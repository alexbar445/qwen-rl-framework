"""Top-level run lifecycle: task allowlists, complete cohorts, and finalization."""
from __future__ import annotations

from qwen_rl.contracts.sampling_topology import owner_slots

import asyncio
import copy
import logging
import random
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from ...contracts.environment import (
    Environment, GroupResult, RewardScorer, Task, score_group, validate_environment,
    validate_group_result, validate_reward_scorer,
)
from ...bepa.contracts import StateTransitionError
from ...bepa.seed_scheduler import SeedAwareGenerationGate
from ...rollout.groups import PolicySnapshot, RolloutBatch
from ...rollout.pipeline import AsyncGroupStream
from ...rollout.scheduler import FilteredLimit
from ...worker.placement import EpisodeWorkerPool
from ..backend import (
    CheckpointOverlapCapability,
    CentralGradientRunEngineProtocol,
    LearnerRunEngineProtocol,
    TrainerFailureStageCapability,
)
from ..compute import ComputeOwner
from ..monitoring import RunCommitView
from ..telemetry import TrainingTelemetry
from ..stop_control import wait_for_engine_durable
from .bepa import BepaRunSession, BepaSeedService
from .groups import GroupCollectionServices, GroupCollectionSession
from .handoff import PolicyHandoffCoordinator
from .checkpoint_overlap import CheckpointRolloutOverlap
from .evaluation import EvaluationCoordinator
from .state import PolicyReferenceLedger, RunExecutionState, RunState, json_safe
from .update import UpdateServices, UpdateTransaction

_LOGGER = logging.getLogger(__name__)


def _required_policy_ack_worker_ids(runtime: Any, configured_workers: list[dict[str, Any]]) -> set[str]:
    worker_control = getattr(runtime, "worker_control", None)
    required = set(getattr(worker_control, "required_policy_worker_ids", ()))
    return required or {worker["id"] for worker in configured_workers}


def _shuffled_training_tasks(tasks: list[Task], run_id: str) -> list[Task]:
    """Return a stable, run-specific permutation without mutating the source list."""
    ordered = sorted(tasks, key=lambda task: task.task_id)
    random.Random(f"qwen-rl-training-task-order-v1:{run_id}").shuffle(ordered)
    return ordered


def _select_run_tasks(
    all_train: list[Task],
    train_task_map: dict[str, Task],
    requested_ids: list[str],
    *,
    run_id: str,
    shuffle_full_training_set: bool,
) -> list[Any]:
    """Keep explicit/recovered task order; shuffle only a new implicit full training set."""
    if requested_ids:
        return [train_task_map[task_id] for task_id in requested_ids]
    if shuffle_full_training_set:
        return _shuffled_training_tasks(all_train, run_id)
    return list(all_train)


@dataclass(frozen=True)
class RunCoordinatorServices:
    config: dict[str, Any]
    runtime: RunCoordinatorRuntime
    engine: LearnerRunEngineProtocol
    managers: dict[str, Any]
    worker_pool: EpisodeWorkerPool
    environment: Environment[Task]
    reward_scorer: RewardScorer[Task]
    compute_owner: ComputeOwner
    bepa: Any | None
    evaluation: EvaluationCoordinator
    seed_service: BepaSeedService
    update_lock: asyncio.Lock
    set_monitor_phase: Callable[[str], None]
    latest_policy: Callable[[], dict[str, Any]]
    publish_current: PublishCurrent
    make_recovery_context: RecoveryContextFactory
    engine_recovery_metadata: Callable[[], dict[str, Any] | None]
    assert_workers_current: Callable[[int], Awaitable[None]]
    sync_workers: Callable[[dict[str, Any]], Awaitable[dict[str, dict[str, Any]]]]
    persist_group: PersistGroup
    stream_factory: StreamFactory
    measure_alignment: Callable[
        [LearnerRunEngineProtocol, list[dict[str, Any]], list[str]], dict[str, Any]
    ]
    alignment_from_proximal_scores: AlignmentFromProximalScores
    log_alignment_warning: Callable[[str, dict[str, Any]], None]
    publication_from_update: Callable[[dict[str, Any], int], dict[str, Any]]
    prune_synced_policies: Callable[
        [str | Path, dict[int, dict[str, Any]], dict[str, Any],
         dict[str, dict[str, Any]], set[str]], list[int]
    ]
    diagnostic_course: str
    diagnostic_run_kinds: frozenset[str]
    alignment_thresholds: dict[str, float]
    stop_control: Any | None = None
    arc_sampling: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", validate_environment(self.environment))
        object.__setattr__(self, "reward_scorer", validate_reward_scorer(self.reward_scorer))


class RunRequestProtocol(Protocol):
    kind: str
    steps: int | None
    task_ids: list[str] | None


class RunCoordinatorRuntime(Protocol):
    runs: dict[str, dict[str, Any]]
    run_commit_view: RunCommitView
    training_telemetry: TrainingTelemetry
    policies: dict[int, dict[str, Any]]
    policy_refs: dict[int, int]
    timings: dict[str, Any]
    external_stage_timings: dict[str, Any]
    active_run_id: str | None
    active_sampling_groups: int
    last_policy_acks: dict[str, dict[str, Any]]
    serve_policy_version: int | None
    seed_scheduler: Any | None
    rollout_stream: Any | None


class PublishCurrent(Protocol):
    def __call__(
        self,
        already_published: dict[str, Any] | None = None,
        *,
        pending_checkpoint: bool = False,
    ) -> Awaitable[dict[str, Any]]: ...


class RecoveryContextFactory(Protocol):
    def __call__(
        self,
        *,
        run_id: str,
        request: RunRequestProtocol,
        state: RunState,
        selected_tasks: list[Task],
        next_completed_updates: int,
        next_task_index: int,
        next_group_index: int | None = None,
        bepa_attempt: Any | None = None,
    ) -> dict[str, Any] | None: ...


class PersistGroup(Protocol):
    def __call__(
        self,
        run_dir: Path,
        group_id: str,
        trajectories: list[dict[str, Any]],
        worker_ids: list[str],
        *,
        phase: Literal["pre_update", "post_update"],
        policy_version: int,
        adapter_sha256: str,
        base_fingerprint: str,
    ) -> str: ...


class StreamFactory(Protocol):
    def __call__(self, **kwargs: Any) -> AsyncGroupStream[Any, Any]: ...


class AlignmentFromProximalScores(Protocol):
    def __call__(
        self,
        raw: dict[str, Any] | None,
        worker_ids: list[str],
        *,
        behavior_version: int,
        learner_start_version: int,
    ) -> dict[str, Any]: ...


class RunCoordinator:
    """Run a finite logical update request with explicit per-run state."""

    def __init__(self, services: RunCoordinatorServices) -> None:
        self.services = services

    async def _generate_group(
        self,
        engine: LearnerRunEngineProtocol,
        managers: dict[str, Any],
        config: dict[str, Any],
        task: Task,
        group_id: str,
        policy: PolicySnapshot,
        **kwargs: Any,
    ) -> GroupResult:
        """Generate one group and normalize its rewards at the environment boundary."""
        generated = await self.services.environment.generate_group(
            engine, managers, config, task, group_id, policy, **kwargs,
        )
        group = validate_group_result(generated, policy_version=policy.version)
        return score_group(self.services.reward_scorer, task, group)

    async def execute(self, run_id: str, request: RunRequestProtocol) -> None:
        services = self.services
        runtime = services.runtime
        state = RunState.from_record(runtime.runs[run_id])
        runtime.runs[run_id] = state
        state.set_stage("task_selection")
        started = time.monotonic()
        runtime.run_commit_view.begin(
            run_id,
            runtime.training_telemetry.snapshot(),
            prior_updates=int(state.get("completed_updates", 0)),
        )
        services.set_monitor_phase("task_selection")
        state["metrics"]["workers"] = {
            worker["id"]: {"generation_calls": 0, "generation_seconds": 0.0}
            for worker in services.config["workers"]
        }
        environment_metrics = {
            "scored_episodes": 0,
            "groups_completed": 0,
        }
        if services.environment.metrics_key == "arc":
            environment_metrics["agent_service_http_seconds"] = 0.0
        state["metrics"][services.environment.metrics_key] = environment_metrics

        try:
            all_train = [
                task for task in services.environment.tasks.values()
                if task.split == "train" and task.pool != "diagnostic"
            ]
        except Exception as exc:  # noqa: BLE001 - task source errors are run failures
            await self._fail_before_execution(state, run_id, exc, started)
            return
        train_task_map = {task.task_id: task for task in all_train}
        fixture = services.environment.fixture_task()
        if request.kind in services.diagnostic_run_kinds:
            selected_tasks = [fixture]
            target_steps = request.steps if request.steps is not None else int(
                services.config["dapo"]["updates"]
            )
            max_filtered = int(services.config["dapo"]["max_filtered_groups"])
        else:
            requested_ids = request.task_ids or services.config["rollout"].get("task_ids") or []
            if requested_ids:
                unknown = set(requested_ids) - set(train_task_map)
                if unknown:
                    await self._fail_before_execution(
                        state,
                        run_id,
                        ValueError(f"unknown training tasks: {sorted(unknown)}"),
                        started,
                    )
                    return
            selected_tasks = _select_run_tasks(
                all_train,
                train_task_map,
                requested_ids,
                run_id=run_id,
                shuffle_full_training_set=request.kind == "training",
            )
            target_steps = request.steps if request.steps is not None else int(
                services.config["dapo"]["updates"]
            )
            max_filtered = int(services.config["rollout"]["max_filtered_groups"])
        if not selected_tasks:
            await self._fail_before_execution(
                state,
                run_id,
                ValueError("no real ARC training tasks configured"),
                started,
            )
            return

        state["requested_steps"] = target_steps
        completed_at_start = int(state.get("completed_updates", 0))
        remaining_steps = target_steps - completed_at_start
        if remaining_steps < 0:
            await self._fail_before_execution(
                state,
                run_id,
                RuntimeError("checkpoint progress exceeds the logical run request"),
                started,
                error_type="RecoveryContextError",
            )
            return

        use_bepa = services.bepa is not None and request.kind == "training"
        if use_bepa and "bepa_task_sampling" not in state:
            metadata = services.engine_recovery_metadata() or {}
            previous = (metadata.get("update_context") or {}).get("bepa_task_sampling")
            if previous is not None:
                state["bepa_task_sampling"] = copy.deepcopy(previous)
        if request.kind == 'training' and state.get('category_sampling') is None:
            from .category_sampling import task_set_key, category
            metadata = services.engine_recovery_metadata() or {}
            previous = (metadata.get('update_context') or {}).get('category_sampling')
            if isinstance(previous, dict) and previous.get('task_set') == task_set_key([t for t in selected_tasks if category(t) is not None]):
                state['category_sampling'] = copy.deepcopy(previous)
        if request.kind == "training" and state.get("source_sampling") is None:
            metadata = services.engine_recovery_metadata() or {}
            previous = (metadata.get("update_context") or {}).get("source_sampling")
            if previous is not None:
                # SourceSampler validates the complete identity before dispatch.
                state["source_sampling"] = copy.deepcopy(previous)
        seed_observability = self._initialize_bepa_observability(
            state, selected_tasks, use_bepa,
        )
        resume_cursor = state.get("resume_task_cursor", {})
        task_index = int(resume_cursor.get("next_task_index", 0))
        group_index = int(resume_cursor.get("next_group_index", task_index))
        central_gradient = (
            services.config.get("learner", {}).get("update_backend") == "central_gradient"
        )
        executor_worker_map = services.config.get("learner", {}).get(
            "gradient_executor_worker_ids", {},
        )
        round_size = len(owner_slots(services.config["learner"])) if central_gradient else 0
        needed_groups = round_size if central_gradient else int(
            services.config["learner"]["groups_per_update"]
        )
        execution = RunExecutionState(
            record=state,
            request=request,
            started_monotonic=started,
            selected_tasks=selected_tasks,
            target_steps=target_steps,
            remaining_steps=remaining_steps,
            completed_at_start=completed_at_start,
            task_cursor_base=task_index,
            needed_groups=needed_groups,
            references=PolicyReferenceLedger(services.engine, runtime.policy_refs),
            task_index=task_index,
            group_index=group_index,
        )

        try:
            await self._execute_locked(
                execution,
                use_bepa=use_bepa,
                max_filtered=max_filtered,
                seed_observability=seed_observability,
                fixture=fixture,
            )
            state["summary"] = self.summary(state, execution.worker_sync_acks)
        except asyncio.CancelledError:
            state["status"] = "cancelled"
            state["committed_policy_version"] = int(services.engine.policy_version)
            raise
        except Exception as exc:
            learner_stage = (
                services.engine.trainer.failure_stage
                if isinstance(services.engine, TrainerFailureStageCapability)
                else None
            )
            _LOGGER.exception(
                "RL run %s failed at learner stage %s",
                run_id,
                state.get("failure_stage", "scheduler"),
            )
            state["status"] = "failed"
            state["error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
                "run_stage": state.get("failure_stage", "scheduler"),
                "learner_stage": learner_stage,
            }
        finally:
            phase_controller = getattr(runtime, "gpu_phase_controller", None)
            if services.config["learner"].get("gpu_sharing") == "phase_shared" and phase_controller is not None:
                state["gpu_phases"] = phase_controller.snapshot()
            if services.stop_control is not None:
                stop_receipt = services.stop_control.snapshot()
                if stop_receipt is not None and stop_receipt.get("status") in {
                    "accepted", "waiting_durable",
                }:
                    if stop_receipt.get("boundary_committed") is not True:
                        services.stop_control.block(
                            "run ended before the requested successful durable update",
                            "StopBoundaryNotReached",
                        )
                state["stop_control"] = services.stop_control.snapshot()
            finished_at = time.time()
            summary = self.summary(state, execution.worker_sync_acks)
            state.finish(
                status=state["status"],
                finished_at=finished_at,
                elapsed_seconds=time.monotonic() - started,
                committed_policy_version=int(services.engine.policy_version),
                summary=summary,
            )
            runtime.run_commit_view.finish(run_id)
            services.set_monitor_phase("idle")
            runtime.active_run_id = None
            run_dir = Path(services.config["run_dir"]) / "runs"
            from ...runtime.manifest import write_json_atomic

            run_manifest = run_dir / f"{run_id}.json"
            write_json_atomic(run_manifest, json_safe(state))
            if services.stop_control is not None:
                stop_receipt = services.stop_control.snapshot()
                if (stop_receipt is not None
                        and stop_receipt.get("status") in {"accepted", "waiting_durable"}
                        and stop_receipt.get("boundary_committed") is True):
                    services.stop_control.complete_paused_run()
                    state["stop_control"] = services.stop_control.snapshot()
                    write_json_atomic(run_manifest, json_safe(state))

    async def _fail_before_execution(
        self,
        state: RunState,
        run_id: str,
        error: BaseException,
        started: float,
        *,
        error_type: str | None = None,
    ) -> None:
        services = self.services
        if error_type is None:
            state.fail(error)
        else:
            state["status"] = "failed"
            state["error"] = {"type": error_type, "message": str(error)[:500]}
        state["finished_at"] = time.time()
        state["elapsed_seconds"] = time.monotonic() - started
        services.runtime.run_commit_view.finish(run_id)
        state["summary"] = self.summary(state, {})
        services.runtime.active_run_id = None
        services.set_monitor_phase("idle")
        from ...runtime.manifest import write_json_atomic

        write_json_atomic(
            Path(services.config["run_dir"]) / "runs" / f"{run_id}.json",
            json_safe(state),
        )

    def _initialize_bepa_observability(
        self,
        state: RunState,
        selected_tasks: list[Any],
        use_bepa: bool,
    ) -> dict[str, Any] | None:
        if not use_bepa:
            return None
        ensure_registered = getattr(self.services.bepa, "ensure_registered_tasks", None)
        if callable(ensure_registered):
            ensure_registered(selected_tasks)
        else:
            for task in selected_tasks:
                self.services.bepa.register(task)
        observability = {
            "schema": "bepa-seed-observability-v1",
            "observability_only": True,
            "affects_training_rows_or_rewards": False,
            "engine_epoch": self.services.bepa.epoch,
            "initial_policy_version": int(self.services.engine.policy_version),
            "required_groups": (
                len(owner_slots(self.services.config["learner"]))
                if self.services.config["learner"].get("update_backend") == "central_gradient"
                else int(self.services.config["learner"]["groups_per_update"])
            ),
            "seed_attempt_count": 0,
            "guided_cache_created_count": 0,
            "guided_raw_episodes": [],
            "scheduler": None,
            "training_allowlist_task_count": len(selected_tasks),
        }
        state["bepa_seed_observability"] = observability
        return observability

    async def _execute_locked(
        self,
        execution: RunExecutionState,
        *,
        use_bepa: bool,
        max_filtered: int,
        seed_observability: dict[str, Any] | None,
        fixture: Any,
    ) -> None:
        services = self.services
        runtime = services.runtime
        cfg = services.config
        state = execution.record
        run_id = str(state["run_id"])
        engine = services.engine
        bepa = services.bepa
        selected_tasks = execution.selected_tasks
        needed_groups = execution.needed_groups
        remaining_steps = execution.remaining_steps
        request = execution.request
        target_steps = execution.target_steps
        central_gradient = cfg.get("learner", {}).get("update_backend") == "central_gradient"
        central_engine = cast(CentralGradientRunEngineProtocol, engine) if central_gradient else None
        round_size = (
            len(owner_slots(cfg["learner"]))
            if central_gradient else 0
        )

        async with services.update_lock:
            references = execution.references
            phase_shared = cfg["learner"].get("gpu_sharing") == "phase_shared"
            sequential_updates = phase_shared or cfg["rollout"].get("sequential_updates", False)
            phase_controller = None
            if phase_shared:
                if getattr(getattr(runtime, "gpu_phase_controller", None), "state", None) == "failed":
                    raise RuntimeError("GPU ownership failed; full-state recovery is required")
                from ..gpu_phases import GpuPhaseCoordinator
                phase_controller = GpuPhaseCoordinator(cfg, engine=engine,
                    worker_pool=services.worker_pool, compute_owner=services.compute_owner)
                runtime.gpu_phase_controller = phase_controller
                await phase_controller.begin_sampling(PolicySnapshot.from_policy(services.latest_policy()))
            group_generation_slots = SeedAwareGenerationGate(
                int(cfg["rollout"]["concurrent_tasks"])
            )
            bepa_session = BepaRunSession(
                run_id=state["run_id"],
                execution=execution,
                bepa=bepa,
                seed_service=services.seed_service,
                selected_tasks=selected_tasks,
                policies=runtime.policies,
                current_policy_version=lambda: int(engine.policy_version),
                serve_policy_version=lambda: runtime.serve_policy_version,
                generation_gate=group_generation_slots,
                on_scheduler_change=lambda scheduler: setattr(runtime, "seed_scheduler", scheduler),
            )
            initial_snapshot = PolicySnapshot.from_policy(services.latest_policy())
            execution.initial_snapshot = initial_snapshot
            handoff_coordinator = PolicyHandoffCoordinator(
                execution=execution,
                sync_workers=services.sync_workers,
                set_failure_stage=state.set_stage,
                set_serve_version=lambda version: setattr(runtime, "serve_policy_version", version),
            )
            checkpoint_overlap = (
                CheckpointRolloutOverlap(
                    execution=execution, engine=engine, runtime=runtime,
                    publish_current=services.publish_current,
                    sync_workers=services.sync_workers,
                    stop_control=services.stop_control,
                    pause_every_updates=(
                        int(cfg["evaluation"]["every_updates"])
                        if services.evaluation.pause_training else None
                    ),
                    successful_update_count=(
                        lambda: services.evaluation.count
                        if services.evaluation.pause_training else None
                    ),
                    evaluation_pending=(
                        services.evaluation.has_pending_requests
                        if services.evaluation.pause_training else None
                    ),
                ) if central_gradient else None
            )
            update_transaction = UpdateTransaction(
                execution=execution,
                use_bepa=use_bepa,
                services=UpdateServices(
                    config=cfg,
                    engine=engine,
                    compute_owner=services.compute_owner,
                    policies=runtime.policies,
                    timings=runtime.timings,
                    external_stage_timings=runtime.external_stage_timings,
                    training_telemetry=runtime.training_telemetry,
                    evaluation=services.evaluation,
                    bepa=bepa,
                    set_monitor_phase=services.set_monitor_phase,
                    make_recovery_context=services.make_recovery_context,
                    engine_recovery_metadata=services.engine_recovery_metadata,
                    publish_current=services.publish_current,
                    settle_handoff=handoff_coordinator.settle,
                    alignment_from_proximal_scores=services.alignment_from_proximal_scores,
                    log_alignment_warning=services.log_alignment_warning,
                    publication_from_update=services.publication_from_update,
                    stop_control=services.stop_control,
                    rollout_policy_ready=(
                        checkpoint_overlap.publish if checkpoint_overlap is not None else None
                    ),
                    checkpoint_ready=(
                        phase_controller.checkpoint_ready_callback(asyncio.get_running_loop())
                        if phase_controller is not None
                        and isinstance(engine, CheckpointOverlapCapability)
                        and engine.supports_checkpoint_overlap
                        else None
                    ),
                ),
            )
            group_session = GroupCollectionSession(
                execution=execution,
                selected_tasks=selected_tasks,
                fixture_task=fixture,
                bepa_session=bepa_session,
                services=GroupCollectionServices(
                    config=cfg,
                    engine=engine,
                    managers=services.managers,
                    worker_pool=services.worker_pool,
                    generation_gate=group_generation_slots,
                    generate_group=self._generate_group,
                    metrics_key=services.environment.metrics_key,
                    environment_name=services.environment.name,
                    arc_sampling=services.arc_sampling,
                    persist_group=services.persist_group,
                    current_policy_version=lambda: int(engine.policy_version),
                    set_failure_stage=state.set_stage,
                    change_active_sampling_groups=lambda delta: setattr(
                        runtime,
                        "active_sampling_groups",
                        runtime.active_sampling_groups + delta,
                    ),
                    run_dir=Path(cfg["run_dir"]),
                    diagnostic_course=services.diagnostic_course,
                ),
            )

            if central_gradient:
                # Standalone gradient roles do not use the local manager bootstrap.
                # The API is serving now, so owned Workers can fetch the initial
                # publication through their configured authenticated route.
                state.set_stage("worker_policy_sync")
                await services.sync_workers(services.latest_policy())
            state.set_stage("worker_policy_ack")
            await services.assert_workers_current(initial_snapshot.version)
            runtime.serve_policy_version = initial_snapshot.version
            state["serve_policy_version"] = initial_snapshot.version
            if central_gradient and services.evaluation.pause_training:
                try:
                    services.set_monitor_phase("native_evaluation")
                    state.set_stage("native_evaluation_pause")
                    await services.evaluation.drain_pending_at_boundary()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    state.setdefault("evaluation_warnings", []).append({
                        "type": type(exc).__name__, "message": str(exc)[:300],
                        "stage": "initial_evaluation_before_rollout",
                    })
                finally:
                    services.set_monitor_phase("awaiting_groups")
                    state.set_stage("awaiting_groups")
            if bepa_session.enabled:
                services.set_monitor_phase("initial_plan_reuse")
                state.set_stage("initial_plan_reuse")
            await bepa_session.prepare_initial_collection(initial_snapshot)
            accepted_task_ids_by_version: dict[int, set[str]] = defaultdict(set)
            execution.accepted_task_ids_by_version = accepted_task_ids_by_version
            pending_cache_task_ids = execution.pending_cache_task_ids
            replay_tasks_in_claim = execution.replay_tasks_in_claim
            cohorts_by_batch_id = execution.cohorts_by_batch_id
            stream = None
            runtime.rollout_stream = None

            references.demand_remaining = {}
            if central_gradient:
                for version in range(initial_snapshot.version, initial_snapshot.version + remaining_steps + 1):
                    references.demand_remaining[version] = round_size
            else:
                for update_index in range(remaining_steps):
                    behavior_version = initial_snapshot.version + (
                        update_index if sequential_updates else max(0, update_index - 1)
                    )
                    references.demand_remaining[behavior_version] = (
                        references.demand_remaining.get(behavior_version, 0) + needed_groups
                    )

            try:
                queue_batches = int(cfg["rollout"]["ready_queue_batches"])
                stream = services.stream_factory(
                    initial_snapshot=initial_snapshot,
                    updates=remaining_steps,
                    groups_per_update=needed_groups,
                    max_concurrent=(
                        round_size if central_gradient
                        else int(cfg["rollout"]["concurrent_tasks"])
                    ),
                    max_outstanding_groups=(needed_groups if sequential_updates else queue_batches * needed_groups),
                    **({"sequential_updates": True} if sequential_updates or central_gradient else {}),
                    **({"round_mode": True, "round_size": round_size} if central_gradient else {}),
                    max_filtered=max_filtered,
                    dispatch=group_session.dispatch_group,
                    accept=group_session.accept_group_result,
                    on_error=group_session.on_error,
                    snapshot_version=lambda snapshot: snapshot.version,
                    version_of=lambda item: int(item["policy_version"]),
                    on_filtered=group_session.on_filtered,
                )
                execution.stream = stream
                runtime.rollout_stream = stream
                await stream.start()
                seed_scheduler = bepa_session.start_scheduler()
                if seed_scheduler is not None and seed_observability is not None:
                    seed_observability["scheduler"] = seed_scheduler.snapshot()

                update_index = 0
                while (
                    state["completed_updates"] < target_steps
                    if central_gradient else update_index < remaining_steps
                ):
                    services.set_monitor_phase("awaiting_groups")
                    state.set_stage("learner_claim")
                    learner_start_version = int(engine.policy_version)
                    batch_id = (
                        stream.prefetched_round_id if central_gradient else None
                    ) or uuid.uuid4().hex
                    try:
                        claim_kwargs = {
                            "prepare_group": (bepa_session.prepare_claim_group if use_bepa else None),
                            "on_claim_abort": (bepa_session.abort_claim_reservations if use_bepa else None),
                        }
                        if central_engine is not None:
                            await update_transaction.begin_round(batch_id, stream.round_first_ordinal(batch_id))
                            claim_kwargs["on_group_ready"] = update_transaction.stage_group
                            claim_kwargs["on_group_outcome"] = update_transaction.sampling_outcome
                        if central_gradient:
                            claim = await stream.claim_round(
                                learner_start_version,
                                batch_id,
                                **claim_kwargs,
                            )
                            update_index = claim.update_index
                        else:
                            claim = await stream.claim(
                                learner_start_version,
                                batch_id,
                                **claim_kwargs,
                            )
                    except FilteredLimit:
                        break

                    if central_gradient and claim.actual_groups == 0:
                        outcome_records = [
                            outcome.item.get("record")
                            for outcome in claim.outcomes
                            if isinstance(outcome.item, dict)
                            and isinstance(outcome.item.get("record"), dict)
                        ]
                        cohort = {
                            "round_id": claim.round_id,
                            "policy_version": claim.behavior_version,
                            "adapter_sha256": claim.policy.adapter_sha256,
                            "accepted_groups": 0,
                            "requested_groups": claim.requested_groups,
                            "required_groups": claim.requested_groups,
                            "filtered_groups": claim.filtered_groups,
                            "started_at": min(
                                (item.get("started_at", time.time()) for item in outcome_records),
                                default=time.time(),
                            ),
                            "finished_at": time.time(),
                            "status": "skipped_empty_round",
                            "optimizer_consumed": False,
                            "batch_id": claim.round_id,
                            "dispatches": [
                                {
                                    "group_id": (
                                        outcome.item.get("group_id")
                                        if isinstance(outcome.item, dict) else None
                                    ),
                                    "task_id": (
                                        outcome.item.get("task").task_id
                                        if isinstance(outcome.item, dict)
                                        and outcome.item.get("task") is not None else None
                                    ),
                                    "status": outcome.status,
                                    "reason": outcome.reason,
                                }
                                for outcome in claim.outcomes
                            ],
                        }
                        state.setdefault("cohorts", []).append(cohort)
                        cohorts_by_batch_id[claim.round_id] = cohort
                        state["sampling_window"] = cohort
                        await stream.finish_round(claim.round_id, update_committed=False)
                        state["progress"] = {
                            "groups_attempted": execution.group_index,
                            "updates": state["completed_updates"],
                            "policy_version": int(engine.policy_version),
                        }
                        continue

                    references.satisfy_demand(claim.behavior_version, needed_groups)
                    if not central_gradient:
                        references.retain_demand(claim.policy)
                    for group in claim.items:
                        group["bepa_batch_id"] = claim.batch_id
                        group["record"]["cohort_id"] = claim.batch_id
                        group["record"]["learner_start_version"] = learner_start_version
                        group["record"]["policy_lag"] = learner_start_version - claim.behavior_version
                        accepted_task_ids_by_version[claim.behavior_version].discard(
                            group["task"].task_id
                        )
                    group_records = [group["record"] for group in claim.items]
                    cohort = {
                        "policy_version": claim.behavior_version,
                        "adapter_sha256": claim.policy.adapter_sha256,
                        "accepted_groups": len(claim.items),
                        "required_groups": needed_groups,
                        "filtered_groups": 0,
                        "started_at": min(record["started_at"] for record in group_records),
                        "finished_at": max(record["finished_at"] for record in group_records),
                        "initial_collection": update_index == 0,
                        "accepted_task_ids": [group["task"].task_id for group in claim.items],
                        "group_ids": [group["group_id"] for group in claim.items],
                        "dispatches": [
                            {
                                "group_id": group["group_id"],
                                "task_id": group["task"].task_id,
                                "selected_ready_cache": group["selected_ready_cache"],
                            }
                            for group in claim.items
                        ],
                        "status": "claimed",
                        "batch_id": claim.batch_id,
                    }
                    if central_gradient:
                        cohort.update(
                            round_id=claim.round_id,
                            requested_groups=claim.requested_groups,
                            filtered_groups=claim.filtered_groups,
                        )
                    state.setdefault("cohorts", []).append(cohort)
                    cohorts_by_batch_id[claim.batch_id] = cohort
                    state["sampling_window"] = cohort

                    batch = RolloutBatch(claim.policy, tuple(claim.items))
                    batch.validate(
                        len(claim.items) if central_gradient else needed_groups,
                        int(cfg["rollout"]["group_size"]),
                        learner_start_version,
                    )
                    completed_before = state["completed_updates"]
                    update_to_commit = int(state["completed_updates"]) + 1
                    if services.stop_control is not None:
                        services.stop_control.begin_update(
                            run_id=run_id,
                            update=update_to_commit,
                            policy_version=int(engine.policy_version) + 1,
                        )
                        if services.stop_control.targets_update(
                            run_id=run_id, update=update_to_commit,
                        ):
                            await stream.stop_admission()
                    try:
                        if phase_controller is not None:
                            await phase_controller.begin_training()
                        update_result = await update_transaction.execute(batch, update_index)
                        if checkpoint_overlap is not None:
                            checkpoint_overlap.committed(update_result)
                            await group_session.checkpoint_settled()
                        targeted_stop = False
                        if services.stop_control is not None:
                            recovery = services.engine_recovery_metadata() or {}
                            targeted_stop = services.stop_control.update_committed(
                                run_id=run_id,
                                update=update_to_commit,
                                policy_version=int(engine.policy_version),
                                recovery=recovery,
                                bepa_attempt_id=update_result.get("bepa_attempt_id"),
                                bepa_commit_state=update_result.get(
                                    "bepa_commit_state", "not_applicable",
                                ),
                            )
                            if targeted_stop:
                                try:
                                    await asyncio.to_thread(
                                        wait_for_engine_durable,
                                        engine,
                                    )
                                except Exception as exc:
                                    services.stop_control.block(
                                        f"target checkpoint did not become durable: {exc}",
                                        type(exc).__name__,
                                    )
                                    raise
                                recovery = services.engine_recovery_metadata() or {}
                                services.stop_control.update_committed(
                                    run_id=run_id,
                                    update=update_to_commit,
                                    policy_version=int(engine.policy_version),
                                    recovery=recovery,
                                    bepa_attempt_id=update_result.get("bepa_attempt_id"),
                                    bepa_commit_state=update_result.get(
                                        "bepa_commit_state", "not_applicable",
                                    ),
                                )
                        if phase_controller is not None:
                            await phase_controller.finish_training(update_result["policy"])
                    except BaseException as exc:
                        if checkpoint_overlap is not None:
                            checkpoint_overlap.failed(exc)
                            await group_session.checkpoint_settled()
                        if phase_controller is not None:
                            await phase_controller.abort_checkpoint_overlap(exc)
                        if state["completed_updates"] > completed_before:
                            if central_gradient:
                                await stream.finish_round(claim.round_id, update_committed=True)
                            else:
                                await stream.finish_update()
                        raise
                    stop_at_boundary = await self._finish_update_handoff(
                        stream=stream,
                        handoff_coordinator=handoff_coordinator,
                        phase_controller=phase_controller,
                        update_result=update_result,
                        latest_policy=services.latest_policy,
                        round_id=claim.round_id if central_gradient else None,
                        stop_after_checkpoint=targeted_stop,
                        stop_control=services.stop_control,
                        run_id=run_id,
                        update=update_to_commit,
                        policy_version=int(engine.policy_version),
                    )
                    if stop_at_boundary:
                        if services.stop_control is not None:
                            if targeted_stop:
                                services.stop_control.mark_boundary_committed(
                                    recovery=services.engine_recovery_metadata() or {},
                                )
                            state["stop_control"] = services.stop_control.snapshot()
                        if state["completed_updates"] < state["requested_steps"]:
                            state["status"] = "paused"
                        break
                    if central_gradient:
                        handoff_error = await handoff_coordinator.settle(wait=True)
                        if handoff_error is not None:
                            raise handoff_error
                    if (phase_controller is not None and execution.request.kind == "training"
                            and not targeted_stop
                            and services.evaluation.runtime is not None):
                        services.evaluation.observe_commit(services.latest_policy())
                    if (central_gradient and not targeted_stop and services.evaluation.pause_training
                            and services.evaluation.has_pending_requests()
                        and not (update_result.get("checkpoint_overlap") or {}).get("prefetched_round_id")):
                        try:
                            state.set_stage("native_evaluation_pause")
                            services.set_monitor_phase("native_evaluation")
                            await services.evaluation.drain_pending_at_boundary()
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            state.setdefault("evaluation_warnings", []).append({
                                "type": type(exc).__name__, "message": str(exc)[:300],
                                "stage": "update_boundary_evaluation",
                            })
                        finally:
                            services.set_monitor_phase("awaiting_groups")
                            state.set_stage("awaiting_groups")
                    if not central_gradient:
                        update_index += 1
            except FilteredLimit:
                # Publish terminal status after stream cleanup releases started groups and claims.
                pass
            finally:
                current_task = asyncio.current_task()
                caller_cancelled = current_task is not None and current_task.cancelling() > 0
                stopping_after_checkpoint = bool(
                    services.stop_control is not None
                    and services.stop_control.targets_update(
                        run_id=run_id,
                        update=int(state.get("completed_updates", 0)),
                    )
                    and services.stop_control.snapshot()
                    and services.stop_control.snapshot().get("boundary_committed") is True
                )
                seed_scheduler_snapshot = await bepa_session.close_scheduler(
                    **({"drain": True} if phase_shared and not caller_cancelled
                       and not stopping_after_checkpoint
                       and phase_controller.state != "failed" else {})
                )
                if seed_observability is not None and seed_scheduler_snapshot is not None:
                    seed_observability["scheduler"] = seed_scheduler_snapshot
                if stream is not None:
                    stream_result = await stream.stop(
                        drain=not caller_cancelled and not stopping_after_checkpoint and not (
                            checkpoint_overlap is not None and checkpoint_overlap.discard_speculative
                        ),
                    )
                    unconsumed_items = stream_result.unconsumed_items
                    if central_engine is not None:
                        await asyncio.to_thread(central_engine.cancel_staged_round)
                else:
                    unconsumed_items = []
                handoff_error, cancelled_while_draining_handoff = await handoff_coordinator.drain()
                caller_cancelled = caller_cancelled or cancelled_while_draining_handoff
                for group_record in state.get("groups", []):
                    if checkpoint_overlap is not None:
                        checkpoint_overlap.quarantine(group_record)
                    if (group_record.get("status") in {
                        "scored", "replay_candidate", "replay_reserved", "rescue_waiting_guided",
                    } and not group_record.get("optimizer_consumed")):
                        group_record["status"] = "unconsumed"
                        group_record["optimizer_consumed"] = False
                        group_record["unconsumed_reason"] = (
                            "caller_cancelled" if caller_cancelled
                            else state.get("status", "run_stopped")
                        )
                for cohort in cohorts_by_batch_id.values():
                    if cohort.get("status") == "claimed":
                        cohort.update(
                            status="unconsumed",
                            finished_at=time.time(),
                            optimizer_consumed=False,
                        )
                for item in unconsumed_items:
                    if not isinstance(item, dict):
                        continue
                    task = item.get("task")
                    if task is not None:
                        accepted_task_ids_by_version[int(item["policy_version"])].discard(
                            task.task_id
                        )
                        pending_cache_task_ids.discard(task.task_id)
                    reservation = item.get("reservation")
                    if use_bepa and reservation is not None:
                        reservation_batch_id = getattr(reservation, "batch_id", None)
                        if isinstance(reservation_batch_id, str) and reservation_batch_id:
                            try:
                                await asyncio.to_thread(
                                    bepa.store.release_batch_reservations,
                                    reservation_batch_id,
                                    reason="unconsumed_stream_claim",
                                )
                            except StateTransitionError:
                                # Optimizer-entered/committed reservations stay under ledger control.
                                pass
                        durable = await asyncio.to_thread(
                            bepa.store.get_reservation, reservation.reservation_id,
                        )
                        if durable is not None and durable.state == "released":
                            item["record"].setdefault("released_replay_reservations", []).append(
                                reservation.reservation_id
                            )
                            item["record"].pop("replay_reservation", None)
                            item.pop("reservation", None)
                references.release_all_demand()
                references.release_all_held()
                replay_tasks_in_claim.clear()
                if runtime.last_policy_acks:
                    synced = next(iter(runtime.last_policy_acks.values()))["version"]
                    required_worker_ids = _required_policy_ack_worker_ids(runtime, cfg["workers"])
                    await asyncio.to_thread(
                        services.prune_synced_policies,
                        cfg["run_dir"],
                        runtime.policies,
                        runtime.policies[synced],
                        runtime.last_policy_acks,
                        required_worker_ids,
                        frozenset(getattr(runtime, "policy_refs", {})),
                        getattr(runtime, "policy_file_references", None),
                    )
                if cancelled_while_draining_handoff:
                    raise asyncio.CancelledError
                if handoff_error is not None and not caller_cancelled:
                    raise handoff_error

            if (request.kind == "selftest" and state["completed_updates"] == execution.target_steps
                    and not (services.stop_control is not None
                             and services.stop_control.snapshot()
                             and services.stop_control.snapshot().get("boundary_committed") is True)):
                await self._post_update_selftest(execution, fixture)
            if state["status"] == "running":
                state["status"] = (
                    "succeeded" if state["completed_updates"] == execution.target_steps
                    else "filtered_limit"
                )

    async def _finish_update_handoff(
        self,
        *,
        stream,
        handoff_coordinator,
        phase_controller,
        update_result,
        latest_policy,
        round_id: str | None = None,
        stop_after_checkpoint: bool = False,
        stop_control: Any | None = None,
        run_id: str | None = None,
        update: int | None = None,
        policy_version: int | None = None,
    ) -> bool:
        """Bookkeep one committed update and return sampling ownership safely."""
        try:
            # This is deliberately separate from the update/finish_training
            # guard above: if bookkeeping itself fails, never call it twice.
            if round_id is None:
                await stream.finish_update()
            else:
                await stream.finish_round(round_id, update_committed=True)
            if (stop_after_checkpoint or (stop_control is not None and run_id is not None
                    and update is not None and policy_version is not None
                    and stop_control.targets_current_checkpoint(
                        run_id=run_id, update=update, policy_version=policy_version,
                    ))):
                return True
            overlap = update_result.get("checkpoint_overlap")
            if isinstance(overlap, dict) and overlap.get("worker_acked") is True:
                snapshot = PolicySnapshot.from_policy(update_result["policy"])
                if (overlap.get("policy_version") != snapshot.version
                        or overlap.get("adapter_sha256") != snapshot.adapter_sha256):
                    raise RuntimeError("early Worker ACK differs from the committed policy")
            else:
                await handoff_coordinator.start(update_result, stream)
            if (phase_controller is not None
                    or self.services.config["rollout"].get("sequential_updates", False)):
                error = await handoff_coordinator.settle(wait=True)
                if error is not None:
                    raise error
            if phase_controller is not None:
                await phase_controller.begin_sampling(
                    PolicySnapshot.from_policy(latest_policy())
                )
            return False
        except BaseException as exc:
            if phase_controller is not None and phase_controller.state != "failed":
                phase_controller.fail(exc)
            raise

    async def _post_update_selftest(self, execution: RunExecutionState, fixture: Any) -> None:
        services = self.services
        runtime = services.runtime
        state = execution.record
        engine = services.engine
        post_version = int(engine.policy_version)
        await services.assert_workers_current(post_version)
        post_group_id = f"{state['run_id']}-post-update-{uuid.uuid4().hex[:8]}"
        post = await self._generate_group(
            engine,
            services.managers,
            services.config,
            fixture,
            post_group_id,
            PolicySnapshot.from_policy(services.latest_policy()),
            worker_pool=services.worker_pool,
            diagnostic_course=services.diagnostic_course,
        )
        environment_metrics = post.get("metrics", {})
        bucket = state["metrics"][services.environment.metrics_key]
        bucket["scored_episodes"] += len(post["trajectories"])
        bucket["groups_completed"] += 1
        for name, value in environment_metrics.get("values", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bucket[name] = bucket.get(name, 0.0) + float(value)
        for worker_id, counts in environment_metrics.get("worker_generation", {}).items():
            worker_metrics = state["metrics"]["workers"].setdefault(
                worker_id, {"generation_calls": 0, "generation_seconds": 0.0},
            )
            worker_metrics["generation_calls"] += int(counts.get("generation_calls", 0))
            worker_metrics["generation_seconds"] += float(counts.get("generation_seconds", 0.0))
        post_policy = runtime.policies.get(post_version)
        if post_policy is None:
            raise RuntimeError(f"post-update trajectory policy {post_version} has not been published")
        trajectory_artifact = services.persist_group(
            Path(services.config["run_dir"]),
            post_group_id,
            post["trajectories"],
            post["worker_ids"],
            phase="post_update",
            policy_version=post_version,
            adapter_sha256=PolicySnapshot.from_policy(post_policy).adapter_sha256,
            base_fingerprint=post_policy["base_fingerprint"],
        )
        post_record = {
            "group_id": post_group_id,
            "task_id": fixture.task_id,
            "source": "environment_fixture",
            "diagnostic_course": services.diagnostic_course,
            "reward_source": services.environment.reward_schema,
            "policy_version": post_version,
            "group_size": len(post["trajectories"]),
            "worker_ids": post["worker_ids"],
            "rewards": post["rewards"],
            "sampling_configs": post.get("sampling_configs", {}),
            "worker_generation_calls": {
                key: int(value.get("generation_calls", 0))
                for key, value in environment_metrics.get("worker_generation", {}).items()
            },
            "trajectory_artifact": trajectory_artifact,
            "status": "post_update_scored",
        }
        if "exact" in post:
            post_record["exact"] = post["exact"]
        if "costs" in post:
            post_record["costs"] = post["costs"]
        if "initial_prompt_prefix_sha256" in post:
            post_record["initial_prompt_prefix_sha256"] = post["initial_prompt_prefix_sha256"]
        state["groups"].append(post_record)
        state["post_update_generation"] = post_record
        phase_controller = (getattr(runtime, "gpu_phase_controller", None)
                            if services.config["learner"].get("gpu_sharing") == "phase_shared" else None)
        try:
            if phase_controller is not None:
                await phase_controller.begin_training()
            alignment = await services.compute_owner.run(
                services.measure_alignment,
                engine,
                post["trajectories"],
                post["worker_ids"],
            )
        except Exception as exc:
            if phase_controller is not None:
                phase_controller.fail(exc)
            alignment = {
                "status": "failed",
                "policy_version": post_version,
                "thresholds": dict(services.alignment_thresholds),
                "error": {"type": type(exc).__name__, "message": str(exc)[:300]},
            }
            state["logprob_alignment"] = alignment
            post_record["logprob_alignment"] = alignment
            post_record["status"] = "post_update_alignment_error"
            raise
        if phase_controller is not None:
            await phase_controller.finish_scoring()
            await phase_controller.begin_sampling(PolicySnapshot.from_policy(services.latest_policy()))
        alignment = {"policy_version": post_version, **alignment}
        state["logprob_alignment"] = alignment
        post_record["logprob_alignment"] = alignment
        services.log_alignment_warning("post-update", alignment)

    def summary(
        self,
        state: dict[str, Any],
        policy_acks: dict[str, Any],
    ) -> dict[str, Any]:
        is_identity_diagnostic = state.get("kind") in self.services.diagnostic_run_kinds
        groups = state.get("groups", [])
        scored = [group for group in groups if group.get("status") in {
            "scored", "filtered_zero_variance", "filtered_nonpositive_reward", "pre_update_alignment_error",
            "pre_update_alignment_failed", "post_update_alignment_error",
            "post_update_alignment_failed",
        }]

        def variable(values: list[Any]) -> bool:
            return len(values) > 1 and max(values) != min(values)

        worker_calls = {
            worker_id: int(metrics.get("generation_calls", 0))
            for worker_id, metrics in state.get("metrics", {}).get("workers", {}).items()
        }
        return {
            "kind": state.get("kind"),
            "environment": self.services.environment.name,
            "fixture_kind": "environment_fixture" if is_identity_diagnostic else "training_task",
            "diagnostic_course": state.get("diagnostic_course"),
            "reward_source": self.services.environment.reward_schema,
            "model_generations": sum(worker_calls.values()),
            "worker_generation_calls": worker_calls,
            "scored_episodes": state.get("metrics", {}).get(self.services.environment.metrics_key, {}).get("scored_episodes", 0),
            "groups_completed": state.get("metrics", {}).get(self.services.environment.metrics_key, {}).get("groups_completed", 0),
            "updates": state.get("completed_updates", 0),
            "policy_version": int(self.services.engine.policy_version),
            "initial_policy_version": state.get("initial_policy_version"),
            "initial_adapter_hash": state.get("initial_adapter_hash"),
            "filtered_groups": state.get("filtered_groups", 0),
            "policy_acks": {key: value for key, value in policy_acks.items()},
            "post_update_generation": state.get("post_update_generation"),
            "pre_update_logprob_alignment": state.get("pre_update_logprob_alignment"),
            "pre_update_worker_ids": state.get("pre_update_worker_ids"),
            "logprob_alignment": state.get("logprob_alignment"),
            "bepa_seed_observability": state.get("bepa_seed_observability"),
            "exact_episodes": sum(
                sum(1 for value in group.get("exact", []) if value) for group in scored
            ),
            "environment_evaluation_gain_claimed": False if is_identity_diagnostic else None,
            "timings": state.get("metrics", {}),
        }
