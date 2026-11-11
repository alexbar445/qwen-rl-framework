"""Single-update transaction and durable optimizer-entry lifecycle."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

from ...bepa.contracts import StateTransitionError
from ...rollout.groups import PolicySnapshot, RolloutBatch
from ..backend import CentralGradientRunEngineProtocol, LearnerRunEngineProtocol
from .state import RunExecutionState

_LOGGER = logging.getLogger(__name__)


def _validate_behavior_inputs(
    trajectories: list[dict[str, Any]], worker_ids: list[str],
) -> None:
    """Reject invalid fresh behavior arrays before optimizer entry without model forwards."""
    if len(trajectories) != len(worker_ids) or not trajectories:
        raise ValueError("logprob alignment requires one worker id per trajectory")
    policy_workers: set[str] = set()
    for trajectory, worker_id in zip(trajectories, worker_ids, strict=True):
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("logprob alignment trajectory omitted its Worker id")
        input_ids = trajectory["input_ids"]
        policy_mask = trajectory["policy_mask"]
        behavior = trajectory["behavior_logprobs"]
        if len(input_ids) < 2:
            raise ValueError("logprobs requires at least two input tokens")
        if len(input_ids) != len(policy_mask) or len(input_ids) != len(behavior):
            raise ValueError("logprob alignment token, mask, and behavior arrays are misaligned")
        if any(value not in (0, 1, False, True) for value in policy_mask):
            raise ValueError("policy_mask must contain only 0/1 values")
        if policy_mask[0]:
            raise ValueError("logprob alignment policy mask must leave input position zero inactive")
        active_positions = [
            position for position, active in enumerate(policy_mask)
            if active and position > 0
        ]
        if not active_positions:
            continue
        policy_workers.add(worker_id)
        for position in active_positions:
            try:
                behavior_logprob = float(behavior[position])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"invalid behavior logprob for {worker_id} at input position {position}"
                ) from exc
            if not math.isfinite(behavior_logprob):
                raise FloatingPointError(
                    f"non-finite behavior logprob for {worker_id} at input position {position}"
                )
    missing_workers = set(worker_ids) - policy_workers
    if missing_workers:
        raise ValueError(
            "logprob alignment found no policy tokens for Worker(s): "
            + ", ".join(sorted(missing_workers))
        )


def _rows_with_source_worker_ids(
    rows: list[dict[str, Any]],
    worker_ids: list[str],
    *,
    fresh_rows: list[dict[str, Any]] | None = None,
    fresh_worker_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Copy learner rows and retain the Worker identity of each fresh source row."""
    if (fresh_rows is None) != (fresh_worker_ids is None):
        raise ValueError("BEPA fresh rows and Worker IDs must be provided together")
    if fresh_rows is None:
        if len(rows) != len(worker_ids):
            raise RuntimeError("Learner update rows differ from their source Worker IDs")
        return [
            {**row, "worker_id": worker_id}
            for row, worker_id in zip(rows, worker_ids, strict=True)
        ]

    assert fresh_rows is not None and fresh_worker_ids is not None
    worker_by_row_identity = {
        id(row): worker_id
        for row, worker_id in zip(fresh_rows, fresh_worker_ids, strict=True)
    }
    learner_rows = []
    for row in rows:
        copied = dict(row)
        if copied.get("row_kind", "fresh_student") == "fresh_student":
            worker_id = worker_by_row_identity.get(id(row))
            if not isinstance(worker_id, str) or not worker_id:
                raise RuntimeError("BEPA fresh update row lost its source Worker identity")
            copied["worker_id"] = worker_id
        learner_rows.append(copied)
    return learner_rows


@asynccontextmanager
async def pre_optimizer_batch_guard(store: Any, batch_id: str):
    """Release BEPA reservations when preparation fails before optimizer entry."""
    try:
        yield
    except BaseException:
        try:
            await asyncio.to_thread(
                store.release_batch_reservations,
                batch_id,
                reason="pre_optimizer_failure",
            )
        except StateTransitionError:
            # The store refuses release after optimizer entry or commit. Preserve
            # that durable state and let the original failure propagate.
            _LOGGER.warning(
                "BEPA batch %s could not be released because its update is no longer prepared",
                batch_id,
            )
        raise


@dataclass(frozen=True)
class UpdateServices:
    config: dict[str, Any]
    engine: LearnerRunEngineProtocol
    compute_owner: Any
    policies: dict[int, dict[str, Any]]
    timings: dict[str, Any]
    external_stage_timings: dict[str, Any]
    training_telemetry: Any
    evaluation: Any
    bepa: Any | None
    set_monitor_phase: Callable[[str], None]
    make_recovery_context: Callable[..., dict[str, Any] | None]
    engine_recovery_metadata: Callable[[], dict[str, Any] | None]
    publish_current: Callable[[dict[str, Any] | None], Awaitable[dict[str, Any]]]
    settle_handoff: Callable[..., Awaitable[Exception | None]]
    alignment_from_proximal_scores: Callable[..., dict[str, Any]]
    log_alignment_warning: Callable[[str, dict[str, Any]], None]
    publication_from_update: Callable[[dict[str, Any], int], dict[str, Any]]
    checkpoint_ready: Callable[[], None] | None = None
    rollout_policy_ready: Callable[[dict[str, Any], RolloutBatch], Awaitable[dict[str, Any]]] | None = None
    stop_control: Any | None = None


class UpdateTransaction:
    """Validate, enter, commit, publish, and account for one claimed batch.

    The transaction intentionally keeps the current order: prepare BEPA source
    rows and the durable attempt; validate behavior inputs; call the guarded
    engine update; durably commit BEPA; account for the irreversible optimizer
    step; validate its publication; settle the previous handoff; then publish the
    current policy.
    """

    def __init__(
        self,
        *,
        execution: RunExecutionState,
        services: UpdateServices,
        use_bepa: bool,
    ) -> None:
        self.execution = execution
        self.services = services
        self.use_bepa = use_bepa

    async def execute(self, batch: RolloutBatch, update_index: int) -> dict[str, Any]:
        """Shield the whole ledger commit while propagating waiter cancellation."""
        task = asyncio.create_task(self._commit(batch, update_index))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:  # noqa: BLE001 - drain the transaction's owned task
                    break
            if task.done() and not task.cancelled():
                task.exception()
            raise

    async def begin_round(self, round_id: str, first_ordinal: int) -> None:
        engine = cast(CentralGradientRunEngineProtocol, self.services.engine)
        await asyncio.to_thread(engine.begin_staged_round, round_id, first_ordinal)

    async def sampling_outcome(self, outcome, round_id: str) -> None:
        engine = cast(CentralGradientRunEngineProtocol, self.services.engine)
        await asyncio.to_thread(engine.note_sampling_outcome, round_id, outcome.ordinal)

    async def stage_group(self, group: dict[str, Any], round_id: str) -> None:
        """Reserve/validate the final group projection before asynchronous backward."""
        group["bepa_batch_id"] = round_id
        policy = group["policy_snapshot"]
        batch = RolloutBatch(policy, (group,))
        if self.use_bepa:
            training = await asyncio.to_thread(
                self.services.bepa.training_batch,
                batch,
                round_id,
                int(self.services.engine.policy_version),
            )
            rows = _rows_with_source_worker_ids(
                training.rows,
                training.fresh_worker_ids,
                fresh_rows=training.fresh_rows,
                fresh_worker_ids=training.fresh_worker_ids,
            )
        else:
            rows = _rows_with_source_worker_ids(batch.trajectories, batch.worker_ids)
        engine = cast(CentralGradientRunEngineProtocol, self.services.engine)
        await asyncio.to_thread(
            engine.stage_group, round_id, rows,
            logical_run_id=self.execution.record["run_id"],
        )

    async def _commit(self, batch: RolloutBatch, index: int) -> dict[str, Any]:
        execution = self.execution
        state = execution.record
        services = self.services
        engine = services.engine
        bepa = services.bepa
        services.set_monitor_phase("batch_prepare")
        state.set_stage("batch_prepare")
        completed_handoff_error = await services.settle_handoff(wait=False)
        if completed_handoff_error is not None:
            raise completed_handoff_error

        learner_start_version = int(engine.policy_version)
        services.external_stage_timings["pre_alignment_seconds"] = None
        central_round = engine.config.get("update_backend") == "central_gradient"
        round_group_count = len(batch.groups) if central_round else execution.needed_groups
        if round_group_count < 1:
            raise ValueError("optimizer update requires at least one complete group")
        batch.validate(
            round_group_count,
            int(services.config["rollout"]["group_size"]),
            learner_start_version,
        )
        lag = learner_start_version - batch.policy.version
        parallel_mode = engine.config.get("parallelism")
        use_proximal_alignment = lag == 0 and parallel_mode == "tp"
        trajectories, trajectory_worker_ids = batch.trajectories, batch.worker_ids
        training = None
        attempt = None
        group_cardinalities: dict[str, dict[str, int]] = {}

        async with AsyncExitStack() as pre_optimizer_guard:
            if self.use_bepa:
                batch_id = batch.groups[0]["bepa_batch_id"]
                await pre_optimizer_guard.enter_async_context(
                    pre_optimizer_batch_guard(bepa.store, batch_id)
                )
                training = await asyncio.to_thread(
                    bepa.training_batch, batch, batch_id, learner_start_version,
                )
                learner_rows = _rows_with_source_worker_ids(
                    training.rows,
                    training.fresh_worker_ids,
                    fresh_rows=training.fresh_rows,
                    fresh_worker_ids=training.fresh_worker_ids,
                )
                trajectories = training.fresh_rows
                trajectory_worker_ids = training.fresh_worker_ids
                training_counts: dict[str, int] = defaultdict(int)
                for row in training.rows:
                    training_counts[str(row["group_id"])] += 1
                for group in batch.groups:
                    group_id = str(group["group_id"])
                    raw_size = len(group["trajectories"])
                    training_size = training_counts.get(group_id, 0)
                    if training_size < raw_size:
                        raise RuntimeError(f"training group {group_id} lost fresh source rows")
                    group_cardinalities[group_id] = {
                        "raw": raw_size,
                        "training": training_size,
                    }
                    group["record"]["raw_group_size"] = raw_size
                    group["record"]["training_group_size"] = training_size
                    group["record"]["group_cardinalities"] = group_cardinalities[group_id]
                from ...bepa.contracts import content_hash

                attempt = await asyncio.to_thread(
                    bepa.store.prepare_update,
                    batch_id,
                    input_batch_hash=content_hash(training.rows),
                    engine_epoch=bepa.epoch,
                    start_policy_version=learner_start_version,
                )
                if services.stop_control is not None:
                    services.stop_control.register_bepa_attempt(
                        run_id=str(state["run_id"]),
                        update=int(state["completed_updates"]) + 1,
                        attempt_id=attempt.attempt_id,
                    )

            _validate_behavior_inputs(trajectories, trajectory_worker_ids)

            if use_proximal_alignment:
                pre_alignment = None
            elif lag == 0:
                # Diagnostic-only forward passes are excluded from production updates.
                # The engine still computes p, b correction and guarded loss normally.
                pre_alignment = {
                    "status": "skipped",
                    "reason": "standalone_pre_update_alignment_removed",
                    "behavior_version": batch.policy.version,
                    "learner_start_version": learner_start_version,
                    "policy_lag": lag,
                }
            else:
                pre_alignment = {
                    "status": "policy_lag",
                    "behavior_version": batch.policy.version,
                    "learner_start_version": learner_start_version,
                    "policy_lag": lag,
                }

        state["pre_update_logprob_alignment"] = pre_alignment
        state["pre_update_worker_ids"] = sorted(set(trajectory_worker_ids))
        for group in batch.groups:
            group["record"]["pre_update_logprob_alignment"] = pre_alignment
            group["record"]["learner_start_version"] = learner_start_version
            group["record"]["policy_lag"] = lag
        old_adapter_hash = PolicySnapshot.from_policy(
            services.policies[learner_start_version]
        ).adapter_sha256
        overlap_record: dict[str, Any] | None = None
        loop = asyncio.get_running_loop()

        def rollout_policy_ready(publication: dict[str, Any]) -> None:
            nonlocal overlap_record
            if services.rollout_policy_ready is None:
                return
            task = asyncio.run_coroutine_threadsafe(
                services.rollout_policy_ready(publication, batch), loop,
            )
            overlap_record = task.result()

        update_started = time.time()
        next_completed_updates = int(state["completed_updates"]) + 1
        if central_round:
            next_task_cursor = execution.task_index
            next_group_cursor = execution.group_index
        else:
            next_task_cursor = execution.task_cursor_base + (
                next_completed_updates - execution.completed_at_start
            ) * round_group_count
            next_group_cursor = next_task_cursor
        recovery_context = services.make_recovery_context(
            run_id=state["run_id"],
            request=execution.request,
            state=state,
            selected_tasks=execution.selected_tasks,
            next_completed_updates=next_completed_updates,
            next_task_index=next_task_cursor,
            next_group_index=next_group_cursor,
            bepa_attempt=attempt,
        )
        if not self.use_bepa:
            learner_rows = _rows_with_source_worker_ids(trajectories, trajectory_worker_ids)
        services.set_monitor_phase("optimizer")
        services.training_telemetry.begin(training.rows if self.use_bepa else trajectories)
        telemetry_finalized = False
        staged_round_id = batch.groups[0].get("bepa_batch_id") if central_round else None

        def mark_telemetry_finalized() -> None:
            nonlocal telemetry_finalized
            telemetry_finalized = True

        try:
            try:
                if self.use_bepa:
                    state.set_stage("learner_update")
                    update_kwargs: dict[str, Any] = {
                        "before_optimizer_step": lambda: bepa.store.mark_optimizer_entered(
                            attempt.attempt_id,
                            current_policy_version=learner_start_version,
                        ),
                    }
                    if services.checkpoint_ready is not None:
                        update_kwargs["checkpoint_ready"] = services.checkpoint_ready
                    if services.rollout_policy_ready is not None:
                        update_kwargs["rollout_policy_ready"] = rollout_policy_ready
                    if recovery_context is not None:
                        update_kwargs["recovery_context"] = recovery_context
                    if staged_round_id is not None:
                        update_kwargs["staged_round_id"] = staged_round_id
                    update_result = await services.compute_owner.run(
                        engine.update, learner_rows, **update_kwargs,
                    )
                    publication = update_result["publication"]
                    await asyncio.to_thread(
                        bepa.store.commit_update,
                        attempt.attempt_id,
                        new_policy_version=int(engine.policy_version),
                        adapter_sha256=publication["adapter_hash"],
                        commit_evidence={
                            "updated": True,
                            "gradient_norm": update_result["gradient_norm_before_clip"],
                        },
                    )
                    for group in batch.groups:
                        group["record"]["training_rewards"] = [
                            row["reward"] for row in training.rows
                            if row["group_id"] == group["group_id"]
                        ]
                else:
                    update_kwargs = ({"recovery_context": recovery_context}
                                     if recovery_context is not None else {})
                    if services.checkpoint_ready is not None:
                        update_kwargs["checkpoint_ready"] = services.checkpoint_ready
                    if services.rollout_policy_ready is not None:
                        update_kwargs["rollout_policy_ready"] = rollout_policy_ready
                    if staged_round_id is not None:
                        update_kwargs["staged_round_id"] = staged_round_id
                    update_result = await services.compute_owner.run(
                        engine.update, learner_rows, **update_kwargs,
                    )
            except BaseException:
                if attempt is not None:
                    durable = bepa.store.get_update_attempt(attempt.attempt_id)
                    if durable.state == "prepared":
                        bepa.store.release_batch_reservations(training.batch_id)
                    elif durable.state == "optimizer_entered":
                        reconciled = False
                        if (recovery_context is not None
                                and int(engine.policy_version) == learner_start_version + 1):
                            try:
                                recovery_metadata = services.engine_recovery_metadata()
                                current_version = int(engine.policy_version)
                                if (isinstance(recovery_metadata, dict)
                                        and recovery_metadata.get("enabled") is True
                                        and recovery_metadata.get("dirty") is False
                                        and type(recovery_metadata.get("durable_policy_version")) is int
                                        and recovery_metadata["durable_policy_version"] == current_version
                                        and type(recovery_metadata.get("policy_version")) is int
                                        and recovery_metadata["policy_version"] == current_version):
                                    current_policy = await services.publish_current(None)
                                    reconciled_id = await asyncio.to_thread(
                                        bepa.reconcile_checkpoint_attempt,
                                        policy_version=current_version,
                                        adapter_sha256=PolicySnapshot.from_policy(
                                            current_policy
                                        ).adapter_sha256,
                                    )
                                    reconciled = reconciled_id == attempt.attempt_id
                            except Exception:
                                _LOGGER.exception(
                                    "Could not reconcile checkpoint-backed BEPA attempt %s",
                                    attempt.attempt_id,
                                )
                        durable = bepa.store.get_update_attempt(attempt.attempt_id)
                        if durable.state == "optimizer_entered" and not reconciled:
                            bepa.store.mark_commit_unknown(
                                attempt.attempt_id,
                                reason="engine_update_failed_without_checkpoint_proof",
                            )
                if int(engine.policy_version) == learner_start_version + 1:
                    self._record_commit(
                        batch,
                        update_index=index,
                        update_started=update_started,
                        update_result={
                            "updated": True,
                            "policy_version": int(engine.policy_version),
                            "post_commit_failure": True,
                        },
                        status="committed_post_update_failed",
                        training=training,
                        trajectories=trajectories,
                        group_cardinalities=group_cardinalities,
                        learner_start_version=learner_start_version,
                        lag=lag,
                        on_telemetry_finalized=mark_telemetry_finalized,
                    )
                raise

            if use_proximal_alignment:
                correction = update_result.get("correction") if isinstance(update_result, dict) else None
                raw_alignment = (
                    correction.get("policy_alignment") if isinstance(correction, dict) else None
                )
                pre_alignment = services.alignment_from_proximal_scores(
                    raw_alignment,
                    trajectory_worker_ids,
                    behavior_version=batch.policy.version,
                    learner_start_version=learner_start_version,
                )
                state["pre_update_logprob_alignment"] = pre_alignment
                for group in batch.groups:
                    group["record"]["pre_update_logprob_alignment"] = pre_alignment
                try:
                    services.log_alignment_warning("pre-update", pre_alignment)
                except Exception:
                    _LOGGER.exception("Could not report proximal-scoring logprob alignment")

            if int(engine.policy_version) == learner_start_version + 1:
                self._record_commit(
                    batch,
                    update_index=index,
                    update_started=update_started,
                    update_result=update_result,
                    status="optimizer_committed",
                    training=training,
                    trajectories=trajectories,
                    group_cardinalities=group_cardinalities,
                    learner_start_version=learner_start_version,
                    lag=lag,
                    on_telemetry_finalized=mark_telemetry_finalized,
                )
            update_elapsed = time.time() - update_started
            if not isinstance(update_result, dict) or update_result.get("updated") is not True:
                raise RuntimeError("LearnerEngine.update() did not confirm a DAPO optimizer step")
            gradient_norm = update_result.get("gradient_norm_before_clip")
            if (isinstance(gradient_norm, bool) or not isinstance(gradient_norm, (int, float))
                    or not math.isfinite(float(gradient_norm)) or gradient_norm <= 0.0):
                raise RuntimeError("LearnerEngine.update() did not produce a finite nonzero LoRA gradient")
            update_publication = update_result.get("publication")
            if (not isinstance(update_publication, dict)
                    or not isinstance(update_publication.get("adapter_hash"), str)
                    or update_publication["adapter_hash"] == old_adapter_hash):
                raise RuntimeError("DAPO update did not change the published adapter SHA-256")
            new_version = int(engine.policy_version)
            if new_version != learner_start_version + 1:
                raise RuntimeError("LearnerEngine.update() must advance exactly one policy version")
            services.timings["learner"]["update_seconds"] += update_elapsed
            state["metrics"]["learner"]["update_seconds"] += update_elapsed
            previous_handoff_error = await services.settle_handoff(wait=True)
            publish_started = time.perf_counter()
            policy = await services.publish_current(update_publication)
            state["metrics"]["learner"]["publish_seconds"] += time.perf_counter() - publish_started
            state["progress"] = {
                "groups_attempted": execution.group_index,
                "updates": state["completed_updates"],
                "policy_version": new_version,
            }
            if previous_handoff_error is not None:
                raise previous_handoff_error
            return {
                "policy": policy, "batch": batch, "update": update_result,
                "checkpoint_overlap": overlap_record,
                "bepa_attempt_id": (
                    getattr(attempt, "attempt_id", None) if attempt is not None else None
                ),
                "bepa_commit_state": "committed" if attempt is not None else "not_applicable",
            }
        finally:
            if not telemetry_finalized:
                services.training_telemetry.end()

    def _record_commit(
        self,
        batch: RolloutBatch,
        *,
        update_index: int,
        update_started: float,
        update_result: Any,
        status: str,
        training: Any | None,
        trajectories: list[dict[str, Any]],
        group_cardinalities: dict[str, dict[str, int]],
        learner_start_version: int,
        lag: int,
        on_telemetry_finalized: Callable[[], None],
    ) -> None:
        execution = self.execution
        state = execution.record
        services = self.services
        engine = services.engine
        state["completed_updates"] += 1
        services.training_telemetry.commit(int(engine.policy_version))
        on_telemetry_finalized()
        committed_metadata = services.engine_recovery_metadata() or {}
        durable_post_failure = (
            status == "committed_post_update_failed"
            and committed_metadata.get("dirty") is False
            and committed_metadata.get("durable_policy_version") == int(engine.policy_version)
        )
        if (services.config.get("learner", {}).get("gpu_sharing") != "phase_shared"
                and execution.request.kind == "training" and services.evaluation.runtime is not None
                and (status == "optimizer_committed" or durable_post_failure)):
            current_publication = services.policies.get(int(engine.policy_version))
            try:
                if current_publication is None:
                    current_publication = services.publication_from_update(
                        update_result, int(engine.policy_version),
                    )
            except Exception:
                _LOGGER.exception("Evaluation could not read the committed policy publication")
            services.evaluation.observe_commit(current_publication)

        trajectory_count = len(training.rows) if training is not None else len(trajectories)
        state.setdefault("update_intervals", []).append({
            "started_at": update_started,
            "finished_at": time.time(),
            "status": status,
            "behavior_version": batch.policy.version,
            "learner_start_version": learner_start_version,
            "new_policy_version": int(engine.policy_version),
            "policy_lag": lag,
            "trajectory_count": trajectory_count,
            "group_cardinalities": dict(group_cardinalities),
        })
        batch_id = batch.groups[0]["bepa_batch_id"]
        cohort = execution.cohorts_by_batch_id.get(batch_id)
        if cohort is not None:
            cohort.update(
                status="optimizer_committed",
                update_index=update_index,
                new_policy_version=int(engine.policy_version),
                training_trajectory_count=trajectory_count,
                group_cardinalities=dict(group_cardinalities),
                finished_at=time.time(),
            )
        if isinstance(update_result, dict):
            update_result["trajectory_count"] = trajectory_count
            update_result["group_cardinalities"] = dict(group_cardinalities)
        for group in batch.groups:
            group["record"]["update"] = update_result
            group["record"]["new_policy_version"] = int(engine.policy_version)
            group["record"]["optimizer_consumed"] = True
            execution.accepted_task_ids_by_version[batch.policy.version].discard(
                group["task"].task_id
            )
            if group.get("replay_pending"):
                execution.pending_cache_task_ids.discard(group["task"].task_id)
        execution.replay_tasks_in_claim.clear()
        for group in batch.groups:
            execution.references.release(group["policy_snapshot"])
