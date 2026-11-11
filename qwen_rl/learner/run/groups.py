"""Complete-group collection, validation, and policy-reference ownership."""
from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...rollout.groups import PolicySnapshot
from ...rollout.pipeline import GroupDisposition
from ...worker.placement import assign_task_worker_owner
from .bepa import BepaRunSession
from .state import RunExecutionState
from .category_sampling import CategorySampler, category
from .source_sampling import MixedSourceSampling


class GroupGenerationError(RuntimeError):
    """An incomplete rollout cohort; malformed scored data remains a hard failure."""


class GroupInfrastructureError(GroupGenerationError):
    """Discard a complete group without counting unavailable work as model failure."""

    def __init__(self, message: str, *, worker_ids: tuple[str, ...]):
        super().__init__(message)
        self.worker_ids = worker_ids


def select_dispatch_task(
    selected_tasks: list[Any],
    task_index: int,
    *,
    active_task_ids: set[str],
    accepted_task_ids: set[str],
    pending_cache_task_ids: set[str],
    historical_seen_task_ids: set[str],
    ready_cache_task_id: str | None,
) -> tuple[Any, int, bool]:
    """Select a fresh native task or a validated ready-cache retry."""
    if not selected_tasks:
        raise ValueError("cannot dispatch from an empty training task list")
    if isinstance(task_index, bool) or not isinstance(task_index, int) or task_index < 0:
        raise ValueError("task_index must be a non-negative integer")
    blocked_ready = active_task_ids | accepted_task_ids | pending_cache_task_ids
    next_index = task_index + 1

    if ready_cache_task_id is not None and ready_cache_task_id not in blocked_ready:
        ready = next((task for task in selected_tasks if task.task_id == ready_cache_task_id), None)
        if ready is None:
            raise RuntimeError("ready-cache task is outside the selected training allowlist")
        return ready, next_index, True

    if len(selected_tasks) == 1:
        only_task = selected_tasks[0]
        if only_task.task_id in pending_cache_task_ids:
            raise GroupGenerationError(
                "single-task workload has no unused cache after its replay reservation is pending"
            )
        return only_task, next_index, False

    base = selected_tasks[task_index % len(selected_tasks)]
    if base.task_id not in historical_seen_task_ids and base.task_id not in active_task_ids:
        return base, next_index, False

    unseen = next((task for task in selected_tasks
                   if task.task_id not in historical_seen_task_ids
                   and task.task_id not in active_task_ids), None)
    if unseen is not None:
        return unseen, next_index, False

    retryable = next((task for task in selected_tasks
                      if task.task_id not in active_task_ids
                      and task.task_id not in pending_cache_task_ids), None)
    if retryable is None:
        raise GroupGenerationError(
            "no training task can be dispatched without overlapping an active or pending cache task"
        )
    return retryable, next_index, False


@dataclass(frozen=True)
class GroupCollectionServices:
    config: dict[str, Any]
    engine: Any
    managers: dict[str, Any]
    worker_pool: Any
    generation_gate: Any
    generate_group: Callable[..., Awaitable[dict[str, Any]]]
    metrics_key: str
    environment_name: str
    persist_group: Callable[..., str]
    current_policy_version: Callable[[], int]
    set_failure_stage: Callable[[str], None]
    change_active_sampling_groups: Callable[[int], None]
    run_dir: Path
    diagnostic_course: str


class GroupCollectionSession:
    """Provide AsyncGroupStream callbacks against explicit run-owned state."""

    def __init__(
        self,
        *,
        execution: RunExecutionState,
        services: GroupCollectionServices,
        selected_tasks: list[Any],
        fixture_task: Any,
        bepa_session: BepaRunSession,
    ) -> None:
        self.execution = execution
        self.services = services
        self.selected_tasks = selected_tasks
        self.fixture_task = fixture_task
        self.bepa_session = bepa_session
        self.task_sampling = getattr(bepa_session, "task_sampling", None)
        self.dispatch_condition = asyncio.Condition()
        source_catalog = None
        dataset_dir = services.config.get("dataset_dir")
        if services.environment_name == "arc" and dataset_dir:
            from ...task_catalog import load_source_catalog

            source_catalog = load_source_catalog(dataset_dir)
        self.source_sampling = None
        self.source_round_index = None
        self.source_round_plan = []
        if (execution.request.kind == "training" and source_catalog is not None
                and self.task_sampling is None):
            learner = services.config.get("learner", {})
            if learner.get("update_backend", "trainer") != "central_gradient":
                raise ValueError("catalog-backed mixed source dispatch requires central_gradient")
            from qwen_rl.contracts.sampling_topology import owner_map, owner_slots
            pairs = owner_map(learner)
            if not isinstance(pairs, dict) or not pairs:
                raise ValueError("mixed source dispatch requires explicit executor/Worker pairs")
            self.source_sampling = MixedSourceSampling(
                catalog=source_catalog,
                selected_tasks=selected_tasks,
                worker_ids=tuple(worker for _, worker in sorted(pairs.items())),
                tasks_per_worker=int(learner.get("gradient_tasks_per_worker", 1)),
                record=execution.record,
                worker_slots=tuple(pairs[owner] for owner in owner_slots(learner)),
            )
        elif execution.record.get("source_sampling") is not None and self.task_sampling is None:
            raise ValueError("persisted mixed source sampling requires its explicit task catalog")
        self.category_sampler = None
        category_tasks = (
            self.source_sampling.tasks_for("synthetic_arc")
            if self.source_sampling is not None else selected_tasks
        )
        if (services.environment_name == "arc"
                and services.config.get('rollout', {}).get('category_sampling', True)
                and self.task_sampling is None
                and execution.request.kind == 'training' and len(category_tasks) > 1
                and all(category(task) is not None for task in category_tasks)):
            self.category_sampler = CategorySampler(execution.record, category_tasks)

    def sample_category_task(self, tasks, snapshot):
        execution = self.execution
        blocked = (execution.active_task_ids | execution.accepted_task_ids_by_version[snapshot.version]
                   | execution.pending_cache_task_ids)
        eligible = [t for t in tasks if t.task_id not in blocked]
        unseen = [t for t in eligible if t.task_id not in execution.historical_seen_task_ids]
        if not eligible:
            raise GroupGenerationError('no eligible native task for category sampling')
        round_index = execution.group_index // execution.needed_groups
        scope = f'{execution.record["run_id"]}:{snapshot.version}:{round_index}'
        return self.category_sampler.choose(unseen or eligible, scope)

    async def remove_active_task(self, task_id: str) -> None:
        async with self.dispatch_condition:
            count = self.execution.active_task_counts.get(task_id, 0)
            if count <= 1:
                self.execution.active_task_counts.pop(task_id, None)
                self.execution.active_task_ids.discard(task_id)
            else:
                self.execution.active_task_counts[task_id] = count - 1
            self.dispatch_condition.notify_all()

    async def checkpoint_settled(self) -> None:
        """Wake only waiting selectors; the fence carries success or failure."""
        async with self.dispatch_condition:
            self.dispatch_condition.notify_all()

    async def _select_active_task(self, snapshot, dispatch_tasks, slot_worker, ordinal, *, prefer_active=True):
        """Called with dispatch_condition held; waiting releases it for peers."""
        execution = self.execution
        wait_record = None
        wait_started = None
        try:
            while True:
                fence = getattr(execution, "checkpoint_sampling_fences", {}).get(snapshot.version)
                if fence is not None:
                    fence.check(snapshot)
                active = execution.active_task_ids
                accepted = execution.accepted_task_ids_by_version[snapshot.version]
                pending = execution.pending_cache_task_ids
                blocked = active | accepted | pending
                eligible = {t.task_id for t in dispatch_tasks if t.task_id not in blocked}
                local_a = {t.task_id for t in dispatch_tasks
                           if t.task_id in self.task_sampling.state["active"]}
                transient = (local_a & pending & fence.replay_task_ids - active - accepted
                             if fence is not None and not fence.settled.is_set() else set())
                exclusions = []
                for task_id in sorted(local_a & blocked):
                    reasons = []
                    if task_id in active:
                        reasons.append("active_episode_group")
                    if task_id in accepted:
                        reasons.append("accepted_current_policy")
                    if task_id in pending:
                        reasons.append("previous_checkpoint_replay" if task_id in transient
                                       else "pending_cache_or_guidance")
                    exclusions.append({"task_id": task_id, "reasons": reasons})
                if not prefer_active or local_a & eligible or not transient:
                    selection = self.task_sampling.choose(eligible, prefer_active=prefer_active)
                    if wait_record is not None:
                        wait_record.update(status="released", finished_at=time.time(),
                                           wait_seconds=time.monotonic() - wait_started)
                    return selection, exclusions
                if wait_record is None:
                    wait_started = time.monotonic()
                    wait_record = {
                        "dispatch_ordinal": ordinal, "owner_worker_id": slot_worker,
                        "policy_version": snapshot.version,
                        "previous_round_id": fence.previous_round_id,
                        "reason": "previous_checkpoint_replay", "status": "waiting",
                        "blocked_a_task_ids": sorted(transient), "started_at": time.time(),
                    }
                    execution.record.setdefault("sampling_waits", []).append(wait_record)
                await self.dispatch_condition.wait()
        except BaseException as exc:
            if wait_record is not None:
                wait_record.update(status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                                   finished_at=time.time(), wait_seconds=time.monotonic() - wait_started)
            raise

    def task_worker_owner(self, task_id: str) -> str | None:
        """Resolve and persist central-gradient task affinity from explicit pairs."""
        learner = self.services.config.get("learner", {})
        if learner.get("update_backend", "trainer") != "central_gradient":
            return None
        from qwen_rl.contracts.sampling_topology import owner_map
        executor_workers = owner_map(learner)
        if not isinstance(executor_workers, dict) or not executor_workers:
            raise RuntimeError("central-gradient task affinity requires explicit executor/Worker pairs")
        worker_ids = tuple(
            worker_id
            for _executor_id, worker_id in sorted(executor_workers.items())
        )
        owner = assign_task_worker_owner(
            task_id, worker_ids, self.execution.task_worker_owners,
        )
        # RunState and RunExecutionState share this mapping; copy assignment to
        # the record explicitly so persisted checkpoints and teacher rescue read
        # the same durable owner.
        self.execution.record["task_worker_owners"] = self.execution.task_worker_owners
        return owner

    @property
    def central_gradient_round_mode(self) -> bool:
        return self.services.config.get("learner", {}).get(
            "update_backend", "trainer",
        ) == "central_gradient"

    async def dispatch_group(self, snapshot: PolicySnapshot, ordinal: int) -> dict[str, Any]:
        execution = self.execution
        services = self.services
        record = execution.record
        services.set_failure_stage("rollout_generation")
        execution.references.acquire(snapshot)
        keep_policy_ref = False
        generation_succeeded = False
        selected_task = None
        group_id = None
        owner_worker_id = None
        selected_guided_rescue = False
        category_selection = None
        source_pool = None
        sampling_queue = None
        sampling_priority = None
        a_exclusions = None
        try:
            async with self.dispatch_condition:
                # Assign a slot before selecting a task so persisted task
                # affinity cannot concentrate a round on only some Workers.
                dispatch_tasks = self.selected_tasks
                slot_worker = None
                expected_source_pool = None
                if self.source_sampling is not None:
                    if self.central_gradient_round_mode:
                        from qwen_rl.contracts.sampling_topology import owner_map, owner_slots_for_ordinal
                        learner = services.config["learner"]
                        pairs = owner_map(learner)
                        self.source_sampling.worker_slots = tuple(
                            pairs[owner] for owner in owner_slots_for_ordinal(learner, ordinal))
                    round_size = len(self.source_sampling.worker_slots)
                    round_index, slot = divmod(ordinal, round_size)
                    if round_index != self.source_round_index:
                        self.source_round_plan = self.source_sampling.round_sources()
                        self.source_round_index = round_index
                    expected_source_pool = self.source_round_plan[slot]
                if self.central_gradient_round_mode:
                    from qwen_rl.contracts.sampling_topology import owner_map, owner_slots_for_ordinal
                    learner = services.config["learner"]
                    pairs = owner_map(learner)
                    slots = owner_slots_for_ordinal(learner, ordinal)
                    slot_worker = pairs[slots[ordinal % len(slots)]]
                    if self.source_sampling is not None:
                        dispatch_tasks = self.source_sampling.tasks_for(expected_source_pool)
                    dispatch_tasks = [task for task in dispatch_tasks
                                      if execution.task_worker_owners.get(task.task_id)
                                      in (None, slot_worker)]
                    if not dispatch_tasks:
                        raise GroupGenerationError(
                            f"no training task available for source {expected_source_pool or 'any'} "
                            f"and Worker slot {slot_worker}"
                        )
                if self.task_sampling is not None:
                    prefer_active = self.task_sampling.active_priority_for_dispatch(
                        ordinal, execution.needed_groups,
                    )
                    sampling_priority = "A_first" if prefer_active else "B_arc2_only"
                    selection, a_exclusions = await self._select_active_task(
                        snapshot, dispatch_tasks, slot_worker, ordinal, prefer_active=prefer_active,
                    )
                    if selection is None:
                        record.setdefault("sampling_skips", []).append({
                            "dispatch_ordinal": ordinal, "owner_worker_id": slot_worker,
                            "policy_version": snapshot.version,
                            "reason": "no_eligible_arc2_task",
                        })
                        raise GroupGenerationError("no eligible task in the active ARC2 sampling queue")
                    selected_task, sampling_queue = selection
                    execution.task_index += 1
                    selected_from_cache = False
                    selected_guided_rescue = sampling_queue == "A"
                    source_pool = "arc2_train"
                while self.task_sampling is None:
                    if not self.bepa_session.enabled:
                        use_category_sampling = (
                            self.category_sampler is not None
                            and (self.source_sampling is None
                                 or expected_source_pool == "synthetic_arc")
                        )
                        if use_category_sampling:
                            try:
                                selected_task, category_selection = self.sample_category_task(dispatch_tasks, snapshot)
                            except GroupGenerationError:
                                if self.central_gradient_round_mode or not execution.active_task_ids:
                                    raise
                                await self.dispatch_condition.wait()
                                continue
                            next_index = execution.task_index + 1
                        elif self.source_sampling is not None:
                            selected_task, next_index, selected_from_cache = select_dispatch_task(
                                dispatch_tasks,
                                execution.task_index,
                                active_task_ids=execution.active_task_ids,
                                accepted_task_ids=execution.accepted_task_ids_by_version[snapshot.version],
                                pending_cache_task_ids=execution.pending_cache_task_ids,
                                historical_seen_task_ids=execution.historical_seen_task_ids,
                                ready_cache_task_id=None,
                            )
                        else:
                            selected_task = dispatch_tasks[execution.task_index % len(dispatch_tasks)]
                            next_index = execution.task_index + 1
                        selected_from_cache = False
                        execution.task_index = next_index
                        break
                    excluded_task_ids = (
                        execution.active_task_ids
                        | execution.accepted_task_ids_by_version[snapshot.version]
                        | execution.pending_cache_task_ids
                    )
                    if slot_worker is not None:
                        excluded_task_ids |= {
                            task.task_id for task in self.selected_tasks
                            if (
                                self.source_sampling is not None
                                and (self.source_sampling.pool_for(task) != expected_source_pool
                                     or execution.task_worker_owners.get(task.task_id) != slot_worker)
                            ) or (
                                self.source_sampling is None
                                and execution.task_worker_owners.get(task.task_id)
                                not in (None, slot_worker)
                            )
                        }
                    rescue_task_id = execution.pop_rescue_ready_task(excluded_task_ids)
                    if rescue_task_id is not None:
                        selected_task = next(
                            (task for task in self.selected_tasks
                             if task.task_id == rescue_task_id),
                            None,
                        )
                        if selected_task is None:
                            raise RuntimeError(
                                "queued BEPA rescue task is outside the selected training allowlist"
                            )
                        execution.task_index += 1
                        selected_from_cache = False
                        selected_guided_rescue = True
                        break
                    ready_cache_task_id = None
                    if not self.central_gradient_round_mode:
                        ready_cache_task_id = await asyncio.to_thread(
                            self.bepa_session.bepa.ready_task,
                            [task.task_id for task in self.selected_tasks],
                            services.current_policy_version(),
                            excluded_task_ids,
                        )
                    try:
                        if (self.category_sampler is not None and ready_cache_task_id is None
                                and (self.source_sampling is None
                                     or expected_source_pool == "synthetic_arc")):
                            selected_task, category_selection = self.sample_category_task(dispatch_tasks, snapshot)
                            next_index, selected_from_cache = execution.task_index + 1, False
                        else:
                            selected_task, next_index, selected_from_cache = select_dispatch_task(
                                dispatch_tasks,
                                execution.task_index,
                                active_task_ids=execution.active_task_ids,
                                accepted_task_ids=execution.accepted_task_ids_by_version[snapshot.version],
                                pending_cache_task_ids=execution.pending_cache_task_ids,
                                historical_seen_task_ids=execution.historical_seen_task_ids,
                                ready_cache_task_id=ready_cache_task_id,
                            )
                        execution.task_index = next_index
                        break
                    except GroupGenerationError:
                        if (self.central_gradient_round_mode
                                or (not execution.active_task_ids
                                    and not execution.pending_cache_task_ids)):
                            raise
                        await self.dispatch_condition.wait()

                task_id = selected_task.task_id
                if self.source_sampling is not None:
                    source_pool = self.source_sampling.pool_for(selected_task)
                    if source_pool != expected_source_pool:
                        raise RuntimeError("selected task differs from its mixed source dispatch slot")
                    self.source_sampling.note_dispatch(source_pool)
                if slot_worker is not None:
                    previous_owner = execution.task_worker_owners.setdefault(task_id, slot_worker)
                    if previous_owner != slot_worker:
                        raise RuntimeError("selected task differs from its Worker slot")
                owner_worker_id = self.task_worker_owner(task_id)
                execution.active_task_counts[task_id] = (
                    execution.active_task_counts.get(task_id, 0) + 1
                )
                execution.active_task_ids.add(task_id)
                execution.historical_seen_task_ids.add(task_id)
                group_id = f"{record['run_id']}-{execution.group_index}-{uuid.uuid4().hex[:8]}"
                execution.group_index += 1

            dispatch_record = {
                "dispatch_ordinal": ordinal,
                "group_id": group_id,
                "task_id": selected_task.task_id,
                **({"source_pool": source_pool} if source_pool is not None else {}),
                "owner_worker_id": owner_worker_id,
                "behavior_version": snapshot.version,
                "selected_ready_cache": selected_from_cache,
                "selected_guided_rescue": selected_guided_rescue,
                "category_selection": category_selection,
                **({"sampling_queue": sampling_queue} if sampling_queue is not None else {}),
                **({"sampling_priority": sampling_priority} if sampling_priority is not None else {}),
                **({"a_exclusions": a_exclusions} if a_exclusions is not None else {}),
                "guided_attempt_status": "not_requested_by_queue",
                "started_at": time.time(),
            }
            record.setdefault("stream_dispatches", []).append(dispatch_record)
            if self.central_gradient_round_mode and self.bepa_session.enabled:
                self.bepa_session.prioritize_plan(selected_task.task_id)
            try:
                async with services.generation_gate.lease():
                    services.change_active_sampling_groups(1)
                    try:
                        guided_attempt = None
                        if (self.central_gradient_round_mode
                                and self.bepa_session.enabled):
                            dispatch_record["guided_attempt_status"] = "preparing"
                            guided_attempt = await self.bepa_session.prepare_guided_rescue(
                                selected_task, snapshot, group_id=group_id,
                            )
                            dispatch_record["guided_attempt_status"] = (
                                "started" if guided_attempt is not None else "no_eligible_plan_or_claim"
                            )
                        if guided_attempt is not None and (
                            guided_attempt.get("task") is None
                            or guided_attempt["task"].task_id != selected_task.task_id
                            or guided_attempt.get("snapshot") != snapshot
                            or guided_attempt.get("owner_worker_id") != owner_worker_id
                        ):
                            await self.bepa_session.abort_guided_rescue(
                                guided_attempt,
                                reason="guided rescue attempt changed task, policy, or Worker owner",
                            )
                            raise RuntimeError(
                                "guided rescue attempt changed task, policy, or Worker owner"
                            )
                        generate_kwargs = {
                            "worker_pool": services.worker_pool,
                            "diagnostic_course": (
                                services.diagnostic_course
                                if execution.request.kind in {"selftest", "diagnostic"}
                                else None
                            ),
                        }
                        if owner_worker_id is not None:
                            generate_kwargs["owner_worker_id"] = owner_worker_id
                        if guided_attempt is not None:
                            generate_kwargs["guided_plan"] = guided_attempt["plan"]
                        try:
                            result = await services.generate_group(
                                services.engine,
                                services.managers,
                                services.config,
                                selected_task,
                                group_id,
                                snapshot,
                                **generate_kwargs,
                            )
                        except BaseException as exc:
                            if guided_attempt is not None:
                                await self.bepa_session.abort_guided_rescue(
                                    guided_attempt,
                                    reason=f"{type(exc).__name__}: {str(exc)[:300]}",
                                )
                            raise
                        if guided_attempt is not None:
                            await self.bepa_session.finish_guided_rescue(
                                result, guided_attempt,
                            )
                    finally:
                        services.change_active_sampling_groups(-1)
            except GroupGenerationError as exc:
                record["groups"].append({
                    "group_id": group_id,
                    "task_id": selected_task.task_id,
                    **({"source_pool": source_pool} if source_pool is not None else {}),
                    "owner_worker_id": owner_worker_id,
                    "source": ("environment_fixture" if selected_task is self.fixture_task
                               else services.environment_name),
                    "status": "discarded",
                    "reason": type(exc).__name__,
                    **({"failure_kind": "infrastructure", "failed_worker_ids": list(exc.worker_ids),
                        "optimizer_consumed": False}
                       if isinstance(exc, GroupInfrastructureError) else {}),
                    "error": str(exc)[:300],
                    "policy_version": snapshot.version,
                    "dispatch_ordinal": ordinal,
                    "started_at": dispatch_record["started_at"],
                    "finished_at": time.time(),
                })
                raise
            result.update({
                "task": selected_task,
                "owner_worker_id": owner_worker_id,
                "group_id": group_id,
                "dispatch_ordinal": ordinal,
                "policy_snapshot": snapshot,
                "dispatch_started_at": dispatch_record["started_at"],
                "selected_ready_cache": selected_from_cache,
                "selected_guided_rescue": selected_guided_rescue,
                "guided_plan_applied": guided_attempt is not None,
                "category_selection": category_selection,
                **({"sampling_queue": sampling_queue} if sampling_queue is not None else {}),
                **({"sampling_priority": sampling_priority} if sampling_priority is not None else {}),
                **({"a_exclusions": a_exclusions} if a_exclusions is not None else {}),
                "guided_attempt_status": dispatch_record["guided_attempt_status"],
                **({"source_pool": source_pool} if source_pool is not None else {}),
            })
            if guided_attempt is not None:
                result["guided_rescue_attempt_id"] = guided_attempt["attempt_id"]
                result["replacement_of_group_id"] = guided_attempt.get(
                    "replacement_of_group_id",
                )
            generation_succeeded = True
            keep_policy_ref = True
            return result
        finally:
            if selected_task is not None and not generation_succeeded:
                await self.remove_active_task(selected_task.task_id)
            if not keep_policy_ref:
                execution.references.release(snapshot)

    async def accept_group_result(self, result: dict[str, Any]) -> bool | GroupDisposition:
        try:
            return await self._accept_group_result(result)
        finally:
            await self.remove_active_task(result["task"].task_id)

    async def _accept_group_result(self, result: dict[str, Any]) -> bool | GroupDisposition:
        execution = self.execution
        services = self.services
        record = execution.record
        snapshot: PolicySnapshot = result["policy_snapshot"]
        trajectories = result["trajectories"]
        group_metrics = result.get("metrics", {})
        worker_generation = group_metrics.get("worker_generation", {})
        if result["policy_version"] != snapshot.version:
            raise RuntimeError("group changed behavior policy before learner claim")
        fence = getattr(execution, "checkpoint_sampling_fences", {}).get(snapshot.version)
        if fence is not None:
            # No speculative native observation may graduate an A task before
            # the policy that produced it survives its full checkpoint write.
            await fence.wait(snapshot)
        environment_metrics = record["metrics"].setdefault(
            services.metrics_key, {"scored_episodes": 0, "groups_completed": 0},
        )
        environment_metrics["scored_episodes"] += int(
            group_metrics.get("episode_count", len(trajectories))
        )
        environment_metrics["groups_completed"] += 1
        for name, value in group_metrics.get("values", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                environment_metrics[name] = environment_metrics.get(name, 0.0) + float(value)
        for worker_id, sample_counts in worker_generation.items():
            worker_metrics = record["metrics"]["workers"].setdefault(
                worker_id, {"generation_calls": 0, "generation_seconds": 0.0},
            )
            worker_metrics["generation_calls"] += int(sample_counts.get("generation_calls", 0))
            worker_metrics["generation_seconds"] += float(sample_counts.get("generation_seconds", 0.0))

        task = result["task"]
        group_record = {
            "group_id": result["group_id"],
            "task_id": task.task_id,
            **({"source_pool": result.get("source_pool")} if result.get("source_pool") is not None else {}),
            "source": "environment_fixture" if task is self.fixture_task else services.environment_name,
            "diagnostic_course": record["diagnostic_course"],
            "policy_version": snapshot.version,
            "adapter_sha256": snapshot.adapter_sha256,
            "dispatch_ordinal": result["dispatch_ordinal"],
            "started_at": result["dispatch_started_at"],
            "finished_at": time.time(),
            "selected_ready_cache": result["selected_ready_cache"],
            "selected_guided_rescue": result.get("selected_guided_rescue", False),
            "guided_plan_applied": result.get("guided_plan_applied", False),
            "category_selection": result.get("category_selection"),
            **({"sampling_queue": result["sampling_queue"]} if "sampling_queue" in result else {}),
            **({"sampling_priority": result["sampling_priority"]} if "sampling_priority" in result else {}),
            **({"a_exclusions": result["a_exclusions"]} if "a_exclusions" in result else {}),
            "guided_attempt_status": result.get("guided_attempt_status"),
            **({"family": getattr(task, "family")} if hasattr(task, "family") else {}),
            **({"category": category(task)} if services.environment_name == "arc" else {}),
            "group_size": len(trajectories),
            "worker_ids": result["worker_ids"],
            "generation_worker_ids": result.get("generation_worker_ids", result["worker_ids"]),
            "owner_worker_id": result.get("owner_worker_id"),
            "rewards": result["rewards"],
            "worker_generation_calls": {
                key: int(value.get("generation_calls", 0))
                for key, value in worker_generation.items()
            },
            "status": "scored",
        }
        if "exact" in result:
            group_record["exact"] = result["exact"]
        if "costs" in result:
            group_record["costs"] = result["costs"]
        if "initial_prompt_prefix_sha256" in result:
            group_record["initial_prompt_prefix_sha256"] = result["initial_prompt_prefix_sha256"]
        annotations = group_metrics.get("annotations")
        if annotations:
            group_record["environment_metrics"] = dict(annotations)
        if result.get("guided_trajectory") is not None:
            guided = result["guided_trajectory"]
            group_record["guided_candidate"] = {
                "worker_id": result.get("guided_worker_id"),
                "episode_id": guided.get("episode_id"),
                "reward": guided.get("reward"),
                "exact": guided.get("exact"),
                "cost": guided.get("cost"),
                "policy_version": guided.get("policy_version"),
            }
        owner_worker_id = result.get("owner_worker_id")
        if owner_worker_id is not None:
            from ...worker.episode_queue import sampling_group_owner
            sampling_group_owner(
                result, getattr(getattr(services.engine, 'store', None), 'episode_queue', None),
            )
        group_record["trajectory_artifact"] = services.persist_group(
            services.run_dir,
            result["group_id"],
            result["trajectories"],
            result["worker_ids"],
            phase="pre_update",
            policy_version=snapshot.version,
            adapter_sha256=snapshot.adapter_sha256,
            base_fingerprint=snapshot.base_fingerprint,
        )
        record["groups"].append(group_record)
        result["record"] = group_record

        if (self.category_sampler is not None
                and (self.source_sampling is None or result.get("source_pool") == "synthetic_arc")
                and not result.get('selected_ready_cache')
                and not result.get('selected_guided_rescue')):
            if len(result['exact']) != int(services.config['rollout']['group_size']):
                raise ValueError('category history requires the complete native candidate group')
            self.category_sampler.observe(task, result['group_id'], result['exact'])

        zero_variance = statistics.pstdev(result.get("training_rewards", result["rewards"])) == 0.0
        no_positive_reward = not any(reward > 0 for reward in result["rewards"])
        disposition = await self.bepa_session.observe_scored_group(
            result,
            zero_variance=zero_variance or no_positive_reward,
        )
        for key in (
            "guided_rescue_replay",
            "guided_rescue_attempt_id",
            "guided_plan_id",
            "guided_plan_job_id",
            "guided_cache_entry_id",
            "guided_row_training_disposition",
            "guided_rescue_error",
            "replacement_of_group_id",
        ):
            if key in result:
                group_record[key] = result[key]
        if no_positive_reward and not result.get("replay_pending"):
            record["filtered_groups"] += 1
            group_record.update(
                status=("filtered_zero_variance" if zero_variance else "filtered_nonpositive_reward"),
                filter_reason="no_positive_reward", original_group_is_optimizer_input=False,
                optimizer_consumed=False,
            )
            return False
        if disposition is GroupDisposition.DEFERRED:
            return disposition
        if zero_variance and not result.get("replay_pending"):
            record["filtered_groups"] += 1
            group_record["status"] = (
                "discarded_waiting_for_verified_guidance"
                if result.get("rescue_waiting_guided")
                else "filtered_zero_variance"
            )
            return False
        execution.accepted_task_ids_by_version[snapshot.version].add(task.task_id)
        return True

    async def on_error(self, exc: Exception) -> None:
        if not isinstance(exc, GroupGenerationError):
            raise exc
        self.execution.record["filtered_groups"] += 1
        if isinstance(exc, GroupInfrastructureError):
            record = self.execution.record
            record["infrastructure_failed_groups"] = record.get("infrastructure_failed_groups", 0) + 1

    async def on_filtered(self, result: dict[str, Any] | None, reason: str) -> None:
        if result is None:
            return
        execution = self.execution
        record = execution.record
        snapshot = result["policy_snapshot"]
        task = result["task"]
        version = snapshot.version
        execution.accepted_task_ids_by_version[version].discard(task.task_id)
        if result.get("replay_pending"):
            execution.pending_cache_task_ids.discard(task.task_id)
        group_record = result.get("record")
        if group_record is not None and reason == "claim_validation_failed":
            group_record["status"] = "filtered_replay_claim"
            group_record["filter_reason"] = "cache_unavailable_or_duplicate"
            record["filtered_groups"] += 1
        elif group_record is not None and reason == "group_filtered":
            group_record.setdefault("filter_reason", "group_filter")
        elif group_record is not None and reason == "claim_prepare_exception":
            group_record.update(
                status="filtered_round_claim_prepare",
                filter_reason="claim_prepare_exception",
            )
            record["filtered_groups"] += 1
        elif group_record is not None and reason == "round_deferred_filtered":
            group_record.update(
                status="filtered_round_deferred",
                filter_reason="round_deferred_filtered",
            )
            record["filtered_groups"] += 1
        elif group_record is not None and reason.startswith("rescue_hold_"):
            group_record.update(
                status="filtered_zero_variance",
                filter_reason=reason,
                rescue_state="released_without_cache",
            )
            record["filtered_groups"] += 1
        execution.references.release(snapshot)
