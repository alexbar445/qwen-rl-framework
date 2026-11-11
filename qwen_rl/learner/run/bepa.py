"""Learner-run BEPA seed and replay lifecycle orchestration."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from ...bepa.seed_scheduler import (
    RunSeedScheduler,
    SeedAttemptResult,
    SeedAwareGenerationGate,
)
from ...rollout.groups import PolicySnapshot
from ...rollout.pipeline import GroupDisposition
from ..backend import CentralGradientRunEngineProtocol, LearnerRunEngineProtocol
from .state import PolicyReferenceLedger, RunExecutionState


class BepaSeedService:
    """Own verified-plan seed claims, guided episodes, and cache maintenance."""

    def __init__(
        self,
        *,
        bepa: Any | None,
        engine: LearnerRunEngineProtocol,
        policies: dict[int, dict[str, Any]],
        policy_refs: dict[int, int],
        generate_episode: Callable[..., Awaitable[dict[str, Any]]],
        build_prefix: Callable[..., Awaitable[list[int]]],
        sync_workers: Callable[[dict[str, Any]], Awaitable[dict[str, dict[str, Any]]]],
    ) -> None:
        self.bepa = bepa
        self.engine = engine
        self.policies = policies
        self.policy_refs = policy_refs
        self.generate_episode = generate_episode
        self.build_prefix = build_prefix
        self.sync_workers = sync_workers

    async def seed_plan_once(
        self,
        task: Any,
        plan_record: Any,
        snapshot: PolicySnapshot,
        *,
        run_id: str,
        task_allowlist: list[str],
        generation_gate: SeedAwareGenerationGate | None = None,
        observability: dict[str, Any] | None = None,
        call_observation: dict[str, Any] | None = None,
        phase: str = "continuous",
    ) -> tuple[SeedAttemptResult, Any | None]:
        """Run one fenced whole-episode student seed from a verified teacher plan."""
        bepa = self.bepa
        if bepa is None:
            return SeedAttemptResult.SKIPPED, None
        attempt_id = uuid.uuid4().hex
        attempt_observation = None
        if observability is not None:
            attempt_observation = {
                "attempt_id": attempt_id,
                "plan_id": plan_record.plan_id,
                "plan_job_id": plan_record.job_id,
                "task_id": task.task_id,
                "phase": phase,
                "policy_version": snapshot.version,
                "adapter_sha256": snapshot.adapter_sha256,
                "started_at": time.time(),
                "finished_at": None,
                "status": "claiming",
                "guided_raw": {
                    "episode_id": None,
                    "exact": None,
                    "raw_artifact_ref": None,
                    "raw_row_kind": "guided",
                    "projection_row_kind": "guided_replay",
                    "source_origin": "student",
                    "construction": "native",
                    "conditioning": "task_plan",
                    "observability_only": True,
                    "raw_artifact_is_optimizer_input": False,
                    "projection_may_be_reserved_for_training": True,
                },
            }
            if call_observation is not None:
                call_observation.setdefault("attempts", []).append(attempt_observation)
            observability.setdefault("attempts", []).append(attempt_observation)
            observability.setdefault("guided_raw_episodes", []).append(
                attempt_observation["guided_raw"]
            )

        references = PolicyReferenceLedger(self.engine, self.policy_refs)
        references.acquire(snapshot)
        seed_claim = None
        cache_created = False
        cancellation_recorded = False
        try:
            max_attempts = int(bepa.settings["seed"]["attempts_per_plan"])
            claim_call = asyncio.create_task(asyncio.to_thread(
                bepa.store.claim_seed_plan,
                plan_record.plan_id,
                task_allowlist=[key for key in task_allowlist if key == plan_record.task_key],
                run_id=run_id,
                attempt_id=attempt_id,
                engine_epoch=bepa.epoch,
                current_policy_version=int(self.engine.policy_version),
                policy_version=snapshot.version,
                max_attempts=max_attempts,
            ))
            try:
                seed_claim = await asyncio.shield(claim_call)
            except asyncio.CancelledError:
                try:
                    seed_claim = await claim_call
                except Exception:  # noqa: BLE001 - cancellation path must account for a failed claim
                    seed_claim = None
                if seed_claim is not None:
                    await asyncio.to_thread(
                        bepa.store.mark_seed_failed,
                        seed_claim,
                        reason="cancelled",
                        max_attempts=max_attempts,
                    )
                    cancellation_recorded = True
                raise
            if seed_claim is None:
                if attempt_observation is not None:
                    attempt_observation.update(
                        status="skipped",
                        finished_at=time.time(),
                        skip_reason="plan_or_cache_changed_before_claim",
                    )
                return SeedAttemptResult.SKIPPED, None
            if attempt_observation is not None:
                attempt_observation.update(
                    status="running",
                    generation_id=seed_claim.generation_id,
                    attempt_number=seed_claim.attempt_number,
                    fence=seed_claim.fence,
                )
            if observability is not None:
                observability["seed_attempt_count"] = int(
                    observability.get("seed_attempt_count", 0)
                ) + 1

            bepa.mode = "seed_maintenance"
            plan = bepa.artifacts.read_json(seed_claim.plan_artifact_ref)["plan"]
            group_id = "bepa-seed-" + attempt_id

            async def generate() -> dict[str, Any]:
                return await self.generate_episode(task, group_id, snapshot, plan)

            if generation_gate is None:
                result = await generate()
            else:
                async with generation_gate.lease(seed=True):
                    result = await generate()
            row = result["trajectories"][0]
            if attempt_observation is not None:
                attempt_observation["guided_raw"].update(
                    episode_id=row.get("episode_id"), exact=bool(row.get("exact")),
                )
            if not row["exact"]:
                raise ValueError("guided_student_not_exact")
            prefix = await self.build_prefix(task)
            cache_write = asyncio.create_task(asyncio.to_thread(
                bepa.cache_student,
                task,
                row,
                snapshot,
                prefix,
                guided=True,
                seed_claim=seed_claim,
            ))
            try:
                cache = await asyncio.shield(cache_write)
            except asyncio.CancelledError:
                try:
                    cache = await cache_write
                    cache_created = True
                    if attempt_observation is not None:
                        attempt_observation["guided_raw"].update(
                            raw_artifact_ref=cache.raw_artifact_ref,
                            projection_artifact_ref=cache.projection_artifact_ref,
                            cache_entry_id=cache.cache_entry_id,
                            row_kind=cache.row_kind,
                            committed_uses_at_seed=cache.committed_uses,
                        )
                        attempt_observation.update(
                            status="cache_ready_run_cancelled", finished_at=time.time(),
                        )
                    if observability is not None:
                        observability["guided_cache_created_count"] = int(
                            observability.get("guided_cache_created_count", 0)
                        ) + 1
                except Exception:  # noqa: BLE001, S110 - outer cancellation records the fenced failure
                    # The outer cancellation handler records the single fenced
                    # failure after this non-cancellable store call has drained.
                    pass
                raise
            cache_created = True
            if attempt_observation is not None:
                attempt_observation["guided_raw"].update(
                    raw_artifact_ref=cache.raw_artifact_ref,
                    projection_artifact_ref=cache.projection_artifact_ref,
                    cache_entry_id=cache.cache_entry_id,
                    row_kind=cache.row_kind,
                    committed_uses_at_seed=cache.committed_uses,
                )
                attempt_observation.update(status="cache_ready", finished_at=time.time())
            if observability is not None:
                observability["guided_cache_created_count"] = int(
                    observability.get("guided_cache_created_count", 0)
                ) + 1
            return SeedAttemptResult.READY, cache
        except asyncio.CancelledError:
            if seed_claim is not None and not cache_created and not cancellation_recorded:
                await asyncio.to_thread(
                    bepa.store.mark_seed_failed,
                    seed_claim,
                    reason="cancelled",
                    max_attempts=int(bepa.settings["seed"]["attempts_per_plan"]),
                )
            if attempt_observation is not None:
                attempt_observation.update(status="cancelled", finished_at=time.time())
            raise
        except Exception as exc:  # noqa: BLE001 - convert any seed execution failure into a durable attempt result
            if seed_claim is not None and not cache_created:
                await asyncio.to_thread(
                    bepa.store.mark_seed_failed,
                    seed_claim,
                    reason=type(exc).__name__ + ": " + str(exc)[:240],
                    max_attempts=int(bepa.settings["seed"]["attempts_per_plan"]),
                )
            if attempt_observation is not None:
                attempt_observation.update(
                    status="failed",
                    finished_at=time.time(),
                    error={"type": type(exc).__name__, "message": str(exc)[:240]},
                )
            return SeedAttemptResult.FAILED, None
        finally:
            try:
                references.release(snapshot)
            finally:
                bepa.mode = "idle"

    async def seed_maintenance(
        self,
        selected_tasks: list[Any],
        snapshot: PolicySnapshot,
        *,
        wait: bool = False,
        enqueue: bool = False,
        limit: int = 1,
        max_seed_attempts: int | None = None,
        drain_workers: bool = True,
        observability: dict[str, Any] | None = None,
        phase: str = "maintenance",
    ) -> int:
        bepa = self.bepa
        if bepa is None:
            return 0
        if self.engine.config.get("update_backend") == "central_gradient":
            raise RuntimeError(
                "central_gradient does not support singleton BEPA seed maintenance; "
                "verified plans are consumed only inside a full native-plus-guided task-group round"
            )
        if (max_seed_attempts is not None
                and (isinstance(max_seed_attempts, bool)
                     or not isinstance(max_seed_attempts, int)
                     or max_seed_attempts < 1)):
            raise ValueError("max_seed_attempts must be a positive integer when provided")
        call_observation = None
        if observability is not None:
            call_observation = {
                "phase": phase,
                "observability_only": True,
                "engine_epoch": bepa.epoch,
                "policy_version": snapshot.version,
                "adapter_sha256": snapshot.adapter_sha256,
                "requested_cache_count": limit,
                "max_seed_attempts": max_seed_attempts,
                "started_at": time.time(),
                "finished_at": None,
                "status": "running",
                "seeded_cache_count": 0,
                "attempts": [],
            }
            observability.setdefault("maintenance_calls", []).append(call_observation)
        candidates = selected_tasks[:int(bepa.settings["bootstrap"]["max_tasks"])]
        for task in candidates:
            bepa.register(task)
        ids = {bepa.task_keys[task.task_id]: task for task in candidates}
        keys = list(ids)
        ready_keys = set()
        for task_key in keys:
            ready = await asyncio.to_thread(
                bepa.store.peek_ready,
                [task_key],
                engine_epoch=bepa.epoch,
                current_policy_version=int(self.engine.policy_version),
            )
            if ready is not None:
                ready_keys.add(task_key)
        seeded = min(len(ready_keys), limit)
        pending_ids = {key: task for key, task in ids.items() if key not in ready_keys}
        if seeded >= limit:
            if call_observation is not None:
                call_observation.update(
                    status="ready_cache_sufficient",
                    seeded_cache_count=seeded,
                    finished_at=time.time(),
                )
            return seeded

        def mark_budget_terminal(status: str, budget: dict[str, Any]) -> None:
            operation = bepa.operation
            if isinstance(operation, dict) and operation.get("status") == "running":
                operation["status"] = status
                operation["budget"] = dict(budget)

        async def available_seed_plans():
            if not pending_ids:
                return []
            current_policy_version = int(self.engine.policy_version)
            keys = list(pending_ids)
            await asyncio.to_thread(
                bepa.store.refresh_seed_plans,
                keys,
                engine_epoch=bepa.epoch,
                current_policy_version=current_policy_version,
            )
            return await asyncio.to_thread(
                bepa.store.available_plans,
                keys,
                engine_epoch=bepa.epoch,
                current_policy_version=current_policy_version,
                limit=max(1000, len(keys) * 4),
            )

        plans = await available_seed_plans()
        if enqueue:
            budget = await asyncio.to_thread(bepa.store.budget_snapshot, bepa.scope)
            if budget.get("scope_id") != bepa.scope:
                raise RuntimeError("bootstrap budget snapshot did not match the configured scope")
            if budget["usage_unknown"] and not plans:
                mark_budget_terminal("budget_unknown", budget)
                return seeded
            jobs = await asyncio.to_thread(
                bepa.store.list_jobs,
                task_allowlist=list(pending_ids),
            ) if pending_ids else []
            jobs = [job for job in jobs if job.budget_scope == bepa.scope]
            active_jobs = any(job.state in {"teacher_running", "source_submitted", "source_verified"}
                              for job in jobs)
            if not budget["usage_unknown"]:
                if budget["remaining_tokens"] <= 0 and not plans and not active_jobs:
                    mark_budget_terminal("budget_exhausted", budget)
                    return seeded
                if budget["remaining_tokens"] > 0 or active_jobs:
                    planned_keys = {plan.task_key for plan in plans}
                    for task_key, task in pending_ids.items():
                        if task_key not in planned_keys:
                            bepa.enqueue(task)

        deadline = time.monotonic() + float(
            bepa.settings["teacher"]["budget"]["job_timeout_seconds"]
        ) * max(1, len(candidates))
        attempted_seed_count = 0
        try:
            while seeded < limit:
                if max_seed_attempts is not None and attempted_seed_count >= max_seed_attempts:
                    break
                plans = await available_seed_plans()
                if not plans:
                    if not wait or time.monotonic() >= deadline:
                        break
                    jobs = await asyncio.to_thread(
                        bepa.store.list_jobs, task_allowlist=list(pending_ids),
                    ) if pending_ids else []
                    jobs = [job for job in jobs if job.budget_scope == bepa.scope]
                    budget = await asyncio.to_thread(bepa.store.budget_snapshot, bepa.scope)
                    if budget.get("scope_id") != bepa.scope:
                        raise RuntimeError("bootstrap budget snapshot did not match the configured scope")
                    if budget["usage_unknown"]:
                        mark_budget_terminal("budget_unknown", budget)
                        break
                    active_jobs = any(job.state in {"teacher_running", "source_submitted", "source_verified"}
                                      for job in jobs)
                    if budget["remaining_tokens"] <= 0 and not active_jobs:
                        mark_budget_terminal("budget_exhausted", budget)
                        break
                    if jobs and all(job.state in {
                        "failed", "exhausted", "rejected", "outcome_unknown", "cancelled", "cache_ready",
                    } for job in jobs):
                        break
                    bepa.mode = "waiting_for_seed"
                    await asyncio.sleep(0.5)
                    continue
                plan_record = plans[0]
                task = ids[plan_record.task_key]
                status, _cache = await self.seed_plan_once(
                    task,
                    plan_record,
                    snapshot,
                    run_id=(
                        str(bepa.operation.get("operation_id"))
                        if isinstance(bepa.operation, dict)
                        and isinstance(bepa.operation.get("operation_id"), str)
                        else "bepa-bootstrap-" + uuid.uuid4().hex
                    ),
                    task_allowlist=list(ids),
                    observability=observability,
                    call_observation=call_observation,
                    phase=phase,
                )
                if status is not SeedAttemptResult.SKIPPED:
                    attempted_seed_count += 1
                if status is SeedAttemptResult.READY:
                    seeded += 1
                    ready_keys.add(plan_record.task_key)
                    pending_ids.pop(plan_record.task_key, None)
        finally:
            try:
                if drain_workers:
                    await self.sync_workers(self.policies[snapshot.version])
            finally:
                bepa.mode = "idle"
                if call_observation is not None:
                    call_observation["finished_at"] = time.time()
                    call_observation["seeded_cache_count"] = seeded
                    if call_observation["status"] == "running":
                        call_observation["status"] = "seeded" if seeded else "no_cache_seeded"
        return seeded


class BepaRunSession:
    """BEPA-specific row admission, claim reservations, and rescue scheduling."""

    def __init__(
        self,
        *,
        run_id: str,
        execution: RunExecutionState,
        bepa: Any | None,
        seed_service: BepaSeedService,
        selected_tasks: list[Any],
        policies: dict[int, dict[str, Any]],
        current_policy_version: Callable[[], int],
        serve_policy_version: Callable[[], int | None],
        generation_gate: SeedAwareGenerationGate,
        on_scheduler_change: Callable[[Any | None], None],
    ) -> None:
        self.run_id = run_id
        self.execution = execution
        self.bepa = bepa
        self.seed_service = seed_service
        self.selected_tasks = selected_tasks
        self.policies = policies
        self.current_policy_version = current_policy_version
        self.serve_policy_version = serve_policy_version
        self.generation_gate = generation_gate
        self.on_scheduler_change = on_scheduler_change
        self.enabled = bepa is not None and execution.request.kind == "training"
        engine_config = self.seed_service.engine.config
        self.central_gradient = (
            engine_config.get("update_backend") == "central_gradient"
        )
        self.task_sampling = None
        if (self.enabled and self.central_gradient
                and any(getattr(task, "pool", None) == "arc2_train" for task in selected_tasks)):
            from .task_sampling import ActiveTaskSampling
            self.task_sampling = ActiveTaskSampling(execution.record, selected_tasks)
            self.selected_tasks = list(self.task_sampling.tasks.values())
            execution.rescue_ready_task_ids[:] = [
                task_id for task_id in execution.rescue_ready_task_ids
                if task_id in self.task_sampling.tasks]
            for task_id in execution.rescue_ready_task_ids:
                self.task_sampling.offer(task_id)
        self.task_key_by_id = (
            {task.task_id: bepa.task_keys[task.task_id] for task in self.selected_tasks}
            if self.enabled else {}
        )
        self.selected_task_keys = list(self.task_key_by_id.values())
        self.task_by_key = {
            self.task_key_by_id[task.task_id]: task for task in self.selected_tasks
        } if self.enabled else {}
        self.plan_prefetch = None
        self._rescue_candidates_by_task: dict[str, dict[str, Any]] = {}
        self._rescue_origin_group_by_task: dict[str, str] = {}

    @property
    def stream(self) -> Any | None:
        return self.execution.stream

    def _central_worker_executor_ids(self) -> Mapping[str, str]:
        if not self.central_gradient:
            raise RuntimeError("Worker/executor ownership is only available on central_gradient")
        engine = cast(CentralGradientRunEngineProtocol, self.seed_service.engine)
        return engine.worker_executor_ids

    async def prepare_initial_collection(
        self, snapshot: PolicySnapshot,
    ) -> dict[str, Any]:
        """Reuse selected-task cache/plans before the first native rollout starts.

        This bounded preflight never enqueues or waits for a Teacher job. It may
        perform at most one real Qwen guided attempt from an already verified
        plan, and it skips that attempt when an eligible cache already exists.
        """
        if not self.enabled:
            return {"status": "disabled"}

        retired = await asyncio.to_thread(
            self.bepa.store.retired_rescue_task_keys,
            self.selected_task_keys, engine_epoch=self.bepa.epoch,
        )
        for task_id, task_key in self.task_key_by_id.items():
            if task_key in retired:
                self._drop_priority_rescue(task_id)
                if self.task_sampling is not None:
                    self.task_sampling.observe_progress(task_id, retired=True)
            elif (self.task_sampling is not None
                  and task_id in self.task_sampling.state["review"]):
                # Native observations may be newer than the last full checkpoint.
                self.task_sampling.observe_progress(task_id, retired=False, reopened=True)

        # A rescue job can outlive the run which discovered the failed group.
        # Restore its original Worker pin before any new task dispatch so a
        # ready plan/cache can never move that task to a different executor.
        owner_reader = getattr(self.bepa, "rescue_owner", None)
        registered_reader = getattr(self.bepa, "registered_rescue_owner", None)
        if callable(owner_reader):
            owners = self.execution.task_worker_owners
            for task in self.selected_tasks:
                owner = (registered_reader(task, self.task_key_by_id[task.task_id])
                         if callable(registered_reader) else owner_reader(task))
                if owner is None:
                    continue
                worker_id = owner.get("worker_id") if isinstance(owner, dict) else None
                if not isinstance(worker_id, str) or not worker_id:
                    raise RuntimeError("persisted BEPA rescue owner omitted its Worker ID")
                current = owners.get(task.task_id)
                if current is not None and current != worker_id:
                    raise RuntimeError("restored BEPA task owner conflicts with the current run owner")
                owners[task.task_id] = worker_id
                origin_ids = owner.get("original_group_ids", [])
                if origin_ids:
                    self._rescue_origin_group_by_task[task.task_id] = origin_ids[-1]
            self.execution.record["task_worker_owners"] = owners

        observability = self.execution.record.get("bepa_seed_observability")
        preparation: dict[str, Any] = {
            "status": "checking",
            "started_at": time.time(),
            "policy_version": snapshot.version,
            "max_seed_attempts": 1,
            "plan_window_limit": int(self.bepa.settings["bootstrap"]["max_tasks"]),
        }
        if isinstance(observability, dict):
            observability["initial_collection"] = preparation

        try:
            current_policy_version = self.current_policy_version()
            if current_policy_version != snapshot.version:
                raise RuntimeError("initial seed snapshot differs from the current learner policy")
            ready_cache = None if self.central_gradient else await asyncio.to_thread(
                self.bepa.ready_task,
                [task.task_id for task in self.selected_tasks],
                current_policy_version,
            )
            if ready_cache is not None:
                preparation.update(
                    status="ready_cache_available",
                    task_id=getattr(ready_cache, "task_id", ready_cache),
                )
                return preparation

            candidates = await self.discover_seed_plans()
            window_limit = preparation["plan_window_limit"]
            window: list[dict[str, Any]] = []
            window_task_ids: set[str] = set()
            for candidate in candidates:
                if candidate["task_id"] in window_task_ids:
                    continue
                window_task_ids.add(candidate["task_id"])
                window.append(candidate)
                if len(window) >= window_limit:
                    break
            preparation["candidate_window_task_count"] = len(window)
            if not window:
                preparation.update(status="no_existing_plan", finished_at=time.time())
                return preparation

            candidate = window[0]
            preparation.update(
                task_id=candidate["task_id"],
                plan_id=candidate["plan"].plan_id,
            )
            if self.central_gradient:
                for ready_candidate in window:
                    self._queue_guided_rescue(ready_candidate)
                preparation.update(status="whole_group_rescue_scheduled")
                return preparation

            status, cache = await self.seed_service.seed_plan_once(
                candidate["task"],
                candidate["plan"],
                snapshot,
                run_id=self.run_id,
                task_allowlist=self.selected_task_keys,
                generation_gate=self.generation_gate,
                observability=observability if isinstance(observability, dict) else None,
                phase="initial_collection",
            )
            preparation["status"] = (
                "seeded" if status is SeedAttemptResult.READY else status.value
            )
            if cache is not None:
                preparation["cache_entry_id"] = cache.cache_entry_id
            return preparation
        except asyncio.CancelledError:
            preparation.update(status="cancelled", finished_at=time.time())
            raise
        finally:
            preparation.setdefault("finished_at", time.time())

    def _drop_priority_rescue(self, task_id: str) -> None:
        self.execution.rescue_ready_task_ids[:] = [
            queued for queued in self.execution.rescue_ready_task_ids if queued != task_id]
        self.execution.pending_cache_task_ids.discard(task_id)
        self.execution.rescue_enqueued_task_ids.discard(task_id)
        self._rescue_candidates_by_task.pop(task_id, None)

    def _queue_guided_rescue(self, candidate: dict[str, Any]) -> bool:
        task_id = candidate.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("verified BEPA plan candidate omitted its training task ID")
        if self.task_sampling is not None and task_id not in self.task_sampling.tasks:
            return False
        if self.bepa.store.retired_rescue_task_keys(
                [self.task_key_by_id[task_id]], engine_epoch=self.bepa.epoch):
            self._drop_priority_rescue(task_id)
            if self.task_sampling is not None:
                self.task_sampling.observe_progress(task_id, retired=True)
            return False
        owners = self.execution.task_worker_owners
        owner_worker = owners.get(task_id)
        if owner_worker is not None:
            executor_map = self._central_worker_executor_ids()
            if not executor_map.get(owner_worker):
                raise RuntimeError("verified BEPA plan owner no longer maps to a gradient executor")
        # Plan-ready is the only transition out of this exclusion set. This
        # also covers plans found during initial collection after restart.
        self.execution.pending_cache_task_ids.discard(task_id)
        self._rescue_candidates_by_task[task_id] = candidate
        newly_queued = task_id not in self.execution.rescue_ready_task_ids
        self.execution.enqueue_rescue_ready_task(task_id)
        promoted = False
        if self.task_sampling is not None:
            promoted = self.task_sampling.offer_verified_plan(task_id)
        return newly_queued or promoted

    async def observe_scored_group(
        self,
        result: dict[str, Any],
        *,
        zero_variance: bool,
    ) -> GroupDisposition | None:
        if not self.enabled:
            return None
        bepa = self.bepa
        task = result["task"]
        snapshot = result["policy_snapshot"]
        progress = await asyncio.to_thread(bepa.observe_native_group, task, result, snapshot)
        if progress is not None:
            result['record']['native_rescue_progress'] = progress
            if self.task_sampling is not None:
                self.task_sampling.observe_progress(
                    task.task_id, retired=progress['retired'], reopened=progress['reopened'],
                    needs_practice=not all(result['exact']))
            if progress['retired'] or progress['reopened']:
                self._drop_priority_rescue(task.task_id)
            if progress['retired'] and not any(result['exact']):
                return None  # An already observed old failure cannot reopen the queue twice.
        if result.get("replay_pending"):
            result["record"]["status"] = "replay_candidate"
            result["record"]["replay_candidate"] = True
            self.execution.pending_cache_task_ids.add(task.task_id)
            return None
        if any(result["exact"]):
            await asyncio.to_thread(
                bepa.admit_group,
                result,
                task,
                snapshot,
                f"cache-source-{result['group_id']}",
                current_policy_version=self.current_policy_version(),
            )
            scheduler = self.execution.seed_scheduler
            if scheduler is not None:
                scheduler.wake()
        else:
            current_policy_version = self.current_policy_version()
            cache_ready = result.get("replay_pending") or (
                not self.central_gradient
                and bepa.ready_task([task.task_id], current_policy_version) is not None
            )
            if cache_ready:
                result["replay_pending"] = True
                result["record"]["status"] = "replay_candidate"
                result["record"]["replay_candidate"] = True
                self.execution.pending_cache_task_ids.add(task.task_id)
            else:
                budget_blocked = False
                if task.task_id not in self.execution.rescue_enqueued_task_ids:
                    owner_kwargs: dict[str, str] = {}
                    if self.central_gradient:
                        worker_ids = result.get("worker_ids")
                        if (not isinstance(worker_ids, list) or not worker_ids
                                or any(not isinstance(worker_id, str) or not worker_id
                                       for worker_id in worker_ids)):
                            raise RuntimeError("central BEPA rescue lost its native source Worker identity")
                        from ...worker.episode_queue import sampling_group_owner
                        owner_worker_id = sampling_group_owner(
                            result, getattr(getattr(self.seed_service.engine, 'store', None), 'episode_queue', None),
                        )
                        owners = self.execution.task_worker_owners
                        existing_owner = owners.get(task.task_id)
                        if existing_owner is not None and existing_owner != owner_worker_id:
                            raise RuntimeError("native BEPA rescue Worker differs from its durable task owner")
                        owners[task.task_id] = owner_worker_id
                        executor_map = self._central_worker_executor_ids()
                        owner_executor_id = executor_map.get(owner_worker_id)
                        if not isinstance(owner_executor_id, str) or not owner_executor_id:
                            raise RuntimeError("native BEPA source Worker has no explicit gradient executor")
                        owner_kwargs = {
                            "owner_worker_id": owner_worker_id,
                            "owner_executor_id": owner_executor_id,
                            "original_group_id": result["group_id"],
                        }
                    queued = await asyncio.to_thread(bepa.enqueue, task, **owner_kwargs)
                    if self.central_gradient and queued.get("status") == "queue_full":
                        result["rescue_queue_full"] = True
                    elif queued.get("status") == "budget_blocked":
                        budget_blocked = True
                        result["rescue_budget_blocked"] = True
                        result["rescue_budget_reason"] = queued.get("reason")
                    else:
                        self.execution.rescue_enqueued_task_ids.add(task.task_id)
                        self.execution.pending_cache_task_ids.add(task.task_id)
                        if self.central_gradient:
                            self._rescue_origin_group_by_task[task.task_id] = result["group_id"]
                if budget_blocked:
                    result["record"]["rescue_state"] = "budget_blocked"
                    result["record"]["rescue_status"] = "budget_blocked"
                    if result.get("rescue_budget_reason") is not None:
                        result["record"]["rescue_budget_reason"] = result["rescue_budget_reason"]
                    if not zero_variance:
                        result["record"]["original_group_is_optimizer_input"] = True
                    else:
                        result["record"]["original_group_is_optimizer_input"] = False
                    return None
                scheduler = self.execution.seed_scheduler
                if scheduler is not None:
                    scheduler.wake()
                result["rescue_hold_started_at"] = result["record"]["finished_at"]
                result["rescue_waiting_guided"] = not result.get("rescue_queue_full", False)
                if not zero_variance:
                    result["record"]["rescue_state"] = "queued_with_tool_progress"
                    result["record"]["original_group_is_optimizer_input"] = True
                    return None
                result["record"].update(
                    status=("discarded_waiting_for_verified_guidance"
                            if self.central_gradient else "rescue_waiting_guided"),
                    rescue_hold_started_at=result["record"]["finished_at"],
                    rescue_state=("waiting_for_verified_plan" if not result.get("rescue_queue_full")
                                  else "queue_full"),
                    original_group_is_optimizer_input=False if self.central_gradient else None,
                )
                if self.central_gradient:
                    # The failed original group completes this round as one
                    # filtered outcome. A later verified plan schedules a new
                    # whole native group under the saved Worker pin.
                    return None
                return GroupDisposition.DEFERRED
        return None

    async def prepare_guided_rescue(
        self,
        task: Any,
        snapshot: PolicySnapshot,
        *,
        group_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Claim one verified Teacher plan for the next whole native-plus-guided group."""
        if not self.enabled or not self.central_gradient:
            return None
        if snapshot.version != self.serve_policy_version():
            raise RuntimeError("central guided rescue snapshot differs from the currently served policy")
        task_key = self.bepa.task_keys.get(task.task_id)
        if not isinstance(task_key, str):
            raise RuntimeError("guided rescue task is outside the registered training allowlist")
        fence = getattr(self.execution, "checkpoint_sampling_fences", {}).get(snapshot.version)
        if fence is not None and task.task_id in fence.replay_task_ids:
            await fence.wait(snapshot)
        current_policy_version = self.current_policy_version()
        await asyncio.to_thread(
            self.bepa.store.refresh_seed_plans,
            [task_key], engine_epoch=self.bepa.epoch,
            current_policy_version=current_policy_version, limit=1, whole_group=True,
            include_retired=True, group_id=group_id,
        )
        candidates = await asyncio.to_thread(
            self.bepa.store.available_plans,
            [task_key],
            engine_epoch=self.bepa.epoch,
            current_policy_version=current_policy_version,
            limit=1,
            whole_group=True,
            include_retired=True,
        )
        if not candidates:
            return None
        plan_record = candidates[0]
        attempt_id = group_id or uuid.uuid4().hex
        references = PolicyReferenceLedger(self.seed_service.engine, self.seed_service.policy_refs)
        references.acquire(snapshot)
        claim = None
        try:
            claim_call = asyncio.create_task(asyncio.to_thread(
                self.bepa.store.claim_seed_plan,
                plan_record.plan_id,
                task_allowlist=[task_key],
                run_id=self.run_id,
                attempt_id=attempt_id,
                engine_epoch=self.bepa.epoch,
                current_policy_version=current_policy_version,
                policy_version=snapshot.version,
                max_attempts=int(self.bepa.settings["seed"]["attempts_per_plan"]),
                whole_group=True,
                include_retired=True,
            ))
            try:
                claim = await asyncio.shield(claim_call)
            except asyncio.CancelledError:
                try:
                    claim = await claim_call
                except Exception:  # noqa: BLE001 - cancellation must drain the durable claim call
                    claim = None
                raise
            if claim is None:
                references.release(snapshot)
                return None
            plan_payload = await asyncio.to_thread(
                self.bepa.artifacts.read_json,
                claim.plan_artifact_ref,
            )
            plan_text = plan_payload.get("plan") if isinstance(plan_payload, dict) else None
            if not isinstance(plan_text, str) or not plan_text.strip():
                raise RuntimeError("verified BEPA plan artifact omitted its plan text")
            attempt = {
                "claim": claim,
                "plan": plan_text,
                "snapshot": snapshot,
                "task": task,
                "attempt_id": attempt_id,
                "references": references,
                "plan_record": plan_record,
                "finished": False,
                "replacement_of_group_id": self._rescue_origin_group_by_task.get(task.task_id),
            }
            owners = self.execution.task_worker_owners
            owner_worker = owners.get(task.task_id)
            if owner_worker is not None:
                executor_map = self._central_worker_executor_ids()
                if not isinstance(executor_map.get(owner_worker), str):
                    raise RuntimeError("persisted rescue Worker no longer maps to a gradient executor")
                attempt["owner_worker_id"] = owner_worker
            observability = self.execution.record.get("bepa_seed_observability")
            if isinstance(observability, dict):
                observation = {
                    "attempt_id": attempt_id,
                    "plan_id": plan_record.plan_id,
                    "plan_job_id": plan_record.job_id,
                    "task_id": task.task_id,
                    "phase": "central_whole_group_rescue",
                    "policy_version": snapshot.version,
                    "adapter_sha256": snapshot.adapter_sha256,
                    "started_at": time.time(),
                    "finished_at": None,
                    "status": "running",
                    "whole_group": True,
                    "native_group_size": int(self.seed_service.engine.group_size),
                    "guided_raw": {
                        "episode_id": None,
                        "exact": None,
                        "raw_artifact_ref": None,
                        "raw_row_kind": "guided",
                        "projection_row_kind": "guided_replay",
                        "source_origin": "student",
                        "construction": "native",
                        "conditioning": "task_plan",
                        "observability_only": True,
                        "raw_artifact_is_optimizer_input": False,
                        "projection_may_be_reserved_for_training": True,
                    },
                }
                attempt["observation"] = observation
                observability.setdefault("attempts", []).append(observation)
                observability.setdefault("guided_raw_episodes", []).append(observation["guided_raw"])
                observability["seed_attempt_count"] = int(observability.get("seed_attempt_count", 0)) + 1
            self.bepa.mode = "whole_group_seed"
            return attempt
        except BaseException:
            if claim is None:
                references.release(snapshot)
            else:
                try:
                    await asyncio.to_thread(
                        self.bepa.store.mark_seed_failed,
                        claim,
                        reason="group_plan_preparation_failed",
                        max_attempts=int(self.bepa.settings["seed"]["attempts_per_plan"]),
                    )
                finally:
                    references.release(snapshot)
            raise

    async def finish_guided_rescue(
        self,
        result: dict[str, Any],
        attempt: dict[str, Any],
    ) -> Any | None:
        """Verify/project a fresh guided row alongside its complete native group."""
        if not self.enabled or not self.central_gradient:
            return None
        claim = attempt["claim"]
        task = attempt["task"]
        snapshot = attempt["snapshot"]
        observation = attempt.get("observation")
        cache = None
        try:
            native_workers = result.get("worker_ids")
            guided_worker = result.get("guided_worker_id")
            if not isinstance(native_workers, list) or not native_workers:
                raise RuntimeError("guided rescue group lost its native Worker leases")
            from ...worker.episode_queue import sampling_group_owner
            owner_worker = sampling_group_owner(
                result, getattr(getattr(self.seed_service.engine, 'store', None), 'episode_queue', None),
            )
            if not isinstance(guided_worker, str) or guided_worker != owner_worker:
                raise RuntimeError("guided rescue group lost its single typed Worker lease")
            executor_map = self._central_worker_executor_ids()
            owner_executor = executor_map.get(owner_worker)
            if not isinstance(owner_executor, str) or not owner_executor:
                raise RuntimeError("guided rescue Worker has no explicit gradient executor")
            owners = self.execution.task_worker_owners
            expected_owner = owners.get(task.task_id)
            if expected_owner is not None and expected_owner != owner_worker:
                raise RuntimeError("guided rescue left the original durable Worker owner")
            owners[task.task_id] = owner_worker
            await asyncio.to_thread(
                self.bepa.store.bind_rescue_owner,
                claim.job_id,
                task_key=claim.task_key,
                worker_id=owner_worker,
                executor_id=owner_executor,
                original_group_id=attempt.get("replacement_of_group_id"),
            )

            guided_row = result.get("guided_trajectory")
            if not isinstance(guided_row, dict):
                raise RuntimeError("whole-group guided candidate omitted its raw trajectory")
            if guided_row.get("exact") is not True:
                raise ValueError("guided_student_not_exact")
            prefix = await self.seed_service.build_prefix(task)
            trusted_summary: dict[str, Any] = {}
            cache = await asyncio.to_thread(
                self.bepa.cache_student,
                task,
                guided_row,
                snapshot,
                prefix,
                guided=True,
                seed_claim=claim,
                trusted_summary=trusted_summary,
            )
            result["training_rewards"] = self.bepa.training_group_rewards(
                task, [*result["trajectories"], {**guided_row, **trusted_summary}], snapshot.version,
            )
            training_has_variance = len(set(result["training_rewards"])) > 1
            result["guided_cache_entry_id"] = cache.cache_entry_id
            result["replay_pending"] = training_has_variance
            result["guided_rescue_replay"] = training_has_variance
            result["guided_plan_id"] = attempt["plan_record"].plan_id
            result["guided_plan_job_id"] = attempt["plan_record"].job_id
            result["guided_row_training_disposition"] = (
                "verified_guided_replay" if training_has_variance else "filtered_zero_variance"
            )
            observation_guided = {
                **observation["guided_raw"],
                "episode_id": guided_row.get("episode_id"),
                "exact": True,
                "raw_artifact_ref": cache.raw_artifact_ref,
                "projection_artifact_ref": cache.projection_artifact_ref,
                "cache_entry_id": cache.cache_entry_id,
                "row_kind": cache.row_kind,
                "committed_uses_at_seed": cache.committed_uses,
            } if isinstance(observation, dict) else None
            if isinstance(observation, dict):
                observation.update(
                    status="cache_ready",
                    finished_at=time.time(),
                    guided_raw=observation_guided,
                )
            observability = self.execution.record.get("bepa_seed_observability")
            if isinstance(observability, dict):
                observability["guided_cache_created_count"] = int(
                    observability.get("guided_cache_created_count", 0)
                ) + 1
            self._rescue_origin_group_by_task.pop(task.task_id, None)
            return cache
        except BaseException as exc:
            if cache is None:
                try:
                    await asyncio.to_thread(
                        self.bepa.store.mark_seed_failed,
                        claim,
                        reason=type(exc).__name__ + ": " + str(exc)[:240],
                        max_attempts=int(self.bepa.settings["seed"]["attempts_per_plan"]),
                    )
                except Exception:  # noqa: BLE001 - retain the original whole-group failure
                    pass
            if isinstance(observation, dict) and observation.get("status") == "running":
                observation.update(
                    status="failed",
                    finished_at=time.time(),
                    error={"type": type(exc).__name__, "message": str(exc)[:240]},
                )
            if isinstance(exc, asyncio.CancelledError):
                raise
            result["guided_row_training_disposition"] = "guided_rescue_failed"
            result["guided_rescue_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:240],
            }
            return None
        finally:
            await self._release_guided_attempt(attempt)

    async def abort_guided_rescue(self, attempt: dict[str, Any], reason: str) -> None:
        """Durably fail a claimed group-plan attempt before releasing its snapshot."""
        if not attempt or attempt.get("finished"):
            return
        try:
            await asyncio.to_thread(
                self.bepa.store.mark_seed_failed,
                attempt["claim"],
                reason=str(reason)[:300],
                max_attempts=int(self.bepa.settings["seed"]["attempts_per_plan"]),
            )
            observation = attempt.get("observation")
            if isinstance(observation, dict):
                observation.update(
                    status="failed",
                    finished_at=time.time(),
                    error={"type": "GroupGenerationError", "message": str(reason)[:240]},
                )
        finally:
            await self._release_guided_attempt(attempt)

    async def _release_guided_attempt(self, attempt: dict[str, Any]) -> None:
        if attempt.get("finished"):
            return
        attempt["finished"] = True
        try:
            attempt["references"].release(attempt["snapshot"])
        finally:
            self.bepa.mode = "idle"

    async def prepare_claim_group(
        self,
        result: dict[str, Any],
        behavior_version: int,
        learner_start_version: int,
        batch_id: str,
    ) -> bool:
        del behavior_version
        if not self.enabled or not result.get("replay_pending"):
            return True
        if self.central_gradient and (
            result.get("guided_rescue_replay") is not True
            or not isinstance(result.get("guided_cache_entry_id"), str)
        ):
            raise RuntimeError(
                "central-gradient BEPA replay must come from the same whole native-plus-guided group"
            )
        task = result["task"]
        task_key = self.bepa.task_keys[task.task_id]
        if task_key in self.execution.replay_tasks_in_claim:
            return False
        reservation = await asyncio.to_thread(
            self.bepa.admit_group,
            result,
            task,
            result["policy_snapshot"],
            batch_id,
            current_policy_version=learner_start_version,
        )
        if reservation is None:
            if self.central_gradient and result.get("guided_rescue_replay") is True:
                # A newer cache may supersede the exact same-group guidance.
                # Preserve a valid native group instead of substituting another cache.
                import statistics
                result["replay_pending"] = False
                result["guided_rescue_replay"] = False
                result["guided_row_training_disposition"] = "same_group_cache_unavailable"
                result["record"]["guided_row_training_disposition"] = "same_group_cache_unavailable"
                result.pop("training_rewards", None)
                self.execution.pending_cache_task_ids.discard(task.task_id)
                rewards = result["rewards"]
                return any(reward > 0 for reward in rewards) and statistics.pstdev(rewards) > 0
            return False
        expected_cache = result.get("guided_cache_entry_id")
        if self.central_gradient and reservation.cache_entry_id != expected_cache:
            await asyncio.to_thread(
                self.bepa.store.release_batch_reservations,
                batch_id,
                reason="central_guided_cache_identity_mismatch",
            )
            raise RuntimeError("central BEPA reservation selected a historical guided cache")
        self.execution.replay_tasks_in_claim.add(task_key)
        result["reservation"] = reservation
        result["record"]["replay_reservation"] = reservation.reservation_id
        result["record"]["status"] = "replay_reserved"
        return True

    async def abort_claim_reservations(self, batch_id: str, items: list[Any]) -> None:
        if self.enabled:
            await asyncio.to_thread(
                self.bepa.store.release_batch_reservations,
                batch_id,
                reason="claim_batch_incomplete",
            )
        for item in items:
            reservation = item.pop("reservation", None)
            if reservation is not None:
                item["record"].setdefault("released_replay_reservations", []).append(
                    reservation.reservation_id
                )
                item["record"].pop("replay_reservation", None)
                item["record"]["status"] = "replay_candidate"
        self.execution.replay_tasks_in_claim.clear()

    async def attach_ready_cache_to_deferred(
        self,
        ordinal: int,
        item: dict[str, Any],
        *,
        plan_job_id: str | None = None,
    ) -> bool:
        stream = self.stream
        if not self.enabled or stream is None:
            return False
        task = item["task"]
        current_policy_version = self.current_policy_version()
        cache_task = self.bepa.ready_task([task.task_id], current_policy_version)
        if cache_task is None:
            return False
        item["replay_pending"] = True
        if plan_job_id is not None:
            item["rescue_plan_job_id"] = plan_job_id
        item["rescue_cache_source_policy_version"] = getattr(
            cache_task, "source_policy_version", None,
        )
        item["record"].update(
            status="replay_candidate",
            replay_candidate=True,
            rescue_state="guided_cache_attached",
            rescue_cache_checked_at=time.time(),
            rescue_plan_job_id=plan_job_id,
        )
        self.execution.accepted_task_ids_by_version[
            int(item["policy_version"])
        ].add(task.task_id)
        self.execution.pending_cache_task_ids.add(task.task_id)
        await stream.resolve_deferred(ordinal, accepted=True)
        return True

    def prioritize_plan(self, task_id: str) -> None:
        if self.plan_prefetch is not None:
            self.plan_prefetch.prioritize(task_id)

    def _record_plan_prefetch(self) -> None:
        if self.plan_prefetch is not None:
            self.execution.record["bepa_plan_prefetch"] = asdict(self.plan_prefetch.snapshot())

    async def housekeeping(self) -> None:
        self._record_plan_prefetch()
        stream = self.stream
        if not self.enabled or stream is None:
            return
        deferred = stream.deferred_items()
        if not deferred:
            return
        now = time.time()
        hold_limit = float(self.bepa.settings["teacher"]["budget"]["job_timeout_seconds"])
        remaining: list[tuple[int, dict[str, Any]]] = []
        for ordinal, _policy, item in deferred:
            if await self.attach_ready_cache_to_deferred(ordinal, item):
                continue
            held_at = float(item.get("rescue_hold_started_at", now))
            if now - held_at >= hold_limit:
                await stream.resolve_deferred(
                    ordinal,
                    accepted=False,
                    reason="rescue_hold_timeout",
                )
                continue
            remaining.append((ordinal, item))
        if not remaining:
            return

        task_keys = sorted({
            self.bepa.task_keys[item["task"].task_id]
            for _ordinal, item in remaining
        })
        jobs = await asyncio.to_thread(self.bepa.store.list_jobs, task_allowlist=task_keys)
        jobs_by_key: dict[str, list[Any]] = {}
        for job in jobs:
            jobs_by_key.setdefault(job.task_key, []).append(job)
        terminal_states = {"failed", "exhausted", "rejected", "outcome_unknown", "cancelled"}
        active_states = {
            "queued", "retry_wait", "teacher_running", "source_submitted", "source_verified",
            "plan_ready", "seed_running", "cache_ready",
        }
        for ordinal, item in remaining:
            task = item["task"]
            task_key = self.bepa.task_keys[task.task_id]
            task_jobs = jobs_by_key.get(task_key, [])
            if (task_jobs and all(job.state in terminal_states for job in task_jobs)
                    and not any(job.state in active_states for job in task_jobs)):
                await stream.resolve_deferred(
                    ordinal,
                    accepted=False,
                    reason="rescue_hold_plan_exhausted",
                )

    async def discover_seed_plans(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        current_policy_version = self.current_policy_version()
        candidates: list[dict[str, Any]] = []
        for offset in range(0, len(self.selected_task_keys), 128):
            keys = self.selected_task_keys[offset:offset + 128]
            await asyncio.to_thread(
                self.bepa.store.refresh_seed_plans,
                keys,
                engine_epoch=self.bepa.epoch,
                current_policy_version=current_policy_version,
                limit=1000,
                whole_group=self.central_gradient,
            )
            plans = await asyncio.to_thread(
                self.bepa.store.available_plans,
                keys,
                engine_epoch=self.bepa.epoch,
                current_policy_version=current_policy_version,
                limit=1000,
                whole_group=self.central_gradient,
            )
            for plan in plans:
                task = self.task_by_key.get(plan.task_key)
                if task is not None:
                    candidates.append({"task_id": task.task_id, "task": task, "plan": plan})
        held_order: dict[str, int] = {}
        if self.stream is not None:
            for ordinal, _policy, item in self.stream.deferred_items():
                task_id = item["task"].task_id
                held_order[task_id] = min(ordinal, held_order.get(task_id, ordinal))
        candidates.sort(key=lambda candidate: (
            0 if candidate["task_id"] in held_order else 1,
            held_order.get(candidate["task_id"], 0),
            candidate["plan"].created_at,
            candidate["plan"].plan_id,
        ))
        return candidates

    async def attempt_seed_candidate(self, candidate: dict[str, Any]) -> SeedAttemptResult:
        if self.central_gradient:
            newly_queued = self._queue_guided_rescue(candidate)
            return SeedAttemptResult.SCHEDULED if newly_queued else SeedAttemptResult.SKIPPED
        version = self.serve_policy_version()
        if not isinstance(version, int) or version not in self.policies:
            return SeedAttemptResult.SKIPPED
        snapshot = PolicySnapshot.from_policy(self.policies[version])
        if snapshot.version != version:
            raise RuntimeError("guided seed policy snapshot differs from the ACKed serve version")
        status, cache = await self.seed_service.seed_plan_once(
            candidate["task"],
            candidate["plan"],
            snapshot,
            run_id=self.run_id,
            task_allowlist=self.selected_task_keys,
            generation_gate=self.generation_gate,
            observability=self.execution.record.get("bepa_seed_observability"),
            phase="continuous",
        )
        if status is SeedAttemptResult.READY and cache is not None and self.stream is not None:
            for ordinal, _policy, item in self.stream.deferred_items():
                if item["task"].task_id == candidate["task"].task_id:
                    await self.attach_ready_cache_to_deferred(
                        ordinal,
                        item,
                        plan_job_id=candidate["plan"].job_id,
                    )
                    break
        return status

    async def report_scheduler_error(self, exc: BaseException) -> None:
        self.execution.record["failure_stage"] = "seed_scheduler"
        observability = self.execution.record.get("bepa_seed_observability")
        if isinstance(observability, dict):
            observability["scheduler_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:300],
            }
        if self.stream is not None:
            await self.stream.fail(exc)

    def oldest_rescue_hold_seconds(self) -> float:
        if self.stream is None:
            return 0.0
        starts = [
            float(item.get("rescue_hold_started_at", time.time()))
            for _ordinal, _policy, item in self.stream.deferred_items()
        ]
        return max(0.0, time.time() - min(starts)) if starts else 0.0

    def start_scheduler(self) -> Any | None:
        if not self.enabled:
            return None
        if self.central_gradient:
            from ...bepa.plan_prefetch import TeacherPlanPrefetcher
            self.plan_prefetch = TeacherPlanPrefetcher(self.bepa, self.selected_tasks)
            self.plan_prefetch.start()
            self._record_plan_prefetch()
        scheduler = RunSeedScheduler(
            self.run_id,
            discover=self.discover_seed_plans,
            attempt=self.attempt_seed_candidate,
            on_error=self.report_scheduler_error,
            housekeeping=self.housekeeping,
            oldest_hold_seconds=self.oldest_rescue_hold_seconds,
        )
        self.execution.seed_scheduler = scheduler
        self.on_scheduler_change(scheduler)
        scheduler.start()
        return scheduler

    async def close_scheduler(self, *, drain: bool = False) -> dict[str, Any] | None:
        if self.plan_prefetch is not None:
            await self.plan_prefetch.stop()
            self._record_plan_prefetch()
        scheduler = self.execution.seed_scheduler
        if scheduler is None:
            return None
        if drain:
            await scheduler.close(drain=True)
        else:
            await scheduler.close()
        snapshot = scheduler.snapshot()
        self.execution.seed_scheduler = None
        self.on_scheduler_change(None)
        return snapshot
