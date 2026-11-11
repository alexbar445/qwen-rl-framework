"""Bounded next-round rollout while the central full checkpoint is saving."""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from ...rollout.groups import PolicySnapshot, RolloutBatch
from .state import CheckpointSamplingFence


class CheckpointRolloutOverlap:
    """Publish/ACK inference LoRA without committing the optimizer transaction.

    The coordinator still awaits the durable update before claiming the next
    round. A failed writer quarantines speculative groups rather than treating
    their publication as checkpoint or BEPA commit evidence.
    """

    def __init__(
        self, *, execution: Any, engine: Any, runtime: Any,
        publish_current: Callable[..., Awaitable[dict[str, Any]]],
        sync_workers: Callable[..., Awaitable[dict[str, Any]]],
        stop_control: Any | None = None,
        pause_every_updates: int | None = None,
        successful_update_count: Callable[[], int | None] | None = None,
        evaluation_pending: Callable[[], bool] | None = None,
    ) -> None:
        self.execution = execution
        self.engine = engine
        self.runtime = runtime
        self.publish_current = publish_current
        self.sync_workers = sync_workers
        self.stop_control = stop_control
        self.pause_every_updates = pause_every_updates
        self.successful_update_count = successful_update_count
        self.evaluation_pending = evaluation_pending
        self.discard_speculative = False
        self.invalid_after_version: int | None = None
        if not hasattr(execution, "checkpoint_sampling_fences"):
            execution.checkpoint_sampling_fences = {}

    async def publish(self, publication: dict[str, Any], batch: RolloutBatch) -> dict[str, Any]:
        execution = self.execution
        state = execution.record
        metadata = self.engine.recovery_metadata
        version = int(publication["version"])
        if metadata.get("policy_version") != version or metadata.get("can_publish") is not True:
            raise RuntimeError("checkpoint overlap cannot publish an unready central policy")
        record = {
            "policy_version": version,
            "durable_policy_version_at_publication": metadata.get("durable_policy_version"),
            "state": "publishing",
            "publication_ready_at": time.time(),
            "worker_acked": False,
            "prefetched_round_id": None,
        }
        state.setdefault("checkpoint_overlaps", []).append(record)
        state["checkpoint_overlap"] = record
        if (self.stop_control is not None
                and self.stop_control.should_skip_overlap(version)):
            self.discard_speculative = True
            record.update(state="stop_requested", worker_acked=False)
            return record
        policy = await self.publish_current(publication, pending_checkpoint=True)
        snapshot = PolicySnapshot.from_policy(policy)
        execution.references.retain_demand(snapshot)
        started = time.perf_counter()
        acknowledgements = await self.sync_workers(policy)
        state["metrics"]["learner"]["worker_sync_seconds"] += time.perf_counter() - started
        execution.worker_sync_acks = acknowledgements
        record["worker_acknowledged_at"] = time.time()
        if (self.stop_control is not None
                and self.stop_control.should_skip_overlap(version)):
            self.discard_speculative = True
            record.update(state="stop_requested", worker_acked=True)
            return record
        # Recheck after network I/O: a writer failure must not open admission.
        metadata = self.engine.recovery_metadata
        if metadata.get("can_publish") is not True or metadata.get("policy_version") != version:
            raise RuntimeError("checkpoint failed while acknowledging speculative LoRA")
        for group in batch.groups:
            group["record"]["policy_acks"] = acknowledgements
        stream = execution.stream
        if stream is None:
            raise RuntimeError("rollout stream ended before checkpoint overlap")
        if version in execution.checkpoint_sampling_fences:
            raise RuntimeError("checkpoint sampling fence already exists for this version")
        execution.checkpoint_sampling_fences[version] = CheckpointSamplingFence(
            policy=snapshot,
            previous_round_id=batch.groups[0]["bepa_batch_id"],
            replay_task_ids=frozenset(
                group["task"].task_id for group in batch.groups if group.get("replay_pending")
            ),
        )
        await stream.publish(snapshot)
        state["serve_policy_version"] = version
        self.runtime.serve_policy_version = version
        record.update(
            state="checkpoint_saving", worker_acked=True,
            adapter_sha256=snapshot.adapter_sha256,
        )
        current_count = (
            self.successful_update_count()
            if self.successful_update_count is not None else None
        )
        next_update_count = (
            current_count + 1 if type(current_count) is int
            else int(state["completed_updates"]) + 1
        )
        pause_after_commit = (
            self.pause_every_updates is not None
            and (next_update_count % self.pause_every_updates == 0
                 or (self.evaluation_pending is not None and self.evaluation_pending()))
        )
        record["pause_after_commit"] = pause_after_commit
        remaining_after_commit = int(state["requested_steps"]) - int(state["completed_updates"]) - 1
        stop_targeted = (
            self.stop_control is not None
            and self.stop_control.should_skip_overlap(version)
        )
        if remaining_after_commit > 0 and not pause_after_commit and not stop_targeted:
            round_id = uuid4().hex
            admitted = await stream.prefetch_round(batch.groups[0]["bepa_batch_id"], snapshot, round_id)
            if admitted:
                record["prefetched_round_id"] = round_id
                record["prefetch_started_at"] = time.time()
        return record

    def failed(self, exc: BaseException) -> None:
        metadata = self.engine.recovery_metadata
        live = metadata.get("policy_version")
        for fence in self.execution.checkpoint_sampling_fences.values():
            if not fence.settled.is_set():
                fence.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                fence.settled.set()
        durable = metadata.get("durable_policy_version")
        self.discard_speculative = type(live) is int and type(durable) is int and live > durable
        self.invalid_after_version = durable if self.discard_speculative else None
        record = self.execution.record.get("checkpoint_overlap")
        if (isinstance(record, dict) and record.get("state") != "durable"
                and record.get("policy_version") == live):
            record.update(
                state="failed", discard_speculative=self.discard_speculative,
                error={"type": type(exc).__name__, "message": str(exc)[:300]},
            )

    def committed(self, result: dict[str, Any]) -> None:
        record = result.get("checkpoint_overlap")
        if isinstance(record, dict):
            metadata = self.engine.recovery_metadata
            if metadata.get("durable_policy_version") != record["policy_version"]:
                raise RuntimeError("checkpoint overlap cannot finalize before durability")
            fence = self.execution.checkpoint_sampling_fences.get(record["policy_version"])
            if fence is not None and fence.error is not None:
                raise RuntimeError("failed checkpoint sampling fence cannot be committed")
            record.update(state="durable", checkpoint_durable_observed_at=time.time())
            if fence is not None:
                fence.settled.set()

    def quarantine(self, group: dict[str, Any]) -> None:
        version = group.get("policy_version")
        if (self.discard_speculative and type(version) is int
                and self.invalid_after_version is not None and version > self.invalid_after_version):
            group.update(
                status="quarantined_uncommitted_policy",
                quarantine_reason="asynchronous_checkpoint_not_durable",
                optimizer_consumed=False,
            )
