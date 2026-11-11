"""Typed lifecycle and mutable execution state for one Learner run."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...rollout.groups import PolicySnapshot

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "filtered_limit", "paused"})


class RunState(dict[str, Any]):
    """Persisted run record with guarded stage and terminal transitions.

    It remains a ``dict`` subclass so the existing HTTP and JSON state ABI stays
    byte-for-byte shaped like the historical run record.
    """

    def __init__(self, run_id: str, values: dict[str, Any] | None = None) -> None:
        super().__init__(values or {})
        stored_id = self.get("run_id")
        if stored_id is not None and stored_id != run_id:
            raise ValueError("RunState run_id differs from its persisted record")
        self["run_id"] = run_id

    def set_stage(self, stage: str) -> None:
        if not stage:
            raise ValueError("run stage cannot be empty")
        if self.get("status") in _TERMINAL_STATUSES:
            raise RuntimeError("cannot change the stage of a terminal run")
        self["failure_stage"] = stage

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> RunState:
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("persisted run record must have a non-empty run_id")
        if isinstance(record, cls):
            return record
        return cls(run_id, record)

    def fail(self, exc: BaseException, *, stage: str | None = None, limit: int = 500) -> None:
        if self.get("status") in _TERMINAL_STATUSES and self.get("status") != "failed":
            raise RuntimeError("cannot fail a terminal run with a different outcome")
        if stage is not None:
            self.set_stage(stage)
        self["status"] = "failed"
        self["error"] = {"type": type(exc).__name__, "message": str(exc)[:limit]}

    def finish(
        self,
        *,
        status: str,
        finished_at: float,
        elapsed_seconds: float,
        committed_policy_version: int,
        summary: dict[str, Any],
    ) -> None:
        if status not in _TERMINAL_STATUSES:
            raise ValueError(f"unsupported terminal run status: {status}")
        current = self.get("status")
        if current in _TERMINAL_STATUSES and current != status:
            raise RuntimeError(f"run terminal status changed from {current} to {status}")
        if current != "running" and current != status:
            raise RuntimeError(f"run cannot finish from status {current!r}")
        self["status"] = status
        self["finished_at"] = finished_at
        self["elapsed_seconds"] = elapsed_seconds
        self["committed_policy_version"] = committed_policy_version
        self["summary"] = summary


@dataclass
class PolicyReferenceLedger:
    """Mirror engine policy refs into app-shared delivery-cache retention refs."""

    engine: Any
    shared_refs: dict[int, int]
    held_versions: dict[int, int] = field(default_factory=dict)
    held_snapshots: dict[int, PolicySnapshot] = field(default_factory=dict)
    demand_remaining: dict[int, int] = field(default_factory=dict)
    demand_refs: dict[int, PolicySnapshot] = field(default_factory=dict)

    def acquire(self, snapshot: PolicySnapshot) -> None:
        version = snapshot.version
        self.engine.acquire_policy(version)
        self.held_versions[version] = self.held_versions.get(version, 0) + 1
        self.held_snapshots[version] = snapshot
        self.shared_refs[version] = self.shared_refs.get(version, 0) + 1

    def release(self, snapshot: PolicySnapshot) -> None:
        version = snapshot.version
        held = self.held_versions.get(version, 0)
        shared = self.shared_refs.get(version, 0)
        if held < 1 or shared < 1:
            raise RuntimeError(f"policy reference underflow for version {version}")
        self.engine.release_policy(version)
        if held == 1:
            self.held_versions.pop(version)
            self.held_snapshots.pop(version, None)
        else:
            self.held_versions[version] = held - 1
        if shared == 1:
            self.shared_refs.pop(version)
        else:
            self.shared_refs[version] = shared - 1

    def retain_demand(self, snapshot: PolicySnapshot) -> None:
        version = snapshot.version
        if self.demand_remaining.get(version, 0) > 0 and version not in self.demand_refs:
            self.acquire(snapshot)
            self.demand_refs[version] = snapshot

    def satisfy_demand(self, version: int, groups: int) -> None:
        remaining = self.demand_remaining.get(version, 0) - groups
        self.demand_remaining[version] = remaining
        if remaining < 0:
            raise RuntimeError("claimed groups exceed the finite policy-version demand")
        if remaining == 0:
            snapshot = self.demand_refs.pop(version, None)
            if snapshot is not None:
                self.release(snapshot)

    def release_all_demand(self) -> None:
        for version, snapshot in list(self.demand_refs.items()):
            self.demand_refs.pop(version)
            self.release(snapshot)

    def release_all_held(self) -> None:
        for version, count in list(self.held_versions.items()):
            snapshot = self.held_snapshots[version]
            for _ in range(count):
                self.release(snapshot)


@dataclass
class CheckpointSamplingFence:
    """One published policy awaits durable checkpoint and replay finalization."""

    policy: PolicySnapshot
    previous_round_id: str
    replay_task_ids: frozenset[str]
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    error: str | None = None

    def check(self, policy: PolicySnapshot) -> None:
        if policy != self.policy:
            raise RuntimeError("checkpoint sampling fence policy identity differs")
        if self.error is not None:
            raise RuntimeError(f"checkpoint sampling failed: {self.error}")

    async def wait(self, policy: PolicySnapshot) -> None:
        self.check(policy)
        await self.settled.wait()
        self.check(policy)


@dataclass
class RunExecutionState:
    """Non-persisted cursors, fences, and owned tasks for one run execution."""

    record: RunState
    request: Any
    started_monotonic: float
    selected_tasks: list[Any]
    target_steps: int
    remaining_steps: int
    completed_at_start: int
    task_cursor_base: int
    needed_groups: int
    references: PolicyReferenceLedger
    task_index: int = 0
    group_index: int = 0
    stream: Any | None = None
    seed_scheduler: Any | None = None
    pending_policy_handoff: Any | None = None
    initial_snapshot: PolicySnapshot | None = None
    worker_sync_acks: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_task_counts: dict[str, int] = field(default_factory=dict)
    active_task_ids: set[str] = field(default_factory=set)
    accepted_task_ids_by_version: dict[int, set[str]] = field(default_factory=dict)
    pending_cache_task_ids: set[str] = field(default_factory=set)
    historical_seen_task_ids: set[str] = field(default_factory=set)
    rescue_enqueued_task_ids: set[str] = field(default_factory=set)
    replay_tasks_in_claim: set[str] = field(default_factory=set)
    cohorts_by_batch_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_worker_owners: dict[str, str] = field(default_factory=dict)
    rescue_ready_task_ids: list[str] = field(default_factory=list)
    checkpoint_sampling_fences: dict[int, CheckpointSamplingFence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep task affinity in the serialized run record as well as live state."""
        owners = self.record.setdefault("task_worker_owners", self.task_worker_owners)
        if not isinstance(owners, dict) or any(
            not isinstance(task_id, str) or not task_id
            or not isinstance(worker_id, str) or not worker_id
            for task_id, worker_id in owners.items()
        ):
            raise ValueError("persisted task_worker_owners must map task IDs to Worker IDs")
        self.task_worker_owners = owners
        rescue_ids = self.record.setdefault("rescue_ready_task_ids", self.rescue_ready_task_ids)
        if (not isinstance(rescue_ids, list)
                or any(not isinstance(task_id, str) or not task_id for task_id in rescue_ids)
                or len(set(rescue_ids)) != len(rescue_ids)):
            raise ValueError("persisted rescue_ready_task_ids must be a unique task-ID list")
        self.rescue_ready_task_ids = rescue_ids

    def enqueue_rescue_ready_task(self, task_id: str) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a non-empty string")
        if task_id not in self.rescue_ready_task_ids:
            self.rescue_ready_task_ids.append(task_id)

    def pop_rescue_ready_task(self, excluded_task_ids: set[str]) -> str | None:
        """Pop the oldest currently eligible rescue task while preserving blocked order."""
        if not isinstance(excluded_task_ids, set):
            raise TypeError("excluded_task_ids must be a set")
        for index, task_id in enumerate(self.rescue_ready_task_ids):
            if task_id not in excluded_task_ids:
                return self.rescue_ready_task_ids.pop(index)
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
