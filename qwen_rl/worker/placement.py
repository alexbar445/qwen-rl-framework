"""Capacity-bounded, whole-episode placement across persistent Workers."""
from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, Any

from .errors import WorkerUnavailableError

_KV_ADMISSION_HIGH_WATER = 0.90
_KV_METRIC_MAX_AGE_SECONDS = 5.0


def complete_group_episode_capacity(config: Mapping[str, Any]) -> int:
    """Keep every admitted group runnable while its finished members await scoring.

    These are logical episode leases, not simultaneous GPU decode slots. Worker
    generation admission independently enforces worker_inference.max_num_seqs.
    BEPA can add one guided member to each native group.
    """
    group_size = int(config["rollout"]["group_size"])
    guided_member = int(bool(config.get("bepa", {}).get("enabled", False)))
    groups = int(config["learner"].get("gradient_tasks_per_worker", 1))
    return max(int(config["worker_inference"]["max_num_seqs"]),
               groups * (group_size + guided_member))


def assign_task_worker_owner(
    task_id: str,
    explicit_worker_ids: Sequence[str],
    task_worker_owners: dict[str, str],
) -> str:
    """Persist one task's affinity using only explicitly paired Workers.

    Existing owners are immutable for the life of a run. New tasks use the
    least assigned explicit Worker, with the configured order as a stable tie
    break. An invalid persisted owner fails closed instead of relocating work.
    """
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a non-empty string")
    if isinstance(explicit_worker_ids, (str, bytes)) or not explicit_worker_ids:
        raise ValueError("explicit_worker_ids must contain at least one Worker ID")
    workers = tuple(explicit_worker_ids)
    if (any(not isinstance(worker_id, str) or not worker_id for worker_id in workers)
            or len(set(workers)) != len(workers)):
        raise ValueError("explicit_worker_ids must contain unique non-empty Worker IDs")
    if not isinstance(task_worker_owners, dict):
        raise TypeError("task_worker_owners must be a mutable mapping")
    owner = task_worker_owners.get(task_id)
    if owner is not None:
        if owner not in workers:
            raise RuntimeError(
                f"task {task_id} is pinned to Worker {owner!r}, outside the explicit executor pair set"
            )
        return owner
    counts = {worker_id: 0 for worker_id in workers}
    for assigned in task_worker_owners.values():
        if assigned not in counts:
            raise RuntimeError(
                f"persisted task Worker owner {assigned!r} is outside the explicit executor pair set"
            )
        counts[assigned] += 1
    selected = min(workers, key=lambda worker_id: counts[worker_id])
    task_worker_owners[task_id] = selected
    return selected


class EpisodeWorkerPool:
    """Lease the least-occupied available Worker for one complete episode.

    Local leases bound episode placement. Optional cached health probes exclude
    unavailable Workers and Workers reporting high KV pressure from new leases;
    they never revoke a running episode or move it between Workers.

    ``active_counts`` returns a copy. Pool state is owned by the event loop that
    acquires leases; read the property on that same loop.
    """

    def __init__(self, worker_ids: Sequence[str], capacity: int | Mapping[str, int], *,
                 resource_probe: Callable[[str], Awaitable[Mapping[str, Any]]] | None = None,
                 member_epochs: Mapping[str, str] | None = None,
                 group_capacity: int = 1,
                 resource_probe_interval: float = 0.5,
                 initially_paused: tuple[str, ...] = ()) -> None:
        if isinstance(worker_ids, (str, bytes)) or not worker_ids:
            raise ValueError("worker_ids must contain at least one Worker ID")
        ids = tuple(worker_ids)
        if any(not isinstance(worker_id, str) or not worker_id for worker_id in ids):
            raise ValueError("Worker IDs must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("Worker IDs must be unique")

        if isinstance(capacity, Mapping):
            if set(capacity) != set(ids):
                raise ValueError("capacity keys must match worker_ids")
            capacities = dict(capacity)
        else:
            capacities = {worker_id: capacity for worker_id in ids}
        for worker_id, limit in capacities.items():
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"capacity for {worker_id} must be a positive integer")

        self.episode_queue = None
        self._worker_ids = ids
        self._capacities = capacities
        epochs = dict(member_epochs or {})
        if set(epochs) - set(ids) or any(
                not isinstance(epoch, str) or not epoch for epoch in epochs.values()):
            raise ValueError("member_epochs must contain non-empty epochs for configured Worker IDs")
        self._member_epochs = epochs
        self._active = {worker_id: 0 for worker_id in ids}
        self._active_groups: dict[str, dict[str, str | None]] = {}
        self._group_episodes: dict[str, int] = {}
        if type(group_capacity) is not int or not 1 <= group_capacity <= 8:
            raise ValueError("group_capacity must be an integer in [1, 8]")
        self._group_capacity = group_capacity
        self._active_group_by_worker: dict[str, set[str]] = {}
        self._paused_workers: set[str] = set(initially_paused)
        self._quarantined_workers: set[str] = set()
        self._next_tie_index = 0
        self._normal_waiters = 0
        self._condition = asyncio.Condition()
        if (isinstance(resource_probe_interval, bool)
                or not isinstance(resource_probe_interval, (int, float))
                or not math.isfinite(resource_probe_interval) or resource_probe_interval <= 0):
            raise ValueError("resource_probe_interval must be finite and positive")
        self._resource_probe = resource_probe
        self._probe_interval = resource_probe_interval
        self._probe_lock = asyncio.Lock()
        self._last_probe = float('-inf')
        self._resource_blocked: set[str] = set()
        self._health_unavailable_workers: set[str] = set()
        self._sync_unavailable_workers: set[str] = set()
        self._rebalance_idle_workers: set[str] = set()
        self._rebalance_ready_workers: set[str] = set()
        self._rebalance_policies = {}
        self._resource_status = {worker_id: 'unknown' for worker_id in ids}

    @property
    def active_counts(self) -> dict[str, int]:
        """A read-only-by-copy snapshot of current whole-episode leases."""
        return dict(self._active)

    @property
    def resource_admission(self) -> dict[str, str]:
        """Observed admission state; unknown KV data is never treated as zero usage."""
        return {key: 'quarantined' if key in self.quarantined_workers else
                'phase_paused' if key in self._paused_workers else value
                for key, value in self._resource_status.items()}

    @property
    def quarantined_workers(self) -> frozenset[str]:
        workers = set(self._quarantined_workers)
        if self.episode_queue is not None:
            with self.episode_queue.lock:
                workers.update(self.episode_queue.quarantined)
        return frozenset(workers)

    async def quarantine_worker(self, worker_id: str) -> None:
        """Fence new work without releasing uncertain episode or compute leases."""
        if worker_id not in self._active:
            raise ValueError("cannot quarantine an unknown Worker")
        async with self._condition:
            self._quarantined_workers.add(worker_id)
            self._paused_workers.add(worker_id)
            if self.episode_queue is not None:
                self.episode_queue.quarantine_worker(worker_id)
            self._condition.notify_all()

    async def mark_worker_unavailable(self, worker_id: str, *, source: str) -> None:
        """Wake pinned admissions after a typed health or policy-sync failure."""
        if worker_id not in self._active or source not in {"health", "policy_sync"}:
            raise ValueError("unavailable admission evidence must name a Worker and source")
        async with self._condition:
            unavailable = (
                self._health_unavailable_workers if source == "health"
                else self._sync_unavailable_workers
            )
            unavailable.add(worker_id)
            self._condition.notify_all()

    async def clear_worker_unavailable(self, worker_id: str, *, source: str) -> None:
        """Clear recovered admission evidence without lifting uncertain quarantines."""
        if worker_id not in self._active or source not in {"health", "policy_sync"}:
            raise ValueError("unavailable admission evidence must name a Worker and source")
        async with self._condition:
            unavailable = (
                self._health_unavailable_workers if source == "health"
                else self._sync_unavailable_workers
            )
            unavailable.discard(worker_id)
            self._condition.notify_all()

    def _require_worker_available(self, worker_id: str) -> None:
        if worker_id in self._health_unavailable_workers | self._sync_unavailable_workers:
            raise WorkerUnavailableError(
                f"Worker {worker_id} is unavailable before episode admission",
                worker_id=worker_id,
                quarantinable=False,
            )

    def _require_live_target(self, worker_ids) -> None:
        if set(worker_ids) <= self.quarantined_workers:
            raise WorkerUnavailableError(
                "Worker is quarantined after an uncertain request",
                worker_id=next(iter(worker_ids)),
            )

    @property
    def member_epochs(self) -> dict[str, str]:
        """Pinned membership epochs; pool refreshes may block leases but never replace them."""
        return dict(self._member_epochs)

    @property
    def active_group_leases(self) -> dict[str, dict[str, Any]]:
        """Current complete-group reservations by Worker."""
        return {worker: {"group_ids": sorted(groups), "group_capacity": self._group_capacity,
                         "owners": {group: self._active_groups[group]["owner_id"] for group in groups}}
                for worker, groups in self._active_group_by_worker.items()}

    async def _refresh_resources(self) -> None:
        if self._resource_probe is None:
            return
        async with self._probe_lock:
            if time.monotonic() - self._last_probe < self._probe_interval:
                return
            replies = await asyncio.gather(
                *(self._resource_probe(worker_id) for worker_id in self._worker_ids),
                return_exceptions=True,
            )
            blocked = set()
            health_unavailable = set()
            idle, ready = set(), set()
            policies = {}
            statuses = {}
            for worker_id, reply in zip(self._worker_ids, replies):
                if isinstance(reply, BaseException) or not isinstance(reply, Mapping):
                    statuses[worker_id] = 'health_unavailable'
                    blocked.add(worker_id)
                    if isinstance(reply, WorkerUnavailableError):
                        health_unavailable.add(worker_id)
                    continue
                # A reachable health document clears a previous transport outage,
                # even while the Worker is still syncing its policy.
                health_unavailable.discard(worker_id)
                if reply.get('admission_open') is False:
                    statuses[worker_id] = 'phase_paused'
                    blocked.add(worker_id)
                    continue
                if reply.get('ready') is False:
                    statuses[worker_id] = 'worker_not_ready'
                    blocked.add(worker_id)
                    continue
                if (reply.get('ready') is True and reply.get('admission_open') is True
                        and reply.get('policy_ready') is True):
                    ready.add(worker_id)
                    versions = reply.get('resident_versions', ())
                    if not isinstance(versions, (list, tuple)):
                        versions = ()
                    policies[worker_id] = {(row.get('policy_version'), row.get('adapter_hash'))
                                           for row in versions if isinstance(row, Mapping)
                                           and type(row.get('policy_version')) is int
                                           and isinstance(row.get('adapter_hash'), str)}
                    fields = ('active_generations', 'pending_generations', 'queued_generations', 'episode_refs')
                    if all(type(reply.get(field)) is int and reply[field] == 0 for field in fields):
                        idle.add(worker_id)
                kv = reply.get('kv_cache') or {}
                if not isinstance(kv, Mapping):
                    kv = {}
                usage, age = kv.get('usage_fraction'), kv.get('age_seconds')
                valid = (not isinstance(usage, bool) and isinstance(usage, (int, float))
                         and math.isfinite(usage) and 0 <= usage <= 1
                         and not isinstance(age, bool) and isinstance(age, (int, float))
                         and math.isfinite(age) and 0 <= age <= _KV_METRIC_MAX_AGE_SECONDS)
                valid = valid and kv.get('available') is True and kv.get('stale') is False
                if not valid:
                    statuses[worker_id] = 'kv_unknown_or_stale'
                elif usage >= _KV_ADMISSION_HIGH_WATER:
                    statuses[worker_id] = 'kv_pressure'
                    blocked.add(worker_id)
                else:
                    statuses[worker_id] = 'available'
            async with self._condition:
                self._resource_blocked = blocked
                self._health_unavailable_workers = health_unavailable
                self._rebalance_idle_workers, self._rebalance_ready_workers = idle, ready
                self._rebalance_policies = policies
                self._resource_status = statuses
                self._last_probe = time.monotonic()
                self._condition.notify_all()

    @staticmethod
    def _validate_worker_ids(worker_ids, configured: tuple[str, ...]) -> tuple[str, ...]:
        if worker_ids is None:
            return configured
        if isinstance(worker_ids, (str, bytes)):
            raise ValueError("Worker targets must be IDs, not a string")
        targets = tuple(worker_ids)
        if (not targets or any(not isinstance(worker_id, str) or not worker_id for worker_id in targets)
                or len(set(targets)) != len(targets) or set(targets) - set(configured)):
            raise ValueError("Worker targets must name unique configured Workers")
        return targets

    @asynccontextmanager
    async def lease(
        self,
        *,
        low_priority: bool = False,
        preferred_worker_id: str | None = None,
        group_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Lease one whole episode, optionally pinned inside an admitted group.

        Existing unpinned leases retain balanced placement. A group member stays
        on its reserved Worker through phase pauses and resource-probe changes;
        the group reservation itself is the admission boundary.
        """
        registered = False
        worker_id = None
        external_key = object()
        if preferred_worker_id is not None and preferred_worker_id not in self._active:
            raise ValueError("preferred_worker_id is not a configured Worker")
        if group_id is not None and (not isinstance(group_id, str) or not group_id):
            raise ValueError("group_id must be a non-empty string")
        if group_id is not None and preferred_worker_id is None:
            raise ValueError("group member leases require preferred_worker_id")
        try:
            if not low_priority:
                async with self._condition:
                    self._normal_waiters += 1
                    registered = True
            while True:
                await self._refresh_resources()
                if preferred_worker_id is not None:
                    self._require_worker_available(preferred_worker_id)
                else:
                    unavailable = self._health_unavailable_workers | self._sync_unavailable_workers
                    if set(self._worker_ids) <= unavailable:
                        self._require_worker_available(self._worker_ids[0])
                self._require_live_target((preferred_worker_id,) if preferred_worker_id else self._worker_ids)
                async with self._condition:
                    if group_id is not None:
                        if group_id not in self._active_group_by_worker.get(preferred_worker_id, set()):
                            raise ValueError("group episode lease has no matching active Worker reservation")
                        # This episode belongs to an already admitted complete
                        # group; a later phase pause or KV probe cannot strand
                        # the other members halfway through that legal group.
                        available = (
                            [preferred_worker_id]
                            if self._active[preferred_worker_id] < self._capacities[preferred_worker_id]
                            else []
                        )
                    elif preferred_worker_id is not None:
                        available = [preferred_worker_id] if (
                            preferred_worker_id not in self._resource_blocked
                            and preferred_worker_id not in self._paused_workers
                            and preferred_worker_id not in self._active_group_by_worker
                            and self._active[preferred_worker_id] < self._capacities[preferred_worker_id]
                        ) else []
                    else:
                        available = [key for key in self._worker_ids
                                     if key not in self._resource_blocked and key not in self._paused_workers
                                     and key not in self._active_group_by_worker
                                     and self._active[key] < self._capacities[key]]
                    if self.episode_queue is not None:
                        with self.episode_queue.lock:
                            reserved = {target for _, target in self.episode_queue.transfers.values()}
                            available = [w for w in available if w not in self.episode_queue.compute
                                         and w not in self.episode_queue.quarantined and w not in reserved]
                    free = sum(self._capacities[key] - self._active[key] for key in available)
                    reserve = min(2, sum(self._capacities.values()))
                    admitted = available and (not low_priority or
                                              (self._normal_waiters == 0 and free >= reserve))
                    if admitted:
                        worker_id = (
                            preferred_worker_id if group_id is not None
                            else self._select_worker(available)
                        )
                        if (self.episode_queue is not None
                                and not self.episode_queue.begin_external_episode(external_key, worker_id)):
                            worker_id = None
                        if worker_id is not None:
                            self._active[worker_id] += 1
                            if group_id is not None:
                                self._group_episodes[group_id] += 1
                            if registered:
                                self._normal_waiters -= 1
                                registered = False
                            break
                    try:
                        await asyncio.wait_for(self._condition.wait(), self._probe_interval)
                    except TimeoutError:
                        pass
            yield worker_id
        finally:
            async with self._condition:
                if registered:
                    self._normal_waiters -= 1
                if worker_id is not None:
                    if self.episode_queue is not None:
                        self.episode_queue.finish_episode(external_key)
                    self._active[worker_id] -= 1
                    if group_id is not None:
                        self._group_episodes[group_id] -= 1
                self._condition.notify_all()

    @asynccontextmanager
    async def group_lease(
        self,
        group_id: str,
        *,
        worker_ids: Sequence[str] | None = None,
        preferred_worker_id: str | None = None,
        owner_by_worker: Mapping[str, str] | None = None,
        low_priority: bool = False,
    ) -> AsyncIterator[str]:
        """Reserve one of the Worker's bounded complete-group positions.

        The reservation does not consume episode slots itself. Each candidate
        takes a whole-episode lease pinned to this Worker, including its wait
        for group scoring. Placement capacity must cover every member of all
        admitted groups; GPU generation concurrency is enforced separately.
        """
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")
        targets = self._validate_worker_ids(worker_ids, self._worker_ids)
        if preferred_worker_id is not None:
            if preferred_worker_id not in self._active:
                raise ValueError("preferred_worker_id is not a configured Worker")
            if preferred_worker_id not in targets:
                raise ValueError("preferred_worker_id is outside the eligible Worker set")
        owners = dict(owner_by_worker or {})
        if (set(owners) - set(targets)
                or any(not isinstance(owner, str) or not owner for owner in owners.values())):
            raise ValueError("owner_by_worker must contain valid owners for eligible Workers")

        registered = False
        selected: str | None = None
        try:
            if not low_priority:
                async with self._condition:
                    self._normal_waiters += 1
                    registered = True
            while selected is None:
                await self._refresh_resources()
                if preferred_worker_id is not None:
                    self._require_worker_available(preferred_worker_id)
                else:
                    unavailable = self._health_unavailable_workers | self._sync_unavailable_workers
                    if set(targets) <= unavailable:
                        self._require_worker_available(targets[0])
                self._require_live_target((preferred_worker_id,) if preferred_worker_id else targets)
                async with self._condition:
                    if group_id in self._active_groups:
                        raise ValueError("group_id already has an active Worker reservation")
                    eligible = [worker_id for worker_id in targets
                                if worker_id not in self._resource_blocked
                                and worker_id not in self._paused_workers
                                and len(self._active_group_by_worker.get(worker_id, ())) < self._group_capacity
                                and (self._active[worker_id] == 0 or worker_id in self._active_group_by_worker)
                                and self._capacities[worker_id] >= 1]
                    if preferred_worker_id is not None:
                        eligible = [worker_id for worker_id in eligible
                                    if worker_id == preferred_worker_id]
                    reserve = min(2, sum(self._capacities.values()))
                    admitted = bool(eligible) and (
                        not low_priority or
                        (self._normal_waiters == 0
                         and sum(self._capacities[key] - self._active[key] for key in eligible) >= reserve)
                    )
                    if admitted:
                        selected = self._select_worker(eligible)
                        self._active_groups[group_id] = {
                            "worker_id": selected,
                            "owner_id": owners.get(selected),
                        }
                        self._group_episodes[group_id] = 0
                        self._active_group_by_worker.setdefault(selected, set()).add(group_id)
                        if registered:
                            self._normal_waiters -= 1
                            registered = False
                        break
                    try:
                        await asyncio.wait_for(self._condition.wait(), self._probe_interval)
                    except TimeoutError:
                        pass
            yield selected
        finally:
            async with self._condition:
                if registered:
                    self._normal_waiters -= 1
                if selected is not None:
                    async with asyncio.timeout(None):
                        await self._condition.wait_for(lambda: self._group_episodes[group_id] == 0)
                    reserved = self._active_group_by_worker.get(selected, set())
                    reserved.discard(group_id)
                    if not reserved:
                        self._active_group_by_worker.pop(selected, None)
                    self._active_groups.pop(group_id, None)
                    self._group_episodes.pop(group_id)
                self._condition.notify_all()

    @asynccontextmanager
    async def queued_episode(self, key, *, owner, policy, cost_key, allow_help=True):
        """Keep unstarted candidates central until a physical GPU is admitted."""
        scheduler = self.episode_queue
        scheduler.enqueue(key, owner, policy, cost_key, allow_help=allow_help)
        selected = None
        completed = False
        try:
            while selected is None:
                await self._refresh_resources()
                self._require_worker_available(owner)
                self._require_live_target((owner,))
                async with self._condition:
                    available = set(self._worker_ids) - self._resource_blocked - self._paused_workers
                    selected = scheduler.claim_episode(key, available)
                    if selected is not None:
                        self._active[selected] += 1
                    else:
                        try:
                            await asyncio.wait_for(self._condition.wait(), self._probe_interval)
                        except TimeoutError:
                            pass
            yield selected
            completed = True
        finally:
            async with self._condition:
                current = scheduler.active[key][1] if key in scheduler.active else selected
                already_released = key in scheduler.generation_finished
                scheduler.finish_episode(key, completed=completed)
                if current is not None and not already_released:
                    self._active[current] -= 1
                self._condition.notify_all()

    async def finish_episode_generation(self, key):
        """Release physical admission while the logical episode awaits group scoring."""
        async with self._condition:
            worker = self.episode_queue.note_generation_finished(key)
            if worker is not None:
                self._active[worker] -= 1
            self._condition.notify_all()

    async def rebalance_target(self, key, store):
        await self._refresh_resources()
        async with self._condition:
            available = set(self._worker_ids) - self._resource_blocked - self._paused_workers
            if self._resource_probe is not None:
                with self.episode_queue.lock:
                    reserved = {target for _, target in self.episode_queue.transfers.values()}
                available &= self._rebalance_idle_workers | (reserved & self._rebalance_ready_workers)
                request = self.episode_queue.active.get(key)
                if request is not None:
                    available = {w for w in available if request[0].policy in self._rebalance_policies.get(w, ())}
                if reserved - available:
                    self.episode_queue.cancel_rebalance_plans()
        return store.reserve_episode_rebalance(key, available)

    async def commit_episode_transfer(self, key, source, target):
        async with self._condition:
            self.episode_queue.commit_transfer(key, source, target)
            self._active[source] -= 1
            self._active[target] += 1
            self._condition.notify_all()

    @property
    def admission_paused(self) -> bool:
        """Whether all Workers are centrally paused."""
        return self._paused_workers == set(self._worker_ids)

    @property
    def paused_worker_ids(self) -> frozenset[str]:
        return frozenset(self._paused_workers)

    def _admission_targets(self, worker_ids) -> set[str]:
        if worker_ids is None:
            return set(self._worker_ids)
        if isinstance(worker_ids, (str, bytes)):
            raise ValueError("admission targets must be Worker IDs, not a string")
        targets = tuple(worker_ids)
        if (not targets or any(not isinstance(key, str) for key in targets)
                or len(set(targets)) != len(targets) or set(targets) - set(self._worker_ids)):
            raise ValueError("admission targets must name unique configured Workers")
        return set(targets)

    async def pause_and_drain(self, *, timeout: float, worker_ids=None) -> None:
        """Fence selected whole episodes; other Workers retain their leases/admission."""
        targets = self._admission_targets(worker_ids)
        async with self._condition:
            self._paused_workers.update(targets)
            self._condition.notify_all()
            async with asyncio.timeout(timeout):
                await self._condition.wait_for(
                    lambda: (not any(self._active[key] for key in targets)
                             and not any(key in self._active_group_by_worker for key in targets))
                )

    async def resume_admission(self, *, worker_ids=None) -> None:
        targets = self._admission_targets(worker_ids)
        async with self._condition:
            self._paused_workers.difference_update(targets - self.quarantined_workers)
            self._last_probe = float('-inf')
            self._condition.notify_all()

    def _select_worker(self, candidates: Sequence[str] | None = None) -> str:
        available = list(candidates) if candidates is not None else [
            worker_id
            for worker_id in self._worker_ids
            if worker_id not in self._resource_blocked and worker_id not in self._paused_workers
            and worker_id not in self._active_group_by_worker
            and self._active[worker_id] < self._capacities[worker_id]
        ]
        least_active = min(self._active[worker_id] for worker_id in available)
        count = len(self._worker_ids)
        order = tuple((self._next_tie_index + offset) % count for offset in range(count))
        selected_index = next(
            index
            for index in order
            if self._worker_ids[index] in available
            and self._active[self._worker_ids[index]] == least_active
        )
        self._next_tie_index = (selected_index + 1) % count
        return self._worker_ids[selected_index]
