"""Bounded stream of complete, policy-pinned rollout groups."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Generic, TypeVar

from .scheduler import CollectionResult, FilteredLimit

S = TypeVar("S")
T = TypeVar("T")


@dataclass(slots=True)
class GroupClaim(Generic[S, T]):
    """Exactly one same-policy B-group batch removed from the ready queue."""

    policy: S
    behavior_version: int
    learner_start_version: int
    batch_id: str
    items: list[T]
    update_index: int
    ordinals: list[int]


@dataclass(slots=True)
class GroupDispatchOutcome(Generic[T]):
    """One terminal result from a requested group dispatch."""

    ordinal: int
    status: str
    item: T | None
    reason: str | None = None


@dataclass(slots=True)
class GroupRoundClaim(Generic[S, T]):
    """One central-gradient round: P dispatch outcomes and K valid groups."""

    policy: S
    behavior_version: int
    learner_start_version: int
    round_id: str
    requested_groups: int
    items: list[T]
    update_index: int
    ordinals: list[int]
    outcomes: list[GroupDispatchOutcome[T]]

    @property
    def actual_groups(self) -> int:
        return len(self.items)

    @property
    def filtered_groups(self) -> int:
        return sum(outcome.status == "filtered" for outcome in self.outcomes)

    @property
    def batch_id(self) -> str:
        """BEPA reservation ID for the round claim."""
        return self.round_id


@dataclass(slots=True)
class StreamStopResult(Generic[T]):
    """Successfully scored groups left after admission has stopped."""

    unconsumed_items: list[T]
    cancelled: bool


class GroupDisposition(Enum):
    """A complete group may wait for bounded external preparation."""

    DEFERRED = "deferred"


@dataclass(slots=True)
class _Outcome(Generic[T]):
    item: T | None
    accepted: bool
    error: BaseException | None = None
    deferred: bool = False
    cancelled: bool = False


@dataclass(slots=True)
class _RoundState(Generic[T]):
    round_id: str
    version: int
    expected: int
    admitted_ordinals: list[int]
    outcomes: dict[int, GroupDispatchOutcome[T]]


async def _invoke(callback: Callable[..., Any], *args: Any) -> Any:
    """Accept either a sync or async observation callback."""
    value = callback(*args)
    if inspect.isawaitable(value):
        return await value
    return value


class AsyncGroupStream(Generic[S, T]):
    """Keep complete groups flowing under a bounded, version-aware credit.

    The ready queue contains scored complete groups. Dispatch and acceptance
    happen in background tasks, with FIRST_COMPLETED refill. A group remains
    queued until a learner claims it; unfinished legal groups are never
    cancelled merely because faster siblings became ready.

    For update ``i`` the behavior version is ``max(v0, learner_version - 1)``:
    the initial policy supplies updates 0 and 1, and each later policy supplies
    the following update. This leaves at most one update between a behavior
    policy and the Learner and makes every older-version bucket a complete B
    multiple before the Learner can advance past it.
    """

    def __init__(
        self,
        *,
        initial_snapshot: S,
        updates: int,
        groups_per_update: int,
        max_concurrent: int,
        max_outstanding_groups: int,
        max_filtered: int,
        dispatch: Callable[[S, int], Awaitable[T]],
        accept: Callable[[T], Awaitable[bool | GroupDisposition]],
        on_error: Callable[[Exception], Awaitable[None]],
        snapshot_version: Callable[[S], int] | None = None,
        version_of: Callable[[T], int] | None = None,
        on_filtered: Callable[[T | None, str], Any] | None = None,
        sequential_updates: bool = False,
        round_mode: bool = False,
        round_size: int | None = None,
    ) -> None:
        self._snapshot_version = snapshot_version or (lambda snapshot: int(snapshot.version))
        self._version_of = version_of or (lambda item: int(item.policy_version))
        self.initial_version = self._snapshot_version(initial_snapshot)
        if type(sequential_updates) is not bool:
            raise ValueError("sequential_updates must be a boolean")
        if type(round_mode) is not bool:
            raise ValueError("round_mode must be a boolean")
        if round_mode and not sequential_updates:
            raise ValueError("central-gradient round mode requires sequential policy versions")
        self.sequential_updates = sequential_updates
        self.round_mode = round_mode
        self.updates = self._require_int("updates", updates, minimum=0)
        self.groups_per_update = self._require_int("groups_per_update", groups_per_update, minimum=1)
        self.round_size = self._require_int(
            "round_size", groups_per_update if round_size is None else round_size, minimum=1
        )
        self.max_concurrent = self._require_int("max_concurrent", max_concurrent, minimum=1)
        self.max_outstanding_groups = self._require_int(
            "max_outstanding_groups", max_outstanding_groups, minimum=self.groups_per_update
        )
        if self.round_mode and self.max_outstanding_groups < self.round_size:
            raise ValueError("central-gradient max_outstanding_groups must be at least P")
        self.max_filtered = self._require_int("max_filtered", max_filtered, minimum=1)

        self._dispatch = dispatch
        self._accept = accept
        self._on_error = on_error
        self._on_filtered = on_filtered
        self._snapshots: dict[int, S] = {self.initial_version: initial_snapshot}

        self._demand_total: dict[int, int] = defaultdict(int)
        if not self.round_mode:
            for update_index in range(self.updates):
                behavior_version = self.initial_version + (update_index if sequential_updates else max(0, update_index - 1))
                self._demand_total[behavior_version] += self.groups_per_update
        self._demand_remaining = dict(self._demand_total)

        self._ready: dict[int, list[tuple[int, T]]] = defaultdict(list)
        self._inflight: dict[asyncio.Task[_Outcome[T]], tuple[int, int]] = {}
        self._provisional: list[tuple[int, int, T]] = []
        self._deferred: dict[int, tuple[int, T]] = {}
        self._orphaned_items: list[T] = []
        self._claimed: GroupClaim[S, T] | GroupRoundClaim[S, T] | None = None
        self._condition = asyncio.Condition()
        self._wake = asyncio.Event()
        self._pump_task: asyncio.Task[None] | None = None
        self._next_ordinal = 0
        self._admitted_count = 0
        self._filtered_count = 0
        self._filtered_window_count = 0
        self._updates_claimed = 0
        self._updates_completed = 0
        self._latest_dispatch_version: int | None = None
        self._backpressure_override: str | None = None
        self._fatal_error: BaseException | None = None
        self._terminal_filtered = False
        self._stop_admission = False
        self._stopped = False
        self._stop_cancelled = False
        self._admitted_by_version: dict[int, int] = defaultdict(int)
        self._active_round_id: str | None = None
        self._prefetched_round_id: str | None = None
        self._round_states: dict[str, _RoundState[T]] = {}
        self._round_id_by_ordinal: dict[int, str] = {}
        self._rounds_started = 0

    @staticmethod
    def _require_int(name: str, value: int, *, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return value

    def _pending_count(self, version: int) -> int:
        return len(self._ready.get(version, ())) + sum(
            1 for _, (inflight_version, _ordinal) in self._inflight.items()
            if inflight_version == version
        ) + sum(1 for provisional_version, _ordinal, _item in self._provisional
                if provisional_version == version) + sum(
            1 for deferred_version, _item in self._deferred.values()
            if deferred_version == version
        )

    def _ready_count(self, version: int) -> int:
        return len(self._ready.get(version, ()))

    def _record_filtered(self) -> None:
        """Record a filtered group and close this progress window at its cap."""
        self._filtered_count += 1
        self._filtered_window_count += 1
        if self._filtered_window_count >= self.max_filtered:
            self._terminal_filtered = True

    def _record_round_outcome(
        self,
        ordinal: int,
        status: str,
        item: T | None,
        reason: str | None = None,
    ) -> None:
        round_id = self._round_id_by_ordinal.get(ordinal)
        if round_id is None:
            if self.round_mode:
                raise RuntimeError(f"dispatch ordinal {ordinal} has no owning round")
            return
        round_state = self._round_states.get(round_id)
        if round_state is None:
            raise RuntimeError(f"dispatch ordinal {ordinal} belongs to an unknown round")
        if ordinal in round_state.outcomes:
            raise RuntimeError(f"dispatch ordinal {ordinal} produced more than one round outcome")
        if status not in {"accepted", "filtered"}:
            raise ValueError("round outcome status must be accepted or filtered")
        round_state.outcomes[ordinal] = GroupDispatchOutcome(
            ordinal=ordinal,
            status=status,
            item=item,
            reason=reason,
        )

    def _remember_orphaned_item(self, item: T) -> None:
        if not any(existing is item for existing in self._orphaned_items):
            self._orphaned_items.append(item)

    def _forget_orphaned_item(self, item: T) -> None:
        self._orphaned_items = [
            existing for existing in self._orphaned_items if existing is not item
        ]

    def _outstanding_count(self) -> int:
        return (sum(len(items) for items in self._ready.values()) + len(self._inflight)
                + len(self._provisional) + len(self._deferred))

    def _next_dispatch(self) -> tuple[int, S] | None:
        if self._stop_admission or self._fatal_error is not None:
            return None
        if self.round_mode:
            if self._updates_completed >= self.updates or self._active_round_id is None:
                return None
            round_state = self._round_states.get(self._active_round_id)
            if round_state is None or len(round_state.admitted_ordinals) >= round_state.expected:
                return None
            if self._terminal_filtered and len(round_state.admitted_ordinals) >= round_state.expected:
                return None
        elif self._terminal_filtered:
            return None
        if len(self._inflight) >= self.max_concurrent:
            return None
        if self._outstanding_count() >= self.max_outstanding_groups:
            return None

        for version in sorted(self._demand_remaining):
            target = self._demand_remaining[version]
            if target <= 0:
                continue
            snapshot = self._snapshots.get(version)
            if snapshot is None:
                continue
            if self.round_mode:
                round_state = self._round_states.get(self._active_round_id or "")
                if (round_state is None or version != round_state.version
                        or self._admitted_by_version[version] >= self._demand_total[version]):
                    continue
                return version, snapshot
            pending = self._pending_count(version)
            if pending >= target:
                continue
            return version, snapshot
        return None

    async def _run_one(self, snapshot: S, ordinal: int) -> _Outcome[T]:
        item: T | None = None
        try:
            item = await self._dispatch(snapshot, ordinal)
            if self._version_of(item) != self._snapshot_version(snapshot):
                raise RuntimeError("dispatched group changed its pinned behavior policy version")
            decision = await self._accept(item)
            if decision is GroupDisposition.DEFERRED:
                return _Outcome(item, False, deferred=True)
            if not isinstance(decision, bool):
                raise TypeError("group accept callback must return bool or GroupDisposition.DEFERRED")
            return _Outcome(item, decision)
        except asyncio.CancelledError:
            # Cancellation may arrive after dispatch returned a complete group
            # but while its accept callback is running. Preserve that item for
            # stop() cleanup instead of losing its policy reference.
            return _Outcome(item, False, cancelled=True)
        except Exception as exc:
            try:
                await self._on_error(exc)
            except asyncio.CancelledError:
                return _Outcome(item, False, cancelled=True)
            except BaseException as fatal:
                return _Outcome(item, False, fatal)
            return _Outcome(item, False)

    def _admit(self) -> None:
        while True:
            next_item = self._next_dispatch()
            if next_item is None:
                return
            version, snapshot = next_item
            ordinal = self._next_ordinal
            self._next_ordinal += 1
            self._admitted_count += 1
            self._latest_dispatch_version = version
            self._admitted_by_version[version] += 1
            if self.round_mode:
                if self._active_round_id is None:
                    raise RuntimeError("central-gradient dispatch has no active sampling round")
                round_state = self._round_states[self._active_round_id]
                round_state.admitted_ordinals.append(ordinal)
                self._round_id_by_ordinal[ordinal] = round_state.round_id
            task = asyncio.create_task(self._run_one(snapshot, ordinal))
            self._inflight[task] = (version, ordinal)

    async def _notify(self) -> None:
        self._wake.set()
        async with self._condition:
            self._condition.notify_all()

    async def _pump(self) -> None:
        try:
            while True:
                self._admit()
                if self._stop_admission and not self._inflight:
                    return
                if self._inflight:
                    self._wake.clear()
                    # A policy ACK or consumer claim may have opened dispatch
                    # credit since _admit ran. Recheck before blocking on slow
                    # generation tasks.
                    if self._next_dispatch() is not None:
                        continue
                    wake_task = asyncio.create_task(self._wake.wait())
                    try:
                        done, _ = await asyncio.wait(
                            [*self._inflight, wake_task], return_when=asyncio.FIRST_COMPLETED
                        )
                    finally:
                        if not wake_task.done():
                            wake_task.cancel()
                        await asyncio.gather(wake_task, return_exceptions=True)
                    if wake_task in done:
                        self._wake.clear()
                    completed_groups = [task for task in done if task in self._inflight]
                    for task in sorted(completed_groups, key=lambda completed: self._inflight[completed][1]):
                        version, ordinal = self._inflight.pop(task)
                        try:
                            outcome = task.result()
                        except asyncio.CancelledError:
                            if not self._stop_admission:
                                raise
                            continue
                        if outcome.cancelled:
                            if outcome.item is not None:
                                self._orphaned_items.append(outcome.item)
                            continue
                        if outcome.error is not None:
                            if self._fatal_error is None:
                                self._fatal_error = outcome.error
                            self._stop_admission = True
                            if outcome.item is not None:
                                self._orphaned_items.append(outcome.item)
                            continue
                        if outcome.accepted:
                            if outcome.item is None:
                                raise RuntimeError("accepted rollout group cannot be None")
                            self._ready[version].append((ordinal, outcome.item))
                            self._ready[version].sort(key=lambda row: row[0])
                            if self.round_mode:
                                self._record_round_outcome(ordinal, "accepted", outcome.item)
                        elif outcome.deferred:
                            if outcome.item is None:
                                raise RuntimeError("deferred rollout group cannot be None")
                            if self.round_mode:
                                # A central-gradient round is bounded by its P
                                # dispatch outcomes. Rescue work is queued for a
                                # later round rather than holding this slot open.
                                self._record_filtered()
                                if self._on_filtered is not None:
                                    await _invoke(
                                        self._on_filtered, outcome.item,
                                        "round_deferred_filtered",
                                    )
                                self._record_round_outcome(
                                    ordinal, "filtered", outcome.item,
                                    "round_deferred_filtered",
                                )
                            else:
                                if ordinal in self._deferred:
                                    raise RuntimeError("dispatch ordinal was deferred more than once")
                                self._deferred[ordinal] = (version, outcome.item)
                        else:
                            self._record_filtered()
                            if self._on_filtered is not None:
                                try:
                                    await _invoke(self._on_filtered, outcome.item, "group_filtered")
                                except BaseException:
                                    if outcome.item is not None:
                                        self._orphaned_items.append(outcome.item)
                                    raise
                            if self.round_mode:
                                self._record_round_outcome(
                                    ordinal, "filtered", outcome.item, "group_filtered",
                                )
                    await self._notify()
                    continue

                if self._fatal_error is not None or self._stop_admission:
                    await self._notify()
                    return
                if self._terminal_filtered:
                    await self._notify()
                    # A committed update may reopen this guard while _notify()
                    # yields. Keep the producer alive when that happens.
                    if self._terminal_filtered and not self.round_mode:
                        return
                    if self._terminal_filtered and self.round_mode:
                        round_state = self._round_states.get(self._active_round_id or "")
                        if (round_state is None
                                or len(round_state.outcomes) >= round_state.expected):
                            return
                    continue

                async with self._condition:
                    # Recheck after acquiring the condition to avoid a lost
                    # policy publication or consumer notification.
                    if self._next_dispatch() is None and not self._inflight and not self._stop_admission:
                        await self._condition.wait()
        except BaseException:
            self._stop_admission = True
            pending = list(self._inflight)
            for task in pending:
                if not task.done():
                    task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in pending:
                if task.cancelled():
                    continue
                try:
                    outcome = task.result()
                except BaseException:
                    continue
                if outcome.item is not None:
                    self._orphaned_items.append(outcome.item)
            self._inflight.clear()
            await self._notify()
            raise

    async def start(self) -> None:
        if self._pump_task is not None:
            raise RuntimeError("group stream was already started")
        if self.updates == 0:
            self._stop_admission = True
            return
        self._pump_task = asyncio.create_task(self._pump(), name="rollout-group-stream")
        await self._notify()

    async def publish(self, snapshot: S) -> None:
        """Make a newly all-Worker-ACKed policy available for dispatch."""
        version = self._snapshot_version(snapshot)
        if version in self._snapshots:
            if self._snapshots[version] != snapshot:
                raise ValueError("a policy version cannot be published with two snapshots")
            return
        if version != max(self._snapshots) + 1:
            raise ValueError("published policy versions must advance exactly one step")
        self._snapshots[version] = snapshot
        await self._notify()

    @property
    def prefetched_round_id(self) -> str | None:
        """Return the one next-policy round staged behind the active update."""
        return self._prefetched_round_id

    async def prefetch_round(
        self,
        claimed_round_id: str,
        snapshot: S,
        round_id: str,
    ) -> bool:
        """Dispatch exactly one next-policy round while the active update saves.

        The caller must invoke this only after every Worker has ACKed the new
        policy and made it available for sampling. This only stages producer
        work: the current round remains claimed and the next round cannot be
        claimed until ``finish_round(..., update_committed=True)`` releases the
        current update after its durable boundary.

        Returns False when there is no later successful update to feed, the
        current round had no optimizer input, or the filtered limit is already
        terminal. No policy snapshot or dispatch credit is consumed then.
        """
        if not self.round_mode:
            raise RuntimeError("prefetch_round is available only in central-gradient round mode")
        if not isinstance(claimed_round_id, str) or not claimed_round_id:
            raise ValueError("claimed_round_id must be a non-empty string")
        if not isinstance(round_id, str) or not round_id:
            raise ValueError("round_id must be a non-empty string")
        claim = self._claimed
        if not isinstance(claim, GroupRoundClaim) or claim.round_id != claimed_round_id:
            raise RuntimeError("claimed_round_id does not own the active central-gradient update")
        if self._active_round_id != claimed_round_id:
            raise RuntimeError("the active central-gradient round changed before prefetch")
        current_round = self._round_states.get(claimed_round_id)
        if (current_round is None or len(current_round.outcomes) != current_round.expected
                or len(current_round.admitted_ordinals) != current_round.expected):
            raise RuntimeError("the active round must finish all P outcomes before prefetch")
        if self._prefetched_round_id is not None:
            raise RuntimeError("a next-policy round is already prefetched")
        if round_id in self._round_states:
            raise ValueError("round_id is already active or retained")
        if self._fatal_error is not None or self._stop_admission or self._stopped:
            raise RuntimeError("cannot prefetch after rollout admission has stopped")
        if claim.actual_groups == 0 or self._terminal_filtered:
            return False
        if self._updates_completed + 1 >= self.updates:
            return False
        next_version = claim.behavior_version + 1
        if self._snapshot_version(snapshot) != next_version:
            raise ValueError("prefetched policy must advance exactly one version from the active round")
        await self.publish(snapshot)
        # A stop request can close admission while the snapshot notification
        # yields to the event loop. Do not materialize a next round afterward.
        if self._fatal_error is not None or self._stop_admission or self._stopped:
            return False
        round_state = _RoundState(
            round_id=round_id,
            version=next_version,
            expected=self.round_size,
            admitted_ordinals=[],
            outcomes={},
        )
        self._round_states[round_id] = round_state
        self._prefetched_round_id = round_id
        # Keep the old GroupRoundClaim owned while directing only the producer
        # pump at the newly ACKed version. claim_round remains fenced by _claimed.
        self._active_round_id = round_id
        self._demand_total[next_version] += self.round_size
        self._demand_remaining[next_version] = self._demand_total[next_version]
        await self._notify()
        return True

    def deferred_items(self) -> tuple[tuple[int, S, T], ...]:
        """Return a read-only snapshot of held groups in dispatch order.

        The tuple structure is immutable; items retain their original identity
        so the owner can append verified sidecar data before resolving them.
        """
        return tuple(
            (ordinal, self._snapshots[version], item)
            for ordinal, (version, item) in sorted(self._deferred.items())
        )

    async def resolve_deferred(
        self,
        ordinal: int,
        accepted: bool,
        *,
        reason: str = "deferred_rejected",
    ) -> None:
        """Resolve one held group without charging new producer demand.

        Successful resolution returns the original group and ordinal to its
        behavior-version queue. Rejection releases its credit and lets normal
        finite-demand refill dispatch a replacement.
        """
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if not isinstance(accepted, bool):
            raise TypeError("accepted must be a bool")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")
        held = self._deferred.pop(ordinal, None)
        if held is None:
            raise KeyError(f"dispatch ordinal {ordinal} is not deferred")
        version, item = held
        try:
            item_version = self._version_of(item)
        except BaseException as exc:
            self._orphaned_items.append(item)
            if self._fatal_error is None:
                self._fatal_error = exc
            self._stop_admission = True
            await asyncio.shield(self._notify())
            raise self._fatal_error
        if item_version != version:
            self._orphaned_items.append(item)
            if self._fatal_error is None:
                self._fatal_error = RuntimeError(
                    "deferred group changed its pinned behavior policy version"
                )
            self._stop_admission = True
            await asyncio.shield(self._notify())
            raise self._fatal_error
        if accepted:
            rows = self._ready[version]
            rows.append((ordinal, item))
            rows.sort(key=lambda row: row[0])
        else:
            self._record_filtered()
            try:
                if self._on_filtered is not None:
                    await _invoke(self._on_filtered, item, reason)
            except BaseException as exc:
                self._orphaned_items.append(item)
                if self._fatal_error is None:
                    self._fatal_error = exc
                self._stop_admission = True
                await asyncio.shield(self._notify())
                raise
        await self._notify()

    async def fail(self, exc: BaseException) -> None:
        """Wake claims and stop admissions after an external scheduler failure."""
        if not isinstance(exc, BaseException):
            raise TypeError("exc must be a BaseException")
        if self._fatal_error is None:
            self._fatal_error = exc
        self._stop_admission = True
        await self._notify()

    async def claim(
        self,
        learner_start_version: int,
        batch_id: str | None = None,
        *,
        prepare_group: Callable[[T, int, int, str], Awaitable[bool]] | None = None,
        on_claim_abort: Callable[[str, list[T]], Awaitable[None]] | None = None,
    ) -> GroupClaim[S, T]:
        """Claim B same-version complete groups, refilling invalid candidates."""
        if self.round_mode:
            raise RuntimeError("central-gradient streams must use claim_round")
        if self._pump_task is None and self.updates:
            raise RuntimeError("group stream has not been started")
        if self._claimed is not None:
            raise RuntimeError("the previous optimizer update still owns a group claim")
        if learner_start_version != self.initial_version + self._updates_claimed:
            raise ValueError("Learner version does not match the next requested update")
        behavior_version = (learner_start_version if self.sequential_updates
                            else max(self.initial_version, learner_start_version - 1))
        policy_lag = learner_start_version - behavior_version
        if policy_lag < 0 or policy_lag > 1:
            raise ValueError("Learner policy lag must be 0 or 1")
        policy = self._snapshots.get(behavior_version)
        if policy is None:
            raise RuntimeError(f"behavior policy {behavior_version} is no longer pinned")
        claim_id = batch_id or uuid.uuid4().hex
        prepared: list[T] = []
        active_claim: GroupClaim[S, T] | None = None

        async def abort_prepared() -> None:
            provisional = [
                (version, ordinal, item)
                for version, ordinal, item in self._provisional
                if version == behavior_version
            ]
            if provisional and on_claim_abort is not None:
                await on_claim_abort(claim_id, [item for _, _, item in provisional])
            # Prepared items remain eligible after their temporary reservation
            # is released; return them to their dispatch-ordered ready bucket.
            rows = self._ready[behavior_version]
            rows.extend(
                (ordinal, item)
                for version, ordinal, item in self._provisional
                if version == behavior_version
            )
            self._provisional = [
                row for row in self._provisional if row[0] != behavior_version
            ]
            rows.sort(key=lambda row: row[0])
            prepared.clear()
            await self._notify()

        try:
            while len(prepared) < self.groups_per_update:
                async with self._condition:
                    while True:
                        if self._fatal_error is not None:
                            raise self._fatal_error
                        active_old = any(
                            v < behavior_version and rows for v, rows in self._ready.items()
                        ) or any(
                            v < behavior_version for v, _ in self._inflight.values()
                        ) or any(v < behavior_version for v, _, _ in self._provisional)
                        active_old = active_old or any(
                            v < behavior_version for v, _item in self._deferred.values()
                        )
                        if active_old:
                            raise RuntimeError(
                                "an older behavior-policy bucket must be consumed before this update"
                            )
                        if self._ready.get(behavior_version):
                            break
                        pending_for_version = any(
                            v == behavior_version for v, _ in self._inflight.values()
                        ) or any(
                            v == behavior_version for v, _item in self._deferred.values()
                        )
                        if (self._terminal_filtered or self._stop_admission) and not pending_for_version:
                            raise FilteredLimit(
                                CollectionResult(
                                    items=[item for _, item in self._ready.get(behavior_version, ())],
                                    filtered_count=self._filtered_window_count,
                                    admitted_count=self._admitted_count,
                                ),
                                max_filtered=self.max_filtered,
                            )
                        if self._pump_task is not None and self._pump_task.done():
                            error = self._pump_task.exception()
                            if error is not None:
                                raise error
                            if not pending_for_version:
                                raise FilteredLimit(
                                    CollectionResult(
                                        items=[item for _, item in self._ready.get(behavior_version, ())],
                                        filtered_count=self._filtered_window_count,
                                        admitted_count=self._admitted_count,
                                    ),
                                    max_filtered=self.max_filtered,
                                )
                        await self._condition.wait()

                    rows = self._ready.get(behavior_version, [])
                    if not rows:
                        continue
                    ordinal, item = rows.pop(0)
                    if self._version_of(item) != behavior_version:
                        raise RuntimeError("ready group changed its pinned behavior policy version")
                    self._provisional.append((behavior_version, ordinal, item))

                if prepare_group is not None:
                    prepared_ok = await prepare_group(
                        item, behavior_version, learner_start_version, claim_id
                    )
                    if not prepared_ok:
                        self._provisional = [
                            row for row in self._provisional
                            if not (row[0] == behavior_version and row[1] == ordinal)
                        ]
                        self._record_filtered()
                        if self._on_filtered is not None:
                            try:
                                await _invoke(self._on_filtered, item, "claim_validation_failed")
                            except BaseException:
                                self._orphaned_items.append(item)
                                raise
                        await self._notify()
                        continue
                prepared.append(item)
                await self._notify()

            self._demand_remaining[behavior_version] -= self.groups_per_update
            if self._demand_remaining[behavior_version] < 0:
                raise RuntimeError("group stream claimed more groups than the finite update budget")
            self._updates_claimed += 1
            claim = GroupClaim(
                policy=policy,
                behavior_version=behavior_version,
                learner_start_version=learner_start_version,
                batch_id=claim_id,
                items=list(prepared),
                update_index=self._updates_claimed - 1,
                ordinals=[
                    ordinal for version, ordinal, _item in self._provisional
                    if version == behavior_version
                ],
            )
            active_claim = claim
            self._claimed = claim
            self._provisional = [
                row for row in self._provisional if row[0] != behavior_version
            ]
            prepared.clear()
            await self._notify()
            return claim
        except BaseException:
            if active_claim is not None and self._claimed is active_claim:
                self._claimed = None
                self._updates_claimed -= 1
                self._demand_remaining[behavior_version] += self.groups_per_update
                if on_claim_abort is not None:
                    await on_claim_abort(claim_id, list(active_claim.items))
                self._ready[behavior_version].extend(
                    zip(active_claim.ordinals, active_claim.items)
                )
                self._ready[behavior_version].sort(key=lambda row: row[0])
                active_claim = None
                await asyncio.shield(self._notify())
            await abort_prepared()
            raise

    def round_first_ordinal(self, round_id: str) -> int:
        """Freeze the controller's next/current admission range before staging."""
        state = self._round_states.get(round_id)
        return state.admitted_ordinals[0] if state and state.admitted_ordinals else self._next_ordinal

    async def claim_round(
        self,
        learner_start_version: int,
        round_id: str | None = None,
        *,
        prepare_group: Callable[[T, int, int, str], Awaitable[bool]] | None = None,
        on_group_ready: Callable[[T, str], Awaitable[None]] | None = None,
        on_group_outcome: Callable[[GroupDispatchOutcome[T], str], Awaitable[None]] | None = None,
        on_claim_abort: Callable[[str, list[T]], Awaitable[None]] | None = None,
    ) -> GroupRoundClaim[S, T]:
        """Collect exactly P dispatch outcomes and return their K valid groups.

        Central-gradient mode intentionally does not refill filtered outcomes
        inside a round. A K=0 claim is still returned so the caller can finish
        the round without advancing the optimizer, policy, or successful-step
        counter, then request another P outcomes at the same policy version.
        Once a round starts, the filtered limit may be exceeded by at most
        P-1 outcomes so every already-requested sibling can finish normally.
        """
        if not self.round_mode:
            raise RuntimeError("claim_round is available only in central-gradient round mode")
        if self._pump_task is None:
            raise RuntimeError("group stream has not been started")
        if self._claimed is not None:
            raise RuntimeError("the previous sampling round still owns the stream")
        adopting_prefetch = self._prefetched_round_id is not None
        if self._active_round_id is not None and not adopting_prefetch:
            raise RuntimeError("the previous sampling round still owns the stream")
        if self._updates_completed >= self.updates:
            raise RuntimeError("the requested successful-update budget is complete")
        if self._terminal_filtered and not adopting_prefetch:
            raise FilteredLimit(
                CollectionResult(items=[], filtered_count=self._filtered_window_count,
                                 admitted_count=self._admitted_count),
                max_filtered=self.max_filtered,
            )
        expected_version = self.initial_version + self._updates_completed
        if learner_start_version != expected_version:
            raise ValueError("Learner version does not match the next central-gradient round")
        policy = self._snapshots.get(learner_start_version)
        if policy is None:
            raise RuntimeError(f"current policy {learner_start_version} is no longer pinned")
        selected_round_id = round_id or uuid.uuid4().hex
        if not isinstance(selected_round_id, str) or not selected_round_id:
            raise ValueError("round_id must be a non-empty string")
        if adopting_prefetch:
            if selected_round_id != self._prefetched_round_id:
                raise ValueError("round_id must match the prefetched next-policy round")
            if self._active_round_id != selected_round_id:
                raise RuntimeError("prefetched round is not the active producer round")
            round_state = self._round_states.get(selected_round_id)
            if round_state is None or round_state.version != learner_start_version:
                raise RuntimeError("prefetched round does not match the next durable learner version")
        else:
            if selected_round_id in self._round_states:
                raise ValueError("round_id is already active or retained")
            round_state = _RoundState(
                round_id=selected_round_id,
                version=learner_start_version,
                expected=self.round_size,
                admitted_ordinals=[],
                outcomes={},
            )
            self._round_states[selected_round_id] = round_state
            self._active_round_id = selected_round_id
            self._demand_total[learner_start_version] += self.round_size
            self._demand_remaining[learner_start_version] = self._demand_total[learner_start_version]
            await self._notify()

        try:
            prepared: list[GroupDispatchOutcome[T]] = []
            accepted: list[GroupDispatchOutcome[T]] = []
            seen: set[int] = set()
            try:
                while True:
                    arrivals = [round_state.outcomes[ordinal]
                        for ordinal in round_state.admitted_ordinals
                        if ordinal in round_state.outcomes and ordinal not in seen]
                    for outcome in arrivals:
                        seen.add(outcome.ordinal)
                        if outcome.status != "accepted":
                            if on_group_outcome is not None:
                                await on_group_outcome(outcome, selected_round_id)
                            continue
                        accepted.append(outcome)
                        assert outcome.item is not None
                        if prepare_group is not None:
                            valid = await prepare_group(
                                outcome.item,
                                learner_start_version,
                                learner_start_version,
                                selected_round_id,
                            )
                            if not valid:
                                self._record_filtered()
                                filtered = GroupDispatchOutcome(
                                    ordinal=outcome.ordinal,
                                    status="filtered",
                                    item=outcome.item,
                                    reason="claim_validation_failed",
                                )
                                round_state.outcomes[outcome.ordinal] = filtered
                                self._ready[learner_start_version] = [
                                    (ordinal, item)
                                    for ordinal, item in self._ready.get(learner_start_version, ())
                                    if ordinal != outcome.ordinal
                                ]
                                if self._on_filtered is not None:
                                    await _invoke(
                                        self._on_filtered, outcome.item,
                                        "claim_validation_failed",
                                    )
                                if on_group_outcome is not None:
                                    await on_group_outcome(filtered, selected_round_id)
                                continue
                        prepared.append(outcome)
                        if on_group_ready is not None:
                            await on_group_ready(outcome.item, selected_round_id)
                        if on_group_outcome is not None:
                            await on_group_outcome(outcome, selected_round_id)
                    async with self._condition:
                        if self._fatal_error is not None:
                            raise self._fatal_error
                        if len(seen) == round_state.expected:
                            break
                        if self._pump_task is None or self._pump_task.done():
                            error = self._pump_task.exception() if self._pump_task is not None else None
                            if error is not None:
                                raise error
                            raise RuntimeError("rollout stream stopped before all round outcomes completed")
                        # Check under the condition to avoid missing a producer wake.
                        if any(ordinal not in seen for ordinal in round_state.outcomes):
                            continue
                        await self._condition.wait()
            except BaseException as prepare_error:
                if isinstance(prepare_error, asyncio.CancelledError):
                    raise
                # A round already admitted as many as P siblings. Keep its
                # owner map alive until the producer records every outcome;
                # otherwise a completed sibling can arrive after the round
                # state is removed and be dropped while the pump handles it.
                async with self._condition:
                    while len(round_state.outcomes) < round_state.expected:
                        if self._pump_task is None or self._pump_task.done():
                            break
                        await self._condition.wait()

                accepted_by_ordinal = {outcome.ordinal: outcome for outcome in accepted}
                for ordinal in round_state.admitted_ordinals:
                    outcome = round_state.outcomes.get(ordinal)
                    if outcome is not None and outcome.status == "accepted":
                        accepted_by_ordinal.setdefault(ordinal, outcome)
                accepted = [
                    accepted_by_ordinal[ordinal]
                    for ordinal in round_state.admitted_ordinals
                    if ordinal in accepted_by_ordinal
                ]
                accepted_items = [row.item for row in accepted if row.item is not None]
                pending_cleanup = [
                    current.item
                    for outcome in accepted
                    if (current := round_state.outcomes[outcome.ordinal]).status == "accepted"
                    and current.item is not None
                ]
                for item in pending_cleanup:
                    self._remember_orphaned_item(item)
                # Remove the failed round's ready credit before invoking
                # callbacks: even a callback failure must not strand P slots.
                round_ordinals = set(round_state.admitted_ordinals)
                self._ready[learner_start_version] = [
                    (ordinal, item)
                    for ordinal, item in self._ready.get(learner_start_version, ())
                    if ordinal not in round_ordinals
                ]
                for ordinal in round_ordinals:
                    deferred = self._deferred.pop(ordinal, None)
                    if deferred is not None and deferred[1] is not None:
                        self._remember_orphaned_item(deferred[1])
                try:
                    if accepted_items and on_claim_abort is not None:
                        await on_claim_abort(selected_round_id, accepted_items)
                    for outcome in accepted:
                        current = round_state.outcomes[outcome.ordinal]
                        if current.status != "accepted" or current.item is None:
                            continue
                        self._record_filtered()
                        round_state.outcomes[outcome.ordinal] = GroupDispatchOutcome(
                            ordinal=outcome.ordinal,
                            status="filtered",
                            item=current.item,
                            reason="claim_prepare_exception",
                        )
                        if self._on_filtered is not None:
                            await _invoke(
                                self._on_filtered,
                                current.item,
                                "claim_prepare_exception",
                            )
                            if (on_claim_abort is not None
                                    and all(item is not None for item in pending_cleanup)):
                                self._forget_orphaned_item(current.item)
                except BaseException as cleanup_error:
                    self._fatal_error = cleanup_error
                    self._stop_admission = True
                    raise cleanup_error from prepare_error
                raise

            ordinals = set(round_state.admitted_ordinals)
            self._ready[learner_start_version] = [
                (ordinal, item)
                for ordinal, item in self._ready.get(learner_start_version, ())
                if ordinal not in ordinals
            ]
            claim = GroupRoundClaim(
                policy=policy,
                behavior_version=learner_start_version,
                learner_start_version=learner_start_version,
                round_id=selected_round_id,
                requested_groups=self.round_size,
                items=[row.item for row in prepared if row.item is not None],
                update_index=self._updates_completed,
                ordinals=[row.ordinal for row in prepared],
                outcomes=[
                    round_state.outcomes[ordinal]
                    for ordinal in round_state.admitted_ordinals
                ],
            )
            self._claimed = claim
            if adopting_prefetch:
                self._prefetched_round_id = None
            await self._notify()
            return claim
        except asyncio.CancelledError:
            # Keep the active round fence and its ordinal map until stop() has
            # joined/cancelled the producer-owned tasks. A second claim is
            # rejected explicitly instead of reusing half-finished credits.
            if self._fatal_error is None:
                self._fatal_error = RuntimeError(
                    "central-gradient round was cancelled before all P outcomes completed"
                )
            self._stop_admission = True
            await asyncio.shield(self._notify())
            raise
        except BaseException:
            if self._active_round_id == selected_round_id and self._claimed is None:
                self._active_round_id = None
                self._round_states.pop(selected_round_id, None)
                if self._prefetched_round_id == selected_round_id:
                    self._prefetched_round_id = None
                for ordinal in round_state.admitted_ordinals:
                    self._round_id_by_ordinal.pop(ordinal, None)
                await asyncio.shield(self._notify())
            raise

    async def finish_round(self, round_id: str, *, update_committed: bool) -> None:
        """Release a round claim and count only an actual successful update."""
        if not self.round_mode:
            raise RuntimeError("finish_round is available only in central-gradient round mode")
        if not isinstance(update_committed, bool):
            raise TypeError("update_committed must be a bool")
        claim = self._claimed
        if not isinstance(claim, GroupRoundClaim) or claim.round_id != round_id:
            raise RuntimeError("round_id does not own the active central-gradient claim")
        if update_committed and claim.actual_groups == 0:
            raise RuntimeError("an empty round cannot commit an optimizer update")
        prefetched_round_id = self._prefetched_round_id
        if prefetched_round_id is not None:
            if not update_committed:
                raise RuntimeError(
                    "a failed update with a prefetched round must stop the stream instead of releasing the claim"
                )
            if self._active_round_id != prefetched_round_id:
                raise RuntimeError("prefetched round lost active producer ownership")
        elif self._active_round_id != round_id:
            raise RuntimeError("round_id no longer owns producer admission")
        prefetched_filtered_count = 0
        if update_committed and prefetched_round_id is not None:
            prefetched_state = self._round_states.get(prefetched_round_id)
            if prefetched_state is None:
                raise RuntimeError("prefetched round state disappeared before the prior update committed")
            prefetched_filtered_count = sum(
                outcome.status == "filtered"
                for outcome in prefetched_state.outcomes.values()
            )
        if update_committed:
            self._updates_completed += 1
            self._filtered_window_count = prefetched_filtered_count
            self._terminal_filtered = prefetched_filtered_count >= self.max_filtered
        self._claimed = None
        if prefetched_round_id is None:
            self._active_round_id = None
        round_state = self._round_states.pop(round_id, None)
        if round_state is not None:
            for ordinal in round_state.admitted_ordinals:
                self._round_id_by_ordinal.pop(ordinal, None)
        if (update_committed and self._updates_completed < self.updates
                and self._pump_task is not None and self._pump_task.done()
                and self._fatal_error is None and not self._stop_admission):
            self._pump_task = asyncio.create_task(
                self._pump(), name="rollout-group-stream-round",
            )
        await self._notify()

    async def finish_update(self) -> None:
        """Release a committed claim and reset its bounded no-progress window."""
        if self.round_mode:
            raise RuntimeError("central-gradient streams must use finish_round")
        if self._claimed is None:
            raise RuntimeError("there is no active group claim")
        self._claimed = None
        self._updates_completed += 1
        self._filtered_window_count = 0
        if self._terminal_filtered:
            self._terminal_filtered = False
            if self._fatal_error is None and not self._stop_admission and not self._stopped:
                if self._pump_task is None or self._pump_task.done():
                    self._pump_task = asyncio.create_task(
                        self._pump(), name="rollout-group-stream"
                    )
        await self._notify()

    def snapshot(self) -> dict[str, Any]:
        """Return the stable, lightweight queue and backpressure view."""
        ready_by_version = {
            str(version): len(items)
            for version, items in sorted(self._ready.items())
            if items
        }
        inflight_by_version: dict[str, int] = {}
        for version, _ in self._inflight.values():
            key = str(version)
            inflight_by_version[key] = inflight_by_version.get(key, 0) + 1
        claimed_by_version: dict[str, int] = {}
        if self._claimed is not None:
            claimed_by_version[str(self._claimed.behavior_version)] = len(self._claimed.items)
        preparing_by_version: dict[str, int] = {}
        for version, _ordinal, _item in self._provisional:
            key = str(version)
            preparing_by_version[key] = preparing_by_version.get(key, 0) + 1
        deferred_by_version: dict[str, int] = {}
        for version, _item in self._deferred.values():
            key = str(version)
            deferred_by_version[key] = deferred_by_version.get(key, 0) + 1
        versions = [int(key) for key, count in ready_by_version.items() if count]
        versions.extend(int(key) for key, count in inflight_by_version.items() if count)
        versions.extend(int(key) for key, count in claimed_by_version.items() if count)
        versions.extend(int(key) for key, count in preparing_by_version.items() if count)
        versions.extend(int(key) for key, count in deferred_by_version.items() if count)

        producer_reason = self._backpressure_override
        update_count = self._updates_completed if self.round_mode else self._updates_claimed
        if producer_reason is None:
            if self._fatal_error is not None:
                producer_reason = "group_error"
            elif self._terminal_filtered:
                producer_reason = "filtered_group_limit"
            elif update_count >= self.updates:
                producer_reason = "finite_update_budget"
            elif self._outstanding_count() >= self.max_outstanding_groups:
                producer_reason = "group_credit_full"
            elif len(self._inflight) >= self.max_concurrent:
                producer_reason = "rollout_concurrency_full"
            else:
                missing_version = next((
                    version for version in sorted(self._demand_remaining)
                    if self._demand_remaining[version] > self._pending_count(version)
                ), None)
                if missing_version is not None and missing_version not in self._snapshots:
                    producer_reason = "awaiting_policy_publish"

        learner_reason: str | None = None
        if self._claimed is None and update_count < self.updates:
            required_version = self.initial_version + (
                self._updates_completed if self.round_mode else
                (self._updates_claimed if self.sequential_updates else max(0, self._updates_claimed - 1))
            )
            ready_count = self._ready_count(required_version)
            if self._provisional:
                learner_reason = "claim_validation"
            elif self.round_mode and self._active_round_id is None:
                learner_reason = "awaiting_round_claim"
            elif self.round_mode and self._active_round_id is not None:
                round_state = self._round_states.get(self._active_round_id)
                if round_state is not None and len(round_state.outcomes) < round_state.expected:
                    learner_reason = "central_round_outcomes"
                else:
                    learner_reason = "central_round_ready"
            elif ready_count < self.groups_per_update:
                if any(v == required_version for v, _ in self._inflight.values()):
                    learner_reason = "oldest_version_incomplete"
                elif any(v == required_version for v, _item in self._deferred.values()):
                    learner_reason = "awaiting_deferred_group"
                else:
                    learner_reason = "awaiting_ready_groups"
        elif self._claimed is None:
            required_version = None
        else:
            required_version = self._claimed.behavior_version

        reason = learner_reason or producer_reason

        return {
            "ready_groups": sum(len(items) for items in self._ready.values()),
            "inflight_groups": len(self._inflight),
            "claimed_groups": len(self._claimed.items) if self._claimed is not None else 0,
            "preparing_groups": len(self._provisional),
            "deferred_groups": len(self._deferred),
            "filtered_groups_total": self._filtered_count,
            "filtered_groups_since_update": self._filtered_window_count,
            "max_filtered_groups_per_update": self.max_filtered,
            "outstanding_groups": self._outstanding_count(),
            "maximum_outstanding_groups": self.max_outstanding_groups,
            "ready_by_version": ready_by_version,
            "inflight_by_version": inflight_by_version,
            "claimed_by_version": claimed_by_version,
            "preparing_by_version": preparing_by_version,
            "deferred_by_version": deferred_by_version,
            "backpressure_reason": reason,
            "producer_backpressure_reason": producer_reason,
            "learner_backpressure_reason": learner_reason,
            "learner_required_behavior_version": required_version,
            "oldest_behavior_version": min(versions) if versions else None,
            "latest_dispatch_version": self._latest_dispatch_version,
        }

    async def stop(self, *, drain: bool = True) -> StreamStopResult[T]:
        """Stop admissions and either drain or cancel every started group task."""
        if self._stopped:
            return StreamStopResult([], cancelled=self._stop_cancelled)
        self._stop_admission = True
        self._backpressure_override = "caller_cancelled" if not drain else None
        await self._notify()
        if not drain:
            for task in list(self._inflight):
                if not task.done():
                    task.cancel()
        if self._pump_task is not None:
            try:
                await self._pump_task
            except asyncio.CancelledError:
                if drain:
                    raise
            except Exception:
                # The group failure is also recorded in snapshot()/run state;
                # stop must still return all successful siblings.
                pass
        unconsumed = self._take_unconsumed_items()
        self._active_round_id = None
        self._prefetched_round_id = None
        self._round_states.clear()
        self._round_id_by_ordinal.clear()
        self._stopped = True
        self._stop_cancelled = not drain
        await self._notify()
        return StreamStopResult(unconsumed, cancelled=not drain)

    async def stop_admission(self) -> None:
        """Fence new producer work while retaining the current claimed update."""
        if self._stopped:
            return
        self._stop_admission = True
        await self._notify()

    def _take_unconsumed_items(self) -> list[T]:
        items = list(self._orphaned_items)
        self._orphaned_items.clear()
        items.extend(item for version in sorted(self._ready) for _, item in self._ready[version])
        self._ready.clear()
        items.extend(item for _version, _ordinal, item in self._provisional)
        self._provisional.clear()
        items.extend(item for _ordinal, (_version, item) in sorted(self._deferred.items()))
        self._deferred.clear()
        if self._claimed is not None:
            items.extend(self._claimed.items)
            self._claimed = None
        return items
