"""Bounded, run-owned polling for already verified BEPA seed plans.

The scheduler owns at most one guided Worker attempt at a time. Plan discovery,
attempt execution, and cache handoff are supplied by the Learner so this module
does not know about BEPA persistence, Worker placement, or training batches.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Generic, Sequence, TypeVar


Candidate = TypeVar("Candidate")


class SeedAwareGenerationGate:
    """Share bounded generation capacity while giving a waiting seed fair entry.

    Existing generations keep their permits. Once a seed starts waiting, newly
    admitted native groups wait until one permit drains and the seed acquires it.
    When no seed is waiting, native groups can use the full configured capacity.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("generation capacity must be a positive integer")
        self.capacity = capacity
        self._active = 0
        self._seed_waiters = 0
        self._condition = asyncio.Condition()

    @property
    def active(self) -> int:
        return self._active

    @property
    def seed_waiters(self) -> int:
        return self._seed_waiters

    @asynccontextmanager
    async def lease(self, *, seed: bool = False) -> AsyncIterator[None]:
        registered_seed_waiter = False
        acquired = False
        try:
            async with self._condition:
                if seed:
                    self._seed_waiters += 1
                    registered_seed_waiter = True
                    self._condition.notify_all()
                await self._condition.wait_for(
                    lambda: self._active < self.capacity
                    and (seed or self._seed_waiters == 0)
                )
                self._active += 1
                acquired = True
                if registered_seed_waiter:
                    self._seed_waiters -= 1
                    registered_seed_waiter = False
                self._condition.notify_all()
            yield
        finally:
            async with self._condition:
                if acquired:
                    self._active -= 1
                if registered_seed_waiter:
                    self._seed_waiters -= 1
                self._condition.notify_all()


class SeedAttemptResult(str, Enum):
    """Disposition of one candidate returned by the durable plan query."""

    SKIPPED = "skipped"  # A competing claim/cache made this candidate stale.
    SCHEDULED = "scheduled"  # A central whole-group rescue was queued for rollout.
    FAILED = "failed"  # A real guided Worker attempt completed without a cache.
    READY = "ready"  # A verified student cache was committed.


class RunSeedScheduler(Generic[Candidate]):
    """Poll eligible verified plans and run one guided seed at a time.

    ``discover`` must be bounded and run-scoped. ``attempt`` is responsible for
    the atomic plan claim, fixed Worker policy reference, guided execution, and
    durable failure/success recording. A successful callback may resolve a
    deferred rollout group before returning.
    """

    def __init__(
        self,
        run_id: str,
        *,
        discover: Callable[[], Awaitable[Sequence[Candidate]]],
        attempt: Callable[[Candidate], Awaitable[SeedAttemptResult]],
        on_error: Callable[[BaseException], Awaitable[None]],
        housekeeping: Callable[[], Awaitable[None]] | None = None,
        oldest_hold_seconds: Callable[[], float] | None = None,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        if (isinstance(poll_interval_seconds, bool)
                or not isinstance(poll_interval_seconds, (int, float))
                or poll_interval_seconds <= 0):
            raise ValueError("poll_interval_seconds must be positive")
        self.run_id = run_id
        self._discover = discover
        self._attempt = attempt
        self._on_error = on_error
        self._housekeeping = housekeeping
        self._oldest_hold_seconds = oldest_hold_seconds or (lambda: 0.0)
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self._status = "idle"
        self._active_task_id: str | None = None
        self._attempts = 0
        self._succeeded = 0
        self._failed = 0
        self._skipped = 0
        self._scheduled = 0
        self._checks = 0
        self._candidates = 0
        self._last_check_at: float | None = None
        self._last_error: dict[str, str] | None = None

    def start(self) -> None:
        """Start this scheduler once for its owning run."""
        if self._task is not None:
            raise RuntimeError("run seed scheduler was already started")
        self._status = "running"
        self._task = asyncio.create_task(
            self._run(), name=f"bepa-seed-scheduler-{self.run_id}"
        )

    def wake(self) -> None:
        """Request an immediate plan scan after a run-local state change."""
        if self._task is not None and not self._task.done():
            self._wake.set()

    async def close(self, *, drain: bool = False) -> None:
        """Stop scheduling; optionally finish the current complete guided episode."""
        self._closing = True
        self._wake.set()
        task = self._task
        if task is None:
            self._status = "stopped"
            return
        if not task.done() and not drain:
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if self._status not in {"failed", "stopped"}:
            self._status = "stopped"

    def snapshot(self) -> dict[str, Any]:
        """Return compact run-scoped scheduler telemetry for health and monitor."""
        try:
            oldest_hold = float(self._oldest_hold_seconds())
        except Exception:
            oldest_hold = 0.0
        if oldest_hold < 0:
            oldest_hold = 0.0
        return {
            "run_id": self.run_id,
            "status": self._status,
            "active_task_id": self._active_task_id,
            "attempts": self._attempts,
            "succeeded": self._succeeded,
            "failed": self._failed,
            "skipped": self._skipped,
            "scheduled": self._scheduled,
            "checks": self._checks,
            "candidates": self._candidates,
            "oldest_hold_seconds": oldest_hold,
            "poll_interval_seconds": self._poll_interval_seconds,
            "last_check_at": self._last_check_at,
            "last_error": self._last_error,
        }

    async def _wait(self) -> None:
        self._wake.clear()
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=self._poll_interval_seconds)
        except TimeoutError:
            pass

    async def _run(self) -> None:
        active_attempt: asyncio.Task[SeedAttemptResult] | None = None
        try:
            while not self._closing:
                if self._housekeeping is not None:
                    await self._housekeeping()
                self._checks += 1
                self._last_check_at = time.time()
                candidates = await self._discover()
                self._candidates += len(candidates)
                for candidate in candidates:
                    if self._closing:
                        return
                    self._active_task_id = self._candidate_task_id(candidate)
                    active_attempt = asyncio.create_task(self._attempt(candidate))
                    while not active_attempt.done():
                        await asyncio.wait(
                            {active_attempt}, timeout=self._poll_interval_seconds,
                        )
                        if (not active_attempt.done() and self._housekeeping is not None
                                and not self._closing):
                            await self._housekeeping()
                    result = active_attempt.result()
                    active_attempt = None
                    if not isinstance(result, SeedAttemptResult):
                        raise TypeError("seed attempt callback returned an unknown disposition")
                    if result is SeedAttemptResult.SKIPPED:
                        self._skipped += 1
                        continue
                    self._attempts += 1
                    if result is SeedAttemptResult.SCHEDULED:
                        self._scheduled += 1
                    elif result is SeedAttemptResult.READY:
                        self._succeeded += 1
                    else:
                        self._failed += 1
                    # One guided Worker attempt per scan keeps this loop bounded
                    # and leaves available Worker capacity to the rollout stream.
                    break
                self._active_task_id = None
                await self._wait()
        except asyncio.CancelledError:
            self._status = "stopping"
            raise
        except BaseException as exc:
            self._status = "failed"
            self._last_error = {"type": type(exc).__name__, "message": str(exc)[:300]}
            self._active_task_id = None
            await self._on_error(exc)
            return
        finally:
            if active_attempt is not None:
                if not active_attempt.done():
                    active_attempt.cancel()
                await asyncio.gather(active_attempt, return_exceptions=True)
            self._active_task_id = None
            if self._status not in {"failed", "stopping"}:
                self._status = "stopped" if self._closing else "idle"

    @staticmethod
    def _candidate_task_id(candidate: Any) -> str | None:
        if isinstance(candidate, Mapping):
            task_id = candidate.get("task_id")
            if isinstance(task_id, str):
                return task_id
        task_id = getattr(candidate, "task_id", None)
        if isinstance(task_id, str):
            return task_id
        task_key = getattr(candidate, "task_key", None)
        return task_key if isinstance(task_key, str) else None
