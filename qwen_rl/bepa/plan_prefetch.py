"""Bounded background requests for Teacher plans on selected training tasks."""
from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
import math
import threading
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from ..task_source import TaskSpec
    from .coordinator import BepaCoordinator


_PENDING_JOB_STATES = ("queued", "retry_wait", "teacher_running", "source_submitted", "source_verified")
_RETRY_STATES = frozenset(("queue_full", "budget_blocked", "window_full"))
_DEFAULT_RETRY_SECONDS = 5.0
_MIN_RETRY_SECONDS = 0.1
_MAX_RETRY_SECONDS = 60.0


@dataclass(frozen=True)
class PlanPrefetchSnapshot:
    selected: int
    resolved: int
    remaining: int
    running: bool
    pending_window: int
    scope_pending: int | None
    current_task_id: str | None
    state_counts: Mapping[str, int]
    task_states: Mapping[str, str]
    enqueue_calls: int


class TeacherPlanPrefetcher:
    """Request plans for the full selected training set without waiting on Teacher."""

    def __init__(
        self,
        coordinator: BepaCoordinator,
        selected_tasks: Sequence[TaskSpec],
        *,
        retry_interval_seconds: float = _DEFAULT_RETRY_SECONDS,
    ) -> None:
        if isinstance(retry_interval_seconds, bool) or not isinstance(retry_interval_seconds, (int, float)):
            raise ValueError("retry_interval_seconds must be a finite positive number")
        if not math.isfinite(retry_interval_seconds) or retry_interval_seconds <= 0:
            raise ValueError("retry_interval_seconds must be a finite positive number")

        teacher = coordinator.settings["teacher"]
        max_active = teacher["max_active_jobs"]
        max_pending = teacher["max_pending_jobs"]
        if (type(max_active) is not int or max_active < 1
                or type(max_pending) is not int or max_pending < 1):
            raise ValueError("Teacher active and pending job limits must be positive integers")

        tasks_by_id: dict[str, TaskSpec] = {}
        for task in selected_tasks:
            task_id = getattr(task, "task_id", None)
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("selected training tasks need non-empty task_id values")
            if getattr(task, "split", None) != "train" or getattr(task, "pool", None) == "diagnostic":
                raise ValueError(f"Teacher plan prefetch accepts training tasks only: {task_id}")
            if task_id in tasks_by_id:
                raise ValueError(f"selected training task IDs must be unique: {task_id}")
            tasks_by_id[task_id] = task

        self._coordinator = coordinator
        self._tasks = tasks_by_id
        self._pending = deque(tasks_by_id)
        self._states = {task_id: "pending" for task_id in tasks_by_id}
        self._pending_window = min(2 * max_active, max_pending)
        self._retry_interval = min(max(float(retry_interval_seconds), _MIN_RETRY_SECONDS), _MAX_RETRY_SECONDS)
        self._lock = threading.RLock()
        self._stop_event: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._current_task_id: str | None = None
        self._scope_pending: int | None = None
        self._enqueue_calls = 0

    def prioritize(self, task_id: str) -> bool:
        """Move an unrequested selected task to the front of the background scan."""
        if task_id not in self._tasks:
            raise KeyError(f"task is outside the selected training set: {task_id}")
        with self._lock:
            if task_id not in self._pending:
                return False
            self._pending.remove(task_id)
            self._pending.appendleft(task_id)
            return True

    def start(self) -> asyncio.Task[None]:
        """Start or resume the helper on the current event loop."""
        with self._lock:
            if self._task is not None and not self._task.done():
                return self._task
            self._stop_event = asyncio.Event()
            self._task = asyncio.create_task(self.run())
            return self._task

    async def stop(self) -> None:
        """Stop after the in-flight local enqueue returns; no Teacher wait is involved."""
        with self._lock:
            task = self._task
            stop_event = self._stop_event
        if task is None:
            return
        if stop_event is not None:
            stop_event.set()
        await task

    async def run(self) -> None:
        """Run until all selected tasks are requested or the helper is stopped."""
        with self._lock:
            if self._running:
                raise RuntimeError("Teacher plan prefetch is already running")
            self._running = True
            if self._stop_event is None:
                self._stop_event = asyncio.Event()
            stop_event = self._stop_event
        try:
            while not stop_event.is_set():
                with self._lock:
                    task_id = self._pending[0] if self._pending else None
                    self._current_task_id = task_id
                    if task_id is not None:
                        self._states[task_id] = "checking_capacity"
                if task_id is None:
                    return

                try:
                    pending_jobs = await asyncio.to_thread(self._scope_pending_jobs)
                except asyncio.CancelledError:
                    with self._lock:
                        self._states[task_id] = "capacity_check_cancelled"
                    raise
                except Exception as exc:
                    with self._lock:
                        self._states[task_id] = f"scope_error:{type(exc).__name__}"
                        self._current_task_id = None
                    await self._wait_for_retry(stop_event)
                    continue
                with self._lock:
                    self._scope_pending = pending_jobs
                if pending_jobs >= self._pending_window:
                    with self._lock:
                        self._states[task_id] = "window_full"
                    await self._wait_for_retry(stop_event)
                    continue

                with self._lock:
                    self._states[task_id] = "enqueuing"
                    self._enqueue_calls += 1
                try:
                    result = await asyncio.to_thread(self._coordinator.enqueue, self._tasks[task_id])
                    status = result.get("status") if isinstance(result, Mapping) else None
                    if not isinstance(status, str) or not status:
                        raise ValueError("coordinator.enqueue returned no status")
                except asyncio.CancelledError:
                    with self._lock:
                        self._states[task_id] = "enqueue_cancelled"
                    raise
                except Exception as exc:
                    with self._lock:
                        self._states[task_id] = f"enqueue_error:{type(exc).__name__}"
                        self._remove_pending(task_id)
                        self._pending.append(task_id)
                        self._current_task_id = None
                    await self._wait_for_retry(stop_event)
                    continue

                with self._lock:
                    self._states[task_id] = status
                    if status not in _RETRY_STATES:
                        self._remove_pending(task_id)
                        self._current_task_id = None
                if status in _RETRY_STATES:
                    await self._wait_for_retry(stop_event)
        finally:
            with self._lock:
                self._running = False
                self._current_task_id = None

    def snapshot(self) -> PlanPrefetchSnapshot:
        with self._lock:
            states = dict(self._states)
            counts = dict(Counter(states.values()))
            pending = len(self._pending)
            return PlanPrefetchSnapshot(
                selected=len(self._tasks),
                resolved=len(self._tasks) - pending,
                remaining=pending,
                running=self._running,
                pending_window=self._pending_window,
                scope_pending=self._scope_pending,
                current_task_id=self._current_task_id,
                state_counts=counts,
                task_states=states,
                enqueue_calls=self._enqueue_calls,
            )

    def _scope_pending_jobs(self) -> int:
        rows = self._coordinator.store.list_jobs(
            states=_PENDING_JOB_STATES,
            budget_scope=self._coordinator.scope,
            limit=self._coordinator.settings["teacher"]["max_pending_jobs"] + 1,
        )
        return len(rows)

    def _remove_pending(self, task_id: str) -> None:
        try:
            self._pending.remove(task_id)
        except ValueError:
            pass

    async def _wait_for_retry(self, stop_event: asyncio.Event) -> None:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=self._retry_interval)
        except asyncio.TimeoutError:
            pass
