"""One inference-only pool alternates between rollout help and evaluation."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import AsyncIterator

from ..rollout.groups import PolicySnapshot
from .episode_queue import EpisodeRoleQueue
from .errors import WorkerUnavailableError
from .placement import EpisodeWorkerPool

_RETRY_SECONDS = 5.0


class InferencePhase(StrEnum):
    SYNCING = "syncing"
    ROLLOUT = "rollout"
    DRAINING = "draining"
    EVALUATING = "evaluating"


class SharedInferencePool:
    """Coordinate already-validated helper IDs; cloud owners never wait on this pool.

    The training and evaluation lease pools share physical Workers only through
    this gate. A policy sync is a bounded background operation, and evaluation
    holds the gate for the whole frozen taskset rather than for individual turns.
    """

    def __init__(
        self,
        worker_ids: tuple[str, ...],
        *,
        training_pool: EpisodeWorkerPool,
        evaluation_pool: EpisodeWorkerPool,
        scheduler: EpisodeRoleQueue,
        current_policy: Callable[[], PolicySnapshot],
        sync_policy: Callable[[PolicySnapshot], Awaitable[None]],
        drain_timeout: float,
    ) -> None:
        self.worker_ids = worker_ids
        self.training_pool = training_pool
        self.evaluation_pool = evaluation_pool
        self.scheduler = scheduler
        self.current_policy = current_policy
        self.sync_policy = sync_policy
        self.drain_timeout = drain_timeout
        self.phase = InferencePhase.SYNCING
        self.policy: PolicySnapshot | None = None
        self.evaluation_version: int | None = None
        self.error: str | None = None
        self._operation = asyncio.Lock()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._evaluation_pending = False
        self._closed = False

    def request_sync(self) -> None:
        """Coalesce publication notifications; this call never waits for a Worker."""
        if self._closed:
            return
        if self.phase == InferencePhase.ROLLOUT:
            self.scheduler.set_helper_policy(None)
        self._wake.set()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._sync_loop(), name="inference-pool-sync")

    async def _sync_loop(self) -> None:
        while not self._closed:
            await self._wake.wait()
            self._wake.clear()
            failed = False
            async with self._operation:
                if self._evaluation_pending or self._closed:
                    continue
                try:
                    target = self.current_policy()
                    if self.phase == InferencePhase.ROLLOUT and self.policy == target:
                        self.scheduler.set_helper_policy((target.version, target.adapter_sha256))
                        continue
                    self.phase = InferencePhase.SYNCING
                    self.scheduler.set_helper_policy(None)
                    await self.evaluation_pool.pause_and_drain(
                        timeout=self.drain_timeout, worker_ids=self.worker_ids,
                    )
                    await self.training_pool.pause_and_drain(
                        timeout=self.drain_timeout, worker_ids=self.worker_ids,
                    )
                    while not self._evaluation_pending and not self._closed:
                        target = self.current_policy()
                        await self.sync_policy(target)
                        if self._evaluation_pending:
                            break
                        if target != self.current_policy():
                            continue
                        self.policy = target
                        for worker_id in self.worker_ids:
                            await self.training_pool.clear_worker_unavailable(
                                worker_id, source="policy_sync",
                            )
                        await self.training_pool.resume_admission(worker_ids=self.worker_ids)
                        self.scheduler.set_helper_policy((target.version, target.adapter_sha256))
                        self.phase = InferencePhase.ROLLOUT
                        self.error = None
                        break
                except Exception as error:
                    # The pool stays fenced while optional helpers recover.
                    if isinstance(error, WorkerUnavailableError) and error.worker_id is not None:
                        await self.training_pool.mark_worker_unavailable(
                            error.worker_id, source="policy_sync",
                        )
                    self.phase = InferencePhase.SYNCING
                    self.scheduler.set_helper_policy(None)
                    self.error = f"{type(error).__name__}: {error}"[:500]
                    failed = True
            if failed and not self._closed:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=_RETRY_SECONDS)
                except TimeoutError:
                    self._wake.set()

    @asynccontextmanager
    async def evaluation(self, policy: PolicySnapshot) -> AsyncIterator[None]:
        self._evaluation_pending = True
        self.scheduler.set_helper_policy(None)
        try:
            async with self._operation:
                self.phase = InferencePhase.DRAINING
                await self.training_pool.pause_and_drain(
                    timeout=self.drain_timeout, worker_ids=self.worker_ids,
                )
                await self.evaluation_pool.pause_and_drain(
                    timeout=self.drain_timeout, worker_ids=self.worker_ids,
                )
                await self.evaluation_pool.resume_admission(worker_ids=self.worker_ids)
                self.phase = InferencePhase.EVALUATING
                self.evaluation_version = policy.version
                try:
                    yield
                finally:
                    await self.evaluation_pool.pause_and_drain(
                        timeout=self.drain_timeout, worker_ids=self.worker_ids,
                    )
        finally:
            self.evaluation_version = None
            self.phase = InferencePhase.SYNCING
            self._evaluation_pending = False
            self.request_sync()

    def snapshot(self) -> dict:
        return {
            "worker_ids": list(self.worker_ids),
            "phase": self.phase.value,
            "training_policy_version": self.policy.version if self.policy else None,
            "evaluation_policy_version": self.evaluation_version,
            "error": self.error,
        }

    async def close(self) -> None:
        self._closed = True
        self.scheduler.set_helper_policy(None)
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
