"""Owned Worker ACK and complete-group stream publication lifecycle."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ...rollout.groups import PolicySnapshot
from .state import RunExecutionState


class PolicyHandoffCoordinator:
    """Serialize the pending policy handoff and drain it before run cleanup."""

    def __init__(
        self,
        *,
        execution: RunExecutionState,
        sync_workers: Callable[[dict[str, Any]], Awaitable[dict[str, dict[str, Any]]]],
        set_failure_stage: Callable[[str], None],
        set_serve_version: Callable[[int], None],
    ) -> None:
        self.execution = execution
        self.sync_workers = sync_workers
        self.set_failure_stage = set_failure_stage
        self.set_serve_version = set_serve_version

    async def start(self, result: dict[str, Any], stream: Any) -> None:
        if self.execution.pending_policy_handoff is not None:
            raise RuntimeError("the previous policy handoff is still pending")
        snapshot = PolicySnapshot.from_policy(result["policy"])
        self.execution.references.retain_demand(snapshot)
        handoff_record = {
            "policy_version": snapshot.version,
            "adapter_sha256": snapshot.adapter_sha256,
            "status": "pending",
            "scheduled_at": time.time(),
            "acknowledged_at": None,
            "published_at": None,
            "error": None,
        }
        self.execution.record.setdefault("policy_handoffs", []).append(handoff_record)

        async def run_handoff() -> None:
            try:
                await self._handoff(result, snapshot, handoff_record, stream)
            except BaseException as exc:
                handoff_record.update(
                    status="failed",
                    finished_at=time.time(),
                    error={"type": type(exc).__name__, "message": str(exc)[:300]},
                )
                raise

        self.execution.pending_policy_handoff = asyncio.create_task(
            run_handoff(), name=f"policy-handoff-v{snapshot.version}",
        )

    async def _handoff(
        self,
        result: dict[str, Any],
        snapshot: PolicySnapshot,
        handoff_record: dict[str, Any],
        stream: Any,
    ) -> None:
        self.set_failure_stage("worker_policy_handoff")
        handoff_record["status"] = "syncing"
        handoff_record["started_at"] = time.time()
        sync_started = time.perf_counter()
        acknowledgements = await self.sync_workers(result["policy"])
        self.execution.worker_sync_acks = acknowledgements
        self.execution.record["metrics"]["learner"]["worker_sync_seconds"] += (
            time.perf_counter() - sync_started
        )
        handoff_record["acknowledged_at"] = time.time()
        for group in result["batch"].groups:
            group["record"]["policy_acks"] = acknowledgements
        self.execution.record["serve_policy_version"] = result["policy"]["version"]
        self.set_serve_version(result["policy"]["version"])
        if stream is None:
            raise RuntimeError("rollout stream ended before policy handoff publication")
        handoff_record["status"] = "publishing"
        await stream.publish(snapshot)
        handoff_record["status"] = "published"
        handoff_record["published_at"] = time.time()
        handoff_record["finished_at"] = handoff_record["published_at"]

    async def settle(self, *, wait: bool) -> Exception | None:
        task = self.execution.pending_policy_handoff
        if task is None or (not wait and not task.done()):
            return None
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done() and task.cancelled():
                self.execution.pending_policy_handoff = None
            raise
        except Exception as exc:  # noqa: BLE001 - surface the owned handoff's original failure
            self.execution.pending_policy_handoff = None
            return exc
        self.execution.pending_policy_handoff = None
        return None

    async def drain(self) -> tuple[Exception | None, bool]:
        """Wait for owned ACK/publish work, preserving caller cancellation."""
        task = self.execution.pending_policy_handoff
        if task is None:
            return None, False
        interrupted = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    break
                interrupted = True
            except Exception:  # noqa: BLE001 - keep draining the owned task before cleanup
                break
        self.execution.pending_policy_handoff = None
        if task.cancelled():
            return RuntimeError("policy handoff task was cancelled before completion"), interrupted
        try:
            task.result()
        except Exception as exc:  # noqa: BLE001 - return task failure to the run finalizer
            return exc, interrupted
        return None, interrupted
