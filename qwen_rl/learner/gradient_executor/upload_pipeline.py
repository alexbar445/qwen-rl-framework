"""One immutable CPU contribution in transport while the next row computes."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UploadSlot:
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    transport_allowed: asyncio.Event = field(default_factory=asyncio.Event)
    lease: dict[str, Any] | None = None

    async def handoff(self, claim):
        self.lease = {"job_id": claim['job']['job_id'], "lease_id": claim['lease_id'],
                      "lease_epoch": claim['lease_epoch']}
        self.ready.set()
        await self.transport_allowed.wait()


async def run_upload_pipeline(client, compute, *, stop, max_jobs, retryable):
    from .client import _join_owned_task

    completed = 0
    tasks: set[asyncio.Task] = set()
    pending = None
    pending_slot = None
    retry_delay = client.retry_initial_seconds

    async def consume(task):
        nonlocal completed
        try:
            worked = await task
            completed += int(worked)
            return worked
        finally:
            tasks.discard(task)

    async def settle():
        # run_once joins any blocking CUDA/CPU owner before cancellation ends.
        # Never park the compute session while another job still owns the GPU.
        nonlocal completed
        outstanding = list(tasks)
        for task in outstanding:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(*outstanding, return_exceptions=True)
        completed += sum(result is True for result in results)
        tasks.clear()

    async def pause(delay):
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    try:
        while not stop.is_set() and (max_jobs is None or completed < max_jobs):
            waiter = None
            try:
                if pending is not None and pending.done():
                    await consume(pending)
                    pending = pending_slot = None
                if max_jobs is not None and completed >= max_jobs:
                    break
                slot = UploadSlot()
                current = asyncio.create_task(client.run_once(
                    compute, _upload_slot=slot,
                    _pending_upload=pending_slot.lease if pending_slot is not None else None,
                ))
                tasks.add(current)
                waiter = asyncio.create_task(slot.ready.wait())
                while not current.done() and not slot.ready.is_set():
                    waiting = {current, waiter}
                    if pending is not None:
                        waiting.add(pending)
                    await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                    if pending is not None and pending.done():
                        await consume(pending)
                        pending = pending_slot = None
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
                waiter = None
                if slot.ready.is_set() and not current.done():
                    # The next GPU row may finish before the previous upload.
                    # Backpressure happens here: only one transport can run.
                    if pending is not None:
                        await consume(pending)
                    slot.transport_allowed.set()
                    pending, pending_slot = current, slot
                    if max_jobs is not None and completed + 1 >= max_jobs:
                        await consume(pending)
                        pending = pending_slot = None
                    retry_delay = client.retry_initial_seconds
                    continue
                worked = await consume(current)
                if worked:
                    retry_delay = client.retry_initial_seconds
                    continue
                if pending is not None:
                    await consume(pending)
                    pending = pending_slot = None
                await client.finish_session()
                if max_jobs is None or completed < max_jobs:
                    await pause(client.poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await settle()
                pending = pending_slot = None
                await client.finish_session()
                if not retryable(exc):
                    raise
                await pause(retry_delay)
                retry_delay = min(client.retry_max_seconds, retry_delay * 2)
            finally:
                if waiter is not None:
                    waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)
        if pending is not None:
            await consume(pending)
    finally:
        try:
            await _join_owned_task(asyncio.create_task(settle()))
        finally:
            await _join_owned_task(asyncio.create_task(client.finish_session()))
    return completed
