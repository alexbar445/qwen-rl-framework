"""Join one group's members without letting a failed member strand its siblings."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from ..worker.errors import WorkerUnavailableError, worker_unavailable


async def collect_group_members(
    members: Iterable[Awaitable[Any]],
    *,
    on_unavailable: Callable[[WorkerUnavailableError], Awaitable[None]],
) -> list[Any]:
    """Fail only this group, quarantine unavailable Workers, and join its cleanup.

    Keep the original member order and exceptions so the caller still treats
    malformed policy/sampling evidence as fatal. Other groups are never cancelled.
    """
    tasks = [asyncio.ensure_future(member) for member in members]
    pending = set(tasks)
    unavailable = set()
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            failed = False
            for task in done:
                if task.cancelled():
                    failed = True
                    continue
                error = task.exception()
                if error is None:
                    continue
                failed = True
                failure = worker_unavailable(error)
                if failure is not None and failure.worker_id not in unavailable:
                    unavailable.add(failure.worker_id)
                    await on_unavailable(failure)
            if failed:
                for task in pending:
                    task.cancel()
                break
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Cancellation cleanup can itself discover a lost remote lease.
        for result in results:
            if isinstance(result, BaseException):
                failure = worker_unavailable(result)
                if failure is not None and failure.worker_id not in unavailable:
                    unavailable.add(failure.worker_id)
                    await on_unavailable(failure)
        return results
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
