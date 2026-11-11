"""Serialize model access and drain started thread work before releasing ownership."""
import asyncio


class ComputeOwner:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def run(self, function, *args, **kwargs):
        async with self._lock:
            task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Cancelling an asyncio waiter does not stop a CUDA/optimizer thread.
                # Keep ownership until that thread really exits, even after a second
                # cancellation from a shutting-down parent.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if task.done() and not task.cancelled():
                    task.exception()
                raise
