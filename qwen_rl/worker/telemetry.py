"""Durable, append-only Worker observations independent of console log levels."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


class TelemetryStore:
    """One Worker owns two JSONL streams; receipts wait for a durable batch."""

    _STREAMS = ("requests", "scheduler")

    def __init__(self, directory: Path, *, worker_id: str, telemetry_epoch: str) -> None:
        self.directory = directory
        self.worker_id = worker_id
        self.telemetry_epoch = telemetry_epoch
        self._lock = threading.Lock()
        self._pending: dict[str, deque[tuple[dict[str, Any], asyncio.Future[None]]]] = {
            stream: deque() for stream in self._STREAMS
        }
        self._flush_tasks: dict[str, asyncio.Task[None]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def _encode(self, record: dict[str, Any]) -> bytes:
        payload = {
            "schema": "qwen-rl-worker-telemetry/v1",
            "worker_id": self.worker_id,
            "telemetry_epoch": self.telemetry_epoch,
            "recorded_at": time.time(),
            **record,
        }
        return (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")

    def _append(self, stream: str, record: dict[str, Any]) -> None:
        """Synchronous single-record entry point retained for callers and tests."""
        self._append_many(stream, (record,))

    def _recover_partial_tail(self, path: Path) -> None:
        """Preserve an interrupted final record before restoring the JSONL prefix."""
        if not path.exists() or not path.stat().st_size:
            return
        with path.open("rb+") as source:
            source.seek(-1, os.SEEK_END)
            if source.read(1) == b"\n":
                return
            cursor = source.tell()
            boundary = 0
            while cursor:
                start = max(0, cursor - 65536)
                source.seek(start)
                newline = source.read(cursor - start).rfind(b"\n")
                if newline >= 0:
                    boundary = start + newline + 1
                    break
                cursor = start
            source.seek(boundary)
            with tempfile.NamedTemporaryFile(
                dir=self.directory, prefix=f"{path.name}.", suffix=".partial", delete=False,
            ) as backup:
                shutil.copyfileobj(source, backup)
                backup.flush()
                os.fsync(backup.fileno())
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            source.truncate(boundary)
            source.flush()
            os.fsync(source.fileno())

    def _append_many(
        self, stream: str, records: tuple[dict[str, Any], ...] | list[dict[str, Any]]
    ) -> None:
        if not records:
            return
        data = b"".join(self._encode(record) for record in records)
        with self._lock:
            missing = []
            ancestor = self.directory
            while not ancestor.exists():
                missing.append(ancestor)
                ancestor = ancestor.parent
            self.directory.mkdir(parents=True, exist_ok=True)
            for created in reversed(missing):
                descriptor = os.open(created.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)

            path = self.directory / f"{stream}.jsonl"
            self._recover_partial_tail(path)
            with path.open("ab") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
            # Keep directory-entry durability for every batch. This is required
            # for the first file creation and is conservative if another process
            # removes or replaces the entry while this Worker is running.
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    @staticmethod
    def _consume_future_exception(future: asyncio.Future[None]) -> None:
        # A cancelled caller still waits for its write, but this also prevents an
        # unobserved-future warning if the event loop is tearing down concurrently.
        if not future.cancelled():
            future.exception()

    @staticmethod
    def _resolve_batch(
        batch: list[tuple[dict[str, Any], asyncio.Future[None]]], error: BaseException | None = None
    ) -> None:
        for _record, receipt in batch:
            if receipt.done():
                continue
            if error is None:
                receipt.set_result(None)
            else:
                receipt.set_exception(error)

    def _schedule_flush(self, stream: str, loop: asyncio.AbstractEventLoop) -> None:
        task = self._flush_tasks.get(stream)
        if task is not None and not task.done():
            return
        task = loop.create_task(self._flush_stream(stream))
        self._flush_tasks[stream] = task

    async def _flush_stream(self, stream: str) -> None:
        queue = self._pending[stream]
        current_task = asyncio.current_task()
        try:
            while True:
                # One event-loop yield collects calls already ready in this turn;
                # no fixed delay is introduced on the Worker request path.
                await asyncio.sleep(0)
                if not queue:
                    return
                batch = list(queue)
                queue.clear()
                records = [record for record, _receipt in batch]

                write_task = asyncio.create_task(asyncio.to_thread(self._append_many, stream, records))
                try:
                    await asyncio.shield(write_task)
                except asyncio.CancelledError:
                    # Do not let cancellation detach an fsync still running in a
                    # thread. Receipts resolve only after that real write finishes.
                    try:
                        await write_task
                    except BaseException as exc:  # noqa: BLE001 - settle receipts after the real write ends
                        self._resolve_batch(batch, exc)
                    else:
                        self._resolve_batch(batch)
                    raise
                except Exception as exc:  # noqa: BLE001 - return serialization and I/O failures to every caller
                    # A receipt represents a complete durable batch. No member
                    # can be acknowledged after any part of its batch failed.
                    self._resolve_batch(batch, exc)
                except BaseException as exc:
                    self._resolve_batch(batch, exc)
                    raise
                else:
                    self._resolve_batch(batch)
        except asyncio.CancelledError:
            # The Worker never cancels flush tasks during normal shutdown. If the
            # event loop is forcibly tearing down, fail any queued receipts rather
            # than strand them or leave a new detached task behind.
            self._resolve_batch(list(queue), RuntimeError("telemetry flush cancelled before write"))
            queue.clear()
            raise
        except BaseException as exc:
            self._resolve_batch(list(queue), exc)
            queue.clear()
            raise
        finally:
            if self._flush_tasks.get(stream) is current_task:
                self._flush_tasks.pop(stream, None)

    async def append(self, stream: str, record: dict[str, Any]) -> None:
        if stream not in self._STREAMS:
            raise ValueError("unknown telemetry stream")
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("TelemetryStore.append must use one Worker event loop")

        receipt: asyncio.Future[None] = loop.create_future()
        receipt.add_done_callback(self._consume_future_exception)
        self._pending[stream].append((record, receipt))
        self._schedule_flush(stream, loop)
        try:
            await asyncio.shield(receipt)
        except asyncio.CancelledError:
            # Cancellation cannot acknowledge or abandon an unfinished write.
            await asyncio.shield(receipt)
            raise

    async def drain(self) -> None:
        """Wait for all scheduled stream flushes; used during Worker shutdown."""
        while self._flush_tasks:
            tasks = tuple(self._flush_tasks.values())
            if not tasks:
                return
            await asyncio.gather(*(asyncio.shield(task) for task in tasks))
