"""Process-local gradient storage; never spills to a filesystem."""
from __future__ import annotations

import io
import threading
import uuid


class GradientMemoryPressure(RuntimeError):
    """The caller must retry without treating its gradient as accepted."""


class MemoryPayloads:
    def __init__(self, limit_bytes: int | None = None):
        if limit_bytes is not None and (type(limit_bytes) is not int or limit_bytes <= 0):
            raise ValueError("max_memory_bytes must be None or a positive integer")
        self.limit_bytes = limit_bytes
        self.used_bytes = 0
        self._payloads: dict[str, object] = {}
        self._lock = threading.RLock()

    def reserve(self, amount: int) -> None:
        with self._lock:
            if self.limit_bytes is not None and self.used_bytes + amount > self.limit_bytes:
                raise GradientMemoryPressure("gradient RAM budget exhausted; retry after payload release")
            self.used_bytes += amount

    def release(self, amount: int) -> None:
        with self._lock:
            self.used_bytes -= amount

    def put(self, raw) -> str:
        with self._lock:
            self.reserve(len(raw))
            try:
                key = "ram:" + uuid.uuid4().hex
                self._payloads[key] = raw
            except BaseException:
                self.release(len(raw))
                raise
            return key

    def get(self, key: str) -> bytes:
        with self._lock:
            return self._payloads[key]

    def contains(self, key: str | None) -> bool:
        with self._lock:
            return key in self._payloads

    def drop(self, key: str) -> None:
        with self._lock:
            value = self._payloads.pop(key, None)
            if value is not None:
                self.release(len(value))
                close = getattr(value, "close", None)
                if close is not None:
                    close()

    def clear(self) -> None:
        with self._lock:
            for key in list(self._payloads):
                self.drop(key)

    def status(self) -> dict[str, int | None]:
        with self._lock:
            return {"used_bytes": self.used_bytes, "limit_bytes": self.limit_bytes,
                    "payloads": len(self._payloads)}

    def stream(self):
        return _MemoryUpload(self)


class _MemoryUpload(io.BytesIO):
    def __init__(self, owner: MemoryPayloads):
        super().__init__()
        self.owner = owner
        self.reserved = 0

    def write(self, data: bytes) -> int:
        self.owner.reserve(len(data))
        try:
            result = super().write(data)
        except BaseException:
            self.owner.release(len(data))
            raise
        self.reserved += len(data)
        return result

    def close(self) -> None:
        if not self.closed:
            super().close()
            self.owner.release(self.reserved)
            self.reserved = 0
