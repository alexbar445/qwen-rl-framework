"""Watch retained idle pinned backing until reuse, pressure reclaim or exit."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class IdlePinnedCacheMonitor:
    def __init__(self, reclaim: Callable[..., dict[str, Any]], *, interval: float = 2.0):
        if interval <= 0:
            raise ValueError("idle cache check interval must be positive")
        self._reclaim = reclaim
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None
        self.last_report: dict[str, Any] | None = None

    def start(self) -> None:
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="idle-pinned-cache", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.last_report = self._reclaim()
                if not self.last_report.get("cache_policy", {}).get("retained"):
                    return
            except Exception as exc:
                self.error = exc
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                raise RuntimeError("idle pinned cache monitor did not stop before GPU wake")
            self._thread = None
        if self.error is not None:
            raise RuntimeError("idle pinned cache maintenance failed") from self.error

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self._reclaim(retain_if_safe=False)
