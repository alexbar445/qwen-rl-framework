"""Caching helpers for the dashboard.

Every HTTP request must answer within a click budget, so the two slow things this dashboard
touches are moved off the request path:

* :class:`LiveWatch` owns one background thread that re-runs the (slow, ~0.8 s) watch command
  on an interval.  ``/api/overview`` then answers from memory.  The value carries
  ``age_seconds`` so the UI can state how old the snapshot is instead of implying freshness.
* :class:`Refreshing` is stale-while-revalidate for cheap-but-not-free reads (run records,
  nvidia-smi, the BEPA ledger): the first call for a key produces inline, later calls return
  the cached value immediately and refresh it in the background once it is older than ``ttl``.

``dir_stats`` bounds the artifact walk: ``arc/`` alone holds >1000 files and the policy
directories are gigabytes, so the count is capped and the truncation is reported to the UI
rather than hidden.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

ARTIFACT_SCAN_LIMIT = 800


class Refreshing:
    """Single-slot stale-while-revalidate cache.

    The caller supplies both the key and the producer per call, so one instance can serve a
    constant key (``overview``, ``gpu``) or a per-request key (``run_id``) without extra
    plumbing.  Only the most recent key is kept, which matches how the UI works: one run,
    one page at a time.
    """

    def __init__(self, ttl: float) -> None:
        self.ttl = float(ttl)
        self._condition = threading.Condition(threading.Lock())
        self._value: Any = None
        self._stamp = 0.0
        self._key: Any = object()
        self._refreshing: set[Any] = set()
        self._loading: set[Any] = set()

    def get(self, key: Any, produce: Callable[[Any], Any]) -> Any:
        with self._condition:
            while key in self._loading:
                self._condition.wait()
                if self._value is not None and key == self._key:
                    return self._value
            if self._value is not None and key == self._key:
                if (time.time() - self._stamp) > self.ttl and key not in self._refreshing:
                    self._refreshing.add(key)
                    threading.Thread(target=self._refresh, args=(key, produce), daemon=True).start()
                return self._value
            self._loading.add(key)
        try:
            value = produce(key)
        except BaseException:
            with self._condition:
                self._loading.discard(key)
                self._condition.notify_all()
            raise
        with self._condition:
            self._value, self._key, self._stamp = value, key, time.time()
            self._loading.discard(key)
            self._condition.notify_all()
        return value

    def invalidate(self) -> None:
        with self._condition:
            self._value = None
            self._key = object()
            self._condition.notify_all()

    def _refresh(self, key: Any, produce: Callable[[Any], Any]) -> None:
        try:
            value = produce(key)
            with self._condition:
                if key == self._key:
                    self._value, self._stamp = value, time.time()
        except Exception:
            pass
        finally:
            with self._condition:
                self._refreshing.discard(key)
                self._condition.notify_all()

class LiveWatch:
    """Keeps one canonical watch snapshot warm in the background."""

    def __init__(self, fetch: Callable[[], dict[str, Any]], interval: float = 4.0,
                 initial_delay: float = 0.05) -> None:
        self._fetch = fetch
        self.interval = float(interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._value: dict[str, Any] = {
            "ok": False, "snapshot": None, "age_seconds": None,
            "error": "首次快照采集尚未完成",
        }
        self._stamp = 0.0
        self._thread = threading.Thread(target=self._loop, args=(initial_delay,),
                                        name="dashboard-live-watch", daemon=True)
        self._thread.start()

    def _loop(self, delay: float) -> None:
        if delay:
            self._stop.wait(delay)
        while not self._stop.is_set():
            try:
                value = self._fetch()
            except Exception as exc:  # never let the refresher die
                value = {"ok": False, "snapshot": None, "error": "%s: %s" % (type(exc).__name__, exc)}
            with self._lock:
                self._value, self._stamp = value, time.time()
            self._kick.wait(self.interval)
            self._kick.clear()

    def get(self) -> dict[str, Any]:
        with self._lock:
            value = dict(self._value)
            value["age_seconds"] = (time.time() - self._stamp) if self._stamp else None
            value["refresh_interval_seconds"] = self.interval
        return value

    def request_refresh(self) -> None:
        """Ask for an immediate re-fetch without making the caller wait for it."""
        self._kick.set()

    def stop(self) -> None:
        self._stop.set()
        self._kick.set()
        self._thread.join(timeout=min(5.0, max(1.0, self.interval + 0.5)))

def dir_stats(base: Path, limit: int = ARTIFACT_SCAN_LIMIT) -> tuple[int, int, bool]:
    """Return bounded file count, byte count, and whether the walk hit its cap."""
    files = 0
    total = 0
    try:
        walker = os.walk(base)
    except OSError:
        return 0, 0, False
    for root, _dirs, names in walker:
        for name in names:
            files += 1
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
            if files >= limit:
                return files, total, True
    return files, total, False
