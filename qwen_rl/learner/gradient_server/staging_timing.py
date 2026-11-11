"""Disjoint host-wall-time accounting for pre-optimizer gradient staging."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import time


class StagingTiming:
    def __init__(self):
        self.started_ns = time.perf_counter_ns()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.durations_ns = defaultdict(int)
        self.calls = defaultdict(int)
        self.bytes = defaultdict(int)

    @contextmanager
    def measure(self, name):
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.durations_ns[name] += time.perf_counter_ns() - started
            self.calls[name] += 1

    def snapshot(self, *, status, completed_groups):
        elapsed_ns = time.perf_counter_ns() - self.started_ns
        phases = {name: {
            "seconds": duration / 1e9,
            "calls": self.calls[name],
            "bytes": self.bytes.get(name, 0),
        } for name, duration in sorted(self.durations_ns.items())}
        return {
            "schema": "gradient-staged-materialization-timing/v1",
            "clock": "host_perf_counter_ns",
            "started_at_utc": self.started_utc,
            "elapsed_seconds": elapsed_ns / 1e9,
            "unattributed_seconds": max(0, elapsed_ns - sum(self.durations_ns.values())) / 1e9,
            "status": status,
            "completed_groups": completed_groups,
            "phases": phases,
            "scope": "Disjoint host stages; compute_park_wait waits for proven Learner park, not Worker restart. Worker return is separately fenced before rollout publication. Accumulator writes include CPU add/multiply arithmetic.",
        }


def measured(timing, name):
    return nullcontext() if timing is None else timing.measure(name)
