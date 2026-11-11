"""Read-only rolling rates and sampled worker-activity summaries."""
from __future__ import annotations

from collections import deque
import math
from typing import Any


COMMITTED_COUNTERS = ("input_tokens", "policy_tokens", "trajectories", "updates")
WORKER_ACTIVITY_STATES = (
    "inference_inflight",
    "queued",
    "between_generations",
    "not_ready",
    "unknown",
)


def _valid_counter(value: Any) -> bool:
    return type(value) is int and value >= 0


class RollingCounterWindow:
    """Rates for cumulative counters, with explicit partial-window coverage.

    An epoch or counter reset starts a new series. Missing observations do not
    become zeroes; a later fresh sample measures over the actual elapsed time.
    A rate is unavailable when there are fewer than two fresh samples inside
    the requested window.
    """

    def __init__(self, window_seconds: float = 300.0):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        self.window_seconds = float(window_seconds)
        self._series: dict[str, tuple[Any, deque[tuple[float, dict[str, int]]]]] = {}

    def clear(self, scope: str | None = None) -> None:
        if scope is None:
            self._series.clear()
        else:
            self._series.pop(scope, None)

    def observe(
        self,
        scope: str,
        *,
        epoch: Any,
        counters: dict[str, Any],
        now: float,
        fresh: bool = True,
    ) -> dict[str, Any]:
        if not scope:
            raise ValueError("counter scope must be non-empty")
        if not math.isfinite(now):
            raise ValueError("sample time must be finite")
        if not fresh:
            return self.snapshot(scope, now=now, fresh=False, reason="stale")
        if epoch is None or not isinstance(counters, dict):
            return self.snapshot(scope, now=now, fresh=False, reason="counter_unavailable")
        values = {name: counters.get(name) for name in COMMITTED_COUNTERS}
        if not all(_valid_counter(value) for value in values.values()):
            return self.snapshot(scope, now=now, fresh=False, reason="counter_unavailable")

        prior_epoch, points = self._series.get(scope, (epoch, deque()))
        if (prior_epoch != epoch or (points and now <= points[-1][0])
                or (points and any(values[k] < points[-1][1][k] for k in COMMITTED_COUNTERS))):
            points = deque()
        points.append((now, values))
        cutoff = now - self.window_seconds
        while len(points) > 2 and points[1][0] < cutoff:
            points.popleft()
        self._series[scope] = (epoch, points)
        return self.snapshot(scope, now=now, fresh=True)

    def snapshot(self, scope: str, *, now: float, fresh: bool,
                 reason: str | None = None) -> dict[str, Any]:
        item = self._series.get(scope)
        if item is None:
            return self._empty(fresh=fresh, reason=reason)
        _epoch, points = item
        if not points:
            return self._empty(fresh=fresh, reason=reason)
        latest_at, latest = points[-1]
        cutoff = now - self.window_seconds
        candidates = [point for point in points if point[0] >= cutoff]
        # If the client has only one recent sample, retain the baseline as an
        # unavailable partial window rather than claiming a zero rate.
        selected = candidates[0] if candidates else None
        if selected is None or len(points) < 2 or not fresh:
            return {
                "window_seconds": self.window_seconds,
                "observed_seconds": 0.0 if selected is None else max(0.0, now - selected[0]),
                "coverage_fraction": 0.0,
                "available": False,
                "fresh": fresh,
                "unavailable_reason": reason or ("baseline" if fresh else "stale"),
                "last_sample_age_seconds": max(0.0, now - latest_at),
                "rates_per_hour": {name: None for name in COMMITTED_COUNTERS},
                "deltas": {name: None for name in COMMITTED_COUNTERS},
            }
        first_at, first = selected
        elapsed = now - first_at
        if elapsed <= 0:
            return self._empty(fresh=fresh, reason=reason)
        # Use the latest counter sample as the numerator endpoint. The `now`
        # denominator includes the actual sampling interval since that sample.
        deltas = {name: latest[name] - first[name] for name in COMMITTED_COUNTERS}
        return {
            "window_seconds": self.window_seconds,
            "observed_seconds": elapsed,
            "coverage_fraction": min(1.0, elapsed / self.window_seconds),
            "available": True,
            "fresh": fresh,
            "unavailable_reason": None,
            "last_sample_age_seconds": max(0.0, now - latest_at),
            "rates_per_hour": {name: deltas[name] * 3600.0 / elapsed for name in COMMITTED_COUNTERS},
            "deltas": deltas,
        }

    def _empty(self, *, fresh: bool, reason: str | None = None) -> dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "observed_seconds": 0.0,
            "coverage_fraction": 0.0,
            "available": False,
            "fresh": fresh,
            "unavailable_reason": reason or ("baseline" if fresh else "stale"),
            "last_sample_age_seconds": None,
            "rates_per_hour": {name: None for name in COMMITTED_COUNTERS},
            "deltas": {name: None for name in COMMITTED_COUNTERS},
        }


def classify_worker_activity(worker: dict[str, Any], producer_reason: str | None) -> str:
    """Classify one fresh Worker sample using mutually exclusive evidence."""
    if worker.get("live") is not True:
        return "unknown"
    if worker.get("ready") is not True or worker.get("policy_ready") is False:
        return "not_ready"
    active = worker.get("active")
    queued = worker.get("queued")
    episode_refs = worker.get("episode_refs")
    if not all(_valid_counter(value) for value in (active, queued, episode_refs)):
        return "unknown"
    if active > 0:
        return "inference_inflight"
    if queued > 0:
        return "queued"
    if episode_refs > 0:
        # An episode lease may be waiting between tool/generation calls; it is
        # not sufficient evidence that every such reference is in a tool wait.
        return "between_generations"
    if isinstance(producer_reason, str) and producer_reason:
        return "wait_dispatch:" + producer_reason
    return "unknown"


class SampledActivityWindow:
    """Time-weighted activity categories from sampled state, not GPU idle time."""

    def __init__(self, window_seconds: float = 300.0, max_observation_gap_seconds: float = 8.0,
                 basis: str = "sampled_worker_walltime_not_gpu_utilization"):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        if not math.isfinite(max_observation_gap_seconds) or max_observation_gap_seconds <= 0:
            raise ValueError("max_observation_gap_seconds must be positive and finite")
        self.window_seconds = float(window_seconds)
        self.max_observation_gap_seconds = float(max_observation_gap_seconds)
        self.basis = basis
        self._series: dict[str, deque[tuple[float, str]]] = {}
        self._epochs: dict[str, Any] = {}

    def clear(self) -> None:
        self._series.clear()
        self._epochs.clear()

    def observe(self, key: str, state: str, *, now: float, epoch: Any = None) -> None:
        if not key:
            raise ValueError("activity key must be non-empty")
        if not math.isfinite(now):
            raise ValueError("sample time must be finite")
        if not isinstance(state, str) or not state:
            state = "unknown"
        points = self._series.setdefault(key, deque())
        prior_epoch = self._epochs.get(key)
        if epoch is not None and prior_epoch is not None and epoch != prior_epoch:
            points.clear()
        if epoch is not None:
            self._epochs[key] = epoch
        if points and now <= points[-1][0]:
            points.clear()
        if points and now - points[-1][0] > self.max_observation_gap_seconds:
            # Carry the last observed state only through a bounded sample hold;
            # the uncovered interval is explicitly counted as unknown.
            points.append((points[-1][0] + self.max_observation_gap_seconds, "unknown"))
        points.append((now, state))
        cutoff = now - self.window_seconds
        while len(points) > 2 and points[1][0] <= cutoff:
            points.popleft()

    def summary(self, *, now: float) -> dict[str, Any]:
        if not math.isfinite(now):
            raise ValueError("summary time must be finite")
        per_worker: dict[str, Any] = {}
        aggregate: dict[str, float] = {}
        expected_seconds = 0.0
        observed_seconds = 0.0
        for key, points in self._series.items():
            item = self._summarize_points(points, now=now)
            per_worker[key] = item
            expected_seconds += self.window_seconds
            observed_seconds += item["observed_seconds"]
            for state, seconds in item["state_seconds"].items():
                aggregate[state] = aggregate.get(state, 0.0) + seconds
        known = sum(seconds for state, seconds in aggregate.items() if state != "unknown")
        unknown = aggregate.get("unknown", 0.0)
        denominator = observed_seconds
        states = {
            state: {
                "seconds": seconds,
                "percent_of_observed_worker_walltime": (100.0 * seconds / denominator if denominator else None),
            }
            for state, seconds in sorted(aggregate.items())
        }
        return {
            "window_seconds": self.window_seconds,
            "basis": self.basis,
            "expected_wall_seconds": expected_seconds,
            "observed_wall_seconds": observed_seconds,
            "expected_worker_seconds": expected_seconds,
            "observed_worker_seconds": observed_seconds,
            "coverage_fraction": min(1.0, observed_seconds / expected_seconds) if expected_seconds else 0.0,
            "known_worker_seconds": known,
            "unknown_worker_seconds": unknown,
            "known_percent": (100.0 * known / denominator if denominator else None),
            "unknown_percent": (100.0 * unknown / denominator if denominator else None),
            "states": states,
            "workers": per_worker,
        }

    def _summarize_points(self, points: deque[tuple[float, str]], *, now: float) -> dict[str, Any]:
        if not points:
            return {"observed_seconds": 0.0, "state_seconds": {}, "state_percent": {}, "current_state": "unknown"}
        cutoff = now - self.window_seconds
        start = max(cutoff, points[0][0])
        observed = max(0.0, now - start)
        totals: dict[str, float] = {}
        for index, (at, state) in enumerate(points):
            left = max(start, at)
            right = min(now, points[index + 1][0] if index + 1 < len(points) else now)
            if right > left:
                totals[state] = totals.get(state, 0.0) + right - left
        percentages = {state: 100.0 * seconds / observed for state, seconds in totals.items()} if observed else {}
        return {
            "observed_seconds": observed,
            "state_seconds": totals,
            "state_percent": percentages,
            "current_state": points[-1][1],
        }


def classify_learner_activity(learner: dict[str, Any], *, live: bool) -> str:
    """Classify foreground work or the exact current stream wait reason."""
    if not live or not isinstance(learner, dict):
        return "unknown"
    training = learner.get("training") or {}
    if not isinstance(training, dict):
        return "unknown"
    active = training.get("active_update")
    progress = training.get("progress") or {}
    activity = training.get("activity") or {}
    if active:
        phase = progress.get("stage")
        if not isinstance(phase, str) or phase in {"", "idle"}:
            phase = activity.get("state")
        if not isinstance(phase, str) or phase in {"", "idle"}:
            phase = training.get("stage")
        return f"compute:{phase}" if isinstance(phase, str) and phase not in {"", "idle"} else "compute:active"
    phase = activity.get("state")
    if phase == "awaiting_groups":
        stream = ((learner.get("sampling") or {}).get("stream") or {})
        reason = (stream.get("learner_backpressure_reason") or stream.get("backpressure_reason")) \
            if isinstance(stream, dict) else None
        return f"wait_groups:{reason}" if isinstance(reason, str) and reason else "unknown"
    if phase == "worker_sync":
        return "wait_sync"
    if isinstance(phase, str) and phase not in {"", "idle"}:
        return f"foreground:{phase}"
    run = learner.get("run") or {}
    if isinstance(run, dict) and run.get("status") == "running":
        stream = ((learner.get("sampling") or {}).get("stream") or {})
        reason = (stream.get("learner_backpressure_reason") or stream.get("backpressure_reason")) \
            if isinstance(stream, dict) else None
        if isinstance(reason, str) and reason:
            return f"wait:{reason}"
        return "unknown"
    return "idle"
