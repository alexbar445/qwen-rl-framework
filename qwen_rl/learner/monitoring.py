"""Small CPU-only views for foreground phases and observed run commits."""
from __future__ import annotations

import time
from typing import Callable, Mapping, Any


class ForegroundActivity:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.state = "idle"
        self.started = clock()
        self.seconds: dict[str, float] = {}

    def transition(self, state: str) -> None:
        if state == self.state:
            return
        now = self.clock()
        self.seconds[self.state] = self.seconds.get(self.state, 0.0) + max(0.0, now - self.started)
        self.state, self.started = state, now

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(0.0, self.clock() - self.started)
        return {"state": self.state, "since_elapsed_seconds": elapsed,
                "scope": "current_process_foreground_wall_time",
                "phase_seconds": {**self.seconds,
                                  self.state: self.seconds.get(self.state, 0.0) + elapsed}}


_COMMIT_FIELDS = {
    "updates": "committed_updates",
    "input_tokens": "committed_input_tokens",
    "policy_tokens": "committed_policy_tokens",
    "trajectories": "committed_trajectories",
}


def process_commit_view(totals: Mapping[str, Any]) -> dict[str, Any]:
    return {"scope": "current_process", "telemetry_epoch": totals.get("telemetry_epoch"),
            "elapsed_seconds": totals.get("uptime_seconds"),
            **{name: totals.get(key) for name, key in _COMMIT_FIELDS.items()}}


class RunCommitView:
    """Subtract a real process counter baseline; never invent pre-resume tokens."""
    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.baselines: dict[str, dict[str, Any]] = {}

    def begin(self, run_id: str, totals: Mapping[str, Any], *, prior_updates: int = 0) -> None:
        self.baselines[run_id] = {"started": self.clock(), "totals": dict(totals),
                                 "prior_updates": prior_updates}

    def finish(self, run_id: str) -> None:
        base = self.baselines.get(run_id)
        if base is not None and "finished" not in base:
            base["finished"] = self.clock()

    def snapshot(self, run: Mapping[str, Any], totals: Mapping[str, Any]) -> dict[str, Any]:
        run_id = run.get("run_id")
        base = self.baselines.get(run_id)
        view = {"scope": "current_process_run", "run_id": run_id,
                "telemetry_epoch": totals.get("telemetry_epoch"),
                "logical_completed_updates": run.get("completed_updates"),
                "prior_resumed_token_totals": "unknown"}
        if base is None or base["totals"].get("telemetry_epoch") != totals.get("telemetry_epoch"):
            return {**view, "elapsed_seconds": None,
                    **{name: None for name in _COMMIT_FIELDS}}
        elapsed = max(0.0, base.get("finished", self.clock()) - base["started"])
        if run.get("status") != "running" and "finished" not in base:
            elapsed = None
        view.update(elapsed_seconds=elapsed, prior_completed_updates=base["prior_updates"])
        for name, key in _COMMIT_FIELDS.items():
            current, initial = totals.get(key), base["totals"].get(key)
            view[name] = (current - initial if type(current) in (int, float)
                          and type(initial) in (int, float) and current >= initial else None)
        return view
