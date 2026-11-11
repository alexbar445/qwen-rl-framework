"""CPU counters and monotonic progress for learner work."""
from __future__ import annotations

import copy
import threading
import time
import uuid
from typing import Any


class TrainingTelemetry:
    """Track attempts, forward work, stage progress, and committed updates.

    The learner app and trainer may each own an instance. App-owned instances
    call ``commit`` to count process/run committed work; trainer-owned instances
    can call ``finish_attempt`` to report progress without claiming those
    external commits.
    """

    def __init__(self):
        self.epoch = uuid.uuid4().hex
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.values = dict(
            forward_input_tokens=0,
            forward_policy_tokens=0,
            loss_forwards=0,
            started_updates=0,
            committed_trajectories=0,
            committed_input_tokens=0,
            committed_policy_tokens=0,
            committed_updates=0,
            committed_compute_seconds=0.0,
        )
        self.active: dict[str, Any] | None = None
        self.last_attempt: dict[str, Any] | None = None
        self.last_update: dict[str, Any] | None = None

    def record_forward(self, input_tokens: int, policy_tokens: int) -> None:
        with self.lock:
            self.values["forward_input_tokens"] += int(input_tokens)
            self.values["forward_policy_tokens"] += int(policy_tokens)
            self.values["loss_forwards"] += 1

    def begin(
        self,
        rows,
        *,
        planned_microbatches: int | None = None,
        update_id: str | None = None,
    ) -> None:
        if planned_microbatches is not None and (
            isinstance(planned_microbatches, bool)
            or not isinstance(planned_microbatches, int)
            or planned_microbatches < 1
        ):
            raise ValueError("planned_microbatches must be a positive integer or None")
        now = time.monotonic()
        input_tokens = sum(len(row["input_ids"]) for row in rows)
        policy_tokens = sum(sum(row["policy_mask"][1:]) for row in rows)
        with self.lock:
            if self.active is not None:
                raise RuntimeError("cannot begin a telemetry update while another update is active")
            self.values["started_updates"] += 1
            self.active = dict(
                started_at=now,
                stage_started_at=now,
                stage="starting",
                current_microbatch=0,
                planned_microbatches=planned_microbatches,
                stage_seconds={},
                update_id=update_id,
                trajectories=len(rows),
                input_tokens=input_tokens,
                policy_tokens=policy_tokens,
            )

    def set_stage(
        self,
        stage: str,
        *,
        current_microbatch: int | None = None,
        planned_microbatches: int | None = None,
    ) -> None:
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a non-empty string")
        with self.lock:
            if self.active is None:
                return
            active = self.active
            if planned_microbatches is not None:
                if (isinstance(planned_microbatches, bool)
                        or not isinstance(planned_microbatches, int)
                        or planned_microbatches < 1):
                    raise ValueError("planned_microbatches must be a positive integer")
                active["planned_microbatches"] = planned_microbatches
            if current_microbatch is not None:
                if (isinstance(current_microbatch, bool)
                        or not isinstance(current_microbatch, int)
                        or current_microbatch < 0):
                    raise ValueError("current_microbatch must be a non-negative integer")
                planned = active["planned_microbatches"]
                if planned is not None and current_microbatch > planned:
                    raise ValueError("current_microbatch cannot exceed planned_microbatches")
                active["current_microbatch"] = current_microbatch
            now = time.monotonic()
            self._close_stage_locked(now)
            active["stage"] = stage
            active["stage_started_at"] = now

    def _close_stage_locked(self, now: float) -> None:
        if self.active is None:
            return
        active = self.active
        elapsed = max(0.0, now - active["stage_started_at"])
        name = active["stage"]
        active["stage_seconds"][name] = active["stage_seconds"].get(name, 0.0) + elapsed

    def _stage_seconds_locked(self, now: float) -> dict[str, float]:
        if self.active is None:
            return {}
        active = self.active
        result = dict(active["stage_seconds"])
        current_elapsed = max(0.0, now - active["stage_started_at"])
        name = active["stage"]
        result[name] = result.get(name, 0.0) + current_elapsed
        return result

    def _finish_locked(
        self,
        *,
        status: str,
        failure_stage: str | None = None,
        error_type: str | None = None,
        policy_version: int | None = None,
        quality: dict[str, Any] | None = None,
        count_as_committed: bool = False,
    ) -> dict[str, Any] | None:
        if self.active is None:
            return None
        now = time.monotonic()
        self._close_stage_locked(now)
        active = self.active
        elapsed = max(now - active["started_at"], 1e-9)
        stage_seconds = dict(active["stage_seconds"])
        item = {
            key: active[key]
            for key in (
                "update_id",
                "trajectories",
                "input_tokens",
                "policy_tokens",
                "planned_microbatches",
                "current_microbatch",
            )
        }
        item.update(
            status=status,
            seconds=elapsed,
            elapsed_seconds=elapsed,
            stage=active["stage"],
            stage_seconds=stage_seconds,
        )
        if failure_stage is not None:
            item["failure_stage"] = failure_stage
        if error_type is not None:
            item["error_type"] = error_type
        if policy_version is not None:
            item["policy_version"] = int(policy_version)
        if quality is not None:
            item["quality"] = copy.deepcopy(quality)

        if count_as_committed:
            item.update(
                input_tokens_per_second=item["input_tokens"] / elapsed,
                policy_tokens_per_second=item["policy_tokens"] / elapsed,
                trajectories_per_second=item["trajectories"] / elapsed,
            )
            self.values["committed_trajectories"] += item["trajectories"]
            self.values["committed_input_tokens"] += item["input_tokens"]
            self.values["committed_policy_tokens"] += item["policy_tokens"]
            self.values["committed_updates"] += 1
            self.values["committed_compute_seconds"] += elapsed
            self.last_update = copy.deepcopy(item)

        self.last_attempt = copy.deepcopy(item)
        self.active = None
        return copy.deepcopy(item)

    def commit(
        self,
        version: int,
        *,
        quality: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Count app-owned work only after its caller confirms the commit."""
        with self.lock:
            return self._finish_locked(
                status="committed",
                policy_version=version,
                quality=quality,
                count_as_committed=True,
            )

    def finish_attempt(
        self,
        *,
        status: str,
        failure_stage: str | None = None,
        error_type: str | None = None,
        policy_version: int | None = None,
        quality: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Close a trainer attempt without changing app-owned commit totals."""
        if status not in {"completed", "committed", "failed", "aborted"}:
            raise ValueError(f"unsupported telemetry attempt status: {status!r}")
        with self.lock:
            return self._finish_locked(
                status=status,
                failure_stage=failure_stage,
                error_type=error_type,
                policy_version=policy_version,
                quality=quality,
            )

    def end(self) -> None:
        """Fail-close an app-owned active attempt when its caller exits early."""
        with self.lock:
            failure_stage = None if self.active is None else self.active["stage"]
            self._finish_locked(status="failed", failure_stage=failure_stage)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.monotonic()
            committed = {
                "trajectories": self.values["committed_trajectories"],
                "input_tokens": self.values["committed_input_tokens"],
                "policy_tokens": self.values["committed_policy_tokens"],
                "updates": self.values["committed_updates"],
                "compute_seconds": self.values["committed_compute_seconds"],
            }
            seconds = committed["compute_seconds"]
            if seconds > 0.0:
                committed.update({
                    "trajectories_per_second": committed["trajectories"] / seconds,
                    "input_tokens_per_second": committed["input_tokens"] / seconds,
                    "policy_tokens_per_second": committed["policy_tokens"] / seconds,
                    "updates_per_second": committed["updates"] / seconds,
                })
            if self.active is None:
                progress = {
                    "stage": None,
                    "current_microbatch": 0,
                    "planned_microbatches": None,
                    "stage_elapsed_seconds": 0.0,
                    "stage_seconds": {},
                }
                active = None
            else:
                current_elapsed = max(0.0, now - self.active["stage_started_at"])
                stage_seconds = self._stage_seconds_locked(now)
                progress = {
                    "stage": self.active["stage"],
                    "current_microbatch": self.active["current_microbatch"],
                    "planned_microbatches": self.active["planned_microbatches"],
                    "stage_elapsed_seconds": current_elapsed,
                    "stage_seconds": stage_seconds,
                }
                active = {
                    key: self.active[key]
                    for key in (
                        "update_id",
                        "stage",
                        "current_microbatch",
                        "planned_microbatches",
                        "trajectories",
                        "input_tokens",
                        "policy_tokens",
                    )
                }
                active.update(
                    elapsed_seconds=max(0.0, now - self.active["started_at"]),
                    stage_elapsed_seconds=current_elapsed,
                    stage_seconds=stage_seconds,
                )
            return {
                "telemetry_epoch": self.epoch,
                "uptime_seconds": max(0.0, now - self.started),
                **self.values,
                "committed": committed,
                "progress": progress,
                "active_update": active,
                "last_attempt": None if self.last_attempt is None else copy.deepcopy(self.last_attempt),
                "last_update": None if self.last_update is None else copy.deepcopy(self.last_update),
            }
