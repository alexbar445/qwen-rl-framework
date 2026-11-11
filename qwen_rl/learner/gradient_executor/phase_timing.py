"""Small local monotonic phase receipts for gradient executor jobs."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IDENTITY_FIELDS = (
    "executor_id", "job_id", "window_id", "attempt_id", "lineage_id",
    "engine_epoch", "group_id", "policy_version", "lease_epoch",
)

try:
    _BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip() or None
except OSError:
    _BOOT_ID = None

@dataclass(frozen=True)
class _PhaseToken:
    sequence: int
    name: str
    started_ns: int


class JobPhaseTiming:
    """Thread-safe timer whose offsets start at the executor's claim request."""

    def __init__(self) -> None:
        wall_before_ns = time.time_ns()
        self.started_ns = time.monotonic_ns()
        wall_after_ns = time.time_ns()
        self.wall_time_anchor_ns = (wall_before_ns + wall_after_ns) // 2
        self.wall_time_anchor_uncertainty_ns = max(0, wall_after_ns - wall_before_ns)
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._next_sequence = 0
        self._open: set[int] = set()
        self._phases: list[dict[str, Any]] = []

    def start(self, name: str) -> _PhaseToken:
        if not isinstance(name, str) or not name or any(char.isspace() for char in name):
            raise ValueError("phase name must be a non-empty token")
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            token = _PhaseToken(sequence, name, time.monotonic_ns())
            self._open.add(sequence)
            return token

    def finish(self, token: _PhaseToken, *, outcome: str = "ok") -> None:
        if outcome not in {"ok", "error"}:
            raise ValueError("phase outcome must be ok or error")
        finished_ns = time.monotonic_ns()
        with self._lock:
            if token.sequence not in self._open:
                raise RuntimeError("phase token was already finished or belongs to another timer")
            self._open.remove(token.sequence)
            start_offset_ns = max(0, token.started_ns - self.started_ns)
            end_offset_ns = max(start_offset_ns, finished_ns - self.started_ns)
            self._phases.append({
                "sequence": token.sequence,
                "name": token.name,
                "start_offset_ns": start_offset_ns,
                "duration_ns": max(0, finished_ns - token.started_ns),
                "end_offset_ns": end_offset_ns,
                "outcome": outcome,
            })

    def mark(self, name: str, *, outcome: str = "ok") -> None:
        """Record a zero-duration event such as a cache hit without inventing work."""
        if not isinstance(name, str) or not name or any(char.isspace() for char in name):
            raise ValueError("phase name must be a non-empty token")
        if outcome not in {"ok", "error"}:
            raise ValueError("phase outcome must be ok or error")
        event_ns = time.monotonic_ns()
        offset_ns = max(0, event_ns - self.started_ns)
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            self._phases.append({
                "sequence": sequence,
                "name": name,
                "start_offset_ns": offset_ns,
                "duration_ns": 0,
                "end_offset_ns": offset_ns,
                "outcome": outcome,
            })

    @contextmanager
    def measure(self, name: str):
        token = self.start(name)
        try:
            yield
        except BaseException:
            self.finish(token, outcome="error")
            raise
        else:
            self.finish(token)

    @asynccontextmanager
    async def measure_async(self, name: str):
        token = self.start(name)
        try:
            yield
        except BaseException:
            self.finish(token, outcome="error")
            raise
        else:
            self.finish(token)

    def snapshot(self) -> dict[str, Any]:
        """Return completed phase rows with their local monotonic clock anchor."""
        with self._lock:
            phases = sorted((dict(row) for row in self._phases), key=lambda row: row["sequence"])
            open_phases = sorted(self._open)
        return {
            "schema": "gradient_executor_phase_snapshot/v1",
            "time_unit": "ns",
            "clock": {
                "monotonic_start_ns": self.started_ns,
                "wall_time_anchor_ns": self.wall_time_anchor_ns,
                "wall_time_anchor_uncertainty_ns": self.wall_time_anchor_uncertainty_ns,
                "boot_id": _BOOT_ID,
                "pid": self.pid,
            },
            "phases": phases,
            "open_phase_sequences": open_phases,
        }

    def receipt(
        self,
        identity: Mapping[str, Any],
        *,
        status: str,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"accepted", "failed", "fenced", "cancelled"}:
            raise ValueError("invalid phase receipt status")
        safe_identity: dict[str, str | int] = {}
        for field in _IDENTITY_FIELDS:
            value = identity.get(field)
            if ((isinstance(value, str) and value)
                    or (type(value) is int and value >= 0)):
                safe_identity[field] = value
        if not isinstance(safe_identity.get("job_id"), str):
            raise TypeError("phase receipt identity must include a non-empty job_id")
        with self._lock:
            phases = sorted(self._phases, key=lambda item: item["sequence"])
            if self._open:
                raise RuntimeError("cannot persist a phase receipt with open phases")
        result: dict[str, Any] = {
            "schema": "gradient_executor_phase_timing/v1",
            "time_unit": "ns",
            "offset_origin": "lease_claim_request_start",
            "clock": {
                "monotonic_start_ns": self.started_ns,
                "wall_time_anchor_ns": self.wall_time_anchor_ns,
                "wall_time_anchor_uncertainty_ns": self.wall_time_anchor_uncertainty_ns,
                "boot_id": _BOOT_ID,
                "pid": self.pid,
            },
            "identity": safe_identity,
            "status": status,
            "phases": phases,
        }
        if isinstance(error_type, str) and error_type:
            result["error_type"] = error_type.rsplit(".", 1)[-1]
        return result

    def write_receipt(
        self,
        working_dir: str | os.PathLike[str],
        identity: Mapping[str, Any],
        *,
        status: str,
        error_type: str | None = None,
    ) -> Path:
        payload = self.receipt(identity, status=status, error_type=error_type)
        safe_identity = payload["identity"]
        receipt_key = "\0".join(
            str(safe_identity.get(field, ""))
            for field in ("job_id", "attempt_id", "lease_epoch")
        )
        filename = hashlib.sha256(receipt_key.encode("utf-8")).hexdigest() + ".json"
        directory = Path(working_dir) / "phase-timings"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        raw = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=directory,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return path


@asynccontextmanager
async def measured_async(timer: JobPhaseTiming | None, name: str):
    if timer is None:
        yield
    else:
        async with timer.measure_async(name):
            yield


@contextmanager
def measured(timer: JobPhaseTiming | None, name: str):
    if timer is None:
        yield
    else:
        with timer.measure(name):
            yield
