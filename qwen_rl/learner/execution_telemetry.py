"""Per-executor CUDA-event timings for the latest training microbatches."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

from qwen_rl.runtime.manifest import process_starttime, write_json_atomic
from qwen_rl.runtime.gpu_utilization import GpuUtilizationSampler


_SCHEMA = "qwen-rl-operation-telemetry/v1"
_PHASES = ("forward", "backward")
_MAX_PENDING_EVENTS = 32


def _process_identity() -> dict[str, Any]:
    pid = os.getpid()
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        boot_id = None
    return {
        "pid": pid,
        "process_start_time_ticks": process_starttime(pid),
        "boot_id": boot_id or None,
    }


def _cuda_event_factory(_device: Any):
    import torch

    return torch.cuda.Event(enable_timing=True)


def _cuda_stream_provider(device: Any):
    import torch

    return torch.cuda.current_stream(device=device)


class OperationTelemetry:
    """Poll CUDA events without synchronizing the training stream.

    The writer retains only one latest completed microbatch rate per phase. A
    background thread queries event completion and emits snapshots atomically;
    it never waits for CUDA work or performs a device-wide synchronization.
    """

    def __init__(
        self,
        compute_runtime_dir: str | os.PathLike[str],
        *,
        process_identity: Mapping[str, Any] | None = None,
        event_factory: Callable[[Any], Any] | None = None,
        stream_provider: Callable[[Any], Any] | None = None,
        writer: Callable[[Path, dict[str, Any]], None] = write_json_atomic,
        wall_time: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        poll_interval_seconds: float = 0.05,
        publish_interval_seconds: float = 1.0,
        start_thread: bool = True,
        gpu_uuid: str | None = None,
        gpu_sample_thread: bool = True,
    ) -> None:
        if (not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0
                or not math.isfinite(publish_interval_seconds) or publish_interval_seconds <= 0):
            raise ValueError("operation telemetry intervals must be finite and positive")
        self.path = Path(compute_runtime_dir) / "operation-telemetry.json"
        self.process_identity = dict(process_identity or _process_identity())
        self._event_factory = event_factory or _cuda_event_factory
        self._stream_provider = stream_provider or _cuda_stream_provider
        self._writer = writer
        self._wall_time = wall_time
        self._monotonic = monotonic
        self._poll_interval = float(poll_interval_seconds)
        self._publish_interval = float(publish_interval_seconds)
        self._start_thread = bool(start_thread)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._state = "idle"
        self._active_job: dict[str, Any] | None = None
        self._active_phase: dict[str, Any] | None = None
        self._pending: deque[dict[str, Any]] = deque(maxlen=_MAX_PENDING_EVENTS)
        self._latest_completed: dict[str, dict[str, Any] | None] = {
            phase: None for phase in _PHASES
        }
        self._completed_microbatches = {phase: 0 for phase in _PHASES}
        self._next_phase_sequence = 0
        self._sequence = 0
        self._last_write_monotonic: float | None = None
        self._dirty = False
        self._write_snapshot_locked()
        with self._lock:
            self._ensure_thread_locked()
        self._gpu_sampler = None
        if gpu_uuid:
            try:
                self._gpu_sampler = GpuUtilizationSampler(
                    self.path.parent / "gpu-utilization.jsonl",
                    gpu_uuid,
                    context_provider=self._gpu_context,
                    start_thread=gpu_sample_thread,
                )
            except Exception:
                # The per-second hardware sample is optional observability.
                self._gpu_sampler = None

    def _gpu_context(self) -> dict[str, Any]:
        with self._lock:
            job = self._active_job or {}
            phase = self._active_phase or {}
            return {
                "operation_state": self._state,
                "learner_phase": phase.get("name"),
                "job_id": job.get("job_id"),
                "group_id": job.get("group_id"),
                "policy_version": job.get("policy_version"),
            }

    @staticmethod
    def _job_identity(window: Any, job: Any) -> dict[str, Any]:
        fields = (
            "owner_executor_id", "lineage_id", "engine_epoch", "window_id",
            "attempt_id", "job_id", "group_id", "policy_version",
        )
        identity: dict[str, Any] = {}
        for field in fields:
            value = getattr(job, field, None)
            if isinstance(value, str) and value:
                identity[field] = value
            elif type(value) is int and value >= 0:
                identity[field] = value
            else:
                raise ValueError(f"gradient job is missing telemetry identity {field}")
        context = getattr(window, "update_context", None)
        logical_run_id = context.get("logical_run_id") if isinstance(context, Mapping) else None
        identity["logical_run_id"] = (
            logical_run_id if isinstance(logical_run_id, str) and logical_run_id else None
        )
        return identity

    def begin_job(self, window: Any, job: Any) -> bool:
        """Start a fresh visible job identity and discard older active samples."""
        try:
            identity = self._job_identity(window, job)
        except (TypeError, ValueError):
            return False
        with self._lock:
            if self._closed:
                return False
            self._active_job = identity
            self._active_phase = None
            self._pending.clear()
            self._latest_completed = {phase: None for phase in _PHASES}
            self._completed_microbatches = {phase: 0 for phase in _PHASES}
            self._state = "active"
            self._write_snapshot_locked(force=True)
            self._ensure_thread_locked()
        self._wake.set()
        return True

    def end_job(self) -> None:
        """Mark the process idle while retaining this job's completed samples."""
        self.poll_once()
        with self._lock:
            if self._active_phase is not None:
                self._active_phase = None
            self._active_job = None
            self._state = "idle"
            self._write_snapshot_locked(force=True)
        self._wake.set()

    def begin_phase(
        self,
        name: str,
        microbatch_index: int,
        input_tokens: int,
        *,
        device: Any = None,
    ) -> int | None:
        if (name not in _PHASES or type(microbatch_index) is not int or microbatch_index < 0
                or type(input_tokens) is not int or input_tokens < 0):
            return None
        with self._lock:
            if self._closed or self._active_job is None:
                return None
            if self._active_phase is not None:
                # A missing end hook must not corrupt the next phase's identity.
                self._active_phase = None
            try:
                stream = self._stream_provider(device)
                start_event = self._event_factory(device)
                start_event.record(stream=stream)
            except Exception:
                return None
            sequence = self._next_phase_sequence
            self._next_phase_sequence += 1
            now = self._monotonic()
            self._active_phase = {
                "sequence": sequence,
                "name": name,
                "microbatch_index": microbatch_index,
                "input_tokens": input_tokens,
                "device": device,
                "stream": stream,
                "start_event": start_event,
                "phase_started_monotonic": now,
                "phase_started_at_unix_seconds": self._wall_time(),
                "elapsed_wall_seconds": 0.0,
            }
            self._dirty = True
        self._wake.set()
        return sequence

    def end_phase(self, sequence: int | None, *, succeeded: bool) -> None:
        if type(sequence) is not int:
            return
        with self._lock:
            phase = self._active_phase
            if phase is None or phase["sequence"] != sequence:
                return
            try:
                end_event = self._event_factory(phase["device"])
                end_event.record(stream=phase["stream"])
            except Exception:
                self._active_phase = None
                self._dirty = True
            else:
                ended_at = self._wall_time()
                self._pending.append({
                    "name": phase["name"],
                    "microbatch_index": phase["microbatch_index"],
                    "input_tokens": phase["input_tokens"],
                    "start_event": phase["start_event"],
                    "end_event": end_event,
                    "identity": deepcopy(self._active_job),
                    "sequence": phase["sequence"],
                    "succeeded": bool(succeeded),
                    "end_queued_at_unix_seconds": ended_at,
                })
                self._active_phase = None
                self._dirty = True
        self._wake.set()

    def poll_once(self) -> bool:
        """Poll pending CUDA events once; safe for deterministic CPU tests."""
        changed = False
        now = self._monotonic()
        with self._lock:
            remaining: deque[dict[str, Any]] = deque(maxlen=_MAX_PENDING_EVENTS)
            for pending in self._pending:
                try:
                    ready = bool(pending["end_event"].query())
                except Exception:
                    ready = True
                    pending["succeeded"] = False
                if not ready:
                    remaining.append(pending)
                    continue
                elapsed = self._elapsed_seconds(pending["start_event"], pending["end_event"])
                if pending["succeeded"]:
                    phase_name = pending["name"]
                    self._completed_microbatches[phase_name] += 1
                    if elapsed is not None:
                        self._latest_completed[phase_name] = {
                            "microbatch_index": pending["microbatch_index"],
                            "input_tokens": pending["input_tokens"],
                            "cuda_elapsed_seconds": elapsed,
                            "tokens_per_second": self._rate(pending["input_tokens"], elapsed),
                            "completed_at_unix_seconds": self._wall_time(),
                            "end_queued_at_unix_seconds": pending["end_queued_at_unix_seconds"],
                            "sequence": pending["sequence"],
                            "identity": deepcopy(pending["identity"]),
                        }
                    changed = True
            self._pending = remaining

            phase = self._active_phase
            if phase is not None:
                phase["elapsed_wall_seconds"] = max(
                    0.0, now - phase["phase_started_monotonic"],
                )

            write_due = (self._last_write_monotonic is None
                         or now - self._last_write_monotonic >= self._publish_interval)
            if changed or write_due or self._dirty:
                self._write_snapshot_locked(force=True)
        return changed

    @staticmethod
    def _elapsed_seconds(start_event: Any, end_event: Any) -> float | None:
        try:
            elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
        except Exception:
            return None
        return elapsed if math.isfinite(elapsed) and elapsed > 0 else None

    @staticmethod
    def _rate(input_tokens: int, elapsed: float) -> float | None:
        value = input_tokens / elapsed if elapsed > 0 else float("nan")
        return value if math.isfinite(value) and value >= 0 else None

    def _ensure_thread_locked(self) -> None:
        if not self._start_thread or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._poll_loop, name="qwen-operation-telemetry", daemon=True,
        )
        self._thread.start()

    def _poll_loop(self) -> None:
        while not self._closed:
            try:
                self.poll_once()
            except Exception:
                # Monitoring failures must never abort training.
                pass
            with self._lock:
                busy = self._active_job is not None or bool(self._pending)
            self._wake.wait(timeout=self._poll_interval if busy else self._publish_interval)
            self._wake.clear()

    def _write_snapshot_locked(self, *, force: bool = False) -> None:
        now = self._monotonic()
        if (not force and self._last_write_monotonic is not None
                and now - self._last_write_monotonic < self._publish_interval):
            return
        self._sequence += 1
        active_phase = None
        if self._active_phase is not None:
            phase = self._active_phase
            active_phase = {
                "name": phase["name"],
                "microbatch_index": phase["microbatch_index"],
                "input_tokens": phase["input_tokens"],
                "phase_started_at_unix_seconds": phase["phase_started_at_unix_seconds"],
                "elapsed_wall_seconds": phase["elapsed_wall_seconds"],
                "measurement": "wall_clock_phase_elapsed",
            }
        payload = {
            "schema": _SCHEMA,
            "sequence": self._sequence,
            "sampled_at_unix_seconds": self._wall_time(),
            "state": self._state,
            "process_identity": deepcopy(self.process_identity),
            "active_job": deepcopy(self._active_job),
            "active_phase": active_phase,
            "completed_microbatches": dict(self._completed_microbatches),
            "last_completed": deepcopy(self._latest_completed),
        }
        try:
            self._writer(self.path, payload)
        except Exception:
            # The optional dashboard receipt cannot own training success.
            return
        self._last_write_monotonic = now
        self._dirty = False

    def close(self) -> None:
        self.poll_once()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._write_snapshot_locked(force=True)
        self._wake.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        if self._gpu_sampler is not None:
            self._gpu_sampler.close()
            self._gpu_sampler = None
