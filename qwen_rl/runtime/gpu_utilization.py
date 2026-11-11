"""Append one-second physical-GPU utilization samples for a local executor."""
from __future__ import annotations

import csv
import io
import json
import math
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


_QUERY = (
    "index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
)


def _number(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_gpu_rows(output: str) -> list[dict[str, Any]]:
    """Parse the stable CSV form returned by ``nvidia-smi --query-gpu``."""
    rows = []
    for values in csv.reader(io.StringIO(output)):
        if len(values) != 8:
            continue
        try:
            index = int(values[0].strip())
        except ValueError:
            continue
        rows.append({
            "index": index,
            "uuid": values[1].strip(),
            "name": values[2].strip(),
            "gpu_utilization_percent": _number(values[3]),
            "memory_utilization_percent": _number(values[4]),
            "memory_used_mib": _number(values[5]),
            "memory_total_mib": _number(values[6]),
            "power_draw_w": _number(values[7]),
        })
    return rows


def _same_uuid(left: str, right: str) -> bool:
    left = left.strip().casefold().removeprefix("gpu-")
    right = right.strip().casefold().removeprefix("gpu-")
    return left == right


class GpuUtilizationSampler:
    """Sample one selected GPU without touching its CUDA context or streams."""

    def __init__(
        self,
        output_path: str | Path,
        gpu_uuid: str,
        *,
        context_provider: Callable[[], Mapping[str, Any]] | None = None,
        runner: Callable[..., Any] = subprocess.run,
        wall_time: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        interval_seconds: float = 1.0,
        start_thread: bool = True,
    ) -> None:
        if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
            raise ValueError("GPU utilization sampling requires a physical GPU UUID")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("GPU utilization sample interval must be finite and positive")
        self.path = Path(output_path)
        self.gpu_uuid = gpu_uuid.strip()
        self._context_provider = context_provider
        self._runner = runner
        self._wall_time = wall_time
        self._monotonic = monotonic
        self._interval = float(interval_seconds)
        self._last_sampled_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if start_thread:
            self._thread = threading.Thread(
                target=self._run,
                name="qwen-gpu-utilization",
                daemon=True,
            )
            self._thread.start()

    def _read_selected_gpu(self) -> dict[str, Any]:
        result = self._runner(
            [
                "nvidia-smi",
                f"--query-gpu={_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
        for row in parse_gpu_rows(result.stdout):
            if _same_uuid(row["uuid"], self.gpu_uuid):
                return {"status": "ok", **row}
        return {
            "status": "unavailable",
            "error_type": "SelectedGpuNotVisible",
            "uuid": self.gpu_uuid,
        }

    def sample_once(self) -> dict[str, Any]:
        try:
            gpu = self._read_selected_gpu()
        except Exception as exc:  # telemetry must not affect training
            gpu = {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "uuid": self.gpu_uuid,
            }
        try:
            context = dict(self._context_provider()) if self._context_provider else {}
        except Exception:
            context = {}
        sampled_at = self._wall_time()
        actual_interval = (
            None if self._last_sampled_at is None
            else max(0.0, sampled_at - self._last_sampled_at)
        )
        self._last_sampled_at = sampled_at
        record = {
            "schema": "qwen-rl-gpu-utilization/v1",
            "sampled_at_unix_seconds": sampled_at,
            "sample_interval_seconds": actual_interval,
            "gpu": gpu,
            "executor_context": context,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
        return record

    def _run(self) -> None:
        deadline = self._monotonic()
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception:
                # File or host telemetry failures are not training failures.
                pass
            deadline += self._interval
            now = self._monotonic()
            if deadline <= now:
                missed = int((now - deadline) // self._interval) + 1
                deadline += missed * self._interval
            self._stop.wait(max(0.0, deadline - now))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.5)
