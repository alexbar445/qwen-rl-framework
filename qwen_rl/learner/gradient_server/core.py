"""Central owner for deterministic async LoRA gradient aggregation."""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
import random
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import torch

from .checkpoint_writer import (
    CheckpointWriteError,
    CheckpointWriteResult,
    OwnedCheckpointWriter,
    write_checkpoint_payload,
)
from ...contracts.gradients import (
    GRADIENT_CONTRIBUTION_CONTRACT,
    GRADIENT_CONTRIBUTION_SCHEMA,
    PREPARED_WINDOW_SCHEMA,
    GradientContractError,
    GradientContribution,
    GradientJob,
    GradientJobSpec,
    GradientParameter,
    PreparedWindow,
    compute_profile_fingerprint,
    validate_compute_profile,
)
from ...deployment.instance_relocation import compatibility_matches, normalize_path_mappings

_CHECKPOINT_SCHEMA = "central-gradient-checkpoint/v1"
_POINTER_SCHEMA = "central-gradient-checkpoint-pointer/v1"
_RECEIPT_SCHEMA = "central-gradient-window-receipt/v1"
_TIMING_SCHEMA = "central-gradient-update-timing/v1"
_DDP_INITIALIZATION_SCHEMA = "central-gradient-ddp-initialization/v1"


def _timing_point() -> dict[str, Any]:
    return {
        "monotonic_ns": time.perf_counter_ns(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }


def _timing_event(record: dict[str, Any], name: str, point: Mapping[str, Any] | None = None) -> dict[str, Any]:
    mark = dict(_timing_point() if point is None else point)
    if type(mark.get("monotonic_ns")) is not int or not isinstance(mark.get("utc"), str):
        raise ValueError("timing event requires monotonic_ns and UTC text")
    events = record.setdefault("events", {})
    events[name] = mark
    received = events.get("last_contribution_accepted")
    if isinstance(received, Mapping):
        offsets = record.setdefault("offsets_from_last_contribution", {})
        for event_name, event_mark in events.items():
            offset_ns = event_mark["monotonic_ns"] - received["monotonic_ns"]
            offsets[event_name] = {
                "nanoseconds": offset_ns,
                "milliseconds": round(offset_ns / 1_000_000, 6),
            }
    return mark


def _timing_span(
    record: dict[str, Any],
    name: str,
    start: Mapping[str, Any],
    end: Mapping[str, Any],
) -> None:
    _timing_event(record, f"{name}_start", start)
    _timing_event(record, f"{name}_end", end)
    duration_ns = end["monotonic_ns"] - start["monotonic_ns"]
    record.setdefault("durations", {})[name] = {
        "nanoseconds": duration_ns,
        "milliseconds": round(duration_ns / 1_000_000, 6),
    }


class GradientCheckpointError(RuntimeError):
    """Central optimizer checkpoint is invalid or could not be saved."""


class DDPCounterInitializationError(GradientCheckpointError):
    """The DDP initialization receipt or central counters are inconsistent."""


class DirtyGradientCoreError(RuntimeError):
    """An optimizer/checkpoint boundary is uncertain; updates are fenced."""


class CheckpointPendingError(RuntimeError):
    """A prior optimizer checkpoint is still being durably written."""


def _layout_fingerprint(layout: Sequence[GradientParameter]) -> str:
    value = json.dumps([item.to_dict() for item in layout], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _attempt_key(window_id: str, attempt_id: str) -> str:
    return json.dumps([window_id, attempt_id], separators=(",", ":"))


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with open(temporary, "xb") as stream:
            stream.write((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _capture_rng() -> dict[str, Any]:
    try:
        import numpy as np
        numpy_state = np.random.get_state()
    except ImportError:  # pragma: no cover
        numpy_state = None
    cuda = [item.cpu().clone() for item in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []
    return {
        "schema": "central-gradient-rng/v1",
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch_cpu": torch.random.get_rng_state().cpu().clone(),
        "torch_cuda": cuda,
    }


def _restore_rng(value: Mapping[str, Any]) -> None:
    if value.get("schema") != "central-gradient-rng/v1":
        raise GradientCheckpointError("central RNG state schema is unsupported")
    random.setstate(value["python"])
    if value.get("numpy") is not None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise GradientCheckpointError("NumPy RNG state cannot be restored without NumPy") from exc
        np.random.set_state(value["numpy"])
    torch.random.set_rng_state(value["torch_cpu"].cpu())
    if value.get("torch_cuda"):
        if not torch.cuda.is_initialized():
            raise GradientCheckpointError("checkpoint has CUDA RNG state but CUDA is not initialized")
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _trainable_lora_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named or any("lora_" not in name for name, _ in named):
        raise ValueError("central model must expose LoRA-only trainable parameters")
    if len({name for name, _ in named}) != len(named):
        raise ValueError("duplicate trainable parameter name")
    if any(p.device.type == "meta" or p.is_quantized for _, p in named):
        raise ValueError("central LoRA parameters must be materialized tensors")
    return dict(sorted(named))


def _dtype_name(parameter: torch.nn.Parameter) -> str:
    return str(parameter.dtype).removeprefix("torch.")


class PreparedGradientWindow:
    """Opaque identity for one Core-owned, validated CPU gradient sum."""

    __slots__ = ()


@dataclass(frozen=True)
class _GradientPreparation:
    token: PreparedGradientWindow
    key: str
    fingerprint: str
    gradients: dict[str, torch.Tensor | None]


@dataclass(frozen=True)
class WindowStepReceipt:
    schema: str
    window_id: str
    attempt_id: str
    policy_version_before: int
    policy_version: int
    optimizer_steps: int
    scheduler_steps: int
    n_window_tokens: int
    job_count: int
    gradient_norm_before_clip: float
    max_grad_norm: float
    checkpoint_generation: int

    def __post_init__(self) -> None:
        if self.schema != _RECEIPT_SCHEMA or self.policy_version != self.policy_version_before + 1:
            raise ValueError("invalid window step receipt")
        if not math.isfinite(self.gradient_norm_before_clip) or self.gradient_norm_before_clip < 0:
            raise ValueError("receipt gradient norm must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "window_id": self.window_id, "attempt_id": self.attempt_id,
            "policy_version_before": self.policy_version_before, "policy_version": self.policy_version,
            "optimizer_steps": self.optimizer_steps, "scheduler_steps": self.scheduler_steps,
            "n_window_tokens": self.n_window_tokens, "job_count": self.job_count,
            "gradient_norm_before_clip": self.gradient_norm_before_clip,
            "max_grad_norm": self.max_grad_norm, "checkpoint_generation": self.checkpoint_generation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WindowStepReceipt":
        fields = {
            "schema", "window_id", "attempt_id", "policy_version_before", "policy_version",
            "optimizer_steps", "scheduler_steps", "n_window_tokens", "job_count",
            "gradient_norm_before_clip", "max_grad_norm", "checkpoint_generation",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise GradientCheckpointError("checkpoint window receipt is malformed")
        return cls(**dict(value))


class GradientServerCore:
    """Central AdamW/scheduler owner. Only named trainable LoRA state is saved."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_dir: str | Path,
        *,
        max_grad_norm: float = 1.0,
        policy_version: int | None = None,
        auto_resume: bool = True,
        compatibility: Mapping[str, Any] | None = None,
        compatibility_path_mappings: Mapping[str, str] | None = None,
        lineage_id: str | None = None,
        engine_epoch: str | None = None,
        adapter_state_factory: Callable[[torch.nn.Module], Mapping[str, torch.Tensor]] | None = None,
        before_optimizer_step: Callable[[], None] | None = None,
        async_checkpoint: bool = False,
        preserve_control_files: bool = False,
        snapshot_mmap: bool = False,
    ) -> None:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("central gradient core requires one torch.optim.AdamW owner")
        if scheduler is None or not callable(getattr(scheduler, "step", None)):
            raise ValueError("central gradient core requires one scheduler")
        if isinstance(max_grad_norm, bool) or not isinstance(max_grad_norm, (int, float)):
            raise ValueError("max_grad_norm must be a finite positive number")
        if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0:
            raise ValueError("max_grad_norm must be a finite positive number")
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.root = Path(checkpoint_dir)
        self.latest_path = self.root / "latest.json"
        self.prepared_path = self.root / "prepared-window.json"
        self.pending_path = self.root / "pending-attempt.json"
        self.async_intent_path = self.root / "async-checkpoint-intent.json"
        if type(async_checkpoint) is not bool:
            raise TypeError("async_checkpoint must be a boolean")
        self.async_checkpoint = async_checkpoint
        if type(preserve_control_files) is not bool:
            raise TypeError("preserve_control_files must be a boolean")
        self.preserve_control_files = preserve_control_files
        if type(snapshot_mmap) is not bool:
            raise TypeError("snapshot_mmap must be a boolean")
        self._snapshot_mmap = snapshot_mmap
        self.snapshot_load_metrics: list[dict[str, Any]] = []
        self.max_grad_norm = float(max_grad_norm)
        raw_compatibility = {} if compatibility is None else dict(compatibility)
        try:
            self.compatibility = json.loads(json.dumps(raw_compatibility, sort_keys=True, separators=(",", ":")))
        except (TypeError, ValueError) as exc:
            raise ValueError("compatibility must be JSON-safe metadata") from exc
        try:
            self.compatibility_path_mappings = normalize_path_mappings(compatibility_path_mappings)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"compatibility path mappings are invalid: {exc}") from None
        self._configured_lineage_id = lineage_id
        self._configured_engine_epoch = engine_epoch
        self.lineage_id = uuid4().hex if lineage_id is None else lineage_id
        self.engine_epoch = uuid4().hex if engine_epoch is None else engine_epoch
        if not isinstance(self.lineage_id, str) or re.fullmatch(r"[0-9a-f]{32}", self.lineage_id) is None:
            raise ValueError("lineage_id must be a 32-character lowercase UUID hex string")
        if not isinstance(self.engine_epoch, str) or re.fullmatch(r"[0-9a-f]{32}", self.engine_epoch) is None:
            raise ValueError("engine_epoch must be a 32-character lowercase UUID hex string")
        self.snapshot_id: str | None = None
        self.resumed = False
        self.last_update_context: dict[str, Any] | None = None
        self.checkpoint_path = str(self.root / "latest.json")
        self._adapter_state_factory = adapter_state_factory
        self._before_optimizer_step = before_optimizer_step
        self._lock = threading.RLock()
        self._gradient_preparation: _GradientPreparation | None = None
        self._timing_lock = threading.RLock()
        self._checkpoint_writer = OwnedCheckpointWriter() if async_checkpoint else None
        self._checkpoint_future: Future[CheckpointWriteResult] | None = None
        self._checkpoint_finalized = threading.Event()
        self._checkpoint_finalized.set()
        self._pending_checkpoint_completion: dict[str, Any] | None = None
        self._checkpoint_state = "durable"
        self._checkpoint_intent: dict[str, Any] | None = None
        self._checkpoint_failure: str | None = None
        self._checkpoint_cleanup_error: str | None = None
        self._checkpoint_notification_error: str | None = None
        self._checkpoint_timing_error: str | None = None
        self._staged_callbacks: dict[str, Callable[[Mapping[str, Any]], None]] = {}
        self._async_intent_resolved = False
        self._timing_records: dict[str, dict[str, Any]] = {}
        self._latest_timing_key: str | None = None
        self._latest_timing_identity: tuple[str, str] | None = None
        self._parameters = _trainable_lora_parameters(model)
        self.parameter_layout = tuple(
            GradientParameter(name, tuple(p.shape), _dtype_name(p)) for name, p in self._parameters.items()
        )
        self.layout_fingerprint = _layout_fingerprint(self.parameter_layout)
        self._optimizer_layout = self._read_optimizer_layout()
        self.policy_version = 0 if policy_version is None else self._counter(policy_version, "policy_version")
        self.initial_policy_version = self.policy_version
        self.optimizer_steps = 0
        self.scheduler_steps = 0
        self.durable_policy_version = self.policy_version
        self.durable_optimizer_steps = 0
        self.durable_scheduler_steps = 0
        self._checkpoint_generation = -1
        self._prepared_windows: dict[str, PreparedWindow] = {}
        self._update_contexts: dict[str, dict[str, Any]] = {}
        self._step_callbacks: dict[str, Callable[[], None]] = {}
        self._committed_attempts: dict[str, tuple[str, WindowStepReceipt]] = {}
        self._committed_windows: dict[str, str] = {}
        self._cancelled_windows: dict[str, dict[str, Any]] = {}
        self._pending_attempt: dict[str, Any] | None = None
        self._checkpoint_pending_attempt: dict[str, Any] | None = None
        self._dirty = False
        self._dirty_reason: str | None = None
        self._last_receipt: WindowStepReceipt | None = None
        for parameter in self._parameters.values():
            parameter.grad = None
        if self.root.is_symlink():
            raise GradientCheckpointError("checkpoint directory may not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise GradientCheckpointError("checkpoint path must be a real directory")
        if self.latest_path.exists() or self.latest_path.is_symlink():
            if not auto_resume:
                raise GradientCheckpointError("checkpoint exists and auto_resume is disabled")
            self.resume()
        else:
            if list(self.root.iterdir()):
                raise GradientCheckpointError("checkpoint data exists without latest pointer; legacy import is disabled")
            self._save_checkpoint()  # durable genesis before any optimizer step

    @staticmethod
    def timing_mark() -> dict[str, Any]:
        return _timing_point()

    def _timing_record(self, window_id: str, attempt_id: str) -> dict[str, Any]:
        key = _attempt_key(window_id, attempt_id)
        with self._timing_lock:
            record = self._timing_records.get(key)
            if record is None:
                try:
                    cpu_affinity = sorted(os.sched_getaffinity(0))
                except (AttributeError, OSError):
                    cpu_affinity = None
                record = {
                    "schema": _TIMING_SCHEMA,
                    "runtime": {
                        "torch_num_threads": torch.get_num_threads(),
                        "torch_num_interop_threads": torch.get_num_interop_threads(),
                        "cpu_affinity": cpu_affinity,
                    },
                    "clock": {
                        "monotonic": "time.perf_counter_ns",
                        "wall": "UTC ISO-8601",
                    },
                    "lineage_id": self.lineage_id,
                    "engine_epoch": self.engine_epoch,
                    "window_id": window_id,
                    "attempt_id": attempt_id,
                    "policy_version_before": self.policy_version,
                    "events": {},
                    "durations": {},
                    "offsets_from_last_contribution": {},
                }
                self._timing_records[key] = record
            self._latest_timing_key = key
            self._latest_timing_identity = (window_id, attempt_id)
            return record

    def note_timing_event(
        self,
        window_id: str,
        attempt_id: str,
        event: str,
        point: Mapping[str, Any] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        if not isinstance(event, str) or not event or event.strip() != event:
            raise ValueError("timing event name must be nonempty text")
        record = self._timing_record(window_id, attempt_id)
        with self._timing_lock:
            _timing_event(record, event, point)
            if event == "last_contribution_accepted":
                record["timing_origin"] = "HTTP contribution finish"
            ready_to_persist = record.get("checkpoint_generation") is not None and persist
        if ready_to_persist:
            self.persist_timing_receipt(window_id, attempt_id)

    def promote_timing_event(
        self,
        window_id: str,
        attempt_id: str,
        source_event: str,
        target_event: str,
    ) -> None:
        key = _attempt_key(window_id, attempt_id)
        with self._timing_lock:
            record = self._timing_records.get(key)
            source = None if record is None else record.get("events", {}).get(source_event)
            point = dict(source) if isinstance(source, Mapping) else None
        self.note_timing_event(
            window_id, attempt_id, target_event, point, persist=False,
        )

    def record_timing_span(
        self,
        window_id: str,
        attempt_id: str,
        name: str,
        start: Mapping[str, Any],
        end: Mapping[str, Any] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("timing span name must be nonempty text")
        record = self._timing_record(window_id, attempt_id)
        finish = _timing_point() if end is None else dict(end)
        with self._timing_lock:
            _timing_span(record, name, start, finish)
            ready_to_persist = record.get("checkpoint_generation") is not None and persist
        if ready_to_persist:
            self.persist_timing_receipt(window_id, attempt_id)

    def latest_timing_identity(self) -> tuple[str, str] | None:
        with self._timing_lock:
            return self._latest_timing_identity

    def note_latest_timing_event(
        self,
        event: str,
        point: Mapping[str, Any] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        identity = self.latest_timing_identity()
        if identity is not None:
            self.note_timing_event(*identity, event, point, persist=persist)

    def record_latest_timing_span(
        self,
        name: str,
        start: Mapping[str, Any],
        end: Mapping[str, Any] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        identity = self.latest_timing_identity()
        if identity is not None:
            self.record_timing_span(*identity, name, start, end, persist=persist)

    def persist_timing_receipt(
        self,
        window_id: str,
        attempt_id: str,
    ) -> str | None:
        key = _attempt_key(window_id, attempt_id)
        with self._timing_lock:
            record = self._timing_records.get(key)
            if record is None or type(record.get("checkpoint_generation")) is not int:
                return None
            timing_dir = self.root / "timing-receipts"
            if timing_dir.is_symlink():
                record["receipt_persist_error"] = "timing receipt directory is a symlink"
                return None
            try:
                existed = timing_dir.exists()
                timing_dir.mkdir(parents=True, exist_ok=True)
                if timing_dir.is_symlink() or not timing_dir.is_dir():
                    raise GradientCheckpointError("timing receipt path is unsafe")
                if not existed:
                    _fsync_dir(self.root)
                path = timing_dir / f"update-{record['checkpoint_generation']:08d}.json"
                record["receipt_path"] = str(path)
                record["receipt_updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                _atomic_json(path, copy.deepcopy(record))
                record.pop("receipt_persist_error", None)
                return str(path)
            except Exception as exc:
                record["receipt_persist_error"] = f"{type(exc).__name__}: {exc}"
                return None

    def latest_timing_receipt(self) -> dict[str, Any] | None:
        with self._timing_lock:
            if self._latest_timing_key is None:
                return None
            record = self._timing_records.get(self._latest_timing_key)
            return None if record is None else copy.deepcopy(record)

    def record_staging_timing(self, window_id: str, attempt_id: str, summary: Mapping[str, Any]) -> None:
        """Attach measured pre-optimizer phases to the existing diagnostic receipt."""
        record = self._timing_record(window_id, attempt_id)
        with self._timing_lock:
            record["staged_materialization"] = copy.deepcopy(dict(summary))

    @staticmethod
    def _counter(value: Any, label: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    def _read_optimizer_layout(self) -> tuple[tuple[str, ...], ...]:
        name_by_id = {id(p): name for name, p in self._parameters.items()}
        groups: list[tuple[str, ...]] = []
        seen: set[str] = set()
        for group in self.optimizer.param_groups:
            names = []
            for parameter in group["params"]:
                name = name_by_id.get(id(parameter))
                if name is None or name in seen:
                    raise ValueError("AdamW must own each LoRA parameter exactly once")
                names.append(name)
                seen.add(name)
            if not names:
                raise ValueError("AdamW groups may not be empty")
            groups.append(tuple(names))
        if seen != set(self._parameters):
            raise ValueError("AdamW must own every trainable LoRA parameter")
        return tuple(groups)

    def _capture_trainable_state(self) -> dict[str, torch.Tensor]:
        """Own one immutable CPU copy of the post-update trainable state."""
        result: dict[str, torch.Tensor] = {}
        for name, parameter in self._parameters.items():
            value = parameter.detach()
            if value.device.type == "cpu":
                value = value.clone(memory_format=torch.contiguous_format)
            else:
                value = value.to(device="cpu", memory_format=torch.contiguous_format)
            result[name] = value
        return result

    def _checkpoint_payload(
        self,
        *,
        snapshot_id: str,
        trainable_state: Mapping[str, torch.Tensor] | None = None,
        pending_attempt: Mapping[str, Any] | None = None,
        policy_version: int | None = None,
        optimizer_steps: int | None = None,
        scheduler_steps: int | None = None,
        prepared_windows: Mapping[str, PreparedWindow] | None = None,
        committed_attempts: Mapping[str, tuple[str, WindowStepReceipt]] | None = None,
        committed_windows: Mapping[str, str] | None = None,
        last_receipt: WindowStepReceipt | None = None,
        last_update_context: Mapping[str, Any] | None = None,
        checkpoint_reason: str | None = None,
        rng_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        prepared = self._prepared_windows if prepared_windows is None else prepared_windows
        committed = self._committed_attempts if committed_attempts is None else committed_attempts
        windows = self._committed_windows if committed_windows is None else committed_windows
        receipt = self._last_receipt if last_receipt is None else last_receipt
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "generation": self._checkpoint_generation + 1,
            "snapshot_id": snapshot_id,
            "parameter_layout": [item.to_dict() for item in self.parameter_layout],
            "layout_fingerprint": self.layout_fingerprint,
            "optimizer_layout": [list(group) for group in self._optimizer_layout],
            "initial_policy_version": self.initial_policy_version,
            "compatibility": copy.deepcopy(self.compatibility),
            "lineage_id": self.lineage_id,
            "engine_epoch": self.engine_epoch,
            "policy_version": self.policy_version if policy_version is None else policy_version,
            "optimizer_steps": self.optimizer_steps if optimizer_steps is None else optimizer_steps,
            "scheduler_steps": self.scheduler_steps if scheduler_steps is None else scheduler_steps,
            "trainable_state": (
                self._capture_trainable_state() if trainable_state is None else trainable_state
            ),
            "optimizer_state": _cpu_tree(self.optimizer.state_dict()),
            "scheduler_class": f"{type(self.scheduler).__module__}.{type(self.scheduler).__qualname__}",
            "scheduler_state": _cpu_tree(self.scheduler.state_dict()),
            "rng_state": _capture_rng() if rng_state is None else copy.deepcopy(dict(rng_state)),
            "prepared_windows": [copy.deepcopy(window.to_dict()) for _, window in sorted(prepared.items())],
            "committed_attempts": [
                {"key": key, "fingerprint": fingerprint, "receipt": saved_receipt.to_dict()}
                for key, (fingerprint, saved_receipt) in sorted(committed.items())
            ],
            "committed_windows": dict(sorted(windows.items())),
            "cancelled_windows": [
                copy.deepcopy(record) for _, record in sorted(self._cancelled_windows.items())
            ],
            "pending_attempt": None if pending_attempt is None else dict(pending_attempt),
            "last_receipt": None if receipt is None else receipt.to_dict(),
            "last_update_context": copy.deepcopy(
                self.last_update_context if last_update_context is None else dict(last_update_context)
            ),
            "checkpoint_reason": checkpoint_reason,
        }

    def _materialize_checkpoint_payload(
        self,
        *,
        timing_record: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        checkpoint_started = _timing_point()
        payload_started = _timing_point()
        snapshot_id = overrides.pop("snapshot_id", uuid4().hex)
        payload = self._checkpoint_payload(snapshot_id=snapshot_id, **overrides)
        payload_finished = _timing_point()
        if timing_record is not None:
            _timing_span(timing_record, "checkpoint_payload_materialize", payload_started, payload_finished)
        return payload, checkpoint_started

    def _record_checkpoint_write(
        self,
        payload: Mapping[str, Any],
        result: CheckpointWriteResult,
        *,
        timing_record: dict[str, Any] | None,
        checkpoint_started: Mapping[str, Any],
    ) -> None:
        self._checkpoint_generation = result.generation
        self.snapshot_id = result.snapshot_id
        self.durable_policy_version = int(payload["policy_version"])
        self.durable_optimizer_steps = int(payload["optimizer_steps"])
        self.durable_scheduler_steps = int(payload["scheduler_steps"])
        self._checkpoint_cleanup_error = result.maintenance_error
        if timing_record is not None:
            try:
                with self._timing_lock:
                    for name, (start, end) in result.spans.items():
                        _timing_span(timing_record, name, start, end)
                    timing_record["checkpoint_generation"] = result.generation
                    timing_record["checkpoint_snapshot_id"] = result.snapshot_id
                    timing_record["checkpoint_policy_version"] = payload["policy_version"]
                    timing_record["checkpoint_payload_bytes"] = result.size_bytes
                    if result.maintenance_error is not None:
                        timing_record["checkpoint_maintenance_error"] = result.maintenance_error
                    _timing_span(
                        timing_record, "checkpoint_commit_total",
                        checkpoint_started, _timing_point(),
                    )
            except Exception as exc:
                self._checkpoint_timing_error = f"{type(exc).__name__}: {exc}"

    def _save_checkpoint(
        self,
        *,
        timing_record: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> int:
        payload, checkpoint_started = self._materialize_checkpoint_payload(
            timing_record=timing_record,
            **overrides,
        )
        try:
            result = write_checkpoint_payload(self.root, payload)
        except CheckpointWriteError as exc:
            raise GradientCheckpointError(str(exc)) from exc
        self._record_checkpoint_write(
            payload, result,
            timing_record=timing_record,
            checkpoint_started=checkpoint_started,
        )
        return result.generation

    def save_checkpoint(
        self,
        *,
        update_context: Mapping[str, Any] | None = None,
        reason: str = "manual",
        rng_state: Mapping[str, Any] | None = None,
    ) -> int:
        with self._lock:
            self._ensure_clean()
            self._ensure_checkpoint_available()
            try:
                context = self.last_update_context if update_context is None else dict(update_context)
                result = self._save_checkpoint(
                    last_update_context=context,
                    checkpoint_reason=reason,
                    rng_state=rng_state,
                )
                if update_context is not None:
                    self.last_update_context = dict(update_context)
                return result
            except Exception as exc:
                self._checkpoint_state = "faulted"
                self._checkpoint_failure = f"{type(exc).__name__}: {exc}"
                self._mark_dirty(f"explicit checkpoint save failed: {type(exc).__name__}: {exc}")
                raise

    def initialize_ddp_counters(
        self,
        receipt: Mapping[str, Any],
        *,
        commit: bool,
    ) -> None:
        """Validate DDP counters and, for a fresh core, commit them once."""
        if type(commit) is not bool:
            raise TypeError("commit must be a boolean")
        with self._lock:
            if not isinstance(receipt, Mapping) or receipt.get("schema") != _DDP_INITIALIZATION_SCHEMA:
                raise DDPCounterInitializationError("invalid DDP initialization receipt")

            source_policy = receipt.get("source_policy_version")
            source_optimizer_steps = receipt.get("source_optimizer_steps")
            source_scheduler_steps = receipt.get("source_scheduler_steps")
            initial_policy = receipt.get("source_initial_policy_version")
            if any(type(value) is not int or value < 0 for value in (
                source_policy, source_optimizer_steps, source_scheduler_steps, initial_policy,
            )):
                raise DDPCounterInitializationError("DDP initialization counters are invalid")
            if (source_scheduler_steps != source_optimizer_steps
                    or source_policy != initial_policy + source_optimizer_steps):
                raise DDPCounterInitializationError(
                    "DDP policy, AdamW, and scheduler counters disagree"
                )

            if commit:
                if any(value != 0 for value in (
                    self.policy_version,
                    self.initial_policy_version,
                    self.optimizer_steps,
                    self.scheduler_steps,
                )):
                    raise DDPCounterInitializationError(
                        "central core is not at a fresh genesis before DDP initialization"
                    )
                self.initial_policy_version = initial_policy
                self.policy_version = source_policy
                self.optimizer_steps = source_optimizer_steps
                self.scheduler_steps = source_scheduler_steps
                self.save_checkpoint(reason="ddp_initialization")
                initialized = (
                    self.initial_policy_version == initial_policy
                    and self.optimizer_steps >= source_optimizer_steps
                    and self.scheduler_steps >= source_scheduler_steps
                    and self.policy_version >= source_policy
                )
            else:
                initialized = not (
                    self.initial_policy_version != initial_policy
                    or self.optimizer_steps < source_optimizer_steps
                    or self.scheduler_steps < source_scheduler_steps
                    or self.policy_version < source_policy
                )
            if not initialized:
                raise DDPCounterInitializationError(
                    "central checkpoint predates the complete DDP state import; "
                    "refusing fresh-optimizer resume"
                )

    def _load_snapshot(self, snapshot: Path, expected_generation: int | None = None) -> None:
        if snapshot.is_symlink() or not snapshot.is_file():
            raise GradientCheckpointError("central checkpoint snapshot is missing or unsafe")
        try:
            import resource
            if self._snapshot_mmap:
                import mmap
                if torch.serialization.get_default_mmap_options() != mmap.MAP_PRIVATE:
                    raise GradientCheckpointError("mapped checkpoint recovery requires private mappings")
            load_started = time.perf_counter()
            payload = torch.load(snapshot, map_location="cpu", weights_only=False, mmap=self._snapshot_mmap)
            self.snapshot_load_metrics.append({
                "path": str(snapshot), "mapped": self._snapshot_mmap,
                "seconds": time.perf_counter() - load_started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            })
        except Exception as exc:
            raise GradientCheckpointError(f"cannot load central checkpoint: {type(exc).__name__}: {exc}") from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != _CHECKPOINT_SCHEMA:
            raise GradientCheckpointError("central checkpoint schema is unsupported")
        if expected_generation is not None and payload.get("generation") != expected_generation:
            raise GradientCheckpointError("checkpoint generation differs from latest pointer")
        saved_snapshot_id = payload.get("snapshot_id")
        if not isinstance(saved_snapshot_id, str) or re.fullmatch(r"[0-9a-f]{32}", saved_snapshot_id) is None:
            raise GradientCheckpointError("checkpoint snapshot_id is invalid")
        layout = tuple(GradientParameter.from_dict(item) for item in payload.get("parameter_layout", []))
        if layout != self.parameter_layout or payload.get("layout_fingerprint") != self.layout_fingerprint:
            raise GradientCheckpointError("checkpoint LoRA parameter layout differs from loaded adapter")
        if tuple(tuple(group) for group in payload.get("optimizer_layout", [])) != self._optimizer_layout:
            raise GradientCheckpointError("checkpoint AdamW parameter mapping differs")
        if not compatibility_matches(
            self.compatibility,
            payload.get("compatibility"),
            path_mappings=self.compatibility_path_mappings,
        ):
            raise GradientCheckpointError("central checkpoint compatibility metadata differs")
        saved_lineage = payload.get("lineage_id")
        saved_epoch = payload.get("engine_epoch")
        if not isinstance(saved_lineage, str) or re.fullmatch(r"[0-9a-f]{32}", saved_lineage) is None:
            raise GradientCheckpointError("checkpoint lineage_id is invalid")
        if self._configured_lineage_id is not None and self._configured_lineage_id != saved_lineage:
            raise GradientCheckpointError("checkpoint lineage_id differs from configured lineage")
        if not isinstance(saved_epoch, str) or re.fullmatch(r"[0-9a-f]{32}", saved_epoch) is None:
            raise GradientCheckpointError("checkpoint engine_epoch is invalid")
        if self._configured_engine_epoch is not None and self._configured_engine_epoch != saved_epoch:
            raise GradientCheckpointError("checkpoint engine_epoch differs from configured epoch")
        scheduler_class = f"{type(self.scheduler).__module__}.{type(self.scheduler).__qualname__}"
        if payload.get("scheduler_class") != scheduler_class:
            raise GradientCheckpointError("checkpoint scheduler class differs")
        state = payload.get("trainable_state")
        if not isinstance(state, Mapping) or set(state) != set(self._parameters):
            raise GradientCheckpointError("checkpoint must contain exactly named LoRA trainable tensors")
        for name, parameter in self._parameters.items():
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(parameter.shape)
                or value.dtype != parameter.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise GradientCheckpointError(f"checkpoint LoRA tensor is invalid: {name}")
        for key in ("initial_policy_version", "policy_version", "optimizer_steps", "scheduler_steps"):
            if type(payload.get(key)) is not int or payload[key] < 0:
                raise GradientCheckpointError(f"checkpoint counter is invalid: {key}")
        if payload["policy_version"] != payload["initial_policy_version"] + payload["optimizer_steps"]:
            raise GradientCheckpointError("policy version and optimizer steps disagree")
        if payload["scheduler_steps"] != payload["optimizer_steps"]:
            raise GradientCheckpointError("scheduler and optimizer step counts disagree")
        try:
            with torch.no_grad():
                for name, parameter in self._parameters.items():
                    parameter.copy_(state[name].to(device=parameter.device))
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.scheduler.load_state_dict(payload["scheduler_state"])
            _restore_rng(payload["rng_state"])
        except Exception as exc:
            raise GradientCheckpointError(
                f"optimizer/scheduler/RNG restore failed: {type(exc).__name__}: {exc}"
            ) from exc
        prepared: dict[str, PreparedWindow] = {}
        for raw in payload.get("prepared_windows", []):
            window = PreparedWindow.from_dict(raw)
            key = _attempt_key(window.window_id, window.attempt_id)
            if key in prepared or window.policy_version != payload["policy_version"]:
                raise GradientCheckpointError("prepared window identity/version is invalid")
            prepared[key] = window
        if len(prepared) > 1:
            raise GradientCheckpointError("multiple outstanding prepared windows are unsupported")
        committed: dict[str, tuple[str, WindowStepReceipt]] = {}
        for item in payload.get("committed_attempts", []):
            if not isinstance(item, Mapping) or set(item) != {"key", "fingerprint", "receipt"}:
                raise GradientCheckpointError("committed attempt record is malformed")
            receipt = WindowStepReceipt.from_dict(item["receipt"])
            key = item["key"]
            if not isinstance(key, str) or key != _attempt_key(receipt.window_id, receipt.attempt_id):
                raise GradientCheckpointError("committed attempt identity is invalid")
            if key in committed or not isinstance(item["fingerprint"], str):
                raise GradientCheckpointError("duplicate or invalid committed attempt")
            committed[key] = (item["fingerprint"], receipt)
        committed_windows = payload.get("committed_windows", {})
        if not isinstance(committed_windows, Mapping) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in committed_windows.items()
        ):
            raise GradientCheckpointError("committed window index is malformed")
        cancelled: dict[str, dict[str, Any]] = {}
        for record in payload.get("cancelled_windows", []):
            self._validate_cancellation_record(record)
            key = _attempt_key(record["window_id"], record["attempt_id"])
            if key in cancelled:
                raise GradientCheckpointError("duplicate cancellation record in checkpoint")
            if record["window_id"] in committed_windows or key in committed:
                raise GradientCheckpointError("checkpoint marks a committed window as cancelled")
            cancelled[key] = dict(record)
        self.initial_policy_version = payload["initial_policy_version"]
        self.lineage_id = saved_lineage
        self.engine_epoch = saved_epoch
        self.policy_version = payload["policy_version"]
        self.optimizer_steps = payload["optimizer_steps"]
        self.scheduler_steps = payload["scheduler_steps"]
        self.durable_policy_version = payload["policy_version"]
        self.durable_optimizer_steps = payload["optimizer_steps"]
        self.durable_scheduler_steps = payload["scheduler_steps"]
        self._checkpoint_generation = payload["generation"]
        self.snapshot_id = saved_snapshot_id
        self._prepared_windows = prepared
        self._committed_attempts = committed
        self._committed_windows = dict(committed_windows)
        self._cancelled_windows = cancelled
        pending = payload.get("pending_attempt")
        self._checkpoint_pending_attempt = None if pending is None else dict(pending)
        if pending is not None:
            if not isinstance(pending, Mapping) or set(pending) != {"window_id", "attempt_id", "fingerprint"}:
                raise GradientCheckpointError("pending optimizer fence is malformed")
            self._pending_attempt = dict(pending)
            self._checkpoint_state = "faulted"
            self._mark_dirty("latest checkpoint contains an unfinished optimizer attempt")
        else:
            self._pending_attempt = None
        raw_receipt = payload.get("last_receipt")
        self._last_receipt = None if raw_receipt is None else WindowStepReceipt.from_dict(raw_receipt)
        last_context = payload.get("last_update_context")
        self.last_update_context = None if last_context is None else dict(last_context)
        for parameter in self._parameters.values():
            parameter.grad = None

    def resume(self, checkpoint_path: str | Path | None = None) -> None:
        with self._lock:
            self._ensure_checkpoint_available()
            if checkpoint_path is None:
                if self.latest_path.is_symlink() or not self.latest_path.is_file():
                    raise GradientCheckpointError("central latest pointer is missing or unsafe")
                try:
                    with open(self.latest_path, "r", encoding="utf-8") as stream:
                        pointer = json.load(stream)
                except Exception as exc:
                    raise GradientCheckpointError(f"cannot read latest pointer: {type(exc).__name__}: {exc}") from exc
                required = {
                    "schema", "generation", "snapshot_id", "snapshot", "size_bytes",
                    "policy_version", "optimizer_steps",
                }
                if (
                    not isinstance(pointer, Mapping)
                    or set(pointer) != required
                    or pointer.get("schema") != _POINTER_SCHEMA
                ):
                    raise GradientCheckpointError("central latest pointer schema is unsupported")
                name = pointer.get("snapshot")
                if (
                    not isinstance(name, str)
                    or Path(name).name != name
                    or not name.startswith("state-")
                    or not name.endswith(".pt")
                ):
                    raise GradientCheckpointError("latest snapshot path is invalid")
                snapshot = self.root / name
                if (
                    snapshot.is_symlink()
                    or not snapshot.is_file()
                    or snapshot.stat().st_size != pointer.get("size_bytes")
                ):
                    raise GradientCheckpointError("latest snapshot size or file identity is invalid")
                self._load_snapshot(snapshot, pointer.get("generation"))
                if pointer.get("snapshot_id") != self.snapshot_id:
                    raise GradientCheckpointError("latest pointer snapshot_id differs from full state")
                if (
                    self.policy_version != pointer.get("policy_version")
                    or self.optimizer_steps != pointer.get("optimizer_steps")
                ):
                    raise GradientCheckpointError("latest pointer counters disagree with snapshot")
            else:
                target = Path(checkpoint_path)
                if target.is_dir():
                    raise GradientCheckpointError("explicit resume path must name a snapshot")
                self._load_snapshot(target)
            self._restore_control_files()
            if self._pending_attempt is not None and self._checkpoint_state != "recovery_required":
                self._checkpoint_state = "faulted"
                self._mark_dirty("resume found an incomplete optimizer transaction")
            self.resumed = True

    def _ensure_clean(self) -> None:
        if self._dirty:
            raise DirtyGradientCoreError(self._dirty_reason or "central optimizer state is dirty")

    def _ensure_checkpoint_available(self) -> None:
        if self._checkpoint_state == "saving":
            raise CheckpointPendingError("central checkpoint writer is still active")
        if self._checkpoint_state in {"faulted", "recovery_required"}:
            raise DirtyGradientCoreError(
                self._dirty_reason or "central checkpoint requires explicit recovery"
            )

    def _mark_dirty(self, reason: str) -> None:
        self._dirty = True
        self._dirty_reason = reason

    def _write_prepared_sidecar(self, window: PreparedWindow) -> None:
        _atomic_json(self.prepared_path, {
            "schema": "central-gradient-prepared-window/v1",
            "checkpoint_generation": self._checkpoint_generation,
            "window": window.to_dict(),
        })

    def _write_pending_fence(self, pending: Mapping[str, Any]) -> None:
        _atomic_json(self.pending_path, {
            "schema": "central-gradient-pending-attempt/v1",
            "checkpoint_generation": self._checkpoint_generation,
            "attempt": dict(pending),
        })

    def _write_async_checkpoint_intent(
        self,
        *,
        window: PreparedWindow,
        fingerprint: str,
        target_generation: int,
        target_snapshot_id: str,
        target_policy_version: int,
        target_optimizer_steps: int,
        target_scheduler_steps: int,
    ) -> dict[str, Any]:
        if self.snapshot_id is None:
            raise GradientCheckpointError("async checkpoint requires a durable source snapshot")
        intent = {
            "schema": "central-gradient-async-checkpoint-intent/v1",
            "operation_id": uuid4().hex,
            "source": {
                "generation": self._checkpoint_generation,
                "snapshot_id": self.snapshot_id,
                "policy_version": self.durable_policy_version,
                "optimizer_steps": self.durable_optimizer_steps,
                "scheduler_steps": self.durable_scheduler_steps,
            },
            "target": {
                "generation": target_generation,
                "snapshot_id": target_snapshot_id,
                "policy_version": target_policy_version,
                "optimizer_steps": target_optimizer_steps,
                "scheduler_steps": target_scheduler_steps,
            },
            "window_id": window.window_id,
            "attempt_id": window.attempt_id,
            "fingerprint": fingerprint,
        }
        _atomic_json(self.async_intent_path, intent)
        self._checkpoint_intent = intent
        return intent

    def _restore_async_checkpoint_intent(self) -> None:
        path = self.async_intent_path
        if path.is_symlink():
            raise GradientCheckpointError("async checkpoint intent may not be a symlink")
        if not path.exists():
            return
        try:
            intent = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise GradientCheckpointError(f"cannot read async checkpoint intent: {exc}") from exc
        fields = {"schema", "operation_id", "source", "target", "window_id", "attempt_id", "fingerprint"}
        counter_fields = {"generation", "snapshot_id", "policy_version", "optimizer_steps", "scheduler_steps"}
        if (
            not isinstance(intent, Mapping)
            or set(intent) != fields
            or intent.get("schema") != "central-gradient-async-checkpoint-intent/v1"
            or not isinstance(intent.get("source"), Mapping)
            or set(intent["source"]) != counter_fields
            or not isinstance(intent.get("target"), Mapping)
            or set(intent["target"]) != counter_fields
        ):
            raise GradientCheckpointError("async checkpoint intent schema is invalid")
        source = dict(intent["source"])
        target = dict(intent["target"])
        if (
            not isinstance(intent.get("operation_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", intent["operation_id"]) is None
            or not isinstance(intent.get("window_id"), str) or not intent["window_id"]
            or not isinstance(intent.get("attempt_id"), str) or not intent["attempt_id"]
            or not isinstance(intent.get("fingerprint"), str)
            or re.fullmatch(r"[0-9a-f]{64}", intent["fingerprint"]) is None
            or not isinstance(source.get("snapshot_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", source["snapshot_id"]) is None
            or not isinstance(target.get("snapshot_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", target["snapshot_id"]) is None
        ):
            raise GradientCheckpointError("async checkpoint intent identity is invalid")
        for snapshot in (source, target):
            for field in ("generation", "policy_version", "optimizer_steps", "scheduler_steps"):
                if type(snapshot.get(field)) is not int or snapshot[field] < 0:
                    raise GradientCheckpointError(f"async checkpoint intent counter is invalid: {field}")
            if snapshot["scheduler_steps"] != snapshot["optimizer_steps"]:
                raise GradientCheckpointError("async intent optimizer/scheduler counts disagree")
            if snapshot["policy_version"] != self.initial_policy_version + snapshot["optimizer_steps"]:
                raise GradientCheckpointError("async intent policy/optimizer counters disagree")
        if (
            target["generation"] != source["generation"] + 1
            or target["policy_version"] != source["policy_version"] + 1
            or target["optimizer_steps"] != source["optimizer_steps"] + 1
            or target["scheduler_steps"] != source["scheduler_steps"] + 1
        ):
            raise GradientCheckpointError("async checkpoint intent source/target transition is invalid")
        self._checkpoint_intent = dict(intent)
        key = _attempt_key(intent["window_id"], intent["attempt_id"])
        if (
            self._checkpoint_generation == target["generation"]
            and self.snapshot_id == target["snapshot_id"]
            and self.policy_version == target["policy_version"]
            and self.optimizer_steps == target["optimizer_steps"]
            and self.scheduler_steps == target["scheduler_steps"]
        ):
            committed = self._committed_attempts.get(key)
            if committed is None or committed[0] != intent["fingerprint"]:
                raise GradientCheckpointError("target checkpoint does not contain the async committed attempt")
            try:
                _fsync_dir(self.root)
            except OSError as exc:
                raise GradientCheckpointError(
                    "target pointer is visible but its directory fsync could not be revalidated"
                ) from exc
            self._checkpoint_state = "durable"
            self._async_intent_resolved = True
            return
        if (
            self._checkpoint_generation == source["generation"]
            and self.snapshot_id == source["snapshot_id"]
            and self.policy_version == source["policy_version"]
            and self.optimizer_steps == source["optimizer_steps"]
            and self.scheduler_steps == source["scheduler_steps"]
        ):
            if self.pending_path.exists():
                try:
                    pending = json.loads(self.pending_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise GradientCheckpointError("async source state has an unreadable pending fence") from exc
                if (
                    not isinstance(pending, Mapping)
                    or set(pending) != {"schema", "checkpoint_generation", "attempt"}
                    or pending.get("schema") != "central-gradient-pending-attempt/v1"
                    or pending.get("checkpoint_generation") != source["generation"]
                    or pending.get("attempt") != {
                        "window_id": intent["window_id"],
                        "attempt_id": intent["attempt_id"],
                        "fingerprint": intent["fingerprint"],
                    }
                ):
                    raise GradientCheckpointError("async intent does not match the existing pending fence")
            self._checkpoint_state = "recovery_required"
            self._dirty = True
            self._dirty_reason = (
                "async checkpoint did not reach latest pointer; speculative policy "
                f"{target['policy_version']} requires explicit recovery"
            )
            return
        raise GradientCheckpointError(
            "async checkpoint intent matches neither the source nor target durable snapshot"
        )

    def _remove_control_files(self) -> None:
        changed = False
        for path in (self.prepared_path, self.pending_path, self.async_intent_path):
            try:
                path.unlink()
                changed = True
            except FileNotFoundError:
                pass
        if changed:
            _fsync_dir(self.root)

    @staticmethod
    def _validate_cancellation_record(record: Any) -> None:
        fields = {
            "schema", "window_id", "attempt_id", "fingerprint", "reason",
            "policy_version", "n_window_tokens", "work_order", "update_context",
            "cancelled_at_ns",
        }
        if not isinstance(record, Mapping) or set(record) != fields:
            raise GradientCheckpointError("window cancellation record is malformed")
        if record.get("schema") != "central-gradient-window-cancellation/v1":
            raise GradientCheckpointError("window cancellation schema is unsupported")
        for field in ("window_id", "attempt_id", "reason"):
            value = record.get(field)
            if not isinstance(value, str) or not value or value != value.strip():
                raise GradientCheckpointError(f"window cancellation {field} is invalid")
        if not isinstance(record.get("fingerprint"), str) or re.fullmatch(
            r"[0-9a-f]{64}", record["fingerprint"]
        ) is None:
            raise GradientCheckpointError("window cancellation fingerprint is invalid")
        if type(record.get("policy_version")) is not int or record["policy_version"] < 0:
            raise GradientCheckpointError("window cancellation policy version is invalid")
        if type(record.get("n_window_tokens")) is not int or record["n_window_tokens"] < 1:
            raise GradientCheckpointError("window cancellation token count is invalid")
        order = record.get("work_order")
        if (
            not isinstance(order, list)
            or not order
            or any(not isinstance(item, str) or not item for item in order)
            or len(order) != len(set(order))
        ):
            raise GradientCheckpointError("window cancellation work order is invalid")
        if not isinstance(record.get("update_context"), Mapping):
            raise GradientCheckpointError("window cancellation context is invalid")
        if type(record.get("cancelled_at_ns")) is not int or record["cancelled_at_ns"] <= 0:
            raise GradientCheckpointError("window cancellation timestamp is invalid")
        try:
            json.dumps(record["update_context"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise GradientCheckpointError("window cancellation context is not JSON-safe") from exc

    def _write_cancellations_sidecar(self) -> None:
        _atomic_json(self.root / "cancelled-windows.json", {
            "schema": "central-gradient-window-cancellations/v1",
            "checkpoint_generation": self._checkpoint_generation,
            "cancellations": [
                dict(record) for _, record in sorted(self._cancelled_windows.items())
            ],
        })

    def _restore_cancellations_sidecar(self) -> None:
        path = self.root / "cancelled-windows.json"
        if path.is_symlink():
            raise GradientCheckpointError("window cancellations sidecar may not be a symlink")
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise GradientCheckpointError(f"cannot read cancellations sidecar: {exc}") from exc
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"schema", "checkpoint_generation", "cancellations"}
            or raw.get("schema") != "central-gradient-window-cancellations/v1"
            or type(raw.get("checkpoint_generation")) is not int
            or raw["checkpoint_generation"] > self._checkpoint_generation
            or not isinstance(raw.get("cancellations"), list)
        ):
            raise GradientCheckpointError("window cancellations sidecar schema is invalid")
        for record in raw["cancellations"]:
            self._validate_cancellation_record(record)
            key = _attempt_key(record["window_id"], record["attempt_id"])
            if record["window_id"] in self._committed_windows or key in self._committed_attempts:
                raise GradientCheckpointError("cancellation sidecar conflicts with a committed update")
            previous = self._cancelled_windows.get(key)
            if previous is not None and previous != record:
                raise GradientCheckpointError("cancellation sidecar differs from checkpoint evidence")
            self._cancelled_windows[key] = dict(record)
        for key, prepared in tuple(self._prepared_windows.items()):
            cancelled = self._cancelled_windows.get(key)
            if cancelled is None:
                continue
            if cancelled["fingerprint"] != _digest_json(prepared.to_dict()):
                raise GradientCheckpointError("cancelled checkpoint window differs from cancellation evidence")
            self._prepared_windows.pop(key, None)

    def managed_stop_now_discard_plan(self, pointer: Mapping[str, Any]) -> dict[str, Any]:
        """Validate which prepared Core windows an authorized stopped-instance discard can fence."""
        with self._lock:
            status = self.status()
            if (not isinstance(pointer, Mapping)
                    or pointer.get("schema") != _POINTER_SCHEMA
                    or pointer.get("generation") != status.get("checkpoint_generation")
                    or pointer.get("snapshot_id") != status.get("snapshot_id")
                    or pointer.get("policy_version") != status.get("policy_version")
                    or pointer.get("optimizer_steps") != status.get("optimizer_steps")):
                raise GradientCheckpointError("managed stop selected pointer differs from the loaded Core")
            if self._checkpoint_state == "saving" or self._checkpoint_future is not None:
                raise DirtyGradientCoreError("managed stop cannot discard while checkpoint writing is active")
            if self._checkpoint_pending_attempt is not None:
                raise DirtyGradientCoreError(
                    "latest full checkpoint itself contains an optimizer-entry fence"
                )
            if self._checkpoint_state not in {"durable", "faulted", "recovery_required"}:
                raise DirtyGradientCoreError("managed stop Core checkpoint state is unsupported")
            prepared = tuple(self._prepared_windows.values())
            intent = self._checkpoint_intent
            pending = self._pending_attempt
            if intent is not None:
                target = intent.get("target")
                if (isinstance(target, Mapping)
                        and target.get("generation") == pointer.get("generation")
                        and target.get("snapshot_id") == pointer.get("snapshot_id")
                        and target.get("policy_version") == pointer.get("policy_version")
                        and target.get("optimizer_steps") == pointer.get("optimizer_steps")):
                    key = _attempt_key(intent["window_id"], intent["attempt_id"])
                    committed = self._committed_attempts.get(key)
                    if (self._dirty or self._checkpoint_state != "durable" or pending is not None
                            or prepared or committed is None
                            or committed[0] != intent.get("fingerprint")):
                        raise GradientCheckpointError("target async intent lacks a matching durable Core receipt")
                    return {
                        "checkpoint": {
                            "generation": status["checkpoint_generation"],
                            "snapshot_id": status["snapshot_id"],
                            "lineage_id": status["lineage_id"],
                            "engine_epoch": status["engine_epoch"],
                            "policy_version": status["policy_version"],
                            "optimizer_steps": status["optimizer_steps"],
                            "scheduler_steps": status["scheduler_steps"],
                            "last_update_context": copy.deepcopy(status.get("last_update_context")),
                        },
                        "checkpoint_state": self._checkpoint_state,
                        "dirty": self._dirty,
                        "async_intent": copy.deepcopy(intent),
                        "pending_attempt": None,
                        "windows": [],
                        "committed_receipt": committed[1].to_dict(),
                    }
            if (self._dirty or self._checkpoint_state != "durable" or intent is not None or pending is not None) and not prepared:
                raise DirtyGradientCoreError("ambiguous Core state has no identity-bound prepared window")
            if ((self._dirty or self._checkpoint_state != "durable")
                    and intent is None and pending is None):
                raise DirtyGradientCoreError("managed stop found an unclassified dirty Core state")
            if len(prepared) > 1:
                raise GradientCheckpointError("managed stop found more than one prepared Core window")
            window_records = []
            for window in prepared:
                key, fingerprint = self._prepared_identity(window)
                if window.policy_version != status["policy_version"]:
                    raise GradientCheckpointError("managed stop prepared window policy differs from durable Core")
                if key in self._committed_attempts or window.window_id in self._committed_windows:
                    raise GradientCheckpointError("managed stop prepared window conflicts with a committed Core receipt")
                if pending is not None and pending != {
                    "window_id": window.window_id,
                    "attempt_id": window.attempt_id,
                    "fingerprint": fingerprint,
                }:
                    raise GradientCheckpointError("managed stop pending fence differs from the prepared window")
                if intent is not None:
                    source = intent.get("source")
                    if (not isinstance(source, Mapping)
                            or source.get("generation") != pointer.get("generation")
                            or source.get("snapshot_id") != pointer.get("snapshot_id")
                            or intent.get("window_id") != window.window_id
                            or intent.get("attempt_id") != window.attempt_id
                            or intent.get("fingerprint") != fingerprint):
                        raise GradientCheckpointError("managed stop async intent is not tied to this durable source")
                window_records.append({
                    "window_id": window.window_id,
                    "attempt_id": window.attempt_id,
                    "fingerprint": fingerprint,
                    "policy_version": window.policy_version,
                    "lineage_id": window.lineage_id,
                    "engine_epoch": window.engine_epoch,
                    "job_count": len(window.jobs),
                    "update_context": dict(window.update_context or {}),
                })
            return {
                "checkpoint": {
                    "generation": status["checkpoint_generation"],
                    "snapshot_id": status["snapshot_id"],
                    "lineage_id": status["lineage_id"],
                    "engine_epoch": status["engine_epoch"],
                    "policy_version": status["policy_version"],
                    "optimizer_steps": status["optimizer_steps"],
                    "scheduler_steps": status["scheduler_steps"],
                    "last_update_context": copy.deepcopy(status.get("last_update_context")),
                },
                "checkpoint_state": self._checkpoint_state,
                "dirty": self._dirty,
                "async_intent": copy.deepcopy(intent),
                "pending_attempt": copy.deepcopy(pending),
                "windows": window_records,
                "committed_receipt": None,
            }

    def record_managed_stop_now_discard(
        self, plan: Mapping[str, Any], *, operation_id: str,
    ) -> dict[str, Any]:
        """Record an explicit rollback decision without advancing optimizer state.

        The caller must first fence the corresponding SQLite window and quarantine
        the validated control sidecars while holding instance and optimizer locks.
        """
        if not isinstance(operation_id, str) or re.fullmatch(r"[0-9a-f]{32}", operation_id) is None:
            raise ValueError("managed stop operation_id must be a lowercase UUID hex string")
        with self._lock:
            expected = self.managed_stop_now_discard_plan(plan.get("pointer", {}) if isinstance(plan, Mapping) else {})
            if plan.get("core_plan") != expected:
                raise GradientCheckpointError("managed stop Core plan changed before discard")
            records = []
            for window in tuple(self._prepared_windows.values()):
                key = _attempt_key(window.window_id, window.attempt_id)
                prior = self._cancelled_windows.get(key)
                fingerprint = _digest_json(window.to_dict())
                if prior is not None:
                    if prior["fingerprint"] != fingerprint:
                        raise GradientCheckpointError("managed stop cancellation conflicts with prior Core evidence")
                    records.append(dict(prior))
                    self._prepared_windows.pop(key, None)
                    continue
                record = {
                    "schema": "central-gradient-window-cancellation/v1",
                    "window_id": window.window_id,
                    "attempt_id": window.attempt_id,
                    "fingerprint": fingerprint,
                    "reason": f"managed_stop_now:{operation_id}",
                    "policy_version": window.policy_version,
                    "n_window_tokens": window.n_window_tokens,
                    "work_order": list(window.work_order),
                    "update_context": dict(window.update_context or {}),
                    "cancelled_at_ns": time.time_ns(),
                }
                self._validate_cancellation_record(record)
                self._cancelled_windows[key] = record
                records.append(dict(record))
                self._prepared_windows.pop(key, None)
                self._step_callbacks.pop(key, None)
                self._staged_callbacks.pop(key, None)
            self._write_cancellations_sidecar()
            self._pending_attempt = None
            self._checkpoint_intent = None
            self._async_intent_resolved = False
            self._checkpoint_state = "durable"
            self._dirty = False
            self._dirty_reason = None
            return {"operation_id": operation_id, "cancellations": records,
                    "checkpoint": dict(expected["checkpoint"])}

    def _restore_control_files(self) -> None:
        if (self.prepared_path.is_symlink() or self.pending_path.is_symlink()
                or self.async_intent_path.is_symlink()):
            raise GradientCheckpointError("gradient control sidecars may not be symlinks")
        self._restore_async_checkpoint_intent()
        self._restore_cancellations_sidecar()
        prepared = None
        if self.prepared_path.exists():
            try:
                raw = json.loads(self.prepared_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(raw, Mapping)
                    or set(raw) != {"schema", "checkpoint_generation", "window"}
                    or raw.get("schema") != "central-gradient-prepared-window/v1"
                    or type(raw.get("checkpoint_generation")) is not int
                    or raw["checkpoint_generation"] > self._checkpoint_generation
                ):
                    raise GradientCheckpointError("prepared-window sidecar schema is invalid")
                prepared = PreparedWindow.from_dict(raw["window"])
            except GradientCheckpointError:
                raise
            except Exception as exc:
                raise GradientCheckpointError(f"cannot read prepared-window sidecar: {exc}") from exc
        stale_prepared = False
        if prepared is not None:
            key = _attempt_key(prepared.window_id, prepared.attempt_id)
            cancelled = self._cancelled_windows.get(key)
            if cancelled is not None:
                if cancelled["fingerprint"] != _digest_json(prepared.to_dict()):
                    raise GradientCheckpointError("cancelled prepared sidecar differs from cancellation evidence")
                self._prepared_windows.pop(key, None)
                stale_prepared = True
                prepared = None
        if prepared is not None:
            key = _attempt_key(prepared.window_id, prepared.attempt_id)
            committed = self._committed_attempts.get(key)
            if committed is not None:
                if committed[0] != _digest_json(prepared.to_dict()):
                    raise GradientCheckpointError("stale prepared sidecar conflicts with committed receipt")
                self._prepared_windows.pop(key, None)
                stale_prepared = True
                prepared = None
            elif prepared.policy_version != self.policy_version:
                raise GradientCheckpointError("prepared sidecar policy version is stale")
            else:
                existing = self._prepared_windows.get(key)
                if existing is not None and existing != prepared:
                    raise GradientCheckpointError("prepared sidecar differs from checkpoint state")
                if self._prepared_windows and key not in self._prepared_windows:
                    raise GradientCheckpointError("more than one prepared window is outstanding")
                self._prepared_windows[key] = prepared
        if self._checkpoint_state == "recovery_required" and self._checkpoint_intent is not None:
            intent_key = _attempt_key(
                self._checkpoint_intent["window_id"], self._checkpoint_intent["attempt_id"],
            )
            restored_window = self._prepared_windows.get(intent_key)
            if (
                restored_window is None
                or _digest_json(restored_window.to_dict()) != self._checkpoint_intent["fingerprint"]
            ):
                raise GradientCheckpointError("async intent differs from the restored prepared window")
        fence_resolved = False
        if self.pending_path.exists():
            try:
                raw = json.loads(self.pending_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(raw, Mapping)
                    or set(raw) != {"schema", "checkpoint_generation", "attempt"}
                    or raw.get("schema") != "central-gradient-pending-attempt/v1"
                    or type(raw.get("checkpoint_generation")) is not int
                    or raw["checkpoint_generation"] > self._checkpoint_generation
                ):
                    raise GradientCheckpointError("pending-attempt sidecar schema is invalid")
                pending = raw["attempt"]
                if not isinstance(pending, Mapping) or set(pending) != {"window_id", "attempt_id", "fingerprint"}:
                    raise GradientCheckpointError("pending-attempt sidecar identity is malformed")
                key = _attempt_key(pending["window_id"], pending["attempt_id"])
                committed = self._committed_attempts.get(key)
                if committed is not None and committed[0] == pending["fingerprint"]:
                    self._pending_attempt = None
                    fence_resolved = True
                elif raw["checkpoint_generation"] != self._checkpoint_generation:
                    raise GradientCheckpointError("pending fence does not match latest checkpoint generation")
                self._pending_attempt = dict(pending)
                if not fence_resolved:
                    if self._checkpoint_state != "recovery_required":
                        self._checkpoint_state = "faulted"
                        self._mark_dirty("latest checkpoint has an unresolved pending optimizer fence")
                else:
                    self._pending_attempt = None
            except GradientCheckpointError:
                raise
            except Exception as exc:
                raise GradientCheckpointError(f"cannot read pending-attempt sidecar: {exc}") from exc
        if (
            fence_resolved
            or (stale_prepared and not self.pending_path.exists())
            or self._async_intent_resolved
        ):
            if not self.preserve_control_files:
                self._remove_control_files()
                self._checkpoint_intent = None
                self._async_intent_resolved = False

    def prepare_window(
        self,
        window_id: str,
        attempt_id: str,
        policy_version: int,
        n_window_tokens: int,
        jobs: Sequence[GradientJobSpec],
        *,
        compute_profile: Mapping[str, Any],
        update_context: Mapping[str, Any] | None = None,
        before_optimizer_step: Callable[[], None] | None = None,
    ) -> PreparedWindow:
        """Persist an immutable ordered job set before any contribution work."""
        with self._lock:
            self._ensure_clean()
            self._ensure_checkpoint_available()
            if not isinstance(window_id, str) or not window_id or window_id != window_id.strip():
                raise GradientContractError("window_id must be a non-empty trimmed string")
            if not isinstance(attempt_id, str) or not attempt_id or attempt_id != attempt_id.strip():
                raise GradientContractError("attempt_id must be a non-empty trimmed string")
            if type(policy_version) is not int or policy_version != self.policy_version:
                raise GradientContractError("prepared policy_version differs from central state")
            if type(n_window_tokens) is not int or n_window_tokens < 1:
                raise GradientContractError("n_window_tokens must be positive")
            profile = validate_compute_profile(compute_profile)
            profile_fingerprint = compute_profile_fingerprint(profile)
            for field in ("base_artifact_id", "loss_profile", "correction_cap"):
                if field in self.compatibility and self.compatibility[field] != profile[field]:
                    raise GradientContractError(f"compute profile differs from checkpoint compatibility: {field}")
            context = {} if update_context is None else dict(update_context)
            try:
                context = json.loads(json.dumps(context, sort_keys=True, separators=(",", ":")))
            except (TypeError, ValueError) as exc:
                raise GradientContractError("update_context must be JSON-safe") from exc
            specs = tuple(GradientJobSpec.from_value(item) for item in jobs)
            if not specs or len({spec.job_id for spec in specs}) != len(specs):
                raise GradientContractError("window requires unique jobs")
            if sum(spec.token_count for spec in specs) != n_window_tokens:
                raise GradientContractError("job token counts must sum to the shared denominator")
            bound_jobs = tuple(
                GradientJob(
                    schema="gradient_job/v1",
                    window_id=window_id,
                    attempt_id=attempt_id,
                    lineage_id=self.lineage_id,
                    engine_epoch=self.engine_epoch,
                    job_id=spec.job_id,
                    group_id=spec.group_id,
                    owner_executor_id=spec.owner_executor_id,
                    policy_version=policy_version,
                    n_window_tokens=n_window_tokens,
                    token_count=spec.token_count,
                    work_index=index,
                    layout_fingerprint=self.layout_fingerprint,
                    compute_profile=profile,
                    compute_profile_fingerprint=profile_fingerprint,
                )
                for index, spec in enumerate(specs)
            )
            window = PreparedWindow(
                schema=PREPARED_WINDOW_SCHEMA,
                aggregation=GRADIENT_CONTRIBUTION_CONTRACT,
                window_id=window_id,
                attempt_id=attempt_id,
                lineage_id=self.lineage_id,
                engine_epoch=self.engine_epoch,
                policy_version=policy_version,
                n_window_tokens=n_window_tokens,
                jobs=bound_jobs,
                work_order=tuple(job.job_id for job in bound_jobs),
                parameter_layout=self.parameter_layout,
                layout_fingerprint=self.layout_fingerprint,
                compute_profile=profile,
                update_context=context,
            )
            key = _attempt_key(window_id, attempt_id)
            if any(record["window_id"] == window_id for record in self._cancelled_windows.values()):
                raise GradientContractError("cancelled window_id may not be prepared again")
            if key in self._committed_attempts or window_id in self._committed_windows:
                raise GradientContractError("window or attempt identity was already committed")
            previous = self._prepared_windows.get(key)
            if previous is not None:
                if previous != window:
                    raise GradientContractError("prepared attempt identity was reused with changed data")
                if before_optimizer_step is not None:
                    self._step_callbacks[key] = before_optimizer_step
                return previous
            if self._prepared_windows:
                raise GradientContractError("only one prepared window may be outstanding")
            self._prepared_windows[key] = window
            if before_optimizer_step is not None:
                self._step_callbacks[key] = before_optimizer_step
            try:
                self._write_prepared_sidecar(window)
            except Exception as exc:
                self._mark_dirty(f"prepared-window checkpoint failed: {type(exc).__name__}: {exc}")
                raise DirtyGradientCoreError(self._dirty_reason) from exc
            return window

    def register_before_optimizer_step(
        self,
        window: PreparedWindow,
        callback: Callable[[], None],
    ) -> None:
        """Reattach a runtime callback to a recovered, still-uncommitted window."""
        with self._lock:
            self._ensure_clean()
            self._ensure_checkpoint_available()
            key = _attempt_key(window.window_id, window.attempt_id)
            if self._prepared_windows.get(key) != window:
                raise GradientContractError("callback window is not the active prepared window")
            if not callable(callback):
                raise TypeError("callback must be callable")
            self._step_callbacks[key] = callback

    def register_checkpoint_staged(
        self,
        window: PreparedWindow,
        callback: Callable[[Mapping[str, Any]], None],
    ) -> None:
        """Register a nonblocking metadata notification for async checkpoint staging."""
        if not self.async_checkpoint:
            raise RuntimeError("checkpoint-staged callbacks require async_checkpoint=True")
        if not callable(callback):
            raise TypeError("checkpoint-staged callback must be callable")
        with self._lock:
            self._ensure_clean()
            self._ensure_checkpoint_available()
            key = _attempt_key(window.window_id, window.attempt_id)
            if self._prepared_windows.get(key) != window:
                raise GradientContractError("staged callback window is not the active prepared window")
            if key in self._staged_callbacks:
                raise GradientContractError("checkpoint-staged callback is already registered")
            self._staged_callbacks[key] = callback

    def prepared_window(self, window_id: str, attempt_id: str) -> PreparedWindow | None:
        """Look up a durable prepared window before HTTP resume/auto-apply."""
        with self._lock:
            return self._prepared_windows.get(_attempt_key(window_id, attempt_id))

    def cancel_prepared_window(
        self,
        window: PreparedWindow,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Durably cancel one uncommitted prepared window without an optimizer step.

        The cancellation record is fsynced before prepared work is removed, so
        a restart cannot reactivate stale work.
        """
        with self._lock:
            self._ensure_clean()
            if not isinstance(window, PreparedWindow):
                raise GradientContractError("cancel_prepared_window requires PreparedWindow")
            window_id, attempt_id = window.window_id, window.attempt_id
            if (
                not isinstance(window_id, str) or not window_id or window_id != window_id.strip()
                or not isinstance(attempt_id, str) or not attempt_id or attempt_id != attempt_id.strip()
                or not isinstance(reason, str) or not reason or reason != reason.strip()
            ):
                raise GradientContractError("window_id, attempt_id and cancellation reason must be non-empty")
            if self._pending_attempt is not None or self.pending_path.exists():
                raise DirtyGradientCoreError("cannot cancel a window while an optimizer step is pending")
            key = _attempt_key(window_id, attempt_id)
            prior = self._cancelled_windows.get(key)
            if prior is not None:
                if prior["reason"] != reason:
                    raise GradientContractError("cancelled attempt was retried with a different reason")
                return dict(prior)
            if key in self._committed_attempts or window_id in self._committed_windows:
                raise GradientContractError("cannot cancel an already committed window")
            stored = self._prepared_windows.get(key)
            if stored is None or stored != window:
                raise GradientContractError("window is not the active prepared window")
            record = {
                "schema": "central-gradient-window-cancellation/v1",
                "window_id": window_id,
                "attempt_id": attempt_id,
                "fingerprint": _digest_json(stored.to_dict()),
                "reason": reason,
                "policy_version": stored.policy_version,
                "n_window_tokens": stored.n_window_tokens,
                "work_order": list(stored.work_order),
                "update_context": dict(stored.update_context or {}),
                "cancelled_at_ns": time.time_ns(),
            }
            self._validate_cancellation_record(record)
            self._cancelled_windows[key] = record
            try:
                self._write_cancellations_sidecar()
            except Exception as exc:
                self._mark_dirty(f"window cancellation persistence failed: {type(exc).__name__}: {exc}")
                raise DirtyGradientCoreError(self._dirty_reason) from exc
            self._prepared_windows.pop(key, None)
            self._step_callbacks.pop(key, None)
            self._staged_callbacks.pop(key, None)
            if self._gradient_preparation is not None and self._gradient_preparation.key == key:
                self._gradient_preparation = None
            try:
                self.prepared_path.unlink()
                _fsync_dir(self.root)
            except FileNotFoundError:
                pass
            except OSError:
                # The durable cancellation sidecar wins on restart.
                pass
            return dict(record)

    def _prepared_identity(self, window: PreparedWindow) -> tuple[str, str]:
        if not isinstance(window, PreparedWindow):
            raise GradientContractError("apply_window requires PreparedWindow")
        key = _attempt_key(window.window_id, window.attempt_id)
        fingerprint = _digest_json(window.to_dict())
        if key in self._cancelled_windows:
            raise GradientContractError("prepared window has been cancelled")
        stored = self._prepared_windows.get(key)
        if stored is not None:
            if stored != window:
                raise GradientContractError("window differs from durable prepared identity")
            return key, fingerprint
        committed = self._committed_attempts.get(key)
        if committed is not None and committed[0] == fingerprint:
            return key, fingerprint
        raise GradientContractError("window is not prepared by this central optimizer")

    def ready_jobs(
        self,
        window: PreparedWindow,
        completed_job_ids: Sequence[str] = (),
    ) -> tuple[GradientJob, ...]:
        with self._lock:
            self._ensure_clean()
            key, _ = self._prepared_identity(window)
            if key not in self._prepared_windows:
                return ()
            completed = set(completed_job_ids)
            if any(not isinstance(item, str) for item in completed) or not completed.issubset(window.work_order):
                raise GradientContractError("completed job set is invalid")
            return tuple(job for job in window.jobs if job.job_id not in completed)

    def _validate_contributions(
        self,
        window: PreparedWindow,
        contributions: Sequence[GradientContribution],
    ) -> dict[str, GradientContribution]:
        if not isinstance(contributions, Sequence) or isinstance(contributions, (str, bytes)):
            raise GradientContractError("contributions must be an ordered sequence")
        if len(contributions) != len(window.jobs):
            raise GradientContractError("every prepared job must contribute exactly once")
        expected = {job.job_id: job for job in window.jobs}
        layout = {item.name: item for item in self.parameter_layout}
        result: dict[str, GradientContribution] = {}
        for contribution in contributions:
            if not isinstance(contribution, GradientContribution):
                raise GradientContractError("invalid gradient contribution")
            job = expected.get(contribution.job_id)
            if job is None or contribution.job_id in result:
                raise GradientContractError("contribution coverage is missing, extra or duplicated")
            if (
                contribution.schema != GRADIENT_CONTRIBUTION_SCHEMA
                or contribution.aggregation != GRADIENT_CONTRIBUTION_CONTRACT
                or contribution.window_id != window.window_id
                or contribution.attempt_id != window.attempt_id
                or contribution.lineage_id != self.lineage_id
                or contribution.engine_epoch != self.engine_epoch
                or contribution.policy_version != window.policy_version
                or contribution.group_id != job.group_id
                or contribution.owner_executor_id != job.owner_executor_id
                or contribution.n_window_tokens != window.n_window_tokens
                or contribution.token_count != job.token_count
                or contribution.layout_fingerprint != self.layout_fingerprint
                or contribution.compute_profile_fingerprint != job.compute_profile_fingerprint
            ):
                raise GradientContractError("contribution identity, version, denominator or compute layout mismatch")
            if set(contribution.gradients) != set(layout):
                raise GradientContractError("contribution must preserve presence for every LoRA parameter")
            for name, value in contribution.gradients.items():
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise GradientContractError(f"gradient payload is not a tensor: {name}")
                if tuple(value.shape) != layout[name].shape:
                    raise GradientContractError(f"gradient shape mismatch: {name}")
                if value.dtype is not torch.float32:
                    raise GradientContractError(f"gradient transport dtype must be float32: {name}")
            result[contribution.job_id] = contribution
        if set(result) != set(expected):
            raise GradientContractError("contribution coverage is incomplete")
        if sum(result[job_id].token_count for job_id in window.work_order) != window.n_window_tokens:
            raise GradientContractError("contribution token counts do not cover the window")
        if not any(
            value is not None
            for contribution in result.values()
            for value in contribution.gradients.values()
        ):
            raise GradientContractError("window has no present LoRA gradient")
        return result

    def prepare_window_gradients(
        self, window: PreparedWindow, contributions: Sequence[GradientContribution],
    ) -> PreparedGradientWindow:
        """Validate and sum into private CPU buffers without mutating training state."""
        with self._lock:
            self._ensure_clean()
            self._ensure_checkpoint_available()
            key, fingerprint = self._prepared_identity(window)
            if key in self._committed_attempts or window.policy_version != self.policy_version:
                raise GradientContractError("cannot prepare gradients for a committed or stale window")
            if self._gradient_preparation is not None:
                raise GradientContractError("one gradient preparation is already pending")
            if any(parameter.device.type != "cpu" for parameter in self._parameters.values()):
                raise GradientContractError("pre-return gradient preparation requires CPU parameters")
            record = self._timing_record(window.window_id, window.attempt_id)
            started = _timing_point()
            _timing_event(record, "gradient_preparation_started", started)
            checked = self._validate_contributions(window, contributions)
            validated = _timing_point()
            _timing_span(record, "contribution_validate", started, validated)
            gradients = self._sum_contributions(window, checked)
            finished = _timing_point()
            _timing_span(record, "gradient_aggregate", validated, finished)
            _timing_event(record, "gradient_preparation_completed", finished)
            token = PreparedGradientWindow()
            self._gradient_preparation = _GradientPreparation(token, key, fingerprint, gradients)
            return token

    def discard_prepared_gradients(self, preparation: PreparedGradientWindow) -> bool:
        """Release only this preparation, including after a failed return fence."""
        with self._lock:
            current = self._gradient_preparation
            if current is None or current.token is not preparation:
                return False
            self._gradient_preparation = None
            return True

    def _sum_contributions(
        self, window: PreparedWindow, contributions: Mapping[str, GradientContribution],
    ) -> dict[str, torch.Tensor | None]:
        gradients: dict[str, torch.Tensor | None] = {}
        for name, parameter in self._parameters.items():
            total = None
            for job_id in window.work_order:
                value = contributions[job_id].gradients[name]
                if value is None:
                    continue
                value = value.detach().to(device=parameter.device, dtype=torch.float32)
                if total is None:
                    total = value.clone()
                else:
                    total.add_(value)
            gradients[name] = (
                None if total is None
                else total.to(device=parameter.device, dtype=parameter.dtype).detach()
            )
        if not any(value is not None for value in gradients.values()):
            raise GradientContractError("all jobs reported None for every LoRA parameter")
        return gradients

    def _attach_and_clip_gradients(
        self, gradients: Mapping[str, torch.Tensor | None], *,
        timing_record: dict[str, Any] | None = None,
    ) -> float:
        for name, parameter in self._parameters.items():
            parameter.grad = gradients[name]
        active = [parameter for parameter in self._parameters.values() if parameter.grad is not None]
        clip_started = _timing_point()
        try:
            norm = torch.nn.utils.clip_grad_norm_(active, self.max_grad_norm, error_if_nonfinite=True)
        except RuntimeError as exc:
            if "non-finite" in str(exc).lower():
                raise GradientContractError("global gradient norm is non-finite") from exc
            raise
        value = float(norm.detach().cpu().item())
        if not math.isfinite(value):
            raise GradientContractError("global gradient norm is non-finite")
        if value <= 0.0:
            raise GradientContractError("global gradient norm is zero")
        if timing_record is not None:
            _timing_span(timing_record, "gradient_norm_clip_and_validation", clip_started, _timing_point())
        return value

    def _finish_async_checkpoint(self, result: CheckpointWriteResult) -> WindowStepReceipt:
        with self._lock:
            completion = self._pending_checkpoint_completion
            if completion is None:
                if self._checkpoint_state == "durable" and self._last_receipt is not None:
                    target = self._checkpoint_intent
                    if (target is None or (
                        target["target"]["generation"] == result.generation
                        and target["target"]["snapshot_id"] == result.snapshot_id
                    )):
                        return self._last_receipt
                raise DirtyGradientCoreError("async checkpoint completion has no staged operation")
            expected = completion["target"]
            if (
                result.generation != expected["generation"]
                or result.snapshot_id != expected["snapshot_id"]
            ):
                self._fail_async_checkpoint_locked(
                    RuntimeError("checkpoint writer returned a different target identity")
                )
                raise DirtyGradientCoreError(self._dirty_reason or "async checkpoint identity changed")

            payload_metadata = {
                "policy_version": expected["policy_version"],
                "optimizer_steps": expected["optimizer_steps"],
                "scheduler_steps": expected["scheduler_steps"],
            }
            self._record_checkpoint_write(
                payload_metadata, result,
                timing_record=completion["timing_record"],
                checkpoint_started=completion["checkpoint_started"],
            )
            self.policy_version = expected["policy_version"]
            self.optimizer_steps = expected["optimizer_steps"]
            self.scheduler_steps = expected["scheduler_steps"]
            self._prepared_windows = completion["prepared_windows"]
            self._committed_attempts = completion["committed_attempts"]
            self._committed_windows = completion["committed_windows"]
            self._pending_attempt = None
            self._last_receipt = completion["receipt"]
            self.last_update_context = completion["last_update_context"]
            self._step_callbacks.pop(completion["key"], None)
            self._staged_callbacks.pop(completion["key"], None)
            self._checkpoint_state = "durable"
            self._checkpoint_failure = None
            self._checkpoint_intent = None
            self._pending_checkpoint_completion = None
            self._checkpoint_future = None
            self._checkpoint_finalized.set()

            timing_record = completion["timing_record"]
            try:
                with self._timing_lock:
                    timing_record["policy_version_after"] = self.policy_version
                    timing_record["optimizer_steps_after"] = self.optimizer_steps
                    timing_record["scheduler_steps_after"] = self.scheduler_steps
                    timing_record["gradient_norm_before_clip"] = completion["gradient_norm"]
                    _timing_event(timing_record, "core_checkpoint_committed")
                    _timing_span(
                        timing_record, "core_apply_total",
                        completion["apply_started"], _timing_point(),
                    )
            except Exception as exc:
                self._checkpoint_timing_error = f"{type(exc).__name__}: {exc}"
            cleanup_started = _timing_point()
            try:
                self._remove_control_files()
            except Exception as exc:
                self._checkpoint_cleanup_error = f"control file cleanup: {type(exc).__name__}: {exc}"
            cleanup_finished = _timing_point()
            try:
                with self._timing_lock:
                    _timing_span(timing_record, "core_control_file_cleanup", cleanup_started, cleanup_finished)
            except Exception as exc:
                self._checkpoint_timing_error = f"{type(exc).__name__}: {exc}"
            self.persist_timing_receipt(
                completion["receipt"].window_id,
                completion["receipt"].attempt_id,
            )
            return completion["receipt"]

    def _fail_async_checkpoint_locked(self, exc: BaseException) -> None:
        if self._checkpoint_state == "durable" and self._pending_checkpoint_completion is None:
            return
        self._checkpoint_state = "faulted"
        pointer_state = (
            f", pointer_replaced={exc.pointer_replaced}"
            if isinstance(exc, CheckpointWriteError) else ""
        )
        self._checkpoint_failure = f"{type(exc).__name__}: {exc}{pointer_state}"
        self._dirty = True
        self._dirty_reason = (
            "async central checkpoint failed before durable completion: "
            f"{self._checkpoint_failure}"
        )
        self._checkpoint_future = None
        self._pending_checkpoint_completion = None
        self._checkpoint_finalized.set()

    def _fail_async_checkpoint(self, exc: BaseException) -> None:
        with self._lock:
            self._fail_async_checkpoint_locked(exc)

    def wait_for_durable(self, timeout: float | None = None) -> None:
        """Wait until the staged notification and checkpoint write are finalized."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            future = self._checkpoint_future
            state = self._checkpoint_state
        if future is None and state != "saving":
            self._ensure_clean()
            self._ensure_checkpoint_available()
            return
        if future is not None:
            try:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                future.result(timeout=remaining)
            except TimeoutError:
                raise
            except BaseException as exc:
                self._fail_async_checkpoint(exc)
                raise DirtyGradientCoreError(self._dirty_reason or str(exc)) from exc
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if not self._checkpoint_finalized.wait(remaining):
            raise TimeoutError("timed out waiting for central checkpoint finalization")
        with self._lock:
            if self._dirty or self._checkpoint_state != "durable":
                raise DirtyGradientCoreError(
                    self._dirty_reason or f"checkpoint is in {self._checkpoint_state} state"
                )

    def close(self) -> None:
        """Drain staged notification and the owned checkpoint writer before shutdown."""
        with self._lock:
            self._gradient_preparation = None
        writer = self._checkpoint_writer
        if writer is None:
            return
        with self._lock:
            state = self._checkpoint_state
            future = self._checkpoint_future
        failure: BaseException | None = None
        if state == "saving":
            if future is not None:
                try:
                    future.result()
                except BaseException as exc:
                    self._fail_async_checkpoint(exc)
                    failure = exc
            self._checkpoint_finalized.wait()
        writer.close(wait=True)
        if failure is not None or self._checkpoint_state == "faulted":
            raise DirtyGradientCoreError(
                self._dirty_reason or f"checkpoint writer failed: {failure}"
            ) from failure

    def apply_window(
        self, window: PreparedWindow, contributions: Sequence[GradientContribution], *,
        before_optimizer_step: Callable[[], None] | None = None,
    ) -> WindowStepReceipt:
        return self._apply_window(
            window, contributions, before_optimizer_step=before_optimizer_step,
        )

    def apply_prepared_window(
        self, window: PreparedWindow, preparation: PreparedGradientWindow, *,
        before_optimizer_step: Callable[[], None] | None = None,
    ) -> WindowStepReceipt:
        if not isinstance(preparation, PreparedGradientWindow):
            raise GradientContractError("apply_prepared_window requires a Core-owned preparation")
        return self._apply_window(
            window, (), preparation=preparation, before_optimizer_step=before_optimizer_step,
        )

    def _apply_window(
        self, window: PreparedWindow, contributions: Sequence[GradientContribution], *,
        preparation: PreparedGradientWindow | None = None,
        before_optimizer_step: Callable[[], None] | None = None,
    ) -> WindowStepReceipt:
        apply_started = _timing_point()
        future: Future[CheckpointWriteResult] | None = None
        staged_callback: Callable[[Mapping[str, Any]], None] | None = None
        staged_notice: Mapping[str, Any] | None = None
        with self._lock:
            self._ensure_clean()
            self._ensure_checkpoint_available()
            key, fingerprint = self._prepared_identity(window)
            prior = self._committed_attempts.get(key)
            if prior is not None:
                if prior[0] != fingerprint:
                    raise GradientContractError("committed attempt was reused with changed metadata")
                return prior[1]
            if window.policy_version != self.policy_version:
                raise GradientContractError("prepared window policy version is stale")
            timing_record = self._timing_record(window.window_id, window.attempt_id)
            _timing_event(timing_record, "core_apply_started", apply_started)
            if preparation is None:
                if self._gradient_preparation is not None:
                    raise GradientContractError("a prepared gradient window must be applied or discarded first")
                validation_started = _timing_point()
                checked = self._validate_contributions(window, contributions)
                validation_finished = _timing_point()
                _timing_span(timing_record, "contribution_validate", validation_started, validation_finished)
                gradients = self._sum_contributions(window, checked)
                _timing_span(timing_record, "gradient_aggregate", validation_finished, _timing_point())
            else:
                current = self._gradient_preparation
                if (current is None or current.token is not preparation
                        or current.key != key or current.fingerprint != fingerprint):
                    raise GradientContractError("gradient preparation does not own this Core/window")
                gradients = current.gradients
                self._gradient_preparation = None
            try:
                norm = self._attach_and_clip_gradients(gradients, timing_record=timing_record)
            except Exception:
                for parameter in self._parameters.values():
                    parameter.grad = None
                raise
            pending = {
                "window_id": window.window_id,
                "attempt_id": window.attempt_id,
                "fingerprint": fingerprint,
            }
            try:
                fence_started = _timing_point()
                self._write_pending_fence(pending)
                fence_finished = _timing_point()
                _timing_span(timing_record, "optimizer_write_ahead_fence", fence_started, fence_finished)
                self._pending_attempt = pending
            except Exception as exc:
                for parameter in self._parameters.values():
                    parameter.grad = None
                self._checkpoint_state = "faulted"
                self._checkpoint_failure = f"{type(exc).__name__}: {exc}"
                self._mark_dirty(f"write-ahead checkpoint failed: {type(exc).__name__}: {exc}")
                raise DirtyGradientCoreError(self._dirty_reason) from exc

            try:
                callback = before_optimizer_step or self._step_callbacks.get(key) or self._before_optimizer_step
                callback_started = _timing_point()
                if callback is not None:
                    callback()
                callback_finished = _timing_point()
                _timing_span(timing_record, "before_optimizer_callback", callback_started, callback_finished)
                optimizer_started = _timing_point()
                self.optimizer.step()
                optimizer_finished = _timing_point()
                _timing_span(timing_record, "cpu_adamw_step", optimizer_started, optimizer_finished)
                scheduler_started = _timing_point()
                self.scheduler.step()
                scheduler_finished = _timing_point()
                _timing_span(timing_record, "scheduler_step", scheduler_started, scheduler_finished)
                finite_started = _timing_point()
                for name, parameter in self._parameters.items():
                    if not bool(torch.isfinite(parameter).all()):
                        raise FloatingPointError(f"AdamW produced non-finite parameter: {name}")
                finite_finished = _timing_point()
                _timing_span(timing_record, "post_step_parameter_finite_scan", finite_started, finite_finished)
                zero_grad_started = _timing_point()
                self.optimizer.zero_grad(set_to_none=True)
                zero_grad_finished = _timing_point()
                _timing_span(timing_record, "optimizer_zero_grad", zero_grad_started, zero_grad_finished)

                snapshot_started = _timing_point()
                trainable_state = self._capture_trainable_state()
                snapshot_finished = _timing_point()
                _timing_span(timing_record, "update_snapshot_capture", snapshot_started, snapshot_finished)

                next_policy = self.policy_version + 1
                next_optimizer_steps = self.optimizer_steps + 1
                next_scheduler_steps = self.scheduler_steps + 1
                next_prepared = dict(self._prepared_windows)
                next_prepared.pop(key, None)
                next_committed = dict(self._committed_attempts)
                next_windows = dict(self._committed_windows)
                next_windows[window.window_id] = window.attempt_id
                generation = self._checkpoint_generation + 1
                receipt = WindowStepReceipt(
                    _RECEIPT_SCHEMA, window.window_id, window.attempt_id,
                    self.policy_version, next_policy, next_optimizer_steps, next_scheduler_steps,
                    window.n_window_tokens, len(window.jobs), norm, self.max_grad_norm, generation,
                )
                next_committed[key] = (fingerprint, receipt)

                if not self.async_checkpoint:
                    written = self._save_checkpoint(
                        timing_record=timing_record,
                        pending_attempt=None,
                        policy_version=next_policy,
                        optimizer_steps=next_optimizer_steps,
                        scheduler_steps=next_scheduler_steps,
                        prepared_windows=next_prepared,
                        committed_attempts=next_committed,
                        committed_windows=next_windows,
                        last_receipt=receipt,
                        last_update_context=window.update_context,
                        checkpoint_reason="central_optimizer_step",
                        trainable_state=trainable_state,
                    )
                    if written != generation:
                        raise GradientCheckpointError("completed checkpoint generation changed")
                    self.policy_version = next_policy
                    self.optimizer_steps = next_optimizer_steps
                    self.scheduler_steps = next_scheduler_steps
                    self._prepared_windows = next_prepared
                    self._committed_attempts = next_committed
                    self._committed_windows = next_windows
                    self._pending_attempt = None
                    self._last_receipt = receipt
                    self.last_update_context = copy.deepcopy(dict(window.update_context or {}))
                    self._step_callbacks.pop(key, None)
                    self._staged_callbacks.pop(key, None)
                    cleanup_started = _timing_point()
                    self._remove_control_files()
                    cleanup_finished = _timing_point()
                    _timing_span(timing_record, "core_control_file_cleanup", cleanup_started, cleanup_finished)
                    timing_record["policy_version_after"] = next_policy
                    timing_record["optimizer_steps_after"] = next_optimizer_steps
                    timing_record["scheduler_steps_after"] = next_scheduler_steps
                    timing_record["gradient_norm_before_clip"] = norm
                    _timing_event(timing_record, "core_checkpoint_committed")
                    _timing_span(timing_record, "core_apply_total", apply_started, _timing_point())
                    self.persist_timing_receipt(window.window_id, window.attempt_id)
                    return receipt

                target_snapshot_id = uuid4().hex
                checkpoint_started = _timing_point()
                payload, _ = self._materialize_checkpoint_payload(
                    timing_record=timing_record,
                    snapshot_id=target_snapshot_id,
                    pending_attempt=None,
                    policy_version=next_policy,
                    optimizer_steps=next_optimizer_steps,
                    scheduler_steps=next_scheduler_steps,
                    prepared_windows=next_prepared,
                    committed_attempts=next_committed,
                    committed_windows=next_windows,
                    last_receipt=receipt,
                    last_update_context=window.update_context,
                    checkpoint_reason="central_optimizer_step",
                    trainable_state=trainable_state,
                )
                if payload["generation"] != generation:
                    raise GradientCheckpointError("staged checkpoint generation changed")
                intent = self._write_async_checkpoint_intent(
                    window=window,
                    fingerprint=fingerprint,
                    target_generation=generation,
                    target_snapshot_id=target_snapshot_id,
                    target_policy_version=next_policy,
                    target_optimizer_steps=next_optimizer_steps,
                    target_scheduler_steps=next_scheduler_steps,
                )
                self.policy_version = next_policy
                self.optimizer_steps = next_optimizer_steps
                self.scheduler_steps = next_scheduler_steps
                self._checkpoint_state = "saving"
                self._checkpoint_failure = None
                self._checkpoint_notification_error = None
                self._checkpoint_finalized.clear()
                self._pending_checkpoint_completion = {
                    "key": key,
                    "target": dict(intent["target"]),
                    "receipt": receipt,
                    "prepared_windows": next_prepared,
                    "committed_attempts": next_committed,
                    "committed_windows": next_windows,
                    "last_update_context": copy.deepcopy(dict(window.update_context or {})),
                    "timing_record": timing_record,
                    "checkpoint_started": checkpoint_started,
                    "apply_started": apply_started,
                    "gradient_norm": norm,
                }
                staged_callback = self._staged_callbacks.pop(key, None)
                staged_notice = MappingProxyType({
                    "schema": "central-gradient-checkpoint-staged/v1",
                    "window_id": window.window_id,
                    "attempt_id": window.attempt_id,
                    "fingerprint": fingerprint,
                    "checkpoint_state": "saving",
                    "source_generation": intent["source"]["generation"],
                    "source_snapshot_id": intent["source"]["snapshot_id"],
                    "durable_policy_version": self.durable_policy_version,
                    "target_generation": intent["target"]["generation"],
                    "target_snapshot_id": intent["target"]["snapshot_id"],
                    "policy_version": intent["target"]["policy_version"],
                    "optimizer_steps": intent["target"]["optimizer_steps"],
                    "scheduler_steps": intent["target"]["scheduler_steps"],
                    "trainable_state_snapshot": MappingProxyType(trainable_state),
                })
            except Exception as exc:
                for parameter in self._parameters.values():
                    parameter.grad = None
                if not (self._dirty and self._checkpoint_state == "faulted"):
                    self._checkpoint_state = "faulted"
                    self._checkpoint_failure = f"{type(exc).__name__}: {exc}"
                    self._mark_dirty(f"optimizer/checkpoint boundary failed: {type(exc).__name__}: {exc}")
                raise DirtyGradientCoreError(self._dirty_reason or str(exc)) from exc

        # Give the latest-LoRA publication handler priority over full checkpoint I/O.
        # The callback may wait for rollout ACK/prefetch but never holds the core lock.
        if staged_callback is not None and staged_notice is not None:
            try:
                staged_callback(staged_notice)
            except BaseException as exc:  # noqa: BLE001 -- notification cannot cancel checkpoint durability
                with self._lock:
                    self._checkpoint_notification_error = f"{type(exc).__name__}: {exc}"
        try:
            if self._checkpoint_state != "saving" or self._pending_checkpoint_completion is None:
                raise DirtyGradientCoreError("staged checkpoint state changed before writer submission")
            if self._checkpoint_writer is None:
                raise RuntimeError("async checkpoint writer is unavailable")

            def register_future(registered: Future[CheckpointWriteResult]) -> None:
                with self._lock:
                    if (self._checkpoint_state != "saving"
                            or self._pending_checkpoint_completion is None):
                        raise DirtyGradientCoreError("staged checkpoint state changed before writer submission")
                    self._checkpoint_future = registered

            future = self._checkpoint_writer.submit(
                self.root, payload, on_reserved=register_future,
            )
        except Exception as exc:
            self._fail_async_checkpoint(exc)
            raise DirtyGradientCoreError(self._dirty_reason or str(exc)) from exc
        try:
            result = future.result()
        except BaseException as exc:
            self._fail_async_checkpoint(exc)
            raise DirtyGradientCoreError(self._dirty_reason or str(exc)) from exc
        return self._finish_async_checkpoint(result)

    def result_receipt(self, window_id: str, attempt_id: str) -> WindowStepReceipt | None:
        with self._lock:
            entry = self._committed_attempts.get(_attempt_key(window_id, attempt_id))
            return None if entry is None else entry[1]

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        with self._lock:
            self._ensure_clean()
            if self._adapter_state_factory is None:
                return {name: parameter.detach().cpu().clone() for name, parameter in self._parameters.items()}
            value = self._adapter_state_factory(self.model)
            if not isinstance(value, Mapping) or not value:
                raise ValueError("adapter_state_factory must return non-empty state")
            result = {str(name): tensor.detach().cpu().clone() for name, tensor in value.items()}
            if any("lora_" not in name or not isinstance(tensor, torch.Tensor) for name, tensor in result.items()):
                raise ValueError("adapter_state_factory must return LoRA tensors only")
            return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            window = next(iter(self._prepared_windows.values()), None)
            cleanup_error = self._checkpoint_cleanup_error
            if cleanup_error is None and self._checkpoint_writer is not None:
                cleanup_error = self._checkpoint_writer.maintenance_error
            return {
                "schema": "central-gradient-status/v1",
                "enabled": True,
                "ready": not self._dirty,
                "dirty": self._dirty,
                "dirty_reason": self._dirty_reason,
                "resumed": self.resumed,
                "async_checkpoint_enabled": self.async_checkpoint,
                "checkpoint_state": self._checkpoint_state,
                "can_optimize": (
                    not self._dirty and self._checkpoint_state == "durable"
                    and self._checkpoint_future is None
                ),
                "can_publish": (
                    not self._dirty and self._checkpoint_state in {"durable", "saving"}
                ),
                "durable_policy_version": self.durable_policy_version,
                "durable_optimizer_steps": self.durable_optimizer_steps,
                "durable_scheduler_steps": self.durable_scheduler_steps,
                "checkpoint_intent": copy.deepcopy(self._checkpoint_intent),
                "checkpoint_failure": self._checkpoint_failure,
                "checkpoint_cleanup_error": cleanup_error,
                "checkpoint_notification_error": self._checkpoint_notification_error,
                "checkpoint_timing_error": self._checkpoint_timing_error,
                "lineage_id": self.lineage_id,
                "engine_epoch": self.engine_epoch,
                "snapshot_id": self.snapshot_id,
                "checkpoint_id": self.snapshot_id,
                "checkpoint_path": self.checkpoint_path,
                "last_update_context": self.last_update_context,
                "initial_policy_version": self.initial_policy_version,
                "policy_version": self.policy_version,
                "optimizer_steps": self.optimizer_steps,
                "scheduler_steps": self.scheduler_steps,
                "checkpoint_generation": self._checkpoint_generation,
                "parameter_layout": [item.to_dict() for item in self.parameter_layout],
                "layout_fingerprint": self.layout_fingerprint,
                "pending_attempt": self._pending_attempt,
                "pending_window": None if window is None else {
                    "window_id": window.window_id, "attempt_id": window.attempt_id,
                    "policy_version": window.policy_version, "n_window_tokens": window.n_window_tokens,
                    "job_count": len(window.jobs), "work_order": list(window.work_order),
                },
                "ready_jobs": 0 if window is None else len(window.jobs),
                "committed_windows": len(self._committed_windows),
                "cancelled_windows": [
                    dict(record) for _, record in sorted(self._cancelled_windows.items())
                ],
                "last_cancellation": (
                    dict(max(self._cancelled_windows.values(), key=lambda item: item["cancelled_at_ns"]))
                    if self._cancelled_windows else None
                ),
                "last_receipt": None if self._last_receipt is None else self._last_receipt.to_dict(),
            }
