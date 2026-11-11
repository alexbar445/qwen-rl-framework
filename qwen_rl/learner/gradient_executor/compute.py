"""Single-process TRL DAPO gradient computation for asynchronous Learners."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_COMPUTE_PROFILE_KEYS = {"base_artifact_id", "loss_profile", "correction_cap"}
_LOSS_PROFILE = "dapo_global_token_mean_beta0_v1"


class GradientComputeExecutor:
    """Own one fixed T2 model and return gradients without changing its policy."""

    def can_help(self) -> bool:
        # Dedicated compute has no paired Worker residency to interrupt.
        return True

    def __init__(self, engine: Any):
        trainer = getattr(engine, "trainer", None)
        if trainer is None or not getattr(trainer, "gradient_only", False):
            raise ValueError("gradient executor requires a gradient-only LearnerGRPOTrainer")
        if getattr(engine, "recovery_store", None) is not None:
            raise ValueError("gradient executor cannot own or restore a Trainer optimizer lineage")
        if getattr(engine, "config", {}).get("recovery_dir") is not None:
            raise ValueError("gradient executor config must not select recovery_dir")
        self.engine = engine
        self.trainer = trainer
        self._lock = threading.RLock()
        self._installed_policy_version: int | None = None
        self._validate_loss_profile()
        self._compute_profile = self._make_compute_profile()

    @property
    def packed_weight_offloader(self):
        return getattr(self.engine, "packed_weight_offloader", None)

    @property
    def current_policy_version(self) -> int | None:
        return self._installed_policy_version

    @property
    def policy_version(self) -> int | None:
        return self.current_policy_version

    @property
    def compute_profile(self) -> dict[str, Any]:
        return dict(self._compute_profile)

    def _validate_loss_profile(self) -> None:
        trainer = self.trainer
        if (
            getattr(trainer, "loss_type", None) != "dapo"
            or float(getattr(trainer, "beta", float("nan"))) != 0.0
            or float(getattr(trainer, "epsilon_low", float("nan"))) != 0.2
            or float(getattr(trainer, "epsilon_high", float("nan"))) != 0.28
            or float(getattr(trainer, "top_entropy_quantile", float("nan"))) != 1.0
            or bool(getattr(trainer, "aux_loss_enabled", True))
            or bool(getattr(trainer, "_entropy_bonus_enabled", True))
            or getattr(trainer, "importance_sampling_level", None) != "token"
            or bool(getattr(trainer.args, "average_tokens_across_devices", True))
            or not bool(getattr(trainer, "model_accepts_loss_kwargs", False))
        ):
            raise ValueError("gradient executor Trainer does not match the supported DAPO loss profile")

    def _make_compute_profile(self) -> dict[str, Any]:
        identity = getattr(self.engine, "_ternary_identity", None)
        artifact_id = None
        if isinstance(identity, Mapping):
            artifact_id = (
                identity.get("source", {})
                .get("ninfer_artifact", {})
                .get("artifact_id")
            )
        if not isinstance(artifact_id, str) or len(artifact_id) != 32:
            raise ValueError("gradient executor requires the loaded T2 NInfer artifact identity")
        if any(character not in "0123456789abcdef" for character in artifact_id):
            raise ValueError("loaded T2 artifact ID is malformed")
        cap = float(self.trainer.importance_sampling_cap)
        if not math.isfinite(cap) or cap < 1.0:
            raise ValueError("gradient executor correction cap is invalid")
        return {
            "base_artifact_id": artifact_id,
            "loss_profile": _LOSS_PROFILE,
            "correction_cap": cap,
        }

    def _validate_compute_profile(self, supplied: Mapping[str, Any], prepared_window, job) -> None:
        from qwen_rl.contracts.gradients import compute_profile_fingerprint

        if not isinstance(supplied, Mapping) or set(supplied) != _COMPUTE_PROFILE_KEYS:
            raise ValueError("compute profile must contain exactly base_artifact_id, loss_profile and correction_cap")
        supplied_cap = supplied["correction_cap"]
        if isinstance(supplied_cap, bool) or not isinstance(supplied_cap, (int, float)):
            raise ValueError("compute profile correction_cap must be numeric")
        normalized = {
            "base_artifact_id": supplied["base_artifact_id"],
            "loss_profile": supplied["loss_profile"],
            "correction_cap": float(supplied_cap),
        }
        if normalized != self._compute_profile:
            raise ValueError("central compute profile differs from the loaded T2 executor")
        window_profile = getattr(prepared_window, "compute_profile", None)
        if window_profile is not None and dict(window_profile) != normalized:
            raise ValueError("prepared window compute profile differs from the request profile")
        if dict(getattr(job, "compute_profile", {})) != normalized:
            raise ValueError("gradient job compute profile differs from the request profile")
        if getattr(job, "compute_profile_fingerprint", None) != compute_profile_fingerprint(normalized):
            raise ValueError("gradient job compute profile fingerprint is invalid")

    def install_adapter_version(
        self,
        version: int,
        adapter_state: Mapping[str, torch.Tensor],
    ) -> None:
        """Copy a caller-owned state, then install it before accepting a job."""
        self._install_adapter_version(version, adapter_state, copy_state=True)

    def _install_owned_adapter_version(self, version, adapter_state):
        """Consume lifecycle's private snapshot while its job lock excludes writes."""
        self._install_adapter_version(version, adapter_state, copy_state=False)

    def _install_adapter_version(self, version, adapter_state, *, copy_state):
        if type(version) is not int or version < 0:
            raise ValueError("adapter version must be a non-negative integer")
        from .adapter_snapshot import _OwnedAdapterSnapshot
        with self._lock:
            if self.trainer._gradient_compute_active:
                raise RuntimeError("cannot replace an adapter while a gradient job is active")
            if copy_state:
                snapshot = _OwnedAdapterSnapshot(version, adapter_state)
            elif isinstance(adapter_state, _OwnedAdapterSnapshot):
                snapshot = adapter_state
            else:
                raise ValueError('owned installation requires a private adapter snapshot')
            if snapshot.version != version:
                raise ValueError('private adapter snapshot version differs from installation')
            adapter_state = snapshot._tensors
            parameters = {
                name: parameter
                for name, parameter in self.engine.model.named_parameters()
                if parameter.requires_grad and "lora_" in name
            }
            non_lora = [
                name for name, parameter in self.engine.model.named_parameters()
                if parameter.requires_grad and "lora_" not in name
            ]
            if non_lora or not parameters:
                raise RuntimeError("loaded T2 executor is not frozen-base LoRA-only")
            if set(adapter_state) != set(parameters):
                raise ValueError("adapter state has an incomplete or unexpected live LoRA parameter inventory")
            staged: dict[str, torch.Tensor] = {}
            for name, parameter in parameters.items():
                value = adapter_state[name]
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or tuple(value.shape) != tuple(parameter.shape)
                        or value.dtype != parameter.dtype):
                    raise ValueError(f"adapter tensor {name!r} differs from the live LoRA parameter")
                staged[name] = value

            self._installed_policy_version = None
            from .adapter_transfer import copy_and_verify
            copy_and_verify(parameters, staged)
            self.engine.policy_version = version
            self._installed_policy_version = version

    def compute_gradient_job(
        self,
        prepared_window,
        job,
        prepared_rows: Sequence[Mapping[str, Any]],
        *,
        compute_profile: Mapping[str, Any],
    ):
        """Compute one job using the requested version and immutable window denominator."""
        with self._lock:
            self._validate_compute_profile(compute_profile, prepared_window, job)
            if self._installed_policy_version is None:
                raise RuntimeError("no adapter version is installed on this executor")
            if self._installed_policy_version != job.policy_version:
                raise ValueError("installed adapter version differs from the gradient job")
            if dict(getattr(job, "compute_profile", {})) != self._compute_profile:
                raise ValueError("gradient job compute profile differs from the loaded T2 executor")
            if int(getattr(self.engine, "policy_version", -1)) != self._installed_policy_version:
                # The engine's ordinary update path is disabled in this mode;
                # this also catches accidental version mutation behind the adapter.
                raise RuntimeError("gradient executor engine version changed outside adapter installation")
            operation_telemetry = getattr(self.engine, "operation_telemetry", None)
            telemetry_started = False
            if operation_telemetry is not None:
                try:
                    telemetry_started = bool(operation_telemetry.begin_job(prepared_window, job))
                except Exception:
                    telemetry_started = False
            offloader = self.packed_weight_offloader
            fla_autotune_memory = None
            if self.engine.config.get("model_backend") == "ternary" and torch.cuda.is_initialized():
                from ..fla_autotune_memory import install_fla_autotune_memory_guard
                fla_autotune_memory = install_fla_autotune_memory_guard()
            try:
                if offloader is not None:
                    torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                result = self.trainer.compute_gradient_contribution(
                    prepared_window,
                    job,
                    prepared_rows,
                    installed_policy_version=self._installed_policy_version,
                    compute_profile=self._compute_profile,
                )
                if offloader is not None:
                    from qwen_rl.runtime.manifest import write_json_atomic

                    # Local observability only: preserve the public contribution
                    # contract and avoid adding model/activation data to the wire.
                    write_json_atomic(
                        Path(self.engine.config["compute_runtime_dir"]) / "last-job-execution.json",
                        {
                            "schema": "managed-t2-gradient-execution/v1",
                            "job_id": job.job_id,
                            "owner_executor_id": job.owner_executor_id,
                            "policy_version": job.policy_version,
                            "gradient_compute_seconds": time.monotonic() - started,
                            "optimizer_owned": False,
                            "fla_autotune_memory": dict(fla_autotune_memory or {}),
                            "gdn_warmup": dict(self.trainer._last_gdn_warmup or {}),
                            "gradient_execution_phases": dict(self.trainer._last_gradient_timing),
                            "packed_weight_offload": offloader.report(),
                            "activation_offload_microbatches": list(self.trainer._offload_microbatch_reports),
                            "torch_allocator_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "torch_allocator_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        },
                    )
                return result
            finally:
                if telemetry_started:
                    try:
                        operation_telemetry.end_job()
                    except Exception:
                        pass

    def shutdown(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


def build_compute(config: dict[str, Any], *, cpu_prepared_model=None) -> GradientComputeExecutor:
    """Build the usual T2 Learner model with a gradient-only Trainer boundary."""
    if not isinstance(config, dict):
        raise TypeError("gradient compute config must be a mapping")
    if config.get("parallelism", "single") != "single":
        raise ValueError("gradient executor currently requires parallelism='single'")
    if config.get("recovery_dir") is not None:
        raise ValueError("gradient executor config must not select recovery_dir")
    if config.get("packed_weight_offload") is not True:
        raise ValueError("gradient executor requires explicitly enabled managed T2 packed-weight offload")
    selected = dict(config)
    selected["gradient_only"] = True
    from .t2_engine import GradientT2Engine

    return GradientComputeExecutor(GradientT2Engine(
        selected, cpu_prepared_model=cpu_prepared_model,
    ))
