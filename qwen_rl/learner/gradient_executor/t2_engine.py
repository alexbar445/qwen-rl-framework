"""Managed single-device T2 assembly for gradient-only executor jobs."""
from __future__ import annotations

from pathlib import Path
from typing import Any


_REQUIRED_T2_GRADIENT_FIELDS = (
    "activation_offload_min_tokens",
    "checkpoint_input_recompute_span",
    "packed_weight_offload",
    "packed_weight_offload_max_cpu_master_bytes",
    "packed_weight_offload_max_pinned_bytes",
    "packed_weight_offload_prefetch",
    "packed_weight_offload_store_dir",
)


def _validate_output_head_metadata(head: Any, *, hidden_size: int, vocab_size: int) -> None:
    """Validate the packed head without evaluating its dense ``weight`` property."""
    if (head is None or not callable(getattr(head, "forward", None))
            or getattr(head, "in_features", None) != hidden_size
            or getattr(head, "out_features", None) != vocab_size):
        raise RuntimeError("assembled T2 output head has invalid projection metadata")


class GradientT2Engine:
    """One current T2 model, gradient-only Trainer, and the formal packed streamer.

    Model and evidence construction are shared with RankService DDP. This engine
    owns neither a process group nor an optimizer and installs the same
    identity-bound packed offloader used by the formal rank service.
    """

    def __init__(self, config: dict[str, Any], *, cpu_prepared_model=None):
        if not isinstance(config, dict):
            raise TypeError("GradientT2Engine config must be a dict")
        missing = [key for key in _REQUIRED_T2_GRADIENT_FIELDS if key not in config]
        if missing:
            raise ValueError(f"gradient T2 profile is missing required field(s): {missing}")
        if config.get("model_backend") != "ternary":
            raise ValueError("GradientT2Engine requires model_backend='ternary'")
        if config.get("parallelism") != "single" or config.get("gradient_only") is not True:
            raise ValueError("GradientT2Engine requires single-process gradient-only configuration")
        if config.get("recovery_dir") is not None:
            raise ValueError("GradientT2Engine cannot own a training recovery lineage")
        if config["packed_weight_offload"] is not True:
            raise ValueError("GradientT2Engine requires packed_weight_offload=true")
        if config.get("activation_offload") is not True:
            raise ValueError("GradientT2Engine requires activation_offload=true")
        if type(config.get("gradient_checkpointing")) is not bool:
            raise ValueError("GradientT2Engine requires an explicit gradient_checkpointing boolean")
        if not config["gradient_checkpointing"] and config.get("activation_cache_parameter_names"):
            raise ValueError("selected-output caching requires gradient checkpointing")
        if type(config["activation_offload_min_tokens"]) is not int or config["activation_offload_min_tokens"] < 0:
            raise ValueError("activation_offload_min_tokens must be an integer >= 0")
        span = config["checkpoint_input_recompute_span"]
        if type(span) is not int or span != 1:
            raise ValueError("checkpoint_input_recompute_span must be 1")
        if (type(config["packed_weight_offload_max_cpu_master_bytes"]) is not int
                or config["packed_weight_offload_max_cpu_master_bytes"] <= 0):
            raise ValueError("packed_weight_offload_max_cpu_master_bytes must be a positive integer")
        if (type(config["packed_weight_offload_max_pinned_bytes"]) is not int
                or config["packed_weight_offload_max_pinned_bytes"] <= 0):
            raise ValueError("packed_weight_offload_max_pinned_bytes must be a positive integer")
        if type(config["packed_weight_offload_prefetch"]) is not bool:
            raise ValueError("packed_weight_offload_prefetch must be a boolean")
        if not isinstance(config["packed_weight_offload_store_dir"], str) or not config[
            "packed_weight_offload_store_dir"
        ].strip():
            raise ValueError("packed_weight_offload_store_dir must be an explicit absolute path")
        store_root = Path(config["packed_weight_offload_store_dir"]).expanduser()
        if not store_root.is_absolute() or store_root.is_symlink():
            raise ValueError("packed_weight_offload_store_dir must be an absolute non-symlink directory")
        for name in ("artifact_identity_path", "artifact_identity", "tokenizer_path", "model_path"):
            if name not in config:
                raise ValueError(f"GradientT2Engine requires {name}")

        from qwen_rl.contracts.ternary import validate_artifact_identity

        self.config = dict(config)
        self.model_path = str(config["model_path"])
        self.policy_version = 0
        self.recovery_store = None
        self.gradient_only = True
        self._ternary_identity = validate_artifact_identity(config["artifact_identity"])
        self._packed_weight_store = None
        self._packed_weight_offloader = None
        self._nested_decoder_checkpoints = None
        self._frozen_embedding_input_policy = None
        self._packed_weight_hook_handles: list[Any] = []
        self._head_bridge = None
        self._ternary_regional_compile = None
        self.operation_telemetry = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._compile_cache_monitor = None
        try:
            from .compile_cache import CompileCacheMonitor

            self._compile_cache_monitor = CompileCacheMonitor(config["compute_runtime_dir"])
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "compile-cache monitor initialization failed; continuing T2 executor initialization"
            )

        import os
        import torch

        # Must precede the first CUDA BLAS operation in this executor process.
        # Strict deterministic backward also guards its BF16 LoRA GEMMs.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        if not torch.cuda.is_available():
            raise RuntimeError("GradientT2Engine requires CUDA")
        torch.cuda.set_device(0)
        self.device = torch.device("cuda", torch.cuda.current_device())

        if cpu_prepared_model is None:
            from qwen_rl.learner.parallel.t2_model_assembly import load_t2_model

            self.model, self.tokenizer, self.evidence = load_t2_model(
                self.config, device=self.device, parallelism="single", world_size=1,
            )
        else:
            if (not isinstance(cpu_prepared_model, tuple) or len(cpu_prepared_model) != 3
                    or cpu_prepared_model[0] is None):
                raise ValueError("CPU-prepared T2 model must contain model, tokenizer and evidence")
            self.model, self.tokenizer, self.evidence = cpu_prepared_model
            import torch
            devices = {parameter.device.type for parameter in self.model.parameters()}
            devices.update(buffer.device.type for buffer in self.model.buffers())
            if devices != {"cpu"}:
                raise RuntimeError("CPU-prepared T2 model contains non-CPU tensors")
            from qwen_rl.models.ternary_blas import configure_ternary_blas
            from qwen_rl.models.ternary_rms_norm import install_bounded_qwen_rms_norms
            from qwen_rl.models.ternary_gdn_execution import CompactGDNExecution
            from qwen_rl.models.ternary_lora_scale import install_identity_scale_lora
            from qwen_rl.models.ternary_fused_hadamard import kernel_build_info

            self.evidence = dict(self.evidence)
            self.evidence["blas"] = configure_ternary_blas(self.device)
            self.evidence["hadamard"] = dict(self.evidence["hadamard"])
            self.evidence["hadamard"]["build"] = kernel_build_info(check_device=True)
            self.model.to(device=self.device)
            torch.cuda.synchronize(self.device)
            self.evidence["rms_norm"] = install_bounded_qwen_rms_norms(self.model)
            get_base = getattr(self.model, "get_base_model", None)
            base_model = get_base() if callable(get_base) else self.model
            execution = CompactGDNExecution(base_model.model.layers)
            base_model._t2_gdn_execution = execution
            self.evidence["gdn_execution"] = execution.report
            basis_counts = self.evidence.get("lora_input_basis_counts")
            if not isinstance(basis_counts, dict) or not basis_counts:
                raise RuntimeError("CPU-prepared T2 model is missing its LoRA basis inventory")
            self.evidence["lora_identity_scale"] = install_identity_scale_lora(
                self.model, expected_count=sum(basis_counts.values()),
            )
            self.evidence["cpu_preload_materialized_on_cuda"] = True
        self.model._t2_managed_assembly = True
        # The rank service owns checkpoint setup through Trainer. The managed
        # assembly marker allows the same single-device span policy validation.
        self.model._checkpoint_input_recompute_span = span
        get_base = getattr(self.model, "get_base_model", None)
        self._base_model = get_base() if callable(get_base) else self.model
        self._backbone = getattr(self._base_model, "model", None)
        if self._backbone is None:
            raise RuntimeError("assembled T2 model has no Qwen3.5 text backbone")
        self._head = self._base_model.get_output_embeddings()
        geometry = self._ternary_identity["model"]["geometry"]
        _validate_output_head_metadata(
            self._head, hidden_size=int(geometry["hidden_size"]),
            vocab_size=int(geometry["vocab_size"]),
        )

        from qwen_rl.learner.parallel.t2_packed_offload import (
            install_t2_frozen_embedding_input_policy,
        )

        if not config["gradient_checkpointing"]:
            self.model.gradient_checkpointing_disable()
            gdn = getattr(self.model, "_t2_gdn_execution", None)
            if gdn is not None:
                gdn.set_checkpointing(False)
                self.evidence["gdn_execution"] = gdn.report
        self._frozen_embedding_input_policy = install_t2_frozen_embedding_input_policy(
            self.model, enabled=True,
        )
        from qwen_rl.learner.bounded_offload import DEFAULT_MAX_PINNED_BYTES
        from qwen_rl.learner.trainer import LearnerGRPOTrainer

        activation_pinned_cap = config.get(
            "activation_offload_max_pinned_bytes", DEFAULT_MAX_PINNED_BYTES,
        )
        self.trainer = LearnerGRPOTrainer(
            model=self.model,
            constrained_decoding=config.get("constrained_decoding",False),
            processing_class=self.tokenizer,
            group_size=int(config.get("group_size", 3)),
            groups_per_update=int(config.get("groups_per_update", 2)),
            microbatch_size=int(config.get("microbatch_size", 2)),
            max_padded_tokens_per_microbatch=config.get("max_padded_tokens_per_microbatch"),
            learning_rate=float(config.get("lr", 1e-6)),
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            max_sequence_length=int(config.get("max_sequence_length", 65536)),
            weight_decay=float(config.get("weight_decay", 0.0)),
            adam_betas=tuple(config.get("adam_betas", (0.9, 0.999))),
            adam_eps=float(config.get("adam_eps", 1e-8)),
            output_dir=str(Path(config.get("compute_runtime_dir", ".")) / "trainer"),
            importance_sampling_cap=float(config.get("importance_sampling_cap", 3.0)),
            activation_offload=True,
            activation_offload_min_tokens=config["activation_offload_min_tokens"],
            checkpoint_input_recompute_span=span,
            activation_offload_pin_memory=False,
            activation_offload_max_pinned_bytes=activation_pinned_cap,
            activation_cache_parameter_names=config.get("activation_cache_parameter_names", ()),
            activation_cache_cpu_bytes=config.get("activation_cache_cpu_bytes", 2 * 1024**3),
            activation_cache_staging_bytes=config.get("activation_cache_staging_bytes", 512 * 1024**2),
            gradient_checkpointing=config["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_only=True,
            deterministic_training=True,
            parallel_mode="single",
            parallel_world_size=1,
            parallel_rank=0,
        )
        if not bool(getattr(self.trainer, "gradient_only", False)):
            raise RuntimeError("GradientT2Engine Trainer did not enter gradient-only mode")
        if getattr(self.trainer, "optimizer", None) is not None:
            raise RuntimeError("GradientT2Engine must not create an optimizer")

        from qwen_rl.learner.parallel.qwen35_pipeline import install_trl_fused_head_bridge

        try:
            self._head_bridge = install_trl_fused_head_bridge(self.model)
            self._install_packed_weight_offloader(store_root)
            from qwen_rl.models.ternary_compile import configure_ternary_regional_compile

            self._ternary_regional_compile = configure_ternary_regional_compile(
                self.model,
                enabled=config.get("ternary_regional_compile", True),
            )
        except BaseException:
            try:
                self.shutdown()
            except Exception:
                pass
            raise
        try:
            self.evidence.update({
                "optimizer": "none; gradient-only executor returns DAPO contributions",
                "trainer": "TRL GRPOTrainer.compute_loss + Accelerator.backward; no optimizer step",
                "parallelism": "single",
                "world_size": 1,
                "process_group": "none",
                "gradient_checkpointing": config["gradient_checkpointing"],
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "backward_determinism": {
                    "enabled": True, "warn_only": False,
                    "scope": "exclusive managed executor training microbatch; restored afterwards",
                    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                    "efficient_sdpa_key_splits": 1,
                },
                "activation_offload": True,
                "activation_offload_min_tokens": config["activation_offload_min_tokens"],
                "checkpoint_input_recompute_span": span,
                "frozen_embedding_input_policy": (
                    None if self._frozen_embedding_input_policy is None
                    else self._frozen_embedding_input_policy.report
                ),
                "regional_compile": (
                    self._ternary_regional_compile.evidence
                    if self._ternary_regional_compile is not None else {"enabled": False}
                ),
            })
            self._write_engine_evidence()
            try:
                from qwen_rl.learner.execution_telemetry import OperationTelemetry

                try:
                    gpu_uuid = str(torch.cuda.get_device_properties(self.device).uuid)
                except Exception:
                    gpu_uuid = None
                self.operation_telemetry = OperationTelemetry(
                    config["compute_runtime_dir"], gpu_uuid=gpu_uuid,
                )
                self.trainer._operation_telemetry = self.operation_telemetry
            except Exception:
                # Dashboard telemetry is optional; it cannot make an otherwise
                # valid compute engine fail to start.
                self.operation_telemetry = None
        except BaseException:
            try:
                self.shutdown()
            except Exception:
                pass
            raise

    @property
    def packed_weight_offloader(self):
        return self._packed_weight_offloader

    def _install_packed_weight_offloader(self, store_root: Path) -> None:
        from qwen_rl.learner.parallel.t2_packed_offload import (
            build_t2_packed_weight_layout,
            install_t2_packed_offloader,
            open_t2_shared_packed_store,
            packed_weight_phase_hooks,
            protect_preplaced_model,
            t2_packed_offload_report,
            wrap_packed_weight_backward,
        )

        layout = build_t2_packed_weight_layout(
            self.model,
            device=self.device,
            artifact_identity=self._ternary_identity,
            max_cpu_master_bytes=self.config["packed_weight_offload_max_cpu_master_bytes"],
            max_pinned_bytes=self.config["packed_weight_offload_max_pinned_bytes"],
        )
        store = open_t2_shared_packed_store(layout, store_root)
        offloader = None
        nested = None
        handles: list[Any] = []
        try:
            offloader, nested = install_t2_packed_offloader(
                self.model, layout, store, config=self.config,
            )
            self.trainer.accelerator.device_placement = False
            protect_preplaced_model(self.trainer.accelerator, self.model)
            handles = packed_weight_phase_hooks(self.model, offloader)
            self.trainer.accelerator.backward = wrap_packed_weight_backward(
                self.trainer.accelerator.backward, offloader,
            )
        except BaseException:
            for handle in handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            if nested is not None:
                try:
                    nested.close()
                except Exception:
                    pass
            if offloader is not None:
                try:
                    offloader.close()
                except Exception:
                    pass
            self.model._t2_managed_packed_weights = False
            self.model._t2_packed_weight_offloader = None
            layout.base._t2_packed_weight_offloader = None
            try:
                store.close()
            except Exception:
                pass
            raise
        self._packed_weight_hook_handles = handles
        self._packed_weight_store = store
        self._packed_weight_offloader = offloader
        self._nested_decoder_checkpoints = nested
        self.evidence["packed_weight_offload"] = t2_packed_offload_report(
            layout,
            store,
            offloader,
            nested,
            mode="single_gradient_only_shared_cpu_master",
            max_pinned_bytes=self.config["packed_weight_offload_max_pinned_bytes"],
            retained_device_components=(
                "input_hadamard_signs", "bias", "token_embedding", "output_head", "LoRA",
            ),
        )

    def _write_engine_evidence(self) -> None:
        runtime_dir = self.config.get("compute_runtime_dir")
        if not isinstance(runtime_dir, str) or not runtime_dir.strip():
            raise ValueError("GradientT2Engine requires compute_runtime_dir for its live evidence receipt")
        from qwen_rl.runtime.manifest import write_json_atomic

        source = self._ternary_identity.get("source", {})
        artifact_id = source.get("ninfer_artifact", {}).get("artifact_id")
        payload = {
            "schema_version": "qwen-rl-gradient-t2-engine-evidence/v1",
            "gradient_only": True,
            "optimizer": "none",
            "process_group": "none",
            "model_assembly": {
                "model_backend": "ternary",
                "parallelism": "single",
                "world_size": 1,
                "artifact_id": artifact_id,
                "base_frozen": self.evidence.get("base_frozen"),
                "packed_module_count": self.evidence.get("packed_module_count"),
                "packed_module_input_basis_counts": self.evidence.get(
                    "packed_module_input_basis_counts"
                ),
                "adapter_trainable_parameters": self.evidence.get("adapter_trainable_parameters"),
                "lora_master_dtype": self.evidence.get("lora_master_dtype"),
                "lora_input_basis_counts": self.evidence.get("lora_input_basis_counts"),
                "lora_input_basis_by_module": self.evidence.get("lora_input_basis_by_module"),
                "output_head": self.evidence.get("output_head"),
                "hadamard": self.evidence.get("hadamard"),
            },
            "checkpointing": {
                "gradient_checkpointing": self.config["gradient_checkpointing"],
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "activation_offload": True,
                "activation_offload_min_tokens": self.config["activation_offload_min_tokens"],
                "checkpoint_input_recompute_span": self.config["checkpoint_input_recompute_span"],
            },
            "frozen_embedding_input_policy": self.evidence.get("frozen_embedding_input_policy"),
            "packed_weight_offload": self.evidence.get("packed_weight_offload"),
            "regional_compile": self.evidence.get("regional_compile"),
            "backward_determinism": self.evidence.get("backward_determinism"),
            "gdn_execution": self.evidence.get("gdn_execution"),
        }
        import json

        encoded = json.dumps(payload, allow_nan=False)
        if not encoded:
            raise ValueError("GradientT2Engine evidence receipt is empty")
        self.engine_evidence_path = Path(runtime_dir).expanduser() / "engine-evidence.json"
        write_json_atomic(self.engine_evidence_path, payload)

    def shutdown(self) -> None:
        if self.trainer is not None and getattr(self.trainer, "_gradient_compute_active", False):
            raise RuntimeError("cannot shut down GradientT2Engine during a gradient job")
        errors = []
        if self._compile_cache_monitor is not None:
            try:
                self._compile_cache_monitor.close()
            except Exception:
                pass
            self._compile_cache_monitor = None
        if self.operation_telemetry is not None:
            try:
                self.operation_telemetry.close()
            except BaseException as exc:
                errors.append(exc)
            self.operation_telemetry = None
        if self._ternary_regional_compile is not None:
            try:
                self._ternary_regional_compile.restore()
            except BaseException as exc:
                errors.append(exc)
            self._ternary_regional_compile = None
        if self._nested_decoder_checkpoints is not None:
            try:
                self._nested_decoder_checkpoints.close()
            except BaseException as exc:
                errors.append(exc)
            self._nested_decoder_checkpoints = None
        for handle in self._packed_weight_hook_handles:
            try:
                handle.remove()
            except BaseException as exc:
                errors.append(exc)
        self._packed_weight_hook_handles.clear()
        if self._packed_weight_offloader is not None:
            try:
                self._packed_weight_offloader.close()
            except BaseException as exc:
                errors.append(exc)
            self._packed_weight_offloader = None
        if self._packed_weight_store is not None:
            try:
                self._packed_weight_store.close()
            except BaseException as exc:
                errors.append(exc)
            self._packed_weight_store = None
        if self._frozen_embedding_input_policy is not None:
            try:
                self._frozen_embedding_input_policy.close()
            except BaseException as exc:
                errors.append(exc)
            self._frozen_embedding_input_policy = None
        if errors:
            raise RuntimeError(
                f"GradientT2Engine shutdown had {len(errors)} cleanup error(s)"
            ) from errors[0]


__all__ = ["GradientT2Engine"]
