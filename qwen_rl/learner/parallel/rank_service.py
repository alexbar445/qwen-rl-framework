"""torchrun entrypoint for persistent local Learner ranks."""
from __future__ import annotations

from ...contracts.activation_offload import DEFAULT_OFFLOAD_MIN_TOKENS

import argparse
import copy
import ctypes
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import socket
import signal
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from typing import Any, Callable

from .transport import receive_message, send_message


class RankServiceError(RuntimeError):
    pass


def _merge_pipeline_peft_configs(rank_configs: list[dict[str, Any]]) -> dict[str, Any]:
    from qwen_rl.learner.parallel.pipeline_exports import merge_pipeline_peft_configs

    try:
        return merge_pipeline_peft_configs(rank_configs)
    except Exception as exc:
        raise RankServiceError(f"cannot merge the four-stage PEFT target inventory: {exc}") from exc


def _cpu_tree(value: Any, *, copy_stats: dict[str, int] | None = None) -> Any:
    """Copy optimizer/RNG structures to ordinary CPU tensors without GPU gather."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - pinned Torch has DTensor
        DTensor = ()  # type: ignore[assignment]
    import torch

    if isinstance(value, DTensor):
        local = value.to_local()
        if copy_stats is not None:
            copy_stats["cpu_tensor_bytes"] = copy_stats.get("cpu_tensor_bytes", 0) + (
                local.numel() * local.element_size()
            )
        return local.detach().to(device="cpu", copy=True)
    if isinstance(value, torch.Tensor):
        if copy_stats is not None:
            copy_stats["cpu_tensor_bytes"] = copy_stats.get("cpu_tensor_bytes", 0) + (
                value.numel() * value.element_size()
            )
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_tree(item, copy_stats=copy_stats) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item, copy_stats=copy_stats) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item, copy_stats=copy_stats) for item in value)
    return copy.deepcopy(value)


def _tensor_payload_bytes(value: Any) -> int:
    """Count tensor storage represented by a nested snapshot without copying it."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_payload_bytes(item) for item in value)
    return 0


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _configure_ddp_ignored_frozen_state(model) -> list[str]:
    """Use upstream DDP's ignore list for the identity-bound frozen base.

    Every rank loads the same frozen artifact independently. DDP synchronizes
    only trainable LoRA parameters; frozen packed weights and buffers stay local.
    """
    from torch.nn.parallel import DistributedDataParallel

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RankServiceError("DDP initialization requires trainable adapter parameters")
    ignored = [
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    ]
    ignored.extend(name for name, _ in model.named_buffers())
    ignored = sorted(set(ignored))
    if not callable(getattr(DistributedDataParallel, "_set_params_and_buffers_to_ignore_for_model", None)):
        raise RankServiceError("pinned PyTorch DDP has no upstream state-ignore configuration")
    DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, ignored)
    return ignored


def _restore_fsdp2_optimizer_moments(optimizer) -> int:
    """Restore local Adam moments as DTensors matching their FSDP2 parameters."""
    import torch

    from torch.distributed.tensor import DTensor, distribute_tensor

    restored = 0
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, DTensor) or not isinstance(state, dict):
            continue
        local_shape = tuple(parameter.to_local().shape)
        for key, value in tuple(state.items()):
            if key == "step":
                if isinstance(value, DTensor):
                    state[key] = value.to_local().detach().clone()
                continue
            if isinstance(value, DTensor) or not isinstance(value, torch.Tensor):
                continue
            if value.ndim == 0:
                continue
            if tuple(value.shape) == local_shape:
                local_value = value.to(device=parameter.device)
                state[key] = DTensor.from_local(
                    local_value,
                    parameter.device_mesh,
                    parameter.placements,
                    run_check=False,
                    shape=parameter.shape,
                    stride=parameter.stride(),
                )
            elif tuple(value.shape) == tuple(parameter.shape):
                state[key] = distribute_tensor(
                    value.to(device=parameter.device),
                    parameter.device_mesh,
                    parameter.placements,
                )
            else:
                raise RankServiceError(
                    f"FSDP2 optimizer state {key!r} shape {tuple(value.shape)} matches neither "
                    f"local {local_shape} nor global {tuple(parameter.shape)} parameter shape"
                )
            restored += 1
    return restored


class RankService:
    def __init__(
        self,
        runtime: dict[str, Any],
        *,
        allocation_context_factory: Callable[[], Any] | None = None,
        residency=None,
    ):
        self.runtime = runtime
        if allocation_context_factory is not None and not callable(allocation_context_factory):
            raise TypeError("allocation_context_factory must be callable when set")
        self._allocation_context_factory = allocation_context_factory
        self.residency = residency
        self.config = runtime["config"]
        self.topology = runtime["topology"]
        self.mode = self.topology["mode"]
        self.world_size = int(self.topology["world_size"])
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.dist = None
        self.gloo_group = None
        self.ddp_snapshot_group = None
        self.connection: socket.socket | None = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.pipeline_compute = None
        self.pipeline_layout_signature: str | None = None
        self.t2_tp_placements: dict[str, Any] | None = None
        self.t2_tp_rank_placements: list[dict[str, Any]] | None = None
        self.t2_tp_layout_recipe: str | None = None
        self.t2_tp_layout_signature: str | None = None
        self.evidence: dict[str, Any] = {}
        self.ddp_ignored_state_names: list[str] = []
        self._packed_weight_store = None
        self._packed_weight_offloader = None
        self._packed_weight_hook_handles: list[Any] = []
        self._allocation_context_active = False
        self._frozen_embedding_input_policy = None
        self.current_request_id: str | None = None
        self.current_phase = "process_group_start"
        # Keep persistent model/optimizer storage in one releasable pool;
        # ordinary forward/backward workspaces use the native allocator.
        with self._allocation_context():
            self._load_model()
        if self.residency is not None:
            self._reclaim_private_cache()
        if self.world_size > 1 and self.dist is not None:
            self.dist.barrier()
        if self.rank == 0 and self.connection is not None:
            send_message(self.connection, {"kind": "ready", "payload": self.evidence})

    def _reclaim_private_cache(self) -> dict[str, Any]:
        self._status("private_pool_cache_reclaim_started")
        report = self.residency.reclaim_inactive_cache()
        self.evidence["private_pool_cache_reclaim"] = report
        self._status("private_pool_cache_reclaim_completed", **report)
        return report

    def _execute_allocation_operation(self, operation, payload):
        with self._allocation_context():
            result = self._execute(operation, payload)
        if self.residency is not None and operation == "restore":
            self._reclaim_private_cache()
        return result

    def _allocation_context(self):
        factory = self._allocation_context_factory
        if factory is None:
            self._allocation_context_active = False
            return nullcontext()

        @contextmanager
        def tracked_context():
            with factory() as value:
                self._allocation_context_active = True
                try:
                    yield value
                finally:
                    self._allocation_context_active = False

        return tracked_context()

    def _status(self, phase: str, **values: Any) -> None:
        self.current_phase = phase
        payload = {
            "state": "starting" if self.current_request_id is None else "updating",
            "phase": phase,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "updated_at": time.time(),
            **_json_safe(values),
        }
        print("PARALLEL_RANK_STATUS " + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
        if self.rank == 0 and self.connection is not None:
            send_message(self.connection, {
                "kind": "status",
                "payload": payload,
            })

    def _abort_owned_world(self, operation: str, request_id: str, error: BaseException) -> None:
        details = {
            "operation": operation,
            "request_id": request_id,
            "phase": self.current_phase,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=40),
        }
        print("PARALLEL_RANK_FAILURE " + json.dumps(details, sort_keys=True), file=sys.stderr, flush=True)
        if self.rank == 0 and self.connection is not None:
            try:
                send_message(self.connection, {
                    "kind": "response",
                    "request_id": request_id,
                    "ok": False,
                    "phase": self.current_phase,
                    "error": f"rank {self.rank} {type(error).__name__}: {error}",
                })
            except OSError:
                pass

        expected_parent = int(os.environ.get(
            "QWEN_RL_PARALLEL_RANK_PARENT_PID",
            os.environ.get("QWEN_RL_PARALLEL_TORCHRUN_PID", "-1"),
        ))
        process_group = os.getpgrp()
        if (expected_parent > 0 and os.getppid() == expected_parent
                and os.getsid(0) == process_group):
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise SystemExit(1)

    def _load_model(self) -> None:
        if self.mode == "pp" and (
            self.world_size != 4
            or self.config.get("model_backend") != "ternary"
            or self.config.get("pipeline_stages") != 4
        ):
            raise RankServiceError("PP rank service is available only for the verified T2 four-stage backend")
        import torch
        import torch.distributed as dist

        self.dist = dist
        if not torch.cuda.is_available():
            raise RankServiceError("parallel Learner workers require CUDA")
        torch.cuda.set_device(self.local_rank)
        torch.cuda.set_per_process_memory_fraction(
            float(self.topology["cuda_memory_fraction"]), device=self.local_rank
        )
        if self.mode != "single":
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    timeout=timedelta(seconds=float(self.topology["compute_timeout"])),
                )
            self.gloo_group = dist.new_group(
                ranks=list(range(self.world_size)),
                backend="gloo",
                timeout=timedelta(seconds=float(self.topology["compute_timeout"])),
            )
            if self.mode == "ddp":
                # P2P checkpoint payloads use an isolated Gloo group. Non-snapshot
                # ranks can wait in the command broadcast group without reordering
                # its collectives against rank-0 receive operations.
                self.ddp_snapshot_group = dist.new_group(
                    ranks=list(range(self.world_size)),
                    backend="gloo",
                    timeout=timedelta(seconds=float(self.topology["compute_timeout"])),
                )
        if self.rank == 0:
            self._connect_parent()
        self._status("model_loading")

        if self.mode == "fsdp2":
            self._configure_fsdp2()

        model_backend = self.config.get("model_backend", "nvfp4")
        mesh = None
        if self.mode == "tp" and model_backend != "ternary":
            from accelerate import ParallelismConfig

            mesh = ParallelismConfig(tp_size=self.world_size).build_device_mesh("cuda")
        lora_rank = int(self.config.get("lora_rank", 16))
        lora_alpha = int(self.config.get("lora_alpha", 16))
        gradient_checkpointing = self.config.get("gradient_checkpointing", True)
        if type(gradient_checkpointing) is not bool:
            raise RankServiceError("gradient_checkpointing must be a boolean")
        if model_backend == "ternary" and self.mode == "ddp":
            self.model, self.tokenizer, loader_evidence = self._load_ternary_ddp_model(
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        elif model_backend == "ternary" and self.mode == "tp":
            self.model, self.tokenizer, loader_evidence = self._load_ternary_tp_policy()
        elif model_backend == "ternary" and self.mode == "pp":
            # The rank-local pipeline stage is built below after model dispatch.
            self.model, self.tokenizer, loader_evidence = None, None, {}
        elif model_backend == "nvfp4":
            from qwen_rl.models.nvfp4 import load_nvfp4_policy

            model_path = self.config.get("model_path")
            if not isinstance(model_path, str) or not model_path:
                raise RankServiceError("parallel Learner config requires model_path")
            self.model, self.tokenizer, loader_evidence = load_nvfp4_policy(
                model_path,
                trainable=True,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                gradient_checkpointing=gradient_checkpointing,
                parallel_mode=("tp" if self.mode == "tp" else "single" if self.mode == "single" else "replicated"),
                device_mesh=mesh,
            )
        else:
            raise RankServiceError(
                f"parallel rank service does not support model_backend={model_backend!r} in mode={self.mode!r}"
            )
        from qwen_rl.learner.trainer import LearnerGRPOTrainer
        if self.mode == "pp":
            self.model, self.tokenizer, loader_evidence = self._load_ternary_pipeline_stage(
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        if self.mode == "fsdp2":
            self.model._axolotl_lora_fp32_gradients = True
        self._validate_model()
        if self.mode == "ddp":
            self.ddp_ignored_state_names = _configure_ddp_ignored_frozen_state(self.model)
        self._status("trainer_initializing")
        embedding_input_grad_policy = None
        if (model_backend == "ternary" and self.mode == "ddp"
                and self.config.get("packed_weight_offload", False) and gradient_checkpointing
                and getattr(self.model, "_t2_gdn_execution", None) is not None):
            from .t2_packed_offload import install_t2_frozen_embedding_input_policy

            policy = install_t2_frozen_embedding_input_policy(self.model, enabled=True)
            self._frozen_embedding_input_policy = policy
            embedding_input_grad_policy = policy
        self.trainer = LearnerGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            group_size=int(self.config.get("group_size", 3)),
            groups_per_update=int(self.config.get("groups_per_update", 2)),
            microbatch_size=int(self.config.get("microbatch_size", 2)),
            max_padded_tokens_per_microbatch=self.config.get("max_padded_tokens_per_microbatch"),
            learning_rate=float(self.config.get("lr", 1e-6)),
            max_grad_norm=float(self.config.get("max_grad_norm", 1.0)),
            max_sequence_length=int(self.config.get("max_sequence_length", 65536)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
            adam_betas=tuple(self.config.get("adam_betas", (0.9, 0.999))),
            adam_eps=float(self.config.get("adam_eps", 1e-8)),
            output_dir=str(Path(self.runtime["runtime_dir"]) / "trainer"),
            seed=int(self.config.get("seed", 42)),
            importance_sampling_cap=float(self.config.get("importance_sampling_cap", 3.0)),
            activation_offload=bool(self.config.get("activation_offload", True)),
            activation_offload_min_tokens=self.config.get("activation_offload_min_tokens", DEFAULT_OFFLOAD_MIN_TOKENS),
            checkpoint_input_recompute_span=self.config.get("checkpoint_input_recompute_span", 1),
            activation_offload_pin_memory=(model_backend != "ternary"),
            activation_offload_max_pinned_bytes=self._activation_offload_pinned_budget(),
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": False}
                if model_backend == "ternary" and self.mode == "ddp"
                and self.config.get("packed_weight_offload", False)
                else None
            ),
            parallel_mode=self.mode,
            parallel_world_size=self.world_size,
            parallel_rank=self.rank,
        )
        if model_backend == "ternary" and self.mode == "ddp":
            from qwen_rl.learner.parallel.qwen35_pipeline import install_trl_fused_head_bridge

            self.ternary_head_bridge = install_trl_fused_head_bridge(self.model)
        if self.residency is not None:
            from ..cuda_residency import materialize_adam_state
            materialize_adam_state(self.trainer.optimizer)
        packed_weight_offload_evidence = self._install_t2_ddp_packed_weight_offloader()
        guard = self.trainer._gradient_guard
        old_phase_callback = guard.phase_callback

        def phase_callback(stage: str, **values: Any) -> None:
            if old_phase_callback is not None:
                old_phase_callback(stage)
            self._status(stage, **values)

        guard.phase_callback = phase_callback
        self.trainer._parallel_phase_callback = phase_callback
        if self.rank == 0:
            def progress_callback(stage: str, telemetry: dict[str, Any]) -> None:
                telemetry = copy.deepcopy(telemetry)
                telemetry["measurement_scope"] = (
                    "rank0_partial" if self.mode in {"ddp", "fsdp2", "tp"} else "global"
                )
                self._status(stage, telemetry=telemetry)

            self.trainer._progress_callback = progress_callback
        original_backward = self.trainer.accelerator.backward
        if self._packed_weight_offloader is not None:
            original_backward = self._wrap_packed_weight_backward(
                original_backward, self._packed_weight_offloader
            )

        def traced_backward(loss, *args, **kwargs):
            self._status(
                "backward_started",
                global_step=int(getattr(self.trainer.state, "global_step", 0)),
            )
            try:
                return original_backward(loss, *args, **kwargs)
            finally:
                self._status(
                    "backward_completed",
                    global_step=int(getattr(self.trainer.state, "global_step", 0)),
                )

        self.trainer.accelerator.backward = traced_backward
        if self.mode == "fsdp2":
            self._status("fsdp2_initial_prepare_started")
            # Use the pinned Transformers Trainer preparation path before the
            # controller's version-0 snapshot. FSDP2 installs DTensor layouts
            # here; preparing only inside the first train() would change the
            # recovery layout after that snapshot.
            self.trainer._prepare_for_training(
                max_steps=1,
                train_dataloader=None,
                resume_from_checkpoint=None,
            )
            self._status(
                "fsdp2_initial_prepare_completed",
                local_layout_parameters=len(self._local_parameter_layouts()),
            )
        self.evidence = dict(loader_evidence)
        if packed_weight_offload_evidence:
            self.evidence["packed_weight_offload"] = packed_weight_offload_evidence
        if self.mode == "pp":
            self._configure_pipeline_compute()
            self.evidence.update({
                "pipeline_layout_signature": self.pipeline_layout_signature,
                "pipeline_stage_manifests": copy.deepcopy(self.pipeline_rank_layouts),
                "pipeline_global_lora_target_count": len(
                    self.pipeline_peft_config["default"]["target_modules"]
                ),
                "pipeline_trainable_lora_parameter_count": len(self.pipeline_expected_adapter_names),
            })
        manifest = loader_evidence.get("artifact_manifest", {}) if isinstance(loader_evidence, dict) else {}
        packed_count = loader_evidence.get("packed_module_count") if isinstance(loader_evidence, dict) else None
        if packed_count is None and isinstance(manifest, dict):
            packed_count = manifest.get("packed_module_count")
        if (packed_count is None and model_backend == "ternary" and self.mode == "tp"
                and isinstance(loader_evidence, dict)):
            packed_count = loader_evidence.get("t2_tp_packed_module_count")
        if self.mode == "pp" and packed_count is None:
            from qwen_rl.models.ternary import FrozenTernaryLinear

            packed_count = sum(isinstance(module, FrozenTernaryLinear) for module in self.model.modules())
        if isinstance(packed_count, bool) or not isinstance(packed_count, int) or packed_count <= 0:
            raise RankServiceError("loaded policy evidence must include a positive packed_module_count")
        self.evidence["packed_module_count"] = packed_count
        if bool(self.config.get("activation_offload", True)):
            self.evidence["activation_offload"] = True
            self.evidence["activation_offload_min_tokens"] = self.trainer.activation_offload_min_tokens
            self.evidence["activation_offload_pin_memory"] = model_backend != "ternary"
            pinned_budget = self.trainer.activation_offload_max_pinned_bytes
            if pinned_budget is not None:
                self.evidence["activation_offload_max_pinned_bytes_per_rank"] = pinned_budget
                self.evidence["activation_offload_pinned_budget_bytes_world"] = pinned_budget * self.world_size
        self.evidence.update({
            "parallelism": self.mode,
            "model_backend": "ternary" if self.mode == "pp" else self.config.get("model_backend", "nvfp4"),
            "trainer": (
                "trl.GRPOTrainer._compute_loss via torch Schedule1F1B"
                if self.mode == "pp" else "trl.GRPOTrainer.train"
            ),
            "optimizer": "Trainer-owned AdamW",
            "embedding_input_grad_policy": (embedding_input_grad_policy.report
                                            if embedding_input_grad_policy is not None else None),
            "loss": "upstream TRL DAPO",
            "parallel_data_denominator": "global_policy_tokens",
            "process_group_backend": "nccl" if dist.is_initialized() else "none",
            "state_collective_backend": "gloo" if self.world_size > 1 else "none",
            "rank": self.rank,
            "local_rank": self.local_rank,
            "base_frozen": True,
            "adapter_state_collection": (
                "four_rank_disjoint_lora_inventory_merge"
                if self.mode == "pp"
                else "torch.distributed.checkpoint.get_model_state_dict(ignore_frozen_params=True)"
                if self.mode == "fsdp2"
                else "canonical_t2_tp_full_peft_gather"
                if model_backend == "ternary" and self.mode == "tp"
                else "official_peft_adapter_state"
            ),
            "gradient_clip_scope": (
                "global_pipeline_lora_norm"
                if self.mode == "pp"
                else "global_t2_tp_lora_norm"
                if model_backend == "ternary" and self.mode == "tp"
                else None
            ),
            "t2_tp_gradient_reduction": (
                "placement_aware_sum_partial_mean_complete"
                if model_backend == "ternary" and self.mode == "tp" else None
            ),
            "pipeline_schedule": "torch.distributed.pipelining.Schedule1F1B" if self.mode == "pp" else None,
            "ddp_initial_sync_scope": "trainable_lora_only" if self.mode == "ddp" else None,
            "ddp_ignored_frozen_state_count": len(self.ddp_ignored_state_names),
            "trainable_adapter_parameter_count": sum(
                parameter.numel() for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and "lora_" in name
            ),
        })
        self._configure_ternary_regional_compile()
        process_record = {
            "rank": self.rank,
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "process_group": os.getpgid(0),
        }
        process_records: list[Any] = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(process_records, process_record, group=self.gloo_group)
        else:
            process_records[0] = process_record
        if self.rank == 0:
            self.evidence["owned_rank_processes"] = process_records
        if self.world_size > 1:
            dist.barrier()

    def _configure_pipeline_compute(self) -> None:
        """Bind the real pipeline runner and establish one shared layout identity."""
        from qwen_rl.learner.parallel.pipeline_compute import PipelineComputeAdapter
        from qwen_rl.learner.parallel.pipeline_exports import ordered_optimizer_parameter_group_signature

        self.pipeline_compute = PipelineComputeAdapter(
            rank=self.rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
            config=self.config,
            stage_model=self.model,
            tokenizer=self.tokenizer,
            trainer=self.trainer,
            process_group=None,
            control_group=self.gloo_group,
        )
        base_model = self.model.get_base_model() if callable(getattr(self.model, "get_base_model", None)) else self.model
        stage_layout = {
            "rank": self.rank,
            "stage_index": self.rank,
            "layer_range": list(base_model._qwen35_layer_range),
            "parameters": self.pipeline_compute.local_parameter_layout(),
            "optimizer_parameter_groups": self._named_parameter_groups(),
            "ordered_optimizer_parameter_group_signature": ordered_optimizer_parameter_group_signature(
                self.model,
                self.trainer.optimizer,
            ),
        }
        stage_record = {
            "rank": self.rank,
            "layout": stage_layout,
            "peft_config": self.pipeline_compute.peft_config(),
        }
        records: list[Any] = [None] * self.world_size
        assert self.dist is not None
        self.dist.all_gather_object(records, stage_record, group=self.gloo_group)
        if any(not isinstance(item, dict) or item.get("rank") != index for index, item in enumerate(records)):
            raise RankServiceError("PP ranks returned incomplete or reordered local layout records")
        self.pipeline_rank_layouts = [item["layout"] for item in records]
        self.pipeline_layout_signature = self._layout_signature(self.pipeline_rank_layouts)
        self.pipeline_compute.set_layout_signature(self.pipeline_layout_signature)
        self.pipeline_peft_config = _merge_pipeline_peft_configs(
            [item["peft_config"] for item in records]
        )
        inventories = [
            {entry["name"] for entry in record["layout"]["parameters"]}
            for record in records
        ]
        merged_names: set[str] = set()
        for rank, inventory in enumerate(inventories):
            if not inventory or merged_names.intersection(inventory):
                raise RankServiceError(f"PP stage {rank} has an empty or overlapping LoRA parameter inventory")
            merged_names.update(inventory)
        if len(merged_names) != 2 * len(self.pipeline_peft_config["default"]["target_modules"]):
            raise RankServiceError("PP adapter inventory does not contain one LoRA A/B pair per global target")
        self.pipeline_expected_adapter_names = merged_names

    def _configure_ternary_regional_compile(self) -> None:
        """Install the accepted full-replica T2 recipe once per live model."""
        if self.config.get("model_backend") != "ternary" or self.mode != "ddp":
            return
        from qwen_rl.models.ternary_compile import configure_ternary_regional_compile

        self._ternary_regional_compile = configure_ternary_regional_compile(
            self.model, enabled=self.config.get("ternary_regional_compile", True),
        )
        self.evidence["regional_compile"] = (
            self._ternary_regional_compile.evidence
            if self._ternary_regional_compile is not None else {"enabled": False}
        )

    def _activation_offload_pinned_budget(self) -> int | None:
        """Give each T2 DDP rank its own arena; preserve the TP global budget."""
        explicit = self.config.get("activation_offload_max_pinned_bytes")
        if explicit is not None:
            return explicit
        if self.config.get("model_backend") == "ternary" and self.mode in {"ddp", "tp"}:
            from qwen_rl.learner.bounded_offload import DEFAULT_MAX_PINNED_BYTES

            if self.mode == "ddp":
                return DEFAULT_MAX_PINNED_BYTES
            return DEFAULT_MAX_PINNED_BYTES // self.world_size
        return None

    @staticmethod
    def _t2_tp_weight_offloader(model):
        for name in ("_t2_packed_weight_offloader", "_t2_tp_weight_offloader"):
            streamer = getattr(model, name, None)
            if streamer is not None:
                return streamer
        get_base = getattr(model, "get_base_model", None)
        base = get_base() if callable(get_base) else None
        if base is not None:
            for name in ("_t2_packed_weight_offloader", "_t2_tp_weight_offloader"):
                streamer = getattr(base, name, None)
                if streamer is not None:
                    return streamer
        return None

    @staticmethod
    def _protect_preplaced_model(accelerator, target_model) -> None:
        """Keep Accelerator from moving only this already-placed packed model."""
        from .t2_packed_offload import protect_preplaced_model

        protect_preplaced_model(accelerator, target_model)

    @staticmethod
    def _packed_weight_phase_hooks(model, offloader) -> list[Any]:
        """Set forward phases around calls that bypass accelerator.backward."""
        from .t2_packed_offload import packed_weight_phase_hooks

        return packed_weight_phase_hooks(model, offloader)

    @staticmethod
    def _wrap_packed_weight_backward(backward, offloader):
        """Bracket checkpoint recomputation with reverse-order packed leases."""
        from .t2_packed_offload import wrap_packed_weight_backward

        return wrap_packed_weight_backward(backward, offloader)

    def _install_t2_ddp_packed_weight_offloader(self) -> dict[str, Any]:
        """Share one canonical packed CPU store across local T2 DDP ranks."""
        if not (self.config.get("model_backend") == "ternary" and self.mode == "ddp"
                and self.config.get("packed_weight_offload", False)):
            return {}
        if (self._allocation_context_factory is not None
                and not self._allocation_context_active):
            raise RankServiceError(
                "T2 DDP packed offloader must replace CUDA buffers before the persistent pool exits"
            )

        import torch
        from .t2_packed_offload import (
            build_t2_packed_weight_layout,
            install_t2_packed_offloader,
            open_t2_shared_packed_store,
            t2_packed_offload_report,
        )

        model = self.model
        device = torch.device("cuda", self.local_rank)
        layout = None
        local_error = None
        try:
            layout = build_t2_packed_weight_layout(
                model,
                device=device,
                artifact_identity=self.config.get("artifact_identity"),
                max_cpu_master_bytes=int(self.config.get(
                "packed_weight_offload_max_cpu_master_bytes", 4 * 1024**3
                )),
                max_pinned_bytes=int(self.config.get(
                "packed_weight_offload_max_pinned_bytes", 512 * 1024**2
                )),
            )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

        layout_status = {
            "rank": self.rank,
            "error": local_error,
            "identity": None if layout is None else layout.identity,
            "group_sizes": [] if layout is None else list(layout.group_sizes),
        }
        layout_reports: list[Any] = [None] * self.world_size
        if self.world_size > 1:
            self.dist.all_gather_object(layout_reports, layout_status, group=self.gloo_group)
        else:
            layout_reports[0] = layout_status
        layout_failures = [row for row in layout_reports if row.get("error") is not None]
        if layout_failures:
            raise RankServiceError(f"T2 DDP shared packed layout failed: {layout_failures}")
        if any(row["identity"] != layout_reports[0]["identity"]
               or row["group_sizes"] != layout_reports[0]["group_sizes"]
               for row in layout_reports[1:]):
            raise RankServiceError("T2 DDP ranks disagree on shared packed layout identity")
        assert layout is not None

        recovery_dir = Path(self.config.get("recovery_dir") or self.runtime["runtime_dir"])
        configured_store_root = self.config.get("packed_weight_offload_store_dir")
        store_root = (
            Path(configured_store_root).expanduser()
            if configured_store_root is not None
            else recovery_dir.expanduser().resolve() / "frozen-base"
        )
        self._status(
            "t2_ddp_shared_packed_store_open_started",
            group_count=len(layout.plan.groups),
            cpu_master_bytes=layout.plan.cpu_master_bytes,
        )
        store = None
        store_error = None
        try:
            store = open_t2_shared_packed_store(layout, store_root)
        except Exception as exc:
            store_error = f"{type(exc).__name__}: {exc}"
        store_statuses: list[Any] = [None] * self.world_size
        local_store_status = {"rank": self.rank, "error": store_error}
        if self.world_size > 1:
            self.dist.all_gather_object(store_statuses, local_store_status, group=self.gloo_group)
        else:
            store_statuses[0] = local_store_status
        store_failures = [row for row in store_statuses if row.get("error") is not None]
        if store_failures:
            if store is not None:
                store.close()
            raise RankServiceError(f"T2 DDP shared packed store failed: {store_failures}")
        assert store is not None

        self._status("t2_ddp_shared_packed_offloader_install_started")
        offloader = None
        nested_checkpoints = None
        install_error = None
        try:
            offloader, nested_checkpoints = install_t2_packed_offloader(
                model, layout, store, config=self.config,
            )
        except Exception as exc:
            install_error = f"{type(exc).__name__}: {exc}"
        install_statuses: list[Any] = [None] * self.world_size
        local_install_status = {"rank": self.rank, "error": install_error}
        if self.world_size > 1:
            self.dist.all_gather_object(install_statuses, local_install_status, group=self.gloo_group)
        else:
            install_statuses[0] = local_install_status
        install_failures = [row for row in install_statuses if row.get("error") is not None]
        if install_failures:
            if nested_checkpoints is not None:
                nested_checkpoints.close()
            if offloader is not None:
                try:
                    offloader.close()
                except Exception:
                    pass
            store.close()
            raise RankServiceError(f"T2 DDP packed offloader installation failed: {install_failures}")
        assert offloader is not None

        # Keep one owner reachable through model and rank service; buffers are
        # views into the shared mapping, not per-rank CPU masters.
        self._packed_weight_store = store
        self._packed_weight_offloader = offloader
        self._nested_decoder_checkpoints = nested_checkpoints
        self._protect_preplaced_model(self.trainer.accelerator, model)

        self._packed_weight_hook_handles.extend(
            self._packed_weight_phase_hooks(model, offloader)
        )
        self._status("t2_ddp_shared_packed_offloader_installed")
        return t2_packed_offload_report(
            layout,
            store,
            offloader,
            nested_checkpoints,
            mode="local_ddp_shared_cpu_master",
            max_pinned_bytes=int(self.config.get(
                "packed_weight_offload_max_pinned_bytes", 512 * 1024**2
            )),
            retained_device_components=(
                "input_hadamard_signs", "bias", "token_embedding", "output_head", "LoRA", "optimizer"
            ),
        )

    def _load_ternary_tp_policy(self):
        """Load the explicit four-rank T2 tensor-parallel model and bind its layout."""
        if self.world_size != 4:
            raise RankServiceError("T2 TP rank service requires exactly four ranks")
        from qwen_rl.models.ternary_tp_model import load_t2_tp_policy
        from qwen_rl.models.ternary_tp_state import (
            normalize_t2_tp_placements,
            t2_tp_layout_signature,
        )

        model, tokenizer, evidence = load_t2_tp_policy(
            self.config,
            local_rank=self.local_rank,
            rank=self.rank,
            world_size=self.world_size,
            control_group=self.gloo_group,
        )
        if getattr(model, "_t2_tp_model", False) is not True:
            raise RankServiceError("T2 TP loader did not mark its tensor-parallel model")
        raw_placements = getattr(model, "_t2_tp_placements", None)
        layout_recipe = getattr(model, "_t2_tp_layout_recipe", None)
        if not isinstance(layout_recipe, str) or not layout_recipe.strip():
            raise RankServiceError("T2 TP loader omitted its layout recipe")

        placement_error = None
        local_placements = None
        try:
            local_placements = normalize_t2_tp_placements(
                model, raw_placements, rank=self.rank, world_size=self.world_size
            )
        except Exception as exc:
            placement_error = f"{type(exc).__name__}: {exc}"
        placement_statuses: list[Any] = [None] * self.world_size
        assert self.dist is not None
        self.dist.all_gather_object(
            placement_statuses,
            {"rank": self.rank, "error": placement_error},
            group=self.gloo_group,
        )
        placement_failures = [item for item in placement_statuses if item.get("error") is not None]
        if placement_failures:
            raise RankServiceError(f"T2 TP placement validation failed across ranks: {placement_failures}")

        rank_placements: list[Any] = [None] * self.world_size
        self.dist.all_gather_object(rank_placements, local_placements, group=self.gloo_group)
        try:
            layout_signature = t2_tp_layout_signature(rank_placements, recipe=layout_recipe)
        except Exception as exc:
            raise RankServiceError(f"T2 TP global placement layout is invalid: {exc}") from exc
        self.t2_tp_placements = local_placements
        self.t2_tp_rank_placements = rank_placements
        self.t2_tp_layout_recipe = layout_recipe
        self.t2_tp_layout_signature = layout_signature
        if not isinstance(evidence, dict):
            raise RankServiceError("T2 TP loader omitted structured load evidence")
        evidence.update({
            "model_backend": "ternary",
            "parallelism": "tp",
            "world_size": self.world_size,
            "t2_tp_layout_recipe": layout_recipe,
            "t2_tp_layout_signature": layout_signature,
            "t2_tp_trainable_placement_count": len(local_placements),
            "adapter_state_collection": "canonical_t2_tp_full_peft_gather",
        })
        return model, tokenizer, evidence

    def _load_ternary_ddp_model(self, *, lora_rank: int, lora_alpha: int):
        """Load the complete identity-bound T2 model independently on each DDP rank."""
        import torch
        from .t2_model_assembly import load_t2_model

        def materialize_tokenizer_files(provider, destination):
            tokenizer_files_box = [None]
            if self.rank == 0:
                tokenizer_files_box[0] = provider.materialize_text_resources(destination)
            assert self.dist is not None
            if self.world_size > 1:
                self.dist.broadcast_object_list(tokenizer_files_box, src=0, group=self.gloo_group)
            return tokenizer_files_box[0]

        try:
            return load_t2_model(
                {**self.config, "lora_rank": lora_rank, "lora_alpha": lora_alpha},
                device=torch.device("cuda", self.local_rank),
                parallelism="ddp",
                world_size=self.world_size,
                materialize_tokenizer_files=materialize_tokenizer_files,
            )
        except Exception as exc:
            raise RankServiceError(f"T2 DDP model assembly failed: {type(exc).__name__}: {exc}") from exc

    def _load_ternary_pipeline_stage(self, *, lora_rank: int, lora_alpha: int):
        """Build only this rank's verified T2 Qwen3.5 stage and tokenizer view."""
        import torch
        from transformers import AutoTokenizer

        from qwen_rl.contracts.ternary import load_artifact_identity
        from qwen_rl.learner.parallel.qwen35_pipeline import build_qwen35_pipeline_stage
        from qwen_rl.models.ternary_provider import TernaryModelProvider

        if self.mode != "pp" or self.world_size != 4 or self.config.get("model_backend") != "ternary":
            raise RankServiceError("the four-stage pipeline service accepts only model_backend='ternary'")
        if self.config.get("pipeline_stages") != 4:
            raise RankServiceError("T2 PP requires exactly four declared pipeline stages")
        identity_path = self.config.get("artifact_identity_path")
        identity_document = self.config.get("artifact_identity")
        tokenizer_path = self.config.get("tokenizer_path")
        if not isinstance(identity_path, str) or not identity_path:
            raise RankServiceError("T2 PP requires artifact_identity_path")
        if not isinstance(identity_document, dict):
            raise RankServiceError("T2 PP requires the validated artifact identity document")
        if not isinstance(tokenizer_path, str) or not tokenizer_path:
            raise RankServiceError("T2 PP requires the identity-bound tokenizer_path")
        assert self.dist is not None
        provider = None
        identity_error = None
        try:
            identity = load_artifact_identity(identity_path, verify_files=False)
            if identity.as_dict() != identity_document:
                raise RankServiceError("T2 PP artifact identity sidecar changed after profile validation")
            source = identity_document["source"]["ninfer_artifact"]
            model_path = self.config.get("model_path")
            if not isinstance(model_path, str) or Path(model_path).resolve() != Path(source["path"]).resolve():
                raise RankServiceError("T2 PP model_path differs from the validated artifact identity")
            adapter = identity_document["learner"]["adapter"]
            if lora_rank != adapter["rank"] or lora_alpha != adapter["alpha"]:
                raise RankServiceError("T2 PP LoRA rank/alpha differs from the validated artifact identity")
            provider = TernaryModelProvider(identity_path, verify_files=False)
            if provider.identity.as_dict() != identity_document:
                raise RankServiceError("T2 provider identity changed before stage construction")
        except BaseException as exc:
            identity_error = f"{type(exc).__name__}: {exc}"
        identity_statuses: list[Any] = [None] * self.world_size
        self.dist.all_gather_object(
            identity_statuses,
            {"rank": self.rank, "error": identity_error},
            group=self.gloo_group,
        )
        identity_failures = [item for item in identity_statuses if item.get("error") is not None]
        if identity_failures:
            if provider is not None:
                provider.close()
            raise RankServiceError(f"T2 identity/provider validation failed across ranks: {identity_failures}")
        assert provider is not None

        device = torch.device("cuda", self.local_rank)
        try:
            resource_box: list[Any] = [{"ok": None}]
            if self.rank == 0:
                try:
                    resource_box[0] = {
                        "ok": True,
                        "files": provider.materialize_text_resources(tokenizer_path),
                    }
                except BaseException as exc:
                    resource_box[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            assert self.dist is not None
            self.dist.broadcast_object_list(resource_box, src=0, group=self.gloo_group)
            resource_result = resource_box[0]
            if not isinstance(resource_result, dict) or resource_result.get("ok") is not True:
                error = resource_result.get("error") if isinstance(resource_result, dict) else None
                raise RankServiceError(f"rank 0 failed to materialize T2 text resources: {error}")
            resource_files = resource_result.get("files")
            if not isinstance(resource_files, dict) or not resource_files:
                raise RankServiceError("rank 0 did not materialize the embedded tokenizer resources")
            self.dist.barrier(group=self.gloo_group)
            stage_error = None
            model = tokenizer = loader_evidence = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                model = build_qwen35_pipeline_stage(
                    provider.text_config,
                    provider,
                    stage_index=self.rank,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    device=device,
                    gradient_checkpointing=bool(self.config.get("gradient_checkpointing", True)),
                )
                base_model = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
                loader_evidence = provider.evidence()
                loader_evidence.update({
                    "model_backend": "ternary",
                    "pipeline_stage": self.rank,
                    "pipeline_stages": 4,
                    "pipeline_layer_range": list(base_model._qwen35_layer_range),
                    "pipeline_gradient_checkpointing": dict(base_model._qwen35_gradient_checkpointing),
                    "attention_implementation": dict(base_model._qwen35_attention),
                    "tokenizer_files": resource_files,
                    "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
                    "tokenizer_length": len(tokenizer),
                    "model_vocab_size": int(provider.text_config["vocab_size"]),
                    "identity_validation_scope": "compute profile rehashed source; rank rechecks sidecar and NInfer header ID",
                })
            except Exception as exc:
                stage_error = f"{type(exc).__name__}: {exc}"
            stage_statuses: list[Any] = [None] * self.world_size
            self.dist.all_gather_object(
                stage_statuses,
                {"rank": self.rank, "error": stage_error},
                group=self.gloo_group,
            )
            stage_failures = [item for item in stage_statuses if item.get("error") is not None]
            if stage_failures:
                raise RankServiceError(f"T2 Qwen stage construction failed across ranks: {stage_failures}")
            assert model is not None and tokenizer is not None and loader_evidence is not None
            return model, tokenizer, loader_evidence
        finally:
            provider.close()

    def _connect_parent(self) -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(float(self.topology["compute_timeout"]))
        deadline = time.monotonic() + float(self.topology["compute_timeout"])
        while True:
            try:
                connection.connect(self.runtime["socket_path"])
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("rank 0 could not connect to the Learner controller")
                time.sleep(0.05)
        send_message(connection, {"kind": "auth", "token": self.runtime["auth_token"]})
        response = receive_message(connection)
        if not isinstance(response, dict) or response.get("kind") != "auth_ack":
            raise RankServiceError("Learner controller rejected private-socket authentication")
        self.connection = connection

    def _configure_fsdp2(self) -> None:
        try:
            from axolotl.monkeypatch.accelerate.fsdp2 import patch_accelerate_fsdp2
            from axolotl.monkeypatch.trainer.trl import patch_trl_prepare_fsdp2

            patch_accelerate_fsdp2()
            patch_trl_prepare_fsdp2()
        except Exception as exc:
            raise RankServiceError(f"cannot install pinned Axolotl FSDP2 integration: {exc}") from exc

    def _validate_model(self) -> None:
        trainable = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if not trainable or any("lora_" not in name for name in trainable):
            raise RankServiceError("parallel Learner policy must have trainable LoRA parameters only")
        if any(parameter.requires_grad for name, parameter in self.model.named_parameters() if "lora_" not in name):
            raise RankServiceError("parallel Learner base parameters must remain frozen")

    def serve(self) -> None:
        assert self.dist is not None
        while True:
            command_box: list[Any] = [None]
            try:
                if self.rank == 0:
                    assert self.connection is not None
                    message = receive_message(self.connection)
                    if not isinstance(message, dict) or message.get("kind") != "command":
                        raise RankServiceError(f"malformed Learner command: {message!r}")
                    command_box[0] = message
                    self.current_request_id = str(message.get("request_id"))
                # Commands may carry a full restore envelope. Keep object payloads
                # on the CPU Gloo group instead of allocating NCCL object tensors.
                if self.world_size > 1:
                    self.dist.broadcast_object_list(command_box, src=0, group=self.gloo_group)
            except BaseException as exc:
                self._abort_owned_world("command_dispatch", self.current_request_id or "", exc)
            command = command_box[0]
            request_id = str(command.get("request_id"))
            operation = str(command.get("operation"))
            payload = command.get("payload")
            self.current_request_id = request_id
            self._status(f"{operation}_started", request_id=request_id)
            try:
                if operation in {"park", "wake", "shutdown"}:
                    result = self._execute(operation, payload)
                elif self.residency is not None and operation != "restore":
                    if self.residency.state != "awake":
                        raise RankServiceError("compute requires resident Learner storage")
                    result = self._execute(operation, payload)
                else:
                    result = self._execute_allocation_operation(operation, payload)
            except BaseException as exc:
                self._abort_owned_world(operation, request_id, exc)
            local_result = {"ok": True, "payload": result, "rank": self.rank}
            gather_started = time.perf_counter()
            gathered = self._gather_command_results(local_result, operation=operation, request_id=request_id)
            gather_seconds = time.perf_counter() - gather_started
            # Gloo gather has produced its rank0 receive objects; release the
            # original local payload before combining or serializing the reply.
            del local_result
            if self.rank == 0:
                try:
                    combine_started = time.perf_counter()
                    result = self._combine_results(operation, [item["payload"] for item in gathered])
                    combine_seconds = time.perf_counter() - combine_started
                    if operation == "snapshot":
                        performance = result.setdefault("checkpoint_performance", {})
                        performance["rank_gather_wall_seconds"] = gather_seconds
                        performance["rank_combine_seconds"] = combine_seconds
                    send_message(self.connection, {
                        "kind": "response",
                        "request_id": request_id,
                        "ok": True,
                        "payload": result,
                    })
                except BaseException as exc:
                    self._abort_owned_world("result_packaging", request_id, exc)
            self.current_request_id = None
            if operation == "shutdown":
                break
            # Do not retain the previous restore envelope or optimizer snapshots
            # while waiting for the next controller command.
            del gathered, result, payload, command, command_box
            if self.rank == 0:
                message = None
        if self.world_size > 1:
            self.dist.barrier()
            if self.ddp_snapshot_group is not None:
                self.dist.destroy_process_group(self.ddp_snapshot_group)
                self.ddp_snapshot_group = None
            self.dist.destroy_process_group()

    def _gather_command_results(self, local_result, *, operation=None, request_id=""):
        """Gather small DDP metadata and transfer unique optimizer trees by variable-size P2P."""
        if self.world_size == 1:
            return [local_result]
        if operation == "snapshot" and self.mode == "ddp":
            from .snapshot_dedup import (
                MAX_DDP_OPTIMIZER_PICKLE_BYTES,
                MAX_DDP_SNAPSHOT_METADATA_BYTES,
                measure_small_gather_pickle_bytes,
                preflight_ddp_optimizer_snapshot,
                stage_ddp_snapshot_payload,
                synchronize_ddp_snapshot_payload_ready,
                transfer_ddp_snapshot_heavy_payloads,
            )

            local_error = None
            if not isinstance(local_result, dict):
                payload = None
                local_error = "local snapshot result is not a dictionary"
            else:
                payload = local_result.get("payload")
                if local_result.get("ok") is not True:
                    error_type = local_result.get("error_type")
                    error_text = local_result.get("error")
                    details = ": ".join(
                        str(value) for value in (error_type, error_text)
                        if value is not None and str(value)
                    )
                    local_error = "local snapshot execution failed" + (f": {details}" if details else "")
                elif not isinstance(payload, dict):
                    local_error = "local snapshot result is missing its payload dictionary"
                elif payload.get("ok") is False:
                    detail = payload.get("error")
                    local_error = "local snapshot execution failed" + (
                        f": {detail}" if detail is not None and str(detail) else ""
                    )
                elif "optimizer_state" not in payload:
                    detail = payload.get("error")
                    local_error = (
                        f"local snapshot payload is missing optimizer_state: {detail}"
                        if detail is not None and str(detail)
                        else "local snapshot payload is missing optimizer_state"
                    )
            if self.ddp_snapshot_group is None:
                local_error = local_error or "DDP snapshot transfer group is not initialized"
            state = payload.get("optimizer_state") if isinstance(payload, dict) else None
            try:
                plan = preflight_ddp_optimizer_snapshot(
                    state,
                    rank=self.rank,
                    world_size=self.world_size,
                    all_gather_object=lambda output, value: self.dist.all_gather_object(
                        output, value, group=self.gloo_group
                    ),
                    local_error=local_error,
                )
            except BaseException as exc:
                error = RankServiceError(
                    f"DDP optimizer snapshot preflight failed: {type(exc).__name__}: {exc}"
                )
                self._abort_owned_world("snapshot_optimizer_preflight", request_id, error)

            local_heavy = None
            preparation_error = None
            try:
                local_heavy = stage_ddp_snapshot_payload(payload, rank=self.rank, plan=plan)
            except BaseException as exc:
                preparation_error = exc
            local_pickle_bytes = (
                int(local_heavy["optimizer_pickle_bytes"]) if local_heavy is not None else 0
            )
            try:
                preparation_seconds, preparation_statuses = synchronize_ddp_snapshot_payload_ready(
                    rank=self.rank,
                    world_size=self.world_size,
                    local_error=preparation_error,
                    status_fields={"optimizer_pickle_bytes": local_pickle_bytes},
                    expected_representatives=plan["rank_representatives"],
                    bounded_fields={"optimizer_pickle_bytes": MAX_DDP_OPTIMIZER_PICKLE_BYTES},
                    all_gather_object=lambda output, value: self.dist.all_gather_object(
                        output, value, group=self.gloo_group
                    ),
                )
            except BaseException as exc:
                error = RankServiceError(
                    f"DDP optimizer snapshot serialization failed: {type(exc).__name__}: {exc}"
                )
                self._abort_owned_world("snapshot_optimizer_preflight", request_id, error)
            optimizer_pickle_bytes_by_rank = [
                int(item["optimizer_pickle_bytes"]) for item in preparation_statuses
            ]
            payload["_optimizer_snapshot_pre_gloo"]["payload_ready_collective_seconds"] = (
                preparation_seconds
            )

            receive_buffer = None
            allocation_error = None
            try:
                if self.rank == 0:
                    receive_size = max(optimizer_pickle_bytes_by_rank, default=0)
                    if receive_size > 0:
                        import torch

                        receive_buffer = torch.empty(receive_size, dtype=torch.uint8, device="cpu")
            except BaseException as exc:
                allocation_error = exc
            try:
                allocation_seconds, _allocation_statuses = synchronize_ddp_snapshot_payload_ready(
                    rank=self.rank,
                    world_size=self.world_size,
                    local_error=allocation_error,
                    status_fields={"receive_buffer_ready": allocation_error is None},
                    all_gather_object=lambda output, value: self.dist.all_gather_object(
                        output, value, group=self.gloo_group
                    ),
                )
            except BaseException as exc:
                error = RankServiceError(
                    f"DDP optimizer receive buffer preparation failed: {type(exc).__name__}: {exc}"
                )
                self._abort_owned_world("snapshot_optimizer_preflight", request_id, error)
            payload["_optimizer_snapshot_pre_gloo"]["payload_ready_collective_seconds"] += (
                allocation_seconds
            )

            small_pickle_bytes = 0
            small_payload_error = None
            try:
                small_pickle_bytes = measure_small_gather_pickle_bytes(local_result)
            except BaseException as exc:
                small_payload_error = exc
            try:
                synchronize_ddp_snapshot_payload_ready(
                    rank=self.rank,
                    world_size=self.world_size,
                    local_error=small_payload_error,
                    status_fields={"small_gather_pickle_bytes": small_pickle_bytes},
                    bounded_fields={"small_gather_pickle_bytes": MAX_DDP_SNAPSHOT_METADATA_BYTES},
                    all_gather_object=lambda output, value: self.dist.all_gather_object(
                        output, value, group=self.gloo_group
                    ),
                )
            except BaseException as exc:
                error = RankServiceError(
                    f"DDP small snapshot envelope preparation failed: {type(exc).__name__}: {exc}"
                )
                self._abort_owned_world("snapshot_optimizer_preflight", request_id, error)

            gathered = [None] * self.world_size if self.rank == 0 else None
            self.dist.gather_object(local_result, gathered, dst=0, group=self.gloo_group)
            transfer_status = transfer_ddp_snapshot_heavy_payloads(
                gathered,
                local_heavy,
                rank=self.rank,
                world_size=self.world_size,
                optimizer_pickle_bytes_by_rank=optimizer_pickle_bytes_by_rank,
                receive_buffer=receive_buffer,
                dist=self.dist,
                group=self.ddp_snapshot_group,
            )
            try:
                transfer_seconds, transfer_statuses = synchronize_ddp_snapshot_payload_ready(
                    rank=self.rank,
                    world_size=self.world_size,
                    local_error=transfer_status.get("error"),
                    status_fields={
                        "optimizer_transfer_bytes": int(transfer_status["optimizer_transfer_bytes"]),
                        "optimizer_received_bytes": int(transfer_status["optimizer_received_bytes"]),
                        "transfer_seconds": float(transfer_status["transfer_seconds"]),
                    },
                    expected_transfer_bytes_by_rank=[0, *optimizer_pickle_bytes_by_rank[1:]],
                    expected_received_bytes_by_rank=[sum(optimizer_pickle_bytes_by_rank[1:])]
                    + [0] * (self.world_size - 1),
                    all_gather_object=lambda output, value: self.dist.all_gather_object(
                        output, value, group=self.gloo_group
                    ),
                )
            except BaseException as exc:
                error = RankServiceError(
                    f"DDP optimizer P2P transfer failed: {type(exc).__name__}: {exc}"
                )
                self._abort_owned_world("snapshot_optimizer_transfer", request_id, error)
            if self.rank == 0:
                for rank, status in enumerate(transfer_statuses):
                    gathered[rank]["payload"]["_optimizer_snapshot_pre_gloo"][
                        "optimizer_transfer_bytes"
                    ] = int(status["optimizer_transfer_bytes"])
                    gathered[rank]["payload"]["_optimizer_snapshot_pre_gloo"][
                        "optimizer_received_bytes"
                    ] = int(status["optimizer_received_bytes"])
                    gathered[rank]["payload"]["_optimizer_snapshot_pre_gloo"][
                        "transfer_seconds"
                    ] = float(status["transfer_seconds"])
            return gathered
        gathered = [None] * self.world_size if self.rank == 0 else None
        self.dist.gather_object(local_result, gathered, dst=0, group=self.gloo_group)
        return gathered

    def _execute(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation in {"park", "wake"}:
            if self.residency is None:
                raise RankServiceError("rank residency requires phase_shared")
            if operation == "wake":
                result = self.residency.wake()
                streamer = self._t2_tp_weight_offloader(self._unwrap_model())
                if streamer is not None:
                    streamer.resume("forward")
                return result
            import torch
            model = self._unwrap_model()
            streamer = self._t2_tp_weight_offloader(model)
            if streamer is not None:
                streamer.park()
            # Retire only inactive CuMem segments before sleep would make CPU
            # backups for them. This also synchronizes outstanding CUDA copies.
            private_cache = self._reclaim_private_cache()
            from ..bounded_offload import reclaim_idle_pinned_cache
            self._status("park_host_cache_reclaim_started")
            host_cache = reclaim_idle_pinned_cache(
                empty_host_cache=torch.accelerator.empty_host_cache,
                host_memory_stats=torch.cuda.memory.host_memory_stats,
            )
            self._status("park_host_cache_reclaim_completed", **host_cache)
            if self.world_size > 1:
                # Every rank must return its idle host pages before any rank
                # allocates CPU residency backups on this shared host.
                self.dist.barrier(group=self.gloo_group)
            persistent = [("parameter:" + name, value) for name, value in model.named_parameters()]
            persistent.extend(("buffer:" + name, value) for name, value in model.named_buffers())
            for parameter_index, state in enumerate(self.trainer.optimizer.state.values()):
                persistent.extend((f"optimizer:{parameter_index}:{name}", value)
                                  for name, value in state.items() if isinstance(value, torch.Tensor))
            ownership = self.residency.assert_tensors_owned(persistent)
            frozen = [value for value in model.parameters() if not value.requires_grad]
            frozen.extend(model.buffers())
            mutable = [value for value in model.parameters() if value.requires_grad]
            mutable.extend(value for state in self.trainer.optimizer.state.values()
                           for value in state.values() if isinstance(value, torch.Tensor))
            result = self.residency.park(drain=lambda: None,
                frozen_tensors=frozen, mutable_tensors=mutable)
            result["persistent_storage"] = ownership
            result["pre_park_cache_reclaim"] = {"private_pool": private_cache, "host": host_cache}
            return result
        if operation == "logprobs":
            return self._logprobs(payload)
        if operation == "probe":
            return self._probe(payload)
        if operation == "execution_diagnostics":
            return self._execution_diagnostics(payload)
        if operation == "update":
            return self._update(payload)
        if operation == "snapshot":
            return self._snapshot()
        if operation == "restore":
            return self._restore(payload["snapshot"])
        if operation == "export_adapter":
            return self._export_adapter(payload["path"])
        if operation == "shutdown":
            nested_checkpoints = getattr(self, "_nested_decoder_checkpoints", None)
            if nested_checkpoints is not None:
                nested_checkpoints.close()
            gdn_execution = getattr(self._unwrap_model(), "_t2_gdn_execution", None)
            if gdn_execution is not None:
                gdn_execution.close()
            input_policy = getattr(self, "_frozen_embedding_input_policy", None)
            if input_policy is not None:
                input_policy.close()
                self._frozen_embedding_input_policy = None
            streamer = self._t2_tp_weight_offloader(self._unwrap_model())
            if streamer is not None:
                closed = streamer.close()
                packed_store = self._packed_weight_store
                result = {
                    "closed": True,
                    "t2_tp_weight_offloader": closed,
                }
                if packed_store is not None:
                    packed_store.close()
                    result["shared_packed_store_closed"] = True
                return result
            return {"closed": True}
        raise RankServiceError(f"unsupported parallel learner operation {operation!r}")

    def _logprobs(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "pp":
            raise RankServiceError("T2 PP compute-only mode does not provide a Worker logprob endpoint")
        import torch

        input_ids = payload["input_ids"]
        policy_mask = payload["policy_mask"]
        if not isinstance(input_ids, (list, tuple)) or not isinstance(policy_mask, (list, tuple)):
            raise ValueError("logprobs requires token and mask sequences")
        if len(input_ids) != len(policy_mask) or len(input_ids) < 2:
            raise ValueError("logprobs token and mask lengths differ or are too short")
        device = torch.device("cuda", self.local_rank)
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        mask = torch.tensor([policy_mask], dtype=torch.bool, device=device)
        attention = payload.get("attention_mask")
        attention_tensor = None if attention is None else torch.tensor([attention], dtype=torch.long, device=device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logps, selected = self.trainer.score_token_ids(ids, mask, attention_tensor)
        finally:
            self.model.train(was_training)
        aligned = torch.zeros(len(input_ids), dtype=torch.float32)
        aligned[1:] = logps[0].float().cpu() * selected[0].float().cpu()
        return {"logprobs": aligned.tolist()}

    def _probe(self, payload: dict[str, Any]) -> dict[str, Any]:
        import torch

        lengths = payload.get("lengths")
        if not isinstance(lengths, list) or not lengths or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in lengths
        ):
            raise ValueError("probe lengths must be a non-empty list of integers >= 2")
        if max(lengths) > int(self.config.get("max_sequence_length", 65536)):
            raise ValueError("probe length exceeds configured max_sequence_length")
        if self.mode == "pp":
            self._status("pipeline_probe_started", lengths=lengths)
            result = self.pipeline_compute.probe(lengths)
            self._status("pipeline_probe_completed", finite_logprobs=result.get("finite_logprobs"))
            return dict(result)
        max_length = max(lengths)
        text_config = getattr(self.model.config, "text_config", None)
        vocab = int(getattr(text_config, "vocab_size", getattr(self.model.config, "vocab_size", 0)))
        if vocab < 2:
            raise RuntimeError("loaded model config has no usable vocab_size")
        device = torch.device("cuda", self.local_rank)
        ids = torch.zeros((len(lengths), max_length), dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, length in enumerate(lengths):
            ids[index, :length] = torch.arange(length, device=device, dtype=torch.long).remainder(vocab)
            attention[index, :length] = 1
            mask[index, 1:length] = True
        was_training = self.model.training
        self.model.eval()
        start = time.perf_counter()
        try:
            with torch.no_grad():
                logps, selected = self.trainer.score_token_ids(ids, mask, attention)
                finite = bool(torch.isfinite(logps[selected]).all())
                torch.cuda.synchronize(device)
        finally:
            self.model.train(was_training)
        return {
            "status": "passed" if finite else "failed",
            "rank": self.rank,
            "lengths": list(lengths),
            "selected_policy_tokens": int(selected.sum().item()),
            "finite_logprobs": finite,
            "seconds": time.perf_counter() - start,
        }

    def _execution_diagnostics(self, payload=None) -> dict[str, Any]:
        """Collect explicit T2 TP frozen-state fingerprints on this idle rank."""
        if self.mode != "tp" or self.config.get("model_backend") != "ternary":
            raise RankServiceError("execution diagnostics are available only for T2 TP")
        import torch
        from qwen_rl.models.ternary_tp_diagnostics import (
            collect_t2_tp_execution_diagnostics,
        )

        device = torch.device("cuda", self.local_rank)
        torch.cuda.synchronize(device)
        model = self._unwrap_model()
        report = collect_t2_tp_execution_diagnostics(
            model,
            self.trainer,
            rank=self.rank,
            world_size=self.world_size,
            offloader=self._t2_tp_weight_offloader(model),
            residency_state=None if self.residency is None else str(self.residency.state),
        )
        if payload and ("input_ids" in payload or "policy_mask" in payload):
            from qwen_rl.models.ternary_tp_diagnostics import capture_t2_forward_trace

            ids, mask = payload.get("input_ids"), payload.get("policy_mask")
            if (not isinstance(ids, list) or not isinstance(mask, list)
                    or not 2 <= len(ids) <= 256 or len(ids) != len(mask)):
                raise ValueError("forward diagnostics require aligned inputs of 2..256 tokens")
            plain = self._logprobs(payload)["logprobs"]
            with capture_t2_forward_trace(model) as trace:
                traced = self._logprobs(payload)["logprobs"]
            report["forward_probe"] = {
                "plain_scores": plain, "traced_scores": traced, "trace": trace,
                "plain_traced_max_abs": max(abs(a - b) for a, b in zip(plain, traced, strict=True)),
            }
        return report

    def _update(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("update requires a non-empty trajectory list")
        self.model.train()
        self._status("trainer_update_entered", global_rows=len(rows))

        def before_optimizer_step() -> None:
            assert self.dist is not None
            result_box: list[Any] = [None]
            self._status("parent_optimizer_fence_wait_started")
            if self.rank == 0:
                assert self.connection is not None and self.current_request_id is not None
                send_message(self.connection, {
                    "kind": "before_optimizer_step",
                    "request_id": self.current_request_id,
                    "phase": self.current_phase,
                })
                ack = receive_message(self.connection)
                if (not isinstance(ack, dict) or ack.get("kind") != "callback_ack"
                        or ack.get("request_id") != self.current_request_id):
                    result_box[0] = {"ok": False, "error": "invalid parent pre-optimizer acknowledgement"}
                elif ack.get("error") is not None:
                    result_box[0] = {"ok": False, "error": ack["error"]}
                else:
                    result_box[0] = {"ok": True}
                self._status("parent_optimizer_fence_ack_received", accepted=bool(result_box[0].get("ok")))
            if self.world_size > 1:
                self.dist.broadcast_object_list(result_box, src=0, group=self.gloo_group)
            self._status("parent_optimizer_fence_collective_completed", accepted=bool(result_box[0].get("ok")))
            if not result_box[0].get("ok"):
                error = result_box[0].get("error")
                raise RuntimeError(f"parent pre-optimizer callback rejected update: {error}")

        if self.mode == "pp":
            self.current_phase = "pipeline_update"
            self._status("pipeline_proximal_scoring_started", global_rows=len(rows))
            result = self.pipeline_compute.update_rows(
                rows,
                before_optimizer_step=before_optimizer_step,
            )
            update = result.get("update") if isinstance(result, dict) else None
            if not isinstance(update, dict):
                raise RankServiceError("PP compute adapter returned no update record")
            norm = update.get("global_gradient_norm_before_clip")
            if (isinstance(norm, bool) or not isinstance(norm, (int, float))
                    or not math.isfinite(float(norm)) or float(norm) <= 0.0):
                raise RankServiceError("PP compute adapter returned no finite nonzero global LoRA norm")
            self.trainer.gradient_norm_before_clip = float(norm)
            self.trainer._last_correction_metrics = dict(result.get("correction", {}))
            self.trainer._last_trainer_metrics = dict(result.get("metrics", {}))
            self._status(
                "pipeline_update_completed",
                global_step=int(self.trainer.state.global_step),
                global_gradient_norm_before_clip=float(norm),
            )
            self.current_phase = "post_optimizer_validation"
            return {
                "updated": bool(result.get("updated")),
                "metrics": dict(result.get("metrics", {})),
                "quality": dict(result.get("quality", {"available": False})),
                "correction": dict(result.get("correction", {})),
                "clip": dict(result.get("clip", {})),
                "gradient_norm_before_clip": float(norm),
                "gradient_norm_after_clip": update.get("global_gradient_norm_after_clip"),
                "optimizer_step_count": int(update.get("optimizer_step_count", 0)),
                "trainer_state": asdict(self.trainer.state),
                "telemetry": dict(result.get("telemetry", {})),
                "train_output": {"global_step": int(self.trainer.state.global_step)},
            }

        self.current_phase = "trainer_update"
        output = self.trainer.train_groups(
            rows,
            before_optimizer_step=before_optimizer_step,
        )
        self._status("trainer_update_completed", global_step=int(output.global_step))
        self.current_phase = "post_optimizer_validation"
        telemetry = self.trainer.telemetry.snapshot()
        return {
            "updated": True,
            "metrics": dict(self.trainer._last_trainer_metrics),
            "quality": self.trainer.quality_snapshot(),
            "correction": dict(self.trainer._last_correction_metrics),
            "clip": dict(self.trainer._last_upstream_clip_metrics),
            "gradient_norm_before_clip": self.trainer.gradient_norm_before_clip,
            "gradient_norm_after_clip": self.trainer._gradient_guard.gradient_norm_after_clip,
            "optimizer_step_count": self.trainer._gradient_guard.optimizer_step_count,
            "gradient_accumulation_steps": int(
                self.trainer._last_actual_gradient_accumulation_steps
            ),
            "microbatch_plan": copy.deepcopy(self.trainer._last_microbatch_plan_report),
            "trainer_state": asdict(self.trainer.state),
            "telemetry": telemetry,
            "train_output": {"global_step": int(output.global_step)},
        }

    def _snapshot(self) -> dict[str, Any]:
        if self.mode == "pp":
            self._status("pipeline_snapshot_capture_started")
            state = self.pipeline_compute.capture_state(self.pipeline_layout_signature)
            if state.get("layout_signature") != self.pipeline_layout_signature:
                raise RankServiceError("pipeline rank snapshot layout signature changed during capture")
            return {
                "pipeline_rank_state": state,
                "rank_layout": copy.deepcopy(self.pipeline_rank_layouts[self.rank]),
                "parameter_groups": self._named_parameter_groups(),
                "peft_config": self.pipeline_compute.peft_config(),
                "parallelism": "pp",
                "world_size": self.world_size,
                "layout_signature": self.pipeline_layout_signature,
                "evidence": copy.deepcopy(self.evidence),
            }
        import numpy as np
        import torch
        from peft import get_peft_model_state_dict
        snapshot_started = time.perf_counter()
        self._status("snapshot_adapter_collection_started", mode=self.mode)
        model = self._unwrap_model()
        is_t2_tp = self.mode == "tp" and self.config.get("model_backend") == "ternary"
        storage_attestation = None
        local_artifact_identity = None
        if self.config.get("model_backend", "nvfp4") == "ternary":
            identity = self.evidence.get("artifact_identity", {})
            local_artifact_identity = {
                "model_id": identity.get("model_id"),
                "artifact_id": identity.get("artifact_id"),
                "artifact_sha256": identity.get("artifact_sha256"),
                "conversion_report_sha256": identity.get("conversion_report_sha256"),
                "lora_input_basis": self.evidence.get("quantization", {}).get("lora_input_basis"),
            }
            if not local_artifact_identity["artifact_sha256"]:
                raise RankServiceError("T2 snapshot is missing its rank-local artifact identity")
        else:
            from qwen_rl.models.nvfp4 import nvfp4_local_packed_storage_attestation

            storage_attestation = nvfp4_local_packed_storage_attestation(
                model,
                expected_packed_names=self.evidence.get("peft_runtime_packed_module_names"),
            )
            trainer_global_step = int(getattr(self.trainer.state, "global_step", 0))
            storage_attestation.update({
                "parallelism": self.mode,
                "rank": self.rank,
                "trainer_global_step": trainer_global_step,
                "post_prepare": (
                    self.mode in {"single", "tp", "fsdp2"} or trainer_global_step > 0
                ),
            })
        adapter_collect_started = time.perf_counter()
        if is_t2_tp:
            from qwen_rl.models.ternary_tp_state import (
                collect_t2_tp_peft_state_dict,
                controller_named_adapter_state,
            )

            if self.t2_tp_placements is None:
                raise RankServiceError("T2 TP snapshot has no validated local placement map")
            canonical_state = collect_t2_tp_peft_state_dict(
                model,
                self.t2_tp_placements,
                rank=self.rank,
                world_size=self.world_size,
                group=self.gloo_group,
            )
            adapter_state = (
                controller_named_adapter_state(canonical_state)
                if self.rank == 0 and canonical_state is not None else None
            )
        elif self.mode == "tp":
            from qwen_rl.models.nvfp4 import nvfp4_peft_state_dict

            adapter_state = nvfp4_peft_state_dict(model, parallel_mode="tp", collect_on_this_rank=True)
        elif self.mode == "fsdp2":
            from qwen_rl.models.nvfp4 import nvfp4_peft_state_dict

            wrapped_model = getattr(self.trainer, "model_wrapped", None) or self.trainer.model
            adapter_state = nvfp4_peft_state_dict(
                wrapped_model,
                parallel_mode="fsdp2",
                collect_on_this_rank=self.rank == 0,
            )
        else:
            adapter_state = get_peft_model_state_dict(model, adapter_name="default")
        adapter_collection_seconds = time.perf_counter() - adapter_collect_started
        adapter_cpu_copy_started = time.perf_counter()
        if is_t2_tp:
            pass  # The TP collector returned the full controller-named CPU tensors on rank 0.
        elif self.mode == "fsdp2":
            if adapter_state is not None:
                adapter_state = self._full_adapter_state(adapter_state)
        elif self.rank != 0:
            adapter_state = None
        elif adapter_state is not None:
            adapter_state = self._full_adapter_state(adapter_state)
        adapter_cpu_copy_seconds = time.perf_counter() - adapter_cpu_copy_started
        adapter_cpu_copy_bytes = _tensor_payload_bytes(adapter_state)

        optimizer = self.trainer.optimizer
        optimizer_state_started = time.perf_counter()
        t2_tp_optimizer_contract = None
        optimizer_copy_stats: dict[str, int] = {}
        if is_t2_tp:
            from qwen_rl.models.ternary_tp_state import capture_t2_tp_optimizer_state

            if (self.t2_tp_placements is None or self.t2_tp_layout_signature is None
                    or self.t2_tp_layout_recipe is None):
                raise RankServiceError("T2 TP snapshot has no validated optimizer layout")
            captured = capture_t2_tp_optimizer_state(
                model,
                optimizer,
                self.t2_tp_placements,
                rank=self.rank,
                world_size=self.world_size,
                topology_signature=self.t2_tp_layout_signature,
                layout_recipe=self.t2_tp_layout_recipe,
            )
            optimizer_state = captured["optimizer_state"]
            t2_tp_optimizer_contract = {
                key: value for key, value in captured.items() if key != "optimizer_state"
            }
            optimizer_copy_stats["cpu_tensor_bytes"] = _tensor_payload_bytes(optimizer_state)
            optimizer_state_dict_seconds = time.perf_counter() - optimizer_state_started
            optimizer_cpu_copy_seconds = 0.0
        else:
            optimizer_payload = optimizer.state_dict()
            optimizer_state_dict_seconds = time.perf_counter() - optimizer_state_started
            optimizer_copy_started = time.perf_counter()
            optimizer_state = _cpu_tree(optimizer_payload, copy_stats=optimizer_copy_stats)
            optimizer_cpu_copy_seconds = time.perf_counter() - optimizer_copy_started
            del optimizer_payload

        scheduler_state_started = time.perf_counter()
        scheduler_payload = self.trainer.lr_scheduler.state_dict()
        scheduler_state_dict_seconds = time.perf_counter() - scheduler_state_started
        scheduler_copy_stats: dict[str, int] = {}
        scheduler_copy_started = time.perf_counter()
        scheduler_state = _cpu_tree(scheduler_payload, copy_stats=scheduler_copy_stats)
        scheduler_cpu_copy_seconds = time.perf_counter() - scheduler_copy_started
        del scheduler_payload

        rng_capture_started = time.perf_counter()
        trainer_state = asdict(self.trainer.state)
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state().cpu(),
            "torch_cuda": torch.cuda.get_rng_state(self.local_rank).cpu(),
        }
        rng_capture_seconds = time.perf_counter() - rng_capture_started
        rng_copy_stats: dict[str, int] = {}
        rng_copy_started = time.perf_counter()
        rng_state = _cpu_tree(rng_state, copy_stats=rng_copy_stats)
        rng_cpu_copy_seconds = time.perf_counter() - rng_copy_started

        metadata_started = time.perf_counter()
        layouts = self._local_parameter_layouts()
        parameter_groups = self._named_parameter_groups()
        evidence = copy.deepcopy(self.evidence)
        evidence["local_packed_storage_attestation"] = copy.deepcopy(storage_attestation)
        metadata_seconds = time.perf_counter() - metadata_started
        copy_bytes = (
            adapter_cpu_copy_bytes
            + optimizer_copy_stats.get("cpu_tensor_bytes", 0)
            + scheduler_copy_stats.get("cpu_tensor_bytes", 0)
            + rng_copy_stats.get("cpu_tensor_bytes", 0)
        )
        return {
            # _full_adapter_state already owns one complete CPU copy. A second
            # _cpu_tree pass here duplicated the full adapter before Gloo gather.
            "adapter_state": adapter_state,
            "local_storage_attestation": storage_attestation,
            "local_artifact_identity": local_artifact_identity,
            "peft_config": self._peft_configs(),
            "parameter_groups": parameter_groups,
            "rank_layout": layouts,
            "optimizer_state": optimizer_state,
            "t2_tp_optimizer_contract": t2_tp_optimizer_contract,
            "scheduler_state": scheduler_state,
            "rng_state": rng_state,
            "trainer_state": trainer_state,
            "evidence": evidence,
            "parallelism": self.mode,
            "world_size": self.world_size,
            "t2_tp_layout_signature": self.t2_tp_layout_signature if is_t2_tp else None,
            "t2_tp_layout_recipe": self.t2_tp_layout_recipe if is_t2_tp else None,
            "layout_signature": self._layout_signature(layouts),
            "checkpoint_performance": {
                "rank": self.rank,
                "adapter_collection_seconds": adapter_collection_seconds,
                "adapter_cpu_copy_seconds": adapter_cpu_copy_seconds,
                "adapter_cpu_copy_bytes": adapter_cpu_copy_bytes,
                "optimizer_state_dict_seconds": optimizer_state_dict_seconds,
                "optimizer_cpu_copy_seconds": optimizer_cpu_copy_seconds,
                "optimizer_cpu_copy_bytes": optimizer_copy_stats.get("cpu_tensor_bytes", 0),
                "scheduler_state_dict_seconds": scheduler_state_dict_seconds,
                "scheduler_cpu_copy_seconds": scheduler_cpu_copy_seconds,
                "scheduler_cpu_copy_bytes": scheduler_copy_stats.get("cpu_tensor_bytes", 0),
                "rng_capture_seconds": rng_capture_seconds,
                "rng_cpu_copy_seconds": rng_cpu_copy_seconds,
                "rng_cpu_copy_bytes": rng_copy_stats.get("cpu_tensor_bytes", 0),
                "metadata_seconds": metadata_seconds,
                "rank_tensor_copy_bytes": copy_bytes,
                "rank_snapshot_total_seconds": time.perf_counter() - snapshot_started,
            },
        }

    def _restore(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "pp":
            return self._restore_pipeline(snapshot)
        import numpy as np
        import torch
        from transformers.trainer_callback import TrainerState

        if not isinstance(snapshot, dict):
            raise ValueError("parallel snapshot must be a dictionary")
        if int(snapshot.get("world_size", -1)) != self.world_size:
            raise ValueError("parallel snapshot world size differs from the active topology")
        if snapshot.get("parallelism") != self.mode:
            raise ValueError("parallel snapshot topology differs from the active topology")
        rank_layouts = snapshot.get("rank_layouts")
        if not isinstance(rank_layouts, list) or len(rank_layouts) != self.world_size:
            raise ValueError("parallel snapshot does not contain every rank layout")
        if snapshot.get("layout_signature") != self._layout_signature(rank_layouts):
            raise ValueError("parallel snapshot layout signature is corrupt")
        optimizer_states = snapshot.get("optimizer_states")
        scheduler_states = snapshot.get("scheduler_states")
        rng_states = snapshot.get("rng_states")
        if not all(isinstance(value, list) and len(value) == self.world_size for value in (
            optimizer_states, scheduler_states, rng_states
        )):
            raise ValueError("parallel snapshot must contain complete per-rank optimizer, scheduler, and RNG states")
        model = self._unwrap_model()
        adapter_state = snapshot.get("adapter_state")
        if not isinstance(adapter_state, dict):
            raise ValueError("parallel snapshot has no full PEFT adapter state")
        is_t2_tp = self.mode == "tp" and self.config.get("model_backend") == "ternary"
        if is_t2_tp:
            from qwen_rl.models.ternary_tp_state import (
                canonical_parameter_name,
                load_t2_tp_peft_state_dict,
                restore_t2_tp_optimizer_state,
                validate_t2_tp_optimizer_world,
            )

            if (self.t2_tp_placements is None or self.t2_tp_layout_signature is None
                    or self.t2_tp_layout_recipe is None):
                raise ValueError("active T2 TP model has no validated parameter layout")
            layout_signature = snapshot.get("t2_tp_layout_signature")
            layout_recipe = snapshot.get("t2_tp_layout_recipe")
            if (layout_signature != self.t2_tp_layout_signature
                    or layout_recipe != self.t2_tp_layout_recipe):
                raise ValueError("T2 TP snapshot topology differs from the active model layout")
            rank_optimizer_snapshots = []
            for rank, rank_layout in enumerate(rank_layouts):
                if not isinstance(rank_layout, dict) or rank_layout.get("rank") != rank:
                    raise ValueError("T2 TP snapshot rank layouts are missing or reordered")
                contract = rank_layout.get("t2_tp_optimizer_contract")
                if not isinstance(contract, dict):
                    raise ValueError(f"T2 TP snapshot rank {rank} has no optimizer layout contract")
                local_parameters = rank_layout.get("parameters")
                if not isinstance(local_parameters, list):
                    raise ValueError(f"T2 TP snapshot rank {rank} has no local parameter inventory")
                generic_by_name = {}
                for item in local_parameters:
                    if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                        raise ValueError(f"T2 TP snapshot rank {rank} has malformed parameter metadata")
                    name = canonical_parameter_name(item["name"])
                    if name in generic_by_name:
                        raise ValueError(f"T2 TP snapshot rank {rank} has duplicate canonical parameters")
                    generic_by_name[name] = item
                contract_parameters = contract.get("parameters")
                if not isinstance(contract_parameters, list):
                    raise ValueError(f"T2 TP snapshot rank {rank} has malformed placement inventory")
                contract_by_name = {item.get("name"): item for item in contract_parameters
                                    if isinstance(item, dict)}
                if set(generic_by_name) != set(contract_by_name):
                    raise ValueError(f"T2 TP snapshot rank {rank} parameter inventories disagree")
                for name, item in generic_by_name.items():
                    contract_item = contract_by_name[name]
                    if (tuple(item.get("shape", ())) != tuple(contract_item.get("shape", ()))
                            or item.get("dtype") != contract_item.get("dtype")):
                        raise ValueError(f"T2 TP snapshot rank {rank} parameter geometry differs for {name}")
                rank_optimizer_snapshots.append({
                    **contract,
                    "optimizer_state": optimizer_states[rank],
                })
            validate_t2_tp_optimizer_world(
                rank_optimizer_snapshots,
                topology_signature=layout_signature,
                layout_recipe=layout_recipe,
            )
            load_t2_tp_peft_state_dict(
                model,
                adapter_state,
                self.t2_tp_placements,
                rank=self.rank,
                world_size=self.world_size,
            )
            restore_t2_tp_optimizer_state(
                model,
                self.trainer.optimizer,
                rank_optimizer_snapshots[self.rank],
                self.t2_tp_placements,
                rank=self.rank,
                world_size=self.world_size,
                topology_signature=layout_signature,
                layout_recipe=layout_recipe,
            )
        else:
            self._load_adapter_state(model, adapter_state)
            self.trainer.optimizer.load_state_dict(optimizer_states[self.rank])
            if self.mode == "fsdp2":
                _restore_fsdp2_optimizer_moments(self.trainer.optimizer)
        self.trainer.lr_scheduler.load_state_dict(scheduler_states[self.rank])
        self.trainer.state = TrainerState(**snapshot["trainer_state"])
        rng = rng_states[self.rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.random.set_rng_state(rng["torch_cpu"])
        torch.cuda.set_rng_state(rng["torch_cuda"], self.local_rank)
        return {"restored": True, "rank": self.rank, "global_step": self.trainer.state.global_step}

    def _restore_pipeline(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        from qwen_rl.learner.parallel.pipeline_exports import validate_ordered_optimizer_parameter_group_signature

        if not isinstance(snapshot, dict):
            raise ValueError("pipeline snapshot must be a dictionary")
        if snapshot.get("parallelism") != "pp" or snapshot.get("world_size") != self.world_size:
            raise ValueError("pipeline snapshot topology differs from the active four-stage world")
        rank_layouts = snapshot.get("rank_layouts")
        if not isinstance(rank_layouts, list) or len(rank_layouts) != self.world_size:
            raise ValueError("pipeline snapshot does not contain all four rank layouts")
        if self._layout_signature(rank_layouts) != self.pipeline_layout_signature:
            raise ValueError("pipeline optimizer/parameter group layout differs from the active stages")
        if snapshot.get("layout_signature") != self.pipeline_layout_signature:
            raise ValueError("pipeline snapshot layout signature differs from the active stages")
        rank_parameter_groups = snapshot.get("rank_parameter_groups")
        if (not isinstance(rank_parameter_groups, list) or len(rank_parameter_groups) != self.world_size
                or rank_parameter_groups[self.rank] != self._named_parameter_groups()):
            raise ValueError("pipeline rank-local optimizer parameter ordering changed")
        validate_ordered_optimizer_parameter_group_signature(
            self.model,
            self.trainer.optimizer,
            rank_layouts[self.rank].get("ordered_optimizer_parameter_group_signature"),
        )
        optimizer_states = snapshot.get("optimizer_states")
        scheduler_states = snapshot.get("scheduler_states")
        rng_states = snapshot.get("rng_states")
        if not all(isinstance(value, list) and len(value) == self.world_size for value in (
            optimizer_states, scheduler_states, rng_states
        )):
            raise ValueError("pipeline snapshot is missing rank-local optimizer, scheduler, or RNG state")
        adapter_state = snapshot.get("adapter_state")
        if not isinstance(adapter_state, dict) or set(adapter_state) != self.pipeline_expected_adapter_names:
            raise ValueError("pipeline snapshot does not contain the complete merged LoRA adapter")
        trainer_state = snapshot.get("trainer_state")
        if not isinstance(trainer_state, dict):
            raise ValueError("pipeline snapshot is missing shared Trainer progress state")

        local_state = self.pipeline_compute.capture_state(self.pipeline_layout_signature)
        local_names = set(local_state["adapter_state"])
        if not local_names.issubset(adapter_state):
            raise ValueError("global pipeline adapter is missing this stage's LoRA tensors")
        local_state.update({
            "adapter_state": {name: adapter_state[name] for name in local_names},
            "optimizer_state": optimizer_states[self.rank],
            "scheduler_state": scheduler_states[self.rank],
            "rng_state": rng_states[self.rank],
            "trainer_state": trainer_state,
        })
        restored = self.pipeline_compute.restore_state(local_state, self.pipeline_layout_signature)
        return {
            "restored": True,
            "rank": self.rank,
            "global_step": int(restored.get("global_step", 0)),
        }

    def _load_adapter_state(self, model, state: dict[str, Any]) -> None:
        if self.config.get("model_backend", "nvfp4") == "ternary":
            from .t2_model_assembly import load_t2_adapter_state

            load_t2_adapter_state(model, state)
            return
        from qwen_rl.models.nvfp4 import load_nvfp4_peft_state_dict

        loader_mode = self.mode if self.mode in {"single", "tp", "fsdp2"} else "replicated"
        load_nvfp4_peft_state_dict(
            model,
            state,
            parallel_mode=loader_mode,
            rank=self.rank,
        )

    def _export_adapter(self, destination: str) -> dict[str, Any]:
        from pathlib import Path
        import shutil

        if self.mode == "pp":
            return {
                "rank": self.rank,
                "destination": str(Path(destination).resolve()),
                "adapter_state": self.pipeline_compute.adapter_state_dict(),
                "peft_config": self.pipeline_compute.peft_config(),
                "exported": False,
            }
        model = self._unwrap_model()
        if self.mode == "tp" and self.config.get("model_backend") == "ternary":
            from qwen_rl.models.ternary_tp_state import collect_t2_tp_peft_state_dict

            if self.t2_tp_placements is None:
                raise RankServiceError("T2 TP export has no validated local placement map")
            state = collect_t2_tp_peft_state_dict(
                model,
                self.t2_tp_placements,
                rank=self.rank,
                world_size=self.world_size,
                group=self.gloo_group,
            )
            if self.rank != 0:
                return {"exported": False, "rank": self.rank}
            temporary = Path(destination).with_name(Path(destination).name + ".parallel-tmp")
            temporary.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(temporary, state_dict=state, safe_serialization=True)
        elif self.mode in {"tp", "fsdp2"}:
            from qwen_rl.models.nvfp4 import nvfp4_peft_state_dict

            if self.mode == "fsdp2":
                state_model = getattr(self.trainer, "model_wrapped", None) or self.trainer.model
                state = nvfp4_peft_state_dict(
                    state_model,
                    parallel_mode="fsdp2",
                    collect_on_this_rank=self.rank == 0,
                )
            else:
                state = nvfp4_peft_state_dict(model, parallel_mode="tp", collect_on_this_rank=True)
            if self.rank != 0:
                return {"exported": False, "rank": self.rank}

            # Keep standard PEFT serialization after the topology-specific
            # state collector returns the official canonical adapter inventory.
            temporary = Path(destination).with_name(Path(destination).name + ".parallel-tmp")
            temporary.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(temporary, state_dict=state, safe_serialization=True)
        elif self.rank == 0:
            temporary = Path(destination).with_name(Path(destination).name + ".parallel-tmp")
            temporary.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(temporary, safe_serialization=True)
        else:
            return {"exported": False, "rank": self.rank}
        target = Path(destination)
        if target.exists():
            shutil.rmtree(temporary)
            raise FileExistsError(f"adapter destination already exists: {target}")
        temporary.replace(target)
        return {"exported": True, "path": str(target), "rank": self.rank}

    def _unwrap_model(self):
        model = self.trainer.model
        accelerator = getattr(self.trainer, "accelerator", None)
        if accelerator is not None:
            try:
                model = accelerator.unwrap_model(model)
            except Exception:
                pass
        return model

    def _full_adapter_state(self, state: dict[str, Any]) -> dict[str, Any]:
        import torch
        from torch.distributed.tensor import DTensor

        named_lora = {
            name.replace(".default.", "."): name
            for name, parameter in self._unwrap_model().named_parameters()
            if parameter.requires_grad and "lora_" in name
        }
        full: dict[str, Any] = {}
        for name in sorted(state):
            tensor = state[name]
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            full_name = named_lora.get(name)
            if full_name is None:
                raise RankServiceError(f"PEFT state key has no full named LoRA parameter: {name}")
            full[full_name] = tensor.detach().to(device="cpu", copy=True) if isinstance(tensor, torch.Tensor) else tensor
        if set(full) != {name for name in named_lora.values()}:
            missing = sorted({name for name in named_lora.values()} - set(full))
            raise RankServiceError(f"PEFT state omitted trainable LoRA parameters: {missing[:8]}")
        return full

    def _peft_configs(self) -> dict[str, Any]:
        model = self._unwrap_model()
        configs = getattr(model, "peft_config", None)
        if not isinstance(configs, dict):
            raise RankServiceError("loaded policy has no PEFT config mapping")
        return {name: config.to_dict() for name, config in configs.items()}

    def _local_parameter_layouts(self) -> list[dict[str, Any]]:
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:  # pragma: no cover
            DTensor = ()  # type: ignore[assignment]
        layouts = []
        for index, (name, parameter) in enumerate(self._unwrap_model().named_parameters()):
            if not parameter.requires_grad:
                continue
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            layouts.append({
                "order": index,
                "name": name,
                "shape": list(local.shape),
                "global_shape": list(parameter.shape),
                "local_shape": list(local.shape),
                "dtype": str(parameter.dtype),
                "placements": [str(item) for item in parameter.placements] if isinstance(parameter, DTensor) else [],
            })
        return layouts

    def _named_parameter_groups(self) -> list[dict[str, Any]]:
        model = self._unwrap_model()
        names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
        result = []
        for group in self.trainer.optimizer.param_groups:
            names = []
            for parameter in group["params"]:
                name = names_by_id.get(id(parameter))
                if name is None:
                    # FSDP2 may expose original parameters via the optimizer
                    # even when its wrapper replaced the named parameter view.
                    name = getattr(parameter, "_fqns", None)
                    if isinstance(name, (list, tuple)) and len(name) == 1:
                        name = name[0]
                if not isinstance(name, str):
                    raise RankServiceError("optimizer parameter has no stable model name")
                names.append(name)
            options = {key: _json_safe(value) for key, value in group.items() if key != "params"}
            result.append({"parameter_names": names, "options": options})
        return result

    @staticmethod
    def _layout_signature(rank_layouts: list[Any]) -> str:
        import hashlib

        encoded = json.dumps(_json_safe(rank_layouts), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _combine_results(self, operation: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        if operation in {"park", "wake"}:
            expected = "asleep" if operation == "park" else "awake"
            if len(results) != self.world_size or any(item.get("state") != expected for item in results):
                raise RankServiceError("ranks did not complete the same residency transition")
            return {"state": "parked" if operation == "park" else "awake", "ranks": results}
        if operation == "execution_diagnostics":
            from qwen_rl.models.ternary_tp_diagnostics import (
                combine_t2_tp_execution_diagnostics,
            )

            try:
                return combine_t2_tp_execution_diagnostics(
                    results,
                    parallelism=self.mode,
                    model_backend=str(self.config.get("model_backend", "")),
                    world_size=self.world_size,
                )
            except (TypeError, ValueError) as exc:
                raise RankServiceError(f"invalid T2 TP execution diagnostics: {exc}") from exc
        if operation == "snapshot":
            if self.mode == "pp":
                return self._combine_pipeline_snapshots(results)
            snapshots = results
            first = snapshots[0]
            is_t2_tp = self.mode == "tp" and self.config.get("model_backend") == "ternary"
            rank_performance = [
                item.pop("checkpoint_performance", {}) for item in snapshots
            ]
            if any(item["adapter_state"] is not None for item in snapshots[1:]):
                for item in snapshots[1:]:
                    if item["adapter_state"] is not None:
                        raise RankServiceError("more than rank 0 exported full adapter state")
            if first["adapter_state"] is None:
                raise RankServiceError("rank 0 did not produce full adapter state")
            if is_t2_tp:
                from qwen_rl.models.ternary_tp_state import (
                    canonical_parameter_name,
                    controller_parameter_name,
                    validate_t2_tp_optimizer_world,
                )
            rank_layouts = [
                {
                    "rank": rank,
                    "parameters": [
                        {
                            **{key: value for key, value in parameter.items()
                               if key in {"name", "shape", "dtype"}},
                            **({
                                "name": controller_parameter_name(
                                    canonical_parameter_name(parameter["name"])
                                )
                            } if is_t2_tp else {}),
                        }
                        for parameter in item["rank_layout"]
                    ],
                    **({"t2_tp_optimizer_contract": item.get("t2_tp_optimizer_contract")}
                       if is_t2_tp else {}),
                }
                for rank, item in enumerate(snapshots)
            ]
            first["rank_layouts"] = rank_layouts
            if is_t2_tp:
                layout_signature = first.get("t2_tp_layout_signature")
                layout_recipe = first.get("t2_tp_layout_recipe")
                tp_optimizer_snapshots = []
                for rank, item in enumerate(snapshots):
                    contract = item.get("t2_tp_optimizer_contract")
                    if not isinstance(contract, dict):
                        raise RankServiceError(f"T2 TP rank {rank} omitted optimizer layout metadata")
                    tp_optimizer_snapshots.append({
                        **contract,
                        "optimizer_state": item.get("optimizer_state"),
                    })
                try:
                    validate_t2_tp_optimizer_world(
                        tp_optimizer_snapshots,
                        topology_signature=layout_signature,
                        layout_recipe=layout_recipe,
                    )
                except Exception as exc:
                    raise RankServiceError(f"T2 TP rank optimizer snapshots are incompatible: {exc}") from exc
                if any(
                    item.get("t2_tp_layout_signature") != layout_signature
                    or item.get("t2_tp_layout_recipe") != layout_recipe
                    for item in snapshots
                ):
                    raise RankServiceError("T2 TP ranks disagree on their validated parameter layout")
            if self.config.get("model_backend", "nvfp4") == "ternary":
                identities = [item.get("local_artifact_identity") for item in snapshots]
                if (any(not isinstance(item, dict) for item in identities)
                        or any(item != identities[0] for item in identities[1:])):
                    raise RankServiceError("T2 DDP ranks disagree on the identity-bound frozen base")
                first["rank_artifact_identities"] = identities
                first.pop("local_artifact_identity", None)
                first.pop("local_storage_attestation", None)
            else:
                storage_attestations = [item.get("local_storage_attestation") for item in snapshots]
                if any(not isinstance(item, dict) or item.get("status") != "verified" for item in storage_attestations):
                    raise RankServiceError("snapshot is missing a verified local NVFP4 storage attestation")
                packed_name_sets = [
                    {record.get("name") for record in item.get("packed_tensors", [])}
                    for item in storage_attestations
                ]
                if any(names != packed_name_sets[0] for names in packed_name_sets[1:]):
                    raise RankServiceError("ranks disagree on post-prepare packed NVFP4 module FQNs")
                first["rank_storage_attestations"] = storage_attestations
                first["post_prepare_storage_attested_all_ranks"] = all(
                    item.get("post_prepare") is True for item in storage_attestations
                )
            optimizer_dedup = None
            if self.mode == "ddp":
                from .snapshot_dedup import reconstruct_ddp_optimizer_states

                try:
                    first["optimizer_states"], optimizer_dedup = reconstruct_ddp_optimizer_states(
                        snapshots, world_size=self.world_size
                    )
                except Exception as exc:
                    raise RankServiceError(
                        f"cannot reconstruct pre-Gloo DDP optimizer snapshot slots: {exc}"
                    ) from exc
            else:
                first["optimizer_states"] = [item["optimizer_state"] for item in snapshots]
            first["scheduler_states"] = [item["scheduler_state"] for item in snapshots]
            first["rng_states"] = [item["rng_state"] for item in snapshots]
            parameter_groups = copy.deepcopy(snapshots[0]["parameter_groups"])
            if is_t2_tp:
                for group in parameter_groups:
                    group["parameter_names"] = [
                        controller_parameter_name(canonical_parameter_name(name))
                        for name in group["parameter_names"]
                    ]
            first["parameter_groups"] = parameter_groups
            first["layout_signature"] = self._layout_signature(rank_layouts)
            first.pop("rank_layout", None)
            first.pop("optimizer_state", None)
            first.pop("t2_tp_optimizer_contract", None)
            first.pop("scheduler_state", None)
            first.pop("rng_state", None)
            first["checkpoint_performance"] = {
                "schema": "qwen-rl-checkpoint-performance/v1",
                "rank_local": rank_performance,
                "optimizer_snapshot_dedup": optimizer_dedup,
                "rank_tensor_copy_bytes": sum(
                    int(item.get("rank_tensor_copy_bytes", 0))
                    for item in rank_performance
                    if isinstance(item, dict)
                ),
                "rank_gather_scope": (
                    "signature/readiness collectives, small-envelope gather_object, and unique "
                    "optimizer-state P2P transfer; rank-0 optimizer/adapter trees stay local"
                ),
            }
            return first
        if operation == "update":
            first = dict(results[-1] if self.mode == "pp" else results[0])
            rank_corrections = [copy.deepcopy(item["correction"]) for item in results]
            if self.mode in {"ddp", "fsdp2"}:
                combined_correction = self._combine_corrections(rank_corrections)
            else:
                combined_correction = copy.deepcopy(rank_corrections[0])
            combined_correction["measurement_scope"] = "global"
            if len(rank_corrections) > 1 and self.mode != "pp":
                combined_correction["rank_corrections"] = rank_corrections
            first["correction"] = combined_correction
            first["quality"] = copy.deepcopy(first["quality"])
            if first["quality"].get("available"):
                first["quality"]["importance_correction"] = copy.deepcopy(first["correction"])
                fresh = first["correction"].get("fresh_policy_tokens")
                capped = first["correction"].get("truncated_fraction")
                if isinstance(fresh, int) and fresh > 0 and isinstance(capped, (int, float)):
                    first["quality"]["importance_capped_fraction"] = float(capped)
                    first["quality"]["importance_capped_fraction_denominator_fresh_policy_tokens"] = fresh
            first["telemetry"] = self._combine_telemetry(
                [item["telemetry"] for item in results], mode=self.mode
            )
            first["status"] = {
                "world_size": self.world_size,
                "completed_optimizer_steps": [item["optimizer_step_count"] for item in results],
                "rank_gradient_norms": [item["gradient_norm_before_clip"] for item in results],
                "ranks": [item["telemetry"] for item in results],
            }
            if any(item["optimizer_step_count"] != 1 for item in results):
                raise RankServiceError("not every rank completed exactly one guarded optimizer step")
            return first
        if operation == "logprobs":
            first = results[0]
            if any(item["logprobs"] != first["logprobs"] for item in results[1:]):
                raise RankServiceError("replicated policy ranks returned different log probabilities")
            return first
        if operation == "probe":
            return {
                "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
                "lengths": results[0]["lengths"],
                "selected_policy_tokens": results[0]["selected_policy_tokens"],
                "rank_seconds": [item["seconds"] for item in results],
                "ranks": self.world_size,
            }
        if operation == "export_adapter":
            if self.mode == "pp":
                return self._combine_pipeline_export(results)
            exported = [item for item in results if item.get("exported")]
            if len(exported) != 1:
                raise RankServiceError("adapter export must succeed exactly on rank 0")
            return exported[0]
        if operation == "restore":
            if not all(item.get("restored") for item in results):
                raise RankServiceError("adapter restore failed to complete on every rank")
            return {"restored": True, "world_size": self.world_size, "global_step": results[0]["global_step"]}
        return results[0]

    def _combine_pipeline_export(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        from pathlib import Path
        import shutil

        from qwen_rl.learner.parallel.pipeline_exports import merge_pipeline_adapter_payload
        from qwen_rl.learner.parallel_engine import _AdapterView

        if len(results) != self.world_size or self.world_size != 4:
            raise RankServiceError("PP export requires one adapter shard from each of four stages")
        destinations = {item.get("destination") for item in results}
        if len(destinations) != 1 or not isinstance(next(iter(destinations)), str):
            raise RankServiceError("PP export ranks disagree on the destination path")
        destination = Path(next(iter(destinations)))
        rank_configs = [item.get("peft_config") for item in results]
        rank_states = [{"adapter_state": item.get("adapter_state")} for item in results]
        merged_payload = merge_pipeline_adapter_payload(rank_configs, rank_states)
        state = merged_payload["adapter_state"]
        peft_config = merged_payload["peft_config"]
        if set(state) != self.pipeline_expected_adapter_names:
            raise RankServiceError("PP exported adapter inventory differs from the live stage union")
        if destination.exists():
            raise FileExistsError(f"adapter destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".parallel-tmp")
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            adapter = _AdapterView({"adapter_state": state, "peft_config": peft_config})
            adapter.save_pretrained(
                temporary,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            temporary.replace(destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {
            "exported": True,
            "path": str(destination),
            "files": sorted(path.name for path in destination.iterdir() if path.is_file()),
            "adapter_parameter_count": len(state),
            "target_module_count": len(peft_config["default"]["target_modules"]),
        }

    def _combine_pipeline_snapshots(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        from qwen_rl.learner.parallel.pipeline_exports import merge_pipeline_adapter_payload

        if len(results) != self.world_size or self.world_size != 4:
            raise RankServiceError("PP snapshot must contain one result from each of four stages")
        rank_states = [item.get("pipeline_rank_state") for item in results]
        rank_layouts = [item.get("rank_layout") for item in results]
        rank_groups = [item.get("parameter_groups") for item in results]
        configs = [item.get("peft_config") for item in results]
        if any(not isinstance(state, dict) for state in rank_states):
            raise RankServiceError("PP snapshot is missing a local rank state")
        if any(not isinstance(layout, dict) or layout.get("rank") != rank
               for rank, layout in enumerate(rank_layouts)):
            raise RankServiceError("PP snapshot rank layouts are missing or reordered")
        layout_signature = self._layout_signature(rank_layouts)
        if layout_signature != self.pipeline_layout_signature:
            raise RankServiceError("PP snapshot local layout differs from startup layout")
        if any(state.get("layout_signature") != layout_signature for state in rank_states):
            raise RankServiceError("PP local rank snapshots disagree on the global layout signature")
        merged_payload = merge_pipeline_adapter_payload(configs, rank_states)
        adapter_state = merged_payload["adapter_state"]
        peft_config = merged_payload["peft_config"]
        if set(adapter_state) != self.pipeline_expected_adapter_names:
            raise RankServiceError("PP merged adapter inventory differs from the live stage union")
        if len(rank_groups) != self.world_size or any(not isinstance(groups, list) or not groups for groups in rank_groups):
            raise RankServiceError("PP snapshot is missing rank-local optimizer parameter groups")
        parameter_groups = [
            {"rank": rank, **copy.deepcopy(group)}
            for rank, groups in enumerate(rank_groups)
            for group in groups
        ]
        trainer_states = [state.get("trainer_state") for state in rank_states]
        trainer_signature = json.dumps(_json_safe(trainer_states[0]), sort_keys=True, separators=(",", ":"))
        if any(json.dumps(_json_safe(state), sort_keys=True, separators=(",", ":")) != trainer_signature
               for state in trainer_states[1:]):
            raise RankServiceError("PP ranks disagree on Trainer progress state")
        first = {
            "adapter_state": adapter_state,
            "peft_config": peft_config,
            "parallelism": "pp",
            "world_size": self.world_size,
            "layout_signature": layout_signature,
            "rank_layouts": rank_layouts,
            "parameter_groups": parameter_groups,
            "rank_parameter_groups": rank_groups,
            "optimizer_states": [state["optimizer_state"] for state in rank_states],
            "scheduler_states": [state["scheduler_state"] for state in rank_states],
            "rng_states": [state["rng_state"] for state in rank_states],
            "trainer_state": trainer_states[0],
            "evidence": copy.deepcopy(results[0].get("evidence", {})),
            "rank_evidence": [copy.deepcopy(item.get("evidence", {})) for item in results],
        }
        return first

    @staticmethod
    def _combine_corrections(items: list[dict[str, Any]]) -> dict[str, Any]:
        items = [item for item in items if isinstance(item, dict) and item]
        if not items:
            return {}
        result = {key: items[0][key] for key in (
            "mode", "cap", "target_scoring_strategy", "replay_correction", "approximate_correction"
        ) if key in items[0]}
        additive = (
            "policy_tokens", "fresh_policy_tokens", "replay_policy_tokens",
            "eligible_completion_targets", "policy_targets_projected",
            "non_policy_completion_targets_skipped",
        )
        for key in additive:
            values = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
            if values:
                result[key] = int(sum(values))
        kinds: dict[str, int] = {}
        for item in items:
            value = item.get("policy_tokens_by_kind", {})
            if isinstance(value, dict):
                for name, count in value.items():
                    if isinstance(count, (int, float)):
                        kinds[name] = kinds.get(name, 0) + int(count)
        if kinds:
            result["policy_tokens_by_kind"] = kinds
        fresh_total = int(result.get("fresh_policy_tokens", 0))
        weighted_truncated = [
            (float(item["truncated_fraction"]) * int(item.get("fresh_policy_tokens", 0)))
            for item in items
            if isinstance(item.get("truncated_fraction"), (int, float))
            and isinstance(item.get("fresh_policy_tokens"), (int, float))
        ]
        if fresh_total > 0 and weighted_truncated:
            result["truncated_fraction"] = sum(weighted_truncated) / fresh_total
        for key, op in (("weight_min", min), ("weight_max", max)):
            values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
            if values:
                result[key] = op(values)
        weighted_means = [
            (float(item["weight_mean"]), int(item.get("fresh_policy_tokens", 0)))
            for item in items
            if isinstance(item.get("weight_mean"), (int, float))
            and isinstance(item.get("fresh_policy_tokens"), (int, float))
            and int(item["fresh_policy_tokens"]) > 0
        ]
        if weighted_means:
            denominator = sum(count for _, count in weighted_means)
            result["weight_mean"] = sum(mean * count for mean, count in weighted_means) / denominator
        if len(items) > 1:
            result["per_rank_log_ratio_quantiles"] = [
                {
                    key: item[key]
                    for key in ("log_ratio_min", "log_ratio_median", "log_ratio_p95", "log_ratio_max")
                    if key in item
                }
                for item in items
            ]
        else:
            result.update({
                key: items[0][key]
                for key in ("log_ratio_min", "log_ratio_median", "log_ratio_p95", "log_ratio_max")
                if key in items[0]
            })
        return result

    @staticmethod
    def _combine_telemetry(items: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
        if not items:
            return {}
        result = copy.deepcopy(items[0])
        if mode in {"ddp", "fsdp2"}:
            for key in ("forward_input_tokens", "forward_policy_tokens", "loss_forwards"):
                values = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
                if values:
                    result[key] = sum(values)
            result["progress_measurement_scope"] = "rank0_partial"
        result["measurement_scope"] = "global"
        result["rank_telemetry"] = copy.deepcopy(items)
        return result


def _read_runtime(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("parallel runtime config must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    from ..cuda_allocator import configure_learner_allocator
    configure_learner_allocator()
    if sys.platform == "linux":
        expected_parent = int(os.environ.get(
            "QWEN_RL_PARALLEL_RANK_PARENT_PID",
            os.environ.get("QWEN_RL_PARALLEL_TORCHRUN_PID", "-1"),
        ))
        parent_pid = os.getppid()
        if parent_pid != expected_parent:
            return 1
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if os.getppid() != parent_pid:
            return 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    runtime = _read_runtime(args.config)
    runtime_config = runtime["config"]
    topology = runtime["topology"]
    if (runtime_config.get("gpu_sharing") == "phase_shared"
            and runtime_config.get("model_backend") == "ternary"
            and topology.get("mode") == "tp"):
        from .rank_cache import configure_t2_tp_compiler_cache

        configure_t2_tp_compiler_cache(
            runtime_config.get("recovery_dir"),
            os.environ.get("RANK"),
            os.environ.get("WORLD_SIZE"),
            expected_world_size=topology.get("world_size"),
        )
    residency = None
    if runtime_config.get("gpu_sharing") == "phase_shared":
        import torch
        import torch.distributed as dist
        from ..cuda_residency import CudaResidency
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank),
            timeout=timedelta(seconds=float(topology["compute_timeout"])))
        dist.barrier()
        from ..shared_frozen_backup import FrozenBackupStore
        # runtime_dir is unique per controller service and removed on shutdown.
        # No reuse across lineages, topologies or unrelated jobs.
        frozen_store = FrozenBackupStore(Path(runtime["runtime_dir"]) / "frozen-backups")
        residency = CudaResidency(device=local_rank, frozen_backup_store=frozen_store,
            host_backup_budget_bytes=int(runtime["config"]["residency_host_budget_gib"] * 1024**3))
    service = RankService(runtime,
        allocation_context_factory=residency.allocation_context if residency else None,
        residency=residency)
    service.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
