"""Thin adapter from complete external rollouts to TRL's ordinary GRPOTrainer."""
from __future__ import annotations

import copy
import logging
import math
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import (
    RepeatSampler,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    unsplit_pixel_values_by_grid,
)

from ..contracts.activation_offload import DEFAULT_OFFLOAD_MIN_TOKENS, offload_decision
from ..models.gdn_warmup import GDNWarmupError, warmup_gdn_l2norm_backward
from .saved_tensor_offload import saved_tensor_offload_context
from .telemetry import TrainingTelemetry
from .microbatch import plan_microbatches
from .proximal import ProximalAccumulator
from .proximal_forward import InlineProximalWindow
from .deferred_trl_metrics import (
    DeferredDapoMetricWindow,
    capture_trl_dapo_metric_collectives,
)
from .prepared_groups import (
    PreparedGradientRows,
    rare_success_advantage_scales,
    prepare_gradient_rows as prepare_complete_gradient_rows,
)

_ROW_KIND_TO_ID = {"fresh_student": 0, "native_replay": 1, "guided_replay": 2, "zero_credit": 3}
_ID_TO_ROW_KIND = {value: key for key, value in _ROW_KIND_TO_ID.items()}
_PROMPT_ALIGNED_FIELDS = {"prompt_ids", "prompt_mask"}
_COMPLETION_ALIGNED_FIELDS = {
    "completion_ids",
    "completion_mask",
    "tool_mask",
    "sampling_per_token_logps",
    "old_per_token_logps",
    "ref_per_token_logps",
    "importance_sampling_ratio",
    "behavior_correction",
    "per_token_logps",
    "per_token_entropies",
    "entropies",
    "loss_mask",
}
_FULL_SEQUENCE_ALIGNED_FIELDS = {
    "labels",
    "position_ids",
    "token_type_ids",
    "mm_token_type_ids",
    "image_position_ids",
}
_NON_TOKEN_FIELDS = {
    "image_grid_thw",
    "image_sizes",
    "pixel_attention_mask",
    "pixel_values",
    "spatial_shapes",
}

_TRL_HEAD_INIT_LOCK = threading.RLock()


def _model_parameter_version(model: torch.nn.Module) -> tuple[int, ...]:
    return tuple(parameter._version for parameter in model.parameters() if parameter.requires_grad)


@contextmanager
def _explicit_head_trainer_init_scope(enabled: bool):
    """Let explicit PP/TP layouts install their own selected-token head.

    Pinned TRL has no public disable flag. Scope this dispatch to construction;
    ordinary models retain upstream patching, and the original hook is restored.
    """
    if not enabled:
        yield
        return
    from importlib import import_module

    trl_grpo_trainer = import_module("trl.trainer.grpo_trainer")

    with _TRL_HEAD_INIT_LOCK:
        original = getattr(trl_grpo_trainer, "add_fused_lm_head", None)
        if not callable(original):
            raise TypeError("pinned TRL GRPOTrainer no longer exposes callable add_fused_lm_head")

        def add_fused_lm_head_for_stage(model, *args, **kwargs):
            if (getattr(model, "_qwen35_pipeline_stage", None) is not None
                    or getattr(model, "_t2_tp_model", False)):
                return None
            return original(model, *args, **kwargs)

        trl_grpo_trainer.add_fused_lm_head = add_fused_lm_head_for_stage
        try:
            yield
        finally:
            trl_grpo_trainer.add_fused_lm_head = original



class _T2TensorParallelGRPOConfig(GRPOConfig):
    """The explicit TP loader owns CUDA placement and packed CPU masters."""

    @property
    def place_model_on_device(self):
        return False


class _OptimizerGuard(TrainerCallback):
    """Fail before Trainer's optimizer step when the accumulated gradient is unusable."""

    def __init__(
        self,
        parameters: Sequence[torch.nn.Parameter],
        phase_callback: Callable[[str], None] | None = None,
        *,
        allow_local_zero_gradient: bool = False,
    ):
        self.parameters = list(parameters)
        self.phase_callback = phase_callback
        self.allow_local_zero_gradient = bool(allow_local_zero_gradient)
        self.gradient_norm_before_clip: float | None = None
        self.gradient_norm_after_clip: float | None = None
        self.optimizer_step_seen = False
        self.optimizer_step_count = 0
        self.pre_optimizer_step_seen = False
        self.before_optimizer_step_callback: Callable[[], None] | None = None
        self.before_optimizer_step_called = False

    def begin_update(self, before_optimizer_step: Callable[[], None] | None = None) -> None:
        self.before_optimizer_step_callback = before_optimizer_step
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.gradient_norm_before_clip = None
        self.gradient_norm_after_clip = None
        self.optimizer_step_seen = False
        self.optimizer_step_count = 0
        self.pre_optimizer_step_seen = False
        self.before_optimizer_step_called = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.reset_metrics()
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        pre_clip_norm = self.gradient_norm_before_clip
        if pre_clip_norm is None or not math.isfinite(pre_clip_norm):
            raise FloatingPointError("Trainer produced no finite pre-clip gradient norm; optimizer step skipped")
        if pre_clip_norm <= 0.0:
            raise RuntimeError("DAPO batch produced a zero gradient; optimizer step skipped")
        gradients = [parameter.grad for parameter in self.parameters if parameter.grad is not None]
        if not gradients and not self.allow_local_zero_gradient:
            raise RuntimeError("DAPO batch produced no trainable gradient; optimizer step skipped")
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:  # pragma: no cover - the pinned runtime provides DTensor
            DTensor = ()  # type: ignore[assignment,misc]
        for gradient in gradients:
            local_gradient = gradient.to_local() if isinstance(gradient, DTensor) else gradient
            if not bool(torch.isfinite(local_gradient).all()):
                raise FloatingPointError("non-finite trainable gradient; Trainer optimizer step skipped")
        if gradients and not self.allow_local_zero_gradient:
            norm = torch.linalg.vector_norm(
                torch.stack([gradient.detach().float().norm(2) for gradient in gradients]), ord=2
            )
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite clipped gradient norm; Trainer optimizer step skipped")
            local_norm_value = float(norm.detach().cpu())
        else:
            local_norm_value = 0.0
        if local_norm_value <= 0.0 and not self.allow_local_zero_gradient:
            raise RuntimeError("DAPO batch produced a zero trainable gradient; optimizer step skipped")
        # The upstream clip call supplies the already placement-aware global
        # norm. Once clipped, the global norm is bounded by max_grad_norm even
        # when this rank owns only a zero-gradient shard.
        self.gradient_norm_after_clip = min(pre_clip_norm, float(args.max_grad_norm))
        if self.before_optimizer_step_callback is not None:
            if self.before_optimizer_step_called:
                raise RuntimeError("before_optimizer_step callback was invoked more than once")
            self.before_optimizer_step_callback()
            self.before_optimizer_step_called = True
        if self.phase_callback is not None:
            self.phase_callback("optimizer_step")
        self.pre_optimizer_step_seen = True
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.optimizer_step_seen = True
        self.optimizer_step_count += 1
        if self.phase_callback is not None:
            self.phase_callback("post_optimizer_validation")
        return control


def _external_reward(prompts, completions, reward, **kwargs):
    """Return the trusted ARC service rewards carried by the external rollout."""
    if reward is None:
        raise ValueError("external rollout did not provide rewards")
    return reward


class LearnerGRPOTrainer(GRPOTrainer):
    """Run complete external rollout groups through one persistent TRL trainer.

    `train_groups` is the synchronous update boundary used by LearnerEngine. One
    dataset prompt is staged per group and GRPOTrainer's normal sampler repeats
    each prompt to form a complete group before splitting the batch into microbatches.
    """

    def __init__(
        self,
        model,
        processing_class,
        *,
        group_size: int = 3,
        groups_per_update: int = 2,
        microbatch_size: int = 2,
        max_padded_tokens_per_microbatch: int | None = None,
        learning_rate: float = 1e-6,
        max_grad_norm: float = 1.0,
        max_sequence_length: int = 65536,
        weight_decay: float = 0.0,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        output_dir: str | Path = "/tmp/qwen-rl-trainer",
        seed: int = 42,
        importance_sampling_cap: float = 3.0,
        activation_offload: bool = True,
        activation_offload_min_tokens: int = DEFAULT_OFFLOAD_MIN_TOKENS,
        checkpoint_input_recompute_span: int = 1,
        activation_offload_pin_memory: bool = True,
        activation_offload_max_pinned_bytes: int | None = None,
        activation_cache_parameter_names: Sequence[str] = (),
        activation_cache_cpu_bytes: int = 2 * 1024**3,
        activation_cache_staging_bytes: int = 512 * 1024**2,
        gradient_checkpointing: bool = True,
        gradient_checkpointing_kwargs: Mapping[str, Any] | None = None,
        gradient_only: bool = False,
        constrained_decoding: bool = False,
        deterministic_training: bool = False,
        parallel_mode: str = "single",
        parallel_world_size: int = 1,
        parallel_rank: int = 0,
    ):
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
            raise ValueError("GRPO group_size must be at least 2")
        if (
            isinstance(groups_per_update, bool)
            or not isinstance(groups_per_update, int)
            or groups_per_update < 1
        ):
            raise ValueError("groups_per_update must be a positive integer")
        if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int):
            raise ValueError("microbatch_size must be a positive integer")
        total_samples = group_size * groups_per_update
        if microbatch_size < 1 or microbatch_size > total_samples or total_samples % microbatch_size:
            raise ValueError(
                "microbatch_size must be a positive divisor of "
                f"group_size * groups_per_update ({total_samples})"
            )
        if learning_rate <= 0 or max_grad_norm <= 0 or max_sequence_length < 2:
            raise ValueError("learning_rate, max_grad_norm, and max_sequence_length must be positive")
        if (isinstance(importance_sampling_cap, bool)
                or not isinstance(importance_sampling_cap, (int, float))
                or not math.isfinite(importance_sampling_cap) or importance_sampling_cap < 1.0):
            raise ValueError("importance_sampling_cap must be finite and >= 1")

        if type(activation_offload) is not bool:
            raise ValueError("activation_offload must be a boolean")
        if type(activation_offload_min_tokens) is not int or activation_offload_min_tokens < 0:
            raise ValueError("activation_offload_min_tokens must be an integer >= 0")
        if type(gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be a boolean")
        if type(gradient_only) is not bool:
            raise ValueError("gradient_only must be a boolean")
        if (gradient_checkpointing_kwargs is not None
                and (not isinstance(gradient_checkpointing_kwargs, Mapping)
                     or any(not isinstance(key, str) for key in gradient_checkpointing_kwargs))):
            raise ValueError("gradient_checkpointing_kwargs must be a string-keyed mapping or None")
        if type(activation_offload_pin_memory) is not bool:
            raise ValueError("activation_offload_pin_memory must be a boolean")
        if (activation_offload_max_pinned_bytes is not None
                and (type(activation_offload_max_pinned_bytes) is not int
                     or activation_offload_max_pinned_bytes <= 0)):
            raise ValueError("activation_offload_max_pinned_bytes must be positive when set")
        if type(checkpoint_input_recompute_span) is not int or checkpoint_input_recompute_span != 1:
            raise ValueError("checkpoint_input_recompute_span must be1")
        managed_single = (
            parallel_mode == "single" and gradient_only
            and getattr(model, "_t2_managed_assembly", False) is True
        )
        if type(deterministic_training) is not bool or (deterministic_training and not managed_single):
            raise ValueError("deterministic_training requires the managed single T2 gradient executor")
        self.deterministic_training = deterministic_training
        if managed_single and (
            gradient_checkpointing_kwargs is None
            or gradient_checkpointing_kwargs.get("use_reentrant") is not False
        ):
            raise ValueError("managed T2 assembly requires explicit non-reentrant checkpointing")
        model._checkpoint_input_recompute_span = checkpoint_input_recompute_span
        self.activation_offload = activation_offload
        self.activation_offload_min_tokens = activation_offload_min_tokens
        self.activation_offload_pin_memory = activation_offload_pin_memory
        self.activation_offload_max_pinned_bytes = activation_offload_max_pinned_bytes
        from .activation_cache import resolve_parameters
        resolve_parameters(model, activation_cache_parameter_names)
        if activation_cache_parameter_names and not managed_single:
            raise ValueError("parameter activation cache requires the managed single T2 gradient executor")
        for budget in (activation_cache_cpu_bytes, activation_cache_staging_bytes):
            if type(budget) is not int or budget <= 0:
                raise ValueError("activation cache byte budgets must be positive integers")
        self.activation_cache_parameter_names = tuple(activation_cache_parameter_names)
        self.activation_cache_cpu_bytes = activation_cache_cpu_bytes
        self.activation_cache_staging_bytes = activation_cache_staging_bytes
        self.gradient_checkpointing = gradient_checkpointing
        self.failure_stage = "idle"
        self.telemetry = TrainingTelemetry()
        self._operation_telemetry = None
        self.group_size = int(group_size)
        self.groups_per_update = int(groups_per_update)
        if max_padded_tokens_per_microbatch is not None:
            if (type(max_padded_tokens_per_microbatch) is not int
                    or max_padded_tokens_per_microbatch < 1):
                raise ValueError("max_padded_tokens_per_microbatch must be a positive integer or None")
            if parallel_mode != "single":
                raise ValueError("dynamic microbatches currently require parallel_mode=single")
        self.max_padded_tokens_per_microbatch = max_padded_tokens_per_microbatch
        self._active_microbatch_plan = None
        self._last_microbatch_plan_report: dict[str, Any] = {}
        self.microbatch_size = int(microbatch_size)
        self.total_samples = total_samples
        self.gradient_accumulation_steps = (
            1 if gradient_only else total_samples // self.microbatch_size
        )
        self._active_rows: list[dict[str, Any]] | None = None
        self._fresh_behavior_by_row: dict[int, tuple[float, ...]] = {}
        self._active_group_prompt_keys: list[str] | None = None
        self._active_group_prompt_keys_by_row: list[str] | None = None
        self._active_dataset_prompt_keys: list[str] | None = None
        self._active_rollout_output: dict[str, Any] | None = None
        self._last_train_output = None
        self._last_trainer_metrics: dict[str, float] = {}
        self._last_upstream_clip_metrics: dict[str, float] = {}
        self._last_saved_tensor_offload_report: dict[str, Any] | None = None
        self._offload_microbatch_reports: list[dict[str, Any]] = []
        self._active_training_offload = None
        self._active_offload_decision: dict[str, Any] | None = None
        self._active_offload_entered = False
        self._training_step_active = False
        self._prepared_training_step_input = None
        self._last_actual_gradient_accumulation_steps = self.gradient_accumulation_steps
        self._last_group_cardinalities: dict[Any, int] = {}
        self._last_padding_metrics: dict[str, float] = {}
        self._last_gdn_warmup: dict[str, Any] = {}
        self.importance_sampling_cap = float(importance_sampling_cap)
        if parallel_mode not in {"single", "tp", "ddp", "fsdp2", "pp"}:
            raise ValueError("parallel_mode must be single, tp, ddp, fsdp2 or pp")
        if isinstance(parallel_world_size, bool) or not isinstance(parallel_world_size, int) or parallel_world_size < 1:
            raise ValueError("parallel_world_size must be a positive integer")
        if parallel_mode == "single" and parallel_world_size != 1:
            raise ValueError("single mode requires parallel_world_size=1")
        if parallel_mode in {"tp", "fsdp2"} and parallel_world_size < 2:
            raise ValueError(f"{parallel_mode} mode requires at least two processes")
        if parallel_mode == "pp" and parallel_world_size != 4:
            raise ValueError("pp mode requires exactly four pipeline processes")
        if isinstance(parallel_rank, bool) or not isinstance(parallel_rank, int) or not 0 <= parallel_rank < parallel_world_size:
            raise ValueError("parallel_rank must be inside the selected world")
        self.max_training_sequence_length = int(max_sequence_length)
        self.parallel_mode = parallel_mode
        self.parallel_world_size = parallel_world_size
        self.parallel_rank = parallel_rank
        self.gradient_only = gradient_only
        if type(constrained_decoding) is not bool:
            raise ValueError("constrained_decoding must be boolean")
        if constrained_decoding and parallel_mode != "single":
            raise ValueError("grammar-conditioned training requires a single-device Trainer")
        self.constrained_decoding = constrained_decoding
        if gradient_only and (parallel_mode != "single" or parallel_world_size != 1):
            raise ValueError("gradient-only compute requires a single-process single-device Trainer")
        self._gradient_compute_active = False
        self._gradient_compute_lock = threading.Lock()
        self._deferred_dapo_metric_window: DeferredDapoMetricWindow | None = None
        self._parallel_phase_callback: Callable[..., None] | None = None
        self._progress_callback: Callable[[str, dict[str, Any]], None] | None = None
        self._parallel_global_policy_tokens: int | None = None
        self._parallel_reward_stats: dict[str, float] = {}
        self._proximal_ready = False
        self._inline_proximal: InlineProximalWindow | None = None
        self._inline_proximal_batch: dict[str, Any] | None = None
        self._policy_scoring_target_mask: torch.Tensor | None = None
        self._last_scoring_metrics: dict[str, Any] = {}
        self._last_correction_metrics: dict[str, Any] = {}
        self._last_gradient_diagnostics: dict[str, Any] = {}
        self._last_successful_quality_snapshot: dict[str, Any] = {"available": False}
        self._gradient_guard = _OptimizerGuard(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            phase_callback=lambda stage: setattr(self, "failure_stage", stage),
            allow_local_zero_gradient=parallel_mode in {"tp", "ddp", "fsdp2", "pp"},
        )
        model_device = next(model.parameters()).device

        accelerator_config: dict[str, Any] = {"split_batches": parallel_mode in {"ddp", "fsdp2"}}
        parallel_args: dict[str, Any] = {}
        if parallel_mode in {"tp", "fsdp2"}:
            from accelerate import ParallelismConfig

            if parallel_mode == "tp":
                parallel_args["parallelism_config"] = ParallelismConfig(tp_size=parallel_world_size)
            else:
                parallel_args["parallelism_config"] = ParallelismConfig(dp_shard_size=parallel_world_size)
                parallel_args["fsdp"] = True
                parallel_args["fsdp_config"] = {
                    "version": 2,
                    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"],
                    "reshard_after_forward": True,
                    "cpu_ram_efficient_loading": False,
                }
        # PP ranks keep one disjoint Qwen stage each. Do not configure TP, DDP,
        # or FSDP around these modules; PipelineRankRunner owns stage traffic.

        config_type = (_T2TensorParallelGRPOConfig
                       if getattr(model, "_t2_tp_model", False) else GRPOConfig)
        args = config_type(
            output_dir=str(output_dir),
            # GRPOConfig validates its unused generation geometry before
            # Trainer construction. A gradient-only executor creates one
            # actual row batch per direct compute call, so use one complete
            # group here to keep generation_batch_size divisible by n while
            # the executor continues to split with self.microbatch_size.
            per_device_train_batch_size=self.group_size if self.gradient_only else self.microbatch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            steps_per_generation=self.gradient_accumulation_steps,
            num_generations=self.group_size,
            num_iterations=1,
            disable_dropout=True,
            max_steps=1,
            num_train_epochs=1,
            learning_rate=float(learning_rate),
            lr_scheduler_type="constant",
            warmup_steps=0,
            weight_decay=float(weight_decay),
            adam_beta1=float(adam_betas[0]),
            adam_beta2=float(adam_betas[1]),
            adam_epsilon=float(adam_eps),
            max_grad_norm=float(max_grad_norm),
            beta=0.0,
            loss_type="dapo",
            epsilon=0.2,
            epsilon_high=0.28,
            scale_rewards="group",
            use_vllm=False,
            vllm_importance_sampling_correction=False,
            max_completion_length=int(max_sequence_length),
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": False}
                if getattr(model, "_t2_tp_model", False)
                else dict(gradient_checkpointing_kwargs)
                if gradient_checkpointing_kwargs is not None
                else None
            ),
            remove_unused_columns=False,
            shuffle_dataset=False,
            dataloader_num_workers=0,
            logging_strategy="no",
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            disable_tqdm=True,
            seed=int(seed),
            bf16=model_device.type == "cuda",
            fp16=False,
            average_tokens_across_devices=False,
            accelerator_config=accelerator_config,
            **parallel_args,
        )
        placeholder_dataset = Dataset.from_dict({"prompt": ["qwen-rl-placeholder"]})
        with _explicit_head_trainer_init_scope(
            parallel_mode == "pp" or bool(getattr(model, "_t2_tp_model", False))
        ):
            super().__init__(
                model=model,
                reward_funcs=_external_reward,
                args=args,
                train_dataset=placeholder_dataset,
                processing_class=processing_class,
                rollout_func=self._rollout_group,
                callbacks=[self._gradient_guard],
            )

        if getattr(model, "_t2_tp_model", False):
            # Accelerator.prepare() must not move CPU packed masters back onto
            # the GPU. Inputs and trainable parameters already have explicit
            # rank placement; Trainer's input preparation still sends tokens.
            self.accelerator.device_placement = False
            from .ternary_tp_bridge import install_t2_tp_fused_head_bridge

            self._t2_tp_head_bridge = install_t2_tp_fused_head_bridge(
                model, temperature=float(self.temperature)
            )

        # GRPO computes its own group-global DAPO token denominator. Supplying
        # that real count keeps Transformers from applying a second generic
        # gradient-accumulation division to the already-normalized loss.
        self.model_accepts_loss_kwargs = True
        if self.gradient_only:
            # Pinned TRL's DAPO normalizer also reads this run-loop field. The
            # gradient-only path bypasses `Trainer.train()`, so initialize its
            # current GAS from the construction-time single-step contract.
            self.current_gradient_accumulation_steps = 1
            if self.gradient_checkpointing:
                enable_checkpointing = getattr(self.model, "gradient_checkpointing_enable", None)
                if not callable(enable_checkpointing):
                    raise TypeError("gradient-only Trainer model cannot enable gradient checkpointing")
                enable_checkpointing(
                    gradient_checkpointing_kwargs=self.args.gradient_checkpointing_kwargs
                )
                if not bool(getattr(self.model, "is_gradient_checkpointing", False)):
                    raise RuntimeError("gradient-only Trainer did not enable gradient checkpointing")

        if not self.gradient_only:
            # Build the Trainer-owned optimizer and constant scheduler once.
            # The Learner snapshots both and reuses their state across one-step
            # train() calls, so retain the scheduler instead of letting Trainer
            # reset it.
            self.create_optimizer()
            self.ensure_recovery_scheduler()

    def ensure_recovery_scheduler(self):
        """Create the pinned Trainer scheduler once and keep it across updates."""
        scheduler = self.create_scheduler(num_training_steps=1)
        # Transformers 5.12 resets a scheduler it created at the start of every
        # train() call. Our constant schedule is lineage state, so prevent that
        # per-call reset and allow a recovery snapshot to restore its counters.
        self._created_lr_scheduler = False
        return scheduler

    @property
    def gradient_norm_before_clip(self) -> float | None:
        return self._gradient_guard.gradient_norm_before_clip

    @property
    def failure_stage(self) -> str:
        return getattr(self, "_failure_stage", "idle")

    @failure_stage.setter
    def failure_stage(self, value: str) -> None:
        stage = str(value)
        self._failure_stage = stage
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None or telemetry.active is None:
            return
        total = getattr(self, "_last_actual_gradient_accumulation_steps", None)
        current = None
        normalized = stage
        if stage.startswith("proximal_microbatch_"):
            try:
                current = int(stage.rsplit("_", 1)[1]) + 1
            except ValueError:
                current = None
            normalized = "proximal"
        elif stage == "loss_forward":
            normalized = "forward"
            current = min(self._step + 1, total) if total else None
        elif stage == "loss_backward":
            normalized = "backward"
            current = min(self._step + 1, total) if total else None
        elif stage == "gradient_clip":
            current = total
        elif stage == "optimizer_step":
            current = total
        elif stage in {"optimizer_completed", "post_optimizer_validation"}:
            normalized = "post_optimizer_validation"
            current = total
        telemetry.set_stage(
            normalized,
            current_microbatch=current,
            planned_microbatches=total if isinstance(total, int) and total > 0 else None,
        )
        callback = getattr(self, "_progress_callback", None)
        if callable(callback):
            callback(normalized, telemetry.snapshot())

    @staticmethod
    def _json_scalar_mapping(values: dict[str, Any]) -> dict[str, Any]:
        """Keep bounded JSON scalars and small nested scalar maps only."""
        output: dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, dict):
                nested = LearnerGRPOTrainer._json_scalar_mapping(value)
                if nested:
                    output[str(key)] = nested
            elif isinstance(value, str) or isinstance(value, bool) or isinstance(value, int):
                output[str(key)] = value
            elif isinstance(value, float) and math.isfinite(value):
                output[str(key)] = value
        return output

    def _build_quality_snapshot(self) -> dict[str, Any]:
        metrics = self._json_scalar_mapping(self._last_trainer_metrics)
        quality: dict[str, Any] = {
            "available": False,
            "source": "latest_successful_optimizer_update",
        }
        loss_metrics = {
            key: metrics[key]
            for key in ("train_loss", "loss", "policy_loss")
            if key in metrics
        }
        loss = metrics.get("train_loss", metrics.get("loss"))
        if loss is not None:
            quality["loss"] = loss
        if loss_metrics:
            quality["loss_metrics"] = loss_metrics

        before = self._gradient_guard.gradient_norm_before_clip
        after = self._gradient_guard.gradient_norm_after_clip
        if before is not None and math.isfinite(before):
            quality["gradient_norm_before_clip"] = float(before)
        if after is not None and math.isfinite(after):
            quality["gradient_norm_after_clip"] = float(after)

        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None and optimizer.param_groups:
            learning_rate = optimizer.param_groups[0].get("lr")
            if isinstance(learning_rate, (int, float)) and math.isfinite(float(learning_rate)):
                quality["learning_rate"] = float(learning_rate)

        clip_ratio = {
            key.split("clip_ratio/", 1)[1]: value
            for key, value in metrics.items()
            if "clip_ratio/" in key
        }
        for key, value in self._last_upstream_clip_metrics.items():
            clip_ratio.setdefault(key, value)
        if clip_ratio:
            quality["clip_ratio"] = clip_ratio

        correction = self._json_scalar_mapping(self._last_correction_metrics)
        if correction:
            quality["importance_correction"] = correction
            capped_fraction = correction.get("truncated_fraction")
            fresh_policy_tokens = correction.get("fresh_policy_tokens")
            if isinstance(capped_fraction, (int, float)) and isinstance(fresh_policy_tokens, int):
                quality["importance_capped_fraction"] = float(capped_fraction)
                quality["importance_capped_fraction_denominator_fresh_policy_tokens"] = fresh_policy_tokens

        alignment_keys = {
            "target_scoring_strategy",
            "eligible_completion_targets",
            "policy_targets_projected",
            "non_policy_completion_targets_skipped",
        }
        alignment = {key: correction[key] for key in alignment_keys if key in correction}
        policy_alignment = correction.get("policy_alignment")
        if isinstance(policy_alignment, dict):
            alignment.update(policy_alignment)
        if alignment:
            quality["alignment"] = alignment

        quality["available"] = any(
            key in quality
            for key in ("loss", "gradient_norm_before_clip", "learning_rate", "clip_ratio", "importance_correction")
        )
        return quality

    def quality_snapshot(self) -> dict[str, Any]:
        """Return the latched scalar quality from the latest successful optimizer update."""
        return copy.deepcopy(self._last_successful_quality_snapshot)

    @property
    def optimizer_step_seen(self) -> bool:
        return self._gradient_guard.optimizer_step_seen

    def _clip_grad_norm(self, model):
        self.failure_stage = "gradient_clip"
        unwrapped = self.accelerator.unwrap_model(model) if self.parallel_mode == "tp" else model
        if self.parallel_mode == "tp" and getattr(unwrapped, "_t2_tp_model", False):
            from qwen_rl.models.ternary_tp_state import (
                reduce_t2_tp_gradients_, clip_t2_tp_grad_norm_,
            )

            placements = unwrapped._t2_tp_placements
            self._t2_tp_gradient_reduction = reduce_t2_tp_gradients_(
                unwrapped, placements, rank=self.parallel_rank,
                world_size=self.parallel_world_size,
            )
            clipping = clip_t2_tp_grad_norm_(
                unwrapped, placements, self.args.max_grad_norm,
                rank=self.parallel_rank, world_size=self.parallel_world_size,
            )
            norm = torch.tensor(clipping["gradient_norm_before_clip"],
                                device=self.accelerator.device)
        elif self.parallel_mode == "tp":
            import torch.distributed as dist

            from qwen_rl.models.nvfp4 import nvfp4_tp_parameter_placements

            placement_by_name = nvfp4_tp_parameter_placements(self.accelerator.unwrap_model(model))
            named_trainable = {
                name: parameter
                for name, parameter in self.accelerator.unwrap_model(model).named_parameters()
                if parameter.requires_grad
            }
            if not named_trainable or set(named_trainable) != set(placement_by_name):
                raise RuntimeError("TP LoRA parameter placement metadata is incomplete or stale")
            local_squared_norm = torch.zeros((), dtype=torch.float32, device=self.accelerator.device)
            for name, parameter in named_trainable.items():
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError(f"non-finite TP gradient for {name}")
                role = placement_by_name[name].get("role")
                if role not in {"replicated", "sharded"}:
                    raise RuntimeError(f"unsupported TP LoRA placement for {name}: {role!r}")
                if role == "replicated" and self.parallel_rank != 0:
                    continue
                local_squared_norm += gradient.detach().float().pow(2).sum()
            if not dist.is_initialized() or dist.get_world_size() != self.parallel_world_size:
                raise RuntimeError("TP gradient clipping requires the configured world process group")
            dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)
            total_norm = local_squared_norm.sqrt()
            if not bool(torch.isfinite(total_norm)):
                raise FloatingPointError("non-finite global TP gradient norm")
            parameters = list(named_trainable.values())
            torch.nn.utils.clip_grads_with_norm_(parameters, self.args.max_grad_norm, total_norm)
            norm = total_norm
        else:
            norm = super()._clip_grad_norm(model)
        self._gradient_guard.gradient_norm_before_clip = float(norm.detach().float().cpu())
        return norm

    @staticmethod
    @contextmanager
    def _lora_input_dtype_cast_scope(model):
        """Disable PEFT's activation-to-adapter-weight cast for one model scope."""
        from peft.helpers import disable_input_dtype_casting

        with disable_input_dtype_casting(model):
            yield

    @contextmanager
    def _bf16_lora_input_dtype_scope(self):
        """Keep BF16 activations BF16 through PEFT LoRA projections.

        The packed base and Trainer use BF16 autocast while LoRA parameters
        remain FP32. PEFT normally copies each activation to its FP32 LoRA
        weight dtype before autocast; on long sequences that copy can itself
        exceed the available activation memory. Autocast performs the matmul
        in BF16 without that materialized FP32 input copy.
        """
        parameter = next(self.model.parameters())
        if parameter.device.type != "cuda" or not self.args.bf16:
            yield
            return
        with self._lora_input_dtype_cast_scope(self.model):
            yield

    def score_token_ids(
        self,
        input_ids: torch.Tensor,
        policy_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use TRL's patched fused LM head to score causally shifted policy tokens.

        Returns log probabilities and a Boolean policy mask aligned to targets
        `input_ids[:, 1:]`; masked prompt and environment targets are zero-valued.
        """
        if input_ids.ndim != 2 or policy_mask.shape != input_ids.shape:
            raise ValueError("input_ids and policy_mask must both have shape [batch, sequence]")
        if input_ids.shape[1] < 2:
            raise ValueError("causal token scoring requires at least two input tokens")
        if any(value not in (0, 1, False, True) for value in policy_mask.detach().reshape(-1).tolist()):
            raise ValueError("policy_mask must contain only 0/1 values")
        policy_mask = policy_mask.to(device=input_ids.device, dtype=torch.bool)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        elif attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.long)
        if bool(policy_mask[:, 0].any()):
            raise ValueError("position zero cannot be a policy target under causal shifting")
        labels = input_ids.masked_fill(~policy_mask, -100)
        with self.accelerator.autocast(), self._bf16_lora_input_dtype_scope():
            if getattr(self.model, "_t2_tp_model", False):
                from .ternary_tp_scoring import score_t2_tp_rows

                logps, _ = score_t2_tp_rows(
                    self.model, input_ids, attention_mask, labels,
                    max_tokens=self.max_training_sequence_length,
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    fused_lm_head=True,
                    return_dict=True,
                )
                logps = outputs.log_probs
        shifted_mask = policy_mask[:, 1:]
        if logps.shape != shifted_mask.shape:
            raise RuntimeError(
                f"TRL fused scorer returned {tuple(logps.shape)} for shifted mask {tuple(shifted_mask.shape)}"
            )
        return logps.float(), shifted_mask

    def _get_train_sampler(self, dataset: Dataset | None = None):
        if self._active_rows is None:
            return super()._get_train_sampler(dataset)
        if dataset is None:
            dataset = self.train_dataset
        expected = len(self._active_dataset_prompt_keys or ())
        if len(dataset) != expected:
            raise RuntimeError("staged dataset rows differ from the immutable global training rows")
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=1,
            batch_size=len(dataset),
            repeat_count=self.args.steps_per_generation,
            shuffle=False,
        )

    def get_train_dataloader(self):
        if self._active_rows is None:
            return super().get_train_dataloader()
        if self._active_dataset_prompt_keys is None or len(self.train_dataset) != len(self._active_dataset_prompt_keys):
            raise RuntimeError("staged train dataset lost a global training row")
        # TRL's default generation batch is microbatch_size * steps_per_generation.
        # An odd actual row count needs a shorter final generation batch, so keep
        # each repeated generation batch at exactly the staged row cardinality.
        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=len(self._active_dataset_prompt_keys),
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    @contextmanager
    def _policy_target_mask_scope(self, target_mask: torch.Tensor | None) -> Iterator[None]:
        """Limit fused-head scoring to policy targets for one synchronous call."""
        previous = self._policy_scoring_target_mask
        self._policy_scoring_target_mask = target_mask
        try:
            yield
        finally:
            self._policy_scoring_target_mask = previous

    @staticmethod
    def _completion_policy_mask(inputs: dict[str, Any]) -> torch.Tensor:
        completion_mask = inputs.get("completion_mask")
        if not isinstance(completion_mask, torch.Tensor) or completion_mask.ndim != 2:
            raise ValueError("TRL inputs must include a two-dimensional completion_mask")
        target_mask = completion_mask.bool()
        tool_mask = inputs.get("tool_mask")
        if tool_mask is not None:
            if not isinstance(tool_mask, torch.Tensor) or tool_mask.shape != completion_mask.shape:
                raise ValueError("TRL tool_mask must match completion_mask")
            target_mask = target_mask & tool_mask.bool()
        return target_mask

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        compute_aux_loss=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        num_tiles=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
    ):
        """Use TRL's fused head with env and padding targets omitted for this batch.

        The base model still receives every input token and its original attention
        mask. Only labels used by the fused vocabulary head are narrowed. Outside
        an active external-rollout scope, retain the pinned TRL implementation.
        """
        target_mask = self._policy_scoring_target_mask
        active_rows = self._active_rows
        conditioned = getattr(self,"constrained_decoding",False)
        if active_rows and any(bool(row.get("constraint_segments")) != conditioned for row in active_rows if row.get("row_kind") != "zero_credit"):
            raise ValueError("trajectory constraint provenance differs from the configured policy distribution")
        if active_rows and conditioned:
            if any(value is not None for value in (pixel_values,image_grid_thw,num_images,pixel_attention_mask,
                    spatial_shapes,num_tiles,image_sizes,token_type_ids,mm_token_type_ids,image_position_ids)) or compute_aux_loss:
                raise ValueError("grammar-conditioned scoring supports text-only trajectories")
            labels = input_ids.masked_fill(~attention_mask.bool(), -100).clone()
            labels[:, :-logits_to_keep] = -100
            if target_mask is not None:
                labels[:, -logits_to_keep:] = labels[:, -logits_to_keep:].masked_fill(~target_mask.bool(), -100)
            from .constrained_scoring import score_constrained_rows
            if not hasattr(self, "_grammar_policy_cache"):
                self._grammar_policy_cache = {}
            with self.accelerator.autocast(), self._bf16_lora_input_dtype_scope():
                logps,entropy = score_constrained_rows(model,input_ids,attention_mask,labels,
                    active_rows,self.processing_class,policy_cache=self._grammar_policy_cache)
            return self._capture_proximal_scores((logps[:, -logits_to_keep:],
                entropy[:, -logits_to_keep:] if compute_entropy else None,None))
        if getattr(self.model, "_t2_tp_model", False):
            if any(value is not None for value in (
                pixel_values, image_grid_thw, num_images, pixel_attention_mask,
                spatial_shapes, num_tiles, image_sizes, token_type_ids,
                mm_token_type_ids, image_position_ids,
            )) or compute_aux_loss:
                raise RuntimeError("T2 TP scoring supports text trajectories without auxiliary loss")
            if (type(logits_to_keep) is not int or not 0 < logits_to_keep < input_ids.shape[1]
                    or attention_mask.shape != input_ids.shape):
                raise ValueError("T2 TP completion window or attention mask is invalid")
            labels = input_ids.masked_fill(~attention_mask.bool(), -100).clone()
            labels[:, :-logits_to_keep] = -100
            if target_mask is not None:
                if target_mask.shape != (input_ids.shape[0], logits_to_keep):
                    raise ValueError("T2 TP policy target mask differs from completion window")
                labels[:, -logits_to_keep:] = labels[:, -logits_to_keep:].masked_fill(
                    ~target_mask.to(device=labels.device, dtype=torch.bool), -100
                )
            from .ternary_tp_scoring import score_t2_tp_rows

            with self.accelerator.autocast(), self._bf16_lora_input_dtype_scope():
                logps, entropy = score_t2_tp_rows(
                    model, input_ids, attention_mask, labels,
                    max_tokens=self.max_training_sequence_length,
                )
            return self._capture_proximal_scores((
                logps[:, -logits_to_keep:],
                entropy[:, -logits_to_keep:] if compute_entropy else None,
                None,
            ))
        if target_mask is None:
            result = super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                compute_aux_loss=compute_aux_loss,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_images=num_images,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
                num_tiles=num_tiles,
                image_sizes=image_sizes,
                token_type_ids=token_type_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_position_ids=image_position_ids,
            )
            return self._capture_proximal_scores(result)

        unsupported = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_images": num_images,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
            "num_tiles": num_tiles,
            "image_sizes": image_sizes,
            "token_type_ids": token_type_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "image_position_ids": image_position_ids,
        }
        present = [name for name, value in unsupported.items() if value is not None]
        if present or compute_aux_loss:
            detail = f" ({', '.join(present)})" if present else ""
            raise RuntimeError(
                "policy-target fused scoring supports text-only batches without auxiliary loss" + detail
            )
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or not isinstance(attention_mask, torch.Tensor)
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError("policy-target fused scoring requires matching [batch, sequence] IDs and attention mask")
        if isinstance(logits_to_keep, bool) or not isinstance(logits_to_keep, int):
            raise TypeError("logits_to_keep must be an integer")
        if logits_to_keep <= 0 or logits_to_keep >= input_ids.shape[1]:
            raise ValueError("logits_to_keep must select a nonempty completion after at least one prompt token")

        target_mask = target_mask.to(device=input_ids.device, dtype=torch.bool)
        expected_shape = (input_ids.shape[0], logits_to_keep)
        if tuple(target_mask.shape) != expected_shape:
            raise ValueError(
                f"policy target mask has shape {tuple(target_mask.shape)}, expected {expected_shape}"
            )
        batch_size = input_ids.shape[0] if batch_size is None else batch_size
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("fused scoring batch_size must be a positive integer")

        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.shape[0], batch_size):
            end = min(start + batch_size, input_ids.shape[0])
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]
            labels = input_ids_batch.masked_fill(attention_mask_batch == 0, -100)
            labels[:, :-logits_to_keep] = -100
            completion_labels = labels[:, -logits_to_keep:]
            completion_labels = completion_labels.masked_fill(~target_mask[start:end], -100)
            labels[:, -logits_to_keep:] = completion_labels

            with self.accelerator.autocast():
                outputs = model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    labels=labels,
                    fused_lm_head=True,
                )
            logps = outputs.log_probs
            if not isinstance(logps, torch.Tensor) or logps.ndim != 2 or logps.shape[0] != input_ids_batch.shape[0]:
                raise RuntimeError("TRL fused LM head returned malformed policy log probabilities")
            logps = logps[:, -logits_to_keep:]
            batch_shape = (input_ids_batch.shape[0], logits_to_keep)
            if tuple(logps.shape) != batch_shape:
                raise RuntimeError("TRL fused LM head returned a policy logprob shape that does not align to targets")
            all_logps.append(logps)

            if compute_entropy:
                entropy = outputs.entropy
                if not isinstance(entropy, torch.Tensor) or entropy.shape != outputs.log_probs.shape:
                    raise RuntimeError("TRL fused LM head returned malformed policy entropy")
                entropy = entropy[:, -logits_to_keep:]
                if tuple(entropy.shape) != batch_shape:
                    raise RuntimeError("TRL fused LM head returned policy entropy that does not align to targets")
                all_entropies.append(entropy)

        result = (
            torch.cat(all_logps, dim=0),
            torch.cat(all_entropies, dim=0) if compute_entropy else None,
            None,
        )
        return self._capture_proximal_scores(result)

    def _capture_proximal_scores(self, result):
        if self._inline_proximal_batch is not None:
            if self._inline_proximal is None:
                raise RuntimeError("inline proximal batch has no active window")
            self._inline_proximal.capture(self._inline_proximal_batch, result[0])
        return result

    def _rollout_group(self, prompts: list[Any], trainer):
        if self._active_rows is None or self._active_group_prompt_keys_by_row is None:
            raise RuntimeError("LearnerGRPOTrainer has no staged external rollout group")
        if prompts != self._active_group_prompt_keys_by_row:
            raise ValueError(
                "TRL rollout batch did not preserve the staged row prompts in exact sampler order"
            )

        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        env_masks: list[list[int]] = []
        rewards: list[float] = []
        group_ids: list[Any] = []
        for row in self._active_rows:
            ids = row["input_ids"]
            mask = row["policy_mask"]
            split = mask.index(1)
            prompt_ids.append(ids[:split])
            completion_ids.append(ids[split:])
            # TRL's env_mask follows its tool_mask convention: 1=model token, 0=environment token.
            if row.get("row_kind") == "zero_credit":
                env_masks.append([0] * (len(ids) - split))
            else:
                env_masks.append(mask[split:])
            rewards.append(float(row["reward"]))
            group_ids.append(row["group_id"])

        result = {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            # The upstream rollout API treats `None` as no sampling scores. Mixed
            # fresh/replay batches cannot be represented by a dense b matrix.
            "logprobs": None,
            "env_mask": env_masks,
            "reward": rewards,
            "group_id": group_ids,
        }
        self._active_rollout_output = result
        return result

    def _generate_and_score_completions(self, inputs):
        """Attach stable row identity before TRL shuffles and splits the batch."""
        if self._active_rows is None:
            return super()._generate_and_score_completions(inputs)
        # TRL groups rewards with a fixed reshape by num_generations. The
        # staged batch contains complete groups of different sizes, so ask its
        # mature generation/reward/tensor path for row-level outputs, then
        # restore the same group-normalized advantage formula using group_id.
        mode = "train" if self.model.training else "eval"
        logged_advantages_before = list(self._logs["advantages"])
        metric_lengths = {
            name: len(self._metrics[mode].get(name, []))
            for name in ("frac_reward_zero_std",)
        }
        configured_num_generations = self.num_generations
        self.num_generations = 1
        try:
            result = super()._generate_and_score_completions(inputs)
        finally:
            self.num_generations = configured_num_generations
        if self._active_rows is None:
            if self.model.training:
                raise RuntimeError("TRL generated a training batch without staged BEPA rows")
            return result
        row_count = len(self._active_rows)
        prompt_ids = result.get("prompt_ids")
        if not isinstance(prompt_ids, torch.Tensor) or prompt_ids.shape[0] != row_count:
            raise ValueError("TRL generation output does not match the staged BEPA row count")
        advantages = result.get("advantages")
        if not isinstance(advantages, torch.Tensor) or advantages.ndim != 1 or advantages.shape[0] != row_count:
            raise ValueError("TRL generation output does not carry one row-level advantage per staged row")
        grouped_advantages = torch.empty(row_count, dtype=torch.float32, device=advantages.device)
        if self.parallel_mode in {"ddp", "fsdp2"} or self.gradient_only:
            missing_advantages = [
                index for index, row in enumerate(self._active_rows)
                if "_parallel_advantage" not in row
                or isinstance(row["_parallel_advantage"], bool)
                or not isinstance(row["_parallel_advantage"], (int, float))
                or not math.isfinite(float(row["_parallel_advantage"]))
            ]
            if missing_advantages:
                raise ValueError(
                    "prepared DAPO rows are missing finite advantages at "
                    f"indices {missing_advantages}"
                )
            grouped_advantages = torch.tensor(
                [float(row.get("_parallel_advantage", 0.0)) for row in self._active_rows],
                dtype=torch.float32,
                device=advantages.device,
            )
            group_reward_stds = (
                [] if self.gradient_only
                else [self._parallel_reward_stats["group_reward_std_mean"]]
            )
        else:
            group_indices: dict[Any, list[int]] = {}
            for index, row in enumerate(self._active_rows):
                if row.get("row_kind") == "zero_credit":
                    grouped_advantages[index] = 0.0
                    continue
                group_indices.setdefault(row["group_id"], []).append(index)
            group_reward_stds: list[float] = []
            for group_id, indices in group_indices.items():
                rewards = torch.tensor(
                    [float(self._active_rows[index]["reward"]) for index in indices],
                    dtype=torch.float32,
                    device=advantages.device,
                )
                if rewards.numel() < 2 or not bool(torch.isfinite(rewards).all()):
                    raise ValueError(f"staged group {group_id!r} has invalid rewards for advantage calculation")
                mean = rewards.mean()
                std = rewards.std(correction=1)
                if not bool(torch.isfinite(std)) or float(std) <= 0.0:
                    raise ValueError(f"staged group {group_id!r} has zero or invalid reward variance")
                group_reward_stds.append(float(std.detach().cpu()))
                normalized = (rewards - mean) / (std + 1e-4)
                scales = normalized.new_tensor(
                    rare_success_advantage_scales(self._active_rows, indices)
                )
                grouped_advantages[indices] = torch.where(
                    normalized > 0.0, normalized * scales, normalized,
                )
        result["advantages"] = grouped_advantages
        if self.gradient_only:
            zero_std_fraction = float(self._parallel_reward_stats.get("group_zero_std_fraction", 0.0))
        else:
            zero_std_fraction = self._parallel_reward_stats.get(
                "group_zero_std_fraction",
                sum(math.isclose(value, 0.0, abs_tol=1e-8) for value in group_reward_stds)
                / len(group_reward_stds),
            )
        logged_zero_std = self._metrics[mode]["frac_reward_zero_std"]
        if len(logged_zero_std) > metric_lengths["frac_reward_zero_std"]:
            del logged_zero_std[metric_lengths["frac_reward_zero_std"]:]
        logged_zero_std.append(zero_std_fraction)
        max_log_length = max(len(self._logs["advantages"]), len(logged_advantages_before) + row_count)
        self._logs["advantages"] = deque(
            [*logged_advantages_before, *grouped_advantages.detach().cpu().tolist()],
            maxlen=max_log_length,
        )
        self._last_actual_group_reward_stats = self._parallel_reward_stats or {
            "group_reward_std_min": min(group_reward_stds),
            "group_reward_std_mean": sum(group_reward_stds) / len(group_reward_stds),
            "group_reward_std_max": max(group_reward_stds),
            "group_zero_std_fraction": zero_std_fraction,
        }
        device = prompt_ids.device
        row_ids = torch.arange(row_count, dtype=torch.long, device=device)
        row_kind_ids = []
        for index, row in enumerate(self._active_rows):
            row_kind = row.get("row_kind", "fresh_student")
            if row_kind not in _ROW_KIND_TO_ID:
                raise ValueError(f"staged row {index} has unsupported row_kind={row_kind!r}")
            row_kind_ids.append(_ROW_KIND_TO_ID[row_kind])
        result["bepa_row_ids"] = row_ids
        result["bepa_row_kind_ids"] = torch.tensor(row_kind_ids, dtype=torch.long, device=device)
        return result

    def _trim_microbatch_padding(self, microbatches):
        """Remove only TRL-added prompt-prefix and completion-suffix padding.

        TRL pads the complete generation batch before splitting it into final
        training microbatches. Immutable staged rows and row IDs provide true
        lengths; attention masks prove which columns are padding. Token IDs and
        tool masks are not used to infer validity because real tokens may equal
        the pad ID and environment tokens have tool_mask=0.
        """
        if self._active_rows is None:
            raise RuntimeError("cannot trim generated batches without staged rollout rows")

        metadata_by_batch: list[list[tuple[int, int]]] = []
        seen_row_ids: set[int] = set()
        planned_positions = 0
        effective_positions = 0

        for batch_index, batch in enumerate(microbatches):
            prompt_ids = batch.get("prompt_ids")
            prompt_mask = batch.get("prompt_mask")
            completion_ids = batch.get("completion_ids")
            completion_mask = batch.get("completion_mask")
            row_ids = batch.get("bepa_row_ids")
            row_kind_ids = batch.get("bepa_row_kind_ids")
            if (not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2
                    or not isinstance(prompt_mask, torch.Tensor) or prompt_mask.shape != prompt_ids.shape
                    or not isinstance(completion_ids, torch.Tensor) or completion_ids.ndim != 2
                    or not isinstance(completion_mask, torch.Tensor)
                    or completion_mask.shape != completion_ids.shape):
                raise RuntimeError(f"TRL microbatch {batch_index} lost prompt/completion token masks")
            row_count, prompt_width = prompt_ids.shape
            if completion_ids.shape[0] != row_count:
                raise RuntimeError(f"TRL microbatch {batch_index} prompt/completion row counts differ")
            completion_width = completion_ids.shape[1]
            if (not isinstance(row_ids, torch.Tensor) or row_ids.ndim != 1
                    or not isinstance(row_kind_ids, torch.Tensor) or row_kind_ids.ndim != 1
                    or row_ids.shape[0] != row_count or row_kind_ids.shape[0] != row_count):
                raise RuntimeError(f"TRL microbatch {batch_index} lost BEPA row identity")

            # Transfer the tiny row metadata and attention-mask counts together.
            row_data = torch.stack((
                row_ids.to(dtype=torch.long),
                row_kind_ids.to(dtype=torch.long),
                prompt_mask.to(dtype=torch.long).sum(dim=1),
                completion_mask.to(dtype=torch.long).sum(dim=1),
            ), dim=1).detach().cpu().tolist()
            batch_metadata: list[tuple[int, int]] = []
            prompt_lengths: list[int] = []
            completion_lengths: list[int] = []
            for row_id_value, row_kind_value, prompt_count, completion_count in row_data:
                row_id = int(row_id_value)
                row_kind_id = int(row_kind_value)
                if row_id < 0 or row_id >= len(self._active_rows):
                    raise RuntimeError(f"TRL returned unknown BEPA row_id={row_id} before padding trim")
                if row_id in seen_row_ids:
                    raise RuntimeError(f"TRL duplicated BEPA row_id={row_id} before padding trim")
                seen_row_ids.add(row_id)
                row = self._active_rows[row_id]
                source_mask = row["policy_mask"]
                if not source_mask or 1 not in source_mask:
                    raise RuntimeError(f"staged BEPA row_id={row_id} has no source policy boundary")
                split = source_mask.index(1)
                prompt_length = split
                completion_length = len(row["input_ids"]) - split
                if prompt_length <= 0 or completion_length <= 0:
                    raise RuntimeError(f"staged BEPA row_id={row_id} has an empty prompt or completion")
                if int(prompt_count) != prompt_length or int(completion_count) != completion_length:
                    raise RuntimeError(
                        f"TRL attention-mask lengths disagree with immutable BEPA row_id={row_id} "
                        f"(prompt {prompt_count}/{prompt_length}, completion {completion_count}/{completion_length})"
                    )
                expected_kind = row.get("row_kind", "fresh_student")
                actual_kind = _ID_TO_ROW_KIND.get(row_kind_id)
                if actual_kind != expected_kind:
                    raise RuntimeError(
                        f"TRL BEPA row metadata shifted before padding trim: row_id={row_id} "
                        f"kind={actual_kind!r}, expected {expected_kind!r}"
                    )
                batch_metadata.append((row_id, row_kind_id))
                prompt_lengths.append(prompt_length)
                completion_lengths.append(completion_length)

            expected_prompt = torch.tensor(prompt_lengths, dtype=torch.long, device=prompt_mask.device)
            prompt_positions = torch.arange(prompt_width, device=prompt_mask.device).unsqueeze(0)
            expected_prompt_mask = prompt_positions >= (prompt_width - expected_prompt.unsqueeze(1))
            expected_completion = torch.tensor(
                completion_lengths, dtype=torch.long, device=completion_mask.device
            )
            completion_positions = torch.arange(completion_width, device=completion_mask.device).unsqueeze(0)
            expected_completion_mask = completion_positions < expected_completion.unsqueeze(1)
            mask_checks = torch.stack((
                prompt_mask.eq(expected_prompt_mask).all(),
                completion_mask.eq(expected_completion_mask).all(),
            ))
            if not bool(mask_checks.all().item()):
                raise RuntimeError(
                    f"TRL microbatch {batch_index} attention masks are not left/right padding of the staged rows"
                )

            # Fail closed if upstream adds a token-aligned tensor without a
            # trimming rule. Image payload tensors and row metadata are not token
            # sequences and are excluded from this shape check.
            aligned_widths = {prompt_width, completion_width, prompt_width + completion_width}
            for key, value in batch.items():
                if (key in _PROMPT_ALIGNED_FIELDS or key in _COMPLETION_ALIGNED_FIELDS
                        or key in _FULL_SEQUENCE_ALIGNED_FIELDS or key in _NON_TOKEN_FIELDS
                        or key in {"advantages", "bepa_row_ids", "bepa_row_kind_ids"}
                        or not isinstance(value, torch.Tensor) or value.ndim < 2
                        or value.shape[0] != row_count):
                    continue
                if any(width in aligned_widths for width in value.shape[1:]):
                    raise RuntimeError(
                        f"unsupported token-aligned TRL field {key!r}; add an explicit padding-trim rule"
                    )

            prompt_start = prompt_width - max(prompt_lengths)
            completion_end = max(completion_lengths)
            full_width = prompt_width + completion_width
            full_end = prompt_width + completion_end
            for key, value in list(batch.items()):
                if value is None or not isinstance(value, torch.Tensor):
                    continue
                if key in _PROMPT_ALIGNED_FIELDS:
                    if value.ndim != 2 or value.shape != (row_count, prompt_width):
                        raise RuntimeError(f"unsupported prompt-aligned TRL field shape for {key!r}")
                    batch[key] = value[:, prompt_start:]
                elif key in _COMPLETION_ALIGNED_FIELDS:
                    if value.ndim != 2 or value.shape != (row_count, completion_width):
                        raise RuntimeError(f"unsupported completion-aligned TRL field shape for {key!r}")
                    batch[key] = value[:, :completion_end]
                elif key == "advantages" and value.ndim == 2:
                    if value.shape != (row_count, completion_width):
                        raise RuntimeError("unsupported token-level advantages shape during padding trim")
                    batch[key] = value[:, :completion_end]
                elif key in _FULL_SEQUENCE_ALIGNED_FIELDS:
                    if value.shape[-1] != full_width or not (
                        value.shape[0] == row_count
                        or (value.ndim >= 3 and value.shape[-2] == row_count)
                    ):
                        raise RuntimeError(f"unsupported full-sequence TRL field shape for {key!r}")
                    batch[key] = value[..., prompt_start:full_end]

            planned_positions += row_count * full_width
            effective_positions += row_count * (max(prompt_lengths) + completion_end)
            metadata_by_batch.append(batch_metadata)

        if seen_row_ids != set(range(len(self._active_rows))):
            missing = sorted(set(range(len(self._active_rows)) - seen_row_ids))
            raise RuntimeError(f"TRL omitted BEPA rows before padding trim: {missing}")

        removed_positions = planned_positions - effective_positions
        self._last_padding_metrics = {
            "planned_input_token_positions": float(planned_positions),
            "effective_input_token_positions": float(effective_positions),
            "padding_token_positions_removed": float(removed_positions),
            "padding_token_position_reduction_fraction": (
                float(removed_positions / planned_positions) if planned_positions else 0.0
            ),
        }
        return metadata_by_batch

    @staticmethod
    def _split_generation_batch(generation_batch, num_chunks: int):
        """Split row-aligned TRL tensors evenly while retaining a short tail."""
        first = next(
            (value for value in generation_batch.values()
             if isinstance(value, torch.Tensor) and value.ndim > 0),
            None,
        )
        if first is None:
            first = next((value for value in generation_batch.values() if isinstance(value, list)), None)
        if first is None:
            raise RuntimeError("TRL generation output has no row-aligned field")
        row_count = int(first.shape[0]) if isinstance(first, torch.Tensor) else len(first)
        if num_chunks < 1 or row_count < num_chunks:
            raise RuntimeError(
                f"cannot split {row_count} actual rollout rows across {num_chunks} microbatches"
            )
        quotient, remainder = divmod(row_count, num_chunks)
        sizes = [quotient + (index < remainder) for index in range(num_chunks)]
        chunks = []
        start = 0
        for size in sizes:
            end = start + size
            chunks.append({
                key: (
                    value[start:end]
                    if isinstance(value, list)
                    or (isinstance(value, torch.Tensor) and value.ndim > 0)
                    else value
                )
                for key, value in generation_batch.items()
            })
            start = end
        if start != row_count or sum(sizes) != row_count:
            raise RuntimeError("variable microbatch split did not preserve every staged row")
        return chunks

    def _split_planned_generation_batch(self, generated):
        """Gather every aligned field using the immutable CPU plan, before proximal scoring."""
        plan = self._active_microbatch_plan
        row_ids = generated.get("bepa_row_ids")
        if plan is None or not isinstance(row_ids, torch.Tensor) or row_ids.ndim != 1:
            raise RuntimeError("dynamic microbatches lost their plan or row identities")
        ids = row_ids.detach().cpu().tolist()
        expected = len(self._active_rows)
        if sorted(ids) != list(range(expected)):
            raise RuntimeError("dynamic generation row identities differ from the staged rows")
        positions = {row_id: index for index, row_id in enumerate(ids)}
        batches = []
        for rows in plan.batches:
            order = [positions[row_id] for row_id in rows]
            batch = {}
            for key, value in generated.items():
                if value is None or (isinstance(value, torch.Tensor) and value.ndim == 0):
                    batch[key] = value
                elif isinstance(value, torch.Tensor):
                    if value.shape[0] != expected:
                        raise RuntimeError(f"dynamic sequence field {key!r} is not row-aligned")
                    indices = torch.tensor(order, dtype=torch.long, device=value.device)
                    batch[key] = value.index_select(0, indices)
                elif isinstance(value, (list, tuple)) and len(value) == expected:
                    batch[key] = [value[index] for index in order]
                else:
                    raise RuntimeError(f"dynamic sequence field {key!r} is not row-aligned")
            batches.append(batch)
        if len(batches) != self.args.steps_per_generation:
            raise RuntimeError("dynamic plan differs from the scheduled accumulation count")
        return batches

    @staticmethod
    def _apply_shared_tp_permutation(sequence_dict, permutation):
        """Apply one leader-provided row order to every TRL sequence field."""
        first = next((value for value in sequence_dict.values() if value is not None), None)
        if first is None:
            raise RuntimeError("TRL sequence batch has no row-aligned fields")
        row_count = len(first)
        order = permutation.detach().to(device="cpu", dtype=torch.long).tolist()
        if len(order) != row_count or sorted(order) != list(range(row_count)):
            raise RuntimeError("TP row permutation does not match the generated batch")
        shuffled = {}
        for key, value in sequence_dict.items():
            if value is None:
                shuffled[key] = None
            elif isinstance(value, torch.Tensor) and value.ndim == 0:
                shuffled[key] = value
            elif isinstance(value, torch.Tensor):
                if value.shape[0] != row_count:
                    raise RuntimeError(f"TRL TP sequence field {key!r} is not row-aligned")
                index = permutation.to(device=value.device, dtype=torch.long)
                shuffled[key] = value.index_select(0, index)
            else:
                if len(value) != row_count:
                    raise RuntimeError(f"TRL TP sequence field {key!r} is not row-aligned")
                shuffled[key] = [value[index] for index in order]
        return shuffled

    def _shared_tp_permutation(self, generated) -> torch.Tensor:
        """Create a row permutation on TP rank zero and broadcast it to every rank."""
        if self.parallel_mode != "tp":
            raise RuntimeError("shared TP permutation is only available in TP mode")
        if not torch.distributed.is_initialized():
            raise RuntimeError("TP row shuffling requires the initialized process group")
        row_ids = generated.get("bepa_row_ids") if isinstance(generated, dict) else None
        local_count = int(row_ids.numel()) if isinstance(row_ids, torch.Tensor) and row_ids.ndim == 1 else -1
        device = torch.device("cuda", torch.cuda.current_device())
        minimum = torch.tensor([local_count], dtype=torch.long, device=device)
        maximum = minimum.clone()
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        expected_rows = len(self._active_rows or ())
        if int(minimum.item()) != int(maximum.item()) or int(minimum.item()) != expected_rows:
            raise RuntimeError(
                "TP ranks generated different staged row counts: "
                f"min={int(minimum.item())}, max={int(maximum.item())}, expected={expected_rows}"
            )

        permutation = torch.empty(expected_rows, dtype=torch.long, device=device)
        if self.parallel_rank == 0:
            permutation.copy_(torch.randperm(expected_rows).to(device))
        torch.distributed.broadcast(permutation, src=0)
        return permutation

    def _report_parallel_phase(self, phase: str, **values: Any) -> None:
        callback = self._parallel_phase_callback
        if self.parallel_mode != "single" and callable(callback):
            callback(phase, **values)

    @staticmethod
    def _requires_staged_generation_path(parallel_mode: str, variable_tail: bool) -> bool:
        return variable_tail or parallel_mode == "tp"

    def _begin_activation_offload(self, inputs: Mapping[str, Any]) -> None:
        """Open the saved-tensor scope after TRL has prepared the final batch."""
        if not self.activation_offload:
            return
        if self._active_training_offload is not None:
            raise RuntimeError("nested offloaded training steps are unsupported")
        prompt_ids = inputs.get("prompt_ids")
        completion_ids = inputs.get("completion_ids")
        if (not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2
                or not isinstance(completion_ids, torch.Tensor) or completion_ids.ndim != 2
                or prompt_ids.shape[0] != completion_ids.shape[0]):
            raise RuntimeError("activation offload requires final 2D prompt_ids and completion_ids tensors")
        decision = offload_decision(
            batch_size=int(prompt_ids.shape[0]),
            sequence_length=int(prompt_ids.shape[1] + completion_ids.shape[1]),
            min_tokens=self.activation_offload_min_tokens,
        )
        checkpoint_policy = {
            "configured_span": 1,
            "selected_span": 1,
            "padded_tokens": decision["padded_tokens"],
            "reason": "span1_only",
        }
        decision["checkpoint_input_policy"] = checkpoint_policy
        self._active_offload_decision = decision
        self._active_offload_entered = False
        offload = (
            saved_tensor_offload_context(
                self.model,
                parallel_mode=self.parallel_mode,
                pin_memory=self.activation_offload_pin_memory and next(self.model.parameters()).is_cuda,
                max_pinned_bytes=self.activation_offload_max_pinned_bytes,
                checkpoint_input_recompute_span=checkpoint_policy["selected_span"],
            )
            if decision["enabled"] else nullcontext()
        )
        offload.__enter__()
        self._active_training_offload = offload
        self._active_offload_entered = bool(decision["enabled"])

    def _prepare_inputs(self, generation_batch):
        if self._prepared_training_step_input is generation_batch:
            self._prepared_training_step_input = None
            return generation_batch
        variable_tail = (
            self.model.training
            and self._active_rows is not None
            and len(self._active_rows) % self.args.steps_per_generation != 0
        )
        dynamic = self.model.training and self.max_padded_tokens_per_microbatch is not None
        if not dynamic and not self._requires_staged_generation_path(self.parallel_mode, variable_tail):
            inputs = super()._prepare_inputs(generation_batch)
        else:
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                if isinstance(generation_batch, dict):
                    prompts = generation_batch.get("prompt")
                    prompt_count = len(prompts) if isinstance(prompts, list) else None
                elif isinstance(generation_batch, (list, tuple)):
                    prompt_count = len(generation_batch)
                else:
                    prompt_count = None
                self._report_parallel_phase(
                    "rollout_score_started",
                    rank=self.parallel_rank,
                    step=self._step,
                    generation_rows=prompt_count,
                    active_rows=len(self._active_rows),
                )
                generated = self._generate_and_score_completions(generation_batch)
                row_ids = generated.get("bepa_row_ids")
                self._report_parallel_phase(
                    "rollout_score_completed",
                    rank=self.parallel_rank,
                    generation_rows=(int(row_ids.numel()) if isinstance(row_ids, torch.Tensor) else None),
                )
                generated = split_pixel_values_by_grid(generated)
                if dynamic:
                    # CPU planning already fixed row order; no second shuffle may change its layout.
                    pass
                elif self.parallel_mode == "tp":
                    permutation = self._shared_tp_permutation(generated)
                    generated = self._apply_shared_tp_permutation(generated, permutation)
                else:
                    generated = shuffle_sequence_dict(generated)
                row_ids = generated.get("bepa_row_ids")
                self._report_parallel_phase(
                    "rollout_shuffled",
                    rank=self.parallel_rank,
                    row_ids=(row_ids.detach().cpu().tolist() if isinstance(row_ids, torch.Tensor) else None),
                    prompt_shape=(list(generated["prompt_ids"].shape) if isinstance(generated.get("prompt_ids"), torch.Tensor) else None),
                    completion_shape=(list(generated["completion_ids"].shape) if isinstance(generated.get("completion_ids"), torch.Tensor) else None),
                )
                self._buffered_inputs = (
                    self._split_planned_generation_batch(generated) if dynamic
                    else self._split_generation_batch(generated, self.args.steps_per_generation)
                )
                self._buffered_inputs = [
                    unsplit_pixel_values_by_grid(batch) for batch in self._buffered_inputs
                ]
                for batch_index, batch in enumerate(self._buffered_inputs):
                    batch_row_ids = batch.get("bepa_row_ids")
                    self._report_parallel_phase(
                        "microbatch_layout",
                        rank=self.parallel_rank,
                        microbatch=batch_index,
                        row_ids=(batch_row_ids.detach().cpu().tolist() if isinstance(batch_row_ids, torch.Tensor) else None),
                        prompt_shape=(list(batch["prompt_ids"].shape) if isinstance(batch.get("prompt_ids"), torch.Tensor) else None),
                        completion_shape=(list(batch["completion_ids"].shape) if isinstance(batch.get("completion_ids"), torch.Tensor) else None),
                    )
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        if not self.model.training:
            return inputs
        if not self._proximal_ready:
            # Upstream fixed the final shuffled/split layout. Single and DDP
            # capture p from each first training forward at unchanged parameters.
            # Model-sharded modes retain their separate collective scoring path.
            if not self._buffered_inputs:
                raise RuntimeError("TRL lost the complete generation batch before proximal scoring")
            row_metadata = self._trim_microbatch_padding(self._buffered_inputs)
            if dynamic:
                for index, batch in enumerate(self._buffered_inputs):
                    actual_cost = batch["prompt_ids"].numel() + batch["completion_ids"].numel()
                    if actual_cost != self._active_microbatch_plan.padded_tokens[index]:
                        raise RuntimeError("actual dynamic microbatch padding differs from its immutable plan")
                    if not bool(batch["prompt_mask"].bool().all()):
                        raise RuntimeError("dynamic microbatch introduced leading prompt padding")
            if getattr(self.model, "_t2_tp_model", False):
                self._last_gdn_warmup = {"status": "skipped", "reason": "explicit_tp_no_warmup"}
            else:
                self.failure_stage = "gdn_l2norm_warmup"
                self._report_parallel_phase(
                    "gdn_warmup_started",
                    rank=self.parallel_rank,
                    microbatches=len(self._buffered_inputs),
                    row_ids=[row_id for batch_rows in row_metadata for row_id, _ in batch_rows],
                )
                try:
                    self._last_gdn_warmup = warmup_gdn_l2norm_backward(
                        self.model,
                        self._buffered_inputs,
                        compute_dtype=torch.bfloat16 if self.args.bf16 else None,
                    )
                except GDNWarmupError as exc:
                    self._last_gdn_warmup = {
                        **exc.metadata,
                        "status": "failed",
                        "error": str(exc),
                    }
                    raise
                self._report_parallel_phase("gdn_warmup_completed", rank=self.parallel_rank)
            self._report_parallel_phase(
                "proximal_scoring_started", rank=self.parallel_rank,
                strategy=("shared_training_forward" if self._uses_inline_proximal() else "separate_forward"),
            )
            if self._uses_inline_proximal():
                self._begin_inline_proximal(self._buffered_inputs, row_metadata_by_batch=row_metadata)
            else:
                self._freeze_proximal(self._buffered_inputs, row_metadata_by_batch=row_metadata)
                self._report_parallel_phase("proximal_scoring_completed", rank=self.parallel_rank)
            self._proximal_ready = True
        if self._training_step_active and self.activation_offload:
            self._begin_activation_offload(inputs)
        return inputs

    @contextmanager
    def _training_microbatch_scope(self, inputs):
        """Share one-microbatch offload ownership between Trainer and gradient jobs."""
        if self._active_training_offload is not None or self._training_step_active:
            raise RuntimeError("nested offloaded training steps are unsupported")
        self._training_step_active = True
        self._active_offload_decision = None
        self._active_offload_entered = False
        self._prepared_training_step_input = None
        succeeded = False
        activation_cache = None
        original_backward = None
        previous_determinism = None
        try:
            if getattr(self, "deterministic_training", False):
                # Efficient SDPA needs strict mode for reproducible split-key
                # gradients. Compiled forward/backward must see the SAME flag.
                previous_determinism = (torch.are_deterministic_algorithms_enabled(),
                                        torch.is_deterministic_algorithms_warn_only_enabled())
                torch.use_deterministic_algorithms(True, warn_only=False)
            # Some callers already pass TRL's final batch. Preserve its one-pass
            # preparation contract by bypassing _prepare_inputs exactly once.
            if (self.activation_offload and isinstance(inputs, Mapping)
                    and isinstance(inputs.get("prompt_ids"), torch.Tensor)
                    and isinstance(inputs.get("completion_ids"), torch.Tensor)):
                self._begin_activation_offload(inputs)
                self._prepared_training_step_input = inputs
            if getattr(self, "activation_cache_parameter_names", ()):
                from .activation_cache import ParameterActivationCache
                policy = (self._active_offload_decision or {}).get("checkpoint_input_policy", {})
                if policy.get("selected_span") != 1:
                    raise ValueError("parameter activation cache requires checkpoint span1 for this microbatch")
                activation_cache = ParameterActivationCache(
                    self.model, self.activation_cache_parameter_names,
                    cpu_bytes=self.activation_cache_cpu_bytes,
                    staging_bytes=self.activation_cache_staging_bytes,
                )
                activation_cache.__enter__()
                original_backward = self.accelerator.backward
                def cached_backward(*args, **kwargs):
                    with activation_cache.backward():
                        return original_backward(*args, **kwargs)
                self.accelerator.backward = cached_backward
            # Gradient-only jobs bypass Trainer.train(), so own this scope for
            # the complete microbatch, including checkpoint replay in backward.
            with self._bf16_lora_input_dtype_scope():
                yield
            succeeded = True
        finally:
            if previous_determinism is not None:
                torch.use_deterministic_algorithms(previous_determinism[0],
                                                   warn_only=previous_determinism[1])
            decision = copy.deepcopy(self._active_offload_decision)
            offload_entered = self._active_offload_entered
            offload = self._active_training_offload
            try:
                try:
                    if original_backward is not None:
                        self.accelerator.backward = original_backward
                    if activation_cache is not None:
                        activation_cache.close()
                finally:
                    if offload is not None:
                        offload.__exit__(*sys.exc_info())
            except BaseException:
                succeeded = False
                raise
            finally:
                self._active_training_offload = None
                self._active_offload_decision = None
                self._active_offload_entered = False
                self._training_step_active = False
                self._prepared_training_step_input = None
                weight_offloader = getattr(self.model, "_t2_tp_weight_offloader", None)
                try:
                    if weight_offloader is not None:
                        weight_offloader.set_phase("idle")
                except BaseException:
                    succeeded = False
                    raise
                finally:
                    if decision is not None:
                        report = copy.deepcopy(getattr(offload, "report", None) or {})
                        if activation_cache is not None:
                            report["parameter_activation_cache"] = activation_cache.report
                        report["decision"] = {
                            **decision,
                            "threshold": self.activation_offload_min_tokens,
                        }
                        report["offload_applied"] = bool(offload_entered)
                        report["status"] = "completed" if succeeded else "failed"
                        policy = getattr(offload, "bounded_offload", None)
                        if policy is not None:
                            report["bounded_offload"] = copy.deepcopy(policy.report)
                        report["microbatch"] = len(self._offload_microbatch_reports)
                        self._offload_microbatch_reports.append(report)
                        self._last_saved_tensor_offload_report = {
                            "scope": "training_step",
                            "status": ("completed" if all(item["status"] == "completed"
                                                          for item in self._offload_microbatch_reports) else "failed"),
                            "microbatch_count": len(self._offload_microbatch_reports),
                            "microbatches": copy.deepcopy(self._offload_microbatch_reports),
                        }

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Own activation copies for one complete F/B, closing before optimizer."""
        with self._training_microbatch_scope(inputs):
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.failure_stage = "loss_forward"
        row_ids = inputs.get("bepa_row_ids")
        self._report_parallel_phase(
            "loss_forward_started",
            rank=self.parallel_rank,
            row_ids=(row_ids.detach().cpu().tolist() if isinstance(row_ids, torch.Tensor) else None),
            input_shape=(list(inputs["prompt_ids"].shape) if isinstance(inputs.get("prompt_ids"), torch.Tensor) else None),
            completion_shape=(list(inputs["completion_ids"].shape) if isinstance(inputs.get("completion_ids"), torch.Tensor) else None),
        )
        if self.model.training and self._active_rows is not None:
            target_mask = self._completion_policy_mask(inputs)
            if self._inline_proximal_batch is not None:
                raise RuntimeError("nested inline proximal loss calls are unsupported")
            self._inline_proximal_batch = inputs if self._inline_proximal is not None else None
            try:
                with self._policy_target_mask_scope(target_mask):
                    if self._deferred_dapo_metric_window is not None and not return_outputs:
                        result = self._compute_loss_with_deferred_dapo_metrics(
                            model, inputs, num_items_in_batch=num_items_in_batch
                        )
                    else:
                        result = super().compute_loss(
                            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
                        )
            finally:
                self._inline_proximal_batch = None
        else:
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        self._report_parallel_phase("loss_forward_completed", rank=self.parallel_rank)
        if self.model.training and 'monitor_input_tokens' in inputs:
            self.telemetry.record_forward(inputs['monitor_input_tokens'], inputs['monitor_policy_tokens'])
        self.failure_stage = "loss_backward"
        return result

    def _deferred_dapo_metrics_are_supported(self) -> bool:
        """Use the window collector only for the exact pinned DDP metric path."""
        return (
            self.parallel_mode == "ddp"
            and self.parallel_world_size > 1
            and self.accelerator.num_processes == self.parallel_world_size
            and self.loss_type == "dapo"
            and self.beta == 0.0
            and self.top_entropy_quantile == 1.0
            and not self._entropy_bonus_enabled
            and not self.aux_loss_enabled
        )

    def _compute_loss_with_deferred_dapo_metrics(self, model, inputs, *, num_items_in_batch=None):
        """Defer only TRL's known diagnostic collectives; leave its loss untouched."""
        window = self._deferred_dapo_metric_window
        if window is None:
            raise RuntimeError("deferred DAPO metric window is not active")
        metric_keys = (
            "entropy",
            "clip_ratio/low_mean",
            "clip_ratio/high_mean",
            "clip_ratio/region_mean",
            "clip_ratio/low_min",
            "clip_ratio/high_max",
        )
        train_metrics = self._metrics["train"]
        metric_lengths = {key: len(train_metrics.get(key, ())) for key in metric_keys}
        with capture_trl_dapo_metric_collectives(
            self.accelerator,
            rows_per_microbatch=self.microbatch_size,
        ) as captured:
            result = super().compute_loss(
                model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
            )
        captured.validate()
        for key, previous_length in metric_lengths.items():
            values = train_metrics.get(key)
            if values is None or len(values) != previous_length + 1:
                raise RuntimeError(
                    f"pinned TRL DAPO metric contract changed: expected one {key!r} value per microbatch"
                )
        window.add_microbatch(captured.reduce_inputs, captured.gather_inputs)
        for key, previous_length in metric_lengths.items():
            del train_metrics[key][previous_length:]
        return result

    def _flush_deferred_dapo_metrics(self) -> None:
        window = self._deferred_dapo_metric_window
        if window is None:
            return
        packed = window.pack()
        gathered = self.accelerator.gather(packed)
        metrics = window.decode(gathered, world_size=self.parallel_world_size)
        for key, values in metrics.items():
            self._metrics["train"][key].extend(values)
        self._deferred_dapo_metric_window = None

    def _uses_inline_proximal(self) -> bool:
        """Data-parallel replicas capture local p before the shared optimizer step."""
        return self.parallel_mode in {"single", "ddp"}

    def _begin_inline_proximal(self, microbatches, *, row_metadata_by_batch=None):
        if self._inline_proximal is not None:
            raise RuntimeError("inline proximal window is already active")
        self._inline_proximal = InlineProximalWindow(
            self, microbatches, row_metadata_by_batch,
            row_kind_to_id=_ROW_KIND_TO_ID, id_to_row_kind=_ID_TO_ROW_KIND,
        )

    def _assert_inline_proximal_complete(self):
        if self._uses_inline_proximal():
            if self._inline_proximal is None:
                raise RuntimeError("optimizer step blocked: inline proximal window is missing")
            self._inline_proximal.assert_complete()

    @torch.no_grad()
    def _freeze_proximal(self, microbatches, *, row_metadata_by_batch=None):
        """Model-sharded scoring uses the same validation/correction accumulator."""
        if row_metadata_by_batch is not None and len(row_metadata_by_batch) != len(microbatches):
            raise RuntimeError("TRL proximal row metadata count differs from the final microbatch count")
        collector = ProximalAccumulator(
            self, row_kind_to_id=_ROW_KIND_TO_ID, id_to_row_kind=_ID_TO_ROW_KIND,
        )
        for index, batch in enumerate(microbatches):
            self.failure_stage = f"proximal_microbatch_{index}"
            mask = self._completion_policy_mask(batch)
            input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
            attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
            with self._policy_target_mask_scope(mask):
                proximal, _, _ = self._get_per_token_logps_and_entropies(
                    self.model, input_ids, attention_mask, batch["completion_ids"].size(1),
                    compute_entropy=True, compute_aux_loss=False,
                )
            collector.add(
                batch, proximal,
                row_metadata=None if row_metadata_by_batch is None else row_metadata_by_batch[index],
            )
        collector.finish()

    def _get_num_items_in_batch(self, batch_samples, device):
        if self._active_rows is None:
            return super()._get_num_items_in_batch(batch_samples, device)
        token_count = self._parallel_global_policy_tokens
        if token_count is None:
            token_count = sum(
                sum(int(flag) for flag in row["policy_mask"][1:])
                for row in self._active_rows
                if row.get("row_kind") != "zero_credit"
            )
        if token_count <= 0:
            raise ValueError("staged rollout group has no causal policy tokens")
        # TRL divides the supplied DAPO token count by the process count. In TP,
        # every rank evaluates the same full batch, so counteract that divisor.
        if self.parallel_mode == "tp":
            token_count *= self.parallel_world_size
        return torch.tensor(token_count, dtype=torch.long, device=device)

    def _expand_completion_log_buffers(self, required_rows: int) -> None:
        """Keep TRL's bounded completion logs large enough for this real batch."""
        for name in ("images", "prompt", "completion", "advantages"):
            current = self._logs[name]
            capacity = max(required_rows, current.maxlen or 0)
            self._logs[name] = deque(current, maxlen=capacity)
        for name in ("rewards", "extra"):
            current = self._logs[name]
            factory_capacity = current.default_factory().maxlen or 0
            capacity = max(required_rows, factory_capacity)
            expanded = defaultdict(lambda: deque(maxlen=capacity))
            for key, values in current.items():
                expanded[key] = deque(values, maxlen=capacity)
            self._logs[name] = expanded

    def prepare_gradient_rows(self, trajectories: Sequence[Mapping[str, Any]]) -> PreparedGradientRows:
        """Use the model-free full-window validator before dividing groups into jobs."""
        return prepare_complete_gradient_rows(
            trajectories,
            group_size=self.group_size,
            groups_per_update=self.groups_per_update,
            microbatch_size=self.microbatch_size,
        )

    def compute_gradient_contribution(
        self,
        prepared_window,
        job,
        trajectories: Sequence[Mapping[str, Any]],
        *,
        installed_policy_version: int,
        compute_profile: Mapping[str, Any],
    ):
        """Backpropagate one complete task-group job without entering Trainer.train()."""
        from qwen_rl.contracts.gradients import (
            GRADIENT_CONTRIBUTION_CONTRACT,
            GRADIENT_CONTRIBUTION_SCHEMA,
            GradientContribution,
            GradientJob,
            PreparedWindow,
            compute_profile_fingerprint,
        )

        if not self.gradient_only:
            raise RuntimeError("compute_gradient_contribution requires gradient_only=True")
        if not isinstance(prepared_window, PreparedWindow) or not isinstance(job, GradientJob):
            raise TypeError("gradient compute requires the typed PreparedWindow and GradientJob contracts")
        if job not in prepared_window.jobs:
            raise ValueError("gradient job is not a member of the prepared window")
        if (job.window_id != prepared_window.window_id
                or job.attempt_id != prepared_window.attempt_id
                or job.policy_version != prepared_window.policy_version
                or job.n_window_tokens != prepared_window.n_window_tokens
                or job.layout_fingerprint != prepared_window.layout_fingerprint):
            raise ValueError("gradient job identity differs from its prepared window")
        if (type(installed_policy_version) is not int
                or installed_policy_version != job.policy_version):
            raise ValueError("installed adapter policy version differs from the gradient job")
        if (not isinstance(compute_profile, Mapping)
                or dict(compute_profile) != dict(prepared_window.compute_profile)
                or dict(compute_profile) != dict(job.compute_profile)
                or job.compute_profile_fingerprint != compute_profile_fingerprint(compute_profile)):
            raise ValueError("gradient job compute profile differs from the prepared window")
        if (compute_profile.get("loss_profile") != "dapo_global_token_mean_beta0_v1"
                or float(compute_profile.get("correction_cap", float("nan"))) != self.importance_sampling_cap):
            raise ValueError("gradient job loss/correction profile differs from this Trainer")
        if (self.parallel_mode != "single" or self.parallel_world_size != 1
                or int(self.accelerator.num_processes) != 1
                or int(self.args.gradient_accumulation_steps) != 1
                or int(self.args.steps_per_generation) != 1
                or int(self.accelerator.gradient_accumulation_steps) != 1
                or int(getattr(self, "current_gradient_accumulation_steps", 0)) != 1):
            raise RuntimeError(
                "gradient compute requires single-process Accelerator GAS=1 and TRL steps_per_generation=1"
            )
        if not self._gradient_compute_lock.acquire(blocking=False):
            raise RuntimeError("a gradient job is already active on this Trainer")
        self._gradient_compute_active = True
        was_training = self.model.training

        try:
            return self._compute_gradient_contribution_locked(
                prepared_window, job, trajectories,
                installed_policy_version=installed_policy_version,
                compute_profile=dict(compute_profile),
                contribution_schema=GRADIENT_CONTRIBUTION_SCHEMA,
                aggregation=GRADIENT_CONTRIBUTION_CONTRACT,
                contribution_type=GradientContribution,
                profile_fingerprint=compute_profile_fingerprint(compute_profile),
            )
        finally:
            for parameter in self.model.parameters():
                parameter.grad = None
            self._inline_proximal = None
            self._inline_proximal_batch = None
            self._buffered_inputs = None
            self._active_microbatch_plan = None
            self._proximal_ready = False
            self._policy_scoring_target_mask = None
            self._active_rows = None
            self._fresh_behavior_by_row = {}
            self._active_group_prompt_keys = None
            self._active_group_prompt_keys_by_row = None
            self._active_dataset_prompt_keys = None
            self._parallel_global_policy_tokens = None
            self._parallel_reward_stats = {}
            self._active_rollout_output = None
            self.model.train(was_training)
            self._gradient_partition_context = False
            self._gradient_compute_active = False
            self._gradient_compute_lock.release()

    def _compute_gradient_contribution_locked(
        self,
        prepared_window,
        job,
        trajectories: Sequence[Mapping[str, Any]],
        *,
        installed_policy_version: int,
        compute_profile: Mapping[str, Any],
        contribution_schema: str,
        aggregation: str,
        contribution_type,
        profile_fingerprint: str,
    ):
        job_started = time.monotonic()
        self._last_gradient_timing = {"clock": "host_monotonic", "microbatch_seconds": []}
        if not trajectories:
            raise ValueError("gradient job must contain at least one prepared trajectory")
        if self._training_step_active or self._active_training_offload is not None:
            raise RuntimeError("cannot compute a gradient job during a Trainer training step")

        parameter_rows = sorted(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        )
        other_trainables = [
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        if other_trainables or not parameter_rows:
            raise RuntimeError("gradient executor requires an exclusively trainable LoRA parameter layout")
        actual_layout = tuple(
            (name, tuple(parameter.shape), str(parameter.dtype).removeprefix("torch."))
            for name, parameter in parameter_rows
        )
        contract_layout = tuple(
            (item.name, tuple(item.shape), item.dtype)
            for item in prepared_window.parameter_layout
        )
        if actual_layout != contract_layout:
            raise ValueError("loaded LoRA parameter layout differs from the prepared window")

        wire_rows = [dict(source) for source in trajectories]
        if any("_parallel_advantage" in row for row in wire_rows):
            raise ValueError("gradient rowbundle must not carry centrally computed advantages")
        wire_group_ids = {row.get("group_id") for row in wire_rows}
        if len(wire_group_ids) != 1:
            raise ValueError("each gradient job must contain exactly one complete task group")
        wire_group_id = next(iter(wire_group_ids))
        if wire_group_id != job.group_id:
            raise ValueError("gradient job group_id differs from its complete rowbundle group")
        if not isinstance(job.owner_executor_id, str) or not job.owner_executor_id:
            raise ValueError("gradient job is missing its owning executor identity")

        prepared_job = prepare_complete_gradient_rows(
            wire_rows,
            group_size=self.group_size,
            groups_per_update=1,
            microbatch_size=min(self.microbatch_size, len(wire_rows)),
        )
        from qwen_rl.contracts.gradient_partitions import selected_rows
        rows = selected_rows(prepared_window, job, prepared_job.rows)
        if len(prepared_job.groups) != 1 or wire_group_id not in prepared_job.groups:
            raise ValueError("gradient job must contain exactly one complete task group")
        token_count = sum(sum(row["policy_mask"][1:]) for row in rows)
        if token_count != job.token_count or token_count <= 0:
            raise ValueError(
                f"gradient job token count {token_count} differs from its contract {job.token_count}"
            )
        if job.n_window_tokens != prepared_window.n_window_tokens:
            raise ValueError("gradient job does not use the common window token denominator")

        behavior_versions: set[int] = set()
        for index, row in enumerate(prepared_job.rows):
            row_kind = row.get("row_kind", "fresh_student")
            ids = row["input_ids"]
            if row_kind == "fresh_student":
                behavior_version = row.get("policy_version")
                if type(behavior_version) is not int:
                    raise ValueError(f"fresh gradient row {index} has an invalid behavior policy version")
                policy_lag = installed_policy_version - behavior_version
                if policy_lag not in {0, 1}:
                    raise ValueError(
                        f"fresh gradient row {index} behavior policy lag must be 0 or 1, got {policy_lag}"
                    )
                behavior_versions.add(behavior_version)
                behavior = row.get("behavior_logprobs")
                if (not isinstance(behavior, (list, tuple)) or len(behavior) != len(ids)
                        or any(not math.isfinite(float(value)) for value in behavior)):
                    raise ValueError(f"fresh gradient row {index} has malformed behavior logprobs")
        if len(behavior_versions) > 1:
            raise ValueError("gradient job contains mixed fresh behavior policy versions")

        self._gradient_partition_context = "trajectory_partitions" in (prepared_window.update_context or {})
        self._last_actual_group_reward_stats = dict(prepared_job.reward_stats)
        self._parallel_reward_stats = dict(prepared_job.reward_stats)
        self._parallel_global_policy_tokens = int(prepared_window.n_window_tokens)
        self._last_group_cardinalities = {
            group_id: len(group_rows) for group_id, group_rows in prepared_job.groups.items()
        }

        for index, row in enumerate(rows):
            row["_parallel_source_index"] = index
            row["_parallel_row_id"] = index
            row.setdefault("row_kind", "fresh_student")
            row["input_ids"] = tuple(int(value) for value in row["input_ids"])
            row["policy_mask"] = tuple(int(value) for value in row["policy_mask"])
            if row["row_kind"] == "fresh_student":
                row["behavior_logprobs"] = tuple(float(value) for value in row["behavior_logprobs"])

        if self.max_padded_tokens_per_microbatch is not None:
            plan = plan_microbatches(
                rows,
                max_rows=self.microbatch_size,
                max_padded_tokens=self.max_padded_tokens_per_microbatch,
            )
            self._active_microbatch_plan = plan
            self._last_microbatch_plan_report = {
                "max_rows": self.microbatch_size,
                "max_padded_tokens": self.max_padded_tokens_per_microbatch,
                "same_prompt_length_only": True,
                "row_ids": [list(batch) for batch in plan.batches],
                "batch_sizes": [len(batch) for batch in plan.batches],
                "padded_tokens": list(plan.padded_tokens),
                "actual_tokens": plan.actual_tokens,
                "total_padded_tokens": plan.total_padded_tokens,
                "oversize_singletons": list(plan.oversize_singletons),
            }
            actual_microbatches = len(plan.batches)
        else:
            self._active_microbatch_plan = None
            self._last_microbatch_plan_report = {}
            actual_microbatches = math.ceil(len(rows) / self.microbatch_size)

        group_keys = {
            wire_group_id: f"qwen-rl-gradient-{job.job_id}-0"
        }
        self._active_rows = rows
        self._fresh_behavior_by_row = {
            row_index: tuple(row["behavior_logprobs"])
            for row_index, row in enumerate(rows)
            if row.get("row_kind", "fresh_student") == "fresh_student"
        }
        self._active_group_prompt_keys = list(group_keys.values())
        self._active_group_prompt_keys_by_row = [group_keys[row["group_id"]] for row in rows]
        self._active_dataset_prompt_keys = list(self._active_group_prompt_keys_by_row)
        self._active_rollout_output = None
        self._last_actual_gradient_accumulation_steps = actual_microbatches
        self.model.train()
        prompts = list(self._active_group_prompt_keys_by_row)
        generated = self._generate_and_score_completions([{"prompt": key} for key in prompts])
        generated = split_pixel_values_by_grid(generated)
        if self._active_microbatch_plan is not None:
            generated_microbatches = self._split_planned_generation_batch(generated)
        else:
            generated = shuffle_sequence_dict(generated)
            generated_microbatches = self._split_generation_batch(generated, actual_microbatches)
        # TRL computes this field from the local generated rows. A gradient job
        # contributes to the complete window mean, so every local loss must use
        # the same central window denominator, regardless of job size.
        for batch in generated_microbatches:
            batch["num_items_in_batch"] = torch.tensor(
                prepared_window.n_window_tokens,
                dtype=torch.long,
                device=self.accelerator.device,
            )
        generated_microbatches = [
            unsplit_pixel_values_by_grid(batch) for batch in generated_microbatches
        ]
        row_metadata_by_batch = self._trim_microbatch_padding(generated_microbatches)
        microbatches = list(zip(generated_microbatches, row_metadata_by_batch, strict=True))
        self._last_scoring_metrics = {
            "eligible_completion_targets": sum(
                len(row["input_ids"]) - row["policy_mask"].index(1) for row in rows
            ),
            "policy_targets_projected": sum(sum(row["policy_mask"][row["policy_mask"].index(1):])
                                              for row in rows if row.get("row_kind") != "zero_credit"),
        }
        self._last_scoring_metrics["non_policy_completion_targets_skipped"] = (
            self._last_scoring_metrics["eligible_completion_targets"]
            - self._last_scoring_metrics["policy_targets_projected"]
        )
        # Execute only real training work. FLA still performs any required JIT
        # on its first real call; do not run a separate synthetic backward sweep.
        self._last_gdn_warmup = {
            "status": "skipped", "reason": "gradient_executor_no_synthetic_prewarm",
            "warmup_seconds": 0.0,
        }
        self._last_gradient_timing["preparation_seconds"] = time.monotonic() - job_started

        scheduler_before = (
            copy.deepcopy(self.lr_scheduler.state_dict())
            if self.lr_scheduler is not None else None
        )
        step_before = int(self.state.global_step)
        parameter_versions_before = _model_parameter_version(self.model)
        was_training = self.model.training
        contributions = None
        raw_log_ratios: list[torch.Tensor] = []
        fresh_behavior_tokens = 0
        replay_policy_tokens = 0
        loss_values: list[float] = []
        from .gradient_executor.accumulation import DeviceGradientAccumulator
        device_gradients = DeviceGradientAccumulator(parameter_rows)
        try:
            for _, parameter in parameter_rows:
                parameter.grad = None
            self.model.train()
            self._last_saved_tensor_offload_report = None
            self._offload_microbatch_reports = []
            self._last_correction_metrics = {}
            self._last_upstream_clip_metrics = {}
            self._last_train_output = None
            self._last_trainer_metrics = {}
            self._buffered_inputs = [batch for batch, _ in microbatches]
            self._step = 0
            self._expand_completion_log_buffers(len(rows))
            self._active_rows = rows
            self._fresh_behavior_by_row = {
                row_index: tuple(row["behavior_logprobs"])
                for row_index, row in enumerate(rows)
                if row.get("row_kind", "fresh_student") == "fresh_student"
            }
            self._proximal_ready = True
            self._begin_inline_proximal(
                [batch for batch, _ in microbatches],
                row_metadata_by_batch=[metadata for _, metadata in microbatches],
            )
            forward_backward_started = time.monotonic()
            operation_telemetry = getattr(self, "_operation_telemetry", None)
            for microbatch_index, (inputs, row_metadata) in enumerate(microbatches):
                microbatch_started = time.monotonic()
                self._inline_proximal_batch = None
                num_items = torch.tensor(
                    prepared_window.n_window_tokens,
                    dtype=torch.long,
                    device=self.accelerator.device,
                )
                input_tokens = sum(
                    len(self._active_rows[row_id]["input_ids"])
                    for row_id, _row_kind in row_metadata
                )
                with self._training_microbatch_scope(inputs):
                    forward_token = None
                    if operation_telemetry is not None:
                        try:
                            forward_token = operation_telemetry.begin_phase(
                                "forward", microbatch_index, input_tokens,
                                device=self.accelerator.device,
                            )
                        except Exception:
                            forward_token = None
                    forward_succeeded = False
                    try:
                        loss = self.compute_loss(
                            self.model,
                            inputs,
                            return_outputs=False,
                            num_items_in_batch=num_items,
                        )
                        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                            raise RuntimeError("pinned TRL DAPO compute_loss must return one scalar tensor")
                        if not bool(torch.isfinite(loss.detach())):
                            raise FloatingPointError("TRL DAPO produced a non-finite gradient-job loss")
                        forward_succeeded = True
                    finally:
                        if operation_telemetry is not None and forward_token is not None:
                            try:
                                operation_telemetry.end_phase(forward_token, succeeded=forward_succeeded)
                            except Exception:
                                pass

                    # Gradient checkpoint recomputation occurs inside backward
                    # and is intentionally included in this CUDA-event interval.
                    backward_token = None
                    if operation_telemetry is not None:
                        try:
                            backward_token = operation_telemetry.begin_phase(
                                "backward", microbatch_index, input_tokens,
                                device=self.accelerator.device,
                            )
                        except Exception:
                            backward_token = None
                    backward_succeeded = False
                    try:
                        self.accelerator.backward(loss)
                        backward_succeeded = True
                    finally:
                        if operation_telemetry is not None and backward_token is not None:
                            try:
                                operation_telemetry.end_phase(backward_token, succeeded=backward_succeeded)
                            except Exception:
                                pass
                device_gradients.add_microbatch()
                loss_values.append(float(loss.detach().float().cpu()))
                self._last_gradient_timing["microbatch_seconds"].append(time.monotonic() - microbatch_started)
            self._last_gradient_timing["forward_backward_seconds"] = time.monotonic() - forward_backward_started
            diagnostics_started = time.monotonic()
            self._assert_inline_proximal_complete()
            collector = self._inline_proximal.collector
            fresh_behavior_tokens = int(collector.policy_tokens_by_kind["fresh_student"])
            replay_policy_tokens = int(
                collector.policy_tokens_by_kind["native_replay"]
                + collector.policy_tokens_by_kind["guided_replay"]
            )
            raw_log_ratios = [
                delta.detach().float().cpu().clone()
                for delta in collector.raw_deltas
            ]

            if int(self.state.global_step) != step_before:
                raise RuntimeError("gradient-only computation changed Trainer optimizer progress")
            if _model_parameter_version(self.model) != parameter_versions_before:
                raise RuntimeError("gradient-only computation changed model parameter values")
            scheduler_after = (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None else None
            )
            if scheduler_after != scheduler_before:
                raise RuntimeError("gradient-only computation changed scheduler state")

            if fresh_behavior_tokens <= 0 and replay_policy_tokens <= 0:
                raise ValueError("gradient partition has no policy tokens")
            capped_tokens = 0
            log_ratio_summary = {"log_ratio_scope": "not_applicable_replay_only"}
            if raw_log_ratios:
                combined_log_ratios = torch.cat(raw_log_ratios)
                quantiles = torch.quantile(
                    combined_log_ratios,
                    torch.tensor([0.0, 0.5, 0.95, 1.0], dtype=torch.float32),
                ).tolist()
                capped_tokens = int((combined_log_ratios > math.log(self.importance_sampling_cap)).sum())
                log_ratio_summary = dict(zip(
                    ("log_ratio_min", "log_ratio_median", "log_ratio_p95", "log_ratio_max"),
                    map(float, quantiles), strict=True,
                ))
            loss_sum = float(sum(loss_values))
            self._last_gradient_diagnostics = {
                "fresh_policy_tokens": fresh_behavior_tokens,
                "fresh_behavior_tokens": fresh_behavior_tokens,
                "replay_policy_tokens": replay_policy_tokens,
                "correction_capped_tokens": capped_tokens,
                "correction_capped_fraction": (
                    capped_tokens / fresh_behavior_tokens if fresh_behavior_tokens else 0.0
                ),
                **log_ratio_summary,
                "dapo_loss_sum": loss_sum,
                "dapo_loss_mean": loss_sum / len(loss_values),
                "dapo_loss_microbatch_count": len(loss_values),
            }

            self._last_gradient_timing["diagnostics_seconds"] = time.monotonic() - diagnostics_started
            export_started = time.monotonic()
            gradient_map = device_gradients.export_cpu()
            contributions = contribution_type(
                schema=contribution_schema,
                aggregation=aggregation,
                lineage_id=job.lineage_id,
                engine_epoch=job.engine_epoch,
                window_id=job.window_id,
                attempt_id=job.attempt_id,
                job_id=job.job_id,
                group_id=job.group_id,
                owner_executor_id=job.owner_executor_id,
                policy_version=job.policy_version,
                n_window_tokens=job.n_window_tokens,
                token_count=job.token_count,
                layout_fingerprint=job.layout_fingerprint,
                compute_profile_fingerprint=profile_fingerprint,
                gradients=gradient_map,
                diagnostics=dict(self._last_gradient_diagnostics),
            )
            self._last_gradient_timing["gradient_export_and_validation_seconds"] = time.monotonic() - export_started
            self._last_gradient_timing["total_before_cleanup_seconds"] = time.monotonic() - job_started
        finally:
            # Failed thread tracebacks may outlive GPU session return. Release
            # this job's private device sum before propagating the exception.
            device_gradients.gradients.clear()
            for _, parameter in parameter_rows:
                parameter.grad = None
            self._inline_proximal = None
            self._inline_proximal_batch = None
            self._buffered_inputs = None
            self._active_microbatch_plan = None
            self._proximal_ready = False
            self._policy_scoring_target_mask = None
            self._active_rows = None
            self._fresh_behavior_by_row = {}
            self._active_group_prompt_keys = None
            self._active_group_prompt_keys_by_row = None
            self._active_dataset_prompt_keys = None
            self._parallel_global_policy_tokens = None
            self._parallel_reward_stats = {}
            self._active_rollout_output = None
            self.model.train(was_training)

        return contributions

    def train_groups(
        self,
        trajectories: list[dict[str, Any]],
        *,
        before_optimizer_step: Callable[[], None] | None = None,
        defer_telemetry_finish: bool = False,
    ):
        """Apply one optimizer update to complete fresh and replay GRPO groups."""
        if self.gradient_only:
            raise RuntimeError("gradient-only Trainer cannot apply train_groups optimizer updates")
        if type(defer_telemetry_finish) is not bool:
            raise ValueError("defer_telemetry_finish must be a boolean")
        if not trajectories:
            raise ValueError("train_groups requires at least one trajectory")
        prepared = self.prepare_gradient_rows(trajectories)
        grouped_rows = {group_id: list(rows) for group_id, rows in prepared.groups.items()}
        ordered_group_ids = list(grouped_rows)
        ordered_rows = list(prepared.rows)
        global_policy_tokens = prepared.n_window_tokens
        reward_stats = dict(prepared.reward_stats)
        self._parallel_global_policy_tokens = global_policy_tokens
        self._parallel_reward_stats = reward_stats

        if self.parallel_mode in {"ddp", "fsdp2"}:
            from .parallel.topology import balance_rows_for_data_parallel

            dataset_rows, local_rows_by_rank = balance_rows_for_data_parallel(
                ordered_rows,
                world_size=self.parallel_world_size,
                microbatch_size=self.microbatch_size,
            )
            local_rows = local_rows_by_rank[self.parallel_rank]
            actual_gradient_accumulation_steps = len(local_rows) // self.microbatch_size
        else:
            dataset_rows = []
            for index, row in enumerate(ordered_rows):
                copied = dict(row)
                copied["_parallel_source_index"] = index
                copied["_parallel_row_id"] = index
                dataset_rows.append(copied)
            local_rows = dataset_rows
            actual_gradient_accumulation_steps = math.ceil(len(local_rows) / self.microbatch_size)
        self._active_microbatch_plan = None
        self._last_microbatch_plan_report = {}
        if self.max_padded_tokens_per_microbatch is not None:
            plan = plan_microbatches(
                local_rows, max_rows=self.microbatch_size,
                max_padded_tokens=self.max_padded_tokens_per_microbatch,
            )
            self._active_microbatch_plan = plan
            actual_gradient_accumulation_steps = len(plan.batches)
            self._last_microbatch_plan_report = {
                "max_rows": self.microbatch_size,
                "max_padded_tokens": self.max_padded_tokens_per_microbatch,
                "same_prompt_length_only": True,
                "row_ids": [list(batch) for batch in plan.batches],
                "batch_sizes": [len(batch) for batch in plan.batches],
                "padded_tokens": list(plan.padded_tokens),
                "actual_tokens": plan.actual_tokens,
                "total_padded_tokens": plan.total_padded_tokens,
                "oversize_singletons": list(plan.oversize_singletons),
            }
        previous_gradient_accumulation_steps = self.args.gradient_accumulation_steps
        previous_steps_per_generation = self.args.steps_per_generation
        self._last_actual_gradient_accumulation_steps = actual_gradient_accumulation_steps
        self._last_group_cardinalities = {
            group_id: len(rows) for group_id, rows in grouped_rows.items()
        }
        self._last_actual_group_reward_stats: dict[str, float] = {}
        self._last_scoring_metrics = {}
        self._active_rows = []
        self._fresh_behavior_by_row = {}
        for row in local_rows:
            copied = dict(row)
            copied.setdefault("row_kind", "fresh_student")
            copied["input_ids"] = tuple(row["input_ids"])
            copied["policy_mask"] = tuple(int(value) for value in row["policy_mask"])
            if copied["row_kind"] == "fresh_student":
                behavior = tuple(float(value) for value in row["behavior_logprobs"])
                self._fresh_behavior_by_row[len(self._active_rows)] = behavior
                copied["behavior_logprobs"] = behavior
            self._active_rows.append(copied)
        completion_target_positions = 0
        policy_target_positions = 0
        for row in self._active_rows:
            split = row["policy_mask"].index(1)
            completion_target_positions += len(row["input_ids"]) - split
            if row.get("row_kind") != "zero_credit":
                policy_target_positions += sum(row["policy_mask"][split:])
        self._last_scoring_metrics = {
            "eligible_completion_targets": completion_target_positions,
            "policy_targets_projected": policy_target_positions,
            "non_policy_completion_targets_skipped": completion_target_positions - policy_target_positions,
        }
        update_key = uuid.uuid4().hex
        self._active_group_prompt_keys = [
            f"qwen-rl-group-{update_key}-{index}" for index in range(self.groups_per_update)
        ]
        group_prompt_key = dict(zip(ordered_group_ids, self._active_group_prompt_keys, strict=True))
        for row in dataset_rows:
            group_id = row["group_id"]
            group_prompt_key.setdefault(
                group_id, f"qwen-rl-zero-credit-{update_key}-{group_id}"
            )
        self._active_group_prompt_keys_by_row = [
            group_prompt_key[row["group_id"]] for row in local_rows
        ]
        self._active_dataset_prompt_keys = [group_prompt_key[row["group_id"]] for row in dataset_rows]
        self._active_rollout_output = None
        if before_optimizer_step is not None and not callable(before_optimizer_step):
            raise TypeError("before_optimizer_step must be callable")
        def authorize_optimizer_step():
            if self._active_training_offload is not None:
                raise RuntimeError("activation offload must close before optimizer step")
            self._assert_inline_proximal_complete()
            self._flush_deferred_dapo_metrics()
            if before_optimizer_step is not None:
                before_optimizer_step()

        self._gradient_guard.begin_update(authorize_optimizer_step)
        self._inline_proximal = None
        self._inline_proximal_batch = None
        self._buffered_inputs = None
        self._proximal_ready = False
        self._policy_scoring_target_mask = None
        self._last_correction_metrics = {}
        self._last_padding_metrics = {}
        self._last_gdn_warmup = {}
        self._last_upstream_clip_metrics = {}
        self._step = 0
        self._expand_completion_log_buffers(len(ordered_rows))
        self.telemetry.begin(
            ordered_rows,
            planned_microbatches=actual_gradient_accumulation_steps,
            update_id=update_key,
        )
        accelerator_config = self.args.accelerator_config
        split_batches = (
            accelerator_config.get("split_batches", False)
            if isinstance(accelerator_config, dict)
            else bool(getattr(accelerator_config, "split_batches", False))
        )
        self._report_parallel_phase(
            "rows_staged",
            rank=self.parallel_rank,
            mode=self.parallel_mode,
            world_size=self.parallel_world_size,
            global_rows=len(ordered_rows),
            dataloader_rows=len(dataset_rows),
            local_rows=len(self._active_rows),
            local_row_ids=[row.get("_parallel_row_id") for row in self._active_rows],
            local_row_kinds=[row.get("row_kind", "fresh_student") for row in self._active_rows],
            local_token_lengths=[len(row["input_ids"]) for row in self._active_rows],
            microbatches=actual_gradient_accumulation_steps,
            split_batches=split_batches,
        )
        try:
            self.args.gradient_accumulation_steps = actual_gradient_accumulation_steps
            self.args.steps_per_generation = actual_gradient_accumulation_steps
            # Trainer persists metric deques between train() calls when logging
            # is disabled. Start this one-update call with an empty upstream
            # train history so quality scalars cannot include older attempts.
            for values in self._metrics["train"].values():
                values.clear()
            self._deferred_dapo_metric_window = (
                DeferredDapoMetricWindow(
                    expected_microbatches=actual_gradient_accumulation_steps,
                    rows_per_microbatch=self.microbatch_size,
                )
                if self._deferred_dapo_metrics_are_supported()
                else None
            )
            self.train_dataset = Dataset.from_dict({"prompt": self._active_dataset_prompt_keys})
            self.failure_stage = "trainer_start"
            # Each training_step closes its copy/prefetch state before optimizer.
            self._last_saved_tensor_offload_report = None
            self._offload_microbatch_reports = []
            self._report_parallel_phase("trainer_train_started", rank=self.parallel_rank)
            with self._bf16_lora_input_dtype_scope():
                output = self.train()
            self._report_parallel_phase("trainer_train_completed", rank=self.parallel_rank)
            self.failure_stage = "post_optimizer_validation"
            if self._active_rollout_output is None:
                raise RuntimeError("upstream Trainer.train() completed without invoking the external rollout")
            if (
                not self._gradient_guard.pre_optimizer_step_seen
                or self._gradient_guard.optimizer_step_count != 1
            ):
                raise RuntimeError(
                    "upstream Trainer.train() did not reach exactly one guarded optimizer step "
                    f"(observed {self._gradient_guard.optimizer_step_count})"
                )
            if (before_optimizer_step is not None
                    and not self._gradient_guard.before_optimizer_step_called):
                raise RuntimeError("Trainer did not invoke before_optimizer_step before its optimizer step")
            if self.accelerator.optimizer_step_was_skipped:
                raise RuntimeError("upstream Trainer skipped the optimizer step")
            self._last_train_output = output
            self._last_trainer_metrics = {
                str(key): float(value)
                for key, value in output.metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            # With logging disabled, GRPOTrainer retains its real upstream
            # clip scalars in _metrics until the next train() call. Capture
            # their logger-equivalent mean without reconstructing clip flags.
            for name, values in self._metrics["train"].items():
                if not name.startswith("clip_ratio/") or not values:
                    continue
                finite = [float(value) for value in values if math.isfinite(float(value))]
                if finite:
                    self._last_upstream_clip_metrics[name.split("clip_ratio/", 1)[1]] = sum(finite) / len(finite)
            self._last_trainer_metrics.update({
                f"train/padding/{key}": value
                for key, value in self._last_padding_metrics.items()
            })
            self._last_trainer_metrics.update({
                f"train/fused_head/{key}": float(value)
                for key, value in self._last_scoring_metrics.items()
            })
            self._last_trainer_metrics.update({
                "train/gdn_warmup/microbatches": float(self._last_gdn_warmup.get("microbatches", 0)),
                "train/gdn_warmup/profiles_required": float(self._last_gdn_warmup.get("profiles_required", 0)),
                "train/gdn_warmup/profiles_attempted": float(
                    self._last_gdn_warmup.get("profiles_attempted_this_update", 0)
                ),
                "train/gdn_warmup/profiles_warmed": float(
                    self._last_gdn_warmup.get("profiles_warmed_this_update", 0)
                ),
                "train/gdn_warmup/profiles_already_attempted": float(
                    self._last_gdn_warmup.get("profiles_already_attempted", 0)
                ),
                "train/gdn_warmup/profiles_attempted_total_process": float(
                    self._last_gdn_warmup.get("profiles_attempted_total_process", 0)
                ),
                "train/gdn_warmup/profiles_warmed_total_process": float(
                    self._last_gdn_warmup.get("profiles_warmed_total_process", 0)
                ),
                "train/gdn_warmup/seconds": float(self._last_gdn_warmup.get("warmup_seconds", 0.0)),
            })
            self._last_trainer_metrics.update({
                "train/batch/actual_rows": float(len(ordered_rows)),
                "train/batch/group_count": float(len(grouped_rows)),
                "train/batch/gradient_accumulation_steps": float(actual_gradient_accumulation_steps),
            })
            for cardinality in sorted(set(self._last_group_cardinalities.values())):
                self._last_trainer_metrics[
                    f"train/batch/groups_with_{cardinality}_rows"
                ] = float(sum(size == cardinality for size in self._last_group_cardinalities.values()))
            self._last_trainer_metrics.update({
                f"train/batch/{name}": value
                for name, value in self._last_actual_group_reward_stats.items()
            })
            self._last_successful_quality_snapshot = self._build_quality_snapshot()
            if not defer_telemetry_finish:
                self.telemetry.finish_attempt(
                    status="completed",
                    quality=self._last_successful_quality_snapshot,
                )
            return output
        except BaseException as exc:
            if not defer_telemetry_finish:
                self.telemetry.finish_attempt(
                    status="failed",
                    failure_stage=self.failure_stage,
                    error_type=type(exc).__name__,
                )
            logging.getLogger(__name__).exception("Learner update failed at %s", self.failure_stage)
            raise
        finally:
            self.args.gradient_accumulation_steps = previous_gradient_accumulation_steps
            self.args.steps_per_generation = previous_steps_per_generation
            if self.optimizer is not None:
                # Accelerate skips zero_grad while a gradient-accumulation
                # window is unsynchronized. A failed intermediate microbatch
                # leaves that flag false, so force an unsynchronized clear.
                self.accelerator.gradient_state._set_sync_gradients(True)
                self.optimizer.zero_grad(set_to_none=True)
            self._inline_proximal = None
            self._inline_proximal_batch = None
            self._buffered_inputs = None
            self._active_microbatch_plan = None
            self._proximal_ready = False
            self._policy_scoring_target_mask = None
            self._active_rows = None
            self._fresh_behavior_by_row = {}
            self._active_group_prompt_keys = None
            self._active_group_prompt_keys_by_row = None
            self._active_dataset_prompt_keys = None
            self._parallel_global_policy_tokens = None
            self._parallel_reward_stats = {}
            self._active_rollout_output = None
            self._gradient_guard.before_optimizer_step_callback = None
            self._deferred_dapo_metric_window = None
