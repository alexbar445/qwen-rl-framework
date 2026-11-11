"""Four-stage Learner backend built on PyTorch's distributed pipelining runtime.

The Qwen model assembly remains responsible for producing one complete ``nn.Module``
per stage. This module owns only the real cross-rank 1F1B schedule and the shared
DAPO/LoRA invariants; it deliberately does not implement a P2P scheduler.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

PIPELINE_STAGE_COUNT = 4


class PipelineContractError(ValueError):
    """The stage layout or update data violates the PP Learner contract."""


def _is_lora_parameter(name: str) -> bool:
    return any(part.startswith("lora_") for part in name.split("."))


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _pipeline_activation_metadata(value: torch.Tensor | tuple[torch.Tensor, ...]):
    """Describe floating PipelineStage inputs/outputs as grad-enabled slots."""
    tensors = value if isinstance(value, tuple) else (value,)
    if not tensors or any(not isinstance(item, torch.Tensor) for item in tensors):
        raise PipelineContractError("stage output metadata must be a tensor or non-empty tensor tuple")
    normalized = tuple(
        item.detach().requires_grad_(True)
        if item.is_floating_point() or item.is_complex()
        else item.detach()
        for item in tensors
    )
    return normalized if isinstance(value, tuple) else normalized[0]


def _stage_parameter_signature(module: nn.Module) -> tuple[tuple[str, tuple[int, ...], str, bool], ...]:
    return tuple(sorted(
        (name, tuple(parameter.shape), str(parameter.dtype), bool(parameter.requires_grad))
        for name, parameter in module.named_parameters()
    ))


def trainable_lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return trainable PEFT LoRA parameters and reject trainable base weights."""
    named = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    if not named:
        raise PipelineContractError("pipeline stage has no trainable LoRA parameters")
    invalid = [name for name, _ in named if not _is_lora_parameter(name)]
    if invalid:
        raise PipelineContractError(
            "pipeline stage has trainable parameters outside LoRA: " + ", ".join(invalid[:8])
        )
    return [parameter for _, parameter in named]


def causal_policy_token_count(policy_mask: torch.Tensor, row_credit: torch.Tensor | None = None) -> int:
    """Count global DAPO tokens from a full-sequence model-token mask.

    Position zero has no causal predecessor and is excluded, matching Learner's
    immutable trajectory contract. ``row_credit`` can exclude zero-credit rows.
    """
    if not isinstance(policy_mask, torch.Tensor) or policy_mask.ndim != 2 or policy_mask.shape[1] < 2:
        raise PipelineContractError("policy_mask must be a [batch, sequence>=2] tensor")
    if policy_mask.dtype != torch.bool and not bool(((policy_mask == 0) | (policy_mask == 1)).all()):
        raise PipelineContractError("policy_mask must contain only boolean or 0/1 values")
    active = policy_mask.bool().clone()
    if bool(active[:, 0].any()):
        raise PipelineContractError("position zero cannot be a causal policy target")
    if row_credit is not None:
        if not isinstance(row_credit, torch.Tensor) or row_credit.ndim != 1 or row_credit.shape[0] != active.shape[0]:
            raise PipelineContractError("row_credit must have one value per trajectory")
        if row_credit.dtype != torch.bool and not bool(((row_credit == 0) | (row_credit == 1)).all()):
            raise PipelineContractError("row_credit must contain only boolean or 0/1 values")
        active &= row_credit.bool().unsqueeze(1)
    total = int(active[:, 1:].sum().item())
    if total <= 0:
        raise PipelineContractError("pipeline DAPO batch has no active causal policy tokens")
    return total


def proximal_correction(
    proximal_logps: torch.Tensor,
    behavior_logps: torch.Tensor,
    policy_mask: torch.Tensor,
    *,
    cap: float = 3.0,
) -> torch.Tensor:
    """Return detached ``min(exp(p-b), cap)`` on model-selected policy tokens."""
    if not math.isfinite(cap) or cap < 1.0:
        raise PipelineContractError("importance-sampling cap must be finite and >= 1")
    if proximal_logps.shape != behavior_logps.shape or proximal_logps.shape != policy_mask.shape:
        raise PipelineContractError("p, b, and policy_mask must have identical shapes")
    mask = policy_mask.bool()
    if not bool(torch.isfinite(proximal_logps[mask]).all()) or not bool(torch.isfinite(behavior_logps[mask]).all()):
        raise PipelineContractError("p and b must be finite on selected policy tokens")
    delta = proximal_logps.float() - behavior_logps.float()
    if not bool(torch.isfinite(delta[mask]).all()):
        raise PipelineContractError("p-b is non-finite on selected policy tokens")
    result = torch.zeros_like(delta)
    result[mask] = torch.exp(delta[mask].clamp(max=math.log(cap)))
    return result.detach()


def dapo_global_token_mean(
    per_token_loss: torch.Tensor,
    policy_mask: torch.Tensor,
    *,
    global_policy_tokens: int,
) -> torch.Tensor:
    """Normalize a microbatch's masked TRL DAPO token-loss sum by the full batch.

    Pipeline ranks process the same examples as consecutive model stages, so the
    denominator is computed once from the full update mask, without dividing by
    pipeline world size. Configure ``Schedule1F1B(scale_grads=False)`` so the
    schedule does not apply a second microbatch average.
    """
    if per_token_loss.shape != policy_mask.shape or per_token_loss.ndim != 2:
        raise PipelineContractError("per_token_loss and policy_mask must be matching [batch, sequence] tensors")
    if isinstance(global_policy_tokens, bool) or not isinstance(global_policy_tokens, int) or global_policy_tokens <= 0:
        raise PipelineContractError("global_policy_tokens must be a positive integer")
    mask = policy_mask.to(device=per_token_loss.device, dtype=torch.float32)
    return (per_token_loss.float() * mask).sum() / global_policy_tokens


def validate_trl_contract(
    trainer: Any,
    process_group: Any,
    microbatches: int,
    *,
    expected_process_count: int = PIPELINE_STAGE_COUNT,
) -> None:
    """Fail closed unless TRL's normalizer cancels to the full-update token count."""
    if not dist.is_available() or not dist.is_initialized():
        raise PipelineContractError("initialize torch.distributed before validating the TRL PP denominator")
    if isinstance(microbatches, bool) or not isinstance(microbatches, int) or microbatches < PIPELINE_STAGE_COUNT:
        raise PipelineContractError("Schedule1F1B needs at least one microbatch per pipeline stage")
    if (isinstance(expected_process_count, bool)
            or not isinstance(expected_process_count, int)
            or expected_process_count != PIPELINE_STAGE_COUNT):
        raise PipelineContractError("the TRL denominator contract requires exactly four PP processes")
    group_count = dist.get_world_size(group=process_group)
    accelerator = getattr(trainer, "accelerator", None)
    accelerator_count = getattr(accelerator, "num_processes", None)
    args = getattr(trainer, "args", None)
    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", None)
    steps_per_generation = getattr(args, "steps_per_generation", None)
    counts = {
        "pipeline process count": group_count,
        "Trainer accelerator process count": accelerator_count,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_generation": steps_per_generation,
        "Schedule1F1B microbatches": microbatches,
        "batch context process_count": expected_process_count,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts.values()):
        raise PipelineContractError(f"TRL PP denominator counts must all be integers: {counts}")
    if any(value != PIPELINE_STAGE_COUNT for value in (group_count, accelerator_count, expected_process_count)):
        raise PipelineContractError(f"TRL process counts must match the four-stage PP group: {counts}")
    if not (gradient_accumulation_steps == steps_per_generation == microbatches):
        raise PipelineContractError(
            "TRL normalization requires gradient_accumulation_steps == steps_per_generation == "
            f"Schedule1F1B microbatches: {counts}"
        )


@dataclass(frozen=True)
class PipelineBatchContext:
    """Immutable full-update tensors shared by the PP ranks before the 1F1B step."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    prompt_width: int
    completion_mask: torch.Tensor
    tool_mask: torch.Tensor
    advantages: torch.Tensor
    proximal_logps: torch.Tensor
    behavior_logps: torch.Tensor
    fresh_row_mask: torch.Tensor
    row_credit: torch.Tensor
    global_policy_tokens: int
    process_count: int = PIPELINE_STAGE_COUNT
    importance_sampling_cap: float = 3.0

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise PipelineContractError("pipeline input_ids and attention_mask must match [batch, sequence]")
        batch, sequence = self.input_ids.shape
        if self.position_ids.shape != (4, batch, sequence):
            raise PipelineContractError("Qwen3.5 position_ids must have shape [4,batch,sequence]")
        if isinstance(self.prompt_width, bool) or not 1 <= self.prompt_width < sequence:
            raise PipelineContractError("prompt_width must leave a non-empty completion")
        completion_width = sequence - self.prompt_width
        expected = (batch, completion_width)
        for name, value in (
            ("completion_mask", self.completion_mask),
            ("tool_mask", self.tool_mask),
            ("proximal_logps", self.proximal_logps),
            ("behavior_logps", self.behavior_logps),
        ):
            if value.shape != expected:
                raise PipelineContractError(f"{name} must match the completion shape {expected}")
        if self.advantages.shape != (batch,):
            raise PipelineContractError("group advantages must contain one value per trajectory")
        if self.fresh_row_mask.shape != (batch,) or self.fresh_row_mask.dtype != torch.bool:
            raise PipelineContractError("fresh_row_mask must contain one boolean per trajectory")
        if self.row_credit.shape != (batch,) or self.row_credit.dtype != torch.bool:
            raise PipelineContractError("row_credit must contain one boolean per trajectory")
        if isinstance(self.process_count, bool) or not isinstance(self.process_count, int) or self.process_count < 1:
            raise PipelineContractError("process_count must be a positive integer")
        if (isinstance(self.global_policy_tokens, bool)
                or not isinstance(self.global_policy_tokens, int)
                or self.global_policy_tokens <= 0):
            raise PipelineContractError("global_policy_tokens must be positive")
        if self.input_ids.dtype not in (torch.int32, torch.int64):
            raise PipelineContractError("input_ids must be integer token IDs")
        if int(self.input_ids.max().item()) >= 2**24:
            raise PipelineContractError("packed target cannot represent token IDs above float32's exact integer range")
        if int(self.input_ids.min().item()) < 0:
            raise PipelineContractError("input_ids cannot contain negative token IDs")
        for name, mask in (("attention_mask", self.attention_mask),
                           ("completion_mask", self.completion_mask),
                           ("tool_mask", self.tool_mask)):
            if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
                raise PipelineContractError(f"{name} must contain only boolean or 0/1 values")
        if not bool(torch.isfinite(self.advantages).all()):
            raise PipelineContractError("group advantages must be finite")
        policy_mask = self.policy_mask
        observed_tokens = int(policy_mask.sum().item())
        if observed_tokens != self.global_policy_tokens:
            raise PipelineContractError(
                f"global DAPO denominator {self.global_policy_tokens} differs from the full policy mask {observed_tokens}"
            )
        if not bool(torch.isfinite(self.proximal_logps[policy_mask]).all()):
            raise PipelineContractError("frozen proximal p must be finite on all active policy tokens")
        fresh_policy = policy_mask & self.fresh_row_mask.unsqueeze(1)
        if not bool(torch.isfinite(self.behavior_logps[fresh_policy]).all()):
            raise PipelineContractError("fresh behavior b must be finite on active policy tokens")
        if bool(torch.isfinite(self.behavior_logps[policy_mask & ~self.fresh_row_mask.unsqueeze(1)]).any()):
            raise PipelineContractError("replay rows must keep behavior b absent, represented by NaN")
        if self.proximal_logps.requires_grad or self.behavior_logps.requires_grad or self.advantages.requires_grad:
            raise PipelineContractError("p, b, and group advantages must be frozen before the PP update")
        if not math.isfinite(self.importance_sampling_cap) or self.importance_sampling_cap < 1.0:
            raise PipelineContractError("importance_sampling_cap must be finite and >= 1")

    @property
    def policy_mask(self) -> torch.Tensor:
        return (
            self.completion_mask.bool()
            & self.tool_mask.bool()
            & self.row_credit.unsqueeze(1)
        )

    @property
    def corrected_advantages(self) -> torch.Tensor:
        mask = self.policy_mask
        fresh_mask = mask & self.fresh_row_mask.unsqueeze(1)
        weights = proximal_correction(
            self.proximal_logps,
            self.behavior_logps,
            fresh_mask,
            cap=self.importance_sampling_cap,
        )
        replay_mask = mask & ~self.fresh_row_mask.unsqueeze(1)
        weights[replay_mask] = 1.0
        return self.advantages.float()[:, None] * weights

    @property
    def shape_signature(self) -> tuple[int, ...]:
        """The Pipe static-shape key; rebuild PipelineStage on a different key."""
        batch, sequence = self.input_ids.shape
        return (batch, sequence, self.prompt_width, self.process_count)

    def pipeline_model_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token IDs plus float row indices; side context stays local to each rank."""
        if self.input_ids.shape[0] >= 2**24:
            raise PipelineContractError("float32 PP row_ids require fewer than 2^24 update rows")
        row_ids = torch.arange(
            self.input_ids.shape[0],
            dtype=torch.float32,
            device=self.input_ids.device,
        ).requires_grad_(True)
        return self.input_ids, row_ids

    @contextmanager
    def bind_pipeline_context(self, stage_model: nn.Module):
        """Bind mask and four-axis positions to this rank's Qwen stage for one step."""
        get_base_model = getattr(stage_model, "get_base_model", None)
        stage = get_base_model() if callable(get_base_model) else stage_model
        bind = getattr(stage, "bind_batch_context", None)
        if not callable(bind):
            raise PipelineContractError("rank-local model does not expose bind_batch_context")
        with bind(self.attention_mask, self.position_ids):
            yield stage

    def packed_target(self) -> torch.Tensor:
        """Pack loss fields into one schedule target tensor, split by upstream 1F1B."""
        batch, sequence = self.input_ids.shape
        target = torch.zeros((batch, sequence, 7), dtype=torch.float32, device=self.input_ids.device)
        target[:, :, 0] = self.input_ids.float()
        target[:, :, 1] = self.attention_mask.float()
        completion_slice = slice(self.prompt_width, sequence)
        target[:, completion_slice, 2] = self.completion_mask.float()
        target[:, completion_slice, 3] = self.tool_mask.float()
        target[:, completion_slice, 4] = self.corrected_advantages
        target[:, completion_slice, 5] = self.proximal_logps.float()
        target[:, :, 6] = self.row_credit.float().unsqueeze(1)
        return target

    def trl_inputs(
        self,
        target_microbatch: torch.Tensor,
        *,
        trainer: Any,
        process_group: Any,
        microbatches: int,
    ) -> dict[str, torch.Tensor]:
        """Rebuild the standard TRL DAPO input dict for one final-stage microbatch."""
        validate_trl_contract(
            trainer,
            process_group,
            microbatches,
            expected_process_count=self.process_count,
        )
        if target_microbatch.ndim != 3 or target_microbatch.shape[2] != 7:
            raise PipelineContractError("packed DAPO target must have shape [microbatch,sequence,7]")
        sequence = target_microbatch.shape[1]
        if sequence != self.input_ids.shape[1]:
            raise PipelineContractError("packed target sequence width changed inside one pipeline update")
        ids = target_microbatch[:, :, 0].long()
        attention = target_microbatch[:, :, 1].long()
        row_credit = target_microbatch[:, 0, 6].bool().unsqueeze(1)
        completion = target_microbatch[:, self.prompt_width :, 2].bool() & row_credit
        tool = target_microbatch[:, self.prompt_width :, 3].bool() & row_credit
        old = target_microbatch[:, self.prompt_width :, 5]
        advantages = target_microbatch[:, self.prompt_width :, 4]
        return {
            "prompt_ids": ids[:, : self.prompt_width],
            "prompt_mask": attention[:, : self.prompt_width],
            "completion_ids": ids[:, self.prompt_width :],
            "completion_mask": completion,
            "tool_mask": tool,
            "advantages": advantages,
            "old_per_token_logps": old,
            # Pinned TRL divides num_items_in_batch by process_count. Each PP rank
            # sees the same data, so counteract that divisor to retain the single
            # global effective-token denominator.
            "num_items_in_batch": torch.tensor(
                self.global_policy_tokens * self.process_count,
                dtype=torch.long,
                device=target_microbatch.device,
            ),
        }


@dataclass
class PipelineRankRunner:
    """One rank's upstream PipelineStage and its real 1F1B schedule factory."""

    stage: PipelineStage
    rank: int
    microbatches: int
    process_group: Any = None
    args_chunk_spec: Any = None
    kwargs_chunk_spec: Any = None
    output_merge_spec: Any = None

    @classmethod
    def create(
        cls,
        stage_module: nn.Module,
        *,
        rank: int,
        num_stages: int,
        microbatches: int,
        device: torch.device | str,
        input_args: torch.Tensor | tuple[torch.Tensor, ...],
        output_args: torch.Tensor | tuple[torch.Tensor, ...],
        process_group: Any = None,
        args_chunk_spec: Any = None,
        kwargs_chunk_spec: Any = None,
        output_merge_spec: Any = None,
    ) -> PipelineRankRunner:
        if num_stages != PIPELINE_STAGE_COUNT:
            raise PipelineContractError(f"Bonsai2 PP requires exactly {PIPELINE_STAGE_COUNT} stages")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < num_stages:
            raise PipelineContractError("rank must identify one stage in the four-stage pipeline")
        if isinstance(microbatches, bool) or not isinstance(microbatches, int) or microbatches < 1:
            raise PipelineContractError("microbatches must be a positive integer")
        if microbatches < num_stages:
            raise PipelineContractError("Schedule1F1B requires at least one microbatch per pipeline stage")
        if not isinstance(stage_module, nn.Module):
            raise TypeError("stage_module must be an nn.Module supplied by the model assembler")
        if not dist.is_available() or not dist.is_initialized():
            raise PipelineContractError("initialize torch.distributed before constructing PipelineStage")
        group_size = dist.get_world_size(group=process_group)
        group_rank = dist.get_rank(group=process_group)
        if group_size != num_stages or group_rank != rank:
            raise PipelineContractError(
                f"process group rank/size ({group_rank}/{group_size}) does not match stage ({rank}/{num_stages})"
            )
        for value in (input_args, output_args):
            tensors = value if isinstance(value, tuple) else (value,)
            if not tensors or any(not isinstance(item, torch.Tensor) for item in tensors):
                raise PipelineContractError("stage input_args/output_args must be tensors or non-empty tensor tuples")
        input_args = _pipeline_activation_metadata(input_args)
        stage = PipelineStage(
            stage_module,
            stage_index=rank,
            num_stages=num_stages,
            device=torch.device(device),
            input_args=input_args,
            output_args=_pipeline_activation_metadata(output_args),
            group=process_group,
        )
        return cls(
            stage=stage,
            rank=rank,
            microbatches=microbatches,
            process_group=process_group,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
        )

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.rank == PIPELINE_STAGE_COUNT - 1

    def forward_only(
        self,
        *args: Any,
        return_outputs: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Run the actual distributed schedule without autograd for frozen p scoring."""
        if not self.is_first and (args or kwargs):
            raise PipelineContractError("only stage zero may receive model input tensors")
        schedule = Schedule1F1B(
            self.stage,
            self.microbatches,
            loss_fn=None,
            args_chunk_spec=self.args_chunk_spec if self.is_first else None,
            kwargs_chunk_spec=self.kwargs_chunk_spec if self.is_first else None,
            output_merge_spec=self.output_merge_spec if self.is_last else None,
            scale_grads=False,
        )
        with torch.no_grad():
            return schedule.step(
                *args,
                return_outputs=(return_outputs and self.is_last),
                **kwargs,
            )

    def backward(
        self,
        loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
        *args: Any,
        target: torch.Tensor,
        losses: list[torch.Tensor] | None = None,
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run one 1F1B update; the caller clips and steps once after this returns."""
        if not callable(loss_fn):
            raise TypeError("loss_fn must compute one scalar loss for a final-stage microbatch")
        if not isinstance(target, torch.Tensor) or target.ndim == 0:
            raise PipelineContractError("target must be a batch-first tensor split by the upstream schedule")
        if target.shape[0] % self.microbatches:
            raise PipelineContractError("target batch rows must divide evenly across pipeline microbatches")
        if not self.is_first and (args or kwargs):
            raise PipelineContractError("only stage zero may receive model input tensors")
        if losses is None:
            losses = []
        schedule = Schedule1F1B(
            self.stage,
            self.microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=self.args_chunk_spec if self.is_first else None,
            kwargs_chunk_spec=self.kwargs_chunk_spec if self.is_first else None,
            output_merge_spec=self.output_merge_spec if self.is_last else None,
            scale_grads=False,
        )
        output = schedule.step(
            *args,
            target=target,
            losses=losses,
            return_outputs=(return_outputs and self.is_last),
            **kwargs,
        )
        if self.is_last and len(losses) != self.microbatches:
            raise PipelineContractError(
                f"upstream Schedule1F1B produced {len(losses)} losses for {self.microbatches} microbatches"
            )
        return output


class PipelineUpdateDriver:
    """Connect upstream pipeline backward to a Trainer-owned optimizer/guard.

    ``score_fn`` and ``loss_fn`` are supplied by the Learner integration. The
    loss callback should adapt the final-stage hidden states to the existing
    TRL fused policy-head scorer and call ``LearnerGRPOTrainer._compute_loss``;
    this driver intentionally contains no PPO/DAPO objective or optimizer
    implementation of its own.
    """

    def __init__(self, runner: PipelineRankRunner, trainer: Any, stage_module: nn.Module):
        if not isinstance(runner, PipelineRankRunner):
            raise TypeError("runner must be a PipelineRankRunner")
        if not isinstance(stage_module, nn.Module):
            raise TypeError("stage_module must be the rank-local module whose LoRA parameters are optimized")
        optimizer = getattr(trainer, "optimizer", None)
        scheduler = getattr(trainer, "lr_scheduler", None)
        guard = getattr(trainer, "_gradient_guard", None)
        if not isinstance(optimizer, torch.optim.Optimizer) or guard is None:
            raise PipelineContractError("the existing Trainer must own an optimizer and optimizer guard")
        if scheduler is None or not callable(getattr(scheduler, "step", None)):
            raise PipelineContractError("the existing Trainer must own a learning-rate scheduler")
        trainable_lora_parameters(stage_module)
        self.runner = runner
        self.trainer = trainer
        self.stage_module = stage_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.guard = guard

    def score(self, score_fn: Callable[[torch.Tensor], Any], *model_inputs: Any, **model_kwargs: Any) -> Any:
        """Run a no-grad distributed forward; only the last stage scores targets."""
        if not callable(score_fn):
            raise TypeError("score_fn must score the final-stage hidden states")
        hidden = self.runner.forward_only(*model_inputs, **model_kwargs)
        if not self.runner.is_last:
            return None
        if not isinstance(hidden, torch.Tensor):
            raise PipelineContractError("last stage must return one hidden-state tensor to the scoring callback")
        with torch.no_grad():
            return score_fn(hidden)

    def update(
        self,
        loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
        *model_inputs: Any,
        target: torch.Tensor,
        before_optimizer_step: Callable[[], None] | None = None,
        losses: list[torch.Tensor] | None = None,
        **model_kwargs: Any,
    ) -> dict[str, Any]:
        """Accumulate one 1F1B update, validate/clip globally, then step Trainer state once."""
        validate_trl_contract(
            self.trainer,
            self.runner.process_group,
            self.runner.microbatches,
        )
        if bool(getattr(self.trainer.args, "fp16", False)):
            raise PipelineContractError("the PP Learner requires BF16 or FP32; FP16 GradScaler is unsupported")
        configured_accumulation = int(getattr(self.trainer.args, "gradient_accumulation_steps", self.runner.microbatches))
        if configured_accumulation != self.runner.microbatches:
            raise PipelineContractError(
                "Trainer gradient_accumulation_steps must equal the PipelineSchedule microbatch count"
            )
        self.trainer.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.guard.begin_update(before_optimizer_step)
        losses = [] if losses is None else losses
        loss_context = getattr(self.trainer, "compute_loss_context_manager", None)
        context = loss_context() if callable(loss_context) else nullcontext()
        with context:
            self.runner.backward(
                loss_fn,
                *model_inputs,
                target=target,
                losses=losses,
                return_outputs=False,
                **model_kwargs,
            )
        norm_before_clip = clip_global_lora_grad_norm_(
            self.stage_module,
            float(self.trainer.args.max_grad_norm),
            process_group=self.runner.process_group,
        )
        self.guard.gradient_norm_before_clip = norm_before_clip
        self.guard.on_pre_optimizer_step(self.trainer.args, self.trainer.state, None)
        self.optimizer.step()
        self.scheduler.step()
        self.guard.on_optimizer_step(self.trainer.args, self.trainer.state, None)
        self.trainer.state.global_step = int(self.trainer.state.global_step) + 1
        return {
            "losses": [float(value.detach().float().cpu()) for value in losses] if self.runner.is_last else [],
            "global_gradient_norm_before_clip": norm_before_clip,
            "global_gradient_norm_after_clip": self.guard.gradient_norm_after_clip,
            "optimizer_step_count": int(self.guard.optimizer_step_count),
            "trainer_state": dict(vars(self.trainer.state)),
        }


def clip_global_lora_grad_norm_(
    module: nn.Module,
    max_norm: float,
    *,
    process_group: Any = None,
) -> float:
    """Clip LoRA grads using the norm across all four disjoint pipeline stages."""
    if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)) or not math.isfinite(max_norm) or max_norm <= 0:
        raise PipelineContractError("max_norm must be finite and positive")
    parameters = trainable_lora_parameters(module)
    device = next((parameter.grad.device for parameter in parameters if parameter.grad is not None), parameters[0].device)
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in parameters:
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            total_sq += torch.sum(grad.double().square())
    dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=process_group)
    norm = torch.sqrt(total_sq)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("global pipeline LoRA gradient norm is non-finite")
    norm_value = float(norm.item())
    coefficient = float(max_norm) / (norm_value + 1e-6)
    if coefficient < 1.0:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return norm_value


def capture_pipeline_rank_state(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    rank: int,
    layout_signature: str,
    trainer_state: dict[str, Any],
) -> dict[str, Any]:
    """Capture this stage's LoRA, optimizer, scheduler and complete RNG state."""
    import numpy as np

    if not 0 <= rank < PIPELINE_STAGE_COUNT or not isinstance(layout_signature, str) or not layout_signature:
        raise PipelineContractError("rank and layout_signature must identify the active four-stage layout")
    if not isinstance(trainer_state, dict):
        raise TypeError("trainer_state must be a dictionary")
    trainable_lora_parameters(module)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer owned by this rank")
    named_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and _is_lora_parameter(name)
    }
    stage_device = next(module.parameters()).device
    cuda_rng = (
        torch.cuda.get_rng_state(device=stage_device).cpu()
        if stage_device.type == "cuda"
        else None
    )
    return {
        "rank": rank,
        "num_stages": PIPELINE_STAGE_COUNT,
        "layout_signature": layout_signature,
        "parameter_signature": _stage_parameter_signature(module),
        "adapter_state": named_adapter,
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "scheduler_state": _cpu_tree(scheduler.state_dict()) if scheduler is not None else None,
        "trainer_state": dict(trainer_state),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state().cpu(),
            "torch_cuda": cuda_rng,
        },
    }


def restore_pipeline_rank_state(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: dict[str, Any],
    *,
    rank: int,
    layout_signature: str,
    device: torch.device | str,
) -> dict[str, Any]:
    """Fail closed unless the full stage-local optimizer and RNG can be restored."""
    import numpy as np

    if not isinstance(state, dict):
        raise TypeError("pipeline rank state must be a dictionary")
    if state.get("rank") != rank or state.get("num_stages") != PIPELINE_STAGE_COUNT:
        raise PipelineContractError("checkpoint rank or stage count differs from the active pipeline")
    if state.get("layout_signature") != layout_signature:
        raise PipelineContractError("checkpoint stage layout differs from the active pipeline")
    if state.get("parameter_signature") != _stage_parameter_signature(module):
        raise PipelineContractError("checkpoint local parameter names/shapes/dtypes differ from the active stage")
    target_parameters = {
        name: parameter for name, parameter in module.named_parameters()
        if parameter.requires_grad and _is_lora_parameter(name)
    }
    adapter_state = state.get("adapter_state")
    if not isinstance(adapter_state, dict) or set(adapter_state) != set(target_parameters):
        raise PipelineContractError("checkpoint does not contain the complete local LoRA adapter state")
    with torch.no_grad():
        for name, parameter in target_parameters.items():
            value = adapter_state[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise PipelineContractError(f"checkpoint LoRA tensor {name!r} has the wrong shape")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    optimizer_state = state.get("optimizer_state")
    rng = state.get("rng_state")
    if not isinstance(optimizer_state, dict) or not isinstance(rng, dict):
        raise PipelineContractError("checkpoint is missing local optimizer or RNG state")
    if not all(key in rng for key in ("python", "numpy", "torch_cpu", "torch_cuda")):
        raise PipelineContractError("checkpoint RNG state is incomplete")
    optimizer.load_state_dict(optimizer_state)
    scheduler_state = state.get("scheduler_state")
    if scheduler is None:
        if scheduler_state is not None:
            raise PipelineContractError("checkpoint has scheduler state but active stage has no scheduler")
    elif not isinstance(scheduler_state, dict):
        raise PipelineContractError("checkpoint is missing the local scheduler state")
    else:
        scheduler.load_state_dict(scheduler_state)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    cpu_rng = rng["torch_cpu"]
    if not isinstance(cpu_rng, torch.Tensor):
        raise PipelineContractError("checkpoint is missing the CPU Torch RNG state")
    torch.random.set_rng_state(cpu_rng.cpu())
    device = torch.device(device)
    cuda_rng = rng["torch_cuda"]
    if device.type == "cuda":
        if not isinstance(cuda_rng, torch.Tensor) or not torch.cuda.is_available():
            raise PipelineContractError("CUDA stage checkpoint is missing its CUDA RNG state")
        torch.cuda.set_rng_state(cuda_rng, device=device)
    elif cuda_rng is not None:
        raise PipelineContractError("CPU restore cannot consume a CUDA checkpoint")
    trainer_state = state.get("trainer_state")
    if not isinstance(trainer_state, dict):
        raise PipelineContractError("checkpoint is missing trainer progress state")
    return dict(trainer_state)


def merge_pipeline_adapter_states(
    rank_states: Iterable[dict[str, Any]],
    *,
    expected_names: set[str],
) -> dict[str, torch.Tensor]:
    """Merge disjoint stage-local PEFT tensors before one canonical publication."""
    if not isinstance(expected_names, set) or not expected_names:
        raise PipelineContractError("expected_names must be the complete adapter parameter inventory")
    merged: dict[str, torch.Tensor] = {}
    for rank_state in rank_states:
        adapter = rank_state.get("adapter_state") if isinstance(rank_state, dict) else None
        if not isinstance(adapter, dict):
            raise PipelineContractError("every stage checkpoint must contain adapter_state")
        overlap = set(merged).intersection(adapter)
        if overlap:
            raise PipelineContractError("stage adapter parameter names overlap: " + ", ".join(sorted(overlap)[:8]))
        for name, value in adapter.items():
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise PipelineContractError("adapter state must map stable names to tensors")
            merged[name] = value
    if set(merged) != expected_names:
        missing = sorted(expected_names - set(merged))
        unexpected = sorted(set(merged) - expected_names)
        raise PipelineContractError(f"adapter inventory mismatch; missing={missing[:8]}, unexpected={unexpected[:8]}")
    return merged
