"""Rank-local compute adapter for the frozen four-stage Qwen3.5 PP backend.

This module only prepares the immutable update tensors and binds TRL's existing
fused scorer/loss to ``PipelineUpdateDriver``.  Optimizer state and stepping
remain owned by ``LearnerGRPOTrainer`` and the upstream Torch schedule.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

from .pipeline import (
    PIPELINE_STAGE_COUNT,
    PipelineBatchContext,
    PipelineRankRunner,
    PipelineUpdateDriver,
    capture_pipeline_rank_state,
    causal_policy_token_count,
    restore_pipeline_rank_state,
    trainable_lora_parameters,
)
from .qwen35_pipeline import (
    TRLFusedTernaryHead,
    make_pipeline_proximal_score_fn,
    make_pipeline_trl_dapo_loss_fn,
)


def _setting(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _base_stage(model: Any) -> Any:
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


class PipelineComputeAdapter:
    """Turn validated trajectory rows into one shared PP score/update layout.

    ``process_group`` is the four-rank Torch pipeline group. ``control_group``
    must be the CPU Gloo group used for small object collectives such as the
    frozen proximal-logprob broadcast. The optional factories are narrow CPU
    test seams; production uses the frozen upstream runner and update driver.
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        world_size: int,
        config: Mapping[str, Any] | Any,
        stage_model: torch.nn.Module,
        tokenizer: Any,
        trainer: Any,
        process_group: Any = None,
        *,
        control_group: Any = None,
        runner_factory: Callable[..., Any] | None = None,
        driver_factory: Callable[..., Any] | None = None,
        fused_head: Any = None,
        broadcast_object_list: Callable[..., Any] | None = None,
    ) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < PIPELINE_STAGE_COUNT:
            raise ValueError("rank must identify one of the four pipeline stages")
        if isinstance(local_rank, bool) or not isinstance(local_rank, int) or local_rank < 0:
            raise ValueError("local_rank must be a non-negative integer")
        if world_size != PIPELINE_STAGE_COUNT:
            raise ValueError(f"pipeline world_size must be {PIPELINE_STAGE_COUNT}")
        if not isinstance(stage_model, torch.nn.Module):
            raise TypeError("stage_model must be the rank-local PEFT pipeline module")
        if not callable(getattr(trainer, "_compute_loss", None)):
            raise TypeError("trainer must expose the existing LearnerGRPOTrainer._compute_loss")
        if not callable(getattr(trainer, "_policy_target_mask_scope", None)):
            raise TypeError("trainer must expose the existing policy target mask scope")
        if world_size > 1 and control_group is None and broadcast_object_list is None:
            raise ValueError("the PP adapter requires the existing Gloo control group")

        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.config = config
        self.stage_model = stage_model
        self.base_stage = _base_stage(stage_model)
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.process_group = process_group
        self.control_group = control_group
        self._runner_factory = runner_factory or PipelineRankRunner.create
        self._driver_factory = driver_factory or PipelineUpdateDriver
        self._broadcast_object_list = broadcast_object_list or dist.broadcast_object_list
        self._injected_head = fused_head
        self._runner = None
        self._driver = None
        self._runner_key: tuple[int, ...] | None = None
        self._layout_signature: str | None = None
        self._last_layout_rows: list[dict[str, Any]] = []
        self.last_group_reward_stats: dict[str, Any] = {}

        self.group_size = int(_setting(trainer, "group_size", _setting(config, "group_size", 3)))
        self.groups_per_update = int(
            _setting(trainer, "groups_per_update", _setting(config, "groups_per_update", 2))
        )
        self.microbatch_size = int(
            _setting(trainer, "microbatch_size", _setting(config, "microbatch_size", 1))
        )
        self.importance_sampling_cap = float(
            _setting(trainer, "importance_sampling_cap", _setting(config, "importance_sampling_cap", 3.0))
        )
        self.max_sequence_length = int(
            _setting(config, "max_sequence_length", _setting(getattr(trainer, "args", None), "max_sequence_length", 65536))
        )
        if self.group_size < 2 or self.groups_per_update < 1 or self.microbatch_size < 1:
            raise ValueError("group_size, groups_per_update, and microbatch_size must be positive")
        if not math.isfinite(self.importance_sampling_cap) or self.importance_sampling_cap < 1.0:
            raise ValueError("importance_sampling_cap must be finite and >= 1")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least two")

    @property
    def device(self) -> torch.device:
        try:
            parameter = next(self.stage_model.parameters())
        except StopIteration:
            parameter = None
        if parameter is not None and parameter.device.type != "meta":
            return parameter.device
        return torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    def _pad_token_id(self) -> int:
        value = getattr(self.tokenizer, "pad_token_id", None)
        if value is None:
            value = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= 2**24:
            raise ValueError("tokenizer must provide an integer pad_token_id or eos_token_id below 2^24")
        return value

    def _deterministic_permutation(self, count: int) -> list[int]:
        args = getattr(self.trainer, "args", None)
        state = getattr(self.trainer, "state", None)
        seed = int(_setting(args, "seed", _setting(self.config, "seed", 0)) or 0)
        step = int(_setting(state, "global_step", 0) or 0)
        generator = torch.Generator(device="cpu")
        generator.manual_seed((seed + 1_000_003 * step) % (2**63 - 1))
        return torch.randperm(count, generator=generator).tolist()

    def _group_advantages(self, rows: Sequence[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
        from ..prepared_groups import group_normalized_advantages

        result = group_normalized_advantages(rows)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("group_normalized_advantages must return (advantages, stats)")
        raw_advantages, stats = result
        advantages = [float(value) for value in raw_advantages]
        if len(advantages) != len(rows) or not all(math.isfinite(value) for value in advantages):
            raise ValueError("group_normalized_advantages returned invalid row-aligned values")
        if not isinstance(stats, dict):
            raise TypeError("group_normalized_advantages stats must be a mapping")
        return advantages, dict(stats)

    def _validated_layout_rows(
        self,
        rows: Sequence[dict[str, Any]],
        advantages: Sequence[float] | None = None,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError("pipeline update requires a non-empty validated row list")
        source_rows = [dict(row) for row in rows]
        if advantages is None:
            values, stats = self._group_advantages(source_rows)
            self.last_group_reward_stats = stats
        else:
            values = [float(value) for value in advantages]
            if len(values) != len(source_rows) or not all(math.isfinite(value) for value in values):
                raise ValueError("explicit advantages must be finite and aligned to the input rows")
            self.last_group_reward_stats = {}

        for index, row in enumerate(source_rows):
            ids = row.get("input_ids")
            mask = row.get("policy_mask")
            if not isinstance(ids, (list, tuple)) or not isinstance(mask, (list, tuple)):
                raise TypeError(f"row {index} must include token and policy-mask sequences")
            if len(ids) != len(mask) or len(ids) < 2:
                raise ValueError(f"row {index} token and policy-mask lengths are invalid")
            if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 or token >= 2**24 for token in ids):
                raise ValueError(f"row {index} token IDs must be non-negative integers below 2^24")
            if any(value not in (0, 1, False, True) for value in mask):
                raise ValueError(f"row {index} policy mask must contain only 0/1")
            mask = [int(value) for value in mask]
            try:
                split = mask.index(1)
            except ValueError as exc:
                raise ValueError(f"row {index} has no selected policy token") from exc
            if split == 0 or not any(mask[1:]):
                raise ValueError(f"row {index} has no causal policy token after an initial prompt")
            if len(ids) > self.max_sequence_length:
                raise ValueError(f"row {index} exceeds max_sequence_length")
            row_kind = row.get("row_kind", "fresh_student")
            if row_kind not in {"fresh_student", "native_replay", "guided_replay"}:
                raise ValueError(f"row {index} has unsupported row_kind={row_kind!r}")
            if row_kind == "fresh_student":
                behavior = row.get("behavior_logprobs")
                if not isinstance(behavior, (list, tuple)) or len(behavior) != len(ids):
                    raise ValueError(f"fresh row {index} has invalid behavior logprobs")
                for position, active in enumerate(mask):
                    if active and not math.isfinite(float(behavior[position])):
                        raise ValueError(f"fresh row {index} has non-finite behavior logprobs on policy tokens")
            elif "behavior_logprobs" in row or "policy_version" in row:
                raise ValueError(f"replay row {index} must not contain ordinary behavior logprobs")
            row["input_ids"] = [int(token) for token in ids]
            row["policy_mask"] = mask
            row["row_kind"] = row_kind
            row["_pipeline_split"] = split
            row["_pipeline_advantage"] = values[index]

        permutation = self._deterministic_permutation(len(source_rows))
        ordered_rows = [source_rows[index] for index in permutation]
        ordered_advantages = [values[index] for index in permutation]
        self._last_layout_rows = ordered_rows
        return ordered_rows, ordered_advantages

    def _microbatch_layout(self, row_count: int) -> tuple[int, int]:
        microbatches = max(PIPELINE_STAGE_COUNT, math.ceil(row_count / self.microbatch_size))
        padded_rows = microbatches * self.microbatch_size
        return microbatches, padded_rows

    def _build_context(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        advantages: Sequence[float] | None = None,
    ) -> tuple[PipelineBatchContext, int, int]:
        ordered_rows, ordered_advantages = self._validated_layout_rows(rows, advantages)
        prompt_width = max(int(row["_pipeline_split"]) for row in ordered_rows)
        completion_width = max(len(row["input_ids"]) - int(row["_pipeline_split"]) for row in ordered_rows)
        sequence_length = prompt_width + completion_width
        if sequence_length <= prompt_width:
            raise ValueError("pipeline rows must leave a non-empty completion")
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"left/right padding expands pipeline sequence to {sequence_length}, "
                f"above max_sequence_length={self.max_sequence_length}"
            )

        microbatches, padded_rows = self._microbatch_layout(len(ordered_rows))
        pad_id = self._pad_token_id()
        device = self.device
        input_ids = torch.full((padded_rows, sequence_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((padded_rows, sequence_length), dtype=torch.bool)
        completion_mask = torch.zeros((padded_rows, completion_width), dtype=torch.bool)
        tool_mask = torch.zeros((padded_rows, completion_width), dtype=torch.bool)
        behavior_logps = torch.full((padded_rows, completion_width), float("nan"), dtype=torch.float32)
        advantage_tensor = torch.zeros((padded_rows,), dtype=torch.float32)
        fresh_row_mask = torch.zeros((padded_rows,), dtype=torch.bool)
        row_credit = torch.zeros((padded_rows,), dtype=torch.bool)

        # Extra schedule slots repeat a valid row's activations but carry no loss
        # credit. This keeps attention numerically valid while preserving the
        # full-update token denominator.
        layout_indices = list(range(len(ordered_rows)))
        if padded_rows > len(ordered_rows):
            layout_indices.extend([0] * (padded_rows - len(ordered_rows)))
        for output_index, source_index in enumerate(layout_indices):
            row = ordered_rows[source_index]
            ids = row["input_ids"]
            split = int(row["_pipeline_split"])
            prompt = ids[:split]
            completion = ids[split:]
            prompt_start = prompt_width - len(prompt)
            input_ids[output_index, prompt_start:prompt_width] = torch.tensor(prompt, dtype=torch.long)
            input_ids[output_index, prompt_width:prompt_width + len(completion)] = torch.tensor(completion, dtype=torch.long)
            attention_mask[output_index, prompt_start:prompt_width + len(completion)] = True
            completion_mask[output_index, :len(completion)] = True
            is_real_row = output_index < len(ordered_rows)
            if is_real_row:
                policy = row["policy_mask"][split:]
                tool_mask[output_index, :len(policy)] = torch.tensor(policy, dtype=torch.bool)
                advantage_tensor[output_index] = ordered_advantages[source_index]
                row_credit[output_index] = True
                is_fresh = row["row_kind"] == "fresh_student"
                fresh_row_mask[output_index] = is_fresh
                if is_fresh:
                    behavior = [float(value) for value in row["behavior_logprobs"][split:]]
                    behavior_logps[output_index, :len(behavior)] = torch.tensor(behavior, dtype=torch.float32)

        text_positions = attention_mask.long().cumsum(dim=-1) - 1
        text_positions.masked_fill_(~attention_mask, 0)
        position_ids = text_positions.unsqueeze(0).expand(4, -1, -1).clone()
        full_policy_mask = torch.zeros((padded_rows, sequence_length), dtype=torch.bool)
        full_policy_mask[:, prompt_width:] = tool_mask
        global_policy_tokens = causal_policy_token_count(full_policy_mask, row_credit)

        context = PipelineBatchContext(
            input_ids=input_ids.to(device=device),
            attention_mask=attention_mask.to(device=device),
            position_ids=position_ids.to(device=device),
            prompt_width=prompt_width,
            completion_mask=completion_mask.to(device=device),
            tool_mask=tool_mask.to(device=device),
            advantages=advantage_tensor.to(device=device),
            proximal_logps=torch.zeros((padded_rows, completion_width), dtype=torch.float32, device=device),
            behavior_logps=behavior_logps.to(device=device),
            fresh_row_mask=fresh_row_mask.to(device=device),
            row_credit=row_credit.to(device=device),
            global_policy_tokens=global_policy_tokens,
            process_count=self.world_size,
            importance_sampling_cap=self.importance_sampling_cap,
        )
        return context, microbatches, len(ordered_rows)

    def _head(self) -> Any:
        if self._injected_head is not None:
            return self._injected_head
        if self.rank != self.world_size - 1:
            return None
        existing = getattr(self.base_stage, "_pipeline_trl_fused_head", None)
        if existing is None:
            existing = getattr(self.base_stage, "_ternary_fused_head_adapter", None)
        if existing is not None:
            self._injected_head = existing
            return existing
        output_embeddings = getattr(self.base_stage, "get_output_embeddings", None)
        if not callable(output_embeddings):
            raise TypeError("last pipeline stage must expose its frozen T2 output head")
        config = getattr(self.base_stage, "config", getattr(self.stage_model, "config", None))
        if config is None:
            raise TypeError("last pipeline stage must expose its Qwen3.5 config")
        head = TRLFusedTernaryHead(config, output_embeddings())
        self.base_stage._pipeline_trl_fused_head = head
        self._injected_head = head
        return head

    def _ensure_driver(self, context: PipelineBatchContext, microbatches: int) -> Any:
        key = (*context.shape_signature, microbatches)
        if self._driver is not None and self._runner_key == key:
            return self._driver
        microbatch_rows = int(context.input_ids.shape[0]) // microbatches
        input_args_fn = getattr(self.base_stage, "pipeline_input_args", None)
        output_args_fn = getattr(self.base_stage, "pipeline_output_args", None)
        chunk_specs_fn = getattr(self.base_stage, "input_chunk_specs", None)
        if not all(callable(value) for value in (input_args_fn, output_args_fn, chunk_specs_fn)):
            raise TypeError("Qwen stage must expose pipeline_input_args/output_args/input_chunk_specs")
        runner = self._runner_factory(
            self.stage_model,
            rank=self.rank,
            num_stages=self.world_size,
            microbatches=microbatches,
            device=self.device,
            input_args=input_args_fn(microbatch_rows, int(context.input_ids.shape[1])),
            output_args=output_args_fn(microbatch_rows, int(context.input_ids.shape[1])),
            process_group=self.process_group,
            args_chunk_spec=chunk_specs_fn(),
        )
        self._runner = runner
        self._runner_key = key
        self._driver = self._driver_factory(runner, self.trainer, self.stage_model)
        return self._driver

    def _model_inputs(self, context: PipelineBatchContext) -> tuple[torch.Tensor, ...]:
        return context.pipeline_model_inputs() if self.rank == 0 else ()

    @contextmanager
    def _last_stage_metric_collectives(self):
        """Keep TRL's metric reductions local to the final PP stage.

        The upstream DAPO objective is called only by the last pipeline stage.
        Its post-loss ``gather``/``reduce`` calls therefore cannot be distributed
        PP collectives. That stage already owns each loss microbatch's complete
        metric tensors, while ``num_processes`` remains four for TRL's loss
        denominator contract.
        """
        accelerator = getattr(self.trainer, "accelerator", None)
        if accelerator is None:
            yield
            return
        names = ("gather", "reduce", "gather_for_metrics")
        originals: dict[str, Any] = {}

        def local_metric(value: Any, *args: Any, **kwargs: Any) -> Any:
            return value

        try:
            for name in names:
                method = getattr(accelerator, name, None)
                if callable(method):
                    originals[name] = method
                    setattr(accelerator, name, local_metric)
            yield
        finally:
            for name, method in originals.items():
                setattr(accelerator, name, method)

    def _broadcast_proximal(self, value: torch.Tensor | None, context: PipelineBatchContext) -> torch.Tensor:
        source = self.world_size - 1
        box: list[Any] = [value.detach().to(device="cpu", dtype=torch.float32).clone() if self.rank == source and value is not None else None]
        self._broadcast_object_list(box, src=source, group=self.control_group)
        result = box[0]
        if not isinstance(result, torch.Tensor):
            raise TypeError("last PP stage did not broadcast a proximal-logprob tensor")
        if tuple(result.shape) != tuple(context.proximal_logps.shape):
            raise RuntimeError("broadcast proximal-logprob shape differs from the shared completion layout")
        result = result.to(device=context.input_ids.device, dtype=torch.float32)
        active = context.policy_mask
        if not bool(torch.isfinite(result[active]).all()):
            raise ValueError("frozen proximal logprobs are non-finite on selected policy tokens")
        return torch.where(active, result, torch.zeros_like(result)).detach()

    def _score_context(
        self,
        context: PipelineBatchContext,
        microbatches: int,
    ) -> tuple[PipelineBatchContext, Any]:
        driver = self._ensure_driver(context, microbatches)
        head = self._head()
        score_fn = make_pipeline_proximal_score_fn(
            self.trainer,
            head,
            input_ids=context.input_ids,
            attention_mask=context.attention_mask,
            completion_mask=context.completion_mask,
            tool_mask=context.tool_mask,
        )
        was_training = bool(self.stage_model.training)
        self.stage_model.eval()
        try:
            with context.bind_pipeline_context(self.stage_model):
                scored = driver.score(score_fn, *self._model_inputs(context))
        finally:
            self.stage_model.train(was_training)
        proximal = self._broadcast_proximal(scored, context)
        scored_context = PipelineBatchContext(
                input_ids=context.input_ids,
                attention_mask=context.attention_mask,
                position_ids=context.position_ids,
                prompt_width=context.prompt_width,
                completion_mask=context.completion_mask,
                tool_mask=context.tool_mask,
                advantages=context.advantages,
                proximal_logps=proximal,
                behavior_logps=context.behavior_logps,
                fresh_row_mask=context.fresh_row_mask,
                row_credit=context.row_credit,
                global_policy_tokens=context.global_policy_tokens,
                process_count=context.process_count,
                importance_sampling_cap=context.importance_sampling_cap,
            )
        return scored_context, driver

    def _correction_metrics(self, context: PipelineBatchContext) -> dict[str, Any]:
        policy = context.policy_mask
        fresh = policy & context.fresh_row_mask.unsqueeze(1)
        fresh_tokens = int(fresh.sum().item())
        replay_tokens = int((policy & ~context.fresh_row_mask.unsqueeze(1)).sum().item())
        if fresh_tokens <= 0:
            raise ValueError("PP update requires at least one fresh behavior token")

        deltas = (context.proximal_logps.float() - context.behavior_logps.float())[fresh]
        if not bool(torch.isfinite(deltas).all()):
            raise ValueError("p-b is non-finite on fresh selected policy tokens")
        cap_log = math.log(self.importance_sampling_cap)
        weights = torch.exp(deltas.clamp(max=cap_log)).detach()
        quantiles = torch.quantile(deltas, torch.tensor([0.0, 0.5, 0.95, 1.0], device=deltas.device))
        absolute = deltas.abs()

        kind_counts = {"fresh_student": 0, "native_replay": 0, "guided_replay": 0}
        by_worker_values: dict[str, list[float]] = {}
        alignment_rows: list[tuple[float, str | None, int, float, float, int, str]] = []
        eligible_completion_targets = int(context.completion_mask[context.row_credit].sum().item())
        for row_index, row in enumerate(self._last_layout_rows):
            split = int(row["_pipeline_split"])
            selected_positions = [
                offset for offset, selected in enumerate(row["policy_mask"][split:]) if selected
            ]
            kind_counts[row["row_kind"]] += len(selected_positions)
            if row["row_kind"] != "fresh_student":
                continue
            worker_id = row.get("worker_id")
            if not isinstance(worker_id, str) or not worker_id:
                worker_id = None
            behavior = row["behavior_logprobs"]
            for offset in selected_positions:
                behavior_value = float(behavior[split + offset])
                proximal_value = float(context.proximal_logps[row_index, offset].detach().cpu())
                error = abs(proximal_value - behavior_value)
                if worker_id is not None:
                    by_worker_values.setdefault(worker_id, []).append(error)
                alignment_rows.append((
                    error,
                    worker_id,
                    split + offset,
                    behavior_value,
                    proximal_value,
                    row_index,
                    str(row.get("group_id", "")),
                ))

        def alignment_stats(errors: list[float]) -> dict[str, Any]:
            if not errors:
                return {"tokens": 0, "mean_abs_error": 0.0, "max_abs_error": 0.0, "rms_error": 0.0}
            return {
                "tokens": len(errors),
                "mean_abs_error": sum(errors) / len(errors),
                "max_abs_error": max(errors),
                "rms_error": math.sqrt(sum(value * value for value in errors) / len(errors)),
            }

        worst = max(alignment_rows, key=lambda item: item[0])
        policy_alignment = {
            "source": "proximal_scores_before_optimizer",
            "tokens": fresh_tokens,
            "mean_abs_error": float(absolute.mean().detach().cpu()),
            "max_abs_error": float(absolute.max().detach().cpu()),
            "rms_error": float(torch.sqrt(torch.mean(deltas.square())).detach().cpu()),
            "by_worker": {
                worker_id: alignment_stats(errors)
                for worker_id, errors in sorted(by_worker_values.items())
            },
            "worst_token": {
                "worker_id": worst[1],
                "input_position": worst[2],
                "behavior_logprob": worst[3],
                "learner_logprob": worst[4],
                "absolute_error": worst[0],
                "trajectory_index": worst[5],
                "group_id": worst[6],
            },
        }
        return {
            "mode": "token_truncate",
            "cap": self.importance_sampling_cap,
            "source": "actual_policy_targets_only",
            "target_scoring_strategy": "actual_policy_targets_only",
            "policy_tokens": context.global_policy_tokens,
            "fresh_policy_tokens": fresh_tokens,
            "replay_policy_tokens": replay_tokens,
            "policy_tokens_by_kind": kind_counts,
            "replay_correction": 1.0,
            "approximate_correction": False,
            "log_ratio_min": float(quantiles[0].detach().cpu()),
            "log_ratio_median": float(quantiles[1].detach().cpu()),
            "log_ratio_p95": float(quantiles[2].detach().cpu()),
            "log_ratio_max": float(quantiles[3].detach().cpu()),
            "weight_min": float(weights.min().detach().cpu()),
            "weight_mean": float(weights.mean().detach().cpu()),
            "weight_max": float(weights.max().detach().cpu()),
            "truncated_fraction": float((deltas > cap_log).float().mean().detach().cpu()),
            "eligible_completion_targets": eligible_completion_targets,
            "policy_targets_projected": context.global_policy_tokens,
            "non_policy_completion_targets_skipped": eligible_completion_targets - context.global_policy_tokens,
            "policy_alignment": policy_alignment,
        }

    def score_rows(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Collectively score a validated batch and return its frozen p tensor."""
        context, microbatches, source_rows = self._build_context(rows)
        context, _driver = self._score_context(context, microbatches)
        selected = context.policy_mask
        return {
            "rank": self.rank,
            "source_rows": source_rows,
            "padded_rows": int(context.input_ids.shape[0]),
            "microbatches": microbatches,
            "global_policy_tokens": context.global_policy_tokens,
            "finite_logprobs": bool(torch.isfinite(context.proximal_logps[selected]).all()),
            "proximal_logps": context.proximal_logps.detach().float().cpu(),
            "layout_rows": [dict(row) for row in self._last_layout_rows],
        }

    def update_rows(
        self,
        rows: Sequence[dict[str, Any]],
        before_optimizer_step: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Freeze p in the final layout, then run the upstream 1F1B/TRL update."""
        if before_optimizer_step is not None and not callable(before_optimizer_step):
            raise TypeError("before_optimizer_step must be callable")
        context, microbatches, source_rows = self._build_context(rows)
        context, driver = self._score_context(context, microbatches)
        loss_fn = make_pipeline_trl_dapo_loss_fn(
            self.trainer,
            self._head(),
            context,
            process_group=self.process_group,
            microbatches=microbatches,
        )
        args = getattr(self.trainer, "args", None)
        if args is None:
            raise TypeError("LearnerGRPOTrainer must expose TrainingArguments")
        previous = {
            name: getattr(args, name, None)
            for name in ("gradient_accumulation_steps", "steps_per_generation")
        }
        trainer_step_fields = ("current_gradient_accumulation_steps", "steps_per_generation")
        previous_trainer_steps = {
            name: getattr(self.trainer, name, None)
            for name in trainer_step_fields
        }
        for name in previous:
            setattr(args, name, microbatches)
        for name in trainer_step_fields:
            if hasattr(self.trainer, name):
                setattr(self.trainer, name, microbatches)
        try:
            with self._last_stage_metric_collectives(), context.bind_pipeline_context(self.stage_model):
                update = driver.update(
                    loss_fn,
                    *self._model_inputs(context),
                    target=context.packed_target(),
                    before_optimizer_step=before_optimizer_step,
                )
        finally:
            for name, value in previous.items():
                if value is not None:
                    setattr(args, name, value)
            for name, value in previous_trainer_steps.items():
                if value is not None:
                    setattr(self.trainer, name, value)
        correction = self._correction_metrics(context)
        clip = {
            "global_gradient_norm_before_clip": update.get("global_gradient_norm_before_clip"),
            "global_gradient_norm_after_clip": update.get("global_gradient_norm_after_clip"),
        }
        if hasattr(self.trainer, "_last_correction_metrics"):
            self.trainer._last_correction_metrics = dict(correction)
        if hasattr(self.trainer, "_last_upstream_clip_metrics"):
            self.trainer._last_upstream_clip_metrics = dict(clip)
        if hasattr(self.trainer, "_last_actual_gradient_accumulation_steps"):
            self.trainer._last_actual_gradient_accumulation_steps = microbatches
        losses = update.get("losses", [])
        scalar_losses = [float(value) for value in losses if isinstance(value, (int, float)) and math.isfinite(float(value))]
        metrics = {
            "loss": sum(scalar_losses),
            "loss_metric_scope": "sum_of_global_denominator_microbatch_losses",
        } if scalar_losses else {}
        return {
            "updated": True,
            "rank": self.rank,
            "source_rows": source_rows,
            "padded_rows": int(context.input_ids.shape[0]),
            "microbatches": microbatches,
            "global_policy_tokens": context.global_policy_tokens,
            "group_reward_stats": dict(self.last_group_reward_stats),
            "correction": correction,
            "metrics": metrics,
            "quality": {"available": False},
            "telemetry": {"metric_scope": "global_batch_on_last_pipeline_stage"},
            "clip": clip,
            "update": update,
        }

    def probe(self, lengths: list[int]) -> dict[str, Any]:
        """Run a collective real PP forward and fused selected-token score probe."""
        if not isinstance(lengths, list) or not lengths or any(
            isinstance(length, bool) or not isinstance(length, int) or length < 2
            for length in lengths
        ):
            raise ValueError("probe lengths must be a non-empty list of integers >= 2")
        if max(lengths) > self.max_sequence_length:
            raise ValueError("probe length exceeds configured max_sequence_length")
        vocab_size = int(
            _setting(
                getattr(self.base_stage, "config", None),
                "vocab_size",
                getattr(self.tokenizer, "vocab_size", 0),
            )
        )
        if vocab_size < 2 or vocab_size >= 2**24:
            raise ValueError("pipeline probe requires a usable vocabulary below 2^24 tokens")
        rows = []
        for index, length in enumerate(lengths):
            rows.append({
                "input_ids": [position % vocab_size for position in range(length)],
                "policy_mask": [0, *([1] * (length - 1))],
                "row_kind": "fresh_student",
                "behavior_logprobs": [0.0] * length,
                "policy_version": 0,
                "group_id": f"probe-{index}",
                "reward": 0.0,
            })
        started = time.perf_counter()
        context, microbatches, source_rows = self._build_context(rows, advantages=[0.0] * len(rows))
        context, _driver = self._score_context(context, microbatches)
        selected = context.policy_mask
        finite = bool(torch.isfinite(context.proximal_logps[selected]).all())
        return {
            "status": "passed" if finite else "failed",
            "rank": self.rank,
            "lengths": list(lengths),
            "selected_policy_tokens": context.global_policy_tokens,
            "finite_logprobs": finite,
            "seconds": time.perf_counter() - started,
            "source_rows": source_rows,
            "microbatches": microbatches,
        }

    def local_parameter_layout(self) -> list[dict[str, Any]]:
        trainable_lora_parameters(self.stage_model)
        return [
            {
                "order": order,
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
            }
            for order, (name, parameter) in enumerate(self.stage_model.named_parameters())
            if parameter.requires_grad
        ]

    def peft_config(self) -> dict[str, Any]:
        configs = getattr(self.stage_model, "peft_config", None)
        if not isinstance(configs, dict) or not configs:
            raise RuntimeError("pipeline stage has no local PEFT config")
        return {
            name: config.to_dict() if callable(getattr(config, "to_dict", None)) else dict(config)
            for name, config in configs.items()
        }

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_lora_parameters(self.stage_model)
        return {
            name: parameter.detach().to(device="cpu", copy=True)
            for name, parameter in self.stage_model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        }

    def set_layout_signature(self, layout_signature: str) -> None:
        if not isinstance(layout_signature, str) or not layout_signature.strip():
            raise ValueError("layout_signature must be a non-empty string")
        self._layout_signature = layout_signature

    def capture_state(self, layout_signature: str | None = None) -> dict[str, Any]:
        layout_signature = layout_signature or self._layout_signature
        if layout_signature is None:
            raise RuntimeError("set the collective PP layout signature before capturing rank state")
        optimizer = getattr(self.trainer, "optimizer", None)
        scheduler = getattr(self.trainer, "lr_scheduler", None)
        state = getattr(self.trainer, "state", None)
        if optimizer is None or state is None:
            raise RuntimeError("Trainer optimizer and progress state are required for PP snapshots")
        return capture_pipeline_rank_state(
            self.stage_model,
            optimizer,
            scheduler,
            rank=self.rank,
            layout_signature=layout_signature,
            trainer_state=dict(vars(state)),
        )

    def restore_state(
        self,
        snapshot: dict[str, Any],
        layout_signature: str | None = None,
    ) -> dict[str, Any]:
        layout_signature = layout_signature or self._layout_signature
        if layout_signature is None:
            raise RuntimeError("set the collective PP layout signature before restoring rank state")
        optimizer = getattr(self.trainer, "optimizer", None)
        scheduler = getattr(self.trainer, "lr_scheduler", None)
        if optimizer is None:
            raise RuntimeError("Trainer optimizer is required for PP restore")
        restored = restore_pipeline_rank_state(
            self.stage_model,
            optimizer,
            scheduler,
            snapshot,
            rank=self.rank,
            layout_signature=layout_signature,
            device=self.device,
        )
        state = getattr(self.trainer, "state", None)
        if state is None:
            raise RuntimeError("Trainer progress state is required for PP restore")
        for name, value in restored.items():
            setattr(state, name, value)
        return restored


__all__ = ["PipelineComputeAdapter"]
