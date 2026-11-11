"""Shared proximal-logprob accumulation for eager and in-forward scoring."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch


class ProximalAccumulator:
    """Validate one or more scored microbatches and build detached corrections.

    ``trainer`` is deliberately duck-typed: this module is shared by the eager
    scorer and the first-gradient-forward collector without importing the
    trainer module (which would create a circular import).
    """

    def __init__(
        self,
        trainer: Any,
        *,
        row_kind_to_id: Mapping[str, int],
        id_to_row_kind: Mapping[int, str],
    ) -> None:
        self.trainer = trainer
        self.row_kind_to_id = row_kind_to_id
        self.id_to_row_kind = id_to_row_kind
        if (
            trainer.loss_type != "dapo"
            or trainer.beta != 0
            or trainer.use_vllm
            or trainer.importance_sampling_level != "token"
            or trainer.aux_loss_enabled
            or trainer._entropy_bonus_enabled
            or trainer.off_policy_mask_threshold is not None
            or trainer.top_entropy_quantile != 1.0
            or trainer.args.delta is not None
        ):
            raise ValueError("proximal correction requires unmodified token-level beta=0 DAPO")
        if any(
            isinstance(module, torch.nn.Dropout) and module.p != 0
            for module in trainer.model.modules()
        ):
            raise ValueError("proximal and training forwards require disabled dropout")
        if trainer._active_rows is None:
            raise RuntimeError("cannot freeze proximal logprobs without staged BEPA rows")

        self.raw_deltas: list[torch.Tensor] = []
        self.weights: list[torch.Tensor] = []
        self.alignment_errors_by_worker: dict[str, list[float]] = defaultdict(list)
        self.alignment_records: list[tuple[float, str | None, int, float, float, int, str]] = []
        self.policy_tokens_by_kind = {
            kind: 0 for kind in row_kind_to_id if kind != "zero_credit"
        }
        if any(row.get("row_kind") == "zero_credit" for row in trainer._active_rows):
            self.policy_tokens_by_kind["zero_credit"] = 0
        self.seen_row_ids: set[int] = set()
        self.finished = False
        self._finish_attempted = False
        self.batch_count = 0

    @torch.no_grad()
    def add(
        self,
        batch: dict[str, Any],
        proximal: torch.Tensor,
        *,
        row_metadata: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Validate one final-layout microbatch and attach its detached inputs."""
        if self.finished:
            raise RuntimeError("cannot add a proximal microbatch after finish()")
        trainer = self.trainer
        index = self.batch_count
        if trainer._inline_proximal_batch is None and trainer.failure_stage != f"proximal_microbatch_{index}":
            trainer.failure_stage = f"proximal_microbatch_{index}"
        row_ids_for_log = batch.get("bepa_row_ids")
        trainer._report_parallel_phase(
            "proximal_microbatch_started",
            rank=trainer.parallel_rank,
            microbatch=index,
            microbatches=trainer._last_actual_gradient_accumulation_steps,
            row_ids=(row_ids_for_log.detach().cpu().tolist() if isinstance(row_ids_for_log, torch.Tensor) else None),
            input_shape=(list(batch["prompt_ids"].shape) if isinstance(batch.get("prompt_ids"), torch.Tensor) else None),
            completion_shape=(list(batch["completion_ids"].shape) if isinstance(batch.get("completion_ids"), torch.Tensor) else None),
        )
        mask = trainer._completion_policy_mask(batch)
        row_ids = batch.get("bepa_row_ids")
        row_kind_ids = batch.get("bepa_row_kind_ids")
        if (
            not isinstance(row_ids, torch.Tensor)
            or row_ids.ndim != 1
            or not isinstance(row_kind_ids, torch.Tensor)
            or row_kind_ids.ndim != 1
            or row_ids.shape[0] != mask.shape[0]
            or row_kind_ids.shape[0] != mask.shape[0]
        ):
            raise RuntimeError("TRL dropped BEPA row identity before proximal scoring")
        if row_metadata is None:
            row_metadata = torch.stack([row_ids, row_kind_ids], dim=1).detach().cpu().tolist()
        if len(row_metadata) != mask.shape[0]:
            raise RuntimeError("TRL proximal row metadata count differs from the microbatch row count")
        if proximal.shape != mask.shape:
            raise ValueError("proximal logprob and policy mask shapes differ")
        if not bool(torch.isfinite(proximal[mask]).all()):
            raise ValueError("non-finite proximal logprob")
        trainer._report_parallel_phase(
            "proximal_microbatch_completed",
            rank=trainer.parallel_rank,
            microbatch=index,
            row_ids=(row_ids_for_log.detach().cpu().tolist() if isinstance(row_ids_for_log, torch.Tensor) else None),
        )

        proximal = torch.where(mask, proximal.float(), 0.0).detach()
        correction = torch.zeros_like(proximal, dtype=torch.float32)
        batch_input_tokens = 0
        batch_policy_tokens = 0
        for row_index, (row_id_value, row_kind_value) in enumerate(row_metadata):
            row_id = int(row_id_value)
            row_kind = self.id_to_row_kind.get(int(row_kind_value))
            if row_id < 0 or row_id >= len(trainer._active_rows):
                raise RuntimeError(f"TRL returned unknown BEPA row_id={row_id}")
            if row_id in self.seen_row_ids:
                raise RuntimeError(f"TRL duplicated BEPA row_id={row_id} across microbatches")
            self.seen_row_ids.add(row_id)
            staged_row = trainer._active_rows[row_id]
            batch_input_tokens += len(staged_row["input_ids"])
            if staged_row.get("row_kind") != "zero_credit":
                batch_policy_tokens += sum(staged_row["policy_mask"][1:])
            expected_kind = staged_row.get("row_kind", "fresh_student")
            if row_kind != expected_kind:
                raise RuntimeError(
                    f"TRL BEPA row metadata shifted: row_id={row_id} kind={row_kind!r}, expected {expected_kind!r}"
                )

            expected_split = staged_row["policy_mask"].index(1)
            completion_mask = batch["completion_mask"][row_index].bool()
            source_policy_mask = staged_row["policy_mask"][expected_split:]
            actual_policy_mask = batch.get("tool_mask")
            if actual_policy_mask is None:
                actual_policy_mask_row = completion_mask
            else:
                actual_policy_mask_row = completion_mask & actual_policy_mask[row_index].bool()
            if expected_kind == "zero_credit":
                expected_policy_mask = torch.zeros_like(actual_policy_mask_row, dtype=torch.bool)
            else:
                expected_policy_mask = torch.tensor(
                    source_policy_mask, dtype=torch.bool, device=actual_policy_mask_row.device
                )
            if expected_policy_mask.shape[0] > actual_policy_mask_row.shape[0]:
                raise RuntimeError(f"TRL policy mask is shorter than staged BEPA row_id={row_id}")
            if (
                not torch.equal(actual_policy_mask_row[: expected_policy_mask.shape[0]], expected_policy_mask)
                or bool(actual_policy_mask_row[expected_policy_mask.shape[0] :].any())
            ):
                raise RuntimeError(f"TRL policy mask no longer aligns with staged BEPA row_id={row_id}")

            if expected_kind == "fresh_student":
                behavior = trainer._fresh_behavior_by_row.get(row_id)
                if behavior is None:
                    raise RuntimeError(f"fresh BEPA row_id={row_id} has no immutable behavior logprobs")
                if len(behavior) != len(staged_row["input_ids"]):
                    raise RuntimeError(f"fresh BEPA row_id={row_id} behavior logprobs have wrong length")
                b_values = torch.tensor(
                    behavior[expected_split:], dtype=torch.float32, device=proximal.device
                )
                if not bool(torch.isfinite(b_values[expected_policy_mask]).all()):
                    raise ValueError(f"fresh BEPA row_id={row_id} has non-finite behavior logprobs")
                delta = proximal[row_index, : b_values.shape[0]] - b_values
                if not bool(torch.isfinite(delta[expected_policy_mask]).all()):
                    raise ValueError(f"fresh BEPA row_id={row_id} has non-finite proximal/behavior delta")
                cap = trainer.importance_sampling_cap
                fresh_weights = torch.exp(
                    delta[expected_policy_mask].clamp(max=math.log(cap))
                ).detach()
                correction[row_index, : b_values.shape[0]][expected_policy_mask] = fresh_weights
                self.raw_deltas.append(delta[expected_policy_mask].detach().cpu())
                self.weights.append(fresh_weights.cpu())
                active_offsets = torch.nonzero(expected_policy_mask, as_tuple=False).flatten().cpu().tolist()
                behavior_values = b_values.detach().float().cpu()
                proximal_values = proximal[row_index, : b_values.shape[0]].detach().float().cpu()
                worker_id = staged_row.get("worker_id")
                if not isinstance(worker_id, str) or not worker_id:
                    worker_id = None
                for offset in active_offsets:
                    behavior_value = float(behavior_values[offset])
                    proximal_value = float(proximal_values[offset])
                    error = abs(proximal_value - behavior_value)
                    if worker_id is not None:
                        self.alignment_errors_by_worker[worker_id].append(error)
                    self.alignment_records.append(
                        (
                            error,
                            worker_id,
                            expected_split + offset,
                            behavior_value,
                            proximal_value,
                            row_id,
                            str(staged_row["group_id"]),
                        )
                    )
                self.policy_tokens_by_kind[expected_kind] += sum(source_policy_mask)
            elif expected_kind in {"native_replay", "guided_replay"}:
                correction[row_index, actual_policy_mask_row] = 1.0
                self.policy_tokens_by_kind[expected_kind] += sum(source_policy_mask)

        correction = correction.detach()
        advantage = batch["advantages"]
        if advantage.ndim != 1 or advantage.shape[0] != mask.shape[0]:
            raise ValueError("upstream must provide one original advantage per trajectory")
        batch["monitor_input_tokens"] = batch_input_tokens
        batch["monitor_policy_tokens"] = batch_policy_tokens
        batch["old_per_token_logps"] = proximal
        batch["advantages"] = advantage[:, None] * correction
        # These are correction weights only; no replay behavior probability is fabricated.
        batch["behavior_correction"] = correction
        self.batch_count += 1

    @torch.no_grad()
    def finish(self) -> None:
        """Require complete row coverage and publish aggregate alignment metrics once."""
        if self._finish_attempted:
            raise RuntimeError("ProximalAccumulator.finish() may only be called once")
        self._finish_attempted = True
        trainer = self.trainer
        if self.seen_row_ids != set(range(len(trainer._active_rows or []))):
            missing = sorted(set(range(len(trainer._active_rows or []))) - self.seen_row_ids)
            raise RuntimeError(f"TRL omitted BEPA rows before proximal scoring: {missing}")
        fresh_policy_tokens = self.policy_tokens_by_kind["fresh_student"]
        replay_policy_tokens = (
            self.policy_tokens_by_kind["native_replay"] + self.policy_tokens_by_kind["guided_replay"]
        )
        if fresh_policy_tokens <= 0 or not self.raw_deltas:
            valid_partition_replay = (getattr(trainer, "gradient_only", False)
                                      and getattr(trainer, "_gradient_partition_context", False)
                                      and replay_policy_tokens > 0)
            if trainer.parallel_mode not in {"ddp", "fsdp2"} and not valid_partition_replay:
                raise ValueError("proximal batch has no fresh behavior tokens")
            trainer._last_correction_metrics = {
                "mode": "token_truncate",
                "cap": trainer.importance_sampling_cap,
                "target_scoring_strategy": "actual_policy_targets_only",
                **trainer._last_scoring_metrics,
                "policy_tokens": 0,
                "fresh_policy_tokens": 0,
                "replay_policy_tokens": self.policy_tokens_by_kind["native_replay"] + self.policy_tokens_by_kind["guided_replay"],
                "policy_tokens_by_kind": dict(self.policy_tokens_by_kind),
                "replay_correction": 1.0,
                "approximate_correction": True,
            }
            self.finished = True
            return
        deltas = torch.cat(self.raw_deltas)
        values = torch.cat(self.weights)
        if not deltas.numel():
            raise ValueError("proximal batch has no policy tokens")
        quantiles = torch.quantile(deltas, torch.tensor([0.0, 0.5, 0.95, 1.0]))
        absolute_deltas = deltas.abs()

        def alignment_stats(errors: list[float]) -> dict[str, int | float]:
            if not errors:
                return {"tokens": 0, "mean_abs_error": 0.0, "max_abs_error": 0.0, "rms_error": 0.0}
            return {
                "tokens": len(errors),
                "mean_abs_error": sum(errors) / len(errors),
                "max_abs_error": max(errors),
                "rms_error": math.sqrt(sum(value * value for value in errors) / len(errors)),
            }

        alignment_by_worker = {
            worker_id: alignment_stats(errors)
            for worker_id, errors in sorted(self.alignment_errors_by_worker.items())
        }
        worst_alignment = max(self.alignment_records, key=lambda item: item[0])
        policy_alignment = {
            "source": "proximal_scores_before_optimizer",
            "tokens": int(deltas.numel()),
            "mean_abs_error": float(absolute_deltas.mean()),
            "max_abs_error": float(absolute_deltas.max()),
            "rms_error": float(torch.sqrt(torch.mean(deltas.square()))),
            "by_worker": alignment_by_worker,
            "worst_token": {
                "worker_id": worst_alignment[1],
                "input_position": worst_alignment[2],
                "behavior_logprob": worst_alignment[3],
                "learner_logprob": worst_alignment[4],
                "absolute_error": worst_alignment[0],
                "trajectory_index": worst_alignment[5],
                "group_id": worst_alignment[6],
            },
        }
        trainer._last_correction_metrics = {
            "mode": "token_truncate",
            "cap": trainer.importance_sampling_cap,
            "target_scoring_strategy": "actual_policy_targets_only",
            **trainer._last_scoring_metrics,
            "policy_tokens": int(deltas.numel()),
            "log_ratio_min": float(quantiles[0]),
            "log_ratio_median": float(quantiles[1]),
            "log_ratio_p95": float(quantiles[2]),
            "log_ratio_max": float(quantiles[3]),
            "weight_min": float(values.min()),
            "weight_mean": float(values.mean()),
            "weight_max": float(values.max()),
            "truncated_fraction": float(
                (deltas > math.log(trainer.importance_sampling_cap)).float().mean()
            ),
            "fresh_policy_tokens": fresh_policy_tokens,
            "replay_policy_tokens": replay_policy_tokens,
            "policy_tokens_by_kind": dict(self.policy_tokens_by_kind),
            "replay_correction": 1.0,
            "approximate_correction": True,
            "policy_alignment": policy_alignment,
        }
        self.finished = True
