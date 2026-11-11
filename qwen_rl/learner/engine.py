"""Single-GPU native NVFP4 QLoRA learner backed by the upstream TRL GRPOTrainer."""
from __future__ import annotations

from ..contracts.activation_offload import DEFAULT_OFFLOAD_MIN_TOKENS

import copy
import math
from pathlib import Path
import threading
import uuid
from collections.abc import Callable
from typing import Any

import torch

from qwen_rl.learner.core import LearnerEngineCore
from qwen_rl.learner.recovery import (
    RecoveryError,
    RecoveryStore,
    training_compatibility,
)


class LearnerEngine(LearnerEngineCore):
    """One visible CUDA device, one fresh group update per call."""

    def __init__(self, config: dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError("LearnerEngine config must be a dict")
        self.config = dict(config)
        quantization = self.config.get("quantization", {})
        mode = quantization.get("mode", "torchao") if isinstance(quantization, dict) else None
        if mode != "torchao":
            raise ValueError(
                "LearnerEngine requires quantization.mode='torchao' (native NVFP4); "
                "GPTQ/compressed-tensors support has been removed"
            )
        self.model_path = str(self.config["model_path"])
        self.lora_rank = int(self.config.get("lora_rank", 16))
        self.lora_alpha = int(self.config.get("lora_alpha", 16))
        self.lr = float(self.config.get("lr", 1e-6))
        self.activation_offload = self.config.get("activation_offload", True)
        self.activation_offload_min_tokens = self.config.get("activation_offload_min_tokens", DEFAULT_OFFLOAD_MIN_TOKENS)
        if type(self.activation_offload_min_tokens) is not int or self.activation_offload_min_tokens < 0:
            raise ValueError("activation_offload_min_tokens must be an integer >= 0")
        if type(self.activation_offload) is not bool:
            raise ValueError("activation_offload must be a boolean")
        self.max_sequence_length = int(self.config.get("max_sequence_length", 65536))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        importance_sampling_cap = self.config.get("importance_sampling_cap", 3.0)
        if (isinstance(importance_sampling_cap, bool)
                or not isinstance(importance_sampling_cap, (int, float))
                or not math.isfinite(importance_sampling_cap)
                or importance_sampling_cap < 1.0):
            raise ValueError("importance_sampling_cap must be finite and >= 1")
        self.importance_sampling_cap = float(importance_sampling_cap)
        if self.lr <= 0 or not 2 <= self.max_sequence_length <= 65536 or self.max_grad_norm <= 0:
            raise ValueError("lr and max_grad_norm must be positive; max_sequence_length must be between 2 and 65536")
        self.group_size = self.config.get("group_size", 3)
        self.groups_per_update = self.config.get("groups_per_update", 2)
        self.microbatch_size = self.config.get("microbatch_size", 2)
        self.gradient_only = self.config.get("gradient_only", False)
        if type(self.gradient_only) is not bool:
            raise ValueError("gradient_only must be a boolean")
        self.max_padded_tokens_per_microbatch = self.config.get("max_padded_tokens_per_microbatch")
        if (self.max_padded_tokens_per_microbatch is not None
                and (type(self.max_padded_tokens_per_microbatch) is not int
                     or self.max_padded_tokens_per_microbatch < 1)):
            raise ValueError("max_padded_tokens_per_microbatch must be a positive integer")
        if (self.max_padded_tokens_per_microbatch is not None
                and self.config.get("parallelism", "single") != "single"):
            raise ValueError("max_padded_tokens_per_microbatch requires parallelism='single'")
        for name, value, minimum in (
            ("group_size", self.group_size, 2),
            ("groups_per_update", self.groups_per_update, 1),
            ("microbatch_size", self.microbatch_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.total_samples = self.group_size * self.groups_per_update
        if self.microbatch_size > self.total_samples or self.total_samples % self.microbatch_size:
            raise ValueError(
                "microbatch_size must divide group_size * groups_per_update "
                f"({self.total_samples}) exactly"
            )
        self.gradient_accumulation_steps = self.total_samples // self.microbatch_size

        recovery_dir = self.config.get("recovery_dir")
        self.recovery_store: RecoveryStore | None = None
        self._recovery_lock = threading.RLock()
        self._recovery_compatibility: dict[str, Any] | None = None
        self._recovery_mode = "disabled"
        self._recovery_resumed = False
        self._recovery_dirty = False
        self._durable_policy_version: int | None = None
        self._last_update_context: dict[str, Any] | None = None
        self._engine_epoch: str | None = None
        self._loaded_snapshot: dict[str, Any] | None = None
        self._recovery_metadata: dict[str, Any] = {"enabled": False, "resumed": False}
        if recovery_dir is not None:
            self.recovery_store = RecoveryStore(
                recovery_dir,
                mode=str(self.config.get("recovery_mode", "auto")),
                lineage_id=self.config.get("recovery_lineage_id"),
            )
            self._recovery_mode = "resumed" if self.recovery_store.resumed else "fresh"
            checkpoint_dir = self.recovery_store.checkpoint_dir
        else:
            checkpoint_dir = self.config.get("checkpoint_dir")
            if checkpoint_dir is None:
                checkpoint_dir = Path(__file__).resolve().parents[2] / "outputs/learner-development/checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.publication_dir = self.checkpoint_dir.parent / "publications"
        self.policy_version = 0
        self._configure_publication_state()
        self.model, self.tokenizer, self.evidence = self.load_policy()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._base_model = self.model.get_base_model()
        self._backbone = getattr(self._base_model, "model", None)
        if self._backbone is None:
            raise RuntimeError("Qwen3.5 causal model has no .model text backbone")
        self._head = self._base_model.get_output_embeddings()
        if self._head is None or getattr(self._head, "weight", None) is None:
            raise RuntimeError("Qwen3.5 causal model has no classifier weight")

        self._adapter_parameters = [
            p for name, p in self.model.named_parameters() if p.requires_grad and "lora_" in name
        ]
        if not self._adapter_parameters:
            raise RuntimeError("LearnerEngine has no trainable LoRA parameters")
        if any(
            p.requires_grad and "lora_" not in name
            for name, p in self.model.named_parameters()
        ):
            raise RuntimeError("base weights must remain frozen during QLoRA training")
        from qwen_rl.learner.trainer import LearnerGRPOTrainer

        self.trainer = LearnerGRPOTrainer(
            model=self.model,
            constrained_decoding=self.config.get("constrained_decoding",False),
            processing_class=self.tokenizer,
            group_size=self.group_size,
            groups_per_update=self.groups_per_update,
            microbatch_size=self.microbatch_size,
            max_padded_tokens_per_microbatch=self.max_padded_tokens_per_microbatch,
            learning_rate=self.lr,
            max_grad_norm=self.max_grad_norm,
            max_sequence_length=self.max_sequence_length,
            weight_decay=float(self.config.get("weight_decay", 0.0)),
            adam_betas=tuple(self.config.get("adam_betas", (0.9, 0.999))),
            adam_eps=float(self.config.get("adam_eps", 1e-8)),
            importance_sampling_cap=self.importance_sampling_cap,
            output_dir=self.checkpoint_dir.parent / "trainer",
            activation_offload=self.activation_offload,
            activation_offload_min_tokens=self.activation_offload_min_tokens,
            checkpoint_input_recompute_span=self.config.get("checkpoint_input_recompute_span", 1),
            # Ragged T2 activations otherwise accumulate tens of GiB in the pinned
            # host allocator across microbatches, even after backward releases them.
            activation_offload_pin_memory=self.config.get("model_backend", "default") != "ternary",
            gradient_only=self.gradient_only,
        )
        self.evidence.update({
            "optimizer": (
                "none; gradient-only executor returns DAPO contributions"
                if self.gradient_only else "Trainer-owned AdamW"
            ),
            "trainer": (
                "trl.GRPOTrainer.compute_loss + Accelerator.backward; no optimizer step"
                if self.gradient_only else "trl.GRPOTrainer.train"
            ),
            "activation_offload": self.activation_offload,
            "activation_offload_min_tokens": self.activation_offload_min_tokens,
            "activation_offload_pin_memory": self.trainer.activation_offload_pin_memory,
            "learning_rate": self.lr,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "loss_aggregation": "global-token-mean",
            "advantage": "trl.GRPOTrainer group-normalized rewards",
            "policy_loss": "trl.GRPOTrainer DAPO",
            "logprob_path": "TRL fused/chunked LM head",
            "kl_loss": False,
            "importance_sampling_correction": True,
            "importance_sampling_correction_cap": self.importance_sampling_cap,
            "max_sequence_length": self.max_sequence_length,
            "group_size": self.group_size,
            "groups_per_update": self.groups_per_update,
            "microbatch_size": self.microbatch_size,
            "max_padded_tokens_per_microbatch": self.max_padded_tokens_per_microbatch,
            "dynamic_microbatch_enabled": self.max_padded_tokens_per_microbatch is not None,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        })

        if self.recovery_store is not None:
            self._recovery_compatibility = training_compatibility(
                self.config,
                model=self.model,
                tokenizer=self.tokenizer,
                trainer=self.trainer,
                evidence=self.evidence,
            )
            restored = self.recovery_store.load(
                model=self.model,
                trainer=self.trainer,
                compatibility=self._recovery_compatibility,
            )
            if restored is None:
                self._engine_epoch = uuid.uuid4().hex
                self.policy_version = 0
                self._last_update_context = None
                self._write_recovery_snapshot(reason="lineage_initialized", update_context=None)
            else:
                self.policy_version = int(restored["policy_version"])
                self._engine_epoch = str(restored["engine_epoch"])
                self._recovery_resumed = True
                self._last_update_context = restored.get("update_context")
                pointer = self.recovery_store.pointer or {}
                snapshot_path = self.recovery_store.current_snapshot_path
                self._loaded_snapshot = {
                    "policy_version": self.policy_version,
                    "snapshot_id": restored.get("snapshot_id", pointer.get("snapshot_id")),
                    "snapshot_sha256": restored.get("snapshot_sha256", pointer.get("manifest_sha256")),
                    "path": restored.get("path") if snapshot_path is None else str(snapshot_path),
                }
                self._durable_policy_version = self.policy_version
                self._refresh_recovery_metadata(
                    snapshot_sha256=restored.get("snapshot_sha256", pointer.get("manifest_sha256")),
                    snapshot_id=restored.get("snapshot_id", pointer.get("snapshot_id")),
                    snapshot_path=restored.get("path") if snapshot_path is None else str(snapshot_path),
                    compatibility_sha256=restored.get("compatibility_sha256"),
                )
                self._adopt_restored_publication(restored.get("publication_hashes"))

    def load_policy(self):
        """Load the frozen base plus its trainable adapter through one overridable seam."""
        from qwen_rl.models.nvfp4 import load_nvfp4_policy

        return load_nvfp4_policy(
            self.model_path,
            trainable=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            gradient_checkpointing=bool(self.config.get("gradient_checkpointing", True)),
            parallel_mode="single",
        )

    @property
    def optimizer(self):
        """Expose the persistent optimizer created and stepped by Trainer."""
        return self.trainer.optimizer

    def score_policy_tokens(
        self,
        input_ids: torch.Tensor,
        policy_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate token scoring to TRL's fused LM-head patch."""
        return self.trainer.score_token_ids(input_ids, policy_mask, attention_mask)

    @torch.no_grad()
    def logprobs(self, input_ids: list[int], policy_mask: list[int]) -> list[float]:
        """Recompute policy logprobs aligned to input positions; masked positions are zero."""
        if len(input_ids) != len(policy_mask):
            raise ValueError("input_ids and policy_mask lengths must match")
        if any(value not in (0, 1, False, True) for value in policy_mask):
            raise ValueError("policy_mask must contain only 0/1 values")
        if len(input_ids) > self.max_sequence_length:
            raise ValueError(
                f"sequence length {len(input_ids)} exceeds max_sequence_length={self.max_sequence_length}"
            )
        if len(input_ids) < 2:
            raise ValueError("logprobs requires at least two input tokens")
        token_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        mask = torch.tensor([policy_mask], dtype=torch.bool, device=self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            log_probs, selected = self.score_policy_tokens(token_ids, mask)
        finally:
            self.model.train(was_training)
        aligned = torch.zeros(len(input_ids), dtype=torch.float32, device="cpu")
        aligned[1:] = log_probs[0].float().cpu() * selected[0].float().cpu()
        return aligned.tolist()

    def update(
        self,
        trajectories: list[dict[str, Any]],
        *,
        before_optimizer_step: Callable[[], None] | None = None,
        recovery_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply one Trainer-owned DAPO step over complete external rollout groups."""
        if self.gradient_only:
            raise RuntimeError("gradient-only LearnerEngine cannot apply update optimizer steps")
        if self._recovery_dirty:
            raise RecoveryError("an optimizer step is newer than the durable snapshot; restart from latest state")
        learner_start_version = self.policy_version
        batch, total_tokens, group_prefix_hashes, group_sizes = self._validated_batch(
            trajectories,
            learner_start_version=learner_start_version,
        )
        behavior_versions = {row["policy_version"] for row in batch if row["row_kind"] == "fresh_student"}
        if len(behavior_versions) != 1:
            raise RuntimeError("validated batch lost its single fresh behavior policy version")
        behavior_version = next(iter(behavior_versions))
        policy_lag = learner_start_version - behavior_version
        from qwen_rl.learner.trainer import LearnerGRPOTrainer

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        telemetry = getattr(self.trainer, "telemetry", None)
        try:
            train_kwargs = {"before_optimizer_step": before_optimizer_step}
            if isinstance(self.trainer, LearnerGRPOTrainer):
                train_kwargs["defer_telemetry_finish"] = True
            output = self.trainer.train_groups(batch, **train_kwargs)
        except BaseException as exc:
            if telemetry is not None:
                telemetry.finish_attempt(
                    status="failed",
                    failure_stage=getattr(self.trainer, "failure_stage", None),
                    error_type=type(exc).__name__,
                )
            # Trainer may fail in a post-step assertion after AdamW has already
            # changed the model. Fence this engine so it cannot apply another
            # batch on top of a step that has no durable recovery snapshot.
            if (self.trainer.optimizer_step_seen
                    and not bool(getattr(self.trainer.accelerator, "optimizer_step_was_skipped", False))):
                self.policy_version = learner_start_version + 1
                self._recovery_dirty = self.recovery_store is not None
                self._refresh_after_failed_durability()
            raise
        # train_groups guarantees exactly one guarded, non-skipped optimizer
        # step before returning. Record that irreversible fact before any
        # diagnostic or publication can fail.
        self.policy_version += 1
        try:
            self.trainer.failure_stage = "post_optimizer_validation"
            correction_metrics = getattr(self.trainer, "_last_correction_metrics", None)
            if not isinstance(correction_metrics, dict) or not correction_metrics:
                raise RuntimeError("Trainer did not report behavior-policy importance-correction metrics")
            grad_norm = self.trainer.gradient_norm_before_clip
            if grad_norm is None or not math.isfinite(grad_norm) or grad_norm <= 0.0:
                raise RuntimeError("Trainer did not report a finite nonzero pre-clip gradient norm")
            self.trainer.failure_stage = "publication"
            publication = self.publish()
            self.evidence["last_update_peak_memory_allocated_mib"] = round(
                torch.cuda.max_memory_allocated() / (1024**2), 2
            )
            self.evidence["last_update_peak_memory_reserved_mib"] = round(
                torch.cuda.max_memory_reserved() / (1024**2), 2
            )
            self.evidence["last_update_gradient_norm_before_clip"] = grad_norm
            if self.recovery_store is not None:
                self.trainer.failure_stage = "checkpoint"
                self._write_recovery_snapshot(
                    reason="optimizer_update",
                    update_context=recovery_context,
                )
                self._recovery_dirty = False
            if telemetry is not None:
                telemetry.finish_attempt(
                    status="committed",
                    policy_version=self.policy_version,
                    quality=self.trainer.quality_snapshot(),
                )
        except BaseException as exc:
            if telemetry is not None:
                telemetry.finish_attempt(
                    status="failed",
                    failure_stage=getattr(self.trainer, "failure_stage", None),
                    error_type=type(exc).__name__,
                    policy_version=self.policy_version,
                    quality=self.trainer.quality_snapshot(),
                )
            if self.recovery_store is not None and self.policy_version != self._durable_policy_version:
                self._recovery_dirty = True
                self._refresh_after_failed_durability()
            raise
        actual_gradient_accumulation_steps = int(
            getattr(
                self.trainer,
                "_last_actual_gradient_accumulation_steps",
                self.gradient_accumulation_steps,
            )
        )
        microbatch_plan = copy.deepcopy(
            getattr(self.trainer, "_last_microbatch_plan_report", {})
        )
        self.evidence["last_update_gradient_accumulation_steps"] = actual_gradient_accumulation_steps
        self.evidence["last_microbatch_plan"] = copy.deepcopy(microbatch_plan)
        return {
            "updated": True,
            "policy_version": self.policy_version,
            "behavior_version": behavior_version,
            "learner_start_version": learner_start_version,
            "policy_lag": policy_lag,
            "importance_correction_enabled": True,
            "importance_correction_cap": self.importance_sampling_cap,
            "importance_correction_metrics": dict(correction_metrics),
            "trajectory_count": len(batch),
            "group_count": len(group_sizes),
            "policy_tokens": total_tokens,
            "row_kind_counts": {
                kind: sum(row["row_kind"] == kind for row in batch)
                for kind in ("fresh_student", "native_replay", "guided_replay")
            },
            "groups_per_update": self.groups_per_update,
            "microbatch_size": self.microbatch_size,
            "max_padded_tokens_per_microbatch": self.max_padded_tokens_per_microbatch,
            "gradient_accumulation_steps": actual_gradient_accumulation_steps,
            "microbatch_plan": microbatch_plan,
            "group_cardinalities": {
                str(group_id): int(size) for group_id, size in group_sizes.items()
            },
            "loss_metrics": dict(self.trainer._last_trainer_metrics),
            "group_prompt_prefixes": [
                {
                    "group_id": str(group_id),
                    "sha256": group_prefix_hashes[group_id],
                    "trajectory_count": group_sizes[group_id],
                }
                for group_id in group_prefix_hashes
            ],
            "gradient_norm_before_clip": grad_norm,
            "max_grad_norm": self.max_grad_norm,
            "publication": publication,
        }
