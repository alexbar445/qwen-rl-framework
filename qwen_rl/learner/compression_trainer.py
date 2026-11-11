"""Compression task over shared packed models and upstream Trainer state helpers."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch
from transformers import Trainer, TrainingArguments

from .compression_loss import StudentRows, causal_distillation
from .compression_model import base_model


def effective_vocab_scale_changes(model, *, compute=False):
    """Count FP16 export or compute-dtype changes, not just FP32 master movement."""
    from qwen_rl.models.ternary_recovery import (
        ScaleTernaryEmbedding,
        ScaleTernaryLinear,
    )

    result = {}
    for name, module in model.named_modules():
        if not isinstance(module, (ScaleTernaryEmbedding, ScaleTernaryLinear)):
            continue
        current, original = module.exported_scale_words(), module.base.scale_words
        if compute:
            dtype = getattr(module.base, "compute_dtype", module.base.output_dtype)
            current = current.view(torch.float16).to(dtype)
            original = original.view(torch.float16).to(dtype)
        else:
            current, original = current.view(torch.int16), original.view(torch.int16)
        result[name] = int((current != original).sum())
    return result


class CompressionRecoveryTrainer(Trainer):
    def __init__(self, model, config, teacher_head, *, device):
        args = TrainingArguments(
            output_dir=str(config.run_dir / "trainer"),
            learning_rate=config.lr,
            max_steps=config.steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            lr_scheduler_type="constant",
            weight_decay=0.0,
            max_grad_norm=config.max_grad_norm,
            optim="adamw_torch",
            use_cpu=device.type == "cpu",
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            disable_tqdm=True,
        )
        super().__init__(model=model, args=args)
        self.recovery_config, self.teacher_head = config, teacher_head
        self.recovery_device = device
        self.create_optimizer_and_scheduler(num_training_steps=config.steps)

    def create_optimizer(self):
        from qwen_rl.models.ternary_recovery import (
            ScaleTernaryEmbedding,
            ScaleTernaryLinear,
        )

        optimizer = super().create_optimizer()
        root = base_model(self.model)
        # Vocabulary has its explicit learning rate; decoder scales use config.lr.
        scale_ids = {
            id(module.log_scale_delta)
            for module in (root.model.embed_tokens, root.lm_head)
            if isinstance(module, (ScaleTernaryEmbedding, ScaleTernaryLinear))
        }
        scale_lr = getattr(
            self.recovery_config, "vocab_scale_lr", self.recovery_config.lr
        )
        if scale_ids and scale_lr != self.recovery_config.lr:
            # Partition the upstream AdamW groups before its scheduler is created.
            # All other optimizer options and parameter ordering remain upstream-owned.
            groups = list(optimizer.param_groups)
            optimizer.param_groups.clear()
            for group in groups:
                scales = [p for p in group["params"] if id(p) in scale_ids]
                other = [p for p in group["params"] if id(p) not in scale_ids]
                if other or not scales:
                    optimizer.add_param_group({**group, "params": other})
                if scales:
                    optimizer.add_param_group(
                        {**group, "params": scales, "lr": scale_lr}
                    )
        return optimizer

    def _loss(self, record, denominator):
        root = base_model(self.model)
        ids = record["input_ids"].unsqueeze(0).to(self.recovery_device)
        mask = record["attention_mask"].unsqueeze(0).to(self.recovery_device)
        lm = record["loss_mask"].unsqueeze(0).to(self.recovery_device)
        tx = record["teacher_hidden_states"].unsqueeze(0).to(self.recovery_device)
        hidden = root.model(
            input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
        ).last_hidden_state
        cfg = self.recovery_config
        return causal_distillation(
            hidden,
            tx,
            ids,
            mask,
            lm,
            StudentRows(root.lm_head),
            self.teacher_head,
            ce_weight=cfg.ce_weight,
            kl_weight=cfg.kl_weight,
            chunk_rows=cfg.vocab_chunk_rows,
            denominator=denominator,
        )

    def update_records(self, records):
        started = time.perf_counter()
        if not records:
            raise ValueError("recovery update requires records")
        denominator = sum(
            int(
                (
                    r["attention_mask"][:-1]
                    & r["attention_mask"][1:]
                    & r["loss_mask"][1:]
                ).sum()
            )
            for r in records
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {"scored_tokens": 0, "ce_total": 0.0, "kl_total": 0.0}
        try:
            for record in records:
                offload = (
                    torch.autograd.graph.save_on_cpu(pin_memory=False)
                    if self.recovery_config.activation_offload
                    and self.recovery_device.type == "cuda"
                    else nullcontext()
                )
                with offload:
                    loss, metrics = self._loss(record, denominator)
                    loss.backward()
                for k in totals:
                    totals[k] += metrics[k]
            params = [p for p in self.model.parameters() if p.requires_grad]
            if any(p.grad is None for p in params):
                raise RuntimeError("a selected recovery parameter received no gradient")
            grads = [p.grad for p in params if p.grad is not None]
            if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError("recovery gradient missing or nonfinite")
            norm = torch.nn.utils.clip_grad_norm_(
                params, self.recovery_config.max_grad_norm, error_if_nonfinite=True
            )
            if not math.isfinite(float(norm)) or float(norm) == 0:
                raise FloatingPointError("recovery gradient is zero or nonfinite")
            self.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in params):
                raise FloatingPointError(
                    "recovery optimizer produced nonfinite parameters"
                )
            for module in self.model.modules():
                validate = getattr(module, "exported_scale_words", None)
                if callable(validate):
                    validate()  # Reject an unexportable FP16 scale state before committing.
            scale_changes = effective_vocab_scale_changes(self.model)
            forward_changes = effective_vocab_scale_changes(self.model, compute=True)
            self.lr_scheduler.step()
            self.state.global_step += 1
            self.state.num_input_tokens_seen += sum(
                r["input_ids"].numel() for r in records
            )
            totals.update(
                gradient_norm=float(norm),
                completed_updates=self.state.global_step,
                ce_mean=totals["ce_total"] / denominator,
                kl_mean=totals["kl_total"] / denominator,
                input_tokens=sum(r["input_ids"].numel() for r in records),
                wall_seconds=time.perf_counter() - started,
                effective_vocab_scale_word_changes=scale_changes,
                effective_vocab_forward_scale_changes=forward_changes,
                optimizer_learning_rates=[g["lr"] for g in self.optimizer.param_groups],
            )
            return totals
        finally:
            self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def evaluate_records(self, records):
        self.model.eval()
        totals = {"scored_tokens": 0, "ce_total": 0.0, "kl_total": 0.0}
        for record in records:
            count = int(
                (
                    record["attention_mask"][:-1]
                    & record["attention_mask"][1:]
                    & record["loss_mask"][1:]
                ).sum()
            )
            _, metrics = self._loss(record, count)
            for k in totals:
                totals[k] += metrics[k]
        totals["ce_mean"] = totals["ce_total"] / totals["scored_tokens"]
        totals["perplexity"] = math.exp(totals["ce_mean"])
        totals["kl_mean"] = totals["kl_total"] / totals["scored_tokens"]
        return totals
