"""One-update proximal capture without retaining earlier forward graphs."""
from __future__ import annotations

from typing import Any

import torch

from .proximal import ProximalAccumulator


class InlineProximalWindow:
    """Capture each final microbatch at unchanged pre-update parameters.

    Only detached scores/correction metadata survive each call. The upstream
    loss consumes the same differentiable scores; earlier graphs are not cached.
    """

    def __init__(self, trainer: Any, batches, row_metadata, *, row_kind_to_id, id_to_row_kind):
        if trainer.parallel_mode not in {"single", "ddp"}:
            raise ValueError("inline proximal capture requires single or DDP training")
        count = len(batches)
        if trainer.gradient_only:
            valid_window = (
                count >= 1
                and trainer.num_iterations == 1
                and trainer.args.num_iterations == 1
                and trainer.args.max_steps == 1
                and trainer.args.steps_per_generation == 1
                and trainer.args.gradient_accumulation_steps == 1
                and int(trainer.accelerator.num_processes) == 1
                and int(trainer.accelerator.gradient_accumulation_steps) == 1
            )
        else:
            valid_window = (
                count > 0
                and trainer.num_iterations == 1
                and trainer.args.num_iterations == 1
                and trainer.args.max_steps == 1
                and trainer.args.steps_per_generation == count
                and trainer.args.gradient_accumulation_steps == count
            )
        if not valid_window:
            raise ValueError("inline proximal capture requires one complete generation in one optimizer window")
        if row_metadata is not None and len(row_metadata) != count:
            raise ValueError("inline proximal metadata differs from the final batch count")
        self.trainer = trainer
        self.batches = tuple(batches)
        self.indices = {id(batch): index for index, batch in enumerate(batches)}
        if len(self.indices) != count:
            raise ValueError("inline proximal window contains a repeated batch object")
        self.row_metadata = row_metadata
        self.seen_batches: set[int] = set()
        self.initial_step = int(trainer.state.global_step)
        self.parameters = tuple(parameter for parameter in trainer.model.parameters() if parameter.requires_grad)
        self.versions = tuple(parameter._version for parameter in self.parameters)
        self.collector = ProximalAccumulator(
            trainer, row_kind_to_id=row_kind_to_id, id_to_row_kind=id_to_row_kind,
        )

    def _assert_unchanged(self):
        if int(self.trainer.state.global_step) != self.initial_step:
            raise RuntimeError("optimizer progress changed during inline proximal capture")
        if tuple(parameter._version for parameter in self.parameters) != self.versions:
            raise RuntimeError("trainable parameters changed during inline proximal capture")

    def capture(self, batch, scores: torch.Tensor):
        self._assert_unchanged()
        index = self.indices.get(id(batch))
        if index is None or self.batches[index] is not batch:
            raise RuntimeError("inline proximal scores belong to an unexpected microbatch")
        if index in self.seen_batches:
            raise RuntimeError("inline proximal microbatch was scored more than once")
        if not torch.is_grad_enabled() or not scores.requires_grad:
            raise RuntimeError("inline proximal capture requires the differentiable training forward")
        metadata = None if self.row_metadata is None else self.row_metadata[index]
        self.collector.add(batch, scores.detach(), row_metadata=metadata)
        self.seen_batches.add(index)
        if len(self.seen_batches) == len(self.batches):
            self.collector.finish()
            self.trainer._last_correction_metrics.update(
                proximal_scoring_strategy="shared_training_forward",
                proximal_extra_model_forwards=0,
                proximal_captured_microbatches=len(self.seen_batches),
            )
            self.trainer._report_parallel_phase(
                "proximal_scoring_completed", rank=self.trainer.parallel_rank,
                strategy="shared_training_forward", extra_model_forwards=0,
            )

    def assert_complete(self):
        self._assert_unchanged()
        if len(self.seen_batches) != len(self.batches) or not self.collector.finished:
            raise RuntimeError("optimizer step blocked: inline proximal capture is incomplete")
