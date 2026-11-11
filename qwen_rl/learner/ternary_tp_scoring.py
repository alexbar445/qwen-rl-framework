"""Causal token scoring for padded microbatches on the experimental T2 TP path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from qwen_rl.learner.sequence_packing import pack_independent_sequences


@dataclass(frozen=True)
class _ValidRow:
    index: int
    start: int
    stop: int
    input_ids: list[int]
    labels: list[int]

    @property
    def length(self) -> int:
        return self.stop - self.start


def _valid_rows(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[_ValidRow, ...]:
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    if not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape:
        raise ValueError("labels must match input_ids shape")
    if input_ids.shape[0] < 1 or input_ids.shape[1] < 2:
        raise ValueError("token scoring requires a nonempty batch and sequence width of at least two")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must use an integer token dtype")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError("labels must use an integer token dtype")

    ids_rows = input_ids.detach().to(device="cpu").tolist()
    mask_rows = attention_mask.detach().to(device="cpu").tolist()
    label_rows = labels.detach().to(device="cpu").tolist()
    rows = []
    for row_index, (ids, mask, targets) in enumerate(
        zip(ids_rows, mask_rows, label_rows, strict=True)
    ):
        if any(value not in (0, 1, False, True) for value in mask):
            raise ValueError("attention_mask must contain only 0/1 values")
        active = [index for index, value in enumerate(mask) if bool(value)]
        if len(active) < 2:
            raise ValueError("each row must contain at least two valid tokens")
        start, stop = active[0], active[-1] + 1
        if any(not bool(value) for value in mask[start:stop]):
            raise ValueError("each attention row must be one contiguous segment; interior padding is unsupported")

        valid_ids = ids[start:stop]
        valid_labels = targets[start:stop]
        if any(type(token) is not int or token < 0 for token in valid_ids):
            raise ValueError("valid input token IDs must be nonnegative integers")
        if any(type(target) is not int or (target < 0 and target != -100) for target in valid_labels):
            raise ValueError("labels must be nonnegative token IDs or the -100 ignore index")
        # Position zero has no causal predecessor in TRL's shifted-label path.
        # The caller tensor stays untouched; make only the scoring copy explicit.
        valid_labels[0] = -100
        rows.append(_ValidRow(row_index, start, stop, valid_ids, valid_labels))
    return tuple(rows)


def _whole_row_chunks(rows: tuple[_ValidRow, ...], max_tokens: int) -> tuple[tuple[_ValidRow, ...], ...]:
    if type(max_tokens) is not int or max_tokens < 2:
        raise ValueError("max_tokens must be an explicit integer of at least two")
    chunks = []
    current = []
    used_tokens = 0
    for row in rows:
        if row.length > max_tokens:
            raise ValueError(
                f"valid row {row.index} has {row.length} tokens, above max_tokens={max_tokens}; "
                "rows cannot be truncated or split"
            )
        if current and used_tokens + row.length > max_tokens:
            chunks.append(tuple(current))
            current, used_tokens = [], 0
        current.append(row)
        used_tokens += row.length
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def _check_fused_output(output: Any, *, packed_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = getattr(output, "log_probs", None)
    entropy = getattr(output, "entropy", None)
    expected = (1, packed_tokens - 1)
    if not isinstance(log_probs, torch.Tensor) or tuple(log_probs.shape) != expected:
        shape = tuple(log_probs.shape) if isinstance(log_probs, torch.Tensor) else type(log_probs).__name__
        raise RuntimeError(f"fused T2 head returned log_probs {shape}; expected {expected}")
    if not isinstance(entropy, torch.Tensor) or tuple(entropy.shape) != expected:
        shape = tuple(entropy.shape) if isinstance(entropy, torch.Tensor) else type(entropy).__name__
        raise RuntimeError(f"fused T2 head returned entropy {shape}; expected {expected}")
    return log_probs.float(), entropy.float()


def score_t2_tp_rows(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each padded row's causal targets through whole-row packed TP calls.

    Rows are trimmed only to their one contiguous attention segment, greedily
    packed in input order under max_tokens, then scored with the model's fused
    T2 vocabulary head. Returned tensors align to labels[:, 1:] (TRL's causal
    positions), with differentiable zero padding restored to the original row
    geometry. The caller's IDs, masks and labels are never modified.
    """
    rows = _valid_rows(input_ids, attention_mask, labels)
    chunks = _whole_row_chunks(rows, max_tokens)
    batch, sequence_width = input_ids.shape
    score_rows: list[torch.Tensor | None] = [None] * batch
    entropy_rows: list[torch.Tensor | None] = [None] * batch

    for chunk in chunks:
        pack = pack_independent_sequences(
            [
                {
                    "row_id": str(row.index),
                    "input_ids": row.input_ids,
                    "labels": row.labels,
                }
                for row in chunk
            ],
            max_tokens=max_tokens,
        ).to(input_ids.device)
        packed_labels = pack.tensors["labels"]
        output = model(
            **pack.backbone_kwargs(),
            labels=packed_labels,
            fused_lm_head=True,
        )
        packed_log_probs, packed_entropy = _check_fused_output(
            output, packed_tokens=pack.offsets[-1]
        )

        for chunk_index, row in enumerate(chunk):
            packed_start, packed_stop = pack.offsets[chunk_index : chunk_index + 2]
            row_log_probs = packed_log_probs[0, packed_start : packed_stop - 1]
            row_entropy = packed_entropy[0, packed_start : packed_stop - 1]
            left_padding = row.start
            right_padding = sequence_width - 1 - left_padding - (row.length - 1)
            if right_padding < 0:
                raise RuntimeError("packed score mapping exceeds the original padded row width")
            score_rows[row.index] = F.pad(row_log_probs, (left_padding, right_padding))
            entropy_rows[row.index] = F.pad(row_entropy, (left_padding, right_padding))

    if any(row is None for row in score_rows) or any(row is None for row in entropy_rows):
        raise RuntimeError("whole-row TP scoring failed to return every input row")
    return torch.stack(score_rows, dim=0), torch.stack(entropy_rows, dim=0)


__all__ = ["score_t2_tp_rows"]
