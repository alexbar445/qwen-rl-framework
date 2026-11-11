"""Deterministic CPU planner for length-aware learner microbatches."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MicrobatchPlan:
    """Immutable row-index layout and padding-cost summary for one update."""

    batches: tuple[tuple[int, ...], ...]
    padded_tokens: tuple[int, ...]
    oversize_singletons: tuple[int, ...]
    actual_tokens: int
    total_padded_tokens: int


def plan_microbatches(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int,
    max_padded_tokens: int,
) -> MicrobatchPlan:
    """Greedily group rows under row-count and prompt/completion padding budgets.

    Rows are bucketed by exact prompt length; each bucket is ordered by
    descending total length, then descending completion length, then original
    index. The estimated padded cost for a batch is
    ``B * (max_prompt_len + max_completion_len)``. A row whose own length
    exceeds the token budget is preserved as an oversize singleton.
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    if (isinstance(max_padded_tokens, bool)
            or not isinstance(max_padded_tokens, int)
            or max_padded_tokens <= 0):
        raise ValueError("max_padded_tokens must be a positive integer")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError("rows must be a sequence of mappings")

    # Each entry is (original index, prompt length, completion length, total length).
    measured: list[tuple[int, int, int, int]] = []
    actual_tokens = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a mapping")
        input_ids = row.get("input_ids")
        policy_mask = row.get("policy_mask")
        if (not isinstance(input_ids, Sequence)
                or isinstance(input_ids, (str, bytes, bytearray))):
            raise ValueError(f"row {index} input_ids must be a sequence")
        if (not isinstance(policy_mask, Sequence)
                or isinstance(policy_mask, (str, bytes, bytearray))):
            raise ValueError(f"row {index} policy_mask must be a sequence")
        if not input_ids:
            raise ValueError(f"row {index} input_ids must be non-empty")
        if len(input_ids) != len(policy_mask):
            raise ValueError(f"row {index} input_ids and policy_mask lengths differ")
        if any(value not in (0, 1, False, True) for value in policy_mask):
            raise ValueError(f"row {index} policy_mask must contain only 0/1 values")
        prompt_len = next(
            (position for position, value in enumerate(policy_mask) if value == 1),
            None,
        )
        if prompt_len is None or prompt_len <= 0:
            raise ValueError(f"row {index} must have its first policy token after position 0")
        completion_len = len(input_ids) - prompt_len
        total_len = len(input_ids)
        measured.append((index, prompt_len, completion_len, total_len))
        actual_tokens += total_len

    measured.sort(key=lambda item: (item[1], -item[3], -item[2], item[0]))
    batches: list[tuple[int, ...]] = []
    padded_tokens: list[int] = []
    oversize_singletons: list[int] = []
    current_ids: list[int] = []
    current_prompt_len: int | None = None
    current_max_prompt = 0
    current_max_completion = 0

    def flush() -> None:
        nonlocal current_ids, current_prompt_len, current_max_prompt, current_max_completion
        if not current_ids:
            return
        cost = len(current_ids) * (current_max_prompt + current_max_completion)
        batches.append(tuple(current_ids))
        padded_tokens.append(cost)
        current_ids = []
        current_prompt_len = None
        current_max_prompt = 0
        current_max_completion = 0

    for index, prompt_len, completion_len, total_len in measured:
        if current_ids and prompt_len != current_prompt_len:
            flush()
        if total_len > max_padded_tokens:
            flush()
            batches.append((index,))
            padded_tokens.append(total_len)
            oversize_singletons.append(index)
            continue

        if not current_ids:
            current_ids = [index]
            current_prompt_len = prompt_len
            current_max_prompt = prompt_len
            current_max_completion = completion_len
            continue

        candidate_rows = len(current_ids) + 1
        candidate_prompt = max(current_max_prompt, prompt_len)
        candidate_completion = max(current_max_completion, completion_len)
        candidate_cost = candidate_rows * (candidate_prompt + candidate_completion)
        if candidate_rows <= max_rows and candidate_cost <= max_padded_tokens:
            current_ids.append(index)
            current_max_prompt = candidate_prompt
            current_max_completion = candidate_completion
        else:
            flush()
            current_ids = [index]
            current_prompt_len = prompt_len
            current_max_prompt = prompt_len
            current_max_completion = completion_len

    flush()
    return MicrobatchPlan(
        batches=tuple(batches),
        padded_tokens=tuple(padded_tokens),
        oversize_singletons=tuple(sorted(oversize_singletons)),
        actual_tokens=actual_tokens,
        total_padded_tokens=sum(padded_tokens),
    )
