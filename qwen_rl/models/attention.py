"""Attention adapters for native Qwen Learner backends."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def arc_sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Use Efficient SDPA for the native NVFP4 path with HF masking semantics.

    PyTorch's Efficient SDPA kernel needs matching query and K/V head counts.
    With no explicit mask, expand K/V through Transformers' own helper before
    calling its SDPA implementation. Explicit masks retain HF's normal GQA
    handling so padding, packed, and non-square cached masks stay intact.
    """
    return _arc_sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    )


def _arc_sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Select one explicit public SDPA backend for this model call."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations.sdpa_attention import (
        repeat_kv,
        sdpa_attention_forward,
    )

    if attention_mask is None and query.shape[-3] != key.shape[-3]:
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )
