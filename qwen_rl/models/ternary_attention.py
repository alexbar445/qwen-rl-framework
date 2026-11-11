"""Bound FlashAttention backward workspace without splitting the sequence.

Partition complete GQA groups: no KV head is shared between calls, so their
gradients need no cross-call reduction. Short rows retain the original call.
"""
from __future__ import annotations

import torch


MAX_QUERY_BYTES = 512 * 1024**2
ATTENTION_NAME = "t2_bounded_sdpa"


def bounded_t2_attention_forward(
    module, query, key, value, attention_mask, *,
    query_budget_bytes=MAX_QUERY_BYTES, **kwargs,
):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if type(query_budget_bytes) is not int or query_budget_bytes <= 0:
        raise ValueError("query_budget_bytes must be a positive integer")
    if (not torch.is_grad_enabled() or not module.training
            or kwargs.get("dropout", 0.0) != 0.0):
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    batch, query_heads, length, width = query.shape
    kv_heads = key.shape[1]
    if kv_heads < 1 or query_heads % kv_heads or value.shape[1] != kv_heads:
        raise ValueError("T2 attention requires complete matching GQA groups")
    heads_per_group = query_heads // kv_heads
    # FFPA is the default for the validated dense T2 training geometry. Other
    # masks, dtypes and execution modes retain the original SDPA semantics.
    if (query.is_cuda and query.dtype == torch.bfloat16 and batch == 1
            and query_heads == 24 and kv_heads == 4 and width == 256
            and key.shape[2] == value.shape[2] == length
            and attention_mask is None and getattr(module, "is_causal", False)):
        from .ternary_ffpa import ternary_ffpa

        scale = kwargs.get("scaling")
        output = ternary_ffpa(query, key, value, width ** -0.5 if scale is None else scale)
        return output.transpose(1, 2), None
    group_bytes = batch * length * width * query.element_size() * heads_per_group
    groups_per_call = max(1, query_budget_bytes // max(1, group_bytes))
    if groups_per_call >= kv_heads:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    # split() has one backward join. Independent slice() calls would each
    # allocate a full-sized zero gradient before scattering their head range.
    query_parts = query.split(groups_per_call * heads_per_group, dim=1)
    key_parts = key.split(groups_per_call, dim=1)
    value_parts = value.split(groups_per_call, dim=1)
    if (attention_mask is not None and attention_mask.ndim >= 3
            and attention_mask.shape[-3] == query_heads):
        masks = attention_mask.split(groups_per_call * heads_per_group, dim=-3)
    else:
        masks = (attention_mask,) * len(query_parts)
    outputs = [
        sdpa_attention_forward(module, q, k, v, mask, **kwargs)[0]
        for q, k, v, mask in zip(query_parts, key_parts, value_parts, masks, strict=True)
    ]
    # Transformers SDPA returns [batch, sequence, heads, head_dim].
    return torch.cat(outputs, dim=2), None


def configure_bounded_t2_attention(config):
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(ATTENTION_NAME, bounded_t2_attention_forward)
    AttentionMaskInterface.register(ATTENTION_NAME, sdpa_mask)
    config._attn_implementation = ATTENTION_NAME
    return {"implementation": ATTENTION_NAME, "training_backend": "ffpa-attn-0.2.4",
            "query_budget_bytes": MAX_QUERY_BYTES,
            "partition": "complete_gqa_head_groups", "sequence_split": False}
