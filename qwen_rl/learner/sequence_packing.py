"""Explicit independent-sequence packing using pinned Transformers metadata.

This constructs model inputs, not a replacement for GRPO row/group semantics.
Callers must preserve row offsets, masks, actual b and per-group advantages.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import inspect
import copy
import torch


@dataclass(frozen=True)
class PackedSequences:
    tensors: dict[str, Any]
    offsets: tuple[int, ...]
    row_ids: tuple[str, ...]
    sidecars: tuple[dict[str, Any], ...]

    def to(self, device):
        return PackedSequences({k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in self.tensors.items()}, self.offsets, self.row_ids, self.sidecars)

    @property
    def valid_targets(self):
        return int((self.tensors["labels"][:, 1:] != -100).sum())

    def backbone_kwargs(self):
        return {k: v for k, v in self.tensors.items() if k != "labels"}


def pack_independent_sequences(rows: list[dict[str, Any]], *, max_tokens: int) -> PackedSequences:
    from transformers import DataCollatorWithFlattening
    if type(max_tokens) is not int or max_tokens < 2 or not rows:
        raise ValueError("explicit positive token budget and nonempty sequences required")
    features, offsets, ids, sidecars = [], [0], [], []
    for index, row in enumerate(rows):
        values = row.get("input_ids")
        if not isinstance(values, list) or len(values) < 2 or any(type(x) is not int or x < 0 for x in values):
            raise ValueError("each independent sequence needs at least2 integer tokens")
        labels = list(row.get("labels", values))
        if len(labels) != len(values):
            raise ValueError("labels must align with each sequence")
        mask = row.get("policy_mask")
        if mask is not None:
            if len(mask) != len(values) or mask[0] or any(x not in (0, 1, False, True) for x in mask):
                raise ValueError("policy mask must align and leave each sequence's first token inactive")
            labels = [label if active else -100 for label, active in zip(labels, mask, strict=True)]
        behavior = row.get("behavior_logprobs")
        if behavior is not None and len(behavior) != len(values):
            raise ValueError("actual behavior scores must align; never invent packed probabilities")
        offsets.append(offsets[-1] + len(values))
        if offsets[-1] > max_tokens:
            raise ValueError("packing exceeds explicit token budget; no truncation or dropped rows")
        features.append({"input_ids": values, "labels": labels})
        ids.append(str(row.get("row_id", index)))
        sidecars.append(copy.deepcopy({k: v for k, v in row.items() if k not in ("input_ids", "labels")}))
    result = DataCollatorWithFlattening(return_tensors="pt", return_position_ids=True,
        return_seq_idx=True, return_flash_attn_kwargs=True)(features)
    if result["cu_seq_lens_q"].tolist() != offsets or result["cu_seq_lens_k"].tolist() != offsets:
        raise RuntimeError("upstream collator lost independent sequence boundaries")
    for start in offsets[:-1]:
        if int(result["position_ids"][0, start]) != 0 or int(result["labels"][0, start]) != -100:
            raise RuntimeError("packed position/target boundary isolation failed")
    return PackedSequences(result, tuple(offsets), tuple(ids), tuple(sidecars))


def enable_qwen35_packing(model) -> dict[str, Any]:
    """Require upstream varlen attention, causal-conv seq_idx and FLA resets."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.set_attn_implementation("flash_attention_2")
    if base.config._attn_implementation != "flash_attention_2":
        raise RuntimeError("packing requires validated FlashAttention varlen; no implicit SDPA fallback")
    gdn = []
    for layer in base.model.layers:
        if getattr(layer, "layer_type", None) != "linear_attention":
            continue
        module = layer.linear_attn
        conv, scan = module.causal_conv1d_fn, module.chunk_gated_delta_rule
        if (conv is None or not conv.__module__.startswith("causal_conv1d")
                or "seq_idx" not in inspect.signature(conv).parameters
                or not scan.__module__.startswith("fla.")
                or "cu_seqlens" not in inspect.signature(scan).parameters):
            raise RuntimeError("packing requires upstream trainable causal-conv seq_idx and FLA cu_seqlens")
        gdn.append({"conv": conv.__module__, "scan": scan.__module__})
    if not gdn:
        raise RuntimeError("no Qwen3.5 GDN boundary-capable layers found")
    return {"attention": "flash_attention_2", "GDN_layers": len(gdn),
            "conv_boundary": "seq_idx", "scan_boundary": "cu_seq_lens_q",
            "position_ids": "reset_per_sequence", "loss_boundary": "first_label_minus100",
            "upstream_collator": "transformers.DataCollatorWithFlattening"}
