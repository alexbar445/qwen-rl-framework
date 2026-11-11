"""Install upstream FLA gated norms after T2 weights leave the meta device.

NInfer computes normalization, affine weight and SiLU in FP32 before a final
BF16 store. FLA uses that same rounding boundary, unlike HF's intermediate
BF16 casts. Reduction/activation implementations can still differ numerically.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

RECIPE = "fp32_norm_weight_silu_then_output_cast/v1"


def _fused_norm_type():
    from fla.modules import FusedRMSNormGated

    return FusedRMSNormGated


def install_fused_gated_norms(layers: Iterable[nn.Module]) -> dict[str, Any]:
    """Keep frozen weight objects/state keys; reject incomplete materialization."""
    targets = []
    for layer in layers:
        if getattr(layer, "layer_type", None) != "linear_attention":
            continue
        linear = layer.linear_attn
        original = linear.norm
        weight = original.weight
        eps = original.variance_epsilon
        activation = linear.activation
        if (not isinstance(weight, nn.Parameter) or weight.is_meta
                or weight.ndim != 1 or weight.numel() != 128
                or weight.dtype != torch.bfloat16 or weight.requires_grad):
            raise ValueError("T2 gated norm requires a materialized frozen BF16 weight of width128")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("T2 gated norm epsilon must be finite and positive")
        if activation not in ("silu", "swish"):
            raise ValueError("T2 gated norm requires a SiLU gate")
        targets.append((linear, weight, eps, activation, original.training))
    if not targets:
        raise ValueError("T2 model has no materialized GDN norms")
    fused_type = _fused_norm_type()
    for linear, weight, eps, activation, training in targets:
        fused = fused_type(weight.numel(), eps=eps, activation=activation,
                           device=weight.device, dtype=weight.dtype)
        fused.weight = weight
        fused.train(training)
        linear.norm = fused
    return {
        "implementation": "fla.modules.FusedRMSNormGated",
        "recipe": RECIPE,
        "layer_count": len(targets),
        "accumulation_dtype": "float32",
        "output_dtype": "bfloat16",
        "frozen_weight_identity_preserved": True,
    }
