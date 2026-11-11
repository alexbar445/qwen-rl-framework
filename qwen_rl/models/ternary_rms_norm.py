"""Row-bounded original PyTorch RMSNorm for frozen Qwen3.5 T2 weights.

Qwen normalizes and multiplies by (1 + weight) in FP32 before casting once.
Original forward statistics run in independent row tiles and inverse RMS is
saved for the Liger fused CUDA backward, which skips the frozen weight gradient. Parameter objects and state keys remain.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable

import torch
from torch import nn

RECIPE = "qwen_fp32_row_tiles_saved_inverse_liger_frozen_weight_backward_v2"


def _reference(x, weight, eps, *, return_inverse_rms=False):
    z = x.float()
    inverse_rms = torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)
    output = (z * inverse_rms * (1.0 + weight.float())).type_as(x)
    return (output, inverse_rms) if return_inverse_rms else output


@lru_cache(maxsize=8)
def _liger_rms_backend():
    """Load the pinned fused RMSNorm backward once per Learner process."""
    from liger_kernel.ops.rms_norm import calculate_settings, rms_norm_backward

    return calculate_settings, rms_norm_backward


def _frozen_rms_backward(dy, x, weight, inverse_rms):
    """Bounded input gradient with cached FP32 statistics and no frozen dW."""
    if x.is_cuda:
        calculate_settings, rms_norm_backward = _liger_rms_backend()
        block_size, num_warps = calculate_settings(x.shape[-1])
        # The pinned Liger needs_dw flag removes the unused per-SM dW reduction.
        dx, weight_grad = rms_norm_backward(
            dy.contiguous(), x.contiguous(), weight, inverse_rms.contiguous(),
            offset=1.0, casting_mode="gemma", BLOCK_SIZE=block_size,
            num_warps=num_warps, in_place=False, row_mode=False, needs_dw=False,
        )
        if weight_grad is not None:
            raise RuntimeError("frozen Qwen RMSNorm must not compute a weight gradient")
        return dx
    gain = 1.0 + weight.float()
    # CPU implementation for model assembly and numerical contract tests.
    xhat = x.float() * inverse_rms[:, None]
    weighted_dy = dy.float() * gain
    correction = (xhat * weighted_dy).mean(-1, keepdim=True)
    return ((weighted_dy - xhat * correction) * inverse_rms[:, None]).to(x.dtype)


class _RowwiseQwenNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, tile_rows, reference_fn, backward_fn):
        flat = x.reshape(-1, x.shape[-1])
        output = torch.empty(flat.shape, dtype=x.dtype, device=x.device)
        inverse_rms = torch.empty(flat.shape[0], dtype=torch.float32, device=x.device)
        for start in range(0, flat.shape[0], tile_rows):
            stop = min(start + tile_rows, flat.shape[0])
            tile_output, tile_inverse = reference_fn(
                flat[start:stop], weight, eps, return_inverse_rms=True)
            output[start:stop].copy_(tile_output)
            inverse_rms[start:stop].copy_(tile_inverse.reshape(-1))
        ctx.save_for_backward(flat, weight, inverse_rms)
        ctx.tile_rows, ctx.input_shape = tile_rows, x.shape
        ctx.backward_fn = backward_fn
        return output.reshape(x.shape)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        flat, weight, inverse_rms = ctx.saved_tensors
        dout = dy.reshape_as(flat)
        dx = torch.empty_like(flat)
        for start in range(0, flat.shape[0], ctx.tile_rows):
            stop = min(start + ctx.tile_rows, flat.shape[0])
            grad = ctx.backward_fn(dout[start:stop], flat[start:stop], weight,
                                   inverse_rms[start:stop])
            dx[start:stop].copy_(grad)
        return dx.reshape(ctx.input_shape), None, None, None, None, None



class BoundedQwenRMSNorm(nn.Module):
    def __init__(self, weight: nn.Parameter, eps: float, tile_rows: int | None = None):
        super().__init__()
        if (type(weight) is not nn.Parameter or weight.is_meta or weight.ndim != 1
                or weight.requires_grad or weight.dtype != torch.bfloat16):
            raise ValueError("bounded Qwen RMSNorm requires a materialized frozen BF16 Parameter")
        if isinstance(eps, bool) or not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("bounded Qwen RMSNorm epsilon must be finite and positive")
        if tile_rows is None:
            tile_rows = max(1, (4096 * 5120) // weight.numel())
        if type(tile_rows) is not int or tile_rows <= 0:
            raise ValueError("tile_rows must be a positive integer")
        self.weight, self.eps, self.tile_rows = weight, eps, tile_rows
        # None uses the normal original reference selected at call time;
        # regional compilation attaches an explicit per-model callable.
        self.reference_fn: Callable[..., torch.Tensor] | None = None
        self.backward_fn: Callable[..., torch.Tensor] | None = None

    def forward(self, x):
        reference_fn = self.reference_fn
        if reference_fn is None:
            reference_fn = _reference
        backward_fn = self.backward_fn
        if backward_fn is None:
            backward_fn = _frozen_rms_backward
        return _RowwiseQwenNorm.apply(
            x,
            self.weight,
            self.eps,
            self.tile_rows,
            reference_fn,
            backward_fn,
        )


def install_bounded_qwen_rms_norms(model: nn.Module) -> dict[str, Any]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    targets = []
    for parent in model.modules():
        for name, original in parent.named_children():
            if type(original) is Qwen3_5RMSNorm:
                fused = BoundedQwenRMSNorm(original.weight, original.eps)
                fused.train(original.training)
                targets.append((parent, name, fused))
    if not targets:
        raise ValueError("T2 model has no frozen Qwen3.5 RMSNorm modules")
    if any(fused.weight.is_cuda for _parent, _name, fused in targets):
        _liger_rms_backend()
    for parent, name, fused in targets:
        setattr(parent, name, fused)
    return {"implementation": "row-tiled FP32 Qwen forward with saved inverse RMS and pinned Liger frozen-weight CUDA backward", "recipe": RECIPE,
            "module_count": len(targets), "affine": "1 + original_weight.float()",
            "accumulation_dtype": "float32", "frozen_weight_identity_preserved": True,
            "full_activation_fp32_materialization": False, "saved_statistics_dtype": "float32",
            "saved_statistics_bytes_per_row": 4, "backward_compute_dW": False,
            "backward_kernel_commit": "ff8ce5857420cee06865f40ffa8a60309bf2b458",
            "backward_kernel_loaded_during_model_assembly": True,
            "max_tile_elements": 4096 * 5120}
