"""FFPA Triton attention with checkpoint-friendly saved tensor lifetimes.

The pinned FFPA backend is used directly so Q/K/V/output views stay views until
backward. This is equivalent to FFPA 0.2.4's Triton autograd path with its
FP32 gradient buffers, without its eager contiguous copies in forward.
"""
from __future__ import annotations

import torch


class _TernaryFFPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, scale):
        from ffpa_attn.triton import _ffpa_attn_forward_triton

        output, lse = _ffpa_attn_forward_triton(
            query, key, value, causal=True, softmax_scale=scale,
        )
        ctx.save_for_backward(query, key, value, output, lse)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from ffpa_attn.triton import _ffpa_attn_backward_triton

        query, key, value, output, lse = ctx.saved_tensors
        dq, dk, dv, _ = _ffpa_attn_backward_triton(
            grad_out=grad_output, q=query, k=key, v=value, o=output,
            lse=lse, causal=True, softmax_scale=ctx.scale,
            grad_q_storage_dtype=torch.float32,
            grad_kv_storage_dtype=torch.float32,
        )
        return dq, dk, dv, None


@torch.compiler.disable
def ternary_ffpa(query, key, value, scale):
    """Return attention in BHSD order; caller owns geometry validation."""
    return _TernaryFFPA.apply(query, key, value, scale)
