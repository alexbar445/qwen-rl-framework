"""Single-launch Hadamard with exact ascending/descending FP32 stages.

Used by the managed T2 runtime. Reads arbitrary leading strides directly;
only the result is allocated, and each program owns one 1024-element block.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _transform(X, S, Y, WIDTH: tl.constexpr, SHAPE: tl.constexpr,
               STRIDES: tl.constexpr, INVERSE: tl.constexpr, TRANSPOSE: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, 1024)
    logical = block * 1024 + lane
    row = logical // WIDTH
    column = logical % WIDTH
    offset = column * STRIDES[-1]
    for dimension in tl.static_range(len(SHAPE) - 2, -1, -1):
        if dimension == 0:
            offset += row * STRIDES[0]
        else:
            offset += (row % SHAPE[dimension]) * STRIDES[dimension]
            row = row // SHAPE[dimension]
    x = tl.load(X + offset).to(tl.float32)
    signs = tl.load(S + column).to(tl.float32)
    if TRANSPOSE:
        if INVERSE:
            x = x * signs
        x = x * 0.03125
    elif not INVERSE:
        x = x * signs
    for stage in tl.static_range(10):
        stride = (512 >> stage) if TRANSPOSE else (1 << stage)
        peer = tl.gather(x, lane ^ stride, axis=0)
        x = tl.where((lane & stride) == 0, x + peer, peer - x)
    if not TRANSPOSE:
        x = x * 0.03125
        if INVERSE:
            x = x * signs
    elif not INVERSE:
        x = x * signs
    tl.store(Y + logical, x)


def hadamard(values, signs, *, inverse=False, transpose=False, output_dtype=None):
    if not values.is_cuda or values.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('requires CUDA BF16 or FP32 values')
    if values.ndim < 1 or values.shape[-1] <= 0 or values.shape[-1] % 1024:
        raise ValueError('requires whole 1024-element blocks')
    if (signs.device != values.device or signs.shape != (values.shape[-1],)
            or not signs.is_contiguous() or signs.requires_grad):
        raise ValueError('requires frozen contiguous device-local signs matching width')
    output_dtype = values.dtype if output_dtype is None else output_dtype
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('requires BF16 or FP32 output')
    output = torch.empty(values.shape, dtype=output_dtype, device=values.device)
    if values.numel():
        # Flatten contiguous leading dimensions without copying. The leading row
        # count is absent from the JIT key, so trajectory tails reuse this kernel.
        selected = values.view(-1, values.shape[-1]) if values.is_contiguous() else values
        metadata_shape = (0, *selected.shape[1:])
        _transform[(values.numel() // 1024,)](
            selected, signs, output, values.shape[-1], metadata_shape,
            selected.stride(), inverse, transpose, num_warps=4, enable_fp_fusion=False,
        )
    return output


def forward_contiguous(values, signs, *, inverse=False):
    """Use the existing warp kernel once when selection needs no scratch copy."""
    from qwen_rl.models.ternary_fused_hadamard import _load_kernel

    if (not values.is_cuda or values.dtype != torch.bfloat16 or not values.is_contiguous()
            or values.ndim < 1 or values.shape[-1] <= 0 or values.shape[-1] % 1024
            or values.data_ptr() % 16 or signs.device != values.device
            or signs.dtype != torch.int8 or signs.requires_grad
            or signs.shape != (values.shape[-1],) or not signs.is_contiguous()):
        raise ValueError('requires aligned contiguous CUDA BF16 input and device-local frozen int8 signs')
    output = torch.empty(values.shape, device=values.device, dtype=values.dtype)
    if values.numel():
        width = values.shape[-1]
        _load_kernel()(values.view(-1, width), signs.to(torch.bfloat16), inverse,
                       out=output.view(-1, width))
    return output
