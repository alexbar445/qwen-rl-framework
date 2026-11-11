"""Train group scales over the shared immutable packed-T2 implementation.

Only this recovery module introduces trainable scales. RL's frozen buffers and
LoRA-only guards remain unchanged. Decoding and rotations use the same functions
as the production Learner, including future optimizations to those functions.
"""

from __future__ import annotations

import torch
from torch import nn

from .ternary import (
    T2_GROUP_SIZE,
    T2_PRIMAL_BASIS,
    FrozenTernaryEmbedding,
    FrozenTernaryLinear,
    _dequantize_t2_unchecked,
    _TernaryLinearFunction,
    hadamard_rows,
    validate_t2,
)


class PrimalTernaryLinear(nn.Linear):
    """Explicit unrotated T2 Use, using the same frozen linear/VJP implementation."""

    def __init__(
        self, codes, words, *, output_dtype=torch.bfloat16, compute_dtype=torch.bfloat16
    ):
        nn.Module.__init__(self)
        self.out_features, self.in_features = validate_t2(codes, words)
        self._t2_input_basis = T2_PRIMAL_BASIS
        self.output_dtype, self.compute_dtype = output_dtype, compute_dtype
        self.register_buffer("packed_codes", codes.detach().contiguous())
        self.register_buffer("scale_words", words.detach().contiguous())
        self.register_buffer("bias", None)

    def dequantize_weight(self, dtype=None):
        return decode(self.packed_codes, self.scale_words, dtype or self.compute_dtype)

    @property
    def weight(self):
        if getattr(self, "_peft_weight_metadata_only", False):
            return torch.empty(
                0, device=self.packed_codes.device, dtype=self.output_dtype
            )
        return self.dequantize_weight(self.output_dtype)

    def rotate_inputs(self, inputs):
        return inputs

    def forward(self, inputs):
        return _TernaryLinearFunction.apply(
            inputs, self.packed_codes, self.scale_words, self.bias, self.compute_dtype
        )


def effective_words(base, delta):
    if delta.dtype != torch.float32:
        raise TypeError("recovery log-scales must remain FP32")
    scales = base.contiguous().view(torch.float16).float() * delta.float().exp()
    rounded = scales.to(torch.float16)
    if not bool(torch.isfinite(rounded).all()):
        raise FloatingPointError("recovery scale overflows FP16")
    if bool(((scales != 0) & (rounded == 0)).any()):
        raise FloatingPointError("recovery scale underflows FP16")
    return rounded.contiguous().view(torch.uint16)


def decode(codes, words, dtype):
    return _dequantize_t2_unchecked(
        codes, words, codes.shape[0], codes.shape[1] * 4, dtype
    )


def scale_vjp(codes, base_words, delta, grad_weight):
    # Straight-through derivative of FP16 scale and compute-dtype rounding.
    units = torch.ones_like(base_words, dtype=torch.float16).view(torch.uint16)
    trits = decode(codes, units, torch.float32)
    unrounded = (
        base_words.contiguous().view(torch.float16).float() * delta.float().exp()
    )
    return (grad_weight.float() * trits).reshape(*unrounded.shape, T2_GROUP_SIZE).sum(
        -1
    ) * unrounded


class _ScaledLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, base):
        ctx.base = base
        ctx.save_for_backward(x, delta)
        weight = decode(
            base.packed_codes,
            effective_words(base.scale_words, delta),
            base.compute_dtype,
        )
        return torch.nn.functional.linear(
            x.to(base.compute_dtype), weight, base.bias
        ).to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        x, delta = ctx.saved_tensors
        base = ctx.base
        dx = None
        if ctx.needs_input_grad[0]:
            weight = decode(
                base.packed_codes,
                effective_words(base.scale_words, delta),
                base.compute_dtype,
            )
            dx = dy.to(base.compute_dtype).matmul(weight).to(x.dtype)
            del weight
        ds = torch.empty_like(delta)
        xf = x.reshape(-1, x.shape[-1]).to(base.compute_dtype)
        df = dy.reshape(-1, dy.shape[-1]).to(base.compute_dtype)
        for begin in range(0, base.out_features, 256):
            end = min(begin + 256, base.out_features)
            dw = df[:, begin:end].T.matmul(xf)
            ds[begin:end] = scale_vjp(
                base.packed_codes[begin:end],
                base.scale_words[begin:end],
                delta[begin:end],
                dw,
            )
        return dx, ds, None


class ScaleTernaryLinear(nn.Module):
    """Own a small FP32 log-ratio parameter and reuse the original frozen owner."""

    def __init__(self, base: FrozenTernaryLinear):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.log_scale_delta = nn.Parameter(
            torch.zeros(
                base.scale_words.shape,
                dtype=torch.float32,
                device=base.scale_words.device,
            )
        )

    def forward(self, x):
        return _ScaledLinear.apply(
            self.base.rotate_inputs(x), self.log_scale_delta, self.base
        )

    def rows(self, begin, end, dtype):
        words = effective_words(
            self.base.scale_words[begin:end], self.log_scale_delta[begin:end]
        )
        return decode(self.base.packed_codes[begin:end], words, dtype)

    def exported_scale_words(self):
        return effective_words(self.base.scale_words, self.log_scale_delta)


class _EmbeddingScales(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, delta, base):
        flat = ids.reshape(-1).long()
        ctx.base = base
        ctx.save_for_backward(flat, delta)
        codes = base.packed_codes.index_select(0, flat)
        words = effective_words(
            base.scale_words.index_select(0, flat), delta.index_select(0, flat)
        )
        return decode(codes, words, torch.float32)

    @staticmethod
    def backward(ctx, grad):
        ids, delta = ctx.saved_tensors
        base = ctx.base
        result = torch.zeros_like(delta)
        for begin in range(0, ids.numel(), 256):
            ix = ids[begin : begin + 256]
            local = scale_vjp(
                base.packed_codes.index_select(0, ix),
                base.scale_words.index_select(0, ix),
                delta.index_select(0, ix),
                grad[begin : begin + 256],
            )
            result.index_add_(0, ix, local)
        return None, result, None


class ScaleTernaryEmbedding(nn.Module):
    def __init__(self, base: FrozenTernaryEmbedding):
        super().__init__()
        self.base = base
        self.log_scale_delta = nn.Parameter(
            torch.zeros(
                base.scale_words.shape,
                dtype=torch.float32,
                device=base.scale_words.device,
            )
        )

    def forward(self, ids):
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("embedding IDs must be integers")
        if bool(((ids < 0) | (ids >= self.base.num_embeddings)).any()):
            raise IndexError("embedding ID out of range")
        rotated = _EmbeddingScales.apply(ids, self.log_scale_delta, self.base)
        value = hadamard_rows(
            rotated,
            self.base.hadamard_signs,
            inverse=True,
            output_dtype=self.base.output_dtype,
        )
        if self.base.padding_idx is not None:
            value = value.masked_fill(ids.reshape(-1, 1) == self.base.padding_idx, 0)
        return value.reshape(*ids.shape, self.base.embedding_dim)

    def exported_scale_words(self):
        return effective_words(self.base.scale_words, self.log_scale_delta)
