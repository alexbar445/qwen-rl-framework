"""Frozen T2 ternary weights with recomputed input gradients and PEFT-compatible linears.

Each packed Use declares whether its input is primal or sign/Hadamard rotated. Rotated linears
consume the artifact's ``W'`` basis; primal linears consume the original activations. PEFT LoRA-A
uses the same per-target basis as its frozen base.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

T2_GROUP_SIZE = 128
T2_CODES_PER_BYTE = 4
HADAMARD_BLOCK = 1024
T2_DEQUANTIZE_ROW_CHUNK = 1024
T2_PRIMAL_BASIS = "primal"
T2_ROTATED_BASIS = "prism_rotated"


def _as_scale_words(scales: torch.Tensor) -> torch.Tensor:
    if scales.dtype == torch.uint16:
        return scales.detach().contiguous()
    if scales.dtype == torch.float16:
        return scales.detach().contiguous().view(torch.uint16)
    raise TypeError("T2 scales must be raw uint16 words or float16 values")


def validate_t2(packed_codes: torch.Tensor, scale_words: torch.Tensor) -> tuple[int, int]:
    """Validate packed T2 storage and return ``(rows, input_features)``.

    Scale words remain uint16 in module state so ``Module.to(dtype=...)`` cannot round the
    artifact's exact FP16 multipliers through BF16 or FP32.
    """

    rows, in_features = _validate_t2_shapes(packed_codes, scale_words)
    if (packed_codes.device.type == "cpu" and packed_codes.layout == torch.strided
            and scale_words.layout == torch.strided
            and not packed_codes.is_neg() and not scale_words.is_neg()):
        from .ternary_validation import validate_t2_cpu

        validate_t2_cpu(packed_codes, scale_words)
        return rows, in_features
    # Validate in bounded row chunks. Expanding the 27B embedding table to one int16 field per
    # weight just to check its language would use several GiB of temporary memory.
    for begin in range(0, rows, 8192):
        code_chunk = packed_codes[begin : begin + 8192]
        high = (code_chunk & 0xAA) >> 1
        low = code_chunk & 0x55
        if bool((high & (~low & 0x55)).any()):
            raise ValueError("T2 code 0b10 is invalid; legal two's-complement codes are {-1, 0, +1}")
        scale_chunk = scale_words[begin : begin + 8192].contiguous().view(torch.float16)
        if not bool(torch.isfinite(scale_chunk).all()):
            raise ValueError("T2 scales must be finite FP16 values")
    return rows, in_features


def _validate_t2_shapes(
    packed_codes: torch.Tensor, scale_words: torch.Tensor
) -> tuple[int, int]:
    if scale_words.dtype == torch.float16:
        scale_words = scale_words.contiguous().view(torch.uint16)
    if packed_codes.dtype != torch.uint8 or packed_codes.ndim != 2:
        raise TypeError("packed T2 codes must be a uint8 matrix [out_features, in_features / 4]")
    if scale_words.dtype != torch.uint16 or scale_words.ndim != 2:
        raise TypeError("packed T2 scale words must be a uint16 matrix [out_features, in_features / 128]")
    rows, code_bytes = packed_codes.shape
    if rows <= 0 or code_bytes <= 0:
        raise ValueError("packed T2 dimensions must be positive")
    in_features = code_bytes * T2_CODES_PER_BYTE
    if in_features % T2_GROUP_SIZE:
        raise ValueError("T2 input width must be divisible by 128")
    expected_scales = (rows, in_features // T2_GROUP_SIZE)
    if tuple(scale_words.shape) != expected_scales:
        raise ValueError(f"T2 scales must have shape {expected_scales}, got {tuple(scale_words.shape)}")
    if packed_codes.device != scale_words.device:
        raise ValueError("T2 codes and scales must be on the same device")
    return rows, in_features


def _unpack_t2_fields(packed_codes: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(4, device=packed_codes.device, dtype=torch.int16) * 2
    return ((packed_codes.to(torch.int16).unsqueeze(-1) >> shifts) & 0x3).reshape(
        packed_codes.shape[0], packed_codes.shape[1] * T2_CODES_PER_BYTE
    )


def dequantize_t2(
    packed_codes: torch.Tensor,
    scale_words: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    row_chunk_size: int = T2_DEQUANTIZE_ROW_CHUNK,
) -> torch.Tensor:
    """Decode the NInfer row-split T2 payload into a temporary dense matrix.

    The codes are little-endian 2-bit two's-complement fields: ``00 -> 0``,
    ``01 -> +1``, ``11 -> -1``; ``10`` is rejected. Each contiguous 128-column
    group uses its exact stored binary16 scale.
    """

    if scale_words.dtype == torch.float16:
        scale_words = scale_words.contiguous().view(torch.uint16)
    rows, in_features = validate_t2(packed_codes, scale_words)
    return _dequantize_t2_unchecked(
        packed_codes,
        scale_words,
        rows,
        in_features,
        dtype,
        row_chunk_size=row_chunk_size,
    )


def _dequantize_t2_unchecked(
    packed_codes: torch.Tensor,
    scale_words: torch.Tensor,
    rows: int,
    in_features: int,
    dtype: torch.dtype,
    *,
    row_chunk_size: int = T2_DEQUANTIZE_ROW_CHUNK,
) -> torch.Tensor:
    """Decode into one final dense tensor with row-bounded temporary storage.

    The output itself is ``rows * K * element_size(dtype)``. All code expansion, signed-code,
    scale repetition, and multiply temporaries are limited to ``row_chunk_size * K`` elements;
    the default bounds a 17,408-wide projection's FP32 scratch to well under 512 MiB and avoids
    multi-GiB intermediates for the vocabulary matrix. The caller may lower the chunk size on
    tighter devices or raise it to reduce launch overhead where memory allows.
    """

    if type(row_chunk_size) is not int or row_chunk_size <= 0:
        raise ValueError("T2 dequantization row_chunk_size must be a positive integer")
    if not dtype.is_floating_point:
        raise TypeError("T2 dense output dtype must be floating point")
    if packed_codes.is_cuda:
        from .ternary_dequant import dequantize_t2_cuda

        return dequantize_t2_cuda(
            packed_codes, scale_words, dtype=dtype, row_chunk_size=row_chunk_size
        )
    output = torch.empty((rows, in_features), dtype=dtype, device=packed_codes.device)
    for begin in range(0, rows, row_chunk_size):
        end = min(rows, begin + row_chunk_size)
        fields = _unpack_t2_fields(packed_codes[begin:end])
        magnitude = (fields & 1).to(torch.int16)
        negative = (fields & 2) != 0
        signed_codes = torch.where(negative, -magnitude, magnitude)
        scales = (
            scale_words[begin:end]
            .contiguous()
            .view(torch.float16)
            .to(dtype=dtype)
            .repeat_interleave(T2_GROUP_SIZE, dim=1)
        )
        output[begin:end] = signed_codes.to(dtype=dtype) * scales
    return output


_HADAMARD_TILE_ELEMENTS = 16 * 1024 * 1024  # at most 64 MiB per FP32 working tile


def _select_flat_hadamard_rows(values: torch.Tensor, begin: int, end: int) -> torch.Tensor:
    """Select logical flattened rows without materializing a full noncontiguous input."""
    width = values.shape[-1]
    if values.is_contiguous():
        return values.view(-1, width)[begin:end]
    if values.ndim == 1:
        return values.unsqueeze(0)

    flat_indices = torch.arange(begin, end, device=values.device, dtype=torch.long)
    coordinates: list[torch.Tensor] = [flat_indices] * (values.ndim - 1)
    for dimension in range(values.ndim - 2, -1, -1):
        size = values.shape[dimension]
        coordinates[dimension] = flat_indices.remainder(size)
        flat_indices = torch.div(flat_indices, size, rounding_mode="floor")
    # Advanced indexing creates only this bounded tile, even when flattening the leading
    # dimensions with reshape() would copy the complete noncontiguous activation.
    return values[tuple(coordinates)]


def _hadamard_rows_impl(
    values: torch.Tensor, signs: torch.Tensor, *, inverse: bool,
    block_size: int, output_dtype: torch.dtype,
    compute_dtype: torch.dtype | None = None, transpose: bool = False,
) -> torch.Tensor:
    accumulation = compute_dtype or (torch.float64 if values.dtype == torch.float64 else torch.float32)
    if values.is_cuda and accumulation == torch.float32 and block_size == HADAMARD_BLOCK:
        if (values.dtype == torch.bfloat16 and output_dtype == torch.bfloat16
                and signs.dtype == torch.int8 and not transpose):
            from .ternary_fused_hadamard import fused_hadamard_rows

            return fused_hadamard_rows(
                values, signs, inverse=inverse, block_size=block_size,
                output_dtype=output_dtype, compute_dtype=accumulation, transpose=False,
            )
        if (values.dtype in (torch.bfloat16, torch.float32)
                and output_dtype in (torch.bfloat16, torch.float32)
                and signs.device == values.device and signs.is_contiguous()
                and not signs.requires_grad):
            from .ternary_hadamard_launch import hadamard

            return hadamard(values, signs, inverse=inverse, transpose=transpose,
                            output_dtype=output_dtype)
        from .ternary_hadamard import dao_hadamard_rows

        return dao_hadamard_rows(
            values, signs, inverse=inverse, block_size=block_size,
            output_dtype=output_dtype, compute_dtype=accumulation, transpose=transpose,
        )
    # Portable CPU, FP64 oracle and other explicitly requested block sizes.
    return _hadamard_rows_torch_impl(
        values, signs, inverse=inverse, block_size=block_size,
        output_dtype=output_dtype, compute_dtype=accumulation, transpose=transpose,
    )


def _hadamard_rows_torch_impl(
    values: torch.Tensor,
    signs: torch.Tensor,
    *,
    inverse: bool,
    block_size: int,
    output_dtype: torch.dtype,
    compute_dtype: torch.dtype | None = None,
    transpose: bool = False,
) -> torch.Tensor:
    width = values.shape[-1]
    compute_dtype = compute_dtype or (
        torch.float64 if values.dtype == torch.float64 else torch.float32
    )
    row_count = values.numel() // width
    rows_per_tile = max(1, _HADAMARD_TILE_ELEMENTS // width)
    result = torch.empty(values.shape, device=values.device, dtype=output_dtype)
    output_rows = result.view(row_count, width)
    sign_values = signs.to(device=values.device, dtype=compute_dtype).reshape(1, width)
    normalization = block_size**-0.5

    for begin in range(0, row_count, rows_per_tile):
        end = min(row_count, begin + rows_per_tile)
        selected = _select_flat_hadamard_rows(values, begin, end)
        # Own one contiguous tile even when the input already has the compute dtype;
        # the in-place butterfly must never modify the caller's activation.
        working = selected.to(dtype=compute_dtype, copy=True, memory_format=torch.contiguous_format)

        if transpose:
            # Match the original autograd graph's reverse ordering exactly: scalar/sign
            # multiplies are applied where backward encounters them, then butterfly stages
            # run in descending stride order.
            if inverse:
                working.mul_(sign_values)
            working.mul_(normalization)
        elif not inverse:
            working.mul_(sign_values)

        transformed = working.reshape(working.shape[0], -1, block_size)
        strides: list[int] = []
        stride = 1
        while stride < block_size:
            strides.append(stride)
            stride *= 2
        if transpose:
            strides.reverse()
        for stride in strides:
            pairs = transformed.reshape(transformed.shape[0], -1, 2, stride)
            low = pairs[:, :, 0, :]
            high = pairs[:, :, 1, :]
            difference = low - high
            low.add_(high)
            high.copy_(difference)
            del difference

        if not transpose:
            working.mul_(normalization)
            if inverse:
                working.mul_(sign_values)
        elif not inverse:
            working.mul_(sign_values)
        output_rows[begin:end].copy_(working.reshape(-1, width))
    return result


class _HadamardRowsFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,
        values: torch.Tensor,
        signs: torch.Tensor,
        inverse: bool,
        block_size: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.save_for_backward(signs)
        ctx.inverse = bool(inverse)
        ctx.block_size = int(block_size)
        ctx.input_dtype = values.dtype
        ctx.compute_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        return _hadamard_rows_impl(
            values,
            signs,
            inverse=ctx.inverse,
            block_size=ctx.block_size,
            output_dtype=output_dtype,
            compute_dtype=ctx.compute_dtype,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (signs,) = ctx.saved_tensors
        grad_input = _hadamard_rows_impl(
            grad_output,
            signs,
            inverse=ctx.inverse,
            block_size=ctx.block_size,
            output_dtype=ctx.input_dtype,
            compute_dtype=ctx.compute_dtype,
            transpose=True,
        )
        return grad_input, None, None, None, None


def hadamard_rows(
    values: torch.Tensor,
    signs: torch.Tensor,
    *,
    inverse: bool = False,
    block_size: int = HADAMARD_BLOCK,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply Bonsai's normalized Sylvester transform along the last dimension.

    Forward: ``H(s * x)``. Inverse: ``s * H(x)``. ``H`` is blockwise, symmetric,
    and involutory. FP64 inputs accumulate in FP64; all other inputs accumulate in FP32.
    Work is tiled across flattened rows so no FP32 activation or butterfly scratch tensor
    exceeds ``_HADAMARD_TILE_ELEMENTS`` elements. Backward applies the transpose transform
    and retains only the small sign vector.
    """

    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block_size must be a positive power of two")
    if signs.requires_grad:
        raise ValueError("Hadamard signs are frozen artifact buffers")
    if values.ndim < 1 or values.shape[-1] <= 0 or values.shape[-1] % block_size:
        raise ValueError(f"Hadamard input width must be a positive multiple of {block_size}")
    if signs.ndim != 1 or tuple(signs.shape) != (values.shape[-1],):
        raise ValueError(f"Hadamard signs must have shape ({values.shape[-1]},)")
    if not values.is_floating_point():
        raise TypeError("Hadamard input must be floating point")

    return _HadamardRowsFunction.apply(
        values,
        signs,
        inverse,
        block_size,
        output_dtype if output_dtype is not None else values.dtype,
    )

class _TernaryLinearFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        packed_codes: torch.Tensor,
        scale_words: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
        activation_cache: Any = None,
        decoded_weight_lease: Any = None,
    ) -> torch.Tensor:
        if not inputs.is_floating_point():
            raise TypeError("T2 linear inputs must be floating point")
        if inputs.device != packed_codes.device:
            raise ValueError("T2 linear input and packed weights must be on the same device")
        rows, in_features = packed_codes.shape[0], packed_codes.shape[1] * T2_CODES_PER_BYTE
        if inputs.shape[-1] != in_features:
            raise ValueError(f"T2 linear expected input width {in_features}, got {inputs.shape[-1]}")
        _validate_t2_shapes(packed_codes, scale_words)
        ctx.save_for_backward(packed_codes, scale_words)
        ctx.input_dtype = inputs.dtype
        ctx.compute_dtype = compute_dtype
        ctx.bias_present = bias is not None
        ctx.needs_input_gradient = ctx.needs_input_grad[0]
        ctx.decoded_weight_lease = decoded_weight_lease

        # Keep this Function's original saved tensors and VJP on cache hits.
        # Returning a module output directly from a replay hook would detach
        # the recomputed graph and lose the base's input gradient.
        if activation_cache is not None:
            cached = activation_cache.restore(inputs)
            if cached is not None:
                return cached

        weight = (decoded_weight_lease.get(packed_codes, scale_words, rows, in_features, ctx.compute_dtype)
                  if decoded_weight_lease is not None else
                  _dequantize_t2_unchecked(packed_codes, scale_words, rows, in_features, ctx.compute_dtype))
        dense_bias = bias.to(ctx.compute_dtype) if bias is not None else None
        output = F.linear(inputs.to(ctx.compute_dtype), weight, dense_bias)
        output = output.to(inputs.dtype)
        if activation_cache is not None:
            activation_cache.save(inputs, output)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        packed_codes, scale_words = ctx.saved_tensors
        grad_input = None
        grad_bias = None
        if ctx.needs_input_gradient:
            in_features = packed_codes.shape[1] * T2_CODES_PER_BYTE
            weight = (ctx.decoded_weight_lease.get(
                packed_codes, scale_words, packed_codes.shape[0], in_features, ctx.compute_dtype
            ) if ctx.decoded_weight_lease is not None else _dequantize_t2_unchecked(
                packed_codes, scale_words, packed_codes.shape[0], in_features, ctx.compute_dtype
            ))
            grad_input = grad_output.to(ctx.compute_dtype).matmul(weight).to(ctx.input_dtype)
        if ctx.bias_present and ctx.needs_input_grad[3]:
            reduction_dims = tuple(range(grad_output.ndim - 1))
            grad_bias = (
                grad_output.to(ctx.compute_dtype).sum(dim=reduction_dims)
                if reduction_dims
                else grad_output.to(ctx.compute_dtype)
            )
        return (grad_input, None, None, grad_bias, None, None, None)[:len(ctx.needs_input_grad)]


class FrozenTernaryLinear(nn.Linear):
    """PEFT-compatible frozen ternary linear that never keeps a dense base weight.

    It accepts a primal-basis input, applies the Bonsai sign/Hadamard transform, and multiplies
    by the artifact's rotated ``W'``. The backward pass saves only packed buffers and recomputes
    the decoded matrix only when an input gradient is required. Production compute defaults to
    BF16; parity/oracle callers should pass ``compute_dtype=torch.float32`` explicitly.
    """

    def __init__(
        self,
        packed_codes: torch.Tensor,
        scales: torch.Tensor,
        hadamard_signs: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        nn.Module.__init__(self)
        scale_words = _as_scale_words(scales)
        rows, in_features = validate_t2(packed_codes, scale_words)
        if tuple(hadamard_signs.shape) != (in_features,):
            raise ValueError(f"T2 input Hadamard signs must have shape ({in_features},)")
        if not bool(torch.all((hadamard_signs == 1) | (hadamard_signs == -1))):
            raise ValueError("T2 input Hadamard signs must contain only +1 and -1")
        self.in_features = int(in_features)
        self.out_features = int(rows)
        self._t2_input_basis = T2_ROTATED_BASIS
        self.output_dtype = output_dtype
        if compute_dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            raise TypeError("T2 compute_dtype must be BF16, FP16, FP32 or FP64")
        self.compute_dtype = compute_dtype
        self.register_buffer("packed_codes", packed_codes.detach().contiguous().clone(), persistent=True)
        self.register_buffer("scale_words", scale_words.clone(), persistent=True)
        self.register_buffer(
            "input_hadamard_signs",
            torch.where(hadamard_signs > 0, 1, -1).to(torch.int8).contiguous().clone(),
            persistent=True,
        )
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            if tuple(bias.shape) != (rows,) or bias.device != packed_codes.device:
                raise ValueError(f"T2 bias must have shape ({rows},) on the weight device")
            if not bias.is_floating_point():
                raise TypeError("T2 bias must be floating point")
            self.register_buffer("bias", bias.detach().contiguous().clone(), persistent=True)

    def dequantize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return _dequantize_t2_unchecked(
            self.packed_codes,
            self.scale_words,
            self.out_features,
            self.in_features,
            dtype,
        )

    def rotate_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the artifact-compatible primal-to-W' activation basis ``H(s*x)``."""

        from .ternary_shared_rotation import rotate_base
        return rotate_base(self, inputs)

    @property
    def weight(self) -> torch.Tensor:
        """Transient dense view; adapter setup can request metadata without decoding it."""

        if getattr(self, "_peft_weight_metadata_only", False):
            return torch.empty(0, device=self.packed_codes.device, dtype=self.output_dtype)
        return self.dequantize_weight(self.output_dtype)

    def forward(self, inputs: torch.Tensor, *, decoded_weight_lease=None) -> torch.Tensor:
        rotated = self.rotate_inputs(inputs)
        bias = self.bias
        if bias is not None and bias.dtype != inputs.dtype:
            bias = bias.to(dtype=inputs.dtype)
        arguments = (
            rotated,
            self.packed_codes,
            self.scale_words,
            bias,
            self.compute_dtype,
        )
        cache = getattr(self, "_activation_cache_site", None)
        if decoded_weight_lease is not None:
            return _TernaryLinearFunction.apply(*arguments, cache, decoded_weight_lease)
        return _TernaryLinearFunction.apply(*arguments, cache) if cache is not None else _TernaryLinearFunction.apply(*arguments)


def t2_linear_input_basis(module: nn.Module) -> str | None:
    """Return the declared input basis for a packed T2 linear, if recognized."""
    basis = getattr(module, "_t2_input_basis", None)
    if basis is None:
        if isinstance(module, FrozenTernaryLinear):
            return T2_ROTATED_BASIS
        return None
    if basis not in {T2_PRIMAL_BASIS, T2_ROTATED_BASIS}:
        raise ValueError(f"unsupported T2 linear input basis: {basis!r}")
    return basis


def is_t2_linear(module: nn.Module) -> bool:
    """Whether ``module`` is a frozen packed T2 linear with an explicit basis."""
    return t2_linear_input_basis(module) is not None


def _canonical_lora_target_name(name: str) -> str:
    return name.removeprefix("base_model.model.")


def _is_frozen_dense_gdn_projection(name: str, module: nn.Module) -> bool:
    """Recognize only the primal dense A/B inputs in Qwen3.5's GDN blocks."""
    parts = _canonical_lora_target_name(name).split(".")
    if not (
        len(parts) == 5
        and parts[0] == "model"
        and parts[1] == "layers"
        and parts[2].isdecimal()
        and parts[3] == "linear_attn"
        and parts[4] in {"in_proj_a", "in_proj_b"}
    ):
        return False
    if not isinstance(module, nn.Linear) or module.bias is not None:
        raise TypeError(f"dense GDN primal target {name} must be a bias-free nn.Linear")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError(f"dense GDN base projection {name} must remain frozen")
    return True


class FrozenTernaryEmbedding(nn.Module):
    """Packed T2 token table; decode only requested rows and restore the primal basis."""

    def __init__(
        self,
        packed_codes: torch.Tensor,
        scales: torch.Tensor,
        hadamard_signs: torch.Tensor,
        *,
        padding_idx: int | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        scale_words = _as_scale_words(scales)
        rows, width = validate_t2(packed_codes, scale_words)
        if width % HADAMARD_BLOCK:
            raise ValueError(f"Bonsai hidden width must be divisible by {HADAMARD_BLOCK}")
        if tuple(hadamard_signs.shape) != (width,):
            raise ValueError(f"embedding Hadamard signs must have shape ({width},)")
        if hadamard_signs.dtype not in (torch.int8, torch.int16, torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("embedding Hadamard signs must be signed integer or floating point")
        if not bool(torch.all((hadamard_signs == 1) | (hadamard_signs == -1))):
            raise ValueError("embedding Hadamard signs must contain only +1 and -1")
        if padding_idx is not None and not -rows <= padding_idx < rows:
            raise ValueError("padding_idx is outside the embedding table")
        self.num_embeddings = int(rows)
        self.embedding_dim = int(width)
        self.padding_idx = padding_idx % rows if padding_idx is not None else None
        self.output_dtype = output_dtype
        self.register_buffer("packed_codes", packed_codes.detach().contiguous().clone(), persistent=True)
        self.register_buffer("scale_words", scale_words.clone(), persistent=True)
        self.register_buffer(
            "hadamard_signs",
            torch.where(hadamard_signs > 0, 1, -1).to(torch.int8).contiguous().clone(),
            persistent=True,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("embedding token IDs must use int32 or int64")
        if input_ids.device != self.packed_codes.device:
            raise ValueError("token IDs and packed T2 embedding must be on the same device")
        if input_ids.numel() and bool(((input_ids < 0) | (input_ids >= self.num_embeddings)).any()):
            raise IndexError("embedding token ID is outside the table")
        flat = input_ids.reshape(-1).to(torch.long)
        codes = self.packed_codes.index_select(0, flat)
        scales = self.scale_words.index_select(0, flat)
        if flat.numel() == 0:
            return torch.empty(
                (*input_ids.shape, self.embedding_dim),
                dtype=self.output_dtype,
                device=input_ids.device,
            )
        rotated = _dequantize_t2_unchecked(
            codes, scales, flat.numel(), self.embedding_dim, torch.float32
        )
        primal = hadamard_rows(rotated, self.hadamard_signs, inverse=True, output_dtype=self.output_dtype)
        if self.padding_idx is not None and input_ids.numel():
            primal = primal.masked_fill(input_ids.reshape(-1, 1) == self.padding_idx, 0)
        return primal.reshape(*input_ids.shape, self.embedding_dim)


class _HadamardLoRAA(nn.Linear):
    """Standard PEFT A projection whose input is transformed into the T2 basis."""

    def __init__(self, source: nn.Linear, signs: torch.Tensor) -> None:
        nn.Module.__init__(self)
        self.in_features = int(source.in_features)
        self.out_features = int(source.out_features)
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer(
            "input_hadamard_signs",
            torch.where(signs > 0, 1, -1).to(torch.int8).contiguous().clone(),
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        from .ternary_shared_rotation import rotate_adapter
        rotated = rotate_adapter(self, inputs)
        return F.linear(rotated, self.weight, self.bias)


@contextmanager
def peft_weight_metadata_only(model: nn.Module):
    """Let PEFT read frozen base dtype/device without materializing any dense T2 matrix."""

    modules = [module for module in model.modules() if is_t2_linear(module)]
    for module in modules:
        module._peft_weight_metadata_only = True
    try:
        yield
    finally:
        for module in modules:
            module._peft_weight_metadata_only = False


def attach_ternary_peft(
    model: nn.Module,
    lora_config: Any,
    *,
    input_basis_by_module: Mapping[str, str] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Inject standard PEFT LoRA with memory-bounded setup and the T2 rotation contract."""

    from peft import get_peft_model
    from peft.tuners.lora.layer import LoraLayer

    with peft_weight_metadata_only(model):
        peft_model = get_peft_model(model, lora_config)
    lora_targets = [
        (name, module)
        for name, module in peft_model.named_modules()
        if name and isinstance(module, LoraLayer)
    ]
    if not lora_targets:
        raise RuntimeError("PEFT selected no LoRA target modules")
    expected_basis = None
    if input_basis_by_module is not None:
        if not isinstance(input_basis_by_module, Mapping) or not input_basis_by_module:
            raise ValueError("T2 LoRA input-basis identity must be a nonempty mapping")
        expected_basis = dict(input_basis_by_module)
        for name, basis in expected_basis.items():
            if not isinstance(name, str) or not name:
                raise ValueError("T2 LoRA input-basis module names must be nonempty strings")
            if basis not in {T2_PRIMAL_BASIS, T2_ROTATED_BASIS}:
                raise ValueError(f"unsupported T2 LoRA input basis for {name}: {basis!r}")
        actual_names = [_canonical_lora_target_name(name) for name, _module in lora_targets]
        if len(actual_names) != len(set(actual_names)):
            raise RuntimeError("PEFT selected duplicate canonical T2 LoRA target names")
        if set(actual_names) != set(expected_basis):
            missing = sorted(set(expected_basis) - set(actual_names))
            extra = sorted(set(actual_names) - set(expected_basis))
            raise RuntimeError(
                "PEFT target inventory differs from the T2 input-basis identity; "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
    dropout_values = set()
    for name, module in lora_targets:
        for adapter_name in tuple(module.lora_A.keys()):
            dropout = module.lora_dropout[adapter_name]
            p = float(getattr(dropout, "p", 0.0))
            dropout_values.add(p)
            if p != 0.0:
                raise ValueError(
                    f"Bonsai T2 LoRA requires lora_dropout=0 for rotated-basis parity ({name})"
                )
    rotated_lora_a = install_lora_hadamard_inputs(peft_model)
    basis_inventory = {}
    for name, module in peft_model.named_modules():
        if not name or not isinstance(module, LoraLayer):
            continue
        base = module.get_base_layer()
        basis = t2_linear_input_basis(base)
        if basis is None and _is_frozen_dense_gdn_projection(name, base):
            basis = T2_PRIMAL_BASIS
        if basis is None:
            raise RuntimeError(
                f"T2 LoRA target {name} is neither a packed linear nor a supported dense GDN projection"
            )
        canonical_name = _canonical_lora_target_name(name)
        if expected_basis is not None and expected_basis[canonical_name] != basis:
            raise RuntimeError(
                f"T2 LoRA input basis for {canonical_name} is {basis}, "
                f"identity declares {expected_basis[canonical_name]}"
            )
        basis_inventory[canonical_name] = basis
    if len(basis_inventory) != len(lora_targets):
        raise RuntimeError("every T2 LoRA target must have a supported explicit input basis")
    primal_lora_a = sum(basis == T2_PRIMAL_BASIS for basis in basis_inventory.values())
    if rotated_lora_a + primal_lora_a != len(basis_inventory):
        raise RuntimeError("T2 LoRA input basis inventory does not match the selected targets")
    merge_guards = guard_lora_merge(peft_model)
    trainable = [
        (name, parameter) for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    ]
    lora_trainable = [
        parameter for name, parameter in trainable if "lora_" in name
    ]
    non_lora_trainable = [
        (name, parameter) for name, parameter in trainable if "lora_" not in name
    ]
    if non_lora_trainable:
        raise RuntimeError(
            "PEFT exposed non-LoRA trainable parameters: "
            f"{[name for name, _parameter in non_lora_trainable[:8]]}"
        )
    return peft_model, {
        "selected_target_count": len(lora_targets),
        "rotated_lora_a_count": rotated_lora_a,
        "primal_lora_a_count": primal_lora_a,
        "adapter_input_basis_by_module": basis_inventory,
        "merge_guard_count": merge_guards,
        "trainable_lora_parameter_count": sum(parameter.numel() for parameter in lora_trainable),
        "trainable_lora_parameter_tensor_count": len(lora_trainable),
        "non_lora_trainable_parameter_count": sum(
            parameter.numel() for _name, parameter in non_lora_trainable
        ),
        "all_trainable_names_lora_only": not non_lora_trainable,
        "all_lora_dropout_zero": not any(dropout_values),
        "lora_dropout_values": sorted(dropout_values),
        "selected_target_names": [name for name, _ in lora_targets],
    }


def install_lora_hadamard_inputs(model: nn.Module) -> int:
    """Make standard PEFT A weights consume exactly the frozen T2 layer's rotated input.

    ``lora_A`` remains a regular ``nn.Linear`` subclass with the same weight parameter name and
    shape, so PEFT's adapter config and weight files remain standard. A nonzero LoRA dropout is
    rejected because applying dropout before a basis rotation would change the target operator.
    """

    from peft.tuners.lora.layer import LoraLayer

    count = 0
    for module in list(model.modules()):
        if not isinstance(module, LoraLayer):
            continue
        base = module.get_base_layer()
        if t2_linear_input_basis(base) != T2_ROTATED_BASIS:
            continue
        for adapter_name in tuple(module.lora_A.keys()):
            dropout = module.lora_dropout[adapter_name]
            if float(getattr(dropout, "p", 0.0)) != 0.0:
                raise ValueError("Bonsai T2 LoRA requires lora_dropout=0 for rotated-basis parity")
            source = module.lora_A[adapter_name]
            if isinstance(source, _HadamardLoRAA):
                continue
            if not isinstance(source, nn.Linear):
                raise TypeError(
                    f"standard LoRA A projection must be Linear-compatible, got {type(source).__name__}"
                )
            module.lora_A[adapter_name] = _HadamardLoRAA(source, base.input_hadamard_signs)
            count += 1
    from .ternary_shared_rotation import install_rotation_scopes
    install_rotation_scopes(model)
    return count


def guard_lora_merge(model: nn.Module) -> int:
    """Disable PEFT merging for packed T2 and frozen dense GDN bases."""

    from peft.tuners.lora.layer import LoraLayer

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "merging a PEFT adapter into immutable packed T2 or frozen dense GDN weights is unsupported"
        )

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        base = module.get_base_layer()
        if is_t2_linear(base) or _is_frozen_dense_gdn_projection(name, base):
            module.merge = reject
            module.unmerge = reject
            count += 1
    if count:
        for owner in (model, getattr(model, "base_model", None)):
            if owner is not None:
                owner.merge_adapter = reject
                owner.merge_and_unload = reject
    return count


__all__ = [
    "HADAMARD_BLOCK",
    "T2_DEQUANTIZE_ROW_CHUNK",
    "T2_GROUP_SIZE",
    "T2_PRIMAL_BASIS",
    "T2_ROTATED_BASIS",
    "FrozenTernaryEmbedding",
    "FrozenTernaryLinear",
    "attach_ternary_peft",
    "dequantize_t2",
    "guard_lora_merge",
    "hadamard_rows",
    "install_lora_hadamard_inputs",
    "is_t2_linear",
    "peft_weight_metadata_only",
    "t2_linear_input_basis",
    "validate_t2",
]
