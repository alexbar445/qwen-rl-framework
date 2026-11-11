"""FP32-accumulating Dao Hadamard with bounded IO and original transpose order."""
from __future__ import annotations

from functools import lru_cache

import torch

from .ternary import _select_flat_hadamard_rows
from .ternary_hadamard_io import compiled_hadamard_io

_TILE_ELEMENTS = 8 * 1024 * 1024


@lru_cache(maxsize=8)
def _reverse_bits(block_size: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(block_size, device=device, dtype=torch.long)
    permutation = torch.zeros_like(indices)
    for shift in range(block_size.bit_length() - 1):
        permutation = permutation * 2 + ((indices >> shift) & 1)
    return permutation


def dao_hadamard_rows(
    values: torch.Tensor, signs: torch.Tensor, *, inverse: bool,
    block_size: int, output_dtype: torch.dtype,
    compute_dtype: torch.dtype | None = None, transpose: bool = False,
) -> torch.Tensor:
    """Reuse upstream FP32 butterflies with at most 8M elements per IO tile.

    BF16 forward inputs remain BF16 through sign preparation and are rounded
    once by Dao's BF16 output store. Transpose IO remains FP32: bit reversal
    preserves the original descending-stride VJP. No BF16 butterfly stages or
    block padding are introduced; the largest FP32 IO tile is still 32 MiB.
    """
    from fast_hadamard_transform import hadamard_transform

    if compute_dtype not in (None, torch.float32) or values.dtype == torch.float64:
        raise ValueError('Dao T2 backend requires FP32 accumulation')
    if not values.is_cuda or block_size != 1024:
        raise ValueError('Dao T2 backend requires CUDA and independent 1024-wide blocks')
    width = values.shape[-1]
    row_count = values.numel() // width
    output = torch.empty(values.shape, device=values.device, dtype=output_dtype)
    output_rows = output.view(row_count, width)
    signs = signs.to(device=values.device)
    permutation = _reverse_bits(block_size, values.device)
    rows_per_tile = max(1, _TILE_ELEMENTS // width)
    for begin in range(0, row_count, rows_per_tile):
        end = min(begin + rows_per_tile, row_count)
        selected = _select_flat_hadamard_rows(values, begin, end)
        bf16_no_transpose = (
            not transpose
            and selected.dtype == torch.bfloat16
            and output_dtype == torch.bfloat16
        )
        prepare, finish = compiled_hadamard_io(
            width, selected.dtype, signs.dtype, output_dtype, inverse, transpose,
            values.device.index, selected.stride(), bf16_no_transpose,
        )
        working = prepare(selected, signs, permutation, inverse, transpose)
        transformed = hadamard_transform(working, scale=1.0 if transpose else 1 / 32)
        finish(transformed, signs, permutation, output_rows[begin:end], inverse, transpose)
    return output
