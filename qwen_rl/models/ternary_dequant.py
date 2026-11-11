"""Fused T2 decoding with static full tiles and reusable dynamic tail graphs.

Only compiled callables are cached; the cache retains no decoded weight tensors.
Packed buffers remain frozen. This module does not change the base GEMM.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable
from types import FunctionType

import torch

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


def decode_t2_into(
    codes: torch.Tensor, scales_fp16: torch.Tensor, output: torch.Tensor
) -> None:
    """Graph body: exact signed 2-bit decode, then scale in the output dtype.

    The caller validates storage outside the compiled graph. Keeping the FP16
    scale cast before multiplication preserves the current BF16 recipe.
    """
    columns = torch.arange(output.shape[1], device=codes.device)
    fields = (codes[:, columns // 4].to(torch.int16) >> ((columns % 4) * 2)) & 3
    magnitude = fields & 1
    signed = torch.where((fields & 2) != 0, -magnitude, magnitude)
    scales = scales_fp16.to(dtype=output.dtype)[:, columns // 128]
    output.copy_(signed.to(dtype=output.dtype) * scales)


def compile_t2_decoder(
    rows: int, width: int, dtype: torch.dtype, device_index: int, *, backend=None,
    dynamic_rows: bool = False,
) -> Callable:
    # Dynamo caches per Python code object, not per closure. Give each semantic
    # kernel signature its own code identity; no source generation, global cache
    # limit changes, or weight tensors are involved.
    row_signature = "tail" if dynamic_rows else rows
    name = f"t2_decode_{row_signature}_{width}_{str(dtype).split('.')[-1]}_{device_index}"
    code = decode_t2_into.__code__.replace(co_name=name)
    body = FunctionType(code, decode_t2_into.__globals__, name)
    options = {"fullgraph": True, "dynamic": dynamic_rows}
    if backend is not None:
        options["backend"] = backend
    return torch.compile(body, **options)


@lru_cache(maxsize=128)
def _compiled_decoder(rows: int, width: int, dtype: torch.dtype, device_index: int) -> Callable:
    return compile_t2_decoder(rows, width, dtype, device_index)


@lru_cache(maxsize=128)
def _compiled_tail_decoder(width: int, dtype: torch.dtype, device_index: int) -> Callable:
    # Embedding rows follow the full input length, so its remainder changes
    # between requests. Cache by semantic signature, never by that remainder.
    # Dynamo may specialize singleton rows separately; all other tails reuse
    # the same dynamic graph. Full tiles retain their static decoder above.
    return compile_t2_decoder(0, width, dtype, device_index, dynamic_rows=True)


def _decoder_for_tile(rows: int, row_chunk_size: int, width: int,
                      dtype: torch.dtype, device_index: int) -> Callable:
    if rows < row_chunk_size:
        return _compiled_tail_decoder(width, dtype, device_index)
    return _compiled_decoder(rows, width, dtype, device_index)


def dequantize_t2_cuda(
    codes: torch.Tensor,
    scale_words: torch.Tensor,
    *,
    dtype: torch.dtype,
    row_chunk_size: int = 1024,
) -> torch.Tensor:
    """Decode valid CUDA T2 buffers with bounded row work tiles.

    No exception handling or eager retry is installed: compilation/kernel errors
    propagate. Full code/finite-scale validation remains the artifact loader's job.
    """
    if type(row_chunk_size) is not int or row_chunk_size <= 0:
        raise ValueError("T2 row_chunk_size must be a positive integer")
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError("T2 output dtype must be BF16, FP16, FP32 or FP64")
    if codes.dtype != torch.uint8 or codes.ndim != 2:
        raise TypeError("T2 codes must be a uint8 matrix")
    rows, packed_width = codes.shape
    width = packed_width * 4
    if rows <= 0 or width <= 0 or width % 128:
        raise ValueError("T2 dimensions must be positive with input width divisible by 128")
    if scale_words.dtype != torch.uint16 or tuple(scale_words.shape) != (rows, width // 128):
        raise ValueError("T2 scales must be uint16 words with one scale per 128 columns")
    if not codes.is_cuda or codes.device != scale_words.device:
        raise ValueError("compiled T2 decoding requires both packed buffers on the same CUDA device")
    codes = codes.contiguous()
    scale_words = scale_words.contiguous()

    output = torch.empty((rows, width), dtype=dtype, device=codes.device)
    # Reinterpret the stored scale words, without rounding or copying payloads.
    scales_fp16 = scale_words.view(torch.float16)
    # This is frozen base storage, including when called during autograd or
    # checkpoint recomputation. Keep compiler contexts independent of callers.
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        for begin in range(0, rows, row_chunk_size):
            end = min(begin + row_chunk_size, rows)
            decoder = _decoder_for_tile(end - begin, row_chunk_size, width, dtype, codes.device.index)
            decoder(codes[begin:end], scales_fp16[begin:end], output[begin:end])
    return output
