"""Fuse bounded T2 Hadamard preparation/stores around unchanged Dao kernels.

Only compiler callables and tensor metadata are cached. Row counts are dynamic;
weights, activations and full decoded tensors are never retained by this cache.
"""
from __future__ import annotations

from functools import lru_cache
from types import FunctionType
from typing import Callable

import torch


def prepare(values, signs, permutation, inverse: bool, transpose: bool):
    """Preserve FP32 signs/normalization and descending-stride VJP order."""
    width = values.shape[-1]
    working = values.to(torch.float32)
    sign_values = signs.to(torch.float32).reshape(1, width)
    if transpose:
        if inverse:
            working = working * sign_values
        working = working * (1 / 32)
        working = working.reshape(-1, 1024).index_select(-1, permutation)
    elif not inverse:
        working = working * sign_values
    return working.reshape(-1, 1024).contiguous()


def prepare_bf16_no_transpose(values, signs, permutation, inverse: bool, transpose: bool):
    """Keep BF16 forward input for Dao's FP32-register kernel."""
    width = values.shape[-1]
    working = values
    if not inverse:
        working = working * signs.to(torch.bfloat16).reshape(1, width)
    return working.reshape(-1, 1024).contiguous()


def finish_into(transformed, signs, permutation, output, inverse: bool, transpose: bool):
    """Gather/sign/cast directly into the caller's output slice."""
    working = transformed
    if transpose:
        working = working.index_select(-1, permutation)
    working = working.reshape(output.shape)
    if (inverse and not transpose) or (transpose and not inverse):
        working = working * signs.to(torch.float32).reshape(1, output.shape[-1])
    output.copy_(working)


def finish_bf16_no_transpose(transformed, signs, permutation, output, inverse: bool, transpose: bool):
    """Store Dao's BF16 result without an FP32 round trip."""
    working = transformed.reshape(output.shape)
    if inverse:
        working = working * signs.to(torch.bfloat16).reshape(1, output.shape[-1])
    output.copy_(working)


def compile_hadamard_io(
    signature: tuple, *, backend=None, bf16_no_transpose: bool = False,
) -> tuple[Callable, Callable]:
    # As in ternary_dequant, give semantic signatures separate Dynamo code
    # identities without increasing global recompile limits. Row count is absent
    # from the signature, so new trajectory tails reuse dynamic graphs.
    implementations = (
        (prepare_bf16_no_transpose, finish_bf16_no_transpose)
        if bf16_no_transpose
        else (prepare, finish_into)
    )
    functions = []
    for function in implementations:
        name = f"{function.__name__}_{signature}"
        code = function.__code__.replace(co_name=name)
        body = FunctionType(code, function.__globals__, name)
        options = {"fullgraph": True, "dynamic": True}
        if backend is not None:
            options["backend"] = backend
        functions.append(torch.compile(body, **options))
    return tuple(functions)


@lru_cache(maxsize=128)
def compiled_hadamard_io(
    width: int, input_dtype: torch.dtype, sign_dtype: torch.dtype,
    output_dtype: torch.dtype, inverse: bool, transpose: bool,
    device_index: int, strides: tuple[int, ...], bf16_no_transpose: bool = False,
) -> tuple[Callable, Callable]:
    return compile_hadamard_io((
        width, input_dtype, sign_dtype, output_dtype, inverse, transpose,
        device_index, strides, bf16_no_transpose,
    ), bf16_no_transpose=bf16_no_transpose)
