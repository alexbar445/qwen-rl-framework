"""Bounded BF16 forward transform using the packaged NInfer CUDA kernel.

The frozen T2 modules validate their +/-1 sign buffers at construction. This
route accepts those int8 buffers only; floating-point transforms and the
original descending-stride transpose VJP use a separate exact-stage kernel.
"""
from __future__ import annotations

from functools import lru_cache
import importlib
from typing import Callable

import torch

from .ternary import _select_flat_hadamard_rows

_TILE_ELEMENTS = 8 * 1024 * 1024


@lru_cache(maxsize=1)
def _load_kernel() -> Callable:
    try:
        package = importlib.import_module('qwen_rl_ternary_kernels')
    except ImportError as exc:
        raise RuntimeError(
            'T2 BF16 forward requires qwen-rl-ternary-kernels==0.1.0; '
            'provide a qualified extension wheel matching the Learner Torch/CUDA '
            'environment. This distribution does not bundle that wheel or its '
            'build recipe; see docs/SUPPORT.md before selecting this backend'
        ) from exc
    kernel = getattr(package, 'fused_hadamard_forward', None)
    if getattr(package, 'API_VERSION', None) != 1 or not callable(kernel):
        raise RuntimeError('qwen-rl-ternary-kernels must provide Hadamard API_VERSION=1')
    return kernel


def kernel_build_info(*, check_device: bool = True) -> dict:
    """Return installed wheel metadata, optionally checking the visible CUDA device."""
    if type(check_device) is not bool:
        raise TypeError('check_device must be bool')
    _load_kernel()
    package = importlib.import_module('qwen_rl_ternary_kernels')
    info = package.build_info()
    if not isinstance(info, dict) or info.get('api_version') != 1:
        raise RuntimeError('qwen-rl-ternary-kernels build metadata must declare api_version=1')
    if (info.get('torch_version') != torch.__version__
            or info.get('torch_cuda_version') != torch.version.cuda):
        raise RuntimeError('rebuild qwen-rl-ternary-kernels for the active Torch/CUDA environment')
    targets = info.get('architectures')
    if not isinstance(targets, list) or not targets:
        raise RuntimeError('qwen-rl-ternary-kernels build metadata omits CUDA targets')
    if check_device and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        supported = False
        for target in targets:
            architecture, _, suffix = target.partition('+')
            major, minor = architecture.split('.')
            compiled = (int(major), int(minor))
            supported |= compiled == capability or (suffix == 'PTX' and compiled <= capability)
        if not supported:
            raise RuntimeError(f'rebuild qwen-rl-ternary-kernels for CUDA device capability {capability}')
    return dict(info)


def fused_hadamard_rows(
    values: torch.Tensor, signs: torch.Tensor, *, inverse: bool,
    block_size: int, output_dtype: torch.dtype,
    compute_dtype: torch.dtype | None = None, transpose: bool = False,
) -> torch.Tensor:
    """Apply the forward transform without unbounded contiguous copies or JIT."""
    if (not values.is_cuda or values.dtype != torch.bfloat16
            or output_dtype != torch.bfloat16 or block_size != 1024
            or compute_dtype not in (None, torch.float32) or transpose):
        raise ValueError('fused T2 Hadamard requires CUDA BF16 forward, FP32 accumulation and block_size=1024')
    if values.ndim < 1 or values.shape[-1] <= 0 or values.shape[-1] % block_size:
        raise ValueError('fused T2 Hadamard width must be a positive multiple of 1024')
    width = values.shape[-1]
    if width > _TILE_ELEMENTS:
        raise ValueError('fused T2 Hadamard width exceeds the bounded tile capacity')
    if signs.dtype != torch.int8 or signs.requires_grad:
        raise ValueError('fused T2 Hadamard requires frozen int8 artifact signs')
    if signs.ndim != 1 or signs.shape[0] != width:
        raise ValueError('fused T2 Hadamard signs must match the input width')
    row_count = values.numel() // width
    if (row_count and values.is_contiguous() and values.data_ptr() % 16 == 0
            and signs.device == values.device and signs.is_contiguous()):
        from .ternary_hadamard_launch import forward_contiguous

        return forward_contiguous(values, signs, inverse=inverse)
    output = torch.empty(values.shape, device=values.device, dtype=output_dtype)
    if not row_count:
        return output
    kernel = _load_kernel()
    sign_values = signs.to(device=values.device, dtype=torch.bfloat16).contiguous()
    output_rows = output.view(row_count, width)
    rows_per_tile = max(1, _TILE_ELEMENTS // width)
    for begin in range(0, row_count, rows_per_tile):
        end = min(begin + rows_per_tile, row_count)
        selected = _select_flat_hadamard_rows(values, begin, end).contiguous()
        # Canonicalize singleton-row strides without copying the full activation.
        selected = selected.view(-1).view(end - begin, width)
        if selected.data_ptr() % 16:
            selected = selected.clone(memory_format=torch.contiguous_format)
        kernel(selected, sign_values, inverse, out=output_rows[begin:end])
    return output
