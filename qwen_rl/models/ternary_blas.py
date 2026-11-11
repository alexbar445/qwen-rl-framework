"""Process-wide upstream BLAS preference for the single-device CUDA T2 Learner."""
from __future__ import annotations

from typing import Any

import torch


def configure_ternary_blas(device: torch.device | str) -> dict[str, Any]:
    """Prefer cuBLASLt once at construction; never switch libraries per matmul.

    PyTorch's preference is process-wide and operations unsupported by cuBLASLt
    remain dispatched by upstream Torch. This is not proof of a specific kernel
    for every shape. CPU construction does not touch the CUDA runtime.
    """
    if torch.device(device).type != 'cuda':
        return {'preferred_library': None, 'scope': 'cpu'}
    torch.backends.cuda.preferred_blas_library('cublaslt')
    observed = torch.backends.cuda.preferred_blas_library().name.lower()
    if observed != 'cublaslt':
        raise RuntimeError(f'could not set T2 Learner cuBLASLt preference: {observed}')
    return {'preferred_library': observed, 'scope': 'process', 'compute_dtype': 'bfloat16'}
