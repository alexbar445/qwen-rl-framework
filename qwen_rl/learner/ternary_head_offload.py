"""Regenerate transient frozen T2 head weights instead of copying them to CPU."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

import torch


def _identity(tensor: torch.Tensor) -> tuple:
    return (id(tensor), tensor.data_ptr(), tensor._version, tuple(tensor.shape),
            tensor.dtype, tensor.device, bool(tensor.requires_grad))


_BOUND_HEAD: ContextVar[Any] = ContextVar('t2_bound_head_weight', default=None)


@contextmanager
def transient_head_scope(head: Any, weight: torch.Tensor):
    """Publish a binding only while TRL can save the exact full weight matrix."""
    if not torch.is_grad_enabled():
        # Inference tensors have no version counter and cannot need this hook.
        yield
        return
    token = _BOUND_HEAD.set((head, weight, _identity(weight)))
    try:
        yield
    finally:
        _BOUND_HEAD.reset(token)


@dataclass(frozen=True)
class _HeadTicket:
    head: Any
    buffers: dict[str, tuple]
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class RecomputeTernaryHeadSaveOnCPU:
    """Delegate ordinary saves and regenerate only a known frozen T2 head.

    Tickets contain the packed owner and metadata, never a dense matrix. Buffer
    identity/version guards reject mutations. The temporary decoded matrix has
    the same backward lifetime as the native CPU hook's restored CUDA matrix.
    """

    def __init__(self, pack_hook: Callable, unpack_hook: Callable):
        self._pack = pack_hook
        self._unpack = unpack_hook
        self.report = {
            'status': 'active', 'head_pack_count': 0, 'head_unpack_count': 0,
            'd2h_bytes_avoided': 0, 'h2d_bytes_avoided': 0,
            'native_pack_count': 0, 'native_unpack_count': 0,
            'dense_head_cached': False,
        }

    def pack(self, tensor: torch.Tensor):
        bound = _BOUND_HEAD.get()
        if bound is not None:
            head, weight, identity = bound
            if (tensor.device == weight.device and tensor.dtype == weight.dtype
                    and tuple(tensor.shape) == tuple(weight.shape)
                    and tensor.data_ptr() == weight.data_ptr()
                    and tensor.stride() == weight.stride()
                    and tensor.storage_offset() == weight.storage_offset()):
                if _identity(weight) != identity or tensor.requires_grad:
                    raise RuntimeError('Frozen T2 head binding changed')
                buffers = {name: _identity(getattr(head, name))
                           for name in ('packed_codes', 'scale_words')}
                self.report['head_pack_count'] += 1
                self.report['d2h_bytes_avoided'] += tensor.numel() * tensor.element_size()
                return _HeadTicket(head, buffers, tuple(tensor.shape), tensor.dtype, tensor.device)
        self.report['native_pack_count'] += 1
        return self._pack(tensor)

    def unpack(self, packed):
        if not isinstance(packed, _HeadTicket):
            self.report['native_unpack_count'] += 1
            return self._unpack(packed)
        for name, identity in packed.buffers.items():
            if _identity(getattr(packed.head, name)) != identity:
                raise RuntimeError('Frozen T2 head payload changed before backward')
        with torch.no_grad():
            weight = packed.head.dequantize_weight(dtype=packed.dtype)
        if (tuple(weight.shape) != packed.shape or weight.device != packed.device
                or weight.requires_grad):
            raise RuntimeError('Reconstructed T2 head metadata differs')
        self.report['head_unpack_count'] += 1
        self.report['h2d_bytes_avoided'] += weight.numel() * weight.element_size()
        return weight

    def __enter__(self):
        self.hooks = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self.hooks.__enter__()
        return self

    def __exit__(self, *args):
        try:
            return self.hooks.__exit__(*args)
        finally:
            self.report['status'] = 'failed' if args and args[0] is not None else 'completed'
