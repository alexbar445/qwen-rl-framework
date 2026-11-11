"""Reuse one immutable decoded head matrix until its input gradient is ready."""
from __future__ import annotations


def _identity(tensor):
    return (tensor.data_ptr(), tensor._version, tuple(tensor.shape), tensor.stride(),
            tensor.storage_offset(), tensor.dtype, tensor.device)


class DecodedTernaryWeightLease:
    """Graph-owned storage, released before the backbone starts its backward.

    The lease is passed explicitly through checkpoint recomputation and the
    frozen linear VJP. It never installs mutable state on the model or changes
    saved-tensor hooks. An abandoned graph releases its lease normally.
    """

    def __init__(self, head):
        self._head = head
        self._sources = (_identity(head.packed_codes), _identity(head.scale_words))
        self._shape = (head.out_features, head.in_features)
        self._dtype = head.compute_dtype
        self._weight = None
        self._weight_identity = None

    def get(self, codes, scales, rows, width, dtype):
        if ((_identity(self._head.packed_codes), _identity(self._head.scale_words)) != self._sources
                or (_identity(codes), _identity(scales)) != self._sources
                or (rows, width) != self._shape or dtype != self._dtype):
            raise RuntimeError("decoded T2 weight lease payload changed")
        if self._weight is None:
            from .ternary import _dequantize_t2_unchecked
            self._weight = _dequantize_t2_unchecked(codes, scales, rows, width, dtype)
            self._weight_identity = _identity(self._weight)
        elif _identity(self._weight) != self._weight_identity:
            raise RuntimeError("decoded T2 weight lease matrix changed")
        return self._weight

    def release(self, gradient=None):
        self._weight = None
        self._weight_identity = None
        return gradient
