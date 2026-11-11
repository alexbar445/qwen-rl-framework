"""Exact comparison for checked, live CPU byte vectors without tensor temporaries."""
import ctypes

_libc = ctypes.CDLL(None)
_memcmp = _libc.memcmp
_memcmp.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
_memcmp.restype = ctypes.c_int


def equal_cpu_bytes(left, right):
    import torch

    for value in (left, right):
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.dtype != torch.uint8 or value.ndim != 1
                or value.layout != torch.strided or value.is_neg() or value.is_conj()
                or not value.is_contiguous()):
            raise ValueError('byte comparison requires contiguous CPU uint8 vectors')
    if left.numel() != right.numel():
        return False
    # Hold both tensors throughout the call. size_t preserves large lengths;
    # the explicit empty case never passes null storage pointers to libc.
    return left.numel() == 0 or _memcmp(left.data_ptr(), right.data_ptr(), left.numel()) == 0
