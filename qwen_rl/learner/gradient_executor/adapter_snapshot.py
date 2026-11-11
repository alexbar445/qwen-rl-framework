"""Private CPU adapter snapshots validated before GPU ownership is acquired."""
import math
from collections.abc import Mapping
from types import MappingProxyType

import torch


def finite_cpu_tensor(value):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or value.layout != torch.strided
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
        raise ValueError('adapter snapshot requires dense CPU floating tensors')
    if not value.numel():
        return True
    # aminmax propagates NaNs and both extrema retain any infinity. Avoid
    # creating a full-size isfinite mask; installed values are not transformed.
    low, high = torch.aminmax(value.detach())
    return math.isfinite(low.item()) and math.isfinite(high.item())


class _OwnedAdapterSnapshot:
    """Internal capsule: caller storage is never retained or exported publicly."""
    __slots__ = ('version', '_tensors')

    def __init__(self, version, state):
        if type(version) is not int or version < 0:
            raise ValueError('adapter version must be a non-negative integer')
        if not isinstance(state, Mapping) or not state:
            raise ValueError('adapter state must be a non-empty tensor mapping')
        tensors = {}
        for name, value in state.items():
            if (not isinstance(name, str) or not name or not isinstance(value, torch.Tensor)
                    or value.device.type != 'cpu' or value.layout != torch.strided):
                raise ValueError('staged adapter state must contain named CPU tensors')
            tensor = value.detach().contiguous().clone()
            if not finite_cpu_tensor(tensor):
                raise ValueError(f'adapter tensor {name!r} differs from a finite snapshot')
            tensors[name] = tensor
        self.version = version
        self._tensors = MappingProxyType(tensors)
