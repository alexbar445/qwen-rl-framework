"""FP32 device accumulation without keeping low precision .grad between rows."""
from __future__ import annotations

import torch


class DeviceGradientAccumulator:
    def __init__(self, parameters):
        self.parameters = tuple(parameters)
        self.gradients = {name: None for name, _ in self.parameters}

    @torch.no_grad()
    def add_microbatch(self):
        targets, sources = [], []
        for name, parameter in self.parameters:
            value = parameter.grad
            if value is None:
                continue
            value = value.detach().to(dtype=torch.float32)
            if self.gradients[name] is None:
                # Detach retains this storage when parameter.grad is cleared.
                self.gradients[name] = value
            else:
                targets.append(self.gradients[name])
                sources.append(value)
            parameter.grad = None
        if targets:
            torch._foreach_add_(targets, sources)

    def export_cpu(self):
        result = {}
        for name, value in self.gradients.items():
            # The central owner performs the single authoritative finite check
            # together with global norm computation and clipping. Scanning each
            # full device gradient here would duplicate that synchronization.
            if value is None:
                result[name] = None
            elif value.device.type == "cuda":
                # CUDA -> CPU .to() already owns a CPU copy. contiguous() may
                # make another owning copy for strided inputs, so a final clone
                # would only duplicate the storage again.
                result[name] = value.to(device="cpu").contiguous()
            else:
                # CPU .to() can return the original storage; preserve the
                # independent snapshot contract for CPU and other inputs.
                result[name] = value.to(device="cpu").contiguous().clone()
        return result
