"""Recompute the cheap SwiGLU activation while keeping the original VJP math.

The managed T2 RMSNorm already saves BF16 inputs and tiny FP32 statistics,
not full FP32 intermediates. This installation only removes the additional
SiLU output that eager autograd otherwise retains for the gated product.
"""
import torch
from torch.nn import functional as F


class _RecomputeGatedProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return F.silu(gate) * up

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        # These are precisely MulBackward and SiLUBackward's original ops.
        dgate = torch.ops.aten.silu_backward.default(dy * up, gate)
        dup = dy * F.silu(gate)
        return dgate, dup


class CheapActivationRecompute:
    def __init__(self, layers):
        self.layers = tuple(layers)
        self.owners = []
        self.calls = 0

    @property
    def report(self):
        return dict(mlp_modules=len(self.layers), calls=self.calls,
                    implementation="original SiLU/product ops; recompute SiLU in backward",
                    removed_saved_tensors_per_mlp=1,
                    rmsnorm="existing bounded BF16-input / FP32 inverse-RMS implementation preserved")

    def __enter__(self):
        from transformers.activations import SiLUActivation
        # Validate the complete scope before changing any forward owner.
        for layer in self.layers:
            activation = layer.mlp.act_fn
            if (type(activation) not in (torch.nn.SiLU, SiLUActivation)
                    or getattr(activation, "inplace", False)):
                raise ValueError("cheap recompute requires non-inplace SiLU MLPs")
        for layer in self.layers:
            mlp = layer.mlp
            self.owners.append((mlp, mlp.__dict__.get("forward")))
            def forward(x, _mlp=mlp):
                self.calls += 1
                gate, up = _mlp.gate_proj(x), _mlp.up_proj(x)
                return _mlp.down_proj(_RecomputeGatedProduct.apply(gate, up))
            mlp.forward = forward
        return self

    def __exit__(self, *_):
        for mlp, owner in reversed(self.owners):
            if owner is None:
                del mlp.forward
            else:
                mlp.forward = owner
        self.owners.clear()
