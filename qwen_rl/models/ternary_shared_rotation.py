"""Reuse T2 rotation values within one PEFT invocation, with separate VJPs.

Scopes are context-local and cleared even when forward raises. No activations
are retained between calls, and PEFT still owns adapter selection and casting.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _RotationFrame:
    owner: Any
    base: Any
    adapters: tuple[Any, ...]
    inputs: torch.Tensor | None = None
    input_version: int | None = None
    rotated: torch.Tensor | None = None


_FRAMES: ContextVar[tuple[_RotationFrame, ...]] = ContextVar('t2_rotation_frames', default=())


def _version(x):
    return None if torch.is_inference(x) else x._version


def _enter(module, _args):
    _FRAMES.set((*_FRAMES.get(), _RotationFrame(module, module.get_base_layer(), tuple(module.lora_A.values()))))


def _leave(module, _args, _output):
    frames = _FRAMES.get()
    if frames and frames[-1].owner is module:
        _FRAMES.set(frames[:-1])


class _ReuseRotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, values, signs):
        ctx.save_for_backward(signs)
        ctx.input_dtype = inputs.dtype
        ctx.compute_dtype = torch.float64 if inputs.dtype == torch.float64 else torch.float32
        return values

    @staticmethod
    def backward(ctx, grad_output):
        from .ternary import _hadamard_rows_impl, HADAMARD_BLOCK
        (signs,) = ctx.saved_tensors
        # Match A's original VJP and cast independently of the base's VJP.
        # Passing detached values prevents A's gradient from entering base's H.
        grad = _hadamard_rows_impl(
            grad_output, signs, inverse=False, block_size=HADAMARD_BLOCK,
            output_dtype=ctx.input_dtype, compute_dtype=ctx.compute_dtype, transpose=True,
        )
        return grad, None, None


def rotate_base(module, inputs):
    from .ternary import hadamard_rows
    rotated = hadamard_rows(inputs, module.input_hadamard_signs, output_dtype=inputs.dtype)
    frames = _FRAMES.get()
    if frames and frames[-1].base is module:
        frame = frames[-1]
        frame.inputs, frame.input_version, frame.rotated = inputs, _version(inputs), rotated
    return rotated


def rotate_adapter(module, inputs):
    from .ternary import hadamard_rows
    frames = _FRAMES.get()
    if frames:
        frame = frames[-1]
        if (any(a is module for a in frame.adapters) and frame.inputs is inputs
                and frame.input_version == _version(inputs) and frame.rotated is not None):
            return _ReuseRotation.apply(inputs, frame.rotated.detach(), module.input_hadamard_signs)
    # PEFT may cast or select a subset of inputs. Those are different operands
    # and must receive their own transform to preserve PEFT's exact semantics.
    return hadamard_rows(inputs, module.input_hadamard_signs, output_dtype=inputs.dtype)


def install_rotation_scopes(model) -> int:
    from peft.tuners.lora.layer import LoraLayer
    from .ternary import T2_ROTATED_BASIS, t2_linear_input_basis
    count = 0
    for module in model.modules():
        if (isinstance(module, LoraLayer)
                and t2_linear_input_basis(module.get_base_layer()) == T2_ROTATED_BASIS
                and not hasattr(module, '_t2_rotation_scope_handles')):
            module._t2_rotation_scope_handles = (
                module.register_forward_pre_hook(_enter),
                module.register_forward_hook(_leave, always_call=True),
            )
            count += 1
    return count
