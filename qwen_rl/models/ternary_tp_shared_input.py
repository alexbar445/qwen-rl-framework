"""Forward-scoped sharing for repeated T2 TP projection inputs.

The upstream Qwen3.5 attention, GDN and MLP forwards call sibling column
projections with the same activation object. Sharing one replicated-input
autograd edge combines their input gradients before one tensor-parallel SUM.
BF16/FP16 projection gradients are lifted to FP32 at each sibling edge, summed
at the shared edge, then cast once to the activation dtype before that SUM.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.autograd import Function


class _SharedLowPrecisionInput(Function):
    """Keep the exact activation value while routing its VJP through FP32."""

    @staticmethod
    def forward(ctx, accumulation_input: torch.Tensor, activation: torch.Tensor):
        ctx.accumulation_dtype = accumulation_input.dtype
        return activation

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient.to(dtype=ctx.accumulation_dtype), None


@dataclass
class _Region:
    module_name: str
    expected_slots: frozenset[str] | None = None
    seen_slots: set[str] = field(default_factory=set)
    source: torch.Tensor | None = None
    group: Any = None
    shared_input: torch.Tensor | None = None

    def replicate(self, value: torch.Tensor, group: Any, slot: str,
                  create: Callable[[], torch.Tensor]) -> torch.Tensor:
        if not isinstance(slot, str) or not slot:
            raise RuntimeError(f"{self.module_name}: shared TP projection slot is missing")
        if self.expected_slots is not None and slot not in self.expected_slots:
            raise RuntimeError(f"{self.module_name}: unexpected shared TP projection {slot!r}")
        if slot in self.seen_slots:
            raise RuntimeError(f"{self.module_name}: shared TP projection {slot!r} ran twice")
        self.seen_slots.add(slot)

        if self.source is None:
            self.source, self.group = value, group
            mapped = create()
            if (value.dtype in (torch.float16, torch.bfloat16)
                    and value.requires_grad and torch.is_grad_enabled()):
                # Each projection keeps its own low-precision base/LoRA VJPs.
                # The shared edge receives their separate contributions in
                # FP32; the cast back happens once before the TP SUM.
                self.shared_input = mapped.to(dtype=torch.float32)
                return _SharedLowPrecisionInput.apply(self.shared_input, value)
            self.shared_input = mapped
            return mapped

        if value is not self.source:
            raise RuntimeError(
                f"{self.module_name}: sibling TP projections did not receive one shared activation"
            )
        if group is not self.group:
            raise RuntimeError(f"{self.module_name}: sibling TP projections use different groups")
        if self.shared_input is None:
            raise RuntimeError(f"{self.module_name}: shared TP input was not initialized")
        if (self.shared_input.dtype == torch.float32
                and value.dtype in (torch.float16, torch.bfloat16)):
            return _SharedLowPrecisionInput.apply(self.shared_input, value)
        return self.shared_input

    def finish(self) -> None:
        if self.expected_slots is None:
            return
        missing = self.expected_slots.difference(self.seen_slots)
        if missing:
            names = ", ".join(sorted(missing))
            raise RuntimeError(f"{self.module_name}: shared TP region skipped projections: {names}")


_ACTIVE_REGION: ContextVar[_Region | None] = ContextVar("t2_tp_shared_input_region", default=None)
_HOOK_FRAMES: ContextVar[tuple[tuple[nn.Module, Token], ...]] = ContextVar(
    "t2_tp_shared_input_hook_frames", default=()
)


@contextmanager
def shared_tp_input_region(*, module_name: str = "manual", expected_slots=None) -> Iterator[None]:
    """Open a fresh sharing region; useful for focused projection oracles."""
    slots = None if expected_slots is None else frozenset(expected_slots)
    region = _Region(module_name=module_name, expected_slots=slots)
    token = _ACTIVE_REGION.set(region)
    try:
        yield
        region.finish()
    finally:
        _ACTIVE_REGION.reset(token)


def share_replicated_tp_input(value: torch.Tensor, *, group: Any, slot: str,
                              create: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Return a sibling-shared TP mapping or fail if its owner scope is absent."""
    region = _ACTIVE_REGION.get()
    if region is None:
        raise RuntimeError(
            f"shared TP projection {slot!r} ran outside its owner forward region"
        )
    return region.replicate(value, group, slot, create)


def _enter_region(module: nn.Module, expected_slots: frozenset[str], module_name: str):
    token = _ACTIVE_REGION.set(_Region(module_name=module_name, expected_slots=expected_slots))
    _HOOK_FRAMES.set((*_HOOK_FRAMES.get(), (module, token)))


def _leave_region(module: nn.Module, output):
    frames = _HOOK_FRAMES.get()
    if not frames or frames[-1][0] is not module:
        raise RuntimeError("shared TP owner forward hooks exited out of order")
    _HOOK_FRAMES.set(frames[:-1])
    _, token = frames[-1]
    region = _ACTIVE_REGION.get()
    _ACTIVE_REGION.reset(token)
    # Non-reentrant checkpoint recomputation can stop early by raising an
    # internal exception. always_call hooks must clear the ContextVar, while
    # only a completed owner forward is checked for its full projection set.
    if output is not None and region is not None:
        region.finish()
    return output


def install_shared_input_regions(model: nn.Module) -> list[dict[str, Any]]:
    """Scope every owner with two or more direct TP column projections.

    The complete model assembler invokes this after MLP, attention and GDN
    sharding, so this single scan covers the production MLP and token-mixer
    owners without changing their upstream ``forward`` implementations.
    """
    column_modules = [
        module for module in model.modules()
        if getattr(module, "_t2_tp_column_parallel_input", False) is True
    ]
    covered: set[int] = set()
    records: list[dict[str, Any]] = []
    for module_name, owner in model.named_modules():
        children = [
            (name, child) for name, child in owner.named_children()
            if getattr(child, "_t2_tp_column_parallel_input", False) is True
        ]
        if len(children) < 2:
            continue
        slots = tuple(name for name, _ in children)
        groups = [getattr(child, "tp_group", None) for _, child in children]
        if any(group is not groups[0] for group in groups[1:]):
            raise ValueError(f"{module_name or '<root>'}: sibling TP projections use different groups")
        if getattr(owner, "_t2_tp_shared_input_handles", None) is not None:
            raise ValueError(f"{module_name or '<root>'}: shared TP input region is already installed")
        for slot, child in children:
            if id(child) in covered:
                raise ValueError(f"{module_name or '<root>'}.{slot}: TP projection has multiple owners")
            covered.add(id(child))
            child._t2_tp_shared_input_required = True
            child._t2_tp_shared_input_slot = slot

        owner_name = module_name or "<root>"
        expected = frozenset(slots)
        pre_handle = owner.register_forward_pre_hook(
            lambda module, _inputs, names=expected, name=owner_name:
                _enter_region(module, names, name)
        )
        post_handle = owner.register_forward_hook(
            lambda module, _inputs, output: _leave_region(module, output),
            always_call=True,
        )
        owner._t2_tp_shared_input_handles = (pre_handle, post_handle)
        records.append({
            "module": owner_name,
            "projections": list(slots),
            "projection_count": len(slots),
            "input_identity": "same_tensor_within_owner_forward",
            "low_precision_local_accumulation": "fp32_then_cast_once",
            "tp_gradient_reduction": "one_sum_per_shared_input",
        })

    uncovered = [module for module in column_modules if id(module) not in covered]
    if uncovered:
        names = [type(module).__name__ for module in uncovered[:4]]
        raise ValueError(
            "every T2 TP column projection must belong to a multi-projection owner region; "
            f"uncovered: {names}"
        )
    return records


__all__ = ["install_shared_input_regions", "share_replicated_tp_input", "shared_tp_input_region"]
