"""T2-aware saved-tensor policies for Learner training."""
from __future__ import annotations

from typing import Any

import torch

from .bounded_offload import DEFAULT_MAX_PINNED_BYTES
from .saved_activation_ledger import LedgerSavedTensorContext


def _base_model(model: Any) -> Any:
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


class TensorParallelActivationOffload:
    """Own bounded activation copies for one complete TP forward/backward.

    The vocabulary-sharded head recomputes bounded weight chunks itself. There
    is no dense full head ticket or persistent dequantized weight to exempt.
    """

    def __init__(self, policy):
        self.bounded_offload = policy

    @property
    def report(self):
        return self.bounded_offload.report

    def __enter__(self):
        self.bounded_offload.__enter__()
        return self

    def __exit__(self, *args):
        try:
            return self.bounded_offload.__exit__(*args)
        finally:
            self.bounded_offload.close()


def _build_saved_tensor_policy(
    model: Any,
    *,
    parallel_mode: str,
    pin_memory: bool,
    max_pinned_bytes: int | None = None,
):
    """Use native hooks with T2-specific activation offload where required."""
    if max_pinned_bytes is not None and (type(max_pinned_bytes) is not int or max_pinned_bytes <= 0):
        raise ValueError("max_pinned_bytes must be a positive integer when set")
    base_offload = torch.autograd.graph.save_on_cpu(pin_memory=pin_memory)
    if parallel_mode == "tp" and getattr(model, "_t2_tp_model", False):
        if pin_memory:
            raise ValueError("T2 TP native spill requires pin_memory=False")
        from .async_offload import AsyncActivationOffload

        return TensorParallelActivationOffload(AsyncActivationOffload(
            base_offload.pack_hook, base_offload.unpack_hook,
            max_pinned_bytes=(max_pinned_bytes or DEFAULT_MAX_PINNED_BYTES),
        ))
    if parallel_mode in {"single", "ddp"}:
        from ..models.ternary import FrozenTernaryLinear

        base = _base_model(model)
        getter = getattr(base, "get_output_embeddings", None)
        head = getter() if callable(getter) else None
        if isinstance(head, FrozenTernaryLinear):
            if pin_memory:
                raise ValueError("T2 saved-tensor offload requires pin_memory=False")
            from .async_offload import AsyncActivationOffload, AsyncTernaryHeadSaveOnCPU

            policy = AsyncActivationOffload(
                base_offload.pack_hook,
                base_offload.unpack_hook,
                max_pinned_bytes=(max_pinned_bytes or DEFAULT_MAX_PINNED_BYTES),
            )
            return AsyncTernaryHeadSaveOnCPU(policy)
    return base_offload



def saved_tensor_offload_context(
    model: Any,
    *,
    parallel_mode: str,
    pin_memory: bool,
    max_pinned_bytes: int | None = None,
    checkpoint_input_recompute_span: int | None = None,
):
    """Install one native policy with live payload accounting and exact-source reuse."""
    model_span = getattr(model, "_checkpoint_input_recompute_span", 1)
    if type(model_span) is not int or model_span != 1:
        raise ValueError("checkpoint_input_recompute_span must be1")
    if (checkpoint_input_recompute_span is not None
            and (type(checkpoint_input_recompute_span) is not int or checkpoint_input_recompute_span != 1)):
        raise ValueError("checkpoint_input_recompute_span must be1")
    policy = _build_saved_tensor_policy(
        model, parallel_mode=parallel_mode, pin_memory=pin_memory,
        max_pinned_bytes=max_pinned_bytes,
    )
    return LedgerSavedTensorContext(policy)
