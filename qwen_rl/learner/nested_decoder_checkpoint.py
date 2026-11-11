"""Bound long-sequence recomputation to attention using Torch checkpoints.

The existing outer HF layer checkpoint remains responsible for CPU boundaries.
Inner inputs belong to that outer checkpoint, rather than adding persistent CPU
copies. Recompute-only packed leases reconnect submodule calls to the streaming
weight lifecycle after the outer decoder call has returned.
"""
from contextlib import nullcontext
from functools import wraps
import math

import torch
from torch.utils.checkpoint import checkpoint
from torch.utils._pytree import tree_flatten, tree_unflatten

MIN_TOKENS = 16500
CHUNKED_TOKENS = 65536
MLP_CHUNK_TOKENS = 8192


class NestedDecoderCheckpoints:
    def __init__(self, layers, offloader, *, min_tokens=MIN_TOKENS):
        if type(min_tokens) is not int or min_tokens < 1:
            raise ValueError("nested checkpoint min_tokens must be a positive integer")
        self.layers, self.offloader = tuple(layers), offloader
        self.min_tokens, self._entries = min_tokens, []
        self._calls = 0

    @property
    def report(self):
        return {"implementation": "torch nested non-reentrant attention checkpoints",
                "module_count": len(self._entries), "min_tokens": self.min_tokens,
                "outer_layer_checkpoint_preserved": True, "tensor_kwargs_flattened": True, "mlp_checkpoint": "chunked_64k", "checkpointed_calls": self._calls,
                "recompute_weights": "managed packed group lease"}

    def install(self):
        if self._entries:
            raise RuntimeError("nested checkpoints already installed")
        if not callable(getattr(self.offloader, "recompute_group", None)):
            raise TypeError("nested decoder checkpoints require managed recompute leases")
        targets = []
        for index, layer in enumerate(self.layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                attention = getattr(layer, "linear_attn", None)
            if attention is None or getattr(layer, "mlp", None) is None:
                raise ValueError(f"decoder {index} lacks attention/MLP modules")
            targets.append((index, attention, attention.forward,
                            attention.__dict__.get("forward")))
        for index, module, raw, previous in targets:
            wrapped = self._wrap(index, module, raw)
            module.forward = wrapped
            self._entries.append((module, wrapped, previous))
        for index, layer in enumerate(self.layers):
            attention = getattr(layer, "self_attn", None)
            for module in (layer.mlp, getattr(attention, "o_proj", None)):
                if module is None:
                    continue
                previous = module.__dict__.get("forward")
                wrapped = self._wrap_chunked(index, module, module.forward)
                module.forward = wrapped
                self._entries.append((module, wrapped, previous))
        return self.report

    def _contexts(self,index):
        return nullcontext(), self.offloader.recompute_group(index)

    def _wrap(self, index, module, raw):
        @wraps(raw)
        def forward(*args, **kwargs):
            x = args[0] if args else kwargs.get("hidden_states",kwargs.get("x"))
            if not torch.is_grad_enabled() or not module.training:
                return raw(*args, **kwargs)
            if not isinstance(x, torch.Tensor):
                raise ValueError("nested checkpoint requires hidden_states")
            if math.prod(x.shape[:-1]) < self.min_tokens:
                return raw(*args, **kwargs)
            for name in ("past_key_values", "past_key_value", "cache_params"):
                if kwargs.get(name) is not None:
                    raise ValueError("nested decoder checkpoints reject mutable cache state")
            if kwargs.get("use_cache", False):
                raise ValueError("nested decoder checkpoints require use_cache=False")
            self._calls += 1
            # Torch's checkpoint input saver does not descend into a kwargs
            # dict. Keeping hidden_states there pins one full norm output per
            # decoder outside saved-tensor hooks. A TreeSpec contains metadata,
            # while every Tensor leaf now enters the parent checkpoint hooks.
            flat, spec = tree_flatten((args, kwargs))
            def invoke(*values):
                positional, named = tree_unflatten(values, spec)
                return raw(*positional, **named)
            return checkpoint(invoke, *flat, use_reentrant=False,
                              context_fn=lambda: self._contexts(index))
        return forward

    def _wrap_chunked(self, index, module, raw):
        @wraps(raw)
        def forward(x, *args, **kwargs):
            if (not torch.is_grad_enabled() or not module.training or x.ndim != 3
                    or x.shape[0] != 1 or x.shape[1] < CHUNKED_TOKENS):
                return raw(x, *args, **kwargs)
            if args or kwargs:
                raise ValueError("chunked T2 projections require one input tensor")
            parts = [checkpoint(raw, part, use_reentrant=False,
                                context_fn=lambda: self._contexts(index))
                     for part in x.split(MLP_CHUNK_TOKENS, dim=1)]
            return torch.cat(parts, dim=1)
        return forward

    def close(self):
        for module, wrapped, _ in self._entries:
            if module.forward is not wrapped:
                raise RuntimeError("nested checkpoint forward changed outside its owner")
        for module, _, previous in reversed(self._entries):
            if previous is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = previous
        self._entries.clear()
