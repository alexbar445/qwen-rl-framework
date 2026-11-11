"""Managed frozen embedding input-gradient policy for non-reentrant T2 DDP."""
from __future__ import annotations

from types import MethodType


class FrozenEmbeddingInputPolicy:
    """Keep frozen input IDs off the gradient graph for non-reentrant LoRA.

    Install before TRL construction. HF/TRL may request the hook again when a
    train call enables checkpointing; validate that mode and keep it disabled.
    Externally supplied requires-grad inputs_embeds are never modified.
    """
    def __init__(self, model, *, use_reentrant):
        if use_reentrant is not False:
            raise ValueError("frozen embedding input policy requires non-reentrant checkpointing")
        embedding = model.get_input_embeddings()
        if any(p.requires_grad for p in embedding.parameters()):
            raise ValueError("input policy requires a frozen token embedding")
        self.model = model
        self.original = model.enable_input_require_grads
        self.previous = model.__dict__.get('enable_input_require_grads')
        handles = getattr(model, '_require_grads_hooks', ())
        self.had_hook = any(h.id in embedding._forward_hooks for h in handles)
        if self.had_hook:
            model.disable_input_require_grads()
        self.suppressed_calls = 0

        def keep_frozen(root):
            if any(p.requires_grad for p in root.get_input_embeddings().parameters()):
                raise RuntimeError("frozen embedding became trainable")
            for layer in root.modules():
                if getattr(layer, 'gradient_checkpointing', False):
                    fn = getattr(layer, '_gradient_checkpointing_func', None)
                    if getattr(fn, 'keywords', {}).get('use_reentrant') is not False:
                        raise RuntimeError("frozen input policy requires explicit non-reentrant checkpoints")
            self.suppressed_calls += 1

        self.optimized = MethodType(keep_frozen, model)
        model.enable_input_require_grads = self.optimized

    @property
    def report(self):
        return {'recipe':'frozen_embedding_non_reentrant_no_input_grad/v1',
                'base_embedding_frozen':True,'use_reentrant':False,
                'previous_hook_removed':self.had_hook,
                'suppressed_enable_calls':self.suppressed_calls,
                'adapter_parameters_unchanged':True}

    def close(self):
        if self.model.enable_input_require_grads is not self.optimized:
            raise RuntimeError("frozen input policy owner changed")
        if self.previous is None:
            self.model.__dict__.pop('enable_input_require_grads',None)
        else:
            self.model.enable_input_require_grads = self.previous
        if self.had_hook:
            self.original()
