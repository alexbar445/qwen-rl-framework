"""Bound the live buffers of uncached T2 GDN without changing its kernels.

Keep HF's explicit head expansion and Parameter namespace. Compact QKV before
FLA while preserving projection order and backward accumulation. Cached decoding
belongs to the Worker; this execution plan is only for the uncached Learner.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

from torch.utils.checkpoint import checkpoint

import torch
import torch.nn.functional as F

RECIPE = "fla_explicit_heads_compact_qkv_grouped_core/v1"
CORE_HEAD_BLOCK = 16
CORE_MIN_TOKENS = 16500


def _chunk_core(module, query, key, value, g, beta, cu_seqlens):
    """Run independent heads through FLA without splitting the time axis."""
    def run(q, k, v, decay, gate):
        return module.chunk_gated_delta_rule(
            q, k, v, g=decay, beta=gate, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )[0]

    tokens = query.shape[0] * query.shape[1]
    if tokens < CORE_MIN_TOKENS or query.shape[2] <= CORE_HEAD_BLOCK:
        return run(query, key, value, g, beta)
    parts = [torch.split(tensor, CORE_HEAD_BLOCK, dim=dimension)
             for tensor, dimension in ((query, 2), (key, 2), (value, 2), (g, -1), (beta, -1))]
    outputs = []
    for tensors in zip(*parts, strict=True):
        if torch.is_grad_enabled() and getattr(module, "_t2_core_checkpointing", True):
            outputs.append(checkpoint(run, *tensors, use_reentrant=False))
        else:
            outputs.append(run(*tensors))
    return torch.cat(outputs, dim=2)


def _uncached_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    if cache_params is not None:
        raise ValueError("T2 Learner compact GDN execution requires cache_params=None")
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, length, _ = hidden_states.shape
    mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
    b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
    if self.causal_conv1d_fn is not None:
        mixed = self.causal_conv1d_fn(
            x=mixed,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=kwargs.get("seq_idx"),
        )
    else:
        mixed = F.silu(self.conv1d(mixed)[:, :, :mixed.shape[-1]])
    mixed = mixed.transpose(1, 2)
    query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, self.head_k_dim).contiguous()
    key = key.reshape(batch, length, -1, self.head_k_dim).contiguous()
    value = value.reshape(batch, length, -1, self.head_v_dim).contiguous()
    # FLA's input guard also makes contiguous copies. Retire their oversized
    # common storage before requesting the kernel's working buffers.
    del mixed
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    del a, b
    # Preserve HF's expansion and backward head-reduction order. Native GVA's
    # Q/K gradient comparison did not pass the diagnostic's standard gate.
    repeats = self.num_v_heads // self.num_k_heads
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    core = _chunk_core(self, query, key, value, g, beta, kwargs.get("cu_seq_lens_q"))
    del query, key, value, g, beta
    core = self.norm(core.reshape(-1, self.head_v_dim), z).reshape(batch, length, -1)
    return self.out_proj(core)


class CompactGDNExecution:
    """Own per-instance execution changes; preserve parameter and state names."""

    def __init__(self, layers):
        self._entries = []
        self.checkpointing = True
        targets = []
        for layer in layers:
            if getattr(layer, "layer_type", None) != "linear_attention":
                continue
            module = layer.linear_attn
            if module.num_k_heads <= 0 or module.num_v_heads % module.num_k_heads:
                raise ValueError("compact GDN requires divisible key/value head counts")
            if module.num_v_heads < module.num_k_heads:
                raise ValueError("compact GDN requires divisible key/value head counts")
            if not getattr(module.chunk_gated_delta_rule, "__module__", "").startswith("fla."):
                raise ValueError("compact GDN requires the installed FLA chunk_gated_delta_rule")
            targets.append(module)
        if not targets:
            raise ValueError("T2 model has no GDN layers")
        # Validate every target before mutating any module.
        for module in targets:
            original = module.forward
            previous = module.__dict__.get("forward")
            optimized = MethodType(_uncached_forward, module)
            module._t2_core_checkpointing = True
            module.forward = optimized
            self._entries.append((module, optimized, original, previous))

    @property
    def report(self):
        return {
            "recipe": RECIPE,
            "module_count": len(self._entries),
            "native_gva": False,
            "core_max_heads": CORE_HEAD_BLOCK,
            "core_min_tokens": CORE_MIN_TOKENS,
            "core_checkpoint_for_grad_enabled_groups": self.checkpointing,
            "complete_sequence_per_core": True,
            "qkv_compacted_before_core": True,
            "z_projection_after_core": False,
            "cache_supported": False,
            "parameter_namespace_preserved": True,
        }

    def set_checkpointing(self, enabled):
        if type(enabled) is not bool or not self._entries:
            raise ValueError("GDN checkpoint mode requires an installed plan and a boolean")
        for module, *_ in self._entries:
            module._t2_core_checkpointing = enabled
        self.checkpointing = enabled

    @contextmanager
    def reference_context(self):
        """Diagnostic reference, preserving later per-instance checkpoint wrappers."""
        forwards = [(module, module.forward, original) for module, _, original, _ in self._entries]
        try:
            for module, _, original in forwards:
                module.forward = original
            yield
        finally:
            for module, _, original in forwards:
                if module.forward is not original:
                    raise RuntimeError("GDN reference owner changed")
            for module, current, _ in forwards:
                module.forward = current

    def close(self):
        # The checkpoint wrapper must retire first; never overwrite another owner.
        for module, optimized, _, _ in self._entries:
            if module.forward is not optimized:
                raise RuntimeError("GDN forward changed outside its owner")
        for module, _, _, previous in self._entries:
            if previous is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = previous
            del module._t2_core_checkpointing
        self._entries.clear()
