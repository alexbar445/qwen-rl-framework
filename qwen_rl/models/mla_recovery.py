"""Differentiable MLA reconstruction for recovery using upstream attention math.

Replace only K/V projections. Original Q/K norms, partial RoPE, attention gates,
SDPA and GDN stay in the shared model. This is a training forward, not a latent
decode-cache implementation; recovery always disables use_cache.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MLAFactors(nn.Module):
    _compression_recovery_family = "mla_factors"

    def __init__(self, down, up_k, up_v, rope, *, kv_heads, head_dim, rope_dim):
        super().__init__()
        rank, hidden = down.shape
        if rank <= 0 or hidden <= 0:
            raise ValueError("MLA rank and hidden width must be positive")
        expected = [
            (kv_heads * (head_dim - rope_dim), rank),
            (kv_heads * head_dim, rank),
            (kv_heads * rope_dim, hidden),
        ]
        if (
            not 0 < rope_dim <= head_dim
            or [tuple(x.shape) for x in [up_k, up_v, rope]] != expected
        ):
            raise ValueError(
                "MLA factor geometry differs from original KV/RoPE geometry"
            )
        for name, value in [
            ("down", down),
            ("up_k", up_k),
            ("up_v", up_v),
            ("rope", rope),
        ]:
            if value.ndim != 2 or not bool(torch.isfinite(value).all()):
                raise ValueError("MLA factors must be finite matrices")
            self.register_parameter(name, nn.Parameter(value.detach().float().clone()))
        self.kv_heads, self.head_dim, self.rope_dim = kv_heads, head_dim, rope_dim

    def latent(self, x):
        return F.linear(x, self.down.to(x.dtype))

    def key(self, x):
        latent = self.latent(x)
        kr = F.linear(x, self.rope.to(x.dtype)).reshape(
            *x.shape[:-1], self.kv_heads, self.rope_dim
        )
        kc = F.linear(latent, self.up_k.to(x.dtype)).reshape(
            *x.shape[:-1], self.kv_heads, self.head_dim - self.rope_dim
        )
        return torch.cat((kr, kc), -1).flatten(-2)

    def value(self, x):
        return F.linear(self.latent(x), self.up_v.to(x.dtype))


class MLAProjection(nn.Module):
    def __init__(self, factors, kind):
        super().__init__()
        self.factors, self.kind = factors, kind
        self.in_features = factors.down.shape[1]
        self.out_features = factors.kv_heads * factors.head_dim

    def forward(self, x):
        return self.factors.key(x) if self.kind == "key" else self.factors.value(x)


def install_mla_factors(model, bundle, *, base_sha256):
    """Load a separately initialized, base-bound factor bundle; never guess SVD here."""
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    from qwen_rl.learner.recovery import _sha256_file

    path = Path(bundle)
    spec = json.loads(path.read_text())
    if (
        spec.get("schema") != "qwen-rl-mla-factors/v1"
        or spec.get("base_artifact_sha256") != base_sha256
        or spec.get("input_basis") != "primal"
        or spec.get("rope_policy") != "preserve_original_prefix"
        or spec.get("qk_norm") != "upstream_dynamic"
    ):
        raise ValueError("MLA bundle schema or base identity differs")
    tensor_path = (path.parent / spec["tensor_file"]).resolve()
    if _sha256_file(tensor_path) != spec["tensor_sha256"]:
        raise ValueError("MLA tensor SHA mismatch")
    tensors = load_file(str(tensor_path), device="cpu")
    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers, cfg = root.model.layers, root.config
    full = [i for i, kind in enumerate(cfg.layer_types) if kind == "full_attention"]
    if spec.get("layers") != full:
        raise ValueError("MLA bundle must cover exactly the full-attention layers")
    rope_dim = int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"])
    expected_keys = {f"{i}.{n}" for i in full for n in ["down", "up_k", "up_v", "rope"]}
    if set(tensors) != expected_keys:
        raise ValueError("MLA bundle has missing or unexpected factor keys")
    for i in full:
        attn = layers[i].self_attn
        device = next(attn.parameters()).device
        factors = MLAFactors(
            *(tensors[f"{i}.{n}"].to(device) for n in ["down", "up_k", "up_v", "rope"]),
            kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rope_dim=rope_dim,
        )
        if factors.down.shape[1] != cfg.hidden_size:
            raise ValueError("MLA bundle hidden size differs")
        attn.k_proj = MLAProjection(factors, "key")
        attn.v_proj = MLAProjection(factors, "value")
    return {
        "layers": full,
        "training_forward": "upstream_attention_with_reconstructed_KV",
        "latent_decode_cache_validated": False,
        "tensor_sha256": spec["tensor_sha256"],
    }
