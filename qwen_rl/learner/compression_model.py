"""Recovery adapter around the same T2 provider and optimized HF model builder."""

from __future__ import annotations

from torch import nn

from qwen_rl.models.mla_recovery import MLAFactors, install_mla_factors
from qwen_rl.models.ternary import FrozenTernaryEmbedding, FrozenTernaryLinear
from qwen_rl.models.ternary_recovery import ScaleTernaryEmbedding, ScaleTernaryLinear


def base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def recovery_store(path):
    from qwen_rl.models.ternary_artifacts import TernaryArtifactStore

    class RecoveryArtifactStore(TernaryArtifactStore):
        def load_linear(self, parameter, **kwargs):
            return super().load_linear(
                parameter, primal=self.input_basis(parameter) == "primal", **kwargs
            )

        def load_fused_linear(self, parameters, **kwargs):
            modes = {self.input_basis(n) for n in parameters}
            if len(modes) != 1:
                raise ValueError(
                    "fused projections cannot combine different input transforms"
                )
            return super().load_fused_linear(
                parameters, primal=modes.pop() == "primal", **kwargs
            )

    return RecoveryArtifactStore(path)


def attach_recovery_lora(model):
    from peft import get_peft_model
    from peft.tuners.lora.layer import LoraLayer

    from qwen_rl.models.ternary import (
        guard_lora_merge,
        install_lora_hadamard_inputs,
        peft_weight_metadata_only,
    )
    from qwen_rl.models.ternary_recovery import PrimalTernaryLinear

    from .parallel.qwen35_pipeline import (
        DEFAULT_LORA_TARGET_SUFFIXES,
        _make_lora_config,
        _target_module_names,
    )

    primal = [m for m in model.modules() if isinstance(m, PrimalTernaryLinear)]
    targets = [
        name
        for name in _target_module_names(model, DEFAULT_LORA_TARGET_SUFFIXES)
        if isinstance(model.get_submodule(name), nn.Linear)
    ]
    try:
        for m in primal:
            m._peft_weight_metadata_only = True
        with peft_weight_metadata_only(model):
            model = get_peft_model(
                model,
                _make_lora_config(
                    targets,
                    16,
                    16,
                    task_type=None,
                ),
            )
    finally:
        for m in primal:
            m._peft_weight_metadata_only = False
    install_lora_hadamard_inputs(model)
    guard_lora_merge(model)

    def reject(*args, **kwargs):
        raise RuntimeError(
            "packed recovery bases require explicit composition, not PEFT merge"
        )

    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.merge = module.unmerge = reject
    model.merge_adapter = model.merge_and_unload = reject
    return model


def select_trainables(model, families):
    root = base_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    full_scales = "ternary_scales" in families
    if full_scales and "vocab_scales" in families:
        raise ValueError("ternary_scales already includes vocab_scales")
    if full_scales:
        from qwen_rl.models.ternary_recovery import PrimalTernaryLinear
        # Snapshot owners before wrapping; shared fused projections keep one owner.
        for name, module in list(root.named_modules()):
            if isinstance(module, FrozenTernaryEmbedding):
                root.set_submodule(name, ScaleTernaryEmbedding(module))
            elif isinstance(module, (FrozenTernaryLinear, PrimalTernaryLinear)):
                root.set_submodule(name, ScaleTernaryLinear(module))
    elif "vocab_scales" in families:
        embedding, head = root.model.embed_tokens, root.lm_head
        if not isinstance(embedding, FrozenTernaryEmbedding) or not isinstance(head, FrozenTernaryLinear):
            raise ValueError("vocab_scales requires actual frozen T2 vocabulary matrices")
        root.model.embed_tokens = ScaleTernaryEmbedding(embedding)
        root.lm_head = ScaleTernaryLinear(head)
    found = {f: 0 for f in families}
    allowed = set()
    for module in model.modules():
        if isinstance(module, MLAFactors) and "mla_factors" in families:
            for p in module.parameters(recurse=False):
                p.requires_grad_(True)
                allowed.add(id(p))
                found["mla_factors"] += p.numel()
        if (
            isinstance(module, (ScaleTernaryEmbedding, ScaleTernaryLinear))
            and ("vocab_scales" in families or full_scales)
        ):
            allowed.add(id(module.log_scale_delta))
            found["ternary_scales" if full_scales else "vocab_scales"] += module.log_scale_delta.numel()
    for name, p in model.named_parameters():
        if "lora" in families and ("lora_A." in name or "lora_B." in name):
            p.requires_grad_(True)
            allowed.add(id(p))
            found["lora"] += p.numel()
    if any(n == 0 for n in found.values()):
        raise ValueError(f"requested trainable family has no parameters: {found}")
    if any(id(p) not in allowed for p in model.parameters() if p.requires_grad):
        raise RuntimeError("unexpected trainable base parameter")
    return found


def load_model(config, device):
    from .parallel.qwen35_pipeline import build_qwen35_causal_lm

    with recovery_store(config.artifact["path"]) as store:
        if store.artifact_id.hex() != config.artifact["artifact_id"]:
            raise ValueError("base artifact ID changed")
        store.materialize_text_resources(config.run_dir / "tokenizer")
        model, evidence = build_qwen35_causal_lm(
            store.text_config,
            store,
            lora_rank=16,
            lora_alpha=16,
            device=device,
            with_lora=False,
        )
    if config.mla_bundle:
        evidence["mla"] = install_mla_factors(
            model, config.mla_bundle, base_sha256=config.artifact["sha256"]
        )
    if "lora" in config.trainable:
        model = attach_recovery_lora(model)
    evidence["trainable_families"] = select_trainables(model, config.trainable)
    if "ternary_scales" in config.trainable:
        evidence["ternary_scale_owners"] = {
            name: {"groups": module.log_scale_delta.numel(), "shape": list(module.log_scale_delta.shape),
                   "input_basis": "hadamard1024_signed" if isinstance(module.base, FrozenTernaryLinear) else ("embedding_inverse_hadamard" if isinstance(module, ScaleTernaryEmbedding) else "primal")}
            for name, module in model.named_modules()
            if isinstance(module, (ScaleTernaryEmbedding, ScaleTernaryLinear))
        }
    root = base_model(model)
    root.config.use_cache = False
    if config.gradient_checkpointing:
        root.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # No TRL/DAPO patch, Worker, rollout, publication or behavior-probability path.
    model.train()
    return model, evidence
