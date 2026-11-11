"""Native TorchAO NVFP4 model loading and explicit local quantization helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .artifact import read_nvfp4_artifact
from .contract import (
    _enforce_numerical_policy,
    _set_embedding_and_head_dtype,
    _text_lora_targets,
)


ParallelMode = Literal["single", "tp", "replicated", "fsdp2"]
_LORA_SUFFIXES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})


def quantize_nvfp4_linears(
    model: nn.Module,
    *,
    modules_to_not_convert: list[str],
) -> dict[str, Any]:
    """Quantize eligible Linear modules through TorchAO's official quantize_ API."""
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    from torchao.quantization import quantize_

    modules = dict(model.named_modules())
    raw_gdn_skips = [
        name for name in modules_to_not_convert
        if name.endswith(("in_proj_a", "in_proj_b"))
    ]
    if not raw_gdn_skips:
        raise ValueError("NVFP4 recipe has no raw GDN A/B projection skips")
    missing = sorted(set(raw_gdn_skips) - set(modules))
    if missing:
        raise ValueError(f"NVFP4 model is missing declared GDN projections: {missing[:8]}")
    if "lm_head" not in modules:
        raise ValueError("NVFP4 model is missing the declared lm_head skip")

    for name in raw_gdn_skips:
        module = modules[name]
        if not isinstance(module, nn.Linear):
            raise TypeError(f"declared GDN skip is not a Linear module: {name}")
        module.weight = nn.Parameter(module.weight.detach().to(torch.float32), requires_grad=False)
    head = modules["lm_head"]
    if not isinstance(head, nn.Linear):
        raise TypeError("declared lm_head skip is not a Linear module")
    head.weight = nn.Parameter(head.weight.detach().to(torch.bfloat16), requires_grad=False)

    skipped = set(modules_to_not_convert)
    eligible: list[str] = []
    noneligible: list[str] = []

    def filter_fn(module: nn.Module, module_fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if module_fqn in skipped:
            return False
        if module.weight.ndim != 2 or any(dim % 16 for dim in module.weight.shape):
            noneligible.append(module_fqn)
            return False
        if module.weight.dtype != torch.bfloat16:
            module.weight = nn.Parameter(module.weight.detach().to(torch.bfloat16), requires_grad=False)
        eligible.append(module_fqn)
        return True

    quantize_(model, NVFP4WeightOnlyConfig(), filter_fn=filter_fn)
    packed: list[dict[str, Any]] = []
    for name in eligible:
        weight = getattr(modules[name], "weight", None)
        if type(weight).__name__ != "NVFP4Tensor":
            raise RuntimeError(f"TorchAO did not pack eligible Linear module {name}")
        if weight.orig_dtype != torch.bfloat16 or weight.qdata.dtype != torch.uint8:
            raise RuntimeError(f"TorchAO NVFP4 storage contract differs for {name}")
        packed.append({
            "name": f"{name}.weight",
            "qdata_dtype": str(weight.qdata.dtype),
            "qdata_shape": list(weight.qdata.shape),
            "qdata_numel": int(weight.qdata.numel()),
            "orig_dtype": str(weight.orig_dtype),
            "block_size": int(weight.block_size),
            "swizzled_scales": bool(weight.is_swizzled_scales),
        })
    if not packed:
        raise RuntimeError("TorchAO quantize_ produced no NVFP4 Linear weights")
    for name in raw_gdn_skips:
        if modules[name].weight.dtype != torch.float32:
            raise RuntimeError(f"GDN skip weight was not kept in FP32: {name}")
    if head.weight.dtype != torch.bfloat16:
        raise RuntimeError("lm_head skip weight was not kept in BF16")
    return {
        "eligible_modules": sorted(eligible),
        "noneligible_modules": sorted(noneligible),
        "packed_tensors": sorted(packed, key=lambda item: item["name"]),
        "fp32_gdn_modules": sorted(raw_gdn_skips),
        "bf16_head_module": "lm_head",
        "packed_module_count": len(packed),
        "packed_uint8_tensor_count": len(packed),
        "packed_uint8_numel": sum(item["qdata_numel"] for item in packed),
    }


def _unwrap_local_tensor(value: Any) -> Any:
    if type(value).__name__ == "DTensor" and hasattr(value, "to_local"):
        return value.to_local()
    return value


def _runtime_packed_inventory(model: nn.Module) -> list[dict[str, Any]]:
    records_by_name: dict[str, dict[str, Any]] = {}
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        raw_weight = getattr(module, "weight", None)
        weight = _unwrap_local_tensor(raw_weight)
        if type(weight).__name__ != "NVFP4Tensor":
            continue

        # PEFT's BaseTunerLayer exposes `weight` as an alias for
        # `base_layer.weight`. The wrapped layer and its child therefore both
        # appear to own the same packed tensor while walking named_modules().
        # Keep the child parameter (the one named_parameters() can validate)
        # and map its path back to the source model's canonical FQN.
        base_layer = getattr(module, "base_layer", None)
        base_weight = _unwrap_local_tensor(getattr(base_layer, "weight", None))
        if type(base_weight).__name__ == "NVFP4Tensor":
            continue

        canonical_module_name = module_name.removeprefix("base_model.model.")
        if canonical_module_name.endswith(".base_layer"):
            canonical_module_name = canonical_module_name.removesuffix(".base_layer")
        canonical_name = f"{canonical_module_name}.weight"
        parameter_name = f"{module_name}.weight"
        qdata = weight.qdata
        record = {
            "name": canonical_name,
            "runtime_module_name": module_name,
            "parameter_name": parameter_name,
            "weight_storage_type": type(weight).__name__,
            "weight_is_dtensor": type(raw_weight).__name__ == "DTensor",
            "qdata_dtype": str(qdata.dtype),
            "qdata_shape_local": list(qdata.shape),
            "qdata_numel_local": int(qdata.numel()),
            "qdata_bytes_local": int(qdata.numel() * qdata.element_size()),
            "weight_shape_local": list(weight.shape),
            "weight_shape_global": list(raw_weight.shape),
            "orig_dtype": str(weight.orig_dtype),
        }
        previous = records_by_name.get(canonical_name)
        if previous is None or len(parameter_name) > len(previous["parameter_name"]):
            records_by_name[canonical_name] = record
    return sorted(records_by_name.values(), key=lambda item: item["name"])


def nvfp4_local_packed_storage_attestation(
    model: nn.Module,
    *,
    expected_packed_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Verify the actual local packed storage and frozen status after placement."""
    inventory = _runtime_packed_inventory(model)
    if not inventory:
        raise RuntimeError("post-prepare model has no local TorchAO NVFP4 packed weights")
    names = {item["name"] for item in inventory}
    if expected_packed_names is not None and names != set(expected_packed_names):
        raise RuntimeError(
            "post-prepare NVFP4 packed FQNs differ from the loaded artifact: "
            f"missing={sorted(set(expected_packed_names) - names)[:8]}, "
            f"unexpected={sorted(names - set(expected_packed_names))[:8]}"
        )
    named_parameters = dict(model.named_parameters(remove_duplicate=False))
    for item in inventory:
        parameter = named_parameters.get(item.get("parameter_name", item["name"]))
        if parameter is None:
            raise RuntimeError(f"post-prepare packed weight has no named parameter: {item['name']}")
        if parameter.requires_grad:
            raise RuntimeError(f"post-prepare packed base parameter is trainable: {item['name']}")
        if item["qdata_dtype"] != "torch.uint8" or item["qdata_bytes_local"] <= 0:
            raise RuntimeError(f"post-prepare packed weight has invalid local storage: {item['name']}")
    return {
        "status": "verified",
        "source": "post_prepare_module_local_weights",
        "packed_module_count_local": len(inventory),
        "packed_uint8_tensor_count_local": len(inventory),
        "packed_uint8_bytes_local": sum(item["qdata_bytes_local"] for item in inventory),
        "all_packed_base_parameters_frozen": True,
        "qdata_dtype": "torch.uint8",
        "packed_tensors": inventory,
    }


def _guard_lora_merge(model: nn.Module) -> int:
    from peft.tuners.lora.layer import LoraLayer

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("merging a PEFT adapter into immutable NVFP4 base weights is unsupported")

    count = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.merge = reject
            module.unmerge = reject
            count += 1
    for owner in (model, getattr(model, "base_model", None)):
        if owner is not None:
            owner.merge_adapter = reject
            owner.merge_and_unload = reject
    return count


def _select_nvfp4_auto_model_class(
    config: dict[str, Any],
    *,
    auto_causal_lm: Any,
    auto_image_text: Any,
) -> Any:
    """Choose the Transformers loader matching the checkpoint's serialized FQNs."""
    architectures = config.get("architectures")
    if (isinstance(architectures, list)
            and "Qwen3_5ForConditionalGeneration" in architectures
            and config.get("model_type") == "qwen3_5"
            and isinstance(config.get("text_config"), dict)
            and config["text_config"].get("model_type") == "qwen3_5_text"):
        return auto_image_text
    if (isinstance(architectures, list)
            and "Qwen3_5ForCausalLM" in architectures
            and config.get("model_type") == "qwen3_5_text"):
        return auto_causal_lm
    raise ValueError("native NVFP4 loader supports only Qwen3.5 ConditionalGeneration or text CausalLM artifacts")


def _validate_nvfp4_constructor_fqns(
    auto_model_class: Any,
    model_config: Any,
    artifact_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build a meta-only model and prove its eligible Linear FQNs match the artifact."""
    model_config.quantization_config = None
    with torch.device("meta"):
        skeleton = auto_model_class.from_config(
            # Constructor FQN checks do not execute attention; built-in SDPA
            # avoids requiring the Learner's runtime-only custom interface.
            model_config, attn_implementation="sdpa"
        )
    try:
        real_parameters = [
            (name, parameter.device.type)
            for name, parameter in skeleton.named_parameters()
            if parameter.device.type != "meta"
        ]
        if real_parameters:
            raise RuntimeError(
                "NVFP4 constructor FQN preflight allocated non-meta parameters: "
                f"{real_parameters[:8]}"
            )
        modules = dict(skeleton.named_modules())
        skip_modules = set(artifact_manifest["modules_to_not_convert"])
        expected = {
            f"{name}.weight"
            for name, module in modules.items()
            if name and isinstance(module, nn.Linear) and name not in skip_modules
            and module.weight.ndim == 2 and all(int(dim) % 16 == 0 for dim in module.weight.shape)
        }
        inventory = artifact_manifest.get("packed_tensor_inventory")
        if not isinstance(inventory, list) or not inventory:
            raise RuntimeError("NVFP4 artifact has no packed FQN inventory for constructor validation")
        actual = {item.get("name") for item in inventory if isinstance(item, dict)}
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(
                "NVFP4 checkpoint packed FQNs differ from its Transformers architecture; "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        return {
            "model_class": type(skeleton).__name__,
            "constructor_device": "meta",
            "constructor_parameter_count": sum(parameter.numel() for parameter in skeleton.parameters()),
            "constructor_packed_module_count": len(expected),
        }
    finally:
        del skeleton


def _enable_native_nvfp4_lora_training(model: nn.Module) -> bool:
    """Apply Axolotl's pinned, narrowly scoped TorchAO LoRA-training gate."""
    try:
        from axolotl.monkeypatch.torchao_lora import enable_native_nvfp4_lora_training
    except ImportError as exc:
        raise RuntimeError(
            "native NVFP4 LoRA training requires the pinned Axolotl TorchAO training gate"
        ) from exc
    if enable_native_nvfp4_lora_training(model) is not True:
        raise RuntimeError(
            "pinned Axolotl refused the native NVFP4 LoRA training path; "
            "base weights must remain frozen ordinary NVFP4 tensors"
        )
    return True


def _configure_gradient_checkpointing(
    model: nn.Module,
    *,
    trainable: bool,
    requested: bool,
) -> bool:
    if type(requested) is not bool:
        raise TypeError("gradient_checkpointing must be a boolean")
    if not trainable or not requested:
        return False
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise RuntimeError(
            "gradient checkpointing was requested but the loaded model does not support it"
        )
    enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return True


def load_nvfp4_policy(
    model_path: str | Path,
    *,
    trainable: bool,
    gradient_checkpointing: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    parallel_mode: ParallelMode = "single",
    device_mesh: Any | None = None,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load a validated native NVFP4 artifact for single, TP, or replicated DP use."""
    if parallel_mode not in ("single", "tp", "replicated"):
        raise ValueError(f"unsupported NVFP4 parallel_mode: {parallel_mode}")
    if lora_rank <= 0 or lora_alpha <= 0:
        raise ValueError("lora_rank and lora_alpha must be positive")
    if type(gradient_checkpointing) is not bool:
        raise TypeError("gradient_checkpointing must be a boolean")
    numerical_policy = dict(_enforce_numerical_policy())
    numerical_policy["packed_weight_compute_dtype"] = "torch.bfloat16"

    config, base_fingerprint, artifact_identity = read_nvfp4_artifact(model_path)
    root = Path(model_path).expanduser().resolve()
    artifact_manifest = artifact_identity["artifact_manifest"]
    if not torch.cuda.is_available():
        raise RuntimeError("native NVFP4 policy loading requires a visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("native NVFP4 policy loading requires BF16 support")
    if parallel_mode == "single" and torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"single-device NVFP4 loading requires one visible CUDA device; found {torch.cuda.device_count()}"
        )
    if parallel_mode == "tp":
        if device_mesh is None or int(device_mesh.size()) < 2:
            raise ValueError("NVFP4 TP loading requires a multi-rank device_mesh")
    elif device_mesh is not None:
        raise ValueError("NVFP4 single/replicated loading must not receive a TP device_mesh")

    try:
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AttentionInterface,
            AttentionMaskInterface,
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            TorchAoConfig,
        )
        from transformers.masking_utils import sdpa_mask
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    except ImportError as exc:
        raise RuntimeError(
            "native NVFP4 loading requires the pinned Transformers and TorchAO packages"
        ) from exc

    from .attention import arc_sdpa_attention_forward

    AttentionInterface.register("arc_sdpa", arc_sdpa_attention_forward)
    AttentionMaskInterface.register("arc_sdpa", sdpa_mask)
    skip_modules = artifact_manifest["modules_to_not_convert"]
    quantization_config = TorchAoConfig(
        NVFP4WeightOnlyConfig(), modules_to_not_convert=skip_modules
    )
    auto_model_class = _select_nvfp4_auto_model_class(
        config,
        auto_causal_lm=AutoModelForCausalLM,
        auto_image_text=AutoModelForImageTextToText,
    )
    model_config = AutoConfig.from_pretrained(str(root), local_files_only=True)
    constructor_evidence = _validate_nvfp4_constructor_fqns(
        auto_model_class, model_config, artifact_manifest
    )
    model_config.quantization_config = quantization_config.to_dict()

    torch.cuda.reset_peak_memory_stats()
    load_kwargs: dict[str, Any] = {
        "quantization_config": quantization_config,
        "config": model_config,
        "dtype": torch.bfloat16,
        "attn_implementation": "arc_sdpa",
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if parallel_mode == "tp":
        from axolotl.monkeypatch.torchao_tp import native_nvfp4_tp_checkpoint_loading

        load_kwargs.update({
            "tp_plan": "auto",
            "tp_size": int(device_mesh.size()),
            "device_mesh": device_mesh,
        })
        with native_nvfp4_tp_checkpoint_loading(device_mesh):
            base = auto_model_class.from_pretrained(str(root), **load_kwargs)
    else:
        load_kwargs["device_map"] = {"": torch.cuda.current_device()}
        base = auto_model_class.from_pretrained(str(root), **load_kwargs)

    runtime_packed = _runtime_packed_inventory(base)
    expected_packed_names = {
        item["name"] for item in artifact_manifest["packed_tensor_inventory"]
    }
    actual_packed_names = {item["name"] for item in runtime_packed}
    if actual_packed_names != expected_packed_names:
        missing = sorted(expected_packed_names - actual_packed_names)
        unexpected = sorted(actual_packed_names - expected_packed_names)
        raise RuntimeError(
            f"loaded NVFP4 tensor inventory mismatch; missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    embedding_dtypes = _set_embedding_and_head_dtype(base)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    text_config = getattr(base.config, "text_config", None)
    if trainable:
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = False
    gradient_checkpointing_enabled = _configure_gradient_checkpointing(
        base, trainable=trainable, requested=gradient_checkpointing
    )

    targets = _text_lora_targets(base)
    if any(name.rsplit(".", 1)[-1] not in _LORA_SUFFIXES or "linear_attn" in name for name in targets):
        raise RuntimeError("NVFP4 LoRA targets must be attention/MLP rowwise or colwise projections")
    model = get_peft_model(
        base,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    native_lora_training_gate_enabled = (
        _enable_native_nvfp4_lora_training(model) if trainable else False
    )

    tp_prepared: list[str] = []
    tp_skipped: list[str] = []
    if parallel_mode == "tp":
        from axolotl.monkeypatch.torchao_tp_lora import prepare_native_nvfp4_tp_lora

        result = prepare_native_nvfp4_tp_lora(model, merge_aware=False)
        tp_prepared = list(result.prepared)
        tp_skipped = list(result.skipped)
        if not tp_prepared:
            raise RuntimeError("upstream Axolotl prepared no NVFP4 TP LoRA modules")

    peft_runtime_packed = _runtime_packed_inventory(model)
    peft_base_names = {item["name"] for item in peft_runtime_packed}
    if peft_base_names != actual_packed_names:
        raise RuntimeError("PEFT wrapping changed the canonical packed NVFP4 module inventory")

    merge_guard_count = _guard_lora_merge(model)
    if trainable:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_("lora_" in name)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.train(mode=trainable)

    adapter_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    if trainable and not adapter_parameters:
        raise RuntimeError("PEFT did not expose trainable native NVFP4 LoRA parameters")
    if any("lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise RuntimeError("native NVFP4 base has a trainable non-LoRA parameter")
    base_trainable = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if "lora_" not in name and parameter.requires_grad
    )
    if base_trainable:
        raise RuntimeError("native NVFP4 base contains trainable parameters")

    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=True, use_fast=True,
    )
    source_identity = artifact_identity["source_fingerprint"]
    evidence: dict[str, Any] = {
        "model_path": str(root),
        "base_fingerprint": base_fingerprint,
        **artifact_identity,
        "parallel_mode": parallel_mode,
        "numerical_policy": numerical_policy,
        "quantization": {
            "mode": "torchao",
            "quant_method": "torchao",
            "quantizer": "NVFP4WeightOnlyConfig",
            "bits": 4,
            "compute_dtype": "torch.bfloat16",
            "weight_dtype": "nvfp4",
            "packed_resident": True,
            "dequantized_weight_cache": False,
            "modules_to_not_convert": skip_modules,
            "base_artifact_fingerprint": artifact_identity["artifact_fingerprint"],
            "packed_uint8_tensor_count": artifact_manifest["packed_uint8_tensor_count"],
            "packed_uint8_numel": artifact_manifest["packed_uint8_numel"],
        },
        "source_fingerprint": source_identity,
        "packed_module_count": artifact_manifest["packed_module_count"],
        "packed_module_names": [item["name"] for item in artifact_manifest["packed_tensor_inventory"]],
        "packed_buffer_numel": artifact_manifest["packed_uint8_numel"],
        "packed_uint8_tensor_count": artifact_manifest["packed_uint8_tensor_count"],
        "base_trainable_parameters": base_trainable,
        "base_frozen": base_trainable == 0,
        "adapter_trainable_parameters": sum(parameter.numel() for parameter in adapter_parameters),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target_module_count": len(targets),
        "lora_target_modules": targets,
        "embedding_dtypes": embedding_dtypes,
        "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
        "gradient_checkpointing_requested": gradient_checkpointing,
        "use_cache": getattr(text_config, "use_cache", getattr(base.config, "use_cache", None)),
        "device": str(torch.cuda.current_device()),
        "lora_merge_guard_count": merge_guard_count,
        "native_nvfp4_lora_training_gate": native_lora_training_gate_enabled,
        "native_nvfp4_lora_training_gate_name": (
            "axolotl.enable_native_nvfp4_lora_training"
            if native_lora_training_gate_enabled else None
        ),
        "runtime_packed_module_count": len(runtime_packed),
        "runtime_packed_module_names": [item["name"] for item in runtime_packed],
        "peft_runtime_packed_module_count": len(peft_runtime_packed),
        "peft_runtime_packed_module_names": [item["name"] for item in peft_runtime_packed],
        "constructor_fqn_preflight": constructor_evidence,
        "tp_lora_prepared_modules": tp_prepared,
        "tp_lora_skipped_modules": tp_skipped,
        "load_peak_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "load_peak_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
    }
    return model, tokenizer, evidence


def nvfp4_peft_state_dict(
    model: nn.Module,
    *,
    parallel_mode: ParallelMode,
    collect_on_this_rank: bool = True,
) -> dict[str, torch.Tensor] | None:
    """Return the canonical PEFT state dict, gathering TP factors through Axolotl.

    The TP collector returns full named-parameter FQNs (including
    ``lora_A.default.weight``), while PEFT's regular serializer omits the
    adapter-name segment. Normalize both provider contracts against the
    loaded model's actual trainable LoRA FQN inventory before returning state.
    """
    if parallel_mode == "tp":
        from axolotl.monkeypatch.torchao_tp_lora import native_nvfp4_tp_peft_state_dict

        state = native_nvfp4_tp_peft_state_dict(
            model, collect_on_this_rank=collect_on_this_rank
        )
        if not collect_on_this_rank:
            return None
    elif parallel_mode == "fsdp2":
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=True,
            ),
        )
        if not collect_on_this_rank:
            return None
        full_lora_names = {
            name for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad and "lora_" in name
        }
        unexpected = sorted(
            name for name in state if "lora_" in name and name not in full_lora_names
        )
        if unexpected:
            raise RuntimeError(f"DCP returned trainable LoRA FQNs outside the model inventory: {unexpected[:8]}")
        state = {name: value for name, value in state.items() if name in full_lora_names}
    else:
        if not collect_on_this_rank:
            return None
        from peft import get_peft_model_state_dict

        state = get_peft_model_state_dict(model)
    if state is None:
        raise RuntimeError("NVFP4 PEFT state collector returned no state on the collection rank")
    return _canonical_nvfp4_peft_state(model, state)


def _canonical_nvfp4_peft_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Project serializer keys onto the model's canonical LoRA FQN inventory."""
    if not isinstance(state, dict):
        raise TypeError("NVFP4 PEFT state collector must return a dictionary")

    full_to_canonical: dict[str, str] = {}
    expected: set[str] = set()
    for full_name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad or "lora_" not in full_name:
            continue
        parts = full_name.split(".")
        adapter_segments = [
            index + 1 for index, part in enumerate(parts[:-1])
            if part in {"lora_A", "lora_B"} and parts[index + 1] == "default"
        ]
        if len(adapter_segments) != 1:
            raise RuntimeError(f"trainable LoRA FQN has no unique default adapter segment: {full_name}")
        canonical_parts = parts.copy()
        del canonical_parts[adapter_segments[0]]
        canonical_name = ".".join(canonical_parts)
        if canonical_name in expected:
            raise RuntimeError(f"model has duplicate canonical LoRA FQN: {canonical_name}")
        expected.add(canonical_name)
        full_to_canonical[full_name] = canonical_name

    if not expected:
        raise RuntimeError("loaded NVFP4 model has no trainable LoRA FQNs")

    canonical_state: dict[str, torch.Tensor] = {}
    for source_name, value in state.items():
        if not isinstance(source_name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("NVFP4 PEFT state entries must map names to tensors")
        if source_name in full_to_canonical:
            canonical_name = full_to_canonical[source_name]
        elif source_name in expected:
            canonical_name = source_name
        else:
            raise RuntimeError(f"PEFT state key is outside the model LoRA FQN inventory: {source_name}")
        if canonical_name in canonical_state:
            raise RuntimeError(f"PEFT state contains duplicate canonical LoRA FQN: {canonical_name}")
        canonical_state[canonical_name] = value

    if set(canonical_state) != expected:
        raise RuntimeError(
            "PEFT state keys differ from the model LoRA FQN inventory: "
            f"missing={sorted(expected - set(canonical_state))[:8]}, "
            f"unexpected={sorted(set(canonical_state) - expected)[:8]}"
        )
    return canonical_state


def nvfp4_tp_parameter_placements(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Describe each native TP LoRA parameter as replicated or sharded."""
    import torch.distributed as dist

    from axolotl.monkeypatch.torchao_tp_lora import _lora_tp_plan, _tp_group, _tp_lora_layout

    named_parameters = dict(model.named_parameters(remove_duplicate=False))
    names_by_value: dict[int, list[str]] = {}
    for name, value in named_parameters.items():
        if "lora_" in name:
            names_by_value.setdefault(id(value), []).append(name)

    placements: dict[str, dict[str, Any]] = {}
    for module_name, module in model.named_modules():
        if not getattr(module, "_axolotl_native_nvfp4_tp_lora_prepared", False):
            continue
        weight = module.base_layer.weight
        layout = _tp_lora_layout(weight, _lora_tp_plan(model, module_name))
        process_group = _tp_group(weight.device_mesh)
        tp_rank = dist.get_rank(process_group) if dist.is_initialized() else 0
        tp_world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        for factor_name, factor in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
            role = "replicated" if (layout == "colwise") == (factor_name == "lora_A") else "sharded"
            shard_dim = None if role == "replicated" else (0 if factor_name == "lora_B" else 1)
            for adapter, layer in factor.items():
                names = names_by_value.get(id(layer.weight))
                if not names:
                    raise RuntimeError(f"TP LoRA parameter has no stable FQN: {module_name}.{factor_name}.{adapter}")
                for name in names:
                    placements[name] = {
                        "role": role,
                        "layout": layout,
                        "shard_dim": shard_dim,
                        "tp_rank": tp_rank,
                        "tp_world_size": tp_world_size,
                    }
    expected = {
        name for name, parameter in named_parameters.items()
        if "lora_" in name and parameter.requires_grad
    }
    if set(placements) != expected:
        raise RuntimeError(
            "native TP LoRA placement inventory is incomplete: "
            f"missing={sorted(expected - set(placements))[:8]}, "
            f"unexpected={sorted(set(placements) - expected)[:8]}"
        )
    return placements


def load_nvfp4_peft_state_dict(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    parallel_mode: ParallelMode,
    rank: int,
) -> None:
    """Restore a full PEFT state, splitting TP-sharded LoRA factors per rank."""
    import torch.distributed as dist

    if parallel_mode not in ("single", "tp", "replicated", "fsdp2"):
        raise ValueError(f"unsupported NVFP4 parallel_mode: {parallel_mode}")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if not isinstance(state, dict) or not state:
        raise ValueError("NVFP4 PEFT state must be a non-empty dictionary")

    # Axolotl's TP gather uses full parameter FQNs (`lora_A.default.weight`),
    # while PEFT's ordinary state dict omits the default adapter segment.
    canonical_state: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("NVFP4 PEFT state entries must map names to tensors")
        canonical = name.replace(".default.", ".")
        if canonical in canonical_state:
            raise ValueError(f"NVFP4 PEFT state has duplicate canonical key: {canonical}")
        canonical_state[canonical] = value

    named_parameters = dict(model.named_parameters(remove_duplicate=False))
    names_by_canonical: dict[str, list[str]] = {}
    for name, parameter in named_parameters.items():
        if "lora_" in name:
            canonical = name.replace(".default.", ".")
            names_by_canonical.setdefault(canonical, []).append(name)
    names_by_value: dict[int, list[str]] = {}
    for name, parameter in named_parameters.items():
        if "lora_" in name:
            names_by_value.setdefault(id(parameter), []).append(name)
    expected = set(names_by_canonical)
    supplied = set(canonical_state)
    if supplied != expected:
        raise ValueError(
            "NVFP4 PEFT state inventory differs from model: "
            f"missing={sorted(expected - supplied)[:8]}, unexpected={sorted(supplied - expected)[:8]}"
        )

    tp_slices: dict[str, tuple[int, int, int]] = {}
    if parallel_mode == "tp":
        from axolotl.monkeypatch.torchao_tp_lora import _lora_tp_plan, _tp_group, _tp_lora_layout

        for module_name, module in model.named_modules():
            if not getattr(module, "_axolotl_native_nvfp4_tp_lora_prepared", False):
                continue
            weight = module.base_layer.weight
            layout = _tp_lora_layout(weight, _lora_tp_plan(model, module_name))
            process_group = _tp_group(weight.device_mesh)
            world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
            local_rank = dist.get_rank(process_group) if dist.is_initialized() else 0
            if local_rank != rank:
                raise ValueError(f"TP adapter rank mismatch: supplied {rank}, process group rank is {local_rank}")
            for factor_name, factor in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
                shard_dim = (
                    0 if layout == "colwise" and factor_name == "lora_B"
                    else 1 if layout == "rowwise" and factor_name == "lora_A"
                    else None
                )
                for adapter, layer in factor.items():
                    parameter_names = names_by_value.get(id(layer.weight), [])
                    if len(parameter_names) != 1:
                        raise RuntimeError(f"TP LoRA factor has no canonical key: {module_name}.{factor_name}.{adapter}")
                    canonical = parameter_names[0].replace(".default.", ".")
                    if shard_dim is not None:
                        full = canonical_state[canonical]
                        local_shape = tuple(layer.weight.shape)
                        if full.ndim != len(local_shape) or any(
                            full.shape[axis] != local_shape[axis] * world_size
                            for axis in range(full.ndim) if axis == shard_dim
                        ) or any(
                            full.shape[axis] != local_shape[axis]
                            for axis in range(full.ndim) if axis != shard_dim
                        ):
                            raise ValueError(f"full TP factor shape is incompatible for {canonical}")
                        tp_slices[canonical] = (shard_dim, local_shape[shard_dim], local_rank)

    try:
        from torch.distributed.tensor import DTensor, distribute_tensor
    except ImportError:  # pragma: no cover - DTensor is present in the pinned runtime
        DTensor = ()  # type: ignore[assignment,misc]
        distribute_tensor = None

    with torch.no_grad():
        for canonical, parameter_names in names_by_canonical.items():
            value = canonical_state[canonical]
            if canonical in tp_slices:
                dim, local_size, local_rank = tp_slices[canonical]
                value = value.narrow(dim, local_rank * local_size, local_size)
            for name in parameter_names:
                parameter = named_parameters[name]
                if parallel_mode == "fsdp2" and isinstance(parameter, DTensor):
                    if tuple(parameter.shape) != tuple(value.shape):
                        raise ValueError(
                            f"NVFP4 FSDP2 PEFT tensor shape mismatch for {name}: "
                            f"snapshot={tuple(value.shape)}, model={tuple(parameter.shape)}"
                        )
                    assert distribute_tensor is not None
                    distributed_value = distribute_tensor(
                        value.to(device=parameter.device, dtype=parameter.dtype),
                        parameter.device_mesh,
                        parameter.placements,
                    )
                    parameter.copy_(distributed_value)
                    continue
                if tuple(parameter.shape) != tuple(value.shape):
                    raise ValueError(
                        f"NVFP4 PEFT tensor shape mismatch for {name}: "
                        f"snapshot={tuple(value.shape)}, model={tuple(parameter.shape)}"
                    )
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
