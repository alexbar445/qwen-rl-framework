"""Opt-in upstream kernel patches for isolated Learner experiments."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any


LIGER_KERNEL_VERSION = "0.8.4"


def _forward_identity(module: Any) -> tuple[Any, Any]:
    method = module.forward
    return getattr(method, "__self__", None), getattr(method, "__func__", method)


def _parameter_identity(model: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            name,
            id(parameter),
            tuple(parameter.shape),
            str(parameter.dtype),
            bool(parameter.requires_grad),
        )
        for name, parameter in model.named_parameters()
    )


def _qwen3_5_model_types() -> tuple[type, type, type | None]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextModel

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    except ImportError:
        Qwen3_5ForConditionalGeneration = None
    return Qwen3_5ForCausalLM, Qwen3_5TextModel, Qwen3_5ForConditionalGeneration


def _text_model(model: Any, model_type: str) -> Any:
    causal_lm_type, text_model_type, conditional_generation_type = _qwen3_5_model_types()
    if model_type == "qwen3_5_text":
        if isinstance(model, causal_lm_type):
            text_model = getattr(model, model.base_model_prefix, None)
        elif isinstance(model, text_model_type):
            text_model = model
        else:
            raise TypeError(
                "config.model_type='qwen3_5_text' requires Qwen3_5ForCausalLM or Qwen3_5TextModel; "
                f"got {type(model).__module__}.{type(model).__qualname__}"
            )
    elif model_type == "qwen3_5":
        if conditional_generation_type is None or not isinstance(model, conditional_generation_type):
            raise TypeError(
                "config.model_type='qwen3_5' requires Qwen3_5ForConditionalGeneration; "
                f"got {type(model).__module__}.{type(model).__qualname__}"
            )
        model_body = getattr(model, "model", None)
        text_model = getattr(model_body, "language_model", None)
    else:
        raise TypeError(f"unsupported Qwen3.5 dense config model_type: {model_type!r}")

    if text_model is None or not hasattr(text_model, "layers") or not hasattr(text_model, "norm"):
        raise TypeError(
            "Qwen3.5 model type did not expose a text model with layers/norm: "
            f"{type(model).__module__}.{type(model).__qualname__}"
        )
    return text_model


def _gated_norm_snapshot(model: Any) -> dict[str, tuple[int, type, tuple[Any, Any]]]:
    result = {}
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if name.endswith(".linear_attn.norm") or "RMSNormGated" in class_name:
            result[name] = (id(module), type(module), _forward_identity(module))
    return result


def _same_method(left: tuple[Any, Any], right: tuple[Any, Any]) -> bool:
    return left[0] is right[0] and left[1] is right[1]


def apply_qwen3_5_liger_kernels(model: Any, *, kernels: str) -> dict[str, Any]:
    """Patch selected Liger 0.8.4 kernels on a loaded dense Qwen3.5 model.

    The public upstream patcher runs on the actual base-model instance before
    any benchmark update. Cross-entropy, fused-head, and RoPE patches are
    explicitly disabled so TRL's installed DAPO/fused-head forward remains
    intact. Existing parameters and Gated DeltaNet's fused gated norm must be
    identical before and after the patch. ``kernels`` is exactly one of
    ``rms``, ``swiglu``, or ``both``; an unselected module family must retain
    its original forward method.
    """
    if type(kernels) is not str or kernels not in {"rms", "swiglu", "both"}:
        raise ValueError("Liger kernel selection must be exactly 'rms', 'swiglu', or 'both'")
    use_rms_norm = kernels in {"rms", "both"}
    use_swiglu = kernels in {"swiglu", "both"}

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in {"qwen3_5", "qwen3_5_text"}:
        raise TypeError(
            "Liger Learner patch requires the official dense Qwen3.5 model_type "
            "'qwen3_5' or 'qwen3_5_text'"
        )

    try:
        installed_version = version("liger-kernel")
    except PackageNotFoundError as exc:
        raise RuntimeError("the isolated Learner probe requires installed liger-kernel==0.8.4") from exc
    if installed_version != LIGER_KERNEL_VERSION:
        raise RuntimeError(
            f"the isolated Learner probe requires liger-kernel=={LIGER_KERNEL_VERSION}, "
            f"found {installed_version}"
        )

    text_model = _text_model(model, model_type)
    target_norms = [text_model.norm]
    target_mlps = []
    for layer in text_model.layers:
        target_norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
        target_mlps.append(layer.mlp)
    if not target_norms or not target_mlps:
        raise RuntimeError("Qwen3.5 text model has no regular RMSNorm/SwiGLU targets")

    original_forward = _forward_identity(model)
    original_parameters = _parameter_identity(model)
    original_norm_forwards = [_forward_identity(module) for module in target_norms]
    original_mlp_forwards = [_forward_identity(module) for module in target_mlps]
    gated_norms_before = _gated_norm_snapshot(model)
    if not gated_norms_before:
        raise RuntimeError("Qwen3.5 model has no Gated DeltaNet gated RMSNorm to preserve")

    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

    flags = {
        "rope": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": False,
        "rms_norm": use_rms_norm,
        "swiglu": use_swiglu,
    }
    apply_liger_kernel_to_qwen3_5(model=model, **flags)

    patched_forward = _forward_identity(model)
    if not _same_method(original_forward, patched_forward):
        raise RuntimeError("Liger changed the Qwen3.5 base-model forward")
    if _parameter_identity(model) != original_parameters:
        raise RuntimeError("Liger changed the Qwen3.5 parameter inventory or parameter objects")

    patched_norms = sum(
        not _same_method(before, _forward_identity(module))
        for before, module in zip(original_norm_forwards, target_norms, strict=True)
    )
    patched_mlps = sum(
        not _same_method(before, _forward_identity(module))
        for before, module in zip(original_mlp_forwards, target_mlps, strict=True)
    )
    preserved_norms = len(target_norms) - patched_norms
    preserved_mlps = len(target_mlps) - patched_mlps
    expected_patched_norms = len(target_norms) if use_rms_norm else 0
    expected_patched_mlps = len(target_mlps) if use_swiglu else 0
    if patched_norms != expected_patched_norms or patched_mlps != expected_patched_mlps:
        raise RuntimeError(
            "Liger patch selection did not match requested Qwen3.5 module forwards "
            f"(norms={patched_norms}/{expected_patched_norms}, swiglu={patched_mlps}/{expected_patched_mlps})"
        )
    if preserved_norms != (0 if use_rms_norm else len(target_norms)) or preserved_mlps != (
        0 if use_swiglu else len(target_mlps)
    ):
        raise RuntimeError("Liger changed a module family disabled by the explicit selection")

    gated_norms_after = _gated_norm_snapshot(model)
    if gated_norms_after != gated_norms_before:
        raise RuntimeError("Liger changed a Gated DeltaNet gated RMSNorm module or forward")

    return {
        "kernel": f"liger_qwen3_5_{kernels}",
        "kernel_selection": kernels,
        "model_type": model_type,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "liger_kernel_version": installed_version,
        "api_flags": flags,
        "regular_rms_norm_modules_patched": patched_norms,
        "regular_rms_norm_modules_preserved": preserved_norms,
        "swiglu_modules_patched": patched_mlps,
        "swiglu_modules_preserved": preserved_mlps,
        "gated_rms_norm_modules_preserved": len(gated_norms_before),
        "parameter_inventory_preserved": True,
        "base_model_forward_preserved": True,
    }
