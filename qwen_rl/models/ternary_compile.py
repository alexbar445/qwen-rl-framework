"""Opt-in, per-model regional compilation for frozen T2 Learner models.

The installer is deliberately separate from model construction. It compiles the bounded Qwen RMSNorm forward affine and the PEFT LoRA
add/cast epilogue. RMS statistics stay eager and are saved for the pinned
Liger fused CUDA backward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from peft.tuners.lora.layer import Linear
from torch import nn

from ..contracts.ternary import SUPPORTED_T2_LORA_PROFILES
from .ternary import is_t2_linear
from .ternary_lora_scale import IdentityScaleLoRA
from .ternary_rms_norm import BoundedQwenRMSNorm, _reference, _frozen_rms_backward


_COMPILE_OPTIONS = {"fullgraph": True, "dynamic": True}
_INDUCTOR_OPTIONS = {"emulate_precision_casts": True}

def _qwen_norm_forward_affine(
    z: torch.Tensor,
    inverse_rms: torch.Tensor,
    weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Pointwise affine only; FP32 cast/reduction/rsqrt stay in eager PyTorch."""
    return (z * inverse_rms * (1.0 + weight.float())).to(output_dtype)


class _HybridQwenNormForward:
    """Eager FP32 RMS statistics followed by one compiled pointwise affine."""

    def __init__(self, affine: Callable[..., torch.Tensor]):
        self._affine = affine

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float, *, return_inverse_rms=False):
        z = x.float()
        variance = z.pow(2).mean(-1, keepdim=True)
        inverse_rms = torch.rsqrt(variance + eps)
        output = self._affine(z, inverse_rms, weight, x.dtype)
        return (output, inverse_rms) if return_inverse_rms else output


def _lora_add_cast(base: torch.Tensor, update: torch.Tensor, scaling: float) -> torch.Tensor:
    """Match IdentityScaleLoRA plus PEFT's final cast to the base output dtype."""
    if scaling != 1.0:
        update = update * scaling
    if base.dtype == update.dtype and base.shape == update.shape:
        combined = base.add_(update)
    else:
        combined = base + update
    return combined.to(base.dtype)


class _CompiledIdentityScaleLoRA(IdentityScaleLoRA):
    """PEFT variant object that owns the model's shared compiled epilogue."""

    def __init__(self, epilogue: Callable[..., torch.Tensor]):
        self._epilogue = epilogue

    def forward(self, module, active_adapter, x, result, **kwargs):
        update = module.lora_B[active_adapter](
            module.lora_A[active_adapter](module.lora_dropout[active_adapter](x))
        )
        # AOT functionalization materializes another full 64K MLP output for
        # the compiled in-place add. Eager add reuses the existing buffer.
        if result.numel() >= 65536 * 17408 and result.dtype == update.dtype and result.shape == update.shape:
            return _lora_add_cast(result, update, module.scaling[active_adapter])
        return self._epilogue(result, update, module.scaling[active_adapter])


@dataclass
class RegionalCompileInstallation:
    """Live compiled regions plus reversible A/B toggles for one model instance."""

    evidence: dict[str, Any]
    _norm_callable: Callable[..., torch.Tensor]
    _lora_callable: Callable[..., torch.Tensor]
    _norms: list[tuple[
        BoundedQwenRMSNorm,
        Callable[..., torch.Tensor] | None,
        Callable[..., torch.Tensor] | None,
        Callable[..., torch.Tensor],
        Callable[..., torch.Tensor] | None,
    ]]
    _loras: list[tuple[nn.Module, str, IdentityScaleLoRA, _CompiledIdentityScaleLoRA]]
    _enabled: bool = True
    _restored: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._restored

    def _check_owned_state(self) -> None:
        if self._restored:
            raise RuntimeError("regional compile installation has been restored")
        for module, baseline_forward, baseline_backward, compiled_forward, compiled_backward in self._norms:
            expected_forward = compiled_forward if self._enabled else baseline_forward
            expected_backward = compiled_backward if self._enabled else baseline_backward
            if (module.reference_fn is not expected_forward
                    or module.backward_fn is not expected_backward):
                raise RuntimeError("a bounded RMSNorm reference changed outside this installation")
        for module, adapter, original, compiled in self._loras:
            expected = compiled if self._enabled else original
            if module.lora_variant.get(adapter) is not expected:
                raise RuntimeError("a PEFT LoRA variant changed outside this installation")

    def set_enabled(self, enabled: bool) -> "RegionalCompileInstallation":
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        self._check_owned_state()
        for module, baseline_forward, baseline_backward, compiled_forward, compiled_backward in self._norms:
            module.reference_fn = compiled_forward if enabled else baseline_forward
            module.backward_fn = compiled_backward if enabled else baseline_backward
        for module, adapter, original, compiled in self._loras:
            module.lora_variant[adapter] = compiled if enabled else original
        self._enabled = enabled
        self.evidence["enabled"] = enabled
        return self

    def enable(self) -> "RegionalCompileInstallation":
        return self.set_enabled(True)

    def disable(self) -> "RegionalCompileInstallation":
        return self.set_enabled(False)

    def restore(self) -> dict[str, Any]:
        if self._restored:
            return {"restored": True, "already_restored": True}
        self._check_owned_state()
        for module, baseline_forward, baseline_backward, _compiled_forward, _compiled_backward in self._norms:
            module.reference_fn = baseline_forward
            module.backward_fn = baseline_backward
        for module, adapter, original, _compiled in self._loras:
            module.lora_variant[adapter] = original
        self._enabled = False
        self._restored = True
        self.evidence["enabled"] = False
        self.evidence["restored"] = True
        return {"restored": True, "already_restored": False}


def _validate_model(model: nn.Module):
    if not isinstance(model, nn.Module):
        raise TypeError("regional compilation requires a torch.nn.Module")

    norms = [(name, module) for name, module in model.named_modules()
             if isinstance(module, BoundedQwenRMSNorm)]
    if not norms:
        raise ValueError("install bounded Qwen RMSNorms before regional compilation")
    for name, module in norms:
        if module.reference_fn is not None or module.backward_fn is not None:
            raise ValueError(f"bounded RMSNorm {name!r} already has regional callables installed")

    lora_layers = []
    unsupported_layers = []
    for name, module in model.named_modules():
        if not isinstance(module, Linear) or not getattr(module, "lora_A", None):
            continue
        if not is_t2_linear(module.get_base_layer()):
            unsupported_layers.append(name)
            continue
        lora_layers.append((name, module))
    if unsupported_layers:
        raise ValueError(
            "regional LoRA compilation only supports frozen T2 base layers; "
            f"unsupported adapter layers: {unsupported_layers[:4]}"
        )
    if not lora_layers:
        raise ValueError("install IdentityScaleLoRA on frozen T2 layers before regional compilation")

    targets = []
    for name, module in lora_layers:
        adapter_names = set(module.lora_A)
        if not adapter_names or adapter_names != set(module.lora_B) or len(adapter_names) != 1:
            raise ValueError(f"LoRA layer {name!r} must have exactly one matching adapter")
        adapter = next(iter(adapter_names))
        active = list(module.active_adapters)
        if active != [adapter]:
            raise ValueError(
                f"LoRA layer {name!r} must have only adapter {adapter!r} active; got {active!r}"
            )
        if (module.merged or module.lora_variant.get(adapter) is None
                or type(module.lora_variant[adapter]) is not IdentityScaleLoRA
                or any(module.use_dora.values())):
            raise ValueError(
                f"LoRA layer {name!r} must have an unmerged vanilla T2 IdentityScaleLoRA adapter"
            )
        rank = module.r.get(adapter)
        alpha = module.lora_alpha.get(adapter)
        scaling = module.scaling.get(adapter)
        if (type(rank) is not int or type(alpha) is not int
                or (rank, alpha) not in SUPPORTED_T2_LORA_PROFILES):
            raise ValueError(
                f"LoRA layer {name!r} requires a supported rank/alpha 16/16 or 32/32 adapter; "
                f"got rank={rank!r}, alpha={alpha!r}"
            )
        if type(scaling) not in (int, float) or scaling != 1.0:
            raise ValueError(
                f"LoRA layer {name!r} requires alpha/rank scaling 1.0; got {scaling!r}"
            )
        if any(parameter.requires_grad for parameter in module.get_base_layer().parameters()):
            raise ValueError(f"LoRA layer {name!r} has a trainable T2 base parameter")
        adapter_parameters = list(module.lora_A[adapter].parameters()) + list(
            module.lora_B[adapter].parameters()
        )
        if not adapter_parameters or any(not parameter.requires_grad for parameter in adapter_parameters):
            raise ValueError(f"LoRA layer {name!r} must retain trainable A/B parameters")
        targets.append((name, module, adapter, module.lora_variant[adapter], rank, alpha, scaling))

    profiles = {(rank, alpha) for *_prefix, rank, alpha, _scaling in targets}
    if len(profiles) != 1:
        raise ValueError(f"T2 LoRA layers must share one rank/alpha profile; got {sorted(profiles)}")

    return norms, targets


def install_ternary_regional_compile(
    model: nn.Module,
    *,
    backend: str = "inductor",
    compile_fn: Callable[..., Callable[..., torch.Tensor]] | None = None,
) -> RegionalCompileInstallation:
    """Compile bounded RMSNorm and T2 LoRA add/cast once for one model.

    Call after `install_bounded_qwen_rms_norms` and `install_identity_scale_lora`.
    The returned installation starts enabled and supports repeatable eager/compiled
    A/B toggles without constructing another compiler wrapper.
    """
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a nonempty torch.compile backend name")
    if compile_fn is not None and not callable(compile_fn):
        raise TypeError("compile_fn must be callable")
    norms, targets = _validate_model(model)

    def compile_region(function):
        kwargs = dict(_COMPILE_OPTIONS, backend=backend)
        if backend == "inductor":
            kwargs["options"] = dict(_INDUCTOR_OPTIONS)
        compiler = torch.compile if compile_fn is None else compile_fn
        return compiler(function, **kwargs)

    evidence = {
        "implementation": "per-model compiled Qwen RMSNorm forward affine plus saved-statistics Liger frozen-weight backward",
        "backend": backend,
        "compile_options": {
            "fullgraph": True,
            "dynamic": True,
            "emulate_precision_casts": backend == "inductor",
        },
        "norm_compile_scope": "pointwise_affine_only_after_eager_fp32_cast_variance_and_rsqrt",
        "norm_backward": "saved_inverse_rms_liger_needs_dw_false_backward",
        "norm_module_count": len(norms),
        "norm_callable_count": 1,
        "norm_affine_callable_count": 1,
        "norm_modules": [name for name, _module in norms],
        "lora_epilogue_count": len(targets),
        "lora_epilogue_callable_count": 1,
        "lora_layers": [name for name, _module, *_rest in targets],
        "adapters": sorted({adapter for _name, _module, adapter, *_rest in targets}),
        "rank": targets[0][4],
        "alpha": targets[0][5],
        "scaling": 1.0,
        "state_dict_namespace_preserved": True,
        "parameter_objects_preserved": True,
        "frozen_t2_base_required": True,
        "enabled": True,
        "restored": False,
    }
    norm_affine = compile_region(_qwen_norm_forward_affine)
    norm_callable = _HybridQwenNormForward(norm_affine)
    lora_callable = compile_region(_lora_add_cast)
    compiled_norms = [
        (module, module.reference_fn, module.backward_fn, norm_callable, _frozen_rms_backward)
        for _name, module in norms
    ]
    compiled_loras = [
        (module, adapter, original, _CompiledIdentityScaleLoRA(lora_callable))
        for _name, module, adapter, original, _rank, _alpha, _scaling in targets
    ]

    installation = RegionalCompileInstallation(
        evidence=evidence,
        _norm_callable=norm_callable,
        _lora_callable=lora_callable,
        _norms=compiled_norms,
        _loras=compiled_loras,
    )
    try:
        for module, _baseline_forward, _baseline_backward, compiled_forward, compiled_backward in compiled_norms:
            module.reference_fn = compiled_forward
            module.backward_fn = compiled_backward
        for module, adapter, _original, compiled in compiled_loras:
            module.lora_variant[adapter] = compiled
    except Exception:
        for module, baseline_forward, baseline_backward, _compiled_forward, _compiled_backward in compiled_norms:
            module.reference_fn = baseline_forward
            module.backward_fn = baseline_backward
        for module, adapter, original, _compiled in compiled_loras:
            module.lora_variant[adapter] = original
        raise
    return installation


def configure_ternary_regional_compile(
    model: nn.Module, *, enabled: bool = True,
) -> RegionalCompileInstallation | None:
    """Apply the default T2 single/DDP recipe, or explicitly keep eager mode."""
    if type(enabled) is not bool:
        raise TypeError("ternary_regional_compile must be bool")
    if not enabled:
        return None
    return install_ternary_regional_compile(model)
