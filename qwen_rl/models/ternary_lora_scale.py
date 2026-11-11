"""Remove identity scaling through the pinned PEFT vanilla-variant interface.

Only the scale multiplication is omitted when its runtime value is exactly 1.
PEFT still owns casting, adapter selection/disable, dropout and final dtype.
The frozen T2 base produces an owned output and saves only packed weights for
backward, so equal-dtype LoRA additions can reuse that output allocation.
"""
from __future__ import annotations

from typing import Any

from peft.tuners.lora.layer import Linear, LoraVariant
from torch import nn

from .ternary import is_t2_linear, t2_linear_input_basis


class IdentityScaleLoRA(LoraVariant):
    @staticmethod
    def forward(module, active_adapter, x, result, **kwargs):
        update = module.lora_B[active_adapter](
            module.lora_A[active_adapter](module.lora_dropout[active_adapter](x))
        )
        scaling = module.scaling[active_adapter]
        if scaling != 1.0:
            update = update * scaling
        if result.dtype == update.dtype and result.shape == update.shape:
            # A frozen packed T2 linear returns a fresh GEMM output; its custom VJP
            # saves packed codes/scales, never this output. AddBackward needs
            # neither operand value. Preserve normal type promotion otherwise.
            return result.add_(update)
        return result + update

    @staticmethod
    def init(module, adapter_name):
        # Installation is after PEFT creates the ordinary A/B parameters.
        return None

    @staticmethod
    def merge_safe(module, active_adapter, orig_weight):
        raise RuntimeError('frozen T2 adapters must remain separate from the packed base')

    @staticmethod
    def merge_unsafe(module, active_adapter, orig_weight):
        raise RuntimeError('frozen T2 adapters must remain separate from the packed base')

    @staticmethod
    def unmerge(module, active_adapter, orig_weight):
        raise RuntimeError('frozen T2 adapters must remain separate from the packed base')


def install_identity_scale_lora(model: nn.Module, *, expected_count: int = 256) -> dict[str, Any]:
    """Validate the whole vanilla T2 adapter inventory before installing."""
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError('expected T2 adapter count must be a positive integer')
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, Linear) or not is_t2_linear(module.get_base_layer()):
            continue
        if module.merged or module.lora_variant or any(module.use_dora.values()):
            raise ValueError('identity-scale optimization requires unmerged vanilla T2 LoRA')
        if not module.lora_A or set(module.lora_A) != set(module.lora_B):
            raise ValueError('T2 LoRA requires matching nonempty A/B adapter names')
        targets.append((name, module))
    if len(targets) != expected_count:
        raise ValueError(f'T2 LoRA inventory mismatch: {len(targets)} != {expected_count}')
    for _, module in targets:
        for adapter_name in module.lora_A:
            module.lora_variant[adapter_name] = IdentityScaleLoRA()
    return {
        'implementation': 'PEFT0.21.2 LoraVariant',
        'recipe': 'identity_scale_and_owned_equal_dtype_output_add/v2',
        'layer_count': len(targets),
        'input_basis_counts': {
            basis: sum(t2_linear_input_basis(module.get_base_layer()) == basis for _, module in targets)
            for basis in ('primal', 'prism_rotated')
        },
        'parameter_and_state_namespace_preserved': True,
    }
