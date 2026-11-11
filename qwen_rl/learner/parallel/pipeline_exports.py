"""Bounded PP PEFT publication and optimizer-layout recovery helpers.

Kept separate from the active pipeline runner so these contracts can be staged
and reviewed without racing the training-loop owner.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from typing import Any

import torch
from torch import nn

PIPELINE_LAYER_RANGES = ((0, 16), (16, 32), (32, 48), (48, 64))


class PipelineExportError(ValueError):
    """A rank-local adapter/config/recovery layout cannot be safely merged."""


def _validated_layer_ranges(layer_ranges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not isinstance(layer_ranges, (list, tuple)) or len(layer_ranges) != 4:
        raise PipelineExportError("pipeline PEFT export requires four layer ranges")
    normalized = []
    expected_start = 0
    for rank, value in enumerate(layer_ranges):
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in value)):
            raise PipelineExportError(f"rank {rank} layer range must be a pair of integers")
        start, end = value
        if start != expected_start or end <= start:
            raise PipelineExportError("layer ranges must be contiguous, disjoint, and nonempty from layer zero")
        normalized.append((start, end))
        expected_start = end
    if normalized[0][0] != 0 or normalized[-1][1] != 64:
        raise PipelineExportError("pipeline PEFT export must cover exactly all 64 Qwen3.5 layers")
    return tuple(normalized)


def _config_dict(value: Any, rank: int) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    elif callable(getattr(value, "to_dict", None)):
        result = value.to_dict()
    else:
        raise PipelineExportError(f"rank {rank} PEFT config must be a mapping or expose to_dict()")
    if not isinstance(result, dict):
        raise PipelineExportError(f"rank {rank} PEFT config to_dict() did not return a dictionary")
    return result


def _normalize_target_name(target: Any) -> tuple[str, int]:
    if not isinstance(target, str) or not target or target != target.strip():
        raise PipelineExportError("PEFT target names must be nonempty, trimmed strings")
    candidates = [target]
    for prefix in ("base_model.model.", "base_model."):
        if target.startswith(prefix):
            candidates.append(target.removeprefix(prefix))
    for candidate in candidates:
        parts = candidate.split(".")
        if len(parts) >= 4 and parts[:2] == ["model", "layers"] and parts[2].isdigit():
            return candidate, int(parts[2])
    raise PipelineExportError(
        f"PP PEFT target has an unsupported module namespace: {target!r}; expected model.layers.N.*"
    )


def _freeze_value(value: Any) -> Any:
    """Convert config/optimizer values into a stable, hashable representation."""
    if isinstance(value, Enum):
        return (f"{type(value).__module__}.{type(value).__qualname__}", _freeze_value(value.value))
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return ("tensor", str(tensor.dtype), tuple(tensor.shape), _freeze_value(tensor.tolist()))
    if isinstance(value, Mapping):
        pairs = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise PipelineExportError("optimizer/config mapping keys must be strings")
            pairs.append((key, _freeze_value(item)))
        return tuple(sorted(pairs))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PipelineExportError("optimizer/config numeric options must be finite")
        return value
    raise PipelineExportError(f"unsupported optimizer/config option type: {type(value).__name__}")


def merge_pipeline_peft_configs(
    rank_configs: Sequence[Mapping[str, Any]],
    *,
    layer_ranges: Sequence[tuple[int, int]] = PIPELINE_LAYER_RANGES,
) -> dict[str, dict[str, Any]]:
    """Merge four disjoint rank-local target inventories into one PEFT config.

    Each input is the PEFT mapping ``{"default": config_dict}`` captured from
    one stage. Only exact ``model.layers.N.*`` targets are accepted; regex or
    suffix patterns cannot prove a complete, disjoint 64-layer inventory.
    """
    ranges = _validated_layer_ranges(layer_ranges)
    if not isinstance(rank_configs, (list, tuple)) or len(rank_configs) != 4:
        raise PipelineExportError("PEFT config merge requires exactly four rank-local configs")

    configs: list[dict[str, Any]] = []
    targets_by_rank: list[set[str]] = []
    layers_by_rank: list[set[int]] = []
    for rank, rank_config in enumerate(rank_configs):
        if not isinstance(rank_config, Mapping) or set(rank_config) != {"default"}:
            raise PipelineExportError("each pipeline rank must provide exactly one default PEFT config")
        config = _config_dict(rank_config["default"], rank)
        raw_targets = config.get("target_modules")
        if isinstance(raw_targets, str) or not isinstance(raw_targets, (list, tuple, set, frozenset)) or not raw_targets:
            raise PipelineExportError(f"rank {rank} config must contain exact target_modules")

        start, end = ranges[rank]
        local_targets: set[str] = set()
        local_layers: set[int] = set()
        for raw_target in raw_targets:
            target, layer_index = _normalize_target_name(raw_target)
            if not start <= layer_index < end:
                raise PipelineExportError(
                    f"rank {rank} target {raw_target!r} falls outside its layer range [{start}, {end})"
                )
            if target in local_targets:
                raise PipelineExportError(f"rank {rank} target inventory contains duplicate {target!r}")
            local_targets.add(target)
            local_layers.add(layer_index)
        targets_by_rank.append(local_targets)
        layers_by_rank.append(local_layers)
        configs.append(deepcopy(config))

    union: set[str] = set()
    for targets in targets_by_rank:
        overlap = union.intersection(targets)
        if overlap:
            raise PipelineExportError("rank target inventories overlap: " + ", ".join(sorted(overlap)[:8]))
        union.update(targets)

    covered_layers = set().union(*layers_by_rank)
    if covered_layers != set(range(64)):
        missing = sorted(set(range(64)) - covered_layers)
        raise PipelineExportError(f"PEFT target inventory does not cover all 64 layers: {missing[:8]}")

    base = configs[0]
    base_without_targets = {key: value for key, value in base.items() if key != "target_modules"}
    canonical_base = _freeze_value(base_without_targets)
    for rank, config in enumerate(configs[1:], start=1):
        non_target = {key: value for key, value in config.items() if key != "target_modules"}
        if _freeze_value(non_target) != canonical_base:
            raise PipelineExportError(f"rank {rank} PEFT settings differ from rank 0 outside target_modules")
    base["target_modules"] = sorted(
        union,
        key=lambda name: (int(name.split(".")[2]), name),
    )
    return {"default": base}


def _expected_adapter_names(config: Mapping[str, Any]) -> set[str]:
    rank = config.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise PipelineExportError("merged LoRA config requires one positive integer rank")
    if config.get("bias", "none") != "none" or bool(config.get("use_dora", False)):
        raise PipelineExportError("PP standard export supports bias-free ordinary LoRA only")
    if bool(config.get("lora_bias", False)):
        raise PipelineExportError("PP standard export does not support LoRA bias parameters")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise PipelineExportError("PP standard export does not support per-module rank/alpha patterns")
    if config.get("modules_to_save"):
        raise PipelineExportError("PP standard export does not support modules_to_save")
    targets = config.get("target_modules")
    if not isinstance(targets, list) or not targets:
        raise PipelineExportError("merged PEFT config has no global target_modules list")
    expected = set()
    for module_name in targets:
        normalized, _ = _normalize_target_name(module_name)
        prefix = f"base_model.model.{normalized}"
        expected.add(prefix + ".lora_A.default.weight")
        expected.add(prefix + ".lora_B.default.weight")
    return expected


def merge_pipeline_adapter_payload(
    rank_configs: Sequence[Mapping[str, Any]],
    rank_states: Sequence[Mapping[str, Any]],
    *,
    layer_ranges: Sequence[tuple[int, int]] = PIPELINE_LAYER_RANGES,
) -> dict[str, Any]:
    """Merge a standard, reloadable PEFT payload from four stage snapshots."""
    if not isinstance(rank_states, (list, tuple)) or len(rank_states) != 4:
        raise PipelineExportError("adapter state merge requires exactly four rank-local snapshots")
    merged_config = merge_pipeline_peft_configs(rank_configs, layer_ranges=layer_ranges)
    config = merged_config["default"]
    expected_names = _expected_adapter_names(config)
    try:
        from .pipeline import merge_pipeline_adapter_states

        adapter_state = merge_pipeline_adapter_states(rank_states, expected_names=expected_names)
    except Exception as error:
        if isinstance(error, PipelineExportError):
            raise
        raise PipelineExportError(f"cannot merge complete PP adapter tensors: {error}") from error

    lora_rank = int(config["r"])
    for module_name in config["target_modules"]:
        prefix = f"base_model.model.{module_name}"
        a = adapter_state[prefix + ".lora_A.default.weight"]
        b = adapter_state[prefix + ".lora_B.default.weight"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != lora_rank or b.shape[1] != lora_rank:
            raise PipelineExportError(f"LoRA A/B factors for {module_name!r} have incompatible shapes")
        if not a.is_floating_point() or not b.is_floating_point() or a.dtype != b.dtype:
            raise PipelineExportError(f"LoRA A/B factors for {module_name!r} have incompatible dtypes")
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise PipelineExportError(f"LoRA A/B factors for {module_name!r} contain non-finite values")
    return {"peft_config": merged_config, "adapter_state": adapter_state}


def _is_lora_name(name: str) -> bool:
    return any(part.startswith("lora_") for part in name.split("."))


def _optimizer_option(value: Any) -> Any:
    try:
        return _freeze_value(value)
    except PipelineExportError as error:
        raise PipelineExportError(f"optimizer option cannot be pinned in checkpoint signature: {error}") from error


def ordered_optimizer_parameter_group_signature(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[Any, ...]:
    """Return exact ordered optimizer identity for rank-local recovery."""
    if not isinstance(module, nn.Module) or not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("module and optimizer must be a torch module and optimizer")
    name_by_id: dict[int, str] = {}
    for name, parameter in module.named_parameters():
        if id(parameter) in name_by_id:
            raise PipelineExportError("module has aliased parameters with ambiguous optimizer names")
        name_by_id[id(parameter)] = name
    expected_trainable = {
        id(parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and _is_lora_name(name)
    }
    trainable_non_lora = [
        name for name, parameter in module.named_parameters()
        if parameter.requires_grad and not _is_lora_name(name)
    ]
    if trainable_non_lora:
        raise PipelineExportError("optimizer signature requires LoRA-only trainables: " + ", ".join(trainable_non_lora[:8]))
    if not expected_trainable:
        raise PipelineExportError("module has no trainable LoRA parameters")

    seen: set[int] = set()
    groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        params = group.get("params")
        if not isinstance(params, (list, tuple)):
            raise PipelineExportError(f"optimizer group {group_index} has malformed params")
        ordered_parameters = []
        for parameter in params:
            identity = id(parameter)
            name = name_by_id.get(identity)
            if name is None:
                raise PipelineExportError(f"optimizer group {group_index} contains an unknown parameter")
            if identity in seen:
                raise PipelineExportError(f"optimizer parameter {name!r} appears more than once")
            if identity not in expected_trainable or not parameter.requires_grad:
                raise PipelineExportError(f"optimizer parameter {name!r} is not a trainable LoRA parameter")
            seen.add(identity)
            ordered_parameters.append((name, tuple(parameter.shape), str(parameter.dtype)))
        options = tuple(
            (key, _optimizer_option(value))
            for key, value in sorted(group.items())
            if key != "params"
        )
        groups.append((tuple(ordered_parameters), options))
    missing = expected_trainable - seen
    if missing:
        missing_names = sorted(name_by_id[identity] for identity in missing)
        raise PipelineExportError("optimizer omits trainable LoRA parameters: " + ", ".join(missing_names[:8]))
    optimizer_class = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    return optimizer_class, tuple(groups)


def validate_ordered_optimizer_parameter_group_signature(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_signature: tuple[Any, ...],
) -> None:
    """Fail closed if checkpoint optimizer names/order/options/shape/dtype drift."""
    current = ordered_optimizer_parameter_group_signature(module, optimizer)
    if current != expected_signature:
        raise PipelineExportError("ordered optimizer parameter-group signature differs from checkpoint")


__all__ = [
    "PIPELINE_LAYER_RANGES",
    "PipelineExportError",
    "merge_pipeline_adapter_payload",
    "merge_pipeline_peft_configs",
    "ordered_optimizer_parameter_group_signature",
    "validate_ordered_optimizer_parameter_group_signature",
]
