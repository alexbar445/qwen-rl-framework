"""Validation for complete, topology-bound distributed optimizer snapshots."""
from __future__ import annotations

from typing import Any

from .recovery import RecoveryError

OPTIMIZER_SCHEMA = "qwen-rl-parallel-optimizer/v1"


def validate_parallel_optimizer(state: Any, groups: list[dict], model: Any) -> None:
    if not isinstance(state, dict) or state.get("schema") != OPTIMIZER_SCHEMA:
        raise RecoveryError("invalid parallel optimizer snapshot schema")
    mode, size = state.get("parallelism"), state.get("world_size")
    if mode not in {"single", "tp", "ddp", "fsdp2", "pp"} or isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise RecoveryError("invalid parallel optimizer topology")
    if (mode == "single" and size != 1) or (mode in {"tp", "fsdp2", "pp"} and size < 2):
        raise RecoveryError("parallel optimizer mode disagrees with world size")
    if mode == "pp" and size != 4:
        raise RecoveryError("pipeline optimizer snapshots require exactly four stages")
    if state.get("parameter_groups") != groups:
        raise RecoveryError("parallel optimizer global parameter groups changed")
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    names = set(parameters)
    layouts, states = state.get("rank_layouts"), state.get("rank_states")
    if not isinstance(layouts, list) or not isinstance(states, list) or len(layouts) != size or len(states) != size:
        raise RecoveryError("parallel optimizer snapshot is missing rank state")
    if mode == "pp":
        rank_groups = state.get("rank_parameter_groups")
        if not isinstance(rank_groups, list) or len(rank_groups) != size:
            raise RecoveryError("pipeline optimizer snapshot is missing rank-local parameter groups")
        seen_names: set[str] = set()
        for rank, (layout, optimizer, expected_groups) in enumerate(
            zip(layouts, states, rank_groups, strict=True)
        ):
            if not isinstance(layout, dict) or layout.get("rank") != rank:
                raise RecoveryError("pipeline optimizer rank layout/order changed")
            entries = layout.get("parameters")
            if not isinstance(entries, list):
                raise RecoveryError("pipeline optimizer rank layout is malformed")
            local_names = {entry.get("name") for entry in entries if isinstance(entry, dict)}
            if (len(local_names) != len(entries) or not local_names
                    or any(not isinstance(name, str) or name not in names for name in local_names)):
                raise RecoveryError("pipeline optimizer rank has unknown or duplicate parameter names")
            if seen_names.intersection(local_names):
                raise RecoveryError("pipeline optimizer stage parameter inventories overlap")
            seen_names.update(local_names)
            shapes: dict[str, tuple[int, ...]] = {}
            for entry in entries:
                name, shape = entry["name"], entry.get("shape")
                if (not isinstance(shape, (tuple, list)) or not shape
                        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape)):
                    raise RecoveryError("pipeline optimizer has invalid local parameter geometry")
                expected_shape = tuple(parameters[name].shape)
                local_shape = tuple(shape)
                if local_shape != expected_shape:
                    raise RecoveryError("pipeline optimizer stage parameter geometry differs from its full adapter")
                if entry.get("dtype") != str(parameters[name].dtype):
                    raise RecoveryError("pipeline optimizer parameter dtype changed")
                shapes[name] = local_shape
            if not isinstance(expected_groups, list) or not expected_groups:
                raise RecoveryError("pipeline rank has no optimizer parameter groups")
            flattened_names: list[str] = []
            for group in expected_groups:
                if (not isinstance(group, dict) or not isinstance(group.get("options"), dict)
                        or not isinstance(group.get("parameter_names"), list)):
                    raise RecoveryError("pipeline rank optimizer parameter groups are malformed")
                flattened_names.extend(group["parameter_names"])
            if len(flattened_names) != len(set(flattened_names)) or set(flattened_names) != local_names:
                raise RecoveryError("pipeline rank optimizer groups do not cover its stage parameters exactly")
            if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict) or not isinstance(optimizer.get("param_groups"), list):
                raise RecoveryError("pipeline rank optimizer is not a complete state_dict")
            local_optimizer_groups = optimizer["param_groups"]
            if len(local_optimizer_groups) != len(expected_groups):
                raise RecoveryError("pipeline rank optimizer group count changed")
            ids: dict[Any, str] = {}
            for saved, expected in zip(local_optimizer_groups, expected_groups, strict=True):
                from .recovery import _jsonable
                actual_options = {key: _jsonable(value) for key, value in saved.items() if key != "params"}
                if actual_options != _jsonable(expected["options"]):
                    raise RecoveryError("pipeline rank optimizer hyperparameters changed")
                values = saved.get("params")
                expected_names = expected["parameter_names"]
                if not isinstance(values, list) or len(values) != len(expected_names):
                    raise RecoveryError("pipeline rank optimizer parameter ordering changed")
                for key, name in zip(values, expected_names, strict=True):
                    if key in ids:
                        raise RecoveryError("pipeline optimizer has duplicate parameter IDs")
                    ids[key] = name
            for key, values in optimizer["state"].items():
                if key not in ids or not isinstance(values, dict):
                    raise RecoveryError("pipeline optimizer has unknown parameter state")
                for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = values.get(field)
                    if value is not None and (not hasattr(value, "shape") or tuple(value.shape) != shapes[ids[key]]):
                        raise RecoveryError("pipeline optimizer moment shape differs from its local parameter")
        if seen_names != names:
            raise RecoveryError("pipeline optimizer stage inventories do not cover the full LoRA adapter")
        flattened_global = []
        for rank, local_groups in enumerate(rank_groups):
            flattened_global.extend(
                {"rank": rank, "parameter_names": group["parameter_names"], "options": group["options"]}
                for group in local_groups
            )
        if flattened_global != groups:
            raise RecoveryError("pipeline global optimizer groups differ from rank-local group inventory")
        return
    for rank, (layout, optimizer) in enumerate(zip(layouts, states, strict=True)):
        if not isinstance(layout, dict) or layout.get("rank") != rank:
            raise RecoveryError("parallel optimizer rank layout/order changed")
        entries = layout.get("parameters")
        if not isinstance(entries, list) or {e.get("name") for e in entries if isinstance(e, dict)} != names or len(entries) != len(names):
            raise RecoveryError("parallel optimizer rank parameter coverage is incomplete")
        shapes = {}
        for entry in entries:
            name, shape = entry["name"], entry.get("shape")
            if not isinstance(shape, (tuple, list)) or not shape or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape):
                raise RecoveryError("parallel optimizer has invalid local parameter geometry")
            global_shape = tuple(parameters[name].shape)
            local_shape = tuple(shape)
            if len(local_shape) != len(global_shape) or any(a > b for a, b in zip(local_shape, global_shape, strict=True)):
                raise RecoveryError("parallel optimizer local parameter geometry exceeds its global shape")
            if mode in {"single", "ddp"} and local_shape != global_shape:
                raise RecoveryError("replicated optimizer has sharded parameter geometry")
            if entry.get("dtype") != str(parameters[name].dtype):
                raise RecoveryError("parallel optimizer parameter dtype changed")
            shapes[name] = local_shape
        if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict) or not isinstance(optimizer.get("param_groups"), list):
            raise RecoveryError("parallel rank optimizer is not a complete state_dict")
        local_groups = optimizer["param_groups"]
        if len(local_groups) != len(groups):
            raise RecoveryError("parallel optimizer rank group count changed")
        ids = {}
        for saved, expected in zip(local_groups, groups, strict=True):
            from .recovery import _jsonable
            actual_options = {key: _jsonable(value) for key, value in saved.items() if key != "params"}
            if actual_options != _jsonable(expected["options"]):
                raise RecoveryError("parallel optimizer rank hyperparameters changed")
            values = saved.get("params")
            expected_names = expected["parameter_names"]
            if not isinstance(values, list) or len(values) != len(expected_names):
                raise RecoveryError("parallel optimizer rank parameter ordering changed")
            for key, name in zip(values, expected_names, strict=True):
                if key in ids:
                    raise RecoveryError("parallel optimizer has duplicate parameter IDs")
                ids[key] = name
        for key, values in optimizer["state"].items():
            if key not in ids or not isinstance(values, dict):
                raise RecoveryError("parallel optimizer has unknown parameter state")
            for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = values.get(field)
                if value is not None and (not hasattr(value, "shape") or tuple(value.shape) != shapes[ids[key]]):
                    raise RecoveryError("parallel optimizer moment shape differs from its local parameter")
