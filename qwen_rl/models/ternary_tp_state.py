"""Strict adapter, gradient and optimizer state contracts for T2 tensor parallelism."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

TP_STATE_RECIPE = "qwen_rl_t2_tensor_parallel_state/v1"
TP_OPTIMIZER_SCHEMA = "qwen-rl-t2-tp-optimizer/v1"
_FIELDS = {
    "role", "global_shape", "local_shape", "shard_dim", "shard_start",
    "shard_end", "shard_segments", "tp_rank", "tp_world_size",
    "grad_reduce", "norm_count",
}


class T2TensorParallelStateError(ValueError):
    """A T2 TP state is incomplete or belongs to another layout."""


def canonical_parameter_name(name: str) -> str:
    """Remove the default adapter segment while preserving the model namespace."""
    if not isinstance(name, str) or not name or name != name.strip():
        raise T2TensorParallelStateError("parameter names must be non-empty trimmed strings")
    parts = name.split(".")
    indices = [
        index + 1 for index, part in enumerate(parts[:-1])
        if part in {"lora_A", "lora_B"} and parts[index + 1] == "default"
    ]
    if len(indices) > 1:
        raise T2TensorParallelStateError(f"ambiguous default adapter segment in {name}")
    if indices:
        del parts[indices[0]]
    return ".".join(parts)


def controller_parameter_name(canonical_name: str) -> str:
    """Convert one canonical bare LoRA A/B parameter FQN to its PEFT FQN."""
    name = canonical_parameter_name(canonical_name)
    if name != canonical_name:
        raise T2TensorParallelStateError(
            f"controller parameter conversion requires a canonical name without default segment: {canonical_name}"
        )
    module_name, factor = _standard_lora_factor(name)
    prefix = f"{module_name}." if module_name else ""
    return f"{prefix}{factor}.default.weight"


def controller_named_adapter_state(canonical_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add PEFT's default adapter segment for the rank-controller state view.

    Canonical collector keys intentionally omit this segment. The controller's
    local AdapterView uses live model FQNs, so conversion happens explicitly at
    that boundary and never changes tensor values.
    """
    if not isinstance(canonical_state, Mapping) or not canonical_state:
        raise T2TensorParallelStateError("canonical adapter state must be a non-empty mapping")
    result = {}
    for source_name, tensor in canonical_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise T2TensorParallelStateError("canonical adapter values must be tensors")
        target = controller_parameter_name(source_name)
        if target in result:
            raise T2TensorParallelStateError(f"controller adapter key collision: {target}")
        result[target] = tensor
    return result


def _shape(value: Any, field: str) -> tuple[int, ...]:
    if (not isinstance(value, (tuple, list)) or not value
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in value)):
        raise T2TensorParallelStateError(f"{field} must be a non-empty positive shape")
    return tuple(value)


def _segments(record: Mapping[str, Any], name: str, global_shape: tuple[int, ...],
              local_shape: tuple[int, ...], dim: int) -> list[tuple[int, int]]:
    raw = record["shard_segments"]
    start, end = record["shard_start"], record["shard_end"]
    if raw is None:
        if (isinstance(start, bool) or not isinstance(start, int)
                or isinstance(end, bool) or not isinstance(end, int)
                or not 0 <= start < end <= global_shape[dim]):
            raise T2TensorParallelStateError(f"invalid shard interval for {name}")
        result = [(start, end)]
    else:
        if start is not None or end is not None or not isinstance(raw, (list, tuple)) or not raw:
            raise T2TensorParallelStateError(f"invalid discontiguous shard metadata for {name}")
        result = []
        previous = -1
        for interval in raw:
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in interval)):
                raise T2TensorParallelStateError(f"invalid shard segment for {name}")
            left, right = interval
            if not 0 <= left < right <= global_shape[dim] or left < previous:
                raise T2TensorParallelStateError(f"unordered or invalid shard segments for {name}")
            result.append((left, right))
            previous = right
    if sum(right - left for left, right in result) != local_shape[dim]:
        raise T2TensorParallelStateError(f"shard segments do not match local geometry for {name}")
    return result


def _placement(record: Any, *, name: str, rank: int, world_size: int,
               local_shape: Sequence[int] | None = None) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise T2TensorParallelStateError(f"placement for {name} must be an object")
    missing, extra = _FIELDS - set(record), set(record) - _FIELDS
    if missing or extra:
        raise T2TensorParallelStateError(
            f"placement for {name} has invalid fields: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    role = record["role"]
    if role not in {"replicated", "sharded"}:
        raise T2TensorParallelStateError(f"unsupported placement role for {name}: {role!r}")
    global_shape = _shape(record["global_shape"], "global_shape")
    actual_local = _shape(record["local_shape"] if local_shape is None else local_shape, "local_shape")
    if _shape(record["local_shape"], "local_shape") != actual_local:
        raise T2TensorParallelStateError(f"placement local shape differs from parameter for {name}")
    if (isinstance(record["tp_rank"], bool) or not isinstance(record["tp_rank"], int)
            or record["tp_rank"] != rank or isinstance(record["tp_world_size"], bool)
            or not isinstance(record["tp_world_size"], int) or record["tp_world_size"] != world_size
            or world_size < 1 or not 0 <= rank < world_size):
        raise T2TensorParallelStateError(f"placement for {name} belongs to another rank or TP world")
    dim, start, end = record["shard_dim"], record["shard_start"], record["shard_end"]
    segments = record["shard_segments"]
    reduction, norm_count = record["grad_reduce"], record["norm_count"]
    if role == "replicated":
        if (actual_local != global_shape or dim is not None or start is not None
                or end is not None or segments is not None):
            raise T2TensorParallelStateError(f"replicated placement for {name} must cover the full tensor")
        if reduction not in {"sum_partial", "mean_complete"}:
            raise T2TensorParallelStateError(
                f"replicated parameter {name} requires sum_partial or mean_complete gradients"
            )
        if norm_count != "count_once":
            raise T2TensorParallelStateError(f"replicated parameter {name} must count its norm once")
    else:
        if isinstance(dim, bool) or not isinstance(dim, int) or not 0 <= dim < len(global_shape):
            raise T2TensorParallelStateError(f"invalid shard axis for {name}")
        if reduction != "none" or norm_count != "local_shard":
            raise T2TensorParallelStateError(
                f"sharded parameter {name} must use grad_reduce=none and norm_count=local_shard"
            )
        parts = _segments(record, name, global_shape, actual_local, dim)
        expected = list(global_shape)
        expected[dim] = sum(right - left for left, right in parts)
        if tuple(expected) != actual_local:
            raise T2TensorParallelStateError(f"shard geometry does not match local shape for {name}")
    return {
        "role": role,
        "global_shape": list(global_shape),
        "local_shape": list(actual_local),
        "shard_dim": dim,
        "shard_start": start,
        "shard_end": end,
        "shard_segments": None if segments is None else [list(x) for x in _segments(
            record, name, global_shape, actual_local, dim
        )],
        "tp_rank": rank,
        "tp_world_size": world_size,
        "grad_reduce": reduction,
        "norm_count": norm_count,
    }


def _trainables(model: nn.Module) -> dict[str, nn.Parameter]:
    result, ids = {}, {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        key = canonical_parameter_name(name)
        if key in result and result[key] is not parameter:
            raise T2TensorParallelStateError(f"duplicate canonical trainable name: {key}")
        previous = ids.get(id(parameter))
        if previous is not None and previous != key:
            raise T2TensorParallelStateError(f"tied trainable parameters need an explicit shared layout: {previous}, {key}")
        result[key], ids[id(parameter)] = parameter, key
    if not result:
        raise T2TensorParallelStateError("model has no trainable parameters")
    return result


def normalize_t2_tp_placements(
    model: nn.Module,
    placements: Mapping[str, Mapping[str, Any]],
    *,
    rank: int,
    world_size: int,
) -> dict[str, dict[str, Any]]:
    """Validate module-provided records against all local trainable parameters."""
    if not isinstance(placements, Mapping):
        raise T2TensorParallelStateError("TP placements must be a mapping")
    parameters, normalized = _trainables(model), {}
    for source_name, record in placements.items():
        name = canonical_parameter_name(source_name)
        if name in normalized or name not in parameters:
            raise T2TensorParallelStateError(f"duplicate or unexpected TP placement key: {name}")
        normalized[name] = _placement(
            record, name=name, rank=rank, world_size=world_size,
            local_shape=tuple(parameters[name].shape),
        )
    if set(normalized) != set(parameters):
        raise T2TensorParallelStateError(
            "TP placement inventory differs from trainables: "
            f"missing={sorted(set(parameters) - set(normalized))[:8]}, "
            f"unexpected={sorted(set(normalized) - set(parameters))[:8]}"
        )
    return dict(sorted(normalized.items()))


def _check_rank_placements(placements_by_rank):
    if not isinstance(placements_by_rank, (list, tuple)) or not placements_by_rank:
        raise T2TensorParallelStateError("one placement map is required for each TP rank")
    world_size, normalized = len(placements_by_rank), []
    for rank, mapping in enumerate(placements_by_rank):
        if not isinstance(mapping, Mapping) or not mapping:
            raise T2TensorParallelStateError(f"rank {rank} TP placement map is empty")
        row = {}
        for source_name, record in mapping.items():
            name = canonical_parameter_name(source_name)
            if name in row:
                raise T2TensorParallelStateError(f"duplicate placement for rank {rank}: {name}")
            row[name] = _placement(record, name=name, rank=rank, world_size=world_size)
        normalized.append(dict(sorted(row.items())))
    names = set(normalized[0])
    if any(set(row) != names for row in normalized[1:]):
        raise T2TensorParallelStateError("TP ranks have different trainable inventories")
    for name in sorted(names):
        rows = [row[name] for row in normalized]
        first = rows[0]
        for row in rows[1:]:
            if any(row[key] != first[key] for key in
                   ("role", "global_shape", "shard_dim", "grad_reduce", "norm_count")):
                raise T2TensorParallelStateError(f"placement semantics differ across ranks for {name}")
        if first["role"] == "replicated":
            continue
        dim = first["shard_dim"]
        intervals = sorted(
            interval for row in rows
            for interval in _segments(row, name, tuple(first["global_shape"]),
                                      tuple(row["local_shape"]), dim)
        )
        cursor = 0
        for left, right in intervals:
            if left != cursor:
                raise T2TensorParallelStateError(f"TP shards for {name} have a gap or overlap at {cursor}")
            cursor = right
        if cursor != first["global_shape"][dim]:
            raise T2TensorParallelStateError(f"TP shards for {name} do not cover the global tensor")
    return normalized, world_size


def t2_tp_layout_signature(placements_by_rank, *, recipe=TP_STATE_RECIPE) -> str:
    if not isinstance(recipe, str) or not recipe or recipe != recipe.strip():
        raise T2TensorParallelStateError("TP layout recipe must be a non-empty trimmed string")
    normalized, world_size = _check_rank_placements(placements_by_rank)
    value = {"recipe": recipe, "world_size": world_size, "placements": normalized}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _standard_lora_factor(name: str) -> tuple[str, str]:
    parts = name.split(".")
    indices = [
        index for index, part in enumerate(parts)
        if part in {"lora_A", "lora_B"}
    ]
    if (len(indices) != 1 or indices[0] != len(parts) - 2
            or parts[-1] != "weight"):
        raise T2TensorParallelStateError(
            f"trainable T2 LoRA parameter must end in one standard lora_A/B.weight FQN: {name}"
        )
    factor = parts[-2]
    module_name = ".".join(parts[:-2])
    return module_name, factor


def _named_local_lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Build the complete local adapter state from registered trainable A/B parameters."""
    trainables = _trainables(model)
    state = {}
    for name, parameter in trainables.items():
        if "lora_" not in name:
            continue
        _standard_lora_factor(name)
        if name in state:
            raise T2TensorParallelStateError(f"duplicate trainable LoRA parameter FQN: {name}")
        state[name] = parameter.detach().cpu().contiguous().clone()
    if not state:
        raise T2TensorParallelStateError("model has no trainable LoRA A/B parameters")
    return dict(sorted(state.items()))


def _canonical_peft_state(model, state):
    """Validate an externally supplied PEFT map against named local A/B parameters.

    This is an assertion-only path for callers that already have a serializer
    result. TP collection itself never depends on a potentially partial PEFT
    serializer view.
    """
    if not isinstance(state, Mapping):
        raise T2TensorParallelStateError("PEFT state must be a mapping")
    parameters = {
        name: parameter for name, parameter in _trainables(model).items()
        if "lora_" in name
    }
    result = {}
    for source_name, value in state.items():
        name = canonical_parameter_name(source_name)
        if name not in parameters or name in result or not isinstance(value, torch.Tensor):
            raise T2TensorParallelStateError(f"unexpected or duplicate PEFT state key: {name}")
        _standard_lora_factor(name)
        if tuple(value.shape) != tuple(parameters[name].shape) or value.dtype != parameters[name].dtype:
            raise T2TensorParallelStateError(f"PEFT tensor local shape/dtype differs from model for {name}")
        result[name] = value.detach().cpu().contiguous().clone()
    missing = sorted(set(parameters) - set(result))
    if missing:
        raise T2TensorParallelStateError(
            "provided PEFT state omits named trainable LoRA parameters: "
            + ", ".join(missing[:8])
        )
    return dict(sorted(result.items()))


def _assemble_local_chunks(local_value, record, target):
    dim = record["shard_dim"]
    local_cursor = 0
    segments = _segments(record, "adapter", tuple(record["global_shape"]),
                         tuple(record["local_shape"]), dim)
    for left, right in segments:
        width = right - left
        target.narrow(dim, left, width).copy_(local_value.narrow(dim, local_cursor, width))
        local_cursor += width


def _partition_full_tensor(full_value, record, target):
    dim = record["shard_dim"]
    local_cursor = 0
    for left, right in _segments(record, "adapter", tuple(record["global_shape"]),
                                 tuple(record["local_shape"]), dim):
        width = right - left
        target.narrow(dim, local_cursor, width).copy_(full_value.narrow(dim, left, width))
        local_cursor += width


def merge_t2_tp_peft_state_dicts(rank_states, placements_by_rank) -> dict[str, torch.Tensor]:
    """Merge unequal and discontiguous LoRA shards into standard full PEFT tensors."""
    placements, world_size = _check_rank_placements(placements_by_rank)
    if not isinstance(rank_states, (list, tuple)) or len(rank_states) != world_size:
        raise T2TensorParallelStateError("PEFT export is missing TP rank state")
    states = []
    for rank, state in enumerate(rank_states):
        if not isinstance(state, Mapping):
            raise T2TensorParallelStateError(f"rank {rank} PEFT state is malformed")
        row = {}
        for source_name, value in state.items():
            name = canonical_parameter_name(source_name)
            if name in row or not isinstance(value, torch.Tensor):
                raise T2TensorParallelStateError(f"invalid or duplicate PEFT tensor on rank {rank}: {name}")
            row[name] = value.detach().cpu().contiguous()
        expected = {name for name in placements[rank] if "lora_" in name}
        if set(row) != expected:
            raise T2TensorParallelStateError(f"rank {rank} PEFT inventory differs from its placement map")
        states.append(row)
    output = {}
    for name in sorted(key for key in placements[0] if "lora_" in key):
        records, values = [row[name] for row in placements], [row[name] for row in states]
        first, reference = records[0], values[0]
        if tuple(reference.shape) != tuple(first["local_shape"]):
            raise T2TensorParallelStateError(f"rank 0 tensor shape differs from placement for {name}")
        if any(value.dtype != reference.dtype for value in values[1:]):
            raise T2TensorParallelStateError(f"PEFT shard dtype differs across TP ranks for {name}")
        global_value = torch.empty(tuple(first["global_shape"]), dtype=reference.dtype)
        if first["role"] == "replicated":
            for rank, (record, value) in enumerate(zip(records, values, strict=True)):
                if tuple(value.shape) != tuple(record["local_shape"]):
                    raise T2TensorParallelStateError(f"replicated tensor shape differs on rank {rank}: {name}")
                if not torch.equal(value, reference):
                    raise T2TensorParallelStateError(f"replicated adapter differs on rank {rank}: {name}")
            global_value.copy_(reference)
        else:
            for rank, (record, value) in enumerate(zip(records, values, strict=True)):
                if tuple(value.shape) != tuple(record["local_shape"]):
                    raise T2TensorParallelStateError(f"shard tensor shape differs on rank {rank}: {name}")
                _assemble_local_chunks(value, record, global_value)
        output[name] = global_value
    if not output:
        raise T2TensorParallelStateError("T2 TP export has no LoRA parameters")
    return output


def collect_t2_tp_peft_state_dict(
    model, placements, *, rank, world_size, group=None, collection_rank=0, local_state=None
):
    """Gather local PEFT states; only collection_rank receives a full adapter."""
    local_layout = normalize_t2_tp_placements(
        model, placements, rank=rank, world_size=world_size
    )
    if not 0 <= collection_rank < world_size:
        raise ValueError("collection_rank is outside the TP group")
    if dist.is_initialized():
        if (dist.get_rank(group), dist.get_world_size(group)) != (rank, world_size):
            raise T2TensorParallelStateError("active process group differs from PEFT collection topology")
    elif world_size != 1 or rank != 0:
        raise T2TensorParallelStateError("multi-rank PEFT collection requires a process group")
    # Paired T2 modules intentionally expose bare lora_A/B.weight parameters,
    # outside PEFT BaseTunerLayer's adapter-name ModuleDict. Build state from the
    # authoritative trainable model inventory, never from PEFT's selected-adapter
    # serializer (which may omit those bare factors).
    state = _named_local_lora_state(model)
    expected = {name for name in local_layout if "lora_" in name}
    if set(state) != expected:
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        raise T2TensorParallelStateError(
            "named trainable LoRA state differs from placement inventory: "
            f"missing={missing[:8]}, unexpected={extra[:8]}"
        )
    if local_state is not None:
        asserted_state = _canonical_peft_state(model, local_state)
        mismatched = sorted(
            name for name in state
            if not torch.equal(state[name], asserted_state[name])
        )
        if mismatched:
            raise T2TensorParallelStateError(
                "provided PEFT state differs from named trainable LoRA parameters: "
                + ", ".join(mismatched[:8])
            )
    if world_size == 1:
        return merge_t2_tp_peft_state_dicts([state], [local_layout])
    gathered = [None] * world_size if rank == collection_rank else None
    dist.gather_object((state, local_layout), gathered, dst=collection_rank, group=group)
    if rank != collection_rank:
        return None
    return merge_t2_tp_peft_state_dicts(
        [item[0] for item in gathered], [item[1] for item in gathered]
    )


def load_t2_tp_peft_state_dict(model, state, placements, *, rank, world_size) -> None:
    """Initialize local TP adapter tensors from a full standard PEFT state."""
    local = normalize_t2_tp_placements(model, placements, rank=rank, world_size=world_size)
    parameters = _trainables(model)
    expected = {name for name in local if "lora_" in name}
    canonical = {}
    for source_name, value in state.items():
        name = canonical_parameter_name(source_name)
        if name in canonical or not isinstance(value, torch.Tensor):
            raise T2TensorParallelStateError(f"invalid or duplicate full PEFT tensor: {name}")
        canonical[name] = value.detach().cpu()
    if set(canonical) != expected:
        raise T2TensorParallelStateError("full PEFT adapter inventory differs from TP model")
    with torch.no_grad():
        for name in sorted(expected):
            record, full, parameter = local[name], canonical[name], parameters[name]
            if tuple(full.shape) != tuple(record["global_shape"]) or full.dtype != parameter.dtype:
                raise T2TensorParallelStateError(f"full PEFT global shape/dtype differs for {name}")
            if record["role"] == "replicated":
                local_value = full
            else:
                local_value = torch.empty(tuple(record["local_shape"]), dtype=full.dtype)
                _partition_full_tensor(full, record, local_value)
            if tuple(local_value.shape) != tuple(parameter.shape):
                raise T2TensorParallelStateError(f"partitioned PEFT shape differs from parameter for {name}")
            parameter.copy_(local_value.to(device=parameter.device))


def reduce_t2_tp_gradients_(model, placements, *, rank, world_size, group=None):
    """Sum partial replicated gradients and average already-complete replicas."""
    local = normalize_t2_tp_placements(model, placements, rank=rank, world_size=world_size)
    parameters = _trainables(model)
    if dist.is_initialized():
        if (dist.get_rank(group), dist.get_world_size(group)) != (rank, world_size):
            raise T2TensorParallelStateError("gradient process group differs from placement topology")
    elif world_size != 1:
        raise T2TensorParallelStateError("multi-rank gradient reduction requires a process group")
    missing = [name for name, value in parameters.items() if value.grad is None]
    if world_size > 1:
        reports = [None] * world_size
        dist.all_gather_object(reports, missing, group=group)
        missing = sorted({name for row in reports for name in row})
    if missing:
        raise T2TensorParallelStateError(f"TP reduction is missing trainable gradients: {missing[:8]}")
    if any(not torch.isfinite(value.grad).all() for value in parameters.values()):
        raise FloatingPointError("non-finite T2 TP gradient")
    reduced = 0
    with torch.no_grad():
        for name in sorted(local):
            record, gradient = local[name], parameters[name].grad
            if record["role"] == "sharded" or world_size == 1:
                continue
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=group)
            if record["grad_reduce"] == "mean_complete":
                gradient.div_(world_size)
            reduced += 1
    return {
        "collectives": reduced,
        "sum_partial": sum(r["role"] == "replicated" and r["grad_reduce"] == "sum_partial" for r in local.values()),
        "mean_complete": sum(r["role"] == "replicated" and r["grad_reduce"] == "mean_complete" for r in local.values()),
        "local_shards": sum(r["role"] == "sharded" for r in local.values()),
    }


def clip_t2_tp_grad_norm_(
    model, placements, max_norm, *, rank, world_size, group=None, error_if_nonfinite=True
):
    """Clip the global norm; count full replicas once and every local shard."""
    if (isinstance(max_norm, bool) or not isinstance(max_norm, (int, float))
            or not math.isfinite(max_norm) or max_norm <= 0):
        raise ValueError("max_norm must be finite and positive")
    if type(error_if_nonfinite) is not bool:
        raise ValueError("error_if_nonfinite must be boolean")
    local = normalize_t2_tp_placements(model, placements, rank=rank, world_size=world_size)
    parameters = _trainables(model)
    if dist.is_initialized():
        if (dist.get_rank(group), dist.get_world_size(group)) != (rank, world_size):
            raise T2TensorParallelStateError("norm process group differs from placement topology")
    elif world_size != 1:
        raise T2TensorParallelStateError("multi-rank norm requires a process group")
    missing = [name for name, value in parameters.items() if value.grad is None]
    if world_size > 1:
        reports = [None] * world_size
        dist.all_gather_object(reports, missing, group=group)
        missing = sorted({name for row in reports for name in row})
    if missing:
        raise T2TensorParallelStateError(f"global norm is missing trainable gradients: {missing[:8]}")
    first = next(iter(parameters.values()))
    square = torch.zeros((), device=first.device, dtype=torch.float64)
    nonfinite = False
    for name, parameter in parameters.items():
        record = local[name]
        if record["role"] == "replicated" and rank != 0:
            continue
        gradient = parameter.grad.detach()
        nonfinite = nonfinite or not bool(torch.isfinite(gradient).all())
        square += gradient.double().square().sum()
    if world_size > 1:
        flags = [None] * world_size
        dist.all_gather_object(flags, nonfinite, group=group)
        nonfinite = any(flags)
    if nonfinite and error_if_nonfinite:
        raise FloatingPointError("non-finite T2 TP global gradient norm")
    if world_size > 1:
        dist.all_reduce(square, op=dist.ReduceOp.SUM, group=group)
    norm = float(square.sqrt().item())
    scale = min(1.0, float(max_norm) / (norm + 1e-6))
    with torch.no_grad():
        for parameter in parameters.values():
            parameter.grad.mul_(scale)
    return {
        "gradient_norm_before_clip": norm,
        "clip_coefficient": scale,
        "max_norm": float(max_norm),
        "replicated_norm_count": "once",
        "sharded_norm_count": "every_local_shard",
    }


def _metadata(value):
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise T2TensorParallelStateError("optimizer metadata cannot contain NaN or infinity")
        return value
    if hasattr(value, "item"):
        return _metadata(value.item())
    raise T2TensorParallelStateError(f"unsupported optimizer metadata value: {type(value).__name__}")


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _optimizer_bindings(model, optimizer):
    parameters = _trainables(model)
    name_by_id = {id(parameter): name for name, parameter in parameters.items()}
    raw = optimizer.state_dict()
    groups, seen_names, seen_ids = [], set(), set()
    if len(raw["param_groups"]) != len(optimizer.param_groups):
        raise T2TensorParallelStateError("optimizer group count differs from live optimizer")
    for live, saved in zip(optimizer.param_groups, raw["param_groups"], strict=True):
        names = []
        for parameter, state_id in zip(live["params"], saved["params"], strict=True):
            name = name_by_id.get(id(parameter))
            if name is None or name in seen_names or state_id in seen_ids:
                raise T2TensorParallelStateError("optimizer parameter mapping is missing or duplicated")
            names.append(name)
            seen_names.add(name)
            seen_ids.add(state_id)
        options = {key: _metadata(value) for key, value in live.items() if key != "params"}
        groups.append({"parameter_names": names, "options": options})
    if seen_names != set(parameters):
        raise T2TensorParallelStateError("optimizer does not cover every trainable parameter")
    name_by_state_id = {
        state_id: name
        for group, saved in zip(groups, raw["param_groups"], strict=True)
        for name, state_id in zip(group["parameter_names"], saved["params"], strict=True)
    }
    return parameters, groups, raw, name_by_state_id


def capture_t2_tp_optimizer_state(
    model, optimizer, placements, *, rank, world_size, topology_signature,
    layout_recipe=TP_STATE_RECIPE,
):
    """Capture exact rank-local FQN order, geometry, options and optimizer moments."""
    if not isinstance(topology_signature, str) or len(topology_signature) != 64:
        raise T2TensorParallelStateError("optimizer snapshot needs a SHA256 TP layout signature")
    if not isinstance(layout_recipe, str) or not layout_recipe or layout_recipe != layout_recipe.strip():
        raise T2TensorParallelStateError("optimizer snapshot needs a non-empty TP layout recipe")
    local = normalize_t2_tp_placements(model, placements, rank=rank, world_size=world_size)
    parameters, groups, raw, id_to_name = _optimizer_bindings(model, optimizer)
    rows = [{
        "name": name,
        "shape": list(parameters[name].shape),
        "dtype": str(parameters[name].dtype),
        "placement": local[name],
    } for name in sorted(parameters)]
    return {
        "schema": TP_OPTIMIZER_SCHEMA,
        "recipe": TP_STATE_RECIPE,
        "layout_recipe": layout_recipe,
        "rank": rank,
        "world_size": world_size,
        "topology_signature": topology_signature,
        "parameters": rows,
        "parameter_groups": groups,
        "state_parameter_names": {str(key): value for key, value in id_to_name.items()},
        "optimizer_state": _cpu_tree(raw),
    }


def validate_t2_tp_optimizer_state(
    snapshot, model, optimizer, placements, *, rank, world_size, topology_signature,
    layout_recipe=TP_STATE_RECIPE,
):
    if not isinstance(snapshot, Mapping) or snapshot.get("schema") != TP_OPTIMIZER_SCHEMA:
        raise T2TensorParallelStateError("invalid T2 TP optimizer schema")
    expected = capture_t2_tp_optimizer_state(
        model, optimizer, placements, rank=rank, world_size=world_size,
        topology_signature=topology_signature, layout_recipe=layout_recipe,
    )
    for key in ("recipe", "layout_recipe", "rank", "world_size", "topology_signature",
                "parameters", "parameter_groups", "state_parameter_names"):
        if snapshot.get(key) != expected[key]:
            raise T2TensorParallelStateError(f"T2 TP optimizer {key} differs from active layout")
    state = snapshot.get("optimizer_state")
    if not isinstance(state, Mapping) or not isinstance(state.get("state"), Mapping):
        raise T2TensorParallelStateError("T2 TP optimizer snapshot has no state mapping")
    if state.get("param_groups") != expected["optimizer_state"]["param_groups"]:
        raise T2TensorParallelStateError("T2 TP optimizer parameter order/hyperparameters changed")
    id_to_name = {int(key): value for key, value in snapshot["state_parameter_names"].items()}
    if set(state["state"]) - set(id_to_name):
        raise T2TensorParallelStateError("optimizer snapshot contains unknown parameter state")
    shapes = {row["name"]: tuple(row["shape"]) for row in snapshot["parameters"]}
    for parameter_id, values in state["state"].items():
        name = id_to_name[parameter_id]
        if not isinstance(values, Mapping):
            raise T2TensorParallelStateError("optimizer parameter state is malformed")
        for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            tensor = values.get(field)
            if tensor is not None and (
                not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shapes[name]
            ):
                raise T2TensorParallelStateError(f"optimizer {field} shape differs from {name}")


def _validate_rank_optimizer_payload(snapshot):
    parameters = {row["name"]: row for row in snapshot["parameters"]}
    groups = snapshot["parameter_groups"]
    name_map = snapshot.get("state_parameter_names")
    state = snapshot.get("optimizer_state")
    if not isinstance(groups, list) or not isinstance(name_map, Mapping):
        raise T2TensorParallelStateError("TP optimizer mapping metadata is malformed")
    if not isinstance(state, Mapping) or not isinstance(state.get("state"), Mapping):
        raise T2TensorParallelStateError("TP optimizer rank snapshot has no state dictionary")
    saved_groups = state.get("param_groups")
    if not isinstance(saved_groups, list) or len(saved_groups) != len(groups):
        raise T2TensorParallelStateError("TP optimizer state group inventory changed")
    ids, mapped_names = set(), set()
    for saved, expected in zip(saved_groups, groups, strict=True):
        if not isinstance(expected, Mapping) or not isinstance(expected.get("parameter_names"), list):
            raise T2TensorParallelStateError("TP optimizer parameter group metadata is malformed")
        names, params = expected["parameter_names"], saved.get("params")
        options = {key: _metadata(value) for key, value in saved.items() if key != "params"}
        if options != expected.get("options") or not isinstance(params, list) or len(params) != len(names):
            raise T2TensorParallelStateError("TP optimizer group options or parameter order changed")
        for parameter_id, name in zip(params, names, strict=True):
            if parameter_id in ids or name not in parameters or name in mapped_names:
                raise T2TensorParallelStateError("TP optimizer ID-to-FQN map is duplicate or unknown")
            if name_map.get(str(parameter_id)) != name:
                raise T2TensorParallelStateError("TP optimizer saved ID does not map to its ordered FQN")
            ids.add(parameter_id)
            mapped_names.add(name)
    if mapped_names != set(parameters) or set(name_map) != {str(value) for value in ids}:
        raise T2TensorParallelStateError("TP optimizer ID-to-FQN map is incomplete")
    if set(state["state"]) - ids:
        raise T2TensorParallelStateError("TP optimizer has state for an unknown parameter ID")
    for parameter_id, values in state["state"].items():
        if not isinstance(values, Mapping):
            raise T2TensorParallelStateError("TP optimizer parameter state is malformed")
        name = name_map[str(parameter_id)]
        shape = tuple(parameters[name]["shape"])
        for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            tensor = values.get(field)
            if tensor is not None and (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape):
                raise T2TensorParallelStateError(f"TP optimizer {field} shape differs from {name}")


def _state_value_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_state_value_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_state_value_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def validate_t2_tp_optimizer_world(
    rank_snapshots, *, topology_signature, layout_recipe=TP_STATE_RECIPE
):
    """Fail closed unless every ordered TP rank has a matching complete state."""
    if not isinstance(rank_snapshots, (list, tuple)) or not rank_snapshots:
        raise T2TensorParallelStateError("TP optimizer snapshot is missing rank state")
    world_size = len(rank_snapshots)
    if not isinstance(topology_signature, str) or len(topology_signature) != 64:
        raise T2TensorParallelStateError("TP optimizer world needs a SHA256 layout signature")
    if not isinstance(layout_recipe, str) or not layout_recipe or layout_recipe != layout_recipe.strip():
        raise T2TensorParallelStateError("TP optimizer world needs a non-empty layout recipe")
    rows, shared_groups = [], None
    for rank, snapshot in enumerate(rank_snapshots):
        if (not isinstance(snapshot, Mapping) or snapshot.get("schema") != TP_OPTIMIZER_SCHEMA
                or snapshot.get("recipe") != TP_STATE_RECIPE
                or snapshot.get("layout_recipe") != layout_recipe or snapshot.get("rank") != rank
                or snapshot.get("world_size") != world_size
                or snapshot.get("topology_signature") != topology_signature):
            raise T2TensorParallelStateError("TP optimizer rank identity/order/topology changed")
        groups = snapshot.get("parameter_groups")
        if shared_groups is None:
            shared_groups = groups
        elif groups != shared_groups:
            raise T2TensorParallelStateError("TP optimizer group mappings differ across ranks")
        parameters = snapshot.get("parameters")
        if not isinstance(parameters, list):
            raise T2TensorParallelStateError("TP optimizer parameter layout is malformed")
        row = {}
        for item in parameters:
            if not isinstance(item, Mapping) or item.get("name") in row:
                raise T2TensorParallelStateError("duplicate/malformed TP optimizer parameter")
            row[item["name"]] = item
        _validate_rank_optimizer_payload(snapshot)
        rows.append(row)
    names = set(rows[0])
    if any(set(row) != names for row in rows[1:]):
        raise T2TensorParallelStateError("TP optimizer parameter inventories differ across ranks")
    placement_maps = [
        {name: row[name]["placement"] for name in sorted(names)}
        for row in rows
    ]
    if t2_tp_layout_signature(placement_maps, recipe=layout_recipe) != topology_signature:
        raise T2TensorParallelStateError("TP optimizer topology signature does not match rank placements")
    for name in sorted(names):
        records = [row[name]["placement"] for row in rows]
        _check_rank_placements([{name: record} for record in records])
        first = rows[0][name]
        for rank, row in enumerate(rows):
            if row[name].get("dtype") != first.get("dtype"):
                raise T2TensorParallelStateError(f"TP optimizer dtype differs across ranks: {name}")
            if tuple(row[name].get("shape", ())) != tuple(records[rank]["local_shape"]):
                raise T2TensorParallelStateError(f"TP optimizer shape differs from placement: {name}")
        if records[0]["role"] == "replicated":
            states_by_rank = []
            for snapshot in rank_snapshots:
                id_to_name = {
                    value: int(key) for key, value in snapshot["state_parameter_names"].items()
                }
                optimizer_state = snapshot["optimizer_state"]["state"]
                name_to_state = {
                    parameter_name: optimizer_state.get(parameter_id, {})
                    for parameter_name, parameter_id in id_to_name.items()
                }
                states_by_rank.append(name_to_state[name])
            if any(not _state_value_equal(states_by_rank[0], value) for value in states_by_rank[1:]):
                raise T2TensorParallelStateError(
                    f"replicated TP optimizer moments differ across ranks for {name}"
                )


def restore_t2_tp_optimizer_state(
    model, optimizer, snapshot, placements, *, rank, world_size, topology_signature,
    layout_recipe=TP_STATE_RECIPE,
):
    """Restore only a complete optimizer snapshot matching this TP rank exactly."""
    validate_t2_tp_optimizer_state(
        snapshot, model, optimizer, placements, rank=rank, world_size=world_size,
        topology_signature=topology_signature, layout_recipe=layout_recipe,
    )
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer_state"]))
