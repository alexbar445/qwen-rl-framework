"""Explicit, read-only execution diagnostics for the complete T2 TP model.

This module is imported only for the opt-in ``execution_diagnostics`` command.
It does not run during model startup, snapshot collection, or restore.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

EXECUTION_DIAGNOSTICS_SCHEMA = "qwen-rl-t2-tp-execution-diagnostics/v1"
HASH_TILE_BYTES = 32 * 1024 * 1024


def _tensor_tiles(tensor, *, max_elements: int) -> Iterator[Any]:
    """Yield logical-order tensor views bounded by ``max_elements``.

    ``narrow`` and ``select`` keep non-contiguous tensors as views. A later
    per-tile CPU conversion can therefore never materialize a full model tensor.
    """
    if tensor.numel() == 0:
        return
    if tensor.numel() <= max_elements:
        yield tensor
        return
    if tensor.ndim == 0:
        yield tensor
        return

    first = int(tensor.shape[0])
    if first <= 1:
        yield from _tensor_tiles(tensor.select(0, 0), max_elements=max_elements)
        return
    trailing = math.prod(int(size) for size in tensor.shape[1:])
    rows_per_tile = max(1, max_elements // trailing) if trailing <= max_elements else 1
    for start in range(0, first, rows_per_tile):
        count = min(rows_per_tile, first - start)
        yield from _tensor_tiles(tensor.narrow(0, start, count), max_elements=max_elements)


def iter_tensor_byte_chunks(
    tensor,
    *,
    tile_bytes: int = HASH_TILE_BYTES,
) -> Iterator[bytes]:
    """Yield canonical tensor bytes in bounded CPU chunks.

    The byte stream follows logical row-major tensor order and excludes device
    placement, storage offsets, strides, and addresses. Each temporary CPU tile
    is at most ``tile_bytes``; unsupported sparse/non-strided tensors fail closed.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("diagnostic tensor hashing requires a torch.Tensor")
    if tensor.layout != torch.strided:
        raise TypeError("diagnostic tensor hashing only supports strided tensors")
    if isinstance(tile_bytes, bool) or not isinstance(tile_bytes, int) or tile_bytes <= 0:
        raise ValueError("tile_bytes must be a positive integer")
    element_size = int(tensor.element_size())
    if tile_bytes < element_size:
        raise ValueError("tile_bytes must fit at least one tensor element")
    max_elements = max(1, tile_bytes // element_size)
    for tile in _tensor_tiles(tensor.detach(), max_elements=max_elements):
        cpu_tile = tile.to(device="cpu")
        if not cpu_tile.is_contiguous():
            cpu_tile = cpu_tile.contiguous()
        raw = cpu_tile.view(torch.uint8).reshape(-1).numpy().tobytes()
        for offset in range(0, len(raw), tile_bytes):
            yield raw[offset : offset + tile_bytes]


def tensor_sha256(tensor, *, tile_bytes: int = HASH_TILE_BYTES) -> str:
    digest = hashlib.sha256()
    for chunk in iter_tensor_byte_chunks(tensor, tile_bytes=tile_bytes):
        digest.update(chunk)
    return digest.hexdigest()


def _tensor_record(name: str, tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype),
        "sha256": tensor_sha256(tensor),
    }


def _named_tensors(model, method_name: str, *, predicate=None) -> list[tuple[str, Any]]:
    method = getattr(model, method_name, None)
    if not callable(method):
        return []
    try:
        entries = method(remove_duplicate=False)
    except TypeError:
        entries = method()
    result = []
    for name, value in entries:
        if predicate is None or predicate(value):
            result.append((str(name), value))
    return sorted(result, key=lambda item: item[0])


def _safe_metadata_value(value: Any) -> Any:
    """Return address-free JSON metadata for known scalar/sequence attributes."""
    if isinstance(value, Enum):
        return _safe_metadata_value(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (tuple, list)):
        return [_safe_metadata_value(item) for item in value]
    return None


def _safe_call(function, *args) -> Any:
    if not callable(function):
        return None
    try:
        return function(*args)
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        return None


def _torch_flags() -> dict[str, Any]:
    import torch

    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    preferred_blas = getattr(torch.backends.cuda, "preferred_blas_library", None)
    preferred_blas_value = _safe_call(preferred_blas)
    return {
        "version": str(torch.__version__),
        "default_dtype": str(torch.get_default_dtype()),
        "allow_tf32_matmul": _safe_metadata_value(getattr(matmul, "allow_tf32", None)),
        "allow_tf32_cudnn": _safe_metadata_value(getattr(cudnn, "allow_tf32", None)),
        "allow_bf16_reduced_precision_reduction": _safe_metadata_value(
            getattr(matmul, "allow_bf16_reduced_precision_reduction", None)
        ),
        "allow_fp16_reduced_precision_reduction": _safe_metadata_value(
            getattr(matmul, "allow_fp16_reduced_precision_reduction", None)
        ),
        "preferred_blas_library": None if preferred_blas_value is None else str(preferred_blas_value),
        "deterministic_algorithms": _safe_call(torch.are_deterministic_algorithms_enabled),
        "deterministic_warn_only": _safe_call(
            getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
        ),
        "cudnn_deterministic": _safe_metadata_value(getattr(cudnn, "deterministic", None)),
        "cudnn_benchmark": _safe_metadata_value(getattr(cudnn, "benchmark", None)),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "flash_attention_deterministic": os.environ.get("FLASH_ATTENTION_DETERMINISTIC"),
    }


def _autocast_metadata(trainer) -> dict[str, Any]:
    import torch

    args = getattr(trainer, "args", None)
    try:
        cuda_enabled = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        cuda_enabled = bool(torch.is_autocast_enabled())
    try:
        cpu_enabled = bool(torch.is_autocast_enabled("cpu"))
    except TypeError:
        cpu_enabled = None
    get_dtype = getattr(torch, "get_autocast_dtype", None)
    cuda_dtype = _safe_call(get_dtype, "cuda")
    cpu_dtype = _safe_call(get_dtype, "cpu")
    return {
        "cuda_enabled": cuda_enabled,
        "cpu_enabled": cpu_enabled,
        "cuda_dtype": None if cuda_dtype is None else str(cuda_dtype),
        "cpu_dtype": None if cpu_dtype is None else str(cpu_dtype),
        "trainer_args_bf16": _safe_metadata_value(getattr(args, "bf16", None)),
        "trainer_temperature": _safe_metadata_value(getattr(trainer, "temperature", None)),
    }


def _base_model(model):
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        candidate = getter()
        if candidate is not None:
            return candidate
    return model


def _head_metadata(base_model) -> dict[str, Any]:
    getter = getattr(base_model, "get_output_embeddings", None)
    head = _safe_call(getter) if callable(getter) else None
    if head is None:
        head = getattr(base_model, "lm_head", None)
    return {
        "temperature": _safe_metadata_value(getattr(base_model, "_t2_tp_temperature", None)),
        "logit_scale": _safe_metadata_value(getattr(base_model, "_t2_tp_logit_scale", None)),
        "softcap": _safe_metadata_value(getattr(base_model, "_t2_tp_softcap", None)),
        "backend": _safe_metadata_value(getattr(head, "scoring_backend", None)),
        "chunk_rows": _safe_metadata_value(getattr(head, "chunk_rows", None)),
        "max_logits_bytes": _safe_metadata_value(getattr(head, "max_logits_bytes", None)),
    }


def _module_modes(model) -> list[dict[str, Any]]:
    entries = []
    for name, module in _named_modules(model):
        entries.append({"name": name, "training": bool(module.training)})
    return entries


def _named_modules(model) -> Iterable[tuple[str, Any]]:
    method = getattr(model, "named_modules", None)
    if not callable(method):
        return ()
    try:
        return sorted(method(remove_duplicate=False), key=lambda item: str(item[0]))
    except TypeError:
        return sorted(method(), key=lambda item: str(item[0]))


def _rope_runtime(model, named_buffers: list[tuple[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    buffer_names = {name for name, _ in named_buffers}
    rope_records: list[dict[str, Any]] = []
    tensor_attributes: list[dict[str, Any]] = []
    for module_name, module in _named_modules(model):
        module_type = type(module).__name__
        if "RotaryEmbedding" not in module_type:
            continue
        row: dict[str, Any] = {
            "module_name": str(module_name),
            "module_type": f"{type(module).__module__}.{module_type}",
            "rope_type": _safe_metadata_value(getattr(module, "rope_type", None)),
            "max_seq_len_cached": _safe_metadata_value(getattr(module, "max_seq_len_cached", None)),
            "original_max_seq_len": _safe_metadata_value(getattr(module, "original_max_seq_len", None)),
            "attention_scaling": _safe_metadata_value(getattr(module, "attention_scaling", None)),
            "mrope_section": _safe_metadata_value(getattr(module, "mrope_section", None)),
            "tensor_attributes": {},
        }
        for attribute in ("inv_freq", "original_inv_freq"):
            value = getattr(module, attribute, None)
            if not isinstance(value, torch.Tensor):
                continue
            qualified_name = f"{module_name}.{attribute}" if module_name else attribute
            if qualified_name in buffer_names:
                row["tensor_attributes"][attribute] = {
                    "source": "named_buffer",
                    "name": qualified_name,
                }
            else:
                record_name = f"rope_attribute:{qualified_name}"
                tensor_attributes.append(_tensor_record(record_name, value))
                row["tensor_attributes"][attribute] = {
                    "source": "module_attribute",
                    "name": record_name,
                }
        rope_records.append(row)
    return rope_records, tensor_attributes


def _gdn_callables(model) -> list[dict[str, Any]]:
    result = []
    for name, module in _named_modules(model):
        owner_type = type(module)
        owner_name = owner_type.__name__
        if "gateddeltanet" not in owner_name.lower() and "gated_delta_net" not in owner_name.lower():
            continue
        forward = getattr(module, "forward", None)
        target = getattr(forward, "__func__", forward)
        result.append({
            "name": str(name),
            "owner_type": f"{owner_type.__module__}.{owner_name}",
            "module": getattr(target, "__module__", None),
            "qualname": getattr(target, "__qualname__", None),
        })
    return result


def _autotune_json_value(value: Any) -> Any:
    """Keep Triton cache keys address-free and JSON-safe."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, Enum):
        return _autotune_json_value(value.value)
    if isinstance(value, (tuple, list)):
        return [_autotune_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _autotune_json_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "name": name}
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _triton_config_summary(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    kwargs = getattr(config, "kwargs", None)
    summary = {
        "kwargs": _autotune_json_value(kwargs) if isinstance(kwargs, dict) else None,
        "num_warps": _autotune_json_value(getattr(config, "num_warps", None)),
        "num_stages": _autotune_json_value(getattr(config, "num_stages", None)),
        "num_ctas": _autotune_json_value(getattr(config, "num_ctas", None)),
        "maxnreg": _autotune_json_value(getattr(config, "maxnreg", None)),
        "enable_fp_fusion": _autotune_json_value(getattr(config, "enable_fp_fusion", None)),
    }
    if not isinstance(kwargs, dict) and isinstance(config, (str, int, float, bool)):
        summary["value"] = config
    return summary


def _fla_norm_autotune_summary(module) -> dict[str, Any]:
    """Read, but never invoke, the upstream FLA layer-norm Triton autotuner."""
    module_name = type(module).__module__
    candidate_modules = [module_name, "fla.modules.fused_norm_gate"]
    imported = None
    last_import_error = None
    for candidate in dict.fromkeys(candidate_modules):
        try:
            imported = importlib.import_module(candidate)
        except Exception as exc:  # noqa: BLE001 - optional metadata must not alter forward.
            last_import_error = type(exc).__name__
            continue
        kernel = getattr(imported, "layer_norm_gated_fwd_kernel", None)
        if kernel is not None:
            break
        imported = None
    else:
        return {
            "kernel": "layer_norm_gated_fwd_kernel",
            "status": "unavailable",
            "reason": (f"upstream kernel import failed: {last_import_error}"
                       if last_import_error else "upstream kernel object was not found"),
            "implementation_module": module_name,
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        }

    kernel = getattr(imported, "layer_norm_gated_fwd_kernel", None)
    autotuner = getattr(kernel, "fn", kernel)
    cache = getattr(autotuner, "cache", None)
    best_config = getattr(autotuner, "best_config", None)
    if not isinstance(cache, dict):
        return {
            "kernel": "layer_norm_gated_fwd_kernel",
            "status": "unavailable",
            "reason": "autotuner cache is not a mapping",
            "implementation_module": getattr(imported, "__name__", module_name),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "last_best_config": _triton_config_summary(best_config),
        }

    cache_entries = [
        {"key": _autotune_json_value(key), "config": _triton_config_summary(config)}
        for key, config in sorted(
            cache.items(),
            key=lambda item: json.dumps(_autotune_json_value(item[0]), sort_keys=True),
        )
    ]
    selected_keys = [
        _autotune_json_value(key)
        for key, config in cache.items()
        if config is best_config
    ]
    return {
        "kernel": "layer_norm_gated_fwd_kernel",
        "status": "available",
        "implementation_module": getattr(imported, "__name__", module_name),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "key_names": _autotune_json_value(getattr(autotuner, "keys", None)),
        "cache_entry_count": len(cache_entries),
        "cache_entries": cache_entries,
        "last_selected_keys": selected_keys,
        "last_best_config": _triton_config_summary(best_config),
    }


def collect_t2_tp_execution_diagnostics(
    model,
    trainer,
    *,
    rank: int,
    world_size: int,
    offloader=None,
    residency_state: str | None = None,
) -> dict[str, Any]:
    """Hash frozen local weights/buffers and capture execution-mode metadata.

    The caller must serialize this with model commands and synchronize CUDA
    before calling. This function performs no model forward or state mutation.
    """
    if getattr(model, "_t2_tp_model", False) is not True:
        raise ValueError("execution diagnostics require a loaded T2 TP model")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if rank >= world_size:
        raise ValueError("rank must be smaller than world_size")

    buffers = _named_tensors(model, "named_buffers")
    frozen_parameters = _named_tensors(
        model, "named_parameters", predicate=lambda parameter: not bool(parameter.requires_grad)
    )
    buffer_records = [_tensor_record(name, tensor) for name, tensor in buffers]
    frozen_records = [_tensor_record(name, tensor) for name, tensor in frozen_parameters]
    base_model = _base_model(model)
    rope_runtime, rope_tensor_attributes = _rope_runtime(model, buffers)
    offloader_state = None
    if offloader is not None:
        offloader_state = {
            "phase": _safe_metadata_value(getattr(offloader, "_phase", None)),
            "parked": _safe_metadata_value(getattr(offloader, "_parked", None)),
            "closed": _safe_metadata_value(getattr(offloader, "_closed", None)),
        }

    return {
        "schema": EXECUTION_DIAGNOSTICS_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "residency_state": residency_state,
        "frozen_parameters": frozen_records,
        "buffers": buffer_records,
        "rope_runtime": rope_runtime,
        "rope_tensor_attributes": rope_tensor_attributes,
        "module_modes": _module_modes(model),
        "batch_profile": {
            "autocast": _autocast_metadata(trainer),
            "base_t2_head": _head_metadata(base_model),
        },
        "torch": _torch_flags(),
        "gdn_callables": _gdn_callables(model),
        "offloader": offloader_state,
    }


def combine_t2_tp_execution_diagnostics(
    rank_records: list[dict[str, Any]],
    *,
    parallelism: str,
    model_backend: str,
    world_size: int,
) -> dict[str, Any]:
    """Validate and return one explicit per-rank diagnostic payload."""
    if parallelism != "tp" or model_backend != "ternary":
        raise ValueError("execution diagnostics are available only for T2 TP")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if not isinstance(rank_records, list) or len(rank_records) != world_size:
        raise ValueError("execution diagnostics must contain exactly one record per rank")
    ranks: list[dict[str, Any] | None] = [None] * world_size
    for record in rank_records:
        if not isinstance(record, dict) or record.get("schema") != EXECUTION_DIAGNOSTICS_SCHEMA:
            raise ValueError("rank returned an invalid execution diagnostic schema")
        rank = record.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError("rank returned an invalid execution diagnostic identity")
        if ranks[rank] is not None:
            raise ValueError("duplicate rank execution diagnostic")
        if record.get("world_size") != world_size:
            raise ValueError("rank execution diagnostic world size differs")
        ranks[rank] = record
    if any(record is None for record in ranks):
        raise ValueError("execution diagnostics omit one or more ranks")
    return {
        "schema": EXECUTION_DIAGNOSTICS_SCHEMA,
        "parallelism": "tp",
        "model_backend": "ternary",
        "world_size": world_size,
        "ranks": [record for record in ranks if record is not None],
    }


__all__ = [
    "EXECUTION_DIAGNOSTICS_SCHEMA",
    "HASH_TILE_BYTES",
    "capture_t2_forward_trace",
    "collect_t2_tp_execution_diagnostics",
    "combine_t2_tp_execution_diagnostics",
    "iter_tensor_byte_chunks",
    "tensor_sha256",
]


@contextmanager
def capture_t2_forward_trace(model):
    """Capture the selected T2 forward trace and remove hooks on every exit.

    Each record keeps the first positional ``input`` summary and adds summaries
    for every tensor nested in positional and keyword arguments. Only the first
    layer's ``linear_attn.norm`` records its own parameters and named attributes.
    """
    import torch

    base = _base_model(model)
    records, handles = [], []

    def tensor_items(value, path):
        if isinstance(value, torch.Tensor):
            yield path, value
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                yield from tensor_items(value[key], f"{path}[{key!r}]")
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from tensor_items(item, f"{path}[{index}]")

    def summary(tensor):
        return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype), "sha256": tensor_sha256(tensor)}

    def attribute_summary(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if callable(value):
            target = getattr(value, "__func__", value)
            return {
                "module": getattr(target, "__module__", None),
                "qualname": getattr(target, "__qualname__", None),
            }
        if isinstance(value, (tuple, list)):
            return [attribute_summary(item) for item in value]
        return None

    def hook(name, module):
        def capture(_module, inputs, kwargs, output):
            positional_tensors = [
                {"path": path, **summary(tensor)}
                for path, tensor in tensor_items(inputs, "args")
            ]
            keyword_tensors = [
                {"path": path, **summary(tensor)}
                for path, tensor in tensor_items(kwargs, "kwargs")
            ]
            input_tensors = positional_tensors + keyword_tensors
            output_tensors = list(tensor_items(output, "output"))
            record = {
                "name": name,
                "module_type": type(module).__qualname__,
                "input": None if not positional_tensors else {
                    key: value for key, value in positional_tensors[0].items() if key != "path"
                },
                "inputs": input_tensors,
                "output": None if not output_tensors else summary(output_tensors[0][1]),
            }
            if name == "layers.0.linear_attn.norm":
                record["named_parameters"] = [
                    {
                        "name": parameter_name,
                        **summary(parameter),
                    }
                    for parameter_name, parameter in sorted(
                        module.named_parameters(recurse=False), key=lambda item: item[0]
                    )
                ]
                record["attributes"] = {
                    attribute: attribute_summary(getattr(module, attribute, None))
                    for attribute in ("eps", "activation", "norm_before_gate")
                }
                try:
                    record["upstream_fla_autotune"] = _fla_norm_autotune_summary(module)
                except Exception as exc:  # noqa: BLE001 - diagnostics must not alter forward.
                    record["upstream_fla_autotune"] = {
                        "kernel": "layer_norm_gated_fwd_kernel",
                        "status": "unavailable",
                        "reason": type(exc).__name__,
                        "implementation_module": type(module).__module__,
                        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
                    }
            records.append(record)
        return capture

    try:
        for name, module in base.model.named_modules():
            if (name in {"embed_tokens", "norm"}
                    or re.fullmatch(r"layers\.\d+", name) is not None
                    or re.fullmatch(r"layers\.0\.(linear_attn|mlp)\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj|norm|gate_proj|up_proj|down_proj)", name) is not None):
                handles.append(module.register_forward_hook(hook(name, module), with_kwargs=True))
        yield records
    finally:
        for handle in handles:
            handle.remove()
