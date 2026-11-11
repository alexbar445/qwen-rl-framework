"""Torch-free validation and content identity for native TorchAO NVFP4 artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

from .artifacts import source_model_fingerprint


SOURCE_FINGERPRINT_FILE = "source_fingerprint.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STRUCTURAL_FILES = {
    "config.json",
    "model.safetensors.index.json",
    SOURCE_FINGERPRINT_FILE,
}
_SAFE_TENSOR_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"NVFP4 artifact has invalid {field}")
    return value


def _read_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"NVFP4 artifact is missing {field}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"NVFP4 artifact has invalid {field}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"NVFP4 artifact {field} must contain an object")
    return value


def _source_manifest_fingerprint(manifest: dict[str, Any]) -> str:
    fields = {
        "config_sha256": manifest.get("config_sha256"),
        "index_sha256": manifest.get("index_sha256"),
        "weight_files": manifest.get("weight_files"),
        "tokenizer_files": manifest.get("tokenizer_files"),
    }
    _require_sha256(fields["config_sha256"], "source config SHA-256")
    if fields["index_sha256"] is not None:
        _require_sha256(fields["index_sha256"], "source index SHA-256")
    content_verification = manifest.get("content_verification")
    if content_verification not in (None, "disabled"):
        raise ValueError("source manifest has an unsupported content_verification value")
    for collection_name in ("weight_files", "tokenizer_files"):
        collection = fields[collection_name]
        if not isinstance(collection, list):
            raise ValueError(f"source manifest {collection_name} must be a list")
        declared_hashes: list[str | None] = []
        for record in collection:
            if not isinstance(record, dict):
                raise ValueError(f"source manifest {collection_name} has a malformed entry")
            name, size, digest = record.get("name"), record.get("size"), record.get("sha256")
            if (not isinstance(name, str) or not name or Path(name).is_absolute()
                    or isinstance(size, bool) or not isinstance(size, int) or size < 0):
                raise ValueError(f"source manifest {collection_name} has a malformed entry")
            if digest is None and collection_name == "weight_files" and content_verification == "disabled":
                declared_hashes.append(None)
                continue
            _require_sha256(digest, f"source {collection_name} SHA-256")
            declared_hashes.append(digest)
        if collection_name == "weight_files" and content_verification == "disabled":
            if not collection or any(digest is not None for digest in declared_hashes):
                raise ValueError("metadata-only source weight identity must omit every weight SHA-256")
        elif collection_name == "weight_files" and any(digest is None for digest in declared_hashes):
            raise ValueError("source weight SHA-256 may be omitted only for metadata-only identities")
    if not fields["weight_files"]:
        raise ValueError("NVFP4 source manifest contains no source weights")
    return _canonical_sha256(fields)


def _safe_relative_file(root: Path, relative: Any, field: str) -> tuple[str, Path]:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"NVFP4 {field} path must be relative")
    unresolved = root / relative
    resolved = unresolved.resolve()
    if (unresolved.is_symlink() or not resolved.is_file()
            or root not in resolved.parents):
        raise ValueError(f"NVFP4 {field} path is missing, symlinked, or escapes the artifact: {relative}")
    return relative, resolved


def _recorded_files(
    root: Path,
    records: Any,
    field: str,
    *,
    verify_payload_sha: bool = True,
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"NVFP4 artifact has no {field} manifest")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"NVFP4 {field} entries must be objects")
        relative, path = _safe_relative_file(root, record.get("path"), field)
        if relative in seen:
            raise ValueError(f"NVFP4 {field} declares duplicate path: {relative}")
        seen.add(relative)
        size, digest = record.get("size"), record.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"NVFP4 {field} has an invalid size: {relative}")
        if path.stat().st_size != size:
            raise ValueError(f"NVFP4 {field} size mismatch: {relative}")
        _require_sha256(digest, f"{field} SHA-256")
        if verify_payload_sha:
            actual_digest = _sha256(path)
            if actual_digest != digest:
                raise ValueError(f"NVFP4 {field} SHA-256 mismatch: {relative}")
            recorded_digest = actual_digest
        else:
            # Keep the sidecar's existing SHA as an opaque identity. It was not
            # recomputed from the weight payload during this load.
            recorded_digest = digest
        result.append({"path": relative, "size": size, "sha256": recorded_digest})
    return sorted(result, key=lambda item: item["path"])


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("text_config")
    return value if isinstance(value, dict) else config


def expected_nvfp4_skip_modules(config: dict[str, Any]) -> list[str]:
    """Return exact HF source FQNs and vLLM fused FQNs for the Qwen3.5 recipe."""
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        raise ValueError("NVFP4 Qwen config has no architectures list")
    if "Qwen3_5ForConditionalGeneration" in architectures:
        hf_layer_prefix = "model.language_model.layers"
    elif "Qwen3_5ForCausalLM" in architectures:
        hf_layer_prefix = "model.layers"
    else:
        raise ValueError("native NVFP4 export supports Qwen3.5 CausalLM architectures only")

    text = _text_config(config)
    layer_types = text.get("layer_types")
    layer_count = text.get("num_hidden_layers")
    if (not isinstance(layer_types, list) or not layer_types
            or isinstance(layer_count, bool) or not isinstance(layer_count, int)
            or len(layer_types) != layer_count):
        raise ValueError("Qwen3.5 text config has invalid layer_types/num_hidden_layers")
    if text.get("model_type") not in ("qwen3_5_text", "qwen3_5"):
        raise ValueError("native NVFP4 export requires a Qwen3.5 text config")

    result: list[str] = []
    for index, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention":
            continue
        result.extend((
            f"{hf_layer_prefix}.{index}.linear_attn.in_proj_a",
            f"{hf_layer_prefix}.{index}.linear_attn.in_proj_b",
            f"model.layers.{index}.linear_attn.in_proj_ba",
        ))
    if not result:
        raise ValueError("Qwen3.5 config has no linear-attention GDN layers")
    result.append("lm_head")
    return result


def _shape_numel(shape: Any, name: str) -> int:
    if (not isinstance(shape, list)
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape)):
        raise ValueError(f"safetensors tensor {name} has an invalid shape")
    return math.prod(shape)


def _safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors header: {path.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 0 or header_size > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header size in {path.name}")
        raw_header = stream.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"truncated safetensors metadata: {path.name}")
    try:
        parsed = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensors header JSON in {path.name}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"safetensors header is not an object: {path.name}")
    metadata = parsed.pop("__metadata__", {})
    if (not isinstance(metadata, dict)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in metadata.items())):
        raise ValueError(f"safetensors metadata is invalid: {path.name}")
    data_bytes = path.stat().st_size - 8 - header_size
    occupied: list[tuple[int, int]] = []
    for name, entry in parsed.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError(f"malformed safetensors tensor header in {path.name}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        numel = _shape_numel(shape, name)
        if (not isinstance(dtype, str) or dtype not in _SAFE_TENSOR_BYTES
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)):
            raise ValueError(f"malformed safetensors dtype/offsets for {name}")
        start, end = offsets
        if start < 0 or end < start or end > data_bytes:
            raise ValueError(f"safetensors offsets escape data section for {name}")
        if end - start != numel * _SAFE_TENSOR_BYTES[dtype]:
            raise ValueError(f"safetensors byte length disagrees with shape/dtype for {name}")
        occupied.append((start, end))
    occupied.sort()
    if any(left[1] > right[0] for left, right in zip(occupied, occupied[1:])):
        raise ValueError(f"overlapping safetensors tensor offsets in {path.name}")
    return parsed, metadata


def _tensor_inventory(root: Path, weight_files: list[dict[str, Any]]) -> tuple[
    dict[str, dict[str, Any]], dict[str, str], dict[str, str], dict[str, dict[str, str]]
]:
    tensors: dict[str, dict[str, Any]] = {}
    metadata: dict[str, str] = {}
    tensor_to_shard: dict[str, str] = {}
    metadata_by_shard: dict[str, dict[str, str]] = {}
    for record in weight_files:
        relative = record["path"]
        shard = root / relative
        if shard.suffix.lower() != ".safetensors":
            raise ValueError(f"native NVFP4 artifacts must use safetensors shards: {relative}")
        entries, shard_metadata = _safetensors_header(shard)
        for name, item in entries.items():
            if name in tensors:
                raise ValueError(f"duplicate safetensors tensor across shards: {name}")
            tensors[name] = item
            tensor_to_shard[name] = relative
        for name, value in shard_metadata.items():
            if name == "tensor_names":
                try:
                    logical_names = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid tensor_names metadata in {relative}") from exc
                if (not isinstance(logical_names, list)
                        or any(not isinstance(item, str) for item in logical_names)):
                    raise ValueError(f"invalid tensor_names metadata in {relative}")
                metadata[f"{relative}\u0000tensor_names"] = value
                metadata_by_shard.setdefault(relative, {})[name] = value
                continue
            if name in metadata and metadata[name] != value:
                raise ValueError(f"conflicting tensor metadata for {name}")
            metadata[name] = value
            metadata_by_shard.setdefault(relative, {})[name] = value
    return tensors, metadata, tensor_to_shard, metadata_by_shard


def _validate_tensor_names_metadata(
    metadata_by_shard: dict[str, dict[str, str]],
    tensors: dict[str, dict[str, Any]],
    tensor_to_shard: dict[str, str],
) -> None:
    component_suffixes = (
        "._weight_qdata", "._weight_scale", "._weight_per_tensor_scale",
    )
    for shard, shard_metadata in metadata_by_shard.items():
        value = shard_metadata.get("tensor_names")
        if value is None:
            continue
        names = json.loads(value)
        raw_names = {
            name for name, owner in tensor_to_shard.items()
            if owner == shard and not name.endswith(component_suffixes)
        }
        subclass_names = {
            name for name, value in shard_metadata.items()
            if name != "tensor_names"
            and isinstance(json.loads(value), dict)
            and json.loads(value).get("_type") != "Tensor"
        }
        if set(names) != raw_names | subclass_names:
            raise ValueError(f"TorchAO tensor_names metadata differs from safetensors keys in {shard}")


def _packed_inventory(
    tensors: dict[str, dict[str, Any]], metadata: dict[str, str], tensor_to_shard: dict[str, str]
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    qdata_suffix = "._weight_qdata"
    for qdata_key in sorted(name for name in tensors if name.endswith(qdata_suffix)):
        prefix = qdata_key[:-len(qdata_suffix)]
        logical_name = prefix + ".weight"
        qdata = tensors[qdata_key]
        if qdata.get("dtype") != "U8":
            raise ValueError(f"NVFP4 packed qdata must use uint8 storage: {qdata_key}")
        scale_key = prefix + "._weight_scale"
        per_tensor_scale_key = prefix + "._weight_per_tensor_scale"
        for component in (scale_key, per_tensor_scale_key):
            if component not in tensors:
                raise ValueError(f"NVFP4 packed weight is missing component {component}")
            if tensor_to_shard[component] != tensor_to_shard[qdata_key]:
                raise ValueError(f"NVFP4 weight components span shards: {logical_name}")
        tensor_metadata_raw = metadata.get(logical_name)
        if not isinstance(tensor_metadata_raw, str):
            raise ValueError(f"NVFP4 weight is missing TorchAO metadata: {logical_name}")
        try:
            tensor_metadata = json.loads(tensor_metadata_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid TorchAO metadata for {logical_name}") from exc
        data = tensor_metadata.get("_data") if isinstance(tensor_metadata, dict) else None
        orig_dtype = data.get("orig_dtype") if isinstance(data, dict) else None
        if (tensor_metadata.get("_type") != "NVFP4Tensor"
                or not isinstance(orig_dtype, dict)
                or orig_dtype.get("_type") != "torch.dtype"
                or orig_dtype.get("_data") != "bfloat16"
                or data.get("block_size") != 16
                or data.get("is_swizzled_scales") is not True):
            raise ValueError(f"NVFP4 weight metadata does not match BF16 swizzled block-16 recipe: {logical_name}")
        tensor_data_names = tensor_metadata.get("_tensor_data_names")
        if not isinstance(tensor_data_names, list) or not {
            "qdata", "scale", "per_tensor_scale"
        }.issubset(tensor_data_names):
            raise ValueError(f"NVFP4 tensor metadata is missing packed components: {logical_name}")
        inventory.append({
            "name": logical_name,
            "qdata_key": qdata_key,
            "shard_path": tensor_to_shard[qdata_key],
            "qdata_dtype": qdata["dtype"],
            "qdata_shape": qdata["shape"],
            "qdata_numel": _shape_numel(qdata["shape"], qdata_key),
            "scales_keys": [scale_key, per_tensor_scale_key],
            "orig_dtype": "bfloat16",
            "block_size": 16,
            "swizzled_scales": True,
        })
    if not inventory:
        raise ValueError("NVFP4 artifact contains no packed uint8 weight tensors")
    return inventory


def _validate_skip_tensors(
    config: dict[str, Any], skip_modules: list[str], tensors: dict[str, dict[str, Any]], metadata: dict[str, str]
) -> dict[str, Any]:
    text = _text_config(config)
    if "Qwen3_5ForConditionalGeneration" in config["architectures"]:
        hf_prefix = "model.language_model.layers"
    else:
        hf_prefix = "model.layers"
    raw_skips = [
        name for name in skip_modules
        if name.startswith(hf_prefix + ".") and name.endswith(("in_proj_a", "in_proj_b"))
    ]
    linear_count = sum(item == "linear_attention" for item in text["layer_types"])
    if len(raw_skips) != linear_count * 2:
        raise ValueError("NVFP4 skip metadata does not contain both raw GDN projections for every linear layer")
    raw_inventory = {}
    for name in raw_skips:
        key = f"{name}.weight"
        entry = tensors.get(key)
        if not isinstance(entry, dict) or entry.get("dtype") != "F32":
            raise ValueError(f"GDN skip tensor must be raw FP32: {key}")
        expected_shape = [text.get("linear_num_value_heads"), text.get("hidden_size")]
        if entry.get("shape") != expected_shape:
            raise ValueError(f"GDN skip tensor has an unexpected shape: {key}")
        if key not in metadata or json.loads(metadata[key]).get("_type") != "Tensor":
            raise ValueError(f"GDN skip tensor is missing raw tensor metadata: {key}")
        raw_inventory[key] = {"dtype": entry["dtype"], "shape": entry["shape"]}
    head = tensors.get("lm_head.weight")
    if not isinstance(head, dict) or head.get("dtype") != "BF16":
        raise ValueError("NVFP4 artifact output head must be a raw BF16 tensor")
    expected_head_shape = [text.get("vocab_size"), text.get("hidden_size")]
    if head.get("shape") != expected_head_shape:
        raise ValueError("NVFP4 artifact output head has an unexpected shape")
    if "lm_head.weight" not in metadata or json.loads(metadata["lm_head.weight"]).get("_type") != "Tensor":
        raise ValueError("NVFP4 artifact output head must have raw tensor metadata")
    if any(name.startswith("lm_head._weight_") for name in tensors):
        raise ValueError("NVFP4 artifact contains packed lm_head components")
    return {"gdn_fp32": raw_inventory, "lm_head_bf16": {"dtype": "BF16", "shape": head["shape"]}}


def read_nvfp4_artifact(root: str | Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Validate a complete TorchAO NVFP4 checkpoint without importing torch."""
    requested_root = Path(root).expanduser()
    if requested_root.is_symlink():
        raise ValueError("NVFP4 model path cannot be a symlink")
    artifact_root = requested_root.resolve()
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"NVFP4 model directory does not exist: {artifact_root}")

    sidecar_path = artifact_root / SOURCE_FINGERPRINT_FILE
    lineage = _read_json(sidecar_path, SOURCE_FINGERPRINT_FILE)
    base_fingerprint = _require_sha256(lineage.get("base_fingerprint"), "source base_fingerprint")
    if lineage.get("quantization_method") != "torchao":
        raise ValueError("NVFP4 source manifest must declare quantization_method=torchao")
    if lineage.get("quantizer") != "NVFP4WeightOnlyConfig":
        raise ValueError("NVFP4 source manifest has the wrong quantizer")
    source_manifest = lineage.get("source_manifest")
    if not isinstance(source_manifest, dict) or _source_manifest_fingerprint(source_manifest) != base_fingerprint:
        raise ValueError("NVFP4 source lineage does not match its content-derived base fingerprint")
    if lineage.get("source_weight_files") != source_manifest.get("weight_files"):
        raise ValueError("NVFP4 source_weight_files differ from the fixed source manifest")

    config_path = artifact_root / "config.json"
    config = _read_json(config_path, "config.json")
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "torchao":
        raise ValueError("NVFP4 config.json must declare quant_method=torchao")
    quant_type = quantization.get("quant_type")
    default_type = quant_type.get("default") if isinstance(quant_type, dict) else None
    if not isinstance(default_type, dict) or default_type.get("_type") != "NVFP4WeightOnlyConfig":
        raise ValueError("NVFP4 config.json must use TorchAO NVFP4WeightOnlyConfig")
    default_data = default_type.get("_data")
    if not isinstance(default_data, dict) or default_data.get("use_dynamic_per_tensor_scale") is not True:
        raise ValueError("NVFP4 config.json must declare TorchAO's dynamic per-tensor scale recipe")
    expected_skips = expected_nvfp4_skip_modules(config)
    actual_skips = quantization.get("modules_to_not_convert")
    if actual_skips != expected_skips:
        raise ValueError("NVFP4 config.json modules_to_not_convert do not match actual Qwen3.5 FQNs")
    if lineage.get("modules_to_not_convert") != actual_skips:
        raise ValueError("NVFP4 source manifest skip metadata differs from config.json")

    index_path = artifact_root / "model.safetensors.index.json"
    index = _read_json(index_path, "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("NVFP4 safetensors index has no weight_map")
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("NVFP4 safetensors index has a malformed tensor key")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("NVFP4 safetensors index has a malformed shard path")

    weight_files = _recorded_files(
        artifact_root,
        lineage.get("quantized_weight_files"),
        "quantized_weight_files",
        verify_payload_sha=False,
    )
    declared_paths = {record["path"] for record in weight_files}
    discovered_paths = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*.safetensors")
        if path.is_file() and not path.is_symlink()
    }
    if discovered_paths != declared_paths:
        raise ValueError("NVFP4 safetensors inventory differs from source_fingerprint.json")
    tensors, tensor_metadata, tensor_to_shard, metadata_by_shard = _tensor_inventory(artifact_root, weight_files)
    if set(tensors) != set(weight_map):
        raise ValueError("NVFP4 safetensors index does not cover every serialized tensor exactly")
    for tensor_name, shard_name in weight_map.items():
        _safe_relative_file(artifact_root, shard_name, "index shard")
        if tensor_to_shard.get(tensor_name) != shard_name:
            raise ValueError(f"NVFP4 safetensors index points to the wrong shard for {tensor_name}")
    _validate_tensor_names_metadata(metadata_by_shard, tensors, tensor_to_shard)

    packed = _packed_inventory(tensors, tensor_metadata, tensor_to_shard)
    skip_inventory = _validate_skip_tensors(config, actual_skips, tensors, tensor_metadata)
    packed_names = {item["name"] for item in packed}
    if any(f"{name}.weight" in packed_names for name in actual_skips):
        raise ValueError("an explicitly skipped module was serialized as NVFP4")

    mtp_manifest = lineage.get("mtp_preservation")
    if mtp_manifest is not None:
        if not isinstance(mtp_manifest, dict):
            raise ValueError("NVFP4 MTP preservation metadata must be an object")
        mtp_path = mtp_manifest.get("path")
        mtp_file, _ = _safe_relative_file(artifact_root, mtp_path, "MTP sidecar")
        if mtp_file not in declared_paths or Path(mtp_file).name != "model_mtp.safetensors":
            raise ValueError("NVFP4 MTP sidecar is not declared as the dedicated model_mtp.safetensors shard")
        mtp_tensors = mtp_manifest.get("tensors")
        mtp_count = mtp_manifest.get("tensor_count")
        if (isinstance(mtp_count, bool) or not isinstance(mtp_count, int) or mtp_count <= 0
                or not isinstance(mtp_tensors, list) or len(mtp_tensors) != mtp_count):
            raise ValueError("NVFP4 MTP tensor inventory is malformed")
        source_shards = mtp_manifest.get("source_shards")
        source_weight_names = {item["name"] for item in source_manifest["weight_files"]}
        if (not isinstance(source_shards, list) or not source_shards
                or any(not isinstance(name, str) or name not in source_weight_names for name in source_shards)):
            raise ValueError("NVFP4 MTP source shard lineage is invalid")
        seen_mtp: set[str] = set()
        for item in mtp_tensors:
            if not isinstance(item, dict):
                raise ValueError("NVFP4 MTP tensor entry is malformed")
            name, dtype, shape = item.get("name"), item.get("dtype"), item.get("shape")
            if (not isinstance(name, str) or not name.startswith("mtp.") or name in seen_mtp
                    or not isinstance(dtype, str) or not isinstance(shape, list)):
                raise ValueError("NVFP4 MTP tensor entry is malformed")
            seen_mtp.add(name)
            entry = tensors.get(name)
            if (entry is None or entry.get("dtype") != dtype or entry.get("shape") != shape
                    or tensor_to_shard.get(name) != mtp_file):
                raise ValueError(f"NVFP4 MTP tensor differs from its raw sidecar inventory: {name}")
            packed_prefix = name.removesuffix(".weight")
            if f"{packed_prefix}._weight_qdata" in tensors:
                raise ValueError(f"NVFP4 MTP tensor was packed: {name}")
        actual_mtp_names = {name for name in tensors if name.startswith("mtp.")}
        if seen_mtp != actual_mtp_names:
            raise ValueError("NVFP4 MTP sidecar inventory does not cover every serialized MTP tensor")
        if mtp_manifest.get("quantized") is not False or mtp_manifest.get("runtime_enabled") is not False:
            raise ValueError("NVFP4 MTP sidecar must remain disabled and unquantized")
        if any(item["name"].startswith("mtp.") for item in packed):
            raise ValueError("NVFP4 MTP sidecar contains packed tensors")
    elif any(name.startswith("mtp.") for name in tensors):
        raise ValueError("NVFP4 artifact contains MTP tensors without a preservation manifest")

    packed_module_count = len(packed)
    packed_uint8_numel = sum(item["qdata_numel"] for item in packed)
    if (lineage.get("packed_module_count") != packed_module_count
            or lineage.get("packed_uint8_tensor_count") != packed_module_count
            or lineage.get("packed_uint8_numel") != packed_uint8_numel):
        raise ValueError("NVFP4 packed tensor counts differ from source_fingerprint.json")

    auxiliary_files = _recorded_files(artifact_root, lineage.get("auxiliary_files"), "auxiliary_files") if lineage.get("auxiliary_files") else []
    structural_paths = _STRUCTURAL_FILES | {record["path"] for record in weight_files}
    actual_auxiliary_paths = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.relative_to(artifact_root).as_posix() not in structural_paths
    }
    if actual_auxiliary_paths != {record["path"] for record in auxiliary_files}:
        raise ValueError("NVFP4 auxiliary/tokenizer inventory differs from source_fingerprint.json")

    tokenizer_manifest = lineage.get("tokenizer_files")
    if not isinstance(tokenizer_manifest, list):
        raise ValueError("NVFP4 source manifest has no tokenizer_files list")
    tokenizer_files = _recorded_files(artifact_root, tokenizer_manifest, "tokenizer_files") if tokenizer_manifest else []
    if not tokenizer_files:
        raise ValueError("NVFP4 artifact contains no tokenizer assets")
    if not any(Path(record["path"]).name in {"tokenizer.json", "vocab.json", "tokenizer.model"}
               for record in tokenizer_files):
        raise ValueError("NVFP4 artifact has no loadable tokenizer vocabulary")

    for record in weight_files + auxiliary_files + tokenizer_files:
        if record["path"] in _STRUCTURAL_FILES:
            raise ValueError("NVFP4 file manifests may not include structural identity files")
    artifact_manifest = {
        "kind": "content-sha256-torchao-nvfp4-artifact",
        "source_fingerprint": base_fingerprint,
        "sidecar_sha256": _sha256(sidecar_path),
        "source_manifest_sha256": _canonical_sha256(source_manifest),
        "config_sha256": _sha256(config_path),
        "index_sha256": _sha256(index_path),
        "weight_files": weight_files,
        "auxiliary_files": auxiliary_files,
        "tokenizer_files": tokenizer_files,
        "modules_to_not_convert": actual_skips,
        "skip_tensor_inventory": skip_inventory,
        "packed_tensor_inventory": packed,
        "packed_module_count": packed_module_count,
        "packed_uint8_tensor_count": packed_module_count,
        "packed_uint8_numel": packed_uint8_numel,
        "mtp_preservation": mtp_manifest,
        "quantization": {
            "mode": "torchao",
            "bits": 4,
            "quantizer": "NVFP4WeightOnlyConfig",
            "compute_dtype": "torch.bfloat16",
            "packed_resident": True,
        },
    }
    artifact_fingerprint = _canonical_sha256(artifact_manifest)
    identity = {
        "source_fingerprint": lineage,
        "artifact_fingerprint": artifact_fingerprint,
        "artifact_manifest": artifact_manifest,
        "content_verification": "disabled",
        "weight_identity_basis": "sidecar_declared_sha256_opaque",
        "source_weight_identity_basis": (
            "metadata_names_and_sizes"
            if lineage["source_manifest"].get("content_verification") == "disabled"
            else "source_manifest_declared_sha256_opaque"
        ),
    }
    return config, base_fingerprint, identity
