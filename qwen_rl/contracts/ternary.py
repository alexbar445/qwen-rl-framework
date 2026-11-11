"""Strict, CPU-only identity and representation contracts for supported Qwen T2 artifacts.

This module records an existing NInfer-v3 container as the immutable source for the
T2 learner loader. It does not decode, repack, or materialize model tensors.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import sys
import struct
from typing import Any, Mapping

SCHEMA_VERSION = "qwen-rl-ternary-artifact/v1"
MODEL_NAME = "bonsai2-27b-arc-k2-mtp"
RECIPE = "bonsai2_27b_ternary"
CONVERTER = "ninfer-v3"
CODER390_MODEL_NAME = "coder390-t2-full-scales-step512"
CODER390_REPORT_KIND = "coder390_full_scales_step512_run_report"
ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_TEXT_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5_text",
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
}
EXPECTED_LAYER_TYPES = [
    "full_attention" if index % 4 == 3 else "linear_attention"
    for index in range(64)
]
EXPECTED_FORMAT_COUNTS = {
    "bf16": 360,
    "fp32": 96,
    "int32": 1,
    "q8_g32_fp16": 5,
    "t2_g128_fp16": 323,
}
EXPECTED_T2_OBJECTS = 323
EXPECTED_T2_LOGICAL_PARAMETERS = 515
T2_FORMAT = "t2_g128_fp16"
T2_LAYOUT = "row_split_k128_v1"
SUPPORTED_T2_LORA_PROFILES = frozenset({(16, 16), (32, 32)})
TERNARY_QG_LAYOUT_VERSION = "qwen3_5_per_head_q_block_then_gate_block/v2"

CODER390_FORMAT_COUNTS = {
    "bf16": 585,
    "fp32": 96,
    "q4_g64_fp16": 54,
    "q5_g64_fp16": 54,
    "q6_g64_fp16": 1,
    "q8_g32_fp16": 7,
    "resource": 6,
    T2_FORMAT: 322,
}
CODER390_T2_OBJECTS = 322
CODER390_T2_LOGICAL_PARAMETERS = 514
CODER390_COMPLETED_UPDATES = 512
CODER390_LAYER_TYPES = EXPECTED_LAYER_TYPES
CODER390_PRIMAL_PARAMETERS = frozenset({
    "text/layers/7/attention/query",
    "text/layers/7/attention/gate",
    "text/layers/7/attention/key",
    "text/layers/7/attention/value",
    "text/layers/15/attention/query",
    "text/layers/15/attention/gate",
    "text/layers/15/attention/key",
    "text/layers/15/attention/value",
    "text/layers/20/gdn/output",
    "text/layers/39/attention/output",
    "text/layers/43/attention/output",
    "text/layers/48/gdn/output",
    "text/layers/63/attention/output",
})


class TernaryContractError(ValueError):
    """An artifact or identity document violates a supported pinned T2 profile."""


def expected_t2_parameters() -> set[str]:
    """Logical T2 parameter names exposed by the existing Bonsai2 NInfer artifact."""
    names = {"text/token_embedding", "text/output_head", "proposal/head"}
    for layer, layer_type in enumerate(EXPECTED_LAYER_TYPES):
        prefix = f"text/layers/{layer}/"
        names.update(
            prefix + "mlp/" + module for module in ("gate", "up", "down")
        )
        if layer_type == "full_attention":
            names.update(
                prefix + "attention/" + module
                for module in ("query", "key", "gate", "value", "output")
            )
        else:
            names.update(
                prefix + "gdn/" + module
                for module in ("query", "key", "value", "z", "output")
            )
    return names


def expected_coder390_t2_parameters() -> set[str]:
    """Text-stage T2 inventory for the Coder390 full-KV artifact (no proposal head)."""
    return expected_t2_parameters() - {"proposal/head"}


def _expected_coder390_t2_shapes() -> dict[str, tuple[int, int]]:
    shapes = {
        "text/token_embedding": (248320, 5120),
        "text/output_head": (248320, 5120),
    }
    for layer, layer_type in enumerate(CODER390_LAYER_TYPES):
        prefix = f"text/layers/{layer}/"
        shapes.update({
            prefix + "mlp/gate": (17408, 5120),
            prefix + "mlp/up": (17408, 5120),
            prefix + "mlp/down": (5120, 17408),
        })
        if layer_type == "full_attention":
            shapes.update({
                prefix + "attention/query": (6144, 5120),
                prefix + "attention/gate": (6144, 5120),
                prefix + "attention/key": (1024, 5120),
                prefix + "attention/value": (1024, 5120),
                prefix + "attention/output": (5120, 6144),
            })
        else:
            shapes.update({
                prefix + "gdn/query": (2048, 5120),
                prefix + "gdn/key": (2048, 5120),
                prefix + "gdn/value": (6144, 5120),
                prefix + "gdn/z": (6144, 5120),
                prefix + "gdn/output": (5120, 6144),
            })
    return shapes


def _coder390_lora_input_basis_by_module(
    input_basis_by_parameter: Mapping[str, str],
) -> dict[str, str]:
    """Map the current default PEFT targets to their effective input bases."""
    result: dict[str, str] = {}
    for layer, layer_type in enumerate(CODER390_LAYER_TYPES):
        prefix = f"text/layers/{layer}/"
        module_prefix = f"model.layers.{layer}."
        for projection in ("gate", "up", "down"):
            basis = input_basis_by_parameter[prefix + "mlp/" + projection]
            result[module_prefix + "mlp." + projection + "_proj"] = (
                "prism_rotated" if basis == "hadamard1024_signed" else "primal"
            )
        if layer_type == "full_attention":
            query_basis = input_basis_by_parameter[prefix + "attention/query"]
            gate_basis = input_basis_by_parameter[prefix + "attention/gate"]
            _expect(query_basis == gate_basis,
                    f"Coder390 layer {layer} query/gate Uses must share an input basis")
            for suffix, logical in (
                ("q_proj", "query"), ("k_proj", "key"),
                ("v_proj", "value"), ("o_proj", "output"),
            ):
                basis = input_basis_by_parameter[prefix + "attention/" + logical]
                result[module_prefix + "self_attn." + suffix] = (
                    "prism_rotated" if basis == "hadamard1024_signed" else "primal"
                )
    return dict(sorted(result.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise TernaryContractError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TernaryContractError(f"{label} is missing or unreadable: {path}") from exc
    if not resolved.is_file():
        raise TernaryContractError(f"{label} must be a regular file: {path}")
    return resolved


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TernaryContractError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TernaryContractError(f"{label} must contain a JSON object")
    return value


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise TernaryContractError(message)


def _validate_t2_lora_profile(adapter: Any, *, label: str = "T2 LoRA") -> None:
    _expect(isinstance(adapter, dict), f"{label} adapter contract is missing")
    rank, alpha = adapter.get("rank"), adapter.get("alpha")
    _expect(type(rank) is int and type(alpha) is int and (rank, alpha) in SUPPORTED_T2_LORA_PROFILES,
            f"{label} profile must be rank/alpha 16/16 or 32/32")
    _expect(adapter.get("scale_alpha_over_rank") == 1.0,
            f"{label} alpha/rank scaling must be exactly 1.0")


def derive_rank32_lora_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a rank-32 policy identity from a validated rank-16 frozen T2 base.

    Only the descriptive Learner adapter rank and alpha change. Frozen artifact,
    source paths, model geometry, basis maps, and conversion report remain intact.
    """
    from copy import deepcopy

    validated = validate_artifact_identity(identity)
    model = validated.get("model")
    _expect(isinstance(model, dict) and model.get("id") == MODEL_NAME,
            "rank-32 identity derivation is supported only for the Bonsai2 T2 profile")
    learner = validated.get("learner")
    adapter = learner.get("adapter") if isinstance(learner, dict) else None
    _expect(isinstance(adapter, dict) and (adapter.get("rank"), adapter.get("alpha")) == (16, 16),
            "rank-32 identity derivation requires a validated rank-16 T2 identity")
    result = deepcopy(validated)
    result["learner"]["adapter"]["rank"] = 32
    result["learner"]["adapter"]["alpha"] = 32
    derived = validate_artifact_identity(result)
    _expect(derived["source"] == validated["source"],
            "rank-32 identity derivation changed frozen artifact or conversion report identity")
    _expect(derived["model"] == validated["model"],
            "rank-32 identity derivation changed frozen model identity")
    return derived


def _validate_conversion_report(report: Mapping[str, Any]) -> dict[str, Any]:
    _expect(report.get("name") == MODEL_NAME, "conversion report is not the Bonsai2 27B ARC-K2 artifact")
    provenance = report.get("provenance")
    _expect(isinstance(provenance, dict), "conversion report omitted provenance")
    _expect(provenance.get("converter") == CONVERTER, "conversion report converter is unsupported")
    _expect(provenance.get("recipe") == RECIPE, "conversion report recipe is unsupported")

    components = report.get("components")
    _expect(isinstance(components, dict), "conversion report omitted components")
    text_component = components.get("text")
    _expect(isinstance(text_component, dict), "conversion report omitted text component")
    text_config = text_component.get("config")
    _expect(isinstance(text_config, dict), "conversion report omitted text model config")
    for key, expected in EXPECTED_TEXT_CONFIG.items():
        _expect(text_config.get(key) == expected, f"text config {key} differs from Bonsai2 geometry")
    _expect(text_config.get("layer_types") == EXPECTED_LAYER_TYPES,
            "text layer_types is not the pinned 3:1 GDN/full-attention sequence")

    mtp_component = components.get("mtp")
    _expect(isinstance(mtp_component, dict), "conversion report omitted MTP component")
    mtp_config = mtp_component.get("config")
    _expect(isinstance(mtp_config, dict), "conversion report omitted MTP component config")
    _expect(mtp_config.get("architectures") == ["Qwen3_5MTP"],
            "conversion report MTP component architecture is unsupported")

    formats = report.get("formats")
    _expect(formats == EXPECTED_FORMAT_COUNTS,
            f"format inventory differs from the pinned Bonsai2 T2 artifact: {formats!r}")
    methods = report.get("methods")
    _expect(isinstance(methods, list), "conversion report methods must be a list")
    _expect(all(isinstance(entry, dict) and isinstance(entry.get("format"), str) for entry in methods),
            "conversion report has a malformed method entry")
    method_formats = dict(Counter(entry["format"] for entry in methods))
    _expect(method_formats == EXPECTED_FORMAT_COUNTS,
            "conversion report method inventory differs from its format totals")
    t2_methods = [entry for entry in methods if entry.get("format") == T2_FORMAT]
    _expect(len(t2_methods) == EXPECTED_T2_OBJECTS, "T2 object count differs from the pinned artifact")
    object_ids: set[str] = set()
    observed_parameters: set[str] = set()
    for entry in t2_methods:
        _expect(entry.get("layout") == T2_LAYOUT, "T2 parameter layout is unsupported")
        _expect(entry.get("method") == "import_encoded", "T2 parameter was not imported as encoded data")
        object_id = entry.get("object")
        _expect(isinstance(object_id, str) and object_id and object_id not in object_ids,
                "T2 method object ID is missing or duplicated")
        object_ids.add(object_id)
        shape = entry.get("shape")
        _expect(isinstance(shape, list) and len(shape) == 2
                and all(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in shape)
                and shape[1] % 128 == 0,
                "T2 method shape must be a matrix with K divisible by group size 128")
        names = entry.get("parameters")
        _expect(isinstance(names, list) and names, "T2 method entry has no logical parameter names")
        _expect(all(isinstance(name, str) and name for name in names),
                "T2 method entry contains an invalid logical parameter name")
        _expect(not (observed_parameters & set(names)), "T2 logical parameter is declared more than once")
        observed_parameters.update(names)
    expected_parameters = expected_t2_parameters()
    _expect(observed_parameters == expected_parameters,
            "T2 logical parameter inventory differs from the pinned Bonsai2 architecture")
    _expect(report.get("artifact_id") and ARTIFACT_ID_RE.fullmatch(str(report["artifact_id"])) is not None,
            "conversion report artifact_id must be a 32-character lowercase hex ID")
    _expect(report.get("payload_bytes") is None or (
        isinstance(report.get("payload_bytes"), int) and not isinstance(report.get("payload_bytes"), bool)
        and report["payload_bytes"] > 0
    ), "conversion report payload_bytes is invalid")

    sources = provenance.get("sources")
    _expect(isinstance(sources, dict), "conversion report provenance omitted sources")
    ternary_source = sources.get("ternary")
    mtp_source = sources.get("mtp")
    _expect(isinstance(ternary_source, dict) and isinstance(ternary_source.get("path"), str),
            "conversion report omitted the original PQ2_0 GGUF path")
    _expect(isinstance(mtp_source, dict) and isinstance(mtp_source.get("path"), str),
            "conversion report omitted the MTP source path")
    return {
        "text_config": text_config,
        "layer_types": list(text_config["layer_types"]),
        "mtp_config": mtp_config,
        "formats": dict(formats),
        "t2_object_count": len(t2_methods),
        "t2_logical_parameter_count": len(observed_parameters),
        "t2_logical_parameters": sorted(observed_parameters),
        "provenance": provenance,
        "artifact_id": report["artifact_id"],
        "payload_bytes": report.get("payload_bytes"),
        "source_paths": {
            "raw_gguf": ternary_source["path"],
            "mtp": mtp_source["path"],
            "base_assets": (sources.get("base") or {}).get("path"),
            "conversion_output": report.get("output"),
        },
        "report_counts": {
            key: report.get(key)
            for key in ("parameters", "uses", "objects", "seconds", "preparation_seconds")
        },
    }


def _read_ninfer_directory(path: Path) -> tuple[dict[str, Any], str, int]:
    """Read the bounded NInfer-v3 directory without touching its tensor payload."""
    with path.open("rb") as stream:
        header = stream.read(32)
        _expect(len(header) == 32, "NInfer-v3 header is incomplete")
        magic, directory_size, artifact_id = struct.unpack("<8sQ16s", header)
        _expect(magic == b"NINFER\x00\x03", "NInfer artifact is not version 3")
        file_size = path.stat().st_size
        _expect(0 < directory_size <= file_size - 32, "NInfer directory size is invalid")
        try:
            directory = json.loads(stream.read(directory_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TernaryContractError("NInfer directory is invalid JSON") from exc
    _expect(isinstance(directory, dict), "NInfer directory must contain a JSON object")
    files = directory.get("files")
    _expect(isinstance(files, list) and len(files) == 1 and isinstance(files[0], dict),
            "T2 learner requires one self-contained NInfer entry file")
    payload_bytes = files[0].get("payload_bytes")
    _expect(type(payload_bytes) is int and payload_bytes > 0, "NInfer payload size is invalid")
    payload_offset = ((32 + directory_size + 4095) // 4096) * 4096
    _expect(payload_offset + payload_bytes == file_size,
            "NInfer file size differs from its header and declared payload")
    return directory, artifact_id.hex(), payload_bytes


def _expected_coder390_text_config() -> dict[str, Any]:
    return {
        **EXPECTED_TEXT_CONFIG,
        "layer_types": list(CODER390_LAYER_TYPES),
        "tie_word_embeddings": False,
    }


def _coder390_header_inventory(
    directory: Mapping[str, Any],
    payload_bytes: int,
) -> dict[str, Any]:
    """Validate Coder390's exact text T2 inventory and per-Use basis from header metadata."""
    components = directory.get("components")
    _expect(isinstance(components, dict), "Coder390 NInfer directory has no component map")
    text_component = components.get("text")
    _expect(isinstance(text_component, dict) and isinstance(text_component.get("config"), dict),
            "Coder390 NInfer directory has no text config")
    text_config = text_component["config"]
    for key, expected in _expected_coder390_text_config().items():
        _expect(text_config.get(key) == expected,
                f"Coder390 text config {key} differs from the pinned full-KV architecture")

    metadata = directory.get("metadata")
    _expect(isinstance(metadata, dict)
            and metadata.get("name") == "Coder390 full T2 scales step512 (no MLA)",
            "NInfer metadata name is not the Coder390 full-scales step512 artifact")
    provenance = directory.get("provenance")
    _expect(isinstance(provenance, dict), "Coder390 NInfer provenance is missing")
    recovery = provenance.get("compression_recovery")
    _expect(isinstance(recovery, dict)
            and recovery.get("completed_updates") == CODER390_COMPLETED_UPDATES
            and recovery.get("trainable") == ["ternary_scales"]
            and recovery.get("mla_enabled") is False
            and recovery.get("packed_codes_requantized") is False,
            "Coder390 NInfer compression-recovery provenance differs from step512 full-scales")
    _expect(isinstance(components.get("mtp"), dict)
            and isinstance(components.get("vision"), dict),
            "Coder390 artifact must retain its separate non-text MTP and vision components")

    object_records = directory.get("objects")
    _expect(isinstance(object_records, list), "Coder390 NInfer object inventory is missing")
    objects: dict[str, dict[str, Any]] = {}
    format_counts: Counter[str] = Counter()
    for obj in object_records:
        _expect(isinstance(obj, dict), "Coder390 NInfer object record is malformed")
        object_id = obj.get("id")
        _expect(isinstance(object_id, str) and object_id and object_id not in objects,
                "Coder390 NInfer object IDs must be unique strings")
        objects[object_id] = obj
        format_counts[obj.get("format") if isinstance(obj.get("format"), str) else "resource"] += 1
    _expect(dict(format_counts) == CODER390_FORMAT_COUNTS,
            "Coder390 NInfer object format inventory differs from full-scales step512")
    t2_objects = {
        object_id: obj for object_id, obj in objects.items()
        if obj.get("kind") == "tensor" and obj.get("format") == T2_FORMAT
    }
    _expect(len(t2_objects) == CODER390_T2_OBJECTS,
            "Coder390 text T2 storage-object count differs from full-scales step512")

    bindings = directory.get("bindings")
    _expect(isinstance(bindings, dict), "Coder390 NInfer binding inventory is missing")
    expected_shapes = _expected_coder390_t2_shapes()
    expected_parameters = expected_coder390_t2_parameters()
    observed_parameters: set[str] = set()
    used_t2_objects: set[str] = set()
    parameter_widths: dict[str, int] = {}
    for name, binding in bindings.items():
        if not isinstance(name, str) or not name.startswith("text/") or not isinstance(binding, dict):
            continue
        raw_parts = binding.get("parts")
        if raw_parts is None:
            object_id = binding.get("object")
            if not isinstance(object_id, str) or object_id not in objects:
                continue
            obj = objects[object_id]
            shape = obj.get("shape")
            if not isinstance(shape, list) or len(shape) != 2:
                continue
            raw_parts = [{"object": object_id, "range": binding.get("range", [0, shape[0] * shape[1]])}]
        if not isinstance(raw_parts, list) or not raw_parts:
            continue
        part_objects = [
            objects.get(part.get("object")) if isinstance(part, dict) else None
            for part in raw_parts
        ]
        if not any(isinstance(obj, dict) and obj.get("format") == T2_FORMAT for obj in part_objects):
            continue
        _expect(name in expected_parameters,
                f"Coder390 text T2 binding {name!r} is outside the pinned learner inventory")
        _expect(all(isinstance(obj, dict) and obj.get("kind") == "tensor"
                    and obj.get("format") == T2_FORMAT and obj.get("layout") == T2_LAYOUT
                    for obj in part_objects),
                f"Coder390 {name} binding mixes T2 and non-T2 storage")
        total_rows = 0
        input_width: int | None = None
        for raw_part, obj in zip(raw_parts, part_objects):
            shape = obj.get("shape")
            _expect(isinstance(shape, list) and len(shape) == 2
                    and all(type(value) is int and value > 0 for value in shape),
                    f"Coder390 {name} T2 object shape must be positive rank 2")
            rows, width = shape
            _expect(width % 128 == 0, f"Coder390 {name} T2 input width must divide group size 128")
            code_row_bytes = (width // 128) * 32
            scale_row_bytes = (width // 128) * 2
            scale_offset = ((rows * code_row_bytes + 255) // 256) * 256
            expected_bytes = scale_offset + rows * scale_row_bytes
            _expect(obj.get("bytes") == expected_bytes,
                    f"Coder390 {name} T2 object byte geometry is invalid")
            offset = obj.get("offset")
            _expect(type(offset) is int and offset >= 0 and offset % 256 == 0
                    and offset + expected_bytes <= payload_bytes,
                    f"Coder390 {name} T2 object payload range is invalid")
            bounds = raw_part.get("range") if isinstance(raw_part, dict) else None
            _expect(isinstance(bounds, list) and len(bounds) == 2
                    and all(type(value) is int for value in bounds),
                    f"Coder390 {name} T2 binding range is malformed")
            begin, end = bounds
            _expect(0 <= begin < end <= rows * width and begin % width == 0 and end % width == 0,
                    f"Coder390 {name} T2 binding range must cover whole in-bounds rows")
            _expect(input_width is None or input_width == width,
                    f"Coder390 {name} T2 parts use different input widths")
            input_width = width
            total_rows += (end - begin) // width
            used_t2_objects.add(obj["id"])
        _expect((total_rows, input_width) == expected_shapes[name],
                f"Coder390 {name} T2 logical shape differs from the pinned architecture")
        observed_parameters.add(name)
        parameter_widths[name] = int(input_width)

    _expect(observed_parameters == expected_parameters,
            "Coder390 text T2 logical parameter inventory differs from the pinned architecture")
    _expect(used_t2_objects == set(t2_objects),
            "Coder390 text T2 inventory does not account for every T2 storage object")

    uses = directory.get("uses")
    _expect(isinstance(uses, list), "Coder390 NInfer Use inventory is missing")
    uses_by_parameter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for use in uses:
        if isinstance(use, dict) and isinstance(use.get("parameter"), str):
            uses_by_parameter[use["parameter"]].append(use)

    basis_by_parameter: dict[str, str] = {}
    for name in sorted(expected_parameters - {"text/token_embedding"}):
        parameter_uses = uses_by_parameter.get(name, [])
        _expect(bool(parameter_uses), f"Coder390 {name} has no NInfer Use")
        modes: set[str] = set()
        for use in parameter_uses:
            auxiliaries = use.get("auxiliaries", {})
            _expect(isinstance(auxiliaries, dict) and set(auxiliaries) <= {"hadamard_signs"},
                    f"Coder390 {name} has an unsupported input transform")
            if "hadamard_signs" not in auxiliaries:
                modes.add("primal")
                continue
            sign_ref = auxiliaries["hadamard_signs"]
            _expect(isinstance(sign_ref, dict) and isinstance(sign_ref.get("object"), str),
                    f"Coder390 {name} Hadamard sign reference is malformed")
            sign_obj = objects.get(sign_ref["object"])
            width = parameter_widths[name]
            _expect(isinstance(sign_obj, dict) and sign_obj.get("kind") == "tensor"
                    and sign_obj.get("format") == "bf16"
                    and sign_obj.get("layout") == "contiguous_le_v1"
                    and sign_obj.get("shape") == [width]
                    and sign_obj.get("bytes") == width * 2,
                    f"Coder390 {name} Hadamard sign tensor metadata is invalid")
            modes.add("hadamard1024_signed")
        _expect(len(modes) == 1, f"Coder390 {name} Uses disagree on input basis")
        basis_by_parameter[name] = modes.pop()
    basis_by_parameter["text/token_embedding"] = basis_by_parameter["text/output_head"]
    _expect({name for name, basis in basis_by_parameter.items() if basis == "primal"}
            == set(CODER390_PRIMAL_PARAMETERS),
            "Coder390 per-Use basis inventory differs from the pinned full-scales artifact")
    _expect(len(basis_by_parameter) == CODER390_T2_LOGICAL_PARAMETERS,
            "Coder390 per-Use basis inventory count is invalid")
    _expect(basis_by_parameter["text/token_embedding"] == "hadamard1024_signed"
            and basis_by_parameter["text/output_head"] == "hadamard1024_signed",
            "Coder390 vocabulary T2 pair must use the rotated input basis")

    expected_t2_objects = {
        object_id: obj for object_id, obj in t2_objects.items()
        if any(
            isinstance(part, dict) and part.get("object") == object_id
            for name, binding in bindings.items() if name in observed_parameters
            for part in (binding.get("parts") or [binding])
        )
    }
    return {
        "text_config": text_config,
        "t2_object_count": len(expected_t2_objects),
        "t2_logical_parameter_count": len(observed_parameters),
        "t2_logical_parameters": sorted(observed_parameters),
        "format_inventory": dict(sorted(format_counts.items())),
        "input_basis_counts": dict(Counter(basis_by_parameter.values())),
        "input_basis_by_parameter": dict(sorted(basis_by_parameter.items())),
        "input_basis_by_module": _coder390_lora_input_basis_by_module(basis_by_parameter),
        "artifact_provenance": dict(provenance),
    }


def _validate_coder390_run_report(
    report: Mapping[str, Any],
    artifact_id: str,
    file_size: int,
    artifact_path: Path,
) -> dict[str, Any]:
    _expect(report.get("status") == "ready_native_validated",
            "Coder390 run report is not a ready native-validated artifact")
    artifact = report.get("artifact")
    _expect(isinstance(artifact, dict)
            and artifact.get("status") == "composed_cpu_bytes_verified"
            and artifact.get("artifact_id") == artifact_id
            # The report records the build location; deployment identity binds
            # the current file by its header ID, byte count and declared digest.
            and isinstance(artifact.get("path"), str)
            and Path(artifact["path"]).is_absolute()
            and Path(artifact["path"]).name == artifact_path.name
            and artifact.get("bytes") == file_size
            and artifact.get("source_scale_tensors") == 402
            and artifact.get("logical_matrices") == CODER390_T2_LOGICAL_PARAMETERS
            and artifact.get("overlaid_storage_objects") == CODER390_T2_OBJECTS
            and artifact.get("payload_objects_verified") == sum(CODER390_FORMAT_COUNTS.values())
            and artifact.get("text_layers") == 64
            and artifact.get("mla_enabled") is False
            and artifact.get("packed_codes_preserved") is True
            and artifact.get("non_scale_payload_preserved") is True,
            "Coder390 run report artifact inventory differs from full-scales step512")
    _expect(isinstance(artifact.get("sha256"), str)
            and SHA256_RE.fullmatch(artifact["sha256"]) is not None,
            "Coder390 run report artifact SHA-256 is invalid")
    roundtrip = report.get("production_learner_roundtrip")
    _expect(isinstance(roundtrip, dict) and roundtrip.get("status") == "passed"
            and roundtrip.get("production_reader_scale_owners_exact") == 402
            and roundtrip.get("qgate_layout") == "original head_interleave"
            and roundtrip.get("gdn_qkv_layout") == "original concat"
            and roundtrip.get("gpu_used") is False,
            "Coder390 production Learner roundtrip evidence is incomplete")
    training = report.get("training")
    _expect(isinstance(training, dict)
            and training.get("completed_updates") == CODER390_COMPLETED_UPDATES
            and training.get("trainable") == ["ternary_scales"]
            and training.get("mla_enabled") is False
            and training.get("lora_enabled") is False,
            "Coder390 run report training receipt differs from full-scales step512")
    return {
        "status": report["status"],
        "artifact_id": artifact_id,
        "t2_object_count": artifact["overlaid_storage_objects"],
        "t2_logical_parameter_count": artifact["logical_matrices"],
        "source_scale_tensors": artifact["source_scale_tensors"],
        "completed_updates": training["completed_updates"],
        "trainable": list(training["trainable"]),
        "production_learner_roundtrip_status": roundtrip["status"],
        "format_inventory": dict(CODER390_FORMAT_COUNTS),
    }


def _build_coder390_identity(
    artifact_path: Path,
    report_path: Path,
    report: Mapping[str, Any],
    *,
    declared_artifact_sha256: str | None,
) -> dict[str, Any]:
    _expect(isinstance(declared_artifact_sha256, str)
            and SHA256_RE.fullmatch(declared_artifact_sha256) is not None,
            "deployment base SHA-256 declaration is invalid")
    directory, artifact_id, payload_bytes = _read_ninfer_directory(artifact_path)
    report_metadata = _validate_coder390_run_report(
        report, artifact_id, artifact_path.stat().st_size, artifact_path
    )
    _expect(report["artifact"]["sha256"] == declared_artifact_sha256,
            "deployment-declared Coder390 SHA differs from its run report")
    inventory = _coder390_header_inventory(directory, payload_bytes)
    _expect(inventory["t2_object_count"] == report_metadata["t2_object_count"]
            and inventory["t2_logical_parameter_count"] == report_metadata["t2_logical_parameter_count"],
            "Coder390 run report counts differ from the NInfer header inventory")
    artifact_provenance = inventory["artifact_provenance"]
    sources = artifact_provenance.get("sources")
    mtp_source_path = (sources.get("mtp") or {}).get("path") if isinstance(sources, dict) else None
    report_record = _file_record(report_path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "frozen_ninfer_t2_learner_base",
        "model": {
            "id": CODER390_MODEL_NAME,
            "architecture": "Qwen3_5ForCausalLM",
            "model_type": "qwen3_5_text",
            "geometry": {
                key: inventory["text_config"][key]
                for key in (
                    "num_hidden_layers", "hidden_size", "intermediate_size",
                    "num_attention_heads", "num_key_value_heads", "head_dim",
                    "vocab_size", "max_position_embeddings", "linear_conv_kernel_dim",
                    "linear_key_head_dim", "linear_num_key_heads",
                    "linear_num_value_heads", "linear_value_head_dim",
                )
            },
            "layer_types": list(inventory["text_config"]["layer_types"]),
            "layer_schedule": "three_linear_attention_then_one_full_attention",
        },
        "source": {
            "ninfer_artifact": {
                "path": str(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
                "sha256": declared_artifact_sha256,
                "artifact_id": artifact_id,
                "name": "Coder390 full T2 scales step512 (no MLA)",
                "profile": CODER390_MODEL_NAME,
                "payload_bytes_from_header": payload_bytes,
            },
            "artifact_report": {
                **report_record,
                "report_kind": CODER390_REPORT_KIND,
                "artifact_size_bytes": artifact_path.stat().st_size,
                "declared_artifact_sha256": declared_artifact_sha256,
                **report_metadata,
                "input_basis_counts": inventory["input_basis_counts"],
            },
            "header_provenance": artifact_provenance,
            "raw_gguf": {
                "path": None,
                "status": "not_recorded",
                "sha256": None,
                "size_bytes": None,
            },
        },
        "representation": {
            "format": T2_FORMAT,
            "layout": T2_LAYOUT,
            "bits_per_weight_including_scale": 2.125,
            "group_size_k": 128,
            "codes_per_byte": 4,
            "code_bits": 2,
            "code_field_order": "least_significant_field_first",
            "code_mapping": {"00": 0, "01": 1, "11": -1},
            "invalid_code": {"bits": "10", "value": -2},
            "scale": {
                "storage": "raw_ieee754_binary16_bits",
                "shape_per_parameter": "[out_features, in_features / 128]",
                "meaning": "multiplicative_group_scale",
                "zero_point": None,
            },
            "packed_code_shape_per_parameter": "[out_features, in_features / 4]",
            "format_counts": inventory["format_inventory"],
            "t2_object_count": inventory["t2_object_count"],
            "t2_logical_parameter_count": inventory["t2_logical_parameter_count"],
            "t2_logical_parameters": inventory["t2_logical_parameters"],
            "input_basis_counts": inventory["input_basis_counts"],
        },
        "rotation": {
            "input_basis": "per_use",
            "input_basis_by_parameter": inventory["input_basis_by_parameter"],
            "activation_transform": "Use-defined: H_1024(signs * x) when hadamard signs are present; identity for primal Uses",
            "weight_relation": "Use-defined: rotated weights consume H_1024(signs * x); primal weights consume x",
            "transform": "per_use_hadamard1024_or_identity",
            "block_size": 1024,
            "axis": "input_last_dimension",
            "signs": {
                "binding": "per_Use.auxiliaries.hadamard_signs.object",
                "storage_kind": "auxiliary_tensor",
                "format": "bf16",
                "layout": "contiguous_le_v1",
                "value_set": [-1, 1],
                "input_widths": [5120, 6144, 17408],
            },
            "residual_stream_basis": "primal",
            "token_embedding_inverse": "signs * H_1024(z)",
        },
        "learner": {
            "reader": "qwen_rl.models.ternary.TernaryArtifactStore",
            "reader_format": "NInfer-v3 .ninfer",
            "base_trainable": False,
            "trainable_parameters": "LoRA factors only",
            "adapter": {
                "format": "standard_peft_lora",
                "rank": 16,
                "alpha": 16,
                "scale_alpha_over_rank": 1.0,
                "dropout": 0.0,
                "bias": "none",
                "input_basis": "per_use",
                "input_basis_by_module": inventory["input_basis_by_module"],
                "input_rule": "LoRA A consumes each target projection's Use-defined input basis",
                "output_basis": "projection_output",
                "factor_shapes": {"A": "[rank, in_features]", "B": "[out_features, rank]"},
                "lora_on_mtp_proposal_head": False,
            },
        },
        "mtp_checkpoint": {
            "local_copy": None,
            "source_path": mtp_source_path,
            "source_status": "embedded_component",
            "role": "embedded_non_text_component_not_loaded_by_learner",
            "is_lora_adapter": False,
            "loader_separation": "learner_loads_text_t2_bindings_only",
        },
        "verification": {
            "artifact_content_sha256_verified": False,
            "base_content_check": "deployment_declared",
            "conversion_report_sha256_verified": False,
            "artifact_report_sha256_verified": True,
            "ninfer_payload_decoded": False,
            "raw_gguf_content_verified": False,
            "mtp_copy_hashed": False,
            "gpu_loaded_or_scored": False,
            "limitations": [
                "NInfer payload SHA-256 is deployment-declared and was not recomputed.",
                "The run report records training and runtime receipts; the identity builder reads only its metadata and the NInfer header directory.",
                "Embedded non-text components are not loaded by the T2 learner provider.",
            ],
        },
        "observed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return validate_artifact_identity(identity)


def _source_file_status(path_value: str | None) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value:
        return {"path": None, "status": "path_missing", "sha256": None, "size_bytes": None}
    path = Path(path_value).expanduser()
    if path.is_symlink():
        return {"path": str(path), "status": "symlink_unverified", "sha256": None, "size_bytes": None}
    if not path.exists():
        return {"path": str(path), "status": "missing_unverified", "sha256": None, "size_bytes": None}
    if not path.is_file():
        return {"path": str(path), "status": "not_regular_unverified", "sha256": None, "size_bytes": None}
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "status": "present_unverified",
        "sha256": None,
        "size_bytes": resolved.stat().st_size,
    }


def build_artifact_identity(
    ninfer_path: str | Path,
    conversion_report_path: str | Path,
    *,
    mtp_checkpoint_path: str | Path | None = None,
    declared_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Create an identity manifest for the frozen T2 reader without copying tensor payloads."""
    artifact_path = _regular_file(ninfer_path, "NInfer T2 artifact")
    report_path = _regular_file(conversion_report_path, "NInfer conversion report")
    report = _read_json(report_path, "NInfer conversion report")
    if report.get("status") == "ready_native_validated":
        return _build_coder390_identity(
            artifact_path,
            report_path,
            report,
            declared_artifact_sha256=declared_artifact_sha256,
        )
    metadata = _validate_conversion_report(report)
    if declared_artifact_sha256 is not None:
        _expect(isinstance(declared_artifact_sha256, str) and
                SHA256_RE.fullmatch(declared_artifact_sha256) is not None,
                "deployment base SHA-256 declaration is invalid")
        artifact_record = {"path": str(artifact_path), "size_bytes": artifact_path.stat().st_size,
                           "sha256": declared_artifact_sha256}
        _check_pinned_header(artifact_path, metadata["artifact_id"])
    else:
        raise TernaryContractError(
            "a deployment-declared base SHA-256 is required; model payload hashing is disabled"
        )
    report_record = _file_record(report_path)
    payload_bytes = metadata["payload_bytes"]
    if payload_bytes is not None and payload_bytes > artifact_record["size_bytes"]:
        raise TernaryContractError("conversion report payload_bytes exceeds the .ninfer file size")
    _expect(artifact_path.suffix == ".ninfer", "NInfer T2 artifact path must end in .ninfer")

    mtp_copy: dict[str, Any] | None = None
    if mtp_checkpoint_path is not None:
        mtp_path = _regular_file(mtp_checkpoint_path, "MTP checkpoint")
        _expect(mtp_path.suffix == ".safetensors", "MTP checkpoint must be a .safetensors file")
        mtp_copy = {
            "path": str(mtp_path),
            "size_bytes": mtp_path.stat().st_size,
            "sha256": None,
        }

    raw_gguf = _source_file_status(metadata["source_paths"]["raw_gguf"])
    source_mtp = _source_file_status(metadata["source_paths"]["mtp"])
    base_assets = _source_file_status(metadata["source_paths"]["base_assets"])
    _expect(metadata["artifact_id"] == report.get("artifact_id"), "artifact ID changed while reading report")

    layer_types = metadata["layer_types"]
    logical_parameters = metadata["t2_logical_parameters"]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "frozen_ninfer_t2_learner_base",
        "model": {
            "id": MODEL_NAME,
            "architecture": "Qwen3_5ForCausalLM",
            "model_type": "qwen3_5_text",
            "geometry": {
                key: metadata["text_config"][key]
                for key in (
                    "num_hidden_layers", "hidden_size", "intermediate_size",
                    "num_attention_heads", "num_key_value_heads", "head_dim",
                    "vocab_size", "max_position_embeddings", "linear_conv_kernel_dim",
                    "linear_key_head_dim", "linear_num_key_heads",
                    "linear_num_value_heads", "linear_value_head_dim",
                )
            },
            "layer_types": layer_types,
            "layer_schedule": "three_linear_attention_then_one_full_attention",
        },
        "source": {
            "ninfer_artifact": {
                **artifact_record,
                "artifact_id": metadata["artifact_id"],
                "name": MODEL_NAME,
                "recipe": RECIPE,
                "converter": CONVERTER,
                "payload_bytes_from_report": payload_bytes,
            },
            "conversion_report": {
                **report_record,
                "format_inventory": metadata["formats"],
                "t2_object_count": metadata["t2_object_count"],
                "t2_logical_parameter_count": metadata["t2_logical_parameter_count"],
                "report_counts": metadata["report_counts"],
                "recorded_output_path": metadata["source_paths"]["conversion_output"],
            },
            "raw_gguf": {
                **raw_gguf,
                "format": "Prism PQ2_0 GGUF",
                "lineage_scope": "recorded_path_only_when_content_hash_is_unavailable",
            },
            "mtp_source": {
                **source_mtp,
                "role": "mtp_checkpoint_source",
                "is_lora_adapter": False,
            },
            "base_assets_source": base_assets,
            "lineage_status": (
                "deployment_declared_base" if declared_artifact_sha256 is not None else (
                    "converted_ninfer_content_hashed_raw_gguf_missing"
                    if raw_gguf["status"] == "missing_unverified"
                    else "converted_ninfer_content_hashed_raw_gguf_status_recorded")
            ),
        },
        "representation": {
            "format": T2_FORMAT,
            "layout": T2_LAYOUT,
            "bits_per_weight_including_scale": 2.125,
            "group_size_k": 128,
            "codes_per_byte": 4,
            "code_bits": 2,
            "code_field_order": "least_significant_field_first",
            "code_mapping": {"00": 0, "01": 1, "11": -1},
            "invalid_code": {"bits": "10", "value": -2},
            "scale": {
                "storage": "raw_ieee754_binary16_bits",
                "shape_per_parameter": "[out_features, in_features / 128]",
                "meaning": "multiplicative_group_scale",
                "zero_point": None,
            },
            "packed_code_shape_per_parameter": "[out_features, in_features / 4]",
            "format_counts": metadata["formats"],
            "t2_object_count": metadata["t2_object_count"],
            "t2_logical_parameter_count": metadata["t2_logical_parameter_count"],
            "t2_logical_parameters": logical_parameters,
        },
        "rotation": {
            "input_basis": "prism_rotated",
            "activation_transform": "R(x)=H_1024(signs * x)",
            "weight_relation": "W_rotated = W_primal * diag(signs) * H_1024",
            "transform": "normalized_sylvester_walsh_hadamard",
            "block_size": 1024,
            "axis": "input_last_dimension",
            "signs": {
                "binding": "per_Use.auxiliaries.hadamard_signs.object",
                "storage_kind": "auxiliary_tensor",
                "format": "bf16",
                "layout": "contiguous_le_v1",
                "value_set": [-1, 1],
                "input_widths": [5120, 6144, 17408],
                "shape_rule": "[input_width]",
            },
            "residual_stream_basis": "primal",
            "token_embedding_inverse": "signs * H_1024(z)",
        },
        "learner": {
            "reader": "qwen_rl.models.ternary.TernaryArtifactStore",
            "reader_format": "NInfer-v3 .ninfer",
            "base_trainable": False,
            "trainable_parameters": "LoRA factors only",
            "adapter": {
                "format": "standard_peft_lora",
                "rank": 16,
                "alpha": 16,
                "scale_alpha_over_rank": 1.0,
                "dropout": 0.0,
                "bias": "none",
                "input_basis": "prism_rotated",
                "input_rule": "LoRA A consumes the same transformed input as its frozen T2 projection",
                "output_basis": "projection_output",
                "factor_shapes": {"A": "[rank, in_features]", "B": "[out_features, rank]"},
                "lora_on_mtp_proposal_head": False,
            },
        },
        "mtp_checkpoint": {
            "local_copy": mtp_copy,
            "source_path": metadata["source_paths"]["mtp"],
            "source_status": source_mtp["status"],
            "role": "MTP_checkpoint_only",
            "is_lora_adapter": False,
            "loader_separation": "model_artifact_component_not_policy_adapter",
        },
        "verification": {
            "artifact_content_sha256_verified": False,
            "base_content_check": "deployment_declared",
            "conversion_report_sha256_verified": True,
            "ninfer_payload_decoded": False,
            "raw_gguf_content_verified": raw_gguf["status"] == "content_hashed",
            "mtp_copy_hashed": mtp_copy is not None,
            "gpu_loaded_or_scored": False,
            "limitations": [
                "NInfer payload content is not verified at runtime; its SHA is deployment-declared.",
                "The original PQ2_0 GGUF is not content-verified when its source path is unavailable.",
                "MTP checkpoint identity does not attest LoRA training or adapter origin.",
            ],
        },
        "observed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return validate_artifact_identity(identity)


def _validate_coder390_identity(
    identity: Mapping[str, Any],
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    model = identity.get("model")
    _expect(isinstance(model, dict) and model.get("id") == CODER390_MODEL_NAME,
            "Coder390 artifact model ID is unsupported")
    _expect(model.get("architecture") == "Qwen3_5ForCausalLM"
            and model.get("model_type") == "qwen3_5_text",
            "Coder390 text architecture is unsupported")
    _expect(model.get("layer_types") == CODER390_LAYER_TYPES
            and model.get("layer_schedule") == "three_linear_attention_then_one_full_attention",
            "Coder390 layer schedule is unsupported")
    geometry = model.get("geometry")
    _expect(isinstance(geometry, dict), "Coder390 model geometry is missing")
    for key, expected in _expected_coder390_text_config().items():
        if key in ("architectures", "model_type", "layer_types", "tie_word_embeddings"):
            continue
        _expect(geometry.get(key) == expected, f"Coder390 geometry {key} is invalid")

    source = identity.get("source")
    _expect(isinstance(source, dict), "Coder390 source identity is missing")
    ninfer = source.get("ninfer_artifact")
    report = source.get("artifact_report")
    _expect(isinstance(ninfer, dict) and isinstance(report, dict),
            "Coder390 NInfer artifact or run-report identity is missing")
    _expect(isinstance(ninfer.get("path"), str) and Path(ninfer["path"]).is_absolute()
            and Path(ninfer["path"]).suffix == ".ninfer",
            "Coder390 NInfer artifact path is invalid")
    _expect(type(ninfer.get("size_bytes")) is int and ninfer["size_bytes"] > 0,
            "Coder390 NInfer artifact size is invalid")
    _expect(isinstance(ninfer.get("sha256"), str)
            and SHA256_RE.fullmatch(ninfer["sha256"]) is not None,
            "Coder390 deployment-declared NInfer SHA-256 is invalid")
    _expect(isinstance(ninfer.get("artifact_id"), str)
            and ARTIFACT_ID_RE.fullmatch(ninfer["artifact_id"]) is not None,
            "Coder390 NInfer artifact ID is invalid")
    _expect(ninfer.get("profile") == CODER390_MODEL_NAME
            and ninfer.get("name") == "Coder390 full T2 scales step512 (no MLA)",
            "Coder390 NInfer artifact profile is invalid")
    _expect(type(ninfer.get("payload_bytes_from_header")) is int
            and ninfer["payload_bytes_from_header"] > 0,
            "Coder390 NInfer payload size is invalid")

    _expect(isinstance(report.get("path"), str) and Path(report["path"]).is_absolute()
            and Path(report["path"]).name == "run-report.json",
            "Coder390 run-report path is invalid")
    _expect(type(report.get("size_bytes")) is int and report["size_bytes"] > 0
            and isinstance(report.get("sha256"), str)
            and SHA256_RE.fullmatch(report["sha256"]) is not None,
            "Coder390 run-report file identity is invalid")
    _expect(report.get("report_kind") == CODER390_REPORT_KIND
            and report.get("status") == "ready_native_validated"
            and report.get("artifact_id") == ninfer["artifact_id"]
            and report.get("artifact_size_bytes") == ninfer["size_bytes"]
            and report.get("declared_artifact_sha256") == ninfer["sha256"],
            "Coder390 run-report identity differs from the NInfer artifact")
    _expect(report.get("t2_object_count") == CODER390_T2_OBJECTS
            and report.get("t2_logical_parameter_count") == CODER390_T2_LOGICAL_PARAMETERS
            and report.get("source_scale_tensors") == 402
            and report.get("completed_updates") == CODER390_COMPLETED_UPDATES
            and report.get("trainable") == ["ternary_scales"]
            and report.get("production_learner_roundtrip_status") == "passed"
            and report.get("format_inventory") == CODER390_FORMAT_COUNTS,
            "Coder390 run-report training or inventory receipt is invalid")
    expected_format_counts = dict(sorted(CODER390_FORMAT_COUNTS.items()))
    _expect(report.get("format_inventory") == expected_format_counts,
            "Coder390 run-report format inventory is invalid")

    raw_gguf = source.get("raw_gguf")
    _expect(isinstance(raw_gguf, dict) and raw_gguf.get("status") == "not_recorded"
            and raw_gguf.get("path") is None and raw_gguf.get("sha256") is None,
            "Coder390 raw GGUF lineage must remain explicitly unrecorded")
    _expect(isinstance(source.get("header_provenance"), dict),
            "Coder390 NInfer header provenance is missing")

    representation = identity.get("representation")
    _expect(isinstance(representation, dict)
            and representation.get("format") == T2_FORMAT
            and representation.get("layout") == T2_LAYOUT
            and representation.get("group_size_k") == 128
            and representation.get("code_bits") == 2
            and representation.get("code_mapping") == {"00": 0, "01": 1, "11": -1}
            and representation.get("invalid_code") == {"bits": "10", "value": -2},
            "Coder390 T2 representation contract is invalid")
    scale = representation.get("scale")
    _expect(isinstance(scale, dict) and scale.get("storage") == "raw_ieee754_binary16_bits"
            and scale.get("zero_point") is None,
            "Coder390 T2 scale contract is invalid")
    expected_parameters = expected_coder390_t2_parameters()
    _expect(representation.get("t2_object_count") == CODER390_T2_OBJECTS
            and representation.get("t2_logical_parameter_count") == CODER390_T2_LOGICAL_PARAMETERS
            and isinstance(representation.get("t2_logical_parameters"), list)
            and set(representation["t2_logical_parameters"]) == expected_parameters,
            "Coder390 text T2 logical parameter inventory is invalid")
    _expect(representation.get("format_counts") == expected_format_counts
            and representation.get("input_basis_counts") == {
                "hadamard1024_signed": CODER390_T2_LOGICAL_PARAMETERS - len(CODER390_PRIMAL_PARAMETERS),
                "primal": len(CODER390_PRIMAL_PARAMETERS),
            }, "Coder390 T2 format or Use-basis counts are invalid")

    expected_parameter_basis = {
        name: "primal" if name in CODER390_PRIMAL_PARAMETERS else "hadamard1024_signed"
        for name in sorted(expected_parameters)
    }
    rotation = identity.get("rotation")
    _expect(isinstance(rotation, dict) and rotation.get("input_basis") == "per_use"
            and rotation.get("input_basis_by_parameter") == expected_parameter_basis
            and rotation.get("transform") == "per_use_hadamard1024_or_identity",
            "Coder390 per-Use rotation map is invalid")
    _expect(rotation.get("activation_transform") ==
            "Use-defined: H_1024(signs * x) when hadamard signs are present; identity for primal Uses",
            "Coder390 activation transform contract is invalid")
    _expect(rotation.get("weight_relation") ==
            "Use-defined: rotated weights consume H_1024(signs * x); primal weights consume x",
            "Coder390 weight/input-basis relation is invalid")
    signs = rotation.get("signs")
    _expect(isinstance(signs, dict)
            and signs.get("binding") == "per_Use.auxiliaries.hadamard_signs.object"
            and signs.get("storage_kind") == "auxiliary_tensor"
            and signs.get("format") == "bf16"
            and signs.get("layout") == "contiguous_le_v1"
            and signs.get("value_set") == [-1, 1]
            and signs.get("input_widths") == [5120, 6144, 17408],
            "Coder390 Hadamard sign metadata is invalid")

    learner = identity.get("learner")
    _expect(isinstance(learner, dict) and learner.get("base_trainable") is False,
            "Coder390 T2 base must remain frozen")
    adapter = learner.get("adapter")
    _expect(isinstance(adapter, dict)
            and (adapter.get("rank"), adapter.get("alpha")) == (16, 16)
            and adapter.get("scale_alpha_over_rank") == 1.0,
            "Coder390 T2 LoRA remains rank/alpha 16/16 with identity scaling")
    _expect(isinstance(adapter, dict)
            and adapter.get("format") == "standard_peft_lora"
            and adapter.get("dropout") == 0.0 and adapter.get("bias") == "none"
            and adapter.get("input_basis") == "per_use"
            and adapter.get("input_rule") ==
            "LoRA A consumes each target projection's Use-defined input basis"
            and adapter.get("output_basis") == "projection_output"
            and adapter.get("lora_on_mtp_proposal_head") is False,
            "Coder390 per-Use LoRA profile is invalid")
    expected_module_basis = _coder390_lora_input_basis_by_module(expected_parameter_basis)
    _expect(adapter.get("input_basis_by_module") == expected_module_basis,
            "Coder390 PEFT target input-basis map is invalid")

    mtp = identity.get("mtp_checkpoint")
    _expect(isinstance(mtp, dict) and mtp.get("local_copy") is None
            and mtp.get("source_status") == "embedded_component"
            and mtp.get("role") == "embedded_non_text_component_not_loaded_by_learner"
            and mtp.get("is_lora_adapter") is False
            and mtp.get("loader_separation") == "learner_loads_text_t2_bindings_only",
            "Coder390 embedded MTP component metadata is invalid")
    _expect(mtp.get("source_path") ==
            (source["header_provenance"].get("sources", {}).get("mtp") or {}).get("path"),
            "Coder390 embedded MTP source path differs from NInfer provenance")

    verification = identity.get("verification")
    _expect(isinstance(verification, dict)
            and verification.get("artifact_content_sha256_verified") is False
            and verification.get("base_content_check") == "deployment_declared"
            and verification.get("conversion_report_sha256_verified") is False
            and verification.get("artifact_report_sha256_verified") is True
            and verification.get("ninfer_payload_decoded") is False,
            "Coder390 artifact verification claims are invalid")
    if verify_files:
        artifact_path = _regular_file(ninfer["path"], "Coder390 NInfer artifact")
        _expect(artifact_path.stat().st_size == ninfer["size_bytes"],
                "Coder390 NInfer artifact size changed")
        report_path = _regular_file(report["path"], "Coder390 run report")
        _expect(report_path.stat().st_size == report["size_bytes"],
                "Coder390 run-report size changed")
        _expect(_sha256(report_path) == report["sha256"],
                "Coder390 run-report metadata SHA-256 changed")
        run_report = _read_json(report_path, "Coder390 run report")
        _expect(run_report.get("artifact", {}).get("sha256") == ninfer["sha256"],
                "Coder390 run-report declared artifact SHA changed")
        _check_pinned_header(artifact_path, ninfer["artifact_id"])
        directory, actual_id, payload_bytes = _read_ninfer_directory(artifact_path)
        _expect(actual_id == ninfer["artifact_id"]
                and payload_bytes == ninfer["payload_bytes_from_header"],
                "Coder390 NInfer header identity or payload size changed")
        actual_inventory = _coder390_header_inventory(directory, payload_bytes)
        _expect(actual_inventory["input_basis_by_parameter"] == expected_parameter_basis
                and actual_inventory["input_basis_by_module"] == expected_module_basis
                and actual_inventory["t2_logical_parameters"] == representation["t2_logical_parameters"]
                and actual_inventory["format_inventory"] == representation["format_counts"]
                and actual_inventory["artifact_provenance"] == source["header_provenance"],
                "Coder390 NInfer header metadata differs from its identity manifest")
        _validate_coder390_run_report(
            run_report, actual_id, artifact_path.stat().st_size, artifact_path
        )
    return dict(identity)


def validate_artifact_identity(identity: Mapping[str, Any], *, verify_files: bool = False) -> dict[str, Any]:
    """Validate the stable identity contract and optionally rehash all present inputs."""
    _expect(isinstance(identity, Mapping), "artifact identity must be an object")
    _expect(identity.get("schema_version") == SCHEMA_VERSION, "artifact identity schema is unsupported")
    _expect(identity.get("artifact_kind") == "frozen_ninfer_t2_learner_base",
            "artifact identity kind is unsupported")
    model = identity.get("model")
    if isinstance(model, dict) and model.get("id") == CODER390_MODEL_NAME:
        return _validate_coder390_identity(identity, verify_files=verify_files)
    _expect(isinstance(model, dict) and model.get("id") == MODEL_NAME,
            "artifact identity model ID is unsupported")
    _expect(model.get("architecture") == "Qwen3_5ForCausalLM", "artifact architecture is unsupported")
    _expect(model.get("layer_types") == EXPECTED_LAYER_TYPES, "artifact layer schedule is unsupported")
    geometry = model.get("geometry")
    _expect(isinstance(geometry, dict), "artifact geometry is missing")
    for key, expected in EXPECTED_TEXT_CONFIG.items():
        if key in ("architectures", "model_type"):
            continue
        _expect(geometry.get(key) == expected, f"artifact geometry {key} is invalid")

    source = identity.get("source")
    _expect(isinstance(source, dict), "artifact source identity is missing")
    ninfer = source.get("ninfer_artifact")
    report = source.get("conversion_report")
    _expect(isinstance(ninfer, dict) and isinstance(report, dict), "artifact or report identity is missing")
    for item, label in ((ninfer, "NInfer artifact"), (report, "conversion report")):
        _expect(isinstance(item.get("path"), str) and Path(item["path"]).is_absolute(),
                f"{label} path must be absolute")
        _expect(isinstance(item.get("size_bytes"), int) and item["size_bytes"] > 0,
                f"{label} size is invalid")
        digest = item.get("sha256")
        if label == "conversion report":
            _expect(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
                    f"{label} SHA-256 is invalid")
        else:
            _expect(digest is None or (
                isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None
            ), f"{label} declared SHA-256 metadata is invalid")
    _expect(Path(ninfer["path"]).suffix == ".ninfer", "NInfer artifact path must end in .ninfer")
    _expect(ninfer.get("artifact_id") and ARTIFACT_ID_RE.fullmatch(str(ninfer["artifact_id"])) is not None,
            "NInfer artifact ID is invalid")
    _expect(ninfer.get("recipe") == RECIPE and ninfer.get("converter") == CONVERTER,
            "NInfer conversion recipe or converter is invalid")
    _expect(report.get("format_inventory") == EXPECTED_FORMAT_COUNTS,
            "conversion report format inventory is unsupported")
    _expect(report.get("t2_object_count") == EXPECTED_T2_OBJECTS,
            "conversion report T2 object count is unsupported")
    _expect(report.get("t2_logical_parameter_count") == EXPECTED_T2_LOGICAL_PARAMETERS,
            "conversion report T2 logical parameter count is unsupported")

    representation = identity.get("representation")
    _expect(isinstance(representation, dict), "T2 representation contract is missing")
    _expect(representation.get("format") == T2_FORMAT, "T2 format is unsupported")
    _expect(representation.get("layout") == T2_LAYOUT, "T2 layout is unsupported")
    _expect(representation.get("group_size_k") == 128 and representation.get("code_bits") == 2,
            "T2 packing geometry is unsupported")
    _expect(representation.get("code_mapping") == {"00": 0, "01": 1, "11": -1},
            "T2 code mapping is invalid")
    _expect(representation.get("invalid_code") == {"bits": "10", "value": -2},
            "T2 reserved code policy is invalid")
    scale = representation.get("scale")
    _expect(isinstance(scale, dict) and scale.get("storage") == "raw_ieee754_binary16_bits"
            and scale.get("zero_point") is None, "T2 scale contract is invalid")
    _expect(representation.get("t2_logical_parameter_count") == EXPECTED_T2_LOGICAL_PARAMETERS,
            "T2 logical parameter inventory count is invalid")
    logical_parameters = representation.get("t2_logical_parameters")
    _expect(isinstance(logical_parameters, list)
            and set(logical_parameters) == expected_t2_parameters(),
            "T2 logical parameter inventory is invalid")

    rotation = identity.get("rotation")
    _expect(isinstance(rotation, dict) and rotation.get("input_basis") == "prism_rotated",
            "T2 projection input basis must be prism_rotated")
    _expect(rotation.get("activation_transform") == "R(x)=H_1024(signs * x)",
            "T2 activation transform is invalid")
    _expect(rotation.get("weight_relation") == "W_rotated = W_primal * diag(signs) * H_1024",
            "T2 rotated-weight relation is invalid")
    signs = rotation.get("signs")
    _expect(isinstance(signs, dict)
            and signs.get("binding") == "per_Use.auxiliaries.hadamard_signs.object"
            and signs.get("storage_kind") == "auxiliary_tensor"
            and signs.get("format") == "bf16"
            and signs.get("layout") == "contiguous_le_v1"
            and signs.get("value_set") == [-1, 1]
            and signs.get("input_widths") == [5120, 6144, 17408],
            "T2 Hadamard sign tensor contract is invalid")

    learner = identity.get("learner")
    _expect(isinstance(learner, dict) and learner.get("base_trainable") is False,
            "T2 base must remain frozen")
    adapter = learner.get("adapter")
    _validate_t2_lora_profile(adapter)
    _expect(adapter.get("format") == "standard_peft_lora"
            and adapter.get("dropout") == 0.0 and adapter.get("bias") == "none",
            "standard T2 LoRA profile is unsupported")
    from .lora_targets import gdn_lora_enabled, projection_input_bases
    if gdn_lora_enabled(identity):
        _expect(adapter.get("input_basis") == "per_use",
                "all-GDN LoRA must declare per-Use projection inputs")
        _expect(adapter.get("input_basis_by_module") == projection_input_bases(identity),
                "all-GDN LoRA input bases must exactly cover all declared projections")
    else:
        _expect(adapter.get("input_basis") == "prism_rotated",
                "LoRA input basis must match the frozen T2 projection input basis")
    _expect(adapter.get("output_basis") == "projection_output",
            "LoRA output basis must match the projection output basis")
    _expect(adapter.get("lora_on_mtp_proposal_head") is False,
            "MTP proposal head cannot be declared LoRA-enabled")

    mtp = identity.get("mtp_checkpoint")
    _expect(isinstance(mtp, dict) and mtp.get("is_lora_adapter") is False
            and mtp.get("role") == "MTP_checkpoint_only",
            "MTP checkpoint must be recorded separately from LoRA adapters")
    local_copy = mtp.get("local_copy")
    if local_copy is not None:
        _expect(isinstance(local_copy, dict) and isinstance(local_copy.get("path"), str)
                and Path(local_copy["path"]).is_absolute()
                and Path(local_copy["path"]).suffix == ".safetensors"
                and isinstance(local_copy.get("size_bytes"), int) and local_copy["size_bytes"] > 0
                and (local_copy.get("sha256") is None or (
                    isinstance(local_copy.get("sha256"), str)
                    and SHA256_RE.fullmatch(local_copy["sha256"]) is not None
                )),
                "MTP checkpoint copy identity is invalid")

    raw_gguf = source.get("raw_gguf")
    _expect(isinstance(raw_gguf, dict), "raw GGUF lineage status is missing")
    if raw_gguf.get("status") == "missing_unverified":
        _expect(raw_gguf.get("sha256") is None, "missing raw GGUF must not claim a content hash")
    elif raw_gguf.get("status") == "content_hashed":
        _expect(isinstance(raw_gguf.get("sha256"), str)
                and SHA256_RE.fullmatch(raw_gguf["sha256"]) is not None,
                "present raw GGUF content hash is invalid")

    if verify_files:
        payloads = [ninfer]
        if local_copy is not None:
            payloads.append(local_copy)
        for source_key in ("raw_gguf", "mtp_source", "base_assets_source"):
            source_item = source.get(source_key)
            if isinstance(source_item, dict) and source_item.get("status") == "present_unverified":
                payloads.append(source_item)
        for item in payloads:
            path = _regular_file(item["path"], "identity input")
            _expect(path.stat().st_size == item["size_bytes"],
                    f"identity input size changed: {item['path']}")
        report_path = _regular_file(report["path"], "conversion report")
        _expect(report_path.stat().st_size == report["size_bytes"],
                "conversion report size changed")
        _expect(_sha256(report_path) == report["sha256"],
                "conversion report metadata SHA-256 changed")
        _check_pinned_header(Path(ninfer["path"]), ninfer["artifact_id"])
    return dict(identity)


@dataclass(frozen=True, slots=True)
class TernaryArtifactIdentity:
    """Validated wrapper for the JSON identity consumed by runtime config."""

    document: dict[str, Any]

    @classmethod
    def from_document(cls, value: Mapping[str, Any], *, verify_files: bool = False) -> "TernaryArtifactIdentity":
        return cls(validate_artifact_identity(value, verify_files=verify_files))

    @property
    def artifact_id(self) -> str:
        return str(self.document["source"]["ninfer_artifact"]["artifact_id"])

    @property
    def artifact_sha256(self) -> str:
        return str(self.document["source"]["ninfer_artifact"]["sha256"])

    @property
    def input_basis(self) -> str:
        return str(self.document["learner"]["adapter"]["input_basis"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.document)


def artifact_report(identity: Mapping[str, Any] | TernaryArtifactIdentity) -> dict[str, Any]:
    """Return the report descriptor defined by an identity's exact model profile."""
    document = identity.document if isinstance(identity, TernaryArtifactIdentity) else identity
    _expect(isinstance(document, Mapping), "artifact identity must be an object")
    model = document.get("model")
    source = document.get("source")
    _expect(isinstance(model, dict) and isinstance(source, dict),
            "artifact identity has no model or source report")
    if model.get("id") == MODEL_NAME:
        report = source.get("conversion_report")
    elif model.get("id") == CODER390_MODEL_NAME:
        report = source.get("artifact_report")
    else:
        raise TernaryContractError("artifact report lookup has an unsupported model profile")
    _expect(isinstance(report, dict), "artifact identity report descriptor is missing")
    return dict(report)


def write_artifact_identity(
    ninfer_path: str | Path,
    conversion_report_path: str | Path,
    output_path: str | Path,
    *,
    mtp_checkpoint_path: str | Path | None = None,
    declared_artifact_sha256: str | None = None,
    overwrite: bool = False,
) -> TernaryArtifactIdentity:
    identity = build_artifact_identity(
        ninfer_path,
        conversion_report_path,
        mtp_checkpoint_path=mtp_checkpoint_path,
        declared_artifact_sha256=declared_artifact_sha256,
    )
    output = Path(output_path).expanduser()
    if output.is_symlink():
        raise TernaryContractError("identity output path must not be a symlink")
    if output.exists() and not overwrite:
        raise TernaryContractError(f"identity output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise TernaryContractError(f"identity temporary path already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    return TernaryArtifactIdentity.from_document(identity)


def load_artifact_identity(path: str | Path, *, verify_files: bool = True) -> TernaryArtifactIdentity:
    manifest_path = _regular_file(path, "T2 artifact identity manifest")
    identity = _read_json(manifest_path, "T2 artifact identity manifest")
    return TernaryArtifactIdentity.from_document(identity, verify_files=verify_files)


def _check_pinned_header(path: Path, artifact_id: str) -> None:
    with path.open("rb") as stream:
        header = stream.read(32)
    _expect(len(header) == 32, "NInfer pinned base header is incomplete")
    magic, directory_bytes, actual_id = struct.unpack("<8sQ16s", header)
    _expect(magic == b"NINFER\x00\x03" and actual_id.hex() == artifact_id,
            "NInfer pinned base header or artifact ID differs from deployment")
    _expect(0 < directory_bytes <= path.stat().st_size - 32,
            "NInfer pinned base directory size is invalid")


def load_pinned_artifact_identity(path: str | Path) -> TernaryArtifactIdentity:
    """Load deployment identity without scanning model payloads; no content-hash claim."""
    identity = load_artifact_identity(path, verify_files=False)
    source = identity.document["source"]
    base, report = source["ninfer_artifact"], artifact_report(identity)
    for item in (base, report):
        actual = _regular_file(item["path"], "pinned identity input")
        _expect(actual.stat().st_size == item["size_bytes"], "pinned identity input size changed")
    _check_pinned_header(Path(base["path"]), base["artifact_id"])
    _expect(_sha256(Path(report["path"])) == report["sha256"], "artifact report SHA changed")
    return identity


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create or verify a frozen Qwen T2 learner artifact identity contract."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="describe T2 inputs using deployment-declared base identity")
    create.add_argument("--ninfer", required=True, help="existing frozen T2 .ninfer artifact")
    create.add_argument("--report", required=True, help="matching artifact conversion or run report JSON")
    create.add_argument("--mtp-checkpoint", help="optional separate MTP .safetensors copy")
    create.add_argument(
        "--declared-artifact-sha256",
        required=True,
        help="deployment-declared base identity; model payload is not scanned",
    )
    create.add_argument("--output", help="identity manifest output path; omit for stdout")
    create.add_argument("--overwrite", action="store_true", help="replace an existing identity manifest")

    verify = subparsers.add_parser("verify", help="validate metadata SHA, file sizes, and NInfer header identity")
    verify.add_argument("--manifest", required=True, help="identity manifest created by the create command")

    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            if args.output:
                identity = write_artifact_identity(
                    args.ninfer,
                    args.report,
                    args.output,
                    mtp_checkpoint_path=args.mtp_checkpoint,
                    declared_artifact_sha256=args.declared_artifact_sha256,
                    overwrite=args.overwrite,
                )
                _print_json({"status": "created", "manifest": str(Path(args.output).resolve()),
                             "artifact_id": identity.artifact_id, "artifact_sha256": identity.artifact_sha256})
            else:
                _print_json(build_artifact_identity(
                    args.ninfer,
                    args.report,
                    mtp_checkpoint_path=args.mtp_checkpoint,
                    declared_artifact_sha256=args.declared_artifact_sha256,
                ))
            return 0
        identity = load_artifact_identity(args.manifest, verify_files=True)
        _print_json({"status": "verified", "manifest": str(Path(args.manifest).resolve()),
                     "artifact_id": identity.artifact_id, "artifact_sha256": identity.artifact_sha256,
                     "artifact_sha256_verified": False,
                     "lora_input_basis": identity.input_basis})
        return 0
    except (OSError, TernaryContractError) as exc:
        print(f"ternary contract error: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "CODER390_MODEL_NAME",
    "SCHEMA_VERSION",
    "TernaryArtifactIdentity",
    "TernaryContractError",
    "artifact_report",
    "build_artifact_identity",
    "expected_coder390_t2_parameters",
    "expected_t2_parameters",
    "load_artifact_identity",
    "validate_artifact_identity",
    "write_artifact_identity",
]


if __name__ == "__main__":
    raise SystemExit(main())
