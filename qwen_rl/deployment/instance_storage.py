"""Read-only disk-capacity estimates for a stopped managed instance.

The packed T2 cache estimate comes from the NInfer directory and the pinned
artifact layout. Checkpoint headroom comes from the current durable snapshot,
which the checkpoint writer replaces atomically. This module does not load a
model, initialize CUDA, hash model payloads, or remove files.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Callable, Mapping

_SCHEMA = "qwen-rl-instance-storage-preflight/v1"
_CHECKPOINT_NAME = re.compile(r"state-(\d{8})-([0-9a-f]{32})\.pt\Z")
_STORE_KEY = re.compile(r"[0-9a-f]{64}\Z")
_T2_FORMAT = "t2_g128_fp16"
_T2_LAYOUT = "row_split_k128_v1"
_STORE_LAYOUT = "qwen35_t2_ddp_decoder_packed/v1"
_ALIGNMENT = 16
_MIB = 1024**2
_MIN_HEADROOM_BYTES = 256 * _MIB
_HEADROOM_FRACTION = 0.05
_MAX_SMALL_JSON_BYTES = 16 * _MIB
_GENESIS_METADATA_SLACK_BYTES = 1 * _MIB
_COMPILER_CACHE_NAMES = (
    "hf", "huggingface", "vllm", "triton", "torch", "torchinductor",
    "extensions", "cuda", "flashinfer",
)


class StoragePreflightError(ValueError):
    """Storage requirements could not be established from local metadata."""


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _identity_path(runtime: Mapping[str, Any]) -> Path:
    model = runtime.get("model") if isinstance(runtime.get("model"), Mapping) else {}
    value = model.get("artifact_identity_path") or runtime.get("artifact_identity_path")
    if not isinstance(value, str) or not value.strip():
        raise StoragePreflightError("artifact identity path is unavailable")
    path = Path(value).expanduser()
    if not path.is_absolute():
        from ..paths import ROOT
        path = ROOT / path
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_SMALL_JSON_BYTES:
        raise StoragePreflightError("artifact identity is unavailable")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_SMALL_JSON_BYTES:
        raise StoragePreflightError(f"{label} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise StoragePreflightError(f"{label} is unreadable") from None
    if not isinstance(value, dict):
        raise StoragePreflightError(f"{label} is malformed")
    return value


def _artifact_layout(runtime: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    identity_file = _identity_path(runtime)
    identity = _read_json(identity_file, "artifact identity")
    source = identity.get("source")
    ninfer = source.get("ninfer_artifact") if isinstance(source, Mapping) else None
    report = ((source.get("conversion_report") or source.get("artifact_report"))
              if isinstance(source, Mapping) else None)
    report_path = (report.get("path") if isinstance(report, Mapping) else None)
    if not isinstance(ninfer, Mapping) or not isinstance(ninfer.get("artifact_id"), str):
        raise StoragePreflightError("artifact identity has no NInfer artifact ID")
    if not isinstance(report, Mapping):
        raise StoragePreflightError("artifact identity has no conversion report descriptor")
    report_inventory = report.get("format_inventory", report.get("format_counts"))
    if not isinstance(report_inventory, Mapping) or not report_inventory.get(_T2_FORMAT):
        raise StoragePreflightError("conversion report has no T2 inventory")
    if not isinstance(report_path, str) or not report_path:
        raise StoragePreflightError("conversion report path is unavailable")
    report_file = Path(report_path).expanduser()
    if not report_file.is_absolute():
        from ..paths import ROOT
        report_file = ROOT / report_file
    if report_file.is_symlink() or not report_file.is_file() or report_file.stat().st_size > _MAX_SMALL_JSON_BYTES:
        raise StoragePreflightError("conversion report is unavailable")
    try:
        report_doc = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise StoragePreflightError("conversion report is unreadable") from None
    if not isinstance(report_doc, dict):
        raise StoragePreflightError("conversion report is malformed")
    report_artifact_id = report_doc.get("artifact_id")
    if report.get("report_kind") == "coder390_full_scales_step512_run_report":
        artifact_record = report_doc.get("artifact")
        report_artifact_id = artifact_record.get("artifact_id") if isinstance(artifact_record, Mapping) else None
        actual_t2_count = (artifact_record.get("overlaid_storage_objects")
                           if isinstance(artifact_record, Mapping) else None)
    else:
        actual_inventory = report_doc.get("formats")
        actual_t2_count = (actual_inventory.get(_T2_FORMAT)
                           if isinstance(actual_inventory, Mapping) else None)
    if report_artifact_id != ninfer["artifact_id"]:
        raise StoragePreflightError("conversion report artifact ID differs from the identity")
    expected_t2_count = report_inventory.get(_T2_FORMAT)
    if (type(actual_t2_count) is not int or actual_t2_count <= 0
            or type(expected_t2_count) is not int or actual_t2_count != expected_t2_count):
        raise StoragePreflightError("conversion report T2 inventory differs from the identity")

    inference = (runtime.get("worker_inference")
                 if isinstance(runtime.get("worker_inference"), Mapping) else {})
    artifact_value = inference.get("ninfer_artifact_path") or runtime.get("model_path")
    if not isinstance(artifact_value, str) or not artifact_value.strip():
        raise StoragePreflightError("NInfer artifact path is unavailable")
    artifact_path = Path(artifact_value).expanduser()
    if not artifact_path.is_absolute():
        from ..paths import ROOT
        artifact_path = ROOT / artifact_path

    # This existing contract helper reads only the fixed NInfer header and its
    # JSON directory. It never scans the packed model payload.
    from ..contracts.ternary import _read_ninfer_directory
    try:
        directory, artifact_id, _payload_bytes = _read_ninfer_directory(artifact_path)
    except Exception as error:
        raise StoragePreflightError(f"NInfer header/layout is unavailable ({type(error).__name__})") from None
    if artifact_id != ninfer["artifact_id"]:
        raise StoragePreflightError("NInfer header artifact ID differs from the identity")
    header_objects = directory.get("objects")
    header_t2_count = (sum(isinstance(obj, Mapping) and obj.get("format") == _T2_FORMAT
                           for obj in header_objects)
                       if isinstance(header_objects, list) else 0)
    if header_t2_count != actual_t2_count:
        raise StoragePreflightError("NInfer header T2 inventory differs from the conversion report")
    representation = identity.get("representation")
    if (not isinstance(representation, Mapping)
            or representation.get("format") != _T2_FORMAT
            or representation.get("layout") != _T2_LAYOUT
            or representation.get("group_size_k") != 128):
        raise StoragePreflightError("artifact identity has an unsupported packed T2 layout")
    return identity, directory


def estimate_packed_layout(directory: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate the canonical decoder cache bytes from NInfer header metadata.

    Each logical decoder matrix is treated as its own pair of packed buffers.
    The live model may fuse adjacent projections, so this is a small conservative
    alignment upper bound that follows the same 16-byte buffer alignment used by
    ``build_packed_weight_plan``. Tensor bytes come from the actual header shapes.
    """
    components = directory.get("components")
    text = components.get("text") if isinstance(components, Mapping) else None
    config = text.get("config") if isinstance(text, Mapping) else None
    layer_count = config.get("num_hidden_layers") if isinstance(config, Mapping) else None
    bindings = directory.get("bindings")
    objects_list = directory.get("objects")
    if (type(layer_count) is not int or layer_count <= 0
            or not isinstance(bindings, Mapping) or not isinstance(objects_list, list)):
        raise StoragePreflightError("NInfer header lacks decoder layout metadata")
    objects = {item.get("id"): item for item in objects_list
               if isinstance(item, Mapping) and isinstance(item.get("id"), str)}
    if len(objects) != len(objects_list):
        raise StoragePreflightError("NInfer header object inventory is malformed")

    layers: list[list[tuple[int, int]]] = [[] for _ in range(layer_count)]
    observed: set[str] = set()
    for name, binding in bindings.items():
        if not isinstance(name, str) or not name.startswith("text/layers/"):
            continue
        pieces = name.split("/", 3)
        if len(pieces) != 4 or not pieces[2].isdecimal():
            raise StoragePreflightError("NInfer decoder binding name is malformed")
        layer = int(pieces[2])
        if not 0 <= layer < layer_count:
            raise StoragePreflightError("NInfer decoder binding is outside the configured layer count")
        if not isinstance(binding, Mapping):
            raise StoragePreflightError("NInfer decoder binding is malformed")
        parts = binding.get("parts")
        if parts is None:
            parts = [binding]
        if not isinstance(parts, list) or not parts:
            raise StoragePreflightError("NInfer decoder binding has no storage parts")
        rows_total = 0
        width: int | None = None
        for part in parts:
            if not isinstance(part, Mapping) or not isinstance(part.get("object"), str):
                raise StoragePreflightError("NInfer decoder storage part is malformed")
            obj = objects.get(part["object"])
            if not isinstance(obj, Mapping) or obj.get("format") != _T2_FORMAT:
                # Non-T2 decoder bindings are outside this packed cache.
                continue
            if obj.get("layout") != _T2_LAYOUT:
                raise StoragePreflightError("NInfer decoder T2 layout is unsupported")
            shape = obj.get("shape")
            if (not isinstance(shape, list) or len(shape) != 2
                    or any(type(value) is not int or value <= 0 for value in shape)):
                raise StoragePreflightError("NInfer decoder T2 shape is malformed")
            rows, current_width = shape
            if current_width % 128:
                raise StoragePreflightError("NInfer decoder T2 width does not divide group size 128")
            bounds = part.get("range", [0, rows * current_width])
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(value) is not int for value in bounds)):
                raise StoragePreflightError("NInfer decoder T2 row range is malformed")
            begin, end = bounds
            if (not 0 <= begin < end <= rows * current_width
                    or begin % current_width or end % current_width):
                raise StoragePreflightError("NInfer decoder T2 range does not cover complete rows")
            if width is not None and width != current_width:
                raise StoragePreflightError("NInfer decoder T2 parts use different widths")
            width = current_width
            rows_total += (end - begin) // current_width
        if rows_total:
            if name in observed:
                raise StoragePreflightError("NInfer decoder binding is duplicated")
            observed.add(name)
            assert width is not None
            code_bytes = rows_total * (width // 4)
            scale_bytes = rows_total * (width // 128) * 2
            layers[layer].append((code_bytes, scale_bytes))

    if not observed or any(not entries for entries in layers):
        raise StoragePreflightError("NInfer header does not describe every packed decoder layer")
    groups = []
    raw_total = 0
    aligned_total = 0
    for index, entries in enumerate(layers):
        cursor = 0
        for code_bytes, scale_bytes in sorted(entries):
            raw_total += code_bytes + scale_bytes
            cursor = _align_up(cursor) + code_bytes
            cursor = _align_up(cursor) + scale_bytes
        groups.append({"index": index, "logical_matrix_count": len(entries),
                       "payload_bytes": sum(code + scale for code, scale in entries),
                       "storage_bytes_upper_bound": cursor})
        aligned_total += cursor
    return {
        "schema": "qwen-rl-packed-layout-estimate/v1",
        "layout": _STORE_LAYOUT,
        "source": "NInfer-v3 header shapes plus identity conversion-report T2 inventory",
        "layer_count": layer_count,
        "logical_matrix_count": len(observed),
        "payload_bytes": raw_total,
        "alignment_upper_bound_bytes": aligned_total,
        "alignment_padding_upper_bound_bytes": aligned_total - raw_total,
        "groups": groups,
    }


def _signature(info: os.stat_result) -> dict[str, int]:
    return {name: int(getattr(info, "st_" + name))
            for name in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}


def _same_json(left: Any, right: Any) -> bool:
    try:
        return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
            right, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError):
        return False


def _packed_cache_state(root: Path, identity: Mapping[str, Any], estimate: Mapping[str, Any]) -> dict[str, Any]:
    expected_upper = estimate["alignment_upper_bound_bytes"]
    expected_raw = estimate["payload_bytes"]
    matches: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    unclassified_key = False
    if root.is_symlink():
        raise StoragePreflightError("packed cache root must not be a symlink")
    if root.exists() and not root.is_dir():
        raise StoragePreflightError("packed cache root is not a directory")
    if root.is_dir():
        for entry in root.iterdir():
            if entry.is_symlink():
                inventory.append({"path": str(entry), "state": "symlink_not_followed",
                                  "allocated_bytes": 0, "reference_status": "unknown"})
                continue
            if not entry.is_dir():
                continue
            manifest_path = entry / "manifest.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                unclassified_key = unclassified_key or bool(_STORE_KEY.fullmatch(entry.name))
                inventory.append({"path": str(entry), "state": "unclassified",
                                  "allocated_bytes": _tree_inventory(entry, seen_inodes=set())[0],
                                  "reference_status": "unknown"})
                continue
            try:
                manifest = _read_json(manifest_path, "packed cache manifest")
            except StoragePreflightError:
                unclassified_key = unclassified_key or bool(_STORE_KEY.fullmatch(entry.name))
                inventory.append({"path": str(entry), "state": "manifest_unreadable",
                                  "allocated_bytes": _tree_inventory(entry, seen_inodes=set())[0],
                                  "reference_status": "unknown"})
                continue
            descriptor = manifest.get("descriptor")
            descriptor_identity = descriptor.get("identity") if isinstance(descriptor, Mapping) else None
            is_current_identity = (isinstance(descriptor_identity, Mapping)
                                   and descriptor_identity.get("layout") == _STORE_LAYOUT
                                   and _same_json(descriptor_identity.get("artifact_identity"), identity))
            groups = descriptor.get("group_sizes") if isinstance(descriptor, Mapping) else None
            payload_path = entry / "packed.bin"
            payload_info = None
            try:
                if not payload_path.is_symlink() and payload_path.is_file():
                    payload_info = payload_path.stat()
            except OSError:
                payload_info = None
            expected_signature = manifest.get("file_signature")
            signature_valid = (payload_info is not None
                               and expected_signature == _signature(payload_info))
            size_valid = (payload_info is not None
                          and expected_raw <= payload_info.st_size <= expected_upper)
            groups_valid = (isinstance(groups, list)
                            and len(groups) == estimate["layer_count"]
                            and all(type(size) is int and size > 0 for size in groups)
                            and sum(groups) == (payload_info.st_size if payload_info else -1))
            entry_state = "valid" if signature_valid and size_valid and groups_valid else "partial"
            reference = "current_runtime_cache" if is_current_identity else "noncurrent_identity_retained"
            size, _inventory_state = _tree_inventory(entry, seen_inodes=set())
            inventory.append({"path": str(entry), "state": entry_state if is_current_identity else "noncurrent_identity",
                              "allocated_bytes": size, "reference_status": reference})
            if is_current_identity:
                matches.append({"path": str(entry), "state": entry_state,
                                "allocated_bytes": size,
                                "payload_bytes": payload_info.st_size if payload_info else 0})
    valid = next((row for row in matches if row["state"] == "valid"), None)
    if valid is not None:
        return {"state": "valid", "path": valid["path"], "existing_bytes": valid["payload_bytes"],
                "incremental_bytes": 0, "expected_bytes": valid["payload_bytes"],
                "duplicate_partial_entries": [row for row in matches if row["state"] == "partial"],
                "entries": inventory}
    partial = next((row for row in matches if row["state"] == "partial"), None)
    cache_state = "partial" if partial else "unclassified" if unclassified_key else "missing"
    return {"state": cache_state,
            "path": partial["path"] if partial else str(root),
            "existing_bytes": partial["payload_bytes"] if partial else 0,
            # The shared store does not resume partial files; budget a complete
            # replacement and do not credit bytes in the incomplete entry.
            "incremental_bytes": expected_upper,
            "expected_bytes": expected_upper,
            "matching_partial_entries": matches if partial else [],
            "entries": inventory}


def _read_checkpoint_record(root: Path) -> dict[str, Any] | None:
    """Read a pointer and stat its snapshot; None is allowed only when latest is absent."""
    pointer_path = root / "latest.json"
    try:
        pointer_info = pointer_path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise StoragePreflightError("checkpoint pointer is unavailable") from None
    if not stat.S_ISREG(pointer_info.st_mode) or stat.S_ISLNK(pointer_info.st_mode):
        raise StoragePreflightError("checkpoint pointer is not a regular file")
    pointer = _read_json(pointer_path, "checkpoint pointer")
    snapshot_name = pointer.get("snapshot")
    match = _CHECKPOINT_NAME.fullmatch(snapshot_name) if isinstance(snapshot_name, str) else None
    generation, snapshot_id = pointer.get("generation"), pointer.get("snapshot_id")
    policy_version, optimizer_steps = pointer.get("policy_version"), pointer.get("optimizer_steps")
    if (pointer.get("schema") != "central-gradient-checkpoint-pointer/v1"
            or match is None or type(generation) is not int or generation < 0
            or not isinstance(snapshot_id, str)
            or generation != int(match.group(1)) or snapshot_id != match.group(2)
            or type(pointer.get("size_bytes")) is not int or pointer["size_bytes"] <= 0
            or type(policy_version) is not int or policy_version < 0
            or type(optimizer_steps) is not int or optimizer_steps < 0):
        raise StoragePreflightError("durable checkpoint pointer is unsupported or inconsistent")
    snapshot_path = root / pointer["snapshot"]
    try:
        snapshot_info = snapshot_path.lstat()
    except OSError:
        raise StoragePreflightError("durable checkpoint snapshot is unavailable") from None
    if (not stat.S_ISREG(snapshot_info.st_mode) or stat.S_ISLNK(snapshot_info.st_mode)
            or snapshot_info.st_size <= 0 or snapshot_info.st_size != pointer["size_bytes"]):
        raise StoragePreflightError("durable checkpoint snapshot differs from its pointer")
    return {"root": root, "snapshot_path": snapshot_path, "snapshot_bytes": snapshot_info.st_size,
            "pointer_bytes": pointer_info.st_size, "generation": generation,
            "policy_version": policy_version, "optimizer_steps": optimizer_steps}


def _pointer_json_bytes(snapshot_bytes: int, *, generation: int,
                        policy_version: int, optimizer_steps: int) -> int:
    snapshot_id = "f" * 32
    value = {
        "schema": "central-gradient-checkpoint-pointer/v1",
        "generation": generation,
        "snapshot_id": snapshot_id,
        "snapshot": f"state-{generation:08d}-{snapshot_id}.pt",
        "size_bytes": snapshot_bytes,
        "policy_version": policy_version,
        "optimizer_steps": optimizer_steps,
    }
    return len((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def _genesis_checkpoint_upper_bound(runtime: Mapping[str, Any]) -> dict[str, Any]:
    learner = runtime.get("learner") if isinstance(runtime.get("learner"), Mapping) else {}
    adapter_value = learner.get("initial_adapter_path")
    if not isinstance(adapter_value, str) or not adapter_value.strip():
        raise StoragePreflightError("genesis checkpoint estimate requires initial_adapter_path")
    adapter_root = Path(adapter_value).expanduser()
    if not adapter_root.is_absolute():
        from ..paths import ROOT
        adapter_root = ROOT / adapter_root
    if adapter_root.is_symlink() or not adapter_root.is_dir():
        raise StoragePreflightError("genesis initial adapter directory is unavailable")
    paths = {
        "config": adapter_root / "adapter_config.json",
        "weights": adapter_root / "adapter_model.safetensors",
        "provenance": adapter_root / "provenance.json",
    }
    infos = {}
    for name, path in paths.items():
        try:
            info = path.lstat()
        except OSError:
            raise StoragePreflightError(f"genesis initial adapter {name} file is unavailable") from None
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise StoragePreflightError(f"genesis initial adapter {name} file is unsafe")
        infos[name] = info
    config = _read_json(paths["config"], "initial adapter config")
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if (type(rank) is not int or rank <= 0 or type(alpha) not in {int, float}
            or not math.isfinite(float(alpha)) or alpha <= 0
            or config.get("bias", "none") != "none" or config.get("use_dora", False)
            or config.get("use_rslora", False) or config.get("rank_pattern")
            or config.get("alpha_pattern")):
        raise StoragePreflightError("genesis initial adapter does not use the supported LoRA layout")
    try:
        from ..models.artifact import _safetensors_header

        tensors, _metadata = _safetensors_header(paths["weights"])
    except Exception as error:
        raise StoragePreflightError(
            f"genesis initial adapter header is unavailable ({type(error).__name__})"
        ) from None
    pairs: dict[str, set[str]] = {}
    parameter_count = 0
    serialized_tensor_bytes = 0
    for name, item in tensors.items():
        if not isinstance(item, Mapping):
            raise StoragePreflightError("genesis initial adapter tensor header is malformed")
        suffix = next((suffix for suffix in (".lora_A.weight", ".lora_B.weight")
                       if name.endswith(suffix)), None)
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if (suffix is None or not isinstance(shape, list) or len(shape) != 2
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)):
            raise StoragePreflightError("genesis initial adapter contains an unsupported LoRA tensor")
        if ((suffix == ".lora_A.weight" and shape[0] != rank)
                or (suffix == ".lora_B.weight" and shape[1] != rank)):
            raise StoragePreflightError("genesis initial adapter tensor ranks differ from adapter_config.json")
        side = "A" if suffix == ".lora_A.weight" else "B"
        stem = name[:-len(suffix)]
        sides = pairs.setdefault(stem, set())
        if side in sides:
            raise StoragePreflightError("genesis initial adapter has duplicate LoRA factors")
        sides.add(side)
        parameter_count += math.prod(shape)
        serialized_tensor_bytes += offsets[1] - offsets[0]
    if not pairs or any(sides != {"A", "B"} for sides in pairs.values()):
        raise StoragePreflightError("genesis initial adapter lacks complete LoRA factor pairs")
    header_bytes = infos["weights"].st_size - serialized_tensor_bytes
    if header_bytes < 8:
        raise StoragePreflightError("genesis initial adapter file size differs from its header")

    # The central owner promotes LoRA masters to FP32. AdamW adds two FP32
    # moments per parameter after the first update; include per-tensor step
    # scalars and measured adapter metadata for a genesis upper bound.
    trainable_bytes = max(parameter_count * 4, infos["weights"].st_size)
    adam_bytes = parameter_count * 8 + len(tensors) * 8
    metadata_bytes = (header_bytes + infos["config"].st_size + infos["provenance"].st_size
                      + _GENESIS_METADATA_SLACK_BYTES)
    snapshot_bytes = trainable_bytes + adam_bytes + metadata_bytes
    pointer_bytes = _pointer_json_bytes(
        snapshot_bytes, generation=0, policy_version=0, optimizer_steps=0,
    )
    return {
        "root": None, "snapshot_path": None, "snapshot_bytes": snapshot_bytes,
        "pointer_bytes": pointer_bytes, "generation": None, "policy_version": 0,
        "optimizer_steps": 0, "source": "genesis_upper_bound_from_initial_adapter_header",
        "initial_adapter_path": str(adapter_root),
        "initial_adapter_file_bytes": infos["weights"].st_size,
        "initial_adapter_header_bytes": header_bytes,
        "trainable_parameter_count": parameter_count,
        "trainable_state_bytes_upper_bound": trainable_bytes,
        "adam_moment_and_step_bytes_upper_bound": adam_bytes,
        "metadata_bytes_upper_bound": metadata_bytes,
    }


def _checkpoint_temp_write(runtime: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    learner = runtime.get("learner") if isinstance(runtime.get("learner"), Mapping) else {}
    recovery_value = learner.get("recovery_dir")
    if not isinstance(recovery_value, str) or not recovery_value.strip():
        raise StoragePreflightError("learner recovery directory is unavailable")
    target_root = Path(recovery_value).expanduser()
    if not target_root.is_absolute():
        from ..paths import ROOT
        target_root = ROOT / target_root
    if target_root.is_symlink() or (target_root.exists() and not target_root.is_dir()):
        raise StoragePreflightError("checkpoint destination is unsafe")

    # Never use the source filesystem for the disk-capacity budget. An explicit
    # preview source supplies only metadata for the expected snapshot size.
    destination_record = _read_checkpoint_record(target_root)
    source_value = plan.get("storage_checkpoint_source")
    source_record = None
    source_root = None
    if source_value is not None:
        if not isinstance(source_value, (str, Path)) or not str(source_value).strip():
            raise StoragePreflightError("storage_checkpoint_source is invalid")
        source_root = Path(source_value).expanduser()
        if not source_root.is_absolute() or source_root.is_symlink() or not source_root.is_dir():
            raise StoragePreflightError("storage_checkpoint_source is unavailable")
        source_record = _read_checkpoint_record(source_root)
        if source_record is None:
            raise StoragePreflightError("storage checkpoint preview source has no durable checkpoint")
    if destination_record is None and target_root.exists():
        try:
            has_orphaned_state = next(target_root.iterdir(), None) is not None
        except OSError:
            raise StoragePreflightError("checkpoint destination could not be inspected") from None
        if has_orphaned_state:
            raise StoragePreflightError("checkpoint destination has files but no durable latest pointer")

    adapter_estimate = _genesis_checkpoint_upper_bound(runtime)
    record = source_record or destination_record
    observed_bytes = record["snapshot_bytes"] if record is not None else None
    snapshot_bytes = max(observed_bytes or 0, adapter_estimate["snapshot_bytes"])
    current_pointer_bytes = record["pointer_bytes"] if record is not None else 0
    generation = record.get("generation") if record is not None else None
    policy_version = record.get("policy_version", 0) if record is not None else 0
    optimizer_steps = record.get("optimizer_steps", 0) if record is not None else 0
    next_pointer_bytes = _pointer_json_bytes(
        snapshot_bytes,
        generation=(generation + 1) if generation is not None else 0,
        policy_version=(policy_version + 1) if generation is not None else 0,
        optimizer_steps=(optimizer_steps + 1) if generation is not None else 0,
    )
    pointer_temp_bytes = max(current_pointer_bytes, next_pointer_bytes)
    source_observed = source_record is not None and source_root.resolve() != target_root.resolve()
    if record is None:
        estimate_source = "genesis_upper_bound_from_initial_adapter_header"
    elif observed_bytes < adapter_estimate["snapshot_bytes"]:
        estimate_source = "max_current_snapshot_and_initial_adapter_upper_bound"
    else:
        estimate_source = "current_snapshot_at_least_initial_adapter_upper_bound"
    return {
        "path": str(target_root), "snapshot_path": str(record["snapshot_path"])
        if record is not None else None,
        "observation_path": str(source_root if source_observed else target_root),
        "source_observed": source_observed,
        "snapshot_bytes": snapshot_bytes,
        "observed_snapshot_bytes": observed_bytes,
        "adapter_checkpoint_upper_bound_bytes": adapter_estimate["snapshot_bytes"],
        "pointer_temporary_bytes": pointer_temp_bytes,
        "temporary_write_bytes": snapshot_bytes + pointer_temp_bytes,
        "estimate_source": estimate_source,
        "method": ("source snapshot metadata; writes budgeted on target recovery filesystem"
                   if source_observed else
                   "max of current snapshot stat and initial-adapter FP32 LoRA/AdamW geometry"
                   if record is not None else
                   "genesis upper bound from safetensors header and FP32 AdamW geometry"),
        **{key: adapter_estimate[key] for key in (
            "initial_adapter_path", "initial_adapter_file_bytes", "initial_adapter_header_bytes",
            "trainable_parameter_count", "trainable_state_bytes_upper_bound",
            "adam_moment_and_step_bytes_upper_bound", "metadata_bytes_upper_bound",
        ) if key in adapter_estimate},
    }


def _existing_anchor(path: Path) -> Path:
    current = path.expanduser()
    while not current.exists():
        if current.is_symlink():
            raise OSError("destination path contains a symlink")
        parent = current.parent
        if parent == current:
            raise OSError("no existing filesystem anchor")
        current = parent
    if current.is_file():
        current = current.parent
    return current.resolve(strict=True)


def _filesystem_info(path: Path) -> dict[str, Any]:
    anchor = _existing_anchor(path)
    info = os.stat(anchor)
    stats = os.statvfs(anchor)
    free_by_statvfs = int(stats.f_bavail * stats.f_frsize)
    free_by_shutil = int(shutil.disk_usage(anchor).free)
    filesystem_id = f"{int(info.st_dev)}:{int(getattr(stats, 'f_fsid', 0))}"
    mount = anchor
    while mount.parent != mount:
        try:
            parent_info = os.stat(mount.parent)
            parent_vfs = os.statvfs(mount.parent)
        except OSError:
            break
        parent_id = f"{int(parent_info.st_dev)}:{int(getattr(parent_vfs, 'f_fsid', 0))}"
        if parent_id != filesystem_id:
            break
        mount = mount.parent
    mount_info = _mount_info(anchor)
    return {"filesystem_id": filesystem_id,
            "mount_path": mount_info[0] if mount_info is not None else str(mount),
            "filesystem_type": mount_info[1] if mount_info is not None else None,
            "free_bytes": min(free_by_statvfs, free_by_shutil)}


def _mount_unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _mount_info(path: Path) -> tuple[str, str] | None:
    """Return the longest matching mount point and its filesystem type."""
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    matches: list[tuple[int, str, str]] = []
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        left, right = before.split(), after.split()
        if len(left) < 5 or not right:
            continue
        mountpoint = Path(_mount_unescape(left[4]))
        try:
            if path == mountpoint or path.is_relative_to(mountpoint):
                matches.append((len(str(mountpoint)), str(mountpoint), right[0]))
        except OSError:
            continue
    if not matches:
        return None
    _length, mountpoint, filesystem_type = max(matches)
    return mountpoint, filesystem_type


def _cgroup_memory_files() -> tuple[Path, Path]:
    """Locate this process's memory controller files from proc cgroup metadata."""
    try:
        memberships = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        mounts = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        raise StoragePreflightError("cgroup memory membership is unavailable") from None

    groups: list[tuple[str, str, str]] = []
    for line in memberships:
        pieces = line.split(":", 2)
        if len(pieces) == 3:
            groups.append((pieces[1], _mount_unescape(pieces[2]), "v2" if pieces[1] == "" else "v1"))
    candidates: list[tuple[int, Path, Path]] = []
    for line in mounts:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        left, right = before.split(), after.split()
        if len(left) < 5 or not right:
            continue
        filesystem_type = right[0]
        super_options = set(right[2].split(",")) if len(right) > 2 else set()
        for controllers, group_path, version in groups:
            if version == "v2" and filesystem_type == "cgroup2":
                controller_files = ("memory.max", "memory.current")
            elif (version == "v1" and filesystem_type == "cgroup"
                  and "memory" in super_options):
                controller_files = ("memory.limit_in_bytes", "memory.usage_in_bytes")
            else:
                continue
            mount_root = Path(_mount_unescape(left[3]))
            mountpoint = Path(_mount_unescape(left[4]))
            try:
                if group_path == str(mount_root):
                    relative = Path()
                elif group_path.startswith(str(mount_root).rstrip("/") + "/"):
                    relative = Path(group_path).relative_to(mount_root)
                elif str(mount_root) == "/":
                    relative = Path(group_path.lstrip("/"))
                else:
                    continue
            except ValueError:
                continue
            directory = mountpoint / relative
            limit_file, usage_file = directory / controller_files[0], directory / controller_files[1]
            if limit_file.is_file() and usage_file.is_file():
                candidates.append((len(str(mountpoint)), limit_file, usage_file))
    if not candidates:
        raise StoragePreflightError("cgroup memory controller files are unavailable")
    _length, limit_file, usage_file = max(candidates, key=lambda item: item[0])
    return limit_file, usage_file


def _memory_capacity() -> dict[str, Any]:
    """Observe host MemAvailable and this process's cgroup memory budget."""
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
    except OSError:
        raise StoragePreflightError("host memory availability is unavailable") from None
    available_match = re.search(r"^MemAvailable:\s+(\d+)\s+kB\s*$", meminfo, re.MULTILINE)
    if available_match is None:
        raise StoragePreflightError("host MemAvailable is unavailable")
    host_available = int(available_match.group(1)) * 1024
    limit_path, usage_path = _cgroup_memory_files()
    try:
        limit_text = limit_path.read_text(encoding="ascii").strip()
        current_text = usage_path.read_text(encoding="ascii").strip()
    except OSError:
        raise StoragePreflightError("cgroup memory budget is unreadable") from None
    try:
        current = int(current_text)
        limit = None if limit_text == "max" else int(limit_text)
    except ValueError:
        raise StoragePreflightError("cgroup memory budget is malformed") from None
    if current < 0 or (limit is not None and limit <= 0):
        raise StoragePreflightError("cgroup memory budget is invalid")
    # cgroup v1 uses a very large sentinel for an unlimited controller.
    if limit is not None and limit >= 2**60:
        limit = None
    return {
        "host_available_bytes": host_available,
        "cgroup_max_bytes": limit,
        "cgroup_current_bytes": current,
        "cgroup_available_bytes": max(0, limit - current) if limit is not None else None,
    }


def _allocated(info: os.stat_result) -> int:
    return int(getattr(info, "st_blocks", 0) * 512) or int(info.st_size)


def _tree_inventory(path: Path, *, seen_inodes: set[tuple[int, int]]) -> tuple[int, str]:
    if path.is_symlink():
        return 0, "symlink_not_followed"
    if not path.exists():
        return 0, "missing"
    total = 0
    try:
        info = path.stat(follow_symlinks=False)
        if stat.S_ISREG(info.st_mode):
            key = (int(info.st_dev), int(info.st_ino))
            if key not in seen_inodes:
                seen_inodes.add(key)
                total += _allocated(info)
            return total, "present"
        if not stat.S_ISDIR(info.st_mode):
            return 0, "unsupported_entry"
        for base, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(base) / name).is_symlink()]
            for name in files:
                item = Path(base) / name
                try:
                    entry = item.stat(follow_symlinks=False)
                except OSError:
                    continue
                if not stat.S_ISREG(entry.st_mode):
                    continue
                key = (int(entry.st_dev), int(entry.st_ino))
                if key not in seen_inodes:
                    seen_inodes.add(key)
                    total += _allocated(entry)
    except OSError:
        return total, "unreadable"
    return total, "present"


def _proc_path_reference(path: Path, *, proc_root: Path = Path("/proc")) -> bool | None:
    """Audit command, cwd, and open FDs; None means the scan was incomplete."""
    target = str(path.resolve())
    try:
        processes = tuple(proc_root.iterdir())
    except OSError:
        return None
    for process in processes:
        if not process.name.isdecimal():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="surrogateescape")
            cwd = os.readlink(process / "cwd")
        except FileNotFoundError:
            continue
        except OSError:
            return None
        if target in command or cwd == target or cwd.startswith(target + os.sep):
            return True
        try:
            descriptors = tuple((process / "fd").iterdir())
        except FileNotFoundError:
            continue
        except OSError:
            return None
        for descriptor in descriptors:
            try:
                opened = os.readlink(descriptor)
            except FileNotFoundError:
                continue
            except OSError:
                return None
            if opened == target or opened.startswith(target + os.sep):
                return True
    return False


def _inventory(runtime: Mapping[str, Any], deployment: Any, plan: Mapping[str, Any],
               packed_root: Path, packed: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cleanup: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def add(path: Path, *, category: str, kind: str, reference: str) -> None:
        size, state = _tree_inventory(path, seen_inodes=seen)
        rows.append({"category": category, "kind": kind, "path": str(path),
                     "allocated_bytes": size, "state": state,
                     "reference_status": reference})

    learner = runtime.get("learner") if isinstance(runtime.get("learner"), Mapping) else {}
    if isinstance(learner.get("recovery_dir"), str):
        add(Path(learner["recovery_dir"]).expanduser(), category="durable",
            kind="checkpoint", reference="retained_by_checkpoint_policy")
    bepa = runtime.get("bepa") if isinstance(runtime.get("bepa"), Mapping) else {}
    if bepa.get("enabled") is True:
        value = bepa.get("state_dir") or (Path(str(runtime.get("run_dir") or "")) / "bepa")
        add(Path(value).expanduser(), category="durable", kind="bepa_state",
            reference="configured_runtime_state")
    evaluation = runtime.get("evaluation") if isinstance(runtime.get("evaluation"), Mapping) else {}
    if evaluation.get("enabled") is True and isinstance(evaluation.get("state_dir"), str):
        add(Path(evaluation["state_dir"]).expanduser(), category="durable", kind="evaluation_state",
            reference="configured_runtime_state")

    packed_state = packed.get("state")
    add(packed_root, category="rebuildable", kind="packed_cache_root",
        reference=("current_cache_valid" if packed_state == "valid" else
                   "current_cache_missing" if packed_state == "missing" else
                   "cache_identity_unknown" if packed_state == "unclassified" else
                   "current_cache_partial"))
    for entry in packed.get("entries", []):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            continue
        entry_path = Path(entry["path"])
        is_partial = entry.get("state") == "partial"
        reference_result = _proc_path_reference(entry_path) if is_partial else None
        reference_status = (
            "proven_unreferenced" if reference_result is False else
            "referenced" if reference_result is True else
            "unknown" if is_partial else str(entry.get("reference_status") or "unknown")
        )
        allocated = entry.get("allocated_bytes", 0)
        rows.append({"category": "rebuildable", "kind": "packed_cache_entry",
                     "path": str(entry_path), "allocated_bytes": allocated,
                     "state": entry.get("state", "unknown"), "reference_status": reference_status,
                     "included_in_parent_inventory": True})
        if is_partial and reference_result is False:
            cleanup.append({"path": str(entry_path), "allocated_bytes": allocated,
                            "reason": "incomplete packed cache can be rebuilt from the immutable NInfer artifact",
                            "reference_status": "proven_unreferenced"})

    cache_root_value = getattr(deployment, "cache_root", None) or plan.get("cache_root")
    if cache_root_value:
        cache_root = Path(cache_root_value).expanduser()
        for name in _COMPILER_CACHE_NAMES:
            candidate = cache_root / name
            if candidate == packed_root or not candidate.exists():
                continue
            add(candidate, category="rebuildable", kind="compiler_cache",
                reference="preserve_across_restarts")

    transfer_path = plan.get("transfer_cache_dir")
    if isinstance(transfer_path, str) and transfer_path.strip():
        add(Path(transfer_path).expanduser(), category="rebuildable", kind="transfer_cache",
            reference="transfer_cache_contract_unknown")
    else:
        rows.append({"category": "rebuildable", "kind": "transfer_cache", "path": None,
                     "allocated_bytes": None, "state": "not_configured",
                     "reference_status": "not_configured_for_instance"})
    return rows, cleanup


def preflight_storage(deployment: Any, plan: Mapping[str, Any], runtime: Mapping[str, Any], *,
                      filesystem_fn: Callable[[Path], Mapping[str, Any]] | None = None,
                      memory_fn: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Return a read-only estimate and inventory for all known startup writes."""
    filesystem_fn = filesystem_fn or _filesystem_info
    memory_fn = memory_fn or _memory_capacity
    report: dict[str, Any] = {
        "schema": _SCHEMA,
        "status": "blocked",
        "scope": "cpu_metadata_only_no_model_load_no_cuda_no_model_payload_hash_no_cleanup",
        "packed_cache": None,
        "packed_cache_capacity": None,
        "checkpoint_temp_write": None,
        "filesystems": [],
        "inventory": [],
        "cleanup_plan": {"allowlisted_paths": [], "applied": False},
        "errors": [],
    }
    try:
        identity, directory = _artifact_layout(runtime)
        estimate = estimate_packed_layout(directory)
        executor_profiles = [row.get("compute_profile") for row in plan.get("executors", [])
                             if isinstance(row, Mapping) and isinstance(row.get("compute_profile"), Mapping)]
        store_paths = {profile.get("packed_weight_offload_store_dir") for profile in executor_profiles
                       if isinstance(profile.get("packed_weight_offload_store_dir"), str)}
        if len(store_paths) > 1:
            raise StoragePreflightError("planned executors disagree on the packed cache root")
        from .instance_plan import _default_packed_cache_root
        selected_root = plan.get("packed_cache_root")
        if selected_root is not None and (not isinstance(selected_root, str) or not selected_root.strip()):
            raise StoragePreflightError("planned packed cache root is malformed")
        if store_paths and selected_root is not None and str(next(iter(store_paths))) != selected_root:
            raise StoragePreflightError("planned packed cache roots disagree")
        if selected_root is not None:
            packed_root = Path(selected_root).expanduser()
        elif store_paths:
            packed_root = Path(next(iter(store_paths))).expanduser()
        elif getattr(deployment, "packed_cache_root", None) is not None:
            packed_root = Path(deployment.packed_cache_root).expanduser()
        else:
            packed_root = _default_packed_cache_root(deployment)
        if not packed_root.is_absolute():
            from ..paths import ROOT
            packed_root = ROOT / packed_root
        packed = _packed_cache_state(packed_root, identity, estimate)
        report["packed_cache"] = {**packed, "layout_estimate": estimate,
                                  "expected_bytes": estimate["alignment_upper_bound_bytes"]}
        if packed["state"] in {"partial", "unclassified"}:
            partial = packed["state"] == "partial"
            report["errors"].append({
                "reason_code": "packed_cache_partial" if partial else "packed_cache_identity_unknown",
                "field": "learner.packed_weight_offload_store_dir",
                "message": (
                    "The matching packed cache is incomplete and must be safely retired before startup."
                    if partial else
                    "A packed-cache entry lacks readable identity metadata, so startup capacity cannot be established."
                ),
            })

        checkpoint = _checkpoint_temp_write(runtime, plan)
        report["checkpoint_temp_write"] = checkpoint
        packed_filesystem = filesystem_fn(packed_root)
        operations = [
            {"kind": "packed_cache", "path": packed_root,
             "incremental_bytes": packed["incremental_bytes"], "filesystem": packed_filesystem},
            {"kind": "checkpoint_atomic_temp", "path": Path(checkpoint["path"]),
             "incremental_bytes": checkpoint["temporary_write_bytes"]},
        ]
        grouped: dict[str, dict[str, Any]] = {}
        for operation in operations:
            info = operation.get("filesystem") or filesystem_fn(Path(operation["path"]))
            fs_id = str(info.get("filesystem_id"))
            free = info.get("free_bytes")
            if not fs_id or type(free) is not int or free < 0:
                raise StoragePreflightError("filesystem free-space observation is invalid")
            row = grouped.setdefault(fs_id, {
                "filesystem_id": fs_id, "mount_path": str(info.get("mount_path") or "unknown"),
                "filesystem_type": info.get("filesystem_type"),
                "free_bytes": free, "incremental_bytes": 0, "peak_bytes": 0,
                "headroom_bytes": 0, "required_free_bytes": 0, "paths": [],
            })
            row["free_bytes"] = min(row["free_bytes"], free)
            amount = operation["incremental_bytes"]
            row["incremental_bytes"] += amount
            row["peak_bytes"] += amount
            row["paths"].append({"kind": operation["kind"], "path": str(operation["path"]),
                                 "incremental_bytes": amount})
        for row in grouped.values():
            row["headroom_bytes"] = max(
                _MIN_HEADROOM_BYTES,
                math.ceil(row["peak_bytes"] * _HEADROOM_FRACTION),
            )
            row["required_free_bytes"] = row["peak_bytes"] + row["headroom_bytes"]
            row["status"] = "passed" if row["free_bytes"] >= row["required_free_bytes"] else "blocked"
            if row["status"] == "blocked":
                report["errors"].append({
                    "reason_code": "insufficient_tmpfs_space"
                    if row.get("filesystem_type") == "tmpfs" else "insufficient_disk_space",
                    "field": ",".join(sorted({item["kind"] for item in row["paths"]})),
                    "message": "A filesystem has less free space than the estimated startup peak plus headroom.",
                })
            report["filesystems"].append(row)

        packed_filesystem_type = packed_filesystem.get("filesystem_type")
        default_shm_root = Path("/dev/shm").resolve(strict=False)
        packed_root_resolved = packed_root.resolve(strict=False)
        uses_default_shm_path = (packed_root_resolved == default_shm_root
                                 or default_shm_root in packed_root_resolved.parents)
        if uses_default_shm_path and packed_filesystem_type != "tmpfs":
            raise StoragePreflightError("the configured /dev/shm packed cache is not on a verified tmpfs mount")
        if packed_filesystem_type == "tmpfs":
            amount = packed["incremental_bytes"]
            packed_headroom = max(_MIN_HEADROOM_BYTES, math.ceil(amount * _HEADROOM_FRACTION))
            learner = runtime.get("learner") if isinstance(runtime.get("learner"), Mapping) else {}
            gradient_memory_limit = learner.get("gradient_memory_limit_bytes")
            if gradient_memory_limit is not None and (
                    type(gradient_memory_limit) is not int or gradient_memory_limit <= 0):
                raise StoragePreflightError("gradient contribution memory limit is invalid")
            gradient_scratch = (math.ceil(gradient_memory_limit * 0.20)
                                if gradient_memory_limit is not None else 0)
            required_memory = amount + packed_headroom + (gradient_memory_limit or 0) + gradient_scratch
            memory = dict(memory_fn())
            host_available = memory.get("host_available_bytes")
            cgroup_available = memory.get("cgroup_available_bytes")
            if "cgroup_available_bytes" not in memory:
                raise StoragePreflightError("cgroup memory availability is unavailable")
            if type(host_available) is not int or host_available < 0:
                raise StoragePreflightError("host MemAvailable observation is invalid")
            if cgroup_available is not None and (type(cgroup_available) is not int or cgroup_available < 0):
                raise StoragePreflightError("cgroup memory availability is invalid")
            cgroup_status = ("unlimited" if cgroup_available is None else
                             "passed" if cgroup_available >= required_memory else "blocked")
            host_status = "passed" if host_available >= required_memory else "blocked"
            packed_fs_row = next((row for row in report["filesystems"]
                                  if row["filesystem_id"] == packed_filesystem.get("filesystem_id")), None)
            filesystem_status = packed_fs_row.get("status") if packed_fs_row is not None else "unknown"
            memory_status = ("passed" if cgroup_status in {"passed", "unlimited"}
                             and host_status == "passed" else "blocked")
            capacity_status = ("passed" if memory_status == "passed" and filesystem_status == "passed"
                               else "blocked")
            report["packed_cache_capacity"] = {
                "status": capacity_status,
                "filesystem_space_status": filesystem_status,
                "filesystem_type": packed_filesystem_type,
                "mount_path": str(packed_filesystem.get("mount_path") or "unknown"),
                "free_bytes": packed_filesystem.get("free_bytes"),
                "incremental_bytes": amount,
                "packed_headroom_bytes": packed_headroom,
                "gradient_memory_limit_bytes": gradient_memory_limit,
                "gradient_memory_mode": "bounded" if gradient_memory_limit is not None else "unbounded",
                "gradient_scratch_allowance_bytes": gradient_scratch,
                "required_bytes": required_memory,
                "host_available_bytes": host_available,
                "host_memory_status": host_status,
                "cgroup_max_bytes": memory.get("cgroup_max_bytes"),
                "cgroup_current_bytes": memory.get("cgroup_current_bytes"),
                "cgroup_available_bytes": cgroup_available,
                "cgroup_memory_status": cgroup_status,
            }
            if memory_status == "blocked":
                report["errors"].append({
                    "reason_code": "insufficient_tmpfs_memory",
                    "field": "learner.packed_weight_offload_store_dir",
                    "message": "The packed tmpfs cache exceeds host or cgroup memory headroom.",
                })
        else:
            report["packed_cache_capacity"] = {
                "status": "not_applicable",
                "filesystem_type": packed_filesystem_type or "unknown",
                "mount_path": str(packed_filesystem.get("mount_path") or "unknown"),
            }

        inventory, cleanup = _inventory(runtime, deployment, plan, packed_root, packed)
        if checkpoint.get("source_observed") is True:
            inventory.append({
                "category": "durable_source_observation",
                "kind": "checkpoint_snapshot",
                "path": checkpoint.get("snapshot_path"),
                "allocated_bytes": checkpoint.get("observed_snapshot_bytes"),
                "state": "present",
                "reference_status": "source_preview_only_not_destination_capacity",
            })
        elif checkpoint.get("estimate_source") == "genesis_upper_bound_from_initial_adapter_header":
            inventory.append({
                "category": "configured_input",
                "kind": "genesis_initial_adapter",
                "path": checkpoint.get("initial_adapter_path"),
                "allocated_bytes": checkpoint.get("initial_adapter_file_bytes"),
                "state": "present",
                "reference_status": "configured_genesis_source",
            })
        report["inventory"] = inventory
        report["cleanup_plan"]["allowlisted_paths"] = cleanup
        report["status"] = "blocked" if report["errors"] else "passed"
        return report
    except StoragePreflightError as error:
        report["errors"].append({"reason_code": "storage_capacity_unknown", "field": "storage",
                                 "message": str(error)})
        return report
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        report["errors"].append({"reason_code": "storage_capacity_unknown", "field": "storage",
                                 "message": f"Storage metadata could not be read ({type(error).__name__})."})
        return report
