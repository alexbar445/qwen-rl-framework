"""Worker-side attestations for the explicitly selected base artifact format.

This module stays stdlib-only so artifact identity can be checked before the
vLLM runtime imports Torch or initializes a CUDA device.
"""
from __future__ import annotations

import importlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_MODES = frozenset({"torchao"})
_TORCHAO_QUANTIZER = "NVFP4WeightOnlyConfig"
_NVFP4_COMPUTE_DTYPE = "bfloat16"
_NVFP4_ARTIFACT_COMPUTE_DTYPE = "torch.bfloat16"
_TERNARY_MODE = "ternary"
_TERNARY_BASE_WEIGHT_PROFILE = "t2_g128_fp16_rotated"
_CODER390_MODEL_ID = "coder390-t2-full-scales-step512"
_CODER390_BASE_ARTIFACT_KIND = "coder390_t2_full_scales_step512"
_CODER390_BASE_WEIGHT_PROFILE = "t2_g128_fp16_per_use"
_BONSAI2_MODEL_ID = "bonsai2-27b-arc-k2-mtp"
_TERNARY_RUNTIME_PROFILES = {
    _BONSAI2_MODEL_ID: ("bonsai2_t2", "prism_rotated", _TERNARY_BASE_WEIGHT_PROFILE, 323, 515),
    _CODER390_MODEL_ID: (
        _CODER390_BASE_ARTIFACT_KIND, "per_use", _CODER390_BASE_WEIGHT_PROFILE, 322, 514,
    ),
}
_TERNARY_RUNTIME_FORMAT_SETS = (
    ("t2_g128_fp16",),
    ("q8_g32_fp16", "t2_g128_fp16"),
)


class WorkerArtifactError(ValueError):
    """The configured Worker mode and immutable model artifact do not agree."""


def ternary_runtime_profile(model_id: str) -> tuple[str, str, str]:
    """Return the exact native T2 kind, input basis, and weight profile for a known model ID."""
    if not isinstance(model_id, str):
        raise WorkerArtifactError("native T2 model identity must be a string")
    profile = _TERNARY_RUNTIME_PROFILES.get(model_id)
    if profile is None:
        raise WorkerArtifactError(f"unsupported native T2 model identity: {model_id!r}")
    return profile[:3]


def validate_ternary_runtime_profile(
    health: dict[str, Any], *, model_id: str | None = None,
) -> tuple[str, str, str]:
    """Check a Worker/runtime identity and its source quantization against the known T2 profiles."""
    if not isinstance(health, dict):
        raise WorkerArtifactError("native T2 runtime profile evidence must be an object")
    if model_id is not None and not isinstance(model_id, str):
        raise WorkerArtifactError("native T2 expected model identity must be a string")
    if model_id is None:
        model_id = next(
            (candidate for candidate, profile in _TERNARY_RUNTIME_PROFILES.items()
             if profile[0] == health.get("base_artifact_kind")),
            None,
        )
    profile = _TERNARY_RUNTIME_PROFILES.get(model_id) if isinstance(model_id, str) else None
    if profile is None:
        raise WorkerArtifactError("native T2 runtime profile names an unsupported model")
    kind, base_basis, weight_profile, t2_object_count, t2_logical_parameter_count = profile
    gdn_enabled = health.get("gdn_projections", False)
    if type(gdn_enabled) is not bool:
        raise WorkerArtifactError("native T2 runtime GDN projection state must be a boolean")
    adapter_basis = "per_use" if gdn_enabled else base_basis
    identity = health.get("artifact_identity")
    if identity is not None:
        try:
            from ..contracts.lora_targets import gdn_lora_enabled, projection_input_bases

            identity_gdn = gdn_lora_enabled(identity)
            learner = identity.get("learner", {})
            adapter = learner.get("adapter", {})
            rotation = identity.get("rotation", {})
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise WorkerArtifactError("native T2 runtime artifact identity is invalid") from exc
        if identity_gdn is not gdn_enabled:
            raise WorkerArtifactError("native T2 runtime GDN state differs from its artifact identity")
        if model_id == _BONSAI2_MODEL_ID:
            if rotation.get("input_basis") != "prism_rotated":
                raise WorkerArtifactError("Bonsai2 frozen T2 rotation identity changed")
        elif rotation.get("input_basis") != "per_use":
            raise WorkerArtifactError("Coder390 frozen T2 rotation identity changed")
        if adapter.get("input_basis") != adapter_basis:
            raise WorkerArtifactError("native T2 adapter input basis differs from its identity")
        module_basis = adapter.get("input_basis_by_module")
        if adapter_basis == "per_use":
            try:
                expected_module_basis = projection_input_bases(identity)
            except (KeyError, TypeError, ValueError) as exc:
                raise WorkerArtifactError("native T2 per-use projection basis map is invalid") from exc
            if not isinstance(module_basis, dict) or module_basis != expected_module_basis:
                raise WorkerArtifactError("native T2 per-use projection basis map differs from its identity")
    elif gdn_enabled:
        raise WorkerArtifactError("all-GDN native T2 runtime omitted its artifact identity")
    if (health.get("base_artifact_kind") != kind
            or health.get("input_basis") != adapter_basis
            or health.get("base_weight_profile") != weight_profile):
        raise WorkerArtifactError("native T2 runtime profile differs from the expected model identity")
    quantization = health.get("quantization") or health.get("source_quantization")
    if (not isinstance(quantization, dict)
            or quantization.get("mode") != _TERNARY_MODE
            or quantization.get("lora_input_basis") != adapter_basis
            or quantization.get("t2_object_count") != t2_object_count
            or quantization.get("t2_logical_parameter_count") != t2_logical_parameter_count):
        raise WorkerArtifactError("native T2 source quantization differs from its model profile")
    return kind, adapter_basis, weight_profile


@dataclass(frozen=True)
class WorkerArtifactEvidence:
    model_path: str
    mode: str
    config: dict[str, Any]
    base_fingerprint: str
    artifact_fingerprint: str
    source_fingerprint: dict[str, Any]
    artifact_manifest: dict[str, Any]
    quantization: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_config": self.config,
            "base_fingerprint": self.base_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "artifact_manifest": self.artifact_manifest,
            "quantization": self.quantization,
        }


@dataclass(frozen=True)
class TernaryWorkerArtifactEvidence:
    """Identity-bound native T2 base evidence, separate from HF Q4 lineage."""

    model_path: str
    ninfer_artifact_path: str
    conversion_report_path: str
    base_fingerprint: str
    artifact_fingerprint: str
    ninfer_artifact_id: str
    ninfer_artifact_sha256: str
    source_report_sha256: str
    source_report_kind: str
    base_artifact_kind: str
    gdn_projections: bool
    input_basis: str
    base_weight_profile: str
    artifact_identity: dict[str, Any]
    quantization: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "ninfer_artifact_path": self.ninfer_artifact_path,
            "ninfer_conversion_report": self.conversion_report_path,
            "base_fingerprint": self.base_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "ninfer_artifact_id": self.ninfer_artifact_id,
            "ninfer_artifact_sha256": self.ninfer_artifact_sha256,
            "ninfer_artifact_sha256_verified": False,
            "source_report_sha256": self.source_report_sha256,
            "source_report_kind": self.source_report_kind,
            "base_artifact_kind": self.base_artifact_kind,
            "gdn_projections": self.gdn_projections,
            "input_basis": self.input_basis,
            "base_weight_profile": self.base_weight_profile,
            "artifact_identity": self.artifact_identity,
            "quantization": self.quantization,
        }


def artifact_lineage(evidence: WorkerArtifactEvidence) -> dict[str, Any]:
    """Project the common source and packed-artifact identity from CPU-only evidence."""
    manifest = evidence.artifact_manifest
    quantization = evidence.quantization
    weight_count = quantization.get("packed_module_count")
    result = {
        **evidence.source_fingerprint,
        "model_path": evidence.model_path,
        "base_fingerprint": evidence.base_fingerprint,
        "artifact_fingerprint": evidence.artifact_fingerprint,
        "quantization_method": evidence.mode,
        "source_fingerprint": evidence.source_fingerprint,
        "artifact_manifest": manifest,
        "quantization": quantization,
        "quantized_weight_count": weight_count,
    }
    for key in ("packed_module_count", "packed_uint8_tensor_count", "packed_uint8_numel"):
        if key in quantization:
            result[key] = quantization[key]
    return result


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise WorkerArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _expected_nvfp4_skips(config: dict[str, Any]) -> list[str]:
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        raise WorkerArtifactError("TorchAO NVFP4 Worker artifact must declare a Qwen3.5 architecture")
    if "Qwen3_5ForCausalLM" in architectures and config.get("model_type") == "qwen3_5_text":
        text_config = config
        hf_layer_prefix = "model.layers"
    elif (
        "Qwen3_5ForConditionalGeneration" in architectures
        and config.get("model_type") == "qwen3_5"
        and isinstance(config.get("text_config"), dict)
        and config["text_config"].get("model_type") == "qwen3_5_text"
    ):
        text_config = config["text_config"]
        hf_layer_prefix = "model.language_model.layers"
    else:
        raise WorkerArtifactError(
            "TorchAO NVFP4 Worker artifact must declare Qwen3_5ForCausalLM "
            "or Qwen3_5ForConditionalGeneration with a Qwen3.5 text config"
        )
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        raise WorkerArtifactError("TorchAO NVFP4 config must declare layer_types")
    skips: list[str] = []
    for index, layer_type in enumerate(layer_types):
        if layer_type == "linear_attention":
            vllm_prefix = f"model.layers.{index}.linear_attn"
            skips.extend(
                (
                    f"{hf_layer_prefix}.{index}.linear_attn.in_proj_a",
                    f"{hf_layer_prefix}.{index}.linear_attn.in_proj_b",
                    # vLLM fuses the two projections in this module for either
                    # supported Qwen3.5 architecture.
                    f"{vllm_prefix}.in_proj_ba",
                )
            )
        elif layer_type != "full_attention":
            raise WorkerArtifactError(f"unsupported Qwen3.5 layer type: {layer_type!r}")
    if not skips:
        raise WorkerArtifactError("TorchAO NVFP4 Worker artifact has no GDN layers")
    skips.append("lm_head")
    return skips


def _validate_skip_metadata(
    config: dict[str, Any],
    lineage: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    expected = _expected_nvfp4_skips(config)
    quantization_config = config.get("quantization_config")
    config_skips = (
        quantization_config.get("modules_to_not_convert")
        if isinstance(quantization_config, dict)
        else None
    )
    lineage_skips = lineage.get("modules_to_not_convert")
    manifest_skips = manifest.get("modules_to_not_convert")
    for name, actual in (
        ("config.json", config_skips),
        ("source_fingerprint.json", lineage_skips),
        ("artifact manifest", manifest_skips),
    ):
        if not isinstance(actual, list) or actual != expected:
            raise WorkerArtifactError(
                f"{name} modules_to_not_convert differs from the exact "
                f"Qwen3.5 GDN/head recipe: expected {expected!r}, got {actual!r}"
            )
    return expected


def _validate_packed_inventory(manifest: dict[str, Any]) -> dict[str, int]:
    inventory = manifest.get("packed_tensor_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise WorkerArtifactError("TorchAO artifact has no packed tensor inventory")
    weight_files = manifest.get("weight_files")
    if not isinstance(weight_files, list) or not weight_files:
        raise WorkerArtifactError("TorchAO artifact has no weight shard inventory")
    shard_paths = {
        record.get("path")
        for record in weight_files
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    names: set[str] = set()
    qdata_keys: set[str] = set()
    qdata_numel = 0
    for record in inventory:
        if not isinstance(record, dict):
            raise WorkerArtifactError("packed tensor inventory entries must be objects")
        name = record.get("name")
        qdata_key = record.get("qdata_key")
        shard_path = record.get("shard_path")
        if not isinstance(name, str) or not name or name in names:
            raise WorkerArtifactError("packed tensor inventory has an invalid or duplicate module name")
        if not isinstance(qdata_key, str) or not qdata_key or qdata_key in qdata_keys:
            raise WorkerArtifactError("packed tensor inventory has an invalid or duplicate qdata key")
        if not isinstance(shard_path, str) or shard_path not in shard_paths:
            raise WorkerArtifactError(f"packed tensor {name} references an unknown shard")
        if record.get("qdata_dtype") != "U8":
            raise WorkerArtifactError(f"packed tensor {name} is not stored as uint8 qdata")
        shape = record.get("qdata_shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape)
        ):
            raise WorkerArtifactError(f"packed tensor {name} has an invalid qdata shape")
        numel = record.get("qdata_numel")
        if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
            raise WorkerArtifactError(f"packed tensor {name} has an invalid qdata numel")
        if math.prod(shape) != numel:
            raise WorkerArtifactError(f"packed tensor {name} qdata shape and numel disagree")
        scale_keys = record.get("scales_keys")
        if (
            not isinstance(scale_keys, list)
            or not scale_keys
            or any(not isinstance(key, str) or not key for key in scale_keys)
        ):
            raise WorkerArtifactError(f"packed tensor {name} has no scale tensor inventory")
        if not isinstance(record.get("orig_dtype"), str) or not record["orig_dtype"]:
            raise WorkerArtifactError(f"packed tensor {name} has no original dtype")
        names.add(name)
        qdata_keys.add(qdata_key)
        qdata_numel += numel

    counts = {
        "packed_module_count": len(names),
        "packed_uint8_tensor_count": len(qdata_keys),
        "packed_uint8_numel": qdata_numel,
    }
    for key, actual in counts.items():
        declared = manifest.get(key)
        if isinstance(declared, bool) or not isinstance(declared, int) or declared != actual:
            raise WorkerArtifactError(
                f"TorchAO artifact {key} disagrees with packed tensor inventory"
            )
    return counts


def _canonical_quantization(
    mode: str,
    config: dict[str, Any],
    lineage: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise WorkerArtifactError("model config has no quantization_config")
    config_method = quantization_config.get("quant_method")
    lineage_method = lineage.get("quantization_method")
    if config_method != mode or lineage_method != mode:
        raise WorkerArtifactError(
            "configured quantization mode differs from model config or source lineage"
        )
    quantization = {
        "mode": mode,
        "bits": 4,
        "packed_resident": True,
        "compute_dtype": _NVFP4_ARTIFACT_COMPUTE_DTYPE,
    }
    if lineage.get("quantizer") != _TORCHAO_QUANTIZER:
        raise WorkerArtifactError("TorchAO lineage must name NVFP4WeightOnlyConfig")
    quant_type = quantization_config.get("quant_type")
    default_type = quant_type.get("default") if isinstance(quant_type, dict) else None
    if not isinstance(default_type, dict) or default_type.get("_type") != _TORCHAO_QUANTIZER:
        raise WorkerArtifactError("model config must declare TorchAO NVFP4WeightOnlyConfig")
    skips = _validate_skip_metadata(config, lineage, manifest)
    counts = _validate_packed_inventory(manifest)
    quantization.update(
        {
            "quantizer": _TORCHAO_QUANTIZER,
            "modules_to_not_convert": skips,
            **counts,
        }
    )
    return quantization, counts


def read_worker_artifact(
    model_path: str | Path,
    configured_mode: str,
) -> WorkerArtifactEvidence:
    """Read and attest only the artifact format explicitly selected by config."""
    if not isinstance(configured_mode, str) or configured_mode not in _SUPPORTED_MODES:
        raise WorkerArtifactError("vLLM Worker quantization must explicitly be torchao NVFP4")
    requested = Path(model_path).expanduser()
    if requested.is_symlink():
        raise WorkerArtifactError("Worker model_path cannot be a symlink")
    root = requested.resolve()
    if not root.is_dir():
        raise WorkerArtifactError(f"Worker model directory does not exist: {root}")

    try:
        module = importlib.import_module("qwen_rl.models.artifact")
        reader = getattr(module, "read_nvfp4_artifact")
        config, base_fingerprint, identity = reader(root)
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        raise WorkerArtifactError(str(exc)) from exc
    if not isinstance(config, dict) or not isinstance(identity, dict):
        raise WorkerArtifactError("artifact reader returned malformed evidence")
    _require_sha256(base_fingerprint, "base_fingerprint")
    artifact_fingerprint = _require_sha256(
        identity.get("artifact_fingerprint"), "artifact_fingerprint"
    )
    lineage = identity.get("source_fingerprint")
    manifest = identity.get("artifact_manifest")
    if not isinstance(lineage, dict) or not isinstance(manifest, dict):
        raise WorkerArtifactError("artifact reader omitted source lineage or content manifest")
    if lineage.get("base_fingerprint") != base_fingerprint:
        raise WorkerArtifactError("source lineage base_fingerprint differs from artifact identity")
    if manifest.get("source_fingerprint") != base_fingerprint:
        raise WorkerArtifactError("artifact content manifest differs from source base_fingerprint")
    quantization, counts = _canonical_quantization(
        configured_mode, config, lineage, manifest
    )
    return WorkerArtifactEvidence(
        model_path=str(root),
        mode=configured_mode,
        config=config,
        base_fingerprint=base_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        source_fingerprint=lineage,
        artifact_manifest=manifest,
        quantization=quantization,
    )


def read_ternary_worker_artifact(
    ninfer_path: str | Path,
    conversion_report_path: str | Path,
    *, declared_artifact_sha256: str | None = None,
    artifact_identity_path: str | Path | None = None,
    expected_lora_rank: int | None = None,
    expected_lora_alpha: int | None = None,
) -> TernaryWorkerArtifactEvidence:
    """Attest one frozen Bonsai2 T2 `.ninfer` artifact through its strict identity contract."""
    artifact_path = Path(ninfer_path).expanduser()
    report_path = Path(conversion_report_path).expanduser()
    for path, label in ((artifact_path, "T2 NInfer artifact"), (report_path, "T2 conversion report")):
        if path.is_symlink():
            raise WorkerArtifactError(f"{label} cannot be a symlink")
        if not path.is_file():
            raise WorkerArtifactError(f"{label} must be a regular file")
    artifact_path = artifact_path.resolve(strict=True)
    report_path = report_path.resolve(strict=True)
    if artifact_path.suffix != ".ninfer":
        raise WorkerArtifactError("T2 NInfer artifact path must end in .ninfer")
    try:
        from qwen_rl.contracts.ternary import (
            SUPPORTED_T2_LORA_PROFILES,
            TernaryContractError,
            build_artifact_identity,
            load_pinned_artifact_identity,
        )

        if (expected_lora_rank is None) != (expected_lora_alpha is None):
            raise WorkerArtifactError("expected T2 LoRA rank and alpha must be provided together")
        expected_profile = None
        if expected_lora_rank is not None:
            expected_profile = (expected_lora_rank, expected_lora_alpha)
            if (type(expected_lora_rank) is not int or type(expected_lora_alpha) is not int
                    or expected_profile not in SUPPORTED_T2_LORA_PROFILES):
                raise WorkerArtifactError("expected T2 LoRA profile must be rank/alpha 16/16 or 32/32")
        if artifact_identity_path is None:
            if expected_profile not in (None, (16, 16)):
                raise WorkerArtifactError("rank-32 T2 Worker requires an explicit artifact identity path")
            identity = build_artifact_identity(
                artifact_path,
                report_path,
                declared_artifact_sha256=declared_artifact_sha256,
            )
        else:
            identity_path = Path(artifact_identity_path).expanduser()
            if identity_path.is_symlink() or not identity_path.is_file():
                raise WorkerArtifactError("T2 artifact identity must be a regular file, not a symlink")
            identity = load_pinned_artifact_identity(identity_path.resolve(strict=True)).as_dict()
        if expected_profile is not None:
            adapter_profile = identity.get("learner", {}).get("adapter", {})
            if (adapter_profile.get("rank"), adapter_profile.get("alpha")) != expected_profile:
                raise WorkerArtifactError("T2 artifact identity rank/alpha differs from Worker configuration")
    except (OSError, ValueError, TernaryContractError) as exc:
        raise WorkerArtifactError(f"Bonsai2 T2 artifact identity is invalid: {exc}") from exc

    source = identity.get("source")
    artifact = source.get("ninfer_artifact") if isinstance(source, dict) else None
    report = (
        source.get("conversion_report") or source.get("artifact_report")
        if isinstance(source, dict) else None
    )
    representation = identity.get("representation")
    rotation = identity.get("rotation")
    adapter = identity.get("learner", {}).get("adapter") if isinstance(identity.get("learner"), dict) else None
    model = identity.get("model")
    if not isinstance(artifact, dict) or not isinstance(report, dict):
        raise WorkerArtifactError("T2 identity omitted artifact or source report")
    if artifact.get("path") != str(artifact_path) or report.get("path") != str(report_path):
        raise WorkerArtifactError("T2 identity paths differ from the configured Worker inputs")
    artifact_sha = _require_sha256(artifact.get("sha256"), "T2 NInfer artifact SHA-256")
    if declared_artifact_sha256 is not None and artifact_sha != declared_artifact_sha256:
        raise WorkerArtifactError("T2 identity base SHA differs from the deployment declaration")
    report_sha = _require_sha256(report.get("sha256"), "T2 source report SHA-256")
    artifact_id = artifact.get("artifact_id")
    if not isinstance(artifact_id, str) or re.fullmatch(r"[0-9a-f]{32}", artifact_id) is None:
        raise WorkerArtifactError("T2 identity has an invalid NInfer artifact ID")
    if not isinstance(representation, dict) or representation.get("format") != "t2_g128_fp16":
        raise WorkerArtifactError("T2 identity does not prove the T2 G128 representation")
    model_id = model.get("id") if isinstance(model, dict) else None
    input_basis = rotation.get("input_basis") if isinstance(rotation, dict) else None
    adapter_basis = adapter.get("input_basis") if isinstance(adapter, dict) else None
    try:
        from ..contracts.lora_targets import gdn_lora_enabled, projection_input_bases

        gdn_projections = gdn_lora_enabled(identity)
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerArtifactError(f"T2 identity has an invalid GDN LoRA declaration: {exc}") from exc
    if model_id == _BONSAI2_MODEL_ID:
        if input_basis != "prism_rotated":
            raise WorkerArtifactError("Bonsai2 frozen T2 Uses must retain prism_rotated inputs")
        if gdn_projections:
            module_basis = adapter.get("input_basis_by_module") if isinstance(adapter, dict) else None
            try:
                expected_module_basis = projection_input_bases(identity)
            except (KeyError, TypeError, ValueError) as exc:
                raise WorkerArtifactError("Bonsai2 all-GDN LoRA basis map is invalid") from exc
            if (adapter_basis != "per_use" or not isinstance(module_basis, dict)
                    or module_basis != expected_module_basis):
                raise WorkerArtifactError(
                    "Bonsai2 all-GDN LoRA identity must bind all projection input bases"
                )
        elif adapter_basis != "prism_rotated":
            raise WorkerArtifactError("Bonsai2 default LoRA identity must retain prism_rotated inputs")
        base_artifact_kind = "bonsai2_t2"
        base_weight_profile = _TERNARY_BASE_WEIGHT_PROFILE
    elif model_id == _CODER390_MODEL_ID:
        module_basis = adapter.get("input_basis_by_module") if isinstance(adapter, dict) else None
        try:
            expected_module_basis = projection_input_bases(identity)
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkerArtifactError("Coder390 T2 LoRA basis map is invalid") from exc
        if (input_basis != "per_use" or adapter_basis != "per_use"
                or not isinstance(module_basis, dict) or module_basis != expected_module_basis):
            raise WorkerArtifactError("Coder390 T2 identity must bind each LoRA target input basis")
        base_artifact_kind = _CODER390_BASE_ARTIFACT_KIND
        base_weight_profile = _CODER390_BASE_WEIGHT_PROFILE
    else:
        raise WorkerArtifactError("T2 identity names an unsupported model profile")

    quantization = {
        "mode": _TERNARY_MODE,
        "bits": 2,
        "packed_resident": True,
        "compute_dtype": "bfloat16",
        "format": representation["format"],
        "layout": representation["layout"],
        "group_size_k": representation["group_size_k"],
        "t2_object_count": representation["t2_object_count"],
        "t2_logical_parameter_count": representation["t2_logical_parameter_count"],
        "artifact_id": artifact_id,
        "artifact_sha256": artifact_sha,
        "artifact_sha256_verified": False,
        "lora_input_basis": adapter_basis,
    }
    if gdn_projections:
        quantization["gdn_projections"] = True
    # Bonsai2's source and runnable artifact identities are both the immutable
    # .ninfer content digest. The raw GGUF is separately marked verified or
    # missing by the identity contract and is never promoted to this value.
    return TernaryWorkerArtifactEvidence(
        model_path=str(artifact_path),
        ninfer_artifact_path=str(artifact_path),
        conversion_report_path=str(report_path),
        base_fingerprint=artifact_sha,
        artifact_fingerprint=artifact_sha,
        ninfer_artifact_id=artifact_id,
        ninfer_artifact_sha256=artifact_sha,
        source_report_sha256=report_sha,
        source_report_kind=report.get("report_kind", "ninfer_conversion_report"),
        base_artifact_kind=base_artifact_kind,
        gdn_projections=gdn_projections,
        input_basis=adapter_basis,
        base_weight_profile=base_weight_profile,
        artifact_identity=identity,
        quantization=quantization,
    )


def validate_ternary_runtime_quantization(value: Any) -> dict[str, Any]:
    """Accept only the loaded T2 route, optionally with its active Q8 MTP weights."""
    if not isinstance(value, dict):
        raise WorkerArtifactError("NInfer T2 runtime_quantization must be an object")
    formats = value.get("formats")
    if (not isinstance(formats, list)
            or tuple(formats) not in _TERNARY_RUNTIME_FORMAT_SETS
            or value.get("representation") != "native_t2"):
        raise WorkerArtifactError(
            "NInfer T2 runtime must use native_t2 with T2 and optional Q8 formats only"
        )
    return dict(value)


def _validate_quantization_payload(
    configured_mode: str,
    evidence: dict[str, Any],
    source_name: str,
    *,
    require_skip_modules: bool,
) -> dict[str, Any]:
    if configured_mode == _TERNARY_MODE:
        if not isinstance(evidence, dict):
            raise WorkerArtifactError(f"{source_name} quantization evidence must be an object")
        required = (
            "mode", "bits", "packed_resident", "compute_dtype", "format", "layout",
            "group_size_k", "t2_object_count", "t2_logical_parameter_count",
            "artifact_id", "artifact_sha256", "artifact_sha256_verified", "lora_input_basis",
        )
        if evidence.get("mode") != _TERNARY_MODE:
            raise WorkerArtifactError(f"{source_name} quantization mode differs from explicit Worker config")
        if any(key not in evidence for key in required):
            raise WorkerArtifactError(f"{source_name} T2 quantization contract is incomplete")
        t2_object_count = evidence.get("t2_object_count")
        t2_logical_parameter_count = evidence.get("t2_logical_parameter_count")
        if (evidence.get("bits") != 2 or isinstance(evidence.get("bits"), bool)
                or evidence.get("packed_resident") is not True
                or evidence.get("compute_dtype") != "bfloat16"
                or evidence.get("format") != "t2_g128_fp16"
                or evidence.get("layout") != "row_split_k128_v1"
                or evidence.get("group_size_k") != 128
                or isinstance(t2_object_count, bool) or not isinstance(t2_object_count, int)
                or t2_object_count <= 0
                or isinstance(t2_logical_parameter_count, bool)
                or not isinstance(t2_logical_parameter_count, int)
                or t2_logical_parameter_count <= 0
                or evidence.get("lora_input_basis") not in {"prism_rotated", "per_use"}):
            raise WorkerArtifactError(f"{source_name} T2 quantization contract is invalid")
        gdn_projections = evidence.get("gdn_projections", False)
        if type(gdn_projections) is not bool:
            raise WorkerArtifactError(f"{source_name} GDN projection state must be a boolean")
        if (not isinstance(evidence.get("artifact_id"), str)
                or re.fullmatch(r"[0-9a-f]{32}", evidence["artifact_id"]) is None):
            raise WorkerArtifactError(f"{source_name} T2 artifact ID is invalid")
        _require_sha256(evidence.get("artifact_sha256"), f"{source_name} declared T2 artifact SHA-256")
        if evidence.get("artifact_sha256_verified") is not False:
            raise WorkerArtifactError(f"{source_name} T2 artifact SHA verification state is invalid")
        normalized = {key: evidence[key] for key in required}
        normalized["gdn_projections"] = gdn_projections
        return normalized
    if not isinstance(configured_mode, str) or configured_mode not in _SUPPORTED_MODES:
        raise WorkerArtifactError("Worker quantization must explicitly be torchao or ternary")
    if not isinstance(evidence, dict):
        raise WorkerArtifactError(f"{source_name} quantization evidence must be an object")
    expected_compute_dtype = _NVFP4_COMPUTE_DTYPE

    required = ("mode", "bits", "packed_resident", "compute_dtype")
    if evidence.get("mode") != configured_mode:
        raise WorkerArtifactError(
            f"{source_name} quantization mode differs from explicit Worker config"
        )
    bits = evidence.get("bits")
    if (
        isinstance(bits, bool)
        or not isinstance(bits, int)
        or bits != 4
        or evidence.get("packed_resident") is not True
        or evidence.get("compute_dtype") != expected_compute_dtype
    ):
        raise WorkerArtifactError(f"{source_name} quantization contract is invalid")
    if any(key not in evidence for key in required):
        raise WorkerArtifactError(f"{source_name} quantization contract is incomplete")
    if evidence.get("quantizer") != _TORCHAO_QUANTIZER:
        raise WorkerArtifactError(
            f"{source_name} must name NVFP4WeightOnlyConfig explicitly"
        )
    skips = evidence.get("modules_to_not_convert")
    if require_skip_modules and (
        not isinstance(skips, list)
        or not skips
        or len(skips) != len(set(skips))
        or any(not isinstance(item, str) or not item for item in skips)
    ):
        raise WorkerArtifactError(
            f"{source_name} has no valid exact modules_to_not_convert"
        )
    return {key: evidence[key] for key in required}


def validate_worker_quantization(
    configured_mode: str,
    worker_quantization: dict[str, Any],
) -> dict[str, Any]:
    """Require loaded Worker evidence to match its explicit inference setting."""
    return _validate_quantization_payload(
        configured_mode,
        worker_quantization,
        "Worker artifact",
        require_skip_modules=True,
    )


def validate_policy_quantization(
    configured_mode: str,
    worker_quantization: dict[str, Any],
    policy_quantization: dict[str, Any],
) -> dict[str, Any]:
    """Require the policy manifest and loaded base to name the same explicit recipe."""
    worker_contract = validate_worker_quantization(
        configured_mode, worker_quantization
    )
    policy_contract = _validate_quantization_payload(
        configured_mode,
        policy_quantization,
        "policy manifest",
        require_skip_modules=False,
    )
    if configured_mode == "torchao":
        worker_skips = worker_quantization["modules_to_not_convert"]
        policy_skips = policy_quantization.get("modules_to_not_convert")
        if policy_skips is not None and policy_skips != worker_skips:
            raise WorkerArtifactError(
                "policy modules_to_not_convert differs from loaded Worker artifact"
            )
    if worker_contract != policy_contract:
        raise WorkerArtifactError(
            "policy quantization contract differs from loaded Worker artifact"
        )
    return worker_contract
