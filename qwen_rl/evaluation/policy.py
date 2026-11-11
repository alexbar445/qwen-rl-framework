"""Evaluation policy identities and CPU-only zero-residual SFT adapter export."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..rollout.groups import PolicySnapshot
from ..runtime.manifest import write_json_atomic


BASELINE_POLICY_VERSION = 0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_LORA_KEY = re.compile(r"^(?P<prefix>.+)\.lora_(?P<side>A|B)\.weight$")


def policy_quantization_mode(quantization: Any) -> str:
    """Reject unsupported policy formats before archive or baseline publication."""
    if not isinstance(quantization, dict):
        raise ValueError("policy quantization must be an object")
    mode = quantization.get("mode")
    expected_bits = {"torchao": 4, "ternary": 2}
    if (not isinstance(mode, str) or mode not in expected_bits
            or type(quantization.get("bits")) is not int
            or quantization["bits"] != expected_bits[mode]):
        raise ValueError("policy quantization must be NVFP4 torchao or T2 ternary")
    return mode


@dataclass(frozen=True)
class EvaluationPolicy:
    role: Literal["baseline_sft", "rl_checkpoint"]
    snapshot: PolicySnapshot
    source: str
    policy_archive_ref: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"baseline_sft", "rl_checkpoint"}:
            raise ValueError("evaluation policy role must be baseline_sft or rl_checkpoint")
        if not isinstance(self.snapshot, PolicySnapshot):
            raise TypeError("evaluation policy requires a PolicySnapshot")
        if (isinstance(self.snapshot.version, bool) or not isinstance(self.snapshot.version, int)
                or self.snapshot.version < 0):
            raise ValueError("evaluation policy version must be a nonnegative integer")
        if not isinstance(self.snapshot.adapter_sha256, str) or not _SHA256.fullmatch(self.snapshot.adapter_sha256):
            raise ValueError("evaluation policy adapter SHA must be lowercase SHA-256")
        if (not isinstance(self.snapshot.base_fingerprint, str) or not self.snapshot.base_fingerprint
                or not isinstance(self.snapshot.artifact_fingerprint, str)
                or not _SHA256.fullmatch(self.snapshot.artifact_fingerprint)):
            raise ValueError("evaluation policy requires valid base and artifact fingerprints")
        if self.role == "baseline_sft" and self.snapshot.version != BASELINE_POLICY_VERSION:
            raise ValueError("SFT baseline evaluation policy must use version 0")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("evaluation policy source must be non-empty")
        if self.policy_archive_ref is not None and (
            not isinstance(self.policy_archive_ref, str) or not self.policy_archive_ref
        ):
            raise ValueError("policy_archive_ref must be a non-empty string or None")

    @property
    def version(self) -> int:
        return self.snapshot.version

    @property
    def adapter_sha256(self) -> str:
        return self.snapshot.adapter_sha256

    @property
    def base_fingerprint(self) -> str:
        return self.snapshot.base_fingerprint

    @property
    def artifact_fingerprint(self) -> str:
        return self.snapshot.artifact_fingerprint

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "version": self.version,
            "adapter_sha256": self.adapter_sha256,
            "base_fingerprint": self.base_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "source": self.source,
            "policy_archive_ref": self.policy_archive_ref,
        }


@dataclass(frozen=True)
class ExportedEvaluationPolicy:
    policy: EvaluationPolicy
    directory: Path
    publication: dict[str, Any]
    provenance_path: Path


def _verify_standard_lora_config(config: dict[str, Any]) -> None:
    if config.get("peft_type") != "LORA":
        raise ValueError("baseline export requires a standard PEFT LORA adapter")
    if config.get("bias") != "none":
        raise ValueError("baseline export requires bias='none'")
    if config.get("use_dora") is not False:
        raise ValueError("baseline export requires use_dora=false")
    if config.get("lora_bias") is not False:
        raise ValueError("baseline export requires lora_bias=false")
    for name in ("modules_to_save", "trainable_token_indices", "target_parameters", "layer_replication"):
        if config.get(name) not in (None, [], {}):
            raise ValueError(f"baseline export does not support PEFT {name}")
    for name in ("use_rslora", "use_qalora"):
        if config.get(name) is not False:
            raise ValueError(f"baseline export requires {name}=false")
    if config.get("use_bdlora") is not None:
        raise ValueError("baseline export does not support use_bdlora")


def _publication(directory: Path, snapshot: PolicySnapshot, quantization: dict[str, Any]) -> dict[str, Any]:
    files = []
    for name in _ADAPTER_FILES:
        path = directory / name
        files.append({"name": name, "sha256": None, "size": path.stat().st_size})
    return {
        "version": BASELINE_POLICY_VERSION,
        "directory": str(directory),
        "base_fingerprint": snapshot.base_fingerprint,
        "artifact_fingerprint": snapshot.artifact_fingerprint,
        "source_adapter_sha256": snapshot.adapter_sha256,
        "quantization": dict(quantization),
        "files": files,
    }


def _baseline_construction(source_policy: EvaluationPolicy) -> tuple[str, str]:
    if source_policy.version == BASELINE_POLICY_VERSION:
        return "version0_sft_adapter_identity_copy", "version0_sft_adapter_identity_copy"
    return "zero_lora_B_from_published_adapter", "zero_residual_sft_lora"


def _load_existing_export(
    directory: Path,
    *,
    source_policy: EvaluationPolicy,
    source_config_size: int,
    quantization: dict[str, Any],
) -> ExportedEvaluationPolicy | None:
    provenance_path = directory / "evaluation-policy.json"
    if not provenance_path.is_file():
        return None
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, ValueError):
        return None
    expected_source = source_policy.as_dict()
    construction, baseline_source = _baseline_construction(source_policy)
    source_files = provenance.get("source_files")
    source_config = source_files.get("adapter_config.json") if isinstance(source_files, dict) else None
    baseline_policy = provenance.get("baseline_policy")
    if (provenance.get("schema") != "qwen-rl-evaluation-policy-v1"
            or provenance.get("construction") != construction
            or provenance.get("source_policy") != expected_source
            or not isinstance(source_config, dict)
            or source_config.get("sha256") is not None
            or source_config.get("size") != source_config_size
            or not isinstance(baseline_policy, dict)
            or baseline_policy.get("adapter_sha256") != source_policy.adapter_sha256
            or baseline_policy.get("base_fingerprint") != source_policy.base_fingerprint
            or baseline_policy.get("artifact_fingerprint") != source_policy.artifact_fingerprint
            or provenance.get("quantization") != quantization):
        return None
    for name in _ADAPTER_FILES:
        expected = provenance.get("files", {}).get(name)
        path = directory / name
        if (not isinstance(expected, dict) or not path.is_file()
                or expected.get("sha256") is not None
                or expected.get("size") != path.stat().st_size):
            return None
    snapshot = PolicySnapshot(
        BASELINE_POLICY_VERSION, source_policy.adapter_sha256,
        source_policy.base_fingerprint, source_policy.artifact_fingerprint,
    )
    policy = EvaluationPolicy("baseline_sft", snapshot, baseline_source, str(directory))
    return ExportedEvaluationPolicy(policy, directory, _publication(directory, snapshot, quantization), provenance_path)


def export_zero_residual_sft_policy(
    source_directory: str | Path,
    destination: str | Path,
    *,
    source_policy: EvaluationPolicy,
    quantization: dict[str, Any],
) -> ExportedEvaluationPolicy:
    """Copy the published LoRA A matrices and zero B matrices on CPU for the SFT baseline.

    The frozen base is never loaded or changed. Published version 0 is accepted only when
    every LoRA B tensor is already zero, then its two adapter files are copied byte-for-byte.
    Later policies retain their A matrices while all B matrices are zeroed on CPU.
    """
    if source_policy.role not in {"baseline_sft", "rl_checkpoint"}:
        raise ValueError("source_policy must carry a verified evaluation policy identity")
    source_directory = Path(source_directory).resolve()
    destination = Path(destination).resolve()
    if source_directory == destination or source_directory in destination.parents:
        raise ValueError("baseline destination must be outside the source policy directory")
    policy_quantization_mode(quantization)
    source_files = {name: source_directory / name for name in _ADAPTER_FILES}
    if any(not path.is_file() for path in source_files.values()):
        raise FileNotFoundError("published source policy must contain both standard PEFT files")
    source_config_size = source_files["adapter_config.json"].stat().st_size
    config = json.loads(source_files["adapter_config.json"].read_text())
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must be an object")
    _verify_standard_lora_config(config)

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("zero-residual adapter export requires safetensors and torch") from exc

    metadata = None
    with safe_open(source_files["adapter_model.safetensors"], framework="pt", device="cpu") as stream:
        metadata = stream.metadata()
    state = load_file(str(source_files["adapter_model.safetensors"]), device="cpu")
    sides: dict[str, set[str]] = {}
    for key, tensor in state.items():
        match = _LORA_KEY.fullmatch(key)
        if match is None or not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError(f"baseline export encountered a nonstandard or non-CPU adapter tensor: {key}")
        sides.setdefault(match.group("prefix"), set()).add(match.group("side"))
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"baseline export encountered a non-finite adapter tensor: {key}")
    if not sides or any(pair != {"A", "B"} for pair in sides.values()):
        raise ValueError("published adapter must have one matching LoRA A/B tensor pair per module")
    for prefix in sides:
        key_a = f"{prefix}.lora_A.weight"
        key_b = f"{prefix}.lora_B.weight"
        matrix_a, matrix_b = state[key_a], state[key_b]
        if (matrix_a.ndim != 2 or matrix_b.ndim != 2
                or matrix_a.shape[0] != matrix_b.shape[1]):
            raise ValueError(f"published LoRA A/B tensors have incompatible shapes for {prefix}")
        if source_policy.version == BASELINE_POLICY_VERSION:
            if torch.count_nonzero(matrix_b).item() != 0:
                raise ValueError("published version-0 adapter must have exactly zero LoRA B matrices")
        else:
            state[key_b] = torch.zeros_like(matrix_b, device="cpu")

    if destination.exists():
        existing = _load_existing_export(
            destination, source_policy=source_policy, source_config_size=source_config_size,
            quantization=quantization,
        )
        if existing is None:
            raise FileExistsError("baseline destination exists but does not match the requested policy identity")
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent))
    construction, baseline_source = _baseline_construction(source_policy)
    try:
        if source_policy.version == BASELINE_POLICY_VERSION:
            for name in _ADAPTER_FILES:
                shutil.copyfile(source_files[name], stage / name)
        else:
            (stage / "adapter_config.json").write_text(
                json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            save_file(state, str(stage / "adapter_model.safetensors"), metadata=metadata)
        staged_snapshot = PolicySnapshot(
            BASELINE_POLICY_VERSION,
            source_policy.adapter_sha256,
            source_policy.base_fingerprint,
            source_policy.artifact_fingerprint,
        )
        file_map = {
            name: {"sha256": None, "size": (stage / name).stat().st_size}
            for name in _ADAPTER_FILES
        }
        provenance = {
            "schema": "qwen-rl-evaluation-policy-v1",
            "construction": construction,
            "source_policy": source_policy.as_dict(),
            "source_files": {
                name: {"sha256": None, "size": path.stat().st_size}
                for name, path in source_files.items()
            },
            "baseline_policy": {
                "role": "baseline_sft",
                "version": BASELINE_POLICY_VERSION,
                "adapter_sha256": staged_snapshot.adapter_sha256,
                "base_fingerprint": staged_snapshot.base_fingerprint,
                "artifact_fingerprint": staged_snapshot.artifact_fingerprint,
                "source": baseline_source,
            },
            "quantization": dict(quantization),
            "files": file_map,
        }
        write_json_atomic(stage / "evaluation-policy.json", provenance)
        try:
            stage.rename(destination)
        except FileExistsError:
            existing = _load_existing_export(
                destination, source_policy=source_policy, source_config_size=source_config_size,
                quantization=quantization,
            )
            if existing is None:
                raise
            return existing
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    _, baseline_source = _baseline_construction(source_policy)
    policy = EvaluationPolicy("baseline_sft", staged_snapshot, baseline_source, str(destination))
    return ExportedEvaluationPolicy(
        policy, destination, _publication(destination, staged_snapshot, quantization),
        destination / "evaluation-policy.json",
    )
