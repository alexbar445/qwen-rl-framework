"""Strict compute-only profile for a frozen Bonsai2 T2 Learner.

This profile deliberately has no Worker configuration. The T2 identity manifest
is the only model/quantization source; tokenizer resources are materialized from
the same NInfer container into the explicit run directory.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from qwen_rl.runtime.config import LearnerConfig

from qwen_rl.contracts.worker import MAX_CONTEXT_TOKENS

from qwen_rl.contracts.ternary import (
    TERNARY_QG_LAYOUT_VERSION,
    TernaryArtifactIdentity,
    load_pinned_artifact_identity,
)

SCHEMA_VERSION = "qwen-rl-ternary-compute/v1"
class TernaryComputeConfigError(ValueError):
    """A compute-only T2 profile is incomplete or incompatible with its identity."""


@dataclass(frozen=True)
class TernaryComputeConfig:
    artifact_identity_path: Path
    run_dir: Path
    parallelism: str
    gpus: tuple[int, ...]
    pipeline_stages: int | None
    lr: float
    max_grad_norm: float
    max_sequence_length: int
    group_size: int
    groups_per_update: int
    microbatch_size: int
    max_padded_tokens_per_microbatch: int | None
    importance_sampling_cap: float
    activation_offload: bool
    activation_offload_min_tokens: int
    activation_cache_parameter_names: tuple[str, ...]
    activation_cache_cpu_bytes: int
    activation_cache_staging_bytes: int
    checkpoint_input_recompute_span: int
    packed_weight_offload: bool
    packed_weight_offload_prefetch: bool
    packed_weight_offload_max_cpu_master_bytes: int
    packed_weight_offload_max_pinned_bytes: int
    packed_weight_offload_store_dir: Path | None
    gradient_checkpointing: bool
    ternary_regional_compile: bool
    weight_decay: float
    adam_betas: tuple[float, float]
    adam_eps: float
    compute_timeout: float
    cuda_memory_fraction: float
    compute_python: str | None
    compute_pythonpath: str | None
    recovery_mode: str
    recovery_lineage_id: str | None
    identity: TernaryArtifactIdentity
    constrained_decoding: bool = False

    @property
    def model_path(self) -> str:
        return str(Path(self.identity.document["source"]["ninfer_artifact"]["path"]))

    @property
    def tokenizer_path(self) -> Path:
        return self.run_dir / "tokenizer"

    def engine_config(self) -> dict[str, Any]:
        """Translate the profile into the existing engine and rank-service contract."""
        adapter = self.identity.document["learner"]["adapter"]
        return {
            "model_path": self.model_path,
            "artifact_identity_path": str(self.artifact_identity_path),
            "artifact_identity": self.identity.as_dict(),
            "model_backend": "ternary",
            "tokenizer_path": str(self.tokenizer_path),
            "gpus": list(self.gpus),
            "parallelism": self.parallelism,
            "pipeline_stages": self.pipeline_stages,
            "compute_python": self.compute_python,
            "compute_pythonpath": self.compute_pythonpath,
            "compute_timeout": self.compute_timeout,
            "cuda_memory_fraction": self.cuda_memory_fraction,
            "compute_runtime_dir": str(self.run_dir / "compute"),
            "lora_rank": adapter["rank"],
            "lora_alpha": adapter["alpha"],
            "lr": self.lr,
            "checkpoint_dir": str(self.run_dir / "checkpoints"),
            "max_sequence_length": self.max_sequence_length,
            "group_size": self.group_size,
            "groups_per_update": self.groups_per_update,
            "microbatch_size": self.microbatch_size,
            "max_padded_tokens_per_microbatch": self.max_padded_tokens_per_microbatch,
            "importance_sampling_cap": self.importance_sampling_cap,
            "constrained_decoding": self.constrained_decoding,
            "activation_offload": self.activation_offload,
            "activation_cache_parameter_names": list(self.activation_cache_parameter_names),
            "activation_cache_cpu_bytes": self.activation_cache_cpu_bytes,
            "activation_cache_staging_bytes": self.activation_cache_staging_bytes,
            "activation_offload_min_tokens": self.activation_offload_min_tokens,
            "checkpoint_input_recompute_span": self.checkpoint_input_recompute_span,
            "packed_weight_offload": self.packed_weight_offload,
            "packed_weight_offload_prefetch": self.packed_weight_offload_prefetch,
            "packed_weight_offload_max_cpu_master_bytes": self.packed_weight_offload_max_cpu_master_bytes,
            "packed_weight_offload_max_pinned_bytes": self.packed_weight_offload_max_pinned_bytes,
            "packed_weight_offload_store_dir": (
                str(self.packed_weight_offload_store_dir)
                if self.packed_weight_offload_store_dir is not None else None
            ),
            "gradient_checkpointing": self.gradient_checkpointing,
            "ternary_regional_compile": self.ternary_regional_compile,
            "weight_decay": self.weight_decay,
            "adam_betas": list(self.adam_betas),
            "adam_eps": self.adam_eps,
            "max_grad_norm": self.max_grad_norm,
            "recovery_dir": str(self.run_dir / "training"),
            "recovery_mode": self.recovery_mode,
            "recovery_lineage_id": self.recovery_lineage_id,
            "recovery_semantics": {
                "profile_schema": SCHEMA_VERSION,
                "rollout_source": "caller_supplied_complete_trajectory_batch/v1",
                "base_artifact_sha256": self.identity.artifact_sha256,
                "qg_layout": TERNARY_QG_LAYOUT_VERSION,
                "policy_distribution": "grammar_conditioned_policy/v1" if self.constrained_decoding else "raw_policy",
            },
        }


# The standalone profile schema is the parsed config itself, plus its version
# marker. Keeping this set derived from the typed boundary prevents a producer
# from accepting/emitting a second hand-maintained list of compute fields.
_COMPUTE_PROFILE_INTERNAL_FIELDS = frozenset({"identity"})
_LOCAL_EXECUTOR_OMITTED_FIELDS = frozenset({"pipeline_stages", "recovery_lineage_id"})
TERNARY_COMPUTE_PROFILE_FIELDS = frozenset(
    item.name for item in fields(TernaryComputeConfig)
    if item.init and item.name not in _COMPUTE_PROFILE_INTERNAL_FIELDS
) | {"schema_version"}


def project_ternary_compute_profile(
    learner: LearnerConfig,
    *,
    group_size: int,
    artifact_identity_path: str,
    run_dir: str,
    store_dir: str,
    constrained_decoding: bool = False,
) -> dict[str, Any]:
    """Project runtime learner settings into one local executor profile.

    The executor is a single visible CUDA device and computes one group. Values
    shared with ``LearnerConfig`` are copied by name from that typed config;
    fields owned only by the standalone parser keep its defaults. Shared fields
    intentionally inherit runtime defaults, which can differ from standalone
    defaults (the packed-master limit is 4 GiB at runtime and 8 GiB standalone).
    """
    fixed_profile_values: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_identity_path": artifact_identity_path,
        "run_dir": run_dir,
        "parallelism": "single",
        "gpus": [0],
        "max_grad_norm": 1.0,
        "group_size": group_size,
        "groups_per_update": 1,
        "packed_weight_offload_store_dir": store_dir,
        "recovery_mode": "auto",
    }
    profile: dict[str, Any] = {}
    for name in TERNARY_COMPUTE_PROFILE_FIELDS:
        if name in fixed_profile_values:
            continue
        if name in _LOCAL_EXECUTOR_OMITTED_FIELDS:
            # PP topology and recovery lineage belong outside local executors.
            continue
        if not hasattr(learner, name):
            continue
        value = getattr(learner, name)
        if name == "max_padded_tokens_per_microbatch" and value is None:
            continue
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, Path):
            value = str(value)
        profile[name] = value
    profile.update(fixed_profile_values)
    if constrained_decoding:
        profile["constrained_decoding"] = True
    return profile


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TernaryComputeConfigError(f"{name} must be a mapping with string keys")
    return value


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TernaryComputeConfigError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise TernaryComputeConfigError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise TernaryComputeConfigError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise TernaryComputeConfigError(f"{name} must be <= {maximum}")
    return number


def load_ternary_compute_config(path: str | Path) -> TernaryComputeConfig:
    """Load a no-Worker T2 profile and validate pinned artifact metadata."""
    profile_path = Path(path).expanduser().resolve()
    try:
        text = profile_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TernaryComputeConfigError(f"cannot read compute profile: {profile_path}") from exc
    try:
        loaded = json.loads(text) if text.lstrip().startswith("{") else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise TernaryComputeConfigError("compute profile is invalid JSON/YAML") from exc
    data = _mapping(loaded, "compute profile")
    training = data.get("training")
    if isinstance(training, dict) and training.get("objective") == "compression_recovery":
        raise TernaryComputeConfigError(
            "compression_recovery profiles are no longer supported by learner compute"
        )
    extra = sorted(set(data) - TERNARY_COMPUTE_PROFILE_FIELDS)
    if extra:
        raise TernaryComputeConfigError(f"unknown compute profile field(s): {extra}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise TernaryComputeConfigError(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("artifact_identity_path", "run_dir"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise TernaryComputeConfigError(f"{key} must be a non-empty path string")
    for key in ("compute_python", "compute_pythonpath"):
        if data.get(key) is not None and (not isinstance(data[key], str) or not data[key].strip()):
            raise TernaryComputeConfigError(f"{key} must be a non-empty string when set")
    base = profile_path.parent
    identity_path = (base / data["artifact_identity_path"]).resolve() if not Path(data["artifact_identity_path"]).is_absolute() else Path(data["artifact_identity_path"]).resolve()
    run_dir = (base / data["run_dir"]).resolve() if not Path(data["run_dir"]).is_absolute() else Path(data["run_dir"]).resolve()
    mode = data.get("parallelism")
    if not isinstance(mode, str) or mode not in {"single", "pp"}:
        raise TernaryComputeConfigError("parallelism must be single or pp")
    raw_gpus = data.get("gpus")
    if not isinstance(raw_gpus, list) or not raw_gpus or any(type(gpu) is not int or gpu < 0 for gpu in raw_gpus):
        raise TernaryComputeConfigError("gpus must be a non-empty list of distinct nonnegative CUDA indices")
    gpus = tuple(raw_gpus)
    if len(set(gpus)) != len(gpus):
        raise TernaryComputeConfigError("gpus must contain distinct CUDA indices")
    pipeline_stages = data.get("pipeline_stages")
    if mode == "single":
        if len(gpus) != 1 or pipeline_stages is not None:
            raise TernaryComputeConfigError("single requires one GPU and omits pipeline_stages")
    else:
        if len(gpus) != 4 or type(pipeline_stages) is not int or pipeline_stages != 4:
            raise TernaryComputeConfigError("pp requires exactly four GPUs and pipeline_stages=4")
        if data.get("max_padded_tokens_per_microbatch") is not None:
            raise TernaryComputeConfigError(
                "max_padded_tokens_per_microbatch requires parallelism=single"
            )
        for key in ("compute_python", "compute_pythonpath"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise TernaryComputeConfigError(f"pp requires explicit {key}")

    integer_fields = {
        "max_sequence_length": (2, MAX_CONTEXT_TOKENS if mode == "single" else 65536),
        "group_size": (2, None),
        "groups_per_update": (1, None),
        "microbatch_size": (1, None),
    }
    values: dict[str, int] = {}
    for key, (minimum, maximum) in integer_fields.items():
        value = data.get(key)
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            upper = f" and <= {maximum}" if maximum is not None else ""
            raise TernaryComputeConfigError(f"{key} must be an integer >= {minimum}{upper}")
        values[key] = value
    dynamic_microbatch_cap = data.get("max_padded_tokens_per_microbatch")
    if dynamic_microbatch_cap is not None and (
            type(dynamic_microbatch_cap) is not int or dynamic_microbatch_cap < 1):
        raise TernaryComputeConfigError(
            "max_padded_tokens_per_microbatch must be a positive integer"
        )
    samples = values["group_size"] * values["groups_per_update"]
    if samples % values["microbatch_size"]:
        raise TernaryComputeConfigError("microbatch_size must divide group_size * groups_per_update")
    if mode == "pp" and samples // values["microbatch_size"] < 4:
        raise TernaryComputeConfigError("pp requires at least four microbatches for four pipeline stages")
    raw_betas = data.get("adam_betas", [0.9, 0.999])
    if not isinstance(raw_betas, list) or len(raw_betas) != 2:
        raise TernaryComputeConfigError("adam_betas must contain exactly two finite values")
    betas = tuple(_finite_number(value, f"adam_betas[{index}]", minimum=0.0)
                  for index, value in enumerate(raw_betas))
    if any(beta >= 1 for beta in betas):
        raise TernaryComputeConfigError("adam_betas values must be below 1")
    activation_offload = data.get("activation_offload", True)
    if type(activation_offload) is not bool:
        raise TernaryComputeConfigError("activation_offload must be a boolean")
    ternary_regional_compile = data.get("ternary_regional_compile", True)
    if type(ternary_regional_compile) is not bool:
        raise TernaryComputeConfigError("ternary_regional_compile must be a boolean")
    gradient_checkpointing = data.get("gradient_checkpointing", True)
    if type(gradient_checkpointing) is not bool:
        raise TernaryComputeConfigError("gradient_checkpointing must be a boolean")

    selectors = data.get("activation_cache_parameter_names", [])
    if not isinstance(selectors, (list, tuple)) or any(not isinstance(v, str) or not v for v in selectors):
        raise TernaryComputeConfigError("activation_cache_parameter_names must be nonempty strings")
    memory_values = {"activation_cache_parameter_names": tuple(selectors)}
    for key, default, minimum in (
        ("activation_offload_min_tokens", 0, 0),
        ("activation_cache_cpu_bytes", 2 * 1024**3, 1),
        ("activation_cache_staging_bytes", 512 * 1024**2, 1),
        ("checkpoint_input_recompute_span", 1, 1),
        ("packed_weight_offload_max_cpu_master_bytes", 8 * 1024**3, 1),
        ("packed_weight_offload_max_pinned_bytes", 512 * 1024**2, 1),
    ):
        value = data.get(key, default)
        if type(value) is not int or value < minimum:
            raise TernaryComputeConfigError(f"{key} must be an integer >= {minimum}")
        memory_values[key] = value
    if memory_values["checkpoint_input_recompute_span"] != 1:
        raise TernaryComputeConfigError("checkpoint_input_recompute_span must be 1")
    for key, default in (("packed_weight_offload", False), ("packed_weight_offload_prefetch", True)):
        value = data.get(key, default)
        if type(value) is not bool:
            raise TernaryComputeConfigError(f"{key} must be a boolean")
        memory_values[key] = value
    store_dir = data.get("packed_weight_offload_store_dir")
    if store_dir is not None:
        if not isinstance(store_dir, str) or not store_dir.strip():
            raise TernaryComputeConfigError("packed_weight_offload_store_dir must be a non-empty path")
        store_dir = Path(store_dir).expanduser()
        store_dir = (base / store_dir).resolve() if not store_dir.is_absolute() else store_dir.resolve()
    memory_values["packed_weight_offload_store_dir"] = store_dir
    recovery_mode = data.get("recovery_mode", "auto")
    lineage_id = data.get("recovery_lineage_id")
    if recovery_mode not in {"auto", "fresh"}:
        raise TernaryComputeConfigError("recovery_mode must be auto or fresh")
    if recovery_mode == "fresh":
        if not isinstance(lineage_id, str) or not lineage_id.strip():
            raise TernaryComputeConfigError("fresh recovery requires recovery_lineage_id")
    elif lineage_id is not None:
        raise TernaryComputeConfigError("recovery_lineage_id is only valid with recovery_mode=fresh")

    # Reject malformed/unsupported runtime geometry before opening the pinned
    # NInfer artifact. Model context capacity is an artifact fact (262144 here);
    # single-card gradient execution follows the Worker wire context ceiling;
    # the separately configured PP path retains its 65536 limit.
    identity = load_pinned_artifact_identity(identity_path)

    constrained = data.get("constrained_decoding",False)
    if type(constrained) is not bool or (constrained and mode != "single"):
        raise TernaryComputeConfigError("constrained_decoding requires a boolean and single-device profile")
    return TernaryComputeConfig(
        constrained_decoding=constrained,
        artifact_identity_path=identity_path,
        run_dir=run_dir,
        parallelism=mode,
        gpus=gpus,
        pipeline_stages=pipeline_stages,
        lr=_finite_number(data.get("lr", 1e-6), "lr", minimum=1e-300),
        max_grad_norm=_finite_number(data.get("max_grad_norm", 1.0), "max_grad_norm", minimum=1e-300),
        max_sequence_length=values["max_sequence_length"],
        group_size=values["group_size"],
        groups_per_update=values["groups_per_update"],
        microbatch_size=values["microbatch_size"],
        max_padded_tokens_per_microbatch=dynamic_microbatch_cap,
        importance_sampling_cap=_finite_number(
            data.get("importance_sampling_cap", 3.0), "importance_sampling_cap", minimum=1.0
        ),
        activation_offload=activation_offload,
        **memory_values,
        gradient_checkpointing=gradient_checkpointing,
        ternary_regional_compile=ternary_regional_compile,
        weight_decay=_finite_number(data.get("weight_decay", 0.0), "weight_decay", minimum=0.0),
        adam_betas=(float(betas[0]), float(betas[1])),
        adam_eps=_finite_number(data.get("adam_eps", 1e-8), "adam_eps", minimum=1e-300),
        compute_timeout=_finite_number(data.get("compute_timeout", 3600.0), "compute_timeout", minimum=1e-300),
        cuda_memory_fraction=(
            _finite_number(
                data.get("cuda_memory_fraction", 1.0), "cuda_memory_fraction",
                minimum=1e-300, maximum=1.0,
            )
        ),
        compute_python=data.get("compute_python"),
        compute_pythonpath=data.get("compute_pythonpath"),
        recovery_mode=recovery_mode,
        recovery_lineage_id=lineage_id,
        identity=identity,
    )
