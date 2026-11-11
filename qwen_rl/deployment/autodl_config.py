"""Write local AutoDL instance configs from an existing T2 runtime profile."""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ..runtime.config import load_config
from .instance_relocation import normalize_path_mappings
from .instance_plan import InstanceDeployment, load_deployment

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUNTIME_PATH_FIELDS = (
    ("model_path",),
    ("dataset_dir",),
    ("run_dir",),
    ("tokenizer_path",),
    ("artifact_identity_path",),
    ("model", "artifact_identity_path"),
    ("worker_inference", "model_path"),
    ("worker_inference", "ninfer_artifact_path"),
    ("worker_inference", "ninfer_conversion_report"),
    ("worker_inference", "ninfer_converter_python"),
    ("worker_inference", "ninfer_converter_path"),
    ("learner", "initial_adapter_path"),
    ("learner", "recovery_dir"),
    ("learner", "gradient_initial_training_dir"),
    ("learner", "packed_weight_offload_store_dir"),
    ("arc", "contract_path"),
    ("bepa", "state_dir"),
    ("evaluation", "manifest_path"),
    ("evaluation", "sft_exclusion_proof"),
    ("evaluation", "initial_count_proof"),
    ("evaluation", "state_dir"),
)
_RUNTIME_OVERRIDE_FIELDS = {
    "model_path",
    "dataset_dir",
    "artifact_identity_path",
    "worker_inference.model_path",
    "worker_inference.ninfer_artifact_path",
    "worker_inference.ninfer_conversion_report",
    "worker_inference.ninfer_artifact_sha256",
    "learner.gradient_tasks_per_worker",
    "rollout.group_size",
}


class AutoDLConfigError(ValueError):
    """An AutoDL instance config cannot be generated safely."""


def _absolute_text(value: str | Path, name: str) -> str:
    if not isinstance(value, (str, Path)):
        raise AutoDLConfigError(f"{name} must be an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute() or "\x00" in str(path):
        raise AutoDLConfigError(f"{name} must be an absolute path")
    return str(path)


def _path_mappings(value: Mapping[str | Path, str | Path] | None) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    mappings = []
    for source, target in value.items():
        source_text = _absolute_text(source, "path mapping source").rstrip("/") or "/"
        target_text = _absolute_text(target, "path mapping target").rstrip("/") or "/"
        if source_text == "/":
            raise AutoDLConfigError("path mapping source cannot be the filesystem root")
        mappings.append((source_text, target_text))
    return tuple(sorted(mappings, key=lambda pair: len(pair[0]), reverse=True))


def _map_runtime_paths(runtime: dict[str, Any], mappings: tuple[tuple[str, str], ...]) -> None:
    if not mappings:
        return
    for path in _RUNTIME_PATH_FIELDS:
        row: Any = runtime
        for key in path[:-1]:
            if not isinstance(row, dict):
                row = None
                break
            row = row.get(key)
        if not isinstance(row, dict):
            continue
        value = row.get(path[-1])
        if not isinstance(value, str) or not value.startswith("/"):
            continue
        for source, target in mappings:
            if value == source:
                row[path[-1]] = target
                break
            prefix = source.rstrip("/") + "/"
            if value.startswith(prefix):
                row[path[-1]] = target.rstrip("/") + value[len(source.rstrip("/")):]
                break


def _apply_runtime_overrides(
    runtime: dict[str, Any], overrides: Mapping[str, str | Path | int | None] | None,
) -> None:
    if overrides is None:
        return
    if not isinstance(overrides, Mapping):
        raise AutoDLConfigError("runtime_overrides must be a path-field mapping")
    for name, value in overrides.items():
        if name not in _RUNTIME_OVERRIDE_FIELDS:
            raise AutoDLConfigError(f"unsupported runtime override: {name}")
        if name in {"learner.gradient_tasks_per_worker", "rollout.group_size"}:
            if type(value) is not int:
                raise AutoDLConfigError(f"runtime_overrides.{name} must be an integer")
            parent, key = name.split(".")
            section = runtime.get(parent)
            if not isinstance(section, dict):
                raise AutoDLConfigError(f"runtime profile has no mapping for override: {name}")
            section[key] = value
            continue
        if name == "worker_inference.ninfer_artifact_sha256":
            if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
                raise AutoDLConfigError(f"runtime_overrides.{name} must be a non-empty string or null")
            inference = runtime.get("worker_inference")
            if not isinstance(inference, dict):
                raise AutoDLConfigError("runtime profile has no mapping for override: worker_inference.ninfer_artifact_sha256")
            inference["ninfer_artifact_sha256"] = value
            continue
        path_text = _absolute_text(value, f"runtime_overrides.{name}")
        if name == "artifact_identity_path":
            model = runtime.get("model")
            if isinstance(model, dict):
                model[name] = path_text
            else:
                runtime[name] = path_text
            continue
        parts = name.split(".")
        row: Any = runtime
        for key in parts[:-1]:
            if not isinstance(row, dict) or not isinstance(row.get(key), dict):
                raise AutoDLConfigError(f"runtime profile has no mapping for override: {name}")
            row = row[key]
        row[parts[-1]] = path_text


def write_autodl_configs(
    *,
    runtime_yaml: str | Path,
    install_root: str | Path,
    venv_python: str | Path,
    native_binary: str | Path,
    output_dir: str | Path,
    state_dir: str | Path,
    gpu_indices: Sequence[int],
    cache_root: str | Path | None = None,
    packed_cache_root: str | Path | None = None,
    credential_paths: Mapping[str, str | Path] | None = None,
    path_mappings: Mapping[str | Path, str | Path] | None = None,
    runtime_overrides: Mapping[str, str | Path | int | None] | None = None,
    teacher_base_url: str | None = None,
    runtime_config_reference: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write a local runtime YAML and instance deployment JSON.

    The source Worker/executor mappings and training settings stay in the
    runtime profile. :func:`instance_plan.build_plan` expands those owners to
    the explicitly selected GPUs.
    """
    source_path = Path(runtime_yaml).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise AutoDLConfigError("runtime_yaml must name a regular file")
    install_root_text = _absolute_text(install_root, "install_root")
    if not Path(install_root_text).is_dir():
        raise AutoDLConfigError("install_root must name an existing source directory")
    python_text = _absolute_text(venv_python, "venv_python")
    binary_text = _absolute_text(native_binary, "native_binary")
    output_path = Path(_absolute_text(output_dir, "output_dir"))
    state_path = Path(_absolute_text(state_dir, "state_dir"))
    runtime_reference_text = (
        _absolute_text(runtime_config_reference, "runtime_config_reference")
        if runtime_config_reference is not None else None
    )

    if (isinstance(gpu_indices, (str, bytes)) or not isinstance(gpu_indices, Sequence)
            or len(gpu_indices) not in {2, 4, 8}
            or any(type(gpu) is not int or gpu < 0 for gpu in gpu_indices)
            or len(set(gpu_indices)) != len(gpu_indices)):
        raise AutoDLConfigError("gpu_indices must contain 2, 4 or 8 unique nonnegative integers")

    try:
        runtime = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise AutoDLConfigError("runtime_yaml cannot be read as YAML") from None
    if not isinstance(runtime, dict) or any(not isinstance(key, str) for key in runtime):
        raise AutoDLConfigError("runtime_yaml must contain a mapping with string keys")
    learner = runtime.get("learner")
    if not isinstance(learner, dict):
        raise AutoDLConfigError("runtime_yaml learner must be a mapping")

    path_map = _path_mappings(path_mappings)
    _map_runtime_paths(runtime, path_map)
    _apply_runtime_overrides(runtime, runtime_overrides)
    if isinstance(runtime.get("model"), dict) and runtime["model"].get("backend") == "ternary":
        from .instance_artifact import ModelInstallation
        try:
            ModelInstallation.from_runtime(runtime).apply_to_runtime(runtime)
        except (TypeError, ValueError) as error:
            raise AutoDLConfigError(f"model installation paths are invalid: {error}") from None
    if path_map:
        existing = learner.get("recovery_path_mappings", {})
        try:
            merged = dict(normalize_path_mappings(existing))
            for source, target in path_map:
                if source in merged and merged[source] != target:
                    raise ValueError("recovery path map source has conflicting targets")
                merged[source] = target
            learner["recovery_path_mappings"] = normalize_path_mappings(merged)
        except ValueError as error:
            raise AutoDLConfigError(f"learner.recovery_path_mappings is invalid: {error}") from None
    learner["compute_python"] = python_text
    learner["compute_pythonpath"] = install_root_text

    credential_files: dict[str, Path] = {}
    for name, path in (credential_paths or {}).items():
        if not isinstance(name, str) or _ENV_NAME.fullmatch(name) is None:
            raise AutoDLConfigError("credential path names must be valid environment variable names")
        credential_files[name] = Path(_absolute_text(path, f"credential_paths.{name}"))

    output_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime_path = output_path / "runtime.yaml"
    deployment_path = output_path / "deployment.json"
    with tempfile.TemporaryDirectory(prefix=".autodl-config-", dir=output_path) as temporary:
        stage_dir = Path(temporary)
        staged_runtime = stage_dir / runtime_path.name
        staged_deployment = stage_dir / deployment_path.name
        staged_runtime.write_text(
            yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True), encoding="utf-8",
        )
        candidate = InstanceDeployment(
            path=staged_deployment,
            runtime_config=staged_runtime,
            state_dir=state_path,
            gpus=tuple(gpu_indices),
            engine_binary=binary_text,
            worker_python=python_text,
            python=python_text,
            credential_files=credential_files,
            teacher_base_url=teacher_base_url,
            cache_root=(Path(_absolute_text(cache_root, "cache_root")) if cache_root is not None else None),
            packed_cache_root=(Path(_absolute_text(packed_cache_root, "packed_cache_root"))
                               if packed_cache_root is not None else None),
        )
        candidate_data = candidate.public_dict()
        candidate_data.pop("path")
        staged_deployment.write_text(
            json.dumps(candidate_data, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        load_config(staged_runtime, for_local_worker_plan=True)
        load_deployment(staged_deployment)

        final_data = dict(candidate_data)
        final_data["runtime_config"] = runtime_reference_text or str(runtime_path)
        staged_final_deployment = stage_dir / "deployment-final.json"
        staged_final_deployment.write_text(
            json.dumps(final_data, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staged_runtime, runtime_path)
        os.replace(staged_final_deployment, deployment_path)
    return runtime_path, deployment_path
