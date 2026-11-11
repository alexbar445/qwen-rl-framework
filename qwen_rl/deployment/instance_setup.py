"""First-run configuration helper for a local single-instance deployment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import yaml

from .autodl_config import _map_runtime_paths, _path_mappings, write_autodl_configs
from .instance_plan import build_plan, load_deployment
from ..cluster.settings import validate_credential_origin
from ..runtime.config import BepaTeacherConfig, load_config


_SETUP_FIELDS = frozenset({
    "model_path", "dataset_dir", "artifact_identity_path", "conversion_report",
    "python", "native_binary", "install_root", "runtime_config", "gpus",
    "output_dir", "state_dir", "credential_files",
    "packed_cache_root",
    "learner.gradient_tasks_per_worker", "rollout.group_size",
})


class InstanceSetupError(ValueError):
    """A first-run instance configuration cannot be generated safely."""

    def __init__(self, message: str, *, field: str | None = None):
        super().__init__(message)
        self.field = field if field in _SETUP_FIELDS else None


def _error_field(name: str) -> str | None:
    if name.startswith("credential_paths."):
        return "credential_files"
    return name if name in _SETUP_FIELDS else None


def _read_template(path: Path, mappings: Mapping[str | Path, str | Path] | None) -> dict[str, Any]:
    try:
        if path.stat().st_size > 2 * 1024**2:
            raise InstanceSetupError("runtime_config exceeds the supported size", field="runtime_config")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise InstanceSetupError("runtime_config cannot be read as YAML", field="runtime_config") from None
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise InstanceSetupError("runtime_config must contain a mapping with string keys", field="runtime_config")
    try:
        _map_runtime_paths(value, _path_mappings(mappings))
    except ValueError:
        raise InstanceSetupError("runtime_config path mappings are invalid", field="runtime_config") from None
    return value


def _value_at(value: Mapping[str, Any], *path: str) -> Any:
    row: Any = value
    for key in path:
        if not isinstance(row, Mapping):
            return None
        row = row.get(key)
    return row


def _identity_path(profile: Mapping[str, Any]) -> str | None:
    model = profile.get("model")
    value = model.get("artifact_identity_path") if isinstance(model, Mapping) else None
    value = value or profile.get("artifact_identity_path")
    return value if isinstance(value, str) and value.strip() else None


def _parse_gpus(value: Sequence[int] | str | None, *, interactive: bool,
                input_fn: Callable[[str], str]) -> tuple[int, ...]:
    if value is None:
        if not interactive:
            raise InstanceSetupError("gpus is required unless interactive mode is enabled", field="gpus")
        answer = _ask(input_fn, "GPU indices (comma separated; 2, 4 or 8, default 0,1): ", field="gpus")
        value = answer.strip() or "0,1"
    if isinstance(value, str):
        try:
            values = tuple(int(part.strip()) for part in value.split(","))
        except ValueError:
            raise InstanceSetupError("gpus must be comma-separated nonnegative integers", field="gpus") from None
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(value)
    else:
        raise InstanceSetupError("gpus must be a sequence of GPU indices", field="gpus")
    if (len(values) not in {2, 4, 8} or any(type(gpu) is not int or gpu < 0 for gpu in values)
            or len(set(values)) != len(values)):
        raise InstanceSetupError("gpus must contain 2, 4 or 8 unique nonnegative integers", field="gpus")
    return values


def _ask(input_fn: Callable[[str], str], prompt: str, *, field: str | None = None) -> str:
    try:
        value = input_fn(prompt)
    except (EOFError, OSError):
        raise InstanceSetupError("interactive input is unavailable", field=field) from None
    if not isinstance(value, str):
        raise InstanceSetupError("interactive input must be text", field=field)
    return value


def _prompt_path(input_fn: Callable[[str], str], label: str, current: str | Path | None,
                 *, required: bool = False, field: str | None = None) -> str | None:
    default = str(current) if current is not None and str(current).strip() else None
    suffix = "" if required or default is None else f" [{default}]"
    value = _ask(input_fn, f"{label}{suffix}: ", field=field).strip()
    if value:
        return value
    return None if required or default is None else default


def _path(
    value: str | Path,
    name: str,
    *,
    kind: str,
    executable: bool = False,
    preserve_path: bool = False,
) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
        raise InstanceSetupError(f"{name} must be a path", field=_error_field(name))
    candidate = Path(value).expanduser().absolute()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise InstanceSetupError(f"{name} does not exist", field=_error_field(name)) from None
    if kind == "file":
        valid = resolved.is_file()
    elif kind == "directory":
        valid = resolved.is_dir()
    elif kind == "file_or_directory":
        valid = resolved.is_file() or resolved.is_dir()
    else:
        raise InstanceSetupError(f"unsupported path kind for {name}", field=_error_field(name))
    if not valid:
        expected = {"file": "regular file", "directory": "directory",
                    "file_or_directory": "file or directory"}[kind]
        raise InstanceSetupError(f"{name} must name an existing {expected}", field=_error_field(name))
    if executable and not os.access(resolved, os.X_OK):
        raise InstanceSetupError(f"{name} must be executable", field=_error_field(name))
    return candidate if preserve_path else resolved


def _compare_path(value: str | Path | None) -> str | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    return os.path.normpath(str(Path(value).expanduser().absolute()))


def _reference_path(value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
        raise InstanceSetupError(f"{name} must be a path", field=_error_field(name))
    return Path(value).expanduser().absolute()


def _installed_prefix(python: str | Path) -> Path | None:
    if not isinstance(python, (str, Path)) or not str(python).strip():
        return None
    candidate = Path(python).expanduser().absolute()
    if len(candidate.parents) >= 3 and candidate.parent.name == "bin" and candidate.parent.parent.name == ".venv":
        return candidate.parents[2]
    return None


def _default_native_binary(python: str | Path) -> Path | None:
    prefix = _installed_prefix(python)
    if prefix is not None:
        candidate = prefix / "native/bin/ninfer-serve"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("ninfer-serve")
    if found:
        return Path(found)
    fallback = Path("/usr/local/bin/ninfer-serve")
    return fallback if fallback.is_file() and os.access(fallback, os.X_OK) else None


def _default_install_root(python: str | Path) -> Path:
    prefix = _installed_prefix(python)
    if prefix is not None and (prefix / "source").is_dir():
        return prefix / "source"
    return Path(__file__).resolve().parents[2]


def _credential_names(profile: Mapping[str, Any]) -> tuple[str, ...]:
    names = ["ARC_ENV_TOKEN"]
    bepa = profile.get("bepa")
    teacher = bepa.get("teacher") if isinstance(bepa, Mapping) else None
    if isinstance(bepa, Mapping) and bepa.get("enabled") is True and isinstance(teacher, Mapping):
        defaults = BepaTeacherConfig()
        for key in ("api_key_env", "token_env"):
            name = teacher.get(key, getattr(defaults, key))
            if isinstance(name, str) and name.strip() and name not in names:
                names.append(name)
    return tuple(names)


def setup_instance(
    *,
    runtime_config: str | Path,
    output_dir: str | Path,
    gpus: Sequence[int] | str | None,
    model_path: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    state_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    packed_cache_root: str | Path | None = None,
    python: str | Path | None = None,
    native_binary: str | Path | None = None,
    install_root: str | Path | None = None,
    credential_paths: Mapping[str, str | Path] | None = None,
    teacher_base_url: str | None = None,
    artifact_identity_path: str | Path | None = None,
    conversion_report: str | Path | None = None,
    path_mappings: Mapping[str | Path, str | Path] | None = None,
    tasks_per_worker: int | None = None,
    trajectories_per_group: int | None = None,
    interactive: bool = False,
    input_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Generate a fresh runtime/deployment pair without starting any roles.

    A changed model artifact requires explicit matching identity and conversion
    report files. Credentials are referenced by path only and are never read or
    copied by this helper.
    """
    template_path = _path(runtime_config, "runtime_config", kind="file")
    if input_fn is None:
        input_fn = input
    if not isinstance(output_dir, (str, Path)) or not str(output_dir).strip() or "\x00" in str(output_dir):
        raise InstanceSetupError("output_dir must be a path", field="output_dir")
    output_path = Path(output_dir).expanduser().absolute()
    if output_path.exists() or output_path.is_symlink():
        raise InstanceSetupError("output_dir must be new; choose a fresh directory for each setup", field="output_dir")
    if credential_paths is not None and not isinstance(credential_paths, Mapping):
        raise InstanceSetupError("credential_paths must map environment names to file paths", field="credential_files")
    profile = _read_template(template_path, path_mappings)
    selected_gpus = _parse_gpus(gpus, interactive=interactive, input_fn=input_fn)
    if tasks_per_worker is not None and (type(tasks_per_worker) is not int or not 1 <= tasks_per_worker <= 8):
        raise InstanceSetupError(
            "tasks_per_worker must be an integer from 1 to 8",
            field="learner.gradient_tasks_per_worker",
        )
    if trajectories_per_group is not None and (
            type(trajectories_per_group) is not int or trajectories_per_group < 2):
        raise InstanceSetupError(
            "trajectories_per_group must be an integer of at least 2",
            field="rollout.group_size",
        )

    current_model = profile.get("model_path")
    current_dataset = profile.get("dataset_dir")
    current_identity = _identity_path(profile)
    current_report = _value_at(profile, "worker_inference", "ninfer_conversion_report")
    current_teacher_url = _value_at(profile, "bepa", "teacher", "base_url")
    bepa_enabled = _value_at(profile, "bepa", "enabled") is True
    if isinstance(current_teacher_url, str):
        try:
            current_teacher_url = validate_credential_origin(current_teacher_url)
        except ValueError:
            current_teacher_url = None
    current_python = _value_at(profile, "learner", "compute_python")
    running_in_venv = Path(sys.prefix).absolute() != Path(sys.base_prefix).absolute()
    if running_in_venv:
        # The manager may have been launched through prefix/.venv/bin/python.
        # Preserve that executable path: resolving it can escape the venv through
        # Python's intentional symlink to the base interpreter.
        current_python = Path(sys.executable).absolute()
    elif (not isinstance(current_python, str) or not current_python.strip()
            or not Path(current_python).expanduser().is_file()
            or not os.access(Path(current_python).expanduser(), os.X_OK)):
        current_python = sys.executable
    current_state = state_dir if state_dir is not None else output_path / "state"
    selected_python: str | Path = python if python is not None else current_python
    selected_native: str | Path | None = native_binary

    if interactive:
        if model_path is None:
            model_path = _prompt_path(input_fn, "Model artifact path", current_model, field="model_path")
        if dataset_dir is None:
            dataset_dir = _prompt_path(input_fn, "Dataset directory", current_dataset, field="dataset_dir")
        model_changed = (
            model_path is not None and _compare_path(model_path) != _compare_path(current_model)
        )
        if artifact_identity_path is None:
            artifact_identity_path = _prompt_path(
                input_fn, "Artifact identity file" if not model_changed else "Artifact identity file for selected model",
                current_identity if not model_changed else None, required=model_changed,
                field="artifact_identity_path",
            )
        if conversion_report is None:
            conversion_report = _prompt_path(
                input_fn, "Conversion report file" if not model_changed else "Conversion report for selected model",
                current_report if not model_changed else None, required=model_changed,
                field="conversion_report" if not model_changed else "artifact_identity_path",
            )
        if python is None:
            selected_python = _prompt_path(
                input_fn, "Python interpreter", current_python, field="python",
            ) or current_python
        if native_binary is None:
            selected_native = _prompt_path(
                input_fn, "NInfer native binary", _default_native_binary(selected_python),
                field="native_binary",
            )
        if state_dir is None:
            state_dir = _prompt_path(
                input_fn, "Instance state directory", current_state, field="state_dir",
            ) or current_state
        if bepa_enabled and teacher_base_url is None:
            teacher_base_url = _prompt_path(
                input_fn, "Teacher base URL", current_teacher_url, field="runtime_config",
            )
        supplied_credentials = dict(credential_paths or {})
        for name in _credential_names(profile):
            if name in supplied_credentials:
                continue
            value = _prompt_path(
                input_fn, f"Credential file path for {name} (blank to skip)", None,
                field="credential_files",
            )
            if value:
                supplied_credentials[name] = value
        credential_paths = supplied_credentials
    else:
        model_changed = model_path is not None and _compare_path(model_path) != _compare_path(current_model)
        if native_binary is None:
            selected_native = _default_native_binary(selected_python)

    if model_changed and (artifact_identity_path is None or conversion_report is None):
        raise InstanceSetupError(
            "changing model_path requires explicit artifact_identity_path and conversion_report",
            field="artifact_identity_path",
        )
    if teacher_base_url is not None:
        if not bepa_enabled:
            raise InstanceSetupError("teacher_base_url requires an enabled BEPA teacher", field="runtime_config")
        try:
            teacher_base_url = validate_credential_origin(teacher_base_url)
        except ValueError:
            raise InstanceSetupError("teacher_base_url is not a permitted URL", field="runtime_config") from None

    effective_model = model_path if model_path is not None else current_model
    effective_dataset = dataset_dir if dataset_dir is not None else current_dataset
    model_file_or_dir = _path(effective_model, "model_path", kind="file_or_directory") if effective_model else None
    if model_file_or_dir is None:
        raise InstanceSetupError("runtime profile must provide a model_path", field="model_path")
    dataset_path = _path(effective_dataset, "dataset_dir", kind="directory") if effective_dataset else None
    if dataset_path is None:
        raise InstanceSetupError("runtime profile must provide a dataset_dir", field="dataset_dir")

    overrides: dict[str, str | Path | int | None] = {}
    if model_path is not None:
        model_text = str(model_file_or_dir)
        overrides.update({
            "model_path": model_text,
            "worker_inference.model_path": model_text,
            "worker_inference.ninfer_artifact_path": model_text,
        })
        if model_changed:
            # The selected identity is the source of truth for this declaration.
            overrides["worker_inference.ninfer_artifact_sha256"] = None
    if dataset_dir is not None:
        overrides["dataset_dir"] = str(dataset_path)
    if artifact_identity_path is not None:
        overrides["artifact_identity_path"] = str(_path(
            artifact_identity_path, "artifact_identity_path", kind="file",
        ))
    if conversion_report is not None:
        overrides["worker_inference.ninfer_conversion_report"] = str(_path(
            conversion_report, "conversion_report", kind="file",
        ))
    if tasks_per_worker is not None:
        overrides["learner.gradient_tasks_per_worker"] = tasks_per_worker
    if trajectories_per_group is not None:
        overrides["rollout.group_size"] = trajectories_per_group

    python_path = _path(selected_python, "python", kind="file", executable=True, preserve_path=True)
    native_path = _path(selected_native, "native_binary", kind="file", executable=True) if selected_native else None
    if native_path is None:
        raise InstanceSetupError("native_binary is required; pass its prepared executable path", field="native_binary")
    source_root = _path(
        install_root if install_root is not None else _default_install_root(selected_python),
        "install_root", kind="directory",
    )
    selected_state = _reference_path(state_dir if state_dir is not None else current_state, "state_dir")
    selected_credentials = {
        name: _reference_path(path, f"credential_paths.{name}")
        for name, path in (credential_paths or {}).items()
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_context = tempfile.TemporaryDirectory(prefix=f".{output_path.name}.setup-", dir=output_path.parent)
    except OSError:
        raise InstanceSetupError("cannot prepare output_dir", field="output_dir") from None
    with temporary_context as temporary:
        staging_dir = Path(temporary) / "config"
        try:
            staged_runtime, staged_deployment = write_autodl_configs(
                runtime_yaml=template_path,
                install_root=source_root,
                venv_python=python_path,
                native_binary=native_path,
                output_dir=staging_dir,
                state_dir=selected_state,
                cache_root=(_reference_path(cache_root, "cache_root") if cache_root is not None else None),
                packed_cache_root=(_reference_path(packed_cache_root, "packed_cache_root")
                                   if packed_cache_root is not None else None),
                gpu_indices=selected_gpus,
                credential_paths=selected_credentials,
                path_mappings=path_mappings,
                runtime_overrides=overrides,
                teacher_base_url=teacher_base_url,
                runtime_config_reference=output_path / "runtime.yaml",
            )
            runtime = load_config(staged_runtime, for_local_worker_plan=True)
            _path(runtime.model_path, "model_path", kind="file_or_directory")
            _path(runtime.dataset_dir, "dataset_dir", kind="directory")

            # Synthetic identities exercise the planner without querying real GPUs.
            raw_deployment = json.loads(staged_deployment.read_text(encoding="utf-8"))
            raw_deployment["runtime_config"] = str(staged_runtime)
            validation_path = Path(temporary) / "deployment-validation.json"
            validation_path.write_text(json.dumps(raw_deployment), encoding="utf-8")
            deployment = load_deployment(validation_path)
            plan = build_plan(deployment, inventory=[
                {"index": gpu, "uuid": f"GPU-{gpu:08d}", "compute_capability": None}
                for gpu in selected_gpus
            ])
        except InstanceSetupError:
            raise
        except ValueError:
            raise InstanceSetupError("runtime_config validation failed", field="runtime_config") from None

        if output_path.exists() or output_path.is_symlink():
            raise InstanceSetupError("output_dir appeared during setup; choose a fresh directory", field="output_dir")
        try:
            os.replace(staging_dir, output_path)
        except OSError:
            raise InstanceSetupError("cannot publish output_dir", field="output_dir") from None

    from ..runtime.config import _resolve_path

    final_runtime = output_path / "runtime.yaml"
    final_deployment = output_path / "deployment.json"
    command = "qwen-rl instance --deployment " + shlex.quote(str(final_deployment)) + " doctor"
    return {
        "configuration_paths": {
            "runtime_config": str(final_runtime),
            "deployment": str(final_deployment),
        },
        "next_commands": [command],
        "resolved_state_paths": {
            "instance_state_dir": str(selected_state),
            "run_dir": str(runtime.run),
            "packed_cache_root": plan["packed_cache_root"],
            "recovery_dir": str(_resolve_path(runtime.learner.recovery_dir)
                                if runtime.learner.recovery_dir else runtime.run / "training"),
            "bepa_state_dir": (str(_resolve_path(runtime.bepa.state_dir)
                                   if runtime.bepa.state_dir else runtime.run / "bepa")
                               if runtime.bepa.enabled else None),
        },
        "workload": {
            "paired_workers": len(selected_gpus),
            "tasks_per_worker": runtime.learner.gradient_tasks_per_worker,
            "groups_per_update": plan["runtime"]["learner"]["groups_per_update"],
            "trajectories_per_group": runtime.rollout.group_size,
            "nominal_native_trajectories_per_update": (
                plan["runtime"]["learner"]["groups_per_update"] * runtime.rollout.group_size
            ),
        },
        "gpu_inventory_validation": "synthetic_planner_only",
        "training_started": False,
    }
