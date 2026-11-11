"""CPU-only startup checks for the managed single-instance deployment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time
from typing import Any, Mapping

from .instance_plan import InstanceDeploymentError

_SCHEMA = "qwen-rl-instance-preflight/v1"
_PROBE_TIMEOUT_SECONDS = 180
_MAX_DEPENDENCY_CONFLICTS = 20
_MAX_PIP_OUTPUT_CHARS = 64 * 1024
_PACKAGE = r"[A-Za-z0-9][A-Za-z0-9._+-]{0,79}"
_VERSION = r"[A-Za-z0-9][A-Za-z0-9._+!~-]{0,79}"
_VERSION_OP = r"(?:===|==|!=|~=|<=|>=|<|>)"
_REQUIREMENT = (rf"(?P<required_package>{_PACKAGE})"
                rf"(?:\[[A-Za-z0-9_,.-]{{1,80}}\])?"
                rf"(?:\s*{_VERSION_OP}\s*{_VERSION})"
                rf"(?:,\s*{_VERSION_OP}\s*{_VERSION})*")
_PIP_CONFLICT = re.compile(
    rf"^(?P<package>{_PACKAGE}) (?P<package_version>{_VERSION}) has requirement "
    rf"(?P<required>{_REQUIREMENT}), but you have "
    rf"(?P<installed_package>{_PACKAGE}) (?P<installed_version>{_VERSION})\.$"
)
_MISSING_MODULE = re.compile(
    r"^ModuleNotFoundError: No module named ['\"]([A-Za-z_][A-Za-z0-9_.]{0,119})['\"]$"
)
_SECRET_ENV_NAME = re.compile(r"(?i)(token|secret|password|credential|auth|api[_-]?key|private[_-]?key)")
# Keep this allowlist aligned with instance._initialize_caches; it is also the
# only set of environment names that can appear as a cache failure field.
_CACHE_ENV_NAMES = (
    "HOME", "XDG_CACHE_HOME", "HF_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH",
    "FLASHINFER_WORKSPACE_BASE",
)
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "MKL_THREADING_LAYER": "SEQUENTIAL",
}

_DATASET_PROBE = r"""
import sys
from pathlib import Path
from qwen_rl.task_source import load_tasks
from qwen_rl.task_source_generated import _generator
from qwen_rl.runtime.config import _resolve_path

dataset, verifier = _generator()
assert callable(dataset.load_split)
assert callable(verifier.verify)
checked = verifier.verify({"verifier": {"targets": {"test/0": [[0]]}}},
                          {"test/0": [[0]]})
assert checked == {"all_exact": True, "per_target_exact": {"test/0": True}}
tasks = load_tasks(pools=("arc2_train",), variants=False,
                   dataset_dir=_resolve_path(sys.argv[1]))
assert tasks and isinstance(tasks[0].challenge, dict)
"""

_MANAGER_PROBE = r"""
import sys
from pathlib import Path
from qwen_rl.deployment.instance_compat import verify_cpu
verify_cpu(native_binary=Path(sys.argv[1]), cuobjdump=None)
"""

_TOKENIZER_PROBE = r"""
import sys
from pathlib import Path
from transformers import AutoTokenizer
from qwen_rl.models.ternary_provider import TernaryModelProvider

with TernaryModelProvider(sys.argv[1], verify_files=False) as provider:
    resources = provider.materialize_text_resources(Path(sys.argv[2]))
assert resources
tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True)
assert len(tokenizer) > 0
"""

_WORKER_PROBE = r"""
from importlib import metadata
from qwen_rl.worker import app, ninfer_backend, ninfer_launcher

# These exact interfaces are pinned by the current worker install profile.
for name, expected in (("vllm", "0.29.0"), ("torchao", "0.18.0"),
                       ("transformers", "5.12.1")):
    assert metadata.version(name) == expected
assert app and ninfer_backend and ninfer_launcher
"""


class InstancePreflightError(InstanceDeploymentError):
    """A safe, structured CPU preflight failure that blocks instance startup."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        errors = report.get("errors") if isinstance(report, Mapping) else None
        first = errors[0] if isinstance(errors, list) and errors and isinstance(errors[0], Mapping) else {}
        self.reason_code = first.get("reason_code")
        self.field = first.get("field")
        super().__init__(str(first.get("message") or "instance CPU startup preflight is blocked"))


def _command(supervisor: Any) -> str:
    path = getattr(getattr(supervisor, "deployment", None), "path", None)
    if isinstance(path, Path) and str(path):
        return f"qwen-rl instance --deployment {shlex.quote(str(path))} doctor"
    return "qwen-rl instance --deployment <deployment.json> doctor"


class _PreflightReport(dict):
    def __init__(self, value, *, progress=None):
        super().__init__(value)
        self.started = time.monotonic()
        self.active_stages = {}
        self.progress = progress


def _stage(report: dict[str, Any], name: str, status: str, message: str,
           *, field: str | None = None) -> None:
    item: dict[str, Any] = {"stage": name, "status": status, "message": message}
    if isinstance(report, _PreflightReport):
        now = time.monotonic()
        item["elapsed_seconds"] = round(now - report.started, 3)
        if status == "running":
            report.active_stages[name] = now
        elif name in report.active_stages:
            item["duration_seconds"] = round(now - report.active_stages.pop(name), 3)
    if field is not None:
        item["field"] = field
    report["stages"].append(item)
    if isinstance(report, _PreflightReport) and report.progress is not None:
        report.progress(dict(item))


def _block(report: dict[str, Any], code: str, field: str, message: str,
           next_command: str) -> None:
    report["errors"].append({
        "reason_code": code,
        "field": field,
        "message": message,
        "next_command": next_command,
    })
    if report["next_command"] is None:
        report["next_command"] = next_command


def _child_environment() -> dict[str, str]:
    # Match the managed runtime's inherited allowlist, while making these probes
    # incapable of seeing a GPU and bounding common CPU thread pools.
    from .local_gradients import _SAFE_INHERITED_ENV

    env = {name: value for name, value in os.environ.items()
           if name in _SAFE_INHERITED_ENV}
    source_root = str(Path(__file__).resolve().parents[2])
    env.update(
        ARC_RL_ROOT=source_root,
        PYTHONPATH=source_root,
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        NVIDIA_VISIBLE_DEVICES="void",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        **_THREAD_ENV,
    )
    return env


def _run(python: str, code: str, arguments: tuple[str, ...], *, env: dict[str, str]) -> None:
    _run_command([python, "-B", "-c", code, *arguments], env=env)


def _run_command(command: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=_PROBE_TIMEOUT_SECONDS,
        check=True,
    )


def _record_probe(report: dict[str, Any], *, stage: str, field: str, python: str,
                  code: str, arguments: tuple[str, ...], env: dict[str, str]) -> bool:
    _stage(report, stage, "running", "Checking the configured CPU interpreter.", field=field)
    try:
        _run(python, code, arguments, env=env)
    except subprocess.TimeoutExpired:
        _stage(report, stage, "blocked", "The bounded CPU probe exceeded its time limit.", field=field)
        _block(report, "cpu_probe_timeout", field,
               "The configured interpreter did not finish its CPU startup probe.",
               f"{shlex.quote(python)} -m pip check")
        return False
    except subprocess.CalledProcessError as error:
        missing = _missing_module(error.stderr)
        if missing is not None:
            message = f"Configured interpreter is missing module {missing}."
            _stage(report, stage, "blocked", message, field=field)
            _block(report, "module_not_found", field, message, f"{shlex.quote(python)} -m pip check")
            report["errors"][-1]["missing_module"] = missing
            return False
        _stage(report, stage, "blocked", "The configured interpreter failed its CPU startup probe.", field=field)
        _block(report, "cpu_probe_failed", field,
               "The configured interpreter failed its CPU startup probe.", f"{shlex.quote(python)} -m pip check")
        return False
    except (OSError, ValueError):
        _stage(report, stage, "blocked", "The configured interpreter failed its CPU startup probe.", field=field)
        _block(report, "cpu_probe_failed", field,
               "The configured interpreter failed its CPU startup probe.", f"{shlex.quote(python)} -m pip check")
        return False
    _stage(report, stage, "passed", "The configured interpreter passed its CPU startup probe.", field=field)
    return True


def _parallel_probes(report: dict[str, Any], probes: list[dict[str, Any]]) -> dict[str, bool]:
    """Overlap independent CPU subprocesses; only this caller writes the report."""
    if not probes:
        return {}
    outcomes = {}
    with ThreadPoolExecutor(max_workers=min(3, len(probes)), thread_name_prefix="startup-probe") as pool:
        pending = {}
        for probe in probes:
            _stage(report, probe["stage"], "running", "Checking the configured CPU interpreter.",
                   field=probe["field"])
            local = {"stages": [], "errors": [], "next_command": None}
            pending[pool.submit(_record_probe, local, **probe)] = (probe, local)
        for future in as_completed(pending):
            probe, local = pending[future]
            outcomes[probe["stage"]] = future.result()
            for event in local["stages"]:
                if event["status"] != "running":
                    _stage(report, event["stage"], event["status"], event["message"],
                           field=event.get("field"))
            report["errors"].extend(local["errors"])
            if report["next_command"] is None:
                report["next_command"] = local["next_command"]
    return outcomes


def _record_pip_check(report: dict[str, Any], *, stage: str, field: str, python: str,
                      env: dict[str, str]) -> bool:
    _stage(report, stage, "running", "Checking installed dependencies.", field=field)
    try:
        result = subprocess.run(
            [python, "-B", "-m", "pip", "check"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _stage(report, stage, "blocked", "The bounded dependency check exceeded its time limit.", field=field)
        _block(report, "cpu_probe_timeout", field,
               "The configured interpreter did not finish its dependency check.",
               f"{shlex.quote(python)} -m pip check")
        return False
    except (OSError, ValueError):
        _stage(report, stage, "blocked", "The configured environment failed its dependency check.", field=field)
        _block(report, "dependency_check_failed", field,
               "The configured environment has missing or incompatible dependencies.",
               f"{shlex.quote(python)} -m pip check")
        return False
    if result.returncode != 0:
        output = _redact_known_environment_values((result.stdout or "") + "\n" + (result.stderr or ""))
        output = output[:_MAX_PIP_OUTPUT_CHARS]
        conflicts = [parsed for line in output.splitlines()
                     if (parsed := _parse_pip_conflict(line)) is not None]
        if conflicts:
            prior = report.setdefault("dependency_conflicts", [])
            remaining = max(0, _MAX_DEPENDENCY_CONFLICTS - len(prior))
            prior.extend(conflicts[:remaining])
            if len(conflicts) > remaining or len(output) >= _MAX_PIP_OUTPUT_CHARS:
                report["dependency_conflicts_truncated"] = True
            message = f"pip check found {len(conflicts)} dependency constraint conflict(s)."
        else:
            message = "pip check failed without a safely recognized dependency conflict."
        _stage(report, stage, "blocked", message, field=field)
        _block(report, "dependency_check_failed", field, message,
               f"{shlex.quote(python)} -m pip check")
        if conflicts:
            report["errors"][-1]["dependency_conflict_count"] = len(conflicts)
        return False
    _stage(report, stage, "passed", "The configured environment passed pip dependency checks.", field=field)
    return True


def _absolute_python_path(python: str) -> str:
    """Normalize a configured executable path without resolving symlinks."""
    return os.path.abspath(os.path.expanduser(python))


def _redact_known_environment_values(value: str) -> str:
    secrets = {item for name, item in os.environ.items()
               if _SECRET_ENV_NAME.search(name) and item}
    for secret in sorted(secrets, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    return value


def _parse_pip_conflict(line: str) -> dict[str, str] | None:
    if len(line) > 512:
        return None
    match = _PIP_CONFLICT.fullmatch(line.strip())
    if match is None:
        return None
    values = match.groupdict()
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    if normalize(values["required_package"]) != normalize(values["installed_package"]):
        return None
    return {
        "declaring_package": values["package"],
        "declaring_version": values["package_version"],
        "required_constraint": values["required"],
        "installed_package": values["installed_package"],
        "installed_version": values["installed_version"],
    }


def _missing_module(stderr: Any) -> str | None:
    if not isinstance(stderr, str):
        return None
    safe = _redact_known_environment_values(stderr)
    for line in reversed(safe.splitlines()[-5:]):
        match = _MISSING_MODULE.fullmatch(line.strip())
        if match is not None:
            return match.group(1)
    return None


def _path_status(value: Any, *, kind: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
        return "invalid"
    path = Path(value).expanduser()
    if not path.is_absolute():
        from ..paths import ROOT
        path = ROOT / path
    try:
        info = path.stat()
    except FileNotFoundError:
        return "missing"
    except (OSError, RuntimeError, ValueError):
        return "unavailable"
    if kind == "directory":
        valid = stat.S_ISDIR(info.st_mode)
        readable = os.R_OK | os.X_OK
    elif kind == "file_or_directory":
        valid = stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        readable = os.R_OK | (os.X_OK if stat.S_ISDIR(info.st_mode) else 0)
    else:
        valid = stat.S_ISREG(info.st_mode)
        readable = os.R_OK
    if not valid:
        return "unavailable"
    return "passed" if os.access(path, readable) else "unavailable"


def _writable_target(value: Any) -> bool:
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
        return False
    path = Path(value).expanduser()
    if not path.is_absolute():
        from ..paths import ROOT
        path = ROOT / path
    try:
        current = path
        while not current.exists():
            if current.is_symlink():
                return False
            parent = current.parent
            if parent == current:
                return False
            current = parent
        info = current.stat()
        if not stat.S_ISDIR(info.st_mode):
            return False
        return os.access(current, os.R_OK | os.W_OK | os.X_OK)
    except (OSError, RuntimeError, ValueError):
        return False


def _runtime_paths(runtime: Mapping[str, Any], supervisor: Any) -> list[tuple[str, Any, str]]:
    model = runtime.get("model") if isinstance(runtime.get("model"), Mapping) else {}
    inference = (runtime.get("worker_inference")
                 if isinstance(runtime.get("worker_inference"), Mapping) else {})
    learner = runtime.get("learner") if isinstance(runtime.get("learner"), Mapping) else {}
    arc = runtime.get("arc") if isinstance(runtime.get("arc"), Mapping) else {}
    evaluation = (runtime.get("evaluation")
                 if isinstance(runtime.get("evaluation"), Mapping) else {})
    paths: list[tuple[str, Any, str]] = [
        ("model_path", runtime.get("model_path"), "file_or_directory"),
        ("dataset_dir", runtime.get("dataset_dir"), "directory"),
        ("artifact_identity_path", model.get("artifact_identity_path")
         or runtime.get("artifact_identity_path"), "file"),
        ("worker_inference.ninfer_artifact_path",
         inference.get("ninfer_artifact_path"), "file"),
        ("worker_inference.ninfer_conversion_report",
         inference.get("ninfer_conversion_report"), "file"),
        ("learner.initial_adapter_path", learner.get("initial_adapter_path"), "directory"),
        ("arc.contract_path", arc.get("contract_path"), "file"),
    ]
    if evaluation.get("enabled") is True:
        paths.extend((
            ("evaluation.manifest_path", evaluation.get("manifest_path"), "file"),
            ("evaluation.sft_exclusion_proof", evaluation.get("sft_exclusion_proof"), "file"),
        ))
        if evaluation.get("initial_count_proof"):
            paths.append(("evaluation.initial_count_proof",
                          evaluation.get("initial_count_proof"), "file"))
        bepa = runtime.get("bepa") if isinstance(runtime.get("bepa"), Mapping) else {}
        state_dir = bepa.get("state_dir") or (Path(str(runtime.get("run_dir") or "")) / "bepa")
        paths.append(("bepa.state_dir/state.sqlite3",
                      Path(state_dir).expanduser() / "state.sqlite3", "file"))
    deployment = supervisor.deployment
    paths.append(("runtime_config", deployment.runtime_config, "file"))
    if deployment.gateway_config is not None:
        paths.append(("gateway_config", deployment.gateway_config, "file"))
    return [(field, value, kind) for field, value, kind in paths if value is not None]


def _check_paths(report: dict[str, Any], runtime: Mapping[str, Any], supervisor: Any,
                 *, command: str) -> None:
    failed = False
    for field, value, kind in _runtime_paths(runtime, supervisor):
        result = _path_status(value, kind=kind)
        if result == "passed":
            continue
        failed = True
        code = "required_file_missing" if result == "missing" else "required_path_unavailable"
        _block(report, code, field, "A configured runtime input is missing or inaccessible.", command)
    if not failed:
        _stage(report, "prepared_inputs", "passed",
               "Configured model, dataset, artifact, report, adapter, contract, and runtime paths are accessible.")
    else:
        _stage(report, "prepared_inputs", "blocked",
               "One or more configured runtime inputs are missing or inaccessible.")

    writable: list[tuple[str, Any]] = [
        ("state_dir", supervisor.deployment.state_dir),
        ("run_dir", runtime.get("run_dir")),
    ]
    learner = runtime.get("learner") if isinstance(runtime.get("learner"), Mapping) else {}
    writable.append(("learner.recovery_dir", learner.get("recovery_dir")))
    bepa = runtime.get("bepa") if isinstance(runtime.get("bepa"), Mapping) else {}
    if bepa.get("enabled") is True:
        writable.append(("bepa.state_dir", _bepa_state_dir(runtime)))
    evaluation = runtime.get("evaluation") if isinstance(runtime.get("evaluation"), Mapping) else {}
    if evaluation.get("enabled") is True:
        writable.append(("evaluation.state_dir", evaluation.get("state_dir")))
    for field, value in writable:
        if value is not None and not _writable_target(value):
            _block(report, "runtime_output_unavailable", field,
                   "A managed runtime output location is not writable.", command)
    if not any(error["reason_code"] == "runtime_output_unavailable" for error in report["errors"]):
        _stage(report, "managed_outputs", "passed",
               "Managed state, run, and recovery locations have writable directory access.")
    else:
        _stage(report, "managed_outputs", "blocked",
               "One or more managed runtime output locations are unavailable.")

    tokenizer_path = _tokenizer_path(runtime)
    if tokenizer_path is None:
        _block(report, "runtime_path_invalid", "tokenizer_path",
               "The Learner tokenizer destination cannot be resolved.", command)
        _stage(report, "tokenizer_destination", "blocked",
               "The Learner tokenizer destination cannot be resolved.", field="tokenizer_path")
    else:
        tokenizer_status = _path_status(tokenizer_path, kind="directory")
        destination_ready = (tokenizer_status == "passed"
                             or (tokenizer_status == "missing" and _writable_target(tokenizer_path)))
        if destination_ready:
            _stage(report, "tokenizer_destination", "passed",
                   "The configured or normal-app fallback tokenizer destination is accessible.",
                   field="tokenizer_path")
        else:
            _block(report, "required_path_unavailable", "tokenizer_path",
                   "The Learner tokenizer destination is missing or inaccessible.", command)
            _stage(report, "tokenizer_destination", "blocked",
                   "The Learner tokenizer destination is missing or inaccessible.", field="tokenizer_path")


def _tokenizer_path(runtime: Mapping[str, Any]) -> Path | None:
    value = runtime.get("tokenizer_path")
    if not value:
        run_dir = runtime.get("run_dir")
        if not isinstance(run_dir, (str, Path)) or not str(run_dir).strip():
            return None
        value = Path(run_dir).expanduser() / "tokenizer"
    try:
        from ..runtime.config import _resolve_path
        return _resolve_path(str(value))
    except (OSError, RuntimeError, ValueError):
        return None


def _bepa_state_dir(runtime: Mapping[str, Any]) -> Path:
    bepa = runtime.get("bepa") if isinstance(runtime.get("bepa"), Mapping) else {}
    value = bepa.get("state_dir") or (Path(str(runtime.get("run_dir") or "")) / "bepa")
    try:
        from ..runtime.config import _resolve_path
        return _resolve_path(str(value))
    except (OSError, RuntimeError, ValueError):
        return Path("\x00invalid")


def _check_credentials(report: dict[str, Any], runtime: Mapping[str, Any], supervisor: Any,
                       plan: Mapping[str, Any] | None, *, command: str) -> None:
    external_names = ["ARC_ENV_TOKEN"]
    bepa = runtime.get("bepa") if isinstance(runtime.get("bepa"), Mapping) else {}
    teacher = bepa.get("teacher") if isinstance(bepa.get("teacher"), Mapping) else {}
    if bepa.get("enabled") is True:
        external_names.extend((teacher.get("api_key_env", "DEEPSEEK_API_KEY"),
                               teacher.get("token_env", "BEPA_TEACHER_TOKEN")))
    values: list[str] = []
    good = True
    for name in dict.fromkeys(external_names):
        value = os.environ.get(name) if isinstance(name, str) else None
        if value is None:
            path = supervisor.deployment.credential_files.get(name) if isinstance(name, str) else None
            if path is None:
                _block(report, "required_credential_missing", "credential_files",
                       "A required credential is unavailable.", command)
                good = False
                continue
            try:
                from .instance import _secret
                value = _secret(Path(path))
            except Exception:
                _block(report, "credential_file_invalid", "credential_files",
                       "A configured credential file is unavailable or invalid.", command)
                good = False
                continue
        if not value or any(char.isspace() for char in value):
            _block(report, "credential_environment_invalid", "credential_environment",
                   "A required credential is empty or malformed.", command)
            good = False
            continue
        values.append(value)
    if len(values) != len(external_names) or len(set(values)) != len(values):
        if len(values) == len(external_names) and len(set(values)) != len(values):
            _block(report, "credential_values_not_distinct", "credential_files",
                   "Required credentials must be distinct.", command)
            good = False

    generated_paths: list[tuple[str, Any]] = []
    if isinstance(plan, Mapping):
        for row in plan.get("workers", []):
            if isinstance(row, Mapping):
                generated_paths.append(("credential_files", row.get("auth_token_path")))
        for row in plan.get("executors", []):
            if isinstance(row, Mapping):
                generated_paths.append(("credential_files", row.get("token_path")))
    for field, value in generated_paths:
        if not isinstance(value, str) or not value.strip():
            _block(report, "credential_path_invalid", field,
                   "A managed role credential path is invalid.", command)
            good = False
            continue
        path = Path(value).expanduser()
        try:
            if path.exists():
                from .local import read_or_create_secret
                read_or_create_secret(path, label="managed role", create=False)
            else:
                parent = path.parent
                while not parent.exists() and parent != parent.parent:
                    if parent.is_symlink():
                        raise OSError
                    parent = parent.parent
                if not parent.is_dir() or not os.access(parent, os.R_OK | os.W_OK | os.X_OK):
                    raise OSError
        except Exception:
            _block(report, "credential_path_invalid", field,
                   "A managed role credential path is unavailable or invalid.", command)
            good = False
    if good:
        _stage(report, "credentials", "passed",
               "Required credentials and managed token destinations are available; values are redacted.")
    else:
        _stage(report, "credentials", "blocked",
               "One or more required credentials or managed token destinations are unavailable.",
               field="credential_files")


def _cache_failure_field(error: Exception) -> str:
    prefix = "instance cache directory is not writable: "
    message = str(error)
    if message.startswith(prefix):
        named = message[len(prefix):]
        if named in _CACHE_ENV_NAMES:
            return named
    filename = getattr(error, "filename", None)
    if isinstance(filename, str) and filename:
        failed = os.path.abspath(os.path.expanduser(filename))
        for name in _CACHE_ENV_NAMES:
            configured = os.environ.get(name)
            if not configured:
                continue
            target = os.path.abspath(os.path.expanduser(configured))
            if (failed == target or failed.startswith(target + os.sep)
                    or target.startswith(failed + os.sep)):
                return name
    return next((name for name in _CACHE_ENV_NAMES if os.environ.get(name)), "cache_directories")


def _initialize_cache_stage(report: dict[str, Any], *, command: str) -> bool:
    try:
        from .instance import _initialize_caches
        _initialize_caches()
    except Exception as error:
        field = _cache_failure_field(error)
        message = "A configured runtime cache directory cannot be created or written."
        _stage(report, "cache_directories", "blocked", message, field=field)
        _block(report, "cache_directory_unavailable", field, message, command)
        return False
    _stage(report, "cache_directories", "passed",
           "Configured runtime cache directories are present and writable.")
    return True


def _static_input_stage(supervisor: Any, report: dict[str, Any], *, command: str) -> bool:
    path = getattr(supervisor.deployment, "path", None)
    if not isinstance(path, Path) or not path.is_file():
        _stage(report, "existing_static_check", "skipped",
               "No serialized deployment file is available; the supplied instance plan is used.")
        return True
    static_report: dict[str, Any] = {"checks": [], "blockers": []}
    try:
        from .instance_diagnostics import _Issue, _preflight, _record, safe_failure
        _preflight(static_report, path)
    except Exception as error:
        from .instance_diagnostics import _Issue, _record, safe_failure
        failure = safe_failure(error, operation="doctor", deployment_path=path)
        issue = _Issue(failure["reason_code"], failure.get("field"))
        _record(static_report, issue.code, "blocked", failure["message"],
                field=issue.field, failure=failure)
    blockers = static_report.get("blockers")
    if isinstance(blockers, list) and blockers:
        for blocker in blockers:
            if not isinstance(blocker, Mapping):
                continue
            _block(report,
                   str(blocker.get("reason_code") or "static_input_invalid"),
                   str(blocker.get("field") or "deployment"),
                   str(blocker.get("message") or "Existing static input checks reported a blocker."),
                   str(blocker.get("next_command") or command))
        _stage(report, "existing_static_check", "blocked",
               "Existing read-only static input checks reported a blocker.", field="deployment")
        return False
    _stage(report, "existing_static_check", "passed",
           "Existing read-only static checks validated deployment inputs, credentials, and executable paths.")
    return True


def _storage_stage(supervisor: Any, plan: Mapping[str, Any], runtime: Mapping[str, Any],
                   report: dict[str, Any], *, command: str) -> bool:
    try:
        from .instance_storage import preflight_storage

        storage = preflight_storage(supervisor.deployment, plan, runtime)
    except Exception as error:
        storage = {
            "schema": "qwen-rl-instance-storage-preflight/v1",
            "status": "blocked",
            "errors": [{"reason_code": "storage_capacity_unknown", "field": "storage",
                        "message": f"Storage metadata could not be checked ({type(error).__name__})."}],
        }
    report["storage"] = storage
    errors = storage.get("errors") if isinstance(storage, Mapping) else None
    if storage.get("status") != "passed":
        if isinstance(errors, list) and errors:
            for item in errors:
                if not isinstance(item, Mapping):
                    continue
                _block(report,
                       str(item.get("reason_code") or "storage_preflight_blocked"),
                       str(item.get("field") or "storage"),
                       str(item.get("message") or "Storage capacity could not be established."),
                       command)
        else:
            _block(report, "storage_capacity_unknown", "storage",
                   "Storage capacity could not be established.", command)
        _stage(report, "storage_capacity", "blocked",
               "Startup disk requirements do not fit or could not be established.", field="storage")
        return False
    _stage(report, "storage_capacity", "passed",
           "Packed-cache creation and checkpoint temporary writes fit with per-filesystem headroom.")
    return True


def preflight(supervisor: Any, *, plan: Mapping[str, Any] | None = None, progress=None) -> dict[str, Any]:
    """Run CPU startup checks for an instance supervisor.

    A live deployment returns ``reused`` immediately so callers can preserve the
    existing workers and checkpoint writer. A stopped deployment is checked
    without model loading, CUDA initialization, GPU work, or model-file hashing.
    Embedded tokenizer resources may be materialized at the normal app path.
    """
    from .instance import configure_cache_environment
    configure_cache_environment(supervisor.deployment)
    command = _command(supervisor)
    report: dict[str, Any] = _PreflightReport({
        "schema": _SCHEMA,
        "status": "blocked",
        "stages": [],
        "errors": [],
        "dependency_conflicts": [],
        "next_command": None,
        "scope": "cpu_only_no_model_load_no_cuda_no_model_payload_hash",
    }, progress=progress)

    try:
        from .instance_migration import assert_not_migrating
        assert_not_migrating(supervisor.deployment)
    except Exception:
        _stage(report, "migration_fence", "blocked",
               "A migration fence blocks this deployment.", field="state_dir")
        _block(report, "migration_in_progress", "state_dir",
               "A migration fence blocks CPU preflight.", command)
        return report
    _stage(report, "migration_fence", "passed", "No active migration fence was found.")

    try:
        current = supervisor.status()
        state = current.get("status") if isinstance(current, Mapping) else None
    except Exception:
        _stage(report, "instance_status", "blocked",
               "The existing instance status could not be read.", field="state_dir")
        _block(report, "instance_status_failed", "state_dir",
               "The existing instance status could not be read.", command)
        return report
    if state == "running":
        _stage(report, "instance_status", "reused",
               "The managed instance is already running; existing workers are preserved.")
        report["status"] = "reused"
        return report
    if state != "stopped":
        _stage(report, "instance_status", "blocked",
               "The instance is not in a safe stopped state for startup.", field="state_dir")
        _block(report, "instance_not_stopped", "state_dir",
               "The instance is not in a safe stopped state for startup.", command)
        return report
    _stage(report, "instance_status", "passed", "The managed instance is stopped.")

    from .instance_source_package import source_package_preflight
    source_package = source_package_preflight(Path(__file__).resolve().parents[2])
    report["source_package"] = source_package
    _stage(report, "source_package", source_package["status"],
           "Required runtime source files are present." if source_package["status"] == "passed"
           else "The runtime source package is incomplete.")
    if source_package["status"] != "passed":
        for issue in source_package["errors"]:
            _block(report, issue["reason_code"], issue["field"], issue["message"], command)
        return report
    static_ok = _static_input_stage(supervisor, report, command=command)
    effective_plan: Mapping[str, Any] | None = plan
    if not isinstance(effective_plan, Mapping):
        try:
            effective_plan = supervisor.prepare()
            _stage(report, "instance_prepare", "passed",
                   "Existing CPU-only prepare validated the topology and runtime configuration.")
        except Exception:
            _stage(report, "instance_prepare", "blocked",
                   "Existing CPU-only prepare rejected the deployment.", field="runtime_config")
            _block(report, "instance_prepare_failed", "runtime_config",
                   "Existing CPU-only prepare rejected the deployment.", command)
            return report
    elif static_ok:
        _stage(report, "instance_prepare", "passed",
               "The caller supplied the plan produced by existing instance prepare.")

    if not isinstance(effective_plan, Mapping):
        _block(report, "instance_prepare_failed", "runtime_config",
               "The prepared runtime plan is unavailable.", command)
        return report
    runtime = effective_plan.get("runtime")
    if not isinstance(runtime, Mapping):
        _stage(report, "runtime_config", "blocked",
               "The prepared runtime configuration is unavailable.", field="runtime_config")
        _block(report, "runtime_config_invalid", "runtime_config",
               "The prepared runtime configuration is unavailable.", command)
        return report
    _stage(report, "runtime_config", "passed", "Prepared runtime settings are available.")

    _check_paths(report, runtime, supervisor, command=command)
    _storage_stage(supervisor, effective_plan, runtime, report, command=command)
    _check_credentials(report, runtime, supervisor, effective_plan, command=command)
    if report["errors"]:
        return report

    from .teacher_readiness import teacher_provider_status
    _stage(report, "teacher_provider", "running", "Checking the non-billable Teacher model catalog.")
    provider = teacher_provider_status(runtime, supervisor.deployment.credential_files)
    report["teacher_provider"] = provider
    _stage(report, "teacher_provider", provider["status"],
           "Teacher provider readiness checked without submitting a completion.")
    if provider["status"] == "blocked":
        _block(report, provider["reason_code"], "bepa.teacher.base_url",
               "The enabled Teacher provider is unavailable; restore its authenticated model catalog before startup.", command)
        return report

    deployment = supervisor.deployment
    executables = (("python", deployment.python), ("worker_python", deployment.worker_python),
                   ("engine_binary", deployment.engine_binary))
    executable_ok = True
    for field, value in executables:
        try:
            usable = isinstance(value, str) and Path(value).is_file() and os.access(value, os.X_OK)
        except (OSError, RuntimeError, ValueError):
            usable = False
        if not usable:
            executable_ok = False
            _block(report, "executable_unavailable", field,
                   "A configured runtime executable is missing or inaccessible.", command)
    if executable_ok:
        _stage(report, "runtime_executables", "passed",
               "Configured manager Python, Worker Python, and NInfer executable are accessible.")
    else:
        _stage(report, "runtime_executables", "blocked",
               "One or more configured runtime executables are unavailable.")

    cache_ok = _initialize_cache_stage(report, command=command) if executable_ok else False
    if executable_ok and cache_ok:
        env = _child_environment()
        pip_checks: dict[str, bool] = {}
        dataset_checks: dict[str, str] = {}
        probes: list[dict[str, Any]] = []
        reused_datasets: list[tuple[str, str]] = []
        for role, python in (("manager_python", deployment.python),
                             ("worker_python", deployment.worker_python)):
            field = "python" if role == "manager_python" else "worker_python"
            interpreter_key = _absolute_python_path(python)
            if interpreter_key in pip_checks:
                pip_ok = pip_checks[interpreter_key]
                _stage(report, f"{role}_pip_check", "reused" if pip_ok else "blocked",
                       "The same configured Python path already passed pip check."
                       if pip_ok else "The same configured Python path already failed pip check.",
                       field=field)
            else:
                pip_ok = _record_pip_check(
                    report, stage=f"{role}_pip_check", field=field,
                    python=python, env=env,
                )
                pip_checks[interpreter_key] = pip_ok
            if not pip_ok:
                continue
            if role == "manager_python":
                probes.append(dict(stage="pinned_runtime_interfaces", field="python",
                              python=python, code=_MANAGER_PROBE,
                              arguments=(deployment.engine_binary,), env=env))
                model = runtime.get("model") if isinstance(runtime.get("model"), Mapping) else {}
                identity_path = (model.get("artifact_identity_path")
                                 or runtime.get("artifact_identity_path") or "")
                tokenizer_path = _tokenizer_path(runtime)
                probes.append(dict(stage="tokenizer_load", field="tokenizer_path",
                              python=python, code=_TOKENIZER_PROBE,
                              arguments=(str(identity_path), str(tokenizer_path or "")), env=env))
            else:
                probes.append(dict(stage="worker_runtime_interfaces", field="worker_python",
                              python=python, code=_WORKER_PROBE, arguments=(), env=env))
            dataset_stage = f"{role}_arc_task_data"
            if interpreter_key in dataset_checks:
                reused_datasets.append((dataset_stage, dataset_checks[interpreter_key]))
            else:
                dataset_checks[interpreter_key] = dataset_stage
                probes.append(dict(
                    stage=dataset_stage, field="dataset_dir",
                    python=python, code=_DATASET_PROBE,
                    arguments=(str(runtime.get("dataset_dir") or ""),), env=env,
                ))
        outcomes = _parallel_probes(report, probes)
        for alias, original in reused_datasets:
            ok = outcomes[original]
            _stage(report, alias, "reused" if ok else "blocked",
                   "The same configured interpreter already checked this dataset.", field="dataset_dir")

    report["status"] = "blocked" if report["errors"] else "passed"
    return report
