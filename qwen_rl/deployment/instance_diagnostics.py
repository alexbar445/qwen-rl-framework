"""Read-only operator diagnostics for one managed instance."""
from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import yaml

_SCHEMA = "qwen-rl-instance-diagnostics/v1"
_OPERATOR_SCHEMA = "qwen-rl-instance-operator-status/v1"
_MAX_JSON = 1024 * 1024
_MAX_YAML = 2 * 1024 * 1024
_REQUIRED_KEYS = {"runtime_config", "state_dir", "gpus", "engine_binary", "worker_python"}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FIELDS = {
    "deployment", "deployment.schema", "deployment.runtime_config", "deployment.gpus",
    "deployment.state_dir", "deployment.engine_binary", "deployment.worker_python",
    "deployment.python", "deployment.credential_files", "deployment.gateway_config", "runtime_config",
    "runtime_config_or_gpus", "model_path", "dataset_dir", "artifact_identity_path",
    "worker_inference.ninfer_conversion_report", "learner.initial_adapter_path", "gpus",
    "operator.action", "state_dir.active_manifest", "output_dir", "conversion_report",
    "python", "native_binary", "install_root", "state_dir", "credential_files",
    "credential_environment", "learner.gradient_tasks_per_worker", "rollout.group_size",
}
_FIELD_LABELS = {field: field for field in _FIELDS}
_FIELD_LABELS.update({"deployment": "deployment JSON", "artifact_identity_path": "model.artifact_identity_path",
                      "runtime_config_or_gpus": "runtime_config or gpus",
                      "state_dir.active_manifest": "state_dir active manifest"})
_COMMANDS = {
    "check": "python -m qwen_rl instance --deployment <deployment.json> check",
    "status": "python -m qwen_rl instance --deployment <deployment.json> status --summary",
    "watch": "python -m qwen_rl instance --deployment <deployment.json> watch --details --once",
    "serve": "python -m qwen_rl instance --deployment <deployment.json> serve",
    "gpu_inventory": "nvidia-smi",
    "setup": "python -m qwen_rl instance setup --interactive --output-dir <new-config-dir>",
    "resume_plan": "python -m qwen_rl instance --deployment <deployment.json> resume-plan --inspect-checkpoint --json",
}
_MESSAGES = {
    "unknown_failure": "Instance operation failed; sensitive error details are suppressed.",
    "setup_input_invalid": "A setup input is invalid. Correct the named input and select a new configuration directory.",
    "deployment_file_missing": "The deployment JSON file is missing.",
    "deployment_file_invalid": "The deployment path is not a readable regular JSON file.",
    "deployment_json_invalid": "The deployment file is not valid JSON.",
    "deployment_schema_invalid": "The deployment schema is unsupported.",
    "deployment_field_missing": "A required deployment field is missing.",
    "deployment_field_invalid": "A deployment field has an invalid value.",
    "runtime_path_invalid": "A configured runtime path field has an invalid value.",
    "runtime_config_missing": "The configured runtime YAML file is missing.",
    "runtime_config_invalid": "The runtime YAML is invalid or unsupported.",
    "required_file_missing": "A configured model, data, identity, or credential path is missing.",
    "required_path_unavailable": "A configured required path could not be inspected.",
    "required_credential_missing": "A required credential is absent from the environment and has no configured file.",
    "credential_environment_invalid": "A required credential environment variable is present but empty or contains whitespace; the manager will not fall back to a file.",
    "credential_file_metadata_invalid": "A configured credential file fails the manager's private-file metadata checks.",
    "credential_file_unavailable": "A configured credential file could not be inspected.",
    "executable_unavailable": "A configured runtime executable is missing or is not executable.",
    "configured_gpu_missing": "One or more configured GPU indices are absent from local inventory.",
    "instance_check_failed": "The existing check could not validate runtime configuration or GPU inventory.",
    "instance_status_failed": "The existing read-only instance status could not be read.",
    "operator_blocked": "Existing instance metadata reports an actionable blocker.",
}


def _with_gpu_override(command: str, gpus: tuple[int, ...] | None) -> str:
    if (gpus and isinstance(gpus, tuple) and len(set(gpus)) == len(gpus)
            and all(type(gpu) is int and gpu >= 0 for gpu in gpus)):
        return command.replace(" instance ", " instance --gpus " + ",".join(map(str, gpus)) + " ", 1)
    return command


def _command_for(name: str, deployment_path: Path | None = None,
                 gpus: tuple[int, ...] | None = None) -> str:
    command = _with_gpu_override(_COMMANDS[name], gpus)
    if deployment_path is None or "<deployment.json>" not in command:
        return command
    try:
        path = Path(deployment_path).expanduser().resolve(strict=True)
        text = str(path)
        if not path.is_file() or re.fullmatch(r"[A-Za-z0-9_./ -]+", text) is None:
            return command
    except (OSError, ValueError, RuntimeError):
        return command
    return command.replace("<deployment.json>", shlex.quote(text))


def _safe_command(command: Any) -> bool:
    if not isinstance(command, str):
        return False
    if command in _COMMANDS.values():
        return True
    try:
        values = shlex.split(command)
        if len(values) > 5 and values[4] == "--gpus":
            if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", values[5]) is None:
                return False
            del values[4:6]
        for template in _COMMANDS.values():
            if "<deployment.json>" not in template:
                continue
            expected = shlex.split(template)
            index = expected.index("<deployment.json>")
            if (len(values) == len(expected) and values[:index] == expected[:index]
                    and values[index + 1:] == expected[index + 1:]
                    and values[index].startswith("/")
                    and re.fullmatch(r"[A-Za-z0-9_./ -]+", values[index]) is not None):
                return True
    except ValueError:
        pass
    return False


class _Issue(Exception):
    """Typed diagnostic failure with safe, allowlisted fields only."""

    def __init__(self, code: str, field: str | None = None):
        super().__init__(code)
        self.code, self.field = code, field


def safe_failure(
    error: Exception,
    *,
    operation: str,
    deployment_path: Path | None,
) -> dict[str, Any]:
    """Return a safe failure projection; never expose exception text or file paths."""
    code, field = "unknown_failure", None
    if isinstance(error, _Issue) and error.code in _MESSAGES:
        code, field = error.code, error.field if error.field in _FIELDS else None
    elif (type(error).__module__ in {"qwen_rl.deployment.instance_setup", "qwen_rl.deployment.autodl_config"}
          and operation == "setup"):
        code = "setup_input_invalid"
        field = getattr(error, "field", None)
        field = field if field in _FIELDS else "runtime_config"
    elif (type(error).__module__ == "qwen_rl.deployment.instance_plan"
          and type(error).__name__ == "InstanceDeploymentError"):
        code, field = (("instance_status_failed", "state_dir.active_manifest") if operation == "status"
                       else ("instance_check_failed", "runtime_config"))
    elif isinstance(error, yaml.YAMLError) and operation in {"doctor", "check"}:
        code, field = "runtime_config_invalid", "runtime_config"
    elif isinstance(error, json.JSONDecodeError) and operation in {"doctor", "check"}:
        code, field = "deployment_json_invalid", "deployment"
    elif isinstance(error, FileNotFoundError) and deployment_path is not None:
        try:
            if not Path(deployment_path).expanduser().exists():
                code, field = "deployment_file_missing", "deployment"
        except (OSError, RuntimeError, ValueError):
            pass
    if code == "unknown_failure" and isinstance(error, ValueError) and operation in {"doctor", "check"}:
        code, field = "runtime_config_invalid", "runtime_config"
    command = _command_for("setup" if operation == "setup" else
                           "status" if operation == "status" else "check", deployment_path)
    if code == "configured_gpu_missing":
        command = _COMMANDS["gpu_inventory"]
    failure = {"reason_code": code, "message": _MESSAGES[code], "next_command": command}
    if field:
        failure["field"] = field
    return failure


def _load_instance_api():
    """Import the existing instance implementation only when doctor is run."""
    from .instance import InstanceDeploymentError, InstanceSupervisor, load_deployment
    return InstanceDeploymentError, InstanceSupervisor, load_deployment


def _read_deployment(path: Path) -> tuple[dict[str, Any], Path]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        info = resolved.stat()
    except FileNotFoundError:
        raise _Issue("deployment_file_missing", "deployment") from None
    except (OSError, RuntimeError, ValueError):
        raise _Issue("deployment_file_invalid", "deployment") from None
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_JSON:
        raise _Issue("deployment_file_invalid", "deployment")
    try:
        with resolved.open("rb") as stream:
            raw = json.loads(stream.read(_MAX_JSON + 1).decode("utf-8"))
    except Exception:
        raise _Issue("deployment_json_invalid", "deployment") from None
    if not isinstance(raw, dict):
        raise _Issue("deployment_json_invalid", "deployment")
    if raw.get("schema") != "qwen-rl-instance/v1":
        raise _Issue("deployment_schema_invalid", "deployment.schema")
    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise _Issue("deployment_field_missing", f"deployment.{sorted(missing)[0]}")
    from .instance_plan import _TOP_KEYS
    if set(raw) - _TOP_KEYS:
        raise _Issue("deployment_field_invalid", "deployment")
    return raw, resolved


def _resolve_path(value: Any, *, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        code = "deployment_field_invalid" if field.startswith("deployment.") else "runtime_path_invalid"
        raise _Issue(code, field)
    path = Path(value).expanduser()
    try:
        return (path if path.is_absolute() else base / path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise _Issue("required_path_unavailable", field) from None


def _record(
    report: dict[str, Any], code: str, status: str, message: str,
    *, field: str | None = None, failure: dict[str, Any] | None = None,
) -> None:
    check: dict[str, Any] = {"code": code, "status": status, "message": message}
    if field in _FIELDS:
        check["field"] = field
    if failure is not None:
        check["reason_code"] = failure["reason_code"]
        check["next_command"] = failure["next_command"]
        report["blockers"].append(failure)
    report["checks"].append(check)


def _check_path(report: dict[str, Any], path: Path, field: str, code: str, *, directory: bool = False) -> None:
    try:
        info = path.stat()
        present = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        missing = False
    except FileNotFoundError:
        present, missing = False, True
    except OSError:
        present, missing = False, False
    if present:
        _record(report, code, "passed", "Configured path is present; contents were not inspected.", field=field)
        return
    failure = safe_failure(_Issue("required_file_missing" if missing else "required_path_unavailable", field),
                           operation="doctor", deployment_path=None)
    _record(report, code, "blocked", failure["message"], field=field, failure=failure)


def _check_executable(report: dict[str, Any], value: Any, field: str) -> None:
    try:
        executable = isinstance(value, str) and bool(value) and Path(value).is_file() and os.access(value, os.X_OK)
    except (OSError, RuntimeError, ValueError):
        executable = False
    if executable:
        _record(report, "runtime_executable", "passed", "Configured runtime executable exists and is executable.", field=field)
        return
    failure = safe_failure(_Issue("executable_unavailable", field), operation="doctor", deployment_path=None)
    _record(report, "runtime_executable", "blocked", failure["message"], field=field, failure=failure)


def _check_credential_file(report: dict[str, Any], value: Any, base: Path) -> None:
    field = "credential_files"
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        failure = safe_failure(_Issue("credential_file_metadata_invalid", field),
                               operation="doctor", deployment_path=None)
        _record(report, "credential_file", "blocked", failure["message"], field=field, failure=failure)
        return
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        info = path.lstat()
    except FileNotFoundError:
        failure = safe_failure(_Issue("required_file_missing", field), operation="doctor", deployment_path=None)
        _record(report, "credential_file", "blocked", failure["message"], field=field, failure=failure)
        return
    except (OSError, RuntimeError, ValueError):
        failure = safe_failure(_Issue("credential_file_unavailable", field),
                               operation="doctor", deployment_path=None)
        _record(report, "credential_file", "blocked", failure["message"], field=field, failure=failure)
        return
    acceptable = (
        stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) & 0o077 == 0
        and info.st_nlink == 1 and 0 < info.st_size <= 16384
    )
    if not acceptable:
        failure = safe_failure(_Issue("credential_file_metadata_invalid", field),
                               operation="doctor", deployment_path=None)
        _record(report, "credential_file", "blocked", failure["message"], field=field, failure=failure)
        return
    _record(report, "credential_file", "metadata_only",
            "Private file metadata matches manager requirements; credential contents were not read.", field=field)


def _check_required_credentials(
    report: dict[str, Any], runtime: Mapping[str, Any], configured: Mapping[str, Any], base: Path,
) -> None:
    names = ["ARC_ENV_TOKEN"]
    bepa = runtime.get("bepa") if isinstance(runtime.get("bepa"), dict) else {}
    if bepa.get("enabled") is True:
        teacher = bepa.get("teacher") if isinstance(bepa.get("teacher"), dict) else {}
        names.extend((teacher.get("api_key_env", "DEEPSEEK_API_KEY"),
                      teacher.get("token_env", "BEPA_TEACHER_TOKEN")))
    for name in dict.fromkeys(names):
        if not isinstance(name, str) or _ENV_NAME.fullmatch(name) is None:
            raise _Issue("runtime_config_invalid", "runtime_config")
        if name in os.environ:
            value = os.environ[name]
            if not value or any(char.isspace() for char in value):
                failure = safe_failure(_Issue("credential_environment_invalid", "credential_environment"),
                                       operation="doctor", deployment_path=None)
                _record(report, "credential_environment", "blocked", failure["message"],
                        field=failure.get("field"), failure=failure)
            else:
                _record(report, "credential_environment", "passed",
                        "Required credential is present in the environment; value redacted.",
                        field="credential_environment")
            continue
        if name not in configured:
            failure = safe_failure(_Issue("required_credential_missing", "credential_files"),
                                   operation="doctor", deployment_path=None)
            _record(report, "credential_file", "blocked", failure["message"],
                    field=failure.get("field"), failure=failure)
        else:
            _check_credential_file(report, configured[name], base)


def _preflight(report: dict[str, Any], deployment_path: Path) -> tuple[dict[str, Any], Path]:
    raw, resolved = _read_deployment(deployment_path)
    _record(report, "deployment_json", "passed", "Deployment JSON uses the supported schema.", field="deployment")
    base = resolved.parent
    runtime_path = _resolve_path(raw.get("runtime_config"), base=base, field="deployment.runtime_config")
    if raw.get("gateway_config") is not None:
        path = _resolve_path(raw["gateway_config"], base=base, field="deployment.gateway_config")
        _check_path(report, path, "deployment.gateway_config", "gateway_config_file")
    credentials = raw.get("credential_files", {})
    if not isinstance(credentials, dict):
        raise _Issue("deployment_field_invalid", "deployment.credential_files")

    try:
        info = runtime_path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_YAML:
            raise _Issue("runtime_config_invalid", "runtime_config")
        with runtime_path.open("rb") as stream:
            runtime = yaml.safe_load(stream.read(_MAX_YAML + 1).decode("utf-8"))
    except _Issue:
        raise
    except FileNotFoundError:
        raise _Issue("runtime_config_missing", "runtime_config") from None
    except Exception:
        raise _Issue("runtime_config_invalid", "runtime_config") from None
    if not isinstance(runtime, dict):
        raise _Issue("runtime_config_invalid", "runtime_config")
    _record(report, "runtime_config", "passed", "Runtime YAML parsed; semantic checks use the existing instance check.",
            field="runtime_config")

    _check_executable(report, raw.get("python", sys.executable), "deployment.python")
    _check_executable(report, raw.get("worker_python"), "deployment.worker_python")
    _check_executable(report, raw.get("engine_binary"), "deployment.engine_binary")
    _check_required_credentials(report, runtime, credentials, base)

    from qwen_rl.paths import ROOT
    model = runtime.get("model") if isinstance(runtime.get("model"), dict) else {}
    worker = runtime.get("worker_inference") if isinstance(runtime.get("worker_inference"), dict) else {}
    learner = runtime.get("learner") if isinstance(runtime.get("learner"), dict) else {}
    paths = (
        ("model_path", runtime.get("model_path"), False),
        ("dataset_dir", runtime.get("dataset_dir"), True),
        ("artifact_identity_path", model.get("artifact_identity_path", runtime.get("artifact_identity_path")), False),
        ("worker_inference.ninfer_conversion_report", worker.get("ninfer_conversion_report"), False),
        ("learner.initial_adapter_path", learner.get("initial_adapter_path"), True),
    )
    for field, value, is_dir in paths:
        if value is not None:
            path = _resolve_path(value, base=ROOT, field=field)
            _check_path(report, path, field, field, directory=is_dir)
    return raw, runtime_path


def diagnose_instance(deployment_path: Path, *, gpus: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Run existing read-only status/check paths plus bounded file-presence checks."""
    deployment_path = Path(deployment_path)
    report: dict[str, Any] = {
        "schema": _SCHEMA, "status": "blocked", "scope": "read_only_static_validation_and_metadata",
        "checks": [], "blockers": [], "operator": None,
        "not_performed": [
            "CUDA runtime initialization or GPU compute smoke", "Worker/provider health polling",
            "model load or checkpoint restore", "checkpoint/model payload read or SHA-256 scan",
            "file-backed credential contents or token validity",
            "paid API requests, service start/stop, training submission, or state writes",
        ],
        "next_command": None,
    }
    try:
        _preflight(report, deployment_path)
    except _Issue as issue:
        failure = safe_failure(issue, operation="doctor", deployment_path=deployment_path)
        _record(report, issue.code, "blocked", failure["message"], field=failure.get("field"), failure=failure)
        report["next_command"] = failure["next_command"]
        return report

    try:
        from dataclasses import replace
        InstanceDeploymentError, InstanceSupervisor, load_deployment = _load_instance_api()
        deployment = load_deployment(deployment_path)
        if gpus is not None:
            deployment = replace(deployment, gpus=tuple(gpus))
        supervisor = InstanceSupervisor(deployment)
    except Exception:
        issue = _Issue("deployment_field_invalid", "deployment")
        failure = safe_failure(issue, operation="doctor", deployment_path=deployment_path)
        _record(report, "deployment_validation", "blocked", failure["message"],
                field=failure.get("field"), failure=failure)
        report["next_command"] = failure["next_command"]
        return report

    operator = None
    try:
        status = supervisor.status()
        if isinstance(status, Mapping) and isinstance(status.get("operator"), Mapping):
            operator = dict(status["operator"])
        _record(report, "instance_status", "passed", "Existing read-only status query completed.")
    except Exception:
        failure = safe_failure(_Issue("instance_status_failed", "state_dir.active_manifest"),
                               operation="doctor", deployment_path=deployment_path)
        _record(report, "instance_status", "blocked", failure["message"], field=failure.get("field"),
                failure=failure)
    try:
        checked = supervisor.prepare()
        if isinstance(checked, Mapping) and isinstance(checked.get("operator"), Mapping):
            operator = dict(checked["operator"])
        _record(report, "instance_check", "passed", "Existing check validated topology and GPU inventory.",
                field="runtime_config")
    except Exception as error:
        if isinstance(error, InstanceDeploymentError):
            try:
                from .instance_plan import _gpu_inventory
                inventory = _gpu_inventory(None)
                code, field = (("configured_gpu_missing", "gpus") if set(deployment.gpus) - set(inventory)
                               else ("instance_check_failed", "runtime_config"))
            except Exception:
                code, field = "instance_check_failed", "runtime_config_or_gpus"
            failure = safe_failure(_Issue(code, field), operation="doctor", deployment_path=deployment_path)
        else:
            failure = safe_failure(error, operation="doctor", deployment_path=deployment_path)
        _record(report, "instance_check", "blocked", failure["message"],
                field=failure.get("field"), failure=failure)

    if operator is not None:
        report["operator"] = operator
        action = operator.get("action")
        if isinstance(action, Mapping) and action.get("blocked") is True:
            failure = safe_failure(_Issue("operator_blocked", "operator.action"),
                                   operation="doctor", deployment_path=deployment_path)
            _record(report, "operator_action", "blocked", failure["message"],
                    field=failure.get("field"), failure=failure)
    report["status"] = "blocked" if report["blockers"] else "passed"
    report["next_command"] = report["blockers"][0]["next_command"] if report["blockers"] else _command_for("resume_plan", deployment_path, gpus)
    return report


def _safe(value: Any, values: set[str], default: str = "unknown") -> str:
    return value if isinstance(value, str) and value in values else default


def _operator_blocked_reason(operator: Mapping[str, Any]) -> tuple[str, str] | None:
    action = operator.get("action")
    if not isinstance(action, Mapping) or action.get("blocked") is not True:
        return None
    reasons = {
        "owned_instance_process_state_requires_inspection": "The instance process state needs inspection.",
        "checkpoint_metadata_or_transaction_sidecar_is_invalid": "Checkpoint metadata or a transaction sidecar is invalid.",
        "checkpoint_transaction_sidecar_remains_without_a_live_owner": "A checkpoint transaction is pending without a live owner.",
    }
    reason = action.get("blocked_reason")
    text = reasons.get(reason) if isinstance(reason, str) else None
    return text or "Existing instance metadata reports an actionable blocker.", _COMMANDS["status"]


def format_operator_summary(operator: dict[str, Any], *, deployment_path: Path | None = None) -> str:
    """Render only known fields from the existing operator summary schema."""
    if not isinstance(operator, dict) or operator.get("schema") != _OPERATOR_SCHEMA:
        return "Operator summary: unavailable (unsupported schema)."
    process = operator.get("process") if isinstance(operator.get("process"), Mapping) else {}
    process_state = _safe(process.get("state"), {"stopped", "running", "failed", "orphaned", "stop_incomplete"})
    lines = [f"Process: {process_state}."]
    reason = _operator_blocked_reason(operator)
    if reason:
        lines.extend((f"Blocker: {reason[0]}", f"Next command: {reason[1]}"))
    else:
        lines.append("Action: no actionable blocker is reported by existing metadata.")

    action = operator.get("action") if isinstance(operator.get("action"), Mapping) else {}
    next_steps = {
        "review_resume_receipt_then_use_existing_instance_start_path": ("Review the resume receipt before managed startup.", _COMMANDS["status"]),
        "use_existing_watch_for_current_worker_readiness_and_logical_progress": ("Use existing watch for current readiness and progress.", _COMMANDS["watch"]),
        "use_existing_instance_start_path_with_managed_restore_guards": ("Review the resume plan before managed startup.", _command_for("resume_plan", deployment_path)),
    }
    step = next_steps.get(action.get("next_action")) if isinstance(action.get("next_action"), str) else None
    if not reason and step:
        lines.extend((f"Next step: {step[0]}", f"Next command: {step[1]}"))

    deployment = operator.get("deployment") if isinstance(operator.get("deployment"), Mapping) else {}
    effective = deployment.get("effective") if isinstance(deployment.get("effective"), Mapping) else {}
    lines.append(f"Effective runtime metadata: {_safe(effective.get('state'), {'readable_metadata', 'invalid', 'unknown'})}.")
    workers = operator.get("workers") if isinstance(operator.get("workers"), Mapping) else {}
    ack, expected = workers.get("startup_policy_ack_count"), workers.get("expected_count")
    counts = f"{ack}/{expected}" if all(type(n) is int and n >= 0 for n in (ack, expected)) else "unknown"
    startup = _safe(workers.get("startup_readiness"), {"verified", "partial", "unknown"})
    lines.append(f"Workers: startup acknowledgements {startup} ({counts}); current readiness unknown.")
    checkpoint = operator.get("checkpoint") if isinstance(operator.get("checkpoint"), Mapping) else {}
    state = _safe(checkpoint.get("state"), {"metadata_consistent", "pending", "invalid", "missing", "unknown"})
    lines.append(f"Checkpoint: {state}; restore validation not_performed.")
    logical = operator.get("logical_run") if isinstance(operator.get("logical_run"), Mapping) else {}
    run_id = logical.get("run_id")
    safe_run_id = run_id if isinstance(run_id, str) and re.fullmatch(r"[a-f0-9]{32}", run_id) else "unknown"
    logical_state = _safe(logical.get("state"), {"pending_at_resume_acceptance", "complete_at_resume_acceptance", "none_at_resume_acceptance", "unknown"})
    remaining = logical.get("remaining_steps")
    remaining_text = str(remaining) if type(remaining) is int and remaining >= 0 else "unknown"
    lines.append(f"Logical run: {safe_run_id}; {logical_state}; remaining at resume acceptance {remaining_text} (not live progress).")
    durable = operator.get("durable_optimizer_advancement") if isinstance(operator.get("durable_optimizer_advancement"), Mapping) else {}
    version, updates = durable.get("durable_policy_version"), durable.get("durable_optimizer_steps")
    version_text = str(version) if type(version) is int and version >= 0 else "unknown"
    updates_text = str(updates) if type(updates) is int and updates >= 0 else "unknown"
    lines.append(f"Durable pointer counters: policy {version_text}, optimizer updates {updates_text} (not new updates since this query).")
    lines.append("Running image identity: unknown.")
    return "\n".join(lines)


def format_diagnostics(report: dict[str, Any]) -> str:
    """Format doctor output using allowlisted codes and fields, never raw messages."""
    if not isinstance(report, dict) or report.get("schema") != _SCHEMA:
        return "Instance doctor: report unavailable (unsupported schema)."
    status = _safe(report.get("status"), {"passed", "blocked"})
    lines = [f"Instance doctor: {status.upper()} (read-only static validation and metadata)."]
    blockers = report.get("blockers")
    if isinstance(blockers, list) and blockers:
        lines.append("Blockers:")
        for item in blockers:
            if not isinstance(item, Mapping):
                continue
            code, field = item.get("reason_code"), item.get("field")
            message = _MESSAGES.get(code, "An actionable blocker was reported.") if isinstance(code, str) else "An actionable blocker was reported."
            suffix = f" Field: {_FIELD_LABELS[field]}." if isinstance(field, str) and field in _FIELD_LABELS else ""
            lines.append(f"- {message}{suffix}")
            command = item.get("next_command")
            if _safe_command(command):
                lines.append(f"  Next command: {command}")
    else:
        lines.append("No actionable blocker was observed by these checks.")
    checks = report.get("checks")
    if isinstance(checks, list):
        if any(isinstance(check, Mapping) and check.get("code") == "credential_file"
               and check.get("status") == "metadata_only" for check in checks):
            lines.append("Credential files: private metadata checked; contents and token validity not checked.")
        if any(isinstance(check, Mapping) and check.get("code") == "credential_environment"
               and check.get("status") == "passed" for check in checks):
            lines.append("Credential environment: required variable presence and whitespace checked; value redacted.")
    operator = report.get("operator")
    if isinstance(operator, dict):
        lines.append("Existing operator summary:")
        lines.extend(f"  {line}" for line in format_operator_summary(operator).splitlines())
    command = report.get("next_command")
    if _safe_command(command):
        lines.append(f"Next command: {command}")
    lines.append("Not performed: CUDA compute smoke, Worker/provider health polling, model load/restore, payload SHA scans, file-backed credential contents, paid API requests, service lifecycle, training submission, or state writes.")
    return "\n".join(lines)
