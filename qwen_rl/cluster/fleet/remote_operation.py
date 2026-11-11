"""Pinned-SSH adapter for owned, detached, long-running node commands."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import ssh
from .remote_operation_worker import SCHEMA, _embedded_source

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SECRET = re.compile(r"(?i)(?:bearer\s+[a-z0-9._~-]{8,}|(?:password|passwd|secret|token|credential|api[_-]?key|authorization)\s*[:=]\s*[^\s]+)")
_SECRET_FLAG = re.compile(r"(?i)^--?(?:password|passwd|secret|token|credential|api[_-]?key|authorization)(?:-(?:file|path)|_(?:file|path))?(?:$|=)")


class FleetRemoteOperationError(RuntimeError):
    """Sanitized Fleet remote operation failure with a stable machine code."""

    def __init__(self, code: str, message: str | None = None):
        self.code = code
        super().__init__(message or code)


def _setting(settings: Any, name: str) -> str:
    value = settings.get(name) if isinstance(settings, Mapping) else getattr(settings, name, None)
    if not isinstance(value, str) or not value:
        raise FleetRemoteOperationError("settings_invalid")
    return value


def _node_value(node: Any, name: str) -> str:
    value = node.get(name) if isinstance(node, Mapping) else getattr(node, name, None)
    if not isinstance(value, str) or not value:
        raise FleetRemoteOperationError("node_identity_invalid")
    return value


def _argv(value: list[str]) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 256:
        raise FleetRemoteOperationError("argv_invalid")
    clean: list[str] = []
    consume_file = False
    for item in value:
        if not isinstance(item, str) or not item or len(item) > 8192 or "\x00" in item:
            raise FleetRemoteOperationError("argv_invalid")
        if consume_file:
            if not item.startswith(("/", "./", "../", "~/", "@")):
                raise FleetRemoteOperationError("inline_credential_refused")
            consume_file = False
        elif _SECRET_FLAG.match(item):
            flag, separator, argument = item.partition("=")
            if flag.lower().endswith(("-file", "_file", "-path", "_path")):
                if separator and not argument.startswith(("/", "./", "../", "~/", "@")):
                    raise FleetRemoteOperationError("inline_credential_refused")
                consume_file = not separator
            else:
                raise FleetRemoteOperationError("inline_credential_refused")
        elif _SECRET.search(item):
            raise FleetRemoteOperationError("inline_credential_refused")
        clean.append(item)
    if consume_file:
        raise FleetRemoteOperationError("inline_credential_refused")
    return clean


def _sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _remote(node: Any, settings: Any, meta: dict[str, Any], mode: str) -> dict[str, Any]:
    from .remote_operation_worker import __file__ as worker_file

    source = Path(worker_file).read_text(encoding="utf-8")
    launcher = _embedded_source(source)
    payload = base64.b64encode(json.dumps(meta, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode()).decode("ascii")
    if len(payload) > 60000:
        raise FleetRemoteOperationError("request_too_large")
    state_dir = Path(_setting(settings, "state_dir"))
    identity_file = Path(_setting(settings, "ssh_admin_identity_file"))
    try:
        result = ssh.run_admin_ssh(
            node, ["python3", "-c", launcher, payload, "--" + mode],
            state_dir=state_dir, identity_file=identity_file,
            timeout=30, capture_output=True,
        )
    except Exception as exc:
        # A lost startup reply may follow a successful remote launch. Never retry it here.
        raise FleetRemoteOperationError("blocked", "pinned SSH did not return a verifiable operation receipt") from exc
    text = getattr(result, "stdout", "")
    if not isinstance(text, str) or len(text.encode("utf-8")) > 131072:
        raise FleetRemoteOperationError("remote_receipt_invalid")
    try:
        value = json.loads(text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise FleetRemoteOperationError("remote_receipt_invalid") from exc
    if not isinstance(value, dict):
        raise FleetRemoteOperationError("remote_receipt_invalid")
    if value.get("error_code") is not None:
        code = value["error_code"]
        raise FleetRemoteOperationError(code if isinstance(code, str) else "remote_refused")
    return value


def execute_remote(
    node: Any,
    settings: Any,
    operation_id: str,
    step_id: str,
    argv: list[str],
    *,
    timeout_seconds: int,
    progress: Callable[[dict[str, Any]], Any] | None = None,
    expected_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start or resume one remote action and poll its private status receipt.

    Commands run as direct argv with ``shell=False``. Long Docker CLI invocations
    are unsupported for timeout cleanup: their container may outlive the CLI PID.
    """
    if not _ID.fullmatch(operation_id or "") or not _ID.fullmatch(step_id or ""):
        raise FleetRemoteOperationError("operation_identity_invalid")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 604800:
        raise FleetRemoteOperationError("timeout_invalid")
    command = _argv(argv)
    if expected_json is not None:
        if not isinstance(expected_json, dict):
            raise FleetRemoteOperationError("expected_json_invalid")
        try:
            if len(json.dumps(expected_json, ensure_ascii=False, allow_nan=False).encode()) > 65536:
                raise ValueError
        except (TypeError, ValueError):
            raise FleetRemoteOperationError("expected_json_invalid") from None
    node_id = _node_value(node, "node_id")
    boot_id = _node_value(node, "boot_id")
    argv_sha = _sha(command)
    plan_sha = _sha({"node_id": node_id, "boot_id": boot_id, "operation_id": operation_id,
                     "step_id": step_id, "argv_sha256": argv_sha, "timeout_seconds": timeout_seconds,
                     "expected_json": expected_json})
    meta = {"schema": SCHEMA, "node_id": node_id, "boot_id": boot_id,
            "operation_id": operation_id, "step_id": step_id, "argv": command,
            "argv_sha256": argv_sha, "plan_sha256": plan_sha,
            "timeout_seconds": timeout_seconds, "expected_json": expected_json}
    call_started = time.monotonic()
    state = _remote(node, settings, meta, "inspect")
    if state.get("status") == "missing":
        call_started = time.monotonic()
        state = _remote(node, settings, meta, "start")
    local_deadline: float | None = None
    while state.get("status") == "running":
        deadline = state.get("deadline")
        server_now = state.get("server_now")
        started_at = state.get("started_at")
        if (not isinstance(deadline, (int, float)) or not isinstance(server_now, (int, float))
                or not isinstance(started_at, (int, float))):
            raise FleetRemoteOperationError("remote_receipt_invalid")
        elapsed = max(0, int(float(server_now) - float(started_at)))
        estimate = call_started + max(0.0, float(deadline) - float(server_now))
        local_deadline = estimate if local_deadline is None else min(local_deadline, estimate)
        if progress is not None:
            progress({"elapsed_seconds": elapsed, "status": "running"})
        if time.monotonic() >= local_deadline:
            state = _remote(node, settings, meta, "timeout")
            break
        time.sleep(min(5.0, max(0.0, local_deadline - time.monotonic())))
        call_started = time.monotonic()
        state = _remote(node, settings, meta, "inspect")
    status = state.get("status")
    if status == "completed":
        result = {"status": "completed", "exit_code": state.get("exit_code")}
        if expected_json is not None:
            result["result_json"] = state.get("result_json")
        return result
    if status == "failed":
        raise FleetRemoteOperationError(state.get("error_code") or "remote_command_failed")
    if status == "interrupted":
        raise FleetRemoteOperationError("interrupted", "remote supervisor disappeared; operator intervention is required")
    if status == "running":
        raise FleetRemoteOperationError("blocked")
    raise FleetRemoteOperationError("remote_receipt_invalid")
