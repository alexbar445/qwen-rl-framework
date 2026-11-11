"""Safe, bounded failure diagnosis from logs recorded by the instance owner."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import traceback
from typing import Any, Mapping

_TAIL_BYTES = 64 * 1024
_ROLE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_MODULE = re.compile(r"No module named ['\"]([A-Za-z_][A-Za-z0-9_.]{0,119})['\"]")
_QUOTED_PATH = re.compile(r"['\"](?P<path>/[^'\"\r\n\x00]{1,512})['\"]")
_UNQUOTED_PATH = re.compile(r"(?P<path>/[A-Za-z0-9_./+:-]{1,512})")


def _component(row: Mapping[str, Any]) -> str | None:
    role = row.get("role")
    if not isinstance(role, str) or not _ROLE_ID.fullmatch(role):
        return None
    identifier = row.get("worker_id") or row.get("executor_id")
    if isinstance(identifier, str) and _ROLE_ID.fullmatch(identifier):
        return f"{role}:{identifier}"
    return role


def _role_rows(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    rows = state.get("roles")
    if isinstance(rows, list):
        result.extend(row for row in rows if isinstance(row, Mapping))
    gradient = state.get("gradient")
    if isinstance(gradient, Mapping) and isinstance(gradient.get("roles"), list):
        result.extend(row for row in gradient["roles"] if isinstance(row, Mapping))
    return result


def _safe_owned_log(supervisor: Any, row: Mapping[str, Any]) -> tuple[Path | None, str]:
    path_value = row.get("log_file") or row.get("log_path")
    # The identity tuple is part of the instance manifest and ties this path to
    # a role launched by this supervisor, including roles that have since exited.
    if (not isinstance(path_value, str) or type(row.get("pid")) is not int
            or type(row.get("starttime")) is not int or not isinstance(row.get("argv"), list)):
        return None, "unavailable"
    state_dir = getattr(getattr(supervisor, "deployment", None), "state_dir", None)
    if not isinstance(state_dir, Path):
        return None, "unavailable"
    try:
        root = state_dir.resolve(strict=True)
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = root / path
        info = path.lstat()
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077
                or info.st_nlink != 1):
            return None, "unavailable"
        return path, "available"
    except (OSError, RuntimeError, ValueError):
        return None, "unavailable"


def _tail(path: Path) -> bytes | None:
    try:
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
                or before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) & 0o077
                or before.st_nlink != 1):
            return None
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
        try:
            opened = os.fstat(fd)
            if (not stat.S_ISREG(opened.st_mode) or opened.st_dev != before.st_dev
                    or opened.st_ino != before.st_ino):
                return None
            start = max(0, opened.st_size - _TAIL_BYTES)
            os.lseek(fd, start, os.SEEK_SET)
            chunks = bytearray()
            while len(chunks) < _TAIL_BYTES:
                block = os.read(fd, _TAIL_BYTES - len(chunks))
                if not block:
                    break
                chunks.extend(block)
            return bytes(chunks)
        finally:
            os.close(fd)
    except OSError:
        return None


def _line_failure(line: str) -> dict[str, Any] | None:
    lowered = line.lower()
    if "no space left on device" in lowered or "errno 28" in lowered:
        result: dict[str, Any] = {"reason_code": "disk_full", "enospc": True}
    elif "ninfer t2 rl capabilities are incomplete:" in lowered:
        result = {"reason_code": "native_capabilities_incomplete", "enospc": False}
        known = {"ready", "protocol version", "backend", "build ID", "base artifact kind",
                 "base fingerprint", "artifact fingerprint", "NInfer artifact ID", "NInfer artifact SHA",
                 "input basis", "base weight profile", "source T2 quantization", "runtime T2 formats"}
        fields = line.split("incomplete:", 1)[-1].strip().split(",")
        result["missing_capabilities"] = [field.strip() for field in fields if field.strip() in known]
    elif "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
        result = {"reason_code": "cuda_out_of_memory", "enospc": False}
    elif "address already in use" in lowered:
        result = {"reason_code": "port_in_use", "enospc": False}
    elif "permission denied" in lowered or "errno 13" in lowered:
        result = {"reason_code": "permission_denied", "enospc": False}
    elif "modulenotfounderror" in lowered or "no module named" in lowered:
        result = {"reason_code": "module_not_found", "enospc": False}
        module = _MODULE.search(line)
        if module:
            result["missing_module"] = module.group(1)
    elif "filenotfounderror" in lowered or "no such file or directory" in lowered:
        result = {"reason_code": "file_not_found", "enospc": False}
    elif "traceback (most recent call last)" in lowered:
        result = {"reason_code": "role_exception", "enospc": False}
    else:
        return None
    path_match = _QUOTED_PATH.search(line) or _UNQUOTED_PATH.search(line)
    if path_match:
        value = path_match.group("path")
        if not any(ord(character) < 32 for character in value):
            result["failing_path"] = value
    return result


def _free_bytes(path_value: str | None) -> int | None:
    if not path_value:
        return None
    try:
        path = Path(path_value)
        while not path.exists() and path != path.parent:
            path = path.parent
        target = path if path.is_dir() else path.parent
        return int(shutil.disk_usage(target).free)
    except (OSError, RuntimeError, ValueError):
        return None


def _next_command(supervisor: Any, failure: Mapping[str, Any]) -> str:
    path = failure.get("failing_path")
    reason = failure.get("reason_code")
    deployment_path = getattr(getattr(supervisor, "deployment", None), "path", None)
    if reason == "disk_full" and isinstance(path, str):
        return f"df -h -- {shlex.quote(path)}"
    if reason == "permission_denied" and isinstance(path, str):
        return f"ls -ld -- {shlex.quote(path)}"
    if reason == "module_not_found":
        python = getattr(getattr(supervisor, "deployment", None), "python", "python")
        return f"{shlex.quote(str(python))} -m pip check"
    if reason == "port_in_use":
        return "ss -ltn"
    if isinstance(deployment_path, (Path, str)) and str(deployment_path):
        return f"qwen-rl instance --deployment {shlex.quote(str(deployment_path))} watch --details --once"
    return "qwen-rl instance --deployment <deployment.json> watch --details --once"


def failure_diagnosis(supervisor: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize a failed instance from at most 64 KiB per owned role log.

    Log text and exception messages are never returned. The helper is read-only
    and does not inspect, signal, or otherwise act on unrelated processes.
    """
    status = state.get("status") if isinstance(state, Mapping) else None
    gradient = state.get("gradient")
    gradient_failed = isinstance(gradient, Mapping) and gradient.get("status") in {"failed", "stop_incomplete"}
    stopped_after_failure = status == "stopped" and bool(state.get("error") or state.get("error_message") or gradient_failed)
    if status not in {"failed", "orphaned", "stop_incomplete"} and not stopped_after_failure:
        return {"schema": "qwen-rl-instance-failure-diagnosis/v1", "status": "unknown",
                "reason_code": "instance_not_failed", "component": None,
                "failing_path": None, "enospc": False, "free_bytes": None,
                "components": [], "next_command": _next_command(supervisor, {})}

    components: list[dict[str, Any]] = []
    matches: list[tuple[int, Mapping[str, Any], str, dict[str, Any]]] = []
    priority = {"disk_full": 0, "permission_denied": 1, "file_not_found": 2,
                "module_not_found": 3, "port_in_use": 4,
                "native_capabilities_incomplete": 4, "cuda_out_of_memory": 5, "role_exception": 6}
    for row in _role_rows(state):
        component = _component(row)
        if component is None:
            continue
        path, path_state = _safe_owned_log(supervisor, row)
        component_info = {"component": component, "log_status": path_state}
        components.append(component_info)
        if path is None:
            continue
        tail = _tail(path)
        if tail is None:
            component_info["log_status"] = "unavailable"
            continue
        component_info["log_status"] = "readable"
        # Keep only a parsed code/path, never the line itself.
        for line_number, line in enumerate(tail.decode("utf-8", "replace").splitlines()):
            failure = _line_failure(line)
            if failure is not None:
                matches.append((priority.get(str(failure["reason_code"]), 99), row,
                                component, {**failure, "line_order": line_number}))

    if not matches:
        diagnosis: dict[str, Any] = {"schema": "qwen-rl-instance-failure-diagnosis/v1",
            "status": "unknown", "reason_code": "owned_role_failure_unclassified",
            "component": None, "failing_path": None, "enospc": False, "free_bytes": None,
            "components": components}
        diagnosis["next_command"] = _next_command(supervisor, diagnosis)
        return diagnosis

    _rank, _row, component, failure = min(matches, key=lambda match: (match[0], -match[3]["line_order"]))
    failing_path = failure.get("failing_path")
    diagnosis = {"schema": "qwen-rl-instance-failure-diagnosis/v1", "status": "diagnosed",
                 "reason_code": failure["reason_code"], "component": component,
                 "failing_path": failing_path, "enospc": failure["enospc"],
                 "free_bytes": _free_bytes(failing_path) if failure["enospc"] else None,
                 "components": components}
    if failure.get("missing_module"):
        diagnosis["missing_module"] = failure["missing_module"]
    if "missing_capabilities" in failure:
        diagnosis["missing_capabilities"] = failure["missing_capabilities"]
    diagnosis["next_command"] = _next_command(supervisor, diagnosis)
    return diagnosis


def source_failure_locations(error: BaseException) -> list[dict[str, Any]]:
    """Expose source locations, never exception values or traceback source text."""
    root = Path(__file__).resolve().parents[2]
    locations = []
    for frame, lineno in traceback.walk_tb(error.__traceback__):
        try:
            source = Path(frame.f_code.co_filename).resolve().relative_to(root)
        except ValueError:
            continue
        locations.append({"file": str(source), "line": lineno, "function": frame.f_code.co_name})
    return locations[-6:]
