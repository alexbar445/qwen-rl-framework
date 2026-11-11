"""Private durable state and detached workers for multi-step Fleet operations."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..identity import atomic_write
from .settings import FleetSettings

_OP_ID = re.compile(r"^[0-9a-f]{32}$")
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_STATUSES = {"pending", "running", "blocked", "failed", "succeeded"}
_FIELDS = {"status", "phase", "completed_steps", "done", "total", "progress",
           "reason_code", "next_action", "pending_nodes", "node", "scope"}
_PROGRESS = {"bytes_done", "bytes_total", "items_done", "items_total", "nodes_done",
             "nodes_total", "percent", "current_step", "current_node"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id(value: str) -> str:
    if not isinstance(value, str) or not _OP_ID.fullmatch(value):
        raise ValueError("operation_id_invalid")
    return value


def _dir(path: Path, *, create: bool = False) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if not create:
            raise ValueError("operation_directory_missing")
        path.mkdir(mode=0o700)
        info = path.lstat()
    if (path.is_symlink() or not path.is_dir() or
            (os.name == "posix" and (info.st_uid != os.geteuid() or info.st_mode & 0o777 != 0o700))):
        raise ValueError("operation_directory_untrusted")
    return path


def _file(path: Path, *, create: bool = False) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if not create:
            raise ValueError("operation_file_missing")
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        info = path.lstat()
    if (path.is_symlink() or not path.is_file() or
            (os.name == "posix" and (info.st_uid != os.geteuid() or info.st_mode & 0o777 != 0o600))):
        raise ValueError("operation_file_untrusted")
    return path


def _root(settings: FleetSettings, *, create: bool = False) -> Path:
    state = Path(settings.state_dir).expanduser().absolute()
    try:
        info = state.lstat()
    except OSError as exc:
        raise ValueError("fleet_state_directory_unavailable") from exc
    if (state.is_symlink() or not state.is_dir() or
            (os.name == "posix" and info.st_uid != os.geteuid())):
        raise ValueError("fleet_state_directory_untrusted")
    return _dir(state / "site-operations", create=create)


def operation_dir(settings: FleetSettings, operation_id: str, *, create: bool = False) -> Path:
    path = _root(settings, create=create) / _id(operation_id)
    if create and not path.exists():
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
    return _dir(path, create=create)


def _read(path: Path) -> dict[str, Any]:
    _file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("operation_json_invalid")  # noqa: TRY004 - invalid persisted data, not a caller type error.
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(path, data.encode(), mode=0o600)


def _status(directory: Path, operation_id: str) -> dict[str, Any]:
    value = _read(directory / "status.json")
    if value.get("operation_id") != operation_id or value.get("schema") != "qwen-fleet-operation-status/v1":
        raise ValueError("operation_status_invalid")
    return value


def _worker(directory: Path) -> dict[str, Any] | None:
    path = directory / "worker.json"
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    return _read(path)


def _proc(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def _alive(worker: Mapping[str, Any] | None) -> bool:
    if not worker or type(worker.get("pid")) is not int or not isinstance(worker.get("starttime"), str):
        return False
    return _proc(worker["pid"]) == worker["starttime"]


def _code(value: Any, label: str, *, optional: bool = True) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _SAFE.fullmatch(value):
        raise ValueError(f"operation_{label}_invalid")
    return value


def _codes(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) > 10000:
        raise ValueError(f"operation_{label}_invalid")
    return [_code(item, label, optional=False) for item in value]  # type: ignore[misc]


def _clean_progress(value: Any) -> dict[str, int | str | None]:
    if not isinstance(value, Mapping) or set(value) - _PROGRESS:
        raise ValueError("operation_progress_invalid")
    result: dict[str, int | str | None] = {}
    for key, item in value.items():
        if key in {"current_step", "current_node"}:
            result[key] = _code(item, "progress_value")
        elif type(item) is int and item >= 0:
            result[key] = item
        else:
            raise ValueError("operation_progress_invalid")
    return result


def _clean_status(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("status") not in _STATUSES:
        raise ValueError("operation_status_invalid")
    if value.get("scope") not in {None, "preparation", "deployment"}:
        raise ValueError("operation_scope_invalid")
    plan_sha = value.get("plan_sha256")
    if not isinstance(plan_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", plan_sha):
        raise ValueError("operation_status_invalid")
    done, total = value.get("done", 0), value.get("total")
    if any(count is not None and (type(count) is not int or count < 0) for count in (done, total)):
        raise ValueError("operation_progress_invalid")
    updated = value.get("last_update")
    if not isinstance(updated, str) or len(updated) > 40:
        raise ValueError("operation_timestamp_invalid")
    try:
        datetime.fromisoformat(updated)
    except ValueError as exc:
        raise ValueError("operation_timestamp_invalid") from exc
    return {
        "status": value["status"], "plan_sha256": plan_sha, "scope": value.get("scope"),
        "phase": _code(value.get("phase"), "phase"),
        "completed_steps": _codes(value.get("completed_steps", []), "completed_steps"),
        "done": done, "total": total, "progress": _clean_progress(value.get("progress", {})),
        "reason_code": _code(value.get("reason_code"), "reason_code"),
        "next_action": _code(value.get("next_action"), "next_action"),
        "pending_nodes": _codes(value.get("pending_nodes", []), "pending_nodes"),
        "node": _code(value.get("node"), "node"), "last_update": updated,
    }


def _summary(directory: Path, operation_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    state = _clean_status(raw)
    worker = _worker(directory)
    status = state["status"]
    if status in {"pending", "running"}:
        status = "running" if _alive(worker) else ("interrupted" if worker else status)
    return {
        "schema": "qwen-fleet-operation-status/v1", "operation_id": operation_id,
        **state, "status": status, "operation_dir": str(directory),
        "plan_file": str(directory / "plan.json"), "logfile": str(directory / "log.txt"),
        "receiptfile": str(directory / "receipts"),
        "pid": worker.get("pid") if worker and type(worker.get("pid")) is int else None,
    }


def job_status(settings: FleetSettings, operation_id: str) -> dict[str, Any]:
    operation_id = _id(operation_id)
    directory = operation_dir(settings, operation_id)
    return _summary(directory, operation_id, _status(directory, operation_id))


def write_progress(settings: FleetSettings, operation_id: str, **fields: Any) -> dict[str, Any]:
    """Atomically update only the public, non-secret operation status fields."""
    operation_id = _id(operation_id)
    if not fields or set(fields) - _FIELDS:
        raise ValueError("operation_status_fields_invalid")
    directory = operation_dir(settings, operation_id)
    raw = _status(directory, operation_id)
    candidate = {**raw, **fields, "last_update": _now()}
    if "status" in fields and fields["status"] not in _STATUSES:
        raise ValueError("operation_status_invalid")
    clean = {"schema": raw["schema"], "operation_id": operation_id,
             "plan_sha256": raw["plan_sha256"], "created_at": raw.get("created_at"),
             **_clean_status(candidate)}
    _write(directory / "status.json", clean)
    return _summary(directory, operation_id, clean)


def _argv(config_path: Path, operation_id: str) -> list[str]:
    return [sys.executable, "-B", "-m", "qwen_rl.cluster.fleet.operations",
            "--config", str(config_path), "--operation-id", operation_id]


def _spawn(config_path: Path, operation_id: str, log_fd: int) -> subprocess.Popen[bytes]:
    code_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(code_root), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.Popen(_argv(config_path, operation_id), stdin=subprocess.DEVNULL,
                            stdout=log_fd, stderr=log_fd, close_fds=True,
                            start_new_session=True, cwd=code_root, env=env)


def _inputs(plan_path: str | Path, config_path: str | Path) -> tuple[bytes, Path, Path]:
    plan, config = Path(plan_path).expanduser().absolute(), Path(config_path).expanduser().absolute()
    _file(plan)
    _file(config)
    return plan.read_bytes(), config, plan


def start_job(settings: FleetSettings, plan_path: str | Path, *, config_path: str | Path,
              resume_operation_id: str | None = None) -> dict[str, Any]:
    plan, child_config, plan_file = _inputs(plan_path, config_path)
    if settings.public_upload_dir:
        public = Path(settings.public_upload_dir).expanduser().resolve()
        if any(p.resolve().is_relative_to(public) for p in (plan_file, child_config)):
            raise ValueError("operation_input_inside_public_tree")
    sha = hashlib.sha256(plan).hexdigest()
    operation_id = _id(resume_operation_id) if resume_operation_id else uuid4().hex
    directory = operation_dir(settings, operation_id, create=resume_operation_id is None)
    launcher = _file(directory / "launcher.lock", create=True)
    _file(directory / "worker.lock", create=True)

    with launcher.open("rb") as fence:
        fcntl.flock(fence.fileno(), fcntl.LOCK_EX)
        status_path = directory / "status.json"
        if resume_operation_id is None:
            atomic_write(directory / "plan.json", plan, mode=0o600)
            _dir(directory / "receipts", create=True)
            _file(directory / "events.jsonl", create=True)
            _file(directory / "log.txt", create=True)
            raw = {"schema": "qwen-fleet-operation-status/v1", "operation_id": operation_id,
                   "plan_sha256": sha, "status": "pending", "phase": "queued",
                   "completed_steps": [], "done": 0, "total": None, "progress": {},
                   "reason_code": None, "next_action": None, "pending_nodes": [], "node": None,
                   "created_at": _now(), "last_update": _now()}
            _write(status_path, raw)
        else:
            raw = _status(directory, operation_id)
            _file(directory / "events.jsonl")
            _dir(directory / "receipts")
            if raw.get("plan_sha256") != sha or hashlib.sha256(_file(directory / "plan.json").read_bytes()).hexdigest() != sha:
                raise ValueError("operation_plan_changed")
            current = _summary(directory, operation_id, raw)
            if current["status"] in {"running", "succeeded"} or _alive(_worker(directory)):
                return current
            if current["status"] not in {"pending", "interrupted", "failed", "blocked"}:
                raise ValueError("operation_not_resumable")
            raw.update(status="pending", reason_code=None, next_action=None)
            _write(status_path, raw)

        log = _file(directory / "log.txt")
        fd = os.open(log, os.O_WRONLY | os.O_APPEND)
        try:
            child = _spawn(child_config, operation_id, fd)
        except OSError as exc:
            write_progress(settings, operation_id, status="failed", phase="launch",
                           reason_code="worker_launch_failed", next_action="inspect_log")
            raise RuntimeError("worker_launch_failed") from exc
        finally:
            os.close(fd)
        start = _proc(child.pid)
        _write(directory / "worker.json", {"pid": child.pid, "starttime": start, "started_at": _now()})
        return _summary(directory, operation_id, _status(directory, operation_id))


@contextmanager
def operation_worker(settings: FleetSettings, operation_id: str) -> Iterator[Path]:
    """Hold a nonblocking, per-operation worker lease for the full execution."""
    operation_id = _id(operation_id)
    directory = operation_dir(settings, operation_id)
    worker_lock = _file(directory / "worker.lock").open("rb")
    try:
        fcntl.flock(worker_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with _file(directory / "launcher.lock").open("rb") as launch_lock:
            fcntl.flock(launch_lock.fileno(), fcntl.LOCK_EX)
            worker, start = _worker(directory), _proc(os.getpid())
            if start is None:
                raise RuntimeError("operation_worker_identity_unavailable")
            if worker is not None and (worker.get("pid") != os.getpid() or worker.get("starttime") != start):
                raise RuntimeError("operation_worker_identity_mismatch")
            worker = worker or {"pid": os.getpid(), "starttime": start, "started_at": _now()}
            worker["claimed_at"] = _now()
            _write(directory / "worker.json", worker)
        yield directory
    finally:
        worker_lock.close()
