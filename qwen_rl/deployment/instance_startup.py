"""One-use handoff for CPU preflight performed by ``instance resume``."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping

_HANDOFF_SCHEMA = "qwen-rl-instance-startup-handoff/v1"
_HANDOFF_DIR = "startup-handoffs"
_MAX_RECEIPT_BYTES = 2 * 1024 * 1024
_MAX_INPUT_ENTRIES = 50_000
_MAX_SOURCE_BYTES = 1024 * 1024
_HANDOFF_TTL_SECONDS = 900
_SOURCE_FILES = (
    "qwen_rl/deployment/instance.py",
    "qwen_rl/deployment/instance_plan.py",
    "qwen_rl/deployment/instance_preflight.py",
    "qwen_rl/deployment/instance_compat.py",
    "qwen_rl/deployment/instance_startup.py",
    "qwen_rl/deployment/instance_recovery.py",
    "qwen_rl/deployment/instance_storage.py",
    "qwen_rl/deployment/instance_source_package.py",
    "qwen_rl/deployment/autodl_install.py",
    "qwen_rl/evaluation/relocation.py",
    "qwen_rl/runtime/config.py",
    "qwen_rl/contracts/ternary.py",
    "qwen_rl/models/ternary_provider.py",
    "qwen_rl/deployment/local_gradients.py",
    "qwen_rl/task_source.py",
    "qwen_rl/task_catalog.py",
    "qwen_rl/task_source_generated.py",
    "qwen_rl/worker/app.py",
    "qwen_rl/worker/ninfer_backend.py",
    "qwen_rl/worker/ninfer_launcher.py",
)


def _safe_report(supervisor: Any, code: str, field: str, message: str) -> dict[str, Any]:
    deployment = getattr(supervisor, "deployment", None)
    path = getattr(deployment, "path", None)
    command = (f"qwen-rl instance --deployment {shlex.quote(str(path))} doctor"
               if isinstance(path, (Path, str)) and str(path)
               else "qwen-rl instance --deployment <deployment.json> doctor")
    return {
        "schema": "qwen-rl-instance-preflight/v1",
        "status": "blocked",
        "stages": [{"stage": "preflight_handoff", "status": "blocked", "message": message,
                    "field": field}],
        "errors": [{"reason_code": code, "field": field, "message": message,
                    "next_command": command}],
        "dependency_conflicts": [],
        "next_command": command,
        "scope": "one_use_same_resume_operation_handoff",
    }


def _fail(supervisor: Any, code: str, field: str, message: str) -> None:
    from .instance_preflight import InstancePreflightError

    raise InstancePreflightError(_safe_report(supervisor, code, field, message))


def _private_dir(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        info = path.lstat()
    except OSError:
        raise ValueError("private directory is unavailable") from None
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private directory has unsafe metadata")


def _state_dir(supervisor: Any) -> Path:
    state_dir = getattr(getattr(supervisor, "deployment", None), "state_dir", None)
    if not isinstance(state_dir, Path):
        raise ValueError("deployment state directory is invalid")
    _private_dir(state_dir)
    return state_dir


def _open_private(path: Path, flags: int) -> int:
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
        os.close(fd)
        raise ValueError("private file has unsafe metadata")
    return fd


def _stat_record(path: Path, info: os.stat_result) -> dict[str, Any]:
    return {"path": str(path), "mode": stat.S_IMODE(info.st_mode), "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            "device": info.st_dev, "inode": info.st_ino,
            "file_type": stat.S_IFMT(info.st_mode)}


def _configured_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        from ..paths import ROOT
        path = ROOT / path
    return path


def _input_tree_fingerprint(supervisor: Any, plan: Mapping[str, Any]) -> str:
    """Fingerprint input path metadata only; model and dataset bytes are never read."""
    from .instance_preflight import _runtime_paths

    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("prepared runtime inputs are unavailable")
    aliases: dict[Path, list[str]] = {}
    for field, value, _kind in _runtime_paths(runtime, supervisor):
        path = _configured_path(value)
        resolved = path.resolve(strict=True)
        # Opening an imported BEPA ledger may switch journal mode during the
        # CPU restore. Its live state belongs to the ledger owner, not this
        # immutable dependency/input handoff.
        if field == "bepa.state_dir/state.sqlite3":
            continue
        aliases.setdefault(resolved, []).append(field)

    records: list[dict[str, Any]] = []
    visited_dirs: set[tuple[int, int]] = set()

    def visit(path: Path, label: str) -> None:
        if len(records) >= _MAX_INPUT_ENTRIES:
            raise ValueError("input metadata exceeds bounded entry count")
        info = path.stat()
        records.append({"label": label, **_stat_record(path, info)})
        if not stat.S_ISDIR(info.st_mode):
            return
        identity = (info.st_dev, info.st_ino)
        if identity in visited_dirs:
            return
        visited_dirs.add(identity)
        with os.scandir(path) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            item = path / name
            try:
                link_info = item.lstat()
                item_label = f"{label}/{name}"
                if stat.S_ISLNK(link_info.st_mode):
                    target = item.resolve(strict=True)
                    target_info = target.stat()
                    records.append({"label": item_label, "link": os.readlink(item),
                                    **_stat_record(item, link_info),
                                    "target": _stat_record(target, target_info)})
                    if stat.S_ISDIR(target_info.st_mode):
                        visit(target, item_label + "->target")
                elif stat.S_ISDIR(link_info.st_mode):
                    visit(item, item_label)
                else:
                    records.append({"label": item_label, **_stat_record(item, link_info)})
                if len(records) > _MAX_INPUT_ENTRIES:
                    raise ValueError("input metadata exceeds bounded entry count")
            except (OSError, RuntimeError, ValueError):
                raise ValueError("an input path changed or cannot be inspected") from None

    try:
        for path, fields in sorted(aliases.items(), key=lambda row: str(row[0])):
            visit(path, ",".join(sorted(fields)))
    except (OSError, RuntimeError, ValueError):
        raise ValueError("configured input metadata is unavailable") from None
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    result: dict[str, str] = {}
    from .autodl_install import CRITICAL_INSTALLED_FILES
    required = (name.removeprefix("source/") for name in CRITICAL_INSTALLED_FILES
                if name.startswith("source/"))
    for relative in sorted(set(_SOURCE_FILES).union(required)):
        path = root / relative
        try:
            before = path.lstat()
            if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
                    or before.st_size > _MAX_SOURCE_BYTES):
                raise ValueError
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                opened = os.fstat(fd)
                if (not stat.S_ISREG(opened.st_mode) or opened.st_dev != before.st_dev
                        or opened.st_ino != before.st_ino or opened.st_size > _MAX_SOURCE_BYTES):
                    raise ValueError
                digest = hashlib.sha256()
                total = 0
                while True:
                    chunk = os.read(fd, 64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MAX_SOURCE_BYTES:
                        raise ValueError
                    digest.update(chunk)
                after = os.fstat(fd)
                if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                    raise ValueError
            finally:
                os.close(fd)
            result[relative] = digest.hexdigest()
        except (OSError, ValueError):
            raise ValueError("a small startup source file is unavailable or changed") from None
    return result


def _plan_fingerprint(plan: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in plan.items() if key != "operator"}
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _configured_executable_identity(value: str | Path) -> dict[str, Any]:
    configured = Path(os.path.abspath(Path(str(value)).expanduser()))
    try:
        link_info = configured.lstat()
        target = configured.resolve(strict=True)
        target_info = target.stat()
        if not stat.S_ISREG(target_info.st_mode) or not os.access(configured, os.X_OK):
            raise ValueError
        return {"configured_path": str(configured),
                "configured_stat": _stat_record(configured, link_info),
                "symlink_target": os.readlink(configured) if stat.S_ISLNK(link_info.st_mode) else None,
                "target_path": str(target), "target_stat": _stat_record(target, target_info)}
    except (OSError, RuntimeError, ValueError):
        raise ValueError("configured executable is unavailable") from None


def _executable_metadata(supervisor: Any) -> dict[str, dict[str, Any]]:
    deployment = supervisor.deployment
    values = {"python": deployment.python, "worker_python": deployment.worker_python,
              "engine_binary": deployment.engine_binary}
    try:
        return {name: _configured_executable_identity(value) for name, value in values.items()}
    except ValueError:
        raise ValueError("configured executable metadata is unavailable") from None


def _interpreter_metadata() -> dict[str, Any]:
    return {"executable": os.path.abspath(sys.executable),
            "implementation": sys.implementation.name,
            "version": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
            "prefix": sys.prefix, "base_prefix": sys.base_prefix}


_INTERPRETER_PROBE = (
    "import json,os,sys;"
    "print(json.dumps({'executable':os.path.abspath(sys.executable),"
    "'implementation':sys.implementation.name,"
    "'version':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro],"
    "'prefix':sys.prefix,'base_prefix':sys.base_prefix},sort_keys=True))"
)


def _configured_interpreter_metadata(value: str | Path) -> dict[str, Any]:
    """Query the configured Python with a short stdlib-only probe."""
    identity = _configured_executable_identity(value)
    current = _interpreter_metadata()
    if identity["configured_path"] == current["executable"]:
        return {"configured": identity, "runtime": current}
    try:
        result = subprocess.run(
            [identity["configured_path"], "-I", "-B", "-c", _INTERPRETER_PROBE],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, env=dict(os.environ), timeout=20, check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        runtime = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ValueError("configured manager interpreter metadata is unavailable") from None
    if (not isinstance(runtime, dict) or not isinstance(runtime.get("executable"), str)
            or not isinstance(runtime.get("prefix"), str)
            or not isinstance(runtime.get("base_prefix"), str)
            or runtime.get("implementation") != "cpython"
            or not isinstance(runtime.get("version"), list)
            or len(runtime["version"]) != 3
            or any(type(part) is not int or part < 0 for part in runtime["version"])):
        raise ValueError("configured manager interpreter metadata is invalid")
    return {"configured": identity, "runtime": runtime}


def _lock_identity(supervisor: Any, *, require_held: bool) -> dict[str, int]:
    path = _state_dir(supervisor) / "operation.lock"
    fd = -1
    try:
        fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise ValueError
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            locked = True
        else:
            locked = False
            fcntl.flock(fd, fcntl.LOCK_UN)
        if locked != require_held:
            raise ValueError
        return {"device": info.st_dev, "inode": info.st_ino}
    except (OSError, ValueError):
        raise ValueError("resume operation lock is unavailable or not held") from None
    finally:
        if fd >= 0:
            os.close(fd)


def _parent_record(supervisor: Any) -> dict[str, Any]:
    from .local_gradients import _proc_record

    current = _proc_record(os.getpid())
    if current is None:
        raise ValueError("resume parent process identity is unavailable")
    return {"pid": os.getpid(), "starttime": current[0], "argv": list(current[1])}


def create_handoff(supervisor: Any, plan: Mapping[str, Any],
                   preflight_report: Mapping[str, Any]) -> Path:
    """Write one private preflight receipt for the child launched by this resume."""
    if not isinstance(preflight_report, Mapping) or preflight_report.get("status") != "passed":
        _fail(supervisor, "startup_handoff_preflight_not_passed", "runtime_config",
              "A startup handoff requires a passed CPU preflight report.")
    try:
        state_dir = _state_dir(supervisor)
    except (OSError, RuntimeError, TypeError, ValueError):
        _fail(supervisor, "startup_handoff_state_dir_unavailable", "state_dir",
              "The private instance state directory is unavailable.")
    try:
        lock = _lock_identity(supervisor, require_held=True)
    except ValueError:
        _fail(supervisor, "startup_handoff_operation_lock_required", "state_dir.operation_lock",
              "The resume operation lock must remain held while creating the startup handoff.")
    try:
        parent = _parent_record(supervisor)
        plan_digest = _plan_fingerprint(plan)
        interpreter = _configured_interpreter_metadata(supervisor.deployment.python)
        executables = _executable_metadata(supervisor)
        sources = _source_fingerprints()
        inputs = _input_tree_fingerprint(supervisor, plan)
    except (OSError, RuntimeError, TypeError, ValueError):
        _fail(supervisor, "startup_handoff_evidence_unavailable", "runtime_config",
              "Startup preflight evidence could not be safely recorded.")
    handoff_dir = state_dir / _HANDOFF_DIR
    try:
        _private_dir(handoff_dir, create=True)
        path = handoff_dir / (uuid.uuid4().hex + ".json")
        report_copy = dict(preflight_report)
        report_copy["source_fingerprints"] = sources
        report_copy["input_metadata_fingerprint"] = inputs
        if isinstance(preflight_report, dict):
            preflight_report.update(
                source_fingerprints=sources,
                input_metadata_fingerprint=inputs,
            )
        payload = {
            "schema": _HANDOFF_SCHEMA,
            "created_at": time.time(),
            "parent": parent,
            "operation_lock": lock,
            "plan_fingerprint": plan_digest,
            "interpreter": interpreter,
            "executables": executables,
            "source_fingerprints": sources,
            "input_metadata_fingerprint": inputs,
            "preflight_report": report_copy,
        }
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode()
        if len(encoded) > _MAX_RECEIPT_BYTES:
            raise ValueError
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
        return path
    except (OSError, TypeError, ValueError):
        _fail(supervisor, "startup_handoff_write_failed", "state_dir",
              "The one-use startup preflight receipt could not be written safely.")


def _read_and_claim(supervisor: Any, path: str | Path) -> dict[str, Any]:
    state_dir = _state_dir(supervisor)
    handoff_dir = state_dir / _HANDOFF_DIR
    _private_dir(handoff_dir)
    candidate = Path(path)
    expected_parent = Path(os.path.abspath(handoff_dir))
    if Path(os.path.abspath(candidate.parent)) != expected_parent or candidate.name in {"", ".", ".."}:
        _fail(supervisor, "startup_handoff_invalid", "state_dir", "The startup handoff path is invalid.")
    fd = -1
    try:
        before = candidate.lstat()
        if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
                or before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_nlink != 1 or before.st_size > _MAX_RECEIPT_BYTES):
            raise ValueError
        fd = _open_private(candidate, os.O_RDONLY)
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        chunks = bytearray()
        while len(chunks) <= _MAX_RECEIPT_BYTES:
            block = os.read(fd, min(64 * 1024, _MAX_RECEIPT_BYTES + 1 - len(chunks)))
            if not block:
                break
            chunks.extend(block)
        if len(chunks) > _MAX_RECEIPT_BYTES:
            raise ValueError
        current = candidate.lstat()
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError
        candidate.unlink()
        directory_fd = os.open(handoff_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        value = json.loads(chunks.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        _fail(supervisor, "startup_handoff_missing_or_replayed", "state_dir",
              "The startup handoff is missing, unsafe, or has already been used.")
    finally:
        if fd >= 0:
            os.close(fd)


def _check_parent_and_lock(supervisor: Any, receipt: Mapping[str, Any]) -> None:
    from .local_gradients import _proc_record

    parent = receipt.get("parent")
    if not isinstance(parent, Mapping) or type(parent.get("pid")) is not int or type(parent.get("starttime")) is not int:
        _fail(supervisor, "startup_handoff_parent_not_alive", "state_dir.operation_lock",
              "The resume parent no longer owns the startup operation.")
    current = _proc_record(parent["pid"])
    expected_argv = parent.get("argv")
    if (current is None or current[0] != parent["starttime"] or not isinstance(expected_argv, list)
            or list(current[1]) != expected_argv):
        _fail(supervisor, "startup_handoff_parent_not_alive", "state_dir.operation_lock",
              "The resume parent no longer owns the startup operation.")
    expected_lock = receipt.get("operation_lock")
    try:
        actual_lock = _lock_identity(supervisor, require_held=True)
    except ValueError:
        _fail(supervisor, "startup_handoff_operation_lock_lost", "state_dir.operation_lock",
              "The resume operation lock is no longer held.")
    if actual_lock != expected_lock:
        _fail(supervisor, "startup_handoff_operation_lock_changed", "state_dir.operation_lock",
              "The resume operation lock changed during startup.")


def consume_handoff(supervisor: Any, plan: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Claim, validate, and consume the same-operation preflight report once."""
    receipt = _read_and_claim(supervisor, path)
    if receipt.get("schema") != _HANDOFF_SCHEMA:
        _fail(supervisor, "startup_handoff_invalid", "state_dir", "The startup handoff format is unsupported.")
    created_at = receipt.get("created_at")
    if (not isinstance(created_at, (int, float)) or isinstance(created_at, bool)
            or time.time() - created_at < 0 or time.time() - created_at > _HANDOFF_TTL_SECONDS):
        _fail(supervisor, "startup_handoff_expired", "state_dir", "The startup handoff has expired.")
    _check_parent_and_lock(supervisor, receipt)
    if receipt.get("plan_fingerprint") != _plan_fingerprint(plan):
        _fail(supervisor, "startup_handoff_plan_changed", "runtime_config",
              "The prepared startup configuration changed after CPU preflight.")
    try:
        current_interpreter = _interpreter_metadata()
        current_executables = _executable_metadata(supervisor)
    except (OSError, RuntimeError, TypeError, ValueError):
        _fail(supervisor, "startup_handoff_executable_unavailable", "python",
              "Configured interpreter or executable metadata is unavailable after CPU preflight.")
    expected_interpreter = receipt.get("interpreter")
    if (not isinstance(expected_interpreter, Mapping)
            or expected_interpreter.get("runtime") != current_interpreter):
        _fail(supervisor, "startup_handoff_interpreter_changed", "python",
              "The serving process is not using the interpreter that passed CPU preflight.")
    if expected_interpreter.get("configured") != current_executables.get("python"):
        _fail(supervisor, "startup_handoff_executable_changed", "python",
              "The configured manager Python changed after CPU preflight.")
    if receipt.get("executables") != current_executables:
        _fail(supervisor, "startup_handoff_executable_changed", "python",
              "A configured runtime executable changed after CPU preflight.")
    try:
        current_sources = _source_fingerprints()
    except (OSError, RuntimeError, TypeError, ValueError):
        _fail(supervisor, "startup_handoff_source_unavailable", "python",
              "Startup source metadata is unavailable after CPU preflight.")
    if receipt.get("source_fingerprints") != current_sources:
        _fail(supervisor, "startup_handoff_source_changed", "python",
              "Startup source changed after CPU preflight.")
    try:
        current_inputs = _input_tree_fingerprint(supervisor, plan)
    except (OSError, RuntimeError, TypeError, ValueError):
        _fail(supervisor, "startup_handoff_input_unavailable", "runtime_config",
              "Configured input metadata is unavailable after CPU preflight.")
    if receipt.get("input_metadata_fingerprint") != current_inputs:
        _fail(supervisor, "startup_handoff_input_changed", "runtime_config",
              "Configured input metadata changed after CPU preflight.")
    report = receipt.get("preflight_report")
    if not isinstance(report, dict) or report.get("status") != "passed":
        _fail(supervisor, "startup_handoff_invalid", "runtime_config",
              "The startup handoff does not contain a passed preflight report.")
    result = dict(report)
    result["source_fingerprints"] = current_sources
    result["handoff"] = {"status": "passed", "one_use": True,
                         "scope": "same_resume_parent_and_operation_lock"}
    from .local_gradients import _write_private_json
    # The parent may have timed out while metadata was being checked. Never
    # acknowledge or start roles after its operation lock has been released.
    _check_parent_and_lock(supervisor, receipt)
    _write_private_json(Path(str(path) + ".accepted"), {
        "schema": "qwen-rl-instance-startup-handoff-accepted/v1",
        "pid": os.getpid(), "plan_fingerprint": _plan_fingerprint(plan), "status": "passed",
    })
    return result
