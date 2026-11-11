"""Trusted stdlib-only remote supervisor for one pinned Fleet operation."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pwd
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "qwen-fleet-remote-operation/v1"
OWNER = ".owner.json"
STATUS = "status.json"
LOG = "log.txt"
STDERR_LOG = "stderr.log"
RECEIPT = "receipt.json"


class Refused(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _read(path: Path) -> dict:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise Refused("private_file_invalid")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            if os.fstat(stream.fileno()).st_ino != info.st_ino:
                raise Refused("private_file_changed")
            value = json.loads(stream.read(131073))
        if not isinstance(value, dict):
            raise Refused("private_json_invalid")
        return value
    except Refused:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Refused("private_file_unavailable") from exc


def _write(path: Path, value: dict, *, exclusive: bool = False) -> None:
    data = _json(value)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    if exclusive:
        flags |= os.O_EXCL
    else:
        flags |= os.O_TRUNC
    try:
        fd = os.open(path, flags, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise
    except OSError as exc:
        raise Refused("private_file_write_failed") from exc


def _private_bytes(path: Path, maximum: int) -> bytes:
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > maximum):
            raise Refused("private_file_invalid")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if opened.st_ino != info.st_ino or opened.st_dev != info.st_dev:
                raise Refused("private_file_changed")
            return stream.read(maximum + 1)
    except Refused:
        raise
    except OSError as exc:
        raise Refused("private_file_unavailable") from exc


def _dir(path: Path, *, create: bool, mode: int = 0o700) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if not create:
            raise Refused("operation_missing")
        try:
            path.mkdir(mode=mode)
            info = path.lstat()
        except FileExistsError:
            info = path.lstat()
        except OSError as exc:
            raise Refused("operation_directory_create_failed") from exc
    except OSError as exc:
        raise Refused("operation_directory_inspection_failed") from exc
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != mode):
        raise Refused("operation_directory_untrusted")


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise Refused("boot_id_unavailable") from exc
    if not value or len(value) > 100:
        raise Refused("boot_id_unavailable")
    return value


def _proc_start(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except (OSError, IndexError, ValueError):
        return None


def _safe_request(meta: dict) -> None:
    if meta.get("schema") != SCHEMA or meta.get("boot_id") != _boot_id():
        raise Refused("boot_id_mismatch")
    for name in ("operation_id", "step_id", "node_id"):
        value = meta.get(name)
        if (not isinstance(value, str) or not value or len(value) > 128
                or not value[0].isalnum()
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-" for c in value)):
            raise Refused("request_identity_invalid")
    argv = meta.get("argv")
    if (not isinstance(argv, list) or not argv or len(argv) > 256
            or any(not isinstance(x, str) or not x or "\x00" in x or len(x) > 8192 for x in argv)):
        raise Refused("argv_invalid")
    digest = hashlib.sha256(json.dumps(argv, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if digest != meta.get("argv_sha256"):
        raise Refused("argv_hash_mismatch")
    for key in ("plan_sha256", "argv_sha256"):
        value = meta.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise Refused("request_hash_invalid")
    expected = meta.get("expected_json")
    if expected is not None and (not isinstance(expected, dict) or len(_json(expected)) > 65537):
        raise Refused("expected_json_invalid")
    plan = {key: meta.get(key) for key in ("node_id", "boot_id", "operation_id", "step_id",
                                           "timeout_seconds", "expected_json")}
    plan["argv_sha256"] = meta["argv_sha256"]
    if hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False).encode()).hexdigest() != meta["plan_sha256"]:
        raise Refused("plan_hash_mismatch")
    if type(meta.get("timeout_seconds")) is not int or not 1 <= meta["timeout_seconds"] <= 604800:
        raise Refused("timeout_invalid")


def _paths(meta: dict, *, create: bool) -> tuple[Path, Path]:
    _safe_request(meta)
    uid = os.geteuid()
    try:
        home = Path(pwd.getpwuid(uid).pw_dir)
    except (KeyError, OSError) as exc:
        raise Refused("home_unavailable") from exc
    if not home.is_absolute() or home == Path("/"):
        raise Refused("home_invalid")
    current = Path(home.anchor)
    for part in home.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            raise Refused("home_unavailable") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise Refused("home_path_untrusted")
    if home.lstat().st_uid != uid:
        raise Refused("home_owner_invalid")
    for part in (".local", "share"):
        current /= part
        try:
            _dir(current, create=create, mode=0o700 if not current.exists() else stat.S_IMODE(current.lstat().st_mode))
        except OSError as exc:
            raise Refused("operation_parent_invalid") from exc
    root = current / "qwen-fleet-operations"
    _dir(root, create=create)
    with _locked(root, name=".bootstrap.lock"):
        op_dir = root / meta["operation_id"]
        _dir(op_dir, create=create)
        op_owner = {"schema": SCHEMA, "uid": uid, "node_id": meta["node_id"], "operation_id": meta["operation_id"]}
        _owner(op_dir, op_owner, create=create)
        step_dir = op_dir / meta["step_id"]
        _dir(step_dir, create=create)
        owner = {**op_owner, "step_id": meta["step_id"], "boot_id": meta["boot_id"],
                 "plan_sha256": meta["plan_sha256"], "argv_sha256": meta["argv_sha256"]}
        _owner(step_dir, owner, create=create)
    return op_dir, step_dir


def _owner(directory: Path, expected: dict, *, create: bool) -> None:
    path = directory / OWNER
    try:
        actual = _read(path)
    except Refused as exc:
        if exc.code != "private_file_unavailable" or not create:
            raise Refused("operation_owner_invalid") from exc
        try:
            if any(directory.iterdir()):
                raise Refused("operation_unmarked_not_empty")
            _write(path, expected, exclusive=True)
            actual = expected
        except FileExistsError:
            actual = _read(path)
    if actual != expected:
        raise Refused("operation_identity_mismatch")


def _status(step_dir: Path, *, create: bool = False) -> dict:
    path = step_dir / STATUS
    try:
        return _read(path)
    except Refused as exc:
        if exc.code != "private_file_unavailable" or not create:
            raise
        value = {"schema": SCHEMA, "status": "pending", "created_at": time.time()}
        _write(path, value, exclusive=True)
        return value


def _same(status: dict, meta: dict) -> None:
    for key in ("node_id", "boot_id", "operation_id", "step_id", "plan_sha256", "argv_sha256"):
        if status.get(key) != meta.get(key):
            raise Refused("operation_identity_mismatch")


def _alive(status: dict, key: str = "supervisor") -> bool:
    process = status.get(key)
    return (isinstance(process, dict) and type(process.get("pid")) is int
            and isinstance(process.get("starttime"), str)
            and _proc_start(process["pid"]) == process["starttime"])


def _terminate(status: dict, key: str = "child") -> None:
    process = status.get(key)
    if not _alive(status, key):
        return
    pid, started = process["pid"], process["starttime"]
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    until = time.monotonic() + 1.0
    while time.monotonic() < until and _proc_start(pid) == started:
        time.sleep(0.05)
    if _proc_start(pid) == started:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _embedded_source(source: str) -> str:
    marker = "from __future__ import annotations\n"
    if marker not in source:
        raise Refused("worker_source_invalid")
    return source.replace(marker, marker + "_SOURCE = " + repr(source) + "\n", 1)


@contextmanager
def _locked(step_dir: Path, *, name: str = ".lock"):
    path = step_dir / name
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = None
    try:
        try:
            fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            os.fchmod(fd, 0o600)
        except FileExistsError:
            fd = os.open(path, flags)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o600):
                raise Refused("operation_lock_untrusted")
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        raise Refused("operation_lock_failed") from exc
    try:
        yield
    finally:
        os.close(fd)


def _result(step_dir: Path, meta: dict, status: dict) -> dict:
    if status.get("status") in {"pending", "running"}:
        if _alive(status):
            return {"status": "running", "deadline": status.get("deadline"),
                    "started_at": status.get("started_at"), "server_now": time.time()}
        status.update(status="interrupted", finished_at=time.time(), error_code="supervisor_lost")
        _write(step_dir / STATUS, status)
    result = {"status": status.get("status"), "deadline": status.get("deadline"),
              "exit_code": status.get("exit_code"), "error_code": status.get("error_code")}
    if status.get("status") == "completed" and meta.get("expected_json") is not None:
        try:
            data = _private_bytes(step_dir / LOG, 65536)
            if len(data) > 65536:
                raise ValueError
            value = json.loads(data)
            if not isinstance(value, dict) or not _json_subset(value, meta["expected_json"]):
                raise ValueError
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            status.update(status="failed", error_code="expected_json_invalid")
            _write(step_dir / STATUS, status)
            raise Refused("expected_json_invalid") from exc
        _write(step_dir / RECEIPT, value)
        result["result_json"] = value
    return result


def _json_subset(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _json_subset(actual[key], value)
                                                for key, value in expected.items())
    return type(actual) is type(expected) and actual == expected


def _launch(step_dir: Path, meta: dict) -> dict:
    existing = _status(step_dir, create=True)
    if existing.get("status") != "pending":
        _same(existing, meta)
        return _result(step_dir, meta, existing)
    read_fd, write_fd = os.pipe()
    encoded = sys.argv[1]
    try:
        source = _embedded_source(_SOURCE)
        subprocess.Popen([sys.executable, "-c", source, encoded, "--supervisor", str(write_fd)],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True, pass_fds=(write_fd,),
                         start_new_session=True)
        os.close(write_fd)
        ready = os.read(read_fd, 1)
        if ready != b"1":
            raise Refused("supervisor_start_failed")
    except (OSError, subprocess.SubprocessError) as exc:
        raise Refused("supervisor_start_failed") from exc
    finally:
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass
    return _result(step_dir, meta, _status(step_dir))


def _supervise(meta: dict, ready_fd: int) -> None:
    _, step_dir = _paths(meta, create=False)
    status = _status(step_dir)
    _same(status, meta)
    started = time.time()
    status.update(status="running", started_at=started, deadline=started + meta["timeout_seconds"],
                  supervisor={"pid": os.getpid(), "starttime": _proc_start(os.getpid())},
                  exit_code=None, error_code=None)
    _write(step_dir / STATUS, status)
    os.write(ready_fd, b"1")
    os.close(ready_fd)
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        log_fd = os.open(step_dir / LOG, flags, 0o600)
        os.fchmod(log_fd, 0o600)
        stderr_fd = os.open(step_dir / STDERR_LOG, flags, 0o600)
        os.fchmod(stderr_fd, 0o600)
        with os.fdopen(log_fd, "wb") as log, os.fdopen(stderr_fd, "wb") as stderr_log:
            with _locked(step_dir):
                latest = _status(step_dir)
                if latest.get("error_code") == "timeout":
                    return
                proc = subprocess.Popen(meta["argv"], stdin=subprocess.DEVNULL, stdout=log, stderr=stderr_log,
                                        shell=False, close_fds=True)
                status["child"] = {"pid": proc.pid, "starttime": _proc_start(proc.pid)}
                _write(step_dir / STATUS, status)
            if latest.get("error_code") == "timeout":
                _terminate(status)
            code = proc.wait(timeout=max(0.0, status["deadline"] - time.time()))
        with _locked(step_dir):
            latest = _status(step_dir)
            if latest.get("error_code") == "timeout":
                status = latest
            else:
                status.update(status="completed" if code == 0 else "failed", exit_code=code,
                              finished_at=time.time(), error_code=None if code == 0 else "remote_command_failed")
            _write(step_dir / STATUS, status)
    except subprocess.TimeoutExpired:
        _terminate(status)
        proc.wait()
        with _locked(step_dir):
            latest = _status(step_dir)
            if latest.get("error_code") != "timeout":
                status.update(status="failed", exit_code=None, finished_at=time.time(), error_code="timeout")
            else:
                status = latest
            _write(step_dir / STATUS, status)
    except (OSError, ValueError, TypeError, Refused, subprocess.SubprocessError):
        _terminate(status)
        with _locked(step_dir):
            status.update(status="failed", exit_code=None, finished_at=time.time(), error_code="supervisor_failed")
            _write(step_dir / STATUS, status)


def _timeout(step_dir: Path, meta: dict) -> dict:
    status = _status(step_dir)
    _same(status, meta)
    if status.get("status") not in {"pending", "running"}:
        return _result(step_dir, meta, status)
    status.update(status="failed", error_code="timeout", finished_at=time.time())
    _write(step_dir / STATUS, status)
    _terminate(status)
    return _result(step_dir, meta, status)


def dispatch(meta: dict, mode: str) -> dict:
    _safe_request(meta)
    if mode == "timeout":
        _, step = _paths(meta, create=False)
        with _locked(step):
            return _timeout(step, meta)
    if mode == "start":
        _, step = _paths(meta, create=True)
        with _locked(step):
            status = _status(step, create=True)
            if status.get("status") == "pending" and "node_id" not in status:
                status.update({key: meta[key] for key in ("node_id", "boot_id", "operation_id", "step_id", "plan_sha256", "argv_sha256")})
                _write(step / STATUS, status)
            _same(status, meta)
            return _launch(step, meta)
    if mode == "inspect":
        try:
            _, step = _paths(meta, create=False)
        except Refused as exc:
            if exc.code == "operation_missing":
                return {"status": "missing"}
            raise
        with _locked(step):
            status = _status(step)
            _same(status, meta)
            return _result(step, meta, status)
    raise Refused("mode_invalid")


def main() -> int:
    try:
        meta = json.loads(__import__("base64").b64decode(sys.argv[1], validate=True))
        if not isinstance(meta, dict):
            raise Refused("metadata_invalid")
        mode = "supervisor" if len(sys.argv) > 2 and sys.argv[2] == "--supervisor" else "timeout" if len(sys.argv) > 2 and sys.argv[2] == "--timeout" else "start" if len(sys.argv) > 2 and sys.argv[2] == "--start" else "inspect"
        if mode == "supervisor":
            _supervise(meta, int(sys.argv[3]))
            return 0
        result = dispatch(meta, mode)
        sys.stdout.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
        return 0
    except Refused as exc:
        sys.stdout.write(json.dumps({"error_code": exc.code}, separators=(",", ":")) + "\n")
        return 0
    except (OSError, ValueError, TypeError, KeyError, IndexError, subprocess.SubprocessError):
        sys.stdout.write('{"error_code":"remote_worker_failed"}\n')
        return 0


_SOURCE = globals().get("_SOURCE") or Path(__file__).read_text(encoding="utf-8")
if __name__ == "__main__":
    raise SystemExit(main())
