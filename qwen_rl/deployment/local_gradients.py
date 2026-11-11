"""Host-owned lifecycle for a local central-gradient Learner deployment.

The host creates private, per-run role configs; a small in-container supervisor
owns the CPU gradient server and its single-GPU executors. Worker containers
remain owned by :class:`LocalWorkerFleet`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .local import DeploymentError, read_or_create_secret
from qwen_rl.learner.ternary_config import project_ternary_compute_profile

_SCHEMA = "qwen-rl-local-gradient-run/v1"
_PLAN_SCHEMA = "qwen-rl-local-gradient-plan/v1"
_SAFE_TOKEN_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_INHERITED_ENV = {
    "PATH", "HOME", "XDG_CACHE_HOME", "LD_LIBRARY_PATH", "CUDA_HOME", "HF_HOME", "ARC_RL_ROOT",
    "PYTHONPATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "PYTHONDONTWRITEBYTECODE",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH",
    "FLASHINFER_WORKSPACE_BASE", "RL_PYTHON", "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES", "PYTORCH_ALLOC_CONF", "FLA_CACHE_RESULTS",
    "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE",
    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO", "TRITON_CACHE_AUTOTUNING",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES", "TORCH_COMPILE_JOB_ID",
}


class LocalGradientError(DeploymentError):
    """A user-safe central-gradient host lifecycle failure."""


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_private_dir(path.parent)
    payload = (json.dumps(dict(value), sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temp_path = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def _write_private_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_private_dir(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temp_path = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def _require_private_dir(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError:
        raise LocalGradientError("central-gradient state directory cannot be inspected") from None
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
        raise LocalGradientError("central-gradient state directory must be private and user-owned")


def _read_private_json(path: Path) -> dict[str, Any] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise LocalGradientError("central-gradient state file cannot be inspected") from None
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1):
        raise LocalGradientError("central-gradient state file has unsafe ownership or mode")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise LocalGradientError("central-gradient state file is unreadable") from None
    if not isinstance(value, dict):
        raise LocalGradientError("central-gradient state file is invalid")
    return value


def _proc_record(pid: int, proc_root: Path = Path("/proc")) -> tuple[int, tuple[str, ...]] | None:
    try:
        stat_text = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
        command_text = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    close = stat_text.rfind(")")
    if close < 0:
        return None
    fields = stat_text[close + 1:].split()
    if len(fields) <= 19:
        return None
    try:
        starttime = int(fields[19])
    except ValueError:
        return None
    command = tuple(part.decode("utf-8", "replace") for part in command_text.split(b"\0") if part)
    return starttime, command


def _record_matches(record: Mapping[str, Any], proc_root: Path = Path("/proc")) -> bool:
    pid, starttime, argv = record.get("pid"), record.get("starttime"), record.get("argv")
    if (type(pid) is not int or pid <= 1 or type(starttime) is not int
            or not isinstance(argv, list) or any(not isinstance(item, str) for item in argv)):
        return False
    current = _proc_record(pid, proc_root)
    return current == (starttime, tuple(argv))


def _process_record(pid: int, argv: Sequence[str]) -> dict[str, Any]:
    deadline = time.monotonic() + 2.0
    while True:
        current = _proc_record(pid)
        if current is not None and current[1] == tuple(argv):
            return {"pid": pid, "starttime": current[0], "argv": list(argv)}
        if time.monotonic() >= deadline:
            raise LocalGradientError("central-gradient process identity could not be established")
        # /proc can briefly expose the pre-exec command after Popen returns.
        time.sleep(0.01)


def _terminate_record(record: Mapping[str, Any], *, proc_root: Path = Path("/proc"),
                      timeout: float = 20.0) -> str:
    pid = int(record["pid"])
    current = _proc_record(pid, proc_root)
    if current is None:
        return "exited"
    if not _record_matches(record, proc_root):
        return "identity_mismatch"
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return "exited"
    deadline = time.monotonic() + timeout * 0.6
    while time.monotonic() < deadline:
        if not _record_matches(record, proc_root):
            return "SIGINT"
        time.sleep(0.1)
    if _record_matches(record, proc_root):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return "exited"
    deadline = time.monotonic() + timeout * 0.3
    while time.monotonic() < deadline:
        if not _record_matches(record, proc_root):
            return "SIGTERM"
        time.sleep(0.1)
    if _record_matches(record, proc_root):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return "exited"
    deadline = time.monotonic() + timeout * 0.1
    while time.monotonic() < deadline:
        if not _record_matches(record, proc_root):
            return "SIGKILL"
        time.sleep(0.1)
    return "stop_incomplete"


def _atomic_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    _write_private_json(path, manifest)


def _http_request_json(url: str, token: str, *, method: str = "GET",
                       body: Mapping[str, Any] | None = None,
                       timeout: float = 3.0) -> tuple[dict[str, Any], Any]:
    data = json.dumps(dict(body), sort_keys=True).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json",
                 **({"Content-Type": "application/json"} if data is not None else {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status not in {200, 202}:
            raise RuntimeError("endpoint returned a non-success response")
        value = json.loads(response.read(2_000_001))
    if not isinstance(value, dict):
        raise TypeError("endpoint returned an invalid object")
    return value, response.headers


def _http_json(url: str, token: str, *, timeout: float = 3.0) -> dict[str, Any]:
    return _http_request_json(url, token, timeout=timeout)[0]


def _sync_initial_workers(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = plan.get("initial_policy_sync", [])
    if not isinstance(entries, list):
        raise LocalGradientError("central initial policy sync plan is malformed")
    prepared: list[dict[str, Any]] = []
    seen_executors: set[str] = set()
    seen_workers: set[str] = set()
    common_identity: tuple[int, str, str, str] | None = None
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise LocalGradientError("central initial policy sync entry is malformed")
        executor_id = entry.get("executor_id")
        token_env = entry.get("token_env")
        server_url = entry.get("server_url")
        worker_ids = entry.get("worker_ids")
        if (not isinstance(executor_id, str) or not executor_id
                or executor_id in seen_executors
                or not isinstance(token_env, str) or _SAFE_TOKEN_ENV.fullmatch(token_env) is None
                or not isinstance(server_url, str) or not server_url
                or not isinstance(worker_ids, list) or not worker_ids
                or any(not isinstance(worker_id, str) or not worker_id for worker_id in worker_ids)
                or len(worker_ids) != len(set(worker_ids))
                or seen_workers.intersection(worker_ids)):
            raise LocalGradientError("central initial policy sync contains a duplicate or invalid executor/Worker cohort")
        seen_executors.add(executor_id)
        seen_workers.update(worker_ids)
        token = os.environ.get(token_env)
        if not token:
            raise LocalGradientError(f"central initial policy sync credential is unavailable: {token_env}")
        root = server_url.rstrip("/") + f"/gradient/v1/executors/{executor_id}"
        policy, headers = _http_request_json(root + "/policy/current", token)
        version = policy.get("version")
        lineage_id = headers.get("X-Lineage-ID")
        engine_epoch = headers.get("X-Engine-Epoch")
        adapter_hash = policy.get("source_adapter_sha256")
        if (type(version) is not int or version < 0
                or not isinstance(lineage_id, str) or not lineage_id
                or not isinstance(engine_epoch, str) or not engine_epoch
                or not isinstance(adapter_hash, str) or not adapter_hash):
            raise LocalGradientError("central policy manifest is missing its current identity")
        identity = (version, lineage_id, engine_epoch, adapter_hash)
        if common_identity is not None and identity != common_identity:
            raise LocalGradientError("central initial policy sync entries do not share one active policy identity")
        common_identity = identity
        prepared.append({
            "executor_id": executor_id, "root": root, "token": token,
            "worker_ids": list(worker_ids), "identity": identity,
        })

    abort_pending = threading.Event()

    def sync_entry(entry: Mapping[str, Any]) -> dict[str, dict[str, Any]] | None:
        if abort_pending.is_set():
            return None
        version, lineage_id, engine_epoch, adapter_hash = entry["identity"]
        try:
            result, _response_headers = _http_request_json(
                entry["root"] + "/workers/ensure", entry["token"], method="POST",
                body={"lineage_id": lineage_id, "engine_epoch": engine_epoch,
                      "policy_version": version, "worker_ids": entry["worker_ids"]},
                timeout=float(plan["request_timeout_seconds"]),
            )
            rows = result.get("workers")
            expected_ids = set(entry["worker_ids"])
            if (not isinstance(rows, dict) or set(rows) != expected_ids
                    or result.get("policy_version") != version
                    or result.get("lineage_id") != lineage_id
                    or result.get("engine_epoch") != engine_epoch):
                raise LocalGradientError("central initial policy sync returned incomplete Worker acknowledgements")
            acknowledgements: dict[str, dict[str, Any]] = {}
            for worker_id, row in rows.items():
                if (not isinstance(row, dict) or row.get("worker_id") != worker_id
                        or row.get("version") != version or row.get("adapter_hash") != adapter_hash):
                    raise LocalGradientError(f"Worker {worker_id} did not ACK the current central policy")
                acknowledgements[worker_id] = {
                    "worker_id": worker_id, "version": version,
                    "adapter_hash": adapter_hash, "executor_id": entry["executor_id"],
                }
            return acknowledgements
        except BaseException:
            # Mark abort before the thread returns so the executor cannot start
            # another queued ensure while the main thread processes the failure.
            abort_pending.set()
            raise

    acknowledgements: dict[str, dict[str, Any]] = {}
    if prepared:
        # Each entry is an independently authorized disjoint cohort. Keep the
        # fan-out bounded; per-Worker policy locks serialize local updates, and
        # cohort-wide pruning only runs when one request covers every Worker.
        with ThreadPoolExecutor(max_workers=min(4, len(prepared)), thread_name_prefix="initial-worker-ack") as pool:
            futures = [pool.submit(sync_entry, entry) for entry in prepared]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        acknowledgements.update(result)
            except BaseException:
                abort_pending.set()
                for pending in futures:
                    pending.cancel()
                # The executor context waits for already-started requests before
                # _supervise can stop the central writer. Preserve the root error.
                raise
        if set(acknowledgements) != seen_workers:
            raise LocalGradientError("central initial policy sync ended before every Worker ACK")
    return acknowledgements


def _resume_pending_run(plan: Mapping[str, Any], manifest: dict[str, Any]) -> dict[str, Any] | None:
    health = _http_json(plan["server_health"]["url"], os.environ[plan["server_health"]["token_env"]])
    recovery = health.get("recovery")
    pending = recovery.get("pending_run") if isinstance(recovery, dict) else None
    if pending in (None, False):
        manifest["resume"] = {"status": "no_pending_run"}
        return manifest["resume"]
    if not isinstance(pending, dict):
        raise LocalGradientError("central recovery reports an invalid pending logical run")
    lineage_id = recovery.get("lineage_id")
    pending_run_id = pending.get("run_id")
    if not isinstance(lineage_id, str) or not lineage_id or not isinstance(pending_run_id, str) or not pending_run_id:
        raise LocalGradientError("central recovery lacks a pending run identity")
    ledger_path = Path(str(plan["resume_ledger_path"]))
    ledger = _read_private_json(ledger_path) or {"schema": "qwen-rl-central-resume-ledger/v1", "attempts": []}
    if ledger.get("schema") != "qwen-rl-central-resume-ledger/v1" or not isinstance(ledger.get("attempts"), list):
        raise LocalGradientError("central resume ledger is invalid")
    generation = str(plan["run_id"])
    prior = next((row for row in ledger["attempts"] if isinstance(row, dict)
                  and row.get("server_generation") == generation), None)
    if prior is not None:
        if prior.get("state") in {"accepted", "no_pending_run"}:
            manifest["resume"] = prior.get("response") or {"status": prior["state"]}
            return manifest["resume"]
        raise LocalGradientError("a prior resume request has an unknown outcome; refusing to replay it")
    attempt = {"server_generation": generation, "lineage_id": lineage_id,
               "pending_run_id": pending_run_id, "state": "requesting", "requested_at": time.time()}
    ledger["attempts"].append(attempt)
    _write_private_json(ledger_path, ledger)
    try:
        response, _headers = _http_request_json(
            str(plan["resume_url"]), os.environ[plan["server_health"]["token_env"]],
            method="POST", body={"lineage_id": lineage_id}, timeout=float(plan["request_timeout_seconds"]),
        )
    except BaseException as exc:
        attempt["state"] = "outcome_unknown"
        attempt["failure_type"] = type(exc).__name__
        attempt["finished_at"] = time.time()
        _write_private_json(ledger_path, ledger)
        raise LocalGradientError("central resume request outcome is unknown; it will not be replayed in this process") from exc
    if response.get("status") == "no_pending_run" and response.get("run_id") is None:
        attempt["state"] = "no_pending_run"
    else:
        remaining = response.get("remaining_steps")
        if (response.get("resumed") is not True or response.get("status") != "running"
                or response.get("run_id") != pending_run_id
                or type(remaining) is not int or remaining < 0):
            attempt["state"] = "outcome_unknown"
            _write_private_json(ledger_path, ledger)
            raise LocalGradientError("central resume response did not match the pending logical run")
        attempt["state"] = "accepted"
    attempt["response"] = response
    attempt["finished_at"] = time.time()
    _write_private_json(ledger_path, ledger)
    manifest["resume"] = response
    return response


def _stop_role_group(records, child_handles, role_grace: float) -> dict[str, str]:
    """Signal an owned peer group together, with one deadline per escalation phase."""
    pending = []
    outcomes = {}

    def finish(record, outcome):
        record["status"] = outcome
        outcomes[str(record.get("executor_id") or record["role"])] = outcome

    for record in records:
        process = child_handles.get(int(record["pid"]))
        if process is not None and process.poll() is not None:
            finish(record, "exited")
        elif process is None or not _record_matches(record):
            finish(record, "exited" if _proc_record(int(record["pid"])) is None else "identity_mismatch")
        else:
            pending.append((process, record))

    for sig, seconds in ((signal.SIGINT, role_grace * 0.6),
                         (signal.SIGTERM, role_grace * 0.3),
                         (signal.SIGKILL, 2.0)):
        deadline = time.monotonic() + seconds
        signaled = []
        for process, record in pending:
            if process.poll() is not None:
                finish(record, "exited")
            elif not _record_matches(record):
                finish(record, "identity_mismatch")
            else:
                try:
                    process.send_signal(sig)
                except ProcessLookupError:
                    finish(record, "exited")
                else:
                    signaled.append((process, record))
        pending = []
        for process, record in signaled:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pending.append((process, record))
            else:
                finish(record, sig.name)
        if not pending:
            break
    for _process, record in pending:
        finish(record, "stop_incomplete")
    return outcomes


def _supervise(plan_path: Path) -> int:
    plan = _read_private_json(plan_path)
    if plan is None or plan.get("schema") != _PLAN_SCHEMA:
        raise LocalGradientError("central-gradient launch plan is missing or unsupported")
    manifest_path = Path(str(plan["manifest_path"]))
    _require_private_dir(manifest_path.parent)
    supervisor_argv = list(plan["supervisor_argv"])
    manifest: dict[str, Any] = {
        "schema": _SCHEMA, "deployment_id": plan["deployment_id"],
        "run_id": plan["run_id"], "config_fingerprint": plan["config_fingerprint"],
        "status": "starting", "server_ready": False,
        "supervisor": _process_record(os.getpid(), supervisor_argv),
        "roles": [], "worker_pins": plan.get("worker_pins", []),
        "started_at": time.time(), "stop_outcomes": {}, "timings": {},
    }
    if "source_config_fingerprint" in plan:
        manifest["source_config_fingerprint"] = plan["source_config_fingerprint"]
    _atomic_manifest(manifest_path, manifest)
    children: list[dict[str, Any]] = []
    child_handles: dict[int, subprocess.Popen[bytes]] = {}
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def spawn(role: Mapping[str, Any]) -> dict[str, Any]:
        env_names = role.get("environment_names")
        argv = role.get("argv")
        if (not isinstance(env_names, list) or not isinstance(argv, list)
                or any(not isinstance(name, str) or _SAFE_TOKEN_ENV.fullmatch(name) is None for name in env_names)
                or any(not isinstance(arg, str) for arg in argv)):
            raise LocalGradientError("central-gradient process plan is invalid")
        secrets_to_filter = plan.get("secret_environment_names", [])
        child_env = {name: os.environ[name] for name in _SAFE_INHERITED_ENV if name in os.environ}
        for secret_name in secrets_to_filter:
            child_env.pop(secret_name, None)
        for name in env_names:
            if name in os.environ:
                child_env[name] = os.environ[name]
        if {name for name in env_names if name in child_env} != set(env_names):
            raise LocalGradientError("central-gradient process is missing a required environment value")
        log_path = Path(str(role["log_path"]))
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise LocalGradientError("central-gradient log file has unsafe ownership or mode")
            process = subprocess.Popen(
                argv, env=child_env, cwd=str(plan["workdir"]), stdin=subprocess.DEVNULL,
                stdout=fd, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
            )
        finally:
            os.close(fd)
        try:
            identity = _process_record(process.pid, argv)
        except BaseException:
            # The Popen handle owns this child even before its durable record exists.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            raise
        record = {
            "role": role["role"], "executor_id": role.get("executor_id"),
            **identity,
            "environment_names": list(env_names), "log_file": str(log_path),
            "status": "running", "started_at": time.time(),
        }
        children.append(record)
        child_handles[process.pid] = process
        manifest["roles"] = children
        _atomic_manifest(manifest_path, manifest)
        return record

    def stop_children() -> dict[str, str]:
        role_grace = float(plan.get("role_stop_timeout_seconds", 20.0))
        if not 0 < role_grace <= 86400:
            role_grace = 20.0
        # Keep the central writer alive until all producers/executors have stopped.
        peers = [record for record in reversed(children) if record["role"] != "server"]
        servers = [record for record in children if record["role"] == "server"]
        outcomes = _stop_role_group(peers, child_handles, role_grace)
        if all(value in {"SIGINT", "SIGTERM", "SIGKILL", "exited"} for value in outcomes.values()):
            outcomes.update(_stop_role_group(servers, child_handles, role_grace))
        else:
            # Preserve the writer and its Workers if a peer cannot be released safely.
            for record in servers:
                record["status"] = "stop_incomplete"
                outcomes[record["role"]] = "stop_incomplete"
        manifest["roles"] = children
        manifest["stop_outcomes"] = outcomes
        return outcomes

    try:
        roles = plan.get("roles")
        if not isinstance(roles, list) or not roles or roles[0].get("role") != "server":
            raise LocalGradientError("central-gradient launch plan must start with the server")
        server_ready_started = time.monotonic()
        try:
            spawn(roles[0])
            health = plan.get("server_health")
            if not isinstance(health, dict):
                raise LocalGradientError("central-gradient launch plan has no server readiness probe")
            deadline = time.monotonic() + float(plan["server_ready_timeout_seconds"])
            while not stop_requested and time.monotonic() < deadline:
                server = child_handles[int(children[0]["pid"])]
                if server.poll() is not None:
                    raise LocalGradientError("central-gradient server exited before becoming ready")
                try:
                    payload = _http_json(health["url"], os.environ[health["token_env"]])
                    if payload.get("ready") is True and payload.get("role") == "learner":
                        manifest["server_ready"] = True
                        break
                except (OSError, urllib.error.URLError, ValueError, KeyError, RuntimeError):
                    pass
                time.sleep(0.5)
        finally:
            manifest["timings"]["server_ready_duration_seconds"] = time.monotonic() - server_ready_started
            _atomic_manifest(manifest_path, manifest)
        if stop_requested:
            raise LocalGradientError("central-gradient start was interrupted")
        if not manifest["server_ready"]:
            raise LocalGradientError("central-gradient server did not become ready before timeout")
        initial_sync_started = time.monotonic()
        try:
            manifest["policy_acks"] = _sync_initial_workers(plan)
        finally:
            manifest["timings"]["initial_policy_sync_duration_seconds"] = time.monotonic() - initial_sync_started
        if stop_requested:
            raise LocalGradientError("central-gradient start was interrupted")
        resume_started = time.monotonic()
        try:
            manifest["resume"] = _resume_pending_run(plan, manifest)
        finally:
            manifest["timings"]["resume_pending_run_duration_seconds"] = time.monotonic() - resume_started
        _atomic_manifest(manifest_path, manifest)
        for role in roles[1:]:
            if stop_requested:
                raise LocalGradientError("central-gradient start was interrupted")
            spawn(role)
        manifest["status"] = "running"
        manifest["started_at"] = time.time()
        _atomic_manifest(manifest_path, manifest)
        while not stop_requested:
            for record in children:
                process = child_handles[int(record["pid"])]
                code = process.poll()
                if code is not None:
                    record["status"] = "exited"
                    record["exit_code"] = code
                    raise LocalGradientError(
                        f"central-gradient {record.get('executor_id') or record['role']} exited unexpectedly"
                    )
            time.sleep(0.5)
    except BaseException as exc:  # noqa: BLE001 - every supervisor failure must drain owned children.
        manifest["status"] = "stopping" if stop_requested else "failed"
        manifest["error"] = type(exc).__name__
        if isinstance(exc, LocalGradientError):
            manifest["error_message"] = str(exc)
        _atomic_manifest(manifest_path, manifest)
    finally:
        outcomes = stop_children()
        clean_stop = all(value in {"SIGINT", "SIGTERM", "SIGKILL", "exited"}
                         for value in outcomes.values())
        manifest["status"] = ("stopped" if stop_requested and clean_stop else
                              "stop_incomplete" if not clean_stop else manifest.get("status", "failed"))
        manifest["stopped_at"] = time.time()
        _atomic_manifest(manifest_path, manifest)
    return 0 if manifest["status"] == "stopped" else 1


def _container_status(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_private_json(manifest_path)
    if manifest is None or manifest.get("schema") != _SCHEMA:
        return {"backend": "central_gradient", "status": "degraded", "deployment_status": "unknown",
                "run_id": None, "roles": [], "reason": "owned manifest is missing or unsupported"}
    supervisor = manifest.get("supervisor")
    supervisor_live = isinstance(supervisor, dict) and _record_matches(supervisor)
    roles = []
    for record in manifest.get("roles", []):
        if not isinstance(record, dict):
            continue
        roles.append({
            "role": record.get("role"), "executor_id": record.get("executor_id"),
            "pid": record.get("pid"), "starttime": record.get("starttime"),
            "identity_matches": _record_matches(record),
            "status": record.get("status"), "log_file": record.get("log_file"),
        })
    worker_rows = []
    for pin in manifest.get("worker_pins", []):
        if not isinstance(pin, dict):
            continue
        token = os.environ.get(pin.get("auth_token_env", ""), "")
        try:
            health = _http_json(pin["endpoint"] + "/health", token) if token else {}
        except (OSError, urllib.error.URLError, ValueError, KeyError, RuntimeError):
            health = {}
        identity_matches = bool(
            health.get("worker_id") == pin.get("worker_id")
            and health.get("boot_id") == pin.get("boot_id")
            and health.get("telemetry_epoch") == pin.get("telemetry_epoch")
            and health.get("ninfer_artifact_sha256") == pin.get("model_sha256")
            and health.get("gpu_uuid") == pin.get("gpu_uuid")
            and health.get("base_fingerprint") == pin.get("base_fingerprint")
            and health.get("artifact_fingerprint") == pin.get("artifact_fingerprint")
            and health.get("residency_enabled") is pin.get("residency_enabled")
            and isinstance(health.get("deployment"), dict)
            and health["deployment"].get("scope") == "service"
            and health["deployment"].get("site_id") == pin.get("site_id")
            and health["deployment"].get("node_id") == pin.get("node_id")
        )
        worker_rows.append({"worker_id": pin.get("worker_id"), "identity_matches": identity_matches,
                            "ready": health.get("ready"), "residency_state": health.get("residency_state")})
    workers_live = all(row["identity_matches"] for row in worker_rows)
    if manifest.get("status") == "stop_incomplete":
        status = "degraded"
    elif (manifest.get("status") == "stopped" and not supervisor_live
          and not any(row["identity_matches"] for row in roles)):
        status = "stopped"
    elif supervisor_live and manifest.get("status") == "running" and all(row["identity_matches"] for row in roles) and workers_live:
        status = "running"
    elif supervisor_live:
        status = "starting" if manifest.get("status") == "starting" else "degraded"
    else:
        status = "degraded" if any(row["identity_matches"] for row in roles) else "stopped"
    return {
        "backend": "central_gradient", "status": status,
        "deployment_status": ("workers_live" if status == "stopped" and any(row["identity_matches"] for row in worker_rows)
                              else status),
        "launch_pending": manifest.get("status") == "prepared" and not supervisor_live and not roles,
        "run_id": manifest.get("run_id"), "server_ready": manifest.get("server_ready"),
        "roles": roles, "started_at": manifest.get("started_at"),
        "timings": manifest.get("timings", {}),
        "stopped_at": manifest.get("stopped_at"),
        "workers": worker_rows, "policy_acks": manifest.get("policy_acks", {}),
        "resume": manifest.get("resume"),
        "stop_outcomes": manifest.get("stop_outcomes", {}),
    }


def _container_stop(manifest_path: Path, *, timeout: float = 60.0) -> dict[str, Any]:
    status = _container_status(manifest_path)
    if status["status"] == "stopped":
        return status
    manifest = _read_private_json(manifest_path)
    if manifest is None:
        return status
    supervisor = manifest.get("supervisor")
    if isinstance(supervisor, dict) and _record_matches(supervisor):
        try:
            os.kill(int(supervisor["pid"]), signal.SIGINT)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = _container_status(manifest_path)
            if current["status"] == "stopped":
                return current
            time.sleep(0.2)
        return _container_status(manifest_path)
    outcomes: dict[str, str] = {}
    for record in reversed(manifest.get("roles", [])):
        if not isinstance(record, dict):
            continue
        key = str(record.get("executor_id") or record.get("role"))
        outcomes[key] = _terminate_record(record, timeout=timeout / max(1, len(manifest.get("roles", []))))
    manifest["stop_outcomes"] = outcomes
    manifest["status"] = "stopped" if all(value in {"SIGINT", "SIGTERM", "SIGKILL", "exited"}
                                           for value in outcomes.values()) else "stop_incomplete"
    manifest["stopped_at"] = time.time()
    _atomic_manifest(manifest_path, manifest)
    return _container_status(manifest_path)


def _internal_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="qwen_rl.deployment.local_gradients")
    parser.add_argument("command", choices=("_supervise", "_status", "_stop"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(list(argv))
    try:
        if args.command == "_supervise":
            if args.plan is None:
                raise LocalGradientError("_supervise requires --plan")
            return _supervise(args.plan)
        if args.manifest is None:
            raise LocalGradientError(f"{args.command} requires --manifest")
        result = _container_status(args.manifest) if args.command == "_status" else _container_stop(args.manifest)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0 if result["status"] != "degraded" and result["status"] != "stop_incomplete" else 2
    except (LocalGradientError, OSError, ValueError) as exc:
        print(f"qwen-rl local gradient lifecycle failed ({type(exc).__name__})", file=sys.stderr)
        return 2


def _load_runtime_config(path: Path):
    from ..runtime.config import load_config

    try:
        return load_config(path, for_local_worker_plan=True)
    except Exception as exc:
        raise LocalGradientError(f"local central-gradient runtime config is invalid ({type(exc).__name__})") from exc


def is_central_gradient_config(path: Path) -> bool:
    """Read the configured backend without starting or probing a role."""
    return _load_runtime_config(path).learner.update_backend == "central_gradient"


def _local_pairs(runtime: Any, plan: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = dict(runtime.learner.gradient_executor_worker_ids)
    local_ids = set(runtime.learner.gradient_local_executor_worker_ids)
    specs = {spec.worker_id: spec for spec in plan.specs}
    local_pairs = {executor_id: worker_id for executor_id, worker_id in pairs.items()
                   if worker_id in local_ids}
    if not local_pairs or set(local_pairs.values()) != local_ids or not local_ids <= set(specs):
        raise LocalGradientError(
            "each locally managed central-gradient executor must map to one managed local Worker"
        )
    by_worker = {worker.id: worker for worker in runtime.workers}
    if set(pairs.values()) - set(by_worker):
        raise LocalGradientError("central-gradient executor mapping references a missing Worker")
    return local_pairs, specs


def _compute_profile(runtime: Any, *, artifact_identity_path: str,
                     run_dir: str, store_dir: str) -> dict[str, Any]:
    return project_ternary_compute_profile(
        runtime.learner,
        group_size=runtime.rollout.group_size,
        constrained_decoding=runtime.rollout.constrained_decoding,
        artifact_identity_path=artifact_identity_path,
        run_dir=run_dir,
        store_dir=store_dir,
    )


class LocalGradientHost:
    """Bridge the existing local deployment and WorkerFleet to gradient processes."""

    def __init__(self, deployment: Any):
        self.deployment = deployment

    def _run_root(self, fleet: Any) -> Path:
        return fleet.state_root / "gradient-runs"

    @staticmethod
    def _container_child_path(plan: Any, host_path: Path) -> str:
        relative = host_path.relative_to(plan.runtime_config_host_path.parent)
        return str(Path(plan.runtime_config_container_path).parent / relative)

    def _source_config_fingerprint(self) -> str:
        configured = Path(self.deployment.config.default_config)
        try:
            relative = configured.relative_to("/work")
            source = self.deployment.root / relative
            return hashlib.sha256(source.read_bytes()).hexdigest()
        except (OSError, ValueError):
            raise LocalGradientError("central-gradient source config cannot be read") from None

    @staticmethod
    def _pin_runtime_service_workers(plan: Any, pins: Mapping[str, Mapping[str, Any]]) -> None:
        host_path = plan.runtime_config_host_path
        try:
            config = yaml.safe_load(host_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            raise LocalGradientError("generated central-gradient runtime config cannot be read") from None
        workers = config.get("workers") if isinstance(config, dict) else None
        if not isinstance(workers, list):
            raise LocalGradientError("generated central-gradient runtime config has no Worker list")
        seen: set[str] = set()
        for worker in workers:
            if not isinstance(worker, dict) or worker.get("id") not in pins:
                continue
            worker.update({
                "boot_id": pins[worker["id"]]["boot_id"],
                "telemetry_epoch": pins[worker["id"]]["telemetry_epoch"],
                "model_sha256": pins[worker["id"]]["model_sha256"],
            })
            seen.add(worker["id"])
        if seen != set(pins):
            raise LocalGradientError("generated central-gradient config omits a locally pinned Worker")
        _write_private_bytes(host_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode())

    def _restore_active_pins(self, fleet: Any, plan: Any, run_dir: Path) -> None:
        pins = _read_private_json(run_dir / "worker-pins.json")
        if pins is None or pins.get("schema") != "qwen-rl-local-gradient-worker-pins/v1":
            raise LocalGradientError("active central-gradient Worker pins are missing")
        self._pin_runtime_service_workers(plan, pins.get("workers", {}))

    def _active(self, fleet: Any) -> tuple[dict[str, Any], Path] | None:
        root = self._run_root(fleet)
        value = _read_private_json(root / "active.json")
        if value is None:
            runs_root = root / "runs"
            if runs_root.exists():
                _require_private_dir(runs_root)
                runtime_paths = fleet.existing_runtime_paths()
                if runtime_paths is None:
                    raise LocalGradientError(
                        "central-gradient run directories exist without an active pointer or managed Worker profile"
                    )
                runtime_host_parent = fleet.runtime_config_host_path.parent
                runtime_container_parent = Path(runtime_paths[0]).parent
                for run_dir in runs_root.iterdir():
                    if run_dir.is_symlink() or not run_dir.is_dir():
                        raise LocalGradientError("central-gradient run root contains an unverified entry")
                    manifest = _read_private_json(run_dir / "manifest.json")
                    if manifest is None:
                        raise LocalGradientError(
                            "a retained central-gradient run directory has no ownership manifest"
                        )
                    try:
                        relative = (run_dir / "manifest.json").relative_to(runtime_host_parent)
                    except ValueError:
                        raise LocalGradientError("central-gradient run directory escaped its managed root") from None
                    manifest_container = str(runtime_container_parent / relative)
                    tokens = self._worker_status_environment(fleet, run_dir)
                    status = self._invoke("_status", manifest_container,
                                          timeout=self.deployment.config.docker_timeout,
                                          environment=tokens)
                    if status.get("status") != "stopped":
                        raise LocalGradientError(
                            "central-gradient run manifests exist without an active pointer; refusing lifecycle changes"
                        )
            return None
        if value.get("schema") != _SCHEMA or not isinstance(value.get("run_id"), str):
            raise LocalGradientError("central-gradient active-run marker is invalid")
        run_dir = root / "runs" / value["run_id"]
        if run_dir.parent != (root / "runs"):
            raise LocalGradientError("central-gradient active-run path is invalid")
        return value, run_dir

    def _invoke(self, command: str, manifest_container_path: str, *, timeout: float,
                environment: Mapping[str, str] | None = None) -> dict[str, Any]:
        cfg = self.deployment.config
        argv = ["docker", "exec", "-u", cfg.container_uid, "-w", cfg.container_workdir,
                ]
        env_values = self.deployment._base_child_env()
        for name, value in (environment or {}).items():
            if not isinstance(name, str) or _SAFE_TOKEN_ENV.fullmatch(name) is None or not isinstance(value, str):
                raise LocalGradientError("central-gradient status environment is invalid")
            env_values[name] = value
            argv.extend(("-e", name))
        argv.extend((cfg.container, cfg.container_python, "-B", "-m", "qwen_rl.deployment.local_gradients",
                     command, "--manifest", manifest_container_path))
        try:
            result = self.deployment.runner(argv, env=env_values, text=True,
                                            capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            raise LocalGradientError("central-gradient container control command timed out") from None
        except OSError:
            raise LocalGradientError("central-gradient container control command could not start") from None
        if result.returncode not in {0, 2}:
            raise LocalGradientError("central-gradient container control command failed")
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            raise LocalGradientError("central-gradient container returned invalid control data") from None
        if not isinstance(payload, dict):
            raise LocalGradientError("central-gradient container returned invalid control data")
        if result.returncode == 2 and payload.get("status") != "stopped":
            raise LocalGradientError("central-gradient process ownership is degraded; refusing unsafe control")
        return payload

    @staticmethod
    def _worker_status_environment(fleet: Any, run_dir: Path) -> dict[str, str]:
        worker_pins = _read_private_json(run_dir / "worker-pins.json")
        tokens: dict[str, str] = {}
        if isinstance(worker_pins, dict) and isinstance(worker_pins.get("workers"), dict):
            for worker_id, pin in worker_pins["workers"].items():
                if not isinstance(pin, dict):
                    continue
                token_env = pin.get("auth_token_env")
                if isinstance(token_env, str):
                    tokens[token_env] = read_or_create_secret(
                        fleet.state_root / "tokens" / f"{worker_id}.token",
                        label=f"local Worker {worker_id}", create=False,
                    )
        return tokens

    def status(self, fleet: Any) -> dict[str, Any]:
        active = self._active(fleet)
        if active is None:
            return {"backend": "central_gradient", "status": "stopped", "run_id": None, "roles": []}
        marker, run_dir = active
        manifest_container = marker.get("manifest_container_path")
        if not isinstance(manifest_container, str):
            raise LocalGradientError("central-gradient active-run marker has no manifest path")
        tokens = self._worker_status_environment(fleet, run_dir)
        return self._invoke("_status", manifest_container, timeout=self.deployment.config.docker_timeout,
                            environment=tokens)

    def check(self, plan: Any) -> dict[str, Any]:
        runtime = plan.runtime_config
        _local_pairs(runtime, plan)
        fleet = self.deployment._worker_fleet()
        container_gpus = fleet.manager_gpu_inventory()
        for spec in plan.specs:
            if spec.gpu_uuid not in container_gpus.values():
                raise LocalGradientError(
                    f"manager does not expose Worker physical GPU {spec.gpu_index} UUID {spec.gpu_uuid}"
                )
        active = self._active(fleet)
        if active is not None:
            marker, run_dir = active
            status = self.status(fleet)
            if status.get("status") in {"running", "starting"}:
                if marker.get("source_config_fingerprint") != self._source_config_fingerprint():
                    raise LocalGradientError("active central-gradient runtime uses a different source config")
                if marker.get("worker_fingerprint") != plan.fingerprint:
                    raise LocalGradientError("active central-gradient runtime uses a different Worker config")
                self._restore_active_pins(fleet, plan, run_dir)
        return {"backend": "central_gradient", "workers": len(plan.specs),
                "managed_local_executors": len(runtime.learner.gradient_local_executor_worker_ids),
                "allowlisted_executors": len(runtime.learner.gradient_executor_token_envs),
                "checkpoint_mode": "central_full_state_auto_resume",
                "state_dir": runtime.learner.recovery_dir}

    def _assert_legacy_runtime_inactive(self, credentials: Mapping[str, str]) -> None:
        legacy_profile = self.deployment.root / "configs" / "local-bonsai2-phase-dp4-bepa.yaml"
        if legacy_profile.is_file():
            snapshot = self.deployment._manager_snapshot(
                {"ARC_ENV_TOKEN": credentials["ARC_ENV_TOKEN"]},
                runtime_paths=("/work/configs/local-bonsai2-phase-dp4-bepa.yaml", None),
            )
            if snapshot is None:
                raise LocalGradientError("cannot verify the existing DP4 RuntimeManager before central startup")
            live = [row for row in snapshot["processes"] if row.get("identity_matches") is True]
            if live:
                raise LocalGradientError(
                    "the formal DP4 RuntimeManager still owns roles; stop that deployment before central startup"
                )

        legacy_deployment_path = self.deployment.root / "configs" / "local-bonsai2-phase-dp4-bepa-deployment.json"
        if not legacy_deployment_path.is_file():
            return
        try:
            from .local import LocalDeployment, load_deployment

            legacy_config = load_deployment(legacy_deployment_path)
            legacy_deployment = LocalDeployment(
                self.deployment.root, legacy_config,
                runner=self.deployment.runner, environ=self.deployment.environ,
            )
            legacy_fleet = legacy_deployment._worker_fleet()
            if not legacy_fleet._check_state_path(required=False):
                return
            legacy_fleet._verify_owner()
            worker_manifest = legacy_fleet._read_manifest()
        except Exception as exc:
            raise LocalGradientError(f"cannot verify formal DP4 Worker ownership ({type(exc).__name__})") from exc
        if worker_manifest is None:
            return
        expected_common = {
            "io.qwen-rl.owner": "local-worker-fleet/v1",
            "io.qwen-rl.deployment": worker_manifest.get("deployment_id"),
            "io.qwen-rl.fingerprint": worker_manifest.get("fingerprint"),
        }
        for row in worker_manifest["workers"]:
            if not isinstance(row, dict) or not isinstance(row.get("container_name"), str):
                raise LocalGradientError("formal DP4 Worker manifest contains an invalid container identity")
            container = legacy_fleet._inspect_container(row["container_name"])
            if container is None or not container["running"]:
                continue
            labels = container["labels"]
            expected = {
                **expected_common,
                "io.qwen-rl.worker": row.get("worker_id"),
                "io.qwen-rl.config-sha256": row.get("config_sha256"),
                "io.qwen-rl.image-id": row.get("worker_image_id"),
            }
            if (container.get("id") != row.get("container_id")
                    or any(labels.get(key) != value for key, value in expected.items())):
                raise LocalGradientError("an unverified container occupies the formal DP4 Worker identity")
            raise LocalGradientError(
                f"formal DP4 Worker {row['worker_id']} is still running; stop that deployment first"
            )

    def _assert_learner_port_free(self, runtime: Any) -> None:
        cfg = self.deployment.config
        argv = ["docker", "exec", "-u", cfg.container_uid, cfg.container,
                cfg.container_python, "-B", "-c",
                ("import socket,sys\ns=socket.socket(); s.settimeout(1)\n"
                "try:\n s.connect(('127.0.0.1', int(sys.argv[1])))\n"
                "except ConnectionRefusedError:\n raise SystemExit(1)\n"
                "except OSError:\n raise SystemExit(2)\n"
                "else:\n raise SystemExit(0)\n"), str(runtime.learner.port)]
        try:
            result = self.deployment.runner(argv, env=self.deployment._base_child_env(), text=True,
                                            capture_output=True, timeout=cfg.docker_timeout, check=False)
        except (subprocess.TimeoutExpired, OSError):
            raise LocalGradientError("cannot verify the manager-container Learner port before startup") from None
        if result.returncode == 0:
            raise LocalGradientError(
                f"Learner port {runtime.learner.port} is already occupied by a runtime outside this deployment"
            )
        if result.returncode != 1:
            raise LocalGradientError("Learner port ownership preflight returned an invalid result")

    def _worker_identity(self, fleet: Any, plan: Any, spec: Any, *, timeout: float,
                         expect_residency: bool) -> dict[str, str]:
        deadline = time.monotonic() + timeout
        expected_sha = plan.runtime_config.worker_inference.ninfer_artifact_sha256
        source_input = next((item for item in plan.ninfer_inputs
                             if item.field_name == "ninfer_artifact_path"), None)
        if (not isinstance(expected_sha, str) or source_input is None
                or source_input.sha256 != expected_sha):
            raise LocalGradientError("Worker model pin differs from the runtime's declared artifact identity")
        while time.monotonic() < deadline:
            try:
                payload = fleet.worker_identity(plan, spec)
                deployment = payload.get("deployment")
                if (payload.get("ready") is True and payload.get("role") == "worker"
                        and payload.get("worker_id") == spec.worker_id
                        and payload.get("boot_id") == plan.boot_id
                        and payload.get("gpu_uuid") == spec.gpu_uuid
                        and payload.get("residency_enabled") is expect_residency
                        and isinstance(payload.get("telemetry_epoch"), str)
                        and bool(payload["telemetry_epoch"])
                        and payload.get("ninfer_artifact_sha256") == expected_sha
                        and payload.get("base_fingerprint") == expected_sha
                        and payload.get("artifact_fingerprint") == expected_sha
                        and isinstance(deployment, dict)
                        and deployment.get("scope") == "service"
                        and deployment.get("site_id") == spec.site_id
                        and deployment.get("node_id") == spec.node_id):
                    return {
                        "boot_id": payload["boot_id"],
                        "telemetry_epoch": payload["telemetry_epoch"],
                        "model_sha256": expected_sha,
                        "gpu_uuid": spec.gpu_uuid,
                        "base_fingerprint": expected_sha,
                        "artifact_fingerprint": expected_sha,
                    }
                raise LocalGradientError(f"Worker {spec.worker_id} health differs from its pinned service identity")
            except LocalGradientError:
                raise
            except (OSError, urllib.error.URLError, ValueError):
                time.sleep(0.25)
        raise LocalGradientError(f"Worker {spec.worker_id} did not become ready for central gradient sharing")

    def _make_run(self, plan: Any, credentials: Mapping[str, str],
                  worker_started: Sequence[tuple[str, str]]) -> tuple[Path, str, dict[str, Any]]:
        runtime = plan.runtime_config
        local_pairs, specs = _local_pairs(runtime, plan)
        pairs = dict(runtime.learner.gradient_executor_worker_ids)
        fleet = self.deployment._worker_fleet()
        root = self._run_root(fleet)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_private_dir(root)
        run_id = f"{int(time.time())}-{uuid.uuid4().hex[:12]}"
        runs_root = root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_private_dir(runs_root)
        run_dir = runs_root / run_id
        run_dir.mkdir(exist_ok=False, mode=0o700)
        (run_dir / "logs").mkdir(mode=0o700)
        _write_private_json(run_dir / "manifest.json", {
            "schema": _SCHEMA, "deployment_id": plan.deployment_id, "run_id": run_id,
            "status": "prepared", "roles": [],
        })
        run_dir_container = Path(self._container_child_path(plan, run_dir))
        logs_container = run_dir_container / "logs"
        gradient_tokens = fleet.state_root / "gradient-tokens"
        gradient_tokens.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_private_dir(gradient_tokens)
        executor_envs = runtime.learner.gradient_executor_token_envs
        executor_tokens: dict[str, str] = {}
        for executor_id, env_name in executor_envs.items():
            if not isinstance(executor_id, str) or not executor_id or _SAFE_TOKEN_ENV.fullmatch(env_name) is None:
                raise LocalGradientError("central-gradient executor identity is invalid")
            if executor_id in local_pairs:
                executor_tokens[executor_id] = read_or_create_secret(
                    gradient_tokens / f"{executor_id}.token", label=f"gradient executor {executor_id}", create=True,
                )
            else:
                token = fleet.command_env.get(env_name)
                if not token:
                    raise LocalGradientError(f"remote executor credential is unavailable in environment: {env_name}")
                executor_tokens[executor_id] = token
        all_worker_tokens = {spec.auth_token_env: spec.auth_token for spec in plan.specs}
        for worker in runtime.workers:
            if worker.scope == "gateway":
                token = fleet.command_env.get(worker.auth_token_env)
                if not token:
                    raise LocalGradientError(f"Gateway Worker credential is unavailable: {worker.auth_token_env}")
                all_worker_tokens[worker.auth_token_env] = token
        secret_values = [*credentials.values(), *executor_tokens.values(), *all_worker_tokens.values()]
        if len(secret_values) != len(set(secret_values)):
            raise LocalGradientError("ARC, Teacher, Worker and executor credentials must be distinct")

        server_config = plan.runtime_config_container_path
        learner = runtime.learner
        server_url_host = "127.0.0.1" if learner.host in {"0.0.0.0", "::", "[::]"} else learner.host
        server_url = f"http://{server_url_host}:{learner.port}"
        runtime_env_names = tuple(self.deployment.config.runtime_env)
        base_names = set(runtime_env_names) | {"PYTHONUNBUFFERED"}
        executor_thread_env = {
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        }
        env_values = self.deployment._base_child_env({
            **dict(credentials), **all_worker_tokens,
            **{runtime.learner.gradient_executor_token_envs[key]: value
               for key, value in executor_tokens.items()},
            **executor_thread_env,
            "PYTHONUNBUFFERED": "1",
        })
        server_names = set(base_names) | set(credentials) | set(all_worker_tokens) | set(executor_thread_env)
        server_names.update(runtime.learner.gradient_executor_token_envs.values())
        for name in runtime.learner.gradient_executor_token_envs.values():
            if name not in env_values:
                raise LocalGradientError(f"central-gradient credential environment is missing: {name}")
        roles: list[dict[str, Any]] = [{
            "role": "server", "argv": [self.deployment.config.container_python, "-B", "-m", "qwen_rl",
                                          "gradients", "server", "--config", server_config],
            "environment_names": sorted(server_names), "log_path": str(logs_container / "server.log"),
        }]
        initial_policy_sync = [{
            "executor_id": executor_id, "token_env": executor_envs[executor_id],
            "server_url": server_url, "worker_ids": [worker_id],
        } for executor_id, worker_id in pairs.items()]
        if runtime.bepa.enabled:
            teacher_id = f"teacher-{run_id}"
            teacher = runtime.bepa.teacher
            roles.append({
                "role": "teacher", "worker_id": teacher_id,
                "argv": [self.deployment.config.container_python, "-B", "-m", "qwen_rl", "runtime",
                         "teacher", "--config", server_config, "--worker-id", teacher_id],
                "environment_names": sorted(base_names | {teacher.api_key_env, teacher.token_env}),
                "log_path": str(logs_container / "teacher.log"),
            })
        generated_runtime = yaml.safe_load(plan.runtime_config_host_path.read_text(encoding="utf-8"))
        artifact_identity_path = (generated_runtime.get("artifact_identity_path")
                                  if isinstance(generated_runtime, dict) else None)
        if not isinstance(artifact_identity_path, str) or not artifact_identity_path:
            raise LocalGradientError("generated central-gradient config has no mapped artifact identity")
        frozen_store = str(Path(runtime.run_dir) / "frozen-base")
        executor_base_names = base_names | set(executor_thread_env)
        worker_pins: dict[str, dict[str, Any]] = {}
        local_ids = set(runtime.learner.gradient_local_executor_worker_ids)
        identities: dict[str, dict[str, str]] = {}
        for spec in plan.specs:
            identity = self._worker_identity(
                fleet, plan, spec, timeout=self.deployment.config.worker_start_timeout,
                expect_residency=spec.worker_id in local_ids,
            )
            identities[spec.worker_id] = identity
            worker_pins[spec.worker_id] = {
                "worker_id": spec.worker_id, "auth_token_env": spec.auth_token_env,
                "endpoint": f"http://127.0.0.1:{spec.host_port}",
                "site_id": spec.site_id, "node_id": spec.node_id, "gpu": spec.gpu_index,
                "residency_enabled": spec.worker_id in local_ids,
                **identity,
            }
        executor_root = run_dir / "executors"
        executor_root.mkdir(mode=0o700)
        _require_private_dir(executor_root)
        for executor_id, worker_id in local_pairs.items():
            spec = specs[worker_id]
            identity = identities[worker_id]
            work_dir_host = executor_root / executor_id
            work_dir_host.mkdir(exist_ok=False, mode=0o700)
            work_dir_container = run_dir_container / "executors" / executor_id
            profile = _compute_profile(
                runtime, artifact_identity_path=artifact_identity_path,
                run_dir=str(work_dir_container), store_dir=frozen_store,
            )
            profile_host = work_dir_host / "compute-profile.json"
            _write_private_json(profile_host, profile)
            profile_container = str(work_dir_container / "compute-profile.json")
            executor_config_host = work_dir_host / "executor.json"
            executor_config_container = str(work_dir_container / "executor.json")
            worker_token_env = spec.auth_token_env
            executor_token_env = executor_envs[executor_id]
            executor_config = {
                "local_work_stealing": runtime.learner.gradient_local_work_stealing,
                "schema_version": "qwen-rl-gradient-executor/v1",
                "server_url": server_url,
                "executor_id": executor_id,
                "token_env": executor_token_env,
                "cuda_visible_devices": spec.gpu_uuid,
                "compute_profile": profile_container,
                "working_dir": str(work_dir_container),
                "policy_cache_dir": str(Path(runtime.run_dir) / "executor-policy-cache"),
                "poll_interval_seconds": 0.5,
                "heartbeat_interval_seconds": 10.0,
                "sharing": {
                    "node_scope": {"site_id": spec.site_id, "node_id": spec.node_id},
                    "gpu_index": spec.gpu_index,
                    "gpu_uuid": spec.gpu_uuid,
                    "shared_worker_ids": [worker_id],
                    "workers": [{
                        "id": worker_id, "scope": "service", "site_id": spec.site_id,
                        "node_id": spec.node_id, "gpu": spec.gpu_index, "gpu_uuid": spec.gpu_uuid, "port": spec.host_port,
                        "endpoint": f"http://127.0.0.1:{spec.host_port}",
                        "auth_token_env": worker_token_env,
                        "boot_id": identity["boot_id"],
                        "telemetry_epoch": identity["telemetry_epoch"],
                        "model_sha256": identity["model_sha256"],
                    }],
                    "host_backup_budget_gib": runtime.learner.residency_host_budget_gib,
                    "gpu_sharing_timeout_seconds": runtime.learner.gpu_sharing_timeout,
                    "request_timeout_seconds": runtime.learner.request_timeout,
                },
            }
            _write_private_json(executor_config_host, executor_config)
            worker_env = {
                name: value for name, value in env_values.items()
                if name in executor_base_names or name in {executor_token_env, worker_token_env}
            }
            if set(worker_env) != executor_base_names | {executor_token_env, worker_token_env}:
                raise LocalGradientError("executor credential environment is incomplete")
            roles.append({
                "role": "executor", "executor_id": executor_id,
                "argv": [self.deployment.config.container_python, "-B", "-m", "qwen_rl", "gradients",
                         "executor", "--config", executor_config_container],
                "environment_names": sorted(worker_env),
                "log_path": str(logs_container / f"executor-{executor_id}.log"),
            })
        self._pin_runtime_service_workers(plan, worker_pins)
        _write_private_json(run_dir / "worker-pins.json", {
            "schema": "qwen-rl-local-gradient-worker-pins/v1", "workers": worker_pins,
        })
        # The supervisor receives credentials by variable name through docker exec;
        # role manifests and configs contain names only, never values.
        all_names = set(base_names) | server_names | set(executor_thread_env)
        secret_names = set(credentials) | set(all_worker_tokens) | set(executor_envs.values())
        plan_path_host = run_dir / "launch-plan.json"
        plan_path_container = str(run_dir_container / "launch-plan.json")
        manifest_path_host = run_dir / "manifest.json"
        manifest_path_container = str(run_dir_container / "manifest.json")
        supervisor_argv = [self.deployment.config.container_python, "-B", "-m",
                           "qwen_rl.deployment.local_gradients", "_supervise", "--plan", plan_path_container]
        runtime_bytes = plan.runtime_config_host_path.read_bytes()
        config_fingerprint = hashlib.sha256(runtime_bytes).hexdigest()
        source_config_fingerprint = self._source_config_fingerprint()
        resume_ledger_host = fleet.state_root / "gradient-resume-ledger.json"
        resume_ledger_container = self._container_child_path(plan, resume_ledger_host)
        launch_plan = {
            "schema": _PLAN_SCHEMA, "deployment_id": plan.deployment_id, "run_id": run_id,
            "config_fingerprint": config_fingerprint, "manifest_path": manifest_path_container,
            "supervisor_argv": supervisor_argv, "workdir": self.deployment.config.container_workdir,
            "server_ready_timeout_seconds": self.deployment.config.start_timeout,
            "server_health": {"url": server_url + "/health", "token_env": "ARC_ENV_TOKEN"},
            "resume_url": server_url + "/resume",
            "resume_ledger_path": resume_ledger_container,
            "request_timeout_seconds": runtime.learner.request_timeout,
            "initial_policy_sync": initial_policy_sync,
            "worker_pins": list(worker_pins.values()),
            "secret_environment_names": sorted(secret_names),
            "roles": roles,
        }
        _write_private_json(plan_path_host, launch_plan)
        _write_private_json(run_dir / "manifest.json", {
            "schema": _SCHEMA, "deployment_id": plan.deployment_id, "run_id": run_id,
            "config_fingerprint": config_fingerprint, "status": "prepared", "roles": [],
            "worker_pins": list(worker_pins.values()),
        })
        marker = {
            "schema": _SCHEMA, "run_id": run_id,
            "manifest_container_path": manifest_path_container,
            "plan_container_path": plan_path_container,
            "config_fingerprint": config_fingerprint,
            "source_config_fingerprint": source_config_fingerprint,
            "worker_fingerprint": plan.fingerprint,
        }
        _write_private_json(root / "active.json", marker)
        return plan_path_host, plan_path_container, {
            "plan": launch_plan, "env_values": env_values, "environment_names": sorted(all_names),
            "manifest_container_path": manifest_path_container,
            "manifest_path_host": manifest_path_host,
            "status_environment": all_worker_tokens,
            "run_id": run_id,
        }

    def _launch_supervisor(self, plan: Any, launch: Mapping[str, Any]) -> dict[str, Any]:
        cfg = self.deployment.config
        env_values = dict(launch["env_values"])
        argv = ["docker", "exec", "-d", "-u", cfg.container_uid, "-w", cfg.container_workdir]
        for name in launch["environment_names"]:
            if name not in env_values:
                raise LocalGradientError(f"central-gradient environment is missing: {name}")
            argv.extend(("-e", name))
        argv.extend((cfg.container, cfg.container_python, "-B", "-m", "qwen_rl.deployment.local_gradients",
                     "_supervise", "--plan", launch["plan"]["supervisor_argv"][-1]))
        try:
            result = self.deployment.runner(argv, env=env_values, text=True, capture_output=True,
                                            timeout=cfg.docker_timeout, check=False)
        except subprocess.TimeoutExpired:
            raise LocalGradientError("central-gradient supervisor launch timed out") from None
        except OSError:
            raise LocalGradientError("central-gradient supervisor could not be launched") from None
        if result.returncode != 0:
            raise LocalGradientError("central-gradient supervisor launch failed")
        deadline = time.monotonic() + cfg.start_timeout
        while time.monotonic() < deadline:
            try:
                status = self._invoke("_status", launch["manifest_container_path"], timeout=cfg.docker_timeout,
                                      environment=launch["status_environment"])
            except LocalGradientError:
                time.sleep(0.5)
                continue
            if status.get("status") == "running":
                return status
            # docker exec -d returns before Python imports and writes the first
            # supervisor manifest. A prepared launch is not a terminal failure.
            if status.get("status") in {"degraded", "stopped", "failed"} and not status.get("launch_pending"):
                raise LocalGradientError("central-gradient supervisor failed during startup")
            time.sleep(0.5)
        raise LocalGradientError("central-gradient supervisor did not finish startup before timeout")

    def start(self, fleet: Any, plan: Any, arguments: Sequence[str]) -> int:
        if arguments:
            if "--fresh" in arguments:
                raise LocalGradientError("central-gradient local start never resets optimizer state; --fresh is unsupported")
            raise LocalGradientError("central-gradient local start takes no arguments; submit updates with local run")
        runtime = plan.runtime_config
        self.check(plan)
        active = self.status(fleet)
        if active["status"] in {"running", "starting"}:
            self._assert_legacy_runtime_inactive(
                self.deployment._load_credentials(teacher=False),
            )
            active_marker, active_dir = self._active(fleet) or ({}, Path())
            if (active_marker.get("worker_fingerprint") != plan.fingerprint
                    or active_marker.get("source_config_fingerprint") != self._source_config_fingerprint()):
                raise LocalGradientError("active central-gradient runtime uses a different config; stop it first")
            self._restore_active_pins(fleet, plan, active_dir)
            deadline = time.monotonic() + self.deployment.config.start_timeout
            while active.get("status") == "starting" and time.monotonic() < deadline:
                time.sleep(0.5)
                active = self.status(fleet)
            if active.get("status") != "running":
                raise LocalGradientError("active central-gradient supervisor did not become ready")
            print(json.dumps(active, indent=2, ensure_ascii=False))
            return 0
        if active["status"] == "degraded":
            raise LocalGradientError("central-gradient runtime is degraded; run local stop before restarting")

        credentials = self.deployment._load_credentials(teacher=runtime.bepa.enabled)
        self._assert_legacy_runtime_inactive(credentials)
        self._assert_learner_port_free(runtime)
        gateway = None
        bridge_created = False
        if runtime.bepa.enabled:
            gateway, container_ip = self.deployment._inspect_network()
            bridge_created = self.deployment._ensure_bridge(gateway, container_ip)
        started: tuple[tuple[str, str], ...] = ()
        launch: dict[str, Any] | None = None
        try:
            self.deployment._preflight(credentials, gateway)
            started = fleet.start(plan)
            _host_plan, _container_plan, launch = self._make_run(plan, credentials, started)
            status = self._launch_supervisor(plan, launch)
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0
        except (Exception, KeyboardInterrupt) as original:
            if launch is not None:
                try:
                    self._invoke("_stop", launch["manifest_container_path"],
                                  timeout=self.deployment.config.command_timeout,
                                  environment=launch["status_environment"])
                except (LocalGradientError, OSError, subprocess.SubprocessError) as cleanup_error:
                    original.add_note(f"central-gradient process cleanup could not be confirmed: {type(cleanup_error).__name__}")
            try:
                inactive = self.status(fleet)["status"] == "stopped"
            except (LocalGradientError, OSError):
                inactive = False
            if inactive and started:
                try:
                    fleet.rollback_started(plan, started)
                except (DeploymentError, OSError, subprocess.SubprocessError) as cleanup_error:
                    original.add_note(f"owned Worker rollback failed: {type(cleanup_error).__name__}")
            if bridge_created and inactive:
                try:
                    bridge_state = self.deployment._state()
                    if bridge_state is not None and self.deployment._process_matches(bridge_state):
                        self.deployment._terminate_bridge(bridge_state)
                except (DeploymentError, OSError) as cleanup_error:
                    original.add_note(f"owned Teacher bridge cleanup failed: {type(cleanup_error).__name__}")
            raise

    def stop(self, fleet: Any, arguments: Sequence[str]) -> int:
        if arguments:
            raise LocalGradientError("central-gradient local stop takes no arguments")
        credentials = self.deployment._load_credentials(teacher=False)
        self._assert_legacy_runtime_inactive(credentials)
        runtime_source = Path(self.deployment.config.default_config).relative_to("/work")
        runtime = _load_runtime_config(self.deployment.root / runtime_source)
        active = self._active(fleet)
        if active is not None:
            marker, _run_dir = active
            status = self._invoke("_stop", marker["manifest_container_path"],
                                  timeout=self.deployment.config.command_timeout,
                                  environment=self._worker_status_environment(fleet, _run_dir))
            if status.get("status") != "stopped":
                raise LocalGradientError("central-gradient processes did not stop cleanly; Workers were left untouched")
        self._assert_learner_port_free(runtime)
        fleet.stop_owned()
        bridge = self.deployment._state()
        if bridge is not None:
            self.deployment._terminate_bridge(bridge)
        print(json.dumps({"backend": "central_gradient", "status": "stopped"}, indent=2))
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    return _internal_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
