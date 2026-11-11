"""Explicit rollback of uncommitted work for a stopped managed instance."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
from typing import Any, Mapping

from ..learner.stop_control import StopAfterCheckpointControl

_PROOF_SCHEMA = "qwen-rl-instance-stop-now-proof/v1"
_RECEIPT_SCHEMA = "qwen-rl-managed-stop-now/v1"
_TRANSACTION_SCHEMA = "qwen-rl-managed-stop-now-transaction/v1"
_CONTROL_FILES = (
    "async-checkpoint-intent.json",
    "pending-attempt.json",
    "prepared-window.json",
)
_STOP_AFTER_STATUSES = frozenset({"accepted", "waiting_durable", "completed", "blocked"})
_STOP_AFTER_TRANSACTION_STATES = frozenset({"not_started", "in_progress", "committed", "already_durable"})
_STOP_AFTER_BEPA_STATES = frozenset({
    "pending", "prepared", "optimizer_entered", "commit_unknown", "committed", "released",
    "unknown", "not_applicable",
})
_TERMINAL_DESCENDANT_OUTCOMES = frozenset({
    "SIGTERM", "SIGTERM_sent", "owner_shutdown_requested", "SIGKILL", "SIGKILL_sent",
    "exited", "already_exited", "exited_during_capture", "selected",
})
_OPERATION_ID = re.compile(r"^[0-9a-f]{32}$")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_MAX_CONTROL_BYTES = 4 * 1024 * 1024


def _safe_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"managed stop directory is unavailable: {path.name}") from exc
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
        raise RuntimeError(f"managed stop directory is unsafe: {path.name}")


def _read_regular(path: Path, *, max_bytes: int = _MAX_CONTROL_BYTES) -> bytes | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"cannot inspect managed stop file: {path.name}") from exc
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1 or info.st_uid != os.getuid()):
        raise RuntimeError(f"managed stop file is unsafe: {path.name}")
    if info.st_size > max_bytes:
        raise RuntimeError(f"managed stop file exceeds its bounded size: {path.name}")
    data = path.read_bytes()
    if len(data) != info.st_size:
        raise RuntimeError(f"managed stop file changed while reading: {path.name}")
    return data


def _read_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _atomic_write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    _safe_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temp_path = Path(temporary)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def _operation_lock_is_held(state_dir: Path) -> None:
    path = state_dir / "operation.lock"
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1):
            raise RuntimeError("managed stop requires the private per-deployment operation lock")
        descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise RuntimeError("managed stop requires the per-deployment operation lock") from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise RuntimeError("managed stop operation lock changed during validation")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise RuntimeError("managed stop caller does not hold the per-deployment operation lock")
    finally:
        os.close(descriptor)


def _expected_process_records(manifest: Mapping[str, Any], nested: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    owner = manifest.get("owner")
    roles = manifest.get("roles")
    if (owner is not None and not isinstance(owner, Mapping)) or not isinstance(roles, list):
        raise RuntimeError("stopped instance manifest has no complete process inventory")
    if isinstance(owner, Mapping):
        records.append(owner)
    records.extend(row for row in roles if isinstance(row, Mapping))
    if len(records) != (1 if isinstance(owner, Mapping) else 0) + len(roles):
        raise RuntimeError("stopped instance manifest contains an invalid role record")
    if nested is not None:
        supervisor = nested.get("supervisor")
        nested_roles = nested.get("roles")
        if not isinstance(supervisor, Mapping) or not isinstance(nested_roles, list):
            raise RuntimeError("nested gradient manifest has no complete process inventory")
        records.append(supervisor)
        records.extend(row for row in nested_roles if isinstance(row, Mapping))
        if len(records) != (1 if isinstance(owner, Mapping) else 0) + 1 + len(roles) + len(nested_roles):
            raise RuntimeError("nested gradient manifest contains an invalid role record")
    unique: dict[tuple[int, int, tuple[str, ...]], Mapping[str, Any]] = {}
    for record in records:
        unique.setdefault(_record_identity(record), record)
    return list(unique.values())


def normalize_stop_process_proof(
    manifest: Mapping[str, Any], nested: Mapping[str, Any] | None,
    selected_snapshot: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split exact manifest roots from the separately audited process descendants."""
    expected = _expected_process_records(manifest, nested)
    root_by_pid = {}
    for row in expected:
        identity = _record_identity(row)
        if identity[0] in root_by_pid:
            raise RuntimeError("stop snapshot has ambiguous manifest root PIDs")
        root_by_pid[identity[0]] = (identity, row)
    snapshot: dict[tuple[int, int, tuple[str, ...]], Mapping[str, Any]] = {}
    for row in selected_snapshot:
        pid = row.get("pid")
        if type(pid) is int and pid in root_by_pid:
            identity = _record_identity(row)
            if identity != root_by_pid[pid][0]:
                raise RuntimeError("stop snapshot reuses a manifest-owned process PID")
        else:
            identity = _descendant_identity(row)
        if identity in snapshot:
            raise RuntimeError("stop snapshot contains a duplicate process identity")
        if row.get("exited") is not True:
            raise RuntimeError("stop snapshot contains a live or unverified process")
        snapshot[identity] = row
    _validate_terminal_descendant_leaves(selected_snapshot)

    def outcome(row: Mapping[str, Any]) -> str:
        value = row.get("outcome")
        if value in {"SIGTERM", "SIGTERM_sent", "owner_shutdown_requested"}:
            return "SIGTERM"
        if value in {"SIGKILL", "SIGKILL_sent"}:
            return "SIGKILL"
        if value in {"exited", "already_exited", "exited_during_capture", "selected"}:
            return "exited"
        raise RuntimeError("stop snapshot has an unsupported terminal process outcome")

    expected_ids = {_record_identity(row) for row in expected}
    stopped_owner = manifest.get("stopped_owner")
    verified: list[dict[str, Any]] = []
    for record in expected:
        identity = _record_identity(record)
        row = snapshot.get(identity)
        if row is None:
            raise RuntimeError("stop snapshot omits a manifest-owned process identity")
        verified.append({**dict(record), "outcome": outcome(row), "exited": True})

    root_by_id = {_record_identity(row): row for row in expected}
    root_by_pid = {identity[0]: (identity, row) for identity, row in root_by_id.items()}
    descendants = []
    for identity, row in snapshot.items():
        if identity in expected_ids:
            continue
        if row.get("root_role", row.get("role")) == "instance-supervisor":
            if stopped_owner is not None and identity != _record_identity(stopped_owner, allow_pid1=True):
                raise RuntimeError("stop snapshot supervisor identity differs from the preserved owner")
            continue
        root_pid = row.get("root_pid")
        root_start = row.get("root_starttime")
        root_pair = root_by_pid.get(root_pid)
        if (root_pair is None or root_pair[0][1] != root_start
                or row.get("root_role") != root_pair[1].get("role")):
            raise RuntimeError("stop descendant is not rooted at a manifest-owned role")
        descendants.append({**dict(row), "outcome": outcome(row), "exited": True})
    descendants.sort(key=lambda row: (row["root_pid"], row["pid"], row["starttime"]))
    _validate_descendant_audit(expected, descendants)
    return verified, descendants


def _validate_descendant_audit(
    roots: list[Mapping[str, Any]], descendants: list[Mapping[str, Any]],
) -> None:
    root_by_pid = {}
    for row in roots:
        pid, _starttime, _argv = _record_identity(row)
        if pid in root_by_pid:
            raise RuntimeError("stop descendant audit has ambiguous manifest root identities")
        root_by_pid[pid] = row
    by_pid: dict[int, Mapping[str, Any]] = {}
    for row in descendants:
        pid, starttime, argv = _descendant_identity(row)
        if (row.get("exited") is not True
                or row.get("outcome") not in {"SIGTERM", "SIGKILL", "exited"}):
            raise RuntimeError("stop descendant audit includes a live or unverified process")
        if pid in by_pid:
            raise RuntimeError("stop descendant audit repeats a process identity")
        by_pid[pid] = row
        root = root_by_pid.get(row.get("root_pid"))
        if (root is None or root.get("starttime") != row.get("root_starttime")
                or root.get("role") != row.get("root_role")):
            raise RuntimeError("stop descendant audit names an unknown role root")
        expected_gpu = root.get("gpu_uuid")
        if expected_gpu is not None and row.get("gpu_uuid") != expected_gpu:
            raise RuntimeError("stop descendant audit GPU identity differs from its Worker")
        native_child = any("ninfer-serve" in Path(arg).name or "residency_exec" in arg
                           for arg in argv)
        if native_child and row.get("gpu_uuid_verified") is not True:
            raise RuntimeError("native Worker descendant lacks UUID ownership evidence")

    _validate_terminal_descendant_leaves(descendants)

    for row in descendants:
        root_pid = int(row["root_pid"])
        current = row
        visited = set()
        while True:
            key = (int(current["pid"]), int(current["starttime"]))
            if key in visited:
                raise RuntimeError("stop descendant audit contains an ancestry cycle")
            visited.add(key)
            parent_pid = current.get("parent_pid")
            parent_start = current.get("parent_starttime")
            root = root_by_pid[root_pid]
            if parent_pid == root_pid and parent_start == root.get("starttime"):
                break
            parent = by_pid.get(parent_pid)
            if (parent is None or parent.get("starttime") != parent_start
                    or parent.get("root_pid") != root_pid):
                raise RuntimeError("stop descendant audit has an incomplete parent chain")
            current = parent


def _validate_terminal_descendant_leaves(records: list[Mapping[str, Any]]) -> None:
    for row in records:
        if row.get("argv") != []:
            continue
        if any(
            child.get("parent_pid") == row.get("pid")
            and child.get("parent_starttime") == row.get("starttime")
            for child in records
        ):
            raise RuntimeError("terminal descendant with an unknown command has recorded children")


def _descendant_identity(record: Mapping[str, Any]) -> tuple[int, int, tuple[str, ...]]:
    """Validate descendant identity, narrowly permitting a verified terminal zombie command line."""
    argv = record.get("argv")
    if argv != []:
        return _record_identity(record)
    pid, starttime = record.get("pid"), record.get("starttime")
    if (type(pid) is not int or pid < 2 or type(starttime) is not int
            or record.get("exited") is not True
            or record.get("outcome") not in _TERMINAL_DESCENDANT_OUTCOMES
            or record.get("terminal_state") not in {"Z", "X"}
            or record.get("terminal_identity_verified") is not True):
        raise RuntimeError("empty descendant command line lacks verified terminal identity")
    return pid, starttime, ()


def _record_identity(
    record: Mapping[str, Any], *, allow_pid1: bool = False,
) -> tuple[int, int, tuple[str, ...]]:
    pid, starttime, argv = record.get("pid"), record.get("starttime"), record.get("argv")
    if (type(pid) is not int or pid < (1 if allow_pid1 else 2) or type(starttime) is not int
            or not isinstance(argv, list) or not argv
            or any(not isinstance(item, str) for item in argv)):
        raise RuntimeError("managed process identity is malformed")
    return pid, starttime, tuple(argv)


def _idle_guard_matches(record: Mapping[str, Any]) -> bool:
    from .local_gradients import _proc_record

    pid, starttime, argv = _record_identity(record, allow_pid1=True)
    return _proc_record(pid) == (starttime, argv)


def _validate_stop_proof(deployment: Any, manifest_path: Path, proof: Mapping[str, Any]) -> dict[str, Any]:
    from .instance import _SCHEMA as _INSTANCE_SCHEMA, _read_private_json

    def identity_state(record: Mapping[str, Any]) -> str:
        from .instance_stop_control import _identity_state

        return _identity_state(dict(record))

    expected_manifest_path = (deployment.state_dir / "active.json").resolve()
    if Path(manifest_path).expanduser().resolve() != expected_manifest_path:
        raise RuntimeError("stop proof does not refer to this deployment's active manifest")
    if (proof.get("schema") != _PROOF_SCHEMA or proof.get("status") != "stopped"
            or not isinstance(proof.get("operation_id"), str)
            or _OPERATION_ID.fullmatch(proof["operation_id"]) is None
            or proof.get("deployment_id") != hashlib.sha256(
                str(deployment.path).encode("utf-8")
            ).hexdigest()[:20]):
        raise RuntimeError("managed stop proof is incomplete or belongs to another deployment")
    manifest = _read_private_json(expected_manifest_path)
    if (not isinstance(manifest, dict) or manifest.get("schema") != _INSTANCE_SCHEMA
            or manifest.get("status") != "stopped"):
        raise RuntimeError("managed stop requires the selected instance manifest to be stopped")
    run_id = manifest.get("run_id")
    if (not isinstance(run_id, str) or _OPERATION_ID.fullmatch(run_id) is None
            or proof.get("run_id") != run_id):
        raise RuntimeError("managed stop proof run identity differs from the stopped manifest")
    nested_path_value = manifest.get("gradient_manifest")
    nested_path = None
    nested = None
    nested_missing = False
    if nested_path_value is not None:
        if not isinstance(nested_path_value, str) or not nested_path_value:
            raise RuntimeError("stopped instance has an invalid nested gradient manifest path")
        nested_path = Path(nested_path_value).expanduser()
        try:
            nested_path.resolve().relative_to(deployment.state_dir.resolve())
        except (OSError, ValueError) as exc:
            raise RuntimeError("nested gradient manifest escapes the selected deployment state") from exc
        current = deployment.state_dir
        for part in nested_path.absolute().relative_to(deployment.state_dir.resolve()).parts:
            current = current / part
            if current.is_symlink():
                raise RuntimeError("nested gradient manifest path contains a symlink")
        if nested_path.exists():
            if proof.get("nested_manifest_absent") is True:
                raise RuntimeError("stop proof marks an existing nested manifest absent")
            nested = _read_private_json(nested_path)
            if not isinstance(nested, dict) or nested.get("status") != "stopped":
                raise RuntimeError("nested gradient manager is not proven stopped")
        else:
            if proof.get("nested_manifest_absent") is not True:
                raise RuntimeError("missing nested gradient manifest requires an explicit empty-inventory proof")
            nested_missing = True
    if proof.get("nested_manifest_path") != (str(nested_path) if nested_path is not None else None):
        raise RuntimeError("stop proof nested manifest identity differs from the stopped manifest")

    expected = _expected_process_records(manifest, nested)
    verified = proof.get("verified_processes")
    if not isinstance(verified, list) or len(verified) != len(expected):
        raise RuntimeError("stop proof does not cover every managed process identity")
    expected_ids = {_record_identity(record) for record in expected}
    verified_ids = set()
    for row in verified:
        if (not isinstance(row, Mapping) or row.get("exited") is not True
                or row.get("outcome") not in {"SIGTERM", "SIGKILL", "exited"}):
            raise RuntimeError("stop proof contains an unverified process outcome")
        verified_ids.add(_record_identity(row))
    if verified_ids != expected_ids:
        raise RuntimeError("stop proof process identities differ from the managed manifests")
    descendants = proof.get("descendant_audit")
    if not isinstance(descendants, list):
        raise RuntimeError("stop proof has no separate descendant ownership audit")
    _validate_descendant_audit(expected, descendants)
    owner = manifest.get("owner")
    if isinstance(owner, Mapping):
        raise RuntimeError("stopped instance manifest must clear the active owner")
    else:
        guard = proof.get("idle_guard")
        if guard is None:
            if "idle_guard" not in proof:
                raise RuntimeError("stop proof omits its idle guard decision")
        else:
            if (not isinstance(guard, Mapping) or guard.get("status") != "alive_without_owner_lock"
                    or not _idle_guard_matches(guard)):
                raise RuntimeError("stopped instance idle guard identity is not live and verified")
            guard_identity = _record_identity(guard, allow_pid1=True)
            if guard_identity in expected_ids:
                raise RuntimeError("idle guard process is incorrectly listed as a stopped role")
    stopped_owner = manifest.get("stopped_owner")
    proof_owner = proof.get("stopped_owner")
    if stopped_owner is None:
        if proof_owner is not None:
            raise RuntimeError("stop proof includes an owner that the manifest did not preserve")
        if guard is None:
            raise RuntimeError("a missing idle guard requires a preserved terminal owner identity")
    else:
        if not isinstance(stopped_owner, Mapping) or not isinstance(proof_owner, Mapping):
            raise RuntimeError("stopped owner identity is missing from the stop proof")
        stopped_owner_identity = _record_identity(stopped_owner, allow_pid1=True)
        if stopped_owner_identity != _record_identity(proof_owner, allow_pid1=True):
            raise RuntimeError("stopped owner identity differs from the proof")
        owner_outcome = proof_owner.get("outcome")
        same_as_guard = (
            isinstance(guard, Mapping)
            and stopped_owner_identity == _record_identity(guard, allow_pid1=True)
        )
        if guard is None:
            # A missing idle guard is permitted only when the original, non-PID1
            # owner itself has a fresh terminal identity check and no old guard
            # or managed process remains live or ambiguous.
            _record_identity(stopped_owner)
            if stopped_owner_identity in expected_ids:
                raise RuntimeError("stopped owner is incorrectly listed as a managed role")
            if owner_outcome != "exited" or identity_state(stopped_owner) != "exited":
                raise RuntimeError("original instance owner is not freshly proven exited")
            old_guard = manifest.get("idle_guard")
            if old_guard is not None:
                if not isinstance(old_guard, Mapping):
                    raise RuntimeError("stopped manifest idle guard identity is malformed")
                _record_identity(old_guard, allow_pid1=True)
                if identity_state(old_guard) != "exited":
                    raise RuntimeError("stopped manifest still has a live idle guard")
            for record in [*expected, *descendants]:
                if identity_state(record) != "exited":
                    raise RuntimeError("a managed process remains live or ambiguous without an idle guard")
        elif same_as_guard:
            if owner_outcome != "demoted_to_idle_guard" or not _idle_guard_matches(guard):
                raise RuntimeError("live owner was not proven as the ownerless idle guard")
        else:
            if stopped_owner_identity[0] == 1:
                raise RuntimeError("PID 1 original owner requires a matching live idle guard")
            if owner_outcome != "exited" or identity_state(stopped_owner) != "exited":
                raise RuntimeError("original instance owner did not freshly exit before cleanup")
    for record in [*expected, *descendants]:
        if identity_state(record) != "exited":
            raise RuntimeError("a managed process remains live or ambiguous before cleanup")
    return {"manifest": manifest, "nested": nested, "nested_path": nested_path,
            "manifest_path": expected_manifest_path, "run_id": run_id,
            "verified_process_count": len(verified), "nested_manifest_missing": nested_missing}


def _sidecar_capture(root: Path) -> dict[str, dict[str, str | int]]:
    result = {}
    for name in _CONTROL_FILES:
        data = _read_regular(root / name)
        if data is None:
            continue
        result[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "base64": base64.b64encode(data).decode("ascii"),
        }
    return result


def _capture_superseded_stop_control(runtime: Any) -> dict[str, Any] | None:
    control_dir = Path(runtime.run) / "control"
    try:
        info = control_dir.lstat()
    except FileNotFoundError:
        return None
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
        raise RuntimeError("Learner stop-control directory is unsafe")
    path = control_dir / "stop-after-checkpoint.json"
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise RuntimeError("Learner stop-after-checkpoint receipt is unsafe")
    data = _read_regular(path, max_bytes=64 * 1024)
    if data is None:
        return None
    receipt = _read_json_bytes(data, "Learner stop-after-checkpoint receipt")
    if set(receipt) != StopAfterCheckpointControl.RECEIPT_FIELDS or receipt.get("schema") != StopAfterCheckpointControl.SCHEMA:
        raise RuntimeError("Learner stop-after-checkpoint receipt schema is unsupported")
    if receipt.get("mode") != "after_checkpoint" or receipt.get("status") not in _STOP_AFTER_STATUSES:
        raise RuntimeError("Learner stop-after-checkpoint receipt state is unsupported")
    if type(receipt.get("request_accepted")) is not bool:
        raise RuntimeError("Learner stop receipt acceptance flag is invalid")
    if not (receipt["status"] in {"accepted", "waiting_durable"}
            or (receipt["status"] == "blocked" and receipt["request_accepted"] is True)):
        return None
    if receipt["status"] in {"accepted", "waiting_durable"} and receipt["request_accepted"] is not True:
        raise RuntimeError("pending Learner stop receipt lacks acceptance evidence")
    operation_id = receipt.get("operation_id")
    if (not isinstance(operation_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", operation_id)
            or not isinstance(receipt.get("run_id"), str) or not receipt["run_id"]
            or not isinstance(receipt.get("lineage_id"), str)
            or not isinstance(receipt.get("engine_epoch"), str)
            or _HEX32.fullmatch(receipt["lineage_id"]) is None
            or _HEX32.fullmatch(receipt["engine_epoch"]) is None):
        raise RuntimeError("pending Learner stop-after-checkpoint receipt identity is invalid")
    for name in ("target_update", "target_policy_version"):
        if type(receipt.get(name)) is not int or receipt[name] < 0:
            raise RuntimeError(f"pending Learner stop receipt {name} is invalid")
    for name in ("completed_update", "durable_policy_version"):
        value = receipt.get(name)
        if value is not None and (type(value) is not int or value < 0):
            raise RuntimeError(f"pending Learner stop receipt {name} is invalid")
    snapshot_id = receipt.get("checkpoint_snapshot_id")
    if snapshot_id is not None and (not isinstance(snapshot_id, str) or _HEX32.fullmatch(snapshot_id) is None):
        raise RuntimeError("pending Learner stop receipt checkpoint identity is invalid")
    if (receipt.get("transaction_status") not in _STOP_AFTER_TRANSACTION_STATES
            or receipt.get("bepa_commit_state") not in _STOP_AFTER_BEPA_STATES
            or type(receipt.get("boundary_committed")) is not bool):
        raise RuntimeError("pending Learner stop receipt transaction state is invalid")
    for name in ("created_at", "updated_at"):
        value = receipt.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"pending Learner stop receipt {name} is invalid")
    error = receipt.get("error")
    if error is not None and (not isinstance(error, Mapping) or set(error) != {"type", "message"}
                              or any(not isinstance(error.get(key), str) for key in ("type", "message"))):
        raise RuntimeError("pending Learner stop receipt error is invalid")
    bepa_attempt_id = receipt.get("bepa_attempt_id")
    if bepa_attempt_id is not None and (not isinstance(bepa_attempt_id, str) or not bepa_attempt_id):
        raise RuntimeError("pending Learner stop receipt BEPA attempt identity is invalid")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "base64": base64.b64encode(data).decode("ascii"),
        "receipt": receipt,
    }


def _validate_superseded_stop_control(
    captured: Any,
    *,
    proof: Mapping[str, Any],
    core_status: Mapping[str, Any],
) -> dict[str, Any] | None:
    if captured is None:
        return None
    if not isinstance(captured, Mapping) or not isinstance(captured.get("receipt"), Mapping):
        raise RuntimeError("captured Learner stop-after-checkpoint control is malformed")
    receipt = captured["receipt"]
    expected = {
        "operation_id": receipt.get("operation_id"),
        "run_id": receipt.get("run_id"),
        "target_update": receipt.get("target_update"),
        "lineage_id": receipt.get("lineage_id"),
        "engine_epoch": receipt.get("engine_epoch"),
        "target_policy_version": receipt.get("target_policy_version"),
    }
    proof_values = {
        "operation_id": proof.get("supersedes_operation_id"),
        "run_id": proof.get("superseded_run_id", proof.get("superseded_learner_run_id")),
        "target_update": proof.get("superseded_target_update"),
        "lineage_id": proof.get("superseded_lineage_id"),
        "engine_epoch": proof.get("superseded_engine_epoch"),
        "target_policy_version": proof.get("superseded_target_policy_version"),
    }
    if any(proof_values[key] != value for key, value in expected.items()):
        raise RuntimeError("stop-now proof does not identify the pending Learner stop request")
    if (receipt.get("lineage_id") != core_status.get("lineage_id")
            or receipt.get("engine_epoch") != core_status.get("engine_epoch")):
        raise RuntimeError("pending Learner stop request belongs to another durable Core lineage")
    return {
        "operation_id": receipt["operation_id"],
        "run_id": receipt["run_id"],
        "target_update": receipt["target_update"],
        "target_policy_version": receipt["target_policy_version"],
        "lineage_id": receipt["lineage_id"],
        "engine_epoch": receipt["engine_epoch"],
        "status": receipt["status"],
        "request_accepted": receipt["request_accepted"],
        "sha256": captured["sha256"],
        "size_bytes": captured["size_bytes"],
        "source_path": captured["path"],
        "base64": captured["base64"],
    }


def _move_superseded_stop_control(directory: Path, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError("validated superseded stop receipt is malformed")
    try:
        data = base64.b64decode(value["base64"], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise RuntimeError("superseded stop receipt encoding is invalid") from exc
    if len(data) != value.get("size_bytes") or hashlib.sha256(data).hexdigest() != value.get("sha256"):
        raise RuntimeError("superseded stop receipt identity changed after validation")
    source = Path(str(value.get("source_path")))
    target = directory / "files" / "stop-after-checkpoint.json"
    existing_target = _read_regular(target, max_bytes=64 * 1024)
    source_data = _read_regular(source, max_bytes=64 * 1024)
    if existing_target is not None:
        if existing_target != data:
            raise RuntimeError("quarantined stop-after-checkpoint receipt changed")
        if source_data is not None:
            if source_data != data:
                raise RuntimeError("stop-after-checkpoint receipt changed during retry")
            source.unlink()
    elif source_data is not None:
        if source_data != data:
            raise RuntimeError("stop-after-checkpoint receipt changed after validation")
        os.replace(source, target)
        for parent in (source.parent, target.parent):
            descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    else:
        _atomic_write(target, data)
    return {key: value[key] for key in (
        "operation_id", "run_id", "target_update", "target_policy_version", "lineage_id",
        "engine_epoch", "status", "sha256", "size_bytes",
    )} | {"request_accepted": value["request_accepted"],
         "quarantine_path": str(target)}


def _captured_bytes(entries: Mapping[str, Any], name: str) -> bytes:
    item = entries.get(name)
    if not isinstance(item, Mapping) or not isinstance(item.get("base64"), str):
        raise RuntimeError("managed stop transaction omitted a captured control sidecar")
    try:
        data = base64.b64decode(item["base64"], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise RuntimeError("managed stop captured sidecar encoding is invalid") from exc
    if (len(data) != item.get("size_bytes")
            or hashlib.sha256(data).hexdigest() != item.get("sha256")):
        raise RuntimeError("managed stop captured sidecar identity is invalid")
    return data


def _quarantine_dir(root: Path, operation_id: str) -> Path:
    parent = root / ".stop-now-quarantine"
    _safe_directory(parent, create=True)
    target = parent / operation_id
    _safe_directory(target, create=True)
    files = target / "files"
    _safe_directory(files, create=True)
    return target


def _transaction_path(directory: Path) -> Path:
    return directory / "transaction.json"


def _write_transaction(path: Path, transaction: Mapping[str, Any]) -> None:
    existing = _read_regular(path, max_bytes=16 * 1024 * 1024)
    encoded = (json.dumps(dict(transaction), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if existing is not None:
        prior = _read_json_bytes(existing, "managed stop transaction")
        immutable_keys = ("schema", "operation_id", "deployment_id", "run_id", "checkpoint", "sidecars")
        if any(prior.get(key) != transaction.get(key) for key in immutable_keys):
            raise RuntimeError("managed stop retry transaction identity changed")
    _atomic_write(path, encoded)


def _read_transaction(path: Path) -> dict[str, Any] | None:
    data = _read_regular(path, max_bytes=16 * 1024 * 1024)
    if data is None:
        return None
    value = _read_json_bytes(data, "managed stop transaction")
    if value.get("schema") != _TRANSACTION_SCHEMA:
        raise RuntimeError("managed stop transaction schema is unsupported")
    return value


def _move_control_files(root: Path, directory: Path, entries: Mapping[str, Any]) -> list[dict[str, Any]]:
    target_root = directory / "files"
    moved = []
    for name in _CONTROL_FILES:
        entry = entries.get(name)
        if entry is None:
            continue
        captured = _captured_bytes(entries, name)
        source = root / name
        target = target_root / name
        existing_target = _read_regular(target)
        source_bytes = _read_regular(source)
        if existing_target is not None:
            if existing_target != captured:
                raise RuntimeError(f"quarantined control sidecar changed: {name}")
            if source_bytes is not None:
                if source_bytes != captured:
                    raise RuntimeError(f"control sidecar changed during retry: {name}")
                source.unlink()
        elif source_bytes is not None:
            if source_bytes != captured:
                raise RuntimeError(f"control sidecar changed after validation: {name}")
            os.replace(source, target)
            parent_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            target_parent_fd = os.open(target_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(target_parent_fd)
            finally:
                os.close(target_parent_fd)
        else:
            _atomic_write(target, captured)
        moved.append({"name": name, "sha256": entry["sha256"], "size_bytes": entry["size_bytes"],
                      "path": str(target)})
    return moved


@contextmanager
def _cuda_hidden():
    names = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_recovery_engine(deployment: Any, runtime: Any, owner_lock: Any):
    from . import instance_recovery
    from ..learner.app import _load_engine

    config = instance_recovery._engine_runtime(deployment, runtime)
    from .instance_recovery import _RecoveryTokenizer, _cpu_thread_limit, _offline_executor_tokens

    with _offline_executor_tokens(config["learner"]["gradient_executor_token_envs"]):
        with _cpu_thread_limit(), _cuda_hidden():
            return _load_engine(
                config,
                recovery_semantics=None,
                tokenizer=_RecoveryTokenizer(),
                process_owner_lock=owner_lock,
                recovery_only=True,
            )


def _close_recovery_engine(engine: Any) -> None:
    if engine is None:
        return
    core = getattr(engine, "core", None)
    if core is not None:
        try:
            core.close()
        except Exception:
            # A dirty Core is expected before an explicit source-checkpoint rollback.
            pass


def _core_identity(status: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "lineage_id", "engine_epoch", "snapshot_id", "checkpoint_generation",
        "policy_version", "optimizer_steps", "scheduler_steps", "last_update_context",
    )
    return {field: status.get(field) for field in fields}


def _assert_core_ready(before: Mapping[str, Any], after: Mapping[str, Any], pointer: Mapping[str, Any]) -> None:
    if _core_identity(before) != _core_identity(after):
        raise RuntimeError("managed stop changed durable Core state or the saved run cursor")
    if (after.get("checkpoint_state") != "durable" or after.get("dirty") is not False
            or after.get("pending_attempt") is not None or after.get("checkpoint_intent") is not None
            or after.get("pending_window") is not None
            or after.get("snapshot_id") != pointer.get("snapshot_id")
            or after.get("checkpoint_generation") != pointer.get("generation")):
        raise RuntimeError("latest full checkpoint did not reload as a clean durable Core")


def _bepa_ledger_snapshot(runtime: Any) -> dict[str, Any]:
    """Read budget, reservation, update-attempt, and usage rows without opening BepaStore."""
    if not runtime.bepa.enabled:
        return {"state": "disabled", "row_counts": {}, "digest": None, "budgets": []}
    from ..runtime.config import _resolve_path

    root = (_resolve_path(runtime.bepa.state_dir)
            if runtime.bepa.state_dir else runtime.run / "bepa")
    database_path = root / "state.sqlite3"
    try:
        info = database_path.lstat()
    except FileNotFoundError:
        return {"state": "missing", "row_counts": {}, "digest": None, "budgets": []}
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError("BEPA state database is unsafe during managed stop")
    uri = database_path.resolve().as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN")
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        names = (
            "budget_scopes", "rescue_jobs", "reservations", "update_attempts",
            "usage_events", "cache_entries", "episodes", "events",
        )
        if not set(names).issubset(tables):
            raise RuntimeError("BEPA state database schema is incomplete during managed stop")
        rows = {}
        for name in names:
            primary_key = {
                "budget_scopes": "scope_id", "rescue_jobs": "job_id",
                "reservations": "reservation_id", "update_attempts": "attempt_id",
                "usage_events": "scope_id,idempotency_key", "cache_entries": "cache_entry_id",
                "episodes": "episode_id", "events": "sequence",
            }[name]
            rows[name] = [tuple(row) for row in connection.execute(
                f"SELECT * FROM {name} ORDER BY {primary_key}"
            ).fetchall()]
        budgets = [
            {key: row[key] for key in (
                "scope_id", "max_total_tokens", "max_jobs", "used_tokens", "reserved_tokens", "usage_unknown",
            )}
            for row in connection.execute(
                "SELECT scope_id,max_total_tokens,max_jobs,used_tokens,reserved_tokens,usage_unknown "
                "FROM budget_scopes ORDER BY scope_id"
            ).fetchall()
        ]
        connection.execute("COMMIT")
    except sqlite3.Error as exc:
        raise RuntimeError("BEPA ledger could not be read consistently during managed stop") from exc
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "state": "read_only_snapshot",
        "row_counts": {name: len(value) for name, value in rows.items()},
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "budgets": budgets,
    }


def _verify_prior_receipt(path: Path, *, operation_id: str, deployment_id: str,
                          run_id: str, pointer: Mapping[str, Any]) -> dict[str, Any] | None:
    data = _read_regular(path, max_bytes=2 * 1024 * 1024)
    if data is None:
        return None
    receipt = _read_json_bytes(data, "managed stop receipt")
    checkpoint = receipt.get("checkpoint")
    if (receipt.get("schema") != _RECEIPT_SCHEMA or receipt.get("status") != "passed"
            or receipt.get("operation_id") != operation_id
            or receipt.get("deployment_id") != deployment_id
            or receipt.get("run_id") != run_id
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("generation") != pointer.get("generation")
            or checkpoint.get("snapshot_id") != pointer.get("snapshot_id")):
        raise RuntimeError("prior managed stop receipt differs from the selected durable checkpoint")
    return receipt


def discard_stopped_instance_work(
    deployment: Any,
    *,
    manifest_path: Path,
    operation_id: str,
    stop_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Discard only current speculative central-gradient work after managed stop-now.

    The caller must hold the deployment operation lock and provide the complete
    stop proof. This function acquires the stopped instance-owner and optimizer
    locks, selects ``latest.json`` only after those fences, and never changes the
    selected checkpoint payload or BEPA database.
    """
    from .instance_plan import InstanceDeployment

    if not isinstance(deployment, InstanceDeployment):
        raise TypeError("managed stop discard requires an InstanceDeployment")
    if not isinstance(operation_id, str) or _OPERATION_ID.fullmatch(operation_id) is None:
        raise ValueError("operation_id must be a lowercase UUID hex string")
    if not isinstance(stop_receipt, Mapping) or stop_receipt.get("operation_id") != operation_id:
        raise RuntimeError("stop proof operation identity differs from the requested discard")
    _operation_lock_is_held(deployment.state_dir)
    proof_state = _validate_stop_proof(deployment, manifest_path, stop_receipt)
    from .instance_recovery import _checkpoint_preflight, _instance_owner_lock, _resolved_recovery_dir
    from .instance_plan import _load_runtime
    from ..learner.gradient_server.ownership import ProcessOwnerLock

    owner_descriptor = _instance_owner_lock(deployment.state_dir)
    optimizer_owner_lock = None
    engine = None
    try:
        # Re-read proof under both lifetime locks before selecting latest.
        proof_state = _validate_stop_proof(deployment, manifest_path, stop_receipt)
        runtime = _load_runtime(deployment)
        if runtime.learner.update_backend != "central_gradient":
            raise RuntimeError("stop-now discard supports only the managed central-gradient backend")
        root = _resolved_recovery_dir(runtime)
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError("central recovery directory is missing or unsafe")
        optimizer_owner_lock = ProcessOwnerLock.for_checkpoint_dir(root)
        bepa_before = _bepa_ledger_snapshot(runtime)
        pointer, checkpoint_error = _checkpoint_preflight(root)
        if pointer is None:
            raise RuntimeError(f"latest full checkpoint is not recoverable ({checkpoint_error})")
        deployment_id = stop_receipt["deployment_id"]
        run_id = proof_state["run_id"]
        directory = _quarantine_dir(root, operation_id)
        receipt_path = directory / "receipt.json"
        prior_receipt = _verify_prior_receipt(
            receipt_path, operation_id=operation_id, deployment_id=deployment_id,
            run_id=run_id, pointer=pointer,
        )
        if prior_receipt is not None:
            return prior_receipt

        pending_transactions = []
        quarantine_root = directory.parent
        for child in quarantine_root.iterdir():
            if child == directory or child.is_symlink() or not child.is_dir():
                continue
            pending = _read_transaction(child / "transaction.json")
            if pending is not None and pending.get("status") != "complete":
                pending_transactions.append(child.name)
        if pending_transactions:
            raise RuntimeError(
                "another managed stop transaction is incomplete; retry it with operation_id "
                + sorted(pending_transactions)[0]
            )

        transaction_path = _transaction_path(directory)
        transaction = _read_transaction(transaction_path)
        if transaction is not None:
            immutable = {
                "operation_id": operation_id,
                "deployment_id": deployment_id,
                "run_id": run_id,
                "checkpoint": {"generation": pointer["generation"],
                               "snapshot_id": pointer["snapshot_id"],
                               "policy_version": pointer["policy_version"],
                               "optimizer_steps": pointer["optimizer_steps"]},
            }
            if any(transaction.get(key) != value for key, value in immutable.items()):
                raise RuntimeError("managed stop retry does not match its validated transaction")
        else:
            sidecars = _sidecar_capture(root)
            superseded_control = _capture_superseded_stop_control(runtime)
            transaction = {
                "schema": _TRANSACTION_SCHEMA,
                "status": "captured",
                **{
                    "operation_id": operation_id,
                    "deployment_id": deployment_id,
                    "run_id": run_id,
                    "checkpoint": {"generation": pointer["generation"],
                                   "snapshot_id": pointer["snapshot_id"],
                                   "policy_version": pointer["policy_version"],
                                   "optimizer_steps": pointer["optimizer_steps"]},
                    "sidecars": sidecars,
                    "superseded_control_capture": superseded_control,
                },
            }
            _write_transaction(transaction_path, transaction)

        if transaction.get("status") == "captured":
            from .instance_recovery import _intent_disposition

            disposition, intent = _intent_disposition(root, pointer)
            if disposition in {"invalid", "neither"}:
                raise RuntimeError("optimizer control sidecars do not match the selected durable checkpoint")
            if transaction.get("sidecars") and not isinstance(transaction.get("sidecars"), Mapping):
                raise RuntimeError("managed stop transaction sidecar capture is invalid")
            engine = _load_recovery_engine(deployment, runtime, optimizer_owner_lock)
            before = engine.core.status()
            core_plan = engine.core.managed_stop_now_discard_plan(pointer)
            superseded_control = _validate_superseded_stop_control(
                transaction.get("superseded_control_capture"),
                proof=stop_receipt,
                core_status=before,
            )
            windows = core_plan["windows"]
            if len(windows) > 1:
                raise RuntimeError("managed stop found multiple speculative Core windows")
            window_dict = None
            committed_receipt = None
            from ..learner.gradient_server.store import SQLiteGradientJobStore

            excluded_windows = {(row["window_id"], row["attempt_id"]) for row in windows}
            if disposition == "target" and isinstance(intent, Mapping):
                excluded_windows.add((intent["window_id"], intent["attempt_id"]))
            staged_inspection = SQLiteGradientJobStore.inspect_staged_windows_for_managed_stop(
                Path(runtime.run) / "gradient-server" / "jobs.sqlite3",
                expected_identity={"lineage_id": before["lineage_id"],
                                   "engine_epoch": before["engine_epoch"]},
                excluded_windows=excluded_windows,
                durable_policy_version=before["policy_version"],
            )
            if staged_inspection["other_pending_windows"]:
                raise RuntimeError("gradient database has an unowned pending optimizer window")
            staged_windows = staged_inspection["staged_windows"]
            if windows:
                window_record = windows[0]
                window = engine.core.prepared_window(window_record["window_id"], window_record["attempt_id"])
                if window is None:
                    raise RuntimeError("managed stop Core window disappeared during validation")
                window_dict = window.to_dict()
            elif before.get("dirty") is True or before.get("checkpoint_state") != "durable":
                raise RuntimeError("ambiguous optimizer state has no discardable Core window")
            if disposition == "target":
                if not isinstance(intent, Mapping):
                    raise RuntimeError("target checkpoint has no validated async intent")
                receipt_value = engine.core.result_receipt(intent["window_id"], intent["attempt_id"])
                committed_receipt = receipt_value.to_dict() if receipt_value is not None else None
                target = intent.get("target")
                if (not isinstance(committed_receipt, Mapping) or not isinstance(target, Mapping)
                        or committed_receipt.get("policy_version") != target.get("policy_version")
                        or committed_receipt.get("checkpoint_generation") != target.get("generation")):
                    raise RuntimeError("target checkpoint lacks its committed Core receipt")

            transaction.update({
                "status": "validated",
                "classification": (
                    "committed_checkpoint_reconciled" if disposition == "target"
                    else "uncommitted_work_discarded" if windows or staged_windows
                    else "clean_checkpoint_no_uncommitted_work"
                ),
                "intent_disposition": disposition,
                "intent": None if intent is None else dict(intent),
                "core_plan": core_plan,
                "window": window_dict,
                "staged_windows": staged_windows,
                "superseded_stop_control": superseded_control,
                "committed_receipt": committed_receipt,
                "core_before": _core_identity(before),
            })
            _write_transaction(transaction_path, transaction)

        if transaction.get("status") == "validated":
            sidecars = transaction.get("sidecars", {})
            if not isinstance(sidecars, Mapping):
                raise RuntimeError("managed stop transaction has invalid sidecar entries")
            moved = _move_control_files(root, directory, sidecars)
            transaction["superseded_stop_quarantine"] = _move_superseded_stop_control(
                directory, transaction.get("superseded_stop_control"),
            )
            staged_windows = transaction.get("staged_windows", [])
            core_before = transaction.get("core_before")
            if not isinstance(core_before, Mapping):
                raise RuntimeError("staged discard transaction omitted its Core identity")
            from ..learner.gradient_server.store import SQLiteGradientJobStore

            staged_result = SQLiteGradientJobStore.discard_staged_windows_for_managed_stop(
                Path(runtime.run) / "gradient-server" / "jobs.sqlite3",
                expected_identity={"lineage_id": core_before["lineage_id"],
                                   "engine_epoch": core_before["engine_epoch"]},
                expected_windows=staged_windows,
                operation_id=operation_id,
            )
            transaction["staged_result"] = staged_result
            window_data = transaction.get("window")
            core_before = transaction.get("core_before")
            if transaction.get("classification") == "uncommitted_work_discarded":
                if not isinstance(window_data, Mapping):
                    raise RuntimeError("validated discard transaction omitted its Core window")
                from qwen_rl.contracts.gradients import PreparedWindow
                from ..learner.gradient_server.store import SQLiteGradientJobStore

                window = PreparedWindow.from_dict(window_data)
                sqlite_result = SQLiteGradientJobStore.discard_window_for_managed_stop(
                    Path(runtime.run) / "gradient-server" / "jobs.sqlite3",
                    window,
                    operation_id=operation_id,
                )
                transaction["sqlite_result"] = sqlite_result
                if engine is not None:
                    engine.core.record_managed_stop_now_discard(
                        {"pointer": dict(pointer), "core_plan": transaction["core_plan"]},
                        operation_id=operation_id,
                    )
            elif transaction.get("classification") == "committed_checkpoint_reconciled":
                intent = transaction.get("intent")
                receipt = transaction.get("committed_receipt")
                if not isinstance(intent, Mapping) or not isinstance(receipt, Mapping):
                    raise RuntimeError("validated committed transaction omitted its Core evidence")
                if not isinstance(core_before, Mapping):
                    raise RuntimeError("validated committed transaction omitted its checkpoint identity")
                from ..learner.gradient_server.store import SQLiteGradientJobStore

                sqlite_result = SQLiteGradientJobStore.reconcile_committed_window_for_managed_stop(
                    Path(runtime.run) / "gradient-server" / "jobs.sqlite3",
                    window_id=intent["window_id"], attempt_id=intent["attempt_id"],
                    expected_identity={"lineage_id": core_before["lineage_id"],
                                       "engine_epoch": core_before["engine_epoch"]},
                    expected_fingerprint=intent["fingerprint"], receipt=receipt,
                )
                transaction["sqlite_result"] = sqlite_result
            transaction["status"] = "quarantined"
            transaction["quarantined_files"] = moved
            _write_transaction(transaction_path, transaction)

        if transaction.get("status") == "quarantined":
            # A process may have stopped between DB fencing and Core cancellation.
            window_data = transaction.get("window")
            staged_windows = transaction.get("staged_windows", [])
            core_before = transaction.get("core_before")
            if not isinstance(core_before, Mapping):
                raise RuntimeError("staged discard transaction omitted its Core identity")
            transaction["superseded_stop_quarantine"] = _move_superseded_stop_control(
                directory, transaction.get("superseded_stop_control"),
            )
            from ..learner.gradient_server.store import SQLiteGradientJobStore

            transaction["staged_result"] = SQLiteGradientJobStore.discard_staged_windows_for_managed_stop(
                Path(runtime.run) / "gradient-server" / "jobs.sqlite3",
                expected_identity={"lineage_id": core_before["lineage_id"],
                                   "engine_epoch": core_before["engine_epoch"]},
                expected_windows=staged_windows,
                operation_id=operation_id,
            )
            if (transaction.get("classification") == "uncommitted_work_discarded"
                    and isinstance(window_data, Mapping)):
                from qwen_rl.contracts.gradients import PreparedWindow
                from ..learner.gradient_server.store import SQLiteGradientJobStore

                window = PreparedWindow.from_dict(window_data)
                sqlite_result = SQLiteGradientJobStore.discard_window_for_managed_stop(
                    Path(runtime.run) / "gradient-server" / "jobs.sqlite3",
                    window,
                    operation_id=operation_id,
                )
                transaction["sqlite_result"] = sqlite_result
            engine = _load_recovery_engine(deployment, runtime, optimizer_owner_lock)
            after = engine.core.status()
            if (transaction.get("classification") == "uncommitted_work_discarded"
                    and after.get("pending_window") is not None):
                core_plan = engine.core.managed_stop_now_discard_plan(pointer)
                engine.core.record_managed_stop_now_discard(
                    {"pointer": dict(pointer), "core_plan": core_plan},
                    operation_id=operation_id,
                )
                after = engine.core.status()
            if (transaction.get("classification") == "committed_checkpoint_reconciled"
                    and (after.get("dirty") is True or after.get("checkpoint_state") != "durable")):
                raise RuntimeError("committed target checkpoint did not reload as durable")
            _close_recovery_engine(engine)
            engine = None

            # Reload after the source sidecars are isolated; this is the final normal-resume Core gate.
            engine = _load_recovery_engine(deployment, runtime, optimizer_owner_lock)
            final_status = engine.core.status()
            original = transaction.get("core_before")
            if not isinstance(original, Mapping):
                original = _core_identity(final_status)
            _assert_core_ready(original, final_status, pointer)
            bepa_after = _bepa_ledger_snapshot(runtime)
            if bepa_after != bepa_before:
                raise RuntimeError("BEPA used/unknown budget or reservation ledger changed during stop-now")
            final_windows = [row for row in final_status.get("cancelled_windows", [])
                             if isinstance(row, Mapping)
                             and row.get("reason") == f"managed_stop_now:{operation_id}"]
            moved = transaction.get("quarantined_files", [])
            receipt = {
                "schema": _RECEIPT_SCHEMA,
                "status": "passed",
                "operation_id": operation_id,
                "deployment_id": deployment_id,
                "run_id": run_id,
                "classification": transaction.get("classification"),
                "checkpoint": {
                    "generation": pointer["generation"],
                    "snapshot_id": pointer["snapshot_id"],
                    "policy_version": pointer["policy_version"],
                    "optimizer_steps": pointer["optimizer_steps"],
                    "snapshot": pointer["snapshot"],
                    "size_bytes": pointer["size_bytes"],
                },
                "preserved_core_state": _core_identity(final_status),
                "quarantined_files": moved,
                "quarantined_window": transaction.get("window"),
                "staged_windows_discarded": transaction.get("staged_windows", []),
                "staged_discard": transaction.get("staged_result", {
                    "cancelled_windows": [], "payload_disposition": "metadata_only",
                }),
                "superseded_stop_after_checkpoint": transaction.get("superseded_stop_quarantine"),
                "sqlite": transaction.get("sqlite_result"),
                "core_cancellations": final_windows,
                "bepa_ledger": {"before": bepa_before, "after": bepa_after,
                                "unchanged": True},
                "checkpoint_payload_hash_scan": False,
                "core_resume_ready": True,
                "validation": "actual_cpu_central_core_loaded_twice_under_optimizer_lock",
                "managed_processes_stopped": True,
                "verified_process_count": proof_state["verified_process_count"],
                "nested_manifest_missing": proof_state["nested_manifest_missing"],
                "source_sidecar_recovery": "identity_checked_and_quarantined",
            }
            _atomic_json(receipt_path, receipt)
            transaction["status"] = "complete"
            transaction["receipt"] = str(receipt_path)
            _write_transaction(transaction_path, transaction)
            return receipt

        raise RuntimeError("managed stop transaction has an unsupported state")
    finally:
        _close_recovery_engine(engine)
        try:
            if optimizer_owner_lock is not None:
                optimizer_owner_lock.close()
        finally:
            os.close(owner_descriptor)
