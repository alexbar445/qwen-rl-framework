"""Offline, single-writer migration bundles for managed RL instances.

Checkpoint payloads are copied as opaque files. This module never loads a
checkpoint pickle, hashes model weights, or starts a runtime role.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import shlex
import sqlite3
import stat
import uuid
from typing import Any, Callable, Iterator, Mapping

from .instance_plan import InstanceDeployment

_SCHEMA = "qwen-rl-instance-migration/v1"
_FENCE_SCHEMA = "qwen-rl-instance-migration-fence/v1"
_CLAIM_SCHEMA = "qwen-rl-instance-migration-claim/v1"
_SNAPSHOT_NAME = re.compile(r"state-(\d{8})-([0-9a-f]{32})\.pt\Z")
_FENCE_NAME = "migration-fence.json"
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_FILES = 200_000


class InstanceMigrationError(ValueError):
    """A managed instance migration could not be prepared safely."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _fail(message: str) -> None:
    raise InstanceMigrationError(message)


def _notify(progress: Callable[[dict[str, Any]], None] | None, stage: str, **details: Any) -> None:
    if progress is not None:
        progress({"stage": stage, **details})


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        _fail("migration directories must be private, regular, and user-owned")


def _safe_path(path: str | Path, *, must_exist: bool = False) -> Path:
    value = Path(path).expanduser()
    try:
        resolved = value.resolve(strict=must_exist)
    except OSError:
        _fail("migration path is unavailable")
    if value.exists() and value.is_symlink():
        _fail("migration roots cannot be symlinks")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _safe_relative(value: Any) -> str:
    if (not isinstance(value, str) or not value or len(value) > 4096
            or "\\" in value
            or any(ord(char) < 32 or ord(char) in {127, 0x85, 0x2028, 0x2029} for char in value)):
        _fail("migration bundle contains an invalid relative path")
    path = PurePosixPath(value)
    pieces = value.split("/")
    if path.is_absolute() or any(piece in {"", ".", ".."} for piece in pieces):
        _fail("migration bundle contains an unsafe relative path")
    return path.as_posix()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _private_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_json(path: Path, *, limit: int = _MAX_MANIFEST_BYTES, private: bool = False) -> dict[str, Any]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_size > limit:
            _fail("migration metadata is not a supported regular file")
        if private and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077 or info.st_nlink != 1):
            _fail("migration receipt must be private and user-owned")
        value = json.loads(path.read_text(encoding="utf-8"))
    except InstanceMigrationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        _fail("migration metadata cannot be read")
    if not isinstance(value, dict):
        _fail("migration metadata must be an object")
    return value


def _deployment_id(deployment: InstanceDeployment) -> str:
    return hashlib.sha256(str(deployment.path.resolve()).encode("utf-8")).hexdigest()[:20]


def _fence_path(deployment: InstanceDeployment) -> Path:
    return deployment.state_dir / _FENCE_NAME


def assert_not_migrating(deployment: InstanceDeployment) -> None:
    """Fail closed when a source deployment has an unresolved migration fence."""
    path = _fence_path(deployment)
    if not path.exists() and not path.is_symlink():
        return
    fence = _read_json(path, private=True)
    if (fence.get("schema") != _FENCE_SCHEMA
            or not isinstance(fence.get("migration_id"), str)
            or fence.get("state") not in {"exporting", "exported", "claimed"}):
        _fail("instance migration fence is malformed; refusing to resume this source deployment")
    _fail("source instance is fenced by an unfinished migration; explicitly cancel it before resume")


def _create_fence(deployment: InstanceDeployment, migration_id: str, *, state: str) -> Path:
    path = _fence_path(deployment)
    _private_dir(path.parent)
    data = (json.dumps({
        "schema": _FENCE_SCHEMA,
        "migration_id": migration_id,
        "source_deployment_id": _deployment_id(deployment),
        "state": state,
        "created_at": _now(),
    }, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _read_json(path, private=True)
        if existing.get("migration_id") == migration_id:
            return path
        _fail("source deployment already has a migration fence")
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_dir(path.parent)
    return path


def _update_fence(path: Path, *, migration_id: str, state: str, manifest_sha256: str | None = None) -> None:
    current = _read_json(path, private=True)
    if current.get("schema") != _FENCE_SCHEMA or current.get("migration_id") != migration_id:
        _fail("source migration fence no longer matches this bundle")
    updated = dict(current)
    updated.update(state=state, updated_at=_now())
    if manifest_sha256 is not None:
        updated["manifest_sha256"] = manifest_sha256
    _atomic_json(path, updated)


@contextmanager
def _owner_lock(deployment: InstanceDeployment) -> Iterator[None]:
    """Serialize a fence write with a stopped deployment's next serve call."""
    _private_dir(deployment.state_dir)
    lock_path = deployment.state_dir / "owner.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            _fail("instance owner lock is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("instance is active or starting; stop it through its supervisor before export")
        yield
    finally:
        os.close(descriptor)


@contextmanager
def _migration_operation_lock(deployment: InstanceDeployment) -> Iterator[None]:
    """Serialize export/cancel for one source without relying on a running owner lock."""
    _private_dir(deployment.state_dir)
    path = deployment.state_dir / "migration-operation.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            _fail("migration operation lock is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("another migration operation owns this source deployment")
        yield
    finally:
        os.close(descriptor)


@contextmanager
def _checkpoint_owner_lock(recovery_root: Path) -> Iterator[None]:
    """Hold the existing optimizer ownership lock without writing its PID payload."""
    from ..learner.gradient_server.ownership import _lock_path_for_checkpoint_dir

    lock_path = _lock_path_for_checkpoint_dir(recovery_root)
    try:
        info = lock_path.lstat()
    except OSError:
        _fail("full-checkpoint optimizer lock is unavailable; refusing a potentially live snapshot")
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid():
        _fail("full-checkpoint optimizer lock is unsafe")
    descriptor = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_dev != info.st_dev
                or opened.st_ino != info.st_ino):
            _fail("full-checkpoint optimizer lock changed during validation")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("full-checkpoint optimizer is still active; wait for its managed stop")
        yield
        try:
            current = lock_path.lstat()
            opened = os.fstat(descriptor)
        except OSError:
            _fail("full-checkpoint optimizer lock disappeared during migration capture")
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            _fail("full-checkpoint optimizer lock changed during migration capture")
    finally:
        os.close(descriptor)


def _load_runtime(deployment: InstanceDeployment) -> Any:
    from ..runtime.config import load_config

    try:
        return load_config(deployment.runtime_config, for_local_worker_plan=True)
    except Exception as exc:
        raise InstanceMigrationError(f"migration runtime configuration is invalid ({type(exc).__name__})") from exc


def _resolved_config_path(value: str | Path | None, *, label: str, must_exist: bool) -> Path | None:
    if value is None or not str(value).strip():
        return None
    try:
        from ..runtime.config import _resolve_path
        path = _resolve_path(str(value)).resolve(strict=must_exist)
    except OSError:
        _fail(f"required migration input is unavailable: {label}")
    return path


def _ninfer_artifact_identity(identity: Any, *, label: str) -> tuple[str, str]:
    source = identity.get("source") if isinstance(identity, dict) else None
    artifact = source.get("ninfer_artifact") if isinstance(source, dict) else None
    artifact_id = artifact.get("artifact_id") if isinstance(artifact, dict) else None
    artifact_sha256 = artifact.get("sha256") if isinstance(artifact, dict) else None
    if (not isinstance(artifact_id, str) or re.fullmatch(r"[0-9a-f]{32}", artifact_id) is None
            or not isinstance(artifact_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None):
        _fail(f"{label} lacks its pinned NInfer artifact ID and SHA-256 identity")
    return artifact_id, artifact_sha256


def _require_matching_ninfer_artifact(source: Mapping[str, Any], target_runtime: Any) -> None:
    source_id = source.get("artifact_id")
    source_sha256 = source.get("artifact_sha256")
    if (not isinstance(source_id, str) or re.fullmatch(r"[0-9a-f]{32}", source_id) is None
            or not isinstance(source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None):
        _fail("exported source has an invalid NInfer artifact ID or SHA-256 identity")
    target_id, target_sha256 = _ninfer_artifact_identity(
        getattr(target_runtime, "artifact_identity", None), label="target runtime",
    )
    if target_id != source_id:
        _fail("target model artifact identity differs from the exported source; base weights are never copied")
    if target_sha256 != source_sha256:
        _fail("target model artifact metadata differs from the exported source")


def _credential_excludes(layout: Mapping[str, Any]) -> tuple[Path, ...]:
    configured = layout.get("credential_paths", {})
    if not isinstance(configured, Mapping):
        _fail("migration credential path list is malformed")
    result = []
    for name, value in configured.items():
        if not isinstance(name, str) or not isinstance(value, str):
            _fail("migration credential path list is malformed")
        raw_path = Path(value).expanduser()
        if not raw_path.is_absolute():
            _fail("configured credential path must be absolute")
        try:
            path = raw_path.resolve(strict=False)
        except (OSError, RuntimeError):
            _fail("configured credential path cannot be resolved")
        if not path.is_absolute():
            _fail("configured credential path must be absolute")
        result.append(path)
    return tuple(result)


def _source_layout(deployment: InstanceDeployment, runtime: Any) -> dict[str, Any]:
    run_root = _resolved_config_path(runtime.run_dir, label="run_dir", must_exist=True)
    recovery_root = _resolved_config_path(runtime.learner.recovery_dir, label="recovery_dir", must_exist=True)
    state_root = _resolved_config_path(deployment.state_dir, label="state_dir", must_exist=False)
    model_root = _resolved_config_path(runtime.model_path, label="model_path", must_exist=False)
    dataset_root = _resolved_config_path(runtime.dataset_dir, label="dataset_dir", must_exist=True)
    identity = _resolved_config_path(runtime.artifact_identity_path, label="artifact identity", must_exist=True)
    report = _resolved_config_path(runtime.worker_inference.ninfer_conversion_report,
                                   label="NInfer conversion report", must_exist=True)
    adapter = _resolved_config_path(runtime.learner.initial_adapter_path,
                                    label="initial adapter", must_exist=True)
    if None in (run_root, recovery_root, state_root, model_root, dataset_root, identity, report, adapter):
        _fail("migration requires explicit run, checkpoint, model, dataset, identity, report, and adapter paths")
    assert run_root and recovery_root and state_root and model_root and dataset_root and identity and report and adapter
    artifact_id, artifact_sha256 = _ninfer_artifact_identity(
        runtime.artifact_identity, label="source runtime",
    )
    if not run_root.is_dir() or not recovery_root.is_dir() or not dataset_root.is_dir():
        _fail("run, checkpoint, and dataset roots must be directories")
    if model_root.exists() and _overlap(model_root, run_root):
        if _is_within(run_root, model_root):
            _fail("run_dir cannot be nested under the model base; refusing to scan or copy model weights")
    if _overlap(dataset_root, recovery_root) or _overlap(dataset_root, run_root):
        _fail("dataset_dir cannot overlap durable run or checkpoint state")
    if _is_within(run_root, recovery_root):
        _fail("run_dir cannot be nested under the full-checkpoint root")
    if state_root == run_root:
        _fail("deployment state_dir cannot be identical to run_dir")
    if _is_within(run_root, state_root):
        _fail("run_dir cannot be nested under deployment state_dir")

    raw_credentials = deployment.credential_files
    if not isinstance(raw_credentials, Mapping):
        _fail("deployment credential_files must map names to paths")
    credential_paths: dict[str, str] = {}
    for name, value in raw_credentials.items():
        if not isinstance(name, str):
            _fail("deployment credential_files has an invalid name")
        credential = _resolved_config_path(value, label=f"credential file {name}", must_exist=False)
        if credential is None:
            _fail("deployment credential_files has an empty path")
        credential_paths[name] = str(credential)

    optional_paths: dict[str, str | None] = {}
    arc = getattr(runtime, "arc", None)
    contract = _resolved_config_path(getattr(arc, "contract_path", None),
                                     label="ARC contract", must_exist=True)
    optional_paths["arc_contract_path"] = str(contract) if contract is not None else None
    tokenizer = _resolved_config_path(getattr(runtime, "tokenizer_path", None),
                                      label="tokenizer", must_exist=True)
    # Tokenizer files inside the pinned model directory follow the target
    # model path and are never copied with the base artifact.
    optional_paths["tokenizer_path"] = (
        str(tokenizer) if tokenizer is not None and not _is_within(tokenizer, model_root) else None
    )
    evaluation = getattr(runtime, "evaluation", None)
    for attribute, key, label in (
        ("manifest_path", "evaluation_manifest_path", "evaluation manifest"),
        ("sft_exclusion_proof", "evaluation_sft_exclusion_proof", "evaluation exclusion proof"),
        ("initial_count_proof", "evaluation_initial_count_proof", "evaluation count proof"),
    ):
        path = _resolved_config_path(getattr(evaluation, attribute, None), label=label, must_exist=True)
        optional_paths[key] = str(path) if path is not None else None
    evaluation_state = _resolved_config_path(getattr(evaluation, "state_dir", None),
                                              label="evaluation state", must_exist=False)
    if evaluation_state is not None and evaluation_state.exists() and not evaluation_state.is_dir():
        _fail("evaluation state path must be a directory")
    if evaluation_state is not None:
        if _overlap(evaluation_state, recovery_root):
            _fail("evaluation state cannot overlap full-checkpoint state")
        if _overlap(evaluation_state, run_root) and not _is_within(evaluation_state, run_root):
            _fail("evaluation state cannot contain or overlap the durable run root")
    optional_paths["evaluation_state_dir"] = str(evaluation_state) if evaluation_state is not None else None

    bepa_enabled = bool(runtime.bepa.enabled)
    bepa_root: Path | None = None
    if bepa_enabled:
        raw_bepa = runtime.bepa.state_dir or str(run_root / "bepa")
        bepa_root = _resolved_config_path(raw_bepa, label="BEPA state", must_exist=True)
        if bepa_root is None or not bepa_root.is_dir():
            _fail("enabled BEPA state must be an existing directory")
        db_path = bepa_root / "state.sqlite3"
        if not db_path.is_file() or db_path.is_symlink():
            _fail("enabled BEPA state is missing its durable state.sqlite3 database")
        if _overlap(bepa_root, recovery_root):
            _fail("BEPA state cannot be nested under the full-checkpoint root")
        if _overlap(bepa_root, run_root) and not _is_within(bepa_root, run_root):
            _fail("BEPA state cannot contain or overlap the durable run root")

    return {
        "deployment_id": _deployment_id(deployment),
        "deployment_path": str(deployment.path.resolve()),
        "state_dir": str(state_root),
        "credential_paths": credential_paths,
        "runtime_config": str(deployment.runtime_config.resolve()),
        "run_dir": str(run_root),
        "recovery_dir": str(recovery_root),
        "model_path": str(model_root),
        "artifact_sha256": artifact_sha256,
        "dataset_dir": str(dataset_root),
        "artifact_identity_path": str(identity),
        "artifact_id": artifact_id,
        "conversion_report_path": str(report),
        "initial_adapter_path": str(adapter),
        "bepa_enabled": bepa_enabled,
        "bepa_state_dir": str(bepa_root) if bepa_root is not None else None,
        **optional_paths,
    }


def _source_entries(root: Path, *, exclude_roots: tuple[Path, ...] = ()) -> list[tuple[Path, str, os.stat_result]]:
    """List regular files without following links or descending into exclusions."""
    root = _safe_path(root, must_exist=True)
    try:
        root_info = root.lstat()
    except OSError:
        _fail("migration input root is unavailable")
    if not stat.S_ISDIR(root_info.st_mode):
        _fail("migration directory input is not a regular directory")
    result: list[tuple[Path, str, os.stat_result]] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            _fail("migration input directory cannot be read")
        for child in children:
            if any(_is_within(child, excluded) for excluded in exclude_roots):
                continue
            try:
                info = child.lstat()
            except OSError:
                _fail("migration input changed while it was being enumerated")
            if stat.S_ISLNK(info.st_mode):
                _fail("migration inputs cannot contain symlinks")
            if stat.S_ISDIR(info.st_mode):
                stack.append(child)
            elif stat.S_ISREG(info.st_mode):
                relative = child.relative_to(root).as_posix()
                result.append((child, _safe_relative(relative), info))
            else:
                _fail("migration inputs cannot contain special files")
            if len(result) > _MAX_FILES:
                _fail("migration input exceeds the supported file count")
    return result


def _copy_file(
    source: Path,
    target: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    stage: str = "copy",
) -> tuple[os.stat_result, str]:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists() or target.is_symlink():
        _fail("migration staging path already exists")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(source_fd, "rb") as source_stream:
            before = os.fstat(source_stream.fileno())
            if not stat.S_ISREG(before.st_mode) or (expected_size is not None and before.st_size != expected_size):
                _fail("migration source file is not the pinned regular file")
            target_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(target_fd, "wb") as target_stream:
                digest = hashlib.sha256()
                copied_bytes = 0
                next_progress = 64 * 1024 * 1024
                while chunk := source_stream.read(1024 * 1024):
                    target_stream.write(chunk)
                    digest.update(chunk)
                    copied_bytes += len(chunk)
                    if copied_bytes >= next_progress:
                        _notify(progress, stage, current_file_bytes=copied_bytes,
                                current_file_total_bytes=before.st_size)
                        next_progress = copied_bytes + 64 * 1024 * 1024
                target_stream.flush()
                os.fsync(target_stream.fileno())
                _notify(progress, stage, current_file_bytes=copied_bytes,
                        current_file_total_bytes=before.st_size, file_complete=True)
            sha256 = digest.hexdigest()
            if expected_sha256 is not None and sha256 != expected_sha256:
                _fail("migration file content changed after import; SHA-256 differs from its manifest")
            after = os.fstat(source_stream.fileno())
            if (after.st_dev != before.st_dev or after.st_ino != before.st_ino
                    or after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns):
                _fail("migration source changed while it was copied")
        os.replace(temporary, target)
        _fsync_dir(target.parent)
        return before, sha256
    except InstanceMigrationError:
        temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise InstanceMigrationError("migration file could not be copied atomically") from exc


def _sqlite_identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return None
        with path.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                return None
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
    except OSError:
        return None


def _sqlite_sidecar_identity(path: Path) -> dict[Path, tuple[int, int, int, int] | None]:
    result = {}
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(path) + suffix)
        try:
            info = sidecar.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                _fail("SQLite sidecar is unsafe")
            result[sidecar] = ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                               if suffix != "-shm" else (info.st_dev, info.st_ino, info.st_size, 0))
        except FileNotFoundError:
            result[sidecar] = None
    return result


def _copy_sqlite_snapshot(
    source: Path,
    target: Path,
    *,
    progress: Callable[[dict[str, Any]], None] | None,
    stage: str,
) -> tuple[os.stat_result, str]:
    """Take a consistent read-only SQLite backup, including committed WAL rows."""
    before = _sqlite_identity(source)
    if before is None:
        _fail("SQLite migration source is invalid")
    sidecars_before = _sqlite_sidecar_identity(source)
    journal_identity = sidecars_before.get(Path(str(source) + "-journal"))
    if journal_identity is not None and journal_identity[2] > 0:
        _fail("SQLite source has a nonempty rollback journal; refusing an ambiguous snapshot")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists() or target.is_symlink():
        _fail("migration staging path already exists")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    scratch_dir = target.parent / f".{target.name}.{uuid.uuid4().hex}.sqlite-source"
    source_connection: sqlite3.Connection | None = None
    target_connection: sqlite3.Connection | None = None
    try:
        scratch_dir.mkdir(mode=0o700)
        scratch_database = scratch_dir / source.name
        _copy_file(source, scratch_database, expected_size=before[2],
                   progress=progress, stage=f"{stage}_source_snapshot")
        wal_path = Path(str(source) + "-wal")
        wal_identity = sidecars_before.get(wal_path)
        if wal_identity is not None:
            _copy_file(wal_path, Path(str(scratch_database) + "-wal"), expected_size=wal_identity[2],
                       progress=progress, stage=f"{stage}_wal_snapshot")
        source_after_copy = _sqlite_identity(source)
        sidecars_after_copy = _sqlite_sidecar_identity(source)
        if source_after_copy != before or sidecars_after_copy != sidecars_before:
            _fail("SQLite source changed while its consistent input snapshot was copied")
        # Open only the private scratch copy. SQLite may create/update its SHM
        # lock index there; the source DB and sidecars remain untouched.
        source_connection = sqlite3.connect(scratch_database.as_uri() + "?mode=ro", uri=True, timeout=5.0)
        target_connection = sqlite3.connect(temporary)
        def report(_status: int, remaining: int, total: int) -> None:
            _notify(progress, stage, pages_remaining=remaining, pages_total=total)
        source_connection.backup(target_connection, pages=256, progress=report, sleep=0.01)
        target_connection.commit()
        target_connection.close()
        target_connection = None
        source_connection.close()
        source_connection = None
        source_after = _sqlite_identity(source)
        sidecars_after = _sqlite_sidecar_identity(source)
        if source_after != before or sidecars_after != sidecars_before:
            _fail("SQLite source changed while its consistent backup was captured")
        verification = sqlite3.connect(temporary.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=5.0)
        try:
            check = verification.execute("PRAGMA quick_check").fetchone()
            if not check or check[0] != "ok":
                _fail("SQLite backup failed its read-only integrity check")
        finally:
            verification.close()
        descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            copied_info = os.fstat(descriptor)
            if not stat.S_ISREG(copied_info.st_mode):
                _fail("SQLite backup output is not a regular file")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        digest = hashlib.sha256()
        with temporary.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        os.replace(temporary, target)
        _fsync_dir(target.parent)
        return target.stat(), digest.hexdigest()
    except InstanceMigrationError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, sqlite3.Error) as exc:
        temporary.unlink(missing_ok=True)
        raise InstanceMigrationError(f"SQLite state could not be backed up consistently ({type(exc).__name__})") from exc
    finally:
        if target_connection is not None:
            target_connection.close()
        if source_connection is not None:
            source_connection.close()
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _copy_input(
    source: Path,
    bundle_root: Path,
    relative_root: str,
    *,
    kind: str,
    exclude_roots: tuple[Path, ...] = (),
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    source = _safe_path(source, must_exist=True)
    if any(_is_within(source, excluded) for excluded in exclude_roots):
        _fail("required migration input overlaps a configured credential path")
    destination_root = bundle_root / _safe_relative(relative_root)
    try:
        root_info = source.lstat()
    except OSError:
        _fail("migration input is unavailable")
    entries: list[dict[str, Any]] = []
    if stat.S_ISREG(root_info.st_mode):
        relative = _safe_relative(relative_root)
        copied, digest = _copy_file(source, bundle_root / relative, expected_size=root_info.st_size,
                                    progress=progress, stage=f"prepare_{kind}")
        entries.append({"path": relative, "size": copied.st_size, "sha256": digest, "kind": kind,
                        "source_path": str(source), "source_device": copied.st_dev,
                        "source_inode": copied.st_ino, "source_mtime_ns": copied.st_mtime_ns})
    elif stat.S_ISDIR(root_info.st_mode):
        destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for index, (file_path, suffix, info) in enumerate(
            _source_entries(source, exclude_roots=exclude_roots), start=1,
        ):
            relative = _safe_relative(f"{relative_root}/{suffix}")
            copied, digest = _copy_file(file_path, bundle_root / relative, expected_size=info.st_size,
                                        progress=progress, stage=f"prepare_{kind}")
            entries.append({"path": relative, "size": copied.st_size, "sha256": digest, "kind": kind,
                            "source_path": str(file_path), "source_device": copied.st_dev,
                            "source_inode": copied.st_ino, "source_mtime_ns": copied.st_mtime_ns})
            _notify(progress, f"prepare_{kind}", files_done=index)
        if not entries:
            _fail(f"migration input is empty: {kind}")
    else:
        _fail("migration input must be a regular file or directory")
    return entries


def _manifest_path(bundle_dir: Path) -> Path:
    return bundle_dir / "manifest.json"


def _read_manifest(bundle_dir: Path, *, require_exported: bool = False) -> dict[str, Any]:
    root = _safe_path(bundle_dir, must_exist=True)
    if not root.is_dir():
        _fail("migration bundle must be a directory")
    manifest = _read_json(_manifest_path(root))
    if (manifest.get("schema") != _SCHEMA or not isinstance(manifest.get("migration_id"), str)
            or manifest.get("status") not in {"prepared", "exported"}):
        _fail("migration bundle manifest schema or status is unsupported")
    if require_exported and manifest.get("status") != "exported":
        _fail("migration bundle has not captured stopped durable state")
    source = manifest.get("source")
    if not isinstance(source, dict):
        _fail("migration bundle source identity is missing")
    source_id = source.get("artifact_id")
    source_sha256 = source.get("artifact_sha256")
    if (not isinstance(source_id, str) or re.fullmatch(r"[0-9a-f]{32}", source_id) is None
            or not isinstance(source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None):
        _fail("migration bundle source NInfer artifact ID or SHA-256 is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) > _MAX_FILES:
        _fail("migration bundle file list is invalid")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            _fail("migration bundle file entry is invalid")
        relative = _safe_relative(entry.get("path"))
        digest = entry.get("sha256")
        if (relative in seen or type(entry.get("size")) is not int or entry["size"] < 0
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            _fail("migration bundle file list contains a duplicate or invalid size")
        seen.add(relative)
    return manifest


def _verify_bundle_files(bundle: Path, manifest: Mapping[str, Any]) -> None:
    for entry in manifest["files"]:
        _safe_bundle_file(bundle, entry["path"], entry["size"], entry["sha256"])


def _manifest_sha256(bundle: Path) -> str:
    return hashlib.sha256(_manifest_path(bundle).read_bytes()).hexdigest()


def _file_sha256(path: Path, *, max_size: int | None = None) -> tuple[int, str]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            _fail("migration identity input must be a regular non-symlink file")
        if max_size is not None and info.st_size > max_size:
            _fail("migration identity input exceeds its metadata size limit")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = path.lstat()
        if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)):
            _fail("migration identity input changed while it was read")
        return info.st_size, digest.hexdigest()
    except InstanceMigrationError:
        raise
    except OSError:
        _fail("migration identity input is unavailable")


def _verify_installed_file(root: Path, relative: str, *, size: int, sha256: str) -> None:
    relative = _safe_relative(relative)
    current = root
    parts = relative.split("/")
    for part in parts[:-1]:
        current /= part
        try:
            info = current.lstat()
        except OSError:
            _fail("claimed migration state is missing a directory")
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            _fail("claimed migration state contains an unsafe directory")
    path = current / parts[-1]
    actual_size, actual_sha = _file_sha256(path)
    if actual_size != size or actual_sha != sha256:
        _fail("claimed migration state file changed after import; SHA-256 differs")


def _verify_claimed_import(
    bundle: Path,
    manifest: Mapping[str, Any],
    installed_root: Path,
    claim: Mapping[str, Any],
    *,
    destination_id: str,
    destination_identity: Mapping[str, Any],
    requested_path_mappings: Mapping[str, str],
) -> None:
    manifest_sha = _manifest_sha256(bundle)
    claim_status = claim.get("status")
    if (claim.get("schema") != _CLAIM_SCHEMA or claim_status not in {"imported", "claimed"}
            or claim.get("migration_id") != manifest.get("migration_id")
            or claim.get("destination_id") != destination_id
            or claim.get("manifest_sha256") != manifest_sha
            or claim.get("requested_path_mappings") != dict(requested_path_mappings)):
        _fail("claimed migration receipt does not match the current bundle and target configuration")
    if claim_status == "claimed" and claim.get("destination") != destination_identity:
        _fail("claimed destination configuration identity changed after migration import")
    _verify_bundle_files(bundle, manifest)
    operational = claim.get("operational_file_receipts", {})
    if not isinstance(operational, dict):
        _fail("claimed migration receipt has invalid operational file identities")
    for entry in manifest["files"]:
        receipt = operational.get(entry["path"], {})
        if not isinstance(receipt, dict):
            _fail("claimed migration receipt has invalid installed file metadata")
        expected_size = receipt.get("size", entry["size"])
        expected_sha = receipt.get("sha256", entry["sha256"])
        _verify_installed_file(installed_root, entry["path"], size=expected_size, sha256=expected_sha)
    configuration = claim.get("configuration", {})
    if configuration:
        if not isinstance(configuration, dict):
            _fail("claimed migration receipt has invalid generated configuration metadata")
        for key, digest_key in (("runtime_config", "runtime_config_sha256"),
                                ("deployment", "deployment_config_sha256")):
            value, expected = configuration.get(key), configuration.get(digest_key)
            if not isinstance(value, str) or not isinstance(expected, str):
                _fail("claimed migration receipt is missing generated configuration identities")
            _size, actual = _file_sha256(Path(value), max_size=2 * 1024 * 1024)
            if actual != expected:
                _fail("generated destination configuration changed after migration import")
        try:
            deployment_data = json.loads(Path(configuration["deployment"]).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            _fail("generated destination deployment is unavailable")
        if deployment_data.get("runtime_config") != configuration["runtime_config"]:
            _fail("generated destination deployment no longer references its recorded runtime config")


def _source_migration_receipt(
    source: Mapping[str, Any], bundle: Path, manifest: Mapping[str, Any], checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": _FENCE_SCHEMA,
        "migration_id": manifest["migration_id"],
        "status": "exported",
        "source_deployment_id": source["deployment_id"],
        "source_state_dir": source["state_dir"],
        "source_run_dir": source["run_dir"],
        "source_recovery_dir": source["recovery_dir"],
        "source_bepa_state_dir": source["bepa_state_dir"],
        "checkpoint": dict(checkpoint),
        "manifest_sha256": _manifest_sha256(bundle),
        "bundle_dir": str(bundle),
        "exported_at": manifest["exported_at"],
    }


def _safe_bundle_file(bundle_root: Path, relative: str, expected_size: int,
                      expected_sha256: str | None = None) -> Path:
    relative = _safe_relative(relative)
    current = bundle_root
    parts = relative.split("/")
    for part in parts[:-1]:
        current = current / part
        try:
            info = current.lstat()
        except OSError:
            _fail("migration bundle is missing a listed parent directory")
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            _fail("migration bundle path contains a symlink or non-directory")
    path = current / parts[-1]
    try:
        info = path.lstat()
    except OSError:
        _fail("migration bundle is missing a listed file")
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_size != expected_size:
        _fail("migration bundle file is not the listed regular file")
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError:
            _fail("migration bundle file could not be verified")
        if digest.hexdigest() != expected_sha256:
            _fail("migration bundle file SHA-256 differs from its manifest")
    return path


def _validate_checkpoint(root: Path) -> dict[str, Any]:
    latest = root / "latest.json"
    pointer = _read_json(latest, limit=1024 * 1024)
    required = {"schema", "generation", "snapshot_id", "snapshot", "size_bytes", "policy_version", "optimizer_steps"}
    if set(pointer) != required or pointer.get("schema") != "central-gradient-checkpoint-pointer/v1":
        _fail("latest full-checkpoint pointer schema is unsupported")
    snapshot_name = pointer.get("snapshot")
    snapshot_id = pointer.get("snapshot_id")
    match = _SNAPSHOT_NAME.fullmatch(snapshot_name) if isinstance(snapshot_name, str) else None
    if (match is None or not isinstance(snapshot_id, str) or len(snapshot_id) != 32
            or any(char not in "0123456789abcdef" for char in snapshot_id)):
        _fail("latest full-checkpoint pointer has an unsafe snapshot identity")
    for field in ("generation", "size_bytes", "policy_version", "optimizer_steps"):
        if type(pointer.get(field)) is not int or pointer[field] < 0:
            _fail("latest full-checkpoint pointer has an invalid counter")
    if (int(match.group(1)) != pointer["generation"] or match.group(2) != snapshot_id):
        _fail("latest full-checkpoint filename does not match its generation and snapshot ID")
    snapshot = root / snapshot_name
    try:
        info = snapshot.lstat()
    except OSError:
        _fail("latest full-checkpoint payload is missing")
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_size != pointer["size_bytes"]:
        _fail("latest full-checkpoint payload size or type does not match its pointer")
    return {"snapshot_id": snapshot_id, "policy_version": pointer["policy_version"],
            "optimizer_steps": pointer["optimizer_steps"], "snapshot_size": info.st_size,
            "pointer_sha256": hashlib.sha256(latest.read_bytes()).hexdigest()}


def _copy_checkpoint_snapshot(
    source_root: Path,
    staging_root: Path,
    relative_root: str,
    *,
    progress: Callable[[dict[str, Any]], None] | None,
    exclude_roots: tuple[Path, ...] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with _checkpoint_owner_lock(source_root):
        before = _validate_checkpoint(source_root)
        pointer = _read_json(source_root / "latest.json", limit=1024 * 1024)
        if hashlib.sha256((source_root / "latest.json").read_bytes()).hexdigest() != before["pointer_sha256"]:
            _fail("latest full-checkpoint pointer changed during migration capture")
        selected: set[str] = {"latest.json", pointer["snapshot"]}
        generations = {pointer["generation"]}

        intent_path = source_root / "async-checkpoint-intent.json"
        if intent_path.exists() or intent_path.is_symlink():
            intent = _read_json(intent_path, limit=1024 * 1024)
            fields = {"schema", "operation_id", "source", "target", "window_id", "attempt_id", "fingerprint"}
            snapshot_fields = {"generation", "snapshot_id", "policy_version", "optimizer_steps", "scheduler_steps"}
            if (
                set(intent) != fields
                or intent.get("schema") != "central-gradient-async-checkpoint-intent/v1"
                or not isinstance(intent.get("source"), dict)
                or not isinstance(intent.get("target"), dict)
                or set(intent["source"]) != snapshot_fields
                or set(intent["target"]) != snapshot_fields
                or not isinstance(intent.get("operation_id"), str)
                or re.fullmatch(r"[0-9a-f]{32}", intent["operation_id"]) is None
                or not isinstance(intent.get("window_id"), str) or not intent["window_id"]
                or not isinstance(intent.get("attempt_id"), str) or not intent["attempt_id"]
                or not isinstance(intent.get("fingerprint"), str)
                or re.fullmatch(r"[0-9a-f]{64}", intent["fingerprint"]) is None
            ):
                _fail("async checkpoint intent is ambiguous; refusing to export checkpoint history")
            snapshots = (intent["source"], intent["target"])
            for snapshot in (intent["source"], intent["target"]):
                generation = snapshot.get("generation")
                snapshot_id = snapshot.get("snapshot_id")
                if (
                    type(generation) is not int or generation < 0 or generation > 99_999_999
                    or not isinstance(snapshot_id, str)
                    or re.fullmatch(r"[0-9a-f]{32}", snapshot_id) is None
                    or any(type(snapshot.get(field)) is not int or snapshot[field] < 0
                           for field in ("policy_version", "optimizer_steps", "scheduler_steps"))
                    or snapshot["scheduler_steps"] != snapshot["optimizer_steps"]
                ):
                    _fail("async checkpoint intent names an ambiguous snapshot")
                name = f"state-{generation:08d}-{snapshot_id}.pt"
                if not _SNAPSHOT_NAME.fullmatch(name):
                    _fail("async checkpoint intent names an unsupported snapshot")
                selected.add(name)
                generations.add(generation)
            source, target = snapshots
            if (
                target["generation"] != source["generation"] + 1
                or target["policy_version"] != source["policy_version"] + 1
                or target["optimizer_steps"] != source["optimizer_steps"] + 1
                or target["scheduler_steps"] != source["scheduler_steps"] + 1
            ):
                _fail("async checkpoint intent transition is ambiguous")
            pointer_matches_source = (
                pointer["generation"] == source["generation"]
                and pointer["snapshot_id"] == source["snapshot_id"]
                and pointer["policy_version"] == source["policy_version"]
                and pointer["optimizer_steps"] == source["optimizer_steps"]
            )
            pointer_matches_target = (
                pointer["generation"] == target["generation"]
                and pointer["snapshot_id"] == target["snapshot_id"]
                and pointer["policy_version"] == target["policy_version"]
                and pointer["optimizer_steps"] == target["optimizer_steps"]
            )
            if not (pointer_matches_source or pointer_matches_target):
                _fail("async checkpoint intent does not match the latest durable pointer")

        entries: list[dict[str, Any]] = []
        for name in sorted(selected):
            source = source_root / name
            if any(_is_within(source, excluded) for excluded in exclude_roots):
                _fail("required checkpoint file overlaps a configured credential path")
            try:
                info = source.lstat()
            except FileNotFoundError:
                if name == "latest.json" or name == pointer["snapshot"]:
                    _fail("latest full-checkpoint file disappeared during migration capture")
                # Async intent is written before the target payload. A named
                # snapshot that has not reached disk is part of that valid
                # recovery state and must remain absent in the export.
                continue
            except OSError:
                _fail("selected full-checkpoint file cannot be inspected")
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                _fail("selected full-checkpoint file is not a regular file")
            if name == "latest.json" and info.st_size > 1024 * 1024:
                _fail("latest full-checkpoint pointer is too large")
            relative = _safe_relative(f"{relative_root}/{name}")
            copied, digest = _copy_file(
                source,
                staging_root / relative,
                expected_size=info.st_size,
                progress=progress,
                stage="capture_full_checkpoint",
            )
            entries.append({"path": relative, "size": copied.st_size, "sha256": digest,
                            "kind": "full_checkpoint"})

        for name in (
            "prepared-window.json",
            "pending-attempt.json",
            "async-checkpoint-intent.json",
            "cancelled-windows.json",
        ):
            source = source_root / name
            if any(_is_within(source, excluded) for excluded in exclude_roots):
                _fail("required checkpoint sidecar overlaps a configured credential path")
            if not source.exists() and not source.is_symlink():
                continue
            try:
                info = source.lstat()
            except OSError:
                _fail("gradient recovery sidecar cannot be inspected")
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_size > 8 * 1024 * 1024:
                _fail("gradient recovery sidecar is not a supported small regular file")
            relative = _safe_relative(f"{relative_root}/{name}")
            copied, digest = _copy_file(source, staging_root / relative, expected_size=info.st_size,
                                        progress=progress, stage="capture_checkpoint_control")
            entries.append({"path": relative, "size": copied.st_size, "sha256": digest,
                            "kind": "full_checkpoint"})

        receipt_dir = source_root / "timing-receipts"
        if receipt_dir.exists() or receipt_dir.is_symlink():
            try:
                receipt_info = receipt_dir.lstat()
            except OSError:
                _fail("checkpoint timing receipt directory cannot be inspected")
            if not stat.S_ISDIR(receipt_info.st_mode) or stat.S_ISLNK(receipt_info.st_mode):
                _fail("checkpoint timing receipt directory is unsafe")
            for generation in sorted(generations):
                name = f"update-{generation:08d}.json"
                source = receipt_dir / name
                if any(_is_within(source, excluded) for excluded in exclude_roots):
                    _fail("required checkpoint receipt overlaps a configured credential path")
                if not source.exists() and not source.is_symlink():
                    continue
                try:
                    info = source.lstat()
                except OSError:
                    _fail("checkpoint timing receipt cannot be inspected")
                if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_size > 8 * 1024 * 1024:
                    _fail("checkpoint timing receipt is not a supported small regular file")
                relative = _safe_relative(f"{relative_root}/timing-receipts/{name}")
                copied, digest = _copy_file(source, staging_root / relative, expected_size=info.st_size,
                                            progress=progress, stage="capture_checkpoint_receipt")
                entries.append({"path": relative, "size": copied.st_size, "sha256": digest,
                                "kind": "full_checkpoint"})
        copied = _validate_checkpoint(staging_root / relative_root)
        after = _validate_checkpoint(source_root)
        if before != copied or before != after:
            _fail("full-checkpoint pointer or snapshot identity changed during migration capture")
        return before, entries


def _audit_bepa_artifact_refs(root: Path) -> int:
    """Read-only verify that every durable BEPA reference resolves in its artifact store."""
    from ..bepa.contracts import canonical_json_bytes, content_hash

    db_path = root / "state.sqlite3"
    if db_path.is_symlink() or not db_path.is_file():
        _fail("imported BEPA store is missing its SQLite database")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        required_tables = {"rescue_jobs", "episodes", "cache_entries", "plans"}
        if not required_tables <= tables:
            _fail("imported BEPA store schema is unsupported; artifact references are unknown")
        references: set[str] = set()
        queries = (
            ("SELECT source_artifact_refs_json,source_receipt_ref,plan_artifact_ref FROM rescue_jobs", "job"),
            ("SELECT raw_artifact_ref,projection_artifact_ref,receipt_artifact_ref FROM episodes", "episode"),
            ("SELECT raw_artifact_ref,projection_artifact_ref,receipt_artifact_ref FROM cache_entries", "cache"),
            ("SELECT plan_artifact_ref,source_receipt_ref FROM plans", "plan"),
        )
        for query, kind in queries:
            for row in connection.execute(query):
                values = dict(row)
                for key, value in values.items():
                    if value is None:
                        continue
                    if kind == "job" and key == "source_artifact_refs_json":
                        try:
                            refs = json.loads(value)
                        except (TypeError, ValueError):
                            _fail("BEPA source artifact references are malformed")
                        if not isinstance(refs, dict) or any(not isinstance(ref, str) for ref in refs.values()):
                            _fail("BEPA source artifact references are malformed")
                        references.update(refs.values())
                    else:
                        if not isinstance(value, str):
                            _fail("BEPA artifact reference has an invalid type")
                        references.add(value)
        for digest in references:
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                _fail("BEPA store contains an invalid artifact reference")
            artifact = root / "artifacts" / digest / "artifact.json"
            for parent in (root / "artifacts", root / "artifacts" / digest):
                try:
                    info = parent.lstat()
                except OSError:
                    _fail("BEPA store references a missing artifact directory")
                if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    _fail("BEPA store artifact path contains a symlink or non-directory")
            try:
                info = artifact.lstat()
                payload = artifact.read_bytes()
                value = json.loads(payload)
            except (OSError, UnicodeError, json.JSONDecodeError):
                _fail("BEPA store references a missing or invalid artifact payload")
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                _fail("BEPA artifact payload is not a regular file")
            if content_hash(value) != digest or canonical_json_bytes(value) != payload:
                _fail("BEPA store artifact payload failed its content identity check")
        return len(references)
    except InstanceMigrationError:
        raise
    except sqlite3.Error as exc:
        raise InstanceMigrationError(f"BEPA SQLite store could not be inspected read-only ({type(exc).__name__})") from exc
    finally:
        if connection is not None:
            connection.close()


def _sqlite_integrity_check(path: Path) -> None:
    if _sqlite_identity(path) is None:
        _fail("migration SQLite file is invalid")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=5.0)
    try:
        row = connection.execute("PRAGMA quick_check").fetchone()
        if not row or row[0] != "ok":
            _fail("migration SQLite file failed its read-only integrity check")
    except sqlite3.Error as exc:
        raise InstanceMigrationError(f"migration SQLite file could not be checked ({type(exc).__name__})") from exc
    finally:
        connection.close()


def _static_sources(layout: Mapping[str, Any]) -> tuple[tuple[str, Path], ...]:
    result = [
        ("runtime_config", Path(layout["runtime_config"])),
        ("dataset", Path(layout["dataset_dir"])),
        ("artifact_identity", Path(layout["artifact_identity_path"])),
        ("conversion_report", Path(layout["conversion_report_path"])),
        ("initial_adapter", Path(layout["initial_adapter_path"])),
    ]
    for name, key in (
        ("arc_contract", "arc_contract_path"),
        ("tokenizer", "tokenizer_path"),
        ("evaluation_manifest", "evaluation_manifest_path"),
        ("evaluation_sft_exclusion_proof", "evaluation_sft_exclusion_proof"),
        ("evaluation_initial_count_proof", "evaluation_initial_count_proof"),
    ):
        value = layout.get(key)
        if isinstance(value, str):
            result.append((name, Path(value)))
    return tuple(result)


def prepare_instance_migration(
    deployment: InstanceDeployment,
    output_dir: str | Path,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Stage reusable static inputs while the source instance may remain live."""
    if not isinstance(deployment, InstanceDeployment):
        _fail("prepare requires a loaded source instance deployment")
    if progress is not None and not callable(progress):
        _fail("progress must be a callable or None")
    output = _safe_path(output_dir)
    runtime = _load_runtime(deployment)
    layout = _source_layout(deployment, runtime)
    credential_paths = _credential_excludes(layout)
    for root_name in ("state_dir", "run_dir", "recovery_dir", "model_path", "dataset_dir"):
        source_root = Path(layout[root_name]).resolve()
        if _overlap(output, source_root):
            _fail("migration output must be outside the source instance and data roots")
    if output.exists():
        manifest = _read_manifest(output)
        if (manifest.get("source", {}).get("deployment_id") != layout["deployment_id"]
                or manifest.get("source", {}).get("credential_paths") != layout["credential_paths"]
                or manifest.get("status") != "prepared"):
            _fail("migration output already contains a different or finalized bundle")
        return {"status": "prepared", "migration_id": manifest["migration_id"],
                "bundle_dir": str(output), "file_count": len(manifest["files"]),
                "bytes_total": sum(row["size"] for row in manifest["files"]),
                "source_paths": manifest["source"], "reused_staging": True}
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output.with_name(f".{output.name}.prepare-{uuid.uuid4().hex}.tmp")
    _private_dir(temporary)
    try:
        files: list[dict[str, Any]] = []
        relative_names = {
            "runtime_config": "static/runtime.yaml",
            "dataset": "static/dataset",
            "artifact_identity": "static/artifact/identity.json",
            "conversion_report": "static/artifact/conversion-report.json",
            "initial_adapter": "static/artifact/initial-adapter",
            "arc_contract": "static/contracts/arc-contract",
            "tokenizer": "static/tokenizer",
            "evaluation_manifest": "static/evaluation/manifest",
            "evaluation_sft_exclusion_proof": "static/evaluation/sft-exclusion-proof",
            "evaluation_initial_count_proof": "static/evaluation/initial-count-proof",
        }
        for name, source in _static_sources(layout):
            _notify(progress, "prepare_static_input", input=name)
            files.extend(_copy_input(
                source, temporary, relative_names[name], kind=name,
                exclude_roots=credential_paths, progress=progress,
            ))
        manifest = {
            "schema": _SCHEMA,
            "migration_id": uuid.uuid4().hex,
            "status": "prepared",
            "created_at": _now(),
            "source": layout,
            "static_roots": {name: {"source_path": str(source), "bundle_path": relative_names[name]}
                             for name, source in _static_sources(layout)},
            "files": sorted(files, key=lambda row: row["path"]),
            "checkpoint": None,
        }
        _atomic_json(temporary / "manifest.json", manifest)
        if output.exists() or output.is_symlink():
            _fail("migration output appeared during staging")
        os.replace(temporary, output)
        _fsync_dir(output.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"status": "prepared", "migration_id": manifest["migration_id"],
            "bundle_dir": str(output), "file_count": len(files),
            "bytes_total": sum(row["size"] for row in files),
            "source_paths": layout, "reused_staging": False}


def _check_static_source_unchanged(bundle: Path, manifest: Mapping[str, Any]) -> None:
    credential_paths = _credential_excludes(manifest.get("source", {}))
    for name, root_info in manifest["static_roots"].items():
        source_root = Path(root_info["source_path"])
        expected_rows = [row for row in manifest["files"] if row.get("kind") == name]
        expected = {row.get("source_path"): row for row in expected_rows}
        try:
            root_stat = source_root.lstat()
        except OSError:
            _fail("prepared static input source is unavailable; reprepare before export")
        if any(_is_within(source_root, excluded) for excluded in credential_paths):
            _fail("prepared static input overlaps a configured credential path")
        if stat.S_ISDIR(root_stat.st_mode):
            actual = {str(path): info for path, _relative, info in
                      _source_entries(source_root, exclude_roots=credential_paths)}
        elif stat.S_ISREG(root_stat.st_mode):
            actual = {str(source_root): root_stat}
        else:
            _fail("prepared static input changed type; reprepare before export")
        if set(actual) != set(expected):
            _fail("prepared static input file set changed; reprepare before export")
        for source_path, info in actual.items():
            row = expected[source_path]
            if (not stat.S_ISREG(info.st_mode) or info.st_size != row.get("size")
                    or info.st_dev != row.get("source_device") or info.st_ino != row.get("source_inode")
                    or info.st_mtime_ns != row.get("source_mtime_ns")):
                _fail("prepared static input changed; reprepare before export")
            _safe_bundle_file(bundle, row["path"], row["size"], row["sha256"])


def _copy_tree_into(source: Path, staging_root: Path, relative_root: str, *, kind: str,
                    exclude_roots: tuple[Path, ...] = (),
                    progress: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    source = _safe_path(source, must_exist=True)
    if any(_is_within(source, excluded) for excluded in exclude_roots):
        _fail("durable migration root overlaps an excluded path")
    entries: list[dict[str, Any]] = []
    destination = staging_root / _safe_relative(relative_root)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_files = _source_entries(source, exclude_roots=exclude_roots)
    sqlite_files = {path for path, _suffix, _info in source_files if _sqlite_identity(path) is not None}
    sqlite_sidecars = {Path(str(database) + suffix) for database in sqlite_files
                       for suffix in ("-wal", "-shm", "-journal")}
    for index, (file_path, suffix, info) in enumerate(source_files, start=1):
        if file_path in sqlite_sidecars:
            continue
        relative = _safe_relative(f"{relative_root}/{suffix}")
        if file_path in sqlite_files:
            copied, digest = _copy_sqlite_snapshot(file_path, staging_root / relative,
                                                   progress=progress, stage=f"capture_{kind}_sqlite")
        else:
            copied, digest = _copy_file(file_path, staging_root / relative, expected_size=info.st_size,
                                        progress=progress, stage=f"capture_{kind}")
        entries.append({"path": relative, "size": copied.st_size, "sha256": digest, "kind": kind})
        _notify(progress, f"capture_{kind}", files_done=index)
    return entries


def _clean_stale_export_staging(bundle: Path) -> None:
    pattern = re.compile(r"\.durable-[0-9a-f]{32}\.tmp\Z")
    for child in bundle.iterdir():
        if pattern.fullmatch(child.name) is None:
            continue
        if child.is_symlink() or not child.is_dir():
            _fail("stale durable staging path is unsafe")
        shutil.rmtree(child)


def source_revision(source: Mapping[str, Any]) -> dict[str, Any]:
    """Observe all durable roots, including SQLite WAL, without reading model payloads."""
    files: dict[str, list[int]] = {}
    absent: list[str] = []
    excludes = list(_credential_excludes(source))
    for name in ("model_path", "dataset_dir"):
        if source.get(name):
            excludes.append(Path(source[name]))
    run = Path(source["run_dir"])
    excludes.extend((run / "frozen-base", run / "executor-policy-cache"))
    state = Path(source["state_dir"])
    for name in ("run_dir", "recovery_dir", "bepa_state_dir", "evaluation_state_dir"):
        value = source.get(name)
        if not value:
            continue
        root = Path(value)
        if not root.exists():
            absent.append(str(root))
            continue
        protected = (*excludes, state) if _is_within(state, root) else tuple(excludes)
        for path, _relative, info in _source_entries(root, exclude_roots=protected):
            if path.name.endswith("-shm"):
                continue
            files[str(path)] = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    return {"files": dict(sorted(files.items())), "absent_roots": sorted(absent)}


def _assert_stop_requests_resolved(deployment: InstanceDeployment, run_dir: Path) -> None:
    """Do not transfer a Learner stop latch without its source-side watcher."""
    paths = (
        (deployment.state_dir / "stop-operation.json", "qwen-rl-instance-stop-operation/v1", True),
        (run_dir / "control" / "stop-after-checkpoint.json", "qwen-rl-stop-after-checkpoint/v1", False),
    )
    for path, schema, manager in paths:
        if not path.exists() and not path.is_symlink():
            continue
        receipt = _read_json(path, private=True)
        if receipt.get("schema") != schema:
            _fail("source stop operation is malformed; resolve it before migration")
        if manager:
            status = receipt.get("manager_status")
            rejected = (receipt.get("mode") == "after_checkpoint" and status == "blocked"
                        and (receipt.get("learner_receipt") or {}).get("request_accepted") is False)
            resolved = status == "completed" or rejected
        else:
            from ..learner.stop_control import StopAfterCheckpointControl
            try:
                StopAfterCheckpointControl._validate(receipt)
            except (TypeError, ValueError):
                _fail("source Learner stop receipt is malformed; resolve it before migration")
            accepted = receipt.get("request_accepted")
            if type(accepted) is not bool:
                _fail("source Learner stop receipt has no acceptance proof")
            resolved = receipt.get("status") == "completed" or (
                receipt.get("status") == "blocked" and accepted is False
            )
        if not resolved:
            command = shlex.join(["qwen-rl", "instance", "--deployment", str(deployment.path), "stop-now"])
            _fail("source has an unresolved stop operation; finish it before migration or discard with: " + command)


def export_instance_migration(
    deployment: InstanceDeployment,
    output_dir: str | Path,
    *,
    stop: bool = False,
    refresh: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not isinstance(deployment, InstanceDeployment):
        _fail("export requires a loaded source instance deployment")
    from .instance_operations import operation_lock
    with operation_lock(deployment.state_dir):
        with _migration_operation_lock(deployment):
            return _export_instance_migration_locked(deployment, output_dir, stop=stop, refresh=refresh, progress=progress)


def _export_instance_migration_locked(
    deployment: InstanceDeployment,
    output_dir: str | Path,
    *,
    stop: bool = False,
    refresh: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Capture full durable state after an explicit managed stop or proven stop."""
    if not isinstance(deployment, InstanceDeployment):
        _fail("export requires a loaded source instance deployment")
    if type(stop) is not bool or type(refresh) is not bool:
        _fail("stop must be an explicit boolean")
    if progress is not None and not callable(progress):
        _fail("progress must be a callable or None")
    bundle = _safe_path(output_dir, must_exist=True)
    manifest = _read_manifest(bundle)
    source = manifest.get("source")
    if source.get("deployment_id") != _deployment_id(deployment):
        _fail("prepared migration bundle belongs to another source deployment")
    canonical_run = Path(_load_runtime(deployment).run_dir).expanduser().resolve()
    if (source.get("run_dir") != str(canonical_run)
            or source.get("state_dir") != str(deployment.state_dir.resolve())):
        _fail("source run or state paths differ from the selected deployment")
    _assert_stop_requests_resolved(deployment, canonical_run)
    if manifest["status"] == "exported" and refresh:
        with _bundle_import_lock(bundle):
            # Import re-reads this manifest after acquiring the same lock. Re-read
            # here too, so a claim created after our initial read cannot race the reset.
            manifest = _read_manifest(bundle)
            source = manifest.get("source")
            if source.get("deployment_id") != _deployment_id(deployment):
                _fail("prepared migration bundle belongs to another source deployment")
            if manifest["status"] != "exported":
                _fail("migration bundle changed before refresh could reset its durable snapshot")
            claim_path = bundle / "claim.json"
            if claim_path.exists() or claim_path.is_symlink():
                _fail("cannot refresh a migration already claimed by a destination")
            # Static staging remains useful; every durable root is captured again.
            manifest = {**manifest, "status": "prepared", "checkpoint": None,
                        "files": [row for row in manifest["files"] if row["path"].startswith("static/")]}
            manifest.pop("source_revision", None)
            _atomic_json(_manifest_path(bundle), manifest)
    if manifest["status"] == "exported":
        from .instance import InstanceSupervisor

        fence_path = _fence_path(deployment)
        with _owner_lock(deployment):
            if InstanceSupervisor(deployment).status().get("status") != "stopped" or not fence_path.exists():
                _fail("exported bundle cannot be reconciled without its stopped, fenced source deployment")
            fence = _read_json(fence_path, private=True)
            if (fence.get("schema") != _FENCE_SCHEMA or fence.get("migration_id") != manifest["migration_id"]
                    or fence.get("source_deployment_id") != source["deployment_id"]):
                _fail("exported bundle source fence does not match its migration identity")
            _verify_bundle_files(bundle, manifest)
            with _checkpoint_owner_lock(Path(source["recovery_dir"])):
                current_checkpoint = _validate_checkpoint(Path(source["recovery_dir"]))
            if source_revision(source) != manifest.get("source_revision"):
                _fail("source durable state changed after export, even if the checkpoint version is unchanged; refresh before import")
            if current_checkpoint != manifest.get("checkpoint"):
                _fail("source full checkpoint changed after the bundle was committed")
            manifest_sha = _manifest_sha256(bundle)
            _update_fence(fence_path, migration_id=manifest["migration_id"],
                          state="exported", manifest_sha256=manifest_sha)
            receipt_path = deployment.state_dir / "migration-receipts" / f"{manifest['migration_id']}.json"
            receipt = _source_migration_receipt(source, bundle, manifest, current_checkpoint)
            if receipt_path.exists():
                existing = _read_json(receipt_path, private=True)
                if existing != receipt:
                    _fail("source migration receipt conflicts with the committed bundle")
            else:
                receipts = deployment.state_dir / "migration-receipts"
                _private_dir(receipts)
                _atomic_json(receipt_path, receipt)
            return {**receipt, "file_count": len(manifest["files"]),
                    "bytes_total": sum(row["size"] for row in manifest["files"]), "reused_export": True}
    _check_static_source_unchanged(bundle, manifest)

    from .instance import InstanceSupervisor

    supervisor = InstanceSupervisor(deployment)
    _notify(progress, "checking_source_status")
    current = supervisor.status()
    if not stop and current.get("status") != "stopped":
        _fail("export requires a stopped source; pass explicit stop to use InstanceSupervisor.stop()")
    if stop and current.get("status") not in {"stopped", "starting", "running", "stopping", "failed"}:
        _fail("source process ownership is not proven; refusing to stop or export it")

    migration_id = manifest["migration_id"]
    source_owner_guard = None
    source_owner_guard_entered = False
    try:
        if current.get("status") == "stopped":
            source_owner_guard = _owner_lock(deployment)
            source_owner_guard.__enter__()
            source_owner_guard_entered = True
            fence_path = _create_fence(deployment, migration_id, state="exporting")
        else:
            # The live supervisor holds owner.lock. Fencing before stop prevents a
            # second serve from racing the managed stop and durable capture.
            fence_path = _create_fence(deployment, migration_id, state="exporting")
        if stop:
            _notify(progress, "stopping_source")
            stopped = supervisor.stop(progress=progress)
            if stopped.get("status") != "stopped":
                _fail("managed instance did not reach stopped state")
        elif current.get("status") != "stopped":
            _fail("source instance is not stopped")
        if source_owner_guard is None:
            # The stopped source released its lifetime owner lock. Hold it
            # through snapshot publication so serve and cancel cannot race the
            # durable capture.
            source_owner_guard = _owner_lock(deployment)
            source_owner_guard.__enter__()
            source_owner_guard_entered = True
        if supervisor.status().get("status") != "stopped":
            _fail("source instance is not proven stopped after managed stop")
        _notify(progress, "source_stopped")
        _clean_stale_export_staging(bundle)

        runtime = _load_runtime(deployment)
        layout = _source_layout(deployment, runtime)
        if layout["deployment_id"] != source["deployment_id"]:
            _fail("source deployment identity changed after static staging")
        if (layout["state_dir"] != source.get("state_dir")
                or layout["credential_paths"] != source.get("credential_paths")):
            _fail("source state or credential paths changed after static staging")
        _assert_stop_requests_resolved(deployment, Path(layout["run_dir"]))
        credential_paths = _credential_excludes(layout)
        checkpoint_root = Path(layout["recovery_dir"])
        revision_before = source_revision(layout)
        _notify(progress, "capturing_durable_state")
        exclude: list[Path] = [Path(layout["model_path"]), Path(layout["dataset_dir"])]
        for key in ("artifact_identity_path", "conversion_report_path", "initial_adapter_path"):
            exclude.append(Path(layout[key]))
        exclude_roots = (*exclude, *credential_paths)

        durable_tmp = bundle / f".durable-{uuid.uuid4().hex}.tmp"
        _private_dir(durable_tmp)
        try:
            durable_root = bundle / "durable"
            if durable_root.exists() or durable_root.is_symlink():
                if durable_root.is_symlink() or not durable_root.is_dir():
                    _fail("incomplete durable staging path is unsafe")
                shutil.rmtree(durable_root)
            dynamic_files: list[dict[str, Any]] = []
            run_root = Path(layout["run_dir"])
            run_excludes = exclude_roots
            state_root = Path(layout["state_dir"])
            if _is_within(state_root, run_root):
                run_excludes = (*run_excludes, state_root)
            if _is_within(checkpoint_root, run_root):
                optimizer_lock = checkpoint_root.parent / f".{checkpoint_root.name}.gradient-optimizer.lock"
                run_excludes = (*run_excludes, checkpoint_root, optimizer_lock)
            dynamic_files.extend(_copy_tree_into(run_root, durable_tmp, "durable/run",
                                                 kind="run", exclude_roots=run_excludes, progress=progress))
            checkpoint, checkpoint_files = _copy_checkpoint_snapshot(
                checkpoint_root, durable_tmp, "durable/recovery", progress=progress,
                exclude_roots=credential_paths,
            )
            dynamic_files.extend(checkpoint_files)
            _notify(progress, "checkpoint_captured", policy_version=checkpoint["policy_version"],
                    optimizer_steps=checkpoint["optimizer_steps"], checkpoint_bytes=checkpoint["snapshot_size"])
            bepa_root = Path(layout["bepa_state_dir"]) if layout["bepa_state_dir"] else None
            if bepa_root is not None and not _is_within(bepa_root, Path(layout["run_dir"])):
                dynamic_files.extend(_copy_tree_into(bepa_root, durable_tmp, "durable/bepa", kind="bepa",
                                                     exclude_roots=credential_paths,
                                                     progress=progress))
            evaluation_state = Path(layout["evaluation_state_dir"]) if layout["evaluation_state_dir"] else None
            if (evaluation_state is not None and not _is_within(evaluation_state, Path(layout["run_dir"]))
                    and (bepa_root is None or not _is_within(evaluation_state, bepa_root))):
                if evaluation_state.exists():
                    dynamic_files.extend(_copy_tree_into(evaluation_state, durable_tmp,
                                                         "durable/evaluation-state", kind="evaluation",
                                                         exclude_roots=credential_paths,
                                                         progress=progress))
                else:
                    (durable_tmp / "durable/evaluation-state").mkdir(parents=True, exist_ok=True, mode=0o700)
            if not any(row["kind"] == "full_checkpoint" and row["path"] == "durable/recovery/latest.json"
                       for row in dynamic_files):
                _fail("latest full-checkpoint pointer was not captured")
            # The only state copied from the instance manager root is a small
            # resume receipt; active.json, locks, role manifests and PID data
            # are deliberately excluded.
            resume_ledger = state_root / "resume-ledger.json"
            if (resume_ledger.exists()
                    and not any(_is_within(resume_ledger, path) for path in credential_paths)):
                info = resume_ledger.lstat()
                if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    _fail("instance resume ledger is unsafe")
                copied, digest = _copy_file(resume_ledger, durable_tmp / "durable/instance-resume-ledger.json",
                                            expected_size=info.st_size, progress=progress,
                                            stage="capture_instance_receipt")
                dynamic_files.append({"path": "durable/instance-resume-ledger.json",
                                      "size": copied.st_size, "sha256": digest,
                                      "kind": "instance_receipt"})
            os.replace(durable_tmp / "durable", durable_root)
            shutil.rmtree(durable_tmp, ignore_errors=True)
            _fsync_dir(bundle)
        except BaseException:
            shutil.rmtree(durable_tmp, ignore_errors=True)
            # Partial durable files are not named by a committed exported
            # manifest, so an importer will continue to reject this bundle.
            raise

        manifest["files"] = sorted([*manifest["files"], *dynamic_files], key=lambda row: row["path"])
        revision_after = source_revision(layout)
        if revision_after != revision_before:
            _fail("source durable state changed during final snapshot; retry export")
        manifest["source_revision"] = revision_after
        manifest["checkpoint"] = checkpoint
        manifest["status"] = "exported"
        manifest["exported_at"] = _now()
        _atomic_json(_manifest_path(bundle), manifest)
        _notify(progress, "bundle_committed", file_count=len(manifest["files"]),
                bytes_total=sum(row["size"] for row in manifest["files"]))
        manifest_sha = _manifest_sha256(bundle)
        _update_fence(fence_path, migration_id=migration_id, state="exported", manifest_sha256=manifest_sha)
        receipt = _source_migration_receipt(source, bundle, manifest, checkpoint)
        receipts = deployment.state_dir / "migration-receipts"
        _private_dir(receipts)
        _atomic_json(receipts / f"{migration_id}.json", receipt)
        return {**receipt, "file_count": len(manifest["files"]),
                "bytes_total": sum(row["size"] for row in manifest["files"]), "reused_export": False}
    except BaseException:
        # Keep the source fence after every failure. Explicit cancellation is
        # required before any future source resume.
        raise
    finally:
        if source_owner_guard_entered and source_owner_guard is not None:
            source_owner_guard.__exit__(None, None, None)


def _normalize_path_mappings(
    path_mappings: Mapping[str | Path, str | Path] | None,
    *,
    staging_roots: Mapping[str | Path, str | Path] | None = None,
) -> dict[str, str]:
    if path_mappings is None:
        return {}
    from .instance_relocation import normalize_path_mappings

    try:
        return normalize_path_mappings(path_mappings, staging_roots=staging_roots)
    except (TypeError, ValueError) as exc:
        raise InstanceMigrationError(f"path_mappings are invalid ({type(exc).__name__})") from exc


def _automatic_path_mappings(source: Mapping[str, Any], installed: Path) -> dict[str, str]:
    """Paths that participate in checkpoint semantic comparison (not storage roots)."""
    result = {
        source["dataset_dir"]: str(installed / "static/dataset"),
        source["initial_adapter_path"]: str(installed / "static/artifact/initial-adapter"),
    }
    return result


def _storage_path_mappings(source: Mapping[str, Any], installed: Path) -> dict[str, str]:
    run_target = installed / "durable/run"
    result = {
        source["run_dir"]: str(run_target),
        source["recovery_dir"]: str(installed / "durable/recovery"),
        source["dataset_dir"]: str(installed / "static/dataset"),
        source["artifact_identity_path"]: str(installed / "static/artifact/identity.json"),
        source["conversion_report_path"]: str(installed / "static/artifact/conversion-report.json"),
        source["initial_adapter_path"]: str(installed / "static/artifact/initial-adapter"),
    }
    bepa = source.get("bepa_state_dir")
    if isinstance(bepa, str):
        old_bepa, old_run = Path(bepa), Path(source["run_dir"])
        result[bepa] = (str(run_target / old_bepa.relative_to(old_run))
                        if _is_within(old_bepa, old_run) else str(installed / "durable/bepa"))
    static_roots = {
        "arc_contract_path": "arc_contract",
        "tokenizer_path": "tokenizer",
        "evaluation_manifest_path": "evaluation_manifest",
        "evaluation_sft_exclusion_proof": "evaluation_sft_exclusion_proof",
        "evaluation_initial_count_proof": "evaluation_initial_count_proof",
    }
    static_targets = {
        "arc_contract": "static/contracts/arc-contract",
        "tokenizer": "static/tokenizer",
        "evaluation_manifest": "static/evaluation/manifest",
        "evaluation_sft_exclusion_proof": "static/evaluation/sft-exclusion-proof",
        "evaluation_initial_count_proof": "static/evaluation/initial-count-proof",
    }
    for source_key, static_name in static_roots.items():
        value = source.get(source_key)
        if isinstance(value, str):
            result[value] = str(installed / static_targets[static_name])
    evaluation_state = source.get("evaluation_state_dir")
    if isinstance(evaluation_state, str):
        old_state, old_run = Path(evaluation_state), Path(source["run_dir"])
        if _is_within(old_state, old_run):
            result[evaluation_state] = str(run_target / old_state.relative_to(old_run))
        elif isinstance(bepa, str) and _is_within(old_state, Path(bepa)):
            bepa_target = Path(result[bepa])
            result[evaluation_state] = str(bepa_target / old_state.relative_to(Path(bepa)))
        else:
            result[evaluation_state] = str(installed / "durable/evaluation-state")
    return result


def _map_target(value: str, mappings: Mapping[str, str]) -> str:
    path = Path(value)
    for old, new in sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True):
        try:
            suffix = path.relative_to(Path(old))
        except ValueError:
            continue
        return str(Path(new) / suffix)
    return value


def _compose_checkpoint_mappings(
    existing: Mapping[str, str],
    current_to_new: Mapping[str, str],
) -> dict[str, str]:
    """Compose prior relocations so saved original roots still reach this import."""
    from .instance_relocation import normalize_path_mappings

    prior = normalize_path_mappings(existing)
    direct = normalize_path_mappings(current_to_new)
    composed = {old: _map_target(current, direct) for old, current in prior.items()}
    for current, new in direct.items():
        if any(_is_within(Path(current), Path(previous_target)) for previous_target in prior.values()):
            continue
        composed[current] = new
    try:
        return normalize_path_mappings(composed)
    except (TypeError, ValueError) as exc:
        raise InstanceMigrationError(f"checkpoint path mappings are ambiguous ({type(exc).__name__})") from exc


def _write_target_configuration(
    bundle: Path,
    manifest: Mapping[str, Any],
    output: Path,
    installed: Path,
    target: InstanceDeployment,
    mappings: Mapping[str, str],
) -> dict[str, Any]:
    """Generate a target deployment from source training semantics and target host inputs."""
    from .instance_setup import setup_instance
    from ..runtime.config import load_config

    try:
        target_runtime = load_config(target.runtime_config, for_local_worker_plan=True)
    except Exception as exc:
        raise InstanceMigrationError(f"target runtime configuration is invalid ({type(exc).__name__})") from exc
    _require_matching_ninfer_artifact(manifest["source"], target_runtime)
    if not target_runtime.artifact_identity_path or not target_runtime.worker_inference.ninfer_conversion_report:
        _fail("target deployment must declare its verified artifact identity and conversion report")

    target_model = str(Path(target_runtime.model_path).expanduser().resolve(strict=True))
    effective_mappings = dict(mappings)
    source_model = str(manifest["source"]["model_path"])
    if source_model in effective_mappings and effective_mappings[source_model] != target_model:
        _fail("explicit model path mapping differs from the model declared by the target deployment")
    if source_model != target_model:
        effective_mappings[source_model] = target_model
    else:
        # A baked image may expose the same shared model path at source and
        # destination. Identity maps are invalid checkpoint mappings.
        effective_mappings.pop(source_model, None)

    source = manifest["source"]
    runtime_mappings = _storage_path_mappings(source, installed)
    runtime_mappings[source["model_path"]] = target_model
    for old, new in effective_mappings.items():
        if old in runtime_mappings and runtime_mappings[old] != new:
            if old == source_model and new == target_model:
                continue
            _fail("explicit path map conflicts with a bundled runtime path")
        runtime_mappings[old] = new

    # Rewrite all source roots for the new deployment while only persisting
    # identity-bearing roots in learner.recovery_path_mappings. Storage roots
    # can be nested (for example BEPA under run_dir) and are not checkpoint
    # compatibility identities.
    from .autodl_config import _map_runtime_paths, _path_mappings as _runtime_path_pairs
    import yaml

    try:
        template = yaml.safe_load((bundle / "static/runtime.yaml").read_text(encoding="utf-8"))
    except Exception as exc:
        raise InstanceMigrationError(f"bundled runtime template cannot be read ({type(exc).__name__})") from exc
    if not isinstance(template, dict):
        _fail("bundled runtime template must be a mapping")
    _map_runtime_paths(template, _runtime_path_pairs(runtime_mappings))
    worker_template = template.get("worker_inference")
    if isinstance(worker_template, dict):
        for field in ("ninfer_url", "ninfer_converter_python", "ninfer_converter_path"):
            worker_template[field] = getattr(target_runtime.worker_inference, field, None)
    learner = template.get("learner")
    if not isinstance(learner, dict):
        _fail("bundled runtime template has no learner mapping")
    for field in ("gradient_initial_training_dir", "packed_weight_offload_store_dir"):
        configured = learner.get(field)
        if configured:
            target_value = getattr(target_runtime.learner, field, None)
            if target_value:
                target_path = _resolved_config_path(target_value, label=field, must_exist=True)
                learner[field] = str(target_path)
            elif not _is_within(Path(configured).resolve(strict=False), installed):
                _fail(f"source {field} is external and has no verified destination equivalent")
    existing_maps = learner.get("recovery_path_mappings", {})
    learner["recovery_path_mappings"] = _compose_checkpoint_mappings(existing_maps, effective_mappings)
    effective_mappings = learner["recovery_path_mappings"]
    template_path = output / f".runtime-template-{manifest['migration_id']}.yaml"
    template_data = yaml.safe_dump(template, sort_keys=False, allow_unicode=True).encode("utf-8")
    descriptor = os.open(template_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(template_data)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_dir(output)

    config_parent = output / "config"
    _private_dir(config_parent)
    config_dir = config_parent / f"migration-{manifest['migration_id']}"
    if config_dir.exists():
        try:
            runtime_path = config_dir / "runtime.yaml"
            deployment_path = config_dir / "deployment.json"
            if runtime_path.is_file() and deployment_path.is_file():
                try:
                    generated_runtime = load_config(runtime_path, for_local_worker_plan=True)
                    generated_deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise InstanceMigrationError(
                        f"existing generated destination config is invalid ({type(exc).__name__})"
                    ) from exc
                if (generated_runtime.learner.recovery_path_mappings != effective_mappings
                        or generated_deployment.get("runtime_config") != str(runtime_path)):
                    _fail("existing generated destination config does not match this migration")
                target_values = target.public_dict()
                if generated_deployment.get("cache_root") != target_values.get("cache_root"):
                    _fail("existing generated deployment differs from the target cache_root")
                if generated_deployment.get("packed_cache_root") != target_values.get("packed_cache_root"):
                    _fail("existing generated deployment differs from the target packed_cache_root")
                if any(generated_deployment.get(key) != target_values[key] for key in (
                    "state_dir", "gpus", "engine_binary", "worker_python", "python", "start_timeout",
                    "stop_timeout", "gateway_config", "credential_files", "teacher_base_url",
                    "dashboard_port", "dashboard_host",
                )):
                    _fail("existing generated deployment differs from the current destination settings")
                if (Path(generated_runtime.model_path).resolve() != Path(target_runtime.model_path).resolve()
                        or Path(generated_runtime.artifact_identity_path).resolve()
                        != Path(target_runtime.artifact_identity_path).resolve()
                        or Path(generated_runtime.worker_inference.ninfer_conversion_report).resolve()
                        != Path(target_runtime.worker_inference.ninfer_conversion_report).resolve()
                        or Path(generated_runtime.dataset_dir).resolve()
                        != (installed / "static/dataset").resolve()):
                    _fail("existing generated runtime differs from the current target artifact or dataset")
                return {"runtime_config": str(runtime_path), "deployment": str(deployment_path),
                        "next_resume_command": f"qwen-rl instance --deployment {shlex.quote(str(deployment_path))} resume --wait ready",
                        "path_mappings": effective_mappings,
                        "runtime_config_sha256": _file_sha256(runtime_path, max_size=2 * 1024 * 1024)[1],
                        "deployment_config_sha256": _file_sha256(deployment_path, max_size=2 * 1024 * 1024)[1]}
            _fail("destination config directory is incomplete; refusing to overwrite it")
        finally:
            template_path.unlink(missing_ok=True)

    temporary_config = config_parent / f".migration-{manifest['migration_id']}.{uuid.uuid4().hex}.tmp"
    try:
        setup_result = setup_instance(
            runtime_config=template_path,
            output_dir=temporary_config,
            gpus=target.gpus,
            model_path=target_runtime.model_path,
            dataset_dir=installed / "static/dataset",
            artifact_identity_path=target_runtime.artifact_identity_path,
            conversion_report=target_runtime.worker_inference.ninfer_conversion_report,
            state_dir=target.state_dir,
            packed_cache_root=target.packed_cache_root,
            python=target.python,
            native_binary=target.engine_binary,
            install_root=target_runtime.learner.compute_pythonpath,
            credential_paths=target.credential_files,
            teacher_base_url=target.teacher_base_url,
            path_mappings=None,
        )
        temporary_deployment_path = Path(setup_result["configuration_paths"]["deployment"])
        temporary_runtime_path = Path(setup_result["configuration_paths"]["runtime_config"])
        generated = _read_json(temporary_deployment_path)
        generated_runtime_path = config_dir / "runtime.yaml"
        generated_deployment_path = config_dir / "deployment.json"
        target_values = target.public_dict()
        for key in ("state_dir", "gpus", "engine_binary", "worker_python", "python", "start_timeout",
                    "stop_timeout", "gateway_config", "credential_files", "teacher_base_url",
                    "dashboard_port", "dashboard_host"):
            generated[key] = target_values[key]
        if target.cache_root is not None:
            generated["cache_root"] = str(target.cache_root)
        if target.packed_cache_root is not None:
            generated["packed_cache_root"] = str(target.packed_cache_root)
        generated["runtime_config"] = str(generated_runtime_path)
        _atomic_json(temporary_deployment_path, generated)
        final_runtime = load_config(temporary_runtime_path, for_local_worker_plan=True)
        final_deployment = json.loads(temporary_deployment_path.read_text(encoding="utf-8"))
    except InstanceMigrationError:
        shutil.rmtree(temporary_config, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temporary_config, ignore_errors=True)
        raise InstanceMigrationError(f"generated destination config is invalid ({type(exc).__name__})") from exc
    finally:
        template_path.unlink(missing_ok=True)
    try:
        if config_dir.exists() or config_dir.is_symlink():
            _fail("destination config appeared during migration; refusing to replace it")
        if final_runtime.learner.recovery_path_mappings != effective_mappings:
            _fail("generated destination runtime omitted the migration recovery path map")
        if final_deployment.get("runtime_config") != str(generated_runtime_path):
            _fail("generated destination deployment does not reference its runtime config")
        os.replace(temporary_config, config_dir)
        _fsync_dir(config_parent)
    except BaseException as exc:
        shutil.rmtree(temporary_config, ignore_errors=True)
        if isinstance(exc, InstanceMigrationError):
            raise
        raise InstanceMigrationError("generated destination configuration could not be published") from exc
    return {"runtime_config": str(generated_runtime_path), "deployment": str(generated_deployment_path),
            "next_resume_command": f"qwen-rl instance --deployment {shlex.quote(str(generated_deployment_path))} resume --wait ready",
            "path_mappings": effective_mappings,
            "runtime_path_mappings": runtime_mappings,
            "runtime_config_sha256": _file_sha256(generated_runtime_path, max_size=2 * 1024 * 1024)[1],
            "deployment_config_sha256": _file_sha256(generated_deployment_path, max_size=2 * 1024 * 1024)[1]}


def _target_identity(output: Path, deployment: InstanceDeployment | str | Path | None) -> tuple[str, dict[str, Any] | None]:
    if deployment is None:
        stable_identity = {"output_dir": str(output)}
        identity = dict(stable_identity)
    else:
        from .instance_plan import load_deployment
        from ..runtime.config import load_config

        target = deployment if isinstance(deployment, InstanceDeployment) else load_deployment(deployment)
        runtime = load_config(target.runtime_config, for_local_worker_plan=True)
        identity_size, identity_sha = _file_sha256(
            Path(runtime.artifact_identity_path).resolve(strict=True), max_size=16 * 1024 * 1024,
        )
        report_size, report_sha = _file_sha256(
            Path(runtime.worker_inference.ninfer_conversion_report).resolve(strict=True),
            max_size=16 * 1024 * 1024,
        )
        artifact = (runtime.artifact_identity.get("source", {}).get("ninfer_artifact", {})
                    if isinstance(runtime.artifact_identity, dict) else {})
        stable_identity = {"output_dir": str(output), "deployment_id": _deployment_id(target),
                           "deployment_path": str(target.path.resolve()),
                           "state_dir": str(target.state_dir.resolve()),
                           "runtime_config": str(target.runtime_config.resolve())}
        identity = {**stable_identity,
                    "deployment_config_sha256": _file_sha256(target.path, max_size=2 * 1024 * 1024)[1],
                    "runtime_config_sha256": _file_sha256(target.runtime_config, max_size=2 * 1024 * 1024)[1],
                    "artifact_id": artifact.get("artifact_id"),
                    "artifact_identity_size": identity_size, "artifact_identity_sha256": identity_sha,
                    "conversion_report_size": report_size, "conversion_report_sha256": report_sha}
    encoded = json.dumps(stable_identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20], identity


@contextmanager
def _bundle_import_lock(bundle: Path) -> Iterator[None]:
    path = bundle / ".import.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            _fail("migration import lock is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("migration import is already in progress for this bundle")
        yield
    finally:
        os.close(descriptor)


@contextmanager
def _destination_import_lock(deployment: InstanceDeployment | None) -> Iterator[None]:
    if deployment is None:
        yield
        return
    from .instance import InstanceSupervisor

    with _owner_lock(deployment):
        if InstanceSupervisor(deployment).status().get("status") != "stopped":
            _fail("destination instance must be stopped before importing migration state")
        yield


def import_instance_migration(
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    deployment: InstanceDeployment | str | Path | None = None,
    path_mappings: Mapping[str | Path, str | Path] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Install an exported bundle atomically without deserializing checkpoint data."""
    bundle = _safe_path(bundle_dir, must_exist=True)
    output = _safe_path(output_dir)
    if progress is not None and not callable(progress):
        _fail("progress must be a callable or None")
    if _overlap(bundle, output):
        _fail("migration output must be separate from its bundle")
    manifest = _read_manifest(bundle, require_exported=True)
    for entry in manifest["files"]:
        _safe_bundle_file(bundle, entry["path"], entry["size"])
    source = manifest["source"]
    source_roots = [Path(source[key]).resolve(strict=False)
                    for key in ("state_dir", "run_dir", "recovery_dir", "model_path", "dataset_dir")]
    if any(_overlap(output, root) for root in source_roots):
        _fail("destination must be separate from all source instance and data roots")
    target_id, target_identity = _target_identity(output, deployment)
    requested_mappings = _normalize_path_mappings(path_mappings)
    manifest_sha = _manifest_sha256(bundle)
    target_deployment: InstanceDeployment | None = None
    if deployment is not None:
        from .instance_plan import load_deployment
        from ..runtime.config import load_config

        target_deployment = deployment if isinstance(deployment, InstanceDeployment) else load_deployment(deployment)
        try:
            target_runtime = load_config(target_deployment.runtime_config, for_local_worker_plan=True)
        except Exception as exc:
            raise InstanceMigrationError(f"target runtime configuration is invalid ({type(exc).__name__})") from exc
        _require_matching_ninfer_artifact(source, target_runtime)
        target_roots = [target_deployment.state_dir.resolve(strict=False),
                        Path(target_runtime.run_dir).resolve(strict=False),
                        Path(target_runtime.learner.recovery_dir).resolve(strict=False)]
        if target_runtime.bepa.enabled:
            raw_bepa = target_runtime.bepa.state_dir or (Path(target_runtime.run_dir) / "bepa")
            target_roots.append(Path(raw_bepa).resolve(strict=False))
        if any(_overlap(target_root, source_root) for target_root in target_roots for source_root in source_roots):
            _fail("destination deployment still points into source state; relocate paths before import")

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    _private_dir(output)
    install_root = output / "migration-state"
    claim_path = bundle / "claim.json"
    with _destination_import_lock(target_deployment), _bundle_import_lock(bundle):
        # Re-read after acquiring the bundle-local lock so concurrent importers
        # cannot act on stale claim or manifest state.
        manifest = _read_manifest(bundle, require_exported=True)
        source = manifest["source"]
        if claim_path.exists():
            claim = _read_json(claim_path, private=True)
            if (claim.get("schema") != _CLAIM_SCHEMA or claim.get("migration_id") != manifest["migration_id"]
                    or claim.get("destination_id") != target_id
                    or claim.get("manifest_sha256") != manifest_sha
                    or claim.get("requested_path_mappings") != requested_mappings):
                _fail("migration bundle has already been claimed by another destination")
            if claim.get("status") not in {"claiming", "imported", "claimed"}:
                _fail("migration bundle claim state is unsupported")
        else:
            claim = {
                "schema": _CLAIM_SCHEMA, "migration_id": manifest["migration_id"],
                "destination_id": target_id, "destination": target_identity,
                "manifest_sha256": manifest_sha, "requested_path_mappings": requested_mappings,
                "status": "claiming", "claimed_at": _now(),
            }
            try:
                descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                _fail("migration bundle was claimed concurrently")
            with os.fdopen(descriptor, "wb") as stream:
                stream.write((json.dumps(claim, sort_keys=True, separators=(",", ":")) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_dir(bundle)

        installed_claim: dict[str, Any] | None = None
        reused_install = False
        if install_root.exists() or install_root.is_symlink():
            if install_root.is_symlink() or not install_root.is_dir():
                _fail("destination migration-state path is unsafe")
            installed_claim = _read_json(install_root / "claim.json", private=True)
            if (installed_claim.get("schema") != _CLAIM_SCHEMA
                    or installed_claim.get("migration_id") != manifest["migration_id"]
                    or installed_claim.get("destination_id") != target_id
                    or installed_claim.get("manifest_sha256") != manifest_sha
                    or installed_claim.get("requested_path_mappings") != requested_mappings):
                _fail("destination migration-state belongs to another handoff")
            if installed_claim.get("status") == "claimed":
                if installed_claim.get("destination") != target_identity:
                    _fail("claimed destination configuration identity changed after migration import")
                _verify_claimed_import(
                    bundle, manifest, install_root, installed_claim,
                    destination_id=target_id, destination_identity=target_identity,
                    requested_path_mappings=requested_mappings,
                )
                if claim.get("status") != "claimed":
                    _atomic_json(claim_path, installed_claim)
                return {**installed_claim, "reused_import": True}
            if installed_claim.get("status") != "imported":
                _fail("destination migration-state has an unsupported claim state")
            _verify_claimed_import(
                bundle, manifest, install_root, installed_claim,
                destination_id=target_id, destination_identity=target_identity,
                requested_path_mappings=requested_mappings,
            )
            installed_claim = {**installed_claim, "destination": target_identity}
            claim = {**claim, "destination": target_identity}
            _atomic_json(install_root / "claim.json", installed_claim)
            _atomic_json(claim_path, claim)
            if claim.get("status") == "claimed":
                _fail("claimed migration bundle is missing its final destination state")
            reused_install = True
        else:
            if claim.get("status") in {"imported", "claimed"}:
                _fail("migration bundle claim receipt names destination state that is missing")
            temporary = output / f".migration-state.import-{uuid.uuid4().hex}.tmp"
            _private_dir(temporary)
            try:
                for entry in manifest["files"]:
                    relative = _safe_relative(entry["path"])
                    source_file = _safe_bundle_file(bundle, relative, entry["size"])
                    _copy_file(source_file, temporary / relative, expected_size=entry["size"],
                               expected_sha256=entry["sha256"], progress=progress, stage="import")
                    _notify(progress, "import", file_complete=True)
                bepa_root = source.get("bepa_state_dir")
                if isinstance(bepa_root, str):
                    old_bepa, old_run = Path(bepa_root), Path(source["run_dir"])
                    staged_bepa = (temporary / "durable/run" / old_bepa.relative_to(old_run)
                                   if _is_within(old_bepa, old_run) else temporary / "durable/bepa")
                    references = _audit_bepa_artifact_refs(staged_bepa)
                    _notify(progress, "audit_bepa_references", artifact_count=references)
                evaluation_state = source.get("evaluation_state_dir")
                if isinstance(evaluation_state, str):
                    old_state, old_run = Path(evaluation_state), Path(source["run_dir"])
                    nested_bepa = isinstance(bepa_root, str) and _is_within(old_state, Path(bepa_root))
                    if not _is_within(old_state, old_run) and not nested_bepa:
                        (temporary / "durable/evaluation-state").mkdir(parents=True, exist_ok=True, mode=0o700)
                operational_file_receipts: dict[str, dict[str, Any]] = {}
                gradient_store_path_relocation: dict[str, Any] | None = None
                gradient_database = temporary / "durable/run/gradient-server/jobs.sqlite3"
                if gradient_database.exists():
                    if _sqlite_identity(gradient_database) is None:
                        _fail("imported gradient store is not a valid SQLite database")
                    # A clean idle store can have no artifact files. File-only
                    # transport omits its empty directories; rebuild the root
                    # before validating any recorded active file references.
                    (gradient_database.parent / "artifacts").mkdir(mode=0o700, exist_ok=True)
                    from ..learner.gradient_server.store import relocate_gradient_store_paths

                    source_run = Path(source["run_dir"])
                    source_artifact_root = source_run / "gradient-server/artifacts"
                    final_artifact_root = install_root / "durable/run/gradient-server/artifacts"
                    gradient_store_path_relocation = relocate_gradient_store_paths(
                        gradient_database,
                        path_mappings={source_run: install_root / "durable/run"},
                        source_artifact_root=source_artifact_root,
                        allowed_root=final_artifact_root,
                    )
                    _sqlite_integrity_check(gradient_database)
                    installed_size, installed_sha = _file_sha256(gradient_database)
                    operational_file_receipts["durable/run/gradient-server/jobs.sqlite3"] = {
                        "size": installed_size, "sha256": installed_sha,
                    }
                    _notify(progress, "relocate_gradient_store_paths", **gradient_store_path_relocation)
                evaluation_relocation = None
                if isinstance(evaluation_state, str):
                    from ..evaluation.relocation import relocate_evaluation_references
                    storage_paths = _storage_path_mappings(source, install_root)
                    final_eval = Path(storage_paths[evaluation_state])
                    staged_eval = temporary / final_eval.relative_to(install_root)
                    staged_eval.mkdir(parents=True, exist_ok=True, mode=0o700)
                    evaluation_mappings = dict(storage_paths)
                    for old, new in requested_mappings.items():
                        if old in evaluation_mappings and evaluation_mappings[old] != new:
                            _fail("explicit path map conflicts with a bundled migration root")
                        evaluation_mappings[old] = new
                    evaluation_relocation = relocate_evaluation_references(
                        staged_eval, path_mappings=evaluation_mappings,
                    )
                    for changed in evaluation_relocation["files_updated"]:
                        relative = str(Path(changed).relative_to(temporary))
                        size, digest = _file_sha256(Path(changed))
                        operational_file_receipts[relative] = {"size": size, "sha256": digest}
                    _notify(progress, "relocate_evaluation_references",
                            references_updated=evaluation_relocation["references_updated"])
                _atomic_json(temporary / "source-manifest.json", manifest)
                mappings = _automatic_path_mappings(source, install_root)
                for old, new in requested_mappings.items():
                    if old in mappings and mappings[old] != new:
                        _fail("explicit path map conflicts with a bundled migration root")
                    mappings[old] = new
                mappings = _normalize_path_mappings(
                    mappings, staging_roots={install_root: temporary},
                )
                imported_claim = {
                    **claim, "status": "imported", "installed_root": str(install_root),
                    "path_mappings": mappings, "checkpoint": manifest.get("checkpoint"),
                    "operational_file_receipts": operational_file_receipts,
                    "gradient_store_path_relocation": gradient_store_path_relocation,
                    "evaluation_path_relocation": evaluation_relocation,
                    "imported_at": _now(),
                }
                _atomic_json(temporary / "claim.json", imported_claim)
                _fsync_dir(temporary)
                os.replace(temporary, install_root)
                _fsync_dir(output)
                installed_claim = imported_claim
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            claim = {**claim, "status": "imported", "installed_root": str(install_root),
                     "imported_at": installed_claim["imported_at"]}
            _atomic_json(claim_path, claim)

        assert installed_claim is not None
        mappings = dict(installed_claim["path_mappings"])
        storage_mappings = dict(installed_claim.get("storage_path_mappings") or
                                _storage_path_mappings(source, install_root))
        configuration: dict[str, Any] = {}
        if deployment is not None:
            target_deployment = deployment if isinstance(deployment, InstanceDeployment) else target_deployment
            latest_target_id, latest_target_identity = _target_identity(output, deployment)
            if latest_target_id != target_id or latest_target_identity != target_identity:
                _fail("target deployment configuration changed during import; retry with its current config")
            _notify(progress, "generate_destination_configuration")
            configuration = _write_target_configuration(
                bundle, manifest, output, install_root, target_deployment, mappings,
            )
            mappings = dict(configuration.pop("path_mappings"))
            latest_target_id, latest_target_identity = _target_identity(output, deployment)
            if latest_target_id != target_id or latest_target_identity != target_identity:
                _fail("target deployment configuration changed during import; retry with its current config")
            _notify(progress, "destination_configuration_ready")
        final_claim = {
            **installed_claim, "status": "claimed", "path_mappings": mappings,
            "storage_path_mappings": storage_mappings,
            "configuration": configuration, "claimed_at": _now(),
            "manifest_sha256": manifest_sha, "requested_path_mappings": requested_mappings,
        }
        _atomic_json(install_root / "claim.json", final_claim)
        _atomic_json(claim_path, final_claim)
        _notify(progress, "import_claimed", migration_id=manifest["migration_id"])
        return {**final_claim, "reused_import": reused_install}


def cancel_instance_migration(
    deployment: InstanceDeployment,
    *,
    acknowledge_destination_stopped: bool = False,
) -> dict[str, Any]:
    if not isinstance(deployment, InstanceDeployment):
        _fail("cancel requires a loaded source instance deployment")
    with _migration_operation_lock(deployment):
        return _cancel_instance_migration_locked(
            deployment, acknowledge_destination_stopped=acknowledge_destination_stopped,
        )


def _cancel_instance_migration_locked(
    deployment: InstanceDeployment,
    *,
    acknowledge_destination_stopped: bool = False,
) -> dict[str, Any]:
    """Release a source fence only after explicit destination-stopped acknowledgement."""
    if acknowledge_destination_stopped is not True:
        _fail("cancelling a migration requires explicit acknowledgement that the destination is stopped")
    fence_path = _fence_path(deployment)
    if not fence_path.exists() and not fence_path.is_symlink():
        return {"status": "no_migration_fence", "source_deployment_id": _deployment_id(deployment)}
    fence = _read_json(fence_path, private=True)
    if fence.get("schema") != _FENCE_SCHEMA or not isinstance(fence.get("migration_id"), str):
        _fail("source migration fence is malformed; manual recovery is required")
    from .instance import InstanceSupervisor

    with _owner_lock(deployment):
        if InstanceSupervisor(deployment).status().get("status") != "stopped":
            _fail("source instance must be stopped before migration cancellation")
        current_fence = _read_json(fence_path, private=True)
        if current_fence.get("migration_id") != fence["migration_id"]:
            _fail("source migration fence changed during cancellation")
        archived = deployment.state_dir / f"migration-cancelled-{fence['migration_id']}.json"
        if archived.exists():
            prior = _read_json(archived, private=True)
            if prior.get("migration_id") != fence["migration_id"] or prior.get("status") != "cancelled":
                _fail("migration cancellation receipt conflicts with the source fence")
        else:
            if archived.is_symlink():
                _fail("migration cancellation receipt path is unsafe")
            current_fence.update(status="cancelled", cancelled_at=_now(),
                                 destination_stopped_acknowledged=True)
            _atomic_json(archived, current_fence)
        fence_path.unlink()
        _fsync_dir(deployment.state_dir)
    return {"status": "cancelled", "migration_id": fence["migration_id"],
            "source_deployment_id": _deployment_id(deployment),
            "destination_stopped_acknowledged": True}
