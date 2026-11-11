"""Verified resumable file transfer to approved Fleet nodes over OpenSSH SFTP."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import FleetNodeView
from .settings import FleetSettings
from .ssh import FleetSSHError, admin_ssh_argv, run_admin_ssh

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPERATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_REMOTE_HELPER = Path(__file__).with_name("transfer_remote.py")
_PROGRESS_INTERVAL = 5.0
_SFTP_BUFFER_SIZE = 16_384
_SFTP_REQUESTS = 128


class FleetTransferError(RuntimeError):
    """Expected secret-safe failure while transferring files to a Fleet node."""


@dataclass(frozen=True)
class TransferFile:
    """One local source pinned by size and SHA-256 to a HOME-relative path."""

    source: Path
    remote_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class _SourceSnapshot:
    source: Path
    device: int
    inode: int
    size: int
    mtime_ns: int


def _valid_relative(value: str, *, label: str) -> tuple[str, ...]:
    if (not isinstance(value, str) or not value or len(value) > 4096
            or any(ord(char) < 32 or ord(char) in {127, 0x85, 0x2028, 0x2029} for char in value)):
        raise FleetTransferError(f"{label} must be a safe relative path")
    path = PurePosixPath(value)
    parts = value.split("/")
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise FleetTransferError(f"{label} must be a safe relative path")
    return tuple(parts)


def _validate_node(node: FleetNodeView) -> None:
    if not isinstance(node, FleetNodeView):
        raise FleetTransferError("transfer requires a registered Fleet node view")
    if (node.approval != "approved" or node.heartbeat_stale or node.reverse_port is None
            or (node.desired_action == "deactivate"
                and node.desired_generation != node.observed_generation)):
        raise FleetTransferError("Fleet node is not currently eligible for transfer")
    if type(node.account.uid) is not int or node.account.uid <= 0:
        raise FleetTransferError("Fleet node account UID is invalid")


def _validate_source(item: TransferFile) -> _SourceSnapshot:
    if not isinstance(item, TransferFile) or not isinstance(item.source, Path):
        raise FleetTransferError("transfer manifest contains an invalid source file")
    source = item.source
    if (not source.is_absolute() or ".." in source.parts or type(item.size) is not int or item.size < 0
            or not isinstance(item.sha256, str) or _SHA256.fullmatch(item.sha256) is None):
        raise FleetTransferError("transfer manifest contains an invalid source pin")
    if any(ord(char) < 32 or ord(char) in {127, 0x85, 0x2028, 0x2029} for char in str(source)):
        raise FleetTransferError("transfer source path contains a control character")
    _valid_relative(item.remote_path, label="remote file path")

    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except OSError as exc:
            raise FleetTransferError("transfer source is unavailable") from exc
        if current == source:
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise FleetTransferError("transfer source must be an account-owned regular non-symlink file")
        elif not stat.S_ISDIR(info.st_mode):
            raise FleetTransferError("transfer source path contains a symlink or non-directory")
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(opened.st_mode) or opened.st_size != item.size
                    or opened.st_uid != info.st_uid or opened.st_ino != info.st_ino
                    or opened.st_dev != info.st_dev):
                raise FleetTransferError("transfer source changed during validation")
            total = 0
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
            after = os.fstat(stream.fileno())
            if (total != item.size or after.st_size != opened.st_size
                    or after.st_mtime_ns != opened.st_mtime_ns or after.st_ino != opened.st_ino):
                raise FleetTransferError("transfer source changed during validation")
    except FleetTransferError:
        raise
    except OSError as exc:
        raise FleetTransferError("transfer source is unavailable") from exc
    if digest.hexdigest() != item.sha256:
        raise FleetTransferError("transfer source SHA-256 does not match its manifest")
    return _SourceSnapshot(source, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _assert_source_unchanged(snapshot: _SourceSnapshot) -> None:
    try:
        info = snapshot.source.lstat()
    except OSError as exc:
        raise FleetTransferError("transfer source changed during upload") from exc
    if (not stat.S_ISREG(info.st_mode) or info.st_dev != snapshot.device or info.st_ino != snapshot.inode
            or info.st_size != snapshot.size or info.st_mtime_ns != snapshot.mtime_ns):
        raise FleetTransferError("transfer source changed during upload")


def _prefix_sha(snapshot: _SourceSnapshot, size: int) -> str:
    _assert_source_unchanged(snapshot)
    if not 0 <= size <= snapshot.size:
        raise FleetTransferError("remote partial size is outside the source file")
    digest = hashlib.sha256()
    try:
        descriptor = os.open(snapshot.source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            remaining = size
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise FleetTransferError("transfer source changed during resume validation")
                digest.update(chunk)
                remaining -= len(chunk)
    except FleetTransferError:
        raise
    except OSError as exc:
        raise FleetTransferError("transfer source changed during resume validation") from exc
    _assert_source_unchanged(snapshot)
    return digest.hexdigest()


def _manifest_sha(files: Sequence[TransferFile]) -> str:
    items = sorted(({"remote_path": item.remote_path, "size": item.size, "sha256": item.sha256}
                    for item in files), key=lambda value: value["remote_path"])
    body = json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _remote_payload(
    node: FleetNodeView, remote_dir: str, files: Sequence[TransferFile], operation_id: str,
) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "account": {"uid": node.account.uid, "gid": node.account.gid, "home": node.account.home},
        "remote_dir": remote_dir,
        "operation_id": operation_id,
        "manifest_sha256": _manifest_sha(files),
        "files": [{"remote_path": item.remote_path, "size": item.size, "sha256": item.sha256}
                  for item in files],
    }


def _call_remote(
    node: FleetNodeView, settings: FleetSettings, action: str, payload: dict[str, Any], *, timeout: float = 60,
) -> dict[str, Any]:
    try:
        helper_source = _REMOTE_HELPER.read_text(encoding="utf-8")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        result = run_admin_ssh(
            node, ["python3", "-B", "-c", helper_source, action, encoded],
            state_dir=Path(settings.state_dir).expanduser(),
            identity_file=Path(settings.ssh_admin_identity_file).expanduser(),
            timeout=timeout, capture_output=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise FleetTransferError("transfer_timeout") from exc
    except FleetSSHError as exc:
        if isinstance(exc.__cause__, subprocess.TimeoutExpired):
            raise FleetTransferError("transfer_timeout") from exc
        raise FleetTransferError("Fleet node SSH transfer check failed") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise FleetTransferError("Fleet node SSH transfer check failed") from exc
    stdout = getattr(result, "stdout", None)
    if getattr(result, "returncode", 0) != 0 or not isinstance(stdout, str) or len(stdout) > 2_000_000:
        raise FleetTransferError("Fleet node returned an invalid transfer response")
    try:
        value = json.loads(stdout)
    except (TypeError, ValueError) as exc:
        raise FleetTransferError("Fleet node returned an invalid transfer response") from exc
    if not isinstance(value, dict) or type(value.get("ok")) is not bool:
        raise FleetTransferError("Fleet node returned an invalid transfer response")
    if value["ok"] is not True:
        code = value.get("code")
        if not isinstance(code, str) or re.fullmatch(r"[a-z0-9_]{1,80}", code) is None:
            code = "remote_refused"
        raise FleetTransferError(f"Fleet node refused transfer ({code})")
    return value


def _sftp_argv(node: FleetNodeView, settings: FleetSettings) -> list[str]:
    try:
        ssh_argv = admin_ssh_argv(
            node, None,
            state_dir=Path(settings.state_dir).expanduser(),
            identity_file=Path(settings.ssh_admin_identity_file).expanduser(),
        )
    except FleetSSHError as exc:
        raise FleetTransferError("pinned Fleet SSH configuration is unavailable") from exc
    try:
        config_path = ssh_argv[ssh_argv.index("-F") + 1]
        alias = ssh_argv[-1]
    except (ValueError, IndexError) as exc:
        raise FleetTransferError("pinned Fleet SSH configuration is invalid") from exc
    executable = shutil.which("sftp") or "/usr/bin/sftp"
    return [executable, "-F", config_path, "-B", str(_SFTP_BUFFER_SIZE),
            "-R", str(_SFTP_REQUESTS), "-q", "-b", "-", alias]


def _sftp_quote(value: str) -> str:
    if any(ord(char) < 32 or ord(char) in {127, 0x85, 0x2028, 0x2029} for char in value):
        raise FleetTransferError("SFTP path contains a control character")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    for character in "*?[]":
        escaped = escaped.replace(character, "\\" + character)
    return '"' + escaped + '"'


def _batch(
    node: FleetNodeView, remote_dir: str, files: Sequence[TransferFile],
    snapshots: dict[str, _SourceSnapshot], prepared: dict[str, Any],
) -> bytes:
    entries = prepared.get("files")
    if not isinstance(entries, list) or len(entries) != len(files):
        raise FleetTransferError("Fleet node returned an invalid transfer plan")
    by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("remote_path"), str):
            raise FleetTransferError("Fleet node returned an invalid transfer plan")
        if entry["remote_path"] in by_path:
            raise FleetTransferError("Fleet node returned a duplicate transfer plan entry")
        by_path[entry["remote_path"]] = entry
    home = PurePosixPath(node.account.home)
    lines = []
    for item in files:
        entry = by_path.get(item.remote_path)
        if entry is None or entry.get("size") != item.size or entry.get("sha256") != item.sha256:
            raise FleetTransferError("Fleet node returned an invalid transfer plan")
        final = home.joinpath(*_valid_relative(remote_dir, label="remote directory"),
                              *_valid_relative(item.remote_path, label="remote file path"))
        expected_partial = final.with_name(f".{final.name}.{item.sha256}.partial").as_posix()
        if (entry.get("final_path") != final.as_posix()
                or entry.get("partial_path") != expected_partial):
            raise FleetTransferError("Fleet node returned an invalid transfer plan")
        if entry.get("state") == "skipped":
            continue
        if entry.get("state") != "upload":
            raise FleetTransferError("Fleet node returned an invalid transfer plan")
        offset = entry.get("offset")
        remote_partial = entry.get("partial_path")
        prefix_hash = entry.get("prefix_sha256")
        if (type(offset) is not int or offset < 0 or offset > item.size
                or remote_partial != expected_partial
                or not isinstance(prefix_hash, str) or _SHA256.fullmatch(prefix_hash) is None):
            raise FleetTransferError("Fleet node returned an invalid partial-file receipt")
        snapshot = snapshots[item.remote_path]
        _assert_source_unchanged(snapshot)
        if _prefix_sha(snapshot, offset) != prefix_hash:
            raise FleetTransferError("remote partial does not match the source prefix")
        lines.append(f"reput {_sftp_quote(str(item.source))} {_sftp_quote(remote_partial)}")
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _run_sftp(argv: list[str], batch: bytes) -> subprocess.Popen:
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if process.stdin is None:
            _stop_sftp(process)
            raise FleetTransferError("OpenSSH SFTP did not accept a batch")
        process.stdin.write(batch)
        process.stdin.close()
        return process
    except FleetTransferError:
        if process is not None and process.poll() is None:
            _stop_sftp(process)
        raise
    except (OSError, subprocess.SubprocessError, BrokenPipeError) as exc:
        if process is not None and process.poll() is None:
            _stop_sftp(process)
        raise FleetTransferError("OpenSSH SFTP transfer could not be started") from exc


def _stop_sftp(process: subprocess.Popen) -> None:
    """Stop and reap only the local SFTP child created for this transfer."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    except TypeError:
        # Small injected test doubles may not accept Popen's timeout keyword.
        process.wait()


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None, *, remote_path: str,
    done: int, total: int, status: str,
) -> None:
    if callback is not None:
        callback({"file": remote_path, "bytes_done": done, "bytes_total": total, "status": status})


def _remote_sizes(value: dict[str, Any], files: Sequence[TransferFile]) -> dict[str, int]:
    sizes = value.get("sizes")
    if not isinstance(sizes, dict):
        raise FleetTransferError("Fleet node returned invalid transfer statistics")
    result: dict[str, int] = {}
    for item in files:
        size = sizes.get(item.remote_path)
        if type(size) is not int or not 0 <= size <= item.size:
            raise FleetTransferError("Fleet node returned invalid transfer statistics")
        result[item.remote_path] = size
    return result


def transfer_files(
    node: FleetNodeView,
    settings: FleetSettings,
    remote_dir: str,
    files: Sequence[TransferFile],
    operation_id: str,
    progress: Callable[[dict[str, Any]], None] | None = None,
    *,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Upload a pinned batch to a private node directory and return its safe receipt.

    The registered node and management SSH identity are reused from Fleet's
    existing pinned OpenSSH configuration. Remote files are verified by size
    and SHA-256 before an atomic rename and a private durable receipt.
    """
    _validate_node(node)
    if not isinstance(settings, FleetSettings):
        raise FleetTransferError("transfer requires validated Fleet settings")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86_400:
        raise FleetTransferError("timeout_seconds must be an integer from 1 through 86400")
    _valid_relative(remote_dir, label="remote directory")
    if not isinstance(operation_id, str) or _OPERATION_ID.fullmatch(operation_id) is None:
        raise FleetTransferError("operation ID is invalid")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)) or not files:
        raise FleetTransferError("transfer requires at least one manifest file")
    normalized = tuple(files)
    names: set[str] = set()
    final_paths: set[str] = set()
    partial_paths: set[str] = set()
    snapshots: dict[str, _SourceSnapshot] = {}
    for item in normalized:
        if not isinstance(item, TransferFile):
            raise FleetTransferError("transfer manifest contains an invalid file entry")
        parts = _valid_relative(item.remote_path, label="remote file path")
        if item.remote_path in names:
            raise FleetTransferError("transfer manifest contains duplicate remote paths")
        names.add(item.remote_path)
        snapshots[item.remote_path] = _validate_source(item)
        if (parts[-1] == ".qwen-fleet-transfer-owner.json"
                or item.remote_path == f".qwen-fleet-transfer-{operation_id}.receipt.json"):
            raise FleetTransferError("transfer manifest collides with a reserved Fleet path")
        relative_parent = PurePosixPath(item.remote_path).parent
        partial_name = f".{parts[-1]}.{item.sha256}.partial"
        partial = (relative_parent / partial_name).as_posix()
        final_paths.add(PurePosixPath(item.remote_path).as_posix())
        partial_paths.add(partial)
    if final_paths & partial_paths:
        raise FleetTransferError("transfer manifest contains conflicting final and partial paths")
    # Source hashing above is local and can be expensive. Bound the remote
    # preparation, SFTP transfer, progress checks, and receipt finalization.
    deadline = time.monotonic() + timeout_seconds

    def call_remote(action: str) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FleetTransferError("transfer_timeout")
        value = _call_remote(node, settings, action, payload, timeout=remaining)
        if time.monotonic() >= deadline:
            raise FleetTransferError("transfer_timeout")
        return value

    payload = _remote_payload(node, remote_dir, normalized, operation_id)
    prepared = call_remote("prepare")
    # A retry with the same operation ID may already have a durable receipt.
    if prepared.get("receipt_exists") is True:
        for snapshot in snapshots.values():
            _assert_source_unchanged(snapshot)
        finalized = call_remote("finalize")
        return _safe_result(finalized, payload, normalized)
    batch = _batch(node, remote_dir, normalized, snapshots, prepared)
    if batch:
        sizes = _remote_sizes(call_remote("progress"), normalized)
        for item in normalized:
            entry = next(entry for entry in prepared["files"] if entry["remote_path"] == item.remote_path)
            if entry.get("state") != "skipped":
                _emit_progress(progress, remote_path=item.remote_path, done=sizes[item.remote_path],
                               total=item.size, status="uploading")
        if time.monotonic() >= deadline:
            raise FleetTransferError("transfer_timeout")
        argv = _sftp_argv(node, settings)
        if time.monotonic() >= deadline:
            raise FleetTransferError("transfer_timeout")
        process = _run_sftp(argv, batch)
        last_poll = time.monotonic()
        try:
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FleetTransferError("transfer_timeout")
                time.sleep(min(0.2, _PROGRESS_INTERVAL, remaining))
                now = time.monotonic()
                if now >= deadline:
                    raise FleetTransferError("transfer_timeout")
                if now - last_poll >= _PROGRESS_INTERVAL:
                    sizes = _remote_sizes(call_remote("progress"), normalized)
                    for item in normalized:
                        entry = next(entry for entry in prepared["files"] if entry["remote_path"] == item.remote_path)
                        if entry.get("state") != "skipped":
                            _emit_progress(progress, remote_path=item.remote_path, done=sizes[item.remote_path],
                                           total=item.size, status="uploading")
                    last_poll = now
        except BaseException:
            _stop_sftp(process)
            raise
        return_code = process.wait()
        if return_code != 0:
            raise FleetTransferError(f"OpenSSH SFTP exited with status {return_code}")
        for snapshot in snapshots.values():
            _assert_source_unchanged(snapshot)
    elif time.monotonic() >= deadline:
        raise FleetTransferError("transfer_timeout")
    finalized = call_remote("finalize")
    for item in normalized:
        _emit_progress(progress, remote_path=item.remote_path, done=item.size, total=item.size,
                       status="skipped" if next(entry for entry in prepared["files"]
                                                if entry["remote_path"] == item.remote_path).get("state") == "skipped"
                       else "complete")
    return _safe_result(finalized, payload, normalized)


def _safe_result(value: dict[str, Any], payload: dict[str, Any], files: Sequence[TransferFile]) -> dict[str, Any]:
    receipt = value.get("receipt")
    if (not isinstance(receipt, dict) or receipt.get("node_id") != payload["node_id"]
            or receipt.get("manifest_sha256") != payload["manifest_sha256"]
            or receipt.get("operation_id") != payload["operation_id"]
            or receipt.get("remote_dir") != payload["remote_dir"]
            or receipt.get("schema") != "qwen-fleet-transfer-receipt/v1"):
        raise FleetTransferError("Fleet node returned an invalid transfer receipt")
    receipt_files = receipt.get("files")
    if not isinstance(receipt_files, list) or len(receipt_files) != len(files):
        raise FleetTransferError("Fleet node returned an invalid transfer receipt")
    recorded = {entry.get("remote_path"): entry for entry in receipt_files if isinstance(entry, dict)}
    if len(recorded) != len(files):
        raise FleetTransferError("Fleet node returned an invalid transfer receipt")
    for item in files:
        entry = recorded.get(item.remote_path)
        if (entry is None or entry.get("size") != item.size or entry.get("sha256") != item.sha256
                or entry.get("status") != "verified"):
            raise FleetTransferError("Fleet node returned an invalid transfer receipt")
    receipt_bytes = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if value.get("receipt_sha256") != receipt_sha:
        raise FleetTransferError("Fleet node returned an invalid transfer receipt")
    return {
        "status": "verified",
        "node_id": payload["node_id"],
        "operation_id": payload["operation_id"],
        "remote_dir": payload["remote_dir"],
        "files": [{"remote_path": item.remote_path, "size": item.size, "sha256": item.sha256,
                   "status": "verified"} for item in files],
        "file_count": len(files),
        "bytes_total": sum(item.size for item in files),
        "receipt_sha256": receipt_sha,
    }
