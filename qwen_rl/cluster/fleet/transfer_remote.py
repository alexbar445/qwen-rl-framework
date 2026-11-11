"""Small stdlib-only remote half of the verified Fleet SFTP transfer protocol.

This module is sent as ``python3 -c`` over the already pinned Fleet SSH route.
It deliberately has no imports from the application package, so registered
nodes do not need a second deployment just to receive files.
"""
from __future__ import annotations

import hashlib
import json
import os
import pwd
import secrets
import stat
import sys
from pathlib import Path
from typing import Any

_OWNER = ".qwen-fleet-transfer-owner.json"
_OWNER_SCHEMA = "qwen-fleet-transfer-owner/v1"
_RECEIPT_SCHEMA = "qwen-fleet-transfer-receipt/v1"


class Refused(Exception):
    """A safe, stable remote refusal code."""


def _relative(value: Any, *, label: str) -> tuple[str, ...]:
    if (not isinstance(value, str) or not value or len(value) > 4096
            or any(ord(char) < 32 or ord(char) in {127, 0x85, 0x2028, 0x2029} for char in value)):
        raise Refused("invalid_relative_path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise Refused("invalid_relative_path")
    if label == "file" and parts[-1] == _OWNER:
        raise Refused("reserved_remote_path")
    return tuple(parts)


def _private_directory(path: Path, uid: int, gid: int, *, create: bool, exact_private: bool = False) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if not create:
            raise Refused("remote_directory_missing")
        try:
            path.mkdir(mode=0o700)
            metadata = path.lstat()
        except OSError as exc:
            raise Refused("remote_directory_create_failed") from exc
    except OSError as exc:
        raise Refused("remote_directory_inspection_failed") from exc
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != uid or metadata.st_gid != gid
            or stat.S_IMODE(metadata.st_mode) & 0o022
            or (exact_private and stat.S_IMODE(metadata.st_mode) != 0o700)):
        raise Refused("remote_directory_ownership_invalid")


def _atomic_create(path: Path, content: bytes, *, error_code: str) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        # link() publishes the complete private file atomically and refuses to
        # replace a path created by another process after our lstat check.
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise Refused(error_code) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _safe_home(payload: dict[str, Any], *, effective_uid: int | None = None,
               passwd_home: str | None = None) -> tuple[Path, int, int]:
    account = payload.get("account")
    if not isinstance(account, dict):
        raise Refused("account_invalid")
    uid, gid, home_value = account.get("uid"), account.get("gid"), account.get("home")
    if type(uid) is not int or uid <= 0 or type(gid) is not int or gid < 0:
        raise Refused("account_invalid")
    if effective_uid is None:
        effective_uid = os.geteuid()
    if effective_uid != uid:
        raise Refused("remote_account_mismatch")
    try:
        if passwd_home is None:
            passwd_home = pwd.getpwuid(uid).pw_dir
    except KeyError as exc:
        raise Refused("registered_home_unavailable") from exc
    if (not isinstance(home_value, str) or home_value != passwd_home or not home_value.startswith("/")
            or home_value == "/" or ".." in Path(home_value).parts):
        raise Refused("registered_home_mismatch")
    home = Path(home_value)
    current = Path(home.anchor)
    for component in home.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise Refused("registered_home_unavailable") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise Refused("registered_home_symlink_or_non_directory")
    metadata = home.lstat()
    if (metadata.st_uid != uid or metadata.st_gid != gid or stat.S_IMODE(metadata.st_mode) & 0o022):
        raise Refused("registered_home_ownership_invalid")
    return home, uid, gid


def _root(payload: dict[str, Any], *, effective_uid: int | None = None,
          passwd_home: str | None = None) -> tuple[Path, int, int, str]:
    home, uid, gid = _safe_home(payload, effective_uid=effective_uid, passwd_home=passwd_home)
    remote_dir = payload.get("remote_dir")
    parts = _relative(remote_dir, label="directory")
    target = home.joinpath(*parts)
    current = home
    for index, component in enumerate(parts):
        current /= component
        _private_directory(current, uid, gid, create=True,
                           exact_private=index == len(parts) - 1)
    marker_value = {
        "schema": _OWNER_SCHEMA,
        "node_id": payload.get("node_id"),
        "uid": uid,
        "manifest_sha256": payload.get("manifest_sha256"),
    }
    marker_bytes = (json.dumps(marker_value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    marker = target / _OWNER
    try:
        marker_info = marker.lstat()
    except FileNotFoundError:
        try:
            entries = list(target.iterdir())
        except OSError as exc:
            raise Refused("remote_directory_inspection_failed") from exc
        if entries:
            raise Refused("remote_directory_unmarked_not_empty")
        _atomic_create(marker, marker_bytes, error_code="owner_marker_create_failed")
        marker_info = marker.lstat()
    except OSError as exc:
        raise Refused("owner_marker_inspection_failed") from exc
    if (not stat.S_ISREG(marker_info.st_mode) or marker_info.st_uid != uid
            or stat.S_IMODE(marker_info.st_mode) != 0o600 or marker_info.st_size > 4096):
        raise Refused("owner_marker_invalid")
    try:
        existing = marker.read_bytes()
    except OSError as exc:
        raise Refused("owner_marker_invalid") from exc
    if existing != marker_bytes:
        raise Refused("owner_marker_identity_mismatch")
    return target, uid, gid, str(remote_dir)


def _ensure_file_parent(root: Path, rel_parts: tuple[str, ...], uid: int, gid: int) -> Path:
    current = root
    for component in rel_parts[:-1]:
        current /= component
        _private_directory(current, uid, gid, create=True, exact_private=True)
    return current / rel_parts[-1]


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise Refused("remote_file_not_private_regular")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(opened.st_mode) or opened.st_ino != info.st_ino
                    or opened.st_dev != info.st_dev or opened.st_uid != os.geteuid()):
                raise Refused("remote_file_changed_during_inspection")
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
            if (after.st_dev != opened.st_dev or after.st_ino != opened.st_ino
                    or after.st_size != opened.st_size or after.st_mtime_ns != opened.st_mtime_ns):
                raise Refused("remote_file_changed_during_inspection")
    except Refused:
        raise
    except FileNotFoundError as exc:
        raise Refused("remote_file_missing") from exc
    except OSError as exc:
        raise Refused("remote_file_inspection_failed") from exc
    return size, digest.hexdigest()


def _validate_files(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise Refused("manifest_invalid")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            raise Refused("manifest_invalid")
        remote_path = item.get("remote_path")
        parts = _relative(remote_path, label="file")
        size, sha = item.get("size"), item.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(sha, str) or len(sha) != 64:
            raise Refused("manifest_invalid")
        if any(char not in "0123456789abcdef" for char in sha):
            raise Refused("manifest_invalid")
        if remote_path in seen:
            raise Refused("manifest_duplicate_path")
        seen.add(remote_path)
        result.append({"remote_path": remote_path, "parts": parts, "size": size, "sha256": sha})
    return result


def _check_operation(payload: dict[str, Any]) -> str:
    operation_id = payload.get("operation_id")
    if (not isinstance(operation_id, str) or not operation_id or len(operation_id) > 100
            or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in operation_id)):
        raise Refused("operation_id_invalid")
    if (not isinstance(payload.get("node_id"), str) or not payload["node_id"]
            or not isinstance(payload.get("manifest_sha256"), str)
            or len(payload["manifest_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in payload["manifest_sha256"])):
        raise Refused("request_identity_invalid")
    return operation_id


def prepare(payload: dict[str, Any], *, effective_uid: int | None = None,
            passwd_home: str | None = None) -> dict[str, Any]:
    operation_id = _check_operation(payload)
    files = _validate_files(payload)
    root, uid, gid, remote_dir = _root(payload, effective_uid=effective_uid, passwd_home=passwd_home)
    prepared = []
    for item in files:
        final_path = _ensure_file_parent(root, item["parts"], uid, gid)
        partial_path = final_path.with_name(f".{final_path.name}.{item['sha256']}.partial")
        try:
            final_info = final_path.lstat()
        except FileNotFoundError:
            final_info = None
        except OSError as exc:
            raise Refused("remote_file_inspection_failed") from exc
        if final_info is not None:
            if (not stat.S_ISREG(final_info.st_mode) or final_info.st_uid != uid
                    or stat.S_IMODE(final_info.st_mode) & 0o077):
                raise Refused("remote_final_path_is_foreign")
            actual_size, actual_sha = _hash_file(final_path)
            if actual_size != item["size"] or actual_sha != item["sha256"]:
                raise Refused("remote_final_path_is_foreign")
            prepared.append({"remote_path": item["remote_path"], "state": "skipped",
                             "size": item["size"], "sha256": item["sha256"],
                             "partial_path": str(partial_path), "final_path": str(final_path), "offset": item["size"]})
            continue
        try:
            partial_info = partial_path.lstat()
        except FileNotFoundError:
            try:
                descriptor = os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
                os.close(descriptor)
                partial_info = partial_path.lstat()
            except OSError as exc:
                raise Refused("remote_partial_create_failed") from exc
        except OSError as exc:
            raise Refused("remote_partial_inspection_failed") from exc
        if (not stat.S_ISREG(partial_info.st_mode) or partial_info.st_uid != uid
                or stat.S_IMODE(partial_info.st_mode) != 0o600
                or partial_info.st_size > item["size"]):
            raise Refused("remote_partial_is_foreign")
        prefix_size, prefix_sha = _hash_file(partial_path)
        if prefix_size != partial_info.st_size:
            raise Refused("remote_partial_changed_during_inspection")
        prepared.append({"remote_path": item["remote_path"], "state": "upload",
                         "size": item["size"], "sha256": item["sha256"],
                         "partial_path": str(partial_path), "final_path": str(final_path),
                         "offset": prefix_size, "prefix_sha256": prefix_sha})
    receipt_path = root / f".qwen-fleet-transfer-{operation_id}.receipt.json"
    try:
        receipt_info = receipt_path.lstat()
    except FileNotFoundError:
        receipt_info = None
    except OSError as exc:
        raise Refused("receipt_inspection_failed") from exc
    if receipt_info is not None and (not stat.S_ISREG(receipt_info.st_mode) or receipt_info.st_uid != uid
                                     or stat.S_IMODE(receipt_info.st_mode) != 0o600):
        raise Refused("receipt_path_is_foreign")
    return {"ok": True, "remote_dir": remote_dir, "files": prepared,
            "receipt_path": str(receipt_path), "receipt_exists": receipt_info is not None}


def progress(payload: dict[str, Any], *, effective_uid: int | None = None,
             passwd_home: str | None = None) -> dict[str, Any]:
    _check_operation(payload)
    files = _validate_files(payload)
    root, uid, gid, _ = _root(payload, effective_uid=effective_uid, passwd_home=passwd_home)
    sizes: dict[str, int] = {}
    for item in files:
        final_path = _ensure_file_parent(root, item["parts"], uid, gid)
        try:
            final_info = final_path.lstat()
        except FileNotFoundError:
            final_info = None
        if final_info is not None:
            if not stat.S_ISREG(final_info.st_mode) or final_info.st_uid != uid:
                raise Refused("remote_final_path_is_foreign")
            sizes[item["remote_path"]] = item["size"]
            continue
        partial_path = final_path.with_name(f".{final_path.name}.{item['sha256']}.partial")
        try:
            info = partial_path.lstat()
        except FileNotFoundError:
            sizes[item["remote_path"]] = 0
            continue
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size > item["size"]):
            raise Refused("remote_partial_is_foreign")
        sizes[item["remote_path"]] = info.st_size
    return {"ok": True, "sizes": sizes}


def _atomic_receipt(path: Path, content: bytes, uid: int) -> None:
    if os.geteuid() != uid:
        raise Refused("remote_account_mismatch")
    _atomic_create(path, content, error_code="receipt_write_failed")


def finalize(payload: dict[str, Any], *, effective_uid: int | None = None,
             passwd_home: str | None = None) -> dict[str, Any]:
    operation_id = _check_operation(payload)
    files = _validate_files(payload)
    root, uid, gid, remote_dir = _root(payload, effective_uid=effective_uid, passwd_home=passwd_home)
    receipt_path = root / f".qwen-fleet-transfer-{operation_id}.receipt.json"
    try:
        existing = receipt_path.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise Refused("receipt_inspection_failed") from exc
    if existing is not None:
        if (not stat.S_ISREG(existing.st_mode) or existing.st_uid != uid
                or stat.S_IMODE(existing.st_mode) != 0o600 or existing.st_size > 1_000_000):
            raise Refused("receipt_path_is_foreign")
        try:
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
        except (OSError, UnicodeError, ValueError) as exc:
            raise Refused("receipt_path_is_foreign") from exc
        if (not isinstance(receipt, dict) or receipt.get("operation_id") != operation_id
                or receipt.get("manifest_sha256") != payload["manifest_sha256"]
                or receipt.get("node_id") != payload["node_id"]):
            raise Refused("receipt_path_is_foreign")
        canonical = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        if receipt_bytes != canonical:
            raise Refused("receipt_path_is_foreign")
        return {"ok": True, "remote_dir": remote_dir, "receipt": receipt,
                "receipt_path": str(receipt_path), "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "already_complete": True}
    completed = []
    for item in files:
        final_path = _ensure_file_parent(root, item["parts"], uid, gid)
        partial_path = final_path.with_name(f".{final_path.name}.{item['sha256']}.partial")
        try:
            final_info = final_path.lstat()
        except FileNotFoundError:
            final_info = None
        if final_info is not None:
            if not stat.S_ISREG(final_info.st_mode) or final_info.st_uid != uid:
                raise Refused("remote_final_path_is_foreign")
            actual_size, actual_sha = _hash_file(final_path)
            if actual_size != item["size"] or actual_sha != item["sha256"]:
                raise Refused("remote_final_path_is_foreign")
        else:
            actual_size, actual_sha = _hash_file(partial_path)
            if actual_size != item["size"] or actual_sha != item["sha256"]:
                raise Refused("remote_file_verification_failed")
            try:
                os.replace(partial_path, final_path)
                os.chmod(final_path, 0o600, follow_symlinks=False)
            except OSError as exc:
                raise Refused("remote_atomic_rename_failed") from exc
        completed.append({"remote_path": item["remote_path"], "size": item["size"],
                          "sha256": item["sha256"], "status": "verified"})
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "node_id": payload["node_id"],
        "operation_id": operation_id,
        "manifest_sha256": payload["manifest_sha256"],
        "remote_dir": remote_dir,
        "files": completed,
    }
    content = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _atomic_receipt(receipt_path, content, uid)
    return {"ok": True, "remote_dir": remote_dir, "receipt": receipt,
            "receipt_path": str(receipt_path), "receipt_sha256": hashlib.sha256(content).hexdigest(),
            "already_complete": False}


def handle(action: str, payload: dict[str, Any], *, effective_uid: int | None = None,
           passwd_home: str | None = None) -> dict[str, Any]:
    """Run one bounded action; identity overrides exist only for focused unit tests."""
    actions = {"prepare": prepare, "progress": progress, "finalize": finalize}
    function = actions.get(action)
    if function is None or not isinstance(payload, dict):
        return {"ok": False, "code": "invalid_request"}
    try:
        return function(payload, effective_uid=effective_uid, passwd_home=passwd_home)
    except Refused as exc:
        return {"ok": False, "code": str(exc)}
    except Exception:  # noqa: BLE001 - remote protocol errors must never expose payload or credential values.
        return {"ok": False, "code": "remote_error"}


def main() -> int:
    try:
        action = sys.argv[1]
        payload = json.loads(sys.argv[2])
        result = handle(action, payload)
    except (IndexError, TypeError, ValueError):
        result = {"ok": False, "code": "invalid_request"}
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
