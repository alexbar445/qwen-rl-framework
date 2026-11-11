"""Verify, extract, and install a trusted Learner bundle on its target host.

The standalone HTTPS starter is stdlib-only and verifies the outer bundle SHA
before it imports this package module. This module then validates every member
against the bundle manifest, remaps host-only profile paths, and delegates host
package/image/service work to the L4 callback.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .bundle import (
    BUNDLE_SCHEMA,
    DEFAULT_INSTALL_ROOT,
    IMAGE_MEMBER,
    MANIFEST_NAME,
    LearnerBundleError,
    _json_bytes,
    _member_role,
    _profile_path_map,
    _safe_relative,
    _set_pointer,
    _sha_size,
    _unique_object,
)

INSTALL_MARKER = ".qwen-learner-install.json"
INSTALL_MARKER_SCHEMA = "qwen-rl-learner-install/v1"
MAX_BUNDLE_BYTES = 8 * 1024**4
MAX_MEMBER_BYTES = 4 * 1024**4
MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_TAR_HEADER_OVERHEAD = 256 * 1024**2
MAX_ARCHIVE_MEMBERS = 100_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_FILE_MODES = {0o644, 0o755}


class LearnerBootstrapError(ValueError):
    """A pinned Learner bundle or destination violates the install contract."""


@dataclass(frozen=True, slots=True)
class LearnerInstallReceipt:
    install_root: Path
    deployment_id: str
    bundle_sha256: str
    image_id: str
    profile_sha256: str
    state: str
    host_state: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "install_root": str(self.install_root),
            "deployment_id": self.deployment_id,
            "bundle_sha256": self.bundle_sha256,
            "image_id": self.image_id,
            "profile_sha256": self.profile_sha256,
            "state": self.state,
            "host_state": self.host_state,
        }


def _strict_json(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite JSON")),
        )
    except Exception as exc:
        raise LearnerBootstrapError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise LearnerBootstrapError(f"{label} root must be a JSON object")
    return value


def _open_bundle(path: str | Path):
    bundle_path = Path(path).expanduser()
    try:
        metadata = bundle_path.lstat()
    except OSError as exc:
        raise LearnerBootstrapError("Learner bundle cannot be inspected") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise LearnerBootstrapError("Learner bundle must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_BUNDLE_BYTES:
        raise LearnerBootstrapError("Learner bundle exceeds the compressed size limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(bundle_path, flags)
    except OSError as exc:
        raise LearnerBootstrapError("Learner bundle cannot be safely opened") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != metadata.st_size:
            raise LearnerBootstrapError("Learner bundle changed while it was being opened")
        return os.fdopen(descriptor, "rb"), opened.st_size
    except BaseException:
        os.close(descriptor)
        raise


def _hash_stream(stream) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _manifest_structure(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    expected_keys = {
        "schema_version", "deployment_id", "default_install_root", "image_id",
        "image_archive", "profile", "path_map", "directories", "files",
        "unpacked_size_bytes", "includes_full_state",
    }
    if set(manifest) != expected_keys or manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise LearnerBootstrapError("Learner bundle manifest does not match its strict schema")
    if not isinstance(manifest["deployment_id"], str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,63}", manifest["deployment_id"]):
        raise LearnerBootstrapError("Learner bundle deployment_id is invalid")
    if not isinstance(manifest["image_id"], str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", manifest["image_id"]):
        raise LearnerBootstrapError("Learner bundle image ID is invalid")
    if not isinstance(manifest["default_install_root"], str):
        raise LearnerBootstrapError("Learner bundle default install root is invalid")
    root = Path(manifest["default_install_root"])
    if (not root.is_absolute() or root == Path("/") or ".." in root.parts
            or any(ord(char) < 32 for char in str(root))):
        raise LearnerBootstrapError("Learner bundle default install root is invalid")
    if type(manifest["includes_full_state"]) is not bool:
        raise LearnerBootstrapError("Learner bundle state inclusion flag is invalid")
    if (isinstance(manifest["unpacked_size_bytes"], bool)
            or not isinstance(manifest["unpacked_size_bytes"], int)
            or not 0 <= manifest["unpacked_size_bytes"] <= MAX_BUNDLE_BYTES):
        raise LearnerBootstrapError("Learner bundle unpacked size is invalid")

    directories = manifest["directories"]
    if not isinstance(directories, list) or len(directories) > MAX_ARCHIVE_MEMBERS:
        raise LearnerBootstrapError("Learner bundle directory list is invalid")
    directory_set: set[str] = set()
    for value in directories:
        try:
            relative = _safe_relative(value).as_posix()
        except Exception as exc:
            raise LearnerBootstrapError("Learner bundle contains an unsafe directory path") from exc
        if relative != value or relative in directory_set:
            raise LearnerBootstrapError("Learner bundle directory paths are duplicated or noncanonical")
        directory_set.add(relative)
    if directories != sorted(directory_set, key=lambda name: (name.count("/"), name)):
        raise LearnerBootstrapError("Learner bundle directory list is not in canonical order")

    files = manifest["files"]
    if not isinstance(files, list) or len(files) > MAX_ARCHIVE_MEMBERS:
        raise LearnerBootstrapError("Learner bundle file list is invalid")
    files_by_path: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    allowed_roles = {
        "control", "profile", "image", "runtime_config", "model", "dataset",
        "state", "resource", "credential",
    }
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "role", "sha256", "size_bytes", "mode"}:
            raise LearnerBootstrapError("Learner bundle file manifest entry is malformed")
        try:
            relative = _safe_relative(row["path"]).as_posix()
        except Exception as exc:
            raise LearnerBootstrapError("Learner bundle contains an unsafe file path") from exc
        if relative != row["path"] or relative in files_by_path or relative in directory_set:
            raise LearnerBootstrapError("Learner bundle file paths are duplicated or noncanonical")
        if (not isinstance(row["role"], str) or row["role"] not in allowed_roles
                or row["role"] != _member_role(relative)):
            raise LearnerBootstrapError("Learner bundle file role does not match its path")
        if not isinstance(row["sha256"], str) or not _SHA256.fullmatch(row["sha256"]):
            raise LearnerBootstrapError("Learner bundle file SHA-256 is invalid")
        if (isinstance(row["size_bytes"], bool) or not isinstance(row["size_bytes"], int)
                or row["size_bytes"] < 0 or row["size_bytes"] > MAX_BUNDLE_BYTES):
            raise LearnerBootstrapError("Learner bundle file size is invalid")
        if isinstance(row["mode"], bool) or row["mode"] not in {0o600, 0o644, 0o755}:
            raise LearnerBootstrapError("Learner bundle file mode is not permitted")
        if row["role"] in {"profile", "image", "runtime_config", "model", "dataset", "state", "resource", "credential"}:
            if row["mode"] != 0o600:
                raise LearnerBootstrapError("private Learner assets must be bundled with mode 0600")
        elif row["role"] == "control" and row["mode"] not in _CONTROL_FILE_MODES:
            raise LearnerBootstrapError("control source file mode is invalid")
        files_by_path[relative] = row
        total_bytes += row["size_bytes"]
    if list(files_by_path) != sorted(files_by_path):
        raise LearnerBootstrapError("Learner bundle file list is not in canonical order")
    if len(files_by_path) + len(directory_set) + 1 > MAX_ARCHIVE_MEMBERS:
        raise LearnerBootstrapError("Learner bundle contains too many archive members")
    if total_bytes != manifest["unpacked_size_bytes"]:
        raise LearnerBootstrapError("Learner bundle unpacked size differs from its file table")
    if total_bytes > MAX_BUNDLE_BYTES:
        raise LearnerBootstrapError("Learner bundle exceeds the unpacked size limit")
    for relative in set(files_by_path) | directory_set:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() not in {".", ""}:
            if parent.as_posix() not in directory_set:
                raise LearnerBootstrapError("Learner bundle omits a parent directory from its manifest")
            parent = parent.parent

    image = manifest["image_archive"]
    if not isinstance(image, dict) or set(image) != {"path", "sha256", "size_bytes"}:
        raise LearnerBootstrapError("Learner bundle image descriptor is invalid")
    image_entry = files_by_path.get(IMAGE_MEMBER)
    if (image["path"] != IMAGE_MEMBER or image_entry is None
            or not isinstance(image["sha256"], str) or not _SHA256.fullmatch(image["sha256"])
            or isinstance(image["size_bytes"], bool) or not isinstance(image["size_bytes"], int)
            or image["sha256"] != image_entry["sha256"]
            or image["size_bytes"] != image_entry["size_bytes"]
            or image_entry["role"] != "image"):
        raise LearnerBootstrapError("Learner bundle image archive differs from its content manifest")
    profile = manifest["profile"]
    profile_entry = files_by_path.get("learner-deployment.json")
    if (not isinstance(profile, dict) or set(profile) != {"path", "sha256"}
            or profile["path"] != "learner-deployment.json"
            or not isinstance(profile["sha256"], str) or not _SHA256.fullmatch(profile["sha256"])
            or profile_entry is None or profile["sha256"] != profile_entry["sha256"]
            or profile_entry["role"] != "profile"):
        raise LearnerBootstrapError("Learner bundle deployment profile differs from its content manifest")
    path_map = manifest["path_map"]
    if not isinstance(path_map, list) or any(
            not isinstance(item, dict) or set(item) != {"pointer", "target"} for item in path_map):
        raise LearnerBootstrapError("Learner bundle profile path map is invalid")
    targets = set()
    pointers = set()
    for item in path_map:
        if not isinstance(item["pointer"], str) or not item["pointer"].startswith("/"):
            raise LearnerBootstrapError("Learner bundle path map pointer is invalid")
        try:
            target = _safe_relative(item["target"]).as_posix()
        except Exception as exc:
            raise LearnerBootstrapError("Learner bundle path map contains an unsafe target") from exc
        if item["pointer"] in pointers or target in targets:
            raise LearnerBootstrapError("Learner bundle path map contains duplicates")
        pointers.add(item["pointer"])
        targets.add(target)
    return files_by_path, image_entry, directory_set


def _read_manifest(stream) -> tuple[dict[str, Any], bytes]:
    """Read just the first bounded manifest member from the gzip stream.

    A Learner bundle may contain a multi-gigabyte model and image. Do not call
    TarFile.getmembers() here: installation validates and hashes payloads in
    one streaming extraction pass after this small header is parsed.
    """
    stream.seek(0)
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            member = archive.next()
            if member is None or member.name != MANIFEST_NAME:
                raise LearnerBootstrapError("Learner bundle manifest must be the first archive member")
            if (not member.isfile() or member.issparse() or member.size > MAX_MANIFEST_BYTES
                    or member.linkname or member.pax_headers):
                raise LearnerBootstrapError("Learner bundle manifest member is invalid")
            manifest_stream = archive.extractfile(member)
            if manifest_stream is None:
                raise LearnerBootstrapError("Learner bundle manifest cannot be read")
            with manifest_stream:
                manifest_bytes = manifest_stream.read(MAX_MANIFEST_BYTES + 1)
            if len(manifest_bytes) != member.size or len(manifest_bytes) > MAX_MANIFEST_BYTES:
                raise LearnerBootstrapError("Learner bundle manifest size is invalid")
    except LearnerBootstrapError:
        raise
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise LearnerBootstrapError("Learner bundle is not a valid gzip tar archive") from exc
    stream.seek(0)
    return _strict_json(manifest_bytes, label="Learner bundle manifest"), manifest_bytes


def _write_extracted_file(destination: Path, member: tarfile.TarInfo, entry: dict[str, Any], source) -> None:
    target = destination.joinpath(*PurePosixPath(member.name).parts)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise LearnerBootstrapError("Learner archive path has an unsafe parent")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, entry["mode"])
    except OSError as exc:
        raise LearnerBootstrapError("Learner archive could not create a safe output file") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            while chunk := source.read(8 * 1024 * 1024):
                total += len(chunk)
                if total > entry["size_bytes"] or total > MAX_MEMBER_BYTES:
                    raise LearnerBootstrapError("Learner archive member exceeded its declared size")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total != entry["size_bytes"] or not hmac.compare_digest(digest.hexdigest(), entry["sha256"]):
            raise LearnerBootstrapError("Learner archive member failed its size or SHA-256 check")
        os.chmod(target, entry["mode"])
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def extract_verified_bundle(
    bundle_path: str | Path,
    expected_sha256: str,
    destination: str | Path,
    *,
    expected_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Verify and extract manifest-declared regular files into a new directory."""
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise LearnerBootstrapError("expected bundle SHA-256 is invalid")
    target = Path(destination).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise LearnerBootstrapError("bundle extraction destination already exists")
    try:
        stream, file_size = _open_bundle(bundle_path)
    except LearnerBundleError as exc:
        raise LearnerBootstrapError(str(exc)) from exc
    with stream:
        if expected_size_bytes is not None and (
            isinstance(expected_size_bytes, bool) or not isinstance(expected_size_bytes, int)
            or expected_size_bytes != file_size
        ):
            raise LearnerBootstrapError("Learner bundle compressed size differs from the expected size")
        digest = _hash_stream(stream)
        if not hmac.compare_digest(digest, expected_sha256):
            raise LearnerBootstrapError("Learner bundle SHA-256 does not match the pinned value")
        manifest, manifest_bytes = _read_manifest(stream)
        files, _image, directories = _manifest_structure(manifest)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.mkdir(mode=0o700)
        try:
            for relative in sorted(directories, key=lambda name: (name.count("/"), name)):
                path = target.joinpath(*PurePosixPath(relative).parts)
                path.mkdir(mode=0o700)
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                names: set[str] = set()
                total_header_bytes = 0
                while True:
                    member = archive.next()
                    if member is None:
                        break
                    if len(names) >= MAX_ARCHIVE_MEMBERS + 1:
                        raise LearnerBootstrapError("Learner bundle contains too many archive members")
                    try:
                        canonical_name = _safe_relative(member.name).as_posix()
                    except Exception as exc:
                        raise LearnerBootstrapError("Learner archive contains an unsafe member path") from exc
                    if member.name != canonical_name:
                        raise LearnerBootstrapError("Learner archive contains a noncanonical member path")
                    if member.name in names:
                        raise LearnerBootstrapError("Learner archive contains a duplicate member")
                    names.add(member.name)
                    if member.name == MANIFEST_NAME:
                        if (len(names) != 1 or not member.isfile() or member.issparse()
                                or member.size != len(manifest_bytes) or member.linkname
                                or member.pax_headers):
                            raise LearnerBootstrapError("Learner manifest archive header changed")
                        source = archive.extractfile(member)
                        if source is None:
                            raise LearnerBootstrapError("Learner manifest cannot be read")
                        with source:
                            data = source.read(MAX_MANIFEST_BYTES + 1)
                        if data != manifest_bytes:
                            raise LearnerBootstrapError("Learner bundle manifest changed between reads")
                        output = target / MANIFEST_NAME
                        descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                        with os.fdopen(descriptor, "wb") as destination_stream:
                            destination_stream.write(data)
                            destination_stream.flush()
                            os.fsync(destination_stream.fileno())
                        continue
                    entry = files.get(member.name)
                    if entry is None:
                        if (member.name not in directories or not member.isdir() or member.size != 0
                                or member.linkname or member.pax_headers):
                            raise LearnerBootstrapError("Learner archive member is not declared")
                        continue
                    if (not member.isfile() or member.issparse() or member.linkname or member.pax_headers
                            or member.size != entry["size_bytes"]):
                        raise LearnerBootstrapError("Learner archive file type or size differs from its manifest")
                    total_header_bytes += member.size
                    if total_header_bytes > manifest["unpacked_size_bytes"]:
                        raise LearnerBootstrapError("Learner archive members exceed the declared unpacked size")
                    source = archive.extractfile(member)
                    if source is None:
                        raise LearnerBootstrapError("Learner archive file cannot be read")
                    with source:
                        _write_extracted_file(target, member, entry, source)
                    if getattr(archive.fileobj, "tell", lambda: 0)() > MAX_BUNDLE_BYTES + MAX_TAR_HEADER_OVERHEAD:
                        raise LearnerBootstrapError("Learner bundle expanded beyond its size limit")
                if names != {MANIFEST_NAME, *files.keys(), *directories}:
                    raise LearnerBootstrapError("Learner archive members differ from the manifest")
            _fsync_tree_dirs(target)
        except BaseException:
            shutil.rmtree(target, ignore_errors=True)
            raise
    return manifest


def _fsync_tree_dirs(root: Path) -> None:
    for base, _dirs, _files in os.walk(root, topdown=False, followlinks=False):
        descriptor = os.open(base, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_private_json(path: Path, value: dict[str, Any]) -> bytes:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.parent / f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    descriptor = os.open(
        temp,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)
    return payload


def _remapped_profile(profile: dict[str, Any], manifest: dict[str, Any], final_root: Path) -> dict[str, Any]:
    expected_map = _profile_path_map(profile)
    if manifest["path_map"] != expected_map:
        raise LearnerBootstrapError("Learner bundle path map does not match its strict profile")
    default_final = Path(manifest["default_install_root"]) / manifest["deployment_id"]
    for item in expected_map:
        target = _safe_relative(item["target"]).as_posix()
        expected_default = str(default_final.joinpath(*PurePosixPath(target).parts))
        current = _get_pointer(profile, item["pointer"])
        if current != expected_default:
            raise LearnerBootstrapError("Learner bundle profile host path differs from its path map")
    result = json.loads(json.dumps(profile, ensure_ascii=False, allow_nan=False))
    prior_container_paths = _container_paths(result)
    for item in expected_map:
        target = _safe_relative(item["target"]).as_posix()
        _set_pointer(result, item["pointer"], str(final_root.joinpath(*PurePosixPath(target).parts)))
    if _container_paths(result) != prior_container_paths:
        raise LearnerBootstrapError("bundle installation changed a profile container_path")
    return result


def _get_pointer(document: dict[str, Any], pointer: str) -> Any:
    current: Any = document
    for part in pointer.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _container_paths(profile: dict[str, Any]) -> tuple[str, ...]:
    result = []
    for name in sorted(profile["mounts"]):
        result.append(profile["mounts"][name]["container_path"])
    for name in sorted(profile["credentials"]):
        result.append(profile["credentials"][name]["container_path"])
    for resource in profile.get("resources", []):
        result.append(resource["container_path"])
    return tuple(result)


def _install_base(path: str | Path) -> Path:
    base = Path(path).expanduser()
    if not base.is_absolute() or base == Path("/") or ".." in base.parts:
        raise LearnerBootstrapError("install_root must be an absolute non-root path")
    current = Path(base.anchor)
    for part in base.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise LearnerBootstrapError("Learner install root cannot be inspected") from exc
        if stat.S_ISLNK(info.st_mode):
            raise LearnerBootstrapError("Learner install root cannot traverse symlinks")
        if current == base and not stat.S_ISDIR(info.st_mode):
            raise LearnerBootstrapError("Learner install root must be a directory")
    base.mkdir(parents=True, mode=0o700, exist_ok=True)
    metadata = base.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077):
        raise LearnerBootstrapError("Learner install root must be a private owned directory")
    return base


@contextmanager
def _deployment_install_lock(base: Path, deployment_id: str):
    """Serialize one deployment ID; kernel locks release automatically on exit."""
    lock_path = base / f".{deployment_id}.install-lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise LearnerBootstrapError("Learner deployment install lock cannot be safely opened") from exc
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1):
            raise LearnerBootstrapError("Learner deployment lock has unsafe owner or mode")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LearnerBootstrapError("another Learner install for this deployment ID is active") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)
        try:
            directory_fd = os.open(base, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass


def _marker_data(final_root: Path, manifest: dict[str, Any], bundle_sha256: str) -> dict[str, Any]:
    profile_path = final_root / manifest["profile"]["path"]
    profile_sha, profile_size = _sha_size(profile_path)
    manifest_sha, manifest_size = _sha_size(final_root / MANIFEST_NAME)
    return {
        "schema_version": INSTALL_MARKER_SCHEMA,
        "deployment_id": manifest["deployment_id"],
        "bundle_sha256": bundle_sha256,
        "image_id": manifest["image_id"],
        "manifest_sha256": manifest_sha,
        "manifest_size_bytes": manifest_size,
        "profile_sha256": profile_sha,
        "profile_size_bytes": profile_size,
    }


def _verify_existing_install_layout(final_root: Path, manifest: dict[str, Any], profile: dict[str, Any]) -> None:
    file_entries = {entry["path"]: entry for entry in manifest["files"]}
    expected_files = set(file_entries)
    expected_directories = set(manifest["directories"])
    expected = expected_files | expected_directories | {MANIFEST_NAME, INSTALL_MARKER}
    uid_text, gid_text = profile["container_user"].split(":", 1)
    owners = {(os.geteuid(), os.getegid()), (int(uid_text), int(gid_text))}
    try:
        root_metadata = final_root.lstat()
    except OSError as exc:
        raise LearnerBootstrapError("existing Learner install root cannot be inspected") from exc
    if (not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode)
            or (root_metadata.st_uid, root_metadata.st_gid) not in owners
            or stat.S_IMODE(root_metadata.st_mode) & 0o077):
        raise LearnerBootstrapError("existing Learner install root has unsafe owner or permissions")

    for relative in expected_directories:
        path = final_root.joinpath(*PurePosixPath(relative).parts)
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise LearnerBootstrapError("existing Learner install has a missing bundle directory") from exc
        if (not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
                or (metadata.st_uid, metadata.st_gid) not in owners
                or stat.S_IMODE(metadata.st_mode) & 0o077):
            raise LearnerBootstrapError("existing Learner install has an unsafe bundle directory")

    for base, directories, files in os.walk(final_root, topdown=True, followlinks=False):
        for name in [*directories, *files]:
            path = Path(base) / name
            relative = path.relative_to(final_root).as_posix()
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise LearnerBootstrapError("existing Learner install changed during validation") from exc
            if stat.S_ISLNK(metadata.st_mode) or not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                raise LearnerBootstrapError("existing Learner install contains a link or special file")
            if relative == "state" or relative.startswith("state/"):
                continue
            if relative == "venv" or relative.startswith("venv/"):
                if ((metadata.st_uid, metadata.st_gid) not in owners
                        or stat.S_IMODE(metadata.st_mode) & 0o022):
                    raise LearnerBootstrapError("host control venv has unsafe ownership or permissions")
                continue
            host_control_root = f".host-control/{manifest['deployment_id']}"
            if relative == ".host-control" or relative == host_control_root or relative.startswith(
                host_control_root + "/"
            ):
                if ((metadata.st_uid, metadata.st_gid) not in owners
                        or stat.S_IMODE(metadata.st_mode) & 0o077):
                    raise LearnerBootstrapError("Learner host-control metadata has unsafe ownership or permissions")
                continue
            if relative.startswith(".host-control/"):
                raise LearnerBootstrapError("existing Learner install contains an unknown host-control directory")
            if relative not in expected:
                raise LearnerBootstrapError("existing Learner install contains an undeclared file or directory")
            expected_mode = file_entries[relative]["mode"] if relative in file_entries else 0o700
            if relative in {MANIFEST_NAME, INSTALL_MARKER}:
                expected_mode = 0o600
            if ((metadata.st_uid, metadata.st_gid) not in owners
                    or stat.S_IMODE(metadata.st_mode) != expected_mode):
                raise LearnerBootstrapError("existing Learner bundle member has unsafe owner or mode")


def _verify_existing_install(
    final_root: Path,
    manifest: dict[str, Any],
    bundle_sha256: str,
    *,
    allow_rootless_container_user: bool = False,
    rootless_host_owner: tuple[int, int] | None = None,
) -> None:
    marker_path = final_root / INSTALL_MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        raise LearnerBootstrapError("refusing to overwrite an unowned Learner install directory")
    profile_path = final_root / manifest["profile"]["path"]
    if profile_path.is_symlink() or not profile_path.is_file():
        raise LearnerBootstrapError("existing Learner install deployment profile is missing or unsafe")
    profile = _strict_json(profile_path.read_bytes(), label="installed Learner profile")
    _verify_existing_install_layout(final_root, manifest, profile)
    marker = _strict_json(marker_path.read_bytes(), label="Learner install marker")
    expected_marker = _marker_data(final_root, manifest, bundle_sha256)
    if marker != expected_marker:
        raise LearnerBootstrapError("existing Learner install marker differs; refusing to overwrite it")
    for entry in manifest["files"]:
        # State is mutable application data. The original bundle hash proves
        # what was installed, but a subsequent successful training run is
        # allowed to advance checkpoints without making the install unusable.
        if entry["role"] == "state":
            continue
        path = final_root.joinpath(*PurePosixPath(entry["path"]).parts)
        if entry["role"] == "profile":
            expected_hash = expected_marker["profile_sha256"]
            expected_size = expected_marker["profile_size_bytes"]
        else:
            expected_hash = entry["sha256"]
            expected_size = entry["size_bytes"]
        if path.is_symlink() or not path.is_file():
            raise LearnerBootstrapError("existing Learner install contains a missing or unsafe file")
        observed_hash, observed_size = _sha_size(path)
        if observed_hash != expected_hash or observed_size != expected_size:
            raise LearnerBootstrapError("existing Learner install contents differ from its pinned bundle")
    _verify_preserved_state(
        final_root / "state",
        profile,
        allow_rootless_container_user=allow_rootless_container_user,
        rootless_host_owner=rootless_host_owner,
    )


def _verify_preserved_state(
    path: Path,
    profile: dict[str, Any],
    *,
    allow_rootless_container_user: bool = False,
    rootless_host_owner: tuple[int, int] | None = None,
) -> None:
    """Allow checkpoint writes while requiring a private, non-link state tree."""
    uid_text, gid_text = profile["container_user"].split(":", 1)
    expected = {(int(uid_text), int(gid_text)), (0, 0)}
    if profile["container_user"] == "0:0":
        if not allow_rootless_container_user or rootless_host_owner is None:
            raise LearnerBootstrapError("rootless state validation requires a measured host UID/GID mapping")
        mapped_uid, mapped_gid = rootless_host_owner
        if (isinstance(mapped_uid, bool) or not isinstance(mapped_uid, int) or mapped_uid <= 0
                or isinstance(mapped_gid, bool) or not isinstance(mapped_gid, int) or mapped_gid < 0):
            raise LearnerBootstrapError("rootless state mapping must identify a non-root host account")
        expected.add((mapped_uid, mapped_gid))
    elif allow_rootless_container_user or rootless_host_owner is not None:
        raise LearnerBootstrapError("rootless state mapping cannot be applied to a non-root profile")
    if path.is_symlink() or not path.is_dir():
        raise LearnerBootstrapError("existing Learner state root is missing or unsafe")
    for base, directories, files in os.walk(path, topdown=True, followlinks=False):
        for name in [".", *directories, *files]:
            candidate = Path(base) if name == "." else Path(base) / name
            try:
                metadata = candidate.lstat()
            except OSError as exc:
                raise LearnerBootstrapError("existing Learner state changed during validation") from exc
            if stat.S_ISLNK(metadata.st_mode) or not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                raise LearnerBootstrapError("existing Learner state contains a link or special file")
            if (metadata.st_uid, metadata.st_gid) not in expected:
                raise LearnerBootstrapError("existing Learner state has an unexpected owner")
            mode = stat.S_IMODE(metadata.st_mode)
            if mode & 0o077:
                raise LearnerBootstrapError("existing Learner state must remain private")
            if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
                raise LearnerBootstrapError("existing Learner state contains a hard-linked file")


def _load_deployment(profile_path: Path, *, allow_rootless_container_user: bool = False):
    try:
        from .config import load_learner_deployment

        # L4 runs the full asset/owner check after it safely adopts the
        # root-extracted 0600 credentials for the profile's numeric UID.
        return load_learner_deployment(
            profile_path,
            allow_gpu_autodetect=False,
            allow_rootless_container_user=allow_rootless_container_user,
        )
    except Exception as exc:
        if isinstance(exc, LearnerBootstrapError):
            raise
        raise LearnerBootstrapError("installed Learner deployment profile is invalid") from exc


def _prepare_host(
    deployment: Any,
    *,
    install_root: Path,
    host_prepare: Callable[..., Any] | None,
    runner: Any | None,
    systemd_dir: Path | None,
    allow_package_changes: bool,
) -> Any:
    if host_prepare is None:
        try:
            from .host import prepare_learner_host
        except Exception as exc:
            raise LearnerBootstrapError("Learner host preparation module is not installed") from exc
        host_prepare = prepare_learner_host
    kwargs: dict[str, Any] = {"install_root": install_root}
    if runner is not None:
        kwargs["runner"] = runner
    if systemd_dir is not None:
        kwargs["systemd_dir"] = systemd_dir
    if allow_package_changes:
        kwargs["allow_package_changes"] = True
    try:
        return host_prepare(deployment, **kwargs)
    except Exception as exc:
        raise LearnerBootstrapError("Learner host setup did not complete") from exc


def install_learner_bundle(
    bundle_path: str | Path,
    expected_sha256: str,
    *,
    install_root: str | Path = DEFAULT_INSTALL_ROOT,
    expected_size_bytes: int | None = None,
    host_prepare: Callable[..., Any] | None = None,
    runner: Any | None = None,
    systemd_dir: str | Path | None = None,
    allow_package_changes: bool = False,
    allow_rootless_container_user: bool = False,
    rootless_host_owner: tuple[int, int] | None = None,
) -> LearnerInstallReceipt:
    """Install a verified offline/private bundle and delegate host setup to L4.

    Existing bundles for the same deployment_id are reusable only when their
    marker and every payload hash match exactly. A different or unowned install
    is never replaced, even if its Learner service appears inactive.
    """
    if type(allow_package_changes) is not bool:
        raise TypeError("allow_package_changes must be bool")
    if type(allow_rootless_container_user) is not bool:
        raise TypeError("allow_rootless_container_user must be bool")
    if rootless_host_owner is not None:
        if (not allow_rootless_container_user or not isinstance(rootless_host_owner, tuple)
                or len(rootless_host_owner) != 2
                or isinstance(rootless_host_owner[0], bool) or not isinstance(rootless_host_owner[0], int)
                or rootless_host_owner[0] <= 0 or isinstance(rootless_host_owner[1], bool)
                or not isinstance(rootless_host_owner[1], int) or rootless_host_owner[1] < 0
                or rootless_host_owner != (os.getuid(), os.getgid())):
            raise LearnerBootstrapError("rootless state mapping must match the invoking host account")
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise LearnerBootstrapError("pinned bundle SHA-256 is invalid")
    deployment_id = None
    inspected_manifest: dict[str, Any] | None = None
    # Read only the pinned manifest first to select the per-deployment target;
    # full external SHA and member validation happen inside extraction below.
    try:
        stream, _size = _open_bundle(bundle_path)
        with stream:
            if expected_size_bytes is not None and (
                isinstance(expected_size_bytes, bool) or not isinstance(expected_size_bytes, int)
                or expected_size_bytes != _size
            ):
                raise LearnerBootstrapError("Learner bundle compressed size differs from the expected size")
            if not hmac.compare_digest(_hash_stream(stream), expected_sha256):
                raise LearnerBootstrapError("Learner bundle SHA-256 does not match the pinned value")
            inspected_manifest, _raw_manifest = _read_manifest(stream)
            _manifest_structure(inspected_manifest)
            deployment_id = inspected_manifest["deployment_id"]
    except LearnerBundleError as exc:
        raise LearnerBootstrapError(str(exc)) from exc
    base = _install_base(install_root)
    with _deployment_install_lock(base, deployment_id):
        return _install_verified_manifest(
            bundle_path,
            expected_sha256,
            expected_size_bytes=expected_size_bytes,
            base=base,
            deployment_id=deployment_id,
            inspected_manifest=inspected_manifest,
            host_prepare=host_prepare,
            runner=runner,
            systemd_dir=systemd_dir,
            allow_package_changes=allow_package_changes,
            allow_rootless_container_user=allow_rootless_container_user,
            rootless_host_owner=rootless_host_owner,
        )


def _install_verified_manifest(
    bundle_path: str | Path,
    expected_sha256: str,
    *,
    expected_size_bytes: int | None,
    base: Path,
    deployment_id: str,
    inspected_manifest: dict[str, Any],
    host_prepare: Callable[..., Any] | None,
    runner: Any | None,
    systemd_dir: str | Path | None,
    allow_package_changes: bool,
    allow_rootless_container_user: bool,
    rootless_host_owner: tuple[int, int] | None,
) -> LearnerInstallReceipt:
    final_root = base / deployment_id
    if final_root.is_symlink():
        raise LearnerBootstrapError("Learner deployment directory cannot be a symbolic link")
    if final_root.exists():
        if not final_root.is_dir():
            raise LearnerBootstrapError("Learner deployment path already exists and is not a directory")
        _verify_existing_install(
            final_root,
            inspected_manifest,
            expected_sha256,
            allow_rootless_container_user=allow_rootless_container_user,
            rootless_host_owner=rootless_host_owner,
        )
        deployment = _load_deployment(
            final_root / inspected_manifest["profile"]["path"],
            allow_rootless_container_user=allow_rootless_container_user,
        )
        host_result = _prepare_host(
            deployment,
            install_root=final_root,
            host_prepare=host_prepare,
            runner=runner,
            systemd_dir=Path(systemd_dir) if systemd_dir is not None else None,
            allow_package_changes=allow_package_changes,
        )
        return LearnerInstallReceipt(
            install_root=final_root,
            deployment_id=deployment_id,
            bundle_sha256=expected_sha256,
            image_id=inspected_manifest["image_id"],
            profile_sha256=_marker_data(final_root, inspected_manifest, expected_sha256)["profile_sha256"],
            state="reused_verified_install",
            host_state=_host_state(host_result),
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{deployment_id}.staging-", dir=base))
    shutil.rmtree(staging)
    try:
        manifest = extract_verified_bundle(
            bundle_path,
            expected_sha256,
            staging,
            expected_size_bytes=expected_size_bytes,
        )
        if manifest != inspected_manifest:
            raise LearnerBootstrapError("Learner bundle manifest changed after inspection")
        profile_path = staging / manifest["profile"]["path"]
        profile_document = _strict_json(profile_path.read_bytes(), label="Learner deployment profile")
        remapped = _remapped_profile(profile_document, manifest, final_root)
        _write_private_json(profile_path, remapped)
        deployment = _load_deployment(
            profile_path,
            allow_rootless_container_user=allow_rootless_container_user,
        )
        marker = _marker_data(staging, manifest, expected_sha256)
        _write_private_json(staging / INSTALL_MARKER, marker)
        _fsync_tree_dirs(staging)
        if final_root.exists() or final_root.is_symlink():
            raise LearnerBootstrapError("another install created this deployment_id during extraction")
        try:
            os.rename(staging, final_root)
        except FileExistsError as exc:
            raise LearnerBootstrapError("Learner deployment was installed concurrently") from exc
        parent_fd = os.open(base, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # L4 creates/chowns mount directories and installs/loads the exact image;
    # it receives only the strict path-only profile and the final private root.
    deployment = _load_deployment(
        final_root / manifest["profile"]["path"],
        allow_rootless_container_user=allow_rootless_container_user,
    )
    host_result = _prepare_host(
        deployment,
        install_root=final_root,
        host_prepare=host_prepare,
        runner=runner,
        systemd_dir=Path(systemd_dir) if systemd_dir is not None else None,
        allow_package_changes=allow_package_changes,
    )
    marker = _strict_json((final_root / INSTALL_MARKER).read_bytes(), label="Learner install marker")
    return LearnerInstallReceipt(
        install_root=final_root,
        deployment_id=deployment_id,
        bundle_sha256=expected_sha256,
        image_id=manifest["image_id"],
        profile_sha256=marker["profile_sha256"],
        state="installed_verified",
        host_state=_host_state(host_result),
    )


def _host_state(value: Any) -> str | None:
    if isinstance(value, Mapping):
        state = value.get("state") or value.get("status")
        return state if isinstance(state, str) else None
    state = getattr(value, "state", None) or getattr(value, "status", None)
    return state if isinstance(state, str) else None
