"""Deterministic private Learner bundle construction and starter rendering.

The bundle is a private deployment artifact. It contains only the host control
closure, the pinned Learner image archive, the profile's explicitly mounted
assets, and mode-0600 credential files. Trainer code and model/data assets are
never copied into the public Worker artifact store.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shlex
import stat
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

DEFAULT_INSTALL_ROOT = Path("/var/lib/qwen-rl/learner")
BUNDLE_SCHEMA = "qwen-rl-learner-bundle/v1"
MANIFEST_NAME = "bundle-manifest.json"
IMAGE_MEMBER = "image/learner.tar.gz"
MAX_MEMBER_BYTES = 1 << 42  # 4 TiB; checked before and while streaming.
MAX_UNPACKED_BYTES = 1 << 43  # 8 TiB, including the manifest and tar headers.
MAX_MEMBER_COUNT = 100_000
CHUNK_SIZE = 8 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_KEY_MARKER = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
)

# Deliberately limited to host control and its small import closure. The full
# Learner/trainer tree is delivered only inside the pinned Learner image.
_SOURCE_FILES = (
    "qwen_rl/__init__.py",
    "qwen_rl/__main__.py",
    "qwen_rl/cli.py",
    "qwen_rl/deployment/__init__.py",
    "qwen_rl/models/__init__.py",
    "qwen_rl/models/artifact.py",
    "qwen_rl/models/artifacts.py",
    "qwen_rl/worker/__init__.py",
    "qwen_rl/worker/client.py",
    "qwen_rl/worker/generation.py",
    "qwen_rl/worker/native_artifact.py",
    "qwen_rl/worker/placement.py",
    "qwen_rl/worker/policy_sync.py",
    "qwen_rl/paths.py",
    "qwen_rl/contracts/__init__.py",
    "qwen_rl/contracts/worker.py",
)
_SOURCE_TREES = (
    "qwen_rl/cluster",
    "qwen_rl/deployment/learner",
    "qwen_rl/runtime",
)
_DEPLOY_TREE = "deploy/learner"
_DEPLOY_FILES = (
    "deploy/learner/Dockerfile",
    "deploy/learner/Dockerfile.dockerignore",
    "deploy/learner/qwen-rl-learner@.service",
    "deploy/learner/install.py",
    "deploy/learner/host-requirements.txt",
)


class LearnerBundleError(ValueError):
    """A private Learner bundle input or archive violates its contract."""


@dataclass(frozen=True, slots=True)
class LearnerBundle:
    path: Path
    sha256: str
    size_bytes: int
    file_count: int
    image_id: str
    manifest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "image_id": self.image_id,
        }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LearnerBundleError("Learner profile contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise LearnerBundleError("Learner profile contains a non-finite JSON value")


def _profile_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        path = _ensure_source_path(str(Path(value).expanduser().absolute()), label="Learner profile", expect="file")
        try:
            with _open_regular(path, "Learner profile") as stream:
                profile_text = stream.read().decode("utf-8")
            raw = json.loads(
                profile_text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except LearnerBundleError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LearnerBundleError("Learner profile could not be read as strict JSON") from exc
    elif isinstance(value, Mapping):
        raw = dict(value)
    elif callable(getattr(value, "model_dump", None)):
        raw = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        raw = asdict(value)
    else:
        raise TypeError("profile must be a path, mapping, or LearnerDeployment value")
    if not isinstance(raw, dict):
        raise LearnerBundleError("Learner profile root must be a JSON object")
    try:
        # Reject non-JSON values in typed objects and detach nested references.
        raw = json.loads(json.dumps(raw, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise LearnerBundleError("Learner profile contains a non-JSON value") from exc
    try:
        from .config import check_learner_profile, load_learner_deployment

        # Rootless execution is still gated at installation by the live daemon
        # namespace and UID/GID bind probe; the bundle records only the profile.
        deployment = load_learner_deployment(
            raw,
            allow_gpu_autodetect=False,
            allow_rootless_container_user=True,
        )
        check_learner_profile(
            deployment,
            require_assets=False,
            allow_gpu_autodetect=False,
            allow_rootless_container_user=True,
        )
        return deployment.model_dump(mode="json")
    except Exception as exc:
        if isinstance(exc, LearnerBundleError):
            raise
        raise LearnerBundleError(f"Learner profile failed strict deployment validation: {exc}") from exc


def _safe_relative(value: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or "\\" in value
            or "\x00" in value or value.endswith("/") or "//" in value
            or any(ord(char) < 32 for char in value)):
        raise LearnerBundleError("bundle path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise LearnerBundleError("bundle path is unsafe")
    return path


def _ensure_source_path(path_value: str, *, label: str, expect: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise LearnerBundleError(f"{label} host_path must be absolute")
    if any(part in {".", ".."} for part in path.parts):
        raise LearnerBundleError(f"{label} host_path contains traversal components")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise LearnerBundleError(f"{label} source is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise LearnerBundleError(f"{label} source traverses a symbolic link")
    if expect == "file":
        if not path.is_file():
            raise LearnerBundleError(f"{label} source must be a regular file")
    elif expect == "directory":
        if not path.is_dir():
            raise LearnerBundleError(f"{label} source must be a directory")
    else:
        raise AssertionError("unsupported path expectation")
    return path


def _sha_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        with _open_regular(path, "bundle input") as stream:
            while chunk := stream.read(CHUNK_SIZE):
                total += len(chunk)
                if total > MAX_MEMBER_BYTES:
                    raise LearnerBundleError("Learner bundle member exceeds the per-file size limit")
                digest.update(chunk)
    except LearnerBundleError:
        raise
    except OSError as exc:
        raise LearnerBundleError("Learner bundle input could not be read") from exc
    return digest.hexdigest(), total


def _open_regular(path: Path, label: str):
    """Open a regular file without following a last-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LearnerBundleError(f"{label} is not safely readable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise LearnerBundleError(f"{label} is not a regular file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _contains_private_key_marker(path: Path) -> bool:
    marker_tail = b""
    with _open_regular(path, "private-material scan") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            sample = marker_tail + chunk
            if _PRIVATE_KEY_MARKER.search(sample):
                return True
            marker_tail = sample[-128:]
    return False


def _scan_tree(
    source: Path,
    archive_prefix: str,
    *,
    mode: int,
    reject_private_key_material: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect regular files and directories without following symlinks."""
    source = _ensure_source_path(str(source), label=archive_prefix, expect="directory")
    files: list[dict[str, Any]] = []
    directories: list[str] = [_safe_relative(archive_prefix).as_posix()]
    for base, dirnames, filenames in os.walk(source, topdown=True, followlinks=False):
        base_path = Path(base)
        for dirname in sorted(dirnames):
            path = base_path / dirname
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise LearnerBundleError("asset tree contains a non-directory or symbolic link")
            rel = path.relative_to(source).as_posix()
            target = _safe_relative(f"{archive_prefix}/{rel}").as_posix()
            directories.append(target)
        for filename in sorted(filenames):
            path = base_path / filename
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise LearnerBundleError("asset tree contains a non-regular file or symbolic link")
            if reject_private_key_material and (
                _is_sensitive_file(path) or _contains_private_key_marker(path)
            ):
                raise LearnerBundleError("read-only resources cannot carry private-key material")
            rel = path.relative_to(source).as_posix()
            target = _safe_relative(f"{archive_prefix}/{rel}").as_posix()
            digest, size = _sha_size(path)
            files.append({
                "path": target,
                "source": path,
                "sha256": digest,
                "size_bytes": size,
                "mode": mode,
            })
    return files, directories


def _single_file(source: Path, target: str, *, mode: int, label: str) -> dict[str, Any]:
    source = _ensure_source_path(str(source), label=label, expect="file")
    metadata = source.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_MEMBER_BYTES:
        raise LearnerBundleError(f"{label} must be a bounded regular file")
    digest, size = _sha_size(source)
    return {
        "path": _safe_relative(target).as_posix(),
        "source": source,
        "sha256": digest,
        "size_bytes": size,
        "mode": mode,
    }


def _source_closure(project_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    root = Path(project_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise LearnerBundleError("project_root must be a real directory")
    root = root.resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}
    directories: set[str] = set()

    def add_file(relative: str, *, mode: int = 0o644) -> None:
        target = _safe_relative(f"source/{relative}").as_posix()
        source = root / relative
        item = _single_file(source, target, mode=mode, label=relative)
        with _open_regular(source, relative) as stream:
            source_prefix = stream.read(1024 * 1024)
        if _PRIVATE_KEY_MARKER.search(source_prefix):
            raise LearnerBundleError(f"host control source contains private key material: {relative}")
        files[target] = item

    def add_python_tree(relative: str) -> None:
        source_root = root / relative
        if not source_root.is_dir() or source_root.is_symlink():
            raise LearnerBundleError(f"required host control source is missing: {relative}")
        for base, dirnames, filenames in os.walk(source_root, topdown=True, followlinks=False):
            base_path = Path(base)
            kept: list[str] = []
            for dirname in sorted(dirnames):
                subdir = base_path / dirname
                metadata = subdir.lstat()
                if not stat.S_ISDIR(metadata.st_mode):
                    raise LearnerBundleError("host control source contains a symbolic directory")
                if dirname in {"__pycache__", "tests", "test", ".git"}:
                    continue
                kept.append(dirname)
                archive_dir = _safe_relative(f"source/{subdir.relative_to(root).as_posix()}").as_posix()
                directories.add(archive_dir)
            dirnames[:] = kept
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                add_file((base_path / filename).relative_to(root).as_posix())

    for relative in _SOURCE_FILES:
        add_file(relative)
    for relative in _SOURCE_TREES:
        add_python_tree(relative)

    # Deploy assets are a fixed control surface. Do not walk arbitrary files or
    # examples from this directory into a private Learner bundle.
    deploy_root = root / _DEPLOY_TREE
    if not deploy_root.is_dir() or deploy_root.is_symlink():
        raise LearnerBundleError("deploy/learner must contain the pinned Learner image/install assets")
    for relative in _DEPLOY_FILES:
        source = root / relative
        if not source.exists():
            continue
        add_file(relative, mode=0o755 if relative.endswith("install.py") else 0o644)
    if "source/deploy/learner/Dockerfile" not in files:
        raise LearnerBundleError("deploy/learner/Dockerfile is missing from the Learner bundle source closure")
    required_deploy_files = {
        "source/deploy/learner/Dockerfile.dockerignore",
        "source/deploy/learner/qwen-rl-learner@.service",
        "source/deploy/learner/install.py",
        "source/deploy/learner/host-requirements.txt",
    }
    missing = sorted(required_deploy_files - files.keys())
    if missing:
        raise LearnerBundleError(
            "Learner bundle source closure is missing required install inputs: "
            + ", ".join(missing)
        )

    # Make the imported package structure explicit even when a directory is
    # empty in this source checkout; only Python files are admitted from code.
    for directory in ("source", "source/qwen_rl", "source/qwen_rl/deployment",
                      "source/qwen_rl/deployment/learner", "source/qwen_rl/cluster",
                      "source/qwen_rl/runtime"):
        directories.add(directory)
    return list(files.values()), sorted(directories)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                        allow_nan=False) + "\n").encode("utf-8")


def _install_root(value: str | Path, deployment_id: str) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute() or root == Path("/") or ".." in root.parts:
        raise LearnerBundleError("install_root must be an absolute non-root path")
    if any(ord(char) < 32 for char in str(root)):
        raise LearnerBundleError("install_root contains control characters")
    return root / deployment_id


def _profile_path_map(profile: dict[str, Any]) -> list[dict[str, str]]:
    rows = [
        {"pointer": "/mounts/runtime_config/host_path", "target": "runtime-config.json"},
        {"pointer": "/mounts/model/host_path", "target": "model"},
        {"pointer": "/mounts/dataset/host_path", "target": "dataset"},
        {"pointer": "/mounts/state/host_path", "target": "state"},
    ]
    for name in sorted(profile["credentials"]):
        rows.append({"pointer": f"/credentials/{name}/host_path", "target": f"credentials/{name}"})
    for index, _resource in enumerate(profile.get("resources", [])):
        rows.append({"pointer": f"/resources/{index}/host_path", "target": f"resources/{index:04d}"})
    ssh = profile.get("ssh")
    if isinstance(ssh, dict) and ssh.get("enabled") is True:
        rows.append({
            "pointer": "/ssh/client_private_key_host_path",
            "target": "credentials/ssh-client-key",
        })
    return rows


def _member_role(member_path: str) -> str:
    if member_path == "learner-deployment.json":
        return "profile"
    if member_path == IMAGE_MEMBER:
        return "image"
    if member_path == "runtime-config.json":
        return "runtime_config"
    first = member_path.split("/", 1)[0]
    if first == "resources":
        return "resource"
    if first in {"model", "dataset", "state", "credentials"}:
        return "credential" if first == "credentials" else first
    if first == "source":
        return "control"
    raise LearnerBundleError("bundle member has no recognized trust role")


def _set_pointer(document: dict[str, Any], pointer: str, value: str) -> None:
    parts = pointer.lstrip("/").split("/")
    current: Any = document
    for part in parts[:-1]:
        part = part.replace("~1", "/").replace("~0", "~")
        current = current[int(part)] if isinstance(current, list) else current[part]
    leaf = parts[-1].replace("~1", "/").replace("~0", "~")
    if isinstance(current, list):
        current[int(leaf)] = value
    else:
        current[leaf] = value


def build_learner_bundle(
    project_root: str | Path,
    *,
    profile: Any,
    image_archive: str | Path,
    destination: str | Path,
    include_state: bool = False,
    install_root: str | Path = DEFAULT_INSTALL_ROOT,
) -> LearnerBundle:
    """Create a deterministic private bundle from the profile's explicit host paths.

    Source assets are derived from the strict profile rather than accepted as
    independent CLI paths, preventing a bundle from mounting different model,
    data, runtime config, or credentials than the reviewed profile describes.
    """
    if not isinstance(include_state, bool):
        raise TypeError("include_state must be a bool")
    document = _profile_mapping(profile)
    try:
        from .config import check_learner_profile

        deployment = check_learner_profile(
            document,
            require_assets=False,
            allow_gpu_autodetect=False,
            allow_rootless_container_user=True,
        )
        document = deployment.model_dump(mode="json")
    except Exception as exc:
        if isinstance(exc, LearnerBundleError):
            raise
        raise LearnerBundleError(
            f"Learner profile or mounted assets failed strict validation: {exc}"
        ) from exc
    deployment_id = document["deployment_id"]
    final_root = _install_root(install_root, deployment_id)
    source_files, source_directories = _source_closure(Path(project_root))
    files_by_path = {item["path"]: item for item in source_files}
    directories = set(source_directories)

    def add_tree(source_value: str, target_root: str, *, mode: int, label: str) -> None:
        tree_files, tree_dirs = _scan_tree(
            _ensure_source_path(source_value, label=label, expect="directory"),
            target_root,
            mode=mode,
            reject_private_key_material=label.startswith("read-only resource"),
        )
        directories.update(tree_dirs)
        for item in tree_files:
            if item["path"] in files_by_path or item["path"] == MANIFEST_NAME:
                raise LearnerBundleError("bundle paths collide")
            files_by_path[item["path"]] = item

    def add_file(source_value: str, target: str, *, mode: int, label: str) -> None:
        item = _single_file(Path(source_value), target, mode=mode, label=label)
        if item["path"] in files_by_path or item["path"] == MANIFEST_NAME:
            raise LearnerBundleError("bundle paths collide")
        files_by_path[item["path"]] = item

    mounts = document["mounts"]
    runtime_config = _ensure_source_path(
        mounts["runtime_config"]["host_path"], label="runtime_config", expect="file",
    )
    add_file(str(runtime_config), "runtime-config.json", mode=0o600, label="runtime_config")
    add_tree(mounts["model"]["host_path"], "model", mode=0o600, label="model")
    add_tree(mounts["dataset"]["host_path"], "dataset", mode=0o600, label="dataset")

    state_path_value = mounts["state"]["host_path"]
    if include_state:
        add_tree(state_path_value, "state", mode=0o600, label="full learner state")
    else:
        state_source = Path(state_path_value).expanduser()
        if state_source.exists() or state_source.is_symlink():
            _ensure_source_path(str(state_source), label="learner state", expect="directory")
            if any(state_source.iterdir()):
                raise LearnerBundleError(
                    "learner state is nonempty; pass include_state=True or select an empty state directory"
                )
        directories.add("state")

    sensitive_sources: list[Path] = []
    for env_name, credential in sorted(document["credentials"].items()):
        path = _ensure_source_path(credential["host_path"], label=f"credential {env_name}", expect="file")
        metadata = path.lstat()
        if (stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()):
            raise LearnerBundleError(f"credential {env_name} must be owned by the bundler and mode 0600")
        sensitive_sources.append(path)
        add_file(str(path), f"credentials/{env_name}", mode=0o600, label=f"credential {env_name}")

    ssh = document["ssh"]
    if isinstance(ssh, dict) and ssh.get("enabled") is True:
        ssh_key = _ensure_source_path(
            ssh["client_private_key_host_path"], label="SSH client private key", expect="file",
        )
        _validate_private_file(ssh_key, "SSH client private key")
        sensitive_sources.append(ssh_key)
        add_file(str(ssh_key), "credentials/ssh-client-key", mode=0o600, label="SSH client private key")

    asset_directories = [
        _ensure_source_path(mounts["model"]["host_path"], label="model", expect="directory"),
        _ensure_source_path(mounts["dataset"]["host_path"], label="dataset", expect="directory"),
    ]
    if include_state:
        asset_directories.append(_ensure_source_path(state_path_value, label="full learner state", expect="directory"))
    for index, resource in enumerate(document.get("resources", [])):
        resource_path = Path(resource["host_path"]).expanduser()
        if resource_path.exists() and resource_path.is_dir():
            asset_directories.append(_ensure_source_path(
                str(resource_path), label=f"read-only resource {index}", expect="directory",
            ))
    for secret_path in sensitive_sources:
        if any(secret_path == directory or directory in secret_path.parents for directory in asset_directories):
            raise LearnerBundleError("credential or SSH private key overlaps a model/data/state/resource mount")

    if not any(name.startswith("model/") for name in files_by_path):
        raise LearnerBundleError("Learner model mount must contain at least one regular file")
    if not any(name.startswith("dataset/") for name in files_by_path):
        raise LearnerBundleError("Learner dataset mount must contain at least one regular file")

    for index, resource in enumerate(document.get("resources", [])):
        source = _ensure_source_path(resource["host_path"], label=f"read-only resource {index}", expect=(
            "directory" if Path(resource["host_path"]).is_dir() else "file"
        ))
        target = f"resources/{index:04d}"
        if source.is_dir():
            add_tree(str(source), target, mode=0o600,
                     label=f"read-only resource {index}")
        else:
            if _is_sensitive_file(source) or _contains_private_key_marker(source):
                raise LearnerBundleError("read-only resources cannot carry private-key material")
            add_file(str(source), target, mode=0o600, label=f"read-only resource {index}")

    image_export = image_archive
    export_descriptor = getattr(image_export, "descriptor", None)
    image_value = getattr(image_export, "path", image_export)
    image = _ensure_source_path(str(image_value), label="pinned Learner image archive", expect="file")
    if export_descriptor is not None and getattr(export_descriptor, "image_id", None) != document["image_id"]:
        raise LearnerBundleError("exported image descriptor differs from the profile-pinned image_id")
    with _open_regular(image, "pinned Learner image archive") as stream:
        if stream.read(2) != b"\x1f\x8b":
            raise LearnerBundleError("pinned Learner image archive must be a deterministic gzip Docker save")
    image_item = _single_file(image, IMAGE_MEMBER, mode=0o600, label="pinned Learner image archive")
    files_by_path[image_item["path"]] = image_item
    directories.update({"image", "credentials"})

    path_map = _profile_path_map(document)
    for row in path_map:
        target = _safe_relative(row["target"]).as_posix()
        _set_pointer(document, row["pointer"], str(final_root.joinpath(*PurePosixPath(target).parts)))
    profile_payload = _json_bytes(document)
    profile_path = "learner-deployment.json"
    profile_item = {
        "path": profile_path,
        "source_bytes": profile_payload,
        "sha256": hashlib.sha256(profile_payload).hexdigest(),
        "size_bytes": len(profile_payload),
        "mode": 0o600,
    }
    files_by_path[profile_path] = profile_item

    # Tar directories are explicit, including parents of individual code and
    # credential files, and stay within the same allowlisted namespace.
    for member_path in files_by_path:
        parent = PurePosixPath(member_path).parent
        while parent.as_posix() not in {".", ""}:
            directories.add(_safe_relative(parent.as_posix()).as_posix())
            parent = parent.parent

    file_entries = []
    total_bytes = 0
    for member_path, item in sorted(files_by_path.items()):
        total_bytes += int(item["size_bytes"])
        file_entries.append({
            "path": member_path,
            "role": _member_role(member_path),
            "sha256": item["sha256"],
            "size_bytes": int(item["size_bytes"]),
            "mode": int(item["mode"]),
        })
    if len(file_entries) + len(directories) + 1 > MAX_MEMBER_COUNT:
        raise LearnerBundleError("Learner bundle contains too many files")
    if total_bytes > MAX_UNPACKED_BYTES:
        raise LearnerBundleError("Learner bundle exceeds its unpacked size limit")
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "deployment_id": deployment_id,
        "default_install_root": str(Path(install_root).expanduser()),
        "image_id": document["image_id"],
        "image_archive": {
            "path": IMAGE_MEMBER,
            "sha256": image_item["sha256"],
            "size_bytes": image_item["size_bytes"],
        },
        "profile": {"path": profile_path, "sha256": profile_item["sha256"]},
        "path_map": path_map,
        "directories": sorted(directories, key=lambda name: (name.count("/"), name)),
        "files": file_entries,
        "unpacked_size_bytes": total_bytes,
        "includes_full_state": include_state,
    }
    manifest_bytes = _json_bytes(manifest)
    target = Path(destination).expanduser().absolute()
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise LearnerBundleError("bundle destination must be a regular file")
    if target.exists():
        raise LearnerBundleError("refusing to overwrite an existing Learner bundle")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temp = Path(temp_name)
    published = False
    try:
        with temp.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as zipped, \
                    tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                info = _tar_info(MANIFEST_NAME, len(manifest_bytes), 0o600, kind="file")
                archive.addfile(info, BytesIO(manifest_bytes))
                for directory in sorted(directories, key=lambda name: (name.count("/"), name)):
                    archive.addfile(_tar_info(directory, 0, 0o700, kind="directory"))
                for member_path, item in sorted(
                    files_by_path.items(),
                    key=lambda pair: (pair[1]["path"].split("/", 1)[0] != "source", pair[0]),
                ):
                    source_bytes = item.get("source_bytes")
                    if source_bytes is not None:
                        stream = BytesIO(source_bytes)
                        archive.addfile(_tar_info(member_path, len(source_bytes), item["mode"], kind="file"), stream)
                        continue
                    source = Path(item["source"])
                    observed_hash, observed_size = _sha_size(source)
                    if (observed_hash != item["sha256"] or observed_size != item["size_bytes"]):
                        raise LearnerBundleError("Learner bundle input changed while the archive was being built")
                    info = _tar_info(member_path, observed_size, item["mode"], kind="file")
                    with source.open("rb") as stream:
                        archive.addfile(info, stream)
            raw.flush()
            os.fsync(raw.fileno())
        # Publish without replacing an archive created concurrently after the
        # initial existence check. The link is atomic because temp is a sibling.
        try:
            os.link(temp, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise LearnerBundleError("refusing to overwrite a concurrently created Learner bundle") from exc
        published = True
        temp.unlink()
        bundle_hash, bundle_size = _sha_size(target)
    except BaseException:
        temp.unlink(missing_ok=True)
        # Only remove a target we created. If another writer won the link race,
        # target remains theirs and must never be deleted here.
        if published:
            target.unlink(missing_ok=True)
        raise
    return LearnerBundle(
        path=target,
        sha256=bundle_hash,
        size_bytes=bundle_size,
        file_count=len(file_entries),
        image_id=manifest["image_id"],
        manifest=manifest,
    )


def _tar_info(name: str, size: int, mode: int, *, kind: str) -> tarfile.TarInfo:
    relative = _safe_relative(name)
    info = tarfile.TarInfo(relative.as_posix())
    info.size = size
    info.mode = mode
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    if kind == "directory":
        info.type = tarfile.DIRTYPE
        info.size = 0
    else:
        info.type = tarfile.REGTYPE
    return info


def _is_sensitive_file(path: Path) -> bool:
    return path.suffix.lower() in {".key", ".pem", ".p12", ".pfx"} or path.name.lower() in {
        "id_ed25519", "id_rsa", "client.key", "client.pem",
    }


def _validate_private_file(path: Path, label: str) -> None:
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1 or metadata.st_uid != os.getuid()):
        raise LearnerBundleError(f"private {label} must be an owned regular file with mode 0600")
    if metadata.st_size > 4 * 1024 * 1024:
        raise LearnerBundleError(f"private {label} exceeds the 4 MiB limit")


def _https_url(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) > 2048 or any(ord(char) < 32 for char in value):
        raise LearnerBundleError(f"{label} must be a credential-free HTTPS URL")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise LearnerBundleError(f"{label} must be a credential-free HTTPS URL") from exc
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise LearnerBundleError(f"{label} must be a credential-free HTTPS URL")
    return value


def render_learner_starter_command(
    *,
    installer_url: str,
    installer_sha256: str,
    bundle_url: str,
    bundle_sha256: str,
    install_root: str | Path = DEFAULT_INSTALL_ROOT,
) -> str:
    """Render one pasteable starter with public URLs/hashes only.

    Private bundle authentication is prompted by the verified stdlib installer;
    it is never embedded in shell text, URLs, or process arguments.
    """
    installer_url = _https_url(installer_url, "installer_url")
    bundle_url = _https_url(bundle_url, "bundle_url")
    for label, value in (("installer_sha256", installer_sha256), ("bundle_sha256", bundle_sha256)):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise LearnerBundleError(f"{label} must be a lowercase SHA-256 value")
    root = Path(install_root)
    if not root.is_absolute() or root == Path("/") or ".." in root.parts:
        raise LearnerBundleError("install_root must be an absolute non-root path")
    if any(ord(char) < 32 for char in str(root)):
        raise LearnerBundleError("install_root contains control characters")
    fetch_installer = (
        "import hashlib,sys,urllib.request\n"
        "url,digest,target=sys.argv[1:]\n"
        "request=urllib.request.Request(url,headers={'Accept':'text/x-python'})\n"
        "with urllib.request.urlopen(request,timeout=60) as response:\n"
        " final=response.geturl(); data=response.read(1048577)\n"
        " if len(data)>1048576 or not final.startswith('https://'): raise SystemExit('installer download rejected')\n"
        " if hashlib.sha256(data).hexdigest()!=digest: raise SystemExit('installer SHA-256 mismatch')\n"
        "open(target,'xb').write(data)\n"
    )
    return "\n".join((
        "set -eu",
        "umask 077",
        "_qwen_tmp=\"$(mktemp -d \"${TMPDIR:-/tmp}/qwen-learner-install.XXXXXXXX\")\"",
        "trap 'rm -rf -- \"$_qwen_tmp\"' EXIT HUP INT TERM",
        f"python3 -c {shlex.quote(fetch_installer)} {shlex.quote(installer_url)} {shlex.quote(installer_sha256)} \"$_qwen_tmp/install.py\"",
        "chmod 0700 \"$_qwen_tmp/install.py\"",
        f"sudo python3 \"$_qwen_tmp/install.py\" --bundle-url {shlex.quote(bundle_url)} --bundle-sha256 {shlex.quote(bundle_sha256)} --install-root {shlex.quote(str(root))} --allow-package-changes",
    ))
