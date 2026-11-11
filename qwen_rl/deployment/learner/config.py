"""CPU-safe profile parsing and validation for a portable Learner deployment.

This module deliberately imports only the Python standard library at module
load time so a private bundle installer can inspect a profile before loading
the Learner image or its GPU libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


PROFILE_SCHEMA = "qwen-rl-learner-deployment/v1"
PROFILE_CONTAINER_PATH = "/etc/qwen-rl/learner-deployment.json"
CONTAINER_WORKDIR = "/work"
CONTAINER_PYTHON = "/work/envs/rl-vllm/bin/python"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_ENV_NAMES = {
    "PATH", "PYTHONPATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD",
    "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES",
}
_GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{36}$")
_SSH_KEY = re.compile(r"^(ssh-ed25519|ssh-rsa) [A-Za-z0-9+/=]+(?: .*)?$")
_TOP_KEYS = {
    "schema_version", "deployment_id", "container_name", "image_id",
    "container_user", "container_workdir", "gpu_uuids", "publish", "mounts",
    "credentials", "resources", "ssh",
}
_MOUNT_KEYS = {"host_path", "container_path"}
_MOUNT_NAMES = {"runtime_config", "model", "dataset", "state"}
_PUBLISH_KEYS = {"host_address", "host_port", "container_port"}
_SSH_KEYS = {
    "enabled", "central_host", "central_user", "central_ssh_port",
    "central_host_public_key", "remote_api_port", "remote_ssh_port",
    "local_listener_address", "local_listener_port", "local_ssh_user",
    "administrator_public_key", "client_private_key_host_path",
}
_PROTECTED_HOST_ROOTS = (
    Path("/proc"), Path("/sys"), Path("/dev"),
    Path("/var/run"), Path("/var/lib/docker"),
)
_PROTECTED_CONTAINER_ROOTS = (
    PurePosixPath("/proc"), PurePosixPath("/sys"),
    PurePosixPath("/dev"), PurePosixPath("/var/run"),
)


class LearnerDeploymentError(RuntimeError):
    """A strict profile or host-resource validation failure."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("profile contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("profile JSON cannot contain non-finite values")


def _object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LearnerDeploymentError(f"Learner profile {label} fields do not match the strict schema")
    return value


def _string(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise LearnerDeploymentError(f"Learner profile {label} must be a non-empty string")
    return value


def _port(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise LearnerDeploymentError(f"Learner profile {label} must be a valid TCP port")
    return value


def _absolute_host_path(value: Any, label: str) -> Path:
    text = _string(value, label)
    path = Path(text).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise LearnerDeploymentError(f"Learner profile {label} must be an absolute path without '..'")
    return path


def _container_path(value: Any, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if not path.is_absolute() or ".." in path.parts or str(path) != text or str(path) == "/":
        raise LearnerDeploymentError(f"Learner profile {label} must be a normalized absolute container path")
    if any(path == root or root in path.parents for root in _PROTECTED_CONTAINER_ROOTS):
        raise LearnerDeploymentError(f"Learner profile {label} targets a protected container path")
    if path == PurePosixPath(PROFILE_CONTAINER_PATH):
        raise LearnerDeploymentError(f"Learner profile {label} overlaps the deployment profile mount")
    return text


@dataclass(frozen=True)
class LearnerMount:
    host_path: Path
    container_path: str

    def model_dump(self, *, mode: str = "python") -> dict[str, str]:
        if mode not in {"python", "json"}:
            raise ValueError("mode must be 'python' or 'json'")
        return {"host_path": str(self.host_path), "container_path": self.container_path}


@dataclass(frozen=True)
class LearnerResourceMount(LearnerMount):
    read_only: bool = True

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        value = super().model_dump(mode=mode)
        value["read_only"] = self.read_only
        return value


@dataclass(frozen=True)
class LearnerPublishSettings:
    host_address: str
    host_port: int
    container_port: int

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        if mode not in {"python", "json"}:
            raise ValueError("mode must be 'python' or 'json'")
        return {"host_address": self.host_address, "host_port": self.host_port,
                "container_port": self.container_port}


@dataclass(frozen=True)
class LearnerSSHSettings:
    enabled: bool
    central_host: str
    central_user: str
    central_ssh_port: int
    central_host_public_key: str
    remote_api_port: int
    remote_ssh_port: int
    local_listener_address: str
    local_listener_port: int
    local_ssh_user: str
    administrator_public_key: str
    client_private_key_host_path: Path

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        if mode not in {"python", "json"}:
            raise ValueError("mode must be 'python' or 'json'")
        return {
            "enabled": self.enabled,
            "central_host": self.central_host,
            "central_user": self.central_user,
            "central_ssh_port": self.central_ssh_port,
            "central_host_public_key": self.central_host_public_key,
            "remote_api_port": self.remote_api_port,
            "remote_ssh_port": self.remote_ssh_port,
            "local_listener_address": self.local_listener_address,
            "local_listener_port": self.local_listener_port,
            "local_ssh_user": self.local_ssh_user,
            "administrator_public_key": self.administrator_public_key,
            "client_private_key_host_path": str(self.client_private_key_host_path),
        }


@dataclass(frozen=True)
class LearnerCredential:
    env_name: str
    host_path: Path
    container_path: str

    def model_dump(self, *, mode: str = "python") -> dict[str, str]:
        if mode not in {"python", "json"}:
            raise ValueError("mode must be 'python' or 'json'")
        return {"host_path": str(self.host_path), "container_path": self.container_path}


@dataclass(frozen=True)
class LearnerDeployment:
    deployment_id: str
    container_name: str
    image_id: str
    container_user: str
    container_workdir: str
    gpu_uuids: tuple[str, ...]
    publish: LearnerPublishSettings
    runtime_config: LearnerMount
    model: LearnerMount
    dataset: LearnerMount
    state: LearnerMount
    credentials: Mapping[str, LearnerCredential]
    resources: tuple[LearnerResourceMount, ...]
    ssh: LearnerSSHSettings | None
    profile_path: Path

    @property
    def host_control_path(self) -> Path:
        """Host-only lifecycle metadata, always outside declared container mounts."""
        return self.profile_path.parent / ".host-control" / self.deployment_id

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        if mode not in {"python", "json"}:
            raise ValueError("mode must be 'python' or 'json'")
        return {
            "schema_version": PROFILE_SCHEMA,
            "deployment_id": self.deployment_id,
            "container_name": self.container_name,
            "image_id": self.image_id,
            "container_user": self.container_user,
            "container_workdir": self.container_workdir,
            "gpu_uuids": list(self.gpu_uuids),
            "publish": self.publish.model_dump(mode=mode),
            "mounts": {
                "runtime_config": self.runtime_config.model_dump(mode=mode),
                "model": self.model.model_dump(mode=mode),
                "dataset": self.dataset.model_dump(mode=mode),
                "state": self.state.model_dump(mode=mode),
            },
            "credentials": {name: mount.model_dump(mode=mode)
                            for name, mount in self.credentials.items()},
            "resources": [mount.model_dump(mode=mode) for mount in self.resources],
            "ssh": self.ssh.model_dump(mode=mode) if self.ssh is not None else None,
        }


def _parse_mount(value: Any, label: str) -> LearnerMount:
    row = _object(value, _MOUNT_KEYS, label)
    return LearnerMount(
        host_path=_absolute_host_path(row["host_path"], f"{label}.host_path"),
        container_path=_container_path(row["container_path"], f"{label}.container_path"),
    )


def _parse_resource(value: Any, label: str) -> LearnerResourceMount:
    row = _object(value, _MOUNT_KEYS | {"read_only"}, label)
    if row["read_only"] is not True:
        raise LearnerDeploymentError(f"Learner profile {label} must be read-only")
    mount = _parse_mount({"host_path": row["host_path"], "container_path": row["container_path"]}, label)
    return LearnerResourceMount(mount.host_path, mount.container_path, read_only=True)


def _parse_ssh(value: Any) -> LearnerSSHSettings | None:
    if value is None:
        return None
    row = _object(value, _SSH_KEYS, "ssh")
    if not isinstance(row["enabled"], bool):
        raise LearnerDeploymentError("Learner profile ssh.enabled must be boolean")
    if not row["enabled"]:
        if any(item not in ("", None, 0) for key, item in row.items() if key != "enabled"):
            raise LearnerDeploymentError("disabled SSH profile must not carry inactive targets or credentials")
        return None
    for key in ("central_host", "central_user", "central_host_public_key",
                "administrator_public_key", "local_ssh_user"):
        _string(row[key], f"ssh.{key}")
    for key in ("central_ssh_port", "remote_api_port", "remote_ssh_port", "local_listener_port"):
        _port(row[key], f"ssh.{key}")
    if row["remote_api_port"] == row["remote_ssh_port"]:
        raise LearnerDeploymentError("SSH remote API and SSH ports must be distinct")
    try:
        listener = ipaddress.ip_address(_string(row["local_listener_address"], "ssh.local_listener_address"))
    except ValueError:
        raise LearnerDeploymentError("SSH local listener address must be an IP literal") from None
    if not listener.is_loopback:
        raise LearnerDeploymentError("SSH local listener must bind to loopback")
    if not _SSH_KEY.fullmatch(_string(row["central_host_public_key"], "ssh.central_host_public_key")):
        raise LearnerDeploymentError("SSH central host public key must be an ssh-ed25519 or ssh-rsa key")
    if not _SSH_KEY.fullmatch(_string(row["administrator_public_key"], "ssh.administrator_public_key")):
        raise LearnerDeploymentError("SSH administrator public key must be an ssh-ed25519 or ssh-rsa key")
    return LearnerSSHSettings(
        enabled=True,
        central_host=row["central_host"],
        central_user=row["central_user"],
        central_ssh_port=row["central_ssh_port"],
        central_host_public_key=row["central_host_public_key"],
        remote_api_port=row["remote_api_port"],
        remote_ssh_port=row["remote_ssh_port"],
        local_listener_address=str(listener),
        local_listener_port=row["local_listener_port"],
        local_ssh_user=row["local_ssh_user"],
        administrator_public_key=row["administrator_public_key"],
        client_private_key_host_path=_absolute_host_path(
            row["client_private_key_host_path"], "ssh.client_private_key_host_path",
        ),
    )


def load_learner_deployment(
    path: str | Path | Mapping[str, Any],
    *,
    profile_path: str | Path | None = None,
    allow_gpu_autodetect: bool = False,
    allow_rootless_container_user: bool = False,
) -> LearnerDeployment:
    """Load a strict profile mapping/file without importing model or GPU packages."""
    if type(allow_gpu_autodetect) is not bool or type(allow_rootless_container_user) is not bool:
        raise ValueError("profile loader options must be bool")
    if isinstance(path, Mapping):
        raw = dict(path)
        selected_profile_path = (Path(profile_path).expanduser().resolve(strict=False)
                                 if profile_path is not None else Path.cwd() / "learner-deployment.json")
    else:
        if profile_path is not None:
            raise ValueError("profile_path is only used with an in-memory profile mapping")
        selected_profile_path = Path(path).expanduser().resolve(strict=True)
        try:
            profile_stat = selected_profile_path.lstat()
            if not stat.S_ISREG(profile_stat.st_mode):
                raise LearnerDeploymentError("Learner profile must be a regular file")
            raw = json.loads(
                selected_profile_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            if isinstance(exc, LearnerDeploymentError):
                raise
            raise LearnerDeploymentError("Learner profile JSON cannot be read") from None
    top = _object(raw, _TOP_KEYS, "root")
    if top["schema_version"] != PROFILE_SCHEMA:
        raise LearnerDeploymentError("Learner profile schema_version is unsupported")
    deployment_id = _string(top["deployment_id"], "deployment_id")
    container_name = _string(top["container_name"], "container_name")
    if not _SAFE_ID.fullmatch(deployment_id) or not _SAFE_ID.fullmatch(container_name):
        raise LearnerDeploymentError("Learner profile deployment and container names contain unsafe characters")
    image_id = _string(top["image_id"], "image_id")
    if not _IMAGE_ID.fullmatch(image_id):
        raise LearnerDeploymentError("Learner profile image_id must be an exact sha256 image ID")
    container_user = _string(top["container_user"], "container_user")
    user_match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", container_user)
    if user_match is None and not (allow_rootless_container_user and container_user == "0:0"):
        raise LearnerDeploymentError("Learner profile container_user must be a non-root numeric uid:gid pair")
    if top["container_workdir"] != CONTAINER_WORKDIR:
        raise LearnerDeploymentError("Learner profile container_workdir must be /work")

    gpu_rows = top["gpu_uuids"]
    if not isinstance(gpu_rows, list) or any(
            not isinstance(value, str) or not _GPU_UUID.fullmatch(value) for value in gpu_rows):
        raise LearnerDeploymentError("Learner profile gpu_uuids must be exact GPU UUID strings")
    if not gpu_rows and not allow_gpu_autodetect:
        raise LearnerDeploymentError("Learner profile requires exact GPU UUIDs before check or start")
    if len(set(gpu_rows)) != len(gpu_rows):
        raise LearnerDeploymentError("Learner profile gpu_uuids contains duplicates")

    publish = _object(top["publish"], _PUBLISH_KEYS, "publish")
    address_text = _string(publish["host_address"], "publish.host_address")
    try:
        address = ipaddress.ip_address(address_text)
    except ValueError:
        raise LearnerDeploymentError("Learner profile publish.host_address must be an IP literal") from None
    if not (address.is_private or address.is_loopback) or address.is_unspecified:
        raise LearnerDeploymentError(
            "Learner profile publish.host_address must be a concrete loopback or private IP address"
        )
    host_port = _port(publish["host_port"], "publish.host_port")
    container_port = _port(publish["container_port"], "publish.container_port")

    mounts_raw = _object(top["mounts"], _MOUNT_NAMES, "mounts")
    mounts = {name: _parse_mount(mounts_raw[name], f"mounts.{name}") for name in _MOUNT_NAMES}
    credentials_raw = top["credentials"]
    if not isinstance(credentials_raw, dict) or not credentials_raw:
        raise LearnerDeploymentError("Learner profile credentials must be a non-empty env-name mapping")
    credentials: dict[str, LearnerCredential] = {}
    for name, value in credentials_raw.items():
        if (not isinstance(name, str) or not _ENV_NAME.fullmatch(name)
                or name in _FORBIDDEN_ENV_NAMES
                or name.startswith("CUDA_")):
            raise LearnerDeploymentError(f"Learner profile credential environment name is not allowed: {name}")
        credential = _parse_mount(value, f"credentials.{name}")
        expected_secret_path = f"/run/secrets/{name}"
        if credential.container_path != expected_secret_path:
            raise LearnerDeploymentError(
                f"Learner credential container_path must be {expected_secret_path}"
            )
        credentials[name] = LearnerCredential(
            env_name=name, host_path=credential.host_path,
            container_path=credential.container_path,
        )

    resources_raw = top["resources"]
    if not isinstance(resources_raw, list):
        raise LearnerDeploymentError("Learner profile resources must be a list")
    resources = tuple(_parse_resource(row, f"resources[{index}]")
                      for index, row in enumerate(resources_raw))
    ssh = _parse_ssh(top["ssh"])
    if ssh is not None and ssh.remote_api_port != host_port:
        raise LearnerDeploymentError("SSH remote_api_port must match publish.host_port")

    all_mounts = [*mounts.values(), *credentials.values(), *resources]
    destinations = [PurePosixPath(mount.container_path) for mount in all_mounts]
    destinations.append(PurePosixPath(PROFILE_CONTAINER_PATH))
    for index, destination in enumerate(destinations):
        for other in destinations[index + 1:]:
            if destination == other or destination in other.parents or other in destination.parents:
                raise LearnerDeploymentError("Learner profile container mount destinations overlap")

    return LearnerDeployment(
        deployment_id=deployment_id,
        container_name=container_name,
        image_id=image_id,
        container_user=container_user,
        container_workdir=CONTAINER_WORKDIR,
        gpu_uuids=tuple(gpu_rows),
        publish=LearnerPublishSettings(address_text, host_port, container_port),
        runtime_config=mounts["runtime_config"],
        model=mounts["model"],
        dataset=mounts["dataset"],
        state=mounts["state"],
        credentials=credentials,
        resources=resources,
        ssh=ssh,
        profile_path=selected_profile_path,
    )


def _path_has_symlink(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            return True
    return False


def _check_mount_tree(path: Path, *, label: str) -> None:
    """Allow only regular files and directories anywhere in a bind source."""
    try:
        root_info = path.lstat()
    except OSError:
        raise LearnerDeploymentError(f"Learner {label} path cannot be inspected") from None
    if stat.S_ISREG(root_info.st_mode):
        return
    if not stat.S_ISDIR(root_info.st_mode):
        raise LearnerDeploymentError(
            f"Learner {label} bind source must contain only regular files and directories"
        )
    pending = [path]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(metadata.st_mode):
                        pending.append(Path(entry.path))
                    elif not stat.S_ISREG(metadata.st_mode):
                        raise LearnerDeploymentError(
                            f"Learner {label} bind source contains a symbolic link, socket, or special file"
                        )
    except LearnerDeploymentError:
        raise
    except OSError:
        raise LearnerDeploymentError(f"Learner {label} tree cannot be inspected safely") from None


def _check_host_source(path: Path, *, label: str, required: bool,
                       expected_kind: str | None = None, private_file: bool = False,
                       uid: int | None = None, owner_required: bool = True) -> Path:
    path = Path(path)
    if path == Path("/") or path == Path("/run"):
        raise LearnerDeploymentError(f"Learner {label} uses a protected host path")
    for protected in _PROTECTED_HOST_ROOTS:
        if path == protected or protected in path.parents:
            raise LearnerDeploymentError(f"Learner {label} uses a protected host path")
    if path.name in {"docker.sock", "containerd.sock"}:
        raise LearnerDeploymentError(f"Learner {label} cannot mount a container-engine socket")
    if _path_has_symlink(path):
        raise LearnerDeploymentError(f"Learner {label} path cannot contain symbolic links")
    if not path.exists():
        if required:
            raise LearnerDeploymentError(f"Learner {label} path does not exist: {path}")
        if not path.parent.exists():
            raise LearnerDeploymentError(f"Learner {label} parent directory must exist before start")
        existing = path.parent
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not existing.is_dir() or _path_has_symlink(existing):
            raise LearnerDeploymentError(f"Learner {label} parent path is not safe")
        return path
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError:
        raise LearnerDeploymentError(f"Learner {label} path cannot be inspected") from None
    if not path.is_absolute() or resolved != path or stat.S_ISLNK(info.st_mode):
        raise LearnerDeploymentError(f"Learner {label} path must be canonical and non-symlink")
    if expected_kind == "file" and not stat.S_ISREG(info.st_mode):
        raise LearnerDeploymentError(f"Learner {label} must be a regular file")
    if expected_kind == "directory" and not stat.S_ISDIR(info.st_mode):
        raise LearnerDeploymentError(f"Learner {label} must be a directory")
    if private_file:
        expected_uid = os.getuid() if uid is None else uid
        if (not stat.S_ISREG(info.st_mode) or (owner_required and info.st_uid != expected_uid)
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            requirement = "private owned regular file" if owner_required else "private regular file"
            raise LearnerDeploymentError(f"Learner {label} must be a {requirement} with mode 0600")
    _check_mount_tree(resolved, label=label)
    return resolved


def _runtime_config(profile: LearnerDeployment):
    try:
        from ...runtime.config import load_config
        return load_config(profile.runtime_config.host_path)
    except Exception as exc:
        raise LearnerDeploymentError(f"Learner runtime config is invalid ({type(exc).__name__})") from exc


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def _inside_container_path(value: str, root: str) -> bool:
    path, parent = PurePosixPath(value), PurePosixPath(root)
    return path == parent or parent in path.parents


def check_learner_profile(
    profile: LearnerDeployment | str | Path | Mapping[str, Any],
    *,
    require_assets: bool = True,
    allow_gpu_autodetect: bool = False,
    allow_rootless_container_user: bool = False,
    host_uid: int | None = None,
    host_gid: int | None = None,
) -> LearnerDeployment:
    """Validate static profile safety and its actual RuntimeConfig consumer contract.

    This check is read-only. It never creates directories or credential files.
    """
    if (type(require_assets) is not bool or type(allow_gpu_autodetect) is not bool
            or type(allow_rootless_container_user) is not bool):
        raise ValueError("profile check options must be bool")
    deployment = (profile if isinstance(profile, LearnerDeployment)
                  else load_learner_deployment(
                      profile,
                      allow_gpu_autodetect=allow_gpu_autodetect,
                      allow_rootless_container_user=allow_rootless_container_user,
                  ))
    uid = int(deployment.container_user.split(":", 1)[0])
    gid = int(deployment.container_user.split(":", 1)[1])
    if (uid == 0 or gid == 0) and (uid, gid) != (0, 0):
        raise LearnerDeploymentError("Learner profile root container identity must be exactly 0:0")
    if (uid, gid) == (0, 0):
        if not allow_rootless_container_user:
            raise LearnerDeploymentError("root container identity is allowed only for verified rootless current-user mode")
        if host_uid is None and host_gid is None:
            mapped_uid, mapped_gid = os.getuid(), os.getgid()
        else:
            mapped_uid, mapped_gid = host_uid, host_gid
            if (isinstance(mapped_uid, bool) or not isinstance(mapped_uid, int) or mapped_uid <= 0
                    or isinstance(mapped_gid, bool) or not isinstance(mapped_gid, int) or mapped_gid < 0):
                raise LearnerDeploymentError("rootless container mapping must identify a non-root host account")
            if require_assets and (mapped_uid != os.getuid() or mapped_gid != os.getgid()):
                raise LearnerDeploymentError("rootless container mapping must match the invoking host account")
    else:
        mapped_uid = uid if host_uid is None else host_uid
        mapped_gid = gid if host_gid is None else host_gid
        if (isinstance(mapped_uid, bool) or not isinstance(mapped_uid, int) or mapped_uid < 1
                or isinstance(mapped_gid, bool) or not isinstance(mapped_gid, int) or mapped_gid < 1
                or (mapped_uid, mapped_gid) != (uid, gid)):
            raise LearnerDeploymentError("non-root container identity must match its host UID/GID")
    paths = (
        (deployment.runtime_config, "runtime config", "file", True),
        (deployment.model, "model", "directory", True),
        (deployment.dataset, "dataset", "directory", True),
        (deployment.state, "state", "directory", False),
    )
    resolved: dict[str, Path] = {}
    for mount, label, kind, required in paths:
        resolved[label] = _check_host_source(
            mount.host_path, label=label, required=require_assets and required,
            expected_kind=kind,
        )
    if require_assets:
        state_path = deployment.state.host_path
        if state_path.exists():
            metadata = state_path.stat()
            if metadata.st_uid != mapped_uid or metadata.st_gid != mapped_gid:
                raise LearnerDeploymentError("Learner state directory owner must match container_user")
        elif (os.getuid(), os.getgid()) != (mapped_uid, mapped_gid):
            raise LearnerDeploymentError(
                "a missing Learner state directory can only be created when container_user matches the caller"
            )
    for name, mount in deployment.credentials.items():
        resolved[f"credential {name}"] = _check_host_source(
            mount.host_path, label=f"credential {name}", required=require_assets,
            expected_kind="file", private_file=True, uid=mapped_uid,
            owner_required=require_assets,
        )
    for index, mount in enumerate(deployment.resources):
        resolved[f"resource {index}"] = _check_host_source(
            mount.host_path, label=f"resource {index}", required=require_assets,
        )
    if deployment.ssh is not None:
        _check_host_source(
            deployment.ssh.client_private_key_host_path,
            label="SSH client key", required=require_assets, expected_kind="file",
            private_file=True, uid=os.geteuid(), owner_required=require_assets,
        )
        if require_assets:
            try:
                import pwd
                account = pwd.getpwnam(deployment.ssh.local_ssh_user)
            except (KeyError, ImportError):
                raise LearnerDeploymentError("Learner SSH local_ssh_user must be an existing account") from None
            if not account.pw_shell or Path(account.pw_shell).name in {"nologin", "false"}:
                raise LearnerDeploymentError("Learner SSH local_ssh_user must have a usable shell")

    # Directory source mounts must not overlap on the host; one source bound to
    # multiple destinations can expose checkpoints or assets across roles.
    source_paths = [(name, path) for name, path in resolved.items()
                    if path.exists() and path.is_dir()]
    for index, (name, source) in enumerate(source_paths):
        for other_name, other in source_paths[index + 1:]:
            if source == other or source in other.parents or other in source.parents:
                raise LearnerDeploymentError(f"Learner host mounts overlap: {name} and {other_name}")

    config = _runtime_config(deployment)
    if any(worker.scope == "local" for worker in config.workers):
        raise LearnerDeploymentError("Learner container config must use only external service or Gateway Workers")
    if not config.workers:
        raise LearnerDeploymentError("Learner container config requires at least one external Worker")
    if (Path(config.model_path) != Path(deployment.model.container_path)
            or Path(config.worker_inference.model_path) != Path(deployment.model.container_path)):
        raise LearnerDeploymentError("runtime model paths must match the profile model container_path")
    if Path(config.dataset_dir) != Path(deployment.dataset.container_path):
        raise LearnerDeploymentError("runtime dataset_dir must match the profile dataset container_path")
    if not any(_inside_container_path(str(config.contract), mount.container_path)
               for mount in deployment.resources):
        raise LearnerDeploymentError("ARC contract must be declared in a readonly resources[] mount")
    if not _inside_container_path(config.run_dir, deployment.state.container_path):
        raise LearnerDeploymentError("runtime run_dir must be inside the profile state container_path")
    if config.learner.host != "0.0.0.0":
        raise LearnerDeploymentError(
            "portable runtime learner.host must explicitly be 0.0.0.0 for Docker-published access"
        )
    if config.learner.port != deployment.publish.container_port:
        raise LearnerDeploymentError("runtime Learner port must match publish.container_port")
    if config.learner.gpus != tuple(range(len(deployment.gpu_uuids))):
        if deployment.gpu_uuids or not allow_gpu_autodetect:
            raise LearnerDeploymentError("runtime learner.gpus must be logical indices 0..N-1 for gpu_uuids")
    if config.quantization.mode == "torchao" and config.learner.parallelism not in {"single", "tp", "ddp", "fsdp2"}:
        raise LearnerDeploymentError("Native Learner profile has an unsupported parallelism mode")
    expected_credentials = {"ARC_ENV_TOKEN"}
    expected_credentials.update(
        worker.auth_token_env for worker in config.workers
        if worker.scope in {"service", "gateway"}
    )
    if config.bepa.enabled:
        expected_credentials.update({config.bepa.teacher.token_env, config.bepa.teacher.api_key_env})
    if set(deployment.credentials) != expected_credentials:
        raise LearnerDeploymentError(
            "profile credentials must exactly match ARC, external Worker, and enabled BEPA env names"
        )
    for worker in config.workers:
        if worker.scope not in {"service", "gateway"}:
            continue
        hostname = urlsplit(worker.endpoint).hostname
        if not hostname:
            raise LearnerDeploymentError(f"external Worker {worker.id} endpoint must have a reachable host")
        if hostname.casefold() == "localhost":
            raise LearnerDeploymentError(
                f"external Worker {worker.id} endpoint cannot use host loopback from inside the Learner container"
            )
        try:
            endpoint_address = ipaddress.ip_address(hostname)
        except ValueError:
            endpoint_address = None
        if endpoint_address is not None and (endpoint_address.is_loopback or endpoint_address.is_unspecified):
            raise LearnerDeploymentError(
                f"external Worker {worker.id} endpoint cannot use loopback or unspecified host IP from inside the Learner container"
            )
    if config.bepa.enabled and not _inside_container_path(
            config.bepa.state_dir, deployment.state.container_path):
        raise LearnerDeploymentError("BEPA state_dir must be inside the profile state container_path")
    if config.evaluation.enabled:
        required_resources = [config.evaluation.manifest_path, config.evaluation.sft_exclusion_proof]
        if config.evaluation.initial_count_proof:
            required_resources.append(config.evaluation.initial_count_proof)
        for source_path in required_resources:
            if not any(_inside_container_path(source_path, mount.container_path)
                       for mount in deployment.resources):
                raise LearnerDeploymentError(
                    "evaluation manifests and proofs must be declared in readonly resources[] mounts"
                )
        if config.evaluation.state_dir and not _inside_container_path(
                config.evaluation.state_dir, deployment.state.container_path):
            raise LearnerDeploymentError("evaluation state_dir must be inside the profile state container_path")
    return deployment


def resolve_gpu_uuids(
    profile: LearnerDeployment,
    detected_uuids: Sequence[str],
) -> tuple[str, ...]:
    """Resolve an install-time empty UUID list only when GPU count matches config."""
    detected = tuple(detected_uuids)
    if any(not isinstance(value, str) or not _GPU_UUID.fullmatch(value) for value in detected):
        raise LearnerDeploymentError("GPU inventory contains a malformed UUID")
    if len(set(detected)) != len(detected) or not detected:
        raise LearnerDeploymentError("GPU autodetection must find distinct visible GPUs")
    if profile.gpu_uuids:
        if detected != profile.gpu_uuids:
            raise LearnerDeploymentError("visible GPU UUID order differs from the explicit profile selection")
        return profile.gpu_uuids
    config = _runtime_config(profile)
    if len(config.learner.gpus) != len(detected):
        raise LearnerDeploymentError("auto-detected GPU count must match the logical Learner GPU count")
    if tuple(config.learner.gpus) != tuple(range(len(detected))):
        raise LearnerDeploymentError("runtime learner.gpus must be logical indices 0..N-1 for autodetection")
    return detected
