"""Read-only Ubuntu readiness checks and explicit Learner host installation."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .config import (
    LearnerDeployment,
    LearnerDeploymentError,
    check_learner_profile,
    load_learner_deployment,
)
from .ssh import LearnerSSHError, write_learner_ssh_state

GiB = 1024 ** 3
MIN_STATE_FREE_BYTES = 10 * GiB
MIN_SYSTEM_FREE_BYTES = 4 * GiB
BASE_HOST_PACKAGES = (
    "ca-certificates", "gnupg", "openssh-client", "docker.io",
    "docker-compose-v2", "nvidia-container-toolkit",
)


class LearnerHostError(RuntimeError):
    """Expected, user-safe error while preparing an Ubuntu Learner host."""


def _cluster_host_module():
    """Load the heavier shared Ubuntu helper only in the prepared host venv."""
    try:
        return importlib.import_module("qwen_rl.cluster.host")
    except ImportError as exc:
        raise LearnerHostError(
            "Learner host-control dependencies are unavailable; run the verified per-install venv bootstrap first"
        ) from exc


@dataclass(frozen=True)
class LearnerHostPrerequisites:
    status: Literal["ready", "needs_driver", "needs_reboot", "needs_host_packages"]
    needs_driver: bool
    needs_reboot: bool
    needs_host_packages: bool
    driver_reason: str | None
    selected_gpu_uuids: tuple[str, ...]
    available_gpu_uuids: tuple[str, ...]
    missing_gpu_uuids: tuple[str, ...]
    missing_packages: tuple[str, ...]
    ubuntu_version: str | None
    architecture: str | None
    docker_runtime: str
    free_bytes_by_mount: tuple[tuple[str, int], ...]

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        if mode not in {"python", "json"}:
            raise ValueError("mode must be 'python' or 'json'")
        return {
            "status": self.status,
            "needs_driver": self.needs_driver,
            "needs_reboot": self.needs_reboot,
            "needs_host_packages": self.needs_host_packages,
            "driver_reason": self.driver_reason,
            "selected_gpu_uuids": list(self.selected_gpu_uuids),
            "available_gpu_uuids": list(self.available_gpu_uuids),
            "missing_gpu_uuids": list(self.missing_gpu_uuids),
            "missing_packages": list(self.missing_packages),
            "ubuntu_version": self.ubuntu_version,
            "architecture": self.architecture,
            "docker_runtime": self.docker_runtime,
            "free_bytes_by_mount": dict(self.free_bytes_by_mount),
        }


@dataclass(frozen=True)
class LearnerHostInstallReceipt:
    status: Literal["started", "staged", "needs_driver", "needs_reboot", "needs_host_packages"]
    deployment_id: str
    service_unit: str | None
    ssh_units: tuple[str, ...]
    service_started: bool
    image_id: str
    installed_packages: tuple[str, ...]
    prerequisites: LearnerHostPrerequisites

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        if mode not in {"python", "json"}:
            raise ValueError("mode must be 'python' or 'json'")
        return {
            "status": self.status,
            "deployment_id": self.deployment_id,
            "service_unit": self.service_unit,
            "ssh_units": list(self.ssh_units),
            "service_started": self.service_started,
            "image_id": self.image_id,
            "installed_packages": list(self.installed_packages),
            "prerequisites": self.prerequisites.model_dump(mode=mode),
        }


def _run(
    argv: list[str], *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 120.0, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    if not argv or any(not isinstance(part, str) or "\x00" in part for part in argv):
        raise LearnerHostError("invalid host command arguments")
    try:
        kwargs = {"check": False, "capture_output": True, "text": True, "timeout": timeout}
        if env is not None:
            kwargs["env"] = env
        result = runner(argv, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LearnerHostError(f"could not run {Path(argv[0]).name}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        suffix = f": {detail[-1][:240]}" if detail else ""
        raise LearnerHostError(f"{Path(argv[0]).name} exited with status {result.returncode}{suffix}")
    return result


def _nearest_existing(path: Path) -> Path:
    candidate = Path(path)
    if candidate.exists() and not candidate.is_dir():
        candidate = candidate.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir() or candidate.is_symlink():
        raise LearnerHostError("Learner mount filesystem anchor is not a safe directory")
    return candidate


def _free_bytes(path: Path) -> int:
    anchor = _nearest_existing(path)
    try:
        usage = os.statvfs(anchor)
    except OSError as exc:
        raise LearnerHostError("could not inspect free space for a Learner mount") from exc
    return int(usage.f_bavail * usage.f_frsize)


def _container_access(path: Path, *, uid: int, gid: int, label: str) -> None:
    """Prove access to the bind-mount root and its contents from numeric mode bits.

    Docker resolves host ancestors as its daemon UID before creating a bind
    mount; the container process sees the mounted inode directly, so only the
    source root and descendants need container-UID read/search permissions.
    """
    path = Path(path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LearnerHostError(f"{label} bind-mount root is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise LearnerHostError(f"{label} cannot contain symbolic links")
    owner_bits = (stat.S_IMODE(metadata.st_mode) >> 6) & 0o7
    group_bits = (stat.S_IMODE(metadata.st_mode) >> 3) & 0o7
    other_bits = stat.S_IMODE(metadata.st_mode) & 0o7
    bits = owner_bits if metadata.st_uid == uid else group_bits if metadata.st_gid == gid else other_bits
    needed = 0o5 if stat.S_ISDIR(metadata.st_mode) else 0o4
    if bits & needed != needed:
        raise LearnerHostError(f"{label} permissions do not allow Learner UID/GID access")
    if not path.is_dir():
        return
    for parent, directories, files in os.walk(path, followlinks=False):
        parent_path = Path(parent)
        for name in [*directories, *files]:
            child = parent_path / name
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise LearnerHostError(f"{label} tree changed during permission validation") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise LearnerHostError(f"{label} cannot contain symbolic links")
            if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                raise LearnerHostError(f"{label} tree contains an unsupported filesystem object")
            owner_bits = (stat.S_IMODE(metadata.st_mode) >> 6) & 0o7
            group_bits = (stat.S_IMODE(metadata.st_mode) >> 3) & 0o7
            other_bits = stat.S_IMODE(metadata.st_mode) & 0o7
            bits = owner_bits if metadata.st_uid == uid else group_bits if metadata.st_gid == gid else other_bits
            needed = 0o5 if stat.S_ISDIR(metadata.st_mode) else 0o4
            if bits & needed != needed:
                raise LearnerHostError(f"{label} tree permissions do not allow Learner UID/GID access")


def _validate_profile_assets(
    deployment: LearnerDeployment,
    *,
    allow_rootless_container_user: bool = False,
    host_uid: int | None = None,
    host_gid: int | None = None,
) -> LearnerDeployment:
    try:
        return check_learner_profile(
            deployment,
            require_assets=True,
            allow_gpu_autodetect=False,
            allow_rootless_container_user=allow_rootless_container_user,
            host_uid=host_uid,
            host_gid=host_gid,
        )
    except (LearnerDeploymentError, OSError, ValueError) as exc:
        raise LearnerHostError(f"Learner profile or mounted assets are invalid: {exc}") from exc


def _host_asset_space(
    deployment: LearnerDeployment, *, validate_access: bool = True,
    host_uid: int | None = None, host_gid: int | None = None,
) -> tuple[tuple[str, int], ...]:
    uid_text, gid_text = deployment.container_user.split(":", 1)
    uid, gid = int(uid_text), int(gid_text)
    access_uid = uid if host_uid is None else host_uid
    access_gid = gid if host_gid is None else host_gid
    mounts = [
        ("runtime_config", deployment.runtime_config.host_path, "file"),
        ("model", deployment.model.host_path, "tree"),
        ("dataset", deployment.dataset.host_path, "tree"),
        ("state", deployment.state.host_path, "state"),
        *((f"resource_{index}", mount.host_path, "tree")
          for index, mount in enumerate(deployment.resources)),
    ]
    free: dict[str, int] = {}
    for name, raw_path, kind in mounts:
        path = Path(raw_path)
        if validate_access and kind != "state":
            _container_access(path, uid=access_uid, gid=access_gid, label=name)
        free[name] = _free_bytes(path)
        if kind == "state" and free[name] < MIN_STATE_FREE_BYTES:
            raise LearnerHostError("Learner state filesystem needs at least 10 GiB free before installation")
    system_free = _free_bytes(Path("/"))
    free["system"] = system_free
    if system_free < MIN_SYSTEM_FREE_BYTES:
        raise LearnerHostError("Ubuntu system filesystem needs at least 4 GiB free before package installation")
    return tuple(sorted(free.items()))


def _read_reboot_required(path: Path) -> bool:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise LearnerHostError("could not inspect the Ubuntu reboot-required marker") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise LearnerHostError("Ubuntu reboot-required marker is not a regular file")
    return True


def _status(*, needs_driver: bool, needs_reboot: bool, needs_host_packages: bool) -> Literal[
    "ready", "needs_driver", "needs_reboot", "needs_host_packages",
]:
    if needs_reboot:
        return "needs_reboot"
    if needs_driver:
        return "needs_driver"
    if needs_host_packages:
        return "needs_host_packages"
    return "ready"


def probe_learner_host(
    deployment: LearnerDeployment,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    os_release: Path = Path("/etc/os-release"),
    reboot_required_path: Path = Path("/var/run/reboot-required"),
) -> LearnerHostPrerequisites:
    """Read-only Ubuntu, exact-GPU, Docker/NVIDIA-runtime and disk check.

    It does not load a model, create directories, mutate package state, or
    choose another GPU when a configured UUID is absent.
    """
    deployment = _validate_profile_assets(deployment)
    free_bytes = _host_asset_space(deployment)
    cluster_host = _cluster_host_module()
    try:
        base = cluster_host.preflight_host(
            model_path=deployment.model.host_path,
            state_dir=deployment.state.host_path,
            runner=runner,
            os_release=os_release,
            minimum_free_bytes=0,
            require_runtime=True,
            require_gpu=True,
        )
    except (cluster_host.HostError, OSError, ValueError) as exc:
        raise LearnerHostError(f"Ubuntu host preflight failed: {exc}") from exc
    available = tuple(gpu.uuid for gpu in base.gpus)
    selected = tuple(deployment.gpu_uuids)
    missing = tuple(uuid for uuid in selected if uuid not in set(available))
    needs_driver = base.needs_driver or bool(missing)
    needs_host_packages = base.needs_host_packages
    missing_packages: list[str] = []
    if base.docker_runtime == "docker_missing":
        missing_packages.extend(("docker.io", "docker-compose-v2"))
    elif base.docker_runtime == "docker":
        missing_packages.append("nvidia-container-toolkit")
    elif base.docker_runtime == "daemon_unavailable":
        missing_packages.append("docker-daemon")
    if base.needs_host_packages and not missing_packages:
        missing_packages.append("docker-compose-or-nvidia-container-runtime")
    if deployment.ssh is not None and not deployment.ssh.local_listener_address.startswith("127."):
        # IPv6 loopback remains valid; explicit parsing in the SSH layer checks it.
        import ipaddress
        if not ipaddress.ip_address(deployment.ssh.local_listener_address).is_loopback:
            raise LearnerHostError("optional Learner sshd must bind only to loopback")
    if deployment.ssh is not None and not (
            Path(shutil.which("sshd") or "/usr/sbin/sshd").is_file()):
        needs_host_packages = True
        missing_packages.append("openssh-server")
    if not _package_installed("python3-venv", runner=runner):
        needs_host_packages = True
        missing_packages.append("python3-venv")
    needs_reboot = _read_reboot_required(reboot_required_path)
    return LearnerHostPrerequisites(
        status=_status(needs_driver=needs_driver, needs_reboot=needs_reboot,
                       needs_host_packages=needs_host_packages),
        needs_driver=needs_driver,
        needs_reboot=needs_reboot,
        needs_host_packages=needs_host_packages,
        driver_reason=("configured_gpu_uuid_not_present" if missing else base.driver_reason),
        selected_gpu_uuids=selected,
        available_gpu_uuids=available,
        missing_gpu_uuids=missing,
        missing_packages=tuple(dict.fromkeys(missing_packages)),
        ubuntu_version=base.ubuntu_version,
        architecture=base.architecture,
        docker_runtime=base.docker_runtime,
        free_bytes_by_mount=free_bytes,
    )


def _trusted_path(path: Path, *, owner_uid: int, kind: Literal["file", "directory"], label: str) -> None:
    try:
        metadata = Path(path).lstat()
    except OSError as exc:
        raise LearnerHostError(f"trusted Learner {label} is missing") from exc
    valid_kind = stat.S_ISREG(metadata.st_mode) if kind == "file" else stat.S_ISDIR(metadata.st_mode)
    if Path(path).is_symlink() or not valid_kind:
        raise LearnerHostError(f"trusted Learner {label} must be a regular non-symlink {kind}")
    if metadata.st_uid != owner_uid or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise LearnerHostError(f"trusted Learner {label} must be owned by the host installer and not group/world writable")


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in Path(path).parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise LearnerHostError(f"Learner {label} path component is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise LearnerHostError(f"Learner {label} path cannot traverse a symbolic link")


def _trusted_tree(root: Path, *, owner_uid: int, label: str) -> None:
    _trusted_path(root, owner_uid=owner_uid, kind="directory", label=label)
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            child = Path(parent) / name
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise LearnerHostError(f"trusted Learner {label} changed during validation") from exc
            valid_kind = stat.S_ISDIR(metadata.st_mode) if name in directories else stat.S_ISREG(metadata.st_mode)
            if child.is_symlink() or not valid_kind:
                raise LearnerHostError(f"trusted Learner {label} cannot contain symlinks or special files")
            if metadata.st_uid != owner_uid or stat.S_IMODE(metadata.st_mode) & 0o022:
                raise LearnerHostError(f"trusted Learner {label} contains user-writable execution files")


def _trusted_venv_tree(root: Path, *, owner_uid: int) -> None:
    """Validate a venv while allowing Python's internal ``lib64 -> lib`` link."""
    root = Path(root)
    _trusted_path(root, owner_uid=owner_uid, kind="directory", label="host control venv")
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            child = Path(parent) / name
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise LearnerHostError("trusted Learner host control venv changed during validation") from exc
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    resolved = child.resolve(strict=True)
                    resolved.relative_to(root)
                    target = resolved.stat()
                except (OSError, ValueError) as exc:
                    raise LearnerHostError("host control venv symlinks must resolve within the venv") from exc
                if not (stat.S_ISDIR(target.st_mode) or stat.S_ISREG(target.st_mode)):
                    raise LearnerHostError("host control venv symlinks must resolve to ordinary files")
                if target.st_uid != owner_uid or stat.S_IMODE(target.st_mode) & 0o022:
                    raise LearnerHostError("host control venv symlink target has unsafe owner or mode")
                continue
            valid_kind = stat.S_ISDIR(metadata.st_mode) if name in directories else stat.S_ISREG(metadata.st_mode)
            if not valid_kind:
                raise LearnerHostError("host control venv cannot contain special files")
            if metadata.st_uid != owner_uid or stat.S_IMODE(metadata.st_mode) & 0o022:
                raise LearnerHostError("host control venv contains user-writable execution files")


def _seal_new_venv(root: Path, *, owner_uid: int) -> None:
    """Remove group/world write bits Python deliberately adds to activation scripts."""
    root = Path(root)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise LearnerHostError("new host control venv is missing") from exc
    if not stat.S_ISDIR(root_info.st_mode) or root.is_symlink() or root_info.st_uid != owner_uid:
        raise LearnerHostError("new host control venv must be a regular host-owned directory")
    if stat.S_IMODE(root_info.st_mode) & 0o022:
        os.chmod(root, stat.S_IMODE(root_info.st_mode) & ~0o022)
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            child = Path(parent) / name
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise LearnerHostError("new host control venv changed during sealing") from exc
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    child.resolve(strict=True).relative_to(root)
                except (OSError, ValueError) as exc:
                    raise LearnerHostError("new host control venv symlinks must resolve within the venv") from exc
                continue
            if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                raise LearnerHostError("new host control venv cannot contain special files")
            if metadata.st_uid != owner_uid:
                raise LearnerHostError("new host control venv contains files owned by another user")
            mode = stat.S_IMODE(metadata.st_mode)
            if mode & 0o022:
                os.chmod(child, mode & ~0o022)


def _bundle_member(root: Path, path: Path, *, owner_uids: tuple[int, ...], label: str,
                   kind: Literal["file", "directory", "either"] = "either") -> None:
    root = Path(root)
    path = Path(path)
    _reject_symlink_chain(root, label="install root")
    _reject_symlink_chain(path, label=label)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise LearnerHostError(f"Learner {label} escapes the verified private bundle") from exc
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise LearnerHostError(f"verified Learner bundle member is missing: {label}") from exc
        if current.is_symlink():
            raise LearnerHostError(f"verified Learner bundle member traverses a symbolic link: {label}")
        final = index == len(relative.parts) - 1
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise LearnerHostError(f"verified Learner bundle parent is not a directory: {label}")
        if final:
            matches = (
                kind == "either" and (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode))
                or kind == "file" and stat.S_ISREG(metadata.st_mode)
                or kind == "directory" and stat.S_ISDIR(metadata.st_mode)
            )
            if not matches:
                raise LearnerHostError(f"verified Learner bundle member has the wrong kind: {label}")
        if metadata.st_uid not in owner_uids or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise LearnerHostError(f"verified Learner bundle member has unsafe ownership or write permissions: {label}")


def _validate_bundle_path_map(root: Path, deployment: LearnerDeployment, *, owner_uid: int) -> None:
    container_uid = int(deployment.container_user.split(":", 1)[0])
    container_owners = tuple({owner_uid, container_uid})
    expected: list[tuple[Path, Path, str, Literal["file", "directory", "either"], tuple[int, ...]]] = [
        (deployment.runtime_config.host_path, root / "runtime-config.json", "runtime config", "file",
         container_owners),
        (deployment.model.host_path, root / "model", "model", "directory", container_owners),
        (deployment.dataset.host_path, root / "dataset", "dataset", "directory", container_owners),
        (deployment.state.host_path, root / "state", "state", "directory", container_owners),
        *((credential.host_path, root / "credentials" / name, f"credential {name}", "file",
           container_owners)
          for name, credential in deployment.credentials.items()),
        *((resource.host_path, root / "resources" / f"{index:04d}", f"resource {index}", "either", container_owners)
          for index, resource in enumerate(deployment.resources)),
        (deployment.profile_path, root / "learner-deployment.json", "profile", "file", container_owners),
        (root / "image" / "learner.tar.gz", root / "image" / "learner.tar.gz", "image archive", "file", (owner_uid,)),
    ]
    if deployment.ssh is not None:
        expected.append((deployment.ssh.client_private_key_host_path,
                         root / "credentials" / "ssh-client-key", "SSH client key", "file", (owner_uid,)))
    for configured, bundled, label, kind, owner_uids in expected:
        if Path(configured) != bundled:
            raise LearnerHostError(f"Learner {label} path does not match its verified bundle member")
        _bundle_member(root, bundled, owner_uids=owner_uids, label=label, kind=kind)
        if label in {"runtime config", "SSH client key", "profile"} or label.startswith("credential "):
            info = bundled.lstat()
            if stat.S_IMODE(info.st_mode) != 0o600:
                raise LearnerHostError(f"Learner {label} must be a mode 0600 private file")


def _validate_runtime_json(deployment: LearnerDeployment) -> None:
    path = Path(deployment.runtime_config.host_path)
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            raise LearnerHostError("Learner runtime config is too large")
        value = json.loads(path.read_text(encoding="utf-8"))
    except LearnerHostError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LearnerHostError("Learner runtime config must be a readable JSON object") from exc
    if not isinstance(value, dict):
        raise LearnerHostError("Learner runtime config must be a JSON object")


def _validate_host_requirements(root: Path) -> Path:
    path = root / "source" / "deploy" / "learner" / "host-requirements.txt"
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LearnerHostError("pinned host-requirements.txt is missing or unreadable") from exc
    rows = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not rows or any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9.!+_-]*", row) is None
                       for row in rows):
        raise LearnerHostError("host requirements must contain only exact package==version pins")
    return path


def _early_gpu_reboot_gate(
    deployment: LearnerDeployment,
    *,
    runner: Callable[..., subprocess.CompletedProcess],
    os_release: Path,
    reboot_required_path: Path,
    free_bytes: tuple[tuple[str, int], ...],
) -> LearnerHostPrerequisites | None:
    try:
        release_fields = {}
        for line in Path(os_release).read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                release_fields[key] = value.strip().strip('"')
    except OSError as exc:
        raise LearnerHostError("cannot read Ubuntu release metadata") from exc
    if release_fields.get("ID") != "ubuntu" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise LearnerHostError("Learner host install requires x86_64 Ubuntu")
    needs_reboot = _read_reboot_required(reboot_required_path)
    nvidia_smi = shutil.which("nvidia-smi")
    available: tuple[str, ...] = ()
    reason: str | None = None
    if nvidia_smi is None:
        reason = "nvidia_driver_missing"
    else:
        try:
            result = runner([nvidia_smi, "--query-gpu=uuid", "--format=csv,noheader"],
                            check=False, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is None or result.returncode != 0:
            reason = "nvidia_driver_unavailable"
        else:
            available = tuple(line.strip() for line in (result.stdout or "").splitlines() if line.strip())
            if not available:
                reason = "no_nvidia_gpu_detected"
    selected = tuple(deployment.gpu_uuids)
    missing = tuple(uuid for uuid in selected if uuid not in set(available))
    needs_driver = bool(missing) or reason is not None
    return LearnerHostPrerequisites(
        status=_status(needs_driver=needs_driver, needs_reboot=needs_reboot, needs_host_packages=False),
        needs_driver=needs_driver,
        needs_reboot=needs_reboot,
        needs_host_packages=False,
        driver_reason=("configured_gpu_uuid_not_present" if missing else reason),
        selected_gpu_uuids=selected,
        available_gpu_uuids=available,
        missing_gpu_uuids=missing,
        missing_packages=(),
        ubuntu_version=release_fields.get("VERSION_ID"),
        architecture=platform.machine(),
        docker_runtime="not_checked",
        free_bytes_by_mount=free_bytes,
    )


def _validate_install_root(
    install_root: Path, deployment: LearnerDeployment, *, owner_uid: int,
    secure_parents: bool = False, allow_rootless_container_user: bool = False,
) -> Path:
    root = Path(install_root).expanduser().absolute()
    _reject_symlink_chain(root, label="install root")
    if secure_parents:
        for parent in root.parents:
            try:
                metadata = parent.lstat()
            except OSError as exc:
                raise LearnerHostError("Learner install-root parent cannot be inspected") from exc
            if (not stat.S_ISDIR(metadata.st_mode) or parent.is_symlink()
                    or metadata.st_uid != owner_uid or stat.S_IMODE(metadata.st_mode) & 0o022):
                raise LearnerHostError("production Learner install-root parents must be host-owned and not group/world writable")
            if parent == Path("/"):
                break
    _trusted_path(root, owner_uid=owner_uid, kind="directory", label="install root")
    profile_path = root / "learner-deployment.json"
    profile_uid = int(deployment.container_user.split(":", 1)[0])
    _bundle_member(root, profile_path, owner_uids=tuple({owner_uid, profile_uid}),
                   kind="file", label="deployment profile")
    try:
        persisted = load_learner_deployment(
            profile_path,
            allow_rootless_container_user=allow_rootless_container_user,
        )
    except LearnerDeploymentError as exc:
        raise LearnerHostError("Learner install-root profile does not match the strict schema") from exc
    if persisted.deployment_id != root.name:
        raise LearnerHostError("Learner deployment profile ID does not match its owned install-root directory")
    if persisted.model_dump(mode="json") != deployment.model_dump(mode="json"):
        raise LearnerHostError("passed Learner profile differs from the remapped install-root profile")
    _validate_bundle_path_map(root, deployment, owner_uid=owner_uid)
    _validate_runtime_json(deployment)
    _validate_host_requirements(root)
    _trusted_tree(root / "source", owner_uid=owner_uid, label="source tree")
    return root


def _validate_bundle_image(root: Path, *, owner_uid: int) -> tuple[Path, int]:
    archive = root / "image" / "learner.tar.gz"
    try:
        info = archive.lstat()
    except OSError as exc:
        raise LearnerHostError("verified Learner image archive is missing") from exc
    if not stat.S_ISREG(info.st_mode) or archive.is_symlink() or info.st_size <= 0:
        raise LearnerHostError("verified Learner image archive is not a non-empty regular file")
    if info.st_uid != owner_uid or stat.S_IMODE(info.st_mode) & 0o022:
        raise LearnerHostError("verified Learner image archive has unsafe owner or write permissions")
    return archive, int(info.st_size)


def _check_image_space(root: Path, archive_size: int) -> None:
    docker_root = Path("/var/lib/docker")
    free = _free_bytes(docker_root if docker_root.exists() else Path("/var/lib"))
    required = max(8 * GiB, archive_size * 2 + 4 * GiB)
    if free < required:
        raise LearnerHostError("Docker filesystem lacks the required space to load the pinned Learner image")


def _tree_metadata(path: Path) -> list[tuple[Path, os.stat_result]]:
    path = Path(path)
    if path.is_symlink():
        raise LearnerHostError("Learner state cannot contain a symbolic link at its root")
    if not path.exists():
        raise LearnerHostError("verified Learner container asset directory disappeared")
    root_info = path.lstat()
    if not stat.S_ISDIR(root_info.st_mode):
        raise LearnerHostError("Learner container asset root must be a regular directory")
    rows: list[tuple[Path, os.stat_result]] = [(path, root_info)]
    for parent, directories, files in os.walk(path, followlinks=False):
        parent_path = Path(parent)
        for name in [*directories, *files]:
            child = parent_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                raise LearnerHostError("Learner container asset contains an unsupported filesystem object")
            if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
                raise LearnerHostError("Learner container asset contains a hard-linked file")
            rows.append((child, metadata))
    return rows


def _container_asset_paths(deployment: LearnerDeployment) -> tuple[Path, ...]:
    return (
        Path(deployment.runtime_config.host_path),
        Path(deployment.model.host_path),
        Path(deployment.dataset.host_path),
        Path(deployment.state.host_path),
        *(Path(item.host_path) for item in deployment.credentials.values()),
        *(Path(item.host_path) for item in deployment.resources),
        Path(deployment.profile_path),
    )


def _adopt_bundle_assets(
    deployment: LearnerDeployment,
    *,
    install_root: Path,
    host_owner_uid: int,
    host_owner_gid: int,
) -> None:
    """One-time owner transition for verified bundle mounts and profile.

    A private host-control marker makes retries verification-only. If a process
    dies during first adoption, remaining root-owned leaves are finished while
    already container-owned leaves and their contents stay unchanged.
    """
    uid_text, gid_text = deployment.container_user.split(":", 1)
    uid, gid = int(uid_text), int(gid_text)
    control = _prepare_host_control_dir(deployment, owner_uid=host_owner_uid)
    marker = control / "l4-container-assets.json"
    profile_bytes = Path(deployment.profile_path).read_bytes()
    marker_bytes = (json.dumps({
        "deployment_id": deployment.deployment_id,
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "container_uid": uid,
        "container_gid": gid,
    }, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    already_adopted = False
    if marker.exists():
        try:
            metadata = marker.lstat()
            marker_contents = marker.read_bytes()
        except OSError as exc:
            raise LearnerHostError("Learner asset ownership marker cannot be read") from exc
        if (not stat.S_ISREG(metadata.st_mode) or marker.is_symlink() or metadata.st_uid != host_owner_uid
                or stat.S_IMODE(metadata.st_mode) != 0o600 or marker_contents != marker_bytes):
            raise LearnerHostError("Learner asset ownership marker is invalid; refusing to mutate mounted assets")
        already_adopted = True

    trees: list[list[tuple[Path, os.stat_result]]] = []
    for path in _container_asset_paths(deployment):
        rows = _tree_metadata(path) if path.is_dir() else [(path, path.lstat())]
        for item, metadata in rows:
            if stat.S_ISLNK(metadata.st_mode) or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                raise LearnerHostError("Learner container asset contains an unsupported filesystem object")
            if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
                raise LearnerHostError("Learner container asset contains a hard-linked file")
            if metadata.st_uid not in {host_owner_uid, uid}:
                raise LearnerHostError("Learner container asset has an unexpected owner")
            if metadata.st_gid not in {host_owner_gid, gid}:
                raise LearnerHostError("Learner container asset has an unexpected group")
            if stat.S_IMODE(metadata.st_mode) & 0o022:
                raise LearnerHostError("Learner container asset is group/world writable before adoption")
            if stat.S_IMODE(metadata.st_mode) & 0o6000:
                raise LearnerHostError("Learner container asset cannot carry setuid/setgid/sticky bits")
        trees.append(rows)

    if already_adopted:
        for rows in trees:
            for item, metadata in rows:
                if (metadata.st_uid != uid or metadata.st_gid != gid
                        or stat.S_IMODE(metadata.st_mode) & 0o077):
                    raise LearnerHostError("marked Learner container asset is no longer private and container-owned")
        return

    for rows in trees:
        for item, metadata in rows:
            new_mode = 0o700 if stat.S_ISDIR(metadata.st_mode) else 0o600
            if (metadata.st_uid, metadata.st_gid, stat.S_IMODE(metadata.st_mode)) == (uid, gid, new_mode):
                continue
            try:
                os.chown(item, uid, gid)
                os.chmod(item, new_mode)
            except OSError as exc:
                raise LearnerHostError("could not safely assign bundle assets to the Learner UID/GID") from exc
    descriptor, temporary_name = tempfile.mkstemp(prefix=".l4-container-assets.", dir=control)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(marker_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        os.chown(marker, host_owner_uid, -1)
        os.chmod(marker, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _prepare_container_files(deployment: LearnerDeployment) -> None:
    uid_text, gid_text = deployment.container_user.split(":", 1)
    uid, gid = int(uid_text), int(gid_text)
    for path in _container_asset_paths(deployment):
        rows = _tree_metadata(path) if path.is_dir() else [(path, path.lstat())]
        for _item, metadata in rows:
            if metadata.st_uid != uid or metadata.st_gid != gid or stat.S_IMODE(metadata.st_mode) & 0o077:
                raise LearnerHostError("Learner bind assets must be private and owned by the container UID/GID")


def _prepare_host_control_dir(deployment: LearnerDeployment, *, owner_uid: int) -> Path:
    """Create only the host-owned manager metadata directory, outside bind mounts."""
    path = Path(deployment.host_control_path)
    parent = path.parent
    parent_exists = parent.exists()
    if path.exists():
        for candidate in (parent, path):
            try:
                metadata = candidate.lstat()
            except OSError as exc:
                raise LearnerHostError("Learner host-control directory cannot be inspected") from exc
            if (not stat.S_ISDIR(metadata.st_mode) or candidate.is_symlink()
                    or metadata.st_uid != owner_uid or stat.S_IMODE(metadata.st_mode) & 0o077):
                raise LearnerHostError("existing Learner host-control directory has unsafe owner or permissions")
    else:
        try:
            if parent_exists:
                metadata = parent.lstat()
                if (not stat.S_ISDIR(metadata.st_mode) or parent.is_symlink()
                        or metadata.st_uid != owner_uid or stat.S_IMODE(metadata.st_mode) & 0o077):
                    raise LearnerHostError("existing Learner host-control parent has unsafe owner or permissions")
            else:
                parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if owner_uid != os.geteuid():
                    os.chown(parent, owner_uid, -1)
                os.chmod(parent, 0o700)
            path.mkdir(mode=0o700)
            if owner_uid != os.geteuid():
                os.chown(path, owner_uid, -1)
            os.chmod(path, 0o700)
        except OSError as exc:
            raise LearnerHostError("could not create private Learner host-control metadata directory") from exc
    return path


def _package_installed(package: str, *, runner: Callable[..., subprocess.CompletedProcess]) -> bool:
    dpkg = shutil.which("dpkg-query") or "/usr/bin/dpkg-query"
    try:
        result = runner([dpkg, "-W", "-f=${db:Status-Status}", package], check=False,
                        capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and (result.stdout or "").strip() == "installed"


def _install_python_venv(*, root: Path, runner: Callable[..., subprocess.CompletedProcess]) -> Path:
    interpreter = Path("/usr/bin/python3")
    if not interpreter.is_file():
        raise LearnerHostError("system Python 3 is unavailable for the Learner host control venv")
    venv_path = root / "venv"
    python = venv_path / "bin" / "python"
    pip = venv_path / "bin" / "pip"
    requirements = _validate_host_requirements(root)
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = venv_path / ".host-requirements.sha256"
    if venv_path.exists():
        if not python.is_file() or marker.is_symlink() or not marker.is_file():
            raise LearnerHostError("existing Learner venv is incomplete; refusing to replace it")
        _trusted_venv_tree(venv_path, owner_uid=os.geteuid())
        try:
            installed_digest = marker.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise LearnerHostError("existing Learner host requirements marker is unreadable") from exc
        if installed_digest != digest:
            raise LearnerHostError("existing Learner venv has different pinned requirements; refusing to replace it")
        return python
    trusted_env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root" if os.geteuid() == 0 else str(Path.home()),
        "PYTHONNOUSERSITE": "1",
        "LANG": "C.UTF-8",
    }
    previous_umask = os.umask(0o077)
    try:
        _run([str(interpreter), "-m", "venv", "--copies", str(venv_path)], runner=runner, timeout=180)
        _run([str(pip), "install", "--isolated", "--disable-pip-version-check", "--no-input",
              "--no-compile", "-r", str(requirements)], runner=runner, timeout=1800, env=trusted_env)
    finally:
        os.umask(previous_umask)
    _seal_new_venv(venv_path, owner_uid=os.geteuid())
    marker.write_text(digest + "\n", encoding="ascii")
    os.chmod(marker, 0o600)
    _trusted_venv_tree(venv_path, owner_uid=os.geteuid())
    return python


def _systemd_arg(value: str | Path) -> str:
    text = str(value)
    if not text or any(char in text for char in "\r\n\x00"):
        raise LearnerHostError("systemd arguments must be nonempty single-line values")
    if not any(char.isspace() or char in '\\"%$' for char in text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def _systemd_path(value: str | Path) -> str:
    """Escape a path directive value without adding ExecStart-style quotes."""
    text = str(value)
    if not Path(text).is_absolute() or any(char in text for char in "\r\n\x00"):
        raise LearnerHostError("systemd path directives require absolute single-line paths")
    escaped: list[str] = []
    for char in text:
        code = ord(char)
        if char == "%":
            escaped.append("%%")
        elif char == "\\" or code < 0x20 or char in {' ', '"'} or code == 0x7f:
            escaped.append(f"\\x{code:02x}")
        else:
            escaped.append(char)
    return "".join(escaped)


def _render_service_unit(root: Path, deployment: LearnerDeployment, python: Path) -> str:
    template = root / "source" / "deploy" / "learner" / "qwen-rl-learner@.service"
    if template.is_symlink() or not template.is_file():
        raise LearnerHostError("owned Learner systemd template is missing from the verified bundle")
    content = template.read_text(encoding="utf-8")
    profile_path = root / "learner-deployment.json"
    values = {
        "@LEARNER_DEPLOYMENT@": _systemd_arg(profile_path),
        "@LEARNER_DEPLOYMENT_PATH@": _systemd_path(profile_path),
        "@INSTALL_ROOT@": _systemd_path(root),
        "@HOST_CONTROL_PATH@": _systemd_path(deployment.host_control_path),
        "@SOURCE_ROOT@": _systemd_arg(root / "source"),
        "@VENV_PYTHON@": _systemd_arg(python),
    }
    for placeholder, replacement in values.items():
        content = content.replace(placeholder, replacement)
    if any(placeholder in content for placeholder in values):
        raise LearnerHostError("Learner systemd template contains an unresolved placeholder")
    return content


def _write_owned_file(path: Path, body: bytes, *, mode: int) -> None:
    cluster_host = _cluster_host_module()
    try:
        cluster_host.write_system_file(path, body, mode=mode)
    except cluster_host.HostError as exc:
        raise LearnerHostError(f"could not safely write owned Learner host file {Path(path).name}") from exc


def _install_systemd_unit(path: Path, body: str) -> None:
    path = Path(path)
    encoded = body.encode("utf-8")
    default_systemd_dir = Path("/etc/systemd/system").resolve(strict=False)
    expected_uid = 0 if path.parent.resolve(strict=False) == default_systemd_dir else os.getuid()
    if path.exists():
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise LearnerHostError("existing Learner systemd unit cannot be inspected") from exc
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise LearnerHostError("existing Learner systemd unit path is not a regular file")
        if path.read_bytes() != encoded:
            raise LearnerHostError("an existing Learner systemd unit differs; refusing to replace an active installation")
        if metadata.st_uid != expected_uid or stat.S_IMODE(metadata.st_mode) != 0o644:
            raise LearnerHostError("identical Learner systemd unit has unsafe owner or mode; refusing to trust it")
        return
    _write_owned_file(path, encoded, mode=0o644)
    written = path.lstat()
    if written.st_uid != expected_uid or stat.S_IMODE(written.st_mode) != 0o644:
        raise LearnerHostError("installed Learner systemd unit has unsafe owner or mode")


def _docker_argv(*arguments: str, context: str | None = None,
                 host: str | None = None) -> list[str]:
    docker = shutil.which("docker") or "/usr/bin/docker"
    if host is not None:
        return [docker, "--host", host, *arguments]
    if context is None:
        # The standalone privileged installer retains its established default.
        return [docker, "--host", "unix:///var/run/docker.sock", *arguments]
    if (not isinstance(context, str) or not context
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", context) is None):
        raise LearnerHostError("verified Docker context name is invalid")
    return [docker, "--context", context, *arguments]


def _docker_image_id(*, image_id: str, runner: Callable[..., subprocess.CompletedProcess],
                     docker_context: str | None = None, docker_host: str | None = None) -> str | None:
    try:
        result = runner(_docker_argv("image", "inspect", "--format={{.Id}}", image_id,
                                     context=docker_context, host=docker_host),
                        check=False, capture_output=True,
                        text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _load_pinned_image(
    *, image_id: str, archive: Path, runner: Callable[..., subprocess.CompletedProcess],
    docker_context: str | None = None, docker_host: str | None = None,
) -> None:
    before = _docker_image_id(image_id=image_id, runner=runner, docker_context=docker_context,
                              docker_host=docker_host)
    if before is None:
        _run(_docker_argv("load", "--input", str(archive), context=docker_context,
                          host=docker_host),
             runner=runner, timeout=7200)
    after = _docker_image_id(image_id=image_id, runner=runner, docker_context=docker_context,
                             docker_host=docker_host)
    if after != image_id:
        raise LearnerHostError("loaded Docker image does not match the profile's pinned image ID")


def _host_packages(*, deployment: LearnerDeployment,
                   runner: Callable[..., subprocess.CompletedProcess]) -> tuple[str, ...]:
    cluster_host = _cluster_host_module()
    packages = (*BASE_HOST_PACKAGES, *(("openssh-server",) if deployment.ssh is not None else ()))
    try:
        installed = cluster_host.install_host_packages(
            packages=packages,
            runner=runner,
            allow_package_changes=True,
        )
    except cluster_host.HostError as exc:
        raise LearnerHostError(f"explicit Ubuntu package installation failed: {exc}") from exc
    return tuple(installed)


def _install_python_venv_package(*, runner: Callable[..., subprocess.CompletedProcess]) -> None:
    if _package_installed("python3-venv", runner=runner):
        return
    apt = shutil.which("apt-get") or "/usr/bin/apt-get"
    _run([apt, "update"], runner=runner, timeout=900)
    _run([apt, "install", "--yes", "--no-install-recommends", "python3-venv"],
         runner=runner, timeout=900)


def _make_receipt(deployment: LearnerDeployment, prerequisites: LearnerHostPrerequisites,
                  installed: tuple[str, ...] = ()) -> LearnerHostInstallReceipt:
    return LearnerHostInstallReceipt(
        status=prerequisites.status,
        deployment_id=deployment.deployment_id,
        service_unit=None,
        ssh_units=(),
        service_started=False,
        image_id=deployment.image_id,
        installed_packages=installed,
        prerequisites=prerequisites,
    )


def _activate_host_venv_imports(root: Path) -> None:
    source_root = root / "source"
    site_dirs = sorted((root / "venv" / "lib").glob("python*/site-packages"))
    if not site_dirs:
        raise LearnerHostError("prepared Learner venv has no site-packages directory")
    sys.path[:0] = [str(site_dirs[-1]), str(source_root)]
    package = sys.modules.get("qwen_rl")
    source_package = source_root / "qwen_rl"
    if package is not None and source_package.is_dir():
        package.__path__ = [str(source_package)]
    importlib.invalidate_caches()


def _receipt_from_json(value: object) -> LearnerHostInstallReceipt:
    if not isinstance(value, dict) or not isinstance(value.get("prerequisites"), dict):
        raise LearnerHostError("Learner host child returned an invalid install receipt")
    raw = dict(value["prerequisites"])
    try:
        raw["selected_gpu_uuids"] = tuple(raw["selected_gpu_uuids"])
        raw["available_gpu_uuids"] = tuple(raw["available_gpu_uuids"])
        raw["missing_gpu_uuids"] = tuple(raw["missing_gpu_uuids"])
        raw["missing_packages"] = tuple(raw["missing_packages"])
        raw["free_bytes_by_mount"] = tuple(sorted(dict(raw["free_bytes_by_mount"]).items()))
        prerequisites = LearnerHostPrerequisites(**raw)
        return LearnerHostInstallReceipt(
            status=value["status"],
            deployment_id=value["deployment_id"],
            service_unit=value["service_unit"],
            ssh_units=tuple(value["ssh_units"]),
            service_started=value["service_started"],
            image_id=value["image_id"],
            installed_packages=tuple(value["installed_packages"]),
            prerequisites=prerequisites,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LearnerHostError("Learner host child returned an invalid install receipt") from exc


def _run_venv_host_stage(
    *,
    root: Path,
    systemd_dir: Path,
    allow_package_changes: bool,
    reboot_required_path: Path,
    runner: Callable[..., subprocess.CompletedProcess],
) -> LearnerHostInstallReceipt:
    python = root / "venv" / "bin" / "python"
    argv = [
        str(python), "-m", "qwen_rl.deployment.learner.host", "--install-after-venv",
        "--install-root", str(root), "--systemd-dir", str(systemd_dir),
        "--reboot-required-path", str(reboot_required_path),
        "--allow-package-changes", "true" if allow_package_changes else "false",
    ]
    environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root" if os.geteuid() == 0 else str(Path.home()),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(root / "source"),
        "LANG": "C.UTF-8",
    }
    try:
        result = runner(argv, check=False, capture_output=True, text=True, timeout=7200, env=environment)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LearnerHostError("could not re-enter the prepared Learner host venv") from exc
    if result.returncode != 0:
        raise LearnerHostError("Learner host venv setup failed; inspect local package and service logs")
    try:
        payload = json.loads(result.stdout or "")
    except json.JSONDecodeError as exc:
        raise LearnerHostError("Learner host venv returned an invalid install receipt") from exc
    return _receipt_from_json(payload)


def _prepare_after_venv(
    deployment: LearnerDeployment,
    *,
    install_root: Path,
    systemd_dir: Path,
    allow_package_changes: bool,
    runner: Callable[..., subprocess.CompletedProcess],
    reboot_required_path: Path,
) -> LearnerHostInstallReceipt:
    deployment = _validate_profile_assets(deployment)
    prerequisites = probe_learner_host(
        deployment,
        runner=runner,
        reboot_required_path=reboot_required_path,
    )
    installed: tuple[str, ...] = ()
    if prerequisites.needs_host_packages and allow_package_changes:
        installed = _host_packages(deployment=deployment, runner=runner)
        prerequisites = probe_learner_host(
            deployment,
            runner=runner,
            reboot_required_path=reboot_required_path,
        )
    if prerequisites.status != "ready":
        return _make_receipt(deployment, prerequisites, installed)

    _prepare_container_files(deployment)
    root = Path(install_root)
    python = root / "venv" / "bin" / "python"
    _load_pinned_image(image_id=deployment.image_id,
                       archive=root / "image" / "learner.tar.gz", runner=runner)

    production_systemd = Path(systemd_dir).resolve(strict=False) == Path("/etc/systemd/system").resolve(strict=False)
    owner_uid = 0 if production_systemd else os.getuid()
    _prepare_host_control_dir(deployment, owner_uid=owner_uid)
    unit_body = _render_service_unit(root, deployment, python)
    unit_name = f"qwen-rl-learner@{deployment.deployment_id}.service"
    _install_systemd_unit(Path(systemd_dir) / unit_name, unit_body)

    ssh_units: tuple[str, ...] = ()
    if deployment.ssh is not None:
        ssh_state_root = (Path("/var/lib/qwen-rl-learner-ssh") if production_systemd
                          else root.parent / ".learner-ssh-state")
        ssh_state = ssh_state_root / deployment.deployment_id
        try:
            ssh_setup = write_learner_ssh_state(deployment, state_dir=ssh_state, runner=runner)
        except LearnerSSHError as exc:
            raise LearnerHostError(f"optional Learner SSH preparation failed: {exc}") from exc
        sshd_unit_name = ssh_setup.sshd_unit_name
        tunnel_unit_name = ssh_setup.tunnel_unit_name
        _install_systemd_unit(Path(systemd_dir) / sshd_unit_name, ssh_setup.sshd_unit)
        _install_systemd_unit(Path(systemd_dir) / tunnel_unit_name, ssh_setup.tunnel_unit)
        ssh_units = (sshd_unit_name, tunnel_unit_name)

    if not production_systemd:
        return LearnerHostInstallReceipt(
            status="staged",
            deployment_id=deployment.deployment_id,
            service_unit=unit_name,
            ssh_units=ssh_units,
            service_started=False,
            image_id=deployment.image_id,
            installed_packages=installed,
            prerequisites=prerequisites,
        )
    systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
    _run([systemctl, "daemon-reload"], runner=runner, timeout=60)
    _run([systemctl, "enable", "--now", unit_name], runner=runner, timeout=180)
    for ssh_unit in ssh_units:
        _run([systemctl, "enable", "--now", ssh_unit], runner=runner, timeout=120)
    return LearnerHostInstallReceipt(
        status="started",
        deployment_id=deployment.deployment_id,
        service_unit=unit_name,
        ssh_units=ssh_units,
        service_started=True,
        image_id=deployment.image_id,
        installed_packages=installed,
        prerequisites=prerequisites,
    )


def _current_user_systemd_environment(account: Any, docker_context: str,
                                      docker_config: str | None, docker_host: str | None) -> dict[str, str]:
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(account.home),
        "LANG": "C.UTF-8",
        "XDG_RUNTIME_DIR": f"/run/user/{account.uid}",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{account.uid}/bus",
    }
    if docker_host is None:
        environment["DOCKER_CONTEXT"] = docker_context
    else:
        environment["DOCKER_HOST"] = docker_host
    if docker_config is not None:
        environment["DOCKER_CONFIG"] = docker_config
    return environment


def _validate_current_user_assets(deployment: LearnerDeployment, *, uid: int, gid: int) -> None:
    state_root = Path(deployment.state.host_path)
    if not state_root.exists():
        if state_root.is_symlink():
            raise LearnerHostError("Learner state path cannot be a symbolic link")
        try:
            state_root.mkdir(mode=0o700, parents=True)
        except OSError as exc:
            raise LearnerHostError("could not create private Learner state under the registered account") from exc
    for path in _container_asset_paths(deployment):
        rows = _tree_metadata(path) if path.is_dir() else [(path, path.lstat())]
        for item, metadata in rows:
            if metadata.st_uid != uid or metadata.st_gid != gid:
                raise LearnerHostError(
                    "Learner mounted assets must belong to the registered account; refusing cross-UID chown"
                )
            if stat.S_IMODE(metadata.st_mode) & 0o077 or stat.S_IMODE(metadata.st_mode) & 0o7000:
                raise LearnerHostError("Learner mounted assets must remain private and free of special mode bits")
            if stat.S_ISDIR(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) & 0o700 != 0o700:
                raise LearnerHostError("Learner asset directory is not usable by the registered account")
            if stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) & 0o400 == 0:
                raise LearnerHostError("Learner asset file is not readable by the registered account")


def _render_current_user_unit(
    *, deployment: LearnerDeployment, install_root: Path, python: Path,
    source_root: Path, home: Path, docker_context: str, docker_config: str | None,
    docker_host: str | None, allow_rootless_container_user: bool = False,
) -> str:
    profile_path = install_root / "learner-deployment.json"
    values = {
        "@PROFILE_PATH@": _systemd_path(profile_path),
        "@PROFILE_ARG@": _systemd_arg(profile_path),
        "@ROOT_PATH@": _systemd_path(install_root),
        "@PYTHON_ARG@": _systemd_arg(python),
        "@SOURCE_ENV@": _systemd_arg(f"PYTHONPATH={source_root}"),
        "@HOME_ENV@": _systemd_arg(f"HOME={home}"),
        "@ROOTLESS_OPTION@": "--fleet-rootless " if allow_rootless_container_user else "",
        "@DOCKER_CONTEXT_ENV@": _systemd_arg(f"DOCKER_CONTEXT={docker_context}"),
        "@CONTROL_PATH@": _systemd_path(deployment.host_control_path),
        "@STATE_PATH@": _systemd_path(deployment.state.host_path),
    }
    if docker_host is not None:
        values["@DOCKER_HOST_ENV@"] = _systemd_arg(f"DOCKER_HOST={docker_host}")
    body = "\n".join((
        "[Unit]",
        "Description=Qwen RL user-owned Learner deployment",
        "ConditionPathExists=@PROFILE_PATH@",
        "",
        "[Service]",
        "Type=oneshot",
        "UMask=0077",
        "WorkingDirectory=@ROOT_PATH@",
        "Environment=@HOME_ENV@",
        "Environment=PATH=/usr/local/bin:/usr/bin:/bin",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        "Environment=PYTHONNOUSERSITE=1",
        "Environment=@SOURCE_ENV@",
        ("Environment=@DOCKER_HOST_ENV@" if docker_host is not None
         else "Environment=@DOCKER_CONTEXT_ENV@"),
        ("UnsetEnvironment=DOCKER_CONTEXT" if docker_host is not None and docker_config is not None
         else ("UnsetEnvironment=DOCKER_CONTEXT DOCKER_CONFIG" if docker_host is not None
               else ("UnsetEnvironment=DOCKER_HOST" if docker_config is not None
                     else "UnsetEnvironment=DOCKER_HOST DOCKER_CONFIG"))),
        "ExecStart=@PYTHON_ARG@ -m qwen_rl learner @ROOTLESS_OPTION@--deployment @PROFILE_ARG@ start",
        "ExecStop=@PYTHON_ARG@ -m qwen_rl learner @ROOTLESS_OPTION@--deployment @PROFILE_ARG@ stop",
        "TimeoutStartSec=0",
        "TimeoutStopSec=180",
        "RemainAfterExit=yes",
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
        "ProtectHome=no",
        "PrivateTmp=yes",
        "ReadWritePaths=@CONTROL_PATH@ @STATE_PATH@",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ))
    if docker_config is not None:
        docker_config_env = f"Environment={_systemd_arg(f'DOCKER_CONFIG={docker_config}')}\n"
        docker_env = ("Environment=@DOCKER_HOST_ENV@\n" if docker_host is not None
                      else "Environment=@DOCKER_CONTEXT_ENV@\n")
        body = body.replace(docker_env, docker_env + docker_config_env)
    for placeholder, replacement in values.items():
        body = body.replace(placeholder, replacement)
    if any(token in body for token in values):
        raise LearnerHostError("current-user Learner unit has an unresolved placeholder")
    return body


def _install_current_user_unit(path: Path, body: str, *, owner_uid: int, owner_gid: int) -> None:
    path = Path(path)
    encoded = body.encode("utf-8")
    if path.exists() or path.is_symlink():
        try:
            info = path.lstat()
        except OSError as exc:
            raise LearnerHostError("existing user Learner unit cannot be inspected") from exc
        if (path.is_symlink() or not stat.S_ISREG(info.st_mode)
                or info.st_uid != owner_uid or info.st_gid != owner_gid
                or stat.S_IMODE(info.st_mode) != 0o644):
            raise LearnerHostError("existing user Learner unit has unsafe ownership or mode")
        if path.read_bytes() != encoded:
            raise LearnerHostError("existing user Learner unit differs; refusing to replace it")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
                or info.st_uid != owner_uid or info.st_gid != owner_gid
                or stat.S_IMODE(info.st_mode) != 0o644):
            raise LearnerHostError("installed user Learner unit does not have the registered owner")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_current_user_systemd_dir(path: Path, *, home: Path, owner_uid: int, owner_gid: int) -> Path:
    home = Path(home).absolute()
    path = Path(path).absolute()
    if path != home / ".config" / "systemd" / "user":
        raise LearnerHostError("user systemd directory differs from the registered account path")
    current = Path(home.anchor)
    for component in home.parts[1:]:
        current = current / component
        try:
            info = current.lstat()
        except OSError as exc:
            raise LearnerHostError("registered account HOME path is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise LearnerHostError("registered account HOME cannot traverse a symbolic link")
    home_info = home.lstat()
    if (home_info.st_uid != owner_uid or home_info.st_gid != owner_gid
            or stat.S_IMODE(home_info.st_mode) & 0o022):
        raise LearnerHostError("registered account HOME has unsafe ownership or permissions")
    current = home
    for component in (".config", "systemd", "user"):
        current = current / component
        if current.exists() or current.is_symlink():
            try:
                info = current.lstat()
            except OSError as exc:
                raise LearnerHostError("user systemd directory cannot be inspected") from exc
            if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
                    or info.st_uid != owner_uid or info.st_gid != owner_gid
                    or stat.S_IMODE(info.st_mode) & 0o022):
                raise LearnerHostError("user systemd directory has unsafe ownership or permissions")
        else:
            try:
                current.mkdir(mode=0o700)
            except OSError as exc:
                raise LearnerHostError("could not create a user systemd directory below HOME") from exc
    os.chmod(path, 0o700)
    return path


def _prepare_current_user_learner_host(
    deployment: LearnerDeployment,
    *,
    install_root: Path,
    account: Any,
    systemd_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess],
    docker_context: str,
    docker_rootless: bool,
    docker_host: str | None,
    docker_config: str | None,
    python_executable: Path,
    source_root: Path,
    mapping_probe: Callable[[LearnerDeployment, Path], Mapping[str, Any]],
) -> LearnerHostInstallReceipt:
    if (not hasattr(account, "uid") or not hasattr(account, "gid") or not hasattr(account, "home")
            or account.uid <= 0 or account.uid != os.geteuid() or account.uid != os.getuid()
            or account.gid < 0 or account.gid != os.getgid()):
        raise LearnerHostError("registered Learner account does not match the invoking UID")
    home = Path(account.home)
    if not home.is_absolute() or home == Path("/") or ".." in home.parts:
        raise LearnerHostError("registered Learner account home is invalid")
    expected_root = home / ".local" / "share" / "qwen-fleet" / "roles" / "learner" / deployment.deployment_id
    if Path(install_root).expanduser().absolute() != expected_root:
        raise LearnerHostError("Fleet Learner install root is outside the registered account's canonical role path")
    expected_systemd_dir = home / ".config" / "systemd" / "user"
    systemd_dir = Path(systemd_dir).expanduser().absolute()
    if systemd_dir != expected_systemd_dir:
        raise LearnerHostError("Fleet Learner unit path is outside the registered user's systemd directory")
    if not isinstance(docker_context, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", docker_context) is None:
        raise LearnerHostError("verified Docker context name is invalid")
    if type(docker_rootless) is not bool:
        raise LearnerHostError("verified Docker namespace kind is invalid")
    if getattr(deployment, "ssh", None) is not None:
        raise LearnerHostError(
            "Fleet current-user Learner setup does not install standalone Learner SSH units; disable profile.ssh"
        )
    if not callable(mapping_probe):
        raise LearnerHostError("Docker bind-mount UID/GID mapping probe is required")
    try:
        python_path = Path(python_executable).expanduser().absolute()
        resolved_python = python_path.resolve(strict=True)
        resolved_source = Path(source_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise LearnerHostError("pinned Fleet Learner Python or source is unavailable") from exc
    fleet_root = home / ".local" / "share" / "qwen-fleet"
    if fleet_root not in python_path.parents:
        raise LearnerHostError("Learner service Python is outside the pinned user Fleet runtime")
    if (not resolved_python.is_file() or not os.access(resolved_python, os.X_OK)
            or not resolved_source.is_dir() or not (resolved_source / "qwen_rl" / "__init__.py").is_file()):
        raise LearnerHostError("pinned Fleet Learner Python or source is incomplete")

    root = _validate_install_root(
        Path(install_root),
        deployment,
        owner_uid=account.uid,
        allow_rootless_container_user=docker_rootless,
    )
    _archive, _archive_size = _validate_bundle_image(root, owner_uid=account.uid)
    _validate_runtime_json(deployment)
    _validate_host_requirements(root)
    try:
        declared_uid, declared_gid = (int(part) for part in deployment.container_user.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise LearnerHostError("Learner profile container UID/GID is invalid") from exc
    if docker_rootless and (declared_uid, declared_gid) != (0, 0):
        raise LearnerHostError("rootless Docker Learner profile container_user must be exactly 0:0")
    if not docker_rootless and (declared_uid, declared_gid) != (account.uid, account.gid):
        raise LearnerHostError(
            f"rootful Docker Learner profile container_user {declared_uid}:{declared_gid} must match "
            f"registered account {account.uid}:{account.gid}"
        )
    free_bytes = _host_asset_space(deployment, validate_access=False)
    _prepare_host_control_dir(deployment, owner_uid=account.uid)
    _load_pinned_image(
        image_id=deployment.image_id,
        archive=root / "image" / "learner.tar.gz",
        runner=runner,
        docker_context=docker_context,
        docker_host=docker_host,
    )
    try:
        mapping = mapping_probe(deployment, deployment.host_control_path / "uid-map-probe")
    except Exception as exc:
        raise LearnerHostError("Docker container UID/GID mapping could not be measured") from exc
    try:
        container_uid, container_gid = (int(part) for part in deployment.container_user.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise LearnerHostError("Learner profile container UID/GID is invalid") from exc
    if (mapping.get("uid_mapping_verified") is not True
            or mapping.get("container_uid") != container_uid
            or mapping.get("container_gid") != container_gid
            or mapping.get("host_uid") != account.uid
            or mapping.get("host_gid") != account.gid
            or mapping.get("rootless") is not docker_rootless
            or mapping.get("context") != docker_context
            or mapping.get("host_override") is not (docker_host is not None)
            or not isinstance(mapping.get("endpoint"), str) or not mapping.get("endpoint")
            or not isinstance(mapping.get("daemon_id"), str) or not mapping.get("daemon_id")
            or mapping.get("probe_image") != deployment.image_id
            or mapping.get("probe_python") != "/work/envs/rl-vllm/bin/python"
            or (docker_host is not None and mapping.get("endpoint") != docker_host)):
        observed = (f"{mapping.get('container_uid')}:{mapping.get('container_gid')} -> "
                    f"{mapping.get('host_uid')}:{mapping.get('host_gid')}")
        raise LearnerHostError(
            f"Learner profile container_user {container_uid}:{container_gid} does not match "
            f"the measured Docker bind mapping {observed}; expected it to map to "
            f"account {account.uid}:{account.gid}"
        )
    deployment = _validate_profile_assets(
        deployment,
        allow_rootless_container_user=docker_rootless,
        host_uid=account.uid if docker_rootless else None,
        host_gid=account.gid if docker_rootless else None,
    )
    _validate_current_user_assets(deployment, uid=account.uid, gid=account.gid)
    free_bytes = _host_asset_space(
        deployment,
        validate_access=True,
        host_uid=account.uid if docker_rootless else None,
        host_gid=account.gid if docker_rootless else None,
    )
    if docker_config is not None:
        config_path = Path(docker_config).expanduser().absolute()
        try:
            _reject_symlink_chain(config_path)
            config_info = config_path.lstat()
        except OSError as exc:
            raise LearnerHostError("Docker user configuration is unavailable to the Fleet account") from exc
        if (config_path.is_symlink() or not stat.S_ISDIR(config_info.st_mode)
                or config_info.st_uid != account.uid or config_path.resolve() == home
                or home not in config_path.resolve().parents):
            raise LearnerHostError("Docker user configuration must be an account-owned directory under HOME")

    unit_dir = _ensure_current_user_systemd_dir(
        systemd_dir, home=home, owner_uid=account.uid, owner_gid=account.gid,
    )
    unit_name = f"qwen-rl-learner@{deployment.deployment_id}.service"
    unit_path = unit_dir / unit_name
    body = _render_current_user_unit(
        deployment=deployment,
        install_root=root,
        python=python_path,
        source_root=resolved_source,
        home=home,
        docker_context=docker_context,
        docker_config=docker_config,
        docker_host=docker_host,
        allow_rootless_container_user=docker_rootless,
    )
    _install_current_user_unit(unit_path, body, owner_uid=account.uid, owner_gid=account.gid)
    environment = _current_user_systemd_environment(account, docker_context, docker_config, docker_host)
    systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
    _run([systemctl, "--user", "daemon-reload"], runner=runner, timeout=60, env=environment)
    _run([systemctl, "--user", "enable", "--now", unit_name], runner=runner,
         timeout=1200, env=environment)
    active = runner([systemctl, "--user", "is-active", "--quiet", unit_name], check=False,
                    capture_output=True, text=True, timeout=20, env=environment)
    if active.returncode != 0:
        raise LearnerHostError("user Learner systemd unit did not remain active after startup")
    prerequisites = LearnerHostPrerequisites(
        status="ready", needs_driver=False, needs_reboot=False, needs_host_packages=False,
        driver_reason=None, selected_gpu_uuids=tuple(deployment.gpu_uuids),
        available_gpu_uuids=tuple(deployment.gpu_uuids), missing_gpu_uuids=(), missing_packages=(),
        ubuntu_version=None, architecture=platform.machine(), docker_runtime="verified_user_context",
        free_bytes_by_mount=free_bytes,
    )
    return LearnerHostInstallReceipt(
        status="started", deployment_id=deployment.deployment_id,
        service_unit=unit_name, ssh_units=(), service_started=True,
        image_id=deployment.image_id, installed_packages=(), prerequisites=prerequisites,
    )


def prepare_learner_host(
    deployment: LearnerDeployment,
    *,
    install_root: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    systemd_dir: Path = Path("/etc/systemd/system"),
    allow_package_changes: bool = False,
    reboot_required_path: Path = Path("/var/run/reboot-required"),
    mode: Literal["standalone", "current_user"] = "standalone",
    account: Any | None = None,
    docker_context: str | None = None,
    docker_rootless: bool = False,
    docker_host: str | None = None,
    docker_config: str | None = None,
    python_executable: Path | None = None,
    source_root: Path | None = None,
    mapping_probe: Callable[[LearnerDeployment, Path], Mapping[str, Any]] | None = None,
) -> LearnerHostInstallReceipt:
    """Prepare and start only the trusted Learner deployment owned by this ID.

    Asset, UID/GID readability, credential mode, image archive, and disk checks
    happen before any apt command. Missing drivers or a reboot marker always
    suppress service activation; this code never chooses a CPU fallback.
    """
    if mode not in {"standalone", "current_user"}:
        raise LearnerHostError("Learner host mode is invalid")
    if type(allow_package_changes) is not bool:
        raise LearnerHostError("allow_package_changes must be boolean")
    if mode == "current_user":
        if allow_package_changes:
            raise LearnerHostError("current-user Learner setup cannot change host packages")
        if (account is None or docker_context is None or python_executable is None
                or source_root is None or mapping_probe is None):
            raise LearnerHostError("current-user Learner setup requires account, Docker, and pinned-source evidence")
        return _prepare_current_user_learner_host(
            deployment,
            install_root=Path(install_root),
            account=account,
            systemd_dir=Path(systemd_dir),
            runner=runner,
            docker_context=docker_context,
            docker_rootless=docker_rootless,
            docker_host=docker_host,
            docker_config=docker_config,
            python_executable=Path(python_executable),
            source_root=Path(source_root),
            mapping_probe=mapping_probe,
        )
    systemd_dir = Path(systemd_dir).expanduser().resolve(strict=False)
    production_systemd = systemd_dir == Path("/etc/systemd/system").resolve(strict=False)
    if production_systemd and os.geteuid() != 0:
        raise LearnerHostError("installing a system Learner unit requires root")
    host_owner_uid = 0 if production_systemd else os.getuid()
    root = _validate_install_root(
        install_root, deployment, owner_uid=host_owner_uid, secure_parents=production_systemd,
    )
    _archive, archive_size = _validate_bundle_image(root, owner_uid=host_owner_uid)
    _check_image_space(root, archive_size)
    # Bundle import intentionally produces root-owned 0600 files. Adopt their
    # ownership only after verifying file kind and mode, then run the strict
    # runtime/profile check before package-manager changes.
    _validate_runtime_json(deployment)
    _validate_host_requirements(root)
    free_bytes = _host_asset_space(deployment, validate_access=False)
    _adopt_bundle_assets(
        deployment,
        install_root=root,
        host_owner_uid=host_owner_uid,
        host_owner_gid=os.getgid(),
    )
    free_bytes = _host_asset_space(deployment, validate_access=True)
    early = _early_gpu_reboot_gate(
        deployment,
        runner=runner,
        os_release=Path("/etc/os-release"),
        reboot_required_path=reboot_required_path,
        free_bytes=free_bytes,
    )
    if early.status != "ready":
        return _make_receipt(deployment, early)
    if not _package_installed("python3-venv", runner=runner):
        if not allow_package_changes:
            prerequisites = LearnerHostPrerequisites(
                status="needs_host_packages", needs_driver=False, needs_reboot=False,
                needs_host_packages=True, driver_reason=early.driver_reason,
                selected_gpu_uuids=early.selected_gpu_uuids,
                available_gpu_uuids=early.available_gpu_uuids,
                missing_gpu_uuids=early.missing_gpu_uuids, missing_packages=("python3-venv",),
                ubuntu_version=early.ubuntu_version, architecture=early.architecture,
                docker_runtime="not_checked",
                free_bytes_by_mount=free_bytes,
            )
            return _make_receipt(deployment, prerequisites)
        _install_python_venv_package(runner=runner)
    _install_python_venv(root=root, runner=runner)
    if runner is subprocess.run:
        return _run_venv_host_stage(
            root=root,
            systemd_dir=systemd_dir,
            allow_package_changes=allow_package_changes,
            reboot_required_path=reboot_required_path,
            runner=runner,
        )
    _activate_host_venv_imports(root)
    deployment = load_learner_deployment(root / "learner-deployment.json")
    return _prepare_after_venv(
        deployment,
        install_root=root,
        systemd_dir=systemd_dir,
        allow_package_changes=allow_package_changes,
        runner=runner,
        reboot_required_path=reboot_required_path,
    )


def _install_after_venv_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--install-after-venv", action="store_true")
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--systemd-dir", type=Path, required=True)
    parser.add_argument("--reboot-required-path", type=Path, required=True)
    parser.add_argument("--allow-package-changes", choices=("true", "false"), required=True)
    args = parser.parse_args(argv)
    if not args.install_after_venv:
        return 2
    try:
        root = Path(args.install_root).expanduser().resolve(strict=True)
        deployment = load_learner_deployment(root / "learner-deployment.json")
        receipt = _prepare_after_venv(
            deployment,
            install_root=root,
            systemd_dir=args.systemd_dir.expanduser().resolve(strict=False),
            allow_package_changes=args.allow_package_changes == "true",
            runner=subprocess.run,
            reboot_required_path=args.reboot_required_path,
        )
    except (LearnerHostError, LearnerDeploymentError, OSError, ValueError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(receipt.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_install_after_venv_main())
