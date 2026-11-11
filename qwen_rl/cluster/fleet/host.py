"""Read host inventory and bind Fleet enrollment to the effective Unix account."""
from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
import pwd
from typing import Callable, Sequence

from pydantic import ValidationError

from .. import host as cluster_host
from ..contracts import GPUCapability
from .contracts import FleetNodeRegistration, UserAccount


class FleetHostError(RuntimeError):
    """Expected, redacted failure while observing or preparing a Fleet host."""


@dataclass(frozen=True)
class HostInventory:
    hostname: str
    os_release: str
    architecture: str
    boot_id: str
    lan_host: str | None
    gpus: tuple[GPUCapability, ...]
    account: UserAccount

    def registration(
        self,
        *,
        node_id: str,
        agent_version: str,
        ssh_port: int,
        ssh_public_key: str,
        ssh_host_public_key: str,
    ) -> FleetNodeRegistration:
        try:
            return FleetNodeRegistration(
                node_id=node_id,
                boot_id=self.boot_id,
                hostname=self.hostname,
                os_release=self.os_release,
                architecture=self.architecture,
                agent_version=agent_version,
                account=self.account,
                lan_host=self.lan_host,
                ssh_port=ssh_port,
                ssh_public_key=ssh_public_key,
                ssh_host_public_key=ssh_host_public_key,
                gpus=list(self.gpus),
            )
        except ValidationError as exc:
            raise FleetHostError("local node inventory did not match the Fleet registration contract") from exc


def _os_release(path: Path = Path("/etc/os-release")) -> tuple[str, str]:
    try:
        fields = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                fields[key] = value.strip().strip('"')
    except OSError as exc:
        raise FleetHostError("Ubuntu release metadata is unavailable") from exc
    release = fields.get("ID", "unknown")
    version = fields.get("VERSION_ID", "unknown")
    return release, f"{release} {version}"[:256]


def current_user_account() -> UserAccount:
    """Resolve the process account from its effective UID and the system passwd database."""
    effective_uid = os.geteuid()
    try:
        entry = pwd.getpwuid(effective_uid)
    except (KeyError, OSError) as exc:
        raise FleetHostError("effective UID has no passwd account") from exc
    if entry.pw_uid != effective_uid:
        raise FleetHostError("passwd lookup did not match the effective UID")
    try:
        return UserAccount(username=entry.pw_name, uid=entry.pw_uid,
                           gid=entry.pw_gid, home=entry.pw_dir)
    except ValidationError as exc:
        raise FleetHostError("effective account is not eligible for Fleet registration") from exc


def collect_inventory(
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    os_release_path: Path = Path("/etc/os-release"),
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    hostname_provider: Callable[[], str] = socket.gethostname,
    architecture_provider: Callable[[], str] = platform.machine,
) -> HostInventory:
    """Collect role-neutral facts. Missing GPU drivers do not prevent enrollment."""
    _release_id, release = _os_release(os_release_path)
    account = current_user_account()
    hostname = hostname_provider()
    architecture = architecture_provider()
    if not hostname or any(ord(char) < 32 for char in hostname):
        raise FleetHostError("host name is invalid")
    try:
        gpus = tuple(cluster_host.probe_gpus(runner=runner))
    except cluster_host.HostError as exc:
        raise FleetHostError("GPU inventory could not be read") from exc
    try:
        lan_host = cluster_host.detect_lan_host(runner=runner)
    except cluster_host.HostError:
        # A LAN address is optional for the Fleet reverse-tunnel path.
        lan_host = None
    try:
        boot_id = cluster_host.read_boot_id(boot_id_path)
    except cluster_host.HostError as exc:
        raise FleetHostError("Linux boot identity is unavailable") from exc
    return HostInventory(hostname, release, architecture, boot_id, lan_host, gpus, account)


def host_has_nvidia_device(*, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> bool | None:
    """Return PCI NVIDIA presence, or None when pciutils cannot establish it."""
    lspci = shutil.which("lspci")
    if not lspci:
        return None
    try:
        result = runner([lspci, "-n"], check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return any(re.search(r"(?:^|\s)10de:", line, re.IGNORECASE) for line in (result.stdout or "").splitlines())


def select_one_gpu(gpus: Sequence[GPUCapability], requested_uuid: str | None) -> GPUCapability:
    """Resolve exactly one GPU; never guess an index or device on ambiguity."""
    if requested_uuid is not None:
        matches = [gpu for gpu in gpus if gpu.uuid == requested_uuid]
        if len(matches) != 1:
            raise FleetHostError("the centrally selected GPU UUID is not present exactly once")
        return matches[0]
    if len(gpus) != 1:
        raise FleetHostError("GPU selection is required when the host has zero or multiple GPUs")
    return gpus[0]
