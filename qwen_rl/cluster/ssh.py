"""Pinned OpenSSH inventory generation for central-to-site management."""
from __future__ import annotations

import base64
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contracts import GPUCapability
from .identity import IdentityError, atomic_write


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_SAFE_SITE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_KEY_TYPES = {"ssh-ed25519", "ecdsa-sha2-nistp256", "ssh-rsa"}


class SSHConfigError(RuntimeError):
    """Expected, user-safe inventory or OpenSSH configuration error."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SSHNodeRecord(_Strict):
    node_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    site_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    boot_id: str = Field(min_length=1, max_length=100)
    role: str
    lan_host: str = Field(min_length=1, max_length=253)
    worker_port: int = Field(ge=1, le=65535)
    ssh_port: int = Field(ge=1, le=65535)
    ssh_host_public_key: str = Field(min_length=1, max_length=2000)
    gpus: list[GPUCapability] = Field(default_factory=list)


@dataclass(frozen=True)
class SiteSSHFiles:
    site_id: str
    nodes: tuple[SSHNodeRecord, ...]
    inventory_path: Path
    known_hosts_path: Path


def _safe_key(value: str) -> tuple[str, str]:
    parts = value.strip().split()
    if len(parts) < 2 or parts[0] not in _KEY_TYPES:
        raise SSHConfigError("SSH public key is malformed or unsupported")
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except ValueError as exc:
        raise SSHConfigError("SSH public key base64 is invalid") from exc
    if len(blob) < 8 or any(ord(char) < 32 for char in " ".join(parts[2:])):
        raise SSHConfigError("SSH public key record is invalid")
    return parts[0], parts[1]


def _private_file(path: Path) -> None:
    try:
        info = Path(path).lstat()
    except OSError as exc:
        raise SSHConfigError(f"required private file is unavailable: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        raise SSHConfigError(f"private file must be a regular mode 0600 file: {path}")
    if os.name == "posix" and info.st_uid != os.getuid():
        raise SSHConfigError(f"private file has a different owner: {path}")


def _config_value(value: str | Path) -> str:
    text = str(value)
    if not text or any(char in text for char in "\r\n\x00"):
        raise SSHConfigError("SSH config values must be non-empty single-line strings")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


class SSHInventory:
    """Validated view of Registry's site inventory and pinned known_hosts."""

    def __init__(self, files: SiteSSHFiles):
        self.files = files
        by_id = {node.node_id: node for node in files.nodes}
        if len(by_id) != len(files.nodes):
            raise SSHConfigError("site inventory contains duplicate node IDs")
        if any(node.site_id != files.site_id for node in files.nodes):
            raise SSHConfigError("site inventory contains a node from another site")
        gateways = [node for node in files.nodes if node.role == "gateway"]
        if len(gateways) != 1:
            raise SSHConfigError("site inventory must contain exactly one Gateway node")
        if any(node.role not in {"gateway", "worker"} for node in files.nodes):
            raise SSHConfigError("site inventory has an unsupported node role")
        self.nodes = by_id
        self.gateway = gateways[0]
        self._validate_known_hosts()

    @classmethod
    def load(cls, state_dir: Path, site_id: str) -> "SSHInventory":
        if not _SAFE_SITE.fullmatch(site_id):
            raise SSHConfigError("site_id is invalid")
        directory = Path(state_dir) / "ssh" / "sites" / site_id
        inventory_path = directory / "inventory.json"
        known_hosts_path = directory / "known_hosts"
        try:
            for path in (inventory_path, known_hosts_path):
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
                    raise SSHConfigError("Registry SSH files must be private regular files")
            raw = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SSHConfigError("Registry SSH inventory cannot be read") from exc
        if not isinstance(raw, list):
            raise SSHConfigError("Registry inventory must be a JSON list")
        try:
            records = tuple(SSHNodeRecord.model_validate(row) for row in raw)
        except ValidationError as exc:
            raise SSHConfigError("Registry inventory does not match the T1 node schema") from exc
        return cls(SiteSSHFiles(site_id, records, inventory_path, known_hosts_path))

    def _validate_known_hosts(self) -> None:
        try:
            lines = self.files.known_hosts_path.read_text(encoding="ascii").splitlines()
        except (OSError, UnicodeError) as exc:
            raise SSHConfigError("Registry known_hosts cannot be read") from exc
        pins: dict[str, tuple[str, str]] = {}
        for line in lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3 or "," in parts[0] or "|" in parts[0]:
                raise SSHConfigError("Registry known_hosts must use plain node_id aliases")
            alias = parts[0]
            if alias in pins:
                raise SSHConfigError("Registry known_hosts has duplicate aliases")
            pins[alias] = _safe_key(" ".join(parts[1:3]))
        if set(pins) != set(self.nodes):
            raise SSHConfigError("Registry known_hosts aliases do not match inventory node IDs")
        for node_id, node in self.nodes.items():
            if pins[node_id] != _safe_key(node.ssh_host_public_key):
                raise SSHConfigError(f"host-key pin does not match inventory for node {node_id}")

    @property
    def gateway_alias(self) -> str:
        return f"qwen-gateway-{self.files.site_id}"

    def write_config(
        self,
        destination: Path,
        *,
        central_identity_file: Path,
        gateway_remote_port: int,
        user: str = "root",
    ) -> Path:
        """Write an isolated `ssh -F` config; never edit ~/.ssh/config."""
        _private_file(central_identity_file)
        if not 1024 <= gateway_remote_port <= 65535:
            raise SSHConfigError("Gateway reverse-forward port is invalid")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", user):
            raise SSHConfigError("SSH management user is invalid")
        known_hosts = _config_value(self.files.known_hosts_path)
        identity = _config_value(central_identity_file)
        lines: list[str] = []
        shared = [
            f"  User {user}", f"  IdentityFile {identity}", "  IdentitiesOnly yes",
            f"  UserKnownHostsFile {known_hosts}", "  StrictHostKeyChecking yes",
            "  GlobalKnownHostsFile /dev/null", "  VerifyHostKeyDNS no", "  UpdateHostKeys no",
            "  CheckHostIP no", "  PasswordAuthentication no",
            "  KbdInteractiveAuthentication no", "  BatchMode yes",
            "  ConnectTimeout 10", "  ServerAliveInterval 15", "  ServerAliveCountMax 3",
        ]
        gateway = self.gateway
        lines.extend([
            f"Host {self.gateway_alias}", "  HostName 127.0.0.1",
            f"  Port {gateway_remote_port}", f"  HostKeyAlias {gateway.node_id}", *shared,
        ])
        for node in sorted(self.nodes.values(), key=lambda item: item.node_id):
            if node.role == "gateway":
                continue
            try:
                ipaddress.ip_address(node.lan_host)
                host = node.lan_host
            except ValueError:
                if any(char.isspace() or ord(char) < 32 for char in node.lan_host) or node.lan_host.startswith("-"):
                    raise SSHConfigError(f"invalid LAN host for node {node.node_id}")
                host = node.lan_host
            if not _SAFE_ID.fullmatch(node.node_id):
                raise SSHConfigError("node ID is invalid")
            lines.extend([
                f"Host {node.node_id}", f"  HostName {host}", f"  Port {node.ssh_port}",
                f"  HostKeyAlias {node.node_id}", f"  ProxyJump {self.gateway_alias}", *shared,
            ])
        content = ("\n".join(lines) + "\n").encode("utf-8")
        try:
            atomic_write(Path(destination), content, mode=0o600)
        except IdentityError as exc:
            raise SSHConfigError("could not safely write isolated SSH config") from exc
        return Path(destination)

    def ssh_argv(self, node_id: str, command: Sequence[str] | None, *, config_path: Path) -> list[str]:
        if node_id not in self.nodes:
            raise SSHConfigError("node is not present in the site inventory")
        if self.nodes[node_id].role == "gateway":
            alias = self.gateway_alias
        else:
            alias = node_id
        argv = [shutil.which("ssh") or "ssh", "-F", str(config_path), "-T", alias]
        if command is not None:
            if not command or any(not isinstance(arg, str) or "\x00" in arg for arg in command):
                raise SSHConfigError("remote command must be a non-empty argument list")
            argv.append(shlex.join(list(command)))
        return argv

    def validate_config(
        self,
        config_path: Path,
        *,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> dict[str, str]:
        """Ask OpenSSH to parse each alias without making a network connection."""
        ssh = shutil.which("ssh") or "ssh"
        checked: dict[str, str] = {}
        aliases = [(self.gateway_alias, self.gateway.node_id)] + [
            (node_id, node_id) for node_id, node in self.nodes.items() if node.role == "worker"
        ]
        for alias, host_key_alias in aliases:
            try:
                result = runner([ssh, "-G", "-F", str(config_path), alias], check=False,
                                capture_output=True, text=True, timeout=15)
            except (OSError, subprocess.SubprocessError) as exc:
                raise SSHConfigError("could not run OpenSSH config validation") from exc
            if result.returncode != 0:
                raise SSHConfigError(f"OpenSSH rejected config for alias {alias}")
            values = {}
            for line in result.stdout.splitlines():
                key, _, value = line.partition(" ")
                values[key.lower()] = value.strip()
            if values.get("stricthostkeychecking") not in {"yes", "true"}:
                raise SSHConfigError("OpenSSH host-key checking is not strict")
            if Path(values.get("userknownhostsfile", "")).expanduser() != self.files.known_hosts_path:
                raise SSHConfigError("OpenSSH does not use the Registry host-key pins")
            if values.get("hostkeyalias") != host_key_alias:
                raise SSHConfigError("OpenSSH HostKeyAlias does not match the Registry identity")
            if values.get("identityfile") != str(Path(values.get("identityfile", "")).expanduser()):
                raise SSHConfigError("OpenSSH identity path is not normalized")
            checked[alias] = host_key_alias
        return checked


def run_ssh(
    inventory: SSHInventory,
    node_id: str,
    command: Sequence[str],
    *,
    config_path: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess:
    """Execute one OpenSSH command as an argv vector, with no local shell."""
    argv = inventory.ssh_argv(node_id, command, config_path=config_path)
    try:
        result = runner(argv, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SSHConfigError("OpenSSH command failed to start") from exc
    if result.returncode != 0:
        raise SSHConfigError(f"SSH command for {node_id} exited with status {result.returncode}")
    return result
