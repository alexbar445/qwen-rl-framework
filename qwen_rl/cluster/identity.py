"""Durable node identity, private file primitives, and isolated SSH setup."""
from __future__ import annotations

import base64
import ipaddress
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from .node_config import NodeConfiguration


_NODE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_SSH_KEY_TYPES = {
    "ssh-ed25519", "ecdsa-sha2-nistp256", "ssh-rsa",
    "sk-ssh-ed25519@openssh.com", "sk-ecdsa-sha2-nistp256@openssh.com",
}


class IdentityError(RuntimeError):
    """Expected failure while managing private node identity or SSH state."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def canonical_public_key(value: str) -> str:
    parts = value.strip().split()
    if len(parts) < 2 or parts[0] not in _SSH_KEY_TYPES:
        raise ValueError("unsupported SSH public key")
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except ValueError as exc:
        raise ValueError("invalid SSH public key base64") from exc
    if len(blob) < 8:
        raise ValueError("SSH public key blob is too short")
    size = struct.unpack("!I", blob[:4])[0]
    algorithm = parts[0].encode("ascii")
    if size != len(algorithm) or blob[4:4 + size] != algorithm:
        raise ValueError("SSH key type does not match the key blob")
    return " ".join(parts[:2]) + (" " + parts[2] if len(parts) > 2 else "")


def ensure_private_dir(path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise IdentityError(f"private state path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    if os.name == "posix" and path.stat().st_uid != os.getuid():
        raise IdentityError(f"private state directory has a different owner: {path}")
    return path


def atomic_write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    """Atomically replace a local file without following a target symlink."""
    path = Path(path).absolute()
    ensure_private_dir(path.parent)
    if path.is_symlink():
        raise IdentityError(f"refusing to replace a symlink: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        os.chmod(path, mode)
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def set_file_owner(path: Path, *, uid: int, gid: int) -> None:
    metadata = Path(path).lstat()
    if not stat.S_ISREG(metadata.st_mode) or Path(path).is_symlink():
        raise IdentityError("refusing to change owner of a non-regular file")
    if os.geteuid() != 0 and (metadata.st_uid != uid or os.getegid() != gid):
        raise IdentityError("setting worker file ownership requires the configured service UID/GID")
    try:
        os.chown(path, uid, gid)
    except OSError as exc:
        raise IdentityError("could not set the Worker service file owner") from exc
    os.chmod(path, 0o600)


def set_dir_owner(path: Path, *, uid: int, gid: int, mode: int = 0o700) -> None:
    metadata = Path(path).lstat()
    if not stat.S_ISDIR(metadata.st_mode) or Path(path).is_symlink():
        raise IdentityError("refusing to change owner of a non-directory")
    if os.geteuid() != 0 and (metadata.st_uid != uid or os.getegid() != gid):
        raise IdentityError("setting Worker state ownership requires the configured service UID/GID")
    try:
        os.chown(path, uid, gid)
    except OSError as exc:
        raise IdentityError("could not set Worker state directory owner") from exc
    os.chmod(path, mode)


def allow_group_traverse(path: Path, *, gid: int) -> None:
    """Allow a configured service group to traverse one private state root only."""
    path = Path(path)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise IdentityError("service state root is not a regular directory")
    if os.geteuid() != 0 and (metadata.st_uid != os.getuid() or gid != os.getegid()):
        raise IdentityError("sharing state-root traversal requires root or the configured service group")
    try:
        os.chown(path, metadata.st_uid, gid)
        os.chmod(path, 0o710)
    except OSError as exc:
        raise IdentityError("could not enable private service traversal to node state") from exc


def read_private_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise IdentityError(f"private file must be regular mode 0600: {path}")
        if os.name == "posix" and info.st_uid != os.getuid():
            raise IdentityError(f"private file has a different owner: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IdentityError(f"cannot read private JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise IdentityError(f"private JSON root must be an object: {path}")
    return value


def read_ssh_public_key(path: Path) -> str:
    try:
        value = Path(path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise IdentityError(f"SSH public key is unavailable: {path}") from exc
    parts = value.split()
    if len(parts) < 2 or parts[0] not in {
        "ssh-ed25519", "ecdsa-sha2-nistp256", "ssh-rsa",
        "sk-ssh-ed25519@openssh.com", "sk-ecdsa-sha2-nistp256@openssh.com",
    }:
        raise IdentityError("SSH public key has an unsupported format")
    try:
        key = base64.b64decode(parts[1], validate=True)
    except ValueError as exc:
        raise IdentityError("SSH public key has invalid base64") from exc
    if not key:
        raise IdentityError("SSH public key is empty")
    return " ".join(parts[:2]) + (" " + parts[2] if len(parts) > 2 else "")


def _run_command(
    argv: Sequence[str], *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 60.0, input_text: str | None = None,
) -> subprocess.CompletedProcess:
    if not argv or any(not isinstance(part, str) or "\x00" in part for part in argv):
        raise IdentityError("invalid command argument")
    try:
        result = runner(list(argv), check=False, capture_output=True, text=True,
                        timeout=timeout, input=input_text)
    except (OSError, subprocess.SubprocessError) as exc:
        raise IdentityError(f"could not run {Path(argv[0]).name}") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        detail = stderr[-1][:240] if stderr else "command failed"
        raise IdentityError(f"{Path(argv[0]).name} failed: {detail}")
    return result


class NodeIdentity:
    """Durable per-node UUID and reverse-tunnel keypair."""

    def __init__(
        self, directory: Path, *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        key_creator: Callable[[Path, Path, str], None] | None = None,
    ):
        self.directory = ensure_private_dir(Path(directory))
        self.node_id_path = self.directory / "node-identity.json"
        self.key_path = self.directory / "ssh" / "id_ed25519"
        self.public_key_path = self.directory / "ssh" / "id_ed25519.pub"
        self.host_key_path = self.directory / "ssh" / "ssh_host_ed25519_key"
        self.host_public_key_path = self.directory / "ssh" / "ssh_host_ed25519_key.pub"
        self.runner = runner
        self.key_creator = key_creator

    def _ensure_keypair(self, private_path: Path, public_path: Path, comment: str) -> str:
        if private_path.exists():
            info = private_path.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                raise IdentityError(f"SSH private key must be a regular mode 0600 file: {private_path}")
            if not public_path.is_file():
                raise IdentityError(f"SSH public key is missing: {public_path}")
        else:
            if public_path.exists():
                raise IdentityError(f"SSH public key exists without private key: {public_path}")
            if self.key_creator is None:
                _run_command([shutil.which("ssh-keygen") or "ssh-keygen", "-q", "-t", "ed25519",
                              "-N", "", "-C", comment, "-f", str(private_path)],
                             runner=self.runner, timeout=30)
            else:
                try:
                    self.key_creator(private_path, public_path, comment)
                except Exception as exc:
                    raise IdentityError("configured SSH key creator failed") from exc
            if not private_path.is_file() or not public_path.is_file():
                raise IdentityError("ssh-keygen did not create a complete node keypair")
            os.chmod(private_path, 0o600)
            os.chmod(public_path, 0o644)
        return read_ssh_public_key(public_path)

    def load_or_create(self) -> tuple[str, str, str]:
        if self.node_id_path.exists():
            identity = read_private_json(self.node_id_path)
            node_id = identity.get("node_id")
            if not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id):
                raise IdentityError("stored node identity is invalid")
        else:
            node_id = "node-" + str(uuid4())
            atomic_write(self.node_id_path, _json_bytes({"node_id": node_id}))

        ensure_private_dir(self.key_path.parent)
        reverse_key = self._ensure_keypair(self.key_path, self.public_key_path, f"qwen-cluster:{node_id}")
        host_key = self._ensure_keypair(self.host_key_path, self.host_public_key_path,
                                        f"qwen-cluster-host:{node_id}")
        return node_id, reverse_key, host_key


def write_node_ssh_state(
    *, state_dir: Path, admin_public_key: str, lan_host: str, ssh_port: int,
    role: Literal["gateway", "worker"], runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[Path, Path]:
    """Prepare a separate cluster-only sshd; the host sshd config is untouched."""
    try:
        key = canonical_public_key(admin_public_key)
    except ValueError as exc:
        raise IdentityError("admin SSH key is invalid") from exc
    try:
        address = ipaddress.ip_address(lan_host)
    except ValueError as exc:
        raise IdentityError("isolated sshd requires a numeric LAN address") from exc
    if address.is_loopback or address.is_link_local or not 1 <= ssh_port <= 65535:
        raise IdentityError("isolated sshd address or port is invalid")
    ssh_dir = ensure_private_dir(Path(state_dir) / "ssh")
    host_key = ssh_dir / "ssh_host_ed25519_key"
    host_public_key = ssh_dir / "ssh_host_ed25519_key.pub"
    if not host_key.is_file() or not host_public_key.is_file():
        raise IdentityError("isolated sshd host keypair is missing")
    auth_file = ssh_dir / "authorized_keys"
    config_file = ssh_dir / "sshd_config"
    atomic_write(auth_file, (key + "\n").encode("ascii"))
    lines = [
        f"Port {ssh_port}", f"ListenAddress {address}", f"HostKey {host_key}",
        f"PidFile {Path(state_dir).absolute() / 'ssh' / 'sshd.pid'}",
        f"AuthorizedKeysFile {auth_file}", "PubkeyAuthentication yes",
        "PasswordAuthentication no", "KbdInteractiveAuthentication no",
        "PermitRootLogin prohibit-password", "AllowUsers root", "UsePAM yes",
        "X11Forwarding no", "AllowAgentForwarding no", "PermitTTY no",
        "PermitTunnel no", "GatewayPorts no",
        f"AllowTcpForwarding {'yes' if role == 'gateway' else 'no'}",
        "ClientAliveInterval 60", "ClientAliveCountMax 3",
    ]
    atomic_write(config_file, ("\n".join(lines) + "\n").encode("ascii"))
    sshd = shutil.which("sshd") or ("/usr/sbin/sshd" if Path("/usr/sbin/sshd").is_file() else None)
    if sshd is not None:
        _run_command([sshd, "-t", "-f", str(config_file)], runner=runner, timeout=15)
    return auth_file, config_file


def reverse_ssh_argv(config: NodeConfiguration, *, identity_file: Path, known_hosts_file: Path) -> list[str]:
    """Build a safe OpenSSH reverse tunnel command for Gateway nodes only."""
    if config.role != "gateway" or config.reverse_ssh is None:
        raise IdentityError("reverse SSH is only configured for Gateway nodes")
    target = config.reverse_ssh
    try:
        address = ipaddress.ip_address(target.central_host)
    except ValueError:
        address = None
    if address is not None and address.version == 6:
        host_alias = f"[{target.central_host}]:{target.central_port}"
    else:
        host_alias = f"[{target.central_host}]:{target.central_port}" if target.central_port != 22 else target.central_host
    atomic_write(Path(known_hosts_file), f"{host_alias} {target.central_host_key.strip()}\n".encode("ascii"))
    return [
        shutil.which("ssh") or "ssh", "-F", "/dev/null", "-N", "-T", "-p", str(target.central_port),
        "-i", str(Path(identity_file)), "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3", "-o", "IdentitiesOnly=yes",
        "-o", "ClearAllForwardings=yes", "-o", "GatewayPorts=no",
        "-o", "ProxyCommand=none", "-o", "ProxyJump=none",
        "-o", "GlobalKnownHostsFile=/dev/null", "-o", "VerifyHostKeyDNS=no",
        "-o", "UpdateHostKeys=no", "-o", "CheckHostIP=no",
        "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts_file}",
        "-R", f"127.0.0.1:{target.remote_api_port}:127.0.0.1:{target.local_api_port}",
        "-R", f"127.0.0.1:{target.remote_ssh_port}:127.0.0.1:{config.ssh_port}",
        f"{target.central_user}@{target.central_host}",
    ]
