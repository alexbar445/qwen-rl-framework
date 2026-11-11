"""Optional, isolated SSH management endpoints for a trusted Learner host."""
from __future__ import annotations

import ipaddress
import os
import pwd
import re
import shutil
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_rl.cluster.identity import (
    atomic_write,
    canonical_public_key,
    ensure_private_dir,
)


class LearnerSSHError(RuntimeError):
    """Expected failure while preparing an isolated Learner SSH endpoint."""


@dataclass(frozen=True)
class LearnerSSHSetup:
    state_dir: Path
    sshd_config_path: Path
    authorized_keys_path: Path
    host_key_path: Path
    host_public_key_path: Path
    central_known_hosts_path: Path
    sshd_unit_name: str
    tunnel_unit_name: str
    sshd_unit: str
    tunnel_unit: str
    tunnel_argv: tuple[str, ...]


def _single_line(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
        raise LearnerSSHError(f"SSH {field} must be a non-empty single-line value")
    return value


def _loopback(value: str, field: str) -> str:
    value = _single_line(value, field)
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise LearnerSSHError(f"SSH {field} must be a loopback IP address") from exc
    if not address.is_loopback:
        raise LearnerSSHError(f"SSH {field} must be a loopback IP address")
    return str(address)


def _port(value: int, field: str, *, minimum: int = 1024) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 65535:
        raise LearnerSSHError(f"SSH {field} must be between {minimum} and 65535")
    return value


def _systemd_arg(value: str | Path) -> str:
    text = str(value)
    if not text or any(char in text for char in "\r\n\x00"):
        raise LearnerSSHError("systemd SSH command arguments must be single-line")
    if not any(char.isspace() or char in '\\"%$' for char in text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def _systemd_path(value: str | Path) -> str:
    """Escape one raw path directive without turning it into a quoted argument."""
    text = str(value)
    if not Path(text).is_absolute() or any(char in text for char in "\r\n\x00"):
        raise LearnerSSHError("systemd path directives require absolute single-line paths")
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


def _run(
    argv: list[str], *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
        raise LearnerSSHError("invalid SSH command arguments")
    try:
        result = runner(argv, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LearnerSSHError(f"could not run {Path(argv[0]).name}") from exc
    if result.returncode != 0:
        raise LearnerSSHError(f"{Path(argv[0]).name} exited with status {result.returncode}")
    return result


def _private_regular_file(path: Path, *, field: str) -> None:
    try:
        info = Path(path).lstat()
    except OSError as exc:
        raise LearnerSSHError(f"SSH {field} is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or Path(path).is_symlink():
        raise LearnerSSHError(f"SSH {field} must be a regular file")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise LearnerSSHError(f"SSH {field} must have mode 0600")


def _settings(deployment: Any) -> Any:
    settings = getattr(deployment, "ssh", None)
    if settings is None:
        raise LearnerSSHError("Learner SSH is disabled in this deployment profile")
    return settings


def _validate_settings(deployment: Any) -> tuple[Any, ...]:
    settings = _settings(deployment)
    central_host = _single_line(settings.central_host, "central_host")
    try:
        address = ipaddress.ip_address(central_host)
    except ValueError:
        if (central_host.startswith("-") or len(central_host) > 253
                or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", central_host) is None):
            raise LearnerSSHError("SSH central_host must be a DNS name or IP address")
    else:
        if address.is_unspecified or address.is_multicast:
            raise LearnerSSHError("SSH central_host must be a unicast address")
    central_user = _single_line(settings.central_user, "central_user")
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", central_user) is None:
        raise LearnerSSHError("SSH central_user contains invalid characters")
    local_user = _single_line(settings.local_ssh_user, "local_ssh_user")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,31}", local_user) is None:
        raise LearnerSSHError("SSH local_ssh_user contains invalid characters")
    try:
        account = pwd.getpwnam(local_user)
    except KeyError as exc:
        raise LearnerSSHError("SSH local_ssh_user must already exist on this host") from exc
    if not account.pw_shell or Path(account.pw_shell).name in {"nologin", "false"}:
        raise LearnerSSHError("SSH local_ssh_user must have a usable shell")
    central_port = _port(settings.central_ssh_port, "central_ssh_port", minimum=1)
    remote_api = _port(settings.remote_api_port, "remote_api_port")
    remote_ssh = _port(settings.remote_ssh_port, "remote_ssh_port")
    if remote_api == remote_ssh:
        raise LearnerSSHError("SSH remote API and SSH ports must be distinct")
    local_address = _loopback(settings.local_listener_address, "local_listener_address")
    local_port = _port(settings.local_listener_port, "local_listener_port")
    host_key = canonical_public_key(settings.central_host_public_key)
    admin_key = canonical_public_key(settings.administrator_public_key)
    identity_path = Path(settings.client_private_key_host_path).expanduser().absolute()
    _private_regular_file(identity_path, field="client private key")
    if any(char in str(identity_path) for char in "\r\n\x00"):
        raise LearnerSSHError("SSH client key path must be a single-line path")
    publish_address = _loopback(deployment.publish.host_address, "Learner published host address")
    api_port = _port(deployment.publish.host_port, "Learner published host port", minimum=1)
    return (settings, central_host, central_user, central_port, remote_api, remote_ssh,
            local_address, local_port, publish_address, api_port, local_user, host_key, admin_key,
            identity_path)


def write_learner_ssh_state(
    deployment: Any,
    *,
    state_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> LearnerSSHSetup:
    """Write isolated loopback-only sshd state and a pinned reverse tunnel.

    The host's native sshd configuration and units are not read or changed.
    The profile must name an existing non-root account; this function never
    creates host accounts or modifies their home directories.
    """
    try:
        (_settings, central_host, central_user, central_port, remote_api, remote_ssh,
         local_address, local_port, publish_address, api_port, local_user,
         central_host_key, admin_key, client_key) = _validate_settings(deployment)
    except (AttributeError, TypeError, ValueError) as exc:
        if isinstance(exc, LearnerSSHError):
            raise
        raise LearnerSSHError("Learner SSH profile is incomplete or invalid") from exc
    ssh_state = ensure_private_dir(Path(state_dir))
    # OpenSSH reads AuthorizedKeysFile as the target user. The directory is
    # made searchable but not listable after all atomic writes complete.
    host_key_path = ssh_state / "ssh_host_ed25519_key"
    host_public_path = ssh_state / "ssh_host_ed25519_key.pub"
    if host_key_path.exists() != host_public_path.exists():
        raise LearnerSSHError("Learner SSH host keypair is incomplete; refusing to replace identity")
    ssh_keygen = shutil.which("ssh-keygen") or "/usr/bin/ssh-keygen"
    if not host_key_path.exists():
        _run([ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-C", "qwen-rl-learner-sshd",
              "-f", str(host_key_path)], runner=runner)
    keygen_public = _run([ssh_keygen, "-y", "-P", "", "-f", str(client_key)], runner=runner).stdout
    try:
        canonical_public_key(keygen_public)
    except ValueError as exc:
        raise LearnerSSHError("SSH client private key is not a valid unencrypted OpenSSH key") from exc
    _private_regular_file(host_key_path, field="sshd host private key")
    try:
        canonical_public_key(host_public_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise LearnerSSHError("Learner SSH host public key is malformed") from exc

    authorized_keys = ssh_state / "authorized_keys"
    config_path = ssh_state / "sshd_config"
    known_hosts = ssh_state / "central_known_hosts"
    # `restrict` disables agent, X11, port, and stream-local forwarding for
    # this admin key. Management remains limited to ordinary SSH commands.
    atomic_write(authorized_keys, ("restrict " + admin_key + "\n").encode("ascii"), mode=0o600)
    try:
        os.chown(authorized_keys, pwd.getpwnam(local_user).pw_uid, pwd.getpwnam(local_user).pw_gid)
    except OSError as exc:
        raise LearnerSSHError("could not assign the isolated authorized key to the configured local account") from exc
    host_alias = central_host if central_port == 22 else f"[{central_host}]:{central_port}"
    atomic_write(known_hosts, f"{host_alias} {central_host_key}\n".encode("ascii"), mode=0o600)
    root_policy = "prohibit-password" if local_user == "root" else "no"
    lines = [
        f"Port {local_port}", f"ListenAddress {local_address}", f"HostKey {host_key_path}",
        f"PidFile {ssh_state / 'sshd.pid'}", f"AuthorizedKeysFile {authorized_keys}",
        "PubkeyAuthentication yes", "AuthenticationMethods publickey",
        "PasswordAuthentication no", "KbdInteractiveAuthentication no", "PermitEmptyPasswords no",
        f"PermitRootLogin {root_policy}", f"AllowUsers {local_user}", "UsePAM yes", "StrictModes yes",
        "X11Forwarding no", "AllowAgentForwarding no", "AllowTcpForwarding no",
        "AllowStreamLocalForwarding no", "PermitTunnel no", "PermitTTY no", "GatewayPorts no",
        "UseDNS no", "ClientAliveInterval 60", "ClientAliveCountMax 3",
    ]
    atomic_write(config_path, ("\n".join(lines) + "\n").encode("ascii"), mode=0o600)
    os.chmod(ssh_state, 0o711)
    sshd = shutil.which("sshd") or "/usr/sbin/sshd"
    if not Path(sshd).is_file():
        raise LearnerSSHError("optional Learner SSH requires an already-installed OpenSSH server")
    _run([sshd, "-t", "-f", str(config_path)], runner=runner)

    ssh = shutil.which("ssh") or "/usr/bin/ssh"
    identity = str(client_key)
    tunnel_argv = [
        ssh, "-F", "/dev/null", "-N", "-T", "-p", str(central_port), "-i", identity,
        "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "IdentitiesOnly=yes",
        "-o", "ClearAllForwardings=yes", "-o", "GatewayPorts=no", "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known_hosts}", "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "VerifyHostKeyDNS=no", "-o", "UpdateHostKeys=no", "-o", "CheckHostIP=no",
        "-o", "ProxyCommand=none", "-o", "ProxyJump=none", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3", "-R", f"127.0.0.1:{remote_api}:{publish_address}:{api_port}",
        "-R", f"127.0.0.1:{remote_ssh}:{local_address}:{local_port}",
        f"{central_user}@{central_host}",
    ]
    deployment_id = deployment.deployment_id
    learner_unit_name = f"qwen-rl-learner@{deployment_id}.service"
    sshd_unit_name = f"qwen-rl-learner-sshd@{deployment_id}.service"
    tunnel_unit_name = f"qwen-rl-learner-tunnel@{deployment_id}.service"
    sshd_unit = "\n".join([
        "[Unit]", "Description=Isolated Qwen Learner SSH management daemon",
        "After=network.target", "", "[Service]", "Type=simple", "User=root", "UMask=0077",
        f"ExecStart={_systemd_arg(sshd)} -D -e -f {_systemd_arg(config_path)}",
        "Restart=on-failure", "RestartSec=5s", "NoNewPrivileges=yes", "ProtectSystem=strict",
        "ProtectHome=read-only", "PrivateTmp=yes", f"ReadWritePaths={_systemd_path(ssh_state)} /run",
        "", "[Install]", "WantedBy=multi-user.target", "",
    ])
    tunnel_unit = "\n".join([
        "[Unit]", "Description=Pinned Qwen Learner reverse SSH management tunnel",
        f"Wants=network-online.target {learner_unit_name} {sshd_unit_name}",
        f"After=network-online.target {learner_unit_name} {sshd_unit_name}",
        f"Requires={learner_unit_name} {sshd_unit_name}", "", "[Service]",
        "Type=simple", "User=root", "UMask=0077",
        "ExecStart=" + " ".join(_systemd_arg(arg) for arg in tunnel_argv),
        "Restart=always", "RestartSec=5s", "NoNewPrivileges=yes", "ProtectSystem=strict",
        "ProtectHome=read-only", "PrivateTmp=yes", f"ReadWritePaths={_systemd_path(ssh_state)} /run",
        "", "[Install]", "WantedBy=multi-user.target", "",
    ])
    return LearnerSSHSetup(
        state_dir=ssh_state,
        sshd_config_path=config_path,
        authorized_keys_path=authorized_keys,
        host_key_path=host_key_path,
        host_public_key_path=host_public_path,
        central_known_hosts_path=known_hosts,
        sshd_unit_name=sshd_unit_name,
        tunnel_unit_name=tunnel_unit_name,
        sshd_unit=sshd_unit,
        tunnel_unit=tunnel_unit,
        tunnel_argv=tuple(tunnel_argv),
    )
