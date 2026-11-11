"""Owned central SSH relay for site-initiated reverse management tunnels.

The relay has its own host key, listener, config, and authorized-key file. It
never reads or writes the host's native sshd configuration. Site-scoped grants
come from Registry's private ``ssh/sites/<site>`` exports.
"""
from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from typing import Callable

from .identity import IdentityError, atomic_write, ensure_private_dir


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_SAFE_SITE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_AUTH_LINE = re.compile(
    r'^restrict,command="/usr/bin/false",port-forwarding,'
    r'permitlisten="127\.0\.0\.1:([0-9]{1,5})",'
    r'permitlisten="127\.0\.0\.1:([0-9]{1,5})" '
    r'(ssh-ed25519) ([A-Za-z0-9+/]+={0,2}) node=([A-Za-z0-9_-]{1,100})$'
)


class RelayError(RuntimeError):
    """Expected, user-safe central relay setup error."""


@dataclass(frozen=True)
class RelaySetup:
    state_dir: Path
    config_path: Path
    authorized_keys_path: Path
    host_key_path: Path
    host_public_key_path: Path
    host_public_key: str
    sshd_path: Path
    listen_host: str
    port: int
    user: str
    sites: tuple[str, ...]
    reverse_listen_ports: tuple[int, ...]

    @property
    def start_argv(self) -> tuple[str, ...]:
        """Foreground command suitable for a service manager or isolated run."""
        return (str(self.sshd_path), "-D", "-e", "-f", str(self.config_path))

    @property
    def validate_argv(self) -> tuple[str, ...]:
        return (str(self.sshd_path), "-t", "-f", str(self.config_path))


@dataclass(frozen=True)
class RelaySystemdPlan:
    """Rendered user-unit location and explicit activation commands."""

    staged_unit_path: Path
    user_unit_path: Path
    install_argv: tuple[str, ...]
    daemon_reload_argv: tuple[str, ...]
    enable_start_argv: tuple[str, ...]


def _run(
    argv: list[str], *, runner: Callable[..., subprocess.CompletedProcess], timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    try:
        result = runner(argv, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RelayError(f"could not run {Path(argv[0]).name}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RelayError(f"{Path(argv[0]).name} rejected relay setup" + (f": {detail}" if detail else ""))
    return result


def _secure_dir(path: Path, *, create: bool) -> Path:
    path = Path(path).expanduser().absolute()
    # Refuse symlinked path components so private state cannot be redirected.
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise RelayError(f"required relay directory is missing: {current}")
            current.mkdir(mode=0o700)
            info = current.lstat()
        except OSError as exc:
            raise RelayError(f"cannot inspect relay directory: {current}") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise RelayError(f"relay path component is not a real directory: {current}")
    if create:
        try:
            ensure_private_dir(path)
        except IdentityError as exc:
            raise RelayError("could not safely prepare relay state directory") from exc
    return path


def _private_regular(path: Path) -> bytes:
    try:
        info = Path(path).lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise RelayError(f"Registry SSH export must be a private regular file: {path}")
        if info.st_uid != os.getuid():
            raise RelayError(f"Registry SSH export has a different owner: {path}")
        return Path(path).read_bytes()
    except RelayError:
        raise
    except OSError as exc:
        raise RelayError(f"Registry SSH export cannot be read: {path}") from exc


def _validate_ed25519_key(encoded: str) -> None:
    try:
        blob = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RelayError("Registry Gateway key has invalid base64") from exc
    algorithm = b"ssh-ed25519"
    if len(blob) != 4 + len(algorithm) + 4 + 32:
        raise RelayError("Registry Gateway key is not a complete Ed25519 key")
    if int.from_bytes(blob[:4], "big") != len(algorithm) or blob[4:4 + len(algorithm)] != algorithm:
        raise RelayError("Registry Gateway key type does not match its key data")
    offset = 4 + len(algorithm)
    if int.from_bytes(blob[offset:offset + 4], "big") != 32:
        raise RelayError("Registry Gateway Ed25519 key has an invalid public-key length")


def _site_grants(registry_state_dir: Path) -> tuple[list[str], tuple[str, ...], tuple[int, ...]]:
    root = Path(registry_state_dir).expanduser().absolute()
    try:
        root.lstat()
    except FileNotFoundError:
        # The relay host key must be available while building reverse-SSH
        # bootstrap defaults, before Controller initialization creates its
        # Registry directory. No registered files means no authorized keys.
        return [], (), ()
    root = _secure_dir(root, create=False)
    info = root.lstat()
    if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
        raise RelayError(f"Registry SSH directory must be owned mode 0700: {root}")
    ssh_root = root / "ssh"
    try:
        ssh_root.lstat()
    except FileNotFoundError:
        return [], (), ()
    sites_root = ssh_root / "sites"
    _secure_dir(ssh_root, create=False)
    info = ssh_root.lstat()
    if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
        raise RelayError(f"Registry SSH directory must be owned mode 0700: {ssh_root}")
    try:
        sites_root.lstat()
    except FileNotFoundError:
        return [], (), ()
    _secure_dir(sites_root, create=False)
    info = sites_root.lstat()
    if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
        raise RelayError(f"Registry SSH directory must be owned mode 0700: {sites_root}")
    keys: list[str] = []
    site_ids: list[str] = []
    ports: list[int] = []
    seen_nodes: set[str] = set()
    seen_ports: dict[int, str] = {}
    try:
        site_dirs = sorted(sites_root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise RelayError("Registry site SSH exports cannot be enumerated") from exc
    for directory in site_dirs:
        site_id = directory.name
        if not _SAFE_SITE.fullmatch(site_id):
            raise RelayError("Registry SSH export has an invalid site directory name")
        _secure_dir(directory, create=False)
        info = directory.lstat()
        if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
            raise RelayError(f"Registry site directory must be owned mode 0700: {directory}")
        inventory_path = directory / "inventory.json"
        auth_path = directory / "authorized_keys"
        try:
            inventory = json.loads(_private_regular(inventory_path).decode("utf-8"))
            auth_text = _private_regular(auth_path).decode("ascii")
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RelayError(f"Registry SSH export for site {site_id} is malformed") from exc
        if not isinstance(inventory, list):
            raise RelayError(f"Registry SSH inventory for site {site_id} is not a list")
        nodes: dict[str, dict[str, object]] = {}
        for row in inventory:
            if not isinstance(row, dict):
                raise RelayError(f"Registry SSH inventory for site {site_id} has a malformed node")
            node_id = row.get("node_id")
            if not isinstance(node_id, str) or not _SAFE_ID.fullmatch(node_id):
                raise RelayError(f"Registry SSH inventory for site {site_id} has an invalid node ID")
            if row.get("site_id") != site_id or row.get("role") not in {"worker", "gateway"}:
                raise RelayError(f"Registry SSH inventory for site {site_id} crosses its site boundary")
            if node_id in seen_nodes:
                raise RelayError(f"Registry SSH inventory repeats node ID {node_id}")
            seen_nodes.add(node_id)
            nodes[node_id] = row
        gateway_ids = [node_id for node_id, row in nodes.items() if row["role"] == "gateway"]
        lines = [line for line in auth_text.splitlines() if line]
        if not lines:
            # A site can enrol Workers before its Gateway. It receives no relay
            # authority until Registry exports a Gateway key with exact ports.
            continue
        if len(gateway_ids) != 1 or len(lines) != 1:
            raise RelayError(f"site {site_id} must have exactly one authorized Gateway key")
        match = _AUTH_LINE.fullmatch(lines[0])
        if match is None:
            raise RelayError(f"Registry Gateway grant for site {site_id} is not least-privilege")
        first_port, second_port = int(match.group(1)), int(match.group(2))
        key_type, encoded, node_id = match.group(3), match.group(4), match.group(5)
        if any(port < 1024 or port > 65535 for port in (first_port, second_port)) or first_port == second_port:
            raise RelayError(f"Registry Gateway grant for site {site_id} has invalid listeners")
        if node_id != gateway_ids[0] or nodes[node_id]["role"] != "gateway":
            raise RelayError(f"Registry Gateway grant for site {site_id} belongs to a different node")
        if key_type != "ssh-ed25519":
            raise RelayError(f"Registry Gateway grant for site {site_id} uses an unsupported key")
        _validate_ed25519_key(encoded)
        for port in (first_port, second_port):
            previous_site = seen_ports.get(port)
            if previous_site is not None:
                raise RelayError(
                    f"reverse listener port {port} is assigned to both {previous_site} and {site_id}"
                )
            seen_ports[port] = site_id
            ports.append(port)
        keys.append(lines[0])
        site_ids.append(site_id)
    return keys, tuple(site_ids), tuple(sorted(ports))


def refresh_central_relay_authorized_keys(
    registry_state_dir: Path, relay_state_dir: Path,
) -> tuple[Path, tuple[str, ...], tuple[int, ...]]:
    """Publish current Registry grants atomically without restarting sshd.

    Call this after successful node registration. sshd reads AuthorizedKeysFile
    for each authentication, so replacing this file does not require a daemon
    reload. Duplicate listener ports across sites fail closed.
    """
    state = _secure_dir(Path(relay_state_dir), create=True)
    keys, sites, ports = _site_grants(Path(registry_state_dir))
    path = state / "authorized_keys"
    try:
        atomic_write(path, ("\n".join(keys) + ("\n" if keys else "")).encode("ascii"), mode=0o600)
    except IdentityError as exc:
        raise RelayError("could not safely publish central relay authorized keys") from exc
    return path, sites, ports


def _listen_address(value: str) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() or ord(char) < 32 for char in value):
        raise RelayError("relay listen address must be a numeric IP address")
    try:
        return str(ipaddress.ip_address(value))
    except ValueError as exc:
        raise RelayError("relay listen address must be a numeric IP address") from exc


def _ssh_public_key(path: Path) -> str:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            raise RelayError("central relay host private key must be a private regular file")
        if info.st_uid != os.getuid():
            raise RelayError("central relay host private key has a different owner")
    except OSError as exc:
        raise RelayError("central relay host private key is unavailable") from exc
    public_path = Path(str(path) + ".pub")
    try:
        public_info = public_path.lstat()
        if (not stat.S_ISREG(public_info.st_mode) or stat.S_IMODE(public_info.st_mode) != 0o600
                or public_info.st_uid != os.getuid()):
            raise RelayError("central relay host public key must be an owned mode 0600 regular file")
        text = public_path.read_text(encoding="ascii").strip()
    except RelayError:
        raise
    except (OSError, UnicodeError) as exc:
        raise RelayError("central relay host public key is unavailable") from exc
    parts = text.split()
    if len(parts) < 2 or parts[0] != "ssh-ed25519":
        raise RelayError("central relay host public key is malformed")
    _validate_ed25519_key(parts[1])
    return "ssh-ed25519 " + parts[1]


def prepare_central_relay(
    *,
    registry_state_dir: Path,
    state_dir: Path,
    listen_host: str,
    port: int,
    user: str,
    sshd_path: Path | None = None,
    ssh_keygen_path: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> RelaySetup:
    """Create and validate a standalone central sshd configuration.

    The caller owns service activation. The returned paths and ``start_argv``
    provide the stable integration surface for the central CLI/server.
    """
    if type(port) is not int or not 1024 <= port <= 65535:
        raise RelayError("central relay port must be between 1024 and 65535")
    if not isinstance(user, str) or user == "root" or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", user):
        raise RelayError("central relay user is invalid")
    listen_host = _listen_address(listen_host)
    state = _secure_dir(Path(state_dir), create=True)
    sshd = Path(sshd_path) if sshd_path is not None else Path(shutil.which("sshd") or "/usr/sbin/sshd")
    if not sshd.is_file() or sshd.is_symlink():
        raise RelayError("OpenSSH server binary is unavailable")
    keygen = Path(ssh_keygen_path) if ssh_keygen_path is not None else Path(shutil.which("ssh-keygen") or "/usr/bin/ssh-keygen")
    if not keygen.is_file() or keygen.is_symlink():
        raise RelayError("OpenSSH key generator is unavailable")
    host_key = state / "ssh_host_ed25519_key"
    public_key_path = Path(str(host_key) + ".pub")
    if host_key.exists() != public_key_path.exists():
        raise RelayError("central relay host keypair is incomplete; refusing to replace it")
    if not host_key.exists():
        _run([str(keygen), "-q", "-t", "ed25519", "-N", "", "-C", "qwen-cluster-central-relay", "-f", str(host_key)], runner=runner)
        try:
            os.chmod(host_key, 0o600)
            os.chmod(public_key_path, 0o600)
        except OSError as exc:
            raise RelayError("could not secure central relay host key files") from exc
    host_public_key = _ssh_public_key(host_key)
    derived_host_key = _run([str(keygen), "-y", "-f", str(host_key)], runner=runner).stdout.strip().split()
    if len(derived_host_key) < 2 or " ".join(derived_host_key[:2]) != host_public_key:
        raise RelayError("central relay public key does not match its private host key")
    authorized_keys, sites, ports = refresh_central_relay_authorized_keys(registry_state_dir, state)
    config_path = state / "sshd_config"
    lines = [
        f"Port {port}", f"ListenAddress {listen_host}",
        f"PidFile {state / 'sshd.pid'}", f"HostKey {host_key}",
        f"AuthorizedKeysFile {authorized_keys}",
        "PubkeyAuthentication yes", "AuthenticationMethods publickey",
        "PasswordAuthentication no", "KbdInteractiveAuthentication no",
        "ChallengeResponseAuthentication no", "UsePAM no", "StrictModes yes",
        f"AllowUsers {user}", "PermitRootLogin no", "ForceCommand /usr/bin/false",
        "PermitTTY no", "PermitUserRC no", "PermitTunnel no",
        "AllowAgentForwarding no", "X11Forwarding no",
        "AllowTcpForwarding remote", "AllowStreamLocalForwarding no",
        "PermitOpen none", "GatewayPorts no", "LogLevel VERBOSE",
        "ClientAliveInterval 60", "ClientAliveCountMax 3",
    ]
    try:
        atomic_write(config_path, ("\n".join(lines) + "\n").encode("utf-8"), mode=0o600)
    except IdentityError as exc:
        raise RelayError("could not safely write central relay sshd config") from exc
    setup = RelaySetup(
        state_dir=state,
        config_path=config_path,
        authorized_keys_path=authorized_keys,
        host_key_path=host_key,
        host_public_key_path=public_key_path,
        host_public_key=host_public_key,
        sshd_path=sshd,
        listen_host=listen_host,
        port=port,
        user=user,
        sites=sites,
        reverse_listen_ports=ports,
    )
    _run(list(setup.validate_argv), runner=runner, timeout=15.0)
    return setup


def _systemd_arg(value: str | Path) -> str:
    text = str(value)
    if not text or any(char in text for char in "\r\n\x00"):
        raise RelayError("systemd arguments must be nonempty single-line values")
    if not any(char.isspace() or char in '\\"%$' for char in text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def render_central_relay_user_unit(setup: RelaySetup) -> str:
    """Render a non-root systemd user unit for the selected SSH login user."""
    try:
        current_user = pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError) as exc:
        raise RelayError("could not resolve the current service account") from exc
    if setup.user != current_user or setup.user == "root":
        raise RelayError("a user relay unit must run under the same non-root account as its SSH login")
    config = _systemd_arg(setup.config_path)
    sshd = _systemd_arg(setup.sshd_path)
    state = _systemd_arg(setup.state_dir)
    return "\n".join([
        "[Unit]",
        "Description=Qwen cluster central SSH relay",
        "After=network-online.target",
        "Wants=network-online.target",
        f"ConditionPathExists={config}",
        "",
        "[Service]",
        "Type=simple",
        "UMask=0077",
        f"ExecStartPre={sshd} -t -f {config}",
        f"ExecStart={sshd} -D -e -f {config}",
        "Restart=on-failure",
        "RestartSec=3s",
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
        "PrivateTmp=yes",
        f"ReadWritePaths={state}",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ])


def stage_central_relay_user_unit(setup: RelaySetup) -> RelaySystemdPlan:
    """Stage the rendered user unit and return, but do not run, activation commands.

    This helper writes only below the relay-owned state directory. It does not
    edit systemd directories, contact the service manager, or enable a service.
    """
    text = render_central_relay_user_unit(setup)
    staged_dir = _secure_dir(setup.state_dir / "systemd", create=True)
    staged = staged_dir / "qwen-cluster-relay.service"
    try:
        atomic_write(staged, text.encode("utf-8"), mode=0o600)
    except IdentityError as exc:
        raise RelayError("could not safely stage the central relay user unit") from exc
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()
    if not config_home.is_absolute():
        raise RelayError("XDG_CONFIG_HOME must be an absolute path")
    destination = config_home / "systemd" / "user" / "qwen-cluster-relay.service"
    return RelaySystemdPlan(
        staged_unit_path=staged,
        user_unit_path=destination,
        install_argv=("install", "-D", "-m", "0644", str(staged), str(destination)),
        daemon_reload_argv=("systemctl", "--user", "daemon-reload"),
        enable_start_argv=("systemctl", "--user", "enable", "--now", "qwen-cluster-relay.service"),
    )
