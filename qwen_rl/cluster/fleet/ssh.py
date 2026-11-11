"""Central Fleet SSH relay and pinned OpenSSH administrator client."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import pwd
import re
import shlex
import shutil
import stat
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from starlette.websockets import WebSocketDisconnect

from ..identity import (
    IdentityError,
    atomic_write,
    canonical_public_key,
    ensure_private_dir,
)
from .contracts import UserAccount
from .paths import UserPaths

if TYPE_CHECKING:
    from .contracts import FleetNodeView, FleetSSHGrant


_NODE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_SAFE_USER = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_NODE_SSH_UNIT = "qwen-fleet-node-ssh.service"
_TUNNEL_UNIT = "qwen-fleet-node-tunnel.service"
_RELAY_ALIAS = "qwen-fleet-central-relay"
_MAX_FRAME_BYTES = 65_536
_FLEET_WS_PATH = "/v1/fleet/ssh"


class FleetSSHError(RuntimeError):
    """Expected, redacted Fleet SSH configuration or OpenSSH error."""


@dataclass(frozen=True)
class FleetNodeSSHConfig:
    """Approved central grant and persistent node-owned SSH identity paths."""

    controller_url: str
    ca_bundle: Path
    bearer_file: Path
    reverse_port: int
    relay_host_key: str
    relay_user: str
    tunnel_identity_file: Path
    host_identity_file: Path
    admin_ssh_public_key: str
    management_generation: int
    python_executable: Path
    ssh_port: int = 22022
    relay_port: int = 22230


@dataclass(frozen=True)
class FleetNodeSSHReceipt:
    node_ssh_unit: str
    tunnel_unit: str
    user_ssh_config: Path
    tunnel_config: Path
    source_root: Path
    authorized_keys: Path
    relay_known_hosts: Path
    reverse_port: int
    host_public_key: str
    host_key_fingerprint: str
    management_generation: int
    changed: bool


@dataclass(frozen=True)
class FleetRelaySetup:
    relay_state_dir: Path
    admin_identity_file: Path
    admin_public_key: str
    relay_host_key: str
    relay_host_public_key: Path
    config_path: Path
    authorized_keys_path: Path
    unit_path: Path
    start_argv: tuple[str, ...]


def _private_regular(path: Path, *, exact_mode: int = 0o600) -> None:
    try:
        info = Path(path).lstat()
    except OSError as exc:
        raise FleetSSHError(f"required private file is unavailable: {path}") from exc
    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != exact_mode
            or (os.name == "posix" and info.st_uid != os.getuid())):
        raise FleetSSHError(f"private file must be owned regular mode {exact_mode:04o}: {path}")


def _public_regular(path: Path) -> None:
    try:
        info = Path(path).lstat()
    except OSError as exc:
        raise FleetSSHError(f"required public file is unavailable: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o022:
        raise FleetSSHError(f"public file must be a regular file not writable by group or others: {path}")


def _ed25519_public(value: str, label: str) -> str:
    try:
        key = canonical_public_key(value)
    except (IdentityError, ValueError) as exc:
        raise FleetSSHError(f"{label} is not a valid SSH public key") from exc
    parts = key.split()
    if parts[0] != "ssh-ed25519":
        raise FleetSSHError(f"{label} must use Ed25519")
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except ValueError as exc:
        raise FleetSSHError(f"{label} has invalid base64") from exc
    algorithm = b"ssh-ed25519"
    if (len(blob) != 4 + len(algorithm) + 4 + 32
            or int.from_bytes(blob[:4], "big") != len(algorithm)
            or blob[4:4 + len(algorithm)] != algorithm
            or int.from_bytes(blob[4 + len(algorithm):8 + len(algorithm)], "big") != 32):
        raise FleetSSHError(f"{label} is not a complete Ed25519 key")
    return "ssh-ed25519 " + parts[1]


def _fingerprint(public_key: str) -> str:
    parts = _ed25519_public(public_key, "SSH public key").split()
    digest = base64.b64encode(hashlib.sha256(base64.b64decode(parts[1])).digest()).decode("ascii").rstrip("=")
    return "SHA256:" + digest


def _port(value: int, label: str) -> int:
    if type(value) is not int or not 1024 <= value <= 65535:
        raise FleetSSHError(f"{label} must be an integer from 1024 through 65535")
    return value


def _node_ssh_port(value: int) -> int:
    if type(value) is not int or value != 22022:
        raise FleetSSHError("Fleet management SSH must use the dedicated loopback port 22022")
    return value


def _loopback_host(value: str) -> str:
    if not isinstance(value, str):
        raise FleetSSHError("SSH relay host must be a numeric loopback address")
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise FleetSSHError("SSH relay host must be a numeric loopback address") from exc
    if not address.is_loopback:
        raise FleetSSHError("SSH relay host must be loopback")
    return str(address)


def _validate_websocket_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise FleetSSHError("Fleet SSH URL is invalid") from exc
    if (parsed.scheme != "wss" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path != _FLEET_WS_PATH or parsed.query or parsed.fragment
            or (port is not None and not 1 <= port <= 65535)):
        raise FleetSSHError("Fleet SSH URL must use the configured WSS origin at /v1/fleet/ssh")
    return value


def _controller_websocket_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise FleetSSHError("Fleet controller URL is invalid") from exc
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
            or (port is not None and not 1 <= port <= 65535)):
        raise FleetSSHError("Fleet controller URL must be a valid HTTPS origin")
    return f"wss://{parsed.netloc}{_FLEET_WS_PATH}"


def _private_dir(path: Path, *, create: bool) -> Path:
    path = Path(path).expanduser().absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise FleetSSHError(f"required SSH directory is missing: {current}")
            current.mkdir(mode=0o700)
            info = current.lstat()
        except OSError as exc:
            raise FleetSSHError(f"cannot inspect SSH directory: {current}") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise FleetSSHError(f"SSH path component is not a real directory: {current}")
    if create:
        try:
            ensure_private_dir(path)
        except IdentityError as exc:
            raise FleetSSHError("could not safely prepare Fleet SSH directory") from exc
    else:
        info = path.lstat()
        if stat.S_IMODE(info.st_mode) != 0o700 or (os.name == "posix" and info.st_uid != os.getuid()):
            raise FleetSSHError(f"Fleet SSH directory must be owned mode 0700: {path}")
    return path


def _write_unit(path: Path, content: str) -> bool:
    """Atomically write a systemd unit; skip byte-identical content and report whether it wrote.

    Rewriting an unchanged unit still bumps its mtime/inode, which makes ``systemd --user``
    report ``NeedDaemonReload=yes`` and turns every reconcile tick into a daemon-reload
    (visible in the journal as "Reloading requested from client PID ..." several times a
    minute).  Only a real content change may count as a unit change.
    """
    path = Path(path).absolute()
    try:
        if path.is_symlink():
            raise FleetSSHError(f"refusing to replace symlinked unit file: {path}")
        if path.is_file():
            try:
                if path.read_text(encoding="utf-8") == content:
                    return False
            except (OSError, UnicodeError):
                pass
        import tempfile
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            os.fchmod(target.fileno(), 0o644)
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        return True
    except FleetSSHError:
        raise
    except OSError as exc:
        raise FleetSSHError("could not safely write Fleet SSH unit") from exc


def _config_quote(value: str | Path) -> str:
    text = str(value)
    if not text or any(char in text for char in "\r\n\x00"):
        raise FleetSSHError("OpenSSH configuration value must be a non-empty single line")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _systemd_quote(value: str | Path) -> str:
    text = str(value)
    if not text or any(char in text for char in "\r\n\x00"):
        raise FleetSSHError("systemd argument must be a non-empty single line")
    if not any(char.isspace() or char in '\\"%$' for char in text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def _systemd_path(value: str | Path) -> str:
    """Escape one systemd path directive value without adding quote characters."""
    text = str(value)
    if not text or any(char in text for char in "\r\n\x00"):
        raise FleetSSHError("systemd path must be a non-empty single line")
    escaped: list[str] = []
    for char in text:
        if char == "\\":
            escaped.append("\\\\")
        elif char == "%":
            escaped.append("%%")
        elif char == '"' or char.isspace():
            escaped.append(f"\\x{ord(char):02x}")
        else:
            escaped.append(char)
    return "".join(escaped)


def _run(
    argv: Sequence[str], *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 30.0, input_text: str | None = None,
) -> subprocess.CompletedProcess:
    try:
        result = runner(list(argv), check=False, capture_output=True, text=True,
                        timeout=timeout, input=input_text)
    except (OSError, subprocess.SubprocessError) as exc:
        raise FleetSSHError(f"could not run {Path(argv[0]).name}") from exc
    if result.returncode != 0:
        raise FleetSSHError(f"{Path(argv[0]).name} rejected Fleet SSH configuration")
    return result


def _ssh_key_from_private(
    private_key: Path, *, runner: Callable[..., subprocess.CompletedProcess],
) -> str:
    _private_regular(private_key)
    result = _run([shutil.which("ssh-keygen") or "/usr/bin/ssh-keygen", "-y", "-f", str(private_key)],
                  runner=runner, timeout=20.0)
    return _ed25519_public(result.stdout.strip(), "derived node SSH key")


def render_relay_authorized_key(node_id: str, public_key: str, reverse_port: int) -> str:
    """Render one key that can request only its assigned loopback listener."""
    if not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id):
        raise FleetSSHError("Fleet node ID is invalid")
    port = _port(reverse_port, "reverse SSH listener")
    key = _ed25519_public(public_key, "node tunnel public key")
    return (f'restrict,command="/usr/bin/false",port-forwarding,'
            f'permitlisten="127.0.0.1:{port}" {key} node={node_id}')


def _grant_tuple(grant: Any) -> tuple[str, str, int]:
    if isinstance(grant, (tuple, list)) and len(grant) == 3:
        node_id, public_key, reverse_port = grant
    else:
        node_id = _node_value(grant, "node_id", required=True)
        public_key = _node_value(grant, "ssh_public_key", required=True)
        reverse_port = _node_value(grant, "reverse_port", required=True)
    return node_id, public_key, reverse_port


def render_relay_authorized_keys(grants: Iterable[Any]) -> bytes:
    """Render a complete least-privilege relay authorization file.

    Duplicate node IDs, public keys, or reverse listener ports are rejected so
    a Registry refresh cannot silently transfer a listener between nodes.
    """
    lines: list[str] = []
    node_ids: set[str] = set()
    fingerprints: set[str] = set()
    ports: set[int] = set()
    for grant in grants:
        node_id, public_key, reverse_port = _grant_tuple(grant)
        line = render_relay_authorized_key(node_id, public_key, reverse_port)
        key = _ed25519_public(public_key, "node tunnel public key")
        fingerprint = _fingerprint(key)
        if node_id in node_ids or fingerprint in fingerprints or reverse_port in ports:
            raise FleetSSHError("relay grants contain a duplicate node, key, or listener")
        node_ids.add(node_id)
        fingerprints.add(fingerprint)
        ports.add(reverse_port)
        lines.append(line)
    return ("\n".join(sorted(lines)) + ("\n" if lines else "")).encode("ascii")


def write_relay_authorized_keys(path: Path, grants: Iterable[Any]) -> Path:
    """Atomically refresh the owned relay's key file after Registry changes."""
    try:
        atomic_write(Path(path), render_relay_authorized_keys(grants), mode=0o600)
    except IdentityError as exc:
        raise FleetSSHError("could not safely publish Fleet relay authorization") from exc
    return Path(path)


def _ensure_ed25519_keypair(
    private_key: Path, comment: str, *, runner: Callable[..., subprocess.CompletedProcess],
) -> tuple[Path, str]:
    public_path = Path(str(private_key) + ".pub")
    if private_key.exists() != public_path.exists():
        raise FleetSSHError(f"Fleet SSH keypair is incomplete: {private_key}")
    if not private_key.exists():
        _run([shutil.which("ssh-keygen") or "/usr/bin/ssh-keygen", "-q", "-t", "ed25519", "-N", "",
              "-C", comment, "-f", str(private_key)], runner=runner, timeout=30.0)
        try:
            os.chmod(private_key, 0o600)
            os.chmod(public_path, 0o644)
        except OSError as exc:
            raise FleetSSHError("could not secure Fleet SSH identity") from exc
    _private_regular(private_key)
    key = _ssh_key_from_private(private_key, runner=runner)
    try:
        public = canonical_public_key(public_path.read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError) as exc:
        raise FleetSSHError("Fleet SSH public identity is unavailable or malformed") from exc
    if key != _ed25519_public(public, "Fleet SSH public key"):
        raise FleetSSHError("Fleet SSH public and private identity do not match")
    return private_key, key


def prepare_fleet_ssh(
    state_dir: Path, *, relay_user: str, relay_host: str = "127.0.0.1", relay_port: int = 22230,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> FleetRelaySetup:
    """Create the separate Fleet relay/admin identities and owned relay config.

    The helper never writes ``~/.ssh`` or the host sshd configuration and does
    not install, reload, or start a system service.
    """
    if not isinstance(relay_user, str) or not _SAFE_USER.fullmatch(relay_user) or relay_user == "root":
        raise FleetSSHError("Fleet relay user must be a non-root local account")
    host = _loopback_host(relay_host)
    port = _port(relay_port, "Fleet relay port")
    try:
        account = pwd.getpwnam(relay_user)
        current_user = pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError) as exc:
        raise FleetSSHError("Fleet relay user must be the existing local service account") from exc
    if account.pw_uid != os.getuid() or relay_user != current_user:
        raise FleetSSHError("Fleet relay setup must run as its non-root service account")
    root = _private_dir(Path(state_dir).expanduser().absolute() / "ssh", create=True)
    admin_dir = _private_dir(root / "admin", create=True)
    relay_dir = _private_dir(root / "relay", create=True)
    admin_key_path, admin_key = _ensure_ed25519_keypair(
        admin_dir / "id_ed25519", "qwen-fleet-central-admin", runner=runner,
    )
    relay_key_path, relay_key = _ensure_ed25519_keypair(
        relay_dir / "ssh_host_ed25519_key", "qwen-fleet-central-relay", runner=runner,
    )
    if hmac.compare_digest(admin_key, relay_key):
        raise FleetSSHError("Fleet admin and relay host keys must be distinct")
    config_path = relay_dir / "sshd_config"
    authorized_keys = relay_dir / "authorized_keys"
    unit_dir = _private_dir(relay_dir / "systemd", create=True)
    unit_path = unit_dir / "qwen-fleet-ssh-relay.service"
    if not authorized_keys.exists():
        try:
            atomic_write(authorized_keys, b"", mode=0o600)
        except IdentityError as exc:
            raise FleetSSHError("could not safely create Fleet relay authorization file") from exc
    else:
        _private_regular(authorized_keys)
    lines = [
        f"Port {port}", f"ListenAddress {host}", f"PidFile {relay_dir / 'sshd.pid'}",
        f"HostKey {relay_key_path}", f"AuthorizedKeysFile {authorized_keys}",
        "PubkeyAuthentication yes", "AuthenticationMethods publickey", "PasswordAuthentication no",
        "KbdInteractiveAuthentication no", "ChallengeResponseAuthentication no", "UsePAM no",
        "StrictModes yes", f"AllowUsers {relay_user}", "PermitRootLogin no", "ForceCommand /usr/bin/false",
        "PermitTTY no", "PermitUserRC no", "PermitTunnel no", "AllowAgentForwarding no",
        "X11Forwarding no", "AllowTcpForwarding remote", "AllowStreamLocalForwarding no",
        "PermitOpen none", "GatewayPorts no", "LogLevel VERBOSE", "ClientAliveInterval 60",
        "ClientAliveCountMax 3", "",
    ]
    try:
        atomic_write(config_path, ("\n".join(lines) + "\n").encode("utf-8"), mode=0o600)
    except IdentityError as exc:
        raise FleetSSHError("could not safely write Fleet relay sshd config") from exc
    sshd = Path(shutil.which("sshd") or "/usr/sbin/sshd")
    if not sshd.is_file() or sshd.is_symlink():
        raise FleetSSHError("OpenSSH server binary is unavailable")
    _run([str(sshd), "-t", "-f", str(config_path)], runner=runner, timeout=20.0)
    _write_unit(unit_path, "\n".join([
        "[Unit]", "Description=Qwen Fleet loopback SSH relay", "After=network.target", "",
        "[Service]", "Type=simple", "UMask=0077", f"ExecStartPre={_systemd_quote(sshd)} -t -f {_systemd_quote(config_path)}",
        f"ExecStart={_systemd_quote(sshd)} -D -e -f {_systemd_quote(config_path)}",
        "Restart=on-failure", "RestartSec=3s", "NoNewPrivileges=yes", "ProtectSystem=strict",
        "ProtectHome=read-only", "PrivateTmp=yes", f"ReadWritePaths={_systemd_quote(relay_dir)}", "",
        "[Install]", "WantedBy=default.target", "",
    ]))
    host_public_path = Path(str(relay_key_path) + ".pub")
    return FleetRelaySetup(
        relay_state_dir=relay_dir,
        admin_identity_file=admin_key_path,
        admin_public_key=admin_key,
        relay_host_key=relay_key,
        relay_host_public_key=host_public_path,
        config_path=config_path,
        authorized_keys_path=authorized_keys,
        unit_path=unit_path,
        start_argv=(str(sshd), "-D", "-e", "-f", str(config_path)),
    )


def _node_ssh_fingerprint(node: Any) -> str:
    return str(_node_value(node, "ssh_host_public_key", required=True))


def _node_value(node: Any, name: str, *, required: bool = False) -> Any:
    if isinstance(node, dict):
        value = node.get(name)
    else:
        value = getattr(node, name, None)
    if required and value is None:
        raise FleetSSHError(f"Fleet node record is missing {name}")
    return value


def _node_management_user(node: Any) -> str:
    account = _node_value(node, "account", required=True)
    if isinstance(account, dict):
        username, uid = account.get("username"), account.get("uid")
    else:
        username, uid = getattr(account, "username", None), getattr(account, "uid", None)
    if (not isinstance(username, str) or not _SAFE_USER.fullmatch(username)
            or username == "root" or type(uid) is not int or uid <= 0):
        raise FleetSSHError("registered Fleet node account is invalid")
    return username


def admin_ssh_argv(
    node: FleetNodeView | Any, command: Sequence[str] | None, *,
    state_dir: Path, identity_file: Path,
    ssh_path: Path | None = None, interactive: bool = False,
) -> list[str]:
    """Build a local OpenSSH command pinned to one approved node and listener."""
    node_id = _node_value(node, "node_id", required=True)
    if not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id):
        raise FleetSSHError("Fleet node ID is invalid")
    reverse_port = _port(_node_value(node, "reverse_port", required=True), "reverse SSH listener")
    user = _node_management_user(node)
    _private_regular(identity_file)
    host_key = _ed25519_public(_node_ssh_fingerprint(node), "registered node host key")
    expected_fingerprint = _node_value(node, "ssh_host_fingerprint")
    if expected_fingerprint is not None and expected_fingerprint != _fingerprint(host_key):
        raise FleetSSHError("registered node host-key fingerprint does not match its public key")
    directory = _private_dir(Path(state_dir).expanduser().absolute() / "ssh" / "fleet" / node_id, create=True)
    known_hosts = directory / "known_hosts"
    config_path = directory / "client.conf"
    try:
        atomic_write(known_hosts, f"{node_id} {host_key}\n".encode("ascii"), mode=0o600)
        content = "\n".join([
            f"Host {node_id}", "  HostName 127.0.0.1", f"  Port {reverse_port}",
            f"  User {user}", f"  HostKeyAlias {node_id}",
            f"  IdentityFile {_config_quote(identity_file.absolute())}", "  IdentitiesOnly yes",
            f"  UserKnownHostsFile {_config_quote(known_hosts)}", "  StrictHostKeyChecking yes",
            "  GlobalKnownHostsFile /dev/null", "  VerifyHostKeyDNS no", "  UpdateHostKeys no",
            "  CheckHostIP no", "  PasswordAuthentication no", "  KbdInteractiveAuthentication no",
            "  BatchMode yes", "  ForwardAgent no", "  ForwardX11 no",
            f"  RequestTTY {'force' if interactive else 'no'}",
            "  ProxyCommand none", "  ProxyJump none", "  ControlMaster no",
            "  ConnectTimeout 10", "  ServerAliveInterval 15", "  ServerAliveCountMax 3", "",
        ])
        atomic_write(config_path, content.encode("utf-8"), mode=0o600)
    except IdentityError as exc:
        raise FleetSSHError("could not safely write pinned central SSH configuration") from exc
    tty_option = "-tt" if interactive else "-T"
    argv = [str(ssh_path or Path(shutil.which("ssh") or "/usr/bin/ssh")), "-F", str(config_path), tty_option, node_id]
    if command is not None:
        if not command or any(not isinstance(part, str) or "\x00" in part for part in command):
            raise FleetSSHError("remote command must be a non-empty argument list")
        argv.append(shlex.join(list(command)))
    return argv


def admin_local_forward_argv(
    node: FleetNodeView | Any, *, local_port: int, remote_port: int,
    state_dir: Path, identity_file: Path,
    ssh_path: Path | None = None,
) -> list[str]:
    """Build a narrowly scoped loopback-to-loopback central SSH forward."""
    local = _port(local_port, "local SSH forward port")
    remote = _port(remote_port, "node loopback forward port")
    node_id = _node_value(node, "node_id", required=True)
    if not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id):
        raise FleetSSHError("Fleet node ID is invalid")
    base = admin_ssh_argv(node, None, state_dir=state_dir, identity_file=identity_file,
                          ssh_path=ssh_path)
    return [*base[:-1], "-N", "-L", f"127.0.0.1:{local}:127.0.0.1:{remote}", base[-1]]


def run_admin_ssh(
    node: FleetNodeView | Any, command: Sequence[str] | None, *,
    state_dir: Path, identity_file: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float | None = 120.0, interactive: bool = False,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run pinned SSH with captured results or inherited terminal streams.

    ``interactive=True`` requests a forced PTY, inherited stdin/stdout/stderr,
    and no timeout. ``capture_output=False`` inherits
    stdio without allocating a PTY, which suits streamed CLI command output.
    """
    if interactive:
        capture_output = False
        timeout = None
    argv = admin_ssh_argv(
        node, command, state_dir=state_dir, identity_file=identity_file,
        interactive=interactive,
    )
    try:
        kwargs: dict[str, Any] = {"check": False, "timeout": timeout}
        if capture_output:
            kwargs.update(capture_output=True, text=True)
        result = runner(argv, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        raise FleetSSHError("could not start the Fleet SSH command") from exc
    if result.returncode != 0:
        raise FleetSSHError(f"Fleet SSH command for {_node_value(node, 'node_id', required=True)} exited with status {result.returncode}")
    return result


def _validate_account_paths(account: UserAccount, paths: UserPaths) -> pwd.struct_passwd:
    try:
        current = pwd.getpwuid(os.geteuid())
    except (KeyError, OSError) as exc:
        raise FleetSSHError("Fleet SSH must run as the registered local account") from exc
    if (os.geteuid() == 0 or os.getuid() != os.geteuid() or os.getgid() != os.getegid()
            or account.uid != os.geteuid() or account.gid != current.pw_gid
            or account.username != current.pw_name or account.home != current.pw_dir):
        raise FleetSSHError("Fleet SSH account differs from the current non-root UID, GID, name, or home")
    expected = UserPaths.from_account(account)
    for name in ("home", "root", "management_dir", "runtime_dir", "trust_dir", "state_dir",
                 "roles_dir", "systemd_user_dir"):
        if Path(getattr(paths, name)) != Path(getattr(expected, name)):
            raise FleetSSHError("Fleet SSH paths must come from the registered local account")
    return current


def _safe_user_systemd_dir(path: Path, *, home: Path) -> Path:
    """Create missing unit-directory components without changing existing modes."""
    path = Path(path).expanduser().absolute()
    home = Path(home).expanduser().absolute()
    if not path.is_relative_to(home):
        raise FleetSSHError("Fleet user systemd directory must be beneath the registered account home")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise FleetSSHError("could not create Fleet user systemd directory") from exc
            try:
                info = current.lstat()
            except OSError as exc:
                raise FleetSSHError("Fleet user systemd directory cannot be inspected") from exc
        except OSError as exc:
            raise FleetSSHError("Fleet user systemd directory cannot be inspected") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise FleetSSHError("Fleet user systemd path contains a non-directory component")
        if ((current == home or current.is_relative_to(home))
                and (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022)):
            raise FleetSSHError("Fleet user systemd directories must be user-owned and not group/world writable")
    return path


def _source_root(paths: UserPaths) -> Path:
    """Pin user-unit imports to the loaded release beneath Fleet-owned paths."""
    source = Path(__file__).resolve().parents[3]
    releases = Path(paths.management_dir).resolve() / "releases"
    try:
        relative = source.relative_to(releases)
    except ValueError as exc:
        raise FleetSSHError("Fleet SSH source must be a content-addressed user management release") from exc
    if len(relative.parts) != 1 or re.fullmatch(r"[0-9a-f]{64}", relative.name) is None:
        raise FleetSSHError("Fleet SSH source release must use its 64-character SHA256 directory")
    return source


def _management_payload(config: FleetNodeSSHConfig, account: UserAccount) -> dict[str, Any]:
    python = Path(config.python_executable)
    paths = UserPaths.from_account(account)
    managed_roots = (Path(paths.management_dir), Path(paths.runtime_dir))
    if not python.is_absolute() or not any(python.is_relative_to(root) for root in managed_roots):
        raise FleetSSHError("Fleet user Python path must be inside the account management or runtime directory")
    if type(config.management_generation) is not int or config.management_generation < 1:
        raise FleetSSHError("management generation must be positive for approved Fleet SSH")
    if (not isinstance(config.relay_user, str) or not _SAFE_USER.fullmatch(config.relay_user)
            or config.relay_user == "root"):
        raise FleetSSHError("central relay SSH user is invalid")
    return {
        "account": account.model_dump(mode="json"),
        "websocket_url": _controller_websocket_url(config.controller_url),
        "ca_bundle": str(Path(config.ca_bundle).absolute()),
        "bearer_file": str(Path(config.bearer_file).absolute()),
        "reverse_port": _port(config.reverse_port, "allocated reverse SSH listener"),
        "ssh_port": _node_ssh_port(config.ssh_port),
        "relay_host_key": _ed25519_public(config.relay_host_key, "central relay host key"),
        "relay_user": config.relay_user,
        "tunnel_identity_file": str(Path(config.tunnel_identity_file).absolute()),
        "host_identity_file": str(Path(config.host_identity_file).absolute()),
        "admin_ssh_public_key": _ed25519_public(config.admin_ssh_public_key, "central admin public key"),
        "relay_port": _port(config.relay_port, "Fleet central relay port"),
        "python_executable": str(python.absolute()),
        "source_root": str(_source_root(UserPaths.from_account(account))),
        "management_generation": config.management_generation,
    }


def _user_unit_text(
    config: FleetNodeSSHConfig, server_config: Path, tunnel_config: Path,
    source_root: Path, home: Path,
) -> tuple[str, str]:
    python = _systemd_quote(Path(config.python_executable).absolute())
    working_dir = _systemd_path(home)
    python_path_env = _systemd_quote(f"PYTHONPATH={source_root}")
    server = "\n".join([
        "[Unit]", "Description=Qwen Fleet user SSH server", "",
        "[Service]", "Type=simple", "UMask=0077", f"WorkingDirectory={working_dir}",
        f"Environment={python_path_env}",
        f"ExecStart={python} -m qwen_rl.cluster.fleet.user_ssh serve --config {_systemd_quote(server_config)}",
        "Restart=on-failure", "RestartSec=3s", "NoNewPrivileges=yes", "",
        "[Install]", "WantedBy=default.target", "",
    ])
    tunnel = "\n".join([
        "[Unit]", "Description=Qwen Fleet authenticated reverse SSH tunnel",
        f"After={_NODE_SSH_UNIT}", f"Requires={_NODE_SSH_UNIT}", "",
        "[Service]", "Type=simple", "UMask=0077", f"WorkingDirectory={working_dir}",
        f"Environment={python_path_env}",
        f"ExecStart={python} -m qwen_rl.cluster.fleet.user_ssh tunnel --config {_systemd_quote(tunnel_config)}",
        "Restart=always", "RestartSec=5s", "WatchdogSec=60s", "WatchdogSignal=SIGKILL", "LimitCORE=0", "NotifyAccess=main",
        "NoNewPrivileges=yes", "",
        "[Install]", "WantedBy=default.target", "",
    ])
    return server, tunnel


def ensure_node_ssh(
    config: FleetNodeSSHConfig, *, account: UserAccount, paths: UserPaths,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> FleetNodeSSHReceipt:
    """Stage account-owned AsyncSSH server and WSS reverse-tunnel user units."""
    _validate_account_paths(account, paths)
    payload = _management_payload(config, account)
    state = _private_dir(Path(paths.state_dir), create=True)
    ssh_dir = _private_dir(state / "ssh", create=True)
    systemd_dir = _safe_user_systemd_dir(Path(paths.systemd_user_dir), home=Path(paths.home))
    manifest_path = ssh_dir / "fleet-management.json"
    try:
        if manifest_path.exists():
            _private_regular(manifest_path)
        old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FleetSSHError("Fleet SSH ownership record cannot be read") from exc
    changed = True
    if old is not None:
        if not isinstance(old, dict) or type(old.get("management_generation")) is not int:
            raise FleetSSHError("Fleet SSH ownership record is malformed")
        prior_generation = old["management_generation"]
        if config.management_generation < prior_generation:
            raise FleetSSHError("stale Fleet management generation cannot replace SSH setup")
        if config.management_generation == prior_generation and old.get("config") != payload:
            raise FleetSSHError("Fleet SSH settings changed without a new management generation")
        changed = old.get("config") != payload

    from .user_ssh import read_asyncssh_public_key

    ca_file = Path(config.ca_bundle).absolute()
    bearer_file = Path(config.bearer_file).absolute()
    _public_regular(ca_file)
    _private_regular(bearer_file)
    tunnel_key_path = Path(config.tunnel_identity_file).absolute()
    host_key_path = Path(config.host_identity_file).absolute()
    tunnel_key = read_asyncssh_public_key(tunnel_key_path)
    host_key = read_asyncssh_public_key(host_key_path)
    if hmac.compare_digest(tunnel_key, host_key):
        raise FleetSSHError("node tunnel and SSH host keys must be distinct")
    admin_key = _ed25519_public(config.admin_ssh_public_key, "central admin public key")
    relay_key = _ed25519_public(config.relay_host_key, "central relay host key")
    known_hosts = ssh_dir / "relay_known_hosts"
    authorized_keys = ssh_dir / "authorized_keys"
    server_config = ssh_dir / "user_ssh.json"
    tunnel_config = ssh_dir / "reverse_tunnel.json"
    try:
        atomic_write(known_hosts, f"{_RELAY_ALIAS} {relay_key}\n".encode("ascii"), mode=0o600)
        atomic_write(authorized_keys, f"{admin_key} fleet-central-admin\n".encode("ascii"), mode=0o600)
        server_payload = {
            "version": 1, "account": account.model_dump(mode="json"),
            "host_key": str(host_key_path), "authorized_keys": str(authorized_keys),
            "listen_host": "127.0.0.1", "listen_port": _node_ssh_port(config.ssh_port),
        }
        tunnel_payload = {
            "version": 1, "account": account.model_dump(mode="json"),
            "controller_url": str(config.controller_url), "ca_bundle": str(ca_file),
            "bearer_file": str(bearer_file),
            "reverse_port": _port(config.reverse_port, "allocated reverse SSH listener"),
            "relay_port": _port(config.relay_port, "Fleet central relay port"),
            "relay_user": config.relay_user, "relay_host_key": relay_key,
            "relay_known_hosts": str(known_hosts), "tunnel_key": str(tunnel_key_path),
            "listen_host": "127.0.0.1", "listen_port": _port(config.reverse_port, "assigned reverse listener"),
            "local_host": "127.0.0.1", "local_port": _node_ssh_port(config.ssh_port),
        }
        atomic_write(server_config, (json.dumps(server_payload, sort_keys=True, indent=2) + "\n").encode(), mode=0o600)
        atomic_write(tunnel_config, (json.dumps(tunnel_payload, sort_keys=True, indent=2) + "\n").encode(), mode=0o600)
        manifest = {"management_generation": config.management_generation, "config": payload,
                    "needs_restart": bool(changed or (old and old.get("needs_restart"))),
                    "units_changed": bool(changed or (old and old.get("units_changed")))}
        atomic_write(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(), mode=0o600)
    except IdentityError as exc:
        raise FleetSSHError("could not safely write Fleet user SSH configuration") from exc

    source_root = Path(payload["source_root"])
    server_unit, tunnel_unit = _user_unit_text(
        config, server_config, tunnel_config, source_root, Path(paths.home),
    )
    units_changed = False
    units_changed |= _write_unit(systemd_dir / _NODE_SSH_UNIT, server_unit)
    units_changed |= _write_unit(systemd_dir / _TUNNEL_UNIT, tunnel_unit)
    if units_changed:
        try:
            manifest["units_changed"] = True
            atomic_write(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(), mode=0o600)
        except IdentityError as exc:
            raise FleetSSHError("could not safely record the Fleet SSH unit change") from exc
    return FleetNodeSSHReceipt(
        node_ssh_unit=_NODE_SSH_UNIT, tunnel_unit=_TUNNEL_UNIT,
        user_ssh_config=server_config, tunnel_config=tunnel_config,
        source_root=source_root,
        authorized_keys=authorized_keys, relay_known_hosts=known_hosts,
        reverse_port=_port(config.reverse_port, "allocated reverse listener"),
        host_public_key=host_key, host_key_fingerprint=_fingerprint(host_key),
        management_generation=config.management_generation, changed=changed,
    )


def _units_need_daemon_reload(
    runner: Callable[..., subprocess.CompletedProcess], systemctl: str, units: tuple[str, ...],
) -> bool:
    """True when systemd --user still holds a stale copy of one of these unit files.

    ``NeedDaemonReload=yes`` is systemd's own "the file on disk differs from what I loaded"
    flag, so it is exactly the condition under which a reload does any work.  An
    unanswerable query (unknown unit, missing binary, timeout) counts as yes, so a fresh or
    broken node still gets its reload.
    """
    for unit in units:
        try:
            result = runner([systemctl, "--user", "show", "--value", "-p", "NeedDaemonReload", unit],
                            check=False, capture_output=True, text=True, timeout=15.0)
        except (OSError, subprocess.SubprocessError):
            return True
        if getattr(result, "returncode", 1) != 0:
            return True
        if (getattr(result, "stdout", "") or "").strip().lower() != "no":
            return True
    return False


def start_node_ssh(
    config: FleetNodeSSHConfig, *, account: UserAccount, paths: UserPaths,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[str, ...]:
    """Start or restart only the Fleet-owned user SSH and tunnel units."""
    _validate_account_paths(account, paths)
    ssh_dir = _private_dir(Path(paths.state_dir) / "ssh", create=False)
    manifest_path = ssh_dir / "fleet-management.json"
    try:
        _private_regular(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FleetSSHError("Fleet SSH setup must be prepared before starting its user units") from exc
    if not isinstance(manifest, dict) or manifest.get("management_generation") != config.management_generation:
        raise FleetSSHError("Fleet SSH start requested for a different management generation")
    expected = _management_payload(config, account)
    if manifest.get("config") != expected:
        raise FleetSSHError("Fleet SSH start settings differ from the staged ownership record")
    systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
    if manifest.get("units_changed") or _units_need_daemon_reload(
            runner, systemctl, (_NODE_SSH_UNIT, _TUNNEL_UNIT)):
        _run([systemctl, "--user", "daemon-reload"], runner=runner, timeout=30.0)
        manifest["units_changed"] = False
        manifest["_manifest_dirty"] = True
    changed = bool(manifest.get("needs_restart"))
    for unit in (_NODE_SSH_UNIT, _TUNNEL_UNIT):
        try:
            active = runner([systemctl, "--user", "is-active", "--quiet", unit], check=False,
                            capture_output=True, text=True, timeout=15.0).returncode == 0
        except (OSError, subprocess.SubprocessError):
            active = False
        _run([systemctl, "--user", "enable", unit], runner=runner, timeout=30.0)
        if active and changed:
            _run([systemctl, "--user", "restart", unit], runner=runner, timeout=120.0)
        elif not active:
            _run([systemctl, "--user", "start", unit], runner=runner, timeout=120.0)
    write_manifest = bool(manifest.pop("_manifest_dirty", False))
    if changed:
        manifest["needs_restart"] = False
        write_manifest = True
    if write_manifest:
        try:
            atomic_write(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(), mode=0o600)
        except IdentityError as exc:
            raise FleetSSHError("could not finalize Fleet SSH unit activation receipt") from exc
    return (_NODE_SSH_UNIT, _TUNNEL_UNIT)


async def handle_ssh_websocket(
    websocket: Any, grant: FleetSSHGrant | Any, relay_host: str = "127.0.0.1",
    relay_port: int = 22230, max_frame_bytes: int = _MAX_FRAME_BYTES,
) -> None:
    """Bridge an authenticated, already-accepted WebSocket to the fixed relay.

    The only TCP destination is the configured numeric loopback relay.  No
    address, port, node ID, or other target is taken from a WebSocket message.
    """
    host = _loopback_host(relay_host)
    port = _port(relay_port, "central relay port")
    if type(max_frame_bytes) is not int or not 1 <= max_frame_bytes <= _MAX_FRAME_BYTES:
        raise FleetSSHError("SSH WebSocket frame limit is invalid")
    node_id = _node_value(grant, "node_id", required=True)
    if not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id):
        raise FleetSSHError("approved Fleet SSH grant has an invalid node ID")
    _port(_node_value(grant, "reverse_port", required=True), "approved reverse listener")
    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None
    try:
        reader, writer = await asyncio.open_connection(host, port)

        async def websocket_to_relay() -> None:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                payload = message.get("bytes")
                if not isinstance(payload, bytes):
                    await websocket.close(code=1003, reason="binary SSH stream required")
                    return
                if len(payload) > max_frame_bytes:
                    await websocket.close(code=1009, reason="SSH frame too large")
                    return
                writer.write(payload)
                await writer.drain()

        async def relay_to_websocket() -> None:
            while chunk := await reader.read(max_frame_bytes):
                await websocket.send_bytes(chunk)

        pumps = [asyncio.create_task(websocket_to_relay()), asyncio.create_task(relay_to_websocket())]
        done, pending = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        error = next((task.exception() for task in done if not task.cancelled() and task.exception()), None)
        if error is not None:
            raise error
        try:
            await websocket.close(code=1000)
        except (RuntimeError, WebSocketDisconnect):
            pass
    except WebSocketDisconnect:
        return
    except (OSError, asyncio.TimeoutError, RuntimeError):
        try:
            await websocket.close(code=1011, reason="SSH relay unavailable")
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass


def _read_token(path: Path) -> str:
    _private_regular(path)
    try:
        token = Path(path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise FleetSSHError("Fleet bearer file cannot be read") from exc
    if not token or any(char.isspace() or ord(char) < 33 for char in token):
        raise FleetSSHError("Fleet bearer file is malformed")
    return token
