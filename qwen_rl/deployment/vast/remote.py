"""Attach a prepared Qwen Worker or Learner to an already running Vast SSH instance.

This module never calls the Vast control-plane API.  A normal invocation only
builds a local plan; ``--execute`` is required before it opens SSH, uploads role
credentials, or starts the pinned role script.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from qwen_rl.runtime.manifest import process_starttime

from .config import VastDeploymentError

SCHEMA = "qwen-rl-vast-remote-attach/v2"
SSH_KEEPALIVE_INTERVAL = 15
SSH_KEEPALIVE_COUNT = 3
LOCAL_CONNECT_TIMEOUT = 2.0
REMOTE_COMMAND_TIMEOUT = 20.0
READINESS_TIMEOUT = 900.0
READINESS_POLL_SECONDS = 2.0
_HOSTNAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")
_USERNAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_ROLE_FILES = {
    "worker": ("deployment.json", "worker.json", "model-manifest.json", "start-worker.sh"),
    "learner": ("deployment.json", "runtime.json", "asset-manifest.json", "start-learner.sh"),
}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class VastRemoteError(RuntimeError):
    """An expected, redacted failure while attaching to a prepared instance."""


@dataclass(frozen=True)
class VastSSHEndpoint:
    host: str
    port: int
    user: str
    identity_file: Path
    known_hosts_file: Path


@dataclass(frozen=True)
class VastPreparedFile:
    name: str
    source: Path
    remote_path: str
    sha256: str
    mode: int


@dataclass(frozen=True)
class VastCredentialFile:
    env_name: str | None
    source: Path
    remote_path: str
    sha256: str


@dataclass(frozen=True)
class VastReverseRoute:
    name: str
    remote_port: int
    local_url: str
    local_port: int


@dataclass(frozen=True)
class VastAttachPlan:
    role: str
    deployment_dir: Path
    deployment_id: str
    endpoint: VastSSHEndpoint
    remote_root: str
    service_port: int
    local_forward_port: int
    forward_target_port: int
    health_url: str
    health_credential_env: str
    credential_files: tuple[VastCredentialFile, ...]
    required_credential_envs: tuple[str, ...]
    credential_map_remote_path: str | None
    prepared_files: tuple[VastPreparedFile, ...]
    package_sha256s: Mapping[str, str]
    credential_sha256s: Mapping[str, str]
    credential_map_sha256: str | None
    reverse_routes: tuple[VastReverseRoute, ...]
    local_dependency_ports: tuple[int, ...]
    service_identity: str
    start_command: tuple[str, ...]
    image_ref: str

    @property
    def remote_config_root(self) -> str:
        return f"{self.remote_root}/config"

    @property
    def remote_secrets_root(self) -> str:
        return f"{self.remote_root}/secrets"

    @property
    def remote_start_script(self) -> str:
        return next(item.remote_path for item in self.prepared_files if item.name.startswith("start-"))

    @property
    def remote_credential_map(self) -> str | None:
        return self.credential_map_remote_path

    @property
    def service_url(self) -> str:
        return self.health_url.removesuffix("/health")

    @property
    def health_credential(self) -> VastCredentialFile:
        if self.role == "worker":
            return self.credential_files[0]
        return next(item for item in self.credential_files
                    if item.env_name == self.health_credential_env)

    @property
    def remote_receipt(self) -> str:
        return self.start_command[self.start_command.index("--receipt") + 1]

    def model_dump(self) -> dict[str, Any]:
        """Return non-secret attach metadata suitable for dry-run output."""
        report = {
            "schema": SCHEMA,
            "status": "dry_run",
            "role": self.role,
            "deployment_id": self.deployment_id,
            "instance": {"host": self.endpoint.host, "ssh_port": self.endpoint.port,
                         "ssh_user": self.endpoint.user},
            "network_mode": "ssh_tunnel",
            "health_url": self.health_url,
            "forwards": _forward_metadata(self),
            "remote_root": self.remote_root,
            "required_credential_envs": list(self.required_credential_envs),
            "package_sha256s": dict(self.package_sha256s),
            "image_ref": self.image_ref,
            "remote_command": list(self.start_command),
            "execute_required": True,
            "secret_transferred": False,
            "remote_connection_attempted": False,
        }
        if self.role == "worker":
            report["worker_id"] = self.service_identity
            report["worker_url"] = self.service_url
        else:
            report["learner_url"] = self.service_url
        return report


@dataclass
class _OwnedChild:
    role: str
    process: subprocess.Popen[Any]
    starttime: str
    log_path: Path

    def is_alive(self) -> bool:
        return self.process.poll() is None and process_starttime(self.process.pid) == self.starttime


def _safe_owned_file(path: Path, label: str, *, mode: int | None = None,
                     allow_group_read: bool = False) -> os.stat_result:
    try:
        metadata = Path(path).lstat()
    except OSError:
        raise VastRemoteError(f"{label} is unavailable") from None
    permissions = stat.S_IMODE(metadata.st_mode)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1 or Path(path).is_symlink()):
        raise VastRemoteError(f"{label} must be an owned regular file")
    if mode is not None and permissions != mode:
        raise VastRemoteError(f"{label} must have mode {mode:04o}")
    if mode is None and permissions & 0o022:
        raise VastRemoteError(f"{label} must not be group/world writable")
    if not allow_group_read and permissions & 0o077:
        raise VastRemoteError(f"{label} must be private to the current user")
    return metadata


def _safe_directory(path: Path, label: str) -> None:
    try:
        metadata = Path(path).lstat()
    except OSError:
        raise VastRemoteError(f"{label} is unavailable") from None
    if (not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077):
        raise VastRemoteError(f"{label} must be a private owned directory")


def _ensure_private_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError:
        raise VastRemoteError(f"could not create {label}") from None
    _safe_directory(path, label)
    if stat.S_IMODE(path.lstat().st_mode) != 0o700:
        raise VastRemoteError(f"{label} must have mode 0700")


def _port(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise VastRemoteError(f"{label} must be a valid TCP port")
    return value


def _host(value: Any) -> str:
    if not isinstance(value, str) or not value or any(ord(char) < 33 or ord(char) == 127 for char in value):
        raise VastRemoteError("SSH host is invalid")
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        if not _HOSTNAME.fullmatch(value) or ".." in value:
            raise VastRemoteError("SSH host must be an explicit host name or IP address") from None
        return value


def _loopback_http_url(value: Any, label: str) -> tuple[str, int]:
    if not isinstance(value, str):
        raise VastRemoteError(f"{label} must be an HTTP loopback URL")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError:
        raise VastRemoteError(f"{label} must contain a valid TCP port") from None
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or port is None
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise VastRemoteError(f"{label} must use http://127.0.0.1:<port> without credentials or path")
    return value.rstrip("/"), port


def _sha256(path: Path, label: str) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise VastRemoteError(f"prepared {label} cannot be read") from None
    return digest


def _forward_metadata(plan: VastAttachPlan) -> dict[str, Any]:
    return {
        "local_service": (
            f"127.0.0.1:{plan.local_forward_port}:127.0.0.1:{plan.forward_target_port}"
        ),
        "remote_origins": [
            {
                "name": route.name,
                "forward": f"127.0.0.1:{route.remote_port}:127.0.0.1:{route.local_port}",
            }
            for route in plan.reverse_routes
        ],
    }


def _credential_map_bytes(credentials: Sequence[VastCredentialFile]) -> bytes:
    mapping = {item.env_name: item.remote_path for item in credentials if item.env_name is not None}
    if len(mapping) != len(credentials):
        raise VastRemoteError("credential map can contain only named Learner environment credentials")
    return (json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _credential_source(value: str | Path, env_name: str | None, *, root: Path) -> Path:
    path = Path(value).expanduser().absolute()
    _safe_owned_file(path, f"credential file{f' for {env_name}' if env_name else ''}", mode=0o600)
    try:
        path.relative_to(root)
    except ValueError:
        return path
    raise VastRemoteError("credential files must stay outside the prepared deployment directory")


def _verify_credential_source(credential: VastCredentialFile) -> None:
    label = f"{credential.env_name} credential file" if credential.env_name else "Worker token file"
    _safe_owned_file(credential.source, label, mode=0o600)
    if _sha256(credential.source, label) != credential.sha256:
        raise VastRemoteError(f"{label} changed after attach planning")


def build_attach_plan(
    deployment_dir: str | Path,
    *,
    expected_role: str,
    host: str,
    ssh_port: int,
    ssh_user: str,
    identity_file: str | Path,
    known_hosts_file: str | Path,
    token_file: str | Path | None = None,
    credentials: Sequence[tuple[str, str | Path]] = (),
) -> VastAttachPlan:
    """Validate local metadata and return a no-I/O attach plan.

    The SSH endpoint, port, key, and known-hosts file must be supplied by the
    caller.  This function never contacts Vast or reads the token contents.
    """
    root = Path(deployment_dir).expanduser().absolute()
    if expected_role not in _ROLE_FILES:
        raise VastRemoteError("attach requires an explicit supported deployment role")
    try:
        from .prepare import check_prepared_deployment

        check_prepared_deployment(root, expected_role=expected_role)
    except VastDeploymentError as exc:
        detail = str(exc).lower()
        if "worker.json" in detail or "worker config" in detail:
            label = "prepared Worker configuration"
        elif "start-worker.sh" in detail:
            label = "prepared Worker start script"
        elif "manifest" in detail:
            label = "prepared model manifest"
        else:
            label = "prepared deployment metadata"
        raise VastRemoteError(f"{label} failed canonical package validation") from None
    try:
        profile = json.loads((root / "deployment.json").read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise VastRemoteError("canonical prepared deployment files could not be reread") from None
    if profile.get("role") != expected_role:
        raise VastRemoteError("prepared deployment role differs from the explicitly selected attach role")
    try:
        deployment_id = profile["deployment_id"]
        remote_root = profile["remote_root"]
        local_forward_port = _port(profile["local_forward_port"], "local forward port")
        image_ref = profile["image_ref"]
        if expected_role == "worker":
            worker_config = json.loads((root / "worker.json").read_bytes())
            service_port = _port(profile["worker_port"], "Worker port")
            forward_target_port = service_port
            worker_id = worker_config["worker_id"]
            health_url = f"http://127.0.0.1:{local_forward_port}/health"
            learner_url, learner_port = _loopback_http_url(
                profile["learner_local_url"], "prepared Learner origin URL")
            reverse_routes = (VastReverseRoute(
                name="learner_policy", remote_port=_port(profile["remote_policy_port"], "remote policy port"),
                local_url=learner_url, local_port=learner_port,
            ),)
            local_dependency_ports = (learner_port,)
            required_envs: tuple[str, ...] = ()
            credential_map_remote = None
            identity = worker_id
            filenames = _ROLE_FILES["worker"]
            prepared_sha = {
                "deployment.json": _sha256(root / "deployment.json", "deployment.json"),
                "worker.json": profile["worker_json_sha256"],
                "model-manifest.json": profile["model_manifest_sha256"],
                "start-worker.sh": profile["start_worker_script_sha256"],
            }
            remote_config = f"{remote_root}/config"
            start_command = (
                "/bin/bash", f"{remote_config}/start-worker.sh",
                "--deployment", f"{remote_config}/deployment.json",
                "--worker-config", f"{remote_config}/worker.json",
                "--model-manifest", f"{remote_config}/model-manifest.json",
                "--token-file", f"{remote_root}/secrets/worker_auth_token",
                "--model-root", profile["model_root_container"],
                "--receipt", f"{remote_root}/state/vast-download-receipt.json",
            )
        else:
            service_port = _port(profile["learner_port"], "Learner port")
            forward_target_port = service_port
            health_url = f"http://127.0.0.1:{local_forward_port}/health"
            routes: list[VastReverseRoute] = []
            for row in profile["worker_routes"]:
                route_url, route_port = _loopback_http_url(row["local_url"], "prepared Worker origin URL")
                route = VastReverseRoute(
                    name=row["worker_id"], remote_port=_port(row["remote_port"], "remote Worker route port"),
                    local_url=route_url, local_port=route_port,
                )
                duplicate_port = next((item for item in routes if item.remote_port == route.remote_port), None)
                if duplicate_port is not None:
                    if duplicate_port.local_port != route.local_port:
                        raise VastRemoteError("one remote Worker route port cannot target multiple local origins")
                    continue
                routes.append(route)
            reverse_routes = tuple(routes)
            local_dependency_ports = tuple(dict.fromkeys(route.local_port for route in routes))
            required_envs = tuple(profile["credential_envs"])
            if (not required_envs or len(set(required_envs)) != len(required_envs)
                    or any(not isinstance(name, str) or not _ENV_NAME.fullmatch(name) for name in required_envs)
                    or "ARC_ENV_TOKEN" not in required_envs):
                raise VastRemoteError("prepared Learner credential environment contract is invalid")
            credential_map_remote = f"{remote_root}/secrets/credential-map.json"
            identity = deployment_id
            filenames = _ROLE_FILES["learner"]
            prepared_sha = {
                "deployment.json": _sha256(root / "deployment.json", "deployment.json"),
                "runtime.json": profile["runtime_config_sha256"],
                "asset-manifest.json": profile["asset_manifest_sha256"],
                "start-learner.sh": profile["start_learner_script_sha256"],
            }
            remote_config = f"{remote_root}/config"
            start_command = (
                "/bin/bash", f"{remote_config}/start-learner.sh",
                "--deployment", f"{remote_config}/deployment.json",
                "--runtime-config", f"{remote_config}/runtime.json",
                "--asset-manifest", f"{remote_config}/asset-manifest.json",
                "--asset-root", profile["asset_root_container"],
                "--state-root", profile["state_root_container"],
                "--credential-map", credential_map_remote,
                "--receipt", f"{profile['state_root_container'].rstrip('/')}/vast-learner-entrypoint.json",
            )
    except (KeyError, IndexError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        raise VastRemoteError("prepared deployment role metadata is incomplete or invalid") from None

    package_sha256s: dict[str, str] = {}
    credential_sha256s: dict[str, str] = {}
    prepared_files: list[VastPreparedFile] = []
    for name in filenames:
        expected_sha = prepared_sha[name]
        observed_sha = _sha256(root / name, name)
        if expected_sha != observed_sha:
            raise VastRemoteError(f"prepared {name} differs from deployment.json")
        package_sha256s[name] = expected_sha
        mode = 0o600
        prepared_files.append(VastPreparedFile(
            name=name, source=root / name, remote_path=f"{remote_root}/config/{name}",
            sha256=expected_sha, mode=mode,
        ))

    if not isinstance(ssh_user, str) or not _USERNAME.fullmatch(ssh_user):
        raise VastRemoteError("SSH user is invalid")
    endpoint = VastSSHEndpoint(
        host=_host(host),
        port=_port(ssh_port, "SSH port"),
        user=ssh_user,
        identity_file=Path(identity_file).expanduser().absolute(),
        known_hosts_file=Path(known_hosts_file).expanduser().absolute(),
    )
    _safe_owned_file(endpoint.identity_file, "SSH identity file", mode=0o600)
    known_hosts_metadata = _safe_owned_file(endpoint.known_hosts_file, "SSH known_hosts file",
                                            allow_group_read=True)
    if known_hosts_metadata.st_size <= 0:
        raise VastRemoteError("SSH known_hosts file must contain pinned host keys")
    credential_files: list[VastCredentialFile] = []
    if expected_role == "worker":
        if token_file is None or credentials:
            raise VastRemoteError("Worker attach requires only --token-file")
        token_path = _credential_source(token_file, None, root=root)
        credential_files.append(VastCredentialFile(
            env_name=None, source=token_path, remote_path=f"{remote_root}/secrets/worker_auth_token",
            sha256=_sha256(token_path, "Worker token file"),
        ))
        if tuple(required_envs):
            raise VastRemoteError("Worker deployment cannot declare Learner credential environment names")
    else:
        if token_file is not None:
            raise VastRemoteError("Learner attach requires --credential ENV=/abs/file, not --token-file")
        parsed_credentials: dict[str, Path] = {}
        for env_name, source in credentials:
            if not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name) or env_name in parsed_credentials:
                raise VastRemoteError("Learner credential environment names must be unique valid environment variables")
            parsed_credentials[env_name] = _credential_source(source, env_name, root=root)
        if set(parsed_credentials) != set(required_envs):
            raise VastRemoteError("Learner credential files must exactly match the prepared credential_envs")
        if len(set(parsed_credentials.values())) != len(parsed_credentials):
            raise VastRemoteError("each Learner environment credential must use a separate private file")
        for env_name in required_envs:
            source = parsed_credentials[env_name]
            credential_files.append(VastCredentialFile(
                env_name=env_name, source=source,
                remote_path=f"{remote_root}/secrets/{env_name}",
                sha256=_sha256(source, f"{env_name} credential file"),
            ))

    for credential in credential_files:
        credential_key = credential.env_name or "worker_auth_token"
        credential_sha256s[credential_key] = credential.sha256
    for name in credential_files:
        if name.env_name is not None and name.env_name == "ARC_ENV_TOKEN":
            health_credential_env = name.env_name
    if expected_role == "worker":
        health_credential_env = "worker-token"
    map_digest = None
    if credential_map_remote is not None:
        credential_map_data = _credential_map_bytes(credential_files)
        map_digest = hashlib.sha256(credential_map_data).hexdigest()
    if map_digest is not None:
        credential_sha256s["credential-map.json"] = map_digest

    return VastAttachPlan(
        role=expected_role,
        deployment_dir=root,
        deployment_id=deployment_id,
        endpoint=endpoint,
        remote_root=remote_root,
        service_port=service_port,
        local_forward_port=local_forward_port,
        forward_target_port=forward_target_port,
        health_url=health_url,
        health_credential_env=health_credential_env,
        credential_files=tuple(credential_files),
        required_credential_envs=tuple(required_envs),
        credential_map_remote_path=credential_map_remote,
        prepared_files=tuple(prepared_files),
        package_sha256s=package_sha256s,
        credential_sha256s=credential_sha256s,
        credential_map_sha256=map_digest,
        reverse_routes=reverse_routes,
        local_dependency_ports=local_dependency_ports,
        service_identity=identity,
        start_command=start_command,
        image_ref=image_ref,
    )


def _ssh_security_options(plan: VastAttachPlan, *, clear_forwardings: bool = True) -> list[str]:
    options = [
        "-F", "/dev/null",
        "-p", str(plan.endpoint.port),
        "-i", str(plan.endpoint.identity_file),
        "-l", plan.endpoint.user,
        "-oBatchMode=yes",
        "-oIdentitiesOnly=yes",
        "-oIdentityAgent=none",
        "-oStrictHostKeyChecking=yes",
        f"-oUserKnownHostsFile={plan.endpoint.known_hosts_file}",
        "-oGlobalKnownHostsFile=/dev/null",
        "-oUpdateHostKeys=no",
        "-oVerifyHostKeyDNS=no",
        "-oCheckHostIP=no",
        "-oForwardAgent=no",
        "-oForwardX11=no",
        "-oProxyCommand=none",
        "-oProxyJump=none",
        f"-oServerAliveInterval={SSH_KEEPALIVE_INTERVAL}",
        f"-oServerAliveCountMax={SSH_KEEPALIVE_COUNT}",
    ]
    if clear_forwardings:
        options.append("-oClearAllForwardings=yes")
    return options


def build_tunnel_argv(plan: VastAttachPlan, *, ssh_bin: str = "ssh") -> list[str]:
    argv = [
        ssh_bin, *_ssh_security_options(plan, clear_forwardings=False),
        "-N", "-T",
        "-oExitOnForwardFailure=yes",
        "-L", f"127.0.0.1:{plan.local_forward_port}:127.0.0.1:{plan.forward_target_port}",
    ]
    for route in plan.reverse_routes:
        argv.extend((
            "-R", f"127.0.0.1:{route.remote_port}:127.0.0.1:{route.local_port}",
        ))
    argv.append(
        plan.endpoint.host,
    )
    return argv


_REMOTE_STATE_PROBE = r'''import hashlib,json,os,socket,stat,sys
probe_marker="qwen_vast_remote_state_probe_v2"
root,port,deployment_id,role,expected_config_json,expected_secret_json,map_sha,markers_json=sys.argv[1:]
expected_config=json.loads(expected_config_json)
expected_secrets=json.loads(expected_secret_json)
markers=json.loads(markers_json)
uid=os.geteuid()
def lstat(path):
    try: return os.lstat(path)
    except FileNotFoundError: return None
def safe_dir(path):
    info=lstat(path)
    return "missing" if info is None else "ok" if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_uid==uid and stat.S_IMODE(info.st_mode)==0o700 else "unsafe"
def file_digest(path,mode=None):
    info=lstat(path)
    if info is None: return "missing"
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid!=uid or info.st_nlink!=1: return "unsafe"
    if mode is not None and stat.S_IMODE(info.st_mode)!=mode: return "unsafe"
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def names(path):
    try: return set(os.listdir(path))
    except FileNotFoundError: return set()
    except OSError: return {"<unsafe>"}
info=lstat(root)
root_safe=bool(info and stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_uid==uid and stat.S_IMODE(info.st_mode)==0o700)
dirs={name:safe_dir(os.path.join(root,name)) for name in ("config","secrets","state")} if root_safe else {}
observed={name:file_digest(os.path.join(root,"config",name),0o600) for name in expected_config} if dirs and dirs.get("config")=="ok" else {}
temp_path=os.path.join(root,"config",".deployment."+deployment_id+".tmp") if dirs and dirs.get("config")=="ok" else ""
pending_digest=file_digest(temp_path,0o600) if temp_path else "missing"
secret_observed={name:file_digest(os.path.join(root,"secrets",name),0o600) for name in expected_secrets} if dirs and dirs.get("secrets")=="ok" else {}
map_state="missing"
if map_sha!="": map_state=file_digest(os.path.join(root,"secrets","credential-map.json"),0o600) if dirs.get("secrets")=="ok" else "unsafe" if dirs.get("secrets")=="unsafe" else "missing"
root_names=names(root) if root_safe else set()
config_names=names(os.path.join(root,"config")) if dirs.get("config")=="ok" else set()
secret_names=names(os.path.join(root,"secrets")) if dirs.get("secrets")=="ok" else set()
state_names=names(os.path.join(root,"state")) if dirs.get("state")=="ok" else set()
allowed_state={"vast-download-receipt.json"} if role=="worker" else set()
allowed_secrets=set(expected_secrets) | ({"credential-map.json"} if map_sha!="" else set())
layout_ok=(root_names <= {"config","secrets","state"}
    and config_names <= set(expected_config) | {".deployment."+deployment_id+".tmp"}
    and secret_names <= allowed_secrets and state_names <= allowed_state)
if info is None:
    root_state="absent"
elif not root_safe:
    root_state="foreign"
elif any(value=="unsafe" for value in dirs.values()) or not layout_ok:
    root_state="foreign"
elif observed.get("deployment.json")==expected_config.get("deployment.json"):
    try:
        with open(os.path.join(root,"config","deployment.json"),encoding="utf-8") as stream: deployment=json.load(stream)
    except Exception: deployment={}
    config_ok=all(value in ("missing",expected_config[name]) for name,value in observed.items())
    secrets_ok=all(value in ("missing",expected_secrets[name]) for name,value in secret_observed.items())
    map_ok=map_state in ("missing",map_sha) if map_sha!="" else map_state=="missing"
    if deployment.get("deployment_id")!=deployment_id or deployment.get("role")!=role: root_state="foreign"
    elif not config_ok or not secrets_ok or not map_ok: root_state="foreign"
    else: root_state="owned"
elif observed.get("deployment.json")=="missing" and pending_digest==expected_config.get("deployment.json") and all(value=="missing" for name,value in observed.items() if name!="deployment.json") and not secret_names:
    root_state="pending_owner"
elif all(value=="missing" for value in observed.values()) and pending_digest=="missing" and not secret_names and not state_names:
    root_state="empty"
else:
    root_state="foreign"
credentials_match=(all(secret_observed.get(name)==digest for name,digest in expected_secrets.items())
    and (map_sha=="" or map_state==map_sha))
credential_state="matching" if credentials_match else "partial" if root_state in ("owned","empty","pending_owner","absent") else "foreign"
parent=os.path.dirname(root)
parent_info=lstat(parent)
parent_writable=bool(parent_info and stat.S_ISDIR(parent_info.st_mode) and not stat.S_ISLNK(parent_info.st_mode) and os.access(parent,os.W_OK))
active_start=False
try:
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name)==os.getpid(): continue
        try: args=open("/proc/"+name+"/cmdline","rb").read(16384).decode(errors="replace")
        except OSError: continue
        if probe_marker in args: continue
        if root in args and any(marker in args for marker in markers): active_start=True; break
except OSError: pass
sock=socket.socket(); sock.settimeout(.4)
try: service_port_open=sock.connect_ex(("127.0.0.1",int(port)))==0
except OSError: service_port_open=False
finally: sock.close()
print(json.dumps({"root_state":root_state,"directories":dirs,"files":observed,"credential_state":credential_state,"parent_writable":parent_writable,"service_port_open":service_port_open,"active_start":active_start},separators=(",",":")))
'''


def _build_remote_preflight_argv(plan: VastAttachPlan, *, ssh_bin: str) -> list[str]:
    expected_config = {item.name: item.sha256 for item in plan.prepared_files}
    expected_secrets = {
        item.remote_path.rsplit("/", 1)[-1]: item.sha256
        for item in plan.credential_files
    }
    markers = [plan.remote_start_script.rsplit("/", 1)[-1]]
    markers.extend(("download.py", "entrypoint.sh") if plan.role == "worker"
                   else ("learner-entrypoint.sh", "learner_entrypoint", "run_container_supervisor"))
    arguments = [
        "python3", "-c", _REMOTE_STATE_PROBE, plan.remote_root,
        str(plan.service_port), plan.deployment_id, plan.role,
        json.dumps(expected_config, sort_keys=True, separators=(",", ":")),
        json.dumps(expected_secrets, sort_keys=True, separators=(",", ":")),
        plan.credential_map_sha256 or "",
        json.dumps(markers, separators=(",", ":")),
    ]
    command = " ".join(shlex.quote(argument) for argument in arguments)
    return [ssh_bin, *_ssh_security_options(plan), "-T", plan.endpoint.host, command]


def _sftp_quote(value: str) -> str:
    if any(char in value for char in "\r\n\x00"):
        raise VastRemoteError("SFTP path contains invalid control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_sftp_batch(
    plan: VastAttachPlan,
    *,
    root_state: str = "absent",
    credential_map_source: Path | None = None,
) -> str:
    """Build an SFTP batch containing paths only; credential bytes are never included."""
    root = plan.remote_root
    if root_state not in {"absent", "empty", "pending_owner", "owned"}:
        raise VastRemoteError("remote deployment ownership is not proven")
    for item in plan.prepared_files:
        _safe_owned_file(item.source, f"prepared {item.name}", mode=item.mode)
        if _sha256(item.source, item.name) != item.sha256:
            raise VastRemoteError(f"prepared {item.name} changed after attach planning")
    if plan.credential_map_remote_path is not None:
        if credential_map_source is None:
            raise VastRemoteError("Learner credential map staging file is required")
        _safe_owned_file(credential_map_source, "Learner credential map staging file", mode=0o600)
        if _sha256(credential_map_source, "Learner credential map") != plan.credential_map_sha256:
            raise VastRemoteError("Learner credential map staging file differs from the prepared credential contract")
    dirs = (root, f"{root}/config", f"{root}/secrets", f"{root}/state")
    lines = [f"{'mkdir' if root_state == 'absent' else '-mkdir'} {_sftp_quote(path)}" for path in dirs]
    if root_state in {"absent", "empty", "pending_owner"}:
        deployment_file = next(item for item in plan.prepared_files if item.name == "deployment.json")
        local_deployment = _sftp_quote(str(deployment_file.source))
        remote_temp = f"{plan.remote_config_root}/.deployment.{plan.deployment_id}.tmp"
        lines.append(f"put -p {local_deployment} {_sftp_quote(remote_temp)}")
        lines.append(f"rename {_sftp_quote(remote_temp)} {_sftp_quote(f'{plan.remote_config_root}/deployment.json')}")
    for item in plan.prepared_files:
        if item.name == "deployment.json":
            continue
        lines.append(f"put -p {_sftp_quote(str(item.source))} {_sftp_quote(item.remote_path)}")
    for credential in plan.credential_files:
        _verify_credential_source(credential)
        lines.append(f"put -p {_sftp_quote(str(credential.source))} {_sftp_quote(credential.remote_path)}")
    if credential_map_source is not None and plan.credential_map_remote_path is not None:
        lines.append(
            f"put -p {_sftp_quote(str(credential_map_source))} "
            f"{_sftp_quote(plan.credential_map_remote_path)}"
        )
    lines.extend(f"chmod 700 {_sftp_quote(path)}" for path in dirs)
    lines.extend(
        f"chmod {item.mode:o} {_sftp_quote(item.remote_path)}"
        for item in plan.prepared_files
    )
    lines.extend(
        f"chmod 600 {_sftp_quote(item.remote_path)}"
        for item in plan.credential_files
    )
    if plan.credential_map_remote_path is not None:
        lines.append(f"chmod 600 {_sftp_quote(plan.credential_map_remote_path)}")
    return "\n".join(lines) + "\n"


def _build_sftp_argv(plan: VastAttachPlan, *, sftp_bin: str) -> list[str]:
    return [
        sftp_bin, "-F", "/dev/null", "-P", str(plan.endpoint.port),
        "-i", str(plan.endpoint.identity_file), "-o", f"User={plan.endpoint.user}",
        "-oBatchMode=yes", "-oIdentitiesOnly=yes", "-oIdentityAgent=none",
        "-oStrictHostKeyChecking=yes",
        f"-oUserKnownHostsFile={plan.endpoint.known_hosts_file}",
        "-oGlobalKnownHostsFile=/dev/null", "-oUpdateHostKeys=no",
        "-oVerifyHostKeyDNS=no", "-oCheckHostIP=no",
        "-oClearAllForwardings=yes", "-oForwardAgent=no", "-oForwardX11=no",
        "-oProxyCommand=none", "-oProxyJump=none",
        "-b", "-", plan.endpoint.host,
    ]


def _build_start_argv(plan: VastAttachPlan, *, ssh_bin: str) -> list[str]:
    remote_command = " ".join(shlex.quote(argument) for argument in plan.start_command)
    return [ssh_bin, *_ssh_security_options(plan), "-T", plan.endpoint.host, remote_command]


def _run_ssh_operation(
    argv: Sequence[str], *, label: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    timeout: float = REMOTE_COMMAND_TIMEOUT,
    batch: str | None = None,
) -> None:
    kwargs: dict[str, Any] = {
        "capture_output": True, "text": True, "timeout": timeout, "check": False,
    }
    if batch is not None:
        kwargs["input"] = batch
    try:
        result = runner(list(argv), **kwargs)
    except (OSError, subprocess.SubprocessError):
        raise VastRemoteError(f"{label} could not be completed") from None
    if result.returncode != 0:
        # OpenSSH/SFTP diagnostics are intentionally discarded: logs must never
        # accidentally include an auth header or token content from a helper.
        raise VastRemoteError(f"{label} failed with exit code {result.returncode}")


def _read_secret(path: Path, *, expected_sha256: str | None = None) -> str:
    """Read an owned 0600 credential for the authenticated health GET only."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise VastRemoteError("health credential file could not be opened securely") from None
    try:
        metadata = os.fstat(fd)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1):
            raise VastRemoteError("health credential file lost its private-file guarantees")
        chunks: list[bytes] = []
        remaining = 16385
        while remaining:
            block = os.read(fd, min(4096, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        if not raw or len(raw) > 16384:
            raise VastRemoteError("health credential file is empty or too large")
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise VastRemoteError("health credential file changed after attach planning")
        token = raw.decode("utf-8").strip()
        if not token or any(ord(char) < 32 or ord(char) == 127 for char in token):
            raise VastRemoteError("health credential file is invalid")
        return token
    except UnicodeError:
        raise VastRemoteError("health credential file is not valid UTF-8") from None
    finally:
        os.close(fd)


def _probe_health_result(
    plan: VastAttachPlan, token: str | None, *, timeout: float = 3.0,
    getter: Callable[..., Any] = urlopen,
) -> tuple[int | None, dict[str, Any] | None]:
    headers = {"Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    request = Request(plan.health_url, headers=headers)
    try:
        with getter(request, timeout=timeout) as response:
            body = response.read(1_000_001)
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        return int(exc.code), None
    except (OSError, URLError, TimeoutError):
        return None, None
    if len(body) > 1_000_000:
        return status, None
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, UnicodeError, TypeError):
        return status, None
    if not isinstance(value, dict) or value.get("role") != plan.role or type(value.get("ready")) is not bool:
        return status, None
    health: dict[str, Any] = {"role": plan.role, "ready": value["ready"]}
    if plan.role == "worker":
        if type(value.get("policy_ready")) is not bool:
            return status, None
        health.update({
            "policy_ready": value["policy_ready"],
            "worker_id": value.get("worker_id") if isinstance(value.get("worker_id"), str) else None,
            "backend": value.get("backend") if isinstance(value.get("backend"), str) else None,
            "policy_version": value.get("policy_version") if type(value.get("policy_version")) is int else None,
            "adapter_hash": value.get("adapter_hash") if isinstance(value.get("adapter_hash"), str) else None,
        })
    else:
        health.update({
            key: value[key]
            for key in ("policy_version", "serve_policy_version")
            if type(value.get(key)) is int
        })
    return status, health


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _safe_directory(parent, "local attach state directory")
    if path.is_symlink():
        raise VastRemoteError("local attach receipt path cannot be a symbolic link")
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _local_dependency_preflight(
    plan: VastAttachPlan,
    *,
    connect: Callable[..., Any] = socket.create_connection,
) -> None:
    for port in plan.local_dependency_ports:
        try:
            with connect(("127.0.0.1", port), timeout=LOCAL_CONNECT_TIMEOUT):
                continue
        except OSError:
            label = "Learner" if plan.role == "worker" else "Worker"
            raise VastRemoteError(
                f"local {label} loopback dependency on port {port} is not reachable; startup was not attempted"
            ) from None


def _capture_starttime(process: subprocess.Popen[Any], *, sleep: Callable[[float], None]) -> str:
    for _ in range(10):
        starttime = process_starttime(process.pid)
        if starttime is not None:
            return starttime
        if process.poll() is not None:
            break
        sleep(0.02)
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            pass
    raise VastRemoteError("could not record the owned SSH process identity")


def _terminate_owned(child: _OwnedChild, *, timeout: float = 3.0) -> None:
    process = child.process
    if process.poll() is not None:
        return
    if process_starttime(process.pid) != child.starttime:
        return
    try:
        os.kill(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    if process_starttime(process.pid) == child.starttime:
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return


class VastAttachSession:
    """Foreground owner for role-aware SSH forwards and remote startup."""

    def __init__(
        self,
        plan: VastAttachPlan,
        *,
        popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        getter: Callable[..., Any] = urlopen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        connect: Callable[..., Any] = socket.create_connection,
        ssh_bin: str = "ssh",
        sftp_bin: str = "sftp",
        readiness_timeout: float = READINESS_TIMEOUT,
    ):
        if plan.role not in _ROLE_FILES:
            raise VastRemoteError("attach session requires an explicit supported role")
        self.plan = plan
        self.popen = popen
        self.runner = runner
        self.getter = getter
        self.sleep = sleep
        self.monotonic = monotonic
        self.connect = connect
        self.ssh_bin = ssh_bin
        self.sftp_bin = sftp_bin
        if (isinstance(readiness_timeout, bool) or not isinstance(readiness_timeout, (int, float))
                or readiness_timeout <= 0):
            raise VastRemoteError("readiness timeout must be positive")
        self.readiness_timeout = float(readiness_timeout)
        self.children: list[_OwnedChild] = []
        self.health: dict[str, Any] | None = None
        self.closed = False
        self.service_reused = False
        self.receipt_path = plan.deployment_dir / "state" / f"attach-{uuid.uuid4().hex}.json"
        self.session_id = self.receipt_path.stem.removeprefix("attach-")
        self.credential_map_source: Path | None = None
        self._started_at: str | None = None

    def _session_dir(self) -> Path:
        output_root = self.plan.deployment_dir.parent
        _safe_directory(output_root, "prepared output root")
        sessions_dir = output_root / "sessions"
        deployment_sessions_dir = sessions_dir / self.plan.deployment_id
        session_dir = deployment_sessions_dir / self.session_id
        _ensure_private_directory(sessions_dir, "SSH logs directory")
        _ensure_private_directory(deployment_sessions_dir, "deployment SSH logs directory")
        _ensure_private_directory(session_dir, "SSH log session directory")
        return session_dir

    def _stage_credential_map(self) -> Path | None:
        if self.plan.credential_map_remote_path is None:
            return None
        if self.credential_map_source is not None:
            return self.credential_map_source
        path = self._session_dir() / "credential-map.json"
        descriptor = os.open(
            path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600,
        )
        try:
            data = _credential_map_bytes(self.plan.credential_files)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            path.unlink(missing_ok=True)
            raise VastRemoteError("could not stage the private Learner credential map") from None
        if _sha256(path, "Learner credential map") != self.plan.credential_map_sha256:
            path.unlink(missing_ok=True)
            raise VastRemoteError("staged Learner credential map differs from the prepared contract")
        self.credential_map_source = path
        return path

    def _spawn(self, role: str, argv: Sequence[str]) -> _OwnedChild:
        log_path = self._session_dir() / f"{role}.log"
        try:
            fd = os.open(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except OSError:
            raise VastRemoteError("could not create a private SSH process log") from None
        try:
            with os.fdopen(fd, "wb") as log_stream:
                os.fchmod(log_stream.fileno(), 0o600)
                metadata = os.fstat(log_stream.fileno())
                if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                        or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1):
                    raise VastRemoteError("SSH process log lost its private-file guarantees")
                process = self.popen(
                    list(argv), stdin=subprocess.DEVNULL, stdout=log_stream,
                    stderr=log_stream, close_fds=True, start_new_session=True,
                )
        except VastRemoteError:
            log_path.unlink(missing_ok=True)
            raise
        except OSError:
            log_path.unlink(missing_ok=True)
            raise VastRemoteError(f"could not start the owned {role} SSH process") from None
        child = _OwnedChild(role=role, process=process,
                            starttime=_capture_starttime(process, sleep=self.sleep),
                            log_path=log_path)
        self.children.append(child)
        return child

    def _run_ssh_check(self) -> dict[str, Any]:
        argv = _build_remote_preflight_argv(self.plan, ssh_bin=self.ssh_bin)
        try:
            result = self.runner(list(argv), capture_output=True, text=True,
                                 timeout=REMOTE_COMMAND_TIMEOUT, check=False)
        except (OSError, subprocess.SubprocessError):
            raise VastRemoteError("read-only remote attach identity check failed") from None
        if result.returncode != 0:
            raise VastRemoteError("read-only remote attach identity check failed")
        try:
            state = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            raise VastRemoteError("remote attach identity check returned invalid status") from None
        if (not isinstance(state, dict)
                or state.get("root_state") not in {"absent", "empty", "pending_owner", "owned", "foreign"}
                or state.get("credential_state") not in {"matching", "partial", "foreign"}
                or type(state.get("service_port_open")) is not bool
                or type(state.get("parent_writable")) is not bool
                or type(state.get("active_start")) is not bool):
            raise VastRemoteError("remote attach identity check returned incomplete status")
        return state

    def _upload_prepared_files_and_credentials(self, *, root_state: str) -> None:
        credential_map = self._stage_credential_map()
        batch = build_sftp_batch(
            self.plan, root_state=root_state, credential_map_source=credential_map,
        )
        argv = _build_sftp_argv(self.plan, sftp_bin=self.sftp_bin)
        _run_ssh_operation(argv, label="prepared deployment SFTP upload", runner=self.runner,
                           timeout=120.0, batch=batch)

    def _wait_for_existing_service(self, credential: str, *, tunnel: _OwnedChild) -> dict[str, Any]:
        status, unauthenticated = _probe_health_result(self.plan, None, getter=self.getter)
        if status != 401 and not (status == 200 and unauthenticated is not None):
            raise VastRemoteError(
                "remote service port is occupied but does not report the prepared role; refusing duplicate start"
            )
        if (self.plan.role == "worker" and unauthenticated is not None
                and unauthenticated.get("worker_id") != self.plan.service_identity):
            raise VastRemoteError("existing Worker identity differs from the prepared deployment")
        deadline = self.monotonic() + self.readiness_timeout
        while True:
            if not tunnel.is_alive():
                raise VastRemoteError(f"SSH tunnel exited while reattaching to the existing {self.plan.role}")
            status, health = _probe_health_result(self.plan, credential, getter=self.getter)
            if status != 200 or health is None:
                raise VastRemoteError("existing service did not authenticate with the prepared credential; refusing overwrite")
            if (self.plan.role == "worker"
                    and health.get("worker_id") != self.plan.service_identity):
                raise VastRemoteError("existing Worker identity differs from the prepared deployment")
            self.health = health
            if health["ready"] is True:
                return health
            if self.monotonic() >= deadline:
                raise VastRemoteError(f"existing {self.plan.role} is present but did not become ready before timeout")
            self.sleep(min(READINESS_POLL_SECONDS, max(0.0, deadline - self.monotonic())))

    def _start_remote_command(self) -> _OwnedChild:
        return self._spawn("remote_role_exec", _build_start_argv(self.plan, ssh_bin=self.ssh_bin))

    def _wait_forward_ready(self, tunnel: _OwnedChild) -> None:
        deadline = self.monotonic() + 15.0
        while True:
            if not tunnel.is_alive():
                raise VastRemoteError("SSH tunnel exited before local service forwarding became ready")
            try:
                with self.connect(("127.0.0.1", self.plan.local_forward_port), timeout=1.0):
                    return
            except OSError:
                if self.monotonic() >= deadline:
                    raise VastRemoteError("local SSH service forward did not become ready") from None
                self.sleep(0.1)

    def _report(self, *, status: str) -> dict[str, Any]:
        report: dict[str, Any] = {
            "status": status,
            "role": self.plan.role,
            "service_reused": self.service_reused,
            "receipt": str(self.receipt_path),
            "health_url": self.plan.health_url,
            "health": dict(self.health or {}),
            "secret_transferred": True,
        }
        if self.plan.role == "worker":
            report["worker_url"] = self.plan.service_url
            report["policy_ready"] = bool((self.health or {}).get("policy_ready"))
            report["policy_acknowledged"] = False
        else:
            report["learner_url"] = self.plan.service_url
        return report

    def _write_receipt(self, *, status: str, error_code: str | None = None) -> None:
        receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "role": self.plan.role,
            "status": status,
            "deployment_id": self.plan.deployment_id,
            "instance": {"host": self.plan.endpoint.host,
                         "ssh_port": self.plan.endpoint.port,
                         "ssh_user": self.plan.endpoint.user},
            "network_mode": "ssh_tunnel",
            "service_identity": self.plan.service_identity,
            "service_url": self.plan.service_url,
            "health_url": self.plan.health_url,
            "forwards": _forward_metadata(self.plan),
            "package_sha256s": dict(self.plan.package_sha256s),
            "credential_envs": list(self.plan.required_credential_envs),
            "started_at": self._started_at,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "owned_processes": [
                {"role": child.role, "pid": child.process.pid,
                 "starttime": child.starttime, "log_path": str(child.log_path)}
                for child in self.children
            ],
        }
        if self.health is not None:
            receipt["health"] = dict(self.health)
            receipt["readiness_stage"] = (
                "worker_engine_ready" if self.plan.role == "worker" and self.health.get("ready")
                else "learner_ready" if self.plan.role == "learner" and self.health.get("ready")
                else "starting"
            )
            receipt["service_reused"] = self.service_reused
        if error_code is not None:
            receipt["error_code"] = error_code
        _write_private_json(self.receipt_path, receipt)

    def _verify_no_credential_embedded(self) -> None:
        prepared_bytes = [item.source.read_bytes() for item in self.plan.prepared_files]
        unique_values: set[bytes] = set()
        for credential in self.plan.credential_files:
            _verify_credential_source(credential)
            value = _read_secret(credential.source, expected_sha256=credential.sha256).encode("utf-8")
            if self.plan.role == "learner" and value in unique_values:
                raise VastRemoteError("Learner environment credentials must contain distinct secret values")
            unique_values.add(value)
            if any(value in content for content in prepared_bytes):
                raise VastRemoteError("a prepared deployment file contains a separately supplied credential")

    def start(self) -> dict[str, Any]:
        """Connect, transfer role credentials privately, start, and await role readiness."""
        if self.children or self.closed:
            raise VastRemoteError("Vast attach session cannot be started more than once")
        self._started_at = datetime.now(timezone.utc).isoformat()
        try:
            _local_dependency_preflight(self.plan, connect=self.connect)
            tunnel = self._spawn("ssh_tunnel", build_tunnel_argv(self.plan, ssh_bin=self.ssh_bin))
            remote_state = self._run_ssh_check()
            root_state = remote_state["root_state"]
            service_port_open = remote_state["service_port_open"]
            if root_state == "foreign":
                raise VastRemoteError("remote deployment root is foreign, incomplete, or differs from the prepared identity")
            if root_state in {"absent", "empty", "pending_owner"}:
                if service_port_open:
                    raise VastRemoteError("remote service port is already occupied without a matching deployment identity")
                if not remote_state["parent_writable"]:
                    raise VastRemoteError("remote deployment parent is not an existing writable directory")
            if root_state == "owned" and remote_state["active_start"] and not service_port_open:
                raise VastRemoteError(f"matching remote {self.plan.role} startup is already in progress; refusing duplicate start")
            if root_state == "owned" and service_port_open and self.plan.role == "learner" and not remote_state["active_start"]:
                raise VastRemoteError("remote Learner port is open without the matching owned startup process; refusing reuse")
            if root_state == "owned" and service_port_open and remote_state["credential_state"] != "matching":
                raise VastRemoteError("remote service is running but its private credential files do not match; refusing reuse")
            if not tunnel.is_alive():
                raise VastRemoteError("SSH tunnel exited before its forwards became ready")
            self._wait_forward_ready(tunnel)
            if root_state == "owned" and service_port_open:
                _verify_credential_source(self.plan.health_credential)
                credential = _read_secret(
                    self.plan.health_credential.source,
                    expected_sha256=self.plan.health_credential.sha256,
                )
                self._wait_for_existing_service(credential, tunnel=tunnel)
                self.service_reused = True
                success_status = "worker_started" if self.plan.role == "worker" else "learner_ready"
                self._write_receipt(status=success_status)
                return self._report(status=success_status)

            self._verify_no_credential_embedded()
            self._upload_prepared_files_and_credentials(root_state=root_state)
            if not tunnel.is_alive():
                raise VastRemoteError("SSH tunnel exited during prepared-file upload")
            _verify_credential_source(self.plan.health_credential)
            remote_exec = self._start_remote_command()
            health_credential = _read_secret(
                self.plan.health_credential.source,
                expected_sha256=self.plan.health_credential.sha256,
            )
            deadline = self.monotonic() + self.readiness_timeout
            last_health: dict[str, Any] | None = None
            while True:
                if not tunnel.is_alive() or not remote_exec.is_alive():
                    raise VastRemoteError(f"an owned SSH process exited before {self.plan.role} readiness")
                status_code, last_health = _probe_health_result(
                    self.plan, health_credential, getter=self.getter,
                )
                if status_code == 401:
                    raise VastRemoteError("/health rejected the prepared health credential")
                if status_code == 200 and last_health is None:
                    raise VastRemoteError("/health returned invalid status for the selected deployment role")
                if last_health is not None:
                    if (self.plan.role == "worker"
                            and last_health.get("worker_id") != self.plan.service_identity):
                        raise VastRemoteError("Worker health returned a different worker_id")
                    self.health = last_health
                    if last_health["ready"] is True:
                        success_status = "worker_started" if self.plan.role == "worker" else "learner_ready"
                        self._write_receipt(status=success_status)
                        return self._report(status=success_status)
                if self.monotonic() >= deadline:
                    self.health = last_health
                    error_code = "worker_engine_not_ready" if self.plan.role == "worker" else "learner_not_ready"
                    self._write_receipt(status="readiness_timeout", error_code=error_code)
                    raise VastRemoteError(f"{self.plan.role} did not report /health.ready before the bounded readiness timeout")
                self.sleep(min(READINESS_POLL_SECONDS, max(0.0, deadline - self.monotonic())))
        except BaseException as exc:
            if not isinstance(exc, KeyboardInterrupt):
                try:
                    self._write_receipt(status="failed", error_code="attach_failed")
                except (OSError, VastRemoteError, TypeError, ValueError) as receipt_error:
                    exc.add_note(f"attach receipt could not be written ({type(receipt_error).__name__})")
            self.close()
            raise

    def wait_foreground(self, *, poll_seconds: float = 1.0) -> int:
        """Hold the local SSH children until one exits or the caller interrupts."""
        if not self.children or self.closed:
            raise VastRemoteError("Vast attach session is not running")
        try:
            while True:
                exited = next((child for child in self.children if child.process.poll() is not None), None)
                if exited is not None:
                    status = int(exited.process.returncode or 0)
                    self._write_receipt(status="ssh_session_ended", error_code=exited.role)
                    return status
                if any(process_starttime(child.process.pid) != child.starttime for child in self.children):
                    self._write_receipt(status="ownership_lost", error_code="ssh_pid_identity_changed")
                    return 2
                self.sleep(poll_seconds)
        finally:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for child in reversed(self.children):
            _terminate_owned(child)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _build_parser(expected_role: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"qwen-rl vast {expected_role} attach")
    parser.add_argument("--deployment-dir", type=Path, required=True)
    parser.add_argument("--host", required=True, help="explicit SSH host from the existing Vast instance")
    parser.add_argument("--ssh-port", type=int, required=True, help="explicit mapped SSH port")
    parser.add_argument("--ssh-user", required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--known-hosts-file", type=Path, required=True)
    parser.add_argument("--token-file", type=Path,
                        help="private Worker token file; required only for role=worker")
    parser.add_argument("--credential", action="append", default=[], metavar="ENV=/ABS/FILE",
                        help="private Learner environment credential file; repeat for every prepared credential_env")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true",
                       help="upload the prepared role bundle and private credentials, then start it in the existing instance")
    modes.add_argument("--dry-run", action="store_true",
                       help="print validated attach metadata without network or file mutation")
    parser.add_argument("--readiness-timeout", type=float, default=READINESS_TIMEOUT)
    return parser


def main(argv: Sequence[str] | None = None, *, expected_role: str) -> int:
    if expected_role not in _ROLE_FILES:
        raise VastRemoteError("attach requires the role selected by the public vast subcommand")
    args = _build_parser(expected_role).parse_args(argv)
    try:
        credentials: list[tuple[str, Path]] = []
        for value in args.credential:
            env_name, separator, source = value.partition("=")
            if not separator or not _ENV_NAME.fullmatch(env_name) or not source or not Path(source).is_absolute():
                raise VastRemoteError("--credential must use ENV=/absolute/private/file syntax")
            credentials.append((env_name, Path(source)))
        plan = build_attach_plan(
            args.deployment_dir,
            expected_role=expected_role,
            host=args.host,
            ssh_port=args.ssh_port,
            ssh_user=args.ssh_user,
            identity_file=args.identity_file,
            known_hosts_file=args.known_hosts_file,
            token_file=args.token_file,
            credentials=credentials,
        )
        if not args.execute:
            print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))
            return 0
        session = VastAttachSession(plan, readiness_timeout=args.readiness_timeout)
        report = session.start()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return session.wait_foreground()
    except KeyboardInterrupt:
        return 130
    except VastRemoteError as exc:
        print(f"Vast attach failed: {exc}", file=sys.stderr)
        return 2
