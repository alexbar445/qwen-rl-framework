"""Validated node contracts and on-disk role configuration."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import stat
from pathlib import Path
import subprocess
from typing import Any, Callable, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)
import yaml

from .identity import (
    NodeIdentity,
    allow_group_traverse,
    atomic_write,
    canonical_public_key,
    ensure_private_dir,
    read_private_json,
    set_dir_owner,
    set_file_owner,
    write_node_ssh_state,
)
from .images import ImageArchiveDescriptor
from .worker_release import (
    ResolvedWorkerImage,
    WorkerRelease,
    WorkerReleaseError,
    load_worker_release,
    select_worker_image,
)
from .contracts import GPUCapability, NodeRegistration


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_LOCAL_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{16,512}$")
_NODE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_SITE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class NodeConfigurationError(RuntimeError):
    """Expected failure while validating or writing node configuration."""


def _is_lan_or_local_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        normalized = host.rstrip(".").lower()
        return ("." not in normalized or normalized.endswith(
            (".local", ".lan", ".cluster.local", ".svc")))


class StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelDefaults(StrictSettings):
    artifact_id: str = Field(min_length=1, max_length=300)
    manifest_url: str = Field(min_length=1, max_length=2048)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_path: str = Field(min_length=1, max_length=4096)
    base_fingerprint: str = Field(min_length=1, max_length=500)
    artifact_fingerprint: str = Field(min_length=1, max_length=500)
    container_path: str = Field(default="/models/base", min_length=1, max_length=4096)

    @field_validator("manifest_url")
    @classmethod
    def valid_manifest_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("model.manifest_url must be an absolute HTTP(S) URL without credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("model.manifest_url cannot contain a query or fragment")
        if "\r" in value or "\n" in value or "\x00" in value:
            raise ValueError("model.manifest_url must be a single-line URL")
        if not _is_lan_or_local_host(parsed.hostname):
            raise ValueError("model.manifest_url must resolve to the site LAN or local DNS")
        return value

    @field_validator("cache_path", "container_path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute() or "\x00" in value:
            raise ValueError("model paths must be absolute")
        return value

    @field_validator("artifact_id", "base_fingerprint", "artifact_fingerprint")
    @classmethod
    def fingerprints(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("model artifact and fingerprint identities must be SHA-256 values")
        return value

    @model_validator(mode="after")
    def ids_match_and_manifest_is_content_addressed(self) -> "ModelDefaults":
        if self.artifact_id != self.artifact_fingerprint:
            raise ValueError("model.artifact_id must equal model.artifact_fingerprint")
        if not urlsplit(self.manifest_url).path.rstrip("/").endswith(f"/artifacts/{self.manifest_sha256}"):
            raise ValueError("model.manifest_url must use its pinned /artifacts/<sha256> route")
        return self


class WorkerDefaults(StrictSettings):
    port: int = Field(ge=1, le=65535)
    inference: dict[str, Any]
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    request_timeout: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    @field_validator("inference")
    @classmethod
    def inference_object(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError("worker.inference must be a non-empty object")
        return value


class ReverseSSHDefaults(StrictSettings):
    central_host: str = Field(min_length=1, max_length=253)
    central_port: int = Field(default=22, ge=1, le=65535)
    central_user: str = Field(min_length=1, max_length=128)
    local_api_port: int = Field(default=8310, ge=1, le=65535)
    remote_api_port: int = Field(ge=1024, le=65535)
    remote_ssh_port: int = Field(ge=1024, le=65535)
    central_host_key: str = Field(min_length=1, max_length=2000)

    @field_validator("central_host")
    @classmethod
    def single_line_host(cls, value: str) -> str:
        if any(char.isspace() or ord(char) < 32 for char in value) or value.startswith("-"):
            raise ValueError("reverse SSH host is invalid")
        try:
            ipaddress.ip_address(value)
        except ValueError:
            if (len(value) > 253 or not re.fullmatch(
                    r"(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*\.?",
                    value)):
                raise ValueError("reverse SSH host must be a DNS name or IP address")
        return value

    @field_validator("central_user")
    @classmethod
    def safe_user(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
            raise ValueError("reverse SSH user is invalid")
        return value

    @field_validator("central_host_key")
    @classmethod
    def safe_host_key(cls, value: str) -> str:
        if "\r" in value or "\n" in value or "\x00" in value:
            raise ValueError("reverse SSH host key must be a single line")
        canonical_public_key(value)
        return value.strip()

    @model_validator(mode="after")
    def distinct_listeners(self) -> "ReverseSSHDefaults":
        if self.remote_api_port == self.remote_ssh_port:
            raise ValueError("reverse SSH API and SSH listener ports must be distinct")
        return self


class BootstrapDefaults(StrictSettings):
    """Deployment-specific subset of the registration response."""

    gateway_url: str = Field(min_length=1, max_length=2048)
    worker_image: str | None = Field(default=None, min_length=1, max_length=512)
    worker_image_mode: Literal["registry", "archive"] = "registry"
    image_archive: ImageArchiveDescriptor | None = None
    worker_release: WorkerRelease | None = None
    gateway_image: ImageArchiveDescriptor | None = None
    model: ModelDefaults
    worker: WorkerDefaults
    lineage_id: str = Field(min_length=1, max_length=128)
    admin_ssh_public_key: str = Field(min_length=1, max_length=2000)
    ssh_port: int = Field(default=2222, ge=1, le=65535)
    worker_compose_file: str = Field(default="/opt/qwen-rl/deploy/worker/compose.yaml", min_length=1, max_length=4096)
    site_compose_file: str = Field(default="/opt/qwen-rl/deploy/cluster/compose.site.yaml", min_length=1, max_length=4096)
    worker_uid: int = Field(default=10001, ge=1)
    worker_gid: int = Field(default=10001, ge=1)
    heartbeat_interval_seconds: int = Field(default=15, ge=2, le=3600)
    reverse_ssh: ReverseSSHDefaults | None = None

    @field_validator("admin_ssh_public_key")
    @classmethod
    def valid_admin_key(cls, value: str) -> str:
        if "\r" in value or "\n" in value or "\x00" in value:
            raise ValueError("admin_ssh_public_key must be a single line")
        return canonical_public_key(value)

    @field_validator("gateway_url")
    @classmethod
    def valid_gateway_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("gateway_url must be an absolute HTTP(S) URL without credentials")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("gateway_url must be a site origin without a path, query or fragment")
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError as exc:
            raise ValueError("gateway_url must use a routable private LAN IP literal") from exc
        if (not address.is_private or address.is_loopback or address.is_link_local
                or address.is_unspecified or address.is_multicast):
            raise ValueError("gateway_url must use a routable private LAN IP literal")
        return value.rstrip("/")

    @field_validator("worker_image")
    @classmethod
    def digest_pinned_image(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not (_IMAGE_DIGEST.fullmatch(value) or _LOCAL_IMAGE_ID.fullmatch(value)):
            raise ValueError("worker_image must be a registry digest or local sha256 image ID")
        return value

    @field_validator("worker_release", mode="before")
    @classmethod
    def strict_worker_release(cls, value: Any) -> WorkerRelease | None:
        if value is None:
            return None
        try:
            return load_worker_release(value)
        except WorkerReleaseError as exc:
            raise ValueError(f"worker_release is invalid: {exc}") from exc

    @model_validator(mode="after")
    def worker_model_path_matches(self) -> "BootstrapDefaults":
        import uuid
        try:
            parsed_lineage = uuid.UUID(self.lineage_id)
        except ValueError as exc:
            raise ValueError("lineage_id must be a canonical UUID") from exc
        if str(parsed_lineage) != self.lineage_id.lower():
            raise ValueError("lineage_id must use canonical UUID formatting")
        if self.worker_release is None:
            if self.worker_image is None or self.gateway_image is not None:
                raise ValueError("single-image mode requires worker_image and forbids gateway_image")
            if self.worker_image_mode == "registry":
                if not _IMAGE_DIGEST.fullmatch(self.worker_image) or self.image_archive is not None:
                    raise ValueError("registry image mode requires only a digest-pinned worker_image")
            elif self.image_archive is None or self.worker_image != self.image_archive.image_id:
                raise ValueError("archive image mode requires worker_image to match the pinned archive image ID")
            if self.image_archive is not None:
                self.image_archive.validate_content_url()
        else:
            if self.worker_image is not None or self.image_archive is not None:
                raise ValueError("worker release mode is mutually exclusive with a single worker_image")
            if self.gateway_image is None:
                raise ValueError("worker release mode requires a separate gateway_image archive descriptor")
            self.gateway_image.validate_content_url()
            configured_backend = self.worker.inference.get("backend", "vllm")
            if configured_backend != self.worker_release.backend:
                raise ValueError(
                    f"Worker release backend {self.worker_release.backend!r} does not match inference backend "
                    f"{configured_backend!r}"
                )
        configured = self.worker.inference.get("model_path")
        if configured is not None and configured != self.model.container_path:
            raise ValueError("worker.inference.model_path must match model.container_path")
        for field_name in ("worker_compose_file", "site_compose_file"):
            if not Path(getattr(self, field_name)).is_absolute():
                raise ValueError(f"{field_name} must be an absolute installed path")
        return self


class NodeConfiguration(StrictSettings):
    schema_version: str = "qwen-rl-lan-node/v1"
    controller_url: str
    site_id: str
    node_id: str
    role: Literal["gateway", "worker"]
    boot_id: str
    state_dir: str
    worker_config_path: str | None
    worker_token_path: str
    compose_env_path: str | None
    gateway_compose_env_path: str | None
    gateway_config_path: str | None
    gateway_host_config_path: str | None = None
    worker_compose_file: str | None
    site_compose_file: str
    gateway_node_token_path: str | None
    gateway_bearer_path: str | None
    worker_url: str | None
    gateway_url: str
    lan_host: str
    worker_image: str
    worker_image_mode: Literal["registry", "archive"] = "registry"
    image_archive: ImageArchiveDescriptor | None = None
    worker_release_selection: ResolvedWorkerImage | None = None
    model: ModelDefaults
    gpu_uuid: str | None
    worker_uid: int = Field(ge=0)
    worker_gid: int = Field(ge=0)
    owner_uid: StrictInt | None = Field(default=None, ge=1)
    owner_gid: StrictInt | None = Field(default=None, ge=0)
    write_training_ssh_state: StrictBool = True
    ssh_port: int = Field(ge=1, le=65535)
    sshd_config_path: str
    authorized_keys_path: str
    reverse_ssh_key_path: str
    host_key_path: str
    dragonfly_worker_peer_path: str | None
    dragonfly_seed_peer_path: str | None
    dragonfly_dfget_path: str = "/usr/local/bin/dfget"
    dragonfly_dfdaemon_path: str = "/usr/local/bin/dfdaemon"
    dragonfly_socket_path: str = "/run/dragonfly/dfdaemon.sock"
    heartbeat_interval_seconds: int = Field(ge=2, le=3600)
    needs_driver: bool = False
    needs_host_packages: bool = False
    reverse_ssh: ReverseSSHDefaults | None = None

    @field_validator("dragonfly_dfget_path", "dragonfly_dfdaemon_path", "dragonfly_socket_path")
    @classmethod
    def absolute_dragonfly_runtime_paths(cls, value: str) -> str:
        if not Path(value).is_absolute() or any(char in value for char in "\r\n\x00"):
            raise ValueError("Dragonfly runtime paths must be absolute single-line paths")
        return value

    @model_validator(mode="after")
    def node_identity_mappings_are_coherent(self) -> "NodeConfiguration":
        if (self.owner_uid is None) != (self.owner_gid is None):
            raise ValueError("host owner UID and GID must be supplied together")
        if ((self.worker_uid == 0 or self.worker_gid == 0)
                and (self.worker_uid, self.worker_gid) != (0, 0)):
            raise ValueError("rootless container identity must map UID and GID zero together")
        if (self.worker_uid, self.worker_gid) == (0, 0) and (
            self.owner_uid is None or self.owner_uid == 0
        ):
            raise ValueError("rootless container root requires a non-root account-owned host mapping")
        if self.worker_release_selection is not None:
            selected = self.worker_release_selection
            if (self.role != "worker" or self.worker_image_mode != "archive"
                    or self.image_archive != selected.descriptor
                    or self.worker_image != selected.descriptor.image_id
                    or self.gpu_uuid is None):
                raise ValueError("resolved Worker release must be pinned to this Worker GPU and image archive")
        elif self.image_archive is not None and self.worker_image != self.image_archive.image_id:
            raise ValueError("archive image mode requires worker_image to match the pinned archive image ID")
        return self


def validate_controller_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise NodeConfigurationError("controller URL must use HTTPS and contain no credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise NodeConfigurationError("controller URL must be an HTTPS origin without a path, query, or fragment")
    return value.rstrip("/")


def build_worker_config(
    *, node_id: str, defaults: BootstrapDefaults,
) -> dict[str, Any]:
    inference = dict(defaults.worker.inference)
    inference["model_path"] = defaults.model.container_path
    return {
        "worker_id": node_id,
        "deployment_scope": "gateway",
        "artifact_transport": "dragonfly",
        "host": "0.0.0.0",
        "port": defaults.worker.port,
        "inference": inference,
        "learner_url": defaults.gateway_url,
        "adapter_cache": "/state/adapters",
        "telemetry_dir": "/state/telemetry",
        "lora_rank": defaults.worker.lora_rank,
        "lora_alpha": defaults.worker.lora_alpha,
        "request_timeout": defaults.worker.request_timeout,
        "model_artifact_id": defaults.model.artifact_id,
        "model_manifest_sha256": defaults.model.manifest_sha256,
        "base_fingerprint": defaults.model.base_fingerprint,
        "artifact_fingerprint": defaults.model.artifact_fingerprint,
    }


def build_gateway_config(
    *, node_id: str, site_id: str, state_dir: Path, controller_url: str,
    defaults: BootstrapDefaults, host_secrets: bool = False,
) -> dict[str, Any]:
    gateway_root = Path(state_dir).absolute() / "gateway"
    if host_secrets:
        node_token = f"file:{gateway_root / 'node_token'}"
        gateway_bearer = f"file:{gateway_root / 'gateway_bearer'}"
    else:
        node_token = "file:/run/secrets/gateway_node_token"
        gateway_bearer = "file:/run/secrets/gateway_bearer"
    return {
        "site_id": site_id,
        "node_id": node_id,
        "controller_url": controller_url,
        "node_token": node_token,
        "gateway_bearer": gateway_bearer,
        "state_dir": str(gateway_root / "state"),
        "policy_source_url": controller_url,
        "policy_lineage_id": defaults.lineage_id,
        "artifact_seed_origin": defaults.gateway_url,
        "request_timeout_seconds": 180.0,
        "registry_poll_seconds": 1.0,
        "registry_stale_seconds": 5.0,
    }


def _dotenv_value(value: str | Path) -> str:
    text = str(value)
    if any(char in text for char in "\r\n\x00"):
        raise NodeConfigurationError("Compose environment values must be single-line")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'


def _write_dfdaemon_peer_config(config: NodeConfiguration, *, seed: bool) -> dict[str, str]:
    expected_role = "gateway" if seed else "worker"
    if config.role != expected_role:
        raise NodeConfigurationError(f"only {expected_role} nodes may use this Dragonfly peer role")
    from .dragonfly import validate_dfdaemon_config

    gateway = urlsplit(config.gateway_url)
    gateway_host = gateway.hostname
    if not gateway_host:
        raise NodeConfigurationError("Gateway URL has no scheduler host")
    scheduler_host = f"[{gateway_host}]" if ":" in gateway_host else gateway_host
    dragonfly_dir = ensure_private_dir(Path(config.state_dir) / "dragonfly")
    peer_name = "seed-peer.yaml" if seed else "worker-peer.yaml"
    config_path = dragonfly_dir / peer_name
    dynconfig_path = dragonfly_dir / "dynconfig.yaml"
    cache_dir = dragonfly_dir / "cache"
    storage_dir = dragonfly_dir / "storage"
    plugin_dir = dragonfly_dir / "plugins"
    for directory in (cache_dir, storage_dir, plugin_dir):
        ensure_private_dir(directory)
    worker_peer = {
        "console": False,
        "log": {"level": "info"},
        "host": {"idc": config.site_id, "location": "lan", "ip": config.lan_host},
        "server": {"pluginDir": str(plugin_dir), "cacheDir": str(cache_dir)},
        "download": {
            "protocol": "tcp",
            "server": {"socketPath": config.dragonfly_socket_path},
            "pieceTimeout": "40s",
            "concurrentPieceCount": 10,
        },
        "upload": {"server": {"port": 4000}, "disableShared": False},
        "scheduler": {"enableBackToSource": seed, "announceInterval": "10s",
                      "scheduleTimeout": "3h", "maxScheduleCount": 5},
        "dynconfig": {"path": str(dynconfig_path), "refreshInterval": "1m"},
        "storage": {"server": {"tcpPort": 4005, "quicPort": 4006}, "dir": str(storage_dir), "keep": True},
        "gc": {"interval": "900s", "policy": {"taskTTL": "21600s"}},
        "proxy": {"server": {"port": 4001}, "disableBackToSource": not seed, "prefetch": False},
        "metrics": {"server": {"port": 4002}},
    }
    if seed:
        worker_peer["seedPeer"] = {"enable": True, "type": "super"}
    else:
        worker_peer["seedPeer"] = {"enable": False}
    scheduler_port = 8002
    dynamic = {"scheduler": {"addrs": [f"{scheduler_host}:{scheduler_port}"]}}
    atomic_write(config_path, yaml.safe_dump(worker_peer, sort_keys=False).encode("utf-8"))
    atomic_write(dynconfig_path, yaml.safe_dump(dynamic, sort_keys=False).encode("utf-8"))
    validate_dfdaemon_config(config_path, allow_lan_source_fallback=seed)
    return {"peer_config": str(config_path), "dynconfig": str(dynconfig_path),
            "scheduler": f"{scheduler_host}:{scheduler_port}"}


def write_worker_dfdaemon_config(config: NodeConfiguration) -> dict[str, str]:
    """Write a Worker peer config with source fallback disabled."""
    return _write_dfdaemon_peer_config(config, seed=False)


def write_seed_dfdaemon_config(config: NodeConfiguration) -> dict[str, str]:
    """Write the Gateway seed peer config; only this role may use LAN source fallback."""
    return _write_dfdaemon_peer_config(config, seed=True)


def load_node_config(path: Path) -> tuple[NodeConfiguration, dict[str, Any]]:
    raw = read_private_json(path)
    try:
        config = NodeConfiguration.model_validate(raw)
    except ValidationError as exc:
        raise NodeConfigurationError("node config does not match the T4 schema") from exc
    if (config.owner_uid is not None
            and (os.geteuid() != config.owner_uid or os.getegid() != config.owner_gid)):
        raise NodeConfigurationError("node config host owner mapping differs from the invoking account")
    secrets = read_private_json(Path(config.state_dir) / "credentials.json")
    if secrets.get("site_id") != config.site_id or secrets.get("controller_url") != config.controller_url:
        raise NodeConfigurationError("node credentials do not match node config")
    if not _TOKEN.fullmatch(str(secrets.get("node_token", ""))) or not _TOKEN.fullmatch(str(secrets.get("worker_token", ""))):
        raise NodeConfigurationError("node credentials are invalid")
    return config, secrets


def node_token_for_rejoin(state_dir: Path, *, controller_url: str, site_id: str) -> str | None:
    """Reuse a node token only when its durable credentials match the target origin/site."""
    secret_path = Path(state_dir) / "credentials.json"
    if not secret_path.exists():
        return None
    prior = read_private_json(secret_path)
    candidate = prior.get("node_token")
    if (prior.get("site_id") == site_id and prior.get("controller_url") == controller_url
            and isinstance(candidate, str) and _TOKEN.fullmatch(candidate)):
        return candidate
    return None


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def prepare_node_configuration(
    *,
    state_dir: Path,
    controller_url: str,
    registration: NodeRegistration,
    identity: NodeIdentity,
    response: dict[str, Any],
    chosen_gpu: GPUCapability | None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    model_cache_path: Path | None = None,
    worker_compose_file: Path | None = None,
    container_uid: int | None = None,
    container_gid: int | None = None,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
    dragonfly_dfget_path: Path | None = None,
    dragonfly_dfdaemon_path: Path | None = None,
    dragonfly_socket_path: Path | None = None,
    write_training_ssh_state: bool = True,
) -> NodeConfiguration:
    """Write a registered node's credential, SSH, and role-specific service state."""
    state_dir = Path(state_dir).expanduser().absolute()
    role = registration.role
    if (owner_uid is None) != (owner_gid is None):
        raise NodeConfigurationError("host owner UID and GID must be provided together")
    if (container_uid is None) != (container_gid is None):
        raise NodeConfigurationError("container UID and GID must be provided together")
    if (owner_uid is None) != (container_uid is None):
        raise NodeConfigurationError("user-owned node state requires an explicit container UID/GID mapping")
    if owner_uid is not None and (os.geteuid() != owner_uid or os.getegid() != owner_gid):
        raise NodeConfigurationError("user-owned node state requires the invoking UID and primary GID")
    if owner_uid is not None and owner_uid == 0:
        raise NodeConfigurationError("user-owned node state requires a non-root host owner")
    if owner_uid is not None and any(path is None for path in (
        dragonfly_dfget_path, dragonfly_dfdaemon_path, dragonfly_socket_path,
    )):
        raise NodeConfigurationError("user-owned node state requires explicit Dragonfly binary and socket paths")
    if owner_uid is not None:
        try:
            root_info = state_dir.lstat()
        except OSError as exc:
            raise NodeConfigurationError("user-owned node state directory is unavailable") from exc
        if (not stat.S_ISDIR(root_info.st_mode) or state_dir.is_symlink()
                or root_info.st_uid != owner_uid or root_info.st_gid != owner_gid):
            raise NodeConfigurationError("user-owned node state directory has unexpected ownership")
    if container_uid is not None:
        if (isinstance(container_uid, bool) or not isinstance(container_uid, int) or container_uid < 0
                or isinstance(container_gid, bool) or not isinstance(container_gid, int) or container_gid < 0):
            raise NodeConfigurationError("container UID and GID must be nonnegative integers")
        if owner_uid is None and (container_uid == 0 or container_gid == 0):
            raise NodeConfigurationError("rootless container identities require account-owned node state")
    if role != "worker" and (model_cache_path is not None or worker_compose_file is not None):
        raise NodeConfigurationError("model cache and Worker Compose paths apply only to Worker enrollment")
    defaults_value = dict(response["bootstrap_defaults"])
    defaults_model = dict(defaults_value.get("model") or {})
    if model_cache_path is not None:
        defaults_model["cache_path"] = str(Path(model_cache_path).expanduser().absolute())
    if defaults_model:
        defaults_value["model"] = defaults_model
    if worker_compose_file is not None:
        defaults_value["worker_compose_file"] = str(Path(worker_compose_file).expanduser().absolute())
        if owner_uid is not None:
            source_root = Path(worker_compose_file).expanduser().absolute().parent.parent
            defaults_value["site_compose_file"] = str(source_root / "cluster" / "compose.site.yaml")
    try:
        defaults = BootstrapDefaults.model_validate(defaults_value)
    except ValidationError as exc:
        raise NodeConfigurationError("registration bootstrap_defaults do not match the node schema") from exc
    if defaults.ssh_port != registration.ssh_port:
        raise NodeConfigurationError("registered ssh_port does not match the configured isolated sshd port")
    if defaults.worker.port != registration.worker_port:
        raise NodeConfigurationError("registered worker port does not match the configured worker port")

    worker_image = defaults.worker_image
    worker_image_mode = defaults.worker_image_mode
    image_archive = defaults.image_archive
    worker_release_selection: ResolvedWorkerImage | None = None
    if defaults.worker_release is not None:
        if role == "worker":
            if chosen_gpu is None:
                raise NodeConfigurationError("Worker release image selection requires a selected GPU")
            if not chosen_gpu.compute_capability:
                raise NodeConfigurationError(
                    f"selected GPU {chosen_gpu.uuid} has no compute capability; cannot select a Worker release image"
                )
            try:
                worker_release_selection = select_worker_image(
                    defaults.worker_release, chosen_gpu.compute_capability,
                )
            except WorkerReleaseError as exc:
                raise NodeConfigurationError(
                    f"cannot select a Worker release image for GPU {chosen_gpu.uuid} compute capability "
                    f"{chosen_gpu.compute_capability}: {exc}"
                ) from exc
            image_archive = worker_release_selection.descriptor
            worker_image = image_archive.image_id
            worker_image_mode = "archive"
        else:
            if defaults.gateway_image is None:
                raise NodeConfigurationError("Worker release defaults have no separate Gateway image")
            image_archive = defaults.gateway_image
            worker_image = image_archive.image_id
            worker_image_mode = "archive"

    if worker_image is None:
        raise NodeConfigurationError("node role has no resolved container image")
    credentials = {
        "controller_url": controller_url,
        "site_id": registration.site_id,
        "node_token": response["node_token"],
        "worker_token": response["worker_token"],
    }
    atomic_write(state_dir / "credentials.json", _json_bytes(credentials))

    worker_uid = container_uid if container_uid is not None else defaults.worker_uid
    worker_gid = container_gid if container_gid is not None else defaults.worker_gid
    node_id = registration.node_id
    worker_config_path = state_dir / "worker" / "worker.json" if role == "worker" else None
    worker_token_path = (
        state_dir / "worker" / "worker_auth_token"
        if role == "worker" else state_dir / "gateway" / "gateway_bearer"
    )
    compose_env_path = state_dir / "worker" / "compose.env" if role == "worker" else None
    gateway_config_path = state_dir / "gateway" / "gateway.json" if role == "gateway" else None
    gateway_host_config_path = state_dir / "gateway" / "gateway-host.json" if role == "gateway" else None
    gateway_compose_env_path = state_dir / "gateway" / "compose.env" if role == "gateway" else None
    gateway_node_token_path = state_dir / "gateway" / "node_token" if role == "gateway" else None
    gateway_bearer_path = state_dir / "gateway" / "gateway_bearer" if role == "gateway" else None
    if write_training_ssh_state:
        ssh_authorized_keys, sshd_config = write_node_ssh_state(
            state_dir=state_dir,
            admin_public_key=defaults.admin_ssh_public_key,
            lan_host=registration.lan_host,
            ssh_port=defaults.ssh_port,
            role=role,
            runner=runner,
        )
    else:
        ssh_directory = ensure_private_dir(state_dir / "ssh")
        ssh_authorized_keys = ssh_directory / "authorized_keys"
        sshd_config = ssh_directory / "sshd_config"
    try:
        node_config = NodeConfiguration(
            controller_url=controller_url,
            site_id=registration.site_id,
            node_id=node_id,
            role=role,
            boot_id=registration.boot_id,
            state_dir=str(state_dir),
            lan_host=registration.lan_host,
            worker_config_path=str(worker_config_path) if worker_config_path else None,
            worker_token_path=str(worker_token_path),
            compose_env_path=str(compose_env_path) if compose_env_path else None,
            gateway_compose_env_path=str(gateway_compose_env_path) if gateway_compose_env_path else None,
            gateway_config_path=str(gateway_config_path) if gateway_config_path else None,
            gateway_host_config_path=str(gateway_host_config_path) if gateway_host_config_path else None,
            worker_compose_file=defaults.worker_compose_file if role == "worker" else None,
            site_compose_file=defaults.site_compose_file,
            gateway_node_token_path=str(gateway_node_token_path) if gateway_node_token_path else None,
            gateway_bearer_path=str(gateway_bearer_path) if gateway_bearer_path else None,
            worker_url=f"http://127.0.0.1:{defaults.worker.port}" if role == "worker" else None,
            gateway_url=defaults.gateway_url,
            worker_image=worker_image,
            worker_image_mode=worker_image_mode,
            image_archive=image_archive,
            worker_release_selection=worker_release_selection,
            model=defaults.model,
            gpu_uuid=chosen_gpu.uuid if chosen_gpu else None,
            worker_uid=worker_uid,
            worker_gid=worker_gid,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            write_training_ssh_state=write_training_ssh_state,
            ssh_port=defaults.ssh_port,
            sshd_config_path=str(sshd_config),
            authorized_keys_path=str(ssh_authorized_keys),
            reverse_ssh_key_path=str(identity.key_path),
            host_key_path=str(identity.host_key_path),
            dragonfly_worker_peer_path=(
                str(state_dir / "dragonfly" / "worker-peer.yaml") if role == "worker" else None
            ),
            dragonfly_seed_peer_path=(
                str(state_dir / "dragonfly" / "seed-peer.yaml") if role == "gateway" else None
            ),
            dragonfly_dfget_path=str(Path(dragonfly_dfget_path).absolute()) if dragonfly_dfget_path else "/usr/local/bin/dfget",
            dragonfly_dfdaemon_path=str(Path(dragonfly_dfdaemon_path).absolute()) if dragonfly_dfdaemon_path else "/usr/local/bin/dfdaemon",
            dragonfly_socket_path=str(Path(dragonfly_socket_path).absolute()) if dragonfly_socket_path else "/run/dragonfly/dfdaemon.sock",
            heartbeat_interval_seconds=defaults.heartbeat_interval_seconds,
            reverse_ssh=defaults.reverse_ssh if role == "gateway" else None,
            needs_driver=(role == "worker" and chosen_gpu is None),
            needs_host_packages=False,
        )
    except ValidationError as exc:
        raise NodeConfigurationError("constructed node configuration did not match its contract") from exc

    if role == "worker":
        write_worker_dfdaemon_config(node_config)
        assert worker_config_path is not None and compose_env_path is not None
        worker_state = ensure_private_dir(worker_config_path.parent)
        writable_dirs = (
            "adapters", "telemetry", "jit", "jit/vllm", "jit/triton", "jit/torch",
            "jit/flashinfer", "hf-cache", "home", "results", "p2p-artifacts",
        )
        for name in writable_dirs:
            ensure_private_dir(worker_state / name)
        atomic_write(worker_config_path, _json_bytes(build_worker_config(node_id=node_id, defaults=defaults)))
        atomic_write(worker_token_path, response["worker_token"].encode("ascii"))
        if owner_uid is None:
            set_file_owner(worker_config_path, uid=defaults.worker_uid, gid=defaults.worker_gid)
            set_file_owner(worker_token_path, uid=defaults.worker_uid, gid=defaults.worker_gid)
            for name in writable_dirs:
                set_dir_owner(worker_state / name, uid=defaults.worker_uid, gid=defaults.worker_gid)
        else:
            owned_paths = [worker_config_path, worker_token_path,
                           *(worker_state / name for name in writable_dirs)]
            for path in owned_paths:
                info = path.lstat()
                if path.is_symlink() or info.st_uid != owner_uid or info.st_gid != owner_gid:
                    raise NodeConfigurationError("user-owned Worker state has an unexpected UID or GID")
        os.chmod(worker_state, 0o700 if owner_uid is not None else 0o755)
        compose_values = {
            "COMPOSE_FILE": defaults.worker_compose_file,
            "QWEN_WORKER_IMAGE": node_config.worker_image,
            "QWEN_NODE_ID": node_id,
            "QWEN_SITE_ID": registration.site_id,
            "WORKER_CONFIG_FILE": worker_config_path,
            "WORKER_AUTH_TOKEN_FILE": worker_token_path,
            "MODEL_DIR": defaults.model.cache_path,
            "MODEL_CONTAINER_PATH": defaults.model.container_path,
            "WORKER_STATE_DIR": worker_state,
            "WORKER_PORT": str(defaults.worker.port),
            "WORKER_BIND_ADDRESS": "0.0.0.0",
            "QWEN_UID": str(worker_uid),
            "QWEN_GID": str(worker_gid),
        }
        if chosen_gpu is not None:
            compose_values["GPU_UUID"] = chosen_gpu.uuid
            compose_values["QWEN_GPU_UUID"] = chosen_gpu.uuid
        compose_env = "".join(f"{name}={_dotenv_value(value)}\n" for name, value in compose_values.items())
        atomic_write(compose_env_path, compose_env.encode("utf-8"))
    else:
        assert (gateway_config_path is not None and gateway_host_config_path is not None
                and gateway_node_token_path is not None and gateway_bearer_path is not None)
        gateway_state = ensure_private_dir(gateway_config_path.parent / "state")
        atomic_write(gateway_node_token_path, response["node_token"].encode("ascii"))
        atomic_write(gateway_bearer_path, response["worker_token"].encode("ascii"))
        atomic_write(gateway_config_path, _json_bytes(build_gateway_config(
            node_id=node_id, site_id=registration.site_id, state_dir=state_dir,
            controller_url=controller_url, defaults=defaults,
        )))
        atomic_write(gateway_host_config_path, _json_bytes(build_gateway_config(
            node_id=node_id, site_id=registration.site_id, state_dir=state_dir,
            controller_url=controller_url, defaults=defaults, host_secrets=True,
        )))
        ensure_private_dir(gateway_state / "spool")
        gateway_owned_paths = (
            gateway_config_path.parent,
            gateway_config_path,
            gateway_host_config_path,
            gateway_node_token_path,
            gateway_bearer_path,
            gateway_state,
            gateway_state / "spool",
        )
        if owner_uid is None:
            for private_file in (gateway_config_path, gateway_host_config_path,
                                 gateway_node_token_path, gateway_bearer_path):
                set_file_owner(private_file, uid=defaults.worker_uid, gid=defaults.worker_gid)
            set_dir_owner(gateway_state, uid=defaults.worker_uid, gid=defaults.worker_gid)
            set_dir_owner(gateway_state / "spool", gid=defaults.worker_gid, uid=defaults.worker_uid)
            os.chmod(gateway_config_path.parent, 0o755)
        else:
            for path in gateway_owned_paths:
                metadata = path.lstat()
                if (path.is_symlink() or metadata.st_uid != owner_uid or metadata.st_gid != owner_gid
                        or not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode))):
                    raise NodeConfigurationError("user-owned Gateway state has an unexpected UID or GID")
                os.chmod(path, 0o700 if stat.S_ISDIR(metadata.st_mode) else 0o600)
        site_values = {
            "COMPOSE_FILE": defaults.site_compose_file,
            "QWEN_WORKER_IMAGE": node_config.worker_image,
            "QWEN_NODE_ID": node_id,
            "QWEN_SITE_ID": registration.site_id,
            "GATEWAY_CONFIG_FILE": gateway_config_path,
            "GATEWAY_NODE_TOKEN_FILE": gateway_node_token_path,
            "GATEWAY_BEARER_FILE": gateway_bearer_path,
            "GATEWAY_STATE_DIR": gateway_state,
            "GATEWAY_BIND_ADDRESS": "0.0.0.0",
            "GATEWAY_PORT": str(urlsplit(defaults.gateway_url).port or (
                443 if defaults.gateway_url.startswith("https:") else 80
            )),
            "QWEN_UID": str(worker_uid),
            "QWEN_GID": str(worker_gid),
        }
        assert gateway_compose_env_path is not None
        site_env = "".join(f"{name}={_dotenv_value(value)}\n" for name, value in site_values.items())
        atomic_write(gateway_compose_env_path, site_env.encode("utf-8"))
        write_seed_dfdaemon_config(node_config)

    atomic_write(state_dir / "node.json", _json_bytes(node_config.model_dump(mode="json")))
    atomic_write(state_dir / "bootstrap-defaults.json", _json_bytes(defaults.model_dump(mode="json")))
    if role == "gateway" and owner_uid is None:
        allow_group_traverse(state_dir, gid=defaults.worker_gid)
    return node_config
