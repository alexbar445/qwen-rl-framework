"""User-owned Fleet Worker lifecycle over the existing node and Compose runtime."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from ..bootstrap import join_cluster
from ..identity import atomic_write
from ..node import NodeDaemon, install_node_model_artifact, install_node_worker_image
from ..node_config import NodeConfiguration, load_node_config
from .compose import ComposePluginError, ensure_compose_plugin
from .contracts import UserAccount
from .contracts import WorkerDeployment as WorkerDeploymentSpec
from .host import (
    FleetHostError,
    HostInventory,
    current_user_account,
    host_has_nvidia_device,
    select_one_gpu,
)
from .paths import UserPaths
from .roles import (
    DockerAccess,
    FleetRoleError,
    RoleResult,
    StopResult,
    download_verified_artifact,
    ensure_user_dir,
    inspect_docker_access,
    operation_dir,
    probe_docker_uid_mapping,
    run_owned,
)

_SOURCE_BUNDLE_MAX_BYTES = 192 * 1024 * 1024
_ROLE_MARKER = "fleet-worker-ownership.json"
_JOURNAL = "worker-operation.json"
_WORKER_UNIT = "qwen-fleet-worker.service"
_DRAGONFLY_UNIT = "qwen-fleet-dragonfly.service"
_NODE_UNIT = "qwen-fleet-worker-node.service"
_WORKER_UNITS = (_WORKER_UNIT, _DRAGONFLY_UNIT, _NODE_UNIT)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, _json_bytes(value), mode=0o600)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
            raise FleetRoleError("Worker ownership record is not a private account-owned file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except FleetRoleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FleetRoleError("Worker ownership record cannot be read") from exc
    if not isinstance(value, dict):
        raise FleetRoleError("Worker ownership record has an invalid shape")
    return value


def _sha256_file(path: Path, *, maximum_bytes: int = 256 * 1024 * 1024) -> str:
    if path.is_symlink() or not path.is_file():
        raise FleetRoleError("Worker source path is not a regular file")
    info = path.stat()
    if info.st_size > maximum_bytes:
        raise FleetRoleError("Worker source file exceeds its size limit")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _systemd_value(value: str | Path) -> str:
    text = str(value)
    if not text or any(char in text for char in "\r\n\x00"):
        raise FleetRoleError("user service argument is invalid")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def _systemd_path(value: str | Path) -> str:
    text = str(value)
    if not Path(text).is_absolute() or any(char in text for char in "\r\n\x00"):
        raise FleetRoleError("user service working directory must be an absolute single-line path")
    escaped: list[str] = []
    for char in text:
        code = ord(char)
        if char == "%":
            escaped.append("%%")
        elif char == "\\" or code < 0x20 or char == " " or code == 0x7f:
            escaped.append(f"\\x{code:02x}")
        else:
            escaped.append(char)
    return "".join(escaped)


def _unit_bytes(lines: list[str]) -> bytes:
    if any(any(char in line for char in "\r\n\x00") for line in lines):
        raise FleetRoleError("user service unit contains an invalid line")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _source_extractor():
    source = Path(__file__).resolve().parents[3] / "deploy" / "bootstrap" / "install.py"
    if source.is_symlink() or not source.is_file():
        raise FleetRoleError("pinned source-bundle validator is unavailable")
    spec = importlib.util.spec_from_file_location("_qwen_fleet_worker_bundle_validator", source)
    if spec is None or spec.loader is None:
        raise FleetRoleError("pinned source-bundle validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_source_bundle(archive: Path, root: Path, digest: str) -> dict[str, Any]:
    root = Path(root)
    marker = root / ".fleet-source-bundle.json"
    ownership_path = root.parent / f".{digest}.source-ownership.json"
    expected_ownership = {
        "schema": "qwen-fleet-worker-source/v1",
        "managed_by": "qwen-fleet-worker",
        "sha256": digest,
    }
    ownership = _read_json(ownership_path)
    if ownership is None:
        if root.exists() or root.is_symlink():
            raise FleetRoleError("existing Worker source release has no Fleet ownership record")
        ownership = {**expected_ownership, "state": "extracting"}
        _write_json(ownership_path, ownership)
    elif any(ownership.get(key) != value for key, value in expected_ownership.items()):
        raise FleetRoleError("existing Worker source ownership record conflicts with its content hash")
    elif ownership.get("state") == "ready":
        if root.is_symlink():
            raise FleetRoleError("owned Worker source release path is a symlink")
        if root.is_dir():
            if _read_json(marker) != {"sha256": digest, "managed_by": "qwen-fleet-worker"}:
                raise FleetRoleError("existing Worker source release marker is invalid")
            return {"root": str(root), "sha256": digest, "reused": True}
        ownership = {**expected_ownership, "state": "extracting"}
        _write_json(ownership_path, ownership)
    elif ownership.get("state") != "extracting":
        raise FleetRoleError("existing Worker source ownership state is invalid")
    if root.exists() or root.is_symlink():
        try:
            info = root.lstat()
        except OSError as exc:
            raise FleetRoleError("partial Worker source release cannot be inspected") from exc
        if (root.is_symlink() or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
            raise FleetRoleError("partial Worker source release is not owned by this account")
        shutil.rmtree(root)
    extractor = _source_extractor()
    try:
        result = extractor.extract_verified_bundle(archive, root)
    except Exception as exc:
        raise FleetRoleError("pinned Worker source bundle failed its path, type, or size checks") from exc
    _write_json(marker, {"sha256": digest, "managed_by": "qwen-fleet-worker"})
    _write_json(ownership_path, {**expected_ownership, "state": "ready"})
    return {"root": str(root), "sha256": digest, **result, "reused": False}


def _node_identity(state_dir: Path) -> str:
    identity = _read_json(Path(state_dir) / "node-identity.json")
    node_id = identity.get("node_id") if identity else None
    if not isinstance(node_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,100}", node_id) is None:
        raise FleetRoleError("training node identity is not ready")
    return node_id


def _project_name(node_id: str) -> str:
    token = hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:20]
    return f"qwen-fleet-worker-{token}"


def _compose_env_value(value: str | Path) -> str:
    text = str(value)
    if any(char in text for char in "\r\n\x00"):
        raise FleetRoleError("Worker Compose environment contains an invalid value")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'


def _write_compose_environment(
    config: NodeConfiguration,
    *,
    fleet_node_id: str,
    deployment_id: str,
    account: UserAccount,
    paths: UserPaths,
    project: str,
) -> Path:
    if not config.compose_env_path:
        raise FleetRoleError("Worker Compose environment path is missing")
    env_path = Path(config.compose_env_path)
    if env_path.is_symlink() or not env_path.is_file():
        raise FleetRoleError("Worker Compose environment is not a private regular file")
    values = {
        "COMPOSE_PROJECT_NAME": project,
        "DFGET_PATH": config.dragonfly_dfget_path,
        "DFDAEMON_PATH": config.dragonfly_dfdaemon_path,
        "DRAGONFLY_RUNTIME_DIR": str(paths.runtime_dir / "dragonfly"),
        "QWEN_FLEET_MANAGED": "true",
        "QWEN_FLEET_NODE_ID": fleet_node_id,
        "QWEN_FLEET_DEPLOYMENT_ID": deployment_id,
        "QWEN_FLEET_ACCOUNT_UID": str(account.uid),
        "QWEN_UID": str(config.worker_uid),
        "QWEN_GID": str(config.worker_gid),
    }
    existing: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep and re.fullmatch(r"[A-Z][A-Z0-9_]{0,100}", key):
            existing[key] = value
    existing.update({key: _compose_env_value(value) for key, value in values.items()})
    atomic_write(env_path, ("".join(f"{key}={value}\n" for key, value in sorted(existing.items()))).encode(),
                 mode=0o600)
    info = env_path.stat()
    if info.st_uid != account.uid or info.st_gid != account.gid:
        raise FleetRoleError("Worker Compose environment is not owned by the registered account")
    return env_path


def _node_config_owned(config_path: Path, account: UserAccount) -> bool:
    try:
        info = config_path.lstat()
    except OSError:
        return False
    return (stat.S_ISREG(info.st_mode) and not config_path.is_symlink()
            and info.st_uid == account.uid and info.st_gid == account.gid)


class WorkerRoleAdapter:
    """Apply and drain a Worker using only account-owned files and user units."""

    role = "worker"

    def __init__(
        self,
        *,
        state_dir: Path,
        ca_bundle: Path,
        python_executable: Path,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        artifact_transport: httpx.BaseTransport | None = None,
        account: UserAccount | None = None,
        paths: UserPaths | None = None,
        stop_timeout_seconds: float = 180.0,
        health_timeout_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.account = account or current_user_account()
        expected_paths = UserPaths.from_account(self.account)
        self.paths = paths or expected_paths
        if self.paths != expected_paths:
            raise FleetRoleError("Fleet user paths do not match the registered account HOME")
        self.state_dir = Path(state_dir)
        if self.state_dir != self.paths.state_dir:
            raise FleetRoleError("Fleet role operation state must use the registered account's state path")
        self.ca_bundle = Path(ca_bundle)
        self.python_executable = Path(python_executable)
        self.runner = runner
        self.artifact_transport = artifact_transport
        self.stop_timeout_seconds = stop_timeout_seconds
        self.health_timeout_seconds = health_timeout_seconds
        self.sleep = sleep

    @property
    def role_dir(self) -> Path:
        return self.paths.roles_dir / "worker"

    @property
    def training_state_dir(self) -> Path:
        return self.paths.state_dir / "worker"

    @property
    def user_service_dir(self) -> Path:
        return self.paths.systemd_user_dir

    @property
    def ownership_path(self) -> Path:
        return self.role_dir / _ROLE_MARKER

    def _fleet_node_id(self) -> str:
        identity = _read_json(self.state_dir / "node-identity.json")
        value = identity.get("node_id") if identity else None
        if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{16,100}", value) is None:
            raise FleetRoleError("Fleet node identity is unavailable")
        return value

    def _load_ownership(self) -> dict[str, Any] | None:
        return _read_json(self.ownership_path)

    def _check_ownership(self, deployment_id: str, node_id: str,
                         owned_receipt: Mapping[str, Any] | None) -> dict[str, Any] | None:
        ownership = self._load_ownership()
        if ownership is None:
            for unit in _WORKER_UNITS:
                unit_path = self.user_service_dir / unit
                if unit_path.exists() or unit_path.is_symlink():
                    raise FleetRoleError("unowned Fleet Worker user service already exists")
            return None
        expected = {
            "schema": "qwen-fleet-worker-ownership/v1",
            "managed_by": "qwen-fleet-worker",
            "role": self.role,
            "fleet_node_id": node_id,
            "account_uid": self.account.uid,
            "account_gid": self.account.gid,
        }
        if any(ownership.get(key) != value for key, value in expected.items()):
            raise FleetRoleError("existing Worker ownership record belongs to another account or node")
        owned_deployments = {ownership.get("deployment_id"), ownership.get("previous_deployment_id")}
        if deployment_id not in owned_deployments and ownership.get("lifecycle_state") != "stopped":
            raise FleetRoleError("existing Worker state belongs to another Fleet deployment")
        if owned_receipt is not None:
            for key in ("fleet_node_id", "account_uid", "account_gid"):
                if key in owned_receipt and owned_receipt.get(key) != ownership.get(key):
                    raise FleetRoleError("Worker receipt differs from durable account ownership")
            receipt_deployment = owned_receipt.get("deployment_id")
            if (receipt_deployment is not None
                    and receipt_deployment not in {ownership.get("deployment_id"),
                                                   ownership.get("previous_deployment_id")}):
                raise FleetRoleError("Worker receipt differs from durable deployment ownership")
        units = ownership.get("unit_sha256")
        if not isinstance(units, dict):
            raise FleetRoleError("Worker ownership record has no user-service hashes")
        previous_units = ownership.get("previous_unit_sha256")
        if not isinstance(previous_units, dict):
            raise FleetRoleError("Worker ownership record has invalid previous user-service hashes")
        installing = ownership.get("lifecycle_state") == "installing"
        for unit in _WORKER_UNITS:
            unit_path = self.user_service_dir / unit
            if not unit_path.exists() and not unit_path.is_symlink():
                if (installing or (unit not in units and unit not in previous_units)):
                    continue
                raise FleetRoleError("owned Worker user-service file is missing")
            if unit_path.is_symlink() or not unit_path.is_file():
                raise FleetRoleError("Worker user-service path is not a regular file")
            observed_hash = _sha256_file(unit_path)
            expected_hashes = {units.get(unit)}
            if installing:
                expected_hashes.add(previous_units.get(unit))
            if observed_hash not in expected_hashes:
                raise FleetRoleError("Worker user-service file differs from the owned Fleet version")
        return ownership

    def _docker_access(self, expected: Mapping[str, Any] | None = None) -> DockerAccess:
        return inspect_docker_access(runner=self.runner, expected=expected)

    def _docker_project_containers(self, access: DockerAccess, project: str) -> list[str] | None:
        result = self.runner(
            access.command("ps", "--all", "--quiet", "--filter", f"label=com.docker.compose.project={project}"),
            check=False, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    def _inspect_container_labels(self, access: DockerAccess, container_id: str) -> dict[str, str] | None:
        result = self.runner(
            access.command("inspect", "--format", "{{json .Config.Labels}}", container_id),
            check=False, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        try:
            labels = json.loads((result.stdout or "").strip())
        except json.JSONDecodeError:
            return None
        return labels if isinstance(labels, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                           for k, v in labels.items()) else None

    def _docker_project_networks(self, access: DockerAccess, project: str) -> list[str] | None:
        result = self.runner(
            access.command("network", "ls", "--quiet", "--filter", f"label=com.docker.compose.project={project}"),
            check=False, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    def _inspect_network_labels(self, access: DockerAccess, network_id: str) -> dict[str, str] | None:
        result = self.runner(
            access.command("network", "inspect", "--format", "{{json .Labels}}", network_id),
            check=False, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        try:
            labels = json.loads((result.stdout or "").strip())
        except json.JSONDecodeError:
            return None
        return labels if isinstance(labels, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                           for k, v in labels.items()) else None

    def _project_is_owned(self, access: DockerAccess, project: str, node_id: str,
                          deployment_id: str, *, allow_absent: bool) -> bool:
        containers = self._docker_project_containers(access, project)
        if containers is None:
            return False
        networks = self._docker_project_networks(access, project)
        if networks is None:
            return False
        if not containers and not networks:
            return allow_absent
        expected_labels = {
            "com.docker.compose.project": project,
            "com.qwen-fleet.managed": "true",
            "com.qwen-fleet.node-id": node_id,
            "com.qwen-fleet.deployment-id": deployment_id,
            "com.qwen-fleet.account-uid": str(self.account.uid),
        }
        for container in containers:
            labels = self._inspect_container_labels(access, container)
            if labels is None or any(labels.get(key) != value for key, value in expected_labels.items()):
                return False
        for network in networks:
            labels = self._inspect_network_labels(access, network)
            if labels is None or any(labels.get(key) != value for key, value in expected_labels.items()):
                return False
        return True

    def prepare_install(self, deployment: WorkerDeploymentSpec, *, generation: int,
                        owned_receipt: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if not isinstance(deployment, WorkerDeploymentSpec):
            raise FleetRoleError("Worker adapter received another role")
        node_id = self._fleet_node_id()
        account = current_user_account()
        if account != self.account:
            raise FleetRoleError("effective account changed since Fleet adapter creation")
        ownership = self._check_ownership(deployment.deployment_id, node_id, owned_receipt)
        access = self._docker_access(ownership.get("docker") if ownership else None)
        project = str(ownership.get("compose_project") or _project_name(node_id)) if ownership else _project_name(node_id)
        if ownership is None:
            containers = self._docker_project_containers(access, project)
            networks = self._docker_project_networks(access, project)
            if containers is None or networks is None or containers or networks:
                raise FleetRoleError("Docker resources exist without a Fleet Worker ownership record")
        elif not self._project_is_owned(access, project, node_id, deployment.deployment_id,
                                        allow_absent=True):
            raise FleetRoleError("existing Docker project is not owned by this Fleet Worker")
        config_path = self.training_state_dir / "node.json"
        if config_path.exists() or config_path.is_symlink():
            if ownership is None or not _node_config_owned(config_path, self.account):
                raise FleetRoleError("training Worker state exists without a matching Fleet ownership record")
            try:
                config, _credentials = load_node_config(config_path)
            except Exception as exc:
                raise FleetRoleError("existing Fleet Worker configuration cannot be verified") from exc
            if config.role != "worker" or config.controller_url != deployment.controller_url:
                raise FleetRoleError("existing Worker identity cannot change Controller or role")
            if config.site_id != deployment.site_id:
                raise FleetRoleError("existing Worker site differs from this assignment")
            if (config.gpu_uuid != deployment.gpu_uuid
                    and ownership.get("lifecycle_state") != "stopped"):
                raise FleetRoleError("an active Worker cannot change its assigned GPU")
            if ownership.get("training_node_id") not in {None, config.node_id}:
                raise FleetRoleError("Worker training identity differs from its durable receipt")
            target_state = "owned"
            training_node_id = config.node_id
        else:
            if ownership is not None and ownership.get("training_node_id"):
                raise FleetRoleError("recorded Worker configuration is missing")
            target_state = "absent_before_install"
            training_node_id = None
        return {
            "role": self.role,
            "generation": generation,
            "deployment_id": deployment.deployment_id,
            "fleet_node_id": node_id,
            "account_uid": self.account.uid,
            "account_gid": self.account.gid,
            "target_state": target_state,
            "training_state_dir": str(self.training_state_dir),
            "training_node_id": training_node_id,
            "controller_url": deployment.controller_url,
            "site_id": deployment.site_id,
            "gpu_uuid": deployment.gpu_uuid,
            "compose_project": project,
            "docker": access.receipt(),
        }

    def _operation_record(self, generation: int, deployment_id: str) -> tuple[Path, dict[str, Any]]:
        directory = operation_dir(self.state_dir, generation, self.role, deployment_id, account=self.account)
        path = directory / _JOURNAL
        record = _read_json(path) or {
            "schema": "qwen-fleet-worker-operation/v1",
            "role": self.role,
            "generation": generation,
            "deployment_id": deployment_id,
            "state": "started",
        }
        if (record.get("role") != self.role or record.get("generation") != generation
                or record.get("deployment_id") != deployment_id):
            raise FleetRoleError("Worker retry journal differs from its assignment generation")
        return path, record

    @staticmethod
    def _record_step(path: Path, record: dict[str, Any], state: str, **receipts: Any) -> dict[str, Any]:
        record = {**record, **receipts, "state": state, "updated_at": time.time()}
        _write_json(path, record)
        return record

    def _persist_ownership(
        self,
        *,
        fleet_node_id: str,
        deployment_id: str,
        generation: int,
        config: NodeConfiguration | None,
        docker: Mapping[str, Any],
        project: str,
        source_root: Path,
        source_bundle_sha256: str,
        user_units: Mapping[str, bytes] | None,
        docker_mapping: Mapping[str, Any] | None = None,
        previous_deployment_id: str | None = None,
        compose_plugin: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        ensure_user_dir(self.role_dir, account=self.account)
        existing = self._load_ownership()
        if user_units is None:
            unit_hashes = dict((existing or {}).get("unit_sha256") or {})
            previous_unit_hashes = dict((existing or {}).get("previous_unit_sha256") or {})
        else:
            unit_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in user_units.items()}
            if existing and existing.get("lifecycle_state") == "installing":
                previous_unit_hashes = dict(
                    existing.get("previous_unit_sha256") or existing.get("unit_sha256") or {}
                )
            else:
                previous_unit_hashes = dict((existing or {}).get("unit_sha256") or {})
        receipt = {
            "schema": "qwen-fleet-worker-ownership/v1",
            "managed_by": "qwen-fleet-worker",
            "role": self.role,
            "lifecycle_state": "installing",
            "fleet_node_id": fleet_node_id,
            "account_uid": self.account.uid,
            "account_gid": self.account.gid,
            "deployment_id": deployment_id,
            "previous_deployment_id": previous_deployment_id,
            "generation": generation,
            "training_node_id": config.node_id if config else None,
            "training_state_dir": str(self.training_state_dir),
            "controller_url": config.controller_url if config else None,
            "site_id": config.site_id if config else None,
            "gpu_uuid": config.gpu_uuid if config else None,
            "compose_project": project,
            "compose_file": str(source_root / "deploy" / "worker" / "compose.yaml"),
            "source_root": str(source_root),
            "artifact_sha256": source_bundle_sha256,
            "docker": dict(docker),
            "compose_plugin": dict(compose_plugin) if compose_plugin is not None else None,
            "docker_mapping": dict(docker_mapping) if docker_mapping is not None else None,
            "unit_sha256": unit_hashes,
            "previous_unit_sha256": previous_unit_hashes,
        }
        _write_json(self.ownership_path, receipt)
        return receipt

    def _mark_ownership_active(self, ownership: dict[str, Any]) -> dict[str, Any]:
        unit_hashes = ownership.get("unit_sha256")
        if not isinstance(unit_hashes, dict):
            raise FleetRoleError("Worker ownership record has no user-service hashes")
        for unit in _WORKER_UNITS:
            path = self.user_service_dir / unit
            if (not path.is_file() or path.is_symlink()
                    or unit_hashes.get(unit) != _sha256_file(path)):
                raise FleetRoleError("Worker user services do not match their durable ownership hashes")
        active = {**ownership, "lifecycle_state": "active", "previous_unit_sha256": {}}
        _write_json(self.ownership_path, active)
        return active

    def _user_units(self, config: NodeConfiguration, source_root: Path,
                    project: str, access: DockerAccess) -> dict[str, bytes]:
        compose_file = Path(config.worker_compose_file or "")
        env_file = Path(config.compose_env_path or "")
        if (compose_file != source_root / "deploy" / "worker" / "compose.yaml"
                or not env_file.is_file() or env_file.is_symlink()):
            raise FleetRoleError("Worker Compose paths do not match the pinned source release")
        if not self.python_executable.is_absolute() or not self.python_executable.is_file():
            raise FleetRoleError("managed Python runtime is unavailable")
        state_dir = Path(config.state_dir)
        node_config = state_dir / "node.json"
        dfdaemon_config = Path(config.dragonfly_worker_peer_path or "")
        dynconfig = dfdaemon_config.parent / "dynconfig.yaml"
        log_dir = ensure_user_dir(self.role_dir / "dragonfly" / "logs", account=self.account)
        if not dfdaemon_config.is_file() or not dynconfig.is_file():
            raise FleetRoleError("Worker Dragonfly peer configuration is unavailable")
        if config.dragonfly_dfdaemon_path == "/usr/local/bin/dfdaemon":
            raise FleetRoleError("Fleet Worker cannot use a system Dragonfly binary")
        compose = access.command(
            "compose", "--project-name", project, "--file", str(compose_file),
            "--env-file", str(env_file),
        )
        compose_start = [*compose, "up", "--detach", "--no-build", "--pull", "never", "worker"]
        compose_stop = [*compose, "down", "--timeout", "60"]
        docker_environment: list[str] = []
        docker_config = os.environ.get("DOCKER_CONFIG")
        if docker_config:
            docker_config = str(Path(docker_config).expanduser().absolute())
            docker_environment.append(f"Environment=DOCKER_CONFIG={_systemd_value(docker_config)}")
        for name in ("DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
            value = os.environ.get(name)
            if value:
                docker_environment.append(f"Environment={name}={_systemd_value(value)}")
        node_unit = _unit_bytes([
            "[Unit]", "Description=Fleet-owned Qwen training node heartbeat reporter",
            "After=network-online.target", "Wants=network-online.target", "",
            "[Service]", "Type=simple", "UMask=0077", "NoNewPrivileges=yes",
            f"WorkingDirectory={_systemd_path(source_root)}",
            f"Environment=HOME={_systemd_value(self.account.home)}",
            f"Environment=PYTHONPATH={_systemd_value(source_root)}",
            "ExecStart=" + " ".join(_systemd_value(part) for part in [
                str(self.python_executable), "-m", "qwen_rl.cluster", "node", "run",
                "--config", str(node_config),
            ]),
            "Restart=on-failure", "RestartSec=5s", "",
            "[Install]", "WantedBy=default.target",
        ])
        dragonfly_unit = _unit_bytes([
            "[Unit]", "Description=Fleet-owned pinned Dragonfly Worker peer",
            "After=network-online.target", "Wants=network-online.target", "",
            "[Service]", "Type=simple", "UMask=0077", "NoNewPrivileges=yes",
            f"WorkingDirectory={_systemd_path(self.role_dir)}",
            f"Environment=HOME={_systemd_value(self.account.home)}",
            "ExecStart=" + " ".join(_systemd_value(part) for part in [
                config.dragonfly_dfdaemon_path, "--config", str(dfdaemon_config),
                "--dynconfig", str(dynconfig), "--log-dir", str(log_dir),
            ]),
            "Restart=on-failure", "RestartSec=5s", "",
            "[Install]", "WantedBy=default.target",
        ])
        worker_unit = _unit_bytes([
            "[Unit]", "Description=Fleet-owned pinned Qwen vLLM Worker container",
            f"Requires={_DRAGONFLY_UNIT}", f"After={_DRAGONFLY_UNIT}", "",
            "[Service]", "Type=oneshot", "RemainAfterExit=yes", "UMask=0077",
            "NoNewPrivileges=yes", f"WorkingDirectory={_systemd_path(source_root)}",
            f"Environment=HOME={_systemd_value(self.account.home)}",
            *docker_environment,
            "ExecStart=" + " ".join(_systemd_value(part) for part in compose_start),
            "ExecStartPre=" + " ".join(_systemd_value(part) for part in [
                "/usr/bin/test", "-d", config.model.cache_path,
            ]),
            "ExecStartPre=" + " ".join(_systemd_value(part) for part in [
                "/usr/bin/test", "-f", str(self.role_dir / "image-verified.json"),
            ]),
            "ExecStop=" + " ".join(_systemd_value(part) for part in compose_stop),
            "TimeoutStartSec=2h", "TimeoutStopSec=120s", "",
            "[Install]", "WantedBy=default.target",
        ])
        return {_NODE_UNIT: node_unit, _DRAGONFLY_UNIT: dragonfly_unit, _WORKER_UNIT: worker_unit}

    def _systemctl(self, *arguments: str, timeout: float = 120) -> subprocess.CompletedProcess:
        systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
        return run_owned([systemctl, "--user", *arguments], runner=self.runner, timeout=timeout)

    def _systemctl_active(self, unit: str) -> bool | None:
        systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
        result = self.runner([systemctl, "--user", "is-active", "--quiet", unit],
                             check=False, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return True
        if result.returncode == 3:
            return False
        return None

    def _activate_units(self, units: Mapping[str, bytes], *, start_worker: bool) -> None:
        ensure_user_dir(self.user_service_dir, account=self.account)
        for name, content in units.items():
            target = self.user_service_dir / name
            if target.exists() and target.is_symlink():
                raise FleetRoleError("Fleet Worker user-service path is a symlink")
            atomic_write(target, content, mode=0o600)
            info = target.stat()
            if info.st_uid != self.account.uid or info.st_gid != self.account.gid:
                raise FleetRoleError("Fleet Worker user-service is not owned by the registered account")
        self._systemctl("daemon-reload", timeout=30)
        self._systemctl("enable", "--now", _DRAGONFLY_UNIT)
        if not start_worker:
            return
        self._systemctl("enable", "--now", _NODE_UNIT)
        self._systemctl("enable", "--now", _WORKER_UNIT, timeout=7200)

    def _wait_for_socket(self, socket_path: Path, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                info = socket_path.lstat()
                if (stat.S_ISSOCK(info.st_mode) and info.st_uid == self.account.uid
                        and info.st_gid == self.account.gid):
                    return True
            except OSError:
                pass
            self.sleep(0.25)
        return False

    def _load_config(self, receipt: Mapping[str, Any]) -> tuple[Path, NodeConfiguration, dict[str, Any]]:
        config_path = Path(str(receipt.get("training_state_dir", self.training_state_dir))) / "node.json"
        if config_path.parent.resolve(strict=False) != self.training_state_dir.resolve(strict=False):
            raise FleetRoleError("Worker receipt points outside this account's role state")
        if not _node_config_owned(config_path, self.account):
            raise FleetRoleError("Worker node config is not a regular file owned by this account")
        try:
            config, credentials = load_node_config(config_path)
        except Exception as exc:
            raise FleetRoleError("Worker node state cannot be validated") from exc
        if config.role != self.role:
            raise FleetRoleError("training node state does not identify a Worker")
        private_paths = [
            config_path,
            config_path.parent / "credentials.json",
            config.worker_config_path,
            config.worker_token_path,
            config.compose_env_path,
        ]
        for raw_path in private_paths:
            if not raw_path:
                raise FleetRoleError("Worker private state path is missing")
            path = Path(raw_path)
            try:
                info = path.lstat()
            except OSError as exc:
                raise FleetRoleError("Worker private state file is unavailable") from exc
            if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != self.account.uid or info.st_gid != self.account.gid):
                raise FleetRoleError("Worker private state file has an unexpected owner or mode")
        for key, observed in (("training_node_id", config.node_id), ("controller_url", config.controller_url),
                              ("site_id", config.site_id), ("gpu_uuid", config.gpu_uuid)):
            if receipt.get(key) is not None and receipt.get(key) != observed:
                raise FleetRoleError(f"Worker {key.replace('_', ' ')} differs from its receipt")
        return config_path, config, credentials

    def _local_health(self, config_path: Path) -> tuple[dict[str, Any] | None, bool, str | None]:
        try:
            daemon = NodeDaemon(config_path)
            daemon.model_verified = True
            return asyncio.run(daemon._local_health())
        except Exception:  # noqa: BLE001 - readiness must fail closed on any local health error.
            return None, False, "worker_health_unavailable"

    def _record_model_receipt(self, config: NodeConfiguration, receipt: Mapping[str, Any]) -> dict[str, Any]:
        expected = {
            "artifact_id": config.model.artifact_id,
            "manifest_sha256": config.model.manifest_sha256,
            "base_fingerprint": config.model.base_fingerprint,
            "artifact_fingerprint": config.model.artifact_fingerprint,
            "path": config.model.cache_path,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise FleetRoleError("installed Worker model receipt differs from its pinned descriptor")
        path = Path(config.model.cache_path)
        try:
            info = path.lstat()
        except OSError as exc:
            raise FleetRoleError("verified Worker model cache path is unavailable") from exc
        if (not stat.S_ISDIR(info.st_mode) or path.is_symlink()
                or info.st_uid != self.account.uid or info.st_gid != self.account.gid):
            raise FleetRoleError("verified Worker model cache path is unavailable")
        record = {**expected, "verified": True}
        _write_json(self.role_dir / "model-verified.json", record)
        return record

    def _record_image_receipt(self, config: NodeConfiguration, receipt: Mapping[str, Any]) -> dict[str, Any]:
        if config.worker_image_mode == "archive":
            descriptor = config.image_archive
            expected = {
                "state": "loaded_verified", "image_id": descriptor.image_id if descriptor else None,
                "archive_sha256": descriptor.archive_sha256 if descriptor else None,
            }
        else:
            expected = {"state": "pulled", "worker_image": config.worker_image}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise FleetRoleError("installed Worker image receipt differs from its pinned descriptor")
        record = {**expected, "verified": True}
        if config.worker_release_selection is not None:
            selection = config.worker_release_selection
            release_record = {
                "release_identity": selection.release_identity.model_dump(mode="json"),
                "compute_capability": selection.compute_capability,
                "build_arch": selection.build_arch,
                "cuda_version": selection.cuda_version,
            }
            if receipt.get("worker_release_selection") != release_record:
                raise FleetRoleError("installed Worker image receipt differs from the selected release build")
            record["worker_release_selection"] = release_record
        _write_json(self.role_dir / "image-verified.json", record)
        return record

    def _training_drain(self, config: NodeConfiguration, credentials: Mapping[str, str]) -> bool:
        endpoint = f"{config.controller_url}/v1/nodes/{quote(config.node_id, safe='')}/heartbeat"
        payload = {
            "boot_id": config.boot_id,
            "state": "draining",
            "worker_health": None,
            "image_digest": (config.worker_image.split("@", 1)[1]
                             if "@" in config.worker_image else config.worker_image),
            "error": "fleet-role-transition",
        }
        try:
            with httpx.Client(verify=str(self.ca_bundle), timeout=15, follow_redirects=False,
                              transport=self.artifact_transport, trust_env=False) as client:
                response = client.post(endpoint, json=payload,
                                       headers={"Authorization": f"Bearer {credentials['node_token']}"})
        except httpx.HTTPError:
            return False
        return response.status_code in {200, 204}

    def _worker_health(self, config: NodeConfiguration, credentials: Mapping[str, str]) -> dict[str, Any] | None:
        if not config.worker_url:
            return None
        parsed = urlsplit(config.worker_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", config.lan_host} or parsed.port != 8241:
            return None
        try:
            with httpx.Client(timeout=5, follow_redirects=False, trust_env=False,
                              transport=self.artifact_transport) as client:
                response = client.get(config.worker_url.rstrip("/") + "/health",
                                      headers={"Authorization": f"Bearer {credentials['worker_token']}"})
        except httpx.HTTPError:
            return None
        if response.status_code != 200 or len(response.content) > 2_000_000:
            return None
        try:
            value = response.json()
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _worker_quiescent(health: Mapping[str, Any] | None) -> bool:
        if not isinstance(health, Mapping) or health.get("ready") is not True:
            return False
        return all(health.get(key) == 0 for key in ("active_generations", "pending_generations", "episode_refs"))

    def _wait_for_drain(self, config: NodeConfiguration,
                        credentials: Mapping[str, str]) -> bool:
        deadline = time.monotonic() + self.stop_timeout_seconds
        consecutive_empty = 0
        while time.monotonic() < deadline:
            health = self._worker_health(config, credentials)
            if health is None:
                return False
            if self._worker_quiescent(health):
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    return True
            else:
                consecutive_empty = 0
            self.sleep(0.5)
        return False

    def install(self, deployment: WorkerDeploymentSpec, *, generation: int,
                inventory: HostInventory, newly_managed: bool,
                owned_receipt: Mapping[str, Any] | None = None,
                install_intent: Mapping[str, Any] | None = None) -> RoleResult:
        base: dict[str, Any] = {
            "role": self.role,
            "generation": generation,
            "deployment_id": getattr(deployment, "deployment_id", None),
            "source_bundle_sha256": getattr(getattr(deployment, "source_bundle", None), "sha256", None),
        }
        if not isinstance(deployment, WorkerDeploymentSpec):
            return RoleResult("failed", base, "role_document_mismatch", "Worker adapter received another role", owned=False)
        if inventory.account != self.account or current_user_account() != self.account:
            return RoleResult("failed", base, "account_identity_mismatch",
                              "Worker inventory does not match the invoking account", owned=False)
        try:
            intent = dict(install_intent or self.prepare_install(
                deployment, generation=generation, owned_receipt=owned_receipt,
            ))
        except FleetRoleError as exc:
            return RoleResult("failed", base, "worker_ownership_preflight_failed", str(exc), owned=False)
        if any(intent.get(key) != value for key, value in {
            "role": self.role, "generation": generation, "deployment_id": deployment.deployment_id,
            "fleet_node_id": self._fleet_node_id(), "account_uid": self.account.uid,
            "account_gid": self.account.gid, "controller_url": deployment.controller_url,
            "site_id": deployment.site_id, "gpu_uuid": deployment.gpu_uuid,
        }.items()):
            return RoleResult("failed", base, "worker_install_intent_mismatch",
                              "Worker intent differs from the approved assignment", owned=False)
        if owned_receipt is not None and owned_receipt.get("deployment_id") not in {None, deployment.deployment_id}:
            return RoleResult("failed", base, "worker_state_identity_conflict",
                              "existing Worker state belongs to another Fleet deployment", owned=False)
        try:
            selected_gpu = select_one_gpu(inventory.gpus, deployment.gpu_uuid)
        except FleetHostError as exc:
            if not inventory.gpus and host_has_nvidia_device(runner=self.runner) is not False:
                return RoleResult("needs_driver", {**base, "gpu_inventory": "unavailable"},
                                  "gpu_not_ready", "NVIDIA device identity is not available")
            return RoleResult("failed", base, "gpu_selection_required", str(exc))
        try:
            operation_path, operation = self._operation_record(generation, deployment.deployment_id)
            prior_ownership = self._load_ownership()
            for path in (self.paths.root, self.paths.management_dir, self.paths.runtime_dir,
                         self.paths.trust_dir, self.paths.state_dir, self.paths.roles_dir,
                         self.paths.systemd_user_dir):
                ensure_user_dir(path, account=self.account)
            role_dir = ensure_user_dir(self.role_dir, account=self.account)
            state_dir = ensure_user_dir(self.training_state_dir, account=self.account)
            role_source_dir = ensure_user_dir(role_dir / "sources", account=self.account)
            current_access = self._docker_access(intent.get("docker"))
            compose_receipt = ensure_compose_plugin(
                current_access, account=self.account, runner=self.runner,
            )
            operation = self._record_step(
                operation_path.parent / _JOURNAL, operation, "compose_verified",
                compose_plugin=compose_receipt,
            )
            project = str(intent["compose_project"])
            source_archive = operation_path.parent / "worker-source.tar.gz"
        except ComposePluginError as exc:
            return RoleResult("failed", {**base, **intent}, "worker_compose_plugin_unavailable",
                              str(exc)[:400], owned=True)
        except (FleetRoleError, OSError, ValueError) as exc:
            return RoleResult("failed", {**base, **intent}, "worker_prepare_failed",
                              str(exc)[:400], owned=True)
        try:
            source_archive = download_verified_artifact(
                deployment.source_bundle, source_archive, ca_bundle=self.ca_bundle,
                max_bytes=_SOURCE_BUNDLE_MAX_BYTES, transport=self.artifact_transport,
            )
            source_root = role_source_dir / deployment.source_bundle.sha256
            source_receipt = _extract_source_bundle(source_archive, source_root, deployment.source_bundle.sha256)
            operation = self._record_step(
                operation_path.parent / _JOURNAL, operation, "source_verified",
                source_bundle_sha256=deployment.source_bundle.sha256,
                source_root=str(source_root), source_receipt=source_receipt,
            )
            try:
                selected_gpu = next(gpu for gpu in inventory.gpus if gpu.uuid == deployment.gpu_uuid)
            except StopIteration as exc:
                raise FleetRoleError("assigned GPU disappeared before Worker enrollment") from exc
            project_root = source_root
            compose_file = project_root / "deploy" / "worker" / "compose.yaml"
            if compose_file.is_symlink() or not compose_file.is_file():
                raise FleetRoleError("pinned Worker source bundle has no Compose definition")
            package_name = "dragonfly-client-v1.5.7-x86_64-unknown-linux-musl.tar.gz"
            dragonfly_archive = project_root / "assets" / package_name
            if dragonfly_archive.is_symlink() or not dragonfly_archive.is_file():
                raise FleetRoleError("pinned Worker source bundle omits the Dragonfly client package")
            dragonfly_dir = ensure_user_dir(role_dir / "dragonfly", account=self.account)
            binary_dir = ensure_user_dir(dragonfly_dir / "bin", account=self.account)
            from ..host import install_dragonfly_binaries

            binaries = install_dragonfly_binaries(
                bin_dir=binary_dir, source_archive=dragonfly_archive, runner=self.runner,
            )
            if set(binaries) != {"dfget", "dfdaemon"}:
                raise FleetRoleError("pinned Dragonfly package did not produce both Worker binaries")
            runtime_dir = ensure_user_dir(self.paths.runtime_dir / "dragonfly", account=self.account)
            cache_dir = ensure_user_dir(role_dir / "model-cache", account=self.account)
            config_path = state_dir / "node.json"
            if config_path.is_file() and prior_ownership:
                config, credentials = load_node_config(config_path)
                if (config.controller_url != deployment.controller_url or config.site_id != deployment.site_id
                        or (config.gpu_uuid != deployment.gpu_uuid
                            and prior_ownership.get("lifecycle_state") != "stopped")):
                    raise FleetRoleError("existing Worker training registration conflicts with this deployment")
            secret = deployment.worker_join_code.get_secret_value()
            expected_inside_uid = 0 if current_access.rootless else self.account.uid
            expected_inside_gid = 0 if current_access.rootless else self.account.gid
            from .user_ssh import create_asyncssh_ed25519_keypair

            config = asyncio.run(join_cluster(
                controller_url=deployment.controller_url,
                site_id=deployment.site_id,
                code=secret,
                role="worker",
                state_dir=state_dir,
                lan_host=inventory.lan_host,
                gpu_uuid=selected_gpu.uuid,
                ca_bundle=self.ca_bundle,
                boot_id=inventory.boot_id,
                gpu_inventory=(selected_gpu,),
                model_cache_path=cache_dir,
                worker_compose_file=compose_file,
                container_uid=expected_inside_uid,
                container_gid=expected_inside_gid,
                owner_uid=self.account.uid,
                owner_gid=self.account.gid,
                dragonfly_dfget_path=Path(binaries["dfget"]),
                dragonfly_dfdaemon_path=Path(binaries["dfdaemon"]),
                dragonfly_socket_path=runtime_dir / "dfdaemon.sock",
                write_training_ssh_state=False,
                key_creator=create_asyncssh_ed25519_keypair,
            ))
            config_path, credentials = load_node_config(config_path)
            fleet_node_id = str(intent["fleet_node_id"])
            ownership = self._persist_ownership(
                fleet_node_id=fleet_node_id,
                deployment_id=deployment.deployment_id,
                generation=generation,
                config=config,
                docker=current_access.receipt(),
                project=project,
                source_root=project_root,
                source_bundle_sha256=deployment.source_bundle.sha256,
                user_units=None,
                previous_deployment_id=(prior_ownership.get("deployment_id") if prior_ownership else None),
                compose_plugin=compose_receipt,
            )
            compose_env = _write_compose_environment(
                config, fleet_node_id=fleet_node_id, deployment_id=deployment.deployment_id,
                account=self.account, paths=self.paths, project=project,
            )
            access_receipt = current_access.receipt()
            source_model = config.model.model_dump()
            config_receipt = {
                "training_node_id": config.node_id,
                "training_state_dir": str(state_dir),
                "controller_url": config.controller_url,
                "site_id": config.site_id,
                "gpu_uuid": config.gpu_uuid,
                "worker_image": config.worker_image,
                "model": source_model,
                "compose_env_path": str(compose_env),
            }
            units = self._user_units(config, project_root, project, current_access)
            ownership = self._persist_ownership(
                fleet_node_id=fleet_node_id,
                deployment_id=deployment.deployment_id,
                generation=generation,
                config=config,
                docker=access_receipt,
                project=project,
                source_root=project_root,
                source_bundle_sha256=deployment.source_bundle.sha256,
                user_units=units,
                previous_deployment_id=(prior_ownership.get("deployment_id") if prior_ownership else None),
                compose_plugin=compose_receipt,
            )
            operation = self._record_step(
                operation_path.parent / _JOURNAL, operation, "node_enrolled",
                **config_receipt, ownership_sha256=hashlib.sha256(_json_bytes(ownership)).hexdigest(),
            )
            self._activate_units(units, start_worker=False)
            ownership = self._mark_ownership_active(ownership)
            socket_path = Path(config.dragonfly_socket_path)
            if not self._wait_for_socket(socket_path):
                return RoleResult("loading", {**intent, **ownership, "dragonfly_socket_ready": False},
                                  "dragonfly_not_ready", "User Dragonfly service has not opened its runtime socket")
            operation = self._record_step(operation_path.parent / _JOURNAL, operation, "dragonfly_ready",
                                          dragonfly_socket=str(socket_path))
            model_raw = asyncio.run(install_node_model_artifact(
                config, worker_token=credentials["worker_token"],
            ))
            model_receipt = self._record_model_receipt(config, model_raw)
            operation = self._record_step(operation_path.parent / _JOURNAL, operation, "model_verified",
                                          model_receipt=model_receipt)
            image_raw = asyncio.run(install_node_worker_image(
                config,
                worker_token=credentials["worker_token"],
                docker_binary=current_access.executable,
                docker_args_prefix=tuple(current_access.command()[1:]),
                command_runner=self.runner,
            ))
            image_receipt = self._record_image_receipt(config, image_raw)
            mapping_probe = probe_docker_uid_mapping(
                current_access,
                image=config.worker_image,
                account=self.account,
                probe_dir=operation_path.parent / "uid-map-probe",
                python_executable="python3",
                runner=self.runner,
                expected={
                    "container_uid": config.worker_uid,
                    "container_gid": config.worker_gid,
                    "probe_image": config.worker_image,
                    "probe_python": "python3",
                },
            )
            _write_json(self.role_dir / "docker-mapping-verified.json", mapping_probe)
            ownership = {**ownership, "docker_mapping": mapping_probe}
            _write_json(self.ownership_path, ownership)
            operation = self._record_step(operation_path.parent / _JOURNAL, operation, "image_and_uid_verified",
                                          image_receipt=image_receipt, docker_mapping=mapping_probe)
            self._activate_units(units, start_worker=False)
            self._systemctl("enable", "--now", _NODE_UNIT)
            self._systemctl("enable", "--now", _WORKER_UNIT, timeout=7200)
            operation = self._record_step(operation_path.parent / _JOURNAL, operation, "worker_started")
            daemon = NodeDaemon(config_path)
            heartbeat = asyncio.run(daemon.heartbeat_once())
            if heartbeat.get("node_id") not in {None, config.node_id} or heartbeat.get("state") not in {None, "ready"}:
                raise FleetRoleError("training Controller did not confirm the Worker ready heartbeat")
            heartbeat_receipt = {
                "accepted": True,
                "state": heartbeat.get("state", "ready"),
                "node_id": heartbeat.get("node_id", config.node_id),
            }
            operation = self._record_step(operation_path.parent / _JOURNAL, operation,
                                          "heartbeat_confirmed", node_heartbeat=heartbeat_receipt)
            result = self.inspect({
                **intent,
                **ownership,
                "docker_mapping": mapping_probe,
                "model_receipt": model_receipt,
                "image_receipt": image_receipt,
                "node_heartbeat": heartbeat_receipt,
            })
            if result.status == "ready":
                self._record_step(operation_path.parent / _JOURNAL, operation, "ready",
                                  readiness_receipt=result.receipt)
            return result
        except FleetRoleError as exc:
            return RoleResult("failed", {**base, **intent}, "worker_install_failed", str(exc), owned=True)
        except Exception as exc:  # noqa: BLE001 - never echo provider or credential-bearing exceptions.
            return RoleResult("failed", {**base, **intent}, "worker_install_failed",
                              f"Worker install did not complete ({type(exc).__name__})", owned=True)

    def inspect(self, receipt: Mapping[str, Any]) -> RoleResult:
        safe = dict(receipt)
        if safe.get("role") != self.role:
            return RoleResult("failed", safe, "role_receipt_mismatch", "Worker receipt does not match the recorded role")
        try:
            generation = safe.get("generation")
            deployment_id_value = safe.get("deployment_id")
            journal: dict[str, Any] | None = None
            if (type(generation) is int and generation >= 1 and isinstance(deployment_id_value, str)):
                journal_path = self.state_dir / "operations" / f"generation-{generation}" / _JOURNAL
                journal = _read_json(journal_path)
                if journal is not None and (journal.get("role") != self.role
                                            or journal.get("generation") != generation
                                            or journal.get("deployment_id") != deployment_id_value):
                    return RoleResult("failed", safe, "worker_journal_mismatch",
                                      "Worker durable operation record belongs to another assignment")
            node_id = self._fleet_node_id()
            deployment_id = str(safe.get("deployment_id", ""))
            ownership = self._check_ownership(deployment_id, node_id, safe)
            if ownership is None:
                return RoleResult("loading", safe, "worker_state_pending", "Worker ownership record is not ready")
            access = self._docker_access(ownership.get("docker"))
            mapping = (safe.get("docker_mapping") or ownership.get("docker_mapping")
                       or (journal.get("docker_mapping") if journal else None))
            if not isinstance(mapping, Mapping) or mapping.get("uid_mapping_verified") is not True:
                mapping = _read_json(self.role_dir / "docker-mapping-verified.json")
            if not isinstance(mapping, Mapping) or mapping.get("uid_mapping_verified") is not True:
                return RoleResult("loading", {**safe, **ownership, "docker": access.receipt()},
                                  "docker_uid_mapping_pending", "Docker bind-mount UID mapping has not been verified")
            config_path, config, _credentials = self._load_config({**ownership, **safe})
            unit_statuses = {unit: self._systemctl_active(unit) for unit in _WORKER_UNITS}
            if any(active is None for active in unit_statuses.values()):
                return RoleResult("loading", {**safe, **ownership, "user_service_status": unit_statuses},
                                  "user_service_status_unknown", "Fleet user-service status cannot be established")
            if not all(unit_statuses.values()):
                return RoleResult("loading", {**safe, **ownership, "user_service_status": unit_statuses},
                                  "user_service_not_active", "Fleet Worker user services are not all active")
            if not self._project_is_owned(access, ownership["compose_project"], node_id,
                                          deployment_id, allow_absent=False):
                return RoleResult("loading", {**safe, **ownership}, "worker_container_ownership_unknown",
                                  "Fleet Worker container ownership could not be established")
            model = _read_json(self.role_dir / "model-verified.json")
            image = _read_json(self.role_dir / "image-verified.json")
            expected_model = {
                "artifact_id": config.model.artifact_id,
                "manifest_sha256": config.model.manifest_sha256,
                "base_fingerprint": config.model.base_fingerprint,
                "artifact_fingerprint": config.model.artifact_fingerprint,
                "path": config.model.cache_path,
                "verified": True,
            }
            if not model or any(model.get(key) != value for key, value in expected_model.items()):
                return RoleResult("loading", {**safe, **ownership}, "model_receipt_missing",
                                  "Pinned model verification receipt is unavailable")
            if not image or image.get("verified") is not True:
                return RoleResult("loading", {**safe, **ownership}, "image_receipt_missing",
                                  "Pinned Worker image verification receipt is unavailable")
            if config.worker_image_mode == "archive":
                image_expected = {
                    "state": "loaded_verified",
                    "image_id": config.image_archive.image_id if config.image_archive else None,
                    "archive_sha256": config.image_archive.archive_sha256 if config.image_archive else None,
                    "verified": True,
                }
            else:
                image_expected = {"state": "pulled", "worker_image": config.worker_image, "verified": True}
            if any(image.get(key) != value for key, value in image_expected.items()):
                return RoleResult("loading", {**safe, **ownership}, "image_receipt_mismatch",
                                  "Pinned Worker image verification receipt differs from node config")
            health, ready, error = self._local_health(config_path)
            if not ready:
                return RoleResult("loading", {**safe, **ownership, "worker_health": health},
                                  error or "worker_not_ready", "Worker model and local health receipt are not ready")
            mapping_value = dict(mapping)
            if mapping_value.get("uid_mapping_verified") is not True:
                return RoleResult("loading", {**safe, **ownership}, "docker_uid_mapping_pending",
                                  "Docker UID/GID mapping has not been verified")
            heartbeat = safe.get("node_heartbeat") or (journal.get("node_heartbeat") if journal else None)
            if not isinstance(heartbeat, Mapping) or heartbeat.get("accepted") is not True:
                return RoleResult("loading", {**safe, **ownership}, "node_heartbeat_pending",
                                  "Training Controller ready heartbeat has not been confirmed")
            ready_receipt = {
                **ownership,
                "artifact_sha256": ownership.get("artifact_sha256"),
                "account_uid": self.account.uid,
                "account_gid": self.account.gid,
                "readiness_verified": True,
                "docker": access.receipt(),
                "docker_mapping": mapping_value,
                "docker_access": {**access.receipt(), **mapping_value},
                "user_service_units": list(_WORKER_UNITS),
                "user_services": [
                    {"unit": unit, "path": str(self.user_service_dir / unit), "active": True}
                    for unit in _WORKER_UNITS
                ],
                "user_units_active": True,
                "model_verified": True,
                "image_verified": True,
                "worker_health": health,
                "worker_ready": True,
                "node_heartbeat": dict(heartbeat),
                "target_state": "owned",
            }
            return RoleResult("ready", ready_receipt)
        except FleetRoleError as exc:
            return RoleResult("loading", safe, "worker_state_pending", str(exc))
        except Exception:  # noqa: BLE001 - readiness must fail closed without exposing local secrets.
            return RoleResult("loading", safe, "worker_health_unavailable", "Worker readiness cannot be established")

    def stop_owned(self, receipt: Mapping[str, Any]) -> StopResult:
        safe = dict(receipt)
        if safe.get("role") != self.role:
            return StopResult("failed", {}, "role_receipt_mismatch", "Worker receipt does not match the recorded role")
        try:
            node_id = self._fleet_node_id()
            ownership = self._check_ownership(str(safe.get("deployment_id", "")), node_id, safe)
        except FleetRoleError as exc:
            return StopResult("blocked_busy", safe, "worker_ownership_unknown", str(exc))
        if ownership is None:
            try:
                access = self._docker_access()
                project = _project_name(node_id)
                containers = self._docker_project_containers(access, project)
                networks = self._docker_project_networks(access, project)
            except FleetRoleError as exc:
                return StopResult("blocked_busy", safe, "worker_ownership_unknown", str(exc))
            if containers is None or networks is None or containers or networks:
                return StopResult("blocked_busy", safe, "worker_ownership_unknown",
                                  "Docker project resources exist without a Fleet Worker ownership record")
            return StopResult("stopped", {"role": self.role, "deployment_id": safe.get("deployment_id"),
                                           "preserved_state": True, "no_role_artifacts": True})
        try:
            access = self._docker_access(ownership.get("docker"))
            _config_path, config, credentials = self._load_config({**ownership, **safe})
        except FleetRoleError as exc:
            return StopResult("blocked_busy", safe, "worker_ownership_unknown", str(exc))
        node_unit_state = self._systemctl_active(_NODE_UNIT)
        if node_unit_state is None:
            return StopResult("blocked_busy", safe, "worker_reporter_status_unknown",
                              "Could not prove the Fleet node reporter state")
        if node_unit_state:
            stopped = self.runner(
                [shutil.which("systemctl") or "/usr/bin/systemctl", "--user", "stop", _NODE_UNIT],
                check=False, capture_output=True, text=True, timeout=60,
            )
            if stopped.returncode != 0:
                return StopResult("blocked_busy", safe, "worker_reporter_stop_failed",
                                  "Could not stop the owned Fleet node reporter")
        if not self._training_drain(config, credentials):
            return StopResult("blocked_busy", safe, "training_registry_unreachable",
                              "Could not fence new Worker leases in the training Controller")
        containers = self._docker_project_containers(access, ownership["compose_project"])
        if containers is None:
            return StopResult("blocked_busy", safe, "worker_container_state_unknown",
                              "Could not establish whether the owned Worker container is running")
        networks = self._docker_project_networks(access, ownership["compose_project"])
        if networks is None:
            return StopResult("blocked_busy", safe, "worker_network_state_unknown",
                              "Could not establish whether the owned Worker network exists")
        if ((containers or networks)
                and not self._project_is_owned(access, ownership["compose_project"], node_id,
                                               str(ownership["deployment_id"]), allow_absent=False)):
            return StopResult("blocked_busy", safe, "worker_resource_ownership_unknown",
                              "Docker project includes resources outside the recorded Worker ownership")
        if containers and not self._wait_for_drain(config, credentials):
            return StopResult("blocked_busy", safe, "worker_requests_busy",
                              "Worker still has active generations or episode references")
        compose_file = Path(ownership["compose_file"])
        env_file = Path(config.compose_env_path or "")
        source_root = Path(ownership["source_root"])
        if (compose_file.resolve(strict=False) != (source_root / "deploy" / "worker" / "compose.yaml").resolve(strict=False)
                or not env_file.is_file() or env_file.is_symlink()):
            return StopResult("blocked_busy", safe, "worker_compose_ownership_unknown",
                              "Worker Compose files do not match the owned source release")
        worker_unit_active = self._systemctl_active(_WORKER_UNIT)
        if worker_unit_active is None:
            return StopResult("blocked_busy", safe, "worker_user_service_status_unknown",
                              "Could not establish the Fleet Worker service state")
        if worker_unit_active:
            try:
                self._systemctl("disable", "--now", _WORKER_UNIT, timeout=120)
            except FleetRoleError:
                return StopResult("failed", safe, "worker_user_service_stop_failed",
                                  "Owned Fleet Worker service shutdown did not complete")
        remaining_before_fallback = self._docker_project_containers(access, ownership["compose_project"])
        if remaining_before_fallback is None:
            return StopResult("blocked_busy", safe, "worker_container_state_unknown",
                              "Could not establish whether the owned Worker container stopped")
        remaining_networks = self._docker_project_networks(access, ownership["compose_project"])
        if remaining_networks is None:
            return StopResult("blocked_busy", safe, "worker_network_state_unknown",
                              "Could not establish whether the owned Worker network stopped")
        if remaining_before_fallback or remaining_networks:
            if not self._project_is_owned(access, ownership["compose_project"], node_id,
                                          str(ownership["deployment_id"]), allow_absent=False):
                return StopResult("blocked_busy", safe, "worker_container_ownership_unknown",
                                  "Remaining Docker resources are outside the recorded Worker ownership")
            argv = access.command("compose", "--project-name", ownership["compose_project"],
                                  "--file", str(compose_file), "--env-file", str(env_file),
                                  "down", "--timeout", "60")
            try:
                run_owned(argv, runner=self.runner, timeout=120)
            except FleetRoleError:
                return StopResult("failed", safe, "worker_compose_stop_failed",
                                  "Owned Worker Compose shutdown did not complete")
        remaining = self._docker_project_containers(access, ownership["compose_project"])
        remaining_networks = self._docker_project_networks(access, ownership["compose_project"])
        if remaining is None or remaining_networks is None or remaining or remaining_networks:
            return StopResult("blocked_busy", safe, "worker_container_stop_unknown",
                              "Could not prove that the owned Worker project resources stopped")
        for unit in (_WORKER_UNIT, _DRAGONFLY_UNIT, _NODE_UNIT):
            status = self._systemctl_active(unit)
            if status is None:
                return StopResult("blocked_busy", safe, "worker_user_service_status_unknown",
                                  "Could not establish an owned Worker user-service state")
            if (self.user_service_dir / unit).is_file():
                try:
                    self._systemctl("disable", "--now", unit, timeout=90)
                except FleetRoleError:
                    return StopResult("failed", safe, "worker_user_service_stop_failed",
                                      "Owned Worker user-service shutdown did not complete")
        stopped_ownership = {**ownership, "lifecycle_state": "stopped", "stopped_at": time.time()}
        _write_json(self.ownership_path, stopped_ownership)
        return StopResult("stopped", {
            "role": self.role,
            "fleet_node_id": node_id,
            "deployment_id": ownership["deployment_id"],
            "training_node_id": config.node_id,
            "training_state_dir": str(self.training_state_dir),
            "preserved_state": True,
            "stopped_user_services": list(_WORKER_UNITS),
            "stopped_compose_project": ownership["compose_project"],
        })
