"""Pure planner for a single-instance, process-owned NInfer deployment."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from ..runtime.config import RuntimeConfig, load_config
from .local_gradients import _compute_profile
from .workers import _auth_environment_name

_SCHEMA = "qwen-rl-instance/v1"
_DEFAULT_GATEWAY_PORT = 8310
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOP_KEYS = {
    "schema", "runtime_config", "state_dir", "gpus", "engine_binary", "worker_python",
    "python", "start_timeout", "stop_timeout", "gateway_config", "credential_files",
    "teacher_base_url", "dashboard_port", "dashboard_host", "cache_root", "packed_cache_root",
}


class InstanceDeploymentError(ValueError):
    """The instance profile or its local runtime topology is unsupported."""


@dataclass(frozen=True)
class InstanceDeployment:
    path: Path
    runtime_config: Path
    state_dir: Path
    gpus: tuple[int, ...]
    engine_binary: str
    worker_python: str
    python: str = field(default_factory=lambda: sys.executable)
    start_timeout: float = 1800.0
    stop_timeout: float = 1200.0
    gateway_config: Path | None = None
    credential_files: Mapping[str, Path] = field(default_factory=dict)
    teacher_base_url: str | None = None
    dashboard_port: int | None = None
    dashboard_host: str = "0.0.0.0"
    cache_root: Path | None = None
    packed_cache_root: Path | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "path": str(self.path),
            "runtime_config": str(self.runtime_config),
            "state_dir": str(self.state_dir),
            "gpus": list(self.gpus),
            "engine_binary": self.engine_binary,
            "worker_python": self.worker_python,
            "python": self.python,
            "start_timeout": self.start_timeout,
            "stop_timeout": self.stop_timeout,
            "gateway_config": str(self.gateway_config) if self.gateway_config is not None else None,
            "credential_files": {name: str(path) for name, path in self.credential_files.items()},
            "teacher_base_url": self.teacher_base_url,
            "dashboard_port": self.dashboard_port,
            "dashboard_host": self.dashboard_host,
            **({"cache_root": str(self.cache_root)} if self.cache_root is not None else {}),
            **({"packed_cache_root": str(self.packed_cache_root)}
               if self.packed_cache_root is not None else {}),
        }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InstanceDeploymentError(f"duplicate deployment key: {key}")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise InstanceDeploymentError("deployment JSON contains a non-finite number")


def _path(value: Any, *, base: Path, name: str, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise InstanceDeploymentError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        path = path.resolve(strict=must_exist)
    except OSError:
        raise InstanceDeploymentError(f"{name} path is unavailable") from None
    if must_exist and not path.is_file():
        raise InstanceDeploymentError(f"{name} must name a regular file")
    return path


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise InstanceDeploymentError(f"{name} must be a non-empty string")
    return value


def _positive_number(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise InstanceDeploymentError(f"{name} must be a finite positive number")
    return float(value)


def load_deployment(path: str | Path) -> InstanceDeployment:
    """Load a strict instance JSON; relative paths are anchored to that JSON."""
    source = Path(path).expanduser().resolve(strict=True)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"),
                         object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except InstanceDeploymentError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise InstanceDeploymentError("instance deployment JSON cannot be read") from None
    if not isinstance(raw, dict) or set(raw) - _TOP_KEYS:
        raise InstanceDeploymentError("instance deployment fields do not match the strict schema")
    if raw.get("schema") != _SCHEMA:
        raise InstanceDeploymentError("instance deployment schema is unsupported")
    required = {"runtime_config", "state_dir", "gpus", "engine_binary", "worker_python"}
    if required - set(raw):
        raise InstanceDeploymentError("instance deployment is missing required fields")

    gpu_values = raw["gpus"]
    if (not isinstance(gpu_values, list) or not gpu_values
            or any(type(value) is not int or value < 0 for value in gpu_values)
            or len(set(gpu_values)) != len(gpu_values)):
        raise InstanceDeploymentError("gpus must be a non-empty list of unique nonnegative integers")

    base = source.parent
    credential_data = raw.get("credential_files", {})
    if not isinstance(credential_data, dict):
        raise InstanceDeploymentError("credential_files must be an environment-name to path mapping")
    credential_files: dict[str, Path] = {}
    for name, value in credential_data.items():
        if not isinstance(name, str) or _ENV_NAME.fullmatch(name) is None:
            raise InstanceDeploymentError("credential_files keys must be valid environment variable names")
        credential_files[name] = _path(value, base=base, name=f"credential_files.{name}", must_exist=False)

    gateway_value = raw.get("gateway_config")
    gateway_config = (_path(gateway_value, base=base, name="gateway_config", must_exist=True)
                      if gateway_value is not None else None)
    teacher_base_url = raw.get("teacher_base_url")
    if teacher_base_url is not None:
        teacher_base_url = _string(teacher_base_url, "teacher_base_url")
    dashboard_port = raw.get("dashboard_port")
    if dashboard_port is not None and (type(dashboard_port) is not int or not 1 <= dashboard_port <= 65535):
        raise InstanceDeploymentError("dashboard_port must be a valid TCP port or null")
    dashboard_host = raw.get("dashboard_host", "0.0.0.0")
    if dashboard_host not in {"0.0.0.0", "127.0.0.1", "::", "::1"}:
        raise InstanceDeploymentError("dashboard_host must be an explicit bind address")
    def executable(value: Any, name: str) -> str:
        configured = Path(_string(value, name)).expanduser()
        # Preserve lexical virtualenv symlinks while fixing the configuration
        # base once, before the supervisor changes its working directory.
        return os.path.abspath(configured if configured.is_absolute() else base / configured)

    python = executable(raw.get("python", sys.executable), "python")
    return InstanceDeployment(
        path=source,
        runtime_config=_path(raw["runtime_config"], base=base, name="runtime_config", must_exist=True),
        state_dir=_path(raw["state_dir"], base=base, name="state_dir", must_exist=False),
        gpus=tuple(gpu_values),
        engine_binary=executable(raw["engine_binary"], "engine_binary"),
        worker_python=executable(raw["worker_python"], "worker_python"),
        python=python,
        start_timeout=_positive_number(raw.get("start_timeout"), "start_timeout", 1800.0),
        stop_timeout=_positive_number(raw.get("stop_timeout"), "stop_timeout", 1200.0),
        gateway_config=gateway_config,
        credential_files=credential_files,
        teacher_base_url=teacher_base_url,
        dashboard_port=dashboard_port, dashboard_host=dashboard_host,
        cache_root=(_path(raw["cache_root"], base=base, name="cache_root", must_exist=False)
                    if raw.get("cache_root") is not None else None),
        packed_cache_root=(_path(raw["packed_cache_root"], base=base,
                                 name="packed_cache_root", must_exist=False)
                           if raw.get("packed_cache_root") is not None else None),
    )


def _default_packed_cache_root(deployment: InstanceDeployment) -> Path:
    """Return a stable tmpfs namespace for this deployment's rebuildable packed weights."""
    state_name = re.sub(r"[^A-Za-z0-9._-]+", "-", deployment.state_dir.name).strip(".-") or "instance"
    scope = hashlib.sha256(str(deployment.state_dir).encode("utf-8")).hexdigest()[:12]
    return Path("/dev/shm/qwen-rl-cache") / f"{state_name}-{scope}" / "packed-frozen-base"


def _gpu_inventory(inventory: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if inventory is None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,compute_cap", "--format=csv,noheader,nounits"],
                check=False, capture_output=True, text=True, timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise InstanceDeploymentError("cannot query local GPU inventory with nvidia-smi") from None
        if result.returncode != 0:
            raise InstanceDeploymentError("local GPU inventory query failed")
        inventory = []
        for line in result.stdout.splitlines():
            pieces = [part.strip() for part in line.split(",")]
            if len(pieces) != 3:
                raise InstanceDeploymentError("local GPU inventory is malformed")
            try:
                index = int(pieces[0])
            except ValueError:
                raise InstanceDeploymentError("local GPU inventory has an invalid index") from None
            inventory.append({"index": index, "uuid": pieces[1], "compute_capability": pieces[2] or None})
    if not isinstance(inventory, list):
        raise InstanceDeploymentError("GPU inventory must be a list")
    result: dict[int, dict[str, Any]] = {}
    uuids: set[str] = set()
    for row in inventory:
        if not isinstance(row, dict):
            raise InstanceDeploymentError("GPU inventory rows must be objects")
        index, uuid = row.get("index"), row.get("uuid")
        capability = row.get("compute_capability")
        if (type(index) is not int or index < 0 or not isinstance(uuid, str)
                or not uuid.startswith("GPU-") or index in result or uuid in uuids
                or (capability is not None and not isinstance(capability, str))):
            raise InstanceDeploymentError("GPU inventory contains an invalid or duplicate device identity")
        result[index] = {"index": index, "uuid": uuid, "compute_capability": capability}
        uuids.add(uuid)
    if not result:
        raise InstanceDeploymentError("GPU inventory is empty")
    return result


def _load_runtime(deployment: InstanceDeployment) -> RuntimeConfig:
    try:
        runtime = load_config(deployment.runtime_config, for_local_worker_plan=True)
    except Exception as exc:
        raise InstanceDeploymentError(f"source runtime profile is invalid ({type(exc).__name__})") from exc
    if (runtime.model_backend != "ternary" or runtime.worker_inference.backend != "ninfer"
            or runtime.worker_inference.quantization != "ternary"
            or runtime.learner.update_backend != "central_gradient"):
        raise InstanceDeploymentError("instance deployment requires explicit ternary/ninfer/central_gradient")
    if runtime.learner.recovery_mode != "auto" or runtime.learner.recovery_lineage_id is not None:
        raise InstanceDeploymentError("instance deployment cannot reset or use fresh recovery")
    if not runtime.learner.groups_per_update_auto:
        raise InstanceDeploymentError("source runtime profile must use groups_per_update=auto")
    workers = list(runtime.workers)
    helper_ids = set(runtime.rollout.helper_worker_ids)
    if not workers or any(worker.scope not in {"local", "service", "gateway"} for worker in workers):
        raise InstanceDeploymentError("instance deployment v1 supports local Worker owners and Gateway helpers")
    worker_ids = {worker.id for worker in workers}
    owner_ids = set(runtime.learner.gradient_executor_worker_ids.values())
    source_owner_ids = worker_ids - helper_ids
    if any(worker.scope == "gateway" for worker in workers if worker.id not in helper_ids):
        raise InstanceDeploymentError("managed instance gradient owners must be local or service Workers")
    learner = runtime.learner
    if (owner_ids != source_owner_ids
            or set(learner.gradient_local_executor_worker_ids) != source_owner_ids
            or set(learner.gradient_worker_sync_ids) != source_owner_ids):
        raise InstanceDeploymentError("every source owner Worker must have an explicit local executor and sync owner")
    helper_auth_envs = {worker.auth_token_env for worker in workers if worker.id in helper_ids}
    remote_auth_envs = {learner.gradient_executor_token_envs[e]
                        for e in learner.gradient_remote_executor_ids}
    allowed_credentials = {"ARC_ENV_TOKEN", *helper_auth_envs, *remote_auth_envs}
    if runtime.bepa.enabled:
        allowed_credentials.update((runtime.bepa.teacher.api_key_env, runtime.bepa.teacher.token_env))
    if not set(deployment.credential_files) <= allowed_credentials:
        raise InstanceDeploymentError("credential_files may name only ARC_ENV_TOKEN and configured BEPA credentials")
    return runtime


def _engine_base_port(runtime: RuntimeConfig) -> int:
    value = runtime.worker_inference.ninfer_url
    try:
        parsed = urlsplit(value or "")
        port = parsed.port
    except ValueError:
        raise InstanceDeploymentError("source NInfer URL must be a loopback HTTP origin with an explicit port") from None
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or port is None:
        raise InstanceDeploymentError("source NInfer URL must be a loopback HTTP origin with an explicit port")
    return port


def _validate_gateway_port(deployment: InstanceDeployment, ports: set[int]) -> None:
    if deployment.gateway_config is None:
        return
    if _DEFAULT_GATEWAY_PORT in ports:
        raise InstanceDeploymentError(
            f"Gateway default port {_DEFAULT_GATEWAY_PORT} conflicts with a planned Learner or Worker port"
        )


def build_plan(deployment: InstanceDeployment, *,
               inventory: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return process/config specs without creating files, reading credentials, or launching roles."""
    runtime = _load_runtime(deployment)
    devices = _gpu_inventory(inventory)
    missing_gpus = set(deployment.gpus) - set(devices)
    if missing_gpus:
        raise InstanceDeploymentError(f"configured GPU index is not present in inventory: {sorted(missing_gpus)}")

    source_owner_ids = set(runtime.learner.gradient_executor_worker_ids.values())
    source_worker = next(worker for worker in runtime.workers if worker.id in source_owner_ids)
    site_id, node_id = source_worker.site_id, source_worker.node_id
    worker_base_port = source_worker.port
    engine_base_port = _engine_base_port(runtime)
    count = len(deployment.gpus)
    worker_ports = [worker_base_port + index for index in range(count)]
    engine_ports = [engine_base_port + index for index in range(count)]
    used_ports = {runtime.learner.port}
    for port in (*worker_ports, *engine_ports):
        if not 1 <= port <= 65535 or port in used_ports:
            raise InstanceDeploymentError("generated Worker or NInfer ports are invalid or collide")
        used_ports.add(port)
    _validate_gateway_port(deployment, used_ports)

    if (deployment.dashboard_port is not None
            and (deployment.dashboard_port in used_ports
                 or (deployment.gateway_config is not None and deployment.dashboard_port == _DEFAULT_GATEWAY_PORT))):
        raise InstanceDeploymentError("dashboard port collides with a configured role")
    runtime_data = runtime.as_dict()
    if deployment.teacher_base_url is not None:
        if not runtime.bepa.enabled:
            raise InstanceDeploymentError("teacher_base_url requires an enabled BEPA teacher")
        from ..cluster.settings import validate_credential_origin

        try:
            runtime_data["bepa"]["teacher"]["base_url"] = validate_credential_origin(
                deployment.teacher_base_url
            )
        except ValueError as exc:
            raise InstanceDeploymentError("teacher_base_url is not a permitted credential destination") from exc
    learner_data = runtime_data["learner"]
    runtime_data["rollout"]["concurrent_tasks"] = max(runtime.rollout.concurrent_tasks, count)
    executor_ids = [f"executor-{index}" for index in range(count)]
    worker_ids = [f"worker-{index}" for index in range(count)]
    helper_ids = set(runtime.rollout.helper_worker_ids)
    helper_rows = [row for row in runtime_data["workers"] if row["id"] in helper_ids]
    if set(worker_ids) & helper_ids:
        raise InstanceDeploymentError("Gateway helper IDs cannot collide with generated managed Worker IDs")
    executor_envs = {executor_id: f"QWEN_GRADIENT_EXECUTOR_{index}_TOKEN"
                     for index, executor_id in enumerate(executor_ids)}
    remote_envs = {executor_id: runtime.learner.gradient_executor_token_envs[executor_id]
                   for executor_id in runtime.learner.gradient_remote_executor_ids}
    if set(remote_envs) & set(executor_envs):
        raise InstanceDeploymentError("remote executor IDs cannot collide with generated executor IDs")
    executor_envs.update(remote_envs)
    worker_auth_envs = {worker_id: _auth_environment_name(worker_id) for worker_id in worker_ids}
    helper_auth_envs = {row["auth_token_env"] for row in helper_rows}
    all_env_names = [*executor_envs.values(), *worker_auth_envs.values(), *helper_auth_envs, "ARC_ENV_TOKEN"]
    if runtime.bepa.enabled:
        all_env_names.extend((runtime.bepa.teacher.api_key_env, runtime.bepa.teacher.token_env))
    if len(set(all_env_names)) != len(all_env_names):
        raise InstanceDeploymentError("generated Worker/executor credential names collide with source credentials")

    learner_data["groups_per_update_auto"] = True
    learner_data["gradient_executor_token_envs"] = executor_envs
    from qwen_rl.contracts.sampling_topology import owner_slots, remap_paired_quotas
    new_pairs = dict(zip(executor_ids, worker_ids, strict=True))
    if set(new_pairs) & set(learner_data.get("sampling_only_worker_ids", {})):
        raise InstanceDeploymentError("generated executor IDs collide with sampling-only owners")
    learner_data["sampling_group_quotas"] = remap_paired_quotas(learner_data, new_pairs)
    learner_data["gradient_executor_worker_ids"] = new_pairs
    learner_data["groups_per_update"] = len(owner_slots(learner_data))
    if learner_data.get("sampling_only_worker_ids"):
        runtime_data["rollout"]["concurrent_tasks"] = max(
            runtime_data["rollout"]["concurrent_tasks"], learner_data["groups_per_update"])
    learner_data["gradient_local_executor_worker_ids"] = worker_ids
    learner_data["gradient_worker_sync_ids"] = worker_ids

    worker_rows: list[dict[str, Any]] = []
    worker_specs: list[dict[str, Any]] = []
    executor_specs: list[dict[str, Any]] = []
    inference_template = asdict(runtime.worker_inference)
    options = inference_template.get("ninfer_options")
    if isinstance(options, dict) and options.get("lora_gdn_projections") is False:
        options.pop("lora_gdn_projections")
    artifact_identity_path = runtime.artifact_identity_path
    if not isinstance(artifact_identity_path, str) or not artifact_identity_path:
        raise InstanceDeploymentError("ternary source profile has no artifact identity path")
    server_host = runtime.learner.host
    if server_host in {"0.0.0.0", "::", "[::]"}:
        server_host = "127.0.0.1"
    server_url = f"http://{server_host}:{runtime.learner.port}"
    frozen_store = str(deployment.packed_cache_root or _default_packed_cache_root(deployment))
    pending_root = deployment.state_dir / "gradient-runs" / "pending"

    for index, (worker_id, executor_id, gpu_index, worker_port, engine_port) in enumerate(
            zip(worker_ids, executor_ids, deployment.gpus, worker_ports, engine_ports, strict=True)):
        device = devices[gpu_index]
        gpu_uuid = device["uuid"]
        worker_root = deployment.state_dir / "workers" / worker_id
        worker_config_path = worker_root / "worker.json"
        token_path = deployment.state_dir / "tokens" / f"{worker_id}.token"
        inference = dict(inference_template)
        inference["ninfer_url"] = f"http://127.0.0.1:{engine_port}"
        worker_config = {
            "worker_id": worker_id,
            "node_id": node_id,
            "host": "127.0.0.1",
            "port": worker_port,
            "learner_url": server_url,
            "artifact_identity_path": str(Path(artifact_identity_path).resolve()),
            "adapter_cache": str(
                deployment.cache_root / "workers" / worker_id / "adapters"
                if deployment.cache_root is not None else worker_root / "adapters"
            ),
            "shared_policy_root": str(runtime.run / "policies"),
            "telemetry_dir": str(worker_root / "telemetry"),
            "lora_rank": runtime.lora.rank,
            "lora_alpha": runtime.lora.alpha,
            "request_timeout": runtime.learner.request_timeout,
            "inference": inference,
            "deployment_scope": "service",
            "artifact_transport": "direct",
            "residency": {
                "enabled": True,
                "weight_source": runtime.learner.worker_suspend_weight_source,
                "engine_lifecycle": runtime.learner.worker_engine_lifecycle,
                "drain_timeout_seconds": runtime.learner.gpu_sharing_timeout,
            },
        }
        runtime_worker = {
            "id": worker_id, "host": None, "port": worker_port, "gpu": gpu_index,
            "scope": "service", "site_id": site_id, "node_id": node_id,
            "endpoint": f"http://127.0.0.1:{worker_port}", "url_prefix": "",
            "auth_token_env": worker_auth_envs[worker_id],
            "boot_id": "", "telemetry_epoch": "", "model_sha256": "",
        }
        worker_rows.append(runtime_worker)
        worker_state = {
            "worker_id": worker_id, "gpu_index": gpu_index, "gpu_uuid": gpu_uuid,
            "site_id": site_id, "node_id": node_id, "port": worker_port,
            "engine_port": engine_port, "auth_token_env": worker_auth_envs[worker_id],
            "auth_token_path": str(token_path), "config_path": str(worker_config_path),
            "state_dir": str(worker_root), "worker_config": worker_config,
            "environment": {
                "CUDA_VISIBLE_DEVICES": gpu_uuid, "GPU_UUID": gpu_uuid, "PYTHONUNBUFFERED": "1",
            },
            "launcher_argv": [
                deployment.python, "-B", "-m", "qwen_rl.worker.ninfer_launcher",
                "--worker-config", str(worker_config_path), "--token-file", str(token_path),
                "--engine-binary", deployment.engine_binary, "--worker-python", deployment.worker_python,
                "--gpu-uuid", gpu_uuid, "--engine-host", "127.0.0.1", "--engine-port", str(engine_port),
                "--worker-host", "127.0.0.1", "--worker-port", str(worker_port),
                "--ready-timeout", str(deployment.start_timeout),
            ],
        }
        worker_specs.append(worker_state)

        work_dir = pending_root / "executors" / executor_id
        compute_profile = _compute_profile(
            runtime, artifact_identity_path=artifact_identity_path,
            run_dir=str(work_dir), store_dir=frozen_store,
        )
        worker_identity = {
            "id": worker_id, "scope": "service", "site_id": site_id, "node_id": node_id,
            "gpu": gpu_index, "gpu_uuid": gpu_uuid, "port": worker_port,
            "endpoint": f"http://127.0.0.1:{worker_port}",
            "auth_token_env": worker_auth_envs[worker_id],
            "boot_id": "", "telemetry_epoch": "", "model_sha256": "",
        }
        executor_config = {
            "local_work_stealing": runtime.learner.gradient_local_work_stealing,
            "schema_version": "qwen-rl-gradient-executor/v1",
            "server_url": server_url,
            "executor_id": executor_id,
            "token_env": executor_envs[executor_id],
            "cuda_visible_devices": gpu_uuid,
            "compute_profile": str(work_dir / "compute-profile.json"),
            "working_dir": str(work_dir),
            "policy_cache_dir": str(
                (deployment.cache_root or Path(runtime.run_dir)) / "executor-policy-cache"
            ),
            "poll_interval_seconds": 0.5,
            "heartbeat_interval_seconds": 10.0,
            "sharing": {
                "node_scope": {"site_id": site_id, "node_id": node_id},
                "gpu_index": gpu_index, "gpu_uuid": gpu_uuid,
                "shared_worker_ids": [worker_id], "workers": [worker_identity],
                "host_backup_budget_gib": runtime.learner.residency_host_budget_gib,
                "gpu_sharing_timeout_seconds": runtime.learner.gpu_sharing_timeout,
                "request_timeout_seconds": runtime.learner.request_timeout,
            },
        }
        executor_specs.append({
            "executor_id": executor_id, "worker_id": worker_id,
            "gpu_index": gpu_index, "gpu_uuid": gpu_uuid,
            "token_env": executor_envs[executor_id],
            "token_path": str(deployment.state_dir / "gradient-tokens" / f"{executor_id}.token"),
            "compute_profile": compute_profile,
            "executor_config": executor_config,
            "environment": {
                "CUDA_VISIBLE_DEVICES": gpu_uuid, "GPU_UUID": gpu_uuid,
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            },
            "environment_names": [worker_auth_envs[worker_id], executor_envs[executor_id]],
        })

    runtime_data["workers"] = [*worker_rows, *helper_rows]
    return {
        "deployment": deployment.public_dict(),
        **({"cache_root": str(deployment.cache_root)} if deployment.cache_root is not None else {}),
        "packed_cache_root": frozen_store,
        "runtime": runtime_data,
        "workers": worker_specs,
        "executors": executor_specs,
    }
