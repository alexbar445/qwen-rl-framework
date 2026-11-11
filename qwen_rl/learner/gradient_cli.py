"""Operator entry points for the shared-gradient Learner.

This module is intentionally safe to import from ``qwen-rl --help``: it uses
only the standard library until a role is actually started. In particular, a
compute executor sets ``CUDA_VISIBLE_DEVICES`` before importing its model
factory or Torch.
"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import MutableMapping
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit


EXECUTOR_SCHEMA = "qwen-rl-gradient-executor/v1"
_EXECUTOR_KEYS = {
    "schema_version", "server_url", "executor_id", "token_env",
    "cuda_visible_devices", "compute_profile", "working_dir",
    "poll_interval_seconds", "heartbeat_interval_seconds",
    "sharing", "local_work_stealing", "policy_cache_dir", "upload_pipeline", "gradient_wire_dtype",
}
_EXECUTOR_REQUIRED_KEYS = _EXECUTOR_KEYS - {"sharing", "local_work_stealing", "policy_cache_dir", "upload_pipeline", "gradient_wire_dtype"}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_DEVICE_LIST = re.compile(r"^[0-9]+(?:,[0-9]+)*$")
_GPU_UUID = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class GradientCLIError(ValueError):
    """A gradient CLI config or startup contract is invalid."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GradientCLIError("config contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise GradientCLIError("config JSON cannot contain non-finite numbers")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        text = path.expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        raise GradientCLIError(f"cannot read config file: {path}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except GradientCLIError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GradientCLIError("executor config must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise GradientCLIError("executor config must be a JSON object")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(ch) < 32 for ch in value):
        raise GradientCLIError(f"{name} must be a non-empty string")
    return value.strip()


def _env_name(value: Any, name: str) -> str:
    result = _nonempty_string(value, name)
    if not _ENV_NAME.fullmatch(result):
        raise GradientCLIError(f"{name} must name an environment variable")
    return result


def _positive_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GradientCLIError(f"{name} must be a number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise GradientCLIError(f"{name} must be finite") from exc
    lower_ok = result >= 0 if allow_zero else result > 0
    if not lower_ok or result == float("inf") or result != result:
        comparator = "non-negative" if allow_zero else "positive"
        raise GradientCLIError(f"{name} must be finite and {comparator}")
    return result


def _http_origin(value: str, name: str) -> tuple[str, int]:
    parsed = urlsplit(value)
    try:
        port = parsed.port
        host = parsed.hostname
    except ValueError as exc:
        raise GradientCLIError(f"{name} must have a valid host and port") from exc
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    if (parsed.scheme not in {"http", "https"} or not host
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise GradientCLIError(f"{name} must be an explicit HTTP(S) origin")
    if not 1 <= port <= 65535:
        raise GradientCLIError(f"{name} must use a valid TCP port")
    return host, port


def _loopback_url(value: str, name: str) -> tuple[str, int]:
    host, port = _http_origin(value, name)
    try:
        is_loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise GradientCLIError(f"{name} must be an explicit loopback HTTP(S) origin")
    return host, port


def _validate_sharing(value: Any, *, cuda_visible_devices: str, server_url: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GradientCLIError("sharing must be an explicit same-node Worker mapping")
    base_keys = {
        "node_scope", "gpu_index", "shared_worker_ids", "workers",
        "host_backup_budget_gib", "gpu_sharing_timeout_seconds", "request_timeout_seconds",
    }
    selected_uuid = cuda_visible_devices if _GPU_UUID.fullmatch(cuda_visible_devices) else None
    allowed = base_keys | ({"gpu_uuid"} if selected_uuid is not None else set())
    if set(value) != allowed:
        raise GradientCLIError("sharing fields must exactly identify the node, GPU, Workers, and time budgets")
    _http_origin(server_url, "server_url for same-node sharing")
    selected_gpu = value["gpu_index"]
    if selected_uuid is None:
        try:
            gpu_index = int(cuda_visible_devices)
        except (TypeError, ValueError) as exc:
            raise GradientCLIError(
                "same-node sharing requires one numeric CUDA index or one full GPU UUID"
            ) from exc
        if str(gpu_index) != cuda_visible_devices:
            raise GradientCLIError("same-node sharing requires one canonical numeric CUDA device index")
        if type(selected_gpu) is not int or selected_gpu != gpu_index:
            raise GradientCLIError("sharing.gpu_index must match cuda_visible_devices")
    else:
        if type(selected_gpu) is not int or selected_gpu < 0:
            raise GradientCLIError("sharing.gpu_index must be a non-negative physical GPU index")
        raw_uuid = value["gpu_uuid"]
        gpu_uuid = _nonempty_string(raw_uuid, "sharing.gpu_uuid")
        if raw_uuid != gpu_uuid or gpu_uuid != selected_uuid:
            raise GradientCLIError("sharing.gpu_uuid must exactly match cuda_visible_devices")
        gpu_index = selected_gpu
    node_scope = value["node_scope"]
    if not isinstance(node_scope, dict) or set(node_scope) != {"site_id", "node_id"}:
        raise GradientCLIError("sharing.node_scope must pin site_id and node_id")
    node = {
        key: _nonempty_string(node_scope[key], f"sharing.node_scope.{key}")
        for key in ("site_id", "node_id")
    }
    if any(not _SAFE_ID.fullmatch(item) for item in node.values()):
        raise GradientCLIError("sharing node identity contains unsupported characters")
    selected = value["shared_worker_ids"]
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(item, str) or not _SAFE_ID.fullmatch(item) for item in selected)
            or len(set(selected)) != len(selected)):
        raise GradientCLIError("sharing.shared_worker_ids must explicitly name unique Workers")
    workers = value["workers"]
    worker_keys = {
        "id", "scope", "site_id", "node_id", "gpu", "port", "endpoint",
        "auth_token_env", "boot_id", "telemetry_epoch", "model_sha256", "url_prefix", "gpu_uuid",
    }
    if (not isinstance(workers, list) or len(workers) != len(selected)
            or any(not isinstance(worker, dict) or set(worker) - worker_keys for worker in workers)):
        raise GradientCLIError("sharing.workers must contain exactly the selected pinned Worker records")
    if [worker.get("id") for worker in workers] != selected:
        raise GradientCLIError("sharing.workers order and IDs must exactly match shared_worker_ids")
    normalized_workers = []
    used_auth_envs: set[str] = set()
    for worker in workers:
        required_worker_keys = worker_keys - {"url_prefix", "gpu_uuid"}
        if required_worker_keys - set(worker):
            raise GradientCLIError("each shared Worker requires a complete node, GPU, endpoint, and identity pin")
        if selected_uuid is not None:
            worker_uuid = _nonempty_string(
                worker.get("gpu_uuid"), "shared Worker gpu_uuid",
            )
            if worker.get("gpu_uuid") != worker_uuid or worker_uuid != selected_uuid:
                raise GradientCLIError(
                    f"shared Worker {worker.get('id')} gpu_uuid must match sharing.gpu_uuid"
                )
        elif "gpu_uuid" in worker:
            raise GradientCLIError("shared Worker gpu_uuid is only valid with a UUID CUDA selector")
        worker_id = _nonempty_string(worker["id"], "shared Worker id")
        if worker["scope"] != "service":
            raise GradientCLIError("same-node sharing requires identity-pinned service Workers")
        if worker["site_id"] != node["site_id"] or worker["node_id"] != node["node_id"]:
            raise GradientCLIError(f"shared Worker {worker_id} differs from the pinned local node")
        if type(worker["gpu"]) is not int or worker["gpu"] != gpu_index:
            expected = "sharing.gpu_index" if selected_uuid is not None else "cuda_visible_devices"
            raise GradientCLIError(f"shared Worker {worker_id} GPU must match {expected}")
        if type(worker["port"]) is not int or not 1 <= worker["port"] <= 65535:
            raise GradientCLIError(f"shared Worker {worker_id} requires a valid local port")
        endpoint = _nonempty_string(worker["endpoint"], f"shared Worker {worker_id} endpoint")
        _host, endpoint_port = _loopback_url(endpoint, f"shared Worker {worker_id} endpoint")
        if endpoint_port != worker["port"]:
            raise GradientCLIError(f"shared Worker {worker_id} endpoint port differs from its pinned port")
        if worker.get("url_prefix", "") != "":
            raise GradientCLIError(f"shared Worker {worker_id} cannot use a routed URL prefix")
        auth_env = _env_name(worker["auth_token_env"], f"shared Worker {worker_id} auth_token_env")
        if auth_env in used_auth_envs:
            raise GradientCLIError("shared Workers must use distinct authentication environment variables")
        used_auth_envs.add(auth_env)
        for key in ("boot_id", "telemetry_epoch"):
            _nonempty_string(worker[key], f"shared Worker {worker_id} {key}")
        digest = worker["model_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise GradientCLIError(f"shared Worker {worker_id} requires a pinned model_sha256")
        normalized_worker = {**worker, "url_prefix": ""}
        normalized_worker.pop("gpu_uuid", None)
        normalized_workers.append(normalized_worker)
    normalized = {
        "node_scope": node,
        "gpu_index": gpu_index,
        "shared_worker_ids": list(selected),
        "workers": normalized_workers,
        "host_backup_budget_gib": _positive_number(value["host_backup_budget_gib"], "sharing.host_backup_budget_gib"),
        "gpu_sharing_timeout_seconds": _positive_number(
            value["gpu_sharing_timeout_seconds"], "sharing.gpu_sharing_timeout_seconds",
        ),
        "request_timeout_seconds": _positive_number(
            value["request_timeout_seconds"], "sharing.request_timeout_seconds",
        ),
    }
    if selected_uuid is not None:
        normalized["gpu_uuid"] = selected_uuid
    return normalized


def _validate_executor_config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    unknown = sorted(set(config) - _EXECUTOR_KEYS)
    missing = sorted(_EXECUTOR_REQUIRED_KEYS - set(config))
    if unknown or missing:
        fields = []
        if missing:
            fields.append(f"missing {', '.join(missing)}")
        if unknown:
            fields.append(f"unknown {', '.join(unknown)}")
        raise GradientCLIError("executor config fields are invalid: " + "; ".join(fields))
    if config.get("schema_version") != EXECUTOR_SCHEMA:
        raise GradientCLIError(f"schema_version must be {EXECUTOR_SCHEMA}")
    if type(config.get("local_work_stealing", False)) is not bool:
        raise GradientCLIError("local_work_stealing must be bool")
    dtype = config.get("gradient_wire_dtype", "float32")
    if not isinstance(dtype, str) or dtype not in {"float32", "bfloat16"}:
        raise GradientCLIError("gradient_wire_dtype must be float32 or bfloat16")
    config["executor_id"] = _nonempty_string(config["executor_id"], "executor_id")
    if not _SAFE_ID.fullmatch(config["executor_id"]):
        raise GradientCLIError("executor_id contains unsupported characters")
    config["token_env"] = _env_name(config["token_env"], "token_env")
    config["compute_profile"] = _nonempty_string(config["compute_profile"], "compute_profile")
    config["working_dir"] = _nonempty_string(config["working_dir"], "working_dir")
    working_dir = Path(config["working_dir"]).expanduser()
    if not working_dir.is_absolute() or ".." in working_dir.parts:
        raise GradientCLIError("working_dir must be an absolute path without '..'")
    config["working_dir"] = str(working_dir.resolve())
    if config.get("policy_cache_dir") is not None:
        cache_dir = Path(_nonempty_string(config["policy_cache_dir"], "policy_cache_dir")).expanduser()
        if not cache_dir.is_absolute() or ".." in cache_dir.parts or cache_dir.is_symlink():
            raise GradientCLIError("policy_cache_dir must be absolute without '..' or symlinks")
        config["policy_cache_dir"] = str(cache_dir)
    devices = _nonempty_string(config["cuda_visible_devices"], "cuda_visible_devices")
    if _GPU_UUID.fullmatch(devices):
        if config["cuda_visible_devices"] != devices:
            raise GradientCLIError("cuda_visible_devices GPU UUID must use its exact canonical spelling")
    else:
        if not _DEVICE_LIST.fullmatch(devices):
            raise GradientCLIError(
                "cuda_visible_devices must be one CUDA index or one full GPU UUID"
            )
        device_ids = devices.split(",")
        if len(device_ids) != 1:
            raise GradientCLIError("gradient executors require exactly one visible CUDA device")
        if str(int(device_ids[0])) != device_ids[0]:
            raise GradientCLIError("cuda_visible_devices must use canonical nonnegative CUDA indices")
    config["cuda_visible_devices"] = devices
    server_url = _nonempty_string(config["server_url"], "server_url")
    _http_origin(server_url, "server_url")
    config["server_url"] = server_url.rstrip("/")
    if config.get("sharing") is not None:
        config["sharing"] = _validate_sharing(
            config["sharing"],
            cuda_visible_devices=devices,
            server_url=config["server_url"],
        )
    for key in ("poll_interval_seconds", "heartbeat_interval_seconds"):
        config[key] = _positive_number(config[key], key)
    if type(config.get('upload_pipeline', True)) is not bool:
        raise GradientCLIError('upload_pipeline must be boolean')
    return config


def _validate_distributed_environment(env: Mapping[str, str]) -> None:
    """Reject a launcher environment that would initialize a multi-rank executor."""
    parsed: dict[str, int] = {}
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        value = env.get(name)
        if value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise GradientCLIError(f"inherited {name} is not an integer") from exc
        if number < 0:
            raise GradientCLIError(f"inherited {name} must be non-negative")
        parsed[name] = number
    if (parsed.get("WORLD_SIZE", 1) != 1
            or parsed.get("LOCAL_WORLD_SIZE", 1) != 1
            or parsed.get("RANK", 0) != 0
            or parsed.get("LOCAL_RANK", 0) != 0):
        raise GradientCLIError(
            "inherited distributed rank environment is incompatible with a single-device gradient executor"
        )


def _runtime_config(path: Path):
    """Load the existing strict runtime profile without importing model code."""
    from qwen_rl.runtime.config import load_config

    try:
        config = load_config(path.expanduser().resolve())
    except Exception as exc:
        # Runtime parser exceptions can include source values. Keep operator
        # output limited to the failure class and a stable message.
        raise GradientCLIError(f"runtime config is invalid ({type(exc).__name__})") from exc
    learner = config.learner
    if getattr(learner, "update_backend", None) != "central_gradient":
        raise GradientCLIError("server runtime must set learner.update_backend='central_gradient'")
    if tuple(getattr(learner, "gpus", ())) != ():
        raise GradientCLIError("central gradient server must set learner.gpus to an empty list")
    if getattr(learner, "parallelism", None) != "single":
        raise GradientCLIError("central gradient server must set learner.parallelism='single'")
    if (getattr(learner, "gpu_sharing", "dedicated") != "dedicated"
            or tuple(getattr(learner, "shared_worker_ids", ()) or ())):
        raise GradientCLIError("gradient server does not enable phase-shared Worker residency")
    adapter_path = getattr(learner, "initial_adapter_path", None)
    if not isinstance(adapter_path, str) or not Path(adapter_path).is_absolute():
        raise GradientCLIError("server requires an explicit absolute learner.initial_adapter_path")
    token_envs = getattr(learner, "gradient_executor_token_envs", None)
    if not isinstance(token_envs, Mapping) or not token_envs:
        raise GradientCLIError("server requires learner.gradient_executor_token_envs allowlist entries")
    for executor_id, env_name in token_envs.items():
        if not isinstance(executor_id, str) or not _SAFE_ID.fullmatch(executor_id):
            raise GradientCLIError("gradient executor allowlist contains an invalid executor id")
        _env_name(env_name, f"gradient executor {executor_id} token_env")
    executor_workers = getattr(learner, "gradient_executor_worker_ids", None)
    if (not isinstance(executor_workers, Mapping)
            or set(executor_workers) | set(getattr(learner, "gradient_remote_executor_ids", ())) != set(token_envs)
            or set(executor_workers) & set(getattr(learner, "gradient_remote_executor_ids", ()))
            or any(not isinstance(worker_id, str) or not worker_id for worker_id in executor_workers.values())
            or len(set(executor_workers.values())) != len(executor_workers)):
        raise GradientCLIError(
            "server requires one explicit learner.gradient_executor_worker_ids entry per allowlisted executor"
        )
    configured_worker_ids = {worker.id for worker in config.workers}
    if not set(executor_workers.values()).issubset(configured_worker_ids):
        raise GradientCLIError("gradient executor Worker mapping references an unconfigured Worker")
    resolved_groups = getattr(learner, "groups_per_update", None)
    from qwen_rl.contracts.sampling_topology import owner_slots
    if (type(resolved_groups) is not int or resolved_groups != len(owner_slots(vars(learner)))
            or (hasattr(learner, "groups_per_update_auto")
                and getattr(learner, "groups_per_update_auto") is not True)):
        raise GradientCLIError(
            "central gradient groups_per_update must be resolved by RuntimeConfig auto to the explicit Worker pair count"
        )
    worker_sync_ids = set(getattr(learner, "gradient_worker_sync_ids", ()) or ())
    if not set(executor_workers.values()).issubset(worker_sync_ids):
        raise GradientCLIError("every gradient executor Worker must be in learner.gradient_worker_sync_ids")
    if getattr(config, "model_backend", None) != "ternary":
        raise GradientCLIError("central gradient server currently requires model.backend='ternary'")
    recovery_dir = getattr(learner, "recovery_dir", None)
    if not isinstance(recovery_dir, str) or not Path(recovery_dir).is_absolute():
        raise GradientCLIError("server requires an explicit absolute learner.recovery_dir for central checkpoint resume")
    if getattr(learner, "recovery_mode", None) != "auto":
        raise GradientCLIError("central gradient server requires learner.recovery_mode='auto'")
    if getattr(learner, "recovery_lineage_id", None) is not None:
        raise GradientCLIError("central gradient server must not override its checkpoint lineage")
    if Path(recovery_dir).resolve() == (config.run / "training").resolve():
        raise GradientCLIError("central gradient checkpoints must use a dedicated directory outside run_dir/training")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-rl gradients",
        description="Run or inspect the central LoRA gradient service.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("server", "start the configured central gradient Learner API"),
        ("executor", "join the central service as one single-device compute executor"),
        ("status", "read central gradient service status"),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--config", type=Path, required=True,
                             help="runtime JSON/YAML for server, executor JSON for executor")
    return parser


def _server_command(
    config_path: Path,
    *,
    env: Mapping[str, str],
    app_factory: Callable[[dict[str, Any]], Any] | None,
    server_runner: Callable[..., Any] | None,
) -> int:
    runtime = _runtime_config(config_path)
    if not env.get("ARC_ENV_TOKEN"):
        raise GradientCLIError("required central Learner token environment variable is unset: ARC_ENV_TOKEN")
    # Validate presence without putting secret values into messages or logs.
    for env_name in runtime.learner.gradient_executor_token_envs.values():
        if not env.get(env_name):
            raise GradientCLIError(f"required token environment variable is unset: {env_name}")
    for worker in runtime.workers:
        if worker.scope in {"service", "gateway"} and not env.get(worker.auth_token_env):
            raise GradientCLIError(
                f"required Worker token environment variable is unset: {worker.auth_token_env}"
            )
    bepa = getattr(runtime, "bepa", None)
    teacher = getattr(bepa, "teacher", None)
    if getattr(bepa, "enabled", False) and teacher is not None:
        for env_name in (teacher.api_key_env, teacher.token_env):
            if env_name and not env.get(env_name):
                raise GradientCLIError(f"required central Teacher credential environment variable is unset: {env_name}")
    # The coordinator keeps only canonical CPU LoRA parameters and optimizer
    # state. Hide CUDA before importing the Learner app or any model runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if app_factory is None:
        from qwen_rl.learner.app import create_learner_app

        app_factory = create_learner_app
    app = app_factory(runtime.as_dict())
    host = runtime.learner.host
    port = runtime.learner.port
    if server_runner is None:
        import uvicorn

        server_runner = uvicorn.run
    server_runner(app, host=host, port=port, workers=1, access_log=False, log_level="info")
    return 0


def _executor_engine_config(config: dict[str, Any], *, env: MutableMapping[str, str],
                            torch_loaded: Callable[[], bool] | None,
                            profile_loader: Callable[[Path], Any] | None) -> tuple[Any, dict[str, Any]]:
    _validate_distributed_environment(env)
    if torch_loaded is None:
        torch_loaded = lambda: "torch" in sys.modules
    if torch_loaded():
        raise GradientCLIError(
            "Torch was imported before executor device selection; start the executor in a fresh process"
        )
    env["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    if env is not os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    if config.get("sharing") is not None:
        from qwen_rl.learner.cuda_allocator import configure_phase_shared_allocator

        configure_phase_shared_allocator(
            os.environ, model_backend="ternary", gpu_sharing="phase_shared",
        )
        if env is not os.environ:
            configure_phase_shared_allocator(
                env, model_backend="ternary", gpu_sharing="phase_shared",
            )

    profile_path = Path(config["compute_profile"]).expanduser()
    if not profile_path.is_absolute():
        # Relative paths are anchored to the executor config, not the caller's
        # current directory, so service launchers remain reproducible.
        profile_path = Path(config["_config_path"]).parent / profile_path
    profile_path = profile_path.resolve()
    try:
        if profile_loader is None:
            from qwen_rl.learner.ternary_config import load_ternary_compute_config

            profile_loader = load_ternary_compute_config
        profile = profile_loader(profile_path)
    except Exception as exc:
        raise GradientCLIError(f"compute profile is invalid ({type(exc).__name__})") from exc
    if profile.parallelism != "single" or len(profile.gpus) != 1:
        raise GradientCLIError("gradient executor compute profile must use exactly one GPU")
    engine_config = profile.engine_config()
    # A gradient executor evaluates contributions only. It must not restore a
    # regular Trainer/DP4 optimizer lineage or own optimizer state.
    engine_config.pop("recovery_dir", None)
    engine_config.pop("recovery_mode", None)
    engine_config.pop("recovery_lineage_id", None)
    engine_config["checkpoint_dir"] = str(Path(config["working_dir"]) / "checkpoints")
    engine_config["gpus"] = [0]  # the one selected physical device is index 0 after masking
    engine_config["parallelism"] = "single"
    engine_config["gradient_only"] = True
    return profile, engine_config


def _gradient_compute_profile(profile: Any) -> dict[str, Any]:
    from qwen_rl.learner.gradient_executor.compute import _LOSS_PROFILE

    identity = profile.identity.document
    artifact_id = identity.get("source", {}).get("ninfer_artifact", {}).get("artifact_id")
    if not isinstance(artifact_id, str) or re.fullmatch(r"[0-9a-f]{32}", artifact_id) is None:
        raise GradientCLIError("compute profile does not contain a pinned T2 artifact ID")
    return {
        "base_artifact_id": artifact_id,
        "loss_profile": _LOSS_PROFILE,
        "correction_cap": float(profile.importance_sampling_cap),
    }


def _shared_worker_policy_callbacks(
    config: Mapping[str, Any], sharing: Mapping[str, Any], client_config: Mapping[str, Any],
    *, env: Mapping[str, str],
) -> tuple[Callable[[], Mapping[str, Any]], Callable[..., Mapping[str, Any]]]:
    server_url = str(config["server_url"]).rstrip("/")
    executor_id = str(client_config["executor_id"])
    gradient_token = str(client_config["token"])
    api_root = f"{server_url}/gradient/v1/executors/{executor_id}"
    workers = {row["id"]: row for row in sharing["workers"]}

    def latest_policy() -> Mapping[str, Any]:
        import httpx

        try:
            with httpx.Client(
                timeout=sharing["request_timeout_seconds"],
                headers={"Accept": "application/json", "Authorization": f"Bearer {gradient_token}"},
                trust_env=False,
            ) as client:
                response = client.get(api_root + "/policy/current")
                response.raise_for_status()
                manifest = response.json()
                lineage_id = response.headers.get("X-Lineage-ID")
                engine_epoch = response.headers.get("X-Engine-Epoch")
        except httpx.HTTPError as exc:
            raise GradientCLIError(f"cannot read the current Learner policy ({type(exc).__name__})") from exc
        if not isinstance(manifest, dict) or not isinstance(lineage_id, str) or not lineage_id:
            raise GradientCLIError("current Learner policy manifest lacks its central lineage identity")
        if not isinstance(engine_epoch, str) or not engine_epoch:
            raise GradientCLIError("current Learner policy manifest lacks its central engine epoch")
        manifest["lineage_id"] = lineage_id
        manifest["engine_epoch"] = engine_epoch
        return manifest

    def ensure_policy(policy: Mapping[str, Any], worker_ids: tuple[str, ...]) -> Mapping[str, Any]:
        async def ensure_selected() -> dict[str, dict[str, Any]]:
            import httpx

            from qwen_rl.rollout.groups import PolicySnapshot

            version = policy.get("version")
            if type(version) is not int or version < 0:
                raise GradientCLIError("current Learner policy has an invalid version")
            identity = PolicySnapshot.from_policy(dict(policy))
            if (not isinstance(policy.get("lineage_id"), str)
                    or not isinstance(policy.get("engine_epoch"), str)):
                raise GradientCLIError("current Learner policy identity is incomplete")

            async def current_worker_identity(worker_id: str) -> dict[str, Any]:
                worker = workers[worker_id]
                token = env[worker["auth_token_env"]]
                expected_headers = {
                    "X-Expected-Worker-Boot-ID": worker["boot_id"],
                    "X-Expected-Worker-Telemetry-Epoch": worker["telemetry_epoch"],
                }
                async with httpx.AsyncClient(
                    timeout=sharing["request_timeout_seconds"],
                    headers={
                        "Authorization": f"Bearer {token}",
                        **expected_headers,
                    },
                    trust_env=False,
                ) as client:
                    worker_url = str(worker["endpoint"]).rstrip("/")
                    health_response = await client.get(worker_url + "/health")
                    health_response.raise_for_status()
                    for header, expected in (
                        ("X-Worker-Boot-ID", worker["boot_id"]),
                        ("X-Worker-Telemetry-Epoch", worker["telemetry_epoch"]),
                        ("X-Worker-Node-ID", worker["node_id"]),
                    ):
                        if health_response.headers.get(header) != expected:
                            raise GradientCLIError(f"shared Worker {worker_id} changed pinned {header} identity")
                    health = health_response.json()
                    deployment = health.get("deployment") if isinstance(health, dict) else None
                    if (not isinstance(health, dict) or health.get("worker_id") != worker_id
                            or health.get("boot_id") != worker["boot_id"]
                            or health.get("telemetry_epoch") != worker["telemetry_epoch"]
                            or health.get("ninfer_artifact_sha256") != worker["model_sha256"]
                            or health.get("residency_enabled") is not True
                            or not isinstance(deployment, Mapping)
                            or deployment.get("site_id") != worker["site_id"]
                            or deployment.get("node_id") != worker["node_id"]):
                        raise GradientCLIError(f"shared Worker {worker_id} health differs from its pinned identity")
                    status_response = await client.get(worker_url + "/residency")
                    status_response.raise_for_status()
                    for header, expected in (
                        ("X-Worker-Boot-ID", worker["boot_id"]),
                        ("X-Worker-Telemetry-Epoch", worker["telemetry_epoch"]),
                        ("X-Worker-Node-ID", worker["node_id"]),
                    ):
                        if status_response.headers.get(header) != expected:
                            raise GradientCLIError(f"shared Worker {worker_id} changed pinned {header} identity")
                    status = status_response.json()
                    if (not isinstance(status, dict) or status.get("worker_id") != worker_id
                            or status.get("enabled") is not True
                            or status.get("state") != "resident_fenced"
                            or status.get("admission_open") is not False):
                        raise GradientCLIError(f"shared Worker {worker_id} is not identity-pinned and fenced")
                    return status

            worker_states = await asyncio.gather(*(current_worker_identity(worker_id) for worker_id in worker_ids))
            if all(
                status.get("policy_version") == version
                and status.get("adapter_sha256") == identity.adapter_sha256
                for status in worker_states
            ):
                return {
                    status["worker_id"]: {
                        "worker_id": status["worker_id"],
                        "version": version,
                        "adapter_hash": identity.adapter_sha256,
                    }
                    for status in worker_states
                }

            body = {
                "lineage_id": policy["lineage_id"],
                "engine_epoch": policy["engine_epoch"],
                "policy_version": version,
                "worker_ids": list(worker_ids),
            }
            try:
                async with httpx.AsyncClient(
                    timeout=sharing["request_timeout_seconds"],
                    headers={"Authorization": f"Bearer {gradient_token}"},
                    trust_env=False,
                ) as client:
                    response = await client.post(api_root + "/workers/ensure", json=body)
                    response.raise_for_status()
                    result = response.json()
            except httpx.HTTPError as exc:
                raise GradientCLIError(f"central selected-Worker policy sync failed ({type(exc).__name__})") from exc
            rows = result.get("workers") if isinstance(result, dict) else None
            if (not isinstance(rows, Mapping) or result.get("policy_version") != version
                    or result.get("lineage_id") != policy["lineage_id"]
                    or result.get("engine_epoch") != policy["engine_epoch"]
                    or set(rows) != set(worker_ids)):
                raise GradientCLIError("central selected-Worker sync returned incomplete acknowledgements")
            acknowledgements = {}
            for worker_id, row in rows.items():
                if not isinstance(row, Mapping):
                    raise GradientCLIError(f"Worker {worker_id} returned a malformed policy acknowledgement")
                worker_id = row.get("worker_id")
                if (row.get("version") != version
                        or row.get("adapter_hash") != identity.adapter_sha256):
                    raise GradientCLIError(f"Worker {worker_id} ACK differs from the central policy identity")
                acknowledgements[worker_id] = {
                    "worker_id": worker_id,
                    "version": version,
                    "adapter_hash": identity.adapter_sha256,
                }
            return acknowledgements

        return asyncio.run(ensure_selected())

    return latest_policy, ensure_policy


def _build_shared_compute(
    config: Mapping[str, Any],
    sharing: Mapping[str, Any],
    client_config: Mapping[str, Any],
    profile: Any,
    engine_config: dict[str, Any],
    *,
    env: Mapping[str, str],
) -> Any:
    """Construct the persistent model owner and guarded same-node lifecycle."""
    from qwen_rl.learner.cuda_residency import CudaResidency
    from qwen_rl.learner.gradient_executor.lifecycle import (
        LifecycleGradientCompute,
        SameNodeGradientCycle,
    )
    from qwen_rl.learner.gradient_executor.owner import PersistentGradientOwner

    phase_config = {
        "model_backend": "ternary",
        "node_scope": dict(sharing["node_scope"]),
        "learner": {
            "gpu_sharing": "phase_shared",
            "shared_worker_ids": list(sharing["shared_worker_ids"]),
            "gpus": [sharing["gpu_index"]],
            "gpu_sharing_timeout": sharing["gpu_sharing_timeout_seconds"],
        },
        "workers": [dict(row) for row in sharing["workers"]],
    }
    residency = CudaResidency(
        host_backup_budget_bytes=int(sharing["host_backup_budget_gib"] * 1024**3),
        device=0,
    )

    def model_factory(cpu_prepared_model=None) -> Any:
        from qwen_rl.learner.gradient_executor.compute import build_compute

        return build_compute(dict(engine_config), cpu_prepared_model=cpu_prepared_model)

    def model_preparer() -> Any:
        from qwen_rl.learner.parallel.t2_model_assembly import prepare_t2_model_in_cpu

        return prepare_t2_model_in_cpu(engine_config)

    owner = PersistentGradientOwner(
        residency,
        model_factory=model_factory,
        model_getter=lambda compute: compute.engine.model,
        model_preparer=model_preparer,
    )
    latest_policy, ensure_policy = _shared_worker_policy_callbacks(
        config, sharing, client_config, env=env,
    )
    cycle = SameNodeGradientCycle(
        phase_config,
        compute_wake=owner.wake,
        compute_park=owner.park,
        compute_health=owner.health,
        latest_policy=latest_policy,
        ensure_policy=ensure_policy,
    )
    return LifecycleGradientCompute(
        owner,
        cycle,
        compute_profile=_gradient_compute_profile(profile),
        switch_timing_dir=str(Path(engine_config["compute_runtime_dir"]) / "gpu-switches"),
    )


def _executor_command(
    config_path: Path,
    *,
    env: MutableMapping[str, str],
    torch_loaded: Callable[[], bool] | None,
    profile_loader: Callable[[Path], Any] | None,
    compute_factory: Callable[[dict[str, Any]], Any] | None,
    sharing_compute_factory: Callable[..., Any] | None,
    executor_runner: Callable[[dict[str, Any], Any], Any] | None,
) -> int:
    config = _validate_executor_config(config_path.expanduser().resolve())
    config["_config_path"] = str(config_path.expanduser().resolve())
    token = env.get(config["token_env"])
    if not token:
        raise GradientCLIError(f"required token environment variable is unset: {config['token_env']}")
    sharing = config.get("sharing")
    if sharing is not None:
        for worker in sharing["workers"]:
            if not env.get(worker["auth_token_env"]):
                raise GradientCLIError(
                    "required shared Worker token environment variable is unset: "
                    f"{worker['auth_token_env']}"
                )
    profile, engine_config = _executor_engine_config(
        config, env=env, torch_loaded=torch_loaded, profile_loader=profile_loader,
    )
    # The resolved credential exists only in the in-memory client config. It is
    # never serialized, printed, or included in an exception message here.
    client_config = {
        "server_url": config["server_url"],
        "executor_id": config["executor_id"],
        "token": token,
        "compute_profile": config["compute_profile"],
        "working_dir": config["working_dir"],
        "local_work_stealing": config.get("local_work_stealing", False),
        "policy_cache_dir": config.get("policy_cache_dir"),
        "poll_interval": config["poll_interval_seconds"],
        "upload_pipeline": config.get('upload_pipeline', True),
        **({"gradient_wire_dtype": config["gradient_wire_dtype"]} if "gradient_wire_dtype" in config else {}),
        "heartbeat_interval": config["heartbeat_interval_seconds"],
        "compute_config": engine_config,
    }
    if executor_runner is None:
        from qwen_rl.learner.gradient_executor.client import run_executor

        executor_runner = run_executor

    def lazy_compute_factory(_client_config: Mapping[str, Any]) -> Any:
        if sharing is not None:
            if sharing_compute_factory is not None:
                return sharing_compute_factory(config, sharing, _client_config, profile, engine_config, env)
            return _build_shared_compute(
                config, sharing, _client_config, profile, engine_config, env=env,
            )
        if compute_factory is not None:
            return compute_factory(engine_config)
        from qwen_rl.learner.gradient_executor.compute import build_compute

        return build_compute(engine_config)

    result = executor_runner(client_config, lazy_compute_factory)
    if asyncio.iscoroutine(result):
        asyncio.run(result)
    return 0


def _status_url(runtime_or_executor: Any, executor_id: str) -> str:
    if isinstance(runtime_or_executor, dict):
        base = runtime_or_executor["server_url"].rstrip("/")
        return f"{base}/gradient/v1/executors/{executor_id}/status"
    host = runtime_or_executor.learner.host
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{runtime_or_executor.learner.port}/gradient/v1/executors/{executor_id}/status"


def _read_status(url: str, token: str | None, *, timeout: float = 10.0) -> dict[str, Any]:
    import httpx

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with httpx.Client(timeout=timeout, headers=headers, trust_env=False) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                chunks = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 2_000_000:
                        raise GradientCLIError("gradient status response exceeds 2 MB")
                    chunks.append(chunk)
                payload = b"".join(chunks)
    except httpx.HTTPStatusError as exc:
        raise GradientCLIError(
            f"gradient status request failed with HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise GradientCLIError(f"cannot reach gradient status endpoint ({type(exc).__name__})") from exc
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GradientCLIError("gradient status endpoint returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise GradientCLIError("gradient status endpoint returned an invalid object")
    return value


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if isinstance(key, str) and any(
                marker in key.lower() for marker in ("token", "secret", "authorization", "api_key", "password")
            ):
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = _redact_secrets(item)
        return cleaned
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


_DIAGNOSTIC_AUTH_RE = re.compile(
    r"(?i)(authorization\s*[:=]\s*)(?:(?:bearer|basic)\s+)?\S+"
)
_DIAGNOSTIC_BEARER_RE = re.compile(r"(?i)(\b(?:bearer|basic)\s+)\S+")
_DIAGNOSTIC_CREDENTIAL_RE = re.compile(
    r"(?i)(\b(?:access[_-]?token|refresh[_-]?token|token|api[_-]?key|secret|password|credential)\b\s*[:=]\s*)\S+"
)

def _redact_diagnostic_text(value: str, *environments: Mapping[str, str]) -> str:
    """Remove credential values before an executor traceback reaches its log."""
    secrets: set[str] = set()
    markers = ("token", "secret", "password", "credential", "api_key", "apikey", "private_key")
    for environment in environments:
        if not isinstance(environment, Mapping):
            continue
        for name, secret in environment.items():
            if (isinstance(name, str) and any(marker in name.lower() for marker in markers)
                    and isinstance(secret, str) and secret):
                secrets.add(secret)
    for secret in sorted(secrets, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    value = _DIAGNOSTIC_AUTH_RE.sub("[REDACTED]", value)
    value = _DIAGNOSTIC_BEARER_RE.sub("[REDACTED]", value)
    return _DIAGNOSTIC_CREDENTIAL_RE.sub(r"\1[REDACTED]", value)

def _status_command(config_path: Path, *, env: Mapping[str, str],
                    status_reader: Callable[[str, str | None], dict[str, Any]] | None) -> int:
    path = config_path.expanduser().resolve()
    # Executor configs are standalone strict JSON. Server configs are the
    # existing full RuntimeConfig accepted by the Learner app.
    try:
        raw = _load_json(path)
    except GradientCLIError:
        raw = {}
    if raw.get("schema_version") == EXECUTOR_SCHEMA:
        config = _validate_executor_config(path)
        token = env.get(config["token_env"])
        if not token:
            raise GradientCLIError(f"required token environment variable is unset: {config['token_env']}")
        url = _status_url(config, config["executor_id"])
    else:
        runtime = _runtime_config(path)
        token = None
        executor_id = None
        for candidate_id, env_name in runtime.learner.gradient_executor_token_envs.items():
            candidate = env.get(env_name)
            if candidate:
                executor_id, token = candidate_id, candidate
                break
        if executor_id is None:
            raise GradientCLIError("no gradient executor status token is present in the environment")
        url = _status_url(runtime, executor_id)
    if status_reader is None:
        status_reader = _read_status
    value = status_reader(url, token)
    print(json.dumps(_redact_secrets(value), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    mutable_env: MutableMapping[str, str] | None = None,
    torch_loaded: Callable[[], bool] | None = None,
    profile_loader: Callable[[Path], Any] | None = None,
    server_app_factory: Callable[[dict[str, Any]], Any] | None = None,
    server_runner: Callable[..., Any] | None = None,
    compute_factory: Callable[[dict[str, Any]], Any] | None = None,
    sharing_compute_factory: Callable[..., Any] | None = None,
    executor_runner: Callable[[dict[str, Any], Any], Any] | None = None,
    status_reader: Callable[[str, str | None], dict[str, Any]] | None = None,
) -> int:
    """Run one gradient operator command.

    Keyword-only dependencies are the CPU-only injection seam used by focused
    CLI tests; normal invocations use the production application factories.
    """
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    read_env = os.environ if env is None else env
    if mutable_env is not None:
        write_env = mutable_env
    elif isinstance(env, MutableMapping):
        write_env = env
    else:
        write_env = os.environ
    try:
        if args.command == "server":
            return _server_command(
                args.config, env=read_env, app_factory=server_app_factory,
                server_runner=server_runner,
            )
        if args.command == "executor":
            if not isinstance(write_env, MutableMapping):
                raise GradientCLIError("executor environment must be mutable")
            return _executor_command(
                args.config, env=write_env, torch_loaded=torch_loaded,
                profile_loader=profile_loader,
                compute_factory=compute_factory, executor_runner=executor_runner,
                sharing_compute_factory=sharing_compute_factory,
            )
        return _status_command(args.config, env=read_env, status_reader=status_reader)
    except GradientCLIError as exc:
        print(f"qwen-rl gradients: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"qwen-rl gradients: {args.command} failed ({type(exc).__name__})", file=sys.stderr)
        if args.command in {"executor", "server"}:
            formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            print(_redact_diagnostic_text(formatted, read_env, write_env), file=sys.stderr, end="")
        return 1


__all__ = ["GradientCLIError", "build_parser", "main"]
