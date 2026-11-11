"""Deployment selection and shared, read-only live metrics sampling.

This module deliberately imports no model, training, CUDA, or vLLM packages.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time
from typing import Any

import yaml

from ..runtime.watch import Rates
from ..runtime.watch_metrics import (
    COMMITTED_COUNTERS,
    RollingCounterWindow,
    SampledActivityWindow,
    classify_learner_activity,
    classify_worker_activity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_WATCH_BODY_BYTES = 8 * 1024 * 1024
MAX_HISTORY_SAMPLES = 3600
DEFAULT_HISTORY_SAMPLES = 600
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
WORKER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INSTANCE_WATCH_ENV_LOCK = threading.Lock()


def safe_run_id(value: Any) -> str | None:
    if (not isinstance(value, str) or not RUN_ID_RE.fullmatch(value)
            or value in {".", ".."} or Path(value).name != value):
        return None
    return value


def _read_mapping(path: Path) -> dict[str, Any] | None:
    try:
        info = path.lstat()
        if not path.is_file() or path.is_symlink() or info.st_size > MAX_CONFIG_BYTES:
            return None
        text = path.read_text(encoding="utf-8")
        value = json.loads(text) if text.lstrip().startswith("{") else yaml.safe_load(text)
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _container_path(value: Any) -> PurePosixPath | None:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        return None
    return path


@dataclass(frozen=True)
class DashboardSource:
    kind: str
    root: Path
    deployment_path: Path | None
    runtime_config_path: Path | None
    runtime: dict[str, Any] | None
    profile_config_path: Path | None
    mounts: tuple[tuple[PurePosixPath, Path], ...]
    explicit_bepa_store: Path | None = None
    deployment_id: str | None = None
    resolution_error: str | None = None
    native_runtime: Any = None
    runtime_registry_root: Path | None = None
    controller_id: str | None = None
    instance_manifest_path: Path | None = None

    @property
    def identity(self) -> str:
        parts = [self.kind, str(self.deployment_path or ""), str(self.root)]
        if self.deployment_id:
            parts.append(self.deployment_id)
        if self.instance_manifest_path:
            parts.append(str(self.instance_manifest_path))
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]

    def map_path(self, value: Any) -> Path | None:
        """Map a profile/container path to its host mount without escaping the mount."""
        text = str(value) if isinstance(value, (str, Path)) else ""
        if not text or "\x00" in text:
            return None
        raw = Path(text).expanduser()
        if not raw.is_absolute():
            raw = PROJECT_ROOT / raw
        else:
            posix = _container_path(text)
            if posix is not None:
                for container, host in sorted(self.mounts, key=lambda row: len(row[0].parts), reverse=True):
                    if posix == container or container in posix.parents:
                        relative = Path(*posix.relative_to(container).parts)
                        base = host.resolve(strict=False)
                        candidate = (base / relative).resolve(strict=False)
                        try:
                            candidate.relative_to(base)
                        except ValueError:
                            return None
                        return candidate
            if self.kind != "offline":
                return None
        candidate = raw.resolve(strict=False)
        if self.kind == "offline":
            allowed_roots = (self.root.resolve(strict=False), PROJECT_ROOT.resolve(strict=False))
            if not any(candidate == base or base in candidate.parents for base in allowed_roots):
                return None
        return candidate

    def matches_profile(self, manifest: dict[str, Any], run_id: str) -> bool:
        if self.kind == "offline":
            return True
        manifest_run = self.map_path(manifest.get("run_dir"))
        expected = (self.root / "runs" / run_id).resolve(strict=False)
        if manifest_run == expected:
            return True
        configured = self.map_path(manifest.get("config"))
        return bool(self.profile_config_path and configured == self.profile_config_path.resolve(strict=False))


def _runtime_root(runtime: dict[str, Any] | None, source: DashboardSource | None = None) -> Path | None:
    if not isinstance(runtime, dict):
        return None
    value = runtime.get("run_dir")
    return source.map_path(value) if source is not None else None


def _local_source(
    *,
    deployment: Path | str | None,
    root: Path | str | None,
    explicit_bepa_store: Path | str | None,
) -> DashboardSource:
    deployment_path = Path(deployment).expanduser().resolve(strict=False) if deployment else None
    raw: dict[str, Any] | None = None
    try:
        info = deployment_path.lstat() if deployment_path is not None else None
        if info is not None and stat_is_regular_file(info.st_mode) and not deployment_path.is_symlink() and info.st_size <= MAX_CONFIG_BYTES:
            raw = json.loads(deployment_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raw = None
    if not isinstance(raw, dict) or raw.get("schema_version") != "qwen-rl-local-deployment/v1":
        if root is None:
            raise ValueError("supply a valid --deployment profile or select offline --root")
        override = Path(root).expanduser().resolve(strict=False)
        return DashboardSource(
            "local", override, deployment_path, None, None, None,
            ((PurePosixPath("/work"), PROJECT_ROOT),),
            Path(explicit_bepa_store).expanduser().resolve(strict=False) if explicit_bepa_store else None,
            resolution_error="local deployment profile is unavailable",
        )
    workdir = raw.get("container_workdir")
    config_value = raw.get("default_config")
    container_config = _container_path(config_value)
    if workdir != "/work" or container_config is None or PurePosixPath("/work") not in (container_config, *container_config.parents):
        raise ValueError("local deployment default_config must be a safe path under /work")
    profile_path = (PROJECT_ROOT / Path(*container_config.relative_to(PurePosixPath("/work")).parts)).resolve(strict=False)
    runtime = _read_mapping(profile_path)
    mounts = ((PurePosixPath("/work"), PROJECT_ROOT),)
    registry_root = None
    controller_id = None
    from .central import container_bindings, find_registry, is_central
    if is_central(runtime) and root is None:
        controller_id, mounts = container_bindings(raw.get("container"))
        registry_root = find_registry(PROJECT_ROOT, profile_path)
        generated = registry_root.parent / "runtime-service.yaml"
        runtime = _read_mapping(generated)
        if not is_central(runtime):
            raise ValueError("central generated runtime config is unavailable")
    else:
        generated = profile_path
    temporary = DashboardSource(
        "local", PROJECT_ROOT, deployment_path, generated, runtime, profile_path, mounts,
        Path(explicit_bepa_store).expanduser().resolve(strict=False) if explicit_bepa_store else None,
        str(raw.get("container") or ""),
        None if runtime else "runtime profile is missing or invalid",
    )
    temporary = replace(temporary, runtime_registry_root=registry_root, controller_id=controller_id)
    configured_root = _runtime_root(runtime, temporary)
    selected_root = Path(root).expanduser().resolve(strict=False) if root is not None else configured_root
    if selected_root is None:
        raise ValueError("local runtime profile does not declare a readable run_dir")
    return replace(temporary, root=selected_root)


def stat_is_regular_file(mode: int) -> bool:
    import stat
    return stat.S_ISREG(mode)


def _learner_source(
    *,
    deployment: Path | str | None,
    root: Path | str | None,
    explicit_bepa_store: Path | str | None,
) -> DashboardSource:
    if deployment is None:
        raise ValueError("--source learner requires --deployment PROFILE")
    deployment_path = Path(deployment).expanduser().resolve(strict=True)
    from ..deployment.learner.config import load_learner_deployment

    profile = load_learner_deployment(deployment_path)
    runtime_path = profile.runtime_config.host_path.resolve(strict=False)
    runtime = _read_mapping(runtime_path)
    state_container = PurePosixPath(profile.state.container_path)
    mounts = ((state_container, profile.state.host_path),)
    temporary = DashboardSource(
        "learner", profile.state.host_path.resolve(strict=False), deployment_path, runtime_path,
        runtime, runtime_path, mounts,
        Path(explicit_bepa_store).expanduser().resolve(strict=False) if explicit_bepa_store else None,
        profile.deployment_id,
        None if runtime else "Learner runtime profile is missing or invalid",
    )
    configured_root = _runtime_root(runtime, temporary)
    selected_root = Path(root).expanduser().resolve(strict=False) if root is not None else configured_root
    if selected_root is None:
        raise ValueError("Learner runtime profile run_dir is outside the profile state mount")
    return DashboardSource(
        temporary.kind, selected_root, temporary.deployment_path, temporary.runtime_config_path,
        temporary.runtime, temporary.profile_config_path, temporary.mounts,
        temporary.explicit_bepa_store, temporary.deployment_id, temporary.resolution_error,
    )


def _runtime_source(*, deployment: Path | str | None,
                    instance_manifest: Path | str | None = None) -> DashboardSource:
    """Load one in-container RuntimeConfig without a deployment manager."""
    if deployment is None:
        raise ValueError("--source runtime requires --deployment CONFIG")
    supplied_path = Path(deployment).expanduser()
    try:
        info = supplied_path.lstat()
    except OSError as exc:
        raise ValueError("runtime config is missing or unreadable") from exc
    if (not stat_is_regular_file(info.st_mode) or supplied_path.is_symlink()
            or info.st_size > MAX_CONFIG_BYTES):
        raise ValueError("runtime config must be a bounded regular file")
    deployment_path = supplied_path.resolve(strict=True)
    if (instance_manifest is not None and (info.st_uid != os.getuid()
            or info.st_nlink != 1 or info.st_mode & 0o777 != 0o600)):
        raise ValueError("instance runtime config must be a private regular file")
    instance_manifest_path = None
    if instance_manifest is not None:
        candidate = Path(instance_manifest).expanduser()
        if not candidate.is_absolute():
            candidate = deployment_path.parent / candidate
        try:
            manifest_info = candidate.lstat()
        except OSError as exc:
            raise ValueError("instance gradient manifest is missing or unreadable") from exc
        if (not stat_is_regular_file(manifest_info.st_mode) or candidate.is_symlink()
                or manifest_info.st_uid != os.getuid() or manifest_info.st_nlink != 1
                or manifest_info.st_mode & 0o777 != 0o600 or manifest_info.st_size > MAX_CONFIG_BYTES):
            raise ValueError("instance gradient manifest must be a private bounded regular file")
        instance_manifest_path = candidate.resolve(strict=True)
        if (deployment_path.name != "runtime.yaml"
                or instance_manifest_path.name != "gradient-manifest.json"
                or deployment_path.parent != instance_manifest_path.parent):
            raise ValueError("instance manifest must be gradient-manifest.json beside runtime.yaml")

    # This is the native container path: validate with the same strict loader used
    # by runtime commands, without Docker or LearnerManager ownership checks.
    from ..runtime.config import load_config

    config = load_config(deployment_path)
    runtime = config.as_dict()
    configured_run_dir = config.run_dir
    runtime["run_dir"] = str(config.run)
    workers = runtime.get("workers")
    if isinstance(workers, tuple):
        runtime["workers"] = list(workers)
    try:
        config_sha = hashlib.sha256(deployment_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("runtime config became unavailable while loading") from exc
    mounts = ((PurePosixPath("/work"), PROJECT_ROOT),)
    if instance_manifest_path is not None:
        mounts += ((PurePosixPath("/state"), Path("/state")),)
    temporary = DashboardSource(
        "runtime", PROJECT_ROOT, deployment_path, deployment_path, runtime,
        deployment_path, mounts, deployment_id=config_sha, native_runtime=config,
        instance_manifest_path=instance_manifest_path,
    )
    selected_root = _runtime_root(runtime, temporary)
    if selected_root is None and instance_manifest_path is not None:
        from .central import current_instance
        if current_instance(temporary) is None:
            raise ValueError("instance gradient manifest does not prove a live matching runtime")
        configured_root = Path(configured_run_dir)
        try:
            native_root = configured_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("managed native runtime run_dir is unavailable") from exc
        if (not configured_root.is_absolute() or native_root == Path("/")
                or not native_root.is_dir() or configured_root != native_root):
            raise ValueError("managed native runtime run_dir must be a canonical non-root directory")
        temporary = replace(
            temporary,
            mounts=temporary.mounts + ((PurePosixPath(str(native_root)), native_root),),
        )
        selected_root = native_root
    if selected_root is None:
        raise ValueError("runtime config run_dir must resolve under an allowed runtime mount")
    resolved = DashboardSource(
        temporary.kind, selected_root, temporary.deployment_path,
        temporary.runtime_config_path, temporary.runtime,
        temporary.profile_config_path, temporary.mounts,
        deployment_id=temporary.deployment_id,
        native_runtime=config,
        instance_manifest_path=temporary.instance_manifest_path,
    )
    if instance_manifest_path is not None:
        from .central import current_instance
        if current_instance(resolved) is None:
            raise ValueError("instance gradient manifest does not prove a live matching runtime")
    return resolved


def resolve_source(
    kind: str,
    *,
    deployment: Path | str | None = None,
    root: Path | str | None = None,
    bepa_store: Path | str | None = None,
    instance_manifest: Path | str | None = None,
) -> DashboardSource:
    """Resolve exactly one local, native runtime, Learner, or offline source."""
    if kind == "offline":
        if instance_manifest is not None:
            raise ValueError("--instance-manifest is available only with --source runtime")
        if root is None:
            raise ValueError("offline source requires --root")
        selected_root = Path(root).expanduser().resolve(strict=False)
        explicit = Path(bepa_store).expanduser().resolve(strict=False) if bepa_store else None
        return DashboardSource(
            "offline", selected_root, None, None, None, None,
            ((PurePosixPath("/work"), PROJECT_ROOT),), explicit,
            resolution_error=None if selected_root.is_dir() else "offline root does not exist",
        )
    if kind == "local":
        if instance_manifest is not None:
            raise ValueError("--instance-manifest is available only with --source runtime")
        return _local_source(deployment=deployment, root=root, explicit_bepa_store=bepa_store)
    if kind == "learner":
        if instance_manifest is not None:
            raise ValueError("--instance-manifest is available only with --source runtime")
        return _learner_source(deployment=deployment, root=root, explicit_bepa_store=bepa_store)
    if kind == "runtime":
        if root is not None or bepa_store is not None:
            raise ValueError("runtime source uses its config run_dir; omit --root and --bepa-store")
        return _runtime_source(deployment=deployment, instance_manifest=instance_manifest)
    raise ValueError("source must be local, runtime, learner, or offline")


def safe_runtime_document(path: Path) -> dict[str, Any] | None:
    return _read_mapping(path)


def runtime_paths(runtime: dict[str, Any], source: DashboardSource) -> tuple[dict[str, Path | None], dict[str, str]]:
    """Resolve only the run, BEPA, recovery and evaluation paths from one runtime file."""
    errors: dict[str, str] = {}
    run_value = runtime.get("run_dir")
    run_path = source.map_path(run_value)
    if run_path is None:
        errors["run_dir"] = "run_dir is missing or outside the selected deployment mount"
    bepa = runtime.get("bepa") if isinstance(runtime.get("bepa"), dict) else {}
    learner = runtime.get("learner") if isinstance(runtime.get("learner"), dict) else {}
    evaluation = runtime.get("evaluation") if isinstance(runtime.get("evaluation"), dict) else {}
    store_value = bepa.get("state_dir")
    if not store_value and run_value:
        if source.kind in {"local", "learner"}:
            store_path = source.map_path(str(run_value).rstrip("/") + "/bepa")
        else:
            store_path = (run_path / "bepa") if run_path else None
    else:
        store_path = source.map_path(store_value)
    recovery_value = learner.get("recovery_dir")
    if not recovery_value and run_value:
        recovery_value = str(run_value).rstrip("/") + "/training"
    recovery_path = source.map_path(recovery_value)
    evaluation_path = source.map_path(evaluation.get("state_dir")) if evaluation.get("state_dir") else None
    if store_path is None and store_value:
        errors["bepa_store"] = "BEPA state_dir is outside the selected deployment mount"
    if recovery_path is None and recovery_value:
        errors["recovery_dir"] = "recovery_dir is outside the selected deployment mount"
    if evaluation.get("state_dir") and evaluation_path is None:
        errors["evaluation_state_dir"] = "evaluation state_dir is outside the selected deployment mount"
    return {
        "run_dir": run_path,
        "bepa_store": store_path,
        "recovery_dir": recovery_path,
        "evaluation_state_dir": evaluation_path,
    }, errors


def sanitized_runtime(runtime: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(runtime, dict):
        return None
    learner = runtime.get("learner") if isinstance(runtime.get("learner"), dict) else {}
    bepa = runtime.get("bepa") if isinstance(runtime.get("bepa"), dict) else {}
    evaluation = runtime.get("evaluation") if isinstance(runtime.get("evaluation"), dict) else {}
    return {
        "run_dir": runtime.get("run_dir"),
        "bepa": {
            "enabled": bepa.get("enabled"),
            "budget_scope": bepa.get("budget_scope"),
            "state_dir": bepa.get("state_dir"),
        },
        "learner": {"recovery_dir": learner.get("recovery_dir"),
                    "update_backend": learner.get("update_backend")},
        "evaluation": {
            "enabled": evaluation.get("enabled"),
            "state_dir": evaluation.get("state_dir"),
            "every_updates": evaluation.get("every_updates"),
            "concurrency": evaluation.get("concurrency"),
            "pause_training": evaluation.get("pause_training"),
        },
    }


def _read_response(response: Any) -> dict[str, Any] | None:
    try:
        if response.status_code != 200 or len(response.content) > MAX_WATCH_BODY_BYTES:
            return None
        value = response.json()
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _instance_worker_environment(source: DashboardSource) -> dict[str, str]:
    """Read only the private Worker token files owned by the explicit instance run."""
    manifest = source.instance_manifest_path
    if manifest is None:
        return {}
    if not isinstance(source.runtime, dict):
        raise ValueError("instance runtime config is unavailable")
    from .central import current_instance
    if current_instance(source) is None:
        raise ValueError("instance runtime generation is no longer verified")
    state_root = manifest.parent.parent.parent
    tokens_root = state_root / "tokens"
    if tokens_root.is_symlink() or not tokens_root.is_dir():
        raise ValueError("instance Worker credential directory is unavailable")
    values: dict[str, str] = {}
    for worker in source.runtime.get("workers", ()):
        if not isinstance(worker, dict) or worker.get("scope") not in {"service", "gateway"}:
            continue
        worker_id = worker.get("id")
        env_name = worker.get("auth_token_env")
        if (not isinstance(worker_id, str) or not WORKER_ID_RE.fullmatch(worker_id)
                or not isinstance(env_name, str) or not _ENV_NAME_RE.fullmatch(env_name)
                or env_name in values):
            raise ValueError("instance Worker credential binding is invalid")
        token_path = tokens_root / (worker_id + ".token")
        try:
            descriptor = os.open(token_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                info = os.fstat(descriptor)
                if (not stat_is_regular_file(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_nlink != 1 or info.st_mode & 0o777 != 0o600
                        or info.st_size > 16384):
                    raise ValueError("unsafe instance Worker credential")
                with os.fdopen(descriptor, "rb", closefd=False) as handle:
                    raw = handle.read(16385)
            finally:
                os.close(descriptor)
            if len(raw) > 16384:
                raise ValueError("instance Worker credential is too large")
            token = raw.decode("utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ValueError("instance Worker credential is unavailable") from exc
        if not token or any(character.isspace() for character in token):
            raise ValueError("instance Worker credential is invalid")
        values[env_name] = token
    return values


def runtime_watch_snapshot(source: DashboardSource, *, active_run_id: str | None = None,
                           selected_run_id: str | None = None,
                           timeout: float = 8.0) -> dict[str, Any]:
    """Poll the existing runtime WatchClient using the dashboard process environment."""
    if source.kind != "runtime" or source.native_runtime is None:
        return {"ok": False, "snapshot": None, "error": "native runtime source is not selected"}
    if selected_run_id is not None and selected_run_id != active_run_id:
        return {"ok": False, "snapshot": None, "error": "historical run has no live snapshot"}

    async def poll() -> dict[str, Any]:
        from ..runtime.watch import WatchClient

        config = source.native_runtime
        if source.instance_manifest_path is not None:
            config = replace(config, run_dir=str(source.instance_manifest_path.parent))
        client = WatchClient(config, timeout=timeout, window=30.0,
                             log_lines=0, gpu=True)
        try:
            return await client.poll()
        finally:
            await client.close()

    try:
        import asyncio
        worker_environment = _instance_worker_environment(source)
        if worker_environment:
            with _INSTANCE_WATCH_ENV_LOCK:
                previous = {name: os.environ.get(name) for name in worker_environment}
                try:
                    os.environ.update(worker_environment)
                    snapshot = asyncio.run(poll())
                finally:
                    for name, value in previous.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
        else:
            snapshot = asyncio.run(poll())
    except Exception as exc:
        return {"ok": False, "snapshot": None,
                "error": "runtime watch failed: " + type(exc).__name__}
    if isinstance(snapshot, dict):
        snapshot["gpu_source"] = "selected_runtime_host"
    return {"ok": True, "snapshot": snapshot, "error": None}


def learner_watch_snapshot(source: DashboardSource, *, active_run_id: str | None = None,
                           selected_run_id: str | None = None, timeout: float = 8.0) -> dict[str, Any]:
    """Read authenticated Learner metadata after the profile manager verifies ownership."""
    if source.kind != "learner" or source.deployment_path is None:
        return {"ok": False, "snapshot": None, "error": "Learner source is not selected"}
    if selected_run_id is not None and selected_run_id != active_run_id:
        return {"ok": False, "snapshot": None, "error": "historical run has no live snapshot"}
    try:
        from ..deployment.learner.config import load_learner_deployment
        from ..deployment.learner.manager import LearnerManager
        import httpx

        deployment = load_learner_deployment(source.deployment_path)
        manager = LearnerManager(deployment)
        status = manager.status()
        runtime_status = status.get("runtime") if isinstance(status, dict) else None
        container_id = status.get("container_id") if isinstance(status, dict) else None
        if (status.get("status") == "profile_changed" or not isinstance(runtime_status, dict)
                or not isinstance(container_id, str) or not container_id):
            return {"ok": False, "snapshot": None, "error": "selected Learner container is not verified"}
        manager_run_id = runtime_status.get("run_id")
        if active_run_id is not None and manager_run_id != active_run_id:
            return {"ok": False, "snapshot": None, "error": "Learner runtime identity differs from selected manifest"}
        if selected_run_id is not None and manager_run_id != selected_run_id:
            return {"ok": False, "snapshot": None, "error": "Learner runtime identity differs from selected run"}

        token = manager._read_arc_token()
        base_url = manager._published_health_url().rsplit("/", 1)[0]
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            headers = {"Authorization": "Bearer " + token, "Accept": "application/json"}
            health_response = client.get(base_url + "/health", headers=headers)
            monitor_response = client.get(base_url + "/monitor", headers=headers)
        health = _read_response(health_response)
        monitor = _read_response(monitor_response)
        telemetry_epoch = monitor.get("telemetry_epoch") if isinstance(monitor, dict) else None
        monitor_run_id = monitor.get("runtime_run_id") if isinstance(monitor, dict) else None
        if (not isinstance(health, dict) or health.get("role") != "learner"
                or not isinstance(monitor, dict) or monitor.get("schema") != "qwen-rl-monitor-v1"):
            return {"ok": False, "snapshot": None, "error": "Learner live metadata is unavailable"}
        if monitor_run_id is not None and manager_run_id is not None and monitor_run_id != manager_run_id:
            return {"ok": False, "snapshot": None, "error": "Learner runtime identity differs from owned status"}
        if selected_run_id is not None and monitor_run_id != selected_run_id:
            return {"ok": False, "snapshot": None, "error": "Learner monitor has no matching runtime identity"}
        if active_run_id is not None and monitor_run_id != active_run_id:
            return {"ok": False, "snapshot": None, "error": "Learner monitor has no matching runtime identity"}

        owned_identity = {
            "container_id": container_id,
            "profile_sha256": manager.profile_sha256,
            "telemetry_epoch": telemetry_epoch,
            "deployment_id": deployment.deployment_id,
        }
        runtime = source.runtime or {}
        configured_workers = runtime.get("workers") if isinstance(runtime, dict) else []
        worker_rows = []
        for item in configured_workers if isinstance(configured_workers, list) else []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                worker_rows.append({
                    "id": item["id"], "live": False, "age_seconds": None,
                    "active": None, "maximum": None, "queued": None,
                    "generated_tokens": None, "tokens_per_second": None,
                    "ready": None, "policy_ready": None, "episode_refs": None,
                    "telemetry_epoch": None, "activity_state": "unknown",
                })
        training = monitor.get("training") if isinstance(monitor.get("training"), dict) else {}
        process = training.get("process_committed") if isinstance(training.get("process_committed"), dict) else {}
        run = training.get("run_committed") if isinstance(training.get("run_committed"), dict) else {}
        def totals(value: dict[str, Any]) -> dict[str, Any]:
            return {name: value.get(name) for name in COMMITTED_COUNTERS}
        process_epoch = process.get("telemetry_epoch") or monitor.get("telemetry_epoch")
        committed = {
            "window_seconds": 300.0,
            "process": {
                "live": True, "scope": process.get("scope", "current_process"),
                "telemetry_epoch": process_epoch, "elapsed_seconds": process.get("elapsed_seconds"),
                "totals": totals(process), "lifetime_rates_per_hour": {}, "rolling": {},
            },
            "run": {
                "live": True, "scope": run.get("scope"),
                "run_id": run.get("run_id") or monitor_run_id,
                "telemetry_epoch": run.get("telemetry_epoch"),
                "elapsed_seconds": run.get("elapsed_seconds"),
                "logical_completed_updates": run.get("logical_completed_updates"),
                "prior_completed_updates": run.get("prior_completed_updates"),
                "prior_resumed_token_totals": run.get("prior_resumed_token_totals", "unknown"),
                "totals": totals(run), "lifetime_rates_per_hour": {}, "rolling": {},
            },
        }
        snapshot = {
            "schema": "qwen-rl-watch-v1",
            "sampled_at": monitor.get("sampled_at", time.time()),
            "lifecycle": {
                "status": status.get("status"),
                "run_id": monitor_run_id if isinstance(monitor_run_id, str) else None,
                "owned_source_verified": True,
            },
            "owned_source_verified": True,
            "source_identity": owned_identity,
            "membership": None,
            "learner": monitor,
            "learner_health": health,
            "workers": worker_rows,
            "throughput": {
                "aggregate_output_tokens_per_second": None,
                "learner_forward_input_tokens_per_second": None,
                "learner_forward_scope": None,
            },
            "committed": committed,
            "worker_activity": {},
            "learner_activity": {},
            "freshness": {"learner": {"live": True, "age_seconds": 0}},
            "errors": {},
            "gpus": None,
            "remote_hardware": None,
        }
        return {"ok": True, "snapshot": snapshot, "error": None}
    except Exception as exc:
        return {"ok": False, "snapshot": None,
                "error": "Learner source read failed: " + type(exc).__name__}


class LiveSampler:
    """Thread-safe rolling metrics and bounded history shared by dashboard polling."""

    def __init__(self, *, history_limit: int = MAX_HISTORY_SAMPLES,
                 rate_window_seconds: float = 30.0,
                 activity_window_seconds: float = 300.0,
                 max_observation_gap_seconds: float = 12.0) -> None:
        self.history_limit = max(2, min(int(history_limit), MAX_HISTORY_SAMPLES))
        self.rate_window_seconds = float(rate_window_seconds)
        self.activity_window_seconds = float(activity_window_seconds)
        self.max_observation_gap_seconds = float(max_observation_gap_seconds)
        self._lock = threading.RLock()
        self._history: deque[dict[str, Any]] = deque(maxlen=self.history_limit)
        self._samples_seen = 0
        self._source_identity: str | None = None
        self._runtime_identity: tuple[Any, ...] | None = None
        self._last_observed: float | None = None
        self._rates = Rates(self.rate_window_seconds)
        self._committed_rates = RollingCounterWindow(self.activity_window_seconds)
        self._worker_activity = SampledActivityWindow(
            self.activity_window_seconds, self.max_observation_gap_seconds,
        )
        self._learner_activity = SampledActivityWindow(
            self.activity_window_seconds, self.max_observation_gap_seconds,
            basis="sampled_learner_walltime_not_gpu_utilization",
        )

    def reset(self, source_identity: str | None = None) -> None:
        with self._lock:
            self._history.clear()
            self._samples_seen = 0
            self._source_identity = source_identity
            self._runtime_identity = None
            self._reset_windows()

    def _reset_windows(self) -> None:
        self._last_observed = None
        self._rates.series.clear()
        self._committed_rates.clear()
        self._worker_activity.clear()
        self._learner_activity.clear()

    @staticmethod
    def _fresh(item: dict[str, Any] | None, *, stale_after_seconds: float = 15.0) -> bool:
        if not isinstance(item, dict) or item.get("live") is not True:
            return False
        age = item.get("age_seconds")
        return (age is None or (
            isinstance(age, (int, float)) and not isinstance(age, bool)
            and math.isfinite(float(age)) and 0 <= float(age) <= stale_after_seconds
        ))

    def observe(self, result: dict[str, Any], *, source_identity: str,
                expected_worker_ids: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
        with self._lock:
            if source_identity != self._source_identity:
                self._history.clear()
                self._samples_seen = 0
                self._source_identity = source_identity
                self._runtime_identity = None
                self._reset_windows()
            snapshot = result.get("snapshot") if isinstance(result, dict) else None
            if not isinstance(snapshot, dict):
                # Do not average a resumed counter over an interval with no observations.
                self._rates.series.clear()
                self._committed_rates.clear()
                self._append_error(result, source_identity)
                return result
            lifecycle = snapshot.get("lifecycle") if isinstance(snapshot.get("lifecycle"), dict) else {}
            run_id = lifecycle.get("run_id")
            telemetry = snapshot.get("committed") if isinstance(snapshot.get("committed"), dict) else {}
            process = telemetry.get("process") if isinstance(telemetry.get("process"), dict) else {}
            process_epoch = process.get("telemetry_epoch")
            identity_learner = snapshot.get("learner") or {}
            logical_run = identity_learner.get("run") or {}
            runtime_identity = (run_id, process_epoch, source_identity, logical_run.get("run_id"))
            if self._runtime_identity is not None and runtime_identity != self._runtime_identity:
                self._history.clear()
                self._samples_seen = 0
                self._reset_windows()
            self._runtime_identity = runtime_identity
            now = time.monotonic()
            if self._last_observed is not None and now - self._last_observed > self.max_observation_gap_seconds:
                self._reset_windows()
            self._last_observed = now
            freshness = snapshot.get("freshness") if isinstance(snapshot.get("freshness"), dict) else {}
            learner_fresh = self._fresh(freshness.get("learner"))
            learner = snapshot.get("learner") if isinstance(snapshot.get("learner"), dict) else {}
            training = learner.get("training") if isinstance(learner.get("training"), dict) else {}
            learner_activity = classify_learner_activity(learner, live=learner_fresh)
            self._learner_activity.observe(
                "learner", learner_activity, now=now,
                epoch=training.get("telemetry_epoch") or process_epoch,
            )
            snapshot["learner_activity"] = self._learner_activity.summary(now=now)

            worker_rows = snapshot.get("workers") if isinstance(snapshot.get("workers"), list) else []
            by_id = {row.get("id"): row for row in worker_rows if isinstance(row, dict)
                     and isinstance(row.get("id"), str)}
            worker_ids = tuple(dict.fromkeys(
                value for value in (*expected_worker_ids, *by_id.keys())
                if isinstance(value, str) and WORKER_ID_RE.fullmatch(value)
            ))
            stream = learner.get("sampling") if isinstance(learner.get("sampling"), dict) else {}
            stream = stream.get("stream") if isinstance(stream.get("stream"), dict) else {}
            producer_reason = stream.get("producer_backpressure_reason")
            rates: list[float | None] = []
            for worker_id in worker_ids:
                row = by_id.get(worker_id)
                freshness_item = freshness.get(worker_id)
                if not isinstance(freshness_item, dict) and isinstance(row, dict):
                    freshness_item = {"live": row.get("live"), "age_seconds": row.get("age_seconds")}
                worker_fresh = self._fresh(freshness_item)
                telemetry_epoch = row.get("telemetry_epoch") if isinstance(row, dict) else None
                generated = row.get("generated_tokens") if isinstance(row, dict) else None
                rate = self._rates.sample(
                    worker_id, telemetry_epoch if worker_fresh else None,
                    generated if worker_fresh else None, now,
                )
                rates.append(rate if worker_fresh else None)
                if not isinstance(row, dict):
                    row = {"id": worker_id, "live": False, "age_seconds": None,
                           "telemetry_epoch": None, "generated_tokens": None}
                    worker_rows.append(row)
                    by_id[worker_id] = row
                row["tokens_per_second"] = rate if worker_fresh else None
                row["activity_state"] = classify_worker_activity(
                    {**row, "live": worker_fresh}, producer_reason,
                )
                self._worker_activity.observe(
                    worker_id, row["activity_state"], now=now,
                    epoch=telemetry_epoch if worker_fresh else None,
                )
            snapshot["workers"] = worker_rows
            snapshot["worker_activity"] = self._worker_activity.summary(now=now)
            aggregate = (sum(rate for rate in rates if rate is not None)
                         if rates and all(rate is not None for rate in rates) else None)
            throughput = snapshot.get("throughput") if isinstance(snapshot.get("throughput"), dict) else {}
            throughput["aggregate_output_tokens_per_second"] = aggregate
            native = snapshot.get("native_metrics") or {}
            native_rows = {r.get("id"): r for r in native.get("workers", []) if isinstance(r, dict)}
            native_rates = []
            for worker_id in worker_ids:
                row = by_id.get(worker_id, {})
                observation = native_rows.get(worker_id, {})
                epoch = observation.get("telemetry_epoch")
                stamp = observation.get("monotonic_sample")
                valid = (observation.get("live") is True and row.get("live") is True
                         and epoch == row.get("telemetry_epoch")
                         and isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
                         and math.isfinite(stamp) and 0 <= now - stamp <= 5.0)
                native_rate = self._rates.sample("native_decode:" + worker_id,
                    epoch if valid else None, observation.get("decode_tokens") if valid else None,
                    stamp if valid else now)
                row["native_decode_tokens_per_second"] = native_rate
                native_rates.append(native_rate)
            throughput["native_decode_tokens_per_second"] = (
                sum(native_rates) if native_rates and all(v is not None for v in native_rates) else None)
            throughput["native_decode_scope"] = "engine_committed_decode_walltime_rolling"
            forward = training.get("forward") if isinstance(training.get("forward"), dict) else {}
            forward_epoch = forward.get("telemetry_epoch")
            forward_count = forward.get("forward_input_tokens")
            forward_scope = forward.get("measurement_scope", "global")
            lifecycle = snapshot.get("lifecycle") if isinstance(snapshot.get("lifecycle"), dict) else {}
            source_bound_central = (
                snapshot.get("owned_source_verified") is True
                and lifecycle.get("backend") == "central_gradient"
            )
            try:
                valid_forward_count = (
                    type(forward_count) in (int, float)
                    and math.isfinite(forward_count) and forward_count >= 0
                )
            except OverflowError:
                valid_forward_count = False
            has_forward_counters = isinstance(forward_epoch, str) and bool(forward_epoch) and valid_forward_count
            if source_bound_central and not has_forward_counters:
                # Central GPU executors do backward work, but the Learner watch
                # does not expose their forward counters or scope.
                forward_epoch = None
                forward_count = None
                forward_scope = "not_reported"
            learner_rate = self._rates.sample(
                "learner_forward",
                f"{forward_epoch}:{forward_scope}" if learner_fresh and forward_epoch else None,
                forward_count if learner_fresh else None,
                now,
            )
            throughput["learner_forward_input_tokens_per_second"] = learner_rate
            throughput["learner_forward_scope"] = forward_scope
            snapshot["throughput"] = throughput

            committed = snapshot.get("committed") if isinstance(snapshot.get("committed"), dict) else {}
            for name, scope in (("process", "process"), ("run", "current_process_run")):
                item = committed.get(name) if isinstance(committed.get(name), dict) else {}
                totals = item.get("totals") if isinstance(item.get("totals"), dict) else {}
                live = self._fresh({"live": item.get("live", learner_fresh), "age_seconds": 0})
                if name == "run":
                    raw_run = training.get("run_committed") if isinstance(training.get("run_committed"), dict) else {}
                    epoch = (
                        raw_run.get("telemetry_epoch") or item.get("telemetry_epoch"),
                        raw_run.get("run_id") or item.get("run_id"),
                        raw_run.get("scope") or item.get("scope"),
                    )
                    if not all(value is not None for value in epoch):
                        epoch = None
                    elif epoch is not None:
                        item["telemetry_epoch"] = epoch[0]
                else:
                    epoch = item.get("telemetry_epoch")
                rolling = self._committed_rates.observe(
                    scope, epoch=epoch, counters=totals, now=now,
                    fresh=live and learner_fresh,
                )
                item["rolling"] = rolling
                committed[name] = item
            snapshot["committed"] = committed
            result["snapshot"] = snapshot
            self._append_snapshot(snapshot, result, source_identity, now)
            return result

    def _append_error(self, result: dict[str, Any], source_identity: str) -> None:
        self._samples_seen += 1
        self._history.append({
            "sampled_at": time.time(),
            "source_identity": source_identity,
            "owned_source_identity": None,
            "runtime_identity": {
                "runtime_run_id": None,
                "source_identity": source_identity,
                "telemetry_epoch": None,
            },
            "runtime_run_id": None,
            "expected_runtime_run_id": result.get("expected_runtime_run_id"),
            "ok": False,
            "error": result.get("error"),
            "counters": {},
            "output_tokens_per_second": None,
            "updates_per_hour": None,
            "throughput": {},
            "worker_activity": {},
            "learner_activity": {},
        })

    def _append_snapshot(self, snapshot: dict[str, Any], result: dict[str, Any],
                         source_identity: str, monotonic_now: float) -> None:
        lifecycle = snapshot.get("lifecycle") if isinstance(snapshot.get("lifecycle"), dict) else {}
        committed = snapshot.get("committed") if isinstance(snapshot.get("committed"), dict) else {}
        process = committed.get("process") if isinstance(committed.get("process"), dict) else {}
        run = committed.get("run") if isinstance(committed.get("run"), dict) else {}
        owned_identity = snapshot.get("source_identity")
        if not isinstance(owned_identity, dict):
            owned_identity = None
        runtime_run_id = lifecycle.get("run_id") if isinstance(lifecycle.get("run_id"), str) else None
        counters: dict[str, Any] = {}
        for name, item in (("process", process), ("run", run)):
            counters[name] = {
                "scope": item.get("scope"),
                "telemetry_epoch": item.get("telemetry_epoch"),
                "runtime_run_id": item.get("run_id"),
                "totals": deepcopy(item.get("totals") or {}),
                "rolling": deepcopy(item.get("rolling") or {}),
            }
        worker_counters = {}
        for row in snapshot.get("workers", []):
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            worker_counters[row["id"]] = {
                "live": row.get("live") is True,
                "telemetry_epoch": row.get("telemetry_epoch"),
                "generated_tokens": row.get("generated_tokens"),
                "tokens_per_second": row.get("tokens_per_second"),
                "activity_state": row.get("activity_state"),
            }
        throughput = snapshot.get("throughput") if isinstance(snapshot.get("throughput"), dict) else {}
        rolling = run.get("rolling") if isinstance(run.get("rolling"), dict) else {}
        rates_per_hour = rolling.get("rates_per_hour") if isinstance(rolling.get("rates_per_hour"), dict) else {}

        self._samples_seen += 1
        operation = snapshot.get("execution_telemetry") if isinstance(snapshot.get("execution_telemetry"), dict) else {}
        executor_rates = {}
        for executor in operation.get("executors", []) if isinstance(operation.get("executors"), list) else []:
            if not isinstance(executor, dict) or not isinstance(executor.get("executor_id"), str):
                continue
            measurements = executor.get("measurements") if isinstance(executor.get("measurements"), dict) else {}
            executor_rates[executor["executor_id"]] = {
                "status": executor.get("status"),
                "sample_age_seconds": executor.get("sample_age_seconds"),
                "current_phase": deepcopy(executor.get("current_phase")),
                "phase_elapsed_seconds": executor.get("phase_elapsed_seconds"),
                "measurements": {
                    phase: deepcopy(measurements.get(phase))
                    for phase in ("forward", "backward")
                },
            }

        self._history.append({
            "sampled_at": snapshot.get("sampled_at", time.time()),
            "source_identity": source_identity,
            "owned_source_identity": deepcopy(owned_identity),
            "runtime_identity": {
                "runtime_run_id": runtime_run_id,
                "source_identity": source_identity,
                "owned_source_identity": deepcopy(owned_identity),
                "telemetry_epoch": process.get("telemetry_epoch"),
            },
            "runtime_run_id": runtime_run_id,
            "ok": result.get("ok") is True,
            "error": None if result.get("ok") is True else result.get("error"),
            "counters": {**counters, "workers": worker_counters},
            "native_decode_tokens_per_second": throughput.get("native_decode_tokens_per_second"),
            "series_epoch": repr((self._runtime_identity, tuple((r.get("id"), r.get("telemetry_epoch")) for r in snapshot.get("workers", [])))),
            "output_tokens_per_second": throughput.get("aggregate_output_tokens_per_second"),
            "executor_rates": executor_rates,
            "updates_per_hour": rates_per_hour.get("updates"),
            "throughput": {
                "aggregate_output_tokens_per_second": throughput.get("aggregate_output_tokens_per_second"),
                "learner_forward_input_tokens_per_second": throughput.get("learner_forward_input_tokens_per_second"),
            },
            "worker_activity": deepcopy(snapshot.get("worker_activity") or {}),
            "learner_activity": deepcopy(snapshot.get("learner_activity") or {}),
            "monotonic_sample": monotonic_now,
        })

    def history(self, limit: int = DEFAULT_HISTORY_SAMPLES) -> list[dict[str, Any]]:
        cap = max(1, min(int(limit), self.history_limit))
        with self._lock:
            return deepcopy(list(self._history)[-cap:])

    def history_state(self, limit: int = DEFAULT_HISTORY_SAMPLES) -> dict[str, Any]:
        cap = max(1, min(int(limit), self.history_limit))
        with self._lock:
            samples = deepcopy(list(self._history)[-cap:])
            return {
                "samples": samples,
                "truncated": self._samples_seen > len(samples),
                "total": self._samples_seen,
                "limit": cap,
                "retained": len(self._history),
            }
