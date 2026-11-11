"""Start, inspect, self-test and stop only the processes in one local run manifest."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
from importlib import metadata
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from ..paths import ROOT
from ..rollout.groups import PolicySnapshot
from .config import RuntimeConfig, load_config
from .manifest import (active_manifest_path, process_is_running, process_matches, process_starttime,
                       read_json, write_json_atomic)


class RuntimeErrorDetail(RuntimeError):
    """An expected runtime management failure with a user-safe message."""


_SELFTEST_LOGPROB_THRESHOLDS = {
    "mean_abs_error": 0.01,
    "max_abs_error": 0.1,
    "rms_error": 0.02,
}
_LOGGER = logging.getLogger(__name__)
_EXTERNAL_POLICY_ACK_REFRESH_SECONDS = 15.0


def _validate_selftest_alignment(
    alignment: Any,
    worker_ids: set[str],
    *,
    phase: str,
) -> None:
    """Independently enforce fixed aggregate and per-Worker selftest gates."""
    if not worker_ids:
        raise RuntimeErrorDetail("selftest alignment requires sampled Worker identities")
    if (not isinstance(alignment, dict)
            or not isinstance(alignment.get("status"), str)
            or alignment["status"] not in {"passed", "warning"}):
        raise RuntimeErrorDetail(f"selftest lacks a valid {phase} logprob alignment status")
    if alignment.get("thresholds") != _SELFTEST_LOGPROB_THRESHOLDS:
        raise RuntimeErrorDetail(f"selftest {phase} alignment did not retain the fixed logprob thresholds")

    def validate_metrics(
        metrics: Any, label: str, *, check_reported_status: bool = True,
    ) -> tuple[str, list[str]]:
        if not isinstance(metrics, dict):
            raise RuntimeErrorDetail(f"selftest {phase} alignment omitted {label} metrics")
        tokens = metrics.get("tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise RuntimeErrorDetail(f"selftest {phase} alignment has invalid {label} token count")
        exceeded: list[str] = []
        for metric, limit in _SELFTEST_LOGPROB_THRESHOLDS.items():
            value = metrics.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeErrorDetail(
                    f"selftest {phase} alignment has invalid {label} {metric} metric"
                )
            try:
                numeric_value = float(value)
            except OverflowError:
                numeric_value = math.inf
            if not math.isfinite(numeric_value) or numeric_value < 0.0:
                raise RuntimeErrorDetail(
                    f"selftest {phase} alignment has invalid {label} {metric} metric"
                )
            if numeric_value > limit:
                exceeded.append(metric)
        expected_status = "warning" if exceeded else "passed"
        if check_reported_status and metrics.get("status") != expected_status:
            raise RuntimeErrorDetail(
                f"selftest {phase} {label} alignment status does not match its measured metrics"
            )
        return expected_status, exceeded

    aggregate_status, aggregate_warnings = validate_metrics(
        alignment, "aggregate", check_reported_status=False,
    )
    by_worker = alignment.get("by_worker")
    if not isinstance(by_worker, dict) or set(by_worker) != worker_ids:
        raise RuntimeErrorDetail(f"selftest {phase} alignment must contain exactly the sampled Workers")
    warning_workers: dict[str, list[str]] = {}
    for worker_id in sorted(worker_ids):
        worker_metrics = by_worker[worker_id]
        status, exceeded = validate_metrics(worker_metrics, f"Worker {worker_id}")
        if status == "warning":
            warning_workers[worker_id] = exceeded
    expected_status = (
        "warning" if aggregate_status == "warning" or warning_workers else "passed"
    )
    if alignment.get("status") != expected_status:
        raise RuntimeErrorDetail(
            f"selftest {phase} aggregate alignment status does not match its measured metrics"
        )
    if expected_status == "warning":
        _LOGGER.warning(
            "selftest %s logprob alignment exceeded engineering tolerance; continuing: "
            "aggregate_metrics=%s worker_metrics=%s",
            phase,
            aggregate_warnings,
            warning_workers,
        )


def validate_paths(config: RuntimeConfig, *, role: str = "cluster") -> None:
    """Validate only the files this process role needs to serve its contract."""
    worker_model = Path(config.worker_inference.model_path).expanduser()
    required_by_role = {
        "cluster": {"model": config.model, "dataset": config.dataset,
                    "ARC contract": config.contract, "Worker model": worker_model},
        "learner": {"model": config.model, "dataset": config.dataset,
                    "ARC contract": config.contract},
        "worker": {"Worker model": worker_model},
        "teacher": {},
    }
    if role not in required_by_role:
        raise ValueError(f"unknown runtime role for path validation: {role}")
    required = required_by_role[role]
    if config.quantization.mode == "torchao" and role in {"cluster", "learner"}:
        required = {**required, "native compute interpreter": Path(config.learner.compute_python)}
        for index, entry in enumerate(config.learner.compute_pythonpath.split(os.pathsep)):
            if not entry:
                raise RuntimeErrorDetail("native compute_pythonpath contains an empty entry")
            required[f"native compute module path {index}"] = Path(entry)
    if config.evaluation.enabled and role in {'cluster', 'learner'}:
        required = {**required,
                    'evaluation manifest': Path(config.evaluation.manifest_path),
                    'SFT exclusion proof': Path(config.evaluation.sft_exclusion_proof)}
        if config.evaluation.initial_count_proof:
            required['evaluation count proof'] = Path(config.evaluation.initial_count_proof)
    for label, path in required.items():
        if not path.exists():
            raise RuntimeErrorDetail(f"missing {label} path: {path}")


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("ARC_ENV_TOKEN")
    if not token:
        raise RuntimeErrorDetail("ARC_ENV_TOKEN must be set in the container environment")
    return {"Authorization": f"Bearer {token}"}


def validate_environment(config: RuntimeConfig | None = None) -> None:
    """Fail before loading a model when runtime authentication is not configured."""
    _auth_headers()
    if config is not None:
        validate_worker_environment(config)


def _worker_headers(worker, *, require_epoch: bool = False) -> dict[str, str]:
    token = os.environ.get(worker.auth_token_env)
    if not token:
        raise RuntimeErrorDetail(f"{worker.auth_token_env} must be set for Worker {worker.id} access")
    if worker.scope in {"gateway", "service"} and token == os.environ.get("ARC_ENV_TOKEN"):
        raise RuntimeErrorDetail(
            f"External Worker credential for Worker {worker.id} must differ from ARC_ENV_TOKEN"
        )
    headers = {"Authorization": f"Bearer {token}"}
    if worker.scope in {"gateway", "service"}:
        if worker.boot_id and worker.telemetry_epoch:
            headers.update({
                "X-Expected-Worker-Boot-ID": worker.boot_id,
                "X-Expected-Worker-Telemetry-Epoch": worker.telemetry_epoch,
            })
        elif require_epoch:
            raise RuntimeErrorDetail(
                f"external Worker {worker.id} has no pinned boot and telemetry identity; resolve a fresh active snapshot first"
            )
    return headers


def validate_worker_environment(config: RuntimeConfig) -> None:
    """Check external Worker bearer variables without logging their values."""
    arc_token = os.environ.get("ARC_ENV_TOKEN")
    names = {worker.auth_token_env for worker in config.workers
             if worker.scope in {"gateway", "service"}}
    missing = sorted(name for name in names if not os.environ.get(name))
    if missing:
        raise RuntimeErrorDetail("Worker service environment variables are missing: " + ", ".join(missing))
    if arc_token and any(
            worker.auth_token_env != "ARC_ENV_TOKEN" and os.environ[worker.auth_token_env] == arc_token
            for worker in config.workers if worker.scope in {"gateway", "service"}):
        raise RuntimeErrorDetail("Gateway and service Worker credentials must be distinct from ARC_ENV_TOKEN")


def _worker_request(worker, method: str, path: str, *, timeout: float, **kwargs) -> httpx.Response:
    if path.startswith(("http://", "https://")) or "?" in path or "#" in path:
        raise ValueError("Worker route must be a relative path without query parameters")
    url = f"{worker.url}/{path.lstrip('/')}"
    # The first direct-service health read establishes the process telemetry
    # epoch. It still checks the configured host boot pin and response identity;
    # every other external request requires both pins before it is sent.
    require_epoch = not (
        worker.scope == "service" and path.lstrip("/") == "health"
        and not worker.telemetry_epoch
    )
    try:
        response = httpx.request(
            method, url, headers=_worker_headers(worker, require_epoch=require_epoch), timeout=timeout, **kwargs,
        )
    except httpx.HTTPError as exc:
        raise RuntimeErrorDetail(
            f"HTTP {method} Worker {worker.id} failed ({type(exc).__name__})"
        ) from exc
    if worker.scope == "gateway" and response.is_success:
        expected = {
            "X-Worker-Node-ID": worker.node_id,
            "X-Worker-Boot-ID": worker.boot_id,
            "X-Worker-Telemetry-Epoch": worker.telemetry_epoch,
        }
        if not all(response.headers.get(name) == value for name, value in expected.items()):
            raise RuntimeErrorDetail(
                f"Gateway response identity for Worker {worker.id} differs from the active membership snapshot"
            )
    elif worker.scope == "service" and response.is_success:
        node_id = response.headers.get("X-Worker-Node-ID")
        boot_id = response.headers.get("X-Worker-Boot-ID")
        telemetry_epoch = response.headers.get("X-Worker-Telemetry-Epoch")
        if (node_id != worker.node_id or not boot_id or not telemetry_epoch
                or (worker.boot_id and boot_id != worker.boot_id)
                or (worker.telemetry_epoch and telemetry_epoch != worker.telemetry_epoch)):
            raise RuntimeErrorDetail(
                f"direct service identity for Worker {worker.id} differs from its host/epoch snapshot"
            )
    return response


def _request(method: str, url: str, *, timeout: float, **kwargs) -> httpx.Response:
    try:
        return httpx.request(method, url, headers=_auth_headers(), timeout=timeout, **kwargs)
    except httpx.HTTPError as exc:
        raise RuntimeErrorDetail(f"HTTP {method} {url} failed ({type(exc).__name__})") from exc


def _health(config: RuntimeConfig, role: str, worker_id: str | None = None) -> dict[str, Any] | None:
    if role == "learner":
        url = config.learner_url
        request = lambda: _request("GET", f"{url}/health", timeout=3.0)
    else:
        worker = next((item for item in config.workers if item.id == worker_id), None)
        if worker is None:
            return None
        request = lambda: _worker_request(worker, "GET", "health", timeout=3.0)
    try:
        response = request()
        if response.status_code != 200:
            return {"http_status": response.status_code, "ready": False}
        payload = response.json()
        return payload if isinstance(payload, dict) else {"ready": False, "error": "invalid_json"}
    except (RuntimeErrorDetail, ValueError):
        return None


def _valid_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _external_worker_snapshot(config: RuntimeConfig) -> tuple[RuntimeConfig, dict[str, Any]]:
    """Freeze direct service health and Gateway membership before Learner startup.

    Direct service Workers are admitted from their own health/identity response;
    Gateway Workers are admitted from the site-scoped model-ready inventory.
    The resulting effective config pins node boot and Worker process epochs for
    every subsequent RPC. Only ``scope=local`` Workers are manager children.
    """
    deployed = list(config.workers)
    active = [worker for worker in deployed if worker.scope == "local"]
    active_rows = [
        {"worker_id": worker.id, "scope": "local", "site_id": worker.site_id,
         "node_id": worker.node_id, "gpu": worker.gpu, "port": worker.port,
         "active": True, "reason": "locally_managed"}
        for worker in active
    ]
    excluded: list[dict[str, str]] = []
    for worker in deployed:
        if worker.scope != "service":
            continue
        reason = None
        token = os.environ.get(worker.auth_token_env)
        if not token:
            reason = "service_token_unavailable"
        elif token == os.environ.get("ARC_ENV_TOKEN"):
            reason = "service_token_equals_central_arc_token"
        health = _health(config, "worker", worker.id) if reason is None else None
        if reason is None and (not health or not _is_runtime_worker(
                health, worker.id, config.quantization.mode, _config_artifact_sha256(config))):
            reason = "service_worker_not_ready_or_invalid"
        if reason is None:
            boot_id = health.get("boot_id")
            telemetry_epoch = health.get("telemetry_epoch")
            artifact_sha = health.get("artifact_fingerprint")
            base_sha = health.get("base_fingerprint")
            deployment = health.get("deployment")
            if not isinstance(boot_id, str) or not boot_id.strip():
                reason = "service_boot_id_unknown"
            elif worker.boot_id != boot_id:
                reason = "service_host_boot_id_changed"
            elif not isinstance(telemetry_epoch, str) or not telemetry_epoch.strip():
                reason = "service_telemetry_epoch_unknown"
            elif not _valid_sha256(artifact_sha):
                reason = "service_artifact_fingerprint_unknown"
            elif not _valid_sha256(base_sha):
                reason = "service_base_fingerprint_unknown"
            elif worker.model_sha256 and worker.model_sha256 != artifact_sha:
                reason = "service_artifact_fingerprint_mismatch"
            elif (not isinstance(deployment, dict)
                  or deployment.get("scope") != "service"
                  or deployment.get("node_id") != worker.node_id
                  or deployment.get("site_id") != worker.site_id):
                reason = "service_node_identity_mismatch"
        if reason:
            excluded.append({"worker_id": worker.id, "reason": reason})
            continue
        pinned = replace(
            worker, boot_id=boot_id, telemetry_epoch=telemetry_epoch,
            model_sha256=artifact_sha,
        )
        active.append(pinned)
        active_rows.append({
            "worker_id": worker.id,
            "scope": "service",
            "site_id": worker.site_id,
            "node_id": worker.node_id,
            "endpoint": worker.endpoint,
            "base_path": "",
            "gpu": worker.gpu,
            "port": worker.port,
            "boot_id": boot_id,
            "telemetry_epoch": telemetry_epoch,
            "model_sha256": artifact_sha,
            "base_fingerprint": base_sha,
            "model_ready": True,
            "eligible_at_snapshot": health.get("policy_ready") is True,
            "active": True,
            "reason": "direct_service_health_snapshot",
        })

    gateway_groups: dict[tuple[str, str, str], list[Any]] = {}
    for worker in deployed:
        if worker.scope == "gateway":
            gateway_groups.setdefault(
                (worker.site_id, worker.endpoint.rstrip("/"), worker.auth_token_env), []
            ).append(worker)

    for (site_id, endpoint, token_env), workers in gateway_groups.items():
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeErrorDetail(f"Gateway bearer environment variable {token_env} is missing")
        if token == os.environ.get("ARC_ENV_TOKEN"):
            raise RuntimeErrorDetail(f"Gateway bearer {token_env} must differ from ARC_ENV_TOKEN")
        try:
            response = httpx.get(
                f"{endpoint}/v1/workers?readiness=model",
                headers={"Authorization": f"Bearer {token}"},
                timeout=min(10.0, config.learner.request_timeout),
            )
        except httpx.HTTPError as exc:
            for worker in workers:
                excluded.append({"worker_id": worker.id, "reason": f"gateway_unavailable:{type(exc).__name__}"})
            continue
        if response.status_code != 200:
            for worker in workers:
                excluded.append({"worker_id": worker.id, "reason": f"gateway_http_{response.status_code}"})
            continue
        try:
            payload = response.json()
        except ValueError:
            for worker in workers:
                excluded.append({"worker_id": worker.id, "reason": "gateway_invalid_json"})
            continue
        rows = payload.get("workers") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            for worker in workers:
                excluded.append({"worker_id": worker.id, "reason": "gateway_workers_missing"})
            continue
        by_id: dict[str, dict[str, Any]] = {}
        duplicate_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("worker_id"), str):
                continue
            worker_id = row["worker_id"]
            if worker_id in by_id:
                duplicate_ids.add(worker_id)
            by_id[worker_id] = row
        if duplicate_ids:
            raise RuntimeErrorDetail(
                f"Gateway site {site_id} returned duplicate Worker IDs: {sorted(duplicate_ids)}"
            )
        for worker in workers:
            row = by_id.get(worker.id)
            if row is None:
                excluded.append({"worker_id": worker.id, "reason": "not_in_eligible_gateway_snapshot"})
                continue
            if any(name in row for name in ("worker_token", "node_token", "gateway_bearer", "token")):
                raise RuntimeErrorDetail(f"Gateway site {site_id} exposed a credential in its Worker list")
            health = row.get("worker_health")
            reason = None
            if row.get("node_id") != worker.node_id or row.get("worker_id") != worker.id:
                reason = "registered_identity_mismatch"
            elif row.get("site_id") != worker.site_id or site_id != worker.site_id:
                reason = "site_scope_mismatch"
            elif type(row.get("eligible")) is not bool:
                reason = "generation_eligibility_unknown"
            elif row.get("model_ready") is not True:
                reason = "controller_model_not_ready"
            elif not isinstance(row.get("boot_id"), str) or not row["boot_id"]:
                reason = "boot_id_unknown"
            elif row.get("base_path") != worker.url_prefix:
                reason = "gateway_route_prefix_mismatch"
            elif not isinstance(health, dict) or not _is_vllm_worker(
                    health, worker.id, config.quantization.mode):
                reason = "worker_health_not_ready_or_invalid"
            elif not _valid_sha256(health.get("artifact_fingerprint")):
                reason = "worker_model_fingerprint_unknown"
            elif not _valid_sha256(health.get("base_fingerprint")):
                reason = "worker_base_fingerprint_unknown"
            elif worker.model_sha256 and health.get("artifact_fingerprint") != worker.model_sha256:
                reason = "configured_model_fingerprint_mismatch"
            elif not isinstance(health.get("telemetry_epoch"), str) or not health["telemetry_epoch"]:
                reason = "worker_epoch_unknown"
            else:
                gpus = row.get("gpus")
                indices = {
                    item.get("index") for item in gpus
                    if isinstance(item, dict) and type(item.get("index")) is int and item["index"] >= 0
                } if isinstance(gpus, list) else set()
                if worker.gpu not in indices:
                    reason = "configured_gpu_not_registered_on_node"
            if reason:
                excluded.append({"worker_id": worker.id, "reason": reason})
                continue
            pinned = replace(
                worker, boot_id=row["boot_id"], telemetry_epoch=health["telemetry_epoch"],
                model_sha256=health["artifact_fingerprint"],
            )
            active.append(pinned)
            active_rows.append({
                "worker_id": worker.id,
                "scope": "gateway",
                "site_id": worker.site_id,
                "node_id": worker.node_id,
                "endpoint": worker.endpoint,
                "base_path": worker.url_prefix,
                "gpu": worker.gpu,
                "port": worker.port,
                "boot_id": row["boot_id"],
                "telemetry_epoch": health["telemetry_epoch"],
                "model_sha256": health["artifact_fingerprint"],
                "base_fingerprint": health["base_fingerprint"],
                "model_ready": row["model_ready"],
                "eligible_at_snapshot": row["eligible"],
                "active": True,
                "reason": "model_ready_gateway_snapshot",
            })

    if not active:
        detail = ", ".join(f"{row['worker_id']}={row['reason']}" for row in excluded)
        raise RuntimeErrorDetail(
            "active membership snapshot contains no ready configured Workers"
            + (f" ({detail})" if detail else "")
        )
    snapshot_id = uuid.uuid4().hex
    receipt = {
        "snapshot_id": snapshot_id,
        "created_at": time.time(),
        "phase": "startup_policy_sync",
        "deployed_workers": [
            {"worker_id": worker.id, "scope": worker.scope, "site_id": worker.site_id,
             "node_id": worker.node_id, "endpoint": worker.endpoint or worker.url,
             "url_prefix": worker.url_prefix, "auth_token_env": worker.auth_token_env,
             "gpu": worker.gpu, "port": worker.port}
            for worker in deployed
        ],
        "startup_workers": active_rows,
        "active_workers": [],
        "excluded_workers": excluded,
    }
    return replace(config, workers=tuple(active)), receipt


def _wait_until(predicate, timeout: float, description: str, interval: float = 1.0) -> Any:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
    raise RuntimeErrorDetail(f"timed out waiting for {description}")


def _valid_quantization(value: Any, expected_mode: str | None = None) -> bool:
    from ..models.quantization import quantization_evidence
    try:
        quantization_evidence(value, expected_mode=expected_mode)
    except ValueError:
        return False
    return True


def _t2_quantization_matches(value: Any, expected_artifact_sha256: str | None) -> bool:
    if not isinstance(value, dict) or not isinstance(expected_artifact_sha256, str) \
            or len(expected_artifact_sha256) != 64:
        return False
    from ..contracts.ternary import TERNARY_QG_LAYOUT_VERSION

    profile = (value.get("lora_input_basis"), value.get("t2_object_count"),
               value.get("t2_logical_parameter_count"))
    gdn_enabled = value.get("gdn_projections", False)
    if type(gdn_enabled) is not bool:
        return False
    profiles = ({("per_use", 323, 515)} if gdn_enabled else
                {("prism_rotated", 323, 515), ("per_use", 322, 514)})
    return (
        value.get("mode") == "ternary"
        and value.get("bits") == 2 and not isinstance(value.get("bits"), bool)
        and value.get("packed_resident") is True
        and value.get("compute_dtype") == "bfloat16"
        and value.get("format") == "t2_g128_fp16"
        and value.get("layout") == "row_split_k128_v1"
        and value.get("group_size_k") == 128
        and profile in profiles
        and value.get("qg_layout") == TERNARY_QG_LAYOUT_VERSION
        and value.get("artifact_sha256") == expected_artifact_sha256
    )


def _is_quantized_learner(
    health: dict[str, Any], expected_mode: str | None = None,
    expected_artifact_sha256: str | None = None,
) -> bool:
    if health.get("ready") is not True or health.get("role") != "learner":
        return False
    if expected_mode == "ternary":
        quantization = health.get("quantization")
        evidence = health.get("evidence")
        if not isinstance(quantization, dict) or not isinstance(evidence, dict):
            return False
        expected_sha = expected_artifact_sha256
        artifact_sha = quantization.get("artifact_sha256")
        adapter_count = evidence.get("adapter_trainable_parameters")
        # Native T2 reports the verified source object inventory in quantization;
        # it does not claim a separate runtime tensor count.
        return (
            evidence.get("quantization") == quantization
            and _t2_quantization_matches(quantization, expected_sha)
            and artifact_sha == expected_sha
            and health.get("artifact_fingerprint") == expected_sha
            and evidence.get("base_fingerprint") == expected_sha
            and evidence.get("artifact_fingerprint") == expected_sha
            and evidence.get("is_loaded_in_4bit") is False
            and evidence.get("base_frozen") is True
            and evidence.get("base_trainable_parameters") == 0
            and isinstance(adapter_count, int) and not isinstance(adapter_count, bool) and adapter_count > 0
        )
    from ..models.quantization import quantization_evidence
    try:
        quantization_evidence(health.get("quantization"), expected_mode=expected_mode)
    except ValueError:
        return False
    evidence = health.get("evidence")
    if (not isinstance(evidence, dict) or evidence.get("is_loaded_in_4bit") is not True
            or evidence.get("base_trainable_parameters") != 0):
        return False
    count = evidence.get("packed_module_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return False
    return True


def _learner_recovery_is_clean(health: dict[str, Any]) -> bool:
    """Require a clean full-state snapshot at the current policy version."""
    recovery = health.get("recovery")
    if not isinstance(recovery, dict) or recovery.get("dirty") is not False:
        return False
    current = health.get("policy_version")
    durable = recovery.get("durable_policy_version")
    return (isinstance(current, int) and not isinstance(current, bool)
            and isinstance(durable, int) and not isinstance(durable, bool)
            and current == durable)


def _is_vllm_worker(health: dict[str, Any], worker_id: str, expected_mode: str | None = None) -> bool:
    quantization = health.get("quantization")
    count = health.get("module_count")
    return (
        health.get("ready") is True and health.get("role") == "worker"
        and health.get("worker_id") == worker_id and health.get("backend") == "vllm"
        and health.get("is_loaded_in_4bit") is True
        and health.get("base_trainable_parameters") == 0
        and isinstance(quantization, dict) and quantization.get("bits") == 4
        and _valid_quantization(quantization, expected_mode)
        and not isinstance(count, bool) and isinstance(count, int) and count > 0
    )


def _is_ninfer_ternary_worker(
    health: dict[str, Any], worker_id: str, expected_artifact_sha256: str | None,
) -> bool:
    try:
        from ..worker.native_artifact import validate_ternary_runtime_profile

        validate_ternary_runtime_profile(health)
    except (ImportError, TypeError, ValueError):
        return False
    if (health.get("ready") is not True or health.get("role") != "worker"
            or health.get("worker_id") != worker_id or health.get("backend") != "ninfer"
            or health.get("quantization_mode") != "ternary"
            or health.get("is_loaded_in_4bit") is not False):
        return False
    if not isinstance(expected_artifact_sha256, str) or len(expected_artifact_sha256) != 64:
        return False
    quantization = health.get("quantization")
    runtime_quantization = health.get("runtime_quantization")
    if not isinstance(quantization, dict):
        return False
    if (quantization.get("mode") != "ternary" or quantization.get("bits") != 2
            or isinstance(quantization.get("bits"), bool)
            or quantization.get("packed_resident") is not True
            or quantization.get("compute_dtype") != "bfloat16"
            or quantization.get("format") != "t2_g128_fp16"
            or quantization.get("layout") != "row_split_k128_v1"
            or quantization.get("group_size_k") != 128
            or quantization.get("artifact_sha256") != expected_artifact_sha256):
        return False
    try:
        from ..worker.native_artifact import validate_ternary_runtime_quantization

        validate_ternary_runtime_quantization(runtime_quantization)
    except (ImportError, ValueError):
        return False
    tensor_count = health.get("quantized_tensor_count")
    return (
        health.get("ninfer_artifact_sha256") == expected_artifact_sha256
        and health.get("base_fingerprint") == expected_artifact_sha256
        and health.get("artifact_fingerprint") == expected_artifact_sha256
        and health.get("base_trainable_parameters") == 0
        and isinstance(tensor_count, int) and not isinstance(tensor_count, bool) and tensor_count > 0
    )


def _is_runtime_worker(
    health: dict[str, Any], worker_id: str, expected_mode: str | None = None,
    expected_artifact_sha256: str | None = None,
) -> bool:
    if expected_mode == "ternary":
        return _is_ninfer_ternary_worker(health, worker_id, expected_artifact_sha256)
    return _is_vllm_worker(health, worker_id, expected_mode)


def _config_artifact_sha256(config: RuntimeConfig) -> str | None:
    identity = config.artifact_identity
    if config.model_backend != "ternary" or not isinstance(identity, dict):
        return None
    source = identity.get("source")
    artifact = source.get("ninfer_artifact") if isinstance(source, dict) else None
    value = artifact.get("sha256") if isinstance(artifact, dict) else None
    return value if isinstance(value, str) else None


def _run_location(config: RuntimeConfig, run_id: str) -> tuple[Path, Path]:
    run_root = config.run
    run_dir = run_root / "runs" / run_id
    return run_root, run_dir / "manifest.json"


@contextmanager
def _manager_lock(run_root: Path):
    """Serialize runtime selection and startup for one configured output root."""
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / ".runtime-manager.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _config_compatibility_sha256(config: RuntimeConfig) -> str:
    """Hash the source config semantically, excluding manager-injected recovery fields."""
    data = config.as_dict()
    if config.model_backend != "ternary":
        # Keep existing Q4/NVFP4 profile hashes stable for running 9B users.
        for key in ("model_backend", "artifact_identity_path", "artifact_identity",
                    "tokenizer_path", "recovery_semantics"):
            data.pop(key, None)
    worker_model_path = Path(config.worker_inference.model_path).expanduser()
    if not worker_model_path.is_absolute():
        worker_model_path = ROOT / worker_model_path
    data["model_path"] = str(config.model)
    data["dataset_dir"] = str(config.dataset)
    data["run_dir"] = str(config.run)
    data["worker_inference"]["model_path"] = str(worker_model_path.resolve())
    data["arc"]["contract_path"] = str(config.contract)
    for key in ("recovery_dir", "recovery_mode", "recovery_lineage_id"):
        data["learner"].pop(key, None)
    state_dir = config.bepa.state_dir
    if state_dir:
        path = Path(state_dir).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        data["bepa"]["state_dir"] = str(path.resolve())
    else:
        data["bepa"]["state_dir"] = str((config.run / "bepa").resolve())
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_fingerprint() -> str:
    """Fingerprint the source tree that child roles will import at launch."""
    source_root = ROOT / "qwen_rl"
    paths = [path for path in source_root.rglob("*.py")
             if path.is_file() and not path.is_symlink()]
    pyproject = ROOT / "pyproject.toml"
    if pyproject.is_file() and not pyproject.is_symlink():
        paths.append(pyproject)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _stop_instruction(config_path: Path) -> str:
    return ("stop the owned runtime with `./bin/qwen-rl runtime stop` (or `python -m "
            f"qwen_rl.runtime.cli stop --config {config_path}` inside the runtime environment), then retry")


def _manifest_config_matches(
    manifest: dict[str, Any], config: RuntimeConfig, config_path: Path,
) -> bool:
    current_hash = _config_compatibility_sha256(config)
    recorded_hash = manifest.get("config_compatibility_sha256")
    if isinstance(recorded_hash, str):
        return recorded_hash == current_hash
    # Existing active manifests predate the semantic hash. Reuse those only when the
    # exact source path and bytes still match; do not infer compatibility from code.
    try:
        raw_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError:
        return False
    return (manifest.get("config") == str(config_path)
            and manifest.get("config_sha256") == raw_hash)


def _expected_runtime_roles(config: RuntimeConfig, manifest: dict[str, Any]) -> set[tuple[str, str | None]]:
    # The central manager owns only local child processes. Service and Gateway
    # Workers are health checked and policy-synced, but are never spawned or
    # signalled from this host.
    expected = {("learner", None), *(('worker', worker.id) for worker in config.workers
                                      if worker.scope == "local")}
    teacher = manifest.get("endpoints", {}).get("teacher")
    teacher_id = teacher.get("worker_id") if isinstance(teacher, dict) else None
    if isinstance(teacher_id, str) and teacher_id:
        expected.add(("teacher", teacher_id))
    return expected


def _reuse_existing_runtime(
    config: RuntimeConfig,
    config_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    fresh: bool,
) -> dict[str, Any] | None:
    records = manifest.get("processes", [])
    if not isinstance(records, list):
        records = []
    valid_records = [record for record in records if isinstance(record, dict)]
    live = [record for record in valid_records if _owned_process_matches(record, manifest)]
    mismatched = [record for record in valid_records
                  if (isinstance(record.get("pid"), int)
                      and not isinstance(record.get("pid"), bool)
                      and process_is_running(record["pid"])
                      and not _owned_process_matches(record, manifest))]
    if mismatched:
        raise RuntimeErrorDetail(
            f"runtime {manifest.get('run_id')} contains a live PID that no longer matches its "
            "registered identity; no process was signalled. Inspect `qwen-rl status` and clear "
            "the port owner explicitly before retrying."
        )
    if not live:
        return None
    run_id = manifest.get("run_id")
    stop = _stop_instruction(config_path)
    if fresh:
        raise RuntimeErrorDetail(
            f"runtime {run_id} still has owned live roles; `--fresh` requires stopping them first; "
            f"{stop}. No process was stopped."
        )
    if not _manifest_config_matches(manifest, config, config_path):
        raise RuntimeErrorDetail(
            f"runtime {run_id} is active with a different configuration; {stop}. "
            "The existing roles were left running."
        )

    expected = _expected_runtime_roles(config, manifest)
    if not expected:
        raise RuntimeErrorDetail(f"runtime {run_id} has an invalid role manifest; {stop}.")

    unhealthy: list[str] = []
    for record in live:
        role, worker_id = record.get("role"), record.get("worker_id")
        health = (_health(config, "learner") if role == "learner"
                  else _health(config, "worker", worker_id) if role == "worker" else None)
        if role == "learner":
            if not health or not _is_quantized_learner(health, config.quantization.mode, _config_artifact_sha256(config)):
                unhealthy.append("Learner")
            elif not _learner_recovery_is_clean(health):
                unhealthy.append("Learner (recovery state is not durable)")
        elif role == "worker" and (
                not isinstance(worker_id, str) or not health
                or not _is_runtime_worker(health, worker_id, config.quantization.mode, _config_artifact_sha256(config))):
            unhealthy.append(f"Worker {worker_id}")
        elif role == "teacher" and not _owned_process_matches(record, manifest):
            unhealthy.append("Teacher")
    live_roles = [(record.get("role"), record.get("worker_id")) for record in live]
    is_complete = (set(live_roles) == expected and len(live_roles) == len(expected)
                   and not unhealthy)
    remote_ok, remote_errors = _active_remote_health(config, manifest)
    if not remote_ok:
        raise RuntimeErrorDetail(
            f"runtime {run_id} has an unavailable or changed active external Worker membership "
            f"({', '.join(remote_errors)}); no processes were signalled"
        )
    if is_complete:
        return {
            "run_id": run_id,
            "status": manifest.get("status", "ready"),
            "manifest": str(manifest_path),
            "selftest": manifest.get("selftest"),
            "reused": True,
            "source_fingerprint_at_launch": manifest.get("source_fingerprint_at_launch"),
            "code_hot_reloaded": False,
            "recovery": manifest.get("recovery"),
        }

    if not _manifest_config_matches(manifest, config, config_path):
        raise RuntimeErrorDetail(
            f"runtime {run_id} is partial and its configuration differs; {stop}. "
            "The existing roles were left running."
        )

    # A missing role endpoint that still answers is unowned from this manifest's
    # perspective. Leave it untouched and refuse to create a port collision.
    present_roles = set(live_roles)
    for role, worker_id in expected - present_roles:
        health = (_health(config, "learner") if role == "learner"
                  else _health(config, "worker", worker_id) if role == "worker" else None)
        if health is not None:
            label = "Learner" if role == "learner" else f"Worker {worker_id}"
            raise RuntimeErrorDetail(
                f"runtime {run_id} is partial and {label}'s endpoint is not owned by its live "
                f"manifest processes; no process was signalled and {stop}"
            )

    recovery = _inspect_training_recovery(config)
    if recovery.get("status") != "resume":
        raise _recovery_gate_error(recovery, partial=True)
    validate_paths(config, role="cluster")
    if config.bepa.enabled:
        validate_bepa_environment(config)
    _stop_owned_roles_for_recovery(manifest, manifest_path, live)
    return None


def _reject_unmanaged_runtime(config: RuntimeConfig) -> None:
    """Prevent a managed start from colliding with standalone Learner/Worker roles."""
    health = _health(config, "learner")
    if health is not None:
        raise RuntimeErrorDetail(
            f"the Learner endpoint {config.learner_url} responds but is not owned by the active "
            "manager manifest; stop its owning process or clear the port explicitly before starting. "
            "No process was signalled."
        )
    for worker in config.workers:
        if worker.scope != "local":
            continue
        health = _health(config, "worker", worker.id)
        if health is not None:
            raise RuntimeErrorDetail(
                f"Worker {worker.id} endpoint {worker.url} responds but is not owned by the active "
                "manager manifest; stop its owning process or clear the port explicitly before starting. "
                "No process was signalled."
            )


def _active_remote_health(
    config: RuntimeConfig, manifest: dict[str, Any],
) -> tuple[bool, list[str]]:
    membership = manifest.get("membership")
    rows = membership.get("active_workers") if isinstance(membership, dict) else None
    if not isinstance(rows, list):
        if not any(worker.scope in {"gateway", "service"} for worker in config.workers):
            return True, []
        return False, ["membership_snapshot_missing"]
    configured_remote = {worker.id: worker for worker in config.workers
                         if worker.scope in {"gateway", "service"}}
    if configured_remote and membership.get("phase") != "active":
        return False, ["membership_policy_ack_pending"]
    errors: list[str] = []
    seen_remote: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("scope") not in {"gateway", "service"}:
            continue
        worker = configured_remote.get(row.get("worker_id"))
        if worker is None or worker.scope != row.get("scope"):
            errors.append(str(row.get("worker_id") or "unknown_worker") + ":not_configured")
            continue
        seen_remote.add(worker.id)
        if (row.get("node_id") != worker.node_id or row.get("site_id") != worker.site_id
                or (worker.model_sha256 and row.get("model_sha256") != worker.model_sha256)):
            errors.append(f"{worker.id}:membership_identity_changed")
            continue
        boot_id = row.get("boot_id")
        telemetry_epoch = row.get("telemetry_epoch")
        if (not isinstance(boot_id, str) or not boot_id
                or not isinstance(telemetry_epoch, str) or not telemetry_epoch):
            errors.append(f"{worker.id}:membership_epoch_missing")
            continue
        if worker.scope == "service" and worker.boot_id != boot_id:
            errors.append(f"{worker.id}:host_boot_changed")
            continue
        pinned = replace(
            worker,
            boot_id=boot_id,
            telemetry_epoch=telemetry_epoch,
            model_sha256=str(row.get("model_sha256") or ""),
        )
        health = _health(replace(config, workers=(pinned,)), "worker", pinned.id)
        if (not health or not _is_runtime_worker(
                health, pinned.id, config.quantization.mode, _config_artifact_sha256(config))
                or health.get("boot_id") != pinned.boot_id
                or health.get("telemetry_epoch") != pinned.telemetry_epoch
                or health.get("artifact_fingerprint") != row.get("model_sha256")
                or health.get("base_fingerprint") != row.get("base_fingerprint")):
            errors.append(f"{pinned.id}:unavailable_or_epoch_changed")
            continue
        if worker.scope == "service":
            deployment = health.get("deployment")
            if (not isinstance(deployment, dict) or deployment.get("scope") != "service"
                    or deployment.get("node_id") != worker.node_id
                    or deployment.get("site_id") != worker.site_id):
                errors.append(f"{pinned.id}:deployment_identity_changed")
    deployed = membership.get("deployed_workers")
    excluded = membership.get("excluded_workers")
    if isinstance(deployed, list) and isinstance(excluded, list):
        expected_remote = {
            row.get("worker_id") for row in deployed
            if isinstance(row, dict) and row.get("scope") in {"gateway", "service"}
        } - {
            row.get("worker_id") for row in excluded
            if isinstance(row, dict) and isinstance(row.get("worker_id"), str)
        }
    else:
        expected_remote = seen_remote
    if expected_remote != seen_remote:
        errors.extend(f"{worker_id}:membership_missing"
                      for worker_id in sorted(expected_remote - seen_remote))
    return not errors, errors


def _recovery_config(config: RuntimeConfig) -> dict[str, Any]:
    data = config.as_dict()
    data["model_path"] = str(config.model)
    data["dataset_dir"] = str(config.dataset)
    data["run_dir"] = str(config.run)
    data["arc"]["contract_path"] = str(config.contract)
    state_dir = config.bepa.state_dir
    if state_dir:
        state_path = Path(state_dir).expanduser()
        if not state_path.is_absolute():
            state_path = ROOT / state_path
        data["bepa"]["state_dir"] = str(state_path.resolve())
    else:
        data["bepa"]["state_dir"] = str((config.run / "bepa").resolve())
    data["learner"]["recovery_dir"] = str((config.run / "training").resolve())
    data["learner"]["recovery_mode"] = "auto"
    data["learner"]["recovery_lineage_id"] = None
    return data


def _effective_runtime_config(
    config: RuntimeConfig,
    artifact_dir: Path,
    recovery_dir: Path,
    *,
    recovery_mode: str,
    lineage_id: str | None,
) -> dict[str, Any]:
    data = config.as_dict()
    data["run_dir"] = str(artifact_dir)
    data["learner"]["recovery_dir"] = str(recovery_dir.resolve())
    data["learner"]["recovery_mode"] = recovery_mode
    data["learner"]["recovery_lineage_id"] = lineage_id
    if config.bepa.enabled:
        state_dir = Path(config.bepa.state_dir).expanduser() if config.bepa.state_dir else config.run / "bepa"
        if not state_dir.is_absolute():
            state_dir = ROOT / state_dir
        data["bepa"]["state_dir"] = str(state_dir.resolve())
    if config.model_backend == "ternary":
        data["model_backend"] = "ternary"
        data["artifact_identity_path"] = str(Path(config.artifact_identity_path).resolve())
        data["artifact_identity"] = config.artifact_identity
        data["tokenizer_path"] = str((artifact_dir.parent / "tokenizer").resolve())
    return data


def _rollout_contract_update_config_view(config: RuntimeConfig, *, run_root: Path) -> str:
    """Normalize manager fields and routing while preserving all other config semantics."""
    data = config.as_dict()
    data["model_path"] = str(config.model)
    data["dataset_dir"] = str(config.dataset)
    data["run_dir"] = str(run_root)
    worker_model_path = Path(config.worker_inference.model_path).expanduser()
    if not worker_model_path.is_absolute():
        worker_model_path = ROOT / worker_model_path
    data["worker_inference"]["model_path"] = str(worker_model_path.resolve())
    data["worker_inference"].pop("max_generated_tokens", None)
    data["rollout"].pop("turn_max_tokens", None)
    data["arc"].pop("contract_path", None)
    data["arc"].pop("max_generated_tokens", None)
    data["workers"] = sorted(({"id": worker.id} for worker in config.workers), key=lambda row: row["id"])
    for key in ("recovery_dir", "recovery_mode", "recovery_lineage_id"):
        data["learner"].pop(key, None)
    # Runtime manifests inject a run-local tokenizer path that is absent from the source config.
    data.pop("tokenizer_path", None)
    state_dir = config.bepa.state_dir
    if state_dir:
        path = Path(state_dir).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        data["bepa"]["state_dir"] = str(path.resolve())
    else:
        data["bepa"]["state_dir"] = str((config.run / "bepa").resolve())
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def update_rollout_contract(config: RuntimeConfig, config_path: Path) -> dict[str, Any]:
    """Run the explicit CPU-only episode-budget revision while all managed roles are stopped."""
    from ..learner.recovery import RecoveryStore, preflight_compatibility

    config_path = Path(config_path).expanduser().resolve()
    with _manager_lock(config.run):
        current = _current_manifest(config)
        if current is None:
            raise RuntimeErrorDetail("rollout contract update requires the stopped runtime manifest")
        manifest_path, manifest = current
        recorded_config = manifest.get("config")
        if not isinstance(recorded_config, str) or not recorded_config.strip():
            raise RuntimeErrorDetail("stopped runtime manifest has no recorded source config path")
        recorded_config_path = Path(recorded_config).expanduser()
        if not recorded_config_path.is_absolute():
            raise RuntimeErrorDetail("stopped runtime manifest has an invalid source config path")
        runtime_run_dir_value = manifest.get("run_dir")
        if not isinstance(runtime_run_dir_value, str) or not runtime_run_dir_value.strip():
            raise RuntimeErrorDetail("stopped runtime manifest has no runtime run directory")
        runtime_run_dir = Path(runtime_run_dir_value).expanduser().resolve()
        expected_runs_root = (config.run / "runs").resolve()
        if runtime_run_dir.parent != expected_runs_root:
            raise RuntimeErrorDetail("stopped runtime manifest belongs to a different run directory")
        if manifest.get("status") != "stopped":
            raise RuntimeErrorDetail("rollout contract update requires a normally stopped runtime")
        records = manifest.get("processes")
        if not isinstance(records, list):
            raise RuntimeErrorDetail("stopped runtime manifest has no managed process inventory")
        for record in records:
            if not isinstance(record, dict):
                raise RuntimeErrorDetail("stopped runtime manifest has an invalid process record")
            pid = record.get("pid")
            if isinstance(pid, int) and not isinstance(pid, bool) and process_is_running(pid):
                if _owned_process_matches(record, manifest):
                    raise RuntimeErrorDetail("managed runtime roles are still active; no recovery state was changed")
                raise RuntimeErrorDetail("a recorded runtime PID is live but its identity is unverified")

        effective_path_value = manifest.get("effective_config")
        effective_sha = manifest.get("effective_config_sha256")
        if not isinstance(effective_path_value, str) or not isinstance(effective_sha, str):
            raise RuntimeErrorDetail("stopped runtime manifest has no frozen effective config identity")
        effective_path = Path(effective_path_value).expanduser()
        try:
            if (effective_path.resolve() != runtime_run_dir / "runtime.yaml"
                    or effective_path.is_symlink() or not effective_path.is_file()
                    or hashlib.sha256(effective_path.read_bytes()).hexdigest() != effective_sha):
                raise RuntimeErrorDetail("frozen effective config is missing or its SHA does not match the manifest")
            old_config = load_config(effective_path, for_local_worker_plan=True)
        except RuntimeErrorDetail:
            raise
        except Exception as exc:
            raise RuntimeErrorDetail(
                f"cannot load the frozen effective config ({type(exc).__name__})"
            ) from exc
        if _rollout_contract_update_config_view(old_config, run_root=config.run) != _rollout_contract_update_config_view(config, run_root=config.run):
            raise RuntimeErrorDetail(
                "normalized runtime config changed outside the episode contract, three generation caps, "
                "manager recovery fields, or Worker routing"
            )
        if (old_config.bepa.budget_scope != config.bepa.budget_scope
                or old_config.bepa.teacher.budget != config.bepa.teacher.budget):
            raise RuntimeErrorDetail("BEPA state, budget scope, or teacher ledger limits changed")

        old_limits = {
            "rollout.turn_max_tokens": old_config.rollout.turn_max_tokens,
            "worker_inference.max_generated_tokens": old_config.worker_inference.max_generated_tokens,
            "arc.max_generated_tokens": old_config.arc.max_generated_tokens,
        }
        new_limits = {
            "rollout.turn_max_tokens": config.rollout.turn_max_tokens,
            "worker_inference.max_generated_tokens": config.worker_inference.max_generated_tokens,
            "arc.max_generated_tokens": config.arc.max_generated_tokens,
        }
        runtime_limits = {
            name: {"from": old_limits[name], "to": new_limits[name]}
            for name in old_limits
        }
        recovery_config = _recovery_config(config)
        expected = preflight_compatibility(recovery_config)
        try:
            result = RecoveryStore(config.run / "training").update_rollout_contract(
                expected_compatibility=expected,
                old_contract_path=old_config.contract,
                new_contract_path=config.contract,
                turn_max_tokens=config.rollout.turn_max_tokens,
                runtime_limits=runtime_limits,
            )
        except Exception as exc:
            if isinstance(exc, RuntimeErrorDetail):
                raise
            raise RuntimeErrorDetail(
                f"rollout contract update refused ({type(exc).__name__}: {str(exc)[:300]})"
            ) from exc
        return {
            **result,
            "runtime_status": manifest.get("status"),
            "manager_manifest": str(manifest_path),
            "recorded_source_config": str(recorded_config_path.resolve()),
            "selected_source_config": str(config_path),
            "managed_roles_active": False,
            "bepa_state_unchanged": True,
        }


def _inspect_training_recovery(config: RuntimeConfig) -> dict[str, Any]:
    """Perform the engine's CPU-only checkpoint gate before stopping or spawning roles."""
    try:
        from ..learner.recovery import inspect_recovery, preflight_compatibility
    except ImportError as exc:
        raise RuntimeErrorDetail(
            f"training recovery verifier is unavailable ({type(exc).__name__}); refusing to start"
        ) from exc
    try:
        expected = preflight_compatibility(_recovery_config(config))
        result = inspect_recovery(config.run / "training", expected_compatibility=expected)
    except Exception as exc:
        raise RuntimeErrorDetail(
            f"training recovery preflight failed ({type(exc).__name__}: {str(exc)[:300]}); "
            "refusing to start from an unverifiable state; use `--fresh` after stopping old roles "
            "to create a new lineage"
        ) from exc
    if not isinstance(result, dict) or result.get("status") not in {
            "resume", "empty", "legacy_adapter_only", "corrupt", "incompatible"}:
        raise RuntimeErrorDetail("training recovery verifier returned an invalid checkpoint status")
    return result


def _activate_runtime_locked(
    config: RuntimeConfig,
    config_path: Path,
    *,
    start_teacher: bool,
    resume_interrupted_run: bool,
) -> dict[str, Any]:
    """Activate optional runtime roles only on an explicit deployment action."""
    current = _current_manifest(config)
    if current is None:
        raise RuntimeErrorDetail("no managed Learner runtime is active")
    manifest_path, manifest = current
    if not _manifest_config_matches(manifest, config, config_path):
        raise RuntimeErrorDetail("active Learner runtime does not match the selected deployment profile")
    validate_environment(config)
    active_config = _manifest_active_config(config, manifest)
    expected_core = {
        ("learner", None),
        *(('worker', worker.id) for worker in active_config.workers if worker.scope == "local"),
    }
    records = [record for record in manifest.get("processes", []) if isinstance(record, dict)]
    live_core = {
        (record.get("role"), record.get("worker_id"))
        for record in records
        if (record.get("role") == "learner" or record.get("role") == "worker")
        and _owned_process_matches(record, manifest)
    }
    if live_core != expected_core:
        raise RuntimeErrorDetail("active Learner runtime is not READY for explicit activation")
    learner_health = _health(active_config, "learner")
    if (not learner_health
            or not _is_quantized_learner(learner_health, active_config.quantization.mode, _config_artifact_sha256(active_config))
            or not _learner_recovery_is_clean(learner_health)):
        raise RuntimeErrorDetail("active Learner health is not durable and READY for explicit activation")
    for worker in active_config.workers:
        if worker.scope != "local":
            continue
        health = _health(active_config, "worker", worker.id)
        if not health or not _is_runtime_worker(health, worker.id, active_config.quantization.mode, _config_artifact_sha256(active_config)):
            raise RuntimeErrorDetail(f"local Worker {worker.id} is not READY for explicit activation")
    external_ready, external_errors = _active_remote_health(active_config, manifest)
    if not external_ready:
        raise RuntimeErrorDetail(
            "external Worker membership is not READY for explicit activation "
            f"({', '.join(external_errors)})"
        )
    started_teacher = False
    teacher_id = None
    existing_teacher = None
    endpoints = manifest.setdefault("endpoints", {})
    if start_teacher and config.bepa.enabled:
        current_teacher = endpoints.get("teacher")
        if isinstance(current_teacher, dict):
            current_id = current_teacher.get("worker_id")
            existing_teacher = next((record for record in manifest.get("processes", [])
                                     if isinstance(record, dict) and record.get("role") == "teacher"
                                     and record.get("worker_id") == current_id
                                     and _owned_process_matches(record, manifest)), None)
        else:
            current_id = None
        if existing_teacher is not None:
            teacher_id = existing_teacher["worker_id"]

    recovery: dict[str, Any] | None = None
    if resume_interrupted_run:
        recovery_value = learner_health.get("recovery") if isinstance(learner_health, dict) else None
        recovery = recovery_value if isinstance(recovery_value, dict) else None
        if not isinstance(recovery, dict):
            raise RuntimeErrorDetail("Learner health omitted its recovery state")
        pending_run = recovery.get("pending_run")
    else:
        pending_run = None
    resume_marker = manifest.get("resume_activation")
    if isinstance(resume_marker, dict) and resume_marker.get("state") in {"requesting", "outcome_unknown"}:
        raise RuntimeErrorDetail(
            "a prior resume POST has an unknown outcome; inspect the Learner run before requesting another resume"
        )
    resume_already_accepted = (
        isinstance(resume_marker, dict)
        and resume_marker.get("state") == "accepted"
    )
    needs_resume = pending_run not in (None, False) and not resume_already_accepted
    resumed = resume_already_accepted
    needs_teacher = (start_teacher and config.bepa.enabled and existing_teacher is None)
    if not needs_resume and not needs_teacher:
        return {
            "status": manifest.get("status", "ready"),
            "run_id": manifest.get("run_id"),
            "teacher_started": False,
            "teacher_worker_id": teacher_id,
            "resumed": False,
            "resume": manifest.get("resume"),
            "manifest": str(manifest_path),
        }

    # Validate Teacher credentials before any resume POST, but create the
    # process only after the Learner has acknowledged that one-shot request.
    if needs_teacher:
        validate_bepa_environment(config)

    if needs_resume:
        lineage_id = recovery.get("lineage_id") if recovery else None
        if not isinstance(lineage_id, str) or not lineage_id:
            raise RuntimeErrorDetail("pending Learner run has no verified lineage ID")
        resume_result = _request_resume_once(
            active_config, manifest, manifest_path, lineage_id,
            timeout=active_config.learner.request_timeout,
        )
        resumed = resume_result.get("resumed") is True
        if resumed:
            manifest["status"] = "resume_started"
            write_json_atomic(manifest_path, manifest)

    if needs_teacher:
        effective_text = manifest.get("effective_config")
        run_dir_text = manifest.get("run_dir")
        if not isinstance(effective_text, str) or not effective_text:
            raise RuntimeErrorDetail("active Learner manifest omitted its effective config")
        if not isinstance(run_dir_text, str) or not run_dir_text:
            raise RuntimeErrorDetail("active Learner manifest omitted its run directory")
        teacher_id = f"teacher-{manifest['run_id']}-{uuid.uuid4().hex[:8]}"
        endpoints["teacher"] = {"worker_id": teacher_id, "port": 0, "gpu": -1}
        external_tokens = tuple(sorted({worker.auth_token_env for worker in active_config.workers
                                        if worker.scope in {"gateway", "service"}}))
        record = _spawn(
            manifest, manifest_path, role="teacher", worker_id=teacher_id,
            port=0, gpu=-1, config_path=Path(effective_text), source_config=config_path,
            run_dir=Path(run_dir_text), secret_env_names=external_tokens,
        )
        record["state"] = "running"
        record["started_after_policy_acks"] = bool(manifest.get("policy_acks"))
        started_teacher = True
        write_json_atomic(manifest_path, manifest)
    manifest_recovery = manifest.get("recovery")
    if isinstance(manifest_recovery, dict) and recovery is not None:
        manifest_recovery["pending_run"] = recovery.get("pending_run")
    write_json_atomic(manifest_path, manifest)
    if not resumed and started_teacher:
        manifest["activation"] = {
            "teacher_started_at": time.time(),
            "teacher_worker_id": teacher_id,
        }
        write_json_atomic(manifest_path, manifest)
    return {
        "status": manifest.get("status", "ready"),
        "run_id": manifest.get("run_id"),
        "teacher_started": started_teacher,
        "teacher_worker_id": teacher_id,
        "resumed": resumed,
        "resume": manifest.get("resume"),
        "manifest": str(manifest_path),
    }


def activate_runtime(
    config: RuntimeConfig,
    config_path: str | Path,
    *,
    start_teacher: bool = False,
    resume_interrupted_run: bool = False,
) -> dict[str, Any]:
    """Explicitly activate BEPA Teacher and/or resume a saved logical run."""
    if type(start_teacher) is not bool or type(resume_interrupted_run) is not bool:
        raise ValueError("start_teacher and resume_interrupted_run must be bool")
    config_path = Path(config_path).expanduser().resolve()
    with _manager_lock(config.run):
        return _activate_runtime_locked(
            config, config_path, start_teacher=start_teacher,
            resume_interrupted_run=resume_interrupted_run,
        )


def _recovery_gate_error(result: dict[str, Any], *, partial: bool) -> RuntimeErrorDetail:
    status = result.get("status")
    detail = result.get("error") or result.get("reason")
    if status == "legacy_adapter_only":
        message = "only adapter checkpoints are present; optimizer state is unavailable"
    elif status == "empty":
        message = "no full training checkpoint is available"
    elif status == "incompatible":
        message = "the latest full training checkpoint is incompatible with this training contract"
    else:
        message = "the latest full training checkpoint is corrupt or incomplete"
    if isinstance(detail, str) and detail:
        message += f" ({detail[:300]})"
    if partial:
        message += "; remaining owned roles were left running"
    message += "; use `--fresh` only after explicitly stopping the old roles to create a new lineage"
    return RuntimeErrorDetail(message)


def _stop_owned_roles_for_recovery(
    manifest: dict[str, Any], manifest_path: Path, records: list[dict[str, Any]],
    *, timeout: float = 120.0,
) -> None:
    """Send SIGINT only to exact manifest identities; never escalate an automatic recovery stop."""
    targets = [record for record in reversed(records)
               if _owned_process_matches(record, manifest)]
    manifest["recovery_stop_started_at"] = time.time()
    manifest["recovery_stop_outcomes"] = {}
    write_json_atomic(manifest_path, manifest)
    for record in targets:
        pid = record.get("pid")
        key = str(record.get("worker_id") or record.get("role"))
        if not _owned_process_matches(record, manifest):
            continue
        try:
            os.kill(pid, signal.SIGINT)
            manifest["recovery_stop_outcomes"][key] = "SIGINT"
        except ProcessLookupError:
            manifest["recovery_stop_outcomes"][key] = "exited"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = []
        for record in targets:
            pid = record.get("pid")
            if not isinstance(pid, int) or isinstance(pid, bool):
                continue
            current_starttime = process_starttime(pid)
            if process_is_running(pid) and (
                    str(current_starttime) != str(record.get("starttime"))
                    or not _owned_process_matches(record, manifest)):
                manifest["status"] = "recovery_stop_incomplete"
                manifest["recovery_stop_finished_at"] = time.time()
                write_json_atomic(manifest_path, manifest)
                raise RuntimeErrorDetail(
                    f"automatic recovery stop found a PID identity change for "
                    f"{record.get('worker_id') or record.get('role')}; no further signal was sent"
                )
            if process_is_running(pid):
                alive.append(record)
        if not alive:
            break
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    remaining = []
    for record in targets:
        pid = record.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            continue
        current_starttime = process_starttime(pid)
        if process_is_running(pid):
            if (str(current_starttime) != str(record.get("starttime"))
                    or not _owned_process_matches(record, manifest)):
                manifest["status"] = "recovery_stop_incomplete"
                manifest["recovery_stop_finished_at"] = time.time()
                write_json_atomic(manifest_path, manifest)
                raise RuntimeErrorDetail(
                    f"automatic recovery stop found a PID identity change for "
                    f"{record.get('worker_id') or record.get('role')}; no further signal was sent"
                )
            remaining.append(record)
    if remaining:
        manifest["status"] = "recovery_stop_incomplete"
        manifest["recovery_stop_finished_at"] = time.time()
        write_json_atomic(manifest_path, manifest)
        names = [str(item.get("worker_id") or item.get("role")) for item in remaining]
        raise RuntimeErrorDetail(
            f"automatic recovery could not gracefully stop owned roles {names}; "
            "no process was force-killed and no new roles were started"
        )
    manifest["status"] = "stopped_for_recovery"
    manifest["recovery_stop_finished_at"] = time.time()
    write_json_atomic(manifest_path, manifest)


def _current_manifest(config: RuntimeConfig) -> tuple[Path, dict[str, Any]] | None:
    path = active_manifest_path(config.run)
    if path is None or not path.exists():
        return None
    return path, read_json(path)


def _manifest_active_config(config: RuntimeConfig, manifest: dict[str, Any]) -> RuntimeConfig:
    """Load the exact worker set used by this run, never the deployed superset."""
    membership = manifest.get("membership")
    has_remote = any(worker.scope in {"gateway", "service"} for worker in config.workers)
    if has_remote and isinstance(membership, dict) and membership.get("phase") != "active":
        raise RuntimeErrorDetail("external Worker membership has not completed policy ACK; no run request was sent")
    effective_path = manifest.get("effective_config")
    path = Path(effective_path).expanduser() if isinstance(effective_path, str) and effective_path else None
    if path is None or not path.is_file():
        if has_remote:
            raise RuntimeErrorDetail("active external-Worker run is missing its effective membership config")
        return config
    try:
        active = load_config(path)
    except Exception as exc:
        raise RuntimeErrorDetail(
            f"could not load the run's effective membership config ({type(exc).__name__})"
        ) from exc
    rows = membership.get("active_workers") if isinstance(membership, dict) else None
    if isinstance(rows, list):
        expected_ids = [row.get("worker_id") for row in rows if isinstance(row, dict)]
        if expected_ids != [worker.id for worker in active.workers]:
            raise RuntimeErrorDetail("effective config Worker IDs differ from the active membership snapshot")
    return active


def _existing_live_run(config: RuntimeConfig) -> bool:
    current = _current_manifest(config)
    if current is None:
        return False
    manifest = current[1]
    return any(isinstance(record, dict) and _owned_process_matches(record, manifest)
               for record in manifest.get("processes", []))


def _command(
    role: str, config_path: Path, worker_id: str | None = None, *,
    environment_file: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    args = [sys.executable, "-m", "qwen_rl.runtime.cli", role, "--config", str(config_path)]
    identity = ["-m", "qwen_rl.runtime.cli", role, "--config", str(config_path)]
    if environment_file is not None:
        if role != "learner":
            raise ValueError("private Worker environment files may only be passed to the Learner")
        environment_path = str(Path(environment_file).expanduser().resolve())
        args.extend(["--environment-file", environment_path])
        identity.extend(["--environment-file", environment_path])
    if worker_id is not None:
        args.extend(["--worker-id", worker_id])
        identity.extend(["--worker-id", worker_id])
    return args, identity


def _owned_process_matches(record: dict[str, Any], manifest: dict[str, Any]) -> bool:
    """Validate this run's stored role/port/GPU/config before checking the live PID."""
    role = record.get("role")
    worker_id = record.get("worker_id")
    endpoints = manifest.get("endpoints")
    effective_config = manifest.get("effective_config")
    if not isinstance(endpoints, dict) or not isinstance(effective_config, str):
        return False
    if role == "learner" and worker_id is None:
        expected = endpoints.get("learner")
    elif role == "worker" and isinstance(worker_id, str):
        workers = endpoints.get("workers")
        expected = workers.get(worker_id) if isinstance(workers, dict) else None
    elif role == "teacher" and isinstance(worker_id, str):
        expected = endpoints.get("teacher")
        if not isinstance(expected, dict) or expected.get("worker_id") != worker_id:
            return False
    else:
        return False
    if not isinstance(expected, dict):
        return False
    expected_gpu = expected.get("process_gpu") if role == "learner" else expected.get("gpu")
    environment_file = expected.get("environment_file") if role == "learner" else None
    if environment_file is not None and not isinstance(environment_file, str):
        return False
    if (record.get("config") != effective_config
            or record.get("source_config") != manifest.get("config")
            or record.get("port") != expected.get("port")
            or record.get("gpu") != expected_gpu
            or record.get("environment_file") != environment_file):
        return False
    expected_argv, expected_identity = _command(
        role, Path(effective_config), worker_id, environment_file=environment_file)
    if record.get("argv") != expected_argv or record.get("identity_args") != expected_identity:
        return False
    return process_matches(record)


def _require_registered_roles_alive(
    records: list[dict[str, Any]], manifest: dict[str, Any],
) -> None:
    """Fail readiness polling as soon as any spawned role exits or loses identity."""
    for record in records:
        if _owned_process_matches(record, manifest):
            continue
        role = record.get("role")
        worker_id = record.get("worker_id")
        label = "Learner" if role == "learner" else (
            f"Worker {worker_id}" if role == "worker" and worker_id else str(role)
        )
        pid = record.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool) and not process_is_running(pid):
            reason = "process exited"
        else:
            reason = "process no longer matches its registered identity"
        log_path = record.get("log")
        log_reference = log_path if isinstance(log_path, str) and log_path else "unavailable"
        recovery_detail = None
        if role == "learner" and isinstance(log_path, str):
            try:
                with Path(log_path).open("rb") as stream:
                    stream.seek(max(0, Path(log_path).stat().st_size - 16_384))
                    tail = stream.read().decode("utf-8", errors="replace")
                recovery_detail = next((line.strip()[-600:] for line in reversed(tail.splitlines())
                    if any(marker in line.lower() for marker in (
                        "recoveryerror", "recovery", "full optimizer", "adapter-only", "--fresh",
                    ))), None)
            except OSError:
                recovery_detail = None
        detail_suffix = f"; startup detail: {recovery_detail}" if recovery_detail else ""
        raise RuntimeErrorDetail(
            f"{label} {reason} before readiness{detail_suffix}; see log {log_reference}"
        )


def _ready_role_health(
    config: RuntimeConfig,
    role: str,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    worker_id: str | None = None,
) -> dict[str, Any] | None:
    _require_registered_roles_alive(records, manifest)
    health = _health(config, role, worker_id)
    if role == "learner":
        return health if health and _is_quantized_learner(health, config.quantization.mode, _config_artifact_sha256(config)) else None
    if role == "worker":
        return health if health and worker_id and _is_runtime_worker(health, worker_id, config.quantization.mode, _config_artifact_sha256(config)) else None
    raise ValueError(f"unsupported readiness role: {role}")


def _role_child_environment(
    role: str,
    environment: dict[str, str],
    *,
    gpu: int | None,
    secret_env_names: tuple[str, ...] = (),
) -> dict[str, str]:
    child_env = dict(environment)
    if role == "learner":
        from ..learner.cuda_allocator import configure_learner_allocator
        configure_learner_allocator(child_env)
    if role == "teacher":
        child_env.pop("CUDA_VISIBLE_DEVICES", None)
        child_env.pop("ARC_ENV_TOKEN", None)
        for name in secret_env_names:
            child_env.pop(name, None)
    elif gpu is None:
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        for name in secret_env_names:
            if name != "ARC_ENV_TOKEN":
                child_env.pop(name, None)
    else:
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for name in secret_env_names:
            if name != "ARC_ENV_TOKEN":
                child_env.pop(name, None)
    return child_env


def _spawn(manifest: dict[str, Any], manifest_path: Path, *, role: str, worker_id: str | None,
           port: int, gpu: int | None, config_path: Path, source_config: Path,
           run_dir: Path, secret_env_names: tuple[str, ...] = (),
           environment_file: str | Path | None = None) -> dict[str, Any]:
    command, identity = _command(role, config_path, worker_id, environment_file=environment_file)
    log_path = run_dir / "logs" / f"{role}-{worker_id or 'main'}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The source solver is CPU-only and receives the teacher token. Learner and
    # Worker children retain ARC_ENV_TOKEN but never inherit provider credentials.
    child_env = _role_child_environment(role, dict(os.environ), gpu=gpu,
                                        secret_env_names=secret_env_names)
    child_env["ARC_RL_ROOT"] = str(ROOT)
    child_env["QWEN_RL_MANIFEST_RUN_ID"] = str(manifest["run_id"])
    with log_path.open("ab", buffering=0) as log_file:
        process = subprocess.Popen(
            command, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
            stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
        )
    deadline = time.monotonic() + 5.0
    starttime = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeErrorDetail(f"{role} process exited during startup; see {log_path}")
        starttime = process_starttime(process.pid)
        if starttime is not None:
            break
        time.sleep(0.05)
    if starttime is None:
        # This is the exact Popen child created above, before its identity can be reused.
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        raise RuntimeErrorDetail(f"could not capture {role} process starttime")
    record = {
        "role": role,
        "worker_id": worker_id,
        "pid": process.pid,
        "starttime": starttime,
        "port": port,
        "gpu": gpu,
        "config": str(config_path),
        "source_config": str(source_config),
        "identity_args": identity,
        "argv": command,
        "log": str(log_path),
        "started_at": time.time(),
        "state": "starting",
    }
    if environment_file is not None:
        record["environment_file"] = str(Path(environment_file).expanduser().resolve())
    manifest["processes"].append(record)
    write_json_atomic(manifest_path, manifest)
    return record


def _policy_snapshot(config: RuntimeConfig) -> dict[str, Any]:
    response = _request("GET", f"{config.learner_url}/policy", timeout=config.learner.request_timeout)
    if response.status_code != 200:
        raise RuntimeErrorDetail(f"learner GET /policy returned HTTP {response.status_code}")
    payload = response.json()
    if (not isinstance(payload, dict)
            or isinstance(payload.get("version"), bool)
            or not isinstance(payload.get("version"), int)
            or payload["version"] < 0):
        raise RuntimeErrorDetail("learner /policy returned an invalid version")
    if config.model_backend == "ternary":
        expected_sha = _config_artifact_sha256(config)
        if (not _t2_quantization_matches(payload.get("quantization"), expected_sha)
                or payload.get("base_fingerprint") != expected_sha
                or payload.get("artifact_fingerprint") != expected_sha):
            raise RuntimeErrorDetail("learner /policy T2 identity differs from the verified base artifact")
    else:
        from ..models.quantization import quantization_evidence
        try:
            quantization_evidence(payload.get("quantization"), expected_mode=config.quantization.mode)
        except ValueError as exc:
            raise RuntimeErrorDetail("learner /policy returned invalid quantization evidence") from exc
    if not isinstance(payload.get("base_fingerprint"), str) or not payload["base_fingerprint"]:
        raise RuntimeErrorDetail("learner /policy omitted base_fingerprint")
    if not isinstance(payload.get("artifact_fingerprint"), str) or len(payload["artifact_fingerprint"]) != 64:
        raise RuntimeErrorDetail("policy omitted its packed artifact identity")
    entries = payload.get("files")
    if not isinstance(entries, list):
        raise RuntimeErrorDetail("learner /policy files must be a manifest list")
    native_t2 = config.model_backend == "ternary"
    expected_names = ({"policy.lora.ninfer", "policy.lora.ninfer.json"}
                      if native_t2 else {"adapter_config.json", "adapter_model.safetensors"})
    hashes: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeErrorDetail("learner /policy contains an invalid file manifest entry")
        name, digest, url = entry.get("name"), entry.get("sha256"), entry.get("url")
        if name not in expected_names \
                or name in hashes or not isinstance(digest, str) or len(digest) != 64 \
                or any(char not in "0123456789abcdefABCDEF" for char in digest) \
                or not isinstance(url, str) or not url:
            raise RuntimeErrorDetail("learner /policy contains a malformed policy file entry")
        hashes[name] = digest.lower()
    if set(hashes) != expected_names:
        transport = "native T2" if native_t2 else "standard PEFT"
        raise RuntimeErrorDetail(f"learner /policy must publish exactly the two {transport} files")
    if native_t2:
        source_digest = payload.get("source_adapter_sha256")
        if (not isinstance(source_digest, str) or len(source_digest) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in source_digest)):
            raise RuntimeErrorDetail("learner /policy omitted a valid source_adapter_sha256")
    return payload


def _policy_snapshot_identity(policy: dict[str, Any]) -> PolicySnapshot:
    """Build a policy identity from either a wire manifest list or internal file map."""
    files = policy.get("files")
    normalized = policy
    if isinstance(files, list):
        file_map: dict[str, dict[str, str]] = {}
        for entry in files:
            if not isinstance(entry, dict):
                raise RuntimeErrorDetail("policy identity contains an invalid manifest entry")
            name, digest = entry.get("name"), entry.get("sha256")
            if not isinstance(name, str) or not isinstance(digest, str):
                raise RuntimeErrorDetail("policy identity manifest entry omitted name or SHA-256")
            file_map[name] = {"sha256": digest.lower()}
        normalized = {**policy, "files": file_map}
    elif not isinstance(files, dict):
        raise RuntimeErrorDetail("policy identity requires a file manifest")
    if isinstance(normalized.get("source_adapter_sha256"), str):
        normalized = {**normalized, "source_adapter_sha256": normalized["source_adapter_sha256"].lower()}
    try:
        return PolicySnapshot.from_policy(normalized)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeErrorDetail("policy manifest has an invalid adapter identity") from exc


def _policy_error(response: httpx.Response) -> tuple[str | None, str | None]:
    """(code, message) of a WorkerFailure response body, if it carries one."""
    try:
        body = response.json()
    except Exception:
        return None, None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None, None
    code, message = error.get("code"), error.get("message")
    return (code if isinstance(code, str) else None), (message if isinstance(message, str) else None)


def _wait_external_worker_policy_eligibility(
    config: RuntimeConfig,
    *,
    version: int,
    adapter_hash: str,
    artifact_fingerprint: str,
    base_fingerprint: str,
) -> None:
    """Wait for fresh Controller health to expose the ACKed policy as eligible.

    A Worker may accept its first policy before the next node heartbeat. The
    Gateway keeps generation disabled during that interval, so the manager waits
    for an exact version/SHA receipt before starting the selftest or returning.
    """
    groups: dict[tuple[str, str], list[Any]] = {}
    direct_services = []
    for worker in config.workers:
        if worker.scope == "gateway":
            groups.setdefault((worker.site_id, worker.endpoint.rstrip("/")), []).append(worker)
        elif worker.scope == "service":
            direct_services.append(worker)
    if not groups and not direct_services:
        return
    deadline = time.monotonic() + _EXTERNAL_POLICY_ACK_REFRESH_SECONDS
    pending = [worker.id for workers in groups.values() for worker in workers]
    pending.extend(worker.id for worker in direct_services)
    last_reason = "Controller has not reported eligible policy health"
    while time.monotonic() < deadline:
        pending = []
        for (_site_id, endpoint), workers in groups.items():
            token_env = workers[0].auth_token_env
            token = os.environ.get(token_env)
            if not token:
                raise RuntimeErrorDetail(f"Gateway bearer environment variable {token_env} is missing")
            try:
                response = httpx.get(
                    f"{endpoint}/v1/workers?readiness=policy",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=min(5.0, max(0.1, deadline - time.monotonic())),
                )
            except httpx.HTTPError as exc:
                pending.extend(worker.id for worker in workers)
                last_reason = f"Gateway eligibility snapshot unavailable ({type(exc).__name__})"
                continue
            if response.status_code != 200:
                pending.extend(worker.id for worker in workers)
                last_reason = f"Gateway eligibility snapshot returned HTTP {response.status_code}"
                continue
            try:
                payload = response.json()
            except ValueError:
                pending.extend(worker.id for worker in workers)
                last_reason = "Gateway eligibility snapshot was invalid JSON"
                continue
            rows = payload.get("workers") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                pending.extend(worker.id for worker in workers)
                last_reason = "Gateway eligibility snapshot omitted its Worker list"
                continue
            by_id: dict[str, dict[str, Any]] = {}
            duplicate_ids: set[str] = set()
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("worker_id"), str):
                    continue
                if row["worker_id"] in by_id:
                    duplicate_ids.add(row["worker_id"])
                by_id[row["worker_id"]] = row
            if duplicate_ids:
                raise RuntimeErrorDetail(
                    f"Gateway eligibility snapshot returned duplicate Worker IDs: {sorted(duplicate_ids)}"
                )
            for worker in workers:
                row = by_id.get(worker.id)
                if row is None:
                    pending.append(worker.id)
                    last_reason = "Controller has not published this Worker as policy-eligible"
                    continue
                health = row.get("worker_health")
                if (row.get("site_id") != worker.site_id or row.get("node_id") != worker.node_id
                        or row.get("base_path") != worker.url_prefix):
                    raise RuntimeErrorDetail(
                        f"Gateway policy-eligible identity for Worker {worker.id} changed during startup"
                    )
                if row.get("boot_id") != worker.boot_id:
                    raise RuntimeErrorDetail(
                        f"Gateway Worker {worker.id} boot_id changed before initial rollout admission"
                    )
                if (not isinstance(health, dict)
                        or health.get("telemetry_epoch") != worker.telemetry_epoch):
                    raise RuntimeErrorDetail(
                        f"Gateway Worker {worker.id} telemetry epoch changed before initial rollout admission"
                    )
                ready = (
                    row.get("eligible") is True
                    and row.get("model_ready") is True
                    and _is_vllm_worker(health, worker.id)
                    and health.get("policy_ready") is True
                    and health.get("policy_version") == version
                    and health.get("adapter_hash") == adapter_hash
                    and health.get("artifact_fingerprint") == artifact_fingerprint
                    and health.get("base_fingerprint") == base_fingerprint
                )
                if not ready:
                    pending.append(worker.id)
                    last_reason = "Controller health has not ACKed the configured Learner policy"
        for worker in direct_services:
            health = _health(config, "worker", worker.id)
            deployment = health.get("deployment") if isinstance(health, dict) else None
            if (not health or not _is_runtime_worker(health, worker.id, config.quantization.mode, _config_artifact_sha256(config))
                    or health.get("boot_id") != worker.boot_id
                    or health.get("telemetry_epoch") != worker.telemetry_epoch
                    or not isinstance(deployment, dict)
                    or deployment.get("scope") != "service"
                    or deployment.get("node_id") != worker.node_id
                    or deployment.get("site_id") != worker.site_id
                    or health.get("policy_ready") is not True
                    or health.get("policy_version") != version
                    or health.get("adapter_hash") != adapter_hash
                    or health.get("artifact_fingerprint") != artifact_fingerprint
                    or health.get("base_fingerprint") != base_fingerprint):
                pending.append(worker.id)
                last_reason = "direct service Worker health has not ACKed the configured Learner policy"
        if not pending:
            return
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    raise RuntimeErrorDetail(
        "external Workers did not become policy-eligible within 15 seconds: "
        f"{sorted(set(pending))} ({last_reason}); no rollout or selftest was started"
    )


def _sync_initial_policy(config: RuntimeConfig, timeout: float) -> dict[str, dict[str, Any]]:
    policy = _policy_snapshot(config)
    version = policy["version"]
    expected_files = {item["name"]: {"sha256": item["sha256"].lower()}
                      for item in policy["files"]}
    expected_adapter_hash = _policy_snapshot_identity(policy).adapter_sha256
    ack: dict[str, dict[str, Any]] = {}
    for worker in config.workers:
        if worker.scope in {"gateway", "service"} and worker.model_sha256 != policy["artifact_fingerprint"]:
            raise RuntimeErrorDetail(
                f"external Worker {worker.id} model SHA does not match the Learner artifact; it was not policy-ACKed"
            )
        deadline = time.monotonic() + timeout
        while True:
            response = _worker_request(
                worker, "POST", "policy", timeout=timeout, json={"version": version},
            )
            code, message = _policy_error(response)
            # A Worker can legitimately be busy with its own policy update; this is transient.
            if response.status_code == 409 and code == "worker_busy" and time.monotonic() < deadline:
                time.sleep(5.0)
                continue
            break
        if response.status_code != 200:
            raise RuntimeErrorDetail(
                f"worker {worker.id} policy sync returned HTTP {response.status_code}"
                + (f" ({code}: {message})" if code else ""))
        result = response.json()
        if not isinstance(result, dict) or result.get("version") != version \
                or result.get("worker_id") != worker.id \
                or result.get("files") != expected_files \
                or result.get("adapter_hash") != expected_adapter_hash:
            raise RuntimeErrorDetail(f"worker {worker.id} did not ACK policy version {version}")
        ack[worker.id] = result
    learner_health = _health(config, "learner")
    if (not isinstance(learner_health, dict)
            or not _is_quantized_learner(
                learner_health, config.quantization.mode, _config_artifact_sha256(config))):
        raise RuntimeErrorDetail("Learner health no longer matches its verified model evidence")
    for worker in config.workers:
        health = _health(config, "worker", worker.id)
        if (not health or not _is_runtime_worker(health, worker.id, config.quantization.mode, _config_artifact_sha256(config))
                or health.get("policy_version") != version
                or health.get("artifact_fingerprint") != policy["artifact_fingerprint"]
                or health.get("base_fingerprint") != policy["base_fingerprint"]
                or health.get("adapter_hash") != expected_adapter_hash):
            raise RuntimeErrorDetail(f"worker {worker.id} health does not match policy version {version}")
    _wait_external_worker_policy_eligibility(
        config,
        version=version,
        adapter_hash=expected_adapter_hash,
        artifact_fingerprint=policy["artifact_fingerprint"],
        base_fingerprint=policy["base_fingerprint"],
    )
    if config.learner.gpu_sharing == "phase_shared":
        for worker in config.workers:
            if worker.id not in config.learner.shared_worker_ids:
                continue
            response = _worker_request(worker, "POST", "residency", timeout=timeout,
                                       json={"action": "admit", "policy_version": version,
                                             "adapter_sha256": expected_adapter_hash})
            if response.status_code != 200 or response.json().get("state") != "resident":
                raise RuntimeErrorDetail(f"Worker {worker.id} could not enter sampling after initial ACK")
    return ack


def _trigger_selftest(config: RuntimeConfig, timeout: float) -> dict[str, Any]:
    response = _request("POST", f"{config.learner_url}/run", timeout=config.learner.request_timeout,
                        json={"kind": "selftest", "steps": 1})
    if response.status_code != 202:
        raise RuntimeErrorDetail(f"learner POST /run selftest returned HTTP {response.status_code}")
    payload = response.json()
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeErrorDetail("learner did not return a selftest run_id")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reply = _request("GET", f"{config.learner_url}/run", timeout=config.learner.request_timeout,
                         params={"run_id": run_id})
        if reply.status_code != 200:
            raise RuntimeErrorDetail(f"learner GET /run returned HTTP {reply.status_code}")
        state = reply.json()
        if isinstance(state, dict) and state.get("status") in {"succeeded", "failed", "filtered_limit"}:
            if state.get("status") != "succeeded":
                raise RuntimeErrorDetail(f"real-model connectivity selftest ended {state.get('status')}")
            summary = state.get("summary")
            if not isinstance(summary, dict):
                raise RuntimeErrorDetail("selftest result lacks its acceptance summary")
            current_policy = _policy_snapshot(config)
            required = {
                "fixture_kind": "identity_connectivity",
                "diagnostic_course": "identity-workload-v1",
                "reward_source": "arc_service",
                "updates": 1,
            }
            for field, expected in required.items():
                if summary.get(field) != expected:
                    raise RuntimeErrorDetail(f"selftest acceptance field {field} did not match")
            for field in ("model_generations", "scored_episodes"):
                value = summary.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < (config.learner.groups_per_update + 1) * config.rollout.group_size:
                    raise RuntimeErrorDetail(f"selftest acceptance field {field} must cover the sampling and post-update groups")
            initial_version = summary.get("initial_policy_version")
            initial_adapter_hash = summary.get("initial_adapter_hash")
            current_adapter_hash = _policy_snapshot_identity(current_policy).adapter_sha256
            if (isinstance(initial_version, bool) or not isinstance(initial_version, int)
                    or initial_version + 1 != current_policy["version"]
                    or not isinstance(initial_adapter_hash, str)
                    or initial_adapter_hash == current_adapter_hash):
                raise RuntimeErrorDetail("selftest did not publish a changed adapter at the next version")
            groups = state.get("groups") if isinstance(state, dict) else None
            if not isinstance(groups, list):
                raise RuntimeErrorDetail("selftest result omitted its group records")
            training_groups = [group for group in groups if isinstance(group, dict)
                               and group.get("new_policy_version") == current_policy["version"]]
            if len(training_groups) != config.learner.groups_per_update:
                raise RuntimeErrorDetail("selftest did not record its configured number of complete DAPO training groups")
            update = training_groups[0].get("update")
            if not isinstance(update, dict) or update.get("updated") is not True \
                    or update.get("policy_version") != current_policy["version"]:
                raise RuntimeErrorDetail("selftest did not record a successful LearnerEngine update")
            gradient_norm = update.get("gradient_norm_before_clip")
            if (isinstance(gradient_norm, bool) or not isinstance(gradient_norm, (int, float))
                    or not math.isfinite(float(gradient_norm)) or gradient_norm <= 0.0):
                raise RuntimeErrorDetail("selftest DAPO update has no finite nonzero gradient norm")
            publication = update.get("publication")
            if not isinstance(publication, dict) or publication.get("adapter_hash") != current_adapter_hash:
                raise RuntimeErrorDetail("selftest Learner update did not publish the current adapter SHA")
            post_update = summary.get("post_update_generation")
            if (not isinstance(post_update, dict)
                    or post_update.get("policy_version") != current_policy["version"]
                    or post_update.get("group_size") != config.rollout.group_size
                    or post_update.get("reward_source") != "arc_service"):
                raise RuntimeErrorDetail("selftest lacks a complete ARC-scored post-update group")
            expected_worker_ids = {worker.id for worker in config.workers}
            pre_workers = summary.get("pre_update_worker_ids")
            post_workers = post_update.get("worker_ids")
            if (not isinstance(pre_workers, list) or not pre_workers or not set(pre_workers) <= expected_worker_ids
                    or not isinstance(post_workers, list) or len(post_workers) != config.rollout.group_size
                    or not set(post_workers) <= expected_worker_ids):
                raise RuntimeErrorDetail("selftest omitted its actual sampled Worker identities")
            _validate_selftest_alignment(
                summary.get("pre_update_logprob_alignment"),
                set(pre_workers),
                phase="pre-update",
            )
            _validate_selftest_alignment(
                summary.get("logprob_alignment"),
                set(post_workers),
                phase="post-update",
            )
            worker_calls = summary.get("worker_generation_calls")
            if not isinstance(worker_calls, dict) or any(
                    isinstance(worker_calls.get(worker.id), bool)
                    or not isinstance(worker_calls.get(worker.id), int)
                    or worker_calls[worker.id] < 1 for worker in config.workers):
                raise RuntimeErrorDetail("selftest did not generate at least once on each Worker")
            post_worker_calls = post_update.get("worker_generation_calls")
            if not isinstance(post_worker_calls, dict) or any(
                    isinstance(post_worker_calls.get(worker.id), bool)
                    or not isinstance(post_worker_calls.get(worker.id), int)
                    or post_worker_calls[worker.id] < 1 for worker in config.workers):
                raise RuntimeErrorDetail("selftest did not regenerate on every Worker after update ACK")
            if not isinstance(summary.get("policy_acks"), dict) \
                    or set(summary["policy_acks"]) != {worker.id for worker in config.workers}:
                raise RuntimeErrorDetail("selftest did not receive ACKs from all configured workers")
            expected_files = {item["name"]: {"sha256": item["sha256"].lower()}
                              for item in current_policy["files"]}
            expected_adapter_hash = _policy_snapshot_identity(current_policy).adapter_sha256
            if summary.get("policy_version") != current_policy["version"]:
                raise RuntimeErrorDetail("selftest result and current Learner policy version disagree")
            for worker in config.workers:
                ack = summary["policy_acks"][worker.id]
                if not isinstance(ack, dict) or ack.get("version") != current_policy["version"] \
                        or ack.get("worker_id") != worker.id or ack.get("files") != expected_files \
                        or ack.get("adapter_hash") != expected_adapter_hash:
                    raise RuntimeErrorDetail(f"Worker {worker.id} adapter ACK does not match current Learner policy")
            return state
        time.sleep(1.0)
    raise RuntimeErrorDetail("timed out waiting for real-model identity connectivity selftest")


def _request_resume(
    config: RuntimeConfig, lineage_id: str, *, timeout: float,
) -> dict[str, Any]:
    response = _request("POST", f"{config.learner_url}/resume", timeout=timeout,
                        json={"lineage_id": lineage_id})
    if response.status_code not in {200, 202}:
        raise RuntimeErrorDetail(f"learner POST /resume returned HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeErrorDetail("learner POST /resume returned an invalid response")
    if response.status_code == 200:
        if payload.get("status") != "no_pending_run" or payload.get("run_id") is not None:
            raise RuntimeErrorDetail("learner POST /resume returned an invalid no-pending-run response")
        return payload
    remaining = payload.get("remaining_steps")
    if (payload.get("status") not in {"running", "succeeded", "filtered_limit", "failed", "cancelled"}
            or payload.get("resumed") is not True
            or not isinstance(payload.get("run_id"), str) or not payload["run_id"]
            or isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0):
        raise RuntimeErrorDetail("learner POST /resume returned an invalid resumed-run response")
    return payload


def _request_resume_once(
    config: RuntimeConfig,
    manifest: dict[str, Any],
    manifest_path: Path,
    lineage_id: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    """Persist resume intent before the POST so uncertain responses are never replayed."""
    previous = manifest.get("resume_activation")
    if isinstance(previous, dict):
        state = previous.get("state")
        if state in {"accepted", "no_pending_run"}:
            result = previous.get("response")
            return result if isinstance(result, dict) else {
                "status": "accepted" if state == "accepted" else "no_pending_run",
                "run_id": previous.get("run_id"),
            }
        raise RuntimeErrorDetail(
            "a prior resume POST has an unknown outcome; inspect the Learner run before requesting another resume"
        )
    manifest["resume_activation"] = {
        "state": "requesting",
        "lineage_id": lineage_id,
        "requested_at": time.time(),
    }
    write_json_atomic(manifest_path, manifest)
    try:
        response = _request_resume(config, lineage_id, timeout=timeout)
    except BaseException as exc:
        manifest["resume_activation"]["state"] = "outcome_unknown"
        manifest["resume_activation"]["failure_type"] = type(exc).__name__
        manifest["resume_activation"]["finished_at"] = time.time()
        write_json_atomic(manifest_path, manifest)
        if isinstance(exc, Exception):
            raise RuntimeErrorDetail(
                "resume request outcome is unknown; it will not be automatically resubmitted"
            ) from exc
        raise
    accepted = response.get("resumed") is True
    state = "accepted" if accepted else "no_pending_run"
    manifest["resume_activation"].update({
        "state": state,
        "run_id": response.get("run_id"),
        "finished_at": time.time(),
        "response": response,
    })
    manifest["resume"] = response
    write_json_atomic(manifest_path, manifest)
    return response


def start(config: RuntimeConfig, config_path: str | Path, *, ready_timeout: float = 900.0,
          selftest_timeout: float = 3600.0, foreground: bool = False,
          skip_selftest: bool = False, fresh: bool = False,
          start_teacher: bool = True,
          resume_interrupted_run: bool = True,
          worker_environment_file: str | Path | None = None) -> dict[str, Any]:
    if type(skip_selftest) is not bool:
        raise ValueError("skip_selftest must be a bool")
    if type(fresh) is not bool:
        raise ValueError("fresh must be a bool")
    if type(start_teacher) is not bool or type(resume_interrupted_run) is not bool:
        raise ValueError("start_teacher and resume_interrupted_run must be bool")
    if ready_timeout <= 0 or (not skip_selftest and selftest_timeout <= 0):
        raise ValueError("start timeouts must be positive")
    external_token_envs = {
        worker.auth_token_env for worker in config.workers
        if worker.scope in {"service", "gateway"}
    }
    if external_token_envs and worker_environment_file is None:
        raise RuntimeErrorDetail(
            "service and Gateway Workers require their owned private environment file for the Learner child"
        )
    resolved_worker_environment_file = (
        str(Path(worker_environment_file).expanduser().resolve())
        if worker_environment_file is not None and external_token_envs else None
    )
    config_path = Path(config_path).expanduser().resolve()
    with _manager_lock(config.run):
        current = _current_manifest(config)
        if current is not None:
            validate_environment(config)
            manifest_path, manifest = current
            current_live = any(
                isinstance(record, dict) and _owned_process_matches(record, manifest)
                for record in manifest.get("processes", [])
            )
            learner_endpoint = manifest.get("endpoints", {}).get("learner", {})
            active_environment_file = (learner_endpoint.get("environment_file")
                                       if isinstance(learner_endpoint, dict) else None)
            if current_live and active_environment_file != resolved_worker_environment_file:
                raise RuntimeErrorDetail(
                    "active runtime uses a different Worker environment-file binding; "
                    "the existing roles were left running"
                )
            resume_activation = manifest.get("resume_activation")
            stopped_accepted_resume = (
                isinstance(resume_activation, dict)
                and resume_activation.get("state") == "accepted"
                and manifest.get("status") == "stopped"
                and not current_live
            )
            if (isinstance(resume_activation, dict)
                    and resume_activation.get("state") in {"requesting", "outcome_unknown", "accepted"}
                    and not stopped_accepted_resume
                    and not _existing_live_run(config)):
                raise RuntimeErrorDetail(
                    "the active manifest records an accepted or uncertain resume; inspect its Learner run before starting again"
                )
            reused = _reuse_existing_runtime(
                config, config_path, manifest_path, manifest, fresh=fresh,
            )
            if reused is not None:
                if start_teacher or resume_interrupted_run:
                    activation = _activate_runtime_locked(
                        config, config_path, start_teacher=start_teacher,
                        resume_interrupted_run=resume_interrupted_run,
                    )
                    return {**reused, **activation}
                return reused
        else:
            validate_paths(config, role="cluster")
            validate_environment(config)
        _reject_unmanaged_runtime(config)
        if not fresh:
            recovery = _inspect_training_recovery(config)
            if recovery.get("status") not in {"resume", "empty"}:
                raise _recovery_gate_error(recovery, partial=False)
        validate_paths(config, role="cluster")
        if config.bepa.enabled and start_teacher:
            validate_bepa_environment(config)
        return _start_new_run(
            config, config_path, ready_timeout=ready_timeout,
            selftest_timeout=selftest_timeout, foreground=foreground,
            skip_selftest=skip_selftest, fresh=fresh,
            start_teacher=start_teacher,
            resume_interrupted_run=resume_interrupted_run and not fresh,
            worker_environment_file=resolved_worker_environment_file,
        )


def _start_new_run(config: RuntimeConfig, config_path: Path, *, ready_timeout: float,
                   selftest_timeout: float, foreground: bool, skip_selftest: bool,
                   fresh: bool, start_teacher: bool = True,
                   resume_interrupted_run: bool = True,
                   worker_environment_file: str | None = None) -> dict[str, Any]:
    deployed_config = config
    config, membership = _external_worker_snapshot(deployed_config)
    external_token_envs = tuple(sorted({worker.auth_token_env for worker in config.workers
                                        if worker.scope in {"gateway", "service"}}))
    if external_token_envs and worker_environment_file is None:
        raise RuntimeErrorDetail("active service Workers require an owned private environment file")
    run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:10]}"
    run_root, manifest_path = _run_location(config, run_id)
    run_dir = manifest_path.parent
    run_dir.mkdir(parents=True, exist_ok=False)
    learner_process_gpu = (
        config.learner.gpus[0]
        if config.learner.parallelism == "single" and config.model_backend == "ternary"
        else None
    )
    recovery_dir = config.run / "training"
    recovery_lineage_id = uuid.uuid4().hex if fresh else None
    manifest: dict[str, Any] = {
        "schema_version": "qwen-rl-run/v1",
        "run_id": run_id,
        "created_at": time.time(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config_compatibility_sha256": _config_compatibility_sha256(deployed_config),
        "source_fingerprint_at_launch": _source_fingerprint(),
        "recovery": {
            "requested_mode": "fresh" if fresh else "auto",
            "recovery_dir": str(recovery_dir.resolve()),
            "requested_lineage_id": recovery_lineage_id,
            "resumed": None,
            "lineage_id": recovery_lineage_id,
            "loaded_snapshot": None,
            "pending_run": None,
        },
        "effective_config": str(run_dir / "runtime.yaml"),
        "run_dir": str(run_dir),
        "membership": membership,
        "topology": {"node_scope": {"site_id": config.node_scope.site_id,
                                      "node_id": config.node_scope.node_id},
                     "learner_gpus": list(config.learner.gpus),
                     "parallelism": config.learner.parallelism,
                     "workers": {worker.id: worker.gpu for worker in config.workers},
                     "worker_scopes": {worker.id: {"scope": worker.scope,
                         "site_id": worker.site_id, "node_id": worker.node_id,
                         "gpu": worker.gpu, "port": worker.port}
                         for worker in config.workers}},
        "endpoints": {"learner": {"port": config.learner.port,
                                    "gpus": list(config.learner.gpus),
                                    "parallelism": config.learner.parallelism,
                                    "process_gpu": learner_process_gpu,
                                    "environment_file": worker_environment_file},
                      "workers": {worker.id: {"port": worker.port, "gpu": worker.gpu,
                                               "scope": worker.scope,
                                               "site_id": worker.site_id,
                                               "node_id": worker.node_id,
                                               "url": worker.url,
                                               "url_prefix": worker.url_prefix,
                                               "auth_token_env": worker.auth_token_env,
                                               "boot_id": worker.boot_id,
                                               "telemetry_epoch": worker.telemetry_epoch}
                                  for worker in config.workers},
                      "teacher": ({"worker_id": f"teacher-{run_id}", "port": 0, "gpu": -1}
                                  if config.bepa.enabled and start_teacher else None)},
        "processes": [],
        "status": "starting",
        "policy_acks": {},
        "selftest": None,
    }
    if config.model_backend == "ternary":
        from ..contracts.ternary import TERNARY_QG_LAYOUT_VERSION

        manifest["model_identity"] = {
            "backend": "ternary",
            "quantization_mode": "ternary",
            "artifact_identity_path": str(Path(config.artifact_identity_path).resolve()),
            "artifact_id": config.artifact_identity["source"]["ninfer_artifact"]["artifact_id"],
            "artifact_sha256": _config_artifact_sha256(config),
            "qg_layout": TERNARY_QG_LAYOUT_VERSION,
        }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(run_root / "active.json", {"run_id": run_id,
                                                  "manifest": str(manifest_path.relative_to(run_root))})
    effective_config_path = run_dir / "runtime.yaml"
    effective_config = _effective_runtime_config(
        config, run_dir / "artifacts", recovery_dir,
        recovery_mode="fresh" if fresh else "auto",
        lineage_id=recovery_lineage_id,
    )
    effective_config_path.write_text(json.dumps(effective_config, ensure_ascii=False, indent=2) + "\n")
    manifest["effective_config_sha256"] = hashlib.sha256(
        effective_config_path.read_bytes()).hexdigest()
    write_json_atomic(manifest_path, manifest)
    try:
        learner_secret_env_names = tuple(sorted({
            config.bepa.teacher.api_key_env,
            *external_token_envs,
        }))
        learner = _spawn(manifest, manifest_path, role="learner", worker_id=None,
                         port=config.learner.port, gpu=learner_process_gpu,
                         config_path=effective_config_path, source_config=config_path, run_dir=run_dir,
                         secret_env_names=learner_secret_env_names,
                         environment_file=worker_environment_file)

        # Spawn the worker pool immediately so its model load overlaps the learner's load.
        # Workers only need the learner once the policy sync starts, and a learner failure tears
        # the whole cluster down below, so waiting for the learner first only serialised startup.
        worker_processes = []
        for worker in config.workers:
            if worker.scope != "local":
                continue
            record = _spawn(manifest, manifest_path, role="worker", worker_id=worker.id,
                            port=worker.port, gpu=worker.gpu,
                            config_path=effective_config_path, source_config=config_path, run_dir=run_dir,
                            secret_env_names=(config.bepa.teacher.api_key_env,
                                              config.bepa.teacher.token_env,
                                              *external_token_envs))
            worker_processes.append(record)

        readiness_processes = [learner, *worker_processes]
        _wait_until(lambda: _ready_role_health(
            config, "learner", readiness_processes, manifest,
        ), ready_timeout, "Learner /health")
        learner["ready_at"] = time.time()
        learner["ready_seconds"] = learner["ready_at"] - learner["started_at"]
        learner["state"] = "ready"
        write_json_atomic(manifest_path, manifest)

        for worker in config.workers:
            if worker.scope != "local":
                continue
            _wait_until(lambda w=worker: _ready_role_health(
                config, "worker", readiness_processes, manifest, w.id,
            ), ready_timeout, f"Worker {worker.id} vLLM /health")
            record = next(item for item in worker_processes if item["worker_id"] == worker.id)
            record["ready_at"] = time.time()
            record["ready_seconds"] = record["ready_at"] - record["started_at"]
            record["state"] = "ready"
            write_json_atomic(manifest_path, manifest)

        policy_sync_started = time.monotonic()
        manifest["policy_acks"] = _sync_initial_policy(config, timeout=ready_timeout)
        membership = manifest["membership"]
        startup_workers = membership.get("startup_workers", [])
        active_policy_versions = {
            result.get("version") for result in manifest["policy_acks"].values()
            if isinstance(result, dict)
        }
        if len(active_policy_versions) != 1:
            raise RuntimeErrorDetail("initial policy ACKs do not share one active Learner version")
        active_version = next(iter(active_policy_versions))
        if type(active_version) is not int or active_version < 0:
            raise RuntimeErrorDetail("initial policy ACKs omitted a valid active Learner version")
        membership["active_workers"] = [
            {**row, "active": True, "policy_acked": True,
             "policy_version": active_version,
             **({"eligible_after_policy_sync": True}
                if row.get("scope") in {"gateway", "service"} else {})}
            for row in startup_workers if isinstance(row, dict)
        ]
        membership["phase"] = "active"
        membership["activated_at"] = time.time()
        membership["policy_version"] = active_version
        manifest["timings"] = {"policy_sync_seconds": time.monotonic() - policy_sync_started}
        manifest["status"] = "ready"
        write_json_atomic(manifest_path, manifest)

        learner_health = _health(config, "learner")
        if not learner_health or not _is_quantized_learner(learner_health, config.quantization.mode, _config_artifact_sha256(config)):
            raise RuntimeErrorDetail("Learner health became unavailable after initial policy ACKs")
        if not _learner_recovery_is_clean(learner_health):
            raise RuntimeErrorDetail(
                "Learner policy is newer than its durable full training checkpoint; refusing startup selftest or resume"
            )
        recovery = learner_health.get("recovery")
        if (not isinstance(recovery, dict) or type(recovery.get("resumed")) is not bool
                or not isinstance(recovery.get("lineage_id"), str)
                or not recovery["lineage_id"]):
            raise RuntimeErrorDetail(
                "Learner health omitted verified recovery metadata; refusing to run a startup optimizer selftest"
            )
        if fresh and (recovery["resumed"] or recovery["lineage_id"] != recovery_lineage_id):
            raise RuntimeErrorDetail("Learner did not create the requested fresh training lineage")
        manifest["recovery"] = {
            **manifest["recovery"],
            "resumed": recovery["resumed"],
            "lineage_id": recovery["lineage_id"],
            "mode": recovery.get("mode"),
            "loaded_snapshot": recovery.get("loaded_snapshot"),
            "pending_run": recovery.get("pending_run"),
            "engine_epoch": recovery.get("engine_epoch"),
        }
        write_json_atomic(manifest_path, manifest)

        if recovery["resumed"]:
            manifest["selftest_skipped_reason"] = "restored_training_state"
            manifest["status"] = "resumed_ready"
            write_json_atomic(manifest_path, manifest)
        elif not skip_selftest:
            manifest["selftest"] = _trigger_selftest(config, timeout=selftest_timeout)
            manifest["status"] = "selftest_succeeded"
            write_json_atomic(manifest_path, manifest)
        else:
            manifest["selftest_skipped_reason"] = "requested"
            manifest["status"] = "ready"
            write_json_atomic(manifest_path, manifest)
        if config.bepa.enabled and start_teacher:
            teacher_id = f"teacher-{run_id}"
            teacher = _spawn(
                manifest, manifest_path, role="teacher", worker_id=teacher_id,
                port=0, gpu=-1, config_path=effective_config_path,
                source_config=config_path, run_dir=run_dir,
                secret_env_names=external_token_envs,
            )
            teacher["state"] = "running"
            teacher["started_after_policy_acks"] = bool(manifest.get("policy_acks"))
            write_json_atomic(manifest_path, manifest)

        pending_run = recovery.get("pending_run")
        if (resume_interrupted_run and recovery["resumed"]
                and pending_run not in (None, False)):
            lineage_id = recovery["lineage_id"]
            resume_result = _request_resume_once(
                config, manifest, manifest_path, lineage_id,
                timeout=config.learner.request_timeout,
            )
            if resume_result.get("resumed") is True:
                manifest["status"] = "resume_started"
                write_json_atomic(manifest_path, manifest)
    except BaseException:
        manifest["status"] = "startup_failed"
        write_json_atomic(manifest_path, manifest)
        stopped = stop_manifest(manifest, manifest_path, timeout=10.0)
        manifest["status"] = "startup_failed"
        manifest["stop_outcomes"] = stopped["stop_outcomes"]
        write_json_atomic(manifest_path, manifest)
        raise
    result = {"run_id": run_id, "status": manifest["status"],
              "manifest": str(manifest_path), "selftest": manifest["selftest"],
              "selftest_skipped_reason": manifest.get("selftest_skipped_reason"),
              "recovery": manifest.get("recovery"), "resume": manifest.get("resume"),
              "source_fingerprint_at_launch": manifest.get("source_fingerprint_at_launch"),
              "reused": False, "code_hot_reloaded": False}
    if foreground:
        try:
            while any(_owned_process_matches(record, manifest)
                     for record in manifest.get("processes", []) if isinstance(record, dict)):
                time.sleep(1.0)
        except KeyboardInterrupt:
            stop_manifest(manifest, manifest_path, timeout=15.0)
    return result


def status(config: RuntimeConfig, config_path: str | Path | None = None) -> dict[str, Any]:
    current = _current_manifest(config)
    if current is None:
        return {"status": "stopped", "run_id": None, "processes": [],
                "reuse_ready": False, "config_compatible": False,
                "source_fingerprint_at_launch": None}
    manifest_path, manifest = current
    source_path = (Path(config_path).expanduser().resolve() if config_path is not None
                   else Path(str(manifest.get("config", ""))).expanduser())
    config_compatible = _manifest_config_matches(manifest, config, source_path)
    records = manifest.get("processes", [])
    processes = []
    live_roles: list[tuple[str, str | None]] = []
    role_health: dict[tuple[str, str | None], dict[str, Any] | None] = {}
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        is_live = _owned_process_matches(record, manifest)
        role = record.get("role")
        worker_id = record.get("worker_id")
        health = (_health(config, role, record.get("worker_id")) if is_live and
                  role in {"learner", "worker"} else None)
        if is_live:
            live_roles.append((role, worker_id))
            role_health[(role, worker_id)] = health
        processes.append({"role": role, "worker_id": worker_id,
                          "pid": record.get("pid"), "starttime": record.get("starttime"),
                          "port": record.get("port"), "gpu": record.get("gpu"),
                          "identity_matches": is_live, "health": health})
    expected = _expected_runtime_roles(config, manifest)
    healthy = bool(expected) and set(live_roles) == expected and len(live_roles) == len(expected)
    if healthy:
        for role, worker_id in expected:
            if role == "learner":
                health = role_health.get((role, worker_id))
                healthy = bool(health and _is_quantized_learner(health, config.quantization.mode, _config_artifact_sha256(config))
                               and _learner_recovery_is_clean(health))
            elif role == "worker":
                health = role_health.get((role, worker_id))
                healthy = bool(isinstance(worker_id, str) and health
                               and _is_runtime_worker(health, worker_id, config.quantization.mode, _config_artifact_sha256(config)))
            if not healthy:
                break
    remote_healthy, remote_errors = _active_remote_health(config, manifest)
    healthy = healthy and remote_healthy
    return {"status": manifest.get("status", "unknown"), "run_id": manifest.get("run_id"),
            "manifest": str(manifest_path), "processes": processes,
            "policy_acks": manifest.get("policy_acks", {}),
            "membership": manifest.get("membership"),
            "remote_membership_healthy": remote_healthy,
            "remote_membership_errors": remote_errors,
            "selftest": manifest.get("selftest"),
            "config_compatible": config_compatible,
            "reuse_ready": bool(config_compatible and healthy),
            "source_fingerprint_at_launch": manifest.get("source_fingerprint_at_launch"),
            "recovery": manifest.get("recovery"),
            "resume": manifest.get("resume")}


def validate_bepa_environment(config: RuntimeConfig) -> None:
    """Fail before GPU startup when the enabled CPU teacher lacks isolated credentials."""
    teacher = config.bepa.teacher
    names = {teacher.api_key_env, teacher.token_env}
    if (teacher.api_key_env == teacher.token_env or "ARC_ENV_TOKEN" in names
            or not teacher.api_key_env or not teacher.token_env):
        raise RuntimeErrorDetail("BEPA provider and teacher token environment names must be distinct from ARC_ENV_TOKEN")
    missing = [name for name in (teacher.api_key_env, teacher.token_env) if not os.environ.get(name)]
    if missing:
        raise RuntimeErrorDetail("BEPA teacher credentials are missing from environment: " + ", ".join(missing))
    arc_token = os.environ.get("ARC_ENV_TOKEN")
    if arc_token and (os.environ[teacher.token_env] == arc_token or os.environ[teacher.api_key_env] == arc_token):
        raise RuntimeErrorDetail("BEPA credentials must use values distinct from ARC_ENV_TOKEN")
    try:
        sdk_version = metadata.version("openai")
    except metadata.PackageNotFoundError:
        raise RuntimeErrorDetail("BEPA teacher requires the optional teacher dependencies: install .[teacher]") from None
    if sdk_version != "3.26.1":
        raise RuntimeErrorDetail("BEPA teacher requires openai==3.26.1; install .[teacher] before GPU startup")


def run_training(config: RuntimeConfig, *, steps: int, task_ids: list[str] | None = None) -> dict[str, Any]:
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if task_ids is not None and (
            not isinstance(task_ids, list) or not task_ids
            or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise ValueError("task_ids must be a non-empty list of distinct non-empty strings")
    body: dict[str, Any] = {"kind": "training", "steps": steps}
    if task_ids is not None:
        body["task_ids"] = task_ids
    response = _request("POST", f"{config.learner_url}/run", timeout=config.learner.request_timeout, json=body)
    if response.status_code != 202:
        raise RuntimeErrorDetail(f"learner POST /run training returned HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("run_id"), str):
        raise RuntimeErrorDetail("learner POST /run returned an invalid response")
    return payload


def bepa_request(
    config: RuntimeConfig,
    operation: str,
    *,
    task_ids: list[str] | None = None,
    max_job_tokens: int | None = None,
    max_total_tokens: int | None = None,
    job_id: str | None = None,
    plan_id: str | None = None,
    evidence_ref: str | None = None,
    reason: str | None = None,
    accept_unreported_usage: bool = False,
) -> dict[str, Any] | list[Any]:
    if operation not in {"bootstrap", "status", "jobs", "plans", "forget-plan", "budget", "settle-lower-bound"}:
        raise ValueError("unknown BEPA operation")
    if operation == "bootstrap":
        if task_ids is not None and (
                not task_ids or any(not isinstance(item, str) or not item for item in task_ids)
                or len(set(task_ids)) != len(task_ids)):
            raise ValueError("task_ids must be distinct non-empty strings when supplied")
        body: dict[str, Any] = {}
        if task_ids is not None:
            body["task_ids"] = task_ids
        response = _request("POST", f"{config.learner_url}/bepa/bootstrap",
                            timeout=config.learner.request_timeout, json=body)
        if response.status_code not in {200, 202}:
            raise RuntimeErrorDetail(f"learner POST /bepa/bootstrap returned HTTP {response.status_code}")
    elif operation == "forget-plan":
        if not isinstance(plan_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", plan_id) is None:
            raise ValueError("forget-plan requires a valid --plan-id")
        response = _request("POST", f"{config.learner_url}/bepa/plans/{plan_id}/forget",
                            timeout=config.learner.request_timeout)
        if response.status_code != 200:
            raise RuntimeErrorDetail(f"learner BEPA forget-plan returned HTTP {response.status_code}")
    elif operation == "settle-lower-bound":
        if (accept_unreported_usage is not True or not isinstance(job_id, str)
                or re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", job_id) is None
                or not isinstance(evidence_ref, str) or re.fullmatch(r"[0-9a-f]{64}", evidence_ref) is None
                or not isinstance(reason, str) or not reason.strip() or len(reason) > 500):
            raise ValueError("settle-lower-bound requires job/evidence/reason and --accept-unreported-usage")
        response = _request("POST", f"{config.learner_url}/bepa/jobs/{job_id}/settle-lower-bound",
                            timeout=config.learner.request_timeout,
                            json={"evidence_ref": evidence_ref, "reason": reason,
                                  "accept_unreported_usage": True})
        if response.status_code != 200:
            raise RuntimeErrorDetail(f"learner BEPA lower-bound settlement returned HTTP {response.status_code}")
    elif operation == "budget":
        if max_job_tokens is None and max_total_tokens is None:
            raise ValueError("budget requires --max-job-tokens and/or --max-total-tokens")
        body = {key: value for key, value in {
            "max_job_tokens": max_job_tokens, "max_total_tokens": max_total_tokens,
        }.items() if value is not None}
        response = _request("POST", f"{config.learner_url}/bepa/budget",
                            timeout=config.learner.request_timeout, json=body)
        if response.status_code != 200:
            raise RuntimeErrorDetail(f"learner POST /bepa/budget returned HTTP {response.status_code}")
    else:
        response = _request("GET", f"{config.learner_url}/bepa/{operation}",
                            timeout=config.learner.request_timeout)
        if response.status_code != 200:
            raise RuntimeErrorDetail(f"learner GET /bepa/{operation} returned HTTP {response.status_code}")
    payload = response.json()
    if operation in {"jobs", "plans"} and isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise RuntimeErrorDetail(f"learner BEPA {operation} returned an invalid response")
    return payload


async def run_teacher(config: RuntimeConfig, *, worker_id: str, once: bool = False) -> dict[str, Any]:
    if not config.bepa.enabled:
        raise RuntimeErrorDetail("BEPA teacher runner requires bepa.enabled=true")
    validate_bepa_environment(config)
    os.environ.pop("ARC_ENV_TOKEN", None)
    from ..bepa.teacher import TeacherProviderConfig, TeacherRunner, TeacherRunnerError

    teacher = config.bepa.teacher
    budget = teacher.budget
    runner = TeacherRunner(
        central_base_url=config.learner_url,
        provider=TeacherProviderConfig(
            base_url=teacher.base_url,
            model=teacher.model,
            provider="hp-hermes-router",
            api_key_env=teacher.api_key_env,
            timeout_seconds=budget.job_timeout_seconds,
            max_completion_tokens=teacher.max_completion_tokens,
        ),
        central_token_env=teacher.token_env,
        max_job_tokens=budget.max_job_tokens,
        max_total_tokens=budget.max_total_tokens,
        job_timeout_seconds=budget.job_timeout_seconds,
        evidence_dir=Path(config.bepa.state_dir) / "teacher-outbox",
    )
    try:
        return await runner.run_forever(worker_id=worker_id, once=once)
    except TeacherRunnerError as exc:
        raise RuntimeErrorDetail(json.dumps(exc.safe_diagnostic(), sort_keys=True)) from None


def stop_manifest(manifest: dict[str, Any], manifest_path: Path, *, timeout: float = 15.0) -> dict[str, Any]:
    if timeout <= 0:
        raise ValueError("shutdown timeout must be positive")
    records = [record for record in manifest.get("processes", []) if isinstance(record, dict)]
    ordered = list(reversed(records))
    outcomes: dict[str, str] = {}

    def signal_matching(sig: int, label: str) -> None:
        for record in ordered:
            key = str(record.get("worker_id") or record.get("role"))
            if not _owned_process_matches(record, manifest):
                outcomes.setdefault(key, "exited" if not process_is_running(record.get("pid", -1))
                                    else "identity_mismatch")
                continue
            try:
                os.kill(record["pid"], sig)
                outcomes[key] = label
            except ProcessLookupError:
                outcomes[key] = "exited"

    def wait_gone(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not any(_owned_process_matches(record, manifest) for record in ordered):
                return True
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return not any(_owned_process_matches(record, manifest) for record in ordered)

    signal_matching(signal.SIGINT, "SIGINT")
    if not wait_gone(timeout * 0.5):
        signal_matching(signal.SIGTERM, "SIGTERM")
    if not wait_gone(timeout * 0.35):
        signal_matching(signal.SIGKILL, "SIGKILL")
    wait_gone(timeout * 0.15)
    manifest["status"] = "stopped" if not any(
        _owned_process_matches(record, manifest) for record in ordered) else "stop_incomplete"
    manifest["stop_outcomes"] = outcomes
    manifest["stopped_at"] = time.time()
    write_json_atomic(manifest_path, manifest)
    return {"run_id": manifest.get("run_id"), "status": manifest["status"],
            "stop_outcomes": outcomes}


def stop(config: RuntimeConfig, *, timeout: float = 15.0) -> dict[str, Any]:
    current = _current_manifest(config)
    if current is None:
        return {"status": "stopped", "run_id": None, "stop_outcomes": {}}
    manifest_path, manifest = current
    return stop_manifest(manifest, manifest_path, timeout=timeout)


def request_evaluation(
    config: RuntimeConfig, *, read_only: bool = False, retry_errors: str | None = None,
) -> dict[str, Any]:
    if retry_errors is not None:
        if read_only or re.fullmatch(r"eval-[0-9a-f]{32}", retry_errors) is None:
            raise RuntimeErrorDetail("evaluation retry requires an exact request ID and cannot be read-only")
        method = "POST"
        endpoint = f"{config.learner_url}/evaluation/{retry_errors}/retry-errors"
    else:
        method = "GET" if read_only else "POST"
        endpoint = f"{config.learner_url}/evaluation"
    response = _request(method, endpoint, timeout=config.learner.request_timeout)
    if response.status_code != 200:
        raise RuntimeErrorDetail(f'learner evaluation returned HTTP {response.status_code}: {response.text[:300]}')
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeErrorDetail('evaluation endpoint returned invalid metadata')
    return value


def request_selftest(config: RuntimeConfig, *, timeout: float = 3600.0) -> dict[str, Any]:
    if timeout <= 0:
        raise ValueError("selftest timeout must be positive")
    current = _current_manifest(config)
    if current is None or not any(
            isinstance(record, dict) and _owned_process_matches(record, current[1])
            for record in current[1].get("processes", [])):
        raise RuntimeErrorDetail("no active run in the configured run_dir")
    manifest_path, manifest = current
    active_config = _manifest_active_config(config, manifest)
    result = _trigger_selftest(active_config, timeout)
    manifest["selftest"] = result
    write_json_atomic(manifest_path, manifest)
    return result
