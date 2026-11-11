"""Build and verify a private, CPU-prepared Vast Worker deployment package."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .config import (
    VastDeploymentError,
    VastLearnerProfile,
    VastProfile,
    VastWorkerRoute,
    digest_pinned_image,
    load_model_manifest,
    load_vast_learner_profile,
    load_vast_profile,
    loopback_http_url,
    parse_json_bytes,
    read_regular_file,
    validate_deployment_payload,
)

_PRIVATE_MODE = 0o600
_PRIVATE_DIR_MODE = 0o700
_GB = 1_000_000_000  # Vast's --disk argument is expressed in decimal GB.
_ATTACH_RECEIPT = re.compile(r"^state/attach-[0-9a-f]{32}\.json$")
_ATTACH_STATUSES = frozenset({
    "worker_started", "learner_ready", "readiness_timeout", "failed",
    "ssh_session_ended", "ownership_lost",
})


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PRIVATE_MODE)
    try:
        os.fchmod(descriptor, _PRIVATE_MODE)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _ensure_private_output_root(path: Path) -> None:
    """Create a new output root securely or require an existing private root."""
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if cursor.exists() and (cursor.is_symlink() or not cursor.is_dir()):
        raise VastDeploymentError("output_root has a symlink or non-directory component")
    for candidate in reversed(missing):
        try:
            candidate.mkdir(mode=_PRIVATE_DIR_MODE)
            os.chmod(candidate, _PRIVATE_DIR_MODE)
        except FileExistsError:
            if candidate.is_symlink() or not candidate.is_dir():
                raise VastDeploymentError("output_root has a symlink or non-directory component")
    if path.is_symlink() or not path.is_dir():
        raise VastDeploymentError("output_root must be a regular directory")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise VastDeploymentError("existing output_root must be private (no group/other permissions)")


def _container_file_path(value: Any, *, label: str, root: str) -> str:
    if not isinstance(value, str) or not value:
        raise VastDeploymentError(f"worker.json inference.{label} is required for the T2 artifact handoff")
    path = PurePosixPath(value)
    model_root = PurePosixPath(root)
    if (not path.is_absolute() or ".." in path.parts or str(path) != value
            or path == model_root or model_root not in path.parents):
        raise VastDeploymentError(f"worker.json inference.{label} must be a normalized file under model_root_container")
    return str(path.relative_to(model_root))


def _reject_embedded_worker_credentials(value: Any, *, path: str = "worker.json") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (normalized in {"token", "worker_token", "arc_env_token", "api_key", "apikey"}
                    or any(word in normalized for word in ("password", "secret", "bearer"))):
                raise VastDeploymentError(
                    f"target worker.json contains credential field {path}.{key}; attach token separately"
                )
            _reject_embedded_worker_credentials(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_embedded_worker_credentials(child, path=f"{path}[{index}]")


def _validate_worker_config(worker_bytes: bytes, *, worker_port: int,
                            remote_policy_port: int, model_root_container: str,
                            manifest_paths: set[str]) -> dict[str, Any]:
    worker = parse_json_bytes(worker_bytes, label="target worker.json")
    if not isinstance(worker, dict):
        raise VastDeploymentError("target worker.json must be a JSON object")
    _reject_embedded_worker_credentials(worker)
    worker_id = worker.get("worker_id")
    if not isinstance(worker_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", worker_id):
        raise VastDeploymentError("target worker.json must provide a safe non-empty worker_id")
    if worker.get("host") != "127.0.0.1":
        raise VastDeploymentError("target worker.json host must be 127.0.0.1 for the SSH-only API tunnel")
    if (isinstance(worker.get("port"), bool) or not isinstance(worker.get("port"), int)
            or worker.get("port") != worker_port):
        raise VastDeploymentError("target worker.json port must equal worker_port")
    expected_policy_url = f"http://127.0.0.1:{remote_policy_port}"
    if worker.get("learner_url") != expected_policy_url:
        raise VastDeploymentError("target worker.json learner_url must use the configured remote policy loopback port")
    if worker.get("deployment_scope") != "service" or worker.get("artifact_transport") != "direct":
        raise VastDeploymentError("target worker.json must use deployment_scope=service and artifact_transport=direct")

    inference = worker.get("inference")
    if not isinstance(inference, dict):
        raise VastDeploymentError("target worker.json must contain an inference object")
    if inference.get("backend") != "ninfer":
        raise VastDeploymentError("target worker.json must explicitly select inference.backend=ninfer")
    if inference.get("model_path") != model_root_container:
        raise VastDeploymentError("target worker.json inference.model_path must equal model_root_container")
    engine_url = inference.get("ninfer_url")
    if not isinstance(engine_url, str):
        raise VastDeploymentError("target worker.json must provide the NInfer engine loopback URL")
    _engine_origin, engine_port = loopback_http_url(engine_url, "worker.json inference.ninfer_url")
    if engine_port in {worker_port, remote_policy_port}:
        raise VastDeploymentError("NInfer engine port conflicts with a Worker or tunnel port")

    path_fields = ("ninfer_artifact_path", "ninfer_conversion_report")
    required_model_files = {
        _container_file_path(inference.get(key), label=key, root=model_root_container)
        for key in path_fields
    }
    # Preserve the exact target worker.json. Every other explicit artifact/file
    # path beneath model_path is also required to be listed in the immutable
    # manifest (for example future T2 identity or MTP payload sidecars).
    for key, value in inference.items():
        if (key in {"model_path", "ninfer_converter_path", "ninfer_converter_python"}
                or not key.endswith(("_path", "_file")) or not isinstance(value, str)
                or not value.startswith(model_root_container.rstrip("/") + "/")):
            continue
        required_model_files.add(_container_file_path(value, label=key, root=model_root_container))
    missing = sorted(required_model_files - manifest_paths)
    if missing:
        raise VastDeploymentError(
            "immutable model manifest is missing worker.json referenced T2/NInfer files: "
            + ", ".join(missing)
        )
    return worker


def _storage_receipt(profile: VastProfile, *, model_size_bytes: int, largest_file_bytes: int) -> dict[str, Any]:
    return _storage_receipt_values(
        disk_gb=profile.disk_gb,
        state_reserve_gb=profile.state_reserve_gb,
        image_layer_reserve_gb=profile.image_layer_reserve_gb,
        model_size_bytes=model_size_bytes,
        largest_file_bytes=largest_file_bytes,
    )


def _storage_receipt_values(*, disk_gb: int, state_reserve_gb: int,
                            image_layer_reserve_gb: int, model_size_bytes: int,
                            largest_file_bytes: int) -> dict[str, Any]:
    state_reserve = state_reserve_gb * _GB
    image_reserve = image_layer_reserve_gb * _GB
    required_bytes = model_size_bytes + largest_file_bytes + state_reserve + image_reserve
    disk_bytes = disk_gb * _GB
    if disk_bytes <= required_bytes:
        raise VastDeploymentError(
            "profile.disk_gb is too small: it must exceed asset/model bytes + largest-file staging bytes "
            "+ state_reserve_gb + image_layer_reserve_gb"
        )
    return {
        "disk_gb": disk_gb,
        "disk_bytes": disk_bytes,
        "model_bytes": model_size_bytes,
        "largest_file_staging_bytes": largest_file_bytes,
        "state_reserve_gb": state_reserve_gb,
        "image_layer_reserve_gb": image_layer_reserve_gb,
        "required_bytes_exclusive": required_bytes,
        "remaining_bytes_after_declared_reserves": disk_bytes - required_bytes,
        "image_layer_reserve_verified": False,
    }


def _start_script(profile: VastProfile) -> bytes:
    # Values are validated POSIX absolute paths without shell metacharacters.
    # quote() keeps deployment ids and path components inert even if policy is
    # widened in a future schema.
    import shlex

    remote_config = profile.remote_root + "/config"
    safe_deployment = remote_config + "/deployment.json"
    safe_worker_config = remote_config + "/worker.json"
    safe_manifest = remote_config + "/model-manifest.json"
    safe_token = profile.remote_root + "/secrets/worker_auth_token"
    safe_receipt = profile.remote_root + "/state/vast-download-receipt.json"
    default_script = f'''#!/usr/bin/env bash
set -Eeuo pipefail

ENTRYPOINT=/opt/qwen-vast/entrypoint.sh
NINFER=/usr/local/bin/ninfer-serve
EXPECTED_DEPLOYMENT={shlex.quote(safe_deployment)}
EXPECTED_WORKER_CONFIG={shlex.quote(safe_worker_config)}
EXPECTED_MODEL_MANIFEST={shlex.quote(safe_manifest)}
EXPECTED_TOKEN_FILE={shlex.quote(safe_token)}
EXPECTED_MODEL_ROOT={shlex.quote(profile.model_root_container)}
EXPECTED_RECEIPT={shlex.quote(safe_receipt)}

if [[ $# -eq 0 || "${{1:-}}" == "--wait-for-attach" ]]; then
  [[ -x "$ENTRYPOINT" ]] || {{ echo "qwen-vast: image entrypoint is missing" >&2; exit 70; }}
  [[ -x "$NINFER" ]] || {{ echo "qwen-vast: NInfer binary is missing" >&2; exit 70; }}
  echo "qwen-vast: image preflight passed; waiting for the SSH tunnel and private Worker bundle."
  exec sleep infinity
fi

deployment=""
worker_config=""
model_manifest=""
token_file=""
model_root="$EXPECTED_MODEL_ROOT"
receipt="$EXPECTED_RECEIPT"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deployment) deployment="$2"; shift 2 ;;
    --worker-config) worker_config="$2"; shift 2 ;;
    --model-manifest) model_manifest="$2"; shift 2 ;;
    --token-file) token_file="$2"; shift 2 ;;
    --model-root) model_root="$2"; shift 2 ;;
    --receipt) receipt="$2"; shift 2 ;;
    *) echo "qwen-vast: unsupported start-worker argument: $1" >&2; exit 64 ;;
  esac
done
[[ "$deployment" == "$EXPECTED_DEPLOYMENT" ]] || {{ echo "qwen-vast: deployment.json path differs from the prepared package" >&2; exit 64; }}
[[ "$worker_config" == "$EXPECTED_WORKER_CONFIG" ]] || {{ echo "qwen-vast: worker.json path differs from the prepared package" >&2; exit 64; }}
[[ "$model_manifest" == "$EXPECTED_MODEL_MANIFEST" ]] || {{ echo "qwen-vast: model manifest path differs from the prepared package" >&2; exit 64; }}
[[ "$token_file" == "$EXPECTED_TOKEN_FILE" ]] || {{ echo "qwen-vast: token path differs from the private secret location" >&2; exit 64; }}
[[ "$model_root" == "$EXPECTED_MODEL_ROOT" ]] || {{ echo "qwen-vast: model root differs from deployment.json" >&2; exit 64; }}
[[ "$receipt" == "$EXPECTED_RECEIPT" ]] || {{ echo "qwen-vast: receipt path differs from the private state location" >&2; exit 64; }}
[[ -f "$deployment" && ! -L "$deployment" ]] || {{ echo "qwen-vast: deployment.json is missing or unsafe" >&2; exit 66; }}
[[ -f "$worker_config" && ! -L "$worker_config" ]] || {{ echo "qwen-vast: worker.json is missing or unsafe" >&2; exit 66; }}
[[ -f "$model_manifest" && ! -L "$model_manifest" ]] || {{ echo "qwen-vast: model manifest is missing or unsafe" >&2; exit 66; }}
[[ -f "$token_file" && ! -L "$token_file" ]] || {{ echo "qwen-vast: Worker token file is missing or unsafe" >&2; exit 66; }}
token_mode="$(stat -c '%a' "$token_file")"
[[ "$token_mode" == "600" ]] || {{ echo "qwen-vast: Worker token file must have mode 0600" >&2; exit 77; }}
mkdir -p {shlex.quote(profile.remote_root + "/state")}
exec "$ENTRYPOINT" \\
  --worker-config "$worker_config" \\
  --token-file "$token_file" \\
  --manifest "$model_manifest" \\
  --model-root "$model_root" \\
  --receipt "$receipt"
'''
    return default_script.encode("utf-8")


def _create_recipe(profile: VastProfile, script_path: Path, storage: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "role": "worker",
        "provider": "vast.ai",
        "action": "create_instance",
        "status": "awaiting_operator_offer_id",
        "offer_id": None,
        "required_operator_inputs": ["offer_id from a separately reviewed Vast offer"],
        "parameters": {
            "image": profile.image_ref,
            "disk_gb": profile.disk_gb,
            "ssh": True,
            "onstart": str(script_path),
            "public_worker_api_ports": [],
            "network_mode": "ssh_tunnel",
        },
        "worker_network": {
            "central_worker_url": profile.worker_url,
            "remote_policy_url": profile.remote_policy_url,
            "learner_local_url": profile.learner_local_url,
            "public_worker_api": False,
        },
        "storage_calculation": storage,
        "cost_timing": {
            "image_layers_may_be_pulled_after_instance_creation": True,
            "cold_model_download_occurs_after_instance_creation": True,
            "prewarmed_vast_host_cache_is_not_assumed": True,
        },
        "command_template": [
            "vastai", "create", "instance", "<offer-id>",
            "--image", profile.image_ref,
            "--disk", str(profile.disk_gb),
            "--ssh",
            "--onstart", str(script_path),
        ],
        "executed": False,
    }


def _check_input(profile: VastProfile) -> tuple[bytes, bytes, Any, dict[str, Any], dict[str, Any]]:
    if not profile.worker_config_path.exists():
        raise VastDeploymentError(
            "T2 target worker.json has not been delivered; prepare will not synthesize a Worker config"
        )
    if profile.worker_config_path.is_symlink():
        raise VastDeploymentError("T2 target worker.json must be a regular file, not a symlink")
    worker_bytes = read_regular_file(profile.worker_config_path, label="target worker.json")
    if not profile.model_manifest_path.exists():
        raise VastDeploymentError(
            "T2 immutable model manifest has not been delivered; prepare will not invent model URLs or files"
        )
    if profile.model_manifest_path.is_symlink():
        raise VastDeploymentError("T2 model manifest must be a regular file, not a symlink")
    manifest = load_model_manifest(profile.model_manifest_path, profile.model_manifest_sha256)
    manifest_paths = {item.path for item in manifest.files}
    worker = _validate_worker_config(
        worker_bytes,
        worker_port=profile.worker_port,
        remote_policy_port=profile.remote_policy_port,
        model_root_container=profile.model_root_container,
        manifest_paths=manifest_paths,
    )
    largest = max(item.size for item in manifest.files)
    storage = _storage_receipt(profile, model_size_bytes=manifest.size_bytes, largest_file_bytes=largest)
    return worker_bytes, manifest.raw_bytes, manifest, worker, storage


def check_vast_worker_profile(path: str | Path) -> dict[str, Any]:
    """Perform only local filesystem/JSON checks; do not query Vast or start GPU work."""
    profile = load_vast_profile(path)
    _worker_bytes, _manifest_bytes, manifest, worker, storage = _check_input(profile)
    return {
        "status": "cpu_preflight_passed",
        "role": "worker",
        "deployment_id": profile.deployment_id,
        "worker_id": worker["worker_id"],
        "image_ref": profile.image_ref,
        "image_runtime_contract": "declared_not_validated",
        "runtime_not_validated": True,
        "model_files": len(manifest.files),
        "model_bytes": manifest.size_bytes,
        "worker_json_sha256": hashlib.sha256(_worker_bytes).hexdigest(),
        "model_manifest_sha256": manifest.sha256,
        "storage": storage,
        "network_mode": "ssh_tunnel",
        "worker_url": profile.worker_url,
        "remote_policy_url": profile.remote_policy_url,
        "executed_vast_api": False,
    }


def _parse_runtime_document(raw: bytes) -> dict[str, Any]:
    if raw.lstrip().startswith(b"{"):
        value = parse_json_bytes(raw, label="runtime config")
    else:
        try:
            import yaml
        except ImportError as exc:
            raise VastDeploymentError(
                "YAML runtime config cannot be parsed because PyYAML is unavailable"
            ) from exc
        class UniqueSafeLoader(yaml.SafeLoader):
            def construct_mapping(self, node, deep=False):
                mapping = {}
                for key_node, value_node in node.value:
                    key = self.construct_object(key_node, deep=deep)
                    if key in mapping:
                        raise yaml.constructor.ConstructorError(
                            "while constructing a mapping", node.start_mark,
                            f"duplicate key {key!r}", key_node.start_mark)
                    mapping[key] = self.construct_object(value_node, deep=deep)
                return mapping

        try:
            value = yaml.load(raw.decode("utf-8"), Loader=UniqueSafeLoader)
        except (UnicodeDecodeError, TypeError, yaml.YAMLError) as exc:
            raise VastDeploymentError("runtime config is not valid UTF-8 YAML") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise VastDeploymentError("runtime config must be an object with string keys")
    return value


def _check_runtime_has_no_inline_credentials(value: Any, *, path: str = "runtime config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            credential_marker = any(
                marker in normalized
                for marker in ("password", "secret", "api_key", "apikey", "bearer", "authorization")
            )
            token_value = normalized in {"token", "access_token", "refresh_token", "worker_token",
                                         "arc_env_token", "auth_token"}
            if ((credential_marker or token_value) and not normalized.endswith("_env")):
                raise VastDeploymentError(
                    f"runtime config contains inline credential field {path}.{key}; bind a credential environment name"
                )
            _check_runtime_has_no_inline_credentials(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_runtime_has_no_inline_credentials(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and "://" in value:
        try:
            parsed = urlsplit(value)
            _ = parsed.port
        except ValueError as exc:
            raise VastDeploymentError(f"runtime config {path} contains an invalid URL") from exc
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise VastDeploymentError(f"runtime config {path} URL must not embed credentials or query data")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise VastDeploymentError(f"runtime config {label} must be an object")
    return value


def _runtime_container_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise VastDeploymentError(f"runtime config {label} must be an absolute container path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value or path == PurePosixPath("/"):
        raise VastDeploymentError(f"runtime config {label} must be a normalized absolute path")
    return path


def _path_within(path: PurePosixPath, root: PurePosixPath, *, strict: bool = False) -> bool:
    if path == root:
        return not strict
    return root in path.parents


def _require_asset_reference(path: PurePosixPath, *, asset_root: PurePosixPath,
                             manifest_paths: set[str], label: str,
                             directory: bool = False) -> None:
    if not _path_within(path, asset_root):
        raise VastDeploymentError(f"runtime config {label} must be inside asset_root_container")
    relative = path.relative_to(asset_root).as_posix()
    if directory:
        if relative == "." or not any(item.startswith(relative.rstrip("/") + "/") for item in manifest_paths):
            raise VastDeploymentError(f"runtime config {label} has no files in the immutable asset manifest")
    elif relative not in manifest_paths:
        raise VastDeploymentError(f"runtime config {label} is not listed in the immutable asset manifest")


def _require_resource_reference(path: PurePosixPath, *, asset_root: PurePosixPath,
                                state_root: PurePosixPath, remote_root: PurePosixPath,
                                manifest_paths: set[str], label: str,
                                baked_resources: list[str]) -> None:
    if _path_within(path, state_root) or _path_within(path, remote_root):
        raise VastDeploymentError(f"runtime config {label} cannot refer to Learner state or SSH staging paths")
    if _path_within(path, asset_root):
        _require_asset_reference(path, asset_root=asset_root, manifest_paths=manifest_paths,
                                 label=label)
        return
    work_root = PurePosixPath("/work")
    if _path_within(path, work_root, strict=True):
        baked_resources.append(str(path))
        return
    raise VastDeploymentError(
        f"runtime config {label} must be in asset_root_container or an explicit baked /work resource"
    )


def _check_worker_routes(runtime: dict[str, Any], profile: VastLearnerProfile) -> set[str]:
    raw_workers = runtime.get("workers")
    if not isinstance(raw_workers, list) or not raw_workers:
        raise VastDeploymentError("runtime config workers must be a non-empty list of external Workers")
    route_by_id = {route.worker_id: route for route in profile.worker_routes}
    if len(route_by_id) != len(profile.worker_routes):
        raise VastDeploymentError("profile worker routes contain duplicate worker IDs")
    seen_workers: set[str] = set()
    credentials: set[str] = {"ARC_ENV_TOKEN"}
    service_credentials: set[str] = set()
    gateway_credentials: dict[tuple[str, str, int], str] = {}
    for index, raw_worker in enumerate(raw_workers):
        worker = _mapping(raw_worker, f"workers[{index}]")
        worker_id = worker.get("id")
        if (not isinstance(worker_id, str) or not worker_id
                or worker_id in seen_workers or worker_id not in route_by_id):
            raise VastDeploymentError("runtime Workers and profile.worker_routes must have the same unique worker IDs")
        seen_workers.add(worker_id)
        if worker.get("scope") not in {"service", "gateway"}:
            raise VastDeploymentError("Vast Learner runtime Workers must use scope=service or scope=gateway")
        if worker.get("host") is not None:
            raise VastDeploymentError(f"runtime Worker {worker_id} must not configure a local host")
        endpoint = worker.get("endpoint")
        if not isinstance(endpoint, str):
            raise VastDeploymentError(f"runtime Worker {worker_id} endpoint is required")
        normalized_endpoint, remote_port = loopback_http_url(
            endpoint, f"runtime workers[{index}].endpoint")
        route = route_by_id[worker_id]
        if (endpoint != normalized_endpoint
                or normalized_endpoint != f"http://127.0.0.1:{route.remote_port}"
                or remote_port != route.remote_port
                or worker.get("port") != route.remote_port
                or isinstance(worker.get("port"), bool)):
            raise VastDeploymentError(
                f"runtime Worker {worker_id} endpoint must already match its declared SSH remote_port"
            )
        prefix = worker.get("url_prefix", "")
        if worker["scope"] == "service" and prefix != "":
            raise VastDeploymentError(f"service Worker {worker_id} cannot use a Gateway route prefix")
        if worker["scope"] == "gateway" and prefix != f"/v1/workers/{worker_id}":
            raise VastDeploymentError(f"gateway Worker {worker_id} route prefix must match its exact ID")
        auth_name = worker.get("auth_token_env")
        if not isinstance(auth_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", auth_name):
            raise VastDeploymentError(f"runtime Worker {worker_id} must declare auth_token_env explicitly")
        if auth_name == "ARC_ENV_TOKEN":
            raise VastDeploymentError(f"runtime Worker {worker_id} must not reuse ARC_ENV_TOKEN")
        if auth_name in {"PATH", "HOME", "PYTHONPATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES",
                         "LD_PRELOAD", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "ARC_RL_ROOT"}:
            raise VastDeploymentError(f"runtime Worker {worker_id} uses a reserved credential environment name")
        if worker["scope"] == "service":
            if auth_name in service_credentials:
                raise VastDeploymentError("each service Worker must use a distinct auth_token_env")
            service_credentials.add(auth_name)
        else:
            # A Gateway may share one credential only within the same exact endpoint/prefix.
            gateway_key = (normalized_endpoint, str(prefix), route.remote_port)
            previous = gateway_credentials.get(gateway_key)
            if previous is not None and previous != auth_name:
                raise VastDeploymentError("Workers routed through one Gateway must share its auth_token_env")
            gateway_credentials[gateway_key] = auth_name
        credentials.add(auth_name)
    if seen_workers != set(route_by_id):
        raise VastDeploymentError("runtime Worker IDs differ from profile.worker_routes")
    gateway_credentials_used = set(gateway_credentials.values())
    if service_credentials & gateway_credentials_used:
        raise VastDeploymentError("service Worker and Gateway credentials must use distinct environment names")

    bepa = runtime.get("bepa", {})
    if not isinstance(bepa, dict):
        raise VastDeploymentError("runtime config bepa must be an object")
    if bepa.get("enabled", False):
        teacher = _mapping(bepa.get("teacher"), "bepa.teacher")
        teacher_names: set[str] = set()
        for field in ("api_key_env", "token_env"):
            name = teacher.get(field)
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise VastDeploymentError(f"runtime config bepa.teacher.{field} must be an explicit environment name")
            if name in {"ARC_ENV_TOKEN", "PATH", "HOME", "PYTHONPATH", "CUDA_HOME",
                        "CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "LD_LIBRARY_PATH",
                        "NVIDIA_VISIBLE_DEVICES", "ARC_RL_ROOT"}:
                raise VastDeploymentError(f"runtime config bepa.teacher.{field} uses a reserved credential name")
            teacher_names.add(name)
        if len(teacher_names) != 2 or teacher_names & credentials:
            raise VastDeploymentError("BEPA, ARC and external Worker credentials must have distinct names")
        credentials.update(teacher_names)
    return credentials


def _validate_runtime_document(raw: bytes, profile: VastLearnerProfile,
                               manifest: Any) -> tuple[dict[str, Any], str, list[str]]:
    runtime = _parse_runtime_document(raw)
    _check_runtime_has_no_inline_credentials(runtime)
    allowed_top = {
        "model_path", "dataset_dir", "run_dir", "worker_inference", "node_scope", "learner",
        "workers", "rollout", "dapo", "quantization", "lora", "arc", "bepa", "evaluation",
    }
    required_top = {"model_path", "dataset_dir", "run_dir", "worker_inference", "workers", "learner"}
    if required_top - set(runtime) or set(runtime) - allowed_top:
        missing = sorted(required_top - set(runtime))
        extra = sorted(set(runtime) - allowed_top)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        raise VastDeploymentError("runtime config top-level structure is invalid: " + "; ".join(details))

    asset_root = PurePosixPath(profile.asset_root_container)
    state_root = PurePosixPath(profile.state_root_container)
    manifest_paths = {item.path for item in manifest.files}
    model_path = _runtime_container_path(runtime.get("model_path"), "model_path")
    dataset_path = _runtime_container_path(runtime.get("dataset_dir"), "dataset_dir")
    _require_asset_reference(model_path, asset_root=asset_root, manifest_paths=manifest_paths,
                             label="model_path", directory=True)
    _require_asset_reference(dataset_path, asset_root=asset_root, manifest_paths=manifest_paths,
                             label="dataset_dir", directory=True)
    inference = _mapping(runtime.get("worker_inference"), "worker_inference")
    if _runtime_container_path(inference.get("model_path"), "worker_inference.model_path") != model_path:
        raise VastDeploymentError("runtime model_path and worker_inference.model_path must identify the same asset")

    learner = _mapping(runtime.get("learner"), "learner")
    if (learner.get("host") != "127.0.0.1" or learner.get("port") != profile.learner_port
            or isinstance(learner.get("port"), bool)):
        raise VastDeploymentError("runtime learner.host/port must match the declared loopback listener")
    expected_gpus = list(range(profile.gpu_count))
    if learner.get("gpus") != expected_gpus:
        raise VastDeploymentError("runtime learner.gpus must be exactly 0..gpu_count-1")

    expected_credentials = _check_worker_routes(runtime, profile)
    profile_credentials = set(profile.credential_envs)
    if profile_credentials != expected_credentials:
        raise VastDeploymentError("profile.credential_envs must exactly match RuntimeConfig credential environment names")

    run_path = _runtime_container_path(runtime.get("run_dir"), "run_dir")
    state_paths = [("run_dir", run_path)]
    recovery = learner.get("recovery_dir")
    if recovery not in (None, ""):
        state_paths.append(("learner.recovery_dir", _runtime_container_path(recovery, "learner.recovery_dir")))
    bepa = runtime.get("bepa", {})
    if isinstance(bepa, dict) and bepa.get("enabled", False):
        value = bepa.get("state_dir")
        state_paths.append(("bepa.state_dir", _runtime_container_path(value, "bepa.state_dir")))
    evaluation = runtime.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise VastDeploymentError("runtime config evaluation must be an object")
    if evaluation.get("enabled", False):
        value = evaluation.get("state_dir")
        state_paths.append(("evaluation.state_dir", _runtime_container_path(value, "evaluation.state_dir")))
    for label, path in state_paths:
        if not _path_within(path, state_root, strict=True):
            raise VastDeploymentError(f"runtime {label} must be a private descendant of state_root_container")

    baked_resources: list[str] = []
    arc = runtime.get("arc", {})
    if not isinstance(arc, dict):
        raise VastDeploymentError("runtime config arc must be an object")
    contract = _runtime_container_path(arc.get("contract_path", "/work/configs/train-contract.json"),
                                       "arc.contract_path")
    _require_resource_reference(contract, asset_root=asset_root, state_root=state_root,
                                 remote_root=PurePosixPath(profile.remote_root),
                                 manifest_paths=manifest_paths,
                                 label="arc.contract_path", baked_resources=baked_resources)
    if evaluation.get("enabled", False):
        for key in ("manifest_path", "sft_exclusion_proof", "initial_count_proof"):
            value = _runtime_container_path(evaluation.get(key), f"evaluation.{key}")
            _require_resource_reference(value, asset_root=asset_root, state_root=state_root,
                                        remote_root=PurePosixPath(profile.remote_root),
                                        manifest_paths=manifest_paths,
                                        label=f"evaluation.{key}", baked_resources=baked_resources)

    validation_mode = "static_deferred"
    try:
        from qwen_rl.runtime.config import load_config
    except ImportError:
        load_config = None
    if load_config is not None:
        try:
            load_config(profile.runtime_config_path)
        except ImportError:
            validation_mode = "static_deferred"
        except Exception as exc:
            raise VastDeploymentError(
                f"RuntimeConfig rejected the preserved source ({type(exc).__name__})"
            ) from exc
        else:
            validation_mode = "full_cpu"
    return runtime, validation_mode, sorted(set(baked_resources))


def _learner_storage(profile: VastLearnerProfile, *, asset_bytes: int,
                     largest_file_bytes: int) -> dict[str, Any]:
    values = _storage_receipt_values(
        disk_gb=profile.disk_gb,
        state_reserve_gb=profile.state_reserve_gb,
        image_layer_reserve_gb=profile.image_layer_reserve_gb,
        model_size_bytes=asset_bytes,
        largest_file_bytes=largest_file_bytes,
    )
    return {
        "disk_gb": values["disk_gb"],
        "disk_bytes": values["disk_bytes"],
        "asset_bytes": values["model_bytes"],
        "largest_file_staging_bytes": values["largest_file_staging_bytes"],
        "state_reserve_gb": values["state_reserve_gb"],
        "image_layer_reserve_gb": values["image_layer_reserve_gb"],
        "required_bytes_exclusive": values["required_bytes_exclusive"],
        "remaining_bytes_after_declared_reserves": values["remaining_bytes_after_declared_reserves"],
        "image_layer_reserve_verified": False,
    }


def _check_learner_input(profile: VastLearnerProfile) -> tuple[bytes, bytes, Any, dict[str, Any], str, list[str], dict[str, Any]]:
    runtime_bytes = read_regular_file(profile.runtime_config_path, label="Learner runtime config")
    manifest = load_model_manifest(profile.asset_manifest_path, profile.asset_manifest_sha256)
    runtime, validation_mode, baked_resources = _validate_runtime_document(runtime_bytes, profile, manifest)
    storage = _learner_storage(
        profile, asset_bytes=manifest.size_bytes, largest_file_bytes=manifest.largest_file_bytes)
    return runtime_bytes, manifest.raw_bytes, manifest, runtime, validation_mode, baked_resources, storage


def check_vast_learner_profile(path: str | Path) -> dict[str, Any]:
    """Validate a complete Learner profile and assets without network/GPU/Vast calls."""
    profile = load_vast_learner_profile(path)
    runtime_bytes, _manifest_bytes, manifest, runtime, validation_mode, baked, storage = _check_learner_input(profile)
    return {
        "status": "cpu_preflight_passed",
        "role": "learner",
        "deployment_id": profile.deployment_id,
        "image_ref": profile.image_ref,
        "runtime_config_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
        "asset_manifest_sha256": manifest.sha256,
        "asset_files": len(manifest.files),
        "asset_bytes": manifest.size_bytes,
        "worker_ids": sorted(worker["id"] for worker in runtime["workers"]),
        "credential_envs": sorted(profile.credential_envs),
        "runtime_config_validation": validation_mode,
        "baked_resources_pending_entrypoint_check": baked,
        "storage": storage,
        "network_mode": "ssh_tunnel",
        "learner_url": profile.learner_url,
        "local_learner_url": profile.local_learner_url,
        "executed_vast_api": False,
        "runtime_not_validated": True,
    }


def _start_learner_script(profile: VastLearnerProfile) -> bytes:
    import shlex

    remote_config = profile.remote_root + "/config"
    expected = {
        "deployment": remote_config + "/deployment.json",
        "runtime": remote_config + "/runtime.json",
        "manifest": remote_config + "/asset-manifest.json",
        "asset_root": profile.asset_root_container,
        "state_root": profile.state_root_container,
        "credential_map": profile.remote_root + "/secrets/credential-map.json",
        "receipt": profile.state_root_container + "/vast-learner-entrypoint.json",
    }
    assignments = "\n".join(
        f"EXPECTED_{name.upper()}={shlex.quote(value)}" for name, value in expected.items())
    arguments = {
        "deployment": "deployment",
        "runtime": "runtime_config",
        "manifest": "asset_manifest",
        "asset_root": "asset_root",
        "state_root": "state_root",
        "credential_map": "credential_map",
        "receipt": "receipt",
    }
    checks = "\n".join(
        f'[[ "${{{variable}}}" == "$EXPECTED_{name.upper()}" ]] || fail "{name} path differs from deployment.json"'
        for name, variable in arguments.items())
    script = f'''#!/usr/bin/env bash
set -Eeuo pipefail

fail() {{ echo "qwen-vast-learner: $1" >&2; exit "${{2:-64}}"; }}
ENTRYPOINT=/opt/qwen-vast/learner-entrypoint.sh
PYTHON=/work/envs/rl-vllm/bin/python
SSHD=/usr/sbin/sshd
{assignments}

if [[ $# -eq 0 || "${{1:-}}" == "--wait-for-attach" ]]; then
  [[ -x "$ENTRYPOINT" && -x "$PYTHON" && -x "$SSHD" ]] || fail "prebuilt Learner runtime or SSH server is unavailable" 70
  echo "qwen-vast-learner: image preflight passed; waiting for the SSH tunnel and private Learner bundle."
  exec sleep infinity
fi

deployment=""; runtime_config=""; asset_manifest=""; asset_root=""; state_root=""; credential_map=""; receipt=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deployment) [[ $# -ge 2 ]] || fail "missing deployment path"; deployment="$2"; shift 2 ;;
    --runtime-config) [[ $# -ge 2 ]] || fail "missing runtime config path"; runtime_config="$2"; shift 2 ;;
    --asset-manifest) [[ $# -ge 2 ]] || fail "missing asset manifest path"; asset_manifest="$2"; shift 2 ;;
    --asset-root) [[ $# -ge 2 ]] || fail "missing asset root"; asset_root="$2"; shift 2 ;;
    --state-root) [[ $# -ge 2 ]] || fail "missing state root"; state_root="$2"; shift 2 ;;
    --credential-map) [[ $# -ge 2 ]] || fail "missing credential map"; credential_map="$2"; shift 2 ;;
    --receipt) [[ $# -ge 2 ]] || fail "missing receipt"; receipt="$2"; shift 2 ;;
    *) fail "unsupported Learner startup argument" ;;
  esac
done
{checks}
[[ -x "$ENTRYPOINT" && -x "$PYTHON" && -x "$SSHD" ]] || fail "prebuilt Learner runtime or SSH server is unavailable" 70
for file in "$deployment" "$runtime_config" "$asset_manifest" "$credential_map"; do
  [[ -f "$file" && ! -L "$file" ]] || fail "private Learner input is missing or unsafe" 66
  mode="$(stat -c '%a' "$file")"
  [[ "$mode" == "600" ]] || fail "private Learner input must have mode 0600" 77
done
if [[ -L "$asset_root" ]]; then fail "immutable asset root must not be a symlink" 77; fi
if [[ -e "$asset_root" && ! -d "$asset_root" ]]; then fail "immutable asset root exists with an unsafe type" 66; fi
[[ ! -L "$state_root" ]] || fail "private state root must not be a symlink" 77
mkdir -p -- "$state_root"
chmod 700 -- "$state_root"
[[ "$(stat -c '%u:%a' "$state_root")" == "$(id -u):700" ]] || fail "state root must be owned by the SSH runtime UID with mode 0700" 77
exec "$ENTRYPOINT" \\
  --deployment "$deployment" \\
  --runtime-config "$runtime_config" \\
  --asset-manifest "$asset_manifest" \\
  --asset-root "$asset_root" \\
  --state-root "$state_root" \\
  --credential-map "$credential_map" \\
  --receipt "$receipt"
'''
    return script.encode("utf-8")


def _learner_recipe(profile: VastLearnerProfile, script_path: Path,
                    storage: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "role": "learner",
        "provider": "vast.ai",
        "action": "create_instance",
        "status": "awaiting_operator_offer_id",
        "offer_id": None,
        "required_operator_inputs": ["offer_id from a separately reviewed Vast offer"],
        "parameters": {
            "image": profile.image_ref,
            "disk_gb": profile.disk_gb,
            "ssh": True,
            "onstart": str(script_path),
            "public_learner_api_ports": [],
            "network_mode": "ssh_tunnel",
        },
        "learner_network": {
            "remote_learner_url": profile.learner_url,
            "local_learner_url": profile.local_learner_url,
            "public_learner_api": False,
            "worker_routes": [route.model_dump() for route in profile.worker_routes],
        },
        "storage_calculation": storage,
        "cost_timing": {
            "image_layers_may_be_pulled_after_instance_creation": True,
            "asset_download_occurs_after_instance_creation": True,
            "prewarmed_vast_host_cache_is_not_assumed": True,
        },
        "command_template": [
            "vastai", "create", "instance", "<offer-id>",
            "--image", profile.image_ref,
            "--disk", str(profile.disk_gb),
            "--ssh",
            "--onstart", str(script_path),
        ],
        "executed": False,
    }


def _write_learner_tree(staging: Path, profile: VastLearnerProfile,
                        runtime_bytes: bytes, manifest_bytes: bytes, manifest: Any,
                        runtime_validation: str, baked_resources: list[str],
                        storage: dict[str, Any]) -> None:
    (staging / "state").mkdir(mode=_PRIVATE_DIR_MODE)
    os.chmod(staging / "state", _PRIVATE_DIR_MODE)
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    script = _start_learner_script(profile)
    script_sha = hashlib.sha256(script).hexdigest()
    _write_private(staging / "runtime.json", runtime_bytes)
    _write_private(staging / "asset-manifest.json", manifest_bytes)
    _write_private(staging / "deployment.json", _json_bytes(profile.deployment_payload(
        runtime_config_sha256=runtime_sha,
        asset_manifest_sha256=manifest.sha256,
        start_learner_script_sha256=script_sha,
        runtime_config_validation=runtime_validation,
    )))
    _write_private(staging / "start-learner.sh", script)
    recipe = _learner_recipe(profile, profile.output_dir / "start-learner.sh", storage)
    _write_private(staging / "vast-create-recipe.json", _json_bytes(recipe))
    receipt = {
        "schema_version": 2,
        "role": "learner",
        "deployment_id": profile.deployment_id,
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_config_sha256": runtime_sha,
        "asset_manifest_sha256": manifest.sha256,
        "asset_files": len(manifest.files),
        "asset_bytes": manifest.size_bytes,
        "credential_envs": sorted(profile.credential_envs),
        "worker_routes": [route.model_dump() for route in profile.worker_routes],
        "runtime_config_validation": runtime_validation,
        "baked_resources_pending_entrypoint_check": baked_resources,
        "storage": storage,
        "network_mode": "ssh_tunnel",
        "runtime_contract": "declared_not_validated",
        "runtime_not_validated": True,
        "vast_api_called": False,
        "blocking_prerequisites": [
            "select and review a Vast offer ID before any create action",
            "entrypoint must verify image resources, runtime, credentials and visible GPU before supervisor startup",
        ],
    }
    _write_private(staging / "state" / "receipt.json", _json_bytes(receipt))
    for path in staging.rglob("*"):
        os.chmod(path, _PRIVATE_DIR_MODE if path.is_dir() else _PRIVATE_MODE)
    os.chmod(staging, _PRIVATE_DIR_MODE)


def prepare_vast_learner_deployment(path: str | Path) -> dict[str, Any]:
    """Create a private CPU-prepared Learner bundle; no network, GPU, or Vast API calls."""
    profile = load_vast_learner_profile(path)
    runtime_bytes, manifest_bytes, manifest, _runtime, mode, baked, storage = _check_learner_input(profile)
    _ensure_private_output_root(profile.output_root)
    destination = profile.output_dir
    if destination.exists() or destination.is_symlink():
        raise VastDeploymentError("deployment output already exists; prepare never overwrites an existing package")
    staging = Path(tempfile.mkdtemp(prefix=f".{profile.deployment_id}.", dir=profile.output_root))
    os.chmod(staging, _PRIVATE_DIR_MODE)
    destination_created = False
    try:
        _write_learner_tree(staging, profile, runtime_bytes, manifest_bytes, manifest, mode, baked, storage)
        try:
            os.mkdir(destination, _PRIVATE_DIR_MODE)
        except FileExistsError as exc:
            raise VastDeploymentError(
                "deployment output appeared during prepare; refusing to overwrite it"
            ) from exc
        destination_created = True
        for member in staging.iterdir():
            os.rename(member, destination / member.name)
        staging.rmdir()
        directory_fd = os.open(profile.output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if destination_created:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    checked = check_prepared_deployment(destination, expected_role="learner")
    return {
        **checked,
        "output_dir": str(destination),
        "recipe_path": str(destination / "vast-create-recipe.json"),
        "receipt_path": str(destination / "state" / "receipt.json"),
    }


def _write_tree(staging: Path, profile: VastProfile, worker_bytes: bytes,
                manifest_bytes: bytes, manifest: Any, storage: dict[str, Any]) -> None:
    (staging / "state").mkdir(mode=_PRIVATE_DIR_MODE)
    os.chmod(staging / "state", _PRIVATE_DIR_MODE)
    worker_sha = hashlib.sha256(worker_bytes).hexdigest()
    script = _start_script(profile)
    script_sha = hashlib.sha256(script).hexdigest()
    _write_private(staging / "worker.json", worker_bytes)
    _write_private(staging / "model-manifest.json", manifest_bytes)
    _write_private(staging / "deployment.json", _json_bytes(
        profile.deployment_payload(worker_json_sha256=worker_sha,
                                   start_worker_script_sha256=script_sha)))
    _write_private(staging / "start-worker.sh", script)
    recipe = _create_recipe(profile, profile.output_dir / "start-worker.sh", storage)
    _write_private(staging / "vast-create-recipe.json", _json_bytes(recipe))
    receipt = {
        "schema_version": 2,
        "role": "worker",
        "deployment_id": profile.deployment_id,
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "worker_json_sha256": worker_sha,
        "start_worker_script_sha256": script_sha,
        "model_manifest_sha256": manifest.sha256,
        "model_files": len(manifest.files),
        "model_bytes": manifest.size_bytes,
        "storage": storage,
        "network_mode": "ssh_tunnel",
        "runtime_contract": "declared_not_validated",
        "runtime_not_validated": True,
        "vast_api_called": False,
        "blocking_prerequisites": ["select and review a Vast offer ID before any create action"],
    }
    _write_private(staging / "state" / "receipt.json", _json_bytes(receipt))
    for path in staging.rglob("*"):
        if path.is_dir():
            os.chmod(path, _PRIVATE_DIR_MODE)
        else:
            os.chmod(path, _PRIVATE_MODE)
    os.chmod(staging, _PRIVATE_DIR_MODE)


def prepare_vast_worker_deployment(path: str | Path) -> dict[str, Any]:
    """Create a new private deployment package without network/GPU/Vast API calls."""
    profile = load_vast_profile(path)
    worker_bytes, manifest_bytes, manifest, _worker, storage = _check_input(profile)
    _ensure_private_output_root(profile.output_root)
    destination = profile.output_dir
    if destination.exists() or destination.is_symlink():
        raise VastDeploymentError("deployment output already exists; prepare never overwrites an existing package")
    staging = Path(tempfile.mkdtemp(prefix=f".{profile.deployment_id}.", dir=profile.output_root))
    os.chmod(staging, _PRIVATE_DIR_MODE)
    destination_created = False
    try:
        _write_tree(staging, profile, worker_bytes, manifest_bytes, manifest, storage)
        try:
            os.mkdir(destination, _PRIVATE_DIR_MODE)
        except FileExistsError as exc:
            raise VastDeploymentError(
                "deployment output appeared during prepare; refusing to overwrite it"
            ) from exc
        destination_created = True
        for member in staging.iterdir():
            os.rename(member, destination / member.name)
        staging.rmdir()
        directory_fd = os.open(profile.output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if destination_created:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    checked = check_prepared_deployment(destination, expected_role="worker")
    return {
        **checked,
        "output_dir": str(destination),
        "recipe_path": str(destination / "vast-create-recipe.json"),
        "receipt_path": str(destination / "state" / "receipt.json"),
    }


def _check_prepared_worker_deployment(path: str | Path) -> dict[str, Any]:
    """Verify the private prepared package and its exact source hashes."""
    directory = Path(path).expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise VastDeploymentError("prepared deployment must be an existing private directory")
    if stat.S_IMODE(directory.stat().st_mode) != _PRIVATE_DIR_MODE:
        raise VastDeploymentError("prepared deployment directory must have mode 0700")
    required = {
        "deployment.json", "worker.json", "model-manifest.json",
        "start-worker.sh", "vast-create-recipe.json", "state/receipt.json",
    }
    entries = tuple(directory.rglob("*"))
    observed = {str(item.relative_to(directory)) for item in entries if item.is_file() or item.is_symlink()}
    observed_dirs = {str(item.relative_to(directory)) for item in entries
                     if item.is_dir() and not item.is_symlink()}
    attach_receipts = {item for item in observed if _ATTACH_RECEIPT.fullmatch(item)}
    if observed != required | attach_receipts or observed_dirs != {"state"}:
        raise VastDeploymentError("prepared deployment files do not match the fixed package layout")
    for relative in sorted(required | attach_receipts):
        item = directory / relative
        if item.is_symlink() or not item.is_file():
            raise VastDeploymentError(f"prepared deployment member is unsafe: {relative}")
        if stat.S_IMODE(item.stat().st_mode) != _PRIVATE_MODE:
            raise VastDeploymentError(f"prepared deployment member must have mode 0600: {relative}")
    state_dir = directory / "state"
    if state_dir.is_symlink() or not state_dir.is_dir() or stat.S_IMODE(state_dir.stat().st_mode) != _PRIVATE_DIR_MODE:
        raise VastDeploymentError("prepared state directory must be private with mode 0700")

    deployment_bytes = read_regular_file(directory / "deployment.json", label="deployment.json")
    payload = validate_deployment_payload(parse_json_bytes(deployment_bytes, label="deployment.json"),
                                          expected_role="worker")
    worker_bytes = read_regular_file(directory / "worker.json", label="prepared worker.json")
    if hashlib.sha256(worker_bytes).hexdigest() != payload.get("worker_json_sha256"):
        raise VastDeploymentError("prepared worker.json SHA-256 differs from deployment.json")
    script_bytes = read_regular_file(directory / "start-worker.sh", label="prepared start-worker.sh")
    if hashlib.sha256(script_bytes).hexdigest() != payload.get("start_worker_script_sha256"):
        raise VastDeploymentError("prepared start-worker.sh SHA-256 differs from deployment.json")
    manifest = load_model_manifest(directory / "model-manifest.json", payload.get("model_manifest_sha256"))
    required_image = digest_pinned_image(payload.get("image_ref"))
    _validate_worker_config(
        worker_bytes,
        worker_port=payload["worker_port"],
        remote_policy_port=payload["remote_policy_port"],
        model_root_container=payload["model_root_container"],
        manifest_paths={item.path for item in manifest.files},
    )
    storage = _storage_receipt_values(
        disk_gb=payload["disk_gb"],
        state_reserve_gb=payload["state_reserve_gb"],
        image_layer_reserve_gb=payload["image_layer_reserve_gb"],
        model_size_bytes=manifest.size_bytes,
        largest_file_bytes=manifest.largest_file_bytes,
    )
    receipt_bytes = read_regular_file(directory / "state" / "receipt.json", label="state/receipt.json")
    receipt = parse_json_bytes(receipt_bytes, label="state/receipt.json")
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != 2
            or receipt.get("role") != "worker" or receipt.get("status") != "prepared"
            or receipt.get("runtime_not_validated") is not True
            or receipt.get("vast_api_called") is not False
            or receipt.get("worker_json_sha256") != payload.get("worker_json_sha256")
            or receipt.get("start_worker_script_sha256") != payload.get("start_worker_script_sha256")
            or receipt.get("model_manifest_sha256") != manifest.sha256
            or receipt.get("storage") != storage):
        raise VastDeploymentError("state receipt differs from the immutable prepared package")
    recipe_bytes = read_regular_file(directory / "vast-create-recipe.json", label="vast-create-recipe.json")
    recipe = parse_json_bytes(recipe_bytes, label="vast-create-recipe.json")
    parameters = recipe.get("parameters") if isinstance(recipe, dict) else None
    if (not isinstance(recipe, dict) or recipe.get("executed") is not False
            or recipe.get("offer_id") is not None or recipe.get("action") != "create_instance"
            or recipe.get("storage_calculation") != storage
            or not isinstance(parameters, dict)
            or parameters.get("image") != required_image
            or parameters.get("disk_gb") != payload["disk_gb"]
            or parameters.get("onstart") != str(directory / "start-worker.sh")
            or parameters.get("network_mode") != "ssh_tunnel"
            or parameters.get("public_worker_api_ports") != []):
        raise VastDeploymentError("create recipe must be a review-only template with no offer selected")
    for relative in sorted(attach_receipts):
        attached = parse_json_bytes(
            read_regular_file(directory / relative, label=relative), label=relative)
        package_hashes = {
            "deployment.json": hashlib.sha256(deployment_bytes).hexdigest(),
            "worker.json": payload["worker_json_sha256"],
            "model-manifest.json": manifest.sha256,
            "start-worker.sh": payload["start_worker_script_sha256"],
        }
        if (not isinstance(attached, dict)
                or attached.get("schema") != "qwen-rl-vast-remote-attach/v2"
                or attached.get("role") != "worker"
                or attached.get("deployment_id") != payload.get("deployment_id")
                or attached.get("network_mode") != "ssh_tunnel"
                or attached.get("package_sha256s") != package_hashes
                or attached.get("status") not in _ATTACH_STATUSES):
            raise VastDeploymentError(f"{relative} does not match the prepared deployment")
    return {
        "status": "prepared",
        "role": "worker",
        "deployment_id": payload.get("deployment_id"),
        "image_ref": required_image,
        "worker_json_sha256": payload["worker_json_sha256"],
        "model_manifest_sha256": manifest.sha256,
        "model_files": len(manifest.files),
        "model_bytes": manifest.size_bytes,
        "network_mode": payload.get("network_mode"),
        "worker_url": payload.get("worker_url"),
        "runtime_not_validated": True,
        "attach_receipts": len(attach_receipts),
        "executed_vast_api": False,
    }


def _check_prepared_learner_deployment(path: str | Path) -> dict[str, Any]:
    directory = Path(path).expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise VastDeploymentError("prepared Learner deployment must be an existing private directory")
    if stat.S_IMODE(directory.stat().st_mode) != _PRIVATE_DIR_MODE:
        raise VastDeploymentError("prepared Learner deployment directory must have mode 0700")
    required = {
        "deployment.json", "runtime.json", "asset-manifest.json",
        "start-learner.sh", "vast-create-recipe.json", "state/receipt.json",
    }
    entries = tuple(directory.rglob("*"))
    observed = {str(item.relative_to(directory)) for item in entries
                if item.is_file() or item.is_symlink()}
    observed_dirs = {str(item.relative_to(directory)) for item in entries
                     if item.is_dir() and not item.is_symlink()}
    attach_receipts = {item for item in observed if _ATTACH_RECEIPT.fullmatch(item)}
    if observed != required | attach_receipts or observed_dirs != {"state"}:
        raise VastDeploymentError("prepared Learner files do not match the fixed package layout")
    for relative in sorted(required | attach_receipts):
        item = directory / relative
        if item.is_symlink() or not item.is_file():
            raise VastDeploymentError(f"prepared Learner member is unsafe: {relative}")
        if stat.S_IMODE(item.stat().st_mode) != _PRIVATE_MODE:
            raise VastDeploymentError(f"prepared Learner member must have mode 0600: {relative}")
    state_dir = directory / "state"
    if state_dir.is_symlink() or not state_dir.is_dir() or stat.S_IMODE(state_dir.stat().st_mode) != _PRIVATE_DIR_MODE:
        raise VastDeploymentError("prepared Learner state directory must be private with mode 0700")

    deployment_bytes = read_regular_file(directory / "deployment.json", label="deployment.json")
    payload = validate_deployment_payload(
        parse_json_bytes(deployment_bytes, label="deployment.json"), expected_role="learner")
    runtime_bytes = read_regular_file(directory / "runtime.json", label="prepared runtime.json")
    if hashlib.sha256(runtime_bytes).hexdigest() != payload["runtime_config_sha256"]:
        raise VastDeploymentError("prepared runtime.json SHA-256 differs from deployment.json")
    manifest = load_model_manifest(directory / "asset-manifest.json", payload["asset_manifest_sha256"])
    profile = VastLearnerProfile(
        role="learner",
        deployment_id=payload["deployment_id"],
        output_root=directory.parent,
        remote_root=payload["remote_root"],
        image_ref=payload["image_ref"],
        platform=payload["platform"],
        cuda_arch=payload["cuda_arch"],
        runtime_contract=payload["runtime_contract"],
        runtime_config_path=directory / "runtime.json",
        asset_manifest_path=directory / "asset-manifest.json",
        asset_manifest_sha256=payload["asset_manifest_sha256"],
        asset_root_container=payload["asset_root_container"],
        state_root_container=payload["state_root_container"],
        learner_port=payload["learner_port"],
        local_forward_port=payload["local_forward_port"],
        worker_routes=tuple(VastWorkerRoute(**item) for item in payload["worker_routes"]),
        credential_envs=tuple(payload["credential_envs"]),
        gpu_name=payload["gpu_name"],
        gpu_count=payload["gpu_count"],
        disk_gb=payload["disk_gb"],
        state_reserve_gb=payload["state_reserve_gb"],
        image_layer_reserve_gb=payload["image_layer_reserve_gb"],
        profile_path=directory / "deployment.json",
    )
    _runtime, validation_mode, baked_resources = _validate_runtime_document(runtime_bytes, profile, manifest)
    if validation_mode != payload["runtime_config_validation"]:
        raise VastDeploymentError("prepared runtime validation mode differs from deployment.json")
    start_script = read_regular_file(directory / "start-learner.sh", label="prepared start-learner.sh")
    if hashlib.sha256(start_script).hexdigest() != payload["start_learner_script_sha256"]:
        raise VastDeploymentError("prepared start-learner.sh SHA-256 differs from deployment.json")
    storage = _learner_storage(
        profile, asset_bytes=manifest.size_bytes, largest_file_bytes=manifest.largest_file_bytes)
    receipt = parse_json_bytes(
        read_regular_file(directory / "state" / "receipt.json", label="state/receipt.json"),
        label="state/receipt.json")
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != 2
            or receipt.get("role") != "learner" or receipt.get("status") != "prepared"
            or receipt.get("runtime_not_validated") is not True
            or receipt.get("vast_api_called") is not False
            or receipt.get("runtime_config_sha256") != payload["runtime_config_sha256"]
            or receipt.get("asset_manifest_sha256") != manifest.sha256
            or receipt.get("asset_files") != len(manifest.files)
            or receipt.get("asset_bytes") != manifest.size_bytes
            or receipt.get("credential_envs") != sorted(profile.credential_envs)
            or receipt.get("worker_routes") != [route.model_dump() for route in profile.worker_routes]
            or receipt.get("runtime_config_validation") != validation_mode
            or receipt.get("baked_resources_pending_entrypoint_check") != baked_resources
            or receipt.get("storage") != storage):
        raise VastDeploymentError("Learner state receipt differs from the immutable prepared package")
    recipe = parse_json_bytes(
        read_regular_file(directory / "vast-create-recipe.json", label="vast-create-recipe.json"),
        label="vast-create-recipe.json")
    parameters = recipe.get("parameters") if isinstance(recipe, dict) else None
    if (not isinstance(recipe, dict) or recipe.get("schema_version") != 2
            or recipe.get("role") != "learner" or recipe.get("executed") is not False
            or recipe.get("offer_id") is not None or recipe.get("action") != "create_instance"
            or recipe.get("storage_calculation") != storage
            or not isinstance(parameters, dict)
            or parameters.get("image") != payload["image_ref"]
            or parameters.get("disk_gb") != payload["disk_gb"]
            or parameters.get("onstart") != str(directory / "start-learner.sh")
            or parameters.get("network_mode") != "ssh_tunnel"
            or parameters.get("public_learner_api_ports") != []):
        raise VastDeploymentError("Learner create recipe must be review-only with no offer selected")
    package_hashes = {
        "deployment.json": hashlib.sha256(deployment_bytes).hexdigest(),
        "runtime.json": payload["runtime_config_sha256"],
        "asset-manifest.json": manifest.sha256,
        "start-learner.sh": payload["start_learner_script_sha256"],
    }
    for relative in sorted(attach_receipts):
        attached = parse_json_bytes(read_regular_file(directory / relative, label=relative), label=relative)
        if (not isinstance(attached, dict)
                or attached.get("schema") != "qwen-rl-vast-remote-attach/v2"
                or attached.get("role") != "learner"
                or attached.get("deployment_id") != payload["deployment_id"]
                or attached.get("network_mode") != "ssh_tunnel"
                or attached.get("package_sha256s") != package_hashes
                or attached.get("status") not in _ATTACH_STATUSES):
            raise VastDeploymentError(f"{relative} does not match the prepared Learner deployment")
    return {
        "status": "prepared",
        "role": "learner",
        "deployment_id": payload["deployment_id"],
        "image_ref": payload["image_ref"],
        "runtime_config_sha256": payload["runtime_config_sha256"],
        "asset_manifest_sha256": manifest.sha256,
        "asset_files": len(manifest.files),
        "asset_bytes": manifest.size_bytes,
        "worker_routes": payload["worker_routes"],
        "credential_envs": payload["credential_envs"],
        "runtime_config_validation": validation_mode,
        "baked_resources_pending_entrypoint_check": baked_resources,
        "storage": storage,
        "network_mode": payload["network_mode"],
        "learner_url": payload["learner_url"],
        "local_learner_url": payload["local_learner_url"],
        "runtime_not_validated": True,
        "attach_receipts": len(attach_receipts),
        "executed_vast_api": False,
    }


def check_vast_profile(path: str | Path, *, expected_role: str) -> dict[str, Any]:
    if expected_role == "worker":
        return check_vast_worker_profile(path)
    if expected_role == "learner":
        return check_vast_learner_profile(path)
    raise VastDeploymentError("profile check requires an explicit supported role")


def prepare_vast_deployment(path: str | Path, *, expected_role: str) -> dict[str, Any]:
    if expected_role == "worker":
        return prepare_vast_worker_deployment(path)
    if expected_role == "learner":
        return prepare_vast_learner_deployment(path)
    raise VastDeploymentError("prepare requires an explicit supported role")


def check_prepared_deployment(path: str | Path, *, expected_role: str) -> dict[str, Any]:
    if expected_role not in {"worker", "learner"}:
        raise VastDeploymentError("prepared package check requires an explicit supported role")
    directory = Path(path).expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise VastDeploymentError("prepared deployment must be an existing private directory")
    deployment = parse_json_bytes(
        read_regular_file(directory / "deployment.json", label="deployment.json"),
        label="deployment.json")
    validate_deployment_payload(deployment, expected_role=expected_role)
    if expected_role == "worker":
        return _check_prepared_worker_deployment(directory)
    return _check_prepared_learner_deployment(directory)
