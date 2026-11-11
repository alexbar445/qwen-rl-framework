"""Strict, CPU-only profile and model-manifest validation for Vast Workers."""
from __future__ import annotations

import ipaddress
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PROFILE_SCHEMA_VERSION = 2
WORKER_PROFILE_FIELDS = frozenset({
    "schema_version", "role", "deployment_id", "output_root", "remote_root",
    "image_ref", "platform", "cuda_arch", "runtime_contract",
    "worker_config_path", "model_manifest_path", "model_manifest_sha256",
    "model_root_container", "worker_port", "local_forward_port",
    "remote_policy_port", "learner_local_url", "gpu_name", "disk_gb",
    "state_reserve_gb", "image_layer_reserve_gb",
})
LEARNER_PROFILE_FIELDS = frozenset({
    "schema_version", "role", "deployment_id", "output_root", "remote_root",
    "image_ref", "platform", "cuda_arch", "runtime_contract",
    "runtime_config_path", "asset_manifest_path", "asset_manifest_sha256",
    "asset_root_container", "state_root_container", "learner_port",
    "local_forward_port", "worker_routes", "credential_envs", "gpu_name",
    "gpu_count", "disk_gb",
    "state_reserve_gb", "image_layer_reserve_gb",
})
WORKER_DEPLOYMENT_FIELDS = frozenset({
    "schema_version", "role", "deployment_id", "network_mode", "remote_root", "image_ref",
    "platform", "cuda_arch", "runtime_contract", "model_root_container",
    "worker_port", "local_forward_port", "remote_policy_port", "learner_local_url",
    "worker_url", "remote_policy_url", "runtime_not_validated", "worker_json_sha256",
    "start_worker_script_sha256", "model_manifest_sha256", "gpu_name", "disk_gb",
    "state_reserve_gb", "image_layer_reserve_gb",
})
LEARNER_DEPLOYMENT_FIELDS = frozenset({
    "schema_version", "role", "deployment_id", "network_mode", "remote_root", "image_ref",
    "platform", "cuda_arch", "runtime_contract", "asset_root_container",
    "state_root_container", "learner_port", "local_forward_port", "learner_url",
    "local_learner_url", "worker_routes",
    "credential_envs", "gpu_name", "gpu_count", "disk_gb", "state_reserve_gb",
    "image_layer_reserve_gb", "runtime_config_sha256", "asset_manifest_sha256",
    "start_learner_script_sha256", "runtime_config_validation", "runtime_not_validated",
})
WORKER_RUNTIME_CONTRACT_FIELDS = frozenset({
    "worker_entrypoint", "download_tool", "ninfer_binary",
    "worker_runtime", "ninfer_runtime", "torch_runtime",
})
LEARNER_RUNTIME_CONTRACT_FIELDS = frozenset({
    "learner_entrypoint", "learner_runtime", "torch_runtime",
})
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:([0-9a-f]{64})$")
_SAFE_REMOTE_ROOT = re.compile(r"^/[A-Za-z0-9._/-]+$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CUDA_ARCH = re.compile(r"^[0-9]{2,3}a?$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_FORBIDDEN_CREDENTIAL_ENVS = frozenset({
    "PATH", "HOME", "SHELL", "USER", "LOGNAME", "PWD", "PYTHONPATH",
    "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES", "ARC_RL_ROOT", "HF_HOME", "TMPDIR",
})


class VastDeploymentError(RuntimeError):
    """Profile, manifest, or prepared-package validation failed."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("JSON cannot contain non-finite values")


def parse_json_bytes(data: bytes, *, label: str) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise VastDeploymentError(f"{label} is not valid strict JSON: {exc}") from exc


def _object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != fields:
        missing = sorted(fields - frozenset(value)) if isinstance(value, dict) else sorted(fields)
        extra = sorted(frozenset(value) - fields) if isinstance(value, dict) else []
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        suffix = ": " + "; ".join(details) if details else ""
        raise VastDeploymentError(f"{label} fields do not match the strict schema{suffix}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or _CONTROL.search(value):
        raise VastDeploymentError(f"{label} must be a non-empty string without control characters")
    upper = value.upper()
    if (value.startswith("<") or value.endswith(">") or "REPLACE_ME" in upper
            or "FILL_ME" in upper or "TODO" in upper or "PLACEHOLDER" in upper
            or "example.invalid" in value):
        raise VastDeploymentError(f"{label} contains a placeholder value")
    return value


def _positive_int(value: Any, label: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VastDeploymentError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise VastDeploymentError(f"{label} exceeds its maximum value")
    return value


def _safe_local_path(value: Any, label: str) -> Path:
    raw = _text(value, label)
    path = Path(raw).expanduser()
    if not path.is_absolute() or ".." in Path(raw).parts or path == Path("/"):
        raise VastDeploymentError(f"{label} must be an absolute path without '..'")
    return path


def _safe_container_path(value: Any, label: str) -> str:
    raw = _text(value, label)
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts or str(path) != raw or path == PurePosixPath("/"):
        raise VastDeploymentError(f"{label} must be a normalized absolute container path without '..'")
    return raw


def _safe_remote_root(value: Any) -> str:
    raw = _text(value, "profile.remote_root")
    path = PurePosixPath(raw)
    if (not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts
            or str(path) != raw or raw.startswith(("/proc/", "/sys/"))):
        raise VastDeploymentError("profile.remote_root must be a normalized private absolute path")
    if not _SAFE_REMOTE_ROOT.fullmatch(raw):
        raise VastDeploymentError("profile.remote_root may contain only letters, digits, slash, dot, dash, and underscore")
    return raw


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    if not _SHA256.fullmatch(text) or set(text) == {"0"}:
        raise VastDeploymentError(f"{label} must be a real lowercase SHA-256 digest")
    return text


def digest_pinned_image(value: Any) -> str:
    text = _text(value, "profile.image_ref")
    match = _IMAGE_REF.fullmatch(text)
    if match is None or set(match.group(1)) == {"0"}:
        raise VastDeploymentError("profile.image_ref must pin a non-placeholder image by @sha256:<64 lowercase hex>")
    if "//" in text or ".." in text.split("@", 1)[0]:
        raise VastDeploymentError("profile.image_ref is not a valid digest-pinned image reference")
    return text


def loopback_http_url(value: Any, label: str) -> tuple[str, int]:
    from urllib.parse import urlsplit
    text = _text(value, label)
    parsed = urlsplit(text)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as exc:
        raise VastDeploymentError(f"{label} must use a numeric loopback address and explicit port") from exc
    if (parsed.scheme != "http" or address != ipaddress.ip_address("127.0.0.1")
            or port is None or not 1 <= port <= 65535
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise VastDeploymentError(f"{label} must be an HTTP loopback origin inside the SSH tunnel")
    return text.rstrip("/"), port


@dataclass(frozen=True)
class VastProfile:
    role: str
    deployment_id: str
    output_root: Path
    remote_root: str
    image_ref: str
    platform: str
    cuda_arch: str
    runtime_contract: dict[str, Any]
    worker_config_path: Path
    model_manifest_path: Path
    model_manifest_sha256: str
    model_root_container: str
    worker_port: int
    local_forward_port: int
    remote_policy_port: int
    learner_local_url: str
    learner_local_port: int
    gpu_name: str
    disk_gb: int
    state_reserve_gb: int
    image_layer_reserve_gb: int
    profile_path: Path

    @property
    def worker_url(self) -> str:
        return f"http://127.0.0.1:{self.local_forward_port}"

    @property
    def remote_policy_url(self) -> str:
        return f"http://127.0.0.1:{self.remote_policy_port}"

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.deployment_id

    def deployment_payload(self, *, worker_json_sha256: str,
                           start_worker_script_sha256: str) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "role": self.role,
            "deployment_id": self.deployment_id,
            "network_mode": "ssh_tunnel",
            "remote_root": self.remote_root,
            "image_ref": self.image_ref,
            "platform": self.platform,
            "cuda_arch": self.cuda_arch,
            "runtime_contract": dict(self.runtime_contract),
            "model_root_container": self.model_root_container,
            "worker_port": self.worker_port,
            "local_forward_port": self.local_forward_port,
            "remote_policy_port": self.remote_policy_port,
            "learner_local_url": self.learner_local_url,
            "worker_url": self.worker_url,
            "remote_policy_url": self.remote_policy_url,
            "runtime_not_validated": True,
            "worker_json_sha256": worker_json_sha256,
            "start_worker_script_sha256": start_worker_script_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "gpu_name": self.gpu_name,
            "disk_gb": self.disk_gb,
            "state_reserve_gb": self.state_reserve_gb,
            "image_layer_reserve_gb": self.image_layer_reserve_gb,
        }


def validate_deployment_payload(value: Any, *, expected_role: str | None = None) -> dict[str, Any]:
    """Validate the canonical, non-secret JSON shared with the SSH attach code."""
    if not isinstance(value, dict):
        raise VastDeploymentError("deployment.json must be an object")
    role = value.get("role")
    if expected_role is not None and role != expected_role:
        raise VastDeploymentError(f"deployment.json role must be explicitly '{expected_role}'")
    fields_for_role = (WORKER_DEPLOYMENT_FIELDS if role == "worker"
                       else LEARNER_DEPLOYMENT_FIELDS if role == "learner" else None)
    if fields_for_role is None:
        raise VastDeploymentError("deployment.json requires an explicit supported role")
    row = _object(value, fields_for_role, "deployment.json")
    if row["schema_version"] != PROFILE_SCHEMA_VERSION or isinstance(row["schema_version"], bool):
        raise VastDeploymentError("deployment.json schema_version is unsupported")
    if role != "worker":
        return validate_learner_deployment_payload(row)
    deployment_id = _text(row["deployment_id"], "deployment.json deployment_id")
    if not _SAFE_ID.fullmatch(deployment_id):
        raise VastDeploymentError("deployment.json deployment_id is invalid")
    if row["network_mode"] != "ssh_tunnel":
        raise VastDeploymentError("deployment.json network_mode must be ssh_tunnel")
    remote_root = _safe_remote_root(row["remote_root"])
    digest_pinned_image(row["image_ref"])
    if row["platform"] != "linux/amd64" or row["cuda_arch"] != "120a":
        raise VastDeploymentError("deployment.json must declare linux/amd64 and native CUDA arch 120a")
    if row["runtime_not_validated"] is not True:
        raise VastDeploymentError("deployment.json must preserve runtime_not_validated=true")
    runtime = _object(row["runtime_contract"], WORKER_RUNTIME_CONTRACT_FIELDS,
                      "deployment.json runtime_contract")
    expected_paths = {
        "worker_entrypoint": "/opt/qwen-vast/entrypoint.sh",
        "download_tool": "/opt/qwen-vast/download.py",
        "ninfer_binary": "/usr/local/bin/ninfer-serve",
    }
    for key, expected in expected_paths.items():
        if runtime[key] != expected:
            raise VastDeploymentError(f"deployment.json runtime_contract.{key} is invalid")
    if any(runtime[key] is not True for key in ("worker_runtime", "ninfer_runtime", "torch_runtime")):
        raise VastDeploymentError("deployment.json runtime_contract is incomplete")
    model_root = PurePosixPath(_safe_container_path(
        row["model_root_container"], "deployment.json model_root_container"))
    remote_path = PurePosixPath(remote_root)
    if remote_path == model_root or remote_path in model_root.parents or model_root in remote_path.parents:
        raise VastDeploymentError("deployment config root and model cache root must be disjoint")
    worker_port = _positive_int(row["worker_port"], "deployment.json worker_port", maximum=65535)
    local_forward = _positive_int(row["local_forward_port"], "deployment.json local_forward_port", maximum=65535)
    remote_policy = _positive_int(row["remote_policy_port"], "deployment.json remote_policy_port", maximum=65535)
    if local_forward == worker_port or remote_policy == worker_port:
        raise VastDeploymentError("deployment.json tunnel ports collide")
    learner_url, learner_port = loopback_http_url(row["learner_local_url"], "deployment.json learner_local_url")
    worker_url, worker_url_port = loopback_http_url(row["worker_url"], "deployment.json worker_url")
    policy_url, policy_url_port = loopback_http_url(row["remote_policy_url"], "deployment.json remote_policy_url")
    if (learner_url != row["learner_local_url"] or learner_port in {local_forward, remote_policy}
            or worker_url != f"http://127.0.0.1:{local_forward}" or worker_url_port != local_forward
            or policy_url != f"http://127.0.0.1:{remote_policy}" or policy_url_port != remote_policy):
        raise VastDeploymentError("deployment.json loopback URLs differ from their tunnel port fields")
    _sha256(row["worker_json_sha256"], "deployment.json worker_json_sha256")
    _sha256(row["start_worker_script_sha256"], "deployment.json start_worker_script_sha256")
    _sha256(row["model_manifest_sha256"], "deployment.json model_manifest_sha256")
    if _text(row["gpu_name"], "deployment.json gpu_name") != "RTX 5090":
        raise VastDeploymentError("deployment.json gpu_name must be RTX 5090")
    _positive_int(row["disk_gb"], "deployment.json disk_gb")
    _positive_int(row["state_reserve_gb"], "deployment.json state_reserve_gb")
    _positive_int(row["image_layer_reserve_gb"], "deployment.json image_layer_reserve_gb")
    return row


def validate_learner_deployment_payload(value: Any) -> dict[str, Any]:
    row = _object(value, LEARNER_DEPLOYMENT_FIELDS, "Learner deployment.json")
    if row["schema_version"] != PROFILE_SCHEMA_VERSION or isinstance(row["schema_version"], bool):
        raise VastDeploymentError("Learner deployment.json schema_version is unsupported")
    deployment_id = _text(row["deployment_id"], "Learner deployment.json deployment_id")
    if not _SAFE_ID.fullmatch(deployment_id) or row["role"] != "learner":
        raise VastDeploymentError("Learner deployment.json must have explicit role=learner and safe id")
    if row["network_mode"] != "ssh_tunnel":
        raise VastDeploymentError("Learner deployment.json network_mode must be ssh_tunnel")
    remote_root = _safe_remote_root(row["remote_root"])
    digest_pinned_image(row["image_ref"])
    if row["platform"] != "linux/amd64" or not isinstance(row["cuda_arch"], str) or not _CUDA_ARCH.fullmatch(row["cuda_arch"]):
        raise VastDeploymentError("Learner deployment.json platform or CUDA architecture is invalid")
    if row["runtime_not_validated"] is not True:
        raise VastDeploymentError("Learner deployment.json must preserve runtime_not_validated=true")
    runtime = _object(row["runtime_contract"], LEARNER_RUNTIME_CONTRACT_FIELDS,
                      "Learner deployment.json runtime_contract")
    if (runtime["learner_entrypoint"] != "/opt/qwen-vast/learner-entrypoint.sh"
            or runtime["learner_runtime"] is not True or runtime["torch_runtime"] is not True):
        raise VastDeploymentError("Learner runtime contract is incomplete")
    asset_root = PurePosixPath(_safe_container_path(
        row["asset_root_container"], "Learner deployment.json asset_root_container"))
    state_root = PurePosixPath(_safe_container_path(
        row["state_root_container"], "Learner deployment.json state_root_container"))
    _learner_disjoint_roots(remote_root, str(asset_root), str(state_root))
    learner_port = _positive_int(row["learner_port"], "Learner deployment.json learner_port", maximum=65535)
    local_forward = _positive_int(row["local_forward_port"], "Learner deployment.json local_forward_port", maximum=65535)
    if learner_port == local_forward:
        raise VastDeploymentError("Learner deployment listener and local forward ports must differ")
    learner_url, learner_url_port = loopback_http_url(row["learner_url"], "Learner deployment.json learner_url")
    local_url, local_url_port = loopback_http_url(row["local_learner_url"],
                                                 "Learner deployment.json local_learner_url")
    if (learner_url != f"http://127.0.0.1:{learner_port}" or learner_url_port != learner_port
            or local_url != f"http://127.0.0.1:{local_forward}" or local_url_port != local_forward):
        raise VastDeploymentError("Learner deployment loopback URLs differ from port fields")
    _parse_worker_routes(row["worker_routes"], learner_port=learner_port,
                         local_forward_port=local_forward)
    envs = row["credential_envs"]
    if (not isinstance(envs, list) or not envs
            or any(not isinstance(name, str) for name in envs)
            or len(set(envs)) != len(envs)):
        raise VastDeploymentError("Learner deployment credential_envs must be unique and non-empty")
    for index, name in enumerate(envs):
        text = _text(name, f"deployment.credential_envs[{index}]")
        if not _ENV_NAME.fullmatch(text) or text in _FORBIDDEN_CREDENTIAL_ENVS:
            raise VastDeploymentError("Learner deployment contains an unsafe credential environment name")
    if not isinstance(row["gpu_name"], str) or not row["gpu_name"].strip():
        raise VastDeploymentError("Learner deployment gpu_name is required")
    _positive_int(row["gpu_count"], "Learner deployment gpu_count", maximum=16)
    _positive_int(row["disk_gb"], "Learner deployment disk_gb")
    _positive_int(row["state_reserve_gb"], "Learner deployment state_reserve_gb")
    _positive_int(row["image_layer_reserve_gb"], "Learner deployment image_layer_reserve_gb")
    for key in ("runtime_config_sha256", "asset_manifest_sha256", "start_learner_script_sha256"):
        _sha256(row[key], f"Learner deployment.{key}")
    if row["runtime_config_validation"] not in {"full_cpu", "static_deferred"}:
        raise VastDeploymentError("Learner runtime_config_validation is invalid")
    return row


def load_vast_profile(path: str | Path) -> VastProfile:
    """Load and validate a complete target profile without network/GPU access."""
    profile_path = Path(path).expanduser()
    if profile_path.is_symlink() or not profile_path.is_file():
        raise VastDeploymentError("Vast profile must be an existing regular file, not a symlink")
    try:
        raw = parse_json_bytes(profile_path.read_bytes(), label="Vast profile")
    except OSError as exc:
        raise VastDeploymentError(f"cannot read Vast profile: {exc}") from exc
    row = _object(raw, WORKER_PROFILE_FIELDS, "Vast Worker profile")
    if row["schema_version"] != PROFILE_SCHEMA_VERSION or isinstance(row["schema_version"], bool):
        raise VastDeploymentError(f"Vast Worker profile schema_version must be {PROFILE_SCHEMA_VERSION}")
    if row["role"] != "worker":
        raise VastDeploymentError("Vast Worker profile role must be explicitly 'worker'")

    deployment_id = _text(row["deployment_id"], "profile.deployment_id")
    if not _SAFE_ID.fullmatch(deployment_id):
        raise VastDeploymentError("profile.deployment_id is not a safe deployment identifier")
    output_root = _safe_local_path(row["output_root"], "profile.output_root")
    remote_root = _safe_remote_root(row["remote_root"])
    image_ref = digest_pinned_image(row["image_ref"])
    platform = _text(row["platform"], "profile.platform")
    if platform != "linux/amd64":
        raise VastDeploymentError("Vast Worker image platform must be linux/amd64")
    cuda_arch = _text(row["cuda_arch"], "profile.cuda_arch")
    if cuda_arch != "120a":
        raise VastDeploymentError("Vast RTX 5090 Worker image must declare native CUDA arch 120a")

    runtime = _object(row["runtime_contract"], WORKER_RUNTIME_CONTRACT_FIELDS, "runtime_contract")
    expected_paths = {
        "worker_entrypoint": "/opt/qwen-vast/entrypoint.sh",
        "download_tool": "/opt/qwen-vast/download.py",
        "ninfer_binary": "/usr/local/bin/ninfer-serve",
    }
    for key, expected in expected_paths.items():
        if _text(runtime[key], f"runtime_contract.{key}") != expected:
            raise VastDeploymentError(f"runtime_contract.{key} must be {expected}")
    for key in ("worker_runtime", "ninfer_runtime", "torch_runtime"):
        if runtime[key] is not True:
            raise VastDeploymentError(f"runtime_contract.{key} must assert the packaged runtime component")

    worker_config_path = _safe_local_path(row["worker_config_path"], "profile.worker_config_path")
    model_manifest_path = _safe_local_path(row["model_manifest_path"], "profile.model_manifest_path")
    manifest_sha = _sha256(row["model_manifest_sha256"], "profile.model_manifest_sha256")
    model_root = _safe_container_path(row["model_root_container"], "profile.model_root_container")
    remote_path = PurePosixPath(remote_root)
    model_path = PurePosixPath(model_root)
    if remote_path == model_path or remote_path in model_path.parents or model_path in remote_path.parents:
        raise VastDeploymentError("profile.remote_root and model_root_container must be disjoint")

    worker_port = _positive_int(row["worker_port"], "profile.worker_port", maximum=65535)
    local_forward_port = _positive_int(row["local_forward_port"], "profile.local_forward_port", maximum=65535)
    remote_policy_port = _positive_int(row["remote_policy_port"], "profile.remote_policy_port", maximum=65535)
    if local_forward_port == worker_port:
        # They are different machines, but requiring distinct values prevents a
        # confusing local/remote port collision in hand-written tunnel commands.
        raise VastDeploymentError("local_forward_port and worker_port must differ")
    if remote_policy_port == worker_port:
        raise VastDeploymentError("remote_policy_port and worker_port must differ on the Vast container")
    learner_url, learner_port = loopback_http_url(row["learner_local_url"], "profile.learner_local_url")
    if learner_port in {local_forward_port}:
        raise VastDeploymentError("learner_local_url port must differ from local_forward_port")
    if remote_policy_port == learner_port:
        raise VastDeploymentError("remote_policy_port must differ from learner_local_url port")

    gpu_name = _text(row["gpu_name"], "profile.gpu_name")
    if gpu_name != "RTX 5090":
        raise VastDeploymentError("this Vast deployment profile is restricted to RTX 5090")
    disk_gb = _positive_int(row["disk_gb"], "profile.disk_gb")
    state_reserve_gb = _positive_int(row["state_reserve_gb"], "profile.state_reserve_gb")
    image_layer_reserve_gb = _positive_int(
        row["image_layer_reserve_gb"], "profile.image_layer_reserve_gb")

    return VastProfile(
        role="worker",
        deployment_id=deployment_id,
        output_root=output_root,
        remote_root=remote_root,
        image_ref=image_ref,
        platform=platform,
        cuda_arch=cuda_arch,
        runtime_contract=dict(runtime),
        worker_config_path=worker_config_path,
        model_manifest_path=model_manifest_path,
        model_manifest_sha256=manifest_sha,
        model_root_container=model_root,
        worker_port=worker_port,
        local_forward_port=local_forward_port,
        remote_policy_port=remote_policy_port,
        learner_local_url=learner_url,
        learner_local_port=learner_port,
        gpu_name=gpu_name,
        disk_gb=disk_gb,
        state_reserve_gb=state_reserve_gb,
        image_layer_reserve_gb=image_layer_reserve_gb,
        profile_path=profile_path,
    )


@dataclass(frozen=True)
class VastWorkerRoute:
    worker_id: str
    local_url: str
    remote_port: int

    def model_dump(self) -> dict[str, Any]:
        return {"worker_id": self.worker_id, "local_url": self.local_url,
                "remote_port": self.remote_port}


@dataclass(frozen=True)
class VastLearnerProfile:
    role: str
    deployment_id: str
    output_root: Path
    remote_root: str
    image_ref: str
    platform: str
    cuda_arch: str
    runtime_contract: dict[str, Any]
    runtime_config_path: Path
    asset_manifest_path: Path
    asset_manifest_sha256: str
    asset_root_container: str
    state_root_container: str
    learner_port: int
    local_forward_port: int
    worker_routes: tuple[VastWorkerRoute, ...]
    credential_envs: tuple[str, ...]
    gpu_name: str
    gpu_count: int
    disk_gb: int
    state_reserve_gb: int
    image_layer_reserve_gb: int
    profile_path: Path

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.deployment_id

    @property
    def learner_url(self) -> str:
        return f"http://127.0.0.1:{self.learner_port}"

    @property
    def local_learner_url(self) -> str:
        return f"http://127.0.0.1:{self.local_forward_port}"

    def deployment_payload(
        self, *, runtime_config_sha256: str, asset_manifest_sha256: str,
        start_learner_script_sha256: str, runtime_config_validation: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "role": "learner",
            "deployment_id": self.deployment_id,
            "network_mode": "ssh_tunnel",
            "remote_root": self.remote_root,
            "image_ref": self.image_ref,
            "platform": self.platform,
            "cuda_arch": self.cuda_arch,
            "runtime_contract": dict(self.runtime_contract),
            "asset_root_container": self.asset_root_container,
            "state_root_container": self.state_root_container,
            "learner_port": self.learner_port,
            "local_forward_port": self.local_forward_port,
            "learner_url": self.learner_url,
            "local_learner_url": self.local_learner_url,
            "worker_routes": [route.model_dump() for route in self.worker_routes],
            "credential_envs": list(self.credential_envs),
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "disk_gb": self.disk_gb,
            "state_reserve_gb": self.state_reserve_gb,
            "image_layer_reserve_gb": self.image_layer_reserve_gb,
            "runtime_config_sha256": runtime_config_sha256,
            "asset_manifest_sha256": asset_manifest_sha256,
            "start_learner_script_sha256": start_learner_script_sha256,
            "runtime_config_validation": runtime_config_validation,
            "runtime_not_validated": True,
        }


def _validate_runtime_contract(value: Any) -> dict[str, Any]:
    runtime = _object(value, LEARNER_RUNTIME_CONTRACT_FIELDS, "learner runtime_contract")
    if runtime["learner_entrypoint"] != "/opt/qwen-vast/learner-entrypoint.sh":
        raise VastDeploymentError("runtime_contract.learner_entrypoint must be /opt/qwen-vast/learner-entrypoint.sh")
    if runtime["learner_runtime"] is not True or runtime["torch_runtime"] is not True:
        raise VastDeploymentError("learner runtime_contract must declare learner_runtime and torch_runtime")
    return dict(runtime)


def _learner_disjoint_roots(remote_root: str, asset_root: str, state_root: str) -> None:
    paths = {
        "remote_root": PurePosixPath(remote_root),
        "asset_root_container": PurePosixPath(asset_root),
        "state_root_container": PurePosixPath(state_root),
    }
    names = tuple(paths)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            left, right = paths[first], paths[second]
            if left == right or left in right.parents or right in left.parents:
                raise VastDeploymentError(f"{first} and {second} must be disjoint container paths")


def _parse_worker_routes(value: Any, *, learner_port: int, local_forward_port: int) -> tuple[VastWorkerRoute, ...]:
    if not isinstance(value, list) or not value:
        raise VastDeploymentError("profile.worker_routes must be a non-empty list")
    result: list[VastWorkerRoute] = []
    worker_ids: set[str] = set()
    port_targets: dict[int, str] = {}
    local_ports: set[int] = set()
    for index, raw in enumerate(value):
        item = _object(raw, frozenset({"worker_id", "local_url", "remote_port"}),
                       f"profile.worker_routes[{index}]")
        worker_id = _text(item["worker_id"], f"worker_routes[{index}].worker_id")
        if not _SAFE_ID.fullmatch(worker_id) or worker_id in worker_ids:
            raise VastDeploymentError("worker_routes must have unique safe worker_id values")
        worker_ids.add(worker_id)
        local_url, local_port = loopback_http_url(item["local_url"],
                                                  f"worker_routes[{index}].local_url")
        if local_port in {local_forward_port, learner_port}:
            raise VastDeploymentError("worker local_url port conflicts with a Learner tunnel/listener port")
        remote_port = _positive_int(item["remote_port"],
                                    f"worker_routes[{index}].remote_port", maximum=65535)
        if remote_port == learner_port:
            raise VastDeploymentError("worker remote_port conflicts with learner_port")
        previous = port_targets.get(remote_port)
        if previous is not None and previous != local_url:
            raise VastDeploymentError("one remote worker port cannot tunnel to multiple local origins")
        port_targets[remote_port] = local_url
        local_ports.add(local_port)
        result.append(VastWorkerRoute(worker_id, local_url, remote_port))
    return tuple(result)


def load_vast_learner_profile(path: str | Path) -> VastLearnerProfile:
    """Load the explicit role=learner profile; never infer or translate roles."""
    profile_path = Path(path).expanduser()
    if profile_path.is_symlink() or not profile_path.is_file():
        raise VastDeploymentError("Vast Learner profile must be an existing regular file, not a symlink")
    try:
        raw = parse_json_bytes(profile_path.read_bytes(), label="Vast Learner profile")
    except OSError as exc:
        raise VastDeploymentError(f"cannot read Vast Learner profile: {exc}") from exc
    row = _object(raw, LEARNER_PROFILE_FIELDS, "Vast Learner profile")
    if row["schema_version"] != PROFILE_SCHEMA_VERSION or isinstance(row["schema_version"], bool):
        raise VastDeploymentError(f"Vast Learner profile schema_version must be {PROFILE_SCHEMA_VERSION}")
    if row["role"] != "learner":
        raise VastDeploymentError("Vast Learner profile role must be explicitly 'learner'")
    deployment_id = _text(row["deployment_id"], "learner profile.deployment_id")
    if not _SAFE_ID.fullmatch(deployment_id):
        raise VastDeploymentError("Learner deployment_id is invalid")
    output_root = _safe_local_path(row["output_root"], "learner profile.output_root")
    remote_root = _safe_remote_root(row["remote_root"])
    image_ref = digest_pinned_image(row["image_ref"])
    platform = _text(row["platform"], "learner profile.platform")
    if platform != "linux/amd64":
        raise VastDeploymentError("Vast Learner image platform must be linux/amd64")
    cuda_arch = _text(row["cuda_arch"], "learner profile.cuda_arch")
    if not _CUDA_ARCH.fullmatch(cuda_arch):
        raise VastDeploymentError("learner profile.cuda_arch must be an explicit numeric CUDA architecture")
    runtime_contract = _validate_runtime_contract(row["runtime_contract"])
    runtime_config_path = _safe_local_path(row["runtime_config_path"], "profile.runtime_config_path")
    asset_manifest_path = _safe_local_path(row["asset_manifest_path"], "profile.asset_manifest_path")
    asset_manifest_sha = _sha256(row["asset_manifest_sha256"], "profile.asset_manifest_sha256")
    asset_root = _safe_container_path(row["asset_root_container"], "profile.asset_root_container")
    state_root = _safe_container_path(row["state_root_container"], "profile.state_root_container")
    _learner_disjoint_roots(remote_root, asset_root, state_root)
    learner_port = _positive_int(row["learner_port"], "profile.learner_port", maximum=65535)
    local_forward_port = _positive_int(row["local_forward_port"], "profile.local_forward_port", maximum=65535)
    if learner_port == local_forward_port:
        raise VastDeploymentError("learner_port and local_forward_port must differ")
    routes = _parse_worker_routes(row["worker_routes"], learner_port=learner_port,
                                  local_forward_port=local_forward_port)
    envs = row["credential_envs"]
    if not isinstance(envs, list) or not envs:
        raise VastDeploymentError("profile.credential_envs must be a non-empty explicit list")
    checked_envs: list[str] = []
    for index, value in enumerate(envs):
        name = _text(value, f"credential_envs[{index}]")
        if not _ENV_NAME.fullmatch(name) or name in _FORBIDDEN_CREDENTIAL_ENVS:
            raise VastDeploymentError(f"credential_envs[{index}] is not an allowed secret environment name")
        if name in checked_envs:
            raise VastDeploymentError("credential_envs cannot contain duplicates")
        checked_envs.append(name)
    gpu_name = _text(row["gpu_name"], "profile.gpu_name")
    gpu_count = _positive_int(row["gpu_count"], "profile.gpu_count", maximum=16)
    disk_gb = _positive_int(row["disk_gb"], "profile.disk_gb")
    state_reserve_gb = _positive_int(row["state_reserve_gb"], "profile.state_reserve_gb")
    image_layer_reserve_gb = _positive_int(row["image_layer_reserve_gb"], "profile.image_layer_reserve_gb")
    return VastLearnerProfile(
        role="learner", deployment_id=deployment_id, output_root=output_root,
        remote_root=remote_root, image_ref=image_ref, platform=platform, cuda_arch=cuda_arch,
        runtime_contract=runtime_contract, runtime_config_path=runtime_config_path,
        asset_manifest_path=asset_manifest_path, asset_manifest_sha256=asset_manifest_sha,
        asset_root_container=asset_root, state_root_container=state_root,
        learner_port=learner_port, local_forward_port=local_forward_port,
        worker_routes=routes, credential_envs=tuple(checked_envs), gpu_name=gpu_name,
        gpu_count=gpu_count, disk_gb=disk_gb, state_reserve_gb=state_reserve_gb,
        image_layer_reserve_gb=image_layer_reserve_gb, profile_path=profile_path,
    )


@dataclass(frozen=True)
class ModelManifest:
    files: tuple[Any, ...]
    sha256: str
    size_bytes: int
    largest_file_bytes: int
    raw_bytes: bytes


def load_model_manifest(path: str | Path, expected_sha256: str) -> ModelManifest:
    """Read immutable file metadata; never fetch or mutate model data."""
    manifest_path = Path(path).expanduser()
    raw_bytes = read_regular_file(manifest_path, label="model manifest")
    expected_sha = _sha256(expected_sha256, "profile.model_manifest_sha256")
    from .download import DownloadError, parse_manifest_bytes

    try:
        parsed = parse_manifest_bytes(raw_bytes, expected_sha256=expected_sha)
    except DownloadError as exc:
        raise VastDeploymentError(str(exc)) from exc
    return ModelManifest(parsed.files, parsed.sha256, parsed.total_size, parsed.largest_size, raw_bytes)


def read_regular_file(path: Path, *, label: str) -> bytes:
    """Read an existing regular file without following a final symlink."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise VastDeploymentError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise VastDeploymentError(f"{label} must be a regular file, not a symlink or special file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise VastDeploymentError(f"cannot read {label}: {exc}") from exc
