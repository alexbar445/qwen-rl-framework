"""Build strict site bootstrap defaults from published runtime descriptors."""
from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from ..contracts import worker as worker_wire
from ..runtime.config import (
    WorkerInferenceConfig,
)
from ..worker.backend import NINFER_FIELDS
from .node_config import BootstrapDefaults, ReverseSSHDefaults
from .images import ExportedImageArchive, ImageArchiveDescriptor
from .worker_release import WorkerRelease, WorkerReleaseError, load_worker_release

DEFAULT_GATEWAY_PORT = 8310
DEFAULT_WORKER_PORT = 8241
DEFAULT_MODEL_CACHE_ROOT = Path("/var/lib/qwen-cluster/models")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FAMILY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
_SENSITIVE_KEY_PARTS = frozenset({
    "token", "bearer", "authorization", "secret", "secrets", "password",
    "passwords", "credential", "credentials", "apikey", "privatekey",
})


class DeploymentConfigError(ValueError):
    """An input descriptor cannot produce safe, validated site defaults."""


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeploymentConfigError(f"duplicate descriptor JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise DeploymentConfigError("descriptor JSON contains a non-finite number")


def _mapping_source(value: Mapping[str, Any] | str | Path, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        data: Any = dict(value)
    else:
        try:
            data = json.loads(
                Path(value).expanduser().read_text(encoding="utf-8"),
                object_pairs_hook=_unique_pairs,
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DeploymentConfigError(f"{label} descriptor cannot be read as strict JSON") from exc
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise DeploymentConfigError(f"{label} descriptor must be a JSON object")
    return data


def _has_private_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).lower()
            normalized = re.sub(r"[^a-z0-9]", "", normalized_key)
            components = set(re.findall(r"[a-z0-9]+", normalized_key))
            if components & _SENSITIVE_KEY_PARTS or "apikey" in normalized or "privatekey" in normalized:
                return True
            if _has_private_field(item):
                return True
    elif isinstance(value, list):
        return any(_has_private_field(item) for item in value)
    return False


def _private_gateway_origin(gateway_lan_ip: str, gateway_port: int) -> str:
    if isinstance(gateway_port, bool) or not isinstance(gateway_port, int) or not 1 <= gateway_port <= 65535:
        raise DeploymentConfigError("gateway_port must be a valid TCP port")
    if not isinstance(gateway_lan_ip, str) or not gateway_lan_ip.strip():
        raise DeploymentConfigError("gateway_lan_ip must be explicitly provided")
    try:
        address = ipaddress.ip_address(gateway_lan_ip)
    except ValueError as exc:
        raise DeploymentConfigError("gateway_lan_ip must be an explicit private LAN IP address") from exc
    if (not address.is_private or address.is_loopback or address.is_link_local
            or address.is_unspecified or address.is_multicast):
        raise DeploymentConfigError("gateway_lan_ip must be a routable private LAN address")
    host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return f"http://{host}:{gateway_port}"


def _same_origin(url: Any, origin: str, *, label: str, sha256: str) -> None:
    if not isinstance(url, str):
        raise DeploymentConfigError(f"{label} must be a URL string")
    parsed = urlsplit(url)
    expected = urlsplit(origin)
    try:
        actual_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        expected_port = expected.port or (443 if expected.scheme == "https" else 80)
    except ValueError as exc:
        raise DeploymentConfigError(f"{label} has an invalid port") from exc
    if (parsed.scheme != expected.scheme or parsed.hostname != expected.hostname
            or actual_port != expected_port or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path != f"/v1/artifacts/{sha256}"):
        raise DeploymentConfigError(f"{label} must use the explicit site Gateway content route")


def _validate_model_descriptor(
    descriptor_source: Mapping[str, Any] | str | Path,
    *, lineage_id: str, gateway_origin: str, container_path: str,
    cache_root: Path,
) -> dict[str, str]:
    descriptor = _mapping_source(descriptor_source, label="model")
    if _has_private_field(descriptor):
        raise DeploymentConfigError("model descriptor cannot contain credentials")
    if descriptor.get("schema") != "qwen-model/v1":
        raise DeploymentConfigError("model descriptor schema must be qwen-model/v1")
    if descriptor.get("lineage_id") != lineage_id:
        raise DeploymentConfigError("published model lineage differs from the requested site lineage")
    artifact_id = descriptor.get("artifact_id")
    artifact_fingerprint = descriptor.get("artifact_fingerprint")
    base_fingerprint = descriptor.get("base_fingerprint")
    manifest_sha = descriptor.get("manifest_sha256")
    if (not isinstance(artifact_id, str) or not _SHA256.fullmatch(artifact_id)
            or artifact_fingerprint != artifact_id):
        raise DeploymentConfigError("published model artifact identity is invalid")
    if not isinstance(base_fingerprint, str) or not _SHA256.fullmatch(base_fingerprint):
        raise DeploymentConfigError("published model base fingerprint is invalid")
    if not isinstance(manifest_sha, str) or not _SHA256.fullmatch(manifest_sha):
        raise DeploymentConfigError("published model manifest SHA-256 is invalid")
    if descriptor.get("sha256", manifest_sha) != manifest_sha:
        raise DeploymentConfigError("published model SHA and manifest_sha256 disagree")
    _same_origin(descriptor.get("manifest_url"), gateway_origin,
                 label="model manifest_url", sha256=manifest_sha)

    files = descriptor.get("files")
    if not isinstance(files, list) or not files:
        raise DeploymentConfigError("published model descriptor must contain its verified file records")
    names: set[str] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise DeploymentConfigError(f"published model files[{index}] must be an object")
        name, digest, size = entry.get("name"), entry.get("sha256"), entry.get("size_bytes")
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or ".." in Path(name).parts or name in names):
            raise DeploymentConfigError(f"published model files[{index}] has an invalid or duplicate name")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise DeploymentConfigError(f"published model files[{index}] has an invalid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise DeploymentConfigError(f"published model files[{index}] has an invalid size")
        _same_origin(entry.get("url"), gateway_origin,
                     label=f"published model files[{index}].url", sha256=digest)
        names.add(name)

    # Project source metadata into ModelDefaults' exact contract; publisher-only
    # file inventories and local receipt paths are not NodeModel fields.
    return {
        "artifact_id": artifact_id,
        "manifest_url": descriptor["manifest_url"],
        "manifest_sha256": manifest_sha,
        "cache_path": str(cache_root / artifact_id),
        "base_fingerprint": base_fingerprint,
        "artifact_fingerprint": artifact_fingerprint,
        "container_path": container_path,
    }


def _image_descriptor(value: Any) -> ImageArchiveDescriptor:
    if isinstance(value, ExportedImageArchive):
        return value.descriptor
    if isinstance(value, ImageArchiveDescriptor):
        return value
    if isinstance(value, (str, Path)):
        raw = _mapping_source(value, label="image")
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise DeploymentConfigError("image descriptor must come from the archive export")
    if _has_private_field(raw):
        raise DeploymentConfigError("image descriptor cannot contain credentials")
    try:
        return ImageArchiveDescriptor.model_validate(raw, strict=True)
    except (ValidationError, TypeError, ValueError) as exc:
        raise DeploymentConfigError("image archive descriptor is invalid") from exc


def _load_profile(value: Mapping[str, Any] | str | Path, *, container_path: str) -> dict[str, Any]:
    profile = _mapping_source(value, label="inference profile")
    if _has_private_field(profile):
        raise DeploymentConfigError("inference profile cannot contain credentials")
    profile_id = profile.get("profile_id")
    family = profile.get("model_family")
    if (not isinstance(profile_id, str) or not profile_id.strip()
            or not isinstance(family, str) or not _FAMILY.fullmatch(family)):
        raise DeploymentConfigError("inference profile requires a valid profile_id and model_family")
    inference = profile.get("inference")
    if not isinstance(inference, Mapping) or not inference:
        raise DeploymentConfigError("inference profile must contain an inference object")
    if "model_path" in inference:
        raise DeploymentConfigError("inference profile cannot choose a deployment model_path")
    allowed = {item.name for item in fields(WorkerInferenceConfig)} - {"model_path"}
    extra = set(inference) - allowed
    if extra:
        raise DeploymentConfigError(f"inference profile has unsupported fields: {sorted(extra)}")
    selected_backend = inference.get("backend", "vllm")
    if selected_backend == "ninfer":
        required = {
            "backend", "quantization", "dtype", "max_model_len", "max_num_seqs", "max_loras",
            *NINFER_FIELDS,
        }
    else:
        required = allowed - {
            "kv_cache_memory_bytes", "backend", "ninfer_artifact_sha256", *NINFER_FIELDS,
        }
    missing = required - set(inference)
    if missing:
        raise DeploymentConfigError(f"inference profile is missing explicit fields: {sorted(missing)}")
    try:
        runtime_inference = WorkerInferenceConfig(model_path=container_path, **dict(inference))
        # Profiles describe only Worker inference. Validate the typed shared
        # fields and the exact backend contract without manufacturing Learner
        # settings or loading the model.
        from ..worker.backend import validate_backend_config

        validate_backend_config(asdict(runtime_inference))
        if (runtime_inference.dtype != "bfloat16"
                or not runtime_inference.enable_prefix_caching
                or runtime_inference.mamba_cache_mode != "align"):
            raise ValueError("Qwen Worker profiles require BF16 and prefix caching with aligned Mamba state")
        if (isinstance(runtime_inference.gpu_memory_utilization, bool)
                or not isinstance(runtime_inference.gpu_memory_utilization, (int, float))
                or not 0 < runtime_inference.gpu_memory_utilization < 1
                or any(type(value) is not int or value < 1 for value in (
                    runtime_inference.max_model_len,
                    runtime_inference.max_num_seqs,
                    runtime_inference.max_num_batched_tokens,
                ))
                or type(runtime_inference.max_loras) is not int
                or runtime_inference.max_loras < 2):
            raise ValueError("Worker context, concurrency, and adapter capacities are invalid")
        if (type(runtime_inference.max_generated_tokens) is not int
                or not 1 <= runtime_inference.max_generated_tokens <= worker_wire.MAX_GENERATED_TOKENS
                or runtime_inference.max_generated_tokens > runtime_inference.max_model_len
                or runtime_inference.max_model_len > worker_wire.MAX_CONTEXT_TOKENS):
            raise ValueError("Worker context or generation budget exceeds the shared runtime contract")
        if runtime_inference.kv_cache_dtype not in {"auto", "fp8", "fp8_e5m2", "fp8_e4m3"}:
            raise ValueError("Worker KV cache dtype is unsupported")
        if (runtime_inference.kv_cache_memory_bytes is not None
                and (type(runtime_inference.kv_cache_memory_bytes) is not int
                     or runtime_inference.kv_cache_memory_bytes <= 0)):
            raise ValueError("Worker KV cache memory budget must be positive")
        speculative = runtime_inference.speculative_config
        if speculative is not None:
            if (not isinstance(speculative, Mapping)
                    or speculative.get("method") not in {"ngram", "mtp"}
                    or type(speculative.get("num_speculative_tokens")) is not int
                    or speculative["num_speculative_tokens"] < 1):
                raise ValueError("Worker speculative decoding settings are invalid")
    except (TypeError, ValueError) as exc:
        raise DeploymentConfigError("inference profile does not satisfy the current Worker runtime contract") from exc
    return {"model_family": family, "inference": asdict(runtime_inference)}


def prepare_bootstrap_defaults(
    model_descriptor: Mapping[str, Any] | str | Path,
    image_descriptor: ExportedImageArchive | ImageArchiveDescriptor | Mapping[str, Any] | str | Path | None = None,
    *,
    lineage_id: str,
    gateway_lan_ip: str,
    admin_ssh_public_key: str,
    inference_profile: Mapping[str, Any] | str | Path,
    worker_release: WorkerRelease | Mapping[str, Any] | str | Path | None = None,
    gateway_image_descriptor: ExportedImageArchive | ImageArchiveDescriptor | Mapping[str, Any] | str | Path | None = None,
    reverse_ssh: ReverseSSHDefaults | Mapping[str, Any] | None = None,
    gateway_port: int = DEFAULT_GATEWAY_PORT,
    worker_port: int = DEFAULT_WORKER_PORT,
    model_cache_root: str | Path = DEFAULT_MODEL_CACHE_ROOT,
) -> BootstrapDefaults:
    """Prepare strict site defaults from already-published model and image descriptors.

    The model cache directory and in-container model path are derived from the
    published artifact identity and profile family. The source model descriptor
    is projected into the smaller ``ModelDefaults`` contract instead of passing
    its `files` and local manifest-path metadata into that strict schema.
    """
    if not isinstance(lineage_id, str) or not lineage_id.strip():
        raise DeploymentConfigError("lineage_id must be explicit")
    gateway_origin = _private_gateway_origin(gateway_lan_ip, gateway_port)
    cache_root = Path(model_cache_root).expanduser()
    if not cache_root.is_absolute():
        raise DeploymentConfigError("model_cache_root must be absolute")
    cache_root = cache_root.resolve(strict=False)

    profile_source = _mapping_source(inference_profile, label="inference profile")
    family = profile_source.get("model_family")
    if not isinstance(family, str) or not _FAMILY.fullmatch(family):
        raise DeploymentConfigError("inference profile model_family is invalid")
    container_path = f"/models/{family}"
    profile = _load_profile(profile_source, container_path=container_path)

    model = _validate_model_descriptor(
        model_descriptor, lineage_id=lineage_id, gateway_origin=gateway_origin,
        container_path=container_path, cache_root=cache_root,
    )
    if (image_descriptor is None) == (worker_release is None):
        raise DeploymentConfigError("configure exactly one of image_descriptor or worker_release")
    release: WorkerRelease | None = None
    gateway_image: ImageArchiveDescriptor | None = None
    image: ImageArchiveDescriptor | None = None
    if worker_release is not None:
        try:
            release = load_worker_release(worker_release)
        except WorkerReleaseError as exc:
            raise DeploymentConfigError(f"worker_release is invalid: {exc}") from exc
        profile_backend = profile["inference"].get("backend", "vllm")
        if release.backend != profile_backend:
            raise DeploymentConfigError(
                f"worker_release backend {release.backend!r} does not match inference profile backend "
                f"{profile_backend!r}"
            )
        if gateway_image_descriptor is None:
            raise DeploymentConfigError("worker_release requires a separate gateway_image_descriptor")
        gateway_image = _image_descriptor(gateway_image_descriptor)
        _same_origin(gateway_image.url, gateway_origin, label="Gateway image archive URL",
                     sha256=gateway_image.archive_sha256)
        for variant in release.variants:
            _same_origin(variant.image.url, gateway_origin, label="Worker release image archive URL",
                         sha256=variant.image.archive_sha256)
    else:
        if gateway_image_descriptor is not None:
            raise DeploymentConfigError("gateway_image_descriptor is only valid with worker_release")
        image = _image_descriptor(image_descriptor)
        _same_origin(image.url, gateway_origin, label="image archive URL", sha256=image.archive_sha256)
    if isinstance(worker_port, bool) or not isinstance(worker_port, int) or not 1 <= worker_port <= 65535:
        raise DeploymentConfigError("worker_port must be a valid TCP port")

    reverse_data: dict[str, Any] | None
    if isinstance(reverse_ssh, ReverseSSHDefaults):
        reverse_data = reverse_ssh.model_dump(mode="json")
    elif reverse_ssh is None:
        reverse_data = None
    elif isinstance(reverse_ssh, Mapping):
        reverse_data = dict(reverse_ssh)
    else:
        raise DeploymentConfigError("reverse_ssh must match ReverseSSHDefaults")
    if reverse_data is not None:
        local_api_port = reverse_data.get("local_api_port", DEFAULT_GATEWAY_PORT)
        if local_api_port != gateway_port:
            raise DeploymentConfigError("reverse_ssh.local_api_port must match gateway_port")

    defaults_data: dict[str, Any] = {
        "gateway_url": gateway_origin,
        "model": model,
        "worker": {
            "port": worker_port,
            "inference": profile["inference"],
            "lora_rank": 16,
            "lora_alpha": 16,
            "request_timeout": 180.0,
        },
        "lineage_id": lineage_id,
        "admin_ssh_public_key": admin_ssh_public_key,
    }
    if release is not None:
        defaults_data["worker_release"] = release.model_dump(mode="json", by_alias=True)
        defaults_data["gateway_image"] = gateway_image.model_dump(mode="json")
    else:
        assert image is not None
        defaults_data.update(
            worker_image=image.image_id,
            worker_image_mode="archive",
            image_archive=image.model_dump(mode="json"),
        )
    if reverse_data is not None:
        defaults_data["reverse_ssh"] = reverse_data
    try:
        return BootstrapDefaults.model_validate(defaults_data, strict=True)
    except (ValidationError, TypeError, ValueError) as exc:
        raise DeploymentConfigError("derived BootstrapDefaults failed strict validation") from exc


__all__ = [
    "DEFAULT_GATEWAY_PORT",
    "DEFAULT_MODEL_CACHE_ROOT",
    "DEFAULT_WORKER_PORT",
    "DeploymentConfigError",
    "prepare_bootstrap_defaults",
]
