"""Materialize the central Learner runtime from a site Gateway inventory.

This module is intentionally offline: it turns an existing runtime YAML and a
Gateway ``GET /v1/workers?readiness=model`` response into a validated runtime
YAML plus an in-memory binding for the private Gateway bearer. It never sends
the runtime configuration, ARC credentials, or task data to a site.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import shlex
import stat
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from ..runtime.config import load_config
from .contracts import GPUCapability

DEFAULT_EXPECTED_WORKERS = 50
_SITE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_NODE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_ASSIGNMENT = re.compile(r"^export ([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_WORKER_TOKEN = re.compile(r"^[A-Za-z0-9_-]{16,512}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_KEY_PARTS = frozenset({"token", "secret", "apikey", "privatekey"})
_MAX_RUNTIME_ENV_BYTES = 65536


class RuntimeEntryError(ValueError):
    """An unsafe or inconsistent runtime materialization input."""


@dataclass(frozen=True)
class RuntimeMaterialization:
    """Validated runtime YAML and process-local secret bindings.

    ``environment`` is omitted from repr so logging the returned result cannot
    print a bearer. Callers may pass it directly to a child-process environment
    or install it in the current process before invoking the runtime manager.
    """

    yaml_text: str
    worker_ids: tuple[str, ...]
    environment: Mapping[str, str] = field(repr=False, compare=False)

    @property
    def worker_count(self) -> int:
        return len(self.worker_ids)


def _object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeEntryError(f"duplicate inventory JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise RuntimeEntryError("inventory JSON contains a non-finite number")


def load_gateway_inventory(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    """Load the public Gateway worker-list response without accepting duplicate JSON keys."""
    if isinstance(source, Mapping):
        value: Any = deepcopy(dict(source))
    else:
        try:
            value = json.loads(
                Path(source).expanduser().read_text(encoding="utf-8"),
                object_pairs_hook=_object_pairs_no_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeEntryError("Gateway inventory cannot be read as strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"workers"} or not isinstance(value["workers"], list):
        raise RuntimeEntryError("Gateway inventory must match the {workers: [...]} response schema")
    return value


def read_private_gateway_bearer(path: str | Path) -> str:
    """Read a mode-0400/0600 Gateway bearer using the cluster token-file contract."""
    from .server import read_token_file

    try:
        token = read_token_file(path)
    except (OSError, ValueError) as exc:
        raise RuntimeEntryError("Gateway bearer must be read from a private regular file") from exc
    _validate_bearer(token)
    return token


def _validate_bearer(value: Any) -> str:
    if not isinstance(value, str) or len(value) < 16 or any(char.isspace() for char in value):
        raise RuntimeEntryError("Gateway bearer must be a nonempty single-line token of at least 16 characters")
    return value


def load_runtime_environment(path: str | Path, *, required_names: set[str]) -> dict[str, str]:
    """Read the generated shell-export file as data without evaluating it.

    ``required_names`` contains the service/Gateway ``WorkerConfig.auth_token_env``
    values from the validated runtime config. The file must contain exactly one
    generated export for each name; secret values are omitted from diagnostics.
    """
    if not isinstance(required_names, (set, frozenset)):
        raise RuntimeEntryError("required_names must be a set of Worker environment names")
    names = set(required_names)
    if any(not isinstance(name, str) or not _ENV_NAME.fullmatch(name) for name in names):
        raise RuntimeEntryError("required_names contains an invalid Worker environment variable name")
    if "ARC_ENV_TOKEN" in names:
        raise RuntimeEntryError("central ARC_ENV_TOKEN is not a Worker environment binding")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(Path(path).expanduser(), flags)
    except OSError:
        raise RuntimeEntryError("runtime environment file is unavailable or is not a regular file") from None
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1):
            raise RuntimeEntryError("runtime environment file must be an owned mode-0600 regular file")
        if info.st_size > _MAX_RUNTIME_ENV_BYTES:
            raise RuntimeEntryError("runtime environment file is too large")
        chunks: list[bytes] = []
        remaining = _MAX_RUNTIME_ENV_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
    except OSError:
        raise RuntimeEntryError("runtime environment file could not be read safely") from None
    finally:
        os.close(descriptor)

    if len(content) > _MAX_RUNTIME_ENV_BYTES:
        raise RuntimeEntryError("runtime environment file is too large")
    try:
        text = content.decode("utf-8")
    except UnicodeError:
        raise RuntimeEntryError("runtime environment file must contain UTF-8 export lines") from None
    if "\r" in text or "\x00" in text:
        raise RuntimeEntryError("runtime environment file contains an invalid line ending or byte")
    if not names and text in {"", "\n"}:
        return {}
    if not text.endswith("\n"):
        raise RuntimeEntryError("runtime environment file must end with a newline")

    values: dict[str, str] = {}
    for line_number, line in enumerate(text[:-1].split("\n"), start=1):
        match = _ENV_ASSIGNMENT.fullmatch(line)
        if match is None:
            raise RuntimeEntryError(f"runtime environment line {line_number} must be one export NAME=value")
        name, encoded_value = match.groups()
        if name == "ARC_ENV_TOKEN":
            raise RuntimeEntryError("runtime environment file cannot contain central ARC_ENV_TOKEN")
        if name not in names:
            raise RuntimeEntryError(f"runtime environment contains an unexpected name: {name}")
        if name in values:
            raise RuntimeEntryError(f"runtime environment contains a duplicate name: {name}")
        try:
            parsed = shlex.split(encoded_value, comments=False, posix=True)
        except ValueError:
            raise RuntimeEntryError(f"runtime environment line {line_number} has invalid shell quoting") from None
        if len(parsed) != 1 or not _WORKER_TOKEN.fullmatch(parsed[0]):
            raise RuntimeEntryError(f"runtime environment value for {name} is not a valid Worker token")
        values[name] = parsed[0]

    missing = names - values.keys()
    if missing:
        raise RuntimeEntryError("runtime environment is missing required names: " + ", ".join(sorted(missing)))
    return values


def _validate_endpoint(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeEntryError("Gateway endpoint must be an explicit HTTP(S) origin")
    parsed = urlsplit(value)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise RuntimeEntryError("Gateway endpoint has an invalid port") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment):
        raise RuntimeEntryError("Gateway endpoint must be an origin without credentials or a path")
    # Reuse the cluster's origin policy: plaintext bearer traffic is restricted
    # to private/LAN destinations, while public origins require TLS.
    from .settings import validate_credential_origin

    try:
        validate_credential_origin(value)
    except ValueError as exc:
        raise RuntimeEntryError("Gateway bearer requires a private LAN HTTP origin or HTTPS") from exc
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _default_auth_env(site_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]", "_", site_id).upper()
    return f"QWEN_GATEWAY_{suffix}_BEARER"


def _load_template(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        value: Any = deepcopy(dict(source))
    else:
        try:
            value = yaml.safe_load(Path(source).expanduser().read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise RuntimeEntryError("runtime template cannot be read as YAML") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeEntryError("runtime template must be a YAML mapping with string keys")
    return value


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                normalized = re.sub(r"[^a-z0-9]", "", key.lower())
                components = set(re.findall(r"[a-z0-9]+", key.lower()))
                public_health_key = normalized in {"publictokencount", "supportsfulltokenlogprobs"}
                if (not public_health_key and
                        (components & _SENSITIVE_KEY_PARTS or "apikey" in normalized or "privatekey" in normalized)):
                    return True
            if _contains_sensitive_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _validate_health(
    row: Mapping[str, Any],
    worker_id: str,
    *,
    expected_backend: str,
    expected_quantization: str,
) -> tuple[str, str]:
    health = row.get("worker_health")
    if not isinstance(health, Mapping):
        raise RuntimeEntryError(f"Gateway Worker {worker_id} is missing worker_health")
    if (health.get("ready") is not True or health.get("role") != "worker"
            or health.get("worker_id") != worker_id or health.get("backend") != expected_backend):
        raise RuntimeEntryError(f"Gateway Worker {worker_id} does not have the configured ready backend")
    artifact_sha = health.get("artifact_fingerprint")
    base_sha = health.get("base_fingerprint")
    if not isinstance(artifact_sha, str) or not _SHA256.fullmatch(artifact_sha):
        raise RuntimeEntryError(f"Gateway Worker {worker_id} has no valid artifact fingerprint")
    if not isinstance(base_sha, str) or not _SHA256.fullmatch(base_sha):
        raise RuntimeEntryError(f"Gateway Worker {worker_id} has no valid base fingerprint")
    telemetry_epoch = health.get("telemetry_epoch")
    if not isinstance(telemetry_epoch, str) or not telemetry_epoch.strip():
        raise RuntimeEntryError(f"Gateway Worker {worker_id} has no telemetry epoch")
    quantization = health.get("quantization")
    if expected_quantization == "ternary":
        if expected_backend != "ninfer":
            raise RuntimeEntryError("T2 Workers require the NInfer backend")
        from .registry import worker_health_is_model_valid

        if (not isinstance(quantization, Mapping)
                or quantization.get("mode") != "ternary"
                or not worker_health_is_model_valid(worker_id, dict(health))):
            raise RuntimeEntryError(f"Gateway Worker {worker_id} has invalid native T2 health evidence")
    else:
        if expected_quantization != "torchao" or expected_backend != "vllm":
            raise RuntimeEntryError("runtime templates support only vLLM NVFP4 or NInfer T2")
        from .registry import worker_health_is_model_valid

        if (not isinstance(quantization, Mapping)
                or quantization.get("mode") != "torchao"
                or not worker_health_is_model_valid(worker_id, dict(health))):
            raise RuntimeEntryError(f"Gateway Worker {worker_id} has invalid NVFP4 health evidence")
    return artifact_sha, base_sha


def _worker_rows(
    inventory: Mapping[str, Any], *, site_id: str, endpoint: str,
    auth_token_env: str, expected_workers: int, expected_backend: str,
    expected_quantization: str,
) -> list[dict[str, Any]]:
    raw_rows = inventory["workers"]
    if len(raw_rows) != expected_workers:
        raise RuntimeEntryError(
            f"Gateway inventory has {len(raw_rows)} model-ready Workers; expected exactly {expected_workers}"
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    artifact_shas: set[str] = set()
    base_shas: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise RuntimeEntryError(f"Gateway inventory workers[{index}] must be an object")
        if _contains_sensitive_key(raw):
            raise RuntimeEntryError("Gateway inventory contains a private credential field")
        worker_id = raw.get("worker_id")
        node_id = raw.get("node_id")
        if (not isinstance(worker_id, str) or not _NODE_ID.fullmatch(worker_id)
                or node_id != worker_id):
            raise RuntimeEntryError(f"Gateway inventory workers[{index}] has an invalid Worker/node identity")
        if worker_id in seen:
            raise RuntimeEntryError(f"Gateway inventory contains duplicate Worker ID {worker_id}")
        seen.add(worker_id)
        if raw.get("site_id") != site_id:
            raise RuntimeEntryError(f"Gateway Worker {worker_id} is outside the selected site scope")
        if raw.get("model_ready") is not True or type(raw.get("eligible")) is not bool:
            raise RuntimeEntryError(f"Gateway Worker {worker_id} is not from a model-ready inventory")
        worker_port = raw.get("worker_port")
        if isinstance(worker_port, bool) or not isinstance(worker_port, int) or not 1 <= worker_port <= 65535:
            raise RuntimeEntryError(f"Gateway Worker {worker_id} has no valid registered worker_port")
        base_path = raw.get("base_path")
        expected_base_path = f"/v1/workers/{worker_id}"
        if base_path != expected_base_path:
            raise RuntimeEntryError(f"Gateway Worker {worker_id} has an unexpected route prefix")

        gpus = raw.get("gpus")
        if not isinstance(gpus, list) or len(gpus) != 1:
            raise RuntimeEntryError(
                f"Gateway Worker {worker_id} must report exactly one GPU to resolve its node-local device"
            )
        try:
            gpu = GPUCapability.model_validate(gpus[0], strict=True)
        except Exception as exc:
            raise RuntimeEntryError(f"Gateway Worker {worker_id} has an invalid GPU capability") from exc
        if not isinstance(raw.get("boot_id"), str) or not raw["boot_id"].strip():
            raise RuntimeEntryError(f"Gateway Worker {worker_id} has no registered boot ID")
        artifact_sha, base_sha = _validate_health(
            raw, worker_id, expected_backend=expected_backend,
            expected_quantization=expected_quantization,
        )
        artifact_shas.add(artifact_sha)
        base_shas.add(base_sha)

        rows.append({
            "id": worker_id,
            "host": None,
            "port": worker_port,
            "gpu": gpu.index,
            "scope": "gateway",
            "site_id": site_id,
            "node_id": worker_id,
            "endpoint": endpoint,
            "url_prefix": expected_base_path,
            "auth_token_env": auth_token_env,
            "model_sha256": artifact_sha,
        })
    if len(artifact_shas) != 1 or len(base_shas) != 1:
        raise RuntimeEntryError("Gateway model-ready Workers do not share one artifact and base fingerprint")
    return sorted(rows, key=lambda row: row["id"])


def _validate_yaml_config(value: Mapping[str, Any]) -> None:
    yaml_text = yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True)
    try:
        with tempfile.TemporaryDirectory(prefix="qwen-runtime-entry-") as temp_dir:
            path = Path(temp_dir) / "runtime.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            load_config(path)
    except (OSError, ValueError) as exc:
        raise RuntimeEntryError(f"materialized runtime configuration is invalid: {exc}") from exc


def materialize_runtime_config(
    template: str | Path | Mapping[str, Any],
    inventory: str | Path | Mapping[str, Any],
    *,
    site_id: str,
    gateway_endpoint: str,
    gateway_bearer: str,
    expected_workers: int = DEFAULT_EXPECTED_WORKERS,
    auth_token_env: str | None = None,
    central_arc_token: str | None = None,
) -> RuntimeMaterialization:
    """Build a validated central runtime using every model-ready Worker in one site.

    Worker ports come from each Gateway inventory row, never a guessed default.
    GPU comes from the Gateway's single reported node capability; ambiguous
    multi-GPU rows fail.
    """
    if not isinstance(site_id, str) or not _SITE_ID.fullmatch(site_id):
        raise RuntimeEntryError("site_id is invalid")
    if isinstance(expected_workers, bool) or not isinstance(expected_workers, int) or expected_workers < 1:
        raise RuntimeEntryError("expected_workers must be a positive integer")
    endpoint = _validate_endpoint(gateway_endpoint)
    bearer = _validate_bearer(gateway_bearer)
    central_token = os.environ.get("ARC_ENV_TOKEN") if central_arc_token is None else central_arc_token
    if central_token and hmac.compare_digest(bearer, central_token):
        raise RuntimeEntryError("Gateway bearer must be distinct from the central ARC_ENV_TOKEN")

    token_env = auth_token_env or _default_auth_env(site_id)
    if not isinstance(token_env, str) or not _ENV_NAME.fullmatch(token_env) or token_env == "ARC_ENV_TOKEN":
        raise RuntimeEntryError("Gateway auth_token_env must be a dedicated environment variable name")
    template_data = _load_template(template)
    inventory_data = load_gateway_inventory(inventory)
    worker_inference = template_data.get("worker_inference")
    if not isinstance(worker_inference, Mapping):
        raise RuntimeEntryError("runtime template is missing worker_inference settings")
    expected_backend = worker_inference.get("backend", "vllm")
    expected_quantization = worker_inference.get("quantization", "torchao")
    if not isinstance(expected_backend, str) or expected_backend not in {"vllm", "ninfer"}:
        raise RuntimeEntryError("runtime template selects an unsupported Worker backend")
    if not isinstance(expected_quantization, str) or expected_quantization not in {"torchao", "ternary"}:
        raise RuntimeEntryError("runtime template has an invalid Worker source quantization")
    if ((expected_backend, expected_quantization) not in {("vllm", "torchao"), ("ninfer", "ternary")}):
        raise RuntimeEntryError("runtime templates support only vLLM NVFP4 or NInfer T2")
    workers = _worker_rows(
        inventory_data, site_id=site_id, endpoint=endpoint, auth_token_env=token_env,
        expected_workers=expected_workers, expected_backend=expected_backend,
        expected_quantization=expected_quantization,
    )
    config_data = deepcopy(template_data)
    config_data["workers"] = workers
    _validate_yaml_config(config_data)
    yaml_text = yaml.safe_dump(config_data, sort_keys=False, allow_unicode=True)
    return RuntimeMaterialization(
        yaml_text=yaml_text,
        worker_ids=tuple(row["id"] for row in workers),
        environment=MappingProxyType({token_env: bearer}),
    )


__all__ = [
    "DEFAULT_EXPECTED_WORKERS",
    "RuntimeEntryError",
    "RuntimeMaterialization",
    "load_gateway_inventory",
    "load_runtime_environment",
    "materialize_runtime_config",
    "read_private_gateway_bearer",
]
