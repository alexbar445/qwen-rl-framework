"""Authenticated standalone generation service for a quantized policy."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import logging
import math
import os
import re
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..contracts import worker as wire
from ..transport.local_policy import link_native_policy, shared_policy_root
from ..transport.tensor_codec import (
    MAX_TENSOR_RAW_BYTES,
    TENSOR_PAYLOAD_CONTENT_TYPE,
    decode_tensor_payload,
    max_encoded_size,
)
from .backend import (
    BackendError,
    DEFAULT_WORKER_MAX_MODEL_LEN,
    WorkerBackend,
    load_worker_backend,
    validate_backend_config,
)
from .native_artifact import (
    WorkerArtifactError,
)
from .residency import (
    ResidencyActionRequest,
    ResidencyStatusResponse,
    parse_residency_settings,
)
from .telemetry import TelemetryStore

TELEMETRY_INTERVAL_SECONDS = 10.0
_MAX_POLICY_JSON_BYTES = 16 * 1024 * 1024


@dataclass
class AdapterVersion:
    version: int
    adapter_hash: str
    file_hashes: dict[str, tuple[str | None, int | None]]
    artifact_id: str | None
    request: Any
    episode_ids: set[str] = field(default_factory=set)
    generation_refs: int = 0


@dataclass
class EpisodeLease:
    policy_version: int
    adapter_hash: str
    purpose: str = "training"
    released: bool = False


@dataclass
class WorkerFailure(Exception):
    status_code: int
    code: str
    message: str


_NATIVE_RESIDENCY_ERROR_MAX_CHARS = 512
_NATIVE_AUTH_VALUE_RE = re.compile(
    r"""(?i)\bauthorization\s*[:=]\s*(?:(?:bearer|basic)\s+)?(?:"[^"]*"|'[^']*'|[^\s,;]+)"""
)
_NATIVE_BEARER_VALUE_RE = re.compile(
    r"""(?i)\b(?:bearer|basic)\s+(?:"[^"]*"|'[^']*'|[^\s,;]+)"""
)
_NATIVE_CREDENTIAL_FIELD_RE = re.compile(
    r"""(?i)\b(?:access[_-]?token|refresh[_-]?token|token|api[_-]?key|secret|password|credential)\b\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)"""
)
_NATIVE_CREDENTIAL_URL_RE = re.compile(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@")


def _safe_native_residency_detail(error: Exception, *, limit: int) -> str:
    """Keep useful native failure context while excluding credential values."""
    detail = str(error)
    secret_markers = (
        "token", "secret", "password", "credential", "api_key", "apikey", "private_key",
    )
    secrets = {
        value
        for name, value in os.environ.items()
        if (any(marker in name.lower() for marker in secret_markers)
            and isinstance(value, str) and len(value) >= 4)
    }
    for secret in sorted(secrets, key=len, reverse=True):
        detail = detail.replace(secret, "[REDACTED]")
    detail = _NATIVE_AUTH_VALUE_RE.sub("[REDACTED]", detail)
    detail = _NATIVE_BEARER_VALUE_RE.sub("[REDACTED]", detail)
    detail = _NATIVE_CREDENTIAL_FIELD_RE.sub("[REDACTED]", detail)
    detail = _NATIVE_CREDENTIAL_URL_RE.sub(r"\1[REDACTED]@", detail)
    detail = re.sub(r"[\x00-\x1f\x7f]+", " ", detail).strip()
    if len(detail) > limit:
        detail = detail[:limit - 3].rstrip() + "..."
    return detail


def _native_residency_failure(action: str, error: Exception) -> WorkerFailure:
    prefix = f"NInfer model {action} failed ({type(error).__name__})"
    detail = _safe_native_residency_detail(
        error,
        limit=max(0, _NATIVE_RESIDENCY_ERROR_MAX_CHARS - len(prefix) - 2),
    )
    message = f"{prefix}: {detail}" if detail else prefix
    return WorkerFailure(502, f"native_{action}_failed", message)


def _normalize_sampling_params(params: dict[str, Any], *, limit: int) -> tuple[int, dict[str, Any]]:
    try:
        return wire.normalize_sampling_params(params, limit=limit)
    except ValueError as exc:
        raise WorkerFailure(422, "invalid_sampling_params", str(exc)) from exc


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    return parsed.scheme, parsed.hostname or "", parsed.port


class WorkerRuntime:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        model_loader: Callable[..., Any],
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.worker_id = config["worker_id"]
        self.inference = dict(config["inference"])
        self.backend_name = self.inference.get("backend", "vllm")
        artifact_identity_path = config.get("artifact_identity_path")
        nested_identity_path = self.inference.get("artifact_identity_path")
        if (artifact_identity_path is not None
                and nested_identity_path is not None
                and nested_identity_path != artifact_identity_path):
            raise ValueError("Worker artifact_identity_path differs from its pinned runtime identity path")
        if artifact_identity_path is None:
            artifact_identity_path = nested_identity_path
        if artifact_identity_path is not None:
            if not isinstance(artifact_identity_path, str) or not Path(artifact_identity_path).is_absolute():
                raise ValueError("Worker artifact_identity_path must be an absolute path")
            if self.backend_name == "ninfer":
                self.inference["artifact_identity_path"] = artifact_identity_path
        self.residency_settings = parse_residency_settings(config)
        if self.residency_settings.enabled and self.backend_name != "ninfer":
            raise ValueError("residency.enabled requires inference.backend=ninfer")
        # Effective per-request generation bound. It comes from this deployment's worker.json
        # (rendered from worker_inference.max_generated_tokens), never from the image, and the
        # wire ceiling still applies.
        configured_limit = self.inference.get("max_generated_tokens", wire.MAX_GENERATED_TOKENS)
        if isinstance(configured_limit, bool) or not isinstance(configured_limit, int) or configured_limit < 1:
            raise WorkerFailure(503, "invalid_worker_config", "inference.max_generated_tokens must be a positive integer")
        self.max_generated_tokens = min(configured_limit, wire.MAX_GENERATED_TOKENS)
        # Separate replicas must not share a backend's default random stream.
        self.inference["seed"] = int.from_bytes(
            hashlib.sha256(self.worker_id.encode()).digest()[:4], "big")
        self.learner_url = config["learner_url"].rstrip("/")
        self.cache = Path(config["adapter_cache"]).expanduser()
        self.shared_policy_root = (
            shared_policy_root(config["shared_policy_root"])
            if "shared_policy_root" in config else None
        )
        if self.shared_policy_root is not None and self.backend_name != "ninfer":
            raise ValueError("shared_policy_root requires the NInfer backend")
        self._local_policy_copy_bytes = 0
        self.lora_rank = int(config.get("lora_rank", 16))
        self.lora_alpha = int(config.get("lora_alpha", 16))
        self.request_timeout = float(config.get("request_timeout", 30.0))
        self.model_loader = model_loader
        self.http_transport = http_transport
        self.backend: WorkerBackend | None = None
        self.startup_evidence: wire.WorkerStartupEvidence | None = None
        self.evidence: dict[str, Any] = {}
        self.quantization: dict[str, Any] | None = None
        self.runtime_quantization: dict[str, Any] | None = None
        self.base_fingerprint: str | None = None
        self.policy_artifact_fingerprint: str | None = None
        self.active_version = -1
        self.adapter_hash: str | None = None
        self.active_file_hashes: dict[str, tuple[str | None, int | None]] | None = None
        self.active_adapter_request: Any = None
        self.ready = False
        self.policy_ready = False
        self.max_num_seqs = int(self.inference["max_num_seqs"])
        self.max_loras = int(self.inference.get("max_loras", 2))
        self.generation_slots = asyncio.Semaphore(self.max_num_seqs)
        self.generation_state = asyncio.Condition()
        self.active_generations = 0
        self.pending_generations = 0
        self.adapters: dict[int, AdapterVersion] = {}
        # Released lease IDs remain as tombstones for this Worker epoch. This
        # makes delayed acquire retries harmless and prevents ID reuse.
        self.episode_leases: dict[str, EpisodeLease] = {}
        self.telemetry_epoch = str(uuid4())
        self.telemetry = TelemetryStore(Path(config["telemetry_dir"]).expanduser(),
            worker_id=self.worker_id, telemetry_epoch=self.telemetry_epoch)
        self.telemetry_task: asyncio.Task | None = None
        self.telemetry_error: str | None = None
        self._request_telemetry_failed = False
        self.started_monotonic = time.monotonic()
        self.completed_generations = 0
        self.failed_generations = 0
        self.generated_tokens_total = 0
        self.policy_lock = asyncio.Lock()
        self.residency_operation_lock = asyncio.Lock()
        self.residency_state = "resident"
        self.admission_open = True
        self.last_residency_transition: dict[str, Any] | None = None
        self.last_policy_sync_timing: dict[str, Any] = {}
        self.last_policy_prepare_timing: dict[str, Any] = {}
        self.prepared_manifest: wire.PolicyManifest | None = None
        self.learner_client: httpx.AsyncClient | None = None

    def auth_headers(self) -> dict[str, str]:
        token = os.environ.get("ARC_ENV_TOKEN")
        if not token:
            raise WorkerFailure(503, "auth_not_configured", "ARC_ENV_TOKEN is required")
        return {"Authorization": f"Bearer {token}"}

    async def start(self) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
        # A new Worker epoch has no live leases for previous process versions.
        # Keep only its adapter capacity; registry eviction cannot see orphans
        # left by earlier epochs, which otherwise accumulate on tmpfs.
        await asyncio.to_thread(self._prune_previous_epoch_adapters)
        self.learner_client = httpx.AsyncClient(
            base_url=f"{self.learner_url}/",
            headers=self.auth_headers(),
            timeout=httpx.Timeout(self.request_timeout),
            transport=self.http_transport,
        )
        try:
            loaded_backend = self.model_loader(
                self.inference,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
            self.backend = await loaded_backend if inspect.isawaitable(loaded_backend) else loaded_backend
            startup = getattr(self.backend, "startup_evidence", None)
            if not isinstance(startup, wire.WorkerStartupEvidence):
                raise RuntimeError("inference backend omitted normalized startup evidence")
            if startup.backend != self.backend_name:
                raise RuntimeError("normalized backend evidence does not match the selected backend")
            configured_quantization = self.inference.get("quantization")
            if (startup.quantization.get("mode") != configured_quantization
                    or startup.runtime_quantization.get("mode") != configured_quantization):
                raise RuntimeError("backend startup evidence differs from the explicit Worker quantization mode")
            if startup.supports_full_token_logprobs is not True:
                raise RuntimeError("inference backend does not prove full sampled-token logprobs")
            if startup.base_trainable_parameters != 0:
                raise RuntimeError("inference backend did not prove that the base model is frozen")

            configured_context = self.inference.get("max_model_len", DEFAULT_WORKER_MAX_MODEL_LEN)
            if (isinstance(configured_context, bool) or not isinstance(configured_context, int)
                    or configured_context < 1):
                raise RuntimeError("Worker max_model_len must be a positive integer")
            if startup.max_model_len < configured_context:
                raise RuntimeError("inference backend context capacity is below Worker max_model_len")
            if startup.max_generated_tokens < self.max_generated_tokens:
                raise RuntimeError("inference backend generation capacity is below the deployment token limit")

            requested_max_num_seqs = self.inference.get("max_num_seqs")
            requested_max_loras = self.inference.get("max_loras", self.max_loras)
            if (isinstance(requested_max_num_seqs, bool)
                    or not isinstance(requested_max_num_seqs, int)
                    or requested_max_num_seqs < 1
                    or startup.max_num_seqs < requested_max_num_seqs):
                raise RuntimeError("inference backend concurrency is below Worker admission capacity")
            if (isinstance(requested_max_loras, bool)
                    or not isinstance(requested_max_loras, int)
                    or requested_max_loras < 1
                    or startup.max_loras < requested_max_loras):
                raise RuntimeError("inference backend adapter capacity is below Worker capacity")

            self.evidence = startup.backend_evidence
            self.startup_evidence = startup
            self.behavior_distribution = startup.behavior_distribution
            self.logical_model_vocab_size = startup.logical_model_vocab_size
            self.public_token_count = startup.public_token_count
            self.quantization = startup.quantization
            self.runtime_quantization = startup.runtime_quantization
            self.base_fingerprint = startup.base_fingerprint
            self.policy_artifact_fingerprint = startup.artifact_fingerprint
            self.ready = True
            await self.persist_scheduler_snapshot("startup")
            self.telemetry_task = asyncio.create_task(self._persist_stats_loop())
        except Exception:
            self.ready = False
            self.policy_ready = False
            if self.backend is not None:
                await asyncio.to_thread(self.backend.close)
                self.backend = None
            if self.learner_client is not None:
                await self.learner_client.aclose()
                self.learner_client = None
            raise

    def _prune_previous_epoch_adapters(self) -> None:
        if self.adapters or self.active_generations or self.pending_generations:
            raise RuntimeError("cannot prune previous-epoch adapters after Worker admission")
        versions = sorted(
            (p for p in self.cache.iterdir()
             if p.is_dir() and not p.is_symlink()
             and p.name.startswith("v") and p.name[1:].isascii()
             and p.name[1:].isdecimal()),
            key=lambda p: int(p.name[1:]), reverse=True,
        )
        for path in versions[self.max_loras:]:
            shutil.rmtree(path)

    async def close(self) -> None:
        self.ready = False
        self.policy_ready = False
        if self.telemetry_task is not None:
            self.telemetry_task.cancel()
            await asyncio.gather(self.telemetry_task, return_exceptions=True)
            self.telemetry_task = None
        try:
            if self.backend is not None:
                await self.persist_scheduler_snapshot("shutdown")
        finally:
            try:
                await self.telemetry.drain()
            finally:
                await self._close_resources()

    async def _close_resources(self) -> None:
        if self.backend is not None:
            await asyncio.to_thread(self.backend.close)
            self.backend = None
        if self.learner_client is not None:
            await self.learner_client.aclose()
            self.learner_client = None

    async def persist_scheduler_snapshot(self, stage: str) -> None:
        """Keep structural metrics even if backend console logging is disabled."""
        log_stats_error = None
        try:
            await self.backend.log_stats()
        except Exception as exc:
            log_stats_error = type(exc).__name__
        snapshot = getattr(self.backend, "resource_snapshot", None)
        await self._write_telemetry("scheduler", {
            "kind": "scheduler_snapshot", "stage": stage,
            "kv_cache": snapshot() if callable(snapshot) else None,
            "log_stats_error": log_stats_error,
        })

    async def _write_telemetry(self, stream: str, record: dict[str, Any]) -> None:
        try:
            await self.telemetry.append(stream, record)
        except Exception as exc:
            self.telemetry_error = type(exc).__name__
            if stream == "scheduler":
                logging.getLogger(__name__).warning("Worker scheduler telemetry unavailable; retrying next interval")
                return
            self._request_telemetry_failed = True
            self.ready = False
            self.policy_ready = False
            raise WorkerFailure(503, "telemetry_unavailable",
                                "required Worker telemetry could not be saved") from exc
        else:
            if stream == "scheduler" and not self._request_telemetry_failed:
                self.telemetry_error = None

    async def _persist_stats_loop(self) -> None:
        while True:
            await asyncio.sleep(TELEMETRY_INTERVAL_SECONDS)
            try:
                await self.persist_scheduler_snapshot("periodic")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.telemetry_error = self.telemetry_error or type(exc).__name__
                logging.getLogger(__name__).exception("Worker scheduler snapshot failed; retrying next interval")

    async def fetch_manifest(self, expected_version: int | None) -> wire.PolicyManifest:
        assert self.learner_client is not None
        manifest_path = "policy" if expected_version is None else f"policy/{expected_version}"
        try:
            response = await self.learner_client.get(manifest_path)
        except httpx.HTTPError as exc:
            raise WorkerFailure(502, "learner_unavailable", "could not fetch learner policy manifest") from exc
        if response.status_code == 404 and expected_version is not None:
            raise WorkerFailure(
                409, "policy_version_unavailable",
                "requested learner policy version is no longer published",
            )
        if response.status_code != 200:
            raise WorkerFailure(502, "learner_policy_error",
                                f"learner GET /{manifest_path} returned HTTP {response.status_code}")
        try:
            manifest = wire.PolicyManifest.model_validate(response.json())
        except Exception as exc:
            raise WorkerFailure(502, "invalid_policy_manifest", "learner policy manifest did not match the contract") from exc
        if expected_version is not None and manifest.version != expected_version:
            raise WorkerFailure(409, "policy_version_unavailable", "learner manifest version does not match the requested version")
        if manifest.artifact_fingerprint != self.policy_artifact_fingerprint:
            raise WorkerFailure(409, "artifact_fingerprint_mismatch", "policy and Worker use different base artifact weights")
        if manifest.base_fingerprint != self.base_fingerprint:
            raise WorkerFailure(409, "base_fingerprint_mismatch", "learner policy was built from a different base model")
        try:
            assert self.backend is not None
            self.backend.validate_policy_quantization(manifest.quantization)
        except WorkerArtifactError as exc:
            raise WorkerFailure(
                409,
                "policy_quantization_mismatch",
                str(exc),
            ) from exc
        names = [item.name for item in manifest.files]
        native_manifest = manifest.quantization.get("mode") == "ternary"
        expected_files = wire.NATIVE_ADAPTER_FILES if native_manifest else wire.ADAPTER_FILES
        if set(names) != expected_files or len(names) != len(expected_files):
            expected = "native NInfer adapter files" if native_manifest else "PEFT adapter files"
            raise WorkerFailure(
                502, "invalid_policy_manifest",
                f"manifest must contain exactly the two {expected}",
            )
        learner_origin = _origin(self.learner_url)
        for item in manifest.files:
            expected_path = f"/policy/{manifest.version}/{item.name}"
            target = urljoin(f"{self.learner_url}/", item.url)
            parsed_target = urlsplit(target)
            if (
                _origin(target) != learner_origin
                or parsed_target.path.rstrip("/") != expected_path
                or parsed_target.username is not None
                or parsed_target.password is not None
                or parsed_target.query
                or parsed_target.fragment
            ):
                raise WorkerFailure(502, "invalid_policy_url", "adapter file URL must target its versioned learner endpoint")
        return manifest

    async def _download_adapter(self, manifest: wire.PolicyManifest) -> Path:
        assert self.learner_client is not None
        self._local_policy_copy_bytes = 0
        if self.shared_policy_root is not None:
            try:
                path, copied = await asyncio.to_thread(
                    link_native_policy, self.shared_policy_root, self.cache, manifest,
                )
            except (OSError, ValueError, TypeError) as exc:
                raise WorkerFailure(503, "shared_policy_unavailable",
                                    "managed local Native policy is unavailable or invalid") from exc
            self._local_policy_copy_bytes = copied
            return path
        final_dir = self.cache / f"v{manifest.version}"
        if final_dir.is_dir() and self._cache_matches(final_dir, manifest):
            return final_dir
        stage = self.cache / f".v{manifest.version}.{uuid4().hex}.tmp"
        stage.mkdir(parents=False, exist_ok=False)
        try:
            for item in manifest.files:
                target = urljoin(f"{self.learner_url}/", item.url)
                is_tensor = not item.name.endswith(".json")
                expected_content_type = (
                    TENSOR_PAYLOAD_CONTENT_TYPE if is_tensor else "application/json"
                )
                raw_limit = MAX_TENSOR_RAW_BYTES if item.size_bytes is None else item.size_bytes
                if is_tensor:
                    if raw_limit > MAX_TENSOR_RAW_BYTES:
                        raise WorkerFailure(
                            413, "adapter_size_limit", "published adapter exceeds the tensor transfer limit",
                        )
                    wire_limit = max_encoded_size(raw_limit)
                else:
                    if raw_limit > _MAX_POLICY_JSON_BYTES:
                        raise WorkerFailure(
                            413, "adapter_size_limit", "published adapter metadata exceeds the transfer limit",
                        )
                    wire_limit = raw_limit
                try:
                    async with self.learner_client.stream("GET", target) as response:
                        if response.status_code != 200:
                            raise WorkerFailure(
                                502, "adapter_download_failed",
                                f"learner adapter endpoint returned HTTP {response.status_code}",
                            )
                        actual_content_type = response.headers.get("content-type", "")
                        actual_content_type = actual_content_type.split(";", 1)[0].strip().lower()
                        if actual_content_type != expected_content_type:
                            raise WorkerFailure(
                                502, "adapter_content_type_mismatch",
                                f"learner adapter payload type did not match {item.name}",
                            )
                        content_encoding = response.headers.get("content-encoding", "identity").lower()
                        if content_encoding not in {"", "identity"}:
                            raise WorkerFailure(
                                502, "adapter_content_encoding_mismatch",
                                "learner adapter payload used an unsupported content encoding",
                            )
                        declared_wire_size: int | None = None
                        raw_content_length = response.headers.get("content-length")
                        if raw_content_length is not None:
                            if not raw_content_length.isascii() or not raw_content_length.isdecimal():
                                raise WorkerFailure(
                                    502, "adapter_transfer_size_invalid",
                                    "learner adapter payload has an invalid Content-Length",
                                )
                            declared_wire_size = int(raw_content_length)
                            if declared_wire_size > wire_limit:
                                raise WorkerFailure(
                                    413, "adapter_transfer_size_limit",
                                    "learner adapter payload exceeds the transfer limit",
                                )
                        transfer = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(transfer) + len(chunk) > wire_limit:
                                raise WorkerFailure(
                                    413, "adapter_transfer_size_limit",
                                    "learner adapter payload exceeds the transfer limit",
                                )
                            transfer.extend(chunk)
                except httpx.HTTPError as exc:
                    raise WorkerFailure(502, "adapter_download_failed", "could not download learner adapter file") from exc
                if declared_wire_size is not None and len(transfer) != declared_wire_size:
                    raise WorkerFailure(422, "adapter_transfer_size_mismatch", f"wire size mismatch for {item.name}")
                if is_tensor:
                    try:
                        contents = decode_tensor_payload(
                            bytes(transfer),
                            max_raw_bytes=raw_limit,
                            safetensors=item.name == "adapter_model.safetensors",
                        )
                    except (TypeError, ValueError) as exc:
                        raise WorkerFailure(
                            422, "adapter_payload_invalid", f"invalid compressed tensor payload for {item.name}",
                        ) from exc
                else:
                    contents = bytes(transfer)
                if len(contents) > raw_limit:
                    raise WorkerFailure(413, "adapter_size_limit", f"adapter file exceeds the limit for {item.name}")
                if item.size_bytes is not None and len(contents) != item.size_bytes:
                    raise WorkerFailure(422, "adapter_size_mismatch", f"size mismatch for {item.name}")
                (stage / item.name).write_bytes(contents)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            stage.replace(final_dir)
            return final_dir
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    @staticmethod
    def _cache_matches(path: Path, manifest: wire.PolicyManifest) -> bool:
        try:
            entries = list(path.iterdir())
        except OSError:
            return False
        if {entry.name for entry in entries} != {item.name for item in manifest.files}:
            return False
        for item in manifest.files:
            file_path = path / item.name
            if file_path.is_symlink() or not file_path.is_file():
                return False
            if item.size_bytes is not None and file_path.stat().st_size != item.size_bytes:
                return False
        return True

    async def _add_policy_adapter(self, version: int, path: Path, adapter_hash: str) -> Any:
        try:
            return await self.backend.add_adapter(version, path, adapter_hash=adapter_hash)
        except Exception as exc:
            logging.getLogger(__name__).exception("inference backend adapter load failed for policy %s", version)
            raise WorkerFailure(500, "adapter_load_failed", "inference backend could not load the versioned policy adapter") from exc

    async def _remove_policy_adapter(self, request: Any) -> None:
        try:
            await self.backend.remove_adapter(request)
        except Exception as exc:
            raise WorkerFailure(500, "adapter_remove_failed", "inference backend could not remove the previous policy adapter") from exc

    async def begin_policy_update(self, *, allow_draining: bool = False) -> bool:
        async with self.generation_state:
            if self.residency_state == "suspended" or (
                self.residency_state == "draining" and not allow_draining
            ):
                return False
        if self.policy_lock.locked():
            return False
        await self.policy_lock.acquire()
        async with self.generation_state:
            if self.residency_state == "suspended" or (
                self.residency_state == "draining" and not allow_draining
            ):
                self.policy_lock.release()
                self.generation_state.notify_all()
                return False
        return True

    async def prepare_policy(self, version: int) -> dict[str, Any]:
        """Prepare immutable CPU files while fenced; never install or admit a policy."""
        prepare = getattr(self.backend, "prepare_adapter", None)
        if not self.residency_settings.enabled or not callable(prepare):
            raise WorkerFailure(409, "policy_preparation_unavailable", "backend cannot prepare a suspended policy")
        if self.policy_lock.locked():
            raise WorkerFailure(409, "worker_busy", "another policy operation is active")
        await self.policy_lock.acquire()
        started = time.perf_counter()
        timings: dict[str, Any] = {"version": version, "status": "running"}
        self.last_policy_prepare_timing = dict(timings)
        try:
            async with self.generation_state:
                if (self.residency_state not in {"suspended", "resident_fenced"}
                        or self.admission_open or not self._episode_generation_quiescent_locked()):
                    raise WorkerFailure(409, "worker_not_fenced", "CPU preparation requires a quiescent fenced Worker")
                if version < self.active_version:
                    raise WorkerFailure(409, "stale_policy_version", "cannot prepare an older policy")
            manifest = await self.fetch_manifest(version)
            hashes = {
                item.name: (
                    item.sha256.lower() if item.sha256 is not None else None,
                    item.size_bytes,
                )
                for item in manifest.files
            }
            previous = self.prepared_manifest
            if previous is not None and (previous.version != version or previous.adapter_hash != manifest.adapter_hash or
                    {
                        item.name: (
                            item.sha256.lower() if item.sha256 is not None else None,
                            item.size_bytes,
                        )
                        for item in previous.files
                    } != hashes):
                raise WorkerFailure(409, "immutable_policy_violation", "a different prepared policy is pending installation")
            existing = self.adapters.get(version)
            if existing is not None and (
                    existing.file_hashes != hashes or existing.adapter_hash != manifest.adapter_hash
                    or existing.artifact_id != manifest.artifact_id):
                raise WorkerFailure(409, "immutable_policy_violation", "prepared policy differs from installed policy")
            download_started = time.perf_counter()
            path = await self._download_adapter(manifest)
            timings["adapter_download_seconds"] = time.perf_counter() - download_started
            timings["adapter_local_policy"] = self.shared_policy_root is not None
            timings["adapter_local_copy_bytes"] = self._local_policy_copy_bytes
            prepared = await prepare(version, path, adapter_hash=manifest.adapter_hash)
            prepared_hash = getattr(prepared, "adapter_hash", None)
            if (not isinstance(prepared_hash, str)
                    or not hmac.compare_digest(prepared_hash, manifest.adapter_hash)):
                raise WorkerFailure(
                    409, "adapter_hash_mismatch",
                    "prepared native policy source SHA differs from the published manifest",
                )
            timings.update(prepared.sync_timings)
            self.prepared_manifest = manifest
            timings["status"] = "succeeded"
            return {"worker_id": self.worker_id, "version": version, "prepared": True,
                    "files": {
                        name: {"sha256": digest, "size_bytes": size}
                        for name, (digest, size) in hashes.items()
                    },
                    "artifact_id": manifest.artifact_id,
                    "preparation_timing": timings}
        except BaseException:
            timings["status"] = "failed"
            raise
        finally:
            timings["worker_policy_prepare_seconds"] = time.perf_counter() - started
            self.last_policy_prepare_timing = dict(timings)
            await self.end_policy_update()

    async def end_policy_update(self) -> None:
        self.policy_lock.release()
        async with self.generation_state:
            self.generation_state.notify_all()

    def _episode_refs_locked(self) -> int:
        return sum(len(entry.episode_ids) for entry in self.adapters.values())

    def _generation_refs_locked(self) -> int:
        return sum(entry.generation_refs for entry in self.adapters.values())

    def _drain_complete_locked(self) -> bool:
        return (
            self._episode_generation_quiescent_locked()
            and not self.policy_lock.locked()
        )

    def _episode_generation_quiescent_locked(self) -> bool:
        return (
            self._episode_refs_locked() == 0
            and self.pending_generations == 0
            and self.active_generations == 0
            and self._generation_refs_locked() == 0
        )

    def _residency_counts_locked(self) -> dict[str, Any]:
        return {
            "active_generations": self.active_generations,
            "pending_generations": self.pending_generations,
            "queued_generations": max(0, self.pending_generations - self.active_generations),
            "episode_refs": self._episode_refs_locked(),
            "generation_refs": self._generation_refs_locked(),
            "policy_update_active": self.policy_lock.locked(),
            "drain_complete": self._drain_complete_locked(),
        }

    async def residency_status(self) -> dict[str, Any]:
        async with self.generation_state:
            counts = self._residency_counts_locked()
            state = self.residency_state
            admission_open = self.admission_open
            policy_version = self.active_version
            adapter_sha256 = self.adapter_hash
            ready = self.ready and state != "suspended"
            policy_ready = self.policy_ready

        native = None
        native_error = None
        model_id = getattr(self.backend, "model_id", None)
        if self.residency_settings.enabled:
            get_residency = getattr(self.backend, "get_residency", None)
            if not callable(get_residency):
                native_error = "selected backend does not expose NInfer residency"
            else:
                try:
                    native = await get_residency()
                    model_id = getattr(self.backend, "model_id", model_id)
                except Exception as exc:
                    native_error = f"{type(exc).__name__}: {exc}"

        return {
            "worker_id": self.worker_id,
            "enabled": self.residency_settings.enabled,
            "state": state,
            "ready": ready,
            "policy_ready": policy_ready,
            "admission_open": admission_open,
            **counts,
            "policy_version": policy_version,
            "adapter_sha256": adapter_sha256,
            "native_model_id": model_id,
            "native": native,
            "native_error": native_error,
            "last_transition": self.last_residency_transition,
        }

    async def _begin_residency_drain(self) -> bool:
        async with self.generation_state:
            if self.residency_state == "suspended":
                return self._drain_complete_locked()
            self.residency_state = "draining"
            self.admission_open = False
            self.generation_state.notify_all()
            try:
                await asyncio.wait_for(
                    self.generation_state.wait_for(self._drain_complete_locked),
                    timeout=self.residency_settings.drain_timeout_seconds,
                )
            except asyncio.TimeoutError:
                return False
            return True

    async def _suspend_residency(self) -> None:
        if not self.residency_settings.enabled:
            raise WorkerFailure(409, "residency_disabled", "Worker residency is disabled in its configuration")
        async with self.generation_state:
            if self.residency_state != "draining":
                raise WorkerFailure(409, "worker_drain_required", "drain the Worker before suspending its model")
            if not self._drain_complete_locked():
                raise WorkerFailure(409, "worker_busy", "Worker still has episode, generation, or policy-update references")
        suspend_model = getattr(self.backend, "suspend_model", None)
        if not callable(suspend_model):
            raise WorkerFailure(409, "residency_backend_unavailable", "selected backend cannot suspend its model")
        try:
            native = await suspend_model()
        except Exception as exc:
            raise _native_residency_failure("suspend", exc) from exc
        if native.get("state") != "suspended" or native.get("registry_admission_held") is not True:
            if native.get("state") == "suspended":
                async with self.generation_state:
                    self.residency_state = "suspended"
                    self.admission_open = False
                    self.generation_state.notify_all()
            raise WorkerFailure(502, "native_suspend_unverified", "NInfer did not confirm a held suspended state")
        async with self.generation_state:
            self.residency_state = "suspended"
            self.admission_open = False
            self.generation_state.notify_all()

    async def _wake_residency(self) -> None:
        if not self.residency_settings.enabled:
            raise WorkerFailure(409, "residency_disabled", "Worker residency is disabled in its configuration")
        async with self.generation_state:
            if self.residency_state == "resident_fenced":
                return
            if self.residency_state != "suspended":
                raise WorkerFailure(409, "worker_not_suspended", "Worker model is not suspended")
        resume_model = getattr(self.backend, "resume_model", None)
        if not callable(resume_model):
            raise WorkerFailure(409, "residency_backend_unavailable", "selected backend cannot resume its model")
        try:
            native = await resume_model()
        except Exception as exc:
            raise _native_residency_failure("resume", exc) from exc
        if native.get("state") != "resident":
            raise WorkerFailure(502, "native_resume_unverified", "NInfer did not confirm resident model memory")
        async with self.generation_state:
            self.residency_state = "resident_fenced"
            self.admission_open = False
            self.generation_state.notify_all()

    async def _admit_residency(self, request: ResidencyActionRequest) -> None:
        if not await self.begin_policy_update(allow_draining=True):
            raise WorkerFailure(409, "worker_busy", "policy update or residency transition is already active")
        try:
            async with self.generation_state:
                if self.residency_state not in {"resident", "resident_fenced", "draining"}:
                    raise WorkerFailure(409, "worker_not_resident", "Worker model must be resident before admission opens")
                if not self.ready or not self.policy_ready:
                    raise WorkerFailure(503, "worker_not_ready", "Worker model and policy must be ready before admission opens")
                already_admitted = self.residency_state == "resident" and self.admission_open
                if request.policy_version != self.active_version:
                    raise WorkerFailure(409, "policy_identity_mismatch", "admit policy_version must match the active Worker policy")
                entry = self.adapters.get(self.active_version)
                if (entry is None or self.adapter_hash is None
                        or not hmac.compare_digest(entry.adapter_hash, request.adapter_sha256)
                        or not hmac.compare_digest(self.adapter_hash, request.adapter_sha256)):
                    raise WorkerFailure(409, "policy_identity_mismatch", "admit adapter SHA differs from the active Worker policy")
                if not already_admitted and not self._episode_generation_quiescent_locked():
                    raise WorkerFailure(409, "worker_busy", "Worker still has episode or generation references")

            if self.residency_settings.enabled:
                get_residency = getattr(self.backend, "get_residency", None)
                if not callable(get_residency):
                    raise WorkerFailure(409, "residency_backend_unavailable", "selected backend cannot verify model residency")
                try:
                    native = await get_residency()
                except Exception as exc:
                    raise WorkerFailure(503, "native_residency_unavailable", f"NInfer residency check failed ({type(exc).__name__})") from exc
                if native.get("state") != "resident" or native.get("registry_admission_held") is not False:
                    raise WorkerFailure(409, "native_model_not_resident", "NInfer model is not resident with registry admission verified")

            async with self.generation_state:
                if self.residency_state not in {"resident", "resident_fenced", "draining"}:
                    raise WorkerFailure(409, "worker_not_resident", "Worker residency changed before admission opened")
                already_admitted = self.residency_state == "resident" and self.admission_open
                if not self.ready or not self.policy_ready:
                    raise WorkerFailure(503, "worker_not_ready", "Worker model and policy changed before admission opened")
                if request.policy_version != self.active_version:
                    raise WorkerFailure(409, "policy_identity_mismatch", "active Worker policy changed before admission opened")
                entry = self.adapters.get(self.active_version)
                if (entry is None or self.adapter_hash is None
                        or not hmac.compare_digest(entry.adapter_hash, request.adapter_sha256)
                        or not hmac.compare_digest(self.adapter_hash, request.adapter_sha256)):
                    raise WorkerFailure(409, "policy_identity_mismatch", "active Worker adapter SHA changed before admission opened")
                if not already_admitted and not self._episode_generation_quiescent_locked():
                    raise WorkerFailure(409, "worker_busy", "Worker acquired work before admission opened")
                self.residency_state = "resident"
                self.admission_open = True
                self.generation_state.notify_all()
        finally:
            await self.end_policy_update()

    async def transition_residency(self, request: ResidencyActionRequest) -> dict[str, Any]:
        async with self.residency_operation_lock:
            started = time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            outcome = "succeeded"
            error = None
            try:
                if request.action == "drain":
                    if not await self._begin_residency_drain():
                        outcome = "timed_out"
                elif request.action == "suspend":
                    await self._suspend_residency()
                elif request.action == "wake":
                    await self._wake_residency()
                else:
                    await self._admit_residency(request)
            except WorkerFailure as exc:
                outcome = "failed"
                error = f"{exc.code}: {exc.message}"
                raise
            except Exception as exc:
                outcome = "failed"
                error = f"{type(exc).__name__}: {exc}"
                raise WorkerFailure(500, "residency_transition_failed", "Worker residency transition failed") from exc
            finally:
                self.last_residency_transition = {
                    "action": request.action,
                    "outcome": outcome,
                    "started_at": started_at,
                    "duration_seconds": max(0.0, time.monotonic() - started),
                    "error": error,
                }
        return await self.residency_status()

    async def apply_manifest(
        self,
        manifest: wire.PolicyManifest,
        *,
        activate: bool = True,
    ) -> dict[str, Any]:
        if type(activate) is not bool:
            raise ValueError("activate must be a boolean")
        apply_started = time.perf_counter()
        sync_timings: dict[str, Any] = {
            "adapter_download_seconds": 0.0,
            "native_retire_seconds": 0.0,
            "worker_backend_add_seconds": 0.0,
            "adapter_cache_cleanup_seconds": 0.0,
        }
        self.last_policy_sync_timing = {}
        adapter_hash = manifest.adapter_hash
        native_manifest = manifest.quantization.get("mode") == "ternary"
        file_hashes = {
            item.name: (
                item.sha256.lower() if item.sha256 is not None else None,
                item.size_bytes,
            )
            for item in manifest.files
        }
        async with self.generation_state:
            if activate and manifest.version < self.active_version:
                raise WorkerFailure(
                    409, "stale_policy_version",
                    "cannot install a policy older than the Worker newest version",
                )
            existing = self.adapters.get(manifest.version)
            if existing is not None:
                if (existing.file_hashes != file_hashes or existing.adapter_hash != adapter_hash
                        or existing.artifact_id != manifest.artifact_id):
                    raise WorkerFailure(
                        409, "immutable_policy_violation",
                        "same policy version was published with different immutable metadata",
                    )
                if activate:
                    self.active_version = manifest.version
                    self._set_active_adapter()
                self._finish_policy_sync_timing(sync_timings, apply_started)
                return {
                    "policy_version": manifest.version,
                    "adapter_hash": existing.adapter_hash,
                    "changed": False,
                }
        download_started = time.perf_counter()
        adapter_path = await self._download_adapter(manifest)
        sync_timings["adapter_download_seconds"] = max(
            0.0, time.perf_counter() - download_started,
        )
        sync_timings["adapter_local_policy"] = self.shared_policy_root is not None
        sync_timings["adapter_local_copy_bytes"] = self._local_policy_copy_bytes
        self._finish_policy_sync_timing(sync_timings, apply_started)
        evicted: AdapterVersion | None = None
        async with self.generation_state:
            if activate and manifest.version < self.active_version:
                raise WorkerFailure(
                    409, "stale_policy_version",
                    "cannot install a policy older than the Worker newest version",
                )
            existing = self.adapters.get(manifest.version)
            if existing is not None:
                if (existing.file_hashes != file_hashes or existing.adapter_hash != adapter_hash
                        or existing.artifact_id != manifest.artifact_id):
                    raise WorkerFailure(
                        409, "immutable_policy_violation",
                        "same policy version was published with different immutable metadata",
                    )
                if activate:
                    self.active_version = manifest.version
                    self._set_active_adapter()
                self._finish_policy_sync_timing(sync_timings, apply_started)
                return {
                    "policy_version": manifest.version,
                    "adapter_hash": existing.adapter_hash,
                    "changed": False,
                }

            previous_active_version = self.active_version

            if len(self.adapters) >= self.max_loras:
                candidates = sorted(
                    (entry for entry in self.adapters.values()
                     if not entry.episode_ids and entry.generation_refs == 0
                     and not (not activate and entry.version == self.active_version)),
                    key=lambda entry: entry.version,
                )
                if not candidates:
                    raise WorkerFailure(
                        409, "adapter_capacity_busy",
                        f"all {self.max_loras} resident adapters have episode or generation references",
                    )
                evicted = candidates[0]
                retire_started = time.perf_counter()
                await self._remove_policy_adapter(evicted.request)
                sync_timings["native_retire_seconds"] = max(
                    0.0, time.perf_counter() - retire_started,
                )
                self._finish_policy_sync_timing(sync_timings, apply_started)
                del self.adapters[evicted.version]

            try:
                backend_add_started = time.perf_counter()
                new_request = await self._add_policy_adapter(
                    manifest.version, adapter_path, adapter_hash,
                )
                if native_manifest:
                    native_adapter_hash = getattr(new_request, "adapter_hash", None)
                    native_artifact_id = getattr(new_request, "adapter_ninfer_artifact_id", None)
                    if (not isinstance(native_adapter_hash, str)
                            or not hmac.compare_digest(native_adapter_hash, adapter_hash)
                            or native_artifact_id != manifest.artifact_id):
                        try:
                            await self._remove_policy_adapter(new_request)
                        except Exception:
                            logging.getLogger(__name__).exception(
                                "could not remove native policy after adapter identity mismatch",
                            )
                        raise WorkerFailure(
                            502, "adapter_hash_mismatch",
                            "native backend handle source SHA differs from the published manifest",
                        )
                sync_timings["worker_backend_add_seconds"] = max(
                    0.0, time.perf_counter() - backend_add_started,
                )
                backend_timings = getattr(new_request, "sync_timings", None)
                if isinstance(backend_timings, dict):
                    sync_timings.update(backend_timings)
                self._finish_policy_sync_timing(sync_timings, apply_started)
            except Exception:
                if evicted is not None:
                    try:
                        evicted.request = await self._add_policy_adapter(
                            evicted.version,
                            self.cache / f"v{evicted.version}",
                            evicted.adapter_hash,
                        )
                        self.adapters[evicted.version] = evicted
                        self._set_active_adapter()
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "could not restore evicted policy version %s after adapter load failure",
                            evicted.version,
                        )
                        self.active_version = (
                            previous_active_version
                            if not activate else max(self.adapters, default=-1)
                        )
                        self._set_active_adapter()
                raise

            self.adapters[manifest.version] = AdapterVersion(
                version=manifest.version,
                adapter_hash=adapter_hash,
                file_hashes=file_hashes,
                artifact_id=manifest.artifact_id,
                request=new_request,
            )
            if evicted is not None:
                self.episode_leases = {
                    episode_id: lease
                    for episode_id, lease in self.episode_leases.items()
                    if lease.policy_version != evicted.version or not lease.released
                }
            if activate:
                self.active_version = manifest.version
                self._set_active_adapter()
                self.policy_ready = True

        if evicted is not None:
            cleanup_started = time.perf_counter()
            try:
                await asyncio.to_thread(shutil.rmtree, self.cache / f"v{evicted.version}")
            except FileNotFoundError:
                pass
            except OSError:
                logging.getLogger(__name__).warning(
                    "could not remove cached policy version %s", evicted.version,
                )
            finally:
                sync_timings["adapter_cache_cleanup_seconds"] = max(
                    0.0, time.perf_counter() - cleanup_started,
                )
        self._finish_policy_sync_timing(sync_timings, apply_started)
        return {
            "policy_version": manifest.version,
            "adapter_hash": adapter_hash,
            "changed": True,
            "activated": activate,
        }

    def _finish_policy_sync_timing(
        self,
        timings: dict[str, Any],
        started: float,
    ) -> None:
        timings["worker_apply_manifest_seconds"] = max(
            0.0, time.perf_counter() - started,
        )
        self.last_policy_sync_timing = dict(timings)

    def _set_active_adapter(self) -> None:
        entry = self.adapters.get(self.active_version)
        if entry is None:
            self.active_version = max(self.adapters, default=-1)
            entry = self.adapters.get(self.active_version)
        self.active_adapter_request = entry.request if entry is not None else None
        self.adapter_hash = entry.adapter_hash if entry is not None else None
        self.active_file_hashes = dict(entry.file_hashes) if entry is not None else None
        self.policy_ready = entry is not None

    def _acquire_episode_locked(self, payload: wire.EpisodeLeaseRequest) -> dict[str, Any]:
        previous = self.episode_leases.get(payload.episode_id)
        if previous is not None:
            if (previous.policy_version != payload.policy_version
                    or previous.adapter_hash != payload.adapter_hash
                    or previous.purpose != payload.purpose):
                raise WorkerFailure(
                    409, "episode_lease_mismatch",
                    "episode_id is already bound to a different policy, SHA, or purpose",
                )
            if previous.released:
                raise WorkerFailure(
                    409, "episode_lease_released",
                    "episode_id was already released and cannot be reacquired",
                )
            return {
                "episode_id": payload.episode_id,
                "policy_version": previous.policy_version,
                "adapter_hash": previous.adapter_hash,
                "purpose": previous.purpose,
                "acquired": False,
            }

        entry = self.adapters.get(payload.policy_version)
        if not self.admission_open:
            raise WorkerFailure(
                409, "worker_draining",
                "Worker is fenced from accepting a new whole-episode lease",
            )
        if entry is None:
            raise WorkerFailure(
                409, "policy_version_unavailable",
                "requested policy version is not resident on this Worker",
            )
        if not hmac.compare_digest(entry.adapter_hash, payload.adapter_hash):
            raise WorkerFailure(
                409, "adapter_hash_mismatch",
                "requested adapter SHA does not match the resident policy version",
            )
        entry.episode_ids.add(payload.episode_id)
        self.episode_leases[payload.episode_id] = EpisodeLease(
            policy_version=payload.policy_version,
            adapter_hash=payload.adapter_hash,
            purpose=payload.purpose,
        )
        return {
            "episode_id": payload.episode_id,
            "policy_version": payload.policy_version,
            "adapter_hash": payload.adapter_hash,
            "purpose": payload.purpose,
            "acquired": True,
        }

    async def acquire_episode(self, payload: wire.EpisodeLeaseRequest) -> dict[str, Any]:
        async with self.generation_state:
            previous = self.episode_leases.get(payload.episode_id)
            if previous is not None:
                return self._acquire_episode_locked(payload)
            if not self.admission_open:
                raise WorkerFailure(
                    409, "worker_draining",
                    "Worker is fenced from accepting a new whole-episode lease",
                )
            if payload.policy_version in self.adapters:
                return self._acquire_episode_locked(payload)
            if payload.purpose != "evaluation":
                return self._acquire_episode_locked(payload)

        # Only an authenticated, explicitly evaluation-scoped episode can load a
        # non-head policy. The lease is created while the policy-update fence is
        # still held, so a concurrent train sync cannot evict it first.
        if not await self.begin_policy_update():
            raise WorkerFailure(409, "worker_draining", "Worker is draining or another policy update is in progress")
        try:
            async with self.generation_state:
                if payload.episode_id in self.episode_leases or payload.policy_version in self.adapters:
                    return self._acquire_episode_locked(payload)
            manifest = await self.fetch_manifest(payload.policy_version)
            declared_adapter = manifest.adapter_hash
            if not hmac.compare_digest(declared_adapter, payload.adapter_hash):
                raise WorkerFailure(
                    409, "adapter_hash_mismatch",
                    "requested adapter SHA does not match the published historical policy",
                )
            await self.apply_manifest(manifest, activate=False)
            async with self.generation_state:
                return self._acquire_episode_locked(payload)
        finally:
            await self.end_policy_update()

    async def release_episode(self, payload: wire.EpisodeLeaseRequest) -> dict[str, Any]:
        async with self.generation_state:
            lease = self.episode_leases.get(payload.episode_id)
            if lease is not None and (
                lease.policy_version != payload.policy_version
                or lease.adapter_hash != payload.adapter_hash
                or lease.purpose != payload.purpose
            ):
                raise WorkerFailure(
                    409, "episode_lease_mismatch",
                    "episode_id is bound to a different policy, SHA, or purpose",
                )
            if lease is None:
                entry = self.adapters.get(payload.policy_version)
                if entry is None:
                    return {"episode_id": payload.episode_id, "purpose": payload.purpose, "released": False}
                if not hmac.compare_digest(entry.adapter_hash, payload.adapter_hash):
                    raise WorkerFailure(
                        409, "adapter_hash_mismatch",
                        "release adapter SHA does not match the resident policy version",
                    )
                lease = EpisodeLease(
                    policy_version=payload.policy_version,
                    adapter_hash=payload.adapter_hash,
                    purpose=payload.purpose,
                    released=True,
                )
                self.episode_leases[payload.episode_id] = lease
                return {"episode_id": payload.episode_id, "purpose": payload.purpose, "released": False}
            if lease.released:
                return {"episode_id": payload.episode_id, "purpose": lease.purpose, "released": False}
            entry = self.adapters.get(lease.policy_version)
            if entry is not None:
                entry.episode_ids.discard(payload.episode_id)
            lease.released = True
            self.generation_state.notify_all()
            return {"episode_id": payload.episode_id, "purpose": lease.purpose, "released": True}

    async def generate(self, request: wire.GenerateRequest, max_tokens: int, safe_params: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        receipt = {
            "attempt_id": uuid4().hex, "request_id": request.request_id,
            "episode_id": request.episode_id, "policy_version": request.policy_version,
            "adapter_sha256": request.adapter_hash, "purpose": request.purpose,
            "prompt_tokens": len(request.prompt_ids), "max_tokens": max_tokens,
        }
        await self._write_telemetry("requests", {"kind": "generation_started", **receipt})
        result = None
        failure = None
        try:
            result = await self._generate_admitted(request, max_tokens, safe_params)
            return result
        except BaseException as exc:
            failure = exc
            raise
        finally:
            await self._write_telemetry("requests", {
                "kind": "generation_finished", **receipt,
                "outcome": ("succeeded" if failure is None else
                            "cancelled" if isinstance(failure, asyncio.CancelledError) else "failed"),
                "metrics": result["extra_fields"]["generation_metrics"] if result is not None else None,
                "request_wall_seconds": time.perf_counter() - started,
                "error_code": getattr(failure, "code", None),
                "error_type": type(failure).__name__ if failure is not None else None,
                "error_message": (
                    failure.message.replace("\n", " ").replace("\r", " ")[:512]
                    if isinstance(failure, WorkerFailure) and failure.code == "generation_failed" else None
                ),
            })

    async def _generate_admitted(self, request: wire.GenerateRequest, max_tokens: int,
                                 safe_params: dict[str, Any]) -> dict[str, Any]:
        async with self.generation_state:
            if not self.ready or not self.policy_ready or self.residency_state == "suspended":
                raise WorkerFailure(503, "worker_not_ready", "worker model is not ready")
            lease = self.episode_leases.get(request.episode_id)
            if lease is None or lease.released:
                raise WorkerFailure(
                    409, "episode_lease_missing",
                    "generation requires a live whole-episode lease",
                )
            if (lease.policy_version != request.policy_version
                    or lease.adapter_hash != request.adapter_hash
                    or lease.purpose != request.purpose):
                raise WorkerFailure(
                    409, "episode_lease_mismatch",
                    "generation policy, adapter SHA, or purpose differs from its episode lease",
                )
            entry = self.adapters.get(request.policy_version)
            if entry is None:
                raise WorkerFailure(409, "policy_version_unavailable", "requested policy version is not resident")
            if not hmac.compare_digest(entry.adapter_hash, request.adapter_hash):
                raise WorkerFailure(409, "adapter_hash_mismatch", "request adapter SHA differs from resident policy")
            entry.generation_refs += 1
            self.pending_generations += 1
        try:
            result = await self._generate(request, max_tokens, safe_params, entry)
        except BaseException:
            self.failed_generations += 1
            raise
        else:
            self.completed_generations += 1
            return result
        finally:
            async with self.generation_state:
                entry.generation_refs -= 1
                self.pending_generations -= 1
                self.generation_state.notify_all()

    async def _generate(
        self,
        request: wire.GenerateRequest,
        max_tokens: int,
        safe_params: dict[str, Any],
        entry: AdapterVersion,
    ) -> dict[str, Any]:
        async with self.generation_slots:
            async with self.generation_state:
                context_limit = min(
                    self.inference.get("max_model_len", DEFAULT_WORKER_MAX_MODEL_LEN),
                    wire.MAX_CONTEXT_TOKENS,
                )
                if len(request.prompt_ids) + max_tokens > context_limit:
                    raise WorkerFailure(422, "context_length_exceeded", f"prompt plus max_tokens must not exceed {context_limit}")
                if not self.ready or not self.policy_ready:
                    raise WorkerFailure(503, "worker_not_ready", "worker model is not ready")
                if self.adapters.get(request.policy_version) is not entry:
                    raise WorkerFailure(409, "policy_version_unavailable", "requested policy version was evicted")
                adapter_request = entry.request
                if adapter_request is None:
                    raise WorkerFailure(503, "worker_not_ready", "worker has no active policy adapter")
                self.active_generations += 1
            try:
                started = time.perf_counter()
                last_token_count = 0

                def record_token_count(count: int) -> None:
                    nonlocal last_token_count
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        return
                    if count > last_token_count:
                        self.generated_tokens_total += count - last_token_count
                        last_token_count = count

                generated = await self.backend.generate(
                    request_id=request.request_id,
                    prompt_ids=request.prompt_ids,
                    max_tokens=max_tokens,
                    sampling_params=safe_params,
                    lora_request=adapter_request,
                    token_count_callback=record_token_count,
                    **({"shared_prefix_tokens": request.shared_prefix_tokens} if self.backend_name == "ninfer" else {}),
                    **({"seed": request.seed} if request.seed is not None else {}),
                )
                token_ids, log_probs, stop_reason = generated.token_ids, generated.log_probs, generated.stop_reason
                # Reconcile the streamed cumulative output with the final response.
                # This also covers a backend that only reports its terminal update.
                record_token_count(len(token_ids))
                if not token_ids or len(token_ids) != len(log_probs):
                    raise WorkerFailure(502, "score_alignment_error", "sampled token IDs and logprobs are empty or misaligned")
                if any(not math.isfinite(value) for value in log_probs):
                    raise WorkerFailure(502, "invalid_log_probability", "backend returned a non-finite sampled-token logprob")
                constraint = safe_params.get("constraint")
                if constraint is not None and (
                    generated.metrics.get("behavior_distribution") != "grammar_conditioned_policy/v1"
                    or generated.metrics.get("grammar_sha256") != constraint["sha256"]
                    or generated.metrics.get("grammar_compiler_sha256") != constraint["compiler_sha256"]
                ):
                    raise WorkerFailure(502, "constraint_not_applied", "backend did not prove the requested grammar-conditioned policy")
                elapsed = time.perf_counter() - started
                return {
                    "token_ids": token_ids,
                    "log_probs": log_probs,
                    "stop_reason": stop_reason,
                    "extra_fields": {
                        "generation_metrics": generated.metrics,
                        "adapter_sha256": entry.adapter_hash,
                        "episode_purpose": request.purpose,
                        "base_fingerprint": self.base_fingerprint,
                        "artifact_fingerprint": self.policy_artifact_fingerprint,
                        "ninfer_artifact_id": self.evidence.get("ninfer_artifact_id"),
                        "ninfer_artifact_sha256": self.evidence.get("ninfer_artifact_sha256"),
                        "base_artifact_kind": self.evidence.get("base_artifact_kind"),
                        "input_basis": self.evidence.get("input_basis"),
                        "base_weight_profile": self.evidence.get("base_weight_profile"),
                        "min_global_steps": entry.version,
                        "max_global_steps": entry.version,
                        "sampling_config": {
                            "distribution": "grammar_conditioned_policy/v1" if constraint is not None else "raw_policy",
                            "behavior_distribution": "grammar_conditioned_policy/v1" if constraint is not None else self.behavior_distribution,
                            **({"grammar_sha256": constraint["sha256"], "grammar_compiler_sha256":constraint["compiler_sha256"]} if constraint is not None else {}),
                            "logical_model_vocab_size": generated.metrics.get("logical_model_vocab_size",self.logical_model_vocab_size) if constraint is not None else self.logical_model_vocab_size,
                            "public_token_count": generated.metrics.get("public_token_count",self.public_token_count) if constraint is not None else self.public_token_count,
                            "temperature": safe_params["temperature"],
                            "top_p": safe_params["top_p"],
                            "top_k": safe_params["top_k"],
                            "repetition_penalty": safe_params["repetition_penalty"],
                            "warpers_neutralized": True,
                            "seed_mode": "request" if request.seed is not None else "worker_rng",
                            "seed": request.seed,
                            "neutralized_fields": list(wire.NEUTRALIZED_SAMPLING_FIELDS),
                        },
                    },
                    "worker_id": self.worker_id,
                    "timing": {"generation_seconds": elapsed},
                }
            except WorkerFailure:
                raise
            except BackendError as exc:
                raise WorkerFailure(502, "generation_failed", str(exc)) from exc
            except Exception as exc:
                raise WorkerFailure(500, "generation_failed", f"inference backend generation failed ({type(exc).__name__})") from exc
            finally:
                async with self.generation_state:
                    self.active_generations -= 1
                    self.generation_state.notify_all()


def create_worker_app(
    config: dict[str, Any],
    *,
    model_loader: Callable[..., Any] | None = None,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create a worker ASGI app; model loading and policy sync happen at startup."""
    required = {"worker_id", "inference", "learner_url", "adapter_cache", "telemetry_dir"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"worker config missing required fields: {', '.join(sorted(missing))}")
    if not isinstance(config["worker_id"], str) or not config["worker_id"].strip():
        raise ValueError("worker_id must be a non-empty string")
    inference = config["inference"]
    if not isinstance(inference, dict) or not isinstance(inference.get("model_path"), str) or not inference["model_path"].strip():
        raise ValueError("inference must be an object with a non-empty model_path")
    validate_backend_config(inference)
    max_num_seqs = inference.get("max_num_seqs")
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs < 1:
        raise ValueError("inference.max_num_seqs must be a positive integer")
    max_loras = inference.get("max_loras", 2)
    if isinstance(max_loras, bool) or not isinstance(max_loras, int) or max_loras < 2:
        raise ValueError("inference.max_loras must be an integer of at least 2")
    if any(not isinstance(config[name], str) or not config[name].strip() for name in ("learner_url", "adapter_cache", "telemetry_dir")):
        raise ValueError("learner_url, adapter_cache and telemetry_dir must be non-empty strings")
    for name in ("lora_rank", "lora_alpha"):
        value = config.get(name, 16)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    timeout = config.get("request_timeout", 30.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("request_timeout must be a positive finite number")
    learner_url = str(config["learner_url"])
    if urlsplit(learner_url).scheme not in {"http", "https"} or not urlsplit(learner_url).netloc:
        raise ValueError("learner_url must be an absolute HTTP(S) URL")
    loader = model_loader or load_worker_backend
    runtime = WorkerRuntime(config, model_loader=loader, http_transport=http_transport)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime.start()
        app.state.runtime = runtime
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Qwen RL generation worker", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.runtime = runtime

    @app.middleware("http")
    async def require_worker_auth(request: Request, call_next):
        try:
            expected = runtime.auth_headers()["Authorization"]
        except WorkerFailure as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, expected):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthorized", "message": "valid bearer authorization is required"}})
        return await call_next(request)

    @app.exception_handler(WorkerFailure)
    async def worker_failure_handler(_request: Request, exc: WorkerFailure):
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})

    @app.get("/health")
    async def health():
        state = app.state.runtime
        resource_snapshot = getattr(state.backend, "resource_snapshot", None)
        async with state.generation_state:
            readiness = state.ready and state.residency_state != "suspended"
            residency_state = state.residency_state
            admission_open = state.admission_open
            residency_counts = state._residency_counts_locked()
        return {
            "ready": readiness,
            "policy_ready": state.policy_ready,
            "residency_enabled": state.residency_settings.enabled,
            "residency_state": residency_state,
            "admission_open": admission_open,
            "role": "worker",
            "worker_id": state.worker_id,
            "backend": state.evidence.get("backend", "vllm"),
            "quantization_mode": state.quantization.get("mode") if state.quantization else None,
            "quantization": state.quantization,
            "runtime_quantization": state.runtime_quantization,
            "is_loaded_in_4bit": state.evidence.get("is_loaded_in_4bit"),
            "module_count": state.evidence.get("module_count"),
            "quantized_tensor_count": state.evidence.get("quantized_tensor_count"),
            "supports_full_token_logprobs": state.evidence.get("supports_full_token_logprobs"),
            "behavior_distribution": state.evidence.get("behavior_distribution"),
            "logical_model_vocab_size": state.evidence.get("logical_model_vocab_size"),
            "public_token_count": state.evidence.get("public_token_count"),
            "quantized_weight_count": state.quantization.get("quantized_weight_count") if state.quantization else None,
            "packed_module_count": state.evidence.get("packed_module_count"),
            "packed_uint8_tensor_count": state.evidence.get("packed_uint8_tensor_count"),
            "packed_uint8_numel": state.evidence.get("packed_uint8_numel"),
            "kv_cache_enabled": state.evidence.get("kv_cache_enabled"),
            "prefix_caching_enabled": state.evidence.get("prefix_caching_enabled"),
            "speculative_decoding": state.evidence.get("speculative_decoding"),
            "max_model_len": state.evidence.get("max_model_len"),
            "max_num_seqs": state.max_num_seqs,
            "active_generations": residency_counts["active_generations"],
            "pending_generations": residency_counts["pending_generations"],
            "queued_generations": residency_counts["queued_generations"],
            "telemetry_epoch": state.telemetry_epoch,
            "telemetry": {"directory": str(state.telemetry.directory),
                          "error": state.telemetry_error,
                          "interval_seconds": TELEMETRY_INTERVAL_SECONDS},
            "uptime_seconds": max(0.0, time.monotonic() - state.started_monotonic),
            "completed_generations": state.completed_generations,
            "failed_generations": state.failed_generations,
            "generated_tokens_total": state.generated_tokens_total,
            "max_generated_tokens": state.max_generated_tokens,
            "base_trainable_parameters": state.evidence.get("base_trainable_parameters"),
            "policy_version": state.active_version,
            "adapter_hash": state.adapter_hash,
            "resident_versions": [
                {
                    "policy_version": entry.version,
                    "adapter_hash": entry.adapter_hash,
                    "episode_refs": len(entry.episode_ids),
                    "generation_refs": entry.generation_refs,
                }
                for entry in sorted(state.adapters.values(), key=lambda item: item.version)
            ],
            "episode_refs": residency_counts["episode_refs"],
            "max_loras": state.max_loras,
            "kv_cache": resource_snapshot() if callable(resource_snapshot) else None,
            "base_fingerprint": state.base_fingerprint,
            "artifact_fingerprint": state.evidence.get("artifact_fingerprint"),
            "source_artifact_fingerprint": state.evidence.get("source_artifact_fingerprint"),
            "ninfer_artifact_id": state.evidence.get("ninfer_artifact_id"),
            "ninfer_artifact_sha256": state.evidence.get("ninfer_artifact_sha256"),
            "base_artifact_kind": state.evidence.get("base_artifact_kind"),
            "input_basis": state.evidence.get("input_basis"),
            "base_weight_profile": state.evidence.get("base_weight_profile"),
            "last_policy_sync_timing": dict(state.last_policy_sync_timing),
            "last_policy_prepare_timing": dict(state.last_policy_prepare_timing),
        }

    @app.get("/residency", response_model=ResidencyStatusResponse)
    async def get_residency():
        return await app.state.runtime.residency_status()

    @app.post("/residency", response_model=ResidencyStatusResponse)
    async def transition_residency(payload: ResidencyActionRequest):
        transition = asyncio.create_task(app.state.runtime.transition_residency(payload))
        try:
            return await asyncio.shield(transition)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            try:
                await asyncio.shield(transition)
            except Exception:
                pass
            raise

    @app.post("/policy/prepare")
    async def prepare_policy(payload: wire.PolicyRequest):
        # A disconnected/cancelled caller cannot orphan conversion writes or release
        # the policy fence before this CPU-only operation has actually finished.
        owned = asyncio.create_task(app.state.runtime.prepare_policy(payload.version))
        try:
            return await asyncio.shield(owned)
        except asyncio.CancelledError as cancellation:
            current = asyncio.current_task()
            while not owned.done():
                if current is not None:
                    current.uncancel()
                try:
                    await asyncio.shield(owned)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not owned.cancelled():
                owned.exception()
            raise cancellation

    @app.post("/policy")
    async def sync_policy(payload: wire.PolicyRequest):
        state = app.state.runtime
        sync_started = time.perf_counter()
        lock_started = time.perf_counter()
        if not await state.begin_policy_update():
            raise WorkerFailure(409, "worker_draining", "Worker is draining or another policy update is in progress")
        state.last_policy_sync_timing = {}
        sync_timing: dict[str, Any] = {
            "policy_update_lock_wait_seconds": max(0.0, time.perf_counter() - lock_started),
        }
        try:
            manifest_started = time.perf_counter()
            try:
                manifest = await state.fetch_manifest(payload.version)
            finally:
                sync_timing["policy_manifest_fetch_seconds"] = max(
                    0.0, time.perf_counter() - manifest_started,
                )
            prepared = state.prepared_manifest
            if prepared is not None and (
                    prepared.version != manifest.version
                    or prepared.adapter_hash != manifest.adapter_hash
                    or prepared.artifact_id != manifest.artifact_id
                    or {item.name: item.size_bytes for item in prepared.files}
                    != {item.name: item.size_bytes for item in manifest.files}):
                raise WorkerFailure(409, "immutable_policy_violation", "install differs from the prepared policy")
            result = await state.apply_manifest(manifest)
            state.prepared_manifest = None
            file_hashes = {
                name: {"sha256": digest, "size_bytes": size}
                for name, (digest, size) in state.adapters[result["policy_version"]].file_hashes.items()
            }
            sync_timing.update(state.last_policy_sync_timing)
            return {
                "worker_id": state.worker_id,
                "version": result["policy_version"],
                "files": file_hashes,
                "artifact_id": manifest.artifact_id,
                "adapter_hash": result["adapter_hash"],
                "changed": result["changed"],
                "sync_timing": sync_timing,
            }
        finally:
            sync_timing.update(state.last_policy_sync_timing)
            sync_timing["worker_policy_sync_seconds"] = max(
                0.0, time.perf_counter() - sync_started,
            )
            state.last_policy_sync_timing = dict(sync_timing)
            await state.end_policy_update()

    @app.post("/episode/acquire")
    async def acquire_episode(payload: wire.EpisodeLeaseRequest):
        return await app.state.runtime.acquire_episode(payload)

    @app.post("/episode/release")
    async def release_episode(payload: wire.EpisodeLeaseRequest):
        return await app.state.runtime.release_episode(payload)

    @app.post("/generate", response_model=wire.GenerationResponse)
    async def generate(payload: wire.GenerateRequest):
        state = app.state.runtime
        max_tokens, safe_params = _normalize_sampling_params(
            payload.sampling_params, limit=state.max_generated_tokens)
        return await state.generate(payload, max_tokens, safe_params)

    return app
