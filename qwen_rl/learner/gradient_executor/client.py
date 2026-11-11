"""Outbound HTTP executor loop for centrally leased gradient jobs."""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from qwen_rl.learner.gradient_transport import tensor_codec
from qwen_rl.transport.tensor_codec import MAX_TENSOR_RAW_BYTES

from qwen_rl.contracts.gradients import (
    GradientContractError,
    GradientContribution,
    GradientJob,
    PreparedWindow,
    validate_compute_profile,
)

from .phase_timing import JobPhaseTiming, measured_async


_TENSOR_TRANSPORT_HEADER = "X-Tensor-Transport"
_LOGGER = logging.getLogger(__name__)


class GradientExecutorError(RuntimeError):
    pass


class GradientLeaseLost(GradientExecutorError):
    """A fenced/expired lease whose compute owner has already been joined."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_binary_fsync(path: Path, raw: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _retryable_executor_failure(error: Exception) -> bool:
    if isinstance(error, (GradientLeaseLost, httpx.TransportError)):
        return True
    if not isinstance(error, httpx.HTTPStatusError):
        return False
    status = error.response.status_code
    if status == 429 or status >= 500:
        return True
    if status != 409:
        return False
    try:
        detail = str(error.response.json().get("detail", "")).lower()
    except Exception:
        detail = ""
    return any(word in detail for word in ("lease", "fenced", "expired", "cancelled", "canceled"))


def _normalize_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return validate_compute_profile(value)
    except (GradientContractError, TypeError) as exc:
        raise GradientExecutorError(f"invalid compute profile: {exc}") from exc


async def _join_owned_task(task: asyncio.Task[Any]) -> Any:
    """Join a blocking thread even if its asyncio waiter has been cancelled."""
    cancellation_received = False
    while not task.done():
        try:
            await asyncio.wait({task}, timeout=0.025)
        except asyncio.CancelledError:
            cancellation_received = True
            continue
        except BaseException:
            break
    if task.done() and not task.cancelled():
        error = task.exception()
        if error is not None and not cancellation_received:
            raise error
    if cancellation_received:
        raise asyncio.CancelledError
    return task.result()


class GradientExecutorClient:
    """Claims work over HTTP; it never accepts inbound connections."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = dict(config)
        self._session_id = uuid.uuid4().hex
        self._work_session = False
        self._session_compute = None
        self._help_ready = False
        server_url = self.config.get("server_url")
        executor_id = self.config.get("executor_id")
        token = self.config.get("token")
        if not isinstance(server_url, str) or not server_url.startswith(("http://", "https://")):
            raise ValueError("server_url must be an HTTP(S) URL")
        if not isinstance(executor_id, str) or not executor_id:
            raise ValueError("executor_id must be nonempty")
        if not isinstance(token, str) or not token:
            raise ValueError("token must contain the resolved executor capability token")
        api_prefix = self.config.get("api_prefix")
        if api_prefix is None:
            trimmed = server_url.rstrip("/")
            api_prefix = "" if trimmed.endswith("/gradient/v1") else (
                "/v1" if trimmed.endswith("/gradient") else "/gradient/v1"
            )
            server_url = trimmed
        if not isinstance(api_prefix, str) or not api_prefix.startswith("/"):
            raise ValueError("api_prefix must begin with '/'")
        self.base_url = server_url.rstrip("/") + api_prefix.rstrip("/")
        host = urlsplit(server_url).hostname
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        self.contribution_compression = self.config.get("contribution_compression", not loopback)
        if type(self.contribution_compression) is not bool:
            raise ValueError("contribution_compression must be bool")
        self.contribution_transport = self.config.get("contribution_transport", "auto")
        if self.contribution_transport not in {"auto", "http"}:
            raise ValueError("contribution_transport must be auto or http")
        self.gradient_wire_dtype = self.config.get("gradient_wire_dtype", "float32")
        if not isinstance(self.gradient_wire_dtype, str) or self.gradient_wire_dtype not in {"float32", "bfloat16"}:
            raise ValueError("gradient_wire_dtype must be float32 or bfloat16")
        self.executor_id = executor_id
        self.token = token
        self.working_dir = Path(self.config.get("working_dir", ".gradient-executor")).expanduser().resolve()
        self.rowbundle_dir = self.working_dir / "rowbundles"
        shared_cache = self.config.get("policy_cache_dir")
        self._shared_policy_cache = shared_cache is not None
        if shared_cache is not None:
            root = Path(shared_cache).expanduser()
            if not root.is_absolute() or root.is_symlink():
                raise ValueError("policy_cache_dir must be an absolute non-symlink directory")
            origin = hashlib.sha256(self.base_url.encode()).hexdigest()[:24]
            self.policy_dir = root / origin
        else:
            self.policy_dir = self.working_dir / "policy"
        for directory in (self.rowbundle_dir, self.policy_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.max_payload_bytes = self.config.get("max_payload_bytes", MAX_TENSOR_RAW_BYTES)
        if type(self.max_payload_bytes) is not int or self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be a positive integer")
        self.poll_interval = float(self.config.get("poll_interval", 1.0))
        if not math.isfinite(self.poll_interval) or self.poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        self.retry_initial_seconds = float(self.config.get("retry_initial_seconds", 0.25))
        self.retry_max_seconds = float(self.config.get("retry_max_seconds", 5.0))
        if (not math.isfinite(self.retry_initial_seconds) or self.retry_initial_seconds <= 0
                or not math.isfinite(self.retry_max_seconds)
                or self.retry_max_seconds < self.retry_initial_seconds):
            raise ValueError("retry backoff bounds must be finite, positive, and ordered")
        self.heartbeat_interval = self.config.get("heartbeat_interval")
        if (self.heartbeat_interval is not None
                and (isinstance(self.heartbeat_interval, bool)
                     or not math.isfinite(float(self.heartbeat_interval))
                     or float(self.heartbeat_interval) <= 0)):
            raise ValueError("heartbeat_interval must be finite and positive")
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(connect=15.0, read=300.0, write=300.0, pool=15.0),
            follow_redirects=False,
            transport=transport,
        )
        self._closed = False
        self._installed_policy_version: int | None = None
        self._active_central_identity: tuple[str, str] | None = None

    def _headers(self, lease_id: str, lease_epoch: int) -> dict[str, str]:
        return {"X-Lease-ID": lease_id, "X-Lease-Epoch": str(lease_epoch)}

    def _job_url(self, job_id: str, suffix: str = "") -> str:
        return f"/executors/{self.executor_id}/jobs/{job_id}{suffix}"

    async def _download(self, url: str, destination: Path, *, headers: Mapping[str, str],
                        max_bytes: int, allow_missing: bool = False,
                        expected_response_headers: Mapping[str, str] | None = None,
                        tensor_payload: bool = False) -> bool:
        delay = self.retry_initial_seconds
        for attempt in range(3):
            try:
                return await self._download_once(
                    url, destination, headers=headers, max_bytes=max_bytes,
                    allow_missing=allow_missing,
                    expected_response_headers=expected_response_headers,
                    tensor_payload=tensor_payload,
                )
            except httpx.TransportError:
                if attempt == 2:
                    raise
            except httpx.HTTPStatusError as exc:
                if attempt == 2 or not (exc.response.status_code == 429
                                        or exc.response.status_code >= 500):
                    raise
            await asyncio.sleep(delay)
            delay = min(self.retry_max_seconds, delay * 2.0)
        raise AssertionError("download retry loop exhausted without a response")

    async def _download_once(self, url: str, destination: Path, *, headers: Mapping[str, str],
                             max_bytes: int, allow_missing: bool,
                             expected_response_headers: Mapping[str, str] | None,
                             tensor_payload: bool) -> bool:
        temporary: Path | None = None
        try:
            async with self._http.stream("GET", url, headers=dict(headers)) as response:
                if allow_missing and response.status_code == 404:
                    return False
                if response.status_code in {409, 410}:
                    # All downloads here are bound to a job lease. Streaming
                    # error bodies have not been read, so response.json() cannot
                    # be used by the generic retry classifier at this point.
                    raise GradientLeaseLost("scheduler fenced the leased download")
                response.raise_for_status()
                wire_limit = max_bytes
                if tensor_payload:
                    _, _, content_type, transport_name, max_encoded_size = tensor_codec()
                    wire_limit = max_encoded_size(max_bytes)
                    received_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    content_encoding = response.headers.get("content-encoding")
                    if (received_type != content_type
                            or response.headers.get(_TENSOR_TRANSPORT_HEADER.lower()) != transport_name
                            or (content_encoding is not None and content_encoding.strip().lower() != "identity")):
                        raise GradientExecutorError("policy tensor response omitted the required transport identity")
                for key, expected in (expected_response_headers or {}).items():
                    if response.headers.get(key.lower()) != expected:
                        raise GradientExecutorError(f"download {key} identity differs from the leased job")
                declared = response.headers.get("content-length")
                if declared is not None and int(declared) > wire_limit:
                    raise GradientExecutorError("download exceeds max_payload_bytes")
                destination.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temp_name = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".part",
                                                         dir=destination.parent)
                temporary = Path(temp_name)
                size = 0
                with os.fdopen(descriptor, "wb") as stream:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > wire_limit:
                            raise GradientExecutorError("download exceeds max_payload_bytes")
                        await _join_owned_task(asyncio.create_task(asyncio.to_thread(stream.write, chunk)))
                    if declared is not None and size != int(declared):
                        raise GradientExecutorError("download body length differs from Content-Length")
                    await _join_owned_task(asyncio.create_task(asyncio.to_thread(stream.flush)))
                    if not tensor_payload:
                        await _join_owned_task(asyncio.create_task(asyncio.to_thread(os.fsync, stream.fileno())))
                if tensor_payload:
                    # The packed wire frame is a disposable decode input. Only the
                    # decoded policy bytes below are installed as the durable cache.
                    packed = await _join_owned_task(asyncio.create_task(
                        asyncio.to_thread(temporary.read_bytes),
                    ))
                    _, decode, _, _, _ = tensor_codec()
                    try:
                        raw = await _join_owned_task(asyncio.create_task(
                            asyncio.to_thread(decode, packed, max_raw_bytes=max_bytes, safetensors=True),
                        ))
                    except ValueError as exc:
                        raise GradientExecutorError(
                            "policy tensor transport frame is invalid or exceeds max_payload_bytes"
                        ) from exc
                    if len(raw) > max_bytes:
                        raise GradientExecutorError("decoded policy tensor exceeds max_payload_bytes")
                    await _join_owned_task(asyncio.create_task(
                        asyncio.to_thread(_write_binary_fsync, temporary, raw),
                    ))
                os.replace(temporary, destination)
                temporary = None
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return True
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

    async def _claim(self, pending_upload=None) -> dict[str, Any] | None:
        options = {"help_ready": self._help_ready, "session_id": self._session_id}
        if pending_upload is not None:
            options['pending_upload'] = pending_upload
        response = await self._http.post(f"/executors/{self.executor_id}/claim",
                                         json=options)
        response.raise_for_status()
        value = response.json()
        if value.get("status") == "idle":
            return None
        if value.get("status") != "leased":
            raise GradientExecutorError("scheduler returned an unknown claim state")
        return value

    async def _heartbeat_loop(self, job_id: str, lease_id: str, lease_epoch: int,
                               interval: float, stopped: asyncio.Event, lost: asyncio.Event,
                               fatal: asyncio.Event) -> None:
        while not stopped.is_set():
            try:
                await asyncio.wait_for(stopped.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                response = await self._http.post(
                    self._job_url(job_id, "/heartbeat"),
                    headers=self._headers(lease_id, lease_epoch),
                )
                if response.status_code in {409, 410}:
                    lost.set()
                    return
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    fatal.set()
                    return
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    if exc.response.status_code in {409, 410}:
                        lost.set()
                    else:
                        fatal.set()
                    return
            except httpx.HTTPError:
                # A transient network error gets another interval; server expiry is
                # enforced by the lease epoch if connectivity does not recover.
                continue

    async def _local_rowbundle_path(self, claim, job, headers):
        local = claim.get("local_contribution_transport")
        if (not isinstance(local, dict) or local.get("supports_rowbundle_path") is not True
                or self._local_transport(claim) is None):
            return None
        namespace = local.get("namespace")
        response = await self._http.get(
            self._job_url(job.job_id, "/rowbundle/local"), headers=headers,
        )
        response.raise_for_status()
        descriptor = response.json()
        info = claim["rowbundle"]
        expected_size = info.get("size")
        if type(expected_size) is not int or expected_size < 0 or expected_size > self.max_payload_bytes:
            raise GradientExecutorError("leased rowbundle size exceeds max_payload_bytes")
        if (
            descriptor.get("namespace") != namespace
            or descriptor.get("job_id") != job.job_id
            or descriptor.get("lease_id") != claim.get("lease_id")
            or descriptor.get("lease_epoch") != claim.get("lease_epoch")
            or descriptor.get("semantic_hash") != info.get("semantic_hash")
            or type(descriptor.get("size")) is not int
            or descriptor["size"] != expected_size
        ):
            raise GradientExecutorError("local rowbundle descriptor differs from the leased claim")
        path_value = descriptor.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise GradientExecutorError("local rowbundle descriptor omitted its path")
        path = Path(path_value)
        if not path.is_absolute() or path.is_symlink():
            raise GradientExecutorError("local rowbundle path must be absolute and immutable")
        try:
            path_stat = path.lstat()
        except OSError as exc:
            raise GradientExecutorError("local rowbundle path is unavailable") from exc
        if not stat.S_ISREG(path_stat.st_mode):
            raise GradientExecutorError("local rowbundle path is not a regular file")
        if path_stat.st_size != expected_size:
            raise GradientExecutorError("local rowbundle size differs from its leased descriptor")
        return path

    def _validate_rowbundle_path(self, path, claim, job, *, group_size):
        info = claim["rowbundle"]
        semantic_hash = info["semantic_hash"]
        if path.stat().st_size != int(info["size"]):
            raise GradientExecutorError("downloaded rowbundle byte size differs from claim")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GradientExecutorError("downloaded rowbundle is invalid JSON") from exc
        if hashlib.sha256(_canonical_json(raw)).hexdigest() != semantic_hash:
            raise GradientExecutorError("downloaded rowbundle semantic identity differs from claim")
        expected_profile = _normalize_profile(claim["compute_profile"])
        if raw.get("compute_profile") != expected_profile:
            raise GradientExecutorError("rowbundle compute profile differs from the central claim")
        if type(raw.get("group_size")) is not int or raw["group_size"] != group_size:
            raise GradientExecutorError("rowbundle group_size differs from the central claim")
        expected = {
            "schema": "gradient_rowbundle/v1",
            "window_id": job.window_id,
            "attempt_id": job.attempt_id,
            "lineage_id": job.lineage_id,
            "engine_epoch": job.engine_epoch,
            "job_id": job.job_id,
            "group_id": job.group_id,
            "owner_executor_id": job.owner_executor_id,
            "source_worker_id": claim.get("source_worker_id"),
            "policy_version": job.policy_version,
            "n_window_tokens": job.n_window_tokens,
            "token_count": job.token_count,
            "layout_fingerprint": job.layout_fingerprint,
            "compute_profile_fingerprint": job.compute_profile_fingerprint,
            "group_size": group_size,
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise GradientExecutorError("rowbundle identity/count/layout differs from the leased job")
        rows = raw.get("rows")
        if (not isinstance(rows, list) or len(rows) not in {group_size, group_size + 1}
                or any(not isinstance(row, dict) for row in rows)
                or "_parallel_advantage" in raw or "advantages" in raw):
            raise GradientExecutorError("rowbundle must contain one complete group without central advantages")
        fresh_rows = [row for row in rows if row.get("row_kind", "fresh_student") == "fresh_student"]
        replay_rows = [row for row in rows if row.get("row_kind") in {"native_replay", "guided_replay"}]
        if (len(fresh_rows) != group_size or len(replay_rows) > 1
                or len(replay_rows) != len(rows) - group_size):
            raise GradientExecutorError(
                f"rowbundle must contain {group_size} fresh candidates and at most one replay"
            )
        if any("_parallel_advantage" in row or "advantages" in row for row in rows):
            raise GradientExecutorError("rowbundle rows must not contain centrally prepared advantages")
        if any(row.get("group_id") != job.group_id for row in rows):
            raise GradientExecutorError("rowbundle row group differs from its leased job")
        if any(row.get("source_worker_id") != claim["source_worker_id"] for row in fresh_rows):
            raise GradientExecutorError("fresh row Worker identity differs from its assigned executor")
        return rows

    async def _download_rowbundle(self, claim: dict[str, Any], job: GradientJob,
                                  headers: Mapping[str, str], *, group_size: int) -> list[dict[str, Any]]:
        info = claim["rowbundle"]
        semantic_hash = info["semantic_hash"]
        if not isinstance(semantic_hash, str) or len(semantic_hash) != 64:
            raise GradientExecutorError("claim has an invalid rowbundle semantic hash")
        local_path = await self._local_rowbundle_path(claim, job, headers)
        if local_path is not None:
            return await _join_owned_task(asyncio.create_task(asyncio.to_thread(
                self._validate_rowbundle_path, local_path, claim, job, group_size=group_size,
            )))

        path = self.rowbundle_dir / f"{semantic_hash}.json"
        if not path.is_file() or path.stat().st_size != int(info["size"]):
            await self._download(self._job_url(job.job_id, "/rowbundle"), path, headers=headers,
                                 max_bytes=self.max_payload_bytes,
                                 expected_response_headers={
                                     "X-Rowbundle-Semantic-Hash": semantic_hash,
                                     "X-Lineage-ID": job.lineage_id,
                                     "X-Engine-Epoch": job.engine_epoch,
                                     "X-Owner-Executor-ID": job.owner_executor_id,
                                     "X-Source-Worker-ID": claim["source_worker_id"],
                                 })
        try:
            return await _join_owned_task(asyncio.create_task(asyncio.to_thread(
                self._validate_rowbundle_path, path, claim, job, group_size=group_size,
            )))
        except BaseException:
            # Only the executor's downloaded cache is owned here. Local rowbundle
            # paths go through the branch above and are never removed by the client.
            path.unlink(missing_ok=True)
            raise

    async def _policy_state(self, claim: dict[str, Any], job: GradientJob,
                            lease_headers: Mapping[str, str]) -> tuple[Path, dict[str, Any] | None]:
        if self._local_transport(claim) is not None:
            response = await self._http.get(
                self._job_url(job.job_id, f"/policy/{job.policy_version}/local"), headers=lease_headers)
            response.raise_for_status()
            value = response.json()
            from qwen_rl.learner.gradient_ipc import namespace_identity
            if value["namespace"] != namespace_identity():
                raise GradientExecutorError("local policy namespace changed after claim")
            state_path = Path(value["state_path"])
            if not state_path.is_absolute() or state_path.is_symlink():
                raise GradientExecutorError("local policy requires an immutable absolute file")
            await _join_owned_task(asyncio.create_task(asyncio.to_thread(
                _validate_adapter_state, state_path, claim["prepared_window"],
            )))
            config = json.loads(Path(value["config_path"]).read_text()) if value["config_path"] else None
            return state_path, config
        from qwen_rl.learner.policy_cache import policy_lock
        root = self.policy_dir / job.lineage_id / job.engine_epoch
        async with policy_lock(root / f".{job.policy_version}.download.lock", exclusive=True):
            directory = root / str(job.policy_version)
            if directory.is_symlink():
                raise GradientExecutorError("policy cache version cannot be a symlink")
            return await self._policy_state_locked(claim, job, lease_headers)

    async def _policy_state_locked(self, claim: dict[str, Any], job: GradientJob,
                            lease_headers: Mapping[str, str]) -> tuple[Path, dict[str, Any] | None]:
        directory = self.policy_dir / job.lineage_id / job.engine_epoch / str(job.policy_version)
        state_path = directory / "adapter_state.safetensors"
        config_path = directory / "adapter_config.json"
        identity_headers = {**lease_headers}
        if not state_path.is_file():
            await self._download(
                self._job_url(job.job_id, f"/policy/{job.policy_version}/state"), state_path,
                headers=identity_headers, max_bytes=self.max_payload_bytes,
                tensor_payload=True,
                expected_response_headers={
                    "X-Policy-Version": str(job.policy_version),
                    "X-Layout-Fingerprint": job.layout_fingerprint,
                    "X-Lineage-ID": job.lineage_id,
                    "X-Engine-Epoch": job.engine_epoch,
                },
            )
        if not config_path.is_file():
            await self._download(
                self._job_url(job.job_id, f"/policy/{job.policy_version}/config"), config_path,
                headers=identity_headers, max_bytes=min(self.max_payload_bytes, 8 * 1024 * 1024),
                allow_missing=True,
                expected_response_headers={
                    "X-Policy-Version": str(job.policy_version),
                    "X-Layout-Fingerprint": job.layout_fingerprint,
                    "X-Lineage-ID": job.lineage_id,
                    "X-Engine-Epoch": job.engine_epoch,
                },
            )
        config = None
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                config_path.unlink(missing_ok=True)
                raise GradientExecutorError("canonical adapter_config.json is invalid") from exc
            if not isinstance(config, dict):
                raise GradientExecutorError("canonical adapter_config.json must be an object")
        await _join_owned_task(asyncio.create_task(asyncio.to_thread(
            _validate_adapter_state, state_path, claim["prepared_window"],
        )))
        return state_path, config

    async def _install_policy(
        self,
        compute: Any,
        version: int,
        state_path: Path,
        *,
        expected_names,
        phase_timing: JobPhaseTiming | None = None,
    ) -> None:
        if self._installed_policy_version == version:
            if phase_timing is not None:
                phase_timing.mark("adapter_cache_reuse")
            return
        from ..training_state import load_training_state

        async with measured_async(phase_timing, "adapter_read_cpu"):
            adapter_state = await _join_owned_task(asyncio.create_task(
                asyncio.to_thread(load_training_state, state_path, expected_names),
            ))
        install = getattr(compute, "install_adapter_version", None)
        if not callable(install):
            raise GradientExecutorError("compute factory must expose install_adapter_version(version, state)")
        async with measured_async(phase_timing, "adapter_cpu_stage"):
            await _join_owned_task(asyncio.create_task(asyncio.to_thread(install, version, adapter_state)))
        self._installed_policy_version = version

    async def _compute(self, compute: Any, window: PreparedWindow, job: GradientJob,
                       rows: Sequence[Mapping[str, Any]], compute_profile: Mapping[str, Any],
                       phase_timing: JobPhaseTiming | None = None) -> GradientContribution:
        run = getattr(compute, "compute_gradient_job", None)
        if not callable(run):
            raise GradientExecutorError("compute factory must expose compute_gradient_job(window,job,rows)")
        timed_run = getattr(compute, "compute_gradient_job_with_timing", None)
        if phase_timing is not None and callable(timed_run):
            result = await _join_owned_task(asyncio.create_task(asyncio.to_thread(
                timed_run, window, job, rows,
                compute_profile=compute_profile, phase_timing=phase_timing,
            )))
        else:
            async with measured_async(phase_timing, "gradient_job"):
                result = await _join_owned_task(asyncio.create_task(asyncio.to_thread(
                    run, window, job, rows, compute_profile=compute_profile,
                )))
        if not isinstance(result, GradientContribution):
            raise GradientExecutorError("compute_gradient_job must return GradientContribution")
        return result

    def _validate_contribution(self, contribution: GradientContribution, job: GradientJob,
                               window: PreparedWindow, profile: Mapping[str, Any]) -> None:
        expected_fields = {
            "schema": "gradient_contribution/v1",
            "aggregation": "global_token_mean_contribution/v1",
            "window_id": job.window_id,
            "attempt_id": job.attempt_id,
            "lineage_id": job.lineage_id,
            "engine_epoch": job.engine_epoch,
            "job_id": job.job_id,
            "group_id": job.group_id,
            "owner_executor_id": job.owner_executor_id,
            "policy_version": job.policy_version,
            "n_window_tokens": job.n_window_tokens,
            "token_count": job.token_count,
            "layout_fingerprint": job.layout_fingerprint,
            "compute_profile_fingerprint": job.compute_profile_fingerprint,
        }
        for field, expected in expected_fields.items():
            observed = getattr(contribution, field, None)
            if observed != expected:
                raise GradientExecutorError(f"computed contribution {field} differs from its leased job")
        if getattr(contribution, "compute_profile", profile) != profile:
            raise GradientExecutorError("computed contribution compute profile differs from the lease")
        layout = {parameter.name: tuple(parameter.shape) for parameter in window.parameter_layout}
        gradients = contribution.gradients
        if set(gradients) != set(layout):
            raise GradientExecutorError("computed gradient names differ from the complete parameter layout")
        import torch
        for name, tensor in gradients.items():
            if tensor is None:
                continue
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32 or tuple(tensor.shape) != layout[name]):
                raise GradientExecutorError(f"computed gradient {name} violates CPU float32 shape contract")

    async def _serialize_contribution(
        self,
        contribution: GradientContribution,
        executor_id: str,
        lease_id: str,
        lease_epoch: int,
    ) -> bytes:
        from safetensors.torch import save

        raw_metadata = contribution.to_metadata_json()
        contribution_metadata = json.loads(raw_metadata)
        contribution_metadata["present_parameters"] = sorted(
            name for name, tensor in contribution.gradients.items() if tensor is not None
        )
        wrapped = {
            "contribution": contribution_metadata,
            "executor_id": executor_id,
            "lease_id": lease_id,
            "lease_epoch": lease_epoch,
        }
        file_metadata = {
            "format": "pt",
            "gradient_contribution": json.dumps(wrapped, sort_keys=True, separators=(",", ":"), allow_nan=False),
        }
        import torch
        dtype = torch.bfloat16 if self.gradient_wire_dtype == "bfloat16" else torch.float32
        def serialize():
            tensors = {name: tensor.to(dtype=dtype).contiguous()
                       for name, tensor in contribution.gradients.items() if tensor is not None}
            return save(tensors, metadata=file_metadata)
        # Keep raw SafeTensors bytes in memory; the encoded retry frame also remains process-local.
        raw = await _join_owned_task(asyncio.create_task(asyncio.to_thread(
            serialize,
        )))
        if len(raw) > self.max_payload_bytes:
            raise GradientExecutorError("serialized contribution exceeds max_payload_bytes")
        return raw

    async def _encode_contribution(self, raw: bytes) -> bytes:
        size = len(raw)
        if size > self.max_payload_bytes:
            raise GradientExecutorError("serialized contribution exceeds max_payload_bytes")
        encode, _, _, _, max_encoded_size = tensor_codec()
        try:
            packed = await _join_owned_task(asyncio.create_task(
                asyncio.to_thread(encode, raw, safetensors=True,
                                  compression=self.contribution_compression),
            ))
        except ValueError as exc:
            raise GradientExecutorError("serialized contribution could not be losslessly framed") from exc
        if len(packed) > max_encoded_size(size):
            raise GradientExecutorError("compressed contribution exceeds max_payload_bytes")
        # Retries own the same immutable frame. After a process restart the
        # scheduler fences/reclaims the job; only the central receipt commits it.
        return packed

    def _prune_policy_cache(self, lineage_id: str, engine_epoch: str) -> None:
        lineage_root = self.policy_dir / lineage_id / engine_epoch
        if not lineage_root.is_dir() or lineage_root.is_symlink():
            return
        versions = sorted(
            (int(path.name), path)
            for path in lineage_root.iterdir()
            if path.is_dir() and not path.is_symlink() and re.fullmatch(r"[0-9]+", path.name)
        )
        for version, path in versions[:-2]:
            if self._shared_policy_cache:
                import fcntl
                from qwen_rl.learner.policy_cache import open_lock
                fd = open_lock(lineage_root / f".{version}.use.lock")
                try:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                finally:
                    os.close(fd)
            else:
                shutil.rmtree(path)

    async def _stream_upload(self, packed: bytes, url: str, headers: Mapping[str, str]) -> httpx.Response:
        size = len(packed)
        _, _, content_type, transport_name, _ = tensor_codec()

        async def chunks():
            for offset in range(0, size, 1024 * 1024):
                yield packed[offset:offset + 1024 * 1024]

        delay = self.retry_initial_seconds
        for attempt in range(3):
            request = self._http.build_request(
                "PUT", url,
                headers={**headers, "Content-Type": content_type,
                         _TENSOR_TRANSPORT_HEADER: transport_name,
                         "Content-Length": str(size)},
                content=chunks(),
            )
            try:
                response = await self._http.send(request)
            except httpx.TransportError:
                if attempt == 2:
                    raise
            else:
                if response.status_code not in {429} and response.status_code < 500:
                    return response
                if attempt == 2:
                    return response
                await response.aclose()
            await asyncio.sleep(delay)
            delay = min(self.retry_max_seconds, delay * 2.0)
        raise AssertionError("upload retry loop exhausted without a response")

    def _local_transport(self, claim):
        descriptor = claim.get("local_contribution_transport")
        if self.contribution_transport == "http" or not isinstance(descriptor, dict):
            return None
        from qwen_rl.learner.gradient_ipc import namespace_identity
        # Abstract Unix sockets and FDs are usable only in the exact same host
        # namespace. A remote peer retains the explicit network transport.
        if descriptor.get("namespace") != namespace_identity():
            return None
        return descriptor["address"]

    async def _upload_shared(self, address, job, lease_id, lease_epoch, payload):
        from qwen_rl.learner.gradient_ipc import send_contribution
        delay = self.retry_initial_seconds
        for attempt in range(3):
            try:
                result = await _join_owned_task(asyncio.create_task(asyncio.to_thread(
                    send_contribution, address, self.token, self.executor_id,
                    job.job_id, lease_id, lease_epoch, payload,
                )))
            except (OSError, ConnectionError) as exc:
                if attempt == 2:
                    raise httpx.TransportError("local gradient connection failed") from exc
            else:
                response = httpx.Response(result["status_code"], json=result,
                                          request=httpx.Request("PUT", self._job_url(job.job_id, "/contribution")))
                if response.status_code not in {429, 503} or attempt == 2:
                    if response.status_code in {409, 410}:
                        raise GradientLeaseLost("server fenced this shared contribution lease")
                    response.raise_for_status()
                    break
            await asyncio.sleep(delay)
            delay = min(self.retry_max_seconds, delay * 2)
        # Keep optimizer application on the existing ASGI event-loop owner. The
        # data socket never runs Adam or bypasses the normal window/session fence.
        response = await self._http.post(self._job_url(job.job_id, "/contribution/complete"),
                                         headers=self._headers(lease_id, lease_epoch))
        response.raise_for_status()
        return response

    async def run_once(self, compute: Any, *, _upload_slot=None, _pending_upload=None) -> bool:
        """Claim and complete at most one job; return False when the queue is empty."""
        phase_timing = JobPhaseTiming()
        can_help = getattr(compute, "can_help", None)
        self._help_ready = (bool(await asyncio.to_thread(can_help))
                            if self.config.get("local_work_stealing", False) and callable(can_help) else False)
        async with measured_async(phase_timing, "lease_claim_wait"):
            claim = await self._claim(_pending_upload)
        if claim is None:
            if _upload_slot is None:
                await self.finish_session()
            return False
        raw_job = claim.get("job")
        raw_job = raw_job if isinstance(raw_job, Mapping) else {}
        phase_identity = {
            "executor_id": claim.get("executor_id"),
            "job_id": raw_job.get("job_id"),
            "window_id": raw_job.get("window_id"),
            "attempt_id": raw_job.get("attempt_id"),
            "lineage_id": raw_job.get("lineage_id", claim.get("lineage_id")),
            "engine_epoch": raw_job.get("engine_epoch", claim.get("engine_epoch")),
            "group_id": raw_job.get("group_id"),
            "policy_version": raw_job.get("policy_version"),
            "lease_epoch": claim.get("lease_epoch"),
        }
        phase_status = "failed"
        phase_error_type: str | None = None
        executor_id = claim["executor_id"]
        lease_id = claim["lease_id"]
        lease_epoch = claim["lease_epoch"]
        lease_seconds = float(claim["lease_seconds"])
        interval = min(float(self.heartbeat_interval or lease_seconds / 3.0), lease_seconds / 3.0)
        stopped, lost, fatal = asyncio.Event(), asyncio.Event(), asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat_loop(
            claim["job"]["job_id"], lease_id, lease_epoch, interval, stopped, lost, fatal,
        ))
        policy_lease = None
        packed_contribution: bytes | None = None
        shared_contribution = None
        accepted = False
        fenced = False
        rowbundle_hash = claim.get("rowbundle", {}).get("semantic_hash")
        rowbundle_path = (
            self.rowbundle_dir / f"{rowbundle_hash}.json"
            if isinstance(rowbundle_hash, str) and re.fullmatch(r"[0-9a-f]{64}", rowbundle_hash)
            else None
        )
        try:
            job = GradientJob.from_json(json.dumps(claim["job"], sort_keys=True, separators=(",", ":")))
            window = PreparedWindow.from_json(
                json.dumps(claim["prepared_window"], sort_keys=True, separators=(",", ":"))
            )
            profile = _normalize_profile(claim["compute_profile"])
            if profile != _normalize_profile(window.compute_profile):
                raise GradientExecutorError("claim compute profile differs from PreparedWindow")
            if _normalize_profile(job.compute_profile) != profile:
                raise GradientExecutorError("claim compute profile differs from GradientJob")
            if (job.lineage_id != claim.get("lineage_id") or job.engine_epoch != claim.get("engine_epoch")
                    or window.lineage_id != job.lineage_id or window.engine_epoch != job.engine_epoch):
                raise GradientExecutorError("claim central lineage/engine epoch differs from the immutable job")
            central_identity = (job.lineage_id, job.engine_epoch)
            if (self._active_central_identity is not None
                    and self._active_central_identity != central_identity):
                raise GradientExecutorError("executor daemon cannot switch central lineage/engine epoch")
            self._active_central_identity = central_identity
            if not any(item.job_id == job.job_id and item.to_json() == job.to_json() for item in window.jobs):
                raise GradientExecutorError("claimed job is not the exact job bound in PreparedWindow")
            actual_profile = _normalize_profile(compute.compute_profile)
            if actual_profile != profile:
                raise GradientExecutorError("executor engine base/loss/correction profile differs from the job")
            if self.executor_id != executor_id:
                raise GradientExecutorError("scheduler leased work to a different executor identity")
            partitioned = "trajectory_partitions" in (window.update_context or {})
            if job.owner_executor_id != self.executor_id and not partitioned:
                raise GradientExecutorError("scheduler leased an unpartitioned job outside its source owner")
            if partitioned:
                self._work_session = True
                self._session_compute = compute
            if (not isinstance(claim.get("source_worker_id"), str)
                    or not claim["source_worker_id"]):
                raise GradientExecutorError("claim omitted the owner executor's Worker identity")
            group_size = claim.get("group_size")
            if type(group_size) is not int or group_size < 2:
                raise GradientExecutorError("claim omitted the centrally configured group_size")
            compute_config = self.config.get("compute_config")
            executor_group_size = (
                compute_config.get("group_size") if isinstance(compute_config, Mapping) else None
            )
            if executor_group_size is not None and (
                    type(executor_group_size) is not int or executor_group_size != group_size):
                raise GradientExecutorError(
                    "centrally configured group_size differs from the executor compute profile"
                )
            context_group_size = (window.update_context or {}).get("group_size")
            if context_group_size is not None and (
                    type(context_group_size) is not int or context_group_size != group_size):
                raise GradientExecutorError("claim group_size differs from the prepared window configuration")
            if self._shared_policy_cache:
                from qwen_rl.learner.policy_cache import policy_lock
                root = self.policy_dir / job.lineage_id / job.engine_epoch
                pending_lease = policy_lock(root / f".{job.policy_version}.use.lock")
                await pending_lease.__aenter__()
                policy_lease = pending_lease
            lease_headers = self._headers(lease_id, lease_epoch)
            async with measured_async(phase_timing, "adapter_download_validate"):
                state_path, _ = await self._policy_state(claim, job, lease_headers)
            await self._install_policy(
                compute, job.policy_version, state_path,
                expected_names=tuple(p.name for p in window.parameter_layout),
                phase_timing=phase_timing,
            )
            async with measured_async(phase_timing, "rowbundle_download_validate"):
                rows = await self._download_rowbundle(
                    claim, job, lease_headers, group_size=group_size,
                )
            if fatal.is_set():
                raise GradientExecutorError("executor heartbeat was rejected by the scheduler")
            if lost.is_set():
                raise GradientLeaseLost("lease was fenced before compute started")
            if partitioned:
                from qwen_rl.contracts.gradient_partitions import selected_rows
                selected_rows(window, job, rows)
                start_session = getattr(compute, "start_session", None)
                if callable(start_session):
                    async with measured_async(phase_timing, "compute_session_start"):
                        await _join_owned_task(asyncio.create_task(asyncio.to_thread(start_session)))
            if fatal.is_set():
                raise GradientExecutorError("executor heartbeat was rejected during session preparation")
            if lost.is_set():
                raise GradientLeaseLost("lease was fenced during session preparation; compute skipped")
            contribution = await self._compute(
                compute, window, job, rows, profile, phase_timing=phase_timing,
            )
            local_address = self._local_transport(claim)
            if local_address is not None:
                from qwen_rl.learner.gradient_ipc import SharedGradientPayload
                self._validate_contribution(contribution, job, window, profile)
                async with measured_async(phase_timing, "contribution_shared_pack"):
                    shared_contribution = await _join_owned_task(asyncio.create_task(asyncio.to_thread(
                        SharedGradientPayload.pack, contribution,
                    )))
                del contribution
            if _upload_slot is not None and partitioned and shared_contribution is None:
                from dataclasses import replace

                def snapshot():
                    gradients = {}
                    for name, value in contribution.gradients.items():
                        if value is not None and value.device.type != 'cpu':
                            raise GradientExecutorError('upload pipeline requires CPU-owned gradients')
                        gradients[name] = None if value is None else value.detach().clone()
                    return replace(contribution, gradients=gradients)

                async with measured_async(phase_timing, 'contribution_cpu_snapshot'):
                    contribution = await _join_owned_task(asyncio.create_task(asyncio.to_thread(snapshot)))
                async with measured_async(phase_timing, 'transport_backpressure'):
                    await _upload_slot.handoff(claim)
            if shared_contribution is not None:
                if _upload_slot is not None and partitioned:
                    async with measured_async(phase_timing, "transport_backpressure"):
                        await _upload_slot.handoff(claim)
                if fatal.is_set() or lost.is_set():
                    raise GradientLeaseLost("lease fenced before shared gradient handoff")
                async with measured_async(phase_timing, "contribution_shared_upload"):
                    response = await self._upload_shared(local_address, job, lease_id, lease_epoch,
                                                         shared_contribution)
            else:
                async with measured_async(phase_timing, "contribution_validate"):
                    await _join_owned_task(asyncio.create_task(asyncio.to_thread(
                        self._validate_contribution, contribution, job, window, profile,
                    )))
                if fatal.is_set():
                    raise GradientExecutorError("executor heartbeat was rejected during compute")
                if lost.is_set():
                    raise GradientLeaseLost("lease was fenced during compute; contribution discarded")
                async with measured_async(phase_timing, "contribution_serialize"):
                    raw_contribution = await self._serialize_contribution(
                        contribution, executor_id, lease_id, lease_epoch,
                    )
                del contribution
                async with measured_async(phase_timing, "contribution_encode"):
                    packed_contribution = await self._encode_contribution(raw_contribution)
                del raw_contribution
                if fatal.is_set():
                    raise GradientExecutorError("executor heartbeat was rejected while serializing")
                if lost.is_set():
                    raise GradientLeaseLost("lease was fenced while serializing; contribution discarded")
                async with measured_async(phase_timing, "contribution_upload"):
                    response = await self._stream_upload(
                        packed_contribution, self._job_url(job.job_id, "/contribution"), lease_headers,
                    )
                    if response.status_code in {409, 410}:
                        raise GradientLeaseLost("server fenced this contribution lease")
                    response.raise_for_status()
            async with measured_async(phase_timing, "receipt_wait"):
                receipt_response = await self._http.get(
                    self._job_url(job.job_id, "/receipt"), headers=lease_headers,
                )
                if receipt_response.status_code == 202:
                    raise GradientExecutorError("server accepted upload without a durable contribution receipt")
                receipt_response.raise_for_status()
                payload = receipt_response.json()
                if payload.get("job_receipt", {}).get("status") != "accepted":
                    raise GradientExecutorError("server returned no accepted receipt")
            accepted = True
            phase_status = "accepted"
            return True
        except GradientLeaseLost as exc:
            fenced = True
            phase_status = "fenced"
            phase_error_type = type(exc).__name__
            raise
        except httpx.HTTPStatusError as exc:
            fenced = _retryable_executor_failure(exc) and exc.response.status_code in {409, 410}
            phase_status = "fenced" if fenced else "failed"
            phase_error_type = type(exc).__name__
            raise
        except asyncio.CancelledError as exc:
            # Blocking compute/install operations are joined by _join_owned_task before
            # propagating cancellation; the lease heartbeat remains active until this
            # finally block runs.
            phase_status = "cancelled"
            phase_error_type = type(exc).__name__
            raise
        except BaseException as exc:
            phase_status = "failed"
            phase_error_type = type(exc).__name__
            raise
        finally:
            try:
                async with measured_async(phase_timing, "executor_cleanup"):
                    session_error = None
                    if self._work_session and not accepted and _upload_slot is None:
                        try:
                            await self.finish_session()
                        except BaseException as exc:
                            session_error = exc
                    stopped.set()
                    try:
                        await asyncio.shield(heartbeat)
                    except asyncio.CancelledError:
                        while not heartbeat.done():
                            try:
                                await asyncio.shield(heartbeat)
                            except asyncio.CancelledError:
                                continue
                    if session_error is not None:
                        raise session_error
                    packed_contribution = None
                    if shared_contribution is not None:
                        shared_contribution.close()
                    if rowbundle_path is not None and (accepted or fenced or lost.is_set()):
                        rowbundle_path.unlink(missing_ok=True)
                    if accepted or fenced or lost.is_set():
                        self._prune_policy_cache(claim["lineage_id"], claim["engine_epoch"])
            except BaseException as exc:
                if phase_status != "accepted":
                    phase_status = "failed"
                phase_error_type = type(exc).__name__
                raise
            finally:
                if policy_lease is not None:
                    await policy_lease.__aexit__(None, None, None)
                if self._shared_policy_cache and self._active_central_identity is not None:
                    self._prune_policy_cache(*self._active_central_identity)
                if phase_identity.get("job_id"):
                    try:
                        await _join_owned_task(asyncio.create_task(asyncio.to_thread(
                            phase_timing.write_receipt,
                            self.working_dir,
                            phase_identity,
                            status=phase_status,
                            error_type=phase_error_type,
                        )))
                    except Exception as timing_error:
                        _LOGGER.warning(
                            "gradient timing_error while writing local receipt: %s",
                            type(timing_error).__name__,
                        )

    async def run(self, compute: Any, *, stop_event: asyncio.Event | None = None,
                  max_jobs: int | None = None) -> int:
        if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 0):
            raise ValueError("max_jobs must be a non-negative integer")
        completed = 0
        stop = stop_event or asyncio.Event()
        pipeline = self.config.get('upload_pipeline', True)
        if type(pipeline) is not bool:
            raise ValueError('upload_pipeline must be boolean')
        if pipeline:
            from .upload_pipeline import run_upload_pipeline

            return await run_upload_pipeline(self, compute, stop=stop, max_jobs=max_jobs,
                                             retryable=_retryable_executor_failure)
        retry_delay = self.retry_initial_seconds
        try:
            while not stop.is_set() and (max_jobs is None or completed < max_jobs):
                try:
                    worked = await self.run_once(compute)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not _retryable_executor_failure(exc):
                        raise
                    _LOGGER.warning("executor %s retrying fenced/transient work: %s",
                                    self.executor_id, type(exc).__name__)
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=retry_delay)
                    except asyncio.TimeoutError:
                        pass
                    retry_delay = min(self.retry_max_seconds, retry_delay * 2.0)
                    continue
                if worked:
                    completed += 1
                    retry_delay = self.retry_initial_seconds
                    continue
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self.finish_session()
        return completed

    async def finish_session(self) -> None:
        if not self._work_session:
            return

        async def finish_and_release():
            loop = asyncio.get_running_loop()

            async def attest(action, expected):
                # The exact-session transition is idempotent, including when
                # SQLite committed but the response was lost. Identity/lease
                # refusals still fail immediately and keep the Worker fenced.
                for attempt in range(3):
                    try:
                        response = await self._http.post(
                            f"/executors/{self.executor_id}/session/{action}", json={"session_id": self._session_id},
                        )
                        response.raise_for_status()
                    except (httpx.TransportError, httpx.HTTPStatusError) as error:
                        if attempt == 2 or not _retryable_executor_failure(error):
                            raise
                        await asyncio.sleep(min(self.retry_max_seconds, self.retry_initial_seconds * 2**attempt))
                        continue
                    if response.json() != {"status": expected}:
                        raise RuntimeError(f"central server did not acknowledge compute session {action}")
                    return

            def on_parked():
                asyncio.run_coroutine_threadsafe(attest("parked", "parked"), loop).result()

            end_with_callback = getattr(self._session_compute, "end_session_with_parked_callback", None)
            if callable(end_with_callback):
                await asyncio.to_thread(end_with_callback, on_parked)
            else:
                end_session = getattr(self._session_compute, "end_session", None)
                if callable(end_session):
                    await asyncio.to_thread(end_session)
            await attest("release", "released")
            self._work_session = False
            self._session_compute = None
            self._session_id = uuid.uuid4().hex

        # Cancellation must join both the ownership transition and its final
        # release receipt; otherwise an orphaned active session can block sync.
        await _join_owned_task(asyncio.create_task(finish_and_release()))

    async def close(self) -> None:
        if self._closed:
            return
        await self.finish_session()
        self._closed = True
        await self._http.aclose()

    async def __aenter__(self) -> "GradientExecutorClient":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()


def _validate_adapter_state(path: Path, prepared_window: Mapping[str, Any]) -> None:
    try:
        from ..training_state import validate_training_state
        layout = {entry['name']:tuple(entry['shape']) for entry in prepared_window['parameter_layout']}
        validate_training_state(path, layout)
    except Exception as exc:
        raise GradientExecutorError(f"canonical policy safetensors validation failed: {exc}") from exc


async def run_executor(config: Mapping[str, Any], compute_factory: Callable[[Mapping[str, Any]], Any],
                       *, transport: httpx.AsyncBaseTransport | None = None,
                       stop_event: asyncio.Event | None = None, max_jobs: int | None = None) -> int:
    """Build the compute owner once and keep polling until stopped or capped in a test."""
    if not callable(compute_factory):
        raise TypeError("compute_factory must be callable")
    client = GradientExecutorClient(config, transport=transport)
    compute = None
    try:
        # First import of the native codec can be slow. Do it before holding a
        # sixty-second lease rather than blocking its heartbeat on cold import.
        await _join_owned_task(asyncio.create_task(asyncio.to_thread(tensor_codec)))
        compute = compute_factory(config)
        if asyncio.iscoroutine(compute):
            compute = await compute
        prepare_model = getattr(compute, "prepare_cpu_model", None)
        if callable(prepare_model):
            started = time.monotonic()
            await _join_owned_task(asyncio.create_task(asyncio.to_thread(prepare_model)))
            _LOGGER.info("executor %s prepared the packed T2 model in host RAM in %.2f seconds",
                         client.executor_id,
                         time.monotonic() - started)
        return await client.run(compute, stop_event=stop_event, max_jobs=max_jobs)
    finally:
        # A compute factory may own a thread/device lifecycle. It is closed only after
        # `run` has joined every active run_once and blocking compute operation.
        close = getattr(compute, "close", None) or getattr(compute, "shutdown", None)
        try:
            if callable(close):
                await _join_owned_task(asyncio.create_task(asyncio.to_thread(close)))
        finally:
            await client.close()
