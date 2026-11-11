"""Bounded versioned synchronization of a published policy to one Worker."""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import httpx

from .errors import WorkerUnavailableError


_RETRYABLE_CONFLICT_CODES = frozenset({"worker_busy", "adapter_capacity_busy"})
_MAX_RETRY_WINDOW_SECONDS = 5.0
_RETRY_DELAY_SECONDS = 0.1
_sleep = asyncio.sleep


class WorkerPolicySyncError(RuntimeError):
    """A Worker rejected a policy sync or returned an invalid acknowledgement."""


def _worker_error_code(response: httpx.Response) -> str | None:
    """Return only the structured Worker error code, never its free-form message."""
    try:
        body = response.json()
    except Exception:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and code else None


def _http_failure(worker_id: str, response: httpx.Response, code: str | None) -> WorkerPolicySyncError:
    code_text = code if code is not None else "error_code_unavailable"
    return WorkerPolicySyncError(
        f"worker {worker_id} policy sync returned HTTP {response.status_code} ({code_text})"
    )


async def sync_worker_policy(
    client: httpx.AsyncClient,
    *,
    worker_id: str,
    worker_url: str,
    version: int,
    expected_files: dict[str, dict[str, Any]],
    expected_adapter_hash: str,
    request_timeout: float,
) -> dict[str, Any]:
    """Install one version, retrying only short-lived Worker admission conflicts.

    The first HTTP request uses the configured request timeout. After a retryable
    409, retries share a deadline capped at five seconds (and at the configured
    timeout). Each retry is bounded by the remaining deadline. Caller cancellation
    propagates through the sleep and HTTP request.
    """
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("version must be a nonnegative integer")
    if (isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float))
            or not math.isfinite(request_timeout) or request_timeout <= 0):
        raise ValueError("request_timeout must be positive and finite")
    if not isinstance(expected_adapter_hash, str) or not expected_adapter_hash:
        raise ValueError("expected_adapter_hash must be a non-empty string")

    url = f"{worker_url.rstrip('/')}/policy"
    payload = {"version": version}
    sync_started = time.perf_counter()
    attempt_count = 1

    async def post(timeout: float, *, prior_conflict_code: str | None = None) -> httpx.Response:
        try:
            return await client.post(url, json=payload, timeout=timeout)
        except httpx.HTTPError as exc:
            prior_conflict = (
                f" after retryable HTTP 409 ({prior_conflict_code})"
                if prior_conflict_code is not None else ""
            )
            raise WorkerUnavailableError(
                f"worker {worker_id} policy sync request failed{prior_conflict} "
                f"({type(exc).__name__})", worker_id=worker_id,
            ) from exc

    response = await post(request_timeout)
    code = _worker_error_code(response)
    retryable = response.status_code == 409 and code in _RETRYABLE_CONFLICT_CODES
    if retryable:
        deadline = time.monotonic() + min(request_timeout, _MAX_RETRY_WINDOW_SECONDS)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerPolicySyncError(
                    f"worker {worker_id} policy sync returned HTTP 409 "
                    f"({code}; retry window expired)"
                )
            await _sleep(min(_RETRY_DELAY_SECONDS, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerPolicySyncError(
                    f"worker {worker_id} policy sync returned HTTP 409 "
                    f"({code}; retry window expired)"
                )
            try:
                async with asyncio.timeout(remaining):
                    attempt_count += 1
                    response = await post(
                        min(request_timeout, remaining), prior_conflict_code=code,
                    )
            except TimeoutError as exc:
                raise WorkerPolicySyncError(
                    f"worker {worker_id} policy sync returned HTTP 409 "
                    f"({code}; retry window expired)"
                ) from exc
            code = _worker_error_code(response)
            if response.status_code == 409 and code in _RETRYABLE_CONFLICT_CODES:
                continue
            break

    if response.status_code != 200:
        if response.status_code == 500 and code == "adapter_load_failed":
            raise WorkerUnavailableError(
                f"worker {worker_id} policy sync returned HTTP 500 (adapter_load_failed)",
                worker_id=worker_id,
            )
        raise _http_failure(worker_id, response, code)
    try:
        acknowledgement = response.json()
    except Exception as exc:
        raise WorkerPolicySyncError(
            f"worker {worker_id} returned invalid JSON for policy version {version} ACK"
        ) from exc
    if not isinstance(acknowledgement, dict):
        raise WorkerPolicySyncError(
            f"worker {worker_id} returned an invalid policy version {version} ACK"
        )
    if acknowledgement.get("worker_id") != worker_id:
        raise WorkerPolicySyncError(
            f"worker {worker_id} ACKed policy version {version} under a different worker id"
        )
    acknowledged_version = acknowledgement.get("version")
    if (isinstance(acknowledged_version, bool) or not isinstance(acknowledged_version, int)
            or acknowledged_version != version):
        raise WorkerPolicySyncError(
            f"worker {worker_id} did not ACK policy version {version}"
        )
    if acknowledgement.get("files") != expected_files:
        raise WorkerPolicySyncError(
            f"worker {worker_id} policy version {version} file metadata ACK mismatch"
        )
    if acknowledgement.get("adapter_hash") != expected_adapter_hash:
        raise WorkerPolicySyncError(
            f"worker {worker_id} policy version {version} adapter identity ACK mismatch"
        )
    # Timing is observability only and never replaces the Worker/version/file-metadata/adapter-identity acknowledgement.
    raw_timing = acknowledgement.get("sync_timing")
    timing: dict[str, Any] = {}
    if isinstance(raw_timing, dict):
        for name, value in raw_timing.items():
            if not isinstance(name, str) or not name:
                continue
            if (name.endswith("_seconds") and not isinstance(value, bool)
                    and isinstance(value, (int, float))):
                try:
                    seconds = float(value)
                except (OverflowError, ValueError):
                    continue
                if math.isfinite(seconds) and seconds >= 0:
                    timing[name] = seconds
            elif type(value) is bool:
                timing[name] = value
            elif type(value) is int and value >= 0:
                timing[name] = value
    timing["sync_client_elapsed_seconds"] = max(0.0, time.perf_counter() - sync_started)
    timing["sync_client_attempt_count"] = attempt_count
    acknowledgement["sync_timing"] = timing
    return acknowledgement
