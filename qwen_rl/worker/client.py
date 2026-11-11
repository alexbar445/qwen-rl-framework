"""HTTP client implementing the subset expected by verl's ArcAgentLoop."""
from __future__ import annotations

import asyncio
import math
import os
import re
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import ValidationError
from qwen_rl.contracts import worker as wire
from .errors import WorkerUnavailableError

if TYPE_CHECKING:
    from verl.workers.rollout.replica import TokenOutput


_EVALUATION_ADMISSION_RETRY_CODES = frozenset({"worker_busy", "adapter_capacity_busy", "worker_draining"})
_EVALUATION_ADMISSION_RETRY_MAX_SECONDS = 5.0
_EVALUATION_ADMISSION_RETRY_DELAY_SECONDS = 0.1


class GenerationContractError(RuntimeError):
    """Malformed sampling evidence is fatal, rather than an infrastructure retry."""


# Rejected requests on a restarted/unavailable Worker contain no sample to trust.
_UNAVAILABLE_CODES = frozenset({
    "worker_epoch_changed", "worker_identity_unavailable", "worker_not_found",
    "worker_not_model_ready", "worker_not_eligible", "gateway_rejoin_required",
    "worker_result_indeterminate", "generation_indeterminate",
})
_CONTRACT_CODES = frozenset({
    "score_alignment_error", "invalid_log_probability", "policy_version_mismatch",
    "episode_lease_missing", "episode_lease_mismatch", "adapter_hash_mismatch",
    "policy_version_unavailable", "upstream_worker_identity_mismatch",
})


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _safe_sampling_params(params: dict[str, Any]) -> dict[str, Any]:
    try:
        _, normalized = wire.normalize_sampling_params(
            params, limit=wire.MAX_GENERATED_TOKENS,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return normalized


def _safe_request_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= wire.MAX_REQUEST_SEED:
        raise ValueError(f"seed must be an integer between 0 and {wire.MAX_REQUEST_SEED}")
    return seed


class HTTPGenerationManager:
    """Small async server-manager facade with a fixed episode policy version."""

    def __init__(
        self,
        worker_url: str,
        policy_version: int | None = None,
        *,
        scope: str = "local",
        url_prefix: str = "",
        auth_token_env: str = "ARC_ENV_TOKEN",
        worker_id: str | None = None,
        node_id: str | None = None,
        expected_worker_boot_id: str | None = None,
        expected_worker_telemetry_epoch: str | None = None,
        timeout_seconds: float = 180.0,
        http_transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not worker_url.startswith(("http://", "https://")):
            raise ValueError("worker_url must be an absolute HTTP(S) URL")
        if policy_version is not None and (
            isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0
        ):
            raise ValueError("policy_version must be a nonnegative integer")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self.worker_url = worker_url.rstrip("/")
        if url_prefix and (not url_prefix.startswith("/") or url_prefix.endswith("/")):
            raise ValueError("url_prefix must be an absolute path without a trailing slash")
        if not isinstance(auth_token_env, str) or not auth_token_env:
            raise ValueError("auth_token_env must be a non-empty environment variable name")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", auth_token_env) is None:
            raise ValueError("auth_token_env must be a valid environment variable name")
        if not isinstance(scope, str) or scope not in {"local", "service", "gateway"}:
            raise ValueError("worker scope must be local, service, or gateway")
        if scope == "gateway" and not url_prefix:
            raise ValueError("Gateway Worker calls require a route prefix")
        if scope != "gateway" and url_prefix:
            raise ValueError("local and service Worker calls cannot use a Gateway route prefix")
        if scope in {"service", "gateway"} and (not isinstance(worker_id, str) or not worker_id):
            raise ValueError(f"{scope.title()} Worker calls require a Worker ID")
        if scope == "service" and (
            not isinstance(node_id, str) or not node_id.strip() or any(char.isspace() for char in node_id)
        ):
            raise ValueError("service Worker calls require the physical node ID")
        if expected_worker_boot_id is not None and (
            not isinstance(expected_worker_boot_id, str) or not expected_worker_boot_id
        ):
            raise ValueError("expected Worker boot ID must be a non-empty string")
        if expected_worker_telemetry_epoch is not None and (
            not isinstance(expected_worker_telemetry_epoch, str)
            or not expected_worker_telemetry_epoch
        ):
            raise ValueError("expected Worker telemetry epoch must be a non-empty string")
        has_boot_id = isinstance(expected_worker_boot_id, str) and bool(expected_worker_boot_id)
        has_telemetry_epoch = (
            isinstance(expected_worker_telemetry_epoch, str)
            and bool(expected_worker_telemetry_epoch)
        )
        if has_boot_id != has_telemetry_epoch:
            raise ValueError("Worker boot ID and telemetry epoch must be pinned together")
        remote = scope == "gateway"
        if remote and (not has_boot_id or not has_telemetry_epoch):
            raise ValueError("Gateway Worker calls require a pinned ID, boot_id, and telemetry epoch")
        if remote and auth_token_env == "ARC_ENV_TOKEN":
            raise ValueError("Gateway Worker calls require a Gateway token separate from ARC_ENV_TOKEN")
        if scope == "service" and auth_token_env == "ARC_ENV_TOKEN":
            raise ValueError("service Worker calls require a Worker-specific token environment name")
        self.scope = scope
        self.url_prefix = url_prefix
        self.auth_token_env = auth_token_env
        self.worker_id = worker_id
        self.node_id = node_id or worker_id
        self.expected_worker_boot_id = expected_worker_boot_id
        self.expected_worker_telemetry_epoch = expected_worker_telemetry_epoch
        self._pinned_policy_version = policy_version
        self.timeout_seconds = timeout_seconds
        self.http_transport = http_transport
        self._client = client
        self._owns_client = client is None
        self._episode_leases: dict[str, tuple[int, str, str]] = {}
        self._cleanup_tasks: set[asyncio.Task[None]] = set()

    @property
    def policy_version(self) -> int | None:
        """The constructor-pinned cohort version, if this manager has one."""
        return self._pinned_policy_version

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            token = os.environ.get(self.auth_token_env)
            if not token:
                raise RuntimeError(f"{self.auth_token_env} is required for worker access")
            headers = {"Authorization": f"Bearer {token}"}
            self._client = httpx.AsyncClient(
                base_url=f"{self.worker_url}/",
                headers=headers,
                timeout=httpx.Timeout(self.timeout_seconds),
                # ARC tool execution can leave this connection idle past the
                # Worker server's timeout; generation must not be replayed.
                limits=httpx.Limits(max_keepalive_connections=0),
                transport=self.http_transport,
            )
        return self._client

    def _check_availability_response(self, response: httpx.Response, operation: str, code: str) -> None:
        if code not in _CONTRACT_CODES and (response.status_code >= 500 or code in _UNAVAILABLE_CODES):
            raise WorkerUnavailableError(
                f"worker {operation} returned HTTP {response.status_code} ({code})",
                worker_id=self.worker_id,
            )

    def _path(self, path: str) -> str:
        if not isinstance(path, str) or not path:
            raise ValueError("Worker request path must be non-empty")
        return f"{self.url_prefix.rstrip('/')}/{path.lstrip('/')}" if self.url_prefix else path.lstrip("/")

    def _has_worker_identity_pin(self) -> bool:
        return bool(self.expected_worker_boot_id and self.expected_worker_telemetry_epoch)

    def _worker_identity_headers(self) -> dict[str, str]:
        if self.scope not in {"service", "gateway"} or not self._has_worker_identity_pin():
            return {}
        return {
            "X-Expected-Worker-Boot-ID": self.expected_worker_boot_id,
            "X-Expected-Worker-Telemetry-Epoch": self.expected_worker_telemetry_epoch,
        }

    def _require_worker_identity_pin(self) -> None:
        if self.scope in {"service", "gateway"} and not self._has_worker_identity_pin():
            raise GenerationContractError(
                f"{self.scope.title()} Worker is unpinned; resolve /health before active requests",
            )

    def pin_worker_identity(self, boot_id: str, telemetry_epoch: str) -> None:
        """Pin a discovered service Worker before active policy/episode requests."""
        if self.scope not in {"service", "gateway"}:
            raise ValueError("only service and Gateway Worker clients have a membership identity")
        if (not isinstance(boot_id, str) or not boot_id
                or not isinstance(telemetry_epoch, str) or not telemetry_epoch):
            raise ValueError("Worker boot_id and telemetry_epoch must be non-empty strings")
        self.expected_worker_boot_id = boot_id
        self.expected_worker_telemetry_epoch = telemetry_epoch

    def _verify_worker_identity(self, response: httpx.Response, *, allow_epoch_change: bool = False) -> None:
        if self.scope not in {"service", "gateway"}:
            return
        observed = {"X-Worker-Node-ID": self.node_id}
        if self._has_worker_identity_pin() and not allow_epoch_change:
            observed.update({
                "X-Worker-Boot-ID": self.expected_worker_boot_id,
                "X-Worker-Telemetry-Epoch": self.expected_worker_telemetry_epoch,
            })
        if not response.is_success and not any(response.headers.get(name) for name in observed):
            return
        for name, expected in observed.items():
            if response.headers.get(name) != expected:
                raise GenerationContractError(
                    f"Worker response {name} differs from the pinned service identity",
                )

    async def get_health(self) -> dict[str, Any]:
        """Read authenticated Worker health for capacity and resource admission."""
        client = self._get_client()
        try:
            response = await client.get(
                self._path("health"), headers=self._worker_identity_headers(),
            )
        except httpx.HTTPError as exc:
            raise WorkerUnavailableError(
                "could not read Worker health", worker_id=self.worker_id,
            ) from exc
        self._verify_worker_identity(response)
        if response.status_code >= 500:
            raise WorkerUnavailableError(
                f"Worker health returned HTTP {response.status_code}",
                worker_id=self.worker_id,
            )
        if response.status_code != 200:
            raise RuntimeError(f"Worker health returned HTTP {response.status_code}")
        try:
            health = response.json()
        except ValueError as exc:
            raise GenerationContractError("Worker health response was not valid JSON") from exc
        if (not isinstance(health, dict)
                or health.get("role") != "worker"
                or type(health.get("ready")) is not bool):
            raise GenerationContractError("Worker health response did not match the health contract")
        if self.worker_id is not None and health.get("worker_id") != self.worker_id:
            raise GenerationContractError("Worker health identifies a different Worker ID")
        if self.scope == "service":
            deployment = health.get("deployment")
            boot_id = health.get("boot_id")
            telemetry_epoch = health.get("telemetry_epoch")
            if (not isinstance(deployment, dict)
                    or deployment.get("scope") != "service"
                    or deployment.get("node_id") != self.node_id
                    or not isinstance(boot_id, str) or not boot_id
                    or not isinstance(telemetry_epoch, str) or not telemetry_epoch):
                raise GenerationContractError("service Worker health has incomplete node or process identity")
            if (self._has_worker_identity_pin()
                    and (boot_id != self.expected_worker_boot_id
                         or telemetry_epoch != self.expected_worker_telemetry_epoch)):
                raise GenerationContractError("service Worker health differs from its pinned identity")
        kv_cache = health.get("kv_cache")
        if kv_cache is not None and not isinstance(kv_cache, dict):
            raise GenerationContractError("Worker KV cache health was not an object or null")
        if isinstance(kv_cache, dict):
            usage = kv_cache.get("usage_fraction")
            age = kv_cache.get("age_seconds")
            for flag in ("available", "stale"):
                if flag in kv_cache and type(kv_cache[flag]) is not bool:
                    raise GenerationContractError(f"Worker KV {flag} field was not boolean")
            if usage is not None and (
                isinstance(usage, bool) or not isinstance(usage, (int, float))
                or not math.isfinite(usage) or not 0 <= usage <= 1
            ):
                raise GenerationContractError("Worker KV usage was outside its measured fraction contract")
            if age is not None and (
                isinstance(age, bool) or not isinstance(age, (int, float))
                or not math.isfinite(age) or age < 0
            ):
                raise GenerationContractError("Worker KV sample age was outside its nonnegative contract")
            for count in ("num_running", "num_waiting", "num_preempted"):
                value = kv_cache.get(count)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                ):
                    raise GenerationContractError(f"Worker KV {count} field was not a nonnegative integer")
            sampled_at = kv_cache.get("sampled_at")
            if sampled_at is not None and (
                isinstance(sampled_at, bool) or not isinstance(sampled_at, (int, float))
                or not math.isfinite(sampled_at) or sampled_at < 0
            ):
                raise GenerationContractError("Worker KV sampled_at field was not a finite monotonic time")
            if kv_cache.get("error") is not None and not isinstance(kv_cache.get("error"), str):
                raise GenerationContractError("Worker KV error field was not a string or null")
        return health

    @staticmethod
    def _validate_episode_context(episode_id: str, policy_version: int, adapter_hash: str) -> tuple[str, int, str]:
        if not isinstance(episode_id, str) or not episode_id or len(episode_id) > 200:
            raise ValueError("episode_id must be a non-empty string of at most 200 characters")
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")
        if not isinstance(adapter_hash, str) or not _SHA256.fullmatch(adapter_hash):
            raise ValueError("adapter_hash must be a lowercase SHA-256 hex digest")
        return episode_id, policy_version, adapter_hash

    @staticmethod
    def _validate_purpose(purpose: str) -> str:
        if not isinstance(purpose, str) or purpose not in {"training", "evaluation"}:
            raise ValueError("episode purpose must be training or evaluation")
        return purpose

    async def acquire_episode(
        self,
        *,
        episode_id: str,
        policy_version: int,
        adapter_hash: str,
        purpose: str = "training",
    ) -> bool:
        """Pin one immutable adapter for a whole episode, including tool gaps."""
        episode_id, policy_version, adapter_hash = self._validate_episode_context(
            episode_id, policy_version, adapter_hash,
        )
        purpose = self._validate_purpose(purpose)
        if self.policy_version is not None and policy_version != self.policy_version:
            raise ValueError("episode policy_version differs from this cohort-pinned manager")
        self._require_worker_identity_pin()
        previous = self._episode_leases.get(episode_id)
        expected_lease = (policy_version, adapter_hash, purpose)
        if previous is not None and previous != expected_lease:
            raise GenerationContractError("episode_id is already pinned to another policy version or adapter SHA")
        payload = {
            "episode_id": episode_id,
            "policy_version": policy_version,
            "adapter_hash": adapter_hash,
        }
        if purpose == "evaluation":
            payload["purpose"] = purpose
        client = self._get_client()
        loop = asyncio.get_running_loop()
        retry_budget = min(self.timeout_seconds, _EVALUATION_ADMISSION_RETRY_MAX_SECONDS)
        acquire_started_at = loop.time()
        retry_deadline: float | None = None
        while True:
            try:
                if retry_deadline is None:
                    response = await client.post(
                        self._path("episode/acquire"), json=payload,
                        headers=self._worker_identity_headers(),
                    )
                else:
                    remaining = retry_deadline - loop.time()
                    if remaining <= 0:
                        raise RuntimeError("worker evaluation episode acquire retry deadline expired")
                    response = await asyncio.wait_for(
                        client.post(
                            self._path("episode/acquire"), json=payload,
                            headers=self._worker_identity_headers(),
                        ),
                        timeout=remaining,
                    )
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(self._best_effort_release(payload))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    pass
                raise
            except asyncio.TimeoutError as exc:
                if retry_deadline is None:
                    await self._best_effort_release(payload)
                    raise WorkerUnavailableError("worker episode acquire failed before receiving a response", worker_id=self.worker_id) from exc
                self._schedule_best_effort_release(payload)
                raise RuntimeError("worker evaluation episode acquire retry timed out") from exc
            except httpx.HTTPError as exc:
                await self._cleanup_acquire_failure(payload, deadline_active=retry_deadline is not None)
                raise WorkerUnavailableError("worker episode acquire failed before receiving a response", worker_id=self.worker_id) from exc

            if response.status_code == 200:
                break
            code = self._error_code(response)
            if (purpose == "evaluation" and response.status_code == 409
                    and code in _EVALUATION_ADMISSION_RETRY_CODES):
                if retry_deadline is None:
                    # Count the first busy response in the total retry window, while
                    # leaving a slow successful adapter load on the normal request timeout.
                    retry_deadline = acquire_started_at + retry_budget
                remaining = retry_deadline - loop.time()
                if remaining <= 0:
                    raise RuntimeError(
                        f"worker evaluation episode acquire retry deadline expired after HTTP 409 ({code})",
                    )
                await asyncio.sleep(min(_EVALUATION_ADMISSION_RETRY_DELAY_SECONDS, remaining))
                continue
            break
        if response.status_code != 200:
            if response.status_code >= 500:
                await self._cleanup_acquire_failure(payload, deadline_active=retry_deadline is not None)
            self._check_availability_response(response, "episode acquire", code)
            raise GenerationContractError(
                f"worker episode acquire returned HTTP {response.status_code} ({code})",
            )
        self._verify_worker_identity(response)
        try:
            result = response.json()
            if (not isinstance(result, dict)
                    or result.get("episode_id") != episode_id
                    or result.get("policy_version") != policy_version
                    or result.get("adapter_hash") != adapter_hash
                    or result.get("purpose") != purpose
                    or result.get("acquired") is not False and result.get("acquired") is not True):
                raise ValueError("episode lease acknowledgement changed its identity")
        except (ValueError, TypeError) as exc:
            await self._cleanup_acquire_failure(payload, deadline_active=retry_deadline is not None)
            raise GenerationContractError("worker episode acquire response did not match the lease contract") from exc
        self._episode_leases[episode_id] = expected_lease
        return bool(result["acquired"])

    async def release_episode(
        self,
        *,
        episode_id: str,
        policy_version: int,
        adapter_hash: str,
        purpose: str | None = None,
    ) -> bool:
        """Release a whole-episode lease; a repeated exact release is harmless."""
        episode_id, policy_version, adapter_hash = self._validate_episode_context(
            episode_id, policy_version, adapter_hash,
        )
        previous = self._episode_leases.get(episode_id)
        purpose = self._validate_purpose(
            previous[2] if purpose is None and previous is not None
            else "training" if purpose is None else purpose
        )
        expected = (policy_version, adapter_hash, purpose)
        if previous is not None and previous != expected:
            raise GenerationContractError("episode_id release tuple differs from its acquired policy version or SHA")
        payload = {
            "episode_id": episode_id,
            "policy_version": policy_version,
            "adapter_hash": adapter_hash,
        }
        if purpose == "evaluation":
            payload["purpose"] = purpose
        client = self._get_client()
        request_task = asyncio.create_task(client.post(
            self._path("episode/release"), json=payload,
            headers=self._worker_identity_headers(),
        ))
        cancelled = False
        try:
            response = await asyncio.shield(request_task)
        except asyncio.CancelledError:
            cancelled = True
            try:
                response = await asyncio.shield(request_task)
            except Exception:
                raise asyncio.CancelledError
        except httpx.HTTPError as exc:
            raise WorkerUnavailableError("worker episode release failed before receiving a response", worker_id=self.worker_id) from exc
        if response.status_code != 200:
            if cancelled:
                raise asyncio.CancelledError
            code = self._error_code(response)
            self._check_availability_response(response, "episode release", code)
            raise GenerationContractError(
                f"worker episode release returned HTTP {response.status_code} ({code})",
            )
        self._verify_worker_identity(response, allow_epoch_change=True)
        try:
            result = response.json()
            if (not isinstance(result, dict)
                    or result.get("episode_id") != episode_id
                    or result.get("purpose", "training") != purpose
                    or result.get("released") is not False and result.get("released") is not True):
                raise ValueError("episode release acknowledgement changed its identity")
        except (ValueError, TypeError) as exc:
            if cancelled:
                raise asyncio.CancelledError from exc
            raise GenerationContractError("worker episode release response did not match the lease contract") from exc
        self._episode_leases.pop(episode_id, None)
        if cancelled:
            raise asyncio.CancelledError
        return bool(result["released"])

    async def _best_effort_release(self, payload: dict[str, Any]) -> None:
        """Undo an acquire whose remote outcome is ambiguous, preserving its exact identity."""
        try:
            await self._get_client().post(
                self._path("episode/release"), json=payload,
                headers=self._worker_identity_headers(),
            )
        except httpx.HTTPError:
            pass

    async def _cleanup_acquire_failure(self, payload: dict[str, Any], *, deadline_active: bool) -> None:
        """Clean up ambiguous outcomes without extending an evaluation retry deadline."""
        if deadline_active:
            self._schedule_best_effort_release(payload)
        else:
            await self._best_effort_release(payload)

    def _schedule_best_effort_release(self, payload: dict[str, Any]) -> None:
        """Release an ambiguous timed-out acquire without extending its retry deadline."""
        task = asyncio.create_task(self._best_effort_release(payload))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        task.add_done_callback(self._consume_task_result)

    @staticmethod
    def _consume_task_result(task: asyncio.Task[Any]) -> None:
        """Retrieve background cleanup errors so a failed release stays best-effort."""
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _error_code(response: httpx.Response) -> str:
        try:
            body = response.json()
            error = body.get("error", body.get("detail", {})) if isinstance(body, dict) else {}
            code = error.get("code") if isinstance(error, dict) else None
            return code if isinstance(code, str) else "worker_error"
        except ValueError:
            return "worker_error"

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        episode_id: str | None = None,
        adapter_hash: str | None = None,
        policy_version: int | None = None,
        seed: int | None = None,
        shared_prefix_tokens: int = 0,
        **_unused: Any,
    ) -> TokenOutput:
        if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
            raise ValueError("request_id must be a non-empty string of at most 200 characters")
        if not isinstance(prompt_ids, list) or not prompt_ids or len(prompt_ids) > wire.MAX_CONTEXT_TOKENS:
            raise ValueError(f"prompt_ids must contain between 1 and {wire.MAX_CONTEXT_TOKENS} token IDs")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in prompt_ids):
            raise ValueError("prompt_ids must be nonnegative integers")
        if type(shared_prefix_tokens) is not int or not 0 <= shared_prefix_tokens < len(prompt_ids):
            raise ValueError("shared prefix must end before the final prompt token")
        safe_params = _safe_sampling_params(sampling_params)
        safe_seed = _safe_request_seed(seed)
        if len(prompt_ids) + safe_params["max_tokens"] > wire.MAX_CONTEXT_TOKENS:
            raise ValueError(f"prompt plus max_tokens must not exceed {wire.MAX_CONTEXT_TOKENS}")
        if policy_version is not None and (
            isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0
        ):
            raise ValueError("policy_version must be a nonnegative integer")
        if self.policy_version is not None:
            if policy_version is not None and policy_version != self.policy_version:
                raise ValueError("generate policy_version differs from this cohort-pinned manager")
            expected_version = self.policy_version
        else:
            expected_version = policy_version
        if episode_id is None or adapter_hash is None or expected_version is None:
            raise ValueError("generate requires an acquired episode_id, adapter_hash, and policy_version")
        episode_id, expected_version, adapter_hash = self._validate_episode_context(
            episode_id, expected_version, adapter_hash,
        )
        lease = self._episode_leases.get(episode_id)
        if lease is None or lease[:2] != (expected_version, adapter_hash):
            raise GenerationContractError("generate requires the exact live whole-episode lease")
        purpose = lease[2]
        self._require_worker_identity_pin()
        client = self._get_client()
        payload = {
            "policy_version": expected_version,
            "episode_id": episode_id,
            "adapter_hash": adapter_hash,
            "request_id": request_id,
            "prompt_ids": prompt_ids,
            "shared_prefix_tokens": shared_prefix_tokens,
            "sampling_params": safe_params,
        }
        if purpose == "evaluation":
            payload["purpose"] = purpose
        if safe_seed is not None:
            payload["seed"] = safe_seed
        try:
            response = await client.post(
                self._path("generate"), json=payload,
                headers=self._worker_identity_headers(),
            )
        except httpx.HTTPError as exc:
            raise WorkerUnavailableError(
                f"worker generation request failed before receiving a response ({type(exc).__name__})",
                worker_id=self.worker_id,
            ) from exc
        if response.status_code != 200:
            code = self._error_code(response)
            self._check_availability_response(response, "generation", code)
            failure = GenerationContractError if code in _CONTRACT_CODES else RuntimeError
            raise failure(f"worker generation returned HTTP {response.status_code} ({code})")
        self._verify_worker_identity(response)
        try:
            result = wire.GenerationResponse.model_validate(response.json())
        except (ValidationError, ValueError, TypeError) as exc:
            raise GenerationContractError("worker generation response did not match TokenOutput contract") from exc
        if len(result.token_ids) != len(result.log_probs) or not result.token_ids:
            raise GenerationContractError("worker returned misaligned token IDs and behavior log probabilities")
        if any(not math.isfinite(value) for value in result.log_probs):
            raise GenerationContractError("worker returned a non-finite behavior log probability")
        if result.extra_fields.get("adapter_sha256") != adapter_hash:
            raise GenerationContractError("worker response adapter SHA differs from the episode lease")
        if result.extra_fields.get("episode_purpose", "training") != purpose:
            raise GenerationContractError("worker response episode purpose differs from the episode lease")
        sampling_receipt = result.extra_fields.get("sampling_config")
        if not isinstance(sampling_receipt, dict):
            raise GenerationContractError("worker response omitted its sampling receipt")
        expected_seed_mode = "request" if safe_seed is not None else "worker_rng"
        if (sampling_receipt.get("seed_mode") != expected_seed_mode
                or sampling_receipt.get("seed") != safe_seed):
            raise GenerationContractError("worker response sampling seed differs from the request")
        if not result.timing or any(not math.isfinite(value) or value < 0 for value in result.timing.values()):
            raise GenerationContractError("worker returned invalid generation timing metadata")
        minimum = result.extra_fields.get("min_global_steps")
        maximum = result.extra_fields.get("max_global_steps")
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or minimum != maximum
            or minimum != expected_version
        ):
            raise GenerationContractError("worker response omitted or changed the frozen policy version")
        extra_fields = dict(result.extra_fields)
        extra_fields["worker_id"] = result.worker_id
        extra_fields["timing"] = result.timing
        from verl.workers.rollout.replica import TokenOutput

        return TokenOutput(
            token_ids=result.token_ids,
            log_probs=result.log_probs,
            stop_reason=result.stop_reason,
            extra_fields=extra_fields,
        )

    async def aclose(self) -> None:
        if self._cleanup_tasks:
            pending = tuple(self._cleanup_tasks)
            _, pending = await asyncio.wait(pending, timeout=min(1.0, self.timeout_seconds))
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
