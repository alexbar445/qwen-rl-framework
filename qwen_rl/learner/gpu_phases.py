"""Fixed-topology GPU ownership phases for the Learner and Worker systems."""
from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import time
from typing import Any

import httpx


_LOGGER = logging.getLogger(__name__)
_MAX_WORKER_ERROR_BODY_BYTES = 16_384
_MAX_WORKER_ERROR_CODE_CHARS = 96
_MAX_WORKER_ERROR_MESSAGE_CHARS = 512
_AUTH_VALUE_RE = re.compile(
    r"(?i)(authorization\s*[:=]\s*)(?:(?:bearer|basic)\s+)?\S+"
)
_BEARER_VALUE_RE = re.compile(r"(?i)(\b(?:bearer|basic)\s+)\S+")
_CREDENTIAL_FIELD_RE = re.compile(
    r"(?i)(\b(?:access[_-]?token|refresh[_-]?token|token|api[_-]?key|secret|password|credential)\b\s*[:=]\s*)\S+"
)


def _diagnostic_secret_values(token: str) -> list[str]:
    secrets = {token} if token else set()
    markers = ("token", "secret", "password", "credential", "api_key", "apikey", "private_key")
    for name, value in os.environ.items():
        if (any(marker in name.lower() for marker in markers)
                and isinstance(value, str) and value):
            secrets.add(value)
    return sorted(secrets, key=len, reverse=True)


def _safe_diagnostic_fragment(value: Any, secrets: list[str], *, limit: int) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = str(value)
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    text = _AUTH_VALUE_RE.sub("[REDACTED]", text)
    text = _BEARER_VALUE_RE.sub("[REDACTED]", text)
    text = _CREDENTIAL_FIELD_RE.sub(r"\1[REDACTED]", text)
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit - 3].rstrip() + "..."
    return text or None

def _worker_http_error_note(worker: dict[str, Any], route: str, payload: dict[str, Any],
                            response: httpx.Response, token: str) -> str:
    secrets = _diagnostic_secret_values(token)
    error_code = None
    error_message = None
    body_size = len(response.content)
    if body_size > _MAX_WORKER_ERROR_BODY_BYTES:
        error_message = (
            f"response body omitted ({body_size} bytes exceeds "
            f"{_MAX_WORKER_ERROR_BODY_BYTES}-byte diagnostic limit)"
        )
    elif body_size == 0:
        error_message = "response body was empty"
    else:
        try:
            body = response.json()
        except (ValueError, UnicodeDecodeError):
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                error_code = error.get("code")
                error_message = error.get("message")
            if error_message is None:
                detail = body.get("detail")
                if isinstance(detail, str):
                    error_message = detail
                elif isinstance(detail, dict):
                    error_code = error_code or detail.get("code")
                    error_message = detail.get("message")
        if error_message is None:
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type == "text/plain":
                error_message = response.text
            else:
                error_message = f"non-JSON response body omitted ({body_size} bytes)"
    worker_id = _safe_diagnostic_fragment(worker.get("id", "unknown"), secrets, limit=64) or "unknown"
    action = _safe_diagnostic_fragment(payload.get("action", "unknown"), secrets, limit=64) or "unknown"
    route_path = str(route).split("?", 1)[0].split("#", 1)[0]
    route_path = _safe_diagnostic_fragment(route_path, secrets, limit=96) or "unknown"
    code = _safe_diagnostic_fragment(
        error_code, secrets, limit=_MAX_WORKER_ERROR_CODE_CHARS,
    ) or "unavailable"
    message = _safe_diagnostic_fragment(
        error_message, secrets, limit=_MAX_WORKER_ERROR_MESSAGE_CHARS,
    ) or "unavailable"
    return (
        f"Worker control HTTP failure: worker={worker_id} action={action} route={route_path} "
        f"status={response.status_code} code={code} message={message}"
    )


async def _settle_owned_task(task: asyncio.Task):
    """Join an owned task even if its waiter is cancelled; return its outcome."""
    current = asyncio.current_task()
    joined_cancellations = 0
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            joined_cancellations += 1
            if current is not None:
                current.uncancel()
        except BaseException:
            break
    try:
        return task.result(), None, joined_cancellations
    except BaseException as exc:
        return None, exc, joined_cancellations


async def _await_owned_task(task: asyncio.Task):
    """Shield thread-backed work and join it before propagating cancellation."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        await _settle_owned_task(task)
        raise cancellation


class WorkerResidencyFleet:
    """The controller owns transitions; Workers independently fence every lease."""

    def __init__(self, config: dict[str, Any], *, transport=None):
        from ..contracts.gpu_sharing import shared_workers
        self.workers = shared_workers(config)
        self.timeout = float(config["learner"]["gpu_sharing_timeout"])
        self.transport = transport
        self._native_policy_preparation = config.get("model_backend") == "ternary"

    def idle_for_help(self) -> bool:
        """Unknown/busy Worker state is not permission to borrow its GPU.

        This is a scheduling observation, not the GPU ownership fence. The
        ordinary drain/suspend protocol still protects episodes after this probe.
        """
        for worker in self.workers:
            token = os.environ.get(worker.get("auth_token_env", "ARC_ENV_TOKEN"))
            if not token:
                return False
            headers = {"Authorization": f"Bearer {token}"}
            expected = {}
            if worker.get("scope") == "service":
                url = worker["endpoint"].rstrip("/") + worker.get("url_prefix", "").rstrip("/")
                for field, suffix in (("boot_id", "Boot-ID"), ("telemetry_epoch", "Telemetry-Epoch")):
                    headers[f"X-Expected-Worker-{suffix}"] = worker[field]
                    expected[f"X-Worker-{suffix}"] = worker[field]
                expected["X-Worker-Node-ID"] = worker["node_id"]
            else:
                url = f"http://{worker.get('host', '127.0.0.1')}:{worker['port']}"
            try:
                with httpx.Client(timeout=min(self.timeout, 2.0), trust_env=False,
                                  transport=self.transport, headers=headers) as client:
                    response = client.get(url + "/residency")
                    response.raise_for_status()
                    if any(response.headers.get(k) != v for k, v in expected.items()):
                        return False
                    state = response.json()
                if (state.get("worker_id") != worker["id"] or state.get("state") != "resident"
                        or state.get("ready") is not True or state.get("policy_ready") is not True
                        or state.get("enabled") is not True or state.get("native_error") is not None
                        or state.get("policy_update_active") is not False
                        or any(type(state.get(k)) is not int or state[k] != 0 for k in
                               ("episode_refs", "generation_refs", "active_generations", "pending_generations"))):
                    return False
            except (httpx.HTTPError, ValueError, KeyError):
                return False
        return True

    def _post(self, worker, route, payload):
        token = os.environ.get(worker.get("auth_token_env", "ARC_ENV_TOKEN"))
        if not token:
            raise RuntimeError(f"missing residency credentials for {worker['id']}")
        headers = {"Authorization": f"Bearer {token}"}
        expected_headers = {}
        if worker.get("scope") == "service":
            url = worker["endpoint"].rstrip('/') + worker.get("url_prefix", "").rstrip('/')
            for field, header in (("boot_id", "Boot-ID"), ("telemetry_epoch", "Telemetry-Epoch")):
                if not worker.get(field):
                    raise RuntimeError("phase Worker membership must be pinned")
                headers[f"X-Expected-Worker-{header}"] = worker[field]
                expected_headers[f"X-Worker-{header}"] = worker[field]
            expected_headers["X-Worker-Node-ID"] = worker["node_id"]
        else:
            url = f"http://{worker.get('host', '127.0.0.1')}:{worker['port']}"
        with httpx.Client(timeout=self.timeout, trust_env=False, transport=self.transport,
                          headers=headers) as client:
            response = client.post(url.rstrip('/') + route, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                diagnostic = _worker_http_error_note(worker, route, payload, response, token)
                exc.add_note(diagnostic)
                _LOGGER.warning("%s", diagnostic)
                raise
            if any(response.headers.get(key) != value for key, value in expected_headers.items()):
                raise RuntimeError("Worker control response changed its membership identity")
            result = response.json()
        return result

    def _one(self, worker, action, policy_version=None, adapter_sha256=None):
        payload = {"action": action}
        if action == "admit":
            payload.update(policy_version=policy_version, adapter_sha256=adapter_sha256)
        result = self._post(worker, "/residency", payload)
        expected = {"drain": "draining", "suspend": "suspended",
                    "wake": "resident_fenced", "admit": "resident"}[action]
        if (not isinstance(result, dict) or result.get("worker_id") != worker["id"]
                or result.get("state") != expected
                or result.get("admission_open") is not (action == "admit")):
            raise RuntimeError(f"Worker {worker['id']} did not prove residency action {action}")
        if action == "drain" and result.get("drain_complete") is not True:
            raise RuntimeError(f"Worker {worker['id']} still holds episodes")
        return result

    def apply(self, action: str, *, policy_version=None, adapter_sha256=None) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = {row["id"]: executor.submit(self._one, row, action, policy_version, adapter_sha256) for row in self.workers}
            return {key: future.result() for key, future in futures.items()}


    def _prepare_one(self, worker, policy):
        result = self._post(worker, "/policy/prepare", {"version": policy["version"]})
        expected_files = {name: {"sha256": item["sha256"].lower()}
                          for name, item in policy["files"].items()}
        if (not isinstance(result, dict) or result.get("worker_id") != worker["id"]
                or type(result.get("version")) is not int
                or result.get("version") != policy["version"]
                or result.get("prepared") is not True or result.get("files") != expected_files):
            raise RuntimeError(f"Worker {worker['id']} did not prepare the exact policy manifest")
        return result

    def prepare_policy(self, policy):
        # Only explicitly shared local Workers participate. Fixed-role remote
        # Workers continue through the ordinary policy synchronization path.
        # T2 preparation only downloads native files and checks metadata; let
        # these bounded requests overlap. Q4 still has a CPU converter to bound.
        slots = len(self.workers) if self._native_policy_preparation else 1
        with ThreadPoolExecutor(max_workers=slots) as executor:
            futures = {row["id"]: executor.submit(self._prepare_one, row, policy)
                       for row in self.workers}
            return {key: future.result() for key, future in futures.items()}


class GpuPhaseCoordinator:
    """Switch ownership without replacing ranks, parameters, or optimizer objects.

    Failed transitions stay fenced. A caller must not wake the other system on
    an uncertain release; full-state recovery remains the restart boundary.
    """

    def __init__(self, config, *, engine, worker_pool, compute_owner, fleet=None):
        self.engine = engine
        self.pool = worker_pool
        self.compute = compute_owner
        self.fleet = fleet if fleet is not None else WorkerResidencyFleet(config)
        self.timeout = float(config["learner"]["gpu_sharing_timeout"])
        self.shared_worker_ids = tuple(config["learner"]["shared_worker_ids"])
        learner_config = config["learner"]
        self._checkpoint_overlap_enabled = (
            config.get("model_backend") == "ternary"
            and learner_config.get("parallelism") == "ddp"
            and learner_config.get("gpu_sharing") == "phase_shared"
            and isinstance(learner_config.get("recovery_dir"), str)
            and bool(learner_config["recovery_dir"].strip())
        )
        self.state = "resident_fenced"
        self.history = deque(maxlen=16)
        self.totals: dict[str, dict[str, float | int]] = {}
        self._lock = asyncio.Lock()
        self._sampling_failure_details: dict[str, Any] | None = None
        self._checkpoint_phase_entry: dict[str, Any] | None = None
        self._checkpoint_phase_started: float | None = None
        self._checkpoint_park_task: asyncio.Task | None = None
        self._checkpoint_wake_task: asyncio.Task | None = None
        self._checkpoint_wake_status = "not_started"
        self._checkpoint_wake_seconds: float | None = None
        self._checkpoint_wake_started_at: float | None = None
        self._checkpoint_wake_finished_at: float | None = None
        self._checkpoint_wake_error: BaseException | None = None

    @property
    def supports_checkpoint_overlap(self) -> bool:
        """Whether this is the recoverable T2 DDP phase-shared path."""
        return self._checkpoint_overlap_enabled

    def checkpoint_ready_callback(self, loop: asyncio.AbstractEventLoop):
        """Return the blocking callback used at the post-snapshot checkpoint seam.

        The update thread owns the synchronous engine call while this callback
        bridges back to the controller loop to park only the now-idle rank
        service and start the fenced Worker wake.
        """
        if not self.supports_checkpoint_overlap:
            raise RuntimeError("checkpoint overlap requires recoverable T2 DDP phase sharing")
        if not loop.is_running():
            raise RuntimeError("checkpoint overlap callback requires a running controller loop")

        def ready() -> None:
            asyncio.run_coroutine_threadsafe(self.begin_checkpoint_overlap(), loop).result()

        return ready

    def snapshot(self):
        wake_pending = (self._checkpoint_wake_task is not None
                        and not self._checkpoint_wake_task.done())
        return {
            "mode": "phase_shared",
            "shared_worker_ids": list(self.shared_worker_ids),
            "state": self.state,
            "transitions": list(self.history),
            "totals": {key: dict(value) for key, value in self.totals.items()},
            "checkpoint_overlap": {
                "active": self._checkpoint_phase_entry is not None,
                "pending": wake_pending,
                "wake_status": self._checkpoint_wake_status,
                "wake_seconds": self._checkpoint_wake_seconds,
                "wake_started_at": self._checkpoint_wake_started_at,
                "wake_finished_at": self._checkpoint_wake_finished_at,
                "wake_error_type": (None if self._checkpoint_wake_error is None
                                     else type(self._checkpoint_wake_error).__name__),
            },
        }

    async def begin_checkpoint_overlap(self) -> None:
        """Park idle T2 DDP ranks, then launch Worker wake behind the fence.

        This method is entered by ``checkpoint_ready_callback`` after the
        update thread has refreshed and published its immutable CPU snapshot.
        It deliberately does not use ComputeOwner: the surrounding update
        still owns that lock, while rank GPU computation has already ended.
        """
        if not self.supports_checkpoint_overlap:
            raise RuntimeError("checkpoint overlap requires recoverable T2 DDP phase sharing")
        async with self._lock:
            if self.state != "training":
                raise RuntimeError(f"checkpoint overlap is invalid in {self.state}")
            if self._checkpoint_phase_entry is not None or (
                    self._checkpoint_wake_task is not None
                    and not self._checkpoint_wake_task.done()):
                raise RuntimeError("a checkpoint Worker wake is already pending")

            started = time.monotonic()
            entry = {
                "state": "checkpointing",
                "started_at": time.time(),
                "overlap": {"worker_wake": {"status": "not_started"}},
            }
            self.state = "checkpointing"
            self.history.append(entry)
            self._checkpoint_phase_entry = entry
            self._checkpoint_phase_started = started
            self._checkpoint_wake_status = "not_started"
            self._checkpoint_wake_seconds = None
            self._checkpoint_wake_started_at = None
            self._checkpoint_wake_finished_at = None
            self._checkpoint_wake_error = None

            park_task = asyncio.create_task(
                asyncio.to_thread(self.engine.park_for_checkpoint),
                name="learner-park-for-checkpoint",
            )
            self._checkpoint_park_task = park_task
            try:
                async with asyncio.timeout(self.timeout):
                    park_result = await _await_owned_task(park_task)
            except BaseException as exc:
                self._close_checkpoint_phase(status="failed", error=exc)
                self.state = "failed"
                raise

            entry["overlap"]["checkpoint_park"] = park_result
            entry["overlap"]["worker_wake"] = {"status": "pending"}
            self._checkpoint_wake_status = "pending"
            self._checkpoint_wake_task = asyncio.create_task(
                self._run_checkpoint_worker_wake(),
                name="checkpoint-worker-wake",
            )

    async def _run_checkpoint_worker_wake(self):
        started = time.monotonic()
        self._checkpoint_wake_status = "running"
        self._checkpoint_wake_started_at = time.time()
        entry = self._checkpoint_phase_entry
        if entry is not None:
            entry["overlap"]["worker_wake"] = {
                "status": "running",
                "started_at": self._checkpoint_wake_started_at,
            }
        try:
            result = await asyncio.to_thread(self.fleet.apply, "wake")
        except BaseException as exc:
            self._checkpoint_wake_error = exc
            self._checkpoint_wake_status = "failed"
            raise
        else:
            self._checkpoint_wake_status = "succeeded"
            return result
        finally:
            self._checkpoint_wake_seconds = time.monotonic() - started
            self._checkpoint_wake_finished_at = time.time()
            entry = self._checkpoint_phase_entry
            if entry is not None:
                wake = entry["overlap"].setdefault("worker_wake", {})
                wake.update(
                    status=self._checkpoint_wake_status,
                    finished_at=self._checkpoint_wake_finished_at,
                    seconds=self._checkpoint_wake_seconds,
                )
                if self._checkpoint_wake_error is not None:
                    wake["error_type"] = type(self._checkpoint_wake_error).__name__

    def _close_checkpoint_phase(self, *, status: str, result=None, error=None) -> None:
        entry = self._checkpoint_phase_entry
        started = self._checkpoint_phase_started
        if entry is None:
            return
        entry["status"] = status
        if result is not None:
            entry["result"] = result
        if error is not None:
            entry.update(error_type=type(error).__name__)
        entry["seconds"] = 0.0 if started is None else time.monotonic() - started
        totals = self.totals.setdefault(
            "checkpointing", {"count": 0, "seconds": 0.0, "failures": 0},
        )
        totals["count"] += 1
        totals["seconds"] += entry["seconds"]
        totals["failures"] += int(status == "failed")
        self._checkpoint_phase_entry = None
        self._checkpoint_phase_started = None

    async def abort_checkpoint_overlap(self, primary: BaseException) -> None:
        """Drain overlap work and fence the coordinator after update failure."""
        async with self._lock:
            had_checkpoint_phase = self._checkpoint_phase_entry is not None
            cleanup_errors: list[BaseException] = []
            for owned in (self._checkpoint_park_task, self._checkpoint_wake_task):
                if owned is None:
                    continue
                _result, error, joined_cancellations = await _settle_owned_task(owned)
                if error is not None and error is not primary:
                    cleanup_errors.append(error)
                if joined_cancellations:
                    primary.add_note(
                        f"joined {joined_cancellations} cancellation(s) while draining checkpoint overlap"
                    )
            if self._checkpoint_phase_entry is not None:
                self._close_checkpoint_phase(status="failed", error=primary)
            if self.state != "failed":
                self.state = "failed"
                if not had_checkpoint_phase and not any(
                        row.get("state") == "failed" for row in self.history):
                    self.history.append({
                        "state": "failed",
                        "status": "failed",
                        "started_at": time.time(),
                        "error_type": type(primary).__name__,
                    })
            for error in cleanup_errors:
                primary.add_note(
                    "checkpoint overlap cleanup failed: "
                    f"{type(error).__name__}: {str(error)[:300]}"
                )

    def fail(self, error):
        """Record uncertain ownership without attempting an unsafe automatic wake."""
        self.state = "failed"
        self.history.append({"state": "failed", "status": "failed",
                             "started_at": time.time(), "error_type": type(error).__name__})

    async def _transition(self, target, operation):
        started = time.monotonic()
        self.state = target
        entry = {"state": target, "started_at": time.time()}
        self.history.append(entry)
        try:
            async with asyncio.timeout(self.timeout):
                result = await operation()
        except BaseException as exc:
            self.state = "failed"
            entry.update(status="failed", error_type=type(exc).__name__)
            if target == "sampling" and self._sampling_failure_details is not None:
                entry["admission_failure_details"] = self._sampling_failure_details
                notes = getattr(exc, "__notes__", None)
                if notes:
                    entry["error_notes"] = list(notes)
            raise
        else:
            entry.update(status="succeeded", result=result)
            return result
        finally:
            entry["seconds"] = time.monotonic() - started
            totals = self.totals.setdefault(target, {"count": 0, "seconds": 0.0, "failures": 0})
            totals["count"] += 1
            totals["seconds"] += entry["seconds"]
            totals["failures"] += int(entry.get("status") == "failed")

    async def begin_sampling(self, policy):
        async with self._lock:
            if self._checkpoint_phase_entry is not None or (
                    self._checkpoint_wake_task is not None
                    and not self._checkpoint_wake_task.done()):
                raise RuntimeError("sampling admission is blocked by checkpoint overlap")
            if self.state not in {"resident_fenced", "sampling"}:
                raise RuntimeError(f"sampling admission is invalid in {self.state}")
            self._sampling_failure_details = None

            async def compensate_admission() -> dict[str, Any]:
                async def run_step(name: str, operation) -> dict[str, Any]:
                    owned = asyncio.create_task(operation(), name=f"admission-compensation-{name}")
                    _result, error, joined_cancellations = await _settle_owned_task(owned)
                    outcome: dict[str, Any] = {
                        "status": "failed" if error is not None else "succeeded",
                    }
                    if error is not None:
                        outcome.update(error_type=type(error).__name__, message=str(error)[:300])
                    if joined_cancellations:
                        outcome["joined_cancellations"] = joined_cancellations
                    return outcome

                async def drain_workers():
                    return await asyncio.to_thread(self.fleet.apply, "drain")

                # Close the central lease pool first. Even if it times out after
                # closing, still ask every shared Worker to fence itself.
                pool_result = await run_step(
                    "pause_pool",
                    lambda: self.pool.pause_and_drain(
                        timeout=self.timeout, worker_ids=self.shared_worker_ids,
                    ),
                )
                worker_result = await run_step("drain_workers", drain_workers)
                return {"pool_pause_and_drain": pool_result, "worker_drain": worker_result}

            async def admit():
                fanout = asyncio.create_task(
                    asyncio.to_thread(
                        self.fleet.apply,
                        "admit",
                        policy_version=policy.version,
                        adapter_sha256=policy.adapter_sha256,
                    ),
                    name=f"worker-admit-v{policy.version}",
                )
                try:
                    result = await _await_owned_task(fanout)
                    await self.pool.resume_admission(worker_ids=self.shared_worker_ids)
                    return result
                except BaseException as primary:
                    _result, fanout_error, joined_cancellations = await _settle_owned_task(fanout)
                    compensation = await compensate_admission()
                    details: dict[str, Any] = {"compensation": compensation}
                    if fanout_error is not None and fanout_error is not primary:
                        details["admission_fanout_error"] = {
                            "error_type": type(fanout_error).__name__,
                            "message": str(fanout_error)[:300],
                        }
                    if joined_cancellations:
                        details["joined_cancellations"] = joined_cancellations
                    self._sampling_failure_details = details
                    failures = [
                        f"{name}: {value.get('error_type', value.get('status'))}"
                        for name, value in compensation.items()
                        if value["status"] == "failed"
                    ]
                    if failures and hasattr(primary, "add_note"):
                        primary.add_note(
                            "Worker admission compensation failed: " + "; ".join(failures)
                        )
                    raise

            await self._transition("sampling", admit)

    async def begin_training(self):
        async with self._lock:
            if self.state != "sampling":
                raise RuntimeError(f"training ownership is invalid in {self.state}")
            async def release_workers():
                await self.pool.pause_and_drain(timeout=self.timeout, worker_ids=self.shared_worker_ids)
                await asyncio.to_thread(self.fleet.apply, "drain")
                return await asyncio.to_thread(self.fleet.apply, "suspend")
            await self._transition("draining_workers", release_workers)
            await self._transition("training", lambda: self.compute.run(self.engine.wake))

    async def finish_training(self, policy):
        async with self._lock:
            if self._checkpoint_phase_entry is not None:
                if self.state != "checkpointing":
                    raise RuntimeError(f"checkpoint completion is invalid in {self.state}")
                await self._finish_checkpoint_overlap_locked(policy)
                return
            if self.supports_checkpoint_overlap:
                error = RuntimeError(
                    "recoverable T2 DDP update completed without the checkpoint overlap callback"
                )
                self.fail(error)
                raise error
        # Scoring and non-T2 paths retain the explicit serial Learner park,
        # followed by the ordinary fenced Worker wake/preparation join.
        await self._finish_compute(lambda: self._wake_and_prepare(policy))

    async def _prepare_checkpoint_policy(self, policy):
        started = time.monotonic()
        started_at = time.time()
        entry = self._checkpoint_phase_entry
        overlap = None if entry is None else entry["overlap"]
        if overlap is not None:
            overlap["policy_preparation_started_at"] = started_at
        try:
            return await asyncio.to_thread(self.fleet.prepare_policy, policy)
        finally:
            finished_at = time.time()
            if overlap is not None:
                overlap["policy_preparation_finished_at"] = finished_at
                overlap["policy_preparation_seconds"] = time.monotonic() - started

    async def _join_checkpoint_overlap(self, policy_preparation: asyncio.Task):
        wake = self._checkpoint_wake_task
        if wake is None:
            raise RuntimeError("checkpoint Worker wake was not started")
        return await asyncio.gather(wake, policy_preparation, return_exceptions=True)

    async def _finish_checkpoint_overlap_locked(self, policy) -> None:
        entry = self._checkpoint_phase_entry
        phase_started = self._checkpoint_phase_started
        if entry is None or phase_started is None:
            raise RuntimeError("checkpoint overlap history is missing")

        durable_ready_at = time.time()
        durable_ready_monotonic = time.monotonic()
        overlap = entry["overlap"]
        overlap["durable_ready_at"] = durable_ready_at
        overlap["durable_ready_seconds"] = durable_ready_monotonic - phase_started
        overlap["remaining_join_started_at"] = durable_ready_at
        policy_preparation = asyncio.create_task(
            self._prepare_checkpoint_policy(policy),
            name=f"checkpoint-policy-prepare-v{policy['version']}",
        )
        join_task = asyncio.create_task(
            self._join_checkpoint_overlap(policy_preparation),
            name=f"checkpoint-wake-prepare-v{policy['version']}",
        )
        try:
            async with asyncio.timeout(self.timeout):
                results = await asyncio.shield(join_task)
        except BaseException as primary:
            _joined, join_error, joined_cancellations = await _settle_owned_task(join_task)
            overlap["remaining_join_seconds"] = time.monotonic() - durable_ready_monotonic
            if join_error is not None and join_error is not primary:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "checkpoint overlap join failed: "
                        f"{type(join_error).__name__}: {str(join_error)[:300]}"
                    )
            if joined_cancellations and hasattr(primary, "add_note"):
                primary.add_note(
                    f"joined {joined_cancellations} cancellation(s) while draining checkpoint overlap"
                )
            self._close_checkpoint_phase(status="failed", error=primary)
            self.state = "failed"
            raise

        overlap["remaining_join_seconds"] = time.monotonic() - durable_ready_monotonic
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            primary = failures[0]
            if len(failures) > 1 and hasattr(primary, "add_note"):
                primary.add_note(
                    "additional checkpoint overlap branch failed: "
                    f"{type(failures[1]).__name__}: {str(failures[1])[:300]}"
                )
            self._close_checkpoint_phase(status="failed", error=primary)
            self.state = "failed"
            raise primary

        wake_result, preparation_result = results
        result = {
            "wake": {
                "seconds": self._checkpoint_wake_seconds,
                "workers": wake_result,
            },
            "policy_preparation": {
                "seconds": overlap.get("policy_preparation_seconds", 0.0),
                "workers": preparation_result,
            },
        }
        self._close_checkpoint_phase(status="succeeded", result=result)
        self.state = "resident_fenced"

    async def finish_scoring(self):
        """Return GPUs after scoring unchanged weights; no new policy to prepare."""
        await self._finish_compute(lambda: asyncio.to_thread(self.fleet.apply, "wake"))

    async def _finish_compute(self, wake_operation):
        async with self._lock:
            if self.state != "training":
                raise RuntimeError(f"Worker wake is invalid in {self.state}")
            await self._transition("parking_learner", lambda: self.compute.run(self.engine.park))
            await self._transition("resident_fenced", wake_operation)

    async def _wake_and_prepare(self, policy):
        async def measured(operation, *args):
            started = time.monotonic()
            result = await asyncio.to_thread(operation, *args)
            return {"seconds": time.monotonic() - started, "workers": result}

        async def join():
            # GPU wake and CPU-only preparation have disjoint resources. Neither
            # branch opens admission; normal installation follows their joint success.
            return await asyncio.gather(
                measured(self.fleet.apply, "wake"),
                measured(self.fleet.prepare_policy, policy),
                return_exceptions=True,
            )

        owned = asyncio.create_task(join(), name=f"wake-prepare-v{policy['version']}")
        try:
            results = await asyncio.shield(owned)
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
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return {"wake": results[0], "policy_preparation": results[1]}
