"""Guarded same-node Worker/gradient ownership cycles.

The lifecycle composes existing Worker residency HTTP actions with injected
compute-owner callbacks. It deliberately does not create CUDA residency state
or infer a Worker role from placement. Callers must provide an explicit
``phase_shared`` configuration and callbacks backed by the production owners.
"""
from __future__ import annotations

import asyncio
import inspect
import ipaddress
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib.parse import urlsplit

from ...contracts.gpu_sharing import shared_workers
from ...rollout.groups import PolicySnapshot
from ..gpu_phases import WorkerResidencyFleet
from .phase_timing import JobPhaseTiming, measured_async


if TYPE_CHECKING:
    from .adapter_snapshot import _OwnedAdapterSnapshot

class GradientLifecycleError(RuntimeError):
    """A guarded Worker/compute ownership transition could not be proven."""


_WORKER_STATES = {
    "drain": ("draining", False),
    "suspend": ("suspended", False),
    "wake": ("resident_fenced", False),
    "admit": ("resident", True),
}


def _is_loopback_host(host: str | None) -> bool:
    if not isinstance(host, str) or not host:
        return False
    normalized = host.strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_worker_urls(workers: tuple[dict[str, Any], ...]) -> None:
    """Require an explicit same-host HTTP route for every shared Worker."""
    for worker in workers:
        scope = worker.get("scope", "local")
        if scope == "local":
            if not _is_loopback_host(worker.get("host")):
                raise ValueError(
                    f"shared local Worker {worker.get('id')} must use a loopback URL host"
                )
            port = worker.get("port")
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError(f"shared Worker {worker.get('id')} requires a valid local port")
            continue
        if scope != "service":
            raise ValueError("shared Workers must use an explicit same-node local/service URL")
        endpoint = worker.get("endpoint")
        try:
            parsed = urlsplit(endpoint)
            endpoint_port = parsed.port
        except (TypeError, ValueError) as exc:
            raise ValueError(f"shared service Worker {worker.get('id')} has an invalid URL") from exc
        if (parsed.scheme not in {"http", "https"} or not _is_loopback_host(parsed.hostname)
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment or endpoint_port != worker.get("port")):
            raise ValueError(
                f"shared service Worker {worker.get('id')} must use an explicit loopback origin"
            )


async def _invoke(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run sync production hooks off-loop and accept native async hooks."""
    if inspect.iscoroutinefunction(callback):
        return await callback(*args, **kwargs)
    value = await asyncio.to_thread(callback, *args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value


async def _owned_call(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Join callback work through cancellation, then re-raise cancellation."""
    task = asyncio.create_task(_invoke(callback, *args, **kwargs))
    current = asyncio.current_task()
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
            if current is not None:
                current.uncancel()
        except BaseException:
            break
    try:
        result = task.result()
    except BaseException:
        if cancellation is not None:
            raise cancellation
        raise
    if cancellation is not None:
        raise cancellation
    return result


def _require_health_proof(result: Any, expected: str) -> None:
    """Accept only an explicit healthy status or a callback's literal proof."""
    if result is True:
        return
    if not isinstance(result, Mapping):
        raise GradientLifecycleError(f"compute health did not prove {expected} state")
    states = [result.get("state")]
    residency = result.get("residency")
    if isinstance(residency, Mapping):
        states.append(residency.get("state"))
    if expected == "awake":
        valid = any(state in ("awake", "ready", "resident") for state in states)
        if "ready" in result:
            valid = valid and result.get("ready") is True
    else:
        valid = any(state in ("parked", "asleep", "suspended") for state in states)
    error = result.get("error")
    if not valid or (error is not None and error != ""):
        raise GradientLifecycleError(f"compute health did not prove {expected} state")


def _verify_worker_results(
    result: Any, *, action: str, worker_ids: tuple[str, ...],
    policy_version: int | None = None, adapter_sha256: str | None = None,
) -> None:
    if not isinstance(result, Mapping) or set(result) != set(worker_ids):
        raise GradientLifecycleError(f"Worker {action} did not cover exactly the shared Workers")
    expected_state, expected_open = _WORKER_STATES[action]
    for worker_id in worker_ids:
        row = result[worker_id]
        if (not isinstance(row, Mapping) or row.get("worker_id") != worker_id
                or row.get("state") != expected_state
                or row.get("admission_open") is not expected_open):
            raise GradientLifecycleError(f"Worker {worker_id} did not prove {action}")
        if action == "drain" and row.get("drain_complete") is not True:
            raise GradientLifecycleError(f"Worker {worker_id} still holds generation leases")
        if action == "admit" and (
            type(row.get("policy_version")) is not int
            or row.get("policy_version") != policy_version
            or row.get("adapter_sha256") != adapter_sha256
        ):
            raise GradientLifecycleError(
                f"Worker {worker_id} admission did not echo the exact policy identity"
            )


def _verify_policy_acks(
    acknowledgements: Any,
    *,
    worker_ids: tuple[str, ...],
    identity: PolicySnapshot,
) -> None:
    if not isinstance(acknowledgements, Mapping):
        raise GradientLifecycleError("central policy sync returned no Worker ACK mapping")
    for worker_id in worker_ids:
        ack = acknowledgements.get(worker_id)
        if (not isinstance(ack, Mapping) or ack.get("worker_id") != worker_id
                or type(ack.get("version")) is not int
                or ack.get("version") != identity.version
                or ack.get("adapter_hash") != identity.adapter_sha256):
            raise GradientLifecycleError(
                f"Worker {worker_id} did not ACK central policy {identity.version} identity"
            )


class SameNodeGradientCycle:
    """Hold selected same-node Workers behind a compute-owner gradient phase.

    ``compute_wake`` and ``compute_park`` must operate the existing production
    compute owner. ``compute_health(expected_state)`` must independently verify
    the resulting owner state. ``latest_policy`` reads the central published
    policy and ``ensure_policy(policy, worker_ids)`` either installs it through
    the normal policy path or proves it is already installed, returning exact
    per-Worker version/adapter identity ACKs. No callback is guessed or created
    by this class.

    The object can be reused after a successful cycle. Any uncertain transition
    leaves it failed and fenced; recovery requires a new managed lifecycle.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        compute_wake: Callable[[], Any],
        compute_park: Callable[[], Any],
        compute_health: Callable[[str], Any],
        latest_policy: Callable[[], Mapping[str, Any]],
        ensure_policy: Callable[[Mapping[str, Any], tuple[str, ...]], Any],
        fleet: Any | None = None,
    ) -> None:
        learner = config.get("learner")
        if not isinstance(learner, Mapping) or learner.get("gpu_sharing") != "phase_shared":
            raise ValueError("gradient Worker lifecycle requires explicit phase_shared configuration")
        workers = shared_workers(config)
        _validate_worker_urls(workers)
        if any(not callable(item) for item in (
            compute_wake, compute_park, compute_health, latest_policy, ensure_policy,
        )):
            raise TypeError("compute and central policy lifecycle callbacks are required")
        self.worker_ids = tuple(row["id"] for row in workers)
        self.fleet = fleet if fleet is not None else WorkerResidencyFleet(dict(config))
        fleet_ids = tuple(row["id"] for row in getattr(self.fleet, "workers", workers))
        if fleet_ids != self.worker_ids:
            raise ValueError("Worker residency fleet must contain exactly the configured shared Workers")
        self._compute_wake = compute_wake
        self._compute_park = compute_park
        self._compute_health = compute_health
        self._latest_policy = latest_policy
        self._ensure_policy = ensure_policy
        self.state = "sampling"
        self.admitted_policy: Mapping[str, Any] | None = None
        self.admitted_identity: PolicySnapshot | None = None
        self._cycle_lock = asyncio.Lock()
        self._entered = False
        self._wake_started = False
        self._compute_awake = False

    async def _fleet_action(self, action: str, **kwargs: Any) -> Any:
        result = await _owned_call(self.fleet.apply, action, **kwargs)
        _verify_worker_results(
            result,
            action=action,
            worker_ids=self.worker_ids,
            policy_version=kwargs.get("policy_version"),
            adapter_sha256=kwargs.get("adapter_sha256"),
        )
        return result

    async def _health(self, expected: str) -> None:
        result = await _owned_call(self._compute_health, expected)
        _require_health_proof(result, expected)

    async def _policy(self) -> tuple[Mapping[str, Any], PolicySnapshot]:
        policy = await _owned_call(self._latest_policy)
        if not isinstance(policy, Mapping):
            raise GradientLifecycleError("central latest_policy did not return a policy mapping")
        try:
            identity = PolicySnapshot.from_policy(dict(policy))
        except (KeyError, TypeError, ValueError) as exc:
            raise GradientLifecycleError("central latest policy has no valid adapter identity") from exc
        if (len(identity.adapter_sha256) != 64
                or any(character not in "0123456789abcdef" for character in identity.adapter_sha256)):
            raise GradientLifecycleError("central latest policy adapter identity is not lowercase SHA256")
        return policy, identity

    async def _restore_workers_to_latest_policy(
        self, phase_timing: JobPhaseTiming | None = None,
    ) -> None:
        async with measured_async(phase_timing, "worker_wake"):
            await self._fleet_action("wake")
        for _ in range(3):
            async with measured_async(phase_timing, "policy_ensure"):
                policy, identity = await self._policy()
                acknowledgements = await _owned_call(self._ensure_policy, policy, self.worker_ids)
                _verify_policy_acks(acknowledgements, worker_ids=self.worker_ids, identity=identity)
                _, latest_identity = await self._policy()
            if latest_identity == identity:
                async with measured_async(phase_timing, "worker_admit"):
                    await self._fleet_action(
                        "admit",
                        policy_version=identity.version,
                        adapter_sha256=identity.adapter_sha256,
                    )
                self.admitted_policy = policy
                self.admitted_identity = identity
                self.state = "sampling"
                return
        raise GradientLifecycleError("central policy kept changing before Worker admission")

    async def _refence_workers(
        self, phase_timing: JobPhaseTiming | None = None,
    ) -> list[BaseException]:
        failures: list[BaseException] = []
        for action in ("drain", "suspend"):
            try:
                name = "worker_drain" if action == "drain" else "worker_suspend"
                async with measured_async(phase_timing, name):
                    await self._fleet_action(action)
            except BaseException as exc:
                failures.append(exc)
        return failures

    @staticmethod
    def _attach_cleanup_error(primary: BaseException, errors: list[BaseException]) -> None:
        if errors:
            primary.lifecycle_cleanup_errors = tuple(errors)
        if errors and hasattr(primary, "add_note"):
            primary.add_note(
                "lifecycle cleanup failed: " + "; ".join(type(error).__name__ for error in errors)
            )

    async def _enter(
        self, phase_timing: JobPhaseTiming | None,
    ) -> SameNodeGradientCycle:
        async with measured_async(phase_timing, "lifecycle_wait"):
            await self._cycle_lock.acquire()
        if self.state == "failed":
            self._cycle_lock.release()
            raise GradientLifecycleError("failed gradient lifecycle is fenced and cannot be reused")
        if self.state != "sampling":
            self._cycle_lock.release()
            raise GradientLifecycleError(f"cannot enter gradient phase from {self.state}")
        self.admitted_policy = None
        self.admitted_identity = None
        self._entered = False
        self._wake_started = False
        self._compute_awake = False
        self.state = "entering_gradient"
        try:
            async with measured_async(phase_timing, "worker_drain"):
                await self._fleet_action("drain")
            async with measured_async(phase_timing, "worker_suspend"):
                await self._fleet_action("suspend")
            self._wake_started = True
            wake_owner = getattr(self._compute_wake, "__self__", None)
            owner_loaded = getattr(wake_owner, "loaded", None)
            if owner_loaded is False:
                wake_detail = "compute_first_build"
            elif owner_loaded is True:
                wake_detail = "compute_warm_wake"
            else:
                wake_detail = "compute_wake_detail_unknown"
            async with measured_async(phase_timing, "compute_wake"):
                async with measured_async(phase_timing, wake_detail):
                    await _owned_call(self._compute_wake)
            self._compute_awake = True
            async with measured_async(phase_timing, "compute_wake_health"):
                await self._health("awake")
            self.state = "gradient"
            self._entered = True
            return self
        except BaseException as primary:
            self.state = "failed"
            cleanup_errors: list[BaseException] = []
            if self._wake_started:
                try:
                    async with measured_async(phase_timing, "compute_park"):
                        await _owned_call(self._compute_park)
                        await self._health("parked")
                    self._compute_awake = False
                except BaseException as exc:
                    cleanup_errors.append(exc)
            self._attach_cleanup_error(primary, cleanup_errors)
            self._cycle_lock.release()
            raise

    async def __aenter__(self) -> SameNodeGradientCycle:
        return await self._enter(None)

    @asynccontextmanager
    async def timed(self, phase_timing: JobPhaseTiming):
        result = await self._enter(phase_timing)
        try:
            yield result
        except BaseException as exc:
            await self._exit(type(exc), exc, exc.__traceback__, phase_timing)
            raise
        else:
            await self._exit(None, None, None, phase_timing)

    async def _exit(
        self, exc_type, exc, traceback, phase_timing: JobPhaseTiming | None,
        *, on_parked: Callable[[], Any] | None = None,
    ) -> bool:
        if not self._entered:
            return False
        self._entered = False
        try:
            self.state = "parking_compute"
            cleanup_errors: list[BaseException] = []
            try:
                async with measured_async(phase_timing, "compute_park"):
                    await _owned_call(self._compute_park)
                    await self._health("parked")
                self._compute_awake = False
                self.state = "compute_parked"
            except BaseException as park_error:
                cleanup_errors.append(park_error)
                self.state = "failed"

            if not cleanup_errors:
                # CPU optimizer work needs the Learner's proven GPU release,
                # not a fully rebuilt inference engine. The Worker remains
                # fenced and its scheduler session remains active throughout.
                if on_parked is not None:
                    try:
                        async with measured_async(phase_timing, "session_parked_notification"):
                            await _owned_call(on_parked)
                    except BaseException as notification_error:
                        cleanup_errors.append(notification_error)
            if not cleanup_errors:
                try:
                    await self._restore_workers_to_latest_policy(phase_timing)
                except BaseException as restore_error:
                    cleanup_errors.append(restore_error)
                    self.state = "failed"
                    cleanup_errors.extend(await self._refence_workers(phase_timing))

            if cleanup_errors:
                self.state = "failed"
                if exc is not None:
                    self._attach_cleanup_error(exc, cleanup_errors)
                elif len(cleanup_errors) == 1:
                    raise cleanup_errors[0]
                else:
                    primary = cleanup_errors[0]
                    self._attach_cleanup_error(primary, cleanup_errors[1:])
                    raise primary
            return False
        finally:
            self._cycle_lock.release()

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return await self._exit(exc_type, exc, traceback, None)




class LifecycleGradientCompute:
    """Synchronous client adapter around one persistent async ownership loop.

    The existing HTTP client stages CPU adapter tensors before it dispatches a
    gradient job. This adapter keeps those tensors on CPU until the Worker has
    drained and the persistent model owner has woken, then installs the exact
    leased version, computes a CPU contribution, and lets the lifecycle park
    the owner before the client uploads that contribution.
    """

    def __init__(
        self, owner: Any, cycle: SameNodeGradientCycle, *,
        compute_profile: Mapping[str, Any], switch_timing_dir: str | Path | None = None,
    ):
        if not isinstance(compute_profile, Mapping) or not compute_profile:
            raise ValueError("lifecycle gradient compute requires a non-empty compute profile")
        if not callable(getattr(owner, "wake", None)) or not callable(getattr(owner, "park", None)):
            raise TypeError("persistent gradient owner must provide wake() and park()")
        self.owner = owner
        self.cycle = cycle
        self._compute_profile = dict(compute_profile)
        self._switch_timing_dir = Path(switch_timing_dir) if switch_timing_dir else None
        self._state_lock = threading.Lock()
        self._staged_version: int | None = None
        self._staged_state: _OwnedAdapterSnapshot | None = None
        self._installed_policy_version: int | None = None
        self._job_active = False
        self._session_active = False
        self._closed = False
        self._gpu_switch_session_id: str | None = None
        self._gpu_switch_started_monotonic: float | None = None
        self._gpu_switch_started_unix: float | None = None
        self._gpu_switch_cold_start = False
        self._gpu_switch_cpu_prepared = False
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_error: BaseException | None = None
        self._loop_thread = threading.Thread(
            target=self._serve_loop,
            name="gradient-lifecycle-loop",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(timeout=10.0):
            raise RuntimeError("gradient lifecycle event loop did not start")
        if self._loop_error is not None:
            raise RuntimeError("gradient lifecycle event loop failed to start") from self._loop_error

    def _record_gpu_switch(
        self, direction: str, status: str, error: BaseException | None = None,
        phase_timing: JobPhaseTiming | None = None,
    ) -> None:
        """Persist one aggregate ownership transition next to executor evidence."""
        started = self._gpu_switch_started_monotonic
        started_unix = self._gpu_switch_started_unix
        session_id = self._gpu_switch_session_id
        if started is None or started_unix is None or session_id is None:
            return
        try:
            if self._switch_timing_dir is None:
                return
            from qwen_rl.runtime.manifest import write_json_atomic

            ended = time.monotonic()
            payload = {
                "schema": "qwen-rl-gpu-switch/v1",
                "session_id": session_id,
                "direction": direction,
                "status": status,
                "worker_ids": list(self.cycle.worker_ids),
                "cold_model_initialization": self._gpu_switch_cold_start,
                "cpu_model_preloaded": self._gpu_switch_cpu_prepared,
                "started_at_unix_seconds": started_unix,
            }
            if status != "in_progress":
                payload["ended_at_unix_seconds"] = time.time()
                payload["duration_seconds"] = max(0.0, ended - started)
            if error is not None:
                payload["error_type"] = type(error).__name__
            if phase_timing is not None:
                payload["phase_timing"] = phase_timing.snapshot()
            if direction == 'learner_to_worker':
                payload['compute_park_phase_timing'] = getattr(self.owner, 'last_park_timing', None)
                payload['residency_park_phase_timing'] = getattr(
                    getattr(self.owner, 'residency', None), 'last_park_timing', None,
                )
                if status != "in_progress":
                    host_cache = getattr(self.owner, 'last_host_cache_reclaim', None)
                    if isinstance(host_cache, Mapping):
                        payload['host_cache_reclaim'] = dict(host_cache)
            destination = self._switch_timing_dir / f"{session_id}-{direction}.json"
            write_json_atomic(destination, payload)
        except Exception:
            # Telemetry must not strand the GPU lifecycle after a successful fence.
            return

    def _begin_switch_record(self, *, cold_start: bool) -> None:
        self._gpu_switch_session_id = uuid.uuid4().hex
        self._gpu_switch_started_monotonic = time.monotonic()
        self._gpu_switch_started_unix = time.time()
        self._gpu_switch_cold_start = cold_start
        self._gpu_switch_cpu_prepared = bool(getattr(self.owner, "prepared", False))

    def _finish_switch_record(
        self, direction: str, error: BaseException | None = None,
        phase_timing: JobPhaseTiming | None = None,
    ) -> None:
        self._record_gpu_switch(
            direction, "failed" if error is not None else "succeeded", error, phase_timing,
        )
        self._gpu_switch_session_id = None
        self._gpu_switch_started_monotonic = None
        self._gpu_switch_started_unix = None
        self._gpu_switch_cold_start = False
        self._gpu_switch_cpu_prepared = False

    def _serve_loop(self) -> None:
        try:
            asyncio.set_event_loop(self._loop)
            self._loop_ready.set()
            self._loop.run_forever()
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        except BaseException as exc:
            self._loop_error = exc
            self._loop_ready.set()
        finally:
            self._loop.close()

    @property
    def compute_profile(self) -> dict[str, Any]:
        return dict(self._compute_profile)

    @property
    def current_policy_version(self) -> int | None:
        return self._installed_policy_version

    def can_help(self) -> bool:
        if self._session_active:
            return True
        return (self.cycle.state == "sampling"
                and (self.owner.loaded or bool(getattr(self.owner, "prepared", False)))
                and self.cycle.fleet.idle_for_help())

    def prepare_cpu_model(self) -> None:
        """Assemble weights in host RAM while paired Workers keep sampling."""
        prepare = getattr(self.owner, "prepare_cpu", None)
        if callable(prepare):
            prepare()

    def start_session(self) -> None:
        """Retain one explicitly fenced GPU across queued trajectory jobs."""
        if not self._session_active:
            phase_timing = JobPhaseTiming()
            self._begin_switch_record(cold_start=not self.owner.loaded)
            self._record_gpu_switch("worker_to_learner", "in_progress")
            try:
                self._submit(self.cycle._enter(phase_timing))
            except BaseException as exc:
                self._finish_switch_record("worker_to_learner", exc, phase_timing)
                raise
            self._session_active = True
            self._finish_switch_record("worker_to_learner", phase_timing=phase_timing)

    def end_session(self) -> None:
        self._end_session()

    def end_session_with_parked_callback(self, on_parked: Callable[[], Any]) -> None:
        """Attest GPU release before the independently fenced Worker rebuild."""
        self._end_session(on_parked=on_parked)

    def _end_session(self, *, on_parked: Callable[[], Any] | None = None) -> None:
        if self.cycle.state == "failed":
            raise GradientLifecycleError("failed compute session cannot attest GPU return")
        if self._session_active:
            if self.cycle.state != "gradient":
                raise GradientLifecycleError("compute session GPU ownership is uncertain")
            phase_timing = JobPhaseTiming()
            self._begin_switch_record(cold_start=False)
            self._record_gpu_switch("learner_to_worker", "in_progress")
            try:
                self._submit(self.cycle._exit(None, None, None, phase_timing, on_parked=on_parked))
            except BaseException as exc:
                self._finish_switch_record("learner_to_worker", exc, phase_timing)
                raise
            if self.cycle.state != "sampling":
                self._finish_switch_record("learner_to_worker", GradientLifecycleError(
                    "compute session did not restore its Worker",
                ), phase_timing)
                raise GradientLifecycleError("compute session did not restore its Worker")
            self._session_active = False
            self._finish_switch_record("learner_to_worker", phase_timing=phase_timing)

    @asynccontextmanager
    async def _job_scope(self, phase_timing):
        if self._session_active:
            if self.cycle.state != "gradient":
                raise GradientLifecycleError("retained compute session lost its GPU ownership")
            yield self.cycle
        else:
            async with self.cycle.timed(phase_timing) if phase_timing is not None else self.cycle:
                yield self.cycle

    def install_adapter_version(self, version: int, adapter_state: Mapping[str, Any]) -> None:
        """Stage the client-downloaded CPU state without waking the compute GPU."""
        if type(version) is not int or version < 0:
            raise ValueError("adapter version must be a non-negative integer")
        if not isinstance(adapter_state, Mapping) or not adapter_state:
            raise ValueError("adapter state must be a non-empty mapping")
        from .adapter_snapshot import _OwnedAdapterSnapshot
        staged = _OwnedAdapterSnapshot(version, adapter_state)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("lifecycle gradient compute is closed")
            if self._job_active:
                raise RuntimeError("cannot stage an adapter while gradient compute is active")
            self._staged_version = version
            self._staged_state = staged

    def _submit(self, coroutine):
        if threading.current_thread() is self._loop_thread:
            raise RuntimeError("gradient lifecycle methods cannot synchronously re-enter their event loop")
        with self._state_lock:
            if self._closed:
                raise RuntimeError("lifecycle gradient compute is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def compute_gradient_job(
        self,
        prepared_window: Any,
        job: Any,
        prepared_rows: Any,
        *,
        compute_profile: Mapping[str, Any],
    ) -> Any:
        return self._submit(self._compute_gradient_job(
            prepared_window, job, prepared_rows, compute_profile=compute_profile,
            phase_timing=None,
        ))

    def compute_gradient_job_with_timing(
        self,
        prepared_window: Any,
        job: Any,
        prepared_rows: Any,
        *,
        compute_profile: Mapping[str, Any],
        phase_timing: JobPhaseTiming,
    ) -> Any:
        """Internal client hook that records local lifecycle phases for this job."""
        return self._submit(self._compute_gradient_job(
            prepared_window, job, prepared_rows, compute_profile=compute_profile,
            phase_timing=phase_timing,
        ))

    async def _compute_gradient_job(
        self,
        prepared_window: Any,
        job: Any,
        prepared_rows: Any,
        *,
        compute_profile: Mapping[str, Any],
        phase_timing: JobPhaseTiming | None,
    ) -> Any:
        if dict(compute_profile) != self._compute_profile:
            raise GradientLifecycleError("gradient job compute profile differs from configured lifecycle profile")
        with self._state_lock:
            if self._job_active:
                raise GradientLifecycleError("a gradient lifecycle job is already active")
            self._job_active = True
        try:
            async with self._job_scope(phase_timing):
                compute = self.owner.engine
                async with measured_async(phase_timing, "compute_prepare_adapter_sync"):
                    if dict(compute.compute_profile) != self._compute_profile:
                        raise GradientLifecycleError("loaded model compute profile differs from the central profile")
                    if self._staged_version is not None:
                        if self._staged_version != job.policy_version or self._staged_state is None:
                            raise GradientLifecycleError("staged adapter identity differs from the leased gradient job")
                        compute._install_owned_adapter_version(self._staged_version, self._staged_state)
                        with self._state_lock:
                            self._installed_policy_version = self._staged_version
                            self._staged_version = None
                            self._staged_state = None
                    if self._installed_policy_version != job.policy_version:
                        raise GradientLifecycleError("no exact adapter version is staged or installed for this job")
                async with measured_async(phase_timing, "gradient_job"):
                    result = compute.compute_gradient_job(
                        prepared_window,
                        job,
                        prepared_rows,
                        compute_profile=self._compute_profile,
                    )
                if getattr(compute.trainer, "_gradient_compute_active", False):
                    raise GradientLifecycleError("Trainer retained active gradient work after contribution copy")
                return result
        finally:
            with self._state_lock:
                self._job_active = False

    def close(self) -> None:
        """Stop the owned lifecycle loop after the client joins its active job."""
        with self._state_lock:
            if self._job_active:
                raise RuntimeError("cannot close lifecycle gradient compute during an active job")
            if self._closed:
                return
        cleanup_error = None
        try:
            self.end_session()
        except BaseException as exc:
            cleanup_error = exc
        with self._state_lock:
            self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=30.0)
        if self._loop_thread.is_alive():
            raise RuntimeError("gradient lifecycle event loop did not stop")
        if cleanup_error is not None:
            raise cleanup_error

    shutdown = close
