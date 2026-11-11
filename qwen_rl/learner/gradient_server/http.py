"""Authenticated HTTP scheduler for immutable gradient jobs and safetensors results."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Mapping

from qwen_rl.learner.gradient_transport import tensor_codec

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from qwen_rl.learner.gradient_server.store import (
    GradientPayloadTooLarge,
    GradientMemoryPressure,
    GradientStoreError,
    SQLiteGradientJobStore,
    StaleGradientLease,
)


_LOGGER = logging.getLogger(__name__)

_TENSOR_TRANSPORT_HEADER = "X-Tensor-Transport"
_POLICY_TRANSPORT_LOCK = threading.Lock()


def _encode_policy_payload(
    store: SQLiteGradientJobStore,
    path: Path,
    lineage_id: str,
    engine_epoch: str,
    version: int,
    max_payload_bytes: int,
) -> bytes:
    cache_root = store.artifact_dir / "policy-transport"
    identity_root = cache_root / lineage_id / engine_epoch
    destination = identity_root / f"version-{version:06d}.tensor-payload"
    with _POLICY_TRANSPORT_LOCK:
        for directory in (cache_root, cache_root / lineage_id, identity_root):
            if directory.is_symlink():
                raise GradientStoreError("policy tensor transport cache may not use symlink directories")
            directory.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise GradientStoreError("policy tensor transport cache entry may not be a symlink")
        _, _, _, _, max_encoded_size = tensor_codec()
        max_wire_bytes = max_encoded_size(max_payload_bytes)
        if destination.is_file():
            size = destination.stat().st_size
            if size < 1 or size > max_wire_bytes:
                raise GradientPayloadTooLarge("cached compressed policy exceeds max_payload_bytes")
            return destination.read_bytes()
        if path.stat().st_size > max_payload_bytes:
            raise GradientPayloadTooLarge("canonical training state exceeds max_payload_bytes")
        encode, _, _, _, max_encoded_size = tensor_codec()
        try:
            packed = encode(path.read_bytes(), safetensors=True)
        except ValueError as exc:
            raise GradientStoreError("canonical policy tensor payload could not be losslessly framed") from exc
        if len(packed) > max_encoded_size(path.stat().st_size):
            raise GradientPayloadTooLarge("compressed canonical training state exceeds max_payload_bytes")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}-", suffix=".part", dir=identity_root,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(packed)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(identity_root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        versions = sorted(
            (int(candidate.name[len("version-"):-len(".tensor-payload")]), candidate)
            for candidate in identity_root.glob("version-*.tensor-payload")
            if candidate.is_file() and not candidate.is_symlink()
        )
        for _, old_path in versions[:-2]:
            old_path.unlink(missing_ok=True)
        return packed


@dataclass(frozen=True)
class ExecutorCredential:
    token: str = field(repr=False)
    capabilities: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("executor capability token must be a nonempty string")
        if not self.capabilities or any(not isinstance(value, str) or not value for value in self.capabilities):
            raise ValueError("executor capabilities must be an explicit nonempty set")


def _credentials(raw: Mapping[str, Any]) -> dict[str, ExecutorCredential]:
    result: dict[str, ExecutorCredential] = {}
    for executor_id, value in raw.items():
        if isinstance(value, ExecutorCredential):
            credential = value
        elif isinstance(value, Mapping):
            capabilities = value.get("capabilities")
            token = value.get("token")
            if not isinstance(capabilities, (list, tuple, set, frozenset)):
                raise ValueError(f"executor {executor_id!r} must declare an explicit capabilities list")
            if not isinstance(token, str):
                raise ValueError(f"executor {executor_id!r} token must be a string")
            credential = ExecutorCredential(token, frozenset(capabilities))
        else:
            raise TypeError("executor_allowlist values must be ExecutorCredential or mappings")
        result[str(executor_id)] = credential
    if not result:
        raise ValueError("executor_allowlist must contain at least one executor")
    return result


def _bearer(value: str | None) -> str:
    if not isinstance(value, str) or not value.startswith("Bearer "):
        return ""
    return value[7:]


def _authorize(
    allowlist: Mapping[str, ExecutorCredential], executor_id: str, authorization: str | None,
    capability: str,
) -> ExecutorCredential:
    credential = allowlist.get(executor_id)
    presented = _bearer(authorization)
    expected = credential.token if credential is not None else "\x00" * 32
    valid = hmac.compare_digest(
        hashlib.sha256(presented.encode("utf-8")).digest(),
        hashlib.sha256(expected.encode("utf-8")).digest(),
    )
    if credential is None or not valid:
        raise HTTPException(status_code=401, detail="invalid executor capability token")
    if capability not in credential.capabilities:
        raise HTTPException(status_code=403, detail=f"executor lacks capability {capability}")
    return credential


def _lease_headers(lease_id: str | None, lease_epoch: int | None) -> tuple[str, int]:
    if not lease_id or type(lease_epoch) is not int or lease_epoch < 1:
        raise HTTPException(status_code=400, detail="lease fencing headers are required")
    return lease_id, lease_epoch


async def _call_provider(provider: Callable[..., Any], *args: Any) -> Any:
    if inspect.iscoroutinefunction(provider):
        return await provider(*args)
    value = await _run_owned_thread(provider, *args)
    if inspect.isawaitable(value):
        return await value
    return value


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, GradientMemoryPressure):
        return HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"})
    if isinstance(exc, GradientPayloadTooLarge):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, StaleGradientLease):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, TypeError, GradientStoreError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=503, detail="gradient scheduler operation failed")


def _policy_files(policy_provider: Callable[[str, str, int], str | Path] | None,
                  lineage_id: str, engine_epoch: str, version: int) -> tuple[Path, Path | None, Path]:
    if policy_provider is None:
        raise HTTPException(status_code=404, detail="canonical training state is unavailable")
    try:
        provided = Path(policy_provider(lineage_id, engine_epoch, version))
        root = provided.resolve(strict=True)
        if root.is_dir():
            config_path = root / "adapter_config.json"
            state_path = root / "adapter_state.safetensors"
            if (not config_path.is_file() or not state_path.is_file()
                    or config_path.is_symlink() or state_path.is_symlink()):
                raise ValueError("training-state bundle requires regular adapter_config.json and adapter_state.safetensors")
            for child in (config_path.resolve(strict=True), state_path.resolve(strict=True)):
                if not child.is_relative_to(root):
                    raise ValueError("training-state bundle file escapes its immutable version directory")
            return state_path.resolve(), config_path.resolve(), root
        if root.suffix != ".safetensors" or not root.is_file():
            raise ValueError("policy_provider must return a bundle directory or safetensors file")
        return root, None, root.parent
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"canonical training state is unavailable: {exc}") from exc


def _validate_policy_file(path: Path, window: Any, max_payload_bytes: int) -> None:
    if path.stat().st_size > max_payload_bytes:
        raise HTTPException(status_code=413, detail="canonical training state exceeds max_payload_bytes")
    try:
        from ..training_state import validate_training_state
        expected = {parameter.name:tuple(parameter.shape) for parameter in window.parameter_layout}
        validate_training_state(path, expected)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"canonical training-state validation failed: {exc}") from exc


def _timing_mark(core: Any) -> Mapping[str, Any] | None:
    mark = getattr(core, "timing_mark", None)
    return mark() if callable(mark) else None


def _record_timing_span(
    core: Any,
    window_id: str,
    attempt_id: str,
    name: str,
    start: Mapping[str, Any] | None,
) -> None:
    record = getattr(core, "record_timing_span", None)
    if callable(record) and start is not None:
        record(window_id, attempt_id, name, start, persist=False)


async def _run_owned_thread(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Join a central optimizer thread before propagating cancellation to its caller."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while not task.done():
        try:
            await asyncio.wait({task}, timeout=0.025)
        except asyncio.CancelledError:
            cancelled = True
            continue
        except BaseException:
            break
    if cancelled:
        raise asyncio.CancelledError
    return task.result()


def _note_timing(core: Any, window_id: str, attempt_id: str, event: str) -> None:
    note = getattr(core, "note_timing_event", None)
    if callable(note):
        note(window_id, attempt_id, event, persist=False)


async def _persist_timing(core: Any, window_id: str, attempt_id: str) -> None:
    persist = getattr(core, "persist_timing_receipt", None)
    if callable(persist):
        await _run_owned_thread(persist, window_id, attempt_id)


async def _apply_window_if_ready(store: SQLiteGradientJobStore, core: Any,
                                window_id: str, attempt_id: str,
                                lock: asyncio.Lock | None = None,
                                last_contribution_candidate: bool = False) -> dict[str, Any] | None:
    # Staged local means have no optimizer authority. The central engine alone
    # seals a complete round and promotes weighted contributions to its window.
    staged_window = await _run_owned_thread(store.load_window, window_id, attempt_id)
    if (staged_window.update_context or {}).get("gradient_scope") == "local_group_mean_staging/v1":
        return None
    use_lock = lock or asyncio.Lock()
    lock_wait_started = _timing_mark(core)
    contributions = None
    async with use_lock:
        _record_timing_span(core, window_id, attempt_id, "http_apply_lock_wait", lock_wait_started)
        try:
            accepted_count_started = _timing_mark(core)
            ready, total = await _run_owned_thread(store.accepted_count, window_id, attempt_id)
            _record_timing_span(core, window_id, attempt_id, "store_accepted_count", accepted_count_started)
            if total == 0 or ready != total:
                return None
            status_started = _timing_mark(core)
            status = await _run_owned_thread(store.window_status, window_id, attempt_id)
            _record_timing_span(core, window_id, attempt_id, "store_window_status_before_apply", status_started)
            if status["state"] == "applied":
                return status["receipt"]
            if status["state"] == "applying":
                result_receipt = getattr(core, "result_receipt", None)
                if callable(result_receipt):
                    committed = await _run_owned_thread(result_receipt, window_id, attempt_id)
                    if committed is not None:
                        return await _run_owned_thread(
                            store.mark_window_applied, window_id, attempt_id, committed,
                        )
                inspected = await _run_owned_thread(store.inspect_window_for_recovery, window_id, attempt_id)
                if inspected["state"] != "materialized":
                    raise GradientStoreError("volatile gradients lost during apply; managed checkpoint recovery required")
            if last_contribution_candidate:
                promote = getattr(core, "promote_timing_event", None)
                if callable(promote):
                    promote(
                        window_id, attempt_id,
                        "contribution_database_commit_observed",
                        "last_contribution_accepted",
                    )
                else:
                    _note_timing(core, window_id, attempt_id, "last_contribution_accepted")
            _note_timing(core, window_id, attempt_id, "all_contributions_accepted")
            _note_timing(core, window_id, attempt_id, "window_apply_claim_started")
            claim_started = _timing_mark(core)
            claimed = await _run_owned_thread(store.begin_window_apply, window_id, attempt_id)
            _record_timing_span(core, window_id, attempt_id, "store_begin_window_apply", claim_started)
            if not claimed:
                status_started = _timing_mark(core)
                status = await _run_owned_thread(store.window_status, window_id, attempt_id)
                _record_timing_span(core, window_id, attempt_id, "store_window_status_after_claim_race", status_started)
                return status["receipt"]
            _note_timing(core, window_id, attempt_id, "window_apply_claimed")
            _note_timing(core, window_id, attempt_id, "stored_window_load_started")
            window_load_started = _timing_mark(core)
            window = await _run_owned_thread(store.load_window, window_id, attempt_id)
            _record_timing_span(core, window_id, attempt_id, "store_load_window", window_load_started)
            _note_timing(core, window_id, attempt_id, "stored_window_loaded")
            _note_timing(core, window_id, attempt_id, "contribution_files_load_started")
            contributions_load_started = _timing_mark(core)
            contributions = await _run_owned_thread(store.load_contributions, window_id, attempt_id)
            _record_timing_span(core, window_id, attempt_id, "store_load_contributions", contributions_load_started)
            _note_timing(core, window_id, attempt_id, "contribution_files_loaded")
            receipt = await _run_owned_thread(core.apply_window, window, contributions)
            _note_timing(core, window_id, attempt_id, "core_apply_returned")
            _note_timing(core, window_id, attempt_id, "window_database_commit_started")
            mark_applied_started = _timing_mark(core)
            applied = await _run_owned_thread(store.mark_window_applied, window_id, attempt_id, receipt)
            _record_timing_span(core, window_id, attempt_id, "store_mark_window_applied", mark_applied_started)
            _note_timing(core, window_id, attempt_id, "window_database_applied")
            await _persist_timing(core, window_id, attempt_id)
            return applied
        except Exception as exc:
            try:
                await _run_owned_thread(store.mark_window_error, window_id, attempt_id, str(exc))
            except Exception:
                pass
            raise
        finally:
            if contributions is not None:
                # Do not retain decoded tensors in a completed request or its traceback.
                contributions.clear()


async def resume_ready_windows(store: SQLiteGradientJobStore, core: Any) -> None:
    """Re-drive complete windows after app startup; core idempotency resolves commit gaps."""
    status = await _run_owned_thread(store.status)
    for window in status["windows"]:
        if window["state"] not in {"pending", "applying"}:
            continue
        try:
            await _apply_window_if_ready(store, core, window["window_id"], window["attempt_id"])
        except Exception:
            # Preserve failed/pending state; the caller can inspect it and the core's
            # checkpoint fence prevents an uncertain optimizer step from replaying.
            continue


def create_router(
    store: SQLiteGradientJobStore,
    core: Any,
    executor_allowlist: Mapping[str, Any],
    *,
    executor_worker_ids: Mapping[str, str] | None = None,
    policy_provider: Callable[[str, str, int], str | Path] | None = None,
    current_policy_provider: Callable[[], Mapping[str, Any] | None] | None = None,
    worker_sync_provider: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    worker_cohorts: Mapping[str, list[str] | tuple[str, ...] | set[str]] | None = None,
    max_payload_bytes: int | None = None,
    prefix: str = "/v1",
    local_executor_ids: tuple[str, ...] = (),
) -> APIRouter:
    """Build routes that can be mounted into the Learner app at `/gradient`."""
    if not isinstance(store, SQLiteGradientJobStore):
        raise TypeError("store must be a SQLiteGradientJobStore")
    if not callable(getattr(core, "apply_window", None)):
        raise TypeError("core must implement apply_window(window, contributions)")
    if not prefix.startswith("/"):
        raise ValueError("router prefix must begin with '/'")
    allowlist = _credentials(executor_allowlist)
    store.bind_identity(core.lineage_id, core.engine_epoch)
    claim_executors = {
        executor_id for executor_id, credential in allowlist.items()
        if "claim" in credential.capabilities
    }
    sampling_only = getattr(store, "sampling_only_owner_ids", frozenset())
    if (executor_worker_ids is None
            or (set(executor_worker_ids) - sampling_only) | set(store.remote_executor_ids) != claim_executors
            or sampling_only & claim_executors
            or set(executor_worker_ids) & set(store.remote_executor_ids)):
        raise ValueError("claim executors must be paired or explicitly dedicated remote executors")
    owner_worker_map = store.bind_executor_worker_ids(executor_worker_ids)
    cohorts: dict[str, frozenset[str]] = {}
    for executor_id, worker_ids in (worker_cohorts or {}).items():
        if executor_id not in allowlist:
            raise ValueError("worker_cohorts may name only allowlisted executors")
        if not isinstance(worker_ids, (list, tuple, set, frozenset)):
            raise TypeError("each worker cohort must be an explicit worker ID sequence")
        values = tuple(worker_ids)
        if (any(not isinstance(worker_id, str) or not worker_id for worker_id in values)
                or len(values) != len(set(values))):
            raise ValueError("worker cohort IDs must be unique nonempty strings")
        cohorts[executor_id] = frozenset(values)
    payload_limit = store.max_payload_bytes if max_payload_bytes is None else max_payload_bytes
    if type(payload_limit) is not int or payload_limit <= 0:
        raise ValueError("max_payload_bytes must be a positive integer")
    payload_limit = min(payload_limit, store.max_payload_bytes)
    router = APIRouter(prefix=prefix)
    local_ids = frozenset(local_executor_ids)
    if not local_ids <= claim_executors:
        raise ValueError("local gradient transport names an unknown executor")
    router.gradient_ipc = None
    ipc_loop = None
    if local_ids:
        from qwen_rl.learner.gradient_ipc import LocalGradientServer, SharedGradientPayload

        def accept_shared(request, fd):
            executor_id = request["executor_id"]
            if executor_id not in local_ids:
                raise HTTPException(status_code=403, detail="executor has no local gradient transport")
            _authorize(allowlist, executor_id, "Bearer " + request["token"], "contribution.upload")
            lease_id, epoch = _lease_headers(request["lease_id"], request["lease_epoch"])
            stage = store.begin_contribution(executor_id, request["job_id"], lease_id, epoch, None)
            if isinstance(stage, dict):
                return {"job_receipt": stage}
            mapped = None
            descriptor = None
            try:
                _, window = store.expected_job(executor_id, request["job_id"], lease_id, epoch)
                descriptor = os.dup(fd)
                mapped = SharedGradientPayload.receive(descriptor, request["contribution"],
                                                       window.parameter_layout, payload_limit)
                descriptor = None
                accepted = store.finish_shared_contribution(stage, mapped)
                mapped = None
                _note_timing(core, accepted["window_id"], accepted["attempt_id"],
                             "contribution_database_commit_observed")
                return {"job_receipt": accepted}
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if mapped is not None:
                    mapped.close()
                store.abort_contribution(stage)

        router.gradient_ipc = LocalGradientServer(accept_shared, workers=len(local_ids))
    apply_locks: dict[tuple[str, str], asyncio.Lock] = {}
    upload_slot = asyncio.Semaphore(1)

    async def apply_ready_window(
        window_id: str,
        attempt_id: str,
        *,
        last_contribution_candidate: bool = False,
    ) -> dict[str, Any] | None:
        lock = apply_locks.setdefault((window_id, attempt_id), asyncio.Lock())
        return await _apply_window_if_ready(
            store, core, window_id, attempt_id, lock,
            last_contribution_candidate=last_contribution_candidate,
        )

    if router.gradient_ipc is not None:
        def notify_shared(response):
            accepted = response["job_receipt"]
            if ipc_loop is not None and ipc_loop.is_running() and not ipc_loop.is_closed():
                # The socket worker joins this owned operation before shutdown.
                # Client completion retries use the same per-window apply lock.
                future = asyncio.run_coroutine_threadsafe(
                    apply_ready_window(accepted["window_id"], accepted["attempt_id"],
                                       last_contribution_candidate=True), ipc_loop,
                )
                router.gradient_ipc.join_notification(future)
        router.gradient_ipc.after_accept = notify_shared

    @router.post("/executors/{executor_id}/claim")
    async def claim(executor_id: str, request: Request, authorization: str | None = Header(default=None)):
        nonlocal ipc_loop
        ipc_loop = asyncio.get_running_loop()
        _authorize(allowlist, executor_id, authorization, "claim")
        try:
            payload = await request.body()
            options = json.loads(payload) if payload else {}
            if (not isinstance(options, dict) or set(options) - {"help_ready", "session_id", "pending_upload"}
                    or type(options.get("help_ready", False)) is not bool):
                raise ValueError("claim options are malformed")
            claim_value = await _run_owned_thread(store.claim, executor_id, **options)
        except Exception as exc:
            raise _http_error(exc) from exc
        if claim_value is None:
            return {"status": "idle", "retry_after_seconds": 1.0}
        return {
            **({"local_contribution_transport": {
                **router.gradient_ipc.descriptor(), "supports_rowbundle_path": True,
            }}
               if executor_id in local_ids else {}),
            "status": "leased",
            "executor_id": executor_id,
            "lineage_id": claim_value.lineage_id,
            "engine_epoch": claim_value.engine_epoch,
            "source_worker_id": claim_value.source_worker_id,
            "group_size": store.group_size,
            "lease_id": claim_value.lease_id,
            "lease_epoch": claim_value.lease_epoch,
            "lease_seconds": claim_value.lease_seconds,
            "job": json.loads(claim_value.job.to_json()),
            "prepared_window": json.loads(claim_value.window.to_json()),
            "rowbundle": {"semantic_hash": claim_value.rowbundle_hash, "size": claim_value.rowbundle_size},
            "compute_profile": claim_value.compute_profile,
        }

    @router.post("/executors/{executor_id}/session/release")
    async def release_session(executor_id: str, request: Request, authorization: str | None = Header(default=None)):
        _authorize(allowlist, executor_id, authorization, "claim")
        try:
            payload = await request.json()
            if not isinstance(payload, dict) or set(payload) != {"session_id"}:
                raise ValueError("session release identity is malformed")
            await _run_owned_thread(store.release_compute_session, executor_id, payload["session_id"])
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"status": "released"}

    @router.post("/executors/{executor_id}/session/parked")
    async def park_session(executor_id: str, request: Request, authorization: str | None = Header(default=None)):
        _authorize(allowlist, executor_id, authorization, "claim")
        try:
            payload = await request.json()
            if not isinstance(payload, dict) or set(payload) != {"session_id"}:
                raise ValueError("session parked identity is malformed")
            await _run_owned_thread(store.park_compute_session, executor_id, payload["session_id"])
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"status": "parked"}

    @router.get("/executors/{executor_id}/jobs/{job_id}/rowbundle/local")
    async def local_rowbundle(
        executor_id: str,
        job_id: str,
        authorization: str | None = Header(default=None),
        x_lease_id: str | None = Header(default=None),
        x_lease_epoch: int | None = Header(default=None),
    ):
        _authorize(allowlist, executor_id, authorization, "rowbundle.read")
        if executor_id not in local_ids:
            raise HTTPException(status_code=403, detail="executor has no local rowbundle transport")
        lease_id, epoch = _lease_headers(x_lease_id, x_lease_epoch)
        try:
            path, semantic_hash = await _run_owned_thread(
                store.rowbundle_path, executor_id, job_id, lease_id, epoch,
            )
            path = await _run_owned_thread(path.resolve, strict=True)
            size = (await _run_owned_thread(path.stat)).st_size
            return {
                "path": str(path),
                "size": size,
                "semantic_hash": semantic_hash,
                "job_id": job_id,
                "lease_id": lease_id,
                "lease_epoch": epoch,
                "namespace": router.gradient_ipc.namespace,
            }
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/executors/{executor_id}/jobs/{job_id}/rowbundle")
    async def rowbundle(
        executor_id: str,
        job_id: str,
        authorization: str | None = Header(default=None),
        x_lease_id: str | None = Header(default=None),
        x_lease_epoch: int | None = Header(default=None),
    ):
        _authorize(allowlist, executor_id, authorization, "rowbundle.read")
        lease_id, epoch = _lease_headers(x_lease_id, x_lease_epoch)
        try:
            path, semantic_hash = await _run_owned_thread(
                store.rowbundle_path, executor_id, job_id, lease_id, epoch,
            )
            job, _ = await _run_owned_thread(store.expected_job, executor_id, job_id, lease_id, epoch)
        except Exception as exc:
            raise _http_error(exc) from exc
        return FileResponse(
            path, media_type="application/json", filename=f"{job_id}.json",
            headers={"X-Rowbundle-Semantic-Hash": semantic_hash,
                     "X-Lineage-ID": job.lineage_id,
                     "X-Engine-Epoch": job.engine_epoch,
                     "X-Owner-Executor-ID": job.owner_executor_id,
                     "X-Source-Worker-ID": owner_worker_map[job.owner_executor_id]},
        )

    @router.post("/executors/{executor_id}/jobs/{job_id}/heartbeat")
    async def heartbeat(
        executor_id: str,
        job_id: str,
        authorization: str | None = Header(default=None),
        x_lease_id: str | None = Header(default=None),
        x_lease_epoch: int | None = Header(default=None),
    ):
        _authorize(allowlist, executor_id, authorization, "heartbeat")
        lease_id, epoch = _lease_headers(x_lease_id, x_lease_epoch)
        try:
            expires = await _run_owned_thread(store.heartbeat, executor_id, job_id, lease_id, epoch)
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"status": "leased", "lease_epoch": epoch, "lease_expires_at": expires}

    @router.get("/executors/{executor_id}/jobs/{job_id}/policy/{version}/config")
    async def policy_config(
        executor_id: str,
        job_id: str,
        version: int,
        authorization: str | None = Header(default=None),
        x_lease_id: str | None = Header(default=None),
        x_lease_epoch: int | None = Header(default=None),
    ):
        _authorize(allowlist, executor_id, authorization, "policy.read")
        lease_id, epoch = _lease_headers(x_lease_id, x_lease_epoch)
        try:
            job, _ = await _run_owned_thread(store.expected_job, executor_id, job_id, lease_id, epoch)
            if version != job.policy_version:
                raise StaleGradientLease("requested canonical policy version differs from the leased job")
            _, config_path, _ = await _run_owned_thread(
                _policy_files, policy_provider, job.lineage_id, job.engine_epoch, version,
            )
            if config_path is None:
                raise HTTPException(status_code=404, detail="training-state bundle has no adapter config")
            return FileResponse(config_path, media_type="application/json", filename="adapter_config.json",
                                headers={"X-Policy-Version": str(version),
                                         "X-Layout-Fingerprint": job.layout_fingerprint,
                                         "X-Lineage-ID": job.lineage_id,
                                         "X-Engine-Epoch": job.engine_epoch})
        except HTTPException:
            raise
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/executors/{executor_id}/jobs/{job_id}/policy/{version}/local")
    async def local_policy(executor_id: str, job_id: str, version: int,
                           authorization: str | None = Header(default=None),
                           x_lease_id: str | None = Header(default=None),
                           x_lease_epoch: int | None = Header(default=None)):
        _authorize(allowlist, executor_id, authorization, "policy.read")
        if executor_id not in local_ids:
            raise HTTPException(status_code=403, detail="executor has no local policy transport")
        lease_id, epoch = _lease_headers(x_lease_id, x_lease_epoch)
        try:
            job, _ = await _run_owned_thread(store.expected_job, executor_id, job_id, lease_id, epoch)
            if version != job.policy_version:
                raise StaleGradientLease("local policy differs from leased job version")
            state, config, _ = await _run_owned_thread(
                _policy_files, policy_provider, job.lineage_id, job.engine_epoch, version)
            return {"state_path": str(state), "config_path": str(config) if config else None,
                    "namespace": router.gradient_ipc.namespace}
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/executors/{executor_id}/jobs/{job_id}/policy/{version}/state")
    async def policy_state(
        executor_id: str,
        job_id: str,
        version: int,
        authorization: str | None = Header(default=None),
        x_lease_id: str | None = Header(default=None),
        x_lease_epoch: int | None = Header(default=None),
    ):
        _authorize(allowlist, executor_id, authorization, "policy.read")
        lease_id, epoch = _lease_headers(x_lease_id, x_lease_epoch)
        try:
            job, window = await _run_owned_thread(store.expected_job, executor_id, job_id, lease_id, epoch)
            if version != job.policy_version:
                raise StaleGradientLease("requested canonical policy version differs from the leased job")
            state_path, _, _ = await _run_owned_thread(
                _policy_files, policy_provider, job.lineage_id, job.engine_epoch, version,
            )
            await _run_owned_thread(_validate_policy_file, state_path, window, payload_limit)
            packed = await _run_owned_thread(
                _encode_policy_payload,
                store,
                state_path,
                job.lineage_id,
                job.engine_epoch,
                version,
                payload_limit,
            )
            _, _, content_type, transport_name, _ = tensor_codec()
            return Response(
                content=packed,
                media_type=content_type,
                headers={
                    "X-Policy-Version": str(version),
                    "X-Layout-Fingerprint": job.layout_fingerprint,
                    "X-Lineage-ID": job.lineage_id,
                    "X-Engine-Epoch": job.engine_epoch,
                    _TENSOR_TRANSPORT_HEADER: transport_name,
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.put("/executors/{executor_id}/jobs/{job_id}/contribution")
    async def contribution(
        executor_id: str,
        job_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
        x_lease_id: str | None = Header(default=None),
        x_lease_epoch: int | None = Header(default=None),
    ):
        _authorize(allowlist, executor_id, authorization, "contribution.upload")
        lease_id, epoch = _lease_headers(x_lease_id, x_lease_epoch)
        _, _, tensor_content_type, tensor_transport, max_encoded_size = tensor_codec()
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != tensor_content_type:
            raise HTTPException(status_code=415, detail="gradient upload requires the configured tensor transport content type")
        if request.headers.get(_TENSOR_TRANSPORT_HEADER.lower()) != tensor_transport:
            raise HTTPException(status_code=415, detail="gradient upload requires the configured tensor transport header")
        content_encoding = request.headers.get("content-encoding")
        if content_encoding is not None and content_encoding.strip().lower() != "identity":
            raise HTTPException(status_code=415, detail="gradient upload does not accept HTTP content encodings")
        raw_length = request.headers.get("content-length")
        try:
            content_length = int(raw_length) if raw_length is not None else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid Content-Length") from exc
        wire_limit = max_encoded_size(payload_limit)
        if content_length is not None and (content_length < 1 or content_length > wire_limit):
            raise HTTPException(status_code=413, detail="gradient contribution exceeds configured max_payload_bytes")
        try:
            async with upload_slot:
                stage_or_receipt = await _run_owned_thread(
                    store.begin_contribution, executor_id, job_id, lease_id, epoch, content_length,
                )
                if isinstance(stage_or_receipt, dict):
                    return {"job_receipt": stage_or_receipt,
                            "window": await _run_owned_thread(
                                store.window_status, stage_or_receipt["window_id"], stage_or_receipt["attempt_id"])}
                stage = stage_or_receipt
                try:
                    # One bounded RAM stream, including chunked requests; never spill to disk.
                    received = 0
                    async for chunk in request.stream():
                        if received + len(chunk) > wire_limit:
                            raise GradientPayloadTooLarge("gradient contribution exceeds configured max_payload_bytes")
                        await _run_owned_thread(stage.write, chunk, wire_limit)
                        received += len(chunk)
                    if content_length is not None and received != content_length:
                        raise GradientStoreError("Content-Length differs from the received tensor transport body")
                    if not received:
                        raise GradientStoreError("empty gradient tensor transport body is invalid")
                    accepted = await _run_owned_thread(store.finish_contribution, stage)
                    stage = None
                    _note_timing(
                        core, accepted["window_id"], accepted["attempt_id"],
                        "contribution_database_commit_observed",
                    )
                except BaseException:
                    if stage is not None:
                        await _run_owned_thread(store.abort_contribution, stage)
                    raise
            try:
                window_receipt = await apply_ready_window(
                    accepted["window_id"],
                    accepted["attempt_id"],
                    last_contribution_candidate=True,
                )
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"contribution accepted; central apply is pending/failed: {exc}") from exc
            status = await _run_owned_thread(store.window_status, accepted["window_id"], accepted["attempt_id"])
            return {"job_receipt": accepted, "window_receipt": window_receipt, "window": status}
        except HTTPException:
            raise
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.post("/executors/{executor_id}/jobs/{job_id}/contribution/complete")
    async def complete_contribution(executor_id: str, job_id: str,
                                    authorization: str | None = Header(default=None),
                                    x_lease_id: str | None = Header(default=None),
                                    x_lease_epoch: int | None = Header(default=None)):
        _authorize(allowlist, executor_id, authorization, "contribution.upload")
        lease_id, epoch = _lease_headers(x_lease_id, x_lease_epoch)
        try:
            accepted = await _run_owned_thread(store.receipt, executor_id, job_id, lease_id, epoch)
            if accepted is None:
                raise StaleGradientLease("shared gradient contribution has not been accepted")
            receipt = await apply_ready_window(accepted["window_id"], accepted["attempt_id"],
                                               last_contribution_candidate=True)
            status = await _run_owned_thread(store.window_status, accepted["window_id"], accepted["attempt_id"])
            return {"job_receipt": accepted, "window_receipt": receipt, "window": status}
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/executors/{executor_id}/jobs/{job_id}/receipt")
    async def receipt(
        executor_id: str,
        job_id: str,
        authorization: str | None = Header(default=None),
        x_lease_id: str | None = Header(default=None),
        x_lease_epoch: int | None = Header(default=None),
    ):
        _authorize(allowlist, executor_id, authorization, "receipt.read")
        lease_id, epoch = _lease_headers(x_lease_id, x_lease_epoch)
        try:
            job_receipt = await _run_owned_thread(store.receipt, executor_id, job_id, lease_id, epoch)
            if job_receipt is not None:
                window_id, attempt_id = job_receipt["window_id"], job_receipt["attempt_id"]
            else:
                job, _ = await _run_owned_thread(store.expected_job, executor_id, job_id, lease_id, epoch)
                window_id, attempt_id = job.window_id, job.attempt_id
            status = await _run_owned_thread(store.window_status, window_id, attempt_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        if job_receipt is None:
            return JSONResponse(status_code=202, content={"status": "leased", "window": status})
        return {"job_receipt": job_receipt, "window": status}

    @router.get("/executors/{executor_id}/status")
    async def status(executor_id: str, authorization: str | None = Header(default=None)):
        _authorize(allowlist, executor_id, authorization, "status.read")
        return await _run_owned_thread(store.status)

    @router.get("/executors/{executor_id}/policy/current")
    async def current_policy(executor_id: str, authorization: str | None = Header(default=None)):
        _authorize(allowlist, executor_id, authorization, "policy.read")
        if current_policy_provider is None:
            raise HTTPException(status_code=503, detail="published policy manifest provider is unavailable")
        try:
            manifest = await _call_provider(current_policy_provider)
        except Exception as exc:
            _LOGGER.exception("Published policy unavailable for executor %s", executor_id)
            raise HTTPException(status_code=503, detail="published policy is not fenced/available") from exc
        if (manifest is None or not isinstance(manifest, Mapping)
                or type(manifest.get("version")) is not int or manifest["version"] < 0):
            raise HTTPException(status_code=503, detail="published policy is not fenced/available")
        # Return the exact normal Learner publication manifest. The provider owns
        # BEPA fencing and provenance; this route does not rescan or hash files.
        return JSONResponse(
            content=dict(manifest),
            headers={"X-Lineage-ID": core.lineage_id, "X-Engine-Epoch": core.engine_epoch,
                     "X-Policy-Version": str(manifest.get("version", ""))},
        )

    @router.post("/executors/{executor_id}/workers/ensure")
    async def ensure_workers(
        executor_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        _authorize(allowlist, executor_id, authorization, "worker.sync")
        if worker_sync_provider is None or current_policy_provider is None or worker_cohorts is None:
            raise HTTPException(status_code=503, detail="selected Worker policy sync is unavailable")
        try:
            payload = await request.json()
            if not isinstance(payload, dict) or set(payload) != {
                "lineage_id", "engine_epoch", "policy_version", "worker_ids",
            }:
                raise ValueError("worker sync request identity/worker_ids are malformed")
            manifest = await _call_provider(current_policy_provider)
            if not isinstance(manifest, Mapping) or type(manifest.get("version")) is not int:
                raise HTTPException(status_code=503, detail="published policy is not fenced/available")
            if (payload["lineage_id"] != core.lineage_id or payload["engine_epoch"] != core.engine_epoch
                    or type(payload["policy_version"]) is not int
                    or payload["policy_version"] != manifest.get("version")):
                raise HTTPException(status_code=409, detail="requested Worker policy is not the current central lineage/version")
            allowed_workers = set(cohorts.get(executor_id, ()))
            requested_workers = payload["worker_ids"]
            if (not isinstance(requested_workers, list) or not requested_workers
                    or any(not isinstance(worker_id, str) for worker_id in requested_workers)
                    or len(requested_workers) != len(set(requested_workers))
                    or not set(requested_workers).issubset(allowed_workers)):
                raise HTTPException(status_code=403, detail="requested Worker IDs exceed the executor's configured cohort")
            result = await _call_provider(worker_sync_provider, executor_id, payload)
            latest_after_sync = await _call_provider(current_policy_provider)
            if (not isinstance(latest_after_sync, Mapping)
                    or dict(latest_after_sync) != dict(manifest)):
                raise HTTPException(status_code=409, detail="central served policy changed during Worker synchronization")
        except HTTPException:
            raise
        except Exception as exc:
            _LOGGER.exception("Selected Worker policy sync failed for executor %s", executor_id)
            raise HTTPException(status_code=503, detail="selected Worker policy sync failed") from exc
        if not isinstance(result, Mapping):
            raise HTTPException(status_code=503, detail="Worker sync provider returned an invalid acknowledgement")
        if set(result) != set(payload["worker_ids"]):
            raise HTTPException(status_code=503, detail="Worker sync acknowledgement does not cover requested IDs")
        expected_adapter_hash = manifest.get("source_adapter_sha256")
        expected_files = manifest.get("files")
        for worker_id, ack in result.items():
            if (not isinstance(ack, Mapping) or ack.get("worker_id") != worker_id
                    or ack.get("version") != payload["policy_version"]
                    or not isinstance(ack.get("adapter_hash"), str) or not ack["adapter_hash"]
                    or (expected_adapter_hash is not None and ack.get("adapter_hash") != expected_adapter_hash)
                    or (expected_files is not None and ack.get("files") != expected_files)):
                raise HTTPException(status_code=503, detail="Worker sync acknowledgement differs from the current policy")
        return {
            "lineage_id": core.lineage_id,
            "engine_epoch": core.engine_epoch,
            "policy_version": payload["policy_version"],
            "workers": dict(result),
        }

    return router


def create_app(
    store: SQLiteGradientJobStore,
    core: Any,
    executor_allowlist: Mapping[str, Any],
    *,
    executor_worker_ids: Mapping[str, str] | None = None,
    policy_provider: Callable[[str, str, int], str | Path] | None = None,
    current_policy_provider: Callable[[], Mapping[str, Any] | None] | None = None,
    worker_sync_provider: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    worker_cohorts: Mapping[str, list[str] | tuple[str, ...] | set[str]] | None = None,
    max_payload_bytes: int | None = None,
    prefix: str = "/gradient",
    local_executor_ids: tuple[str, ...] = (),
) -> FastAPI:
    """Create a standalone FastAPI app; runtime integration may mount `create_router`."""
    router = create_router(
        store, core, executor_allowlist, executor_worker_ids=executor_worker_ids,
        policy_provider=policy_provider,
        current_policy_provider=current_policy_provider,
        worker_sync_provider=worker_sync_provider,
        worker_cohorts=worker_cohorts,
        max_payload_bytes=max_payload_bytes, prefix="/v1", local_executor_ids=local_executor_ids,
    )
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if router.gradient_ipc is not None:
                await _run_owned_thread(router.gradient_ipc.close)

    app = FastAPI(title="Qwen Learner Gradient Scheduler", lifespan=lifespan)
    app.state.gradient_ipc = router.gradient_ipc
    app.include_router(router, prefix=prefix)
    return app
