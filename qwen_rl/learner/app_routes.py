"""Learner HTTP route groups for status, monitoring, and policy reads."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from ..contracts.worker import ADAPTER_FILES, NATIVE_ADAPTER_FILES
from ..transport.tensor_codec import (
    MAX_TENSOR_RAW_BYTES,
    TENSOR_PAYLOAD_CONTENT_TYPE,
    encode_tensor_payload,
    max_encoded_size,
)
from .monitoring import process_commit_view
from .policy_file_references import PolicyFileReferences


class _PolicyFileResponse(FileResponse):
    def __init__(self, *args: Any, release_reference: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._release_reference = release_reference
        self._reference_released = False

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if not self._reference_released:
                self._reference_released = True
                self._release_reference()


_T = TypeVar("_T")


async def _finish_cache_write(writer: Callable[[], _T]) -> _T:
    task = asyncio.create_task(asyncio.to_thread(writer))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Keep the policy-file reference held until the background writer can
        # no longer recreate a version directory after cleanup.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        try:
            task.result()
        except BaseException:
            pass
        raise



def attach_health_route(
    app: FastAPI,
    *,
    require_token: Callable[..., Any],
    worker_pool: Any,
    recovery_is_dirty: Callable[[], bool],
    checkpoint_save_in_progress: Callable[[dict[str, Any] | None], bool],
    engine_recovery_metadata: Callable[[], dict[str, Any] | None],
    recovery_summary: Callable[[], dict[str, Any] | None],
) -> None:
    cfg = app.state.config
    engine = app.state.engine

    @app.get("/health", dependencies=[Depends(require_token)])
    async def health() -> dict[str, Any]:
        evidence = getattr(engine, "evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        policy = app.state.policies.get(int(engine.policy_version))
        compute = (engine.compute_status if (cfg["quantization"]["mode"] == "torchao"
                   or cfg["learner"].get("parallelism", "single") != "single") else None)
        compute_ready = compute is None or compute.get("state") in {"ready", "updating", "parked"}
        payload = {
            "ready": bool(app.state.initial_policy_ready and policy is not None
                           and app.state.recovery_initialized
                           and (not recovery_is_dirty() or checkpoint_save_in_progress(engine_recovery_metadata()))
                           and compute_ready and getattr(getattr(app.state, "gpu_phase_controller", None), "state", None) != "failed"),
            "compute": compute,
            "role": "learner",
            "max_sequence_length": int(cfg["learner"]["max_sequence_length"]),
            "groups_per_update": int(cfg["learner"]["groups_per_update"]),
            "microbatch_size": int(cfg["learner"]["microbatch_size"]),
            "max_padded_tokens_per_microbatch": cfg["learner"].get("max_padded_tokens_per_microbatch"),
            "group_size": int(cfg["rollout"]["group_size"]),
            "concurrent_tasks": int(cfg["rollout"]["concurrent_tasks"]),
            "worker_episode_leases": worker_pool.active_counts,
            "worker_resource_admission": worker_pool.resource_admission,
            "quantization": evidence.get("quantization"),
            "policy_version": int(engine.policy_version),
            "serve_policy_version": app.state.serve_policy_version,
            "rollout_pipeline": "bounded_complete_group_stream",
            "rollout_stream": app.state.rollout_stream.snapshot() if app.state.rollout_stream else None,
            "seed_scheduler": (app.state.seed_scheduler.snapshot()
                               if app.state.seed_scheduler is not None else None),
            "max_policy_lag": 1,
            "base_fingerprint": (policy or {}).get("base_fingerprint"),
            "artifact_fingerprint": (policy or {}).get("artifact_fingerprint"),
            "evidence": evidence,
            "timings": app.state.timings["learner"],
        }
        recovery = recovery_summary()
        if recovery is not None:
            payload["recovery"] = recovery
        inference_pool = getattr(app.state, 'inference_pool', None)
        if inference_pool is not None:
            payload['inference_pool'] = inference_pool.snapshot()
        return payload


def attach_monitor_route(
    app: FastAPI,
    *,
    require_token: Callable[..., Any],
    worker_pool: Any,
    evaluation_coordinator: Any,
    recovery_summary: Callable[[], dict[str, Any] | None],
    alignment_scope: Callable[[Any], str],
) -> None:
    cfg = app.state.config
    engine = app.state.engine

    @app.get("/monitor", dependencies=[Depends(require_token)])
    async def monitor():
        """Compact live metadata only: no model forwards or GPU synchronization."""
        run = app.state.runs.get(app.state.active_run_id or app.state.latest_run_id) or {}
        groups = run.get("groups", [])
        scored = [g for g in groups if "rewards" in g]
        def variable(values):
            return len(values) > 1 and max(values) != min(values)
        rewards = dict(scored_groups=len(scored),
            native_nonzero_groups=sum(any(r != 0 for r in g["rewards"]) for g in scored),
            native_positive_groups=sum(any(r > 0 for r in g["rewards"]) for g in scored),
            nonpositive_filtered=sum(g.get("filter_reason") == "no_positive_reward" for g in groups),
            native_variable_groups=sum(variable(g["rewards"]) for g in scored),
            all_fail_groups=sum(bool(g.get("exact")) and not any(g["exact"]) for g in scored),
            replay_groups=sum(bool(g.get("replay_reservation")) for g in scored),
            mixed_variable_groups=sum(variable(g.get("training_rewards", [])) for g in scored),
            native_exact_episodes=sum(sum(g.get("exact", [])) for g in scored),
            native_scored_episodes=sum(len(g.get("exact", [])) for g in scored),
            zero_variance_filtered=sum(g.get("status") == "filtered_zero_variance" for g in groups),
            discarded_groups=sum(g.get("status") == "discarded" for g in groups))
        trainer = getattr(engine, "trainer", None)
        forwards = getattr(trainer, "telemetry", None)
        training = app.state.training_telemetry.snapshot()
        training["forward"] = forwards.snapshot() if forwards is not None else None
        # Keep the foreground Learner phase independent from BEPA's background
        # seed mode. The run-owned seed scheduler is reported under sampling.
        training["stage"] = (getattr(trainer, "failure_stage", "optimizer") if training["active_update"]
            else app.state.monitor_phase)
        training["process_committed"] = process_commit_view(training)
        training["run_committed"] = app.state.run_commit_view.snapshot(run, training)
        forward = training["forward"] or {}
        training["last_attempt"] = forward.get("last_attempt")
        progress = dict(forward.get("progress") or {}) if training["active_update"] else {}
        activity = app.state.foreground_activity.snapshot()
        if training["active_update"]:
            activity.update(state=training["stage"],
                            since_elapsed_seconds=progress.get("stage_elapsed_seconds"))
        else:
            progress.update(stage=training["stage"], current_microbatch=0,
                            planned_microbatches=None,
                            stage_elapsed_seconds=activity["since_elapsed_seconds"])
        training["progress"] = progress
        training["activity"] = activity
        training["external_stages"] = dict(app.state.external_stage_timings)
        training["quality"] = trainer.quality_snapshot() if trainer is not None else {}
        training["quality"]["alignment"] = run.get("pre_update_logprob_alignment")
        training["quality"]["alignment_scope"] = alignment_scope(
            run.get("pre_update_logprob_alignment")
        )
        handoffs = run.get("policy_handoffs", [])
        pending_handoff = next(
            (item for item in reversed(handoffs)
             if item.get("status") in {"pending", "syncing", "maintenance", "publishing"}),
            None,
        )
        seed_observability = run.get("bepa_seed_observability")
        if not isinstance(seed_observability, dict):
            seed_observability = {}
        prefetch = run.get("bepa_plan_prefetch")
        prefetch_summary = ({
            key: prefetch.get(key) for key in (
                "selected", "resolved", "remaining", "running", "pending_window",
                "scope_pending", "state_counts", "enqueue_calls",
            )
        } if isinstance(prefetch, dict) else None)
        return dict(schema="qwen-rl-monitor-v1", sampled_at=time.time(),
            gpu_phases=(app.state.gpu_phase_controller.snapshot()
                        if getattr(app.state, "gpu_phase_controller", None) is not None else None),
            runtime_run_id=os.environ.get("QWEN_RL_MANIFEST_RUN_ID"),
            telemetry_epoch=app.state.training_telemetry.epoch,
            policy_version=int(engine.policy_version), serve_policy_version=app.state.serve_policy_version,
            sampling=dict(active_groups=app.state.active_sampling_groups,
                          maximum_groups=int(cfg["rollout"]["concurrent_tasks"]),
                          group_size=int(cfg["rollout"]["group_size"]),
                          episode_leases=worker_pool.active_counts,
                          worker_resource_admission=worker_pool.resource_admission,
                          episode_capacity_per_worker=int(cfg["worker_inference"]["max_num_seqs"]),
                          stream=(app.state.rollout_stream.snapshot()
                                  if app.state.rollout_stream else None),
                          seed_scheduler=(app.state.seed_scheduler.snapshot()
                                          if app.state.seed_scheduler is not None
                                          else seed_observability.get("scheduler")),
                          plan_prefetch=prefetch_summary),
            training=training, rewards=rewards,
            run={key:run.get(key) for key in ("run_id", "kind", "status", "started_at",
                  "finished_at", "completed_updates", "requested_steps", "error")},
            policy_handoffs=handoffs, pending_policy_handoff=pending_handoff,
            learner=dict(learning_rate=cfg["learner"]["lr"],
                last_loss=getattr(trainer, "_last_trainer_metrics", {}).get("train_loss"),
                padding=getattr(trainer, "_last_padding_metrics", {}),
                groups_per_update=int(cfg["learner"]["groups_per_update"]),
                microbatch_size=int(cfg["learner"]["microbatch_size"]),
                max_padded_tokens_per_microbatch=cfg["learner"].get("max_padded_tokens_per_microbatch"),
                gradient_accumulation_steps=getattr(engine, "evidence", {}).get(
                    "last_update_gradient_accumulation_steps"
                ),
                microbatch_plan=getattr(engine, "evidence", {}).get("last_microbatch_plan", {}),
                last_gradient_norm=getattr(engine, "evidence", {}).get("last_update_gradient_norm_before_clip"),
                peak_allocated_mib=getattr(engine, "evidence", {}).get("last_update_peak_memory_allocated_mib")),
            alignment=run.get("pre_update_logprob_alignment"),
            policy_acks={key:value.get("version") for key,value in app.state.last_policy_acks.items()},
            recovery=recovery_summary(),
            evaluation=evaluation_coordinator.status(),
            inference_pool=(app.state.inference_pool.snapshot()
                            if getattr(app.state, 'inference_pool', None) is not None else None))


def attach_policy_routes(
    app: FastAPI,
    *,
    policy_read_authorizer: Callable[..., Any],
    latest_policy: Callable[[], dict[str, Any]],
    policy_file_references: PolicyFileReferences,
    tensor_encoder: Callable[..., bytes] = encode_tensor_payload,
) -> None:
    cfg = app.state.config
    encode_policy_tensor_payload = tensor_encoder

    @app.get("/policy", dependencies=[Depends(policy_read_authorizer)])
    async def get_policy(request: Request) -> dict[str, Any]:
        policy = latest_policy()
        prefix = str(request.base_url).rstrip("/")
        return {
            "version": policy["version"],
            "base_fingerprint": policy["base_fingerprint"],
            "artifact_fingerprint": policy["artifact_fingerprint"],
            "quantization": policy["quantization"],
            "source_adapter_sha256": policy["source_adapter_sha256"],
            "artifact_id": policy.get("artifact_id"),
            "files": [
                {"name": name, "sha256": item["sha256"], "size_bytes": item["size"],
                 "url": f"{prefix}/policy/{policy['version']}/{name}"}
                for name, item in sorted(policy["_files"].items())
            ],
        }

    native_policy_locks: dict[int, asyncio.Lock] = {}

    async def prepare_native_policy(version: int) -> dict[str, Any]:
        # Evaluation archives are pinned into the catalog independently of the
        # training publisher. Serve every Native manifest from the same immutable
        # directory that managed Workers use, including baseline v0 after resume.
        policy = policy_file_references.acquire(version, app.state.policies)
        if policy is None:
            raise HTTPException(status_code=404, detail="Unknown policy version")
        try:
            if policy.get("quantization", {}).get("mode") != "ternary":
                return policy
            from .served_policy import materialize_served_policy

            async with native_policy_locks.setdefault(version, asyncio.Lock()):
                policy = app.state.policies[version]
                directory = Path(cfg["run_dir"]) / "policies" / f"v{version:08d}"
                if all(entry["path"] == directory / name
                       for name, entry in policy["_files"].items()):
                    return policy
                published = {"adapter_hash": policy["source_adapter_sha256"],
                             "native": {"artifact_id": policy["artifact_id"]}}
                try:
                    files, _ = await _finish_cache_write(lambda: materialize_served_policy(
                        cfg["run_dir"], version, policy["_files"], published, "ternary", policy))
                except (OSError, ValueError, RuntimeError) as exc:
                    raise HTTPException(status_code=409, detail="Published Native policy unavailable") from exc
                policy = {**policy, "_files": files}
                app.state.policies[version] = policy
                return policy
        finally:
            policy_file_references.release(version)

    @app.get("/policy/{version}", dependencies=[Depends(policy_read_authorizer)])
    async def get_policy_version(version: int, request: Request) -> dict[str, Any]:
        policy = await prepare_native_policy(version)
        prefix = str(request.base_url).rstrip("/")
        return {
            "version": policy["version"],
            "base_fingerprint": policy["base_fingerprint"],
            "artifact_fingerprint": policy["artifact_fingerprint"],
            "quantization": policy["quantization"],
            "source_adapter_sha256": policy["source_adapter_sha256"],
            "artifact_id": policy.get("artifact_id"),
            "files": [
                {"name": name, "sha256": item["sha256"], "size_bytes": item["size"],
                 "url": f"{prefix}/policy/{policy['version']}/{name}"}
                for name, item in sorted(policy["_files"].items())
            ],
        }

    tensor_payload_locks: dict[str, asyncio.Lock] = {}

    async def cached_tensor_payload(policy: dict[str, Any], filename: str,
                                    entry: dict[str, Any]) -> Path:
        source_path = Path(entry["path"])
        raw_size = entry.get("size")
        version = policy.get("version")
        source_adapter = policy.get("source_adapter_sha256")
        artifact_id = policy.get("artifact_id")
        if (isinstance(raw_size, bool) or not isinstance(raw_size, int)
                or raw_size < 0 or raw_size > MAX_TENSOR_RAW_BYTES
                or isinstance(version, bool) or not isinstance(version, int) or version < 0
                or not isinstance(source_adapter, str) or not source_adapter):
            raise ValueError("published tensor identity or size is invalid")
        if source_path.is_symlink() or not source_path.is_file():
            raise OSError("published tensor file is unavailable")
        source_stat = source_path.stat()
        if source_stat.st_size != raw_size:
            raise ValueError("published tensor file size changed")
        identity = json.dumps(
            {
                "transport": "LORABLS2",
                "version": version,
                "source_adapter_sha256": source_adapter,
                "artifact_id": artifact_id,
                "filename": filename,
                "path": str(source_path.resolve()),
                "raw_size": raw_size,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        cache_key = hashlib.sha256(identity).hexdigest()
        cache_path = (
            Path(cfg["run_dir"]) / "policy-transport-cache" / f"v{version:08d}"
            / f"{filename}.{cache_key}.lorabls2"
        )
        lock = tensor_payload_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            try:
                cache_info = cache_path.lstat()
            except FileNotFoundError:
                cache_info = None
            if cache_info is not None:
                if cache_path.is_symlink() or not cache_path.is_file():
                    raise OSError("cached tensor payload is not a regular file")
                if not 0 < cache_info.st_size <= max_encoded_size(raw_size):
                    cache_path.unlink(missing_ok=True)
                    cache_info = None
            if cache_info is not None:
                return cache_path

            def create_cache_file() -> Path:
                raw_contents = source_path.read_bytes()
                if len(raw_contents) != raw_size:
                    raise ValueError("published tensor file size changed")
                encoded = encode_policy_tensor_payload(
                    raw_contents,
                    safetensors=filename == "adapter_model.safetensors",
                )
                if len(encoded) > max_encoded_size(raw_size):
                    raise ValueError("encoded tensor exceeds the transfer limit")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    temporary.write_bytes(encoded)
                    os.chmod(temporary, 0o600)
                    os.replace(temporary, cache_path)
                finally:
                    temporary.unlink(missing_ok=True)
                return cache_path

            return await _finish_cache_write(create_cache_file)

    @app.get("/policy/{version}/{filename}", dependencies=[Depends(policy_read_authorizer)])
    async def get_policy_file(version: int, filename: str):
        if filename not in ADAPTER_FILES | NATIVE_ADAPTER_FILES:
            raise HTTPException(status_code=404, detail="Unknown adapter file")
        policy = policy_file_references.acquire(version, app.state.policies)
        if policy is None:
            raise HTTPException(status_code=404, detail="Unknown policy version")
        try:
            entry = policy["_files"].get(filename)
            if entry is None or not entry["path"].is_file():
                raise HTTPException(status_code=404, detail="Adapter file unavailable")
            # Manifest sizes describe uncompressed files. Keep JSON sidecars as
            # JSON and frame tensor payloads for the Worker link.
            if filename.endswith(".json"):
                response = _PolicyFileResponse(
                    entry["path"], media_type="application/json", filename=filename,
                    headers={"Cache-Control": "public, max-age=31536000, immutable"},
                    release_reference=lambda: policy_file_references.release(version),
                )
                response.chunk_size = 1024 * 1024
                return response
            raw_size = entry.get("size")
            if (isinstance(raw_size, bool) or not isinstance(raw_size, int)
                    or raw_size < 0 or raw_size > MAX_TENSOR_RAW_BYTES):
                raise HTTPException(status_code=413, detail="Policy tensor exceeds the transfer size limit")
            try:
                payload_path = await cached_tensor_payload(policy, filename, entry)
            except OSError as exc:
                raise HTTPException(status_code=404, detail="Adapter file unavailable") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail="Published policy tensor is invalid") from exc
            response = _PolicyFileResponse(
                payload_path,
                media_type=TENSOR_PAYLOAD_CONTENT_TYPE,
                filename=filename,
                headers={"Cache-Control": "public, max-age=31536000, immutable"},
                release_reference=lambda: policy_file_references.release(version),
            )
            response.chunk_size = 1024 * 1024
            return response
        except BaseException:
            policy_file_references.release(version)
            raise
