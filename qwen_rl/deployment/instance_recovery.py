"""Offline, CPU-only recovery for a stopped managed central-gradient instance."""
from __future__ import annotations

import fcntl
import json
import os
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

_SCHEMA = "qwen-rl-managed-instance-recovery/v1"
_PLAN_SCHEMA = "qwen-rl-instance-resume-plan/v1"
_RESUME_TRANSACTION_SCHEMA = "qwen-rl-instance-resume-transaction/v1"
_INTENT_SCHEMA = "central-gradient-async-checkpoint-intent/v1"
_RECOVERABLE_PLAN_BLOCKERS = frozenset({
    "optimizer_window_prepared",
    "gradient_optimizer_window_pending",
    "checkpoint_metadata_blocked",
})
_VALIDATED_RECOVERY_CLASSIFICATIONS = frozenset({
    "clean_durable_checkpoint",
    "committed_checkpoint_reconciled",
    "prepared_window_cancelled_before_optimizer",
    "deferred_to_learner",
})


class _RecoveryTokenizer:
    """The central CPU optimizer only records tokenizer dimensions during setup."""

    vocab_size = 0

    def __len__(self) -> int:
        return 0


def _blocked(code: str, message: str, *, classification: str = "incompatible") -> dict[str, Any]:
    return {
        "schema": _SCHEMA,
        "status": "blocked",
        "classification": classification,
        "blockers": [{"code": code, "message": message}],
        "actions": [],
    }


def _semantics_deferred() -> dict[str, str]:
    return {
        "status": "not_performed_by_managed_recovery",
        "required_next_action": "normal_resume_endpoint_must_validate_full_source_semantics_before_resuming_a_pending_run",
    }


def _deployment(value: Any):
    from .instance_plan import InstanceDeployment, load_deployment

    if isinstance(value, InstanceDeployment):
        return value
    return load_deployment(Path(value))


def _instance_owner_lock(state_dir: Path):
    """Create/hold the same lock used by serve, leaving its inode in place."""
    state_dir = Path(state_dir)
    info = state_dir.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError("instance state directory is unavailable or unsafe")
    path = state_dir / "owner.lock"
    try:
        prior = path.lstat()
    except FileNotFoundError:
        prior = None
    if prior is not None and (not stat.S_ISREG(prior.st_mode) or stat.S_ISLNK(prior.st_mode)):
        raise RuntimeError("instance owner lock is not a regular file")
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or (prior is not None and (opened.st_dev, opened.st_ino) != (prior.st_dev, prior.st_ino))
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
                or stat.S_ISLNK(current.st_mode)):
            raise RuntimeError("instance owner lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _cpu_thread_limit():
    import torch

    previous = torch.get_num_threads()
    if previous > 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if previous > 1:
            torch.set_num_threads(previous)


def _resolved_recovery_dir(runtime: Any) -> Path:
    from ..runtime.config import _resolve_path

    return (_resolve_path(runtime.learner.recovery_dir)
            if runtime.learner.recovery_dir else runtime.run / "training")


def _plan_blockers(plan: Any) -> list[dict[str, str]]:
    if plan is None:
        return []
    if not isinstance(plan, Mapping) or plan.get("schema") != _PLAN_SCHEMA:
        return [{"code": "resume_plan_invalid", "message": "The supplied read-only resume plan is invalid."}]
    raw = plan.get("blockers", [])
    if not isinstance(raw, list):
        return [{"code": "resume_plan_invalid", "message": "The supplied read-only resume plan has invalid blockers."}]
    blockers = []
    for row in raw:
        if not isinstance(row, Mapping) or not isinstance(row.get("code"), str):
            return [{"code": "resume_plan_invalid", "message": "The supplied read-only resume plan has malformed blockers."}]
        code = row["code"]
        if code in _RECOVERABLE_PLAN_BLOCKERS or code == "selected_gpu_busy":
            continue
        message = row.get("message")
        blockers.append({
            "code": code,
            "message": message if isinstance(message, str) else "The read-only resume plan has an unresolved blocker.",
        })
    return blockers


def _pointer_matches(pointer: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    return all(pointer.get(key) == state.get(key) for key in (
        "generation", "snapshot_id", "policy_version", "optimizer_steps",
    ))


def _intent_disposition(root: Path, pointer: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Classify an async intent without interpreting optimizer tensors."""
    from .instance_status import _read_json

    path = root / "async-checkpoint-intent.json"
    intent, state = _read_json(path)
    if state == "missing":
        pending, pending_state = _read_json(root / "pending-attempt.json")
        if pending_state == "missing":
            return "none", None
        if pending_state != "ok" or not isinstance(pending, Mapping):
            return "invalid", None
        return "ambiguous_pending_fence", None
    if state != "ok" or not isinstance(intent, Mapping) or intent.get("schema") != _INTENT_SCHEMA:
        return "invalid", None
    source, target = intent.get("source"), intent.get("target")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        return "invalid", None
    if _pointer_matches(pointer, target):
        return "target", dict(intent)
    if _pointer_matches(pointer, source):
        return "source", dict(intent)
    return "neither", dict(intent)


def _checkpoint_preflight(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Validate bounded pointer/sidecar metadata before constructing the engine."""
    from .instance_status import _checkpoint_summary, _read_json

    summary, _optimizer = _checkpoint_summary(
        {"learner": {"recovery_dir": str(root)}},
        source="managed_recovery", process_state="stopped", owner_live=False,
    )
    if summary.get("state") not in {"metadata_consistent", "pending"}:
        return None, str(summary.get("state") or "unknown")
    pointer, pointer_state = _read_json(root / "latest.json")
    if pointer_state != "ok" or not isinstance(pointer, Mapping):
        return None, "checkpoint_pointer_invalid"
    return dict(pointer), None


def _engine_runtime(deployment: Any, runtime: Any) -> dict[str, Any]:
    """Project a runtime to the managed target's executor map without GPU discovery."""
    from ..runtime.config import _resolve_path

    data = runtime.as_dict()
    data["model_path"] = str(runtime.model)
    data["dataset_dir"] = str(runtime.dataset)
    data["run_dir"] = str(runtime.run)
    data["arc"]["contract_path"] = str(runtime.contract)
    if runtime.artifact_identity_path:
        data["artifact_identity_path"] = str(_resolve_path(runtime.artifact_identity_path))
    data["tokenizer_path"] = str(
        _resolve_path(runtime.tokenizer_path) if runtime.tokenizer_path else runtime.model
    )
    count = len(deployment.gpus)
    learner = data["learner"]
    learner["recovery_dir"] = str(_resolved_recovery_dir(runtime))
    # The offline Core restore must not invoke Worker residency drain/suspend hooks.
    learner["gpu_sharing"] = "dedicated"
    if learner.get("initial_adapter_path"):
        learner["initial_adapter_path"] = str(_resolve_path(learner["initial_adapter_path"]))
    executor_ids = [f"executor-{index}" for index in range(count)]
    worker_ids = [f"worker-{index}" for index in range(count)]
    helper_ids = set(runtime.rollout.helper_worker_ids)
    if set(worker_ids) & helper_ids:
        raise ValueError("Gateway helper IDs collide with generated managed Worker IDs")
    source_owner_ids = set(runtime.learner.gradient_executor_worker_ids.values())
    source_worker = next(worker for worker in runtime.workers if worker.id in source_owner_ids)
    from .workers import _auth_environment_name

    data["workers"] = [
        {
            "id": worker_id,
            "host": None,
            "port": source_worker.port + index,
            "gpu": gpu,
            "scope": "service",
            "site_id": source_worker.site_id,
            "node_id": source_worker.node_id,
            "endpoint": f"http://127.0.0.1:{source_worker.port + index}",
            "url_prefix": "",
            "auth_token_env": _auth_environment_name(worker_id),
            "boot_id": "",
            "telemetry_epoch": "",
            "model_sha256": "",
        }
        for index, (worker_id, gpu) in enumerate(zip(worker_ids, deployment.gpus, strict=True))
    ] + [worker for worker in data["workers"] if worker["id"] in helper_ids]
    learner["groups_per_update"] = count * runtime.learner.gradient_tasks_per_worker
    learner["groups_per_update_auto"] = True
    learner["gradient_executor_token_envs"] = {
        executor_id: f"QWEN_MANAGED_RECOVERY_{index}_TOKEN"
        for index, executor_id in enumerate(executor_ids)
    }
    learner["gradient_executor_worker_ids"] = dict(zip(executor_ids, worker_ids, strict=True))
    learner["gradient_remote_executor_ids"] = ()
    learner["sampling_only_worker_ids"] = {}
    learner["sampling_group_quotas"] = {}
    learner["sampling_shared_worker_pairs"] = {}
    learner["sampling_episode_slots"] = {}
    learner["gradient_local_executor_worker_ids"] = worker_ids
    learner["gradient_worker_sync_ids"] = worker_ids
    # No sampling or executor dispatch occurs during this operation.
    learner["gradient_local_work_stealing"] = False
    data["rollout"]["concurrent_tasks"] = max(data["rollout"]["concurrent_tasks"], count)
    return data


def _semantic_runtime_data(runtime: Any) -> dict[str, Any]:
    data = runtime.as_dict()
    data["model_path"] = str(runtime.model)
    data["dataset_dir"] = str(runtime.dataset)
    data["run_dir"] = str(runtime.run)
    data["arc"]["contract_path"] = str(runtime.contract)
    return data


def validate_pending_source_semantics(runtime: Any, core: Any) -> dict[str, Any]:
    """Compare current CPU-computable semantics with the restored Core run context."""
    core_status = core.status()
    contexts: list[tuple[str, Mapping[str, Any]]] = []
    last_context = core_status.get("last_update_context")
    if isinstance(last_context, Mapping):
        contexts.append(("last_update_context", last_context))
    pending = core_status.get("pending_window")
    if isinstance(pending, Mapping):
        window = core.prepared_window(pending["window_id"], pending["attempt_id"])
        if window is None:
            return {
                "status": "blocked", "state": "prepared_context_unavailable",
                "mismatch_keys": ["pending_window.update_context"], "matched_contexts": [],
            }
        if isinstance(window.update_context, Mapping):
            contexts.append(("pending_window.update_context", window.update_context))

    jobs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for label, context in contexts:
        logical_job = context.get("logical_job")
        if not isinstance(logical_job, Mapping):
            if isinstance(context.get("source_identity"), Mapping):
                return {
                    "status": "blocked", "state": "logical_run_context_invalid",
                    "mismatch_keys": [f"{label}.logical_job"], "matched_contexts": [],
                }
            continue
        run_id = logical_job.get("run_id")
        requested, completed = logical_job.get("requested_steps"), logical_job.get("completed_updates")
        if (not isinstance(run_id, str) or not run_id
                or type(requested) is not int or requested < 1
                or type(completed) is not int or not 0 <= completed <= requested):
            return {
                "status": "blocked", "state": "logical_run_context_invalid",
                "mismatch_keys": [f"{label}.logical_job.progress"], "matched_contexts": [],
            }
        jobs.append((label, context, logical_job))

    pending_jobs = [row for row in jobs if row[2]["completed_updates"] < row[2]["requested_steps"]]
    if not pending_jobs:
        return {
            "status": "passed", "state": "not_applicable",
            "reason": "no_pending_logical_run", "semantic_keys": [],
            "matched_contexts": [],
        }
    run_ids = {job.get("run_id") for _label, _context, job in pending_jobs}
    if len(run_ids) != 1:
        return {
            "status": "blocked", "state": "pending_run_identity_invalid",
            "mismatch_keys": ["run_id"], "matched_contexts": [],
        }
    run_id = next(iter(run_ids))

    try:
        from ..learner.recovery import preflight_compatibility
        from .instance_relocation import path_identity_matches

        runtime_data = _semantic_runtime_data(runtime)
        current_semantics = preflight_compatibility(runtime_data)["semantics"]
        current_dataset = runtime_data["dataset_dir"]
    except Exception as exc:
        return {
            "status": "blocked", "state": "current_semantics_unavailable",
            "error_type": type(exc).__name__, "matched_contexts": [],
        }

    matched: list[str] = []
    missing_contexts: list[str] = []
    for label, context in contexts:
        logical_job = context.get("logical_job")
        if not isinstance(logical_job, Mapping):
            continue
        if logical_job.get("run_id") != run_id:
            continue
        source_identity = context.get("source_identity")
        if not isinstance(source_identity, Mapping):
            missing_contexts.append(label)
            continue
        saved_dataset = source_identity.get("dataset_dir")
        try:
            dataset_matches = path_identity_matches(
                saved_dataset,
                current_dataset,
                runtime.learner.recovery_path_mappings,
            )
        except (OSError, TypeError, ValueError):
            dataset_matches = False
        if not dataset_matches:
            return {
                "status": "blocked", "state": "dataset_path_mismatch",
                "mismatch_keys": ["dataset_dir"], "matched_contexts": matched,
            }
        saved_semantics = {key: value for key, value in source_identity.items() if key != "dataset_dir"}
        if set(saved_semantics) != set(current_semantics):
            return {
                "status": "blocked", "state": "source_semantics_key_mismatch",
                "mismatch_keys": sorted(set(saved_semantics) ^ set(current_semantics)),
                "matched_contexts": matched,
            }
        mismatches = sorted(
            key for key, expected in current_semantics.items()
            if saved_semantics.get(key) != expected
        )
        if mismatches:
            return {
                "status": "blocked", "state": "source_semantics_mismatch",
                "mismatch_keys": mismatches, "matched_contexts": matched,
            }
        matched.append(label)
    if missing_contexts or not matched:
        return {
            "status": "blocked", "state": "pending_source_identity_missing",
            "mismatch_keys": missing_contexts or ["source_identity"],
            "matched_contexts": matched,
        }
    return {
        "status": "passed", "state": "verified",
        "semantic_keys": sorted(current_semantics),
        "matched_contexts": matched,
        "normal_resume_revalidation_required": True,
    }


@contextmanager
def _offline_executor_tokens(environment_names: Mapping[str, str]):
    """Supply ephemeral credentials only to construct the local, unused HTTP app."""
    previous = {name: os.environ.get(name) for name in environment_names.values()}
    try:
        for name in environment_names.values():
            if not os.environ.get(name):
                os.environ[name] = "offline-" + secrets.token_urlsafe(24)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _build_cpu_engine(deployment: Any, runtime: Any, *, process_owner_lock: Any):
    from qwen_rl.learner.app import _load_engine

    config = _engine_runtime(deployment, runtime)
    return _load_engine(
        config,
        recovery_semantics=None,
        tokenizer=_RecoveryTokenizer(),
        process_owner_lock=process_owner_lock,
        recovery_only=True,
        preserve_control_files=False,
    )


def _publish_cpu_evaluation_policy(engine: Any) -> dict[str, Any]:
    """Publish only the recovered inference policy needed by the CPU eval probe."""
    from ..learner.core import LearnerEngineCore
    from ..learner.app import _evaluation_publication_from_update

    publication_store = getattr(engine, "_publication_store", None)
    if publication_store is None:
        engine._configure_publication_state()
        publication_store = engine._publication_store
    version = int(engine.policy_version)
    if version not in publication_store.published_versions:
        restored_publication = engine.publication_dir / f"version-{version:06d}"
        if restored_publication.exists() or restored_publication.is_symlink():
            publication_store.adopt_restored(
                engine.model, engine.evidence, version, None,
            )
    raw_publication = LearnerEngineCore.publish(engine)
    return _evaluation_publication_from_update({"publication": raw_publication}, version)


def validate_post_recovery_plan(plan: Any, recovery: Any) -> dict[str, Any]:
    """Authorize only a CPU-validated existing logical state after metadata recheck."""
    if (not isinstance(plan, Mapping) or plan.get("schema") != _PLAN_SCHEMA
            or not isinstance(recovery, Mapping) or recovery.get("schema") != _SCHEMA
            or recovery.get("status") != "passed"
            or recovery.get("classification") not in _VALIDATED_RECOVERY_CLASSIFICATIONS):
        return _blocked("post_recovery_evidence_invalid", "Post-recovery plan or CPU restore evidence is invalid.")
    checkpoint = plan.get("checkpoint")
    logical = plan.get("logical_run")
    preserved = recovery.get("preserved")
    topology = plan.get("desired_topology")
    if not all(isinstance(item, Mapping) for item in (checkpoint, logical, preserved, topology)):
        return _blocked("post_recovery_evidence_incomplete", "Post-recovery checkpoint, logical-run, or topology evidence is incomplete.")
    optimizer_transaction_pending = checkpoint.get("optimizer_transaction_pending")
    if type(optimizer_transaction_pending) is not bool:
        return _blocked("post_recovery_transaction_state_unknown", "The checkpoint optimizer transaction state is unknown.")
    if (checkpoint.get("state") != "payload_metadata_inspected"
            or checkpoint.get("generation") != preserved.get("checkpoint_generation")
            or checkpoint.get("snapshot_id") != preserved.get("snapshot_id")
            or checkpoint.get("lineage_id") != preserved.get("lineage_id")
            or checkpoint.get("policy_version") != preserved.get("policy_version")
            or checkpoint.get("optimizer_steps") != preserved.get("optimizer_steps")):
        return _blocked("post_recovery_checkpoint_mismatch", "The inspected durable checkpoint differs from the CPU-restored optimizer state.")
    occupancy = topology.get("gpu_occupancy")
    if (topology.get("hardware_validation") != "passed"
            or not isinstance(occupancy, Mapping) or occupancy.get("state") != "idle"):
        return _blocked("post_recovery_topology_unready", "The target topology is not proven valid and idle.")

    state = logical.get("state")
    if state not in {"pending", "complete", "no_saved_logical_run"}:
        return _blocked("logical_run_state_unsupported", "The checkpoint logical-run state is not permitted for managed startup.")
    blockers = plan.get("blockers", [])
    if not isinstance(blockers, list):
        return _blocked("post_recovery_plan_invalid", "Post-recovery blockers are malformed.")
    if recovery.get("classification") == "deferred_to_learner":
        if state != "pending" or any(not isinstance(row, Mapping) or row.get("code") not in _RECOVERABLE_PLAN_BLOCKERS
               for row in blockers):
            return _blocked("post_recovery_plan_blocked", "A complete prepared window is accompanied by an incompatible blocker.")
        prepared = recovery.get("prepared_window")
        if (not isinstance(prepared, Mapping)
                or not isinstance(prepared.get("window_id"), str) or not prepared["window_id"]
                or not isinstance(prepared.get("attempt_id"), str) or not prepared["attempt_id"]
                or type(prepared.get("accepted_jobs")) is not int
                or type(prepared.get("total_jobs")) is not int
                or prepared["total_jobs"] < 1
                or prepared["accepted_jobs"] != prepared["total_jobs"]
                or recovery.get("validated_required_action") != "launch_normal_engine_to_commit_the_complete_prepared_window"):
            return _blocked("deferred_window_receipt_invalid", "The deferred prepared window lacks a complete accepted-window receipt.")
        if optimizer_transaction_pending and not any(
            isinstance(row, Mapping) and row.get("code") == "optimizer_window_prepared"
            for row in blockers
        ):
            return _blocked("deferred_window_plan_mismatch", "The plan does not identify the prepared optimizer window reported by CPU recovery.")
    elif optimizer_transaction_pending:
        return _blocked("post_recovery_transaction_pending", "A prepared optimizer transaction requires a fully accepted deferred-window receipt.")
    elif state == "no_saved_logical_run":
        if blockers:
            return _blocked("post_recovery_plan_blocked", "The existing checkpoint has unresolved blockers.")
    elif plan.get("status") != "passed" or blockers:
        return _blocked("post_recovery_plan_blocked", "Post-recovery plan still has unresolved blockers.")

    if state == "no_saved_logical_run":
        if preserved.get("run_id") is not None:
            return _blocked("genesis_run_context_mismatch", "The checkpoint has no logical run but the CPU-restored state reports one.")
        return {
            "schema": _SCHEMA,
            "status": "passed",
            "classification": "existing_checkpoint_without_run",
            "logical_run": {"state": state, "run_id": None},
            "next_action": "start_existing_checkpoint_without_submitting_a_run",
            "source_semantics_validation": "normal_app_startup_still_computes_full_semantics",
            "blockers": [],
        }
    if (logical.get("run_id") != preserved.get("run_id")
            or logical.get("requested_steps") != preserved.get("requested_steps")
            or logical.get("completed_updates") != preserved.get("completed_updates")
            or logical.get("remaining_steps") != preserved.get("remaining_steps")):
        return _blocked("logical_run_restore_mismatch", "The post-recovery logical run differs from the CPU-restored run context.")
    if state == "pending":
        if type(logical.get("remaining_steps")) is not int or logical["remaining_steps"] < 1:
            return _blocked("pending_run_progress_invalid", "The pending run has no positive remaining update count.")
        return {
            "schema": _SCHEMA,
            "status": "passed",
            "classification": "pending_logical_run",
            "logical_run": {key: logical.get(key) for key in (
                "state", "run_id", "requested_steps", "completed_updates", "remaining_steps",
            )},
            "next_action": "start_existing_run_through_normal_resume_endpoint",
            "source_semantics_validation": "validated_before_managed_action_and_required_again_on_normal_resume_endpoint",
            "blockers": [],
        }
    if logical.get("remaining_steps") != 0:
        return _blocked("completed_run_progress_invalid", "A completed logical run has a nonzero remaining update count.")
    return {
        "schema": _SCHEMA,
        "status": "passed",
        "classification": "completed_logical_run",
        "logical_run": {key: logical.get(key) for key in (
            "state", "run_id", "requested_steps", "completed_updates", "remaining_steps",
        )},
        "next_action": "start_existing_checkpoint_idle_without_submitting_a_run",
        "source_semantics_validation": "normal_app_startup_still_computes_full_semantics",
        "blockers": [],
    }


def _checkpoint_recovery_sqlite(runtime: Any) -> list[dict[str, Any]]:
    """Finalize stopped runtime databases using the captured runtime configuration."""
    runtime_data = _semantic_runtime_data(runtime) if not isinstance(runtime, Mapping) else runtime
    run_dir = Path(runtime_data["run_dir"])
    databases = [run_dir / "gradient-server" / "jobs.sqlite3"]
    bepa = runtime_data.get("bepa") or {}
    if bepa.get("enabled"):
        state_dir = bepa.get("state_dir")
        if state_dir:
            from ..runtime.config import _resolve_path

            state_dir = _resolve_path(state_dir)
        else:
            state_dir = run_dir / "bepa"
        databases.append(Path(state_dir) / "state.sqlite3")
    return [_checkpoint_sqlite_wal(path) for path in databases]


def _config_file_identity(path: Path) -> tuple[Any, ...]:
    candidate = Path(path)
    info = candidate.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError("deployment configuration is not a regular file")
    return (str(candidate), info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _deployment_configuration_identity(deployment: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(_config_file_identity(path) for path in (
        Path(deployment.path), Path(deployment.runtime_config),
    ))


def _require_configuration_identity(deployment: Any, expected: Any) -> None:
    if _deployment_configuration_identity(deployment) != expected:
        raise RuntimeError("deployment or runtime configuration changed during CPU resume")


def _capture_resume_runtime(
    deployment: Any,
) -> tuple[Any, Any, tuple[tuple[Any, ...], ...]]:
    from .instance_plan import _load_runtime

    identity = _deployment_configuration_identity(deployment)
    loaded_deployment = deployment
    public_dict = getattr(deployment, "public_dict", None)
    if callable(public_dict):
        loaded_deployment = _deployment(deployment.path)
        if loaded_deployment.public_dict() != public_dict():
            raise RuntimeError("loaded deployment differs from its current configuration file")
    _require_configuration_identity(loaded_deployment, identity)
    runtime = _load_runtime(loaded_deployment)
    _require_configuration_identity(loaded_deployment, identity)
    return loaded_deployment, runtime, identity


def _transaction_failure(stage: str, exc: BaseException) -> dict[str, Any]:
    from .instance_failure import source_failure_locations

    return {
        "stage": stage,
        "error_type": type(exc).__name__,
        "failure_locations": source_failure_locations(exc),
    }


def _managed_resume_transaction_local(deployment: Any) -> dict[str, Any]:
    """Run the full CPU recovery transaction in the configured deployment Python."""
    transaction: dict[str, Any] = {
        "schema": _RESUME_TRANSACTION_SCHEMA,
        "status": "blocked",
        "initial_plan": None,
        "recovery": None,
        "sqlite_wal_checkpoints": None,
        "resume_plan": None,
        "post_recovery": None,
    }

    try:
        working_deployment, runtime, configuration_identity = _capture_resume_runtime(deployment)
        from .instance_resume import build_resume_plan

        initial_plan = build_resume_plan(
            working_deployment.path, gpus=working_deployment.gpus, inspect_checkpoint=True,
        )
        _require_configuration_identity(working_deployment, configuration_identity)
    except Exception as exc:
        transaction["failure"] = _transaction_failure("initial_resume_plan", exc)
        return transaction
    transaction["initial_plan"] = initial_plan

    try:
        recovery = _managed_recover_local(
            working_deployment, initial_plan,
            _runtime_snapshot=runtime,
            _configuration_identity=configuration_identity,
        )
    except Exception:
        # The standalone managed recovery worker reports a sanitized worker
        # failure for this same boundary; keep that result shape here too.
        recovery = _blocked(
            "managed_recovery_worker_failed",
            "Deployment Python could not complete managed recovery.",
        )
    transaction["recovery"] = recovery
    if recovery.get("status") != "passed":
        return transaction

    try:
        _require_configuration_identity(working_deployment, configuration_identity)
        transaction["sqlite_wal_checkpoints"] = _checkpoint_recovery_sqlite(runtime)
        _require_configuration_identity(working_deployment, configuration_identity)
    except Exception as exc:
        transaction["failure"] = _transaction_failure("sqlite_wal_checkpoint", exc)
        return transaction

    try:
        _require_configuration_identity(working_deployment, configuration_identity)
        checked_plan = build_resume_plan(
            working_deployment.path, gpus=working_deployment.gpus, inspect_checkpoint=True,
        )
        _require_configuration_identity(working_deployment, configuration_identity)
    except Exception as exc:
        transaction["failure"] = _transaction_failure("post_recovery_resume_plan", exc)
        return transaction
    transaction["resume_plan"] = checked_plan

    try:
        post_recovery = validate_post_recovery_plan(checked_plan, recovery)
    except Exception as exc:
        transaction["failure"] = _transaction_failure("post_recovery_validation", exc)
        return transaction
    transaction["post_recovery"] = post_recovery
    if post_recovery.get("status") == "passed":
        try:
            _require_configuration_identity(working_deployment, configuration_identity)
        except Exception as exc:
            transaction["failure"] = _transaction_failure("post_recovery_validation", exc)
            return transaction
        transaction["status"] = "passed"
    return transaction


def _close_engine(engine: Any) -> None:
    """Release only this temporary CPU engine's handles; do not run shutdown cleanup."""
    cleanup_error: BaseException | None = None
    core = getattr(engine, "core", None)
    if core is not None:
        try:
            core.close()
        except BaseException as exc:
            cleanup_error = exc
    store = getattr(engine, "store", None)
    clear_memory_payloads = getattr(store, "clear_memory_payloads", None)
    if callable(clear_memory_payloads):
        try:
            clear_memory_payloads()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
    publisher = getattr(engine, "_checkpoint_publications", None)
    if publisher is not None:
        try:
            publisher.shutdown(wait=True, cancel_futures=False)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
    owner_lock = getattr(engine, "_process_owner_lock", None)
    if owner_lock is not None:
        try:
            owner_lock.close()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        finally:
            engine._process_owner_lock = None
    if cleanup_error is not None:
        raise cleanup_error


def _preserved_status(
    status: Mapping[str, Any], pending_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = status.get("last_update_context")
    logical = context.get("logical_job") if isinstance(context, Mapping) else None
    pending_logical = (pending_context.get("logical_job")
                       if isinstance(pending_context, Mapping) else None)
    if not isinstance(logical, Mapping) and isinstance(pending_logical, Mapping):
        logical = pending_logical
    run_id = logical.get("run_id") if isinstance(logical, Mapping) else None
    if run_id is None and isinstance(pending_context, Mapping):
        run_id = pending_context.get("run_id")
    return {
        "lineage_id": status.get("lineage_id"),
        "engine_epoch": status.get("engine_epoch"),
        "snapshot_id": status.get("snapshot_id"),
        "checkpoint_generation": status.get("checkpoint_generation"),
        "policy_version": status.get("policy_version"),
        "optimizer_steps": status.get("optimizer_steps"),
        "scheduler_steps": status.get("scheduler_steps"),
        "run_id": run_id,
        "requested_steps": logical.get("requested_steps") if isinstance(logical, Mapping) else None,
        "completed_updates": logical.get("completed_updates") if isinstance(logical, Mapping) else None,
        "remaining_steps": (
            logical.get("requested_steps") - logical.get("completed_updates")
            if isinstance(logical, Mapping)
            and type(logical.get("requested_steps")) is int
            and type(logical.get("completed_updates")) is int
            else None
        ),
    }


def _same_optimizer_state(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    fields = (
        "lineage_id", "engine_epoch", "snapshot_id", "checkpoint_generation",
        "policy_version", "optimizer_steps", "scheduler_steps", "last_update_context",
    )
    return all(before.get(field) == after.get(field) for field in fields)


def _same_optimizer_progress(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    fields = (
        "lineage_id", "engine_epoch", "policy_version", "optimizer_steps",
        "scheduler_steps", "last_update_context",
    )
    return all(before.get(field) == after.get(field) for field in fields)


def _recover_loaded_engine(
    engine: Any,
    *,
    intent: Mapping[str, Any] | None = None,
    plan_blocker_codes: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    core = engine.core
    before = core.status()
    preserved = _preserved_status(before)
    if (before.get("checkpoint_state") == "recovery_required"
            or before.get("pending_attempt") is not None
            or before.get("checkpoint_intent") is not None):
        return {
            **_blocked(
                "ambiguous_optimizer_state",
                "Core has an unresolved optimizer-entry fence or checkpoint intent; no rollback was attempted.",
                classification="ambiguous",
            ),
            "pending_attempt": before.get("pending_attempt"),
            "checkpoint_intent": before.get("checkpoint_intent"),
            "preserved": preserved,
        }
    if before.get("dirty") is True or before.get("checkpoint_state") != "durable":
        return {
            **_blocked(
                "optimizer_state_not_durable",
                "The restored central optimizer is not clean and durable; no optimizer state was changed.",
                classification="ambiguous",
            ),
            "preserved": preserved,
        }

    pending = before.get("pending_window")
    cancellation = None
    if "gradient_optimizer_window_pending" in plan_blocker_codes and pending is None:
        return {
            **_blocked(
                "gradient_store_core_mismatch",
                "The read-only plan found a nonterminal optimizer window but the durable Core has no matching prepared window.",
            ),
            "preserved": preserved,
        }
    if pending is not None:
        window = core.prepared_window(pending["window_id"], pending["attempt_id"])
        if window is None:
            return {
                **_blocked("prepared_window_missing", "Core status names a prepared window that cannot be restored."),
                "preserved": preserved,
            }
        preserved = _preserved_status(before, window.update_context)
        try:
            open_store = getattr(engine, "open_recovery_window_store", None)
            recovery_store = open_store() if callable(open_store) else getattr(engine, "store", None)
            if recovery_store is None:
                raise RuntimeError("gradient scheduler store is unavailable for a prepared Core window")
            inspection = recovery_store.inspect_window_for_recovery(
                window.window_id, window.attempt_id,
            )
        except Exception:
            return {
                **_blocked("gradient_window_inspection_failed", "The prepared gradient window could not be inspected safely."),
                "preserved": preserved,
            }
        if not isinstance(inspection, Mapping):
            return {
                **_blocked("gradient_window_metadata_invalid", "Prepared gradient window inspection returned invalid metadata."),
                "preserved": preserved,
            }
        state = inspection.get("state")
        accepted = inspection.get("accepted_jobs")
        total = inspection.get("total_jobs")
        store_state = inspection.get("store_state")
        if type(accepted) is not int or type(total) is not int or accepted < 0 or total < accepted:
            return {
                **_blocked("gradient_window_metadata_invalid", "Prepared gradient acceptance metadata is invalid."),
                "preserved": preserved,
            }
        identity = inspection.get("identity")
        expected_identity = {
            "lineage_id": before.get("lineage_id"),
            "engine_epoch": before.get("engine_epoch"),
        }
        if identity is not None and identity != expected_identity:
            return {
                **_blocked("gradient_store_core_mismatch", "The gradient store belongs to a different Core lineage or engine epoch."),
                "preserved": preserved,
            }
        if state != "missing" and identity != expected_identity:
            return {
                **_blocked("gradient_store_core_mismatch", "The registered gradient window has no matching Core store identity."),
                "preserved": preserved,
            }
        if state == "missing" and (store_state is not None or accepted != 0 or total != 0):
            return {
                **_blocked("gradient_window_metadata_invalid", "The missing-window inspection conflicts with stored job metadata."),
                "preserved": preserved,
            }
        if state == "materialized" and total > 0 and accepted == total:
            return {
                "schema": _SCHEMA,
                "status": "passed",
                "classification": "deferred_to_learner",
                "actions": [],
                "validated_required_action": "launch_normal_engine_to_commit_the_complete_prepared_window",
                "preserved": preserved,
                "prepared_window": {"window_id": window.window_id, "attempt_id": window.attempt_id,
                                    "accepted_jobs": accepted, "total_jobs": total},
                "bepa_attempt_id": (window.update_context or {}).get("bepa_attempt_id"),
                "bepa_ledger": {"state": "not_mutated", "attempt_id": (window.update_context or {}).get("bepa_attempt_id")},
                "source_semantics_validation": _semantics_deferred(),
                "cancellation_receipt": None,
                "blockers": [],
            }
        if state not in {"missing", "incomplete"}:
            code = "gradient_window_terminal_mismatch" if state == "terminal" else "gradient_window_state_ambiguous"
            return {
                **_blocked(code, "The gradient store window is terminal or in an apply state that cannot be safely cancelled."),
                "preserved": preserved,
            }
        if state == "incomplete" and (store_state != "pending" or total <= 0 or accepted >= total):
            return {
                **_blocked("gradient_window_metadata_invalid", "The incomplete gradient window does not match the store state."),
                "preserved": preserved,
            }
        reason = "managed_resume_incomplete_window_cancelled_before_optimizer_entry"
        if state == "missing":
            # The read-only store API proved there is no scheduler window to
            # fence. The durable Core state and its optimizer fences above are
            # authoritative, so cancel only the Core-prepared window.
            cancelled = True
        else:
            try:
                cancelled = recovery_store.cancel_window(window.window_id, window.attempt_id, reason)
            except Exception:
                return {
                    **_blocked("gradient_window_cancel_failed", "The gradient store could not fence the incomplete window."),
                    "preserved": preserved,
                }
        if not cancelled:
            try:
                inspection = recovery_store.inspect_window_for_recovery(
                    window.window_id, window.attempt_id,
                )
            except Exception:
                return {
                    **_blocked("gradient_window_race_unknown", "Gradient acceptance changed during cancellation; recovery stopped."),
                    "preserved": preserved,
                }
            if (isinstance(inspection, Mapping)
                    and inspection.get("state") == "materialized"
                    and type(inspection.get("accepted_jobs")) is int
                    and type(inspection.get("total_jobs")) is int
                    and inspection["total_jobs"] > 0
                    and inspection["accepted_jobs"] == inspection["total_jobs"]
                    and inspection.get("identity") == expected_identity):
                accepted = inspection["accepted_jobs"]
                total = inspection["total_jobs"]
                return {
                    "schema": _SCHEMA,
                    "status": "passed",
                    "classification": "deferred_to_learner",
                    "actions": [],
                    "validated_required_action": "launch_normal_engine_to_commit_the_complete_prepared_window",
                    "preserved": preserved,
                    "prepared_window": {"window_id": window.window_id, "attempt_id": window.attempt_id,
                                        "accepted_jobs": accepted, "total_jobs": total},
                    "bepa_attempt_id": (window.update_context or {}).get("bepa_attempt_id"),
                    "bepa_ledger": {"state": "not_mutated", "attempt_id": (window.update_context or {}).get("bepa_attempt_id")},
                    "source_semantics_validation": _semantics_deferred(),
                    "cancellation_receipt": None,
                    "blockers": [],
                }
            return {
                **_blocked("gradient_window_race_unknown", "The prepared window changed during cancellation; recovery stopped."),
                "preserved": preserved,
            }
        try:
            cancellation = core.cancel_prepared_window(window, reason=reason)
        except Exception:
            return {
                **_blocked("core_cancellation_receipt_failed", "Core could not persist the pre-optimizer cancellation receipt."),
                "preserved": preserved,
            }

    source_checkpoint = {
        "snapshot_id": before.get("snapshot_id"),
        "checkpoint_generation": before.get("checkpoint_generation"),
    }
    if cancellation is not None:
        try:
            saved_generation = core.save_checkpoint(
                reason="managed_preoptimizer_window_cancelled",
            )
        except Exception:
            after_failed_save = core.status()
            return {
                **_blocked(
                    "cancelled_checkpoint_normalization_failed",
                    "Core recorded cancellation but could not persist a canonical snapshot without the prepared window.",
                    classification="ambiguous",
                ),
                "source_checkpoint": source_checkpoint,
                "preserved": _preserved_status(after_failed_save),
                "cancellation_receipt": cancellation,
            }
    else:
        saved_generation = before.get("checkpoint_generation")
    after = core.status()
    checkpoint_advanced = (
        cancellation is None
        or (type(saved_generation) is int
            and type(source_checkpoint["checkpoint_generation"]) is int
            and saved_generation == source_checkpoint["checkpoint_generation"] + 1
            and after.get("checkpoint_generation") == saved_generation
            and after.get("snapshot_id") != source_checkpoint["snapshot_id"])
    )
    progress_preserved = (_same_optimizer_progress(before, after) if cancellation is not None
                          else _same_optimizer_state(before, after))
    if not checkpoint_advanced or not progress_preserved:
        return {
            **_blocked("optimizer_state_changed", "Managed recovery unexpectedly changed durable optimizer state."),
            "source_checkpoint": source_checkpoint,
            "preserved": _preserved_status(after),
            "after": _preserved_status(after),
        }
    if cancellation is not None:
        return {
            "schema": _SCHEMA,
            "status": "passed",
            "classification": "prepared_window_cancelled_before_optimizer",
            "actions": ["cancelled_incomplete_prepared_window"],
            "source_checkpoint": source_checkpoint,
            "preserved": _preserved_status(after, window.update_context),
            "bepa_attempt_id": cancellation.get("update_context", {}).get("bepa_attempt_id"),
            "bepa_ledger": {"state": "not_mutated", "attempt_id": cancellation.get("update_context", {}).get("bepa_attempt_id")},
            "source_semantics_validation": _semantics_deferred(),
            "cancellation_receipt": cancellation,
            "blockers": [],
        }
    if intent is not None:
        receipt_value = core.result_receipt(intent["window_id"], intent["attempt_id"])
        receipt = receipt_value.to_dict() if receipt_value is not None else after.get("last_receipt")
        target = intent.get("target") if isinstance(intent, Mapping) else None
        if (not isinstance(receipt, Mapping)
                or receipt.get("window_id") != intent.get("window_id")
                or receipt.get("attempt_id") != intent.get("attempt_id")
                or not isinstance(target, Mapping)
                or receipt.get("policy_version") != target.get("policy_version")
                or receipt.get("checkpoint_generation") != target.get("generation")):
            return {
                **_blocked("committed_checkpoint_receipt_missing", "The target checkpoint has no matching committed window receipt."),
                "preserved": _preserved_status(after),
            }
        return {
            "schema": _SCHEMA,
            "status": "passed",
            "classification": "committed_checkpoint_reconciled",
            "actions": ["reconciled_committed_checkpoint_receipt"],
            "preserved": _preserved_status(after),
            "checkpoint_receipt": receipt,
            "bepa_ledger": {"state": "not_mutated", "attempt_id": (after.get("last_update_context") or {}).get("bepa_attempt_id")},
            "source_semantics_validation": _semantics_deferred(),
            "blockers": [],
        }
    return {
        "schema": _SCHEMA,
        "status": "passed",
        "classification": "clean_durable_checkpoint",
        "actions": [],
        "preserved": preserved,
        "bepa_ledger": {"state": "not_mutated"},
        "source_semantics_validation": _semantics_deferred(),
        "blockers": [],
    }


@contextmanager
def _managed_compute_sessions_database(
    database_path: Path, expected_identity: Mapping[str, str], *, read_only: bool = False,
):
    """Open an existing scheduler database without constructing or migrating its Store."""
    import sqlite3

    database = Path(database_path)
    info = database.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1):
        raise RuntimeError("compute session database is not a regular single-link file")
    if (not isinstance(expected_identity, Mapping)
            or set(expected_identity) != {"lineage_id", "engine_epoch"}
            or any(not isinstance(expected_identity[key], str) or not expected_identity[key]
                   for key in ("lineage_id", "engine_epoch"))):
        raise RuntimeError("restored Core identity is unavailable for compute session recovery")

    connection = None
    try:
        connection = sqlite3.connect(
            database.resolve(strict=True).as_uri() + ("?mode=ro" if read_only else "?mode=rw"), uri=True,
            timeout=5.0, isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN" if read_only else "BEGIN IMMEDIATE")

        current = database.lstat()
        if (not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode)
                or current.st_nlink != 1 or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)):
            raise RuntimeError("compute session database changed while it was opened")
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        required_columns = {
            "windows": {"window_id", "attempt_id", "window_json", "state"},
            "jobs": {"window_id", "attempt_id", "job_id", "state", "executor_id"},
            "store_metadata": {"key", "value"},
            "compute_sessions": {"executor_id", "session_id", "round_id", "active"},
        }
        if not set(required_columns) <= tables:
            raise RuntimeError("compute session database schema is incomplete")
        for table, expected in required_columns.items():
            columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            if not expected <= columns:
                raise RuntimeError(f"compute session database {table} schema is incomplete")
        central = connection.execute(
            "SELECT value FROM store_metadata WHERE key='central_identity'"
        ).fetchone()
        try:
            stored_identity = None if central is None else json.loads(central["value"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("compute session database Core identity is malformed") from exc
        if stored_identity != dict(expected_identity):
            raise RuntimeError("compute session database belongs to another Core identity")
        compute_columns = {row["name"] for row in connection.execute(
            "PRAGMA table_info(compute_sessions)"
        )}
        yield connection, ("parked" in compute_columns)
        connection.execute("COMMIT")
    except BaseException as exc:
        if connection is not None and connection.in_transaction:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        if isinstance(exc, sqlite3.Error):
            raise RuntimeError("compute session database transaction failed") from exc
        raise
    finally:
        if connection is not None:
            connection.close()


def _compute_session_records(connection: Any, *, parked_column: bool) -> list[dict[str, Any]]:
    parked_expr = "parked" if parked_column else "NULL"
    rows = connection.execute(
        "SELECT executor_id,session_id,round_id,active," + parked_expr + " AS parked "
        "FROM compute_sessions ORDER BY executor_id"
    ).fetchall()
    result = []
    for row in rows:
        active = row["active"]
        if type(active) is not int or active not in (0, 1):
            raise RuntimeError("compute session database contains an invalid active state")
        if active != 1:
            continue
        if any(not isinstance(row[key], str) or not row[key]
               for key in ("executor_id", "session_id", "round_id")):
            raise RuntimeError("compute session database contains an invalid active session identity")
        parked_value = row["parked"]
        parked_known = type(parked_value) is int and parked_value in (0, 1)
        if parked_column and not parked_known:
            raise RuntimeError("compute session database contains an invalid parked state")
        result.append({
            "executor_id": row["executor_id"],
            "session_id": row["session_id"],
            "round_id": row["round_id"],
            "parked": parked_value if parked_known else None,
            "parked_state": "known" if parked_known else "unknown",
        })
    return result


def _read_active_compute_sessions(
    database_path: Path, *, expected_identity: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Read live scheduler fences; a legacy missing parked column remains unknown."""
    database = Path(database_path)
    try:
        database.lstat()
    except FileNotFoundError:
        if any(Path(f"{database}{suffix}").exists() or Path(f"{database}{suffix}").is_symlink()
               for suffix in ("-wal", "-shm")):
            raise RuntimeError("compute session database sidecar exists without its database")
        return []
    with _managed_compute_sessions_database(database, expected_identity, read_only=True) as (connection, parked_column):
        return _compute_session_records(connection, parked_column=parked_column)


def _release_stopped_compute_sessions(
    database_path: Path,
    *,
    expected_identity: Mapping[str, str],
    expected_sessions: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Clear only exact active rows after the caller has revalidated stop-now proof."""
    if not expected_sessions:
        return []
    with _managed_compute_sessions_database(database_path, expected_identity) as (connection, parked_column):
        active = _compute_session_records(connection, parked_column=parked_column)
        key = lambda row: (row["executor_id"], row["session_id"], row["round_id"])
        expected_keys = [key(row) for row in expected_sessions]
        if len(expected_keys) != len(set(expected_keys)):
            raise RuntimeError("compute session recovery receipt contains duplicate identities")
        if sorted(key(row) for row in active) != sorted(expected_keys):
            raise RuntimeError("active compute sessions changed before stopped recovery")
        released = []
        for session in active:
            cursor = connection.execute(
                "UPDATE compute_sessions SET active=0 "
                "WHERE executor_id=? AND session_id=? AND round_id=? AND active=1",
                key(session),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("active compute session changed during stopped recovery")
            released.append({**session, "active": 0, "release": "stopped_process_proof"})
        return released


def _completed_stop_now_proof(deployment: Any) -> tuple[str, dict[str, Any]]:
    from .instance import _read_private_json
    from .instance_stop_control import STOP_OPERATION_SCHEMA
    from .instance_stop_now import _validate_stop_proof

    operation = _read_private_json(deployment.state_dir / "stop-operation.json")
    if (not isinstance(operation, Mapping)
            or operation.get("schema") != STOP_OPERATION_SCHEMA
            or operation.get("mode") != "now"
            or operation.get("manager_status") != "completed"
            or operation.get("phase") != "completed"):
        raise RuntimeError("completed stop-now operation proof is unavailable")
    operation_id = operation.get("operation_id")
    proof = operation.get("proof")
    if (not isinstance(operation_id, str) or not isinstance(proof, Mapping)
            or proof.get("operation_id") != operation_id):
        raise RuntimeError("completed stop-now operation does not contain its matching proof")
    validated = _validate_stop_proof(
        deployment, deployment.state_dir / "active.json", proof,
    )
    if (not isinstance(validated, Mapping)
            or type(validated.get("verified_process_count")) is not int
            or type(validated.get("nested_manifest_missing")) is not bool):
        raise RuntimeError("stop proof validator returned incomplete process evidence")
    return operation_id, validated


def _compute_sessions_unreleased(
    recovered: Mapping[str, Any], deployment: Any, message: str,
    sessions: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    result = dict(recovered)
    result.update(
        status="blocked",
        classification="ambiguous",
        blockers=[{"code": "compute_sessions_unreleased", "message": message}],
        next_command=shlex.join([
            "qwen-rl", "instance", "--deployment", str(deployment.path), "stop-now",
        ]),
    )
    if sessions is not None:
        result["unreleased_compute_sessions"] = [dict(row) for row in sessions]
    return result


def _managed_recover_local(
    deployment: Any, plan: Any = None, *,
    _runtime_snapshot: Any = None,
    _configuration_identity: Any = None,
) -> dict[str, Any]:
    from .instance_resume import _instance_state
    from .instance_plan import _load_runtime

    blockers = _plan_blockers(plan)
    plan_blocker_codes = frozenset(
        row.get("code") for row in plan.get("blockers", [])
        if isinstance(plan, Mapping) and isinstance(row, Mapping)
        and isinstance(row.get("code"), str)
    ) if isinstance(plan, Mapping) else frozenset()
    if blockers:
        ambiguous = any(row["code"] in {"optimizer_attempt_pending", "optimizer_owner_lock_busy"}
                        for row in blockers)
        return {
            **_blocked(
                "resume_plan_incompatible" if not ambiguous else "ambiguous_optimizer_state",
                "The read-only resume plan contains an incompatible or unresolved blocker.",
                classification="ambiguous" if ambiguous else "incompatible",
            ),
            "preflight_blockers": blockers,
        }
    try:
        state = _instance_state(deployment)
    except Exception:
        state = "unknown"
    if state != "stopped":
        return _blocked(
            "instance_not_proven_stopped",
            "Managed recovery requires the source instance to be proven stopped.",
            classification="ambiguous",
        )
    try:
        owner_descriptor = _instance_owner_lock(deployment.state_dir)
    except BlockingIOError:
        return _blocked("instance_owner_lock_busy", "The instance owner lock is held.", classification="ambiguous")
    except Exception:
        return _blocked("instance_owner_lock_unavailable", "The instance owner lock could not be acquired safely.", classification="ambiguous")

    engine = None
    optimizer_owner_lock = None
    try:
        if _instance_state(deployment) != "stopped":
            return _blocked(
                "instance_not_proven_stopped",
                "Managed process ownership changed while acquiring the instance lock.",
                classification="ambiguous",
            )
        try:
            from .instance_migration import assert_not_migrating

            assert_not_migrating(deployment)
        except Exception:
            return _blocked(
                "instance_migration_fenced",
                "The source instance has an unresolved migration fence; recovery made no checkpoint changes.",
                classification="ambiguous",
            )
        try:
            if _configuration_identity is not None:
                _require_configuration_identity(deployment, _configuration_identity)
            runtime = (_runtime_snapshot if _runtime_snapshot is not None
                       else _load_runtime(deployment))
            if _configuration_identity is not None:
                _require_configuration_identity(deployment, _configuration_identity)
        except Exception:
            if _configuration_identity is not None:
                return _blocked(
                    "deployment_configuration_changed",
                    "Deployment or runtime configuration changed or could not be validated.",
                    classification="ambiguous",
                )
            return _blocked("deployment_runtime_invalid", "Deployment runtime configuration is invalid.")
        root = _resolved_recovery_dir(runtime)
        try:
            pointer, checkpoint_error = _checkpoint_preflight(root)
        except Exception:
            return _blocked("checkpoint_preflight_failed", "Checkpoint metadata could not be inspected safely.")
        if pointer is None:
            return _blocked(
                "checkpoint_not_recoverable",
                f"The configured checkpoint cannot be used for managed recovery ({checkpoint_error}).",
            )
        from ..learner.gradient_server.ownership import ProcessOwnerLock

        try:
            optimizer_owner_lock = ProcessOwnerLock.for_checkpoint_dir(root)
        except RuntimeError:
            return _blocked("optimizer_owner_lock_busy", "The central optimizer owner lock is held.", classification="ambiguous")
        except OSError:
            return _blocked("optimizer_owner_lock_unavailable", "The central optimizer owner lock could not be created or opened safely.", classification="ambiguous")
        try:
            pointer, checkpoint_error = _checkpoint_preflight(root)
        except Exception:
            return _blocked("checkpoint_preflight_failed", "Checkpoint metadata could not be revalidated under its owner lock.", classification="ambiguous")
        if pointer is None:
            return _blocked(
                "checkpoint_changed_during_preflight",
                f"The checkpoint changed while acquiring its owner lock ({checkpoint_error}).",
                classification="ambiguous",
            )
        planned_checkpoint = plan.get("checkpoint") if isinstance(plan, Mapping) else None
        if isinstance(planned_checkpoint, Mapping):
            checkpoint_state = planned_checkpoint.get("state")
            if checkpoint_state == "payload_metadata_inspected":
                planned_generation = planned_checkpoint.get("generation")
                planned_snapshot_id = planned_checkpoint.get("snapshot_id")
                if (type(planned_generation) is not int or not isinstance(planned_snapshot_id, str)
                        or planned_generation != pointer.get("generation")
                        or planned_snapshot_id != pointer.get("snapshot_id")):
                    return _blocked(
                        "resume_plan_stale",
                        "The durable checkpoint changed after the supplied read-only plan was produced.",
                        classification="ambiguous",
                    )
            elif checkpoint_state != "not_inspected_owner_lock_unproven":
                return _blocked(
                    "resume_plan_checkpoint_identity_unavailable",
                    "The supplied plan has no inspected checkpoint identity and does not report an unproven owner lock.",
                    classification="ambiguous",
                )
        disposition, intent = _intent_disposition(root, pointer)
        if disposition in {"source", "ambiguous_pending_fence", "neither"}:
            return _blocked(
                "ambiguous_optimizer_state",
                "Checkpoint metadata cannot prove a committed receipt or a pre-optimizer state; no rollback was attempted.",
                classification="ambiguous",
            )
        if disposition == "invalid":
            return _blocked("checkpoint_transaction_invalid", "Optimizer recovery sidecars are malformed.")
        committed_intent = intent if disposition == "target" else None
        if _configuration_identity is not None:
            try:
                _require_configuration_identity(deployment, _configuration_identity)
            except Exception:
                return _blocked(
                    "deployment_configuration_changed",
                    "Deployment or runtime configuration changed before CPU restore; no checkpoint changes were made.",
                    classification="ambiguous",
                )
        with _cpu_thread_limit():
            try:
                engine_started = time.perf_counter()
                engine = _build_cpu_engine(
                    deployment, runtime, process_owner_lock=optimizer_owner_lock,
                )
                engine_seconds = time.perf_counter() - engine_started
            except Exception as exc:
                message = str(exc)
                if "compatibility" in message.lower() or "differs" in message.lower():
                    return _blocked("checkpoint_incompatible", "The durable checkpoint is incompatible with this deployment.")
                if "already owned or cannot be locked" in message.lower():
                    return _blocked("optimizer_owner_lock_busy", "The central optimizer owner lock is held.", classification="ambiguous")
                return _blocked("managed_recovery_failed", "The CPU central optimizer could not restore the managed checkpoint.")
            core_status = engine.core.status()
            if (core_status.get("checkpoint_state") != "durable"
                    or core_status.get("dirty") is True):
                return _recover_loaded_engine(
                    engine,
                    intent=committed_intent,
                    plan_blocker_codes=plan_blocker_codes,
                )
            semantics = validate_pending_source_semantics(runtime, engine.core)
            if semantics.get("status") != "passed":
                return {
                    **_blocked(
                        "source_semantics_incompatible",
                        "Current dataset, contract, toolkit, or BEPA semantics differ from the pending run.",
                    ),
                    "source_semantics_validation": semantics,
                }
            recovered = _recover_loaded_engine(
                engine,
                intent=committed_intent,
                plan_blocker_codes=plan_blocker_codes,
            )
            if recovered.get("status") == "passed" and getattr(getattr(runtime, "evaluation", None), "enabled", False):
                try:
                    import asyncio
                    from ..evaluation.relocation import preflight_evaluation_initialization
                    from ..task_source import load_tasks

                    training_tasks = load_tasks(pools=("arc2_train",), variants=False,
                                                dataset_dir=runtime.dataset)
                    metadata = engine.recovery_metadata
                    publication = _publish_cpu_evaluation_policy(engine)
                    recovered["evaluation"] = asyncio.run(preflight_evaluation_initialization(
                        _semantic_runtime_data(runtime), recovery=metadata,
                        current_policy=publication, training_tasks=training_tasks,
                    ))
                except Exception as exc:
                    from .instance_failure import source_failure_locations
                    return {**_blocked("evaluation_initialization_failed",
                                       "CPU evaluation initialization failed before GPU startup."),
                            "evaluation_error_type": type(exc).__name__,
                            "field": "evaluation", "failure_locations": source_failure_locations(exc),
                            "next_command": f"qwen-rl instance --deployment {deployment.path} doctor"}
            if recovered.get("status") == "passed":
                recovered["load_measurements"] = {
                    "cpu_engine_seconds": engine_seconds,
                    "initial_adapter": getattr(engine, "startup_load_measurements", {}),
                    "checkpoint_reads": list(getattr(engine.core, "snapshot_load_metrics", ())),
                }
                recovered["source_semantics_validation"] = {
                    **semantics,
                    "normal_resume_revalidation_required": (
                        semantics.get("state") == "verified"
                    ),
                }
                # A durable CPU optimizer checkpoint does not prove that a
                # parked same-node session returned its Worker reservation.
                # Only a fully revalidated stop-now process proof permits
                # releasing that stale scheduler receipt after a crash.
                try:
                    if _instance_state(deployment) != "stopped":
                        return _compute_sessions_unreleased(
                            recovered, deployment,
                            "The instance is no longer proven stopped; compute session recovery made no changes.",
                        )
                    expected_identity = {
                        "lineage_id": core_status.get("lineage_id"),
                        "engine_epoch": core_status.get("engine_epoch"),
                    }
                    database = Path(runtime.run) / "gradient-server" / "jobs.sqlite3"
                    active_sessions = _read_active_compute_sessions(
                        database, expected_identity=expected_identity,
                    )
                except Exception:
                    return _compute_sessions_unreleased(
                        recovered, deployment,
                        "Compute session state could not be safely verified; no session was released.",
                    )
                if active_sessions:
                    try:
                        operation_id, stop_proof = _completed_stop_now_proof(deployment)
                    except Exception:
                        return _compute_sessions_unreleased(
                            recovered, deployment,
                            "Active compute sessions require a complete, revalidated stop-now proof.",
                            active_sessions,
                        )
                    try:
                        released_sessions = _release_stopped_compute_sessions(
                            database,
                            expected_identity=expected_identity,
                            expected_sessions=active_sessions,
                        )
                    except Exception:
                        return _compute_sessions_unreleased(
                            recovered, deployment,
                            "Active compute sessions changed or could not be released atomically.",
                            active_sessions,
                        )
                    recovered["compute_sessions"] = {
                        "state": "released_after_stop_now_proof",
                        "stop_operation_id": operation_id,
                        "verified_process_count": stop_proof["verified_process_count"],
                        "nested_manifest_missing": stop_proof["nested_manifest_missing"],
                        "sessions": released_sessions,
                    }
                    actions = recovered.get("actions")
                    if isinstance(actions, list):
                        actions.append("released_stopped_compute_sessions")
                    else:
                        recovered["actions"] = ["released_stopped_compute_sessions"]
            return recovered
    finally:
        try:
            if engine is not None:
                _close_engine(engine)
        finally:
            try:
                if optimizer_owner_lock is not None:
                    optimizer_owner_lock.close()
            finally:
                os.close(owner_descriptor)


def _python_path(executable: str) -> Path | None:
    candidate = Path(executable).expanduser()
    if not candidate.is_absolute() and candidate.parent == Path("."):
        resolved = shutil.which(executable)
        candidate = Path(resolved) if resolved else candidate
    try:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            return None
        return candidate.absolute()
    except OSError:
        return None


def _run_in_deployment_python(
    deployment: Any, plan: Any = None, *, operation: str = "managed_recover",
) -> dict[str, Any]:
    executable = _python_path(deployment.python)
    if executable is None:
        return _blocked("deployment_python_unavailable", "Configured deployment Python is unavailable.")
    source_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_root) + (os.pathsep + prior_pythonpath if prior_pythonpath else "")
    environment.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void")
    if operation == "managed_recover":
        request: dict[str, Any] = {"plan": plan}
    elif operation == "resume_transaction":
        request = {"operation": operation}
    else:
        return _blocked("managed_recovery_operation_invalid", "The requested CPU recovery operation is unsupported.")
    try:
        payload = json.dumps(request, separators=(",", ":"))
    except (TypeError, ValueError):
        return _blocked("resume_plan_invalid", "The supplied resume plan cannot be sent to deployment Python.")
    try:
        completed = subprocess.run(
            [str(executable), "-B", "-m", "qwen_rl.deployment.instance_recovery",
             "--_managed-recovery-worker", "--deployment", str(deployment.path)],
            input=payload, capture_output=True, text=True, timeout=600,
            cwd=source_root, env=environment, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _blocked("managed_recovery_worker_failed", "Deployment Python could not complete managed recovery.")
    if completed.returncode not in {0, 1} or len(completed.stdout) > 1024 * 1024:
        return _blocked("managed_recovery_worker_failed", "Deployment Python returned an invalid managed recovery result.")
    try:
        result = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError):
        return _blocked("managed_recovery_worker_failed", "Deployment Python did not return a valid recovery receipt.")
    expected_schema = (_SCHEMA if operation == "managed_recover"
                       else _RESUME_TRANSACTION_SCHEMA)
    if not isinstance(result, dict) or result.get("schema") != expected_schema:
        return _blocked("managed_recovery_worker_failed", "Deployment Python returned an unsupported recovery receipt.")
    return result


def _checkpoint_sqlite_wal(path: Path) -> dict[str, Any]:
    """Flush an inactive SQLite WAL before the read-only resume plan inspects it."""
    import sqlite3

    database = Path(path)
    wal = Path(f"{database}-wal")
    shm = Path(f"{database}-shm")
    if not database.exists():
        if wal.exists() or wal.is_symlink() or shm.exists() or shm.is_symlink():
            raise RuntimeError("SQLite sidecar exists without its database")
        return {"database": database.name, "status": "missing"}
    info = database.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError("SQLite recovery database is not a regular file")
    sidecars = [sidecar for sidecar in (wal, shm) if sidecar.exists() or sidecar.is_symlink()]
    if not sidecars:
        return {"database": database.name, "status": "not_needed"}
    for sidecar in sidecars:
        sidecar_info = sidecar.lstat()
        if not stat.S_ISREG(sidecar_info.st_mode) or stat.S_ISLNK(sidecar_info.st_mode):
            raise RuntimeError("SQLite recovery sidecar is not a regular file")

    watched = {str(database.resolve()), str(wal.resolve()), str(shm.resolve())}
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            descriptors = (process / "fd").iterdir()
            for descriptor in descriptors:
                try:
                    target = os.readlink(descriptor)
                except FileNotFoundError:
                    continue
                except PermissionError as exc:
                    if os.geteuid() == 0:
                        raise RuntimeError("SQLite recovery process references could not be checked") from exc
                    continue
                target = target.removesuffix(" (deleted)")
                if target in watched:
                    raise RuntimeError("SQLite recovery database is still open in a process")
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            if os.geteuid() == 0:
                raise RuntimeError("SQLite recovery process references could not be checked") from exc

    if wal.exists() and wal.stat().st_size == 0:
        # The managed recovery worker has returned under the instance operation lock.
        # An empty WAL has no committed frames; retire its stale sidecars without
        # opening SQLite, which may reject an empty WAL paired with an old shm index.
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=rw", uri=True, timeout=5.0)
        try:
            journal_mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        except sqlite3.Error as exc:
            raise RuntimeError("SQLite WAL mode could not be restored") from exc
        finally:
            connection.close()
        if journal_mode != "wal":
            raise RuntimeError("SQLite recovery database is not in WAL mode")
        for sidecar in sidecars:
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass
        directory = os.open(database.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {"database": database.name, "status": "empty_sidecars_retired"}

    connection = None
    try:
        connection = sqlite3.connect(
            database.resolve().as_uri() + "?mode=rw", uri=True, timeout=5.0,
        )
        connection.execute("PRAGMA busy_timeout=5000")
        journal_mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        if journal_mode != "wal":
            raise RuntimeError("SQLite recovery database is not in WAL mode")
        row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if row is None or len(row) != 3:
            raise RuntimeError("SQLite WAL checkpoint returned no receipt")
        busy, log_frames, checkpointed_frames = (int(value) for value in row)
        if busy != 0 or log_frames != checkpointed_frames:
            raise RuntimeError("SQLite WAL checkpoint was busy or incomplete")
    except sqlite3.Error as exc:
        raise RuntimeError("SQLite WAL checkpoint failed") from exc
    finally:
        if connection is not None:
            connection.close()

    if wal.exists() or wal.is_symlink() or shm.exists() or shm.is_symlink():
        raise RuntimeError("SQLite WAL sidecars remain after checkpoint")
    descriptor = os.open(database, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(database.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "database": database.name,
        "status": "checkpointed",
        "log_frames": log_frames,
        "checkpointed_frames": checkpointed_frames,
    }


def managed_recover(deployment: Any, plan: Any = None) -> dict[str, Any]:
    """Reconcile only proven committed receipts or cancel incomplete pre-optimizer work.

    The configured runtime's Python builds the existing CPU central-gradient engine.
    This function never starts a Worker, connects to a service, loads a base model,
    initializes CUDA, or calls optimizer.step().
    """
    try:
        loaded = _deployment(deployment)
    except Exception:
        return _blocked("deployment_invalid", "Instance deployment could not be loaded.")
    return _run_in_deployment_python(loaded, plan)


def managed_resume_transaction(deployment: Any) -> dict[str, Any]:
    """Run read-only plan, managed restore, WAL checkpoint, and post-plan in one child."""
    try:
        loaded = _deployment(deployment)
    except Exception:
        failed = _blocked("deployment_invalid", "Instance deployment could not be loaded.")
    else:
        failed = _run_in_deployment_python(loaded, operation="resume_transaction")
    if _valid_resume_transaction(failed):
        return dict(failed)
    recovery = failed if isinstance(failed, Mapping) and failed.get("schema") == _SCHEMA else _blocked(
        "managed_recovery_worker_failed",
        "Deployment Python did not return a valid CPU recovery transaction.",
    )
    return {
        "schema": _RESUME_TRANSACTION_SCHEMA,
        "status": "blocked",
        "initial_plan": None,
        "recovery": dict(recovery),
        "sqlite_wal_checkpoints": None,
        "resume_plan": None,
        "post_recovery": None,
    }


def _valid_resume_transaction(value: Any) -> bool:
    if (not isinstance(value, Mapping)
            or value.get("schema") != _RESUME_TRANSACTION_SCHEMA
            or value.get("status") not in ("passed", "blocked")
            or not all(key in value for key in (
                "initial_plan", "recovery", "sqlite_wal_checkpoints", "resume_plan", "post_recovery",
            ))):
        return False
    if value.get("status") == "passed":
        initial_plan, recovery = value.get("initial_plan"), value.get("recovery")
        resume_plan, post_recovery = value.get("resume_plan"), value.get("post_recovery")
        if (not _valid_resume_plan_receipt(initial_plan)
                or not isinstance(recovery, Mapping) or recovery.get("schema") != _SCHEMA
                or recovery.get("status") != "passed"
                or not isinstance(value.get("sqlite_wal_checkpoints"), list)
                or not _valid_resume_plan_receipt(resume_plan)
                or not isinstance(post_recovery, Mapping)
                or post_recovery.get("schema") != _SCHEMA
                or post_recovery.get("status") != "passed"):
            return False
        try:
            checked_post = validate_post_recovery_plan(resume_plan, recovery)
        except Exception:
            return False
        return checked_post.get("status") == "passed" and dict(checked_post) == dict(post_recovery)
    failure = value.get("failure")
    recovery = value.get("recovery")
    if failure is not None:
        locations = failure.get("failure_locations") if isinstance(failure, Mapping) else None
        return (
            isinstance(failure, Mapping)
            and isinstance(failure.get("stage"), str)
            and isinstance(failure.get("error_type"), str)
            and isinstance(locations, list)
            and all(
                isinstance(row, Mapping)
                and set(row) == {"file", "line", "function"}
                and isinstance(row.get("file"), str)
                and type(row.get("line")) is int
                and isinstance(row.get("function"), str)
                for row in locations
            )
        )
    if isinstance(recovery, Mapping):
        if recovery.get("schema") != _SCHEMA:
            return False
        if recovery.get("status") == "blocked":
            return True
        post_recovery = value.get("post_recovery")
        if (recovery.get("status") != "passed"
                or not _valid_resume_plan_receipt(value.get("initial_plan"))
                or not isinstance(value.get("sqlite_wal_checkpoints"), list)
                or not _valid_resume_plan_receipt(value.get("resume_plan"))
                or not isinstance(post_recovery, Mapping)
                or post_recovery.get("schema") != _SCHEMA
                or post_recovery.get("status") != "blocked"):
            return False
        try:
            checked_post = validate_post_recovery_plan(value["resume_plan"], recovery)
        except Exception:
            return False
        return checked_post.get("status") == "blocked" and dict(checked_post) == dict(post_recovery)
    return False


def _valid_resume_plan_receipt(plan: Any) -> bool:
    return (
        isinstance(plan, Mapping)
        and plan.get("schema") == _PLAN_SCHEMA
        and plan.get("status") in ("passed", "incomplete", "blocked")
        and isinstance(plan.get("desired_topology"), Mapping)
        and isinstance(plan.get("checkpoint"), Mapping)
        and isinstance(plan.get("logical_run"), Mapping)
        and isinstance(plan.get("staged_gradients"), Mapping)
        and isinstance(plan.get("bepa_affinity"), Mapping)
        and isinstance(plan.get("blockers"), list)
    )


def _worker_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_managed-recovery-worker", action="store_true")
    parser.add_argument("--deployment", required=True)
    args = parser.parse_args(argv)
    try:
        if not args._managed_recovery_worker:
            raise ValueError("managed recovery worker flag is required")
        payload = sys.stdin.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError("worker input exceeds limit")
        raw = json.loads(payload or "{}")
        if not isinstance(raw, Mapping):
            raise ValueError("worker input must be an object")
        deployment = _deployment(args.deployment)
        operation = raw.get("operation", "managed_recover")
        if operation == "managed_recover":
            result = _managed_recover_local(deployment, raw.get("plan"))
        elif operation == "resume_transaction":
            result = _managed_resume_transaction_local(deployment)
        else:
            result = _blocked(
                "managed_recovery_operation_invalid",
                "Deployment Python received an unsupported CPU recovery operation.",
            )
    except Exception:
        result = _blocked("managed_recovery_worker_failed", "Deployment Python could not validate the recovery request.")
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(_worker_main())
