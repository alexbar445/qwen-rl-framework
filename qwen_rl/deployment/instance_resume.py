"""Read-only preflight for resuming one managed central-gradient instance."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import quote

_SCHEMA = "qwen-rl-instance-resume-plan/v1"
_CHECKPOINT_SCHEMA = "central-gradient-checkpoint/v1"
_POINTER_SCHEMA = "central-gradient-checkpoint-pointer/v1"
_STAGED_SCOPE = "local_group_mean_staging/v1"
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_SAFE_OWNER_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_MAX_OWNER_ROWS = 20_000
_MAX_TASK_SPEC_BYTES = 256 * 1024
_MAX_EVENT_BYTES = 16 * 1024
_NAMESPACE_SCOPE = (
    "owner-lock probe and process records are visible only in this filesystem/PID namespace; "
    "managed startup guards remain authoritative"
)


class _ReadOnlyInspectionDeferred(RuntimeError):
    def __init__(self, state: str):
        self.state = state


def _block(code: str, message: str, action: str) -> dict[str, str]:
    return {"code": code, "message": message, "next_action": action}


def _operator_command(name: str, deployment_path: Path,
                      gpus: tuple[int, ...] | None = None) -> str:
    from .instance_diagnostics import _command_for

    return _command_for(name, deployment_path, gpus)


def _instance_state(deployment: Any) -> str:
    """Check managed process identity without reading checkpoint files."""
    from .instance import _SCHEMA as instance_schema, _owner_matches, _read_private_json, _record_matches

    active_path = deployment.state_dir / "active.json"
    manifest = _read_private_json(active_path)
    if manifest is None:
        return "stopped" if not active_path.exists() else "unknown"
    if manifest.get("schema") != instance_schema:
        return "unknown"
    if _owner_matches(manifest.get("owner", {})):
        return "active"
    if any(_record_matches(row) for row in manifest.get("roles", []) if isinstance(row, dict)):
        return "active"
    nested_path = manifest.get("gradient_manifest")
    if isinstance(nested_path, str) and nested_path:
        nested = _read_private_json(Path(nested_path))
        if isinstance(nested, dict) and (
            _record_matches(nested.get("supervisor", {}))
            or any(_record_matches(row) for row in nested.get("roles", []) if isinstance(row, dict))
        ):
            return "active"
    return "stopped"


def _probe_checkpoint_lock(recovery_dir: Path) -> tuple[str, Path]:
    """Probe the existing central optimizer lock without creating or writing it."""
    import fcntl

    root = recovery_dir.expanduser().resolve()
    lock_path = root.parent / f".{root.name}.gradient-optimizer.lock"
    try:
        info = lock_path.lstat()
    except FileNotFoundError:
        return "missing", lock_path
    except OSError:
        return "unknown", lock_path
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        return "unknown", lock_path
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(lock_path, flags)
    except FileNotFoundError:
        return "missing", lock_path
    except OSError:
        return "unknown", lock_path
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_dev != info.st_dev
                or opened.st_ino != info.st_ino):
            return "unknown", lock_path
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "busy", lock_path
        except OSError:
            return "unknown", lock_path
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            return "unknown", lock_path
        return "unlocked", lock_path
    finally:
        os.close(descriptor)


def _selected_gpu_occupancy(selected: Mapping[int, str]) -> dict[str, Any]:
    """Read compute-app PIDs without initializing CUDA; only selected UUIDs count."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"state": "unknown", "selected_processes": []}
    if result.returncode != 0:
        return {"state": "unknown", "selected_processes": []}
    uuid_to_index = {uuid: index for index, uuid in selected.items()}
    pids_by_index: dict[int, set[int]] = {index: set() for index in selected}
    for line in result.stdout.splitlines():
        if not line.strip() or line.strip().lower() == "no running processes found":
            continue
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 2:
            return {"state": "unknown", "selected_processes": []}
        gpu_uuid, raw_pid = pieces
        index = uuid_to_index.get(gpu_uuid)
        if index is None:
            continue
        if not raw_pid.isdecimal() or int(raw_pid) < 1:
            return {"state": "unknown", "selected_processes": []}
        pids_by_index[index].add(int(raw_pid))
    processes = [
        {"gpu_index": index, "gpu_uuid": selected[index], "pids": sorted(pids)}
        for index, pids in sorted(pids_by_index.items()) if pids
    ]
    return {"state": "busy" if processes else "idle", "selected_processes": processes}


def _desired_topology(deployment: Any, runtime: Any) -> tuple[
    dict[str, Any], list[dict[str, str]], list[dict[str, str]]
]:
    from . import instance_plan

    count = len(deployment.gpus)
    tasks_per_worker = runtime.learner.gradient_tasks_per_worker
    group_size = runtime.rollout.group_size
    pairs = [
        {"executor_id": f"executor-{index}", "worker_id": f"worker-{index}",
         "gpu_index": gpu}
        for index, gpu in enumerate(deployment.gpus)
    ]
    from qwen_rl.contracts.sampling_topology import owner_slots, remap_paired_quotas
    learner_data = vars(runtime.learner).copy()
    new_pairs = {row["executor_id"]: row["worker_id"] for row in pairs}
    learner_data["sampling_group_quotas"] = remap_paired_quotas(learner_data, new_pairs)
    learner_data["gradient_executor_worker_ids"] = new_pairs
    group_count = len(owner_slots(learner_data))
    sampling_pairs = [{"executor_id": key, "worker_id": value}
                      for key, value in runtime.learner.sampling_only_worker_ids.items()]
    desired: dict[str, Any] = {
        "state": "computed_desired",
        "gpu_assignments": pairs,
        "gpu_occupancy": {"state": "unknown", "selected_processes": []},
        "pair_count": count,
        "gradient_tasks_per_worker": tasks_per_worker,
        "groups_per_update": group_count,
        "group_size": group_size,
        "nominal_native_trajectories_per_update": group_count * group_size,
        "bepa_enabled": bool(runtime.bepa.enabled),
        "bepa_guided_rows_per_rescue_group": 1 if runtime.bepa.enabled else 0,
    }
    try:
        inventory = instance_plan._gpu_inventory(None)
    except Exception:
        desired["hardware_validation"] = "unavailable"
        return desired, [{"executor_id": row["executor_id"], "worker_id": row["worker_id"]}
                         for row in pairs] + sampling_pairs, []
    missing = set(deployment.gpus) - set(inventory)
    if missing:
        desired["hardware_validation"] = "failed"
        return desired, pairs, [_block(
            "configured_gpu_missing",
            "A selected physical GPU index is absent from the local inventory.",
            "review_selected_gpu_indices",
        )]
    try:
        plan = instance_plan.build_plan(deployment, inventory=list(inventory.values()))
    except Exception:
        desired["hardware_validation"] = "failed"
        return desired, pairs, [_block(
            "topology_plan_invalid",
            "The desired runtime topology failed an existing deployment-plan check.",
            "run_instance_doctor",
        )]
    desired["state"] = "hardware_validated_plan"
    desired["hardware_validation"] = "passed"
    worker_gpus = {
        row["worker_id"]: {"gpu_index": row["gpu_index"], "gpu_uuid": row["gpu_uuid"]}
        for row in plan["executors"]
    }
    desired["gpu_assignments"] = [
        {"executor_id": row["executor_id"], "worker_id": row["worker_id"],
         **worker_gpus[row["worker_id"]]}
        for row in plan["executors"]
    ]
    desired["pair_count"] = len(plan["workers"])
    desired["groups_per_update"] = plan["runtime"]["learner"]["groups_per_update"]
    desired["nominal_native_trajectories_per_update"] = (
        desired["groups_per_update"] * desired["group_size"]
    )
    worker_map = {row["executor_id"]: row["worker_id"] for row in plan["executors"]}
    worker_map.update(runtime.learner.sampling_only_worker_ids)
    desired["gpu_occupancy"] = _selected_gpu_occupancy({
        row["gpu_index"]: row["gpu_uuid"] for row in plan["executors"]
    })
    blockers = []
    if desired["gpu_occupancy"]["state"] == "busy":
        blockers.append(_block(
            "selected_gpu_busy",
            "One or more selected GPUs have active compute processes.",
            "inspect_selected_gpu_process_ownership_without_killing_processes",
        ))
    return desired, [{"executor_id": key, "worker_id": value}
                     for key, value in sorted(worker_map.items())], blockers


def _read_checkpoint_metadata(recovery_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Use the bounded existing pointer/sidecar validator; never open the .pt payload."""
    from .instance_status import _checkpoint_summary

    summary, optimizer = _checkpoint_summary(
        {"learner": {"recovery_dir": str(recovery_dir)}},
        source="resume_plan", process_state="stopped", owner_live=False,
    )
    if summary.get("state") not in {"metadata_consistent", "pending"}:
        return summary, None
    pointer, state = _read_json(recovery_dir / "latest.json")
    if state != "ok" or pointer is None:
        summary["state"] = "invalid"
        summary["error_code"] = "checkpoint_pointer_unreadable"
        return summary, None
    return summary, pointer


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    from .instance_status import _read_json as read_status_json

    return read_status_json(path)


def _safe_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _SAFE_TASK_ID.fullmatch(value) else None


def _safe_owner_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _SAFE_OWNER_ID.fullmatch(value) else None


def _string_ids(value: Any, *, allowed: set[str] | None = None) -> list[str] | None:
    if not isinstance(value, list) or len(value) > 100_000:
        return None
    result: list[str] = []
    for item in value:
        safe = _safe_id(item)
        if safe is None or safe in result or (allowed is not None and safe not in allowed):
            return None
        result.append(safe)
    return result


def _snapshot_file(recovery_dir: Path, pointer: Mapping[str, Any]) -> tuple[
    Path | None, tuple[int, int, int, int, int] | None
]:
    from .instance_status import _regular_file_size

    name = pointer.get("snapshot")
    snapshot_id = pointer.get("snapshot_id")
    generation = pointer.get("generation")
    if (pointer.get("schema") != _POINTER_SCHEMA or not isinstance(name, str)
            or not re.fullmatch(r"state-\d{8}-[0-9a-f]{32}\.pt", name)
            or not isinstance(snapshot_id, str) or _HEX32.fullmatch(snapshot_id) is None
            or type(generation) is not int or generation < 0
            or name != f"state-{generation:08d}-{snapshot_id}.pt"):
        return None, None
    path = recovery_dir / name
    size, state = _regular_file_size(path)
    if state != "ok" or size != pointer.get("size_bytes"):
        return None, None
    try:
        info = path.lstat()
    except OSError:
        return None, None
    return path, (info.st_dev, info.st_ino, info.st_size,
                  info.st_mtime_ns, info.st_ctime_ns)


def _primitive_logical_context(payload: Mapping[str, Any], pointer: Mapping[str, Any]) -> tuple[
    dict[str, Any], list[dict[str, str]], bool, bool
]:
    """Return allowlisted context, blockers, pending optimizer state and completeness."""
    blockers: list[dict[str, str]] = []
    complete = True
    pending_attempt = payload.get("pending_attempt")
    prepared = payload.get("prepared_windows")
    if pending_attempt is not None:
        blockers.append(_block(
            "optimizer_attempt_pending",
            "Checkpoint contains an unfinished optimizer attempt.",
            "managed_recovery_required",
        ))
    prepared_count = len(prepared) if isinstance(prepared, list) else -1
    if prepared_count < 0 or prepared_count > 1:
        complete = False
        blockers.append(_block(
            "checkpoint_prepared_windows_invalid",
            "Checkpoint prepared-window metadata is invalid or unsupported.",
            "managed_recovery_required",
        ))
    elif prepared_count:
        blockers.append(_block(
            "optimizer_window_prepared",
            "Checkpoint contains a prepared optimizer window requiring managed recovery.",
            "managed_recovery_required",
        ))

    context = payload.get("last_update_context")
    if context is None:
        return ({"state": "no_saved_logical_run", "run_id": None,
                 "requested_steps": None, "completed_updates": None,
                 "remaining_steps": None, "task_ids": [],
                 "task_worker_owners": {}, "rescue_ready_task_ids": [],
                 "context_state": "absent"}, blockers,
                pending_attempt is not None or bool(prepared_count), False)
    if not isinstance(context, Mapping) or context.get("schema") != "qwen-rl-recovery-context-v1":
        return ({"state": "unknown", "run_id": None, "requested_steps": None,
                 "completed_updates": None, "remaining_steps": None,
                 "task_ids": [], "task_worker_owners": {}, "rescue_ready_task_ids": [],
                 "context_state": "unsupported"}, blockers, pending_attempt is not None or bool(prepared_count), False)
    logical = context.get("logical_job")
    if not isinstance(logical, Mapping):
        blockers.append(_block(
            "logical_run_context_invalid",
            "Checkpoint logical-run metadata is missing or malformed.",
            "managed_recovery_required",
        ))
        return ({"state": "unknown", "run_id": None, "requested_steps": None,
                 "completed_updates": None, "remaining_steps": None,
                 "task_ids": [], "task_worker_owners": {}, "rescue_ready_task_ids": [],
                 "context_state": "invalid"}, blockers, pending_attempt is not None or bool(prepared_count), False)
    run_id = _safe_id(logical.get("run_id"))
    lineage_id = _safe_id(context.get("lineage_id"))
    task_ids = _string_ids(logical.get("task_ids"), allowed=None)
    requested, completed = logical.get("requested_steps"), logical.get("completed_updates")
    if (run_id is None or lineage_id is None or task_ids is None or not task_ids
            or logical.get("kind") not in {"training", "selftest", "diagnostic"}
            or type(requested) is not int or requested < 1
            or type(completed) is not int or completed < 0 or completed > requested
            or context.get("engine_epoch") != payload.get("engine_epoch")):
        blockers.append(_block(
            "logical_run_context_invalid",
            "Checkpoint logical-run identity or progress fields are invalid.",
            "managed_recovery_required",
        ))
        return ({"state": "unknown", "run_id": run_id,
                 "requested_steps": None, "completed_updates": None,
                 "remaining_steps": None, "task_ids": [],
                 "task_worker_owners": {}, "rescue_ready_task_ids": [],
                 "context_state": "invalid"}, blockers, pending_attempt is not None or bool(prepared_count), False)
    cursor = context.get("task_cursor")
    if (not isinstance(cursor, Mapping)
            or any(type(cursor.get(key)) is not int or cursor[key] < 0
                   for key in ("next_task_index", "next_group_index"))):
        blockers.append(_block(
            "logical_run_cursor_invalid",
            "Checkpoint task cursor is invalid.",
            "managed_recovery_required",
        ))
        complete = False
    task_set = set(task_ids)
    owners = context.get("task_worker_owners", {})
    if (not isinstance(owners, Mapping) or len(owners) > len(task_ids)
            or any(_safe_id(task_id) is None or task_id not in task_set
                   or _safe_owner_id(worker_id) is None for task_id, worker_id in owners.items())):
        blockers.append(_block(
            "logical_run_owner_map_invalid",
            "Checkpoint task-owner mapping is malformed.",
            "managed_recovery_required",
        ))
        return ({"state": "unknown", "run_id": run_id,
                 "requested_steps": requested, "completed_updates": completed,
                 "remaining_steps": max(0, requested - completed),
                 "task_ids": task_ids, "task_worker_owners": {},
                 "rescue_ready_task_ids": [], "context_state": "invalid"},
                blockers, pending_attempt is not None or bool(prepared_count), False)
    rescue_ids = _string_ids(context.get("rescue_ready_task_ids", []), allowed=task_set)
    if rescue_ids is None:
        complete = False
        blockers.append(_block(
            "logical_run_rescue_task_ids_invalid",
            "Checkpoint BEPA rescue task identifiers are malformed.",
            "managed_recovery_required",
        ))
        rescue_ids = []
    expected_version = pointer.get("policy_version")
    if type(expected_version) is not int:
        complete = False
    remaining = requested - completed
    return ({
        "state": "pending" if remaining > 0 else "complete",
        "run_id": run_id,
        "lineage_id": lineage_id,
        "snapshot_id": pointer.get("snapshot_id"),
        "requested_steps": requested,
        "completed_updates": completed,
        "remaining_steps": remaining,
        "task_ids": task_ids,
        "task_worker_owners": dict(owners),
        "rescue_ready_task_ids": rescue_ids,
        "checkpoint_policy_version": expected_version,
        "context_state": "verified_shape",
    }, blockers, pending_attempt is not None or bool(prepared_count), complete)


def _inspect_checkpoint(
    recovery_dir: Path, pointer: Mapping[str, Any], current_dataset_dir: str | None = None,
    path_mappings: Mapping[str, str] | None = None,
) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, str]], bool
]:
    path, before = _snapshot_file(recovery_dir, pointer)
    if path is None or before is None:
        return ({"state": "invalid", "error_code": "checkpoint_snapshot_identity_invalid"},
                {"state": "unknown", "task_ids": [], "task_worker_owners": {},
                 "rescue_ready_task_ids": [], "context_state": "unknown"},
                [_block("checkpoint_snapshot_identity_invalid",
                        "The checkpoint snapshot no longer matches its durable pointer.",
                        "inspect_checkpoint_metadata")], False)
    try:
        import torch

        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino, opened.st_size) != before[:3]):
                raise OSError("checkpoint snapshot changed before inspection")
            payload = torch.load(
                f"/proc/self/fd/{descriptor}", map_location="meta",
                weights_only=False, mmap=True,
            )
        finally:
            os.close(descriptor)
    except Exception:
        return ({"state": "incomplete", "error_code": "checkpoint_inspection_failed"},
                {"state": "unknown", "task_ids": [], "task_worker_owners": {},
                 "rescue_ready_task_ids": [], "context_state": "unavailable"},
                [_block("checkpoint_inspection_failed",
                        "Checkpoint metadata could not be safely inspected.",
                        "inspect_checkpoint_metadata")], False)
    try:
        current = path.lstat()
        current_identity = (
            current.st_dev, current.st_ino, current.st_size,
            current.st_mtime_ns, current.st_ctime_ns,
        )
    except OSError:
        current_identity = None
    latest, latest_state = _read_json(recovery_dir / "latest.json")
    if (not isinstance(payload, Mapping) or payload.get("schema") != _CHECKPOINT_SCHEMA):
        return ({"state": "unsupported_checkpoint_schema",
                 "error_code": "checkpoint_schema_unrecognized"},
                {"state": "unknown", "task_ids": [], "task_worker_owners": {},
                 "rescue_ready_task_ids": [], "context_state": "unsupported"},
                [], False)
    if (current_identity != before or latest_state != "ok" or latest != dict(pointer)
            or payload.get("generation") != pointer.get("generation")
            or payload.get("snapshot_id") != pointer.get("snapshot_id")
            or payload.get("policy_version") != pointer.get("policy_version")
            or payload.get("optimizer_steps") != pointer.get("optimizer_steps")
            or type(payload.get("scheduler_steps")) is not int
            or payload.get("scheduler_steps") != payload.get("optimizer_steps")
            or not isinstance(payload.get("engine_epoch"), str)
            or _HEX32.fullmatch(payload["engine_epoch"]) is None):
        return ({"state": "invalid", "error_code": "checkpoint_changed_or_identity_invalid"},
                {"state": "unknown", "task_ids": [], "task_worker_owners": {},
                 "rescue_ready_task_ids": [], "context_state": "invalid"},
                [_block("checkpoint_changed_or_identity_invalid",
                        "Checkpoint identity changed or disagreed with its durable pointer.",
                        "inspect_checkpoint_metadata")], False)
    logical, blockers, optimizer_pending, complete = _primitive_logical_context(payload, pointer)
    context = payload.get("last_update_context")
    source_identity = context.get("source_identity") if isinstance(context, Mapping) else None
    saved_dataset_dir = source_identity.get("dataset_dir") if isinstance(source_identity, Mapping) else None
    if isinstance(saved_dataset_dir, str) and isinstance(current_dataset_dir, str):
        from .instance_relocation import path_identity_matches

        try:
            same_path = Path(saved_dataset_dir).expanduser().resolve() == Path(current_dataset_dir).expanduser().resolve()
            mapped_path = path_identity_matches(saved_dataset_dir, current_dataset_dir, path_mappings)
        except (OSError, TypeError, ValueError):
            same_path = mapped_path = False
        logical["dataset_path_state"] = (
            "same" if same_path else "relocated_by_explicit_map" if mapped_path else "mismatch"
        )
        logical["dataset_semantic_identity_state"] = "not_checked_by_read_only_plan"
        if not mapped_path:
            blockers.append(_block(
                "dataset_path_identity_mismatch",
                "The saved run dataset path differs from the configured dataset and has no matching explicit path map.",
                "review_dataset_path_map_and_semantic_identity",
            ))
    else:
        logical["dataset_path_state"] = "unavailable"
        complete = False
    checkpoint = {
        "state": "payload_metadata_inspected",
        "generation": pointer["generation"],
        "snapshot_id": pointer["snapshot_id"],
        "policy_version": pointer["policy_version"],
        "optimizer_steps": pointer["optimizer_steps"],
        "scheduler_state_present": isinstance(payload.get("scheduler_state"), Mapping),
        "scheduler_class_present": isinstance(payload.get("scheduler_class"), str),
        "lineage_id": payload.get("lineage_id") if _HEX32.fullmatch(
            payload.get("lineage_id", "") if isinstance(payload.get("lineage_id"), str) else ""
        ) else None,
        "optimizer_transaction_pending": optimizer_pending,
        "restore_validation": "not_performed",
        "inspection_scope": "meta_device_metadata_only_trusted_checkpoint",
    }
    if checkpoint["lineage_id"] is None:
        blockers.append(_block("checkpoint_lineage_invalid",
                               "Checkpoint lineage metadata is invalid.",
                               "managed_recovery_required"))
        complete = False
    if not checkpoint["scheduler_state_present"] or not checkpoint["scheduler_class_present"]:
        blockers.append(_block("checkpoint_scheduler_metadata_invalid",
                               "Checkpoint scheduler metadata is missing or invalid.",
                               "managed_recovery_required"))
        complete = False
    return checkpoint, logical, blockers, complete


def _sqlite_sidecar_state(path: Path) -> str:
    for sidecar in (Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            sidecar.lstat()
            return "wal_sidecar_present"
        except FileNotFoundError:
            continue
        except OSError:
            return "wal_sidecar_state_unknown"
    return "absent"


@contextmanager
def _readonly_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Read a checkpointed DB without creating or touching WAL sidecars."""
    sidecar_state = _sqlite_sidecar_state(path)
    if sidecar_state != "absent":
        raise _ReadOnlyInspectionDeferred(sidecar_state)
    try:
        before = path.lstat()
    except OSError:
        raise _ReadOnlyInspectionDeferred("database_identity_unknown") from None
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise _ReadOnlyInspectionDeferred("database_identity_unknown")
    uri = f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=0.5)
    except sqlite3.Error:
        raise
    try:
        connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()
    try:
        after = path.lstat()
    except OSError:
        raise _ReadOnlyInspectionDeferred("database_identity_changed") from None
    if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            or _sqlite_sidecar_state(path) != "absent"):
        raise _ReadOnlyInspectionDeferred("database_changed_during_inspection")


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _inspect_gradient_store(path: Path, desired_pairs: list[dict[str, str]], *,
                            checkpoint_clean: bool) -> tuple[dict[str, Any], list[dict[str, str]], bool]:
    if not path.exists():
        return ({"state": "missing", "staged_windows": 0,
                 "nonterminal_optimizer_windows": 0,
                 "cleanup_eligibility": "no_existing_store"}, [], True)
    blockers: list[dict[str, str]] = []
    result: dict[str, Any] = {"state": "unknown", "staged_windows": None,
                              "nonterminal_optimizer_windows": None,
                              "cleanup_eligibility": "unknown"}
    try:
        with _readonly_connection(path) as connection:
            if not {"windows", "jobs", "store_metadata"} <= {
                row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }:
                return ({**result, "state": "unsupported_schema"},
                    [], False)
            window_columns = _table_columns(connection, "windows")
            job_columns = _table_columns(connection, "jobs")
            metadata_columns = _table_columns(connection, "store_metadata")
            if not {"window_id", "attempt_id", "window_json", "state"} <= window_columns or not {
                "window_id", "attempt_id", "state"
            } <= job_columns or not {"key", "value"} <= metadata_columns:
                return ({**result, "state": "unsupported_schema"},
                        [], False)
            rows = connection.execute(
                "SELECT state,substr(window_json,1,262145),length(window_json) "
                "FROM windows ORDER BY created_at LIMIT 10001"
            ).fetchall()
            if len(rows) > 10_000:
                return ({**result, "state": "too_many_windows"},
                        [_block("gradient_store_inspection_bounded",
                                "Gradient store exceeds the bounded preflight row limit.",
                                "managed_recovery_required")], False)
            staged_counts: dict[str, int] = {}
            staged_keys: set[tuple[str, str]] = set()
            optimizer_pending = 0
            staged_unsafe = 0
            for state, window_json, window_size in rows:
                if (not isinstance(window_json, str) or type(window_size) is not int
                        or window_size > 256 * 1024 or len(window_json) != window_size):
                    return ({**result, "state": "window_metadata_too_large"},
                            [_block("gradient_store_window_metadata_unknown",
                                    "A gradient window exceeds the metadata inspection limit.",
                                    "managed_recovery_required")], False)
                try:
                    window = json.loads(window_json)
                except (TypeError, json.JSONDecodeError):
                    return ({**result, "state": "window_metadata_invalid"},
                            [_block("gradient_store_window_metadata_invalid",
                                    "Gradient store window metadata is invalid.",
                                    "managed_recovery_required")], False)
                context = window.get("update_context") if isinstance(window, dict) else None
                is_staged = isinstance(context, dict) and context.get("gradient_scope") == _STAGED_SCOPE
                if is_staged:
                    staged_counts[state] = staged_counts.get(state, 0) + 1
                    window_id, attempt_id = window.get("window_id"), window.get("attempt_id")
                    if isinstance(window_id, str) and isinstance(attempt_id, str):
                        staged_keys.add((window_id, attempt_id))
                    if state in {"applied", "applying"} or state not in {
                        "pending", "cancelled", "failed",
                    }:
                        staged_unsafe += 1
                elif state in {"pending", "applying"}:
                    optimizer_pending += 1
            metadata_row = connection.execute(
                "SELECT substr(value,1,65537),length(value) FROM store_metadata "
                "WHERE key='executor_worker_ids' LIMIT 1"
            ).fetchone()
            stored_map: dict[str, str] | None = None
            if metadata_row is not None:
                if type(metadata_row[1]) is not int or metadata_row[1] > 64 * 1024:
                    return ({**result, "state": "executor_map_too_large"},
                            [_block("gradient_store_executor_map_bounded",
                                    "Persisted executor mapping exceeds the inspection limit.",
                                    "managed_recovery_required")], False)
                parsed = json.loads(metadata_row[0])
                if (isinstance(parsed, dict)
                        and all(_safe_owner_id(key) and _safe_owner_id(value)
                                for key, value in parsed.items())):
                    stored_map = parsed
                else:
                    return ({**result, "state": "invalid_executor_map"},
                            [_block("gradient_store_executor_map_invalid",
                                    "Persisted executor mapping is invalid.",
                                    "managed_recovery_required")], False)
            active_rows = connection.execute(
                "SELECT window_id,attempt_id FROM jobs WHERE state IN ('queued','leased') LIMIT 10001"
            ).fetchall()
            if len(active_rows) > 10_000:
                return ({**result, "state": "too_many_jobs"},
                        [_block("gradient_store_inspection_bounded",
                                "Gradient store exceeds the bounded preflight job limit.",
                                "managed_recovery_required")], False)
            active_jobs = sum(
                (row[0], row[1]) not in staged_keys
                for row in active_rows
            )
    except _ReadOnlyInspectionDeferred as error:
        return ({**result, "state": error.state}, [], False)
    except sqlite3.Error:
        return ({**result, "state": "unavailable"},
                [_block("gradient_store_unavailable",
                        "Gradient store could not be inspected read-only.",
                        "managed_recovery_required")], False)
    except (ValueError, TypeError, OSError):
        return ({**result, "state": "invalid_metadata"},
                [_block("gradient_store_metadata_invalid",
                        "Gradient store metadata could not be validated.",
                        "managed_recovery_required")], False)

    staged_total = sum(staged_counts.values())
    desired_map = {row["executor_id"]: row["worker_id"] for row in desired_pairs}
    map_change = stored_map is not None and stored_map != desired_map
    safe_stage_rows = staged_unsafe == 0
    if optimizer_pending or active_jobs:
        blockers.append(_block("gradient_optimizer_window_pending",
                               "Nonterminal optimizer-authoritative gradient work remains.",
                               "managed_recovery_required"))
    if staged_unsafe:
        blockers.append(_block("staged_window_optimizer_authority_conflict",
                               "A staged window has a state that cannot be retired as pre-optimizer work.",
                               "managed_recovery_required"))
    cleanup = (
        "candidate_after_clean_durable_core_restore"
        if staged_total and safe_stage_rows and checkpoint_clean and not optimizer_pending and not active_jobs
        else "requires_clean_durable_checkpoint" if staged_total and safe_stage_rows
        else "no_nonterminal_staging" if not staged_total
        else "blocked"
    )
    return ({
        "state": "inspected",
        "staged_windows": staged_total,
        "staged_window_states": dict(sorted(staged_counts.items())),
        "nonterminal_optimizer_windows": optimizer_pending,
        "queued_or_leased_jobs": int(active_jobs),
        "stored_executor_worker_ids": stored_map,
        "desired_executor_worker_ids": desired_map,
        "executor_map_change_required": map_change,
        "cleanup_eligibility": cleanup,
        "cleanup_is_advisory": True,
    }, blockers, True)


def _inspect_bepa_store(
    path: Path,
    task_ids: list[str],
    desired: Mapping[str, str],
    *,
    run_id: str | None = None,
    lineage_id: str | None = None,
    snapshot_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]], bool]:
    if not path.exists():
        return ({"state": "missing", "matched_tasks": 0, "owner_bindings": []}, [], False)
    blockers: list[dict[str, str]] = []
    try:
        with _readonly_connection(path) as connection:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"tasks", "rescue_jobs", "events"} <= tables:
                return ({"state": "unsupported_schema", "matched_tasks": 0,
                         "owner_bindings": []}, [], False)
            if (not {"task_key", "spec_json"} <= _table_columns(connection, "tasks")
                    or not {"job_id", "task_key"} <= _table_columns(connection, "rescue_jobs")
                    or not {"entity_type", "entity_id", "action", "details_json"}
                    <= _table_columns(connection, "events")):
                return ({"state": "unsupported_schema", "matched_tasks": 0,
                         "owner_bindings": []}, [], False)
            rows = connection.execute(
                "SELECT t.task_key,substr(t.spec_json,1,?),length(t.spec_json),"
                "e.sequence,e.action,substr(e.details_json,1,?),length(e.details_json) FROM tasks t "
                "JOIN events e ON e.entity_type='job' "
                "AND json_extract(e.details_json,'$.task_key')=t.task_key "
                "JOIN rescue_jobs j ON e.entity_id=j.job_id "
                "WHERE e.action IN ('rescue_owner_bound','rescue_origin_group_bound',"
                "'rescue_owner_migrated') ORDER BY e.sequence LIMIT ?",
                (_MAX_TASK_SPEC_BYTES + 1, _MAX_EVENT_BYTES + 1, _MAX_OWNER_ROWS + 1),
            ).fetchall()
            if len(rows) > _MAX_OWNER_ROWS:
                return ({"state": "too_many_owner_events", "matched_tasks": 0,
                         "owner_bindings": []},
                        [_block("bepa_owner_inspection_bounded",
                                "BEPA owner history exceeds the bounded preflight row limit.",
                                "managed_recovery_required")], False)
            owner_events_by_key: dict[str, dict[str, Any]] = {}
            for task_key, task_json, task_size, sequence, action, details_json, details_size in rows:
                if (not isinstance(task_json, str) or len(task_json) > _MAX_TASK_SPEC_BYTES
                        or type(task_size) is not int or task_size > _MAX_TASK_SPEC_BYTES
                        or len(task_json) != task_size
                        or not isinstance(details_json, str) or len(details_json) > _MAX_EVENT_BYTES
                        or type(details_size) is not int or details_size > _MAX_EVENT_BYTES
                        or len(details_json) != details_size):
                    return ({"state": "owner_metadata_too_large", "matched_tasks": 0,
                             "owner_bindings": []},
                            [_block("bepa_owner_metadata_bounded",
                                    "BEPA owner metadata exceeds the inspection limit.",
                                    "managed_recovery_required")], False)
                spec = json.loads(task_json)
                details = json.loads(details_json)
                if not isinstance(details, dict):
                    return ({"state": "invalid_owner_metadata", "matched_tasks": 0,
                             "owner_bindings": []},
                            [_block("bepa_owner_metadata_invalid",
                                    "A matching BEPA owner event is malformed.",
                                    "managed_recovery_required")], False)
                from ..contracts.tool_call_failures import ENV_SCHEMA_VERSION
                contract = spec.get("tool_contract") if isinstance(spec, dict) else None
                if not isinstance(contract, dict) or contract.get("arc_env_schema") != ENV_SCHEMA_VERSION:
                    continue
                challenge = spec.get("public_challenge") if isinstance(spec, dict) else None
                public_task_id = challenge.get("task_id") if isinstance(challenge, dict) else None
                if not isinstance(task_key, str) or not task_key or len(task_key) > 256:
                    return ({"state": "invalid_owner_metadata", "matched_tasks": 0,
                             "owner_bindings": []},
                            [_block("bepa_owner_metadata_invalid",
                                    "A matching BEPA task key is invalid.",
                                    "managed_recovery_required")], False)
                owner_fields = (
                    ("from_worker_id", "from_executor_id", "to_worker_id", "to_executor_id")
                    if action == "rescue_owner_migrated" else ("worker_id", "executor_id")
                )
                if (any(_safe_owner_id(details.get(field)) is None for field in owner_fields)
                        or (action == "rescue_owner_migrated"
                            and (details.get("schema") != "bepa-owner-migration/v1"
                                 or not isinstance(details.get("migration_id"), str)
                                 or not details.get("migration_id")
                                 or any(not isinstance(details.get(field), str) or not details[field]
                                        for field in ("task_id", "run_id", "lineage_id", "snapshot_id", "reason"))))
                        or (action == "rescue_origin_group_bound"
                            and (not isinstance(details.get("original_group_id"), str)
                                 or not details["original_group_id"]))):
                    return ({"state": "invalid_owner_metadata", "matched_tasks": 0,
                             "owner_bindings": []},
                            [_block("bepa_owner_metadata_invalid",
                                    "A matching BEPA owner event is invalid.",
                                    "managed_recovery_required")], False)
                migration_task_id = details.get("task_id") if action == "rescue_owner_migrated" else None
                if ((public_task_id is not None and _safe_id(public_task_id) is None)
                        or (migration_task_id is not None and _safe_id(migration_task_id) is None)
                        or (public_task_id is not None and migration_task_id is not None
                            and public_task_id != migration_task_id)):
                    return ({"state": "invalid_owner_metadata", "matched_tasks": 0,
                             "owner_bindings": []},
                            [_block("bepa_owner_metadata_invalid",
                                    "BEPA public and migrated task IDs differ or are invalid.",
                                    "managed_recovery_required")], False)
                task_id_hint = public_task_id or migration_task_id
                entry = owner_events_by_key.setdefault(
                    task_key, {"task_ids": set(), "events": []},
                )
                if task_id_hint is not None:
                    entry["task_ids"].add(task_id_hint)
                entry["events"].append({
                    "sequence": int(sequence), "action": action, "details": details,
                })
    except _ReadOnlyInspectionDeferred as error:
        return ({"state": error.state, "matched_tasks": 0,
                 "owner_bindings": []}, [], False)
    except sqlite3.Error:
        return ({"state": "unavailable", "matched_tasks": 0, "owner_bindings": []},
                [_block("bepa_store_unavailable",
                        "BEPA owner records could not be inspected read-only.",
                        "managed_recovery_required")], False)
    except (ValueError, TypeError, OSError):
        return ({"state": "invalid_metadata", "matched_tasks": 0, "owner_bindings": []},
                [_block("bepa_owner_metadata_invalid",
                        "BEPA owner metadata could not be validated.",
                        "managed_recovery_required")], False)

    from ..bepa.owner_migration import OwnerHistoryError, fold_owner_events
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    unmapped_bindings: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    complete = True
    for task_key, entry in owner_events_by_key.items():
        task_ids_for_key = entry["task_ids"]
        if len(task_ids_for_key) > 1:
            complete = False
            blockers.append(_block(
                "bepa_owner_task_id_conflict",
                "One BEPA task key maps to multiple checkpoint task IDs.",
                "review_bepa_task_owner_migration",
            ))
            continue
        task_id_hint = next(iter(task_ids_for_key), None)
        events = entry["events"]
        try:
            history = fold_owner_events(events)
        except OwnerHistoryError:
            history = None
        if history is None:
            complete = False
            blockers.append(_block(
                "bepa_owner_conflict",
                "A current-contract BEPA owner history conflicts or cannot be replayed.",
                "review_bepa_task_owner_migration",
            ))
            unmapped_bindings.append({"task_key": task_key, "state": "owner_history_conflict"})
            continue
        if task_id_hint in task_ids:
            by_task.setdefault(task_id_hint, {})[task_key] = events
        elif task_id_hint is None:
            owner = history["owner"]
            if desired.get(owner["worker_id"]) != owner["executor_id"]:
                complete = False
                blockers.append(_block(
                    "bepa_owner_task_id_unmapped",
                    "A BEPA owner absent from the target topology has no checkpoint task ID mapping.",
                    "review_bepa_task_owner_migration",
                ))
                unmapped_bindings.append({
                    "task_key": task_key,
                    "effective_owner": owner,
                    **({"owner_migrations": history["owner_migrations"]}
                       if history["owner_migrations"] else {}),
                })

    bindings = []
    for task_id in sorted(by_task):
        task_keys = by_task[task_id]
        if len(task_keys) != 1:
            complete = False
            bindings.append({"task_id": task_id, "state": "multiple_task_contracts"})
            continue
        task_key, events = next(iter(task_keys.items()))
        try:
            history = fold_owner_events(events)
        except OwnerHistoryError:
            history = None
        if history is None:
            complete = False
            blockers.append(_block("bepa_owner_conflict",
                                   "A pending task has conflicting or invalid BEPA owner events.",
                                   "restore_compatible_logical_map_or_review_migration"))
            bindings.append({"task_id": task_id, "task_key": task_key,
                             "state": "owner_history_conflict"})
            continue
        owner = history["owner"]
        binding = {"task_id": task_id, "owners": [owner]}
        if history["owner_migrations"]:
            binding["original_owner"] = history["initial_owner"]
            binding["owner_migrations"] = history["owner_migrations"]
        bindings.append(binding)
        if desired.get(owner["worker_id"]) != owner["executor_id"]:
            blockers.append(_block("bepa_owner_mismatch",
                                   "A pending task's effective BEPA owner is absent from the desired topology.",
                                   "restore_compatible_logical_map_or_review_migration"))
    return ({"state": "inspected" if complete else "ambiguous_task_contracts",
             "matched_tasks": len(by_task),
             "owner_bindings": bindings,
             "owner_migration_supported": True,
             "unmapped_owner_bindings": unmapped_bindings,
             "migration_scope": {"run_id": run_id, "lineage_id": lineage_id,
                                 "snapshot_id": snapshot_id}}, blockers, complete)


def build_resume_plan(
    deployment_path: Path,
    *,
    gpus: tuple[int, ...] | None = None,
    inspect_checkpoint: bool = False,
) -> dict[str, Any]:
    """Build a read-only, bounded plan; it never starts roles or changes recovery state."""
    from dataclasses import replace
    from . import instance_plan

    report: dict[str, Any] = {
        "schema": _SCHEMA,
        "status": "incomplete",
        "desired_topology": {},
        "checkpoint": {"state": "unknown", "restore_validation": "not_performed"},
        "logical_run": {"state": "unknown", "task_ids": [], "task_worker_owners": {}},
        "staged_gradients": {"state": "not_inspected"},
        "bepa_affinity": {"state": "not_inspected"},
        "blockers": [],
        "next_commands": [],
        "advisories": [],
        "namespace_scope": _NAMESPACE_SCOPE,
        "not_performed": [
            "full optimizer/model restore or numerical state validation",
            "GPU compute or Worker/provider health checks",
            "staged-gradient cancellation or executor/Worker rebinding",
            "BEPA budget, reservation, cache, or usage mutation",
            "full ARC source-semantic comparison and worker readiness",
            "proof that a container using a different mount namespace shares this lock inode",
            "security validation of a hostile Torch pickle; explicit inspection assumes the checkpoint is trusted",
        ],
    }
    try:
        deployment = instance_plan.load_deployment(Path(deployment_path))
        if gpus is not None:
            if (not isinstance(gpus, tuple) or not gpus or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
                    or len(set(gpus)) != len(gpus)):
                raise ValueError
            deployment = replace(deployment, gpus=gpus)
        runtime = instance_plan._load_runtime(deployment)
    except Exception:
        report["blockers"].append(_block(
            "deployment_or_runtime_invalid",
            "Deployment or central-gradient runtime configuration could not be validated.",
            "run_instance_doctor",
        ))
        report["next_commands"] = [_operator_command("check", Path(deployment_path), gpus)]
        report["status"] = "blocked"
        return report

    desired, pairs, topology_blockers = _desired_topology(deployment, runtime)
    report["desired_topology"] = desired
    report["blockers"].extend(topology_blockers)
    if desired.get("gpu_occupancy", {}).get("state") == "busy":
        report["advisories"].append(
            "Review the listed selected-GPU PIDs in their owning runtime; this plan never signals or kills processes."
        )
    desired_map = {row["worker_id"]: row["executor_id"] for row in pairs}

    try:
        process_state = _instance_state(deployment)
    except Exception:
        process_state = "unknown"
    if process_state != "stopped":
        report["checkpoint"] = {"state": "not_inspected_while_instance_may_be_live",
                                 "restore_validation": "not_performed"}
        report["staged_gradients"] = {"state": "not_inspected_while_instance_may_be_live"}
        report["bepa_affinity"] = {"state": "not_inspected_while_instance_may_be_live"}
        report["not_performed"].append("checkpoint and SQLite inspection because the source instance is not proven stopped")
        active = process_state == "active"
        report["blockers"].append(_block(
            "instance_active" if active else "instance_ownership_unknown",
            "The managed instance is active." if active else
            "Managed process ownership could not be verified as stopped.",
            "inspect_instance_status_without_stopping_it",
        ))
        report["next_commands"] = [_operator_command("status", Path(deployment_path), gpus)]
        report["status"] = "blocked" if active else "incomplete"
        return report

    from ..runtime.config import _resolve_path

    recovery_dir = (
        _resolve_path(runtime.learner.recovery_dir)
        if runtime.learner.recovery_dir else runtime.run / "training"
    )
    lock_state, lock_path = _probe_checkpoint_lock(recovery_dir)
    report["checkpoint_owner_lock"] = {
        "state": lock_state,
        "file": (lock_path.name if re.fullmatch(r"[A-Za-z0-9._-]+", lock_path.name)
                 else "redacted"),
        "probe": "O_RDONLY_nonblocking_exclusive_flock",
        "scope": "this_filesystem_namespace",
    }
    latest_exists = (recovery_dir / "latest.json").exists()
    if lock_state == "busy":
        report["checkpoint"] = {
            "state": "not_inspected_optimizer_owner_lock_busy",
            "restore_validation": "not_performed",
        }
        report["staged_gradients"] = {"state": "not_inspected_owner_lock_busy"}
        report["bepa_affinity"] = {"state": "not_inspected_owner_lock_busy"}
        report["blockers"].append(_block(
            "optimizer_owner_lock_busy",
            "The central optimizer checkpoint lock is held in this filesystem namespace.",
            "inspect_managed_instance_status_without_stopping_it",
        ))
        report["next_commands"] = [_operator_command("status", Path(deployment_path), gpus)]
        report["status"] = "blocked"
        return report
    if lock_state != "unlocked":
        if latest_exists:
            report["checkpoint"] = {
                "state": "not_inspected_owner_lock_unproven",
                "restore_validation": "not_performed",
            }
            report["staged_gradients"] = {"state": "not_inspected_owner_lock_unproven"}
            report["bepa_affinity"] = {"state": "not_inspected_owner_lock_unproven"}
            report["not_performed"].append(
                "checkpoint payload and SQLite inspection because the owner lock is absent or unavailable"
            )
            report["advisories"].append(
                "The configured recovery directory has a checkpoint but no visible owner lock; "
                "checkpoint and SQLite inspection were skipped, and no lock file was created."
            )
            report["next_commands"] = [_operator_command("status", Path(deployment_path), gpus)]
        else:
            report["checkpoint"] = {"state": "missing", "restore_validation": "not_performed"}
            report["logical_run"] = {"state": "no_checkpoint", "task_ids": [],
                                     "task_worker_owners": {}}
            report["staged_gradients"] = {"state": "not_inspected_without_checkpoint"}
            report["bepa_affinity"] = {"state": "not_inspected_without_checkpoint"}
            report["next_commands"] = [_operator_command("status", Path(deployment_path), gpus)]
        report["status"] = "blocked" if report["blockers"] else "incomplete"
        return report

    checkpoint, pointer = _read_checkpoint_metadata(recovery_dir)
    checkpoint["owner_lock"] = "unlocked_in_this_filesystem_namespace"
    report["checkpoint"] = checkpoint
    if checkpoint.get("state") == "missing":
        report["logical_run"] = {"state": "no_checkpoint", "task_ids": [],
                                 "task_worker_owners": {}}
        report["advisories"].append(
            "No durable checkpoint is present in the configured recovery directory; the plan will not create a fresh training line."
        )
        report["next_commands"].append(_operator_command("status", Path(deployment_path), gpus))
    elif checkpoint.get("state") in {"invalid", "pending"}:
        report["blockers"].append(_block("checkpoint_metadata_blocked",
                                         "Checkpoint pointer or transaction sidecar needs managed recovery.",
                                         "managed_recovery_required"))
    elif checkpoint.get("state") != "metadata_consistent" or pointer is None:
        report["blockers"].append(_block("checkpoint_metadata_unknown",
                                         "Checkpoint metadata is unavailable or could not be validated.",
                                         "inspect_checkpoint_metadata"))

    logical = report["logical_run"]
    checkpoint_clean = checkpoint.get("state") == "metadata_consistent"
    gradient_path = runtime.run / "gradient-server" / "jobs.sqlite3"
    staged_complete = False
    context_complete = False
    if inspect_checkpoint and pointer is not None:
        prior_checkpoint_state = checkpoint.get("state")
        prior_sidecars = checkpoint.get("pending_sidecars", [])
        checkpoint, logical, checkpoint_blockers, context_complete = _inspect_checkpoint(
            recovery_dir, pointer, str(Path(runtime.dataset_dir).expanduser().resolve()),
            runtime.learner.recovery_path_mappings,
        )
        if prior_checkpoint_state == "pending":
            checkpoint["metadata_state"] = "pending"
            checkpoint["pending_sidecars"] = prior_sidecars
        report["checkpoint"] = checkpoint
        report["logical_run"] = logical
        report["blockers"].extend(checkpoint_blockers)
        checkpoint_clean = (
            prior_checkpoint_state == "metadata_consistent"
            and checkpoint.get("state") == "payload_metadata_inspected"
            and not checkpoint.get("optimizer_transaction_pending")
        )
    elif pointer is not None:
        logical.update({"state": "unknown", "context_state": "not_inspected"})
        report["next_commands"].append(
            _operator_command("resume_plan", Path(deployment_path), gpus)
        )

    staged, staged_blockers, staged_complete = _inspect_gradient_store(
        gradient_path, pairs, checkpoint_clean=checkpoint_clean,
    )
    report["staged_gradients"] = staged
    report["blockers"].extend(staged_blockers)

    if context_complete and logical.get("state") == "pending":
        owners = logical.get("task_worker_owners", {})
        mismatches = [
            {"task_id": task_id, "worker_id": worker_id}
            for task_id, worker_id in sorted(owners.items())
            if worker_id not in desired_map
        ]
        logical["owner_map_state"] = "compatible" if not mismatches else "incompatible"
        logical["mismatched_task_owners"] = mismatches
        if mismatches:
            report["blockers"].append(_block(
                "task_owner_migration_unsupported",
                "Pending logical-run tasks retain Worker owners absent from the desired topology; owner migration is unsupported.",
                "restore_compatible_logical_map_or_review_migration",
            ))
            report["advisories"].append(
                "Preserve the saved run ID and optimizer state. Do not use a fresh run, a new BEPA scope, or direct checkpoint JSON edits; no task-owner migration API exists."
            )
        else:
            logical["owner_map_state"] = "compatible"
        bepa_enabled = runtime.bepa.enabled
        if bepa_enabled:
            bepa_root = (
                _resolve_path(runtime.bepa.state_dir)
                if runtime.bepa.state_dir else runtime.run / "bepa"
            )
            bepa, bepa_blockers, bepa_complete = _inspect_bepa_store(
                bepa_root / "state.sqlite3", logical["task_ids"], desired_map,
                run_id=logical.get("run_id"),
                lineage_id=logical.get("lineage_id"),
                snapshot_id=logical.get("snapshot_id"),
            )
            report["bepa_affinity"] = bepa
            report["blockers"].extend(bepa_blockers)
            if not bepa_complete:
                context_complete = False
                report["not_performed"].append(
                    "complete BEPA affinity validation because the read-only store scope was unknown"
                )
        else:
            report["bepa_affinity"] = {"state": "disabled", "matched_tasks": 0,
                                        "owner_bindings": []}
        if not staged_complete:
            context_complete = False
    elif context_complete and logical.get("state") == "complete":
        report["bepa_affinity"] = {"state": "not_needed_no_pending_logical_run"}

    if report["blockers"]:
        report["status"] = "blocked"
    elif (not inspect_checkpoint or checkpoint.get("state") != "payload_metadata_inspected"
          or not context_complete or not checkpoint_clean or not staged_complete
          or desired.get("hardware_validation") != "passed"
          or desired.get("gpu_occupancy", {}).get("state") != "idle"):
        report["status"] = "incomplete"
    else:
        report["status"] = "passed"
        report["next_commands"] = [_operator_command("serve", Path(deployment_path), gpus)]
    if report["status"] == "incomplete" and not report["next_commands"]:
        report["next_commands"] = [_operator_command("resume_plan", Path(deployment_path), gpus)]
    if report["status"] == "blocked" and not report["next_commands"]:
        report["next_commands"] = [_operator_command("status", Path(deployment_path), gpus)]
    return report


def format_resume_plan(report: dict[str, Any]) -> str:
    """Format only allowlisted resume-plan fields for operators."""
    if not isinstance(report, dict) or report.get("schema") != _SCHEMA:
        return "Resume plan: unavailable (unsupported report schema)."
    lines = [f"Resume plan: {report.get('status', 'incomplete')}."]
    namespace_scope = report.get("namespace_scope")
    if isinstance(namespace_scope, str):
        lines.append(f"Scope: {namespace_scope}.")
    topology = report.get("desired_topology")
    if isinstance(topology, Mapping):
        lines.append(
            "Desired topology: "
            f"{topology.get('pair_count', 'unknown')} pairs, "
            f"{topology.get('groups_per_update', 'unknown')} groups/update, "
            f"n={topology.get('group_size', 'unknown')}, "
            f"{topology.get('nominal_native_trajectories_per_update', 'unknown')} "
            "nominal native trajectories/update."
        )
        occupancy = topology.get("gpu_occupancy")
        if isinstance(occupancy, Mapping):
            lines.append(f"Selected GPU occupancy: {occupancy.get('state', 'unknown')}.")
            processes = occupancy.get("selected_processes")
            if isinstance(processes, list):
                for row in processes:
                    if (isinstance(row, Mapping) and type(row.get("gpu_index")) is int
                            and isinstance(row.get("pids"), list)
                            and all(type(pid) is int for pid in row["pids"])):
                        lines.append(
                            f"Selected GPU {row['gpu_index']} compute PIDs: "
                            f"{', '.join(map(str, row['pids']))}."
                        )
    checkpoint = report.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        lines.append(f"Checkpoint: {checkpoint.get('state', 'unknown')}.")
        lines.append(f"Restore validation: {checkpoint.get('restore_validation', 'not_performed')}.")
    owner_lock = report.get("checkpoint_owner_lock")
    if isinstance(owner_lock, Mapping):
        lines.append(f"Checkpoint owner lock: {owner_lock.get('state', 'unknown')}.")
    logical = report.get("logical_run")
    if isinstance(logical, Mapping):
        lines.append(
            f"Logical run: {logical.get('state', 'unknown')}; "
            f"remaining steps {logical.get('remaining_steps', 'unknown')}."
        )
        mismatches = logical.get("mismatched_task_owners")
        if isinstance(mismatches, list):
            for row in mismatches[:100]:
                if isinstance(row, Mapping):
                    lines.append(f"Owner mismatch: task {row.get('task_id')} is pinned to {row.get('worker_id')}.")
    staged = report.get("staged_gradients")
    if isinstance(staged, Mapping):
        lines.append(
            f"Staged gradients: {staged.get('staged_windows', 'unknown')} windows; "
            f"cleanup {staged.get('cleanup_eligibility', staged.get('state', 'unknown'))}."
        )
    blockers = report.get("blockers")
    if isinstance(blockers, list):
        for item in blockers:
            if isinstance(item, Mapping) and isinstance(item.get("message"), str):
                lines.append(f"Blocker [{item.get('code', 'unknown')}]: {item['message']}")
    commands = report.get("next_commands")
    if isinstance(commands, list):
        for command in commands:
            if isinstance(command, str):
                lines.append(f"Next: {command}")
    advisories = report.get("advisories")
    if isinstance(advisories, list):
        for advisory in advisories:
            if isinstance(advisory, str):
                lines.append(f"Note: {advisory}")
    return "\n".join(lines)
