"""Read-only, bounded operator summary for one managed instance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping

import yaml

_SCHEMA = "qwen-rl-instance-operator-status/v1"
_MAX_JSON_BYTES = 128 * 1024
_MAX_RUNTIME_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_STAMP_BYTES = 32 * 1024
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT = re.compile(r"^state-(\d{8})-([0-9a-f]{32})\.pt$")
_RUNTIME_BUILD_FILES = frozenset({
    "qwen_rl/learner/gradient_executor/client.py",
    "qwen_rl/learner/gradient_server/http.py",
    "qwen_rl/learner/gradient_server/store.py",
})
_CHECKPOINT_SIDECARS = {
    "prepared-window.json": "central-gradient-prepared-window/v1",
    "pending-attempt.json": "central-gradient-pending-attempt/v1",
    "async-checkpoint-intent.json": "central-gradient-async-checkpoint-intent/v1",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _read_bounded(path: Path, limit: int) -> tuple[bytes | None, str]:
    """Read one regular file without following its final symlink or exceeding limit."""
    try:
        before = path.lstat()
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "unavailable"
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        return None, "invalid"
    if before.st_size > limit:
        return None, "too_large"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "unavailable"
    try:
        current = os.fstat(descriptor)
        if (not stat.S_ISREG(current.st_mode) or current.st_dev != before.st_dev
                or current.st_ino != before.st_ino):
            return None, "invalid"
        if current.st_size > limit:
            return None, "too_large"
        chunks = bytearray()
        while len(chunks) <= limit:
            chunk = os.read(descriptor, min(64 * 1024, limit + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) > limit:
            return None, "too_large"
        return bytes(chunks), "ok"
    except OSError:
        return None, "unavailable"
    finally:
        os.close(descriptor)


def _regular_file_size(path: Path) -> tuple[int | None, str]:
    """Inspect a snapshot by metadata only; never read its checkpoint payload."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "unavailable"
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        return None, "invalid"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "unavailable"
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_dev != info.st_dev
                or opened.st_ino != info.st_ino):
            return None, "invalid"
        return opened.st_size, "ok"
    except OSError:
        return None, "unavailable"
    finally:
        os.close(descriptor)


def _read_json(path: Path, limit: int = _MAX_JSON_BYTES) -> tuple[dict[str, Any] | None, str]:
    payload, state = _read_bounded(path, limit)
    if state != "ok" or payload is None:
        return None, state
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return None, "invalid"
    if not isinstance(value, dict):
        return None, "invalid"
    return value, "ok"


def _safe_image_reference(value: Any) -> str | None:
    if (not isinstance(value, str) or not value or len(value) > 512
            or any(character.isspace() for character in value) or "://" in value):
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:+-]*(?:@sha256:[0-9a-f]{64})?", value):
        return None
    return value


def _build_identity(status: str, evidence: str, *, revision: str | None = None,
                    files: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "scope": "gradient-overlay", "patch_revision": revision,
            "verified_source_files": files or [], "evidence": evidence}


def read_runtime_build_identity(package_root: str | Path) -> dict[str, Any]:
    """Verify the optional gradient-overlay stamp and its exact source file set."""
    root = Path(package_root).resolve()
    stamp, state = _read_json(root / "runtime-build.json", _MAX_STAMP_BYTES)
    if state == "missing":
        return _build_identity("unknown", "runtime_build_stamp_missing")
    if state != "ok" or stamp is None:
        return _build_identity("invalid", "runtime_build_stamp_unreadable")
    expected_fields = {"schema", "kind", "source_revision", "base_image_reference", "source_files"}
    files = stamp.get("source_files")
    if (set(stamp) != expected_fields
            or stamp.get("schema") != "qwen-rl-runtime-build/v1"
            or stamp.get("kind") != "gradient-overlay"
            or not isinstance(stamp.get("source_revision"), str)
            or _HEX_40.fullmatch(stamp["source_revision"]) is None
            or not isinstance(files, dict) or set(files) != _RUNTIME_BUILD_FILES):
        return _build_identity("invalid", "runtime_build_stamp_schema_invalid")

    repository_root = root.parent
    total = 0
    for relative in sorted(_RUNTIME_BUILD_FILES):
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or relative.startswith("/"):
            return _build_identity("invalid", "runtime_build_source_path_invalid")
        expected_hash = files.get(relative)
        if not isinstance(expected_hash, str) or _HEX_64.fullmatch(expected_hash) is None:
            return _build_identity("invalid", "runtime_build_source_hash_invalid")
        path = repository_root.joinpath(*posix.parts)
        cursor = repository_root
        unsafe_parent = False
        for part in posix.parts[:-1]:
            cursor = cursor / part
            try:
                info = cursor.lstat()
            except OSError:
                unsafe_parent = True
                break
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                unsafe_parent = True
                break
        if unsafe_parent:
            return _build_identity("invalid", "runtime_build_source_path_invalid")
        payload, file_state = _read_bounded(path, _MAX_SOURCE_BYTES)
        if file_state != "ok" or payload is None:
            return _build_identity("invalid", "runtime_build_source_unavailable")
        total += len(payload)
        if total > 3 * _MAX_SOURCE_BYTES or hashlib.sha256(payload).hexdigest() != expected_hash:
            return _build_identity("invalid", "runtime_build_source_hash_mismatch")

    result = _build_identity("verified", "runtime-build.json and matching bounded source SHA-256 values",
                             revision=stamp["source_revision"], files=sorted(_RUNTIME_BUILD_FILES))
    result["base_image_reference_scope"] = "build_arg_label_only"
    image_reference = _safe_image_reference(stamp.get("base_image_reference"))
    if image_reference is not None:
        result["base_image_reference"] = image_reference
    return result


def _sha256_file(path: Path) -> str | None:
    payload, state = _read_bounded(path, _MAX_RUNTIME_CONFIG_BYTES)
    return hashlib.sha256(payload).hexdigest() if state == "ok" and payload is not None else None


def _runtime_config_path(
    state_dir: Path, run_id: Any, configured_path: Any,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(run_id, str) or _HEX_32.fullmatch(run_id) is None:
        return None, "invalid"
    expected = state_dir / "runs" / run_id / "runtime.yaml"
    if not isinstance(configured_path, str) or Path(configured_path) != expected:
        return None, "invalid"
    for directory in (state_dir, state_dir / "runs", expected.parent):
        try:
            info = directory.lstat()
        except OSError:
            return None, "unavailable"
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return None, "invalid"
    payload, state = _read_bounded(expected, _MAX_RUNTIME_CONFIG_BYTES)
    if state != "ok" or payload is None:
        return None, state
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError):
        return None, "invalid"
    if not isinstance(value, dict) or not isinstance(value.get("learner"), dict):
        return None, "invalid"
    return value, "ok"


def _worker_assignments(runtime: Mapping[str, Any], gradient: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    pins: dict[str, Mapping[str, Any]] = {}
    if isinstance(gradient, Mapping) and isinstance(gradient.get("worker_pins"), list):
        for pin in gradient["worker_pins"]:
            if isinstance(pin, Mapping) and isinstance(pin.get("worker_id"), str):
                pins[pin["worker_id"]] = pin
    workers = runtime.get("workers")
    if not isinstance(workers, list):
        return []
    result = []
    for row in workers:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            continue
        gpu = row.get("gpu")
        if type(gpu) is not int or gpu < 0:
            continue
        pin = pins.get(row["id"], {})
        uuid = pin.get("gpu_uuid") if isinstance(pin, Mapping) else None
        if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
            uuid = row.get("gpu_uuid") if isinstance(row.get("gpu_uuid"), str) else None
        assignment = {"worker_id": row["id"], "gpu_index": gpu}
        if uuid is not None:
            assignment["gpu_uuid"] = uuid
        result.append(assignment)
    return result


def _checkpoint_view(state: str, source: str, error: str | None, *,
                     sidecars: list[str] | None = None, live_state: str = "unknown") -> dict[str, Any]:
    return {"state": state, "disk_state": state, "live_state": live_state,
            "metadata_scope": "pointer_sidecars_and_snapshot_stat_only", "restore_validation": "not_performed",
            "source": source, "pending_sidecars": sorted(sidecars or []), "error_code": error}


def _checkpoint_summary(runtime: Mapping[str, Any] | None, *, source: str,
                        process_state: str, owner_live: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    unknown_optimizer = {"state": "unknown", "durable_policy_version": None,
                         "durable_optimizer_steps": None, "advancement_since_baseline": None,
                         "evidence": "no_valid_latest_pointer_metadata"}
    if not isinstance(runtime, Mapping):
        return _checkpoint_view("unknown", source, "effective_runtime_config_unavailable"), unknown_optimizer
    learner = runtime.get("learner")
    recovery_dir = learner.get("recovery_dir") if isinstance(learner, Mapping) else None
    if not isinstance(recovery_dir, str) or not recovery_dir or "\x00" in recovery_dir:
        return _checkpoint_view("unknown", source, "recovery_directory_unavailable"), unknown_optimizer
    root = Path(recovery_dir)
    try:
        root_info = root.lstat()
    except FileNotFoundError:
        return _checkpoint_view("missing", source, None), unknown_optimizer
    except OSError:
        return _checkpoint_view("unknown", source, "recovery_directory_unavailable"), unknown_optimizer
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        return _checkpoint_view("invalid", source, "recovery_directory_invalid"), unknown_optimizer

    sidecars: list[str] = []
    invalid_sidecar = False
    oversized_sidecar = False
    for name, expected_schema in _CHECKPOINT_SIDECARS.items():
        sidecar, sidecar_state = _read_json(root / name)
        if sidecar_state == "missing":
            continue
        if sidecar_state == "too_large":
            sidecars.append(name)
            oversized_sidecar = True
            continue
        if sidecar_state != "ok" or sidecar is None:
            invalid_sidecar = True
            continue
        if sidecar.get("schema") != expected_schema:
            invalid_sidecar = True
            continue
        sidecars.append(name)

    pointer, pointer_state = _read_json(root / "latest.json")
    pointer_error: str | None = None
    counters: dict[str, Any] | None = None
    if pointer_state == "ok" and pointer is not None:
        generation = pointer.get("generation")
        snapshot_id = pointer.get("snapshot_id")
        snapshot_name = pointer.get("snapshot")
        size_bytes = pointer.get("size_bytes")
        policy_version = pointer.get("policy_version")
        optimizer_steps = pointer.get("optimizer_steps")
        match = _SNAPSHOT.fullmatch(snapshot_name) if isinstance(snapshot_name, str) else None
        if (pointer.get("schema") != "central-gradient-checkpoint-pointer/v1"
                or type(generation) is not int or generation < 0
                or not isinstance(snapshot_id, str) or _HEX_32.fullmatch(snapshot_id) is None
                or match is None or int(match.group(1)) != generation or match.group(2) != snapshot_id
                or type(size_bytes) is not int or size_bytes <= 0
                or type(policy_version) is not int or policy_version < 0
                or type(optimizer_steps) is not int or optimizer_steps < 0):
            pointer_error = "checkpoint_pointer_invalid"
        else:
            snapshot_size, snapshot_state = _regular_file_size(root / snapshot_name)
            if snapshot_state != "ok" or snapshot_size != size_bytes:
                pointer_error = "checkpoint_snapshot_missing_or_invalid"
            else:
                counters = {"generation": generation, "snapshot_id": snapshot_id,
                            "snapshot": snapshot_name, "size_bytes": size_bytes,
                            "policy_version": policy_version, "optimizer_steps": optimizer_steps}
    elif pointer_state not in {"missing"}:
        pointer_error = "checkpoint_pointer_unreadable"

    if invalid_sidecar or pointer_error:
        disk_state = "invalid"
        error_code = pointer_error or "checkpoint_sidecar_invalid"
    elif sidecars:
        disk_state = "pending"
        error_code = "checkpoint_sidecar_size_exceeds_status_limit" if oversized_sidecar else "checkpoint_transaction_pending"
    elif counters is not None:
        disk_state = "metadata_consistent"
        error_code = None
    elif pointer_state == "missing":
        disk_state = "missing"
        error_code = None
    else:
        disk_state = "unknown"
        error_code = "checkpoint_metadata_unknown"

    live_state = "stopped" if not owner_live and process_state == "stopped" else "unknown"
    checkpoint = _checkpoint_view(disk_state, source, error_code, sidecars=sidecars, live_state=live_state)
    if counters is not None:
        checkpoint["identity"] = {key: counters[key] for key in ("generation", "snapshot_id", "snapshot", "size_bytes")}
        checkpoint["counters"] = {key: counters[key] for key in ("policy_version", "optimizer_steps")}
        optimizer = {
            "state": "metadata_observed", "durable_policy_version": counters["policy_version"],
            "durable_optimizer_steps": counters["optimizer_steps"],
            "advancement_since_baseline": None,
            "evidence": "latest_pointer_counter_not_a_delta_or_payload_validation",
            "claim_scope": "pointer_metadata_only",
            "restore_validation": "not_performed",
        }
    else:
        optimizer = unknown_optimizer
    if owner_live and disk_state == "metadata_consistent":
        checkpoint["live_state_reason"] = "active_optimizer_state_is_not_observable_from_local_metadata"
    return checkpoint, optimizer


def _resume_summary(gradient: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    resume = gradient.get("resume") if isinstance(gradient, Mapping) else None
    if not isinstance(resume, Mapping):
        return ({"state": "unknown", "accepted": None, "run_id": None},
                {"state": "unknown", "run_id": None, "completed_updates": None,
                 "requested_steps": None, "remaining_steps": None, "source": "unavailable"})
    if resume.get("status") == "no_pending_run" and resume.get("run_id") is None:
        return ({"state": "no_pending_run_at_startup", "accepted": False, "run_id": None},
                {"state": "none_at_resume_acceptance", "run_id": None,
                 "completed_updates": None, "requested_steps": None,
                 "remaining_steps": None, "source": "resume_receipt"})
    remaining = resume.get("remaining_steps")
    run_id = resume.get("run_id")
    if (resume.get("resumed") is True and resume.get("status") == "running"
            and isinstance(run_id, str) and run_id and type(remaining) is int and remaining >= 0):
        resume_view = {"state": "accepted", "accepted": True, "run_id": run_id,
                       "remaining_steps_at_acceptance": remaining}
        run_view = {"state": "pending_at_resume_acceptance" if remaining else "complete_at_resume_acceptance",
                    "run_id": run_id, "completed_updates": None, "requested_steps": None,
                    "remaining_steps": remaining, "source": "resume_receipt"}
        return resume_view, run_view
    return ({"state": "unknown", "accepted": None, "run_id": None},
            {"state": "unknown", "run_id": None, "completed_updates": None,
             "requested_steps": None, "remaining_steps": None, "source": "unavailable"})


def _ack_summary(runtime: Mapping[str, Any] | None, gradient: Mapping[str, Any] | None) -> dict[str, Any]:
    expected = set()
    workers = runtime.get("workers") if isinstance(runtime, Mapping) else None
    if isinstance(workers, list):
        expected = {row["id"] for row in workers if isinstance(row, Mapping) and isinstance(row.get("id"), str)}
    acks = gradient.get("policy_acks") if isinstance(gradient, Mapping) else None
    acknowledged = set()
    ack_rows = list(acks.items()) if isinstance(acks, Mapping) else []
    for worker_id, row in ack_rows:
        if (isinstance(worker_id, str) and isinstance(row, Mapping)
                and row.get("worker_id") == worker_id and type(row.get("version")) is int
                and isinstance(row.get("adapter_hash"), str) and _HEX_64.fullmatch(row["adapter_hash"])):
            acknowledged.add(worker_id)
    server_ready = isinstance(gradient, Mapping) and gradient.get("server_ready") is True
    versions = {row["version"] for worker_id, row in ack_rows
                if worker_id in expected and worker_id in acknowledged}
    hashes = {row["adapter_hash"] for worker_id, row in ack_rows
              if worker_id in expected and worker_id in acknowledged}
    consistent = acknowledged == expected and len(versions) == 1 and len(hashes) == 1
    startup = ("verified" if expected and consistent and server_ready else
               "partial" if acknowledged or server_ready else "unknown")
    return {"expected_count": len(expected), "startup_readiness": startup,
            "startup_policy_ack_count": len(acknowledged),
            "startup_policy_version": next(iter(versions)) if startup == "verified" else None,
            "startup_policy_identity_consistent": consistent if acknowledged else None,
            "current_readiness": "unknown",
            "current_readiness_reason": "status_does_not_poll_workers_or_call_authenticated_health"}


def _safe_error_code(value: Any) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value):
        return value
    return None


def _verified_build_receipt(value: Any) -> bool:
    return (isinstance(value, Mapping) and value.get("status") == "verified"
            and value.get("scope") == "gradient-overlay"
            and isinstance(value.get("patch_revision"), str)
            and _HEX_40.fullmatch(value["patch_revision"]) is not None
            and value.get("verified_source_files") == sorted(_RUNTIME_BUILD_FILES)
            and value.get("evidence") == "runtime-build.json and matching bounded source SHA-256 values"
            and value.get("base_image_reference_scope") == "build_arg_label_only")


def build_operator_summary(
    deployment: Any,
    process: Mapping[str, Any],
    *,
    owner_live: bool = False,
    observer_build: Mapping[str, Any] | None = None,
    manifest_build: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project only bounded metadata and safe identities for operator use."""
    state_dir = Path(deployment.state_dir)
    source_hash = _sha256_file(Path(deployment.runtime_config))
    gradient = process.get("gradient") if isinstance(process.get("gradient"), Mapping) else None
    run_id = process.get("run_id")
    runtime_path = process.get("runtime_config")
    if runtime_path is None:
        runtime, runtime_state = None, "missing"
    else:
        runtime, runtime_state = _runtime_config_path(state_dir, run_id, runtime_path)
    runtime_source = "last_active_runtime" if runtime_path is not None else "none"
    effective = {"state": ("readable_metadata" if runtime_state == "ok"
                            else "invalid" if runtime_state == "invalid" else "unknown"),
                 "source": runtime_source if runtime_state == "ok" else "unavailable",
                 "validation_scope": "bounded_yaml_parse_only",
                 "run_id": run_id if runtime_source == "last_active_runtime" and runtime_state == "ok" else None,
                 "runtime_config_path": runtime_path if runtime_source == "last_active_runtime" and runtime_state == "ok" else None,
                 "workers": _worker_assignments(runtime or {}, gradient) if runtime_state == "ok" else []}

    assignments = effective["workers"] if runtime_state == "ok" else []
    worker_status = _ack_summary(runtime, gradient) if runtime_state == "ok" else {
        "expected_count": 0, "startup_readiness": "unknown", "startup_policy_ack_count": 0,
        "startup_policy_version": None, "current_readiness": "unknown",
        "current_readiness_reason": "effective_runtime_config_unavailable",
    }
    resume, logical_run = _resume_summary(gradient)
    checkpoint, durable_optimizer = _checkpoint_summary(
        runtime if runtime_state == "ok" else None,
        source=runtime_source if runtime_state == "ok" else "unknown",
        process_state=str(process.get("status", "unknown")), owner_live=owner_live,
    )

    if observer_build is None:
        observer_build = read_runtime_build_identity(Path(__file__).resolve().parents[1])
    observer = dict(observer_build)
    live_build: dict[str, Any]
    if owner_live and _verified_build_receipt(manifest_build):
        live_build = {"status": "verified", "patch_revision": manifest_build.get("patch_revision"),
                      "scope": manifest_build.get("scope"), "evidence": "owner_identity_matched_startup_manifest"}
    elif owner_live:
        live_build = {"status": "unknown", "patch_revision": None,
                      "scope": None, "evidence": "matching_owner_has_no_verified_build_stamp"}
    else:
        live_build = {"status": "not_running", "patch_revision": None,
                      "scope": None, "evidence": "owner_identity_not_live"}

    state = str(process.get("status", "unknown"))
    reason = None
    error_code = None
    next_action = "none"
    blocked = False
    if state in {"failed", "orphaned", "stop_incomplete"}:
        blocked = True
        reason = "owned_instance_process_state_requires_inspection"
        error_code = _safe_error_code(process.get("error")) or f"instance_{state}"
        next_action = "inspect_owned_role_logs_and_process_identity"
    elif checkpoint.get("state") == "invalid":
        blocked = True
        reason = "checkpoint_metadata_or_transaction_sidecar_is_invalid"
        error_code = checkpoint.get("error_code") or "checkpoint_metadata_invalid"
        next_action = "inspect_checkpoint_metadata_before_resume"
    elif checkpoint.get("state") == "pending" and not owner_live:
        blocked = True
        reason = "checkpoint_transaction_sidecar_remains_without_a_live_owner"
        error_code = "checkpoint_transaction_pending"
        next_action = "inspect_checkpoint_transaction_before_resume"
    elif logical_run.get("state") == "pending_at_resume_acceptance" and state == "stopped":
        reason = "resume_receipt_records_remaining_logical_work"
        next_action = "review_resume_receipt_then_use_existing_instance_start_path"
    elif owner_live:
        next_action = "use_existing_watch_for_current_worker_readiness_and_logical_progress"
    elif state == "stopped" and checkpoint.get("state") == "metadata_consistent":
        next_action = "use_existing_instance_start_path_with_managed_restore_guards"

    return {
        "schema": _SCHEMA,
        "process": {"state": state, "owner_identity_matches": owner_live,
                    "role_identity_matches": sum(row.get("identity_matches") is True
                                                   for row in process.get("roles", [])
                                                   if isinstance(row, Mapping)),
                    "role_count": sum(isinstance(row, Mapping) for row in process.get("roles", [])),
                    "gradient_state": gradient.get("status") if isinstance(gradient, Mapping) else None},
        "deployment": {
            "desired": {"runtime_config_path": str(deployment.runtime_config),
                        "runtime_config_sha256": source_hash,
                        "gpu_indices": list(deployment.gpus)},
            "effective": effective,
        },
        "runtime_identity": {
            "observer_build": observer,
            "running_supervisor_build": live_build,
            "running_image_id": None,
            "running_image_id_status": "unknown",
            "running_image_id_evidence": "instance control metadata does not expose a launcher image ID",
            "source_revision_scope": "gradient-overlay patch revision only",
        },
        "workers": {**worker_status,
                    "gpu_assignments": assignments,
                    "startup_evidence_at": gradient.get("started_at") if isinstance(gradient, Mapping) else None},
        "resume": resume,
        "logical_run": logical_run,
        "checkpoint": checkpoint,
        "durable_optimizer_advancement": durable_optimizer,
        "action": {"blocked": blocked, "blocked_reason": reason if blocked else None,
                   "error_code": error_code, "next_action": next_action},
    }
