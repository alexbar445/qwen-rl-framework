"""Read-only evidence projections for the local dashboard.

This module consumes only persisted run records and explicitly selected source roots.
It does not import the Learner, Trainer, Worker, model, or provider runtime.  SQLite
connections use URI read-only mode and ``query_only``; recovery/checkpoint inspection
reads small metadata files and stats large payloads without opening them.
"""
from __future__ import annotations

from .event_records import ledger_event, policy_fields, severity_fields, event_timestamp

import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from ..contracts.checkpoints import retained_versions

_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_EVENTS = 60
_MAX_TASKS = 20
_MAX_ROWS = 200
_MAX_SCOPES = 50
_MAX_REQUESTS = 40
_ID32 = re.compile(r"^[0-9a-f]{32}$")
_REQUIRED_RECOVERY_FILES = frozenset({
    "adapter_config.json", "adapter_state.safetensors", "optimizer.pt",
    "scheduler.pt", "rng_state.pth", "trainer_state.json", "state.json",
})
_OPTIONAL_RECOVERY_FILES = frozenset({"scaler.pt"})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _at(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if result >= 0 else None


def _small_json(path: Path) -> tuple[Any | None, str | None]:
    """Read one small metadata JSON file; reject symlinks and oversized inputs."""
    try:
        if path.is_symlink() or not path.is_file():
            return None, "missing_or_unsafe"
        if path.stat().st_size > _MAX_JSON_BYTES:
            return None, "metadata_file_too_large"
        with path.open("rb") as handle:
            payload = handle.read(_MAX_JSON_BYTES + 1)
        if len(payload) > _MAX_JSON_BYTES:
            return None, "metadata_file_too_large"
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "unreadable_or_invalid_json"
    return value, None


def _path(value: Any) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    try:
        return Path(value).expanduser()
    except (OSError, TypeError, ValueError):
        return None


def _source_info(collector: Any, run_id: str | None) -> dict[str, Any]:
    reader = getattr(collector, "source_info", None)
    if not callable(reader):
        return {"available": False, "reason": "collector has no explicit source_info()"}
    try:
        info = reader(run_id=run_id)
    except Exception as exc:
        return {"available": False, "reason": f"source selection failed: {type(exc).__name__}"}
    if not isinstance(info, Mapping):
        return {"available": False, "reason": "collector source_info() returned no object"}
    return dict(info)


def _read_only_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=1000")
    return connection


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' LIMIT 100"
    )}


def _query(connection: sqlite3.Connection, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(sql, args)]


def _first(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def _timestamp_age(value: Any, as_of: float) -> float | None:
    stamp = _number(value)
    return max(0.0, as_of - stamp) if stamp is not None else None


def _selected_run_id(run_detail: Any, live_selected: Any) -> str | None:
    for item in (live_selected, run_detail):
        value = _at(item, "run_id")
        if isinstance(value, str) and value:
            return value
    return None


def _run_recovery(run_detail: Any) -> dict[str, Any]:
    return dict(_mapping(_at(run_detail, "manifest", "recovery")))


def _selected_lineage(
    run_detail: Any, recovery: Mapping[str, Any], source: Mapping[str, Any] | None = None,
    live_snapshot: Any = None,
) -> tuple[str | None, str, str | None]:
    detail = _mapping(run_detail)
    manifest = _mapping(detail.get("manifest"))
    selected_source = _mapping(source)
    if (selected_source.get("kind") == "runtime"
            and selected_source.get("instance_manifest")
            and selected_source.get("is_current") is True):
        snapshot = _mapping(live_snapshot)
        lifecycle = _mapping(snapshot.get("lifecycle"))
        identity = _mapping(snapshot.get("source_identity"))
        learner = _mapping(snapshot.get("learner"))
        live_recovery = _mapping(learner.get("recovery"))
        selected_run_id = detail.get("run_id") if run_detail is not None else selected_source.get("run_id")
        lineage_id = live_recovery.get("lineage_id")
        engine_epoch = live_recovery.get("engine_epoch")
        digest = re.compile(r"^[0-9a-f]{64}$")
        bound = (snapshot.get("owned_source_verified") is True
                 and lifecycle.get("backend") == "central_gradient"
                 and lifecycle.get("run_id") == selected_run_id
                 and identity.get("logical_run_id") == selected_run_id
                 and isinstance(identity.get("instance_run_id"), str)
                 and identity.get("instance_run_id")
                 and isinstance(identity.get("telemetry_epoch"), str)
                 and identity.get("telemetry_epoch")
                 and isinstance(identity.get("runtime_config_sha256"), str)
                 and digest.fullmatch(identity["runtime_config_sha256"])
                 and isinstance(identity.get("gradient_manifest_sha256"), str)
                 and digest.fullmatch(identity["gradient_manifest_sha256"]))
        if (bound and isinstance(lineage_id, str) and _ID32.fullmatch(lineage_id)
                and isinstance(engine_epoch, str) and engine_epoch):
            return lineage_id, "verified_live_instance.recovery.lineage_id", None
        return None, "verified_live_instance", "current instance has no verified live recovery lineage"
    if run_detail is not None and manifest:
        run_recovery = _mapping(manifest.get("recovery"))
        if "lineage_id" in run_recovery:
            value = run_recovery.get("lineage_id")
            if isinstance(value, str) and _ID32.fullmatch(value):
                return value, "selected_run_manifest.recovery.lineage_id", None
            return None, "selected_run_manifest.recovery", "selected run manifest has an invalid recovery lineage"
        runtime = _mapping(_mapping(source).get("runtime"))
        runtime_learner = _mapping(runtime.get("learner"))
        snapshot = _mapping(live_snapshot)
        lifecycle = _mapping(snapshot.get("lifecycle"))
        source_identity = _mapping(snapshot.get("source_identity"))
        learner = _mapping(snapshot.get("learner"))
        live_recovery = _mapping(learner.get("recovery"))
        selected_run_id = detail.get("run_id")
        lineage_id = live_recovery.get("lineage_id")
        engine_epoch = live_recovery.get("engine_epoch")
        bound_identity = all(isinstance(source_identity.get(key), str) and source_identity[key]
                             for key in ("container_id", "profile_sha256", "telemetry_epoch"))
        if (runtime_learner.get("update_backend") == "central_gradient"
                and snapshot.get("owned_source_verified") is True
                and lifecycle.get("backend") == "central_gradient"
                and isinstance(selected_run_id, str)
                and lifecycle.get("run_id") == selected_run_id
                and source_identity.get("control_run_id") == selected_run_id
                and bound_identity
                and isinstance(lineage_id, str) and _ID32.fullmatch(lineage_id)
                and isinstance(engine_epoch, str) and engine_epoch):
            return lineage_id, "verified_live_central_run.recovery.lineage_id", None
        return None, "selected_run_manifest", "selected run manifest has no recovery lineage proof"
    value = _at(recovery, "lineage", "lineage_id")
    if isinstance(value, str) and _ID32.fullmatch(value):
        return value, "recovery.latest_pointer", None
    return None, "unknown", "evaluation lineage cannot be selected from persisted evidence"


def _live_context(
    snapshot: Any, live_selected: Any, recovery: Mapping[str, Any], run_detail: Any,
) -> dict[str, Any]:
    """Use the selected live watch, then selected-run records, then durable metadata."""
    snap = _mapping(snapshot)
    selected = _mapping(live_selected)
    # WatchClient places the Learner /monitor response directly at snapshot.learner.
    learner = _mapping(snap.get("learner"))
    recovery_view = _mapping(learner.get("recovery"))
    run_recovery = _run_recovery(run_detail)
    current_version = _int(selected.get("policy_version"))
    version_basis = "selected_metadata" if current_version is not None else None
    if current_version is None:
        current_version = _int(learner.get("policy_version"))
        version_basis = "live_watch.learner.policy_version" if current_version is not None else None
    if current_version is None:
        records = _at(run_detail, "records")
        if isinstance(records, list):
            for record in reversed(records[-20:]):
                current_version = _int(_at(record, "policy_version"))
                if current_version is not None:
                    version_basis = "selected_run_record.committed_policy_version"
                    break
    if current_version is None:
        current_version = _int(run_recovery.get("policy_version"))
        version_basis = "selected_run_manifest.recovery" if current_version is not None else None
    if current_version is None:
        current_version = _int(recovery.get("policy_version"))
        version_basis = "durable_recovery_pointer" if current_version is not None else "unknown"
    engine_epoch = selected.get("engine_epoch")
    epoch_basis = "selected_metadata" if isinstance(engine_epoch, str) and engine_epoch else None
    if not isinstance(engine_epoch, str) or not engine_epoch:
        engine_epoch = recovery_view.get("engine_epoch")
        epoch_basis = "live_watch.learner.recovery" if isinstance(engine_epoch, str) and engine_epoch else None
    if not isinstance(engine_epoch, str) or not engine_epoch:
        engine_epoch = run_recovery.get("engine_epoch")
        epoch_basis = "selected_run_manifest.recovery" if isinstance(engine_epoch, str) and engine_epoch else None
    if not isinstance(engine_epoch, str) or not engine_epoch:
        engine_epoch = recovery.get("engine_epoch")
        epoch_basis = "durable_recovery_pointer" if isinstance(engine_epoch, str) and engine_epoch else "unknown"
    batches = selected.get("run_batch_ids")
    if not isinstance(batches, list):
        batches = _at(learner, "training", "run_batch_ids")
    if not isinstance(batches, list):
        batches = []
    batches = [value for value in batches if isinstance(value, str) and value][:512]
    return {"policy_version": current_version, "engine_epoch": engine_epoch,
            "run_batch_ids": batches, "basis": version_basis,
            "policy_version_basis": version_basis, "engine_epoch_basis": epoch_basis}


def _checkpoint_tier(version: int, current: int) -> int | None:
    if version > current or version <= 0:
        return None
    if version % 50 == 0:
        return 50
    if version % 10 == 0 and current - version <= 50:
        return 10
    if version % 2 == 0 and current - version <= 10:
        return 2
    return None


def _central_recovery_evidence(state_dir: Path, pointer: Mapping[str, Any]) -> dict[str, Any]:
    version = _int(pointer.get("policy_version"))
    steps = _int(pointer.get("optimizer_steps"))
    snapshot_id = pointer.get("snapshot_id")
    size = _int(pointer.get("size_bytes"))
    if (version is None or steps is None or not isinstance(snapshot_id, str)
            or not _ID32.fullmatch(snapshot_id) or not size
            or pointer.get("snapshot") != f"state-{version:08d}-{snapshot_id}.pt"):
        return {"available": False, "reason": "central checkpoint pointer identity is invalid", "source": str(state_dir)}
    name = pointer["snapshot"]
    path = state_dir / name
    try:
        present = path.is_file() and not path.is_symlink()
        actual_size = path.stat().st_size if present else None
    except OSError:
        present, actual_size = False, None
    complete = present and actual_size == size
    files = {"required": [name], "present": [name] if present else [],
             "missing": [] if present else [name], "complete": complete,
             "sizes_bytes": {name: actual_size}, "size_verification": "passed" if complete else "failed",
             "payload_hashes_recomputed": False}
    latest = {"policy_version": version, "completed_updates": steps, "snapshot_id": snapshot_id,
              "lineage_id": None, "engine_epoch": None, "content_verification": "disabled",
              "backend": "central_gradient", "files": files}
    return {"available": True, "reason": None if complete else "central snapshot missing or size differs",
            "source": str(state_dir), "latest": latest, "inventory": {},
            "checkpoints": [{"version": version, "tier": "latest", "exists": present,
                             "complete": complete, "bytes": actual_size, "files": {name: actual_size},
                             "source": str(state_dir)}],
            "selected_lineage_matches_latest": None,
            "limits": ["Central snapshot payload was only stat'ed; optimizer tensors and lineage were not deserialized."]}


def _recovery_evidence(source: Mapping[str, Any], run_detail: Any = None) -> dict[str, Any]:
    runtime = _mapping(source.get("runtime"))
    paths = _mapping(source.get("paths"))
    state_dir = _path(paths.get("recovery_dir"))
    if state_dir is None:
        return {"available": False, "reason": "selected runtime has no recovery_dir", "source": source.get("runtime_config")}
    if state_dir.is_symlink() or not state_dir.is_dir():
        return {"available": False, "reason": "selected recovery_dir is missing or unsafe", "source": str(state_dir)}
    pointer, pointer_error = _small_json(state_dir / "latest.json")
    if not isinstance(pointer, dict):
        return {"available": False, "reason": f"latest recovery pointer {pointer_error}", "source": str(state_dir)}
    if pointer.get("schema") == "central-gradient-checkpoint-pointer/v1":
        return _central_recovery_evidence(state_dir, pointer)
    lineage_id = pointer.get("lineage_id")
    snapshot_id = pointer.get("snapshot_id")
    version = _int(pointer.get("policy_version"))
    if (pointer.get("schema") != "qwen-rl-latest/v1" or not isinstance(lineage_id, str)
            or not _ID32.fullmatch(lineage_id) or not isinstance(snapshot_id, str)
            or not _ID32.fullmatch(snapshot_id) or version is None):
        return {"available": False, "reason": "latest pointer has unsupported or incomplete identity", "source": str(state_dir)}
    snapshot_dir = state_dir / "lineages" / lineage_id / "snapshots" / snapshot_id
    if any(part.is_symlink() for part in (state_dir / "lineages", state_dir / "lineages" / lineage_id,
                                          state_dir / "lineages" / lineage_id / "snapshots", snapshot_dir)):
        return {"available": False, "reason": "latest snapshot path contains a symlink", "source": str(state_dir)}
    manifest, manifest_error = _small_json(snapshot_dir / "manifest.json")
    metadata, state_error = _small_json(snapshot_dir / "state.json")
    if not isinstance(manifest, dict) or not isinstance(metadata, dict):
        return {"available": False, "reason": f"snapshot metadata unavailable: {manifest_error or state_error}",
                "source": str(state_dir), "lineage_id": lineage_id, "snapshot_id": snapshot_id}
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    listed = set(files)
    missing_manifest = sorted(_REQUIRED_RECOVERY_FILES - listed)
    unknown_manifest = sorted(listed - _REQUIRED_RECOVERY_FILES - _OPTIONAL_RECOVERY_FILES)
    present: list[str] = []
    missing_disk: list[str] = []
    sizes: dict[str, int | None] = {}
    for name in sorted(listed & (_REQUIRED_RECOVERY_FILES | _OPTIONAL_RECOVERY_FILES)):
        file_path = snapshot_dir / name
        try:
            stat = file_path.stat()
            if file_path.is_symlink() or not file_path.is_file():
                missing_disk.append(name)
                sizes[name] = None
            else:
                stat = file_path.stat()
                present.append(name)
                sizes[name] = stat.st_size
        except OSError:
            missing_disk.append(name)
            sizes[name] = None
    expected_sizes = manifest.get("file_sizes")
    size_mismatches = []
    if "file_sizes" in manifest:
        if (not isinstance(expected_sizes, dict) or set(expected_sizes) != listed
                or any(type(value) is not int or value < 0 for value in expected_sizes.values())):
            size_mismatches = ["invalid file_sizes"]
        else:
            size_mismatches = sorted(name for name, expected in expected_sizes.items()
                                     if sizes.get(name) != expected)
    identity_match = (manifest.get("schema") == "qwen-rl-training-state/v1"
                      and manifest.get("lineage_id") == lineage_id
                      and manifest.get("snapshot_id") == snapshot_id
                      and manifest.get("policy_version") == version
                      and metadata.get("lineage_id") == lineage_id
                      and metadata.get("policy_version") == version
                      and metadata.get("engine_epoch") == manifest.get("engine_epoch"))
    complete = not missing_manifest and not unknown_manifest and not missing_disk and not size_mismatches and identity_match
    if complete and expected_sizes is None:
        complete = None
    latest = {
        "lineage_id": lineage_id, "snapshot_id": snapshot_id,
        "content_verification": "disabled",
        "engine_epoch": manifest.get("engine_epoch"),
        "policy_version": version, "completed_updates": _int(manifest.get("completed_updates")),
        "created_at": manifest.get("created_at"), "parent_lineage": metadata.get("parent_lineage"),
        "topology": metadata.get("topology") or manifest.get("topology"),
        "update_context": metadata.get("update_context"),
        "files": {"required": sorted(_REQUIRED_RECOVERY_FILES), "present": present,
                  "missing": sorted(set(missing_manifest) | set(missing_disk)),
                  "unexpected": unknown_manifest, "complete": complete,
                  "sizes_bytes": sizes, "size_mismatches": size_mismatches,
                  "size_verification": "unavailable" if expected_sizes is None else ("failed" if size_mismatches else "passed"),
                  "payload_hashes_recomputed": False},
    }
    # Checkpoint inventory is adapter-file stat metadata only. Large binary payloads
    # are never opened or hashed by the dashboard.
    checkpoint_root = state_dir / "lineages" / lineage_id / "checkpoints"
    checkpoint_root_safe = not checkpoint_root.is_symlink()
    retained = sorted(retained_versions(version))
    checkpoints: list[dict[str, Any]] = []
    present_versions: set[int] = set()
    for checkpoint_version in retained:
        name = f"version-{checkpoint_version:06d}"
        directory = checkpoint_root / name
        exists = checkpoint_root_safe and directory.is_dir() and not directory.is_symlink()
        entries = {}
        if exists:
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                file_path = directory / filename
                try:
                    if file_path.is_file() and not file_path.is_symlink():
                        entries[filename] = file_path.stat().st_size
                except OSError:
                    pass
        complete_checkpoint = exists and set(entries) == {"adapter_config.json", "adapter_model.safetensors"}
        if exists:
            present_versions.add(checkpoint_version)
        checkpoints.append({"version": checkpoint_version, "tier": _checkpoint_tier(checkpoint_version, version),
                            "exists": exists, "complete": complete_checkpoint,
                            "bytes": sum(entries.values()) if entries else (0 if exists else None),
                            "files": entries, "source": str(directory)})
    tier_inventory = {}
    for tier in (2, 10, 50):
        expected = [v for v in retained if _checkpoint_tier(v, version) == tier]
        observed = [item for item in checkpoints if item["tier"] == tier and item["complete"]]
        tier_inventory[str(tier)] = {
            "expected_count": len(expected), "present_complete_count": len(observed),
            "bytes": sum(int(item["bytes"] or 0) for item in observed),
            "missing_versions": [v for v in expected if v not in present_versions],
            "basis": "CheckpointPolicy retained_versions rule; adapter files stat only",
        }
    selected_run_recovery = _run_recovery(run_detail)
    selected_lineage = selected_run_recovery.get("lineage_id")
    selected_lineage = selected_lineage if isinstance(selected_lineage, str) and _ID32.fullmatch(selected_lineage) else None
    return {"available": True, "reason": None if complete else "latest metadata or required files are incomplete",
            "source": str(state_dir), "latest": latest, "checkpoints": checkpoints,
            "inventory": tier_inventory, "lineage": {"lineage_id": lineage_id,
            "parent_lineage": metadata.get("parent_lineage"), "policy_version": version,
            "engine_epoch": manifest.get("engine_epoch"),
            "topology": metadata.get("topology") or manifest.get("topology")},
            "selected_run": ({"lineage_id": selected_lineage,
                "engine_epoch": selected_run_recovery.get("engine_epoch"),
                "policy_version": selected_run_recovery.get("policy_version"),
                "loaded_snapshot": selected_run_recovery.get("loaded_snapshot"),
                "source": "selected run manifest recovery metadata"} if selected_lineage else None),
            "selected_lineage_matches_latest": (selected_lineage == lineage_id if selected_lineage else None),
            "limits": ["Large payloads were only stat'ed; their content hashes were not recomputed."]}


def _evaluation_unavailable(reason: str, source: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"available": False, "reason": reason, "source": source, "lineage_id": None,
            "selected_lineage": None, "evalset_sha256": None, "baseline": None,
            "results": [], "latest": None, **extra}


def _evaluation_evidence(
    source: Mapping[str, Any], recovery: Mapping[str, Any], run_detail: Any = None,
    live_snapshot: Any = None,
) -> dict[str, Any]:
    runtime = _mapping(source.get("runtime"))
    evaluation_config = _mapping(runtime.get("evaluation"))
    paths = _mapping(source.get("paths"))
    selected = _path(paths.get("evaluation_state_dir"))
    if evaluation_config.get("enabled") is False:
        return _evaluation_unavailable("native evaluation is disabled in selected runtime", str(selected) if selected else None)
    if selected is None:
        return _evaluation_unavailable("selected runtime has no evaluation state directory")
    if selected.is_symlink() or not selected.is_dir():
        return _evaluation_unavailable("selected evaluation state directory is missing or unsafe", str(selected))
    expected_lineage, lineage_basis, lineage_error = _selected_lineage(
        run_detail, recovery, source, live_snapshot)
    if expected_lineage is None:
        return _evaluation_unavailable(lineage_error or "evaluation lineage is unavailable", str(selected),
                                       selected_lineage_basis=lineage_basis)
    registry_path = selected / expected_lineage / "registry.json"
    if registry_path.parent.is_symlink() or not registry_path.parent.is_dir():
        return _evaluation_unavailable("selected evaluation lineage directory is missing or unsafe", str(registry_path))
    registry, error = _small_json(registry_path)
    if not isinstance(registry, dict):
        return _evaluation_unavailable(f"selected-lineage registry {error}", str(registry_path))
    if registry.get("schema") != "qwen-rl-evaluation-state-v1":
        return _evaluation_unavailable("evaluation registry has an unsupported schema", str(registry_path))
    registry_lineage = registry.get("lineage_id")
    if registry_lineage != expected_lineage:
        return _evaluation_unavailable("evaluation registry lineage differs from selected recovery lineage",
                str(registry_path), lineage_id=registry_lineage, selected_lineage=expected_lineage)
    evalset_sha = registry.get("evalset_sha256")
    requests = _mapping(registry.get("requests"))
    request_order = registry.get("request_order")
    if not isinstance(request_order, list):
        return _evaluation_unavailable("evaluation registry has no request order", str(registry_path))
    request_order = [item for item in request_order if isinstance(item, str) and item in requests][-_MAX_REQUESTS:]
    by_id: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for request_id in request_order:
        request = _mapping(requests.get(request_id))
        if request.get("lineage_id") != expected_lineage or request.get("evalset_sha256") != evalset_sha:
            continue
        if request.get("request_id") != request_id:
            continue
        summary = _mapping(request.get("summary"))
        task_map = _mapping(request.get("tasks"))
        # Preserve exact row-level state for comparisons, but only if stored as native
        # evaluation outputs with explicit boolean exact values.
        exact_rows: dict[str, bool] = {}
        for task_id, task in list(task_map.items())[:_MAX_ROWS]:
            task = _mapping(task)
            task_summary = _mapping(task.get("summary"))
            exact = task_summary.get("exact")
            if isinstance(task_id, str) and type(exact) is bool and not task_summary.get("error"):
                exact_rows[task_id] = exact
        projected_summary = dict(summary) if summary else None
        request_complete = (request.get("status") == "completed"
                            and summary.get("complete_valid_evaluation") is True)
        if projected_summary is not None:
            projected_summary["full_suite_complete"] = request_complete
            if not request_complete:
                projected_summary["exact_rate"] = None
                projected_summary["gains"] = None
                projected_summary["regressions"] = None
        item = {
            "request_id": request_id, "kind": request.get("kind"), "status": request.get("status"),
            "policy_version": _int(request.get("policy_version")),
            "adapter_sha256": request.get("adapter_sha256"),
            "milestone_update_count": _int(request.get("milestone_update_count")),
            "successful_update_count": _int(request.get("successful_update_count")),
            "created_at": _number(request.get("created_at")),
            "completed_at": _number(request.get("completed_at")),
            "summary": projected_summary,
            "tasks_done": sum(_mapping(task).get("status") == "completed" for task in task_map.values()),
            "task_count": len(request.get("task_ids", [])),
            "result_ref": request.get("result_ref"), "result_sha256": request.get("result_sha256"),
            "comparison_ids": dict(_mapping(request.get("comparison"))),
            "exact_rows": exact_rows, "source": str(registry_path),
        }
        by_id[request_id] = item
        results.append(item)
    baseline_id = registry.get("baseline_request_id")
    baseline = by_id.get(baseline_id) if isinstance(baseline_id, str) else None
    for item in results:
        comparison_ids = item.get("comparison_ids") or {}
        baseline_ref = by_id.get(comparison_ids.get("baseline_request_id")) or baseline
        previous_ref = by_id.get(comparison_ids.get("previous_request_id"))
        def pairing(reference: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
            raw = _mapping(_at(item, "summary", name))
            complete_pair = raw.get("complete_paired_evaluation") is True
            return {
                "request_id": reference.get("request_id") if reference else None,
                "exact_rate": (_at(reference, "summary", "exact_rate")
                               if _at(reference, "summary", "full_suite_complete") is True else None),
                "complete_valid_evaluation": _at(reference, "summary", "full_suite_complete") if reference else None,
                "comparison_available": raw.get("comparison_available"),
                "complete_paired_evaluation": complete_pair if raw else None,
                "expected_task_count": _int(raw.get("expected_task_count")),
                "paired_task_count": _int(raw.get("paired_task_count")),
                "paired_known_count": _int(raw.get("paired_known_count")),
                "paired_gain_count": _int(raw.get("paired_gain_count")),
                "paired_regression_count": _int(raw.get("paired_regression_count")),
                "paired_gains": raw.get("gains") if isinstance(raw.get("gains"), list) else None,
                "paired_regressions": raw.get("regressions") if isinstance(raw.get("regressions"), list) else None,
                "full_suite_gain_count": _int(raw.get("gain_count")) if complete_pair else None,
                "full_suite_regression_count": _int(raw.get("regression_count")) if complete_pair else None,
                "full_suite_gains": raw.get("gains") if complete_pair and isinstance(raw.get("gains"), list) else None,
                "full_suite_regressions": raw.get("regressions") if complete_pair and isinstance(raw.get("regressions"), list) else None,
            }
        item["comparison"] = {
            "baseline_request_id": baseline_ref.get("request_id") if baseline_ref else None,
            "baseline_exact_rate": (_at(baseline_ref, "summary", "exact_rate")
                                     if _at(baseline_ref, "summary", "full_suite_complete") is True else None),
            "baseline_complete": _at(baseline_ref, "summary", "full_suite_complete") if baseline_ref else None,
            "baseline_gains": _at(item, "summary", "vs_baseline", "gains")
                if _at(item, "summary", "vs_baseline", "complete_paired_evaluation") is True else None,
            "baseline_regressions": _at(item, "summary", "vs_baseline", "regressions")
                if _at(item, "summary", "vs_baseline", "complete_paired_evaluation") is True else None,
            "baseline_comparison_complete": _at(item, "summary", "vs_baseline", "complete_paired_evaluation"),
            "baseline_pairing": pairing(baseline_ref, "vs_baseline"),
            "previous_request_id": previous_ref.get("request_id") if previous_ref else None,
            "previous_exact_rate": (_at(previous_ref, "summary", "exact_rate")
                                     if _at(previous_ref, "summary", "full_suite_complete") is True else None),
            "previous_gains": _at(item, "summary", "vs_previous", "gains")
                if _at(item, "summary", "vs_previous", "complete_paired_evaluation") is True else None,
            "previous_regressions": _at(item, "summary", "vs_previous", "regressions")
                if _at(item, "summary", "vs_previous", "complete_paired_evaluation") is True else None,
            "previous_comparison_complete": _at(item, "summary", "vs_previous", "complete_paired_evaluation"),
            "previous_pairing": pairing(previous_ref, "vs_previous"),
        }
        item.pop("exact_rows", None)
    results.sort(key=lambda item: (item.get("created_at") or 0.0, item["request_id"]))
    latest = results[-1] if results else None
    cumulative_count = registry.get("lineage_successful_update_count")
    if type(cumulative_count) is not int or cumulative_count < 0:
        cumulative_count = None
    return {"available": True, "reason": None if results else "selected lineage has no persisted evaluation requests",
            "source_class": "native_evaluation", "metric_scope": "selected frozen evaluation registry",
            "source": str(registry_path), "lineage_id": registry_lineage,
            "evalset_sha256": evalset_sha, "selected_lineage": expected_lineage,
            "selected_lineage_basis": lineage_basis,
            "lineage_successful_update_count": cumulative_count,
            "milestone_interval": registry.get("milestone_interval"),
            "baseline": ({"request_id": baseline.get("request_id"), "status": baseline.get("status"),
                          "summary": baseline.get("summary"), "result_ref": baseline.get("result_ref"),
                          "result_sha256": baseline.get("result_sha256")} if baseline else None),
            "results": results, "latest": latest,
            "limits": ["Only persisted native evaluation registry rows and summaries are used.",
                       "Incomplete paired comparisons keep gains and regressions unknown."]}


def _group_cohorts(run_detail: Any) -> list[dict[str, Any]]:
    detail = _mapping(run_detail)
    records = detail.get("records")
    if not isinstance(records, list):
        return []
    cohorts: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for record in records[-20:]:
        record = _mapping(record)
        groups = record.get("groups")
        for group in groups[:500] if isinstance(groups, list) else []:
            group = _mapping(group)
            version = _int(group.get("policy_version"))
            selected_replay = (group.get("selected_ready_cache") is True
                               or bool(group.get("replay_reservation")))
            row_kinds = group.get("row_kinds")
            replay_kinds = {"native_replay", "guided_replay"}
            fresh_kinds = {"native", "fresh", "fresh_student"}
            if isinstance(row_kinds, list):
                has_replay = any(kind in replay_kinds for kind in row_kinds)
                has_fresh = any(kind in fresh_kinds for kind in row_kinds)
                source_class = ("native_plus_replay" if has_replay and has_fresh
                                else "replay_only" if has_replay
                                else "native_only")
            else:
                source_class = "native_plus_replay" if selected_replay else "native_only"
            key = (version, source_class, group.get("phase"))
            item = cohorts.setdefault(key, {
                "policy_version": version, "source_class": source_class, "phase": group.get("phase"),
                "groups": 0, "native_candidate_rows": 0, "native_candidate_row_groups": 0,
                "prepared_rows": 0, "prepared_row_groups_known": 0,
                "prepared_replay_rows": 0,
                "phase_basis": "group.phase; missing phase is not inferred from policy version",
                "training_rows_observed": 0, "training_row_groups_known": 0,
                "replay_rows_added_observed": 0, "replay_row_groups_known": 0,
                "native_exact_known": 0, "native_exact_success": 0, "native_exact_failure": 0,
                "record_ids": [], "exact_basis": "group.exact for native candidates; replay outcomes are separate",
                "basis": "persisted run-record groups; native evaluation requests excluded",
            })
            item["groups"] += 1
            if record.get("record_id") and record["record_id"] not in item["record_ids"]:
                item["record_ids"].append(record["record_id"])
            rewards = group.get("rewards")
            training_rewards = group.get("training_rewards")
            # Prepared cardinalities precede backward and do not prove an optimizer commit.
            cardinalities = _mapping(group.get("group_cardinalities"))
            raw_size, training_size = _int(cardinalities.get("raw")), _int(cardinalities.get("training"))
            if raw_size is not None and training_size is not None and 0 <= raw_size <= training_size:
                item["prepared_rows"] += training_size
                item["prepared_replay_rows"] += training_size - raw_size
                item["prepared_row_groups_known"] += 1
            if isinstance(rewards, list):
                item["native_candidate_rows"] += len(rewards)
                item["native_candidate_row_groups"] += 1
            if isinstance(training_rewards, list):
                item["training_rows_observed"] += len(training_rewards)
                item["training_row_groups_known"] += 1
                if isinstance(rewards, list):
                    item["replay_rows_added_observed"] += max(0, len(training_rewards) - len(rewards))
                    item["replay_row_groups_known"] += 1
            exact = group.get("exact")
            exact_values = exact if isinstance(exact, list) else [exact]
            for value in exact_values:
                if type(value) is bool:
                    item["native_exact_known"] += 1
                    item["native_exact_success"] += int(value)
                    item["native_exact_failure"] += int(not value)
    for item in cohorts.values():
        if not item["prepared_row_groups_known"]:
            item["prepared_rows"] = None
            item["prepared_replay_rows"] = None
        if not item["training_row_groups_known"]:
            item["training_rows_observed"] = None
        if not item["replay_row_groups_known"]:
            item["replay_rows_added_observed"] = None
    return list(cohorts.values())


def _scope_and_limits(source: Mapping[str, Any]) -> tuple[str | None, int | None, int | None]:
    bepa = _mapping(_at(source, "runtime", "bepa"))
    scope = bepa.get("budget_scope")
    scope = scope if isinstance(scope, str) and scope else None
    return scope, _int(bepa.get("cache", {}).get("max_age_updates")) if isinstance(bepa.get("cache"), Mapping) else None, \
        _int(bepa.get("cache", {}).get("max_committed_uses")) if isinstance(bepa.get("cache"), Mapping) else None


def _bepa_evidence(
    source: Mapping[str, Any], base_bepa: Any, context: Mapping[str, Any], run_detail: Any,
) -> dict[str, Any]:
    paths = _mapping(source.get("paths"))
    raw_path = _path(paths.get("bepa_store"))
    scope_id, max_age, max_uses = _scope_and_limits(source)
    if raw_path is None:
        reason = "selected runtime has no BEPA store path"
        return {"available": False, "reason": reason, "scope": None, "timeline": [],
                "cohorts": _group_cohorts(run_detail), "attempts": [], "source": None}
    db = raw_path / "state.sqlite3" if raw_path.is_dir() else raw_path
    base = _mapping(base_bepa)
    base_truncation = _mapping(base.get("truncation"))
    base_path = _path(base.get("path"))
    try:
        base_matches_store = (base.get("available") is True and base_path is not None
                              and base_path.resolve() == db.parent.resolve())
    except OSError:
        base_matches_store = False
    if db.is_symlink() or not db.is_file():
        return {"available": False, "reason": "selected BEPA database is missing or unsafe",
                "scope": {"id": scope_id, "available": False}, "timeline": [],
                "cohorts": _group_cohorts(run_detail), "attempts": [], "source": str(db)}
    try:
        connection = _read_only_connection(db)
    except sqlite3.Error as exc:
        return {"available": False, "reason": f"read-only BEPA database open failed: {type(exc).__name__}",
                "scope": {"id": scope_id, "available": False}, "timeline": [],
                "cohorts": _group_cohorts(run_detail), "attempts": [], "source": str(db)}
    required = {"budget_scopes", "rescue_jobs", "plans", "seed_attempts", "episodes",
                "cache_entries", "reservations", "update_attempts", "events"}
    try:
        tables = _table_names(connection)
        missing_tables = sorted(required - tables)
        if missing_tables:
            return {"available": False, "reason": "BEPA database schema is incomplete",
                    "missing_tables": missing_tables, "scope": {"id": scope_id, "available": False},
                    "timeline": [], "cohorts": _group_cohorts(run_detail), "attempts": [], "source": str(db)}
        job_states_global = {row["state"]: int(row["count"]) for row in connection.execute(
            "SELECT state,count(*) AS count FROM rescue_jobs GROUP BY state")}
        reservation_states_global = {row["state"]: int(row["count"]) for row in connection.execute(
            "SELECT state,count(*) AS count FROM reservations GROUP BY state")}
        cache_meta_global = {row["row_kind"]: int(row["count"]) for row in connection.execute(
            "SELECT row_kind,count(*) AS count FROM cache_entries GROUP BY row_kind")}
        if base_matches_store and isinstance(base.get("scopes"), list):
            scope_rows = [dict(row) for row in base["scopes"][:_MAX_SCOPES] if isinstance(row, Mapping)]
            scope_rows_truncated = bool(base_truncation.get("scopes")) or len(base.get("scopes", [])) >= _MAX_SCOPES
        else:
            scope_rows = _query(connection,
                "SELECT scope_id,max_total_tokens,max_jobs,used_tokens,reserved_tokens,usage_unknown,created_at "
                "FROM budget_scopes ORDER BY created_at DESC,scope_id LIMIT ?", (_MAX_SCOPES,))
            scope_rows_truncated = len(scope_rows) >= _MAX_SCOPES
        scope_row = None
        scope_jobs: list[dict[str, Any]] = []
        scope_plans: list[dict[str, Any]] = []
        scope_attempts: list[dict[str, Any]] = []
        plan_counts_current: dict[str, int] = {}
        attempt_counts_current: dict[str, int] = {}
        task_keys: list[str] = []
        if scope_id is not None:
            scope_row = _first(_query(connection,
                "SELECT scope_id,max_total_tokens,max_jobs,used_tokens,reserved_tokens,usage_unknown,created_at "
                "FROM budget_scopes WHERE scope_id=? LIMIT 1", (scope_id,)))
            if scope_row is not None:
                if base_matches_store and isinstance(base.get("jobs"), list) and not base_truncation.get("jobs"):
                    scope_jobs = [dict(row) for row in base["jobs"]
                                  if isinstance(row, Mapping) and row.get("budget_scope") == scope_id][:_MAX_ROWS]
                else:
                    scope_jobs = _query(connection,
                        "SELECT job_id,task_key,budget_scope,state,attempts,max_attempts,used_tokens,reserved_tokens,max_tokens,usage_unknown,"
                        "fail_reason,created_at,updated_at,lease_expires_at,source_episode_id,source_receipt_ref,plan_artifact_ref "
                        "FROM rescue_jobs WHERE budget_scope=? ORDER BY updated_at DESC,job_id DESC LIMIT ?",
                        (scope_id, _MAX_ROWS))
                if (base_matches_store and isinstance(base.get("plans"), list)
                        and isinstance(base.get("jobs"), list)
                        and not base_truncation.get("plans") and not base_truncation.get("jobs")):
                    scoped_job_ids = {row.get("job_id") for row in scope_jobs}
                    scope_plans = [dict(row) for row in base["plans"]
                                   if isinstance(row, Mapping) and row.get("job_id") in scoped_job_ids][:_MAX_ROWS]
                else:
                    scope_plans = _query(connection,
                        "SELECT p.plan_id,p.job_id,p.task_key,p.status,p.seed_attempts,p.created_at,p.updated_at,"
                        "p.source_episode_id,p.source_receipt_ref,p.plan_artifact_ref,j.state AS job_state,j.fail_reason "
                        "FROM plans p JOIN rescue_jobs j USING(job_id) WHERE j.budget_scope=? "
                        "ORDER BY p.updated_at DESC,p.plan_id DESC LIMIT ?", (scope_id, _MAX_ROWS))
                scope_attempts = _query(connection,
                    "SELECT s.job_id,s.attempt_id,s.status,s.reason,s.created_at,s.updated_at,j.task_key "
                    "FROM seed_attempts s JOIN rescue_jobs j USING(job_id) WHERE j.budget_scope=? "
                    "ORDER BY s.updated_at DESC,s.attempt_id DESC LIMIT ?", (scope_id, _MAX_ROWS))
                plan_counts_current = {str(row["status"]): int(row["count"]) for row in connection.execute(
                    "SELECT p.status,count(*) AS count FROM plans p JOIN rescue_jobs j USING(job_id) "
                    "WHERE j.budget_scope=? GROUP BY p.status", (scope_id,))}
                attempt_counts_current = {str(row["status"]): int(row["count"]) for row in connection.execute(
                    "SELECT s.status,count(*) AS count FROM seed_attempts s JOIN rescue_jobs j USING(job_id) "
                    "WHERE j.budget_scope=? GROUP BY s.status", (scope_id,))}
                seen: set[str] = set()
                for rows in (scope_jobs, scope_plans, scope_attempts):
                    for row in rows:
                        key = row.get("task_key")
                        if isinstance(key, str) and key not in seen:
                            seen.add(key)
                            task_keys.append(key)
                            if len(task_keys) >= _MAX_TASKS:
                                break
                    if len(task_keys) >= _MAX_TASKS:
                        break
        current_epoch = context.get("engine_epoch")
        current_version = _int(context.get("policy_version"))
        if current_epoch is not None and not isinstance(current_epoch, str):
            current_epoch = None
        cache_rows: list[dict[str, Any]] = []
        if current_epoch:
            cache_candidate_count = int(connection.execute(
                "SELECT count(*) FROM cache_entries WHERE engine_epoch=?", (current_epoch,)).fetchone()[0])
            cache_kind_counts = {str(row["row_kind"]): int(row["count"]) for row in connection.execute(
                "SELECT row_kind,count(*) AS count FROM cache_entries WHERE engine_epoch=? GROUP BY row_kind",
                (current_epoch,))}
            cache_rows = _query(connection,
                "SELECT c.cache_entry_id,c.task_key,c.engine_epoch,c.source_episode_id,c.source_policy_version,"
                "c.source_policy_sha256,c.row_kind,c.raw_artifact_ref,c.projection_artifact_ref,c.receipt_artifact_ref,"
                "c.committed_uses,c.ready_sequence,c.active,c.created_at,e.task_key AS episode_task_key,"
                "e.engine_epoch AS episode_engine_epoch,e.source_policy_version AS episode_policy_version,"
                "e.source_policy_sha256 AS episode_policy_sha256,e.episode_kind,e.raw_artifact_ref AS episode_raw_ref,"
                "e.projection_artifact_ref AS episode_projection_ref,e.receipt_artifact_ref AS episode_receipt_ref,e.exact AS episode_exact "
                "FROM cache_entries c LEFT JOIN episodes e ON e.episode_id=c.source_episode_id "
                "WHERE c.engine_epoch=? ORDER BY c.ready_sequence DESC,c.created_at DESC LIMIT ?",
                (current_epoch, _MAX_ROWS))
        else:
            cache_candidate_count = int(connection.execute("SELECT count(*) FROM cache_entries").fetchone()[0])
            cache_kind_counts = dict(cache_meta_global)
            cache_rows = _query(connection,
                "SELECT c.cache_entry_id,c.task_key,c.engine_epoch,c.source_episode_id,c.source_policy_version,"
                "c.source_policy_sha256,c.row_kind,c.raw_artifact_ref,c.projection_artifact_ref,c.receipt_artifact_ref,"
                "c.committed_uses,c.ready_sequence,c.active,c.created_at,e.task_key AS episode_task_key,"
                "e.engine_epoch AS episode_engine_epoch,e.source_policy_version AS episode_policy_version,"
                "e.source_policy_sha256 AS episode_policy_sha256,e.episode_kind,e.raw_artifact_ref AS episode_raw_ref,"
                "e.projection_artifact_ref AS episode_projection_ref,e.receipt_artifact_ref AS episode_receipt_ref,e.exact AS episode_exact "
                "FROM cache_entries c LEFT JOIN episodes e ON e.episode_id=c.source_episode_id "
                "ORDER BY c.created_at DESC LIMIT ?", (_MAX_ROWS,))
        task_cache: dict[str, dict[str, Any]] = {}
        for row in cache_rows:
            task_cache.setdefault(str(row["task_key"]), row)
        relevant_keys = task_keys[:_MAX_TASKS]
        if relevant_keys:
            marks = ",".join("?" for _ in relevant_keys)
            native_rows = _query(connection,
                f"SELECT episode_id,task_key,engine_epoch,source_policy_version,exact,reason,created_at "
                f"FROM episodes WHERE task_key IN ({marks}) AND episode_kind='native' AND exact=0 "
                "ORDER BY created_at DESC,episode_id DESC LIMIT ?", (*relevant_keys, _MAX_ROWS))
            latest_native: dict[str, dict[str, Any]] = {}
            for row in native_rows:
                latest_native.setdefault(str(row["task_key"]), row)
        else:
            latest_native = {}
        reservations: list[dict[str, Any]] = []
        if relevant_keys:
            marks = ",".join("?" for _ in relevant_keys)
            args: tuple[Any, ...] = tuple(relevant_keys)
            epoch_filter = ""
            if current_epoch:
                epoch_filter = " AND engine_epoch=?"
                args += (current_epoch,)
            reservations = _query(connection,
                "SELECT reservation_id,batch_id,group_id,cache_entry_id,task_key,engine_epoch,state,"
                "update_attempt_id,created_at,updated_at FROM reservations WHERE task_key IN (" + marks + ")" +
                epoch_filter + " ORDER BY updated_at DESC,reservation_id DESC LIMIT ?", args + (_MAX_ROWS,))
        reservations_truncated = len(reservations) >= _MAX_ROWS
        reservation_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in reservations:
            reservation_counts[str(row["cache_entry_id"])][str(row["state"])] += 1
        attempts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scope_attempts:
            attempts_by_task[str(row["task_key"])].append(row)
        plans_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scope_plans:
            plans_by_task[str(row["task_key"])].append(row)
        jobs_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scope_jobs:
            jobs_by_task[str(row["task_key"])].append(row)
        timeline: list[dict[str, Any]] = []
        as_of = time.time()
        for task_key in relevant_keys:
            jobs = jobs_by_task.get(task_key, [])
            job = jobs[0] if jobs else None
            plans = plans_by_task.get(task_key, [])
            plan = plans[0] if plans else None
            seed_rows = attempts_by_task.get(task_key, [])
            seed_attempt = seed_rows[0] if seed_rows else None
            native = latest_native.get(task_key)
            cache = task_cache.get(task_key)
            cache_state = "no_metadata_candidate"
            provenance = "not_applicable"
            reservation = None
            commit = None
            if cache is not None:
                links_ok = (cache.get("episode_exact") == 1
                    and cache.get("episode_task_key") == task_key
                    and cache.get("episode_engine_epoch") == cache.get("engine_epoch")
                    and cache.get("episode_policy_version") == cache.get("source_policy_version")
                    and cache.get("episode_policy_sha256") == cache.get("source_policy_sha256")
                    and cache.get("episode_raw_ref") == cache.get("raw_artifact_ref")
                    and cache.get("episode_projection_ref") == cache.get("projection_artifact_ref")
                    and cache.get("episode_receipt_ref") == cache.get("receipt_artifact_ref"))
                provenance = "stored_episode_linkage" if links_ok else "metadata_mismatch"
                cache_state = "metadata_candidate_exact_episode_linked" if links_ok else "metadata_candidate_unlinked"
                counts = reservation_counts.get(str(cache["cache_entry_id"]), Counter())
                pending = counts.get("reserved", 0) + counts.get("commit_unknown", 0)
                if pending:
                    cache_state = "metadata_candidate_blocked_pending_or_unknown"
                elif current_version is not None and max_age is not None and max_uses is not None and current_epoch:
                    age = current_version - int(cache["source_policy_version"])
                    if age < 0:
                        cache_state = "metadata_candidate_future_policy"
                    elif not bool(cache["active"]):
                        cache_state = "metadata_candidate_inactive"
                    elif age + 1 > max_age:
                        cache_state = "metadata_candidate_age_blocked"
                    elif reservations_truncated:
                        cache_state = "metadata_candidate_reservation_rows_truncated"
                    elif counts.get("committed", 0) + counts.get("reserved", 0) >= max_uses:
                        cache_state = "metadata_candidate_quota_blocked"
                    elif links_ok:
                        cache_state = "metadata_candidate_within_recorded_limits"
                latest_reservation = next((row for row in reservations if row["cache_entry_id"] == cache["cache_entry_id"]), None)
                if latest_reservation:
                    reservation_truncated = reservations_truncated
                    reservation = {"state": latest_reservation.get("state"),
                        "count": None if reservation_truncated else sum(counts.values()),
                        "committed_uses_from_ledger": None if reservation_truncated else counts.get("committed", 0),
                        "committed_uses_observed": counts.get("committed", 0),
                        "reserved_observed": counts.get("reserved", 0),
                        "released_observed": counts.get("released", 0),
                        "commit_unknown_observed": counts.get("commit_unknown", 0),
                        "truncated": reservation_truncated,
                        "reservation_id": latest_reservation.get("reservation_id"),
                        "created_at": latest_reservation.get("created_at"),
                        "updated_at": latest_reservation.get("updated_at"),
                        "basis": "reservation ledger rows; query capped at 200"}
                    if latest_reservation.get("update_attempt_id"):
                        attempt_row = _first(_query(connection,
                            "SELECT attempt_id,state,new_policy_version,created_at,updated_at FROM update_attempts "
                            "WHERE attempt_id=? LIMIT 1", (latest_reservation["update_attempt_id"],)))
                        if attempt_row:
                            commit = {"state": attempt_row.get("state"), "update_attempt_id": attempt_row.get("attempt_id"),
                                "new_policy_version": attempt_row.get("new_policy_version"),
                                "created_at": attempt_row.get("created_at"), "updated_at": attempt_row.get("updated_at"),
                                "basis": "durable update_attempt ledger"}
            teacher_state = "unobserved"
            teacher_at = None
            teacher_time_basis = "unavailable"
            if plan:
                teacher_state = "accepted"
                teacher_at = plan.get("created_at")
                teacher_time_basis = "persisted_plan_created_at"
            elif job:
                teacher_state = ("accepted" if job.get("state") in {"plan_ready", "cache_ready", "seed_running"}
                                 else str(job.get("state") or "unknown"))
                teacher_at = job.get("updated_at") if job.get("state") in {"source_verified", "plan_ready", "cache_ready", "seed_running"} else job.get("created_at")
                teacher_time_basis = "persisted_job_timestamp"
            plan_state = str(plan.get("status")) if plan else "unobserved"
            plan_reuse = ("retained_for_reactivation; cache eligibility is reevaluated"
                          if plan and plan.get("status") in {"consumed", "exhausted"}
                          else "retained" if plan else None)
            guided_state = ("succeeded" if plan and plan.get("status") == "consumed"
                            else "running" if plan and plan.get("status") == "seed_running"
                            else "failed" if seed_attempt and seed_attempt.get("status") == "failed"
                            else "pending" if plan and plan.get("status") == "ready"
                            else "exhausted" if plan and plan.get("status") == "exhausted"
                            else "unobserved")
            guided_at = (seed_attempt or {}).get("updated_at") or (plan or {}).get("updated_at")
            wait_stage = None
            wait_since = None
            if job and job.get("state") in {"queued", "retry_wait", "teacher_running", "source_submitted", "source_verified"}:
                wait_stage = "teacher"
                wait_since = job.get("updated_at") if job.get("state") == "teacher_running" else job.get("created_at")
            elif plan and plan.get("status") == "ready":
                wait_stage, wait_since = "plan", plan.get("updated_at")
            elif plan and plan.get("status") == "seed_running":
                wait_stage, wait_since = "guided", guided_at
            elif reservation and reservation.get("state") in {"reserved", "commit_unknown"}:
                wait_stage, wait_since = "reservation", reservation.get("created_at")
            timeline.append({
                "task_key": task_key, "scope_id": scope_id,
                "stages": {
                    "native_failure": {"state": "recorded_failure" if native else "not_observed",
                        "at": native.get("created_at") if native else None,
                        "episode_id": native.get("episode_id") if native else None,
                        "reason": native.get("reason") if native else None},
                    "teacher": {"state": teacher_state, "at": teacher_at,
                        "time_basis": teacher_time_basis,
                        "job_id": job.get("job_id") if job else None,
                        "job_state": job.get("state") if job else None,
                        "created_at": job.get("created_at") if job else None,
                        "updated_at": job.get("updated_at") if job else None,
                        "reason": job.get("fail_reason") if job else None},
                    "plan": {"state": plan_state, "at": plan.get("updated_at") if plan else None,
                        "plan_id": plan.get("plan_id") if plan else None,
                        "artifact_ref": plan.get("plan_artifact_ref") if plan else None,
                        "source_episode_id": plan.get("source_episode_id") if plan else None,
                        "source_receipt_ref": plan.get("source_receipt_ref") if plan else None,
                        "reuse": plan_reuse, "attempts": plan.get("seed_attempts") if plan else None},
                    "guided": {"state": guided_state, "attempts": plan.get("seed_attempts") if plan else None,
                        "attempt_cap": 2 if plan else None, "latest_at": guided_at,
                        "reason": seed_attempt.get("reason") if seed_attempt else None},
                    "cache": {"state": cache_state,
                        "cache_entry_id": cache.get("cache_entry_id") if cache else None,
                        "row_kind": cache.get("row_kind") if cache else None,
                        "source_policy_version": cache.get("source_policy_version") if cache else None,
                        "metadata_candidate": cache is not None,
                        "provenance": provenance,
                        "raw_artifact_ref": cache.get("raw_artifact_ref") if cache else None,
                        "projection_artifact_ref": cache.get("projection_artifact_ref") if cache else None,
                        "receipt_artifact_ref": cache.get("receipt_artifact_ref") if cache else None,
                        "central_claim_receipt_recheck": "not_performed" if cache else None,
                        "consumability": "unknown" if cache else None,
                        "stored_committed_uses": cache.get("committed_uses") if cache else None,
                        "committed_uses_from_ledger": reservation.get("committed_uses_from_ledger") if reservation else None},
                    "reservation": reservation,
                    "commit": commit,
                },
                "wait": {"stage": wait_stage, "since": wait_since,
                    "elapsed_seconds": _timestamp_age(wait_since, as_of),
                    "time_basis": "persisted_row_timestamp" if wait_since is not None else "unavailable"},
            })
        jobs_current = {row["state"]: int(row["count"]) for row in connection.execute(
            "SELECT state,count(*) AS count FROM rescue_jobs WHERE budget_scope=? GROUP BY state", (scope_id,))} if scope_id else None
        global_events = _query(connection,
            "SELECT sequence,entity_type,entity_id,action,reason,details_json,created_at FROM events "
            "ORDER BY sequence DESC LIMIT ?", (_MAX_EVENTS,))
        event_view = [ledger_event(connection,row,str(db)) for row in global_events]
        scope_total = scope_row.get("max_total_tokens") if scope_row else None
        pipeline: list[dict[str, Any]] = []
        if scope_id is not None and scope_row is not None:
            job_count = sum(jobs_current.values()) if jobs_current is not None else None
            cache_states = Counter(item["stages"]["cache"]["state"] for item in timeline)
            pipeline = [
                {"id": "teacher_jobs", "label": "Teacher jobs", "state": "observed", "count": job_count, "known": True,
                 "by_state": jobs_current, "source": "current budget-scope rescue_jobs", "ref": str(db)},
                {"id": "plans", "label": "Persisted plans", "state": "observed", "count": sum(plan_counts_current.values()), "known": True,
                 "by_state": plan_counts_current, "source": "current budget-scope plan rows", "ref": str(db)},
                {"id": "guided_attempts", "label": "Persisted guided attempts", "state": "observed", "count": sum(attempt_counts_current.values()), "known": True,
                 "by_state": attempt_counts_current, "source": "current budget-scope seed_attempts rows", "ref": str(db)},
                {"id": "cache_candidates", "label": "Cache metadata candidates", "state": "observed", "count": cache_candidate_count, "known": True,
                 "by_state_observed": dict(cache_states), "observed_rows": len(cache_rows),
                 "truncated": len(cache_rows) >= _MAX_ROWS,
                 "source": "cache_entries; metadata linkage only", "ref": str(db)},
                {"id": "reservations", "label": "Reservations", "state": "observed", "count": sum(reservation_states_global.values()), "known": True,
                 "by_state": reservation_states_global, "source": "all-scope reservation ledger", "ref": str(db)},
            ]
        scope_used = scope_row.get("used_tokens") if scope_row else None
        scope_reserved = scope_row.get("reserved_tokens") if scope_row else None
        scope_unknown = bool(scope_row.get("usage_unknown")) if scope_row else None
        scope_remaining = (max(0, int(scope_total) - int(scope_used) - int(scope_reserved))
                           if scope_total is not None and scope_used is not None
                           and scope_reserved is not None and scope_unknown is False else None)
        scope_info = {"id": scope_id, "available": scope_row is not None if scope_id else False,
            "reason": None if scope_row is not None else ("runtime has no explicit budget_scope" if scope_id is None
                else "configured budget_scope is absent from selected BEPA ledger"),
            "max_total_tokens": scope_total, "max_jobs": scope_row.get("max_jobs") if scope_row else None,
            "used_tokens": scope_used, "reserved_tokens": scope_reserved,
            "remaining_tokens": scope_remaining, "usage_unknown": scope_unknown,
            "jobs_by_state": jobs_current,
            "job_count": sum(jobs_current.values()) if jobs_current is not None else None,
            "basis": "configured runtime budget_scope" if scope_row else "scope unavailable"}
        job_fields = ("job_id", "task_key", "budget_scope", "state", "attempts", "max_attempts",
                      "used_tokens", "reserved_tokens", "max_tokens", "usage_unknown", "fail_reason",
                      "created_at", "updated_at", "lease_expires_at", "source_episode_id",
                      "source_receipt_ref", "plan_artifact_ref")
        job_view = [{key: row.get(key) for key in job_fields} for row in scope_jobs]
        job_by_id = {row.get("job_id"): row for row in scope_jobs}
        plan_view = []
        for row in scope_plans:
            job = job_by_id.get(row.get("job_id"), {})
            plan_view.append({
                "plan_id": row.get("plan_id"), "job_id": row.get("job_id"),
                "task_key": row.get("task_key"), "status": row.get("status"),
                "seed_attempts": row.get("seed_attempts"), "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"), "source_episode_id": row.get("source_episode_id"),
                "source_receipt_ref": row.get("source_receipt_ref"),
                "plan_artifact_ref": row.get("plan_artifact_ref"),
                "job_state": row.get("job_state") or job.get("state"),
                "fail_reason": row.get("fail_reason") or job.get("fail_reason"),
            })
        attempts_view = [{"job_id": row.get("job_id"), "task_key": row.get("task_key"),
            "attempt_id": row.get("attempt_id"), "status": row.get("status"),
            "reason": row.get("reason"), "created_at": row.get("created_at"), "updated_at": row.get("updated_at"),
            "source": "persisted seed_attempts", "ref": f"{db}#seed-{row.get('attempt_id')}"}
            for row in scope_attempts]
        cache_entry_view = []
        for row in cache_rows:
            cache_entry_view.append({
                "cache_entry_id": row.get("cache_entry_id"), "task_key": row.get("task_key"),
                "engine_epoch": row.get("engine_epoch"), "source_episode_id": row.get("source_episode_id"),
                "source_policy_version": row.get("source_policy_version"),
                "source_policy_sha256": row.get("source_policy_sha256"), "row_kind": row.get("row_kind"),
                "raw_artifact_ref": row.get("raw_artifact_ref"),
                "projection_artifact_ref": row.get("projection_artifact_ref"),
                "receipt_artifact_ref": row.get("receipt_artifact_ref"),
                "stored_committed_uses": row.get("committed_uses"), "active_metadata": bool(row.get("active")),
                "created_at": row.get("created_at"),
                "episode_linkage": "stored_episode_linkage" if (
                    row.get("episode_exact") == 1 and row.get("episode_task_key") == row.get("task_key")
                    and row.get("episode_engine_epoch") == row.get("engine_epoch")
                    and row.get("episode_policy_version") == row.get("source_policy_version")
                    and row.get("episode_policy_sha256") == row.get("source_policy_sha256")
                    and row.get("episode_raw_ref") == row.get("raw_artifact_ref")
                    and row.get("episode_projection_ref") == row.get("projection_artifact_ref")
                    and row.get("episode_receipt_ref") == row.get("receipt_artifact_ref")
                ) else "metadata_mismatch",
                "central_claim_receipt_recheck": "not_performed", "consumability": "unknown",
            })
        reservation_view = []
        for row in reservations:
            reservation_view.append({
                "reservation_id": row.get("reservation_id"), "batch_id": row.get("batch_id"),
                "group_id": row.get("group_id"), "cache_entry_id": row.get("cache_entry_id"),
                "task_key": row.get("task_key"), "engine_epoch": row.get("engine_epoch"),
                "state": row.get("state"), "update_attempt_id": row.get("update_attempt_id"),
                "created_at": row.get("created_at"), "updated_at": row.get("updated_at"),
                "source": "reservation ledger", "ref": f"{db}#reservation-{row.get('reservation_id')}"})
        scope_rows_view = [{**row, "is_current_scope": row.get("scope_id") == scope_id}
                           for row in scope_rows]
        if scope_row is not None and not any(row.get("scope_id") == scope_id for row in scope_rows_view):
            selected_scope_view = {**scope_row, "is_current_scope": True}
            if scope_rows_view:
                scope_rows_view[-1] = selected_scope_view
            else:
                scope_rows_view.append(selected_scope_view)
        jobs_truncated = len(scope_jobs) >= _MAX_ROWS
        plans_truncated = len(scope_plans) >= _MAX_ROWS or jobs_truncated
        attempts_truncated = len(scope_attempts) >= _MAX_ROWS
        return {"available": True, "reason": None, "source": str(db),
            "scope": scope_info,
            "scopes": scope_rows_view, "scopes_truncated": scope_rows_truncated,
            "scopes_basis": "selected bounded collector projection when same store; otherwise newest rows, capped at 50; current scope included",
            "jobs_basis": "selected bounded collector projection when same store; otherwise configured scope SQL rows",
            "plans_basis": "selected bounded collector projection when same store; otherwise plans joined to configured scope",
            "jobs": job_view, "jobs_truncated": jobs_truncated,
            "plans": plan_view, "plans_truncated": plans_truncated,
            "attempts": attempts_view, "attempts_truncated": attempts_truncated,
            "attempts_basis": "seed_attempts joined to configured budget scope only; newest rows first; capped at 200",
            "cache_entries": cache_entry_view, "cache_entries_truncated": len(cache_rows) >= _MAX_ROWS,
            "cache_entries_basis": ("selected engine epoch" if current_epoch else "all persisted epochs")
                + "; metadata candidates newest first; capped at 200",
            "reservation_ledger": reservation_view,
            "reservation_ledger_truncated": len(reservations) >= _MAX_ROWS,
            "reservation_ledger_basis": "selected engine epoch and observed current-scope task keys; newest rows first; capped at 200",
            "historical": {"job_states_all_scopes": job_states_global,
                "reservation_states_all_scopes": reservation_states_global,
                "cache_metadata_candidates_by_kind_all_epochs": cache_meta_global,
                "basis": "all persisted rows; independent of current budget scope"},
            "cache": {"metadata_candidates": cache_candidate_count,
                "metadata_candidates_observed": len(cache_rows),
                "metadata_candidates_truncated": len(cache_rows) >= _MAX_ROWS,
                "by_kind": cache_kind_counts,
                "current_epoch": current_epoch, "current_policy_version": current_version,
                "metadata_candidate_within_recorded_limits_observed_scope_tasks": sum(
                    item["stages"]["cache"]["state"] == "metadata_candidate_within_recorded_limits"
                    for item in timeline),
                "ledger_committed_reservation_uses_all_scopes": reservation_states_global.get("committed", 0),
                "ledger_committed_reservation_uses_observed_scope_tasks": sum(
                    int(v.get("committed", 0)) for v in reservation_counts.values()),
                "reservation_rows_observed": len(reservations),
                "reservation_rows_truncated": len(reservations) >= _MAX_ROWS,
                "stored_committed_uses_observed": sum(int(row.get("committed_uses") or 0) for row in cache_rows),
                "stored_committed_uses_truncated": len(cache_rows) >= _MAX_ROWS,
                "provenance_recheck": "not_performed", "basis": "selected engine epoch if observed; otherwise all-epoch metadata candidates"},
            "timeline": timeline, "events": event_view,
            "cohorts": _group_cohorts(run_detail), "pipeline": pipeline,
            "limits": ["All-scope history is separate from configured budget-scope rows.",
                "Cache rows are metadata candidates; central claim receipt and source artifact bytes are not revalidated.",
                "Recorded age/quota checks do not establish consumability.",
                "Committed uses are counted from reservation ledger rows, separately from cache counters.",
                "Per-task rows cover at most 20 tasks; row arrays are capped as shown by their truncation fields."]}
    except sqlite3.Error as exc:
        return {"available": False, "reason": f"BEPA evidence query failed: {type(exc).__name__}",
                "scope": {"id": scope_id, "available": False}, "timeline": [],
                "cohorts": _group_cohorts(run_detail), "attempts": [], "source": str(db)}
    finally:
        connection.close()


def _events(
    run_detail: Any, recovery: Mapping[str, Any], evaluation: Mapping[str, Any],
    bepa: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    detail = _mapping(run_detail)
    run_id = detail.get("run_id")
    records = detail.get("records")
    if isinstance(records, list):
        for record in records[-20:]:
            record = _mapping(record)
            record_id = record.get("record_id")
            record_ref = (f"runs/{run_id}/artifacts/runs/{record_id}.json"
                          if isinstance(run_id, str) and isinstance(record_id, str) else None)
            at = _number(record.get("finished_at")) or _number(record.get("started_at"))
            if at is not None:
                items.append({"id": f"run-record:{run_id}:{record_id}",
                    "type": "training_run_record", "at": at,
                    **severity_fields(record.get("status")),
                    **policy_fields(record.get("policy_version"),"record.policy_version","record_final"),
                    "title": f"{record.get('kind') or 'run'} {record.get('status') or 'observed'}",
                    "detail": {"record_id": record_id, "completed_updates": _int(record.get("updates")),
                               "error_present": bool(record.get("error"))},
                    "source": "selected run record", "ref": record_ref})
            handoffs = record.get("policy_handoffs")
            if isinstance(handoffs, list):
                for index, handoff_value in enumerate(handoffs[-20:]):
                    handoff = _mapping(handoff_value)
                    handoff_at = next((_number(handoff.get(key)) for key in
                        ("updated_at", "completed_at", "created_at", "started_at")
                        if _number(handoff.get(key)) is not None), None)
                    if handoff_at is None:
                        continue
                    handoff_id = handoff.get("handoff_id") or handoff.get("id") or index
                    items.append({"id": f"policy-handoff:{run_id}:{handoff_id}",
                        "type": "policy_handoff", "at": handoff_at,
                        **severity_fields(handoff.get("status")),
                        **policy_fields(handoff.get("policy_version"),"handoff.policy_version","handoff_target"),
                        "title": f"policy handoff {handoff.get('status') or 'observed'}",
                        "detail": {"policy_version": _int(handoff.get("policy_version")),
                                   "status": handoff.get("status"),
                                   "adapter_sha256": handoff.get("adapter_sha256")},
                        "source": "selected run record policy_handoffs", "ref": record_ref})
    for event in bepa.get("events", []) if isinstance(bepa.get("events"), list) else []:
        items.append(dict(event))
    for result in evaluation.get("results", []) if isinstance(evaluation.get("results"), list) else []:
        if result.get("completed_at") is not None or result.get("status") in {"pending", "running"}:
            items.append({"id": f"evaluation:{result.get('request_id')}",
                "type": "native_evaluation", "at": result.get("completed_at") or result.get("created_at"),
                **severity_fields(result.get("status")),
                **policy_fields(result.get("policy_version"),"evaluation.policy_version","evaluation_policy"),
                "title": f"{result.get('kind') or 'evaluation'} {result.get('status') or 'unknown'}",
                "detail": {"policy_version": result.get("policy_version"),
                           "exact_rate": _at(result, "summary", "exact_rate"),
                           "task_count": _at(result, "summary", "task_count")},
                "source": "selected-lineage evaluation registry", "ref": result.get("result_ref") or evaluation.get("source")})
    latest = _at(recovery, "latest")
    if isinstance(latest, Mapping):
        items.append({"id": f"recovery:{latest.get('snapshot_id')}", "type": "recovery_snapshot",
            **severity_fields("snapshot_recorded"),
            **policy_fields(latest.get("policy_version"),"snapshot manifest.policy_version","checkpoint_snapshot"),
            "at": latest.get("created_at"), "title": f"policy v{latest.get('policy_version')} full-state snapshot",
            "detail": {"lineage_id": latest.get("lineage_id"), "complete": _at(latest, "files", "complete")},
            "source": "training recovery latest pointer and snapshot manifest", "ref": recovery.get("source")})
    for item in items:
        stamp = event_timestamp(item.get("at"))
        if stamp is not None:
            item["at"] = stamp
    items.sort(key=lambda item: (_number(item.get("at")) or 0.0, str(item.get("id"))), reverse=True)
    return items[:_MAX_EVENTS]


def _insights(bepa: Mapping[str, Any], evaluation: Mapping[str, Any], recovery: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not recovery.get("available"):
        out.append({"id": "recovery_unavailable", "severity": "unknown", "title": "Recovery lineage unavailable",
                    "message": recovery.get("reason"), "source": recovery.get("source"), "ref": recovery.get("source")})
    elif _at(recovery, "latest", "files", "complete") is False:
        out.append({"id": "recovery_incomplete", "severity": "blocked", "title": "Latest full-state metadata is incomplete",
                    "message": _at(recovery, "latest", "files"), "source": recovery.get("source"), "ref": recovery.get("source")})
    if not evaluation.get("available"):
        out.append({"id": "evaluation_unavailable", "severity": "unknown", "title": "Native evaluation evidence unavailable",
                    "message": evaluation.get("reason"), "source": evaluation.get("source"), "ref": evaluation.get("source")})
    elif evaluation.get("baseline") is None:
        out.append({"id": "evaluation_baseline_missing", "severity": "unknown", "title": "Native baseline request is not recorded",
                    "message": "No baseline request is present in the selected lineage registry.",
                    "source": evaluation.get("source"), "ref": evaluation.get("source")})
    if not bepa.get("available"):
        out.append({"id": "bepa_unavailable", "severity": "unknown", "title": "BEPA ledger unavailable",
                    "message": bepa.get("reason"), "source": bepa.get("source"), "ref": bepa.get("source")})
    else:
        scope = _mapping(bepa.get("scope"))
        if scope.get("usage_unknown") is True:
            out.append({"id": "bepa_usage_unknown", "severity": "blocked", "title": "Teacher usage is unknown for the selected scope",
                        "message": "The persisted budget scope marks provider usage unknown.",
                        "source": bepa.get("source"), "ref": bepa.get("source")})
        unknown = _at(bepa, "historical", "reservation_states_all_scopes", "commit_unknown")
        if isinstance(unknown, int) and unknown > 0:
            out.append({"id": "bepa_commit_unknown", "severity": "blocked", "title": "BEPA reservations have unknown commit outcomes",
                        "message": {"count": unknown, "basis": "all-scope reservation ledger"},
                        "source": bepa.get("source"), "ref": bepa.get("source")})
    return out


def evidence_for(
    collector: Any,
    run_detail: Mapping[str, Any] | None,
    snapshot: Mapping[str, Any] | None,
    live_selected: Mapping[str, Any] | bool | None,
    base_bepa: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build dashboard evidence from one explicit run/source selection.

    ``base_bepa`` is the collector's already-bounded projection for the selected run.
    Its rows are reused only when its resolved store matches the explicit source path.
    Paths are chosen only through ``source_info`` and ambiguous selection stays unavailable.
    """
    run_id = _selected_run_id(run_detail, live_selected)
    source = _source_info(collector, run_id)
    if source.get("available") is False and source.get("reason"):
        source_view = {"kind": None, "reason": source.get("reason"), "run_id": run_id}
    else:
        source_view = {"kind": source.get("kind"), "deployment": source.get("deployment"),
                       "runtime_config": source.get("runtime_config"), "run_id": run_id,
                       "paths": dict(_mapping(source.get("paths")))}
    recovery = _recovery_evidence(source, run_detail) if source.get("available", True) else {
        "available": False, "reason": source.get("reason"), "source": None}
    context = _live_context(snapshot, live_selected, recovery, run_detail)
    evaluation = _evaluation_evidence(
        source, recovery, run_detail, snapshot if live_selected is True else None
    ) if source.get("available", True) else {
        "available": False, "reason": source.get("reason"), "source": None}
    bepa = _bepa_evidence(source, base_bepa, context, run_detail) if source.get("available", True) else {
        "available": False, "reason": source.get("reason"), "source": None,
        "scope": None, "timeline": [], "attempts": [], "cohorts": _group_cohorts(run_detail), "events": [], "pipeline": []}
    events = _events(run_detail, recovery, evaluation, bepa)
    return {"source": source_view,
            "selection": {"run_id": run_id, "engine_epoch": context.get("engine_epoch"),
                          "policy_version": context.get("policy_version"), "basis": context.get("basis")},
            "run": {"run_id": run_id, "cohorts": _group_cohorts(run_detail),
                    "record_count": len(_mapping(run_detail).get("records", [])) if isinstance(_mapping(run_detail).get("records"), list) else None},
            "bepa": bepa, "evaluation": evaluation, "recovery": recovery,
            "events": events, "insights": _insights(bepa, evaluation, recovery),
            "pipeline": bepa.get("pipeline", []) if isinstance(bepa.get("pipeline"), list) else []}
