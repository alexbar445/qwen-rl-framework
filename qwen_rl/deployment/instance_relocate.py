"""Resumable same-host relocation using the existing migration and instance owners."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import stat
import time
from typing import Any, Callable, Mapping

import yaml

from . import instance_migration as migration
from .instance_plan import InstanceDeployment, InstanceDeploymentError, load_deployment

_SCHEMA = "qwen-rl-instance-relocate/v2"
_HEADROOM = 512 * 1024**2


class InstanceRelocationError(InstanceDeploymentError):
    def __init__(self, code: str, field: str, message: str):
        self.report = {"schema": _SCHEMA, "status": "blocked", "reason_code": code,
                       "field": field, "message": message}
        super().__init__(message)


def _blocked(code: str, field: str, message: str) -> None:
    raise InstanceRelocationError(code, field, message)


@contextmanager
def _operation_lock(state: Path):
    migration._private_dir(state)
    fd = os.open(state / "relocate-operation.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or info.st_mode & 0o077):
            _blocked("operation_lock_invalid", "state_dir", "Relocation lock is not private and owned.")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _blocked("operation_already_running", "state_dir", "Another relocation owns this deployment.")
        yield
    finally:
        os.close(fd)


def _file_identity(path: Path) -> list[int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        _blocked("input_not_regular", "input", "Relocation requires regular input files.")
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _template(source: InstanceDeployment, layout: Mapping[str, Any], prefix: Path,
              cache: Path, *, preview: bool, model_installation) -> InstanceDeployment:
    """Target environment template; final paths are emitted by the existing importer."""
    template = yaml.safe_load(source.runtime_config.read_text())
    bundle = prefix / "bundle"
    model_installation.apply_to_runtime(template)
    template["dataset_dir"] = str(bundle / "static/dataset")
    template["run_dir"] = str(prefix / "preflight/run")
    learner = template["learner"]
    learner["recovery_dir"] = str(prefix / "preflight/checkpoints")
    learner["initial_adapter_path"] = str(bundle / "static/artifact/initial-adapter")
    for section, field, static_name in (
        ("arc", "contract_path", "contracts/arc-contract"),
        ("evaluation", "manifest_path", "evaluation/manifest"),
        ("evaluation", "sft_exclusion_proof", "evaluation/sft-exclusion-proof"),
        ("evaluation", "initial_count_proof", "evaluation/initial-count-proof"),
    ):
        if isinstance(template.get(section), dict) and template[section].get(field):
            template[section][field] = str(bundle / "static" / static_name)
    if isinstance(template.get("bepa"), dict):
        # Preview only reads the live ledger; final import emits an independent copy.
        template["bepa"]["state_dir"] = (
            str(layout["bepa_state_dir"]) if preview else str(prefix / "preflight/bepa")
        ) if template["bepa"].get("enabled") is True else ""
    if isinstance(template.get("evaluation"), dict):
        template["evaluation"]["state_dir"] = str(prefix / "preflight/evaluation")
    folder = prefix / ("preview-config" if preview else "target-template")
    migration._private_dir(folder)
    runtime_path = folder / "runtime.yaml"
    data = yaml.safe_dump(template, sort_keys=False).encode()
    _atomic_bytes(runtime_path, data)
    target = replace(source, path=folder / "deployment.json", runtime_config=runtime_path,
                     state_dir=prefix / "supervisor", cache_root=cache)
    raw = target.public_dict()
    raw.pop("path", None)
    migration._atomic_json(target.path, raw)
    return target


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    migration._fsync_dir(path.parent)


def _copy_budget(layout: Mapping[str, Any], prefix: Path) -> int:
    """Bound remaining bundle/import/temp copies; omit unreferenced checkpoint history."""
    pointer = migration._read_json(Path(layout["recovery_dir"]) / "latest.json")
    revision = migration.source_revision(layout)
    recovery = Path(layout["recovery_dir"])
    durable = 0
    for name, info in revision["files"].items():
        path = Path(name)
        if path.parent == recovery and migration._SNAPSHOT_NAME.fullmatch(path.name):
            if path.name != pointer["snapshot"]:
                continue
        durable += info[2]
    static_bytes = 0
    for kind, path in migration._static_sources(layout):
        if kind == "runtime_config":
            continue
        if path.is_file():
            static_bytes += path.stat().st_size
        else:
            static_bytes += sum(info.st_size for _, _, info in migration._source_entries(path))
    # Model and report are shared frozen inputs. Budget only bundled static
    # inputs and the final durable state, installation and temporary checkpoint.
    return 2 * static_bytes + 3 * durable


def _online_storage(source: Any, layout: Mapping[str, Any], prefix: Path, cache: Path) -> dict[str, Any]:
    from .instance_storage import preflight_storage, _filesystem_info

    plan = copy.deepcopy(source.prepare())
    runtime = json.loads(json.dumps(plan["runtime"]))
    runtime["run_dir"] = str(prefix / "preflight/run")
    runtime["learner"]["recovery_dir"] = str(prefix / "preflight/checkpoints")
    plan = {**plan, "runtime": runtime, "cache_root": str(cache),
            "storage_checkpoint_source": layout["recovery_dir"]}
    for row in plan["executors"]:
        row["compute_profile"] = {**row["compute_profile"],
                                  "packed_weight_offload_store_dir": str(cache / "packed-frozen-base")}
    report = preflight_storage(source.deployment, plan, runtime)
    destination = _filesystem_info(prefix)
    copy_bytes = _copy_budget(layout, prefix)
    matches = [row for row in report["filesystems"]
               if row["filesystem_id"] == destination["filesystem_id"]]
    required = copy_bytes + _HEADROOM + sum(row["required_free_bytes"] for row in matches)
    report["relocation_copies"] = {**destination, "remaining_peak_bytes": copy_bytes,
                                   "required_free_bytes": required}
    if destination["free_bytes"] < required:
        report["status"] = "blocked"
        report["errors"].append({"reason_code": "relocation_space_insufficient", "field": "data_root",
                                  "message": "Destination lacks space for bundle, import, and atomic checkpoint copies."})
    return report


def _source_file(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> Path | None:
    path = row["path"]
    roots = {"durable/run": "run_dir", "durable/recovery": "recovery_dir",
             "durable/bepa": "bepa_state_dir", "durable/evaluation-state": "evaluation_state_dir"}
    for prefix, key in roots.items():
        if path.startswith(prefix + "/") and manifest["source"].get(key):
            return Path(manifest["source"][key]) / path[len(prefix) + 1:]
    return None


def _cleanup_source(source: Any, target: Any, bundle: Path, record: dict[str, Any],
                    save: Callable[[], None]) -> dict[str, Any]:
    from .instance_operations import _observe
    from .local_gradients import _read_private_json

    target_plan = target.prepare()
    active = _read_private_json(target.active_path) or {}
    current = _observe(target, target_plan)
    verified = record.get("training_observation") or record.get("resume", {}).get("observation", {})
    expected_lineage = record.get("cpu_validated", {}).get("preserved", {}).get("lineage_id")
    if (not record.get("target_verified") or not current.get("ready")
            or (expected_lineage and current.get("lineage_id") != expected_lineage)
            or (type(verified.get("policy_version")) is int
                and (type(current.get("policy_version")) is not int
                     or current["policy_version"] < verified["policy_version"]))
            or active.get("config_fingerprint") != target_plan.get("config_fingerprint")
            or (verified.get("run_id") and current.get("run_id") != verified["run_id"])):
        _blocked("target_not_verified_for_cleanup", "cleanup_source", "Target ownership, original run, and current Worker ACKs must still be healthy before source cleanup.")

    if source.status().get("status") != "stopped":
        _blocked("source_not_stopped", "state_dir", "Source cleanup requires its stopped, fenced manager.")
    manifest = migration._read_manifest(bundle, require_exported=True)
    fence = migration._read_json(migration._fence_path(source.deployment), private=True)
    if fence.get("migration_id") != manifest["migration_id"]:
        _blocked("source_fence_changed", "state_dir", "The source fence no longer belongs to this relocation.")
    migration._verify_bundle_files(bundle, manifest)
    source_files = manifest.get("source_revision", {}).get("files", {})
    plan = record.setdefault("cleanup", {"deleted": [], "retained": [], "allowlist": []})
    protected = {source.deployment.path.resolve(), source.deployment.runtime_config.resolve()}
    protected.update(Path(path).resolve() for path in source.deployment.credential_files.values())
    with migration._owner_lock(source.deployment):
        if source.status().get("status") != "stopped":
            _blocked("source_not_stopped", "state_dir", "Source ownership changed before cleanup.")
        if not plan["allowlist"]:
            for row in manifest["files"]:
                path = _source_file(row, manifest)
                if path is None or path in protected or path.is_relative_to(source.deployment.state_dir):
                    continue
                identity = source_files.get(str(path))
                if identity is not None:
                    plan["allowlist"].append({"path": str(path), "stat": identity,
                                               "bundle_path": row["path"]})
            # A SQLite backup includes committed WAL content. Only sidecars whose
            # exact final revision was captured belong to this deletion plan.
            database_paths = {item["path"] for item in plan["allowlist"] if item["path"].endswith(".sqlite3")}
            for name, identity in source_files.items():
                if name.endswith("-wal") and name[:-4] in database_paths:
                    plan["allowlist"].append({"path": name, "stat": identity, "bundle_path": None})
            save()  # Persist the reviewed copy allowlist before any unlink.
        allowed = {item["path"] for item in plan["allowlist"]}
        current_revision = migration.source_revision(manifest["source"])
        current_files = current_revision["files"]
        expected = {name: identity for name, identity in source_files.items()
                    if name not in allowed or name in current_files}
        if (current_files != expected or current_revision["absent_roots"] != manifest["source_revision"]["absent_roots"]):
            _blocked("source_changed_before_cleanup", "cleanup_source", "Source durable files or SQLite WAL changed after the final snapshot; source is retained.")
        from .instance_file_references import audit_file_references
        references = audit_file_references([Path(item["path"]) for item in plan["allowlist"]
                                            if item["path"] in current_files])
        plan["retained"] = []
        for item in plan["allowlist"]:
            path = Path(item["path"])
            if str(path) in plan["deleted"]:
                continue
            if not path.exists() and not path.is_symlink():
                plan["deleted"].append(str(path)); save(); continue
            if path.resolve() != path or _file_identity(path) != item["stat"]:
                _blocked("source_changed_before_cleanup", "cleanup_source", "A source file changed after its final snapshot; no further files were removed.")
            referenced = references.get(str(path))
            if referenced is not False:
                plan["retained"].append({"path": str(path), "reason": "referenced" if referenced else "reference_audit_unknown"})
                save(); continue
            path.unlink()
            migration._fsync_dir(path.parent)
            plan["deleted"].append(str(path)); save()
    plan["shared_inputs_retained"] = True
    plan["unexported_history_and_caches_retained"] = True
    plan["status"] = "completed_with_retained_files" if plan["retained"] else "passed"
    save()
    return plan


def relocate_instance(source: Any, *, data_root: Path, cache_root: Path,
                      resume: bool = False, cleanup_source: bool = False,
                      timeout_seconds: float | None = None,
                      progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Prepare online, stop/snapshot once, validate on CPU, and optionally resume."""
    if timeout_seconds is not None and (isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        _blocked("invalid_timeout", "timeout", "Timeout must be a positive finite number.")
    if cleanup_source and not resume:
        _blocked("cleanup_requires_resume", "cleanup_source", "Source cleanup requires a verified target resume.")
    data_root, cache_root = Path(data_root).expanduser().resolve(), Path(cache_root).expanduser().resolve()
    op_id = hashlib.sha256(str(source.deployment.path).encode()).hexdigest()[:16]
    prefix = data_root / "instances" / op_id
    cache = cache_root / op_id
    runtime = migration._load_runtime(source.deployment)
    layout = migration._source_layout(source.deployment, runtime)
    for key in ("state_dir", "run_dir", "recovery_dir", "model_path", "dataset_dir"):
        if migration._overlap(prefix, Path(layout[key])) or migration._overlap(cache, Path(layout[key])):
            _blocked("target_overlaps_source", "data_root", "Data/cache destinations must be separate from source state and inputs.")
    migration._private_dir(prefix)
    path = prefix / "operation.json"
    started = time.monotonic()
    next_command = (f"qwen-rl instance --deployment {shlex.quote(str(source.deployment.path))} relocate "
                    f"--data-root {shlex.quote(str(data_root))} --cache-root {shlex.quote(str(cache_root))}"
                    + (" --resume" if resume else "") + (" --cleanup-source" if cleanup_source else ""))
    with _operation_lock(source.deployment.state_dir):
        record = migration._read_json(path, private=True) if path.exists() else {
            "schema": _SCHEMA, "operation_id": op_id, "source": str(source.deployment.path),
            "data_root": str(data_root), "cache_root": str(cache_root), "status": "running", "stages": [],
        }
        if (record.get("schema") != _SCHEMA or record.get("source") != str(source.deployment.path)
                or record.get("cache_root") != str(cache_root)):
            _blocked("operation_identity_changed", "data_root", "The existing operation belongs to different source/cache inputs.")
        if record.get("status") == "passed" and record.get("cleanup_requested", False) == cleanup_source:
            return record
        for field in ("failure", "error_type", "failure_locations", "reason_code",
                      "online_preflight_failure", "target_cpu_failure", "resume"):
            record.pop(field, None)
        record.update(status="running", next_command=next_command, cleanup_requested=cleanup_source)
        def save():
            migration._atomic_json(path, record)
        def emit(event: Mapping[str, Any]):
            item = {**dict(event), "elapsed_seconds": round(time.monotonic() - started, 3)}
            record["current_stage"] = item.get("stage")
            record["stages"] = [*record["stages"][-63:], item]
            save()
            if progress: progress(item)
        save()
        try:
            from .instance import InstanceSupervisor
            from .instance_operations import operation_lock, resume_instance, _observe
            from .instance_preflight import preflight
            from .instance_recovery import managed_recover
            from .instance_artifact import ModelInstallation

            with operation_lock(source.deployment.state_dir):
                bundle = prefix / "bundle"
                if not record.get("imported"):
                    emit({"stage": "online_storage_preflight", "status": "running"})
                    record["storage"] = _online_storage(source, layout, prefix, cache)
                    if record["storage"]["status"] != "passed":
                        _blocked("storage_preflight_blocked", "storage", "Resolve the recorded filesystem capacity blockers before stopping the source.")
                    model_installation = ModelInstallation.from_source_layout(layout).with_identity(
                        prefix / "model/identity.json",
                    )
                    record["model_installation"] = {
                        **model_installation.public_dict(), "storage_kind": "shared_frozen_input",
                    }
                    save()
                    if not bundle.exists():
                        emit({"stage": "online_static_prepare", "status": "running"})
                        record["prepared"] = migration.prepare_instance_migration(source.deployment, bundle, progress=emit)
                        save()
                    if not record.get("imported"):
                        preview = _template(source.deployment, layout, prefix, cache, preview=True, model_installation=model_installation)
                        preview_supervisor = InstanceSupervisor(preview)
                        preview_plan = preview_supervisor.prepare()
                        preview_plan["storage_checkpoint_source"] = layout["recovery_dir"]
                        emit({"stage": "target_environment_preflight", "status": "running"})
                        checked = preflight(preview_supervisor, plan=preview_plan)
                        if checked.get("status") != "passed":
                            record["online_preflight_failure"] = checked
                            _blocked("target_preflight_blocked", "runtime_config", "Target dependencies or inputs failed CPU validation; the source was not stopped.")
                        record["online_preflight"] = checked; save()
                    target_template = _template(source.deployment, layout, prefix, cache, preview=False, model_installation=model_installation)
                    emit({"stage": "stop_and_capture_final_snapshot", "status": "running"})
                    manifest = migration._read_manifest(bundle)
                    # This flow already owns operation.lock for the source.
                    with migration._migration_operation_lock(source.deployment):
                        record["exported"] = migration._export_instance_migration_locked(
                            source.deployment, bundle, stop=True,
                            refresh=manifest["status"] == "exported" and not (bundle / "claim.json").exists(),
                            progress=emit,
                        )
                    save()
                    emit({"stage": "import_final_snapshot", "status": "running"})
                    claim = migration.import_instance_migration(bundle, prefix / "instance",
                                                                 deployment=target_template, progress=emit)
                    record["imported"] = claim
                    record["target_deployment"] = claim["configuration"]["deployment"]
                    save()
                target = InstanceSupervisor(load_deployment(record["target_deployment"]))
                if not resume and not record.get("cpu_validated"):
                    emit({"stage": "target_cpu_restore_and_evaluation", "status": "running"})
                    checked = managed_recover(target.deployment)
                    if checked.get("status") != "passed":
                        record["target_cpu_failure"] = checked
                        _blocked("target_cpu_validation_blocked", "checkpoint", "Target Core recovery/evaluation failed before GPU startup; the source is retained.")
                    record["cpu_validated"] = checked; save()
                if not resume:
                    record.update(status="prepared", next_command=(
                        f"qwen-rl instance --deployment {shlex.quote(record['target_deployment'])} resume --wait ready"))
                    save(); return record
                emit({"stage": "target_resume", "status": "running"})
                resumed = resume_instance(target, wait="ready", timeout_seconds=timeout_seconds, progress=emit)
                record["resume"] = resumed
                if resumed.get("recovery", {}).get("status") == "passed":
                    record["cpu_validated"] = resumed["recovery"]
                save()
                if resumed.get("status") != "passed":
                    record["status"] = resumed.get("status", "blocked"); save(); return record
                logical = resumed.get("resume_plan", {}).get("logical_run") or record.get("cpu_validated", {}).get("preserved", {})
                pending = logical.get("state") == "pending" or (type(logical.get("remaining_steps")) is int and logical["remaining_steps"] > 0)
                baseline = record["exported"]["checkpoint"]["policy_version"]
                if pending:
                    deadline = time.monotonic() + (timeout_seconds or target.deployment.start_timeout)
                    while True:
                        observation = _observe(target, target.prepare())
                        record["training_observation"] = observation
                        if observation.get("ready") and type(observation.get("policy_version")) is int and observation["policy_version"] > baseline:
                            break
                        if (observation.get("instance_status") in {"failed", "orphaned", "stop_incomplete", "stopped"}
                                or observation.get("run_status") in {"failed", "cancelled"}):
                            _blocked("target_failed_before_optimizer_progress", "logical_run", "Target failed before the original run made an optimizer update; source is retained.")
                        if time.monotonic() >= deadline:
                            record.update(status="pending", reason_code="waiting_for_original_run_optimizer_progress")
                            save(); return record
                        emit({"stage": "waiting_for_original_run_update", "status": "running"})
                        time.sleep(2)
                record["target_verified"] = True; save()
                if cleanup_source:
                    emit({"stage": "cleanup_verified_source_files", "status": "running"})
                    _cleanup_source(source, target, bundle, record, save)
                record.update(status="passed", next_command=(
                    f"qwen-rl instance --deployment {shlex.quote(record['target_deployment'])} status --human"))
                emit({"stage": "complete", "status": "passed"})
                return record
        except Exception as error:
            record["status"] = "blocked"
            record["error_type"] = type(error).__name__
            from .instance_failure import source_failure_locations
            record["failure_locations"] = source_failure_locations(error)
            detail = getattr(error, "report", None)
            if isinstance(detail, Mapping): record["failure"] = dict(detail)
            else: record["failure"] = {"reason_code": "relocation_stage_failed", "field": record.get("current_stage")}
            save()
            return record
