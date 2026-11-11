"""Checked background resume of the existing instance supervisor.

The operation lock serializes commands; owner.lock and the optimizer lock retain
process and optimizer ownership. This module never implements another supervisor.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
import time
from typing import Any, Callable
import uuid

from .instance import InstanceSupervisor, _mkdir, _secret, configure_cache_environment
from .local_gradients import _http_json, _read_private_json, _write_private_json


@contextmanager
def operation_lock(state_dir: Path):
    _mkdir(state_dir)
    fd = os.open(state_dir / "operation.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o077:
            raise RuntimeError("operation lock must be private and owned by this user")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _observe(supervisor: InstanceSupervisor, plan: dict[str, Any]) -> dict[str, Any]:
    state = supervisor.status()
    if state["status"] != "running":
        return {"ready": False, "instance_status": state["status"],
                "teacher_provider": _teacher_provider_pending(plan)}
    _config, token, url = supervisor._control_context()
    health = _http_json(url + "/health", token)
    monitor = _http_json(url + "/monitor", token)
    version = health.get("policy_version")
    expected = {row["worker_id"] for row in plan["workers"]}
    acks: dict[str, int] = {}
    local_ready = health.get("ready") is True and health.get("role") == "learner" and type(version) is int
    if local_ready:
        policy = _http_json(url + f"/policy/{version}", token)
        adapter = policy.get("source_adapter_sha256")
        local_ready = policy.get("version") == version and isinstance(adapter, str) and len(adapter) == 64
        frozen_workers = {row["id"]: row for row in _config.get("workers", [])}
        for row in plan["workers"]:
            worker_token = _secret(supervisor.deployment.state_dir / "tokens" / (row["worker_id"] + ".token"))
            worker = _http_json(f"http://127.0.0.1:{row['port']}/health", worker_token)
            pin = frozen_workers.get(row["worker_id"], {})
            matches = (worker.get("ready") is True and worker.get("worker_id") == row["worker_id"]
                       and worker.get("role") == "worker" and worker.get("gpu_uuid") == row["gpu_uuid"]
                       and isinstance(pin.get("boot_id"), str) and worker.get("boot_id") == pin["boot_id"]
                       and isinstance(pin.get("telemetry_epoch"), str)
                       and worker.get("telemetry_epoch") == pin["telemetry_epoch"]
                       and worker.get("base_fingerprint") == policy.get("base_fingerprint")
                       and worker.get("artifact_fingerprint") == policy.get("artifact_fingerprint")
                       and any(isinstance(resident, dict)
                               and resident.get("policy_version") == version
                               and resident.get("adapter_hash") == adapter
                               for resident in worker.get("resident_versions", [])))
            if matches:
                acks[row["worker_id"]] = version
            local_ready = local_ready and matches
        local_ready = local_ready and set(acks) == expected
    recovery = health.get("recovery") or {}
    run = monitor.get("run") or {}
    from .teacher_readiness import teacher_provider_status
    bepa = _config.get("bepa") if isinstance(_config.get("bepa"), dict) else {}
    teacher_required = bepa.get("enabled") is True
    if local_ready or not teacher_required:
        provider = teacher_provider_status(_config, supervisor.deployment.credential_files)
    else:
        provider = _teacher_provider_pending(_config)
    provider_passed = provider.get("status") == ("passed" if teacher_required else "not_required")
    ready = local_ready and provider_passed
    return {"ready": ready, "instance_status": state["status"],
            "policy_version": health.get("policy_version"), "ready_policy_version": version if ready else None,
            "learner_reported_serve_policy_version": monitor.get("serve_policy_version"),
            "run_id": run.get("run_id"), "run_status": run.get("status"),
            "completed_updates": run.get("completed_updates"),
            "recovery_mode": recovery.get("mode"), "lineage_id": recovery.get("lineage_id"), "worker_acks": acks,
            "worker_ack_scope": "authenticated_live_resident_policy_and_pinned_worker_identity",
            "teacher_provider": provider}


def _teacher_provider_pending(config: dict[str, Any]) -> dict[str, Any]:
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else config
    bepa = runtime.get("bepa") if isinstance(runtime.get("bepa"), dict) else {}
    if not bepa.get("enabled"):
        return {"required": False, "status": "not_required"}
    return {"required": True, "status": "not_performed",
            "reason_code": "local_roles_not_ready"}


def _launch(supervisor: InstanceSupervisor, gpus: tuple[int, ...], *, preflight_receipt: Path | None = None) -> tuple[subprocess.Popen, Path]:
    deployment = supervisor.deployment
    log = deployment.state_dir / ("resume-supervisor-" + uuid.uuid4().hex + ".log")
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    command = [deployment.python, "-B", "-m", "qwen_rl", "instance",
               "--deployment", str(deployment.path), "--gpus", ",".join(map(str, gpus)), "serve"]
    if preflight_receipt is not None:
        command.extend(("--preflight-receipt", str(preflight_receipt)))
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=fd, stderr=subprocess.STDOUT,
                                   env=env, cwd=root, start_new_session=True)
    finally:
        os.close(fd)
    return process, log


def resume_instance(supervisor: InstanceSupervisor, *, wait: str = "ready",
                    timeout_seconds: float | None = None,
                    progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Preflight, recover on CPU, start or reuse, then verify authenticated readiness.

    An observation timeout leaves the managed instance running and reports pending;
    it never kills a long-running startup or silently submits a fresh training run.
    """
    if wait not in {"ready", "update"}:
        raise ValueError("wait must be ready or update")
    timeout = supervisor.deployment.start_timeout if timeout_seconds is None else timeout_seconds
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    started = time.monotonic()
    result: dict[str, Any] = {"schema": "qwen-rl-instance-resume/v1", "status": "blocked",
                              "wait": wait, "timeout_scope": "readiness_observation_after_cpu_recovery_and_handoff",
                              "stages": [], "reused": False,
                              "next_command": f"qwen-rl instance --deployment {supervisor.deployment.path} status --human"}

    def emit(stage: str, status: str, **fields: Any) -> None:
        event = {"stage": stage, "status": status, "elapsed_seconds": round(time.monotonic() - started, 3), **fields}
        result["stages"].append(event)
        if progress is not None:
            progress(event)

    handoff_path = None
    handoff_completed = False
    process = None
    try:
        with operation_lock(supervisor.deployment.state_dir):
            from .instance_migration import assert_not_migrating
            assert_not_migrating(supervisor.deployment)
            stop_operation = _read_private_json(supervisor.deployment.state_dir / "stop-operation.json")
            if (stop_operation and stop_operation.get("mode") == "now"
                    and stop_operation.get("manager_status") != "completed"):
                deployment_arg = str(supervisor.deployment.path)
                result.update(
                    reason_code="stop_now_incomplete", field="state_dir.stop_operation",
                    next_command=shlex.join(["qwen-rl", "instance", "--deployment",
                                             deployment_arg, "status", "--human"]),
                    stop_now_command=shlex.join(["qwen-rl", "instance", "--deployment",
                                                 deployment_arg, "stop-now"]),
                    resume_command=shlex.join(["qwen-rl", "instance", "--deployment",
                                               deployment_arg, "resume"]),
                )
                return result
            if (stop_operation and stop_operation.get("mode") == "after_checkpoint"
                    and stop_operation.get("manager_status") == "blocked"
                    and (stop_operation.get("learner_receipt") or {}).get("request_accepted") is True):
                deployment_arg = str(supervisor.deployment.path)
                result.update(
                    reason_code="accepted_checkpoint_stop_blocked", field="state_dir.stop_operation",
                    next_command=shlex.join(["qwen-rl", "instance", "--deployment",
                                             deployment_arg, "status", "--human"]),
                    stop_now_command=shlex.join(["qwen-rl", "instance", "--deployment",
                                                 deployment_arg, "stop-now"]),
                    resume_command=shlex.join(["qwen-rl", "instance", "--deployment",
                                               deployment_arg, "resume"]),
                )
                return result
            configure_cache_environment(supervisor.deployment)
            state = supervisor.status()
            if state["status"] not in {"stopped", "running", "starting"}:
                result.update(reason_code="owned_instance_not_stopped_or_healthy", field="state_dir.active_manifest")
                return result
            plan = supervisor.prepare()
            process = None
            if state["status"] in {"running", "starting"}:
                active = _read_private_json(supervisor.active_path) or {}
                if active.get("config_fingerprint") != plan["config_fingerprint"]:
                    result.update(reason_code="running_configuration_unproven_or_changed", field="runtime_config")
                    return result
                result["reused"] = True
                emit("existing_instance", "passed")
                try:
                    baseline = _observe(supervisor, plan) if state["status"] == "running" else {}
                except (OSError, ValueError, RuntimeError):
                    baseline = {}
            else:
                from .instance_preflight import preflight
                emit("preflight", "running")
                report = preflight(supervisor, plan=plan, progress=lambda event: emit(
                    "preflight." + event["stage"], event["status"],
                    **{key: event[key] for key in ("duration_seconds", "field") if key in event},
                ))
                result["preflight"] = report
                if report.get("status") != "passed":
                    result.update(reason_code="preflight_blocked", field="runtime_config")
                    emit("preflight", "blocked")
                    return result
                emit("preflight", "passed")
                from .instance_startup import create_handoff, _plan_fingerprint
                handoff_path = create_handoff(supervisor, plan, report)
                from .instance_recovery import managed_resume_transaction
                emit("managed_recovery", "running")
                transaction = managed_resume_transaction(supervisor.deployment)
                if isinstance(transaction, dict):
                    result["resume_transaction"] = transaction
                else:
                    transaction = {}
                failure = transaction.get("failure")
                recovery = transaction.get("recovery")
                if isinstance(recovery, dict):
                    result["recovery"] = recovery
                elif not (isinstance(failure, dict)
                          and failure.get("stage") == "initial_resume_plan"):
                    recovery = None
                checkpoints = transaction.get("sqlite_wal_checkpoints")
                if isinstance(checkpoints, list):
                    result["sqlite_wal_checkpoints"] = checkpoints
                checked_plan = transaction.get("resume_plan")
                if isinstance(checked_plan, dict):
                    result["resume_plan"] = checked_plan
                post_recovery = transaction.get("post_recovery")
                if isinstance(post_recovery, dict):
                    result["post_recovery"] = post_recovery
                if isinstance(failure, dict) and failure.get("stage") == "initial_resume_plan":
                    result.update(reason_code="resume_stage_failed",
                                  error_type=failure.get("error_type", "RuntimeError"),
                                  failure_locations=failure.get("failure_locations", []))
                    return result
                if isinstance(recovery, dict) and recovery.get("status") == "passed":
                    emit("managed_recovery", "passed")
                elif isinstance(failure, dict):
                    result.update(reason_code="resume_stage_failed",
                                  error_type=failure.get("error_type", "RuntimeError"),
                                  failure_locations=failure.get("failure_locations", []))
                    return result
                if not isinstance(recovery, dict) or recovery.get("status") != "passed":
                    result.update(reason_code="managed_recovery_blocked", field="learner.recovery_dir")
                    emit("managed_recovery", "blocked")
                    return result
                if isinstance(failure, dict):
                    result.update(reason_code="resume_stage_failed",
                                  error_type=failure.get("error_type", "RuntimeError"),
                                  failure_locations=failure.get("failure_locations", []))
                    return result
                if not isinstance(post_recovery, dict):
                    post_recovery = {"status": "blocked"}
                    result["post_recovery"] = post_recovery
                if post_recovery.get("status") != "passed":
                    result.update(reason_code="resume_plan_blocked", field="logical_run")
                    return result
                from .instance_storage import preflight_storage
                storage = preflight_storage(supervisor.deployment, plan, plan["runtime"])
                result["post_recovery_storage"] = storage
                if storage.get("status") != "passed":
                    result.update(reason_code="post_recovery_storage_blocked", field="storage")
                    return result
                baseline = {"policy_version": recovery.get("preserved", {}).get("policy_version")}
                process, log = _launch(supervisor, supervisor.deployment.gpus,
                                       preflight_receipt=handoff_path)
                result["supervisor_log"] = str(log)
                accepted_path = Path(str(handoff_path) + ".accepted")
                handoff_deadline = time.monotonic() + 30
                while not accepted_path.exists():
                    if process.poll() is not None:
                        result.update(reason_code="owned_supervisor_exited_before_handoff", field="state_dir")
                        return result
                    if time.monotonic() >= handoff_deadline:
                        result.update(reason_code="startup_handoff_timeout", field="state_dir")
                        return result
                    time.sleep(0.05)
                accepted = _read_private_json(accepted_path) or {}
                if (accepted.get("status") != "passed" or accepted.get("pid") != process.pid
                        or accepted.get("plan_fingerprint") != _plan_fingerprint(plan)):
                    result.update(reason_code="startup_handoff_ack_invalid", field="state_dir")
                    return result
                handoff_completed = True
                accepted_path.unlink()
                emit("roles_starting", "running")
            deadline = time.monotonic() + float(timeout)
            last_progress = 0.0
            baseline_version = baseline.get("policy_version")
            baseline_updates = baseline.get("completed_updates")
            while time.monotonic() < deadline:
                if process is not None and process.poll() is not None:
                    from .instance_failure import failure_diagnosis
                    result.update(reason_code="owned_supervisor_exited", field="state_dir.active_manifest",
                                  diagnosis=failure_diagnosis(supervisor, supervisor.status()))
                    emit("roles_starting", "failed")
                    return result
                current = supervisor.status()
                if current["status"] in {"failed", "orphaned", "stop_incomplete"}:
                    from .instance_failure import failure_diagnosis
                    result.update(reason_code="owned_role_failed", field="state_dir.active_manifest",
                                  diagnosis=failure_diagnosis(supervisor, current))
                    return result
                try:
                    observed = _observe(supervisor, plan)
                except (OSError, ValueError, RuntimeError):
                    observed = {
                        "ready": False, "instance_status": current["status"],
                        "teacher_provider": _teacher_provider_pending(plan),
                    }
                if baseline_version is None and type(observed.get("policy_version")) is int:
                    baseline_version = observed["policy_version"]
                if baseline_updates is None and type(observed.get("completed_updates")) is int:
                    baseline_updates = observed["completed_updates"]
                update_seen = ((type(baseline_version) is int and type(observed.get("policy_version")) is int
                                and observed["policy_version"] > baseline_version)
                               or (type(baseline_updates) is int and type(observed.get("completed_updates")) is int
                                   and observed["completed_updates"] > baseline_updates))
                if observed.get("ready") is True and (wait == "ready" or update_seen):
                    result.update(status="passed", observation=observed, optimizer_update_observed=update_seen)
                    emit("ready" if wait == "ready" else "optimizer_progress", "passed")
                    stop_hold = _read_private_json(supervisor.deployment.state_dir / "stop-hold.json")
                    stop_operation = _read_private_json(supervisor.deployment.state_dir / "stop-operation.json")
                    pending_boundary_stop = (
                        stop_hold is not None and stop_operation is not None
                        and stop_hold.get("operation_id") == stop_operation.get("operation_id")
                        and stop_operation.get("mode") == "after_checkpoint"
                        and (
                            stop_operation.get("manager_status") in {
                                "requested", "accepted", "waiting_for_checkpoint", "retrying",
                                "learner_completed", "pending",
                            }
                            or (stop_operation.get("manager_status") == "blocked"
                                and (stop_operation.get("learner_receipt") or {}).get("request_accepted") is True)
                        )
                    )
                    if stop_hold is not None and not pending_boundary_stop:
                        from .instance_migration import assert_not_migrating
                        from .instance_stop_control import clear_stop_hold_locked
                        assert_not_migrating(supervisor.deployment)
                        clear_stop_hold_locked(supervisor)
                    _write_private_json(supervisor.deployment.state_dir / "resume-operation.json", result)
                    return result
                if observed.get("ready") is True and wait == "update" and (
                    not observed.get("run_id") or observed.get("run_status") in {"succeeded", "stopped"}
                ):
                    result.update(reason_code="no_pending_training_updates", field="logical_run", observation=observed)
                    return result
                if observed.get("run_status") in {"failed", "cancelled"}:
                    result.update(reason_code="resumed_run_failed", field="logical_run", observation=observed)
                    return result
                if time.monotonic() - last_progress >= 5:
                    emit("waiting_for_" + wait, "running", instance_status=observed.get("instance_status"),
                         policy_version=observed.get("policy_version"),
                         teacher_provider_status=(observed.get("teacher_provider") or {}).get("status"),
                         teacher_provider_reason=(observed.get("teacher_provider") or {}).get("reason_code"))
                    last_progress = time.monotonic()
                time.sleep(0.25)
            result.update(status="pending", reason_code="observation_timeout", field="readiness")
            emit("waiting_for_" + wait, "pending")
            _write_private_json(supervisor.deployment.state_dir / "resume-operation.json", result)
            return result
    except BlockingIOError:
        result.update(reason_code="operation_already_running", field="state_dir.operation_lock")
    except Exception as exc:
        result.update(reason_code="resume_stage_failed", error_type=type(exc).__name__)
        from .instance_failure import source_failure_locations
        result["failure_locations"] = source_failure_locations(exc)
        report = getattr(exc, "report", None)
        if isinstance(report, dict):
            result["detail"] = report
    finally:
        if process is not None and not handoff_completed:
            # This is the exact child launched above. Before a valid ACK there
            # is no permission to leave it starting roles after our lock exits.
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=30)
                result["startup_process_cleanup"] = "stopped"
            except (OSError, subprocess.TimeoutExpired):
                result["startup_process_cleanup"] = "stop_incomplete"
        if handoff_path is not None:
            handoff_path.unlink(missing_ok=True)
            Path(str(handoff_path) + ".accepted").unlink(missing_ok=True)
    return result
