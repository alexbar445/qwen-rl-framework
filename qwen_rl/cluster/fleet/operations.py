"""Resumable deployment recipes over existing Fleet, SSH and artifact APIs.

The public Fleet CLI launches this worker in the background. It never starts
training; recorded success describes the recipe, not current cluster health.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from ..identity import atomic_write
from .operation_jobs import operation_worker, write_progress
from .operation_plan import (
    AssignStep,
    ExecStep,
    ExportImageStep,
    OperationPlan,
    OperationPlanError,
    TransferStep,
    UpgradeStep,
    WaitRoleStep,
    load_plan,
)
from .registry import FleetRegistry
from .settings import FleetSettings


class OperationBlocked(RuntimeError):
    def __init__(self, code: str, action: str, nodes: list[str] | None = None):
        super().__init__(code)
        self.code, self.action, self.nodes = code, action, nodes or []


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(), mode=0o600)


def _read(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600):
        raise OperationPlanError("operation_receipt_unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise OperationPlanError("operation_receipt_invalid")
    return value


def _registry(settings: FleetSettings) -> FleetRegistry:
    from .cli import _registry as configured_registry
    return configured_registry(settings)


def _resolve_targets(plan: OperationPlan, registry: FleetRegistry) -> dict[str, Any]:
    inventory = registry.list_nodes()
    nodes = {}
    for target in plan.targets:
        matches = [node for node in inventory if target.node in {node.node_id, node.hostname}]
        if len(matches) != 1:
            raise OperationBlocked("node_not_unique", "correct_target", [target.node])
        node = matches[0]
        if node.approval != "approved":
            raise OperationBlocked("node_not_approved", "approve_node", [target.node])
        if any(role not in {None, target.role} for role in (node.desired_role, node.observed_role)):
            raise OperationBlocked("node_role_conflict", "review_role_assignment", [target.node])
        nodes[target.node] = node
    pending = [name for name, node in nodes.items() if node.heartbeat_stale]
    if pending:
        raise OperationBlocked("nodes_offline", "restore_nodes_then_resume", pending)
    return nodes


@contextmanager
def _resources(settings: FleetSettings, plan: OperationPlan, nodes: dict[str, Any]):
    """No two operation workers can mutate the same site or registered node."""
    directory = Path(settings.state_dir) / "site-operations" / "resource-locks"
    if not directory.exists():
        directory.mkdir(mode=0o700)
    metadata = directory.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700):
        raise OperationPlanError("operation_lock_directory_unsafe")
    names = sorted({"site-" + plan.site_id, *("node-" + node.node_id for node in nodes.values())})
    with ExitStack() as stack:
        for name in names:
            fd = os.open(directory / name, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            stack.enter_context(os.fdopen(fd, "r+b"))
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise OperationPlanError("operation_lock_unsafe")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise OperationBlocked("resource_busy", "resume_when_operation_finishes") from exc
        yield


def _fence(directory: Path, nodes: dict[str, Any]) -> None:
    current = {name: {"node_id": node.node_id, "boot_id": node.boot_id,
                      "account": node.account.model_dump(), "ssh_fingerprint": node.ssh_host_fingerprint}
               for name, node in nodes.items()}
    path = directory / "targets.json"
    if path.exists():
        if _read(path) != current:
            raise OperationBlocked("node_identity_changed", "start_new_operation_after_review")
    else:
        _write(path, current)


def _generations(directory: Path, nodes: dict[str, Any]) -> dict[str, int]:
    path = directory / "generations.json"
    if path.exists():
        return _read(path)
    value = {name: node.desired_generation for name, node in nodes.items()}
    _write(path, value)
    return value


def _subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _subset(actual[key], value)
                                                for key, value in expected.items())
    return type(actual) is type(expected) and actual == expected


def _private_key_boundary(step: TransferStep) -> None:
    """Only a matching TLS leaf key may cross the explicitly permitted boundary."""
    for item in step.files:
        source = Path(item.source)
        if not source.exists():
            raise OperationBlocked("input_missing", "supply_inputs_then_resume")
        with source.open("rb") as stream:
            header = stream.read(4096)
        if b"PRIVATE KEY-----" not in header:
            continue
        if step.artifact_scope != "site_private" or step.tls_leaf_certificate is None:
            raise OperationPlanError("private_key_scope_invalid")
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        certificate = x509.load_pem_x509_certificate(Path(step.tls_leaf_certificate).read_bytes())
        try:
            if certificate.extensions.get_extension_for_class(x509.BasicConstraints).value.ca:
                raise OperationPlanError("ca_private_key_forbidden")
        except x509.ExtensionNotFound as exc:
            raise OperationPlanError("tls_leaf_constraints_missing") from exc
        metadata = source.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise OperationPlanError("tls_leaf_key_unsafe")
        key = serialization.load_pem_private_key(source.read_bytes(), password=None)
        fmt, kind = serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        if key.public_key().public_bytes(fmt, kind) != certificate.public_key().public_bytes(fmt, kind):
            raise OperationPlanError("tls_leaf_identity_mismatch")


def _execute_local(step: ExecStep, directory: Path, receipt_path: Path) -> dict[str, Any]:
    # Local producers have no remote supervisor to reconcile lost side effects.
    if receipt_path.exists() and _read(receipt_path).get("state") == "started":
        raise OperationBlocked("local_step_unconfirmed", "reconcile_local_step_before_new_operation")
    _write(receipt_path, {"state": "started", "step_sha256": _fingerprint(step.model_dump(mode="json"))})
    log_path = directory / "receipts" / (step.id + ".log")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    error_fd = os.open(directory / "receipts" / (step.id + ".stderr.log"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as log, os.fdopen(error_fd, "wb") as errors:
        process = subprocess.Popen(step.argv, cwd=step.cwd, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=errors, start_new_session=True)
        from .operation_jobs import _proc
        starttime = _proc(process.pid)
        _write(receipt_path, {"state": "started", "step_sha256": _fingerprint(step.model_dump(mode="json")),
                              "pid": process.pid, "proc_starttime": starttime})
        try:
            process.wait(timeout=step.timeout_seconds)
        except subprocess.TimeoutExpired:
            # Popen created a new session: this group contains only this step's children.
            members = []
            if process.poll() is None and _proc(process.pid) == starttime and os.getpgid(process.pid) == process.pid:
                for entry in Path("/proc").iterdir():
                    if not entry.name.isdigit():
                        continue
                    try:
                        raw = (entry / "stat").read_text()
                        fields = raw[raw.rfind(")") + 2:].split()
                        if int(fields[2]) == process.pid and int(fields[3]) == process.pid:
                            members.append((int(entry.name), fields[19]))
                    except (OSError, ValueError, IndexError):
                        continue
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if _proc(process.pid) == starttime and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            for pid, birth in members:
                if _proc(pid) == birth:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            raise OperationBlocked("local_step_timeout", "reconcile_local_step_before_new_operation") from None
    if process.returncode != 0:
        raise OperationBlocked("local_command_failed", "inspect_private_step_log")
    result = {"exit_code": 0}
    if step.expected_json is not None:
        if log_path.stat().st_size > 65536:
            raise OperationPlanError("command_json_too_large")
        value = json.loads(log_path.read_text())
        if not _subset(value, step.expected_json):
            raise OperationPlanError("command_readiness_failed")
        result["result_json"] = value
    return result


def _export_image(step: ExportImageStep, node: Any, settings: FleetSettings, operation_id: str,
                  directory: Path, progress) -> dict[str, Any]:
    from .remote_operation import execute_remote
    from .transfer import TransferFile, transfer_files

    source = Path(__file__).resolve().parents[3]
    names = ["qwen_rl/__init__.py", "qwen_rl/cluster/__init__.py",
             "qwen_rl/cluster/artifacts.py", "qwen_rl/cluster/images.py"]
    generated = directory / ("export-code-" + step.id)
    if not generated.exists():
        generated.mkdir(mode=0o700)
    helper = generated / "export_image.py"
    program = (
        "import json,os,sys\nfrom pathlib import Path\nos.umask(0o077)\n"
        "from qwen_rl.cluster.images import export_image_archive\n"
        "item=export_image_archive(sys.argv[1],sys.argv[2],artifact_base_url=sys.argv[3])\n"
        "print(json.dumps({'image_id':item.descriptor.image_id,'archive_sha256':item.descriptor.archive_sha256,"
        "'size_bytes':item.descriptor.size_bytes,'path':str(item.path),'published':False}))\n"
    )
    if helper.exists() and helper.read_text() != program:
        raise OperationPlanError("export_helper_changed")
    if not helper.exists():
        atomic_write(helper, program.encode(), mode=0o600)
    files = [TransferFile(source / name, name, (source / name).stat().st_size, _sha(source / name)) for name in names]
    files.append(TransferFile(helper, "export_image.py", helper.stat().st_size, _sha(helper)))
    code_sha = _fingerprint([(item.remote_path, item.sha256) for item in files])
    remote_dir = ".local/share/qwen-fleet-operations/export-code/" + code_sha
    transfer_files(node, settings, remote_dir, files, operation_id + "-export", progress=progress,
                   timeout_seconds=step.timeout_seconds)
    root = Path(node.account.home) / remote_dir
    executable = step.python_executable or str(Path(node.account.home) / ".local/share/qwen-fleet/management/.venv/bin/python")
    argv = ["env", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=" + str(root),
            "DOCKER_CONTEXT=" + step.docker_context, executable, "-B", str(root / "export_image.py"),
            step.image_id, str(Path(node.account.home) / step.content_dir), step.artifact_base_url]
    return execute_remote(node, settings, operation_id, step.id, argv, timeout_seconds=step.timeout_seconds,
                          expected_json={"image_id": step.image_id, "published": False}, progress=progress)


def _execute(step, node, settings, registry, directory, operation_id, receipts, progress):
    if isinstance(step, TransferStep):
        from .transfer import TransferFile, transfer_files
        _private_key_boundary(step)
        inputs = [TransferFile(Path(item.source), item.remote_path, item.size, item.sha256) for item in step.files]
        transfer_id = operation_id + "-" + hashlib.sha256(step.id.encode()).hexdigest()[:12]
        return transfer_files(node, settings, step.remote_dir, inputs, transfer_id, progress=progress,
                              timeout_seconds=step.timeout_seconds)
    if isinstance(step, ExecStep):
        if node is None:
            return _execute_local(step, directory, directory / "receipts" / (step.id + ".json"))
        from .remote_operation import execute_remote
        return execute_remote(node, settings, operation_id, step.id, step.argv,
                              timeout_seconds=step.timeout_seconds, expected_json=step.expected_json, progress=progress)
    if isinstance(step, ExportImageStep):
        return _export_image(step, node, settings, operation_id, directory, progress)
    if isinstance(step, UpgradeStep):
        from .upgrade import upgrade_nodes
        result = upgrade_nodes(registry, settings, [node.node_id])
        if result["nodes"][0]["status"] not in {"upgraded", "already_current"}:
            raise OperationBlocked("management_upgrade_pending", "inspect_upgrade_receipt_then_resume")
        return result
    if isinstance(step, AssignStep):
        from .cli import _read_private_assignment
        if not Path(step.deployment_file).exists():
            raise OperationBlocked("deployment_input_missing", "build_role_document_then_resume")
        deployment = _read_private_assignment(Path(step.deployment_file), public_upload_dir=settings.public_upload_dir)
        declared = receipts["roles"][step.node]
        if deployment.role != declared:
            raise OperationPlanError("deployment_role_conflict")
        try:
            return registry.assign_node(node.node_id, deployment,
                                        expected_generation=receipts["generations"][step.node]).model_dump(mode="json")
        except Exception as exc:
            if getattr(exc, "code", None) == "generation_conflict":
                raise OperationBlocked("node_generation_changed", "review_role_assignment", [step.node]) from None
            raise
    if isinstance(step, WaitRoleStep):
        assignment = receipts[step.assignment_step]["result"]
        deadline = time.monotonic() + step.timeout_seconds
        while time.monotonic() < deadline:
            view = registry.get_node(node.node_id)
            if view.heartbeat_stale:
                raise OperationBlocked("node_offline", "restore_nodes_then_resume", [step.node])
            if (view.desired_generation != assignment["generation"]
                    or view.deployment_sha256 != assignment["deployment_sha256"]):
                raise OperationBlocked("assignment_changed", "review_role_assignment", [step.node])
            if (view.observed_generation == assignment["generation"] and view.observed_role == assignment["role"]
                    and view.observed_state == "ready" and _role_ready(view.observed_role, view.observed_receipt)):
                return {"node_id": view.node_id, "generation": view.observed_generation,
                        "role": view.observed_role, "deployment_sha256": view.deployment_sha256,
                        "readiness_verified": True}
            if view.observed_state in {"failed", "needs_driver", "needs_reboot", "needs_host_packages"}:
                raise OperationBlocked("role_not_ready", "inspect_node_then_resume", [step.node])
            progress({"status": "waiting"})
            time.sleep(5)
        raise OperationBlocked("role_readiness_timeout", "inspect_node_then_resume", [step.node])
    raise OperationPlanError("unsupported_operation_step")


def _role_ready(role: str, receipt: Any) -> bool:
    if not isinstance(receipt, dict) or receipt.get("readiness_verified") is not True:
        return False
    if role == "worker":
        health = receipt.get("worker_health")
        return (all(receipt.get(key) is True for key in ("worker_ready", "model_verified", "image_verified"))
                and isinstance(health, dict) and health.get("ready") is True)
    return (role == "learner" and receipt.get("learner_runtime_ready") is True
            and receipt.get("learner_training_idle") is True)


def run_operation(config_path: Path, operation_id: str) -> int:
    settings = FleetSettings.load(config_path)
    completed: list[str] = []
    phase = "preflight"
    with operation_worker(settings, operation_id) as directory:
        try:
            plan = load_plan(directory / "plan.json")
            saved = _read(directory / "status.json")
            if saved["plan_sha256"] != _sha(directory / "plan.json"):
                raise OperationPlanError("operation_plan_changed")
            registry = _registry(settings)
            write_progress(settings, operation_id, status="running", phase=phase, total=len(plan.steps),
                           reason_code=None, next_action=None, pending_nodes=[], scope=plan.scope)
            nodes = _resolve_targets(plan, registry)
            with _resources(settings, plan, nodes):
                _fence(directory, nodes)
                receipts: dict[str, Any] = {"roles": {target.node: target.role for target in plan.targets},
                                            "generations": _generations(directory, nodes)}
                for step in plan.ordered_steps():
                    phase = step.id
                    path = directory / "receipts" / (step.id + ".json")
                    fingerprint = _fingerprint(step.model_dump(mode="json"))
                    if path.exists():
                        receipt = _read(path)
                        if receipt.get("step_sha256") != fingerprint:
                            raise OperationPlanError("operation_step_changed")
                        if receipt.get("state") == "completed":
                            receipts[step.id] = receipt
                            if isinstance(step, AssignStep):
                                receipts["generations"][step.node] = receipt["result"]["generation"]
                            completed.append(step.id)
                            continue
                    current = _resolve_targets(plan, registry)
                    _fence(directory, current)
                    changed = [name for name, value in current.items()
                               if value.desired_generation != receipts["generations"][name]
                               and not (isinstance(step, AssignStep) and step.node == name)]
                    if changed:
                        raise OperationBlocked("node_generation_changed", "review_role_assignment", changed)
                    node = current.get(getattr(step, "node", None))
                    write_progress(settings, operation_id, status="running", phase=phase,
                                   completed_steps=completed, done=len(completed), node=getattr(step, "node", None), progress={})

                    def progress(value, step_id=step.id):
                        allowed = {key: value[key] for key in ("bytes_done", "bytes_total", "items_done", "items_total")
                                   if key in value and type(value[key]) is int}
                        write_progress(settings, operation_id, progress={**allowed, "current_step": step_id})

                    result = _execute(step, node, settings, registry, directory, operation_id, receipts, progress)
                    receipt = {"state": "completed", "step_sha256": fingerprint, "result": result}
                    _write(path, receipt)
                    receipts[step.id] = receipt
                    if isinstance(step, AssignStep):
                        receipts["generations"][step.node] = result["generation"]
                    elif isinstance(step, UpgradeStep):
                        view = registry.get_node(node.node_id)
                        if view.desired_role is not None:
                            raise OperationBlocked("node_generation_changed", "review_role_assignment", [step.node])
                        receipts["generations"][step.node] = view.desired_generation
                    _write(directory / "generations.json", receipts["generations"])
                    completed.append(step.id)
                    events = directory / "events.jsonl"
                    with events.open("a") as stream:
                        stream.write(json.dumps({"event": "step_completed", "step": step.id, "time": time.time()}) + "\n")
                write_progress(settings, operation_id, status="succeeded", phase="complete", completed_steps=completed,
                               done=len(completed), total=len(plan.steps), progress={}, reason_code=None,
                               next_action=None, pending_nodes=[], node=None)
            return 0
        except OperationBlocked as exc:
            write_progress(settings, operation_id, status="blocked", phase=phase, completed_steps=completed,
                           done=len(completed), reason_code=exc.code, next_action=exc.action, pending_nodes=exc.nodes)
            return 2
        except Exception as exc:  # noqa: BLE001 - persist safe terminal state, never nested credential-bearing errors.
            # Exceptions may carry credentials from nested libraries. Only stable codes escape.
            code = "operation_plan_invalid" if isinstance(exc, OperationPlanError) else "operation_step_failed"
            if type(exc).__name__ == "FleetRemoteOperationError" and getattr(exc, "code", None) in {"blocked", "interrupted"}:
                write_progress(settings, operation_id, status="blocked", phase=phase, completed_steps=completed,
                               done=len(completed), reason_code="remote_step_unconfirmed", next_action="inspect_remote_receipt_then_resume")
                return 2
            write_progress(settings, operation_id, status="failed", phase=phase, completed_steps=completed,
                           done=len(completed), reason_code=code, next_action="inspect_private_step_receipt")
            return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--operation-id", required=True)
    args = parser.parse_args(argv)
    return run_operation(args.config, args.operation_id)


if __name__ == "__main__":
    raise SystemExit(main())
