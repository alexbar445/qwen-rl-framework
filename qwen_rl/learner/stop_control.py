"""One persisted stop-after-checkpoint request for a managed Learner."""
from __future__ import annotations

import copy
import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

StopStatus = Literal["accepted", "waiting_durable", "completed", "blocked"]

def _read_control(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("stop control receipt is not a regular file")
        if info.st_mode & 0o077:
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
            value = json.load(stream)
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise ValueError("stop control receipt must be a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_control(path: Path, value: dict[str, Any]) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent.is_symlink() or not parent.is_dir() or path.is_symlink():
        raise RuntimeError("stop control path is not a safe regular-file location")
    os.chmod(parent, 0o700)
    temporary = parent / f".{path.name}.{uuid4().hex}.tmp"
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(parent)
    finally:
        os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def wait_for_engine_durable(engine: Any) -> None:
    """Use the engine/Core durability fence without starting a new checkpoint."""
    wait = getattr(engine, "wait_for_durable", None)
    if not callable(wait):
        wait = getattr(getattr(engine, "core", None), "wait_for_durable", None)
    if not callable(wait):
        raise RuntimeError("Learner engine has no durable-checkpoint wait fence")
    wait()


class StopAfterCheckpointControl:
    """Serialize one operation and its optimizer/checkpoint boundary receipt."""

    SCHEMA = "qwen-rl-stop-after-checkpoint/v1"
    RECEIPT_FIELDS = frozenset({
        "schema", "operation_id", "mode", "status", "run_id", "lineage_id", "engine_epoch",
        "target_update", "target_policy_version", "completed_update", "durable_policy_version",
        "checkpoint_snapshot_id", "transaction_status", "bepa_attempt_id", "bepa_commit_state",
        "boundary_committed", "request_accepted", "created_at", "updated_at", "error",
    })

    def __init__(self, path: str | Path, *, lineage_id: str | None, engine_epoch: str | None) -> None:
        self.path = Path(path)
        self.lineage_id = lineage_id
        self.engine_epoch = engine_epoch
        self._lock = threading.RLock()
        self._receipt: dict[str, Any] | None = None
        self._active_update: dict[str, Any] | None = None
        self._blocks_new_work = False
        if self.path.exists() or self.path.is_symlink():
            if self.path.is_symlink():
                raise ValueError("stop control receipt cannot be a symlink")
            stored = _read_control(self.path)
            self._validate(stored)
            self._receipt = stored
            self._blocks_new_work = stored.get("status") in {"accepted", "waiting_durable"} or (
                stored.get("status") == "blocked" and stored.get("request_accepted") is True
            )
            if (stored.get("lineage_id") != lineage_id
                    or stored.get("engine_epoch") != engine_epoch):
                if stored.get("status") in {"accepted", "waiting_durable"}:
                    self._blocked_locked(
                        "Learner recovery lineage changed while stop request was pending",
                        "StopControlIdentityError",
                    )

    @staticmethod
    def _validate(value: dict[str, Any]) -> None:
        if set(value) != StopAfterCheckpointControl.RECEIPT_FIELDS:
            raise ValueError("persisted stop control has invalid receipt fields")
        if value.get("schema") != StopAfterCheckpointControl.SCHEMA:
            raise ValueError("unsupported persisted stop control schema")
        if type(value.get("request_accepted")) is not bool:
            raise ValueError("persisted stop control has no request acceptance proof")
        if (not isinstance(value.get("operation_id"), str)
                or not value["operation_id"]
                or value.get("mode") != "after_checkpoint"
                or value.get("status") not in {"accepted", "waiting_durable", "completed", "blocked"}):
            raise ValueError("persisted stop control has invalid operation identity")
        for key in ("target_update", "target_policy_version", "completed_update", "durable_policy_version"):
            item = value.get(key)
            if item is not None and (type(item) is not int or item < 0):
                raise ValueError(f"persisted stop control has invalid {key}")
        if value.get("status") in {"accepted", "waiting_durable", "completed"}:
            if type(value.get("target_update")) is not int or type(value.get("target_policy_version")) is not int:
                raise ValueError("persisted stop control is missing its bound target")
        if value.get("run_id") is not None and (
                not isinstance(value.get("run_id"), str) or not value["run_id"]):
            raise ValueError("persisted stop control has an invalid run_id")
        for key in ("lineage_id", "engine_epoch"):
            item = value.get(key)
            if item is not None and (not isinstance(item, str) or not item):
                raise ValueError(f"persisted stop control has an invalid {key}")

    def _persist_locked(self) -> None:
        if self._receipt is not None:
            self._receipt["updated_at"] = time.time()
            _write_control(self.path, self._receipt)

    def _blocked_locked(self, message: str, error_type: str = "StopControlError") -> None:
        assert self._receipt is not None
        self._blocks_new_work = self._receipt.get("request_accepted") is True
        self._receipt.update(
            status="blocked",
            boundary_committed=False,
            error={"type": error_type, "message": str(message)[:300]},
        )
        self._persist_locked()

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._receipt)

    def get(self, operation_id: str) -> dict[str, Any] | None:
        with self._lock:
            if self._receipt is None or self._receipt.get("operation_id") != operation_id:
                return None
            return copy.deepcopy(self._receipt)

    @property
    def has_pending_request(self) -> bool:
        with self._lock:
            return self._receipt is not None and self._receipt.get("status") in {
                "accepted", "waiting_durable",
            }

    @property
    def blocks_new_work(self) -> bool:
        with self._lock:
            return self._blocks_new_work

    def begin_update(self, *, run_id: str, update: int, policy_version: int) -> None:
        """Mark the claimed update before the coordinator yields to any work."""
        if (not run_id or type(update) is not int or update < 1
                or type(policy_version) is not int or policy_version < 1):
            raise ValueError("in-flight update identity is invalid")
        with self._lock:
            self._active_update = {
                "run_id": run_id,
                "update": update,
                "policy_version": policy_version,
            }
            receipt = self._receipt
            if (receipt is not None and receipt.get("status") in {"accepted", "waiting_durable"}
                    and receipt.get("run_id") == run_id
                    and receipt.get("target_update") == update
                    and receipt.get("target_policy_version") == policy_version):
                receipt["transaction_status"] = "in_progress"
                self._persist_locked()

    def end_update(self, *, run_id: str, update: int) -> None:
        with self._lock:
            if (self._active_update is not None
                    and self._active_update.get("run_id") == run_id
                    and self._active_update.get("update") == update):
                self._active_update = None

    def request(
        self,
        operation_id: str,
        *,
        run_id: str | None,
        completed_update: int,
        requested_steps: int | None,
        current_policy_version: int,
        recovery: dict[str, Any] | None,
        bepa: Any | None,
        supported: bool,
    ) -> tuple[dict[str, Any], bool, bool]:
        """Return (receipt, close_current_stream, needs_idle_wait)."""
        if not operation_id:
            raise ValueError("operation_id must be non-empty")
        with self._lock:
            current = self._receipt
            if current is not None and current.get("operation_id") == operation_id:
                return copy.deepcopy(current), self._is_targeted_update_locked(), current.get("status") == "waiting_durable" and current.get("run_id") is None
            if current is not None and (current.get("status") in {"accepted", "waiting_durable"}
                    or (current.get("status") == "blocked" and current.get("request_accepted") is True)):
                raise ValueError("another stop-after-checkpoint request is still pending")

            durable = recovery.get("durable_policy_version") if isinstance(recovery, dict) else None
            live = recovery.get("policy_version") if isinstance(recovery, dict) else current_policy_version
            snapshot_id = recovery.get("snapshot_id") if isinstance(recovery, dict) else None
            lineage_id = recovery.get("lineage_id") if isinstance(recovery, dict) else self.lineage_id
            engine_epoch = recovery.get("engine_epoch") if isinstance(recovery, dict) else self.engine_epoch
            now = time.time()
            receipt = {
                "schema": self.SCHEMA,
                "operation_id": operation_id,
                "mode": "after_checkpoint",
                "status": "accepted",
                "request_accepted": False,
                "run_id": run_id,
                "lineage_id": lineage_id,
                "engine_epoch": engine_epoch,
                "target_update": None,
                "target_policy_version": None,
                "completed_update": completed_update,
                "durable_policy_version": durable,
                "checkpoint_snapshot_id": snapshot_id,
                "transaction_status": "not_started",
                "bepa_attempt_id": None,
                "bepa_commit_state": "not_applicable",
                "boundary_committed": False,
                "created_at": now,
                "updated_at": now,
                "error": None,
            }
            self._receipt = receipt
            self._blocks_new_work = True
            close_stream = False
            needs_idle_wait = False
            reason = None
            if not supported:
                reason = "stop-after-checkpoint is supported only by managed central_gradient Learners"
            elif (not isinstance(recovery, dict) or recovery.get("enabled") is not True
                    or recovery.get("dirty") is True or type(durable) is not int
                    or type(live) is not int or not isinstance(snapshot_id, str) or not snapshot_id):
                reason = "a clean full-state checkpoint is unavailable"
            elif recovery.get("checkpoint_state") in {"faulted", "recovery_required"}:
                reason = "Learner checkpoint state is faulted and cannot satisfy a stop boundary"
            elif lineage_id != self.lineage_id or engine_epoch != self.engine_epoch:
                reason = "Learner recovery identity differs from the stop control owner"
            elif run_id is not None and self._active_update is not None:
                active = self._active_update
                receipt.update(
                    run_id=active["run_id"], target_update=active["update"],
                    target_policy_version=active["policy_version"],
                    transaction_status="in_progress",
                )
                close_stream = True
            elif run_id is not None:
                if type(requested_steps) is int and completed_update >= requested_steps:
                    if live != durable or recovery.get("dirty") is True:
                        reason = "completed run does not have a clean current checkpoint"
                    else:
                        context = recovery.get("update_context")
                        context = context if isinstance(context, dict) else {}
                        attempt_id = context.get("bepa_attempt_id")
                        if attempt_id is not None:
                            store = getattr(bepa, "store", None)
                            attempt = None if store is None else store.get_update_attempt(attempt_id)
                            if attempt is None or getattr(attempt, "state", None) != "committed":
                                reason = "completed run checkpoint has no committed BEPA attempt proof"
                            else:
                                receipt["bepa_attempt_id"] = attempt_id
                                receipt["bepa_commit_state"] = "committed"
                        receipt.update(
                            target_update=completed_update,
                            target_policy_version=current_policy_version,
                            transaction_status="already_durable",
                            boundary_committed=reason is None,
                        )
                        close_stream = True
                else:
                    receipt.update(
                        target_update=completed_update + 1,
                        target_policy_version=current_policy_version + 1,
                        transaction_status="not_started",
                    )
            elif (recovery.get("checkpoint_state") == "saving" and type(live) is int
                    and type(durable) is int and live == durable + 1):
                logical = ((recovery.get("update_context") or {}).get("logical_job")
                           if isinstance(recovery.get("update_context"), dict) else None)
                receipt.update(
                    target_update=(logical.get("completed_updates", completed_update) + 1
                                   if isinstance(logical, dict)
                                   and type(logical.get("completed_updates", completed_update)) is int
                                   else completed_update + 1),
                    target_policy_version=live,
                    status="waiting_durable",
                    transaction_status="in_progress",
                )
                needs_idle_wait = True
            elif (type(live) is int and live == durable and recovery.get("dirty") is False
                    and recovery.get("checkpoint_state") in {None, "durable"}):
                context = recovery.get("update_context")
                context = context if isinstance(context, dict) else {}
                attempt_id = context.get("bepa_attempt_id")
                if attempt_id is not None:
                    store = getattr(bepa, "store", None)
                    attempt = None if store is None else store.get_update_attempt(attempt_id)
                    if attempt is None or getattr(attempt, "state", None) != "committed":
                        reason = "current checkpoint has no committed BEPA attempt proof"
                    else:
                        receipt["bepa_attempt_id"] = attempt_id
                        receipt["bepa_commit_state"] = "committed"
                receipt.update(
                    target_update=completed_update,
                    target_policy_version=live,
                    transaction_status="already_durable",
                    boundary_committed=True,
                    status="completed",
                )
            else:
                reason = "current Learner state is not at a provable durable checkpoint boundary"
            if reason is not None:
                self._blocked_locked(reason)
            else:
                receipt["request_accepted"] = True
                self._persist_locked()
            return copy.deepcopy(receipt), close_stream, needs_idle_wait

    def _is_targeted_update_locked(self) -> bool:
        receipt = self._receipt
        return bool(
            receipt is not None and receipt.get("status") in {"accepted", "waiting_durable"}
            and self._active_update is not None
            and receipt.get("run_id") == self._active_update.get("run_id")
            and receipt.get("target_update") == self._active_update.get("update")
        )

    def targets_update(self, *, run_id: str, update: int) -> bool:
        with self._lock:
            receipt = self._receipt
            return bool(
                receipt is not None and receipt.get("status") in {"accepted", "waiting_durable"}
                and receipt.get("run_id") == run_id and receipt.get("target_update") == update
            )

    def targets_current_checkpoint(self, *, run_id: str, update: int, policy_version: int) -> bool:
        with self._lock:
            receipt = self._receipt
            return bool(
                receipt is not None and receipt.get("status") in {"accepted", "waiting_durable"}
                and receipt.get("run_id") == run_id
                and receipt.get("target_update") == update
                and receipt.get("target_policy_version") == policy_version
                and receipt.get("transaction_status") == "already_durable"
                and receipt.get("boundary_committed") is True
            )

    def should_skip_overlap(self, policy_version: int) -> bool:
        with self._lock:
            receipt = self._receipt
            return bool(
                receipt is not None and receipt.get("status") in {"accepted", "waiting_durable"}
                and type(receipt.get("target_policy_version")) is int
                and receipt["target_policy_version"] <= policy_version
            )

    def mark_bepa_pending(self, operation_id: str) -> None:
        with self._lock:
            receipt = self._receipt
            if (receipt is not None and receipt.get("operation_id") == operation_id
                    and receipt.get("status") in {"accepted", "waiting_durable"}):
                receipt["bepa_commit_state"] = "pending"
                self._persist_locked()

    def register_bepa_attempt(self, *, run_id: str, update: int, attempt_id: str) -> None:
        with self._lock:
            receipt = self._receipt
            if (receipt is not None and receipt.get("status") in {"accepted", "waiting_durable"}
                    and receipt.get("run_id") == run_id
                    and receipt.get("target_update") == update):
                receipt.update(bepa_attempt_id=attempt_id, bepa_commit_state="prepared")
                self._persist_locked()

    def refresh_bepa_state(self, bepa: Any | None) -> dict[str, Any] | None:
        with self._lock:
            receipt = self._receipt
            if receipt is None:
                return None
            attempt_id = receipt.get("bepa_attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                return copy.deepcopy(receipt)
            store = getattr(bepa, "store", None)
            attempt = None if store is None else store.get_update_attempt(attempt_id)
            state = getattr(attempt, "state", None)
            mapped = state if state in {
                "prepared", "optimizer_entered", "committed", "commit_unknown", "released",
            } else "unknown"
            if receipt.get("bepa_commit_state") != mapped:
                receipt["bepa_commit_state"] = mapped
                self._persist_locked()
            return copy.deepcopy(receipt)

    def update_committed(
        self,
        *,
        run_id: str,
        update: int,
        policy_version: int,
        recovery: dict[str, Any],
        bepa_attempt_id: str | None,
        bepa_commit_state: str,
    ) -> bool:
        """Record full optimizer/BEPA return and clear the in-flight barrier."""
        with self._lock:
            targeted = self.targets_update(run_id=run_id, update=update)
            receipt = self._receipt
            if targeted and receipt is not None:
                receipt.update(
                    transaction_status="committed",
                    completed_update=update,
                    durable_policy_version=recovery.get("durable_policy_version"),
                    checkpoint_snapshot_id=recovery.get("snapshot_id"),
                    bepa_attempt_id=bepa_attempt_id,
                    bepa_commit_state=bepa_commit_state,
                    target_policy_version=policy_version,
                )
                if (recovery.get("dirty") is True
                        or recovery.get("durable_policy_version") != policy_version):
                    receipt["status"] = "waiting_durable"
                self._persist_locked()
            if (self._active_update is not None
                    and self._active_update.get("run_id") == run_id
                    and self._active_update.get("update") == update):
                self._active_update = None
            return targeted

    def mark_boundary_committed(self, *, recovery: dict[str, Any]) -> None:
        with self._lock:
            receipt = self._receipt
            if receipt is None or receipt.get("status") not in {"accepted", "waiting_durable"}:
                return
            target = receipt.get("target_policy_version")
            durable = recovery.get("durable_policy_version")
            if (receipt.get("transaction_status") != "committed"
                    or receipt.get("bepa_commit_state") not in {"committed", "not_applicable"}
                    or type(target) is not int or type(durable) is not int or durable < target):
                self._blocked_locked("stop boundary lacks complete optimizer, BEPA, or checkpoint proof")
                return
            receipt.update(
                completed_update=receipt.get("target_update"),
                durable_policy_version=durable,
                checkpoint_snapshot_id=recovery.get("snapshot_id"),
                boundary_committed=True,
            )
            self._persist_locked()

    def complete_paused_run(self) -> None:
        with self._lock:
            receipt = self._receipt
            if receipt is None or receipt.get("status") not in {"accepted", "waiting_durable"}:
                return
            if receipt.get("boundary_committed") is not True:
                self._blocked_locked("run ended before the requested durable stop boundary")
                return
            receipt["status"] = "completed"
            receipt["error"] = None
            self._blocks_new_work = True
            self._persist_locked()
    def wait_idle_complete(self, *, recovery: dict[str, Any], bepa: Any | None = None) -> None:
        with self._lock:
            receipt = self._receipt
            if receipt is None or receipt.get("status") not in {"accepted", "waiting_durable"}:
                return
            target = receipt.get("target_policy_version")
            durable = recovery.get("durable_policy_version")
            if (recovery.get("dirty") is True or type(target) is not int
                    or type(durable) is not int or durable < target
                    or recovery.get("snapshot_id") is None):
                self._blocked_locked("durable checkpoint wait ended without the requested snapshot")
                return
            context = recovery.get("update_context")
            context = context if isinstance(context, dict) else {}
            if receipt.get("transaction_status") == "in_progress":
                logical = context.get("logical_job")
                if (context.get("new_policy_version") != target
                        or not isinstance(logical, dict)
                        or logical.get("completed_updates") != receipt.get("target_update")):
                    self._blocked_locked("durable checkpoint does not prove the idle stop target")
                    return
            attempt_id = context.get("bepa_attempt_id")
            if attempt_id is not None:
                store = getattr(bepa, "store", None)
                attempt = None if store is None else store.get_update_attempt(attempt_id)
                if attempt is None or getattr(attempt, "state", None) != "committed":
                    self._blocked_locked("idle checkpoint lacks committed BEPA attempt proof")
                    return
                receipt["bepa_attempt_id"] = attempt_id
                receipt["bepa_commit_state"] = "committed"
            receipt.update(
                status="completed", transaction_status="already_durable",
                durable_policy_version=durable,
                checkpoint_snapshot_id=recovery.get("snapshot_id"),
                boundary_committed=True,
                error=None,
            )
            self._blocks_new_work = True
            self._persist_locked()

    def block(self, message: str, error_type: str = "StopControlError") -> None:
        with self._lock:
            if self._receipt is not None and self._receipt.get("status") in {"accepted", "waiting_durable"}:
                self._blocked_locked(message, error_type)

    def reconcile_startup(self, *, recovery: dict[str, Any] | None, bepa: Any | None) -> None:
        """Reconcile a persisted target after engine and BEPA recovery have loaded."""
        with self._lock:
            receipt = self._receipt
            if receipt is None or receipt.get("status") not in {"accepted", "waiting_durable"}:
                return
            if not isinstance(recovery, dict):
                self._blocked_locked("recovery metadata is unavailable after Learner restart")
                return
            target = receipt.get("target_policy_version")
            durable = recovery.get("durable_policy_version")
            if type(target) is not int or type(durable) is not int:
                self._blocked_locked("persisted stop target has no integer checkpoint identity")
                return
            if durable < target:
                return

            context = recovery.get("update_context")
            context = context if isinstance(context, dict) else {}
            logical = context.get("logical_job")
            expected_update = receipt.get("target_update")
            if (context.get("new_policy_version") != target
                    or not isinstance(logical, dict)
                    or logical.get("completed_updates") != expected_update):
                self._blocked_locked("durable checkpoint does not prove the persisted logical stop target")
                return

            attempt_id = receipt.get("bepa_attempt_id") or context.get("bepa_attempt_id")
            if attempt_id is None:
                attempt_id = context.get("bepa_attempt_id")
            if attempt_id is not None:
                store = getattr(bepa, "store", None)
                attempt = None if store is None else store.get_update_attempt(attempt_id)
                if attempt is None or getattr(attempt, "state", None) != "committed":
                    self._blocked_locked("durable stop target has no committed BEPA attempt proof")
                    return
                receipt["bepa_attempt_id"] = attempt_id
                receipt["bepa_commit_state"] = "committed"
            else:
                receipt["bepa_commit_state"] = "not_applicable"
            receipt.update(
                status="completed",
                transaction_status="committed",
                completed_update=expected_update,
                durable_policy_version=durable,
                checkpoint_snapshot_id=recovery.get("snapshot_id"),
                boundary_committed=True,
                error=None,
            )
            self._blocks_new_work = True
            self._persist_locked()
