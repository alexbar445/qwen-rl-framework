"""Explicit authorization for one missing, checkpoint-bound BEPA attempt."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .contracts import ContractError
from .store import BepaStore

SCHEMA = "qwen-rl-bepa-checkpoint-reset/v1"
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RESET_TABLES = (
    "rescue_jobs", "teacher_claim_requests", "job_mutations", "usage_events",
    "episodes", "cache_entries", "reservations", "update_attempts", "plans",
    "seed_attempts", "unexpected_events",
)


class CheckpointResetReceiptError(ContractError):
    """A checkpoint reset receipt is missing, malformed, or bound elsewhere."""


def _checkpoint_binding(
    recovery: Mapping[str, Any], *, policy_version: int, adapter_sha256: str,
) -> dict[str, Any]:
    if (recovery.get("enabled") is not True or recovery.get("resumed") is not True
            or recovery.get("dirty") is not False
            or recovery.get("durable_policy_version") != policy_version):
        raise CheckpointResetReceiptError("BEPA reset receipt requires a resumed full checkpoint")
    loaded = recovery.get("loaded_snapshot")
    context = recovery.get("update_context")
    if not isinstance(loaded, Mapping) or not isinstance(context, Mapping):
        raise CheckpointResetReceiptError("BEPA reset receipt requires checkpoint and update metadata")
    lineage_id = recovery.get("lineage_id")
    engine_epoch = recovery.get("engine_epoch")
    snapshot_id = loaded.get("snapshot_id")
    checkpoint_generation = loaded.get("checkpoint_generation")
    attempt_id = context.get("bepa_attempt_id")
    batch_id = context.get("bepa_batch_id")
    input_batch_hash = context.get("input_batch_hash")
    logical_job = context.get("logical_job")
    completed_updates = logical_job.get("completed_updates") if isinstance(logical_job, Mapping) else None
    durable_optimizer_steps = recovery.get("durable_optimizer_steps")
    optimizer_steps = recovery.get("optimizer_steps")
    scheduler_steps = recovery.get("scheduler_steps")
    if (not isinstance(lineage_id, str) or _HEX32.fullmatch(lineage_id) is None
            or not isinstance(engine_epoch, str) or _HEX32.fullmatch(engine_epoch) is None
            or not isinstance(snapshot_id, str) or _HEX32.fullmatch(snapshot_id) is None
            or isinstance(checkpoint_generation, bool)
            or not isinstance(checkpoint_generation, int) or checkpoint_generation < 0
            or recovery.get("checkpoint_state") != "durable"
            or loaded.get("policy_version") != policy_version
            or recovery.get("policy_version") != policy_version
            or recovery.get("durable_policy_version") != policy_version
            or type(policy_version) is not int or policy_version < 1
            or isinstance(durable_optimizer_steps, bool)
            or not isinstance(durable_optimizer_steps, int) or durable_optimizer_steps < 0
            or optimizer_steps != durable_optimizer_steps
            or scheduler_steps != durable_optimizer_steps
            or not isinstance(attempt_id, str) or not attempt_id
            or not isinstance(batch_id, str) or not batch_id
            or not isinstance(input_batch_hash, str) or _HEX64.fullmatch(input_batch_hash) is None
            or context.get("new_policy_version") != policy_version
            or isinstance(completed_updates, bool) or not isinstance(completed_updates, int)
            or completed_updates < 1
            or not isinstance(adapter_sha256, str) or _HEX64.fullmatch(adapter_sha256) is None):
        raise CheckpointResetReceiptError("BEPA reset receipt checkpoint binding is invalid")
    metadata_snapshot_id = recovery.get("snapshot_id")
    if metadata_snapshot_id is not None and metadata_snapshot_id != snapshot_id:
        raise CheckpointResetReceiptError("BEPA reset receipt snapshot IDs disagree")
    return {
        "lineage_id": lineage_id,
        "engine_epoch": engine_epoch,
        "snapshot_id": snapshot_id,
        "policy_version": policy_version,
        "adapter_sha256": adapter_sha256,
        "attempt_id": attempt_id,
        "batch_id": batch_id,
        "input_batch_hash": input_batch_hash,
    }


def _scope_binding(store: BepaStore, scope_id: str) -> dict[str, Any]:
    if not isinstance(scope_id, str) or not scope_id:
        raise CheckpointResetReceiptError("BEPA reset receipt target scope is invalid")
    try:
        row = store.get_budget_scope(scope_id)
    except KeyError as exc:
        raise CheckpointResetReceiptError("BEPA reset receipt target scope is missing") from exc
    return {"scope_id": scope_id, "max_total_tokens": int(row["max_total_tokens"]),
            "max_jobs": row["max_jobs"]}


def write_checkpoint_reset_receipt(
    store: BepaStore,
    *,
    recovery: Mapping[str, Any],
    policy_version: int,
    adapter_sha256: str,
    target_budget_scope: str,
) -> dict[str, Any]:
    """Persist one receipt after the reset ledger is proven empty for its new scope."""
    binding = _checkpoint_binding(
        recovery, policy_version=policy_version, adapter_sha256=adapter_sha256,
    )
    scope = _scope_binding(store, target_budget_scope)
    if scope["max_total_tokens"] <= 0:
        raise CheckpointResetReceiptError("BEPA reset receipt requires a positive target budget")
    receipt = {
        "schema": SCHEMA,
        "checkpoint": binding,
        "budget_scope": scope,
        "reset_state": {
            "used_tokens": 0,
            "reserved_tokens": 0,
            "usage_unknown": False,
            "row_counts": {name: 0 for name in _RESET_TABLES},
            "ready_sequence": 0,
        },
    }
    try:
        store.write_checkpoint_reset_receipt(target_budget_scope, receipt)
    except ContractError as exc:
        raise CheckpointResetReceiptError(str(exc)) from exc
    return receipt


def read_checkpoint_reset_receipt(
    store: BepaStore,
    *,
    recovery: Mapping[str, Any],
    policy_version: int,
    adapter_sha256: str,
    target_budget_scope: str,
) -> dict[str, Any] | None:
    """Read only a receipt matching this exact checkpoint, publication, and scope."""
    try:
        receipt = store.checkpoint_reset_receipt(target_budget_scope)
    except ContractError as exc:
        raise CheckpointResetReceiptError(str(exc)) from exc
    if receipt is None:
        return None
    expected = {
        "schema": SCHEMA,
        "checkpoint": _checkpoint_binding(
            recovery, policy_version=policy_version, adapter_sha256=adapter_sha256,
        ),
        "budget_scope": _scope_binding(store, target_budget_scope),
        "reset_state": {
            "used_tokens": 0,
            "reserved_tokens": 0,
            "usage_unknown": False,
            "row_counts": {name: 0 for name in _RESET_TABLES},
            "ready_sequence": 0,
        },
    }
    if receipt != expected:
        raise CheckpointResetReceiptError("BEPA reset receipt differs from the restored checkpoint or target scope")
    return receipt
