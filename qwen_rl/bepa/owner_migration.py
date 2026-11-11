"""Pure validation and replay for append-only rescue-owner migrations."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any


OWNER_MIGRATION_SCHEMA = "bepa-owner-migration/v1"
OWNER_MIGRATION_ACTION = "rescue_owner_migrated"
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")


def is_safe_task_id(value: Any) -> bool:
    """Validate the public task-ID spelling used by checkpoint recovery metadata."""
    return isinstance(value, str) and _SAFE_TASK_ID.fullmatch(value) is not None


class OwnerHistoryError(ValueError):
    """Persisted owner events cannot be folded into one valid history."""


def _owner_pair(details: Mapping[str, Any], worker_key: str, executor_key: str) -> dict[str, str]:
    worker_id, executor_id = details.get(worker_key), details.get(executor_key)
    if (not isinstance(worker_id, str) or not worker_id or worker_id.strip() != worker_id
            or not isinstance(executor_id, str) or not executor_id
            or executor_id.strip() != executor_id):
        raise OwnerHistoryError("persisted BEPA owner pair is malformed")
    return {"worker_id": worker_id, "executor_id": executor_id}


def fold_owner_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Replay bound, migration, and origin-group events in sequence order."""
    current: dict[str, str] | None = None
    initial: dict[str, str] | None = None
    migration_ids: list[str] = []
    migration_records: list[dict[str, Any]] = []
    original_group_ids: list[str] = []
    ordered = sorted(events, key=lambda row: row.get("sequence", -1))
    for row in ordered:
        action = row.get("action")
        details = row.get("details")
        if not isinstance(details, Mapping):
            raise OwnerHistoryError("persisted BEPA owner event details are malformed")
        if action == "rescue_owner_bound":
            owner = _owner_pair(details, "worker_id", "executor_id")
            if current is None:
                current = initial = owner
            elif current != owner:
                raise OwnerHistoryError("BEPA task has conflicting persisted owner bindings")
        elif action == OWNER_MIGRATION_ACTION:
            if details.get("schema") != OWNER_MIGRATION_SCHEMA:
                raise OwnerHistoryError("persisted BEPA owner migration schema is unsupported")
            migration_id = details.get("migration_id")
            if not isinstance(migration_id, str) or not migration_id:
                raise OwnerHistoryError("persisted BEPA owner migration ID is malformed")
            if (not is_safe_task_id(details.get("task_id"))
                    or any(not isinstance(details.get(field), str) or not details[field]
                           or details[field].strip() != details[field]
                           for field in ("run_id", "lineage_id", "snapshot_id", "reason"))):
                raise OwnerHistoryError("persisted BEPA owner migration audit scope is malformed")
            source = _owner_pair(details, "from_worker_id", "from_executor_id")
            target = _owner_pair(details, "to_worker_id", "to_executor_id")
            if current is None or current != source:
                raise OwnerHistoryError("BEPA owner migration source differs from the effective owner")
            if source == target:
                raise OwnerHistoryError("BEPA owner migration does not change its owner")
            current = target
            migration_ids.append(migration_id)
            migration_records.append({
                "migration_id": migration_id,
                "run_id": details.get("run_id"),
                "lineage_id": details.get("lineage_id"),
                "snapshot_id": details.get("snapshot_id"),
                "task_id": details.get("task_id"),
                "from": source,
                "to": target,
            })
        elif action == "rescue_origin_group_bound":
            owner = _owner_pair(details, "worker_id", "executor_id")
            if current is None or current != owner:
                raise OwnerHistoryError("BEPA rescue group owner differs from the effective owner at bind time")
            group_id = details.get("original_group_id")
            if not isinstance(group_id, str) or not group_id:
                raise OwnerHistoryError("persisted BEPA rescue origin group ID is malformed")
            if group_id not in original_group_ids:
                original_group_ids.append(group_id)
    if current is None:
        if original_group_ids:
            raise OwnerHistoryError("BEPA rescue group history has no owner binding")
        return None
    return {
        "owner": current,
        "initial_owner": initial,
        "migration_ids": migration_ids,
        "owner_migrations": migration_records,
        "original_group_ids": original_group_ids,
    }
