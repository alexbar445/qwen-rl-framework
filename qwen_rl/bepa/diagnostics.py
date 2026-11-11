"""Read-only BEPA status diagnostics backed by persisted metadata only."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any, Sequence

from .contracts import ContractError

_MAX_DIAGNOSTIC_TASKS = 20
_MAX_DIAGNOSTIC_BATCH_IDS = 512


def _check_positive_int(value: int, field_name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        bound = "non-negative" if allow_zero else "positive"
        raise ContractError(f"{field_name} must be a {bound} integer")
    return value


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f"corrupt JSON in BEPA database: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("corrupt BEPA database: expected JSON object")
    return result


def build_diagnostics(
    store: Any,
    *,
    budget_scope: str,
    engine_epoch: str,
    current_policy_version: int,
    registered_task_keys: Sequence[str] | None = None,
    run_id: str | None = None,
    held_task_keys: Sequence[str] | None = None,
    run_batch_ids: Sequence[str] | None = None,
    task_limit: int = _MAX_DIAGNOSTIC_TASKS,
) -> dict[str, Any]:
    """Build bounded diagnostics without reading artifacts or mutating state."""
    from .store import _cache_metadata_state

    if not budget_scope or not engine_epoch:
        raise ContractError("budget_scope and engine_epoch must be non-empty")
    _check_positive_int(current_policy_version, "current_policy_version", allow_zero=True)
    _check_positive_int(task_limit, "task_limit")
    if task_limit > _MAX_DIAGNOSTIC_TASKS:
        raise ContractError(f"task_limit cannot exceed {_MAX_DIAGNOSTIC_TASKS}")
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ContractError("run_id must be non-empty when provided")
    if (isinstance(registered_task_keys, (str, bytes)) or isinstance(held_task_keys, (str, bytes))
            or isinstance(run_batch_ids, (str, bytes))):
        raise ContractError("registered, held, and run batch keys must be sequences of IDs")
    if registered_task_keys is not None and any(not isinstance(key, str) or not key for key in registered_task_keys):
        raise ContractError("registered_task_keys must contain non-empty strings")
    if held_task_keys is not None and any(not isinstance(key, str) or not key for key in held_task_keys):
        raise ContractError("held_task_keys must contain non-empty strings")
    if run_batch_ids is not None:
        if len(run_batch_ids) > _MAX_DIAGNOSTIC_BATCH_IDS:
            raise ContractError(f"run_batch_ids cannot exceed {_MAX_DIAGNOSTIC_BATCH_IDS}")
        if any(not isinstance(batch_id, str) or not batch_id for batch_id in run_batch_ids):
            raise ContractError("run_batch_ids must contain non-empty strings")

    conn = store._conn()
    as_of = time.time()
    held_keys = tuple(dict.fromkeys(held_task_keys or ()))
    batch_ids = tuple(dict.fromkeys(run_batch_ids or ()))
    all_allowed_keys = [row[0] for row in conn.execute(
        "SELECT task_key FROM tasks WHERE allowed=1 ORDER BY task_key"
    ).fetchall()]
    allowed_key_set = set(all_allowed_keys)
    registered_key_set = None if registered_task_keys is None else set(registered_task_keys)
    # Held tasks prioritize the bounded timeline; metadata counters cover the
    # current registered allowed-task set, even when nothing is held.
    scope_task_keys = (
        all_allowed_keys if registered_key_set is None
        else [key for key in all_allowed_keys if key in registered_key_set]
    )
    scope_key_set = set(scope_task_keys)

    global_jobs = {row["state"]: int(row["count"]) for row in conn.execute(
        "SELECT state,count(*) AS count FROM rescue_jobs GROUP BY state"
    ).fetchall()}
    global_reservations = {row["state"]: int(row["count"]) for row in conn.execute(
        "SELECT state,count(*) AS count FROM reservations GROUP BY state"
    ).fetchall()}
    global_active_caches = int(conn.execute(
        "SELECT count(*) FROM cache_entries WHERE active=1"
    ).fetchone()[0])

    budget = store.get_budget_scope(budget_scope)
    scoped_jobs = {row["state"]: int(row["count"]) for row in conn.execute(
        "SELECT state,count(*) AS count FROM rescue_jobs WHERE budget_scope=? GROUP BY state",
        (budget_scope,),
    ).fetchall()}
    terminal_rows = conn.execute(
        "SELECT state,COALESCE(fail_reason,'unspecified') AS reason,count(*) AS count "
        "FROM rescue_jobs WHERE budget_scope=? AND state IN "
        "('failed','exhausted','outcome_unknown','rejected','cancelled') "
        "GROUP BY state,COALESCE(fail_reason,'unspecified') ORDER BY state,reason",
        (budget_scope,),
    ).fetchall()
    terminal_reasons = [
        {"state": row["state"], "reason": row["reason"], "count": int(row["count"])}
        for row in terminal_rows
    ]
    budget_tokens = {
        "cap": int(budget["max_total_tokens"]),
        "used": int(budget["used_tokens"]),
        "reserved": int(budget["reserved_tokens"]),
        "remaining": max(0, int(budget["max_total_tokens"])
                         - int(budget["used_tokens"]) - int(budget["reserved_tokens"])),
        "usage_unknown": bool(budget["usage_unknown"]),
        "max_jobs": budget["max_jobs"],
    }

    plan_rows = conn.execute(
        "SELECT p.*,j.source_episode_id AS job_source_episode_id,"
        "j.source_receipt_ref AS job_source_receipt_ref,j.plan_artifact_ref AS job_plan_artifact_ref,"
        "j.state AS job_state,j.fail_reason AS job_fail_reason,j.attempts AS job_attempts,"
        "j.max_attempts AS job_max_attempts,j.created_at AS job_created_at,"
        "j.updated_at AS job_updated_at,j.lease_expires_at AS job_lease_expires_at,"
        "j.budget_scope AS job_budget_scope,t.allowed AS task_allowed "
        "FROM plans p JOIN tasks t USING(task_key) LEFT JOIN rescue_jobs j ON j.job_id=p.job_id "
        "WHERE t.allowed=1 ORDER BY p.updated_at DESC,p.plan_id DESC"
    ).fetchall()
    scoped_plan_rows = [row for row in plan_rows if row["task_key"] in scope_key_set]

    def plan_provenance(row: sqlite3.Row) -> str:
        if row["job_state"] is None:
            return "job_missing"
        expected = (row["source_episode_id"], row["source_receipt_ref"], row["plan_artifact_ref"])
        actual = (row["job_source_episode_id"], row["job_source_receipt_ref"], row["job_plan_artifact_ref"])
        if not all(expected) or expected != actual:
            return "metadata_mismatch"
        return "stored_linkage_unchecked"

    plan_provenance_by_id = {row["plan_id"]: plan_provenance(row) for row in plan_rows}
    linked_plan_rows = [row for row in scoped_plan_rows
                          if plan_provenance_by_id[row["plan_id"]] == "stored_linkage_unchecked"]
    plan_states = {state: 0 for state in ("ready", "seed_running", "consumed", "exhausted")}
    for row in linked_plan_rows:
        if row["status"] in plan_states:
            plan_states[row["status"]] += 1

    latest_generations: dict[str, dict[str, Any]] = {}
    generation_integrity: dict[str, str] = {}
    generation_rows = conn.execute(
        "SELECT e.entity_id,e.details_json,e.created_at FROM events e JOIN ("
        "SELECT entity_id,max(sequence) AS sequence FROM events "
        "WHERE entity_type='plan' AND action='seed_generation' GROUP BY entity_id"
        ") latest ON latest.entity_id=e.entity_id AND latest.sequence=e.sequence"
    ).fetchall()
    for event in generation_rows:
        if event["entity_id"] not in plan_provenance_by_id:
            continue
        try:
            details = _json_dict(event["details_json"])
            generation_id = details.get("generation_id")
            parsed = uuid.UUID(generation_id) if isinstance(generation_id, str) else None
            valid = (parsed is not None and str(parsed) == generation_id
                     and isinstance(details.get("engine_epoch"), str)
                     and bool(details["engine_epoch"])
                     and type(details.get("start_policy_version")) is int
                     and details["start_policy_version"] >= 0)
        except (RuntimeError, ValueError, TypeError, KeyError):
            valid = False
            details = {}
        if not valid:
            generation_integrity[event["entity_id"]] = "malformed"
            continue
        generation_integrity[event["entity_id"]] = "valid"
        latest_generations[event["entity_id"]] = {
            **details, "started_at": float(event["created_at"]),
        }

    cache_rows = conn.execute(
        "SELECT c.*,e.task_key AS episode_task_key,e.engine_epoch AS episode_engine_epoch,"
        "e.source_policy_version AS episode_policy_version,e.source_policy_sha256 AS episode_policy_sha256,"
        "e.episode_kind AS episode_kind,e.raw_artifact_ref AS episode_raw_ref,"
        "e.projection_artifact_ref AS episode_projection_ref,e.receipt_artifact_ref AS episode_receipt_ref,"
        "e.exact AS episode_exact FROM cache_entries c JOIN tasks t ON t.task_key=c.task_key "
        "LEFT JOIN episodes e ON e.episode_id=c.source_episode_id "
        "WHERE t.allowed=1 AND c.engine_epoch=? AND c.ready_sequence=("
        "SELECT max(c2.ready_sequence) FROM cache_entries c2 "
        "WHERE c2.task_key=c.task_key AND c2.engine_epoch=c.engine_epoch) "
        "ORDER BY c.ready_sequence DESC,c.created_at DESC",
        (engine_epoch,),
    ).fetchall()
    latest_cache_by_task: dict[str, sqlite3.Row] = {}
    for row in cache_rows:
        latest_cache_by_task.setdefault(row["task_key"], row)
    reservation_counts: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT r.cache_entry_id,r.state,count(*) AS count FROM reservations r "
        "JOIN cache_entries c ON c.cache_entry_id=r.cache_entry_id "
        "WHERE c.engine_epoch=? AND c.ready_sequence=(SELECT max(c2.ready_sequence) "
        "FROM cache_entries c2 WHERE c2.task_key=c.task_key AND c2.engine_epoch=c.engine_epoch) "
        "GROUP BY r.cache_entry_id,r.state", (engine_epoch,),
    ).fetchall():
        reservation_counts.setdefault(row["cache_entry_id"], {})[row["state"]] = int(row["count"])
    def cache_metadata(task_key: str) -> dict[str, Any]:
        cache = latest_cache_by_task.get(task_key)
        if cache is None:
            return {
                "metadata_state": "absent", "source_policy_version": None,
                "source_policy_sha256": None, "age_updates": None,
                "age_limit": store.max_age_updates, "committed_uses": 0,
                "pending_reserved": 0, "pending_commit_unknown": 0,
                "uses_remaining": store.max_committed_uses,
                    "last_reason": None, "last_event_action": None, "last_event_at": None,
                "created_at": None, "provenance_recheck": "not_performed",
                "provenance_state": "not_applicable", "cache_entry_id": None,
            }
        uses = reservation_counts.get(cache["cache_entry_id"], {})
        pending_reserved = int(uses.get("reserved", 0))
        pending_unknown = int(uses.get("commit_unknown", 0))
        committed_reservations = int(uses.get("committed", 0))
        committed_uses = int(cache["committed_uses"])
        reservation_committed_total = committed_reservations + pending_unknown
        age = current_policy_version - int(cache["source_policy_version"])
        operational_state = _cache_metadata_state(
            active=bool(cache["active"]), age_updates=age,
            max_age_updates=store.max_age_updates, reserved=pending_reserved,
            committed=reservation_committed_total, max_committed_uses=store.max_committed_uses,
            pending=bool(pending_reserved or pending_unknown),
        )
        uses_remaining = max(0, store.max_committed_uses - pending_reserved - reservation_committed_total)
        provenance_ok = (
            cache["episode_exact"] == 1
            and cache["episode_task_key"] == task_key
            and cache["episode_engine_epoch"] == cache["engine_epoch"]
            and cache["episode_policy_version"] == cache["source_policy_version"]
            and cache["episode_policy_sha256"] == cache["source_policy_sha256"]
            and cache["episode_raw_ref"] == cache["raw_artifact_ref"]
            and cache["episode_projection_ref"] == cache["projection_artifact_ref"]
            and cache["episode_receipt_ref"] == cache["receipt_artifact_ref"]
        )
        if pending_unknown:
            metadata_state = "blocked_unknown"
        elif pending_reserved:
            metadata_state = "blocked_pending"
        elif not provenance_ok:
            metadata_state = "blocked_unverified"
        elif committed_uses != committed_reservations:
            metadata_state = "blocked_unknown"
        else:
            metadata_state = operational_state
        return {
            "metadata_state": metadata_state,
            "source_policy_version": int(cache["source_policy_version"]),
            "source_policy_sha256": cache["source_policy_sha256"],
            "age_updates": age,
            "age_limit": store.max_age_updates,
            "committed_uses": committed_uses,
            "committed_reservations": committed_reservations,
            "pending_reserved": pending_reserved,
            "pending_commit_unknown": pending_unknown,
            "uses_remaining": uses_remaining,
            "last_reason": None,
            "last_event_action": None,
            "last_event_at": None,
            "created_at": float(cache["created_at"]),
            "active": bool(cache["active"]),
            "provenance_recheck": "not_performed",
            "provenance_state": "stored_episode_linkage" if provenance_ok else "metadata_mismatch",
            "cache_entry_id": cache["cache_entry_id"],
        }

    cache_breakdown = {
        key: 0 for key in ("metadata_eligible", "blocked_age", "blocked_quota", "blocked_pending",
                           "blocked_unknown", "blocked_unverified", "future_policy", "inactive", "absent")
    }
    mismatch_task_keys = {
        row["task_key"] for row in scoped_plan_rows
            if plan_provenance_by_id[row["plan_id"]] != "stored_linkage_unchecked"
    }
    scoped_cache_metadata: dict[str, dict[str, Any]] = {}
    for task_key in scope_task_keys:
        metadata = cache_metadata(task_key)
        scoped_cache_metadata[task_key] = metadata
        cache_breakdown[metadata["metadata_state"]] += 1
        if metadata["provenance_state"] == "metadata_mismatch":
            mismatch_task_keys.add(task_key)

    generation_summary = {
        "running": 0, "attempts_started": 0, "failed_attempts": 0,
        "attempt_cap": 2, "attempt_cap_reached": 0,
        "malformed_generation_identity": 0, "terminal_reasons": [],
    }
    generation_failures: dict[str, dict[str, int]] = {}
    for row in linked_plan_rows:
        generation = latest_generations.get(row["plan_id"])
        if generation is None:
            if generation_integrity.get(row["plan_id"]) == "malformed":
                generation_summary["malformed_generation_identity"] += 1
            continue
        attempts = int(row["seed_attempts"])
        generation_summary["attempts_started"] += attempts
        if row["status"] == "seed_running":
            generation_summary["running"] += 1
        if row["status"] == "exhausted" and attempts >= 2:
            generation_summary["attempt_cap_reached"] += 1
        latest_attempt_succeeded_or_open = row["status"] in {"consumed", "seed_running"}
        generation_summary["failed_attempts"] += max(0, attempts - int(latest_attempt_succeeded_or_open))
        if row["status"] == "exhausted":
            reason = row["job_fail_reason"] or "unknown"
            generation_failures.setdefault(reason, {})[row["status"]] = (
                generation_failures.setdefault(reason, {}).get(row["status"], 0) + 1
            )
    generation_summary["terminal_reasons"] = [
        {"reason": reason, "count": counts.get("exhausted", 0)}
        for reason, counts in sorted(generation_failures.items())
    ]

    current_run: dict[str, Any] = {
        "run_id": run_id,
        "seed_claims": {"claimed": None, "failed": None, "succeeded": None,
                        "open_observed": None, "basis": "run_id_unavailable"},
        "reservations": {"reserved": None, "committed": None, "released": None,
                         "commit_unknown": None, "basis": "batch_ids_unavailable",
                         "observed_batch_count": None, "coverage": "batch_ids_unavailable",
                         "prior_resumed_uses": "unknown"},
        "updates": {"prepared": None, "optimizer_entered": None, "committed": None,
                    "commit_unknown": None, "aborted": None, "basis": "batch_ids_unavailable"},
    }
    if run_id is not None:
        try:
            rows = conn.execute(
                "SELECT action,count(*) AS count FROM events WHERE entity_type='plan' "
                "AND action IN ('seed_claimed','seed_failed','seed_succeeded') "
                "AND json_extract(details_json,'$.run_id')=? GROUP BY action",
                (run_id,),
            ).fetchall()
            action_counts = {row["action"]: int(row["count"]) for row in rows}
            claimed = action_counts.get("seed_claimed", 0)
            failed = action_counts.get("seed_failed", 0)
            succeeded = action_counts.get("seed_succeeded", 0)
            current_run["seed_claims"] = {
                "claimed": claimed, "failed": failed, "succeeded": succeeded,
                "open_observed": max(0, claimed - failed - succeeded),
                "basis": "persisted_seed_claim_run_id",
            }
        except sqlite3.OperationalError:
            current_run["seed_claims"]["basis"] = "json_run_id_query_unavailable"
    if run_batch_ids is not None:
        if batch_ids:
            placeholders = ",".join("?" for _ in batch_ids)
            reservation_states = {"reserved": 0, "committed": 0, "released": 0, "commit_unknown": 0}
            for row in conn.execute(
                f"SELECT state,count(*) AS count FROM reservations WHERE batch_id IN ({placeholders}) GROUP BY state",
                batch_ids,
            ).fetchall():
                reservation_states[row["state"]] = int(row["count"])
            update_states = {"prepared": 0, "optimizer_entered": 0, "committed": 0,
                             "commit_unknown": 0, "aborted": 0}
            for row in conn.execute(
                f"SELECT state,count(*) AS count FROM update_attempts WHERE batch_id IN ({placeholders}) GROUP BY state",
                batch_ids,
            ).fetchall():
                update_states[row["state"]] = int(row["count"])
            coverage = "current_process_observed_cohorts"
            basis = "persisted_reservation_batch_ids"
        else:
            reservation_states = {"reserved": 0, "committed": 0, "released": 0, "commit_unknown": 0}
            update_states = {"prepared": 0, "optimizer_entered": 0, "committed": 0,
                             "commit_unknown": 0, "aborted": 0}
            coverage = "no_batches_observed"
            basis = "empty_current_process_cohort_set"
        current_run["reservations"] = {
            **reservation_states, "basis": basis,
            "observed_batch_count": len(batch_ids), "coverage": coverage,
            "prior_resumed_uses": "unknown",
        }
        current_run["updates"] = {**update_states, "basis": basis}

    # Select at most twenty task rows, with held tasks first and then durable
    # seed/plan/job/cache work in deterministic order.
    task_priorities: dict[str, str] = {}

    def add_task(task_key: str, priority: str) -> None:
        if (task_key in scope_key_set and task_key in allowed_key_set
                and task_key not in task_priorities and len(task_priorities) < task_limit):
            task_priorities[task_key] = priority

    for task_key in held_keys:
        add_task(task_key, "held")
    for row in sorted((item for item in plan_rows if item["status"] == "seed_running"),
                      key=lambda item: (float(item["updated_at"]), item["task_key"], item["plan_id"])):
        add_task(row["task_key"], "seed_running")
    for row in conn.execute(
        "SELECT task_key,updated_at FROM rescue_jobs WHERE budget_scope=? AND state='teacher_running' "
        "ORDER BY updated_at,task_key,job_id LIMIT ?", (budget_scope, task_limit),
    ).fetchall():
        add_task(row["task_key"], "teacher_running")
    for row in sorted((item for item in plan_rows if item["status"] == "ready"),
                      key=lambda item: (float(item["created_at"]), item["task_key"], item["plan_id"])):
        add_task(row["task_key"], "plan_ready")
    for states in (("queued", "retry_wait"), ("source_submitted", "source_verified")):
        placeholders = ",".join("?" for _ in states)
        for row in conn.execute(
            f"SELECT task_key,updated_at FROM rescue_jobs WHERE budget_scope=? AND state IN ({placeholders}) "
            "ORDER BY created_at,task_key,job_id LIMIT ?",
            (budget_scope, *states, task_limit),
        ).fetchall():
            add_task(row["task_key"], "job_queued" if states[0] == "queued" else "job_submitted")
    if batch_ids:
        placeholders = ",".join("?" for _ in batch_ids)
        for row in conn.execute(
            f"SELECT DISTINCT task_key FROM reservations WHERE batch_id IN ({placeholders}) ORDER BY task_key",
            batch_ids,
        ).fetchall():
            add_task(row["task_key"], "current_run_reservation")
    for task_key, cache in latest_cache_by_task.items():
        if cache["active"]:
            add_task(task_key, "cache_active")
    terminal_plans = sorted(
        (item for item in plan_rows if item["status"] in {"exhausted", "consumed"}),
        key=lambda item: (-float(item["updated_at"]), item["task_key"], item["plan_id"]),
    )
    for row in terminal_plans:
        add_task(row["task_key"], "plan_terminal")

    selected_plan_by_task: dict[str, sqlite3.Row] = {}
    for row in plan_rows:
        if row["task_key"] in task_priorities:
            selected_plan_by_task.setdefault(row["task_key"], row)
    current_job_by_task: dict[str, sqlite3.Row] = {}
    for task_key in task_priorities:
        row = conn.execute(
            "SELECT * FROM rescue_jobs WHERE budget_scope=? AND task_key=? "
            "ORDER BY updated_at DESC,job_id DESC LIMIT 1",
            (budget_scope, task_key),
        ).fetchone()
        if row is not None:
            current_job_by_task[task_key] = row

    tasks: list[dict[str, Any]] = []
    for task_key, priority in task_priorities.items():
        plan = selected_plan_by_task.get(task_key)
        job = current_job_by_task.get(task_key)
        cache = latest_cache_by_task.get(task_key)
        cache_info = cache_metadata(task_key)
        if cache is not None:
            last_cache_event = conn.execute(
                "SELECT action,reason,created_at FROM events WHERE entity_type='cache' AND entity_id=? "
                "AND action IN ('activated','expired','expired_after_update','exhausted') "
                "ORDER BY sequence DESC LIMIT 1", (cache["cache_entry_id"],),
            ).fetchone()
            if last_cache_event is not None:
                cache_info["last_reason"] = last_cache_event["reason"]
                cache_info["last_event_action"] = last_cache_event["action"]
                cache_info["last_event_at"] = float(last_cache_event["created_at"])
        plan_events: list[dict[str, Any]] = []
        if plan is not None:
            for event in conn.execute(
                "SELECT action,reason,details_json,created_at FROM events "
                "WHERE entity_type='plan' AND entity_id=? "
                "AND action IN ('ready','seed_generation','seed_claimed','seed_failed','seed_succeeded') "
                "ORDER BY sequence DESC LIMIT 100", (plan["plan_id"],),
            ).fetchall():
                try:
                    details = _json_dict(event["details_json"]) if event["details_json"] else {}
                except RuntimeError:
                    details = {}
                plan_events.append({"action": event["action"], "reason": event["reason"],
                                    "details": details, "created_at": float(event["created_at"])})
        generation = latest_generations.get(plan["plan_id"]) if plan is not None else None
        gen_id = generation.get("generation_id") if generation else None
        gen_events = [event for event in plan_events
                      if gen_id is not None and event["details"].get("generation_id") == gen_id]
        ready_event = next((event for event in plan_events if event["action"] == "ready"), None)
        claim_event = next((event for event in gen_events if event["action"] == "seed_claimed"), None)
        claim_details = claim_event["details"] if claim_event else {}
        claim_attempt_id = claim_details.get("attempt_id")
        terminal_event = next((event for event in gen_events
                               if event["action"] in {"seed_failed", "seed_succeeded"}
                               and claim_attempt_id is not None
                               and event["details"].get("attempt_id") == claim_attempt_id), None)
        last_failure = next((event for event in gen_events if event["action"] == "seed_failed"), None)
        plan_state = plan["status"] if plan is not None else None
        attempt_count = int(plan["seed_attempts"]) if plan is not None and generation else None
        cap_reached = bool(plan_state == "exhausted" and attempt_count is not None and attempt_count >= 2)
        if cap_reached:
            terminal_reason = "guided_attempt_cap_reached"
        elif terminal_event is not None and terminal_event["action"] == "seed_succeeded":
            terminal_reason = "cache_created"
        elif last_failure is not None:
            terminal_reason = last_failure["reason"] or "guided_episode_failed"
        else:
            terminal_reason = None

        plan_job = None
        if plan is not None:
            job_row = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (plan["job_id"],)).fetchone()
            if job_row is not None:
                plan_job = job_row
        display_job = plan_job or job
        job_data = None
        if display_job is not None:
            job_data = {
                "state": display_job["state"], "reason": display_job["fail_reason"],
                "attempts": int(display_job["attempts"]), "max_attempts": int(display_job["max_attempts"]),
                "created_at": float(display_job["created_at"]), "updated_at": float(display_job["updated_at"]),
                "lease_expires_at": (float(display_job["lease_expires_at"])
                                      if display_job["lease_expires_at"] is not None else None),
                "budget_scope": display_job["budget_scope"],
            }

        plan_data = None
        if plan is not None:
            plan_data = {
                "plan_id": plan["plan_id"], "state": plan_state,
                "provenance_state": plan_provenance_by_id.get(plan["plan_id"], "unknown"),
                "available_at": float(ready_event["created_at"]) if ready_event else float(plan["created_at"]),
                "generation_id": gen_id,
                "generation_integrity": generation_integrity.get(plan["plan_id"], "missing"),
                "generation_started_at": generation.get("started_at") if generation else None,
                "generation_start_policy_version": generation.get("start_policy_version") if generation else None,
                "attempt_count": attempt_count,
                "attempt_cap": 2,
                "attempt_cap_reached": cap_reached,
                "attempt_started_at": claim_event["created_at"] if claim_event else None,
                "attempt_deadline": claim_details.get("lease_expires_at"),
                "attempt_run_id": claim_details.get("run_id"),
                "attempt_finished_at": terminal_event["created_at"] if terminal_event else None,
                "terminal_reason": terminal_reason,
                "last_failure_reason": last_failure["reason"] if last_failure else None,
            }

        reservation_data = None
        reservation_candidates = []
        if run_batch_ids is not None and batch_ids:
            placeholders = ",".join("?" for _ in batch_ids)
            reservation_candidates = conn.execute(
                f"SELECT r.*,u.new_policy_version FROM reservations r "
                f"LEFT JOIN update_attempts u ON u.attempt_id=r.update_attempt_id "
                f"WHERE r.task_key=? AND r.engine_epoch=? AND r.batch_id IN ({placeholders}) "
                f"ORDER BY r.updated_at DESC,r.reservation_id DESC LIMIT 1",
                (task_key, engine_epoch, *batch_ids),
            ).fetchall()
        if not reservation_candidates:
            reservation_candidates = conn.execute(
                "SELECT r.*,u.new_policy_version FROM reservations r "
                "LEFT JOIN update_attempts u ON u.attempt_id=r.update_attempt_id "
                "WHERE r.task_key=? AND r.engine_epoch=? "
                "ORDER BY r.updated_at DESC,r.reservation_id DESC LIMIT 1",
                (task_key, engine_epoch),
            ).fetchall()
        if reservation_candidates:
            reservation = reservation_candidates[0]
            current_run_reservation = run_batch_ids is not None and reservation["batch_id"] in batch_ids
            reservation_data = {
                "state": reservation["state"], "reservation_id": reservation["reservation_id"],
                "batch_id": reservation["batch_id"],
                "created_at": float(reservation["created_at"]),
                "committed_at": float(reservation["updated_at"]) if reservation["state"] == "committed" else None,
                "commit_policy_version": reservation["new_policy_version"],
                "basis": "current_process_observed_cohorts" if current_run_reservation else "engine_epoch_latest",
            }

        wait_reason = None
        wait_since = None
        wait_deadline = None
        if plan_state == "seed_running":
            wait_reason = "guided_seed_running"
            wait_since = plan_data["attempt_started_at"] if plan_data else None
            wait_deadline = plan_data["attempt_deadline"] if plan_data else None
        elif cache_info["metadata_state"] in {"blocked_pending", "blocked_unknown", "blocked_age",
                                               "blocked_quota", "blocked_unverified"}:
            wait_reason = cache_info["metadata_state"]
            if reservation_data is not None and reservation_data["state"] in {"reserved", "commit_unknown"}:
                wait_since = reservation_data["created_at"]
            elif (cache_info["metadata_state"] == "blocked_age"
                  and cache_info.get("last_event_action") in {"expired", "expired_after_update"}):
                wait_since = cache_info["last_event_at"]
            elif (cache_info["metadata_state"] == "blocked_quota"
                  and cache_info.get("last_event_action") == "exhausted"):
                wait_since = cache_info["last_event_at"]
        elif display_job is not None and display_job["state"] in {"queued", "retry_wait", "teacher_running",
                                                                   "source_submitted", "source_verified"}:
            wait_reason = f"teacher_job_{display_job['state']}"
            wait_since = float(display_job["created_at"])
            wait_deadline = (float(display_job["lease_expires_at"])
                             if display_job["lease_expires_at"] is not None else None)
        elif plan_state == "ready":
            wait_reason = "verified_plan_ready"
            wait_since = (generation.get("started_at") if generation
                          else (float(plan["created_at"]) if plan is not None else None))
        elif plan_state == "exhausted":
            wait_reason = terminal_reason or "seed_attempts_exhausted"
        elif display_job is not None and display_job["state"] in {"failed", "exhausted", "outcome_unknown", "rejected"}:
            wait_reason = display_job["fail_reason"] or display_job["state"]
            wait_since = float(display_job["updated_at"])

        tasks.append({
            "task_id": task_key,
            "task_key": task_key,
            "priority": priority,
            "job": job_data,
            "provenance": {
                "state": plan_data["provenance_state"] if plan_data else "not_applicable",
                "recheck": "not_performed",
            },
            "plan": plan_data,
            "cache": cache_info,
            "reservation": reservation_data,
            "wait": {
                "reason": wait_reason,
                "since": wait_since,
                "elapsed_seconds": max(0.0, as_of - wait_since) if wait_since is not None else None,
                "deadline": wait_deadline,
                "time_basis": "persisted_event_or_row" if wait_since is not None else "unavailable",
            },
        })

    task_basis = ("current_registered_allowed_tasks" if registered_key_set is not None
                  else "all_allowed_tasks")
    cache_basis = ("latest_cache_per_registered_allowed_task_in_engine_epoch"
                   if registered_key_set is not None else "latest_cache_per_allowed_task_in_engine_epoch")
    return {
        "diagnostics_version": "bepa-watch-v1",
        "as_of": as_of,
        "context": {
            "budget_scope": budget_scope,
            "engine_epoch": engine_epoch,
            "current_policy_version": current_policy_version,
            "run_id": run_id,
            "held_task_keys": list(key for key in held_keys if key in scope_key_set),
            "registered_task_count": len(scope_task_keys),
            "task_basis": task_basis,
            "task_limit": task_limit,
            "task_rows_truncated": len(task_priorities) >= task_limit,
        },
        "historical_global": {
            "job_states": global_jobs,
            "active_cache_entries": global_active_caches,
            "reservation_states": global_reservations,
            "basis": "all_persisted_scopes_and_runs",
        },
        "budget_scope": {
            "scope_id": budget_scope,
            "job_states": scoped_jobs,
            "terminal_reasons": terminal_reasons,
            "tokens": budget_tokens,
        },
        "plans": {
            "task_basis": task_basis,
            "states": plan_states,
            "provenance_basis": "stored_source_receipt_linkage_not_rechecked",
            "unverified_or_mismatch": len(mismatch_task_keys),
            "seed_generation": generation_summary,
            "cache_metadata": {
                **cache_breakdown,
                "provenance_recheck": "not_performed",
                "basis": cache_basis,
            },
            "usage": {
                "committed_uses": sum(
                    int(scoped_cache_metadata[key].get("committed_uses", 0)) for key in scope_task_keys
                ),
                "pending_reserved": sum(
                    int(scoped_cache_metadata[key].get("pending_reserved", 0)) for key in scope_task_keys
                ),
                "pending_commit_unknown": sum(
                    int(scoped_cache_metadata[key].get("pending_commit_unknown", 0)) for key in scope_task_keys
                ),
                "max_committed_uses": store.max_committed_uses,
            },
        },
        "current_run": current_run,
        "tasks": tasks,
    }
