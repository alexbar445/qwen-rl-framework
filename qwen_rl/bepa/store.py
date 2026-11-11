"""Durable local BEPA state: jobs, cache, reservations, and update evidence.

The database is intended for one coordinator process on local disk. SQLite
connections are thread-local; write transactions are serialized and kept short.
No transaction spans model execution, HTTP calls, or ARC tool execution.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .contracts import (
    BudgetUnavailableError,
    CacheEntry,
    ConflictError,
    ContractError,
    JobLease,
    JobRecord,
    LeaseLostError,
    MAX_CACHE_AGE_UPDATES,
    MAX_CACHE_COMMITTED_USES,
    PlanRecord,
    Reservation,
    SourceEpisode,
    SourcePolicy,
    SourceProjection,
    StateTransitionError,
    TaskRecord,
    TaskSpec,
    TrustedReceipt,
    UpdateAttempt,
    canonical_json_bytes,
    content_hash,
    require_sha256,
)
from .artifacts import BepaArtifacts, project_source_episode
from .owner_migration import (
    OWNER_MIGRATION_ACTION,
    OWNER_MIGRATION_SCHEMA,
    OwnerHistoryError,
    fold_owner_events,
    is_safe_task_id,
)


NATIVE_RESCUE_REQUIRED_SUCCESSES = 1


def _advance_native_rescue_versions(versions, policy_version, native_exact):
    versions = list(versions)
    if len(versions) >= NATIVE_RESCUE_REQUIRED_SUCCESSES and not all(native_exact):
        return []  # A review miss starts a new practice cycle, including 5/6.
    if all(native_exact):
        if policy_version not in versions and len(versions) < NATIVE_RESCUE_REQUIRED_SUCCESSES:
            versions.append(policy_version)
    elif not any(native_exact):
        versions = []
    return versions


_SCHEMA = """
CREATE TABLE IF NOT EXISTS counters (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO counters(name, value) VALUES ('ready_sequence', 0);

CREATE TABLE IF NOT EXISTS tasks (
    task_key TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    private_target_hash TEXT NOT NULL,
    allowed INTEGER NOT NULL CHECK(allowed IN (0, 1)),
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS tasks_private_target_idx ON tasks(private_target_hash);

CREATE TABLE IF NOT EXISTS native_rescue_progress (
    puzzle_key TEXT NOT NULL,
    engine_epoch TEXT NOT NULL,
    public_challenge TEXT NOT NULL,
    private_target_hash TEXT NOT NULL,
    success_versions_json TEXT NOT NULL,
    retired INTEGER NOT NULL CHECK(retired IN (0, 1)),
    updated_at REAL NOT NULL,
    PRIMARY KEY(engine_epoch, puzzle_key)
);
CREATE INDEX IF NOT EXISTS native_rescue_retired_idx
ON native_rescue_progress(engine_epoch, retired, private_target_hash);

CREATE TABLE IF NOT EXISTS native_rescue_observations (
    engine_epoch TEXT NOT NULL,
    group_id TEXT NOT NULL,
    puzzle_key TEXT NOT NULL,
    policy_version INTEGER NOT NULL,
    native_exact_json TEXT NOT NULL,
    PRIMARY KEY(engine_epoch, group_id)
);

CREATE TABLE IF NOT EXISTS budget_scopes (
    scope_id TEXT PRIMARY KEY,
    max_total_tokens INTEGER NOT NULL CHECK(max_total_tokens >= 0),
    max_jobs INTEGER,
    used_tokens INTEGER NOT NULL DEFAULT 0 CHECK(used_tokens >= 0),
    reserved_tokens INTEGER NOT NULL DEFAULT 0 CHECK(reserved_tokens >= 0),
    usage_unknown INTEGER NOT NULL DEFAULT 0 CHECK(usage_unknown IN (0, 1)),
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rescue_jobs (
    job_id TEXT PRIMARY KEY,
    task_key TEXT NOT NULL REFERENCES tasks(task_key),
    public_job_key TEXT NOT NULL,
    budget_scope TEXT NOT NULL REFERENCES budget_scopes(scope_id),
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL CHECK(max_attempts > 0),
    used_tokens INTEGER NOT NULL DEFAULT 0 CHECK(used_tokens >= 0),
    reserved_tokens INTEGER NOT NULL DEFAULT 0 CHECK(reserved_tokens >= 0),
    max_tokens INTEGER NOT NULL CHECK(max_tokens >= 0),
    usage_unknown INTEGER NOT NULL DEFAULT 0 CHECK(usage_unknown IN (0, 1)),
    fence INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_expires_at REAL,
    source_artifact_refs_json TEXT,
    source_episode_id TEXT,
    source_receipt_ref TEXT,
    plan_artifact_ref TEXT,
    fail_reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(budget_scope, public_job_key)
);
CREATE INDEX IF NOT EXISTS jobs_claim_idx ON rescue_jobs(state, created_at);
CREATE INDEX IF NOT EXISTS jobs_task_idx ON rescue_jobs(task_key, state);

CREATE TABLE IF NOT EXISTS teacher_claim_requests (
    claim_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    job_id TEXT NOT NULL REFERENCES rescue_jobs(job_id),
    fence INTEGER NOT NULL CHECK(fence >= 1),
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS teacher_claim_job_idx ON teacher_claim_requests(job_id, fence);

CREATE TABLE IF NOT EXISTS job_mutations (
    job_id TEXT NOT NULL REFERENCES rescue_jobs(job_id),
    action TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(job_id, action, idempotency_key)
);

CREATE TABLE IF NOT EXISTS usage_events (
    scope_id TEXT NOT NULL REFERENCES budget_scopes(scope_id),
    idempotency_key TEXT NOT NULL,
    job_id TEXT NOT NULL REFERENCES rescue_jobs(job_id),
    tokens INTEGER,
    created_at REAL NOT NULL,
    PRIMARY KEY(scope_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    task_key TEXT NOT NULL REFERENCES tasks(task_key),
    engine_epoch TEXT NOT NULL,
    source_policy_version INTEGER NOT NULL,
    source_policy_sha256 TEXT NOT NULL,
    episode_kind TEXT NOT NULL,
    raw_artifact_ref TEXT NOT NULL,
    projection_artifact_ref TEXT,
    receipt_artifact_ref TEXT,
    exact INTEGER NOT NULL CHECK(exact IN (0, 1)),
    reason TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cache_entries (
    cache_entry_id TEXT PRIMARY KEY,
    task_key TEXT NOT NULL REFERENCES tasks(task_key),
    engine_epoch TEXT NOT NULL,
    source_episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
    source_policy_version INTEGER NOT NULL,
    source_policy_sha256 TEXT NOT NULL,
    row_kind TEXT NOT NULL CHECK(row_kind IN ('native_replay', 'guided_replay')),
    raw_artifact_ref TEXT NOT NULL,
    projection_artifact_ref TEXT NOT NULL,
    receipt_artifact_ref TEXT NOT NULL,
    committed_uses INTEGER NOT NULL DEFAULT 0 CHECK(committed_uses >= 0),
    ready_sequence INTEGER NOT NULL,
    active INTEGER NOT NULL CHECK(active IN (0, 1)),
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_cache_per_task_epoch
ON cache_entries(task_key, engine_epoch) WHERE active = 1;
CREATE INDEX IF NOT EXISTS cache_ready_idx ON cache_entries(engine_epoch, active, ready_sequence);

CREATE TABLE IF NOT EXISTS reservations (
    reservation_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    cache_entry_id TEXT NOT NULL REFERENCES cache_entries(cache_entry_id),
    task_key TEXT NOT NULL REFERENCES tasks(task_key),
    engine_epoch TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('reserved', 'committed', 'released', 'commit_unknown')),
    update_attempt_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(batch_id, group_id)
);
CREATE INDEX IF NOT EXISTS reservation_cache_idx ON reservations(cache_entry_id, state);
CREATE INDEX IF NOT EXISTS reservation_batch_idx ON reservations(batch_id, state);

CREATE TABLE IF NOT EXISTS update_attempts (
    attempt_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL UNIQUE,
    input_batch_hash TEXT NOT NULL,
    engine_epoch TEXT NOT NULL,
    start_policy_version INTEGER NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('prepared', 'optimizer_entered', 'committed', 'commit_unknown', 'aborted')),
    new_policy_version INTEGER,
    adapter_sha256 TEXT,
    commit_evidence_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL UNIQUE REFERENCES rescue_jobs(job_id),
    task_key TEXT NOT NULL REFERENCES tasks(task_key),
    plan_artifact_ref TEXT NOT NULL,
    source_episode_id TEXT NOT NULL,
    source_receipt_ref TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('ready', 'seed_running', 'consumed', 'exhausted')),
    seed_attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS plans_ready_idx ON plans(status, created_at);

CREATE TABLE IF NOT EXISTS seed_attempts (
    job_id TEXT NOT NULL REFERENCES rescue_jobs(job_id),
    attempt_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('running', 'failed')),
    reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY(job_id, attempt_id)
);

CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    details_json TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_type_action_entity_idx
ON events(entity_type, action, entity_id);
CREATE INDEX IF NOT EXISTS rescue_owner_task_idx
ON events(json_extract(details_json,'$.task_key'),action) WHERE entity_type='job';
CREATE INDEX IF NOT EXISTS plans_forgotten_idx ON events(entity_id)
WHERE entity_type='plan' AND action='manually_forgotten';
"""

@dataclass(frozen=True)
class SeedPlanClaim:
    """Fenced lease for one bounded guided-seed attempt in a plan generation."""

    plan_id: str
    job_id: str
    task_key: str
    plan_artifact_ref: str
    source_episode_id: str
    source_receipt_ref: str
    status: str
    seed_attempts: int
    created_at: float
    engine_epoch: str
    generation_id: str
    run_id: str
    attempt_id: str
    fence: int
    lease_expires_at: float
    attempt_number: int
    policy_version: int


def _now() -> float:
    return time.time()


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f"corrupt JSON in BEPA database: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("corrupt BEPA database: expected JSON object")
    return result


def _check_positive_int(value: int, field_name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        bound = "non-negative" if allow_zero else "positive"
        raise ContractError(f"{field_name} must be a {bound} integer")
    return value


def _cache_metadata_state(
    *,
    active: bool,
    age_updates: int,
    max_age_updates: int,
    reserved: int,
    committed: int,
    max_committed_uses: int,
    pending: bool,
) -> str:
    """Classify one cache from the same persisted limits used by admission."""
    if pending:
        return "pending"
    if not active:
        return "inactive"
    if age_updates < 0:
        return "future_policy"
    if age_updates + 1 > max_age_updates:
        return "blocked_age"
    if reserved + committed >= max_committed_uses:
        return "blocked_quota"
    return "metadata_eligible"


class BepaStore:
    """SQLite repository for one BEPA coordinator on one local machine."""

    def __init__(
        self,
        root: str | Path,
        *,
        busy_timeout_ms: int = 5000,
        max_age_updates: int = MAX_CACHE_AGE_UPDATES,
        max_committed_uses: int = MAX_CACHE_COMMITTED_USES,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        _check_positive_int(busy_timeout_ms, "busy_timeout_ms")
        _check_positive_int(max_age_updates, "max_age_updates")
        _check_positive_int(max_committed_uses, "max_committed_uses")
        self.db_path = self.root / "state.sqlite3"
        self.busy_timeout_ms = busy_timeout_ms
        self.max_age_updates = max_age_updates
        self.max_committed_uses = max_committed_uses
        self._write_lock = threading.RLock()
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._closed = False
        # Schema and WAL mode are initialized before other threads can use it.
        conn = self._new_connection()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        self._refresh_native_rescue_progress()

    def _refresh_native_rescue_progress(self):
        """Rebuild derived priority from immutable observations under the current rule."""
        with self._write() as conn:
            progress = {}
            for row in conn.execute(
                "SELECT * FROM native_rescue_observations ORDER BY rowid"
            ):
                key = (row["engine_epoch"], row["puzzle_key"])
                progress[key] = _advance_native_rescue_versions(
                    progress.get(key, []), row["policy_version"],
                    json.loads(row["native_exact_json"]),
                )
            for row in conn.execute("SELECT * FROM native_rescue_progress").fetchall():
                versions = progress.get((row["engine_epoch"], row["puzzle_key"]), [])
                encoded = canonical_json_bytes(versions).decode("utf-8")
                retired = int(len(versions) >= NATIVE_RESCUE_REQUIRED_SUCCESSES)
                if (encoded, retired) != (row["success_versions_json"], row["retired"]):
                    conn.execute(
                        "UPDATE native_rescue_progress SET success_versions_json=?,retired=?,updated_at=? "
                        "WHERE engine_epoch=? AND puzzle_key=?",
                        (encoded, retired, _now(), row["engine_epoch"], row["puzzle_key"]),
                    )

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=FULL")
        with self._write_lock:
            self._connections.append(conn)
        self._local.connection = conn
        return conn

    def _conn(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("BepaStore is closed")
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = self._new_connection()
        return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """Serialize short writes; callers must not await or run models inside."""
        with self._write_lock:
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.rollback()
                raise
            else:
                conn.commit()

    def close(self) -> None:
        """Close every thread-local connection owned by this repository."""
        with self._write_lock:
            if self._closed:
                return
            self._closed = True
            for conn in self._connections:
                conn.close()
            self._connections.clear()
            if hasattr(self._local, "connection"):
                del self._local.connection

    def __enter__(self) -> "BepaStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def checkpoint(self, *, truncate: bool = False) -> tuple[int, int, int]:
        """Run a bounded SQLite WAL checkpoint and return its SQLite result."""
        mode = "TRUNCATE" if truncate else "PASSIVE"
        row = self._conn().execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return tuple(int(value) for value in row)

    def _event(
        self,
        conn: sqlite3.Connection,
        entity_type: str,
        entity_id: str,
        action: str,
        *,
        reason: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO events(entity_type, entity_id, action, reason, details_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entity_type, entity_id, action, reason,
             canonical_json_bytes(dict(details)).decode("utf-8") if details is not None else None,
             _now()),
        )

    def _verify_artifact_ref(self, digest: str) -> Any:
        """Read and hash-check a published artifact before adding a DB reference."""
        return BepaArtifacts(self.root).read_json(digest)

    @staticmethod
    def _task_spec_from_json(payload: str) -> TaskSpec:
        data = _json_dict(payload)
        return TaskSpec(**data)

    def put_task(self, spec: TaskSpec, *, allowed: bool = True) -> TaskRecord:
        """Persist one full compatibility identity and allowlist bit."""
        now = _now()
        spec_json = canonical_json_bytes(spec.compatibility_payload()).decode("utf-8")
        with self._write() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_key=?", (spec.task_key,)).fetchone()
            if row is not None:
                if row["spec_json"] != spec_json or row["private_target_hash"] != spec.private_target_hash:
                    raise ConflictError("task key exists with different compatibility content")
                conn.execute("UPDATE tasks SET allowed=? WHERE task_key=?", (int(bool(allowed)), spec.task_key))
                created_at = float(row["created_at"])
            else:
                conn.execute(
                    "INSERT INTO tasks(task_key,spec_json,private_target_hash,allowed,created_at) VALUES(?,?,?,?,?)",
                    (spec.task_key, spec_json, spec.private_target_hash, int(bool(allowed)), now),
                )
                self._event(conn, "task", spec.task_key, "stored", details={"allowed": bool(allowed)})
                created_at = now
        return TaskRecord(spec.task_key, spec, bool(allowed), created_at)

    def task_registration_current(self, task_key: str, private_target_hash: str, *, allowed: bool) -> bool:
        """Verify existing durable identity before skipping a registration write."""
        row = self._conn().execute(
            "SELECT spec_json,private_target_hash,allowed FROM tasks WHERE task_key=?", (task_key,),
        ).fetchone()
        if row is None:
            return False
        if (hashlib.sha256(row['spec_json'].encode('utf-8')).hexdigest() != task_key
                or row['private_target_hash'] != private_target_hash):
            raise ConflictError('task key exists with different compatibility content')
        return bool(row['allowed']) == allowed

    def get_task(self, task_key: str) -> TaskSpec | None:
        row = self._conn().execute("SELECT spec_json FROM tasks WHERE task_key=?", (task_key,)).fetchone()
        return self._task_spec_from_json(row[0]) if row else None

    def get_task_record(self, task_key: str) -> TaskRecord | None:
        row = self._conn().execute("SELECT * FROM tasks WHERE task_key=?", (task_key,)).fetchone()
        if row is None:
            return None
        return TaskRecord(task_key, self._task_spec_from_json(row["spec_json"]), bool(row["allowed"]), float(row["created_at"]))

    def configure_budget_scope(
        self,
        scope_id: str,
        *,
        max_total_tokens: int,
        max_jobs: int | None = None,
    ) -> dict[str, Any]:
        """Create a persistent external-provider budget boundary."""
        _check_positive_int(max_total_tokens, "max_total_tokens", allow_zero=True)
        if max_jobs is not None:
            _check_positive_int(max_jobs, "max_jobs")
        if not scope_id:
            raise ContractError("scope_id must be non-empty")
        with self._write() as conn:
            row = conn.execute("SELECT * FROM budget_scopes WHERE scope_id=?", (scope_id,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO budget_scopes(scope_id,max_total_tokens,max_jobs,created_at) VALUES(?,?,?,?)",
                    (scope_id, max_total_tokens, max_jobs, _now()),
                )
                self._event(conn, "budget_scope", scope_id, "configured", details={"max_total_tokens": max_total_tokens, "max_jobs": max_jobs})
            elif row["max_jobs"] != max_jobs:
                raise ConflictError("budget scope already exists with a different job-count limit")
            # Once created, the database cap is authoritative. Runtime cap changes
            # are persisted through update_budget_limits and must survive restarts.
        return self.get_budget_scope(scope_id)

    def latest_max_job_tokens(self, scope_id: str) -> int | None:
        rows = self._conn().execute(
            "SELECT details_json FROM events WHERE entity_type='budget_scope' AND entity_id=? "
            "AND action='max_job_tokens_updated' ORDER BY sequence DESC LIMIT 1", (scope_id,)
        ).fetchall()
        if not rows:
            return None
        try:
            details = json.loads(rows[0]["details_json"])
        except (TypeError, json.JSONDecodeError):
            return None
        value = details.get("max_job_tokens") if isinstance(details, dict) else None
        return value if type(value) is int and value > 0 else None

    def get_budget_scope(self, scope_id: str) -> dict[str, Any]:
        row = self._conn().execute("SELECT * FROM budget_scopes WHERE scope_id=?", (scope_id,)).fetchone()
        if row is None:
            raise KeyError(scope_id)
        return dict(row)

    def update_budget_limits(
        self, scope_id: str, *, max_job_tokens: int | None = None,
        max_total_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Update current-scope limits without altering settled use or active grants."""
        if max_job_tokens is None and max_total_tokens is None:
            raise ContractError("at least one budget limit is required")
        if max_job_tokens is not None:
            _check_positive_int(max_job_tokens, "max_job_tokens")
        if max_total_tokens is not None:
            _check_positive_int(max_total_tokens, "max_total_tokens")
        now = _now()
        with self._write() as conn:
            row = conn.execute("SELECT * FROM budget_scopes WHERE scope_id=?", (scope_id,)).fetchone()
            if row is None:
                raise KeyError(scope_id)
            new_total = int(row["max_total_tokens"]) if max_total_tokens is None else max_total_tokens
            required = int(row["used_tokens"]) + int(row["reserved_tokens"])
            if new_total < required:
                raise ContractError("max_total_tokens cannot be below settled plus reserved usage")
            if max_total_tokens is not None:
                conn.execute("UPDATE budget_scopes SET max_total_tokens=? WHERE scope_id=?",
                             (new_total, scope_id))
            queued_updated = 0
            if max_job_tokens is not None:
                queued_updated = conn.execute(
                    "UPDATE rescue_jobs SET max_tokens=?,updated_at=? "
                    "WHERE budget_scope=? AND state='queued' AND used_tokens=0",
                    (max_job_tokens, now, scope_id),
                ).rowcount
            if max_job_tokens is not None:
                self._event(conn, "budget_scope", scope_id, "max_job_tokens_updated", details={
                    "max_job_tokens": max_job_tokens,
                    "queued_jobs_updated": int(queued_updated), "active_grants_changed": False,
                })
            if max_total_tokens is not None:
                self._event(conn, "budget_scope", scope_id, "max_total_tokens_updated", details={
                    "max_total_tokens": max_total_tokens,
                    "settled_usage_changed": False,
                })
        return {**self.budget_snapshot(scope_id),
                "max_job_tokens": max_job_tokens,
                "queued_jobs_updated": int(queued_updated),
                "active_grants_changed": False}

    def retry_unstarted_budget_jobs(self, scope_id: str, task_keys: Sequence[str]) -> int:
        """Requeue budget-only failures after funding, never replay a provider call."""
        allowed = set(task_keys)
        if not allowed:
            return 0
        with self._write() as conn:
            scope = conn.execute("SELECT * FROM budget_scopes WHERE scope_id=?", (scope_id,)).fetchone()
            if scope is None:
                raise KeyError(scope_id)
            if scope['usage_unknown']:
                return 0
            available = scope['max_total_tokens'] - scope['used_tokens'] - scope['reserved_tokens']
            rows = conn.execute(
                "SELECT job_id,task_key FROM rescue_jobs j WHERE budget_scope=? "
                "AND state='failed' AND fail_reason='token_budget_exhausted' "
                "AND used_tokens=0 AND usage_unknown=0 AND reserved_tokens=0 "
                "AND attempts<max_attempts AND max_tokens>0 AND max_tokens<=? "
                "AND NOT EXISTS (SELECT 1 FROM job_mutations m "
                "WHERE m.job_id=j.job_id AND m.action='provider_receipt')",
                (scope_id, available),
            ).fetchall()
            count = 0
            for row in rows:
                if row['task_key'] not in allowed:
                    continue
                conn.execute("UPDATE rescue_jobs SET state='queued',fail_reason=NULL,updated_at=? WHERE job_id=?",
                             (_now(), row['job_id']))
                self._event(conn, 'job', row['job_id'], 'queued', reason='unstarted_budget_retry',
                            details={'attempt_counter_reset': False, 'provider_calls_replayed': False})
                count += 1
            return count

    def budget_snapshot(self, scope_id: str) -> dict[str, Any]:
        """Return the cap, consumed/reserved tokens, and whether usage is unknown."""
        row = self.get_budget_scope(scope_id)
        result = {
            "scope_id": scope_id,
            "max_total_tokens": int(row["max_total_tokens"]),
            "used_tokens": int(row["used_tokens"]),
            "reserved_tokens": int(row["reserved_tokens"]),
            "remaining_tokens": max(0, int(row["max_total_tokens"]) - int(row["used_tokens"]) - int(row["reserved_tokens"])),
            "usage_unknown": bool(row["usage_unknown"]),
            "max_jobs": row["max_jobs"],
        }
        result.update(self._lower_bound_status(scope_id))
        return result

    def _lower_bound_status(self, scope_id: str) -> dict[str, Any]:
        count = int(self._conn().execute(
            "SELECT COUNT(*) FROM rescue_jobs j WHERE budget_scope=? AND usage_unknown=1 "
            "AND state IN ('failed','source_submitted','source_verified','plan_ready','cache_ready') AND EXISTS (SELECT 1 FROM events e WHERE e.entity_type='job' "
            "AND e.entity_id=j.job_id AND e.action='usage_lower_bound_settled')", (scope_id,),
        ).fetchone()[0])
        return ({"accounting_basis": "confirmed_lower_bound", "lower_bound_jobs": count,
                 "provider_usage_known": False} if count else {})

    @staticmethod
    def _checkpoint_reset_state(conn: sqlite3.Connection, scope_id: str) -> dict[str, Any]:
        tables = (
            "rescue_jobs", "teacher_claim_requests", "job_mutations", "usage_events",
            "episodes", "cache_entries", "reservations", "update_attempts", "plans",
            "seed_attempts",
        )
        scopes = [dict(row) for row in conn.execute(
            "SELECT scope_id,max_total_tokens,max_jobs,used_tokens,reserved_tokens,usage_unknown "
            "FROM budget_scopes ORDER BY scope_id"
        ).fetchall()]
        counts = {
            table: int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in tables
        }
        counts["unexpected_events"] = int(conn.execute(
            "SELECT count(*) FROM events WHERE NOT "
            "((entity_type='task' AND action='stored') OR "
            "(entity_type='budget_scope' AND entity_id=? AND action='configured'))",
            (scope_id,),
        ).fetchone()[0])
        counter = conn.execute("SELECT value FROM counters WHERE name='ready_sequence'").fetchone()
        ready_sequence = int(counter[0]) if counter is not None else -1
        return {"scopes": scopes, "counts": counts, "ready_sequence": ready_sequence}

    def write_checkpoint_reset_receipt(self, scope_id: str, receipt: Mapping[str, Any]) -> None:
        """Atomically attest a checkpoint against a pristine, single-scope ledger."""
        action = "checkpoint_reset_authorized"
        if (not isinstance(receipt, Mapping)
                or receipt.get("schema") != "qwen-rl-bepa-checkpoint-reset/v1"
                or set(receipt) != {"schema", "checkpoint", "budget_scope", "reset_state"}):
            raise ContractError("BEPA checkpoint reset receipt schema is invalid")
        with self._write() as conn:
            state = self._checkpoint_reset_state(conn, scope_id)
            scopes = state["scopes"]
            expected_scope = ({"scope_id": scopes[0]["scope_id"],
                               "max_total_tokens": int(scopes[0]["max_total_tokens"]),
                               "max_jobs": scopes[0]["max_jobs"]}
                              if len(scopes) == 1 else None)
            expected_reset_state = {
                "used_tokens": 0, "reserved_tokens": 0, "usage_unknown": False,
                "row_counts": {name: 0 for name in state["counts"]},
                "ready_sequence": 0,
            }
            if (len(scopes) != 1 or scopes[0]["scope_id"] != scope_id
                    or scopes[0]["used_tokens"] != 0 or scopes[0]["reserved_tokens"] != 0
                    or scopes[0]["usage_unknown"] != 0
                    or any(state["counts"].values())
                    or state["ready_sequence"] != 0
                    or receipt.get("budget_scope") != expected_scope
                    or receipt.get("reset_state") != expected_reset_state):
                raise ContractError("BEPA checkpoint reset receipt requires a pristine single-scope ledger")
            if conn.execute(
                "SELECT 1 FROM events WHERE entity_type='budget_scope' AND entity_id=? AND action=? LIMIT 1",
                (scope_id, action),
            ).fetchone() is not None:
                raise ContractError("BEPA checkpoint reset receipt already exists for this scope")
            self._event(conn, "budget_scope", scope_id, action, details=dict(receipt))

    def checkpoint_reset_receipt(self, scope_id: str) -> dict[str, Any] | None:
        rows = self._conn().execute(
            "SELECT details_json FROM events WHERE entity_type='budget_scope' AND entity_id=? "
            "AND action='checkpoint_reset_authorized' ORDER BY sequence",
            (scope_id,),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise ContractError("BEPA ledger has multiple checkpoint reset receipts for one scope")
        try:
            value = json.loads(rows[0]["details_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractError("BEPA checkpoint reset receipt is malformed") from exc
        if not isinstance(value, dict):
            raise ContractError("BEPA checkpoint reset receipt must be an object")
        return value

    def enqueue_job(
        self,
        task_key: str,
        *,
        public_job_key: str,
        budget_scope: str,
        max_attempts: int,
        max_tokens: int,
    ) -> JobRecord:
        """Deduplicate a rescue job inside one durable budget scope."""
        _check_positive_int(max_attempts, "max_attempts")
        _check_positive_int(max_tokens, "max_tokens", allow_zero=True)
        if not public_job_key:
            raise ContractError("public_job_key must be non-empty")
        now = _now()
        with self._write() as conn:
            task = conn.execute("SELECT allowed FROM tasks WHERE task_key=?", (task_key,)).fetchone()
            if task is None:
                raise KeyError(task_key)
            if not task["allowed"]:
                raise ContractError("cannot enqueue a rescue job outside the training allowlist")
            scope = conn.execute("SELECT * FROM budget_scopes WHERE scope_id=?", (budget_scope,)).fetchone()
            if scope is None:
                raise ContractError("configure the persistent budget scope before enqueueing jobs")
            row = conn.execute(
                "SELECT * FROM rescue_jobs WHERE budget_scope=? AND public_job_key=?",
                (budget_scope, public_job_key),
            ).fetchone()
            if row is not None:
                if (row["task_key"], row["max_attempts"]) != (task_key, max_attempts):
                    raise ConflictError("dedupe key exists with different task or attempt contract")
                return self._job_record(row)
            if scope["usage_unknown"]:
                raise BudgetUnavailableError(budget_scope, "usage_unknown")
            if scope["used_tokens"] >= scope["max_total_tokens"]:
                raise BudgetUnavailableError(budget_scope, "exhausted")
            if scope["max_jobs"] is not None:
                count = conn.execute("SELECT count(*) FROM rescue_jobs WHERE budget_scope=?", (budget_scope,)).fetchone()[0]
                if count >= scope["max_jobs"]:
                    raise ContractError("provider job-count budget is exhausted")
            job_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO rescue_jobs(job_id,task_key,public_job_key,budget_scope,state,max_attempts,max_tokens,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (job_id, task_key, public_job_key, budget_scope, "queued", max_attempts, max_tokens, now, now),
            )
            self._event(conn, "job", job_id, "queued", details={"task_key": task_key, "budget_scope": budget_scope})
        return self.get_job(job_id)

    def bind_rescue_owner(
        self,
        job_id: str,
        *,
        task_key: str,
        worker_id: str,
        executor_id: str,
        original_group_id: str | None = None,
    ) -> dict[str, str]:
        """Bind the executing TaskSpec; the Teacher job keeps its source identity."""
        if (not isinstance(worker_id, str) or not worker_id.strip()
                or not isinstance(executor_id, str) or not executor_id.strip()):
            raise ContractError("rescue owner requires non-empty Worker and executor IDs")
        owner = {"worker_id": worker_id, "executor_id": executor_id}
        if original_group_id is not None and (
            not isinstance(original_group_id, str) or not original_group_id.strip()
        ):
            raise ContractError("original_group_id must be a non-empty string when provided")
        with self._write() as conn:
            job = conn.execute(
                "SELECT task_key FROM rescue_jobs WHERE job_id=?", (job_id,),
            ).fetchone()
            if job is None:
                raise KeyError(job_id)
            target = conn.execute("SELECT * FROM tasks WHERE task_key=?", (task_key,)).fetchone()
            source = conn.execute("SELECT * FROM tasks WHERE task_key=?", (job["task_key"],)).fetchone()
            if target is None or not target["allowed"]:
                raise ContractError("rescue execution task is outside the training allowlist")
            if (source is None or source["private_target_hash"] != target["private_target_hash"]
                    or _json_dict(source["spec_json"])["public_challenge"]
                    != _json_dict(target["spec_json"])["public_challenge"]):
                raise ContractError("rescue execution task differs from the Teacher puzzle")
            history = self._rescue_owner_history(conn, task_key)
            if history is not None and history["owner"] != owner:
                raise ConflictError("rescue task is already bound to a different original Worker/executor")
            if history is None:
                self._event(
                    conn,
                    "job",
                    job_id,
                    "rescue_owner_bound",
                    details={"task_key": task_key, **owner},
                )
            if original_group_id is not None:
                prior_groups = conn.execute(
                    "SELECT details_json FROM events WHERE entity_type='job' "
                    "AND json_extract(details_json,'$.task_key')=? "
                    "AND action='rescue_origin_group_bound' ORDER BY sequence",
                    (task_key,),
                ).fetchall()
                group_ids = {
                    _json_dict(row["details_json"]).get("original_group_id")
                    for row in prior_groups
                }
                if original_group_id not in group_ids:
                    self._event(
                        conn,
                        "job",
                        job_id,
                        "rescue_origin_group_bound",
                        details={
                            "task_key": task_key,
                            "worker_id": worker_id,
                            "executor_id": executor_id,
                            "original_group_id": original_group_id,
                        },
                    )
        return owner

    @staticmethod
    def _rescue_owner_history(conn: sqlite3.Connection, task_key: str) -> dict[str, Any] | None:
        rows = conn.execute(
            "SELECT sequence,action,details_json FROM events WHERE entity_type='job' "
            "AND json_extract(details_json,'$.task_key')=? "
            "AND action IN ('rescue_owner_bound','rescue_origin_group_bound',?) "
            "ORDER BY sequence",
            (task_key, OWNER_MIGRATION_ACTION),
        ).fetchall()
        events = [{
            "sequence": int(row["sequence"]),
            "action": row["action"],
            "details": _json_dict(row["details_json"]),
        } for row in rows]
        try:
            return fold_owner_events(events)
        except OwnerHistoryError as exc:
            raise ConflictError(str(exc)) from exc

    def append_rescue_owner_migrations(
        self,
        *,
        migration_id: str,
        run_id: str,
        lineage_id: str,
        snapshot_id: str,
        reason: str,
        mappings: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        """Append an audited owner migration without rewriting owner or budget history."""
        audit = {"migration_id": migration_id, "run_id": run_id,
                 "lineage_id": lineage_id, "snapshot_id": snapshot_id}
        if any(not isinstance(value, str) or not value or value.strip() != value
               for value in audit.values()):
            raise ContractError("owner migration audit identity fields must be non-empty strings")
        if not isinstance(reason, str) or not reason.strip():
            raise ContractError("owner migration requires a non-empty reason")
        if isinstance(mappings, (str, bytes)) or not isinstance(mappings, Sequence) or not mappings:
            raise ContractError("owner migration mappings must be a non-empty sequence")
        normalized = []
        seen_keys: set[str] = set()
        seen_jobs: set[str] = set()
        seen_task_ids: set[str] = set()
        required = {
            "job_id", "task_key", "task_id", "from_worker_id", "from_executor_id",
            "to_worker_id", "to_executor_id",
        }
        for row in mappings:
            if not isinstance(row, Mapping) or set(row) != required:
                raise ContractError("owner migration mapping has an invalid shape")
            values = dict(row)
            if any(not isinstance(value, str) or not value or value.strip() != value
                   for value in values.values()):
                raise ContractError("owner migration mapping fields must be non-empty strings")
            if values["task_key"] in seen_keys or values["job_id"] in seen_jobs:
                raise ContractError("owner migration task keys and job IDs must be unique")
            if not is_safe_task_id(values["task_id"]):
                raise ContractError("owner migration task_id is invalid")
            if values["task_id"] in seen_task_ids:
                raise ContractError("owner migration checkpoint task IDs must be unique")
            seen_keys.add(values["task_key"])
            seen_jobs.add(values["job_id"])
            seen_task_ids.add(values["task_id"])
            if ((values["from_worker_id"], values["from_executor_id"])
                    == (values["to_worker_id"], values["to_executor_id"])):
                raise ContractError("owner migration must change the Worker/executor pair")
            normalized.append(values)
        normalized.sort(key=lambda row: (row["task_key"], row["job_id"]))

        expected_details = [{
            "schema": OWNER_MIGRATION_SCHEMA,
            **audit,
            **mapping,
            "reason": reason.strip(),
        } for mapping in normalized]
        with self._write() as conn:
            existing_rows = conn.execute(
                "SELECT details_json FROM events WHERE entity_type='job' "
                "AND action=? AND json_extract(details_json,'$.migration_id')=? "
                "ORDER BY sequence",
                (OWNER_MIGRATION_ACTION, migration_id),
            ).fetchall()
            if existing_rows:
                existing = [_json_dict(row["details_json"]) for row in existing_rows]
                if existing == expected_details:
                    return {"status": "already_applied", **audit,
                            "binding_count": len(existing)}
                raise ConflictError("owner migration ID is already used with different contents")

            for mapping in normalized:
                job = conn.execute(
                    "SELECT task_key FROM rescue_jobs WHERE job_id=?", (mapping["job_id"],),
                ).fetchone()
                task = conn.execute(
                    "SELECT allowed FROM tasks WHERE task_key=?", (mapping["task_key"],),
                ).fetchone()
                if job is None or job["task_key"] != mapping["task_key"]:
                    raise ContractError("owner migration job does not name its TaskSpec key")
                if task is None or not task["allowed"]:
                    raise ContractError("owner migration TaskSpec is outside the training allowlist")
                history = self._rescue_owner_history(conn, mapping["task_key"])
                source = {"worker_id": mapping["from_worker_id"],
                          "executor_id": mapping["from_executor_id"]}
                if history is None or history["owner"] != source:
                    raise ConflictError("owner migration source differs from the current BEPA task owner")

            for mapping, details in zip(normalized, expected_details, strict=True):
                self._event(
                    conn, "job", mapping["job_id"], OWNER_MIGRATION_ACTION,
                    reason=reason.strip(), details=details,
                )
        return {"status": "applied", **audit, "binding_count": len(normalized)}

    def rescue_owner_for_task(self, task_key: str) -> dict[str, Any] | None:
        """Read the effective owner by replaying immutable binding and migration events."""
        history = self._rescue_owner_history(self._conn(), task_key)
        if history is None:
            return None
        result = {**history["owner"], "original_group_ids": history["original_group_ids"]}
        if history["owner_migrations"]:
            result["original_owner"] = history["initial_owner"]
            result["owner_migrations"] = history["owner_migrations"]
        return result

    def teacher_pending_task_aliases(self, task_keys: Sequence[str], *, budget_scope: str) -> dict[str, str]:
        """Bind queued Teacher text work to an identical current public puzzle.

        Preserve original task/job records. Every new source uses current tools
        and the central private verifier; tokenized replay keeps full task keys.
        """
        keys = tuple(dict.fromkeys(task_keys))
        if not keys:
            return {}
        rows = self._conn().execute(
            "SELECT DISTINCT origin.task_key AS origin_key,target.task_key AS target_key "
            "FROM rescue_jobs j JOIN tasks origin ON origin.task_key=j.task_key "
            "JOIN tasks target ON target.private_target_hash=origin.private_target_hash "
            "AND json_extract(target.spec_json,'$.public_challenge')="
            "json_extract(origin.spec_json,'$.public_challenge') "
            f"WHERE target.allowed=1 AND target.task_key IN ({','.join('?' for _ in keys)}) "
            "AND origin.allowed=1 AND origin.task_key!=target.task_key "
            "AND j.budget_scope=? AND j.state IN ('queued','retry_wait','teacher_running') "
            "ORDER BY origin.task_key,target.task_key",
            (*keys, budget_scope),
        ).fetchall()
        aliases: dict[str, str] = {}
        for row in rows:
            aliases.setdefault(row['origin_key'], row['target_key'])
        return aliases

    def claim_job(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
        task_allowlist: Sequence[str],
        claim_id: str | None = None,
        single_active_worker: bool = False,
        budget_scope: str | None = None,
    ) -> JobLease | None:
        """Claim work; retrying a claim_id replays its original unexpired lease."""
        if budget_scope is not None and (not isinstance(budget_scope, str) or not budget_scope):
            raise ContractError("budget_scope must be a non-empty string")
        if not worker_id or not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise ContractError("worker_id and positive lease_seconds are required")
        if (claim_id is not None
                and (not isinstance(claim_id, str) or len(claim_id) != 32
                     or any(char not in "0123456789abcdef" for char in claim_id))):
            raise ContractError("claim_id must be 32 lowercase hexadecimal characters")
        if type(single_active_worker) is not bool:
            raise ContractError("single_active_worker must be a boolean")
        if isinstance(task_allowlist, (str, bytes)):
            raise ContractError("task_allowlist must contain non-empty task keys")
        try:
            task_keys = tuple(task_allowlist)
        except TypeError as exc:
            raise ContractError("task_allowlist must contain non-empty task keys") from exc
        if any(not isinstance(key, str) or not key for key in task_keys):
            raise ContractError("task_allowlist must contain non-empty task keys")
        now = _now()
        with self._write() as conn:
            # The Teacher retries one transport-ambiguous POST with the same ID.
            # Replay that exact lease instead of claiming a second job.
            if claim_id is not None:
                prior = conn.execute(
                    "SELECT worker_id,job_id,fence FROM teacher_claim_requests WHERE claim_id=?",
                    (claim_id,),
                ).fetchone()
                if prior is not None:
                    if prior["worker_id"] != worker_id:
                        raise ConflictError("claim_id was reused by a different Teacher worker")
                    job = conn.execute(
                        "SELECT j.*,t.spec_json FROM rescue_jobs j JOIN tasks t USING(task_key) "
                        "WHERE j.job_id=?",
                        (prior["job_id"],),
                    ).fetchone()
                    if job is not None and budget_scope is not None and job["budget_scope"] != budget_scope:
                        raise ConflictError("claim_id belongs to another budget scope")
                    if (job is not None and job["state"] == "teacher_running"
                            and job["lease_owner"] == worker_id
                            and int(job["fence"]) == int(prior["fence"])
                            and job["lease_expires_at"] is not None
                            and float(job["lease_expires_at"]) > now):
                        spec = self._task_spec_from_json(job["spec_json"])
                        return JobLease(
                            job_id=job["job_id"], task_key=job["task_key"],
                            budget_scope=job["budget_scope"], worker_id=worker_id,
                            fence=int(prior["fence"]),
                            lease_expires_at=float(job["lease_expires_at"]),
                            attempt_number=int(job["attempts"]),
                            public_task=spec.public_claim_payload(),
                        )
                    # A lease whose response was lost and then expired is
                    # ambiguous. Fence it as unknown and never reuse this ID.
                    if (job is not None and job["state"] == "teacher_running"
                            and job["lease_owner"] == worker_id
                            and int(job["fence"]) == int(prior["fence"])
                            and (job["lease_expires_at"] is None
                                 or float(job["lease_expires_at"]) <= now)):
                        conn.execute(
                            "UPDATE rescue_jobs SET state='outcome_unknown',usage_unknown=1,"
                            "lease_owner=NULL,lease_expires_at=NULL,"
                            "fail_reason='lease_expired_outcome_unknown',updated_at=? WHERE job_id=?",
                            (now, job["job_id"]),
                        )
                        conn.execute(
                            "UPDATE budget_scopes SET usage_unknown=1 WHERE scope_id=?",
                            (job["budget_scope"],),
                        )
                        self._event(
                            conn, "job", job["job_id"], "outcome_unknown",
                            reason="lease_expired_outcome_unknown",
                            details={"claim_id": claim_id, "fence": int(prior["fence"])},
                        )
                    return None

            # An expired paid-provider lease is ambiguous. Do not silently retry it.
            expired = conn.execute(
                "SELECT job_id,budget_scope FROM rescue_jobs WHERE state='teacher_running' AND lease_expires_at<=?",
                (now,),
            ).fetchall()
            for row in expired:
                conn.execute(
                    "UPDATE rescue_jobs SET state='outcome_unknown',usage_unknown=1,lease_owner=NULL,lease_expires_at=NULL,"
                    "fail_reason='lease_expired_outcome_unknown',updated_at=? WHERE job_id=?",
                    (now, row["job_id"]),
                )
                conn.execute("UPDATE budget_scopes SET usage_unknown=1 WHERE scope_id=?", (row["budget_scope"],))
                self._event(conn, "job", row["job_id"], "outcome_unknown", reason="lease_expired_outcome_unknown")

            if single_active_worker:
                active = conn.execute(
                    "SELECT 1 FROM rescue_jobs WHERE state='teacher_running' "
                    "AND lease_expires_at>? LIMIT 1",
                    (now,),
                ).fetchone()
                if active is not None:
                    return None

            if task_keys:
                rows = conn.execute(
                    "SELECT j.*,t.spec_json,t.allowed,b.max_total_tokens,b.used_tokens AS scope_used_tokens,"
                    "b.reserved_tokens AS scope_reserved_tokens,b.usage_unknown AS scope_usage_unknown,b.max_jobs "
                    "FROM rescue_jobs j JOIN tasks t USING(task_key) JOIN budget_scopes b ON b.scope_id=j.budget_scope "
                    "WHERE j.state IN ('queued','retry_wait') "
                    f"AND j.task_key IN ({','.join('?' for _ in task_keys)}) "
                    "AND (? IS NULL OR j.budget_scope=?) "
                    "ORDER BY j.created_at,j.job_id",
                    (*task_keys, budget_scope, budget_scope),
                ).fetchall()
            else:
                rows = []
            for row in rows:
                if not row["allowed"] or row["usage_unknown"] or row["scope_usage_unknown"]:
                    continue
                if row["used_tokens"] >= row["max_tokens"]:
                    conn.execute("UPDATE rescue_jobs SET state='exhausted',updated_at=? WHERE job_id=?", (now, row["job_id"]))
                    self._event(conn, "job", row["job_id"], "exhausted", reason="token_budget_exhausted")
                    continue
                if row["attempts"] >= row["max_attempts"]:
                    conn.execute("UPDATE rescue_jobs SET state='exhausted',updated_at=? WHERE job_id=?", (now, row["job_id"]))
                    self._event(conn, "job", row["job_id"], "exhausted", reason="attempt_budget_exhausted")
                    continue
                available_total = int(row["max_total_tokens"]) - int(row["scope_used_tokens"]) - int(row["scope_reserved_tokens"])
                if available_total <= 0 and int(row["scope_used_tokens"]) >= int(row["max_total_tokens"]):
                    conn.execute("UPDATE rescue_jobs SET state='exhausted',updated_at=? WHERE job_id=?", (now, row["job_id"]))
                    self._event(conn, "job", row["job_id"], "exhausted", reason="total_token_budget_exhausted")
                    continue
                job_remaining = int(row["max_tokens"]) - int(row["used_tokens"])
                budget_grant = min(available_total, job_remaining)
                if budget_grant <= 0 or int(row["reserved_tokens"]) != 0:
                    continue
                fence = int(row["fence"]) + 1
                expires = now + float(lease_seconds)
                attempts = int(row["attempts"]) + 1
                conn.execute(
                    "UPDATE rescue_jobs SET state='teacher_running',attempts=?,reserved_tokens=?,fence=?,lease_owner=?,lease_expires_at=?,updated_at=? WHERE job_id=?",
                    (attempts, budget_grant, fence, worker_id, expires, now, row["job_id"]),
                )
                conn.execute("UPDATE budget_scopes SET reserved_tokens=reserved_tokens+? WHERE scope_id=?",
                             (budget_grant, row["budget_scope"]))
                self._event(conn, "job", row["job_id"], "claimed", details={"worker_id": worker_id, "fence": fence,
                                                                               "budget_tokens": budget_grant,
                                                                               "claim_id": claim_id,
                                                                               "used_tokens_before_attempt": int(row["used_tokens"])})
                spec = self._task_spec_from_json(row["spec_json"])
                if claim_id is not None:
                    conn.execute(
                        "INSERT INTO teacher_claim_requests(claim_id,worker_id,job_id,fence,created_at) "
                        "VALUES(?,?,?,?,?)",
                        (claim_id, worker_id, row["job_id"], fence, now),
                    )
                return JobLease(
                    job_id=row["job_id"], task_key=row["task_key"], budget_scope=row["budget_scope"],
                    worker_id=worker_id, fence=fence, lease_expires_at=expires,
                    attempt_number=attempts, public_task=spec.public_claim_payload(),
                )
            return None

    def heartbeat_job(self, job_id: str, *, worker_id: str, fence: int, lease_seconds: float) -> JobLease:
        if lease_seconds <= 0:
            raise ContractError("lease_seconds must be positive")
        now = _now()
        with self._write() as conn:
            row = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
            self._assert_current_lease(row, worker_id, fence, now)
            expires = now + float(lease_seconds)
            conn.execute("UPDATE rescue_jobs SET lease_expires_at=?,updated_at=? WHERE job_id=?", (expires, now, job_id))
            self._event(conn, "job", job_id, "heartbeat", details={"fence": fence})
            task = conn.execute("SELECT spec_json FROM tasks WHERE task_key=?", (row["task_key"],)).fetchone()
            return JobLease(job_id, row["task_key"], row["budget_scope"], worker_id, fence, expires,
                            int(row["attempts"]), self._task_spec_from_json(task[0]).public_claim_payload())

    @staticmethod
    def _assert_current_lease(row: sqlite3.Row | None, worker_id: str, fence: int, now: float) -> None:
        if row is None or row["state"] != "teacher_running":
            raise LeaseLostError("job is not in teacher_running state")
        if row["lease_owner"] != worker_id or int(row["fence"]) != int(fence):
            raise LeaseLostError("worker or generation fence does not match the current lease")
        if row["lease_expires_at"] is None or float(row["lease_expires_at"]) <= now:
            raise LeaseLostError("job lease has expired")

    def _record_job_usage(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        tokens: int | None,
        idempotency_key: str,
        usage_is_total: bool = False,
        attempt_fence: int | None = None,
    ) -> None:
        if not idempotency_key:
            raise ContractError("usage idempotency_key must be non-empty")
        if tokens is not None:
            _check_positive_int(tokens, "usage tokens", allow_zero=True)
            if usage_is_total:
                base = 0
                if attempt_fence is not None:
                    for claim in conn.execute(
                        "SELECT details_json FROM events WHERE entity_type='job' AND entity_id=? "
                        "AND action='claimed' ORDER BY sequence DESC", (row['job_id'],)):
                        details = json.loads(claim[0])
                        if details.get('fence') == attempt_fence:
                            base = int(details.get('used_tokens_before_attempt', 0))
                            break
                tokens -= int(row['used_tokens']) - base
                if tokens < 0:
                    raise ConflictError('final provider total is below already confirmed usage')
        old = conn.execute(
            "SELECT job_id,tokens FROM usage_events WHERE scope_id=? AND idempotency_key=?",
            (row["budget_scope"], idempotency_key),
        ).fetchone()
        if old is not None:
            if old["job_id"] != row["job_id"] or old["tokens"] != tokens:
                raise ConflictError("usage idempotency key was reused with different data")
            return
        conn.execute(
            "INSERT INTO usage_events(scope_id,idempotency_key,job_id,tokens,created_at) VALUES(?,?,?,?,?)",
            (row["budget_scope"], idempotency_key, row["job_id"], tokens, _now()),
        )
        if tokens is None:
            conn.execute(
                "UPDATE rescue_jobs SET usage_unknown=1 WHERE job_id=?", (row["job_id"],)
            )
            conn.execute("UPDATE budget_scopes SET usage_unknown=1 WHERE scope_id=?", (row["budget_scope"],))
        else:
            covered = min(int(row["reserved_tokens"]), tokens)
            conn.execute(
                "UPDATE rescue_jobs SET used_tokens=used_tokens+?,reserved_tokens=MAX(0,reserved_tokens-?) WHERE job_id=?",
                (tokens, covered, row["job_id"])
            )
            conn.execute(
                "UPDATE budget_scopes SET used_tokens=used_tokens+?,reserved_tokens=MAX(0,reserved_tokens-?) WHERE scope_id=?",
                (tokens, covered, row["budget_scope"])
            )

    @staticmethod
    def _release_known_job_budget(conn: sqlite3.Connection, job_id: str) -> int:
        """Return unused grant after a known provider outcome."""
        row = conn.execute("SELECT budget_scope,reserved_tokens FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        held = int(row["reserved_tokens"])
        if held:
            conn.execute("UPDATE rescue_jobs SET reserved_tokens=0 WHERE job_id=?", (job_id,))
            conn.execute("UPDATE budget_scopes SET reserved_tokens=MAX(0,reserved_tokens-?) WHERE scope_id=?",
                         (held, row["budget_scope"]))
        return held

    def record_usage(self, job_id: str, *, tokens: int | None, idempotency_key: str) -> dict[str, Any]:
        """Persist provider usage; missing usage blocks all further paid calls."""
        with self._write() as conn:
            row = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            self._record_job_usage(conn, row, tokens=tokens, idempotency_key=idempotency_key)
            self._event(conn, "job", job_id, "usage_recorded", reason="unknown" if tokens is None else None,
                        details={"tokens": tokens, "idempotency_key": idempotency_key})
        return self.get_budget_scope(self.get_job(job_id).budget_scope)

    @staticmethod
    def _unresolved_unknown_jobs(conn, scope_id: str) -> int:
        return int(conn.execute(
            "SELECT COUNT(*) FROM rescue_jobs j WHERE budget_scope=? "
            "AND (usage_unknown=1 OR state='outcome_unknown') "
            "AND NOT (state IN ('failed','source_submitted','source_verified','plan_ready','cache_ready') AND EXISTS (SELECT 1 FROM events e "
            "WHERE e.entity_type='job' AND e.entity_id=j.job_id "
            "AND e.action='usage_lower_bound_settled'))", (scope_id,),
        ).fetchone()[0])

    def settle_unknown_lower_bound(self, job_id: str, *, evidence_ref: str,
                                   fence: int, reason: str, acceptance_source: str = "operator", preserve_completed: bool = False) -> dict[str, Any]:
        """Explicitly accept unreported costs; retain unknown receipts and charge only known usage."""
        require_sha256(evidence_ref, "evidence_ref")
        if acceptance_source not in {"operator", "configured_policy"}:
            raise ContractError("unknown lower-bound acceptance source")
        if type(fence) is not int or fence < 1 or not isinstance(reason, str) or not reason.strip():
            raise ContractError("lower-bound settlement requires a fence and operator reason")
        with self._write() as conn:
            row = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            scope_id = row["budget_scope"]
            prior = conn.execute(
                "SELECT details_json FROM events WHERE entity_type='job' AND entity_id=? "
                "AND action='usage_lower_bound_settled' ORDER BY sequence DESC LIMIT 1", (job_id,),
            ).fetchone()
            if prior is not None:
                details = json.loads(prior["details_json"])
                if (details["evidence_ref"] != evidence_ref or details["fence"] != fence
                        or details["operator_reason"] != reason
                        or details.get("acceptance_source", "operator") != acceptance_source):
                    raise ConflictError("lower-bound settlement already has different evidence or approval")
                charged = details["charged_tokens"]
                released = details["released_reserved_tokens"]
            else:
                allowed = ('source_submitted', 'source_verified', 'plan_ready', 'cache_ready') if preserve_completed else ('outcome_unknown',)
                if (row["state"] not in allowed or not row["usage_unknown"]
                        or row["lease_owner"] is not None or row["lease_expires_at"] is not None
                        or row["fence"] != fence):
                    raise ConflictError("lower-bound settlement requires the unleased unknown attempt")
                charged = int(conn.execute(
                    "SELECT COALESCE(SUM(tokens),0) FROM usage_events WHERE job_id=? AND tokens IS NOT NULL",
                    (job_id,),
                ).fetchone()[0])
                if charged != int(row["used_tokens"]):
                    raise ConflictError("known usage ledger and job debit disagree")
                held = int(row["reserved_tokens"])
                scope = conn.execute("SELECT * FROM budget_scopes WHERE scope_id=?", (scope_id,)).fetchone()
                reserved_sum = int(conn.execute(
                    "SELECT COALESCE(SUM(reserved_tokens),0) FROM rescue_jobs WHERE budget_scope=?",
                    (scope_id,),
                ).fetchone()[0])
                used_sum = int(conn.execute(
                    "SELECT COALESCE(SUM(used_tokens),0) FROM rescue_jobs WHERE budget_scope=?",
                    (scope_id,),
                ).fetchone()[0])
                if (reserved_sum != int(scope["reserved_tokens"]) or used_sum != int(scope["used_tokens"])
                        or held < 0 or charged < 0):
                    raise ConflictError("scope accounting is inconsistent; refusing settlement")
                # Do not add a synthetic zero-usage event or erase the unknown flag.
                released = self._release_known_job_budget(conn, job_id)
                if not preserve_completed:
                    conn.execute(
                        "UPDATE rescue_jobs SET state='failed',fail_reason='usage_lower_bound_settled',updated_at=? "
                        "WHERE job_id=?", (_now(), job_id),
                    )
                self._event(conn, "job", job_id, "usage_lower_bound_settled",
                    reason=("operator_accepted_unreported_usage" if acceptance_source == "operator"
                            else "configured_lower_bound_policy"),
                    details={"acceptance_source": acceptance_source, "evidence_ref": evidence_ref, "fence": fence, "operator_reason": reason,
                             "charged_tokens": charged, "released_reserved_tokens": released,
                             "accounting_basis": "confirmed_lower_bound", "provider_usage_known": False})
                if self._unresolved_unknown_jobs(conn, scope_id) == 0:
                    conn.execute("UPDATE budget_scopes SET usage_unknown=0 WHERE scope_id=?", (scope_id,))
        return {"job_id": job_id, "accounting_basis": "confirmed_lower_bound",
                "charged_tokens": charged, "released_reserved_tokens": released,
                "provider_usage_known": False, "already_settled": prior is not None,
                "budget": self.budget_snapshot(scope_id)}

    @staticmethod
    def provider_usage_key(job_id: str, fence: int, index: int) -> str:
        # First-attempt keys retain their persisted evidence identity.
        prefix = job_id if fence == 1 else f'{job_id}-fence-{fence}'
        return prefix + f'-confirmed-call-{index}'

    def rejected_source_is_recorded(self, job_id: str, fence: int) -> bool:
        row = self._conn().execute(
            "SELECT details_json FROM events WHERE entity_type='job' AND entity_id=? "
            "AND action='source_validated' AND json_extract(details_json,'$.fence')=? "
            "ORDER BY sequence DESC LIMIT 1", (job_id, fence)).fetchone()
        return bool(row and json.loads(row[0]).get('accepted') is False)

    def record_source_validation(self, job_id: str, *, fence: int, accepted: bool,
                                 reason: str | None, receipt_ref: str) -> None:
        """Operator audit only; the Teacher receives no private scoring details."""
        payload = {'fence': fence, 'accepted': accepted, 'reason': reason, 'receipt_ref': receipt_ref}
        with self._write() as conn:
            row = conn.execute('SELECT * FROM rescue_jobs WHERE job_id=?', (job_id,)).fetchone()
            self._assert_current_lease(row, row['lease_owner'] if row else '', fence, _now())
            if self._begin_job_mutation(conn, job_id=job_id, action='source_validation',
                                        idempotency_key=str(fence), payload=payload):
                self._event(conn, 'job', job_id, 'source_validated', reason=reason, details=payload)

    def record_provider_receipt(self, job_id: str, *, worker_id: str, fence: int,
                                index: int, evidence_ref: str, tokens: int | None) -> None:
        require_sha256(evidence_ref, 'evidence_ref')
        _check_positive_int(index, 'provider call index', allow_zero=True)
        with self._write() as conn:
            row = conn.execute('SELECT * FROM rescue_jobs WHERE job_id=?', (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            changed = self._begin_job_mutation(conn, job_id=job_id, action='provider_receipt',
                idempotency_key=f'{fence}:{index}', payload={'worker_id': worker_id, 'fence': fence,
                    'evidence_ref': evidence_ref, 'tokens': tokens})
            if not changed:
                return
            self._assert_current_lease(row, worker_id, fence, _now())
            if tokens is not None:
                self._record_job_usage(conn, row, tokens=tokens,
                    idempotency_key=self.provider_usage_key(job_id, fence, index))
            self._event(conn, 'job', job_id, 'provider_receipt_recorded',
                        details={'index': index, 'evidence_ref': evidence_ref, 'tokens': tokens})

    def reconcile_unknown_usage(
        self,
        job_id: str,
        *,
        evidence_ref: str,
        call_tokens: Sequence[int],
    ) -> dict[str, Any]:
        """Settle an unknown job from verified, complete provider usage receipts."""
        if (not isinstance(evidence_ref, str) or len(evidence_ref) != 64
                or any(char not in "0123456789abcdef" for char in evidence_ref)):
            raise ContractError("evidence_ref must be a lowercase SHA-256 reference")
        if (not isinstance(call_tokens, (list, tuple)) or not call_tokens
                or any(type(value) is not int or value < 0 for value in call_tokens)):
            raise ContractError("call_tokens must contain one nonnegative count per provider call")
        current_tokens = list(call_tokens)
        failure_artifact = self._verify_artifact_ref(evidence_ref)
        if (not isinstance(failure_artifact, Mapping)
                or failure_artifact.get("schema") != "bepa-failed-teacher-v1"
                or failure_artifact.get("job_id") != job_id
                or not isinstance(failure_artifact.get("payload"), Mapping)):
            raise ContractError("usage evidence does not identify this failed Teacher job")
        failure_payload = failure_artifact["payload"]
        if (failure_payload.get("reason") != "usage_unknown"
                or failure_payload.get("usage_known") is not False
                or failure_payload.get("outcome_unknown") is not True):
            raise ContractError("usage evidence is not a recorded unknown-usage failure")
        explicit_fence = "fence" in failure_payload
        current_fence = failure_payload.get("fence", 1)
        if type(current_fence) is not int or current_fence < 1:
            raise ContractError("unknown-usage evidence has an invalid provider fence")
        current_calls = failure_payload.get("usage_calls")
        if not isinstance(current_calls, list) or not current_calls or len(current_calls) != len(current_tokens):
            raise ContractError("unknown-usage evidence has no provider call receipts")

        already_reconciled = False
        scope_id: str
        with self._write() as conn:
            row = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            scope_id = row["budget_scope"]
            journal: list[tuple[int, int, int | None, str, str | None]] = []
            if explicit_fence:
                if current_fence != int(row["fence"]):
                    raise ConflictError("usage evidence does not identify the latest job fence")
                from .token_usage import uncached_input_tokens
                rows = conn.execute(
                    "SELECT details_json FROM events WHERE entity_type='job' AND entity_id=? "
                    "AND action='provider_receipt_recorded' ORDER BY sequence", (job_id,),
                ).fetchall()
                for event_row in rows:
                    details = _json_dict(event_row["details_json"])
                    stored_artifact = self._verify_artifact_ref(details.get("evidence_ref"))
                    if (not isinstance(stored_artifact, Mapping)
                            or stored_artifact.get("schema") != "bepa-provider-call-v1"
                            or stored_artifact.get("job_id") != job_id
                            or type(stored_artifact.get("fence")) is not int
                            or not isinstance(stored_artifact.get("receipt"), Mapping)):
                        raise ContractError("provider receipt journal artifact is missing or invalid")
                    fence = stored_artifact["fence"]
                    receipt = stored_artifact["receipt"]
                    index = receipt.get("index")
                    prefix = job_id if fence == 1 else f"{job_id}-fence-{fence}"
                    if (fence < 1 or type(index) is not int or index < 0
                            or details.get("index") != index
                            or receipt.get("request_id") != f"{prefix}-call-{index}"
                            or receipt.get("budget_token_basis") != "flash_uncached_input"):
                        raise ContractError("provider receipt journal identity is invalid")
                    tokens = uncached_input_tokens(receipt.get("usage"))
                    if details.get("tokens") != tokens:
                        raise ContractError("provider receipt journal usage differs from its artifact")
                    journal.append((fence, index, tokens, receipt["request_id"], receipt.get("response_id")))

                journal.sort()
                fences = sorted({fence for fence, _index, _tokens, _request_id, _response_id in journal})
                if fences != list(range(1, current_fence + 1)):
                    raise ContractError("provider receipt journal is missing an earlier attempt")
                for fence in fences:
                    indices = [index for entry_fence, index, *_ in journal if entry_fence == fence]
                    if indices != list(range(len(indices))):
                        raise ContractError("provider receipt journal has a gap within an attempt")
                current_journal = [entry for entry in journal if entry[0] == current_fence]
                if len(current_journal) != len(current_calls):
                    raise ContractError("latest failure evidence does not cover every persisted provider call")
                for index, (provided, recorded) in enumerate(zip(current_calls, current_journal, strict=True)):
                    if not isinstance(provided, Mapping):
                        raise ContractError("provider call receipt must be an object")
                    prefix = job_id if current_fence == 1 else f"{job_id}-fence-{current_fence}"
                    if (provided.get("index") != index
                            or provided.get("request_id") != f"{prefix}-call-{index}"
                            or provided.get("request_id") != recorded[3]
                            or provided.get("response_id") != recorded[4]
                            or provided.get("budget_token_basis") != "flash_uncached_input"
                            or provided.get("stream_complete") is not True
                            or not isinstance(provided.get("finish_reason"), str)
                            or not provided.get("finish_reason")):
                        raise ContractError("latest failure receipt differs from the provider receipt journal")
                    tokens = uncached_input_tokens(provided.get("usage"))
                    if (tokens is None or tokens != current_tokens[index]
                            or (provided.get("total_tokens") is not None and provided.get("total_tokens") != tokens)
                            or (recorded[2] is not None and recorded[2] != tokens)):
                        raise ContractError("latest failure receipt has unresolved or conflicting usage")

                resolved_tokens = []
                for fence, index, tokens, _request_id, _response_id in journal:
                    if fence == current_fence:
                        resolved_tokens.append(current_tokens[index])
                    elif tokens is None:
                        raise ContractError("an earlier rescue attempt still has unknown provider usage")
                    else:
                        resolved_tokens.append(tokens)
                usage_rows = conn.execute(
                    "SELECT idempotency_key,tokens FROM usage_events WHERE job_id=? AND scope_id=?",
                    (job_id, scope_id),
                ).fetchall()
                usage_by_key = {item["idempotency_key"]: item["tokens"] for item in usage_rows}
                expected_keys = {
                    self.provider_usage_key(job_id, fence, index): tokens
                    for fence, index, tokens, _request_id, _response_id in journal
                }
                call_key_prefixes = (f"{job_id}-confirmed-call-", f"{job_id}-fence-")
                for key in usage_by_key:
                    if key.startswith(call_key_prefixes) and key not in expected_keys:
                        raise ConflictError("usage ledger has a provider call without a receipt journal entry")
                for offset, (fence, index, journal_tokens, _request_id, _response_id) in enumerate(journal):
                    key = self.provider_usage_key(job_id, fence, index)
                    ledger_tokens = usage_by_key.get(key)
                    if (fence < current_fence or journal_tokens is not None) and ledger_tokens != journal_tokens:
                        raise ConflictError("a confirmed provider receipt is missing from the usage ledger")
                    if ledger_tokens is not None and int(ledger_tokens) != resolved_tokens[offset]:
                        raise ConflictError("provider call usage conflicts with the reconciled receipt")
            else:
                resolved_tokens = current_tokens
                journal = [(1, index, tokens, "", None) for index, tokens in enumerate(resolved_tokens)]

            prior = conn.execute(
                "SELECT details_json FROM events WHERE entity_type='job' AND entity_id=? "
                "AND action='usage_reconciled' ORDER BY sequence DESC LIMIT 1", (job_id,),
            ).fetchone()
            if prior is not None:
                details = json.loads(prior["details_json"] or "{}")
                if (details.get("evidence_ref") != evidence_ref or details.get("call_tokens") != resolved_tokens
                        or details.get("fence", 1) != current_fence):
                    raise ConflictError("job usage was already reconciled from different evidence")
                already_reconciled = True
            else:
                if (not (row["state"] == "outcome_unknown" or (
                            row["state"] == "failed" and row["fail_reason"] == "usage_lower_bound_settled"))
                        or not bool(row["usage_unknown"]) or row["lease_owner"] is not None or row["lease_expires_at"] is not None):
                    raise ConflictError("only an unleased outcome_unknown job can be reconciled")
                known_before = conn.execute(
                    "SELECT COALESCE(SUM(tokens),0) FROM usage_events WHERE job_id=? AND tokens IS NOT NULL",
                    (job_id,),
                ).fetchone()[0]
                if int(known_before) != int(row["used_tokens"]):
                    raise ConflictError("settled job usage differs from its durable usage receipts")

                calls_to_settle = (
                    [(item[0], item[1], resolved_tokens[index])
                     for index, item in enumerate(journal)]
                    if explicit_fence else
                    [(1, index, tokens) for index, tokens in enumerate(resolved_tokens)]
                )
                for fence, index, tokens in calls_to_settle:
                    key = self.provider_usage_key(job_id, fence, index)
                    old = conn.execute(
                        "SELECT job_id,tokens FROM usage_events WHERE scope_id=? AND idempotency_key=?",
                        (scope_id, key),
                    ).fetchone()
                    if old is not None:
                        if old["job_id"] != job_id:
                            raise ConflictError("provider usage key belongs to a different job")
                        if old["tokens"] is not None and int(old["tokens"]) != tokens:
                            raise ConflictError("provider usage receipt conflicts with settled tokens")
                        if old["tokens"] is not None:
                            continue
                        conn.execute(
                            "UPDATE usage_events SET tokens=? WHERE scope_id=? AND idempotency_key=?",
                            (tokens, scope_id, key),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO usage_events(scope_id,idempotency_key,job_id,tokens,created_at) "
                            "VALUES(?,?,?,?,?)", (scope_id, key, job_id, tokens, _now()),
                        )
                    latest = conn.execute(
                        "SELECT reserved_tokens FROM rescue_jobs WHERE job_id=?", (job_id,),
                    ).fetchone()
                    covered = min(int(latest["reserved_tokens"]), tokens)
                    conn.execute(
                        "UPDATE rescue_jobs SET used_tokens=used_tokens+?,reserved_tokens=MAX(0,reserved_tokens-?) "
                        "WHERE job_id=?", (tokens, covered, job_id),
                    )
                    conn.execute(
                        "UPDATE budget_scopes SET used_tokens=used_tokens+?,reserved_tokens=MAX(0,reserved_tokens-?) "
                        "WHERE scope_id=?", (tokens, covered, scope_id),
                    )

                known_after = conn.execute(
                    "SELECT COALESCE(SUM(tokens),0) FROM usage_events WHERE job_id=? AND tokens IS NOT NULL",
                    (job_id,),
                ).fetchone()[0]
                latest = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
                total_tokens = sum(resolved_tokens)
                if int(known_after) != total_tokens or int(latest["used_tokens"]) != total_tokens:
                    raise ConflictError("provider receipts do not exactly reconcile the job's settled usage")
                released_tokens = self._release_known_job_budget(conn, job_id)
                conn.execute(
                    "UPDATE rescue_jobs SET state='failed',usage_unknown=0,fail_reason='provider_usage_reconciled',"
                    "lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE job_id=?",
                    (_now(), job_id),
                )
                remaining_unknown = self._unresolved_unknown_jobs(conn, scope_id)
                scope_unblocked = int(remaining_unknown) == 0
                if scope_unblocked:
                    conn.execute("UPDATE budget_scopes SET usage_unknown=0 WHERE scope_id=?", (scope_id,))
                self._event(
                    conn, "job", job_id, "usage_reconciled", reason="verified_provider_receipts",
                    details={"evidence_ref": evidence_ref, "fence": current_fence,
                             "call_tokens": resolved_tokens,
                             "usage_tokens": total_tokens, "released_reserved_tokens": released_tokens,
                             "scope_unblocked": scope_unblocked},
                )
        total_tokens = sum(resolved_tokens)
        return {
            "job_id": job_id,
            "state": self.get_job(job_id).state,
            "usage_tokens": total_tokens,
            "budget": self.budget_snapshot(scope_id),
            "already_reconciled": already_reconciled,
        }

    def _begin_job_mutation(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        action: str,
        idempotency_key: str,
        payload: Any,
    ) -> bool:
        if not idempotency_key:
            raise ContractError("idempotency_key must be non-empty")
        request_hash = content_hash(payload)
        row = conn.execute(
            "SELECT request_hash FROM job_mutations WHERE job_id=? AND action=? AND idempotency_key=?",
            (job_id, action, idempotency_key),
        ).fetchone()
        if row is not None:
            if row["request_hash"] != request_hash:
                raise ConflictError("idempotency key was reused with different request content")
            return False
        conn.execute(
            "INSERT INTO job_mutations(job_id,action,idempotency_key,request_hash,created_at) VALUES(?,?,?,?,?)",
            (job_id, action, idempotency_key, request_hash, _now()),
        )
        return True

    def complete_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        fence: int,
        idempotency_key: str,
        source_artifact_refs: Mapping[str, str],
        usage_tokens: int | None = None,
        usage_is_total: bool = False,
    ) -> JobRecord:
        """Submit immutable raw-source refs; completion never asserts exactness."""
        refs = dict(source_artifact_refs)
        if not refs:
            raise ContractError("source_artifact_refs must not be empty")
        for name, ref in refs.items():
            if not name:
                raise ContractError("artifact reference names must be non-empty")
            require_sha256(ref, f"source_artifact_refs[{name}]")
        payload = {"source_artifact_refs": refs, "usage_tokens": usage_tokens}
        with self._write() as conn:
            row = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            changed = self._begin_job_mutation(conn, job_id=job_id, action="complete", idempotency_key=idempotency_key, payload=payload)
            if not changed:
                return self._job_record(row)
            self._assert_current_lease(row, worker_id, fence, _now())
            self._record_job_usage(conn, row, tokens=usage_tokens, idempotency_key=f"complete:{idempotency_key}", usage_is_total=usage_is_total, attempt_fence=fence)
            if usage_tokens is not None:
                self._release_known_job_budget(conn, job_id)
            conn.execute(
                "UPDATE rescue_jobs SET state='source_submitted',source_artifact_refs_json=?,lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE job_id=?",
                (canonical_json_bytes(refs).decode("utf-8"), _now(), job_id),
            )
            self._event(conn, "job", job_id, "source_submitted", details={"refs": refs})
        return self.get_job(job_id)

    def fail_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        fence: int,
        idempotency_key: str,
        reason: str,
        usage_tokens: int | None,
        usage_is_total: bool = False,
        retryable: bool = False,
        safe_to_retry: bool = False,
        outcome_unknown: bool = False,
        confirmed_call_tokens: Sequence[int | None] = (),
        evidence_ref: str | None = None,
    ) -> JobRecord:
        """Persist failure, budget usage, and whether an attempt is safe to retry."""
        if not reason:
            raise ContractError("failure reason must be a typed non-empty value")
        # Audit references do not change the budget mutation identity.
        payload = {"reason": reason, "usage_tokens": usage_tokens, "retryable": retryable,
                   "safe_to_retry": safe_to_retry, "outcome_unknown": outcome_unknown,
                   "worker_id": worker_id, "fence": fence, "confirmed_call_tokens": list(confirmed_call_tokens)}
        with self._write() as conn:
            row = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            changed = self._begin_job_mutation(conn, job_id=job_id, action="fail", idempotency_key=idempotency_key, payload=payload)
            if not changed:
                return self._job_record(row)
            self._assert_current_lease(row, worker_id, fence, _now())
            for index, tokens in enumerate(confirmed_call_tokens):
                if type(tokens) is int and tokens >= 0:
                    latest = conn.execute('SELECT * FROM rescue_jobs WHERE job_id=?', (job_id,)).fetchone()
                    self._record_job_usage(conn, latest, tokens=tokens,
                        idempotency_key=self.provider_usage_key(job_id, fence, index))
            row = conn.execute('SELECT * FROM rescue_jobs WHERE job_id=?', (job_id,)).fetchone()
            self._record_job_usage(conn, row, tokens=usage_tokens, idempotency_key=f"failure:{idempotency_key}", usage_is_total=usage_is_total, attempt_fence=fence)
            latest = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
            scope = conn.execute("SELECT * FROM budget_scopes WHERE scope_id=?", (row["budget_scope"],)).fetchone()
            budget_ok = (
                not latest["usage_unknown"] and not scope["usage_unknown"]
                and int(latest["used_tokens"]) < int(latest["max_tokens"])
                and int(scope["used_tokens"]) < int(scope["max_total_tokens"])
            )
            if outcome_unknown or usage_tokens is None:
                state = "outcome_unknown"
                if outcome_unknown or usage_tokens is None:
                    conn.execute("UPDATE rescue_jobs SET usage_unknown=1 WHERE job_id=?", (job_id,))
                    conn.execute("UPDATE budget_scopes SET usage_unknown=1 WHERE scope_id=?", (row["budget_scope"],))
            elif retryable and safe_to_retry and budget_ok and latest["attempts"] < latest["max_attempts"]:
                state = "retry_wait"
            elif not budget_ok or latest["attempts"] >= latest["max_attempts"]:
                state = "exhausted"
            else:
                state = "failed"
            if usage_tokens is not None and not outcome_unknown:
                self._release_known_job_budget(conn, job_id)
            conn.execute(
                "UPDATE rescue_jobs SET state=?,fail_reason=?,lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE job_id=?",
                (state, reason, _now(), job_id),
            )
            details = {"retryable": retryable, "safe_to_retry": safe_to_retry,
                       "usage_tokens": usage_tokens, "fence": fence}
            if evidence_ref is not None:
                details["evidence_ref"] = evidence_ref
            self._event(conn, "job", job_id, state, reason=reason, details=details)
        return self.get_job(job_id)

    def mark_source_verified(
        self,
        job_id: str,
        *,
        source_episode_id: str,
        trusted_receipt_ref: str,
        idempotency_key: str,
    ) -> JobRecord:
        """Advance a submitted source only after central verification."""
        require_sha256(trusted_receipt_ref, "trusted_receipt_ref")
        self._verify_artifact_ref(trusted_receipt_ref)
        if not source_episode_id:
            raise ContractError("source_episode_id must be non-empty")
        payload = {"source_episode_id": source_episode_id, "trusted_receipt_ref": trusted_receipt_ref}
        with self._write() as conn:
            row = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            changed = self._begin_job_mutation(conn, job_id=job_id, action="source_verified", idempotency_key=idempotency_key, payload=payload)
            if not changed:
                return self._job_record(row)
            if row["state"] != "source_submitted":
                raise StateTransitionError("only source_submitted jobs can be verified")
            conn.execute(
                "UPDATE rescue_jobs SET state='source_verified',source_episode_id=?,source_receipt_ref=?,updated_at=? WHERE job_id=?",
                (source_episode_id, trusted_receipt_ref, _now(), job_id),
            )
            self._event(conn, "job", job_id, "source_verified", details={"source_episode_id": source_episode_id, "receipt_ref": trusted_receipt_ref})
        return self.get_job(job_id)

    def mark_plan_ready(
        self,
        job_id: str,
        *,
        plan_artifact_ref: str,
        source_episode_id: str,
        trusted_receipt_ref: str,
        idempotency_key: str,
    ) -> PlanRecord:
        """Publish an already centrally validated plan, tied to its source receipt."""
        require_sha256(plan_artifact_ref, "plan_artifact_ref")
        require_sha256(trusted_receipt_ref, "trusted_receipt_ref")
        self._verify_artifact_ref(plan_artifact_ref)
        self._verify_artifact_ref(trusted_receipt_ref)
        payload = {"plan_artifact_ref": plan_artifact_ref, "source_episode_id": source_episode_id,
                   "trusted_receipt_ref": trusted_receipt_ref}
        with self._write() as conn:
            job = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            changed = self._begin_job_mutation(conn, job_id=job_id, action="plan_ready", idempotency_key=idempotency_key, payload=payload)
            if not changed:
                row = conn.execute("SELECT * FROM plans WHERE job_id=?", (job_id,)).fetchone()
                if row is None:
                    raise RuntimeError("idempotent plan mutation has no plan row")
                return self._plan_record(row)
            if job["state"] != "source_verified":
                raise StateTransitionError("a plan requires a centrally verified source")
            if (job["source_episode_id"], job["source_receipt_ref"]) != (source_episode_id, trusted_receipt_ref):
                raise ContractError("plan source or trusted receipt does not match the verified job source")
            plan_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO plans(plan_id,job_id,task_key,plan_artifact_ref,source_episode_id,source_receipt_ref,status,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (plan_id, job_id, job["task_key"], plan_artifact_ref, source_episode_id, trusted_receipt_ref, "ready", _now(), _now()),
            )
            conn.execute("UPDATE rescue_jobs SET state='plan_ready',plan_artifact_ref=?,updated_at=? WHERE job_id=?",
                         (plan_artifact_ref, _now(), job_id))
            self._event(conn, "plan", plan_id, "ready", details={"job_id": job_id, "task_key": job["task_key"]})
            row = conn.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
            return self._plan_record(row)

    def _native_rescue_puzzle(self, task_key: str) -> tuple[str, str, str]:
        task = self.get_task(task_key)
        if task is None:
            raise KeyError(task_key)
        public = canonical_json_bytes(task.public_challenge).decode('utf-8')
        key = content_hash({'public_challenge': task.public_challenge,
                            'private_target_hash': task.private_target_hash})
        return key, public, task.private_target_hash

    def native_rescue_progress(self, task_key: str, *, engine_epoch: str) -> dict[str, Any]:
        puzzle_key, _public, _target = self._native_rescue_puzzle(task_key)
        row = self._conn().execute(
            'SELECT success_versions_json,retired FROM native_rescue_progress '
            'WHERE engine_epoch=? AND puzzle_key=?', (engine_epoch, puzzle_key),
        ).fetchone()
        versions = [] if row is None else json.loads(row['success_versions_json'])
        return {'success_versions': versions, 'consecutive_successes': len(versions),
                'retired': bool(row['retired']) if row is not None else False}

    def retired_rescue_task_keys(self, task_keys: Sequence[str], *, engine_epoch: str) -> set[str]:
        keys = tuple(dict.fromkeys(task_keys))
        if not keys:
            return set()
        rows = self._conn().execute(
            'SELECT t.task_key FROM tasks t JOIN native_rescue_progress p '
            'ON p.private_target_hash=t.private_target_hash '
            "AND p.public_challenge=json_extract(t.spec_json,'$.public_challenge') "
            'WHERE p.engine_epoch=? AND p.retired=1 '
            f"AND t.task_key IN ({','.join('?' for _ in keys)})", (engine_epoch, *keys),
        ).fetchall()
        return {row['task_key'] for row in rows}

    def observe_native_rescue_group(
        self, task_key: str, *, engine_epoch: str, group_id: str,
        policy_version: int, native_exact: Sequence[bool],
    ) -> dict[str, Any]:
        """One complete all-correct native group retires priority, not the plan.

        Every complete all-failure group resets the streak. After retirement,
        any non-perfect native group reopens priority and starts a new cycle.
        Group receipts make retries idempotent; observations survive normal resume.
        """
        _check_positive_int(policy_version, 'policy_version', allow_zero=True)
        if not engine_epoch or not group_id or not native_exact or any(type(x) is not bool for x in native_exact):
            raise ContractError('native rescue progress requires an identified complete boolean result')
        puzzle_key, public, target = self._native_rescue_puzzle(task_key)
        exact_json = canonical_json_bytes(list(native_exact)).decode('utf-8')
        reopened = False
        with self._write() as conn:
            seen = conn.execute(
                'SELECT * FROM native_rescue_observations WHERE engine_epoch=? AND group_id=?',
                (engine_epoch, group_id),
            ).fetchone()
            if seen is not None:
                if (seen['puzzle_key'], seen['policy_version'], seen['native_exact_json']) != (
                        puzzle_key, policy_version, exact_json):
                    raise ConflictError('native group ID was reused with different progress evidence')
            else:
                row = conn.execute(
                    'SELECT * FROM native_rescue_progress WHERE engine_epoch=? AND puzzle_key=?',
                    (engine_epoch, puzzle_key),
                ).fetchone()
                versions = [] if row is None else json.loads(row['success_versions_json'])
                was_retired = bool(row['retired']) if row is not None else False
                versions = _advance_native_rescue_versions(versions, policy_version, native_exact)
                reopened = was_retired and not all(native_exact)
                retired = len(versions) >= NATIVE_RESCUE_REQUIRED_SUCCESSES
                conn.execute(
                    'INSERT INTO native_rescue_progress VALUES(?,?,?,?,?,?,?) '
                    'ON CONFLICT(engine_epoch,puzzle_key) DO UPDATE SET '
                    'success_versions_json=excluded.success_versions_json, '
                    'retired=excluded.retired,updated_at=excluded.updated_at',
                    (puzzle_key, engine_epoch, public, target,
                     canonical_json_bytes(versions).decode('utf-8'), int(retired), _now()),
                )
                conn.execute('INSERT INTO native_rescue_observations VALUES(?,?,?,?,?)',
                             (engine_epoch, group_id, puzzle_key, policy_version, exact_json))
                self._event(conn, 'native_rescue', puzzle_key, 'native_group_observed', details={
                    'engine_epoch': engine_epoch, 'group_id': group_id, 'policy_version': policy_version,
                    'success_versions': versions, 'retired': retired, 'reopened': reopened,
                })
        return {**self.native_rescue_progress(task_key, engine_epoch=engine_epoch), 'reopened': reopened}

    @staticmethod
    def _plans_for_tasks(
        conn: sqlite3.Connection, task_keys: Sequence[str], *,
        statuses: Sequence[str] | None = None, limit: int | None = 1000,
        plan_id: str | None = None,
    ) -> list[sqlite3.Row]:
        """Match retained Teacher text by puzzle, never by execution contracts.

        Keep the original task/job/receipt intact. Only the returned view binds
        the text to a current training task; tokenized caches retain full keys.
        """
        keys = tuple(dict.fromkeys(task_keys))
        if not keys:
            return []
        query = (
            "SELECT p.*, target.task_key AS current_task_key FROM plans p "
            "JOIN tasks origin ON origin.task_key=p.task_key "
            "JOIN tasks target ON target.private_target_hash=origin.private_target_hash "
            "AND json_extract(target.spec_json,'$.public_challenge')="
            "json_extract(origin.spec_json,'$.public_challenge') "
            f"WHERE target.allowed=1 AND target.task_key IN ({','.join('?' for _ in keys)}) "
            "AND NOT EXISTS (SELECT 1 FROM events e WHERE e.entity_type='plan' "
            "AND e.entity_id=p.plan_id AND e.action='manually_forgotten') "
        )
        args: list[Any] = list(keys)
        if statuses is not None:
            query += f"AND p.status IN ({','.join('?' for _ in statuses)}) "
            args.extend(statuses)
        if plan_id is not None:
            query += "AND p.plan_id=? "
            args.append(plan_id)
        query += "ORDER BY p.created_at,p.plan_id,target.task_key"
        if limit is not None:
            query += " LIMIT ?"
            args.append(limit)
        return conn.execute(query, args).fetchall()

    def retained_plans(self, task_keys: Sequence[str]) -> list[PlanRecord]:
        conn = self._conn()
        rows = self._plans_for_tasks(conn, task_keys, limit=None)
        for row in rows:
            self._assert_verified_seed_plan(conn, row)
        return [self._plan_record(row) for row in rows]

    def list_retained_plans(
        self, task_keys: Sequence[str], *, engine_epoch: str | None = None,
    ) -> list[dict[str, Any]]:
        records = [{'plan_id': row['plan_id'], 'task_key': row['current_task_key'],
                    'source_task_key': row['task_key'], 'job_id': row['job_id'],
                    'status': row['status'], 'created_at': row['created_at']}
                   for row in self._plans_for_tasks(self._conn(), task_keys, limit=None)]
        if engine_epoch is not None:
            for record in records:
                record['priority_rescue'] = self.native_rescue_progress(
                    record['task_key'], engine_epoch=engine_epoch,
                )
        return records

    def forget_plan(self, plan_id: str) -> dict[str, str]:
        """Explicitly stop offering one plan; preserve its audit and budget history."""
        with self._write() as conn:
            plan = conn.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
            if plan is None:
                raise KeyError(plan_id)
            if plan['status'] == 'seed_running':
                raise ConflictError('cannot forget a plan while a guided attempt owns it')
            exists = conn.execute(
                "SELECT 1 FROM events WHERE entity_type='plan' AND entity_id=? "
                "AND action='manually_forgotten'", (plan_id,),
            ).fetchone()
            if exists is None:
                self._event(conn, 'plan', plan_id, 'manually_forgotten')
        return {'plan_id': plan_id, 'status': 'forgotten'}

    @staticmethod
    def _latest_seed_generation(conn: sqlite3.Connection, plan_id: str) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT details_json FROM events WHERE entity_type='plan' AND entity_id=? AND action='seed_generation' "
            "ORDER BY sequence DESC LIMIT 1",
            (plan_id,),
        ).fetchone()
        if row is None:
            return None
        if not row["details_json"]:
            raise RuntimeError(f"plan {plan_id} has a malformed durable seed generation event")
        generation = _json_dict(row["details_json"])
        generation_id = generation.get("generation_id")
        try:
            parsed_generation_id = uuid.UUID(generation_id) if isinstance(generation_id, str) else None
        except ValueError:
            parsed_generation_id = None
        if parsed_generation_id is None or str(parsed_generation_id) != generation_id:
            raise RuntimeError(f"plan {plan_id} has a malformed durable seed generation identity")
        if (not isinstance(generation.get("engine_epoch"), str) or not generation["engine_epoch"]
                or isinstance(generation.get("start_policy_version"), bool)
                or not isinstance(generation.get("start_policy_version"), int)
                or generation["start_policy_version"] < 0):
            raise RuntimeError(f"plan {plan_id} has malformed durable seed generation context")
        return generation

    @staticmethod
    def _assert_verified_seed_plan(conn: sqlite3.Connection, plan: sqlite3.Row) -> sqlite3.Row:
        task = conn.execute("SELECT task_key,allowed FROM tasks WHERE task_key=?", (plan["task_key"],)).fetchone()
        job = conn.execute("SELECT * FROM rescue_jobs WHERE job_id=?", (plan["job_id"],)).fetchone()
        if task is None:
            raise ContractError("seed plan has no stored source task")
        if job is None or job["task_key"] != plan["task_key"]:
            raise RuntimeError(f"plan {plan['plan_id']} has no matching durable rescue job")
        if (not plan["source_episode_id"] or not plan["source_receipt_ref"]
                or not plan["plan_artifact_ref"]
                or (job["source_episode_id"], job["source_receipt_ref"], job["plan_artifact_ref"])
                != (plan["source_episode_id"], plan["source_receipt_ref"], plan["plan_artifact_ref"])):
            raise RuntimeError(f"plan {plan['plan_id']} does not match its verified source record")
        verified_rows = conn.execute(
            "SELECT details_json FROM events WHERE entity_type='job' AND entity_id=? AND action='source_verified'",
            (plan["job_id"],),
        ).fetchall()
        verified = False
        for row in verified_rows:
            if not row["details_json"]:
                continue
            details = _json_dict(row["details_json"])
            if (details.get("source_episode_id") == plan["source_episode_id"]
                    and details.get("receipt_ref") == plan["source_receipt_ref"]):
                verified = True
                break
        if not verified:
            raise RuntimeError(f"plan {plan['plan_id']} has no matching source_verified event")
        return job

    def _start_seed_generation(
        self,
        conn: sqlite3.Connection,
        plan: sqlite3.Row,
        *,
        engine_epoch: str,
        current_policy_version: int,
        reason: str,
        previous: Mapping[str, Any] | None = None,
        group_id: str | None = None,
    ) -> str:
        generation_id = str(uuid.uuid4())
        details: dict[str, Any] = {
            "generation_id": generation_id,
            "engine_epoch": engine_epoch,
            "start_policy_version": current_policy_version,
            "job_id": plan["job_id"],
            "task_key": (plan["current_task_key"] if "current_task_key" in plan.keys()
                         else plan["task_key"]),
        }
        if group_id is not None:
            details["group_id"] = group_id
        if previous is not None:
            details["previous_generation_id"] = previous["generation_id"]
            details["previous_engine_epoch"] = previous["engine_epoch"]
            details["previous_start_policy_version"] = previous["start_policy_version"]
        self._event(conn, "plan", plan["plan_id"], "seed_generation", reason=reason, details=details)
        return generation_id

    @staticmethod
    def _latest_seed_claim(conn: sqlite3.Connection, plan_id: str) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT details_json FROM events WHERE entity_type='plan' AND entity_id=? AND action='seed_claimed' "
            "ORDER BY sequence DESC LIMIT 1",
            (plan_id,),
        ).fetchone()
        return _json_dict(row["details_json"]) if row is not None and row["details_json"] else None

    @staticmethod
    def _seed_claim_record(plan: sqlite3.Row, details: Mapping[str, Any]) -> SeedPlanClaim:
        required = ("engine_epoch", "generation_id", "run_id", "attempt_id", "fence", "lease_expires_at",
                    "attempt_number", "policy_version")
        if any(key not in details for key in required):
            raise RuntimeError("durable seed claim is missing its fencing identity")
        return SeedPlanClaim(
            plan_id=plan["plan_id"], job_id=plan["job_id"],
            task_key=str(details.get("task_key", plan["task_key"])),
            plan_artifact_ref=plan["plan_artifact_ref"], source_episode_id=plan["source_episode_id"],
            source_receipt_ref=plan["source_receipt_ref"], status=plan["status"],
            seed_attempts=int(plan["seed_attempts"]), created_at=float(plan["created_at"]),
            engine_epoch=str(details["engine_epoch"]), generation_id=str(details["generation_id"]),
            run_id=str(details["run_id"]), attempt_id=str(details["attempt_id"]),
            fence=int(details["fence"]), lease_expires_at=float(details["lease_expires_at"]),
            attempt_number=int(details["attempt_number"]),
            policy_version=int(details["policy_version"]),
        )

    def _assert_seed_claim(self, conn: sqlite3.Connection, claim: SeedPlanClaim, *, now: float) -> sqlite3.Row:
        plan = conn.execute("SELECT * FROM plans WHERE plan_id=?", (claim.plan_id,)).fetchone()
        if plan is None or plan["job_id"] != claim.job_id or plan["status"] != "seed_running":
            raise LeaseLostError("seed plan is no longer held by this attempt")
        self._assert_verified_seed_plan(conn, plan)
        if not self._plans_for_tasks(conn, [claim.task_key], plan_id=claim.plan_id):
            raise LeaseLostError("seed target is no longer an allowed matching puzzle")
        generation = self._latest_seed_generation(conn, claim.plan_id)
        current = self._latest_seed_claim(conn, claim.plan_id)
        identity = (claim.engine_epoch, claim.generation_id, claim.run_id, claim.attempt_id,
                    claim.fence, claim.lease_expires_at, claim.attempt_number, claim.policy_version)
        if (generation is None or generation["engine_epoch"] != claim.engine_epoch
                or generation["generation_id"] != claim.generation_id or current is None):
            raise LeaseLostError("seed generation changed while this attempt was running")
        current_identity = (current.get("engine_epoch"), current.get("generation_id"), current.get("run_id"),
                            current.get("attempt_id"), current.get("fence"), current.get("lease_expires_at"),
                            current.get("attempt_number"), current.get("policy_version"))
        if (identity != current_identity or float(claim.lease_expires_at) <= now
                or current.get("task_key", plan["task_key"]) != claim.task_key):
            raise LeaseLostError("seed attempt fencing token is stale or expired")
        return plan

    def _current_cache_state(
        self,
        conn: sqlite3.Connection,
        *,
        task_key: str,
        engine_epoch: str,
        current_policy_version: int,
    ) -> tuple[str, sqlite3.Row | None]:
        """Return eligible/pending/absent for cache entries in the current epoch."""
        rows = conn.execute(
            "SELECT * FROM cache_entries WHERE task_key=? AND engine_epoch=? "
            "ORDER BY ready_sequence DESC,created_at DESC",
            (task_key, engine_epoch),
        ).fetchall()
        for cache in rows:
            pending = conn.execute(
                "SELECT count(*) FROM reservations WHERE cache_entry_id=? AND state IN ('reserved','commit_unknown')",
                (cache["cache_entry_id"],),
            ).fetchone()[0]
            age = current_policy_version - int(cache["source_policy_version"])
            reserved, committed = self._active_reservation_counts(conn, cache["cache_entry_id"])
            metadata_state = _cache_metadata_state(
                active=bool(cache["active"]), age_updates=age,
                max_age_updates=self.max_age_updates, reserved=reserved, committed=committed,
                max_committed_uses=self.max_committed_uses, pending=bool(pending),
            )
            if metadata_state == "pending":
                # Wait for an update/release decision before spinning up a seed.
                return "pending", cache
            if metadata_state == "metadata_eligible":
                return "eligible", cache
        return "absent", None

    def _seed_cache_binding(self, conn: sqlite3.Connection, plan_id: str) -> sqlite3.Row | None:
        row = conn.execute(
            "SELECT details_json FROM events WHERE entity_type='plan' AND entity_id=? AND action='seed_cache_bound' "
            "ORDER BY sequence DESC LIMIT 1",
            (plan_id,),
        ).fetchone()
        if row is None or not row["details_json"]:
            return None
        details = _json_dict(row["details_json"])
        cache_id = details.get("cache_entry_id")
        if not isinstance(cache_id, str):
            return None
        return conn.execute("SELECT * FROM cache_entries WHERE cache_entry_id=?", (cache_id,)).fetchone()

    def refresh_seed_plans(
        self,
        task_allowlist: Sequence[str],
        *,
        engine_epoch: str,
        current_policy_version: int,
        limit: int = 1000,
        max_attempts: int = 2,
        whole_group: bool = False,
        include_retired: bool = False,
        group_id: str | None = None,
    ) -> dict[str, int]:
        """Start bounded seed generations for verified plans without a usable cache."""
        if not engine_epoch:
            raise ContractError("engine_epoch must be non-empty")
        _check_positive_int(current_policy_version, "current_policy_version", allow_zero=True)
        _check_positive_int(limit, "limit")
        _check_positive_int(max_attempts, "max_attempts")
        if max_attempts != 2:
            raise ContractError("a seed generation has exactly two guided attempts")
        if type(whole_group) is not bool:
            raise ContractError("whole_group must be a boolean")
        if type(include_retired) is not bool or (include_retired and not whole_group):
            raise ContractError("include_retired requires a boolean whole-group request")
        if group_id is not None and (not whole_group or not isinstance(group_id, str) or not group_id):
            raise ContractError("group_id requires an identified whole-group dispatch")
        keys = tuple(dict.fromkeys(task_allowlist))
        summary = {"started": 0, "reactivated": 0, "deferred": 0, "eligible_cache": 0, "pending_cache": 0}
        if not keys:
            return summary
        if not include_retired:
            retired = self.retired_rescue_task_keys(keys, engine_epoch=engine_epoch)
            keys = tuple(key for key in keys if key not in retired)
        if not keys:
            return summary
        now = _now()
        with self._write() as conn:
            plans = self._plans_for_tasks(
                conn, keys, statuses=('ready', 'seed_running', 'consumed', 'exhausted'), limit=limit,
            )
            for plan in plans:
                self._assert_verified_seed_plan(conn, plan)
                status = plan["status"]
                cache_state, _current_cache = self._current_cache_state(
                    conn, task_key=plan["current_task_key"], engine_epoch=engine_epoch,
                    current_policy_version=current_policy_version,
                )
                if group_id is not None:
                    generation = self._latest_seed_generation(conn, plan["plan_id"])
                    claim = self._latest_seed_claim(conn, plan["plan_id"]) if status == "seed_running" else None
                    if cache_state == "pending":
                        summary["pending_cache"] += 1
                        continue
                    if (claim is not None and generation is not None
                            and claim.get("generation_id") == generation["generation_id"]
                            and float(claim.get("lease_expires_at", 0)) > now):
                        summary["deferred"] += 1
                        continue
                    if (generation is None or generation["engine_epoch"] != engine_epoch
                            or generation.get("task_key") != plan["current_task_key"]
                            or generation.get("group_id") != group_id):
                        self._start_seed_generation(
                            conn, plan, engine_epoch=engine_epoch,
                            current_policy_version=current_policy_version,
                            reason="scheduled_whole_group_guidance", previous=generation,
                            group_id=group_id,
                        )
                        conn.execute("UPDATE plans SET status='ready',seed_attempts=0,updated_at=? WHERE plan_id=?",
                                     (now, plan["plan_id"]))
                        conn.execute("UPDATE rescue_jobs SET state='plan_ready',fail_reason=NULL,updated_at=? WHERE job_id=?",
                                     (now, plan["job_id"]))
                        summary["started" if generation is None else "reactivated"] += 1
                    continue
                if cache_state == "eligible":
                    generation = (self._latest_seed_generation(conn, plan["plan_id"])
                                  if whole_group else None)
                    missing_generation = whole_group and (
                        generation is None or generation["engine_epoch"] != engine_epoch
                        or generation.get("task_key", plan["task_key"]) != plan["current_task_key"])
                    if whole_group and (status in {"consumed", "exhausted"} or missing_generation):
                        self._start_seed_generation(
                            conn,
                            plan,
                            engine_epoch=engine_epoch,
                            current_policy_version=current_policy_version,
                            reason="central_whole_group_guidance",
                            previous=generation,
                        )
                        conn.execute(
                            "UPDATE plans SET status='ready',seed_attempts=0,updated_at=? WHERE plan_id=?",
                            (now, plan["plan_id"]),
                        )
                        conn.execute(
                            "UPDATE rescue_jobs SET state='plan_ready',fail_reason=NULL,updated_at=? WHERE job_id=?",
                            (now, plan["job_id"]),
                        )
                        summary["started" if generation is None else "reactivated"] += 1
                        continue
                    summary["eligible_cache"] += 1
                    continue
                if cache_state == "pending":
                    summary["pending_cache"] += 1
                    continue
                generation = self._latest_seed_generation(conn, plan["plan_id"])
                if status == "seed_running":
                    claim = self._latest_seed_claim(conn, plan["plan_id"])
                    claim_expiry = claim.get("lease_expires_at") if claim else None
                    if claim is not None and not isinstance(claim_expiry, (int, float)):
                        raise RuntimeError(f"plan {plan['plan_id']} has a malformed durable seed claim")
                    if (generation is not None and claim is not None
                            and claim.get("engine_epoch") == engine_epoch
                            and claim.get("generation_id") == generation["generation_id"]
                            and float(claim_expiry) > now):
                        summary["deferred"] += 1
                        continue
                    reason = "seed_lease_expired" if claim is not None else "unfenced_seed_attempt_recovered"
                    attempts = (int(plan["seed_attempts"])
                                if generation is not None and generation["engine_epoch"] == engine_epoch else 0)
                    next_status = "exhausted" if attempts >= max_attempts else "ready"
                    conn.execute("UPDATE plans SET status=?,seed_attempts=?,updated_at=? WHERE plan_id=?",
                                 (next_status, attempts, now, plan["plan_id"]))
                    conn.execute("UPDATE rescue_jobs SET state=?,fail_reason=?,updated_at=? WHERE job_id=?",
                                 ("exhausted" if next_status == "exhausted" else "plan_ready",
                                  reason, now, plan["job_id"]))
                    failure_details = {"attempt_id": claim.get("attempt_id") if claim else None,
                                       "run_id": claim.get("run_id") if claim else None,
                                       "fence": claim.get("fence") if claim else None,
                                       "attempts": attempts, "max_attempts": max_attempts}
                    if generation is not None:
                        failure_details["generation_id"] = generation["generation_id"]
                    self._event(conn, "plan", plan["plan_id"], "seed_failed", reason=reason,
                                details=failure_details)
                    plan = self._plans_for_tasks(
                        conn, [plan['current_task_key']], plan_id=plan['plan_id'], limit=1,
                    )[0]
                    status = plan["status"]

                if generation is None:
                    self._start_seed_generation(
                        conn, plan, engine_epoch=engine_epoch,
                        current_policy_version=current_policy_version,
                        reason="verified_plan_seed_generation",
                    )
                    was_reactivated = status in {"consumed", "exhausted", "seed_running"}
                    conn.execute("UPDATE plans SET status='ready',seed_attempts=0,updated_at=? WHERE plan_id=?",
                                 (now, plan["plan_id"]))
                    conn.execute("UPDATE rescue_jobs SET state='plan_ready',fail_reason=NULL,updated_at=? WHERE job_id=?",
                                 (now, plan["job_id"]))
                    summary["started"] += 1
                    if was_reactivated:
                        summary["reactivated"] += 1
                    continue

                generation_epoch = generation["engine_epoch"]
                generation_start = generation["start_policy_version"]
                contract_changed = generation.get('task_key', plan['task_key']) != plan['current_task_key']
                if generation_epoch != engine_epoch or contract_changed:
                    was_reactivated = status in {"consumed", "exhausted"}
                    self._start_seed_generation(
                        conn, plan, engine_epoch=engine_epoch,
                        current_policy_version=current_policy_version,
                        reason="execution_contract_changed" if contract_changed else "new_engine_epoch",
                        previous=generation,
                    )
                    conn.execute("UPDATE plans SET status='ready',seed_attempts=0,updated_at=? WHERE plan_id=?",
                                 (now, plan["plan_id"]))
                    conn.execute("UPDATE rescue_jobs SET state='plan_ready',fail_reason=NULL,updated_at=? WHERE job_id=?",
                                 (now, plan["job_id"]))
                    summary["started"] += 1
                    if was_reactivated:
                        summary["reactivated"] += 1
                    continue
                if current_policy_version < generation_start:
                    raise ContractError("current policy version precedes this engine epoch's seed generation")
                if status == "ready":
                    if int(plan["seed_attempts"]) >= max_attempts:
                        conn.execute("UPDATE plans SET status='exhausted',updated_at=? WHERE plan_id=?",
                                     (now, plan["plan_id"]))
                        conn.execute("UPDATE rescue_jobs SET state='exhausted',fail_reason='attempt_budget_exhausted',updated_at=? WHERE job_id=?",
                                     (now, plan["job_id"]))
                    continue

                bound_cache = self._seed_cache_binding(conn, plan["plan_id"])
                quota_exhausted = (
                    status == "consumed" and bound_cache is not None
                    and int(bound_cache["committed_uses"]) >= self.max_committed_uses
                )
                generation_age = current_policy_version - generation_start
                cache_age = (current_policy_version - int(bound_cache["source_policy_version"])
                             if bound_cache is not None else None)
                if quota_exhausted:
                    reason = "cache_use_quota_exhausted"
                elif cache_age is not None and cache_age >= self.max_age_updates:
                    reason = "cache_age_elapsed"
                elif bound_cache is None and generation_age >= self.max_age_updates:
                    reason = "seed_generation_age_elapsed"
                else:
                    summary["deferred"] += 1
                    continue
                self._start_seed_generation(
                    conn, plan, engine_epoch=engine_epoch,
                    current_policy_version=current_policy_version, reason=reason,
                    previous=generation,
                )
                conn.execute("UPDATE plans SET status='ready',seed_attempts=0,updated_at=? WHERE plan_id=?",
                             (now, plan["plan_id"]))
                conn.execute("UPDATE rescue_jobs SET state='plan_ready',fail_reason=NULL,updated_at=? WHERE job_id=?",
                             (now, plan["job_id"]))
                summary["reactivated"] += 1
        return summary

    def available_plans(
        self,
        task_allowlist: Sequence[str],
        *,
        engine_epoch: str,
        current_policy_version: int,
        limit: int = 100,
        whole_group: bool = False,
        include_retired: bool = False,
    ) -> list[PlanRecord]:
        """Return verified plans whose task has no eligible or pending cache."""
        _check_positive_int(limit, "limit")
        if not engine_epoch:
            raise ContractError("engine_epoch must be non-empty")
        _check_positive_int(current_policy_version, "current_policy_version", allow_zero=True)
        if type(whole_group) is not bool:
            raise ContractError("whole_group must be a boolean")
        if type(include_retired) is not bool or (include_retired and not whole_group):
            raise ContractError("include_retired requires a boolean whole-group request")
        keys = tuple(dict.fromkeys(task_allowlist))
        if not keys:
            return []
        if not include_retired:
            retired = self.retired_rescue_task_keys(keys, engine_epoch=engine_epoch)
            keys = tuple(key for key in keys if key not in retired)
        conn = self._conn()
        rows = self._plans_for_tasks(conn, keys, statuses=('ready',), limit=None)
        result = []
        for row in rows:
            self._assert_verified_seed_plan(conn, row)
            cache_state, _cache = self._current_cache_state(
                conn, task_key=row["current_task_key"], engine_epoch=engine_epoch,
                current_policy_version=current_policy_version,
            )
            if cache_state == "pending" or (cache_state == "eligible" and not whole_group):
                continue
            result.append(self._plan_record(row))
            if len(result) >= limit:
                break
        return result

    def get_plan_for_job(self, job_id: str) -> PlanRecord | None:
        """Return the durable plan for a job, including plans already consumed."""
        row = self._conn().execute("SELECT * FROM plans WHERE job_id=?", (job_id,)).fetchone()
        return self._plan_record(row) if row is not None else None

    @staticmethod
    def _plan_record(row: sqlite3.Row) -> PlanRecord:
        return PlanRecord(row["plan_id"], row["job_id"],
                          row["current_task_key"] if "current_task_key" in row.keys() else row["task_key"],
                          row["plan_artifact_ref"],
                          row["source_episode_id"], row["source_receipt_ref"], row["status"],
                          int(row["seed_attempts"]), float(row["created_at"]))

    def claim_seed_plan(
        self,
        plan_id: str,
        *,
        task_allowlist: Sequence[str],
        run_id: str,
        attempt_id: str,
        engine_epoch: str,
        current_policy_version: int,
        policy_version: int,
        max_attempts: int = 2,
        lease_seconds: float = 86_400,
        whole_group: bool = False,
        include_retired: bool = False,
    ) -> SeedPlanClaim | None:
        """Atomically fence one bounded guided-seed attempt for an eligible plan."""
        if not all(isinstance(value, str) and value for value in (plan_id, run_id, attempt_id, engine_epoch)):
            raise ContractError("plan, run, attempt, and engine epoch identities are required")
        _check_positive_int(current_policy_version, "current_policy_version", allow_zero=True)
        _check_positive_int(policy_version, "policy_version", allow_zero=True)
        _check_positive_int(max_attempts, "max_attempts")
        if max_attempts != 2:
            raise ContractError("a seed generation has exactly two guided attempts")
        if type(whole_group) is not bool:
            raise ContractError("whole_group must be a boolean")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise ContractError("seed lease_seconds must be positive")
        if type(include_retired) is not bool or (include_retired and not whole_group):
            raise ContractError("include_retired requires a boolean whole-group request")
        keys = tuple(dict.fromkeys(task_allowlist))
        now = _now()
        with self._write() as conn:
            original = conn.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
            if original is None:
                raise KeyError(plan_id)
            matches = self._plans_for_tasks(conn, keys, plan_id=plan_id, limit=1)
            if not matches:
                raise ContractError("seed plan is outside the current run allowlist")
            plan = matches[0]
            if not include_retired and self.retired_rescue_task_keys([plan['current_task_key']], engine_epoch=engine_epoch):
                return None
            self._assert_verified_seed_plan(conn, plan)
            cache_state, _cache = self._current_cache_state(
                conn, task_key=plan["current_task_key"], engine_epoch=engine_epoch,
                current_policy_version=current_policy_version,
            )
            if cache_state == "pending" or (cache_state == "eligible" and not whole_group):
                return None
            generation = self._latest_seed_generation(conn, plan_id)
            if (generation is None or generation["engine_epoch"] != engine_epoch
                    or generation.get('task_key', plan['task_key']) != plan['current_task_key']):
                return None
            if current_policy_version < int(generation["start_policy_version"]):
                raise ContractError("current policy version precedes this engine epoch's seed generation")

            latest_claim = self._latest_seed_claim(conn, plan_id)
            if plan["status"] == "seed_running":
                if (latest_claim is not None and latest_claim.get("attempt_id") == attempt_id
                        and (latest_claim.get("run_id") != run_id
                             or latest_claim.get("engine_epoch") != engine_epoch
                             or latest_claim.get("generation_id") != generation["generation_id"])):
                    raise ConflictError("seed attempt id was reused with a different claim identity")
                if (latest_claim is not None
                        and latest_claim.get("engine_epoch") == engine_epoch
                        and latest_claim.get("generation_id") == generation["generation_id"]
                        and latest_claim.get("attempt_id") == attempt_id
                        and latest_claim.get("run_id") == run_id
                        and float(latest_claim.get("lease_expires_at", 0)) > now):
                    if int(latest_claim.get("policy_version", -1)) != policy_version:
                        raise ConflictError("seed attempt id was reused with a different policy version")
                    return self._seed_claim_record(plan, latest_claim)
                if latest_claim is not None and float(latest_claim.get("lease_expires_at", 0)) > now:
                    return None
                return None
            if plan["status"] != "ready":
                return None
            attempts = int(plan["seed_attempts"])
            if attempts >= max_attempts:
                conn.execute("UPDATE plans SET status='exhausted',updated_at=? WHERE plan_id=?", (now, plan_id))
                conn.execute("UPDATE rescue_jobs SET state='exhausted',fail_reason='attempt_budget_exhausted',updated_at=? WHERE job_id=?",
                             (now, plan["job_id"]))
                return None

            history = conn.execute(
                "SELECT details_json FROM events WHERE entity_type='plan' AND entity_id=? AND action='seed_claimed'",
                (plan_id,),
            ).fetchall()
            prior_claims = [_json_dict(row["details_json"]) for row in history if row["details_json"]]
            if any(item.get("attempt_id") == attempt_id for item in prior_claims):
                raise ConflictError("seed attempt id has already been used")
            previous_fence = max((int(item.get("fence", 0)) for item in prior_claims), default=0)
            fence = previous_fence + 1
            expires = now + float(lease_seconds)
            attempt_number = attempts + 1
            details = {
                "engine_epoch": engine_epoch,
                "task_key": plan["current_task_key"],
                "generation_id": generation["generation_id"],
                "run_id": run_id,
                "attempt_id": attempt_id,
                "fence": fence,
                "lease_expires_at": expires,
                "attempt_number": attempt_number,
                "max_attempts": max_attempts,
                "policy_version": policy_version,
                "eligibility_policy_version": current_policy_version,
            }
            conn.execute("UPDATE plans SET status='seed_running',seed_attempts=?,updated_at=? WHERE plan_id=?",
                         (attempt_number, now, plan_id))
            conn.execute("UPDATE rescue_jobs SET state='seed_running',fail_reason=NULL,updated_at=? WHERE job_id=?",
                         (now, plan["job_id"]))
            self._event(conn, "plan", plan_id, "seed_claimed", details=details)
            claimed = conn.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
            return self._seed_claim_record(claimed, details)

    def mark_seed_failed(
        self,
        claim: SeedPlanClaim,
        *,
        reason: str,
        max_attempts: int = 2,
    ) -> PlanRecord:
        """Record a fenced guided-run failure without losing the verified plan."""
        if not isinstance(claim, SeedPlanClaim) or not reason:
            raise ContractError("a current seed claim and typed failure reason are required")
        _check_positive_int(max_attempts, "max_attempts")
        if max_attempts != 2:
            raise ContractError("a seed generation has exactly two guided attempts")
        now = _now()
        with self._write() as conn:
            plan = self._assert_seed_claim(conn, claim, now=now)
            attempts = int(plan["seed_attempts"])
            status = "exhausted" if attempts >= max_attempts else "ready"
            conn.execute("UPDATE plans SET status=?,updated_at=? WHERE plan_id=?",
                         (status, now, claim.plan_id))
            conn.execute("UPDATE rescue_jobs SET state=?,fail_reason=?,updated_at=? WHERE job_id=?",
                         ("exhausted" if status == "exhausted" else "plan_ready", reason, now, claim.job_id))
            self._event(conn, "plan", claim.plan_id, "seed_failed", reason=reason,
                        details={"attempt_id": claim.attempt_id, "run_id": claim.run_id,
                                 "generation_id": claim.generation_id, "fence": claim.fence,
                                 "attempts": attempts, "max_attempts": max_attempts})
            return self._plan_record(conn.execute("SELECT * FROM plans WHERE plan_id=?", (claim.plan_id,)).fetchone())

    @staticmethod
    def _active_reservation_counts(conn: sqlite3.Connection, cache_entry_id: str) -> tuple[int, int]:
        row = conn.execute(
            "SELECT sum(CASE WHEN state='reserved' THEN 1 ELSE 0 END) AS reserved, "
            "sum(CASE WHEN state IN ('committed','commit_unknown') THEN 1 ELSE 0 END) AS committed "
            "FROM reservations WHERE cache_entry_id=?",
            (cache_entry_id,),
        ).fetchone()
        return int(row["reserved"] or 0), int(row["committed"] or 0)

    def _next_ready_sequence(self, conn: sqlite3.Connection) -> int:
        conn.execute("UPDATE counters SET value=value+1 WHERE name='ready_sequence'")
        return int(conn.execute("SELECT value FROM counters WHERE name='ready_sequence'").fetchone()[0])

    @staticmethod
    def _cache_record(conn: sqlite3.Connection, row: sqlite3.Row) -> CacheEntry:
        reserved, _committed = BepaStore._active_reservation_counts(conn, row["cache_entry_id"])
        return CacheEntry(
            cache_entry_id=row["cache_entry_id"], task_key=row["task_key"], engine_epoch=row["engine_epoch"],
            source_episode_id=row["source_episode_id"], source_policy_version=int(row["source_policy_version"]),
            source_policy_sha256=row["source_policy_sha256"], row_kind=row["row_kind"],
            raw_artifact_ref=row["raw_artifact_ref"], projection_artifact_ref=row["projection_artifact_ref"],
            receipt_artifact_ref=row["receipt_artifact_ref"], committed_uses=int(row["committed_uses"]),
            reserved_uses=reserved, ready_sequence=int(row["ready_sequence"]), active=bool(row["active"]),
        )

    def record_episode(
        self,
        episode: SourceEpisode,
        *,
        raw_artifact_ref: str,
        projection_artifact_ref: str | None = None,
        receipt_artifact_ref: str | None = None,
        exact: bool = False,
        reason: str | None = None,
    ) -> None:
        """Persist source evidence separately from whether it can enter cache."""
        require_sha256(raw_artifact_ref, "raw_artifact_ref")
        if projection_artifact_ref is not None:
            require_sha256(projection_artifact_ref, "projection_artifact_ref")
        if receipt_artifact_ref is not None:
            require_sha256(receipt_artifact_ref, "receipt_artifact_ref")
        with self._write() as conn:
            existing = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode.episode_id,)).fetchone()
            values = (episode.task_key, episode.source_policy.engine_epoch, episode.source_policy.version,
                      episode.source_policy.adapter_sha256, episode.row_kind, raw_artifact_ref,
                      projection_artifact_ref, receipt_artifact_ref, int(bool(exact)), reason)
            if existing is not None:
                old = tuple(existing[key] for key in ("task_key", "engine_epoch", "source_policy_version", "source_policy_sha256",
                                                       "episode_kind", "raw_artifact_ref", "projection_artifact_ref",
                                                       "receipt_artifact_ref", "exact", "reason"))
                if old != values:
                    raise ConflictError("episode id already exists with different source evidence")
                return
            if conn.execute("SELECT 1 FROM tasks WHERE task_key=?", (episode.task_key,)).fetchone() is None:
                raise KeyError(episode.task_key)
            conn.execute(
                "INSERT INTO episodes(episode_id,task_key,engine_epoch,source_policy_version,source_policy_sha256,episode_kind,"
                "raw_artifact_ref,projection_artifact_ref,receipt_artifact_ref,exact,reason,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (episode.episode_id, *values, _now()),
            )
            self._event(conn, "episode", episode.episode_id, "stored", reason=reason,
                        details={"task_key": episode.task_key, "exact": bool(exact)})

    def create_cache_entry(
        self,
        task_key: str,
        *,
        engine_epoch: str,
        episode: SourceEpisode,
        projection: SourceProjection,
        receipt: TrustedReceipt,
        raw_artifact_ref: str,
        projection_artifact_ref: str,
        receipt_artifact_ref: str,
        seed_claim: SeedPlanClaim | None = None,
    ) -> CacheEntry:
        """Atomically activate an exact, current-epoch student replay source.

        Existing reservations continue to pin their old entry. A refresh marks
        that entry inactive and inserts a new FIFO entry with zero uses.
        """
        for name, ref in (("raw_artifact_ref", raw_artifact_ref), ("projection_artifact_ref", projection_artifact_ref),
                          ("receipt_artifact_ref", receipt_artifact_ref)):
            require_sha256(ref, name)
        if not engine_epoch or episode.source_policy.engine_epoch != engine_epoch:
            raise ContractError("cache source must belong to the active engine epoch")
        if seed_claim is not None and episode.row_kind != "guided":
            raise ContractError("a seed claim can only bind a guided cache")
        if episode.task_key != task_key or projection.task_key != task_key or receipt.task_key != task_key:
            raise ContractError("cache source, projection, receipt, and task identities must match")
        if receipt.episode_id != episode.episode_id or projection.episode_id != episode.episode_id:
            raise ContractError("cache source episode identities must match")
        if not receipt.exact or receipt.verifier_id != "central_verifier":
            raise ContractError("only exact central-verifier receipts can activate cache entries")
        if projection.source_policy != episode.source_policy:
            raise ContractError("projection must preserve source policy identity")
        spec_for_projection = self.get_task(task_key)
        if spec_for_projection is None:
            raise KeyError(task_key)
        if project_source_episode(episode, spec_for_projection) != projection:
            raise ContractError("projection does not exactly match source prefix replacement")
        raw_artifact = self._verify_artifact_ref(raw_artifact_ref)
        projection_artifact = self._verify_artifact_ref(projection_artifact_ref)
        receipt_artifact = self._verify_artifact_ref(receipt_artifact_ref)
        if canonical_json_bytes(raw_artifact.get("episode")) != canonical_json_bytes(episode):
            raise ContractError("raw source artifact does not match the source episode envelope")
        if canonical_json_bytes(projection_artifact.get("projection")) != canonical_json_bytes(projection):
            raise ContractError("projection artifact does not match the verified projection")
        if canonical_json_bytes(receipt_artifact.get("receipt")) != canonical_json_bytes(receipt):
            raise ContractError("receipt artifact does not match the central verifier receipt")
        entry_id = str(uuid.uuid4())
        now = _now()
        with self._write() as conn:
            task = conn.execute("SELECT * FROM tasks WHERE task_key=?", (task_key,)).fetchone()
            if task is None:
                raise KeyError(task_key)
            if not task["allowed"]:
                raise ContractError("cannot cache a task outside the training allowlist")
            spec = self._task_spec_from_json(task["spec_json"])
            if receipt.private_target_hash != spec.private_target_hash:
                raise ContractError("trusted receipt targets do not match this task compatibility key")
            if receipt.verifier_contract_hash != content_hash(spec.verifier_contract):
                raise ContractError("trusted receipt verifier contract does not match this task")
            if projection.tokenizer_fingerprint != spec.tokenizer_fingerprint:
                raise ContractError("projection tokenizer does not match task compatibility")
            existing_episode = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode.episode_id,)).fetchone()
            if existing_episode is not None:
                expected = (task_key, engine_epoch, episode.source_policy.version, episode.source_policy.adapter_sha256,
                            episode.row_kind, raw_artifact_ref, projection_artifact_ref, receipt_artifact_ref, 1, None)
                actual = tuple(existing_episode[key] for key in ("task_key", "engine_epoch", "source_policy_version", "source_policy_sha256",
                                                                 "episode_kind", "raw_artifact_ref", "projection_artifact_ref",
                                                                 "receipt_artifact_ref", "exact", "reason"))
                if actual != expected:
                    raise ConflictError("episode id already exists with different cache evidence")
            else:
                conn.execute(
                    "INSERT INTO episodes(episode_id,task_key,engine_epoch,source_policy_version,source_policy_sha256,episode_kind,"
                    "raw_artifact_ref,projection_artifact_ref,receipt_artifact_ref,exact,created_at) VALUES(?,?,?,?,?,?,?,?,?,1,?)",
                    (episode.episode_id, task_key, engine_epoch, episode.source_policy.version,
                     episode.source_policy.adapter_sha256, episode.row_kind, raw_artifact_ref,
                     projection_artifact_ref, receipt_artifact_ref, now),
                )
            conn.execute("UPDATE cache_entries SET active=0 WHERE task_key=? AND engine_epoch=? AND active=1", (task_key, engine_epoch))
            ready_sequence = self._next_ready_sequence(conn)
            row_kind = "native_replay" if episode.row_kind == "native" else "guided_replay"
            conn.execute(
                "INSERT INTO cache_entries(cache_entry_id,task_key,engine_epoch,source_episode_id,source_policy_version,"
                "source_policy_sha256,row_kind,raw_artifact_ref,projection_artifact_ref,receipt_artifact_ref,ready_sequence,active,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,1,?)",
                (entry_id, task_key, engine_epoch, episode.episode_id, episode.source_policy.version,
                 episode.source_policy.adapter_sha256, row_kind, raw_artifact_ref, projection_artifact_ref,
                 receipt_artifact_ref, ready_sequence, now),
            )
            if seed_claim is not None:
                plan = self._assert_seed_claim(conn, seed_claim, now=now)
                if (seed_claim.task_key != task_key or seed_claim.engine_epoch != engine_epoch
                        or plan["job_id"] != seed_claim.job_id
                        or episode.source_policy.version != seed_claim.policy_version):
                    raise ContractError("guided cache success does not match the claimed plan or policy")
                conn.execute("UPDATE plans SET status='consumed',updated_at=? WHERE plan_id=?",
                             (now, seed_claim.plan_id))
                conn.execute("UPDATE rescue_jobs SET state='cache_ready',fail_reason=NULL,updated_at=? WHERE job_id=?",
                             (now, seed_claim.job_id))
                self._event(conn, "plan", seed_claim.plan_id, "seed_succeeded", details={
                    "job_id": seed_claim.job_id,
                    "cache_entry_id": entry_id,
                    "engine_epoch": seed_claim.engine_epoch,
                    "generation_id": seed_claim.generation_id,
                    "run_id": seed_claim.run_id,
                    "attempt_id": seed_claim.attempt_id,
                    "fence": seed_claim.fence,
                })
                self._event(conn, "plan", plan["plan_id"], "seed_cache_bound", details={
                    "job_id": seed_claim.job_id, "cache_entry_id": entry_id,
                    "engine_epoch": engine_epoch, "source_policy_version": episode.source_policy.version,
                })
            self._event(conn, "cache", entry_id, "activated", details={"task_key": task_key, "engine_epoch": engine_epoch,
                                                                         "source_policy_version": episode.source_policy.version,
                                                                         "row_kind": row_kind})
            row = conn.execute("SELECT * FROM cache_entries WHERE cache_entry_id=?", (entry_id,)).fetchone()
            return self._cache_record(conn, row)

    def peek_ready(
        self,
        task_allowlist: Sequence[str],
        *,
        engine_epoch: str,
        current_policy_version: int,
        exclude_task_keys: Sequence[str] = (),
    ) -> TaskRecord | None:
        """Read the oldest eligible cache task without reserving or consuming it."""
        _check_positive_int(current_policy_version, "current_policy_version", allow_zero=True)
        allowed_keys = tuple(dict.fromkeys(task_allowlist))
        if not allowed_keys:
            return None
        excludes = tuple(dict.fromkeys(exclude_task_keys))
        allowed_sql = ",".join("?" for _ in allowed_keys)
        exclude_clause = f"AND c.task_key NOT IN ({','.join('?' for _ in excludes)})" if excludes else ""
        rows = self._conn().execute(
            "SELECT c.*,t.spec_json,t.allowed,t.created_at AS task_created_at FROM cache_entries c JOIN tasks t USING(task_key) "
            f"WHERE c.active=1 AND c.engine_epoch=? AND t.allowed=1 AND c.task_key IN ({allowed_sql}) {exclude_clause} "
            "ORDER BY c.ready_sequence,c.created_at,c.cache_entry_id",
            (engine_epoch, *allowed_keys, *excludes),
        ).fetchall()
        for row in rows:
            age = current_policy_version - int(row["source_policy_version"])
            if age < 0 or age + 1 > self.max_age_updates:
                continue
            reserved, committed = self._active_reservation_counts(self._conn(), row["cache_entry_id"])
            if committed + reserved >= self.max_committed_uses:
                continue
            return TaskRecord(row["task_key"], self._task_spec_from_json(row["spec_json"]), True, float(row["task_created_at"]))
        return None

    def reserve_cache(
        self,
        task_key: str,
        *,
        engine_epoch: str,
        current_policy_version: int,
        batch_id: str,
        group_id: str,
        expected_cache_entry_id: str | None = None,
    ) -> Reservation | None:
        """Reserve exactly the selected failed task's cache; never substitute."""
        _check_positive_int(current_policy_version, "current_policy_version", allow_zero=True)
        if expected_cache_entry_id is not None and (not isinstance(expected_cache_entry_id, str) or not expected_cache_entry_id):
            raise ContractError("expected_cache_entry_id must be a non-empty string")
        if not batch_id or not group_id:
            raise ContractError("batch_id and group_id must be non-empty")
        now = _now()
        with self._write() as conn:
            duplicate = conn.execute("SELECT * FROM reservations WHERE batch_id=? AND group_id=?", (batch_id, group_id)).fetchone()
            if duplicate is not None:
                if duplicate["state"] == "released":
                    raise StateTransitionError("released reservation cannot be reused by the same batch/group")
                entry = conn.execute("SELECT * FROM cache_entries WHERE cache_entry_id=?", (duplicate["cache_entry_id"],)).fetchone()
                if duplicate["task_key"] != task_key:
                    raise ConflictError("batch/group reservation is already bound to a different task")
                if expected_cache_entry_id is not None and duplicate["cache_entry_id"] != expected_cache_entry_id:
                    raise ConflictError("batch/group reservation is already bound to a different cache")
                return self._reservation_record(conn, duplicate, entry)
            task = conn.execute("SELECT allowed FROM tasks WHERE task_key=?", (task_key,)).fetchone()
            if task is None:
                raise KeyError(task_key)
            if not task["allowed"]:
                raise ContractError("cache reservation task is outside the training allowlist")
            same_cohort = conn.execute(
                "SELECT 1 FROM reservations WHERE batch_id=? AND task_key=? AND state IN ('reserved','committed','commit_unknown')",
                (batch_id, task_key),
            ).fetchone()
            if same_cohort:
                raise ContractError("a task may appear only once in a cohort")
            cache = conn.execute(
                "SELECT * FROM cache_entries WHERE task_key=? AND engine_epoch=? AND active=1 ORDER BY ready_sequence DESC LIMIT 1",
                (task_key, engine_epoch),
            ).fetchone()
            if cache is None or (expected_cache_entry_id is not None
                                 and cache["cache_entry_id"] != expected_cache_entry_id):
                return None
            age = current_policy_version - int(cache["source_policy_version"])
            if age < 0 or age + 1 > self.max_age_updates:
                conn.execute("UPDATE cache_entries SET active=0 WHERE cache_entry_id=?", (cache["cache_entry_id"],))
                self._event(conn, "cache", cache["cache_entry_id"], "expired", reason=f"commit_age_updates={age + 1}")
                return None
            reserved, committed = self._active_reservation_counts(conn, cache["cache_entry_id"])
            if reserved + committed >= self.max_committed_uses:
                conn.execute("UPDATE cache_entries SET active=0 WHERE cache_entry_id=?", (cache["cache_entry_id"],))
                self._event(conn, "cache", cache["cache_entry_id"], "exhausted", reason="committed_plus_reserved_limit")
                return None
            reservation_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO reservations(reservation_id,batch_id,group_id,cache_entry_id,task_key,engine_epoch,state,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,'reserved',?,?)",
                (reservation_id, batch_id, group_id, cache["cache_entry_id"], task_key, engine_epoch, now, now),
            )
            conn.execute("UPDATE cache_entries SET ready_sequence=? WHERE cache_entry_id=?",
                         (self._next_ready_sequence(conn), cache["cache_entry_id"]))
            self._event(conn, "reservation", reservation_id, "reserved", details={"batch_id": batch_id,
                                                                                    "group_id": group_id,
                                                                                    "cache_entry_id": cache["cache_entry_id"]})
            row = conn.execute("SELECT * FROM reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
            return self._reservation_record(conn, row, cache)

    @staticmethod
    def _reservation_record(conn: sqlite3.Connection, row: sqlite3.Row, cache: sqlite3.Row) -> Reservation:
        return Reservation(
            reservation_id=row["reservation_id"], batch_id=row["batch_id"], group_id=row["group_id"],
            cache_entry_id=row["cache_entry_id"], task_key=row["task_key"], engine_epoch=row["engine_epoch"],
            source_policy=SourcePolicy(row["engine_epoch"], int(cache["source_policy_version"]), cache["source_policy_sha256"]),
            row_kind=cache["row_kind"], projection_artifact_ref=cache["projection_artifact_ref"],
            receipt_artifact_ref=cache["receipt_artifact_ref"], state=row["state"],
        )

    def release_reservation(self, reservation_id: str, *, reason: str = "precommit_release") -> None:
        """Release only work proven to be before optimizer entry."""
        with self._write() as conn:
            row = conn.execute("SELECT * FROM reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
            if row is None:
                raise KeyError(reservation_id)
            if row["state"] == "released":
                return
            if row["state"] != "reserved":
                raise StateTransitionError("committed or unknown reservations cannot be released")
            if row["update_attempt_id"] is not None:
                attempt = conn.execute("SELECT * FROM update_attempts WHERE attempt_id=?", (row["update_attempt_id"],)).fetchone()
                if attempt is not None and attempt["state"] != "prepared":
                    raise StateTransitionError("only a prepared update can release its reservation")
                if attempt is not None:
                    conn.execute("UPDATE reservations SET state='released',updated_at=? WHERE batch_id=? AND state='reserved'",
                                 (_now(), row["batch_id"]))
                    conn.execute("UPDATE update_attempts SET state='aborted',updated_at=? WHERE attempt_id=?", (_now(), attempt["attempt_id"]))
                    self._event(conn, "update", attempt["attempt_id"], "aborted", reason=reason)
                else:
                    conn.execute("UPDATE reservations SET state='released',updated_at=? WHERE reservation_id=?", (_now(), reservation_id))
            else:
                conn.execute("UPDATE reservations SET state='released',updated_at=? WHERE reservation_id=?", (_now(), reservation_id))
            self._event(conn, "reservation", reservation_id, "released", reason=reason)

    def release_batch_reservations(self, batch_id: str, *, reason: str = "precommit_release") -> int:
        """Release all reservations for a batch only if no update entered optimizer."""
        with self._write() as conn:
            attempt = conn.execute("SELECT * FROM update_attempts WHERE batch_id=?", (batch_id,)).fetchone()
            if attempt is not None and attempt["state"] != "prepared":
                raise StateTransitionError("batch has entered optimizer; reservations are isolated")
            cur = conn.execute("UPDATE reservations SET state='released',updated_at=? WHERE batch_id=? AND state='reserved'",
                               (_now(), batch_id))
            if attempt is not None:
                conn.execute("UPDATE update_attempts SET state='aborted',updated_at=? WHERE batch_id=?", (_now(), batch_id))
                self._event(conn, "update", attempt["attempt_id"], "aborted", reason=reason)
            self._event(conn, "batch", batch_id, "reservations_released", reason=reason, details={"count": cur.rowcount})
            return cur.rowcount

    def prepare_update(
        self,
        batch_id: str,
        *,
        input_batch_hash: str,
        engine_epoch: str,
        start_policy_version: int,
        attempt_id: str | None = None,
    ) -> UpdateAttempt:
        """Durably record prepared update and bind all batch reservations."""
        require_sha256(input_batch_hash, "input_batch_hash")
        _check_positive_int(start_policy_version, "start_policy_version", allow_zero=True)
        if not batch_id or not engine_epoch:
            raise ContractError("batch_id and engine_epoch must be non-empty")
        attempt_id = attempt_id or str(uuid.uuid4())
        now = _now()
        with self._write() as conn:
            existing = conn.execute("SELECT * FROM update_attempts WHERE batch_id=?", (batch_id,)).fetchone()
            if existing is not None:
                expected = (input_batch_hash, engine_epoch, start_policy_version)
                actual = (existing["input_batch_hash"], existing["engine_epoch"], existing["start_policy_version"])
                if actual != expected:
                    raise ConflictError("batch id already has a different update attempt contract")
                return self._update_record(existing)
            reservations = conn.execute("SELECT * FROM reservations WHERE batch_id=?", (batch_id,)).fetchall()
            invalid = [row["reservation_id"] for row in reservations if row["state"] != "reserved"]
            if invalid:
                raise StateTransitionError("all update reservations must remain reserved before preparation")
            if any(row["engine_epoch"] != engine_epoch for row in reservations):
                raise ContractError("update reservations cross engine epochs")
            conn.execute(
                "INSERT INTO update_attempts(attempt_id,batch_id,input_batch_hash,engine_epoch,start_policy_version,state,created_at,updated_at) "
                "VALUES(?,?,?,?,?,'prepared',?,?)",
                (attempt_id, batch_id, input_batch_hash, engine_epoch, start_policy_version, now, now),
            )
            conn.execute("UPDATE reservations SET update_attempt_id=?,updated_at=? WHERE batch_id=? AND state='reserved'",
                         (attempt_id, now, batch_id))
            self._event(conn, "update", attempt_id, "prepared", details={"batch_id": batch_id,
                                                                           "input_batch_hash": input_batch_hash,
                                                                           "start_policy_version": start_policy_version})
            return self._update_record(conn.execute("SELECT * FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone())

    @staticmethod
    def _update_record(row: sqlite3.Row) -> UpdateAttempt:
        evidence = _json_dict(row["commit_evidence_json"]) if row["commit_evidence_json"] else None
        return UpdateAttempt(row["attempt_id"], row["batch_id"], row["input_batch_hash"], row["engine_epoch"],
                             int(row["start_policy_version"]), row["state"],
                             int(row["new_policy_version"]) if row["new_policy_version"] is not None else None,
                             row["adapter_sha256"], evidence, float(row["created_at"]), float(row["updated_at"]))

    def validate_before_optimizer(self, attempt_id: str, *, current_policy_version: int) -> None:
        """Recheck cache age and use credits immediately before GPU optimizer entry."""
        _check_positive_int(current_policy_version, "current_policy_version", allow_zero=True)
        with self._write() as conn:
            attempt = conn.execute("SELECT * FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if attempt is None:
                raise KeyError(attempt_id)
            if attempt["state"] != "prepared":
                raise StateTransitionError("only a prepared attempt can be validated for optimizer entry")
            reservations = conn.execute("SELECT * FROM reservations WHERE update_attempt_id=?", (attempt_id,)).fetchall()
            for reservation in reservations:
                if reservation["state"] != "reserved":
                    raise StateTransitionError("update contains a non-reserved cache reference")
                cache = conn.execute("SELECT * FROM cache_entries WHERE cache_entry_id=?", (reservation["cache_entry_id"],)).fetchone()
                if cache is None or cache["engine_epoch"] != attempt["engine_epoch"]:
                    raise ContractError("update cache entry is missing or from a different engine epoch")
                age = current_policy_version - int(cache["source_policy_version"])
                if age < 0 or age + 1 > self.max_age_updates:
                    raise ContractError(f"cache entry commit age {age + 1} is outside the 0..{self.max_age_updates} update limit")
                reserved, committed = self._active_reservation_counts(conn, cache["cache_entry_id"])
                if committed + reserved > self.max_committed_uses:
                    raise ContractError(f"cache committed plus reserved uses exceed {self.max_committed_uses}")

    def mark_optimizer_entered(self, attempt_id: str, *, current_policy_version: int | None = None) -> UpdateAttempt:
        """Durably fence a batch before its first optimizer operation."""
        with self._write() as conn:
            attempt = conn.execute("SELECT * FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if attempt is None:
                raise KeyError(attempt_id)
            if attempt["state"] == "optimizer_entered":
                return self._update_record(attempt)
            if attempt["state"] != "prepared":
                raise StateTransitionError("only prepared updates can enter optimizer")
            current = int(attempt["start_policy_version"] if current_policy_version is None else current_policy_version)
            if current < int(attempt["start_policy_version"]):
                raise ContractError("current policy version precedes prepared update")
            for reservation in conn.execute("SELECT * FROM reservations WHERE update_attempt_id=?", (attempt_id,)).fetchall():
                if reservation["state"] != "reserved":
                    raise StateTransitionError("update contains a non-reserved cache reference")
                cache = conn.execute("SELECT * FROM cache_entries WHERE cache_entry_id=?", (reservation["cache_entry_id"],)).fetchone()
                age = current - int(cache["source_policy_version"])
                if age < 0 or age + 1 > self.max_age_updates:
                    raise ContractError(f"cache entry commit age {age + 1} is outside the 0..{self.max_age_updates} update limit")
                reserved, committed = self._active_reservation_counts(conn, cache["cache_entry_id"])
                if committed + reserved > self.max_committed_uses:
                    raise ContractError(f"cache committed plus reserved uses exceed {self.max_committed_uses}")
            conn.execute("UPDATE update_attempts SET state='optimizer_entered',updated_at=? WHERE attempt_id=?", (_now(), attempt_id))
            self._event(conn, "update", attempt_id, "optimizer_entered", details={"current_policy_version": current})
            return self._update_record(conn.execute("SELECT * FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone())

    def commit_update(
        self,
        attempt_id: str,
        *,
        new_policy_version: int,
        adapter_sha256: str,
        commit_evidence: Mapping[str, Any],
    ) -> UpdateAttempt:
        """Record known optimizer success and consume reservations durably.

        This is an application-level commit record, not an exactly-once claim
        about the GPU optimizer and SQLite forming one atomic transaction.
        """
        require_sha256(adapter_sha256, "adapter_sha256")
        _check_positive_int(new_policy_version, "new_policy_version")
        evidence = dict(commit_evidence)
        if not evidence:
            raise ContractError("commit_evidence must include durable learner evidence")
        canonical_json_bytes(evidence)
        with self._write() as conn:
            attempt = conn.execute("SELECT * FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if attempt is None:
                raise KeyError(attempt_id)
            if attempt["state"] == "committed":
                if (attempt["new_policy_version"], attempt["adapter_sha256"], attempt["commit_evidence_json"]) != (
                    new_policy_version, adapter_sha256, canonical_json_bytes(evidence).decode("utf-8")
                ):
                    raise ConflictError("committed update was retried with different commit evidence")
                return self._update_record(attempt)
            recovered_unknown = attempt["state"] == "commit_unknown"
            if attempt["state"] not in {"optimizer_entered", "commit_unknown"}:
                raise StateTransitionError("update must durably enter optimizer before commit")
            if recovered_unknown:
                snapshot_id = evidence.get("snapshot_id")
                if (
                    evidence.get("recovered_from_full_snapshot") is not True
                    or evidence.get("checkpoint_policy_version") != new_policy_version
                    or evidence.get("input_batch_hash") != attempt["input_batch_hash"]
                    or not isinstance(snapshot_id, str)
                    or len(snapshot_id) != 32
                    or any(char not in "0123456789abcdef" for char in snapshot_id)
                ):
                    raise StateTransitionError(
                        "commit_unknown requires an identity-matched full-checkpoint proof"
                    )
            if new_policy_version != int(attempt["start_policy_version"]) + 1:
                raise ContractError("successful update must advance policy_version exactly once")
            reservations = conn.execute("SELECT * FROM reservations WHERE update_attempt_id=?", (attempt_id,)).fetchall()
            for reservation in reservations:
                expected_state = "commit_unknown" if recovered_unknown else "reserved"
                if reservation["state"] != expected_state:
                    raise StateTransitionError("update contains a cache reference in an invalid commit state")
                cache = conn.execute("SELECT * FROM cache_entries WHERE cache_entry_id=?", (reservation["cache_entry_id"],)).fetchone()
                if new_policy_version - int(cache["source_policy_version"]) > self.max_age_updates:
                    # Preserve the known update fact and consume the reference;
                    # make the stale entry unavailable for future selections.
                    conn.execute("UPDATE cache_entries SET active=0 WHERE cache_entry_id=?", (cache["cache_entry_id"],))
                    self._event(conn, "cache", cache["cache_entry_id"], "expired_after_update", reason="age_limit_exceeded")
                conn.execute("UPDATE reservations SET state='committed',updated_at=? WHERE reservation_id=?",
                             (_now(), reservation["reservation_id"]))
                conn.execute("UPDATE cache_entries SET committed_uses=committed_uses+1 WHERE cache_entry_id=?",
                             (reservation["cache_entry_id"],))
                updated_cache = conn.execute("SELECT committed_uses FROM cache_entries WHERE cache_entry_id=?", (reservation["cache_entry_id"],)).fetchone()
                active_reserved, active_committed = self._active_reservation_counts(conn, reservation["cache_entry_id"])
                if (int(updated_cache["committed_uses"]) >= self.max_committed_uses
                        or active_reserved + active_committed >= self.max_committed_uses):
                    conn.execute("UPDATE cache_entries SET active=0 WHERE cache_entry_id=?", (reservation["cache_entry_id"],))
                    self._event(conn, "cache", reservation["cache_entry_id"], "exhausted",
                                reason=f"{self.max_committed_uses}_committed_or_reserved_uses")
            evidence_json = canonical_json_bytes(evidence).decode("utf-8")
            conn.execute(
                "UPDATE update_attempts SET state='committed',new_policy_version=?,adapter_sha256=?,commit_evidence_json=?,updated_at=? WHERE attempt_id=?",
                (new_policy_version, adapter_sha256, evidence_json, _now(), attempt_id),
            )
            self._event(
                conn,
                "update",
                attempt_id,
                "checkpoint_reconciled_commit_unknown" if recovered_unknown else "committed",
                details={
                    "new_policy_version": new_policy_version,
                    "adapter_sha256": adapter_sha256,
                    "reservations": len(reservations),
                    "recovered_from_full_snapshot": recovered_unknown,
                },
            )
            return self._update_record(conn.execute("SELECT * FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone())

    def mark_commit_unknown(self, attempt_id: str, *, reason: str) -> UpdateAttempt:
        """Quarantine reservations after optimizer entry without commit proof."""
        if not reason:
            raise ContractError("commit_unknown requires a typed reason")
        with self._write() as conn:
            attempt = conn.execute("SELECT * FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if attempt is None:
                raise KeyError(attempt_id)
            if attempt["state"] == "commit_unknown":
                return self._update_record(attempt)
            if attempt["state"] != "optimizer_entered":
                raise StateTransitionError("commit_unknown applies only after optimizer entry")
            conn.execute("UPDATE reservations SET state='commit_unknown',updated_at=? WHERE update_attempt_id=? AND state='reserved'",
                         (_now(), attempt_id))
            conn.execute("UPDATE cache_entries SET active=0 WHERE cache_entry_id IN "
                         "(SELECT cache_entry_id FROM reservations WHERE update_attempt_id=?)", (attempt_id,))
            conn.execute("UPDATE update_attempts SET state='commit_unknown',updated_at=? WHERE attempt_id=?", (_now(), attempt_id))
            self._event(conn, "update", attempt_id, "commit_unknown", reason=reason)
            return self._update_record(conn.execute("SELECT * FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone())

    def get_update_attempt(self, attempt_id: str) -> UpdateAttempt | None:
        row = self._conn().execute("SELECT * FROM update_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        return self._update_record(row) if row else None

    def latest_committed_update(self, *, engine_epoch: str) -> UpdateAttempt | None:
        """Return the latest committed update for one immutable engine lineage."""
        if not isinstance(engine_epoch, str) or not engine_epoch:
            raise ContractError("engine_epoch must be a non-empty string")
        row = self._conn().execute(
            "SELECT * FROM update_attempts WHERE engine_epoch=? AND state='committed' "
            "ORDER BY new_policy_version DESC, updated_at DESC LIMIT 1",
            (engine_epoch,),
        ).fetchone()
        return self._update_record(row) if row else None

    def recover_interrupted_updates(self) -> dict[str, list[str]]:
        """Resolve durable markers after restart without guessing optimizer outcome.

        A still-prepared attempt is known not to have crossed the durable entry
        marker, so its reservations can be released. An optimizer-entered
        attempt has no proof either way and is marked commit_unknown.
        """
        recovered = {"aborted_before_optimizer": [], "commit_unknown": [],
                     "released_orphan_reservations": []}
        with self._write() as conn:
            # Reservations are created while claiming a complete cohort, before
            # prepare_update binds them to an attempt. After process restart no
            # rollout can still own these unbound references, so they are safe
            # to release; optimizer-entered attempts stay on the proof path below.
            orphaned = conn.execute(
                "SELECT reservation_id,batch_id FROM reservations "
                "WHERE state='reserved' AND update_attempt_id IS NULL ORDER BY created_at,reservation_id"
            ).fetchall()
            for row in orphaned:
                conn.execute("UPDATE reservations SET state='released',updated_at=? WHERE reservation_id=?",
                             (_now(), row["reservation_id"]))
                self._event(conn, "reservation", row["reservation_id"], "released",
                            reason="restart_before_update_preparation",
                            details={"batch_id": row["batch_id"]})
                recovered["released_orphan_reservations"].append(row["reservation_id"])
            prepared = conn.execute("SELECT attempt_id,batch_id FROM update_attempts WHERE state='prepared'").fetchall()
            for row in prepared:
                conn.execute("UPDATE reservations SET state='released',updated_at=? WHERE batch_id=? AND state='reserved'",
                             (_now(), row["batch_id"]))
                conn.execute("UPDATE update_attempts SET state='aborted',updated_at=? WHERE attempt_id=?", (_now(), row["attempt_id"]))
                self._event(conn, "update", row["attempt_id"], "aborted", reason="restart_before_optimizer_entry")
                recovered["aborted_before_optimizer"].append(row["attempt_id"])
            entered = conn.execute("SELECT attempt_id FROM update_attempts WHERE state='optimizer_entered'").fetchall()
            for row in entered:
                conn.execute("UPDATE reservations SET state='commit_unknown',updated_at=? WHERE update_attempt_id=? AND state='reserved'",
                             (_now(), row["attempt_id"]))
                conn.execute("UPDATE cache_entries SET active=0 WHERE cache_entry_id IN "
                             "(SELECT cache_entry_id FROM reservations WHERE update_attempt_id=?)", (row["attempt_id"],))
                conn.execute("UPDATE update_attempts SET state='commit_unknown',updated_at=? WHERE attempt_id=?",
                             (_now(), row["attempt_id"]))
                self._event(conn, "update", row["attempt_id"], "commit_unknown", reason="restart_without_commit_proof")
                recovered["commit_unknown"].append(row["attempt_id"])
        return recovered

    def get_reservation(self, reservation_id: str) -> Reservation | None:
        conn = self._conn()
        row = conn.execute("SELECT * FROM reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
        if row is None:
            return None
        cache = conn.execute("SELECT * FROM cache_entries WHERE cache_entry_id=?", (row["cache_entry_id"],)).fetchone()
        return self._reservation_record(conn, row, cache)

    def get_cache_entry(self, cache_entry_id: str) -> CacheEntry | None:
        """Load one immutable cache identity and its current reservation count."""
        conn = self._conn()
        row = conn.execute("SELECT * FROM cache_entries WHERE cache_entry_id=?", (cache_entry_id,)).fetchone()
        return self._cache_record(conn, row) if row else None

    def get_job(self, job_id: str) -> JobRecord:
        row = self._conn().execute("SELECT * FROM rescue_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job_record(row)

    def get_job_by_public_key(self, budget_scope: str, public_job_key: str) -> JobRecord | None:
        """Look up the immutable rescue-job identity before applying queue limits."""
        row = self._conn().execute(
            "SELECT * FROM rescue_jobs WHERE budget_scope=? AND public_job_key=?",
            (budget_scope, public_job_key),
        ).fetchone()
        return self._job_record(row) if row is not None else None

    @staticmethod
    def _job_record(row: sqlite3.Row) -> JobRecord:
        refs = _json_dict(row["source_artifact_refs_json"]) if row["source_artifact_refs_json"] else None
        return JobRecord(
            job_id=row["job_id"], task_key=row["task_key"], public_job_key=row["public_job_key"],
            budget_scope=row["budget_scope"], state=row["state"], attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]), used_tokens=int(row["used_tokens"]),
            reserved_tokens=int(row["reserved_tokens"]), max_tokens=int(row["max_tokens"]),
            usage_unknown=bool(row["usage_unknown"]), fence=int(row["fence"]),
            lease_owner=row["lease_owner"], lease_expires_at=float(row["lease_expires_at"]) if row["lease_expires_at"] is not None else None,
            source_artifact_refs=refs, plan_artifact_ref=row["plan_artifact_ref"], fail_reason=row["fail_reason"],
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
        )

    def list_jobs(self, *, states: Sequence[str] | None = None, task_allowlist: Sequence[str] | None = None,
                  limit: int = 100, budget_scope: str | None = None,
                  unsettled_usage_only: bool = False, newest_first: bool = False) -> list[JobRecord]:
        """List safe job metadata; private targets and provider credentials are absent."""
        _check_positive_int(limit, "limit")
        clauses: list[str] = []
        args: list[Any] = []
        if unsettled_usage_only:
            clauses.append("usage_unknown=1 AND NOT EXISTS (SELECT 1 FROM events e "
                           "WHERE e.entity_type='job' AND e.entity_id=rescue_jobs.job_id "
                           "AND e.action='usage_lower_bound_settled')")
        if budget_scope is not None:
            clauses.append("budget_scope=?")
            args.append(budget_scope)
        if states:
            state_values = tuple(dict.fromkeys(states))
            if any(state not in {"queued", "retry_wait", "teacher_running", "source_submitted", "source_verified",
                                 "plan_ready", "seed_running", "cache_ready", "failed", "exhausted", "rejected",
                                 "cancelled", "outcome_unknown"} for state in state_values):
                raise ContractError("unknown rescue job state filter")
            clauses.append(f"state IN ({','.join('?' for _ in state_values)})")
            args.extend(state_values)
        if task_allowlist is not None:
            keys = tuple(dict.fromkeys(task_allowlist))
            if not keys:
                return []
            clauses.append(f"task_key IN ({','.join('?' for _ in keys)})")
            args.extend(keys)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        order = "created_at DESC,job_id DESC" if newest_first else "created_at,job_id"
        rows = self._conn().execute(f"SELECT * FROM rescue_jobs {where} ORDER BY {order} LIMIT ?", (*args, limit)).fetchall()
        return [self._job_record(row) for row in rows]

    def status(self) -> dict[str, Any]:
        """Return coordinator counts and budget state without task targets."""
        conn = self._conn()
        counts = {row["state"]: int(row["count"]) for row in conn.execute(
            "SELECT state,count(*) AS count FROM rescue_jobs GROUP BY state"
        ).fetchall()}
        active_cache = int(conn.execute("SELECT count(*) FROM cache_entries WHERE active=1").fetchone()[0])
        reservations = {row["state"]: int(row["count"]) for row in conn.execute(
            "SELECT state,count(*) AS count FROM reservations GROUP BY state"
        ).fetchall()}
        budgets = [dict(row) for row in conn.execute("SELECT * FROM budget_scopes ORDER BY created_at").fetchall()]
        for budget in budgets:
            budget.update(self._lower_bound_status(budget["scope_id"]))
        validations = [dict(row) for row in conn.execute(
            "SELECT e.entity_id AS job_id,e.reason,e.details_json,e.created_at FROM events e "
            "JOIN rescue_jobs j ON j.job_id=e.entity_id WHERE e.entity_type='job' "
            "AND e.action='source_validated' ORDER BY e.sequence DESC LIMIT 20").fetchall()]
        for item in validations:
            item.update(json.loads(item.pop('details_json')))
        failures = [dict(row) for row in conn.execute(
            "SELECT e.entity_id AS job_id,j.budget_scope,e.action AS attempt_state,"
            "j.state AS current_job_state,e.reason,e.details_json,e.created_at FROM events e "
            "JOIN rescue_jobs j ON j.job_id=e.entity_id WHERE e.entity_type='job' "
            "AND e.action IN ('failed','exhausted','outcome_unknown','retry_wait') "
            "ORDER BY e.sequence DESC LIMIT 20").fetchall()]
        for item in failures:
            raw = item.pop('details_json')
            details = {} if raw is None else _json_dict(raw)
            item.update({key: details.get(key) for key in
                         ('fence', 'evidence_ref', 'usage_tokens', 'retryable', 'safe_to_retry')})
        return {"jobs": counts, "active_cache_entries": active_cache, "reservations": reservations,
                "budgets": budgets, "recent_source_validations": validations,
                "recent_failed_attempts": failures}

    def diagnostics(
        self,
        *,
        budget_scope: str,
        engine_epoch: str,
        current_policy_version: int,
        registered_task_keys: Sequence[str] | None = None,
        run_id: str | None = None,
        held_task_keys: Sequence[str] | None = None,
        run_batch_ids: Sequence[str] | None = None,
        task_limit: int = 20,
    ) -> dict[str, Any]:
        """Return bounded read-only diagnostics from the metadata query module."""
        from .diagnostics import build_diagnostics

        return build_diagnostics(
            self, budget_scope=budget_scope, engine_epoch=engine_epoch,
            current_policy_version=current_policy_version, registered_task_keys=registered_task_keys,
            run_id=run_id,
            held_task_keys=held_task_keys, run_batch_ids=run_batch_ids,
            task_limit=task_limit,
        )

    def events(self, *, after_sequence: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        _check_positive_int(after_sequence, "after_sequence", allow_zero=True)
        _check_positive_int(limit, "limit")
        rows = self._conn().execute("SELECT * FROM events WHERE sequence>? ORDER BY sequence LIMIT ?",
                                    (after_sequence, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            if item["details_json"]:
                item["details"] = json.loads(item.pop("details_json"))
            else:
                item.pop("details_json")
                item["details"] = None
            result.append(item)
        return result
