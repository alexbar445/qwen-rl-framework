"""Durable scheduler metadata with process-local gradient payloads."""
from __future__ import annotations


import hashlib
import json
import math
import os
import secrets
import sqlite3
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from qwen_rl.contracts.gradients import (
    GradientContractError,
    GradientJob,
    PreparedWindow,
    guided_replay_matches_fresh_policy_version,
    validate_compute_profile,
)
from qwen_rl.contracts.gradient_partitions import STAGED_SCOPE as STAGED_GROUP_SCOPE
from qwen_rl.transport.tensor_codec import MAX_TENSOR_RAW_BYTES

from ...rewards import FAILURE_REWARD_VALUES
from .work_stealing import WorkStealingStore
from .memory_payloads import MemoryPayloads
from .memory_payloads import GradientMemoryPressure as GradientMemoryPressure


class GradientStoreError(RuntimeError):
    """A scheduler/store contract violation safe to return as an HTTP error."""


class GradientPayloadTooLarge(GradientStoreError):
    pass


class StaleGradientLease(GradientStoreError):
    pass


class GradientWindowConflict(GradientStoreError):
    pass


_ROWBUNDLE_SCHEMA = "gradient_rowbundle/v1"
_CONTRIBUTION_SCHEMA = "gradient_contribution/v1"
_AGGREGATION = "global_token_mean_contribution/v1"


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(raw: str | bytes) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_json_object_no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON number {value}")),
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def relocate_gradient_store_paths(
    database_path: str | Path,
    *,
    path_mappings: Mapping[str | Path, str | Path],
    source_artifact_root: str | Path,
    allowed_root: str | Path,
) -> dict[str, Any]:
    """Relocate only stored rowbundle/contribution paths in an imported store copy.

    ``allowed_root`` is the final imported artifact root. During atomic import,
    files are validated beneath ``database_path.parent / 'artifacts'`` before
    the containing tree is renamed into place.
    """
    if not isinstance(path_mappings, Mapping) or len(path_mappings) != 1:
        raise GradientStoreError("gradient store relocation requires one explicit run-root mapping")

    def absolute(value: str | Path, label: str) -> Path:
        if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
            raise GradientStoreError(f"{label} must be an absolute path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise GradientStoreError(f"{label} must be an absolute path")
        return Path(os.path.normpath(str(path)))

    source_run, target_run = next(iter(path_mappings.items()))
    source_run = absolute(source_run, "source run root")
    target_run = absolute(target_run, "target run root")
    source_root = absolute(source_artifact_root, "source artifact root")
    target_root = absolute(allowed_root, "target artifact root")
    try:
        source_relative = source_root.relative_to(source_run)
    except ValueError as exc:
        raise GradientStoreError("source artifact root is outside the mapped run root") from exc
    if target_run / source_relative != target_root:
        raise GradientStoreError("run-root mapping does not preserve the gradient artifact root")
    if source_run == target_run:
        raise GradientStoreError("gradient store relocation requires distinct source and target roots")

    database = absolute(database_path, "gradient store database")
    staging_root = database.parent / "artifacts"
    if database.name != "jobs.sqlite3" or database.parent != staging_root.parent:
        raise GradientStoreError("gradient store database is outside the imported run layout")
    if (database.parent == source_root.parent
            or database.parent.is_symlink() or database.is_symlink() or not database.is_file()
            or staging_root.is_symlink() or not staging_root.is_dir()):
        raise GradientStoreError("gradient store relocation requires a safe imported database and artifact tree")
    if target_root.exists() and (target_root.is_symlink() or not target_root.is_dir()):
        raise GradientStoreError("target gradient artifact root is unsafe")
    if any(Path(f"{database}{suffix}").exists() or Path(f"{database}{suffix}").is_symlink()
           for suffix in ("-wal", "-shm")):
        raise GradientStoreError("gradient store relocation requires a checkpointed SQLite database")

    def target_path(
        value: Any, category: str, *, required: bool, expected_size: Any,
    ) -> tuple[str, bool]:
        source_path = absolute(value, f"jobs.{category}_path")
        directory = {"rowbundle": "rowbundles", "contribution": "contributions"}[category]
        suffix = {"rowbundle": ".json", "contribution": ".tensor-payload"}[category]
        if source_path.suffix != suffix:
            raise GradientStoreError(f"jobs.{category}_path has an unsupported file suffix")
        source_category = source_root / directory
        target_category = target_root / directory
        if source_path.is_relative_to(source_category):
            relative = source_path.relative_to(source_category)
            destination = target_category / relative
        elif source_path.is_relative_to(target_category):
            destination = source_path
        else:
            raise GradientStoreError(f"jobs.{category}_path is outside the recorded artifact root")
        try:
            target_relative = destination.relative_to(target_root)
        except ValueError as exc:
            raise GradientStoreError(f"mapped jobs.{category}_path escapes the imported artifact root") from exc
        staged = staging_root / target_relative
        current = staging_root
        for component in target_relative.parts:
            current = current / component
            if current.is_symlink():
                raise GradientStoreError(f"mapped jobs.{category}_path contains a symlink")
        if staged.exists() and not stat.S_ISREG(staged.lstat().st_mode):
            raise GradientStoreError(f"mapped jobs.{category}_path is not a regular file")
        exists = staged.is_file()
        if required and not exists:
            raise GradientStoreError(f"active jobs.{category}_path is missing from the imported artifact tree")
        if exists and (type(expected_size) is not int or staged.stat().st_size != expected_size):
            raise GradientStoreError(f"mapped jobs.{category}_path size differs from SQLite metadata")
        if required and type(expected_size) is not int:
            raise GradientStoreError(f"active jobs.{category}_path has no persisted size")
        return str(destination), not exists

    connection = sqlite3.connect(database, timeout=1.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    rewrites: list[tuple[str, str, str, str, str | None]] = []
    rowbundle_rewrite_count = 0
    accepted_rewrite_count = 0
    missing_terminal_references = 0
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if not {"jobs", "windows", "store_metadata"} <= tables:
            raise GradientStoreError("imported gradient database schema is unsupported")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        required_columns = {
            "window_id", "attempt_id", "job_id", "state", "work_order",
            "rowbundle_path", "rowbundle_size", "accepted_path", "accepted_size",
        }
        if not required_columns <= columns:
            raise GradientStoreError("imported gradient database has no recognized path columns")
        triggers = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' AND tbl_name IN ('jobs','windows') LIMIT 1"
        ).fetchone()
        if triggers is not None:
            raise GradientStoreError("gradient store relocation refuses databases with path-column triggers")

        connection.execute("BEGIN IMMEDIATE")
        before = _gradient_store_relocation_counts(connection)
        rows = connection.execute(
            "SELECT j.window_id,j.attempt_id,j.job_id,j.state,j.rowbundle_path,j.rowbundle_size,"
            "j.accepted_path,j.accepted_size,"
            "w.state AS window_state FROM jobs j JOIN windows w USING(window_id,attempt_id) "
            "ORDER BY j.window_id,j.attempt_id,j.work_order"
        ).fetchall()
        total_rows = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if len(rows) != total_rows:
            raise GradientStoreError("gradient store jobs are not bound to a known window")
        terminal_states = {"applied", "cancelled", "failed"}
        for row in rows:
            active = row["window_state"] not in terminal_states
            rowbundle_path, rowbundle_missing = target_path(
                row["rowbundle_path"], "rowbundle", required=active,
                expected_size=row["rowbundle_size"],
            )
            accepted_path = row["accepted_path"]
            original_rowbundle_path = row["rowbundle_path"]
            original_accepted_path = accepted_path
            accepted_missing = False
            if accepted_path is not None and accepted_path.startswith("ram:"):
                # Destination startup fences lost volatile accepts before any apply.
                pass
            elif accepted_path is not None:
                if row["state"] != "accepted":
                    raise GradientStoreError("unaccepted job has a stored contribution path")
                accepted_path, accepted_missing = target_path(
                    accepted_path, "contribution", required=active,
                    expected_size=row["accepted_size"],
                )
            elif active and row["state"] == "accepted":
                raise GradientStoreError("active accepted job has no contribution path")
            if rowbundle_missing:
                missing_terminal_references += 1
            if accepted_missing:
                missing_terminal_references += 1
            rowbundle_changed = rowbundle_path != original_rowbundle_path
            accepted_changed = accepted_path != original_accepted_path
            if rowbundle_changed or accepted_changed:
                rewrites.append((
                    row["window_id"], row["attempt_id"], row["job_id"],
                    rowbundle_path, accepted_path,
                ))
                rowbundle_rewrite_count += int(rowbundle_changed)
                accepted_rewrite_count += int(accepted_changed)

        for window_id, attempt_id, job_id, rowbundle_path, accepted_path in rewrites:
            cursor = connection.execute(
                "UPDATE jobs SET rowbundle_path=?,accepted_path=? "
                "WHERE window_id=? AND attempt_id=? AND job_id=?",
                (rowbundle_path, accepted_path, window_id, attempt_id, job_id),
            )
            if cursor.rowcount != 1:
                raise GradientStoreError("gradient store job changed during path relocation")
        after = _gradient_store_relocation_counts(connection)
        if before != after:
            raise GradientStoreError("gradient store state counts changed during path relocation")
        connection.execute("COMMIT")
        mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if mode == "wal":
            checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is not None and checkpoint[0] != 0:
                raise GradientStoreError("gradient store WAL could not be checkpointed after relocation")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()

    if any(Path(f"{database}{suffix}").exists() or Path(f"{database}{suffix}").is_symlink()
           for suffix in ("-wal", "-shm")):
        raise GradientStoreError("gradient store SQLite sidecars remain after relocation")
    descriptor = os.open(database, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(database.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "schema": "qwen-rl-gradient-store-path-relocation/v1",
        "jobs": int(before["jobs"]),
        "rowbundle_paths_rewritten": rowbundle_rewrite_count,
        "accepted_paths_rewritten": accepted_rewrite_count,
        "jobs_with_path_rewrites": len(rewrites),
        "missing_terminal_references": missing_terminal_references,
        "job_states": before["job_states"],
        "window_states": before["window_states"],
    }


def _gradient_store_relocation_counts(connection: sqlite3.Connection) -> dict[str, Any]:
    jobs = int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
    windows = int(connection.execute("SELECT COUNT(*) FROM windows").fetchone()[0])
    job_states = {str(row[0]): int(row[1]) for row in connection.execute(
        "SELECT state,COUNT(*) FROM jobs GROUP BY state ORDER BY state"
    )}
    window_states = {str(row[0]): int(row[1]) for row in connection.execute(
        "SELECT state,COUNT(*) FROM windows GROUP BY state ORDER BY state"
    )}
    path_counts = connection.execute(
        "SELECT SUM(CASE WHEN rowbundle_path IS NOT NULL THEN 1 ELSE 0 END) AS rowbundles,"
        "SUM(CASE WHEN accepted_path IS NOT NULL THEN 1 ELSE 0 END) AS accepted "
        "FROM jobs"
    ).fetchone()
    return {
        "jobs": jobs, "windows": windows,
        "job_states": job_states, "window_states": window_states,
        "path_counts": {name: int(path_counts[name] or 0) for name in path_counts.keys()},
    }


def _safe_component(value: str, name: str) -> str:
    if (not isinstance(value, str) or not value or len(value) > 200
            or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in value)
            or value in {".", ".."}):
        raise ValueError(f"{name} must be a nonempty safe identifier")
    return value


def _normalize_compute_profile(value: Mapping[str, Any] | None) -> dict[str, Any]:
    try:
        return validate_compute_profile(value)
    except (GradientContractError, TypeError) as exc:
        raise GradientStoreError(f"invalid executor compute_profile: {exc}") from exc


def _rowbundle_bytes(job: GradientJob, source: str | Path,
                     compute_profile: Mapping[str, Any], max_payload_bytes: int,
                     group_size: int, *, window=None) -> tuple[bytes, str]:
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"rowbundle is not a regular file: {path}")
    if path.stat().st_size > max_payload_bytes:
        raise GradientPayloadTooLarge("rowbundle exceeds configured max_payload_bytes")
    try:
        value = _parse_json(path.read_bytes())
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise GradientStoreError(f"rowbundle is not valid JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != _ROWBUNDLE_SCHEMA:
        raise GradientStoreError(f"rowbundle must use schema {_ROWBUNDLE_SCHEMA}")
    if "_parallel_advantage" in value or "advantages" in value:
        raise GradientStoreError("rowbundle must not contain a centrally prepared advantage")
    row_group_size = value.get("group_size")
    if type(row_group_size) is not int or row_group_size != group_size:
        raise GradientStoreError("rowbundle group_size differs from the centrally configured group_size")
    if _normalize_compute_profile(value.get("compute_profile")) != dict(compute_profile):
        raise GradientStoreError("rowbundle compute_profile differs from the centrally registered profile")
    expected = {
        "window_id": job.window_id,
        "attempt_id": job.attempt_id,
        "lineage_id": job.lineage_id,
        "engine_epoch": job.engine_epoch,
        "job_id": job.job_id,
        "group_id": job.group_id,
        "owner_executor_id": job.owner_executor_id,
        "policy_version": job.policy_version,
        "n_window_tokens": job.n_window_tokens,
        "token_count": job.token_count,
        "layout_fingerprint": job.layout_fingerprint,
        "compute_profile_fingerprint": job.compute_profile_fingerprint,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise GradientStoreError("rowbundle identity/count/layout differs from its immutable GradientJob")
    source_worker_id = value.get("source_worker_id")
    if not isinstance(source_worker_id, str) or not source_worker_id.strip():
        raise GradientStoreError("rowbundle source_worker_id must be a nonempty worker identity")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) not in {group_size, group_size + 1}:
        raise GradientStoreError(
            f"one rowbundle must contain {group_size} fresh rows and at most one replay row"
        )
    seen: set[int] = set()
    fresh_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    fresh_versions: set[int] = set()
    causal_tokens = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise GradientStoreError(f"rowbundle row {index} must be an object")
        if "_parallel_advantage" in row or "advantages" in row:
            raise GradientStoreError("rowbundle must not contain a centrally prepared advantage")
        row_id = row.get("row_id")
        if row_id is not None:
            if type(row_id) is not int or row_id < 0 or row_id in seen:
                raise GradientStoreError(f"rowbundle row {index} has a duplicate or invalid row_id")
            seen.add(row_id)
        if row.get("group_id") != job.group_id:
            raise GradientStoreError(f"rowbundle row {index} is outside its complete group")
        row_kind = row.get("row_kind", "fresh_student")
        if row_kind == "fresh_student":
            fresh_rows.append(row)
            if row.get("source_worker_id") != source_worker_id:
                raise GradientStoreError("fresh candidate source_worker_id differs from its assigned owner worker")
            version = row.get("policy_version")
            if type(version) is not int or version not in {job.policy_version, job.policy_version - 1}:
                raise GradientStoreError("fresh candidate policy version is outside the allowed current/lag-one range")
            fresh_versions.add(version)
            behavior = row.get("behavior_logprobs")
            if not isinstance(behavior, list):
                raise GradientStoreError("fresh candidates require full behavior_logprobs")
        elif row_kind in {"native_replay", "guided_replay"}:
            replay_rows.append(row)
            if "behavior_logprobs" in row or "policy_version" in row:
                raise GradientStoreError("replay rows must not carry fresh behavior identity")
            replay_source = row.get("source_worker_id")
            if replay_source is not None and (not isinstance(replay_source, str) or not replay_source.strip()):
                raise GradientStoreError("replay source_worker_id must preserve a nonempty identity when present")
        else:
            raise GradientStoreError(f"rowbundle row {index} has unsupported row_kind={row_kind!r}")
        input_ids, policy_mask = row.get("input_ids"), row.get("policy_mask")
        if (not isinstance(input_ids, list) or not isinstance(policy_mask, list)
                or len(input_ids) < 2 or len(policy_mask) != len(input_ids)):
            raise GradientStoreError(f"rowbundle row {index} has malformed input_ids/policy_mask")
        if any(type(token) is not int or token < 0 for token in input_ids):
            raise GradientStoreError(f"rowbundle row {index} input_ids must be nonnegative integers")
        if any(flag not in (0, 1, False, True) for flag in policy_mask):
            raise GradientStoreError(f"rowbundle row {index} policy_mask must contain only 0/1 values")
        if bool(policy_mask[0]):
            raise GradientStoreError(f"rowbundle row {index} selects position zero as a causal token")
        if row_kind == "fresh_student":
            behavior = row["behavior_logprobs"]
            if (len(behavior) != len(input_ids)
                    or any(isinstance(item, bool) or not isinstance(item, (int, float))
                           or not math.isfinite(float(item)) for item in behavior)):
                raise GradientStoreError("fresh behavior_logprobs must be full-length and finite")
        reward = row.get("reward")
        if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(float(reward))):
            raise GradientStoreError(f"rowbundle row {index} reward must be finite")
        rewards.append(float(reward))
        causal_tokens += sum(int(bool(flag)) for flag in policy_mask[1:])
    if (len(fresh_rows) != group_size or len(replay_rows) > 1
            or len(rows) != group_size + len(replay_rows) or len(fresh_versions) != 1):
        raise GradientStoreError(
            f"a gradient job requires {group_size} fresh candidates and at most one replay"
        )
    if replay_rows:
        replay_kind = replay_rows[0]["row_kind"]
        fresh_rewards = [float(row["reward"]) for row in fresh_rows]
        if replay_kind == "native_replay":
            if any(reward not in FAILURE_REWARD_VALUES for reward in fresh_rewards):
                raise GradientStoreError(
                    "a native replay group requires -1, -0.5 or 0 rewards on its original fresh candidates"
                )
        else:
            if any(reward not in FAILURE_REWARD_VALUES and reward < 1.0
                   for reward in fresh_rewards):
                raise GradientStoreError(
                    "a guided replay group requires fresh failure rewards or exact successes"
                )
            if (any(reward >= 1.0 for reward in fresh_rewards)
                    and not guided_replay_matches_fresh_policy_version(fresh_rows, replay_rows[0])):
                raise GradientStoreError(
                    "a guided replay with native success requires matching fresh and replay source policy versions"
                )
    if not any(reward > 0.0 for reward in rewards):
        raise GradientStoreError("complete group requires at least one positive reward")
    if len(set(rewards)) < 2:
        raise GradientStoreError("complete group rewards must have nonzero variance")
    if window is not None:
        from qwen_rl.contracts.gradient_partitions import selected_rows
        chosen = selected_rows(window, job, rows)
        causal_tokens = sum(sum(row["policy_mask"][1:]) for row in chosen)
    if causal_tokens != job.token_count:
        raise GradientStoreError(
            f"rowbundle causal policy token count {causal_tokens} differs from central job count {job.token_count}"
        )
    canonical = _canonical_json(value)
    # This digest is a semantic identity for immutable row JSON, not a tensor/checkpoint digest.
    return canonical, hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class LeaseClaim:
    job: GradientJob
    window: PreparedWindow
    executor_id: str
    lineage_id: str
    engine_epoch: str
    source_worker_id: str
    lease_id: str
    lease_epoch: int
    lease_seconds: float
    rowbundle_hash: str
    rowbundle_size: int
    compute_profile: dict[str, Any]


@dataclass
class ContributionStage:
    job_id: str
    window_id: str
    attempt_id: str
    executor_id: str
    lease_id: str
    lease_epoch: int
    expected_metadata: dict[str, Any]
    stream: Any
    size: int = 0
    finished: bool = False

    def write(self, chunk: bytes, max_payload_bytes: int) -> None:
        if self.finished:
            raise GradientStoreError("upload stage is already closed")
        if not isinstance(chunk, bytes):
            raise TypeError("request body chunks must be bytes")
        if self.size + len(chunk) > max_payload_bytes:
            raise GradientPayloadTooLarge("gradient contribution exceeds configured max_payload_bytes")
        self.stream.write(chunk)
        self.size += len(chunk)


class SQLiteGradientJobStore(WorkStealingStore):
    """Single-host scheduler state with SQLite fencing and fsync/rename payload commits."""

    def __init__(
        self,
        database_path: str | Path,
        artifact_dir: str | Path,
        *,
        group_size: int = 3,
        lease_seconds: float = 60.0,
        max_payload_bytes: int = MAX_TENSOR_RAW_BYTES,
        max_memory_bytes: int | None = None,
        max_inflight_input_tokens: int | None = None,
    ):
        if type(group_size) is not int or group_size < 2:
            raise ValueError("group_size must be an integer >= 2")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if type(max_payload_bytes) is not int or max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be a positive integer")
        self._memory_payloads = MemoryPayloads(max_memory_bytes)
        if max_inflight_input_tokens is not None and (
                type(max_inflight_input_tokens) is not int or max_inflight_input_tokens < 1):
            raise ValueError('max_inflight_input_tokens must be null or a positive integer')
        self.max_inflight_input_tokens = max_inflight_input_tokens
        self.database_path = Path(database_path)
        self.artifact_dir = Path(artifact_dir)
        self.group_size = group_size
        self.work_stealing_pools = {}
        self.remote_executor_ids = frozenset()
        self.remote_min_queued_trajectories = 1
        self.episode_queue = None
        self.rowbundle_dir = self.artifact_dir / "rowbundles"
        self.contribution_dir = self.artifact_dir / "contributions"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.rowbundle_dir.mkdir(parents=True, exist_ok=True)
        self.contribution_dir.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = float(lease_seconds)
        self.max_payload_bytes = max_payload_bytes
        self._lock = threading.RLock()
        # Sampling completion is a live controller attestation, never restored
        # from an old process's scheduler metadata.
        self._sampling_rounds = {}
        self._changed = threading.Condition(self._lock)
        self._initialize()
        # Persist prune intent before reconcile checks artifact references. This
        # also finishes cleanup interrupted after any prior terminal transition.
        self.prune_terminal_rowbundles()
        self.reconcile()
        self.release_terminal_window_payloads()

    def clear_memory_payloads(self) -> None:
        """Release process-local tensor bytes after the engine has drained its users."""
        self._memory_payloads.clear()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _db(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._db() as db:
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('windows','jobs','store_metadata')"
            )}
            expected_tables = {"windows", "jobs", "store_metadata"}
            if tables and tables != expected_tables:
                raise GradientStoreError(
                    "existing scheduler database is incomplete or incompatible; use a fresh store directory"
                )
            required_columns = {
                "windows": {
                    "window_id", "attempt_id", "window_json", "state", "compute_profile_json",
                    "receipt_json", "error", "created_at", "updated_at",
                },
                "jobs": {
                    "window_id", "attempt_id", "job_id", "job_json", "group_id", "owner_executor_id",
                    "source_worker_id", "rowbundle_path", "rowbundle_hash",
                    "rowbundle_size", "work_order", "state", "executor_id", "lease_id", "lease_epoch",
                    "lease_expires_at", "accepted_path", "accepted_size", "accepted_metadata_json",
                    "accepted_receipt_json", "accepted_at",
                },
                "store_metadata": {"key", "value"},
            }
            for table in tables:
                columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
                if not required_columns[table].issubset(columns):
                    raise GradientStoreError(
                        f"existing scheduler database has an unsupported {table} schema; use a fresh store directory"
                    )
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS windows (
                    window_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    window_json TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending',
                    compute_profile_json TEXT NOT NULL,
                    receipt_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (window_id, attempt_id)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    window_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    job_json TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    owner_executor_id TEXT NOT NULL,
                    source_worker_id TEXT NOT NULL,
                    rowbundle_path TEXT NOT NULL,
                    rowbundle_hash TEXT NOT NULL,
                    rowbundle_size INTEGER NOT NULL,
                    rowbundle_pruned_at REAL,
                    work_order INTEGER NOT NULL,
                    state TEXT NOT NULL DEFAULT 'queued',
                    executor_id TEXT,
                    lease_id TEXT,
                    lease_epoch INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at REAL,
                    accepted_path TEXT,
                    accepted_size INTEGER,
                    accepted_metadata_json TEXT,
                    accepted_receipt_json TEXT,
                    accepted_at REAL,
                    PRIMARY KEY (window_id, attempt_id, job_id),
                    FOREIGN KEY (window_id, attempt_id) REFERENCES windows(window_id, attempt_id)
                );
                CREATE INDEX IF NOT EXISTS jobs_claim_order
                    ON jobs(state, lease_expires_at, window_id, attempt_id, work_order);
                CREATE TABLE IF NOT EXISTS compute_sessions (
                    executor_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    round_id TEXT NOT NULL, active INTEGER NOT NULL,
                    parked INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            # Existing sessions are never assumed to have returned their GPU.
            # The zero default makes an old active row wait for a fresh receipt.
            db.execute("BEGIN IMMEDIATE")
            try:
                session_columns = {
                    row["name"]: row for row in db.execute("PRAGMA table_info(compute_sessions)")
                }
                required_session_columns = {"executor_id", "session_id", "round_id", "active"}
                if not required_session_columns.issubset(session_columns):
                    raise GradientStoreError("existing compute session table has an unsupported schema")
                parked_column = session_columns.get("parked")
                if parked_column is None:
                    db.execute(
                        "ALTER TABLE compute_sessions ADD COLUMN parked INTEGER NOT NULL DEFAULT 0"
                    )
                elif (str(parked_column["type"]).upper() != "INTEGER"
                      or parked_column["notnull"] != 1
                      or str(parked_column["dflt_value"]).strip("()") != "0"):
                    raise GradientStoreError("compute_sessions.parked must be INTEGER NOT NULL DEFAULT 0")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            job_columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            if "rowbundle_pruned_at" not in job_columns:
                db.execute("ALTER TABLE jobs ADD COLUMN rowbundle_pruned_at REAL")

    @classmethod
    def discard_window_for_managed_stop(
        cls,
        database_path: str | Path,
        window: PreparedWindow,
        *,
        operation_id: str,
    ) -> dict[str, Any]:
        """Fence one Core-identified uncommitted window after a managed stop.

        This recovery-only primitive opens the existing database directly. It
        does not initialize/reconcile the store, scan artifacts, or touch other
        windows. Accepted payloads and receipts remain in place as immutable
        evidence; all job leases are invalidated in the same SQLite transaction.
        """
        if not isinstance(window, PreparedWindow):
            raise TypeError("managed stop discard requires a validated PreparedWindow")
        _safe_component(window.window_id, "window_id")
        _safe_component(window.attempt_id, "attempt_id")
        _safe_component(operation_id, "operation_id")
        path = Path(database_path)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return {"state": "missing", "changed": False, "jobs": 0}
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            raise GradientStoreError("managed stop gradient database is unsafe")

        connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            connection.execute("BEGIN IMMEDIATE")
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('windows','jobs','store_metadata')"
            )}
            if tables != {"windows", "jobs", "store_metadata"}:
                raise GradientStoreError("managed stop gradient database schema is incomplete")
            metadata = connection.execute(
                "SELECT value FROM store_metadata WHERE key='central_identity'"
            ).fetchone()
            identity = None if metadata is None else _parse_json(metadata["value"])
            expected_identity = {"lineage_id": window.lineage_id, "engine_epoch": window.engine_epoch}
            if identity != expected_identity:
                raise GradientStoreError("managed stop window belongs to another Core identity")
            row = connection.execute(
                "SELECT window_json,state,receipt_json,error FROM windows "
                "WHERE window_id=? AND attempt_id=?",
                (window.window_id, window.attempt_id),
            ).fetchone()
            jobs = connection.execute(
                "SELECT job_id,job_json,state,lease_epoch FROM jobs "
                "WHERE window_id=? AND attempt_id=? ORDER BY work_order",
                (window.window_id, window.attempt_id),
            ).fetchall()
            if row is None:
                if jobs:
                    raise GradientStoreError("managed stop jobs exist without their registered window")
                connection.execute("COMMIT")
                return {"state": "missing", "changed": False, "jobs": 0}
            try:
                stored_window = PreparedWindow.from_json(row["window_json"])
            except Exception as exc:
                raise GradientStoreError("managed stop stored window is invalid") from exc
            if stored_window.to_dict() != window.to_dict():
                raise GradientStoreError("managed stop Core and SQLite windows differ")
            expected_jobs = {job.job_id: job.to_json() for job in window.jobs}
            actual_jobs = {job["job_id"]: job for job in jobs}
            if (len(jobs) != len(expected_jobs) or set(actual_jobs) != set(expected_jobs)
                    or any(actual_jobs[job_id]["job_json"] != encoded
                           for job_id, encoded in expected_jobs.items())):
                raise GradientStoreError("managed stop SQLite job set differs from the Core window")
            state = row["state"]
            if state == "applied":
                connection.execute("COMMIT")
                return {"state": "applied", "changed": False, "jobs": len(jobs),
                        "receipt_json": row["receipt_json"]}
            if state == "cancelled":
                if any(job["state"] not in {"cancelled", "accepted"} for job in jobs):
                    raise GradientStoreError("cancelled managed stop window has an active job lease")
                connection.execute("COMMIT")
                return {"state": "cancelled", "changed": False, "jobs": len(jobs),
                        "reason": row["error"]}
            if state == "failed":
                if any(job["state"] in {"queued", "leased"} for job in jobs):
                    raise GradientStoreError("failed managed stop window retains active job leases")
                connection.execute("COMMIT")
                return {"state": "failed", "changed": False, "jobs": len(jobs)}
            if state not in {"pending", "applying"} or row["receipt_json"] is not None:
                raise GradientStoreError("managed stop window has an unsupported transaction state")
            if any(job["state"] not in {"queued", "leased", "accepted", "cancelled"}
                   for job in jobs):
                raise GradientStoreError("managed stop window has an unsupported job state")
            reason = f"managed_stop_now:{operation_id}"
            now = time.time()
            connection.execute(
                "UPDATE jobs SET state='cancelled',executor_id=NULL,lease_id=NULL,"
                "lease_epoch=lease_epoch+1,lease_expires_at=NULL "
                "WHERE window_id=? AND attempt_id=? AND state!='cancelled'",
                (window.window_id, window.attempt_id),
            )
            connection.execute(
                "UPDATE windows SET state='cancelled',error=?,updated_at=? "
                "WHERE window_id=? AND attempt_id=?",
                (reason, now, window.window_id, window.attempt_id),
            )
            connection.execute("COMMIT")
            return {"state": "cancelled", "changed": True, "jobs": len(jobs),
                    "previous_state": state, "reason": reason}
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @classmethod
    def reconcile_committed_window_for_managed_stop(
        cls,
        database_path: str | Path,
        *,
        window_id: str,
        attempt_id: str,
        expected_identity: Mapping[str, str],
        expected_fingerprint: str,
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reconcile one SQLite window only when the selected Core snapshot proves commit."""
        _safe_component(window_id, "window_id")
        _safe_component(attempt_id, "attempt_id")
        if (not isinstance(expected_identity, Mapping)
                or set(expected_identity) != {"lineage_id", "engine_epoch"}
                or any(not isinstance(value, str) for value in expected_identity.values())
                or not isinstance(expected_fingerprint, str)
                or len(expected_fingerprint) != 64
                or not isinstance(receipt, Mapping)):
            raise GradientStoreError("committed stop receipt identity is malformed")
        required = {
            "schema", "window_id", "attempt_id", "policy_version", "optimizer_steps",
            "scheduler_steps", "n_window_tokens", "job_count", "gradient_norm_before_clip",
            "max_grad_norm", "checkpoint_generation",
        }
        if set(receipt) != required or receipt.get("window_id") != window_id or receipt.get("attempt_id") != attempt_id:
            raise GradientStoreError("committed stop receipt does not identify the selected window")
        if any(type(receipt.get(key)) is not int or receipt[key] < 0 for key in (
            "policy_version", "optimizer_steps", "scheduler_steps", "checkpoint_generation",
        )):
            raise GradientStoreError("committed stop receipt counters are invalid")
        path = Path(database_path)
        try:
            info = path.lstat()
        except FileNotFoundError:
            raise GradientStoreError("committed Core snapshot has no gradient database") from None
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            raise GradientStoreError("committed stop gradient database is unsafe")
        connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            connection.execute("BEGIN IMMEDIATE")
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('windows','jobs','store_metadata')"
            )}
            if tables != {"windows", "jobs", "store_metadata"}:
                raise GradientStoreError("committed stop gradient database schema is incomplete")
            metadata = connection.execute(
                "SELECT value FROM store_metadata WHERE key='central_identity'"
            ).fetchone()
            identity = None if metadata is None else _parse_json(metadata["value"])
            if identity != dict(expected_identity):
                raise GradientStoreError("committed stop database belongs to another Core identity")
            row = connection.execute(
                "SELECT window_json,state,receipt_json FROM windows WHERE window_id=? AND attempt_id=?",
                (window_id, attempt_id),
            ).fetchone()
            jobs = connection.execute(
                "SELECT job_id,job_json,state FROM jobs WHERE window_id=? AND attempt_id=? ORDER BY work_order",
                (window_id, attempt_id),
            ).fetchall()
            if row is None:
                raise GradientStoreError("committed Core window is absent from the gradient database")
            try:
                window = PreparedWindow.from_json(row["window_json"])
            except Exception as exc:
                raise GradientStoreError("committed stop stored window is invalid") from exc
            encoded_window = json.dumps(
                window.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
            if (window.window_id != window_id or window.attempt_id != attempt_id
                    or window.lineage_id != expected_identity["lineage_id"]
                    or window.engine_epoch != expected_identity["engine_epoch"]
                    or hashlib.sha256(encoded_window).hexdigest() != expected_fingerprint):
                raise GradientStoreError("committed stop SQLite window differs from the Core intent")
            expected_jobs = {job.job_id: job.to_json() for job in window.jobs}
            actual_jobs = {job["job_id"]: job for job in jobs}
            if (len(jobs) != len(expected_jobs) or set(actual_jobs) != set(expected_jobs)
                    or any(actual_jobs[job_id]["job_json"] != encoded
                           for job_id, encoded in expected_jobs.items())):
                raise GradientStoreError("committed stop SQLite jobs differ from the Core window")
            if (int(receipt["job_count"]) != len(jobs)
                    or int(receipt["n_window_tokens"]) != window.n_window_tokens
                    or receipt["policy_version"] != window.policy_version + 1
                    or receipt["optimizer_steps"] != receipt["scheduler_steps"]):
                raise GradientStoreError("committed stop receipt counters differ from its window")
            receipt_json = _canonical_json(dict(receipt)).decode("utf-8")
            if row["state"] == "applied":
                if row["receipt_json"] != receipt_json:
                    raise GradientStoreError("applied SQLite receipt differs from the durable Core receipt")
                connection.execute("COMMIT")
                return {"state": "applied", "changed": False, "jobs": len(jobs)}
            if row["state"] not in {"pending", "applying"} or row["receipt_json"] is not None:
                raise GradientStoreError("committed Core snapshot conflicts with SQLite window state")
            if any(job["state"] != "accepted" for job in jobs):
                raise GradientStoreError("committed Core window lacks complete accepted contributions")
            connection.execute(
                "UPDATE windows SET state='applied',receipt_json=?,error=NULL,updated_at=? "
                "WHERE window_id=? AND attempt_id=?",
                (receipt_json, time.time(), window_id, attempt_id),
            )
            connection.execute("COMMIT")
            return {"state": "applied", "changed": True, "jobs": len(jobs)}
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @classmethod
    def inspect_staged_windows_for_managed_stop(
        cls,
        database_path: str | Path,
        *,
        expected_identity: Mapping[str, str],
        excluded_windows: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
        durable_policy_version: int,
    ) -> dict[str, Any]:
        """Read only current-identity staged windows without store reconciliation."""
        path = Path(database_path)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return {"staged_windows": [], "other_pending_windows": []}
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            raise GradientStoreError("managed stop gradient database is unsafe")
        if (not isinstance(expected_identity, Mapping)
                or set(expected_identity) != {"lineage_id", "engine_epoch"}
                or any(not isinstance(value, str) for value in expected_identity.values())
                or type(durable_policy_version) is not int or durable_policy_version < 0):
            raise GradientStoreError("managed stop staged-window identity is malformed")
        connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            connection.execute("BEGIN")
            metadata = connection.execute(
                "SELECT value FROM store_metadata WHERE key='central_identity'"
            ).fetchone()
            identity = None if metadata is None else _parse_json(metadata["value"])
            if identity != dict(expected_identity):
                raise GradientStoreError("managed stop staged store belongs to another Core identity")
            windows = connection.execute(
                "SELECT window_id,attempt_id,window_json,state,receipt_json FROM windows "
                "WHERE state IN ('pending','applying') ORDER BY created_at,window_id,attempt_id"
            ).fetchall()
            staged: list[dict[str, Any]] = []
            other: list[dict[str, str]] = []
            for row in windows:
                try:
                    window = PreparedWindow.from_json(row["window_json"])
                except Exception as exc:
                    raise GradientStoreError("managed stop found a malformed pending gradient window") from exc
                if (window.lineage_id != expected_identity["lineage_id"]
                        or window.engine_epoch != expected_identity["engine_epoch"]):
                    continue
                key = (window.window_id, window.attempt_id)
                if key in excluded_windows:
                    continue
                scope = (window.update_context or {}).get("gradient_scope")
                if scope != STAGED_GROUP_SCOPE:
                    other.append({"window_id": window.window_id, "attempt_id": window.attempt_id,
                                  "state": row["state"]})
                    continue
                if row["state"] != "pending" or row["receipt_json"] is not None:
                    raise GradientStoreError("staged window has an optimizer-authoritative SQLite state")
                if window.policy_version > durable_policy_version:
                    raise GradientStoreError("staged window is bound to a future Core policy")
                jobs = connection.execute(
                    "SELECT job_id,job_json,state,accepted_path,accepted_size "
                    "FROM jobs WHERE window_id=? AND attempt_id=? ORDER BY work_order",
                    key,
                ).fetchall()
                expected_jobs = {job.job_id: job.to_json() for job in window.jobs}
                actual_jobs = {job["job_id"]: job for job in jobs}
                if (len(jobs) != len(expected_jobs) or set(actual_jobs) != set(expected_jobs)
                        or any(actual_jobs[job_id]["job_json"] != encoded
                               for job_id, encoded in expected_jobs.items())
                        or any(job["state"] not in {"queued", "leased", "accepted", "cancelled"}
                               for job in jobs)):
                    raise GradientStoreError("staged window jobs differ from their Core identity")
                staged.append({
                    "window_id": window.window_id,
                    "attempt_id": window.attempt_id,
                    "lineage_id": window.lineage_id,
                    "engine_epoch": window.engine_epoch,
                    "policy_version": window.policy_version,
                    "fingerprint": hashlib.sha256(window.to_json().encode("utf-8")).hexdigest(),
                    "window": window.to_dict(),
                    "job_count": len(jobs),
                    "accepted_artifacts": [
                        {"job_id": job["job_id"], "path": job["accepted_path"],
                         "size_bytes": job["accepted_size"]}
                        for job in jobs if job["accepted_path"] is not None
                    ],
                    "job_states": {state: sum(job["state"] == state for job in jobs)
                                   for state in ("queued", "leased", "accepted", "cancelled")},
                })
            connection.execute("COMMIT")
            return {"staged_windows": staged, "other_pending_windows": other}
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @classmethod
    def discard_staged_windows_for_managed_stop(
        cls,
        database_path: str | Path,
        *,
        expected_identity: Mapping[str, str],
        expected_windows: list[Mapping[str, Any]],
        operation_id: str,
    ) -> dict[str, Any]:
        """Cancel current-identity pre-optimizer windows and retain receipt metadata.

        Gradient payloads are regenerable intermediates, so this records their
        cancellation metadata without copying or hashing potentially large files.
        Existing terminal-window cleanup can remove those payload paths later.
        """
        if (not isinstance(expected_identity, Mapping)
                or set(expected_identity) != {"lineage_id", "engine_epoch"}
                or not isinstance(expected_windows, list)
                or not isinstance(operation_id, str) or len(operation_id) != 32
                or any(character not in "0123456789abcdef" for character in operation_id)):
            raise GradientStoreError("managed stop staged discard identity is invalid")
        database = Path(database_path)
        try:
            info = database.lstat()
        except FileNotFoundError:
            if expected_windows:
                raise GradientStoreError("staged window database disappeared during managed stop")
            return {"cancelled_windows": [], "payload_disposition": "metadata_only"}
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            raise GradientStoreError("managed stop gradient database is unsafe")
        connection = sqlite3.connect(database, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            connection.execute("BEGIN IMMEDIATE")
            metadata = connection.execute(
                "SELECT value FROM store_metadata WHERE key='central_identity'"
            ).fetchone()
            identity = None if metadata is None else _parse_json(metadata["value"])
            if identity != dict(expected_identity):
                raise GradientStoreError("managed stop staged store belongs to another Core identity")
            expected_keys = {
                (str(item.get("window_id")), str(item.get("attempt_id")))
                for item in expected_windows if isinstance(item, Mapping)
            }
            current_staged: set[tuple[str, str]] = set()
            for pending in connection.execute(
                "SELECT window_id,attempt_id,window_json FROM windows WHERE state='pending'"
            ).fetchall():
                try:
                    prepared = PreparedWindow.from_json(pending["window_json"])
                except Exception as exc:
                    raise GradientStoreError("managed stop found a malformed pending gradient window") from exc
                if (prepared.lineage_id == expected_identity["lineage_id"]
                        and prepared.engine_epoch == expected_identity["engine_epoch"]
                        and (prepared.update_context or {}).get("gradient_scope") == STAGED_GROUP_SCOPE):
                    current_staged.add((pending["window_id"], pending["attempt_id"]))
            if not current_staged.issubset(expected_keys):
                raise GradientStoreError("a staged window appeared after managed stop inspection")
            results = []
            for candidate in expected_windows:
                if not isinstance(candidate, Mapping):
                    raise GradientStoreError("managed stop staged window receipt is malformed")
                window_data = candidate.get("window")
                if not isinstance(window_data, Mapping):
                    raise GradientStoreError("managed stop staged window identity is missing")
                try:
                    window = PreparedWindow.from_dict(window_data)
                except Exception as exc:
                    raise GradientStoreError("managed stop staged window identity is invalid") from exc
                if (window.lineage_id != expected_identity["lineage_id"]
                        or window.engine_epoch != expected_identity["engine_epoch"]
                        or (window.update_context or {}).get("gradient_scope") != STAGED_GROUP_SCOPE
                        or hashlib.sha256(window.to_json().encode("utf-8")).hexdigest()
                        != candidate.get("fingerprint")):
                    raise GradientStoreError("managed stop staged window differs from its inspection receipt")
                key = (window.window_id, window.attempt_id)
                row = connection.execute(
                    "SELECT window_json,state,receipt_json,error FROM windows "
                    "WHERE window_id=? AND attempt_id=?", key,
                ).fetchone()
                if row is None:
                    raise GradientStoreError("managed stop staged window disappeared")
                try:
                    stored = PreparedWindow.from_json(row["window_json"])
                except Exception as exc:
                    raise GradientStoreError("managed stop stored staged window is invalid") from exc
                if stored.to_dict() != window.to_dict():
                    raise GradientStoreError("managed stop staged window changed after inspection")
                jobs = connection.execute(
                    "SELECT job_id,job_json,state,accepted_path,accepted_size,accepted_receipt_json "
                    "FROM jobs WHERE window_id=? AND attempt_id=? ORDER BY work_order", key,
                ).fetchall()
                expected_jobs = {job.job_id: job.to_json() for job in window.jobs}
                actual_jobs = {job["job_id"]: job for job in jobs}
                if (len(jobs) != len(expected_jobs) or set(actual_jobs) != set(expected_jobs)
                        or any(actual_jobs[job_id]["job_json"] != encoded
                               for job_id, encoded in expected_jobs.items())):
                    raise GradientStoreError("managed stop staged jobs differ from their prepared window")
                reason = f"managed_stop_now_staged:{operation_id}"
                expected_artifacts = candidate.get("accepted_artifacts")
                if not isinstance(expected_artifacts, list):
                    raise GradientStoreError("managed stop staged artifact receipt is malformed")
                expected_by_job = {item.get("job_id"): item for item in expected_artifacts
                                   if isinstance(item, Mapping)}
                if len(expected_by_job) != len(expected_artifacts):
                    raise GradientStoreError("managed stop staged artifact receipt has duplicate jobs")
                actual_artifacts = {
                    job["job_id"]: {"path": job["accepted_path"], "size_bytes": job["accepted_size"]}
                    for job in jobs if job["accepted_path"] is not None
                }
                expected_actual = {
                    item["job_id"]: {"path": item.get("path"), "size_bytes": item.get("size_bytes")}
                    for item in expected_artifacts
                }
                if row["state"] == "cancelled" and row["error"] == reason:
                    if any(job["state"] != "cancelled" for job in jobs):
                        raise GradientStoreError("managed stop staged cancellation has active jobs")
                    if actual_artifacts and any(
                        job_id not in expected_actual
                        or expected_actual[job_id] != value
                        for job_id, value in actual_artifacts.items()
                    ):
                        raise GradientStoreError("staged artifact metadata changed after cancellation")
                    results.append({"window_id": window.window_id, "attempt_id": window.attempt_id,
                                    "state": "cancelled", "changed": False,
                                    "accepted_payload_count": len(expected_artifacts)})
                    continue
                if row["state"] != "pending" or row["receipt_json"] is not None:
                    raise GradientStoreError("staged window is no longer pre-optimizer and pending")
                if any(job["state"] not in {"queued", "leased", "accepted", "cancelled"}
                       for job in jobs):
                    raise GradientStoreError("managed stop staged window has an unsupported job state")
                if actual_artifacts != expected_actual:
                    raise GradientStoreError("staged accepted artifacts changed after inspection")
                contribution_root = (database.parent / "artifacts" / "contributions").resolve()
                for job in jobs:
                    raw_path = job["accepted_path"]
                    if raw_path is None or raw_path.startswith("ram:"):
                        # Process-local tensors cannot survive the stopped owner.
                        continue
                    source = Path(raw_path)
                    if source.is_symlink() or not source.is_file():
                        raise GradientStoreError("managed stop staged contribution is missing or unsafe")
                    resolved = source.resolve(strict=True)
                    if not resolved.is_relative_to(contribution_root):
                        raise GradientStoreError("managed stop staged contribution escapes its store")
                    if source.stat().st_size != job["accepted_size"]:
                        raise GradientStoreError("managed stop staged contribution size changed")
                now = time.time()
                connection.execute(
                    "UPDATE jobs SET state='cancelled',executor_id=NULL,lease_id=NULL,"
                    "lease_epoch=lease_epoch+1,lease_expires_at=NULL "
                    "WHERE window_id=? AND attempt_id=? AND state!='cancelled'", key,
                )
                connection.execute(
                    "UPDATE windows SET state='cancelled',error=?,updated_at=? "
                    "WHERE window_id=? AND attempt_id=?",
                    (reason, now, window.window_id, window.attempt_id),
                )
                results.append({"window_id": window.window_id, "attempt_id": window.attempt_id,
                                "state": "cancelled", "changed": True})
            connection.execute("COMMIT")
            return {"cancelled_windows": results,
                    "payload_disposition": "metadata_only_normal_terminal_cleanup_may_remove_regenerable_gradients"}
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _fsync_directory(self, directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _stored_rowbundle_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(self.rowbundle_dir.resolve()):
            raise GradientStoreError("stored rowbundle path escapes its artifact directory")
        if path.exists() and not path.is_file():
            raise GradientStoreError("stored rowbundle path is not a regular file")
        return path

    def _install_rowbundle(self, job: GradientJob, source: str | Path,
                           compute_profile: Mapping[str, Any], *, window=None) -> tuple[Path, str, int]:
        canonical, digest = _rowbundle_bytes(
            job, source, compute_profile, self.max_payload_bytes, self.group_size, window=window,
        )
        if len(canonical) > self.max_payload_bytes:
            raise GradientPayloadTooLarge("rowbundle exceeds configured max_payload_bytes")
        destination = self.rowbundle_dir / f"{digest}.json"
        if destination.exists():
            if destination.stat().st_size != len(canonical) or destination.read_bytes() != canonical:
                raise GradientStoreError("content-addressed rowbundle path contains different bytes")
            return destination, digest, len(canonical)
        descriptor, temp_name = tempfile.mkstemp(prefix=".rowbundle-", suffix=".part", dir=self.rowbundle_dir)
        temp = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(canonical)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, destination)
            destination.chmod(0o444)
            self._fsync_directory(self.rowbundle_dir)
        finally:
            temp.unlink(missing_ok=True)
        return destination, digest, len(canonical)

    def enqueue_window(self, window: PreparedWindow, rowbundles: Mapping[str, str | Path], *,
                       compute_profile: Mapping[str, Any] | None = None) -> None:
        """Persist a prepared window and its centrally validated immutable row JSON."""
        if not isinstance(window, PreparedWindow):
            raise TypeError("window must be a PreparedWindow")
        lineage_id, engine_epoch = window.lineage_id, window.engine_epoch
        _safe_component(lineage_id, "lineage_id")
        _safe_component(engine_epoch, "engine_epoch")
        profile = _normalize_compute_profile(
            compute_profile if compute_profile is not None else getattr(window, "compute_profile", None)
        )
        if getattr(window, "compute_profile", profile) != profile:
            raise GradientStoreError("registered compute_profile differs from PreparedWindow")
        _safe_component(window.window_id, "window_id")
        _safe_component(window.attempt_id, "attempt_id")
        jobs = tuple(window.jobs)
        job_ids = [job.job_id for job in jobs]
        if len(job_ids) != len(set(job_ids)) or not jobs:
            raise GradientStoreError("prepared window must have unique jobs")
        if set(rowbundles) != set(job_ids):
            raise GradientStoreError("rowbundle paths must cover every job exactly once")
        work_order = tuple(getattr(window, "work_order", job_ids))
        if work_order != tuple(job_ids):
            raise GradientStoreError("jobs must be registered in the PreparedWindow deterministic work order")

        owner_worker_map = self.executor_worker_ids()
        prepared: list[tuple[GradientJob, Path, str, int, int, str]] = []
        seen_row_ids: set[int] = set()
        for order, job in enumerate(jobs):
            if (job.window_id != window.window_id or job.attempt_id != window.attempt_id
                    or job.lineage_id != lineage_id or job.engine_epoch != engine_epoch
                    or job.policy_version != window.policy_version
                    or job.n_window_tokens != window.n_window_tokens
                    or job.layout_fingerprint != window.layout_fingerprint):
                raise GradientStoreError("job identity/version/token-layout differs from PreparedWindow")
            _safe_component(job.job_id, "job_id")
            _safe_component(job.owner_executor_id, "owner_executor_id")
            if getattr(job, "compute_profile", profile) != profile:
                raise GradientStoreError("job compute_profile differs from PreparedWindow")
            path, digest, size = self._install_rowbundle(job, rowbundles[job.job_id], profile, window=window)
            value = _parse_json(path.read_bytes())
            from qwen_rl.contracts.gradient_partitions import selected_rows
            for row in selected_rows(window, job, value["rows"]):
                row_id = row.get("row_id")
                if row_id is not None and row_id in seen_row_ids:
                    raise GradientStoreError("row_id appears in more than one job in this window")
                if row_id is not None:
                    seen_row_ids.add(row_id)
            source_worker_id = value["source_worker_id"]
            from ...worker.episode_queue import validate_sampling_placement
            for candidate in value["rows"]:
                if candidate.get("row_kind", "fresh_student") == "fresh_student":
                    try:
                        validate_sampling_placement(candidate, source_worker_id, self.episode_queue)
                    except ValueError as exc:
                        raise GradientStoreError(str(exc)) from exc
            if owner_worker_map and owner_worker_map.get(job.owner_executor_id) != source_worker_id:
                raise GradientStoreError("job owner executor does not map to the rowbundle fresh source Worker")
            prepared.append((job, path, digest, size, order, source_worker_id))
        if sum(job.token_count for job in jobs) != window.n_window_tokens:
            raise GradientStoreError("central job token counts do not sum to the window denominator")

        window_json = window.to_json()
        now = time.time()
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                existing = db.execute(
                    "SELECT window_json,compute_profile_json FROM windows WHERE window_id=? AND attempt_id=?",
                    (window.window_id, window.attempt_id),
                ).fetchone()
                if existing is not None:
                    if (existing["window_json"] != window_json
                            or _parse_json(existing["compute_profile_json"] or "null") != profile):
                        raise GradientWindowConflict("window attempt was already registered with different contents")
                    rows = db.execute(
                        "SELECT job_id,rowbundle_hash FROM jobs WHERE window_id=? AND attempt_id=?",
                        (window.window_id, window.attempt_id),
                    ).fetchall()
                    registered = {row["job_id"]: row["rowbundle_hash"] for row in rows}
                    if registered != {job.job_id: digest for job, _, digest, _, _, _ in prepared}:
                        raise GradientWindowConflict("rowbundle identity changed for an existing attempt")
                    db.execute("COMMIT")
                    return
                identity = {"lineage_id": lineage_id, "engine_epoch": engine_epoch}
                stored_identity = db.execute(
                    "SELECT value FROM store_metadata WHERE key='central_identity'"
                ).fetchone()
                if stored_identity is not None and _parse_json(stored_identity["value"]) != identity:
                    raise GradientWindowConflict(
                        "scheduler database is bound to another central optimizer lineage/engine epoch"
                    )
                if stored_identity is None:
                    db.execute(
                        "INSERT INTO store_metadata(key,value) VALUES('central_identity',?)",
                        (json.dumps(identity, sort_keys=True, separators=(",", ":")),),
                    )
                collisions = db.execute(
                    "SELECT job_id FROM jobs WHERE job_id IN ("
                    + ",".join("?" for _ in job_ids) + ") LIMIT 1",
                    tuple(job_ids),
                ).fetchone()
                if collisions is not None:
                    raise GradientWindowConflict("job_id values must be globally unique across registered windows")
                stored_owners = db.execute(
                    "SELECT value FROM store_metadata WHERE key='executor_worker_ids'"
                ).fetchone()
                bound_owner_map = _parse_json(stored_owners["value"]) if stored_owners else {}
                if bound_owner_map and any(
                    job.owner_executor_id not in bound_owner_map
                    or bound_owner_map[job.owner_executor_id] != source_worker_id
                    for job, _, _, _, _, source_worker_id in prepared
                ):
                    raise GradientWindowConflict("job ownership differs from the persisted executor/Worker map")
                for _, path, _, size, _, _ in prepared:
                    if not path.is_file() or path.stat().st_size != size:
                        raise GradientStoreError("immutable rowbundle disappeared before job registration")
                db.execute(
                    "INSERT INTO windows(window_id,attempt_id,window_json,state,compute_profile_json,created_at,updated_at) "
                    "VALUES(?,?,?,'pending',?,?,?)",
                    (window.window_id, window.attempt_id, window_json,
                     json.dumps(profile, sort_keys=True, separators=(",", ":")), now, now),
                )
                for job, path, digest, size, order, source_worker_id in prepared:
                    db.execute(
                        "INSERT INTO jobs(window_id,attempt_id,job_id,job_json,group_id,owner_executor_id,"
                        "source_worker_id,rowbundle_path,rowbundle_hash,rowbundle_size,work_order,state) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,'queued')",
                        (window.window_id, window.attempt_id, job.job_id, job.to_json(), job.group_id,
                         job.owner_executor_id, source_worker_id, str(path), digest, size, order),
                    )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            self._changed.notify_all()

    def reserve_episode_rebalance(self, key, available):
        """Central-only admission: gradient work takes precedence over migration."""
        with self._lock, self._db() as db:
            scheduler = self.episode_queue
            if scheduler is None:
                return None
            outstanding = db.execute(
                "SELECT 1 FROM jobs j JOIN windows w USING(window_id,attempt_id) "
                "WHERE w.state IN ('pending','applying') AND j.state IN ('queued','leased') LIMIT 1"
            ).fetchone()
            if outstanding:
                scheduler.cancel_rebalance_plans()
                return None
            return scheduler.rebalance_target(key, available)

    def claim(self, executor_id: str, *, help_ready=False, session_id=None,
              pending_upload=None) -> LeaseClaim | None:
        _safe_component(executor_id, "executor_id")
        if pending_upload is not None and (
                not isinstance(pending_upload, dict)
                or set(pending_upload) != {"job_id", "lease_id", "lease_epoch"}
                or not isinstance(pending_upload["job_id"], str)
                or not isinstance(pending_upload["lease_id"], str)
                or type(pending_upload["lease_epoch"]) is not int):
            raise ValueError("pending upload requires an exact job lease identity")
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                # Lock/SQLite contention is not part of an executor's lease.
                now = time.time()
                active = db.execute("SELECT j.*,w.window_json FROM jobs j JOIN windows w "
                                    "USING(window_id,attempt_id) WHERE executor_id=? AND j.state='leased' "
                                    "AND lease_expires_at>?", (executor_id, now)).fetchall()
                pending_window = None
                if len(active) == 1 and pending_upload is not None:
                    prior = active[0]
                    if (prior['job_id'] != pending_upload['job_id']
                            or prior['lease_id'] != pending_upload['lease_id']
                            or prior['lease_epoch'] != pending_upload['lease_epoch']):
                        raise StaleGradientLease("pending upload lease does not belong to this executor")
                    pending_window = json.loads(prior['window_json'])
                if active and (pending_window is None or len(active) > 1
                               or 'trajectory_partitions' not in (pending_window.get('update_context') or {})):
                    db.execute("COMMIT")
                    return None
                candidates = db.execute(
                    "SELECT j.*,w.window_json,w.compute_profile_json,w.created_at AS window_created_at FROM jobs j JOIN windows w USING(window_id,attempt_id) "
                    "WHERE w.state IN ('pending','applying') AND j.owner_executor_id=? "
                    "AND (j.state='queued' OR (j.state='leased' AND j.lease_expires_at<=?)) "
                    "ORDER BY w.created_at,j.work_order",
                    (executor_id, now),
                ).fetchall()
                candidates = [row for row in candidates
                              if self._sampling_allows_compute(db, json.loads(row['window_json']), executor_id)]
                if help_ready:
                    candidates.extend(self._help_candidates(db, executor_id))
                # Preserve FIFO between rounds while comparing all eligible
                # owners within a round. Stable sorting keeps each query's
                # existing tie order for jobs without a declared cost.
                candidates.sort(key=lambda item: item['window_created_at'])
                if pending_window is not None:
                    previous_context = pending_window.get('update_context') or {}
                    def compatible_pending(row):
                        candidate = json.loads(row['window_json'])
                        context = candidate.get('update_context') or {}
                        return ('trajectory_partitions' in context
                                and context.get('staged_round_id') == previous_context.get('staged_round_id')
                                and all(candidate.get(key) == pending_window.get(key)
                                        for key in ('lineage_id', 'engine_epoch', 'policy_version')))
                    candidates = [row for row in candidates if compatible_pending(row)]
                session = db.execute("SELECT * FROM compute_sessions WHERE executor_id=? AND active=1",
                                     (executor_id,)).fetchone()
                if session is not None and session['session_id'] == session_id and not session['parked']:
                    candidates = [row for row in candidates if
                                  (json.loads(row['window_json']).get('update_context') or {}).get('staged_round_id')
                                  == session['round_id']]
                from .work_stealing import longest_available_partition
                if (executor_id in self.remote_executor_ids
                        and self.remote_min_queued_trajectories > 1 and candidates):
                    from .work_stealing import _staged_round_key
                    oldest = _staged_round_key(json.loads(candidates[0]['window_json']))
                    if oldest is not None:
                        candidates = [row for row in candidates
                                      if _staged_round_key(json.loads(row['window_json'])) == oldest]
                        group_running = any(
                            len(state['outcomes']) < len(state['plan'][3])
                            for state in self._sampling_rounds.values()
                        )
                        waiting = sum(row['state'] == 'queued' for row in candidates)
                        if waiting < self.remote_min_queued_trajectories and not group_running:
                            db.execute('COMMIT')
                            return None
                if self.max_inflight_input_tokens is not None:
                    from .work_stealing import partition_peak_input_tokens, _staged_round_key
                    # Reserve at the same atomic boundary as the job lease.
                    # Expired leases still hold memory until an owned terminal
                    # transition proves the compute has stopped; do not reassign
                    # that work while its physical liveness remains unknown.
                    live = db.execute(
                        "SELECT j.job_id,w.window_json FROM jobs j JOIN windows w "
                        "USING(window_id,attempt_id) WHERE j.state='leased'"
                    ).fetchall()
                    costs = [partition_peak_input_tokens(item) for item in live]
                    remaining = self.max_inflight_input_tokens - sum(costs)
                    if candidates:
                        oldest = _staged_round_key(json.loads(candidates[0]['window_json']))
                        candidates = [item for item in candidates if
                                      _staged_round_key(json.loads(item['window_json'])) == oldest]
                    candidates = [item for item in candidates
                                  if item['state'] == 'queued'
                                  and partition_peak_input_tokens(item) <= remaining]
                row = longest_available_partition(candidates)
                if row is None:
                    db.execute("COMMIT")
                    return None
                if pending_window is not None:
                    candidate = json.loads(row['window_json'])
                    context = candidate.get('update_context') or {}
                    previous_context = pending_window.get('update_context') or {}
                    if ('trajectory_partitions' not in context
                            or context.get('staged_round_id') != previous_context.get('staged_round_id')
                            or any(candidate.get(key) != pending_window.get(key)
                                   for key in ('lineage_id', 'engine_epoch', 'policy_version'))):
                        db.execute('COMMIT')
                        return None
                from contextlib import nullcontext
                scheduler = self.episode_queue
                worker = self.executor_worker_ids().get(executor_id)
                admission = (scheduler.compute_admission(worker, session_id)
                             if scheduler is not None and worker is not None else nullcontext(True))
                with admission as allowed:
                    if not allowed:
                        db.execute("COMMIT")
                        return None
                    self._open_work_session(db, executor_id, session_id, row["window_json"])
                    lease_id = secrets.token_urlsafe(24)
                    epoch = int(row["lease_epoch"]) + 1
                    expires_at = time.time() + self.lease_seconds
                    db.execute(
                        "UPDATE jobs SET state='leased',executor_id=?,lease_id=?,lease_epoch=?,lease_expires_at=? "
                        "WHERE window_id=? AND attempt_id=? AND job_id=?",
                        (executor_id, lease_id, epoch, expires_at, row["window_id"], row["attempt_id"], row["job_id"]),
                    )
                    db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        job = GradientJob.from_json(row["job_json"])
        return LeaseClaim(
            job=job,
            window=PreparedWindow.from_json(row["window_json"]),
            executor_id=executor_id,
            lineage_id=job.lineage_id,
            engine_epoch=job.engine_epoch,
            source_worker_id=row["source_worker_id"],
            lease_id=lease_id,
            lease_epoch=epoch,
            lease_seconds=self.lease_seconds,
            rowbundle_hash=row["rowbundle_hash"],
            rowbundle_size=int(row["rowbundle_size"]),
            compute_profile=_parse_json(row["compute_profile_json"]),
        )

    def _leased_row(self, db: sqlite3.Connection, executor_id: str, job_id: str,
                    lease_id: str, lease_epoch: int) -> sqlite3.Row:
        row = db.execute("SELECT * FROM jobs WHERE job_id=? AND executor_id=?", (job_id, executor_id)).fetchone()
        if (row is None
                or row["state"] != "leased" or row["lease_id"] != lease_id
                or int(row["lease_epoch"]) != lease_epoch or row["lease_expires_at"] is None
                or float(row["lease_expires_at"]) <= time.time()):
            raise StaleGradientLease("lease is expired, superseded, or belongs to another executor")
        return row

    def heartbeat(self, executor_id: str, job_id: str, lease_id: str, lease_epoch: int) -> float:
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = self._leased_row(db, executor_id, job_id, lease_id, lease_epoch)
                expires = time.time() + self.lease_seconds
                db.execute(
                    "UPDATE jobs SET lease_expires_at=? WHERE window_id=? AND attempt_id=? AND job_id=?",
                    (expires, row["window_id"], row["attempt_id"], job_id),
                )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return expires

    def rowbundle_path(self, executor_id: str, job_id: str, lease_id: str, lease_epoch: int) -> tuple[Path, str]:
        with self._db() as db:
            row = self._leased_row(db, executor_id, job_id, lease_id, lease_epoch)
        if row["rowbundle_pruned_at"] is not None:
            raise StaleGradientLease("rowbundle was pruned after this lease became terminal")
        path = self._stored_rowbundle_path(row["rowbundle_path"])
        if not path.is_file() or path.stat().st_size != int(row["rowbundle_size"]):
            raise GradientStoreError("immutable rowbundle is missing or has changed")
        return path, row["rowbundle_hash"]

    def expected_job(self, executor_id: str, job_id: str, lease_id: str, lease_epoch: int) -> tuple[GradientJob, PreparedWindow]:
        with self._db() as db:
            row = self._leased_row(db, executor_id, job_id, lease_id, lease_epoch)
            window_row = db.execute(
                "SELECT window_json FROM windows WHERE window_id=? AND attempt_id=?",
                (row["window_id"], row["attempt_id"]),
            ).fetchone()
        return GradientJob.from_json(row["job_json"]), PreparedWindow.from_json(window_row["window_json"])

    def begin_contribution(self, executor_id: str, job_id: str, lease_id: str,
                           lease_epoch: int, content_length: int | None) -> ContributionStage | dict[str, Any]:
        from qwen_rl.transport.tensor_codec import max_encoded_size
        if content_length is not None and content_length > max_encoded_size(self.max_payload_bytes):
            raise GradientPayloadTooLarge("gradient contribution exceeds configured max_payload_bytes")
        with self._lock, self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=? AND executor_id=?", (job_id, executor_id)).fetchone()
            if row is not None and row["state"] == "accepted":
                if row["lease_id"] == lease_id and int(row["lease_epoch"]) == lease_epoch:
                    return _parse_json(row["accepted_receipt_json"])
                raise StaleGradientLease("accepted job cannot be uploaded under a different lease")
            self._leased_row(db, executor_id, job_id, lease_id, lease_epoch)
            job = GradientJob.from_json(row["job_json"])
            expected = {
                "schema": _CONTRIBUTION_SCHEMA,
                "aggregation": _AGGREGATION,
                "window_id": job.window_id,
                "attempt_id": job.attempt_id,
                "lineage_id": job.lineage_id,
                "engine_epoch": job.engine_epoch,
                "job_id": job.job_id,
                "group_id": job.group_id,
                "owner_executor_id": job.owner_executor_id,
                "policy_version": job.policy_version,
                "n_window_tokens": job.n_window_tokens,
                "token_count": job.token_count,
                "layout_fingerprint": job.layout_fingerprint,
                "compute_profile_fingerprint": job.compute_profile_fingerprint,
            }
        stream = self._memory_payloads.stream()
        return ContributionStage(
            job_id=job_id, window_id=job.window_id, attempt_id=job.attempt_id,
            executor_id=executor_id, lease_id=lease_id, lease_epoch=lease_epoch,
            expected_metadata=expected, stream=stream,
        )

    def finish_contribution(self, stage: ContributionStage) -> dict[str, Any]:
        if stage.finished:
            raise GradientStoreError("upload stage has already been finalized")
        final_path = None
        try:
            stage.stream.flush()
            from qwen_rl.transport.tensor_codec import decode_tensor_payload
            try:
                raw = decode_tensor_payload(stage.stream.getvalue(), max_raw_bytes=self.max_payload_bytes, safetensors=True)
            except ValueError as exc:
                raise GradientStoreError("gradient tensor transport frame is invalid or exceeds max_payload_bytes") from exc
            raw_size = len(raw)
            metadata, present, wire_dtype = self._validate_gradient_payload(raw, stage.expected_metadata, stage)
            # SQLite stores only an opaque process-local reference, never tensor bytes.
            final_path = self._memory_payloads.put(raw)
            stage.stream.close()
            stage.finished = True
            return self._commit_contribution(stage, metadata, present, final_path, raw_size,
                                             wire_bytes=stage.size, storage="memory", wire_dtype=wire_dtype)
        except BaseException:
            if final_path is not None:
                self._memory_payloads.drop(final_path)
            self.abort_contribution(stage)
            raise

    def _commit_contribution(self, stage, metadata, present, final_path, raw_size, *, wire_bytes, storage,
                             wire_dtype="float32"):
        now = time.time()
        receipt = {
            "status": "accepted",
            **{key: metadata["contribution"][key]
               for key in stage.expected_metadata if key not in {"schema", "aggregation"}},
            "executor_id": stage.executor_id,
            "lease_epoch": stage.lease_epoch,
            "accepted_at": now,
            "contribution_bytes": raw_size,
            "stored_bytes": raw_size,
            "wire_bytes": wire_bytes,
            "gradient_wire_dtype": wire_dtype,
            "payload_storage": storage,
            "payload_durable": False,
            "present_parameter_count": len(present),
        }
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = self._leased_row(db, stage.executor_id, stage.job_id, stage.lease_id, stage.lease_epoch)
                db.execute(
                    "UPDATE jobs SET state='accepted',accepted_path=?,accepted_size=?,accepted_metadata_json=?,"
                    "accepted_receipt_json=?,accepted_at=?,lease_expires_at=NULL "
                    "WHERE window_id=? AND attempt_id=? AND job_id=?",
                    (str(final_path), raw_size,
                     json.dumps(metadata["contribution"], sort_keys=True, separators=(",", ":")),
                     json.dumps(receipt, sort_keys=True, separators=(",", ":")), now,
                     row["window_id"], row["attempt_id"], row["job_id"]),
                )
                db.execute("UPDATE windows SET updated_at=? WHERE window_id=? AND attempt_id=?",
                           (now, row["window_id"], row["attempt_id"]))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        with self._changed:
            self._changed.notify_all()
        return receipt

    def finish_shared_contribution(self, stage, payload):
        """Retain immutable mapped pages after the same metadata/lease commit as HTTP."""
        metadata = payload.metadata
        present = metadata.get("present_parameters", [])
        if any(metadata.get(key) != value for key, value in stage.expected_metadata.items()):
            raise GradientStoreError("shared gradient identity/version/token-layout differs from its job")
        reference = None
        try:
            reference = self._memory_payloads.put(payload)
            stage.stream.close()
            stage.finished = True
            return self._commit_contribution(stage, {"contribution": metadata}, present,
                                             reference, len(payload), wire_bytes=0,
                                             storage="shared_memory")
        except BaseException:
            if reference is not None:
                self._memory_payloads.drop(reference)
            self.abort_contribution(stage)
            raise

    def abort_contribution(self, stage: ContributionStage) -> None:
        if not stage.finished:
            try:
                stage.stream.close()
            finally:
                stage.finished = True

    def _read_contribution_payload(self, reference: str | Path, *, timing=None) -> bytes:
        from .staging_timing import measured
        try:
            with measured(timing, "gradient_memory_read"):
                raw = self._memory_payloads.get(str(reference))
        except KeyError:
            raise GradientStoreError("volatile gradient is unavailable; recompute from the durable checkpoint") from None
        if timing is not None:
            timing.bytes["gradient_memory_read"] += len(raw)
        return raw

    def _validate_gradient_payload(self, raw: bytes, expected: dict[str, Any],
                                   stage: ContributionStage) -> tuple[dict[str, Any], list[str]]:
        from safetensors.torch import load
        try:
            tensors = load(raw)
            header_size = int.from_bytes(raw[:8], 'little')
            header = json.loads(raw[8:8 + header_size])
            file_metadata = header.get("__metadata__", {})
            raw_metadata = file_metadata.get("gradient_contribution")
            if not raw_metadata:
                raise GradientStoreError("safetensors file lacks gradient_contribution metadata")
            metadata = _parse_json(raw_metadata)
            if (not isinstance(metadata, dict) or set(metadata) != {
                "contribution", "executor_id", "lease_id", "lease_epoch",
            }):
                raise GradientStoreError("safetensors lease metadata is malformed")
            contribution_metadata = metadata["contribution"]
            if not isinstance(contribution_metadata, dict):
                raise GradientStoreError("safetensors contribution metadata must be an object")
            keys = sorted(tensors)
            present = contribution_metadata.get("present_parameters")
            if (not isinstance(present, list) or present != sorted(set(present))
                    or any(not isinstance(name, str) or not name for name in present)):
                raise GradientStoreError("present_parameters must be a sorted unique name array")
            if keys != present:
                raise GradientStoreError("safetensors keys differ from present_parameters")
            if any(contribution_metadata.get(key) != value for key, value in expected.items()):
                raise GradientStoreError("gradient metadata identity/version/token-layout differs from its job")
            if (metadata["executor_id"] != stage.executor_id or metadata["lease_id"] != stage.lease_id
                    or metadata["lease_epoch"] != stage.lease_epoch):
                raise GradientStoreError("gradient safetensors lease identity differs from the upload lease")
            expected_job, expected_window = self.expected_job(
                stage.executor_id, expected["job_id"], stage.lease_id, stage.lease_epoch,
            )
            layout = {parameter.name: tuple(parameter.shape) for parameter in expected_window.parameter_layout}
            if not set(present).issubset(layout):
                raise GradientStoreError("gradient payload contains unknown parameter names")
            if set(present) == set() and expected_job.token_count:
                # A fully absent packet is legal for a sparse update; the central core
                # decides whether all-None gradients should preserve optimizer state.
                pass
            dtypes = set()
            for name in present:
                tensor = tensors[name]
                if tuple(tensor.shape) != layout[name]:
                    raise GradientStoreError(f"gradient shape differs from layout for {name}")
                dtypes.add(str(tensor.dtype))
            if not dtypes <= {"torch.float32", "torch.bfloat16"} or len(dtypes) > 1:
                raise GradientStoreError("gradient payload must have a uniform float32 or bfloat16 dtype")
            return metadata, present, next(iter(dtypes)).removeprefix("torch.") if dtypes else "empty"
        except GradientStoreError:
            raise
        except Exception as exc:
            raise GradientStoreError(f"invalid safetensors contribution: {exc}") from exc

    def accepted_count(self, window_id: str, attempt_id: str) -> tuple[int, int]:
        with self._db() as db:
            row = db.execute(
                "SELECT COUNT(*) AS total,SUM(CASE WHEN state='accepted' THEN 1 ELSE 0 END) AS accepted "
                "FROM jobs WHERE window_id=? AND attempt_id=?",
                (window_id, attempt_id),
            ).fetchone()
        return int(row["accepted"] or 0), int(row["total"] or 0)

    def staged_window_progress(
        self, windows: Sequence[tuple[str, str]],
    ) -> tuple[dict[str, Any], ...]:
        """Read staged window state and accepted-job counts in one metadata query."""
        if not isinstance(windows, (list, tuple)):
            raise TypeError("staged window progress requires a sequence of window/attempt pairs")
        keys = []
        for value in windows:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("staged window progress keys must be window/attempt pairs")
            keys.append((_safe_component(value[0], "window_id"),
                         _safe_component(value[1], "attempt_id")))
        if len(keys) != len(set(keys)):
            raise ValueError("staged window progress keys must be unique")
        if not keys:
            return ()

        requested = ",".join(f"({index},?,?)" for index in range(len(keys)))
        values = tuple(part for key in keys for part in key)
        query = f"""
            WITH requested(ordinal,window_id,attempt_id) AS (VALUES {requested})
            SELECT r.window_id,r.attempt_id,w.window_id AS found_window_id,
                   w.state,w.error,COALESCE(SUM(CASE WHEN j.state='accepted' THEN 1 ELSE 0 END),0) AS accepted,
                   COUNT(j.job_id) AS total
            FROM requested r
            LEFT JOIN windows w
              ON w.window_id=r.window_id AND w.attempt_id=r.attempt_id
            LEFT JOIN jobs j
              ON j.window_id=r.window_id AND j.attempt_id=r.attempt_id
            GROUP BY r.ordinal,r.window_id,r.attempt_id,w.window_id,w.state,w.error
            ORDER BY r.ordinal
        """
        with self._db() as db:
            rows = db.execute(query, values).fetchall()
        if len(rows) != len(keys):
            raise GradientStoreError("staged window progress query lost requested rows")
        result = []
        for row in rows:
            if row["found_window_id"] is None:
                raise KeyError((row["window_id"], row["attempt_id"]))
            result.append({
                "window_id": row["window_id"],
                "attempt_id": row["attempt_id"],
                "state": row["state"],
                "error": row["error"],
                "accepted": int(row["accepted"] or 0),
                "total": int(row["total"] or 0),
            })
        return tuple(result)

    def inspect_window_for_recovery(self, window_id: str, attempt_id: str) -> dict[str, Any]:
        """Read the durable store state needed to classify one Core-prepared window.

        Unlike ``accepted_count``, this distinguishes an absent scheduler window
        from a registered window with no accepted jobs. It validates the stored
        PreparedWindow and job identities before reporting that the store is
        incomplete or materialized; malformed or partial records raise instead
        of being treated as discardable work.
        """
        _safe_component(window_id, "window_id")
        _safe_component(attempt_id, "attempt_id")
        with self._db() as db:
            db.execute("BEGIN")
            try:
                window = db.execute(
                    "SELECT window_json,state FROM windows WHERE window_id=? AND attempt_id=?",
                    (window_id, attempt_id),
                ).fetchone()
                jobs = db.execute(
                    "SELECT job_id,job_json,state,accepted_path FROM jobs WHERE window_id=? AND attempt_id=?",
                    (window_id, attempt_id),
                ).fetchall()
                identity_row = db.execute(
                    "SELECT value FROM store_metadata WHERE key='central_identity'"
                ).fetchone()
                identity = _parse_json(identity_row["value"]) if identity_row else None
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

        if window is None:
            if jobs:
                raise GradientStoreError("gradient jobs exist without a registered window")
            return {
                "state": "missing", "store_state": None,
                "accepted_jobs": 0, "total_jobs": 0, "identity": identity,
            }

        try:
            prepared = PreparedWindow.from_json(window["window_json"])
        except Exception as exc:
            raise GradientStoreError("stored prepared window is invalid") from exc
        if (prepared.window_id != window_id or prepared.attempt_id != attempt_id
                or identity != {"lineage_id": prepared.lineage_id, "engine_epoch": prepared.engine_epoch}):
            raise GradientStoreError("stored prepared window identity differs from scheduler identity")
        expected_jobs = {job.job_id: job.to_json() for job in prepared.jobs}
        actual_jobs = {row["job_id"]: row for row in jobs}
        if (len(actual_jobs) != len(jobs) or set(actual_jobs) != set(expected_jobs)
                or any(actual_jobs[job_id]["job_json"] != expected_json
                       for job_id, expected_json in expected_jobs.items())):
            raise GradientStoreError("registered jobs differ from the stored prepared window")
        if any(row["state"] not in {"queued", "leased", "accepted", "cancelled"} for row in jobs):
            raise GradientStoreError("registered job has an unsupported durable state")

        store_state = window["state"]
        accepted = sum(row["state"] == "accepted" and (
            store_state not in {"pending", "applying"} or self._memory_payloads.contains(row["accepted_path"])
        ) for row in jobs)
        total = len(jobs)
        if store_state == "pending":
            state = "materialized" if accepted == total else "incomplete"
        elif store_state == "applying":
            # This marker is written before Core.apply_window. Core's own
            # pending_attempt fence is the authority for optimizer entry.
            state = "materialized" if accepted == total else "applying"
        elif store_state in {"applied", "cancelled", "failed"}:
            state = "terminal"
        else:
            raise GradientStoreError("registered window has an unsupported durable state")
        if total != len(prepared.jobs) or total == 0:
            raise GradientStoreError("registered job coverage differs from the prepared window")
        return {
            "state": state, "store_state": store_state,
            "accepted_jobs": accepted, "total_jobs": total, "identity": identity,
        }

    def begin_window_apply(self, window_id: str, attempt_id: str) -> bool:
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                window = db.execute(
                    "SELECT state FROM windows WHERE window_id=? AND attempt_id=?",
                    (window_id, attempt_id),
                ).fetchone()
                if window is None:
                    raise KeyError((window_id, attempt_id))
                if window["state"] == "cancelled":
                    db.execute("COMMIT")
                    return False
                total = db.execute(
                    "SELECT COUNT(*) AS n,SUM(CASE WHEN state='accepted' THEN 1 ELSE 0 END) AS accepted "
                    "FROM jobs WHERE window_id=? AND attempt_id=?", (window_id, attempt_id),
                ).fetchone()
                if int(total["n"] or 0) == 0 or int(total["accepted"] or 0) != int(total["n"]):
                    db.execute("COMMIT")
                    return False
                if window["state"] == "applied":
                    db.execute("COMMIT")
                    return False
                if window["state"] == "failed":
                    raise GradientStoreError("window is in a terminal failed state")
                db.execute(
                    "UPDATE windows SET state='applying',updated_at=? WHERE window_id=? AND attempt_id=?",
                    (time.time(), window_id, attempt_id),
                )
                db.execute("COMMIT")
                return True
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def mark_window_applied(self, window_id: str, attempt_id: str, receipt: Any) -> dict[str, Any]:
        receipt_value = _receipt_dict(receipt)
        raw = json.dumps(receipt_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        now = time.time()
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT state,receipt_json FROM windows WHERE window_id=? AND attempt_id=?",
                    (window_id, attempt_id),
                ).fetchone()
                if row is None:
                    raise KeyError((window_id, attempt_id))
                if row["state"] == "applied":
                    existing = _parse_json(row["receipt_json"])
                    if existing != receipt_value:
                        raise GradientWindowConflict("core returned a different receipt for an applied attempt")
                    db.execute("COMMIT")
                    receipt_value = existing
                else:
                    db.execute(
                        "UPDATE windows SET state='applied',receipt_json=?,error=NULL,updated_at=? "
                        "WHERE window_id=? AND attempt_id=?",
                        (raw, now, window_id, attempt_id),
                    )
                    db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        self.release_terminal_payloads(window_id, attempt_id)
        self.prune_terminal_rowbundles()
        with self._changed:
            self._changed.notify_all()
        return receipt_value

    def prune_terminal_rowbundles(self) -> int:
        """Remove applied/cancelled rowbundles after committing deletion intent.

        The rowbundle path, hash, and size stay in SQLite as audit metadata.
        Active references to a shared content-addressed path keep that file.
        """
        now = time.time()
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                rows = db.execute(
                    "SELECT j.rowbundle_path FROM jobs j JOIN windows w USING(window_id,attempt_id) "
                    "WHERE w.state IN ('applied','cancelled') AND j.state NOT IN ('queued','leased') "
                    "AND j.rowbundle_pruned_at IS NULL"
                ).fetchall()
                paths = {self._stored_rowbundle_path(row["rowbundle_path"]) for row in rows}
                db.execute(
                    "UPDATE jobs SET rowbundle_pruned_at=? WHERE rowbundle_pruned_at IS NULL "
                    "AND state NOT IN ('queued','leased') "
                    "AND EXISTS (SELECT 1 FROM windows w WHERE w.window_id=jobs.window_id "
                    "AND w.attempt_id=jobs.attempt_id AND w.state IN ('applied','cancelled'))",
                    (now,),
                )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

        # Recheck under a write transaction so another store cannot register a
        # live reference between the check and unlink. enqueue_window validates
        # file presence inside its own transaction to close the opposite race.
        unlinked = False
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                for path in paths:
                    referenced = db.execute(
                        "SELECT 1 FROM jobs WHERE rowbundle_path=? AND rowbundle_pruned_at IS NULL LIMIT 1",
                        (str(path),),
                    ).fetchone()
                    if referenced is None:
                        path.unlink(missing_ok=True)
                        unlinked = True
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        if unlinked:
            self._fsync_directory(self.rowbundle_dir)
        return len(rows)

    def release_materialized_staged_payloads(self, window_id: str, attempt_id: str) -> int:
        """Drop recomputable staged tensors without fencing receipt readers.

        A restart requeues these missing, pre-optimizer partitions. The final
        optimizer window's payloads cannot enter this path.
        """
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT state,window_json FROM windows WHERE window_id=? AND attempt_id=?",
                                 (window_id, attempt_id)).fetchone()
                if row is None:
                    raise KeyError((window_id, attempt_id))
                window = PreparedWindow.from_json(row["window_json"])
                if (row["state"] != "pending"
                        or (window.update_context or {}).get("gradient_scope") != STAGED_GROUP_SCOPE):
                    raise GradientWindowConflict("only pending pre-optimizer staged payloads may be materialized")
                jobs = db.execute("SELECT state,accepted_path FROM jobs WHERE window_id=? AND attempt_id=?",
                                  (window_id, attempt_id)).fetchall()
                if not jobs or any(job["state"] != "accepted" for job in jobs):
                    raise GradientWindowConflict("staged group must be complete before materialization")
                paths = [self._staged_contribution_path(job["accepted_path"])
                         for job in jobs if job["accepted_path"] is not None]
                db.execute("UPDATE jobs SET accepted_path=NULL WHERE window_id=? AND attempt_id=?",
                           (window_id, attempt_id))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        self._unlink_released_payloads(paths)
        return len(paths)

    def release_terminal_payloads(self, window_id: str, attempt_id: str) -> int:
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT state FROM windows WHERE window_id=? AND attempt_id=?",
                                 (window_id, attempt_id)).fetchone()
                if row is None:
                    raise KeyError((window_id, attempt_id))
                if row["state"] not in {"applied", "cancelled"}:
                    raise GradientWindowConflict("payload release requires a terminal window")
                paths = [self._staged_contribution_path(row["accepted_path"]) for row in db.execute(
                    "SELECT accepted_path FROM jobs WHERE window_id=? AND attempt_id=? AND accepted_path IS NOT NULL",
                    (window_id, attempt_id))]
                db.execute("UPDATE jobs SET accepted_path=NULL WHERE window_id=? AND attempt_id=?",
                           (window_id, attempt_id))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        self._unlink_released_payloads(paths)
        return len(paths)

    def mark_window_error(self, window_id: str, attempt_id: str, error: str) -> None:
        with self._lock, self._db() as db:
            db.execute(
                "UPDATE windows SET error=?,updated_at=? WHERE window_id=? AND attempt_id=? AND state!='applied'",
                (str(error)[:1000], time.time(), window_id, attempt_id),
            )
        with self._changed:
            self._changed.notify_all()

    def cancel_window(self, window_id: str, attempt_id: str, reason: str) -> bool:
        """Fence an incomplete prepared window while retaining accepted artifacts/receipts."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cancellation reason must be nonempty")
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                window = db.execute(
                    "SELECT state FROM windows WHERE window_id=? AND attempt_id=?",
                    (window_id, attempt_id),
                ).fetchone()
                if window is None:
                    raise KeyError((window_id, attempt_id))
                total = db.execute(
                    "SELECT COUNT(*) AS total,SUM(CASE WHEN state='accepted' THEN 1 ELSE 0 END) AS accepted "
                    "FROM jobs WHERE window_id=? AND attempt_id=?", (window_id, attempt_id),
                ).fetchone()
                job_count, accepted_count = int(total["total"] or 0), int(total["accepted"] or 0)
                if window["state"] == "applied":
                    raise GradientWindowConflict("an applied window cannot be cancelled")
                if job_count == 0:
                    raise GradientStoreError("cannot cancel a window without registered jobs")
                if accepted_count == job_count:
                    db.execute("COMMIT")
                    return False
                if window["state"] == "cancelled":
                    db.execute("COMMIT")
                    return True
                now = time.time()
                db.execute(
                    "UPDATE jobs SET state='cancelled',lease_epoch=lease_epoch+1,executor_id=NULL,lease_id=NULL,"
                    "lease_expires_at=NULL WHERE window_id=? AND attempt_id=? AND state!='accepted'",
                    (window_id, attempt_id),
                )
                db.execute(
                    "UPDATE windows SET state='cancelled',error=?,updated_at=? WHERE window_id=? AND attempt_id=?",
                    (reason.strip()[:1000], now, window_id, attempt_id),
                )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        with self._changed:
            self._changed.notify_all()
        self.release_terminal_payloads(window_id, attempt_id)
        self.prune_terminal_rowbundles()
        return True

    def _staged_contribution_path(self, value: str) -> Path:
        if value.startswith("ram:"):
            return Path(value)
        path = Path(value).expanduser()
        if path.is_symlink():
            raise GradientStoreError("staged contribution path may not be a symlink")
        root = self.contribution_dir.resolve()
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise GradientStoreError(
                "staged contribution path escapes its artifact directory"
            )
        if path.exists() and not path.is_file():
            raise GradientStoreError("staged contribution path is not a regular file")
        return path

    def _unlink_released_payloads(self, paths: list[Path]) -> None:
        if not paths:
            return
        for path in paths:
            if str(path).startswith("ram:"):
                self._memory_payloads.drop(str(path))
            else:
                path.unlink(missing_ok=True)
        self._fsync_directory(self.contribution_dir)

    def release_staged_window(
        self, window_id: str, attempt_id: str, *, reason: str,
    ) -> bool:
        """Retire one pre-optimizer local-mean staging window and its tensor files.

        Staged group means are inputs to a later central window; they never have
        optimizer authority themselves. The SQLite deletion intent is committed
        before payload unlink, so startup reconciliation can collect an orphan if
        the process stops between those operations. Accepted receipt metadata is
        retained for audit even though the job is fenced as cancelled.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("staged release reason must be nonempty")
        obsolete: list[Path] = []
        changed = False
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                window = db.execute(
                    "SELECT state,window_json,error FROM windows "
                    "WHERE window_id=? AND attempt_id=?",
                    (window_id, attempt_id),
                ).fetchone()
                if window is None:
                    raise KeyError((window_id, attempt_id))
                prepared = PreparedWindow.from_json(window["window_json"])
                scope = (prepared.update_context or {}).get("gradient_scope")
                if scope != STAGED_GROUP_SCOPE:
                    raise GradientWindowConflict(
                        "only local group-mean staging windows can be released"
                    )
                state = window["state"]
                if state in {"applied", "applying"}:
                    raise GradientWindowConflict(
                        "an optimizer-authoritative window cannot be released "
                        "as staging"
                    )
                if state not in {"pending", "cancelled", "failed"}:
                    raise GradientWindowConflict(
                        "staged window has an unsupported state"
                    )
                jobs = db.execute(
                    "SELECT state,accepted_path FROM jobs "
                    "WHERE window_id=? AND attempt_id=?",
                    (window_id, attempt_id),
                ).fetchall()
                if not jobs:
                    raise GradientStoreError(
                        "cannot release a staged window without registered jobs"
                    )
                if state == "cancelled" and all(
                    row["state"] == "cancelled"
                    and row["accepted_path"] is None
                    for row in jobs
                ):
                    db.execute("COMMIT")
                    return False
                obsolete = list(dict.fromkeys(
                    self._staged_contribution_path(row["accepted_path"])
                    for row in jobs if row["accepted_path"] is not None
                ))
                now = time.time()
                db.execute(
                    "UPDATE jobs SET state='cancelled',executor_id=NULL,lease_id=NULL,"
                    "lease_epoch=lease_epoch+1,lease_expires_at=NULL,"
                    "accepted_path=NULL "
                    "WHERE window_id=? AND attempt_id=?",
                    (window_id, attempt_id),
                )
                prior_error = window["error"]
                detail = f"{prior_error}; {reason}" if prior_error else reason
                db.execute(
                    "UPDATE windows SET state='cancelled',error=?,updated_at=? "
                    "WHERE window_id=? AND attempt_id=?",
                    (detail[:1000], now, window_id, attempt_id),
                )
                db.execute("COMMIT")
                changed = True
            except BaseException:
                db.execute("ROLLBACK")
                raise
        self._unlink_released_payloads(obsolete)
        if changed:
            self.prune_terminal_rowbundles()
            with self._changed:
                self._changed.notify_all()
        return changed

    def release_staged_windows(self, *, reason: str) -> int:
        """Retire all non-applied local staging windows after a clean core restore."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("staged release reason must be nonempty")
        with self._db() as db:
            rows = db.execute(
                "SELECT window_id,attempt_id,state,window_json FROM windows "
                "ORDER BY created_at,window_id,attempt_id"
            ).fetchall()
        released = 0
        for row in rows:
            window = PreparedWindow.from_json(row["window_json"])
            scope = (window.update_context or {}).get("gradient_scope")
            if scope != STAGED_GROUP_SCOPE:
                continue
            if row["state"] in {"applied", "applying"}:
                raise GradientWindowConflict(
                    "optimizer-authoritative state cannot be bulk-released "
                    "as staging"
                )
            if row["state"] not in {"pending", "cancelled", "failed"}:
                raise GradientWindowConflict("staged window has an unsupported state")
            released += int(self.release_staged_window(
                row["window_id"], row["attempt_id"], reason=reason,
            ))
        return released

    def load_window(self, window_id: str, attempt_id: str) -> PreparedWindow:
        with self._db() as db:
            row = db.execute("SELECT window_json FROM windows WHERE window_id=? AND attempt_id=?",
                             (window_id, attempt_id)).fetchone()
        if row is None:
            raise KeyError((window_id, attempt_id))
        return PreparedWindow.from_json(row["window_json"])

    def load_contribution_diagnostics(self, window_id: str, attempt_id: str) -> list[Any]:
        from types import SimpleNamespace
        with self._db() as db:
            rows = db.execute(
                "SELECT accepted_metadata_json FROM jobs WHERE window_id=? AND attempt_id=? "
                "AND state='accepted' ORDER BY work_order", (window_id, attempt_id),
            ).fetchall()
        return [SimpleNamespace(diagnostics=_parse_json(row[0]).get("diagnostics", {}))
                for row in rows if row[0] is not None]

    def load_contributions(self, window_id: str, attempt_id: str) -> list[Any]:
        return list(self.iter_contributions(window_id, attempt_id))

    def iter_contributions(self, window_id: str, attempt_id: str, *, timing=None):
        """Read accepted immutable partitions in logical order, one payload at a time."""
        from qwen_rl.contracts.gradients import GradientContribution
        from safetensors.torch import load
        from .staging_timing import measured

        with self._db() as db:
            rows = db.execute(
                "SELECT job_json,accepted_path,accepted_metadata_json,state,work_order FROM jobs "
                "WHERE window_id=? AND attempt_id=? ORDER BY work_order",
                (window_id, attempt_id),
            ).fetchall()
        if not rows or any(row["state"] != "accepted" for row in rows):
            raise GradientStoreError("window contributions are incomplete")
        window = self.load_window(window_id, attempt_id)
        parameter_names = [parameter.name for parameter in window.parameter_layout]
        for row in rows:
            if not row["accepted_path"]:
                raise GradientStoreError("accepted gradient payload was intentionally pruned after window apply")
            metadata = row["accepted_metadata_json"]
            raw = self._read_contribution_payload(row["accepted_path"], timing=timing)
            with measured(timing, "gradient_decode_convert"):
                tensors = dict(raw.gradients) if hasattr(raw, "gradients") else load(raw)
                import torch
                tensors = {name: None if value is None else value.to(dtype=torch.float32)
                           for name, value in tensors.items()}
                contribution = GradientContribution.from_metadata_json(
                    metadata, gradients=tensors, parameter_names=parameter_names,
                )
            del raw, tensors
            yield contribution
            del contribution

    def receipt(self, executor_id: str, job_id: str, lease_id: str, lease_epoch: int) -> dict[str, Any] | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=? AND executor_id=?", (job_id, executor_id)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if row["lease_id"] != lease_id or int(row["lease_epoch"]) != lease_epoch:
            raise StaleGradientLease("receipt request has a stale lease fence")
        if row["state"] != "accepted":
            return None
        return _parse_json(row["accepted_receipt_json"])

    def window_status(self, window_id: str, attempt_id: str) -> dict[str, Any]:
        with self._db() as db:
            window = db.execute("SELECT * FROM windows WHERE window_id=? AND attempt_id=?",
                                (window_id, attempt_id)).fetchone()
            if window is None:
                raise KeyError((window_id, attempt_id))
            jobs = db.execute(
                "SELECT state,COUNT(*) AS count FROM jobs WHERE window_id=? AND attempt_id=? GROUP BY state",
                (window_id, attempt_id),
            ).fetchall()
            payload_counts = self._payload_retained_counts(db, window_id, attempt_id)
        counts = {row["state"]: int(row["count"]) for row in jobs}
        identity = self._identity()
        return {
            "window_id": window_id,
            "attempt_id": attempt_id,
            "lineage_id": identity["lineage_id"],
            "engine_epoch": identity["engine_epoch"],
            "state": window["state"],
            "jobs": counts,
            "payload_retained_counts": payload_counts,
            "receipt": _parse_json(window["receipt_json"]) if window["receipt_json"] else None,
            "error": window["error"],
        }

    def status(self) -> dict[str, Any]:
        with self._db() as db:
            windows = db.execute(
                "SELECT window_id,attempt_id,state,receipt_json,error,created_at,updated_at,compute_profile_json "
                "FROM windows "
                "ORDER BY created_at,window_id,attempt_id"
            ).fetchall()
            counts = db.execute("SELECT state,COUNT(*) AS count FROM jobs GROUP BY state").fetchall()
            payload_counts = self._payload_retained_counts(db)
            identity_row = db.execute(
                "SELECT value FROM store_metadata WHERE key='central_identity'"
            ).fetchone()
            host_admission = None
            if self.max_inflight_input_tokens is not None:
                from .work_stealing import partition_peak_input_tokens
                live = db.execute(
                    "SELECT j.job_id,w.window_json FROM jobs j JOIN windows w "
                    "USING(window_id,attempt_id) WHERE j.state='leased'"
                ).fetchall()
                host_admission = {
                    'max_inflight_input_tokens': self.max_inflight_input_tokens,
                    'leased_input_tokens': sum(partition_peak_input_tokens(item) for item in live),
                }
        sessions = self._compute_session_rows()
        active_sessions = {
            row["executor_id"]: {key: row[key] for key in ("executor_id", "session_id", "round_id")}
            for row in sessions if row["active"] == 1
        }
        return {
            "identity": _parse_json(identity_row["value"]) if identity_row else None,
            "memory_payloads": self._memory_payloads.status(),
            **({'host_compute_admission': host_admission} if host_admission is not None else {}),
            "compute_sessions": list(active_sessions.values()),
            "parked_compute_sessions": [active_sessions[row["executor_id"]] for row in sessions
                                        if row["active"] == 1 and row["parked"] == 1],
            "episode_queue": self.episode_queue.snapshot() if self.episode_queue is not None else None,
            "work_stealing_pools": dict(self.work_stealing_pools),
            "jobs": {row["state"]: int(row["count"]) for row in counts},
            "payload_retained_counts": payload_counts,
            "windows": [
                {"window_id": row["window_id"], "attempt_id": row["attempt_id"], "state": row["state"],
                 "receipt": _parse_json(row["receipt_json"]) if row["receipt_json"] else None,
                 "error": row["error"],
                 "compute_profile": _parse_json(row["compute_profile_json"])
                 if row["compute_profile_json"] else None,
                 "created_at": row["created_at"], "updated_at": row["updated_at"]}
                for row in windows
            ],
        }

    @staticmethod
    def _payload_retained_counts(db: sqlite3.Connection, window_id: str | None = None,
                                 attempt_id: str | None = None) -> dict[str, int]:
        query = (
            "SELECT COUNT(*) AS accepted_total, "
            "SUM(CASE WHEN j.accepted_path IS NOT NULL THEN 1 ELSE 0 END) AS retained, "
            "SUM(CASE WHEN j.accepted_path IS NULL AND w.state IN ('applied','cancelled') THEN 1 ELSE 0 END) AS pruned, "
            "SUM(CASE WHEN j.accepted_path IS NULL AND w.state NOT IN ('applied','cancelled') THEN 1 ELSE 0 END) AS missing "
            "FROM jobs j JOIN windows w USING(window_id,attempt_id) WHERE j.state='accepted'"
        )
        values: tuple[Any, ...] = ()
        if window_id is not None and attempt_id is not None:
            query += " AND j.window_id=? AND j.attempt_id=?"
            values = (window_id, attempt_id)
        row = db.execute(query, values).fetchone()
        return {key: int(row[key] or 0) for key in ("accepted_total", "retained", "pruned", "missing")}

    def _identity(self) -> dict[str, str]:
        with self._db() as db:
            row = db.execute("SELECT value FROM store_metadata WHERE key='central_identity'").fetchone()
        if row is None:
            raise GradientStoreError("scheduler database has no central identity yet")
        value = _parse_json(row["value"])
        return {"lineage_id": value["lineage_id"], "engine_epoch": value["engine_epoch"]}

    def bind_identity(self, lineage_id: str, engine_epoch: str) -> None:
        identity = {
            "lineage_id": _safe_component(lineage_id, "lineage_id"),
            "engine_epoch": _safe_component(engine_epoch, "engine_epoch"),
        }
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT value FROM store_metadata WHERE key='central_identity'"
                ).fetchone()
                if row is not None and _parse_json(row["value"]) != identity:
                    raise GradientWindowConflict(
                        "scheduler database is bound to another central optimizer lineage/engine epoch"
                    )
                if row is None:
                    db.execute(
                        "INSERT INTO store_metadata(key,value) VALUES('central_identity',?)",
                        (json.dumps(identity, sort_keys=True, separators=(",", ":")),),
                    )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def bind_executor_worker_ids(self, executor_worker_ids: Mapping[str, str]) -> dict[str, str]:
        if not isinstance(executor_worker_ids, Mapping) or not executor_worker_ids:
            raise ValueError("executor_worker_ids must explicitly map executors to Workers")
        normalized = {
            _safe_component(executor_id, "executor_id"): _safe_component(worker_id, "worker_id")
            for executor_id, worker_id in executor_worker_ids.items()
        }
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("each gradient executor must own a distinct Worker identity")
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT value FROM store_metadata WHERE key='executor_worker_ids'"
                ).fetchone()
                stored_map = _parse_json(row["value"]) if row is not None else None
                active_window = db.execute(
                    "SELECT 1 FROM windows WHERE state IN ('pending','applying') LIMIT 1"
                ).fetchone()
                active_job = db.execute(
                    "SELECT 1 FROM jobs WHERE state IN ('queued','leased') LIMIT 1"
                ).fetchone()
                if stored_map is None:
                    # Before the first map is bound, all existing rows still
                    # need to prove their immutable owner/source pair.
                    existing = db.execute(
                        "SELECT DISTINCT owner_executor_id,source_worker_id FROM jobs"
                    ).fetchall()
                    if any(normalized.get(item["owner_executor_id"]) != item["source_worker_id"]
                           for item in existing):
                        raise GradientWindowConflict(
                            "registered job ownership differs from configured executor/Worker map"
                        )
                elif stored_map != normalized and (active_window is not None or active_job is not None):
                    raise GradientWindowConflict(
                        "executor/Worker map can change only after all gradient windows are terminal"
                    )
                elif stored_map == normalized:
                    active_rows = db.execute(
                        "SELECT DISTINCT j.owner_executor_id,j.source_worker_id FROM jobs j "
                        "JOIN windows w USING(window_id,attempt_id) "
                        "WHERE w.state IN ('pending','applying') OR j.state IN ('queued','leased')"
                    ).fetchall()
                    if any(normalized.get(item["owner_executor_id"]) != item["source_worker_id"]
                           for item in active_rows):
                        raise GradientWindowConflict(
                            "active job ownership differs from configured executor/Worker map"
                        )
                if row is None:
                    db.execute(
                        "INSERT INTO store_metadata(key,value) VALUES('executor_worker_ids',?)",
                        (json.dumps(normalized, sort_keys=True, separators=(",", ":")),),
                    )
                elif stored_map != normalized:
                    db.execute(
                        "UPDATE store_metadata SET value=? WHERE key='executor_worker_ids'",
                        (json.dumps(normalized, sort_keys=True, separators=(",", ":")),),
                    )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return normalized

    def release_terminal_window_payloads(self) -> int:
        with self._db() as db:
            windows = db.execute(
                "SELECT DISTINCT w.window_id,w.attempt_id FROM windows w JOIN jobs j "
                "USING(window_id,attempt_id) WHERE w.state IN ('applied','cancelled') "
                "AND j.accepted_path IS NOT NULL"
            ).fetchall()
        return sum(self.release_terminal_payloads(row["window_id"], row["attempt_id"])
                   for row in windows)

    def active_policy_versions(self) -> set[int]:
        with self._db() as db:
            rows = db.execute(
                "SELECT window_json FROM windows WHERE state IN ('pending','applying')"
            ).fetchall()
        return {PreparedWindow.from_json(row["window_json"]).policy_version for row in rows}

    def executor_worker_ids(self) -> dict[str, str]:
        with self._db() as db:
            row = db.execute(
                "SELECT value FROM store_metadata WHERE key='executor_worker_ids'"
            ).fetchone()
        return _parse_json(row["value"]) if row is not None else {}

    def wait_for_window(self, window_id: str, attempt_id: str, timeout: float | None = None,
                        poll_interval: float = 0.1) -> dict[str, Any]:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._changed:
            while True:
                status = self.window_status(window_id, attempt_id)
                if status["error"]:
                    raise GradientStoreError(f"gradient window failed: {status['error']}")
                if status["state"] == "applied":
                    return status["receipt"]
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"gradient window {window_id}/{attempt_id} did not complete before timeout"
                    )
                self._changed.wait(timeout=min(poll_interval, remaining)
                                   if remaining is not None else poll_interval)

    def reconcile(self) -> None:
        """Drop interrupted staging/orphans and fail closed on missing accepted files."""
        with self._lock, self._db() as db:
            self._sampling_rounds.clear()
            references = {
                "rowbundles": {str(Path(row[0]).resolve()) for row in db.execute(
                    "SELECT rowbundle_path FROM jobs WHERE rowbundle_pruned_at IS NULL")},
                "contributions": {str(Path(row[0]).resolve()) for row in db.execute(
                    "SELECT accepted_path FROM jobs WHERE accepted_path IS NOT NULL")},
            }
            accepted = db.execute(
                "SELECT j.window_id,j.attempt_id,j.job_id,j.accepted_path,w.state AS window_state "
                "FROM jobs j JOIN windows w USING(window_id,attempt_id) WHERE j.state='accepted'"
            ).fetchall()
            rowbundle_rows = db.execute(
                "SELECT window_id,attempt_id,job_id,rowbundle_path,rowbundle_size FROM jobs "
                "WHERE rowbundle_pruned_at IS NULL"
            ).fetchall()
        for directory, key in ((self.rowbundle_dir, "rowbundles"), (self.contribution_dir, "contributions")):
            for path in directory.iterdir():
                if path.name.endswith(".part") or str(path.resolve()) not in references[key]:
                    if path.is_file():
                        path.unlink(missing_ok=True)
        missing_by_window: dict[tuple[str, str], list[str]] = {}
        for row in accepted:
            if not row["accepted_path"] and row["window_state"] in {"applied", "cancelled"}:
                # accepted_path=NULL is the durable intent for an intentionally
                # pruned applied payload; metadata/receipt remain authoritative.
                continue
            if not self._memory_payloads.contains(row["accepted_path"]):
                missing_by_window.setdefault((row["window_id"], row["attempt_id"]), []).append(row["job_id"])
        missing_rowbundles: dict[tuple[str, str], list[str]] = {}
        for row in rowbundle_rows:
            path = Path(row["rowbundle_path"])
            if not path.is_file() or path.stat().st_size != int(row["rowbundle_size"]):
                missing_rowbundles.setdefault((row["window_id"], row["attempt_id"]), []).append(row["job_id"])
        now = time.time()
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                for (window_id, attempt_id), job_ids in missing_by_window.items():
                    window_state = db.execute(
                        "SELECT state FROM windows WHERE window_id=? AND attempt_id=?", (window_id, attempt_id)
                    ).fetchone()
                    if window_state and window_state["state"] in {"applied", "cancelled"}:
                        # Full checkpoint/receipt is authoritative; tensors are disposable.
                        db.execute("UPDATE jobs SET accepted_path=NULL WHERE window_id=? AND attempt_id=?",
                                   (window_id, attempt_id))
                        continue
                    if window_state and window_state["state"] == "applying":
                        # Never rewind an optimizer-entry marker just because RAM was lost.
                        db.execute("UPDATE windows SET error=?,updated_at=? WHERE window_id=? AND attempt_id=?",
                                   ("volatile gradients lost during apply; managed checkpoint recovery required", now,
                                    window_id, attempt_id))
                        continue
                    for job_id in job_ids:
                        db.execute(
                            "UPDATE jobs SET state='queued',executor_id=NULL,lease_id=NULL,lease_epoch=lease_epoch+1,lease_expires_at=NULL,"
                            "accepted_path=NULL,accepted_size=NULL,accepted_metadata_json=NULL,"
                            "accepted_receipt_json=NULL,accepted_at=NULL WHERE window_id=? AND attempt_id=? AND job_id=?",
                            (window_id, attempt_id, job_id),
                        )
                    db.execute("UPDATE windows SET state='pending',receipt_json=NULL,error=NULL,updated_at=? "
                               "WHERE window_id=? AND attempt_id=?", (now, window_id, attempt_id))
                for (window_id, attempt_id), job_ids in missing_rowbundles.items():
                    window_state = db.execute(
                        "SELECT state,error FROM windows WHERE window_id=? AND attempt_id=?",
                        (window_id, attempt_id),
                    ).fetchone()
                    if window_state is None:
                        continue
                    detail = "immutable rowbundle is missing or changed: " + ",".join(sorted(job_ids))
                    if window_state["error"]:
                        detail = f"{window_state['error']}; {detail}"
                    if window_state["state"] not in {"applied", "cancelled"}:
                        db.execute(
                            "UPDATE jobs SET state='cancelled',lease_epoch=lease_epoch+1,executor_id=NULL,"
                            "lease_id=NULL,lease_expires_at=NULL WHERE window_id=? AND attempt_id=? AND state!='accepted'",
                            (window_id, attempt_id),
                        )
                        db.execute(
                            "UPDATE windows SET state='failed',error=?,updated_at=? WHERE window_id=? AND attempt_id=?",
                            (detail[:1000], now, window_id, attempt_id),
                        )
                    else:
                        db.execute(
                            "UPDATE windows SET error=?,updated_at=? WHERE window_id=? AND attempt_id=?",
                            (detail[:1000], now, window_id, attempt_id),
                        )
                # A committed core receipt may precede the HTTP receipt transaction.
                # The app retries `apply_window` on startup; its core contract is idempotent.
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        self._fsync_directory(self.rowbundle_dir)
        self._fsync_directory(self.contribution_dir)

        with self._lock, self._db() as db:
            retained = {row[0] for row in db.execute(
                "SELECT accepted_path FROM jobs WHERE accepted_path IS NOT NULL")}
            for path in self.contribution_dir.iterdir():
                if path.is_file() and str(path) not in retained:
                    path.unlink(missing_ok=True)


def _receipt_dict(receipt: Any) -> dict[str, Any]:
    if isinstance(receipt, dict):
        value = receipt
    elif callable(getattr(receipt, "to_json", None)):
        value = _parse_json(receipt.to_json())
    elif hasattr(receipt, "__dataclass_fields__"):
        from dataclasses import asdict
        value = asdict(receipt)
    elif callable(getattr(receipt, "model_dump", None)):
        value = receipt.model_dump(mode="json")
    else:
        raise TypeError("window receipt must be a JSON object or provide to_json/model_dump")
    if not isinstance(value, dict):
        raise TypeError("window receipt must serialize to a JSON object")
    _canonical_json(value)
    return value
