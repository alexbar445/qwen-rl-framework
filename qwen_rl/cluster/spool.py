"""Durable idempotency storage for Worker generation requests.

Generation is an external side effect. Once a request may have reached a
Worker, an absent result is ambiguous and must never be turned into a second
generation attempt. This spool records that boundary before the HTTP call and
stores the exact response bytes before the Gateway returns them.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterator


class GenerationIdentityConflict(Exception):
    """A request ID was reused with a different body."""


class GenerationIndeterminate(Exception):
    """A previous call may have reached a Worker, but its reply is unknown."""


class SpoolUnavailable(Exception):
    """The Gateway could not durably record a generation boundary or result."""


@dataclass(frozen=True)
class GenerationResult:
    status_code: int
    body: bytes
    headers: tuple[tuple[str, str], ...]


@dataclass
class _Flight:
    lock: asyncio.Lock
    references: int = 0


class GenerationSpool:
    """SQLite WAL spool keyed by ``(node_id, request_id)``.

    SQLite's ``synchronous=FULL`` in WAL mode syncs each committed transaction
    before it returns. A ``pending`` row is committed before HTTP dispatch;
    after restart it becomes ``indeterminate`` and can only be inspected, never
    dispatched again.
    """

    def __init__(self, path: str | Path):
        requested_path = Path(path).expanduser()
        requested_parent = requested_path.parent
        if any(ancestor.is_symlink() for ancestor in (requested_parent, *requested_parent.parents)):
            raise SpoolUnavailable("generation spool directory must not be a symlink")
        state_dir = requested_parent.resolve()
        if state_dir == Path("/") or state_dir == Path("/etc") or Path("/etc") in state_dir.parents:
            raise SpoolUnavailable("generation spool directory cannot be / or inside /etc")
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not state_dir.is_dir():
            raise SpoolUnavailable("generation spool path must be a directory")
        state_dir.chmod(0o700)
        self.path = state_dir / requested_path.name
        if self.path.is_symlink():
            raise SpoolUnavailable("generation spool database must not be a symlink")
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if not self.path.is_file():
                raise SpoolUnavailable("generation spool database must be a regular file")
            self.path.chmod(0o600)
        else:
            os.close(descriptor)
        self._flights: dict[tuple[str, str], _Flight] = {}
        self._flights_guard = asyncio.Lock()
        try:
            self._initialize()
        except (OSError, sqlite3.Error) as exc:
            raise SpoolUnavailable("could not initialize generation result spool") from exc

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS generation_results (
                    node_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    body_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'indeterminate', 'response')),
                    status_code INTEGER,
                    response_body BLOB,
                    response_headers TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (node_id, request_id)
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS generation_results_state_idx "
                "ON generation_results(state)"
            )
        finally:
            conn.close()
        self.path.chmod(0o600)
        database_fd = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(database_fd)
        finally:
            os.close(database_fd)
        # Persist the directory entry when the database is first created.
        self._fsync_directory(self.path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _recover_pending_sync(self) -> int:
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE generation_results SET state='indeterminate', "
                "error='gateway restarted with no durable response', updated_at=? "
                "WHERE state='pending'",
                (now,),
            )
            count = cursor.rowcount
            conn.commit()
            return count

    async def recover_pending(self) -> int:
        """Mark every request interrupted by a prior process as unknown."""
        try:
            return await asyncio.to_thread(self._recover_pending_sync)
        except (OSError, sqlite3.Error) as exc:
            raise SpoolUnavailable("could not recover pending generation requests") from exc

    @asynccontextmanager
    async def _singleflight(self, key: tuple[str, str]):
        async with self._flights_guard:
            flight = self._flights.get(key)
            if flight is None:
                flight = _Flight(lock=asyncio.Lock())
                self._flights[key] = flight
            flight.references += 1
        acquired = False
        try:
            await flight.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                flight.lock.release()
            async with self._flights_guard:
                flight.references -= 1
                if flight.references == 0:
                    self._flights.pop(key, None)

    def _reserve_sync(self, node_id: str, request_id: str, body_sha256: str) -> GenerationResult | None:
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT body_sha256, state, status_code, response_body, response_headers "
                "FROM generation_results WHERE node_id=? AND request_id=?",
                (node_id, request_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO generation_results "
                    "(node_id, request_id, body_sha256, state, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'pending', ?, ?)",
                    (node_id, request_id, body_sha256, now, now),
                )
                conn.commit()
                return None

            if row["body_sha256"] != body_sha256:
                conn.rollback()
                raise GenerationIdentityConflict(
                    "request_id is already bound to a different request body"
                )
            if row["state"] == "response":
                headers = tuple(tuple(pair) for pair in json.loads(row["response_headers"] or "[]"))
                result = GenerationResult(
                    status_code=int(row["status_code"]),
                    body=bytes(row["response_body"] or b""),
                    headers=headers,
                )
                conn.commit()
                return result
            if row["state"] == "pending":
                conn.execute(
                    "UPDATE generation_results SET state='indeterminate', "
                    "error='request is pending outside this singleflight', updated_at=? "
                    "WHERE node_id=? AND request_id=? AND state='pending'",
                    (now, node_id, request_id),
                )
                conn.commit()
                raise GenerationIndeterminate(
                    "generation result is unknown; the Worker will not be called again"
                )
            conn.commit()
            raise GenerationIndeterminate(
                "generation result is unknown; the Worker will not be called again"
            )

    def _record_response_sync(
        self,
        node_id: str,
        request_id: str,
        body_sha256: str,
        result: GenerationResult,
    ) -> None:
        now = time.time()
        headers = json.dumps(list(result.headers), separators=(",", ":"), ensure_ascii=True)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE generation_results SET state='response', status_code=?, "
                "response_body=?, response_headers=?, error=NULL, updated_at=? "
                "WHERE node_id=? AND request_id=? AND body_sha256=? "
                "AND state IN ('pending', 'indeterminate')",
                (result.status_code, sqlite3.Binary(result.body), headers, now,
                 node_id, request_id, body_sha256),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise sqlite3.DatabaseError("generation spool row disappeared before response commit")
            conn.commit()

    def _mark_indeterminate_sync(
        self, node_id: str, request_id: str, body_sha256: str, error: str
    ) -> None:
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE generation_results SET state='indeterminate', error=?, updated_at=? "
                "WHERE node_id=? AND request_id=? AND body_sha256=? AND state='pending'",
                (error[:500], now, node_id, request_id, body_sha256),
            )
            conn.commit()

    async def execute(
        self,
        *,
        node_id: str,
        request_id: str,
        body_sha256: str,
        operation: Callable[[], Awaitable[GenerationResult]],
    ) -> GenerationResult:
        """Call ``operation`` once, or return its already-persisted response."""
        key = (node_id, request_id)
        async with self._singleflight(key):
            try:
                replay = await asyncio.to_thread(self._reserve_sync, node_id, request_id, body_sha256)
            except (GenerationIdentityConflict, GenerationIndeterminate):
                raise
            except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
                raise SpoolUnavailable("could not reserve generation request") from exc
            if replay is not None:
                return replay

            try:
                result = await operation()
            except BaseException as exc:
                reason = f"{type(exc).__name__}: {exc}"
                try:
                    await asyncio.shield(asyncio.to_thread(
                        self._mark_indeterminate_sync, node_id, request_id, body_sha256, reason,
                    ))
                except BaseException:
                    pass
                raise

            try:
                await asyncio.to_thread(
                    self._record_response_sync, node_id, request_id, body_sha256, result,
                )
            except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
                try:
                    await asyncio.shield(asyncio.to_thread(
                        self._mark_indeterminate_sync, node_id, request_id, body_sha256,
                        f"response could not be committed: {type(exc).__name__}",
                    ))
                except BaseException:
                    pass
                raise SpoolUnavailable("Worker replied, but its result was not durably committed") from exc
            return result

    def _snapshot_sync(self, node_id: str, request_id: str) -> dict[str, object] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT node_id, request_id, body_sha256, state, status_code, "
                "response_body, response_headers, error FROM generation_results "
                "WHERE node_id=? AND request_id=?",
                (node_id, request_id),
            ).fetchone()
            if row is None:
                return None
            return {
                "node_id": row["node_id"],
                "request_id": row["request_id"],
                "body_sha256": row["body_sha256"],
                "state": row["state"],
                "status_code": row["status_code"],
                "response_body": bytes(row["response_body"] or b""),
                "response_headers": json.loads(row["response_headers"] or "[]"),
                "error": row["error"],
            }

    async def snapshot(self, node_id: str, request_id: str) -> dict[str, object] | None:
        """Read-only evidence helper, also useful for focused acceptance tests."""
        try:
            return await asyncio.to_thread(self._snapshot_sync, node_id, request_id)
        except (OSError, sqlite3.Error) as exc:
            raise SpoolUnavailable("could not inspect generation result spool") from exc
