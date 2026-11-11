"""Durable, role-neutral node registration and desired-state assignments."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import threading
import time
from typing import Any, Callable, Iterator
from pydantic import TypeAdapter
from urllib.parse import urlsplit

from .contracts import (
    ApplyAckReceipt,
    ApplyAckRequest,
    DesiredStateResponse,
    FleetApprovalReceipt,
    FleetAssignmentReceipt,
    FleetDeactivationReceipt,
    FleetHeartbeatReceipt,
    LearnerDeployment,
    FleetNodeRegistration,
    FleetNodeRegistrationReceipt,
    FleetNodeView,
    FleetRejectionAuditRecord,
    FleetRejectionReceipt,
    FleetRoleDeployment,
    FleetSSHGrant,
    LearnerDeploymentTemplate,
    FleetRoleTemplate,
    NodeHeartbeat,
    PrivateDeploymentEnvelope,
    UserAccount,
    WorkerDeployment,
    role_deployment_canonical_json,
    role_deployment_sha256,
    role_template_sha256,
)


_BEARER = re.compile(r"^[A-Za-z0-9_-]{32,512}$")
_SECRET_TEXT = re.compile(r"(?i)\b(bearer\s+|(?:token|secret|password|credential|authorization)\s*[:=]\s*)[^\s,;]+")
_URL_TEXT = re.compile(r"https?://[^\s\])}>,;]+", re.IGNORECASE)
_SECRET_KEY = re.compile(r"(?i)(token|bearer|authorization|secret|password|credential|private.?key|join.?code)")
_ROLE_DEPLOYMENT = TypeAdapter(FleetRoleDeployment)
_ROLE_TEMPLATE = TypeAdapter(FleetRoleTemplate)


def _iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _token_hash(value: str) -> str:
    if not isinstance(value, str) or _BEARER.fullmatch(value) is None:
        raise FleetRegistryError(401, "invalid_node_token", "node bearer token is invalid")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _safe_text(value: str | None, *, maximum: int = 512, secrets_to_redact: tuple[str, ...] = ()) -> str | None:
    if value is None:
        return None
    text = value
    for secret in sorted((item for item in secrets_to_redact if len(item) >= 8), key=len, reverse=True):
        text = text.replace(secret, "[redacted]")
    text = _SECRET_TEXT.sub(r"\1[redacted]", text)
    text = _URL_TEXT.sub("[url]", text)
    text = "".join(ch for ch in text if ch >= " " and ch != "\x7f")
    return text[:maximum]


def _safe_value(value: Any, *, depth: int = 0, secrets_to_redact: tuple[str, ...] = ()) -> Any:
    if depth > 8:
        return "[depth-limited]"
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            key_text = str(key)[:128]
            output[key_text] = "[redacted]" if _SECRET_KEY.search(key_text) else _safe_value(
                item, depth=depth + 1, secrets_to_redact=secrets_to_redact,
            )
        return output
    if isinstance(value, list):
        return [_safe_value(item, depth=depth + 1, secrets_to_redact=secrets_to_redact) for item in value[:1000]]
    if isinstance(value, str):
        return _safe_text(value, maximum=4096, secrets_to_redact=secrets_to_redact)
    if value is None or type(value) in {bool, int, float}:
        return value
    return str(type(value).__name__)


class FleetRegistryError(Exception):
    """Expected, sanitized control-plane failure."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class FleetRegistry:
    """SQLite-backed role-neutral registration and versioned deployment state.

    Node bearers are selected and persisted by the node before registration.
    The Registry stores only SHA-256 digests. Private role deployment documents
    remain in this separate, mode-0600 Fleet database; the existing training
    Registry and its Worker/Gateway tokens are not used or modified.
    """

    def __init__(
        self,
        state_dir: str | Path,
        *,
        heartbeat_ttl_seconds: int = 60,
        max_registered_nodes: int = 1000,
        max_pending_nodes: int = 100,
        max_rejected_tombstones: int = 1000,
        registration_rate_per_minute: int = 120,
        reverse_port_start: int = 24000,
        reverse_port_end: int = 34000,
        public_origin: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if type(heartbeat_ttl_seconds) is not int or heartbeat_ttl_seconds < 1:
            raise ValueError("heartbeat_ttl_seconds must be a positive integer")
        if type(max_registered_nodes) is not int or max_registered_nodes < 1:
            raise ValueError("max_registered_nodes must be a positive integer")
        if type(max_pending_nodes) is not int or not 1 <= max_pending_nodes <= max_registered_nodes:
            raise ValueError("max_pending_nodes must be within the registered node limit")
        if type(max_rejected_tombstones) is not int or max_rejected_tombstones < 1:
            raise ValueError("max_rejected_tombstones must be a positive integer")
        if type(registration_rate_per_minute) is not int or registration_rate_per_minute < 50:
            raise ValueError("registration_rate_per_minute must allow a shared burst of at least 50 nodes")
        if (type(reverse_port_start) is not int or type(reverse_port_end) is not int
                or not 1024 <= reverse_port_start <= reverse_port_end <= 65535
                or reverse_port_end - reverse_port_start + 1 < max_registered_nodes):
            raise ValueError("reverse SSH port range must cover all registered nodes")
        root = Path(state_dir).expanduser()
        if not root.is_absolute():
            raise ValueError("Fleet state_dir must be absolute")
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise ValueError("Fleet state_dir must be a real directory")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        self.state_dir = root.resolve()
        if self.state_dir in {Path("/"), Path("/etc")} or Path("/etc") in self.state_dir.parents:
            raise ValueError("Fleet state_dir cannot be / or inside /etc")
        self.db_path = self.state_dir / "fleet.sqlite3"
        if self.db_path.is_symlink():
            raise ValueError("Fleet database path must not be a symlink")
        if not self.db_path.exists():
            descriptor = os.open(self.db_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        self.db_path.chmod(0o600)
        self.heartbeat_ttl_seconds = heartbeat_ttl_seconds
        self.max_registered_nodes = max_registered_nodes
        self.max_pending_nodes = max_pending_nodes
        self.max_rejected_tombstones = max_rejected_tombstones
        self.registration_rate_per_minute = registration_rate_per_minute
        self.reverse_port_start = reverse_port_start
        self.reverse_port_end = reverse_port_end
        self.public_origin = public_origin.rstrip("/") if public_origin else None
        self._clock = clock or time.time
        self._schema_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._schema_lock:
            connection = self._connect()
            try:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS fleet_nodes (
                        node_id TEXT PRIMARY KEY,
                        token_hash TEXT NOT NULL UNIQUE,
                        approval TEXT NOT NULL CHECK(approval IN ('pending','approved','rejected')),
                        boot_id TEXT NOT NULL,
                        hostname TEXT NOT NULL,
                        os_release TEXT NOT NULL,
                        architecture TEXT NOT NULL,
                        agent_version TEXT NOT NULL,
                        lan_host TEXT,
                        ssh_port INTEGER NOT NULL,
                        ssh_public_key TEXT NOT NULL,
                        ssh_public_key_fingerprint TEXT NOT NULL UNIQUE,
                        ssh_host_public_key TEXT NOT NULL,
                        ssh_host_fingerprint TEXT NOT NULL UNIQUE,
                        gpus_json TEXT NOT NULL,
                        reverse_port INTEGER UNIQUE,
                        registered_at REAL NOT NULL,
                        rejected_at REAL,
                        last_heartbeat_at REAL,
                        desired_generation INTEGER NOT NULL DEFAULT 0 CHECK(desired_generation >= 0),
                        desired_action TEXT NOT NULL DEFAULT 'none' CHECK(desired_action IN ('none','install','deactivate')),
                        desired_role TEXT CHECK(desired_role IS NULL OR desired_role IN ('worker','learner')),
                        deployment_id TEXT,
                        deployment_sha256 TEXT,
                        observed_generation INTEGER NOT NULL DEFAULT 0 CHECK(observed_generation >= 0),
                        observed_state TEXT NOT NULL DEFAULT 'registered',
                        observed_role TEXT CHECK(observed_role IS NULL OR observed_role IN ('worker','learner')),
                        observed_error_code TEXT,
                        observed_error_message TEXT,
                        observed_receipt_json TEXT
                    );
                    CREATE INDEX IF NOT EXISTS fleet_nodes_approval_idx ON fleet_nodes(approval, registered_at, node_id);
                    CREATE TABLE IF NOT EXISTS fleet_node_accounts (
                        node_id TEXT PRIMARY KEY REFERENCES fleet_nodes(node_id) ON DELETE CASCADE,
                        account_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS fleet_deployments (
                        node_id TEXT NOT NULL REFERENCES fleet_nodes(node_id) ON DELETE CASCADE,
                        generation INTEGER NOT NULL CHECK(generation > 0),
                        action TEXT NOT NULL CHECK(action IN ('install','deactivate')),
                        role TEXT CHECK(role IS NULL OR role IN ('worker','learner')),
                        deployment_id TEXT,
                        template_sha256 TEXT,
                        deployment_sha256 TEXT,
                        private_bundle_sha256 TEXT,
                        private_bundle_grant_hash TEXT UNIQUE,
                        deployment_json TEXT,
                        created_at REAL NOT NULL,
                        PRIMARY KEY(node_id, generation)
                    );
                    CREATE TABLE IF NOT EXISTS fleet_private_artifacts (
                        sha256 TEXT PRIMARY KEY,
                        publisher_bearer_hash TEXT NOT NULL UNIQUE,
                        size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS fleet_registration_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS fleet_registration_events_time_idx ON fleet_registration_events(created_at);
                    CREATE TABLE IF NOT EXISTS fleet_rejection_audit (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        node_id TEXT NOT NULL,
                        rejected_at REAL NOT NULL,
                        pruned_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS fleet_rejection_audit_time_idx ON fleet_rejection_audit(pruned_at,event_id);
                    """
                )
                account_columns = {
                    row["name"]: row for row in connection.execute("PRAGMA table_info(fleet_node_accounts)")
                }
                account_foreign_keys = connection.execute(
                    "PRAGMA foreign_key_list(fleet_node_accounts)"
                ).fetchall()
                foreign_key_ok = any(
                    row["table"] == "fleet_nodes" and row["from"] == "node_id"
                    and row["to"] == "node_id" and row["on_delete"].upper() == "CASCADE"
                    for row in account_foreign_keys
                )
                if (set(account_columns) != {"node_id", "account_json"}
                        or account_columns["node_id"]["pk"] != 1
                        or account_columns["account_json"]["notnull"] != 1
                        or not foreign_key_ok):
                    raise RuntimeError("Fleet node account table does not match the required immutable account schema")
                self.db_path.chmod(0o600)
            finally:
                connection.close()

    def register_node(self, registration: FleetNodeRegistration, *, bearer: str) -> FleetNodeRegistrationReceipt:
        if not isinstance(registration, FleetNodeRegistration):
            registration = FleetNodeRegistration.model_validate(registration, strict=True)
        digest = _token_hash(bearer)
        now = self._clock()
        data = registration.model_dump(mode="json")
        gpus_json = _json(data["gpus"])
        account_json = _json(data["account"])
        from ..identity import canonical_public_key

        ssh_key = canonical_public_key(registration.ssh_public_key)
        host_key = canonical_public_key(registration.ssh_host_public_key)
        from .contracts import ssh_key_fingerprint

        ssh_fp = ssh_key_fingerprint(ssh_key)
        host_fp = ssh_key_fingerprint(host_key)
        with self._transaction() as connection:
            existing = connection.execute("SELECT * FROM fleet_nodes WHERE node_id=?", (registration.node_id,)).fetchone()
            if existing is not None:
                if not hmac.compare_digest(existing["token_hash"], digest):
                    raise FleetRegistryError(409, "identity_conflict", "node identity or bearer token conflicts with its registration")
                if existing["ssh_public_key_fingerprint"] != ssh_fp or existing["ssh_host_fingerprint"] != host_fp:
                    raise FleetRegistryError(409, "identity_conflict", "node SSH identity changed for an existing registration")
                if existing["approval"] == "rejected":
                    raise FleetRegistryError(403, "node_rejected", "this stable node identity was rejected by central approval")
                account_row = connection.execute(
                    "SELECT account_json FROM fleet_node_accounts WHERE node_id=?", (registration.node_id,),
                ).fetchone()
                if account_row is None:
                    raise FleetRegistryError(
                        409, "node_account_missing",
                        "existing node has no registered Unix account; account identity cannot be inferred",
                    )
                stored_account = self._decode_account(account_row["account_json"])
                if stored_account.model_dump(mode="json") != data["account"]:
                    raise FleetRegistryError(409, "account_conflict", "Unix account changed for an existing node identity")
                try:
                    connection.execute(
                        """UPDATE fleet_nodes SET boot_id=?,hostname=?,os_release=?,architecture=?,agent_version=?,
                           lan_host=?,ssh_port=?,gpus_json=? WHERE node_id=?""",
                        (registration.boot_id, registration.hostname, registration.os_release,
                         registration.architecture, registration.agent_version, registration.lan_host,
                         registration.ssh_port, gpus_json, registration.node_id),
                    )
                except sqlite3.IntegrityError as exc:
                    raise FleetRegistryError(409, "identity_conflict", "node registration conflicts with an existing SSH identity") from exc
            else:
                # Rejected identities remain durable tombstones, so count them
                # against the lifetime node limit to keep the registry bounded.
                total = int(connection.execute(
                    "SELECT COUNT(*) FROM fleet_nodes WHERE approval IN ('pending','approved')"
                ).fetchone()[0])
                pending = int(connection.execute("SELECT COUNT(*) FROM fleet_nodes WHERE approval='pending'").fetchone()[0])
                if total >= self.max_registered_nodes or pending >= self.max_pending_nodes:
                    raise FleetRegistryError(429, "registration_limit", "Fleet registration capacity is full")
                connection.execute("DELETE FROM fleet_registration_events WHERE created_at <= ?", (now - 60.0,))
                recent = int(connection.execute("SELECT COUNT(*) FROM fleet_registration_events").fetchone()[0])
                if recent >= self.registration_rate_per_minute:
                    raise FleetRegistryError(429, "registration_rate_limited", "Fleet registration rate limit is full")
                try:
                    connection.execute(
                        """INSERT INTO fleet_nodes(
                           node_id,token_hash,approval,boot_id,hostname,os_release,architecture,agent_version,lan_host,
                           ssh_port,ssh_public_key,ssh_public_key_fingerprint,ssh_host_public_key,ssh_host_fingerprint,
                           gpus_json,registered_at
                           ) VALUES(?,?,'pending',?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (registration.node_id, digest, registration.boot_id, registration.hostname,
                         registration.os_release, registration.architecture, registration.agent_version,
                         registration.lan_host, registration.ssh_port, ssh_key, ssh_fp, host_key, host_fp,
                         gpus_json, now),
                    )
                except sqlite3.IntegrityError as exc:
                    raise FleetRegistryError(409, "identity_conflict", "node identity conflicts with an existing registration") from exc
                connection.execute(
                    "INSERT INTO fleet_node_accounts(node_id,account_json) VALUES(?,?)",
                    (registration.node_id, account_json),
                )
                connection.execute("INSERT INTO fleet_registration_events(created_at) VALUES(?)", (now,))
            row = connection.execute("SELECT * FROM fleet_nodes WHERE node_id=?", (registration.node_id,)).fetchone()
            return self._registration_receipt(row)

    def _registration_receipt(self, row: sqlite3.Row) -> FleetNodeRegistrationReceipt:
        return FleetNodeRegistrationReceipt(
            node_id=row["node_id"], approval=row["approval"], registered_at=_iso(row["registered_at"]),
            desired_generation=row["desired_generation"], reverse_port=row["reverse_port"],
        )

    def authenticate_node(self, bearer: str) -> str | None:
        try:
            digest = _token_hash(bearer)
        except FleetRegistryError:
            return None
        with self._reader() as connection:
            row = connection.execute("SELECT node_id FROM fleet_nodes WHERE token_hash=?", (digest,)).fetchone()
        return row["node_id"] if row is not None else None

    def authenticate_ssh_bearer(self, bearer: str) -> FleetSSHGrant | None:
        try:
            digest = _token_hash(bearer)
        except FleetRegistryError:
            return None
        with self._reader() as connection:
            row = connection.execute("SELECT * FROM fleet_nodes WHERE token_hash=?", (digest,)).fetchone()
        if row is None or row["approval"] != "approved" or row["reverse_port"] is None:
            return None
        return FleetSSHGrant(
            node_id=row["node_id"], reverse_port=row["reverse_port"], ssh_port=row["ssh_port"],
            ssh_host_public_key=row["ssh_host_public_key"], ssh_public_key=row["ssh_public_key"],
            ssh_public_key_fingerprint=row["ssh_public_key_fingerprint"],
        )

    def _require_node(self, connection: sqlite3.Connection, node_id: str, bearer: str) -> sqlite3.Row:
        digest = _token_hash(bearer)
        row = connection.execute("SELECT * FROM fleet_nodes WHERE node_id=?", (node_id,)).fetchone()
        if row is None:
            raise FleetRegistryError(404, "node_not_found", "Fleet node was not found")
        if not hmac.compare_digest(row["token_hash"], digest):
            raise FleetRegistryError(401, "invalid_node_token", "node bearer token is invalid")
        return row

    def _assigned_secrets(self, connection: sqlite3.Connection, row: sqlite3.Row) -> tuple[str, ...]:
        if row["desired_action"] != "install":
            return ()
        deployment = connection.execute(
            "SELECT deployment_json FROM fleet_deployments WHERE node_id=? AND generation=? AND action='install'",
            (row["node_id"], row["desired_generation"]),
        ).fetchone()
        if deployment is None or deployment["deployment_json"] is None:
            return ()
        try:
            value = json.loads(deployment["deployment_json"])
        except (TypeError, ValueError):
            return ()
        if not isinstance(value, dict):
            return ()
        secrets_found = tuple(
            secret for key in ("worker_join_code", "bundle_bearer")
            if isinstance((secret := value.get(key)), str)
        )
        return secrets_found

    def heartbeat(self, node_id: str, heartbeat: NodeHeartbeat, *, bearer: str) -> FleetHeartbeatReceipt:
        if not isinstance(heartbeat, NodeHeartbeat):
            heartbeat = NodeHeartbeat.model_validate(heartbeat, strict=True)
        now = self._clock()
        with self._transaction() as connection:
            row = self._require_node(connection, node_id, bearer)
            if heartbeat.boot_id != row["boot_id"]:
                raise FleetRegistryError(409, "boot_id_mismatch", "node must re-register after a system reboot")
            if heartbeat.observed_generation > row["desired_generation"]:
                raise FleetRegistryError(409, "generation_mismatch", "heartbeat reports a future desired generation")
            if heartbeat.state == "ready":
                if (heartbeat.observed_generation != row["desired_generation"]
                        or row["desired_action"] != "install" or heartbeat.role != row["desired_role"]):
                    raise FleetRegistryError(409, "invalid_ready_state", "ready heartbeat must match the current assigned role and generation")
            if heartbeat.state == "deactivated":
                if heartbeat.observed_generation != row["desired_generation"] or row["desired_action"] != "deactivate":
                    raise FleetRegistryError(409, "invalid_deactivated_state", "deactivated heartbeat must match an explicit deactivation generation")
            if heartbeat.role is not None and row["desired_action"] == "install" and heartbeat.observed_generation == row["desired_generation"]:
                if heartbeat.role != row["desired_role"]:
                    raise FleetRegistryError(409, "role_mismatch", "heartbeat role does not match the current assignment")
            known_secrets = (bearer, *self._assigned_secrets(connection, row))
            safe_receipt = _safe_value(heartbeat.receipt, secrets_to_redact=known_secrets) if heartbeat.receipt is not None else None
            connection.execute(
                """UPDATE fleet_nodes SET last_heartbeat_at=?,agent_version=?,observed_generation=?,observed_state=?,
                   observed_role=?,observed_error_code=?,observed_error_message=?,observed_receipt_json=? WHERE node_id=?""",
                (now, heartbeat.agent_version, heartbeat.observed_generation, heartbeat.state, heartbeat.role,
                 _safe_text(heartbeat.error_code, maximum=100, secrets_to_redact=known_secrets),
                 _safe_text(heartbeat.error_message, secrets_to_redact=known_secrets),
                 _json(safe_receipt) if safe_receipt is not None else None, node_id),
            )
            return FleetHeartbeatReceipt(
                node_id=node_id, accepted=True, desired_generation=row["desired_generation"], server_time=_iso(now),
            )

    def desired_state(self, node_id: str, *, bearer: str) -> DesiredStateResponse:
        with self._reader() as connection:
            row = self._require_node(connection, node_id, bearer)
        return DesiredStateResponse(
            approval=row["approval"], generation=row["desired_generation"], action=row["desired_action"],
            role=row["desired_role"], deployment_id=row["deployment_id"],
            deployment_sha256=row["deployment_sha256"], reverse_port=row["reverse_port"],
        )

    def deployment_for(self, node_id: str, generation: int, *, bearer: str) -> PrivateDeploymentEnvelope:
        with self._reader() as connection:
            row = self._require_node(connection, node_id, bearer)
            if row["approval"] != "approved":
                raise FleetRegistryError(403, "node_not_approved", "node is awaiting central approval")
            if generation != row["desired_generation"]:
                raise FleetRegistryError(409, "generation_mismatch", "deployment request is not for the current generation")
            deployment = connection.execute(
                "SELECT deployment_json FROM fleet_deployments WHERE node_id=? AND generation=? AND action='install'",
                (node_id, generation),
            ).fetchone()
        if row["desired_action"] != "install" or deployment is None or deployment["deployment_json"] is None:
            raise FleetRegistryError(409, "deployment_unavailable", "there is no private install document for this generation")
        try:
            value = json.loads(deployment["deployment_json"])
            role_deployment = _ROLE_DEPLOYMENT.validate_python(value, strict=True)
        except Exception as exc:
            raise FleetRegistryError(503, "deployment_integrity_error", "stored deployment document failed validation") from exc
        if role_deployment.deployment_id != row["deployment_id"] or role_deployment_sha256(role_deployment) != row["deployment_sha256"]:
            raise FleetRegistryError(503, "deployment_integrity_error", "stored deployment document does not match its public descriptor")
        return PrivateDeploymentEnvelope(node_id=node_id, generation=generation, deployment=role_deployment)

    def publish_private_artifact(self, sha256: str, *, publisher_bearer: str, size_bytes: int) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None or type(size_bytes) is not int or size_bytes < 0:
            raise FleetRegistryError(422, "invalid_private_artifact", "private artifact metadata is invalid")
        digest = _token_hash(publisher_bearer)
        with self._transaction() as connection:
            row = connection.execute("SELECT publisher_bearer_hash,size_bytes FROM fleet_private_artifacts WHERE sha256=?", (sha256,)).fetchone()
            if row is not None:
                if hmac.compare_digest(row["publisher_bearer_hash"], digest) and row["size_bytes"] == size_bytes:
                    return
                raise FleetRegistryError(409, "private_artifact_exists", "private artifact already exists with a different grant")
            try:
                connection.execute(
                    "INSERT INTO fleet_private_artifacts(sha256,publisher_bearer_hash,size_bytes,created_at) VALUES(?,?,?,?)",
                    (sha256, digest, size_bytes, self._clock()),
                )
            except sqlite3.IntegrityError as exc:
                raise FleetRegistryError(409, "private_artifact_conflict", "private artifact grant conflicts with an existing artifact") from exc

    def authorize_private_artifact(self, sha256: str, *, bearer: str) -> int | None:
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            return None
        try:
            digest = _token_hash(bearer)
        except FleetRegistryError:
            return None
        with self._reader() as connection:
            row = connection.execute(
                """SELECT a.size_bytes FROM fleet_private_artifacts a
                   JOIN fleet_nodes n ON n.approval='approved' AND n.desired_action='install'
                     AND n.desired_role='learner'
                   JOIN fleet_deployments d ON d.node_id=n.node_id AND d.generation=n.desired_generation
                     AND d.action='install'
                   WHERE a.sha256=? AND d.private_bundle_sha256=a.sha256
                     AND d.private_bundle_grant_hash=?""",
                (sha256, digest),
            ).fetchone()
        return int(row["size_bytes"]) if row is not None else None

    def acknowledge(self, node_id: str, ack: ApplyAckRequest, *, bearer: str) -> ApplyAckReceipt:
        if not isinstance(ack, ApplyAckRequest):
            ack = ApplyAckRequest.model_validate(ack, strict=True)
        now = self._clock()
        with self._transaction() as connection:
            row = self._require_node(connection, node_id, bearer)
            if row["approval"] != "approved":
                raise FleetRegistryError(403, "node_not_approved", "node is awaiting central approval")
            if ack.generation != row["desired_generation"] or ack.generation == 0:
                raise FleetRegistryError(409, "generation_mismatch", "acknowledgement is not for the current generation")
            action = row["desired_action"]
            if action == "install":
                if ack.role is not None and ack.role != row["desired_role"]:
                    raise FleetRegistryError(409, "role_mismatch", "acknowledgement role does not match the current assignment")
                if ack.status == "ready" and ack.role != row["desired_role"]:
                    raise FleetRegistryError(409, "role_mismatch", "ready acknowledgement must name the assigned role")
                if ack.status == "deactivated":
                    raise FleetRegistryError(409, "action_mismatch", "install generation cannot be acknowledged as deactivated")
            elif action == "deactivate":
                if ack.status == "ready":
                    raise FleetRegistryError(409, "action_mismatch", "deactivation generation cannot be acknowledged as ready")
            else:
                raise FleetRegistryError(409, "generation_mismatch", "node has no assigned operation for this generation")

            known_secrets = (bearer, *self._assigned_secrets(connection, row))
            safe_receipt = _safe_value(ack.receipt, secrets_to_redact=known_secrets) if ack.receipt is not None else None
            receipt_json = _json(safe_receipt) if safe_receipt is not None else None
            error_code = _safe_text(ack.error_code, maximum=100, secrets_to_redact=known_secrets)
            error_message = _safe_text(ack.error_message, secrets_to_redact=known_secrets)

            terminal = {"ready", "failed", "deactivated"}
            if row["observed_generation"] == ack.generation and row["observed_state"] in terminal:
                identical = (
                    row["observed_state"] == ack.status
                    and row["observed_role"] == ack.role
                    and row["observed_error_code"] == error_code
                    and row["observed_error_message"] == error_message
                    and row["observed_receipt_json"] == receipt_json
                )
                if not identical:
                    raise FleetRegistryError(409, "generation_already_completed", "completed generation cannot be changed; create a new generation to retry")
                return ApplyAckReceipt(node_id=node_id, generation=ack.generation, status=ack.status, accepted=True)

            connection.execute(
                """UPDATE fleet_nodes SET last_heartbeat_at=?,observed_generation=?,observed_state=?,observed_role=?,
                   observed_error_code=?,observed_error_message=?,observed_receipt_json=? WHERE node_id=?""",
                (now, ack.generation, ack.status, ack.role, error_code, error_message, receipt_json, node_id),
            )
        return ApplyAckReceipt(node_id=node_id, generation=ack.generation, status=ack.status, accepted=True)

    def approve_node(self, node_id: str) -> FleetApprovalReceipt:
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM fleet_nodes WHERE node_id=?", (node_id,)).fetchone()
            if row is None:
                raise FleetRegistryError(404, "node_not_found", "Fleet node was not found")
            if row["approval"] == "rejected":
                raise FleetRegistryError(409, "node_rejected", "a rejected node identity cannot be approved")
            port = row["reverse_port"]
            if row["approval"] == "pending":
                port = self._allocate_reverse_port(connection)
                if port is None:
                    raise FleetRegistryError(409, "relay_capacity_full", "no management SSH listener is available")
                connection.execute(
                    "UPDATE fleet_nodes SET approval='approved',reverse_port=?,desired_generation=desired_generation+1 WHERE node_id=?",
                    (port, node_id),
                )
                row = connection.execute("SELECT * FROM fleet_nodes WHERE node_id=?", (node_id,)).fetchone()
            return FleetApprovalReceipt(node_id=node_id, approval="approved", generation=row["desired_generation"],
                                        reverse_port=port)

    def reject_node(self, node_id: str) -> FleetRejectionReceipt:
        with self._transaction() as connection:
            row = connection.execute("SELECT approval,rejected_at FROM fleet_nodes WHERE node_id=?", (node_id,)).fetchone()
            if row is None:
                raise FleetRegistryError(404, "node_not_found", "Fleet node was not found")
            if row["approval"] == "approved":
                raise FleetRegistryError(409, "cannot_reject_approved_node", "approved nodes require role deactivation and session closure")
            if row["approval"] == "rejected":
                return FleetRejectionReceipt(node_id=node_id, approval="rejected", rejected_at=_iso(row["rejected_at"]), changed=False)
            now = self._clock()
            connection.execute("UPDATE fleet_nodes SET approval='rejected',rejected_at=? WHERE node_id=?", (now, node_id))
            rejected_count = int(connection.execute(
                "SELECT COUNT(*) FROM fleet_nodes WHERE approval='rejected'"
            ).fetchone()[0])
            excess = rejected_count - self.max_rejected_tombstones
            if excess > 0:
                stale = connection.execute(
                    """SELECT node_id,rejected_at FROM fleet_nodes WHERE approval='rejected'
                       ORDER BY rejected_at ASC,rowid ASC LIMIT ?""",
                    (excess,),
                ).fetchall()
                for item in stale:
                    connection.execute(
                        "INSERT INTO fleet_rejection_audit(node_id,rejected_at,pruned_at) VALUES(?,?,?)",
                        (item["node_id"], item["rejected_at"], now),
                    )
                    connection.execute("DELETE FROM fleet_nodes WHERE node_id=? AND approval='rejected'", (item["node_id"],))
                connection.execute(
                    """DELETE FROM fleet_rejection_audit WHERE event_id NOT IN
                       (SELECT event_id FROM fleet_rejection_audit ORDER BY pruned_at DESC,event_id DESC LIMIT ?)""",
                    (self.max_rejected_tombstones,),
                )
        return FleetRejectionReceipt(node_id=node_id, approval="rejected", rejected_at=_iso(now), changed=True)

    def _allocate_reverse_port(self, connection: sqlite3.Connection) -> int | None:
        used = {row[0] for row in connection.execute("SELECT reverse_port FROM fleet_nodes WHERE reverse_port IS NOT NULL")}
        for port in range(self.reverse_port_start, self.reverse_port_end + 1):
            if port not in used:
                return port
        return None

    def assign_node(self, node_id: str, deployment: FleetRoleTemplate, *,
                    expected_generation: int | None = None) -> FleetAssignmentReceipt:
        if expected_generation is not None and (type(expected_generation) is not int or expected_generation < 0):
            raise FleetRegistryError(422, "invalid_expected_generation", "expected generation must be a nonnegative integer")
        if not isinstance(deployment, (WorkerDeployment, LearnerDeploymentTemplate)):
            try:
                deployment = _ROLE_TEMPLATE.validate_python(deployment, strict=True)
            except Exception as exc:
                raise FleetRegistryError(422, "invalid_deployment", "role deployment document is invalid") from exc
        template_sha256 = role_template_sha256(deployment)
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM fleet_nodes WHERE node_id=?", (node_id,)).fetchone()
            if row is None:
                raise FleetRegistryError(404, "node_not_found", "Fleet node was not found")
            if row["approval"] != "approved":
                raise FleetRegistryError(409, "node_not_approved", "approve the node before assigning a role")
            private_bundle_sha256 = None
            private_bundle_grant_hash = None
            if isinstance(deployment, LearnerDeploymentTemplate):
                expected_url = (
                    self.public_origin + f"/v1/fleet/private-artifacts/{deployment.private_bundle.sha256}"
                    if self.public_origin else None
                )
                if expected_url is not None and deployment.private_bundle.url == expected_url:
                    if deployment.publisher_bearer is None or deployment.external_bearer is not None:
                        raise FleetRegistryError(422, "invalid_bundle_grant", "managed Learner bundles require their publisher bearer")
                    private_row = connection.execute(
                        "SELECT publisher_bearer_hash,size_bytes FROM fleet_private_artifacts WHERE sha256=?",
                        (deployment.private_bundle.sha256,),
                    ).fetchone()
                    supplied_hash = hashlib.sha256(
                        deployment.publisher_bearer.get_secret_value().encode("ascii")
                    ).hexdigest()
                    artifact_path = self.state_dir / "private-artifacts" / deployment.private_bundle.sha256
                    try:
                        root_info = artifact_path.parent.lstat()
                        artifact_info = artifact_path.lstat()
                    except OSError:
                        root_info = artifact_info = None
                    artifact_ready = (
                        root_info is not None and stat.S_ISDIR(root_info.st_mode)
                        and stat.S_IMODE(root_info.st_mode) == 0o700 and root_info.st_uid == os.geteuid()
                        and artifact_info is not None and stat.S_ISREG(artifact_info.st_mode)
                        and stat.S_IMODE(artifact_info.st_mode) == 0o600 and artifact_info.st_uid == os.geteuid()
                        and private_row is not None and artifact_info.st_size == private_row["size_bytes"]
                    )
                    if (private_row is None or not hmac.compare_digest(private_row["publisher_bearer_hash"], supplied_hash)
                            or not artifact_ready):
                        raise FleetRegistryError(409, "private_artifact_not_ready", "Learner bundle must be published in this Fleet private artifact store")
                    private_bundle_sha256 = deployment.private_bundle.sha256
                else:
                    if deployment.publisher_bearer is not None or deployment.external_bearer is None:
                        raise FleetRegistryError(422, "invalid_external_bundle_grant", "external Learner bundles require an external bearer")
                    if self.public_origin is not None:
                        try:
                            external = urlsplit(deployment.private_bundle.url)
                            local = urlsplit(self.public_origin)
                            external_port = external.port or 443
                            local_port = local.port or 443
                        except ValueError as exc:
                            raise FleetRegistryError(422, "invalid_external_bundle_url", "external bundle URL is invalid") from exc
                        if (external.scheme == local.scheme and external.hostname == local.hostname
                                and external_port == local_port):
                            raise FleetRegistryError(422, "invalid_external_bundle_url", "same-origin Learner bundles must use the managed private artifact path")
            current_assignment = connection.execute(
                "SELECT template_sha256 FROM fleet_deployments WHERE node_id=? AND generation=? AND action='install'",
                (node_id, row["desired_generation"]),
            ).fetchone() if row["desired_action"] == "install" else None
            if (current_assignment is not None
                    and hmac.compare_digest(current_assignment["template_sha256"] or "", template_sha256)):
                return FleetAssignmentReceipt(node_id=node_id, generation=row["desired_generation"], action="install",
                                              role=row["desired_role"], deployment_id=row["deployment_id"],
                                              deployment_sha256=row["deployment_sha256"], changed=False)

            # Replaying the same template is safe; a different concurrent assignment is not.
            if expected_generation is not None and row["desired_generation"] != expected_generation:
                raise FleetRegistryError(409, "generation_conflict", "Fleet desired generation changed before assignment")

            delivered: FleetRoleDeployment
            if isinstance(deployment, LearnerDeploymentTemplate):
                node_bundle_bearer = (
                    secrets.token_urlsafe(32)
                    if deployment.publisher_bearer is not None
                    else deployment.external_bearer.get_secret_value()
                )
                delivered = LearnerDeployment(
                    role="learner", deployment_id=deployment.deployment_id,
                    private_bundle=deployment.private_bundle, bundle_bearer=node_bundle_bearer,
                )
                if deployment.publisher_bearer is not None:
                    private_bundle_grant_hash = hashlib.sha256(node_bundle_bearer.encode("ascii")).hexdigest()
            else:
                delivered = deployment
            document = role_deployment_canonical_json(delivered)
            deployment_sha256 = role_deployment_sha256(delivered)
            generation = row["desired_generation"] + 1
            connection.execute(
                """INSERT INTO fleet_deployments(node_id,generation,action,role,deployment_id,deployment_sha256,
                   template_sha256,private_bundle_sha256,private_bundle_grant_hash,deployment_json,created_at)
                   VALUES(?,?,'install',?,?,?,?,?,?,?,?)""",
                (node_id, generation, delivered.role, delivered.deployment_id, deployment_sha256,
                 template_sha256, private_bundle_sha256, private_bundle_grant_hash, document, self._clock()),
            )
            connection.execute(
                """UPDATE fleet_nodes SET desired_generation=?,desired_action='install',desired_role=?,deployment_id=?,
                   deployment_sha256=? WHERE node_id=?""",
                (generation, delivered.role, delivered.deployment_id, deployment_sha256, node_id),
            )
        return FleetAssignmentReceipt(node_id=node_id, generation=generation, action="install", role=delivered.role,
                                      deployment_id=delivered.deployment_id, deployment_sha256=deployment_sha256, changed=True)

    def deactivate_node(self, node_id: str) -> FleetDeactivationReceipt:
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM fleet_nodes WHERE node_id=?", (node_id,)).fetchone()
            if row is None:
                raise FleetRegistryError(404, "node_not_found", "Fleet node was not found")
            if row["approval"] != "approved":
                raise FleetRegistryError(409, "node_not_approved", "approve the node before deactivation")
            if row["desired_action"] == "deactivate":
                return FleetDeactivationReceipt(node_id=node_id, generation=row["desired_generation"], action="deactivate", changed=False)
            generation = row["desired_generation"] + 1
            connection.execute(
                """INSERT INTO fleet_deployments(node_id,generation,action,role,deployment_id,deployment_sha256,
                   template_sha256,private_bundle_sha256,private_bundle_grant_hash,deployment_json,created_at)
                   VALUES(?,?,'deactivate',NULL,NULL,NULL,NULL,NULL,NULL,NULL,?)""",
                (node_id, generation, self._clock()),
            )
            connection.execute(
                """UPDATE fleet_nodes SET desired_generation=?,desired_action='deactivate',desired_role=NULL,
                   deployment_id=NULL,deployment_sha256=NULL WHERE node_id=?""",
                (generation, node_id),
            )
        return FleetDeactivationReceipt(node_id=node_id, generation=generation, action="deactivate", changed=True)

    def refresh_management(self, node_id: str) -> FleetDeactivationReceipt:
        """Advance an eligible, unassigned node to a new deactivation generation.

        This is an operator-only recovery operation for the node-owned management
        SSH/tunnel units. It deliberately uses the normal deactivation generation
        and audit row, so the existing agent consumes it through the public
        desired-state contract without inventing a role or bypassing the registry.
        """
        now = self._clock()
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM fleet_nodes WHERE node_id=?", (node_id,)).fetchone()
            if row is None:
                raise FleetRegistryError(404, "node_not_found", "Fleet node was not found")
            if row["approval"] != "approved":
                raise FleetRegistryError(409, "node_not_approved", "approve the node before refreshing management")
            if row["desired_action"] not in {"none", "deactivate"}:
                raise FleetRegistryError(409, "active_assignment", "management refresh is only available without an active role assignment")
            if row["desired_role"] is not None or row["observed_role"] is not None:
                raise FleetRegistryError(409, "node_has_active_role", "management refresh is blocked while a Worker or Learner role is present")
            if row["deployment_id"] is not None or row["deployment_sha256"] is not None:
                raise FleetRegistryError(409, "active_assignment", "management refresh is blocked by an active deployment descriptor")

            active_states = {"installing", "loading", "draining", "ready",
                             "needs_reboot", "needs_driver", "needs_host_packages"}
            if row["observed_state"] in active_states:
                raise FleetRegistryError(409, "node_operation_in_progress", "management refresh is blocked while a node operation is active")
            if row["observed_generation"] > row["desired_generation"]:
                raise FleetRegistryError(409, "generation_mismatch", "node observation is ahead of the current desired generation")

            try:
                observed_receipt = (json.loads(row["observed_receipt_json"])
                                    if row["observed_receipt_json"] is not None else None)
            except (TypeError, ValueError) as exc:
                raise FleetRegistryError(503, "registry_integrity_error", "Fleet node receipt could not be decoded") from exc
            if observed_receipt is not None and not isinstance(observed_receipt, dict):
                raise FleetRegistryError(503, "registry_integrity_error", "Fleet node receipt is not an object")
            if observed_receipt is not None:
                if observed_receipt.get("role") in {"worker", "learner"}:
                    raise FleetRegistryError(409, "node_has_active_role", "management refresh is blocked by a Worker or Learner receipt")
                if (observed_receipt.get("action") == "install"
                        or observed_receipt.get("status") in active_states):
                    raise FleetRegistryError(409, "node_operation_in_progress", "management refresh is blocked by the current operation receipt")

            current_deployment = connection.execute(
                "SELECT action,role FROM fleet_deployments WHERE node_id=? AND generation=?",
                (node_id, row["desired_generation"]),
            ).fetchone()
            previous_role = connection.execute(
                "SELECT 1 FROM fleet_deployments WHERE node_id=? AND action='install' LIMIT 1",
                (node_id,),
            ).fetchone()
            if row["desired_action"] == "deactivate":
                if (row["observed_generation"] != row["desired_generation"]
                        or row["observed_state"] != "deactivated"
                        or observed_receipt is None
                        or observed_receipt.get("generation") != row["desired_generation"]
                        or observed_receipt.get("action") != "deactivate"):
                    raise FleetRegistryError(409, "deactivation_unconfirmed", "the current role-free deactivation must be acknowledged before another management refresh")
                if (current_deployment is None or current_deployment["action"] != "deactivate"
                        or current_deployment["role"] is not None):
                    raise FleetRegistryError(503, "registry_integrity_error", "current deactivation audit record is missing or invalid")
                if previous_role is not None and (
                        row["last_heartbeat_at"] is None
                        or now - row["last_heartbeat_at"] > self.heartbeat_ttl_seconds):
                    raise FleetRegistryError(409, "heartbeat_stale", "a fresh heartbeat is required after role history")
            else:
                if current_deployment is not None or row["observed_state"] == "deactivated":
                    raise FleetRegistryError(409, "generation_mismatch", "role-free node state does not match the current desired action")
                if (observed_receipt is not None
                        and observed_receipt.get("generation") == row["desired_generation"]
                        and observed_receipt.get("action") == "deactivate"):
                    raise FleetRegistryError(409, "generation_mismatch", "current deactivation receipt does not match the desired action")
                if previous_role is not None:
                    raise FleetRegistryError(409, "deactivation_required", "a node with role history must acknowledge deactivation before management refresh")

            generation = row["desired_generation"] + 1
            connection.execute(
                """INSERT INTO fleet_deployments(node_id,generation,action,role,deployment_id,deployment_sha256,
                   template_sha256,private_bundle_sha256,private_bundle_grant_hash,deployment_json,created_at)
                   VALUES(?,?,'deactivate',NULL,NULL,NULL,NULL,NULL,NULL,NULL,?)""",
                (node_id, generation, now),
            )
            connection.execute(
                """UPDATE fleet_nodes SET desired_generation=?,desired_action='deactivate',desired_role=NULL,
                   deployment_id=NULL,deployment_sha256=NULL WHERE node_id=?""",
                (generation, node_id),
            )
        return FleetDeactivationReceipt(node_id=node_id, generation=generation, action="deactivate", changed=True)

    def list_nodes(self) -> list[FleetNodeView]:
        with self._reader() as connection:
            rows = connection.execute(
                """SELECT n.*,a.account_json AS account_json FROM fleet_nodes n
                   LEFT JOIN fleet_node_accounts a ON a.node_id=n.node_id ORDER BY n.registered_at,n.node_id"""
            ).fetchall()
        return [self._node_view(row) for row in rows]

    def get_node(self, node_id: str) -> FleetNodeView:
        with self._reader() as connection:
            row = connection.execute(
                """SELECT n.*,a.account_json AS account_json FROM fleet_nodes n
                   LEFT JOIN fleet_node_accounts a ON a.node_id=n.node_id WHERE n.node_id=?""", (node_id,),
            ).fetchone()
        if row is None:
            raise FleetRegistryError(404, "node_not_found", "Fleet node was not found")
        return self._node_view(row)

    def list_ssh_grants(self) -> list[FleetSSHGrant]:
        with self._reader() as connection:
            rows = connection.execute(
                "SELECT * FROM fleet_nodes WHERE approval='approved' AND reverse_port IS NOT NULL ORDER BY reverse_port"
            ).fetchall()
        return [
            FleetSSHGrant(
                node_id=row["node_id"], reverse_port=row["reverse_port"], ssh_port=row["ssh_port"],
                ssh_host_public_key=row["ssh_host_public_key"], ssh_public_key=row["ssh_public_key"],
                ssh_public_key_fingerprint=row["ssh_public_key_fingerprint"],
            )
            for row in rows
        ]

    def list_rejection_audit(self) -> list[FleetRejectionAuditRecord]:
        with self._reader() as connection:
            rows = connection.execute(
                "SELECT node_id,rejected_at,pruned_at FROM fleet_rejection_audit ORDER BY pruned_at DESC,event_id DESC"
            ).fetchall()
        return [
            FleetRejectionAuditRecord(node_id=row["node_id"], rejected_at=_iso(row["rejected_at"]),
                                      pruned_at=_iso(row["pruned_at"]))
            for row in rows
        ]

    def _node_view(self, row: sqlite3.Row) -> FleetNodeView:
        try:
            gpus = json.loads(row["gpus_json"])
            receipt = json.loads(row["observed_receipt_json"]) if row["observed_receipt_json"] else None
        except (TypeError, ValueError) as exc:
            raise FleetRegistryError(503, "registry_integrity_error", "Fleet node record could not be decoded") from exc
        if row["account_json"] is None:
            raise FleetRegistryError(
                503, "registry_account_missing", "Fleet node has no required registered Unix account",
            )
        account = self._decode_account(row["account_json"])
        return FleetNodeView(
            node_id=row["node_id"], boot_id=row["boot_id"], hostname=row["hostname"], os_release=row["os_release"],
            architecture=row["architecture"], agent_version=row["agent_version"], account=account,
            lan_host=row["lan_host"],
            ssh_port=row["ssh_port"], ssh_public_key=row["ssh_public_key"],
            ssh_public_key_fingerprint=row["ssh_public_key_fingerprint"], ssh_host_public_key=row["ssh_host_public_key"],
            ssh_host_fingerprint=row["ssh_host_fingerprint"], gpus=gpus, approval=row["approval"],
            created_at=_iso(row["registered_at"]), rejected_at=_iso(row["rejected_at"]),
            last_heartbeat_at=_iso(row["last_heartbeat_at"]),
            heartbeat_stale=(row["last_heartbeat_at"] is None
                             or self._clock() - row["last_heartbeat_at"] > self.heartbeat_ttl_seconds),
            desired_generation=row["desired_generation"], desired_action=row["desired_action"],
            desired_role=row["desired_role"], deployment_id=row["deployment_id"],
            deployment_sha256=row["deployment_sha256"], reverse_port=row["reverse_port"],
            observed_generation=row["observed_generation"], observed_state=row["observed_state"],
            observed_role=row["observed_role"], observed_error_code=row["observed_error_code"],
            observed_error_message=row["observed_error_message"], observed_receipt=receipt,
        )

    @staticmethod
    def _decode_account(value: str) -> UserAccount:
        try:
            data = json.loads(value)
            return UserAccount.model_validate(data, strict=True)
        except (TypeError, ValueError) as exc:
            raise FleetRegistryError(
                503, "registry_account_invalid", "Fleet node Unix account could not be decoded",
            ) from exc
