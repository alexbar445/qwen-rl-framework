"""Durable node enrollment, liveness, and site-scoped inventory."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import sqlite3
import struct
import tempfile
import threading
import time
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .contracts import EnrolmentRequest, NodeHeartbeat, NodeRegistration


_KEY_TYPES = {
    "ssh-ed25519",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
    "ssh-rsa",
    "sk-ssh-ed25519@openssh.com",
    "sk-ecdsa-sha2-nistp256@openssh.com",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HOST_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_SENSITIVE_HEALTH_KEYS = {"token", "bearer", "authorization", "secret", "password", "credential", "apikey", "privatekey"}
_SITE_WORKER_TOKEN_STATES = frozenset({"registered", "loading", "ready"})


class RegistryError(Exception):
    """An expected registry failure that the HTTP controller can report safely."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _utc_iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _canonical_gpu_uuid(value: str) -> str:
    """Normalize UUID spelling while preserving the complete device identity."""
    normalized = value.strip()
    if normalized[:4].casefold() == "gpu-":
        normalized = normalized[4:].strip()
    return normalized.casefold()


def _canonical_lan_host(value: str) -> str:
    """Normalize a registered Worker host for endpoint ownership checks."""
    try:
        return ipaddress.ip_address(value).compressed.casefold()
    except ValueError:
        return value.rstrip(".").casefold()


def _validated_worker_gpu_uuids(registration: NodeRegistration) -> set[str]:
    """Return a Worker's physical GPU claims, validating any reported inventory."""
    if registration.role != "worker":
        return set()
    if not registration.gpus:
        # Initial enrollment remains possible before NVIDIA driver installation.
        # The NodeAgent cannot report ready until an authenticated rejoin adds
        # the actual GPU inventory.
        return set()
    gpu_uuids: set[str] = set()
    gpu_indices: set[int] = set()
    for gpu in registration.gpus:
        uuid = _canonical_gpu_uuid(gpu.uuid)
        if not uuid or uuid in gpu_uuids:
            raise RegistryError(422, "invalid_gpu_capabilities", "Worker GPU UUIDs must be nonempty and unique")
        if gpu.index in gpu_indices:
            raise RegistryError(422, "invalid_gpu_capabilities", "Worker GPU indices must be unique within a node")
        gpu_uuids.add(uuid)
        gpu_indices.add(gpu.index)
    return gpu_uuids


def _worker_health_matches_gpu_claims(health: Any, gpus: Any) -> bool:
    """Bind a live Worker health identity to one of its registered GPU UUIDs."""
    if not isinstance(health, dict) or not isinstance(gpus, list) or not gpus:
        return False
    health_uuid = health.get("gpu_uuid")
    if not isinstance(health_uuid, str):
        return False
    claimed = {
        _canonical_gpu_uuid(item["uuid"])
        for item in gpus
        if isinstance(item, dict) and isinstance(item.get("uuid"), str)
    }
    actual = _canonical_gpu_uuid(health_uuid)
    return bool(actual and actual in claimed)


def _legacy_site_boot_unique(db_path: Path) -> bool:
    """Inspect existing schema, allowing SQLite crash recovery but no migration."""
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return False
    # Even a schema read needs write access to recover a hot rollback journal.
    uri = db_path.resolve().as_uri() + "?mode=rw"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes'"
            ).fetchone()
            if table is None:
                return False
            for index in connection.execute("PRAGMA index_list(nodes)").fetchall():
                if not index[2]:
                    continue
                columns = [row[2] for row in connection.execute(
                    f"PRAGMA index_info('{str(index[1]).replace(chr(39), chr(39) * 2)}')"
                ).fetchall()]
                if len(columns) == 2 and set(columns) == {"site_id", "boot_id"}:
                    return True
            return False
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        raise ValueError("registry schema cannot be inspected") from exc


def _check_same_boot_worker_ownership(
    connection: sqlite3.Connection,
    registration: NodeRegistration,
    claimed_gpu_uuids: set[str],
) -> None:
    """Allow co-resident Workers only when their GPU and HTTP endpoint claims differ."""
    if registration.role != "worker":
        return
    rows = connection.execute(
        """SELECT node_id, role, lan_host, worker_port, gpus_json
           FROM nodes WHERE site_id=? AND boot_id=? AND node_id!=? AND role='worker'""",
        (registration.site_id, registration.boot_id, registration.node_id),
    ).fetchall()
    endpoint = (_canonical_lan_host(registration.lan_host), registration.worker_port)
    for row in rows:
        try:
            raw_gpus = json.loads(row["gpus_json"])
            if (not isinstance(raw_gpus, list)
                    or any(not isinstance(item, dict) or not isinstance(item.get("uuid"), str)
                           for item in raw_gpus)):
                raise ValueError("malformed GPU list")
            existing_gpu_uuids = [_canonical_gpu_uuid(item["uuid"]) for item in raw_gpus]
            existing_gpu_indices = [item.get("index") for item in raw_gpus]
            if ("" in existing_gpu_uuids
                    or len(set(existing_gpu_uuids)) != len(existing_gpu_uuids)
                    or any(type(index) is not int or index < 0 for index in existing_gpu_indices)
                    or len(set(existing_gpu_indices)) != len(existing_gpu_indices)):
                raise ValueError("invalid GPU UUID or index inventory")
            existing_gpu_uuids = set(existing_gpu_uuids)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RegistryError(503, "registry_data_invalid", "registered Worker GPU claims are unreadable") from exc
        overlap = claimed_gpu_uuids & existing_gpu_uuids
        if overlap:
            raise RegistryError(
                409, "gpu_ownership_conflict",
                f"Worker GPU ownership overlaps node {row['node_id']} in this site and boot",
            )
        existing_endpoint = (_canonical_lan_host(row["lan_host"]), row["worker_port"])
        if endpoint == existing_endpoint:
            raise RegistryError(
                409, "worker_endpoint_conflict",
                f"Worker HTTP endpoint is already assigned to node {row['node_id']} in this site and boot",
            )


def _new_unique_token(connection: sqlite3.Connection, *, reserved_hashes: tuple[str, ...] = ()) -> str:
    while True:
        token = secrets.token_urlsafe(32)
        digest = _token_hash(token)
        if digest in reserved_hashes:
            continue
        exists = connection.execute(
            "SELECT 1 FROM nodes WHERE node_token_hash=? OR worker_token_hash=? LIMIT 1",
            (digest, digest),
        ).fetchone()
        if exists is None:
            return token


def _public_key(value: str, field: str) -> tuple[str, str]:
    """Validate and canonicalize a bare OpenSSH public key, dropping its comment."""
    parts = value.strip().split()
    if len(parts) < 2 or parts[0] not in _KEY_TYPES:
        raise RegistryError(422, "invalid_ssh_key", f"{field} must be a supported bare OpenSSH public key")
    algorithm, encoded = parts[0], parts[1]
    try:
        blob = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RegistryError(422, "invalid_ssh_key", f"{field} has invalid base64 key data") from exc
    if len(blob) < 4:
        raise RegistryError(422, "invalid_ssh_key", f"{field} has a truncated key blob")
    size = struct.unpack("!I", blob[:4])[0]
    if size != len(algorithm.encode("ascii")) or blob[4:4 + size] != algorithm.encode("ascii"):
        raise RegistryError(422, "invalid_ssh_key", f"{field} algorithm does not match its key blob")
    fields: list[bytes] = []
    offset = 4 + size
    while offset < len(blob):
        if len(blob) - offset < 4:
            raise RegistryError(422, "invalid_ssh_key", f"{field} has a truncated key field")
        field_size = struct.unpack("!I", blob[offset:offset + 4])[0]
        offset += 4
        if field_size > len(blob) - offset:
            raise RegistryError(422, "invalid_ssh_key", f"{field} has a truncated key field")
        fields.append(blob[offset:offset + field_size])
        offset += field_size
    if algorithm == "ssh-ed25519" and (len(fields) != 1 or len(fields[0]) != 32):
        raise RegistryError(422, "invalid_ssh_key", f"{field} does not contain a valid Ed25519 key")
    if algorithm.startswith("ecdsa-sha2-"):
        curve = algorithm.removeprefix("ecdsa-sha2-")
        expected_point_lengths = {"nistp256": 65, "nistp384": 97, "nistp521": 133}
        if (len(fields) != 2 or fields[0].decode("ascii", errors="ignore") != curve
                or len(fields[1]) != expected_point_lengths.get(curve, -1) or fields[1][:1] != b"\x04"):
            raise RegistryError(422, "invalid_ssh_key", f"{field} does not contain a valid ECDSA key")
    if algorithm == "ssh-rsa":
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise RegistryError(422, "invalid_ssh_key", f"{field} does not contain a valid RSA key")
        modulus = int.from_bytes(fields[1], "big", signed=False)
        if modulus.bit_length() < 2048:
            raise RegistryError(422, "invalid_ssh_key", f"{field} RSA modulus must be at least 2048 bits")
    if algorithm == "sk-ssh-ed25519@openssh.com" and (len(fields) != 2 or len(fields[0]) != 32 or not fields[1]):
        raise RegistryError(422, "invalid_ssh_key", f"{field} does not contain a valid security key")
    if algorithm == "sk-ecdsa-sha2-nistp256@openssh.com" and (
        len(fields) != 3 or fields[0] != b"nistp256" or len(fields[1]) != 65
        or fields[1][:1] != b"\x04" or not fields[2]
    ):
        raise RegistryError(422, "invalid_ssh_key", f"{field} does not contain a valid ECDSA security key")
    canonical = f"{algorithm} {base64.b64encode(blob).decode('ascii')}"
    fingerprint = hashlib.sha256(blob).hexdigest()
    return canonical, fingerprint


def _validate_lan_host(value: str) -> None:
    """Accept an IP literal or DNS-style host, never a URL or shell fragment."""
    if len(value) > 253 or not value or any(c in value for c in "/\\@?#[]%\r\n\x00"):
        raise RegistryError(422, "invalid_lan_host", "lan_host must be a hostname or IP address")
    try:
        ipaddress.ip_address(value)
        return
    except ValueError:
        pass
    hostname = value[:-1] if value.endswith(".") else value
    labels = hostname.split(".")
    if not hostname or any(not _HOST_LABEL.fullmatch(label) for label in labels):
        raise RegistryError(422, "invalid_lan_host", "lan_host must be a hostname or IP address")


def _valid_t2_worker_health(health: dict[str, Any]) -> bool:
    """Validate the strict T2 contract exposed by the existing Worker health route.

    The artifact and runtime validators are deliberately reused here: registry
    admission must apply the same two-bit format, layout, artifact identity, and
    runtime-format checks as Worker loading. This module is stdlib-only and does
    not initialize the CUDA/NInfer runtime.
    """
    try:
        from ..worker.native_artifact import (
            validate_ternary_runtime_quantization,
            validate_ternary_runtime_profile,
            validate_worker_quantization,
        )

        quantization = health.get("quantization")
        runtime_quantization = health.get("runtime_quantization")
        if not isinstance(quantization, dict) or not isinstance(runtime_quantization, dict):
            return False
        if health.get("quantization_mode") != "ternary":
            return False
        source = validate_worker_quantization("ternary", quantization)
        runtime = validate_ternary_runtime_quantization(runtime_quantization)
        validate_ternary_runtime_profile(health)
    except (ImportError, TypeError, ValueError):
        return False

    artifact_id = health.get("ninfer_artifact_id")
    artifact_sha = health.get("ninfer_artifact_sha256")
    model_vocab = health.get("logical_model_vocab_size")
    public_tokens = health.get("public_token_count")
    if (
        health.get("backend") != "ninfer"
        or health.get("is_loaded_in_4bit") is not False
        or type(health.get("base_trainable_parameters")) is not int
        or health.get("base_trainable_parameters") != 0
        or isinstance(health.get("quantized_tensor_count"), bool)
        or not isinstance(health.get("quantized_tensor_count"), int)
        or health.get("quantized_tensor_count") < 1
        or health.get("supports_full_token_logprobs") is not True
        or health.get("behavior_distribution") != "raw_policy_public_vocab"
        or isinstance(model_vocab, bool)
        or not isinstance(model_vocab, int)
        or model_vocab < 1
        or isinstance(public_tokens, bool)
        or not isinstance(public_tokens, int)
        or not 0 < public_tokens <= model_vocab
        or not isinstance(artifact_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", artifact_id) is None
        or not isinstance(artifact_sha, str)
        or not _SHA256.fullmatch(artifact_sha)
        or source.get("artifact_id") != artifact_id
        or source.get("artifact_sha256") != artifact_sha
        or health.get("base_fingerprint") != artifact_sha
        or health.get("artifact_fingerprint") != artifact_sha
    ):
        return False
    return runtime.get("representation") == "native_t2"


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _valid_nvfp4_worker_health(health: dict[str, Any]) -> bool:
    """Validate the normalized vLLM/TorchAO NVFP4 startup evidence."""
    source = health.get("quantization")
    runtime = health.get("runtime_quantization")
    if not isinstance(source, dict) or not isinstance(runtime, dict):
        return False
    skipped = source.get("modules_to_not_convert")
    if (not isinstance(skipped, list) or not skipped
            or any(not isinstance(name, str) or not name.strip() for name in skipped)
            or len(set(skipped)) != len(skipped)):
        return False
    source_mode = (
        source.get("mode") == "torchao"
        and source.get("bits") == 4
        and source.get("packed_resident") is True
        and source.get("compute_dtype") == "bfloat16"
        and source.get("quantizer") == "NVFP4WeightOnlyConfig"
        and _positive_int(source.get("packed_module_count"))
        and _positive_int(source.get("packed_uint8_tensor_count"))
        and _positive_int(source.get("packed_uint8_numel"))
    )
    runtime_mode = (
        runtime.get("mode") == "torchao"
        and runtime.get("runtime_method") == "torchao"
        and _positive_int(runtime.get("packed_module_count"))
        and _positive_int(runtime.get("packed_uint8_tensor_count"))
        and _positive_int(runtime.get("packed_uint8_numel"))
    )
    capacities = (
        health.get("max_model_len"), health.get("max_generated_tokens"),
        health.get("max_num_seqs"), health.get("max_loras"),
    )
    return (
        source_mode
        and runtime_mode
        and health.get("quantization_mode") == "torchao"
        and health.get("is_loaded_in_4bit") is True
        and _positive_int(health.get("module_count"))
        and health.get("base_trainable_parameters") == 0
        and type(health.get("base_trainable_parameters")) is int
        and health.get("supports_full_token_logprobs") is True
        and health.get("behavior_distribution") == "raw_policy"
        and all(_positive_int(value) for value in capacities)
        and health["max_generated_tokens"] <= health["max_model_len"]
    )


def _valid_worker_model_identity(worker_id: str, health: Any) -> bool:
    if not isinstance(health, dict):
        return False
    if health.get("role") != "worker" or health.get("worker_id") != worker_id:
        return False
    quantization = health.get("quantization")
    is_t2_claim = (
        health.get("base_artifact_kind") == "bonsai2_t2"
        or health.get("quantization_mode") == "ternary"
        or (isinstance(quantization, dict) and quantization.get("mode") == "ternary")
    )
    if is_t2_claim:
        # T2 is a native two-bit NInfer representation, never a 4-bit load.
        if health.get("is_loaded_in_4bit") is not False or not _valid_t2_worker_health(health):
            return False
    else:
        if health.get("backend") != "vllm" or not _valid_nvfp4_worker_health(health):
            return False
    artifact_fingerprint = health.get("artifact_fingerprint")
    base_fingerprint = health.get("base_fingerprint")
    if not isinstance(artifact_fingerprint, str) or not _SHA256.fullmatch(artifact_fingerprint):
        return False
    return isinstance(base_fingerprint, str) and bool(base_fingerprint.strip())


def _valid_worker_model_health(worker_id: str, health: Any) -> bool:
    return (isinstance(health, dict) and health.get("ready") is True
            and _valid_worker_model_identity(worker_id, health))


def _valid_worker_health(worker_id: str, health: Any) -> bool:
    if not _valid_worker_model_health(worker_id, health) or health.get("policy_ready") is not True:
        return False
    version = health.get("policy_version")
    if type(version) is not int or version < 0:
        return False
    adapter_hash = health.get("adapter_hash")
    if not isinstance(adapter_hash, str) or not _SHA256.fullmatch(adapter_hash):
        return False
    return True


def _redact_health(value: Any) -> Any:
    """Keep Worker health useful while preventing nested credential leakage."""
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            components = set(re.findall(r"[a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()))
            public_health_key = normalized in {"publictokencount", "supportsfulltokenlogprobs"}
            if (not public_health_key and
                    ((_SENSITIVE_HEALTH_KEYS & components) or "apikey" in normalized or "privatekey" in normalized)):
                continue
            safe[key] = _redact_health(item)
        return safe
    if isinstance(value, list):
        return [_redact_health(item) for item in value]
    return value


def _replace_secrets(value: Any, secrets_to_redact: list[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets_to_redact:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: _replace_secrets(item, secrets_to_redact) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_secrets(item, secrets_to_redact) for item in value]
    return value


def worker_health_is_valid(worker_id: str, health: Any) -> bool:
    """Public predicate used by the Controller and CPU acceptance tests."""
    return _valid_worker_health(worker_id, health)


def worker_health_is_model_valid(worker_id: str, health: Any) -> bool:
    """Public CPU-only predicate for a model-ready Worker health payload."""
    return _valid_worker_model_health(worker_id, health)


def worker_health_has_model_identity(worker_id: str, health: Any) -> bool:
    """Validate stable model identity without requiring Worker readiness."""
    return _valid_worker_model_identity(worker_id, health)


class Registry:
    """SQLite-backed registration and heartbeat store.

    `state_dir` is kept private (0700); the database contains node and Worker
    bearer tokens and is created with 0600 permissions. Registration and
    heartbeat writes use `BEGIN IMMEDIATE` so node limits and resource ownership
    remain correct across concurrent Controller processes.
    """

    def __init__(
        self,
        state_dir: str | Path,
        *,
        heartbeat_ttl_seconds: int = 45,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if type(heartbeat_ttl_seconds) is not int or heartbeat_ttl_seconds < 1:
            raise ValueError("heartbeat_ttl_seconds must be a positive integer")
        self.state_dir = Path(state_dir).expanduser().resolve()
        if self.state_dir == Path("/") or self.state_dir == Path("/etc") or Path("/etc") in self.state_dir.parents:
            raise ValueError("registry state_dir cannot be / or inside /etc")
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state_dir.chmod(0o700)
        self.db_path = self.state_dir / "registry.sqlite3"
        if self.db_path.is_symlink():
            raise ValueError("registry database path must not be a symlink")
        if _legacy_site_boot_unique(self.db_path):
            raise ValueError(
                "registry schema with unique(site_id, boot_id) requires a fresh state directory; "
                "automatic migration is not supported"
            )
        if not self.db_path.exists():
            descriptor = os.open(self.db_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        self.db_path.chmod(0o600)
        self.heartbeat_ttl_seconds = heartbeat_ttl_seconds
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
                    CREATE TABLE IF NOT EXISTS enrolments (
                        code_hash TEXT PRIMARY KEY,
                        site_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('gateway', 'worker')),
                        max_nodes INTEGER NOT NULL,
                        expires_at REAL NOT NULL,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS nodes (
                        node_id TEXT PRIMARY KEY,
                        site_id TEXT NOT NULL,
                        boot_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('gateway', 'worker')),
                        lan_host TEXT NOT NULL,
                        worker_port INTEGER NOT NULL,
                        ssh_port INTEGER NOT NULL,
                        ssh_public_key TEXT NOT NULL,
                        ssh_public_fingerprint TEXT NOT NULL UNIQUE,
                        ssh_host_public_key TEXT NOT NULL,
                        ssh_host_fingerprint TEXT NOT NULL UNIQUE,
                        gpus_json TEXT NOT NULL,
                        node_token TEXT NOT NULL,
                        node_token_hash TEXT NOT NULL UNIQUE,
                        worker_token TEXT NOT NULL,
                        worker_token_hash TEXT NOT NULL UNIQUE,
                        enrolment_code_hash TEXT NOT NULL REFERENCES enrolments(code_hash),
                        state TEXT NOT NULL DEFAULT 'registered',
                        worker_health_json TEXT,
                        image_digest TEXT,
                        error TEXT,
                        quarantine_reason TEXT,
                        registered_at REAL NOT NULL,
                        last_heartbeat_at REAL
                    );
                    CREATE INDEX IF NOT EXISTS nodes_site_idx ON nodes(site_id, node_id);
                    CREATE TABLE IF NOT EXISTS site_ssh (
                        site_id TEXT PRIMARY KEY,
                        remote_api_port INTEGER NOT NULL,
                        remote_ssh_port INTEGER NOT NULL
                    );
                    """
                )
                enrolment_columns = {
                    row["name"] for row in connection.execute("PRAGMA table_info(enrolments)").fetchall()
                }
                if "role" not in enrolment_columns:
                    raise ValueError("registry enrollment schema requires a fresh state directory with role-bound codes")
                self.db_path.chmod(0o600)
            finally:
                connection.close()

    def create_enrolment(self, request: EnrolmentRequest) -> dict[str, Any]:
        now = self._clock()
        code = secrets.token_urlsafe(32)
        code_hash = _token_hash(code)
        try:
            with self._transaction() as connection:
                connection.execute(
                    "INSERT INTO enrolments(code_hash, site_id, role, max_nodes, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (code_hash, request.site_id, request.role, request.max_nodes, now + request.ttl_seconds, now),
                )
        except sqlite3.IntegrityError as exc:
            raise RegistryError(409, "enrolment_conflict", "could not create enrollment") from exc
        return {
            "site_id": request.site_id,
            "role": request.role,
            "code": code,
            "max_nodes": request.max_nodes,
            "expires_at": _utc_iso(now + request.ttl_seconds),
        }

    def register_node(
        self,
        registration: NodeRegistration,
        *,
        node_token: str | None = None,
        ssh_remote_ports: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        now = self._clock()
        _validate_lan_host(registration.lan_host)
        claimed_gpu_uuids = _validated_worker_gpu_uuids(registration)
        public_key, public_fingerprint = _public_key(registration.ssh_public_key, "ssh_public_key")
        host_key, host_fingerprint = _public_key(registration.ssh_host_public_key, "ssh_host_public_key")
        try:
            gpus_json = _json_dump([gpu.model_dump(mode="json") for gpu in registration.gpus])
        except (TypeError, ValueError) as exc:
            raise RegistryError(422, "invalid_gpu_capabilities", "GPU capabilities are not valid JSON") from exc

        with self._transaction() as connection:
            existing = connection.execute("SELECT * FROM nodes WHERE node_id = ?", (registration.node_id,)).fetchone()
            if existing is not None:
                if existing["site_id"] != registration.site_id or existing["role"] != registration.role:
                    raise RegistryError(409, "node_identity_conflict", "a node cannot change its registered site or role")
                authenticated = node_token is not None and hmac.compare_digest(
                    _token_hash(node_token), existing["node_token_hash"]
                )
                if not authenticated:
                    # Enrollment codes are also the recovery path if the first
                    # registration response was lost before the node persisted
                    # its bearer tokens. Recovery must prove the original SSH
                    # public identities and must use an unexpired site code.
                    code_hash = _token_hash(registration.code)
                    enrolment = connection.execute(
                        "SELECT * FROM enrolments WHERE code_hash=?", (code_hash,)
                    ).fetchone()
                    recovery_allowed = (
                        enrolment is not None
                        and enrolment["site_id"] == registration.site_id
                        and enrolment["role"] == registration.role
                        and now < enrolment["expires_at"]
                        and existing["ssh_public_fingerprint"] == public_fingerprint
                        and existing["ssh_host_fingerprint"] == host_fingerprint
                    )
                    if not recovery_allowed:
                        raise RegistryError(409, "node_already_registered", "node_id is already registered; use its node token or a valid site enrollment code with the same SSH keys")
                if registration.role == "gateway":
                    other_gateway = connection.execute(
                        "SELECT node_id FROM nodes WHERE site_id=? AND role='gateway' AND node_id!=? LIMIT 1",
                        (registration.site_id, registration.node_id),
                    ).fetchone()
                    if other_gateway is not None:
                        raise RegistryError(409, "gateway_already_registered", "site already has a registered Gateway; rejoin that node instead")
                _check_same_boot_worker_ownership(connection, registration, claimed_gpu_uuids)
                try:
                    connection.execute(
                        """UPDATE nodes SET boot_id=?, lan_host=?, worker_port=?, ssh_port=?,
                           ssh_public_key=?, ssh_public_fingerprint=?, ssh_host_public_key=?, ssh_host_fingerprint=?,
                           gpus_json=?, state='registered', worker_health_json=NULL, image_digest=NULL,
                           error=NULL, quarantine_reason=NULL, last_heartbeat_at=NULL
                           WHERE node_id=?""",
                        (
                            registration.boot_id, registration.lan_host, registration.worker_port, registration.ssh_port,
                            public_key, public_fingerprint, host_key, host_fingerprint, gpus_json, registration.node_id,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    detail = str(exc).lower()
                    if "ssh_public_fingerprint" in detail:
                        raise RegistryError(409, "ssh_key_conflict", "ssh_public_key is already assigned to another node") from exc
                    if "ssh_host_fingerprint" in detail:
                        raise RegistryError(409, "ssh_host_key_conflict", "ssh_host_public_key is already assigned to another node") from exc
                    raise RegistryError(409, "node_identity_conflict", "node identity conflicts with an existing registration") from exc
                row = connection.execute("SELECT * FROM nodes WHERE node_id = ?", (registration.node_id,)).fetchone()
                result = self._registration_result(row)
            else:
                code_hash = _token_hash(registration.code)
                enrolment = connection.execute("SELECT * FROM enrolments WHERE code_hash = ?", (code_hash,)).fetchone()
                if enrolment is None or enrolment["site_id"] != registration.site_id:
                    raise RegistryError(403, "invalid_enrolment_code", "a valid site enrollment code is required")
                if enrolment["role"] != registration.role:
                    raise RegistryError(403, "enrolment_role_mismatch", "the enrollment code is for a different node role")
                if now >= enrolment["expires_at"]:
                    raise RegistryError(410, "enrolment_expired", "the enrollment code has expired")
                if registration.role == "gateway":
                    other_gateway = connection.execute(
                        "SELECT node_id FROM nodes WHERE site_id=? AND role='gateway' LIMIT 1",
                        (registration.site_id,),
                    ).fetchone()
                    if other_gateway is not None:
                        raise RegistryError(409, "gateway_already_registered", "site already has a registered Gateway; rejoin that node instead")
                count = connection.execute(
                    "SELECT COUNT(*) FROM nodes WHERE enrolment_code_hash = ?", (code_hash,)
                ).fetchone()[0]
                if count >= enrolment["max_nodes"]:
                    raise RegistryError(409, "enrolment_full", "the enrollment code reached its node limit")
                _check_same_boot_worker_ownership(connection, registration, claimed_gpu_uuids)
                node_secret = _new_unique_token(connection)
                worker_secret = _new_unique_token(connection, reserved_hashes=(_token_hash(node_secret),))
                try:
                    connection.execute(
                        """INSERT INTO nodes(
                           node_id,site_id,boot_id,role,lan_host,worker_port,ssh_port,
                           ssh_public_key,ssh_public_fingerprint,ssh_host_public_key,ssh_host_fingerprint,
                           gpus_json,node_token,node_token_hash,worker_token,worker_token_hash,enrolment_code_hash,
                           registered_at
                           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            registration.node_id, registration.site_id, registration.boot_id, registration.role,
                            registration.lan_host, registration.worker_port, registration.ssh_port,
                            public_key, public_fingerprint, host_key, host_fingerprint, gpus_json,
                            node_secret, _token_hash(node_secret), worker_secret,
                            _token_hash(worker_secret), code_hash, now,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    detail = str(exc).lower()
                    if "ssh_public_fingerprint" in detail:
                        raise RegistryError(409, "ssh_key_conflict", "ssh_public_key is already assigned to another node") from exc
                    if "ssh_host_fingerprint" in detail:
                        raise RegistryError(409, "ssh_host_key_conflict", "ssh_host_public_key is already assigned to another node") from exc
                    raise RegistryError(409, "node_identity_conflict", "node identity conflicts with an existing registration") from exc
                row = connection.execute("SELECT * FROM nodes WHERE node_id = ?", (registration.node_id,)).fetchone()
                result = self._registration_result(row)

        self.write_site_files(registration.site_id, remote_listen_ports=ssh_remote_ports)
        return result

    def heartbeat(self, node_id: str, heartbeat: NodeHeartbeat, *, node_token: str) -> dict[str, Any]:
        now = self._clock()
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
            if row is None or not hmac.compare_digest(_token_hash(node_token), row["node_token_hash"]):
                raise RegistryError(401, "invalid_node_token", "a valid node bearer token is required")
            if heartbeat.boot_id != row["boot_id"]:
                raise RegistryError(409, "boot_id_mismatch", "re-register this node before sending heartbeats for a new boot_id")
            if row["state"] == "quarantined" and heartbeat.state != "quarantined":
                raise RegistryError(409, "rejoin_required", "a quarantined node must re-register with its node token before becoming active")
            if heartbeat.state == "ready" and row["role"] == "worker":
                try:
                    registered_gpus = json.loads(row["gpus_json"])
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RegistryError(503, "registry_data_invalid", "registered Worker GPU inventory is unreadable") from exc
                if not isinstance(registered_gpus, list) or not registered_gpus:
                    raise RegistryError(422, "invalid_worker_health", "ready Worker heartbeat requires registered GPU capability proof")
                if not _valid_worker_health(node_id, heartbeat.worker_health):
                    raise RegistryError(422, "invalid_worker_health", "ready Worker heartbeat requires matching ready policy and artifact proof")
                if not _worker_health_matches_gpu_claims(heartbeat.worker_health, registered_gpus):
                    raise RegistryError(422, "invalid_worker_health", "ready Worker health must identify one of its registered GPUs")
            secrets_to_redact: list[str] = []
            if heartbeat.worker_health is not None or heartbeat.error is not None or heartbeat.image_digest is not None:
                credentials = connection.execute("SELECT node_token, worker_token FROM nodes").fetchall()
                secrets_to_redact = [
                    token
                    for credential in credentials
                    for token in (credential["node_token"], credential["worker_token"])
                    if token
                ]
            if heartbeat.worker_health is not None:
                try:
                    safe_health = _replace_secrets(_redact_health(heartbeat.worker_health), secrets_to_redact)
                    health_json = _json_dump(safe_health)
                except (TypeError, ValueError) as exc:
                    raise RegistryError(422, "invalid_worker_health", "worker_health must contain finite JSON values") from exc
            else:
                health_json = None
            if heartbeat.image_digest is not None and (len(heartbeat.image_digest) > 256 or any(c in heartbeat.image_digest for c in "\r\n\x00")):
                raise RegistryError(422, "invalid_image_digest", "image_digest must be a single-line value of at most 256 characters")
            error = heartbeat.error
            if error is not None:
                error = _replace_secrets(error.replace("\x00", "")[:2000], secrets_to_redact)
            image_digest = _replace_secrets(heartbeat.image_digest, secrets_to_redact)
            quarantine_reason = None
            if heartbeat.state == "quarantined":
                quarantine_reason = error or "node_reported_quarantine"
            connection.execute(
                """UPDATE nodes SET state=?, worker_health_json=?, image_digest=?, error=?,
                   quarantine_reason=?, last_heartbeat_at=? WHERE node_id=?""",
                (heartbeat.state, health_json, image_digest, error, quarantine_reason, now, node_id),
            )
            updated = connection.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
            return self._node_view(updated, now)

    def authenticate_node(self, node_token: str) -> dict[str, str] | None:
        now = self._clock()
        self._quarantine_stale_nodes(now)
        digest = _token_hash(node_token)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT node_id, site_id, role, state, node_token_hash FROM nodes WHERE node_token_hash = ?", (digest,)
            ).fetchone()
        if (row is None or row["state"] == "quarantined"
                or not hmac.compare_digest(digest, row["node_token_hash"])):
            return None
        return {"node_id": row["node_id"], "site_id": row["site_id"], "role": row["role"]}

    def gateway_site(self, node_token: str) -> str | None:
        """Return the authenticated Gateway's site, or None for other identities."""
        principal = self.authenticate_node(node_token)
        if principal is None or principal["role"] != "gateway":
            return None
        return principal["site_id"]

    def site_gateway(self, site_id: str) -> dict[str, Any] | None:
        """Return the sole Gateway record and its service bearer for admin export."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM nodes WHERE site_id = ? AND role = 'gateway' ORDER BY node_id",
                (site_id,),
            ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise RegistryError(409, "multiple_site_gateways", "site has more than one registered Gateway")
        record = self._node_view(rows[0], self._clock())
        record["worker_token"] = rows[0]["worker_token"]
        return record

    def list_nodes(self) -> list[dict[str, Any]]:
        now = self._clock()
        self._quarantine_stale_nodes(now)
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM nodes ORDER BY site_id, node_id").fetchall()
        return [self._node_view(row, now) for row in rows]

    def list_site_nodes(self, site_id: str) -> list[dict[str, Any]]:
        now = self._clock()
        self._quarantine_stale_nodes(now, site_id=site_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM nodes WHERE site_id = ? ORDER BY node_id", (site_id,)
            ).fetchall()
        result = []
        for row in rows:
            record = self._node_view(row, now)
            if row["role"] == "worker" and row["state"] in _SITE_WORKER_TOKEN_STATES:
                record["worker_token"] = row["worker_token"]
            result.append(record)
        return result

    def write_site_files(
        self,
        site_id: str,
        *,
        remote_listen_ports: tuple[int, int] | None = None,
    ) -> dict[str, str]:
        """Atomically emit private, site-scoped SSH inventory for T4.

        Only Gateway keys are authorized to open a reverse management tunnel.
        The generated key options deny sessions and constrain remote listeners
        to loopback; this method never writes to system SSH paths or starts SSH.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", site_id):
            raise ValueError("invalid site_id")
        if remote_listen_ports is not None and (
            len(remote_listen_ports) != 2
            or any(type(port) is not int or not 1 <= port <= 65535 for port in remote_listen_ports)
            or remote_listen_ports[0] == remote_listen_ports[1]
        ):
            raise ValueError("remote_listen_ports must contain two distinct valid TCP ports")
        if remote_listen_ports is not None:
            with self._transaction() as connection:
                connection.execute(
                    """INSERT INTO site_ssh(site_id,remote_api_port,remote_ssh_port) VALUES(?,?,?)
                       ON CONFLICT(site_id) DO UPDATE SET remote_api_port=excluded.remote_api_port,
                       remote_ssh_port=excluded.remote_ssh_port""",
                    (site_id, remote_listen_ports[0], remote_listen_ports[1]),
                )
        else:
            with closing(self._connect()) as connection:
                configured = connection.execute(
                    "SELECT remote_api_port, remote_ssh_port FROM site_ssh WHERE site_id=?",
                    (site_id,),
                ).fetchone()
                if configured is not None:
                    remote_listen_ports = (configured["remote_api_port"], configured["remote_ssh_port"])
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM nodes WHERE site_id = ? ORDER BY node_id", (site_id,)).fetchall()
        ssh_root = self.state_dir / "ssh"
        sites_root = ssh_root / "sites"
        directory = sites_root / site_id
        for parent in (ssh_root, sites_root, directory):
            if parent.is_symlink():
                raise ValueError("SSH export directories must not be symlinks")
            parent.mkdir(mode=0o700, exist_ok=True)
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError("SSH export path must contain only real directories")
            parent.chmod(0o700)
        inventory = []
        authorized = []
        known_hosts = []
        for row in rows:
            inventory.append({
                "node_id": row["node_id"],
                "site_id": row["site_id"],
                "boot_id": row["boot_id"],
                "role": row["role"],
                "lan_host": row["lan_host"],
                "worker_port": row["worker_port"],
                "ssh_port": row["ssh_port"],
                "ssh_host_public_key": row["ssh_host_public_key"],
                "gpus": json.loads(row["gpus_json"]),
            })
            if row["role"] == "gateway":
                options = ['restrict', 'command="/usr/bin/false"']
                if remote_listen_ports is not None:
                    options.extend([
                        "port-forwarding",
                        f'permitlisten="127.0.0.1:{remote_listen_ports[0]}"',
                        f'permitlisten="127.0.0.1:{remote_listen_ports[1]}"',
                    ])
                authorized.append(f'{",".join(options)} {row["ssh_public_key"]} node={row["node_id"]}')
            known_hosts.append(f'{row["node_id"]} {row["ssh_host_public_key"]} node={row["node_id"]}')
        files = {
            "authorized_keys": "\n".join(authorized) + ("\n" if authorized else ""),
            "known_hosts": "\n".join(known_hosts) + ("\n" if known_hosts else ""),
            "inventory.json": json.dumps(inventory, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            "central-relay-sshd_config.fragment": (
                "# Include only in the dedicated cluster relay sshd configuration.\n"
                "AllowTcpForwarding remote\n"
                "PermitOpen none\n"
                "GatewayPorts no\n"
            ),
        }
        for name, content in files.items():
            _atomic_private_write(directory / name, content.encode("utf-8"))
        return {name: str(directory / name) for name in files}

    def _quarantine_stale_nodes(self, now: float, *, site_id: str | None = None) -> None:
        cutoff = now - self.heartbeat_ttl_seconds
        with closing(self._connect()) as connection:
            if site_id is None:
                stale = connection.execute(
                    "SELECT 1 FROM nodes WHERE state != 'quarantined' AND last_heartbeat_at IS NOT NULL AND last_heartbeat_at < ? LIMIT 1",
                    (cutoff,),
                ).fetchone()
            else:
                stale = connection.execute(
                    "SELECT 1 FROM nodes WHERE site_id=? AND state != 'quarantined' AND last_heartbeat_at IS NOT NULL AND last_heartbeat_at < ? LIMIT 1",
                    (site_id, cutoff),
                ).fetchone()
        if stale is None:
            return
        with self._transaction() as connection:
            if site_id is None:
                connection.execute(
                    """UPDATE nodes SET state='quarantined', quarantine_reason='heartbeat_timeout'
                       WHERE state != 'quarantined' AND last_heartbeat_at IS NOT NULL AND last_heartbeat_at < ?""",
                    (cutoff,),
                )
            else:
                connection.execute(
                    """UPDATE nodes SET state='quarantined', quarantine_reason='heartbeat_timeout'
                       WHERE site_id=? AND state != 'quarantined' AND last_heartbeat_at IS NOT NULL AND last_heartbeat_at < ?""",
                    (site_id, cutoff),
                )

    def _node_view(self, row: sqlite3.Row, now: float) -> dict[str, Any]:
        health = json.loads(row["worker_health_json"]) if row["worker_health_json"] else None
        gpus = json.loads(row["gpus_json"])
        has_gpu_claim = (
            row["role"] == "worker"
            and _worker_health_matches_gpu_claims(health, gpus)
        )
        last = row["last_heartbeat_at"]
        age = max(0.0, now - last) if last is not None else None
        fresh = age is not None and age <= self.heartbeat_ttl_seconds
        eligible = bool(
            row["role"] == "worker"
            and has_gpu_claim
            and row["state"] == "ready"
            and fresh
            and _valid_worker_health(row["node_id"], health)
        )
        model_ready = bool(
            row["role"] == "worker"
            and has_gpu_claim
            and row["state"] in {"loading", "ready"}
            and fresh
            and _valid_worker_model_health(row["node_id"], health)
        )
        return {
            "node_id": row["node_id"],
            "site_id": row["site_id"],
            "boot_id": row["boot_id"],
            "role": row["role"],
            "lan_host": row["lan_host"],
            "worker_port": row["worker_port"],
            "ssh_port": row["ssh_port"],
            "ssh_public_key": row["ssh_public_key"],
            "ssh_host_public_key": row["ssh_host_public_key"],
            "gpus": gpus,
            "state": row["state"],
            "worker_health": health,
            "image_digest": row["image_digest"],
            "error": row["error"],
            "quarantine_reason": row["quarantine_reason"],
            "registered_at": _utc_iso(row["registered_at"]),
            "last_heartbeat_at": _utc_iso(last),
            "heartbeat_age_seconds": age,
            "heartbeat_fresh": fresh,
            "model_ready": model_ready,
            "eligible": eligible,
        }

    def _registration_result(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "node": self._node_view(row, self._clock()),
            "node_token": row["node_token"],
            "worker_token": row["worker_token"],
        }


def _atomic_private_write(path: Path, content: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
        path.chmod(0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise
