"""Immutable policy mirroring and content-addressed model artifacts for LAN sites.

The mirror reads policy payloads from the central Learner and stores them by
manifest identity and size. Generic model artifacts keep the content-addressed
SHA routes exposed by this module.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import ipaddress
import json
import mmap
import os
import re
import stat
import tempfile
import threading
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse

from qwen_rl.contracts import worker as wire
from qwen_rl.transport.tensor_codec import (
    MAX_TENSOR_RAW_BYTES,
    TENSOR_PAYLOAD_CONTENT_TYPE,
    TENSOR_PAYLOAD_ENVELOPE,
    inspect_tensor_payload,
    max_encoded_size,
)

POLICY_FILES = wire.ADAPTER_FILES | wire.NATIVE_ADAPTER_FILES
_MAX_POLICY_JSON_BYTES = 16 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LINEAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_POLICY_FIELDS = frozenset({"version", "base_fingerprint", "artifact_fingerprint", "quantization", "files", "source_adapter_sha256", "artifact_id"})
_CHUNK_SIZE = 1024 * 1024
_MANIFEST_SCHEMA = 1


class ArtifactMirrorError(RuntimeError):
    """Base class for an invalid, unavailable, or corrupt artifact."""


class ArtifactNotFound(ArtifactMirrorError):
    """The requested immutable artifact is not present."""


class ArtifactUpstreamError(ArtifactMirrorError):
    """The configured central origin did not return a usable response."""


class ArtifactIntegrityError(ArtifactMirrorError):
    """Bytes did not match the immutable SHA/size contract."""


class PolicyVersionUnavailable(ArtifactNotFound):
    """A requested policy version is not cached or published upstream."""


class ImmutablePolicyConflict(ArtifactMirrorError):
    """One lineage/version was presented with two different manifests."""


def _inspect_tensor_file(path: Path, *, max_raw_bytes: int, safetensors: bool) -> int:
    """Validate framed sizes and SafeTensors header without decompressing or copying the payload."""
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError("cached policy tensor is not a regular file")
    if path.stat().st_size < TENSOR_PAYLOAD_ENVELOPE.size:
        raise ArtifactIntegrityError("policy tensor payload is shorter than its envelope")
    with path.open("rb") as source, mmap.mmap(
        source.fileno(), 0, access=mmap.ACCESS_READ,
    ) as mapped:
        view = memoryview(mapped)
        try:
            return inspect_tensor_payload(
                view,
                max_raw_bytes=max_raw_bytes,
                safetensors=safetensors,
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("policy tensor frame is invalid") from exc
        finally:
            view.release()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.lower()):
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def policy_file_cache_key(
    policy: Mapping[str, Any], filename: str, *, origin: str | None = None,
) -> str:
    """Key a policy file by its small immutable manifest identity, never its bytes."""
    if not isinstance(policy, Mapping):
        raise ValueError("policy metadata must be a mapping")
    version = policy.get("version")
    source_adapter = policy.get("source_adapter_sha256")
    if (isinstance(version, bool) or not isinstance(version, int) or version < 0
            or not isinstance(source_adapter, str) or not _SHA256_RE.fullmatch(source_adapter)):
        raise ValueError("policy metadata omitted its version or adapter identity")
    quantization = policy.get("quantization")
    if not isinstance(quantization, Mapping):
        raise ValueError("policy metadata omitted its quantization contract")
    mode = quantization.get("mode")
    if not isinstance(mode, str) or mode not in {"torchao", "ternary"}:
        raise ValueError("unsupported policy quantization")
    artifact_id = policy.get("artifact_id")
    if quantization.get("mode") == "ternary":
        if not isinstance(artifact_id, str) or re.fullmatch(r"[0-9a-f]{32}", artifact_id) is None:
            raise ValueError("native policy metadata omitted its NInfer artifact ID")
    elif artifact_id is not None:
        raise ValueError("non-native policy metadata cannot carry a native artifact ID")
    files = policy.get("files")
    if not isinstance(files, list):
        raise ValueError("policy metadata omitted its file inventory")
    selected = next((item for item in files if isinstance(item, Mapping)
                     and item.get("name") == filename), None)
    if selected is None:
        raise ValueError("policy metadata has no selected file")
    size = selected.get("size_bytes")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
        raise ValueError("policy file size must be nonnegative or absent")
    normalized_origin = None
    if origin is not None:
        parts = urlsplit(origin)
        if (parts.scheme.lower() not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or parts.path not in ("", "/")):
            raise ValueError("policy cache origin must be an HTTP(S) origin without credentials or path")
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
        normalized_origin = (parts.scheme.lower(), parts.hostname.lower(), port)
    identity = {
        "schema": "qwen-rl-policy-object-key/v2",
        "transport": "LORABLS2" if not filename.endswith(".json") else "identity",
        "origin": normalized_origin,
        "version": version,
        "source_adapter_sha256": source_adapter,
        "artifact_id": artifact_id,
        "base_fingerprint": policy.get("base_fingerprint"),
        "artifact_fingerprint": policy.get("artifact_fingerprint"),
        "quantization": policy.get("quantization"),
        "filename": filename,
        "size_bytes": size,
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _is_private_or_local(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        name = host.rstrip(".").lower()
        return "." not in name or name.endswith((".local", ".lan", ".cluster.local", ".svc"))


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _canonical_json(payload) + b"\n"
    with lock:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(line)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class ArtifactRecord:
    sha256: str
    size: int
    path: Path

    def as_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size": self.size, "path": str(self.path)}


class ContentAddressedStore:
    """Atomic SHA-256 blob store shared by policy and model files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.blob_root = self.root / "blobs" / "sha256"
        self._receipt_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._verified_signatures: dict[str, tuple[int, int, int, int, int]] = {}
        self._blob_locks: dict[str, threading.Lock] = {}

    def path_for(self, sha256: str) -> Path:
        digest = _validate_sha256(sha256)
        return self.blob_root / digest[:2] / digest

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int, int, int]:
        info = path.lstat()
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _blob_lock(self, digest: str) -> threading.Lock:
        with self._state_lock:
            return self._blob_locks.setdefault(digest, threading.Lock())

    def remember_verified(self, sha256: str, path: str | Path, *, expected_size: int | None = None) -> None:
        digest = _validate_sha256(sha256)
        path = Path(path)
        with self._blob_lock(digest):
            signature = self._signature(path)
            if path.is_symlink() or not path.is_file():
                raise ArtifactIntegrityError("content-addressed object is not a regular file")
            if expected_size is not None and signature[2] != expected_size:
                raise ArtifactIntegrityError("verified artifact size does not match its receipt")
            with self._state_lock:
                self._verified_signatures[digest] = signature

    def resolve(self, sha256: str, *, expected_size: int | None = None) -> ArtifactRecord:
        digest = _validate_sha256(sha256)
        path = self.path_for(digest)
        with self._blob_lock(digest):
            try:
                signature = self._signature(path)
            except FileNotFoundError as exc:
                raise ArtifactNotFound(f"artifact SHA-256 {digest} is not cached") from exc
            if path.is_symlink() or not path.is_file():
                raise ArtifactIntegrityError("content-addressed object is not a regular file")
            observed_size = signature[2]
            if expected_size is not None and observed_size != expected_size:
                raise ArtifactIntegrityError("cached artifact size does not match its receipt")
            with self._state_lock:
                already_verified = self._verified_signatures.get(digest) == signature
            if not already_verified:
                observed_sha, hashed_size = _hash_file(path)
                if not hmac.compare_digest(observed_sha, digest):
                    raise ArtifactIntegrityError("cached artifact SHA-256 does not match its content address")
                if hashed_size != observed_size or self._signature(path) != signature:
                    raise ArtifactIntegrityError("cached artifact changed while it was being verified")
                with self._state_lock:
                    self._verified_signatures[digest] = signature
            return ArtifactRecord(digest, observed_size, path)

    def publish_file(self, source_path: str | Path, *, expected_sha256: str | None = None) -> ArtifactRecord:
        """Hash and atomically publish one local model/artifact file."""
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        expected = _validate_sha256(expected_sha256) if expected_sha256 is not None else None
        digest = hashlib.sha256()
        size = 0
        destination: Path | None = None
        temporary: Path | None = None
        try:
            with source_path.open("rb") as source:
                first_chunk = source.read(_CHUNK_SIZE)
                digest.update(first_chunk)
                size += len(first_chunk)
                sha_path = self.blob_root / "00" / f".staging.{uuid4().hex}.tmp"
                # The final prefix directory is not known until hashing finishes. Keep the
                # staging file on the same filesystem so the eventual replace is atomic.
                sha_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = sha_path
                with temporary.open("xb") as staged:
                    staged.write(first_chunk)
                    while chunk := source.read(_CHUNK_SIZE):
                        digest.update(chunk)
                        size += len(chunk)
                        staged.write(chunk)
                    staged.flush()
                    os.fsync(staged.fileno())
            actual = digest.hexdigest()
            if expected is not None and not hmac.compare_digest(actual, expected):
                raise ArtifactIntegrityError("local artifact does not match the requested SHA-256")
            destination = self.path_for(actual)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                try:
                    existing = self.resolve(actual, expected_size=size)
                except ArtifactIntegrityError:
                    os.chmod(temporary, 0o444)
                    os.replace(temporary, destination)
                    temporary = None
                else:
                    temporary.unlink(missing_ok=True)
                    temporary = None
                    return existing
            else:
                os.chmod(temporary, 0o444)
                os.replace(temporary, destination)
                temporary = None
            _fsync_dir(destination.parent)
            self.remember_verified(actual, destination, expected_size=size)
            _append_jsonl(self.root / "published-artifacts.jsonl", {
                "kind": "local_publish", "sha256": actual, "size": size,
                "source_name": source_path.name, "published_at": _utc_now(),
            }, self._receipt_lock)
            return ArtifactRecord(actual, size, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def publish_artifact(source_path: str | Path, cache_dir: str | Path,
                     *, expected_sha256: str | None = None) -> ArtifactRecord:
    """Convenience entry point for the controller's artifact-publish command."""
    return ContentAddressedStore(cache_dir).publish_file(source_path, expected_sha256=expected_sha256)


def _safe_model_relative_path(relative: str) -> str:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("model artifact file names must be nonempty POSIX relative paths")
    path = PurePosixPath(relative)
    if not path.parts or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("model artifact file path escapes the model root")
    return path.as_posix()






class ArtifactMirror:
    """Mirror immutable Learner policy versions and generic content-addressed artifacts.

    `lineage_id` names the immutable manifest namespace. The default is the
    explicit stable namespace ``default``; callers with more than one training
    lineage must pass its actual ID.
    """

    def __init__(
        self,
        upstream_url: str,
        cache_dir: str | Path,
        headers: Mapping[str, str] | None = None,
        *,
        lineage_id: str = "default",
        client: httpx.AsyncClient | None = None,
        request_timeout: float = 120.0,
        latest_ttl_seconds: float = 0.5,
    ) -> None:
        if not isinstance(lineage_id, str) or not _LINEAGE_RE.fullmatch(lineage_id):
            raise ValueError("lineage_id must be a stable path-safe identifier")
        parts = urlsplit(upstream_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None
                or parts.password is not None or parts.query or parts.fragment):
            raise ValueError("upstream_url must be an HTTP(S) origin without credentials, query, or fragment")
        if request_timeout <= 0 or latest_ttl_seconds < 0:
            raise ValueError("request timeout must be positive and latest TTL cannot be negative")
        self.upstream_url = upstream_url.rstrip("/")
        self._upstream_origin = (parts.scheme.lower(), parts.hostname.lower(), parts.port or (443 if parts.scheme == "https" else 80))
        self.cache_dir = Path(cache_dir)
        self.lineage_id = lineage_id
        self.store = ContentAddressedStore(self.cache_dir)
        self.headers = dict(headers or {})
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=request_timeout, follow_redirects=False)
        self._request_timeout = request_timeout
        self._latest_ttl_seconds = latest_ttl_seconds
        self._latest_version: int | None = None
        self._latest_checked_at = 0.0
        self._latest_lock = asyncio.Lock()
        self._locks_guard = asyncio.Lock()
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._receipt_lock = threading.Lock()
        self._manifest_lock = threading.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _key_lock(self, key: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._key_locks.setdefault(key, asyncio.Lock())

    def _manifest_index_path(self, version: int) -> Path:
        return self.cache_dir / "manifests" / self.lineage_id / "versions" / f"v{version}.json"

    def _manifest_record_path(self, policy_sha256: str) -> Path:
        return self.cache_dir / "manifests" / self.lineage_id / "sha256" / f"{policy_sha256}.json"

    def _read_record(self, version: int) -> dict[str, Any] | None:
        index_path = self._manifest_index_path(version)
        if not index_path.is_file():
            return None
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("cached policy version index is unreadable") from exc
        policy_sha = index.get("policy_sha256") if isinstance(index, dict) else None
        if (not isinstance(index, dict) or index.get("schema") != _MANIFEST_SCHEMA
                or index.get("lineage_id") != self.lineage_id or index.get("version") != version
                or not isinstance(policy_sha, str) or not _SHA256_RE.fullmatch(policy_sha)):
            raise ArtifactIntegrityError("cached policy version index has an invalid schema or namespace")
        path = self._manifest_record_path(policy_sha)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("cached SHA-addressed policy manifest is unreadable") from exc
        if (not isinstance(record, dict) or record.get("schema") != _MANIFEST_SCHEMA
                or record.get("lineage_id") != self.lineage_id or record.get("version") != version
                or record.get("policy_sha256") != policy_sha
                or not isinstance(record.get("policy"), dict)
                or not isinstance(record.get("files"), dict)):
            raise ArtifactIntegrityError("cached SHA-addressed policy manifest has an invalid namespace")
        observed = hashlib.sha256(_canonical_json(record["policy"])).hexdigest()
        if not hmac.compare_digest(observed, str(record.get("policy_sha256", ""))):
            raise ArtifactIntegrityError("cached policy manifest digest does not match its metadata")
        return record

    @staticmethod
    def _validate_manifest(payload: Any, requested_version: int | None) -> tuple[int, dict[str, Any]]:
        try:
            manifest = wire.PolicyManifest.model_validate(payload)
        except Exception as exc:
            raise ArtifactUpstreamError("policy response did not match the Worker manifest contract") from exc
        mode = manifest.quantization.get("mode")
        if not isinstance(mode, str) or mode not in {"torchao", "ternary"}:
            raise ArtifactUpstreamError("unsupported policy quantization")
        version = manifest.version
        if requested_version is not None and version != requested_version:
            raise PolicyVersionUnavailable("origin returned a different policy version")
        expected_names = (wire.NATIVE_ADAPTER_FILES
                          if manifest.quantization.get("mode") == "ternary"
                          else wire.ADAPTER_FILES)
        normalized = manifest.model_dump(mode="json")
        entries = normalized.get("files")
        if (not isinstance(entries, list) or len(entries) != len(expected_names)
                or {entry.get("name") for entry in entries if isinstance(entry, dict)} != expected_names):
            raise ArtifactUpstreamError("policy response must list exactly the selected adapter transport files")
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ArtifactUpstreamError("policy response has an invalid file entry")
            if set(entry) != {"name", "sha256", "size_bytes", "url"}:
                raise ArtifactUpstreamError("policy file fields do not match the Worker manifest contract")
            name, digest, size, url = (entry.get("name"), entry.get("sha256"),
                                       entry.get("size_bytes"), entry.get("url"))
            if name not in expected_names or name in seen:
                raise ArtifactUpstreamError("policy response has an unexpected or duplicate file name")
            if digest is not None and (not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest.lower())):
                raise ArtifactUpstreamError(f"policy response has an invalid SHA-256 declaration for {name}")
            if size is None and digest is None:
                raise ArtifactUpstreamError(f"policy response omitted the required size for {name}")
            if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
                raise ArtifactUpstreamError(f"policy response has an invalid size for {name}")
            if not isinstance(url, str) or not url:
                raise ArtifactUpstreamError("policy response omitted a file URL")
            seen.add(name)
        if seen != expected_names:
            raise ArtifactUpstreamError("policy response omitted a required adapter file")
        return version, normalized

    def _rewrite_and_check_urls(self, version: int, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(payload))
        for entry in normalized["files"]:
            name = entry["name"]
            target = urljoin(self.upstream_url + "/", entry["url"])
            parts = urlsplit(target)
            origin = (parts.scheme.lower(), (parts.hostname or "").lower(),
                      parts.port or (443 if parts.scheme == "https" else 80))
            expected_path = f"/policy/{version}/{name}"
            if (origin != self._upstream_origin or parts.path.rstrip("/") != expected_path
                    or parts.username is not None or parts.password is not None
                    or parts.query or parts.fragment):
                raise ArtifactUpstreamError("policy file URL must target its versioned same-origin Learner route")
            entry["url"] = expected_path
        return normalized

    def _save_manifest(self, version: int, policy: dict[str, Any]) -> dict[str, Any]:
        policy_sha = hashlib.sha256(_canonical_json(policy)).hexdigest()
        record_path = self._manifest_record_path(policy_sha)
        index_path = self._manifest_index_path(version)
        with self._manifest_lock:
            prior = self._read_record(version)
            if prior is not None:
                if not hmac.compare_digest(prior["policy_sha256"], policy_sha):
                    raise ImmutablePolicyConflict(
                        f"lineage {self.lineage_id} policy version {version} changed after publication"
                    )
                return prior
            file_meta = {entry["name"]: {
                "sha256": entry["sha256"].lower() if entry["sha256"] is not None else None,
                "size_bytes": entry["size_bytes"],
            } for entry in policy["files"]}
            record = {
                "schema": _MANIFEST_SCHEMA,
                "lineage_id": self.lineage_id,
                "version": version,
                "policy_sha256": policy_sha,
                "policy": policy,
                "files": file_meta,
                "created_at": _utc_now(),
            }
            if record_path.exists():
                try:
                    existing = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ArtifactIntegrityError("content-addressed policy manifest is unreadable") from exc
                if existing != record:
                    # `created_at` is not an identity field; preserve the first durable receipt.
                    identity_fields = ("schema", "lineage_id", "version", "policy_sha256", "policy", "files")
                    if any(existing.get(key) != record.get(key) for key in identity_fields):
                        raise ArtifactIntegrityError("content-addressed policy manifest path has conflicting data")
                    record = existing
            else:
                _write_json_atomic(record_path, record)
            _write_json_atomic(index_path, {
                "schema": _MANIFEST_SCHEMA,
                "lineage_id": self.lineage_id,
                "version": version,
                "policy_sha256": policy_sha,
            })
            return record

    async def _request_policy(self, version: int | None) -> dict[str, Any]:
        route = "policy" if version is None else f"policy/{version}"
        url = f"{self.upstream_url}/{route}"
        started = _utc_now()
        status: int | None = None
        payload_bytes = 0
        try:
            response = await self._client.get(url, headers=self.headers, timeout=self._request_timeout)
            status = response.status_code
            payload_bytes = len(response.content)
            if status == 404 and version is not None:
                raise PolicyVersionUnavailable(f"policy version {version} is unavailable at the origin")
            if status != 200:
                raise ArtifactUpstreamError(f"policy origin returned HTTP {status}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise ArtifactUpstreamError("policy origin returned invalid JSON") from exc
            received_version, payload = self._validate_manifest(payload, version)
            payload = self._rewrite_and_check_urls(received_version, payload)
            return payload
        except PolicyVersionUnavailable as exc:
            self._record_origin({
                "kind": "policy_manifest", "route": route, "requested_version": version,
                "status_code": status, "bytes": payload_bytes, "outcome": "unavailable",
                "error": type(exc).__name__, "started_at": started, "finished_at": _utc_now(),
            })
            raise
        except ArtifactMirrorError as exc:
            self._record_origin({
                "kind": "policy_manifest", "route": route, "requested_version": version,
                "status_code": status, "bytes": payload_bytes, "outcome": "failed",
                "error": type(exc).__name__, "started_at": started, "finished_at": _utc_now(),
            })
            raise
        except asyncio.CancelledError:
            self._record_origin({
                "kind": "policy_manifest", "route": route, "requested_version": version,
                "status_code": status, "bytes": payload_bytes, "outcome": "cancelled",
                "error": "CancelledError", "started_at": started, "finished_at": _utc_now(),
            })
            raise
        except httpx.HTTPError as exc:
            self._record_origin({
                "kind": "policy_manifest", "route": route, "requested_version": version,
                "status_code": status, "bytes": payload_bytes, "outcome": "failed",
                "error": type(exc).__name__, "started_at": started, "finished_at": _utc_now(),
            })
            raise ArtifactUpstreamError(f"policy origin request failed ({type(exc).__name__})") from exc
        else:
            pass
        finally:
            # Success receipt is written by the caller after manifest identity is saved.
            pass

    def _record_origin(self, event: dict[str, Any]) -> None:
        _append_jsonl(self.cache_dir / "origin-requests.jsonl", event, self._receipt_lock)

    async def policy(self, version: int | None = None) -> dict[str, Any]:
        """Return one immutable Worker policy manifest, using cache for pinned versions."""
        if version is not None and (isinstance(version, bool) or not isinstance(version, int) or version < 0):
            raise ValueError("version must be a nonnegative integer or None")
        if version is not None:
            record = self._read_record(version)
            if record is not None:
                return record["policy"]
            lock = await self._key_lock(f"manifest:{version}")
            async with lock:
                record = self._read_record(version)
                if record is not None:
                    return record["policy"]
                started = _utc_now()
                payload = await self._request_policy(version)
                record = self._save_manifest(version, payload)
                self._record_origin({
                    "kind": "policy_manifest", "route": f"policy/{version}",
                    "lineage_id": self.lineage_id, "version": version,
                    "manifest_sha256": record["policy_sha256"], "outcome": "cached",
                    "started_at": started, "finished_at": _utc_now(),
                })
                return record["policy"]

        now = time.monotonic()
        if self._latest_version is not None and now - self._latest_checked_at < self._latest_ttl_seconds:
            record = self._read_record(self._latest_version)
            if record is not None:
                return record["policy"]
        async with self._latest_lock:
            now = time.monotonic()
            if self._latest_version is not None and now - self._latest_checked_at < self._latest_ttl_seconds:
                record = self._read_record(self._latest_version)
                if record is not None:
                    return record["policy"]
            started = _utc_now()
            payload = await self._request_policy(None)
            latest, payload = self._validate_manifest(payload, None)
            # `_request_policy` already rewrote and validated URLs against this version.
            version_lock = await self._key_lock(f"manifest:{latest}")
            async with version_lock:
                record = self._save_manifest(latest, payload)
            self._latest_version = latest
            self._latest_checked_at = time.monotonic()
            self._record_origin({
                "kind": "policy_manifest", "route": "policy", "lineage_id": self.lineage_id,
                "version": latest, "manifest_sha256": record["policy_sha256"],
                "outcome": "cached", "started_at": started, "finished_at": _utc_now(),
            })
            return record["policy"]

    def _policy_object_path(self, version: int, filename: str, cache_key: str) -> Path:
        if (isinstance(version, bool) or not isinstance(version, int) or version < 0
                or filename not in POLICY_FILES or not _SHA256_RE.fullmatch(cache_key)):
            raise ArtifactIntegrityError("policy object identity is invalid")
        cache_root = self.cache_dir / "policy-objects"
        cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root = cache_root / self.lineage_id
        root.mkdir(mode=0o700, exist_ok=True)
        version_dir = root / f"v{version:08d}"
        version_dir.mkdir(mode=0o700, exist_ok=True)
        object_dir = version_dir / cache_key
        object_dir.mkdir(mode=0o700, exist_ok=True)
        for directory in (cache_root, root, version_dir, object_dir):
            try:
                info = directory.lstat()
            except OSError as exc:
                raise ArtifactIntegrityError("policy object cache directory is unavailable") from exc
            if (directory.is_symlink() or not stat.S_ISDIR(info.st_mode)
                    or (directory != self.cache_dir and stat.S_IMODE(info.st_mode) & 0o077)):
                raise ArtifactIntegrityError("policy object cache directory is unsafe")
        return object_dir / filename

    async def _download_policy_object(
        self, version: int, name: str, cache_key: str, expected_size: int | None,
    ) -> Path:
        is_tensor = not name.endswith(".json")
        expected_content_type = TENSOR_PAYLOAD_CONTENT_TYPE if is_tensor else "application/json"
        raw_limit = MAX_TENSOR_RAW_BYTES if expected_size is None else expected_size
        if is_tensor:
            if raw_limit > MAX_TENSOR_RAW_BYTES:
                raise ArtifactIntegrityError("policy tensor exceeds the raw transfer limit")
            wire_limit = max_encoded_size(raw_limit)
        else:
            if raw_limit > _MAX_POLICY_JSON_BYTES:
                raise ArtifactIntegrityError("policy metadata exceeds the transfer limit")
            wire_limit = raw_limit
        target = self._policy_object_path(version, name, cache_key)
        url = f"{self.upstream_url}/policy/{version}/{name}"
        started = _utc_now()
        status: int | None = None
        received = 0
        declared_wire_size: int | None = None
        descriptor, stage_name = tempfile.mkstemp(prefix=f".{cache_key}.", suffix=".tmp", dir=target.parent)
        stage = Path(stage_name)
        outcome = "failed"
        error_name: str | None = None
        try:
            async with self._client.stream("GET", url, headers=self.headers, timeout=self._request_timeout) as response:
                status = response.status_code
                if status != 200:
                    raise ArtifactUpstreamError(f"policy file origin returned HTTP {status}")
                content_type = response.headers.get("content-type", "")
                content_type = content_type.split(";", 1)[0].strip().lower()
                if content_type != expected_content_type:
                    raise ArtifactIntegrityError(f"origin payload content type mismatch for {name}")
                content_encoding = response.headers.get("content-encoding", "identity").lower()
                if content_encoding not in {"", "identity"}:
                    raise ArtifactIntegrityError(f"origin payload content encoding mismatch for {name}")
                raw_content_length = response.headers.get("content-length")
                if raw_content_length is not None:
                    if not raw_content_length.isascii() or not raw_content_length.isdecimal():
                        raise ArtifactIntegrityError(f"origin payload has invalid Content-Length for {name}")
                    declared_wire_size = int(raw_content_length)
                    if declared_wire_size > wire_limit:
                        raise ArtifactIntegrityError(f"origin payload wire size exceeds the limit for {name}")
                with os.fdopen(descriptor, "wb") as output:
                    descriptor = -1
                    async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                        if not chunk:
                            continue
                        received += len(chunk)
                        if received > wire_limit:
                            raise ArtifactIntegrityError(f"origin payload wire size exceeds the limit for {name}")
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            if declared_wire_size is not None and received != declared_wire_size:
                raise ArtifactIntegrityError(f"origin payload wire size mismatch for {name}")
            if is_tensor:
                observed_raw_size = _inspect_tensor_file(
                    stage,
                    max_raw_bytes=raw_limit,
                    safetensors=name == "adapter_model.safetensors",
                )
                if expected_size is not None and observed_raw_size != expected_size:
                    raise ArtifactIntegrityError(f"origin tensor raw size mismatch for {name}")
            elif expected_size is not None and received != expected_size:
                raise ArtifactIntegrityError(f"origin payload size mismatch for {name}")
            os.chmod(stage, 0o444)
            os.replace(stage, target)
            _fsync_dir(target.parent)
            outcome = "cached"
            return target
        except asyncio.CancelledError:
            outcome = "cancelled"
            error_name = "CancelledError"
            raise
        except (ArtifactMirrorError, httpx.HTTPError, OSError) as exc:
            error_name = type(exc).__name__
            if isinstance(exc, httpx.HTTPError):
                raise ArtifactUpstreamError(f"policy file request failed ({type(exc).__name__})") from exc
            if isinstance(exc, OSError):
                raise ArtifactMirrorError("could not atomically save policy payload") from exc
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            stage.unlink(missing_ok=True)
            self._record_origin({
                "kind": "policy_file", "route": f"policy/{version}/{name}",
                "lineage_id": self.lineage_id, "version": version, "filename": name,
                "object_key": cache_key, "expected_size": expected_size,
                "wire_limit_bytes": wire_limit, "bytes": received,
                "status_code": status, "outcome": outcome, "error": error_name,
                "started_at": started, "finished_at": _utc_now(),
            })

    async def policy_file(
        self, version: int, filename: str, *, expected_object_key: str | None = None,
        expected_origin: str | None = None,
    ) -> Path:
        """Return an immutable policy payload addressed by manifest identity and size."""
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("version must be a nonnegative integer")
        if filename not in POLICY_FILES:
            raise ArtifactNotFound("policy filename is not part of the Worker policy contract")
        await self.policy(version)
        record = self._read_record(version)
        if record is None:
            raise PolicyVersionUnavailable(f"policy version {version} is unavailable")
        file_meta = record["files"].get(filename)
        if not isinstance(file_meta, dict):
            raise ArtifactIntegrityError("cached policy is missing file metadata")
        local_cache_key = policy_file_cache_key(record["policy"], filename)
        if expected_object_key is not None:
            if expected_origin is None:
                raise ArtifactIntegrityError("policy object origin is missing")
            expected_key = policy_file_cache_key(record["policy"], filename, origin=expected_origin)
            if not hmac.compare_digest(expected_key, expected_object_key):
                raise ArtifactIntegrityError("policy object key differs from the pinned manifest metadata")
        expected_size = file_meta.get("size_bytes")
        is_tensor = not filename.endswith(".json")
        raw_limit = MAX_TENSOR_RAW_BYTES if expected_size is None else expected_size
        if is_tensor:
            if raw_limit > MAX_TENSOR_RAW_BYTES:
                raise ArtifactIntegrityError("policy tensor exceeds the raw transfer limit")
            wire_limit = max_encoded_size(raw_limit)
        else:
            if raw_limit > _MAX_POLICY_JSON_BYTES:
                raise ArtifactIntegrityError("policy metadata exceeds the transfer limit")
            wire_limit = raw_limit
        target = self._policy_object_path(version, filename, local_cache_key)
        lock = await self._key_lock(f"policy-object:{local_cache_key}")
        async with lock:
            try:
                info = target.lstat()
            except FileNotFoundError:
                info = None
            if info is not None:
                if target.is_symlink() or not stat.S_ISREG(info.st_mode):
                    raise ArtifactIntegrityError("cached policy object is not a regular file")
                if ((not is_tensor and expected_size is not None and info.st_size != expected_size)
                        or (is_tensor and not 0 < info.st_size <= wire_limit)):
                    target.unlink()
                elif is_tensor:
                    if stat.S_IMODE(info.st_mode) != 0o444:
                        raise ArtifactIntegrityError("cached policy object has unsafe mode bits")
                    try:
                        observed_raw_size = _inspect_tensor_file(
                            target,
                            max_raw_bytes=raw_limit,
                            safetensors=filename == "adapter_model.safetensors",
                        )
                    except ArtifactIntegrityError:
                        target.unlink()
                    else:
                        if expected_size is not None and observed_raw_size != expected_size:
                            target.unlink()
                        else:
                            return target
                elif stat.S_IMODE(info.st_mode) != 0o444:
                    raise ArtifactIntegrityError("cached policy object has unsafe mode bits")
                else:
                    return target
            return await self._download_policy_object(version, filename, local_cache_key, expected_size)

    async def receive_artifact(self, sha256: str, chunks: AsyncIterable[bytes],
                               *, expected_size: int | None = None) -> ArtifactRecord:
        """Stream a centrally published model blob into the site's immutable store."""
        digest = _validate_sha256(sha256)
        if expected_size is not None and (
            isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0
        ):
            raise ValueError("expected_size must be a nonnegative integer or None")
        target = self.store.path_for(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = await self._key_lock(f"blob:{digest}")
        async with lock:
            descriptor, stage_name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=target.parent)
            stage = Path(stage_name)
            started = _utc_now()
            digest_state = hashlib.sha256()
            received = 0
            outcome = "failed"
            error_name: str | None = None
            try:
                with os.fdopen(descriptor, "wb") as output:
                    descriptor = -1
                    async for chunk in chunks:
                        if not chunk:
                            continue
                        digest_state.update(chunk)
                        received += len(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                actual_sha = digest_state.hexdigest()
                if not hmac.compare_digest(actual_sha, digest):
                    raise ArtifactIntegrityError("published content SHA-256 differs from its route")
                if expected_size is not None and received != expected_size:
                    raise ArtifactIntegrityError("published content size differs from Content-Length")
                try:
                    existing = await asyncio.to_thread(self.store.resolve, digest, expected_size=received)
                except (ArtifactNotFound, ArtifactIntegrityError):
                    existing = None
                if existing is not None:
                    stage.unlink(missing_ok=True)
                    outcome = "already_present"
                    return existing
                os.chmod(stage, 0o444)
                os.replace(stage, target)
                _fsync_dir(target.parent)
                self.store.remember_verified(digest, target, expected_size=received)
                outcome = "published"
                return ArtifactRecord(digest, received, target)
            except asyncio.CancelledError:
                outcome = "cancelled"
                error_name = "CancelledError"
                raise
            except ArtifactMirrorError as exc:
                error_name = type(exc).__name__
                raise
            except OSError as exc:
                error_name = type(exc).__name__
                raise ArtifactMirrorError("could not persist uploaded artifact") from exc
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                stage.unlink(missing_ok=True)
                _append_jsonl(self.cache_dir / "artifact-publish-receipts.jsonl", {
                    "kind": "central_artifact_publish", "sha256": digest,
                    "observed_sha256": digest_state.hexdigest(), "size_bytes": received,
                    "expected_size": expected_size, "outcome": outcome, "error": error_name,
                    "started_at": started, "finished_at": _utc_now(),
                }, self._receipt_lock)

    def publish_artifact(self, source_path: str | Path,
                         *, expected_sha256: str | None = None) -> ArtifactRecord:
        """Publish a local model file to the content-SHA route."""
        return self.store.publish_file(source_path, expected_sha256=expected_sha256)


    def artifact_file(self, sha256: str) -> Path:
        return self.store.resolve(sha256).path


def create_artifact_router(
    mirror: ArtifactMirror,
    policy_ready_hook: Callable[[dict[str, Any]], Awaitable[Any]] | None = None,
) -> APIRouter:
    """Build reusable policy/content-SHA routes for the site Gateway."""
    router = APIRouter()

    def as_http_error(exc: ArtifactMirrorError) -> HTTPException:
        if isinstance(exc, ArtifactNotFound):
            return HTTPException(status_code=404, detail={"code": "artifact_unavailable"})
        return HTTPException(status_code=502, detail={"code": "artifact_mirror_error"})

    def checked_sha(sha256: str) -> str:
        try:
            return _validate_sha256(sha256)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_artifact_sha256"}) from exc

    async def await_policy_ready(policy: dict[str, Any]) -> None:
        if policy_ready_hook is None:
            return
        try:
            await policy_ready_hook(policy)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "policy_preseed_unavailable"}) from exc

    @router.get("/policy")
    async def get_latest_policy() -> dict[str, Any]:
        try:
            policy = await mirror.policy()
            await await_policy_ready(policy)
            return policy
        except HTTPException:
            raise
        except ArtifactMirrorError as exc:
            raise as_http_error(exc) from exc

    @router.get("/policy/{version}")
    async def get_policy(version: int) -> dict[str, Any]:
        if version < 0:
            raise HTTPException(status_code=422, detail={"code": "invalid_policy_version"})
        try:
            policy = await mirror.policy(version)
            await await_policy_ready(policy)
            return policy
        except HTTPException:
            raise
        except ArtifactMirrorError as exc:
            raise as_http_error(exc) from exc

    @router.get("/policy/{version}/{filename}")
    async def get_policy_file(
        version: int, filename: str, request: Request, policy_object_key: str | None = None,
    ) -> FileResponse:
        if version < 0:
            raise HTTPException(status_code=422, detail={"code": "invalid_policy_version"})
        try:
            expected_origin = f"{request.url.scheme}://{request.url.netloc}"
            path = await mirror.policy_file(
                version, filename, expected_object_key=policy_object_key,
                expected_origin=expected_origin,
            )
        except ArtifactMirrorError as exc:
            raise as_http_error(exc) from exc
        media_type = "application/json" if filename.endswith(".json") else TENSOR_PAYLOAD_CONTENT_TYPE
        response = FileResponse(
            path, media_type=media_type, filename=filename,
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )
        response.chunk_size = _CHUNK_SIZE
        return response

    @router.get("/v1/artifacts/{sha256}")
    async def get_artifact(sha256: str) -> FileResponse:
        digest = checked_sha(sha256)
        try:
            path = await asyncio.to_thread(mirror.artifact_file, digest)
        except ArtifactMirrorError as exc:
            raise as_http_error(exc) from exc
        return FileResponse(path, media_type="application/octet-stream",
                            headers={"Cache-Control": "private, max-age=31536000, immutable",
                                     "ETag": f'"sha256:{digest}"'})

    return router


def create_artifact_publish_router(mirror: ArtifactMirror, *, publish_dependency: Any) -> APIRouter:
    """Build separate management-authenticated HEAD/PUT routes for model publishing.

    `publish_dependency` is required and should be `Depends(...)` for a central
    publisher credential. It is intentionally separate from the registered
    Worker-token dependency on the read-only router.
    """
    if publish_dependency is None:
        raise ValueError("model artifact publishing requires a management auth dependency")
    router = APIRouter(dependencies=[publish_dependency])

    @router.head("/v1/artifacts/{sha256}")
    async def artifact_present(sha256: str) -> Response:
        try:
            digest = _validate_sha256(sha256)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_artifact_sha256"}) from exc
        try:
            path = await asyncio.to_thread(mirror.artifact_file, digest)
        except ArtifactNotFound as exc:
            raise HTTPException(status_code=404, detail={"code": "artifact_unavailable"}) from exc
        except ArtifactMirrorError as exc:
            raise HTTPException(status_code=502, detail={"code": "artifact_mirror_error"}) from exc
        size = (await asyncio.to_thread(path.stat)).st_size
        return Response(status_code=200, headers={
            "Content-Length": str(size),
            "ETag": f'"sha256:{digest}"',
        })

    @router.put("/v1/artifacts/{sha256}")
    async def receive_artifact(sha256: str, request: Request) -> dict[str, Any]:
        try:
            digest = _validate_sha256(sha256)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_artifact_sha256"}) from exc
        raw_size = request.headers.get("content-length")
        if raw_size is not None and not raw_size.isdigit():
            raise HTTPException(status_code=400, detail={"code": "invalid_content_length"})
        try:
            record = await mirror.receive_artifact(
                digest, request.stream(), expected_size=int(raw_size) if raw_size is not None else None,
            )
        except ArtifactIntegrityError as exc:
            raise HTTPException(status_code=422, detail={"code": "artifact_integrity_error"}) from exc
        except ArtifactMirrorError as exc:
            raise HTTPException(status_code=503, detail={"code": "artifact_publish_failed"}) from exc
        return {"sha256": record.sha256, "size_bytes": record.size, "state": "published"}

    return router


def create_artifact_app(
    mirror: ArtifactMirror,
    policy_ready_hook: Callable[[dict[str, Any]], Awaitable[Any]] | None = None,
) -> FastAPI:
    """Create a standalone HTTP app; Gateway callers can include the router directly."""
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await mirror.aclose()

    app = FastAPI(title="LAN artifact mirror", lifespan=lifespan)
    app.include_router(create_artifact_router(mirror, policy_ready_hook))
    return app


def _publish_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish immutable SHA-256 model artifacts")
    commands = parser.add_subparsers(dest="command", required=True)
    file_parser = commands.add_parser("file", help="publish one file")
    file_parser.add_argument("source", type=Path)
    file_parser.add_argument("--cache-dir", required=True, type=Path)
    file_parser.add_argument("--sha256", help="require this SHA-256 while publishing")
    args = parser.parse_args(argv)
    result = publish_artifact(args.source, args.cache_dir,
                              expected_sha256=args.sha256).as_dict()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_publish_cli())
