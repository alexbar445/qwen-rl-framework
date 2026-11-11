"""Pinned Dragonfly dfget transfers with explicit content-SHA and opaque policy modes."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import yaml

from qwen_rl.cluster.artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFound,
    ContentAddressedStore,
    _append_jsonl,
    _fsync_dir,
    _hash_file,
    _validate_sha256,
)

DRAGONFLY_SERVER_VERSION = "2.5.2"
DRAGONFLY_CLIENT_VERSION = "1.5.7"
DRAGONFLY_SERVER_LINUX_AMD64_ARCHIVE_SHA256 = "248554608b7aba01293aa2c80a97ae1b2a98f41106f100ba40b7e8854577d984"
DRAGONFLY_CLIENT_LINUX_AMD64_MUSL_ARCHIVE_SHA256 = "b2287ae6aea393a6311eee76d82f213f1320cdfcae96cac37c02366429e399d9"
DRAGONFLY_DFGET_LINUX_AMD64_MUSL_SHA256 = "fc394751638cafd6601c2337f4151205b8c33b28227039c2e6603d5546b4b8cf"
DRAGONFLY_DFDAEMON_LINUX_AMD64_MUSL_SHA256 = "6afdddbb8569ff584178e3889ccc8ad38a1b8455fb8d933dad377b30c774b65f"
DRAGONFLY_SCHEDULER_GHCR_INDEX = "sha256:d5dea9e662cdc0ceb8c7c624e06ab714aa803c90a124a26a75da0551411878d5"
DRAGONFLY_CLIENT_GHCR_INDEX = "sha256:e5a4479c3e9c57c2b34ac21db346827a2d833be2eacbad474e45b337decc904d"
DRAGONFLY_CLIENT_SOURCE_COMMIT = "ada717934"
_HTTP_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


class DragonflyDownloadError(RuntimeError):
    """Pinned dfget failed or returned content that did not match its manifest."""


class DragonflyConfigError(ValueError):
    """Dragonfly config would permit an unpinned scheduler or WAN source fallback."""


class ModelManifestError(DragonflyDownloadError):
    """Model manifest does not meet the strict qwen-model/v1 contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_model_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ModelManifestError("model file names must be nonempty POSIX relative paths")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ModelManifestError("model file path escapes the model root")
    return path.as_posix()


def _origin(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise DragonflyDownloadError("artifact URL must use HTTP(S) and include a host")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise DragonflyDownloadError("artifact URL cannot include credentials, query, or fragment")
    return (parts.scheme.lower(), parts.hostname.lower(),
            parts.port or (443 if parts.scheme.lower() == "https" else 80))


def _host_is_private_or_local(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        name = host.rstrip(".").lower()
        return "." not in name or name.endswith((".local", ".lan", ".cluster.local", ".svc"))


def _parse_static_scheduler_address(value: str) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int]:
    """Parse the literal IP socket address accepted by client v1.5.7 `scheduler.addrs`."""
    try:
        if value.startswith("["):
            end = value.index("]")
            host, separator, port_text = value[1:end], value[end + 1:end + 2], value[end + 2:]
            if separator != ":" or not port_text.isdigit():
                raise ValueError
        else:
            host, separator, port_text = value.rpartition(":")
            if not separator or ":" in host or not port_text.isdigit():
                raise ValueError
        address = ipaddress.ip_address(host)
        port = int(port_text)
    except (ValueError, ipaddress.AddressValueError):
        raise ValueError("static Scheduler address must be a literal private IP:port") from None
    if (not address.is_private or address.is_loopback or address.is_link_local
            or address.is_multicast or address.is_unspecified or not 1 <= port <= 65535):
        raise ValueError("static Scheduler address must be a literal private IP:port")
    return address, port


def validate_dfdaemon_config(path: str | Path, *, allow_lan_source_fallback: bool = False) -> dict[str, Any]:
    """Validate the official client config fields required for a LAN-only peer."""
    path = Path(path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DragonflyConfigError("dfdaemon config is unreadable YAML") from exc
    if not isinstance(config, dict):
        raise DragonflyConfigError("dfdaemon config must be a YAML mapping")
    scheduler = config.get("scheduler")
    expected_fallback = allow_lan_source_fallback
    if not isinstance(scheduler, dict) or scheduler.get("enableBackToSource") is not expected_fallback:
        raise DragonflyConfigError("dfdaemon source fallback setting does not match the requested role")
    if config.get("manager"):
        raise DragonflyConfigError("LAN static-scheduler dfdaemon must not depend on a Dragonfly manager")
    dynconfig = config.get("dynconfig")
    if not isinstance(dynconfig, dict):
        raise DragonflyConfigError("dfdaemon must declare local dynconfig")
    dynconfig_path = dynconfig.get("path")
    if dynconfig_path:
        candidate = Path(dynconfig_path)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
    else:
        candidate = path.parent / "dynconfig.yaml"
    try:
        dynamic = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DragonflyConfigError("local Dragonfly dynconfig is unreadable YAML") from exc
    scheduler_config = dynamic.get("scheduler") if isinstance(dynamic, dict) else None
    addrs = scheduler_config.get("addrs") if isinstance(scheduler_config, dict) else None
    if not isinstance(addrs, list) or not addrs or any(not isinstance(item, str) or not item for item in addrs):
        raise DragonflyConfigError("local dynconfig must name at least one static Scheduler address")
    for address in addrs:
        try:
            _parse_static_scheduler_address(address)
        except ValueError as exc:
            raise DragonflyConfigError(str(exc)) from exc
    return config


ProcessFactory = Callable[..., Awaitable[asyncio.subprocess.Process]]


class DragonflyDownloader:
    """Use official dfget/dfdaemon P2P for LAN files; never run HTTP fallback here.

    ``download`` keeps strict content-SHA addressing for model/image assets.
    ``download_policy_object`` uses a separate manifest-derived key and size
    checks for policy files whose manifests no longer carry content digests.
    Worker requests disable back-to-source; seed requests may use the explicitly
    allowed site Gateway.
    """

    def __init__(
        self,
        dfget_path: str | Path,
        cache_dir: str | Path,
        *,
        dfdaemon_path: str | Path = "/usr/local/bin/dfdaemon",
        dfdaemon_socket: str | Path = "/var/run/dragonfly/dfdaemon.sock",
        allowed_hosts: Iterable[str],
        expected_version: str = DRAGONFLY_CLIENT_VERSION,
        expected_binary_sha256: str | None = DRAGONFLY_DFGET_LINUX_AMD64_MUSL_SHA256,
        expected_dfdaemon_sha256: str | None = DRAGONFLY_DFDAEMON_LINUX_AMD64_MUSL_SHA256,
        site_id: str = "default",
        site_role: str = "worker",
        scheduler_addrs: Sequence[str] = (),
        timeout: float = 7200.0,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        if not expected_version or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", expected_version):
            raise ValueError("expected_version must be a pinned semantic version")
        if not site_id or any(c in site_id for c in "\r\n\x00"):
            raise ValueError("site_id must be a nonempty single-line identity")
        if site_role not in {"worker", "seed"}:
            raise ValueError("site_role must be worker or seed")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        normalized_hosts = {str(host).lower().rstrip(".") for host in allowed_hosts}
        if not normalized_hosts:
            raise ValueError("allowed_hosts must explicitly name the site artifact source")
        if any(not _host_is_private_or_local(host) for host in normalized_hosts):
            raise ValueError("allowed_hosts may contain only RFC1918/private or local-DNS hosts")
        for address in scheduler_addrs:
            _parse_static_scheduler_address(address)
        self.dfget_path = Path(dfget_path)
        self.dfdaemon_path = Path(dfdaemon_path)
        self.dfdaemon_socket = Path(dfdaemon_socket)
        self.cache_dir = Path(cache_dir)
        self.store = ContentAddressedStore(self.cache_dir)
        self.allowed_hosts = normalized_hosts
        self.expected_version = expected_version
        self.expected_binary_sha256 = (_validate_sha256(expected_binary_sha256)
                                       if expected_binary_sha256 is not None else None)
        self.expected_dfdaemon_sha256 = (_validate_sha256(expected_dfdaemon_sha256)
                                        if expected_dfdaemon_sha256 is not None else None)
        self.site_id = site_id
        self.site_role = site_role
        self.scheduler_addrs = tuple(scheduler_addrs)
        self.timeout = timeout
        self._process_factory = process_factory or asyncio.create_subprocess_exec
        self._binaries_checked = False
        self._binary_lock = asyncio.Lock()
        self._locks_guard = asyncio.Lock()
        self._download_locks: dict[str, asyncio.Lock] = {}
        self._receipt_lock = threading.Lock()

    async def _download_lock(self, identity: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._download_locks.setdefault(identity, asyncio.Lock())

    def _validate_lan_url(self, url: str) -> tuple[str, str, int]:
        origin = _origin(url)
        host = origin[1].rstrip(".")
        if host not in self.allowed_hosts or not _host_is_private_or_local(host):
            raise DragonflyDownloadError("artifact source is not on the configured LAN allowlist")
        return origin

    async def _run(self, executable: str | Path, *argv: str) -> tuple[int, bytes, bytes]:
        try:
            process = await self._process_factory(
                str(executable), *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            label = Path(executable).name
            raise DragonflyDownloadError(f"could not execute pinned {label} ({type(exc).__name__})") from exc
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            try:
                process.kill()
                await process.communicate()
            except (OSError, ProcessLookupError):
                pass
            raise
        return int(process.returncode or 0), stdout or b"", stderr or b""

    async def _verify_pinned_binaries(self) -> None:
        if self._binaries_checked:
            return
        async with self._binary_lock:
            if self._binaries_checked:
                return
            for label, path, expected_sha in (
                ("dfget", self.dfget_path, self.expected_binary_sha256),
                ("dfdaemon", self.dfdaemon_path, self.expected_dfdaemon_sha256),
            ):
                if expected_sha is not None:
                    try:
                        actual_sha, _ = _hash_file(path)
                    except OSError as exc:
                        raise DragonflyDownloadError(f"pinned {label} binary is unavailable") from exc
                    if not hmac.compare_digest(actual_sha, expected_sha):
                        raise DragonflyDownloadError(f"installed {label} binary SHA-256 does not match the release pin")
                result, stdout, stderr = await self._run(path, "--version")
                version_output = (stdout + b"\n" + stderr).decode(errors="replace")
                if result != 0 or not re.search(rf"\b[vV]?{re.escape(self.expected_version)}\b", version_output):
                    raise DragonflyDownloadError(
                        f"{label} version does not match the configured official release pin"
                    )
            self._binaries_checked = True

    def _record(self, record: dict[str, Any]) -> None:
        _append_jsonl(self.cache_dir / "dragonfly-downloads.jsonl", record, self._receipt_lock)

    async def download(
        self,
        url: str,
        sha256: str,
        size: int | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        allow_lan_source_fallback: bool = False,
        force_transfer: bool = False,
    ) -> Path:
        """Download one immutable blob through dfget and atomically cache it."""
        digest = _validate_sha256(sha256)
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise ValueError("size must be a nonnegative integer or None")
        origin = self._validate_lan_url(url)
        if allow_lan_source_fallback and self.site_role != "seed":
            raise DragonflyDownloadError("only the site seed may use its allowlisted LAN source")
        if force_transfer and self.site_role != "seed":
            raise DragonflyDownloadError("only the site seed may force a Dragonfly cache preseed")
        safe_headers: list[tuple[str, str]] = []
        for name, value in (headers or {}).items():
            if (not isinstance(name, str) or not _HTTP_HEADER_NAME_RE.fullmatch(name)
                    or not isinstance(value, str) or any(c in value for c in "\r\n\x00")):
                raise ValueError("dfget headers must be single-line strings")
            safe_headers.append((name, value))

        target = self.store.path_for(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = await self._download_lock(f"content:{digest}")
        async with lock:
            try:
                cached = await asyncio.to_thread(self.store.resolve, digest, expected_size=size)
            except (ArtifactNotFound, ArtifactIntegrityError):
                cached = None
            if cached is not None and not force_transfer:
                self._record({
                    "kind": "dragonfly_download", "site_id": self.site_id,
                    "scheduler_addrs": list(self.scheduler_addrs), "url_host": origin[1],
                    "sha256": digest, "size_bytes": cached.size, "expected_size": size,
                    "client_version": self.expected_version, "outcome": "cache_hit",
                    "back_to_source_disabled": not allow_lan_source_fallback,
                    "force_transfer": False,
                    "exit_code": 0, "verified_at": _now(),
                })
                return cached.path

            await self._verify_pinned_binaries()
            descriptor, stage_name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=target.parent)
            os.close(descriptor)
            stage = Path(stage_name)
            stage.unlink(missing_ok=True)
            argv: list[str] = ["--endpoint", str(self.dfdaemon_socket),
                               "--transfer-from-dfdaemon",
                               "--output", str(stage), "--digest", f"sha256:{digest}"]
            if not allow_lan_source_fallback:
                argv.append("--disable-back-to-source")
            for name, value in safe_headers:
                argv.extend(("--header", f"{name}: {value}"))
            argv.append(url)
            started = _now()
            exit_code: int | None = None
            observed_sha: str | None = None
            observed_size: int | None = None
            try:
                exit_code, _, _ = await self._run(self.dfget_path, *argv)
                if exit_code != 0:
                    raise DragonflyDownloadError(f"pinned dfget returned exit code {exit_code}")
                if not stage.is_file() or stage.is_symlink():
                    raise DragonflyDownloadError("dfget succeeded without writing a regular output file")
                observed_sha, observed_size = await asyncio.to_thread(_hash_file, stage)
                if not hmac.compare_digest(observed_sha, digest):
                    raise DragonflyDownloadError("dfget payload SHA-256 does not match the artifact manifest")
                if size is not None and observed_size != size:
                    raise DragonflyDownloadError("dfget payload size does not match the artifact manifest")
                os.chmod(stage, 0o444)
                os.replace(stage, target)
                _fsync_dir(target.parent)
                self.store.remember_verified(digest, target, expected_size=observed_size)
                self._record({
                    "kind": "dragonfly_download", "site_id": self.site_id,
                    "scheduler_addrs": list(self.scheduler_addrs), "url_host": origin[1],
                    "dfdaemon_socket": str(self.dfdaemon_socket),
                    "sha256": digest, "observed_sha256": observed_sha,
                    "size_bytes": observed_size, "expected_size": size,
                    "client_version": self.expected_version, "outcome": "verified",
                    "transfer_from_dfdaemon": True,
                    "back_to_source_disabled": not allow_lan_source_fallback,
                    "force_transfer": force_transfer,
                    "exit_code": exit_code, "started_at": started, "finished_at": _now(),
                })
                return target
            except asyncio.CancelledError:
                self._record({
                    "kind": "dragonfly_download", "site_id": self.site_id,
                    "scheduler_addrs": list(self.scheduler_addrs), "url_host": origin[1],
                    "dfdaemon_socket": str(self.dfdaemon_socket),
                    "sha256": digest, "observed_sha256": observed_sha,
                    "size_bytes": observed_size, "expected_size": size,
                    "client_version": self.expected_version, "outcome": "cancelled",
                    "back_to_source_disabled": not allow_lan_source_fallback,
                    "force_transfer": force_transfer,
                    "exit_code": exit_code, "error": "CancelledError",
                    "started_at": started, "finished_at": _now(),
                })
                raise
            except (OSError, asyncio.TimeoutError, DragonflyDownloadError) as exc:
                self._record({
                    "kind": "dragonfly_download", "site_id": self.site_id,
                    "scheduler_addrs": list(self.scheduler_addrs), "url_host": origin[1],
                    "dfdaemon_socket": str(self.dfdaemon_socket),
                    "sha256": digest, "observed_sha256": observed_sha,
                    "size_bytes": observed_size, "expected_size": size,
                    "client_version": self.expected_version, "outcome": "failed",
                    "back_to_source_disabled": not allow_lan_source_fallback,
                    "force_transfer": force_transfer,
                    "exit_code": exit_code, "error": type(exc).__name__,
                    "started_at": started, "finished_at": _now(),
                })
                if isinstance(exc, DragonflyDownloadError):
                    raise
                if isinstance(exc, asyncio.TimeoutError):
                    raise DragonflyDownloadError("dfget download timed out") from exc
                raise DragonflyDownloadError("could not persist dfget output") from exc
            finally:
                stage.unlink(missing_ok=True)

    async def download_policy_object(
        self,
        url: str,
        object_key: str,
        size: int | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        allow_lan_source_fallback: bool = False,
        force_transfer: bool = False,
    ) -> Path:
        """Transfer a policy object by manifest key without a payload digest."""
        if not isinstance(object_key, str) or re.fullmatch(r"[0-9a-f]{64}", object_key) is None:
            raise ValueError("policy object key must be a lowercase 64-hex metadata fingerprint")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise ValueError("size must be a nonnegative integer or None")
        parts = urlsplit(url)
        if (parts.username is not None or parts.password is not None or parts.fragment):
            raise DragonflyDownloadError("policy object URL has credentials or a fragment")
        query = parse_qs(parts.query, keep_blank_values=True, strict_parsing=True)
        if query != {"policy_object_key": [object_key]}:
            raise DragonflyDownloadError("policy object URL key differs from its cache identity")
        origin_url = parts._replace(query="", fragment="").geturl()
        origin = self._validate_lan_url(origin_url)
        if allow_lan_source_fallback and self.site_role != "seed":
            raise DragonflyDownloadError("only the site seed may use its allowlisted LAN source")
        if force_transfer and self.site_role != "seed":
            raise DragonflyDownloadError("only the site seed may force a Dragonfly cache preseed")
        safe_headers: list[tuple[str, str]] = []
        for name, value in (headers or {}).items():
            if (not isinstance(name, str) or not _HTTP_HEADER_NAME_RE.fullmatch(name)
                    or not isinstance(value, str) or any(c in value for c in "\r\n\x00")):
                raise ValueError("dfget headers must be single-line strings")
            safe_headers.append((name, value))

        cache_root = self.cache_dir / "policy-objects"
        cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        prefix = cache_root / object_key[:2]
        prefix.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = prefix / object_key
        for directory in (self.cache_dir, cache_root, prefix):
            try:
                info = directory.lstat()
            except OSError as exc:
                raise DragonflyDownloadError("policy object cache directory is unavailable") from exc
            if (directory.is_symlink() or not stat.S_ISDIR(info.st_mode)
                    or (directory != self.cache_dir and stat.S_IMODE(info.st_mode) & 0o077)):
                raise DragonflyDownloadError("policy object cache directory is unsafe")

        lock = await self._download_lock(f"policy-object:{object_key}")
        async with lock:
            try:
                info = target.lstat()
            except FileNotFoundError:
                info = None
            if info is not None:
                if target.is_symlink() or not stat.S_ISREG(info.st_mode):
                    raise DragonflyDownloadError("cached policy object is not a regular file")
                if size is not None and info.st_size != size:
                    target.unlink()
                elif stat.S_IMODE(info.st_mode) != 0o444:
                    raise DragonflyDownloadError("cached policy object has unsafe mode bits")
                elif not force_transfer:
                    self._record({
                        "kind": "dragonfly_policy_object", "site_id": self.site_id,
                        "scheduler_addrs": list(self.scheduler_addrs), "url_host": origin[1],
                        "object_key": object_key, "size_bytes": info.st_size,
                        "expected_size": size, "client_version": self.expected_version,
                        "outcome": "cache_hit", "back_to_source_disabled": not allow_lan_source_fallback,
                        "force_transfer": False, "exit_code": 0, "verified_at": _now(),
                    })
                    return target

            await self._verify_pinned_binaries()
            descriptor, stage_name = tempfile.mkstemp(
                prefix=f".{object_key}.", suffix=".tmp", dir=target.parent,
            )
            os.close(descriptor)
            stage = Path(stage_name)
            stage.unlink(missing_ok=True)
            argv: list[str] = ["--endpoint", str(self.dfdaemon_socket),
                               "--transfer-from-dfdaemon", "--output", str(stage)]
            if not allow_lan_source_fallback:
                argv.append("--disable-back-to-source")
            for name, value in safe_headers:
                argv.extend(("--header", f"{name}: {value}"))
            argv.append(url)
            started = _now()
            exit_code: int | None = None
            observed_size: int | None = None
            try:
                exit_code, _, _ = await self._run(self.dfget_path, *argv)
                if exit_code != 0:
                    raise DragonflyDownloadError(f"pinned dfget returned exit code {exit_code}")
                if not stage.is_file() or stage.is_symlink():
                    raise DragonflyDownloadError("dfget succeeded without writing a regular policy object")
                file_info = stage.lstat()
                if not stat.S_ISREG(file_info.st_mode):
                    raise DragonflyDownloadError("dfget policy object is not a regular file")
                observed_size = file_info.st_size
                if size is not None and observed_size != size:
                    raise DragonflyDownloadError("dfget policy object size does not match its manifest")
                os.chmod(stage, 0o444)
                os.replace(stage, target)
                _fsync_dir(target.parent)
                self._record({
                    "kind": "dragonfly_policy_object", "site_id": self.site_id,
                    "scheduler_addrs": list(self.scheduler_addrs), "url_host": origin[1],
                    "dfdaemon_socket": str(self.dfdaemon_socket), "object_key": object_key,
                    "size_bytes": observed_size, "expected_size": size,
                    "client_version": self.expected_version, "outcome": "transferred",
                    "transfer_from_dfdaemon": True,
                    "back_to_source_disabled": not allow_lan_source_fallback,
                    "force_transfer": force_transfer, "exit_code": exit_code,
                    "started_at": started, "finished_at": _now(),
                })
                return target
            except asyncio.CancelledError:
                self._record({
                    "kind": "dragonfly_policy_object", "site_id": self.site_id,
                    "scheduler_addrs": list(self.scheduler_addrs), "url_host": origin[1],
                    "object_key": object_key, "size_bytes": observed_size, "expected_size": size,
                    "client_version": self.expected_version, "outcome": "cancelled",
                    "back_to_source_disabled": not allow_lan_source_fallback,
                    "force_transfer": force_transfer, "exit_code": exit_code,
                    "error": "CancelledError", "started_at": started, "finished_at": _now(),
                })
                raise
            except (OSError, asyncio.TimeoutError, DragonflyDownloadError) as exc:
                self._record({
                    "kind": "dragonfly_policy_object", "site_id": self.site_id,
                    "scheduler_addrs": list(self.scheduler_addrs), "url_host": origin[1],
                    "object_key": object_key, "size_bytes": observed_size, "expected_size": size,
                    "client_version": self.expected_version, "outcome": "failed",
                    "back_to_source_disabled": not allow_lan_source_fallback,
                    "force_transfer": force_transfer, "exit_code": exit_code,
                    "error": type(exc).__name__, "started_at": started, "finished_at": _now(),
                })
                if isinstance(exc, DragonflyDownloadError):
                    raise
                if isinstance(exc, asyncio.TimeoutError):
                    raise DragonflyDownloadError("dfget policy object transfer timed out") from exc
                raise DragonflyDownloadError("could not persist dfget policy object") from exc
            finally:
                stage.unlink(missing_ok=True)

    @staticmethod
    def _validate_model_manifest(
        raw: bytes,
        manifest_sha256: str,
        *,
        expected_artifact_id: str | None,
        expected_base_fingerprint: str | None,
        expected_artifact_fingerprint: str | None,
        manifest_url: str,
    ) -> dict[str, Any]:
        observed = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(observed, _validate_sha256(manifest_sha256)):
            raise ModelManifestError("model manifest SHA-256 does not match its pinned URL")
        try:
            manifest = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ModelManifestError("model manifest is not valid UTF-8 JSON") from exc
        if (not isinstance(manifest, dict) or manifest.get("schema") != "qwen-model/v1"
                or set(manifest) != {"schema", "artifact_id", "lineage_id", "base_fingerprint",
                                     "artifact_fingerprint", "files"}):
            raise ModelManifestError("model manifest schema must be qwen-model/v1")
        artifact_id = manifest.get("artifact_id")
        base_fp = manifest.get("base_fingerprint")
        artifact_fp = manifest.get("artifact_fingerprint")
        for name, value in (("artifact_id", artifact_id), ("base_fingerprint", base_fp),
                            ("artifact_fingerprint", artifact_fp)):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ModelManifestError(f"model manifest has an invalid {name}")
        if artifact_id != artifact_fp:
            raise ModelManifestError("model artifact_id must equal its verified artifact_fingerprint")
        if expected_artifact_id is not None and artifact_id != expected_artifact_id:
            raise ModelManifestError("model artifact_id differs from the pinned node configuration")
        if expected_base_fingerprint is not None and base_fp != expected_base_fingerprint:
            raise ModelManifestError("model base_fingerprint differs from the pinned node configuration")
        if expected_artifact_fingerprint is not None and artifact_fp != expected_artifact_fingerprint:
            raise ModelManifestError("model artifact_fingerprint differs from the pinned node configuration")
        lineage_id = manifest.get("lineage_id")
        if not isinstance(lineage_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", lineage_id):
            raise ModelManifestError("model manifest has an invalid lineage_id")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ModelManifestError("model manifest must contain at least one file")
        manifest_origin = _origin(manifest_url)
        manifest_path = urlsplit(manifest_url).path.rstrip("/")
        manifest_suffix = f"/v1/artifacts/{manifest_sha256}"
        if not manifest_path.endswith(manifest_suffix):
            raise ModelManifestError("pinned manifest URL must use the content SHA route")
        route_prefix = manifest_path[:-len(manifest_suffix)]
        names: set[str] = set()
        for entry in files:
            if not isinstance(entry, dict):
                raise ModelManifestError("model manifest contains a malformed file record")
            if set(entry) != {"name", "sha256", "size_bytes", "url"}:
                raise ModelManifestError("model file fields do not match qwen-model/v1")
            name = _safe_model_name(entry.get("name"))
            if name in names:
                raise ModelManifestError("model manifest contains duplicate file names")
            names.add(name)
            digest = _validate_sha256(entry.get("sha256", ""))
            size = entry.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ModelManifestError(f"model manifest has invalid size_bytes for {name}")
            url = entry.get("url")
            if not isinstance(url, str):
                raise ModelManifestError(f"model manifest omitted URL for {name}")
            if _origin(url) != manifest_origin:
                raise ModelManifestError("model file URLs must use the manifest's exact LAN origin")
            if urlsplit(url).path.rstrip("/") != f"{route_prefix}/v1/artifacts/{digest}":
                raise ModelManifestError("model file URL must use its content SHA route")
            entry["name"] = name
        return manifest

    async def install_model_artifact(
        self,
        manifest_url: str,
        manifest_sha256: str,
        destination: str | Path,
        *,
        expected_artifact_id: str | None = None,
        expected_base_fingerprint: str | None = None,
        expected_artifact_fingerprint: str | None = None,
        headers: Mapping[str, str] | None = None,
        allow_lan_source_fallback: bool = False,
    ) -> dict[str, Any]:
        """Download a pinned qwen-model/v1 manifest/files and atomically install it."""
        self._validate_lan_url(manifest_url)
        manifest_sha = _validate_sha256(manifest_sha256)
        manifest_blob = await self.download(
            manifest_url, manifest_sha, headers=headers,
            allow_lan_source_fallback=allow_lan_source_fallback,
        )
        raw_manifest = manifest_blob.read_bytes()
        manifest = self._validate_model_manifest(
            raw_manifest, manifest_sha,
            expected_artifact_id=expected_artifact_id,
            expected_base_fingerprint=expected_base_fingerprint,
            expected_artifact_fingerprint=expected_artifact_fingerprint,
            manifest_url=manifest_url,
        )
        target = Path(destination)
        if target.is_symlink():
            raise ModelManifestError("model install destination cannot be a symlink")
        if target.exists():
            if not target.is_dir():
                raise ModelManifestError("model install destination exists and is not a directory")
            expected_names = {entry["name"] for entry in manifest["files"]}
            actual_names: set[str] = set()
            for path in target.rglob("*"):
                if path.is_symlink():
                    raise ModelManifestError("existing model directory contains a symlink")
                if path.is_file():
                    actual_names.add(path.relative_to(target).as_posix())
            if actual_names != expected_names:
                raise ModelManifestError("existing model directory does not match the pinned manifest inventory")
            for entry in manifest["files"]:
                path = target.joinpath(*PurePosixPath(entry["name"]).parts)
                if path.is_symlink():
                    raise ModelManifestError("existing model directory contains a symlink")
                observed_sha, observed_size = await asyncio.to_thread(_hash_file, path)
                if (observed_sha != entry["sha256"] or observed_size != entry["size_bytes"]):
                    raise ModelManifestError("existing model directory has a corrupted file")
            return {"path": str(target), **{key: manifest[key] for key in (
                "schema", "artifact_id", "lineage_id", "base_fingerprint", "artifact_fingerprint")},
                    "manifest_sha256": manifest_sha, "files": len(manifest["files"]),
                    "state": "already_installed"}

        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.parent / f".{target.name}.{uuid4().hex}.staging"
        stage.mkdir(mode=0o700)
        try:
            for entry in manifest["files"]:
                blob = await self.download(
                    entry["url"], entry["sha256"], entry["size_bytes"], headers,
                    allow_lan_source_fallback=allow_lan_source_fallback,
                )
                relative = PurePosixPath(entry["name"])
                output = stage.joinpath(*relative.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(blob, output)
                except OSError:
                    shutil.copyfile(blob, output)
                    observed_sha, observed_size = await asyncio.to_thread(_hash_file, output)
                    if observed_sha != entry["sha256"] or observed_size != entry["size_bytes"]:
                        raise ModelManifestError("copied model file failed SHA/size verification")
                    os.chmod(output, 0o444)
            for directory in sorted((path for path in stage.rglob("*") if path.is_dir()),
                                    key=lambda item: len(item.parts), reverse=True):
                os.chmod(directory, 0o555)
            os.chmod(stage, 0o555)
            os.replace(stage, target)
            _fsync_dir(target.parent)
        except Exception:
            os.chmod(stage, 0o700)
            for directory in (path for path in stage.rglob("*") if path.is_dir()):
                try:
                    os.chmod(directory, 0o700)
                except OSError:
                    pass
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return {"path": str(target), **{key: manifest[key] for key in (
            "schema", "artifact_id", "lineage_id", "base_fingerprint", "artifact_fingerprint")},
                "manifest_sha256": manifest_sha, "files": len(manifest["files"]),
                "state": "installed"}


async def preseed_model_artifact(
    downloader: DragonflyDownloader,
    manifest_url: str,
    manifest_sha256: str,
    *,
    expected_artifact_id: str,
    expected_base_fingerprint: str,
    expected_artifact_fingerprint: str,
    headers: Mapping[str, str],
) -> dict[str, Any]:
    """Populate the selected seed's dfdaemon cache before Workers are released.

    This runs on the site seed host after the Controller publishes the immutable
    blobs into Gateway storage. The seed is the only role allowed to source from
    that explicitly LAN-allowlisted Gateway route; ordinary Worker instances
    cannot opt into the source fallback.
    """
    if downloader.site_role != "seed":
        raise DragonflyDownloadError("model preseed requires a DragonflyDownloader with site_role=seed")
    digest = _validate_sha256(manifest_sha256)
    manifest_path = await downloader.download(
        manifest_url, digest, headers=headers,
        allow_lan_source_fallback=True, force_transfer=True,
    )
    raw_manifest = await asyncio.to_thread(manifest_path.read_bytes)
    manifest = downloader._validate_model_manifest(
        raw_manifest, digest,
        expected_artifact_id=expected_artifact_id,
        expected_base_fingerprint=expected_base_fingerprint,
        expected_artifact_fingerprint=expected_artifact_fingerprint,
        manifest_url=manifest_url,
    )
    for entry in manifest["files"]:
        await downloader.download(
            entry["url"], entry["sha256"], entry["size_bytes"], headers,
            allow_lan_source_fallback=True, force_transfer=True,
        )
    return {
        "state": "preseeded",
        "site_id": downloader.site_id,
        "artifact_id": manifest["artifact_id"],
        "manifest_sha256": digest,
        "base_fingerprint": manifest["base_fingerprint"],
        "artifact_fingerprint": manifest["artifact_fingerprint"],
        "files": len(manifest["files"]),
        "dfdaemon_socket": str(downloader.dfdaemon_socket),
    }


def _read_private_bearer_token(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("token file must be a regular file with mode 0600 or stricter")
    token = path.read_text(encoding="utf-8").strip()
    if not token or any(char.isspace() or char in "\r\n\x00" for char in token):
        raise ValueError("token file must contain one Bearer token")
    return token


def _preseed_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Populate the Dragonfly seed cache from a published qwen model")
    parser.add_argument("--manifest-url", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--base-fingerprint", required=True)
    parser.add_argument("--artifact-fingerprint", required=True)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--scheduler-addr", action="append", required=True)
    parser.add_argument("--dfdaemon-config", required=True, type=Path)
    parser.add_argument("--dfget-path", type=Path, default=Path("/usr/local/bin/dfget"))
    parser.add_argument("--dfdaemon-path", type=Path, default=Path("/usr/local/bin/dfdaemon"))
    parser.add_argument("--dfdaemon-socket", type=Path, default=Path("/var/run/dragonfly/dfdaemon.sock"))
    parser.add_argument("--gateway-token-file", required=True, type=Path,
                        help="0600 file containing the Gateway service Bearer token")
    args = parser.parse_args(argv)
    try:
        validate_dfdaemon_config(args.dfdaemon_config, allow_lan_source_fallback=True)
        token = _read_private_bearer_token(args.gateway_token_file)
        host = urlsplit(args.manifest_url).hostname
        if not host:
            raise ValueError("manifest URL has no host")
        downloader = DragonflyDownloader(
            args.dfget_path, args.cache_dir, dfdaemon_path=args.dfdaemon_path,
            dfdaemon_socket=args.dfdaemon_socket,
            allowed_hosts={host}, site_id=args.site_id, site_role="seed",
            scheduler_addrs=args.scheduler_addr,
        )
        receipt = asyncio.run(preseed_model_artifact(
            downloader, args.manifest_url, args.manifest_sha256,
            expected_artifact_id=args.artifact_id,
            expected_base_fingerprint=args.base_fingerprint,
            expected_artifact_fingerprint=args.artifact_fingerprint,
            headers={"Authorization": f"Bearer {token}"},
        ))
    except (ValueError, DragonflyDownloadError, DragonflyConfigError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_preseed_cli())
