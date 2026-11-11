"""Verify or download the model files declared by a private Vast manifest.

The module uses only the Python standard library and the preinstalled ``curl``
binary. Runtime downloads are resumable and published only after both declared
size and SHA-256 match. URLs are passed to curl through stdin, never argv or
diagnostics, because HTTPS URLs may carry short-lived credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

MANIFEST_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_BYTES = 8 * 1024 * 1024
_DEFAULT_RETRIES = 3
_DEFAULT_CONNECT_TIMEOUT = 30
_DEFAULT_MAX_TIME = 21600
_DEFAULT_RETRY_WINDOW = 180


class DownloadError(RuntimeError):
    """A safe-to-display validation, cache, or transfer failure."""

    def __init__(self, message: str, *, code: str = "invalid_manifest") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ModelManifestFile:
    path: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    files: tuple[ModelManifestFile, ...]
    sha256: str
    total_size: int
    largest_size: int


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _safe_path(value: Any) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise DownloadError("manifest contains an invalid model path")
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts or str(path) != value
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(part == ".qwen-vast-state" for part in path.parts)):
        raise DownloadError("manifest model path must be a normalized relative path")
    return path.as_posix()


def _safe_https_url(value: Any, *, allow_http_for_tests: bool = False) -> str:
    if (not isinstance(value, str) or not value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise DownloadError("manifest contains an invalid model URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise DownloadError("manifest contains an invalid model URL") from exc
    allowed_schemes = {"https"}
    if allow_http_for_tests:
        allowed_schemes.add("http")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    placeholder_hosts = {"example.com", "example.net", "example.org"}
    if (parsed.scheme not in allowed_schemes or not parsed.hostname or parsed.username
            or parsed.password or parsed.fragment or port == 0
            or hostname in placeholder_hosts
            or hostname.endswith((".invalid", ".example", ".test"))):
        raise DownloadError("manifest model URLs must use credential-free HTTPS")
    return value


def parse_manifest(payload: bytes | str, *, allow_http_for_tests: bool = False) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = payload.encode("utf-8") if isinstance(payload, str) else payload
        manifest = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise DownloadError("model manifest is not valid strict JSON") from exc
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}
            or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or isinstance(manifest.get("schema_version"), bool)):
        raise DownloadError("model manifest does not match schema version 1")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise DownloadError("model manifest must list at least one file")
    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in records:
        if not isinstance(item, dict) or set(item) != {"path", "url", "sha256", "size"}:
            raise DownloadError("model manifest file entry does not match the strict schema")
        path = _safe_path(item["path"])
        url = _safe_https_url(item["url"], allow_http_for_tests=allow_http_for_tests)
        sha256 = item["sha256"]
        size = item["size"]
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
            raise DownloadError("model manifest contains an invalid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise DownloadError("model manifest file size must be a positive integer")
        if path in seen_paths:
            raise DownloadError("model manifest contains duplicate paths")
        seen_paths.add(path)
        files.append({"path": path, "url": url, "sha256": sha256, "size": size})

    # A file cannot also be another file's parent or curl's .part staging path.
    path_set = set(seen_paths)
    for name in seen_paths:
        if f"{name}.part" in path_set:
            raise DownloadError("model manifest file paths conflict with resumable staging")
        if any(other.startswith(name + "/") for other in path_set):
            raise DownloadError("model manifest file paths overlap")
    return files, hashlib.sha256(raw).hexdigest()


def parse_manifest_bytes(payload: bytes, expected_sha256: str | None = None) -> ModelManifest:
    """Parse exact manifest bytes for CPU-side deployment profile validation."""
    if not isinstance(payload, bytes):
        raise DownloadError("model manifest payload must be bytes")
    files, observed_sha = parse_manifest(payload)
    if expected_sha256 is not None:
        if _SHA256.fullmatch(expected_sha256) is None:
            raise DownloadError("expected manifest SHA-256 is invalid")
        if not hmac.compare_digest(observed_sha, expected_sha256):
            raise DownloadError("model manifest SHA-256 does not match the pinned digest",
                                code="integrity_failed")
    typed_files = tuple(ModelManifestFile(**item) for item in files)
    return ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        files=typed_files,
        sha256=observed_sha,
        total_size=sum(item.size for item in typed_files),
        largest_size=max(item.size for item in typed_files),
    )


def _private_directory(path: Path) -> None:
    try:
        if path.is_symlink():
            raise DownloadError("private directory must not be a symlink", code="unsafe_path")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.lstat()
    except DownloadError:
        raise
    except OSError as exc:
        raise DownloadError("cannot prepare a private directory", code="unsafe_path") from exc
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise DownloadError("private directory has an unexpected owner or type", code="unsafe_path")
    if stat.S_IMODE(info.st_mode) != 0o700:
        try:
            path.chmod(0o700)
        except OSError as exc:
            raise DownloadError("private directory permissions are unsafe", code="unsafe_path") from exc


def _private_root(value: str | Path) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute() or ".." in root.parts:
        raise DownloadError("model root must be an absolute path without '..'", code="unsafe_path")
    _private_directory(root)
    return root


def _private_receipt_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise DownloadError("receipt path must be absolute and normalized", code="unsafe_path")
    return path


def _ensure_child_directory(path: Path) -> None:
    if path.is_symlink():
        raise DownloadError("model path crosses a symlink", code="unsafe_path")
    _private_directory(path)


def _destination(root: Path, relative: str, *, create_parents: bool) -> Path:
    current = root
    for component in PurePosixPath(relative).parts[:-1]:
        current = current / component
        if create_parents:
            _ensure_child_directory(current)
        elif current.is_symlink():
            raise DownloadError("cached model path has an unsafe parent", code="unsafe_path")
        elif not current.exists():
            return root / PurePosixPath(relative)
        elif not current.is_dir():
            raise DownloadError("cached model path has an unsafe parent", code="unsafe_path")
    target = current / PurePosixPath(relative).name
    if target.is_symlink():
        raise DownloadError("model file must not be a symlink", code="unsafe_path")
    return target


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _check_file(path: Path, *, sha256: str, size: int) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        raise DownloadError("cached model path is not a private regular file", code="unsafe_path")
    if stat.S_IMODE(info.st_mode) != 0o600:
        try:
            path.chmod(0o600)
        except OSError as exc:
            raise DownloadError("cached model file permissions are unsafe", code="unsafe_path") from exc
    if info.st_size != size:
        return False
    try:
        observed_sha, observed_size = _hash_file(path)
    except OSError as exc:
        raise DownloadError("cannot verify cached model file", code="unsafe_path") from exc
    return observed_size == size and hmac.compare_digest(observed_sha, sha256)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    parent = path.parent
    _private_directory(parent)
    if path.is_symlink():
        raise DownloadError("receipt path must not be a symlink", code="unsafe_path")
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        _sync_directory(parent)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise DownloadError("cannot write the private download receipt", code="receipt_error") from exc


def _curl_quote(value: str) -> str:
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise DownloadError("curl argument contains control characters", code="unsafe_path")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _curl_config(
    *, url: str, output: Path, resume: bool, retries: int,
    connect_timeout: int, max_time: int, retry_window: int,
    allow_http_for_tests: bool,
) -> bytes:
    protocols = "=http,https" if allow_http_for_tests else "=https"
    lines = [
        f"url = {_curl_quote(url)}",
        f"output = {_curl_quote(str(output))}",
        "fail", "location", "silent", "show-error",
        f"proto = {_curl_quote(protocols)}",
        f"proto-redir = {_curl_quote(protocols)}",
        "max-redirs = 10",
        f"connect-timeout = {connect_timeout}",
        f"max-time = {max_time}",
        f"retry = {retries}",
        f"retry-max-time = {retry_window}",
    ]
    if retries:
        lines.extend(("retry-all-errors", "retry-delay = 1"))
    if resume:
        lines.append('continue-at = "-"')
    return ("\n".join(lines) + "\n").encode("utf-8")


def _curl_transfer(
    *, curl_binary: str, url: str, part: Path, resume: bool, retries: int,
    connect_timeout: int, max_time: int, retry_window: int,
    allow_http_for_tests: bool,
) -> int:
    config = _curl_config(
        url=url, output=part, resume=resume, retries=retries,
        connect_timeout=connect_timeout, max_time=max_time,
        retry_window=retry_window, allow_http_for_tests=allow_http_for_tests,
    )
    safe_env = os.environ.copy()
    safe_env.pop("ARC_ENV_TOKEN", None)
    safe_env.pop("ARC_ENV_TOKEN_FILE", None)
    safe_env["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            [curl_binary, "--disable", "--config", "-"], input=config,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=max_time + retry_window + connect_timeout + 30,
            check=False, env=safe_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DownloadError("curl could not complete the bounded model transfer", code="transfer_failed") from exc
    return result.returncode


def _download_one(
    *, root: Path, item: dict[str, Any], curl_binary: str, retries: int,
    connect_timeout: int, max_time: int, retry_window: int,
    allow_http_for_tests: bool,
) -> tuple[Path, float]:
    target = _destination(root, item["path"], create_parents=True)
    part = target.with_name(target.name + ".part")
    if part.is_symlink():
        raise DownloadError("partial model file must not be a symlink", code="unsafe_path")
    if _check_file(target, sha256=item["sha256"], size=item["size"]):
        return target, 0.0
    try:
        part_info = part.lstat()
    except FileNotFoundError:
        part_info = None
    if part_info is not None:
        if not stat.S_ISREG(part_info.st_mode) or part_info.st_uid != os.geteuid():
            raise DownloadError("partial model path is not a private regular file", code="unsafe_path")
        if part_info.st_size > item["size"]:
            part.unlink()
            part_info = None
        elif stat.S_IMODE(part_info.st_mode) != 0o600:
            part.chmod(0o600)
    if part_info is None:
        descriptor = os.open(
            part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        os.close(descriptor)
    resume = part.stat().st_size > 0
    started = time.monotonic()
    code = _curl_transfer(
        curl_binary=curl_binary, url=item["url"], part=part, resume=resume,
        retries=retries, connect_timeout=connect_timeout, max_time=max_time,
        retry_window=retry_window, allow_http_for_tests=allow_http_for_tests,
    )
    if code == 33 and resume:
        # Some HTTPS origins ignore Range. Restart once from zero, still under
        # the same bounded curl retry and transfer-time limits.
        part.unlink(missing_ok=True)
        descriptor = os.open(
            part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        os.close(descriptor)
        code = _curl_transfer(
            curl_binary=curl_binary, url=item["url"], part=part, resume=False,
            retries=retries, connect_timeout=connect_timeout, max_time=max_time,
            retry_window=retry_window, allow_http_for_tests=allow_http_for_tests,
        )
    if code != 0:
        raise DownloadError("curl model transfer failed", code="transfer_failed")
    try:
        with part.open("rb") as completed:
            os.fsync(completed.fileno())
        observed_sha, observed_size = _hash_file(part)
    except OSError as exc:
        raise DownloadError("downloaded model file could not be verified", code="integrity_failed") from exc
    if (observed_size != item["size"]
            or not hmac.compare_digest(observed_sha, item["sha256"])):
        part.unlink(missing_ok=True)
        raise DownloadError("downloaded model file does not match its declared size and SHA-256",
                            code="integrity_failed")
    try:
        os.replace(part, target)
        target.chmod(0o600)
        _sync_directory(target.parent)
    except OSError as exc:
        raise DownloadError("verified model file could not be atomically published",
                            code="publish_failed") from exc
    return target, time.monotonic() - started


def validate_worker_config(
    config_path: str | Path, *, manifest_files: Sequence[dict[str, Any]], root: Path,
) -> int:
    """Check that the prepared Worker binds locally and references manifest files."""
    path = Path(config_path)
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            raise DownloadError("Worker config must be a private regular file", code="worker_config_invalid")
        config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=_reject_json_constant)
    except DownloadError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DownloadError("Worker config is invalid or unavailable", code="worker_config_invalid") from exc
    if not isinstance(config, dict) or config.get("host") != "127.0.0.1":
        raise DownloadError("Worker config must bind to 127.0.0.1", code="worker_config_invalid")
    port = config.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise DownloadError("Worker config has an invalid port", code="worker_config_invalid")
    inference = config.get("inference")
    if not isinstance(inference, dict) or inference.get("backend") != "ninfer":
        raise DownloadError("Worker config must explicitly select the NInfer backend",
                            code="worker_config_invalid")
    model_path = inference.get("model_path")
    if not isinstance(model_path, str) or not Path(model_path).is_absolute():
        raise DownloadError("Worker config model path must be absolute", code="worker_config_invalid")
    root_resolved = root.resolve(strict=False)
    model_resolved = Path(model_path).resolve(strict=False)
    if model_resolved != root_resolved:
        raise DownloadError("Worker model path must match the prepared model root",
                            code="worker_config_invalid")
    listed = {item["path"] for item in manifest_files}
    for key in ("ninfer_artifact_path", "ninfer_conversion_report"):
        configured = inference.get(key)
        if not isinstance(configured, str) or not Path(configured).is_absolute():
            raise DownloadError(f"Worker config is missing absolute {key}", code="worker_config_invalid")
        configured_path = Path(configured).resolve(strict=False)
        try:
            relative = configured_path.relative_to(root_resolved).as_posix()
        except ValueError as exc:
            raise DownloadError("Worker model artifact path escapes the model root",
                                code="worker_config_invalid") from exc
        if relative not in listed:
            raise DownloadError("Worker config references a model file absent from the manifest",
                                code="worker_config_invalid")
    return port


def _read_manifest(path: str | Path, *, allow_http_for_tests: bool) -> tuple[list[dict[str, Any]], str]:
    source = Path(path)
    try:
        info = source.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            raise DownloadError("model manifest must be a private regular file", code="unsafe_path")
        payload = source.read_bytes()
    except DownloadError:
        raise
    except OSError as exc:
        raise DownloadError("model manifest is unavailable", code="manifest_unavailable") from exc
    return parse_manifest(payload, allow_http_for_tests=allow_http_for_tests)


def download_manifest(
    manifest_path: str | Path,
    root: str | Path,
    *,
    offline: bool = False,
    receipt_path: str | Path | None = None,
    worker_config: str | Path | None = None,
    expected_sha256: str | None = None,
    curl_binary: str = "curl",
    retries: int = _DEFAULT_RETRIES,
    connect_timeout: int = _DEFAULT_CONNECT_TIMEOUT,
    max_time: int = _DEFAULT_MAX_TIME,
    retry_window: int = _DEFAULT_RETRY_WINDOW,
    allow_http_for_tests: bool = False,
) -> dict[str, Any]:
    """Download and verify a manifest, or in offline mode only verify its cache."""
    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (retries, connect_timeout, max_time, retry_window))
            or connect_timeout == 0 or max_time == 0):
        raise DownloadError("curl retry and timeout limits must be positive", code="invalid_options")
    files, manifest_sha = _read_manifest(manifest_path, allow_http_for_tests=allow_http_for_tests)
    if expected_sha256 is not None:
        if _SHA256.fullmatch(expected_sha256) is None:
            raise DownloadError("expected manifest SHA-256 is invalid", code="integrity_failed")
        if not hmac.compare_digest(manifest_sha, expected_sha256):
            raise DownloadError("model manifest SHA-256 does not match the pinned digest",
                                code="integrity_failed")
    model_root = _private_root(root)
    if worker_config is not None:
        validate_worker_config(worker_config, manifest_files=files, root=model_root)
    output_receipt = (_private_receipt_path(receipt_path) if receipt_path is not None
                      else model_root / ".qwen-vast-state" / "download-receipt.json")
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    cold_seconds = 0.0
    warm_seconds = 0.0
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha,
        "status": "in_progress",
        "mode": "offline" if offline else "online",
        "files": results,
    }
    try:
        for item in files:
            file_started = time.monotonic()
            target = _destination(model_root, item["path"], create_parents=False)
            if _check_file(target, sha256=item["sha256"], size=item["size"]):
                duration = time.monotonic() - file_started
                warm_seconds += duration
                results.append({"path": item["path"], "size": item["size"],
                                "sha256": item["sha256"], "cache": "warm",
                                "duration_seconds": duration})
                continue
            if offline:
                raise DownloadError("offline model cache is missing or does not match the manifest",
                                    code="cache_miss")
            target, duration = _download_one(
                root=model_root, item=item, curl_binary=curl_binary, retries=retries,
                connect_timeout=connect_timeout, max_time=max_time,
                retry_window=retry_window, allow_http_for_tests=allow_http_for_tests,
            )
            cold_seconds += duration
            results.append({"path": item["path"], "size": item["size"],
                            "sha256": item["sha256"], "cache": "cold",
                            "duration_seconds": duration})
        receipt["status"] = "verified"
    except DownloadError as exc:
        receipt["status"] = "failed"
        receipt["failure_code"] = exc.code
        raise
    finally:
        receipt["timings_seconds"] = {
            "cold": cold_seconds,
            "warm": warm_seconds,
            "total": time.monotonic() - started,
        }
        try:
            _write_private_json(output_receipt, receipt)
        except DownloadError:
            if receipt["status"] == "verified":
                raise
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="private schema-v1 model manifest JSON")
    parser.add_argument("--root", required=True, help="absolute private model cache root")
    parser.add_argument("--offline", action="store_true", help="only verify cached files; do not contact URLs")
    parser.add_argument("--receipt", help="private receipt path; defaults inside the model cache")
    parser.add_argument("--worker-config", help="validate that NInfer model paths match the manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = download_manifest(
            args.manifest, args.root, offline=args.offline,
            receipt_path=args.receipt, worker_config=args.worker_config,
        )
    except DownloadError as exc:
        print(f"qwen-vast-download: {exc}", file=sys.stderr)
        return 2
    warm = sum(item["cache"] == "warm" for item in receipt["files"])
    cold = sum(item["cache"] == "cold" for item in receipt["files"])
    print(f"verified {len(receipt['files'])} model files ({cold} downloaded, {warm} cached)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
