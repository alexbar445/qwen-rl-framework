"""Route Worker policy downloads through size-checked Dragonfly object caches.

The Worker keeps its versioned ``/policy`` routes. Policy payloads use an opaque
key derived from the small manifest identity; generic content-addressed model
and image transfers remain in Dragonfly's strict SHA mode.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit

import httpx

from qwen_rl.contracts import worker as wire
from qwen_rl.transport.tensor_codec import (
    MAX_TENSOR_RAW_BYTES,
    TENSOR_PAYLOAD_CONTENT_TYPE,
    max_encoded_size,
)

from .artifacts import (
    ArtifactIntegrityError,
    _inspect_tensor_file,
    policy_file_cache_key,
)

if TYPE_CHECKING:
    from .dragonfly import DragonflyDownloader


_POLICY_METADATA_PATH = re.compile(r"^/policy(?:/(?P<version>[0-9]+))?$")
_POLICY_FILE_PATH = re.compile(r"^/policy/(?P<version>[0-9]+)/(?P<filename>[^/]+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BODY_REWRITTEN_HEADERS = frozenset({
    "content-encoding", "content-length", "transfer-encoding", "connection",
})
_MAX_POLICY_JSON_BYTES = 16 * 1024 * 1024


class _ManifestError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _ManifestError("Gateway policy manifest contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _ManifestError(f"Gateway policy manifest contains an invalid JSON constant: {value}")


@dataclass(frozen=True)
class _ManifestEntry:
    name: str
    sha256: str | None
    size_bytes: int | None
    cache_key: str
    url: str
    version: int


@dataclass(frozen=True)
class _CachedManifest:
    origin: tuple[str, str, int]
    version: int
    canonical: bytes
    files: dict[str, _ManifestEntry]


def _origin(url: str | httpx.URL) -> tuple[str, str, int]:
    parts = urlsplit(str(url))
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment):
        raise _ManifestError("Gateway URL must be an HTTP(S) origin without credentials or query")
    try:
        port = parts.port
    except ValueError as exc:
        raise _ManifestError("Gateway URL has an invalid port") from exc
    return parts.scheme.lower(), parts.hostname.lower(), port or (443 if parts.scheme.lower() == "https" else 80)


def _origin_url(origin: tuple[str, str, int]) -> str:
    scheme, host, port = origin
    default_port = 443 if scheme == "https" else 80
    host_for_url = f"[{host}]" if ":" in host else host
    port_text = "" if port == default_port else f":{port}"
    return f"{scheme}://{host_for_url}{port_text}"


def _validate_manifest(
    payload: Any,
    *,
    expected_version: int | None,
    metadata_url: httpx.URL,
) -> _CachedManifest:
    try:
        manifest = wire.PolicyManifest.model_validate(payload)
    except Exception as exc:
        raise _ManifestError("Gateway policy manifest did not match the Worker contract") from exc
    if expected_version is not None and manifest.version != expected_version:
        raise _ManifestError("Gateway policy version differs from the pinned request")
    expected_names = (wire.NATIVE_ADAPTER_FILES
                      if manifest.quantization.get("mode") == "ternary"
                      else wire.ADAPTER_FILES)
    if len(manifest.files) != len(expected_names) or {item.name for item in manifest.files} != expected_names:
        raise _ManifestError("Gateway policy manifest has the wrong adapter file set")

    origin = _origin(metadata_url)
    origin_url = _origin_url(origin)
    payload_dict = manifest.model_dump(mode="json")
    entries: dict[str, _ManifestEntry] = {}
    for item in manifest.files:
        if item.sha256 is not None and _SHA256.fullmatch(item.sha256.lower()) is None:
            raise _ManifestError("Gateway policy manifest contains a malformed legacy file SHA-256")
        if item.sha256 is None and item.size_bytes is None:
            raise _ManifestError("Gateway policy manifest omitted size for a nullable file digest")
        target = urljoin(f"{origin_url}/", item.url)
        try:
            target_origin = _origin(target)
        except _ManifestError as exc:
            raise _ManifestError("Gateway policy file URL is invalid") from exc
        target_parts = urlsplit(target)
        expected_path = f"/policy/{manifest.version}/{item.name}"
        if target_origin != origin or target_parts.path.rstrip("/") != expected_path:
            raise _ManifestError("Gateway policy file URL escaped its immutable versioned route")
        cache_key = policy_file_cache_key(payload_dict, item.name, origin=_origin_url(origin))
        entries[item.name] = _ManifestEntry(
            item.name, item.sha256, item.size_bytes, cache_key, item.url, manifest.version,
        )

    canonical = json.dumps(
        payload_dict, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return _CachedManifest(origin, manifest.version, canonical, entries)


def _json_error(request: httpx.Request, status_code: int, code: str) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"error": {"code": code}},
        headers={"content-type": "application/json"},
        request=request,
    )


class DragonflyWorkerTransport(httpx.AsyncBaseTransport):
    """HTTPX transport for Worker policy manifests and versioned policy files.

    ``downloader`` must be a worker-role DragonflyDownloader configured for the
    site Gateway. Metadata and policy bytes use versioned routes; payload cache
    keys come from manifest identity and size, not content digests.
    """

    def __init__(
        self,
        downloader: DragonflyDownloader,
        *,
        delegate: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if getattr(downloader, "site_role", None) != "worker":
            raise ValueError("Worker adapter transport requires a worker-role DragonflyDownloader")
        if not callable(getattr(downloader, "download_policy_object", None)):
            raise TypeError("downloader must expose async download_policy_object()")
        self.downloader = downloader
        self._delegate = delegate or httpx.AsyncHTTPTransport(retries=0)
        self._manifests: dict[tuple[tuple[str, str, int], int], _CachedManifest] = {}
        self._manifests_lock = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        # Paths are indexed by a manifest-derived object key, not file content.
        self._paths_by_key: dict[str, Any] = {}
        self._closed = False

    async def _object_lock(self, cache_key: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(cache_key, asyncio.Lock())

    async def _remember_manifest(self, manifest: _CachedManifest) -> None:
        key = (manifest.origin, manifest.version)
        async with self._manifests_lock:
            previous = self._manifests.get(key)
            if previous is not None and previous.canonical != manifest.canonical:
                raise _ManifestError("Gateway changed an immutable policy version manifest")
            self._manifests[key] = manifest

    async def _cached_manifest(
        self, origin: tuple[str, str, int], version: int,
    ) -> _CachedManifest | None:
        async with self._manifests_lock:
            return self._manifests.get((origin, version))

    async def _read_metadata(
        self,
        request: httpx.Request,
        *,
        expected_version: int | None,
    ) -> tuple[httpx.Response, _CachedManifest | None]:
        response = await self._delegate.handle_async_request(request)
        if response.status_code != 200:
            return response, None
        try:
            body = await response.aread()
            payload = json.loads(
                body,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
            manifest = _validate_manifest(
                payload,
                expected_version=expected_version,
                metadata_url=request.url,
            )
            await self._remember_manifest(manifest)
        except (ValueError, TypeError, UnicodeDecodeError):
            await response.aclose()
            return _json_error(request, 502, "invalid_policy_manifest"), None
        except httpx.HTTPError:
            await response.aclose()
            raise
        # The metadata response body is returned exactly as the Gateway sent
        # it; only an internal manifest index is created for file routing.
        return response, manifest

    async def _fetch_pinned_manifest(
        self,
        file_request: httpx.Request,
        *,
        origin: tuple[str, str, int],
        version: int,
    ) -> _CachedManifest | httpx.Response:
        origin_url = _origin_url(origin)
        pinned_url = f"{origin_url}/policy/{version}"
        metadata_request = httpx.Request(
            "GET",
            pinned_url,
            headers=file_request.headers,
        )
        response, manifest = await self._read_metadata(
            metadata_request,
            expected_version=version,
        )
        if response.status_code != 200 or manifest is None:
            body = await response.aread()
            headers = [
                (name, value)
                for name, value in response.headers.multi_items()
                if name.lower() not in _BODY_REWRITTEN_HEADERS
            ]
            result = httpx.Response(
                response.status_code,
                headers=headers,
                content=body,
                request=file_request,
            )
            await response.aclose()
            return result
        await response.aclose()
        return manifest

    async def _download_bytes(
        self,
        request: httpx.Request,
        *,
        origin: tuple[str, str, int],
        entry: _ManifestEntry,
    ) -> bytes:
        authorization_values = request.headers.get_list("authorization")
        if len(authorization_values) != 1 or not authorization_values[0]:
            raise httpx.TransportError("Worker artifact request lacks one authorization bearer")
        auth_header = authorization_values[0]
        lock = await self._object_lock(entry.cache_key)
        async with lock:
            is_tensor = not entry.name.endswith(".json")
            raw_limit = MAX_TENSOR_RAW_BYTES if entry.size_bytes is None else entry.size_bytes
            if is_tensor:
                if raw_limit > MAX_TENSOR_RAW_BYTES:
                    raise httpx.TransportError("Worker tensor manifest exceeds the raw size limit")
                wire_limit = max_encoded_size(raw_limit)
            else:
                if raw_limit > _MAX_POLICY_JSON_BYTES:
                    raise httpx.TransportError("Worker policy metadata exceeds the transfer limit")
                wire_limit = raw_limit
            path = self._paths_by_key.get(entry.cache_key)
            if path is None or not path.is_file() or path.is_symlink():
                origin_url = _origin_url(origin)
                route = f"/policy/{entry.version}/{entry.name}"
                artifact_url = f"{origin_url}{route}?policy_object_key={entry.cache_key}"
                try:
                    path = await self.downloader.download_policy_object(
                        artifact_url,
                        entry.cache_key,
                        None if is_tensor else entry.size_bytes,
                        headers={"Authorization": auth_header},
                        allow_lan_source_fallback=False,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise httpx.TransportError("LAN policy object transfer failed") from None
                if not hasattr(path, "is_file") or not path.is_file() or path.is_symlink():
                    raise httpx.TransportError("LAN policy cache did not return a regular object file")
                self._paths_by_key[entry.cache_key] = path
            try:
                info = await asyncio.to_thread(path.stat)
                if ((not is_tensor and entry.size_bytes is not None
                     and info.st_size != entry.size_bytes)
                        or (not is_tensor and info.st_size > wire_limit)
                        or (is_tensor and not 0 < info.st_size <= wire_limit)):
                    self._paths_by_key.pop(entry.cache_key, None)
                    raise httpx.TransportError("LAN policy object size differs from its transfer contract")
                if is_tensor:
                    try:
                        observed_raw_size = await asyncio.to_thread(
                            _inspect_tensor_file,
                            path,
                            max_raw_bytes=raw_limit,
                            safetensors=entry.name == "adapter_model.safetensors",
                        )
                    except (ArtifactIntegrityError, OSError, TypeError, ValueError, BufferError):
                        self._paths_by_key.pop(entry.cache_key, None)
                        raise httpx.TransportError(
                            "LAN policy tensor frame differs from its manifest contract",
                        ) from None
                    if entry.size_bytes is not None and observed_raw_size != entry.size_bytes:
                        self._paths_by_key.pop(entry.cache_key, None)
                        raise httpx.TransportError("LAN policy tensor raw size differs from its manifest")
                contents = await asyncio.to_thread(path.read_bytes)
            except OSError:
                self._paths_by_key.pop(entry.cache_key, None)
                raise httpx.TransportError("LAN policy cache could not read the object file") from None
            return contents

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._closed:
            raise httpx.TransportError("Worker artifact transport is closed")
        if request.method != "GET":
            return await self._delegate.handle_async_request(request)

        path = request.url.path
        metadata_match = _POLICY_METADATA_PATH.fullmatch(path)
        if metadata_match is not None:
            if request.url.query:
                return await self._delegate.handle_async_request(request)
            expected_version = metadata_match.group("version")
            if expected_version is not None:
                if len(expected_version) > 1 and expected_version.startswith("0"):
                    return _json_error(request, 404, "policy_version_unavailable")
                expected_version = int(expected_version)
            metadata_request = request
            response, _manifest = await self._read_metadata(
                metadata_request,
                expected_version=expected_version,
            )
            return response

        file_match = _POLICY_FILE_PATH.fullmatch(path)
        if file_match is None:
            return await self._delegate.handle_async_request(request)
        if request.url.query:
            return _json_error(request, 400, "adapter_file_query_not_allowed")

        raw_version = file_match.group("version")
        if len(raw_version) > 1 and raw_version.startswith("0"):
            return _json_error(request, 404, "policy_version_unavailable")
        version = int(raw_version)
        filename = file_match.group("filename")
        if filename not in wire.ADAPTER_FILES | wire.NATIVE_ADAPTER_FILES:
            return _json_error(request, 404, "adapter_file_unavailable")
        origin = _origin(request.url)
        manifest = await self._cached_manifest(origin, version)
        if manifest is None:
            manifest_or_response = await self._fetch_pinned_manifest(
                request,
                origin=origin,
                version=version,
            )
            if isinstance(manifest_or_response, httpx.Response):
                return manifest_or_response
            manifest = manifest_or_response
        entry = manifest.files.get(filename)
        if entry is None:
            return _json_error(request, 404, "adapter_file_unavailable")
        try:
            contents = await self._download_bytes(request, origin=origin, entry=entry)
        except asyncio.CancelledError:
            raise
        except httpx.TransportError:
            raise
        headers = {
            "Content-Type": (
                TENSOR_PAYLOAD_CONTENT_TYPE if not entry.name.endswith(".json")
                else "application/json"
            ),
            "Content-Length": str(len(contents)),
            "ETag": f'W/"policy-object-{entry.cache_key}"',
        }
        return httpx.Response(200, headers=headers, content=contents, request=request)

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._delegate.aclose()


def create_worker_http_transport(
    downloader: DragonflyDownloader,
    *,
    delegate: httpx.AsyncBaseTransport | None = None,
) -> DragonflyWorkerTransport:
    """Build the HTTPX transport passed to ``create_worker_app``."""
    return DragonflyWorkerTransport(downloader, delegate=delegate)
