"""Fixed, read-only HTTP upstreams for the dashboard source selector."""
from __future__ import annotations

import ipaddress
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

PROFILE_LIMIT = 64 * 1024
RESPONSE_LIMIT = 2 * 1024 * 1024
MAX_SOURCES = 16
CACHE_ENTRY_LIMIT = 8
CACHE_BYTES_LIMIT = 8 * 1024 * 1024
SNAPSHOT_LIMIT = 2 * 1024 * 1024
WORKSPACE_SCHEMA = "qwen-rl-dashboard-v2"
SNAPSHOT_META_FIELD = "_snapshot_meta"
WORKSPACE_FIELDS = frozenset({
    "schema", "build", "code_sha256", "generated_at", "source", "selection", "runs",
    "watch", "history", "run", "bepa", "evaluation", "recovery", "events", "insights",
    "training_pipeline", "activity", "pipeline", "gpus", "workspace_age_seconds",
    "source_id", "availability", "stale", "observed_age_seconds", "last_observed_logical_run",
})
_SOURCE_ID = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_CREDENTIAL_FIELDS = frozenset({
    "token", "secret", "password", "credential", "credentials", "authorization", "cookie",
    "api_key", "private_key",
})
_CREDENTIAL_SUFFIXES = (
    "_token", "_secret", "_password", "_credential", "_credentials", "_api_key",
    "_private_key", "_authorization", "_cookie",
)
_BEARER = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")
_KEY_VALUE = re.compile(
    r"(?i)\b(?P<key>[A-Za-z][A-Za-z0-9_.-]*)(?P<key_quote>[\"']?)"
    r"(?P<separator>\s*[:=]\s*)(?P<value>\"[^\"]*\"|'[^']*'|[^\s\"',;}]+)"
)

REMOTE_ROUTES: dict[str, frozenset[str]] = {
    "/api/health": frozenset(),
    "/api/workspace": frozenset({"run_id"}),
    "/api/overview": frozenset(),
    "/api/watch": frozenset(),
    "/api/runs": frozenset(),
    "/api/run": frozenset({"run_id"}),
    "/api/trajectory": frozenset({"run_id", "name"}),
    "/api/bepa": frozenset(),
    "/api/artifact": frozenset({"ref", "run_id"}),
    "/api/logs": frozenset({"run_id", "role", "tail"}),
    "/api/gpu": frozenset(),
}


class UpstreamUnavailable(RuntimeError):
    """The fixed upstream did not provide a bounded JSON read response."""


def _validate_url(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("upstream url must be a bounded loopback URL")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid upstream url") from exc
    if (parsed.scheme not in {"http", "https"} or not host or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("upstream url must not contain credentials, path, query, or fragment")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if not loopback:
        raise ValueError("upstream host must be loopback")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("invalid upstream port")
    return value.rstrip("/")


def _clean_string(value: str) -> str:
    value = _BEARER.sub(r"\1 ***", value)
    def redact_pair(match: re.Match[str]) -> str:
        if not _is_credential_name(match.group("key")):
            return match.group(0)
        value = match.group("value")
        replacement = (value[0] + "***" + value[0]
                       if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]
                       else "***")
        return (match.group("key") + match.group("key_quote") + match.group("separator")
                + replacement)
    value = _KEY_VALUE.sub(redact_pair, value)
    return re.sub(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@", r"\1***@", value)


def _is_credential_name(name: str) -> bool:
    """Match credential field names without swallowing token-count metrics."""
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized in _CREDENTIAL_FIELDS or normalized.endswith(_CREDENTIAL_SUFFIXES)


def _valid_workspace_shape(payload: Any) -> bool:
    if not isinstance(payload, dict) or payload.get("schema") != WORKSPACE_SCHEMA:
        return False
    if set(payload) - WORKSPACE_FIELDS - {SNAPSHOT_META_FIELD}:
        return False
    required = {"schema", "generated_at", "source", "selection", "runs", "watch", "history", "run"}
    if not required <= set(payload):
        return False
    if not isinstance(payload.get("generated_at"), (int, float)) or not math.isfinite(payload["generated_at"]):
        return False
    if not all(isinstance(payload.get(name), dict) for name in ("source", "selection", "watch")):
        return False
    if not all(isinstance(payload.get(name), list) for name in ("runs", "history")):
        return False
    if payload.get("run") is not None and not isinstance(payload.get("run"), dict):
        return False
    for name in ("events", "insights", "training_pipeline", "pipeline"):
        if name in payload and not isinstance(payload[name], list):
            return False
    if "gpus" in payload and payload["gpus"] is not None and not isinstance(payload["gpus"], list):
        return False
    if "activity" in payload and payload["activity"] is not None and not isinstance(payload["activity"], dict):
        return False
    logical = payload.get("last_observed_logical_run")
    if logical is not None:
        if (not isinstance(logical, dict)
                or set(logical) != {"run", "provenance", "observed_at", "runtime_run_id"}
                or not isinstance(logical.get("run"), dict)
                or not isinstance(logical.get("provenance"), dict)
                or not isinstance(logical.get("runtime_run_id"), str)
                or not logical["runtime_run_id"]
                or not isinstance(logical.get("observed_at"), (int, float))
                or not math.isfinite(logical["observed_at"])):
            return False
    return True


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_private_json(path: Path) -> tuple[dict[str, Any], os.stat_result] | None:
    try:
        if not path.is_absolute() or not path.parent.is_dir() or path.parent.resolve(strict=True) != path.parent:
            return None
    except OSError:
        return None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > SNAPSHOT_LIMIT):
            return None
        pieces: list[bytes] = []
        remaining = SNAPSHOT_LIMIT + 1
        while remaining:
            block = os.read(fd, min(65536, remaining))
            if not block:
                break
            pieces.append(block)
            remaining -= len(block)
        data = b"".join(pieces)
        if len(data) > SNAPSHOT_LIMIT:
            return None
        parsed = json.loads(data, object_pairs_hook=_strict_object)
        return (parsed, info) if isinstance(parsed, dict) else None
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    finally:
        os.close(fd)


def _atomic_private_json(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("snapshot_path must be absolute and normalized")
    parent = path.parent
    if not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise ValueError("snapshot parent must be an existing real directory")
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None and (not stat.S_ISREG(existing.st_mode) or existing.st_uid != os.geteuid()
                                 or stat.S_IMODE(existing.st_mode) != 0o600):
        raise ValueError("existing snapshot must be an owner-owned regular mode-0600 file")
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > SNAPSHOT_LIMIT:
        raise ValueError("snapshot exceeds size limit")
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def sanitize_payload(value: Any) -> Any:
    """Remove credential-bearing fields and redact common inline credential forms."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            result[name] = "***" if _is_credential_name(name) else sanitize_payload(item)
        return result
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return _clean_string(value)
    return value


def _bounded_query(path: str, query: Mapping[str, list[str]]) -> dict[str, str]:
    allowed = REMOTE_ROUTES.get(path)
    if allowed is None:
        raise LookupError("no such endpoint")
    keys = set(query) - {"source"}
    if keys - allowed:
        raise ValueError("unsupported upstream query parameter")
    result: dict[str, str] = {}
    for key in keys:
        values = query[key]
        if len(values) != 1:
            raise ValueError("duplicate upstream query parameter")
        value = values[0]
        if len(value) > 1024 or any(ord(char) < 32 for char in value):
            raise ValueError("upstream query parameter is too large or invalid")
        result[key] = value
    if len(result) > 8:
        raise ValueError("too many upstream query parameters")
    return result


@dataclass(frozen=True)
class UpstreamSpec:
    id: str
    label: str
    url: str
    snapshot_path: Path | None = None


class RemoteUpstream:
    def __init__(self, spec: UpstreamSpec, *, transport: httpx.BaseTransport | None = None) -> None:
        self.spec = spec
        self._client = httpx.Client(
            trust_env=False,
            timeout=httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            follow_redirects=False,
            transport=transport,
        )
        self._cache_lock = threading.Lock()
        self._last_success: OrderedDict[tuple[str, tuple[tuple[str, str], ...]], tuple[float, Any, int]] = OrderedDict()
        self._cache_bytes = 0
        self._load_persisted_snapshot()

    def get(self, path: str, query: Mapping[str, list[str]]) -> tuple[int, Any]:
        params = _bounded_query(path, query)
        cache_key = (path, tuple(sorted(params.items())))
        try:
            with self._client.stream("GET", self.spec.url + path, params=params,
                                     headers={"Accept": "application/json"}) as response:
                if response.status_code >= 500 or response.is_redirect:
                    raise UpstreamUnavailable("upstream unavailable")
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > RESPONSE_LIMIT:
                        raise UpstreamUnavailable("upstream response exceeded size limit")
                try:
                    payload = json.loads(content)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise UpstreamUnavailable("upstream returned invalid JSON") from exc
                payload = sanitize_payload(payload)
                if (response.status_code == 200 and isinstance(payload, dict)
                        and path in {"/api/workspace", "/api/overview"}
                        and payload.get("stale") is not True
                        and payload.get("availability") != "unavailable"):
                    payload = self._remember(cache_key, payload, len(content))
                return response.status_code, payload
        except UpstreamUnavailable:
            stale = self._stale_copy(path, cache_key)
            if stale is not None:
                return 200, stale
            raise
        except (httpx.HTTPError, OSError) as exc:
            stale = self._stale_copy(path, cache_key)
            if stale is not None:
                return 200, stale
            raise UpstreamUnavailable("upstream unavailable") from exc

    def _remember(self, key: tuple[str, tuple[tuple[str, str], ...]], payload: Any, size: int) -> Any:
        if not isinstance(payload, dict) or size > CACHE_BYTES_LIMIT:
            return payload
        observed_at = time.time()
        payload = deepcopy(payload)
        if key[0] == "/api/workspace":
            payload["source_id"] = self.spec.id
            projection = self._project_last_observed(payload, observed_at)
            payload.pop("last_observed_logical_run", None)
            if projection is not None:
                payload["last_observed_logical_run"] = projection
        try:
            encoded_size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                          separators=(",", ":")).encode("utf-8"))
        except (ValueError, TypeError):
            return payload
        if encoded_size > CACHE_BYTES_LIMIT:
            return payload
        with self._cache_lock:
            previous = self._last_success.pop(key, None)
            if previous is not None:
                self._cache_bytes -= previous[2]
            self._last_success[key] = (observed_at, deepcopy(payload), encoded_size)
            self._cache_bytes += encoded_size
            while (len(self._last_success) > CACHE_ENTRY_LIMIT
                   or self._cache_bytes > CACHE_BYTES_LIMIT):
                _evicted_key, evicted = self._last_success.popitem(last=False)
                self._cache_bytes -= evicted[2]
        if key[0] == "/api/workspace" and self.spec.snapshot_path is not None:
            self._persist_snapshot(key, payload, observed_at)
        return payload

    @staticmethod
    def _project_last_observed(payload: dict[str, Any], observed_at: float) -> dict[str, Any] | None:
        watch = payload.get("watch") or {}
        if not isinstance(watch, dict):
            return None
        snapshot = watch.get("snapshot") if isinstance(watch, dict) else None
        learner = snapshot.get("learner") if isinstance(snapshot, dict) else None
        logical_run = learner.get("run") if isinstance(learner, dict) else None
        selection = payload.get("selection") or {}
        lifecycle = snapshot.get("lifecycle") if isinstance(snapshot, dict) else None
        runtime_run_id = ((selection.get("observed_runtime_run_id") if isinstance(selection, dict) else None)
                          or (lifecycle.get("run_id") if isinstance(lifecycle, dict) else None)
                          or (learner.get("runtime_run_id") if isinstance(learner, dict) else None))
        selected_run_id = selection.get("run_id") if isinstance(selection, dict) else None
        if (watch.get("ok") is not True or not isinstance(logical_run, dict)
                or not isinstance(runtime_run_id, str) or not runtime_run_id
                or (selected_run_id and selected_run_id != runtime_run_id)):
            return None
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        source_identity = selection.get("source_identity") if isinstance(selection, dict) else None
        return {
            "run": deepcopy(logical_run),
            "provenance": {
                "source_id": payload.get("source_id"),
                "source_kind": source.get("kind"),
                "deployment_id": source.get("deployment_id"),
                "source_identity": deepcopy(source_identity),
                "code_sha256": payload.get("code_sha256"),
            },
            "observed_at": observed_at,
            "runtime_run_id": runtime_run_id,
        }

    def _persist_snapshot(self, key: tuple[str, tuple[tuple[str, str], ...]],
                          payload: dict[str, Any], observed_at: float) -> None:
        path = self.spec.snapshot_path
        if path is None or not _valid_workspace_shape(payload):
            return
        old_observation = payload.get("last_observed_logical_run")
        runtime_run_id = old_observation.get("runtime_run_id") if isinstance(old_observation, dict) else None
        if not isinstance(runtime_run_id, str):
            runtime_run_id = self._workspace_runtime_run_id(payload)
        document = deepcopy(payload)
        document[SNAPSHOT_META_FIELD] = {
            "schema": WORKSPACE_SCHEMA,
            "source_id": self.spec.id,
            "endpoint_sha256": hashlib.sha256(self.spec.url.encode("utf-8")).hexdigest(),
            "path": key[0],
            "query": dict(key[1]),
            "observed_at": observed_at,
            "runtime_run_id": runtime_run_id,
        }
        try:
            _atomic_private_json(path, document)
        except (OSError, ValueError, TypeError):
            return

    @staticmethod
    def _workspace_runtime_run_id(payload: dict[str, Any]) -> str | None:
        selection = payload.get("selection") or {}
        watch = payload.get("watch") or {}
        snapshot = watch.get("snapshot") if isinstance(watch, dict) else None
        lifecycle = snapshot.get("lifecycle") if isinstance(snapshot, dict) else None
        learner = snapshot.get("learner") if isinstance(snapshot, dict) else None
        candidate = ((selection.get("observed_runtime_run_id") if isinstance(selection, dict) else None)
                     or (lifecycle.get("run_id") if isinstance(lifecycle, dict) else None)
                     or (learner.get("runtime_run_id") if isinstance(learner, dict) else None)
                     or (selection.get("run_id") if isinstance(selection, dict) else None))
        return candidate if isinstance(candidate, str) and candidate else None

    def _load_persisted_snapshot(self) -> None:
        path = self.spec.snapshot_path
        if path is None:
            return
        loaded = _read_private_json(path)
        if loaded is None:
            return
        document, file_info = loaded
        meta = document.pop(SNAPSHOT_META_FIELD, None)
        if (not isinstance(meta, dict)
                or set(meta) != {"schema", "source_id", "endpoint_sha256", "path", "query", "observed_at", "runtime_run_id"}
                or meta.get("schema") != WORKSPACE_SCHEMA
                or meta.get("source_id") != self.spec.id
                or meta.get("endpoint_sha256") != hashlib.sha256(self.spec.url.encode("utf-8")).hexdigest()
                or meta.get("path") != "/api/workspace"
                or document.get("source_id") != self.spec.id
                or not _valid_workspace_shape(document)):
            return
        query = meta.get("query")
        if not isinstance(query, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in query.items()):
            return
        try:
            bounded = _bounded_query("/api/workspace", {key: [value] for key, value in query.items()})
        except (ValueError, LookupError):
            return
        observed_at = meta.get("observed_at")
        if not isinstance(observed_at, (int, float)) or not math.isfinite(observed_at) or observed_at <= 0:
            return
        runtime_run_id = meta.get("runtime_run_id")
        if runtime_run_id is not None and not isinstance(runtime_run_id, str):
            return
        selected_run_id = (document.get("selection") or {}).get("run_id")
        if bounded.get("run_id") and selected_run_id and bounded["run_id"] != selected_run_id:
            return
        recorded = document.get("last_observed_logical_run")
        if isinstance(recorded, dict):
            if runtime_run_id and recorded.get("runtime_run_id") != runtime_run_id:
                return
            if selected_run_id and recorded.get("runtime_run_id") != selected_run_id:
                document.pop("last_observed_logical_run", None)
        elif runtime_run_id and selected_run_id and runtime_run_id != selected_run_id:
            return
        document = sanitize_payload(document)
        document["source_id"] = self.spec.id
        if not isinstance(recorded, dict):
            projection = self._project_last_observed(document, float(observed_at))
            if projection is not None:
                document["last_observed_logical_run"] = projection
        key = ("/api/workspace", tuple(sorted(bounded.items())))
        size = file_info.st_size
        with self._cache_lock:
            aliases = [key]
            selection = document.get("selection") or {}
            selected = selection.get("run_id") if isinstance(selection, dict) else None
            active = selection.get("active_run_id") if isinstance(selection, dict) else None
            if isinstance(selected, str) and selected and selected == active:
                aliases.append(("/api/workspace", (("run_id", selected),)))
            for alias in set(aliases):
                self._last_success[alias] = (float(observed_at), deepcopy(document), size)
                self._cache_bytes += size
            while (len(self._last_success) > CACHE_ENTRY_LIMIT
                   or self._cache_bytes > CACHE_BYTES_LIMIT):
                _evicted_key, evicted = self._last_success.popitem(last=False)
                self._cache_bytes -= evicted[2]

    def _stale_copy(self, path: str,
                    key: tuple[str, tuple[tuple[str, str], ...]]) -> dict[str, Any] | None:
        if path not in {"/api/workspace", "/api/overview"}:
            return None
        with self._cache_lock:
            cached = self._last_success.get(key)
            if cached is None:
                return None
            self._last_success.move_to_end(key)
            observed_age = max(0.0, time.time() - cached[0])
            payload = deepcopy(cached[1])
        if not isinstance(payload, dict):
            return None
        payload["stale"] = True
        payload["observed_age_seconds"] = observed_age
        payload["availability"] = "unavailable"
        payload["source_id"] = self.spec.id
        watch = payload.get("watch")
        if not isinstance(watch, dict):
            watch = {}
            payload["watch"] = watch
        watch.update(ok=False, snapshot=None,
                     error="远端 source 不可用；以下为上次成功读取的记录")
        payload["gpus"] = None
        if path == "/api/workspace":
            selection = payload.get("selection")
            if not isinstance(selection, dict):
                selection = {}
                payload["selection"] = selection
            selection.update(live_selected=False, view_mode="unavailable")
            payload["training_pipeline"] = []
            payload["activity"] = None
        return payload

    def close(self) -> None:
        self._client.close()


class RemoteRegistry:
    def __init__(self, sources: Mapping[str, Mapping[str, str]] | None = None,
                 default_source: str | None = None,
                 *, transport_factory: Any = None) -> None:
        self._upstreams: dict[str, RemoteUpstream] = {}
        for source_id, raw in (sources or {}).items():
            spec = _normalize_spec(source_id, raw)
            if transport_factory is None:
                transport = None
            else:
                transport = transport_factory(spec.id)
            self._upstreams[spec.id] = RemoteUpstream(spec, transport=transport)
        self.default_source = default_source or "local"
        if self.default_source != "local" and self.default_source not in self._upstreams:
            self.close()
            raise ValueError("default_source must name local or a configured upstream")

    @property
    def source_ids(self) -> frozenset[str]:
        return frozenset(self._upstreams)

    def catalog(self, local_label: str = "本机 dashboard") -> dict[str, Any]:
        entries = [{"id": "local", "label": local_label, "kind": "local", "availability": "local"}]
        entries.extend({"id": item.spec.id, "label": item.spec.label, "kind": "remote",
                        "availability": "configured"}
                       for item in self._upstreams.values())
        return {"schema": "qwen-rl-dashboard-sources-v1", "default_source": self.default_source,
                "sources": entries}

    def get(self, source_id: str, path: str, query: Mapping[str, list[str]]) -> tuple[int, Any]:
        upstream = self._upstreams.get(source_id)
        if upstream is None:
            raise LookupError("unknown source")
        try:
            return upstream.get(path, query)
        except UpstreamUnavailable as exc:
            return 502, {"error": str(exc), "source_id": source_id,
                         "availability": "unavailable", "stale": False}
        except LookupError as exc:
            return 404, {"error": str(exc)}
        except ValueError as exc:
            return 400, {"error": str(exc)}

    def close(self) -> None:
        for upstream in self._upstreams.values():
            upstream.close()
        self._upstreams.clear()


def _normalize_spec(source_id: Any, raw: Mapping[str, Any]) -> UpstreamSpec:
    if not isinstance(source_id, str) or not _SOURCE_ID.fullmatch(source_id) or source_id == "local":
        raise ValueError("invalid remote source id")
    if not isinstance(raw, Mapping) or not {"label", "url"} <= set(raw) or set(raw) - {"label", "url", "snapshot_path"}:
        raise ValueError("remote source must contain label/url and optional snapshot_path")
    label = raw.get("label")
    if not isinstance(label, str) or not label.strip() or len(label) > 128 or any(ord(char) < 32 for char in label):
        raise ValueError("invalid remote source label")
    snapshot_path = None
    if raw.get("snapshot_path") is not None:
        value = raw.get("snapshot_path")
        if not isinstance(value, str) or not value or len(value) > 4096 or any(ord(char) < 32 for char in value):
            raise ValueError("invalid snapshot_path")
        snapshot_path = Path(value)
        if not snapshot_path.is_absolute() or ".." in snapshot_path.parts:
            raise ValueError("snapshot_path must be absolute and normalized")
    return UpstreamSpec(source_id, label.strip(), _validate_url(raw.get("url")), snapshot_path)


def load_remote_profile(path: str | Path) -> tuple[dict[str, dict[str, str]], str]:
    """Load a bounded, credential-free fixed loopback source profile."""
    profile_path = Path(path)
    try:
        if profile_path.stat().st_size > PROFILE_LIMIT:
            raise ValueError("remote profile exceeds size limit")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError("remote profile cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("remote profile is not valid JSON") from exc
    if not isinstance(profile, dict) or set(profile) != {"schema_version", "default_source", "sources"}:
        raise ValueError("remote profile has an invalid shape")
    if profile.get("schema_version") != 1:
        raise ValueError("unsupported remote profile schema_version")
    items = profile.get("sources")
    if not isinstance(items, list) or not items or len(items) > MAX_SOURCES:
        raise ValueError("remote profile sources must contain 1 to 16 entries")
    sources: dict[str, dict[str, str]] = {}
    for item in items:
        if (not isinstance(item, dict) or not {"id", "label", "url"} <= set(item)
                or set(item) - {"id", "label", "url", "snapshot_path"}):
            raise ValueError("remote profile source has an invalid shape")
        source_id = item.get("id")
        raw_spec = {"label": item.get("label"), "url": item.get("url")}
        if "snapshot_path" in item:
            raw_spec["snapshot_path"] = item.get("snapshot_path")
        spec = _normalize_spec(source_id, raw_spec)
        if spec.id in sources:
            raise ValueError("duplicate remote source id")
        sources[spec.id] = {"label": spec.label, "url": spec.url}
        if spec.snapshot_path is not None:
            sources[spec.id]["snapshot_path"] = str(spec.snapshot_path)
    default = profile.get("default_source")
    if not isinstance(default, str) or (default != "local" and default not in sources):
        raise ValueError("default_source must name local or a configured remote source")
    return sources, default


def seed_workspace_snapshot(path: str | Path, *, source_id: str, upstream_url: str,
                            workspace: Mapping[str, Any], query: Mapping[str, str] | None = None,
                            observed_at: float | None = None) -> None:
    """Privately seed a trusted source's last workspace from a captured v2 response."""
    spec = _normalize_spec(source_id, {"label": source_id, "url": upstream_url,
                                       "snapshot_path": str(path)})
    bounded = _bounded_query("/api/workspace", {
        key: [value] for key, value in (query or {}).items()
        if isinstance(key, str) and isinstance(value, str)
    })
    if len(bounded) != len(query or {}):
        raise ValueError("snapshot query must contain string keys and values")
    payload = sanitize_payload(dict(workspace))
    if not _valid_workspace_shape(payload):
        raise ValueError("workspace does not match qwen-rl-dashboard-v2")
    selected_run_id = (payload.get("selection") or {}).get("run_id")
    if bounded.get("run_id") and selected_run_id and bounded["run_id"] != selected_run_id:
        raise ValueError("snapshot query run_id does not match workspace selection")
    observed = time.time() if observed_at is None else observed_at
    if not isinstance(observed, (int, float)) or not math.isfinite(observed) or observed <= 0:
        raise ValueError("observed_at must be a positive timestamp")
    payload["source_id"] = spec.id
    projection = RemoteUpstream._project_last_observed(payload, float(observed))
    if projection is None:
        raise ValueError("workspace has no run observation bound to its selected runtime")
    payload["last_observed_logical_run"] = projection
    payload[SNAPSHOT_META_FIELD] = {
        "schema": WORKSPACE_SCHEMA,
        "source_id": spec.id,
        "endpoint_sha256": hashlib.sha256(spec.url.encode("utf-8")).hexdigest(),
        "path": "/api/workspace",
        "query": bounded,
        "observed_at": float(observed),
        "runtime_run_id": projection["runtime_run_id"],
    }
    _atomic_private_json(spec.snapshot_path, payload)
