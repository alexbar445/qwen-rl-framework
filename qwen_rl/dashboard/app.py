"""Read-only dashboard HTTP surface over deployment-bound observations."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .cache import LiveWatch, Refreshing
from .collect import Collector, SCHEMA
from .evidence import evidence_for
from .operations import operations_for
from .upstream import RemoteRegistry

STATIC_DIR = Path(__file__).resolve().parent / "static"
BUILD = "v2"
LIVE_INTERVAL = 1.0
OVERVIEW_TTL = 1.0
RUN_DETAIL_TTL = 4.0
BEPA_TTL = 3.0
GPU_TTL = 2.0
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


def _run_id(value: str) -> bool:
    return bool(_ID.fullmatch(value)) and value not in {".", ".."}


def _verified_instance_observation(snapshot: Any, source: Any) -> bool:
    if (not isinstance(source, dict) or source.get("kind") != "runtime"
            or not source.get("instance_manifest")
            or not isinstance(snapshot, dict) or snapshot.get("owned_source_verified") is not True):
        return False
    lifecycle = snapshot.get("lifecycle") if isinstance(snapshot.get("lifecycle"), dict) else {}
    identity = snapshot.get("source_identity") if isinstance(snapshot.get("source_identity"), dict) else {}
    return bool(
        lifecycle.get("backend") == "central_gradient"
        and isinstance(identity.get("instance_run_id"), str)
        and lifecycle.get("instance_run_id") == identity.get("instance_run_id")
        and isinstance(identity.get("telemetry_epoch"), str) and identity["telemetry_epoch"]
        and isinstance(identity.get("runtime_config_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", identity["runtime_config_sha256"])
        and isinstance(identity.get("gradient_manifest_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", identity["gradient_manifest_sha256"])
    )


def _source_build() -> str:
    """Fingerprint the code loaded by this server, independent of disk HEAD."""
    digest = hashlib.sha256()
    base = Path(__file__).resolve().parent
    for file in sorted(base.rglob("*")):
        if not file.is_file() or file.suffix not in {".py", ".js", ".css", ".html"}:
            continue
        digest.update(str(file.relative_to(base)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


class Dashboard:
    """Cached reader. Request handlers never invoke training or Teacher calls."""

    def __init__(self, collector: Collector, static_dir: Path = STATIC_DIR, *,
                 background: bool = True,
                 remote_sources: dict[str, dict[str, str]] | None = None,
                 default_source: str | None = None,
                 upstream_transport_factory: Any = None) -> None:
        self.collector = collector
        self.static = Path(static_dir)
        selected_source = getattr(collector, "source", None)
        self.local_source_id = "runtime" if getattr(selected_source, "instance_manifest_path", None) is not None else "local"
        if self.local_source_id in (remote_sources or {}):
            raise ValueError(f"remote source id conflicts with local source {self.local_source_id}")
        registry_default = default_source
        if self.local_source_id != "local" and default_source in (None, "local"):
            registry_default = "local"
        self.remote_registry = RemoteRegistry(
            remote_sources, registry_default, transport_factory=upstream_transport_factory)
        self.default_source = (self.local_source_id if self.remote_registry.default_source == "local"
                               else self.remote_registry.default_source)
        self.code_sha256 = _source_build()
        self.started_at = time.time()
        self._closed = False
        self._live = (LiveWatch(collector.watch_snapshot, interval=LIVE_INTERVAL)
                      if collector.live and background else None)
        self._overview = Refreshing(OVERVIEW_TTL)
        self._run_detail = Refreshing(RUN_DETAIL_TTL)
        self._bepa = Refreshing(BEPA_TTL)
        self._gpu = Refreshing(GPU_TTL)
        if background:
            threading.Thread(target=self.workspace, name="dashboard-warmup", daemon=True).start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._live is not None:
            self._live.stop()
        self.remote_registry.close()
        close = getattr(self.collector, "close", None)
        if callable(close):
            close()

    def live_watch(self) -> dict[str, Any]:
        if self._live is None:
            return self.collector.watch_snapshot()
        return self._live.get()

    def request_live_refresh(self) -> None:
        if self._live is not None:
            self._live.request_refresh()

    def overview(self) -> dict[str, Any]:
        # Inventory can be cached; the selected runtime observation must stay current.
        inventory = self._overview.get("overview", lambda _: self.collector.overview({}))
        return {**inventory, "watch": self.live_watch()}

    def run_detail(self, run_id: str) -> Any:
        if not _run_id(run_id):
            return None
        return self._run_detail.get(run_id, self.collector.run_detail)

    def workspace(self, run_id: str = "") -> dict[str, Any]:
        if run_id and not _run_id(run_id):
            raise ValueError("invalid run_id")
        value = self._compose_workspace(run_id)
        value["workspace_age_seconds"] = max(0.0, time.time() - value["generated_at"])
        return value

    def _compose_workspace(self, requested: str) -> dict[str, Any]:
        overview = self.overview()
        runs = overview.get("runs") or []
        active_id = overview.get("active_run_id")
        watch = overview.get("watch") or {}
        watch_age = watch.get("age_seconds")
        if watch.get("ok") and isinstance(watch_age, (int, float)) and watch_age > 60.0:
            watch = {**watch, "ok": False, "snapshot": None,
                     "error": "即时观测已超过60秒；等待新采样，不显示旧状态为当前"}
        observed = watch.get("snapshot") or {}
        current_source = self.collector.source_info()
        if current_source.get("kind") in {"local", "runtime"}:
            active_id = current_source.get("run_id")
        instance_observation = _verified_instance_observation(observed, current_source)
        standalone = ((not active_id and current_source.get("kind") == "learner"
                       and observed.get("owned_source_verified") is True
                       and (observed.get("lifecycle") or {}).get("run_id") is None)
                      or (not active_id and instance_observation))
        selected = requested or active_id or (runs[0]["run_id"] if runs and not standalone else None)
        detail = self.run_detail(selected) if selected else None
        if requested and detail is None:
            raise LookupError("unknown run_id")
        observed_id = ((observed.get("lifecycle") or {}).get("run_id")
                       or (observed.get("learner") or {}).get("runtime_run_id"))
        source = self.collector.source_info(selected)
        source_identity = observed.get("source_identity") or {}
        owned_observation = bool(
            not selected and not requested and (
                (source.get("kind") == "learner"
                 and observed.get("owned_source_verified") is True
                 and all(isinstance(source_identity.get(key), str) and source_identity[key]
                         for key in ("container_id", "profile_sha256", "telemetry_epoch")))
                or _verified_instance_observation(observed, source)
            ))
        has_fresh_observation = any(
            isinstance(item, dict) and item.get("live") is True
            for item in (observed.get("freshness") or {}).values())
        live_selected = bool(watch.get("ok") and has_fresh_observation and (
            (selected and observed_id == selected) or owned_observation))
        historical_selection = (source.get("kind") == "offline"
            or (requested and selected != active_id)
            or ((detail or {}).get("manifest") or {}).get("status") in {"stopped", "failed"})
        view_mode = "live" if live_selected else ("history" if historical_selection else "unavailable")
        shown_watch = watch if live_selected else {
            "ok": False, "snapshot": None, "age_seconds": watch.get("age_seconds"),
            "error": ("所选历史 run 不使用其他运行的即时指标" if requested and observed_id != selected
                      else watch.get("error") or "当前所选运行没有即时观测"),
        }
        base_bepa = self._bepa.get(selected, self.collector.bepa_detail)
        evidence = evidence_for(self.collector, detail, observed if live_selected else {},
                                live_selected, base_bepa=base_bepa)
        operations = operations_for(observed if live_selected else None)
        history_result = self.collector.history()
        history = history_result["samples"]
        history = [row for row in history if (
            selected is not None and row.get("runtime_run_id") == selected) or (
            selected is not None and row.get("ok") is False and row.get("expected_runtime_run_id") == selected) or (
            owned_observation and row.get("owned_source_identity") == source_identity)]
        return {
            "schema": SCHEMA, "build": BUILD, "code_sha256": self.code_sha256,
            "generated_at": time.time(), "source": source,
            "selection": {"run_id": selected, "active_run_id": active_id,
                          "live_selected": live_selected, "observed_runtime_run_id": observed_id,
                          "observation_only": owned_observation, "source_identity": source_identity,
                          "view_mode": view_mode},
            "runs": runs, "watch": shown_watch,
            # The chart needs timestamps/rates/identity, not every historical telemetry
            # counter and activity breakdown. Rich samples remain on /api/history.
            "history": [{key: value for key, value in row.items() if key in {
                "sampled_at", "source_identity", "owned_source_identity", "runtime_identity", "expected_runtime_run_id",
                "runtime_run_id", "ok", "error", "native_decode_tokens_per_second", "series_epoch", "output_tokens_per_second",
                "executor_rates", "updates_per_hour"}} for row in history],
            "run": detail,
            "bepa": evidence.get("bepa", base_bepa),
            "evaluation": evidence.get("evaluation", {}),
            "recovery": evidence.get("recovery", {}),
            "events": evidence.get("events", []),
            "insights": operations["insights"] + evidence.get("insights", []),
            "training_pipeline": operations["training_pipeline"],
            "activity": operations["activity"],
            "pipeline": evidence.get("pipeline", []),
            "gpus": overview.get("gpus"),
        }

    def api(self, path: str, query: dict[str, list[str]]) -> tuple[int, Any]:
        def one(name: str, default: str = "") -> str:
            return (query.get(name) or [default])[0]

        if path == "/api/sources":
            source_kind = self.collector.source_info().get("kind") or "local"
            catalog = self.remote_registry.catalog("本机 · " + str(source_kind))
            catalog["local_source_id"] = self.local_source_id
            if self.local_source_id != "local":
                local = next(item for item in catalog["sources"] if item["id"] == "local")
                local.update(id=self.local_source_id, kind=source_kind)
            catalog["default_source"] = self.default_source
            return 200, catalog

        requested_sources = query.get("source") or []
        if len(requested_sources) > 1:
            return 400, {"error": "source must be specified once"}
        source_id = requested_sources[0] if requested_sources else self.default_source
        local_query = {key: values for key, values in query.items() if key != "source"}
        if source_id != self.local_source_id:
            if source_id not in self.remote_registry.source_ids:
                return 404, {"error": "unknown source"}
            return self.remote_registry.get(source_id, path, local_query)
        query = local_query

        if path == "/api/health":
            return 200, {"ok": not self._closed, "schema": SCHEMA, "build": BUILD,
                         "code_sha256": self.code_sha256, "started_at": self.started_at,
                         "source": self.collector.source_info(), "live": self.collector.live}
        if path == "/api/workspace":
            try:
                return 200, self.workspace(one("run_id"))
            except ValueError as exc:
                return 400, {"error": str(exc)}
            except LookupError as exc:
                return 404, {"error": str(exc)}
        if path == "/api/overview":
            return 200, self.overview()
        if path == "/api/watch":
            self.request_live_refresh()
            return 200, self.live_watch()
        if path == "/api/runs":
            return 200, {"runs": self.collector.runs(),
                         "active_run_id": self.collector.active_run_id()}
        if path == "/api/run":
            detail = self.run_detail(one("run_id"))
            return (200, detail) if detail else (404, {"error": "unknown run_id"})
        if path == "/api/trajectory":
            if not _run_id(one("run_id")):
                return 404, {"error": "unknown run_id"}
            detail = self.collector.trajectory(one("run_id"), one("name"))
            return (200, detail) if detail else (404, {"error": "unknown trajectory"})
        if path == "/api/bepa":
            return 200, self._bepa.get("bepa", lambda _: self.collector.bepa_detail())
        if path == "/api/artifact":
            selected = one("run_id") or None
            if selected and not _run_id(selected):
                return 404, {"error": "unknown run_id"}
            detail = self.collector.artifact(one("ref"), run_id=selected)
            return (200, detail) if detail else (404, {"error": "unknown artifact"})
        if path == "/api/logs":
            if not _run_id(one("run_id")):
                return 404, {"error": "unknown run_id"}
            return 200, self.collector.logs(one("run_id"), one("role", "manager"), one("tail", "200"))
        if path == "/api/gpu":
            return 200, self._gpu.get("gpu", lambda _: {"gpus": self.collector.gpu()})
        return 404, {"error": "no such endpoint", "path": path}

    def static_file(self, path: str) -> Path | None:
        rel = unquote(path).lstrip("/")
        if rel.startswith("static/"):
            rel = rel[len("static/"):]
        candidate = (self.static / (rel or "index.html")).resolve()
        try:
            candidate.relative_to(self.static.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None


class Handler(BaseHTTPRequestHandler):
    dashboard: Dashboard
    server_version = "qwen-rl-dashboard/" + BUILD
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.path.startswith("/api/"):
            print("%s %s %s" % (self.log_date_time_string(), self.path.split("?")[0], fmt % args), flush=True)

    def do_GET(self) -> None:
        if len(self.path) > 8192:
            self._send(414, b'{"error":"request target too large"}', MIME[".json"])
            return
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
                try:
                    query = parse_qs(parsed.query, max_num_fields=32, keep_blank_values=True)
                except ValueError:
                    self._send(400, b'{"error":"too many query parameters"}', MIME[".json"])
                    return
                status, payload = self.dashboard.api(parsed.path, query)
                self._send(status, json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                              default=str).encode(), MIME[".json"])
                return
            target = self.dashboard.static_file(parsed.path)
            if target is None:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            self._send(200, target.read_bytes(), MIME.get(target.suffix, "application/octet-stream"))
        except BrokenPipeError:
            pass
        except Exception as exc:
            from .collect import _redact
            body = json.dumps({"error": _redact("%s: %s" % (type(exc).__name__, exc))},
                              ensure_ascii=False).encode()
            try:
                self._send(500, body, MIME[".json"])
            except BrokenPipeError:
                pass

    def do_POST(self) -> None:
        self._send(405, b'{"error":"read-only dashboard"}', MIME[".json"])

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                         "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                         "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_close(self) -> None:
        self.RequestHandlerClass.dashboard.close()
        super().server_close()


def serve(collector: Collector, host: str, port: int,
          static_dir: Path = STATIC_DIR, *,
          remote_sources: dict[str, dict[str, str]] | None = None,
          default_source: str | None = None,
          upstream_transport_factory: Any = None) -> ThreadingHTTPServer:
    dashboard = Dashboard(collector, static_dir, remote_sources=remote_sources,
                          default_source=default_source,
                          upstream_transport_factory=upstream_transport_factory)
    handler = type("BoundHandler", (Handler,), {"dashboard": dashboard})
    try:
        return DashboardServer((host, port), handler)
    except Exception:
        dashboard.close()
        raise
