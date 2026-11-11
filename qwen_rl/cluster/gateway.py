"""Site-scoped Gateway for registered HTTP Workers and policy artifacts."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import quote, urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .spool import (
    GenerationIdentityConflict,
    GenerationIndeterminate,
    GenerationResult,
    GenerationSpool,
    SpoolUnavailable,
)
from .settings import validate_credential_origin


_NODE_ID = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")
_HOST_LABEL = re.compile(
    r"^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$"
)
_ARTIFACT_READER_STATES = frozenset({
    "registered", "loading", "ready",
})
_ROUTE_METHODS = {
    "health": frozenset({"GET"}),
    "policy": frozenset({"POST"}),
    "policy/prepare": frozenset({"POST"}),
    "generate": frozenset({"POST"}),
    "episode/acquire": frozenset({"POST"}),
    "episode/release": frozenset({"POST"}),
}
_HOP_BY_HOP_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
    "content-encoding",
})


class GatewaySettings(BaseModel):
    """Immutable settings for one Gateway node.

    ``node_token`` authenticates the Gateway to its site's Controller API.
    ``gateway_bearer`` authenticates central callers to this Gateway. They are
    separate credentials with separate scopes.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    site_id: StrictStr = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    node_id: StrictStr = Field(pattern=_NODE_ID.pattern)
    controller_url: StrictStr = Field(min_length=1, max_length=2048)
    node_token: StrictStr = Field(min_length=16, max_length=4096)
    gateway_bearer: StrictStr = Field(min_length=16, max_length=4096)
    state_dir: StrictStr = Field(min_length=1, max_length=4096)
    policy_source_url: StrictStr | None = None
    policy_distribution: Literal["dragonfly", "http"] = "dragonfly"
    policy_lineage_id: StrictStr | None = None
    artifact_seed_origin: StrictStr | None = None
    request_timeout_seconds: StrictFloat | StrictInt = Field(default=180.0, gt=0, le=3600)
    registry_poll_seconds: StrictFloat | StrictInt = Field(default=1.0, gt=0, le=60)
    registry_stale_seconds: StrictFloat | StrictInt = Field(default=5.0, gt=0, le=300)

    @model_validator(mode="after")
    def validate_gateway_credentials(self) -> "GatewaySettings":
        if hmac.compare_digest(self.node_token, self.gateway_bearer):
            raise ValueError("node_token and gateway_bearer must be distinct credentials")
        if self.policy_source_url is not None and self.policy_lineage_id is None:
            raise ValueError("policy_lineage_id is required when policy_source_url is configured")
        if self.policy_distribution == "http":
            if self.policy_source_url is None:
                raise ValueError("policy_distribution=http requires policy_source_url")
            if self.artifact_seed_origin is not None:
                raise ValueError("artifact_seed_origin cannot be configured when policy_distribution=http")
        elif self.policy_source_url is not None and self.artifact_seed_origin is None:
            raise ValueError("artifact_seed_origin is required for policy_distribution=dragonfly")
        if self.registry_stale_seconds < self.registry_poll_seconds:
            raise ValueError("registry_stale_seconds must be at least registry_poll_seconds")
        return self

    @field_validator("artifact_seed_origin")
    @classmethod
    def validate_seed_origin(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from .artifacts import _is_private_or_local
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or not _is_private_or_local(parsed.hostname) or parsed.username or parsed.password
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError("artifact_seed_origin must be a private site Gateway origin")
        return value.rstrip("/")

    @field_validator("controller_url")
    @classmethod
    def validate_controller_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("controller_url has an invalid port") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("controller_url must be an absolute HTTP(S) origin")
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("controller_url must not contain credentials, query, or fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError("controller_url must be an origin without a path prefix")
        validate_credential_origin(value)
        return value.rstrip("/")

    @field_validator("policy_source_url")
    @classmethod
    def validate_policy_source_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("policy_source_url has an invalid port") from exc
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise ValueError("policy_source_url must be an HTTP(S) origin without credentials or path")
        validate_credential_origin(value)
        return value.rstrip("/")

    @field_validator("policy_lineage_id")
    @classmethod
    def validate_policy_lineage_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = uuid.UUID(value)
        except (ValueError, AttributeError) as exc:
            raise ValueError("policy_lineage_id must be an explicit UUID") from exc
        if value not in {str(parsed), parsed.hex}:
            raise ValueError("policy_lineage_id must use canonical or 32-hex UUID formatting")
        return value

    @classmethod
    def load(cls, path: str | Path) -> "GatewaySettings":
        from .settings import load_settings_json

        values = load_settings_json(path, secret_fields=("node_token", "gateway_bearer"))
        return cls.model_validate(values, strict=True)


class _InventoryUnavailable(Exception):
    pass


class _InventoryAuthorizationDenied(_InventoryUnavailable):
    pass


class _SiteInventory:
    """In-memory site registry snapshot refreshed off the Worker hot path."""

    def __init__(self, *, max_age_seconds: float):
        self.max_age_seconds = max_age_seconds
        self.rows: tuple[dict[str, Any], ...] | None = None
        self.updated_at: float | None = None
        self.last_error: str | None = None
        self._lock = asyncio.Lock()

    async def refresh(self, fetch) -> None:
        async with self._lock:
            try:
                rows = await fetch()
            except _InventoryAuthorizationDenied as exc:
                # Do not keep secret-bearing Worker credentials cached after
                # the Controller revokes this Gateway's inventory access.
                self.rows = None
                self.updated_at = None
                self.last_error = str(exc)[:200] or type(exc).__name__
                raise
            except Exception as exc:
                self.last_error = str(exc)[:200] or type(exc).__name__
                raise
            self.rows = tuple(rows)
            self.updated_at = time.monotonic()
            self.last_error = None

    def health(self) -> dict[str, Any]:
        age = None if self.updated_at is None else max(0.0, time.monotonic() - self.updated_at)
        stale = age is None or age > self.max_age_seconds
        return {
            "age_seconds": age,
            "stale": stale,
            "error": self.last_error,
            "node_count": 0 if self.rows is None else len(self.rows),
        }

    def snapshot(self, *, allow_stale: bool = False) -> list[dict[str, Any]]:
        metadata = self.health()
        if (metadata["stale"] and not allow_stale) or self.rows is None:
            raise _InventoryUnavailable("site inventory is stale or unavailable")
        return list(self.rows)


def _load_settings(settings: GatewaySettings | Mapping[str, Any] | str | Path) -> GatewaySettings:
    if isinstance(settings, GatewaySettings):
        return settings
    if isinstance(settings, (str, Path)):
        return GatewaySettings.load(settings)
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be GatewaySettings or a strict settings mapping")
    return GatewaySettings.model_validate(dict(settings), strict=True)


def _bearer(request: Request) -> str:
    value = request.headers.get("authorization", "")
    scheme, separator, token = value.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token or " " in token:
        return ""
    return token


def _host_origin(host: str, port: int) -> str:
    """Build an origin from a registry host field, never from request input."""
    if not isinstance(host, str) or not host or any(c in host for c in "/\\@?#%\x00\r\n"):
        raise ValueError("registry Worker host is not a plain hostname or IP address")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("registry Worker port is invalid")
    candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        if not _HOST_LABEL.fullmatch(candidate):
            raise ValueError("registry Worker host is not a valid LAN hostname")
        return f"http://{candidate}:{port}"
    if address.version == 6:
        return f"http://[{address.compressed}]:{port}"
    return f"http://{address.compressed}:{port}"


def _response_headers(headers: httpx.Headers | tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    raw = headers.multi_items() if isinstance(headers, httpx.Headers) else headers
    return tuple((name, value) for name, value in raw if name.lower() not in _HOP_BY_HOP_HEADERS)


def _make_response(result: GenerationResult) -> Response:
    headers = dict(result.headers)
    return Response(content=result.body, status_code=result.status_code, headers=headers)


def create_gateway_app(
    settings: GatewaySettings | Mapping[str, Any] | str | Path,
    registry_client: httpx.AsyncClient | None = None,
    artifact_store: Any | None = None,
    policy_ready_hook: Any | None = None,
) -> FastAPI:
    """Create a Gateway app using Controller inventory and per-node tokens.

    ``registry_client`` is an optional HTTPX client, mainly useful for mounting
    the Controller's real HTTP app into a CPU test. Controller inventory is
    fetched at startup and refreshed in the background; Worker requests use
    only a fresh local snapshot.
    """
    cfg = _load_settings(settings)
    spool = GenerationSpool(Path(cfg.state_dir) / "generation-results.sqlite3")
    owns_registry_client = registry_client is None
    owns_artifact_store = False
    if artifact_store is None and cfg.policy_source_url is not None:
        if cfg.policy_lineage_id is None:
            raise ValueError("policy_lineage_id is required with policy_source_url")
        from .artifacts import ArtifactMirror

        artifact_store = ArtifactMirror(
            cfg.policy_source_url,
            Path(cfg.state_dir) / "artifact-cache",
            headers={"Authorization": f"Bearer {cfg.node_token}"},
            lineage_id=cfg.policy_lineage_id,
        )
        owns_artifact_store = True
    controller_client = registry_client or httpx.AsyncClient(
        timeout=cfg.request_timeout_seconds,
        follow_redirects=False,
    )
    worker_client = httpx.AsyncClient(
        timeout=cfg.request_timeout_seconds,
        follow_redirects=False,
        trust_env=False,
    )
    site_inventory = _SiteInventory(max_age_seconds=cfg.registry_stale_seconds)

    async def close_clients() -> None:
        await worker_client.aclose()
        if owns_registry_client:
            await controller_client.aclose()
        if owns_artifact_store:
            await artifact_store.aclose()

    async def fetch_inventory() -> list[dict[str, Any]]:
        url = f"{cfg.controller_url}/v1/sites/{quote(cfg.site_id, safe='')}/nodes"
        try:
            response = await controller_client.get(
                url,
                headers={"Authorization": f"Bearer {cfg.node_token}", "Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise _InventoryUnavailable("Controller request failed") from exc
        if response.status_code in {401, 403}:
            raise _InventoryAuthorizationDenied("Controller denied Gateway inventory access")
        if response.status_code != 200:
            raise _InventoryUnavailable(f"Controller returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise _InventoryUnavailable("Controller returned invalid JSON") from exc
        rows = payload.get("nodes") if isinstance(payload, dict) else None
        if (not isinstance(payload, dict) or payload.get("site_id") != cfg.site_id
                or not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)):
            raise _InventoryUnavailable("Controller response has no node list")
        return rows

    async def refresh_registry_forever() -> None:
        while True:
            await asyncio.sleep(cfg.registry_poll_seconds)
            try:
                await site_inventory.refresh(fetch_inventory)
            except Exception:
                # Retain evidence of the last good snapshot, but fail closed as
                # soon as it is older than the configured freshness bound.
                continue

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await site_inventory.refresh(fetch_inventory)
        except Exception:
            # Keep diagnostics available if Controller is down at startup;
            # inventory-dependent routes remain unavailable until a good read.
            pass
        refresh_task = asyncio.create_task(refresh_registry_forever())
        try:
            await spool.recover_pending()
            yield
        finally:
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass
            await close_clients()

    app = FastAPI(
        title="LAN Worker Gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.spool = spool
    app.state.site_inventory = site_inventory

    async def require_gateway_bearer(request: Request) -> None:
        supplied = _bearer(request)
        if not supplied or not hmac.compare_digest(supplied, cfg.gateway_bearer):
            raise HTTPException(
                status_code=401,
                detail={"code": "unauthorized", "message": "valid Gateway bearer is required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        allow_stale = request.url.path.endswith("/episode/release")
        require_registered_gateway(allow_stale=allow_stale)

    def require_registered_gateway(*, allow_stale: bool = False) -> dict[str, Any]:
        rows = inventory(allow_stale=allow_stale)
        gateway = next((
            row for row in rows
            if row.get("site_id") == cfg.site_id
            and row.get("node_id") == cfg.node_id
            and row.get("role") == "gateway"
        ), None)
        if gateway is None or gateway.get("state") == "quarantined":
            raise HTTPException(
                status_code=403,
                detail={"code": "gateway_rejoin_required", "message": "Gateway must rejoin the Controller before serving requests"},
            )
        return gateway

    def inventory(*, allow_stale: bool = False) -> list[dict[str, Any]]:
        try:
            return site_inventory.snapshot(allow_stale=allow_stale)
        except _InventoryUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "site_inventory_stale", "message": str(exc)},
            ) from exc

    async def require_artifact_reader(request: Request) -> None:
        supplied = _bearer(request)
        if not supplied:
            raise HTTPException(
                status_code=401,
                detail={"code": "unauthorized", "message": "Gateway or registered Worker bearer is required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        if hmac.compare_digest(supplied, cfg.gateway_bearer):
            require_registered_gateway()
            return
        rows = inventory()
        authorized = any(
            row.get("site_id") == cfg.site_id
            and row.get("role") == "worker"
            and row.get("state") in _ARTIFACT_READER_STATES
            and isinstance(row.get("worker_token"), str)
            and hmac.compare_digest(supplied, row["worker_token"])
            for row in rows
        )
        if not authorized:
            raise HTTPException(
                status_code=401,
                detail={"code": "unauthorized", "message": "Gateway or registered Worker bearer is required"},
                headers={"WWW-Authenticate": "Bearer"},
            )

    app.add_exception_handler(SpoolUnavailable, lambda _request, exc: JSONResponse(
        status_code=503,
        content={"error": {"code": "generation_spool_unavailable", "message": str(exc)}},
    ))

    @app.get("/health")
    async def gateway_health() -> dict[str, Any]:
        registry = site_inventory.health()
        return {
            "role": "gateway",
            "ready": not registry["stale"],
            "registry_cache": registry,
        }


    @app.get("/v1/workers", dependencies=[Depends(require_gateway_bearer)])
    async def list_workers(request: Request) -> dict[str, Any]:
        rows = inventory()
        query = list(request.query_params.multi_items())
        if len(query) > 1 or any(name != "readiness" for name, _ in query):
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_worker_query", "message": "only one readiness selector is allowed"},
            )
        readiness = request.query_params.get("readiness", "policy")
        if readiness not in {"model", "policy"}:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_worker_query", "message": "readiness must be model or policy"},
            )
        required_field = "model_ready" if readiness == "model" else "eligible"
        workers = []
        for row in rows:
            if (row.get("site_id") != cfg.site_id or row.get("role") != "worker"
                    or row.get(required_field) is not True):
                continue
            node_id = row.get("node_id")
            worker_port = row.get("worker_port")
            health = row.get("worker_health")
            if (not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id)
                    or type(worker_port) is not int or not 1 <= worker_port <= 65535
                    or not isinstance(row.get("boot_id"), str) or not row["boot_id"]
                    or not isinstance(health, dict)
                    or not isinstance(health.get("telemetry_epoch"), str)
                    or not health["telemetry_epoch"]):
                continue
            workers.append({
                "worker_id": node_id,
                "node_id": node_id,
                "site_id": cfg.site_id,
                "boot_id": row.get("boot_id"),
                "worker_port": worker_port,
                "gpus": row.get("gpus", []),
                "base_path": f"/v1/workers/{node_id}",
                "worker_health": row.get("worker_health"),
                "model_ready": row.get("model_ready") is True,
                "eligible": row.get("eligible") is True,
            })
        return {"workers": workers}

    async def forward(request: Request, node_id: str, worker_path: str) -> Response:
        if request.query_params:
            raise HTTPException(
                status_code=400,
                detail={"code": "query_not_allowed", "message": "Worker ABI routes do not accept query parameters"},
            )
        methods = _ROUTE_METHODS.get(worker_path)
        if methods is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "worker_route_not_found", "message": "Worker ABI route is not allowlisted"},
            )
        if request.method not in methods:
            raise HTTPException(
                status_code=405,
                detail={"code": "worker_method_not_allowed", "message": "method is not allowed for this Worker route"},
                headers={"Allow": ", ".join(sorted(methods))},
            )
        if not _NODE_ID.fullmatch(node_id):
            raise HTTPException(
                status_code=404,
                detail={"code": "worker_not_found", "message": "Worker is not registered"},
            )

        rows = inventory(allow_stale=worker_path == "episode/release")
        node = next((row for row in rows if row.get("node_id") == node_id), None)
        if (node is None or node.get("site_id") != cfg.site_id or node.get("role") != "worker"
                or not isinstance(node.get("worker_token"), str)):
            raise HTTPException(
                status_code=404,
                detail={"code": "worker_not_found", "message": "Worker is not registered"},
            )
        if worker_path in {"health", "policy"} and not (
            node.get("model_ready") is True or node.get("eligible") is True
        ):
            raise HTTPException(
                status_code=503,
                detail={"code": "worker_not_model_ready", "message": "Worker model is not ready"},
            )
        if worker_path in {"generate", "episode/acquire"} and node.get("eligible") is not True:
            raise HTTPException(
                status_code=503,
                detail={"code": "worker_not_eligible", "message": "Worker policy is not ready for new work"},
            )
        try:
            origin = _host_origin(node.get("lan_host"), node.get("worker_port"))
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "worker_inventory_invalid", "message": "Worker has an invalid LAN address"},
            ) from exc
        worker_health = node.get("worker_health")
        boot_id = node.get("boot_id")
        telemetry_epoch = worker_health.get("telemetry_epoch") if isinstance(worker_health, dict) else None

        body = await request.body()
        request_id: str | None = None
        body_sha256: str | None = None
        if worker_path == "generate":
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "invalid_generate_request", "message": "generate body must be valid JSON"},
                ) from exc
            request_id_value = payload.get("request_id") if isinstance(payload, dict) else None
            if not isinstance(request_id_value, str) or not request_id_value or len(request_id_value) > 200:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "invalid_generate_request",
                        "message": "generate body requires a valid request_id",
                    },
                )
            request_id = request_id_value
            body_sha256 = hashlib.sha256(body).hexdigest()

        headers = {"Authorization": f"Bearer {node['worker_token']}"}
        content_type = request.headers.get("content-type")
        if content_type:
            headers["Content-Type"] = content_type
        accept = request.headers.get("accept")
        if accept:
            headers["Accept"] = accept

        def require_pinned_worker_identity() -> tuple[str, str]:
            if worker_path == "episode/release":
                current_epoch = telemetry_epoch if isinstance(telemetry_epoch, str) else ""
                current_boot = boot_id if isinstance(boot_id, str) else ""
                return current_boot, current_epoch
            if (not isinstance(boot_id, str) or not boot_id
                    or not isinstance(telemetry_epoch, str) or not telemetry_epoch):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "worker_identity_unavailable",
                        "message": "Worker boot identity is not available in fresh inventory",
                    },
                )
            expected_boot_id = request.headers.get("x-expected-worker-boot-id")
            expected_telemetry_epoch = request.headers.get("x-expected-worker-telemetry-epoch")
            if expected_boot_id != boot_id or expected_telemetry_epoch != telemetry_epoch:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "worker_epoch_changed",
                        "message": "Worker boot or process epoch differs from the pinned membership",
                    },
                )
            return boot_id, telemetry_epoch

        async def do_request() -> GenerationResult:
            pinned_boot_id, pinned_telemetry_epoch = require_pinned_worker_identity()
            worker_headers = dict(headers)
            if pinned_boot_id:
                worker_headers["X-Expected-Worker-Boot-ID"] = pinned_boot_id
            if pinned_telemetry_epoch:
                worker_headers["X-Expected-Worker-Telemetry-Epoch"] = pinned_telemetry_epoch
            try:
                upstream = await worker_client.request(
                    request.method,
                    f"{origin}/{worker_path}",
                    content=body,
                    headers=worker_headers,
                )
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "worker_result_indeterminate",
                        "message": "Worker request failed after dispatch; generation will not be retried",
                    },
                ) from exc
            identity = {
                "X-Worker-Node-ID": node_id,
                "X-Worker-Boot-ID": pinned_boot_id,
                "X-Worker-Telemetry-Epoch": pinned_telemetry_epoch,
            }
            if worker_path != "episode/release":
                for name, expected in identity.items():
                    observed = upstream.headers.get_list(name)
                    if observed and observed != [expected]:
                        raise HTTPException(502, detail={"code": "upstream_worker_identity_mismatch"})
            identity_names = {name.lower() for name in identity}
            return GenerationResult(
                status_code=upstream.status_code,
                body=upstream.content,
                headers=(
                    *((name, value) for name, value in _response_headers(upstream.headers)
                      if name.lower() not in identity_names),
                    *identity.items(),
                ),
            )

        if request_id is not None and body_sha256 is not None:
            prior = await spool.snapshot(node_id, request_id)
            if prior is None:
                # An identity mismatch is known before dispatch, so it must not
                # create an indeterminate spool row or consume this request ID.
                require_pinned_worker_identity()
            try:
                result = await spool.execute(
                    node_id=node_id,
                    request_id=request_id,
                    body_sha256=body_sha256,
                    operation=do_request,
                )
            except GenerationIdentityConflict as exc:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "request_id_payload_conflict", "message": str(exc)},
                ) from exc
            except GenerationIndeterminate as exc:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "generation_indeterminate", "message": str(exc)},
                ) from exc
            except HTTPException:
                raise
            except SpoolUnavailable:
                raise
            except BaseException as exc:
                # A timeout or cancellation after dispatch leaves the durable row
                # indeterminate. Do not translate it into an automatic retry.
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "worker_result_indeterminate",
                        "message": "Worker result is unknown; this request ID will not be retried",
                    },
                ) from exc
        else:
            result = await do_request()

        return _make_response(result)

    @app.api_route(
        "/v1/workers/{node_id}/{worker_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        dependencies=[Depends(require_gateway_bearer)],
    )
    async def worker_abi_route(request: Request, node_id: str, worker_path: str):
        return await forward(request, node_id, worker_path)

    if artifact_store is not None:
        try:
            from .artifacts import create_artifact_router
        except ImportError as exc:
            raise RuntimeError("cluster artifact router is unavailable") from exc
        if (policy_ready_hook is None and cfg.policy_distribution == "dragonfly"
                and cfg.artifact_seed_origin is not None):
            from .policy_seed import PolicyPreseeder
            policy_ready_hook = PolicyPreseeder(artifact_store,
                artifact_seed_origin=cfg.artifact_seed_origin, site_id=cfg.site_id,
                state_dir=Path(cfg.state_dir) / "policy-p2p", token=cfg.gateway_bearer)
        app.include_router(
            create_artifact_router(artifact_store, policy_ready_hook=policy_ready_hook),
            dependencies=[Depends(require_artifact_reader)],
        )
        try:
            from .artifacts import create_artifact_publish_router
        except ImportError as exc:
            raise RuntimeError("cluster artifact publish router is unavailable") from exc
        app.include_router(
            create_artifact_publish_router(
                artifact_store,
                publish_dependency=Depends(require_gateway_bearer),
            )
        )

    return app
