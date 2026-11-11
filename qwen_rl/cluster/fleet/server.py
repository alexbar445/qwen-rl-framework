"""Authenticated FastAPI surface for the independent Fleet registry."""
from __future__ import annotations

import hmac
import os
from pathlib import Path
import stat
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from .contracts import (
    ApplyAckRequest,
    FleetAssignmentRequest,
    FleetBootstrapConfig,
    FleetNodeRegistration,
    MANAGEMENT_SSH_PORT,
    NodeHeartbeat,
    private_deployment_to_dict,
)
from .private_artifacts import PrivateArtifactError, PrivateArtifactStore
from .registry import FleetRegistry, FleetRegistryError
from .settings import FleetSettings, resolve_settings


SSHWebSocketHandler = Callable[..., Awaitable[None]]


def _bearer(value: str | None) -> str | None:
    if value is None:
        return None
    pieces = value.split()
    if len(pieces) != 2 or pieces[0].lower() != "bearer" or not pieces[1]:
        return None
    return pieces[1]


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _node_bearer(authorization: str | None) -> str:
    value = _bearer(authorization)
    if value is None:
        raise _error(401, "node_auth_required", "a valid Fleet node bearer is required")
    return value


def _admin(authorization: str | None, settings: FleetSettings) -> None:
    value = _bearer(authorization)
    if value is None or not hmac.compare_digest(value, settings.admin_token.get_secret_value()):
        raise _error(401, "admin_auth_required", "a valid Fleet admin bearer is required")


def _check_same_origin(websocket: WebSocket, settings: FleetSettings) -> None:
    origin = websocket.headers.get("origin")
    if origin is None or settings.public_origin is None:
        return
    try:
        observed = urlsplit(origin)
        expected = urlsplit(settings.public_origin)
        observed_port = observed.port or (443 if observed.scheme == "https" else 80)
        expected_port = expected.port or 443
    except ValueError:
        raise _error(403, "origin_forbidden", "Fleet WebSocket origin is not allowed")
    if (observed.scheme != "https" or observed.hostname != expected.hostname or observed_port != expected_port
            or observed.username or observed.password or observed.path not in {"", "/"}):
        raise _error(403, "origin_forbidden", "Fleet WebSocket origin is not allowed")


def require_relay_setup(settings: FleetSettings) -> Path:
    """Ensure the isolated relay authorization file is staged before approval."""
    path = settings.relay_authorized_keys_file
    try:
        metadata = path.lstat()
    except OSError:
        metadata = None
    if (metadata is None or path.is_symlink() or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (os.name == "posix" and metadata.st_uid != os.geteuid())):
        raise FleetRegistryError(503, "ssh_relay_not_initialized", "Fleet SSH relay is not initialized")
    return path


def refresh_relay_authorized_keys(registry: FleetRegistry, settings: FleetSettings) -> None:
    """Atomically refresh this Fleet's isolated relay grants after approval."""
    path = require_relay_setup(settings)
    try:
        from .ssh import write_relay_authorized_keys

        write_relay_authorized_keys(path, registry.list_ssh_grants())
    except FleetRegistryError:
        raise
    except Exception as exc:
        raise FleetRegistryError(503, "ssh_relay_refresh_failed", "Fleet SSH relay authorization could not be refreshed") from exc


def create_fleet_app(
    settings: FleetSettings | Mapping[str, Any] | str | Path,
    registry: FleetRegistry | None = None,
    *,
    ssh_websocket_handler: SSHWebSocketHandler | None = None,
) -> FastAPI:
    """Create the Fleet API; its state is isolated from training Registry state."""
    config = resolve_settings(settings)
    store = registry or FleetRegistry(
        config.state_dir,
        heartbeat_ttl_seconds=config.heartbeat_ttl_seconds,
        max_registered_nodes=config.max_registered_nodes,
        max_pending_nodes=config.max_pending_nodes,
        max_rejected_tombstones=config.max_rejected_tombstones,
        registration_rate_per_minute=config.registration_rate_per_minute,
        reverse_port_start=config.reverse_port_start,
        reverse_port_end=config.reverse_port_end,
        public_origin=config.public_origin,
    )
    store.heartbeat_ttl_seconds = config.heartbeat_ttl_seconds
    store.public_origin = config.public_origin.rstrip("/") if config.public_origin else None
    artifacts = None
    if config.public_upload_dir is not None:
        artifacts = PrivateArtifactStore(config.state_dir, public_upload_dir=config.public_upload_dir)

    app = FastAPI(title="Qwen unified Fleet control plane", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.registry = store
    app.state.settings = config
    app.state.private_artifacts = artifacts

    @app.exception_handler(FleetRegistryError)
    async def registry_error_handler(_request: Request, exc: FleetRegistryError):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, _exc: RequestValidationError):
        # Validation input may contain a private deployment bearer. Never echo it.
        return JSONResponse(status_code=422,
                            content={"error": {"code": "invalid_request", "message": "request did not match the Fleet contract"}})

    @app.get("/v1/fleet/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/fleet/bootstrap/config", response_model=FleetBootstrapConfig)
    async def bootstrap_config():
        return FleetBootstrapConfig(
            ssh_relay_user=config.ssh_relay_user,
            ssh_relay_host_key=config.ssh_relay_host_key,
            ssh_relay_port=config.ssh_relay_port,
            admin_ssh_public_key=config.admin_ssh_public_key,
            ssh_port=MANAGEMENT_SSH_PORT,
        )

    @app.post("/v1/fleet/nodes/register")
    async def register_node(payload: FleetNodeRegistration,
                            authorization: str | None = Header(default=None)):
        bearer = _node_bearer(authorization)
        return store.register_node(payload, bearer=bearer).model_dump(mode="json")

    @app.post("/v1/fleet/nodes/{node_id}/heartbeat")
    async def heartbeat(node_id: str, payload: NodeHeartbeat,
                        authorization: str | None = Header(default=None)):
        bearer = _node_bearer(authorization)
        return store.heartbeat(node_id, payload, bearer=bearer).model_dump(mode="json")

    @app.get("/v1/fleet/nodes/{node_id}/desired")
    async def desired_state(node_id: str, authorization: str | None = Header(default=None)):
        bearer = _node_bearer(authorization)
        return store.desired_state(node_id, bearer=bearer).model_dump(mode="json")

    @app.get("/v1/fleet/nodes/{node_id}/desired/{generation}/deployment")
    async def protected_deployment(node_id: str, generation: int,
                                   authorization: str | None = Header(default=None)):
        bearer = _node_bearer(authorization)
        return private_deployment_to_dict(store.deployment_for(node_id, generation, bearer=bearer))

    @app.post("/v1/fleet/nodes/{node_id}/desired/{generation}/ack")
    async def acknowledge(node_id: str, generation: int, payload: ApplyAckRequest,
                          authorization: str | None = Header(default=None)):
        bearer = _node_bearer(authorization)
        if payload.generation != generation:
            raise _error(409, "generation_mismatch", "acknowledgement body and path generations differ")
        return store.acknowledge(node_id, payload, bearer=bearer).model_dump(mode="json")

    @app.get("/v1/fleet/private-artifacts/{sha256}")
    async def private_artifact(sha256: str, request: Request):
        bearer = _bearer(request.headers.get("authorization"))
        if bearer is None or artifacts is None:
            raise _error(404, "private_artifact_unavailable", "private artifact is unavailable")
        size = store.authorize_private_artifact(sha256, bearer=bearer)
        if size is None:
            # Hide both artifact existence and whether a bearer has an assignment.
            raise _error(404, "private_artifact_unavailable", "private artifact is unavailable")
        try:
            result = artifacts.serve_path(sha256)
        except PrivateArtifactError as exc:
            raise _error(503, "private_artifact_integrity_error", "private artifact ownership check failed") from exc
        if result is None or result[1] != size:
            raise _error(503, "private_artifact_integrity_error", "private artifact is unavailable")
        path, _size = result
        return FileResponse(path, media_type="application/octet-stream", filename=f"learner-bundle-{sha256}.tar.gz",
                            headers={"ETag": f'"sha256:{sha256}"', "Cache-Control": "private, no-store"})

    @app.get("/v1/fleet/admin/nodes")
    async def list_nodes(authorization: str | None = Header(default=None)):
        _admin(authorization, config)
        return {"nodes": [node.model_dump(mode="json") for node in store.list_nodes()]}

    @app.get("/v1/fleet/admin/nodes/{node_id}")
    async def get_node(node_id: str, authorization: str | None = Header(default=None)):
        _admin(authorization, config)
        return store.get_node(node_id).model_dump(mode="json")

    @app.get("/v1/fleet/admin/rejections")
    async def rejection_audit(authorization: str | None = Header(default=None)):
        _admin(authorization, config)
        return {"rejections": [item.model_dump(mode="json") for item in store.list_rejection_audit()]}

    @app.post("/v1/fleet/admin/nodes/{node_id}/approve")
    async def approve_node(node_id: str, authorization: str | None = Header(default=None)):
        _admin(authorization, config)
        require_relay_setup(config)
        result = store.approve_node(node_id)
        refresh_relay_authorized_keys(store, config)
        return result.model_dump(mode="json")

    @app.post("/v1/fleet/admin/nodes/{node_id}/reject")
    async def reject_node(node_id: str, authorization: str | None = Header(default=None)):
        _admin(authorization, config)
        return store.reject_node(node_id).model_dump(mode="json")

    @app.put("/v1/fleet/admin/nodes/{node_id}/assignment")
    async def assign_node(node_id: str, payload: FleetAssignmentRequest,
                          authorization: str | None = Header(default=None)):
        _admin(authorization, config)
        require_relay_setup(config)
        return store.assign_node(node_id, payload.deployment).model_dump(mode="json")

    @app.post("/v1/fleet/admin/nodes/{node_id}/deactivate")
    async def deactivate_node(node_id: str, authorization: str | None = Header(default=None)):
        _admin(authorization, config)
        return store.deactivate_node(node_id).model_dump(mode="json")

    @app.websocket("/v1/fleet/ssh")
    async def ssh_tunnel(websocket: WebSocket):
        try:
            _check_same_origin(websocket, config)
        except HTTPException:
            await websocket.close(code=4403)
            return
        bearer = _bearer(websocket.headers.get("authorization"))
        grant = store.authenticate_ssh_bearer(bearer or "") if bearer else None
        if grant is None:
            await websocket.close(code=4401)
            return
        if ssh_websocket_handler is None:
            try:
                from .ssh import handle_ssh_websocket
            except ImportError:
                await websocket.close(code=1011)
                return
            handler = handle_ssh_websocket
        else:
            handler = ssh_websocket_handler
        await websocket.accept()
        try:
            await handler(websocket, grant, relay_host=config.ssh_relay_host, relay_port=config.ssh_relay_port)
        except WebSocketDisconnect:
            return
        except Exception:
            try:
                await websocket.close(code=1011)
            except RuntimeError:
                pass

    return app
