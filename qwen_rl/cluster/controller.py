"""HTTP enrollment and site-scoped Controller API."""
from __future__ import annotations

import hmac
import json
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from .contracts import EnrolmentRequest, NodeHeartbeat, NodeRegistration
from .registry import Registry, RegistryError
from .settings import load_settings_json, validate_credential_origin


class ControllerSettings(BaseModel):
    """Strict settings loaded from one native JSON document."""

    model_config = ConfigDict(extra="forbid", strict=True)

    state_dir: StrictStr = Field(min_length=1, max_length=4096)
    admin_token: StrictStr = Field(min_length=32, max_length=512, repr=False)
    bootstrap_defaults: dict[str, Any]
    heartbeat_ttl_seconds: StrictInt = Field(default=45, ge=1, le=86400)
    policy_upstream_url: StrictStr | None = None
    policy_upstream_token: StrictStr | None = Field(default=None, min_length=32, max_length=512, repr=False)
    relay_state_dir: StrictStr | None = None

    @model_validator(mode="after")
    def validate_settings(self) -> "ControllerSettings":
        if not self.state_dir.strip() or any(c in self.state_dir for c in "\r\n\x00"):
            raise ValueError("state_dir must be a single-line path")
        if any(c.isspace() or c == "\x00" for c in self.admin_token):
            raise ValueError("admin_token must be a single-line bearer value")
        if not isinstance(self.bootstrap_defaults, dict):
            raise ValueError("bootstrap_defaults must be a JSON object")
        if _contains_secret_field(self.bootstrap_defaults):
            raise ValueError("bootstrap_defaults cannot contain bearer tokens or other credentials")
        try:
            json.dumps(self.bootstrap_defaults, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("bootstrap_defaults must contain finite JSON values") from exc
        if (self.policy_upstream_url is None) != (self.policy_upstream_token is None):
            raise ValueError("policy_upstream_url and policy_upstream_token must be configured together")
        if self.policy_upstream_url is not None:
            parsed = urlsplit(self.policy_upstream_url)
            try:
                parsed.port
            except ValueError as exc:
                raise ValueError("policy_upstream_url has an invalid port") from exc
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("policy_upstream_url must be an absolute HTTP(S) URL without embedded credentials")
            validate_credential_origin(self.policy_upstream_url)
        if self.policy_upstream_token is not None and any(c.isspace() or c == "\x00" for c in self.policy_upstream_token):
            raise ValueError("policy_upstream_token must be a single-line bearer value")
        resolved = Path(self.state_dir).expanduser().resolve()
        if resolved == Path("/") or resolved == Path("/etc") or Path("/etc") in resolved.parents:
            raise ValueError("state_dir cannot be / or inside /etc")
        if self.relay_state_dir is not None:
            if not self.relay_state_dir.strip() or any(c in self.relay_state_dir for c in "\r\n\x00"):
                raise ValueError("relay_state_dir must be a single-line private path")
            relay_path = Path(self.relay_state_dir).expanduser()
            if not relay_path.is_absolute():
                raise ValueError("relay_state_dir must be an absolute native path")
            resolved_relay = relay_path.resolve()
            if (resolved_relay == Path("/") or resolved_relay == Path("/etc")
                    or Path("/etc") in resolved_relay.parents):
                raise ValueError("relay_state_dir cannot be / or inside /etc")
            if resolved_relay == resolved:
                raise ValueError("relay_state_dir must be separate from state_dir")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "ControllerSettings":
        """Load strict JSON and resolve references only in declared secret fields."""
        data = load_settings_json(path, secret_fields=("admin_token", "policy_upstream_token"))
        return cls.model_validate(data, strict=True)


def _contains_secret_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            components = set(re.findall(r"[a-z0-9]+", str(key).lower()))
            if (components & {"token", "bearer", "authorization", "secret", "secrets", "password", "passwords", "credential", "credentials"}
                    or "apikey" in normalized or "privatekey" in normalized):
                return True
            if _contains_secret_field(item):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_field(item) for item in value)
    return False


def _settings(value: ControllerSettings | Mapping[str, Any] | str | Path) -> ControllerSettings:
    if isinstance(value, ControllerSettings):
        return value
    if isinstance(value, (str, Path)):
        return ControllerSettings.load(value)
    if isinstance(value, Mapping):
        return ControllerSettings.model_validate(value, strict=True)
    raise TypeError("settings must be ControllerSettings, a strict mapping, or a JSON file path")


def _bearer(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    pieces = authorization.split()
    if len(pieces) != 2 or pieces[0].lower() != "bearer" or not pieces[1]:
        return None
    return pieces[1]


def _same_secret(expected: str, supplied: str | None) -> bool:
    if supplied is None:
        return False
    return hmac.compare_digest(expected, supplied)


def _gateway_remote_ports(bootstrap_defaults: dict[str, Any]) -> tuple[int, int]:
    reverse_ssh = bootstrap_defaults.get("reverse_ssh")
    if not isinstance(reverse_ssh, dict):
        raise HTTPException(status_code=503, detail="Gateway reverse_ssh defaults are not configured")
    api_port = reverse_ssh.get("remote_api_port")
    ssh_port = reverse_ssh.get("remote_ssh_port")
    if (type(api_port) is not int or type(ssh_port) is not int
            or not 1 <= api_port <= 65535 or not 1 <= ssh_port <= 65535 or api_port == ssh_port):
        raise HTTPException(status_code=503, detail="Gateway reverse_ssh defaults need distinct remote_api_port and remote_ssh_port")
    return api_port, ssh_port


def _bootstrap_defaults_for_role(defaults: dict[str, Any], role: str) -> dict[str, Any]:
    result = dict(defaults)
    if result.get("worker_release") is not None:
        if role == "gateway":
            gateway_image = result.get("gateway_image")
            if not isinstance(gateway_image, dict) or not isinstance(gateway_image.get("image_id"), str):
                raise HTTPException(status_code=503, detail="Worker release has no separate Gateway image")
            result.pop("worker_release", None)
            result.pop("gateway_image", None)
            result["worker_image"] = gateway_image["image_id"]
            result["worker_image_mode"] = "archive"
            result["image_archive"] = gateway_image
    if role != "gateway":
        result.pop("reverse_ssh", None)
    return result


def create_controller_app(
    settings: ControllerSettings | Mapping[str, Any] | str | Path,
    registry: Registry | None = None,
) -> FastAPI:
    """Build the Controller API with independent admin and Gateway identities."""
    config = _settings(settings)
    store = registry or Registry(
        config.state_dir,
        heartbeat_ttl_seconds=config.heartbeat_ttl_seconds,
    )
    # The JSON settings remain authoritative when a test or embedding supplies
    # an already-created Registry instance.
    store.heartbeat_ttl_seconds = config.heartbeat_ttl_seconds

    app = FastAPI(title="Qwen LAN cluster Controller", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.registry = store
    app.state.settings = config

    @app.exception_handler(RegistryError)
    async def registry_error_handler(_request: Request, exc: RegistryError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    def require_admin(authorization: str | None) -> None:
        supplied = _bearer(authorization)
        if not _same_secret(config.admin_token, supplied):
            raise HTTPException(
                status_code=401,
                detail={"code": "admin_auth_required", "message": "a valid admin bearer token is required"},
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.post("/v1/enrolments")
    async def create_enrolment(payload: EnrolmentRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return store.create_enrolment(payload)

    @app.post("/v1/nodes/register")
    async def register_node(payload: NodeRegistration, request: Request):
        authorization = request.headers.get("authorization")
        supplied = _bearer(authorization)
        if authorization is not None and supplied is None:
            raise HTTPException(status_code=401, detail={"code": "invalid_authorization", "message": "use a Bearer node token"})
        # Fleet-managed Gateways use the already authenticated WSS SSH path.
        # Allocate legacy reverse listeners only when an operator explicitly
        # supplies the standalone reverse_ssh defaults.
        remote_ports = (
            _gateway_remote_ports(config.bootstrap_defaults)
            if config.bootstrap_defaults.get("reverse_ssh") is not None else None
        )
        result = store.register_node(payload, node_token=supplied, ssh_remote_ports=remote_ports)
        if config.relay_state_dir is not None:
            from .relay import RelayError, refresh_central_relay_authorized_keys

            try:
                refresh_central_relay_authorized_keys(store.state_dir, Path(config.relay_state_dir).expanduser())
            except RelayError as exc:
                raise HTTPException(
                    status_code=503,
                    detail={"code": "central_relay_refresh_failed", "message": "central relay authorization could not be refreshed"},
                ) from exc
        result["bootstrap_defaults"] = _bootstrap_defaults_for_role(config.bootstrap_defaults, payload.role)
        return result

    @app.post("/v1/nodes/{node_id}/heartbeat")
    async def heartbeat(node_id: str, payload: NodeHeartbeat, authorization: str | None = Header(default=None)):
        token = _bearer(authorization)
        if token is None:
            raise HTTPException(
                status_code=401,
                detail={"code": "node_auth_required", "message": "a valid node bearer token is required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return store.heartbeat(node_id, payload, node_token=token)

    @app.get("/v1/nodes")
    async def list_nodes(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        # Registry node views intentionally contain no bearer tokens.
        return {"nodes": store.list_nodes()}

    @app.get("/v1/sites/{site_id}/gateway")
    async def site_gateway(site_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        gateway = store.site_gateway(site_id)
        if gateway is None:
            raise HTTPException(status_code=404, detail={"code": "gateway_not_found", "message": "site has no registered Gateway"})
        return {"site_id": site_id, "gateway": gateway}

    @app.get("/v1/sites/{site_id}/nodes")
    async def list_site_nodes(site_id: str, authorization: str | None = Header(default=None)):
        token = _bearer(authorization)
        if token is None:
            raise HTTPException(
                status_code=401,
                detail={"code": "gateway_auth_required", "message": "a Gateway node bearer token is required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        principal = store.authenticate_node(token)
        if principal is None:
            raise HTTPException(status_code=401, detail={"code": "invalid_node_token", "message": "node bearer token is invalid"})
        if principal["role"] != "gateway" or principal["site_id"] != site_id:
            raise HTTPException(status_code=403, detail={"code": "site_forbidden", "message": "Gateway tokens can read only their own site"})
        return {"site_id": site_id, "nodes": store.list_site_nodes(site_id)}

    return app
