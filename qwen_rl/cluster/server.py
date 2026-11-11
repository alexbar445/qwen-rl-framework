"""Executable service factories; private Learner credentials remain central."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import httpx

from .controller import ControllerSettings, create_controller_app


def read_token_file(path: str | Path) -> str:
    return load_settings_json_value(path)


def load_settings_json_value(path: str | Path) -> str:
    # Reuse the same permissions contract as file: settings without a second
    # settings parser or accepting inline credentials on the command line.
    import stat
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("token must be a private regular file")
    value = source.read_text().strip()
    if not value or any(char.isspace() for char in value):
        raise ValueError("token must be a nonempty single-line value")
    return value


def create_controller_server(settings: ControllerSettings | str | Path) -> FastAPI:
    from .artifacts import ArtifactMirror, create_artifact_router
    config = settings if isinstance(settings, ControllerSettings) else ControllerSettings.load(settings)
    app = create_controller_app(config)
    # Bootstrap source contains no enrolment codes, keys, datasets or weights.
    # The join command pins each object's SHA and uses the public TLS endpoint.
    @app.get("/v1/bootstrap/{sha256}")
    async def bootstrap_object(sha256: str):
        import asyncio
        import hashlib
        import re
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise HTTPException(422, "invalid bootstrap content SHA")
        path = Path(config.state_dir) / "bootstrap" / sha256
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, "bootstrap content unavailable")
        def verified():
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    digest.update(chunk)
            return digest.hexdigest() == sha256
        if not await asyncio.to_thread(verified):
            raise HTTPException(503, "bootstrap content integrity failure")
        return FileResponse(path, media_type="application/octet-stream", headers={
            "ETag": f'"sha256:{sha256}"', "Cache-Control": "public, max-age=31536000, immutable"})
    if config.policy_upstream_url is not None:
        lineage = config.bootstrap_defaults.get("lineage_id")
        if not isinstance(lineage, str) or not lineage:
            raise ValueError("a policy origin requires an explicit lineage_id")
        mirror = ArtifactMirror(config.policy_upstream_url, Path(config.state_dir) / "policy-mirror",
            headers={"Authorization": f"Bearer {config.policy_upstream_token}"}, lineage_id=lineage)

        async def require_policy_reader(authorization: str | None = Header(default=None)):
            token = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
            if not hmac.compare_digest(token, config.admin_token) and app.state.registry.gateway_site(token) is None:
                raise HTTPException(401, "registered Gateway identity required")

        app.include_router(create_artifact_router(mirror), dependencies=[Depends(require_policy_reader)])
        app.state.policy_mirror = mirror
        original = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(server):
            try:
                async with original(server):
                    yield
            finally:
                await mirror.aclose()

        app.router.lifespan_context = lifespan
    return app


def inspect_selected_gpu(expected_uuid: str) -> dict[str, Any]:
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("cluster Worker must see exactly one selected GPU")
    props = torch.cuda.get_device_properties(0)
    observed = str(props.uuid)
    if observed.removeprefix("GPU-") != expected_uuid.removeprefix("GPU-"):
        raise RuntimeError("visible CUDA device differs from selected GPU UUID")
    return {"gpu_uuid": expected_uuid, "gpu_name": props.name,
            "gpu_memory_total_mib": int(props.total_memory // (1024 * 1024)),
            "gpu_compute_capability": f"{props.major}.{props.minor}"}


def read_kernel_boot_id(path: str | Path = "/proc/sys/kernel/random/boot_id") -> str:
    boot_id = Path(path).read_text(encoding="ascii").strip()
    if not boot_id or any(char.isspace() for char in boot_id):
        raise RuntimeError("Worker could not read the kernel boot ID")
    return boot_id


def create_worker_server(config: dict[str, Any], *, token_file: str | Path | None = None,
                         model_loader: Callable | None = None,
                         gpu_probe: Callable = inspect_selected_gpu,
                         boot_probe: Callable[[], str] = read_kernel_boot_id,
                         http_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    from ..worker.app import create_worker_app
    if not isinstance(config, dict):
        raise ValueError("Worker config must be an object")
    deployment_scope = config.get("deployment_scope")
    artifact_transport = config.get("artifact_transport")
    if (deployment_scope, artifact_transport) not in {
        ("service", "direct"),
        ("gateway", "dragonfly"),
    }:
        raise ValueError(
            "Worker config must explicitly select service/direct or gateway/dragonfly deployment"
        )
    worker_config = dict(config)
    worker_config.pop("deployment_scope")
    worker_config.pop("artifact_transport")
    configured_node_id = worker_config.pop("node_id", None)
    environment_node_id = os.environ.get("QWEN_NODE_ID")
    if configured_node_id is not None and (
        not isinstance(configured_node_id, str) or not configured_node_id.strip()
    ):
        raise ValueError("Worker node_id must be a non-empty string when configured")
    if environment_node_id and configured_node_id and environment_node_id != configured_node_id:
        raise ValueError("QWEN_NODE_ID differs from the Worker config node_id")
    if token_file is not None:
        os.environ["ARC_ENV_TOKEN"] = read_token_file(token_file)
    if not os.environ.get("ARC_ENV_TOKEN"):
        raise ValueError("Worker requires its node-scoped endpoint token")
    expected_uuid = os.environ.get("GPU_UUID")
    if not expected_uuid:
        raise ValueError("GPU_UUID must identify the container's selected device")
    if artifact_transport == "dragonfly" and http_transport is None:
        from urllib.parse import urlsplit
        from .adapter_transport import DragonflyWorkerTransport
        from .dragonfly import DragonflyDownloader
        site_id = os.environ.get("QWEN_SITE_ID")
        if not site_id:
            raise ValueError("QWEN_SITE_ID is required for the node's P2P artifact namespace")
        gateway_host = urlsplit(config["learner_url"]).hostname
        if not gateway_host:
            raise ValueError("Worker requires a site Gateway origin")
        downloader = DragonflyDownloader("/usr/local/bin/dfget",
            Path(config["adapter_cache"]).parent / "p2p-artifacts",
            allowed_hosts=[gateway_host], site_id=site_id)
        http_transport = DragonflyWorkerTransport(downloader)
    inner = create_worker_app(worker_config, model_loader=model_loader, http_transport=http_transport)
    device: dict[str, Any] = {}
    node_id = environment_node_id or configured_node_id or config["worker_id"]

    @asynccontextmanager
    async def lifespan(_app):
        boot_id = boot_probe()
        if not isinstance(boot_id, str) or not boot_id.strip() or any(char.isspace() for char in boot_id):
            raise RuntimeError("Worker boot probe must return a non-empty kernel boot ID")
        device["boot_id"] = boot_id
        device.update(gpu_probe(expected_uuid))
        async with inner.router.lifespan_context(inner):
            yield

    outer = FastAPI(title="Qwen cluster GPU Worker", docs_url=None, redoc_url=None,
                    openapi_url=None, lifespan=lifespan)
    outer.state.worker_app = inner

    def current_identity() -> tuple[str | None, str | None]:
        runtime = getattr(inner.state, "runtime", None)
        telemetry_epoch = getattr(runtime, "telemetry_epoch", None)
        return device.get("boot_id"), telemetry_epoch if isinstance(telemetry_epoch, str) else None

    @outer.middleware("http")
    async def fence_worker_identity(request: Request, call_next):
        path = request.url.path.rstrip("/") or "/"
        if path in {"/policy", "/policy/prepare", "/episode/acquire", "/generate", "/residency"}:
            worker_token = os.environ.get("ARC_ENV_TOKEN")
            if (not worker_token
                    or not hmac.compare_digest(
                        request.headers.get("authorization", ""),
                        f"Bearer {worker_token}",
                    )):
                return await call_next(request)
            current_boot_id, current_telemetry_epoch = current_identity()
            expected_boot_id = request.headers.get("x-expected-worker-boot-id")
            expected_telemetry_epoch = request.headers.get("x-expected-worker-telemetry-epoch")
            if not current_boot_id or not current_telemetry_epoch:
                response = JSONResponse(
                    {"error": {"code": "worker_identity_unavailable",
                                "message": "Worker boot or process identity is unavailable"}},
                    status_code=503,
                )
                return add_identity_headers(response)
            if (expected_boot_id != current_boot_id
                    or expected_telemetry_epoch != current_telemetry_epoch):
                response = JSONResponse(
                    {"error": {"code": "worker_epoch_changed",
                                "message": "Worker boot or process epoch differs from the pinned identity"}},
                    status_code=409,
                )
                return add_identity_headers(response)
        response = await call_next(request)
        if response.status_code == 401:
            return response
        return add_identity_headers(response)

    def add_identity_headers(response):
        boot_id, telemetry_epoch = current_identity()
        response.headers["X-Worker-Node-ID"] = node_id
        if boot_id:
            response.headers["X-Worker-Boot-ID"] = boot_id
        if telemetry_epoch:
            response.headers["X-Worker-Telemetry-Epoch"] = telemetry_epoch
        return response

    @outer.get("/health")
    async def health(request: Request):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=inner), base_url="http://worker") as client:
            reply = await client.get("/health", headers={"Authorization": request.headers.get("authorization", "")})
        if reply.status_code != 200:
            return JSONResponse(reply.json(), status_code=reply.status_code)
        payload = reply.json()
        payload.update(device)
        payload["boot_id"] = device.get("boot_id")
        payload["deployment"] = {"scope": deployment_scope,
                                 "node_id": node_id,
                                 "site_id": os.environ.get("QWEN_SITE_ID")}
        return payload

    outer.mount("/", inner)
    return outer
