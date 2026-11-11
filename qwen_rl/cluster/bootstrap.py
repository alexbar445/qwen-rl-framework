"""Compose node enrollment and authenticated re-registration.

Identity, typed configuration, and host prerequisites live in their own
modules. This module sequences them around the Controller registration API.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

import httpx as _httpx
from pydantic import ValidationError as _ValidationError

from . import contracts as _contracts
from . import host as _host
from . import identity as _identity
from . import node_config as _node_config

_TOKEN = re.compile(r"^[A-Za-z0-9_-]{16,512}$")
_SITE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class BootstrapError(RuntimeError):
    """Expected, user-safe failure while composing node enrollment."""


def _make_registration(
    *, code: str, site_id: str, node_id: str, boot_id: str, role: str,
    lan_host: str, worker_port: int, ssh_port: int,
    ssh_public_key: str, ssh_host_public_key: str, gpus: Sequence[_contracts.GPUCapability],
) -> _contracts.NodeRegistration:
    try:
        return _contracts.NodeRegistration(
            code=code, site_id=site_id, node_id=node_id, boot_id=boot_id, role=role,
            lan_host=lan_host, worker_port=worker_port, ssh_port=ssh_port,
            ssh_public_key=ssh_public_key, ssh_host_public_key=ssh_host_public_key,
            gpus=list(gpus),
        )
    except _ValidationError as exc:
        raise BootstrapError(f"node registration fields are invalid: {exc.errors()[0]['loc']}") from exc


async def _post_registration(
    *, controller_url: str, registration: _contracts.NodeRegistration, node_token: str | None,
    transport: _httpx.AsyncBaseTransport | None, ca_bundle: str | Path | bool = True,
    timeout_seconds: float = 30,
) -> dict[str, Any]:
    try:
        controller_url = _node_config.validate_controller_url(controller_url)
    except _node_config.NodeConfigurationError as exc:
        raise BootstrapError(str(exc)) from exc
    headers = {"Accept": "application/json"}
    if node_token:
        if not _TOKEN.fullmatch(node_token):
            raise BootstrapError("stored node bearer token is invalid")
        headers["Authorization"] = f"Bearer {node_token}"
    try:
        async with _httpx.AsyncClient(verify=ca_bundle, transport=transport, timeout=timeout_seconds,
                                     follow_redirects=False) as client:
            response = await client.post(
                f"{controller_url}/v1/nodes/register", json=registration.model_dump(mode="json"), headers=headers,
            )
    except _httpx.HTTPError as exc:
        raise BootstrapError("HTTPS node registration failed") from exc
    if response.status_code != 200:
        raise BootstrapError(f"node registration returned HTTP {response.status_code}")
    if len(response.content) > 1_000_000:
        raise BootstrapError("node registration response is too large")
    if response.request.url.scheme != "https":
        raise BootstrapError("node registration response did not use HTTPS")
    try:
        payload = response.json()
    except ValueError as exc:
        raise BootstrapError("node registration response is not JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"node", "node_token", "worker_token", "bootstrap_defaults"}:
        raise BootstrapError("node registration response does not match the T1 contract")
    node = payload["node"]
    if not isinstance(node, dict) or node.get("node_id") != registration.node_id or node.get("site_id") != registration.site_id:
        raise BootstrapError("node registration response identity does not match this host")
    for key in ("node_token", "worker_token"):
        if not isinstance(payload[key], str) or not _TOKEN.fullmatch(payload[key]):
            raise BootstrapError(f"node registration returned an invalid {key}")
    try:
        defaults = _node_config.BootstrapDefaults.model_validate(payload["bootstrap_defaults"])
    except _ValidationError as exc:
        raise BootstrapError("node registration bootstrap_defaults do not match the T4 schema") from exc
    if registration.role == "gateway" and defaults.reverse_ssh is not None:
        parsed_gateway = urlsplit(defaults.gateway_url)
        gateway_port = parsed_gateway.port or (443 if parsed_gateway.scheme == "https" else 80)
        if defaults.reverse_ssh.local_api_port != gateway_port:
            raise BootstrapError("reverse_ssh.local_api_port must match the Gateway URL port")
    return {**payload, "bootstrap_defaults": defaults.model_dump(mode="json")}




async def register_node(
    *,
    controller_url: str,
    code: str,
    site_id: str,
    node_id: str,
    boot_id: str,
    role: str,
    lan_host: str,
    worker_port: int,
    ssh_port: int,
    ssh_public_key: str,
    ssh_host_public_key: str,
    gpus: Sequence[_contracts.GPUCapability],
    node_token: str | None = None,
    transport: _httpx.AsyncBaseTransport | None = None,
    ca_bundle: str | Path | bool = True,
) -> dict[str, Any]:
    """Validate and register one node, returning the Controller receipt."""
    registration = _make_registration(
        code=code,
        site_id=site_id,
        node_id=node_id,
        boot_id=boot_id,
        role=role,
        lan_host=lan_host,
        worker_port=worker_port,
        ssh_port=ssh_port,
        ssh_public_key=ssh_public_key,
        ssh_host_public_key=ssh_host_public_key,
        gpus=gpus,
    )
    return await _post_registration(
        controller_url=controller_url,
        registration=registration,
        node_token=node_token,
        transport=transport,
        ca_bundle=ca_bundle,
    )


async def authenticated_node_rejoin(
    *,
    config: _node_config.NodeConfiguration,
    boot_id: str,
    node_token: str,
    worker_token: str,
    reverse_public_key: str,
    host_public_key: str,
    gpus: Sequence[_contracts.GPUCapability],
    worker_port: int,
    transport: _httpx.AsyncBaseTransport | None = None,
) -> None:
    """Re-register durable node identity after its Linux boot ID changes."""
    response = await register_node(
        controller_url=config.controller_url,
        code="authenticated-node-rejoin",
        site_id=config.site_id,
        node_id=config.node_id,
        boot_id=boot_id,
        role=config.role,
        lan_host=config.lan_host,
        worker_port=worker_port,
        ssh_port=config.ssh_port,
        ssh_public_key=reverse_public_key,
        ssh_host_public_key=host_public_key,
        gpus=gpus,
        node_token=node_token,
        transport=transport,
    )
    if response["node_token"] != node_token or response["worker_token"] != worker_token:
        raise BootstrapError("authenticated rejoin returned different node credentials")


async def join_cluster(
    *,
    controller_url: str,
    site_id: str,
    code: str,
    role: str,
    state_dir: Path,
    lan_host: str | None = None,
    ssh_port: int = 2222,
    worker_port: int = 8241,
    gpu_uuid: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    http_transport: _httpx.AsyncBaseTransport | None = None,
    ca_bundle: str | Path | bool = True,
    boot_id: str | None = None,
    host_public_key: str | None = None,
    gpu_inventory: Sequence[_contracts.GPUCapability] | None = None,
    model_cache_path: Path | None = None,
    worker_compose_file: Path | None = None,
    container_uid: int | None = None,
    container_gid: int | None = None,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
    dragonfly_dfget_path: Path | None = None,
    dragonfly_dfdaemon_path: Path | None = None,
    dragonfly_socket_path: Path | None = None,
    write_training_ssh_state: bool = True,
    key_creator: Callable[[Path, Path, str], None] | None = None,
) -> _node_config.NodeConfiguration:
    """Compose durable identity, host facts, registration, and node configuration."""
    try:
        controller_url = _node_config.validate_controller_url(controller_url)
        if not _SITE_ID.fullmatch(site_id):
            raise BootstrapError("site_id is invalid")
        if role not in {"gateway", "worker"}:
            raise BootstrapError("role must be gateway or worker")
        if type(write_training_ssh_state) is not bool:
            raise BootstrapError("write_training_ssh_state must be a boolean")
        if not write_training_ssh_state and key_creator is None:
            raise BootstrapError("training SSH disabled enrollment requires a real SSH key creator")
        if (owner_uid is None) != (owner_gid is None):
            raise BootstrapError("host owner UID and GID must be provided together")
        if (container_uid is None) != (container_gid is None):
            raise BootstrapError("container UID and GID must be provided together")
        if (owner_uid is None) != (container_uid is None):
            raise BootstrapError("user-owned node state requires a container UID/GID mapping")
        if owner_uid is not None and (owner_uid == 0 or os.geteuid() != owner_uid or os.getegid() != owner_gid):
            raise BootstrapError("user-owned node state requires the invoking non-root account")
        if owner_uid is not None and any(path is None for path in (
            dragonfly_dfget_path, dragonfly_dfdaemon_path, dragonfly_socket_path,
        )):
            raise BootstrapError("user-owned node state requires explicit Dragonfly binary and socket paths")
        if container_uid is not None:
            if (isinstance(container_uid, bool) or not isinstance(container_uid, int) or container_uid < 0
                    or isinstance(container_gid, bool) or not isinstance(container_gid, int) or container_gid < 0):
                raise BootstrapError("container UID and GID must be nonnegative integers")
            if owner_uid is None and (container_uid == 0 or container_gid == 0):
                raise BootstrapError("rootless container identities require account-owned node state")
        if role != "worker" and (model_cache_path is not None or worker_compose_file is not None):
            raise BootstrapError("model cache and Worker Compose paths apply only to Worker enrollment")
        if not isinstance(code, str) or len(code) < 16 or "\n" in code or "\x00" in code:
            raise BootstrapError("enrollment code is invalid")
        state_dir = _identity.ensure_private_dir(Path(state_dir))
        identity = _identity.NodeIdentity(state_dir, runner=runner, key_creator=key_creator)
        node_id, public_key, durable_host_key = identity.load_or_create()
        chosen_lan_host = lan_host or _host.detect_lan_host(runner=runner)
        if host_public_key is not None:
            try:
                matches = _identity.canonical_public_key(host_public_key) == _identity.canonical_public_key(durable_host_key)
            except ValueError as exc:
                raise BootstrapError("provided SSH host key is invalid") from exc
            if not matches:
                raise BootstrapError("provided SSH host key differs from durable node identity")
        current_boot_id = boot_id or _host.read_boot_id()
        gpus = (
            list(gpu_inventory) if gpu_inventory is not None else _host.probe_gpus(runner=runner)
        ) if role == "worker" else []
        chosen_gpu = _host.selected_gpu(gpus, gpu_uuid) if role == "worker" and gpus else None
        registration = _make_registration(
            code=code,
            site_id=site_id,
            node_id=node_id,
            boot_id=current_boot_id,
            role=role,
            lan_host=chosen_lan_host,
            worker_port=worker_port,
            ssh_port=ssh_port,
            ssh_public_key=public_key,
            ssh_host_public_key=durable_host_key,
            gpus=gpus,
        )
        response = await _post_registration(
            controller_url=controller_url,
            registration=registration,
            node_token=_node_config.node_token_for_rejoin(state_dir, controller_url=controller_url, site_id=site_id),
            transport=http_transport,
            ca_bundle=ca_bundle,
        )
        return _node_config.prepare_node_configuration(
            state_dir=state_dir,
            controller_url=controller_url,
            registration=registration,
            identity=identity,
            response=response,
            chosen_gpu=chosen_gpu,
            runner=runner,
            model_cache_path=model_cache_path,
            worker_compose_file=worker_compose_file,
            container_uid=container_uid,
            container_gid=container_gid,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            dragonfly_dfget_path=dragonfly_dfget_path,
            dragonfly_dfdaemon_path=dragonfly_dfdaemon_path,
            dragonfly_socket_path=dragonfly_socket_path,
            write_training_ssh_state=write_training_ssh_state,
        )
    except BootstrapError:
        raise
    except (_identity.IdentityError, _host.HostError, _node_config.NodeConfigurationError) as exc:
        raise BootstrapError(str(exc)) from exc
    except (_ValidationError, ValueError) as exc:
        raise BootstrapError("node enrollment data is invalid") from exc
