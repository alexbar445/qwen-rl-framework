"""Node-local configuration and heartbeat daemon for the LAN cluster."""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable
from urllib.parse import quote, urlsplit

import httpx
from pydantic import ValidationError
import shutil

from .bootstrap import BootstrapError, authenticated_node_rejoin
from .contracts import NodeHeartbeat
from .host import probe_gpus, read_boot_id
from .identity import IdentityError, allow_group_traverse, atomic_write, read_ssh_public_key
from .node_config import NodeConfiguration, NodeConfigurationError, load_node_config
from .worker_release import expected_worker_image_labels


class NodeDaemonError(RuntimeError):
    """Expected node-agent failure with credentials omitted."""


async def install_node_model_artifact(
    config: NodeConfiguration,
    *,
    worker_token: str,
    downloader: Any | None = None,
) -> dict[str, Any]:
    """Install the pinned model through the site's Dragonfly path only."""
    if config.role != "worker":
        raise NodeDaemonError("only Worker nodes install the frozen model artifact")
    if config.needs_driver or config.gpu_uuid is None:
        raise NodeDaemonError("node needs an NVIDIA driver and exact GPU UUID before model installation")
    if not worker_token or any(char.isspace() or char in "\r\n\x00" for char in worker_token):
        raise NodeDaemonError("Worker token is invalid")
    if downloader is None:
        from .dragonfly import DragonflyDownloader

        manifest_host = urlsplit(config.model.manifest_url).hostname
        gateway_host = urlsplit(config.gateway_url).hostname
        if not manifest_host or not gateway_host:
            raise NodeDaemonError("model or Gateway URL has no host")
        downloader = DragonflyDownloader(
            Path(config.dragonfly_dfget_path),
            Path(config.state_dir) / "artifact-cache",
            dfdaemon_path=Path(config.dragonfly_dfdaemon_path),
            dfdaemon_socket=Path(config.dragonfly_socket_path),
            allowed_hosts=[manifest_host],
            site_id=config.site_id,
            scheduler_addrs=[f"{gateway_host}:8002"],
        )
    lock_path = Path(config.state_dir) / "model-install.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    os.fchmod(descriptor, 0o600)
    try:
        await asyncio.to_thread(fcntl.flock, descriptor, fcntl.LOCK_EX)
        receipt = await downloader.install_model_artifact(
            config.model.manifest_url,
            config.model.manifest_sha256,
            config.model.cache_path,
            expected_artifact_id=config.model.artifact_id,
            expected_base_fingerprint=config.model.base_fingerprint,
            expected_artifact_fingerprint=config.model.artifact_fingerprint,
            headers={"Authorization": f"Bearer {worker_token}"},
        )
        expected = {
            "artifact_id": config.model.artifact_id,
            "manifest_sha256": config.model.manifest_sha256,
            "base_fingerprint": config.model.base_fingerprint,
            "artifact_fingerprint": config.model.artifact_fingerprint,
            "path": config.model.cache_path,
        }
        if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items()):
            raise NodeDaemonError("installed model receipt differs from the pinned manifest")
        return receipt
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


async def install_node_seed_artifact(
    config: NodeConfiguration,
    *,
    worker_token: str,
    downloader: Any | None = None,
) -> dict[str, Any]:
    """Preseed the Gateway's Dragonfly cache after central artifact publication."""
    if config.role != "gateway":
        raise NodeDaemonError("only the site Gateway may preseed the model artifact")
    if not config.dragonfly_seed_peer_path:
        raise NodeDaemonError("Gateway has no installed Dragonfly seed configuration")
    if not worker_token or any(char.isspace() or char in "\r\n\x00" for char in worker_token):
        raise NodeDaemonError("Gateway artifact bearer is invalid")
    if downloader is None:
        from .dragonfly import DragonflyDownloader, validate_dfdaemon_config

        validate_dfdaemon_config(config.dragonfly_seed_peer_path, allow_lan_source_fallback=True)
        manifest_host = urlsplit(config.model.manifest_url).hostname
        gateway_host = urlsplit(config.gateway_url).hostname
        if not manifest_host or not gateway_host:
            raise NodeDaemonError("model or Gateway URL has no host")
        downloader = DragonflyDownloader(
            shutil.which("dfget") or "/usr/local/bin/dfget",
            Path(config.state_dir) / "artifact-cache",
            allowed_hosts={manifest_host, gateway_host},
            site_id=config.site_id,
            site_role="seed",
            scheduler_addrs=[f"{gateway_host}:8002"],
        )
    from .dragonfly import preseed_model_artifact

    receipt = await preseed_model_artifact(
        downloader,
        config.model.manifest_url,
        config.model.manifest_sha256,
        expected_artifact_id=config.model.artifact_id,
        expected_base_fingerprint=config.model.base_fingerprint,
        expected_artifact_fingerprint=config.model.artifact_fingerprint,
        headers={"Authorization": f"Bearer {worker_token}"},
    )
    expected = {
        "state": "preseeded",
        "site_id": config.site_id,
        "artifact_id": config.model.artifact_id,
        "manifest_sha256": config.model.manifest_sha256,
        "base_fingerprint": config.model.base_fingerprint,
        "artifact_fingerprint": config.model.artifact_fingerprint,
    }
    if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items()):
        raise NodeDaemonError("Dragonfly seed receipt differs from the pinned model manifest")
    if config.worker_image_mode == "archive":
        image = config.image_archive
        if image is None:
            raise NodeDaemonError("archive mode requires a pinned image descriptor")
        await downloader.download(image.url, image.archive_sha256, image.size_bytes,
            headers={"Authorization": f"Bearer {worker_token}"},
            allow_lan_source_fallback=True, force_transfer=True)
        receipt = {**receipt, "image_archive_sha256": image.archive_sha256,
                   "image_id": image.image_id, "image_archive_size_bytes": image.size_bytes}
    return receipt


async def install_node_worker_image(
    config: NodeConfiguration,
    *,
    worker_token: str,
    downloader: Any | None = None,
    docker_binary: str | Path = "docker",
    docker_args_prefix: tuple[str, ...] = (),
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any]:
    """Ensure this node has its explicitly selected, pinned Worker/Gateway image."""
    if config.role != "worker":
        raise NodeDaemonError("only GPU Worker nodes install the vLLM container image")
    if config.worker_image_mode == "registry":
        argv = [str(docker_binary), *docker_args_prefix, "image", "pull", config.worker_image]
        try:
            result = await asyncio.to_thread(
                command_runner, argv, check=False, capture_output=True, text=True, timeout=7200,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise NodeDaemonError("pinned Worker image pull failed") from exc
        if getattr(result, "returncode", None) != 0:
            raise NodeDaemonError("pinned Worker image pull failed")
        receipt = {"state": "pulled", "worker_image": config.worker_image}
        atomic_write(Path(config.state_dir) / "worker-image-install-receipt.json", (
            json.dumps({**receipt, "verified_at": time.time()}, sort_keys=True) + "\n"
        ).encode("utf-8"))
        return receipt

    descriptor = config.image_archive
    if descriptor is None or descriptor.image_id != config.worker_image:
        raise NodeDaemonError("archive image mode has no matching pinned image descriptor")
    if downloader is None:
        from .dragonfly import DragonflyDownloader, validate_dfdaemon_config

        if not config.dragonfly_worker_peer_path:
            raise NodeDaemonError("Worker has no installed Dragonfly peer configuration")
        validate_dfdaemon_config(config.dragonfly_worker_peer_path, allow_lan_source_fallback=False)
        archive_host = urlsplit(descriptor.url).hostname
        gateway_host = urlsplit(config.gateway_url).hostname
        if not archive_host or not gateway_host:
            raise NodeDaemonError("Worker image or Gateway URL has no host")
        downloader = DragonflyDownloader(
            Path(config.dragonfly_dfget_path),
            Path(config.state_dir) / "artifact-cache",
            dfdaemon_path=Path(config.dragonfly_dfdaemon_path),
            dfdaemon_socket=Path(config.dragonfly_socket_path),
            allowed_hosts={archive_host},
            site_id=config.site_id,
            site_role="worker",
            scheduler_addrs=[f"{gateway_host}:8002"],
        )
    from .images import install_image_archive

    receipt = await install_image_archive(
        descriptor, downloader, worker_token=worker_token,
        docker_binary=docker_binary, docker_args_prefix=docker_args_prefix,
        command_runner=command_runner,
    )
    if (not isinstance(receipt, dict) or receipt.get("state") != "loaded_verified"
            or receipt.get("image_id") != descriptor.image_id
            or receipt.get("archive_sha256") != descriptor.archive_sha256
            or receipt.get("size_bytes") != descriptor.size_bytes):
        raise NodeDaemonError("loaded Worker image receipt differs from the pinned archive descriptor")
    if config.worker_release_selection is not None:
        argv = [str(docker_binary), *docker_args_prefix, "image", "inspect", "--format",
                "{{json .Config.Labels}}", config.worker_image]
        try:
            label_result = await asyncio.to_thread(
                command_runner, argv, check=False, capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise NodeDaemonError("Worker release image label inspection failed") from exc
        if getattr(label_result, "returncode", None) != 0:
            raise NodeDaemonError("Worker release image label inspection failed")
        try:
            image_labels = json.loads(label_result.stdout)
        except (json.JSONDecodeError, TypeError):
            raise NodeDaemonError("Worker release image has malformed Docker labels") from None
        expected_labels = expected_worker_image_labels(config.worker_release_selection)
        if (not isinstance(image_labels, dict)
                or any(image_labels.get(key) != value for key, value in expected_labels.items())):
            raise NodeDaemonError("Worker release image labels do not match the pinned release catalog")
        selection = config.worker_release_selection
        receipt = {
            **receipt,
            "worker_release_selection": {
                "release_identity": selection.release_identity.model_dump(mode="json"),
                "compute_capability": selection.compute_capability,
                "build_arch": selection.build_arch,
                "cuda_version": selection.cuda_version,
            },
        }
    persisted_receipt = {key: receipt[key] for key in (
        "state", "image_id", "archive_sha256", "size_bytes", "url_host",
    )}
    if "worker_release_selection" in receipt:
        persisted_receipt["worker_release_selection"] = receipt["worker_release_selection"]
    atomic_write(Path(config.state_dir) / "worker-image-install-receipt.json", (
        json.dumps(persisted_receipt | {"verified_at": time.time()}, sort_keys=True) + "\n"
    ).encode("utf-8"))
    return receipt


class NodeDaemon:
    """Report this node's state using only its node-scoped bearer token."""

    def __init__(
        self,
        config_path: Path,
        *,
        http_transport: httpx.AsyncBaseTransport | None = None,
        local_transport: httpx.AsyncBaseTransport | None = None,
        boot_id_provider: Callable[[], str] = read_boot_id,
        model_installer: Callable[[NodeConfiguration], Any] | None = None,
        seed_installer: Callable[[NodeConfiguration], Any] | None = None,
        request_timeout: float = 10.0,
    ) -> None:
        try:
            self.config, self.credentials = load_node_config(config_path)
        except (BootstrapError, IdentityError, NodeConfigurationError, ValidationError) as exc:
            raise NodeDaemonError("node state is invalid or has unsafe permissions") from exc
        self.config_path = Path(config_path)
        self.http_transport = http_transport
        self.local_transport = local_transport
        self.boot_id_provider = boot_id_provider
        self.model_installer = model_installer
        self.seed_installer = seed_installer
        self.request_timeout = request_timeout
        self.model_verified = self.config.role == "gateway"
        self.model_install_error: str | None = None
        self._model_install_task: asyncio.Task | None = None
        self.seed_verified = self.config.role != "gateway"
        self.seed_install_error: str | None = None
        self._seed_install_task: asyncio.Task | None = None

    async def _install_model_artifact(self) -> None:
        if self.config.role != "worker":
            self.model_verified = True
            return
        backoff = 2.0
        while not self.model_verified:
            try:
                if self.model_installer is None:
                    await install_node_model_artifact(self.config, worker_token=self.worker_token)
                else:
                    receipt = self.model_installer(self.config)
                    if asyncio.iscoroutine(receipt):
                        receipt = await receipt
                    expected = {
                        "artifact_id": self.config.model.artifact_id,
                        "manifest_sha256": self.config.model.manifest_sha256,
                        "base_fingerprint": self.config.model.base_fingerprint,
                        "artifact_fingerprint": self.config.model.artifact_fingerprint,
                        "path": self.config.model.cache_path,
                    }
                    if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items()):
                        raise NodeDaemonError("installed model receipt differs from the pinned manifest")
                self.model_verified = True
                self.model_install_error = None
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.model_verified = False
                self.model_install_error = type(exc).__name__
                logging.getLogger(__name__).warning(
                    "model artifact install failed (%s); retrying after %.0f seconds",
                    type(exc).__name__, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _seed_model_artifact(self) -> None:
        """Retry until the Gateway can fetch and P2P-seed the centrally published blobs."""
        if self.config.role != "gateway":
            self.seed_verified = True
            return
        backoff = 2.0
        while not self.seed_verified:
            try:
                if self.seed_installer is None:
                    receipt = await install_node_seed_artifact(self.config, worker_token=self.worker_token)
                else:
                    receipt = self.seed_installer(self.config)
                    if asyncio.iscoroutine(receipt):
                        receipt = await receipt
                expected = {
                    "state": "preseeded",
                    "site_id": self.config.site_id,
                    "artifact_id": self.config.model.artifact_id,
                    "manifest_sha256": self.config.model.manifest_sha256,
                    "base_fingerprint": self.config.model.base_fingerprint,
                    "artifact_fingerprint": self.config.model.artifact_fingerprint,
                }
                if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items()):
                    raise NodeDaemonError("Dragonfly seed receipt differs from the pinned model manifest")
                if self.config.worker_image_mode == "archive":
                    image = self.config.image_archive
                    if image is None or receipt.get("image_archive_sha256") != image.archive_sha256 or receipt.get("image_id") != image.image_id:
                        raise NodeDaemonError("Dragonfly seed receipt omits the pinned Worker image")
                self.seed_verified = True
                self.seed_install_error = None
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.seed_verified = False
                self.seed_install_error = type(exc).__name__
                logging.getLogger(__name__).warning(
                    "model seed is not ready (%s); retrying after %.0f seconds",
                    type(exc).__name__, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    @property
    def node_token(self) -> str:
        return self.credentials["node_token"]

    @property
    def worker_token(self) -> str:
        return self.credentials["worker_token"]

    async def _local_health(self) -> tuple[dict[str, Any] | None, bool, str | None]:
        if self.config.role == "worker":
            if not self.config.worker_url:
                return None, False, "worker_url_missing"
            url = f"{self.config.worker_url}/health"
            token = self.worker_token
        else:
            url = f"{self.config.gateway_url}/health"
            token = self.worker_token
        try:
            async with httpx.AsyncClient(
                transport=self.local_transport,
                timeout=self.request_timeout,
                follow_redirects=False,
            ) as client:
                response = await client.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
        except httpx.HTTPError:
            return None, False, "local_health_unavailable"
        if response.status_code != 200 or len(response.content) > 2_000_000:
            return None, False, f"local_health_http_{response.status_code}"
        try:
            health = response.json()
        except ValueError:
            return None, False, "local_health_invalid_json"
        if not isinstance(health, dict):
            return None, False, "local_health_invalid_shape"
        expected_role = "worker" if self.config.role == "worker" else "gateway"
        role = health.get("role")
        # A Gateway may use a narrower status envelope, but it must identify its role.
        if (expected_role == "worker" and role != "worker") or (
            expected_role == "gateway" and role not in {"gateway", None}
        ):
            return health, False, "local_health_role_mismatch"
        ready = health.get("ready") is True
        if expected_role == "worker":
            ready = (
                ready and health.get("policy_ready") is True and self.model_verified
                and health.get("base_fingerprint") == self.config.model.base_fingerprint
                and health.get("artifact_fingerprint") == self.config.model.artifact_fingerprint
            )
            if self.model_install_error:
                return health, False, f"model_install_{self.model_install_error}"
            if health.get("base_fingerprint") != self.config.model.base_fingerprint or health.get("artifact_fingerprint") != self.config.model.artifact_fingerprint:
                return health, False, "model_fingerprint_mismatch"
            if not self.model_verified:
                return health, False, "model_artifact_not_verified"
        return health, ready, None if ready else "local_service_not_ready"

    def _persist_boot_id(self, boot_id: str) -> None:
        if self.config.boot_id == boot_id:
            return
        self.config = self.config.model_copy(update={"boot_id": boot_id})
        atomic_write(self.config_path, (json.dumps(self.config.model_dump(mode="json"), sort_keys=True,
                                                   indent=2, ensure_ascii=False) + "\n").encode())
        if self.config.role == "gateway":
            allow_group_traverse(Path(self.config.state_dir), gid=self.config.worker_gid)

    async def _authenticated_rejoin(self, boot_id: str) -> None:
        reverse_public = read_ssh_public_key(Path(self.config.reverse_ssh_key_path).with_suffix(".pub"))
        host_public = read_ssh_public_key(Path(self.config.host_key_path).with_suffix(".pub"))
        gpus = probe_gpus() if self.config.role == "worker" else []
        worker_port = 8241
        if self.config.worker_url:
            worker_port = httpx.URL(self.config.worker_url).port or 8241
        try:
            await authenticated_node_rejoin(
                config=self.config,
                boot_id=boot_id,
                node_token=self.node_token,
                worker_token=self.worker_token,
                reverse_public_key=reverse_public,
                host_public_key=host_public,
                gpus=gpus,
                worker_port=worker_port,
                transport=self.http_transport,
            )
        except BootstrapError as exc:
            raise NodeDaemonError("authenticated rejoin failed") from exc
        self._persist_boot_id(boot_id)

    async def heartbeat_once(self) -> dict[str, Any]:
        boot_id = self.boot_id_provider()
        if self.config.boot_id != boot_id:
            await self._authenticated_rejoin(boot_id)
        self._persist_boot_id(boot_id)
        if self.config.role == "worker" and self.config.needs_driver:
            health, ready, error = None, False, "needs_driver"
            state = "registered"
        elif self.config.needs_host_packages:
            health, ready, error = None, False, "needs_host_packages"
            state = "registered"
        else:
            health, ready, error = await self._local_health()
            state = "ready" if ready else "loading"
            if self.config.role == "gateway" and not self.seed_verified:
                error = f"model_seed_{self.seed_install_error or 'pending'}"
        image_digest = None
        if self.config.role == "worker":
            image_digest = (self.config.worker_image.split("@", 1)[1]
                            if "@" in self.config.worker_image else self.config.worker_image)
        payload = NodeHeartbeat(
            boot_id=boot_id,
            state=state,
            worker_health=health,
            image_digest=image_digest,
            error=error,
        )
        endpoint = f"{self.config.controller_url}/v1/nodes/{quote(self.config.node_id, safe='')}/heartbeat"
        try:
            async with httpx.AsyncClient(
                transport=self.http_transport,
                timeout=self.request_timeout,
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    endpoint,
                    json=payload.model_dump(mode="json"),
                    headers={"Authorization": f"Bearer {self.node_token}", "Accept": "application/json"},
                )
        except httpx.HTTPError as exc:
            raise NodeDaemonError("Controller heartbeat request failed") from exc
        if response.status_code == 409:
            try:
                error = response.json()
            except ValueError:
                error = {}
            detail = (error.get("detail") or error.get("error") or {}) if isinstance(error, dict) else {}
            if isinstance(detail, dict) and detail.get("code") == "rejoin_required":
                await self._authenticated_rejoin(boot_id)
                async with httpx.AsyncClient(
                    transport=self.http_transport,
                    timeout=self.request_timeout,
                    follow_redirects=False,
                ) as client:
                    response = await client.post(
                        endpoint,
                        json=payload.model_dump(mode="json"),
                        headers={"Authorization": f"Bearer {self.node_token}", "Accept": "application/json"},
                    )
        if response.status_code not in {200, 204}:
            raise NodeDaemonError(f"Controller heartbeat returned HTTP {response.status_code}")
        if response.status_code == 204:
            return {"node_id": self.config.node_id, "state": payload.state, "worker_health": health}
        if len(response.content) > 1_000_000:
            raise NodeDaemonError("Controller heartbeat response is too large")
        try:
            result = response.json()
        except ValueError as exc:
            raise NodeDaemonError("Controller heartbeat response is not JSON") from exc
        if not isinstance(result, dict):
            raise NodeDaemonError("Controller heartbeat response must be an object")
        return result

    async def run(self, *, stop_event: asyncio.Event | None = None) -> None:
        stop_event = stop_event or asyncio.Event()
        interval = self.config.heartbeat_interval_seconds
        if (self.config.role == "worker" and not self.config.needs_driver
                and not self.config.needs_host_packages and self._model_install_task is None):
            self._model_install_task = asyncio.create_task(self._install_model_artifact())
        if (self.config.role == "gateway" and not self.config.needs_host_packages
                and self._seed_install_task is None):
            self._seed_install_task = asyncio.create_task(self._seed_model_artifact())
        while not stop_event.is_set():
            try:
                await self.heartbeat_once()
            except NodeDaemonError:
                # The service manager owns restart policy; do not fabricate a healthy state.
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except TimeoutError:
                continue
        for task in (self._model_install_task, self._seed_install_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


def load_worker_configuration(node_config_path: Path) -> dict[str, Any]:
    """Read the generated local Worker config and verify its private token file."""
    try:
        config, credentials = load_node_config(node_config_path)
    except (BootstrapError, IdentityError, NodeConfigurationError) as exc:
        raise NodeDaemonError("node state is invalid") from exc
    if config.role != "worker" or not config.worker_config_path or not config.worker_token_path:
        raise NodeDaemonError("this node does not have a Worker configuration")
    path = Path(config.worker_config_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NodeDaemonError("Worker configuration cannot be read") from exc
    if not isinstance(raw, dict) or raw.get("worker_id") != config.node_id:
        raise NodeDaemonError("Worker configuration identity is invalid")
    try:
        info = path.lstat()
    except OSError as exc:
        raise NodeDaemonError("Worker configuration file is missing") from exc
    if (not path.is_file() or path.is_symlink() or info.st_mode & 0o077
            or info.st_uid != config.worker_uid or info.st_gid != config.worker_gid):
        raise NodeDaemonError("Worker configuration must be a private regular file")
    try:
        token_path = Path(config.worker_token_path)
        env_text = token_path.read_text(encoding="ascii")
        env_info = token_path.lstat()
    except (OSError, UnicodeError) as exc:
        raise NodeDaemonError("Worker token environment file cannot be read") from exc
    if (env_text != credentials["worker_token"] or env_info.st_mode & 0o077
            or not token_path.is_file() or token_path.is_symlink()
            or env_info.st_uid != config.worker_uid or env_info.st_gid != config.worker_gid):
        raise NodeDaemonError("Worker token must be a mode 0600 per-node secret file")
    return raw
