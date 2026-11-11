"""Activate one enrolled node with isolated systemd and role-specific Compose."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .host import (
    HostError,
    HostPrerequisites,
    install_dragonfly_binaries,
    preflight_host,
    write_system_file,
)
from .identity import (
    allow_group_traverse,
    atomic_write,
    reverse_ssh_argv,
)
from .node_config import (
    NodeConfiguration,
    load_node_config,
)
from .node import install_node_model_artifact, install_node_worker_image


class NodeInstallError(RuntimeError):
    """Expected, user-safe host activation failure."""


def _systemd_arg(value: str | Path) -> str:
    text = str(value)
    if not text or any(char in text for char in "\r\n\x00"):
        raise NodeInstallError("systemd arguments must be nonempty single-line values")
    if not any(char.isspace() or char in '\\"%$' for char in text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def _write_unit(unit_dir: Path, name: str, content: str) -> Path:
    if not name.startswith("qwen-cluster-") or not name.endswith(".service") or "/" in name:
        raise NodeInstallError("installer unit name is outside the Qwen cluster namespace")
    path = Path(unit_dir) / name
    try:
        write_system_file(path, content.encode("utf-8"), mode=0o644)
    except HostError as exc:
        raise NodeInstallError(f"could not safely install systemd unit {name}") from exc
    return path


def _run(argv: list[str], runner: Callable[..., subprocess.CompletedProcess], *, timeout: float = 180.0) -> None:
    if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
        raise NodeInstallError("invalid installer command arguments")
    try:
        result = runner(argv, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise NodeInstallError(f"could not run {Path(argv[0]).name}") from exc
    if result.returncode != 0:
        raise NodeInstallError(f"{Path(argv[0]).name} exited with status {result.returncode}")


def _load_renderer(project_root: Path):
    renderer_path = project_root / "deploy" / "dragonfly" / "render_config.py"
    if not renderer_path.is_file() or renderer_path.is_symlink():
        raise NodeInstallError("pinned Dragonfly site config renderer is missing from the installation")
    spec = importlib.util.spec_from_file_location("qwen_cluster_dragonfly_render", renderer_path)
    if spec is None or spec.loader is None:
        raise NodeInstallError("could not load the pinned Dragonfly config renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node_unit(config: NodeConfiguration, python_executable: Path) -> str:
    state = Path(config.state_dir)
    py = _systemd_arg(python_executable)
    cfg = _systemd_arg(state / "node.json")
    dependencies = []
    if config.write_training_ssh_state:
        dependencies.append("qwen-cluster-sshd.service")
    if config.role == "gateway":
        dependencies.append("qwen-cluster-gateway.service")
    wanted = " ".join(dependencies)
    after = " ".join(["network-online.target", *dependencies])
    return f"""[Unit]
Description=Qwen LAN cluster node heartbeat and model preparation agent
Wants=network-online.target {wanted}
After={after}
ConditionPathExists={cfg}

[Service]
Type=simple
User=root
UMask=0077
ExecStart={py} -m qwen_rl.cluster node run --config {cfg}
Restart=always
RestartSec=5s
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths={_systemd_arg(state)} /run

[Install]
WantedBy=multi-user.target
"""


def _replace_unit_directive(unit_text: str, name: str, value: str | Path) -> str:
    lines = unit_text.splitlines()
    prefix = name + "="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = prefix + _systemd_arg(value)
            return "\n".join(lines) + "\n"
    raise NodeInstallError(f"systemd unit is missing {name}")


def _gateway_service_unit(config: NodeConfiguration, python_executable: Path, project_root: Path) -> str:
    if not config.gateway_host_config_path:
        raise NodeInstallError("Gateway host service requires a local secret-backed config")
    parsed = urlsplit(config.gateway_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    state = Path(config.state_dir) / "gateway" / "state"
    template_path = Path(project_root) / "deploy" / "systemd" / "qwen-cluster-gateway.service"
    if not template_path.is_file() or template_path.is_symlink():
        raise NodeInstallError("host Gateway systemd unit template is missing")
    unit = _replace_state_root(template_path.read_text(encoding="utf-8"), Path(config.state_dir))
    unit = _replace_execstart(unit, [
        python_executable, "-m", "qwen_rl.cluster", "gateway", "--config",
        config.gateway_host_config_path, "--host", "0.0.0.0", "--port", str(port),
    ])
    unit = _replace_condition_path(unit, Path(config.gateway_host_config_path))
    unit = _replace_unit_directive(unit, "User", str(config.worker_uid))
    unit = _replace_unit_directive(unit, "Group", str(config.worker_gid))
    return _replace_unit_directive(unit, "ReadWritePaths", state)


def _gateway_tunnel_unit(config: NodeConfiguration, python_executable: Path) -> str:
    state = Path(config.state_dir)
    ssh_argv = reverse_ssh_argv(
        config,
        identity_file=Path(config.reverse_ssh_key_path),
        known_hosts_file=state / "ssh" / "central_known_hosts",
    )
    command = " ".join(_systemd_arg(argument) for argument in ssh_argv)
    return f"""[Unit]
Description=Qwen cluster Gateway reverse SSH management tunnel
Wants=network-online.target qwen-cluster-sshd.service qwen-cluster-gateway.service
After=network-online.target qwen-cluster-sshd.service qwen-cluster-gateway.service
ConditionPathExists={_systemd_arg(state / 'node.json')}

[Service]
Type=simple
User=root
UMask=0077
ExecStart={command}
Restart=always
RestartSec=5s
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths={_systemd_arg(state)} /run

[Install]
WantedBy=multi-user.target
"""


def _oneshot_compose_unit(
    *, name: str, description: str, project_root: Path, compose_file: Path,
    env_file: Path, services: list[str], before: list[str] = (), after: list[str] = (),
    requires: list[str] = (), pre_start: list[list[str]] = (), docker: str = "/usr/bin/docker",
) -> str:
    start = " ".join([
        _systemd_arg(docker), "compose", "--file", _systemd_arg(compose_file),
        "--env-file", _systemd_arg(env_file), "up", "--detach", "--no-build", "--pull", "never",
        *services,
    ])
    dep_lines = []
    if before:
        dep_lines.append("Before=" + " ".join(before))
    if after:
        dep_lines.append("After=" + " ".join(after))
    if requires:
        dep_lines.append("Requires=" + " ".join(requires))
    pre_lines = ["ExecStartPre=" + " ".join(_systemd_arg(arg) for arg in argv) for argv in pre_start]
    return "\n".join([
        "[Unit]", f"Description={description}", *dep_lines,
        "", "[Service]", "Type=oneshot", "User=root", "UMask=0077",
        f"WorkingDirectory={_systemd_arg(project_root)}", *pre_lines, f"ExecStart={start}",
        "TimeoutStartSec=7200",
        "Restart=on-failure", "RestartSec=30s", "RemainAfterExit=yes",
        "NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes",
        "PrivateTmp=yes", f"ReadWritePaths={_systemd_arg(Path(env_file).parent)} /var/lib/docker /run",
        "", "[Install]", "WantedBy=multi-user.target", "",
    ])


def _replace_state_root(unit_text: str, state_dir: Path) -> str:
    return unit_text.replace("/var/lib/qwen-cluster", str(state_dir))


def _replace_execstart(unit_text: str, argv: list[str]) -> str:
    command = " ".join(_systemd_arg(argument) for argument in argv)
    lines = unit_text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("ExecStart="):
            lines[index] = "ExecStart=" + command
            return "\n".join(lines) + "\n"
    raise NodeInstallError("systemd unit is missing ExecStart")


def _replace_condition_path(unit_text: str, path: Path) -> str:
    lines = unit_text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("ConditionPathExists="):
            lines[index] = "ConditionPathExists=" + _systemd_arg(path)
            return "\n".join(lines) + "\n"
    raise NodeInstallError("systemd unit is missing ConditionPathExists")


def _write_scheduler_env(config: NodeConfiguration, dragonfly_dir: Path) -> Path:
    env_path = Path(config.state_dir) / "gateway" / "dragonfly-compose.env"
    values = {
        "DRAGONFLY_SITE_IP": config.lan_host,
        "DRAGONFLY_CONFIG_DIR": dragonfly_dir / "site-config",
        "DRAGONFLY_STATE_DIR": dragonfly_dir / "scheduler" / "state",
        "DRAGONFLY_LOG_DIR": dragonfly_dir / "scheduler" / "logs",
    }
    for path in (values["DRAGONFLY_CONFIG_DIR"], values["DRAGONFLY_STATE_DIR"], values["DRAGONFLY_LOG_DIR"]):
        Path(path).mkdir(parents=True, exist_ok=True, mode=0o700)
    from .identity import atomic_write
    body = "".join(f'{key}="{str(value).replace(chr(34), chr(92) + chr(34))}"\n' for key, value in values.items())
    atomic_write(env_path, body.encode("utf-8"))
    return env_path


async def _wait_for_verified_model(
    config: NodeConfiguration,
    *,
    worker_token: str,
    model_installer: Callable[..., Any] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Retry transient seed/Dragonfly failures before allowing Worker Compose."""
    status_path = Path(config.state_dir) / "model-install-status.json"
    receipt_path = Path(config.state_dir) / "model-install-receipt.json"
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    backoff = 2.0
    expected = {
        "path": config.model.cache_path,
        "artifact_id": config.model.artifact_id,
        "manifest_sha256": config.model.manifest_sha256,
        "base_fingerprint": config.model.base_fingerprint,
        "artifact_fingerprint": config.model.artifact_fingerprint,
    }
    last_error = "not_started"
    while True:
        attempt += 1
        atomic_write(status_path, (json.dumps({
            "state": "installing", "attempt": attempt,
            "manifest_sha256": config.model.manifest_sha256,
            "updated_at": time.time(), "last_error": last_error,
        }, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        try:
            if model_installer is None:
                receipt = await install_node_model_artifact(config, worker_token=worker_token)
            else:
                receipt = model_installer(config, worker_token)
                if asyncio.iscoroutine(receipt):
                    receipt = await receipt
            if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items()):
                raise NodeInstallError("model install receipt does not match the registered manifest")
            receipt_doc = {
                "state": "verified", **expected,
                "files": receipt.get("files") if isinstance(receipt.get("files"), int)
                and not isinstance(receipt.get("files"), bool) else None,
                "verified_at": time.time(),
            }
            atomic_write(receipt_path, (json.dumps(receipt_doc, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            atomic_write(status_path, (json.dumps({
                "state": "verified", "attempt": attempt,
                "manifest_sha256": config.model.manifest_sha256,
                "artifact_id": config.model.artifact_id,
                "base_fingerprint": config.model.base_fingerprint,
                "artifact_fingerprint": config.model.artifact_fingerprint,
                "updated_at": time.time(),
            }, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            return receipt
        except asyncio.CancelledError:
            atomic_write(status_path, (json.dumps({
                "state": "cancelled", "attempt": attempt,
                "manifest_sha256": config.model.manifest_sha256,
                "updated_at": time.time(),
            }, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            raise
        except Exception as exc:
            last_error = type(exc).__name__
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                atomic_write(status_path, (json.dumps({
                    "state": "failed", "attempt": attempt,
                    "manifest_sha256": config.model.manifest_sha256,
                    "updated_at": time.time(), "last_error": last_error,
                }, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
                raise NodeInstallError(
                    f"model artifact is still unavailable after {attempt} attempts; Worker Compose was not started"
                ) from exc
            await asyncio.sleep(min(backoff, remaining))
            backoff = min(backoff * 2, 30.0)


async def _ensure_node_image(
    config: NodeConfiguration,
    *,
    worker_token: str,
    docker: str,
    runner: Callable[..., subprocess.CompletedProcess],
    image_installer: Callable[..., Any] | None,
) -> dict[str, Any]:
    if image_installer is None:
        receipt = await install_node_worker_image(
            config, worker_token=worker_token, docker_binary=docker, command_runner=runner,
        )
    else:
        receipt = image_installer(config, worker_token, docker, runner)
        if asyncio.iscoroutine(receipt):
            receipt = await receipt
    if config.worker_image_mode == "registry":
        valid = (isinstance(receipt, dict) and receipt.get("state") == "pulled"
                 and receipt.get("worker_image") == config.worker_image)
    else:
        descriptor = config.image_archive
        valid = (descriptor is not None and isinstance(receipt, dict)
                 and receipt.get("state") == "loaded_verified"
                 and receipt.get("image_id") == descriptor.image_id
                 and receipt.get("archive_sha256") == descriptor.archive_sha256
                 and receipt.get("size_bytes") == descriptor.size_bytes)
        if valid and config.worker_release_selection is not None:
            selection = config.worker_release_selection
            valid = receipt.get("worker_release_selection") == {
                "release_identity": selection.release_identity.model_dump(mode="json"),
                "compute_capability": selection.compute_capability,
                "build_arch": selection.build_arch,
                "cuda_version": selection.cuda_version,
            }
    if not valid:
        raise NodeInstallError("local Worker image receipt does not match the selected image source")
    return receipt


def _wait_scheduler_ready(
    *, docker: str, compose_file: Path, env_file: Path,
    runner: Callable[..., subprocess.CompletedProcess], timeout_seconds: float = 180.0,
    poll_seconds: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        listing = runner([
            docker, "--host", "unix:///var/run/docker.sock", "compose", "--file", str(compose_file),
            "--env-file", str(env_file), "ps", "--quiet", "scheduler",
        ], check=False, capture_output=True, text=True, timeout=20)
        container_id = (listing.stdout or "").strip().splitlines()
        if listing.returncode == 0 and container_id:
            health = runner([
                docker, "--host", "unix:///var/run/docker.sock", "inspect", "--format",
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}", container_id[0],
            ], check=False, capture_output=True, text=True, timeout=20)
            if health.returncode == 0 and (health.stdout or "").strip() == "healthy":
                return
        time.sleep(poll_seconds)
    raise NodeInstallError("Dragonfly Scheduler did not become healthy before the install timeout")


async def install_node_services(
    config: NodeConfiguration,
    *,
    project_root: Path,
    python_executable: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    systemd_dir: Path = Path("/etc/systemd/system"),
    binary_dir: Path = Path("/usr/local/bin"),
    source_archive: Path | None = None,
    install_client: bool = True,
    model_installer: Callable[..., Any] | None = None,
    preflight_report: HostPrerequisites | Mapping[str, Any] | None = None,
    model_retry_timeout_seconds: float = 7200.0,
    image_installer: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Install and start only this node's private management and role services.

    The default systemd destination is the host service directory. Tests must
    pass a temporary ``systemd_dir`` and a fake ``runner``; this function is
    never called by the repository's unit tests with the production defaults.
    """
    project_root = Path(project_root).expanduser().resolve(strict=True)
    python_executable = Path(python_executable).expanduser().resolve(strict=True)
    systemd_dir = Path(systemd_dir).expanduser().absolute()
    state_dir = Path(config.state_dir).expanduser().resolve(strict=True)
    if config.owner_uid is not None:
        raise NodeInstallError("account-owned Fleet roles require an account-scoped service installer")
    node_config_path = state_dir / "node.json"
    persisted, credentials = load_node_config(node_config_path)
    if persisted.node_id != config.node_id or persisted.site_id != config.site_id or persisted.role != config.role:
        raise NodeInstallError("the supplied node config does not match durable local identity")
    if config.role == "worker" and not config.needs_driver and not config.gpu_uuid:
        raise NodeInstallError("Worker activation requires an exact registered GPU UUID")
    if model_retry_timeout_seconds <= 0:
        raise NodeInstallError("model installation retry timeout must be positive")
    if preflight_report is None:
        observed_preflight = preflight_host(
            model_path=Path(config.model.cache_path) if config.role == "worker" else state_dir / "artifact-cache",
            state_dir=state_dir,
            runner=runner,
            require_gpu=config.role == "worker",
            require_runtime=True,
        )
    else:
        observed_preflight = preflight_report
    preflight = (
        observed_preflight if isinstance(observed_preflight, HostPrerequisites)
        else HostPrerequisites.model_validate(dict(observed_preflight))
    )
    needs_driver = config.role == "worker" and (
        config.needs_driver or preflight.needs_driver
    )
    needs_host_packages = preflight.needs_host_packages
    updated_config = config.model_copy(update={
        "needs_driver": needs_driver,
        "needs_host_packages": needs_host_packages,
    })
    if updated_config != config:
        config = updated_config
        atomic_write(node_config_path, (json.dumps(
            config.model_dump(mode="json"), sort_keys=True, indent=2, ensure_ascii=False,
        ) + "\n").encode("utf-8"))
        if config.role == "gateway":
            allow_group_traverse(state_dir, gid=config.worker_gid)

    units: list[str] = []
    if config.write_training_ssh_state:
        sshd_template = (project_root / "deploy" / "systemd" / "qwen-cluster-sshd.service").read_text(encoding="utf-8")
        sshd_template = _replace_state_root(sshd_template, state_dir)
        _write_unit(systemd_dir, "qwen-cluster-sshd.service", sshd_template)
        units.append("qwen-cluster-sshd.service")
    _write_unit(systemd_dir, "qwen-cluster-node.service", _node_unit(config, python_executable))
    units.append("qwen-cluster-node.service")

    if needs_driver or needs_host_packages:
        _run([shutil.which("systemctl") or "/usr/bin/systemctl", "daemon-reload"], runner, timeout=30)
        recovery_units = (["qwen-cluster-sshd.service"] if config.write_training_ssh_state else [])
        recovery_units.append("qwen-cluster-node.service")
        for unit in recovery_units:
            _run([shutil.which("systemctl") or "/usr/bin/systemctl", "enable", "--now", unit], runner, timeout=120)
        status = "needs_driver" if needs_driver else "needs_host_packages"
        return {"status": status, "node_id": config.node_id, "units": units,
                "worker_compose_started": False, "needs_driver": needs_driver,
                "needs_host_packages": needs_host_packages,
                "preflight": preflight.model_dump(mode="json"),
                "needs_reboot": False}

    if install_client:
        install_dragonfly_binaries(bin_dir=binary_dir, source_archive=source_archive, runner=runner)

    dragonfly_dir = state_dir / "dragonfly"
    if config.role == "gateway":
        renderer = _load_renderer(project_root)
        renderer.render(config.lan_host, dragonfly_dir / "site-config")
        seed_peer = dragonfly_dir / "site-config" / "seed-peer.yaml"
        dynamic = dragonfly_dir / "site-config" / "dynconfig.yaml"
        from .dragonfly import validate_dfdaemon_config
        validate_dfdaemon_config(seed_peer, allow_lan_source_fallback=True)
        config = config.model_copy(update={"dragonfly_seed_peer_path": str(seed_peer)})
        atomic_write(node_config_path, (json.dumps(config.model_dump(mode="json"), sort_keys=True,
                                                    indent=2, ensure_ascii=False) + "\n").encode())
        allow_group_traverse(state_dir, gid=config.worker_gid)
        seed_unit = _replace_state_root(
            (project_root / "deploy" / "systemd" / "qwen-cluster-dragonfly-seed.service").read_text(encoding="utf-8"),
            state_dir,
        )
        seed_unit = _replace_execstart(seed_unit, [
            "/usr/local/bin/dfdaemon", "--config", str(seed_peer), "--dynconfig", str(dynamic),
        ])
        seed_unit = _replace_condition_path(seed_unit, seed_peer)
        _write_unit(systemd_dir, "qwen-cluster-dragonfly-seed.service", seed_unit)
        units.append("qwen-cluster-dragonfly-seed.service")
        scheduler_env = _write_scheduler_env(config, dragonfly_dir)
        scheduler_compose = project_root / "deploy" / "dragonfly" / "docker-compose.yml"
        from .dragonfly import DRAGONFLY_SCHEDULER_GHCR_INDEX
        docker = shutil.which("docker") or "/usr/bin/docker"
        _run([
            docker, "--host", "unix:///var/run/docker.sock", "pull",
            f"ghcr.io/dragonflyoss/dragonfly/scheduler:v2.5.2@{DRAGONFLY_SCHEDULER_GHCR_INDEX}",
        ], runner, timeout=7200)
        scheduler_unit = _oneshot_compose_unit(
            name="qwen-cluster-dragonfly-scheduler.service", description="Qwen site Dragonfly Scheduler",
            project_root=project_root, compose_file=scheduler_compose, env_file=scheduler_env,
            services=["scheduler"], docker=docker,
        )
        _write_unit(systemd_dir, "qwen-cluster-dragonfly-scheduler.service", scheduler_unit)
        units.append("qwen-cluster-dragonfly-scheduler.service")
        gateway_unit = _gateway_service_unit(config, python_executable, project_root)
        _write_unit(systemd_dir, "qwen-cluster-gateway.service", gateway_unit)
        units.append("qwen-cluster-gateway.service")
        start_order = []
        if config.write_training_ssh_state:
            start_order.append("qwen-cluster-sshd.service")
        start_order.extend((
            "qwen-cluster-dragonfly-scheduler.service",
            "qwen-cluster-dragonfly-seed.service",
            "qwen-cluster-gateway.service",
            "qwen-cluster-node.service",
        ))
        if config.write_training_ssh_state and config.reverse_ssh is not None:
            tunnel_unit = _gateway_tunnel_unit(config, python_executable)
            _write_unit(systemd_dir, "qwen-cluster-gateway-tunnel.service", tunnel_unit)
            units.append("qwen-cluster-gateway-tunnel.service")
            start_order.append("qwen-cluster-gateway-tunnel.service")
    else:
        worker_peer = Path(config.dragonfly_worker_peer_path or "")
        if not worker_peer.is_file():
            raise NodeInstallError("Worker Dragonfly peer configuration is missing")
        from .dragonfly import validate_dfdaemon_config
        validate_dfdaemon_config(worker_peer, allow_lan_source_fallback=False)
        worker_unit = _replace_state_root(
            (project_root / "deploy" / "systemd" / "qwen-cluster-dragonfly.service").read_text(encoding="utf-8"),
            state_dir,
        )
        dynconfig = worker_peer.parent / "dynconfig.yaml"
        worker_unit = _replace_execstart(worker_unit, [
            "/usr/local/bin/dfdaemon", "--config", str(worker_peer), "--dynconfig", str(dynconfig),
        ])
        _write_unit(systemd_dir, "qwen-cluster-dragonfly.service", worker_unit)
        units.append("qwen-cluster-dragonfly.service")
        worker_env = Path(config.compose_env_path or "")
        if not worker_env.is_file() or not config.worker_compose_file:
            raise NodeInstallError("Worker Compose configuration is incomplete")
        pre_start = [[str(python_executable), "-m", "qwen_rl.cluster", "node", "verify-model",
                      "--config", str(node_config_path)],
                     [str(python_executable), "-m", "qwen_rl.cluster", "node", "install-image",
                      "--config", str(node_config_path)]]
        docker = shutil.which("docker") or "/usr/bin/docker"
        compose_unit = _oneshot_compose_unit(
            name="qwen-cluster-worker-compose.service", description="Qwen vLLM Worker Compose service",
            project_root=project_root, compose_file=Path(config.worker_compose_file), env_file=worker_env,
            services=["worker"], after=["qwen-cluster-node.service", "qwen-cluster-dragonfly.service"],
            requires=["docker.service", "qwen-cluster-dragonfly.service"], pre_start=pre_start,
            docker=docker,
        )
        _write_unit(systemd_dir, "qwen-cluster-worker-compose.service", compose_unit)
        units.append("qwen-cluster-worker-compose.service")
        start_order = []
        if config.write_training_ssh_state:
            start_order.append("qwen-cluster-sshd.service")
        start_order.extend(("qwen-cluster-dragonfly.service", "qwen-cluster-node.service"))

    systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
    _run([systemctl, "daemon-reload"], runner, timeout=30)
    for unit in start_order:
        _run([systemctl, "enable", "--now", unit], runner, timeout=180)
        if config.role == "gateway" and unit == "qwen-cluster-dragonfly-scheduler.service":
            _wait_scheduler_ready(
                docker=shutil.which("docker") or "/usr/bin/docker",
                compose_file=project_root / "deploy" / "dragonfly" / "docker-compose.yml",
                env_file=state_dir / "gateway" / "dragonfly-compose.env",
                runner=runner,
            )

    if config.role == "worker":
        # Keep boot recovery armed even if this invocation exhausts its model
        # wait. The unit's verified-model pre-start gate prevents early launch.
        _run([systemctl, "enable", "qwen-cluster-worker-compose.service"], runner, timeout=120)
        await _wait_for_verified_model(
            config, worker_token=credentials["worker_token"], model_installer=model_installer,
            timeout_seconds=model_retry_timeout_seconds,
        )
        await _ensure_node_image(
            config, worker_token=credentials["worker_token"],
            docker=shutil.which("docker") or "/usr/bin/docker", runner=runner,
            image_installer=image_installer,
        )
        _run([systemctl, "start", "qwen-cluster-worker-compose.service"], runner, timeout=7200)

    return {"status": "activated", "node_id": config.node_id, "units": units,
            "worker_compose_started": config.role == "worker",
            "gateway_host_started": config.role == "gateway", "needs_reboot": False}
