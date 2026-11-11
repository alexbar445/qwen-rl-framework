"""Shared Fleet role contracts and ownership-safe local helpers."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx

from ..identity import atomic_write
from .contracts import ArtifactReference
from .host import HostInventory
from .paths import AccountPathError, ensure_account_path


class FleetRoleError(RuntimeError):
    """Expected, secret-safe failure while applying a centrally assigned role."""


@dataclass(frozen=True)
class RoleResult:
    status: Literal["ready", "loading", "needs_driver", "needs_host_packages", "needs_reboot", "draining", "failed"]
    receipt: dict[str, Any]
    error_code: str | None = None
    error_message: str | None = None
    owned: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "receipt": dict(self.receipt),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "owned": self.owned,
        }


@dataclass(frozen=True)
class StopResult:
    status: Literal["stopped", "blocked_busy", "failed"]
    receipt: dict[str, Any]
    error_code: str | None = None
    error_message: str | None = None


class RoleAdapter(Protocol):
    role: str

    def prepare_install(self, deployment: Any, *, generation: int,
                        owned_receipt: Mapping[str, Any] | None = None) -> Mapping[str, Any]: ...

    def install(self, deployment: Any, *, generation: int, inventory: HostInventory,
                newly_managed: bool, owned_receipt: Mapping[str, Any] | None = None,
                install_intent: Mapping[str, Any] | None = None) -> RoleResult: ...

    def inspect(self, receipt: Mapping[str, Any]) -> RoleResult: ...

    def stop_owned(self, receipt: Mapping[str, Any]) -> StopResult: ...


def operation_dir(
    state_dir: Path,
    generation: int,
    role: str,
    deployment_id: str,
    *,
    account: Any,
) -> Path:
    """Create or validate one durable, private role-operation directory."""
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise FleetRoleError("assignment generation is invalid")
    if role not in {"worker", "learner"}:
        raise FleetRoleError("role identity is invalid")
    if re.fullmatch(r"[A-Za-z0-9_-]{16,100}", deployment_id) is None:
        raise FleetRoleError("deployment identity is invalid")
    try:
        ensure_account_path(state_dir, account=account, mode=0o700)
        root = ensure_account_path(Path(state_dir) / "operations", account=account, mode=0o700)
    except AccountPathError as exc:
        raise FleetRoleError("role operation state cannot be safely used below HOME") from exc
    os.chmod(root, 0o700)
    target = root / f"generation-{generation}"
    try:
        ensure_account_path(target, account=account, mode=0o700)
    except AccountPathError as exc:
        raise FleetRoleError("role operation state cannot be safely used below HOME") from exc
    os.chmod(target, 0o700)
    marker = target / "assignment.json"
    expected = {"generation": generation, "role": role, "deployment_id": deployment_id}
    if marker.exists():
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FleetRoleError("existing operation identity cannot be read") from exc
        if current != expected:
            raise FleetRoleError("assignment generation conflicts with its durable operation record")
    else:
        atomic_write(marker, (json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n").encode(),
                     mode=0o600)
    return target


def ensure_user_dir(path: Path, *, account: Any, mode: int = 0o700) -> Path:
    """Create a private account-owned directory without changing another owner's files."""
    if os.geteuid() != account.uid:
        raise FleetRoleError("role process is not running as the registered account")
    try:
        target = ensure_account_path(path, account=account, mode=mode)
    except AccountPathError as exc:
        raise FleetRoleError("account role path cannot be safely used below HOME") from exc
    os.chmod(target, mode)
    return target


def verify_existing(path: Path, sha256: str, *, maximum_bytes: int) -> bool:
    """Return whether an existing regular file matches a content pin."""
    try:
        metadata = Path(path).lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise FleetRoleError("pinned artifact file cannot be inspected") from exc
    if (not stat.S_ISREG(metadata.st_mode) or Path(path).is_symlink()
            or metadata.st_size > maximum_bytes):
        raise FleetRoleError("pinned artifact path is not a bounded regular file")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if not hmac.compare_digest(digest.hexdigest(), sha256):
        raise FleetRoleError("existing artifact does not match its SHA-256 pin")
    return True


def download_verified_artifact(
    reference: ArtifactReference,
    destination: Path,
    *,
    ca_bundle: str | Path,
    max_bytes: int = 8 * 1024 * 1024,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Download one HTTPS artifact without redirects and verify its pinned SHA-256."""
    destination = Path(destination)
    if verify_existing(destination, reference.sha256, maximum_bytes=max_bytes):
        return destination
    if destination.exists() or destination.is_symlink():
        raise FleetRoleError("artifact destination already exists")
    temporary = destination.with_name(f".{destination.name}.download-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    total = 0
    try:
        with httpx.Client(verify=str(ca_bundle), transport=transport, timeout=120,
                          follow_redirects=False, trust_env=False) as client, \
                client.stream("GET", reference.url,
                              headers={"Accept": "application/octet-stream"}) as response:
            if (response.status_code != 200 or response.url.scheme != "https"
                    or response.url.host != urlsplit(reference.url).hostname):
                raise FleetRoleError("pinned artifact download returned an unexpected origin or status")
            length = response.headers.get("Content-Length")
            if length is not None and length.isdigit() and int(length) > max_bytes:
                raise FleetRoleError("pinned artifact exceeds its download size limit")
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(descriptor, "wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise FleetRoleError("pinned artifact exceeds its download size limit")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if total == 0 or not hmac.compare_digest(digest.hexdigest(), reference.sha256):
            raise FleetRoleError("pinned artifact SHA-256 did not match")
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        return destination
    except FleetRoleError:
        temporary.unlink(missing_ok=True)
        raise
    except (httpx.HTTPError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise FleetRoleError("pinned artifact download failed") from exc


def run_owned(
    argv: Sequence[str], *, runner: Callable[..., subprocess.CompletedProcess], timeout: float,
) -> subprocess.CompletedProcess:
    """Run one validated command and report only a bounded exit detail."""
    if not argv or any(not isinstance(item, str) or not item or "\x00" in item for item in argv):
        raise FleetRoleError("owned role command is invalid")
    try:
        result = runner(list(argv), check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise FleetRoleError("owned role command could not be started") from exc
    if result.returncode != 0:
        raise FleetRoleError(f"owned role command exited with status {result.returncode}")
    return result


def parse_role_receipt(stdout: str, *, label: str = "role helper") -> dict[str, Any]:
    """Parse a JSON object emitted by a pinned local role helper."""
    try:
        value = json.loads(stdout or "")
    except json.JSONDecodeError as exc:
        raise FleetRoleError(f"{label} returned an invalid receipt") from exc
    if not isinstance(value, dict):
        raise FleetRoleError(f"{label} returned an invalid receipt")
    return value


@dataclass(frozen=True)
class DockerAccess:
    executable: Path
    context: str
    endpoint: str
    daemon_id: str
    rootless: bool
    host_override: bool = False

    def command(self, *arguments: str) -> list[str]:
        target = ["--host", self.endpoint] if self.host_override else ["--context", self.context]
        return [str(self.executable), *target, *arguments]

    def receipt(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "endpoint": self.endpoint,
            "daemon_id": self.daemon_id,
            "rootless": self.rootless,
            "host_override": self.host_override,
        }


def _docker_executable() -> Path:
    path = shutil.which("docker") or "/usr/bin/docker"
    executable = Path(path)
    if not executable.is_absolute() or not executable.is_file():
        raise FleetRoleError("Docker CLI is unavailable")
    return executable


def _docker_command(
    argv: Sequence[str], *, runner: Callable[..., subprocess.CompletedProcess], timeout: float = 20,
) -> subprocess.CompletedProcess:
    try:
        result = runner(list(argv), check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise FleetRoleError("Docker context could not be inspected") from exc
    if result.returncode != 0:
        raise FleetRoleError("Docker context could not be inspected")
    return result


def _local_docker_endpoint(endpoint: Any) -> str:
    if not isinstance(endpoint, str) or len(endpoint) > 4096 or any(ord(char) < 32 for char in endpoint):
        raise FleetRoleError("Docker context endpoint is invalid")
    try:
        parsed = urlsplit(endpoint)
        _ = parsed.port
    except ValueError as exc:
        raise FleetRoleError("Docker context endpoint is invalid") from exc
    if parsed.scheme == "unix":
        if parsed.netloc or not parsed.path.startswith("/") or parsed.query or parsed.fragment:
            raise FleetRoleError("Docker context endpoint is not a local Unix socket")
        return endpoint
    if parsed.scheme == "tcp" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise FleetRoleError("Docker context endpoint is not a local daemon endpoint")
        return endpoint
    raise FleetRoleError("Docker context must point to a local daemon")


def inspect_docker_access(
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    expected: Mapping[str, Any] | None = None,
) -> DockerAccess:
    """Pin the active local Docker context and fail closed on unknown namespaces."""
    executable = _docker_executable()
    env_context = os.environ.get("DOCKER_CONTEXT")
    env_host = os.environ.get("DOCKER_HOST")
    host_override = bool(env_host and not env_context)
    if host_override:
        context = "DOCKER_HOST"
        endpoint = _local_docker_endpoint(env_host)
        info_argv = [str(executable), "--host", endpoint, "info", "--format", "{{json .}}"]
    else:
        context_result = _docker_command([str(executable), "context", "show"], runner=runner)
        context = (context_result.stdout or "").strip()
        if not context or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", context) is None:
            raise FleetRoleError("active Docker context name is invalid")
        inspected = _docker_command(
            [str(executable), "context", "inspect", context, "--format", "{{json .Endpoints.docker.Host}}"],
            runner=runner,
        )
        try:
            endpoint_value = json.loads((inspected.stdout or "").strip())
        except json.JSONDecodeError:
            endpoint_value = (inspected.stdout or "").strip().strip('"')
        endpoint = _local_docker_endpoint(endpoint_value)
        info_argv = [str(executable), "--context", context, "info", "--format", "{{json .}}"]
    info_result = _docker_command(info_argv, runner=runner)
    info = parse_role_receipt(info_result.stdout, label="Docker daemon")
    daemon_id = info.get("ID")
    daemon_name = info.get("Name")
    if not isinstance(daemon_id, str) or not daemon_id.strip():
        if not isinstance(daemon_name, str) or not daemon_name.strip():
            raise FleetRoleError("Docker daemon namespace is not identifiable")
        daemon_id = f"name:{daemon_name.strip()}"
    security_options = info.get("SecurityOptions")
    if not isinstance(security_options, list) or any(not isinstance(item, str) for item in security_options):
        raise FleetRoleError("Docker daemon UID namespace is unknown")
    security_text = " ".join(security_options).lower()
    if "userns" in security_text and "rootless" not in security_text:
        raise FleetRoleError("Docker user namespace remapping is unsupported for this role")
    rootless = "rootless" in security_text
    access = DockerAccess(executable, context, endpoint, daemon_id.strip(), rootless, host_override)
    if expected is not None:
        recorded = access.receipt()
        for key in ("context", "endpoint", "daemon_id", "rootless", "host_override"):
            if key in expected and expected.get(key) != recorded[key]:
                raise FleetRoleError("active Docker context differs from the recorded role receipt")
    return access


def probe_docker_uid_mapping(
    access: DockerAccess,
    *,
    image: str,
    account: Any,
    probe_dir: Path,
    python_executable: str = "python",
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure a container's writable bind-mount UID/GID mapping with a labelled probe."""
    if not image or any(ord(char) < 32 for char in image):
        raise FleetRoleError("Docker mapping probe image is invalid")
    if (not isinstance(python_executable, str) or not python_executable
            or len(python_executable) > 512 or python_executable.startswith("-")
            or any(char.isspace() or ord(char) < 32 for char in python_executable)):
        raise FleetRoleError("Docker mapping probe Python executable is invalid")
    root = ensure_user_dir(probe_dir, account=account, mode=0o700)
    marker = root / "uid-map-probe"
    if marker.exists() or marker.is_symlink():
        raise FleetRoleError("Docker mapping probe path already contains data")
    expected_inside_uid = 0 if access.rootless else account.uid
    expected_inside_gid = 0 if access.rootless else account.gid
    probe_id = uuid.uuid4().hex
    labels = {
        "com.qwen-fleet.managed": "true",
        "com.qwen-fleet.uid-probe": probe_id,
        "com.qwen-fleet.account-uid": str(account.uid),
    }
    code = (
        "import os,pathlib; p=pathlib.Path('/qwen-probe/uid-map-probe'); "
        "p.write_text(f'{os.getuid()}:{os.getgid()}\\n'); print(f'{os.getuid()}:{os.getgid()}')"
    )
    command = access.command(
        "run", "--rm", "--pull", "never", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--user", f"{expected_inside_uid}:{expected_inside_gid}",
        "--mount", f"type=bind,source={root},target=/qwen-probe",
        "--label", f"com.qwen-fleet.managed={labels['com.qwen-fleet.managed']}",
        "--label", f"com.qwen-fleet.uid-probe={probe_id}",
        "--label", f"com.qwen-fleet.account-uid={account.uid}",
        "--entrypoint", python_executable, image, "-c", code,
    )

    def remove_own_probe() -> None:
        try:
            listed = runner(
                access.command("ps", "--all", "--quiet", "--filter",
                               f"label=com.qwen-fleet.uid-probe={probe_id}"),
                check=False, capture_output=True, text=True, timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return
        if listed.returncode != 0:
            return
        for container_id in (listed.stdout or "").splitlines():
            container_id = container_id.strip()
            if re.fullmatch(r"[0-9a-f]{12,64}", container_id) is None:
                continue
            try:
                observed = runner(
                    access.command("inspect", "--format", "{{json .Config.Labels}}", container_id),
                    check=False, capture_output=True, text=True, timeout=20,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if observed.returncode != 0:
                continue
            try:
                labels_value = json.loads((observed.stdout or "").strip())
            except json.JSONDecodeError:
                continue
            if not isinstance(labels_value, dict) or any(labels_value.get(key) != value for key, value in labels.items()):
                continue
            try:
                runner(access.command("rm", "--force", container_id), check=False,
                       capture_output=True, text=True, timeout=20)
            except (OSError, subprocess.SubprocessError):
                continue

    try:
        result = run_owned(command, runner=runner, timeout=90)
        host_info = marker.lstat()
        stdout = (result.stdout or "").strip().splitlines()
        if (not stat.S_ISREG(host_info.st_mode) or marker.is_symlink()
                or host_info.st_uid != account.uid or host_info.st_gid != account.gid
                or marker.read_text(encoding="ascii") != f"{expected_inside_uid}:{expected_inside_gid}\n"
                or not stdout or stdout[-1] != f"{expected_inside_uid}:{expected_inside_gid}"):
            raise FleetRoleError("Docker bind mount does not map the role UID/GID to this account")
        receipt = {
            **access.receipt(),
            "container_uid": expected_inside_uid,
            "container_gid": expected_inside_gid,
            "host_uid": host_info.st_uid,
            "host_gid": host_info.st_gid,
            "uid_mapping_verified": True,
            "probe_image": image,
            "probe_python": python_executable,
        }
        if expected is not None:
            for key in ("container_uid", "container_gid", "host_uid", "host_gid",
                        "probe_image", "probe_python"):
                if key in expected and expected.get(key) != receipt[key]:
                    raise FleetRoleError("Docker UID mapping differs from the recorded role receipt")
        return receipt
    finally:
        remove_own_probe()
        marker.unlink(missing_ok=True)


def default_role_adapters(
    *,
    state_dir: Path,
    ca_bundle: Path,
    python_executable: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    artifact_transport: httpx.BaseTransport | None = None,
    account: Any | None = None,
    paths: Any | None = None,
) -> dict[str, RoleAdapter]:
    """Build role adapters through delayed imports, keeping concrete roles separate."""
    from .host import current_user_account
    from .paths import UserPaths
    from .roles_learner import LearnerRoleAdapter
    from .roles_worker import WorkerRoleAdapter

    account = account or current_user_account()
    paths = paths or UserPaths.from_account(account)
    common = {
        "state_dir": Path(state_dir),
        "ca_bundle": Path(ca_bundle),
        "python_executable": Path(python_executable),
        "runner": runner,
        "artifact_transport": artifact_transport,
        "account": account,
        "paths": paths,
    }
    return {
        "worker": WorkerRoleAdapter(**common),
        "learner": LearnerRoleAdapter(**common),
    }
