"""Host-owned lifecycle for local vLLM Worker sidecar containers."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shlex
import stat
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from ..cluster.images import ImageArchiveDescriptor
from ..cluster.worker_release import (
    ResolvedWorkerImage,
    WorkerRelease,
    WorkerReleaseError,
    WorkerReleaseIdentity,
    expected_worker_image_labels,
    load_worker_release,
    select_worker_image,
)
from .local import DeploymentError

_OWNER_LABEL = "io.qwen-rl.owner"
_DEPLOYMENT_LABEL = "io.qwen-rl.deployment"
_WORKER_LABEL = "io.qwen-rl.worker"
_FINGERPRINT_LABEL = "io.qwen-rl.fingerprint"
_CONFIG_SHA_LABEL = "io.qwen-rl.config-sha256"
_IMAGE_ID_LABEL = "io.qwen-rl.image-id"
_OWNER_VALUE = "local-worker-fleet/v1"
_MANIFEST_SCHEMA = "qwen-rl-local-worker-fleet/v1"
_WORKER_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_BOOT_ID = re.compile(r"^[0-9a-f-]{36}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$")
_HEALTH_PROBE = r'''
import json, os, sys
from urllib.request import Request, urlopen
request = Request(sys.argv[1], headers={
    "Authorization": "Bearer " + os.environ["ARC_ENV_TOKEN"],
    "Accept": "application/json",
})
with urlopen(request, timeout=3.0) as response:
    payload = json.loads(response.read(1_000_001))
if not isinstance(payload, dict):
    raise SystemExit(2)
keys = ("ready", "role", "worker_id", "boot_id", "telemetry_epoch",
        "residency_enabled", "ninfer_artifact_sha256",
        "base_fingerprint", "artifact_fingerprint", "gpu_uuid")
result = {key: payload.get(key) for key in keys}
deployment = payload.get("deployment")
if isinstance(deployment, dict):
    result["deployment"] = {key: deployment.get(key) for key in ("scope", "site_id", "node_id")}
print(json.dumps(result, separators=(",", ":")))
'''


class LocalWorkerFleetError(DeploymentError):
    """A safe local Worker sidecar lifecycle failure."""


@dataclass(frozen=True)
class LocalWorkerSpec:
    worker_id: str
    host_port: int
    gpu_index: int
    gpu_uuid: str
    compute_capability: str | None
    worker_image: str
    worker_image_id: str
    image_archive: ImageArchiveDescriptor | None
    worker_release_identity: WorkerReleaseIdentity | None
    site_id: str
    node_id: str
    auth_token_env: str
    auth_token_path: Path
    config_path: Path
    state_dir: Path
    container_name: str
    config_sha256: str
    ninfer_url: str | None
    auth_token: str = field(repr=False, compare=False)

    def manifest_record(self, *, container_id: str | None = None) -> dict[str, Any]:
        record = {
            "worker_id": self.worker_id,
            "container_name": self.container_name,
            "container_id": container_id,
            "port": self.host_port,
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
            "gpu_compute_capability": self.compute_capability,
            "worker_image": self.worker_image,
            "worker_image_id": self.worker_image_id,
            "auth_token_env": self.auth_token_env,
            "config_sha256": self.config_sha256,
            "config_path": str(self.config_path),
            "state_dir": str(self.state_dir),
        }
        if self.image_archive is not None:
            record["image_archive"] = self.image_archive.model_dump(mode="json")
        if self.worker_release_identity is not None:
            record["worker_release_identity"] = self.worker_release_identity.model_dump(mode="json")
        if self.ninfer_url is not None:
            record["ninfer_url"] = self.ninfer_url
        return record


@dataclass(frozen=True)
class LocalWorkerNInferInput:
    """One NInfer source artifact/report path and how the Worker sees it."""

    field_name: str
    host_path: Path
    container_path: str
    model_mount_covers_file: bool
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class LocalWorkerNInferOrigin:
    worker_id: str
    ordinal: int
    worker_http_port: int
    engine_port: int
    url: str


@dataclass(frozen=True)
class LocalWorkerNInferOriginPlan:
    configured_base_url: str
    learner_http_port: int
    worker_http_ports: tuple[tuple[str, int], ...]
    assignments: tuple[LocalWorkerNInferOrigin, ...]


@dataclass(frozen=True)
class LocalWorkerPlan:
    deployment_id: str
    fingerprint: str
    model_container_path: str
    model_host_path: Path
    boot_id: str
    runtime_config: Any
    runtime_config_host_path: Path
    runtime_config_container_path: str
    runtime_environment_host_path: Path
    runtime_environment_container_path: str
    manifest_path: Path
    ninfer_inputs: tuple[LocalWorkerNInferInput, ...]
    ninfer_origin_plan: LocalWorkerNInferOriginPlan | None
    specs: tuple[LocalWorkerSpec, ...]


def _worker_requires_residency(runtime: Any, worker_id: str) -> bool:
    """Keep phase-sharing explicit for Trainer and central-gradient profiles."""
    if worker_id in runtime.learner.shared_worker_ids:
        return True
    return (runtime.learner.update_backend == "central_gradient"
            and worker_id in runtime.learner.gradient_local_executor_worker_ids)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_private(path: Path, content: bytes) -> None:
    path = Path(path)
    if path.is_symlink():
        raise LocalWorkerFleetError(f"private Worker file is a symbolic link: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _check_private_directory(path.parent)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(path, 0o600)


def _check_private_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LocalWorkerFleetError("local Worker state directory cannot be inspected") from exc
    if (not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077):
        raise LocalWorkerFleetError("local Worker state directory must be private and owned by the caller")


def _check_private_file(path: Path, label: str) -> None:
    try:
        metadata = Path(path).lstat()
    except OSError as exc:
        raise LocalWorkerFleetError(f"{label} cannot be inspected") from exc
    if (not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1):
        raise LocalWorkerFleetError(f"{label} must be a private owned regular file")


def _safe_worker_slug(value: str) -> str:
    if not isinstance(value, str) or not _WORKER_ID.fullmatch(value):
        raise LocalWorkerFleetError("runtime Worker id is invalid for local container ownership")
    return value


def _yaml_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _yaml_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_yaml_value(item) for item in value]
    return value


def _auth_environment_name(worker_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]", "_", _safe_worker_slug(worker_id)).upper()
    return f"QWEN_RL_WORKER_{slug}_TOKEN"


def _mount_source(mounts: Sequence[Mapping[str, Any]], destination: str) -> Path:
    target = PurePosixPath(destination)
    matches: list[tuple[int, Path, Path]] = []
    for mount in mounts:
        if mount.get("Type") != "bind":
            continue
        source_text, destination_text = mount.get("Source"), mount.get("Destination")
        if not isinstance(source_text, str) or not isinstance(destination_text, str):
            continue
        source, target_root = Path(source_text), PurePosixPath(destination_text)
        try:
            relative = target.relative_to(target_root)
        except ValueError:
            continue
        matches.append((len(target_root.parts), source, Path(*relative.parts)))
    if not matches:
        raise LocalWorkerFleetError(f"manager container has no host bind mount for {destination}")
    _depth, source, relative = max(matches, key=lambda item: item[0])
    candidate = source / relative
    path_component = candidate
    has_symlink = False
    while path_component != source.parent:
        if path_component.is_symlink():
            has_symlink = True
            break
        if path_component == source:
            break
        path_component = path_component.parent
    if has_symlink:
        raise LocalWorkerFleetError(f"manager bind path is a symbolic link: {destination}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LocalWorkerFleetError(f"manager bind path is unavailable: {destination}") from exc
    return resolved


def _container_path_for_host(mounts: Sequence[Mapping[str, Any]], source_path: Path) -> str:
    source_path = Path(source_path).resolve(strict=False)
    matches: list[tuple[int, PurePosixPath, Path]] = []
    for mount in mounts:
        if mount.get("Type") != "bind":
            continue
        source_text, destination_text = mount.get("Source"), mount.get("Destination")
        if not isinstance(source_text, str) or not isinstance(destination_text, str):
            continue
        host_root = Path(source_text).resolve(strict=False)
        try:
            relative = source_path.relative_to(host_root)
        except ValueError:
            continue
        matches.append((len(host_root.parts), PurePosixPath(destination_text), relative))
    if not matches:
        raise LocalWorkerFleetError("local Worker service files are outside the manager's bind-mounted project")
    _depth, container_root, relative = max(matches, key=lambda item: item[0])
    return str(container_root.joinpath(*relative.parts))


def _file_content_identity(path: Path, *, label: str) -> tuple[str, int]:
    """Stream one regular bind input once and reject changes during the read."""
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise LocalWorkerFleetError(f"{label} cannot be inspected") from exc
    if not stat.S_ISREG(before_path.st_mode) or stat.S_ISLNK(before_path.st_mode):
        raise LocalWorkerFleetError(f"{label} must be a regular non-symlink file")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            signature = (before.st_dev, before.st_ino, before.st_size,
                         before.st_mtime_ns, before.st_ctime_ns)
            path_signature = (before_path.st_dev, before_path.st_ino, before_path.st_size,
                              before_path.st_mtime_ns, before_path.st_ctime_ns)
            if not stat.S_ISREG(before.st_mode) or signature != path_signature:
                raise LocalWorkerFleetError(f"{label} changed while opening")
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
            after_signature = (after.st_dev, after.st_ino, after.st_size,
                               after.st_mtime_ns, after.st_ctime_ns)
            if signature != after_signature or size != after.st_size:
                raise LocalWorkerFleetError(f"{label} changed while hashing")
    except LocalWorkerFleetError:
        raise
    except OSError as exc:
        raise LocalWorkerFleetError(f"{label} cannot be read safely") from exc
    return digest.hexdigest(), size


def _resolve_ninfer_inputs(
    inference: Any,
    *,
    mounts: Sequence[Mapping[str, Any]],
    model_container_path: str,
    model_host_path: Path,
    model_is_file: bool = False,
) -> tuple[LocalWorkerNInferInput, ...]:
    if inference.backend != "ninfer":
        return ()

    model_root = PurePosixPath(model_container_path)
    inputs: list[LocalWorkerNInferInput] = []
    seen_targets: set[str] = set()
    for field_name in ("ninfer_artifact_path", "ninfer_conversion_report"):
        configured_path = getattr(inference, field_name)
        if not isinstance(configured_path, str) or not configured_path:
            raise LocalWorkerFleetError(f"NInfer Worker requires {field_name}")
        target = PurePosixPath(configured_path)
        if (not target.is_absolute() or str(target) != configured_path
                or ".." in target.parts):
            raise LocalWorkerFleetError(f"NInfer Worker {field_name} must be a canonical absolute path")
        if "," in configured_path or ":" in configured_path:
            raise LocalWorkerFleetError(f"NInfer Worker {field_name} contains a Docker mount separator")
        if configured_path in seen_targets:
            raise LocalWorkerFleetError("NInfer artifact and conversion report must use distinct paths")
        seen_targets.add(configured_path)

        try:
            host_path = _mount_source(mounts, configured_path)
        except LocalWorkerFleetError as exc:
            raise LocalWorkerFleetError(
                f"NInfer Worker {field_name} must resolve through an inspected manager bind mount"
            ) from exc
        if not host_path.is_file():
            raise LocalWorkerFleetError(f"NInfer Worker {field_name} must resolve to a regular file")
        if inference.quantization == "ternary" and field_name == "ninfer_artifact_path":
            sha256 = inference.ninfer_artifact_sha256
            if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise LocalWorkerFleetError("T2 Worker requires deployment base SHA-256 declaration")
            size_bytes = host_path.stat().st_size
        else:
            sha256, size_bytes = _file_content_identity(host_path, label=f"NInfer Worker {field_name}")

        if model_is_file and target == model_root and field_name == "ninfer_artifact_path":
            if host_path != model_host_path:
                raise LocalWorkerFleetError(
                    "T2 Worker model_path and ninfer_artifact_path resolve to different files"
                )
            model_mount_covers_file = True
        else:
            try:
                relative = target.relative_to(model_root)
            except ValueError:
                model_mount_covers_file = False
            else:
                if not relative.parts:
                    raise LocalWorkerFleetError(
                        f"NInfer Worker {field_name} cannot be the Worker model directory"
                    )
                if model_is_file:
                    raise LocalWorkerFleetError(
                        f"NInfer Worker {field_name} cannot be nested below the T2 model file"
                    )
                try:
                    expected_host_path = model_host_path.joinpath(*relative.parts).resolve(strict=True)
                except OSError as exc:
                    raise LocalWorkerFleetError(
                        f"NInfer Worker {field_name} is unavailable below the Worker model bind source"
                    ) from exc
                if expected_host_path != host_path:
                    raise LocalWorkerFleetError(
                        f"NInfer Worker {field_name} under model_path does not resolve to the Worker model bind source"
                    )
                model_mount_covers_file = True

        inputs.append(LocalWorkerNInferInput(
            field_name=field_name,
            host_path=host_path,
            container_path=configured_path,
            model_mount_covers_file=model_mount_covers_file,
            sha256=sha256,
            size_bytes=size_bytes,
        ))
    return tuple(inputs)


def _plan_ninfer_origins(
    inference: Any,
    *,
    learner_http_port: int,
    workers: Sequence[Any],
) -> LocalWorkerNInferOriginPlan | None:
    if inference.backend != "ninfer":
        return None

    configured_base_url = inference.ninfer_url
    if not isinstance(configured_base_url, str) or not configured_base_url:
        raise LocalWorkerFleetError("NInfer Worker requires an explicit ninfer_url base origin")
    try:
        parsed = urlsplit(configured_base_url)
        base_port = parsed.port
    except ValueError as exc:
        raise LocalWorkerFleetError("NInfer Worker ninfer_url has an invalid port") from exc
    host = parsed.hostname
    loopback = host == "localhost"
    if host and not loopback:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
    if (parsed.scheme != "http" or not host or not loopback or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ("", "/")
            or base_port is None or not 1 <= base_port <= 65535):
        raise LocalWorkerFleetError(
            "NInfer Worker ninfer_url must be an explicit loopback HTTP origin with a port and no API path"
        )

    ordered_workers = sorted(workers, key=lambda worker: worker.id)
    worker_ids = [worker.id for worker in ordered_workers]
    if len(set(worker_ids)) != len(worker_ids):
        raise LocalWorkerFleetError("NInfer Worker IDs must be unique when planning engine ports")
    if (isinstance(learner_http_port, bool) or not isinstance(learner_http_port, int)
            or not 1 <= learner_http_port <= 65535):
        raise LocalWorkerFleetError("Learner HTTP port is invalid for NInfer engine port planning")

    worker_http_ports: list[tuple[str, int]] = []
    reserved_ports = {learner_http_port}
    for worker in ordered_workers:
        port = worker.port
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise LocalWorkerFleetError(f"Worker HTTP port is invalid for NInfer Worker {worker.id}")
        if port in reserved_ports:
            raise LocalWorkerFleetError(
                f"Worker HTTP port {port} collides with another Worker or the Learner HTTP port"
            )
        reserved_ports.add(port)
        worker_http_ports.append((worker.id, port))

    assignments: list[LocalWorkerNInferOrigin] = []
    used_engine_ports: set[int] = set()
    host_literal = f"[{host}]" if ":" in host else host
    for ordinal, worker in enumerate(ordered_workers):
        engine_port = base_port + ordinal
        if engine_port > 65535:
            raise LocalWorkerFleetError("NInfer engine port allocation exceeds port 65535")
        if engine_port in reserved_ports or engine_port in used_engine_ports:
            raise LocalWorkerFleetError(
                f"NInfer engine port {engine_port} collides with a Learner or Worker HTTP port"
            )
        used_engine_ports.add(engine_port)
        engine_netloc = f"{host_literal}:{engine_port}"
        engine_url = urlunsplit((parsed.scheme, engine_netloc, parsed.path, "", ""))
        assignments.append(LocalWorkerNInferOrigin(
            worker_id=worker.id,
            ordinal=ordinal,
            worker_http_port=worker.port,
            engine_port=engine_port,
            url=engine_url,
        ))

    return LocalWorkerNInferOriginPlan(
        configured_base_url=configured_base_url,
        learner_http_port=learner_http_port,
        worker_http_ports=tuple(worker_http_ports),
        assignments=tuple(assignments),
    )


def _read_boot_id(path: Path = Path("/proc/sys/kernel/random/boot_id")) -> str:
    try:
        value = Path(path).read_text(encoding="ascii").strip()
    except OSError as exc:
        raise LocalWorkerFleetError("host boot identity is unavailable") from exc
    if not _BOOT_ID.fullmatch(value):
        raise LocalWorkerFleetError("host boot identity is invalid")
    return value


class LocalWorkerFleet:
    """Create and remove only Worker containers bearing this deployment's labels."""

    def __init__(
        self,
        *,
        root: Path,
        deployment_path: Path,
        manager_container: str,
        worker_image: str | None,
        worker_release: str | Path | WorkerRelease | None = None,
        container_uid: str,
        default_config: str,
        container_workdir: str,
        container_python: str,
        worker_start_timeout: float,
        command_env: Mapping[str, str],
        runner: Callable[..., Any],
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    ):
        self.root = Path(root).resolve()
        self.deployment_path = Path(deployment_path).resolve(strict=True)
        self.manager_container = manager_container
        self.worker_image = worker_image
        if (worker_image is None) == (worker_release is None):
            raise LocalWorkerFleetError("configure exactly one of worker_image or worker_release")
        try:
            self.worker_release = load_worker_release(worker_release) if worker_release is not None else None
        except WorkerReleaseError as exc:
            raise LocalWorkerFleetError(f"local Worker release is invalid: {exc}") from exc
        match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", container_uid)
        if match is None or int(match.group(1)) != os.getuid():
            raise LocalWorkerFleetError(
                "Worker container uid must match the owner of the private local runtime files"
            )
        self.container_uid = container_uid
        self.default_config = default_config
        self.container_workdir = container_workdir
        self.container_python = container_python
        self.worker_start_timeout = worker_start_timeout
        self.command_env = dict(command_env)
        self.runner = runner
        self.sleep = sleep
        self.monotonic = monotonic
        self.boot_id_path = Path(boot_id_path)
        self.deployment_id = _sha256(f"{self.deployment_path}\0{self.manager_container}".encode())[:20]
        self.state_root = self.root / "outputs" / "local-worker-services" / self.deployment_id
        self.owner_path = self.state_root / "owner.json"
        self.manifest_path = self.state_root / "manifest.json"
        self.runtime_config_host_path = self.state_root / "runtime-service.yaml"
        self.runtime_environment_host_path = self.state_root / "runtime-workers.env"

    def _run(self, argv: Sequence[str], *, timeout: float,
             injected: Mapping[str, str] | None = None) -> Any:
        environment = dict(self.command_env)
        if injected:
            environment.update(injected)
        try:
            result = self.runner(list(argv), env=environment, text=True,
                                 capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise LocalWorkerFleetError("local Worker Docker operation timed out") from exc
        except OSError as exc:
            raise LocalWorkerFleetError("local Worker Docker command could not be started") from exc
        return result

    def _require_success(self, argv: Sequence[str], *, timeout: float, label: str) -> Any:
        result = self._run(argv, timeout=timeout)
        if result.returncode != 0:
            raise LocalWorkerFleetError(f"{label} failed with status {result.returncode}")
        return result

    def _check_state_path(self, *, required: bool) -> bool:
        """Check every project-owned ancestor without following a symlink."""
        managed_root = self.root
        for component in ("outputs", "local-worker-services", self.deployment_id):
            managed_root = managed_root / component
            try:
                metadata = managed_root.lstat()
            except FileNotFoundError:
                if required:
                    raise LocalWorkerFleetError("local Worker state directory is missing") from None
                continue
            except OSError as exc:
                raise LocalWorkerFleetError("local Worker state path cannot be inspected") from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise LocalWorkerFleetError("local Worker state path contains an unsafe directory")
            if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o002:
                raise LocalWorkerFleetError("local Worker state path has unsafe ownership or write permissions")
        if self.state_root.exists():
            _check_private_directory(self.state_root)
            return True
        return False

    def _initialize_owner(self) -> None:
        # Do not let a pre-existing symlink redirect ownership state outside the
        # project. Check each managed ancestor before creating the leaf.
        self._check_state_path(required=False)
        self.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _check_private_directory(self.state_root)
        expected = {"schema": _MANIFEST_SCHEMA, "deployment_id": self.deployment_id}
        if self.owner_path.exists() or self.owner_path.is_symlink():
            _check_private_file(self.owner_path, "local Worker owner marker")
            try:
                actual = json.loads(self.owner_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise LocalWorkerFleetError("local Worker owner marker is unreadable") from None
            if actual != expected:
                raise LocalWorkerFleetError("local Worker state belongs to a different deployment")
            return
        _write_private(self.owner_path, _canonical_json(expected) + b"\n")

    def _verify_owner(self) -> None:
        _check_private_file(self.owner_path, "local Worker owner marker")
        try:
            owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise LocalWorkerFleetError("local Worker owner marker is unreadable") from None
        if owner != {"schema": _MANIFEST_SCHEMA, "deployment_id": self.deployment_id}:
            raise LocalWorkerFleetError("local Worker state belongs to a different deployment")

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists() and not self.manifest_path.is_symlink():
            return None
        _check_private_file(self.manifest_path, "local Worker manifest")
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise LocalWorkerFleetError("local Worker manifest is unreadable") from None
        if (not isinstance(value, dict) or value.get("schema") != _MANIFEST_SCHEMA
                or value.get("deployment_id") != self.deployment_id
                or not isinstance(value.get("workers"), list)):
            raise LocalWorkerFleetError("local Worker manifest does not match this deployment")
        return value

    def _write_manifest(self, value: dict[str, Any]) -> None:
        _write_private(self.manifest_path, _canonical_json(value) + b"\n")

    def existing_runtime_paths(self) -> tuple[str, str] | None:
        """Return manager-container paths for the persisted service config and env."""
        if (not self._check_state_path(required=False)
                or (not self.manifest_path.exists() and not self.manifest_path.is_symlink())):
            return None
        self._verify_owner()
        self._read_manifest()
        for path in (self.runtime_config_host_path, self.runtime_environment_host_path):
            try:
                metadata = path.lstat()
            except OSError:
                raise LocalWorkerFleetError("persisted local Worker runtime files are unavailable") from None
            if (not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
                    or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600):
                raise LocalWorkerFleetError("persisted local Worker runtime files have unsafe ownership or mode")
        mounts = self._inspect_mounts()
        return (
            _container_path_for_host(mounts, self.runtime_config_host_path),
            _container_path_for_host(mounts, self.runtime_environment_host_path),
        )

    def _inspect_mounts(self) -> list[dict[str, Any]]:
        result = self._require_success(
            ["docker", "inspect", "--format", "{{json .Mounts}}", self.manager_container],
            timeout=30, label="manager mount inspection",
        )
        try:
            mounts = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            raise LocalWorkerFleetError("manager mount inspection returned invalid data") from None
        if not isinstance(mounts, list) or any(not isinstance(row, dict) for row in mounts):
            raise LocalWorkerFleetError("manager mount inspection returned invalid data")
        return mounts

    def _inspect_image(self, image: str, *, expected_image_id: str | None = None) -> str:
        result = self._require_success(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            timeout=30, label="prebuilt Worker image inspection",
        )
        image_id = (result.stdout or "").strip()
        if not _IMAGE_ID.fullmatch(image_id):
            raise LocalWorkerFleetError("prebuilt Worker image did not resolve to an immutable local image ID")
        if expected_image_id is not None and image_id != expected_image_id:
            raise LocalWorkerFleetError("local Worker release image differs from its pinned archive descriptor")
        return image_id

    def _verify_release_image_labels(self, image: str, resolved: ResolvedWorkerImage) -> None:
        result = self._require_success(
            ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", image],
            timeout=30, label="Worker release image label inspection",
        )
        try:
            labels = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            raise LocalWorkerFleetError("local Worker release image has malformed Docker labels") from None
        expected = expected_worker_image_labels(resolved)
        if not isinstance(labels, dict) or any(labels.get(key) != value for key, value in expected.items()):
            raise LocalWorkerFleetError("local Worker release image labels do not match the pinned release catalog")

    def _gpu_inventory(self) -> dict[int, tuple[str, str | None]]:
        result = self._require_success(
            ["nvidia-smi", "--query-gpu=index,uuid,compute_cap", "--format=csv,noheader,nounits"],
            timeout=20, label="host GPU inventory query",
        )
        values: dict[int, tuple[str, str | None]] = {}
        seen_uuids: set[str] = set()
        for line in (result.stdout or "").splitlines():
            parts = line.split(",")
            if len(parts) != 3:
                raise LocalWorkerFleetError("host GPU inventory is malformed")
            try:
                number = int(parts[0].strip())
            except ValueError:
                raise LocalWorkerFleetError("host GPU inventory has an invalid index") from None
            gpu_uuid = parts[1].strip()
            capability = parts[2].strip() or None
            if not gpu_uuid.startswith("GPU-") or number in values or gpu_uuid in seen_uuids:
                raise LocalWorkerFleetError("host GPU inventory has an invalid or duplicate identity")
            values[number] = (gpu_uuid, capability)
            seen_uuids.add(gpu_uuid)
        if not values:
            raise LocalWorkerFleetError("host has no visible NVIDIA GPUs")
        return values

    def manager_gpu_inventory(self) -> dict[int, str]:
        """Map container-visible CUDA ordinals to physical UUIDs before executor launch."""
        result = self._require_success(
            ["docker", "exec", self.manager_container, "nvidia-smi", "--query-gpu=index,uuid",
             "--format=csv,noheader,nounits"],
            timeout=20, label="manager-container GPU inventory query",
        )
        values: dict[int, str] = {}
        seen: set[str] = set()
        for line in (result.stdout or "").splitlines():
            parts = line.split(",")
            if len(parts) != 2:
                raise LocalWorkerFleetError("manager-container GPU inventory is malformed")
            try:
                index = int(parts[0].strip())
            except ValueError:
                raise LocalWorkerFleetError("manager-container GPU index is invalid") from None
            gpu_uuid = parts[1].strip()
            if index in values or not gpu_uuid.startswith("GPU-") or gpu_uuid in seen:
                raise LocalWorkerFleetError("manager-container GPU inventory has duplicate or invalid identities")
            values[index] = gpu_uuid
            seen.add(gpu_uuid)
        if not values:
            raise LocalWorkerFleetError("manager container has no visible NVIDIA GPUs")
        return values

    def _token(self, path: Path, worker_id: str) -> str:
        # Reuse the local launcher's owner/mode/link-count checks. Importing at
        # call time avoids a module cycle while the local command is loading.
        from .local import read_or_create_secret

        return read_or_create_secret(path, label=f"local Worker {worker_id}", create=True)

    def prepare(self) -> LocalWorkerPlan:
        """Materialize a service-scoped copy of the configured runtime profile."""
        self._initialize_owner()
        mounts = self._inspect_mounts()
        profile_path = _mount_source(mounts, self._default_config_container_path)
        from ..runtime.config import load_config

        runtime = load_config(profile_path, for_local_worker_plan=True)
        if (self.worker_release is not None
                and runtime.worker_inference.backend != self.worker_release.backend):
            raise LocalWorkerFleetError(
                f"Worker release backend {self.worker_release.backend!r} does not match runtime profile backend "
                f"{runtime.worker_inference.backend!r}"
            )
        t2_identity_record = None
        t2_identity_container_path = None
        if getattr(runtime, "model_backend", "default") == "ternary":
            identity_value = getattr(runtime, "artifact_identity_path", None)
            if not isinstance(identity_value, str) or not identity_value.strip():
                raise LocalWorkerFleetError("T2 runtime profile has no verified artifact identity path")
            identity_source = Path(identity_value).expanduser()
            if identity_source.is_symlink():
                raise LocalWorkerFleetError("T2 artifact identity file cannot be a symlink")
            try:
                identity_host_path = identity_source.resolve(strict=True)
                identity_sha256, identity_size = _file_content_identity(
                    identity_host_path, label="T2 artifact identity file",
                )
                t2_identity_container_path = _container_path_for_host(mounts, identity_host_path)
            except (OSError, LocalWorkerFleetError) as exc:
                raise LocalWorkerFleetError(
                    "T2 artifact identity file must resolve through an inspected manager bind mount"
                ) from exc
            t2_identity_record = {
                "container_path": t2_identity_container_path,
                "host_path": str(identity_host_path),
                "sha256": identity_sha256,
                "size_bytes": identity_size,
            }
        model_container_path = runtime.worker_inference.model_path
        model_host_path = _mount_source(mounts, model_container_path)
        t2_file_model = (
            runtime.worker_inference.backend == "ninfer"
            and runtime.worker_inference.quantization == "ternary"
        )
        if t2_file_model:
            if not model_host_path.is_file():
                raise LocalWorkerFleetError("T2 Worker model bind source must be a regular NInfer artifact file")
            if runtime.worker_inference.ninfer_artifact_path != model_container_path:
                raise LocalWorkerFleetError(
                    "T2 Worker model_path must equal ninfer_artifact_path so the same base is mounted once"
                )
        elif not model_host_path.is_dir():
            raise LocalWorkerFleetError("Worker model bind source is not a directory")
        ninfer_inputs = _resolve_ninfer_inputs(
            runtime.worker_inference,
            mounts=mounts,
            model_container_path=model_container_path,
            model_host_path=model_host_path,
            model_is_file=t2_file_model,
        )
        managed_workers = [worker for worker in runtime.workers if worker.scope in {"local", "service"}]
        ninfer_origin_plan = _plan_ninfer_origins(
            runtime.worker_inference,
            learner_http_port=runtime.learner.port,
            workers=managed_workers,
        )
        ninfer_urls = ({item.worker_id: item.url for item in ninfer_origin_plan.assignments}
                       if ninfer_origin_plan is not None else {})
        runtime_config_container_path = _container_path_for_host(mounts, self.runtime_config_host_path)
        runtime_environment_container_path = _container_path_for_host(mounts, self.runtime_environment_host_path)
        gpu_inventory = self._gpu_inventory()
        boot_id = _read_boot_id(self.boot_id_path)

        data = _yaml_value(runtime.as_dict())
        if t2_identity_container_path is not None:
            # load_config verified the identity against the host file. The
            # generated manager config must use the inspected bind destination.
            data["artifact_identity_path"] = t2_identity_container_path
        specs: list[LocalWorkerSpec] = []
        inspected_images: dict[str, str] = {}
        environment: dict[str, str] = {}
        used_env_names: set[str] = set()
        for worker in managed_workers:
            worker_id = _safe_worker_slug(worker.id)
            if worker.gpu not in gpu_inventory:
                raise LocalWorkerFleetError(f"configured GPU {worker.gpu} for Worker {worker_id} is unavailable")
            gpu_uuid, compute_capability = gpu_inventory[worker.gpu]
            resolved_image: ResolvedWorkerImage | None = None
            if self.worker_release is not None:
                if compute_capability is None:
                    raise LocalWorkerFleetError(
                        f"Worker {worker_id} GPU {gpu_uuid} has no compute capability; cannot select a Worker release image"
                    )
                try:
                    resolved_image = select_worker_image(self.worker_release, compute_capability)
                except WorkerReleaseError as exc:
                    raise LocalWorkerFleetError(
                        f"cannot select a Worker release image for GPU {gpu_uuid} compute capability "
                        f"{compute_capability}: {exc}"
                    ) from exc
                worker_image = resolved_image.descriptor.image_id
                image_archive = resolved_image.descriptor
                release_identity = resolved_image.release_identity
            else:
                assert self.worker_image is not None
                worker_image = self.worker_image
                image_archive = None
                release_identity = None
            expected_image_id = image_archive.image_id if image_archive is not None else None
            if worker_image not in inspected_images:
                inspected_images[worker_image] = self._inspect_image(
                    worker_image, expected_image_id=expected_image_id,
                )
            if resolved_image is not None:
                self._verify_release_image_labels(worker_image, resolved_image)
            worker_image_id = inspected_images[worker_image]
            auth_env = _auth_environment_name(worker_id)
            if auth_env in used_env_names:
                raise LocalWorkerFleetError("local Worker ids collide after environment-name normalization")
            used_env_names.add(auth_env)
            token_path = self.state_root / "tokens" / f"{worker_id}.token"
            token = self._token(token_path, worker_id)
            environment[auth_env] = token

            worker_state = self.state_root / "workers" / worker_id
            worker_state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _check_private_directory(worker_state.parent)
            worker_state.mkdir(parents=True, exist_ok=True, mode=0o700)
            _check_private_directory(worker_state)
            worker_ninfer_url = ninfer_urls.get(worker.id)
            inference_data = asdict(runtime.worker_inference)
            ninfer_options = inference_data.get("ninfer_options")
            if (isinstance(ninfer_options, dict)
                    and ninfer_options.get("lora_gdn_projections") is False):
                ninfer_options.pop("lora_gdn_projections")
            if worker_ninfer_url is not None:
                inference_data["ninfer_url"] = worker_ninfer_url
            raw = {
                "worker_id": worker_id,
                "node_id": worker.node_id,
                "host": "127.0.0.1",
                "port": worker.port,
                "learner_url": f"http://127.0.0.1:{runtime.learner.port}",
                "adapter_cache": "/state/adapters",
                "telemetry_dir": "/state/telemetry",
                "lora_rank": runtime.lora.rank,
                "lora_alpha": runtime.lora.alpha,
                "request_timeout": runtime.learner.request_timeout,
                "inference": inference_data,
                "deployment_scope": "service",
                "artifact_transport": "direct",
            }
            if t2_identity_container_path is not None:
                raw["artifact_identity_path"] = t2_identity_container_path
            if _worker_requires_residency(runtime, worker.id):
                raw["residency"] = {"enabled": True,
                    "weight_source": runtime.learner.worker_suspend_weight_source,
                    "engine_lifecycle": runtime.learner.worker_engine_lifecycle,
                    "drain_timeout_seconds": runtime.learner.gpu_sharing_timeout}
            content = _canonical_json(raw) + b"\n"
            config_path = worker_state / "worker.json"
            _write_private(config_path, content)
            config_sha = _sha256(content)
            name_suffix = _sha256(worker_id.encode())[:8]
            container_name = f"qwen-rl-local-{self.deployment_id}-{name_suffix}"
            specs.append(LocalWorkerSpec(
                worker_id=worker_id,
                host_port=worker.port,
                gpu_index=worker.gpu,
                gpu_uuid=gpu_uuid,
                compute_capability=compute_capability,
                worker_image=worker_image,
                worker_image_id=worker_image_id,
                image_archive=image_archive,
                worker_release_identity=release_identity,
                site_id=worker.site_id,
                node_id=worker.node_id,
                auth_token_env=auth_env,
                auth_token_path=token_path,
                config_path=config_path,
                state_dir=worker_state,
                container_name=container_name,
                config_sha256=config_sha,
                ninfer_url=worker_ninfer_url,
                auth_token=token,
            ))

        if not specs:
            raise LocalWorkerFleetError("local runtime profile has no Workers to manage")
        for worker in runtime.workers:
            if worker.scope != "gateway":
                continue
            value = self.command_env.get(worker.auth_token_env)
            if not value:
                raise LocalWorkerFleetError(f"missing Gateway credential environment: {worker.auth_token_env}")
            if worker.auth_token_env in used_env_names:
                raise LocalWorkerFleetError("Gateway and managed Worker credentials must have distinct names")
            environment[worker.auth_token_env] = value
        env_text = "".join(
            f"export {name}={shlex.quote(value)}\n" for name, value in sorted(environment.items())
        ).encode("utf-8")
        _write_private(self.runtime_environment_host_path, env_text)

        specs_by_id = {spec.worker_id: spec for spec in specs}
        for worker_data in data["workers"]:
            spec = specs_by_id.get(worker_data["id"])
            if spec is None:
                continue
            worker_data.update({
                "scope": "service",
                "host": None,
                "endpoint": f"http://127.0.0.1:{spec.host_port}",
                "url_prefix": "",
                "auth_token_env": spec.auth_token_env,
                "boot_id": boot_id,
                "telemetry_epoch": "",
                "model_sha256": "",
            })
        runtime_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).encode("utf-8")
        _write_private(self.runtime_config_host_path, runtime_text)

        fingerprint_identity = {
            "deployment_id": self.deployment_id,
            "manager_container": self.manager_container,
            "model_container_path": model_container_path,
            "model_host_path": str(model_host_path),
            "runtime_workers": [
                {"id": spec.worker_id, "site_id": spec.site_id, "node_id": spec.node_id,
                 "port": spec.host_port, "gpu": spec.gpu_index,
                 "gpu_uuid": spec.gpu_uuid, "gpu_compute_capability": spec.compute_capability,
                 "worker_image": spec.worker_image, "worker_image_id": spec.worker_image_id,
                 "worker_release_identity": (spec.worker_release_identity.model_dump(mode="json")
                                             if spec.worker_release_identity is not None else None),
                 "config_sha256": spec.config_sha256}
                for spec in specs
            ],
        }
        external_workers = [row for row in data["workers"] if row["scope"] == "gateway"]
        if external_workers:
            fingerprint_identity["external_workers"] = external_workers
        if ninfer_inputs:
            fingerprint_identity["ninfer_inputs"] = [
                {
                    "field_name": item.field_name,
                    "container_path": item.container_path,
                    "host_path": str(item.host_path),
                    "model_mount_covers_file": item.model_mount_covers_file,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in ninfer_inputs
            ]
        if ninfer_origin_plan is not None:
            fingerprint_identity["ninfer_origin_plan"] = asdict(ninfer_origin_plan)
        if t2_identity_record is not None:
            fingerprint_identity["artifact_identity"] = t2_identity_record
        fingerprint = _sha256(_canonical_json(fingerprint_identity))
        manifest = self._read_manifest()
        if (manifest is not None and manifest.get("fingerprint") not in {None, fingerprint}
                and manifest.get("workers")):
            raise LocalWorkerFleetError(
                "local Worker profile changed while owned sidecars remain; stop the managed runtime first"
            )
        if manifest is not None and manifest.get("workers"):
            prior = {item.get("worker_id"): item for item in manifest["workers"] if isinstance(item, dict)}
            if set(prior) != {spec.worker_id for spec in specs}:
                raise LocalWorkerFleetError("local Worker membership changed while owned sidecars remain")

        prior_workers = {}
        if manifest is not None and manifest.get("fingerprint") == fingerprint:
            prior_workers = {
                row.get("worker_id"): row
                for row in manifest["workers"] if isinstance(row, dict)
            }
        records = [
            spec.manifest_record(container_id=prior_workers.get(spec.worker_id, {}).get("container_id"))
            for spec in specs
        ]
        manifest_value = {
            "schema": _MANIFEST_SCHEMA,
            "deployment_id": self.deployment_id,
            "fingerprint": fingerprint,
            "worker_image": self.worker_image,
            "worker_release_identity": (
                self.worker_release.release_identity.model_dump(mode="json")
                if self.worker_release is not None else None
            ),
            "manager_container": self.manager_container,
            "model_container_path": model_container_path,
            "model_host_path": str(model_host_path),
            "workers": records,
        }
        if ninfer_inputs:
            manifest_value["ninfer_inputs"] = [
                {
                    "field_name": item.field_name,
                    "container_path": item.container_path,
                    "host_path": str(item.host_path),
                    "model_mount_covers_file": item.model_mount_covers_file,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in ninfer_inputs
            ]
        if ninfer_origin_plan is not None:
            manifest_value["ninfer_origin_plan"] = asdict(ninfer_origin_plan)
        if t2_identity_record is not None:
            manifest_value["artifact_identity"] = t2_identity_record
        self._write_manifest(manifest_value)
        return LocalWorkerPlan(
            deployment_id=self.deployment_id,
            fingerprint=fingerprint,
            model_container_path=model_container_path,
            model_host_path=model_host_path,
            boot_id=boot_id,
            runtime_config=runtime,
            runtime_config_host_path=self.runtime_config_host_path,
            runtime_config_container_path=runtime_config_container_path,
            runtime_environment_host_path=self.runtime_environment_host_path,
            runtime_environment_container_path=runtime_environment_container_path,
            manifest_path=self.manifest_path,
            ninfer_inputs=ninfer_inputs,
            ninfer_origin_plan=ninfer_origin_plan,
            specs=tuple(specs),
        )

    @property
    def _default_config_container_path(self) -> str:
        return self._configured_default_config

    @property
    def _configured_default_config(self) -> str:
        value = getattr(self, "default_config", None)
        if not isinstance(value, str):
            raise LocalWorkerFleetError("local runtime profile path was not configured")
        return value

    def configure_default_profile(self, path: str) -> None:
        if not path.startswith(self.container_workdir.rstrip("/") + "/"):
            raise LocalWorkerFleetError("runtime profile must be inside the manager's /work mount")
        self.default_config = path

    def _inspect_container(self, name: str) -> dict[str, Any] | None:
        result = self._run(
            ["docker", "inspect", "--format",
             "{{json .Config.Labels}}|{{.Id}}|{{.State.Running}}|{{.Image}}", name],
            timeout=30,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").lower()
            if "no such object" in stderr or "no such container" in stderr:
                return None
            raise LocalWorkerFleetError("local Worker container inspection failed")
        fields = (result.stdout or "").strip().split("|", 3)
        if len(fields) != 4:
            raise LocalWorkerFleetError("local Worker container inspection returned malformed data")
        try:
            labels = json.loads(fields[0])
        except json.JSONDecodeError:
            raise LocalWorkerFleetError("local Worker container labels are invalid") from None
        if not isinstance(labels, dict) or not _CONTAINER_ID.fullmatch(fields[1]):
            raise LocalWorkerFleetError("local Worker container identity is invalid")
        return {"labels": labels, "id": fields[1], "running": fields[2] == "true", "image_id": fields[3]}

    def _expected_labels(self, plan: LocalWorkerPlan, spec: LocalWorkerSpec) -> dict[str, str]:
        return {
            _OWNER_LABEL: _OWNER_VALUE,
            _DEPLOYMENT_LABEL: plan.deployment_id,
            _WORKER_LABEL: spec.worker_id,
            _FINGERPRINT_LABEL: plan.fingerprint,
            _CONFIG_SHA_LABEL: spec.config_sha256,
            _IMAGE_ID_LABEL: spec.worker_image_id,
        }

    def _container_args(self, plan: LocalWorkerPlan, spec: LocalWorkerSpec) -> list[str]:
        if ("," in str(plan.model_host_path) or ":" in str(plan.model_host_path)
                or "," in str(spec.config_path) or ":" in str(spec.config_path)
                or "," in str(spec.auth_token_path) or ":" in str(spec.auth_token_path)
                or "," in str(spec.state_dir) or ":" in str(spec.state_dir)
                or any("," in str(item.host_path) or ":" in str(item.host_path)
                       or "," in item.container_path or ":" in item.container_path
                       for item in plan.ninfer_inputs)):
            raise LocalWorkerFleetError("local Worker bind paths cannot contain Docker mount separators")
        argv = [
            "docker", "create",
            "--name", spec.container_name,
            "--network", f"container:{self.manager_container}",
            "--gpus", f"device={spec.gpu_uuid}",
            "--user", self.container_uid,
            "--read-only", "--init", "--restart", "no",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--shm-size", "8g", "--tmpfs", "/tmp:rw,nosuid,nodev,size=4g",
            "--env", "ARC_ENV_TOKEN_FILE=/run/secrets/worker_auth_token",
            # Worker stdout goes to `docker logs`; without this it is block-buffered.
            "--env", "PYTHONUNBUFFERED=1",
            "--env", f"GPU_UUID={spec.gpu_uuid}",
            "--env", f"QWEN_SITE_ID={spec.site_id}",
            "--env", f"QWEN_NODE_ID={spec.node_id}",
        ]
        for key, value in self._expected_labels(plan, spec).items():
            argv.extend(("--label", f"{key}={value}"))
        mounts = [
            (spec.config_path, "/config/worker.json", True),
            (plan.model_host_path, plan.model_container_path, True),
            (spec.state_dir, "/state", False),
            (spec.auth_token_path, "/run/secrets/worker_auth_token", True),
        ]
        if plan.ninfer_origin_plan is not None:
            # Shared across deployment IDs/Workers, scoped to an immutable image.
            profile_root = self.root / "outputs" / "ninfer-device-profiles"
            profile_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            _check_private_directory(profile_root)
            profile_cache = profile_root / spec.worker_image_id.removeprefix("sha256:")
            profile_cache.mkdir(exist_ok=True, mode=0o700)
            _check_private_directory(profile_cache)
            mounts.append((profile_cache, "/var/cache/qwen-ninfer-profile", False))
            argv.extend(("--env", "QWEN_NINFER_DEVICE_PROFILE_CACHE=/var/cache/qwen-ninfer-profile"))
        mounts.extend(
            (item.host_path, item.container_path, True)
            for item in plan.ninfer_inputs
            if not item.model_mount_covers_file
        )
        for source, target, readonly in mounts:
            mount = f"type=bind,source={source},target={target}"
            if readonly:
                mount += ",readonly"
            argv.extend(("--mount", mount))
        argv.extend((spec.worker_image_id, "cluster", "worker", "--config", "/config/worker.json",
                     "--host", "127.0.0.1", "--port", str(spec.host_port)))
        return argv

    def _save_worker_id(self, worker_id: str, container_id: str) -> None:
        manifest = self._read_manifest()
        if manifest is None:
            raise LocalWorkerFleetError("local Worker manifest disappeared during start")
        rows = manifest["workers"]
        match = next((row for row in rows if row.get("worker_id") == worker_id), None)
        if match is None:
            raise LocalWorkerFleetError("local Worker manifest lost a planned member")
        match["container_id"] = container_id
        self._write_manifest(manifest)

    def _probe_worker(self, plan: LocalWorkerPlan, spec: LocalWorkerSpec) -> bool:
        url = f"http://127.0.0.1:{spec.host_port}/health"
        env = dict(self.command_env)
        env["ARC_ENV_TOKEN"] = spec.auth_token
        argv = [
            "docker", "exec", "--env", "ARC_ENV_TOKEN", self.manager_container,
            self.container_python, "-B", "-c", _HEALTH_PROBE, url,
        ]
        result = self._run(argv, timeout=8, injected={"ARC_ENV_TOKEN": spec.auth_token})
        if result.returncode != 0:
            return False
        try:
            health = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return False
        return (
            isinstance(health, dict)
            and health.get("ready") is True
            and health.get("role") == "worker"
            and health.get("worker_id") == spec.worker_id
            and isinstance(health.get("deployment"), dict)
            and health["deployment"].get("scope") == "service"
            and health["deployment"].get("site_id") == spec.site_id
            and health["deployment"].get("node_id") == spec.node_id
            and health.get("boot_id") == plan.boot_id
            and isinstance(health.get("telemetry_epoch"), str)
            and bool(health.get("telemetry_epoch"))
        )

    def worker_identity(self, plan: LocalWorkerPlan, spec: LocalWorkerSpec) -> dict[str, Any]:
        """Read the service identity through the manager's shared Docker network."""
        url = f"http://127.0.0.1:{spec.host_port}/health"
        result = self._run([
            "docker", "exec", "--env", "ARC_ENV_TOKEN", self.manager_container,
            self.container_python, "-B", "-c", _HEALTH_PROBE, url,
        ], timeout=8, injected={"ARC_ENV_TOKEN": spec.auth_token})
        if result.returncode != 0:
            raise LocalWorkerFleetError(f"local Worker {spec.worker_id} identity probe failed")
        try:
            health = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            raise LocalWorkerFleetError(f"local Worker {spec.worker_id} identity probe returned invalid data") from None
        if not isinstance(health, dict):
            raise LocalWorkerFleetError(f"local Worker {spec.worker_id} identity probe returned invalid data")
        return health

    def _wait_ready(self, plan: LocalWorkerPlan, spec: LocalWorkerSpec) -> None:
        deadline = self.monotonic() + self.worker_start_timeout
        while True:
            if self._probe_worker(plan, spec):
                return
            container = self._inspect_container(spec.container_name)
            if container is None:
                raise LocalWorkerFleetError(f"local Worker {spec.worker_id} disappeared before readiness")
            self._verify_owned(plan, spec, container)
            if not container["running"]:
                raise LocalWorkerFleetError(f"local Worker {spec.worker_id} exited before readiness; inspect its Docker logs")
            if self.monotonic() >= deadline:
                raise LocalWorkerFleetError(f"local Worker {spec.worker_id} did not become ready before timeout")
            self.sleep(min(2.0, max(0.0, deadline - self.monotonic())))

    def _verify_owned(self, plan: LocalWorkerPlan, spec: LocalWorkerSpec, container: dict[str, Any]) -> None:
        labels = self._expected_labels(plan, spec)
        if any(container["labels"].get(key) != value for key, value in labels.items()):
            raise LocalWorkerFleetError(
                f"container name reserved for Worker {spec.worker_id} is not owned by this deployment"
            )
        if container["image_id"] != spec.worker_image_id:
            raise LocalWorkerFleetError(f"owned Worker {spec.worker_id} container uses a different image")

    def start(self, plan: LocalWorkerPlan) -> tuple[tuple[str, str], ...]:
        manifest = self._read_manifest()
        if manifest is None or manifest.get("fingerprint") != plan.fingerprint:
            raise LocalWorkerFleetError("local Worker manifest is missing or has a different deployment fingerprint")
        created: list[tuple[str, str]] = []
        try:
            for spec in plan.specs:
                container = self._inspect_container(spec.container_name)
                if container is None:
                    result = self._require_success(
                        self._container_args(plan, spec), timeout=120,
                        label=f"local Worker {spec.worker_id} container creation",
                    )
                    container_id = (result.stdout or "").strip()
                    if not _CONTAINER_ID.fullmatch(container_id):
                        raise LocalWorkerFleetError("Docker returned an invalid local Worker container ID")
                    container = self._inspect_container(spec.container_name)
                    if container is None:
                        raise LocalWorkerFleetError("created local Worker container could not be inspected")
                    self._verify_owned(plan, spec, container)
                    if container["id"] != container_id:
                        raise LocalWorkerFleetError("Docker returned a different id than the owned Worker container")
                    self._save_worker_id(spec.worker_id, container_id)
                    created.append((spec.worker_id, container_id))
                else:
                    self._verify_owned(plan, spec, container)
                    self._save_worker_id(spec.worker_id, container["id"])
                if not container["running"]:
                    self._require_success(
                        ["docker", "start", container["id"]], timeout=120,
                        label=f"local Worker {spec.worker_id} start",
                    )
                    if (spec.worker_id, container["id"]) not in created:
                        created.append((spec.worker_id, container["id"]))
            for spec in plan.specs:
                self._wait_ready(plan, spec)
            return tuple(created)
        except BaseException:
            for worker_id, container_id in reversed(created):
                self._remove_container(container_id, worker_id, plan.deployment_id, plan.fingerprint)
            raise

    def rollback_started(self, plan: LocalWorkerPlan, started: Sequence[tuple[str, str]]) -> None:
        """Remove only sidecars this call newly created or started."""
        for worker_id, container_id in reversed(tuple(started)):
            self._remove_container(container_id, worker_id, plan.deployment_id, plan.fingerprint)
            manifest = self._read_manifest()
            if manifest is None:
                raise LocalWorkerFleetError("local Worker manifest disappeared during rollback")
            row = next((item for item in manifest["workers"] if item.get("worker_id") == worker_id), None)
            if row is None or row.get("container_id") != container_id:
                raise LocalWorkerFleetError("local Worker rollback identity differs from its manifest")
            row["container_id"] = None
            self._write_manifest(manifest)

    def _remove_container(self, container_id: str, worker_id: str,
                          deployment_id: str, fingerprint: str) -> None:
        if not _CONTAINER_ID.fullmatch(container_id):
            raise LocalWorkerFleetError("refusing to remove a Worker with an invalid container ID")
        name = f"qwen-rl-local-{deployment_id}-{_sha256(worker_id.encode())[:8]}"
        container = self._inspect_container(name)
        if container is None:
            return
        manifest = self._read_manifest()
        if manifest is None:
            raise LocalWorkerFleetError("local Worker manifest is missing during removal")
        row = next((item for item in manifest["workers"]
                    if isinstance(item, dict) and item.get("worker_id") == worker_id), None)
        if row is None or not isinstance(row.get("config_sha256"), str):
            raise LocalWorkerFleetError("local Worker manifest has no exact ownership fingerprint")
        expected_image_id = row.get("worker_image_id")
        if not isinstance(expected_image_id, str) or not _IMAGE_ID.fullmatch(expected_image_id):
            raise LocalWorkerFleetError("local Worker manifest has no exact per-worker image identity")
        labels = container["labels"]
        if (container["id"] != container_id or labels.get(_OWNER_LABEL) != _OWNER_VALUE
                or labels.get(_DEPLOYMENT_LABEL) != deployment_id
                or labels.get(_WORKER_LABEL) != worker_id
                or labels.get(_FINGERPRINT_LABEL) != fingerprint
                or labels.get(_CONFIG_SHA_LABEL) != row["config_sha256"]
                or labels.get(_IMAGE_ID_LABEL) != expected_image_id
                or container["image_id"] != expected_image_id):
            raise LocalWorkerFleetError("refusing to remove a container that is not exactly manifest-owned")
        if container["running"]:
            self._require_success(["docker", "stop", "--time", "20", container_id],
                                  timeout=45, label=f"local Worker {worker_id} stop")
        self._require_success(["docker", "rm", container_id], timeout=45,
                              label=f"local Worker {worker_id} removal")

    def stop_owned(self) -> None:
        """Stop only the exact containers recorded in this deployment manifest."""
        if not self._check_state_path(required=False):
            return
        self._verify_owner()
        manifest = self._read_manifest()
        if manifest is None:
            return
        deployment_id, fingerprint = manifest["deployment_id"], manifest.get("fingerprint")
        for row in reversed(manifest["workers"]):
            if not isinstance(row, dict):
                raise LocalWorkerFleetError("local Worker manifest contains an invalid row")
            worker_id, container_id = row.get("worker_id"), row.get("container_id")
            if not isinstance(worker_id, str) or container_id is None:
                continue
            if not isinstance(container_id, str):
                raise LocalWorkerFleetError("local Worker manifest contains an invalid container ID")
            self._remove_container(container_id, worker_id, deployment_id, fingerprint)
        manifest["workers"] = []
        self._write_manifest(manifest)
