"""Host-only ownership and lifecycle for a portable Learner container."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import httpx

from .config import (
    CONTAINER_PYTHON,
    PROFILE_CONTAINER_PATH,
    LearnerDeployment,
    LearnerDeploymentError,
    check_learner_profile,
)


_OWNER_LABEL = "io.qwen-rl.owner"
_DEPLOYMENT_LABEL = "io.qwen-rl.deployment"
_PROFILE_LABEL = "io.qwen-rl.profile-sha256"
_IMAGE_LABEL = "io.qwen-rl.image-id"
_OWNER_VALUE = "portable-learner/v1"
_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$")
_PROFILE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_INTERNAL_MODULE = "qwen_rl.deployment.learner.container"
_SECRET_MAX_BYTES = 16_384


class LearnerManagerError(LearnerDeploymentError):
    """An expected, user-safe Learner host management failure."""


class LearnerManager:
    """Create and manage only the container named and labelled by one profile."""

    def __init__(
        self,
        deployment: LearnerDeployment,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        docker: str | None = None,
        allow_rootless_container_user: bool = False,
        host_uid: int | None = None,
        host_gid: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        socket_factory: Callable[..., socket.socket] = socket.socket,
        inventory_reader: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        self.deployment = deployment
        self.runner = runner
        self.docker = docker or shutil.which("docker") or "docker"
        if type(allow_rootless_container_user) is not bool:
            raise ValueError("allow_rootless_container_user must be bool")
        self.allow_rootless_container_user = allow_rootless_container_user
        self.host_uid = os.getuid() if host_uid is None else host_uid
        self.host_gid = os.getgid() if host_gid is None else host_gid
        if (isinstance(self.host_uid, bool) or not isinstance(self.host_uid, int) or self.host_uid < 0
                or isinstance(self.host_gid, bool) or not isinstance(self.host_gid, int) or self.host_gid < 0):
            raise ValueError("Learner host UID/GID mapping must be numeric")
        if allow_rootless_container_user:
            if (deployment.container_user != "0:0" or self.host_uid <= 0
                    or (self.host_uid, self.host_gid) != (os.getuid(), os.getgid())):
                raise LearnerManagerError(
                    "rootless Learner manager requires container 0:0 mapped to the invoking account"
                )
        self.monotonic = monotonic
        self.sleep = sleep
        self.socket_factory = socket_factory
        self.inventory_reader = inventory_reader or self._gpu_inventory
        self.profile_sha256 = hashlib.sha256(
            json.dumps(deployment.model_dump(mode="json"), sort_keys=True,
                       separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.state_dir = deployment.state.host_path
        self.ownership_dir = deployment.host_control_path
        self.ownership_path = self.ownership_dir / "container.json"

    def _assert_control_dir_disjoint(self) -> None:
        control = self.ownership_dir.resolve(strict=False)
        sources = [
            self.deployment.profile_path,
            self.deployment.runtime_config.host_path,
            self.deployment.model.host_path,
            self.deployment.dataset.host_path,
            self.deployment.state.host_path,
            *(item.host_path for item in self.deployment.resources),
            *(item.host_path for item in self.deployment.credentials.values()),
        ]
        for source in sources:
            resolved = Path(source).resolve(strict=False)
            if (control == resolved or control in resolved.parents or resolved in control.parents):
                raise LearnerManagerError(
                    "host ownership metadata directory must be outside every container bind mount"
                )

    def _run(self, argv: Sequence[str], *, timeout: float, label: str,
             capture_output: bool = True) -> subprocess.CompletedProcess[str]:
        if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
            raise ValueError("command arguments must be non-empty single-line strings")
        try:
            result = self.runner(
                list(argv), check=False, capture_output=capture_output, text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LearnerManagerError(f"could not run {Path(argv[0]).name} for {label}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > 400:
                detail = detail[:400]
            suffix = f": {detail}" if detail else ""
            raise LearnerManagerError(f"{Path(argv[0]).name} failed during {label}{suffix}")
        return result

    def _gpu_inventory(self) -> Sequence[str]:
        try:
            result = self.runner(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                check=False, capture_output=True, text=True, timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LearnerManagerError("NVIDIA GPU inventory is unavailable; refusing GPU fallback") from exc
        if result.returncode != 0:
            raise LearnerManagerError("NVIDIA GPU inventory failed; refusing GPU fallback")
        return tuple(line.strip() for line in (result.stdout or "").splitlines() if line.strip())

    def _image_id(self) -> str:
        result = self._run(
            [self.docker, "image", "inspect", "--format", "{{.Id}}", self.deployment.image_id],
            timeout=20.0, label="Learner image verification",
        )
        value = (result.stdout or "").strip()
        if value != self.deployment.image_id:
            raise LearnerManagerError("local Learner image does not match the profile's exact image_id")
        return value

    def _check_port_available(self, *, allow_owned_container: bool) -> bool:
        if allow_owned_container:
            current = self._inspect_container()
            if current is not None and current.get("State", {}).get("Running") is True:
                return True
        family = socket.AF_INET6 if ":" in self.deployment.publish.host_address else socket.AF_INET
        try:
            endpoint = self.socket_factory(family, socket.SOCK_STREAM)
        except OSError as exc:
            raise LearnerManagerError("could not create a host port probe socket") from exc
        try:
            endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            endpoint.bind((self.deployment.publish.host_address, self.deployment.publish.host_port))
            return True
        except OSError:
            raise LearnerManagerError("Learner publish port is already in use") from None
        finally:
            endpoint.close()

    def check(self, *, allow_owned_container: bool = True) -> dict[str, Any]:
        check_options: dict[str, Any] = {"require_assets": True}
        if self.allow_rootless_container_user:
            check_options.update({
                "allow_rootless_container_user": True,
                "host_uid": self.host_uid,
                "host_gid": self.host_gid,
            })
        check_learner_profile(self.deployment, **check_options)
        self._assert_control_dir_disjoint()
        self._image_id()
        visible = tuple(self.inventory_reader())
        if not self.deployment.gpu_uuids or not set(self.deployment.gpu_uuids) <= set(visible):
            raise LearnerManagerError("selected Learner GPU UUIDs are not all visible; no fallback is allowed")
        self._check_port_available(allow_owned_container=allow_owned_container)
        return {
            "status": "checked",
            "deployment_id": self.deployment.deployment_id,
            "image_id": self.deployment.image_id,
            "gpu_uuids": list(self.deployment.gpu_uuids),
            "host_port": self.deployment.publish.host_port,
        }

    def _inspect_container(self, reference: str | None = None) -> dict[str, Any] | None:
        result = self.runner(
            [self.docker, "container", "inspect", "--format", "{{json .}}",
             reference or self.deployment.container_name],
            check=False, capture_output=True, text=True, timeout=15.0,
        )
        if result.returncode != 0:
            text = (result.stderr or result.stdout or "").lower()
            if "no such object" in text or "no such container" in text or "not found" in text:
                return None
            raise LearnerManagerError("Docker could not inspect the profile Learner container")
        try:
            value = json.loads(result.stdout or "")
        except (json.JSONDecodeError, TypeError):
            raise LearnerManagerError("Docker returned malformed Learner container metadata") from None
        if not isinstance(value, dict):
            raise LearnerManagerError("Docker returned invalid Learner container metadata")
        return value

    def _rollback_container(self, container_id: str) -> None:
        """Stop/remove only the exact container ID created or started by this call."""
        if not _CONTAINER_ID.fullmatch(container_id):
            return
        try:
            container = self._inspect_container(container_id)
        except LearnerManagerError:
            return
        if container is None or self._container_id(container) != container_id:
            return
        labels = self._labels(container)
        if (labels.get(_OWNER_LABEL) != _OWNER_VALUE
                or labels.get(_DEPLOYMENT_LABEL) != self.deployment.deployment_id
                or labels.get(_PROFILE_LABEL) != self.profile_sha256
                or labels.get(_IMAGE_LABEL) != self.deployment.image_id
                or container.get("Image") != self.deployment.image_id):
            return
        state = container.get("State", {})
        if state.get("Running") is True:
            self._run([self.docker, "container", "stop", "--time", "60", container_id],
                      timeout=75.0, label="Learner startup rollback")
        self._run([self.docker, "container", "rm", container_id],
                  timeout=30.0, label="Learner startup rollback")
        try:
            owner = json.loads(self.ownership_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            owner = None
        if isinstance(owner, dict) and owner.get("container_id") == container_id:
            self.ownership_path.unlink(missing_ok=True)

    def _create_container(self) -> tuple[str, dict[str, Any]]:
        result = self._run(self._container_args(), timeout=120.0,
                           label="Learner container creation")
        identifier = (result.stdout or "").strip()
        if not _CONTAINER_ID.fullmatch(identifier):
            candidate = self._inspect_container()
            if candidate is not None:
                try:
                    if self._owned(candidate):
                        identifier = self._container_id(candidate)
                except LearnerManagerError:
                    pass
        if not _CONTAINER_ID.fullmatch(identifier):
            raise LearnerManagerError("Docker returned an invalid Learner container ID")
        try:
            container = self._inspect_container(identifier)
            if container is None or self._container_id(container) != identifier:
                raise LearnerManagerError("created Learner container could not be inspected by its exact ID")
            self._owned(container)
            self._verify_mounts(container)
            return identifier, container
        except BaseException:
            self._rollback_container(identifier)
            raise

    @staticmethod
    def _labels(container: Mapping[str, Any]) -> Mapping[str, str]:
        config = container.get("Config")
        labels = config.get("Labels") if isinstance(config, dict) else None
        if not isinstance(labels, dict) or any(not isinstance(key, str) for key in labels):
            return {}
        return labels

    @staticmethod
    def _container_id(container: Mapping[str, Any]) -> str:
        identifier = container.get("Id")
        if not isinstance(identifier, str) or not _CONTAINER_ID.fullmatch(identifier):
            raise LearnerManagerError("Docker returned an invalid Learner container ID")
        return identifier

    def _owned(self, container: Mapping[str, Any], *, check_profile: bool = True) -> str:
        container_id = self._owned_identity(container)
        labels = self._labels(container)
        if check_profile:
            if labels.get(_PROFILE_LABEL) != self.profile_sha256:
                raise LearnerManagerError("owned Learner container uses a different profile")
            if labels.get(_IMAGE_LABEL) != self.deployment.image_id \
                    or container.get("Image") != self.deployment.image_id:
                raise LearnerManagerError("owned Learner container uses a different image ID")
        return container_id

    def _owned_identity(self, container: Mapping[str, Any]) -> str:
        labels = self._labels(container)
        if (labels.get(_OWNER_LABEL) != _OWNER_VALUE
                or labels.get(_DEPLOYMENT_LABEL) != self.deployment.deployment_id):
            raise LearnerManagerError(
                "container name is reserved by an unrelated Learner deployment; no changes were made"
            )
        label_image = labels.get(_IMAGE_LABEL)
        if not isinstance(label_image, str) or not _IMAGE_ID.fullmatch(label_image) \
                or container.get("Image") != label_image:
            raise LearnerManagerError("owned Learner container image labels are inconsistent")
        return self._container_id(container)

    def _expected_mounts(self) -> dict[str, tuple[Path, bool]]:
        expected = {
            PROFILE_CONTAINER_PATH: (self.deployment.profile_path, True),
            self.deployment.runtime_config.container_path: (self.deployment.runtime_config.host_path, True),
            self.deployment.model.container_path: (self.deployment.model.host_path, True),
            self.deployment.dataset.container_path: (self.deployment.dataset.host_path, True),
            self.deployment.state.container_path: (self.deployment.state.host_path, False),
        }
        expected.update({mount.container_path: (mount.host_path, True)
                         for mount in self.deployment.resources})
        expected.update({mount.container_path: (mount.host_path, True)
                         for mount in self.deployment.credentials.values()})
        return {destination: (Path(source).resolve(strict=True), read_only)
                for destination, (source, read_only) in expected.items()}

    def _verify_mounts(self, container: Mapping[str, Any]) -> None:
        mounts = container.get("Mounts")
        if not isinstance(mounts, list):
            raise LearnerManagerError("Docker omitted Learner container mounts")
        observed: dict[str, tuple[Path, bool]] = {}
        for mount in mounts:
            if not isinstance(mount, dict) or mount.get("Type") != "bind":
                raise LearnerManagerError("Learner container has an undeclared non-bind mount")
            source, destination = mount.get("Source"), mount.get("Destination")
            if not isinstance(source, str) or not isinstance(destination, str):
                raise LearnerManagerError("Learner container mount metadata is invalid")
            observed[destination] = (Path(source).resolve(strict=False), mount.get("RW") is True)
        if observed != self._expected_mounts():
            raise LearnerManagerError("Learner container mounts differ from the strict profile")

    def _prepare_state(self) -> None:
        path = self.deployment.state.host_path
        try:
            path.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise LearnerManagerError("could not create the Learner state directory") from exc
        if path.is_symlink() or not path.is_dir():
            raise LearnerManagerError("Learner persistent state path must be a regular directory")
        uid, gid = (int(part) for part in self.deployment.container_user.split(":"))
        if self.allow_rootless_container_user:
            uid, gid = self.host_uid, self.host_gid
        metadata = path.stat()
        if metadata.st_uid != uid or metadata.st_gid != gid:
            raise LearnerManagerError("Learner state directory owner must match container_user")
        path.chmod(0o700)

    def _prepare_ownership_dir(self) -> None:
        self._assert_control_dir_disjoint()
        control_root = self.ownership_dir.parent
        control_root.mkdir(mode=0o700, exist_ok=True)
        if (control_root.is_symlink() or control_root.stat().st_uid != os.getuid()
                or stat.S_IMODE(control_root.stat().st_mode) & 0o077):
            raise LearnerManagerError("host-control root must be caller-owned and private")
        self.ownership_dir.mkdir(mode=0o700, exist_ok=True)
        if (self.ownership_dir.is_symlink() or self.ownership_dir.stat().st_uid != os.getuid()
                or stat.S_IMODE(self.ownership_dir.stat().st_mode) & 0o077):
            raise LearnerManagerError("Learner ownership metadata directory must be caller-owned and private")

    def _container_args(self) -> list[str]:
        d = self.deployment
        # Recheck recursive bind-source contents immediately before asking
        # Docker to create the container; read-only does not make sockets or
        # symlink escapes safe.
        check_options: dict[str, Any] = {"require_assets": True}
        if self.allow_rootless_container_user:
            check_options.update({
                "allow_rootless_container_user": True,
                "host_uid": self.host_uid,
                "host_gid": self.host_gid,
            })
        check_learner_profile(d, **check_options)
        args = [
            self.docker, "container", "create",
            "--name", d.container_name,
            "--label", f"{_OWNER_LABEL}={_OWNER_VALUE}",
            "--label", f"{_DEPLOYMENT_LABEL}={d.deployment_id}",
            "--label", f"{_PROFILE_LABEL}={self.profile_sha256}",
            "--label", f"{_IMAGE_LABEL}={d.image_id}",
            "--user", d.container_user,
            "--workdir", d.container_workdir,
            "--gpus", f"device={','.join(d.gpu_uuids)}",
            "--publish", self._docker_publish_spec(),
            "--read-only",
            "--tmpfs", "/tmp:rw,nosuid,nodev,mode=1777,size=4g",
            "--security-opt", "no-new-privileges:true",
            "--cap-drop", "ALL",
            "--stop-timeout", "60",
        ]
        if self.allow_rootless_container_user:
            args.extend(("--env", "QWEN_LEARNER_ROOTLESS_CONTAINER_USER=1"))
        mounts = [
            (d.profile_path, PROFILE_CONTAINER_PATH, True),
            (d.runtime_config.host_path, d.runtime_config.container_path, True),
            (d.model.host_path, d.model.container_path, True),
            (d.dataset.host_path, d.dataset.container_path, True),
            (d.state.host_path, d.state.container_path, False),
            *((item.host_path, item.container_path, True) for item in d.resources),
            *((item.host_path, item.container_path, True) for item in d.credentials.values()),
        ]
        for source, destination, read_only in mounts:
            if any(char in str(source) for char in ",:\n\r") or any(char in destination for char in ",:\n\r"):
                raise LearnerManagerError("Learner mount paths cannot contain Docker mount delimiters")
            mount_text = f"type=bind,src={Path(source).resolve(strict=True)},dst={destination}"
            if read_only:
                mount_text += ",readonly"
            args.extend(("--mount", mount_text))
        args.extend((d.image_id, "--deployment", PROFILE_CONTAINER_PATH))
        return args

    def _docker_publish_spec(self) -> str:
        address = self.deployment.publish.host_address
        if ipaddress.ip_address(address).version == 6:
            address = f"[{address}]"
        return f"{address}:{self.deployment.publish.host_port}:{self.deployment.publish.container_port}"

    def _published_health_url(self) -> str:
        address = self.deployment.publish.host_address
        if ipaddress.ip_address(address).version == 6:
            address = f"[{address}]"
        return f"http://{address}:{self.deployment.publish.host_port}/health"

    def _read_arc_token(self) -> str:
        credential = self.deployment.credentials.get("ARC_ENV_TOKEN")
        if credential is None:
            raise LearnerManagerError("ARC_ENV_TOKEN credential file is required for host-published health")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(credential.host_path, flags)
        except OSError:
            raise LearnerManagerError("ARC_ENV_TOKEN credential file is unavailable") from None
        try:
            info = os.fstat(descriptor)
            expected_uid = (self.host_uid if self.allow_rootless_container_user
                            else int(self.deployment.container_user.split(":", 1)[0]))
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != expected_uid
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise LearnerManagerError("ARC_ENV_TOKEN must be a mode-0600 regular file owned by container_user")
            content = bytearray()
            while len(content) <= _SECRET_MAX_BYTES:
                chunk = os.read(descriptor, min(4096, _SECRET_MAX_BYTES + 1 - len(content)))
                if not chunk:
                    break
                content.extend(chunk)
            if len(content) > _SECRET_MAX_BYTES:
                raise LearnerManagerError("ARC_ENV_TOKEN credential file is too large")
            try:
                token = content.decode("utf-8").strip()
            except UnicodeError:
                raise LearnerManagerError("ARC_ENV_TOKEN credential file is not UTF-8") from None
            if not token or any(ord(char) < 32 or ord(char) == 127 for char in token):
                raise LearnerManagerError("ARC_ENV_TOKEN credential file is empty or invalid")
            return token
        finally:
            os.close(descriptor)

    def _wait_published_ready(self, timeout: float) -> dict[str, str]:
        """Require authenticated health through the host-published Docker port."""
        if timeout <= 0:
            raise ValueError("published Learner readiness timeout must be positive")
        token = self._read_arc_token()
        url = self._published_health_url()
        deadline = self.monotonic() + timeout
        last_error = "no host-published HTTP response"
        try:
            with httpx.Client(trust_env=False, timeout=min(5.0, timeout)) as client:
                while self.monotonic() < deadline:
                    try:
                        response = client.get(url, headers={"Authorization": f"Bearer {token}"})
                        if response.status_code == 200:
                            payload = response.json()
                            if (isinstance(payload, dict) and payload.get("ready") is True
                                    and payload.get("role") == "learner"):
                                return {
                                    "status": "published_ready",
                                    "host_address": self.deployment.publish.host_address,
                                    "host_port": str(self.deployment.publish.host_port),
                                }
                            last_error = "host-published health response did not identify a READY Learner"
                        else:
                            last_error = f"host-published health returned HTTP {response.status_code}"
                    except (httpx.HTTPError, ValueError):
                        last_error = "host-published health request failed"
                    self.sleep(min(0.5, max(0.0, deadline - self.monotonic())))
        except httpx.HTTPError:
            last_error = "host-published health client could not be created"
        raise LearnerManagerError(
            f"Learner did not become READY through its host-published port: {last_error}"
        )

    def _write_ownership(self, container: Mapping[str, Any]) -> None:
        value = {
            "schema_version": "qwen-rl-learner-container-owner/v1",
            "container_id": self._container_id(container),
            "container_name": self.deployment.container_name,
            "deployment_id": self.deployment.deployment_id,
            "profile_sha256": self.profile_sha256,
            "image_id": self.deployment.image_id,
            "updated_at": time.time(),
        }
        temporary = self.ownership_path.with_name(f".{self.ownership_path.name}.{os.getpid()}.tmp")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(temporary, flags, 0o600)
        except FileExistsError:
            temporary.unlink(missing_ok=True)
            fd = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.ownership_path)
            os.chmod(self.ownership_path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)

    def _operation(self, operation: str, *arguments: str,
                   timeout: float = 30.0, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
        container = self._inspect_container()
        if container is None:
            raise LearnerManagerError("owned Learner container is not present")
        container_id = self._owned(container)
        self._verify_mounts(container)
        argv = [self.docker, "exec", container_id, CONTAINER_PYTHON, "-B", "-m", _INTERNAL_MODULE,
                "--deployment", PROFILE_CONTAINER_PATH, "--operation", operation, *arguments]
        return self._run(argv, timeout=timeout, label=f"Learner {operation}",
                         capture_output=capture_output)

    def _wait_ready(self, timeout: float) -> dict[str, Any]:
        deadline = self.monotonic() + timeout
        last_error = "Learner runtime has not reported READY"
        while self.monotonic() < deadline:
            container = self._inspect_container()
            if container is None:
                raise LearnerManagerError("Learner container disappeared during startup")
            container_id = self._owned(container)
            state = container.get("State", {})
            if state.get("Running") is not True:
                raise LearnerManagerError("Learner container exited before READY")
            try:
                result = self._run(
                    [self.docker, "exec", container_id, CONTAINER_PYTHON, "-B", "-m", _INTERNAL_MODULE,
                     "--deployment", PROFILE_CONTAINER_PATH, "--operation", "status"],
                    timeout=min(10.0, max(0.1, deadline - self.monotonic())),
                    label="Learner readiness query",
                )
                report = json.loads(result.stdout or "")
                if isinstance(report, dict) and report.get("reuse_ready") is True:
                    return report
                last_error = "Learner runtime has not reported READY"
            except (LearnerManagerError, json.JSONDecodeError):
                last_error = "Learner status probe failed while waiting for READY"
            self.sleep(min(1.0, max(0.0, deadline - self.monotonic())))
        raise LearnerManagerError(f"Learner did not become READY before timeout: {last_error}")

    def _activate(self, *, resume: bool, with_teacher: bool) -> dict[str, Any] | None:
        if not resume and not with_teacher:
            return None
        arguments = ["--resume"] if resume else []
        if with_teacher:
            arguments.append("--with-teacher")
        result = self._operation("activate", *arguments, timeout=60.0)
        try:
            report = json.loads(result.stdout or "")
        except json.JSONDecodeError:
            raise LearnerManagerError("Learner activation returned malformed status") from None
        if not isinstance(report, dict):
            raise LearnerManagerError("Learner activation returned invalid status")
        return report

    def start(self, *, timeout: float = 900.0, resume: bool = False,
              with_teacher: bool = False) -> dict[str, Any]:
        if type(resume) is not bool or type(with_teacher) is not bool:
            raise ValueError("resume and with_teacher must be bool")
        if timeout <= 0:
            raise ValueError("Learner start timeout must be positive")
        self.check(allow_owned_container=True)
        self._prepare_state()
        self._prepare_ownership_dir()
        rollback_id: str | None = None
        created = False
        try:
            existing = self._inspect_container()
            if existing is None:
                rollback_id, existing = self._create_container()
                created = True
            else:
                state = existing.get("State", {})
                if state.get("Running") is True:
                    self._owned(existing)
                    self._verify_mounts(existing)
                else:
                    try:
                        self._owned(existing)
                        self._verify_mounts(existing)
                    except LearnerManagerError:
                        old_id = self._owned_identity(existing)
                        self._run([self.docker, "container", "rm", old_id], timeout=30.0,
                                  label="stopped Learner profile replacement")
                        rollback_id, existing = self._create_container()
                        created = True
                    else:
                        rollback_id = self._container_id(existing)
            container_id = self._container_id(existing)
            state = existing.get("State", {})
            if state.get("Running") is not True:
                self._run([self.docker, "container", "start", container_id], timeout=60.0,
                          label="Learner container start")
            refreshed = self._inspect_container(container_id)
            if refreshed is None:
                raise LearnerManagerError("Learner container disappeared after start")
            self._owned(refreshed)
            self._verify_mounts(refreshed)
            self._write_ownership(refreshed)
            startup_deadline = self.monotonic() + timeout
            status = self._wait_ready(timeout)
            published = self._wait_published_ready(
                max(0.1, startup_deadline - self.monotonic()),
            )
            activation = self._activate(resume=resume, with_teacher=with_teacher)
        except BaseException:
            if rollback_id is not None:
                try:
                    self._rollback_container(rollback_id)
                except LearnerManagerError:
                    pass
            raise
        return {
            "status": activation.get("status") if activation else status.get("status", "ready"),
            "deployment_id": self.deployment.deployment_id,
            "container_id": self._container_id(refreshed),
            "created": created,
            "runtime": status,
            "published_health": published,
            "activation": activation,
        }

    def status(self) -> dict[str, Any]:
        container = self._inspect_container()
        if container is None:
            return {"status": "stopped", "deployment_id": self.deployment.deployment_id,
                    "container_id": None, "runtime": None}
        container_id = self._owned(container, check_profile=False)
        state = container.get("State", {})
        labels = self._labels(container)
        profile_matches = labels.get(_PROFILE_LABEL) == self.profile_sha256
        if state.get("Running") is not True:
            return {"status": "exited", "deployment_id": self.deployment.deployment_id,
                    "container_id": container_id, "exit_code": state.get("ExitCode"),
                    "profile_matches": profile_matches, "runtime": None}
        if not profile_matches:
            return {"status": "profile_changed", "deployment_id": self.deployment.deployment_id,
                    "container_id": container_id, "profile_matches": False, "runtime": None}
        self._verify_mounts(container)
        result = self._operation("status")
        try:
            runtime = json.loads(result.stdout or "")
        except json.JSONDecodeError:
            raise LearnerManagerError("Learner runtime status returned malformed JSON") from None
        if not isinstance(runtime, dict):
            raise LearnerManagerError("Learner runtime status must be a JSON object")
        return {"status": runtime.get("status", "unknown"), "deployment_id": self.deployment.deployment_id,
                "container_id": container_id, "runtime": runtime}

    def run(self, *, steps: int, task_ids: Sequence[str] | None = None,
            timeout: float = 60.0) -> dict[str, Any]:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        arguments = ["--steps", str(steps)]
        if task_ids is not None:
            if (not task_ids or any(not isinstance(item, str) or not item for item in task_ids)
                    or len(set(task_ids)) != len(task_ids)):
                raise ValueError("task_ids must be distinct non-empty strings")
            arguments.extend(("--task-ids", *task_ids))
        result = self._operation("run", *arguments, timeout=timeout)
        try:
            report = json.loads(result.stdout or "")
        except json.JSONDecodeError:
            raise LearnerManagerError("Learner run returned malformed status") from None
        if not isinstance(report, dict):
            raise LearnerManagerError("Learner run returned invalid status")
        return report

    def watch(self, *, tail: int = 100, follow: bool = True,
              timeout: float = 3600.0) -> None:
        if isinstance(tail, bool) or not isinstance(tail, int) or tail < 0:
            raise ValueError("tail must be a nonnegative integer")
        container = self._inspect_container()
        if container is None:
            raise LearnerManagerError("Learner container is not running")
        container_id = self._owned(container)
        self._verify_mounts(container)
        argv = [self.docker, "container", "logs", "--tail", str(tail)]
        if follow:
            argv.append("--follow")
        argv.append(container_id)
        self._run(argv, timeout=timeout, label="Learner logs", capture_output=False)

    def stop(self, *, timeout: float = 60.0) -> dict[str, Any]:
        if timeout <= 0:
            raise ValueError("Learner stop timeout must be positive")
        container = self._inspect_container()
        if container is None:
            return {"status": "stopped", "deployment_id": self.deployment.deployment_id,
                    "container_id": None}
        container_id = self._owned(container, check_profile=False)
        if self._labels(container).get(_PROFILE_LABEL) == self.profile_sha256:
            self._verify_mounts(container)
        state = container.get("State", {})
        if state.get("Running") is True:
            self._run([self.docker, "container", "stop", "--time", "60", container_id],
                      timeout=timeout + 10.0, label="Learner container shutdown")
        self._run([self.docker, "container", "rm", container_id],
                  timeout=30.0, label="Learner owned container removal")
        self.ownership_path.unlink(missing_ok=True)
        return {"status": "stopped", "deployment_id": self.deployment.deployment_id,
                "container_id": container_id}
