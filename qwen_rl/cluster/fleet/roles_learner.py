"""User-owned Fleet installation and lifecycle for a trusted Learner bundle."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import httpx

from ..identity import atomic_write
from .contracts import LearnerDeployment as LearnerDeploymentSpec
from .contracts import UserAccount
from .host import HostInventory, current_user_account
from .paths import AccountPathError, UserPaths, ensure_account_path
from .roles import (
    DockerAccess,
    FleetRoleError,
    RoleResult,
    StopResult,
    ensure_user_dir,
    inspect_docker_access,
    operation_dir,
    probe_docker_uid_mapping,
    run_owned,
    verify_existing,
)

MAX_LEARNER_BUNDLE_BYTES = 8 * 1024**4
_DEPLOYMENT_ID = re.compile(r"^[A-Za-z0-9_-]{16,100}$")


def _deployment_root(paths: UserPaths, deployment_id: str) -> Path:
    if _DEPLOYMENT_ID.fullmatch(deployment_id) is None:
        raise FleetRoleError("Learner deployment identity is invalid")
    return paths.roles_dir / "learner" / deployment_id


def _prepare_account_path(path: Path, account: UserAccount) -> Path:
    """Create only real directories below the passwd HOME, rejecting symlink parents."""
    try:
        return ensure_account_path(path, account=account, mode=0o700)
    except AccountPathError as exc:
        raise FleetRoleError("Learner path cannot be safely used below the registered HOME") from exc


def _bundle_marker(root: Path, *, deployment_id: str, bundle_sha256: str,
                   account: UserAccount) -> dict[str, Any] | None:
    from qwen_rl.deployment.learner.bootstrap import INSTALL_MARKER

    path = root / INSTALL_MARKER
    try:
        info = path.lstat()
        if (path.is_symlink() or not stat.S_ISREG(info.st_mode)
                or info.st_uid != account.uid or info.st_gid != account.gid
                or stat.S_IMODE(info.st_mode) != 0o600):
            return None
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (not isinstance(marker, dict) or marker.get("deployment_id") != deployment_id
            or marker.get("bundle_sha256") != bundle_sha256):
        return None
    return marker


def _read_private_bundle(
    deployment: LearnerDeploymentSpec,
    destination: Path,
    *,
    ca_bundle: Path,
    transport: httpx.BaseTransport | None,
) -> Path:
    """Download the assigned private artifact without redirects or bearer persistence."""
    if verify_existing(destination, deployment.private_bundle.sha256,
                       maximum_bytes=MAX_LEARNER_BUNDLE_BYTES):
        return destination
    if destination.exists() or destination.is_symlink():
        raise FleetRoleError("Learner bundle destination already exists")
    parsed = urlsplit(deployment.private_bundle.url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise FleetRoleError("Learner private bundle URL is not HTTPS")
    temporary: Path | None = None
    digest = hashlib.sha256()
    total = 0
    token = deployment.bundle_bearer.get_secret_value()
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{destination.name}.download-", dir=destination.parent)
        temporary = Path(raw_path)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            with httpx.Client(verify=str(ca_bundle), transport=transport, timeout=120,
                              follow_redirects=False, trust_env=False) as client:
                with client.stream(
                    "GET", deployment.private_bundle.url,
                    headers={"Accept": "application/vnd.qwen-rl.learner-bundle",
                             "Authorization": f"Bearer {token}"},
                ) as response:
                    if (response.status_code != 200 or response.url != httpx.URL(deployment.private_bundle.url)
                            or response.url.scheme != "https"
                            or response.url.host != parsed.hostname):
                        raise FleetRoleError("assigned private Learner bundle returned an unexpected origin or status")
                    length = response.headers.get("Content-Length")
                    if length is not None:
                        if not length.isdigit() or int(length) > MAX_LEARNER_BUNDLE_BYTES:
                            raise FleetRoleError("private Learner bundle exceeds its compressed size limit")
                    for chunk in response.iter_bytes(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_LEARNER_BUNDLE_BYTES:
                            raise FleetRoleError("private Learner bundle exceeds its compressed size limit")
                        digest.update(chunk)
                        output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total == 0 or not hmac.compare_digest(digest.hexdigest(), deployment.private_bundle.sha256):
            raise FleetRoleError("private Learner bundle SHA-256 did not match its assignment")
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        return destination
    except FleetRoleError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    except (httpx.HTTPError, OSError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise FleetRoleError("assigned private Learner bundle download failed") from exc


def _bundle_grant_sha256(deployment: LearnerDeploymentSpec) -> str:
    token = deployment.bundle_bearer.get_secret_value().encode("ascii")
    return hashlib.sha256(token).hexdigest()


def _pin_bundle_grant(operation: Path, deployment: LearnerDeploymentSpec) -> str:
    """Bind one generation to its grant without persisting the bearer itself."""
    fingerprint = _bundle_grant_sha256(deployment)
    path = Path(operation) / "bundle-grant.sha256"
    if path.exists() or path.is_symlink():
        descriptor = None
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            info = os.fstat(descriptor)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_gid != os.getegid() or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_size > 128):
                raise FleetRoleError("Learner assignment grant binding has unsafe ownership or mode")
            saved = os.read(descriptor, 128).decode("ascii").strip()
        except FleetRoleError:
            raise
        except (OSError, UnicodeError) as exc:
            raise FleetRoleError("Learner assignment grant binding cannot be read") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if not hmac.compare_digest(saved, fingerprint):
            raise FleetRoleError("Learner bundle grant changed inside one assignment generation")
        return fingerprint
    atomic_write(path, (fingerprint + "\n").encode("ascii"), mode=0o600)
    return fingerprint


class LearnerRoleAdapter:
    role = "learner"

    def __init__(
        self,
        *,
        state_dir: Path,
        ca_bundle: Path,
        python_executable: Path,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        artifact_transport: httpx.BaseTransport | None = None,
        account: UserAccount | None = None,
        paths: UserPaths | None = None,
    ) -> None:
        self.account = account or current_user_account()
        actual = current_user_account()
        if actual != self.account:
            raise FleetRoleError("Learner adapter account differs from the effective passwd account")
        expected_paths = UserPaths.from_account(self.account)
        self.paths = paths or expected_paths
        if self.paths != expected_paths:
            raise FleetRoleError("Learner adapter paths differ from the registered account HOME")
        self.state_dir = Path(state_dir)
        if self.state_dir != self.paths.state_dir:
            raise FleetRoleError("Learner operation state must use the registered account's Fleet state path")
        self.ca_bundle = Path(ca_bundle)
        self.python_executable = Path(python_executable)
        self.install_root = self.paths.roles_dir / "learner"
        self.systemd_dir = self.paths.systemd_user_dir
        self.runner = runner
        self.artifact_transport = artifact_transport

    def _assert_account(self, account: UserAccount | None = None) -> UserAccount:
        actual = current_user_account()
        expected = account or self.account
        if actual != self.account or expected != self.account:
            raise FleetRoleError("Learner role is not running as the registered account")
        return actual

    def _prepare_user_paths(self, account: UserAccount) -> None:
        for path in (self.paths.root, self.paths.state_dir, self.paths.roles_dir, self.install_root):
            _prepare_account_path(path, account)
            ensure_user_dir(path, account=account)

    def _unit(self, deployment_id: str) -> str:
        if _DEPLOYMENT_ID.fullmatch(deployment_id) is None:
            raise FleetRoleError("Learner deployment identity is invalid")
        return f"qwen-rl-learner@{deployment_id}.service"

    def _unit_path(self, deployment_id: str) -> Path:
        return self.systemd_dir / self._unit(deployment_id)

    def _unit_active(self, deployment_id: str) -> bool | None:
        systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
        env = self._service_environment(None, None)
        try:
            result = self.runner(
                [systemctl, "--user", "is-active", "--quiet", self._unit(deployment_id)],
                check=False, capture_output=True, text=True, timeout=20, env=env,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode == 0:
            return True
        if result.returncode in {3, 4}:
            return False
        return None

    def _service_environment(self, docker_context: str | None, docker_config: str | None) -> dict[str, str]:
        environment = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": self.account.home,
            "LANG": "C.UTF-8",
            "XDG_RUNTIME_DIR": f"/run/user/{self.account.uid}",
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{self.account.uid}/bus",
        }
        if docker_context is not None:
            environment["DOCKER_CONTEXT"] = docker_context
        if docker_config is not None:
            environment["DOCKER_CONFIG"] = docker_config
        return environment

    @staticmethod
    def _docker_config_path() -> str | None:
        configured = os.environ.get("DOCKER_CONFIG")
        return str(Path(configured).expanduser().absolute()) if configured is not None else None

    def _docker_runner(self, access: DockerAccess, docker_config: str | None = None):
        environment = self._service_environment(access.context, docker_config)
        if access.host_override:
            environment.pop("DOCKER_CONTEXT", None)
            environment["DOCKER_HOST"] = access.endpoint
        else:
            environment.pop("DOCKER_HOST", None)
            environment["DOCKER_CONTEXT"] = access.context

        def run(argv, **kwargs):
            merged = {**environment, **(kwargs.get("env") or {})}
            merged.pop("DOCKER_HOST", None)
            merged.pop("DOCKER_CONTEXT", None)
            if access.host_override:
                merged["DOCKER_HOST"] = access.endpoint
            else:
                merged["DOCKER_CONTEXT"] = access.context
            kwargs["env"] = merged
            return self.runner(argv, **kwargs)

        return run

    def _docker_access(self, *, expected: Mapping[str, Any] | None = None) -> DockerAccess:
        return inspect_docker_access(runner=self.runner, expected=expected)

    def _container_exists(self, access: DockerAccess, deployment_id: str) -> bool | None:
        result = self.runner(
            access.command("ps", "--all", "--quiet", "--filter",
                           f"label=io.qwen-rl.deployment={deployment_id}"),
            check=False, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        return bool((result.stdout or "").strip())

    def _owned_profile(
        self,
        root: Path,
        deployment_id: str,
        bundle_sha256: str,
        *,
        allow_rootless_container_user: bool = False,
    ) -> tuple[dict[str, Any], Any] | None:
        marker = _bundle_marker(root, deployment_id=deployment_id,
                                 bundle_sha256=bundle_sha256, account=self.account)
        if marker is None:
            return None
        profile_path = root / "learner-deployment.json"
        try:
            from qwen_rl.deployment.learner.config import load_learner_deployment

            deployment = load_learner_deployment(
                profile_path,
                allow_gpu_autodetect=False,
                allow_rootless_container_user=allow_rootless_container_user,
            )
        except Exception:
            return None
        if deployment.deployment_id != deployment_id or marker.get("image_id") != deployment.image_id:
            return None
        return marker, deployment

    def prepare_install(
        self,
        deployment: LearnerDeploymentSpec,
        *,
        generation: int,
        owned_receipt: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if not isinstance(deployment, LearnerDeploymentSpec):
            raise FleetRoleError("Learner adapter received another role")
        account = self._assert_account()
        self._prepare_user_paths(account)
        root = _deployment_root(self.paths, deployment.deployment_id)
        access_expected = None
        if owned_receipt is not None:
            if (owned_receipt.get("role") != self.role
                    or owned_receipt.get("deployment_id") != deployment.deployment_id
                    or owned_receipt.get("account_uid") != account.uid
                    or owned_receipt.get("account_gid") != account.gid
                    or owned_receipt.get("install_root") != str(root)
                    or owned_receipt.get("bundle_sha256") not in {None, deployment.private_bundle.sha256}):
                raise FleetRoleError("existing Learner receipt does not prove ownership of this user install")
            if (owned_receipt.get("generation") == generation
                    and owned_receipt.get("bundle_grant_sha256") not in {
                        None, _bundle_grant_sha256(deployment)}):
                raise FleetRoleError("Learner private bundle grant changed inside one assignment generation")
            access_expected = owned_receipt.get("docker_access")
        access = self._docker_access(expected=access_expected)
        if root.exists() or root.is_symlink():
            if owned_receipt is None or self._owned_profile(
                    root,
                    deployment.deployment_id,
                    deployment.private_bundle.sha256,
                    allow_rootless_container_user=access.rootless,
            ) is None:
                raise FleetRoleError("unowned or differently pinned Learner install already exists")
            return {
                "role": self.role,
                "generation": generation,
                "deployment_id": deployment.deployment_id,
                "bundle_sha256": deployment.private_bundle.sha256,
                "bundle_grant_sha256": _bundle_grant_sha256(deployment),
                "artifact_sha256": deployment.private_bundle.sha256,
                "install_root": str(root),
                "roles_root": str(self.install_root),
                "account_uid": account.uid,
                "account_gid": account.gid,
                "target_state": "owned",
                "docker_access": access.receipt(),
                "docker_uid_mapping": owned_receipt.get("docker_access"),
            }
        if root.is_symlink() or root.exists():
            raise FleetRoleError("Learner install path cannot be adopted")
        unit_path = self._unit_path(deployment.deployment_id)
        if unit_path.exists() or unit_path.is_symlink():
            raise FleetRoleError("unowned Learner user unit already exists")
        unit_active = self._unit_active(deployment.deployment_id)
        container_active = self._container_exists(access, deployment.deployment_id)
        if unit_active is None or unit_active or container_active is None or container_active:
            raise FleetRoleError("Learner user service or labelled container exists without an owned install")
        return {
            "role": self.role,
            "generation": generation,
            "deployment_id": deployment.deployment_id,
            "bundle_sha256": deployment.private_bundle.sha256,
            "bundle_grant_sha256": _bundle_grant_sha256(deployment),
            "artifact_sha256": deployment.private_bundle.sha256,
            "install_root": str(root),
            "roles_root": str(self.install_root),
            "account_uid": account.uid,
            "account_gid": account.gid,
            "target_state": "absent_before_install",
            "docker_access": access.receipt(),
        }

    def _profile_mapping_probe(
        self,
        deployment: Any,
        probe_dir: Path,
        *,
        access: DockerAccess,
        expected: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        mapping = probe_docker_uid_mapping(
            access,
            image=deployment.image_id,
            account=self.account,
            probe_dir=probe_dir,
            python_executable="/work/envs/rl-vllm/bin/python",
            runner=self._docker_runner(access),
            expected=expected,
        )
        expected_container = (0, 0) if access.rootless else (self.account.uid, self.account.gid)
        if deployment.container_user != f"{expected_container[0]}:{expected_container[1]}":
            raise FleetRoleError(
                "Learner profile container_user does not match the verified Docker namespace mode"
            )
        if (mapping.get("rootless") is not access.rootless
                or mapping.get("uid_mapping_verified") is not True
                or (mapping.get("container_uid"), mapping.get("container_gid")) != expected_container
                or (mapping.get("host_uid"), mapping.get("host_gid")) != (self.account.uid, self.account.gid)):
            raise FleetRoleError("Learner Docker UID/GID mapping does not map to the registered account")
        return mapping

    def _install_bundle(self, deployment: LearnerDeploymentSpec, *, generation: int,
                        access: DockerAccess, intent: Mapping[str, Any],
                        operation: Path, previous_receipt: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        from qwen_rl.deployment.learner.bootstrap import install_learner_bundle
        from qwen_rl.deployment.learner.host import prepare_learner_host

        bundle_path = _read_private_bundle(
            deployment,
            operation / "learner-bundle.tar.gz",
            ca_bundle=self.ca_bundle,
            transport=self.artifact_transport,
        )
        mapping_result: dict[str, Any] = {}
        mapping_expected = None
        if previous_receipt is not None:
            candidate = previous_receipt.get("docker_access")
            if isinstance(candidate, Mapping):
                mapping_expected = candidate

        root = _deployment_root(self.paths, deployment.deployment_id)
        rootless_host_owner = None
        if access.rootless:
            if root.exists():
                if mapping_expected is None:
                    raise FleetRoleError("existing rootless Learner state has no prior measured UID/GID mapping")
                from qwen_rl.deployment.learner.config import load_learner_deployment

                existing_profile = load_learner_deployment(
                    root / "learner-deployment.json",
                    allow_gpu_autodetect=False,
                    allow_rootless_container_user=True,
                )
                self._profile_mapping_probe(
                    existing_profile,
                    existing_profile.host_control_path / "uid-map-probe",
                    access=access,
                    expected=mapping_expected,
                )
                rootless_host_owner = (self.account.uid, self.account.gid)

        docker_config = self._docker_config_path()
        source_root = Path(__file__).resolve().parents[3]

        def host_prepare(profile, **kwargs):
            def mapping_probe(current_profile, probe_dir):
                observed = self._profile_mapping_probe(
                    current_profile,
                    probe_dir,
                    access=access,
                    expected=mapping_expected,
                )
                mapping_result.clear()
                mapping_result.update(observed)
                return observed

            try:
                kwargs.update({
                    "mode": "current_user",
                    "account": self.account,
                    "docker_context": access.context,
                    "docker_rootless": access.rootless,
                    "docker_host": access.endpoint if access.host_override else None,
                    "docker_config": docker_config,
                    "python_executable": self.python_executable,
                    "source_root": source_root,
                    "mapping_probe": mapping_probe,
                    "runner": self._docker_runner(access, docker_config),
                    "systemd_dir": self.systemd_dir,
                    "allow_package_changes": False,
                })
                return prepare_learner_host(profile, **kwargs)
            except Exception as exc:
                # The bootstrap deliberately returns a secret-safe generic error;
                # preserve only the user-safe host reason for a precise Fleet ACK.
                host_failure["message"] = str(exc)
                raise

        host_failure: dict[str, str] = {}
        try:
            installed = install_learner_bundle(
                bundle_path,
                deployment.private_bundle.sha256,
                install_root=self.install_root,
                host_prepare=host_prepare,
                runner=self._docker_runner(access, docker_config),
                systemd_dir=self.systemd_dir,
                allow_package_changes=False,
                allow_rootless_container_user=access.rootless,
                rootless_host_owner=rootless_host_owner,
            )
        except Exception as exc:
            detail = host_failure.get("message")
            if detail:
                raise FleetRoleError(detail[:400]) from exc
            raise FleetRoleError("verified private Learner bundle installation failed") from exc
        root = _deployment_root(self.paths, deployment.deployment_id)
        marker = _bundle_marker(root, deployment_id=deployment.deployment_id,
                                bundle_sha256=deployment.private_bundle.sha256, account=self.account)
        if marker is None:
            raise FleetRoleError("verified Learner install marker is missing after host preparation")
        host_receipt = installed.as_dict()
        return marker, {**host_receipt, "docker_uid_mapping": dict(mapping_result)}

    def install(
        self,
        deployment: LearnerDeploymentSpec,
        *,
        generation: int,
        inventory: HostInventory,
        newly_managed: bool,
        owned_receipt: Mapping[str, Any] | None = None,
        install_intent: Mapping[str, Any] | None = None,
    ) -> RoleResult:
        base = {
            "role": self.role,
            "generation": generation,
            "deployment_id": getattr(deployment, "deployment_id", None),
            "bundle_sha256": getattr(getattr(deployment, "private_bundle", None), "sha256", None),
            "bundle_grant_sha256": (_bundle_grant_sha256(deployment)
                                     if isinstance(deployment, LearnerDeploymentSpec) else None),
        }
        if not isinstance(deployment, LearnerDeploymentSpec):
            return RoleResult("failed", base, "role_document_mismatch",
                              "Learner adapter received another role", owned=False)
        try:
            account = self._assert_account(inventory.account)
            if inventory.account != self.account:
                raise FleetRoleError("Learner inventory account differs from the registered Fleet account")
            expected = dict(install_intent or self.prepare_install(
                deployment, generation=generation, owned_receipt=owned_receipt,
            ))
            root = _deployment_root(self.paths, deployment.deployment_id)
            if (expected.get("role") != self.role or expected.get("generation") != generation
                    or expected.get("deployment_id") != deployment.deployment_id
                    or expected.get("bundle_sha256") != deployment.private_bundle.sha256
                    or expected.get("bundle_grant_sha256") != _bundle_grant_sha256(deployment)
                    or expected.get("install_root") != str(root)
                    or expected.get("account_uid") != account.uid
                    or expected.get("account_gid") != account.gid):
                raise FleetRoleError("Learner install intent differs from the approved account assignment")
            if (owned_receipt is not None and owned_receipt.get("bundle_sha256") not in {
                    None, deployment.private_bundle.sha256}):
                raise FleetRoleError("Learner state belongs to a different private bundle pin")
            access = self._docker_access(expected=expected.get("docker_access"))
            operation = operation_dir(
                self.state_dir, generation, self.role, deployment.deployment_id, account=account,
            )
            _pin_bundle_grant(operation, deployment)
            _marker, install_receipt = self._install_bundle(
                deployment,
                generation=generation,
                access=access,
                intent=expected,
                operation=operation,
                previous_receipt=owned_receipt,
            )
            receipt = {
                **base,
                "artifact_sha256": deployment.private_bundle.sha256,
                "install_root": str(root),
                "roles_root": str(self.install_root),
                "account_uid": account.uid,
                "account_gid": account.gid,
                "target_state": "owned",
                "owned_partial": True,
                "preserved_state": True,
                "learner_host_state": install_receipt.get("host_state"),
                "learner_service_unit": self._unit(deployment.deployment_id),
                "learner_service_unit_path": str(self._unit_path(deployment.deployment_id)),
                "learner_service_active": True,
                "learner_runtime_ready": False,
                "docker_access": install_receipt.get("docker_uid_mapping"),
                "management_python": str(self.python_executable.expanduser().absolute()),
                "management_source_root": str(Path(__file__).resolve().parents[3]),
                "image_id": _marker.get("image_id"),
                "profile_sha256": _marker.get("profile_sha256"),
            }
            inspected = self.inspect(receipt)
            if inspected.status == "ready":
                return inspected
            if inspected.status == "failed":
                return RoleResult("failed", {**receipt, **inspected.receipt},
                                  inspected.error_code, inspected.error_message, owned=True)
            return RoleResult("loading", {**receipt, **inspected.receipt},
                              inspected.error_code or "learner_readiness_pending",
                              inspected.error_message or "Learner service has not reported ready", owned=True)
        except FleetRoleError as exc:
            root = _deployment_root(self.paths, deployment.deployment_id)
            owned = _bundle_marker(root, deployment_id=deployment.deployment_id,
                                   bundle_sha256=deployment.private_bundle.sha256,
                                   account=self.account) is not None
            receipt = {
                **base,
                "artifact_sha256": deployment.private_bundle.sha256,
                "install_root": str(root),
                "roles_root": str(self.install_root),
                "account_uid": self.account.uid,
                "account_gid": self.account.gid,
                "target_state": "owned_partial" if owned else "absent_before_install",
                "owned_partial": owned,
                "preserved_state": True,
            }
            return RoleResult("failed", receipt, "learner_install_failed", str(exc)[:400], owned=owned)
        except Exception:
            root = _deployment_root(self.paths, deployment.deployment_id)
            owned = _bundle_marker(root, deployment_id=deployment.deployment_id,
                                   bundle_sha256=deployment.private_bundle.sha256,
                                   account=self.account) is not None
            receipt = {
                **base,
                "artifact_sha256": deployment.private_bundle.sha256,
                "install_root": str(root),
                "roles_root": str(self.install_root),
                "account_uid": self.account.uid,
                "account_gid": self.account.gid,
                "target_state": "owned_partial" if owned else "absent_before_install",
                "owned_partial": owned,
                "preserved_state": True,
            }
            return RoleResult("failed", receipt, "learner_install_failed",
                              "Learner setup did not complete", owned=owned)

    def _load_manager(self, receipt: Mapping[str, Any], access: DockerAccess):
        from qwen_rl.deployment.learner.config import load_learner_deployment
        from qwen_rl.deployment.learner.manager import LearnerManager

        deployment_id = receipt.get("deployment_id")
        if not isinstance(deployment_id, str) or _DEPLOYMENT_ID.fullmatch(deployment_id) is None:
            raise FleetRoleError("Learner receipt deployment identity is invalid")
        root = _deployment_root(self.paths, deployment_id)
        if receipt.get("install_root") != str(root):
            raise FleetRoleError("Learner receipt install path differs from the canonical account path")
        deployment = load_learner_deployment(
            root / "learner-deployment.json",
            allow_gpu_autodetect=False,
            allow_rootless_container_user=access.rootless,
        )
        if deployment.deployment_id != deployment_id:
            raise FleetRoleError("Learner profile identity differs from the Fleet receipt")
        mapping = receipt.get("docker_access")
        self._validate_mapping_receipt(mapping, access, deployment.image_id)
        expected_container = (0, 0) if access.rootless else (self.account.uid, self.account.gid)
        if deployment.container_user != f"{expected_container[0]}:{expected_container[1]}":
            raise FleetRoleError("Learner profile does not match the verified Docker namespace mapping")
        return deployment, LearnerManager(
            deployment,
            runner=self._docker_runner(access, self._docker_config_path()),
            docker=str(access.executable),
            allow_rootless_container_user=access.rootless,
            host_uid=self.account.uid,
            host_gid=self.account.gid,
        )

    def _validate_mapping_receipt(
        self,
        value: Any,
        access: DockerAccess,
        image_id: str,
    ) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or value.get("uid_mapping_verified") is not True:
            raise FleetRoleError("Learner Docker UID/GID mapping receipt is incomplete")
        for key, observed in access.receipt().items():
            if value.get(key) != observed:
                raise FleetRoleError("Learner Docker context receipt differs from the active local daemon")
        expected_container = (0, 0) if access.rootless else (self.account.uid, self.account.gid)
        if (value.get("host_uid"), value.get("host_gid")) != (self.account.uid, self.account.gid):
            raise FleetRoleError("Learner Docker UID/GID mapping does not target the registered account")
        if (value.get("container_uid"), value.get("container_gid")) != expected_container:
            raise FleetRoleError("Learner Docker UID/GID mapping does not match the daemon namespace")
        if (value.get("probe_image") != image_id
                or value.get("probe_python") != "/work/envs/rl-vllm/bin/python"):
            raise FleetRoleError("Learner Docker UID/GID mapping receipt uses another probe")
        return value

    def _monitor(self, deployment: Any, manager: Any) -> dict[str, Any] | None:
        try:
            token = manager._read_arc_token()
            address = deployment.publish.host_address
            if ":" in address:
                address = f"[{address}]"
            url = f"http://{address}:{deployment.publish.host_port}/monitor"
            with httpx.Client(timeout=8, follow_redirects=False, trust_env=False) as client:
                response = client.get(url, headers={"Authorization": f"Bearer {token}"})
        except Exception:
            return None
        if response.status_code != 200 or len(response.content) > 2_000_000:
            return None
        try:
            value = response.json()
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _monitor_is_quiescent(monitor: Mapping[str, Any] | None) -> bool:
        if not isinstance(monitor, Mapping):
            return False
        run = monitor.get("run")
        sampling = monitor.get("sampling")
        training = monitor.get("training")
        if not isinstance(run, Mapping) or not isinstance(sampling, Mapping) or not isinstance(training, Mapping):
            return False
        if run.get("status") == "running" or sampling.get("active_groups") != 0:
            return False
        if training.get("active_update") is not False:
            return False
        activity = training.get("activity")
        if not isinstance(activity, Mapping) or activity.get("state") not in {"idle", "complete", "completed"}:
            return False
        if monitor.get("pending_policy_handoff") is not None:
            return False
        evaluation = monitor.get("evaluation")
        if isinstance(evaluation, Mapping) and evaluation.get("status") in {"running", "queued", "starting"}:
            return False
        return True

    def inspect(self, receipt: Mapping[str, Any]) -> RoleResult:
        safe_receipt = dict(receipt)
        deployment_id = receipt.get("deployment_id")
        if (receipt.get("role") != self.role or not isinstance(deployment_id, str)
                or _DEPLOYMENT_ID.fullmatch(deployment_id) is None):
            return RoleResult("failed", safe_receipt, "learner_receipt_invalid",
                              "Learner receipt identity is invalid", owned=False)
        ownership_proven = False
        try:
            account = self._assert_account()
            if (receipt.get("account_uid") != account.uid or receipt.get("account_gid") != account.gid
                    or receipt.get("artifact_sha256") != receipt.get("bundle_sha256")):
                raise FleetRoleError("Learner receipt account or artifact pin differs from this assignment")
            root = _deployment_root(self.paths, deployment_id)
            marker = _bundle_marker(root, deployment_id=deployment_id,
                                    bundle_sha256=str(receipt.get("bundle_sha256", "")), account=account)
            if marker is None or marker.get("profile_sha256") != receipt.get("profile_sha256"):
                raise FleetRoleError("Learner bundle marker no longer matches the owned receipt")
            ownership_proven = True
            if marker.get("image_id") != receipt.get("image_id"):
                raise FleetRoleError("Learner image pin differs from the owned receipt")
            access_record = receipt.get("docker_access")
            access = self._docker_access(expected=access_record)
            self._validate_mapping_receipt(access_record, access, str(marker.get("image_id", "")))
            unit_path = self._unit_path(deployment_id)
            if receipt.get("learner_service_unit_path") != str(unit_path):
                raise FleetRoleError("Learner user unit path differs from the canonical account path")
            unit_active = self._unit_active(deployment_id)
            if unit_active is None:
                return RoleResult("loading", safe_receipt, "learner_user_service_unknown",
                                  "Could not verify the user Learner service state")
            if not unit_path.is_file() or unit_path.is_symlink() or unit_active is not True:
                return RoleResult("loading", safe_receipt, "learner_user_service_not_active",
                                  "Owned user Learner unit is not active")
            info = unit_path.lstat()
            if (info.st_uid != account.uid or info.st_gid != account.gid
                    or stat.S_IMODE(info.st_mode) != 0o644):
                raise FleetRoleError("Learner user unit ownership or mode changed")
            deployment, manager = self._load_manager(receipt, access)
            if deployment.container_user != f"{access_record.get('container_uid')}:{access_record.get('container_gid')}":
                raise FleetRoleError("Learner profile user no longer matches the measured Docker UID/GID mapping")
            status = manager.status()
            runtime = status.get("runtime") if isinstance(status, dict) else None
            if isinstance(status, dict) and status.get("status") == "resumed_ready":
                return RoleResult("failed", safe_receipt, "learner_resume_detected",
                                  "Learner runtime reports resumed state; Fleet installation does not resume runs",
                                  owned=True)
            if not isinstance(runtime, dict) or status.get("status") not in {"ready", "running"}:
                return RoleResult("loading", safe_receipt, "learner_runtime_not_ready",
                                  "Learner runtime has not reported ready")
            monitor = self._monitor(deployment, manager)
            if not self._monitor_is_quiescent(monitor):
                return RoleResult("loading", safe_receipt, "learner_training_not_idle",
                                  "Learner readiness monitor is unknown or reports active work")
            user_service = {"unit": self._unit(deployment_id), "path": str(unit_path), "active": True}
            ready_receipt = {
                **safe_receipt,
                "readiness_verified": True,
                "artifact_sha256": receipt.get("bundle_sha256"),
                "account_uid": account.uid,
                "account_gid": account.gid,
                "docker_access": dict(access_record),
                "user_services": [user_service],
                "learner_service_unit": self._unit(deployment_id),
                "learner_service_unit_path": str(unit_path),
                "learner_service_active": True,
                "learner_host_state": "started",
                "learner_runtime_ready": True,
                "learner_training_idle": True,
                "owned_partial": False,
                "management_python": str(self.python_executable.expanduser().absolute()),
                "management_source_root": str(Path(__file__).resolve().parents[3]),
                "image_id": deployment.image_id,
                "profile_sha256": marker["profile_sha256"],
                "preserved_state": True,
            }
            return RoleResult("ready", ready_receipt)
        except FleetRoleError as exc:
            return RoleResult("failed", safe_receipt, "learner_ownership_or_readiness_failed",
                              str(exc)[:400], owned=ownership_proven)
        except Exception:
            return RoleResult("loading", safe_receipt, "learner_health_unavailable",
                              "Could not verify Learner runtime readiness")

    def stop_owned(self, receipt: Mapping[str, Any]) -> StopResult:
        if receipt.get("role") != self.role:
            return StopResult("failed", {}, "role_receipt_mismatch", "Learner receipt does not match its role")
        deployment_id = receipt.get("deployment_id")
        if not isinstance(deployment_id, str) or _DEPLOYMENT_ID.fullmatch(deployment_id) is None:
            return StopResult("blocked_busy", dict(receipt), "learner_ownership_unknown",
                              "Learner deployment identity is unavailable")
        try:
            account = self._assert_account()
            if receipt.get("account_uid") != account.uid or receipt.get("account_gid") != account.gid:
                raise FleetRoleError("Learner stop receipt differs from the registered account")
            root = _deployment_root(self.paths, deployment_id)
            exists = root.exists() or root.is_symlink()
            if not exists and receipt.get("target_state") == "absent_before_install":
                access = self._docker_access(expected=receipt.get("docker_access"))
                unit_active = self._unit_active(deployment_id)
                container_exists = self._container_exists(access, deployment_id)
                if unit_active is False and container_exists is False:
                    return StopResult("stopped", {"role": self.role, "deployment_id": deployment_id,
                                                     "preserved_state": True})
            if not exists:
                return StopResult("blocked_busy", dict(receipt), "learner_state_missing",
                                  "Recorded Learner install state is missing")
            marker = _bundle_marker(root, deployment_id=deployment_id,
                                    bundle_sha256=str(receipt.get("bundle_sha256", "")), account=account)
            if marker is None:
                return StopResult("blocked_busy", dict(receipt), "learner_ownership_unknown",
                                  "Learner install marker cannot prove ownership")
            access = self._docker_access(expected=receipt.get("docker_access"))
            deployment, manager = self._load_manager(receipt, access)
            status = manager.status()
            if status.get("status") in {"ready", "resumed_ready", "running"}:
                monitor = self._monitor(deployment, manager)
                if not self._monitor_is_quiescent(monitor):
                    return StopResult("blocked_busy", dict(receipt), "learner_busy_or_unknown",
                                      "Learner run, sampling, training, or handoff state is not safely drained")
            unit_path = self._unit_path(deployment_id)
            if unit_path.exists() or unit_path.is_symlink():
                info = unit_path.lstat()
                if (unit_path.is_symlink() or not stat.S_ISREG(info.st_mode)
                        or info.st_uid != account.uid or info.st_gid != account.gid):
                    return StopResult("blocked_busy", dict(receipt), "learner_unit_ownership_unknown",
                                      "Learner user unit is not owned by the registered account")
                systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
                run_owned(
                    [systemctl, "--user", "disable", "--now", self._unit(deployment_id)],
                    runner=self._docker_runner(access, self._docker_config_path()), timeout=180,
                )
            # Stop only the exact Manager-verified container and leave profile, datasets and checkpoints intact.
            manager.stop(timeout=120)
            stopped = manager.status()
            if stopped.get("status") not in {"stopped", "exited"}:
                return StopResult("blocked_busy", dict(receipt), "learner_shutdown_unverified",
                                  "Owned Learner container did not stop cleanly")
            return StopResult("stopped", {"role": self.role, "deployment_id": deployment_id,
                                            "install_root": str(root), "preserved_state": True,
                                            "learner_service_unit": self._unit(deployment_id),
                                            "container_removed": True})
        except FleetRoleError as exc:
            return StopResult("blocked_busy", dict(receipt), "learner_ownership_or_shutdown_unknown",
                              str(exc)[:400])
        except Exception:
            return StopResult("blocked_busy", dict(receipt), "learner_shutdown_unverified",
                              "Could not prove safe shutdown of the owned Learner")
