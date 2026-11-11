"""Typed public and protected contracts for unified Ubuntu node onboarding."""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Union
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ..contracts import GPUCapability
from ..identity import canonical_public_key


_ID = re.compile(r"^[A-Za-z0-9_-]{16,100}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CODE = re.compile(r"^[A-Za-z0-9_-]{16,512}$")
_SAFE_UNIX_USERNAME = r"^[A-Za-z0-9_.-]{1,128}$"
_WORKER_JOIN_CODE = re.compile(r"^[A-Za-z0-9_-]{16,200}$")
MANAGEMENT_SSH_PORT = 22022


class FleetContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class UserAccount(FleetContract):
    """The real non-root Unix account that owns a managed node."""

    username: StrictStr = Field(min_length=1, max_length=128, pattern=_SAFE_UNIX_USERNAME)
    uid: StrictInt = Field(gt=0)
    gid: StrictInt = Field(ge=0)
    home: StrictStr = Field(min_length=1, max_length=4096)

    @field_validator("username")
    @classmethod
    def valid_username(cls, value: str) -> str:
        if value in {"root", ".", ".."}:
            raise ValueError("username must be a non-root SSH-safe account name")
        return value

    @field_validator("home")
    @classmethod
    def absolute_home(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (not path.is_absolute() or value == "/" or ".." in path.parts
                or any(ord(c) < 32 or c in "\u0085\u2028\u2029" for c in value)):
            raise ValueError("home must be an absolute, non-root, single-line path without parent traversal")
        return value


def ssh_key_fingerprint(value: str) -> str:
    """Return the OpenSSH SHA256 fingerprint of a validated public key."""
    canonical = canonical_public_key(value)
    encoded = canonical.split()[1]
    digest = hashlib.sha256(base64.b64decode(encoded, validate=True)).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def _validate_https_url(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 2048 or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError(f"{label} must be a single-line HTTPS URL")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid HTTPS URL") from exc
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError(f"{label} must be HTTPS without embedded credentials, query, or fragment")
    return value


def _safe_single_line(value: str, label: str, *, max_length: int) -> str:
    if not value or len(value) > max_length or any(c in value for c in "\r\n\x00"):
        raise ValueError(f"{label} must be a nonempty single-line value")
    return value


def _ed25519_public_key(value: str, *, label: str) -> str:
    try:
        canonical = canonical_public_key(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid Ed25519 OpenSSH public key") from exc
    if canonical.split()[0] != "ssh-ed25519":
        raise ValueError(f"{label} must use Ed25519")
    return canonical


class ArtifactReference(FleetContract):
    url: StrictStr = Field(min_length=1, max_length=2048)
    sha256: StrictStr = Field(pattern=_SHA256.pattern)

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        return _validate_https_url(value, label="artifact URL")


class FleetNodeRegistration(FleetContract):
    """Role-neutral host facts bound to the authenticated Unix account."""

    node_id: StrictStr = Field(pattern=_ID.pattern)
    boot_id: StrictStr = Field(min_length=1, max_length=100)
    hostname: StrictStr = Field(min_length=1, max_length=253)
    os_release: StrictStr = Field(min_length=1, max_length=256)
    architecture: StrictStr = Field(min_length=1, max_length=64)
    agent_version: StrictStr = Field(min_length=1, max_length=100)
    account: UserAccount
    lan_host: StrictStr | None = Field(default=None, max_length=253)
    ssh_port: StrictInt = Field(default=MANAGEMENT_SSH_PORT, ge=1, le=65535)
    ssh_public_key: StrictStr = Field(min_length=1, max_length=2000)
    ssh_host_public_key: StrictStr = Field(min_length=1, max_length=2000)
    gpus: list[GPUCapability] = Field(default_factory=list, max_length=100)

    @field_validator("ssh_port")
    @classmethod
    def fixed_management_ssh_port(cls, value: int) -> int:
        if value != MANAGEMENT_SSH_PORT:
            raise ValueError("management SSH must bind the fixed loopback port")
        return value

    @field_validator("boot_id", "hostname", "os_release", "architecture", "agent_version", "lan_host")
    @classmethod
    def single_line_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _safe_single_line(value, "node inventory value", max_length=256)

    @field_validator("ssh_public_key", "ssh_host_public_key")
    @classmethod
    def valid_ssh_key(cls, value: str) -> str:
        return _ed25519_public_key(value, label="SSH identity")


class FleetNodeRegistrationReceipt(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    approval: Literal["pending", "approved", "rejected"]
    registered_at: StrictStr
    desired_generation: StrictInt = Field(ge=0)
    reverse_port: StrictInt | None = Field(default=None, ge=1024, le=65535)


class FleetHeartbeatReceipt(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    accepted: StrictBool = True
    desired_generation: StrictInt = Field(ge=0)
    server_time: StrictStr


class NodeHeartbeat(FleetContract):
    boot_id: StrictStr = Field(min_length=1, max_length=100)
    agent_version: StrictStr = Field(min_length=1, max_length=100)
    state: Literal[
        "registered", "installing", "needs_reboot", "needs_driver", "needs_host_packages",
        "draining", "loading", "ready", "failed", "deactivated",
    ]
    observed_generation: StrictInt = Field(ge=0)
    role: Literal["worker", "learner"] | None = None
    error_code: StrictStr | None = Field(default=None, max_length=100)
    error_message: StrictStr | None = Field(default=None, max_length=512)
    receipt: dict[str, Any] | None = None

    @field_validator("boot_id", "agent_version", "error_code", "error_message")
    @classmethod
    def safe_heartbeat_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _safe_single_line(value, "heartbeat value", max_length=512)

    @field_validator("receipt")
    @classmethod
    def finite_receipt(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None:
            try:
                encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise ValueError("receipt must be finite JSON") from exc
            if len(encoded.encode("utf-8")) > 64_000:
                raise ValueError("receipt exceeds the size limit")
        return value


class DesiredStateResponse(FleetContract):
    approval: Literal["pending", "approved", "rejected"]
    generation: StrictInt = Field(ge=0)
    action: Literal["none", "install", "deactivate"]
    role: Literal["worker", "learner"] | None = None
    deployment_id: StrictStr | None = Field(default=None, max_length=100)
    deployment_sha256: StrictStr | None = Field(default=None, pattern=_SHA256.pattern)
    reverse_port: StrictInt | None = Field(default=None, ge=1024, le=65535)

    @model_validator(mode="after")
    def coherent_assignment(self) -> "DesiredStateResponse":
        if self.action == "install" and (self.role is None or self.deployment_id is None or self.deployment_sha256 is None):
            raise ValueError("install desired state requires a role and deployment descriptor")
        if self.action != "install" and any(x is not None for x in (self.role, self.deployment_id, self.deployment_sha256)):
            raise ValueError("only install desired state may include a role descriptor")
        if self.approval in {"pending", "rejected"} and self.reverse_port is not None:
            raise ValueError("pending or rejected nodes cannot have an SSH reverse listener")
        if self.approval == "rejected" and (self.action != "none" or self.generation != 0):
            raise ValueError("rejected nodes cannot retain a desired role or generation")
        return self


class WorkerDeployment(FleetContract):
    role: Literal["worker"]
    deployment_id: StrictStr = Field(pattern=_ID.pattern)
    source_bundle: ArtifactReference
    controller_url: StrictStr = Field(min_length=1, max_length=2048)
    site_id: StrictStr = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    worker_join_code: SecretStr = Field(min_length=16, max_length=200, repr=False)
    gpu_uuid: StrictStr = Field(min_length=1, max_length=100)

    @field_validator("controller_url")
    @classmethod
    def valid_controller_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            parsed.port
        except ValueError as exc:
            raise ValueError("controller_url must be a valid HTTPS origin") from exc
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or any(c.isspace() or ord(c) < 32 for c in value)):
            raise ValueError("controller_url must be an HTTPS origin without credentials or path")
        return value.rstrip("/")

    @field_validator("worker_join_code")
    @classmethod
    def valid_join_code(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if _WORKER_JOIN_CODE.fullmatch(secret) is None:
            raise ValueError("worker_join_code must be a single-line opaque code")
        return value

    @field_validator("gpu_uuid")
    @classmethod
    def valid_gpu_uuid(cls, value: str) -> str:
        if any(c.isspace() or ord(c) < 32 for c in value):
            raise ValueError("gpu_uuid must be a single-line identifier")
        return value


class LearnerDeployment(FleetContract):
    role: Literal["learner"]
    deployment_id: StrictStr = Field(pattern=_ID.pattern)
    private_bundle: ArtifactReference
    bundle_bearer: SecretStr = Field(min_length=16, max_length=512, repr=False)

    @field_validator("bundle_bearer")
    @classmethod
    def valid_bundle_bearer(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if _SAFE_CODE.fullmatch(secret) is None:
            raise ValueError("bundle_bearer must be a single-line opaque value")
        return value


FleetRoleDeployment = Annotated[Union[WorkerDeployment, LearnerDeployment], Field(discriminator="role")]


class LearnerDeploymentTemplate(FleetContract):
    """Admin-only request template; local and external bundle bearers differ."""

    role: Literal["learner"]
    deployment_id: StrictStr = Field(pattern=_ID.pattern)
    private_bundle: ArtifactReference
    publisher_bearer: SecretStr | None = Field(default=None, min_length=16, max_length=512, repr=False)
    external_bearer: SecretStr | None = Field(default=None, min_length=16, max_length=512, repr=False)

    @field_validator("publisher_bearer", "external_bearer")
    @classmethod
    def valid_template_bearer(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and _SAFE_CODE.fullmatch(value.get_secret_value()) is None:
            raise ValueError("bundle authorization must be a single-line opaque value")
        return value

    @model_validator(mode="after")
    def one_bundle_authorization(self) -> "LearnerDeploymentTemplate":
        if (self.publisher_bearer is None) == (self.external_bearer is None):
            raise ValueError("set exactly one of publisher_bearer or external_bearer")
        return self


FleetRoleTemplate = Annotated[Union[WorkerDeployment, LearnerDeploymentTemplate], Field(discriminator="role")]


class FleetAssignmentRequest(FleetContract):
    deployment: FleetRoleTemplate


class PrivateDeploymentEnvelope(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    generation: StrictInt = Field(ge=1)
    deployment: FleetRoleDeployment


class ApplyAckRequest(FleetContract):
    generation: StrictInt = Field(ge=1)
    status: Literal[
        "installing", "needs_reboot", "needs_driver", "needs_host_packages",
        "draining", "loading", "ready", "failed", "deactivated",
    ]
    role: Literal["worker", "learner"] | None = None
    error_code: StrictStr | None = Field(default=None, max_length=100)
    error_message: StrictStr | None = Field(default=None, max_length=512)
    receipt: dict[str, Any] | None = None

    @field_validator("error_code", "error_message")
    @classmethod
    def safe_ack_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _safe_single_line(value, "ack value", max_length=512)

    @field_validator("receipt")
    @classmethod
    def finite_ack_receipt(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return NodeHeartbeat.finite_receipt(value)


class ApplyAckReceipt(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    generation: StrictInt = Field(ge=1)
    status: Literal[
        "installing", "needs_reboot", "needs_driver", "needs_host_packages",
        "draining", "loading", "ready", "failed", "deactivated",
    ]
    accepted: StrictBool = True


class FleetNodeView(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    boot_id: StrictStr
    hostname: StrictStr
    os_release: StrictStr
    architecture: StrictStr
    agent_version: StrictStr
    account: UserAccount
    lan_host: StrictStr | None
    ssh_port: StrictInt
    ssh_public_key: StrictStr
    ssh_public_key_fingerprint: StrictStr
    ssh_host_public_key: StrictStr
    ssh_host_fingerprint: StrictStr
    gpus: list[GPUCapability]
    approval: Literal["pending", "approved", "rejected"]
    created_at: StrictStr
    rejected_at: StrictStr | None
    last_heartbeat_at: StrictStr | None
    heartbeat_stale: StrictBool
    desired_generation: StrictInt = Field(ge=0)
    desired_action: Literal["none", "install", "deactivate"]
    desired_role: Literal["worker", "learner"] | None
    deployment_id: StrictStr | None
    deployment_sha256: StrictStr | None
    reverse_port: StrictInt | None
    observed_generation: StrictInt = Field(ge=0)
    observed_state: Literal[
        "registered", "installing", "needs_reboot", "needs_driver", "needs_host_packages",
        "draining", "loading", "ready", "failed", "deactivated",
    ]
    observed_role: Literal["worker", "learner"] | None
    observed_error_code: StrictStr | None
    observed_error_message: StrictStr | None
    observed_receipt: dict[str, Any] | None


class FleetSSHGrant(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    reverse_port: StrictInt = Field(ge=1024, le=65535)
    ssh_port: StrictInt = Field(ge=1, le=65535)
    ssh_host_public_key: StrictStr
    ssh_public_key: StrictStr
    ssh_public_key_fingerprint: StrictStr


class FleetApprovalReceipt(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    approval: Literal["approved"]
    generation: StrictInt = Field(ge=1)
    reverse_port: StrictInt = Field(ge=1024, le=65535)


class FleetRejectionReceipt(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    approval: Literal["rejected"]
    rejected_at: StrictStr
    changed: StrictBool


class FleetRejectionAuditRecord(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    rejected_at: StrictStr
    pruned_at: StrictStr


class FleetAssignmentReceipt(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    generation: StrictInt = Field(ge=1)
    action: Literal["install"]
    role: Literal["worker", "learner"]
    deployment_id: StrictStr = Field(pattern=_ID.pattern)
    deployment_sha256: StrictStr = Field(pattern=_SHA256.pattern)
    changed: StrictBool


class FleetDeactivationReceipt(FleetContract):
    node_id: StrictStr = Field(pattern=_ID.pattern)
    generation: StrictInt = Field(ge=1)
    action: Literal["deactivate"]
    changed: StrictBool


class FleetBootstrapConfig(FleetContract):
    """Nonsecret trust anchors needed by the pre-role management agent."""

    ssh_relay_user: StrictStr = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    ssh_relay_host_key: StrictStr = Field(min_length=1, max_length=2000)
    ssh_relay_port: StrictInt = Field(default=22230, ge=1024, le=65535)
    admin_ssh_public_key: StrictStr = Field(min_length=1, max_length=2000)
    ssh_port: StrictInt = Field(default=MANAGEMENT_SSH_PORT, ge=1, le=65535)

    @field_validator("ssh_port")
    @classmethod
    def fixed_management_ssh_port(cls, value: int) -> int:
        if value != MANAGEMENT_SSH_PORT:
            raise ValueError("management SSH must use the fixed loopback port")
        return value

    @field_validator("ssh_relay_host_key", "admin_ssh_public_key")
    @classmethod
    def canonical_ssh_key(cls, value: str) -> str:
        return _ed25519_public_key(value, label="bootstrap SSH trust anchor")


def role_deployment_to_dict(value: FleetRoleDeployment) -> dict[str, Any]:
    """Serialize one protected role document, intentionally revealing secrets for transport/storage."""
    data = value.model_dump(mode="json")
    if isinstance(value, WorkerDeployment):
        data["worker_join_code"] = value.worker_join_code.get_secret_value()
    else:
        data["bundle_bearer"] = value.bundle_bearer.get_secret_value()
    return data


def role_deployment_canonical_json(value: FleetRoleDeployment) -> str:
    return json.dumps(role_deployment_to_dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def role_deployment_sha256(value: FleetRoleDeployment) -> str:
    return hashlib.sha256(role_deployment_canonical_json(value).encode("utf-8")).hexdigest()


def role_template_to_dict(value: FleetRoleTemplate) -> dict[str, Any]:
    """Serialize an admin assignment template for its nonreversible idempotency hash."""
    if isinstance(value, WorkerDeployment):
        return role_deployment_to_dict(value)
    data = value.model_dump(mode="json")
    if value.publisher_bearer is not None:
        data["publisher_bearer"] = value.publisher_bearer.get_secret_value()
    if value.external_bearer is not None:
        data["external_bearer"] = value.external_bearer.get_secret_value()
    return data


def role_template_sha256(value: FleetRoleTemplate) -> str:
    encoded = json.dumps(role_template_to_dict(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def private_deployment_to_dict(value: PrivateDeploymentEnvelope) -> dict[str, Any]:
    return {
        "node_id": value.node_id,
        "generation": value.generation,
        "deployment": role_deployment_to_dict(value.deployment),
    }
