"""Persistent role-neutral node agent for the Fleet control plane."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import ConfigDict, Field, SecretStr, StrictInt, StrictStr, field_validator, model_validator

from ..identity import (
    NodeIdentity,
    atomic_write,
    canonical_public_key,
    read_private_json,
    read_ssh_public_key,
)
from .contracts import (
    MANAGEMENT_SSH_PORT,
    ApplyAckReceipt,
    ApplyAckRequest,
    DesiredStateResponse,
    FleetContract,
    FleetNodeRegistrationReceipt,
    FleetNodeRegistration,
    LearnerDeployment,
    NodeHeartbeat,
    PrivateDeploymentEnvelope,
    UserAccount,
    WorkerDeployment,
    private_deployment_to_dict,
    role_deployment_sha256,
)
from .host import FleetHostError, HostInventory, collect_inventory, current_user_account
from .paths import AccountPathError, UserPaths, ensure_account_path
from .roles import (
    RoleAdapter,
    RoleResult,
    StopResult,
    default_role_adapters,
)
from .watchdog import notify_ready, notify_watchdog


_TOKEN = re.compile(r"^[A-Za-z0-9_-]{32,512}$")
_SAFE_ERROR = re.compile(r"^[a-z0-9_]{1,100}$")
_STATE_SCHEMA = "qwen-fleet-agent-state/v1"
_MAX_RESPONSE_BYTES = 256 * 1024


def _require_private_user_file(path: Path, *, uid: int, label: str) -> None:
    try:
        metadata = Path(path).lstat()
    except OSError as exc:
        raise FleetAgentError(f"{label} is unavailable") from exc
    if (not stat.S_ISREG(metadata.st_mode) or Path(path).is_symlink()
            or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_uid != uid):
        raise FleetAgentError(f"{label} must be a private regular mode-0600 file owned by the invoking account")


def _ensure_account_directory(path: Path, *, account: UserAccount, label: str, create: bool) -> Path:
    try:
        return ensure_account_path(path, account=account, create=create)
    except AccountPathError as exc:
        raise FleetAgentError(f"{label} is outside the invoking account's safe home paths") from exc


def _loaded_source_root() -> Path:
    return Path(__file__).resolve().parents[3]


class FleetAgentError(RuntimeError):
    """Expected, sanitized node agent error."""


class FleetAgentSettings(FleetContract):
    """User-owned bootstrap settings. Identity and account facts are derived locally."""

    controller_url: StrictStr = Field(min_length=1, max_length=2048)
    ca_bundle: Path
    state_dir: Path
    agent_version: StrictStr = Field(min_length=1, max_length=100)
    ssh_relay_user: StrictStr = Field(min_length=1, max_length=128)
    ssh_relay_port: StrictInt = Field(default=22230, ge=1024, le=65535)
    ssh_relay_host_key: StrictStr = Field(min_length=1, max_length=2000)
    admin_ssh_public_key: StrictStr = Field(min_length=1, max_length=2000)
    ssh_port: StrictInt = Field(default=MANAGEMENT_SSH_PORT, ge=1, le=65535)
    poll_interval_seconds: StrictInt = Field(default=15, ge=2, le=3600)
    heartbeat_interval_seconds: StrictInt = Field(default=15, ge=2, le=30)

    @field_validator("controller_url")
    @classmethod
    def https_origin(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            parsed.port
        except ValueError as exc:
            raise ValueError("Fleet controller must be a valid HTTPS origin") from exc
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError("Fleet controller must be an HTTPS origin without credentials or path")
        if any(char.isspace() or ord(char) < 32 for char in value):
            raise ValueError("Fleet controller origin must be a single-line URL")
        return value.rstrip("/")

    @field_validator("ca_bundle", "state_dir")
    @classmethod
    def absolute_local_paths(cls, value: Path) -> Path:
        if not value.is_absolute() or "\x00" in str(value) or any(ord(char) < 32 for char in str(value)):
            raise ValueError("Fleet agent paths must be absolute single-line paths")
        return value

    @field_validator("agent_version")
    @classmethod
    def safe_agent_version(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("agent_version must be a lowercase SHA-256 source identity")
        return value

    @field_validator("ssh_relay_user")
    @classmethod
    def safe_relay_user(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None or value == "root":
            raise ValueError("Fleet relay user is invalid")
        return value

    @field_validator("ssh_relay_host_key", "admin_ssh_public_key")
    @classmethod
    def valid_public_keys(cls, value: str) -> str:
        try:
            canonical = canonical_public_key(value)
        except ValueError as exc:
            raise ValueError("Fleet SSH public key is invalid") from exc
        if canonical.split()[0] != "ssh-ed25519":
            raise ValueError("Fleet SSH public keys must use Ed25519")
        return canonical

    @model_validator(mode="after")
    def fixed_management_port(self) -> "FleetAgentSettings":
        if self.ssh_port != MANAGEMENT_SSH_PORT:
            raise ValueError(f"Fleet management SSH must use port {MANAGEMENT_SSH_PORT}")
        return self

    @classmethod
    def load(cls, path: str | Path, *, account: UserAccount | None = None) -> "FleetAgentSettings":
        account = account or current_user_account()
        if account.uid == 0 or account.uid != os.geteuid():
            raise FleetAgentError("Fleet agent settings account must match the non-root invoking UID")
        paths = UserPaths.from_account(account)
        source = Path(path)
        if source != paths.management_dir / "agent.json":
            raise FleetAgentError("Fleet agent settings must use the canonical user management path")
        _ensure_account_directory(source.parent, account=account, label="Fleet agent settings path", create=False)
        _require_private_user_file(source, uid=account.uid, label="Fleet agent settings")
        try:
            settings = cls.model_validate_json(source.read_bytes())
        except Exception as exc:
            raise FleetAgentError("Fleet agent settings do not match the local schema") from exc
        if (settings.state_dir != paths.state_dir
                or settings.ca_bundle != paths.trust_dir / "upload-ca.pem"):
            raise FleetAgentError("Fleet agent paths must be bound to the invoking account home")
        _ensure_account_directory(settings.ca_bundle.parent, account=account,
                                  label="Fleet TLS trust path", create=False)
        _require_private_user_file(settings.ca_bundle, uid=account.uid, label="Fleet TLS CA bundle")
        return settings

    @property
    def websocket_url(self) -> str:
        parsed = urlsplit(self.controller_url)
        return urlunsplit(("wss", parsed.netloc, "/v1/fleet/ssh", "", ""))

    @property
    def node_id_path(self) -> Path:
        return self.state_dir / "node-identity.json"

    @property
    def bearer_path(self) -> Path:
        return self.state_dir / "node-bearer"

    @property
    def agent_state_path(self) -> Path:
        return self.state_dir / "agent-state.json"

    @property
    def client_key_path(self) -> Path:
        return self.state_dir / "ssh" / "id_ed25519"

    @property
    def host_key_path(self) -> Path:
        return self.state_dir / "ssh" / "ssh_host_ed25519_key"


class FleetHTTPError(FleetAgentError):
    def __init__(self, operation: str, status_code: int):
        self.status_code = status_code
        super().__init__(f"Fleet {operation} returned HTTP {status_code}")


class FleetHTTPClient:
    """Small typed client for the node-only Fleet routes."""

    def __init__(
        self,
        settings: FleetAgentSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        # Construct lazily in the agent's running event loop. The bootstrap CLI
        # builds NodeAgent synchronously before asyncio.run starts.
        if self._client is None:
            self._client = httpx.AsyncClient(
                verify=str(self.settings.ca_bundle), transport=self.transport,
                timeout=self.timeout_seconds, follow_redirects=False,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self, bearer: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {bearer}", "Accept": "application/json"}

    async def _json(self, method: str, path: str, *, operation: str, bearer: str,
                    payload: Mapping[str, Any] | None = None) -> bytes:
        try:
            response = await self._get_client().request(
                method,
                f"{self.settings.controller_url}{path}",
                json=dict(payload) if payload is not None else None,
                headers=self._headers(bearer),
            )
        except httpx.HTTPError as exc:
            raise FleetAgentError(f"Fleet {operation} request failed") from exc
        if response.url.scheme != "https" or response.status_code not in {200, 201}:
            raise FleetHTTPError(operation, response.status_code)
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise FleetAgentError(f"Fleet {operation} response exceeded its size limit")
        return response.content

    async def register(self, bearer: str, payload: FleetNodeRegistration) -> FleetNodeRegistrationReceipt:
        raw = await self._json("POST", "/v1/fleet/nodes/register", operation="registration",
                               bearer=bearer, payload=payload.model_dump(mode="json"))
        try:
            return FleetNodeRegistrationReceipt.model_validate_json(raw)
        except Exception as exc:
            raise FleetAgentError("Fleet registration response did not match its contract") from exc

    async def heartbeat(self, node_id: str, bearer: str, payload: NodeHeartbeat) -> None:
        path = f"/v1/fleet/nodes/{node_id}/heartbeat"
        try:
            response = await self._get_client().post(
                f"{self.settings.controller_url}{path}",
                json=payload.model_dump(mode="json"), headers=self._headers(bearer),
            )
        except httpx.HTTPError as exc:
            raise FleetAgentError("Fleet heartbeat request failed") from exc
        if response.url.scheme != "https" or response.status_code not in {200, 204}:
            raise FleetHTTPError("heartbeat", response.status_code)
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise FleetAgentError("Fleet heartbeat response exceeded its size limit")

    async def desired_state(self, node_id: str, bearer: str) -> DesiredStateResponse:
        raw = await self._json("GET", f"/v1/fleet/nodes/{node_id}/desired", operation="desired state", bearer=bearer)
        try:
            return DesiredStateResponse.model_validate_json(raw)
        except Exception as exc:
            raise FleetAgentError("Fleet desired state did not match its contract") from exc

    async def deployment(self, node_id: str, generation: int, bearer: str) -> PrivateDeploymentEnvelope:
        raw = await self._json(
            "GET", f"/v1/fleet/nodes/{node_id}/desired/{generation}/deployment",
            operation="protected deployment", bearer=bearer,
        )
        try:
            return PrivateDeploymentEnvelope.model_validate_json(raw)
        except Exception as exc:
            raise FleetAgentError("protected Fleet deployment did not match its contract") from exc

    async def acknowledge(self, node_id: str, bearer: str, payload: ApplyAckRequest) -> ApplyAckReceipt:
        raw = await self._json(
            "POST", f"/v1/fleet/nodes/{node_id}/desired/{payload.generation}/ack",
            operation="assignment acknowledgment", bearer=bearer,
            payload=payload.model_dump(mode="json"),
        )
        try:
            return ApplyAckReceipt.model_validate_json(raw)
        except Exception as exc:
            raise FleetAgentError("Fleet acknowledgment response did not match its contract") from exc


class AgentStateStore:
    """Atomic, private local state. Protected deployments live here only after approval."""

    def __init__(self, path: Path, *, node_id: str):
        self.path = Path(path)
        self.node_id = node_id
        self._lock = threading.RLock()
        self._state = self._read()

    def _read(self) -> dict[str, Any]:
        default = {
            "schema": _STATE_SCHEMA,
            "node_id": self.node_id,
            "registered": False,
            "terminal_rejection": False,
            "registered_boot_id": None,
            "approval": "pending",
            "desired_generation": 0,
            "desired_action": "none",
            "desired_role": None,
            "reverse_port": None,
            "report_state": "registered",
            "error_code": None,
            "error_message": None,
            "receipt": None,
            "active_role": None,
            "owned_roles": {},
            "operation": None,
            "role_ever_ready": False,
        }
        if not self.path.exists():
            return default
        try:
            value = read_private_json(self.path)
        except Exception as exc:
            raise FleetAgentError("durable Fleet agent state cannot be safely read") from exc
        if value.get("schema") != _STATE_SCHEMA or value.get("node_id") != self.node_id:
            raise FleetAgentError("durable Fleet agent state identity is invalid")
        for key, default_value in default.items():
            value.setdefault(key, default_value)
        if not isinstance(value.get("owned_roles"), dict):
            raise FleetAgentError("durable Fleet role ownership ledger is invalid")
        return value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def update(self, **values: Any) -> dict[str, Any]:
        with self._lock:
            self._state.update(values)
            encoded = (json.dumps(self._state, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            atomic_write(self.path, encoded, mode=0o600)
            return copy.deepcopy(self._state)


def _load_or_create_bearer(path: Path) -> str:
    path = Path(path)
    if path.exists() or path.is_symlink():
        try:
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or (os.name == "posix" and info.st_uid != os.geteuid())
                    or info.st_nlink != 1):
                raise FleetAgentError("persisted node bearer must be a private regular file")
            token = path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise FleetAgentError("persisted node bearer cannot be read") from exc
    else:
        token = secrets.token_urlsafe(48)
        atomic_write(path, token.encode("ascii"), mode=0o600)
    if _TOKEN.fullmatch(token) is None:
        raise FleetAgentError("persisted node bearer is malformed")
    return token


class NodeAgent:
    """Register neutrally, keep management online, and apply one desired operation at a time."""

    def __init__(
        self,
        settings: FleetAgentSettings | str | Path,
        *,
        http_transport: httpx.AsyncBaseTransport | None = None,
        local_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        role_adapters: Mapping[str, RoleAdapter] | None = None,
        ssh_manager: Callable[..., Any] | None = None,
        account_provider: Callable[[], UserAccount] = current_user_account,
        source_root_provider: Callable[[], Path] = _loaded_source_root,
        inventory_provider: Callable[[], HostInventory] = collect_inventory,
        boot_id_provider: Callable[[], str] | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.account = account_provider()
        if self.account.uid == 0 or self.account.uid != os.geteuid():
            raise FleetAgentError("Fleet node agent must use the invoking non-root account")
        self.paths = UserPaths.from_account(self.account)
        self.settings = (settings if isinstance(settings, FleetAgentSettings)
                         else FleetAgentSettings.load(settings, account=self.account))
        if (self.settings.state_dir != self.paths.state_dir
                or self.settings.ca_bundle != self.paths.trust_dir / "upload-ca.pem"):
            raise FleetAgentError("Fleet agent settings are not bound to the invoking account home")
        _ensure_account_directory(self.paths.management_dir, account=self.account,
                                  label="Fleet management source path", create=False)
        _ensure_account_directory(self.paths.trust_dir, account=self.account,
                                  label="Fleet TLS trust path", create=False)
        expected_source_root = self.paths.management_dir / "releases" / self.settings.agent_version
        source_root = Path(source_root_provider())
        if source_root != expected_source_root:
            raise FleetAgentError("loaded Fleet agent source is not the pinned account-owned release")
        _ensure_account_directory(source_root, account=self.account,
                                  label="Fleet agent source path", create=False)
        self.source_root = source_root
        _require_private_user_file(self.settings.ca_bundle, uid=self.account.uid,
                                   label="Fleet TLS CA bundle")
        self.state_dir = _ensure_account_directory(
            self.paths.state_dir, account=self.account, label="Fleet agent state path", create=True,
        )
        from .user_ssh import create_asyncssh_ed25519_keypair

        self.identity = NodeIdentity(self.state_dir, runner=local_runner,
                                     key_creator=create_asyncssh_ed25519_keypair)
        self.node_id, self.reverse_public_key, self.host_public_key = self.identity.load_or_create()
        self.bearer = _load_or_create_bearer(self.settings.bearer_path)
        self.store = AgentStateStore(self.settings.agent_state_path, node_id=self.node_id)
        self.client = FleetHTTPClient(self.settings, transport=http_transport)
        self.local_runner = local_runner
        self.inventory_provider = inventory_provider
        self.boot_id_provider = boot_id_provider
        self.sleep = sleep
        self.role_adapters = dict(role_adapters or default_role_adapters(
            account=self.account, paths=self.paths, state_dir=self.state_dir,
            ca_bundle=self.settings.ca_bundle, python_executable=Path(sys.executable),
            runner=local_runner,
        ))
        self.ssh_manager = ssh_manager
        self._operation_lock = asyncio.Lock()
        self._inventory: HostInventory | None = None
        self._stop_event = asyncio.Event()

    async def _run_blocking(self, function: Callable[..., Any], *args: Any,
                            label: str = "host-operation", **kwargs: Any) -> Any:
        """Run synchronous host work and persist ambiguity if its await is cancelled.

        A running thread cannot be cancelled by asyncio. Role state is therefore marked
        unknown before cancellation escapes; the next poll inspects or drains the recorded
        ownership instead of blindly rerunning the installer.
        """
        try:
            return await asyncio.to_thread(function, *args, **kwargs)
        except asyncio.CancelledError:
            self._record_interrupted_operation(label)
            raise

    def _record_interrupted_operation(self, label: str) -> None:
        state = self.store.snapshot()
        operation = state.get("operation")
        if not isinstance(operation, dict):
            return
        values: dict[str, Any] = {}
        if label.startswith("role-install"):
            if operation.get("status") not in {"installing", "loading"}:
                return
            operation = dict(operation)
            operation.update(status="loading", error_code="operation_completion_unknown",
                             error_message="role operation was interrupted before a completion receipt was observed")
            values.update(operation=operation, report_state="loading",
                          error_code=operation["error_code"], error_message=operation["error_message"],
                          receipt=self._public_receipt(operation))
            roles = copy.deepcopy(state.get("owned_roles") or {})
            key = operation.get("owned_role_key")
            owned = roles.get(key) if isinstance(key, str) else None
            if isinstance(owned, dict):
                owned.update(status="loading", receipt=dict(operation.get("receipt") or {}))
                roles[key] = owned
                values["owned_roles"] = roles
        elif label.startswith("role-drain") or label.startswith("role-deactivate"):
            if operation.get("status") != "draining":
                return
            operation = dict(operation)
            operation.update(error_code="role_drain_completion_unknown",
                             error_message="role drain was interrupted before its completion receipt was observed")
            values.update(operation=operation, report_state="draining",
                          error_code=operation["error_code"], error_message=operation["error_message"],
                          receipt=self._public_receipt(operation))
        if values:
            self.store.update(**values)

    def _boot_id(self) -> str:
        if self.boot_id_provider is not None:
            return self.boot_id_provider()
        from ..host import read_boot_id

        return read_boot_id()

    def _inventory_now(self) -> HostInventory:
        if self._inventory is None or self._inventory.boot_id != self._boot_id():
            self._inventory = self.inventory_provider()
        if self._inventory.account != self.account:
            raise FleetAgentError("local account inventory does not match the invoking Fleet account")
        return self._inventory

    async def close(self) -> None:
        await self.client.close()

    async def _ensure_registered(self) -> bool:
        boot_id = self._boot_id()
        current = self.store.snapshot()
        if current.get("terminal_rejection") or current.get("approval") == "rejected":
            return False
        if current["registered"] and current.get("registered_boot_id") == boot_id:
            return True
        inventory = await self._run_blocking(self._inventory_now, label="inventory")
        registration = inventory.registration(
            node_id=self.node_id,
            agent_version=self.settings.agent_version,
            ssh_port=self.settings.ssh_port,
            ssh_public_key=self.reverse_public_key,
            ssh_host_public_key=self.host_public_key,
        )
        # The random bearer, stable node ID, and both key pairs were persisted before this first request.
        try:
            response = await self.client.register(self.bearer, registration)
        except FleetHTTPError as exc:
            if exc.status_code == 403:
                self.store.update(
                    registered=False,
                    terminal_rejection=True,
                    approval="rejected",
                    report_state="failed",
                    error_code="node_rejected",
                    error_message="central Fleet registration was rejected",
                )
                return False
            raise
        if response.node_id != self.node_id:
            raise FleetAgentError("Fleet registration returned another node identity")
        if response.approval == "rejected":
            self.store.update(
                registered=False,
                terminal_rejection=True,
                approval="rejected",
                desired_generation=response.desired_generation,
                report_state="failed",
                error_code="node_rejected",
                error_message="central Fleet registration was rejected",
            )
            return False
        self.store.update(
            registered=True,
            registered_boot_id=boot_id,
            approval=response.approval,
            desired_generation=max(current.get("desired_generation", 0), response.desired_generation),
            reverse_port=response.reverse_port,
            registration_time=response.registered_at,
            report_state="registered",
            error_code=None,
            error_message=None,
        )
        return True

    def _ssh_config(self, desired: DesiredStateResponse):
        from .ssh import FleetNodeSSHConfig

        if desired.reverse_port is None:
            raise FleetAgentError("approved Fleet node has no assigned SSH reverse port")
        return FleetNodeSSHConfig(
            controller_url=self.settings.controller_url,
            ca_bundle=self.settings.ca_bundle,
            bearer_file=self.settings.bearer_path,
            reverse_port=desired.reverse_port,
            management_generation=desired.generation,
            ssh_port=self.settings.ssh_port,
            relay_port=self.settings.ssh_relay_port,
            relay_host_key=self.settings.ssh_relay_host_key,
            relay_user=self.settings.ssh_relay_user,
            tunnel_identity_file=self.identity.key_path,
            host_identity_file=self.identity.host_key_path,
            admin_ssh_public_key=self.settings.admin_ssh_public_key,
            python_executable=Path(sys.executable),
        )

    async def _ensure_management_ssh(self, desired: DesiredStateResponse) -> None:
        config = self._ssh_config(desired)
        if self.ssh_manager is not None:
            value = self.ssh_manager(config, account=self.account, paths=self.paths,
                                     runner=self.local_runner)
            if asyncio.iscoroutine(value):
                await value
            return
        from .ssh import ensure_node_ssh, start_node_ssh

        await self._run_blocking(
            ensure_node_ssh, config, account=self.account, paths=self.paths,
            runner=self.local_runner,
            label="management-ssh-prepare",
        )
        await self._run_blocking(start_node_ssh, config, account=self.account, paths=self.paths,
                                 runner=self.local_runner, label="management-ssh-start")

    async def _fetch_deployment(self, desired: DesiredStateResponse) -> PrivateDeploymentEnvelope:
        envelope = await self.client.deployment(self.node_id, desired.generation, self.bearer)
        if envelope.node_id != self.node_id or envelope.generation != desired.generation:
            raise FleetAgentError("protected deployment identity or generation did not match this poll")
        if (envelope.deployment.role != desired.role
                or envelope.deployment.deployment_id != desired.deployment_id
                or not hmac.compare_digest(role_deployment_sha256(envelope.deployment), desired.deployment_sha256 or "")):
            raise FleetAgentError("protected deployment content did not match the approved descriptor")
        return envelope

    async def _prepare_role_install(
        self,
        adapter: RoleAdapter,
        deployment: Any,
        *,
        generation: int,
        owned_receipt: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        prepare = getattr(adapter, "prepare_install", None)
        if not callable(prepare):
            return {"role": deployment.role, "generation": generation,
                    "deployment_id": deployment.deployment_id,
                    "target_state": "adapter_managed"}
        value = await self._run_blocking(
            prepare, deployment, generation=generation, owned_receipt=owned_receipt,
            label="role-preflight",
        )
        if not isinstance(value, Mapping):
            raise FleetAgentError("role adapter returned an invalid read-only ownership preflight")
        intent = dict(value)
        if (intent.get("role") != deployment.role
                or intent.get("generation") != generation
                or intent.get("deployment_id") != deployment.deployment_id):
            raise FleetAgentError("role adapter ownership preflight did not match the protected assignment")
        return intent

    async def tick_once(self) -> dict[str, Any]:
        """Run one register/poll/apply/ACK cycle; safe for fixtures and deterministic tests."""
        async with self._operation_lock:
            if not await self._ensure_registered():
                return self.snapshot()
            desired = await self.client.desired_state(self.node_id, self.bearer)
            state = self.store.snapshot()
            self.store.update(
                approval=desired.approval,
                desired_generation=desired.generation,
                desired_action=desired.action,
                desired_role=desired.role,
                reverse_port=desired.reverse_port,
            )
            if desired.approval == "rejected":
                self.store.update(terminal_rejection=True, registered=False, report_state="failed",
                                  error_code="node_rejected",
                                  error_message="central Fleet registration was rejected", receipt=None)
                return self.snapshot()
            if desired.approval == "pending":
                self.store.update(report_state="registered", error_code=None, error_message=None, receipt=None)
                return self.snapshot()
            try:
                # Admin approval starts the durable management path before any role adapter runs.
                await self._ensure_management_ssh(desired)
            except Exception:
                self.store.update(report_state="failed", error_code="management_ssh_unavailable",
                                  error_message="owned Fleet management SSH could not be started")
                return self.snapshot()
            if desired.action == "none":
                active = state.get("active_role")
                if active and state.get("report_state") == "ready":
                    return self.snapshot()
                self.store.update(report_state="registered", error_code=None, error_message=None, receipt=None)
                return self.snapshot()
            if desired.action == "deactivate":
                return await self._apply_deactivation(desired, state)
            return await self._apply_install(desired, state)

    @staticmethod
    def _owned_role_key(role: str | None, deployment_id: str | None) -> str:
        if role not in {"worker", "learner"} or not isinstance(deployment_id, str):
            raise FleetAgentError("role ownership key is incomplete")
        return f"{role}:{deployment_id}"

    @classmethod
    def _mark_owned(cls, state: Mapping[str, Any], operation: Mapping[str, Any],
                    status: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
        owned_roles = copy.deepcopy(state.get("owned_roles") or {})
        key = operation.get("owned_role_key") or cls._owned_role_key(
            operation.get("role"), operation.get("deployment_id"))
        prior = owned_roles.get(key) if isinstance(owned_roles.get(key), dict) else {}
        owned_roles[key] = {
            "role": operation.get("role"),
            "generation": operation.get("generation"),
            "deployment_id": operation.get("deployment_id"),
            "status": status,
            "receipt": dict(receipt),
            "previous_receipt": prior.get("receipt"),
        }
        return owned_roles

    @classmethod
    def _owned_role_candidates(
        cls, state: Mapping[str, Any], *, exclude_generation: int | None = None,
    ) -> list[dict[str, Any]]:
        owned_roles = state.get("owned_roles")
        candidates: dict[str, dict[str, Any]] = {}
        if isinstance(owned_roles, dict):
            for key, entry in owned_roles.items():
                if (isinstance(key, str) and isinstance(entry, dict)
                        and entry.get("status") != "stopped"
                        and entry.get("role") in {"worker", "learner"}
                        and isinstance(entry.get("receipt"), dict)):
                    candidates[key] = copy.deepcopy(entry)
        active = state.get("active_role")
        if isinstance(active, dict) and active.get("role") in {"worker", "learner"}:
            try:
                key = cls._owned_role_key(active.get("role"), active.get("deployment_id"))
            except FleetAgentError:
                key = "active_role:unknown"
            if key not in candidates and isinstance(active.get("receipt"), dict):
                candidates[key] = {**copy.deepcopy(active), "status": "ready", "receipt": active["receipt"]}
        # A pre-ledger version may have persisted an installer intent without its role receipt.
        # Preserve that ambiguity as a candidate so a later generation cannot overwrite it.
        operation = state.get("operation")
        if (isinstance(operation, dict) and operation.get("ownership_proven") is not False
                and operation.get("status") not in {"ready", "deactivated"}
                and operation.get("role") in {"worker", "learner"}
                and operation.get("generation") != exclude_generation):
            try:
                key = cls._owned_role_key(operation.get("role"), operation.get("deployment_id"))
            except FleetAgentError:
                key = "operation:unknown"
            known = owned_roles.get(key) if isinstance(owned_roles, dict) else None
            if (key not in candidates and not (isinstance(known, dict) and known.get("status") == "stopped")
                    and isinstance(operation.get("receipt"), dict)):
                candidates[key] = {
                    "role": operation.get("role"),
                    "generation": operation.get("generation"),
                    "deployment_id": operation.get("deployment_id"),
                    "status": operation.get("status"),
                    "receipt": copy.deepcopy(operation["receipt"]),
                }
        return list(candidates.values())

    async def _apply_install(self, desired: DesiredStateResponse, state: dict[str, Any]) -> dict[str, Any]:
        operation = state.get("operation")
        same_generation = isinstance(operation, dict) and operation.get("generation") == desired.generation
        if same_generation:
            if operation.get("status") == "ready":
                receipt = operation.get("receipt")
                if (operation.get("ownership_proven") is not True
                        or not isinstance(receipt, Mapping)
                        or not self._ready_receipt_complete(operation, receipt)):
                    return await self._finish_operation(operation, RoleResult(
                        "loading", dict(receipt) if isinstance(receipt, Mapping) else {},
                        "role_readiness_unconfirmed",
                        "saved role readiness or ownership receipt is incomplete"))
                if operation.get("ack_accepted") is True:
                    return self.snapshot()
                await self._send_ack(operation)
                return self.snapshot()
            if operation.get("status") in {"failed", "deactivated"}:
                if operation.get("ack_accepted") is True:
                    return self.snapshot()
                await self._send_ack(operation)
                return self.snapshot()
            if operation.get("status") == "loading":
                await self._resume_nonterminal_install(operation, desired)
                return self.snapshot()
            if operation.get("status") == "draining":
                return await self._drain_previous_role(desired, state)
            if operation.get("status") == "installing":
                # A process restart during a role install has ambiguous completion. Inspect exact
                # ownership once; if not ready, record a terminal failure rather than rerun it.
                role = operation.get("role")
                adapter = self.role_adapters.get(role)
                if adapter is not None and isinstance(operation.get("receipt"), dict):
                    inspected = await self._run_blocking(adapter.inspect, operation["receipt"],
                                                         label="role-recovery-inspect")
                    if inspected.status == "ready":
                        return await self._finish_operation(operation, inspected)
                failed = RoleResult("failed", operation.get("receipt") or {}, "operation_interrupted_unknown",
                                    "role installer stopped before a durable completion receipt")
                return await self._finish_operation(operation, failed)
            # Unknown persisted states are held fail-closed; they never authorize a new install.
            return self.snapshot()
        envelope = await self._fetch_deployment(desired)
        previous = self._owned_role_candidates(state)
        if previous:
            operation = {
                "generation": desired.generation,
                "role": desired.role,
                "deployment_id": desired.deployment_id,
                "deployment_sha256": desired.deployment_sha256,
                "deployment": private_deployment_to_dict(envelope),
                "status": "draining",
                "receipt": {},
                "ack_accepted": False,
            }
            self.store.update(operation=operation, report_state="draining", error_code=None,
                              error_message=None, receipt=self._public_receipt(operation))
            return await self._drain_previous_role(desired, self.store.snapshot())
        operation = {
            "generation": desired.generation,
            "role": desired.role,
            "deployment_id": desired.deployment_id,
            "deployment_sha256": desired.deployment_sha256,
            "deployment": private_deployment_to_dict(envelope),
            "approved_artifact_sha256": self._deployment_artifact_sha256(envelope.deployment),
            "status": "installing",
            "receipt": {"role": desired.role, "generation": desired.generation,
                        "deployment_id": desired.deployment_id},
            "ack_accepted": False,
        }
        operation["owned_role_key"] = self._owned_role_key(desired.role, desired.deployment_id)
        owned_roles = state.get("owned_roles", {})
        reusable_receipt = owned_roles.get(operation["owned_role_key"])
        operation["owned_receipt"] = (
            reusable_receipt.get("receipt")
            if isinstance(reusable_receipt, dict) and isinstance(reusable_receipt.get("receipt"), dict)
            else None
        )
        adapter = self.role_adapters.get(desired.role or "")
        if adapter is None:
            operation["receipt"] = {"role": desired.role, "generation": desired.generation,
                                    "deployment_id": desired.deployment_id}
            operation["ownership_proven"] = False
            self.store.update(operation=operation, report_state="failed", error_code="role_adapter_unavailable",
                              error_message="assigned role has no local installer adapter", receipt=self._public_receipt(operation))
            return await self._finish_operation(operation, RoleResult(
                "failed", operation["receipt"], "role_adapter_unavailable",
                "assigned role has no local installer adapter", owned=False))
        try:
            install_intent = await self._prepare_role_install(
                adapter, envelope.deployment, generation=desired.generation,
                owned_receipt=operation.get("owned_receipt"),
            )
        except Exception as exc:
            operation["status"] = "failed"
            operation["ownership_proven"] = False
            operation["error_code"] = "role_ownership_preflight_failed"
            operation["error_message"] = "existing role state is not owned or safe to reuse"
            operation["receipt"] = {"role": desired.role, "generation": desired.generation,
                                    "deployment_id": desired.deployment_id}
            self.store.update(operation=operation, report_state="failed", error_code=operation["error_code"],
                              error_message=operation["error_message"], receipt=self._public_receipt(operation))
            return await self._finish_operation(operation, RoleResult(
                "failed", operation["receipt"], operation["error_code"],
                operation["error_message"], owned=False))
        operation["install_intent"] = install_intent
        operation["receipt"] = install_intent
        operation["ownership_proven"] = True
        self.store.update(operation=operation, report_state="installing", error_code=None,
                          error_message=None, receipt=self._public_receipt(operation),
                          owned_roles=self._mark_owned(state, operation, "installing", operation["receipt"]))
        inventory = await self._run_blocking(self._inventory_now, label="inventory")
        try:
            result = await self._run_blocking(
                adapter.install,
                envelope.deployment,
                generation=desired.generation,
                inventory=inventory,
                newly_managed=not bool(state.get("role_ever_ready")),
                owned_receipt=operation.get("owned_receipt"),
                install_intent=install_intent,
                label="role-install",
            )
        except Exception as exc:
            result = RoleResult("failed", operation["receipt"], "role_installer_error",
                                f"role installer failed ({type(exc).__name__})")
        return await self._finish_operation(operation, result)

    async def _resume_nonterminal_install(
        self, operation: dict[str, Any], desired: DesiredStateResponse,
    ) -> dict[str, Any]:
        adapter = self.role_adapters.get(operation.get("role", ""))
        if adapter is None:
            return self.snapshot()
        inspected = await self._run_blocking(adapter.inspect, operation.get("receipt") or {},
                                             label="role-inspect")
        await self._finish_operation(operation, inspected)
        return self.snapshot()

    async def _drain_previous_role(self, desired: DesiredStateResponse, state: dict[str, Any]) -> dict[str, Any]:
        operation = state.get("operation") or {}
        previous_roles = self._owned_role_candidates(state, exclude_generation=desired.generation)
        if not previous_roles:
            saved = operation.get("deployment")
            if isinstance(saved, dict):
                envelope = PrivateDeploymentEnvelope.model_validate(saved)
            else:
                envelope = await self._fetch_deployment(desired)
            operation = dict(operation)
            operation.update(
                generation=desired.generation,
                role=envelope.deployment.role,
                deployment_id=envelope.deployment.deployment_id,
                deployment_sha256=desired.deployment_sha256,
                deployment=private_deployment_to_dict(envelope),
                approved_artifact_sha256=self._deployment_artifact_sha256(envelope.deployment),
                status="installing",
                receipt={"role": envelope.deployment.role, "generation": desired.generation,
                         "deployment_id": envelope.deployment.deployment_id},
                ack_accepted=False,
            )
            operation["owned_role_key"] = self._owned_role_key(desired.role, desired.deployment_id)
            owned_roles = state.get("owned_roles", {})
            reusable = owned_roles.get(operation["owned_role_key"]) if isinstance(owned_roles, dict) else None
            operation["owned_receipt"] = (
                reusable.get("receipt") if isinstance(reusable, dict)
                and isinstance(reusable.get("receipt"), dict) else None
            )
            adapter = self.role_adapters.get(desired.role or "")
            if adapter is None:
                operation["ownership_proven"] = False
                self.store.update(operation=operation, report_state="failed")
                return await self._finish_operation(operation, RoleResult(
                    "failed", operation["receipt"], "role_adapter_unavailable",
                    "assigned role has no local installer adapter", owned=False))
            try:
                install_intent = await self._prepare_role_install(
                    adapter, envelope.deployment, generation=desired.generation,
                    owned_receipt=operation.get("owned_receipt"),
                )
            except Exception:
                operation.update(status="failed", ownership_proven=False,
                                 error_code="role_ownership_preflight_failed",
                                 error_message="existing role state is not owned or safe to reuse")
                self.store.update(operation=operation, report_state="failed",
                                  error_code=operation["error_code"], error_message=operation["error_message"])
                return await self._finish_operation(operation, RoleResult(
                    "failed", operation["receipt"], operation["error_code"],
                    operation["error_message"], owned=False))
            operation["install_intent"] = install_intent
            operation["receipt"] = install_intent
            operation["ownership_proven"] = True
            self.store.update(operation=operation,
                              owned_roles=self._mark_owned(state, operation, "installing", operation["receipt"]),
                              report_state="installing")
            inventory = await self._run_blocking(self._inventory_now, label="inventory")
            result = await self._run_blocking(
                adapter.install, envelope.deployment,
                generation=desired.generation, inventory=inventory,
                newly_managed=not bool(state.get("role_ever_ready")),
                owned_receipt=operation.get("owned_receipt"),
                install_intent=install_intent,
                label="role-install",
            )
            return await self._finish_operation(operation, result)
        owned_roles = state.get("owned_roles", {})
        for previous in previous_roles:
            adapter = self.role_adapters.get(previous.get("role", ""))
            if adapter is None or not isinstance(previous.get("receipt"), dict):
                return self._record_drain_block(operation, StopResult(
                    "blocked_busy", dict(previous), "previous_role_ownership_unknown",
                    "cannot prove ownership of the previous role"))
            stop = await self._run_blocking(adapter.stop_owned, previous["receipt"],
                                            label="role-drain")
            if stop.status != "stopped":
                return await self._record_drain_ack(operation, stop)
            key = self._owned_role_key(previous.get("role"), previous.get("deployment_id"))
            entry = owned_roles.get(key) if isinstance(owned_roles, dict) else None
            if not isinstance(entry, dict):
                entry = dict(previous)
            entry.update(status="stopped", stop_receipt=dict(stop.receipt))
            owned_roles[key] = entry
            active = state.get("active_role")
            if isinstance(active, dict) and active.get("role") == previous.get("role"):
                state["active_role"] = None
            self.store.update(owned_roles=owned_roles, active_role=state.get("active_role"))
        envelope = PrivateDeploymentEnvelope.model_validate(operation["deployment"])
        adapter = self.role_adapters.get(envelope.deployment.role)
        if adapter is None:
            return await self._finish_operation(operation, RoleResult(
                "failed", operation.get("receipt") or {}, "role_adapter_unavailable",
                "assigned role has no local installer adapter"))
        operation = dict(operation)
        operation.update(status="installing", receipt={"role": envelope.deployment.role,
                                                        "generation": desired.generation,
                                                        "deployment_id": envelope.deployment.deployment_id})
        operation["owned_role_key"] = self._owned_role_key(envelope.deployment.role,
                                                            envelope.deployment.deployment_id)
        owned_roles = self.store.snapshot().get("owned_roles", {})
        reusable = owned_roles.get(operation["owned_role_key"]) if isinstance(owned_roles, dict) else None
        operation["owned_receipt"] = (
            reusable.get("receipt") if isinstance(reusable, dict)
            and isinstance(reusable.get("receipt"), dict) else None
        )
        try:
            install_intent = await self._prepare_role_install(
                adapter, envelope.deployment, generation=desired.generation,
                owned_receipt=operation.get("owned_receipt"),
            )
        except Exception:
            operation.update(status="failed", ownership_proven=False,
                             error_code="role_ownership_preflight_failed",
                             error_message="existing role state is not owned or safe to reuse")
            self.store.update(operation=operation, report_state="failed",
                              error_code=operation["error_code"], error_message=operation["error_message"])
            return await self._finish_operation(operation, RoleResult(
                "failed", operation["receipt"], operation["error_code"],
                operation["error_message"], owned=False))
        operation["install_intent"] = install_intent
        operation["receipt"] = install_intent
        operation["ownership_proven"] = True
        owned_roles = self._mark_owned(self.store.snapshot(), operation, "installing", operation["receipt"])
        self.store.update(operation=operation, owned_roles=owned_roles,
                          report_state="installing", error_code=None, error_message=None)
        inventory = await self._run_blocking(self._inventory_now, label="inventory")
        result = await self._run_blocking(
            adapter.install, envelope.deployment,
            generation=desired.generation,
            inventory=inventory,
            newly_managed=not bool(self.store.snapshot().get("role_ever_ready")),
            owned_receipt=operation.get("owned_receipt"),
            install_intent=install_intent,
            label="role-install",
        )
        return await self._finish_operation(operation, result)

    def _record_drain_block(self, operation: dict[str, Any], stop: StopResult) -> dict[str, Any]:
        operation = dict(operation)
        operation.update(status="draining", receipt=stop.receipt or operation.get("receipt"),
                         error_code=stop.error_code, error_message=stop.error_message)
        self.store.update(operation=operation, report_state="draining",
                          error_code=stop.error_code or "role_drain_incomplete",
                          error_message=stop.error_message or "previous role has not safely drained",
                          receipt=self._public_receipt(operation))
        return self.snapshot()

    async def _record_drain_ack(self, operation: dict[str, Any], stop: StopResult) -> dict[str, Any]:
        snapshot = self._record_drain_block(operation, stop)
        current = self.store.snapshot().get("operation")
        if isinstance(current, dict):
            await self._send_ack(current, override_status="draining")
        return snapshot

    async def _apply_deactivation(self, desired: DesiredStateResponse, state: dict[str, Any]) -> dict[str, Any]:
        operation = state.get("operation")
        if isinstance(operation, dict) and operation.get("generation") == desired.generation:
            if operation.get("status") == "deactivated":
                await self._send_ack(operation)
                return self.snapshot()
            if operation.get("status") == "draining":
                return await self._drain_for_deactivation(desired, state)
        operation = {
            "generation": desired.generation,
            "role": (state.get("active_role") or {}).get("role"),
            "deployment_id": (state.get("active_role") or {}).get("deployment_id"),
            "status": "draining",
            "receipt": {"generation": desired.generation, "action": "deactivate"},
            "ack_accepted": False,
        }
        self.store.update(operation=operation, report_state="draining", error_code=None, error_message=None)
        return await self._drain_for_deactivation(desired, self.store.snapshot())

    async def _drain_for_deactivation(self, desired: DesiredStateResponse, state: dict[str, Any]) -> dict[str, Any]:
        operation = dict(state.get("operation") or {})
        previous_roles = self._owned_role_candidates(state, exclude_generation=desired.generation)
        if not previous_roles:
            deactivated = {"generation": desired.generation, "action": "deactivate"}
            operation.update(status="deactivated", role=None, receipt=deactivated,
                             error_code=None, error_message=None)
            self.store.update(operation=operation, active_role=None, report_state="deactivated",
                              receipt=deactivated, error_code=None, error_message=None)
            await self._send_ack(operation, override_status="deactivated")
            return self.snapshot()
        owned_roles = state.get("owned_roles", {})
        for previous in previous_roles:
            adapter = self.role_adapters.get(previous.get("role", ""))
            if adapter is None or not isinstance(previous.get("receipt"), dict):
                return await self._record_drain_ack(operation, StopResult(
                    "blocked_busy", dict(previous), "role_ownership_unknown",
                    "cannot prove ownership of the active role"))
            stop = await self._run_blocking(adapter.stop_owned, previous["receipt"],
                                            label="role-deactivate")
            if stop.status != "stopped":
                return await self._record_drain_ack(operation, stop)
            key = self._owned_role_key(previous.get("role"), previous.get("deployment_id"))
            entry = owned_roles.get(key) if isinstance(owned_roles, dict) else None
            if not isinstance(entry, dict):
                entry = dict(previous)
            entry.update(status="stopped", stop_receipt=dict(stop.receipt))
            owned_roles[key] = entry
        deactivated = {"generation": desired.generation, "action": "deactivate",
                       "stopped_roles": sorted({item.get("role") for item in previous_roles})}
        self.store.update(owned_roles=owned_roles, active_role=None)
        operation.update(status="deactivated", role=None, receipt=deactivated,
                         error_code=None, error_message=None)
        self.store.update(operation=operation, report_state="deactivated", receipt=deactivated,
                          error_code=None, error_message=None)
        await self._send_ack(operation, override_status="deactivated")
        return self.snapshot()

    def _ready_receipt_complete(self, operation: Mapping[str, Any], receipt: Mapping[str, Any]) -> bool:
        """Check the shared adapter receipt contract; adapters own role-specific health probes."""
        role = operation.get("role")
        if (receipt.get("role") != role
                or receipt.get("generation") != operation.get("generation")
                or receipt.get("deployment_id") != operation.get("deployment_id")
                or receipt.get("readiness_verified") is not True
                or receipt.get("account_uid") != self.account.uid
                or receipt.get("account_gid") != self.account.gid):
            return False
        docker = receipt.get("docker_access")
        if not isinstance(docker, Mapping):
            return False
        host_uid, host_gid = docker.get("host_uid"), docker.get("host_gid")
        container_uid, container_gid = docker.get("container_uid"), docker.get("container_gid")
        rootless = docker.get("rootless")
        if (docker.get("uid_mapping_verified") is not True
                or type(host_uid) is not int or type(host_gid) is not int
                or host_uid != self.account.uid or host_gid != self.account.gid
                or type(container_uid) is not int or container_uid < 0
                or type(container_gid) is not int or container_gid < 0
                or not isinstance(rootless, bool)
                or (not rootless and (container_uid != self.account.uid or container_gid != self.account.gid))
                or not all(isinstance(docker.get(field), str) and docker[field]
                           for field in ("context", "endpoint", "daemon_id"))):
            return False
        services = receipt.get("user_services")
        if not isinstance(services, list) or not services:
            return False
        for service in services:
            if not isinstance(service, Mapping) or service.get("active") is not True:
                return False
            unit, path = service.get("unit"), service.get("path")
            if (not isinstance(unit, str) or not unit.endswith(".service")
                    or not isinstance(path, str)):
                return False
            unit_path = Path(path)
            if not unit_path.is_absolute() or self.paths.systemd_user_dir not in unit_path.parents:
                return False
        artifact_sha256 = receipt.get("artifact_sha256")
        if not isinstance(artifact_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None:
            return False
        expected_sha256 = operation.get("approved_artifact_sha256")
        if not isinstance(expected_sha256, str):
            serialized = operation.get("deployment")
            deployment = serialized.get("deployment") if isinstance(serialized, Mapping) else None
            if not isinstance(deployment, Mapping):
                return False
            artifact_field = "source_bundle" if role == "worker" else "private_bundle"
            artifact = deployment.get(artifact_field)
            expected_sha256 = artifact.get("sha256") if isinstance(artifact, Mapping) else None
        return isinstance(expected_sha256, str) and hmac.compare_digest(artifact_sha256, expected_sha256)

    @staticmethod
    def _deployment_artifact_sha256(deployment: Any) -> str:
        field = "source_bundle" if deployment.role == "worker" else "private_bundle"
        artifact = getattr(deployment, field, None)
        digest = getattr(artifact, "sha256", None)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise FleetAgentError("approved Fleet deployment has no pinned role artifact")
        return digest

    async def _finish_operation(self, operation: dict[str, Any], result: RoleResult) -> dict[str, Any]:
        if result.status not in ("ready", "loading", "failed"):
            result = RoleResult("failed", dict(result.receipt), "role_status_unsupported",
                                "role adapter returned a status unsupported by user-owned Fleet",
                                owned=result.owned)
        elif result.status == "ready" and not result.owned:
            result = RoleResult("failed", dict(result.receipt), "role_ownership_unconfirmed",
                                "role adapter could not prove ownership of the installed resources",
                                owned=False)
        elif result.status == "ready" and not self._ready_receipt_complete(operation, result.receipt):
            result = RoleResult("loading", dict(result.receipt), "role_readiness_unconfirmed",
                                "role startup or Docker readiness receipt is incomplete",
                                owned=result.owned)
        operation = dict(operation)
        operation.update(
            status=result.status,
            receipt=dict(result.receipt),
            error_code=result.error_code,
            error_message=result.error_message,
            inventory_fingerprint=self._inventory_fingerprint(self._inventory_now()),
            registered_boot_id=self._boot_id(),
            ack_accepted=False,
        )
        values: dict[str, Any] = {
            "operation": operation,
            "report_state": result.status,
            "error_code": result.error_code,
            "error_message": result.error_message,
            "receipt": self._public_receipt(operation),
        }
        role = operation.get("role")
        deployment_id = operation.get("deployment_id")
        if role in {"worker", "learner"} and isinstance(deployment_id, str):
            key = operation.get("owned_role_key") or self._owned_role_key(role, deployment_id)
            operation["owned_role_key"] = key
            values["operation"] = operation
            current_state = self.store.snapshot()
            owned_roles = copy.deepcopy(current_state.get("owned_roles") or {})
            if not result.owned:
                if not operation.get("owned_receipt"):
                    owned_roles.pop(key, None)
                operation["ownership_proven"] = False
                # Keep any prior stopped Fleet receipt unchanged; this failed preflight
                # did not acquire or mutate the target state.
            else:
                operation["ownership_proven"] = True
                owned_roles = self._mark_owned(current_state, operation, result.status, result.receipt)
            values["operation"] = operation
            values["owned_roles"] = owned_roles
        if result.status == "ready":
            values["active_role"] = {
                "role": operation.get("role"),
                "generation": operation.get("generation"),
                "deployment_id": operation.get("deployment_id"),
                "receipt": dict(result.receipt),
            }
            values["role_ever_ready"] = True
        else:
            active = self.store.snapshot().get("active_role")
            if (isinstance(active, dict)
                    and active.get("role") == operation.get("role")
                    and active.get("generation") == operation.get("generation")
                    and active.get("deployment_id") == operation.get("deployment_id")):
                values["active_role"] = None
        self.store.update(**values)
        await self._send_ack(operation)
        return self.snapshot()

    async def _send_ack(self, operation: dict[str, Any], *, override_status: str | None = None) -> bool:
        status = override_status or operation.get("status")
        if status not in {"installing", "loading", "draining", "ready", "failed", "deactivated"}:
            return False
        role = operation.get("role")
        if status == "deactivated":
            role = None
        receipt = self._public_receipt(operation)
        try:
            payload = ApplyAckRequest(
                generation=operation["generation"],
                status=status,
                role=role,
                error_code=operation.get("error_code"),
                error_message=operation.get("error_message"),
                receipt=receipt,
            )
            accepted = await self.client.acknowledge(self.node_id, self.bearer, payload)
        except Exception:
            # Retain terminal receipts and secrets locally; the next poll retries the same ACK
            # without re-running a completed installer.
            return False
        if accepted.node_id != self.node_id or accepted.generation != operation["generation"]:
            return False
        current = self.store.snapshot().get("operation")
        if isinstance(current, dict) and current.get("generation") == operation.get("generation"):
            current["ack_accepted"] = True
            if status in {"ready", "failed", "deactivated"}:
                current.pop("deployment", None)
            self.store.update(operation=current)
        return True

    @staticmethod
    def _inventory_fingerprint(inventory: HostInventory) -> str:
        value = json.dumps([inventory.boot_id, inventory.architecture,
                            sorted(gpu.uuid for gpu in inventory.gpus)], separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _public_receipt(operation: Mapping[str, Any]) -> dict[str, Any]:
        receipt = operation.get("receipt")
        if not isinstance(receipt, dict):
            receipt = {}
        allowed = {
            "role", "generation", "deployment_id", "source_bundle_sha256",
            "bundle_sha256", "worker_ready", "user_units_active", "model_verified", "docker",
            "docker_access", "learner_runtime_ready", "learner_host_state", "learner_service_active",
            "learner_training_idle", "install_root", "profile_sha256", "image_id", "container_user",
            "learner_service_unit_path", "training_node_id", "gpu_uuid", "user_service_units",
            "learner_service_unit", "worker_health", "node_heartbeat",
            "docker_mapping", "image_verified", "management_python", "management_source_root",
            "preserved_state", "action", "status", "heartbeat_ready", "docker_context",
            "readiness_verified", "account_uid", "account_gid", "user_services", "artifact_sha256",
        }
        public = {key: receipt[key] for key in allowed if key in receipt}
        public.update(generation=operation.get("generation"), status=operation.get("status"))
        role = operation.get("role")
        if role in {"worker", "learner"}:
            public.setdefault("role", role)
        deployment_id = operation.get("deployment_id")
        if isinstance(deployment_id, str):
            public.setdefault("deployment_id", deployment_id)
        return public

    def _heartbeat_payload(self) -> NodeHeartbeat | None:
        state = self.store.snapshot()
        if not state.get("registered") or state.get("registered_boot_id") != self._boot_id():
            return None
        operation = state.get("operation") or {}
        active = state.get("active_role") or {}
        role = operation.get("role") or active.get("role")
        receipt = state.get("receipt")
        try:
            return NodeHeartbeat(
                boot_id=self._boot_id(),
                agent_version=self.settings.agent_version,
                state=state.get("report_state", "registered"),
                observed_generation=int(state.get("desired_generation", 0)),
                role=role if role in {"worker", "learner"} else None,
                error_code=state.get("error_code"),
                error_message=state.get("error_message"),
                receipt=receipt if isinstance(receipt, dict) else None,
            )
        except Exception:
            return None

    async def _heartbeat_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            payload = self._heartbeat_payload()
            if payload is not None and not self.store.snapshot().get("terminal_rejection"):
                try:
                    await self.client.heartbeat(self.node_id, self.bearer, payload)
                    notify_watchdog()
                except FleetHTTPError as exc:
                    if exc.status_code in {401, 404}:
                        self.store.update(registered=False)
                except FleetAgentError:
                    pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.settings.heartbeat_interval_seconds)
            except TimeoutError:
                continue

    def snapshot(self) -> dict[str, Any]:
        state = self.store.snapshot()
        return {
            "node_id": self.node_id,
            "approval": state.get("approval"),
            "generation": state.get("desired_generation"),
            "action": state.get("desired_action"),
            "role": state.get("desired_role"),
            "state": state.get("report_state"),
            "error_code": state.get("error_code"),
            "ack_accepted": (state.get("operation") or {}).get("ack_accepted"),
        }

    async def run(self, *, stop_event: asyncio.Event | None = None) -> None:
        stop_event = stop_event or self._stop_event
        notify_ready()
        heartbeat = asyncio.create_task(self._heartbeat_loop(stop_event), name="qwen-fleet-heartbeat")
        try:
            while not stop_event.is_set():
                try:
                    await self.tick_once()
                except (FleetAgentError, FleetHostError) as exc:
                    logging.getLogger(__name__).warning("Fleet control cycle failed (%s)", type(exc).__name__)
                except Exception as exc:  # Keep the public log secret-safe and the service alive.
                    logging.getLogger(__name__).error("Fleet control cycle failed (%s)", type(exc).__name__)
                if self.store.snapshot().get("terminal_rejection"):
                    stop_event.set()
                    break
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.settings.poll_interval_seconds)
                except TimeoutError:
                    continue
        finally:
            stop_event.set()
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            await self.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-fleet-node-agent")
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--once", action="store_true",
                        help="run one register/poll/apply/ACK cycle and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        agent = NodeAgent(args.settings)
    except (FleetAgentError, FleetHostError, OSError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__}, sort_keys=True))
        return 2

    async def execute() -> int:
        if args.once:
            try:
                result = await agent.tick_once()
                print(json.dumps(result, sort_keys=True, separators=(",", ":")))
                await agent.close()
                return 0 if result.get("state") not in {"failed"} else 1
            except (FleetAgentError, FleetHostError, OSError, ValueError) as exc:
                print(json.dumps({"error": type(exc).__name__}, sort_keys=True))
                await agent.close()
                return 1
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, stop_event.set)
        await agent.run(stop_event=stop_event)
        return 0

    return asyncio.run(execute())


if __name__ == "__main__":
    raise SystemExit(main())
