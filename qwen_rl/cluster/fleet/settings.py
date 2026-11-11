"""Private Fleet server settings and bounded control-plane defaults."""
from __future__ import annotations

import ipaddress
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictInt, StrictStr, model_validator
from urllib.parse import urlsplit

from ..identity import canonical_public_key
from ..settings import load_settings_json


class FleetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    state_dir: StrictStr = Field(min_length=1, max_length=4096)
    admin_token: SecretStr = Field(min_length=32, max_length=512, repr=False)
    ssh_admin_identity_file: StrictStr = Field(min_length=1, max_length=4096)
    ssh_relay_user: StrictStr = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    ssh_relay_host_key: StrictStr = Field(min_length=1, max_length=2000)
    admin_ssh_public_key: StrictStr = Field(min_length=1, max_length=2000)
    public_origin: StrictStr | None = Field(default=None, max_length=2048)
    public_upload_dir: StrictStr | None = Field(default=None, max_length=4096)
    heartbeat_ttl_seconds: StrictInt = Field(default=60, ge=1, le=86400)
    max_registered_nodes: StrictInt = Field(default=1000, ge=1, le=100000)
    max_pending_nodes: StrictInt = Field(default=100, ge=1, le=10000)
    max_rejected_tombstones: StrictInt = Field(default=1000, ge=1, le=100000)
    registration_rate_per_minute: StrictInt = Field(default=120, ge=50, le=10000)
    reverse_port_start: StrictInt = Field(default=24000, ge=1024, le=65535)
    reverse_port_end: StrictInt = Field(default=34000, ge=1024, le=65535)
    ssh_relay_host: StrictStr = Field(default="127.0.0.1", min_length=1, max_length=64)
    ssh_relay_port: StrictInt = Field(default=22230, ge=1024, le=65535)
    api_host: StrictStr = Field(default="127.0.0.1", min_length=1, max_length=64)
    api_port: StrictInt = Field(default=18138, ge=1, le=65535)

    @model_validator(mode="after")
    def validate_settings(self) -> "FleetSettings":
        state = Path(self.state_dir).expanduser()
        if not state.is_absolute() or "\x00" in self.state_dir or "\n" in self.state_dir:
            raise ValueError("state_dir must be an absolute, single-line path")
        resolved = state.resolve()
        if resolved in {Path("/"), Path("/etc")} or Path("/etc") in resolved.parents:
            raise ValueError("state_dir cannot be / or inside /etc")
        identity = Path(self.ssh_admin_identity_file).expanduser()
        if not identity.is_absolute() or any(c in self.ssh_admin_identity_file for c in "\r\n\x00"):
            raise ValueError("ssh_admin_identity_file must be an absolute, single-line path")
        if self.ssh_relay_user == "root":
            raise ValueError("ssh_relay_user must be the non-root Fleet service account")
        if self.public_origin is not None:
            try:
                parsed_origin = urlsplit(self.public_origin)
                parsed_origin.port
            except ValueError as exc:
                raise ValueError("public_origin must be a valid HTTPS origin") from exc
            if (parsed_origin.scheme != "https" or not parsed_origin.hostname
                    or parsed_origin.username or parsed_origin.password or parsed_origin.path not in {"", "/"}
                    or parsed_origin.query or parsed_origin.fragment or any(c.isspace() for c in self.public_origin)):
                raise ValueError("public_origin must be an HTTPS origin without credentials or path")
        if self.public_upload_dir is not None:
            upload = Path(self.public_upload_dir).expanduser()
            if (not upload.is_absolute() or any(c in self.public_upload_dir for c in "\r\n\x00")):
                raise ValueError("public_upload_dir must be an absolute, single-line path")
            upload_root = upload.resolve()
            if (resolved == upload_root or resolved.is_relative_to(upload_root) or upload_root.is_relative_to(resolved)):
                raise ValueError("Fleet state must be separate from the entire public upload tree")
            identity_path = identity.resolve()
            if identity_path == upload_root or identity_path.is_relative_to(upload_root):
                raise ValueError("Fleet SSH private identity must be outside the public upload tree")
        if self.max_pending_nodes > self.max_registered_nodes:
            raise ValueError("max_pending_nodes cannot exceed max_registered_nodes")
        if self.reverse_port_start > self.reverse_port_end:
            raise ValueError("reverse SSH port range is inverted")
        if self.reverse_port_end - self.reverse_port_start + 1 < self.max_registered_nodes:
            raise ValueError("reverse SSH port range must cover max_registered_nodes")
        for label, value in (("api_host", self.api_host), ("ssh_relay_host", self.ssh_relay_host)):
            try:
                address = ipaddress.ip_address(value)
            except ValueError as exc:
                raise ValueError(f"{label} must be a numeric loopback address") from exc
            if not address.is_loopback:
                raise ValueError(f"{label} must be a loopback address")
        if self.admin_token.get_secret_value().strip() != self.admin_token.get_secret_value() or any(
            c.isspace() or c == "\x00" for c in self.admin_token.get_secret_value()
        ):
            raise ValueError("admin_token must be a nonempty single-line bearer")
        for label, value in (("ssh_relay_host_key", self.ssh_relay_host_key),
                             ("admin_ssh_public_key", self.admin_ssh_public_key)):
            try:
                canonical = canonical_public_key(value)
                if canonical.split()[0] != "ssh-ed25519":
                    raise ValueError("not Ed25519")
            except ValueError as exc:
                raise ValueError(f"{label} must be a valid Ed25519 OpenSSH public key") from exc
        return self

    @property
    def private_artifacts_dir(self) -> Path:
        return Path(self.state_dir).expanduser().resolve() / "private-artifacts"

    def private_artifact_url(self, sha256: str) -> str:
        if self.public_origin is None or self.public_upload_dir is None:
            raise ValueError("private bundle publication requires public_origin and public_upload_dir settings")
        return self.public_origin.rstrip("/") + f"/v1/fleet/private-artifacts/{sha256}"

    @property
    def relay_authorized_keys_file(self) -> Path:
        return Path(self.state_dir).expanduser().resolve() / "ssh" / "relay" / "authorized_keys"

    @classmethod
    def load(cls, path: str | Path) -> "FleetSettings":
        source = Path(path)
        metadata = source.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600
                or (os.name == "posix" and metadata.st_uid != os.geteuid())):
            raise ValueError("Fleet settings must be a regular private mode-0600 file")
        def unique_pairs(pairs):
            values = {}
            for key, value in pairs:
                if key in values:
                    raise ValueError("duplicate Fleet settings key")
                values[key] = value
            return values

        raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
        if isinstance(raw, dict):
            reference = raw.get("admin_token")
            public_value = raw.get("public_upload_dir")
            if isinstance(public_value, str):
                public_root = Path(public_value).expanduser().resolve()
                resolved_settings = source.resolve()
                if resolved_settings == public_root or resolved_settings.is_relative_to(public_root):
                    raise ValueError("Fleet settings file must be outside the public upload tree")
                identity_value = raw.get("ssh_admin_identity_file")
                if isinstance(identity_value, str):
                    identity_path = Path(identity_value).expanduser()
                    if not identity_path.is_absolute():
                        identity_path = source.resolve().parent / identity_path
                    resolved_identity = identity_path.resolve()
                    if resolved_identity == public_root or resolved_identity.is_relative_to(public_root):
                        raise ValueError("Fleet SSH private identity must be outside the public upload tree")
            if isinstance(reference, str) and reference.startswith("file:"):
                secret_path = Path(reference[5:]).expanduser()
                if not secret_path.is_absolute():
                    secret_path = source.resolve().parent / secret_path
                secret_info = secret_path.lstat()
                if (not stat.S_ISREG(secret_info.st_mode) or stat.S_IMODE(secret_info.st_mode) != 0o600
                        or (os.name == "posix" and secret_info.st_uid != os.geteuid())):
                    raise ValueError("admin token file must be owner-matched regular mode-0600")
            if isinstance(public_value, str) and isinstance(reference, str) and reference.startswith("file:"):
                secret_path = Path(reference[5:]).expanduser()
                if not secret_path.is_absolute():
                    secret_path = source.resolve().parent / secret_path
                public_root = Path(public_value).expanduser().resolve()
                resolved_secret = secret_path.resolve()
                if resolved_secret == public_root or resolved_secret.is_relative_to(public_root):
                    raise ValueError("admin token file must be outside the public upload tree")
        data = load_settings_json(source, secret_fields=("admin_token",))
        return cls.model_validate(data, strict=True)


def resolve_settings(value: FleetSettings | Mapping[str, Any] | str | Path) -> FleetSettings:
    if isinstance(value, FleetSettings):
        return value
    if isinstance(value, (str, Path)):
        return FleetSettings.load(value)
    if isinstance(value, Mapping):
        return FleetSettings.model_validate(value, strict=True)
    raise TypeError("settings must be FleetSettings, a strict mapping, or a private JSON file path")
