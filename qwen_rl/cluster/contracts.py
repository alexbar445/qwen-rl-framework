"""Strict shared identities and capability receipts for the LAN cluster."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GPUCapability(StrictContract):
    uuid: str = Field(min_length=1, max_length=100)
    index: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=200)
    memory_total_mib: int = Field(gt=0)
    compute_capability: str | None = None


class NodeRegistration(StrictContract):
    code: str = Field(min_length=16, max_length=200)
    site_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    node_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    boot_id: str = Field(min_length=1, max_length=100)
    role: Literal["gateway", "worker"]
    lan_host: str = Field(min_length=1, max_length=253)
    worker_port: int = Field(default=8241, ge=1, le=65535)
    ssh_port: int = Field(default=22, ge=1, le=65535)
    ssh_public_key: str = Field(min_length=1, max_length=2000)
    ssh_host_public_key: str = Field(min_length=1, max_length=2000)
    gpus: list[GPUCapability] = Field(default_factory=list)

    @field_validator("lan_host", "ssh_public_key", "ssh_host_public_key")
    @classmethod
    def single_line(cls, value: str) -> str:
        if any(c in value for c in "\r\n\x00"):
            raise ValueError("node addresses and keys must be single-line")
        return value


class NodeHeartbeat(StrictContract):
    boot_id: str = Field(min_length=1, max_length=100)
    state: Literal["registered", "installing", "needs_reboot", "loading", "ready", "draining", "quarantined"]
    worker_health: dict[str, Any] | None = None
    image_digest: str | None = None
    error: str | None = None


class EnrolmentRequest(StrictContract):
    site_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    role: Literal["gateway", "worker"]
    max_nodes: int = Field(default=50, ge=1, le=10000)
    ttl_seconds: int = Field(default=3600, ge=30, le=86400)
