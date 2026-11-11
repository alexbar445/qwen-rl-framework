"""Typed Worker residency control and configuration."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator


@dataclass(frozen=True)
class ResidencySettings:
    enabled: bool = False
    weight_source: str = "host"
    engine_lifecycle: str = "suspend"
    drain_timeout_seconds: float = 900.0


def parse_residency_settings(config: Any) -> ResidencySettings:
    """Read the optional top-level Worker JSON residency section."""
    if not isinstance(config, dict):
        raise ValueError("worker config must be an object")
    value = config.get("residency")
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("residency must be an object")
    extra = sorted(set(value) - {"enabled", "drain_timeout_seconds", "weight_source", "engine_lifecycle"})
    if extra:
        raise ValueError(f"residency contains unsupported field(s): {extra}")
    enabled = value.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("residency.enabled must be a boolean")
    timeout = value.get("drain_timeout_seconds", 900.0)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("residency.drain_timeout_seconds must be finite and positive")
    source = value.get("weight_source", "host")
    if not isinstance(source, str) or source not in {"host", "artifact"}:
        raise ValueError("residency.weight_source must be host or artifact")
    lifecycle = value.get("engine_lifecycle", "suspend")
    if not isinstance(lifecycle, str) or lifecycle not in {"suspend", "restart"}:
        raise ValueError("residency.engine_lifecycle must be suspend or restart")
    if lifecycle == "restart" and not enabled:
        raise ValueError("restart engine lifecycle requires residency.enabled=true")
    if lifecycle == "restart" and config.get("inference", {}).get("backend") != "ninfer":
        raise ValueError("restart engine lifecycle requires NInfer")
    return ResidencySettings(enabled=enabled, drain_timeout_seconds=float(timeout),
                             weight_source=source, engine_lifecycle=lifecycle)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ResidencyActionRequest(_StrictModel):
    action: Literal["drain", "suspend", "wake", "admit"]
    policy_version: StrictInt | None = Field(default=None, ge=0)
    adapter_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_action_fields(self) -> "ResidencyActionRequest":
        if self.action == "admit":
            if self.policy_version is None or self.adapter_sha256 is None:
                raise ValueError("admit requires policy_version and adapter_sha256")
        elif self.policy_version is not None or self.adapter_sha256 is not None:
            raise ValueError("policy identity fields are only valid for admit")
        return self


class ResidencyTransitionRecord(_StrictModel):
    action: Literal["drain", "suspend", "wake", "admit"]
    outcome: Literal["succeeded", "timed_out", "failed"]
    started_at: StrictStr
    duration_seconds: float = Field(ge=0)
    error: StrictStr | None = None


class ResidencyStatusResponse(_StrictModel):
    worker_id: StrictStr
    enabled: bool
    state: Literal["resident", "draining", "suspended", "resident_fenced"]
    ready: bool
    policy_ready: bool
    admission_open: bool
    drain_complete: bool
    active_generations: StrictInt = Field(ge=0)
    pending_generations: StrictInt = Field(ge=0)
    queued_generations: StrictInt = Field(ge=0)
    episode_refs: StrictInt = Field(ge=0)
    generation_refs: StrictInt = Field(ge=0)
    policy_update_active: bool
    policy_version: StrictInt
    adapter_sha256: StrictStr | None
    native_model_id: StrictStr | None
    native: dict[str, Any] | None
    native_error: StrictStr | None
    last_transition: ResidencyTransitionRecord | None
