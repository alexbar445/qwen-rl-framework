"""Metadata contracts for asynchronous, centrally aggregated gradients.

Gradient tensors are deliberately not part of these JSON contracts. The HTTP
transport carries them in a separate safetensors payload; in-process callers
attach them to :class:`GradientContribution.gradients`.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence


GRADIENT_CONTRIBUTION_CONTRACT = "global_token_mean_contribution/v1"
GRADIENT_JOB_SCHEMA = "gradient_job/v1"
PREPARED_WINDOW_SCHEMA = "prepared_gradient_window/v1"
GRADIENT_CONTRIBUTION_SCHEMA = "gradient_contribution/v1"
GRADIENT_LOSS_PROFILE = "dapo_global_token_mean_beta0_v1"


class GradientContractError(ValueError):
    """A gradient packet does not satisfy the shared wire contract."""


def guided_replay_matches_fresh_policy_version(
    fresh_rows: Sequence[Mapping[str, Any]],
    replay_row: Mapping[str, Any],
) -> bool:
    """Whether a guided replay was produced by the exact fresh-group policy.

    Guided rows do not carry fresh behavior logprobs. When they accompany any
    native success, their source policy therefore must be the same policy that
    generated every fresh candidate in the group.
    """
    source_policy = replay_row.get("source_policy")
    if not isinstance(source_policy, Mapping):
        return False
    source_version = source_policy.get("version")
    if type(source_version) is not int or not fresh_rows:
        return False
    return all(
        type(row.get("policy_version")) is int
        and row["policy_version"] == source_version
        for row in fresh_rows
    )


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GradientContractError(f"{label} must be an integer >= {minimum}")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GradientContractError(f"{label} must be a non-empty trimmed string")
    return value


def _uuid_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise GradientContractError(f"{label} must be a 32-character lowercase UUID hex string")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def validate_compute_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"base_artifact_id", "loss_profile", "correction_cap"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise GradientContractError("compute_profile must include base artifact, loss profile and correction cap")
    artifact = value["base_artifact_id"]
    if not isinstance(artifact, str) or re.fullmatch(r"[0-9a-f]{32}", artifact) is None:
        raise GradientContractError("compute_profile base_artifact_id must be 32 lowercase hex characters")
    if value["loss_profile"] != GRADIENT_LOSS_PROFILE:
        raise GradientContractError("unsupported compute loss profile")
    cap = value["correction_cap"]
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(float(cap)) or float(cap) <= 0:
        raise GradientContractError("compute_profile correction_cap must be finite and positive")
    return {
        "base_artifact_id": artifact,
        "loss_profile": GRADIENT_LOSS_PROFILE,
        "correction_cap": float(cap),
    }


def compute_profile_fingerprint(value: Mapping[str, Any]) -> str:
    import hashlib
    profile = validate_compute_profile(value)
    return hashlib.sha256(_canonical_json(profile).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GradientParameter:
    """One named trainable adapter parameter in the immutable layout."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        _nonempty(self.name, "parameter name")
        if not self.name or "lora_" not in self.name:
            raise GradientContractError("gradient layout may contain only LoRA parameters")
        if not isinstance(self.shape, tuple) or any(
            type(size) is not int or size < 0 for size in self.shape
        ):
            raise GradientContractError("parameter shape must be a tuple of non-negative integers")
        _nonempty(self.dtype, "parameter dtype")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GradientParameter":
        if not isinstance(value, Mapping) or set(value) != {"name", "shape", "dtype"}:
            raise GradientContractError("malformed gradient parameter layout entry")
        shape = value["shape"]
        if not isinstance(shape, (list, tuple)):
            raise GradientContractError("parameter shape must be a JSON array")
        return cls(str(value["name"]), tuple(shape), str(value["dtype"]))


@dataclass(frozen=True)
class GradientJobSpec:
    """Unbound work identity used when the central owner prepares a window."""

    job_id: str
    token_count: int
    group_id: str
    owner_executor_id: str

    def __post_init__(self) -> None:
        _nonempty(self.job_id, "job_id")
        _exact_int(self.token_count, "token_count", minimum=1)
        _nonempty(self.group_id, "group_id")
        _nonempty(self.owner_executor_id, "owner_executor_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "token_count": self.token_count,
            "group_id": self.group_id,
            "owner_executor_id": self.owner_executor_id,
        }

    @classmethod
    def from_value(cls, value: "GradientJobSpec") -> "GradientJobSpec":
        if isinstance(value, cls):
            return value
        raise GradientContractError("jobs must use explicit GradientJobSpec owner and group fields")


@dataclass(frozen=True)
class GradientJob:
    """A work item bound to one immutable window, version, denominator and layout."""

    schema: str
    window_id: str
    attempt_id: str
    lineage_id: str
    engine_epoch: str
    job_id: str
    group_id: str
    owner_executor_id: str
    policy_version: int
    n_window_tokens: int
    token_count: int
    work_index: int
    layout_fingerprint: str
    compute_profile: Mapping[str, Any]
    compute_profile_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema != GRADIENT_JOB_SCHEMA:
            raise GradientContractError("unsupported gradient job schema")
        for name in ("window_id", "attempt_id", "job_id", "layout_fingerprint"):
            _nonempty(getattr(self, name), name)
        _nonempty(self.group_id, "group_id")
        _nonempty(self.owner_executor_id, "owner_executor_id")
        _uuid_hex(self.lineage_id, "lineage_id")
        _uuid_hex(self.engine_epoch, "engine_epoch")
        _exact_int(self.policy_version, "policy_version")
        _exact_int(self.n_window_tokens, "n_window_tokens", minimum=1)
        _exact_int(self.token_count, "token_count", minimum=1)
        _exact_int(self.work_index, "work_index")
        profile = validate_compute_profile(self.compute_profile)
        object.__setattr__(self, "compute_profile", MappingProxyType(profile))
        if self.compute_profile_fingerprint != compute_profile_fingerprint(profile):
            raise GradientContractError("gradient job compute profile fingerprint mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "window_id": self.window_id,
            "attempt_id": self.attempt_id,
            "lineage_id": self.lineage_id,
            "engine_epoch": self.engine_epoch,
            "job_id": self.job_id,
            "group_id": self.group_id,
            "owner_executor_id": self.owner_executor_id,
            "policy_version": self.policy_version,
            "n_window_tokens": self.n_window_tokens,
            "token_count": self.token_count,
            "work_index": self.work_index,
            "layout_fingerprint": self.layout_fingerprint,
            "compute_profile": dict(self.compute_profile),
            "compute_profile_fingerprint": self.compute_profile_fingerprint,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GradientJob":
        fields = {
            "schema", "window_id", "attempt_id", "lineage_id", "engine_epoch", "job_id",
            "group_id", "owner_executor_id", "policy_version",
            "n_window_tokens", "token_count", "work_index", "layout_fingerprint",
            "compute_profile", "compute_profile_fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise GradientContractError("malformed gradient job")
        return cls(**dict(value))

    @classmethod
    def from_json(cls, value: str) -> "GradientJob":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GradientContractError("gradient job is not valid JSON") from exc
        return cls.from_dict(payload)


@dataclass(frozen=True)
class PreparedWindow:
    """Central preparation receipt and exact ordered coverage for one update."""

    schema: str
    aggregation: str
    window_id: str
    attempt_id: str
    lineage_id: str
    engine_epoch: str
    policy_version: int
    n_window_tokens: int
    jobs: tuple[GradientJob, ...]
    work_order: tuple[str, ...]
    parameter_layout: tuple[GradientParameter, ...]
    layout_fingerprint: str
    compute_profile: Mapping[str, Any]
    update_context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema != PREPARED_WINDOW_SCHEMA:
            raise GradientContractError("unsupported prepared window schema")
        if self.aggregation != GRADIENT_CONTRIBUTION_CONTRACT:
            raise GradientContractError("unsupported gradient aggregation contract")
        _nonempty(self.window_id, "window_id")
        _nonempty(self.attempt_id, "attempt_id")
        _uuid_hex(self.lineage_id, "lineage_id")
        _uuid_hex(self.engine_epoch, "engine_epoch")
        _nonempty(self.layout_fingerprint, "layout_fingerprint")
        profile = validate_compute_profile(self.compute_profile)
        object.__setattr__(self, "compute_profile", MappingProxyType(profile))
        profile_fingerprint = compute_profile_fingerprint(profile)
        _exact_int(self.policy_version, "policy_version")
        _exact_int(self.n_window_tokens, "n_window_tokens", minimum=1)
        if not self.jobs or any(not isinstance(job, GradientJob) for job in self.jobs):
            raise GradientContractError("prepared window must include gradient jobs")
        if not self.parameter_layout or any(
            not isinstance(item, GradientParameter) for item in self.parameter_layout
        ):
            raise GradientContractError("prepared window must include a LoRA parameter layout")
        names = tuple(item.name for item in self.parameter_layout)
        if len(names) != len(set(names)) or names != tuple(sorted(names)):
            raise GradientContractError("parameter layout names must be unique and sorted")
        job_ids = tuple(job.job_id for job in self.jobs)
        if len(job_ids) != len(set(job_ids)):
            raise GradientContractError("window job ids must be unique")
        group_ids = tuple(job.group_id for job in self.jobs)
        if (len(group_ids) != len(set(group_ids))
                and "trajectory_partitions" not in (self.update_context or {})):
            raise GradientContractError("each complete group must have exactly one gradient job")
        if self.work_order != job_ids:
            raise GradientContractError("work_order must exactly match the ordered jobs")
        if sum(job.token_count for job in self.jobs) != self.n_window_tokens:
            raise GradientContractError("job token coverage must sum to n_window_tokens")
        for index, job in enumerate(self.jobs):
            if (
                job.window_id != self.window_id
                or job.attempt_id != self.attempt_id
                or job.lineage_id != self.lineage_id
                or job.engine_epoch != self.engine_epoch
                or not job.owner_executor_id
                or job.policy_version != self.policy_version
                or job.n_window_tokens != self.n_window_tokens
                or job.work_index != index
                or job.layout_fingerprint != self.layout_fingerprint
                or job.compute_profile_fingerprint != profile_fingerprint
                or dict(job.compute_profile) != profile
            ):
                raise GradientContractError("gradient job is not bound to its prepared window")
        context = {} if self.update_context is None else dict(self.update_context)
        from .gradient_partitions import validate_partitions
        validate_partitions(context, self.jobs)
        try:
            context = json.loads(_canonical_json(context))
        except (TypeError, ValueError) as exc:
            raise GradientContractError("update_context must contain JSON-safe metadata") from exc
        object.__setattr__(self, "update_context", MappingProxyType(context))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "aggregation": self.aggregation,
            "window_id": self.window_id,
            "attempt_id": self.attempt_id,
            "lineage_id": self.lineage_id,
            "engine_epoch": self.engine_epoch,
            "policy_version": self.policy_version,
            "n_window_tokens": self.n_window_tokens,
            "jobs": [job.to_dict() for job in self.jobs],
            "work_order": list(self.work_order),
            "parameter_layout": [item.to_dict() for item in self.parameter_layout],
            "layout_fingerprint": self.layout_fingerprint,
            "compute_profile": dict(self.compute_profile),
            "update_context": dict(self.update_context or {}),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreparedWindow":
        fields = {
            "schema", "aggregation", "window_id", "attempt_id", "lineage_id", "engine_epoch", "policy_version",
            "n_window_tokens", "jobs", "work_order", "parameter_layout", "layout_fingerprint",
            "compute_profile", "update_context",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise GradientContractError("malformed prepared window")
        try:
            return cls(
                schema=value["schema"],
                aggregation=value["aggregation"],
                window_id=value["window_id"],
                attempt_id=value["attempt_id"],
                lineage_id=value["lineage_id"],
                engine_epoch=value["engine_epoch"],
                policy_version=value["policy_version"],
                n_window_tokens=value["n_window_tokens"],
                jobs=tuple(GradientJob.from_dict(job) for job in value["jobs"]),
                work_order=tuple(value["work_order"]),
                parameter_layout=tuple(GradientParameter.from_dict(item) for item in value["parameter_layout"]),
                layout_fingerprint=value["layout_fingerprint"],
                compute_profile=value["compute_profile"],
                update_context=value["update_context"],
            )
        except (TypeError, KeyError) as exc:
            raise GradientContractError("malformed prepared window values") from exc

    @classmethod
    def from_json(cls, value: str) -> "PreparedWindow":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GradientContractError("prepared window is not valid JSON") from exc
        return cls.from_dict(payload)


@dataclass(frozen=True)
class GradientContribution:
    """One job's already globally normalized LoRA gradient contribution.

    ``gradients`` contains every parameter name in the prepared layout. A
    ``None`` value records that the parameter had no gradient for this job.
    Tensor values are an in-memory payload and never enter JSON serialization.
    """

    schema: str
    aggregation: str
    window_id: str
    attempt_id: str
    lineage_id: str
    engine_epoch: str
    job_id: str
    group_id: str
    owner_executor_id: str
    policy_version: int
    n_window_tokens: int
    token_count: int
    layout_fingerprint: str
    compute_profile_fingerprint: str
    gradients: Mapping[str, Any]
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema != GRADIENT_CONTRIBUTION_SCHEMA:
            raise GradientContractError("unsupported gradient contribution schema")
        if self.aggregation != GRADIENT_CONTRIBUTION_CONTRACT:
            raise GradientContractError("unsupported gradient aggregation contract")
        for name in ("window_id", "attempt_id", "job_id", "group_id", "owner_executor_id", "layout_fingerprint"):
            _nonempty(getattr(self, name), name)
        _uuid_hex(self.lineage_id, "lineage_id")
        _uuid_hex(self.engine_epoch, "engine_epoch")
        _exact_int(self.policy_version, "policy_version")
        _exact_int(self.n_window_tokens, "n_window_tokens", minimum=1)
        _exact_int(self.token_count, "token_count", minimum=1)
        if not isinstance(self.compute_profile_fingerprint, str) or re.fullmatch(
            r"[0-9a-f]{64}", self.compute_profile_fingerprint
        ) is None:
            raise GradientContractError("compute_profile_fingerprint must be SHA-256 hex")
        if not isinstance(self.gradients, Mapping) or not self.gradients:
            raise GradientContractError("gradient contribution must include parameter presence")
        diagnostics = {} if self.diagnostics is None else dict(self.diagnostics)
        try:
            diagnostics = json.loads(_canonical_json(diagnostics), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (TypeError, ValueError) as exc:
            raise GradientContractError("gradient diagnostics must be finite JSON metadata") from exc
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "aggregation": self.aggregation,
            "window_id": self.window_id,
            "attempt_id": self.attempt_id,
            "lineage_id": self.lineage_id,
            "engine_epoch": self.engine_epoch,
            "job_id": self.job_id,
            "group_id": self.group_id,
            "owner_executor_id": self.owner_executor_id,
            "policy_version": self.policy_version,
            "n_window_tokens": self.n_window_tokens,
            "token_count": self.token_count,
            "layout_fingerprint": self.layout_fingerprint,
            "present_parameters": sorted(name for name, value in self.gradients.items() if value is not None),
            "compute_profile_fingerprint": self.compute_profile_fingerprint,
            "diagnostics": dict(self.diagnostics or {}),
        }

    def to_metadata_json(self) -> str:
        return _canonical_json(self.to_metadata_dict())

    @classmethod
    def from_metadata_dict(
        cls,
        value: Mapping[str, Any],
        *,
        gradients: Mapping[str, Any],
        parameter_names: tuple[str, ...] | list[str] | None = None,
    ) -> "GradientContribution":
        fields = {
            "schema", "aggregation", "window_id", "attempt_id", "lineage_id", "engine_epoch",
            "job_id", "group_id", "owner_executor_id", "policy_version",
            "n_window_tokens", "token_count", "layout_fingerprint", "present_parameters",
            "compute_profile_fingerprint", "diagnostics",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise GradientContractError("malformed gradient contribution metadata")
        present = value["present_parameters"]
        if not isinstance(present, list) or any(not isinstance(name, str) for name in present):
            raise GradientContractError("present_parameters must be a JSON string array")
        if present != sorted(set(present)):
            raise GradientContractError("present_parameters must be unique and sorted")
        if not isinstance(gradients, Mapping) or set(gradients) != set(present):
            raise GradientContractError("tensor payload keys do not match present_parameters")
        if parameter_names is None:
            parameter_names = list(present)
        if len(parameter_names) != len(set(parameter_names)) or not set(present).issubset(parameter_names):
            raise GradientContractError("parameter_names must cover present_parameters")
        all_gradients = {name: None for name in parameter_names}
        all_gradients.update(gradients)
        return cls(
            schema=value["schema"],
            aggregation=value["aggregation"],
            window_id=value["window_id"],
            attempt_id=value["attempt_id"],
            lineage_id=value["lineage_id"],
            engine_epoch=value["engine_epoch"],
            job_id=value["job_id"],
            group_id=value["group_id"],
            owner_executor_id=value["owner_executor_id"],
            policy_version=value["policy_version"],
            n_window_tokens=value["n_window_tokens"],
            token_count=value["token_count"],
            layout_fingerprint=value["layout_fingerprint"],
            compute_profile_fingerprint=value["compute_profile_fingerprint"],
            gradients=all_gradients,
            diagnostics=value["diagnostics"],
        )

    @classmethod
    def from_metadata_json(
        cls,
        value: str,
        *,
        gradients: Mapping[str, Any],
        parameter_names: tuple[str, ...] | list[str] | None = None,
    ) -> "GradientContribution":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GradientContractError("gradient contribution metadata is not valid JSON") from exc
        return cls.from_metadata_dict(payload, gradients=gradients, parameter_names=parameter_names)
