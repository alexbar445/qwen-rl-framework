"""Typed boundaries between the learner coordinator and compute engines."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable


class PublishedFile(TypedDict):
    name: str
    sha256: str
    size: int


class PublishedPolicy(TypedDict):
    version: int
    directory: str
    base_fingerprint: str
    artifact_fingerprint: str
    quantization: dict[str, Any]
    files: list[PublishedFile]
    adapter_hash: str
    evidence: dict[str, Any]


class LearnerUpdateResult(TypedDict, total=False):
    updated: bool
    policy_version: int
    learner_start_version: int
    behavior_version: int
    policy_lag: int
    policy_tokens: int
    trajectory_count: int
    group_count: int
    gradient_norm_before_clip: float
    publication: PublishedPolicy


@runtime_checkable
class RankComputeBackend(Protocol):
    """Operations implemented by the private single- or multi-rank service."""

    @property
    def cached_status(self) -> Mapping[str, Any]: ...

    def start(self) -> Mapping[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...

    def logprobs(
        self,
        input_ids: list[int],
        policy_mask: list[int],
        attention_mask: list[int] | None = None,
    ) -> Sequence[float]: ...

    def probe(self, lengths: list[int]) -> Mapping[str, Any]: ...

    def update(
        self,
        rows: list[dict[str, Any]],
        *,
        before_optimizer_step: Callable[[], None] | None = None,
        recovery_context: dict[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...

    def export_adapter(self, path: str | Path) -> Mapping[str, Any]: ...

    def restore(self, payload: dict[str, Any]) -> Mapping[str, Any]: ...

    def shutdown(self) -> None: ...


@runtime_checkable
class LearnerEngineProtocol(Protocol):
    """Engine surface consumed by the HTTP coordinator."""

    tokenizer: Any
    evidence: dict[str, Any]
    policy_version: int

    @property
    def recovery_metadata(self) -> dict[str, Any]: ...

    @property
    def retained_policy_versions(self) -> tuple[int, ...]: ...

    def logprobs(self, input_ids: list[int], policy_mask: list[int]) -> list[float]: ...

    def update(
        self,
        trajectories: list[dict[str, Any]],
        *,
        before_optimizer_step: Callable[[], None] | None = None,
        recovery_context: dict[str, Any] | None = None,
    ) -> LearnerUpdateResult: ...

    def publish(self) -> PublishedPolicy: ...

    def acquire_policy(self, version: int) -> None: ...

    def release_policy(self, version: int) -> None: ...

    def save_recovery_snapshot(
        self,
        *,
        update_context: dict[str, Any] | None = None,
        reason: str = "explicit",
    ) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


class LearnerRunEngineProtocol(LearnerEngineProtocol, Protocol):
    """Common engine surface used by run lifecycle orchestration."""

    config: dict[str, Any]
    group_size: int


class CentralGradientRunEngineProtocol(LearnerRunEngineProtocol, Protocol):
    """Capabilities required only by the explicitly selected central backend."""

    worker_executor_ids: Mapping[str, str]

    def begin_staged_round(self, round_id: str, first_ordinal: int) -> None: ...

    def note_sampling_outcome(self, round_id: str, ordinal: int) -> None: ...

    def stage_group(
        self,
        round_id: str,
        trajectories: list[dict[str, Any]],
        *,
        logical_run_id: str | None = None,
    ) -> None: ...

    def cancel_staged_round(self, round_id: str | None = None) -> int: ...


@runtime_checkable
class CheckpointOverlapCapability(Protocol):
    """Optional engine capability used during phase-shared checkpointing."""

    @property
    def supports_checkpoint_overlap(self) -> bool: ...


@runtime_checkable
class TrainerFailureStateProtocol(Protocol):
    failure_stage: str


@runtime_checkable
class TrainerFailureStageCapability(Protocol):
    """Optional diagnostic access to an engine's active Trainer stage."""

    trainer: TrainerFailureStateProtocol
