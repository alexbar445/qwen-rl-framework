"""Typed public contracts for BEPA persistence and replay artifacts.

This module intentionally contains only CPU-friendly dataclasses and JSON
validation. It has no dependency on the Learner, Worker, or verifier runtime.
Private targets never appear in a public job claim; only their digest is kept
inside the compatibility identity used by the coordinator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Literal, Mapping, Sequence


ROW_FRESH_STUDENT = "fresh_student"
ROW_NATIVE_REPLAY = "native_replay"
ROW_GUIDED_REPLAY = "guided_replay"
ROW_KINDS = frozenset({ROW_FRESH_STUDENT, ROW_NATIVE_REPLAY, ROW_GUIDED_REPLAY})
REPLAY_ROW_KINDS = frozenset({ROW_NATIVE_REPLAY, ROW_GUIDED_REPLAY})
MAX_CACHE_AGE_UPDATES = 10
MAX_CACHE_COMMITTED_USES = 5
JOB_STATES = frozenset(
    {
        "queued",
        "retry_wait",
        "teacher_running",
        "source_submitted",
        "source_verified",
        "plan_ready",
        "seed_running",
        "cache_ready",
        "failed",
        "exhausted",
        "rejected",
        "cancelled",
        "outcome_unknown",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BepaError(RuntimeError):
    """Base exception for a rejected BEPA operation."""


class ContractError(BepaError, ValueError):
    """A value does not satisfy the persisted contract."""


class BudgetUnavailableError(BepaError):
    """A provider token budget cannot authorize a new rescue job."""

    def __init__(self, scope_id: str, reason: str) -> None:
        if reason not in {"usage_unknown", "exhausted"}:
            raise ValueError("budget unavailability reason must be typed")
        self.scope_id = scope_id
        self.reason = reason
        detail = "usage is unknown" if reason == "usage_unknown" else "token allowance is exhausted"
        super().__init__(f"provider budget is unavailable for scope {scope_id}: {detail}")


def validate_teacher_plan(plan: Any) -> str:
    """Validate reusable procedural guidance using the central admission contract."""
    if not isinstance(plan, str) or not plan.strip() or len(plan) > 4000:
        raise ContractError("teacher plan must contain 1..4000 characters")
    if any(marker in plan for marker in ("[[", "test_output", "private_target")):
        raise ContractError("plan includes an answer literal or private target reference")
    return plan.strip()


class ConflictError(BepaError):
    """An idempotency key or entity identity was reused inconsistently."""


class LeaseLostError(BepaError):
    """A worker no longer owns the current job lease and fencing token."""


class StateTransitionError(BepaError):
    """A durable state transition is not valid from the current state."""


def _json_value(value: Any) -> Any:
    if hasattr(value, "to_json_dict"):
        return value.to_json_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Encode stable JSON and reject NaN/Infinity before it reaches disk."""
    try:
        return json.dumps(
            _json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractError(f"{field_name} must be a lowercase sha256 hex digest")
    return value


def _require_nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value


def _require_json_object(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field_name} must be a JSON object")
    result = dict(value)
    canonical_json_bytes(result)
    return result


@dataclass(frozen=True)
class TaskSpec:
    """Internal compatibility identity and the public task sent to a runner.

    ``private_target_hash`` is included in ``task_key`` but omitted from
    ``public_claim_payload``. Callers must store only the digest here, never
    the private target values themselves.
    """

    public_challenge: Mapping[str, Any]
    clean_prompt: str
    tool_contract: Mapping[str, Any]
    env_contract: Mapping[str, Any]
    prompt_contract: Mapping[str, Any]
    tokenizer_fingerprint: str
    artifact_fingerprint: str
    private_target_hash: str
    verifier_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_challenge", _require_json_object(self.public_challenge, "public_challenge"))
        object.__setattr__(self, "tool_contract", _require_json_object(self.tool_contract, "tool_contract"))
        object.__setattr__(self, "env_contract", _require_json_object(self.env_contract, "env_contract"))
        object.__setattr__(self, "prompt_contract", _require_json_object(self.prompt_contract, "prompt_contract"))
        object.__setattr__(self, "verifier_contract", _require_json_object(self.verifier_contract, "verifier_contract"))
        _require_nonempty(self.clean_prompt, "clean_prompt")
        _require_nonempty(self.tokenizer_fingerprint, "tokenizer_fingerprint")
        _require_nonempty(self.artifact_fingerprint, "artifact_fingerprint")
        require_sha256(self.private_target_hash, "private_target_hash")

    @classmethod
    def create(
        cls,
        *,
        public_challenge: Mapping[str, Any],
        clean_prompt: str,
        tool_contract: Mapping[str, Any],
        env_contract: Mapping[str, Any],
        prompt_contract: Mapping[str, Any],
        tokenizer_fingerprint: str,
        artifact_fingerprint: str,
        private_target_hash: str,
        verifier_contract: Mapping[str, Any],
    ) -> "TaskSpec":
        return cls(
            public_challenge=public_challenge,
            clean_prompt=clean_prompt,
            tool_contract=tool_contract,
            env_contract=env_contract,
            prompt_contract=prompt_contract,
            tokenizer_fingerprint=tokenizer_fingerprint,
            artifact_fingerprint=artifact_fingerprint,
            private_target_hash=private_target_hash,
            verifier_contract=verifier_contract,
        )

    @property
    def task_key(self) -> str:
        """Hash all public contracts plus private-target and verifier digests."""
        return content_hash(self.compatibility_payload())

    def compatibility_payload(self) -> dict[str, Any]:
        return {
            "public_challenge": dict(self.public_challenge),
            "clean_prompt": self.clean_prompt,
            "tool_contract": dict(self.tool_contract),
            "env_contract": dict(self.env_contract),
            "prompt_contract": dict(self.prompt_contract),
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "private_target_hash": self.private_target_hash,
            "verifier_contract": dict(self.verifier_contract),
        }

    def public_claim_payload(self) -> dict[str, Any]:
        """Data the external runner may see; never includes private digest/key."""
        return {
            "public_challenge": dict(self.public_challenge),
            "clean_prompt": self.clean_prompt,
            "tool_contract": dict(self.tool_contract),
            "env_contract": dict(self.env_contract),
            "prompt_contract": dict(self.prompt_contract),
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
        }

    def public_job_key(self, *, source_contract: Mapping[str, Any], teacher_id: str) -> str:
        """Build a dedupe key from public task/source/teacher identity only."""
        _require_nonempty(teacher_id, "teacher_id")
        return content_hash(
            {
                "public_task": self.public_claim_payload(),
                "source_contract": _require_json_object(source_contract, "source_contract"),
                "teacher_id": teacher_id,
            }
        )


@dataclass(frozen=True)
class TaskRecord:
    task_key: str
    spec: TaskSpec
    allowed: bool
    created_at: float


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    task_key: str
    public_job_key: str
    budget_scope: str
    state: str
    attempts: int
    max_attempts: int
    used_tokens: int
    reserved_tokens: int
    max_tokens: int
    usage_unknown: bool
    fence: int
    lease_owner: str | None
    lease_expires_at: float | None
    source_artifact_refs: Mapping[str, str] | None
    plan_artifact_ref: str | None
    fail_reason: str | None
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class JobLease:
    job_id: str
    task_key: str
    budget_scope: str
    worker_id: str
    fence: int
    lease_expires_at: float
    attempt_number: int
    public_task: Mapping[str, Any]


@dataclass(frozen=True)
class PlanRecord:
    plan_id: str
    job_id: str
    task_key: str
    plan_artifact_ref: str
    source_episode_id: str
    source_receipt_ref: str
    status: str
    seed_attempts: int
    created_at: float


@dataclass(frozen=True)
class UpdateAttempt:
    attempt_id: str
    batch_id: str
    input_batch_hash: str
    engine_epoch: str
    start_policy_version: int
    state: str
    new_policy_version: int | None
    adapter_sha256: str | None
    commit_evidence: Mapping[str, Any] | None
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class SourcePolicy:
    engine_epoch: str
    version: int
    adapter_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.engine_epoch, "engine_epoch")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 0:
            raise ContractError("source policy version must be a non-negative integer")
        require_sha256(self.adapter_sha256, "source policy adapter_sha256")


@dataclass(frozen=True)
class SourceEpisode:
    """Raw student episode metadata before a guidance-free projection."""

    episode_id: str
    task_key: str
    source_policy: SourcePolicy
    row_kind: Literal["native", "guided"]
    input_ids: Sequence[int]
    policy_mask: Sequence[int]
    guided_prefix_length: int
    clean_prefix_ids: Sequence[int]
    raw_trace: Any
    tokenizer_fingerprint: str
    behavior_logprobs: Sequence[float] = field(default_factory=tuple)
    max_context_tokens: int | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.episode_id, "episode_id")
        require_sha256(self.task_key, "task_key")
        if self.row_kind not in {"native", "guided"}:
            raise ContractError("row_kind must be 'native' or 'guided'")
        _require_nonempty(self.tokenizer_fingerprint, "tokenizer_fingerprint")
        if isinstance(self.guided_prefix_length, bool) or not isinstance(self.guided_prefix_length, int):
            raise ContractError("guided_prefix_length must be an integer")
        ids = _validate_token_ids(self.input_ids, "input_ids")
        mask = _validate_mask(self.policy_mask, "policy_mask")
        clean_prefix = _validate_token_ids(self.clean_prefix_ids, "clean_prefix_ids")
        if len(ids) != len(mask):
            raise ContractError("input_ids and policy_mask must have identical lengths")
        if not 0 <= self.guided_prefix_length <= len(ids):
            raise ContractError("guided_prefix_length is outside input_ids")
        if any(mask[: self.guided_prefix_length]):
            raise ContractError("guided prompt prefix must not contain policy-loss tokens")
        if not clean_prefix:
            raise ContractError("clean_prefix_ids must not be empty")
        if not any(mask[self.guided_prefix_length :]):
            raise ContractError("episode has no valid policy-mask token after its prompt prefix")
        if self.max_context_tokens is not None:
            if (isinstance(self.max_context_tokens, bool) or not isinstance(self.max_context_tokens, int)
                    or self.max_context_tokens <= 0):
                raise ContractError("max_context_tokens must be a positive integer")
        logprobs = tuple(float(value) for value in self.behavior_logprobs)
        if any(not math.isfinite(value) for value in logprobs):
            raise ContractError("source episode behavior_logprobs must contain only finite values")
        if len(logprobs) != len(ids) - self.guided_prefix_length:
            raise ContractError("source episode behavior_logprobs must align exactly with the body after the guided prefix")
        canonical_json_bytes(self.raw_trace)
        object.__setattr__(self, "input_ids", tuple(ids))
        object.__setattr__(self, "policy_mask", tuple(mask))
        object.__setattr__(self, "clean_prefix_ids", tuple(clean_prefix))
        object.__setattr__(self, "behavior_logprobs", logprobs)


@dataclass(frozen=True)
class SourceProjection:
    """Immutable clean-prefix replacement while preserving the exact body."""

    episode_id: str
    task_key: str
    source_policy: SourcePolicy
    row_kind: Literal["native_replay", "guided_replay"]
    input_ids: tuple[int, ...]
    policy_mask: tuple[int, ...]
    prompt_token_count: int
    body_token_count: int
    body_token_hash: str
    tokenizer_fingerprint: str

    def __post_init__(self) -> None:
        require_sha256(self.task_key, "task_key")
        if self.row_kind not in REPLAY_ROW_KINDS:
            raise ContractError("projection row_kind must be native_replay or guided_replay")
        _validate_token_ids(self.input_ids, "projection.input_ids")
        _validate_mask(self.policy_mask, "projection.policy_mask")
        if len(self.input_ids) != len(self.policy_mask):
            raise ContractError("projection input_ids and policy_mask are misaligned")
        if len(self.input_ids) != self.prompt_token_count + self.body_token_count:
            raise ContractError("projection lengths do not match prompt/body token counts")
        if not any(self.policy_mask):
            raise ContractError("projection has no valid policy-mask token")
        require_sha256(self.body_token_hash, "body_token_hash")


@dataclass(frozen=True)
class TrustedReceipt:
    """Receipt produced by the central verifier, never accepted from a runner."""

    receipt_id: str
    task_key: str
    episode_id: str
    private_target_hash: str
    verifier_contract_hash: str
    exact: bool
    reward: float
    submission_hash: str
    verifier_id: str = "central_verifier"

    def __post_init__(self) -> None:
        _require_nonempty(self.receipt_id, "receipt_id")
        require_sha256(self.task_key, "task_key")
        _require_nonempty(self.episode_id, "episode_id")
        require_sha256(self.private_target_hash, "private_target_hash")
        require_sha256(self.verifier_contract_hash, "verifier_contract_hash")
        require_sha256(self.submission_hash, "submission_hash")
        _require_nonempty(self.verifier_id, "verifier_id")
        if not isinstance(self.exact, bool):
            raise ContractError("exact must be a boolean issued by the central verifier")
        if not math.isfinite(float(self.reward)):
            raise ContractError("receipt reward must be finite")
        object.__setattr__(self, "reward", float(self.reward))


@dataclass(frozen=True)
class CacheEntry:
    cache_entry_id: str
    task_key: str
    engine_epoch: str
    source_episode_id: str
    source_policy_version: int
    source_policy_sha256: str
    row_kind: Literal["native_replay", "guided_replay"]
    raw_artifact_ref: str
    projection_artifact_ref: str
    receipt_artifact_ref: str
    committed_uses: int
    reserved_uses: int
    ready_sequence: int
    active: bool


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    batch_id: str
    group_id: str
    cache_entry_id: str
    task_key: str
    engine_epoch: str
    source_policy: SourcePolicy
    row_kind: Literal["native_replay", "guided_replay"]
    projection_artifact_ref: str
    receipt_artifact_ref: str
    state: Literal["reserved", "committed", "released", "commit_unknown"]


@dataclass(frozen=True)
class ReplayRow:
    """A replay row has no behavior-logprob or current-policy-version field."""

    input_ids: Sequence[int]
    policy_mask: Sequence[int]
    reward: float
    group_id: str
    row_kind: Literal["native_replay", "guided_replay"]
    source_policy: SourcePolicy
    cache_entry_id: str
    verification_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.row_kind not in REPLAY_ROW_KINDS:
            raise ContractError("replay row_kind must be native_replay or guided_replay")
        if not isinstance(self.source_policy, SourcePolicy):
            raise ContractError("replay source_policy must be a SourcePolicy")
        ids = _validate_token_ids(self.input_ids, "input_ids")
        mask = _validate_mask(self.policy_mask, "policy_mask")
        if len(ids) != len(mask):
            raise ContractError("replay input_ids and policy_mask must have identical lengths")
        if not any(mask):
            raise ContractError("replay row has no valid policy-mask token")
        if not math.isfinite(float(self.reward)):
            raise ContractError("replay reward must be finite")
        _require_nonempty(self.group_id, "group_id")
        _require_nonempty(self.cache_entry_id, "cache_entry_id")
        receipt = _require_json_object(self.verification_receipt, "verification_receipt")
        if (receipt.get("task_key") is None or receipt.get("exact") is not True
                or not isinstance(receipt.get("verifier_contract"), str) or not receipt.get("verifier_contract", "").strip()
                or not isinstance(receipt.get("projection_sha256"), str)):
            raise ContractError("verification_receipt must bind an exact task, verifier contract, and projection hash")
        require_sha256(receipt["task_key"], "verification_receipt.task_key")
        require_sha256(receipt["projection_sha256"], "verification_receipt.projection_sha256")
        object.__setattr__(self, "input_ids", tuple(ids))
        object.__setattr__(self, "policy_mask", tuple(mask))
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "verification_receipt", receipt)

    def to_training_dict(self) -> dict[str, Any]:
        """Serialize the row keys consumed by ``LearnerEngine.update``."""
        return {
            "input_ids": list(self.input_ids),
            "policy_mask": list(self.policy_mask),
            "reward": self.reward,
            "group_id": self.group_id,
            "row_kind": self.row_kind,
            "source_policy": {
                "engine_epoch": self.source_policy.engine_epoch,
                "version": self.source_policy.version,
                "adapter_sha256": self.source_policy.adapter_sha256,
            },
            "cache_entry_id": self.cache_entry_id,
            "verification_receipt": dict(self.verification_receipt),
        }


def _validate_token_ids(values: Sequence[int], field_name: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ContractError(f"{field_name} must be a sequence of token IDs")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"{field_name} must contain non-negative integer token IDs")
        result.append(value)
    if not result:
        raise ContractError(f"{field_name} must not be empty")
    return result


def _validate_mask(values: Sequence[int], field_name: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ContractError(f"{field_name} must be a sequence of 0/1 values")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, float) or value not in (0, 1):
            raise ContractError(f"{field_name} must contain only 0/1 values")
        result.append(int(value))
    return result


def replay_row_from_artifacts(
    *,
    projection: SourceProjection,
    receipt: TrustedReceipt,
    projection_artifact_ref: str,
    group_id: str,
    cache_entry_id: str,
) -> ReplayRow:
    """Build a replay row from matching central artifacts and receipt."""
    if receipt.verifier_id != "central_verifier" or not receipt.exact:
        raise ContractError("only exact central-verifier receipts can build replay rows")
    if (receipt.task_key, receipt.episode_id) != (projection.task_key, projection.episode_id):
        raise ContractError("receipt does not attest this source projection")
    require_sha256(projection_artifact_ref, "projection_artifact_ref")
    return ReplayRow(
        input_ids=projection.input_ids,
        policy_mask=projection.policy_mask,
        reward=receipt.reward,
        group_id=group_id,
        row_kind=projection.row_kind,
        source_policy=projection.source_policy,
        cache_entry_id=cache_entry_id,
        verification_receipt={
            "task_key": receipt.task_key,
            "verifier_contract": receipt.verifier_contract_hash,
            "exact": receipt.exact,
            "projection_sha256": projection_artifact_ref,
        },
    )
