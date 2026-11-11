"""Authenticated, single-process ARC rollout service.

The service owns task targets, runs the frozen toolkit, accounts for original model token
IDs, and releases rewards only after a complete and version-consistent rollout group.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .budget import Contract, token_cost
from .contracts.arc_tools import NATIVE_TOOL_NAMES, is_model_stop, tool_schema_issue, turn_rejection_reason
from .contracts.episode_budget import budget_policy_identity
from .contracts.tool_call_failures import (ENV_SCHEMA_VERSION, MAX_TOOL_CALL_FAILURES,
                                          TOOL_CALL_FAILURE_COUNTING,
                                          tool_call_failure_feedback)
from .environment import ARCToolEnv, EnvironmentTokenFraming
from .prompt import prompt_contract
from .rewards import REWARD_SCHEMA, VERIFICATION_COEFFICIENT, group_rewards, validate_efficiency_coefficient
from .task_source import TaskSpec, load_tasks

_LOGGER = logging.getLogger(__name__)

GENERATED_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "arc-taskgen-v2"


def _generated_verifier():
    """The generator's own trusted verifier (`arc_taskgen.verifier`), imported lazily."""
    vendored = str(GENERATED_VENDOR)
    if vendored not in sys.path:
        sys.path.insert(0, vendored)
    from arc_taskgen import verifier

    return verifier


def generated_group_rewards(group: Any, members: list[Any], epsilon: float, *,
                            verification_coverages: list[float] | None = None) -> list[float]:
    """Score generated-task rows with the shared continuous reward, without length taper.

    The generator verifies private targets; runtime owns the training objective.
    Infrastructure-aborted groups receive no usable reward. All rows must carry
    one real task/policy identity and nonnegative original token counts.
    """
    validate_efficiency_coefficient(epsilon)
    if not members or not isinstance(group.task_id, str) or not group.task_id.strip():
        raise ValueError("reward group requires members and a task identity")
    versions = [str(member.policy_version if member.policy_version is not None else 0)
                for member in members]
    if len(set(versions)) != 1:
        raise ValueError("reward group must use the same policy version")
    costs, exacts = [], []
    for member in members:
        generated, observed = member.generated_tokens, member.env.state.obs_tokens
        if (type(member.exact) is not bool
                or any(type(count) is not int or count < 0 for count in (generated, observed))):
            raise ValueError("reward rows require exact flags and nonnegative token counts")
        costs.append(token_cost(generated, observed))
        exacts.append(member.exact)
    if any(member.env.state.aborted for member in members):
        return [0.0] * len(members)
    return group_rewards(costs, exacts, epsilon,
        successful_tool_calls=[member.env.state.successful_tool_calls for member in members],
        tool_call_failures=[member.tool_call_failures for member in members],
        termination_reasons=[getattr(member, "termination_reason", None) for member in members],
        verification_coverages=verification_coverages)



SCHEMA_VERSION = ENV_SCHEMA_VERSION
_MAX_GROUP_SIZE = 64
_MAX_GENERATED_IDS_PER_TURN = 16_384
_MAX_PARSED_CALLS_PER_TURN = _MAX_GENERATED_IDS_PER_TURN
_GENERATION_LIMIT_STOP_REASONS = {"length"}


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EpisodeRequest(APIModel):
    task_id: str = Field(min_length=1, max_length=200)
    group_id: str = Field(min_length=1, max_length=200)
    sample_id: str = Field(min_length=1, max_length=200)
    group_size: int = Field(ge=1, le=_MAX_GROUP_SIZE)
    validation: bool


class TurnRequest(APIModel):
    sequence: int = Field(ge=0)
    name: Literal["arc_action"] | None
    arguments: dict[str, Any]
    generated_ids: list[int] = Field(max_length=_MAX_GENERATED_IDS_PER_TURN)
    generation_stop_reason: str = Field(min_length=1, max_length=64, strict=True)
    generation_limit_reached: bool
    requested_max_tokens: int = Field(ge=1, le=_MAX_GENERATED_IDS_PER_TURN)
    parse_status: Literal[
        "valid", "no_call", "malformed", "unregistered_tool", "invalid_schema", "multiple_calls",
    ]
    requested_tool_name: str | None = Field(max_length=512, strict=True)
    parsed_call_count: int = Field(ge=0, le=_MAX_PARSED_CALLS_PER_TURN)
    parsed_tool_names: list[str] = Field(max_length=_MAX_PARSED_CALLS_PER_TURN)

    @model_validator(mode="after")
    def validate_turn_diagnostics(self) -> "TurnRequest":
        if len(self.generated_ids) > self.requested_max_tokens:
            raise ValueError("generated_ids exceed requested_max_tokens")
        limit_stop = self.generation_stop_reason in _GENERATION_LIMIT_STOP_REASONS
        expected_limit = len(self.generated_ids) == self.requested_max_tokens or limit_stop
        if self.generation_limit_reached != expected_limit:
            raise ValueError("generation_limit_reached disagrees with token count/stop reason")
        if len(self.parsed_tool_names) != self.parsed_call_count:
            raise ValueError("parsed_tool_names must match parsed_call_count")

        if self.parse_status == "valid":
            if (self.parsed_call_count != 1 or self.name not in NATIVE_TOOL_NAMES
                    or self.requested_tool_name != self.name
                    or self.parsed_tool_names != [self.name]):
                raise ValueError("valid parse status requires one matching registered tool call")
        else:
            if self.name is not None or self.arguments:
                raise ValueError("rejected or absent tool calls must not carry executable name/arguments")
            if self.parse_status == "no_call":
                if self.parsed_call_count or self.requested_tool_name is not None:
                    raise ValueError("no_call status cannot carry parsed calls or a requested tool name")
            elif self.parse_status == "unregistered_tool":
                if (self.parsed_call_count not in {0, 1} or not self.requested_tool_name
                        or (self.parsed_call_count == 1
                            and self.parsed_tool_names != [self.requested_tool_name])
                        or self.requested_tool_name in NATIVE_TOOL_NAMES):
                    raise ValueError("unregistered_tool status must identify one unregistered call")
            elif self.parse_status == "invalid_schema":
                if (self.parsed_call_count != 1 or self.requested_tool_name not in NATIVE_TOOL_NAMES
                        or self.parsed_tool_names != [self.requested_tool_name]):
                    raise ValueError("invalid_schema status must identify one registered call")
            elif self.parse_status == "multiple_calls" and self.parsed_call_count < 2:
                raise ValueError("multiple_calls status requires at least two parsed calls")
        return self


class BudgetReceipt(APIModel):
    token_budget: int = Field(ge=1)
    tokens_used: int = Field(ge=0)
    tokens_remaining: int = Field(ge=0)
    generated_token_budget: int | None = Field(default=None, ge=1)
    observation_token_budget: int | None = Field(default=None, ge=1)
    generated_tokens: int | None = Field(default=None, ge=0)
    observation_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_remaining(self) -> "BudgetReceipt":
        expected = max(0, self.token_budget - self.tokens_used)
        if self.tokens_remaining != expected:
            raise ValueError("tokens_remaining must equal max(0, token_budget - tokens_used)")
        values = (self.generated_token_budget, self.observation_token_budget,
                  self.generated_tokens, self.observation_tokens)
        if values != (None, None, None, None):
            if any(value is None for value in values):
                raise ValueError("separate budget receipt requires both quotas and counters")
            if (self.generated_token_budget + self.observation_token_budget != self.token_budget
                    or self.generated_tokens + self.observation_tokens != self.tokens_used):
                raise ValueError("separate budget receipt counters or quotas differ from totals")
        return self


class EpisodeCreatedResponse(BudgetReceipt):
    episode_id: str
    observation: str
    contract_fingerprint: str
    action_language_fingerprint: str


class TurnResponse(BudgetReceipt):
    observation: str
    observation_role: Literal["tool", "user", "none"]
    done: bool
    infra_abort: bool
    outcome: Literal["tool_result", "rejection", "episode_complete"]
    reason: Literal[
        "action_executed", "model_stop", "generation_limit",
        "malformed_tool_call", "unregistered_tool", "invalid_schema", "multiple_calls",
        "tool_error", "token_budget_exceeded",
        "generated_budget_exceeded", "observation_budget_exceeded",
    ]
    retryable: bool
    retries_remaining: int = Field(ge=0, lt=MAX_TOOL_CALL_FAILURES)

    @model_validator(mode="after")
    def validate_retry(self) -> "TurnResponse":
        if self.reason == "model_stop" and (
                not self.done or self.infra_abort or self.observation or self.observation_role != "none"
                or self.outcome != "episode_complete"):
            raise ValueError("model_stop requires a complete assistant turn without tool feedback")
        retry = self.outcome == "rejection" and not self.done and not self.infra_abort
        if self.retryable != retry or bool(self.retries_remaining) != retry:
            raise ValueError("retry fields must match a nonterminal rejection")
        return self


class FinishRequest(APIModel):
    reason: Literal["model_stop", "turn_limit", "context_limit", "completed", "infra_abort", "budget_reserve"]
    min_policy_version: int | None = None
    max_policy_version: int | None = None


@dataclass
class Episode:
    episode_id: str
    task_id: str
    group_id: str
    sample_id: str
    group_key: tuple[str, bool]
    token_budget: int
    env: ARCToolEnv | None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_sequence: int = 0
    turn_replays: dict[int, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    finish_hash: str | None = None
    finish_response: dict[str, Any] | None = None
    generated_tokens: int = 0
    tool_call_failures: int = 0
    forced_failure: bool = False
    termination_reason: str | None = None
    exact: bool = False
    attempt_exact: dict[str, dict[str, bool]] = field(default_factory=dict)
    per_target_exact: dict[str, bool] = field(default_factory=dict)
    terminal: bool = False
    finished: bool = False
    policy_version: int | None = None
    env_summary: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


@dataclass
class Group:
    key: tuple[str, bool]
    task_id: str
    group_size: int
    validation: bool
    created_at: float
    deadline: float
    members: dict[str, Episode] = field(default_factory=dict)
    invalid_reason: str | None = None
    policy_version: int | None = None
    completed_at: float | None = None
    timeout_pending: bool = False


def _fingerprint_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def _load_task_map(dataset_dir: Path, tasks: dict[str, TaskSpec] | None) -> dict[str, TaskSpec]:
    source = tasks if tasks is not None else {
        spec.task_id: spec for spec in load_tasks(
            pools=("arc2_train", "arc2_eval"), variants=True, dataset_dir=dataset_dir)
    }
    result: dict[str, TaskSpec] = {}
    for key, spec in source.items():
        if not isinstance(spec, TaskSpec):
            raise TypeError(f"tasks[{key!r}] must be a TaskSpec")
        result[str(key)] = spec
        result.setdefault(spec.task_id, spec)
    return result


def create_app(
    *,
    dataset_dir: Path,
    artifact_dir: Path,
    tokenizer: Any,
    contract: Contract,
    group_timeout: float = 1200,
    max_active_groups: int = 64,
    max_generated_tokens: int = 4096,
    overlong_buffer: int = 512,
    tasks: dict[str, TaskSpec] | None = None,
) -> FastAPI:
    """Create the ARC environment API.

    ``ARC_ENV_TOKEN`` is the sole authentication secret. The supplied tokenizer is mandatory:
    production accounting never substitutes a character-count estimate for model tokens.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for exact observation accounting")
    if group_timeout <= 0 or max_active_groups <= 0 or max_generated_tokens <= 0:
        raise ValueError("group_timeout, max_active_groups and max_generated_tokens must be positive")
    if overlong_buffer < 0 or overlong_buffer > max_generated_tokens:
        raise ValueError("overlong_buffer must be between zero and max_generated_tokens")

    # Framing is fixed by this service's tokenizer/template. Keep one immutable
    # measurement for the app lifetime; task/session/token accounting stays per episode.
    environment_token_framing = EnvironmentTokenFraming.measure(tokenizer)

    dataset_dir = Path(dataset_dir)
    artifact_dir = Path(artifact_dir)
    task_map = _load_task_map(dataset_dir, tasks)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, bool], Group] = {}
    episodes: dict[str, Episode] = {}
    registry_lock = asyncio.Lock()

    # Resolve the frozen runtime hash without exposing its path in the public health response.
    from .environment import DEFAULT_TOOLKIT

    from .prompt import load_toolkit_runtime, validate_toolkit_binding
    loaded_toolkit = load_toolkit_runtime(DEFAULT_TOOLKIT)
    action_language_fingerprint = validate_toolkit_binding(DEFAULT_TOOLKIT)["fingerprint"]
    toolkit_runtime = Path(loaded_toolkit.__file__).resolve()
    toolkit_fingerprint = (hashlib.sha256(toolkit_runtime.read_bytes()).hexdigest()
                           if toolkit_runtime.exists() else "unavailable")
    prompt_fingerprint = _fingerprint_payload(
        prompt_contract()
    )

    app = FastAPI(title="ARC environment", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.groups = groups
    app.state.episodes = episodes
    app.state.tasks = task_map
    app.state.registry_lock = registry_lock

    async def require_token(authorization: str | None = Header(default=None)) -> None:
        expected = os.environ.get("ARC_ENV_TOKEN")
        if not expected:
            raise HTTPException(status_code=401, detail="ARC environment token is not configured")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bearer token required")
        supplied = authorization[7:]
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    def append_audit(episode: Episode, event: dict[str, Any]) -> None:
        path = artifact_dir / "audit" / f"{episode.episode_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def episode_budget_receipt(episode: Episode) -> BudgetReceipt:
        assert episode.env is not None
        used = int(episode.env.tokens_used())
        return BudgetReceipt(
            token_budget=episode.token_budget,
            tokens_used=used,
            tokens_remaining=max(0, episode.token_budget - used),
            **({"generated_token_budget": episode.env.contract.generated_token_budget,
                "observation_token_budget": episode.env.contract.observation_token_budget,
                "generated_tokens": episode.env.state.out_tokens,
                "observation_tokens": episode.env.state.obs_tokens}
               if episode.env.contract.generated_token_budget is not None else {}),
        )

    def turn_response(
        episode: Episode,
        *,
        observation: str,
        observation_role: Literal["tool", "user", "none"],
        done: bool,
        infra_abort: bool,
        outcome: Literal["tool_result", "rejection", "episode_complete"],
        reason: Literal[
            "action_executed", "model_stop", "generation_limit",
            "malformed_tool_call", "unregistered_tool", "invalid_schema", "multiple_calls",
            "tool_error", "token_budget_exceeded",
            "generated_budget_exceeded", "observation_budget_exceeded",
        ],
        retryable: bool = False,
        retries_remaining: int = 0,
    ) -> dict[str, Any]:
        return TurnResponse(
            observation=observation,
            observation_role=observation_role,
            done=done,
            infra_abort=infra_abort,
            outcome=outcome,
            reason=reason,
            retryable=retryable,
            retries_remaining=retries_remaining,
            **episode_budget_receipt(episode).model_dump(),
        ).model_dump(mode="json")

    async def store_turn_response(
        episode: Episode,
        request: TurnRequest,
        payload_hash: str,
        response_data: dict[str, Any],
    ) -> dict[str, Any]:
        episode.turn_replays[request.sequence] = (payload_hash, response_data)
        episode.next_sequence += 1
        await asyncio.to_thread(append_audit, episode, {
            "event": "turn_result",
            "time": time.time(),
            "sequence": request.sequence,
            "generation_stop_reason": request.generation_stop_reason,
            "generation_limit_reached": request.generation_limit_reached,
            "requested_max_tokens": request.requested_max_tokens,
            "parse_status": request.parse_status,
            "requested_tool_name": request.requested_tool_name,
            "parsed_call_count": request.parsed_call_count,
            "parsed_tool_names": request.parsed_tool_names,
            "response": response_data,
        })
        return response_data

    def invalidate(group: Group, reason: str) -> None:
        if group.invalid_reason is None:
            group.invalid_reason = reason

    def expire_locked(now: float) -> None:
        """Expire groups on request boundaries; tombstones remain queryable as HTTP 409."""
        remove: list[tuple[str, bool]] = []
        for key, group in groups.items():
            group_complete = (len(group.members) == group.group_size
                              and all(member.finished for member in group.members.values()))
            if group.invalid_reason is None and not group_complete and now >= group.deadline:
                invalidate(group, "group timed out before every member finished")
                group.completed_at = now
                group.timeout_pending = True
            retain_until = (group.completed_at + group_timeout
                            if group.completed_at is not None else group.deadline + group_timeout)
            if now >= retain_until:
                remove.append(key)
        for key in remove:
            group = groups.pop(key)
            for episode in group.members.values():
                episodes.pop(episode.episode_id, None)

    def group_for_episode(episode: Episode) -> Group:
        group = groups.get(episode.group_key)
        if group is None:
            raise HTTPException(status_code=409, detail="Episode group is no longer available")
        if group.invalid_reason:
            raise HTTPException(status_code=409, detail="Episode group was discarded")
        return group

    def persist_episode_summary(episode: Episode) -> None:
        if episode.env is None:
            return
        summary = dict(episode.env_summary or {})
        summary.pop("root", None)
        payload = {
            "episode_id": episode.episode_id,
            "budget_policy": budget_policy_identity(),
            "task_id": episode.task_id,
            "group_id": episode.group_id,
            "sample_id": episode.sample_id,
            "validation": episode.group_key[1],
            "generated_tokens": episode.generated_tokens,
            "tool_call_failures": episode.tool_call_failures,
            "observation_tokens": episode.env.state.obs_tokens,
            "cost": episode.env.cost(),
            "exact": episode.exact,
            "attempt_exact": episode.attempt_exact,
            "per_target_exact": episode.per_target_exact,
            "reward": None if episode.result is None else episode.result["reward"],
            "termination_reason": episode.termination_reason,
            "policy_version": episode.policy_version,
            "contract_fingerprint": contract.fingerprint(),
            "environment": summary,
        }
        _safe_json_write(artifact_dir / "summaries" / f"{episode.episode_id}.json", payload)

    async def freeze_discarded_group(group: Group, reason: str) -> None:
        """Close and summarize all member artifacts after timeout or cancellation."""
        members = sorted(group.members.values(), key=lambda member: member.episode_id)
        acquired: list[asyncio.Lock] = []
        try:
            for member in members:
                await member.lock.acquire()
                acquired.append(member.lock)
            for member in members:
                member.result = None
                member.exact = False
                if member.env is not None:
                    if member.env.session is not None:
                        member.env.state.aborted = True
                        member.env.state.abort_reason = reason
                        member.env_summary = await asyncio.to_thread(member.env.close)
                    await asyncio.to_thread(persist_episode_summary, member)
                await asyncio.to_thread(append_audit, member, {
                    "event": "group_discarded", "time": time.time(), "reason": reason,
                })
        finally:
            for lock in reversed(acquired):
                lock.release()
        discarded_summary = {
            "group_id": group.key[0], "validation": group.validation,
            "task_id": group.task_id, "group_size": group.group_size,
            "discarded": True, "reason": reason,
        }
        await asyncio.to_thread(
            _safe_json_write,
            artifact_dir / "summaries" / f"group-{_fingerprint_payload(group.key)[:24]}.json",
            discarded_summary)

    async def sweep_expired_groups() -> None:
        async with registry_lock:
            expire_locked(time.monotonic())
            expired = [group for group in groups.values() if group.timeout_pending]
            for group in expired:
                group.timeout_pending = False
        for group in expired:
            await freeze_discarded_group(
                group, "group timed out before every member finished")

    def complete_group(group: Group) -> None:
        if group.invalid_reason or len(group.members) != group.group_size:
            return
        members = list(group.members.values())
        if not all(member.finished for member in members):
            return
        costs = [member.env.cost() for member in members]
        exacts = [member.exact for member in members]
        verification = [m.env.training_verification.summary() for m in members]
        coverages = [evidence["coverage"] for evidence in verification]
        spec = task_map.get(group.task_id)
        if spec is not None and spec.instruction is not None:
            # Generated curriculum uses the shared runtime reward without ARC length taper.
            rewards = generated_group_rewards(group, members, contract.epsilon,
                                              verification_coverages=coverages)
        else:
            base_rewards = group_rewards(costs, exacts, contract.epsilon,
                successful_tool_calls=[m.env.state.successful_tool_calls for m in members],
                tool_call_failures=[m.tool_call_failures for m in members],
                termination_reasons=[m.termination_reason for m in members],
                verification_coverages=coverages)
            rewards = base_rewards
        for member, reward, evidence in zip(members, rewards, verification, strict=True):
            member.result = {
                "reward": reward,
                "reward_schema": REWARD_SCHEMA,
                "attempt_exact": member.attempt_exact,
                "per_target_exact": member.per_target_exact,
                "competition_score": (sum(member.per_target_exact.values()) / len(member.per_target_exact)
                                      if member.per_target_exact else 0.0),
                "training_verification": evidence,
                "reward_components": {"exact": float(member.exact),
                    "verification": VERIFICATION_COEFFICIENT * evidence["coverage"] if member.exact else 0.0,
                    "efficiency": reward - 1.0 - VERIFICATION_COEFFICIENT * evidence["coverage"] if member.exact else 0.0,
                    "failure": reward if not member.exact else 0.0},
                "termination_reason": member.termination_reason,
                "successful_tool_calls": member.env.state.successful_tool_calls,
                "tool_call_failures": member.tool_call_failures,
                "exact": member.exact,
                "cost": round(member.env.cost(), 1),
                "generated_tokens": member.generated_tokens,
                "observation_tokens": member.env.state.obs_tokens,
                "group_id": member.group_id,
                "policy_version": member.policy_version,
                "contract_fingerprint": contract.fingerprint(),
            }
            persist_episode_summary(member)
        group.completed_at = time.monotonic()
        group_summary = {
            "reward_schema": REWARD_SCHEMA,
            "budget_policy": budget_policy_identity(),
            "group_id": group.key[0],
            "validation": group.validation,
            "task_id": group.task_id,
            "group_size": group.group_size,
            "policy_version": group.policy_version,
            "contract_fingerprint": contract.fingerprint(),
            "episodes": [
                {"sample_id": member.sample_id, "reward": member.result["reward"],
                 "successful_tool_calls": member.env.state.successful_tool_calls,
                 "tool_call_failures": member.tool_call_failures,
                 "exact": member.result["exact"], "cost": member.result["cost"],
                 "generated_tokens": member.generated_tokens,
                 "observation_tokens": member.env.state.obs_tokens}
                for member in members
            ],
        }
        _safe_json_write(
            artifact_dir / "summaries" / f"group-{_fingerprint_payload(group.key)[:24]}.json",
            group_summary)

    @app.get("/health", dependencies=[Depends(require_token)])
    async def health() -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "max_tool_call_failures": MAX_TOOL_CALL_FAILURES,
            "tool_call_failure_counting": TOOL_CALL_FAILURE_COUNTING,
            "reward_schema": REWARD_SCHEMA,
            "reward_coefficients": {"efficiency": contract.epsilon,
                                    "verification": VERIFICATION_COEFFICIENT},
            "budget_policy": budget_policy_identity(),
            "contract_fingerprint": contract.fingerprint(),
            "toolkit_fingerprint": toolkit_fingerprint,
            "action_language_fingerprint": action_language_fingerprint,
            "prompt_fingerprint": prompt_fingerprint,
            "active_groups": sum(g.invalid_reason is None and g.completed_at is None
                                 and time.monotonic() < g.deadline for g in groups.values()),
            "registered_groups": len(groups),
            "maximum_registered_groups": max_active_groups,
            "active_episodes": sum(not e.finished and not e.terminal for e in episodes.values()),
        }

    @app.post("/episodes", response_model=EpisodeCreatedResponse)
    async def create_episode(request: EpisodeRequest, _auth: None = Depends(require_token)):
        await sweep_expired_groups()
        now = time.monotonic()
        key = (request.group_id, request.validation)
        async with registry_lock:
            task = task_map.get(request.task_id)
            if task is None:
                raise HTTPException(status_code=422, detail="Unknown task_id")
            if request.validation and request.group_size != 1:
                raise HTTPException(status_code=422, detail="validation episodes require group_size=1")
            group = groups.get(key)
            if group is not None:
                if group.invalid_reason:
                    raise HTTPException(status_code=409, detail="Episode group was discarded")
                if (group.task_id != request.task_id or group.group_size != request.group_size
                        or group.validation != request.validation):
                    invalidate(group, "group members disagree on task, size, or validation")
                    group.completed_at = now
                    raise HTTPException(status_code=409, detail="Episode group metadata mismatch")
                if request.sample_id in group.members:
                    raise HTTPException(status_code=409, detail="sample_id must be unique within a group")
                if len(group.members) >= group.group_size:
                    invalidate(group, "group received more samples than group_size")
                    group.completed_at = now
                    raise HTTPException(status_code=409, detail="Episode group is already full")
            else:
                if len(groups) >= max_active_groups:
                    # Completed and expired entries have already been pruned above. Refuse
                    # new work rather than evicting a pending group and silently losing it.
                    raise HTTPException(status_code=429, detail="Maximum active episode groups reached")
                group = Group(key, request.task_id, request.group_size, request.validation,
                              created_at=now, deadline=now + group_timeout)
                groups[key] = group

            episode_id = uuid.uuid4().hex
            token_budget = contract.token_budget
            workdir = artifact_dir / "episodes" / episode_id
            episode = Episode(episode_id, request.task_id, request.group_id, request.sample_id,
                              key, token_budget, None)
            group.members[request.sample_id] = episode
            episodes[episode_id] = episode

        try:
            def create_and_reset() -> tuple[ARCToolEnv, str]:
                env = ARCToolEnv(task.challenge, task.targets, task.public_id,
                                 workdir=workdir, contract=contract, token_budget=token_budget,
                                 tokenizer=tokenizer,
                                 instruction=task.instruction, input_roles=task.input_roles,
                                 submission_targets=task.submission_targets, conventions=task.conventions,
                                 token_framing=environment_token_framing)
                return env, env.reset()

            episode.env, observation = await asyncio.to_thread(create_and_reset)
            await asyncio.to_thread(append_audit, episode, {
                "event": "episode_created", "time": time.time(), "episode_id": episode_id,
                "task_id": request.task_id, "group_id": request.group_id,
                "sample_id": request.sample_id, "group_size": request.group_size,
                "validation": request.validation, "observation": observation,
            })
        except Exception as exc:  # runtime/prompt failures are infrastructure, not task errors
            _LOGGER.exception("ARC episode initialization failed")
            async with registry_lock:
                current = groups.get(key)
                if current is not None:
                    invalidate(current, "episode initialization failed")
                    current.completed_at = time.monotonic()
            raise HTTPException(status_code=409, detail="Episode initialization failed") from exc
        return EpisodeCreatedResponse(
            episode_id=episode_id,
            observation=observation,
            contract_fingerprint=contract.fingerprint(),
            action_language_fingerprint=action_language_fingerprint,
            **episode_budget_receipt(episode).model_dump(),
        )

    @app.post("/episodes/{episode_id}/turn", response_model=TurnResponse)
    async def turn(episode_id: str, request: TurnRequest, _auth: None = Depends(require_token)):
        await sweep_expired_groups()
        async with registry_lock:
            episode = episodes.get(episode_id)
        if episode is None:
            raise HTTPException(status_code=404, detail="Unknown episode_id")
        async with episode.lock:
            async with registry_lock:
                group_for_episode(episode)
            payload_hash = _fingerprint_payload(request.model_dump(mode="json"))
            prior = episode.turn_replays.get(request.sequence)
            if prior is not None:
                if prior[0] != payload_hash:
                    raise HTTPException(status_code=409, detail="Turn sequence payload conflict")
                return prior[1]
            if episode.finished or episode.terminal:
                raise HTTPException(status_code=409, detail="Episode is already terminal")
            if request.sequence != episode.next_sequence:
                raise HTTPException(status_code=409, detail="Turn sequence must be the next sequence")

            assert episode.env is not None

            async def respond(
                *,
                observation: str,
                observation_role: Literal["tool", "user", "none"],
                done: bool,
                infra_abort: bool,
                outcome: Literal["tool_result", "rejection", "episode_complete"],
                reason: Literal[
                    "action_executed", "model_stop", "generation_limit",
                    "malformed_tool_call", "unregistered_tool", "invalid_schema", "multiple_calls",
                    "tool_error", "token_budget_exceeded",
                    "generated_budget_exceeded", "observation_budget_exceeded",
                ],
                retryable: bool = False,
                retries_remaining: int = 0,
            ) -> dict[str, Any]:
                if done:
                    initial_limit = (
                        request.sequence == 0 and request.generation_limit_reached
                        and request.requested_max_tokens == _MAX_GENERATED_IDS_PER_TURN
                        and len(request.generated_ids) == _MAX_GENERATED_IDS_PER_TURN
                    )
                    episode.termination_reason = (
                        "initial_generation_limit" if initial_limit else
                        "generation_limit" if request.generation_limit_reached else
                        "tool_failure_limit" if episode.tool_call_failures >= MAX_TOOL_CALL_FAILURES else reason
                    )
                data = turn_response(
                    episode,
                    observation=observation,
                    observation_role=observation_role,
                    done=done,
                    infra_abort=infra_abort,
                    outcome=outcome,
                    reason=reason,
                    retryable=retryable,
                    retries_remaining=retries_remaining,
                )
                return await store_turn_response(episode, request, payload_hash, data)

            args = request.arguments
            server_schema_issue = (
                tool_schema_issue(request.name, args)
                if request.parse_status == "valid" else None
            )
            server_schema_error = (server_schema_issue["code"]
                                   if server_schema_issue is not None else None)
            episode.generated_tokens += len(request.generated_ids)
            try:
                await asyncio.to_thread(episode.env.record_generation, len(request.generated_ids))
                await asyncio.to_thread(append_audit, episode, {
                    "event": "generation_received",
                    "time": time.time(),
                    "sequence": request.sequence,
                    "name": request.name,
                    "arguments": args,
                    "generated_ids": request.generated_ids,
                    "generation_stop_reason": request.generation_stop_reason,
                    "generation_limit_reached": request.generation_limit_reached,
                    "requested_max_tokens": request.requested_max_tokens,
                    "parse_status": request.parse_status,
                    "requested_tool_name": request.requested_tool_name,
                    "parsed_call_count": request.parsed_call_count,
                    "parsed_tool_names": request.parsed_tool_names,
                })
            except Exception as exc:
                await _abort_episode(episode, "generation accounting failed")
                raise HTTPException(status_code=409, detail="Episode group was discarded") from exc

            # The sampled output is charged before any mutation. An over-budget call fails as
            # policy behavior and cannot execute submit or any other toolkit action.
            if episode.env.budget_exhausted(exceeded=True):
                episode.forced_failure = True
                episode.exact = False
                episode.terminal = True
                await asyncio.to_thread(append_audit, episode, {
                    "event": "task_budget_exceeded",
                    "time": time.time(),
                    "sequence": request.sequence,
                    "tokens_used": episode.env.tokens_used(),
                    "token_budget": episode.token_budget,
                    "executed_action": False,
                })
                return await respond(
                    observation="",
                    observation_role="none",
                    done=True,
                    infra_abort=False,
                    outcome="episode_complete",
                    reason=episode.env.budget_limit_reason(),
                )

            try:
                def decode_generation() -> str:
                    return tokenizer.decode(
                        request.generated_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )

                generated_text = await asyncio.to_thread(decode_generation)
                await asyncio.to_thread(append_audit, episode, {
                    "event": "generation_decoded",
                    "time": time.time(),
                    "sequence": request.sequence,
                    "generated_text": generated_text,
                })
            except Exception as exc:
                await _abort_episode(episode, "tokenizer decode failed")
                raise HTTPException(status_code=409, detail="Episode group was discarded") from exc

            # Charge all sampled tokens, but never execute a truncated call.
            rejection_reason = turn_rejection_reason(
                request.parse_status, request.generation_stop_reason,
                request.generation_limit_reached, server_schema_error,
            )
            if is_model_stop(request.parse_status, request.generation_stop_reason,
                             request.generation_limit_reached):
                episode.terminal = True
                return await respond(
                    observation="", observation_role="none", done=True, infra_abort=False,
                    outcome="episode_complete", reason="model_stop",
                )
            if rejection_reason is not None:
                call_generation_failed = (
                    request.parse_status != "valid" or server_schema_error is not None
                ) and rejection_reason != "generation_limit"
                if call_generation_failed:
                    episode.tool_call_failures += 1
                done = (not call_generation_failed
                        or episode.tool_call_failures >= MAX_TOOL_CALL_FAILURES)
                role = "tool" if request.parse_status == "valid" else "user"
                feedback = tool_call_failure_feedback(
                    rejection_reason, parse_status=request.parse_status,
                    generation_limit_reached=request.generation_limit_reached,
                    requested_tool_name=request.requested_tool_name,
                    validation_message=(server_schema_issue.get("message")
                                        if rejection_reason == "invalid_schema"
                                        and server_schema_issue is not None else None),
                    tool_call_failures=episode.tool_call_failures, done=done,
                )
                retries_remaining = (MAX_TOOL_CALL_FAILURES - episode.tool_call_failures
                                     if not done else 0)
                feedback.update(tool_call_failures=episode.tool_call_failures,
                                max_tool_call_failures=MAX_TOOL_CALL_FAILURES)
                observation = await asyncio.to_thread(
                    episode.env.render_budget_observation,
                    feedback, role=role,
                )
                if episode.env.budget_exhausted():
                    rejection_reason = episode.env.budget_limit_reason()
                    done = True
                if done:
                    episode.forced_failure = True
                    episode.exact = False
                    episode.terminal = True
                await asyncio.to_thread(append_audit, episode, {
                    "event": "tool_rejected", "time": time.time(),
                    "sequence": request.sequence, "reason": rejection_reason,
                    "tool_call_failures": episode.tool_call_failures,
                })
                return await respond(observation=observation, observation_role=role if observation else "none",
                    done=done, infra_abort=False,
                    outcome="episode_complete" if done else "rejection", reason=rejection_reason,
                    retryable=not done,
                    retries_remaining=(MAX_TOOL_CALL_FAILURES - episode.tool_call_failures if not done else 0))

            retryable_tool_error = False
            try:
                step = await asyncio.to_thread(episode.env.step, args)
                observation = step.observation
                infra_abort = bool(step.info.get("aborted"))
                used = episode.env.tokens_used()
                over_budget = episode.env.budget_exhausted(exceeded=True)
                tool_failed = step.info.get("status") != "ok" and not infra_abort
                if tool_failed:
                    episode.tool_call_failures += 1
                tool_failure_limit_reached = (
                    tool_failed and episode.tool_call_failures >= MAX_TOOL_CALL_FAILURES
                )
                done = bool(step.done or infra_abort or tool_failure_limit_reached
                            or episode.env.budget_exhausted())
                if tool_failed and done:
                    episode.forced_failure = True
                    episode.exact = False
                if infra_abort:
                    await _abort_episode(episode, "tool environment infrastructure abort")
                if over_budget:
                    episode.forced_failure = True
                    episode.exact = False
                if done:
                    episode.terminal = True
                retryable_tool_error = tool_failed and not done
                outcome = ("rejection" if retryable_tool_error else
                           "episode_complete" if done or infra_abort else "tool_result")
                reason = ("tool_error" if tool_failed else
                          episode.env.budget_limit_reason() if (over_budget or
                            (episode.env.contract.generated_token_budget is not None and episode.env.budget_exhausted()))
                          else "action_executed")
            except HTTPException:
                raise
            except Exception as exc:
                await _abort_episode(episode, "tool execution failed")
                raise HTTPException(status_code=409, detail="Episode group was discarded") from exc

            return await respond(
                observation=observation,
                observation_role="tool" if observation else "none",
                done=done,
                infra_abort=infra_abort,
                outcome=outcome,
                reason=reason,
                retryable=retryable_tool_error,
                retries_remaining=(MAX_TOOL_CALL_FAILURES - episode.tool_call_failures
                                   if request.name == "arc_action" and retryable_tool_error else 0),
            )

    async def _abort_episode(episode: Episode, reason: str) -> None:
        episode.terminal = True
        async with registry_lock:
            group = groups.get(episode.group_key)
            if group is not None:
                invalidate(group, reason)
                group.completed_at = time.monotonic()
        if episode.env is not None and episode.env.session is not None:
            episode.exact = False
            episode.env_summary = await asyncio.to_thread(episode.env.close)
            await asyncio.to_thread(persist_episode_summary, episode)
        await asyncio.to_thread(append_audit, episode, {
            "event": "discarded", "time": time.time(), "reason": reason,
        })

    @app.post("/episodes/{episode_id}/finish")
    async def finish(episode_id: str, request: FinishRequest, _auth: None = Depends(require_token)):
        await sweep_expired_groups()
        async with registry_lock:
            episode = episodes.get(episode_id)
        if episode is None:
            raise HTTPException(status_code=404, detail="Unknown episode_id")
        async with episode.lock:
            async with registry_lock:
                group = group_for_episode(episode)
            payload_hash = _fingerprint_payload(request.model_dump(mode="json"))
            if episode.finished:
                if episode.finish_hash != payload_hash:
                    raise HTTPException(status_code=409, detail="Finish payload conflict")
                return episode.finish_response

            # A turn cap is a normal policy budget stop: score only real submissions.
            # Infrastructure failure and response truncation invalidate the trajectory.
            if request.reason in {"infra_abort", "context_limit"}:
                await _abort_episode(episode, f"episode ended with {request.reason}")
                raise HTTPException(status_code=409, detail="Episode group was discarded")

            if (request.min_policy_version is None or request.max_policy_version is None
                    or request.min_policy_version != request.max_policy_version):
                await _abort_episode(episode, "missing or mixed policy version within episode")
                raise HTTPException(status_code=409, detail="Episode group was discarded")

            version = request.min_policy_version
            async with registry_lock:
                if group.policy_version is not None and group.policy_version != version:
                    invalidate(group, "group members used different policy versions")
                    group.completed_at = time.monotonic()
                    mismatch = True
                else:
                    mismatch = False
                    group.policy_version = version
            if mismatch:
                await _abort_episode(episode, "group members used different policy versions")
                raise HTTPException(status_code=409, detail="Episode group was discarded")

            assert episode.env is not None
            # Score only the actual submitted grids, including an early normal model stop.
            attempts = await asyncio.to_thread(episode.env.score_attempts)
            eligible = not episode.env.budget_exhausted(exceeded=True) and not episode.env.state.aborted and not episode.forced_failure
            episode.attempt_exact = {target: {str(number): bool(value and eligible)
                                             for number, value in values.items()}
                                     for target, values in attempts.items()}
            episode.per_target_exact = {target: any(values.values())
                                        for target, values in episode.attempt_exact.items()}
            exact = bool(episode.per_target_exact) and all(episode.per_target_exact.values())
            episode.exact = bool(exact and not episode.env.budget_exhausted(exceeded=True)
                                 and not episode.env.state.aborted and not episode.forced_failure)
            if episode.env.state.aborted:
                await _abort_episode(episode, "tool environment infrastructure abort")
                raise HTTPException(status_code=409, detail="Episode group was discarded")

            if episode.termination_reason is None:
                episode.termination_reason = request.reason
            if (not episode.exact and episode.env.budget_exhausted()
                    and episode.termination_reason not in {
                        "initial_generation_limit", "generation_limit", "tool_failure_limit"}):
                # Tool observations also consume the episode budget. A final
                # action can exhaust it without a sampler length stop.
                episode.termination_reason = episode.env.budget_limit_reason()
            episode.policy_version = version
            episode.env_summary = await asyncio.to_thread(episode.env.close)
            finish_audit = {
                "event": "finished", "time": time.time(), "reason": request.reason,
                "policy_version": version, "exact": episode.exact,
                "generated_tokens": episode.generated_tokens,
                "observation_tokens": episode.env.state.obs_tokens,
                "cost": episode.env.cost(),
            }
            try:
                await asyncio.to_thread(persist_episode_summary, episode)
                await asyncio.to_thread(append_audit, episode, finish_audit)
            except Exception as exc:
                await _abort_episode(episode, "finish artifact persistence failed")
                raise HTTPException(status_code=409, detail="Episode group was discarded") from exc
            # A group may be completed by another member's concurrent finish.
            # Publish this member as finished only after its own summary and audit
            # are durable, so complete_group cannot race this summary's .tmp file.
            episode.finished = True
            episode.finish_hash = payload_hash
            response_data = {"finished": True, "group_ready": False}
            async with registry_lock:
                current = group_for_episode(episode)
                try:
                    await asyncio.to_thread(complete_group, current)
                except Exception as exc:
                    invalidate(current, "group result persistence failed")
                    current.completed_at = time.monotonic()
                    for member in current.members.values():
                        member.result = None
                    raise HTTPException(status_code=409, detail="Episode group was discarded") from exc
                response_data["group_ready"] = all(member.finished for member in current.members.values()) \
                    and len(current.members) == current.group_size
            episode.finish_response = response_data
            return response_data

    @app.get("/episodes/{episode_id}/result")
    async def result(episode_id: str, response: Response, _auth: None = Depends(require_token)):
        await sweep_expired_groups()
        async with registry_lock:
            episode = episodes.get(episode_id)
            if episode is None:
                raise HTTPException(status_code=404, detail="Unknown episode_id")
            group = groups.get(episode.group_key)
            if group is None or group.invalid_reason:
                raise HTTPException(status_code=409, detail="Episode group was discarded")
            if not all(member.finished for member in group.members.values()) \
                    or len(group.members) != group.group_size:
                response.status_code = 202
                return {"status": "pending"}
            if episode.result is None:
                response.status_code = 202
                return {"status": "pending"}
            return episode.result

    @app.delete("/episodes/{episode_id}")
    async def cancel_episode(episode_id: str, _auth: None = Depends(require_token)):
        await sweep_expired_groups()
        async with registry_lock:
            episode = episodes.get(episode_id)
            if episode is None:
                raise HTTPException(status_code=404, detail="Unknown episode_id")
            group = groups.get(episode.group_key)
            if group is None:
                raise HTTPException(status_code=409, detail="Episode group was discarded")
            invalidate(group, "episode explicitly cancelled")
            group.completed_at = time.monotonic()
            members = list(group.members.values())
        ordered_members = sorted(members, key=lambda member: member.episode_id)
        acquired: list[asyncio.Lock] = []
        try:
            for member in ordered_members:
                await member.lock.acquire()
                acquired.append(member.lock)
            for member in ordered_members:
                member.result = None
                member.exact = False
                if not member.finished and member.env is not None and member.env.session is not None:
                    member.env.state.aborted = True
                    member.env.state.abort_reason = "episode group explicitly cancelled"
                    member.env_summary = await asyncio.to_thread(member.env.close)
                await asyncio.to_thread(persist_episode_summary, member)
                await asyncio.to_thread(append_audit, member, {
                    "event": "cancelled" if member is episode else "group_discarded",
                    "time": time.time(), "reason": "episode explicitly cancelled",
                })
        finally:
            for lock in reversed(acquired):
                lock.release()
        discarded_summary = {
            "group_id": group.key[0], "validation": group.validation,
            "task_id": group.task_id, "group_size": group.group_size,
            "discarded": True, "reason": "episode explicitly cancelled",
        }
        await asyncio.to_thread(
            _safe_json_write,
            artifact_dir / "summaries" / f"group-{_fingerprint_payload(group.key)[:24]}.json",
            discarded_summary)
        return Response(status_code=204)

    return app
