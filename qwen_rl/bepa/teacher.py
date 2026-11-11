"""CPU-side DeepSeek teacher runner for verified BEPA source traces.

The runner owns only provider requests and the public source conversation. ARC
tool calls are executed by the central service, which retains the private
targets and performs the trusted verification before this module extracts a
student-facing plan.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import re
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, urlsplit

import httpx

from .contracts import ContractError, validate_teacher_plan
from .token_usage import raw_completion_tokens, uncached_input_tokens
from ..contracts.arc_tools import NATIVE_TOOL_NAMES, is_model_stop, tool_schema_error

DEFAULT_PROVIDER = "hp-hermes-router"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_CENTRAL_TOKEN_ENV = "BEPA_TEACHER_TOKEN"
MAX_PLAN_CHARS = 4_000
_LOGGER = logging.getLogger(__name__)
_JOB_ID = re.compile(r"[A-Za-z0-9_.-]{1,200}\Z")
CLAIM_RETRY_INITIAL_SECONDS = 0.25
CLAIM_RETRY_MAX_SECONDS = 5.0
CLAIM_ONCE_MAX_RETRIES = 5
PROVIDER_PROBE_TIMEOUT_SECONDS = 3.0
PROVIDER_READY_TTL_SECONDS = 10.0
PROVIDER_PROBE_RETRY_INITIAL_SECONDS = 1.0
PROVIDER_PROBE_RETRY_MAX_SECONDS = 30.0


def _claim_retry_delay(attempt: int) -> float:
    """Bound claim retry backoff while keeping the loop promptly cancellable."""
    return min(CLAIM_RETRY_MAX_SECONDS, CLAIM_RETRY_INITIAL_SECONDS * (2 ** min(attempt, 8)))


def _provider_probe_delay(attempt: int) -> float:
    """Back off readiness probes while the non-billable provider catalog is unavailable."""
    return min(PROVIDER_PROBE_RETRY_MAX_SECONDS,
               PROVIDER_PROBE_RETRY_INITIAL_SECONDS * (2 ** min(max(attempt, 0), 8)))


_PROVIDER_EXCEPTION_CATEGORIES = {
    "APIConnectionError": "connection_error",
    "APITimeoutError": "timeout",
    "APIStatusError": "http_status",
    "ConnectError": "connect_error",
    "ConnectTimeout": "connect_timeout",
    "PoolTimeout": "pool_timeout",
    "ReadError": "read_error",
    "ReadTimeout": "read_timeout",
    "RemoteProtocolError": "protocol_error",
    "WriteError": "write_error",
    "WriteTimeout": "write_timeout",
}


def _safe_provider_exception(exc: BaseException) -> tuple[str, int | None]:
    """Reduce SDK/transport exceptions to a fixed category and optional HTTP status."""
    category = "provider_error"
    status_code = None
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        current_status = getattr(current, "status_code", None)
        if type(current_status) is int and 100 <= current_status <= 599:
            status_code = current_status
        mapped = _PROVIDER_EXCEPTION_CATEGORIES.get(type(current).__name__)
        if mapped is not None:
            category = mapped
        current = current.__cause__ or current.__context__
    return category, status_code


def _provider_probe_failure_code(category: str, status_code: int | None) -> str:
    if status_code == 404:
        return "provider_models_route_missing"
    if status_code in {401, 403}:
        return "provider_auth_rejected"
    if status_code == 429:
        return "provider_rate_limited"
    if status_code is not None:
        return "provider_models_http_error"
    if category in {"connect_error", "connect_timeout", "connection_error"}:
        return "provider_connection_unavailable"
    if category in {"timeout", "read_timeout", "write_timeout", "pool_timeout"}:
        return "provider_probe_timeout"
    return "provider_probe_failed"


_SAFE_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TOOL_NAMES = NATIVE_TOOL_NAMES


class TeacherRunnerError(RuntimeError):
    """A bounded error category safe to return through the job control plane."""

    def __init__(
        self, code: str, *, status_code: int | None = None,
        endpoint: str | None = None, category: str | None = None,
    ) -> None:
        if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code):
            code = "teacher_runner_error"
        self.code = code
        self.status_code = status_code if type(status_code) is int and 100 <= status_code <= 599 else None
        self.endpoint = endpoint if endpoint in {"complete", "fail"} else None
        self.category = category if category in {
            "http_rejected", "lease_or_state_conflict", "transport_outcome_unknown",
            "response_invalid", "local_persistence", "provider_readiness",
        } else None
        super().__init__(code)

    def safe_diagnostic(self) -> dict[str, Any]:
        # Expose fixed control-plane metadata only; never include response text.
        value: dict[str, Any] = {"category": self.category or "runner_error", "code": self.code}
        if self.endpoint is not None:
            value["endpoint"] = self.endpoint
        if self.status_code is not None:
            value["status_code"] = self.status_code
        return value


@dataclass(frozen=True)
class TeacherProviderConfig:
    """OpenAI-compatible provider settings; credentials are read only from env."""

    base_url: str
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout_seconds: float = 180.0
    max_completion_tokens: int = 32_768

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("provider base_url is required")
        parsed = urlsplit(self.base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("provider base_url must be an absolute HTTP URL without embedded credentials or query")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("provider model is required")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider name is required")
        if not isinstance(self.api_key_env, str) or not _SAFE_ENV_NAME.fullmatch(self.api_key_env):
            raise ValueError("api_key_env must be an environment variable name")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0):
            raise ValueError("provider timeout_seconds must be finite and positive")
        if type(self.max_completion_tokens) is not int or self.max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be a positive integer")

@dataclass
class TeacherResult:
    """Auditable public teacher output ready for the coordinator's complete call."""

    job_id: str
    session_id: str
    native_messages: list[dict[str, Any]]
    usage_calls: list[dict[str, Any]]
    termination: str
    source_accepted: bool = False
    source_receipt: dict[str, Any] | None = None
    plan: str | None = None
    plan_response: dict[str, Any] | None = None
    token_total: int | None = 0
    usage_known: bool = True
    provider: str = DEFAULT_PROVIDER
    requested_model: str = DEFAULT_MODEL
    actual_models: list[str] = field(default_factory=list)

    def completion_payload(self) -> dict[str, Any]:
        """Return the public artifact payload; it contains no verifier answer or score."""
        return {
            "session_id": self.session_id,
            "native_messages": copy.deepcopy(self.native_messages),
            "plan": self.plan,
            "plan_response": copy.deepcopy(self.plan_response),
            "usage_calls": copy.deepcopy(self.usage_calls),
            "token_total": self.token_total,
            "usage_known": self.usage_known,
            "termination": self.termination,
            "source_receipt": copy.deepcopy(self.source_receipt),
            "source_accepted": self.source_accepted,
            "provider": self.provider,
            "requested_model": self.requested_model,
            "actual_models": list(self.actual_models),
        }


@dataclass
class _ProviderStreamResult:
    assistant: dict[str, Any] | None
    usage: dict[str, Any] | None
    actual_model: str | None
    response_id: str | None
    finish_reason: str | None
    complete: bool
    error_code: str | None = None
    error_category: str | None = None
    status_code: int | None = None
    chunks_received: int = 0


@dataclass
class _TokenBudget:
    max_job_tokens: int
    budget_grant_tokens: int
    spent: int = 0
    usage_known: bool = True
    allow_unknown: bool = False
    retry_attempts: int = 0

    @property
    def remaining(self) -> int:
        return max(0, min(self.max_job_tokens, self.budget_grant_tokens) - self.spent)

    def completion_limit(self, request: Mapping[str, Any], configured_cap: int) -> int | None:
        """Reserve a UTF-8 byte upper bound for prompt tokens before a paid call.

        The provider does not expose a shared cross-model tokenizer. Reserving one
        input token per serialized UTF-8 byte, plus framing slack, is deliberately
        conservative for the configured DeepSeek/OpenAI-compatible route.
        """
        if not self.usage_known and not self.allow_unknown:
            return None
        serialized = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        input_upper_bound = len(serialized.encode("utf-8")) + 16
        if input_upper_bound > self.remaining:
            return None
        return configured_cap

    def record(self, usage: Any) -> int | None:
        input_tokens = uncached_input_tokens(usage)
        if input_tokens is None:
            self.usage_known = False
            return None
        self.spent += input_tokens
        return input_tokens


def _integer(value: Any) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _strict_json_loads(value: str | bytes) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("invalid JSON number")),
    )


def _claimed_budget(
    job: Mapping[str, Any], source: Mapping[str, Any], *,
    configured_max_job_tokens: int | None = None,
    configured_max_total_tokens: int | None = None,
) -> _TokenBudget:
    max_job = _integer(job.get("max_job_tokens"))
    job_grant = _integer(job.get("budget_remaining"))
    source_grant = _integer(source.get("budget_remaining"))
    if max_job is not None and configured_max_job_tokens is not None:
        max_job = min(max_job, configured_max_job_tokens)
    grants = [value for value in (job_grant, source_grant) if value is not None]
    budget_grant = min(grants) if len(grants) == 2 else None
    if budget_grant is not None and configured_max_total_tokens is not None:
        budget_grant = min(budget_grant, configured_max_total_tokens)
    if (max_job is None or max_job < 1 or job_grant is None or source_grant is None
            or job_grant < 1 or source_grant < 1 or budget_grant is None or budget_grant < 1):
        raise TeacherRunnerError("token_budget_missing")
    return _TokenBudget(max_job_tokens=max_job, budget_grant_tokens=budget_grant)


def _native_tool_names(tools: Any) -> set[str]:
    if not isinstance(tools, list):
        raise TeacherRunnerError("native_tool_contract_invalid")
    names: list[str] = []
    for item in tools:
        if not isinstance(item, Mapping) or item.get("type") != "function":
            raise TeacherRunnerError("native_tool_contract_invalid")
        function = item.get("function")
        if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
            raise TeacherRunnerError("native_tool_contract_invalid")
        names.append(function["name"])
    if len(names) != len(_TOOL_NAMES) or set(names) != _TOOL_NAMES:
        raise TeacherRunnerError("native_tool_contract_invalid")
    return set(names)


def _initial_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise TeacherRunnerError("public_prompt_invalid")
    messages: list[dict[str, Any]] = []
    for message in value:
        if (not isinstance(message, Mapping) or message.get("role") not in {"system", "developer", "user"}
                or not isinstance(message.get("content"), (str, list))):
            raise TeacherRunnerError("public_prompt_invalid")
        messages.append(copy.deepcopy(dict(message)))
    if not any(message["role"] == "user" for message in messages):
        raise TeacherRunnerError("public_prompt_invalid")
    return messages


def _usage_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json", exclude_none=True)
        return dumped if isinstance(dumped, dict) else None
    return None


async def _stream_completion(
    client: Any,
    request: Mapping[str, Any],
    *,
    request_id: str,
    timeout_seconds: float,
) -> _ProviderStreamResult:
    """Accumulate a real Chat Completions stream using the SDK's SSE decoder."""
    kwargs = {key: value for key, value in request.items()
              if key not in {"stream", "stream_options"}}
    kwargs.update(
        stream=True,
        stream_options={"include_usage": True},
        extra_headers={"X-Client-Request-Id": request_id},
        timeout=timeout_seconds,
    )
    stream = None
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None
    actual_model = response_id = finish_reason = None
    error_code = None
    error_category = None
    status_code = None
    chunks_received = 0
    try:
        stream = await client.chat.completions.create(**kwargs)
        response_status = getattr(getattr(stream, 'response', None), 'status_code', None)
        if type(response_status) is int and 100 <= response_status <= 599:
            status_code = response_status
        async for chunk in stream:
            chunks_received += 1
            model = getattr(chunk, "model", None)
            if isinstance(model, str):
                actual_model = model
            response_id_value = getattr(chunk, "id", None)
            if isinstance(response_id_value, str):
                response_id = response_id_value
            usage_value = getattr(chunk, "usage", None)
            if usage_value is not None:
                usage = _usage_mapping(usage_value)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            if len(choices) != 1:
                error_code = "provider_stream_invalid"
                continue
            choice = choices[0]
            finish = getattr(choice, "finish_reason", None)
            if isinstance(finish, str):
                finish_reason = finish
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            visible = getattr(delta, "content", None)
            if visible is not None:
                if not isinstance(visible, str):
                    error_code = "provider_stream_invalid"
                else:
                    content.append(visible)
            delta_fields = delta.model_dump(mode="json", exclude_unset=True)
            reasoning_piece = delta_fields.get("reasoning_content")
            if reasoning_piece is not None:
                if not isinstance(reasoning_piece, str):
                    error_code = "provider_stream_invalid"
                else:
                    reasoning.append(reasoning_piece)
            for call in delta_fields.get("tool_calls", []) or []:
                if not isinstance(call, Mapping) or type(call.get("index")) is not int:
                    error_code = "provider_stream_invalid"
                    continue
                index = call["index"]
                accumulated = tool_calls.setdefault(index, {
                    "id": None, "type": "function", "name": "", "arguments": "",
                })
                if isinstance(call.get("id"), str):
                    accumulated["id"] = call["id"]
                call_type = call.get("type")
                if isinstance(call_type, str):
                    accumulated["type"] = call_type
                function = call.get("function")
                if function is not None:
                    if not isinstance(function, Mapping):
                        error_code = "provider_stream_invalid"
                        continue
                    name_piece = function.get("name")
                    argument_piece = function.get("arguments")
                    if name_piece is not None:
                        if isinstance(name_piece, str):
                            accumulated["name"] += name_piece
                        else:
                            error_code = "provider_stream_invalid"
                    if argument_piece is not None:
                        if isinstance(argument_piece, str):
                            accumulated["arguments"] += argument_piece
                        else:
                            error_code = "provider_stream_invalid"
    except Exception as exc:
        error_code = "provider_stream_unknown"
        error_category, exception_status = _safe_provider_exception(exc)
        if exception_status is not None:
            status_code = exception_status
    finally:
        if stream is not None:
            try:
                await stream.close()
            except Exception as exc:
                if error_code is None:
                    error_code = "provider_stream_unknown"
                    error_category, exception_status = _safe_provider_exception(exc)
                    if exception_status is not None:
                        status_code = exception_status

    complete = error_code is None and finish_reason is not None
    if error_code is None and not complete:
        error_code = "provider_finish_missing"
        error_category = "stream_ended_without_finish"
    elif complete and usage is None:
        error_code = "provider_usage_missing"
        error_category = "usage_receipt_missing"
    assistant: dict[str, Any] | None = None
    if content or reasoning or tool_calls or complete:
        assistant = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            assistant["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            normalized = []
            for index in sorted(tool_calls):
                call = tool_calls[index]
                # Native Chat Completions history requires the original JSON string.
                # Parse a separate copy only when dispatching the central tool.
                arguments = call["arguments"]
                normalized.append({
                    "id": call["id"], "type": call["type"],
                    "function": {"name": call["name"], "arguments": arguments},
                })
            assistant["tool_calls"] = normalized
    return _ProviderStreamResult(
        assistant=assistant, usage=usage, actual_model=actual_model, response_id=response_id,
        finish_reason=finish_reason, complete=complete, error_code=error_code,
        error_category=error_category, status_code=status_code, chunks_received=chunks_received,
    )


def _validate_tool_call(assistant: Mapping[str, Any], allowed: set[str]) -> tuple[str, str, dict[str, Any]]:
    calls = assistant.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise TeacherRunnerError("native_tool_turn_invalid")
    call = calls[0]
    call_id = call.get("id")
    function = call.get("function")
    if not isinstance(call_id, str) or not call_id or not isinstance(function, Mapping):
        raise TeacherRunnerError("native_tool_turn_invalid")
    name = function.get("name")
    raw_arguments = function.get("arguments")
    if name not in allowed or not isinstance(raw_arguments, str):
        raise TeacherRunnerError("native_tool_turn_invalid")
    try:
        arguments = _strict_json_loads(raw_arguments)
    except (ValueError, TypeError):
        raise TeacherRunnerError("native_tool_arguments_invalid") from None
    if not isinstance(arguments, dict):
        raise TeacherRunnerError("native_tool_arguments_invalid")
    if tool_schema_error(name, arguments) is not None:
        raise TeacherRunnerError("native_tool_arguments_invalid")
    return call_id, name, arguments


def _tool_observation(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise TeacherRunnerError("tool_response_invalid") from None


def _source_receipt(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    # Keep only public identity/attestation handles. In particular, never return
    # exact-match details, target outputs, or private scores to the teacher process.
    allowed = {"receipt_id", "source_hash", "session_id", "accepted_at"}
    return {key: copy.deepcopy(value[key]) for key in allowed if key in value}


_PLAN_SYSTEM = (
    "Turn the accepted public source trace into a concise student-facing discovery scaffold. Help the "
    "student select useful observations and execute a testable investigation. For inductive tasks, leave "
    "the consequential rule selection and input-dependent parameter inference to the student. Preserve "
    "conditions explicitly given by the task. Use real source evidence to choose the useful "
    "representation and the smallest set of measurements that can advance the task. You may name "
    "representations, generic calculations, registered tools and the intermediate artifacts they consume "
    "and produce. Do not disclose the winning transformation, decisive task-specific constants, selected "
    "placement rule or complete construction recipe, including by disguising the answer as a question. "
    "Explain generic coordinate or bbox calculations only where useful; obtain actual values from the "
    "student's own tool results. Distinguish rule constants the student infers from public demonstrations"
    " from instance-dependent parameters recomputed from each input; do not require a newly introduced "
    "output color to be derived from the input palette. Write ordered guidance:\n1. Start with one focused"
    " comparison of public training inputs and outputs. State the feature to measure and how its possible"
    " outcomes affect the next decision. Omit unrelated statistics and cautions. Prefer saved "
    "tool-produced differences, pixel/object tables and numeric results over manually transcribing cells,"
    " counting, or doing coordinate arithmetic in prose. Name the relevant tool, the observation artifact"
    " it returns and the next use of that artifact. Observation/calculation tools documented in the "
    "source public catalog may be prescribed even if not called in the source; label them as future "
    "student actions, not witnessed successes. Check each named tool against its documented input/output "
    "kinds and purpose. A single-grid pixel table is not a two-grid difference report; selection is not "
    "distance measurement. Do not invent capabilities or conversions; consult tools help when a "
    "connection is uncertain. Use inspect to view saved results or targeted grid regions; do not replace "
    "an available structured measurement with full-grid transcription. When grouping is material, "
    "distinguish cue/marker components from edited cells and surrounding context before measuring their "
    "relationships. Infer color roles from the observed changes; do not assume the most frequent color is"
    " the semantic background. Identify the spacing or relationship measurements that can test whether "
    "separate cue components belong together; do not silently equate a group with a directly connected "
    "component.\n2. Ask the student to state a concise candidate and its observed basis. If a material "
    "rival exists, prescribe one focused comparison on a relevant demonstration. Explain the conditional "
    "evidence, without naming the winner. Do not force a rival or a failed guess. If demonstrations do "
    "not distinguish choices, preserve that uncertainty rather than repeat the same check. Public "
    "test-input structure is supporting evidence, not a hidden-answer oracle.\n3. Once observations "
    "support an executable candidate, construct a prediction from one training INPUT and call "
    "verify_example early. Describe the required intermediate artifacts and useful source-supported tool "
    "connections. For construction, list only tools actually invoked successfully in the source, using "
    "their witnessed input/output kinds. Do not list other construction tools, even as optional examples."
    " Prefer the witnessed batch form over one call per cell, and omit speculative alternative chains. "
    "Never construct a prediction from a training output or its descendants. Explicitly label recommended"
    " artifact paths as diagnostic (may read training outputs) or input-only (may feed prediction). Infer"
    " the candidate from diagnostic evidence, then apply it to fresh input-only measurements to compute "
    "edit selections and coordinates. Never instruct the student to obtain edit coordinates from a diff "
    "report, even if a later sentence forbids that. Reports or selections computed using training outputs"
    " are diagnostic only; recompute prediction selections and coordinates from the training input alone,"
    " without consuming those diagnostic artifacts. Do not require exhaustive analysis of every example "
    "before this first check. Every plan must explicitly say that after the first training prediction "
    "passes, construct and verify predictions for all remaining applicable demonstrations before any "
    "submission, even if no check failed.\n4. On failure, use shape_match and pixel_diff_count, then "
    "inspect only the relevant mismatch. Identify which assumption the comparison tests, revise from real"
    " evidence, and rebuild. Once a candidate passes, verify every applicable demonstration; recheck "
    "every demonstration after a rule change. Do not treat protocol recovery as rule discovery or repeat "
    "observations already resolved.\n5. Inspect each required test input afresh, recompute its parameters,"
    " apply the supported procedure and submit every target. Follow the permitted submission strategy; do"
    " not claim uniqueness merely because demonstrations pass. Distinguish actions witnessed in the "
    "source from checks prescribed for the student. Do not invent observations, failures or labeled "
    "counterexamples. Keep the student's short candidate/evidence record understandable after guidance is"
    " removed; omit private chain-of-thought. Name tools and data dependencies in prose. Do not write any"
    " tool invocation or argument template, including parentheses after tool names, symbolic arguments, "
    "ellipses or placeholder paths. Do not copy source artifact IDs, hidden answers or precomputed test "
    "coordinates, or recommend source-failed/unverified construction alternatives. Tools execute through "
    "arc_action; finish with a normal response after submission, not arc_report. For explicit-instruction"
    " tasks without demonstrations, follow the given conditions directly; do not invent a discovery "
    "exercise or verification examples. If the source cannot support an actionable investigation, return "
    "an empty plan. Prefer the shortest task-specific plan that preserves the meaningful decisions; never"
    " exceed 4000 characters. Return exactly one JSON object with one string field: {\"plan\":\"...\"}. "
)


class TeacherRunner:
    """Run one claimed BEPA source job using a configured DeepSeek-compatible route.

    The coordinator remains responsible for claim polling, heartbeat, complete/fail,
    durable usage aggregation, and the global budget ledger. A claim must carry its
    fence and a bounded token grant. Partial streams have bounded retries before
    any tool execution; tool side effects are never automatically replayed.
    """

    def __init__(
        self,
        *,
        central_base_url: str,
        provider: TeacherProviderConfig,
        evidence_dir: str | os.PathLike[str] | None = None,
        central_token_env: str = DEFAULT_CENTRAL_TOKEN_ENV,
        central_timeout_seconds: float = 30.0,
        max_job_tokens: int | None = None,
        max_total_tokens: int | None = None,
        job_timeout_seconds: float | None = None,
        central_transport: httpx.AsyncBaseTransport | None = None,
        provider_transport: Any | None = None,
    ) -> None:
        if not isinstance(central_base_url, str) or not central_base_url.strip():
            raise ValueError("central_base_url is required")
        parsed = urlsplit(central_base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("central_base_url must be an absolute HTTP URL without credentials or query")
        if not isinstance(central_token_env, str) or not _SAFE_ENV_NAME.fullmatch(central_token_env):
            raise ValueError("central_token_env must be an environment variable name")
        if (isinstance(central_timeout_seconds, bool) or not isinstance(central_timeout_seconds, (int, float))
                or not math.isfinite(float(central_timeout_seconds)) or central_timeout_seconds <= 0):
            raise ValueError("central_timeout_seconds must be finite and positive")
        for name, value in (("max_job_tokens", max_job_tokens), ("max_total_tokens", max_total_tokens)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer when provided")
        if max_job_tokens is not None and max_total_tokens is not None and max_total_tokens < max_job_tokens:
            raise ValueError("max_total_tokens must be at least max_job_tokens")
        if (job_timeout_seconds is not None
                and (isinstance(job_timeout_seconds, bool) or not isinstance(job_timeout_seconds, (int, float))
                     or not math.isfinite(float(job_timeout_seconds)) or job_timeout_seconds <= 0)):
            raise ValueError("job_timeout_seconds must be finite and positive when provided")
        self.central_base_url = central_base_url.rstrip("/")
        self.provider_config = provider
        if evidence_dir is not None and (
                not isinstance(evidence_dir, (str, os.PathLike)) or not os.fspath(evidence_dir)):
            raise ValueError("evidence_dir must be a non-empty filesystem path")
        self.evidence_dir = Path(evidence_dir) if evidence_dir is not None else None
        self.central_token_env = central_token_env
        self.central_timeout_seconds = float(central_timeout_seconds)
        self.max_job_tokens = max_job_tokens
        self.max_total_tokens = max_total_tokens
        self.job_timeout_seconds = float(job_timeout_seconds) if job_timeout_seconds is not None else None
        self.central_transport = central_transport
        self.provider_transport = provider_transport

    def _provider_client(self, api_key: str) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise TeacherRunnerError("provider_sdk_unavailable") from None
        http_client = None
        if self.provider_transport is not None:
            try:
                import httpx2 as sdk_httpx
            except ImportError:
                # The pinned teacher image uses httpx2 through OpenAI 3.x. The
                # fallback supports the developer host's OpenAI 2.x test SDK.
                sdk_httpx = httpx
            http_client = sdk_httpx.AsyncClient(
                timeout=self.provider_config.timeout_seconds,
                transport=self.provider_transport,
            )
        return AsyncOpenAI(
            api_key=api_key,
            base_url=self.provider_config.base_url,
            timeout=self.provider_config.timeout_seconds,
            max_retries=0,
            http_client=http_client,
        )

    async def _probe_provider(self, api_key: str) -> None:
        """Check the authenticated, non-billable model catalog before claiming a job."""
        try:
            client = self._provider_client(api_key)
        except TeacherRunnerError as exc:
            raise TeacherRunnerError(exc.code, category="provider_readiness") from None
        try:
            timeout = min(PROVIDER_PROBE_TIMEOUT_SECONDS, self.provider_config.timeout_seconds)
            try:
                catalog = await client.models.list(timeout=timeout)
            except Exception as exc:
                category, status_code = _safe_provider_exception(exc)
                raise TeacherRunnerError(
                    _provider_probe_failure_code(category, status_code),
                    status_code=status_code,
                    category="provider_readiness",
                ) from None
            rows = getattr(catalog, "data", None)
            if not isinstance(rows, (list, tuple)):
                raise TeacherRunnerError(
                    "provider_models_response_invalid", category="provider_readiness",
                )
            available = {
                model_id for row in rows
                if isinstance(model_id := (row.get("id") if isinstance(row, Mapping)
                                           else getattr(row, "id", None)), str)
            }
            if self.provider_config.model not in available:
                raise TeacherRunnerError(
                    "provider_model_unavailable", category="provider_readiness",
                )
        finally:
            try:
                await client.close()
            except Exception:
                pass

    async def _provider_call(self, client, request, *, phase, central_client, fence,
                             worker_id, deadline, budget, result, max_retries=0):
        for attempt in range(max_retries + 1):
            streamed, tokens = await self._provider_call_once(
                client, request, phase=phase, central_client=central_client, fence=fence,
                worker_id=worker_id, deadline=deadline, budget=budget, result=result)
            retryable = (streamed.error_code in {'provider_finish_missing', 'provider_stream_unknown'}
                         and (streamed.status_code is None or streamed.status_code == 200
                              or streamed.status_code >= 500))
            if (not retryable or attempt == max_retries or budget.retry_attempts >= max_retries
                    or not (budget.allow_unknown or budget.usage_known)
                    or budget.completion_limit(request, request['max_tokens']) is None
                    or deadline - time.monotonic() <= 0.25):
                return streamed, tokens
            # A new billed attempt with a distinct call ID; no tool from the partial turn ran.
            budget.retry_attempts += 1
            await asyncio.sleep(0.25)
        raise AssertionError('unreachable provider retry state')

    async def _provider_call_once(
        self,
        client: Any,
        request: Mapping[str, Any],
        *,
        phase: str,
        central_client: httpx.AsyncClient,
        fence: int,
        worker_id: str,
        deadline: float,
        budget: _TokenBudget,
        result: TeacherResult,
    ) -> tuple[_ProviderStreamResult, int | None]:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TeacherRunnerError("job_deadline_expired")
        prefix = result.job_id if fence == 1 else f'{result.job_id}-fence-{fence}'
        request_id = f"{prefix}-call-{len(result.usage_calls)}"
        streamed = await _stream_completion(
            client,
            request,
            request_id=request_id,
            timeout_seconds=min(self.provider_config.timeout_seconds, remaining_seconds),
        )
        token_count = budget.record(streamed.usage)
        receipt = self._usage_receipt(
            len(result.usage_calls), phase, streamed.usage, token_count, streamed.actual_model,
            request_id=request_id, response_id=streamed.response_id,
            finish_reason=streamed.finish_reason, stream_complete=streamed.complete,
            provider_error_category=streamed.error_category,
            provider_status_code=streamed.status_code,
        )
        receipt['provider_error_code'] = streamed.error_code
        receipt['stream_evidence'] = {
            'chunks_received': streamed.chunks_received,
            'finish_observed': streamed.finish_reason is not None,
            'usage_observed': streamed.usage is not None,
            'content_chars': len((streamed.assistant or {}).get('content') or ''),
            'reasoning_chars': len((streamed.assistant or {}).get('reasoning_content') or ''),
            'tool_call_count': len((streamed.assistant or {}).get('tool_calls') or []),
        }
        result.usage_calls.append(receipt)
        if streamed.actual_model and streamed.actual_model not in result.actual_models:
            result.actual_models.append(streamed.actual_model)
        self._sync_budget_result(result, budget)
        # The receipt is durable before any tool execution or next paid call.
        acknowledged = await self._central_post(central_client, result.job_id, 'provider-receipt',
                                 {'fence': fence, 'worker_id': worker_id, 'receipt': receipt})
        ref = acknowledged.get('receipt_ref')
        if (not isinstance(ref, str) or re.fullmatch(r'[0-9a-f]{64}', ref) is None
                or acknowledged.get('confirmed_tokens') != token_count):
            raise TeacherRunnerError('provider_receipt_ack_invalid')
        return streamed, token_count

    async def run_claimed_job(
        self,
        job: Mapping[str, Any],
        *,
        central_client: httpx.AsyncClient | None = None,
    ) -> TeacherResult:
        """Solve and verify a claimed public ARC task, then extract a concise plan."""
        if not isinstance(job, Mapping):
            raise TeacherRunnerError("job_contract_invalid")
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise TeacherRunnerError("job_id_invalid")
        worker_id = job.get("worker_id")
        task_id = job.get("task_id")
        public_task = job.get("public_task")
        if not isinstance(worker_id, str) or not _JOB_ID.fullmatch(worker_id):
            raise TeacherRunnerError("job_worker_invalid")
        if not isinstance(task_id, str) or not task_id or not isinstance(public_task, Mapping):
            raise TeacherRunnerError("job_public_task_invalid")
        actual_fence = job.get("fence")
        if type(actual_fence) is not int or actual_fence < 1:
            raise TeacherRunnerError("job_fence_missing")
        if (_integer(job.get("max_job_tokens")) is None or _integer(job.get("max_job_tokens")) < 1
                or _integer(job.get("budget_remaining")) is None or _integer(job.get("budget_remaining")) < 1):
            raise TeacherRunnerError("token_budget_missing")
        deadline = self._deadline(job)
        if deadline <= time.monotonic():
            raise TeacherRunnerError("job_deadline_expired")

        provider_key = os.environ.get(self.provider_config.api_key_env)
        if not provider_key:
            raise TeacherRunnerError("provider_credential_missing")
        central_token = os.environ.get(self.central_token_env)
        if not central_token:
            raise TeacherRunnerError("central_credential_missing")

        central_headers = {"Authorization": f"Bearer {central_token}"}
        owns_central_client = central_client is None
        if central_client is None:
            central_client = httpx.AsyncClient(
                headers=central_headers, timeout=httpx.Timeout(self.central_timeout_seconds),
                transport=self.central_transport,
            )
        provider_client = self._provider_client(provider_key)
        result: TeacherResult | None = None
        try:
            source = await self._central_post(
                central_client, job_id, "source-start", {"fence": actual_fence},
            )
            session_id = source.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise TeacherRunnerError("source_session_invalid")
            messages = _initial_messages(source.get("messages"))
            self_check = (
                "\nFollow explicit task instructions directly, including their limits, stopping conditions, and "
                "required submissions. Preserve the registered ARC tool protocol and inspect real tool results "
                "instead of assuming an action succeeded. For input/output examples, infer the transformation "
                "from the observed changes and choose a useful representation, such as objects, partitions, "
                "rows/columns, or sequence/order, based on the task. After a candidate fits the demonstrations, "
                "audit assumptions that could change a required test prediction, such as connectivity, grouping, "
                "direction, alignment or tie handling. If a material rival remains, make one focused comparison "
                "with real tool results; skip needless alternatives when no material rival is apparent. State "
                "whether the demonstrations actually distinguish the choices: matching the same examples does "
                "not rule out a rival. Inspect relevant public test-input structure when the choices diverge "
                "there, without treating it as a hidden-answer oracle. Distinguish observations from heuristic "
                "preferences; repeated object counts or visual regularity are not universal constraints. "
                "Never invent an observation or a labeled counterexample, and never special-case an example "
                "by its index. Keep explanations brief and verifiable: state only "
                "the key decision and evidence needed to check it, without long chain-of-thought. Before "
                "submitting, re-read the instruction and check explicit constraints such as coordinates, colors, "
                "direction, maximum repetitions/turns, unchanged cells, and every required target. Use only the "
                "public task and tool observations; do not claim access to hidden answers."
            )
            if messages[0]['role'] == 'system' and isinstance(messages[0]['content'], str):
                messages[0]['content'] += self_check
            else:
                messages.insert(0, {'role': 'system', 'content': self_check.strip()})
            tools = copy.deepcopy(source.get("tools"))
            allowed_tools = _native_tool_names(tools)
            budget = _claimed_budget(
                job, source, configured_max_job_tokens=self.max_job_tokens,
                configured_max_total_tokens=self.max_total_tokens,
            )
            max_tool_calls = _integer(source.get("max_tool_calls"))
            if max_tool_calls is None or not 1 <= max_tool_calls <= 1_000:
                raise TeacherRunnerError("source_tool_budget_invalid")
            source_completion_cap = _integer(source.get("max_completion_tokens"))
            if source_completion_cap is None or source_completion_cap < 1:
                raise TeacherRunnerError("source_completion_budget_invalid")
            completion_cap = min(self.provider_config.max_completion_tokens, source_completion_cap)
            result = TeacherResult(
                job_id=job_id,
                session_id=session_id,
                native_messages=messages,
                usage_calls=[],
                termination="running",
                provider=self.provider_config.provider,
                requested_model=self.provider_config.model,
            )
            accounting_policy = source.get('unknown_usage_policy', 'block')
            if accounting_policy not in {'block', 'confirmed_lower_bound'}:
                raise TeacherRunnerError('source_usage_policy_invalid')
            budget.allow_unknown = accounting_policy == 'confirmed_lower_bound'
            stream_retries = source.get('max_stream_retries', 0)
            if type(stream_retries) is not int or not 0 <= stream_retries <= 2:
                raise TeacherRunnerError('source_stream_retry_budget_invalid')
            tool_sequence = 0
            terminal_generation = {}
            max_model_turns = max_tool_calls + 4
            for model_turn in range(max_model_turns):
                if time.monotonic() >= deadline:
                    result.termination = "job_deadline"
                    break
                if tool_sequence >= max_tool_calls:
                    result.termination = "max_tool_calls"
                    break
                request = {
                    "model": self.provider_config.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "max_tokens": completion_cap,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                call_limit = budget.completion_limit(request, completion_cap)
                if call_limit is None:
                    result.termination = "token_budget_exhausted"
                    break
                request["max_tokens"] = call_limit
                try:
                    streamed, token_count = await self._provider_call(
                        provider_client, request, phase="solve", deadline=deadline,
                        budget=budget, result=result, central_client=central_client,
                        fence=actual_fence, worker_id=worker_id, max_retries=stream_retries,
                    )
                except TeacherRunnerError as exc:
                    result.termination = exc.code
                    break
                assistant = streamed.assistant
                if assistant is not None:
                    messages.append(assistant)
                    result.native_messages = messages
                if streamed.error_code is not None:
                    result.termination = streamed.error_code
                    break
                if assistant is None:
                    result.termination = "provider_stream_invalid"
                    break

                if token_count is None:
                    result.termination = "usage_unknown"
                    break
                if budget.spent > min(budget.max_job_tokens, budget.budget_grant_tokens):
                    result.termination = "token_budget_exceeded"
                    break

                calls = assistant.get("tool_calls")
                if calls is None or calls == []:
                    if is_model_stop('no_call', streamed.finish_reason, False):
                        completion_tokens = raw_completion_tokens(streamed.usage)
                        if completion_tokens is None:
                            result.termination = "usage_missing_completion_tokens"
                            break
                        terminal_generation = {
                            "generation_stop_reason": streamed.finish_reason,
                            "completion_tokens": completion_tokens,
                            "usage": copy.deepcopy(dict(streamed.usage)),
                            "generated_text": assistant.get("content", ""),
                        }
                        result.termination = "model_stop"
                    else:
                        result.termination = ("completion_limit" if streamed.finish_reason == "length"
                                              else "native_tool_turn_invalid")
                    break
                if not isinstance(calls, list):
                    result.termination = "native_tool_turn_invalid"
                    break
                if len(calls) > max_tool_calls - tool_sequence:
                    result.termination = "max_tool_calls"
                    break
                # Validate the complete batch before executing its first stateful ARC action.
                # Providers can emit multiple calls despite parallel_tool_calls=False. Execute
                # those calls in returned order without inventing an episode-ending tool.
                validated_calls = []
                call_ids = set()
                try:
                    for call in calls:
                        call_id, name, arguments = _validate_tool_call(
                            {"tool_calls": [call]}, allowed_tools)
                        if call_id in call_ids:
                            raise TeacherRunnerError("native_tool_turn_invalid")
                        call_ids.add(call_id)
                        validated_calls.append((call_id, name, arguments))
                except TeacherRunnerError as exc:
                    result.termination = exc.code
                    break
                completion_tokens = raw_completion_tokens(streamed.usage)
                if completion_tokens is None or completion_tokens < 1:
                    result.termination = "usage_missing_completion_tokens"
                    break
                for call_index, (call_id, name, arguments) in enumerate(validated_calls):
                    # One provider response produced the whole batch. Attribute its token
                    # receipt to the first event only so replay accounting never double-counts.
                    tool_payload = {
                        "fence": actual_fence,
                        "session_id": session_id,
                        "sequence": tool_sequence,
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                        "generated_text": assistant.get("content", ""),
                        "completion_tokens": completion_tokens if call_index == 0 else 0,
                    }
                    if call_index == 0 and isinstance(streamed.usage, Mapping):
                        tool_payload["usage"] = copy.deepcopy(dict(streamed.usage))
                    tool_sequence += 1
                    try:
                        tool_result = await self._central_post(
                            central_client, job_id, "tool", tool_payload)
                    except TeacherRunnerError:
                        result.termination = "tool_outcome_unknown"
                        break
                    observation = tool_result.get("observation")
                    if tool_result.get("infra_abort") is True:
                        if observation is not None:
                            messages.append({
                                "role": "tool", "name": name, "tool_call_id": call_id,
                                "content": _tool_observation(observation),
                            })
                            result.native_messages = messages
                        result.termination = "tool_infra_abort"
                        break
                    if observation is None:
                        result.termination = "tool_response_invalid"
                        break
                    messages.append({
                        "role": "tool", "name": name, "tool_call_id": call_id,
                        "content": _tool_observation(observation),
                    })
                    result.native_messages = messages
                    if tool_result.get("done") is True:
                        if call_index != len(validated_calls) - 1:
                            result.termination = "native_tool_turn_invalid"
                        else:
                            result.termination = "tool_budget_limit"
                        break
                    if not budget.usage_known and not budget.allow_unknown:
                        result.termination = "usage_unknown"
                        break
                    if budget.spent > min(budget.max_job_tokens, budget.budget_grant_tokens):
                        result.termination = "token_budget_exhausted"
                        break
                if result.termination != "running":
                    break
            else:
                result.termination = "model_turn_limit"

            # Always close the source session through the central owner. Its response
            # contains only a boolean gate and public receipt handles, never private
            # test outputs or scores.
            try:
                finished = await self._central_post(central_client, job_id, "source-finish", {
                    "fence": actual_fence,
                    "session_id": session_id,
                    "termination": result.termination,
                    **terminal_generation,
                })
            except TeacherRunnerError:
                result.termination = "source_finish_unknown"
                return result
            accepted = finished.get("source_accepted")
            if type(accepted) is not bool:
                result.termination = "source_verification_unknown"
                return result
            result.source_accepted = accepted
            result.source_receipt = _source_receipt(finished.get("source_receipt"))
            if not accepted:
                if result.termination == "model_stop":
                    result.termination = "source_rejected"
                return result
            if result.termination != "model_stop":
                return result
            if not budget.usage_known and not budget.allow_unknown:
                result.termination = "usage_unknown"
                return result

            plan_messages = [
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user", "content": json.dumps({"native_messages": messages},
                                                            ensure_ascii=False, separators=(",", ":"))},
            ]
            plan_request = {
                "model": self.provider_config.model,
                "messages": plan_messages,
                "max_tokens": completion_cap,
                "stream": True,
                "stream_options": {"include_usage": True},
                "response_format": {"type": "json_object"},
            }
            plan_limit = budget.completion_limit(plan_request, completion_cap)
            if plan_limit is None:
                result.termination = "plan_budget_exhausted"
                return result
            plan_request["max_tokens"] = plan_limit
            try:
                streamed, token_count = await self._provider_call(
                    provider_client, plan_request, phase="plan", deadline=deadline,
                    budget=budget, result=result, central_client=central_client,
                    fence=actual_fence, worker_id=worker_id, max_retries=stream_retries,
                )
            except TeacherRunnerError as exc:
                result.termination = exc.code
                return result
            assistant = streamed.assistant
            if assistant is not None:
                result.plan_response = assistant
            if streamed.error_code is not None:
                result.termination = "plan_outcome_unknown"
                return result
            if assistant is None:
                result.termination = "plan_response_invalid"
                return result
            if token_count is None:
                result.termination = "plan_usage_unknown"
                return result
            if budget.spent > min(budget.max_job_tokens, budget.budget_grant_tokens):
                result.termination = "token_budget_exceeded"
                return result
            content = assistant.get("content", "")
            try:
                structured = _strict_json_loads(content)
            except (TypeError, ValueError):
                result.termination = "plan_invalid"
                return result
            plan = structured.get("plan") if isinstance(structured, dict) and set(structured) == {"plan"} else None
            try:
                plan = validate_teacher_plan(plan)
            except ContractError:
                result.termination = "plan_invalid"
                return result
            if assistant.get("tool_calls") or streamed.finish_reason == "length":
                result.termination = "plan_invalid"
                return result
            result.plan = plan
            result.termination = "plan_extracted"
            return result
        finally:
            await provider_client.close()
            if owns_central_client:
                await central_client.aclose()

    async def _central_post(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        endpoint: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        url = f"{self.central_base_url}/bepa/jobs/{quote(job_id, safe='')}/{endpoint}"
        try:
            response = await client.post(url, json=dict(payload))
        except httpx.HTTPError:
            raise TeacherRunnerError("central_transport_error") from None
        if response.status_code < 200 or response.status_code >= 300:
            raise TeacherRunnerError("central_request_rejected", status_code=response.status_code)
        try:
            value = response.json()
        except ValueError:
            raise TeacherRunnerError("central_response_invalid", status_code=response.status_code) from None
        if not isinstance(value, dict):
            raise TeacherRunnerError("central_response_invalid", status_code=response.status_code)
        return value

    @staticmethod
    def _deadline(job: Mapping[str, Any]) -> float:
        value = job.get("deadline")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise TeacherRunnerError("job_deadline_missing")
        return time.monotonic() + max(0.0, float(value) - time.time())

    async def run_forever(
        self,
        *,
        worker_id: str,
        once: bool = False,
        poll_interval_seconds: float = 2.0,
    ) -> dict[str, Any]:
        """Claim jobs, heartbeat while they run, and submit one terminal mutation."""
        if not isinstance(worker_id, str) or not _JOB_ID.fullmatch(worker_id):
            raise TeacherRunnerError("worker_id_invalid")
        if (isinstance(poll_interval_seconds, bool)
                or not isinstance(poll_interval_seconds, (int, float))
                or not math.isfinite(float(poll_interval_seconds)) or poll_interval_seconds <= 0):
            raise ValueError("poll_interval_seconds must be finite and positive")
        token = os.environ.get(self.central_token_env)
        if not token:
            raise TeacherRunnerError("central_credential_missing")
        provider_key = os.environ.get(self.provider_config.api_key_env)
        if not provider_key:
            error = TeacherRunnerError(
                "provider_credential_missing", category="provider_readiness",
            )
            if once:
                return {
                    "status": "provider_not_ready", "worker_id": worker_id,
                    "diagnostic": error.safe_diagnostic(),
                    "claimed": 0, "completed": 0, "failed": 0,
                }
            raise error
        headers = {"Authorization": f"Bearer {token}"}
        claimed = completed = failed = 0
        provider_ready_until = 0.0
        probe_failures = 0
        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(self.central_timeout_seconds),
            transport=self.central_transport,
        ) as client:
            while True:
                if time.monotonic() >= provider_ready_until:
                    try:
                        await self._probe_provider(provider_key)
                    except TeacherRunnerError as exc:
                        diagnostic = exc.safe_diagnostic()
                        if once:
                            return {
                                "status": "provider_not_ready", "worker_id": worker_id,
                                "diagnostic": diagnostic,
                                "claimed": claimed, "completed": completed, "failed": failed,
                            }
                        _LOGGER.warning(
                            "provider readiness probe failed before claim: %s",
                            json.dumps(diagnostic, sort_keys=True, separators=(",", ":")),
                        )
                        await asyncio.sleep(_provider_probe_delay(probe_failures))
                        probe_failures += 1
                        continue
                    provider_ready_until = time.monotonic() + PROVIDER_READY_TTL_SECONDS
                    probe_failures = 0
                claim_id = uuid.uuid4().hex
                claim_retry_attempt = 0
                while True:
                    try:
                        response = await client.post(
                            f"{self.central_base_url}/bepa/jobs/claim",
                            json={"worker_id": worker_id, "claim_id": claim_id},
                        )
                    except httpx.TransportError:
                        if once and claim_retry_attempt >= CLAIM_ONCE_MAX_RETRIES:
                            raise TeacherRunnerError("claim_transport_error") from None
                        await asyncio.sleep(_claim_retry_delay(claim_retry_attempt))
                        claim_retry_attempt = min(claim_retry_attempt + 1, 8)
                        continue
                    if response.status_code in {408, 429} or response.status_code >= 500:
                        if once and claim_retry_attempt >= CLAIM_ONCE_MAX_RETRIES:
                            raise TeacherRunnerError(
                                "claim_request_rejected", status_code=response.status_code,
                            )
                        await asyncio.sleep(_claim_retry_delay(claim_retry_attempt))
                        claim_retry_attempt = min(claim_retry_attempt + 1, 8)
                        continue
                    break
                if response.status_code != 200:
                    raise TeacherRunnerError("claim_request_rejected", status_code=response.status_code)
                try:
                    payload = _strict_json_loads(response.content)
                except (ValueError, TypeError):
                    raise TeacherRunnerError("claim_response_invalid") from None
                if not isinstance(payload, Mapping) or "job" not in payload:
                    raise TeacherRunnerError("claim_response_invalid")
                job = payload["job"]

                if job is None:
                    if once:
                        return {"status": "idle", "worker_id": worker_id,
                                "claimed": claimed, "completed": completed, "failed": failed}
                    await asyncio.sleep(float(poll_interval_seconds))
                    continue
                if not isinstance(job, Mapping):
                    raise TeacherRunnerError("claim_response_invalid")
                claimed += 1
                claimed_worker = job.get("worker_id")
                if not isinstance(claimed_worker, str) or claimed_worker != worker_id:
                    raise TeacherRunnerError("claim_worker_mismatch")
                job_id = job.get("job_id")
                fence = job.get("fence")
                deadline_value = job.get("deadline")
                if (not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id)
                        or type(fence) is not int or fence < 1
                        or isinstance(deadline_value, bool)
                        or not isinstance(deadline_value, (int, float))
                        or not math.isfinite(float(deadline_value))):
                    raise TeacherRunnerError("claim_response_invalid")

                deadline_remaining = float(deadline_value) - time.time()
                if deadline_remaining <= 0:
                    raise TeacherRunnerError("claim_deadline_expired")
                heartbeat_timeout = min(
                    deadline_remaining,
                    self.job_timeout_seconds if self.job_timeout_seconds is not None else deadline_remaining,
                )
                heartbeat_interval = min(30.0, max(0.1, heartbeat_timeout / 3.0))
                stop_heartbeat = asyncio.Event()
                job_task = asyncio.create_task(self.run_claimed_job(job, central_client=client))
                heartbeat_task = asyncio.create_task(self._heartbeat_loop(
                    client, job_id=job_id, fence=fence, worker_id=worker_id,
                    interval_seconds=heartbeat_interval, stop=stop_heartbeat,
                ))
                done, _pending = await asyncio.wait(
                    {job_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED,
                )
                if heartbeat_task in done:
                    error = heartbeat_task.exception()
                    if error is not None:
                        if not job_task.done():
                            job_task.cancel()
                            try:
                                await job_task
                            except asyncio.CancelledError:
                                pass
                        stop_heartbeat.set()
                        if not once and isinstance(error, TeacherRunnerError) and (
                            error.code in {"heartbeat_unknown", "lease_lost"}
                            or (error.code == "heartbeat_rejected"
                                and error.status_code is not None and error.status_code >= 500)
                        ):
                            # Cancel the source, never replay its provider request or
                            # mutate an uncertain lease. Central claim fencing keeps
                            # the service idle until that lease is settled/expired.
                            _LOGGER.warning("Teacher job %s cancelled after heartbeat: %s",
                                            job_id, error.code)
                            failed += 1
                            provider_ready_until = 0.0
                            await asyncio.sleep(poll_interval_seconds)
                            continue
                        return {"status": "lease_lost", "worker_id": worker_id,
                                "job_id": job_id,
                                "reason": error.code if isinstance(error, TeacherRunnerError) else "heartbeat_unknown",
                                "claimed": claimed, "completed": completed, "failed": failed}
                stop_heartbeat.set()
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                try:
                    result = await job_task
                except TeacherRunnerError as exc:
                    failure = {"fence": fence, "worker_id": worker_id,
                               "reason": exc.code, "termination": exc.code}
                    try:
                        await self._central_mutation(client, job_id, "fail", failure)
                    except TeacherRunnerError as mutation_exc:
                        return {
                            "status": self._mutation_stop_status(mutation_exc),
                            "worker_id": worker_id, "job_id": job_id,
                            "diagnostic": mutation_exc.safe_diagnostic(),
                            "claimed": claimed, "completed": completed, "failed": failed,
                        }
                    failed += 1
                except Exception:
                    failure = {"fence": fence, "worker_id": worker_id,
                               "reason": "teacher_internal_error", "termination": "teacher_internal_error"}
                    try:
                        await self._central_mutation(client, job_id, "fail", failure)
                    except TeacherRunnerError as mutation_exc:
                        return {
                            "status": self._mutation_stop_status(mutation_exc),
                            "worker_id": worker_id, "job_id": job_id,
                            "diagnostic": mutation_exc.safe_diagnostic(),
                            "claimed": claimed, "completed": completed, "failed": failed,
                        }
                    failed += 1
                else:
                    if result.termination == "plan_extracted" and result.source_accepted:
                        # Persist the exact terminal request before any central mutation. This
                        # outbox is the recovery source if the response is lost or rejected.
                        completion_evidence = result.completion_payload()
                        body = copy.deepcopy(completion_evidence)
                        body.update(fence=fence, worker_id=worker_id)
                        outbox_ref = None
                        mutation_error = None
                        try:
                            outbox_ref = self._persist_completion_evidence(
                                job_id=job_id, fence=fence, worker_id=worker_id, request_body=body,
                            )
                        except TeacherRunnerError as exc:
                            mutation_error = exc
                        if mutation_error is None:
                            try:
                                await self._central_mutation(client, job_id, "complete", body)
                            except TeacherRunnerError as exc:
                                mutation_error = exc
                        if mutation_error is not None:
                            diagnostic = mutation_error.safe_diagnostic()
                            deterministic_rejection = (
                                mutation_error.code == "job_mutation_rejected"
                                and mutation_error.endpoint == "complete"
                                and mutation_error.status_code == 422
                            )
                            evidence_persist_failed = (
                                mutation_error.category == "local_persistence"
                            )
                            if deterministic_rejection or evidence_persist_failed:
                                reason = (
                                    "complete_rejected" if deterministic_rejection
                                    else "completion_evidence_persist_failed"
                                )
                                failure = {
                                    "fence": fence,
                                    "worker_id": worker_id,
                                    "reason": reason,
                                    "termination": reason,
                                    "token_total": result.token_total,
                                    "usage_calls": copy.deepcopy(result.usage_calls),
                                    "usage_known": result.usage_known,
                                    "outcome_unknown": not result.usage_known,
                                    "error": (
                                        "complete_rejected_http_422" if deterministic_rejection
                                        else "completion_evidence_persist_failed"
                                    ),
                                    "evidence": {
                                        "completion": completion_evidence,
                                        "outbox": outbox_ref,
                                        "mutation": diagnostic,
                                    },
                                }
                                try:
                                    await self._central_mutation(client, job_id, "fail", failure)
                                except TeacherRunnerError as fail_exc:
                                    fail_diagnostic = fail_exc.safe_diagnostic()
                                    if fail_exc.status_code == 409:
                                        return {
                                            "status": "lease_lost", "worker_id": worker_id,
                                            "job_id": job_id, "diagnostic": fail_diagnostic,
                                            "evidence_path": outbox_ref["path"] if outbox_ref else None,
                                            "claimed": claimed, "completed": completed, "failed": failed,
                                        }
                                    status = (
                                        "mutation_unknown"
                                        if fail_exc.code in {"job_mutation_unknown", "job_mutation_response_invalid"}
                                        or fail_exc.status_code is None or fail_exc.status_code >= 500
                                        else "mutation_rejected"
                                    )
                                    return {
                                        "status": status, "worker_id": worker_id,
                                        "job_id": job_id, "diagnostic": fail_diagnostic,
                                        "evidence_path": outbox_ref["path"] if outbox_ref else None,
                                        "claimed": claimed, "completed": completed, "failed": failed,
                                    }
                                failed += 1
                            elif mutation_error.status_code == 409:
                                return {
                                    "status": "lease_lost", "worker_id": worker_id,
                                    "job_id": job_id, "diagnostic": diagnostic,
                                    "evidence_path": outbox_ref["path"] if outbox_ref else None,
                                    "claimed": claimed, "completed": completed, "failed": failed,
                                }
                            else:
                                status = (
                                    "mutation_unknown"
                                    if mutation_error.code in {"job_mutation_unknown", "job_mutation_response_invalid"}
                                    or mutation_error.status_code is None
                                    or mutation_error.status_code >= 500
                                    else "mutation_rejected"
                                )
                                return {
                                    "status": status, "worker_id": worker_id,
                                    "job_id": job_id, "diagnostic": diagnostic,
                                    "evidence_path": outbox_ref["path"] if outbox_ref else None,
                                    "claimed": claimed, "completed": completed, "failed": failed,
                                }
                        else:
                            completed += 1
                    else:
                        body = {
                            "fence": fence,
                            "worker_id": worker_id,
                            "reason": result.termination,
                            "termination": result.termination,
                            "token_total": result.token_total,
                            "usage_calls": copy.deepcopy(result.usage_calls),
                            "usage_known": result.usage_known,
                            "outcome_unknown": (not result.usage_known or "unknown" in result.termination),
                            "evidence": result.completion_payload(),
                        }
                        try:
                            await self._central_mutation(client, job_id, "fail", body)
                        except TeacherRunnerError as mutation_error:
                            return {
                                "status": self._mutation_stop_status(mutation_error),
                                "worker_id": worker_id, "job_id": job_id,
                                "diagnostic": mutation_error.safe_diagnostic(),
                                "claimed": claimed, "completed": completed, "failed": failed,
                            }
                        failed += 1
                # A completed job may have taken long enough for the provider route to change.
                # Require a fresh model-list probe before claiming the next job; idle polls use
                # the short readiness TTL instead of probing the catalog every poll interval.
                provider_ready_until = 0.0
                if once:
                    return {"status": "processed", "worker_id": worker_id,
                            "claimed": claimed, "completed": completed, "failed": failed}

    async def _heartbeat_loop(
        self,
        client: httpx.AsyncClient,
        *,
        job_id: str,
        fence: int,
        worker_id: str,
        interval_seconds: float,
        stop: asyncio.Event,
    ) -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
                return
            except TimeoutError:
                pass
            url = f"{self.central_base_url}/bepa/jobs/{quote(job_id, safe='')}/heartbeat"
            # Renewal with the same owner/fence is safe to repeat. This retries
            # only the control-plane heartbeat, never a billable provider call.
            for attempt in range(2):
                try:
                    response = await client.post(url, json={"fence": fence, "worker_id": worker_id})
                    break
                except httpx.HTTPError:
                    if attempt:
                        raise TeacherRunnerError("heartbeat_unknown") from None
            if response.status_code == 409:
                raise TeacherRunnerError("lease_lost", status_code=409)
            if response.status_code < 200 or response.status_code >= 300:
                raise TeacherRunnerError("heartbeat_rejected", status_code=response.status_code)
            try:
                value = _strict_json_loads(response.content) if response.content else {}
            except (ValueError, TypeError):
                raise TeacherRunnerError("heartbeat_response_invalid") from None
            if isinstance(value, Mapping) and value.get("lease_valid") is False:
                raise TeacherRunnerError("lease_lost")

    @staticmethod
    def _mutation_stop_status(error: TeacherRunnerError) -> str:
        if error.status_code == 409:
            return "lease_lost"
        if (error.code in {"job_mutation_unknown", "job_mutation_response_invalid"}
                or error.status_code is None or error.status_code >= 500):
            return "mutation_unknown"
        return "mutation_rejected"

    def _persist_completion_evidence(
        self, *, job_id: str, fence: int, worker_id: str, request_body: Mapping[str, Any],
    ) -> dict[str, str]:
        """Atomically write the exact complete request before any terminal API mutation."""
        if self.evidence_dir is None:
            raise TeacherRunnerError(
                "completion_evidence_dir_missing", category="local_persistence",
            )
        envelope = {
            "schema": "bepa-teacher-completion-outbox-v1",
            "job_id": job_id,
            "fence": fence,
            "worker_id": worker_id,
            "request": dict(request_body),
        }
        try:
            data = json.dumps(
                envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
            self.evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            destination = self.evidence_dir / f"{job_id}.fence-{fence}.json"
            try:
                existing_fd = os.open(
                    destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
            except FileNotFoundError:
                existing_fd = None
            if existing_fd is not None:
                with os.fdopen(existing_fd, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode) or stream.read() != data:
                        raise TeacherRunnerError(
                            "completion_evidence_conflict", category="local_persistence",
                        )
            else:
                fd, temp_name = tempfile.mkstemp(
                    prefix=".teacher-outbox-", suffix=".tmp", dir=self.evidence_dir,
                )
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        os.link(temp_name, destination, follow_symlinks=False)
                    except FileExistsError:
                        existing_fd = os.open(
                            destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                        )
                        with os.fdopen(existing_fd, "rb") as stream:
                            if (not stat.S_ISREG(os.fstat(stream.fileno()).st_mode)
                                    or stream.read() != data):
                                raise TeacherRunnerError(
                                    "completion_evidence_conflict", category="local_persistence",
                                )
                    dir_fd = os.open(
                        self.evidence_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                finally:
                    try:
                        os.unlink(temp_name)
                    except FileNotFoundError:
                        pass
        except TeacherRunnerError:
            raise
        except (OSError, TypeError, ValueError):
            raise TeacherRunnerError(
                "completion_evidence_persist_failed", category="local_persistence",
            ) from None
        return {
            "path": str(destination),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    async def _central_mutation(
        self, client: httpx.AsyncClient, job_id: str, endpoint: str, payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        url = f"{self.central_base_url}/bepa/jobs/{quote(job_id, safe='')}/{endpoint}"
        try:
            response = await client.post(url, json=dict(payload))
        except httpx.HTTPError:
            raise TeacherRunnerError(
                "job_mutation_unknown", endpoint=endpoint,
                category="transport_outcome_unknown",
            ) from None
        if response.status_code < 200 or response.status_code >= 300:
            category = (
                "lease_or_state_conflict" if response.status_code == 409 else "http_rejected"
            )
            raise TeacherRunnerError(
                "job_mutation_rejected", status_code=response.status_code,
                endpoint=endpoint, category=category,
            )
        if not response.content:
            return {}
        try:
            value = _strict_json_loads(response.content)
        except (ValueError, TypeError):
            raise TeacherRunnerError(
                "job_mutation_response_invalid", endpoint=endpoint,
                category="response_invalid",
            ) from None
        if not isinstance(value, dict):
            raise TeacherRunnerError(
                "job_mutation_response_invalid", endpoint=endpoint,
                category="response_invalid",
            )
        return value

    def _usage_receipt(
        self,
        index: int,
        phase: str,
        usage: Any,
        total_tokens: int | None,
        actual_model: str | None,
        *,
        request_id: str,
        response_id: str | None,
        finish_reason: str | None,
        stream_complete: bool,
        provider_error_category: str | None,
        provider_status_code: int | None,
    ) -> dict[str, Any]:
        receipt: dict[str, Any] = {
            "index": index,
            "phase": phase,
            "provider": self.provider_config.provider,
            "requested_model": self.provider_config.model,
            "model": actual_model,
            "request_id": request_id,
            "response_id": response_id,
            "finish_reason": finish_reason,
            "stream_complete": stream_complete,
            "usage": copy.deepcopy(dict(usage)) if isinstance(usage, Mapping) else None,
            "total_tokens": total_tokens,
            "budget_token_basis": "flash_uncached_input",
            "outcome_unknown": total_tokens is None or not stream_complete,
            "usage_known": total_tokens is not None,
            "safe_to_retry": False,
        }
        if provider_error_category is not None:
            receipt["provider_error_category"] = provider_error_category
        if provider_status_code is not None:
            receipt["provider_status_code"] = provider_status_code
        return receipt

    @staticmethod
    def _sync_budget_result(result: TeacherResult, budget: _TokenBudget) -> None:
        result.usage_known = budget.usage_known
        result.token_total = budget.spent if budget.usage_known else None
