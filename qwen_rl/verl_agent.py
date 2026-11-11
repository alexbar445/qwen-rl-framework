"""verl agent-loop adapter for the authenticated ARC environment service."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolSchema,
)

from .prompt import action_tools, validate_service_action_language
from .shared_prefix import stable_system_prefix
from .contracts.arc_tools import NATIVE_TOOL_NAMES, is_model_stop, tool_schema_error
from .contracts.episode_budget import BUDGET_POLICY_ID, context_generation_limit, plan_generation
from .contracts.tool_call_failures import MAX_TOOL_CALL_FAILURES
from .contracts.worker import MAX_CONTEXT_TOKENS

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_SECONDS = 30.0
_RESULT_POLL_SECONDS = 0.5
_RESULT_TRANSPORT_MAX_ATTEMPTS = 3


class ArcServiceError(RuntimeError):
    """An ARC service response violated its HTTP or JSON contract."""


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    getter = getattr(obj, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(obj, key, default)


def _required_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _chat_text(message: Any) -> str:
    if not isinstance(message, Mapping) or message.get("role") != "user":
        raise ValueError("raw_prompt must end in a user message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("ARC visible question must be plain text")
    return content


def _initial_prompt_fingerprint(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    try:
        canonical = json.dumps(
            {"messages": messages, "tools": tools},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("initial ARC prompt and tools must be JSON-compatible") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedInitialPrompt:
    """Initial Continuous Token IDs bound to the messages and tools that rendered them."""

    token_ids: tuple[int, ...]
    fingerprint: str
    shared_prefix_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, tuple) or not self.token_ids:
            raise ValueError("prepared initial prompt token_ids must be a non-empty tuple")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.token_ids):
            raise ValueError("prepared initial prompt token IDs must be nonnegative integers")
        if type(self.shared_prefix_tokens) is not int or not 0 <= self.shared_prefix_tokens < len(self.token_ids):
            raise ValueError("prepared shared prefix must end before the final prompt token")
        if (
            not isinstance(self.fingerprint, str)
            or len(self.fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.fingerprint)
        ):
            raise ValueError("prepared initial prompt fingerprint must be a lowercase SHA-256")


def _decode_call(call: FunctionCall) -> tuple[str, dict[str, Any]] | None:
    try:
        parsed, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
            OpenAIFunctionParsedSchema(name=call.name, arguments=call.arguments)
        )
    except Exception:
        return None
    if has_decode_error:
        return None
    return parsed.name, parsed.arguments


def _policy_turn(parsed_calls: list[FunctionCall], raw_text: str) -> tuple[str | None, dict[str, Any], str, str | None]:
    """Diagnose the sampled call without rewriting it into an executable action."""
    if len(parsed_calls) > 1:
        return None, {}, "multiple_calls", None
    if parsed_calls:
        decoded = _decode_call(parsed_calls[0])
        if decoded is None:
            return None, {}, "malformed", getattr(parsed_calls[0], "name", None)
        name, arguments = decoded
        if name not in NATIVE_TOOL_NAMES:
            return None, {}, "unregistered_tool", name
        if tool_schema_error(name, arguments) is not None:
            return None, {}, "invalid_schema", name
        return name, arguments, "valid", name
    # The upstream parser may discard an unknown/malformed function entirely.
    # Inspect its original spelling for diagnostics only; never execute this text.
    tail = raw_text.rsplit("</think>", 1)[-1]
    candidate = re.search(r"<function=([^>\n]{1,200})>", tail)
    requested_name = candidate.group(1) if candidate else None
    if requested_name and requested_name not in NATIVE_TOOL_NAMES:
        return None, {}, "unregistered_tool", requested_name
    if re.search(r"</?(?:tool_call|function|parameter)(?:[=>\s<]|$)", tail):
        return None, {}, "malformed", requested_name
    return None, {}, "no_call", None


def _budget_receipt(payload: dict[str, Any]) -> tuple[int, int, int]:
    used = _required_int(payload.get("tokens_used"), "ARC tokens_used")
    remaining = _required_int(payload.get("tokens_remaining"), "ARC tokens_remaining")
    budget = _required_int(payload.get("token_budget"), "ARC token_budget", minimum=1)
    if remaining != max(0, budget - used):
        raise ArcServiceError("ARC budget receipt is inconsistent")
    return used, remaining, budget


def _separate_budget_fields(payload: dict[str, Any]) -> dict[str, int]:
    generated_cap = payload.get("generated_token_budget")
    observation_cap = payload.get("observation_token_budget")
    if generated_cap is None and observation_cap is None:
        return {}
    generated_cap = _required_int(generated_cap, "ARC generated_token_budget", minimum=1)
    observation_cap = _required_int(observation_cap, "ARC observation_token_budget", minimum=1)
    generated = _required_int(payload.get("generated_tokens"), "ARC generated_tokens")
    observations = _required_int(payload.get("observation_tokens"), "ARC observation_tokens")
    if (generated_cap + observation_cap != payload["token_budget"]
            or generated + observations != payload["tokens_used"]):
        raise ArcServiceError("ARC separate budget receipt is inconsistent")
    return {"remaining_generated_budget": max(0, generated_cap - generated),
            "remaining_observation_budget": max(0, observation_cap - observations)}


def _close_feedback_turn(tokenizer, token_ids, response_mask, response_logprobs):
    """Close an interrupted ChatML turn as zero-loss context, preserving sampled IDs."""
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if type(end_id) is not int or end_id < 0 or end_id == getattr(tokenizer, "unk_token_id", None):
        raise ArcServiceError("Qwen feedback requires the tokenizer's im_end token")
    newline = tokenizer.encode("\n", add_special_tokens=False)
    suffix = [end_id, *newline]
    closed = bool(token_ids) and (
        token_ids[-1] == end_id or (len(token_ids) >= len(suffix) and token_ids[-len(suffix):] == suffix)
    )
    if closed:
        return token_ids, response_mask, response_logprobs
    return [*token_ids, end_id], [*response_mask, 0], [*response_logprobs, 0.0]


def _assistant_message(content: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    # `content` is the parser's exact pre-tool text. In Qwen3 thinking mode this retains
    # the original <think> block; the normalized parsed tool call is rendered once by CT.
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def _prepare_assistant_turn(
    tokenizer: Any,
    generated_ids: list[int],
    assistant_content: str,
    parsed_calls: list[FunctionCall],
) -> tuple[str | None, dict[str, Any], str, str | None, dict[str, Any]]:
    # Parsed calls are classified without the raw decode. Keep decoding for empty
    # or invalid parser results, where raw text is still needed for diagnostics
    # or the assistant fallback message.
    raw_text = "" if parsed_calls else tokenizer.decode(
        generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    name, arguments, parse_status, requested_name = _policy_turn(parsed_calls, raw_text)
    valid_call = parse_status == "valid"
    if parsed_calls and not valid_call:
        raw_text = tokenizer.decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
    assistant_message = (
        _assistant_message(assistant_content, name, arguments) if valid_call
        else {"role": "assistant", "content": raw_text}
    )
    return name, arguments, parse_status, requested_name, assistant_message


@register("arc")
class ArcAgentLoop(AgentLoopBase):
    """One ARC episode per sample, scored by a group-aware trusted service."""

    def __init__(self, *args, http_transport: httpx.AsyncBaseTransport | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.arc_config = self.config.arc
        self.constrained_decoding = _value(self.arc_config, "constrained_decoding", False)
        if type(self.constrained_decoding) is not bool:
            raise ValueError("arc.constrained_decoding must be boolean")
        self.grammar_constraint = None
        self.constraint_tokenizer_sha256 = None
        if self.constrained_decoding:
            from .prompt import load_toolkit_runtime
            load_toolkit_runtime()
            from arc_toolkit.constrained_actions import constraint_receipt
            receipt = constraint_receipt(thinking=True)
            from .contracts.grammar_compiler import native_grammar_compiler_sha256
            self.grammar_constraint = {**receipt, "version":"structural-tag-policy/v1",
                "compiler_sha256":native_grammar_compiler_sha256()}
            backend = getattr(self.tokenizer, "backend_tokenizer", None)
            if backend is None:
                raise ValueError("constrained decoding requires a serializable fast tokenizer")
            self.constraint_tokenizer_sha256 = hashlib.sha256(backend.to_str().encode()).hexdigest()
        self.env_url = str(self.arc_config.env_url).rstrip("/")
        self.group_timeout = float(self.arc_config.group_timeout)
        if not math.isfinite(self.group_timeout) or self.group_timeout <= 0:
            raise ValueError("arc.group_timeout must be a positive finite number")
        self.turn_max_tokens = _required_int(self.arc_config.turn_max_tokens, "arc.turn_max_tokens", minimum=1)
        self.max_turns = _required_int(self.arc_config.max_turns, "arc.max_turns", minimum=1)
        self.response_length = _required_int(self.rollout_config.response_length, "rollout.response_length", minimum=1)
        self.context_length = _required_int(
            _value(self.arc_config, "context_length", MAX_CONTEXT_TOKENS),
            "arc.context_length", minimum=1,
        )
        if self.context_length > MAX_CONTEXT_TOKENS:
            raise ValueError("ARC context exceeds the Worker wire capacity")
        self.http_transport = http_transport

        if self.rollout_config.multi_turn.format != "qwen3_coder":
            raise ValueError("ARC requires rollout.multi_turn.format=qwen3_coder")

        # Keep the pinned JSON schemas intact for chat-template rendering. verl's parser
        # also receives its upstream typed representation for parameter decoding.
        self.tool_schema_dicts = action_tools()
        self.tool_schemas = [OpenAIFunctionToolSchema.model_validate(schema) for schema in self.tool_schema_dicts]
        self.tool_parser = ToolParser.get_tool_parser("qwen3_coder", self.tokenizer)
        self.tool_names = {schema.function.name for schema in self.tool_schemas}
        if self.tool_names != NATIVE_TOOL_NAMES:
            raise ValueError("pinned ARC tools must contain only arc_action")

    def _auth_headers(self) -> dict[str, str]:
        token = os.environ.get("ARC_ENV_TOKEN")
        if not token:
            raise RuntimeError("ARC_ENV_TOKEN is required for ARC environment service access")
        return {"Authorization": f"Bearer {token}"}

    async def build_initial_prompt_ids(self, messages: list[dict[str, Any]]) -> PreparedInitialPrompt:
        """Render one text-only ARC prompt through this loop's pinned CT builder."""
        if self.processor is not None:
            raise ValueError("prepared ARC initial prompts currently support text-only loops")
        if (
            not isinstance(messages, list)
            or not messages
            or any(not isinstance(message, dict) for message in messages)
            or any(not isinstance(message.get("content"), str) for message in messages)
        ):
            raise ValueError("prepared ARC initial prompts require a non-empty text-only message list")
        prompt_messages = [dict(message) for message in messages]
        fingerprint = _initial_prompt_fingerprint(prompt_messages, self.tool_schema_dicts)
        prefix = await stable_system_prefix(self.ct_build_initial_tokens, prompt_messages, self.tool_schema_dicts)
        token_ids = await self.ct_build_initial_tokens(
            prompt_messages,
            tools=self.tool_schema_dicts,
        )
        if not token_ids or any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("Continuous Token builder returned invalid initial prompt token IDs")
        n = min(len(prefix), len(token_ids) - 1)
        if tuple(token_ids[:n]) != prefix[:n]:
            raise ValueError("actual ARC prompt differs from its stable system/tool prefix")
        return PreparedInitialPrompt(tuple(token_ids), fingerprint, n)

    async def _delete_episode(self, client: httpx.AsyncClient, episode_id: str) -> None:
        try:
            response = await client.delete(f"/episodes/{episode_id}")
            if response.status_code not in (204, 404, 409):
                logger.warning("ARC episode cleanup returned HTTP %d", response.status_code)
        except Exception as exc:  # cleanup is best-effort; never replace the rollout failure
            logger.warning("ARC episode cleanup failed (%s)", type(exc).__name__)

    @staticmethod
    def _policy_version(extra_fields: Any) -> int:
        minimum = _value(extra_fields, "min_global_steps")
        maximum = _value(extra_fields, "max_global_steps")
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
        ):
            raise ValueError("TokenOutput.extra_fields must include integer min_global_steps and max_global_steps")
        if minimum != maximum:
            raise ValueError("a generated turn spans multiple policy versions")
        return minimum

    @staticmethod
    def _expect_status(response: httpx.Response, expected: tuple[int, ...], operation: str) -> None:
        if response.status_code not in expected:
            # Do not include request headers or bodies: ARC_ENV_TOKEN must never reach logs.
            raise ArcServiceError(f"ARC service {operation} returned HTTP {response.status_code}")

    async def _poll_result(
        self, client: httpx.AsyncClient, episode_id: str, deadline: float
    ) -> dict[str, Any]:
        transport_failures = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("ARC episode group did not produce a result before group_timeout")
            try:
                response = await client.get(
                    f"/episodes/{episode_id}/result",
                    timeout=httpx.Timeout(min(_HTTP_TIMEOUT_SECONDS, remaining)),
                )
            except (httpx.ReadError, httpx.RemoteProtocolError) as exc:
                # A result GET does not replay a tool mutation. Preserve the
                # original group deadline and fail after three consecutive
                # transport failures; never retry status or schema violations.
                transport_failures += 1
                if transport_failures >= _RESULT_TRANSPORT_MAX_ATTEMPTS:
                    raise
                logger.warning(
                    "arc_result_transport_retry episode=%s attempt=%d error_type=%s",
                    episode_id, transport_failures + 1, type(exc).__name__,
                )
                await asyncio.sleep(min(_RESULT_POLL_SECONDS, max(0, deadline - time.monotonic())))
                continue
            transport_failures = 0
            self._expect_status(response, (200, 202), "result")
            if response.status_code == 200:
                try:
                    result = response.json()
                except ValueError as exc:
                    raise ArcServiceError("ARC service result was not valid JSON") from exc
                if not isinstance(result, dict):
                    raise ArcServiceError("ARC service result must be a JSON object")
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("ARC episode group did not produce a result before group_timeout")
            await asyncio.sleep(min(_RESULT_POLL_SECONDS, remaining))

    async def _finish_until_accepted(
        self,
        client: httpx.AsyncClient,
        episode_id: str,
        payload: dict[str, Any],
        deadline: float,
    ) -> None:
        # finish is idempotent by the service contract, so a 202 may be polled with the
        # identical payload. This retry path is deliberately not shared with /turn.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("ARC episode finish was not accepted before group_timeout")
            response = await client.post(
                f"/episodes/{episode_id}/finish",
                json=payload,
                timeout=httpx.Timeout(min(_HTTP_TIMEOUT_SECONDS, remaining)),
            )
            self._expect_status(response, (200, 202), "finish")
            if response.status_code == 200:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("ARC episode finish was not accepted before group_timeout")
            await asyncio.sleep(min(_RESULT_POLL_SECONDS, remaining))

    async def _discard_for_context_limit(
        self, client: httpx.AsyncClient, episode_id: str, policy_version: int, deadline: float
    ) -> None:
        try:
            await self._finish_until_accepted(
                client,
                episode_id,
                {
                    "reason": "context_limit",
                    "min_policy_version": policy_version,
                    "max_policy_version": policy_version,
                },
                deadline,
            )
        except ArcServiceError as exc:
            if "HTTP 409" in str(exc):
                raise ArcServiceError("ARC response exceeded rollout.response_length; group discarded") from exc
            raise
        raise ArcServiceError("ARC service accepted context_limit finish unexpectedly")

    @staticmethod
    def _result_values(result: dict[str, Any]) -> tuple[float, bool, float]:
        reward = result.get("reward")
        exact = result.get("exact")
        cost = result.get("cost")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ArcServiceError("ARC service result.reward must be finite numeric")
        if not isinstance(exact, bool):
            raise ArcServiceError("ARC service result.exact must be boolean")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise ArcServiceError("ARC service result.cost must be finite non-negative numeric")
        return float(reward), exact, float(cost)

    def _cap_text_prompt_length(self, prompt_ids: list[int]) -> list[int]:
        """Keep the complete initial prompt, including its system prefix.

        ``prompt_length`` is verl's static prompt/response allocation, not an ARC
        input limit. Worker validation enforces the actual model context capacity.
        Override verl's head truncation without imposing a separate prompt cap.
        """
        return prompt_ids

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_prompt = kwargs.get("raw_prompt")
        if not isinstance(raw_prompt, list) or not raw_prompt:
            raise ValueError("raw_prompt must be a non-empty message list")
        messages = [dict(message) for message in raw_prompt]
        raw_visible_question = _chat_text(messages[-1])

        extra_info = kwargs.get("extra_info") or {}
        task_id = _value(extra_info, "task_id")
        validation = _value(extra_info, "validation")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("extra_info.task_id is required")
        if not isinstance(validation, bool):
            raise ValueError("extra_info.validation must be boolean")
        if "validate" in kwargs and bool(kwargs["validate"]) != validation:
            raise ValueError("extra_info.validation disagrees with rollout validate flag")
        uid = kwargs.get("uid")
        session_id = kwargs.get("session_id")
        if uid is None or session_id is None:
            raise ValueError("verl uid and session_id are required for ARC group accounting")
        group_id, sample_id = str(uid), str(session_id)
        if not group_id or not sample_id:
            raise ValueError("verl uid and session_id must be non-empty")

        rollout = self.config.actor_rollout_ref.rollout
        group_size = _required_int(
            rollout.val_kwargs.n if validation else rollout.n,
            "rollout group size",
            minimum=1,
        )
        if validation and group_size != 1:
            raise ValueError("ARC validation episodes require rollout.val_kwargs.n=1")

        # Build or validate the initial prompt through verl's pinned CT implementation
        # before creating an episode, so tokenizer/template failures cannot leave a
        # live service member. A shared prepared prompt avoids encoding the same group
        # input once for every candidate while keeping each episode's token list private.
        prepared_prompt = kwargs.get("initial_prompt_ids")
        if prepared_prompt is None:
            prepared_prompt = await self.build_initial_prompt_ids(messages)
        else:
            if self.processor is not None:
                raise ValueError("prepared ARC initial prompts currently support text-only loops")
            if not isinstance(prepared_prompt, PreparedInitialPrompt):
                raise ValueError("initial_prompt_ids must be a PreparedInitialPrompt from this ARC loop")
            expected_fingerprint = _initial_prompt_fingerprint(messages, self.tool_schema_dicts)
            if not hmac.compare_digest(prepared_prompt.fingerprint, expected_fingerprint):
                raise ValueError("prepared initial prompt does not match raw_prompt and ARC tools")
        runtime_token_ids = list(prepared_prompt.token_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        assistant_turns = 0
        tool_events: list[dict[str, Any]] = []
        constraint_segments: list[dict[str, Any]] = []
        tool_call_failures = 0
        tool_turns = 0
        context_turns = 0
        generated_seconds = 0.0
        service_seconds = 0.0
        num_preempted: int | None = None
        request_id_base = f"arc-{group_id}-{sample_id}-{uuid4().hex}"
        policy_version: int | None = None
        episode_id: str | None = None
        request_id: str | None = None
        succeeded = False
        started_at = time.monotonic()
        deadline = started_at + self.group_timeout

        headers = self._auth_headers()
        async with httpx.AsyncClient(
            base_url=self.env_url,
            headers=headers,
            timeout=httpx.Timeout(_HTTP_TIMEOUT_SECONDS),
            # GPU work can outlive the server's idle timeout. Reconnect for
            # control RPCs instead of risking a stale socket on a tool mutation.
            limits=httpx.Limits(max_keepalive_connections=0),
            transport=self.http_transport,
        ) as client:
            try:
                failure_phase = "create"
                create_response = await client.post(
                    "/episodes",
                    json={
                        "task_id": task_id,
                        "group_id": group_id,
                        "sample_id": sample_id,
                        "group_size": group_size,
                        "validation": validation,
                    },
                )
                self._expect_status(create_response, (200,), "episode creation")
                try:
                    created = create_response.json()
                except ValueError as exc:
                    raise ArcServiceError("ARC service episode response was not valid JSON") from exc
                if not isinstance(created, dict) or not isinstance(created.get("episode_id"), str):
                    raise ArcServiceError("ARC service episode response must contain episode_id")
                episode_id = created["episode_id"]
                try:
                    validate_service_action_language(created.get("action_language_fingerprint"))
                except RuntimeError as exc:
                    raise ArcServiceError(str(exc)) from exc
                service_observation = created.get("observation")
                if not isinstance(service_observation, str):
                    raise ArcServiceError("ARC service reset observation must be a string")
                if service_observation != raw_visible_question:
                    raise ArcServiceError("ARC service reset observation differs from raw_prompt's visible question")
                _used, remaining_budget, episode_budget = _budget_receipt(created)
                separate_budget = _separate_budget_fields(created)
                episode_caps = (created.get("generated_token_budget"), created.get("observation_token_budget"))

                for sequence in range(self.max_turns):
                    remaining_response = self.response_length - len(response_mask)
                    if remaining_response <= 0:
                        if policy_version is not None:
                            await self._discard_for_context_limit(client, episode_id, policy_version, deadline)
                        raise ArcServiceError("ARC response cap reached before the episode finished")
                    if remaining_budget <= 0:
                        finish_reason = "completed"
                        break
                    generation_plan = plan_generation(
                        remaining_budget=remaining_budget, episode_budget=episode_budget,
                        turn_max_tokens=self.turn_max_tokens, remaining_response=remaining_response,
                        **separate_budget,
                    )
                    # Keep the full history and leave observation/CT closure room.
                    # The episode allowance can exceed the remaining model context.
                    generation_limit = context_generation_limit(
                        generation_plan, prompt_tokens=len(runtime_token_ids),
                        context_limit=self.context_length,
                    )
                    if generation_limit <= 0:
                        logger.info(
                            "arc_rollout_budget_stop group=%s sample=%s prompt_tokens=%d "
                            "context_limit=%d observation_reserve=%d planned_generation=%d",
                            group_id, sample_id, len(runtime_token_ids), self.context_length,
                            generation_plan.observation_reserve, generation_plan.max_tokens,
                        )
                        finish_reason = "budget_reserve"
                        break
                    params = dict(sampling_params)
                    params.pop("max_new_tokens", None)
                    params["max_tokens"] = generation_limit
                    params["logprobs"] = True
                    if self.grammar_constraint is not None:
                        params["constraint"] = self.grammar_constraint
                    # Gateway generation IDs bind immutably to the full request body.
                    # A new turn has a new prompt after the previous ARC observation,
                    # while retries of this same turn must retain this exact ID.
                    request_id = f"{request_id_base}-t{sequence}"

                    n = prepared_prompt.shared_prefix_tokens
                    if tuple(runtime_token_ids[:n]) != prepared_prompt.token_ids[:n]:
                        raise ArcServiceError("ARC continuation changed the stable system/tool prefix")
                    generation_prompt_tokens = len(runtime_token_ids)
                    generation_started = time.perf_counter()
                    failure_phase = "generation"
                    token_output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=runtime_token_ids,
                        sampling_params=params,
                        shared_prefix_tokens=prepared_prompt.shared_prefix_tokens,
                    )
                    generated_seconds += time.perf_counter() - generation_started
                    failure_phase = "generation_output_validation"
                    if token_output.stop_reason in {"aborted", "abort"}:
                        raise ArcServiceError("server_manager.generate reported an aborted request")
                    generated_ids = token_output.token_ids
                    if not isinstance(generated_ids, list) or any(
                        isinstance(token, bool) or not isinstance(token, int) for token in generated_ids
                    ):
                        raise ArcServiceError("TokenOutput.token_ids must be a list of integers")
                    if len(generated_ids) > generation_limit:
                        raise ArcServiceError("Worker output exceeded its requested generation budget")
                    if not generated_ids:
                        raise ArcServiceError("server_manager.generate returned no generated tokens")
                    token_logprobs = token_output.log_probs
                    if token_logprobs is None or len(token_logprobs) != len(generated_ids):
                        raise ArcServiceError("TokenOutput.log_probs must align one-to-one with token_ids")
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in token_logprobs
                    ):
                        raise ArcServiceError("TokenOutput.log_probs must contain finite numeric values")

                    generation_metrics = _value(token_output.extra_fields, "generation_metrics", {})
                    reused = _value(generation_metrics, "reused_prompt_tokens")
                    cache_available = type(reused) is int and 0 <= reused <= generation_prompt_tokens
                    generation_cache = {
                        "available": cache_available,
                        "prompt_tokens": generation_prompt_tokens,
                        "reused_prompt_tokens": reused if cache_available else None,
                        "uncached_prompt_tokens": generation_prompt_tokens - reused if cache_available else None,
                    }
                    turn_version = self._policy_version(token_output.extra_fields)
                    if policy_version is None:
                        policy_version = turn_version
                    elif policy_version != turn_version:
                        raise ArcServiceError("policy version changed between ARC generation turns")

                    segment_start = len(runtime_token_ids)
                    segment_prefix = list(runtime_token_ids)
                    if self.grammar_constraint is not None:
                        receipt = _value(token_output.extra_fields, "sampling_config", {})
                        public_count = _value(receipt,"public_token_count")
                        if (_value(receipt,"distribution") != "grammar_conditioned_policy/v1"
                                or _value(receipt,"grammar_sha256") != self.grammar_constraint["sha256"]
                                or _value(receipt,"grammar_compiler_sha256") != self.grammar_constraint["compiler_sha256"]
                                or type(public_count) is not int or public_count != len(self.tokenizer)):
                            raise ArcServiceError("Worker did not attest the requested grammar and tokenizer domain")
                        constraint_segments.append({"start":segment_start,"stop":segment_start+len(generated_ids),
                            "grammar_sha256":self.grammar_constraint["sha256"],
                            "compiler_sha256":self.grammar_constraint["compiler_sha256"],
                            "tokenizer_sha256":self.constraint_tokenizer_sha256,"public_token_count":public_count})
                    assistant_turns += 1
                    if token_output.num_preempted is not None:
                        num_preempted = (num_preempted or 0) + int(token_output.num_preempted)
                    merge_result, response_mask, aligned_logprobs = await self.ct_merge_assistant_token(
                        runtime_token_ids,
                        generated_ids,
                        response_mask,
                        response_logprobs,
                        assistant_logprobs=[float(value) for value in token_logprobs],
                    )
                    runtime_token_ids = merge_result.token_ids
                    if self.grammar_constraint is not None and (
                        runtime_token_ids[:segment_start] != segment_prefix
                        or runtime_token_ids[segment_start:segment_start+len(generated_ids)] != generated_ids
                    ):
                        raise ArcServiceError("Continuous Token merge changed constrained sampled token IDs")
                    if aligned_logprobs is None:
                        raise ArcServiceError("Continuous Token merge lost generated logprobs")
                    response_logprobs = aligned_logprobs

                    assistant_content, parsed_calls = await self.tool_parser.extract_tool_calls(
                        generated_ids, self.tool_schemas
                    )
                    name, arguments, parse_status, requested_name, assistant_message = _prepare_assistant_turn(
                        self.tokenizer, generated_ids, assistant_content, parsed_calls,
                    )
                    valid_call = parse_status == "valid"
                    stop_reason = token_output.stop_reason
                    if not isinstance(stop_reason, str) or not stop_reason or len(stop_reason) > 64:
                        raise ArcServiceError("Worker returned an invalid generation stop reason")
                    limit_reached = stop_reason == "length" or len(generated_ids) >= generation_limit
                    # The service receives the original model IDs, never a re-encoded
                    # assistant message. Each mutation is issued exactly once.
                    turn_started = time.perf_counter()
                    failure_phase = "turn"
                    turn_response = await client.post(
                        f"/episodes/{episode_id}/turn",
                        json={
                            "sequence": sequence,
                            "name": name,
                            "arguments": arguments,
                            "generated_ids": generated_ids,
                            "generation_stop_reason": stop_reason,
                            "generation_limit_reached": limit_reached,
                            "parse_status": parse_status,
                            "requested_tool_name": requested_name,
                            "requested_max_tokens": generation_limit,
                            "parsed_call_count": len(parsed_calls),
                            "parsed_tool_names": [call.name for call in parsed_calls],
                        },
                    )
                    service_seconds += time.perf_counter() - turn_started
                    self._expect_status(turn_response, (200,), "turn")
                    try:
                        turn_data = turn_response.json()
                    except ValueError as exc:
                        raise ArcServiceError("ARC service turn response was not valid JSON") from exc
                    if not isinstance(turn_data, dict):
                        raise ArcServiceError("ARC service turn response must be a JSON object")
                    observation = turn_data.get("observation")
                    done = turn_data.get("done")
                    infra_abort = turn_data.get("infra_abort")
                    if not isinstance(observation, str) or not isinstance(done, bool) or not isinstance(infra_abort, bool):
                        raise ArcServiceError("ARC service turn response has invalid observation/done/infra_abort fields")

                    tokens_used, remaining_budget, reported_budget = _budget_receipt(turn_data)
                    if episode_caps != (turn_data.get("generated_token_budget"), turn_data.get("observation_token_budget")):
                        raise ArcServiceError("ARC separate episode budgets changed between turns")
                    separate_budget = _separate_budget_fields(turn_data)
                    if reported_budget != episode_budget:
                        raise ArcServiceError("ARC episode budget changed between turns")
                    outcome = turn_data.get("outcome")
                    reason = turn_data.get("reason")
                    retryable = turn_data.get("retryable")
                    retries_remaining = turn_data.get("retries_remaining")
                    observation_role = turn_data.get("observation_role")
                    generation_limit_failure = reason == "generation_limit" or limit_reached
                    pre_feedback_budget_failure = (
                        reason in {"token_budget_exceeded", "generated_budget_exceeded"} and observation_role == "none"
                    )
                    model_stop = (is_model_stop(parse_status, stop_reason, limit_reached)
                                  and not pre_feedback_budget_failure)
                    if (reason == "model_stop") != model_stop:
                        raise ArcServiceError("ARC service completion differs from the sampled stop")
                    failed_tool_call = (
                        (not valid_call
                         or (reason == "invalid_schema" and outcome in {
                             "rejection", "episode_complete"
                         })
                         or (reason == "tool_error" and outcome in {
                             "rejection", "episode_complete"
                         }))
                        and not model_stop and not generation_limit_failure and not pre_feedback_budget_failure
                    )
                    if failed_tool_call:
                        tool_call_failures += 1
                    if (failed_tool_call and done and not infra_abort and remaining_budget > 0
                            and reason not in {"token_budget_exceeded", "generated_budget_exceeded", "observation_budget_exceeded"}
                            and tool_call_failures < MAX_TOOL_CALL_FAILURES
                            and not (episode_caps[1] is not None and observation_role == "none")):
                        raise ArcServiceError("ARC service stopped before the tool-call failure limit")
                    if tool_call_failures > MAX_TOOL_CALL_FAILURES:
                        raise ArcServiceError("ARC service exceeded the tool-call failure limit")
                    expected_retryable = (
                        failed_tool_call and not done
                        and tool_call_failures < MAX_TOOL_CALL_FAILURES
                    )
                    expected_retries_remaining = (
                        MAX_TOOL_CALL_FAILURES - tool_call_failures if expected_retryable else 0
                    )
                    if (outcome not in {"tool_result", "rejection", "episode_complete"}
                            or not isinstance(reason, str) or not reason
                            or type(retryable) is not bool
                            or type(retries_remaining) is not int
                            or retryable is not expected_retryable
                            or retries_remaining != expected_retries_remaining
                            or (expected_retryable and (outcome != "rejection" or done))
                            or (not done and failed_tool_call and not expected_retryable)
                            or (done and failed_tool_call and outcome != "episode_complete")
                            or ((outcome == "rejection") != (retryable and not done))
                            or (not done and remaining_budget <= 0)
                            or observation_role not in {"tool", "user", "none"}
                            or bool(observation) != (observation_role != "none")
                            or (observation_role == "tool" and not valid_call)
                            or (model_stop and (not done or observation_role != "none"
                                                or outcome != "episode_complete"))):
                        raise ArcServiceError("ARC terminal/tool receipt is inconsistent")

                    messages.append(assistant_message)
                    event = {"sequence": sequence, "name": name, "arguments": arguments,
                             "generated_ids": list(generated_ids),
                             "behavior_logprobs": list(token_logprobs),
                             "observation": observation, "done": done, "infra_abort": infra_abort,
                             "generation_stop_reason": stop_reason,
                             "generation_limit_reached": limit_reached,
                             "parse_status": parse_status, "requested_tool_name": requested_name,
                             "requested_max_tokens": generation_limit,
                             "generation_cache": generation_cache,
                             "parsed_call_count": len(parsed_calls),
                             "parsed_tool_names": [call.name for call in parsed_calls],
                             "token_budget": episode_budget, "tokens_used": tokens_used,
                             "tokens_remaining": remaining_budget, "outcome": outcome,
                             "reason": reason, "retryable": retryable,
                             "retries_remaining": retries_remaining,
                             "observation_role": observation_role,
                             "budget_policy": BUDGET_POLICY_ID,
                             "budget_phase": generation_plan.phase}
                    if separate_budget:
                        event.update({key: turn_data[key] for key in (
                            "generated_token_budget", "observation_token_budget",
                            "generated_tokens", "observation_tokens")})
                    tool_events.append(event)
                    if observation:
                        before_ids = list(runtime_token_ids)
                        before_mask, before_probs = list(response_mask), list(response_logprobs)
                        boundary_count = 0
                        if observation_role == "user":
                            runtime_token_ids, response_mask, response_logprobs = _close_feedback_turn(
                                self.tokenizer, runtime_token_ids, response_mask, response_logprobs,
                            )
                            boundary_count = len(runtime_token_ids) - len(before_ids)
                        messages.append({"role": observation_role, "content": observation})
                        previous_messages = messages[:-1]
                        merge_result, response_mask, aligned_logprobs = await self.ct_merge_context_msg(
                            previous_messages, messages, runtime_token_ids, response_mask,
                            response_logprobs, tools=self.tool_schema_dicts,
                        )
                        runtime_token_ids = merge_result.token_ids
                        if aligned_logprobs is None or len(aligned_logprobs) != len(response_mask):
                            raise ArcServiceError("Continuous Token context merge misaligned response logprobs")
                        response_logprobs = aligned_logprobs
                        if (observation_role == "user" or self.constrained_decoding) and (
                                runtime_token_ids[:len(before_ids)] != before_ids
                                or response_mask[:len(before_mask)] != before_mask
                                or response_logprobs[:len(before_probs)] != before_probs
                                or any(response_mask[len(before_mask):])
                                or any(response_logprobs[len(before_probs):])):
                            raise ArcServiceError("terminal feedback changed sampled tokens or their loss metadata")
                        event["feedback_boundary_token_count"] = boundary_count
                        context_turns += 1
                        if observation_role == "tool":
                            tool_turns += 1

                    if infra_abort:
                        raise ArcServiceError("ARC service reported infrastructure abort")
                    if (len(response_mask) > self.response_length
                            or len(runtime_token_ids) > self.context_length):
                        await self._discard_for_context_limit(client, episode_id, policy_version, deadline)
                    if done:
                        finish_reason = "completed" if valid_call else "model_stop"
                        break
                    if len(response_mask) >= self.response_length:
                        await self._discard_for_context_limit(client, episode_id, policy_version, deadline)
                    if sequence + 1 >= self.max_turns:
                        finish_reason = "turn_limit"
                        break
                else:
                    finish_reason = "turn_limit"

                if not response_mask:
                    raise ArcServiceError("ARC rollout produced an empty response")
                if policy_version is None:
                    raise ArcServiceError("ARC rollout did not observe a policy version")

                finish_started = time.perf_counter()
                failure_phase = "finish"
                await self._finish_until_accepted(
                    client,
                    episode_id,
                    {
                        "reason": finish_reason,
                        "min_policy_version": policy_version,
                        "max_policy_version": policy_version,
                    },
                    deadline,
                )
                service_seconds += time.perf_counter() - finish_started

                finished_generation = getattr(self.server_manager, 'finish_episode_generation', None)
                if callable(finished_generation):
                    await finished_generation()
                failure_phase = "result"
                result = await self._poll_result(client, episode_id, deadline)
                failure_phase = "output_validation"
                reward, exact, cost = self._result_values(result)

                prompt_ids = runtime_token_ids[: len(runtime_token_ids) - len(response_mask)]
                response_ids = runtime_token_ids[len(prompt_ids) :]
                if len(response_ids) != len(response_mask) or len(response_logprobs) != len(response_mask):
                    raise ArcServiceError("final response ids, mask, and logprobs are not aligned")
                metrics = AgentLoopMetrics(
                    generate_sequences=generated_seconds,
                    tool_calls=float(tool_turns),
                    compute_score=service_seconds,
                    num_preempted=-1 if num_preempted is None else num_preempted,
                )
                output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    reward_score=reward,
                    num_turns=assistant_turns + context_turns + 1,
                    metrics=metrics,
                    extra_fields={
                        "reward_extra_info": {"exact": exact, "cost": cost,
                                              "successful_tool_calls": result["successful_tool_calls"],
                                              "tool_call_failures": result["tool_call_failures"],
                                              "termination_reason": result.get("termination_reason"),
                                              "training_verification": result["training_verification"],
                                              "reward_components": result["reward_components"],
                                              "reward_schema": result["reward_schema"],
                                              "attempt_exact": result.get("attempt_exact"),
                                              "per_target_exact": result.get("per_target_exact"),
                                              "competition_score": result.get("competition_score"),
                                              "generated_tokens": result.get("generated_tokens"),
                                              "observation_tokens": result.get("observation_tokens")},
                        "min_global_steps": policy_version,
                        "max_global_steps": policy_version,
                        "episode_id": episode_id,
                        "messages": messages,
                        "tool_events": tool_events,
                        **({"constraint_segments":constraint_segments} if self.grammar_constraint is not None else {}),
                    },
                )
                succeeded = True
                return output
            except BaseException as exc:
                # Keep the initiating phase/type before DELETE can cause
                # sibling HTTP 409s. Never log headers, bodies or exception
                # messages, which may contain private request data.
                log_failure = logger.warning if isinstance(exc, asyncio.CancelledError) else logger.error
                log_failure(
                    "primary_arc_rollout_failure group=%s sample=%s episode=%s phase=%s "
                    "request=%s error_type=%s cause_type=%s",
                    group_id, sample_id, episode_id, failure_phase, request_id,
                    type(exc).__name__, type(exc.__cause__).__name__ if exc.__cause__ else None,
                )
                if episode_id is not None and not succeeded:
                    await self._delete_episode(client, episode_id)
                raise
