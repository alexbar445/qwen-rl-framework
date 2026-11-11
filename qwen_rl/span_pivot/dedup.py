"""Conservative removal of proved, retry-only read rechecks.

This module operates on a derived event list. Callers must keep the original
trajectory archive immutable. It recognizes only native OpenAI ``arc_action``
tool calls whose command is an unprefixed ``inspect(...)`` or ``tools(...)``
call with literal keyword arguments. Any ambiguity retains the events.

The event-level ``proposal`` field is audit metadata, not model text. A retry
turn is eligible only when it is exactly tagged ``{"kind": "retry_only"}``
and its reasoning is one of the small, literal retry phrases below.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_ASSISTANT_KEYS = {"event_id", "role", "content", "reasoning_content", "tool_calls", "proposal"}
_TOOL_KEYS = {"event_id", "role", "tool_call_id", "content", "receipt", "name", "status"}
_CALL_KEYS = {"id", "type", "function"}
_FUNCTION_KEYS = {"name", "arguments"}
_ACTION_ARGUMENT_KEYS = {"line", "rationale"}

# These fields are explicitly per-call/per-observation receipt metadata. They
# are ignored only while comparing receipt metadata; the tool response body is
# compared in full, including any timestamps or counters it contains.
_VOLATILE_RECEIPT_FIELDS = frozenset({
    "call_id",
    "completed_at",
    "created_at",
    "duration_ms",
    "elapsed_ms",
    "latency_ms",
    "receipt_id",
    "request_id",
    "sequence",
    "span_id",
    "started_at",
    "timestamp",
    "tool_call_id",
    "trace_id",
})

_RETRY_ONLY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"wait\s*,?\s+let\s+me\s+retry[.!]?",
        r"wait\s*,?\s+let\s+retry[.!]?",
        r"let\s+me\s+retry[.!]?",
        r"retry\s+the\s+same\s+read[.!]?",
        r"retrying\s+the\s+same\s+read[.!]?",
    )
)


@dataclass(frozen=True)
class _ReadCall:
    call_id: str
    function_name: str
    arguments: dict[str, Any]
    line: str
    readset_canonical: str
    readset_sha256: str


@dataclass(frozen=True)
class _Turn:
    assistant_index: int
    tool_index: int
    assistant: dict[str, Any]
    tool: dict[str, Any]
    read_call: _ReadCall


def deduplicate_rechecks(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a copied retained event list and audit records for read rechecks.

    A repeated turn is removed only when it immediately follows a successful
    read-only turn, has the same canonical read set and same state, both real
    tool responses carry successful complete side-effect-free receipts, the
    canonical response bodies and observation hashes match, and the retry
    assistant turn is explicitly retry-only. The source events passed here are
    never mutated.
    """
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise TypeError("events must be a list of event dictionaries")

    original = copy.deepcopy(events)
    turns: list[_Turn] = []
    index = 0
    while index + 1 < len(original):
        assistant = original[index]
        tool = original[index + 1]
        if _role(assistant) != "assistant" or _role(tool) != "tool":
            index += 1
            continue
        read_call = _read_call(assistant)
        if read_call is not None:
            turns.append(_Turn(index, index + 1, assistant, tool, read_call))
        index += 2

    deletions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    removed_indices: set[int] = set()

    previous_turn: _Turn | None = None
    retained_anchor: _Turn | None = None
    for retry in turns:
        if (previous_turn is None or retained_anchor is None
                or retry.assistant_index != previous_turn.tool_index + 1):
            previous_turn = retry
            retained_anchor = retry
            continue

        candidate = _candidate(retained_anchor, retry)
        reason, proof = _prove_candidate(retained_anchor, retry, original)
        if reason is None:
            removed_indices.update((retry.assistant_index, retry.tool_index))
            proof = proof or {}
            deletion = {
                "source_event_ids": [
                    retry.assistant.get("event_id"),
                    retry.tool.get("event_id"),
                ],
                "kept_call_id": retained_anchor.read_call.call_id,
                "reason": "proved_same_state_readonly_recheck",
                "proof": proof,
            }
            deletions.append(deletion)
            candidate.update({
                "status": "deleted",
                "reason": deletion["reason"],
                "proof": copy.deepcopy(proof),
            })
        else:
            candidate.update({"status": "retained", "reason": reason, "proof": proof})
            retained_anchor = retry
        candidates.append(candidate)
        previous_turn = retry

    retained = [event for i, event in enumerate(original) if i not in removed_indices]
    return {"events": retained, "deletions": deletions, "candidates": candidates}


def _role(event: Any) -> str | None:
    return event.get("role") if isinstance(event, Mapping) else None


def _candidate(previous: _Turn, retry: _Turn) -> dict[str, Any] | None:
    """Describe a nearby pair of eligible read calls, even when proof fails."""
    base_call = previous.read_call
    retry_call = retry.read_call
    return {
        "source_event_ids": [retry.assistant.get("event_id"), retry.tool.get("event_id")],
        "kept_call_id": base_call.call_id,
        "candidate_call_id": retry_call.call_id,
        "status": "pending",
        "reason": "proof_pending",
        "proof": None,
    }


def _prove_candidate(
    previous: _Turn,
    retry: _Turn,
    events: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    first = previous.read_call
    second = retry.read_call

    if not _minimal_assistant_event(retry.assistant):
        return "retry_assistant_has_unreviewed_fields", None
    if not _minimal_tool_event(previous.tool) or not _minimal_tool_event(retry.tool):
        return "tool_response_has_unreviewed_fields", None

    proposal = retry.assistant.get("proposal")
    if not isinstance(proposal, Mapping) or dict(proposal) != {"kind": "retry_only"}:
        return "retry_only_proposal_missing", None

    if not _retry_only_reasoning(retry.assistant.get("reasoning_content")):
        return "reasoning_not_literal_retry_only", None
    retry_args = second.arguments
    rationale = retry_args.get("rationale", "")
    if rationale and not _retry_only_reasoning(rationale):
        return "tool_rationale_not_literal_retry_only", None
    if retry.assistant.get("content") not in (None, ""):
        if not (isinstance(retry.assistant.get("content"), str)
                and retry.assistant["content"].strip() == ""):
            return "retry_assistant_has_user_visible_content", None

    first_event_id = previous.assistant.get("event_id")
    first_tool_event_id = previous.tool.get("event_id")
    retry_event_id = retry.assistant.get("event_id")
    retry_tool_event_id = retry.tool.get("event_id")
    source_ids = (first_event_id, first_tool_event_id, retry_event_id, retry_tool_event_id)
    if any(not isinstance(value, str) or not value for value in source_ids):
        return "source_event_id_missing", None
    if len(set(source_ids)) != len(source_ids):
        return "source_event_id_collision", None

    if previous.tool.get("tool_call_id") != first.call_id:
        return "prior_response_not_linked_to_call", None
    if retry.tool.get("tool_call_id") != second.call_id:
        return "retry_response_not_linked_to_call", None
    if first.call_id == second.call_id:
        return "retry_call_id_collision", None

    if first.readset_canonical != second.readset_canonical:
        return "readset_changed", None

    first_response = _successful_response(previous.tool)
    if first_response is None:
        return "prior_response_not_successful_or_complete", None
    retry_response = _successful_response(retry.tool)
    if retry_response is None:
        return "retry_response_not_successful_or_complete", None

    first_receipt, first_body, first_body_hash = first_response
    retry_receipt, retry_body, retry_body_hash = retry_response

    if not _same_stable_receipt_metadata(first_receipt, retry_receipt):
        return "receipt_metadata_changed", None
    if not _same_state(first_receipt, retry_receipt):
        return "state_changed", None
    if first_receipt["observation_hash"] != retry_receipt["observation_hash"]:
        return "observation_hash_changed", None
    if first_body != retry_body:
        return "observation_body_changed", None

    if _references_call_id(events, second.call_id, {
        previous.assistant_index,
        previous.tool_index,
        retry.assistant_index,
        retry.tool_index,
    }):
        return "duplicate_call_id_referenced", None

    proof = {
        "prior_event_ids": [first_event_id, first_tool_event_id],
        "retry_event_ids": [retry_event_id, retry_tool_event_id],
        "kept_call_id": first.call_id,
        "removed_call_id": second.call_id,
        "proposal_kind": "retry_only",
        "response_status": "ok",
        "prior_response_linked": True,
        "retry_response_linked": True,
        "same_readset": True,
        "readset_sha256": first.readset_sha256,
        "same_state": True,
        "canonical_observation_equal": True,
        "observation_hash_equal": True,
        "observation_hash": first_receipt["observation_hash"],
        "observation_body_sha256": first_body_hash,
        "state_hash_before": first_receipt["state_hash_before"],
        "state_hash_after": first_receipt["state_hash_after"],
        "side_effect_free": True,
        "complete": True,
    }
    return None, proof


def _minimal_assistant_event(event: Mapping[str, Any]) -> bool:
    return set(event).issubset(_ASSISTANT_KEYS) and event.get("role") == "assistant"


def _minimal_tool_event(event: Mapping[str, Any]) -> bool:
    if not set(event).issubset(_TOOL_KEYS) or event.get("role") != "tool":
        return False
    if "name" in event and event.get("name") != "arc_action":
        return False
    if "status" in event and event.get("status") != "ok":
        return False
    return True


def _retry_only_reasoning(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    return bool(candidate) and any(pattern.fullmatch(candidate) for pattern in _RETRY_ONLY_PATTERNS)


def _read_call(assistant: Mapping[str, Any]) -> _ReadCall | None:
    if not _minimal_assistant_event(assistant):
        return None
    calls = assistant.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    if not isinstance(call, Mapping) or set(call) != _CALL_KEYS:
        return None
    if call.get("type") != "function" or not isinstance(call.get("id"), str) or not call["id"]:
        return None
    function = call.get("function")
    if not isinstance(function, Mapping) or set(function) != _FUNCTION_KEYS:
        return None
    if function.get("name") != "arc_action":
        return None
    arguments = _decode_arguments(function.get("arguments"))
    if arguments is None or set(arguments) - _ACTION_ARGUMENT_KEYS:
        return None
    line = arguments.get("line")
    rationale = arguments.get("rationale", "")
    if not isinstance(line, str) or not line.strip() or not isinstance(rationale, str):
        return None
    readset = _canonical_readset(line)
    if readset is None:
        return None
    return _ReadCall(
        call_id=call["id"],
        function_name="arc_action",
        arguments=arguments,
        line=line,
        readset_canonical=readset,
        readset_sha256=hashlib.sha256(readset.encode("utf-8")).hexdigest(),
    )


def _decode_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            decoded = json.loads(
                value,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    elif isinstance(value, Mapping):
        decoded = dict(value)
    else:
        return None
    if not isinstance(decoded, dict) or any(not isinstance(key, str) for key in decoded):
        return None
    try:
        _canonical_json(decoded)
    except (TypeError, ValueError):
        return None
    return decoded


def _canonical_readset(line: str) -> str | None:
    from qwen_rl.prompt import load_toolkit_runtime
    runtime = load_toolkit_runtime()
    try:
        function_name, kwargs = runtime._parse_call(line)
    except runtime.CallRejected:
        return None
    if function_name not in {"inspect", "tools"}:
        return None

    if function_name == "inspect":
        if set(kwargs) - {"source", "limit"} or not isinstance(kwargs.get("source"), str):
            return None
        if "limit" in kwargs and type(kwargs["limit"]) is not int:
            return None
    else:
        if set(kwargs) - {"name"} or ("name" in kwargs and not isinstance(kwargs["name"], str)):
            return None

    # Keyword ordering and quote style are presentation details. The function
    # name, keyword names, literal types, and values form the exact read set.
    return _canonical_json({
        "function": function_name,
        "keywords": {key: kwargs[key] for key in sorted(kwargs)},
    })


def _successful_response(tool_event: Mapping[str, Any]) -> tuple[dict[str, Any], str, str] | None:
    parsed = _parse_json_object(tool_event.get("content"))
    if parsed is None or parsed.get("status") != "ok":
        return None
    if "status" in tool_event and tool_event.get("status") != "ok":
        return None
    receipt = tool_event.get("receipt")
    if not isinstance(receipt, Mapping):
        return None
    for key in ("state_hash_before", "state_hash_after", "observation_hash"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            return None
    if receipt.get("side_effect_free") is not True or receipt.get("complete") is not True:
        return None
    try:
        body = _canonical_json(parsed)
    except (TypeError, ValueError):
        return None
    return dict(receipt), body, hashlib.sha256(body.encode("utf-8")).hexdigest()


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            parsed = json.loads(
                value,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    elif isinstance(value, Mapping):
        parsed = dict(value)
    else:
        return None
    if not isinstance(parsed, dict) or any(not isinstance(key, str) for key in parsed):
        return None
    try:
        _canonical_json(parsed)
    except (TypeError, ValueError):
        return None
    return parsed


def _same_state(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    before = first["state_hash_before"]
    after = first["state_hash_after"]
    return (
        before == after
        and second["state_hash_before"] == second["state_hash_after"]
        and before == second["state_hash_before"]
    )


def _same_stable_receipt_metadata(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    proof_fields = {"state_hash_before", "state_hash_after", "observation_hash", "side_effect_free", "complete"}
    try:
        first_stable = {key: value for key, value in first.items() if key not in proof_fields}
        second_stable = {key: value for key, value in second.items() if key not in proof_fields}
        return _canonical_json(_without_volatile_receipt_fields(first_stable)) == _canonical_json(
            _without_volatile_receipt_fields(second_stable)
        )
    except (TypeError, ValueError):
        return False


def _without_volatile_receipt_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_volatile_receipt_fields(item)
            for key, item in value.items()
            if key not in _VOLATILE_RECEIPT_FIELDS
        }
    if isinstance(value, list):
        return [_without_volatile_receipt_fields(item) for item in value]
    return value


def _references_call_id(events: list[dict[str, Any]], call_id: str, excluded_indices: set[int]) -> bool:
    for index, event in enumerate(events):
        if index not in excluded_indices and _contains_string_reference(event, call_id):
            return True
    return False


def _contains_string_reference(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        # A substring match is intentional: it may retain a duplicate when an
        # ID appears incidentally in prose, but it cannot delete a referenced ID.
        return needle in value
    if isinstance(value, Mapping):
        return any(_contains_string_reference(key, needle)
                   or _contains_string_reference(item, needle)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_string_reference(item, needle) for item in value)
    return False


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
