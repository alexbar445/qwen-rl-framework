"""Validate native tool payloads with the pinned toolkit's canonical contract."""
from __future__ import annotations
from typing import Any

NATIVE_TOOL_NAMES = frozenset({"arc_action"})


def is_model_stop(parse_status: str, stop_reason: str, limit_reached: bool) -> bool:
    """Only a complete assistant turn without a tool call ends the episode."""
    return parse_status == "no_call" and stop_reason in {"eos", "stop"} and not limit_reached


def turn_rejection_reason(parse_status: str, stop_reason: str, limit_reached: bool,
                          schema_error: str | None = None) -> str | None:
    """Share truncation, completion and malformed-call boundaries across ARC hosts."""
    if limit_reached and (stop_reason == "length" or parse_status == "no_call"):
        return "generation_limit"
    if schema_error is not None:
        return schema_error
    if parse_status == "valid" or is_model_stop(parse_status, stop_reason, limit_reached):
        return None
    return {
        "no_call": "malformed_tool_call",
        "malformed": "malformed_tool_call",
        "unregistered_tool": "unregistered_tool",
        "invalid_schema": "invalid_schema",
        "multiple_calls": "multiple_calls",
    }[parse_status]


def tool_schema_issue(name: str | None, arguments: dict[str, Any]) -> dict[str, str] | None:
    """Keep parser status codes while sharing executable payload validation."""
    if name not in NATIVE_TOOL_NAMES:
        return {"code": "unregistered_tool", "message": "Only arc_action is a registered native tool."}
    from qwen_rl.prompt import load_toolkit_runtime
    load_toolkit_runtime()
    from arc_toolkit.native_protocol import validate_native_arguments
    try:
        validate_native_arguments(name, arguments)
    except (TypeError, ValueError, OverflowError) as exc:
        return {"code": "invalid_schema", "message": str(exc)[:1000]}
    return None


def tool_schema_error(name: str | None, arguments: dict[str, Any]) -> str | None:
    """Parser classification from the same validation result used by the service."""
    issue = tool_schema_issue(name, arguments)
    return issue["code"] if issue is not None else None
