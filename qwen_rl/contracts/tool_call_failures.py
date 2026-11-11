"""Shared training and evaluation tool-call failure policy."""

ENV_SCHEMA_VERSION = "arc-env/v9"
MAX_TOOL_CALL_FAILURES = 10
TOOL_CALL_FAILURE_COUNTING = "total_per_episode_including_execution_errors"


def tool_call_failure_feedback(reason: str, *, parse_status: str,
                               generation_limit_reached: bool,
                               requested_tool_name: str | None,
                               validation_message: str | None = None,
                               tool_call_failures: int | None = None,
                               done: bool | None = None) -> dict:
    """Render the same corrective payload for training, replay and evaluation."""
    messages = {
        "generation_limit": (
            "The generation limit was reached before an accepted tool call. Keep reasoning brief "
            "and emit one complete tool call."
        ),
        "malformed_tool_call": (
            "The tool call could not be parsed. Emit one complete call using the advertised "
            "chat-template format."
        ),
        "unregistered_tool": (
            "That top-level tool is not registered. Only arc_action is available; "
            "set action to the exact toolkit name and args to its JSON argument object."
        ),
        "invalid_schema": (
            "The tool arguments do not match the registered schema. arc_action requires action and "
            "args. Use the selected action definition. No action was executed."
        ),
        "multiple_calls": "Only one registered tool call is allowed in a turn.",
    }
    message = messages[reason]
    if reason == "invalid_schema" and isinstance(validation_message, str) and validation_message:
        message = f"{message} Detail: {validation_message[:1000]}"
    from qwen_rl.prompt import load_toolkit_runtime
    load_toolkit_runtime()
    from arc_toolkit.diagnostics import normalize_recovery
    recovery = normalize_recovery(requested_tool_name, message,
                                  {"code": reason, "hint": message}, stage="protocol")
    if tool_call_failures is not None and done is not None:
        recovery.update(retryable=not done,
                        retries_remaining=MAX_TOOL_CALL_FAILURES - tool_call_failures if not done else 0,
                        tool_call_failures=tool_call_failures, max_tool_call_failures=MAX_TOOL_CALL_FAILURES)
    return {
        "recovery": recovery,
        "status": "error",
        "error_type": reason,
        "message": message,
        "parse_status": parse_status,
        "generation_limit_reached": generation_limit_reached,
        "requested_tool_name": requested_tool_name,
        "available_tools": ["arc_action"],
    }
