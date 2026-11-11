"""Typed result returned by a Worker generation backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TOOL_CALL_STOP_STRING = "</tool_call>"


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Sampled sequence, its behavior logprobs, termination, and telemetry."""

    token_ids: list[int]
    log_probs: list[float]
    stop_reason: str
    metrics: dict[str, Any]
