#!/usr/bin/env python3
"""Exact-gated cost and training-verification rewards for a complete group."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .contracts.tool_call_failures import MAX_TOOL_CALL_FAILURES

REWARD_SCHEMA = "two-attempt-exact-group-cost-verification/v13"
VERIFICATION_COEFFICIENT = 0.30
FAILURE_REWARDS = {
    "initial_generation_limit": -1.0,
    "generation_limit": -0.5,
    "tool_failure_limit": -0.5,
    "token_budget_exceeded": -0.5,
    "generated_budget_exceeded": -0.5,
    "observation_budget_exceeded": -0.5,
    "budget_reserve": -0.5,
}
FAILURE_REWARD_VALUES = frozenset({0.0, *FAILURE_REWARDS.values()})


def validate_efficiency_coefficient(epsilon: float) -> None:
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or not 0 <= epsilon <= 0.05:
        raise ValueError("efficiency coefficient must be finite and in 0..0.05")


def _efficiencies(costs: Sequence[float]) -> list[float]:
    """Normalize cost within all exact winners: minimum 1, maximum 0, ties 0."""
    if not costs:
        return []
    low, high = min(costs), max(costs)
    if high == low:
        return [0.0] * len(costs)
    return [(high - cost) / (high - low) for cost in costs]


def group_rewards(costs: Iterable[float], exacts: Iterable[bool], epsilon: float, *,
                  successful_tool_calls: Iterable[int], tool_call_failures: Iterable[int],
                  termination_reasons: Iterable[str | None] | None = None,
                  verification_coverages: Iterable[float] | None = None) -> list[float]:
    """Tiered terminal-limit penalties; other failures 0; exact gets 1 + epsilon * E + 0.30 * V."""
    costs, exacts = list(costs), list(exacts)
    calls, failures = list(successful_tool_calls), list(tool_call_failures)
    if len(costs) != len(exacts) or any(type(exact) is not bool for exact in exacts):
        raise ValueError("costs and boolean exacts must be the same length")
    if any(isinstance(c, bool) or not isinstance(c, (int, float))
           or not math.isfinite(c) or c < 0 for c in costs):
        raise ValueError("costs must be finite nonnegative numbers")
    validate_efficiency_coefficient(epsilon)
    if len(calls) != len(costs) or any(type(n) is not int or n < 0 for n in calls):
        raise ValueError("successful_tool_calls must contain one nonnegative integer per trajectory")
    if len(failures) != len(costs) or any(type(n) is not int or not 0 <= n <= MAX_TOOL_CALL_FAILURES for n in failures):
        raise ValueError(f"tool_call_failures must contain one integer in 0..{MAX_TOOL_CALL_FAILURES} per trajectory")
    reasons = [None] * len(costs) if termination_reasons is None else list(termination_reasons)
    if len(reasons) != len(costs) or any(reason is not None and not isinstance(reason, str) for reason in reasons):
        raise ValueError("termination_reasons must contain one string or None per trajectory")
    coverages = [0.0] * len(costs) if verification_coverages is None else list(verification_coverages)
    if len(coverages) != len(costs) or any(type(v) not in (int, float)
            or not math.isfinite(v) or not 0 <= v <= 1 for v in coverages):
        raise ValueError("verification_coverages must contain one finite value in 0..1 per trajectory")
    winners = [i for i, exact in enumerate(exacts) if exact]
    rewards = [FAILURE_REWARDS.get(reason, 0.0) if not exact else 0.0
               for exact, reason in zip(exacts, reasons, strict=True)]
    efficiencies = _efficiencies([costs[i] for i in winners])
    for index, efficiency in zip(winners, efficiencies, strict=True):
        rewards[index] = 1.0 + epsilon * efficiency + VERIFICATION_COEFFICIENT * coverages[index]
    return rewards
