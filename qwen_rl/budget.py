#!/usr/bin/env python3
"""Single source of truth for the episode cost contract and the reward shaping constants.

Contract (RSI design §8):

    cost_i  = w_output * T_out_i + w_observation * T_obs_i
              T_out = tokens the policy itself generates (rendered assistant turn)
              T_obs = tokens of the tool observations fed back ("flash input", rendered)
    reward  = shared same-task group formula in qwen_rl/rewards.py
              (exact success gets 1 + epsilon * efficiency + 0.30 * verification coverage;
              failures retain their terminal-limit penalties)
    budget  = explicit raw episode token budget
              ``call_cap`` is a harness safety
              net that must never bind a legitimate solution
    wall clock is an **infrastructure abort** (rollout discarded), never a failure:
              §8 -- "执行器／机器故障独立记录，避免把环境故障当解题错误"

The runtime loads one complete JSON contract, and records its fingerprint in each run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .rewards import validate_efficiency_coefficient

OUTPUT_TOKEN_WEIGHT = 4.0
OBSERVATION_TOKEN_WEIGHT = 1.0


def token_cost(out_tokens: int, obs_tokens: int) -> float:
    """The fixed cost shared by ordinary ARC, generated tasks and BEPA."""
    return OUTPUT_TOKEN_WEIGHT * out_tokens + OBSERVATION_TOKEN_WEIGHT * obs_tokens


@dataclass(frozen=True)
class Contract:
    """Episode cost model + reward shaping. Immutable: derive with ``replace``."""

    w_output: float = OUTPUT_TOKEN_WEIGHT
    w_observation: float = OBSERVATION_TOKEN_WEIGHT
    epsilon: float = 0.05            # efficiency coefficient in the shared group reward
    token_budget: int = 12000       # raw generated + observation tokens per episode
    call_cap: int = 512              # harness safety net, never a reward term
    abort_seconds: int = 0           # infra abort threshold; 0 = disabled (batched sweeps,
                                     # where wall time is dominated by batch-mate waiting)
    generated_token_budget: int | None = None
    observation_token_budget: int | None = None

    def __post_init__(self) -> None:
        if self.abort_seconds < 0:
            raise ValueError("abort_seconds must be >= 0 (0 = no wall-clock abort)")
        if (type(self.w_output) not in (int, float) or self.w_output != OUTPUT_TOKEN_WEIGHT
                or type(self.w_observation) not in (int, float)
                or self.w_observation != OBSERVATION_TOKEN_WEIGHT):
            raise ValueError("token cost weights must be fixed at output=4 and observation=1")
        validate_efficiency_coefficient(self.epsilon)
        if self.token_budget <= 0:
            raise ValueError("token_budget must be > 0")
        if self.call_cap <= 0:
            raise ValueError("call_cap must be > 0")
        caps = (self.generated_token_budget, self.observation_token_budget)
        if caps != (None, None):
            if any(type(cap) is not int or cap <= 0 for cap in caps):
                raise ValueError("separate token budgets must both be positive integers")
            if sum(caps) != self.token_budget:
                raise ValueError("separate token budgets must sum to token_budget")

    # ------------------------------------------------------------------ costing
    def cost(self, out_tokens: int, obs_tokens: int) -> float:
        return token_cost(out_tokens, obs_tokens)

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def as_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


# --------------------------------------------------------------------------
# The canonical contract: exactly one definition site in the whole workspace.
# --------------------------------------------------------------------------
CONTRACT = Contract()

CONTRACT_FIELDS = tuple(CONTRACT.as_dict())
SEPARATE_BUDGET_FIELDS = ("generated_token_budget", "observation_token_budget")


def load_contract(path: str | Path) -> Contract:
    """Load a complete contract; unknown and missing fields are rejected."""
    data = json.loads(Path(path).read_text())
    unknown = set(data) - set(CONTRACT_FIELDS) - set(SEPARATE_BUDGET_FIELDS)
    if unknown:
        raise ValueError(f"unknown contract field(s): {sorted(unknown)}")
    missing = set(CONTRACT_FIELDS) - set(data)
    if missing:
        raise ValueError(f"missing contract field(s): {sorted(missing)}")
    return Contract(**data)
