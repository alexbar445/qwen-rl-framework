"""Shared, deterministic ARC observation and generation budget policy."""
from dataclasses import dataclass

BUDGET_POLICY_ID = "post-observation-reserve/v2"
_OBSERVATION_RESERVE = 512
_FINALIZATION_RESERVE = 4096
_CLOSING_TRIGGER_BUFFER = 1024


def budget_policy_identity() -> dict:
    return {"id": BUDGET_POLICY_ID, "observation_reserve": _OBSERVATION_RESERVE,
            "finalization_reserve": _FINALIZATION_RESERVE, "closing_trigger_buffer": _CLOSING_TRIGGER_BUFFER,
            "observation_scale_divisor": 16, "finalization_scale_divisor": 8,
            "tokens_left": "post_observation_safe_lower_bound",
            "observation_schema": "arc-budget-observation-v2"}


def _reserves(episode_budget: int) -> tuple[int, int]:
    if type(episode_budget) is not int or episode_budget < 1:
        raise ValueError("episode_budget must be a positive integer")
    # Small diagnostic contracts retain room for actual sampled output.
    return min(_OBSERVATION_RESERVE, episode_budget // 16), min(_FINALIZATION_RESERVE, episode_budget // 8)


def budget_observation_fields(remaining_tokens: int, episode_budget: int) -> dict:
    observation, finalization = _reserves(episode_budget)
    closing = remaining_tokens <= observation + finalization + _CLOSING_TRIGGER_BUFFER
    result = {"budget_policy": BUDGET_POLICY_ID, "budget_phase": "closing" if closing else "normal"}
    if closing:
        result["budget_guidance"] = (
            "Closing budget: keep reasoning brief. Make at most one arc_action per turn; "
            "you may make further calls in later turns while budget remains. "
            "Use catalogued action names and JSON args. If a suitable test artifact exists, "
            "submit it using the actual artifact ID and required target as keyword arguments. Otherwise apply "
            "the best supported rule to the remaining test targets, then submit. Avoid "
            "repeating diagnostics when the rule is established. Only real submissions count."
        )
    return result


@dataclass(frozen=True)
class GenerationBudget:
    max_tokens: int
    phase: str
    observation_reserve: int
    finalization_reserve: int


def plan_generation(*, remaining_budget: int, episode_budget: int,
                    turn_max_tokens: int, remaining_response: int,
                    remaining_generated_budget: int | None = None,
                    remaining_observation_budget: int | None = None) -> GenerationBudget:
    observation, finalization = _reserves(episode_budget)
    if (type(remaining_budget) is not int or remaining_budget < 0
            or type(turn_max_tokens) is not int or turn_max_tokens < 1
            or type(remaining_response) is not int or remaining_response < 0):
        raise ValueError("generation planning requires nonnegative balances and a positive turn cap")
    if remaining_generated_budget is not None or remaining_observation_budget is not None:
        if any(type(n) is not int or n < 0 for n in
               (remaining_generated_budget, remaining_observation_budget)):
            raise ValueError("separate remaining budgets must both be nonnegative integers")
        closing = remaining_generated_budget <= observation + finalization + _CLOSING_TRIGGER_BUFFER
        available = remaining_generated_budget
        if remaining_observation_budget == 0:
            available = 0
        return GenerationBudget(min(turn_max_tokens, remaining_response, available),
                                "closing" if closing else "normal",
                                min(observation, remaining_observation_budget), 0)
    closing = remaining_budget <= observation + finalization + _CLOSING_TRIGGER_BUFFER
    available = (remaining_budget - observation
                 if closing else remaining_budget - observation - finalization)
    return GenerationBudget(max(0, min(turn_max_tokens, remaining_response, available)),
                            "closing" if closing else "normal", observation, finalization)


def context_generation_limit(plan: GenerationBudget, *, prompt_tokens: int, context_limit: int) -> int:
    """Limit generation to the capacity shared by the learner and worker."""
    return max(0, min(plan.max_tokens, context_limit - prompt_tokens - plan.observation_reserve))


def replay_response_offsets(events: list[dict], input_ids: list[int], policy_mask: list[int]) -> list[int]:
    """Recover each sampling window from preserved model spans, including masked context."""
    if not input_ids or len(input_ids) != len(policy_mask) or any(v not in (0, 1) for v in policy_mask):
        raise ValueError("replay requires aligned full input IDs and a binary policy mask")
    positions = [index for index, flag in enumerate(policy_mask) if flag]
    if not positions:
        raise ValueError("replay requires sampled model tokens")
    cursor, offsets = 0, []
    for event in events:
        generated = event['generated_ids']
        if not generated or cursor + len(generated) > len(positions):
            raise ValueError("replay generated IDs do not cover the preserved model spans")
        begin = positions[cursor]
        if (positions[cursor:cursor + len(generated)] != list(range(begin, begin + len(generated)))
                or list(input_ids[begin:begin + len(generated)]) != generated):
            raise ValueError("replay sampled IDs or policy mask differ from the original turn")
        offsets.append(begin - positions[0])
        cursor += len(generated)
    if cursor != len(positions):
        raise ValueError("replay leaves unexplained sampled tokens")
    return offsets
