"""Model-free validation and preparation of complete DAPO trajectory windows."""
from __future__ import annotations


from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch

from ..contracts.gradients import guided_replay_matches_fresh_policy_version
from ..rewards import FAILURE_REWARD_VALUES


@dataclass(frozen=True)
class PreparedGradientRows:
    """A validated full window before complete groups are assigned to jobs."""

    rows: tuple[dict[str, Any], ...]
    groups: Mapping[Any, tuple[dict[str, Any], ...]]
    n_window_tokens: int
    reward_stats: Mapping[str, float]


def rare_success_advantage_scales(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int],
) -> list[float]:
    """Scale exact successes when a validated group has a rare native exact rate.

    The reward contract encodes exact success as reward >= 1; failures are <= 0.
    Replay and zero-credit rows do not enter the native exact-rate denominator.
    Callers apply these scales only to positive, already normalized advantages.
    """
    native = [i for i in indices if rows[i].get("row_kind", "fresh_student") == "fresh_student"]
    native_exact = sum(rows[i]["reward"] >= 1.0 for i in native)
    rare = bool(native) and 100 * native_exact < 34 * len(native)
    return [
        1.5 if rare and rows[i].get("row_kind") != "zero_credit" and rows[i]["reward"] >= 1.0 else 1.0
        for i in indices
    ]


def group_normalized_advantages(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[float], dict[str, float]]:
    """Return group-normalized advantages with the rare-success bonus and reward statistics."""
    if not rows:
        raise ValueError("group-normalized advantages require trajectory rows")
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["group_id"]].append(index)
    advantages = [0.0] * len(rows)
    reward_stds: list[float] = []
    for group_id, indices in groups.items():
        rewards = torch.tensor([float(rows[index]["reward"]) for index in indices], dtype=torch.float32)
        if rewards.numel() < 2 or not bool(torch.isfinite(rewards).all()):
            raise ValueError(f"staged group {group_id!r} has invalid rewards for advantage calculation")
        mean = rewards.mean()
        std = rewards.std(correction=1)
        if not bool(torch.isfinite(std)) or float(std) <= 0.0:
            raise ValueError(f"staged group {group_id!r} has zero or invalid reward variance")
        reward_stds.append(float(std))
        normalized = ((rewards - mean) / (std + 1e-4)).tolist()
        scales = rare_success_advantage_scales(rows, indices)
        for index, value, scale in zip(indices, normalized, scales, strict=True):
            advantages[index] = float(value * scale if value > 0.0 else value)
    return advantages, {
        "group_reward_std_min": min(reward_stds),
        "group_reward_std_mean": sum(reward_stds) / len(reward_stds),
        "group_reward_std_max": max(reward_stds),
        "group_zero_std_fraction": 0.0,
    }


def prepare_gradient_rows(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    group_size: int,
    groups_per_update: int,
    microbatch_size: int,
) -> PreparedGradientRows:
    """Validate complete groups and attach their normalized, rare-success-weighted advantages.

    The ordinary Trainer calls this on its full update. A gradient executor may
    call it on exactly one complete task group, but it must reject any
    `_parallel_advantage` supplied on the wire. Only the global token denominator
    is shared across asynchronous jobs; advantage calculation uses complete
    local groups.
    """
    for name, value, minimum in (
        ("group_size", group_size, 2),
        ("groups_per_update", groups_per_update, 1),
        ("microbatch_size", microbatch_size, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not trajectories:
        raise ValueError("gradient preparation requires at least one trajectory")

    grouped_rows: dict[Any, list[dict[str, Any]]] = {}
    grouped_prefixes: dict[Any, tuple[int, ...]] = {}
    grouped_replay_counts: dict[Any, int] = {}
    row_kinds = {"fresh_student", "native_replay", "guided_replay", "zero_credit"}
    for index, source in enumerate(trajectories):
        row = dict(source)
        row_kind = row.get("row_kind", "fresh_student")
        if row_kind not in row_kinds:
            raise ValueError(f"trajectory {index} has unsupported row_kind={row_kind!r}")
        if row_kind == "fresh_student" and ("behavior_logprobs" not in row or "policy_version" not in row):
            raise ValueError(f"fresh trajectory {index} is missing behavior logprobs or policy version")
        if row_kind != "fresh_student" and ("behavior_logprobs" in row or "policy_version" in row):
            raise ValueError(f"replay trajectory {index} must not contain behavior logprobs or policy version")
        group_id = row["group_id"]
        try:
            hash(group_id)
        except TypeError as exc:
            raise ValueError(f"trajectory {index} group_id must be hashable") from exc
        ids = row["input_ids"]
        mask = row["policy_mask"]
        if len(ids) != len(mask) or not any(mask):
            raise ValueError(f"trajectory {index} has malformed input_ids or policy_mask")
        reward = row.get("reward")
        if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(float(reward))):
            raise ValueError(f"trajectory {index} reward must be finite")
        prefix = tuple(ids[:mask.index(1)])
        if group_id in grouped_prefixes and prefix != grouped_prefixes[group_id]:
            raise ValueError(f"group {group_id!r} has mixed initial prompt prefixes")
        grouped_prefixes.setdefault(group_id, prefix)
        grouped_rows.setdefault(group_id, []).append(row)
        if row_kind != "fresh_student":
            grouped_replay_counts[group_id] = grouped_replay_counts.get(group_id, 0) + 1

    if len(grouped_rows) != groups_per_update:
        raise ValueError(
            f"gradient preparation requires exactly {groups_per_update} distinct groups, "
            f"got {len(grouped_rows)}"
        )
    for group_id, rows in grouped_rows.items():
        fresh_rows = [row for row in rows if row.get("row_kind", "fresh_student") == "fresh_student"]
        replay_rows = [row for row in rows if row.get("row_kind", "fresh_student") != "fresh_student"]
        if len(fresh_rows) != group_size:
            raise ValueError(
                f"group {group_id!r} has {len(fresh_rows)} fresh trajectories; "
                f"expected exactly {group_size} original candidates"
            )
        if grouped_replay_counts.get(group_id, 0) > 1:
            raise ValueError(f"group {group_id!r} contains more than one replay row")
        if replay_rows:
            if len(rows) != group_size + 1 or len(replay_rows) != 1:
                raise ValueError(
                    f"rescued group {group_id!r} must retain all {group_size} fresh rows "
                    "and append exactly one replay row"
                )
            if any(row["row_kind"] not in {"native_replay", "guided_replay"} for row in replay_rows):
                raise ValueError(f"group {group_id!r} has an unsupported replay source")
            replay_kind = replay_rows[0]["row_kind"]
            fresh_rewards = [float(row["reward"]) for row in fresh_rows]
            if replay_kind == "native_replay":
                if any(reward not in FAILURE_REWARD_VALUES for reward in fresh_rewards):
                    raise ValueError(
                        f"rescued group {group_id!r} with native replay requires original fresh "
                        "failure rewards to be -1, -0.5 or 0"
                    )
            else:
                if any(reward not in FAILURE_REWARD_VALUES and reward < 1.0
                       for reward in fresh_rewards):
                    raise ValueError(
                        f"guided replay group {group_id!r} requires fresh failure rewards or exact successes"
                    )
                if (any(reward >= 1.0 for reward in fresh_rewards)
                        and not guided_replay_matches_fresh_policy_version(fresh_rows, replay_rows[0])):
                    raise ValueError(
                        f"guided replay group {group_id!r} with native success requires matching "
                        "fresh and replay source policy versions"
                    )
        elif len(rows) != group_size:
            raise ValueError(
                f"group {group_id!r} has {len(rows)} trajectories; expected {group_size}"
            )
        rewards = torch.tensor([float(row["reward"]) for row in rows], dtype=torch.float32)
        if float(rewards.std(correction=0)) <= 0.0:
            raise ValueError(f"group {group_id!r} has zero reward variance")

    ordered_group_ids = list(grouped_rows)
    ordered_rows = [row for group_id in ordered_group_ids for row in grouped_rows[group_id]]
    if microbatch_size > len(ordered_rows):
        raise ValueError("microbatch_size cannot exceed the actual update row count")
    n_window_tokens = sum(
        sum(int(flag) for flag in row["policy_mask"][1:]) for row in ordered_rows
    )
    if n_window_tokens <= 0:
        raise ValueError("update contains no causal policy tokens")

    advantages, reward_stats = group_normalized_advantages(ordered_rows)
    for row, advantage in zip(ordered_rows, advantages, strict=True):
        row["_parallel_advantage"] = advantage
    return PreparedGradientRows(
        rows=tuple(ordered_rows),
        groups={group_id: tuple(group_rows) for group_id, group_rows in grouped_rows.items()},
        n_window_tokens=n_window_tokens,
        reward_stats=reward_stats,
    )
