"""Shared learner-side trajectory validation, independent of compute backend."""
from __future__ import annotations


import hashlib
import json
import math
import copy
from typing import Any

from ..contracts.gradients import guided_replay_matches_fresh_policy_version
from ..rewards import FAILURE_REWARD_VALUES


def validate_batch(
    trajectories: list[dict[str, Any]],
    *,
    learner_start_version: int,
    max_sequence_length: int,
    group_size: int,
    groups_per_update: int,
) -> tuple[list[dict[str, Any]], int, dict[Any, str], dict[Any, int]]:
    if not trajectories:
        raise ValueError("update requires at least one trajectory")
    if (isinstance(learner_start_version, bool)
            or not isinstance(learner_start_version, int)
            or learner_start_version < 0):
        raise ValueError("learner_start_version must be a non-negative integer")
    normalized: list[dict[str, Any]] = []
    groups: dict[Any, list[float]] = {}
    group_rows: dict[Any, list[dict[str, Any]]] = {}
    behavior_versions: set[int] = set()
    total_tokens = 0
    group_prefixes: dict[Any, tuple[int, ...]] = {}
    group_prefix_hashes: dict[Any, str] = {}
    group_sizes: dict[Any, int] = {}
    group_replay_counts: dict[Any, int] = {}
    row_kinds = {"fresh_student", "native_replay", "guided_replay"}
    for index, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, dict):
            raise ValueError(f"trajectory {index} must be a mapping")
        row_kind = trajectory.get("row_kind", "fresh_student")
        if not isinstance(row_kind, str) or row_kind not in row_kinds:
            raise ValueError(f"trajectory {index} has unsupported row_kind={row_kind!r}")
        required = {"input_ids", "policy_mask", "reward", "group_id"}
        if row_kind == "fresh_student":
            required.update({"behavior_logprobs", "policy_version"})
        missing = required - trajectory.keys()
        if missing:
            raise ValueError(f"trajectory {index} missing fields: {sorted(missing)}")
        if not isinstance(trajectory["input_ids"], (list, tuple)):
            raise ValueError(f"trajectory {index} input_ids must be a sequence")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0
               for token in trajectory["input_ids"]):
            raise ValueError(f"trajectory {index} input_ids must contain non-negative integers")
        ids = list(trajectory["input_ids"])
        raw_mask = trajectory["policy_mask"]
        if not isinstance(raw_mask, (list, tuple)):
            raise ValueError(f"trajectory {index} policy_mask must be a sequence")
        if any(value not in (0, 1, False, True) for value in raw_mask):
            raise ValueError(f"trajectory {index} policy_mask must contain only 0/1")
        mask = [int(value) for value in raw_mask]
        reward_value = trajectory["reward"]
        if isinstance(reward_value, bool) or not isinstance(reward_value, (int, float)):
            raise ValueError(f"trajectory {index} reward must be a finite number")
        reward = float(reward_value)
        group_id = trajectory["group_id"]
        try:
            hash(group_id)
        except TypeError as exc:
            raise ValueError(f"trajectory {index} group_id must be hashable") from exc
        if len(ids) != len(mask):
            raise ValueError(f"trajectory {index} token and mask lengths differ")
        if len(ids) < 2 or len(ids) > max_sequence_length:
            raise ValueError(f"trajectory {index} sequence length is outside the configured limit")
        if any(value not in (0, 1) for value in mask):
            raise ValueError(f"trajectory {index} policy_mask must contain only 0/1")
        active = [position for position, flag in enumerate(mask) if flag and position > 0]
        if not active:
            raise ValueError(f"trajectory {index} has no policy token after causal shift")
        if not math.isfinite(reward):
            raise ValueError(f"trajectory {index} has a non-finite reward")

        normalized_row = {
            "input_ids": ids,
            "policy_mask": mask,
            "reward": reward,
            "group_id": group_id,
            "row_kind": row_kind,
        }
        worker_id = trajectory.get("worker_id")
        if worker_id is not None:
            if not isinstance(worker_id, str) or not worker_id.strip():
                raise ValueError(f"trajectory {index} worker_id must be a non-empty string")
            normalized_row["worker_id"] = worker_id
        for identity in ('source_worker_id', 'generation_worker_id'):
            if identity in trajectory:
                value = trajectory[identity]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"trajectory {index} {identity} must be a non-empty string")
                normalized_row[identity] = value
        if 'generation_worker_history' in trajectory:
            history = trajectory['generation_worker_history']
            if (not isinstance(history, list) or not history
                    or any(not isinstance(worker, str) or not worker for worker in history)):
                raise ValueError('generation Worker history must contain nonempty Worker IDs')
            normalized_row['generation_worker_history'] = list(history)
        if "episode_id" in trajectory:
            normalized_row["episode_id"] = trajectory["episode_id"]
        if "constraint_segments" in trajectory:
            segments = trajectory["constraint_segments"]
            if not isinstance(segments, list):
                raise ValueError(f"trajectory {index} constraint_segments must be a list")
            normalized_row["constraint_segments"] = copy.deepcopy(segments)
        if row_kind == "fresh_student":
            raw_behavior = trajectory["behavior_logprobs"]
            if not isinstance(raw_behavior, (list, tuple)):
                raise ValueError(f"trajectory {index} behavior_logprobs must be a sequence")
            if len(ids) != len(raw_behavior):
                raise ValueError(f"trajectory {index} token and behavior logprob lengths differ")
            behavior = [float(value) for value in raw_behavior]
            if not all(math.isfinite(behavior[pos]) for pos in active):
                raise ValueError(f"trajectory {index} has a non-finite behavior logprob")
            behavior_version = trajectory["policy_version"]
            if (isinstance(behavior_version, bool)
                    or not isinstance(behavior_version, int)
                    or behavior_version < 0):
                raise ValueError(f"trajectory {index} policy_version must be a non-negative integer")
            if behavior_version > learner_start_version:
                raise ValueError(
                    f"trajectory {index} behavior policy_version={behavior_version} is newer than "
                    f"learner_start_version={learner_start_version}"
                )
            policy_lag = learner_start_version - behavior_version
            if policy_lag > 1:
                raise ValueError(
                    f"trajectory {index} behavior policy_version={behavior_version} exceeds "
                    f"max_policy_lag=1 at learner_start_version={learner_start_version}"
                )
            behavior_versions.add(behavior_version)
            normalized_row.update({
                "behavior_logprobs": behavior,
                "policy_version": behavior_version,
            })
        else:
            if "behavior_logprobs" in trajectory or "policy_version" in trajectory:
                raise ValueError(
                    f"trajectory {index} replay rows must not contain ordinary behavior_logprobs or policy_version"
                )
            source_policy = trajectory.get("source_policy")
            if not isinstance(source_policy, dict):
                raise ValueError(f"trajectory {index} replay row is missing source_policy")
            source_epoch = source_policy.get("engine_epoch")
            source_version = source_policy.get("version")
            source_sha = source_policy.get("adapter_sha256")
            if not isinstance(source_epoch, str) or not source_epoch.strip():
                raise ValueError(f"trajectory {index} replay source_policy.engine_epoch must be non-empty")
            if (isinstance(source_version, bool) or not isinstance(source_version, int)
                    or source_version < 0):
                raise ValueError(f"trajectory {index} replay source_policy.version must be non-negative")
            if (not isinstance(source_sha, str) or len(source_sha) != 64
                    or any(char not in "0123456789abcdef" for char in source_sha.lower())):
                raise ValueError(f"trajectory {index} replay source_policy.adapter_sha256 must be a SHA-256 hex digest")
            cache_entry_id = trajectory.get("cache_entry_id")
            if not isinstance(cache_entry_id, str) or not cache_entry_id.strip():
                raise ValueError(f"trajectory {index} replay cache_entry_id must be non-empty")
            receipt = trajectory.get("verification_receipt")
            if not isinstance(receipt, dict):
                raise ValueError(f"trajectory {index} replay row is missing verification_receipt")
            if (not isinstance(receipt.get("task_key"), str) or not receipt["task_key"].strip()
                    or not isinstance(receipt.get("verifier_contract"), str)
                    or not receipt["verifier_contract"].strip()
                    or receipt.get("exact") is not True):
                raise ValueError(f"trajectory {index} replay verification_receipt is incomplete or not exact")
            projection_sha = receipt.get("projection_sha256")
            if (not isinstance(projection_sha, str) or len(projection_sha) != 64
                    or any(char not in "0123456789abcdef" for char in projection_sha.lower())):
                raise ValueError(f"trajectory {index} replay receipt projection_sha256 must be a SHA-256 hex digest")
            normalized_row.update({
                "source_policy": dict(source_policy),
                "cache_entry_id": cache_entry_id,
                "verification_receipt": dict(receipt),
            })
            group_replay_counts[group_id] = group_replay_counts.get(group_id, 0) + 1

        first_policy_position = mask.index(1)
        if first_policy_position == 0:
            raise ValueError(f"trajectory {index} has no initial prompt before its first policy token")
        prefix = tuple(ids[:first_policy_position])
        expected_prefix = group_prefixes.setdefault(group_id, prefix)
        if prefix != expected_prefix:
            raise ValueError(
                f"group {group_id!r} has mixed initial prompt prefixes before its first policy token"
            )
        group_prefix_hashes.setdefault(
            group_id,
            hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest(),
        )
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
        total_tokens += len(active)
        groups.setdefault(group_id, []).append(reward)
        group_rows.setdefault(group_id, []).append(normalized_row)
    for group_id, rewards in groups.items():
        rows = group_rows[group_id]
        fresh_rows = [row for row in rows if row["row_kind"] == "fresh_student"]
        replay_rows = [row for row in rows if row["row_kind"] != "fresh_student"]
        if len(fresh_rows) != group_size:
            raise ValueError(
                f"group {group_id!r} has {len(rows)} trajectories; expected {group_size} "
                f"fresh original candidates, got {len(fresh_rows)} fresh rows"
            )
        if group_replay_counts.get(group_id, 0) > 1:
            raise ValueError(f"group {group_id!r} contains more than one replay row")
        if replay_rows:
            if len(rows) != group_size + 1 or len(replay_rows) != 1:
                raise ValueError(
                    f"rescued group {group_id!r} must retain all {group_size} fresh rows "
                    "and append exactly one replay row"
                )
            if replay_rows[0]["row_kind"] not in {"native_replay", "guided_replay"}:
                raise ValueError(f"group {group_id!r} has an unsupported replay source")
            replay_kind = replay_rows[0]["row_kind"]
            fresh_rewards = [row["reward"] for row in fresh_rows]
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
        if not any(reward > 0.0 for reward in rewards):
            raise ValueError(f"group {group_id!r} has no positive reward")
        mean_reward = sum(rewards) / len(rewards)
        reward_variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        if reward_variance <= 0.0:
            raise ValueError(f"group {group_id!r} has zero reward variance")
    if len(groups) != groups_per_update:
        raise ValueError(
            f"update requires exactly {groups_per_update} complete fresh groups, got {len(groups)}"
        )
    if not behavior_versions:
        raise ValueError("update batch must contain at least one fresh student row")
    if len(behavior_versions) != 1:
        raise ValueError("fresh rows must contain a single behavior policy_version; mixed versions are rejected")
    normalized = [row for group_id in groups for row in group_rows[group_id]]
    return normalized, total_tokens, group_prefix_hashes, group_sizes
