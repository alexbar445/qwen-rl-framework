"""Canonical evaluation row, artifact, and comparison contracts."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json_atomic(path: Path, value: Any) -> str:
    """Write JSON without replacing a partial artifact and return its byte digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return sha256_bytes(payload)


@dataclass(frozen=True)
class EpisodeExecution:
    """One production ARC loop result plus its raw, non-training trajectory."""

    trajectory: dict[str, Any]
    worker_id: str
    episode_id: str | None
    duration_seconds: float
    tool_errors: int
    group_id: str
    worker_lease_id: str | None = None
    sampling_config: dict[str, Any] = field(default_factory=dict)
    turn_seed_receipts: tuple[int, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.trajectory, dict):
            raise TypeError("evaluation trajectory must be a dictionary")
        if not isinstance(self.worker_id, str) or not self.worker_id:
            raise ValueError("evaluation execution requires a worker_id")
        if self.episode_id is not None and (not isinstance(self.episode_id, str) or not self.episode_id):
            raise ValueError("episode_id must be a non-empty string or None")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("evaluation execution requires its runner group_id")
        if (isinstance(self.duration_seconds, bool)
                or not isinstance(self.duration_seconds, (int, float))
                or not math.isfinite(self.duration_seconds) or self.duration_seconds < 0):
            raise ValueError("duration_seconds must be finite and nonnegative")
        if isinstance(self.tool_errors, bool) or not isinstance(self.tool_errors, int) or self.tool_errors < 0:
            raise ValueError("tool_errors must be a nonnegative integer")
        if self.worker_lease_id is not None and (
            not isinstance(self.worker_lease_id, str) or not self.worker_lease_id
        ):
            raise ValueError("worker_lease_id must be a non-empty string or None")
        if not isinstance(self.sampling_config, dict) or not isinstance(self.settings, dict):
            raise ValueError("sampling_config and settings must be dictionaries")
        if (not isinstance(self.turn_seed_receipts, tuple)
                or any(type(seed) is not int or not 0 <= seed <= (1 << 32) - 1
                       for seed in self.turn_seed_receipts)):
            raise ValueError("turn_seed_receipts must contain only 32-bit nonnegative integer seeds")


@dataclass(frozen=True)
class EvaluationEpisodeResult:
    """Compact state row pointing to a separate full episode artifact."""

    task_id: str
    attempt_id: str
    seed: int
    policy_role: str
    policy_version: int
    adapter_sha256: str
    episode_id: str | None
    group_id: str
    worker_id: str | None
    exact: bool | None
    passed: bool | None
    reward: float | None
    generated_tokens: int | None
    observation_tokens: int | None
    prompt_tokens: int | None
    total_tokens: int | None
    duration_seconds: float | None
    tool_calls: int | None
    tool_outcomes_known: int | None
    tool_errors: int | None
    error: str | None
    artifact_ref: str
    artifact_sha256: str
    sampling_config: dict[str, Any] | None
    turn_seed_receipts: tuple[int, ...] | None
    settings: dict[str, Any] | None
    attempt_exact: dict[str, dict[str, bool]] | None = None
    per_target_exact: dict[str, bool] | None = None
    competition_score: float | None = None

    def summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReceipt:
    request_id: str
    evalset_sha256: str
    policy: dict[str, Any]
    artifact_ref: str
    artifact_sha256: str
    summary: dict[str, Any]
    rows: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "qwen-rl-evaluation-receipt-v1",
            "request_id": self.request_id,
            "evalset_sha256": self.evalset_sha256,
            "policy": self.policy,
            "artifact_ref": self.artifact_ref,
            "artifact_sha256": self.artifact_sha256,
            "summary": self.summary,
            "rows": list(self.rows),
        }


def _known_exact(row: Mapping[str, Any]) -> bool | None:
    if row.get("error"):
        return None
    exact = row.get("exact")
    return exact if isinstance(exact, bool) else None


def compare_exact_rows(
    reference_rows: Mapping[str, Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    *,
    expected_task_count: int | None = None,
) -> dict[str, Any]:
    """Compare paired exact outcomes without promoting an incomplete subset to a full claim."""
    def prompt_identity(rows: Mapping[str, Mapping[str, Any]]) -> tuple[set[str], bool, bool]:
        fingerprints: set[str] = set()
        present = False
        complete = True
        for row in rows.values():
            if not isinstance(row, Mapping):
                complete = False
                continue
            settings = row.get("settings")
            if not isinstance(settings, Mapping) or "prompt_fingerprint" not in settings:
                complete = False
                continue
            present = True
            fingerprint = settings.get("prompt_fingerprint")
            if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                    or any(char not in "0123456789abcdef" for char in fingerprint)):
                complete = False
                continue
            fingerprints.add(fingerprint)
        return fingerprints, present, complete

    def unavailable_for_prompt(reason: str) -> dict[str, Any]:
        return {
            "comparison_available": False,
            "comparison_unavailable_reason": reason,
            "expected_task_count": expected_task_count if expected_task_count is not None else len(candidate_rows),
            "paired_task_count": 0,
            "paired_known_count": 0,
            "complete_paired_evaluation": False,
            "gain_count": None,
            "regression_count": None,
            "paired_gain_count": 0,
            "paired_regression_count": 0,
            "gains": [],
            "regressions": [],
            "unmatched_reference_tasks": [],
            "unmatched_candidate_tasks": [],
        }

    if reference_rows and candidate_rows:
        reference_fingerprints, reference_has_prompt, reference_complete = prompt_identity(reference_rows)
        candidate_fingerprints, candidate_has_prompt, candidate_complete = prompt_identity(candidate_rows)
        if reference_has_prompt or candidate_has_prompt:
            if not reference_has_prompt or not candidate_has_prompt:
                return unavailable_for_prompt("prompt_fingerprint_missing")
            if (not reference_complete or not candidate_complete
                    or not reference_fingerprints or not candidate_fingerprints):
                return unavailable_for_prompt("prompt_fingerprint_invalid_or_incomplete")
            if len(reference_fingerprints) != 1 or len(candidate_fingerprints) != 1:
                return unavailable_for_prompt("prompt_fingerprint_inconsistent_within_evaluation")
            if reference_fingerprints != candidate_fingerprints:
                return unavailable_for_prompt("prompt_fingerprint_mismatch")

    # New evaluations bind grammar as well as prompt text. Old result files are
    # historical evidence; a newly stamped run cannot be paired with an unstamped
    # or different parser and presented as a policy-only gain/regression.
    field = "action_language_fingerprint"
    sides = (reference_rows, candidate_rows)
    if any(field in (row.get("settings") or {}) for rows in sides for row in rows.values()):
        identities = []
        for rows in sides:
            settings = [row.get("settings") for row in rows.values()]
            values = [item.get(field) if isinstance(item, Mapping) else None for item in settings]
            if None in values or not values:
                return unavailable_for_prompt("action_language_fingerprint_missing")
            if any(not isinstance(v, str) or len(v) != 64
                   or any(c not in "0123456789abcdef" for c in v) for v in values):
                return unavailable_for_prompt("action_language_fingerprint_invalid")
            values = set(values)
            if len(values) != 1:
                return unavailable_for_prompt("action_language_fingerprint_inconsistent")
            identities.append(values)
        if identities[0] != identities[1]:
            return unavailable_for_prompt("action_language_fingerprint_mismatch")

    shared = sorted(set(reference_rows) & set(candidate_rows))
    gains: list[str] = []
    regressions: list[str] = []
    comparable = 0
    for task_id in shared:
        reference = _known_exact(reference_rows[task_id])
        candidate = _known_exact(candidate_rows[task_id])
        if reference is None or candidate is None:
            continue
        comparable += 1
        if candidate and not reference:
            gains.append(task_id)
        elif reference and not candidate:
            regressions.append(task_id)
    if expected_task_count is None:
        expected_task_count = max(len(reference_rows), len(candidate_rows))
    if isinstance(expected_task_count, bool) or not isinstance(expected_task_count, int) or expected_task_count < 0:
        raise ValueError("expected_task_count must be a nonnegative integer")
    complete = (
        comparable == expected_task_count
        and len(shared) == expected_task_count
        and len(reference_rows) == expected_task_count
        and len(candidate_rows) == expected_task_count
    )
    return {
        "comparison_available": True,
        "expected_task_count": expected_task_count,
        "paired_task_count": len(shared),
        "paired_known_count": comparable,
        "complete_paired_evaluation": complete,
        "gain_count": len(gains) if complete else None,
        "regression_count": len(regressions) if complete else None,
        "paired_gain_count": len(gains),
        "paired_regression_count": len(regressions),
        "gains": gains,
        "regressions": regressions,
        "unmatched_reference_tasks": sorted(set(reference_rows) - set(candidate_rows)),
        "unmatched_candidate_tasks": sorted(set(candidate_rows) - set(reference_rows)),
    }


def aggregate_episode_rows(
    rows: list[Mapping[str, Any]],
    *,
    baseline_rows: Mapping[str, Mapping[str, Any]] | None = None,
    previous_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    exact_values = [_known_exact(row) for row in rows]
    known = [value for value in exact_values if value is not None]
    generated = [row["generated_tokens"] for row in rows
                 if isinstance(row.get("generated_tokens"), int) and not isinstance(row.get("generated_tokens"), bool)]
    observed = [row["observation_tokens"] for row in rows
                if isinstance(row.get("observation_tokens"), int) and not isinstance(row.get("observation_tokens"), bool)]
    durations = [float(row["duration_seconds"]) for row in rows
                 if isinstance(row.get("duration_seconds"), (int, float))
                 and not isinstance(row.get("duration_seconds"), bool)
                 and math.isfinite(float(row["duration_seconds"])) and row["duration_seconds"] >= 0]
    errors = [row for row in rows if row.get("error")]
    known_tool_rows = [
        row for row in rows
        if (isinstance(row.get("tool_calls"), int) and not isinstance(row.get("tool_calls"), bool)
            and row["tool_calls"] >= 0
            and isinstance(row.get("tool_outcomes_known"), int)
            and not isinstance(row.get("tool_outcomes_known"), bool)
            and 0 <= row["tool_outcomes_known"] <= row["tool_calls"]
            and isinstance(row.get("tool_errors"), int) and not isinstance(row.get("tool_errors"), bool)
            and 0 <= row["tool_errors"] <= row["tool_outcomes_known"])
    ]
    tool_calls = sum(int(row["tool_calls"]) for row in known_tool_rows)
    tool_outcomes_known = sum(int(row["tool_outcomes_known"]) for row in known_tool_rows)
    tool_errors = sum(int(row["tool_errors"]) for row in known_tool_rows)
    unknown_tool_episodes = len(rows) - len(known_tool_rows)
    unknown_tool_calls = tool_calls - tool_outcomes_known
    known_tool_error_rate = tool_errors / tool_outcomes_known if tool_outcomes_known else None
    task_count = len(rows)
    solve_count = sum(known)
    exact_known_count = len(known)
    result: dict[str, Any] = {
        "task_count": task_count,
        "exact_known_count": exact_known_count,
        "exact_solve_count": solve_count,
        "exact_rate": (solve_count / task_count) if task_count and exact_known_count == task_count else None,
        "known_exact_rate": (solve_count / exact_known_count) if exact_known_count else None,
        "verified_solve_fraction": (solve_count / task_count) if task_count else None,
        "complete_valid_evaluation": bool(task_count) and exact_known_count == task_count and not errors,
        "unknown_exact_count": task_count - exact_known_count,
        "generated_tokens": sum(generated),
        "observation_tokens": sum(observed),
        "total_tokens": sum(generated) + sum(observed),
        "duration_seconds_total": sum(durations),
        "duration_seconds_mean": (sum(durations) / len(durations)) if durations else None,
        "tool_calls": tool_calls,
        "tool_outcomes_known": tool_outcomes_known,
        "tool_errors": tool_errors,
        "tool_error_rate": known_tool_error_rate if unknown_tool_episodes == 0 and unknown_tool_calls == 0 else None,
        "known_tool_error_rate": known_tool_error_rate,
        "tool_error_rate_known_calls": tool_outcomes_known,
        "tool_error_unknown_calls": unknown_tool_calls,
        "tool_error_unknown_episodes": unknown_tool_episodes,
        "error_count": len(errors),
        "errors": [{"task_id": row.get("task_id"), "error": row.get("error")} for row in errors],
    }
    target_rows = [row.get("per_target_exact") for row in rows]
    complete_targets = bool(rows) and all(
        not row.get("error") and isinstance(targets, dict) and bool(targets)
        and all(type(value) is bool for value in targets.values())
        for row, targets in zip(rows, target_rows)
    )
    known_targets = [value for row, targets in zip(rows, target_rows)
                     if not row.get("error") and isinstance(targets, dict)
                     for value in targets.values() if type(value) is bool]
    result["competition_target_count_known"] = len(known_targets)
    result["competition_target_solve_count"] = sum(known_targets)
    result["competition_score"] = (sum(known_targets) / len(known_targets)
                                   if complete_targets and known_targets else None)
    result["known_competition_score"] = (sum(known_targets) / len(known_targets)
                                         if known_targets else None)
    attempt_rows = [row.get("attempt_exact") for row in rows]
    attempts_complete = complete_targets and all(
        isinstance(attempts, dict) and set(attempts) == set(targets)
        and all(isinstance(slots, dict) and set(slots) <= {"1", "2"}
                and all(type(value) is bool for value in slots.values())
                for slots in attempts.values())
        for attempts, targets in zip(attempt_rows, target_rows)
    )
    for slot in ("1", "2"):
        result[f"attempt_{slot}_target_exact_rate"] = (
            sum(slots.get(slot, False) for attempts in attempt_rows for slots in attempts.values())
            / len(known_targets) if attempts_complete else None
        )
    indexed = {str(row["task_id"]): row for row in rows if isinstance(row.get("task_id"), str)}
    unavailable = {
        "comparison_available": False,
        "expected_task_count": task_count,
        "paired_task_count": 0,
        "paired_known_count": 0,
        "complete_paired_evaluation": False,
        "gain_count": None,
        "regression_count": None,
        "paired_gain_count": 0,
        "paired_regression_count": 0,
        "gains": [],
        "regressions": [],
        "unmatched_reference_tasks": [],
        "unmatched_candidate_tasks": sorted(indexed),
    }
    result["vs_baseline"] = (
        compare_exact_rows(baseline_rows, indexed, expected_task_count=task_count)
        if baseline_rows is not None else dict(unavailable)
    )
    result["vs_previous"] = (
        compare_exact_rows(previous_rows, indexed, expected_task_count=task_count)
        if previous_rows is not None else dict(unavailable)
    )
    baseline = result["vs_baseline"]
    result["gains"] = baseline["gain_count"]
    result["regressions"] = baseline["regression_count"]
    result["paired_gains"] = baseline["paired_gain_count"]
    result["paired_regressions"] = baseline["paired_regression_count"]
    result["paired_known_count"] = baseline["paired_known_count"]
    result["whole_evaluation_comparison_complete"] = baseline["complete_paired_evaluation"]
    return result
