"""Sequential, one-trajectory-per-task orchestration for frozen ARC evaluations."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol

from ..task_source import TaskSpec
from ..observations import decode_observation
from .policy import EvaluationPolicy
from .results import (
    EpisodeExecution,
    EvaluationEpisodeResult,
    EvaluationReceipt,
    aggregate_episode_rows,
    write_json_atomic,
)


class FrozenEvaluationDataset(Protocol):
    tasks: tuple[TaskSpec, ...]
    manifest: dict[str, Any]

    def task_seed(self, task_id: str) -> int: ...


RunEpisode = Callable[..., Awaitable[EpisodeExecution]]
TaskStarted = Callable[[str, str], Any]
TaskResult = Callable[[EvaluationEpisodeResult], Any]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evaluation_concurrency(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= 64:
        raise ValueError("evaluation concurrency must be an integer between 1 and 64")
    return value


def _safe_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _request_tasks(request: Mapping[str, Any], dataset: FrozenEvaluationDataset) -> list[TaskSpec]:
    request_id = _safe_identifier(request.get("request_id"), "request_id")
    if request.get("status") != "running":
        raise ValueError("evaluation request must be claimed before running")
    manifest = dataset.manifest
    if not isinstance(manifest, dict):
        raise TypeError("evaluation dataset manifest must be a dictionary")
    evalset_sha = manifest.get("manifest_sha256")
    if request.get("evalset_sha256") != evalset_sha:
        raise ValueError("evaluation request evalset SHA differs from the loaded frozen taskset")
    tasks = request.get("task_ids")
    expected = [row.get("task_id") for row in manifest.get("tasks", [])
                if isinstance(row, dict)]
    if not isinstance(tasks, list) or tasks != expected:
        raise ValueError("evaluation request task order differs from the frozen taskset manifest")
    by_id = {task.task_id: task for task in dataset.tasks}
    if len(by_id) != len(dataset.tasks) or list(by_id) != expected:
        raise ValueError("loaded task specifications disagree with frozen manifest order")
    return [by_id[task_id] for task_id in tasks]


def _validate_request_policy(request: Mapping[str, Any], policy: EvaluationPolicy) -> None:
    if request.get("policy_version") != policy.version:
        raise ValueError("evaluation request policy version differs from its pinned policy")
    if request.get("adapter_sha256") != policy.adapter_sha256:
        raise ValueError("evaluation request adapter SHA differs from its pinned policy")
    if not isinstance(request.get("lineage_id"), str) or not request["lineage_id"]:
        raise ValueError("evaluation request lineage_id must be non-empty")
    kind = request.get("kind")
    if kind == "baseline" and policy.role != "baseline_sft":
        raise ValueError("baseline request requires the pinned SFT baseline policy")
    if kind == "milestone" and policy.role != "rl_checkpoint":
        raise ValueError("milestone request requires an RL checkpoint policy")
    if kind == "manual" and policy.role not in {"baseline_sft", "rl_checkpoint"}:
        raise ValueError("manual request requires its selected baseline or RL checkpoint policy")
    if kind not in {"baseline", "milestone", "manual"}:
        raise ValueError("evaluation request kind must be baseline, milestone, or manual")


async def _call_hook(hook: Callable[..., Any] | None, *args: Any) -> None:
    if hook is None:
        return
    result = hook(*args)
    if inspect.isawaitable(result):
        await result


def _attempt_id(request_id: str, task_id: str, ordinal: int) -> str:
    payload = f"qwen-rl-evaluation-attempt-v1:{request_id}:{task_id}:{ordinal}"
    return "eval-" + _sha256_text(payload)[:40]


def _artifact_path(output_dir: Path, request_id: str, task_id: str, ordinal: int) -> Path:
    request_key = _sha256_text(request_id)[:20]
    task_key = _sha256_text(task_id)[:24]
    return output_dir / "requests" / request_key / "episodes" / f"{task_key}-attempt-{ordinal:04d}.json"


def _task_public_row(dataset: FrozenEvaluationDataset, task_id: str) -> dict[str, Any]:
    row = next((item for item in dataset.manifest["tasks"] if item["task_id"] == task_id), None)
    if not isinstance(row, dict):
        raise ValueError(f"frozen taskset manifest has no public row for {task_id}")
    forbidden = {"target", "targets", "solution", "solutions", "target_sha256", "answer_sha256"}
    if forbidden.intersection(row):
        raise ValueError("frozen evaluation task manifest contains private target data")
    return dict(row)


def _attempt_ordinal(request: Mapping[str, Any], task_id: str) -> int:
    partial = request.get("partial_tasks", {})
    task = partial.get(task_id, {}) if isinstance(partial, dict) else {}
    attempts = task.get("attempts", []) if isinstance(task, dict) else []
    return len(attempts) + 1 if isinstance(attempts, list) else 1


def _validate_completed_summary(
    *,
    task_id: str,
    expected_seed: int,
    request: Mapping[str, Any],
    policy: EvaluationPolicy,
    summary: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if summary.get("task_id") != task_id or summary.get("seed") != expected_seed:
        raise ValueError("completed evaluation row task ID or seed differs from the frozen taskset")
    if (summary.get("policy_version") != policy.version
            or summary.get("adapter_sha256") != policy.adapter_sha256):
        raise ValueError("completed evaluation row policy differs from the pinned request policy")
    artifact_ref = summary.get("artifact_ref")
    artifact_sha = summary.get("artifact_sha256")
    if (not isinstance(artifact_ref, str) or not artifact_ref
            or not isinstance(artifact_sha, str) or len(artifact_sha) != 64
            or any(char not in "0123456789abcdef" for char in artifact_sha)):
        raise ValueError("completed evaluation row omitted its artifact reference or SHA")
    artifact_path = (output_dir / artifact_ref).resolve()
    if output_dir not in artifact_path.parents or not artifact_path.is_file():
        raise ValueError("completed evaluation artifact is missing or outside its request output root")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if digest != artifact_sha:
        raise ValueError("completed evaluation artifact SHA differs from its immutable result row")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("completed evaluation artifact is not valid JSON") from exc
    artifact_task = artifact.get("task") if isinstance(artifact, dict) else None
    artifact_policy = artifact.get("policy") if isinstance(artifact, dict) else None
    if (not isinstance(artifact, dict)
            or artifact.get("schema") != "qwen-rl-native-evaluation-episode-v1"
            or artifact.get("training_eligible") is not False
            or artifact.get("request_id") != request.get("request_id")
            or artifact.get("evalset_sha256") != request.get("evalset_sha256")
            or not isinstance(artifact_task, dict)
            or artifact_task.get("task_id") != task_id
            or artifact.get("seed") != expected_seed
            or not isinstance(artifact_policy, dict)
            or artifact_policy.get("version") != policy.version
            or artifact_policy.get("adapter_sha256") != policy.adapter_sha256):
        raise ValueError("completed evaluation artifact identity differs from its frozen request")
    return dict(summary)


def _known_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def tool_event_metrics(trajectory: Mapping[str, Any]) -> tuple[int, int, int]:
    """Count recorded toolkit outcomes using the current pinned observation wire."""
    events = trajectory.get("tool_events")
    if not isinstance(events, list):
        raise ValueError("trusted ARC trajectory must contain tool_events")
    calls = 0
    known_outcomes = 0
    errors = 0
    for event in events:
        if not isinstance(event, dict) or event.get("name") not in {"arc_action"}:
            continue
        calls += 1
        observation = event.get("observation")
        try:
            parsed = decode_observation(observation) if isinstance(observation, str) else observation
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        status = parsed.get("status")
        if status == "ok":
            known_outcomes += 1
        elif status == "error":
            known_outcomes += 1
            errors += 1
    return calls, known_outcomes, errors


def _tool_event_metrics(trajectory: Mapping[str, Any], reported_errors: int) -> tuple[int, int, int]:
    calls, known_outcomes, errors = tool_event_metrics(trajectory)
    if errors != reported_errors:
        raise ValueError("reported tool error count differs from the recorded toolkit responses")
    return calls, known_outcomes, errors


def _row_from_execution(
    *,
    task_id: str,
    attempt_id: str,
    seed: int,
    policy: EvaluationPolicy,
    execution: EpisodeExecution,
    artifact_ref: str,
    artifact_sha256: str,
) -> EvaluationEpisodeResult:
    execution.validate()
    trajectory = execution.trajectory
    exact = trajectory.get("exact")
    if not isinstance(exact, bool):
        raise ValueError("trusted ARC trajectory must contain a boolean exact result")
    reward_value = trajectory.get("reward")
    reward = None
    if (not isinstance(reward_value, bool) and isinstance(reward_value, (int, float))
            and math.isfinite(reward_value)):
        reward = float(reward_value)
    generated_tokens = _known_int(trajectory.get("generated_tokens"))
    observation_tokens = _known_int(trajectory.get("observation_tokens"))
    prompt_tokens = _known_int(trajectory.get("prompt_tokens"))
    tool_calls, tool_outcomes_known, tool_errors = _tool_event_metrics(
        trajectory, execution.tool_errors,
    )
    total_tokens = (
        generated_tokens + observation_tokens
        if generated_tokens is not None and observation_tokens is not None else None
    )
    return EvaluationEpisodeResult(
        task_id=task_id,
        attempt_id=attempt_id,
        seed=seed,
        policy_role=policy.role,
        policy_version=policy.version,
        adapter_sha256=policy.adapter_sha256,
        episode_id=execution.episode_id,
        group_id=execution.group_id,
        worker_id=execution.worker_id,
        exact=exact,
        passed=exact,
        reward=reward,
        generated_tokens=generated_tokens,
        observation_tokens=observation_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
        duration_seconds=float(execution.duration_seconds),
        tool_calls=tool_calls,
        tool_outcomes_known=tool_outcomes_known,
        tool_errors=tool_errors,
        error=None,
        artifact_ref=artifact_ref,
        artifact_sha256=artifact_sha256,
        sampling_config=dict(execution.sampling_config),
        turn_seed_receipts=execution.turn_seed_receipts,
        settings=dict(execution.settings),
        attempt_exact=trajectory.get("attempt_exact"),
        per_target_exact=trajectory.get("per_target_exact"),
        competition_score=trajectory.get("competition_score"),
    )


def _row_from_error(
    *,
    task_id: str,
    attempt_id: str,
    seed: int,
    policy: EvaluationPolicy,
    group_id: str,
    elapsed: float,
    error: BaseException,
    artifact_ref: str,
    artifact_sha256: str,
) -> EvaluationEpisodeResult:
    return EvaluationEpisodeResult(
        task_id=task_id,
        attempt_id=attempt_id,
        seed=seed,
        policy_role=policy.role,
        policy_version=policy.version,
        adapter_sha256=policy.adapter_sha256,
        episode_id=None,
        group_id=group_id,
        worker_id=None,
        exact=None,
        passed=None,
        reward=None,
        generated_tokens=None,
        observation_tokens=None,
        prompt_tokens=None,
        total_tokens=None,
        duration_seconds=elapsed,
        tool_calls=None,
        tool_outcomes_known=None,
        tool_errors=None,
        error=f"{type(error).__name__}: {str(error)[:500]}",
        artifact_ref=artifact_ref,
        artifact_sha256=artifact_sha256,
        sampling_config=None,
        turn_seed_receipts=None,
        settings=None,
    )


def derive_turn_seed(task_seed: int, turn_index: int) -> int:
    if (type(task_seed) is not int or not 0 <= task_seed <= (1 << 32) - 1
            or type(turn_index) is not int or turn_index < 0):
        raise ValueError("turn seed inputs are outside their integer contracts")
    payload = f"qwen-arc-eval-seed-v1:{task_seed}:{turn_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _validate_execution_evidence(execution: EpisodeExecution, task_seed: int) -> None:
    settings = execution.settings
    sampling = execution.sampling_config
    required_settings = {
        "enable_thinking", "tool_protocol", "tool_names", "temperature", "top_p", "top_k",
        "repetition_penalty", "turn_max_tokens", "max_turns", "response_length",
        "token_budget", "contract_fingerprint", "prompt_fingerprint",
    }
    if not required_settings <= set(settings):
        raise ValueError(f"evaluation execution omitted settings: {sorted(required_settings - set(settings))}")
    if settings["enable_thinking"] is not True:
        raise ValueError("native ARC evaluation requires Thinking=True")
    if settings["tool_protocol"] != "qwen3_coder":
        raise ValueError("native ARC evaluation requires the Qwen qwen3_coder tool protocol")
    tool_names = settings["tool_names"]
    if not isinstance(tool_names, (list, tuple, set)) or set(tool_names) != {"arc_action"}:
        raise ValueError("native ARC evaluation requires exactly arc_action")
    for name in ("turn_max_tokens", "max_turns", "response_length", "token_budget"):
        value = settings[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"evaluation setting {name} must be a positive integer")
    contract_fingerprint = settings["contract_fingerprint"]
    if (not isinstance(contract_fingerprint, str) or len(contract_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in contract_fingerprint)):
        raise ValueError("evaluation settings require the trusted ARC contract fingerprint")
    prompt_fingerprint = settings["prompt_fingerprint"]
    if (not isinstance(prompt_fingerprint, str) or len(prompt_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in prompt_fingerprint)):
        raise ValueError("evaluation settings require the trusted prompt fingerprint")
    for name in ("temperature", "top_p", "top_k", "repetition_penalty"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"evaluation setting {name} must be finite numeric")
        if sampling.get(name) != value:
            raise ValueError(f"Worker sampling receipt differs from evaluation setting {name}")
    distribution = settings.get("policy_distribution", "raw_policy")
    if distribution not in {"raw_policy", "grammar_conditioned_policy/v1"}:
        raise ValueError("unsupported evaluation policy distribution")
    if (sampling.get("distribution") != distribution
            or sampling.get("warpers_neutralized") is not True
            or sampling.get("seed_mode") != "request"):
        raise ValueError("Worker sampling receipt did not prove the configured seeded policy distribution")
    seeds = execution.turn_seed_receipts
    if not seeds:
        raise ValueError("evaluation execution omitted per-turn seed receipts")
    expected = tuple(derive_turn_seed(task_seed, index) for index in range(len(seeds)))
    if seeds != expected or sampling.get("seed") != seeds[-1]:
        raise ValueError("Worker per-turn seed receipts differ from the frozen task seed")
    segments = execution.trajectory.get("constraint_segments")
    if distribution == "raw_policy":
        if segments:
            raise ValueError("raw-policy evaluation contains conditioned generation segments")
        return
    if sampling.get("behavior_distribution") != distribution:
        raise ValueError("Worker behavior distribution differs from the configured grammar policy")
    for name in ("grammar_sha256", "grammar_compiler_sha256"):
        value = settings.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                or sampling.get(name) != value):
            raise ValueError(f"Worker grammar receipt differs from evaluation setting {name}")
    public = sampling.get("public_token_count")
    logical = sampling.get("logical_model_vocab_size")
    if type(public) is not int or type(logical) is not int or not 0 < public <= logical:
        raise ValueError("Worker grammar receipt has an invalid vocabulary domain")
    if not isinstance(segments, list) or len(segments) != len(seeds):
        raise ValueError("evaluation requires a grammar segment for every seeded generation")
    for segment in segments:
        if (not isinstance(segment, dict)
                or segment.get("grammar_sha256") != settings["grammar_sha256"]
                or segment.get("compiler_sha256") != settings["grammar_compiler_sha256"]
                or segment.get("public_token_count") != public):
            raise ValueError("evaluation generation segment differs from the attested grammar policy")


def _index_completed_rows(
    comparison_rows: Mapping[str, Mapping[str, Mapping[str, Any]]] | None,
    request_id: str | None,
) -> Mapping[str, Mapping[str, Any]] | None:
    if request_id is None:
        return None
    if comparison_rows is None or request_id not in comparison_rows:
        raise ValueError(f"completed comparison rows are required for request {request_id}")
    return comparison_rows[request_id]


async def run_evaluation_request(
    request: Mapping[str, Any],
    dataset: FrozenEvaluationDataset,
    policy: EvaluationPolicy,
    run_episode: RunEpisode,
    output_dir: str | Path,
    *,
    comparison_rows: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    on_task_started: TaskStarted | None = None,
    on_task_result: TaskResult | None = None,
    concurrency: int = 1,
) -> EvaluationReceipt:
    """Run one claimed policy over a frozen taskset with bounded episode concurrency.

    The caller owns Worker policy pinning and the production ARC callback. This runner never
    imports training data, requests a Teacher, populates replay/cache, or returns trajectories
    to an optimizer. It writes every completed episode as an independent hashed artifact.
    """
    concurrency = _evaluation_concurrency(concurrency)
    tasks = _request_tasks(request, dataset)
    _validate_request_policy(request, policy)
    if dataset.manifest.get("base_fingerprint") != policy.base_fingerprint:
        raise ValueError("evaluation taskset was frozen against a different base fingerprint")
    request_id = str(request["request_id"])
    output_dir = Path(output_dir).resolve()
    completed = request.get("completed_tasks", {})
    if not isinstance(completed, dict):
        raise ValueError("evaluation request completed_tasks must be an object")
    comparison = request.get("comparison", {})
    if not isinstance(comparison, dict):
        raise ValueError("evaluation request comparison must be an object")
    baseline_rows = _index_completed_rows(comparison_rows, comparison.get("baseline_request_id"))
    previous_rows = _index_completed_rows(comparison_rows, comparison.get("previous_request_id"))
    expected_task_ids = {task.task_id for task in tasks}
    for label, reference_rows in (("baseline", baseline_rows), ("previous", previous_rows)):
        if reference_rows is not None and set(reference_rows) != expected_task_ids:
            raise ValueError(f"{label} comparison rows do not cover the exact frozen evaluation taskset")
    rows_by_task: dict[str, dict[str, Any]] = {}
    pending: list[tuple[TaskSpec, int]] = []
    for task in tasks:
        seed = dataset.task_seed(task.task_id)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (1 << 32) - 1:
            raise ValueError("frozen task seed must be an integer between 0 and 2**32-1")
        prior = completed.get(task.task_id)
        if prior is not None:
            if not isinstance(prior, dict) or prior.get("status") != "completed":
                raise ValueError("completed task entry has an invalid state")
            summary = prior.get("summary")
            if not isinstance(summary, dict) or summary.get("task_id") != task.task_id:
                raise ValueError("completed task entry omitted its immutable summary")
            rows_by_task[task.task_id] = _validate_completed_summary(
                task_id=task.task_id,
                expected_seed=seed,
                request=request,
                policy=policy,
                summary=summary,
                output_dir=output_dir,
            )
        else:
            pending.append((task, seed))

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(task: TaskSpec, seed: int) -> None:
        async with semaphore:
            ordinal = _attempt_ordinal(request, task.task_id)
            attempt_id = _attempt_id(request_id, task.task_id, ordinal)
            artifact_path = _artifact_path(output_dir, request_id, task.task_id, ordinal)
            artifact_ref = str(artifact_path.relative_to(output_dir))
            await _call_hook(on_task_started, task.task_id, attempt_id)

            started = time.perf_counter()
            execution: EpisodeExecution | None = None
            try:
                execution = await run_episode(
                    task,
                    policy,
                    seed=seed,
                    request_id=request_id,
                    attempt_id=attempt_id,
                )
                if not isinstance(execution, EpisodeExecution):
                    raise TypeError("run_episode must return EpisodeExecution")
                execution.validate()
                _validate_execution_evidence(execution, seed)
                row = _row_from_execution(
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    seed=seed,
                    policy=policy,
                    execution=execution,
                    artifact_ref=artifact_ref,
                    artifact_sha256="0" * 64,
                )
                payload = {
                    "schema": "qwen-rl-native-evaluation-episode-v1",
                    "training_eligible": False,
                    "request_id": request_id,
                    "attempt_id": attempt_id,
                    "evalset_sha256": request["evalset_sha256"],
                    "task": _task_public_row(dataset, task.task_id),
                    "policy": policy.as_dict(),
                    "worker_id": execution.worker_id,
                    "worker_lease_id": execution.worker_lease_id,
                    "episode_id": execution.episode_id,
                    "group_id": execution.group_id,
                    "seed": seed,
                    "duration_seconds": float(execution.duration_seconds),
                    "tool_calls": row.tool_calls,
                    "tool_outcomes_known": row.tool_outcomes_known,
                    "tool_errors": row.tool_errors,
                    "sampling_config": execution.sampling_config,
                    "turn_seed_receipts": list(execution.turn_seed_receipts),
                    "settings": execution.settings,
                    "trajectory": execution.trajectory,
                }
                artifact_sha = write_json_atomic(artifact_path, payload)
                row = replace(row, artifact_sha256=artifact_sha)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                elapsed = time.perf_counter() - started
                group_id = attempt_id
                error_payload = {
                    "schema": "qwen-rl-native-evaluation-episode-v1",
                    "training_eligible": False,
                    "request_id": request_id,
                    "attempt_id": attempt_id,
                    "evalset_sha256": request["evalset_sha256"],
                    "task": _task_public_row(dataset, task.task_id),
                    "policy": policy.as_dict(),
                    "episode_id": None,
                    "group_id": group_id,
                    "seed": seed,
                    "duration_seconds": elapsed,
                    "error": {"type": type(exc).__name__, "message": str(exc)[:500]},
                }
                if isinstance(execution, EpisodeExecution):
                    error_payload.update({
                        "worker_id": execution.worker_id,
                        "worker_lease_id": execution.worker_lease_id,
                        "episode_id": execution.episode_id,
                        "group_id": execution.group_id,
                        "tool_errors": execution.tool_errors,
                        "sampling_config": execution.sampling_config,
                        "turn_seed_receipts": list(execution.turn_seed_receipts),
                        "settings": execution.settings,
                        "trajectory": execution.trajectory,
                    })
                artifact_sha = write_json_atomic(artifact_path, error_payload)
                row = _row_from_error(
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    seed=seed,
                    policy=policy,
                    group_id=group_id,
                    elapsed=elapsed,
                    error=exc,
                    artifact_ref=artifact_ref,
                    artifact_sha256=artifact_sha,
                )
            rows_by_task[task.task_id] = row.summary()
            await _call_hook(on_task_result, row)

    async with asyncio.TaskGroup() as task_group:
        for task, seed in pending:
            task_group.create_task(run_one(task, seed))

    ordered_rows = [rows_by_task[task.task_id] for task in tasks]
    summary = aggregate_episode_rows(
        ordered_rows,
        baseline_rows=baseline_rows,
        previous_rows=previous_rows,
    )
    summary["native_exact"] = {
        "count": summary["exact_solve_count"], "total": summary["exact_known_count"],
    }
    summary["native_pass"] = dict(summary["native_exact"])
    summary["gains"] = summary.get("vs_baseline", {}).get("gain_count", 0)
    summary["regressions"] = summary.get("vs_baseline", {}).get("regression_count", 0)
    summary["trajectory_count_per_task"] = 1
    summary["concurrency"] = concurrency
    request_key = _sha256_text(request_id)[:20]
    receipt_path = output_dir / "requests" / request_key / f"result-attempt-{int(request.get('retry_count', 0)):04d}.json"
    receipt_payload = {
        "schema": "qwen-rl-native-evaluation-result-v1",
        "request_id": request_id,
        "kind": request.get("kind"),
        "successful_update_count": request.get("successful_update_count"),
        "milestone_update_count": request.get("milestone_update_count"),
        "lineage_id": request.get("lineage_id"),
        "evalset_sha256": request["evalset_sha256"],
        "task_ids": [task.task_id for task in tasks],
        "comparison": comparison,
        "policy": policy.as_dict(),
        "summary": summary,
        "rows": ordered_rows,
    }
    result_sha = write_json_atomic(receipt_path, receipt_payload)
    return EvaluationReceipt(
        request_id=request_id,
        evalset_sha256=request["evalset_sha256"],
        policy=policy.as_dict(),
        artifact_ref=str(receipt_path.relative_to(output_dir)),
        artifact_sha256=result_sha,
        summary=summary,
        rows=tuple(ordered_rows),
    )
