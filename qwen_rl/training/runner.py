"""High-level local runner composed from the shared training coordinator."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import importlib
import inspect
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

from qwen_rl.contracts.environment import Reward, ValidatedEnvironment, ValidatedRewardScorer
from qwen_rl.learner.compute import ComputeOwner
from qwen_rl.learner.monitoring import RunCommitView
from qwen_rl.learner.run.bepa import BepaSeedService
from qwen_rl.learner.run.coordinator import RunCoordinator, RunCoordinatorServices
from qwen_rl.learner.run.evaluation import EvaluationCoordinator
from qwen_rl.learner.run.state import RunState, json_safe
from qwen_rl.learner.telemetry import TrainingTelemetry
from qwen_rl.rollout.pipeline import AsyncGroupStream
from qwen_rl.runtime.manifest import read_json, write_json_atomic

from .tasks import RewardFn, Task, load_tasks_jsonl, normalized_tasks_jsonl, tasks_sha256


_CONFIG_SCHEMA = "qwen-rl-local-training-config/v1"
_RUN_SCHEMA = "qwen-rl-local-training-run/v1"
_IDENTITY_SCHEMA = "qwen-rl-local-training-identity/v1"
_SAMPLING_SCHEDULE = "sequential-local/v1"
_LOCAL_WORKER_ID = "local-shared-engine"
_REQUIRED_CONFIG = frozenset({
    "model_id", "model_revision", "device", "dtype", "lora_rank", "lora_alpha",
    "lora_targets", "group_size", "learning_rate", "seed", "epsilon", "clip_ratio",
    "importance_cap", "num_threads", "max_prompt_tokens", "max_filtered_groups",
    "choices", "data_path", "reward_fn", "reward_schema", "source_identity",
})
_ENGINE_EVIDENCE_FIELDS = frozenset({
    "model_id", "model_revision", "source_identity_mode", "revision_is_immutable_hub_commit", "tokenizer_id",
    "tokenizer_revision", "device", "dtype", "attention_implementation", "optimizer",
    "scheduler", "lora_rank", "lora_alpha", "lora_targets", "lora_dropout", "choices",
    "choice_token_ids", "epsilon", "behavior_probability", "loss",
    "importance_correction_cap", "policy_identity_kind", "publication_scope",
    "checkpoint_schema", "base_source_proof", "base_identity", "adapter_structure",
    "adapter_weights_finite",
})


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("config must be a mapping")
    missing = _REQUIRED_CONFIG - set(value)
    if missing:
        raise ValueError(f"training config is missing keys: {sorted(missing)}")
    config = dict(value)
    for key in ("model_id", "model_revision", "device", "dtype", "reward_fn",
                "reward_schema", "source_identity"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"config[{key!r}] must be a non-empty string")
    revision = config["model_revision"]
    model_id = config["model_id"]
    if Path(model_id).is_absolute():
        if not Path(model_id).is_dir():
            raise ValueError("absolute model_id must identify a local model directory")
    elif re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Hub model_revision must be an immutable 40-character commit")
    data_path = config["data_path"]
    if not isinstance(data_path, str) or not Path(data_path).is_absolute():
        raise ValueError("data_path must be an absolute path")
    choices = config["choices"]
    if (not isinstance(choices, (list, tuple)) or len(choices) != 2
            or any(not isinstance(choice, str) or not choice for choice in choices)):
        raise ValueError("choices must contain exactly two non-empty strings")
    try:
        # Normalize tuples to JSON arrays and reject hidden objects/non-finite numbers
        # at this one public configuration boundary.
        return json.loads(_canonical_json(config))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config must contain only finite JSON values: {exc}") from exc


def _reward_spec(function: RewardFn) -> str:
    module_name = getattr(function, "__module__", None)
    function_name = getattr(function, "__name__", None)
    qualname = getattr(function, "__qualname__", None)
    if (not callable(function) or not isinstance(module_name, str) or not module_name
            or not isinstance(function_name, str) or not function_name
            or qualname != function_name):
        raise ValueError("reward_fn must be an importable module-level function")
    return f"{module_name}:{function_name}"


def _load_reward(spec: str) -> tuple[RewardFn, str]:
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("reward_fn must use the 'module:function' form")
    module_name, function_name = spec.split(":", 1)
    if not module_name or not function_name or function_name.startswith("_"):
        raise ValueError("reward_fn must name a public module-level function")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function) or _reward_spec(function) != spec:
        raise TypeError(f"reward_fn is not an importable module-level function: {spec}")
    try:
        inspect.signature(function).bind(Task("probe", "prompt", "answer"), "completion")
    except (TypeError, ValueError) as exc:
        raise TypeError("reward_fn must accept (Task, completion: str)") from exc
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str) or not module_path.endswith(".py"):
        raise ValueError("reward_fn module must have a Python source file for identity binding")
    source_sha256 = _sha256(Path(module_path).resolve().read_bytes())
    return function, source_sha256


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_inputs(
    config: dict[str, Any] | None,
    run_dir: Path,
    *,
    resume: bool,
    reward_fn: RewardFn | None,
) -> tuple[dict[str, Any], tuple[Task, ...], RewardFn, dict[str, Any], dict[str, Any] | None]:
    config_path = run_dir / "config.json"
    tasks_path = run_dir / "input" / "tasks.jsonl"
    stored: dict[str, Any] | None = None
    supplied_config = config is not None
    if resume:
        if not config_path.is_file() or config_path.is_symlink():
            raise FileNotFoundError("resume requires the original run config.json")
        if not tasks_path.is_file() or tasks_path.is_symlink():
            raise FileNotFoundError("resume requires the frozen input/tasks.jsonl")
        stored = read_json(config_path)
        if stored.get("schema") != _CONFIG_SCHEMA or not isinstance(stored.get("config"), dict):
            raise ValueError("unsupported or incomplete persisted training config")
        persisted_config = _normalize_config(stored["config"])
        if config is not None:
            candidate_config = _normalize_config(config)
            if reward_fn is not None:
                candidate_spec = _reward_spec(reward_fn)
                if candidate_config["reward_fn"] != candidate_spec:
                    raise ValueError("provided reward_fn differs from config.reward_fn")
            if candidate_config != persisted_config:
                raise ValueError("resume config differs from the persisted training config")
        config = persisted_config
        if reward_fn is not None and _reward_spec(reward_fn) != config["reward_fn"]:
            raise ValueError("provided reward_fn differs from the persisted training config")
        tasks = load_tasks_jsonl(tasks_path)
        frozen = tasks_path.read_bytes()
        canonical = normalized_tasks_jsonl(tasks)
        if frozen != canonical:
            raise ValueError("frozen task snapshot is not canonical JSONL")
        if supplied_config:
            source_path = Path(config["data_path"])
            if not source_path.is_file():
                raise FileNotFoundError(f"resume task source is missing: {source_path}")
            if normalized_tasks_jsonl(load_tasks_jsonl(source_path)) != frozen:
                raise ValueError("current task source differs from the frozen run task snapshot")
    else:
        if config is None:
            raise ValueError("a new run requires a config")
        config = _normalize_config(config)
        if reward_fn is not None:
            if config["reward_fn"] != _reward_spec(reward_fn):
                raise ValueError("provided reward_fn differs from config.reward_fn")
        tasks = load_tasks_jsonl(config["data_path"])
        _atomic_write_bytes(tasks_path, normalized_tasks_jsonl(tasks))

    assert config is not None
    selected_train = tuple(task for task in tasks if task.split == "train" and task.pool != "diagnostic")
    if not selected_train:
        raise ValueError("task source contains no train tasks")
    function, reward_source_sha256 = _load_reward(config["reward_fn"])
    if reward_fn is not None and function is not reward_fn:
        raise ValueError("provided reward_fn object does not match its configured module function")
    identity = {
        "schema": _IDENTITY_SCHEMA,
        "config_sha256": _sha256(_canonical_json(config)),
        "task_sha256": tasks_sha256(tasks),
        "ordered_task_ids_sha256": _sha256(_canonical_json([task.task_id for task in selected_train])),
        "source_identity": config["source_identity"],
        "reward_schema": config["reward_schema"],
        "reward_fn": config["reward_fn"],
        "reward_module_sha256": reward_source_sha256,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "sampling_schedule": _SAMPLING_SCHEDULE,
    }
    if resume:
        if stored is None or stored.get("identity", {}).get("schema") != _IDENTITY_SCHEMA:
            raise ValueError("resume config is missing its source-bound identity")
        old_identity = stored["identity"]
        for key, value in identity.items():
            if old_identity.get(key) != value:
                raise ValueError(f"resume identity mismatch for {key}")
    return config, tasks, function, identity, stored


class _LocalEnvironment:
    name = "local-hf-completion"
    metrics_key = "local_training"
    arc_sampling = False

    def __init__(
        self,
        tasks: Sequence[Task],
        *,
        group_size: int,
        compute_owner: ComputeOwner,
        reward_schema: str,
        source_identity: str,
        prompt_identity: dict[str, Any],
    ) -> None:
        self.tasks = {task.task_id: task for task in tasks}
        self.group_size = group_size
        self.compute_owner = compute_owner
        self.reward_schema = reward_schema
        self.source_identity = source_identity
        self.prompt_identity = prompt_identity
        self._fixture = Task(
            "local-diagnostic-fixture", "Answer yes or no.", "yes",
            split="diagnostic", pool="diagnostic",
        )

    def fixture_task(self) -> Task:
        return self._fixture

    async def generate_group(
        self,
        engine: Any,
        managers: dict[str, Any],
        config: dict[str, Any],
        task: Task,
        group_id: str,
        policy: Any,
        *,
        worker_pool: Any,
        diagnostic_course: str | None = None,
        owner_worker_id: str | None = None,
        guided_plan: str | None = None,
    ) -> dict[str, Any]:
        del managers, config, worker_pool, diagnostic_course, owner_worker_id, guided_plan
        started = time.perf_counter()
        rows = await self.compute_owner.run(
            engine.sample_group, task.prompt, self.group_size, policy.version,
        )
        if not isinstance(rows, list) or len(rows) != self.group_size:
            raise ValueError("local engine returned an incomplete rollout group")
        trajectories: list[dict[str, Any]] = []
        for row in rows:
            if (not isinstance(row, dict) or type(row.get("policy_version")) is not int
                    or row["policy_version"] != policy.version):
                raise ValueError("local engine rollout policy identity differs from its pinned snapshot")
            if not isinstance(row.get("completion"), str):
                raise ValueError("local engine rollout omitted its completion text")
            trajectories.append({**row, "group_id": group_id, "policy_version": policy.version})
        elapsed = time.perf_counter() - started
        return {
            "trajectories": trajectories,
            "worker_ids": [_LOCAL_WORKER_ID] * len(trajectories),
            "policy_version": policy.version,
            "metrics": {
                "episode_count": len(trajectories),
                "worker_generation": {
                    _LOCAL_WORKER_ID: {"generation_calls": len(trajectories),
                                       "generation_seconds": elapsed},
                },
            },
        }


class _LocalRewardScorer:
    def __init__(self, function: RewardFn) -> None:
        self.function = function

    def score(self, task: Task, trajectories: Sequence[dict[str, Any]]) -> Sequence[Reward]:
        # score_group owns the one validation of user reward return values.
        return [Reward(self.function(task, trajectory["completion"])) for trajectory in trajectories]


def _policy_record(engine: Any) -> dict[str, Any]:
    publication = engine.publish()
    if not isinstance(publication, dict):
        raise TypeError("HFEngine.publish() must return a policy mapping")
    if type(publication.get("version")) is not int or publication["version"] != engine.policy_version:
        raise ValueError("published policy version differs from the engine")
    if publication.get("evidence", {}).get("publication_scope") != "local_only":
        raise ValueError("local runner requires an explicitly local-only policy publication")
    if publication.get("evidence", {}).get("identity_kind") != "metadata_id_not_content_sha256":
        raise ValueError("policy identity must disclose that it is a metadata ID, not a content hash")
    for key in ("source_adapter_sha256", "adapter_hash"):
        digest = publication.get(key)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"published policy {key} must be a 64-character metadata ID")
    for key in ("base_fingerprint", "artifact_fingerprint"):
        if not isinstance(publication.get(key), str) or not publication[key]:
            raise ValueError(f"published policy {key} must be non-empty")
    return publication


def _stable_engine_evidence(engine: Any, config: dict[str, Any]) -> dict[str, Any]:
    evidence = getattr(engine, "evidence", None)
    if not isinstance(evidence, dict):
        raise TypeError("HFEngine.evidence must be a JSON-safe identity mapping")
    missing = _ENGINE_EVIDENCE_FIELDS - set(evidence)
    if missing:
        raise ValueError(f"HFEngine evidence is missing identity fields: {sorted(missing)}")
    if evidence["model_id"] != config["model_id"] or evidence["model_revision"] != config["model_revision"]:
        raise ValueError("HFEngine loaded a different model ID or revision")
    if Path(config["model_id"]).is_absolute():
        if evidence["source_identity_mode"] != "operator_declared_local_revision":
            raise ValueError("local model source must disclose its operator-declared revision")
    elif (evidence["source_identity_mode"] != "immutable_hub_commit"
          or evidence["revision_is_immutable_hub_commit"] is not True):
        raise ValueError("HFEngine did not attest an immutable Hub commit revision")
    if evidence["tokenizer_revision"] != config["model_revision"]:
        raise ValueError("tokenizer revision differs from the pinned model revision")
    if evidence["adapter_weights_finite"] is not True:
        raise ValueError("HFEngine adapter parameter identity is not finite")
    if evidence["publication_scope"] != "local_only":
        raise ValueError("HFEngine policy publication must be local-only")
    if evidence["policy_identity_kind"] != "lineage_version_metadata_id_not_content_sha256":
        raise ValueError("HFEngine policy identity must disclose metadata-only identity")
    return json.loads(_canonical_json(evidence))


def _trajectory_persistor(run_dir: Path):
    def persist_group(
        _group_run_dir: Path,
        group_id: str,
        trajectories: list[dict[str, Any]],
        worker_ids: list[str],
        *,
        phase: str,
        policy_version: int,
        adapter_sha256: str,
        base_fingerprint: str,
    ) -> str:
        payload = {
            "schema": "qwen-rl-local-trajectory-group/v1",
            "group_id": group_id,
            "phase": phase,
            "policy_version": policy_version,
            "policy_identity": adapter_sha256,
            "policy_identity_kind": "metadata_id_not_content_sha256",
            "base_fingerprint": base_fingerprint,
            "worker_ids": worker_ids,
            "trajectories": json_safe(trajectories),
        }
        raw = _canonical_json(payload) + b"\n"
        artifact = run_dir / "trajectories" / f"{group_id}.json"
        if artifact.exists():
            if artifact.read_bytes() != raw:
                raise FileExistsError(f"trajectory group ID already has different content: {group_id}")
        else:
            _atomic_write_bytes(artifact, raw)
        return f"{artifact.relative_to(run_dir)}#sha256={_sha256(raw)}"

    return persist_group


def _new_run_state(
    *,
    run_id: str,
    engine: Any,
    target_steps: int,
    prior: dict[str, Any] | None,
    segment: int,
    parent_run_id: str | None,
) -> RunState:
    now = time.time()
    policy = _policy_record(engine)
    if prior is None:
        values: dict[str, Any] = {
            "run_id": run_id,
            "kind": "training",
            "status": "running",
            "failure_stage": "queued",
            "started_at": now,
            "initial_policy_version": int(engine.policy_version),
            "initial_adapter_hash": policy["adapter_hash"],
            "requested_steps": target_steps,
            "completed_updates": int(engine.policy_version),
            "resume_task_cursor": {"next_task_index": 0, "next_group_index": 0},
            "filtered_groups": 0,
            "groups": [],
            "cohorts": [],
            "policy_handoffs": [],
            "diagnostic_course": None,
            "metrics": {"learner": {
                "model_load_seconds": 0.0,
                "initial_publish_seconds": 0.0,
                "update_seconds": 0.0,
                "publish_seconds": 0.0,
                "worker_sync_seconds": 0.0,
            }},
            "summary": None,
            "error": None,
            "segment": segment,
            "parent_run_id": parent_run_id,
        }
    else:
        values = dict(prior)
        old_status = values.get("status")
        if old_status in {"succeeded", "failed", "filtered_limit", "cancelled", "paused"}:
            values.setdefault("resume_attempts", []).append({
                "status": old_status,
                "finished_at": values.get("finished_at"),
                "error": values.get("error"),
                "resumed_at": now,
            })
        values.update(
            status="running",
            failure_stage="resume",
            requested_steps=target_steps,
            completed_updates=int(engine.policy_version),
            error=None,
            summary=None,
            segment=segment,
            parent_run_id=parent_run_id,
        )
        for key in ("finished_at", "elapsed_seconds", "committed_policy_version"):
            values.pop(key, None)
    return RunState(run_id, values)


def _build_services(
    *,
    config: dict[str, Any],
    run_dir: Path,
    engine: Any,
    tasks: Sequence[Task],
    reward_fn: RewardFn,
    identity: dict[str, Any],
    target_steps: int,
    parent_run_id: str | None,
    segment: int,
    cursor_base: dict[str, int],
    dispatch_count_base: int,
) -> tuple[RunCoordinator, Any]:
    compute_owner = ComputeOwner()
    environment = _LocalEnvironment(
        tasks,
        group_size=int(config["group_size"]),
        compute_owner=compute_owner,
        reward_schema=config["reward_schema"],
        source_identity=config["source_identity"],
        prompt_identity={
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "tokenizer_revision": config["model_revision"],
            "choices": config["choices"],
            "max_prompt_tokens": config["max_prompt_tokens"],
        },
    )
    scorer = _LocalRewardScorer(reward_fn)
    # The JSONL loader and Task constructor already checked these values at the
    # external boundary; pass the typed views so the core does not revalidate them.
    validated_environment = ValidatedEnvironment(
        adapter=environment,
        tasks=MappingProxyType(dict(environment.tasks)),
        fixture=environment.fixture_task(),
        name=environment.name,
        prompt_identity=environment.prompt_identity,
        reward_schema=environment.reward_schema,
        metrics_key=environment.metrics_key,
        source_identity=environment.source_identity,
    )
    validated_scorer = ValidatedRewardScorer(adapter=scorer)
    policy = _policy_record(engine)
    telemetry = TrainingTelemetry()
    runtime = SimpleNamespace(
        runs={},
        run_commit_view=RunCommitView(),
        training_telemetry=telemetry,
        policies={int(engine.policy_version): policy},
        policy_refs={},
        timings={"learner": {"update_seconds": 0.0, "publish_seconds": 0.0,
                              "worker_sync_seconds": 0.0}},
        external_stage_timings={},
        active_run_id=None,
        active_sampling_groups=0,
        last_policy_acks={},
        serve_policy_version=None,
        seed_scheduler=None,
        rollout_stream=None,
    )
    train_task_ids = [task.task_id for task in tasks if task.split == "train" and task.pool != "diagnostic"]
    core_config = {
        "run_dir": str(run_dir),
        "workers": [{"id": _LOCAL_WORKER_ID, "scope": "local_shared_engine"}],
        "learner": {
            "update_backend": "trainer",
            "parallelism": "single",
            "groups_per_update": 1,
            "learning_rate": float(config["learning_rate"]),
        },
        "rollout": {
            "group_size": int(config["group_size"]),
            "concurrent_tasks": 1,
            "ready_queue_batches": 1,
            "max_filtered_groups": int(config["max_filtered_groups"]),
            "task_ids": train_task_ids,
            "category_sampling": False,
            "sequential_updates": True,
        },
        "dapo": {
            "updates": target_steps,
            "max_filtered_groups": int(config["max_filtered_groups"]),
        },
        "evaluation": {"enabled": False, "pause_training": False, "every_updates": 0},
    }

    async def sync_workers(publication: dict[str, Any]) -> dict[str, dict[str, Any]]:
        version = publication.get("version")
        if type(version) is not int or version != int(engine.policy_version):
            raise RuntimeError("local policy ACK cannot acknowledge a stale engine version")
        evidence = publication.get("evidence", {})
        if evidence.get("publication_scope") != "local_only":
            raise RuntimeError("local worker ACK requires a local-only policy publication")
        current = _policy_record(engine)
        if any(current.get(key) != publication.get(key) for key in (
            "version", "source_adapter_sha256", "base_fingerprint", "artifact_fingerprint",
        )):
            raise RuntimeError("local shared engine does not hold the acknowledged publication")
        acknowledgements = {
            _LOCAL_WORKER_ID: {
                "version": version,
                "status": "acknowledged",
                "scope": "local_shared_engine",
                "publication_scope": "local_only",
                "transport": "same_process_engine_reference",
                "remote_worker_qualification": False,
            },
        }
        runtime.last_policy_acks = acknowledgements
        return acknowledgements

    async def assert_workers_current(version: int) -> None:
        current = _policy_record(engine)
        if int(engine.policy_version) != version or current["version"] != version:
            raise RuntimeError("local shared-engine policy changed before its version fence")
        acknowledgement = runtime.last_policy_acks.get(_LOCAL_WORKER_ID)
        if not isinstance(acknowledgement, dict) or acknowledgement.get("version") != version:
            raise RuntimeError("local shared-engine policy has not acknowledged the requested version")

    async def publish_current(
        already_published: dict[str, Any] | None = None,
        *,
        pending_checkpoint: bool = False,
    ) -> dict[str, Any]:
        del pending_checkpoint
        current = _policy_record(engine)
        if already_published is not None:
            for key in ("version", "adapter_hash", "source_adapter_sha256", "base_fingerprint"):
                if current.get(key) != already_published.get(key):
                    raise RuntimeError(f"update publication differs from engine policy field {key}")
        runtime.policies[int(current["version"])] = current
        return current

    async def unused_generate_episode(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("local user-training runner does not enable BEPA generation")

    async def unused_build_prefix(*_args: Any, **_kwargs: Any) -> list[int]:
        raise RuntimeError("local user-training runner does not enable BEPA generation")

    seed_service = BepaSeedService(
        bepa=None,
        engine=engine,
        policies=runtime.policies,
        policy_refs=runtime.policy_refs,
        generate_episode=unused_generate_episode,
        build_prefix=unused_build_prefix,
        sync_workers=sync_workers,
    )
    evaluation = EvaluationCoordinator(
        settings={"enabled": False},
        dataset=None,
        engine=engine,
        policies=runtime.policies,
        policy_refs=runtime.policy_refs,
        current_policy=lambda: runtime.policies[int(engine.policy_version)],
        recovery_metadata=lambda: engine.recovery_metadata,
        assert_workers_current=assert_workers_current,
        run_episode=unused_generate_episode,
    )

    def recovery_context(**kwargs: Any) -> dict[str, Any]:
        dispatches = kwargs["state"].get("stream_dispatches", [])
        if not isinstance(dispatches, list) or len(dispatches) < dispatch_count_base:
            raise RuntimeError("run dispatch history moved behind its resume cursor")
        newly_dispatched = len(dispatches) - dispatch_count_base
        # The shared core's trainer cursor counts committed groups. This local
        # single-inflight adapter includes filtered refill attempts in the durable cursor.
        next_task_index = cursor_base["next_task_index"] + newly_dispatched
        next_group_index = cursor_base["next_group_index"] + newly_dispatched
        return {
            "schema": _RUN_SCHEMA,
            "training_identity": identity,
            "lineage_id": (engine.recovery_metadata or {}).get("lineage_id"),
            "engine_epoch": (engine.recovery_metadata or {}).get("engine_epoch"),
            "logical_job": {
                "run_id": kwargs["run_id"],
                "kind": kwargs["request"].kind,
                "requested_steps": int(kwargs["state"]["requested_steps"]),
                "completed_updates": kwargs["next_completed_updates"],
                "task_ids": train_task_ids,
                "parent_run_id": parent_run_id,
                "segment": segment,
            },
            "task_cursor": {
                "next_task_index": next_task_index,
                "next_group_index": next_group_index,
                "ordered_task_ids": train_task_ids,
                "ordered_task_ids_sha256": identity["ordered_task_ids_sha256"],
                "dispatches_since_resume": newly_dispatched,
            },
            "environment": environment.name,
            "adapter_source_identity": config["source_identity"],
            "reward_schema": config["reward_schema"],
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "sampling_schedule": _SAMPLING_SCHEDULE,
        }

    services = RunCoordinatorServices(
        config=core_config,
        runtime=runtime,
        engine=engine,
        managers={},
        worker_pool=SimpleNamespace(),
        compute_owner=compute_owner,
        bepa=None,
        evaluation=evaluation,
        seed_service=seed_service,
        update_lock=asyncio.Lock(),
        set_monitor_phase=lambda phase: setattr(runtime, "monitor_phase", phase),
        latest_policy=lambda: runtime.policies[int(engine.policy_version)],
        publish_current=publish_current,
        make_recovery_context=recovery_context,
        engine_recovery_metadata=lambda: engine.recovery_metadata,
        assert_workers_current=assert_workers_current,
        sync_workers=sync_workers,
        persist_group=_trajectory_persistor(run_dir),
        stream_factory=lambda **kwargs: AsyncGroupStream(**kwargs),
        measure_alignment=lambda *_args, **_kwargs: {"status": "not_measured"},
        alignment_from_proximal_scores=lambda *_args, **_kwargs: {"status": "not_measured"},
        log_alignment_warning=lambda *_args, **_kwargs: None,
        publication_from_update=lambda result, _version: result["publication"],
        prune_synced_policies=lambda *_args, **_kwargs: [],
        diagnostic_course="local-user-training",
        diagnostic_run_kinds=frozenset(),
        alignment_thresholds={},
        environment=validated_environment,
        reward_scorer=validated_scorer,
        arc_sampling=False,
    )
    return RunCoordinator(services), runtime


def _initial_record(run_id: str, engine: Any, target_steps: int, cursor: dict[str, Any],
                    *, parent_run_id: str | None, segment: int) -> RunState:
    state = _new_run_state(
        run_id=run_id,
        engine=engine,
        target_steps=target_steps,
        prior=None,
        segment=segment,
        parent_run_id=parent_run_id,
    )
    state["resume_task_cursor"] = dict(cursor)
    return state


async def _execute_run(
    *,
    coordinator: RunCoordinator,
    runtime: Any,
    engine: Any,
    run_id: str,
    state: RunState,
    target_steps: int,
    task_ids: list[str],
) -> dict[str, Any]:
    runtime.runs[run_id] = state
    runtime.active_run_id = run_id
    await coordinator.services.sync_workers(coordinator.services.latest_policy())
    await coordinator.execute(
        run_id,
        SimpleNamespace(kind="training", steps=target_steps, task_ids=task_ids),
    )
    return json_safe(dict(state))


def _checkpoint_exists(run_dir: Path) -> bool:
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    return checkpoint.is_file() and not checkpoint.is_symlink()


def _load_engine(config: dict[str, Any], run_dir: Path, *, resume: bool) -> Any:
    # Import lazily so the contracts and torch-free runner tests do not import HF dependencies.
    from .hf_engine import HFEngine

    return HFEngine(config, run_dir, resume=resume)


def _run_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir.parent / f".{run_dir.name}.training.lock"
    handle = lock_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another training process owns {run_dir}") from exc
    return handle


def train(
    config: dict[str, Any] | None,
    run_dir: str | Path,
    steps: int | None = None,
    resume: bool = False,
    *,
    reward_fn: RewardFn | None = None,
) -> dict[str, Any]:
    """Run or resume local HF training toward a total policy-update target."""
    if type(resume) is not bool:
        raise TypeError("resume must be a boolean")
    if not resume and steps is None:
        raise ValueError("a new run requires a total-update target in steps")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps must be a positive total policy-update target")
    requested_run_dir = Path(run_dir).expanduser()
    if requested_run_dir.is_symlink():
        raise ValueError("run_dir must not be a symlink")
    resolved_run_dir = requested_run_dir.resolve()
    if resume and not resolved_run_dir.is_dir():
        raise FileNotFoundError("resume run_dir does not exist")
    lock = _run_lock(resolved_run_dir)
    try:
        if not resume and any(resolved_run_dir.iterdir()):
            raise FileExistsError("a new run requires an empty run_dir")

        normalized, tasks, resolved_reward, identity, stored = _load_inputs(
            config, resolved_run_dir, resume=resume, reward_fn=reward_fn,
        )
        if resume and not _checkpoint_exists(resolved_run_dir):
            raise FileNotFoundError("resume requires a complete checkpoints/latest.pt")
        engine = _load_engine(normalized, resolved_run_dir, resume=resume)
        try:
            evidence = _stable_engine_evidence(engine, normalized)
            policy = _policy_record(engine)
            identity = {**identity, "engine_evidence": evidence}
            if resume:
                assert stored is not None
                if stored.get("identity", {}).get("engine_evidence") != evidence:
                    raise ValueError("resumed model/tokenizer/adapter evidence differs from the original run")
                metadata = engine.recovery_metadata
                context = metadata.get("update_context") if isinstance(metadata, dict) else None
                if int(engine.policy_version) > 0 and not isinstance(context, dict):
                    raise ValueError("resume checkpoint is missing its source-bound update context")
                if isinstance(context, dict):
                    if context.get("training_identity") != identity:
                        raise ValueError("resume checkpoint training identity differs from config/tasks/reward")
                    if context.get("model_revision") != normalized["model_revision"]:
                        raise ValueError("resume checkpoint model revision differs from the requested revision")

            if not resume:
                write_json_atomic(resolved_run_dir / "config.json", {
                    "schema": _CONFIG_SCHEMA,
                    "config": normalized,
                    "identity": identity,
                })

            metadata = engine.recovery_metadata if isinstance(engine.recovery_metadata, dict) else {}
            context = metadata.get("update_context")
            context = context if isinstance(context, dict) else None
            logical_job = context.get("logical_job", {}) if context is not None else {}
            task_cursor = context.get("task_cursor", {}) if context is not None else {}
            active_record_path = resolved_run_dir / "logical_run.json"
            active_record = read_json(active_record_path) if active_record_path.is_file() else None
            prior_target = logical_job.get("requested_steps") if logical_job else None
            prior_run_id = logical_job.get("run_id") if logical_job else None
            prior_cursor = {
                "next_task_index": task_cursor.get("next_task_index", 0),
                "next_group_index": task_cursor.get("next_group_index", 0),
            }
            if context is None and active_record is not None:
                if active_record.get("identity") != identity:
                    raise ValueError("resume logical-run identity differs from current source identity")
                if int(engine.policy_version) != int(active_record.get("policy_version", -1)):
                    raise ValueError("logical-run cursor is not bound to the current durable policy version")
                prior_target = active_record.get("requested_steps")
                prior_run_id = active_record.get("run_id")
                prior_cursor = active_record.get("task_cursor", prior_cursor)
            if resume and context is None and int(engine.policy_version) > 0:
                raise ValueError("resume checkpoint does not identify its logical run and task cursor")

            current_version = int(engine.policy_version)
            requested_target = steps if steps is not None else prior_target
            if requested_target is None:
                raise ValueError("resume requires steps or a persisted total-update target")
            if type(requested_target) is not int or requested_target < current_version:
                raise ValueError("total-update target is below the durable policy version")

            prior_state = None
            if prior_run_id:
                manifest_path = resolved_run_dir / "runs" / f"{prior_run_id}.json"
                if manifest_path.is_file():
                    prior_state = read_json(manifest_path)
                    if prior_state.get("run_id") != prior_run_id:
                        raise ValueError("prior run manifest identity mismatch")
            complete = bool(
                prior_state is not None
                and prior_state.get("status") == "succeeded"
                and int(prior_state.get("completed_updates", -1)) >= int(prior_state.get("requested_steps", 0))
            )
            if resume and complete and requested_target == current_version:
                return {
                    "run_id": prior_run_id,
                    "status": "succeeded",
                    "completed_updates": current_version,
                    "target_updates": requested_target,
                    "updates_this_invocation": 0,
                    "run_dir": str(resolved_run_dir),
                    "checkpoint": str(resolved_run_dir / "checkpoints/latest.pt"),
                    "policy_version": current_version,
                    "summary": prior_state.get("summary"),
                    "identity": identity,
                    "policy_identity_kind": evidence["policy_identity_kind"],
                    "publication_scope": "local_only",
                }
            if resume and complete and requested_target > current_version:
                parent_run_id = prior_run_id
                run_id = uuid.uuid4().hex
                segment = int(prior_state.get("segment", 0)) + 1
                state = _initial_record(
                    run_id, engine, requested_target, prior_cursor,
                    parent_run_id=parent_run_id, segment=segment,
                )
            else:
                if resume:
                    if not isinstance(prior_run_id, str) or not prior_run_id:
                        if active_record is None:
                            raise ValueError("resume checkpoint lacks a logical run ID")
                        prior_run_id = active_record.get("run_id")
                    if not isinstance(prior_run_id, str) or not prior_run_id:
                        raise ValueError("resume checkpoint has an invalid logical run ID")
                    run_id = prior_run_id
                    parent_run_id = logical_job.get("parent_run_id")
                    segment = int(logical_job.get("segment", 0))
                    state = _new_run_state(
                        run_id=run_id,
                        engine=engine,
                        target_steps=requested_target,
                        prior=prior_state,
                        segment=segment,
                        parent_run_id=parent_run_id,
                    )
                    state["resume_task_cursor"] = dict(prior_cursor)
                else:
                    run_id = uuid.uuid4().hex
                    parent_run_id = None
                    segment = 0
                    state = _initial_record(
                        run_id, engine, requested_target,
                        {"next_task_index": 0, "next_group_index": 0},
                        parent_run_id=None, segment=segment,
                    )

            task_ids = [task.task_id for task in tasks if task.split == "train" and task.pool != "diagnostic"]
            logical_payload = {
                "schema": _RUN_SCHEMA,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "segment": segment,
                "status": "running",
                "requested_steps": requested_target,
                "policy_version": current_version,
                "task_cursor": state.get("resume_task_cursor", prior_cursor),
                "identity": identity,
            }
            write_json_atomic(active_record_path, logical_payload)
            coordinator, runtime = _build_services(
                config=normalized,
                run_dir=resolved_run_dir,
                engine=engine,
                tasks=tasks,
                reward_fn=resolved_reward,
                identity=identity,
                target_steps=requested_target,
                parent_run_id=parent_run_id,
                segment=segment,
                cursor_base={
                    "next_task_index": int(state.get("resume_task_cursor", prior_cursor).get("next_task_index", 0)),
                    "next_group_index": int(state.get("resume_task_cursor", prior_cursor).get("next_group_index", 0)),
                },
                dispatch_count_base=(
                    len(state.get("stream_dispatches", []))
                    if isinstance(state.get("stream_dispatches", []), list) else 0
                ),
            )
            result = asyncio.run(_execute_run(
                coordinator=coordinator,
                runtime=runtime,
                engine=engine,
                run_id=run_id,
                state=state,
                target_steps=requested_target,
                task_ids=task_ids,
            ))
            write_json_atomic(active_record_path, {
                **logical_payload,
                "status": result["status"],
                "policy_version": int(engine.policy_version),
                "completed_updates": result.get("completed_updates"),
                "summary": result.get("summary"),
            })
            return {
                "run_id": run_id,
                "status": result["status"],
                "completed_updates": int(engine.policy_version),
                "target_updates": requested_target,
                "updates_this_invocation": int(engine.policy_version) - current_version,
                "run_dir": str(resolved_run_dir),
                "checkpoint": str(resolved_run_dir / "checkpoints/latest.pt"),
                "policy_version": int(engine.policy_version),
                "lineage_id": metadata.get("lineage_id"),
                "segment": segment,
                "parent_run_id": parent_run_id,
                "summary": result.get("summary"),
                "identity": identity,
                "policy_identity_kind": evidence["policy_identity_kind"],
                "publication_scope": "local_only",
                "run_manifest": str(resolved_run_dir / "runs" / f"{run_id}.json"),
            }
        finally:
            engine.shutdown()
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
