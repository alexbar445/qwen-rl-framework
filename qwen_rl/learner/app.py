"""HTTP Learner API, complete-group CPU scheduler and mounted trusted ARC scorer."""
from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager, nullcontext
import hashlib
import hmac
import json
import logging
import math
import os
import shutil
import statistics
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..bepa.seed_scheduler import (
    RunSeedScheduler,
)
from ..contracts.environment import (
    Environment, RewardScorer, Task, validate_environment, validate_recovery_source_identity,
    validate_reward_scorer,
)
from ..contracts.worker import ADAPTER_FILES, NATIVE_ADAPTER_FILES, MAX_CONTEXT_TOKENS
from ..contracts.activation_offload import DEFAULT_OFFLOAD_MIN_TOKENS
from ..contracts.gradient_partitions import DEFAULT_GRADIENT_ACCUMULATION_ROWS
from ..evaluation.policy import EvaluationPolicy
from ..rollout.groups import PolicySnapshot
from ..rollout.pipeline import AsyncGroupStream
from ..runtime.adapter_cache import prune_adapter_cache
from ..runtime.manifest import write_json_atomic
from ..transport.tensor_codec import encode_tensor_payload
from ..worker.placement import EpisodeWorkerPool, complete_group_episode_capacity
from ..worker.inference_pool import SharedInferencePool
from .app_lifecycle import make_learner_lifespan
from .app_routes import (
    _finish_cache_write,
    attach_health_route,
    attach_monitor_route,
    attach_policy_routes,
)
from .compute import ComputeOwner
from .monitoring import ForegroundActivity, RunCommitView
from .policy_maintenance import remove_conflicting_publication_directory
from .policy_file_references import PolicyFileReferences
from .stop_control import StopAfterCheckpointControl, wait_for_engine_durable
from .run.auth import PolicyReadAuthorizer
from .run.bepa import BepaSeedService
from .run.coordinator import RunCoordinator, RunCoordinatorServices
from .run.evaluation import EvaluationCoordinator
from .run.groups import GroupGenerationError
from .run.state import RunState
from .run.workers import WorkerControlPlane
from .telemetry import TrainingTelemetry

_IDENTITY_DIAGNOSTIC_COURSE = "identity-workload-v1"
_IDENTITY_DIAGNOSTIC_RUN_KINDS = frozenset({"selftest", "diagnostic"})
_LOGPROB_MEAN_ABS_TOLERANCE = 0.01
_LOGPROB_MAX_ABS_TOLERANCE = 0.1
_LOGPROB_RMS_TOLERANCE = 0.02


def _prune_synced_policies(
    run_dir: str | Path,
    policies: dict[int, dict[str, Any]],
    policy: dict[str, Any],
    acknowledgements: dict[str, dict[str, Any]],
    worker_ids: set[str],
    keep_versions: frozenset[int] = frozenset(),
    policy_file_references: PolicyFileReferences | None = None,
) -> list[int]:
    """Retire unreferenced delivery files after every required Worker ACKs."""
    version = policy["version"]
    from .policy_maintenance import validate_policy_acknowledgements
    validate_policy_acknowledgements(policy, acknowledgements, worker_ids)
    recent_policy_versions = frozenset(range(max(0, version - 2), version))
    transport_recent_versions = frozenset({version - 1}) if version > 0 else frozenset()
    cleanup_context = (
        policy_file_references.cleanup()
        if policy_file_references is not None else nullcontext(frozenset())
    )
    with cleanup_context as response_versions:
        protected_versions = frozenset(keep_versions) | response_versions
        retained_policy_versions = recent_policy_versions | protected_versions
        removed = prune_adapter_cache(
            Path(run_dir) / "policies", version, width=8,
            keep_versions=retained_policy_versions,
        )
        removed_transport = prune_adapter_cache(
            Path(run_dir) / "policy-transport-cache", version, width=8,
            keep_versions=transport_recent_versions | protected_versions,
        )
        for older in list(policies):
            if older < version and older not in retained_policy_versions:
                del policies[older]
    return sorted(set(removed) | set(removed_transport))




_LOGGER = logging.getLogger(__name__)


def _persist_trajectory_group(
    run_dir: Path,
    group_id: str,
    trajectories: list[dict[str, Any]],
    worker_ids: list[str],
    *,
    phase: Literal["pre_update", "post_update"],
    policy_version: int,
    adapter_sha256: str,
    base_fingerprint: str,
) -> str:
    """Persist the original CPU rollout arrays before later gates or updates can fail."""
    if phase not in {"pre_update", "post_update"}:
        raise ValueError("trajectory phase must be pre_update or post_update")
    if not group_id or Path(group_id).name != group_id:
        raise ValueError("trajectory group_id must be a non-empty filename component")
    if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
        raise ValueError("trajectory policy_version must be a non-negative integer")
    if not isinstance(adapter_sha256, str) or not adapter_sha256:
        raise ValueError("trajectory adapter SHA-256 is required")
    if not isinstance(base_fingerprint, str) or not base_fingerprint:
        raise ValueError("trajectory base fingerprint is required")
    if len(trajectories) < 2 or len(worker_ids) != len(trajectories):
        raise ValueError("trajectory persistence requires a complete group with matching Worker identities")

    rows: list[dict[str, Any]] = []
    for trajectory, worker_id in zip(trajectories, worker_ids):
        if not isinstance(trajectory, dict):
            raise ValueError("trajectory persistence requires trajectory objects")
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("trajectory worker_id must be a non-empty string")
        if (trajectory.get("group_id") != group_id
                or trajectory.get("policy_version") != policy_version):
            raise ValueError("trajectory metadata differs from its group metadata")
        input_ids = trajectory.get("input_ids")
        policy_mask = trajectory.get("policy_mask")
        behavior_logprobs = trajectory.get("behavior_logprobs")
        reward = trajectory.get("reward")
        if (not isinstance(input_ids, list) or not input_ids
                or any(isinstance(token, bool) or not isinstance(token, int) for token in input_ids)
                or not isinstance(policy_mask, list) or len(policy_mask) != len(input_ids)
                or any(value not in (0, 1, False, True) for value in policy_mask)
                or not isinstance(behavior_logprobs, list)
                or len(behavior_logprobs) != len(input_ids)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) for value in behavior_logprobs)
                or isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(float(reward))):
            raise ValueError("trajectory arrays or reward are invalid")
        provenance = {}
        for name in ('source_worker_id', 'generation_worker_id'):
            if name in trajectory:
                value = trajectory[name]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"trajectory {name} must be a non-empty Worker identity")
                provenance[name] = value
        if provenance.get('generation_worker_id', worker_id) != worker_id:
            raise ValueError('trajectory generation Worker differs from its final Worker')
        if 'generation_worker_history' in trajectory:
            history = trajectory['generation_worker_history']
            if (not isinstance(history, list) or not history or history[-1] != worker_id
                    or any(not isinstance(worker, str) or not worker.strip() for worker in history)):
                raise ValueError('trajectory generation Worker history differs from its final Worker')
            provenance['generation_worker_history'] = list(history)
        rows.append({
            **provenance,
            "worker_id": worker_id,
            "input_ids": input_ids,
            "policy_mask": policy_mask,
            "behavior_logprobs": behavior_logprobs,
            "reward": reward,
            "group_id": trajectory["group_id"],
            "policy_version": trajectory["policy_version"],
            **{key: trajectory[key] for key in ("row_kind", "episode_id", "messages", "tool_events", "constraint_segments",
                "exact", "cost", "generated_tokens", "observation_tokens", "successful_tool_calls", "tool_call_failures", "termination_reason", "training_verification",
                "reward_components", "reward_schema", "attempt_exact", "per_target_exact", "competition_score") if key in trajectory},
        })

    root = Path(run_dir)
    relative_path = Path("trajectories") / f"{group_id}.json"
    destination = root / relative_path
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite trajectory evidence: {destination}")
    write_json_atomic(destination, {
        "schema": "qwen-rl-trajectory-group-v1",
        "phase": phase,
        "group_id": group_id,
        "policy_version": policy_version,
        "adapter_sha256": adapter_sha256,
        "base_fingerprint": base_fingerprint,
        "trajectories": rows,
    })
    return relative_path.as_posix()


def _inside_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RunRequest(APIModel):
    kind: Literal["training", "selftest", "diagnostic"] = "training"
    steps: int | None = Field(default=None, ge=1, le=100_000)
    task_ids: list[str] | None = None


class ResumeRequest(APIModel):
    lineage_id: str = Field(min_length=1)


class CheckpointRequest(APIModel):
    reason: Literal["explicit"] = "explicit"


class StopControlRequest(APIModel):
    operation_id: str = Field(
        min_length=1, max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    mode: Literal["after_checkpoint"]


class ProbeRequest(APIModel):
    lengths: list[int] = Field(default_factory=lambda: [64, 1024], min_length=1, max_length=3)


def _cfg_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "as_dict"):
        config = config.as_dict()
    if not isinstance(config, dict):
        raise TypeError("Learner app config must be a strict runtime config or dictionary")
    return config


def _require_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("ARC_ENV_TOKEN")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not expected or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized",
                            headers={"WWW-Authenticate": "Bearer"})


def _current_prompt_identity(environment: Environment[Task]) -> object:
    identity = environment.prompt_identity
    return identity() if callable(identity) else identity


def _identity_fixture() -> Task:
    """Publicly specified copy-through rule used only for connectivity self-test."""
    from ..task_source import TaskSpec

    challenge = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
        ],
        "test": [{"input": [[9, 0], [2, 1]]}],
    }
    return TaskSpec(
        task_id="rl-smoke:identity-copy-through-v1",
        family="rl-smoke-identity-copy-through",
        pool="diagnostic",
        split="train",
        variant=None,
        challenge=challenge,
        targets={"test/0": [[9, 0], [2, 1]]},
        public_id="rl-smoke-identity-copy-through-v1",
    )


def _identity_course_messages(task: Task) -> list[dict[str, str]]:
    """Give every trajectory the same public rule and bounded, executable solve routes."""
    from ..prompt import build_task_messages

    messages = build_task_messages(task)
    messages[0]["content"] += (
        '\n\nThis is a public connectivity diagnostic. The rule is identity: submit the '
        'unchanged public grid test/0/input as the output for test/0. The shortest route is '
        'submit(source="test/0/input"), then end your response normally. If you inspect the public '
        'examples first, use at most these two read-only calls, then submit the unchanged '
        'test/0/input: A.objects(source="train/0/input") and '
        'A.objects(source="train/1/input"). Do not inspect other objects or transform the '
        'grid. Every sample sees these same routes. Use at most one arc_action call in each '
        'assistant turn.'
    )
    return messages


def _identity_course_config(config: dict[str, Any]) -> dict[str, Any]:
    rollout = dict(config["rollout"])
    rollout["max_turns"] = min(int(rollout["max_turns"]), 4)
    rollout["enable_thinking"] = True
    return {**config, "rollout": rollout}


def _same_initial_prompt_prefix_sha256(trajectories: list[dict[str, Any]]) -> str:
    """Return a digest for one group's common IDs before the first policy token."""
    prefixes: list[tuple[int, ...]] = []
    for index, trajectory in enumerate(trajectories):
        ids = [int(token) for token in trajectory["input_ids"]]
        mask = [int(value) for value in trajectory["policy_mask"]]
        try:
            first_policy_position = mask.index(1)
        except ValueError as exc:
            raise ValueError(f"trajectory {index} has no policy token") from exc
        if first_policy_position == 0:
            raise ValueError(f"trajectory {index} has no initial prompt before its first policy token")
        prefixes.append(tuple(ids[:first_policy_position]))
    if not prefixes or any(prefix != prefixes[0] for prefix in prefixes[1:]):
        raise ValueError("group has mixed initial prompt prefixes before its first policy token")
    encoded = json.dumps(prefixes[0], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_engine(
    config: dict[str, Any], *, recovery_semantics: dict[str, Any] | None = None,
    tokenizer: Any = None, process_owner_lock: Any = None, recovery_only: bool = False,
    preserve_control_files: bool | None = None,
):
    if type(recovery_only) is not bool:
        raise TypeError("recovery_only must be a boolean")
    if preserve_control_files is not None and type(preserve_control_files) is not bool:
        raise TypeError("preserve_control_files must be a boolean or None")
    from .engine import LearnerEngine

    learner = config["learner"]
    if (learner.get("max_padded_tokens_per_microbatch") is not None
            and learner.get("parallelism", "single") != "single"):
        raise ValueError("learner.max_padded_tokens_per_microbatch requires learner.parallelism='single'")
    lora = config["lora"]
    model_backend = config.get("model_backend", "default")
    if model_backend == "ternary":
        if config["quantization"].get("mode") != "ternary":
            raise ValueError("T2 Learner requires quantization.mode='ternary'")
        from ..contracts.ternary import (
            TERNARY_QG_LAYOUT_VERSION,
            validate_artifact_identity,
        )

        identity_path = config.get("artifact_identity_path")
        identity = config.get("artifact_identity")
        if not isinstance(identity_path, str) or not identity_path:
            raise ValueError("T2 Learner requires the manager-validated artifact_identity_path")
        if not isinstance(identity, dict):
            raise ValueError("T2 Learner requires the manager-validated artifact_identity document")
        identity = validate_artifact_identity(identity)
        artifact = identity["source"]["ninfer_artifact"]
        if Path(config["model_path"]).expanduser().resolve() != Path(artifact["path"]).expanduser().resolve():
            raise ValueError("T2 model_path differs from the manager-validated artifact identity")
        if config.get("recovery_semantics") is not None and not isinstance(config["recovery_semantics"], dict):
            raise ValueError("T2 recovery_semantics must be a mapping")
        declared_semantics = dict(recovery_semantics or {})
        manager_semantics = dict(config.get("recovery_semantics") or {})
        for key, value in manager_semantics.items():
            if key in declared_semantics and declared_semantics[key] != value:
                raise ValueError(f"T2 manager recovery_semantics.{key} differs from app preflight")
            declared_semantics[key] = value
        for key, expected in (
            ("base_artifact_sha256", artifact["sha256"]),
            ("qg_layout", TERNARY_QG_LAYOUT_VERSION),
        ):
            if key in declared_semantics and declared_semantics[key] != expected:
                raise ValueError(f"T2 recovery_semantics.{key} differs from the verified artifact identity")
            declared_semantics[key] = expected
        engine_class = None
    elif model_backend == "default":
        engine_class = LearnerEngine
    else:
        raise ValueError(f"unsupported Learner model_backend: {model_backend!r}")
    if model_backend == "default" and config["quantization"]["mode"] == "torchao":
        from .parallel_engine import ParallelLearnerEngine
        engine_class = ParallelLearnerEngine
    from qwen_rl.contracts.gradient_partitions import gradient_executor_pools

    engine_config = {
        "gpus": learner["gpus"],
        "parallelism": learner.get("parallelism", "single"),
        "update_backend": learner.get("update_backend", "trainer"),
        "gradient_executor_token_envs": learner.get("gradient_executor_token_envs", {}),
        "gradient_executor_worker_ids": learner.get("gradient_executor_worker_ids", {}),
        "sampling_only_worker_ids": learner.get("sampling_only_worker_ids", {}),
        "sampling_group_quotas": learner.get("sampling_group_quotas", {}),
        "sampling_shared_worker_pairs": learner.get("sampling_shared_worker_pairs", {}),
        "gradient_remote_executor_ids": learner.get("gradient_remote_executor_ids", ()),
        "gradient_remote_min_queued_trajectories": learner.get("gradient_remote_min_queued_trajectories", 1),
        "gradient_local_executor_worker_ids": learner.get("gradient_local_executor_worker_ids", ()),
        "gradient_tasks_per_worker": learner.get("gradient_tasks_per_worker", 1),
        "gradient_accumulation_rows": learner.get("gradient_accumulation_rows", DEFAULT_GRADIENT_ACCUMULATION_ROWS),
        "gradient_memory_limit_bytes": learner.get("gradient_memory_limit_bytes"),
        "gradient_max_inflight_input_tokens": learner.get("gradient_max_inflight_input_tokens"),
        "gradient_work_stealing_pools": gradient_executor_pools(config),
        "gradient_worker_sync_ids": learner.get("gradient_worker_sync_ids", ()),
        "gradient_wait_timeout": learner.get("gradient_wait_timeout", 3600.0),
        "gradient_initial_training_dir": learner.get("gradient_initial_training_dir"),
        "gpu_sharing": learner.get("gpu_sharing", "dedicated"),
        "initial_adapter_path": learner.get("initial_adapter_path"),
        "residency_host_budget_gib": learner.get("residency_host_budget_gib", 12.0),
        "compute_python": learner.get("compute_python"),
        "compute_pythonpath": learner.get("compute_pythonpath"),
        "compute_timeout": learner.get("compute_timeout", 3600.0),
        "gradient_checkpointing": learner.get("gradient_checkpointing", True),
        "cuda_memory_fraction": learner.get("cuda_memory_fraction", 1.0),
        "compute_runtime_dir": str(Path(config["run_dir"]) / "compute"),
        "gradient_server_dir": str(Path(config["run_dir"]) / "gradient-server"),
        "model_path": config["model_path"],
        "lora_rank": lora["rank"],
        "lora_alpha": lora["alpha"],
        "lr": learner["lr"],
        "checkpoint_dir": str(Path(config["run_dir"]) / "checkpoints"),
        "max_sequence_length": learner["max_sequence_length"],
        "group_size": config["rollout"]["group_size"],
        "groups_per_update": learner["groups_per_update"],
        "groups_per_update_auto": learner.get("groups_per_update_auto", False),
        "microbatch_size": learner["microbatch_size"],
        "max_padded_tokens_per_microbatch": learner.get("max_padded_tokens_per_microbatch"),
        "importance_sampling_cap": learner.get("importance_sampling_cap", 3.0),
        "activation_offload": learner.get("activation_offload", True),
        "activation_offload_min_tokens": learner.get("activation_offload_min_tokens", DEFAULT_OFFLOAD_MIN_TOKENS),
        "activation_cache_parameter_names": learner.get("activation_cache_parameter_names", []),
        "activation_cache_cpu_bytes": learner.get("activation_cache_cpu_bytes", 2 * 1024**3),
        "activation_cache_staging_bytes": learner.get("activation_cache_staging_bytes", 512 * 1024**2),
        "checkpoint_input_recompute_span": learner.get("checkpoint_input_recompute_span", 1),
        "ternary_regional_compile": learner.get("ternary_regional_compile", True),
        "packed_weight_offload": learner.get("packed_weight_offload", False),
        "packed_weight_offload_prefetch": learner.get("packed_weight_offload_prefetch", True),
        "packed_weight_offload_max_cpu_master_bytes": learner.get(
            "packed_weight_offload_max_cpu_master_bytes", 4 * 1024**3
        ),
        "packed_weight_offload_max_pinned_bytes": learner.get(
            "packed_weight_offload_max_pinned_bytes", 512 * 1024**2
        ),
        "weight_decay": learner.get("weight_decay", 0.0),
        "adam_betas": learner.get("adam_betas", (0.9, 0.999)),
        "adam_eps": learner.get("adam_eps", 1e-8),
        "max_grad_norm": learner.get("max_grad_norm", 1.0),
        "recovery_dir": str(learner.get("recovery_dir") or (Path(config["run_dir"]) / "training")),
        "recovery_mode": learner.get("recovery_mode", "auto"),
        "recovery_lineage_id": learner.get("recovery_lineage_id"),
        "recovery_path_mappings": learner.get("recovery_path_mappings", {}),
        "recovery_semantics": recovery_semantics or {},
        "dataset_dir": str(Path(config["dataset_dir"]).resolve()),
        "arc": {
            **config["arc"],
            "contract_path": str(Path(config["arc"]["contract_path"]).resolve()),
        },
        "bepa": config.get("bepa", {}),
        "constrained_decoding": config.get("rollout",{}).get("constrained_decoding",False),
    }
    if model_backend == "ternary":
        from .ternary_engine import TernaryLearnerEngine

        engine_config.update({
            "model_backend": "ternary",
            "artifact_identity_path": str(Path(identity_path).expanduser().resolve()),
            "artifact_identity": identity,
            "tokenizer_path": str(Path(config.get("tokenizer_path") or (Path(config["run_dir"]) / "tokenizer")).resolve()),
            "recovery_semantics": declared_semantics,
        })
        if learner.get("parallelism", "single") == "tp":
            engine_config["tp_output_chunk_rows"] = learner.get("tp_output_chunk_rows", 8192)
        if learner.get("parallelism", "single") in {"ddp", "tp"}:
            from .parallel_engine import ParallelLearnerEngine
            engine_class = ParallelLearnerEngine
        else:
            # Retired ordinary single-card construction must fail before a
            # shared Worker is drained or any model/CUDA state is touched.
            if learner.get("update_backend", "trainer") != "central_gradient":
                from .ternary_engine import OutdatedTernaryBuilderError

                raise OutdatedTernaryBuilderError(
                    "outdated single-card T2 builder is disabled; "
                    "select the maintained T2 Learner assembly"
                )
            engine_class = TernaryLearnerEngine
    phase = learner.get("gpu_sharing") == "phase_shared"
    if phase:
        from .gpu_phases import WorkerResidencyFleet
        fleet = WorkerResidencyFleet(config)
        fleet.apply("drain")
        fleet.apply("suspend")
    if learner.get("update_backend", "trainer") == "central_gradient":
        from .gradient_server.engine import build_gradient_learner_engine

        engine = build_gradient_learner_engine(
            engine_config, tokenizer=tokenizer, process_owner_lock=process_owner_lock,
            recovery_only=recovery_only, preserve_control_files=preserve_control_files,
        )
    else:
        if recovery_only:
            raise ValueError("recovery_only is supported only by the central-gradient backend")
        if preserve_control_files is not None:
            raise ValueError("preserve_control_files is supported only by the central-gradient backend")
        if process_owner_lock is not None:
            raise ValueError("process_owner_lock requires the central_gradient backend")
        engine = engine_class(engine_config)
    if phase:
        engine.park()
        fleet.apply("wake")
    return engine


def _load_worker_managers(config: dict[str, Any]) -> dict[str, Any]:
    from ..worker.client import HTTPGenerationManager

    learner_timeout = float(config["learner"].get("request_timeout", 30.0))
    managers = {}
    for worker in config["workers"]:
        scope = worker.get("scope", "local")
        if scope in {"gateway", "service"}:
            if (not worker.get("boot_id") or not worker.get("telemetry_epoch")
                    or not worker.get("model_sha256")):
                raise ValueError(
                    f"{scope.title()} Worker {worker['id']} is missing its active model/member identity; "
                    "start through the runtime manager to resolve a snapshot"
                )
            worker_url = worker["endpoint"]
            prefix = worker.get("url_prefix", "")
        else:
            worker_url = f"http://{worker['host']}:{worker['port']}"
            prefix = ""
        managers[worker["id"]] = HTTPGenerationManager(
            worker_url=worker_url,
            scope=scope,
            url_prefix=prefix,
            auth_token_env=worker.get("auth_token_env", "ARC_ENV_TOKEN"),
            worker_id=worker["id"],
            node_id=worker.get("node_id"),
            expected_worker_boot_id=worker.get("boot_id") or None,
            expected_worker_telemetry_epoch=worker.get("telemetry_epoch") or None,
            timeout_seconds=learner_timeout,
        )
    return managers


def _load_arc_source(config: dict[str, Any]):
    from ..task_source import load_tasks
    from ..budget import load_contract

    dataset_dir = Path(config["dataset_dir"])
    tasks = {task.task_id: task for task in load_tasks(
        pools=("arc2_train", "arc2_eval"), variants=True, dataset_dir=dataset_dir,
    )}
    from ..task_catalog import load_source_catalog
    if load_source_catalog(dataset_dir) is not None:
        requested = config.get("rollout", {}).get("task_ids") or [
            task_id for task_id, task in tasks.items() if task.split == "train"
        ]
        if any(task_id not in tasks or tasks[task_id].split != "train" for task_id in requested):
            raise ValueError("mixed rollout.task_ids must refer only to known training tasks")
        if {tasks[task_id].pool for task_id in requested} != {"synthetic_arc", "arc2_train"}:
            raise ValueError("mixed rollout.task_ids must include both synthetic_arc and arc2_train")
    fixture = _identity_fixture()
    tasks[fixture.task_id] = fixture
    return tasks, load_contract(config["arc"]["contract_path"])


def _recovery_semantics(config: dict[str, Any], tasks: dict[str, Any], contract: Any) -> dict[str, Any]:
    """Use the same semantic identity function as manager preflight and restore."""
    from .recovery import preflight_compatibility

    runtime = dict(config)
    runtime["model_path"] = str(Path(config["model_path"]).resolve())
    runtime["dataset_dir"] = str(Path(config["dataset_dir"]).resolve())
    runtime["run_dir"] = str(Path(config["run_dir"]).resolve())
    runtime["arc"] = {
        **config["arc"],
        "contract_path": str(Path(config["arc"]["contract_path"]).resolve()),
    }
    # `tasks` and `contract` are supplied from the single ARC source load used
    # to build the mounted service; preflight hashes their authoritative source
    # files without loading them again.
    if not isinstance(tasks, dict) or contract is None:
        raise TypeError("recovery semantics require the existing loaded ARC source")
    return preflight_compatibility(runtime)["semantics"]


def _load_arc_service(config: dict[str, Any], engine: Any, *, tasks=None, contract=None):
    from ..service import create_app as create_arc_app

    if tasks is None or contract is None:
        tasks, contract = _load_arc_source(config)
    arc = config["arc"]
    dataset_dir = Path(config["dataset_dir"])
    return create_arc_app(
        dataset_dir=dataset_dir,
        artifact_dir=Path(config["run_dir"]) / "arc",
        tokenizer=engine.tokenizer,
        contract=contract,
        group_timeout=float(arc["group_timeout"]),
        max_active_groups=int(arc["max_active_groups"]),
        max_generated_tokens=int(arc["max_generated_tokens"]),
        overlong_buffer=int(arc["overlong_buffer"]),
        tasks=tasks,
    )


def _policy_files(published: dict[str, Any], version: int) -> dict[str, dict[str, Any]]:
    if not isinstance(published, dict):
        raise TypeError("LearnerEngine.publish() must return a dictionary")
    reported_version = published.get("version")
    if isinstance(reported_version, bool) or reported_version != version:
        raise ValueError("published adapter version disagrees with LearnerEngine.policy_version")
    native = published.get("quantization", {}).get("mode") == "ternary"
    transport = published.get("native") if native else published
    if not isinstance(transport, dict):
        raise ValueError("T2 publication requires Learner-exported native policy files")
    expected_files = NATIVE_ADAPTER_FILES if native else ADAPTER_FILES
    entries = transport.get("files")
    directory_value = transport.get("directory")
    if not isinstance(entries, list) or not isinstance(directory_value, str) or not directory_value:
        raise ValueError("LearnerEngine.publish() must return version, directory, and files list")
    directory = Path(directory_value).resolve()
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid published adapter file entry")
        name = entry.get("name")
        if name not in expected_files or name in result:
            raise ValueError("published policy contains an unexpected transport file")
        declared = entry.get("sha256")
        size = entry.get("size")
        if (
            (declared is not None and (
                not isinstance(declared, str) or len(declared) != 64
            ))
            or isinstance(size, bool) or not isinstance(size, int) or size < 0
        ):
            raise ValueError(f"published adapter file {name} has invalid digest metadata or size")
        path = (directory / name).resolve()
        if directory not in path.parents:
            raise ValueError(f"published adapter file escaped its version directory: {name}")
        if not path.is_file():
            raise ValueError(f"published adapter file missing: {name}")
        if path.stat().st_size != size:
            raise ValueError(f"published adapter size mismatch: {name}")
        result[name] = {"path": path, "sha256": None, "size": size}
    if set(result) != expected_files:
        raise ValueError("publication must contain the exact selected policy transport files")
    return result


def _attach_native_source(policy: dict[str, Any], published: dict[str, Any]) -> None:
    policy["source_adapter_sha256"] = published["adapter_hash"]
    if published["quantization"].get("mode") != "ternary":
        return
    policy["artifact_id"] = published["native"]["artifact_id"]
    policy["_native_export_identity"] = published["native"]["base_identity"]
    peft_dir = Path(published["directory"])
    policy["_peft_files"] = {
        entry["name"]: {"path": peft_dir / entry["name"], "sha256": entry["sha256"]}
        for entry in published["files"]
    }


def _base_fingerprint(published: dict[str, Any]) -> str:
    value = published.get("base_fingerprint")
    if not isinstance(value, str) or not value:
        raise ValueError("LearnerEngine.publish() must include base_fingerprint")
    return value


def _quantization_evidence(published: dict[str, Any]) -> dict[str, Any]:
    value = published.get("quantization")
    if isinstance(value, dict) and value.get("mode") == "ternary":
        from ..contracts.ternary import TERNARY_QG_LAYOUT_VERSION
        from ..worker.native_artifact import validate_policy_quantization

        try:
            validate_policy_quantization("ternary", value, value)
        except ValueError as exc:
            raise ValueError("T2 policy publication omitted its typed ternary quantization evidence") from exc
        if value.get("qg_layout") != TERNARY_QG_LAYOUT_VERSION:
            raise ValueError("T2 policy publication omitted the current Q/G layout identity")
        return dict(value)
    from ..models.quantization import quantization_evidence
    return quantization_evidence(value)


def _validate_engine_evidence(engine: Any, config: dict[str, Any]) -> dict[str, Any]:
    evidence = getattr(engine, "evidence", None)
    if not isinstance(evidence, dict):
        raise TypeError("LearnerEngine must report structured model-load evidence")
    central_gradient = config.get("learner", {}).get("update_backend", "trainer") == "central_gradient"
    if central_gradient:
        if config.get("model_backend") != "ternary":
            raise ValueError("central-gradient evidence currently requires the verified T2 backend")
        from ..contracts.ternary import (
            TERNARY_QG_LAYOUT_VERSION,
            validate_artifact_identity,
        )
        from ..worker.native_artifact import validate_policy_quantization

        raw_identity = config.get("artifact_identity")
        if not isinstance(raw_identity, dict):
            raise ValueError("central-gradient runtime has no manager-validated artifact identity")
        identity = validate_artifact_identity(raw_identity)
        artifact = identity["source"]["ninfer_artifact"]
        if Path(config["model_path"]).expanduser().resolve() != Path(artifact["path"]).expanduser().resolve():
            raise ValueError("central-gradient model path differs from its verified artifact identity")
        if (evidence.get("compute_backend") != "central_gradient"
                or evidence.get("parallelism") != "central_gradient"
                or evidence.get("world_size") != 0
                or evidence.get("full_base_loaded") is not False
                or evidence.get("adapter_state_device") != "cpu"
                or evidence.get("optimizer_owner") != "central_cpu_adamw"):
            raise ValueError("central-gradient engine must prove CPU LoRA-only ownership with no loaded base")
        if (evidence.get("model_backend") != "ternary"
                or evidence.get("is_loaded_in_4bit") is not False
                or evidence.get("base_trainable_parameters") != 0):
            raise ValueError("central-gradient engine reported an unsupported base-model state")
        lora_evidence = evidence.get("lora")
        parameter_count = evidence.get("adapter_trainable_parameters")
        if (isinstance(parameter_count, bool) or not isinstance(parameter_count, int)
                or parameter_count < 1
                or not isinstance(lora_evidence, dict)
                or lora_evidence.get("all_trainable_names_lora_only") is not True
                or lora_evidence.get("non_lora_trainable_parameter_count") != 0
                or lora_evidence.get("trainable_lora_parameter_count") != parameter_count):
            raise ValueError("central-gradient engine must report its complete CPU LoRA inventory")
        adapter = identity["learner"]["adapter"]
        if (config.get("lora", {}).get("rank") != adapter["rank"]
                or config.get("lora", {}).get("alpha") != adapter["alpha"]):
            raise ValueError("central-gradient LoRA configuration differs from the verified artifact identity")
        quantization = evidence.get("quantization")
        try:
            validate_policy_quantization("ternary", quantization, quantization)
        except (TypeError, ValueError) as exc:
            raise ValueError("central-gradient quantization evidence is incomplete or invalid") from exc
        if (quantization.get("qg_layout") != TERNARY_QG_LAYOUT_VERSION
                or quantization.get("artifact_sha256") != artifact["sha256"]
                or quantization.get("artifact_id") != artifact["artifact_id"]):
            raise ValueError("central-gradient quantization evidence differs from the verified T2 artifact")
        identity_evidence = evidence.get("artifact_identity")
        if (not isinstance(identity_evidence, dict)
                or identity_evidence.get("artifact_sha256") != artifact["sha256"]
                or identity_evidence.get("artifact_id") != artifact["artifact_id"]):
            raise ValueError("central-gradient artifact evidence differs from the verified T2 identity")
        if (evidence.get("base_fingerprint") != artifact["sha256"]
                or evidence.get("artifact_fingerprint") != artifact["sha256"]):
            raise ValueError("central-gradient fingerprints must bind the verified T2 artifact")
    elif config.get("model_backend", "default") != "ternary":
        from ..models.quantization import quantization_evidence

        expected_mode = config["quantization"]["mode"]
        if expected_mode != "torchao":
            raise ValueError("non-T2 Learner runtime must select native NVFP4 with quantization.mode='torchao'")
        quantization_evidence(evidence.get("quantization"), expected_mode=expected_mode)
        if evidence.get("base_trainable_parameters") != 0:
            raise ValueError("LearnerEngine evidence must prove the base model is frozen")
    else:
        from ..contracts.ternary import (
            TERNARY_QG_LAYOUT_VERSION,
            validate_artifact_identity,
        )
        from ..worker.native_artifact import validate_policy_quantization

        if config.get("quantization", {}).get("mode") != "ternary":
            raise ValueError("T2 Learner runtime must select quantization.mode='ternary'")
        raw_identity = config.get("artifact_identity")
        if not isinstance(raw_identity, dict):
            raise ValueError("T2 Learner runtime has no manager-validated artifact identity")
        identity = validate_artifact_identity(raw_identity)
        artifact = identity["source"]["ninfer_artifact"]
        expected_sha = artifact["sha256"]
        if Path(config["model_path"]).expanduser().resolve() != Path(artifact["path"]).expanduser().resolve():
            raise ValueError("T2 Learner runtime model path differs from its validated artifact identity")
        expected_mode = config["learner"].get("parallelism", "single")
        expected_world = len(config["learner"]["gpus"])
        if (evidence.get("model_backend") != "ternary" or evidence.get("parallelism") != expected_mode
                or evidence.get("world_size") != expected_world):
            raise ValueError("T2 Learner must prove the configured fixed topology")
        if evidence.get("is_loaded_in_4bit") is not False:
            raise ValueError("T2 Learner must truthfully report is_loaded_in_4bit=false")
        if evidence.get("base_frozen") is not True or evidence.get("base_trainable_parameters") != 0:
            raise ValueError("T2 Learner evidence must prove the frozen base model")
        if (isinstance(evidence.get("adapter_trainable_parameters"), bool)
                or not isinstance(evidence.get("adapter_trainable_parameters"), int)
                or evidence["adapter_trainable_parameters"] < 1):
            raise ValueError("T2 Learner evidence must report positive trainable LoRA parameters")
        lora_evidence = evidence.get("lora")
        if (not isinstance(lora_evidence, dict)
                or lora_evidence.get("all_trainable_names_lora_only") is not True
                or lora_evidence.get("non_lora_trainable_parameter_count") != 0
                or lora_evidence.get("trainable_lora_parameter_count") != evidence["adapter_trainable_parameters"]):
            raise ValueError("T2 Learner evidence must bind trainable parameters to the verified LoRA inventory")
        adapter = identity["learner"]["adapter"]
        lora_config = config.get("lora", {})
        if (lora_config.get("rank") != adapter["rank"]
                or lora_config.get("alpha") != adapter["alpha"]):
            raise ValueError("T2 Learner LoRA configuration differs from the verified artifact identity")
        quantization = evidence.get("quantization")
        try:
            validate_policy_quantization("ternary", quantization, quantization)
        except (TypeError, ValueError) as exc:
            raise ValueError("T2 Learner quantization evidence is incomplete or invalid") from exc
        if (quantization.get("qg_layout") != TERNARY_QG_LAYOUT_VERSION
                or quantization.get("artifact_sha256") != expected_sha
                or quantization.get("artifact_id") != artifact["artifact_id"]):
            raise ValueError("T2 Learner quantization evidence differs from the frozen artifact/QG identity")
        identity_evidence = evidence.get("artifact_identity")
        if (not isinstance(identity_evidence, dict)
                or identity_evidence.get("artifact_sha256") != expected_sha
                or identity_evidence.get("artifact_id") != artifact["artifact_id"]):
            raise ValueError("T2 Learner model evidence differs from the verified artifact identity")
        if evidence.get("base_fingerprint") != expected_sha or evidence.get("artifact_fingerprint") != expected_sha:
            raise ValueError("T2 Learner fingerprints must equal the frozen .ninfer SHA-256")
        expected_quantization = config.get("quantization", {})
        for key in ("mode", "bits", "packed_resident", "compute_dtype"):
            if expected_quantization.get(key) != quantization.get(key):
                raise ValueError(f"T2 runtime quantization.{key} differs from loaded model evidence")

    if (isinstance(evidence.get("adapter_trainable_parameters"), bool)
            or not isinstance(evidence.get("adapter_trainable_parameters"), int)
            or evidence["adapter_trainable_parameters"] < 1):
        raise ValueError("LearnerEngine evidence must report trainable LoRA adapter parameters")
    return evidence


def _evaluation_publication_from_update(
    update_result: dict[str, Any], version: int,
) -> dict[str, Any]:
    published = update_result["publication"]
    source_files = _policy_files(published, version)
    result = {
        "version": version,
        "base_fingerprint": _base_fingerprint(published),
        "artifact_fingerprint": published["artifact_fingerprint"],
        "quantization": _quantization_evidence(published),
        "files": {name: {"sha256": None, "size_bytes": item["size"]} for name, item in source_files.items()},
        "_files": {
            name: {"path": Path(item["path"]), "sha256": None, "size": item["size"]}
            for name, item in source_files.items()
        },
    }
    _attach_native_source(result, published)
    return result


def _run_engine_probe(engine: Any, output_dir: Path, lengths: list[int]) -> dict[str, Any]:
    """Probe the already-loaded Learner; do not allocate a second model on GPU0."""
    if not lengths or lengths[0] != 64 or any(length < 2 for length in lengths):
        raise ValueError("probe lengths must begin with 64 and contain positive sequence lengths")
    probe = getattr(engine, "probe", None)
    if not callable(probe):
        raise ValueError("selected Learner engine does not support a bounded probe")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = probe(lengths)
    (output_dir / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class _TrackingManager:
    def __init__(self, worker_id: str, manager: Any, counters: dict[str, Any],
                 policy: PolicySnapshot, episode_id: str = "identity-test-episode", *,
                 evaluation_seed: int | None = None, generation_finished=None, before_generation=None):
        self._generation_finished = generation_finished
        self._generation_closed = False
        self._before_generation = before_generation
        self._generation_calls = 0
        self.worker_history = [worker_id]
        self.worker_id = worker_id
        self.manager = manager
        self.counters = counters
        self.policy = policy
        self.episode_id = episode_id
        self.evaluation_seed = evaluation_seed
        self.evaluation_turn = 0

    async def finish_episode_generation(self):
        if self._generation_closed:
            return
        if self._generation_finished is not None:
            await self._generation_finished()
        self._generation_closed = True

    async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any],
                       **unused: Any):
        if self._generation_closed:
            raise _PolicyContractError("generation requested after trusted terminal finish")
        if self._generation_calls and self._before_generation is not None:
            await self._before_generation(self)
        self._generation_calls += 1
        started = time.perf_counter()
        try:
            if self.evaluation_seed is not None:
                value = f"qwen-arc-eval-seed-v1:{self.evaluation_seed}:{self.evaluation_turn}".encode()
                seed = int.from_bytes(hashlib.sha256(value).digest()[:4], "big")
                unused["seed"] = seed
                self.counters.setdefault("evaluation_turn_seeds", []).append(seed)
                self.evaluation_turn += 1
            result = await self.manager.generate(request_id=request_id, prompt_ids=prompt_ids,
                                                 sampling_params=sampling_params,
                                                 policy_version=self.policy.version,
                                                 episode_id=self.episode_id,
                                                 adapter_hash=self.policy.adapter_sha256,
                                                 **unused)
            fields = getattr(result, "extra_fields", None)
            if isinstance(fields, dict):
                if (fields.get("worker_id") != self.worker_id
                        or fields.get("adapter_sha256") != self.policy.adapter_sha256
                        or fields.get("base_fingerprint") != self.policy.base_fingerprint
                        or fields.get("artifact_fingerprint") != self.policy.artifact_fingerprint):
                    raise _PolicyContractError("Worker generation policy SHA or model contract differs from its cohort")
                self.counters["sampling_config"] = fields.get("sampling_config")
            return result
        finally:
            self.counters["calls"] += 1
            self.counters["seconds"] += time.perf_counter() - started


def _agent_loop(engine: Any, manager: Any, config: dict[str, Any], group_size: int):
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap

    from ..verl_agent import ArcAgentLoop

    arc_cfg = config["arc"]
    rollout = config["rollout"]
    agent_cfg = OmegaConf.create({
        "arc": {
            "env_url": f"http://127.0.0.1:{config['learner']['port']}/arc",
            "group_timeout": float(arc_cfg["group_timeout"]),
            "turn_max_tokens": int(rollout["turn_max_tokens"]),
            "max_turns": int(rollout["max_turns"]),
            "constrained_decoding": rollout.get("constrained_decoding", False),
            "context_length": min(
                int(config["learner"].get("max_sequence_length", MAX_CONTEXT_TOKENS)),
                int(config.get("worker_inference", {}).get("max_model_len", MAX_CONTEXT_TOKENS)),
            ),
        },
        "actor_rollout_ref": {"rollout": {
            "n": group_size,
            "response_length": int(rollout["response_length"]),
            "prompt_length": int(rollout["prompt_length"]),
            "multi_turn": {"format": "qwen3_coder"},
            "val_kwargs": {"n": 1},
        }},
        "data": {"apply_chat_template_kwargs": {
            "enable_thinking": True,
        }},
    })
    return ArcAgentLoop(
        trainer_config=DictConfigWrap(agent_cfg),
        server_manager=manager,
        tokenizer=engine.tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=DictConfigWrap(agent_cfg.data),
        hf_model_type="qwen3_5",
    )


def _measure_logprob_alignment(
    engine: Any,
    trajectories: list[dict[str, Any]],
    worker_ids: list[str],
) -> dict[str, Any]:
    """Compare actual Worker scores with the Learner path under fixed group and worker gates."""
    if len(trajectories) != len(worker_ids) or not trajectories:
        raise ValueError("logprob alignment requires one worker id per trajectory")
    errors: list[tuple[float, str, int, float, float, int, str]] = []
    per_worker: dict[str, list[float]] = {}
    validated_rows = []
    for trajectory, worker_id in zip(trajectories, worker_ids):
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("logprob alignment trajectory omitted its Worker id")
        input_ids = trajectory["input_ids"]
        policy_mask = trajectory["policy_mask"]
        behavior = trajectory["behavior_logprobs"]
        if len(input_ids) != len(policy_mask) or len(input_ids) != len(behavior):
            raise ValueError("logprob alignment token, mask, and behavior arrays are misaligned")
        validated_rows.append({"input_ids": input_ids, "policy_mask": policy_mask})
    recomputed_rows = [
        engine.logprobs(row["input_ids"], row["policy_mask"])
        for row in validated_rows
    ]
    for trajectory_index, (trajectory, worker_id, recomputed) in enumerate(
        zip(trajectories, worker_ids, recomputed_rows, strict=True)
    ):
        policy_mask = trajectory["policy_mask"]
        behavior = trajectory["behavior_logprobs"]
        if len(recomputed) != len(trajectory["input_ids"]):
            raise ValueError("LearnerEngine.logprobs returned a misaligned token array")
        for position, active in enumerate(policy_mask):
            if not active or position == 0:
                continue
            actual = float(behavior[position])
            current = float(recomputed[position])
            if not math.isfinite(actual) or not math.isfinite(current):
                raise FloatingPointError(
                    f"non-finite logprob for {worker_id} at input position {position}"
                )
            error = abs(current - actual)
            if not math.isfinite(error):
                raise FloatingPointError(
                    f"non-finite logprob error for {worker_id} at input position {position}"
                )
            errors.append((error, worker_id, position, actual, current, trajectory_index, str(trajectory.get("group_id", ""))))
            per_worker.setdefault(worker_id, []).append(error)
    missing_workers = set(worker_ids) - per_worker.keys()
    if missing_workers:
        raise ValueError(
            "logprob alignment found no policy tokens for Worker(s): "
            + ", ".join(sorted(missing_workers))
        )
    if not errors:
        raise ValueError("logprob alignment found no policy tokens")

    thresholds = {
        "mean_abs_error": _LOGPROB_MEAN_ABS_TOLERANCE,
        "max_abs_error": _LOGPROB_MAX_ABS_TOLERANCE,
        "rms_error": _LOGPROB_RMS_TOLERANCE,
    }

    def metrics(values: list[float]) -> dict[str, float | int | str]:
        measured: dict[str, float | int | str] = {
            "tokens": len(values),
            "mean_abs_error": statistics.fmean(values),
            "max_abs_error": max(values),
            "rms_error": math.sqrt(statistics.fmean(value * value for value in values)),
        }
        measured["status"] = (
            "passed"
            if all(float(measured[name]) <= limit for name, limit in thresholds.items())
            else "warning"
        )
        return measured

    aggregate = metrics([item[0] for item in errors])
    by_worker = {worker_id: metrics(values) for worker_id, values in sorted(per_worker.items())}
    worst = max(errors, key=lambda item: item[0])
    warning_metrics = {
        "aggregate": [
            name for name, limit in thresholds.items() if float(aggregate[name]) > limit
        ],
        **{
            worker_id: [
                name for name, limit in thresholds.items()
                if float(worker_metrics[name]) > limit
            ]
            for worker_id, worker_metrics in by_worker.items()
            if worker_metrics["status"] == "warning"
        },
    }
    warning_workers = [worker_id for worker_id, worker_metrics in by_worker.items()
                       if worker_metrics["status"] == "warning"]
    passed = aggregate["status"] == "passed" and all(
        worker_metrics["status"] == "passed" for worker_metrics in by_worker.values()
    )
    warning_errors = [item for item in errors if item[1] in set(warning_workers)]
    worst_warning = max(warning_errors or errors, key=lambda item: item[0])

    def token_record(item: tuple[float, str, int, float, float, int, str]) -> dict[str, Any]:
        return {
            "worker_id": item[1],
            "input_position": item[2],
            "behavior_logprob": item[3],
            "learner_logprob": item[4],
            "absolute_error": item[0],
            "trajectory_index": item[5],
            "group_id": item[6],
        }

    result = {
        **aggregate,
        "status": "passed" if passed else "warning",
        "thresholds": thresholds,
        "by_worker": by_worker,
        "warning_workers": warning_workers,
        "warning_metrics": {name: values for name, values in warning_metrics.items() if values},
        "worst_token": token_record(worst),
        "worst_warning_token": None if passed else token_record(worst_warning),
    }
    return result


def _alignment_from_proximal_scores(
    raw: dict[str, Any] | None,
    worker_ids: list[str],
    *,
    behavior_version: int,
    learner_start_version: int,
) -> dict[str, Any]:
    """Summarize the Trainer's already-computed pre-step policy/behavior deltas."""
    thresholds = {
        "mean_abs_error": _LOGPROB_MEAN_ABS_TOLERANCE,
        "max_abs_error": _LOGPROB_MAX_ABS_TOLERANCE,
        "rms_error": _LOGPROB_RMS_TOLERANCE,
    }
    if not isinstance(raw, dict):
        return {
            "status": "warning",
            "source": "trainer_proximal_scores",
            "behavior_version": behavior_version,
            "learner_start_version": learner_start_version,
            "policy_lag": learner_start_version - behavior_version,
            "tokens": 0,
            "thresholds": thresholds,
            "by_worker": {},
            "warning_workers": [],
            "warning_metrics": {"alignment": ["missing_proximal_metrics"]},
            "error": {
                "type": "AlignmentMetricsMissing",
                "message": "Trainer did not return proximal p-b metrics",
            },
        }

    aggregate = {
        key: raw.get(key)
        for key in ("tokens", "mean_abs_error", "max_abs_error", "rms_error")
    }
    by_worker: dict[str, dict[str, Any]] = {}
    warning_metrics: dict[str, list[str]] = {}

    def breached(metrics: dict[str, Any]) -> list[str]:
        failures = []
        token_count = metrics.get("tokens")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1:
            failures.append("tokens")
        for name, limit in thresholds.items():
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                failures.append(name)
                continue
            try:
                numeric = float(value)
            except (OverflowError, ValueError):
                failures.append(name)
                continue
            if not math.isfinite(numeric) or numeric > limit:
                failures.append(name)
        return failures

    aggregate_breaches = breached(aggregate)
    raw_by_worker = raw.get("by_worker")
    if not isinstance(raw_by_worker, dict):
        raw_by_worker = {}
        warning_metrics["worker_metrics"] = ["missing_or_malformed"]
    for worker_id in sorted(raw_by_worker, key=str):
        source = raw_by_worker[worker_id]
        if not isinstance(worker_id, str) or not worker_id:
            warning_metrics.setdefault("worker_metrics", []).append("invalid_worker_id")
            continue
        if not isinstance(source, dict):
            warning_metrics[worker_id] = ["missing_or_malformed_metrics"]
            continue
        metrics = {
            key: source.get(key)
            for key in ("tokens", "mean_abs_error", "max_abs_error", "rms_error")
        }
        failures = breached(metrics)
        metrics["status"] = "warning" if failures else "passed"
        by_worker[worker_id] = metrics
        if failures:
            warning_metrics[worker_id] = failures

    expected_workers = set(worker_ids)
    missing_workers = sorted(expected_workers - set(by_worker))
    extra_workers = sorted(set(by_worker) - expected_workers) if expected_workers else []
    if aggregate_breaches:
        warning_metrics["aggregate"] = aggregate_breaches
    if missing_workers:
        warning_metrics["missing_workers"] = missing_workers
    if extra_workers:
        warning_metrics["unexpected_workers"] = extra_workers
    warning_workers = sorted(
        worker_id for worker_id, metrics in by_worker.items()
        if metrics["status"] == "warning"
    )
    status = (
        "warning"
        if aggregate_breaches or warning_workers or missing_workers or extra_workers or warning_metrics
        else "passed"
    )
    result = {
        **aggregate,
        "status": status,
        "source": "trainer_proximal_scores_before_optimizer",
        "behavior_version": behavior_version,
        "learner_start_version": learner_start_version,
        "policy_lag": learner_start_version - behavior_version,
        "thresholds": thresholds,
        "by_worker": by_worker,
        "warning_workers": warning_workers,
        "warning_metrics": warning_metrics,
        "worst_token": raw.get("worst_token"),
        "worst_warning_token": raw.get("worst_token") if status == "warning" else None,
    }
    if missing_workers or extra_workers:
        result["worker_coverage"] = {"missing": missing_workers, "unexpected": extra_workers}
    return result


def _alignment_warning_message(phase: str, alignment: dict[str, Any]) -> str:
    worst = alignment.get("worst_warning_token") or alignment.get("worst_token")
    if (not isinstance(worst, dict) or not isinstance(worst.get("worker_id"), str)
            or any(not isinstance(worst.get(key), (int, float)) for key in
                   ("input_position", "behavior_logprob", "learner_logprob", "absolute_error"))):
        return (
            f"{phase} Worker actual-sampling logprob alignment exceeded engineering tolerance; continuing: "
            f"mean_abs={alignment.get('mean_abs_error')} rms={alignment.get('rms_error')} "
            f"max_abs={alignment.get('max_abs_error')} warning_metrics={alignment.get('warning_metrics', {})}"
        )
    return (
        f"{phase} Worker actual-sampling logprob alignment exceeded engineering tolerance; continuing: "
        f"{worst['worker_id']} input_position={worst['input_position']} "
        f"actual={worst['behavior_logprob']:.6f} "
        f"learner={worst['learner_logprob']:.6f} "
        f"abs_error={worst['absolute_error']:.6f}; "
        f"warning_workers={alignment.get('warning_workers', [])} "
        f"warning_metrics={alignment.get('warning_metrics', {})}"
    )


def _alignment_scope(alignment: Any) -> str:
    """Name the evidence path that produced the latest update's alignment record."""
    if isinstance(alignment, dict) and alignment.get("status") == "policy_lag":
        return "policy_lag_status_only"
    if isinstance(alignment, dict) and alignment.get("status") == "skipped":
        return "alignment_not_measured"
    source = alignment.get("source") if isinstance(alignment, dict) else None
    if isinstance(source, str) and source.startswith("trainer_proximal_scores"):
        return "latest_attempt_proximal_scoring"
    return "latest_attempt_pre_alignment"


def _log_alignment_warning(phase: str, alignment: dict[str, Any]) -> None:
    """Log finite tolerance breaches and reject any unexpected measurement status."""
    status = alignment.get("status")
    if status == "warning":
        _LOGGER.warning(_alignment_warning_message(phase, alignment))
    elif status != "passed":
        raise RuntimeError(f"{phase} logprob alignment returned invalid status {status!r}")




class _PolicyContractError(RuntimeError):
    """A Worker changed or omitted the policy identity of a pinned episode."""


def _assert_worker_sampling_contract(
    worker_id: str,
    sampling_config: Any,
    rollout_config: dict[str, Any],
    *,
    backend: str,
    model_vocab_size: int | None,
    tokenizer_length: int,
    sampled_token_ids: list[int],
) -> None:
    """Check the current Worker receipt against the actual sampling and token domain."""
    if not isinstance(sampling_config, dict):
        raise TypeError(f"Worker {worker_id} omitted its actual sampling receipt")
    expected = {
        "distribution": "grammar_conditioned_policy/v1" if rollout_config.get("constrained_decoding",False) else "raw_policy",
        "temperature": float(rollout_config["temperature"]),
        "top_p": float(rollout_config["top_p"]),
        "top_k": int(rollout_config["top_k"]),
        "repetition_penalty": float(rollout_config["repetition_penalty"]),
        "warpers_neutralized": True,
    }
    if any(sampling_config.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"Worker {worker_id} did not prove the configured policy distribution and neutral sampler settings")
    for name in ("temperature", "top_p", "repetition_penalty"):
        value = sampling_config.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise RuntimeError(f"Worker {worker_id} sampling receipt has invalid {name}")
    top_k = sampling_config.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"Worker {worker_id} sampling receipt has invalid top_k")
    neutralized = sampling_config.get("neutralized_fields")
    required_neutralized = {
        "typical_p", "min_p", "epsilon_cutoff", "eta_cutoff",
        "no_repeat_ngram_size", "sequence_bias", "suppress_tokens",
    }
    if (not isinstance(neutralized, list)
            or any(not isinstance(name, str) for name in neutralized)
            or not required_neutralized.issubset(neutralized)):
        raise RuntimeError(f"Worker {worker_id} did not attest all neutralized sampler fields")

    if backend == "ninfer":
        logical_vocab = sampling_config.get("logical_model_vocab_size")
        public_tokens = sampling_config.get("public_token_count")
        expected_behavior = "grammar_conditioned_policy/v1" if rollout_config.get("constrained_decoding",False) else "raw_policy_public_vocab"
        if sampling_config.get("behavior_distribution") != expected_behavior:
            raise RuntimeError(f"NInfer Worker {worker_id} did not attest its actual public-vocabulary sampler")
        if (isinstance(logical_vocab, bool) or not isinstance(logical_vocab, int)
                or logical_vocab <= 0 or logical_vocab != model_vocab_size):
            raise RuntimeError(f"NInfer Worker {worker_id} logical vocabulary differs from the loaded Learner model")
        if (isinstance(public_tokens, bool) or not isinstance(public_tokens, int)
                or not 0 < public_tokens <= logical_vocab or public_tokens != tokenizer_length):
            raise RuntimeError(f"NInfer Worker {worker_id} public vocabulary differs from the loaded tokenizer")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < public_tokens
            for token in sampled_token_ids
        ):
            raise RuntimeError(f"NInfer Worker {worker_id} sampled a token outside the current public tokenizer")
        return

    if backend != "vllm":
        raise RuntimeError(f"Worker {worker_id} uses unsupported sampling backend {backend!r}")
    behavior = sampling_config.get("behavior_distribution")
    if behavior not in (None, "raw_policy"):
        raise RuntimeError(f"Worker {worker_id} did not attest raw-policy sampling")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < tokenizer_length
        for token in sampled_token_ids
    ):
        raise RuntimeError(f"Worker {worker_id} sampled a token outside the current tokenizer")


async def _generate_scored_group(
    engine: Any,
    managers: dict[str, Any],
    config: dict[str, Any],
    task: Task,
    group_id: str,
    policy: PolicySnapshot,
    *,
    worker_pool: EpisodeWorkerPool,
    diagnostic_course: str | None = None,
    plan: str | None = None,
    group_size_override: int | None = None,
    validation: bool = False,
    evaluation_seed: int | None = None,
    validation_low_priority: bool = True,
    owner_worker_id: str | None = None,
    guided_plan: str | None = None,
) -> dict[str, Any]:
    """Generate and ARC-score a fresh complete group through the production AgentLoop path."""
    workers = config["workers"]
    if validation and (plan is not None or group_size_override != 1 or evaluation_seed is None):
        raise ValueError("evaluation requires one native episode, a fixed seed, and no teacher plan")
    if owner_worker_id is not None and (
        not isinstance(owner_worker_id, str) or not owner_worker_id
    ):
        raise ValueError("owner_worker_id must be a non-empty configured Worker ID")
    if guided_plan is not None and (
        not isinstance(guided_plan, str) or not guided_plan.strip()
    ):
        raise ValueError("guided_plan must be a non-empty verified guidance string")
    rollout_config = (_identity_course_config(config)
                      if diagnostic_course == _IDENTITY_DIAGNOSTIC_COURSE else config)
    if diagnostic_course not in {None, _IDENTITY_DIAGNOSTIC_COURSE}:
        raise ValueError(f"unsupported diagnostic course: {diagnostic_course}")
    counters = {worker["id"]: {"calls": 0, "seconds": 0.0} for worker in workers}
    sampling = {
        "temperature": float(config["rollout"]["temperature"]),
        "top_p": float(config["rollout"]["top_p"]),
        "top_k": int(config["rollout"]["top_k"]),
        "repetition_penalty": float(config["rollout"]["repetition_penalty"]),
    }
    group_size = group_size_override or int(config["rollout"]["group_size"])
    episode_group_size = group_size + int(guided_plan is not None)
    expected_version = policy.version
    if diagnostic_course == _IDENTITY_DIAGNOSTIC_COURSE:
        prompt = _identity_course_messages(task)
    else:
        from ..prompt import build_task_messages

        prompt = build_task_messages(task)
    if plan is not None:
        prompt[0]["content"] += "\n\nVerified teacher guidance (training only):\n" + plan
    template_loop = _agent_loop(
        engine, managers[workers[0]["id"]], rollout_config, group_size=episode_group_size,
    )
    initial_prompt = await template_loop.build_initial_prompt_ids(prompt)
    guided_prompt = None
    guided_initial_prompt = None
    if guided_plan is not None:
        guided_prompt = [dict(message) for message in prompt]
        guided_prompt[0]["content"] += (
            "\n\nVerified teacher guidance (training only):\n" + guided_plan
        )
        guided_initial_prompt = await template_loop.build_initial_prompt_ids(guided_prompt)
    queued = owner_worker_id is not None and getattr(worker_pool, "episode_queue", None) is not None and not validation
    leased_worker_id: str | None = owner_worker_id

    async def generate_sample(sample_no, *, guided: bool = False):
        # Hold placement across every tool turn, including observation waits.
        sample_prompt = guided_prompt if guided else prompt
        sample_initial_prompt = guided_initial_prompt if guided else initial_prompt
        if sample_prompt is None or sample_initial_prompt is None:
            raise RuntimeError("guided sample was scheduled without a verified teacher plan")
        lease_options = {'low_priority': validation_low_priority} if validation else {}
        if (owner_worker_id is None and not validation
                and config['rollout'].get('helper_worker_ids')):
            owners = tuple(config['learner']['gradient_executor_worker_ids'].values())
            lease_options['preferred_worker_id'] = owners[sample_no % len(owners)]
        if leased_worker_id is not None:
            lease_options.update(
                preferred_worker_id=leased_worker_id,
                group_id=group_id,
            )
        placement = (worker_pool.queued_episode(
            f"{group_id}:{sample_no}", owner=owner_worker_id,
            policy=(policy.version, policy.adapter_sha256), cost_key=task.task_id, allow_help=not guided,
        ) if queued else worker_pool.lease(**lease_options))
        async with placement as worker_id:
            manager = managers[worker_id]
            episode_id = f"{group_id}-{sample_no}-{uuid.uuid4().hex}"
            acquire_episode = getattr(manager, "acquire_episode", None)
            release_episode = getattr(manager, "release_episode", None)
            if callable(acquire_episode) != callable(release_episode):
                raise RuntimeError("Worker manager must provide both episode lease methods")
            lease_acquired = False
            if callable(acquire_episode):
                lease_kwargs = {'purpose': 'evaluation'} if validation else {}
                await acquire_episode(
                    episode_id=episode_id,
                    policy_version=policy.version,
                    adapter_hash=policy.adapter_sha256,
                    **lease_kwargs,
                )
                lease_acquired = True
            async def generation_finished():
                nonlocal lease_acquired
                # The trusted ARC service accepted terminal finish: this
                # episode can no longer generate, but its group may await score.
                if lease_acquired:
                    await release_episode(
                        episode_id=episode_id, policy_version=policy.version,
                        adapter_hash=policy.adapter_sha256,
                    )
                    lease_acquired = False
                await worker_pool.finish_episode_generation(f"{group_id}:{sample_no}")

            async def rebalance(tracking):
                nonlocal worker_id, manager, acquire_episode, release_episode
                key = f"{group_id}:{sample_no}"
                target = await worker_pool.rebalance_target(key, engine.store)
                if target is None:
                    return

                async def handoff():
                    nonlocal worker_id, manager, acquire_episode, release_episode
                    source = worker_id
                    destination = managers[target]
                    source_release_started = False
                    try:
                        await destination.acquire_episode(
                            episode_id=episode_id, policy_version=policy.version,
                            adapter_hash=policy.adapter_sha256,
                        )
                        source_release_started = True
                        await release_episode(
                            episode_id=episode_id, policy_version=policy.version,
                            adapter_hash=policy.adapter_sha256,
                        )
                        await worker_pool.commit_episode_transfer(key, source, target)
                    except BaseException:
                        if source_release_started:
                            worker_pool.episode_queue.abort_transfer(key, uncertain_worker=source)
                        try:
                            await destination.release_episode(
                                episode_id=episode_id, policy_version=policy.version,
                                adapter_hash=policy.adapter_sha256,
                            )
                        except BaseException:
                            worker_pool.episode_queue.abort_transfer(key, uncertain_worker=target)
                        else:
                            worker_pool.episode_queue.abort_transfer(key)
                        raise
                    worker_id, manager = target, destination
                    acquire_episode, release_episode = destination.acquire_episode, destination.release_episode
                    tracking.worker_id, tracking.manager = target, destination
                    tracking.counters = counters[target]
                    tracking.worker_history.append(target)

                # Cancellation must join acquire/release and central role bookkeeping.
                from .gradient_executor.client import _join_owned_task
                await _join_owned_task(asyncio.create_task(handoff()))

            primary_error = None
            try:
                tracking = _TrackingManager(
                    worker_id, manager, counters[worker_id], policy, episode_id,
                    evaluation_seed=evaluation_seed,
                    generation_finished=generation_finished if queued else None,
                    before_generation=rebalance if queued else None,
                )
                loop = _agent_loop(engine, tracking, rollout_config, group_size=episode_group_size)
                output = await loop.run(
                    sampling,
                    raw_prompt=sample_prompt,
                    initial_prompt_ids=sample_initial_prompt,
                    uid=group_id,
                    session_id=str(sample_no),
                    extra_info={
                        "task_id": task.task_id,
                        "validation": validation,
                        "guided_rescue": guided,
                    },
                    validate=validation,
                )
                return worker_id, output, list(tracking.worker_history)
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                if lease_acquired:
                    from ..worker.errors import WorkerUnavailableError
                    try:
                        await release_episode(
                            episode_id=episode_id,
                            policy_version=policy.version,
                            adapter_hash=policy.adapter_sha256,
                        )
                    except WorkerUnavailableError:
                        if worker_pool is not None:
                            await worker_pool.quarantine_worker(worker_id)
                        if primary_error is None:
                            raise

    async def collect_samples():
        from ..rollout.group_collection import collect_group_members

        async def quarantine(failure):
            if (worker_pool is not None and failure.worker_id is not None
                    and failure.quarantinable):
                await worker_pool.quarantine_worker(failure.worker_id)

        # Queue the owner-only guided member first, while preserving native-first
        # result order and the original per-sample identities.
        guided_task = (asyncio.create_task(generate_sample(group_size, guided=True))
                       if guided_plan is not None else None)
        native_tasks = [asyncio.create_task(generate_sample(sample_no))
                        for sample_no in range(group_size)]
        ordered = [*native_tasks]
        if guided_task is not None:
            ordered.append(guided_task)
        return await collect_group_members(ordered, on_unavailable=quarantine)

    if owner_worker_id is None or queued:
        results = await collect_samples()
    else:
        # The group lease reserves the whole Worker once. Individual
        # trajectories take only pinned episode leases within that reservation;
        # nesting another group lease here would deadlock on the same Worker.
        async with worker_pool.group_lease(
            group_id,
            worker_ids=(owner_worker_id,),
            preferred_worker_id=owner_worker_id,
            owner_by_worker={owner_worker_id: task.task_id},
        ) as leased_worker_id:
            results = await collect_samples()
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        from ..worker.client import GenerationContractError
        fatal = next((error for error in errors
                      if isinstance(error, (GenerationContractError, _PolicyContractError))), None)
        if fatal is not None:
            raise fatal
        from ..worker.errors import worker_unavailable
        unavailable = [failure for error in errors if (failure := worker_unavailable(error)) is not None]
        if unavailable:
            from .run.groups import GroupInfrastructureError
            raise GroupInfrastructureError(
                "complete group discarded after Worker infrastructure failure",
                worker_ids=tuple(sorted({error.worker_id for error in unavailable if error.worker_id is not None})),
            ) from unavailable[0]
        raise GroupGenerationError(
            f"complete group failed: {type(errors[0]).__name__}: {str(errors[0])[:300]}"
        ) from errors[0]
    all_assignments = [result[0] for result in results]
    if owner_worker_id is not None and not queued and set(all_assignments) != {owner_worker_id}:
        raise RuntimeError(
            f"complete group {group_id} escaped its persisted Worker owner {owner_worker_id}"
        )
    native_assignments = all_assignments[:group_size]
    outputs = [result[1] for result in results[:group_size]]
    guided_worker_id = all_assignments[group_size] if guided_plan is not None else None
    guided_output = results[group_size][1] if guided_plan is not None else None
    if len(outputs) != group_size:
        raise RuntimeError("scored group does not contain its configured candidate count")

    trajectories: list[dict[str, Any]] = []
    worker_ids: list[str] = list(native_assignments)
    versions: list[int] = []
    rewards: list[float] = []
    exacts: list[bool] = []
    costs: list[float] = []
    sampling_configs: dict[str, dict[str, Any]] = {}
    guided_trajectory = None

    def trajectory_from_output(output, worker_id: str, *, guided: bool):
        fields = output.extra_fields
        minimum = fields.get("min_global_steps")
        maximum = fields.get("max_global_steps")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum != maximum:
            raise RuntimeError("post-update ArcAgentLoop returned missing or mixed policy versions")
        versions.append(minimum)
        sampling_config = counters[worker_id].get("sampling_config")
        prompt_ids = list(output.prompt_ids)
        response_ids = list(output.response_ids)
        worker_backend = config.get("worker_inference", {}).get("backend", "vllm")
        model_vocab_size = None
        if worker_backend == "ninfer":
            model_vocab_size = getattr(engine, "evidence", {}).get("vocab_size")
            if isinstance(model_vocab_size, bool) or not isinstance(model_vocab_size, int):
                base_model = engine.model.get_base_model()
                model_vocab_size = getattr(getattr(base_model, "config", None), "vocab_size", None)
        _assert_worker_sampling_contract(
            worker_id,
            sampling_config,
            config["rollout"],
            backend=worker_backend,
            model_vocab_size=model_vocab_size,
            tokenizer_length=len(engine.tokenizer),
            sampled_token_ids=response_ids,
        )
        sampling_configs[worker_id] = sampling_config
        response_mask = [int(value) for value in output.response_mask]
        response_logprobs = [float(value) for value in output.response_logprobs]
        if len(response_ids) != len(response_mask) or len(response_ids) != len(response_logprobs):
            raise RuntimeError("post-update token, mask, and actual sampling logprobs are misaligned")
        reward = float(output.reward_score)
        if not math.isfinite(reward):
            raise RuntimeError("trusted ARC scorer returned a non-finite post-update reward")
        reward_info = fields.get("reward_extra_info", {})
        exact = reward_info.get("exact")
        cost = reward_info.get("cost")
        if not isinstance(exact, bool) or isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise RuntimeError("trusted ARC result omitted post-update exact/cost fields")
        return {
            "row_kind": "guided_candidate" if guided else "fresh_student",
            "generation_worker_id": worker_id,
            **({"source_worker_id": owner_worker_id} if queued else {}),
            "input_ids": prompt_ids + response_ids,
            "policy_mask": [0] * len(prompt_ids) + response_mask,
            "behavior_logprobs": [0.0] * len(prompt_ids) + response_logprobs,
            "reward": reward,
            "group_id": group_id,
            "policy_version": minimum,
            "episode_id": fields.get("episode_id"),
            "messages": fields.get("messages", []),
            "tool_events": fields.get("tool_events", []),
            **({"constraint_segments":fields["constraint_segments"]} if fields.get("constraint_segments") else {}),
            "exact": exact, "cost": float(cost),
            "generated_tokens": reward_info.get("generated_tokens"),
            "observation_tokens": reward_info.get("observation_tokens"),
            "successful_tool_calls": reward_info["successful_tool_calls"],
            "tool_call_failures": reward_info["tool_call_failures"],
            "termination_reason": reward_info.get("termination_reason"),
            "training_verification": reward_info["training_verification"],
            "reward_components": reward_info["reward_components"],
            "reward_schema": reward_info["reward_schema"],
            "attempt_exact": reward_info.get("attempt_exact"),
            "per_target_exact": reward_info.get("per_target_exact"),
            "competition_score": reward_info.get("competition_score"),
        }, reward, exact, float(cost)

    for sample_no, output in enumerate(outputs):
        row, reward, exact, cost = trajectory_from_output(
            output, native_assignments[sample_no], guided=False,
        )
        row["generation_worker_history"] = results[sample_no][2]
        trajectories.append(row)
        rewards.append(reward)
        exacts.append(exact)
        costs.append(cost)
    if guided_output is not None and guided_worker_id is not None:
        guided_trajectory, guided_reward, guided_exact, guided_cost = trajectory_from_output(
            guided_output, guided_worker_id, guided=True,
        )
    else:
        guided_reward = guided_exact = guided_cost = None
    if versions != [expected_version] * len(results):
        raise RuntimeError("complete group did not use its fixed behavior policy")
    prompt_prefix_hash = _same_initial_prompt_prefix_sha256(trajectories)
    return {
        "outputs": outputs,
        "trajectories": trajectories,
        "worker_ids": worker_ids,
        "generation_worker_ids": native_assignments,
        "owner_worker_id": owner_worker_id,
        "counters": counters,
        "rewards": rewards,
        "exact": exacts,
        "costs": costs,
        "sampling_configs": sampling_configs,
        "initial_prompt_prefix_sha256": prompt_prefix_hash,
        "policy_version": expected_version,
        "guided_trajectory": guided_trajectory,
        "guided_worker_id": guided_worker_id,
        "guided_output": guided_output,
        "guided_reward": guided_reward,
        "guided_exact": guided_exact,
        "guided_cost": guided_cost,
    }


def create_learner_app(config: Any, *, engine: Any = None,
                       worker_managers: dict[str, Any] | None = None,
                       arc_app: FastAPI | None = None,
                       environment: Environment[Task] | None = None,
                       reward_scorer: RewardScorer[Task] | None = None) -> FastAPI:
    """Create the Learner API with an explicit rollout environment boundary.

    Omitting ``environment`` selects the established ARC adapter. Supplying an
    environment and scorer avoids importing the ARC source, service, prompt, or
    AgentLoop; ARC-only BEPA and held-out evaluation remain adapter-specific.
    """
    cfg = _cfg_dict(config)
    policy_read_authorizer = PolicyReadAuthorizer.from_workers(cfg["workers"])
    eval_settings = cfg.get('evaluation', {})
    generic_environment = environment is not None
    if generic_environment:
        if reward_scorer is None:
            raise ValueError("a custom environment requires a reward_scorer")
        if arc_app is not None:
            raise ValueError("arc_app cannot be combined with a custom environment")
        if cfg.get("bepa", {}).get("enabled"):
            raise ValueError("BEPA currently requires the ARC environment adapter")
        if eval_settings.get("enabled"):
            raise ValueError("held-out evaluation currently requires the ARC environment adapter")
        arc_contract = None
        evaluation_dataset = None
    else:
        evaluation_dataset = EvaluationCoordinator.load_dataset(cfg)
        if arc_app is None:
            task_map, arc_contract = _load_arc_source(cfg)
        else:
            task_map = getattr(arc_app.state, "tasks", None)
            if task_map is None:
                task_map = {}
            elif not isinstance(task_map, dict):
                raise TypeError("injected ARC app state.tasks must be a dictionary")
            if engine is None:
                from ..budget import load_contract

                arc_contract = load_contract(cfg["arc"]["contract_path"])
            else:
                arc_contract = None
        if environment is None:
            from ..integrations.arc.environment import ArcEnvironment, ArcRewardScorer
            from ..prompt import prompt_identity
            from ..rewards import REWARD_SCHEMA

            environment = ArcEnvironment(
                tasks=task_map,
                fixture=_identity_fixture(),
                generator=lambda *args, **kwargs: _generate_scored_group(*args, **kwargs),
                prompt_identity=prompt_identity,
                reward_schema=REWARD_SCHEMA,
            )
            reward_scorer = ArcRewardScorer()
    if environment is None or reward_scorer is None:
        raise RuntimeError("Learner environment and reward scorer were not initialized")
    environment = validate_environment(environment)
    reward_scorer = validate_reward_scorer(reward_scorer)
    task_map = dict(environment.tasks)
    if generic_environment:
        arc_contract = None
    load_started = time.perf_counter()
    recovery_semantics = (
        _recovery_semantics(cfg, task_map, arc_contract)
        if engine is None and not generic_environment else None
    )
    engine = engine if engine is not None else _load_engine(cfg, recovery_semantics=recovery_semantics)
    engine_load_seconds = time.perf_counter() - load_started
    client_init_started = time.perf_counter()
    managers = worker_managers if worker_managers is not None else _load_worker_managers(cfg)
    helper_worker_ids = tuple(cfg['rollout'].get('helper_worker_ids', ()))
    resource_probe = None
    managers_have_health = managers and all(
        callable(getattr(manager, "get_health", None)) for manager in managers.values()
    )
    if worker_managers is None and not managers_have_health:
        raise ValueError("production Worker managers must expose authenticated get_health() probes")
    if managers_have_health:
        async def resource_probe(worker_id: str):
            if worker_id in helper_worker_ids:
                # Optional WAN helpers may be offline; their probes must not
                # inherit the long generation timeout of cloud owner Workers.
                try:
                    async with asyncio.timeout(1.0):
                        return await managers[worker_id].get_health()
                except TimeoutError as exc:
                    from ..worker.errors import WorkerUnavailableError
                    raise WorkerUnavailableError(
                        "Worker health probe timed out", worker_id=worker_id,
                    ) from exc
            return await managers[worker_id].get_health()
    member_epochs = {
        worker["id"]: f"{worker['boot_id']}:{worker['telemetry_epoch']}"
        for worker in cfg["workers"]
        if worker.get("scope") in {"gateway", "service"}
    }
    worker_pool = EpisodeWorkerPool(
        [worker["id"] for worker in cfg["workers"]],
        capacity=complete_group_episode_capacity(cfg),
        group_capacity=int(cfg["learner"].get("gradient_tasks_per_worker", 1)),
        resource_probe=resource_probe,
        member_epochs=member_epochs,
        initially_paused=helper_worker_ids,
    )
    from ..worker.episode_queue import EpisodeRoleQueue, episode_queue_enabled
    if episode_queue_enabled(cfg["learner"], helper_worker_ids=helper_worker_ids):
        store = engine.store
        pairs = cfg["learner"]["gradient_executor_worker_ids"]
        from qwen_rl.contracts.gradient_partitions import local_executor_pools
        if cfg["learner"].get("gradient_remote_executor_ids"):
            local_pools = local_executor_pools(cfg)
            pools = {worker: local_pools.get(executor, worker) for executor, worker in pairs.items()}
        else:
            pools = {worker: store.work_stealing_pools[executor] for executor, worker in pairs.items()}
        external_owners = set(cfg["learner"].get("sampling_only_worker_ids", {}).values())
        pools.update({worker: "sampling-only/" + worker for worker in external_owners})
        for first, second in cfg["learner"].get("sampling_shared_worker_pairs", {}).items():
            pools[first] = pools[second] = "sampling-pair/" + first
        helper_only = tuple(worker for worker in helper_worker_ids if worker not in external_owners)
        scheduler = EpisodeRoleQueue(
            pools, {worker: int(cfg["learner"].get("sampling_episode_slots", {}).get(
                        worker, cfg["worker_inference"]["max_num_seqs"]))
                    for worker in (*pools, *helper_only)},
            helper_worker_ids=helper_only, external_owner_ids=external_owners,
            episode_seconds=cfg["learner"].get("episode_duration_prior_seconds", 120.0),
            switch_seconds=cfg["learner"].get("gpu_role_switch_seconds", 35.0),
            margin_seconds=cfg["learner"].get("gpu_role_switch_margin_seconds", 10.0),
        )
        # Outstanding sessions are fail-closed across controller construction.
        for session in store.active_compute_sessions():
            if session['executor_id'] in pairs:
                scheduler.compute[pairs[session['executor_id']]] = session['session_id']
        worker_pool.episode_queue = store.episode_queue = scheduler
    worker_clients_init_seconds = time.perf_counter() - client_init_started
    arc_service_init_seconds = 0.0
    if not generic_environment:
        arc_init_started = time.perf_counter()
        if arc_app is None:
            service_tasks = dict(task_map)
            if evaluation_dataset is not None:
                EvaluationCoordinator.validate_task_separation(
                    task_map.values(),
                    evaluation_dataset,
                )
                for task in evaluation_dataset.tasks:
                    service_tasks[task.task_id] = task
            arc_app = _load_arc_service(cfg, engine, tasks=service_tasks, contract=arc_contract)
        arc_service_init_seconds = time.perf_counter() - arc_init_started
    evidence = _validate_engine_evidence(engine, cfg)

    state_lock = asyncio.Lock()
    update_lock = asyncio.Lock()
    compute_owner = ComputeOwner()

    def recovered_gradient_step_callback(window):
        context = window.update_context if isinstance(window.update_context, Mapping) else {}
        attempt_id = context.get("bepa_attempt_id")
        if attempt_id is None:
            return None
        if not isinstance(attempt_id, str) or not attempt_id:
            raise RuntimeError("recovered central window has an invalid BEPA attempt identity")
        if bepa is None:
            raise RuntimeError("recovered central window names a BEPA attempt but BEPA is disabled")

        def mark_optimizer_entered() -> None:
            bepa.store.mark_optimizer_entered(
                attempt_id,
                current_policy_version=int(window.policy_version),
            )

        return mark_optimizer_entered

    app = FastAPI(title="Qwen Learner", version="1.0")
    app.state.config = cfg
    app.state.engine = engine
    app.state.environment = environment
    gradient_http_app = getattr(engine, "gradient_http_app", None)
    if gradient_http_app is not None:
        app.state.gradient_engine = engine
        app.mount("/gradient", gradient_http_app, name="gradient-executors")
    app.state.worker_managers = managers
    app.state.worker_membership = {
        worker["id"]: {
            "scope": worker.get("scope", "local"),
            "site_id": worker.get("site_id", cfg.get("node_scope", {}).get("site_id", "central")),
            "node_id": worker.get("node_id", cfg.get("node_scope", {}).get("node_id", "central")),
            "boot_id": worker.get("boot_id") or None,
            "telemetry_epoch": worker.get("telemetry_epoch") or None,
            "model_sha256": worker.get("model_sha256") or None,
        }
        for worker in cfg["workers"]
    }
    app.state.engine_load_seconds = engine_load_seconds
    app.state.timings = {"learner": {"model_load_seconds": engine_load_seconds,
                                    "update_seconds": 0.0,
                                    "publish_seconds": 0.0, "worker_sync_seconds": 0.0},
                        "runtime": {"worker_client_init_seconds": worker_clients_init_seconds,
                                    "arc_service_init_seconds": arc_service_init_seconds}}
    app.state.runs: dict[str, dict[str, Any]] = {}
    app.state.latest_run_id: str | None = None
    app.state.active_task: asyncio.Task | None = None
    app.state.stop_wait_task: asyncio.Task | None = None
    app.state.active_run_id: str | None = None
    app.state.policies: dict[int, dict[str, Any]] = {}
    app.state.policy_refs: dict[int, int] = {}
    app.state.policy_file_references = PolicyFileReferences()
    app.state.serve_policy_version: int | None = None
    app.state.last_policy_acks: dict[str, dict[str, Any]] = {}
    app.state.training_telemetry = TrainingTelemetry()
    app.state.active_sampling_groups = 0
    app.state.rollout_stream: AsyncGroupStream | None = None
    app.state.seed_scheduler: RunSeedScheduler[Any] | None = None
    app.state.monitor_phase = "idle"
    app.state.foreground_activity = ForegroundActivity()
    app.state.run_commit_view = RunCommitView()
    app.state.external_stage_timings = {}
    app.state.evaluation = None
    app.state.evaluation_dataset = evaluation_dataset
    app.state.evaluation_coordinator = None
    app.state.recovery_initialized = False
    app.state.recovery_resolution: dict[str, Any] | None = None
    app.state.resumed_run_id: str | None = None
    bepa = None
    if cfg.get("bepa", {}).get("enabled"):
        from ..bepa.coordinator import BepaCoordinator
        bepa = BepaCoordinator(cfg, engine, task_map)
        registration_context = bepa._registration_context()
        for task in task_map.values():
            if task.split == "train" and task.pool != "diagnostic":
                bepa.register(task)
        if bepa._registration_context() != registration_context:
            raise RuntimeError("BEPA registration contracts changed during startup")
        bepa._registered_context_guard = registration_context
    app.state.bepa = bepa

    def set_monitor_phase(phase: str) -> None:
        app.state.monitor_phase = phase
        app.state.foreground_activity.transition(phase)

    def engine_recovery_metadata() -> dict[str, Any] | None:
        value = getattr(engine, "recovery_metadata", None)
        return value if isinstance(value, dict) else None

    recovery_identity = engine_recovery_metadata() or {}
    app.state.stop_control = StopAfterCheckpointControl(
        Path(cfg["run_dir"]) / "control" / "stop-after-checkpoint.json",
        lineage_id=recovery_identity.get("lineage_id"),
        engine_epoch=recovery_identity.get("engine_epoch"),
    )

    def recovery_is_dirty() -> bool:
        metadata = engine_recovery_metadata()
        if metadata is None:
            return False
        if metadata.get("dirty") is True:
            return True
        durable = metadata.get("durable_policy_version")
        recorded = metadata.get("policy_version")
        if (isinstance(durable, int) and not isinstance(durable, bool)
                and isinstance(recorded, int) and not isinstance(recorded, bool)):
            return durable != int(engine.policy_version) or recorded != int(engine.policy_version)
        return False

    def checkpoint_save_in_progress(metadata: dict[str, Any] | None) -> bool:
        if not isinstance(metadata, dict):
            return False
        live = metadata.get("policy_version")
        durable = metadata.get("durable_policy_version")
        return (
            cfg.get("learner", {}).get("update_backend") == "central_gradient"
            and metadata.get("enabled") is True
            and metadata.get("dirty") is False
            and metadata.get("checkpoint_state") == "saving"
            and metadata.get("can_publish") is True
            and type(live) is int and type(durable) is int
            and live == durable + 1 and live == int(engine.policy_version)
        )

    def recovery_summary() -> dict[str, Any] | None:
        metadata = engine_recovery_metadata()
        if metadata is None:
            return None
        context = metadata.get("update_context")
        logical_job = context.get("logical_job") if isinstance(context, dict) else None
        pending_run = None
        completed_source = None
        if isinstance(logical_job, dict):
            total = logical_job.get("requested_steps")
            completed = logical_job.get("completed_updates")
            run_id = logical_job.get("run_id")
            if (isinstance(total, int) and not isinstance(total, bool)
                    and isinstance(completed, int) and not isinstance(completed, bool)
                    and isinstance(run_id, str) and run_id):
                remaining = max(0, total - completed)
                progress = {
                    "run_id": run_id,
                    "kind": logical_job.get("kind"),
                    "requested_steps": total,
                    "completed_updates": completed,
                    "remaining_steps": remaining,
                    "task_ids": logical_job.get("task_ids"),
                    "task_cursor": context.get("task_cursor"),
                    "source_identity": context.get("source_identity"),
                    "task_worker_owners": context.get("task_worker_owners", {}),
                    "rescue_ready_task_ids": context.get("rescue_ready_task_ids", []),
                }
                if remaining:
                    pending_run = progress
                else:
                    completed_source = {**progress, "status": "complete"}
        return {
            "available": True,
            "mode": metadata.get("mode"),
            "resumed": metadata.get("resumed") is True,
            "lineage_id": metadata.get("lineage_id"),
            "engine_epoch": metadata.get("engine_epoch"),
            "policy_version": metadata.get("policy_version"),
            "durable_policy_version": metadata.get("durable_policy_version"),
            "dirty": recovery_is_dirty() and not checkpoint_save_in_progress(metadata),
            "checkpoint_state": metadata.get("checkpoint_state"),
            "can_publish": metadata.get("can_publish"),
            "can_optimize": metadata.get("can_optimize"),
            "checkpoint_intent": metadata.get("checkpoint_intent"),
            "snapshot_sha256": metadata.get("snapshot_sha256"),
            "loaded_snapshot": metadata.get("loaded_snapshot"),
            "pending_run": pending_run,
            "completed_source": completed_source,
            "bepa": app.state.recovery_resolution,
        }

    def make_recovery_context(*, run_id: str, request: RunRequest, state: dict[str, Any],
                              selected_tasks: list[Task], next_completed_updates: int,
                              next_task_index: int, next_group_index: int | None = None,
                              bepa_attempt=None) -> dict[str, Any] | None:
        metadata = engine_recovery_metadata()
        if metadata is None:
            return None
        task_ids = [task.task_id for task in selected_tasks]
        semantics = recovery_semantics or {}
        return {
            "schema": "qwen-rl-recovery-context-v1",
            "lineage_id": metadata.get("lineage_id"),
            "engine_epoch": (bepa.epoch if bepa is not None else metadata.get("engine_epoch")),
            "new_policy_version": int(engine.policy_version) + 1,
            "logical_job": {
                "run_id": run_id,
                "kind": request.kind,
                "requested_steps": int(state["requested_steps"]),
                "completed_updates": next_completed_updates,
                "task_ids": task_ids,
            },
            "task_cursor": {
                "next_task_index": next_task_index,
                "next_group_index": (
                    next_task_index if next_group_index is None else next_group_index
                ),
                "source": "fresh_rollouts_after_committed_update",
            },
            "task_worker_owners": dict(state.get("task_worker_owners", {})),
            "rescue_ready_task_ids": list(state.get("rescue_ready_task_ids", [])),
            "bepa_task_sampling": copy.deepcopy(state.get("bepa_task_sampling")),
            "task_prompt": _current_prompt_identity(environment),
            "reward_schema": environment.reward_schema,
            "category_sampling": copy.deepcopy(state.get('category_sampling')),
            "source_sampling": copy.deepcopy(state.get("source_sampling")),
            "source_identity": {
                "environment": environment.name,
                **({"adapter_source_identity": environment.source_identity}
                   if generic_environment else {}),
                **({"dataset_dir": str(Path(cfg["dataset_dir"]).resolve())}
                   if cfg.get("dataset_dir") else {}),
                **semantics,
            },
            "bepa_attempt_id": getattr(bepa_attempt, "attempt_id", None),
            "bepa_batch_id": getattr(bepa_attempt, "batch_id", None),
            "input_batch_hash": getattr(bepa_attempt, "input_batch_hash", None),
            "lineage_successful_training_updates": (
                evaluation_coordinator.count + int(request.kind == 'training')
                if evaluation_coordinator.count is not None else
                ((metadata['update_context']['lineage_successful_training_updates']
                  + int(request.kind == 'training'))
                 if type((metadata.get('update_context') or {}).get('lineage_successful_training_updates')) is int
                 else (int(request.kind == 'training') if int(engine.policy_version) == 0 else None))),
        }

    publication_lock = asyncio.Lock()

    def _validate_publication_state(version, already_published, pending_checkpoint):
        if engine.policy_version != version:
            raise RuntimeError('Learner policy advanced during publication')
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise RuntimeError("LearnerEngine.policy_version must be a non-negative integer before publication")
        recovery_metadata = engine_recovery_metadata()
        recovery_enabled = (
            getattr(engine, "recovery_store", None) is not None
            or (isinstance(recovery_metadata, dict) and recovery_metadata.get("enabled") is True)
        )
        if recovery_enabled:
            durable_version = recovery_metadata.get("durable_policy_version") if isinstance(
                recovery_metadata, dict
            ) else None
            recorded_version = recovery_metadata.get("policy_version") if isinstance(
                recovery_metadata, dict
            ) else None
            pending_publication = (
                pending_checkpoint
                and already_published is not None
                and checkpoint_save_in_progress(recovery_metadata)
            )
            if (not isinstance(recovery_metadata, dict)
                    or recovery_metadata.get("enabled") is not True
                    or recovery_metadata.get("dirty") is not False
                    or type(durable_version) is not int
                    or (durable_version != version and not pending_publication)
                    or type(recorded_version) is not int
                    or recorded_version != version):
                raise RuntimeError(
                    f"policy version {version} cannot be published before a matching durable recovery snapshot"
                )

    async def publish_current(already_published=None, *, pending_checkpoint=False):
        async with publication_lock:
            return await _publish_current_unlocked(already_published, pending_checkpoint=pending_checkpoint)

    async def _publish_current_unlocked(
        already_published: dict[str, Any] | None = None, *, pending_checkpoint: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        version = engine.policy_version
        _validate_publication_state(version, already_published, pending_checkpoint)
        if already_published is None:
            published = await compute_owner.run(engine.publish)
        else:
            reported_version = already_published.get("version") if isinstance(already_published, dict) else None
            if type(reported_version) is not int or reported_version != version:
                raise ValueError(
                    "provided update publication version must match LearnerEngine.policy_version"
                )
            published = already_published
        source_files = _policy_files(published, version)
        quantization = _quantization_evidence(published)
        if cfg.get("model_backend", "default") == "ternary":
            expected_sha = evidence["base_fingerprint"]
            if (published.get("base_fingerprint") != expected_sha
                    or published.get("artifact_fingerprint") != expected_sha
                    or quantization != evidence.get("quantization")):
                raise ValueError("T2 published policy differs from the validated frozen artifact and quantization identity")
        from .served_policy import materialize_served_policy
        files, materialization = await _finish_cache_write(lambda: materialize_served_policy(
            cfg['run_dir'], version, source_files, published,
            cfg.get('model_backend', 'default'), app.state.policies.get(version)))
        _validate_publication_state(version, already_published, pending_checkpoint)
        app.state.external_stage_timings['served_policy_materialize'] = {
            'policy_version': version, **materialization}
        policy = {
            "version": version,
            "base_fingerprint": _base_fingerprint(published),
            "artifact_fingerprint": published["artifact_fingerprint"],
            "quantization": quantization,
            "files": {name: {"sha256": None, "size_bytes": entry["size"]} for name, entry in files.items()},
            "_files": files,
            "published_at": time.time(),
        }
        _attach_native_source(policy, published)
        existing = app.state.policies.get(version)
        if existing is not None:
            # A version is immutable: repeated publish is permitted only for byte-identical files.
            if (existing["files"] != policy["files"]
                    or existing.get("source_adapter_sha256") != policy.get("source_adapter_sha256")
                    or existing.get("artifact_id") != policy.get("artifact_id")):
                raise RuntimeError(f"policy version {version} changed after publication")
        app.state.policies[version] = policy
        app.state.timings["learner"]["publish_seconds"] += time.perf_counter() - started
        return policy

    # Publish the initial adapter before /health turns ready.
    initial_policy = asyncio.run(publish_current()) if not _inside_running_loop() else None
    if initial_policy is not None:
        app.state.initial_policy_ready = True
    else:
        app.state.initial_policy_ready = False

    def latest_policy() -> dict[str, Any]:
        version = int(engine.policy_version)
        policy = app.state.policies.get(version)
        if policy is None:
            raise HTTPException(status_code=503, detail="Current adapter has not been published")
        return policy

    def latest_published_policy() -> dict[str, Any]:
        # The CPU optimizer can advance while a parked executor restores its
        # Worker. Until the next manifest is registered, that Worker must still
        # be able to restore the last published version. This is not permission
        # to serve an unregistered optimizer state or skip cohort ACKs.
        from .served_policy import published_policy_for_restore

        try:
            return published_policy_for_restore(
                app.state.policies, int(engine.policy_version), engine_recovery_metadata(),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def current_served_policy_provider() -> dict[str, Any]:
        policy = latest_published_policy()
        return {
            "version": int(policy["version"]),
            "base_fingerprint": policy["base_fingerprint"],
            "artifact_fingerprint": policy["artifact_fingerprint"],
            "quantization": dict(policy["quantization"]),
            "files": {
                name: {"sha256": value.get("sha256"), "size_bytes": value.get("size_bytes")}
                for name, value in policy["files"].items()
            },
            "source_adapter_sha256": policy.get("source_adapter_sha256"),
            "artifact_id": policy.get("artifact_id"),
        }

    central_gradient_worker_ids = tuple(
        cfg["learner"].get("gradient_worker_sync_ids")
        or cfg["learner"].get("gradient_executor_worker_ids", {}).values()
    ) if cfg["learner"].get("update_backend", "trainer") == "central_gradient" else ()

    worker_control = WorkerControlPlane(
        workers=cfg["workers"],
        request_timeout=float(cfg["learner"].get("request_timeout", 30.0)),
        quantization_mode=cfg["quantization"]["mode"],
        policies=app.state.policies,
        policy_refs=app.state.policy_refs,
        policy_file_references=app.state.policy_file_references,
        required_policy_worker_ids=central_gradient_worker_ids,
        run_dir=cfg["run_dir"],
        training_telemetry=app.state.training_telemetry,
        timings=app.state.timings,
        external_stage_timings=app.state.external_stage_timings,
        get_monitor_phase=lambda: app.state.monitor_phase,
        set_monitor_phase=set_monitor_phase,
        set_serve_version=lambda version: setattr(app.state, "serve_policy_version", version),
        set_last_acks=lambda acknowledgements: setattr(
            app.state, "last_policy_acks", acknowledgements,
        ),
        prune_synced_policies=_prune_synced_policies,
        client_factory=lambda **kwargs: httpx.AsyncClient(**kwargs),
    )
    app.state.worker_control = worker_control

    async def assert_workers_current(expected_version: int) -> None:
        await worker_control.assert_workers_current(
            expected_version,
            **({"worker_ids": central_gradient_worker_ids}
               if central_gradient_worker_ids else {}),
        )

    helper_pool: SharedInferencePool | None = None

    async def sync_workers(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if central_gradient_worker_ids:
            acknowledgements = await worker_control.sync_selected_workers(policy, central_gradient_worker_ids)
        else:
            acknowledgements = await worker_control.sync_workers(policy)
        if helper_pool is not None:
            helper_pool.request_sync()
        return acknowledgements

    async def ensure_policy_for_workers(
        policy: dict[str, Any], worker_ids: tuple[str, ...] | list[str],
    ) -> dict[str, dict[str, Any]]:
        return await worker_control.sync_selected_workers(policy, worker_ids)

    if cfg["learner"].get("update_backend", "trainer") == "central_gradient":
        allowed_worker_sync_ids = set(cfg["learner"].get("gradient_worker_sync_ids", ()))

        async def ensure_selected_workers(executor_id: str, payload: Mapping[str, Any]):
            worker_ids = payload.get("worker_ids")
            if (not isinstance(worker_ids, list) or not worker_ids
                    or not set(worker_ids).issubset(allowed_worker_sync_ids)):
                raise ValueError("executor requested Workers outside gradient_worker_sync_ids")
            policy = latest_published_policy()
            if payload.get("policy_version") != policy["version"]:
                raise ValueError("executor requested Worker sync for a non-current policy")
            return await ensure_policy_for_workers(policy, tuple(worker_ids))

        engine.current_served_policy_provider = current_served_policy_provider
        engine.ensure_selected_workers = ensure_selected_workers
        app.state.gradient_policy_bridge = {
            "latest_policy": current_served_policy_provider,
            "ensure_policy": ensure_policy_for_workers,
        }

    evaluation_pool = worker_pool
    helper_evaluation_scope = None
    if helper_worker_ids:
        evaluation_pool = EpisodeWorkerPool(
            helper_worker_ids, capacity=complete_group_episode_capacity(cfg),
            resource_probe=resource_probe,
            member_epochs={name: member_epochs[name] for name in helper_worker_ids},
            initially_paused=helper_worker_ids,
        )

        def helper_training_policy() -> PolicySnapshot:
            return PolicySnapshot.from_policy(latest_published_policy())

        async def sync_helpers(snapshot: PolicySnapshot) -> None:
            health = await asyncio.gather(
                *(worker_control.health_of_worker(name) for name in helper_worker_ids),
            )
            if any(row.get('episode_refs') != 0 for row in health):
                raise RuntimeError('helper Worker still holds an episode lease')
            publication = app.state.policy_file_references.acquire(snapshot.version, app.state.policies)
            if publication is None:
                raise RuntimeError('helper synchronization policy is no longer published')
            try:
                await worker_control.sync_selected_workers(publication, helper_worker_ids, wait_gateway=True)
            finally:
                app.state.policy_file_references.release(snapshot.version)

        local_inference_pool = SharedInferencePool(
            helper_worker_ids, training_pool=worker_pool, evaluation_pool=evaluation_pool,
            scheduler=worker_pool.episode_queue, current_policy=helper_training_policy,
            sync_policy=sync_helpers, drain_timeout=float(cfg['learner']['request_timeout']),
        )
        helper_pool = local_inference_pool

        @asynccontextmanager
        async def helper_evaluation_scope(policy: EvaluationPolicy) -> AsyncIterator[None]:
            async with local_inference_pool.evaluation(policy.snapshot):
                health = await asyncio.gather(
                    *(worker_control.health_of_worker(name) for name in helper_worker_ids),
                )
                if any(row.get('ready') is not True or row.get('policy_ready') is not True
                       or row.get('episode_refs') != 0 for row in health):
                    raise RuntimeError('evaluation helpers are not ready and drained')
                leases: dict[str, str] = {}

                async def acquire(name: str) -> None:
                    episode_id = f'evaluation-pool-{uuid.uuid4().hex}'
                    await managers[name].acquire_episode(
                        episode_id=episode_id, policy_version=policy.version,
                        adapter_hash=policy.adapter_sha256, purpose='evaluation',
                    )
                    leases[name] = episode_id

                try:
                    results = await asyncio.gather(
                        *(acquire(name) for name in helper_worker_ids), return_exceptions=True,
                    )
                    error = next((item for item in results if isinstance(item, BaseException)), None)
                    if error is not None:
                        raise error
                    yield
                finally:
                    async def release() -> None:
                        results = await asyncio.gather(*(
                            managers[name].release_episode(
                                episode_id=episode_id, policy_version=policy.version,
                                adapter_hash=policy.adapter_sha256, purpose='evaluation',
                            ) for name, episode_id in leases.items()), return_exceptions=True,
                        )
                        error = next((item for item in results if isinstance(item, BaseException)), None)
                        if error is not None:
                            raise error

                    from .gradient_executor.client import _join_owned_task
                    await _join_owned_task(asyncio.create_task(release()))

    app.state.inference_pool = helper_pool

    async def run_evaluation_episode(task, policy, *, seed, request_id, attempt_id):
        started = time.monotonic()
        group = await _generate_scored_group(
            engine, managers, cfg, task, attempt_id,
            policy.snapshot, worker_pool=evaluation_pool, group_size_override=1,
            validation=True, evaluation_seed=seed,
            # A paused training boundary (or an idle standalone evaluator) owns
            # the slots; background evaluations keep normal training priority.
            validation_low_priority=not (
                helper_pool is not None or eval_settings.get('pause_training') is True
                or app.state.active_run_id is None
            ),
        )
        row = group['trajectories'][0]
        worker_id = group['worker_ids'][0]
        from ..evaluation.runner import tool_event_metrics
        _, _, tool_errors = tool_event_metrics(row)
        if arc_contract is None:
            from ..budget import load_contract

            actual_contract = load_contract(cfg['arc']['contract_path'])
        else:
            actual_contract = arc_contract
        from ..evaluation.results import EpisodeExecution
        from ..evaluation.runtime import native_episode_settings

        return EpisodeExecution(
            trajectory=row,
            worker_id=worker_id,
            episode_id=row.get('episode_id'),
            duration_seconds=time.monotonic() - started,
            tool_errors=tool_errors,
            group_id=attempt_id,
            sampling_config=group['sampling_configs'][worker_id],
            turn_seed_receipts=tuple(group['counters'][worker_id]['evaluation_turn_seeds']),
            settings=native_episode_settings(cfg['rollout'], actual_contract),
        )

    evaluation_coordinator = EvaluationCoordinator(
        settings=eval_settings,
        dataset=evaluation_dataset,
        engine=engine,
        policies=app.state.policies,
        policy_refs=app.state.policy_refs,
        current_policy=latest_policy,
        recovery_metadata=engine_recovery_metadata,
        assert_workers_current=assert_workers_current,
        run_episode=run_evaluation_episode,
        on_initialized=lambda runtime: setattr(app.state, "evaluation", runtime),
        evaluation_scope=helper_evaluation_scope,
    )
    app.state.evaluation_coordinator = evaluation_coordinator

    async def ensure_evaluation_ready():
        return await evaluation_coordinator.ensure_ready()

    async def start_idle_evaluation():
        if not evaluation_coordinator.pause_training:
            return
        async with state_lock:
            if app.state.active_task is not None and not app.state.active_task.done():
                return

            async def run_idle_evaluation():
                set_monitor_phase("native_evaluation")
                try:
                    await evaluation_coordinator.drain_pending_at_boundary()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    _LOGGER.warning("idle native evaluation failed (%s): %s",
                                    type(exc).__name__, str(exc)[:300])
                finally:
                    set_monitor_phase("idle")

            app.state.active_task = asyncio.create_task(
                run_idle_evaluation(), name="native-evaluation-idle",
            )

    @app.get('/evaluation', dependencies=[Depends(_require_token)])
    async def evaluation_status():
        return evaluation_coordinator.status()

    @app.post('/evaluation', dependencies=[Depends(_require_token)])
    async def evaluate_current_policy():
        if not eval_settings.get('enabled'):
            raise HTTPException(409, 'native evaluation is disabled in this deployment')
        if recovery_is_dirty():
            raise HTTPException(409, 'cannot evaluate an undurable learner policy')
        try:
            request = await evaluation_coordinator.request_current()
            if evaluation_coordinator.pause_training:
                await start_idle_evaluation()
            return request
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/evaluation/{request_id}/retry-errors', dependencies=[Depends(_require_token)])
    async def retry_evaluation_errors(request_id: str):
        if not eval_settings.get('enabled'):
            raise HTTPException(409, 'native evaluation is disabled in this deployment')
        if recovery_is_dirty():
            raise HTTPException(409, 'cannot retry evaluation on an undurable learner policy')
        try:
            request = await evaluation_coordinator.retry_errors(request_id)
            if evaluation_coordinator.pause_training:
                await start_idle_evaluation()
            return request
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    attach_health_route(
        app,
        require_token=_require_token,
        worker_pool=worker_pool,
        recovery_is_dirty=recovery_is_dirty,
        checkpoint_save_in_progress=checkpoint_save_in_progress,
        engine_recovery_metadata=engine_recovery_metadata,
        recovery_summary=recovery_summary,
    )
    attach_monitor_route(
        app,
        require_token=_require_token,
        worker_pool=worker_pool,
        evaluation_coordinator=evaluation_coordinator,
        recovery_summary=recovery_summary,
        alignment_scope=_alignment_scope,
    )
    attach_policy_routes(
        app,
        policy_read_authorizer=policy_read_authorizer,
        latest_policy=latest_policy,
        policy_file_references=app.state.policy_file_references,
        tensor_encoder=lambda raw, **kwargs: encode_tensor_payload(raw, **kwargs),
    )


    async def generate_bepa_seed_episode(task, group_id, snapshot, plan):
        app.state.active_sampling_groups += 1
        try:
            return await _generate_scored_group(
                engine,
                managers,
                cfg,
                task,
                group_id,
                snapshot,
                worker_pool=worker_pool,
                plan=plan,
                group_size_override=1,
            )
        finally:
            app.state.active_sampling_groups -= 1

    async def build_bepa_seed_prefix(task):
        loop = _agent_loop(engine, next(iter(managers.values())), cfg, 1)
        from ..prompt import build_task_messages

        return await loop.ct_build_initial_tokens(
            build_task_messages(task),
            tools=loop.tool_schema_dicts,
        )

    bepa_seed_service = BepaSeedService(
        bepa=bepa,
        engine=engine,
        policies=app.state.policies,
        policy_refs=app.state.policy_refs,
        generate_episode=generate_bepa_seed_episode,
        build_prefix=build_bepa_seed_prefix,
        sync_workers=sync_workers,
    )

    run_coordinator = RunCoordinator(RunCoordinatorServices(
        arc_sampling=not generic_environment,
        config=cfg,
        runtime=app.state,
        engine=engine,
        managers=managers,
        worker_pool=worker_pool,
        environment=environment,
        reward_scorer=reward_scorer,
        compute_owner=compute_owner,
        bepa=bepa,
        evaluation=evaluation_coordinator,
        seed_service=bepa_seed_service,
        update_lock=update_lock,
        set_monitor_phase=set_monitor_phase,
        latest_policy=latest_policy,
        publish_current=publish_current,
        make_recovery_context=make_recovery_context,
        engine_recovery_metadata=engine_recovery_metadata,
        assert_workers_current=assert_workers_current,
        sync_workers=sync_workers,
        persist_group=lambda *args, **kwargs: _persist_trajectory_group(*args, **kwargs),
        stream_factory=lambda **kwargs: AsyncGroupStream(**kwargs),
        measure_alignment=_measure_logprob_alignment,
        alignment_from_proximal_scores=_alignment_from_proximal_scores,
        log_alignment_warning=_log_alignment_warning,
        publication_from_update=_evaluation_publication_from_update,
        prune_synced_policies=_prune_synced_policies,
        diagnostic_course=_IDENTITY_DIAGNOSTIC_COURSE,
        diagnostic_run_kinds=_IDENTITY_DIAGNOSTIC_RUN_KINDS,
        stop_control=app.state.stop_control,
        alignment_thresholds={
            "mean_abs_error": _LOGPROB_MEAN_ABS_TOLERANCE,
            "max_abs_error": _LOGPROB_MAX_ABS_TOLERANCE,
            "rms_error": _LOGPROB_RMS_TOLERANCE,
        },
    ))
    app.state.run_coordinator = run_coordinator

    async def start_bepa_bootstrap(task_ids):
        if cfg["learner"].get("update_backend", "trainer") == "central_gradient":
            raise HTTPException(
                status_code=422,
                detail="central_gradient schedules complete BEPA rescue groups within training runs",
            )
        async with state_lock:
            if app.state.stop_control.blocks_new_work:
                raise HTTPException(409, "A stop-after-checkpoint request owns the Learner")
            if recovery_is_dirty():
                raise HTTPException(503, "Learner policy is newer than its durable recovery snapshot")
            if app.state.active_task is not None and not app.state.active_task.done():
                raise HTTPException(409, "A run or bootstrap is already active")
            recovery = recovery_summary()
            if recovery is not None and recovery.get("pending_run") is not None:
                raise HTTPException(409, "A recovered run must be resumed before BEPA bootstrap")
            allowed = {key: task for key, task in task_map.items()
                       if task.split == "train" and task.pool != "diagnostic"}
            ids = task_ids or cfg["rollout"].get("task_ids") or list(allowed)
            if set(ids) - set(allowed):
                raise HTTPException(422, "Bootstrap task_ids must be training tasks")
            selected = [allowed[key] for key in dict.fromkeys(ids)][:int(bepa.settings["bootstrap"]["max_tasks"])]
            operation = {"operation_id": uuid.uuid4().hex, "status": "running", "seeded": 0,
                         "task_ids": [task.task_id for task in selected], "started_at": time.time()}
            bepa.operation = operation
            async def run_bootstrap():
                try:
                    async with update_lock:
                        snapshot = PolicySnapshot.from_policy(app.state.policies[int(engine.policy_version)])
                        await assert_workers_current(snapshot.version)
                        operation["seeded"] = await bepa_seed_service.seed_maintenance(selected, snapshot, wait=True,
                            enqueue=True, limit=int(bepa.settings["bootstrap"]["target_cache_tasks"]))
                        if operation["status"] == "running":
                            operation["status"] = (
                                "succeeded" if operation["seeded"] >= int(bepa.settings["bootstrap"]["target_cache_tasks"])
                                else "insufficient_learning_signal"
                            )
                except asyncio.CancelledError:
                    operation["status"] = "cancelled"
                    raise
                except Exception as exc:
                    operation.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)[:300]})
                finally:
                    operation["finished_at"] = time.time()
                    await asyncio.to_thread(bepa.artifacts.write_json, operation)
            app.state.active_task = asyncio.create_task(run_bootstrap(), name="bepa-bootstrap")
            return operation

    @app.post("/run", status_code=202, dependencies=[Depends(_require_token)])
    async def start_run(request: RunRequest):
        if (cfg["learner"].get("update_backend", "trainer") == "central_gradient"
                and request.kind != "training"):
            raise HTTPException(
                status_code=422,
                detail="central_gradient supports training runs only; identity alignment requires a local scorer",
            )
        if getattr(getattr(app.state, "gpu_phase_controller", None), "state", None) == "failed":
            raise HTTPException(503, "GPU ownership transition failed; full-state recovery is required")
        if request.kind == 'training':
            await ensure_evaluation_ready()
        async with state_lock:
            if app.state.stop_control.blocks_new_work:
                raise HTTPException(409, "A stop-after-checkpoint request owns the Learner")
            if recovery_is_dirty():
                raise HTTPException(status_code=503, detail="Learner policy is newer than its durable recovery snapshot")
            if app.state.active_task is not None and not app.state.active_task.done():
                raise HTTPException(status_code=409, detail="A run is already active")
            recovery = recovery_summary()
            pending = recovery.get("pending_run") if recovery is not None else None
            if pending is not None:
                raise HTTPException(status_code=409, detail="A recovered run must be resumed before starting a new run")
            if int(engine.policy_version) not in app.state.policies:
                raise HTTPException(status_code=503, detail="Current policy has not been published")
            requested_steps = request.steps if request.steps is not None else int(cfg["dapo"]["updates"])
            if request.kind == "selftest" and requested_steps != 1:
                raise HTTPException(status_code=422, detail="selftest requires exactly one actor update")
            if request.kind == "diagnostic" and requested_steps < 2:
                raise HTTPException(status_code=422, detail="diagnostic requires at least two actor updates")
            if request.kind == "diagnostic" and "task_ids" in request.model_fields_set:
                raise HTTPException(
                    status_code=422,
                    detail="diagnostic uses a fixed identity task and does not accept task_ids",
                )
            run_id = uuid.uuid4().hex
            state = {
                "run_id": run_id,
                "kind": request.kind,
                "diagnostic_course": (
                    _IDENTITY_DIAGNOSTIC_COURSE
                    if request.kind in _IDENTITY_DIAGNOSTIC_RUN_KINDS else None
                ),
                "status": "running",
                "failure_stage": "queued",
                "started_at": time.time(),
                "initial_policy_version": int(engine.policy_version),
                "initial_adapter_hash": PolicySnapshot.from_policy(app.state.policies[int(engine.policy_version)]).adapter_sha256,
                "requested_steps": requested_steps,
                "completed_updates": 0,
                "filtered_groups": 0,
                "groups": [],
                "pre_update_logprob_alignment": None,
                "logprob_alignment": None,
                "policy_handoffs": [],
                "metrics": {"learner": {
                    "model_load_seconds": app.state.timings["learner"]["model_load_seconds"],
                    "initial_publish_seconds": app.state.timings["learner"]["publish_seconds"],
                    "update_seconds": 0.0, "publish_seconds": 0.0,
                    "worker_sync_seconds": 0.0,
                }, "workers": {}, environment.metrics_key: {}},
                "summary": None,
                "error": None,
            }
            state = RunState(run_id, state)
            app.state.runs[run_id] = state
            app.state.latest_run_id = run_id
            app.state.active_run_id = run_id
            app.state.active_task = asyncio.create_task(
                run_coordinator.execute(run_id, request), name=f"rl-run-{run_id}")
        return {"run_id": run_id, "status": "running", "kind": request.kind}

    @app.post("/resume", dependencies=[Depends(_require_token)])
    async def resume_run(request: ResumeRequest):
        metadata = engine_recovery_metadata()
        if metadata is None:
            return JSONResponse(status_code=200, content={"status": "no_pending_run", "run_id": None})
        if not app.state.recovery_initialized:
            raise HTTPException(status_code=503, detail="Learner recovery checks have not completed")
        if recovery_is_dirty():
            raise HTTPException(status_code=503, detail="Learner policy is newer than its durable recovery snapshot")
        lineage_id = metadata.get("lineage_id")
        if not isinstance(lineage_id, str) or request.lineage_id != lineage_id:
            raise HTTPException(status_code=409, detail="Recovery lineage_id does not match the loaded checkpoint")

        async with state_lock:
            recovery = recovery_summary()
            if app.state.stop_control.blocks_new_work:
                stop_receipt = app.state.stop_control.snapshot() or {}
                current_pending = recovery.get("pending_run") if recovery is not None else None
                resumable_stop_target = (
                    stop_receipt.get("status") in {"accepted", "waiting_durable"}
                    and isinstance(current_pending, dict)
                    and stop_receipt.get("run_id") == current_pending.get("run_id")
                    and stop_receipt.get("target_update")
                    == int(current_pending.get("completed_updates", -1)) + 1
                    and stop_receipt.get("target_policy_version")
                    == int(engine.policy_version) + 1
                )
                if not resumable_stop_target:
                    raise HTTPException(409, "A stop-after-checkpoint request owns the Learner")
            if app.state.active_task is not None and not app.state.active_task.done():
                if app.state.active_run_id == app.state.resumed_run_id and app.state.resumed_run_id:
                    state = app.state.runs[app.state.resumed_run_id]
                    remaining = max(0, int(state["requested_steps"]) - int(state["completed_updates"]))
                    return JSONResponse(status_code=202, content={
                        "status": "running", "resumed": True,
                        "run_id": app.state.resumed_run_id, "remaining_steps": remaining,
                        "task_prompt_transition": state.get("resume", {}).get("task_prompt_transition"),
                    })
                raise HTTPException(status_code=409, detail="A run or bootstrap is already active")

            pending = recovery.get("pending_run") if recovery is not None else None
            if pending is None:
                return JSONResponse(status_code=200, content={"status": "no_pending_run", "run_id": None})
            run_id = pending["run_id"]
            # Active-task idempotence is handled above. A finished attempt must
            # revalidate the durable context and Worker policy before retrying.
            if metadata.get("resumed") is not True:
                return JSONResponse(status_code=200, content={"status": "no_pending_run", "run_id": None})
            if recovery.get("bepa") is None and bepa is not None:
                raise HTTPException(status_code=503, detail="BEPA recovery reconciliation is unavailable")

            context = metadata.get("update_context")
            logical_job = context.get("logical_job") if isinstance(context, dict) else None
            source_identity = context.get("source_identity") if isinstance(context, dict) else None
            cursor = context.get("task_cursor") if isinstance(context, dict) else None
            if not isinstance(logical_job, dict) or not isinstance(source_identity, dict) or not isinstance(cursor, dict):
                raise HTTPException(status_code=409, detail="Loaded checkpoint has incomplete logical-run context")
            if generic_environment:
                try:
                    validate_recovery_source_identity(source_identity, environment)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=409,
                        detail="Recovery source identity does not match the active environment adapter",
                    ) from exc
            else:
                from ..deployment.instance_relocation import path_identity_matches

                try:
                    dataset_path_matches = path_identity_matches(
                        source_identity.get("dataset_dir"),
                        str(Path(cfg["dataset_dir"]).resolve()),
                        cfg.get("learner", {}).get("recovery_path_mappings", {}),
                    )
                except (TypeError, ValueError):
                    dataset_path_matches = False
                if not dataset_path_matches:
                    raise HTTPException(status_code=409, detail="Recovery task source does not match the active dataset")
            if recovery_semantics is not None and any(
                source_identity.get(key) != value for key, value in recovery_semantics.items()
            ):
                raise HTTPException(status_code=409, detail="Recovery source semantics do not match the active contracts")
            task_ids = logical_job.get("task_ids")
            if (not isinstance(task_ids, list) or not task_ids
                    or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
                    or len(task_ids) != len(set(task_ids))):
                raise HTTPException(status_code=409, detail="Recovery context has an invalid task allowlist")
            task_worker_owners = context.get("task_worker_owners", {})
            rescue_ready_task_ids = context.get("rescue_ready_task_ids", [])
            if (not isinstance(task_worker_owners, dict)
                    or any(task_id not in task_ids
                           or not isinstance(worker_id, str) or not worker_id
                           for task_id, worker_id in task_worker_owners.items())
                    or not isinstance(rescue_ready_task_ids, list)
                    or any(not isinstance(task_id, str) or task_id not in task_ids
                           for task_id in rescue_ready_task_ids)
                    or len(set(rescue_ready_task_ids)) != len(rescue_ready_task_ids)):
                raise HTTPException(status_code=409, detail="Recovery context has invalid BEPA task affinity")
            if cfg["learner"].get("update_backend") == "central_gradient":
                from qwen_rl.contracts.sampling_topology import owner_map
                allowed_owner_workers = set(owner_map(cfg["learner"]).values())
                if set(task_worker_owners.values()) - allowed_owner_workers:
                    raise HTTPException(
                        status_code=409,
                        detail="Recovery task Worker owners differ from the current explicit executor map",
                    )
            next_task_index = cursor.get("next_task_index")
            next_group_index = cursor.get("next_group_index")
            if (isinstance(next_task_index, bool) or not isinstance(next_task_index, int) or next_task_index < 0
                    or isinstance(next_group_index, bool) or not isinstance(next_group_index, int)
                    or next_group_index < 0):
                raise HTTPException(status_code=409, detail="Recovery context has an invalid task cursor")
            kind = logical_job.get("kind")
            requested_steps = logical_job.get("requested_steps")
            completed_updates = logical_job.get("completed_updates")
            if (kind not in {"training", "selftest", "diagnostic"}
                    or isinstance(requested_steps, bool) or not isinstance(requested_steps, int)
                    or requested_steps < 1
                    or isinstance(completed_updates, bool) or not isinstance(completed_updates, int)
                    or completed_updates < 0 or completed_updates >= requested_steps):
                raise HTTPException(status_code=409, detail="Recovery context has invalid logical-run progress")
            if bepa is not None and context.get("engine_epoch") != bepa.epoch:
                raise HTTPException(status_code=409, detail="Recovery context BEPA epoch does not match the ledger")
            current_task_prompt = _current_prompt_identity(environment)
            prompt_changed = context.get("task_prompt") != current_task_prompt
            if prompt_changed:
                # The queue stores task IDs, while plans/cache use prompt-derived
                # task keys. Do not force old-prompt rescue IDs into native sampling.
                rescue_ready_task_ids = []
            resumed_request = RunRequest(
                kind=kind,
                steps=requested_steps,
                task_ids=task_ids if kind == "training" else None,
            )
            if int(engine.policy_version) not in app.state.policies:
                raise HTTPException(status_code=503, detail="Recovered policy is not published")
            policy = app.state.policies[int(engine.policy_version)]
            try:
                await assert_workers_current(int(engine.policy_version))
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Workers are not ready on the recovered policy: {exc}") from exc
            if kind == 'training':
                await ensure_evaluation_ready()
            state = {
                "run_id": run_id,
                "kind": kind,
                "diagnostic_course": (
                    _IDENTITY_DIAGNOSTIC_COURSE
                    if kind in _IDENTITY_DIAGNOSTIC_RUN_KINDS else None
                ),
                "status": "running",
                "failure_stage": "queued",
                "started_at": time.time(),
                "initial_policy_version": int(engine.policy_version),
                "initial_adapter_hash": PolicySnapshot.from_policy(policy).adapter_sha256,
                "requested_steps": requested_steps,
                "completed_updates": completed_updates,
                "filtered_groups": 0,
                "groups": [],
                "pre_update_logprob_alignment": None,
                "logprob_alignment": None,
                "policy_handoffs": [],
                "resume": {
                    "resumed": True,
                    "lineage_id": lineage_id,
                    "checkpoint_policy_version": int(engine.policy_version),
                    "checkpoint_snapshot_sha256": metadata.get("snapshot_sha256"),
                    "prior_completed_updates": completed_updates,
                    "task_prompt_transition": ({"from": context.get("task_prompt"),
                                                "to": current_task_prompt}
                                               if prompt_changed else None),
                    "remaining_steps": requested_steps - completed_updates,
                },
                "resume_task_cursor": {
                    "next_task_index": next_task_index,
                    "next_group_index": next_group_index,
                },
                "category_sampling": copy.deepcopy(context.get('category_sampling')),
                "source_sampling": copy.deepcopy(context.get("source_sampling")),
                "task_worker_owners": dict(task_worker_owners),
                "rescue_ready_task_ids": list(rescue_ready_task_ids),
                # A/B membership contains ARC2 task IDs, not prompt-scoped
                # plans or replay. Preserve the current task-set-bound schema;
                # ActiveTaskSampling revalidates its identity before dispatch.
                # Prompt-scoped rescue_ready_task_ids remain cleared above.
                "bepa_task_sampling": (
                    copy.deepcopy(context.get("bepa_task_sampling"))
                    if not prompt_changed or (
                        isinstance(context.get("bepa_task_sampling"), dict)
                        and context["bepa_task_sampling"].get("schema") == "bepa-active-arc2/v1"
                    ) else None
                ),
                "metrics": {"learner": {
                    "model_load_seconds": app.state.timings["learner"]["model_load_seconds"],
                    "initial_publish_seconds": app.state.timings["learner"]["publish_seconds"],
                    "update_seconds": 0.0, "publish_seconds": 0.0,
                    "worker_sync_seconds": 0.0,
                }, "workers": {}, environment.metrics_key: {}},
                "summary": None,
                "error": None,
            }
            state = RunState(run_id, state)
            app.state.runs[run_id] = state
            app.state.latest_run_id = run_id
            app.state.active_run_id = run_id
            app.state.resumed_run_id = run_id
            app.state.active_task = asyncio.create_task(
                run_coordinator.execute(run_id, resumed_request), name=f"rl-resume-{run_id}")
            return JSONResponse(status_code=202, content={
                "status": "running", "resumed": True,
                "run_id": run_id, "remaining_steps": requested_steps - completed_updates,
                "task_prompt_transition": state["resume"]["task_prompt_transition"],
            })

    @app.get("/run", dependencies=[Depends(_require_token)])
    async def get_run(run_id: str | None = None):
        selected = run_id or app.state.latest_run_id
        if selected is None or selected not in app.state.runs:
            raise HTTPException(status_code=404, detail="Unknown run_id")
        return app.state.runs[selected]

    async def wait_for_idle_stop(operation_id: str) -> None:
        try:
            await asyncio.to_thread(wait_for_engine_durable, engine)
            app.state.stop_control.wait_idle_complete(
                recovery=engine_recovery_metadata() or {},
                bepa=bepa,
            )
        except Exception as exc:
            app.state.stop_control.block(
                f"current checkpoint did not become durable: {exc}", type(exc).__name__,
            )

    @app.post("/control/stop", dependencies=[Depends(_require_token)])
    async def request_stop(request: StopControlRequest):
        async with state_lock:
            active_run_id = app.state.active_run_id
            active_state = (
                app.state.runs.get(active_run_id)
                if isinstance(active_run_id, str) else None
            )
            if not isinstance(active_state, dict) or active_state.get("status") != "running":
                active_run_id = None
                active_state = None
            recovery = engine_recovery_metadata()
            update_context = recovery.get("update_context") if isinstance(recovery, dict) else None
            logical = update_context.get("logical_job") if isinstance(update_context, dict) else None
            completed_update = (
                int(active_state.get("completed_updates", 0)) if active_state is not None
                else int(logical.get("completed_updates", 0)) if isinstance(logical, dict)
                and type(logical.get("completed_updates")) is int else 0
            )
            requested_steps = (
                int(active_state.get("requested_steps")) if active_state is not None
                and type(active_state.get("requested_steps")) is int else None
            )
            try:
                receipt, close_stream, needs_idle_wait = app.state.stop_control.request(
                    request.operation_id,
                    run_id=active_run_id,
                    completed_update=completed_update,
                    requested_steps=requested_steps,
                    current_policy_version=int(engine.policy_version),
                    recovery=recovery,
                    bepa=bepa,
                    supported=cfg.get("learner", {}).get("update_backend", "trainer") == "central_gradient",
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except OSError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Could not persist stop-after-checkpoint intent: {type(exc).__name__}",
                ) from exc
            if close_stream and app.state.rollout_stream is not None:
                try:
                    await app.state.rollout_stream.stop_admission()
                except Exception as exc:
                    app.state.stop_control.block(
                        f"could not close rollout admission: {exc}", type(exc).__name__,
                    )
                    receipt = app.state.stop_control.get(request.operation_id) or receipt
            if (active_state is not None and active_state.get("kind") == "training"
                    and bepa is not None and receipt.get("status") in {"accepted", "waiting_durable"}):
                app.state.stop_control.mark_bepa_pending(request.operation_id)
                receipt = app.state.stop_control.get(request.operation_id) or receipt
            receipt = app.state.stop_control.refresh_bepa_state(bepa) or receipt
            if (needs_idle_wait and app.state.stop_control.has_pending_request
                    and (app.state.stop_wait_task is None or app.state.stop_wait_task.done())):
                app.state.stop_wait_task = asyncio.create_task(
                    wait_for_idle_stop(request.operation_id),
                    name=f"stop-after-checkpoint-{request.operation_id}",
                )
            receipt = app.state.stop_control.get(request.operation_id)
            if receipt is None:
                raise HTTPException(status_code=409, detail="Stop request was superseded")
            return receipt

    @app.get("/control/stop/{operation_id}", dependencies=[Depends(_require_token)])
    async def get_stop_status(operation_id: str):
        receipt = app.state.stop_control.refresh_bepa_state(bepa)
        if receipt is not None and receipt.get("operation_id") != operation_id:
            receipt = None
        if receipt is None:
            raise HTTPException(status_code=404, detail="Unknown stop operation_id")
        return receipt

    @app.post("/checkpoint", dependencies=[Depends(_require_token)])
    async def save_checkpoint(_request: CheckpointRequest):
        async with state_lock:
            if app.state.active_task is not None and not app.state.active_task.done():
                raise HTTPException(status_code=409, detail="Cannot checkpoint while a run or bootstrap is active")
            if app.state.active_sampling_groups or app.state.training_telemetry.active is not None:
                raise HTTPException(status_code=409, detail="Cannot checkpoint while sampling or optimizer work is active")
            metadata = engine_recovery_metadata()
            save = getattr(engine, "save_recovery_snapshot", None)
            if metadata is None or metadata.get("enabled") is not True or not callable(save):
                raise HTTPException(status_code=409, detail="Full-state recovery is not enabled for this Learner")
            if recovery_is_dirty():
                raise HTTPException(status_code=503, detail="Learner policy is newer than its durable recovery snapshot")
            policy_version = int(engine.policy_version)
            if (metadata.get("snapshot_id")
                    and metadata.get("durable_policy_version") == policy_version
                    and metadata.get("policy_version") == policy_version):
                return {
                    "status": "already_saved",
                    "policy_version": policy_version,
                    "snapshot_id": metadata["snapshot_id"],
                    "snapshot_sha256": metadata["snapshot_sha256"],
                }
            try:
                snapshot = await compute_owner.run(
                    save,
                    update_context=metadata.get("update_context"),
                    reason="explicit_api",
                )
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Full-state checkpoint failed: {exc}") from exc
            return {
                "status": "saved",
                "policy_version": int(engine.policy_version),
                "snapshot_id": snapshot.get("snapshot_id"),
                "snapshot_sha256": snapshot.get("snapshot_sha256"),
            }

    @app.post("/probe", status_code=202, dependencies=[Depends(_require_token)])
    async def start_probe(request: ProbeRequest):
        if cfg["learner"].get("update_backend", "trainer") == "central_gradient":
            raise HTTPException(
                status_code=422,
                detail="central_gradient has no central scorer; Learner probe is unsupported",
            )
        if (any(isinstance(length, bool) or not isinstance(length, int) or length < 2
                for length in request.lengths)
                or len(set(request.lengths)) != len(request.lengths)
                or request.lengths[0] != 64
                or max(request.lengths) > int(cfg["learner"]["max_sequence_length"])):
            raise HTTPException(status_code=422, detail="probe lengths must be unique, start with 64, and fit the configured sequence limit")
        async with state_lock:
            if app.state.active_task is not None and not app.state.active_task.done():
                raise HTTPException(status_code=409, detail="A run or probe is already active")
            run_id = uuid.uuid4().hex
            state = RunState(run_id, {"run_id": run_id, "kind": "probe", "status": "running",
                     "started_at": time.time(), "completed_updates": 0,
                     "filtered_groups": 0, "groups": [], "metrics": {"learner": {},
                     "workers": {}, environment.metrics_key: {}}, "summary": None, "error": None})
            app.state.runs[run_id] = state
            app.state.latest_run_id = run_id
            app.state.active_run_id = run_id
            app.state.active_task = asyncio.create_task(run_probe(run_id, request.lengths),
                                                        name=f"rl-probe-{run_id}")
        return {"run_id": run_id, "status": "running", "kind": "probe"}

    async def run_probe(run_id: str, lengths: list[int]) -> None:
        state = app.state.runs[run_id]
        output_dir = Path(cfg["run_dir"]) / "probes" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            async with update_lock:
                result = await compute_owner.run(_run_engine_probe, engine, output_dir, lengths)
            state["summary"] = result
            status = result.get("status") if isinstance(result, dict) else None
            if not isinstance(status, str) or status not in {"passed", "failed"}:
                state["status"] = "failed"
                state["error"] = {
                    "type": "InvalidProbeResult",
                    "message": "Engine probe returned an invalid status; expected 'passed' or 'failed'",
                }
            elif status == "passed":
                state["status"] = "succeeded"
            else:
                state["status"] = "failed"
                diagnostic = {
                    "type": "ProbeFailed",
                    "message": "Engine probe reported status='failed'",
                }
                if result.get("error") is not None:
                    diagnostic["diagnostic_error"] = result["error"]
                failures = result.get("failures")
                if isinstance(failures, list) and failures:
                    diagnostic["failures"] = failures
                state["error"] = diagnostic
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
            write_json_atomic(output_dir / "probe-results.json", {
                "status": "failed", "diagnostic_only": True,
                "sample_class": "public diagnostic token fixture; not ARC data or ARC reward evidence",
                "policy_version": int(engine.policy_version),
                "error": state["error"], "finished_at": time.time(),
            })
        finally:
            state["finished_at"] = time.time()
            state["elapsed_seconds"] = state["finished_at"] - state["started_at"]
            app.state.active_run_id = None


    def bepa_status_context():
        run = app.state.runs.get(app.state.active_run_id or app.state.latest_run_id) or {}
        stream = app.state.rollout_stream
        held = stream.deferred_items() if stream is not None else ()
        holds = {}
        for _ordinal, _policy, item in held:
            task_id = item["task"].task_id
            held_at = item.get("rescue_hold_started_at")
            holds[task_id] = {"hold_started_at": held_at,
                              "hold_deadline": (float(held_at) + float(bepa.settings["teacher"]["budget"]["job_timeout_seconds"]))
                              if held_at is not None and bepa is not None else None}
        ids = [cohort["batch_id"] for cohort in run.get("cohorts", []) if cohort.get("batch_id")]
        groups = run.get("groups", [])
        return {
            "current_policy_version": int(engine.policy_version),
            "run_id": run.get("run_id"), "held_task_ids": list(holds),
            "run_batch_ids": ids[-512:], "task_limit": 20,
        }, {
            "optimizer_consumed_replay_groups": sum(
                bool(group.get("optimizer_consumed") and group.get("replay_reservation"))
                for group in groups),
            "held_groups": len(held), "holds": holds,
            "observed_cohort_count": len(ids),
            "batch_coverage_truncated": len(ids) > 512,
        }

    from ..bepa.api import attach_routes
    attach_routes(app, bepa, _require_token, start_bepa_bootstrap,
                  status_context=bepa_status_context)
    if arc_app is not None:
        app.mount("/arc", arc_app, name="trusted-arc")
    app.router.lifespan_context = make_learner_lifespan(
        app,
        engine=engine,
        recovered_gradient_step_callback=recovered_gradient_step_callback,
        publish_current=publish_current,
        sync_workers=sync_workers,
        latest_policy=latest_policy,
    )
    return app
