"""A CPU-only scalar-policy example using the framework's run coordinator."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qwen_rl.contracts.environment import Reward, RewardScorer, Task
from qwen_rl.learner.compute import ComputeOwner
from qwen_rl.learner.monitoring import RunCommitView
from qwen_rl.learner.run.coordinator import RunCoordinator, RunCoordinatorServices
from qwen_rl.learner.run.state import RunState
from qwen_rl.learner.telemetry import TrainingTelemetry
from qwen_rl.rollout.pipeline import AsyncGroupStream


_SNAPSHOT_SCHEMA = "qwen-rl-minimal-toy-state/v1"
_BASE_FINGERPRINT = hashlib.sha256(b"qwen-rl-minimal-scalar-policy/v1").hexdigest()
_WORKER_ID = "toy-cpu-worker"


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


@dataclass(frozen=True)
class BinaryChoiceTask:
    task_id: str
    split: str = "train"
    pool: str | None = None
    target: int = 1


class BinaryChoiceEnvironment:
    """A tiny deterministic two-action task; one group contains both actions."""

    name = "binary-choice"
    prompt_identity = "binary-choice/v1:prompt-token-1"
    reward_schema = "binary-choice-match/v1"
    metrics_key = "environment"
    source_identity = "binary-choice-task-source/v1"

    def __init__(self) -> None:
        self.tasks = {"choose-one": BinaryChoiceTask("choose-one")}

    def fixture_task(self) -> BinaryChoiceTask:
        return BinaryChoiceTask("minimal-fixture", split="diagnostic", pool="diagnostic")

    async def generate_group(
        self,
        engine: ToyLearnerEngine,
        managers: dict[str, Any],
        config: dict[str, Any],
        task: BinaryChoiceTask,
        group_id: str,
        policy: Any,
        *,
        worker_pool: Any,
        diagnostic_course: str | None = None,
        owner_worker_id: str | None = None,
        guided_plan: str | None = None,
    ) -> dict[str, Any]:
        del managers, config, worker_pool, diagnostic_course, owner_worker_id, guided_plan
        if int(policy.version) != engine.policy_version:
            raise ValueError("environment received a stale policy snapshot")

        actions = [0, 1]
        engine.rng.shuffle(actions)
        probability_one = engine.probability_one
        trajectories = []
        for action in actions:
            probability = probability_one if action else 1.0 - probability_one
            trajectories.append({
                "group_id": group_id,
                "policy_version": policy.version,
                "input_ids": [1, action + 2],
                "policy_mask": [0, 1],
                "behavior_logprobs": [0.0, math.log(probability)],
                "toy_action": action,
            })
        return {
            "trajectories": trajectories,
            "worker_ids": [_WORKER_ID, _WORKER_ID],
            "policy_version": policy.version,
            "metrics": {
                "worker_generation": {
                    _WORKER_ID: {"generation_calls": 2, "generation_seconds": 0.0},
                },
                "values": {"groups_generated": 1.0},
                "annotations": {"task": task.task_id},
            },
        }


class TargetRewardScorer(RewardScorer):
    """Reward action 1; the toy environment is not a language model benchmark."""

    def score(
        self,
        task: BinaryChoiceTask,
        trajectories: list[dict[str, Any]],
    ) -> list[Reward]:
        return [
            Reward(
                value=float(int(row["toy_action"]) == task.target),
                components={"matched_target": int(row["toy_action"]) == task.target},
            )
            for row in trajectories
        ]


class ToyLearnerEngine:
    """Scalar Bernoulli policy with real Adam state and an exact local snapshot."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        seed: int = 7,
        learning_rate: float = 0.05,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.snapshot_dir = self.checkpoint_dir / "snapshots"
        self.latest_path = self.checkpoint_dir / "latest.json"
        self.config = {
            "update_backend": "trainer",
            "parallelism": "single",
            "learning_rate": float(learning_rate),
        }
        self.group_size = 2
        self.policy_version = 0
        self.theta = 0.0
        self.first_moment = 0.0
        self.second_moment = 0.0
        self.optimizer_step = 0
        self.rng = random.Random(seed)
        self.lineage_id = uuid.uuid4().hex
        self.engine_epoch = uuid.uuid4().hex
        self.update_context: dict[str, Any] | None = None
        self._resumed = False
        self._snapshot_sha256: str | None = None
        self._loaded_snapshot: dict[str, Any] | None = None
        self._restore_or_initialize(seed)

    @property
    def probability_one(self) -> float:
        if self.theta >= 0:
            exp_term = math.exp(-self.theta)
            return 1.0 / (1.0 + exp_term)
        exp_term = math.exp(self.theta)
        return exp_term / (1.0 + exp_term)

    @property
    def adapter_hash(self) -> str:
        return hashlib.sha256(
            f"scalar-policy/v1:{self.theta.hex()}".encode("ascii")
        ).hexdigest()

    @property
    def recovery_metadata(self) -> dict[str, Any]:
        return {
            "schema": _SNAPSHOT_SCHEMA,
            "enabled": True,
            "mode": "resumed" if self._resumed else "fresh",
            "resumed": self._resumed,
            "lineage_id": self.lineage_id,
            "engine_epoch": self.engine_epoch,
            "policy_version": self.policy_version,
            "durable_policy_version": self.policy_version,
            "snapshot_id": f"policy-{self.policy_version}",
            "snapshot_sha256": self._snapshot_sha256,
            "snapshot_path": str(self.snapshot_dir / f"policy-{self.policy_version}.json"),
            "loaded_snapshot": self._loaded_snapshot,
            "update_context": self.update_context,
            "dirty": False,
        }

    @property
    def retained_policy_versions(self) -> tuple[int, ...]:
        return (self.policy_version,)

    def _state(self, *, update_context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema": _SNAPSHOT_SCHEMA,
            "lineage_id": self.lineage_id,
            "engine_epoch": self.engine_epoch,
            "policy_version": self.policy_version,
            "theta": self.theta,
            "adam": {
                "learning_rate": self.config["learning_rate"],
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "first_moment": self.first_moment,
                "second_moment": self.second_moment,
                "step": self.optimizer_step,
            },
            "rng_state": self.rng.getstate(),
            "update_context": update_context,
        }

    def _restore_or_initialize(self, seed: int) -> None:
        if self.latest_path.exists():
            pointer = json.loads(self.latest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(pointer, dict)
                or pointer.get("schema") != f"{_SNAPSHOT_SCHEMA}/latest"
                or type(pointer.get("policy_version")) is not int
                or not isinstance(pointer.get("snapshot"), str)
                or Path(pointer["snapshot"]).name != pointer["snapshot"]
                or not isinstance(pointer.get("sha256"), str)
            ):
                raise ValueError("unsupported toy checkpoint pointer schema")
            snapshot_path = self.snapshot_dir / pointer["snapshot"]
            raw = snapshot_path.read_bytes()
            state = json.loads(raw)
            if (
                not isinstance(state, dict)
                or state.get("schema") != _SNAPSHOT_SCHEMA
                or type(state.get("policy_version")) is not int
                or state["policy_version"] < 0
                or state["policy_version"] != pointer["policy_version"]
            ):
                raise ValueError("unsupported toy checkpoint schema")
            if pointer.get("sha256") != hashlib.sha256(raw).hexdigest():
                raise ValueError("toy checkpoint does not match its latest pointer")
            if float(state["adam"]["learning_rate"]) != self.config["learning_rate"]:
                raise ValueError("toy checkpoint learning rate differs from the requested engine")
            self.lineage_id = str(state["lineage_id"])
            self.engine_epoch = str(state["engine_epoch"])
            self.policy_version = int(state["policy_version"])
            self.theta = float(state["theta"])
            self.first_moment = float(state["adam"]["first_moment"])
            self.second_moment = float(state["adam"]["second_moment"])
            self.optimizer_step = int(state["adam"]["step"])
            self.rng.setstate(_as_tuple(state["rng_state"]))
            self.update_context = state.get("update_context")
            self._resumed = True
            self._snapshot_sha256 = pointer["sha256"]
            self._loaded_snapshot = {
                "policy_version": self.policy_version,
                "snapshot_id": pointer["snapshot"],
                "snapshot_sha256": pointer["sha256"],
                "path": str(snapshot_path),
            }
            return
        self.rng.seed(seed)
        self.save_recovery_snapshot(reason="lineage_initialized")

    def publish(self) -> dict[str, Any]:
        digest = self.adapter_hash
        return {
            "version": self.policy_version,
            "directory": str(self.checkpoint_dir),
            "base_fingerprint": _BASE_FINGERPRINT,
            "artifact_fingerprint": _BASE_FINGERPRINT,
            "quantization": {"mode": "toy_scalar"},
            "adapter_hash": digest,
            "source_adapter_sha256": digest,
            "files": [{"name": "scalar-policy.json", "sha256": digest, "size": 8}],
        }

    def policy_record(self) -> dict[str, Any]:
        publication = self.publish()
        return {
            "version": self.policy_version,
            "base_fingerprint": _BASE_FINGERPRINT,
            "artifact_fingerprint": _BASE_FINGERPRINT,
            "source_adapter_sha256": publication["adapter_hash"],
            "files": {"scalar-policy.json": {"sha256": publication["adapter_hash"]}},
        }

    def logprobs(self, input_ids: list[int], policy_mask: list[int]) -> list[float]:
        if len(input_ids) != len(policy_mask):
            raise ValueError("toy policy token and mask lengths differ")
        probability_one = self.probability_one
        result = [0.0] * len(input_ids)
        for index, active in enumerate(policy_mask):
            if active:
                action = input_ids[index] - 2
                if action not in (0, 1):
                    raise ValueError("toy policy received an unknown action token")
                probability = probability_one if action else 1.0 - probability_one
                result[index] = math.log(probability)
        return result

    def update(
        self,
        rows: list[dict[str, Any]],
        *,
        before_optimizer_step=None,
        recovery_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not rows:
            raise ValueError("toy Adam update requires rows")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["group_id"]), []).append(row)
        if any(len(group) != self.group_size for group in grouped.values()):
            raise ValueError("toy Adam update requires complete groups")

        probability_one = self.probability_one
        total_gradient = 0.0
        for group in grouped.values():
            rewards = [float(row["reward"]) for row in group]
            mean_reward = sum(rewards) / len(rewards)
            variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
            deviation = math.sqrt(variance)
            if deviation == 0.0:
                raise ValueError("toy GRPO group requires nonzero reward variance")
            for row, reward in zip(group, rewards, strict=True):
                action = int(row["toy_action"])
                advantage = (reward - mean_reward) / deviation
                total_gradient -= advantage * (action - probability_one) / len(rows)

        gradient_norm = abs(total_gradient)
        if not math.isfinite(gradient_norm) or gradient_norm == 0.0:
            raise ValueError("toy Adam update produced a zero or invalid gradient")
        if before_optimizer_step is not None:
            before_optimizer_step()

        self.optimizer_step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        self.first_moment = beta1 * self.first_moment + (1.0 - beta1) * total_gradient
        self.second_moment = beta2 * self.second_moment + (1.0 - beta2) * total_gradient**2
        corrected_first = self.first_moment / (1.0 - beta1**self.optimizer_step)
        corrected_second = self.second_moment / (1.0 - beta2**self.optimizer_step)
        self.theta -= self.config["learning_rate"] * corrected_first / (
            math.sqrt(corrected_second) + epsilon
        )
        self.policy_version += 1
        self.update_context = recovery_context
        snapshot = self.save_recovery_snapshot(
            update_context=recovery_context,
            reason="optimizer_update",
        )
        publication = self.publish()
        self._snapshot_sha256 = snapshot["sha256"]
        return {
            "updated": True,
            "policy_version": self.policy_version,
            "learner_start_version": self.policy_version - 1,
            "policy_tokens": len(rows),
            "trajectory_count": len(rows),
            "group_count": len(grouped),
            "gradient_norm_before_clip": gradient_norm,
            "publication": publication,
        }

    def save_recovery_snapshot(
        self,
        *,
        update_context: dict[str, Any] | None = None,
        reason: str = "explicit",
    ) -> dict[str, Any]:
        del reason
        if update_context is None:
            update_context = self.update_context
        state = self._state(update_context=update_context)
        state["adam"]["learning_rate"] = self.config["learning_rate"]
        raw = (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        name = f"policy-{self.policy_version}.json"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.snapshot_dir / name
        self._write_atomic(snapshot_path, raw)
        pointer = {
            "schema": f"{_SNAPSHOT_SCHEMA}/latest",
            "snapshot": name,
            "policy_version": self.policy_version,
            "sha256": digest,
        }
        pointer_raw = (json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        self._write_atomic(self.latest_path, pointer_raw)
        self.update_context = update_context
        self._snapshot_sha256 = digest
        return {"path": str(snapshot_path), "snapshot_id": name, "sha256": digest}

    @staticmethod
    def _write_atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def acquire_policy(self, version: int) -> None:
        if version != self.policy_version:
            raise ValueError("toy engine retains only its current policy")

    def release_policy(self, version: int) -> None:
        if not 0 <= version <= self.policy_version:
            raise ValueError("toy engine released an unknown policy")

    def shutdown(self) -> None:
        return None

    def state_values(self) -> tuple[Any, ...]:
        return (
            self.policy_version,
            self.theta,
            self.first_moment,
            self.second_moment,
            self.optimizer_step,
            self.rng.getstate(),
            self.update_context,
        )


def _coordinator(
    engine: ToyLearnerEngine,
    environment: BinaryChoiceEnvironment,
    scorer: TargetRewardScorer,
    run_dir: Path,
) -> tuple[RunCoordinator, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "runs").mkdir(exist_ok=True)
    policy = engine.policy_record()
    telemetry = TrainingTelemetry()
    runtime = SimpleNamespace(
        runs={},
        run_commit_view=RunCommitView(),
        training_telemetry=telemetry,
        policies={engine.policy_version: policy},
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
    config = {
        "run_dir": str(run_dir),
        "workers": [{"id": _WORKER_ID}],
        "learner": {
            "update_backend": "trainer",
            "parallelism": "single",
            "groups_per_update": 1,
        },
        "rollout": {
            "group_size": 2,
            "concurrent_tasks": 1,
            "ready_queue_batches": 1,
            "max_filtered_groups": 4,
            "task_ids": [],
            "category_sampling": False,
        },
        "dapo": {"updates": 1, "max_filtered_groups": 4},
        "evaluation": {"pause_training": False, "every_updates": 0},
    }

    async def publish_current(_already_published=None, *, pending_checkpoint=False):
        del pending_checkpoint
        published = engine.policy_record()
        runtime.policies[engine.policy_version] = published
        return published

    async def sync_workers(published):
        version = int(published["version"])
        acks = {_WORKER_ID: {"version": version, "status": "acknowledged"}}
        runtime.last_policy_acks = acks
        return acks

    def recovery_context(**kwargs):
        return {
            "schema": "qwen-rl-minimal-run-context/v1",
            "lineage_id": engine.lineage_id,
            "engine_epoch": engine.engine_epoch,
            "new_policy_version": engine.policy_version + 1,
            "logical_job": {
                "run_id": kwargs["run_id"],
                "kind": kwargs["request"].kind,
                "requested_steps": int(kwargs["state"]["requested_steps"]),
                "completed_updates": kwargs["next_completed_updates"],
                "task_ids": [task.task_id for task in kwargs["selected_tasks"]],
            },
            "task_cursor": {
                "next_task_index": kwargs["next_task_index"],
                "next_group_index": kwargs["next_group_index"],
            },
            "environment": environment.name,
            "reward_schema": environment.reward_schema,
        }

    services = RunCoordinatorServices(
        config=config,
        runtime=runtime,
        engine=engine,
        managers={},
        worker_pool=SimpleNamespace(),
        compute_owner=ComputeOwner(),
        bepa=None,
        evaluation=SimpleNamespace(
            runtime=None,
            pause_training=False,
            count=None,
            has_pending_requests=lambda: False,
        ),
        seed_service=SimpleNamespace(engine=engine),
        update_lock=asyncio.Lock(),
        set_monitor_phase=lambda _phase: None,
        latest_policy=lambda: runtime.policies[engine.policy_version],
        publish_current=publish_current,
        make_recovery_context=recovery_context,
        engine_recovery_metadata=lambda: engine.recovery_metadata,
        assert_workers_current=lambda _version: asyncio.sleep(0),
        sync_workers=sync_workers,
        persist_group=lambda *_args, **_kwargs: "toy://trajectory-group",
        stream_factory=lambda **kwargs: AsyncGroupStream(**kwargs),
        measure_alignment=lambda *_args, **_kwargs: {"status": "not_measured"},
        alignment_from_proximal_scores=lambda *_args, **_kwargs: {"status": "not_measured"},
        log_alignment_warning=lambda *_args, **_kwargs: None,
        publication_from_update=lambda result, _version: result["publication"],
        prune_synced_policies=lambda *_args, **_kwargs: [],
        diagnostic_course="minimal-toy",
        diagnostic_run_kinds=frozenset(),
        alignment_thresholds={},
        environment=environment,
        reward_scorer=scorer,
    )
    return RunCoordinator(services), runtime


async def run_training_update(
    engine: ToyLearnerEngine,
    checkpoint_dir: Path,
    *,
    run_id: str,
    steps: int = 1,
) -> RunState:
    environment = BinaryChoiceEnvironment()
    coordinator, runtime = _coordinator(
        engine, environment, TargetRewardScorer(), checkpoint_dir / "run",
    )
    metadata = engine.recovery_metadata
    previous_context = metadata.get("update_context") or {}
    prior_logical_job = previous_context.get("logical_job", {})
    resume_cursor = previous_context.get("task_cursor", {})
    state = RunState(run_id, {
        "run_id": run_id,
        "kind": "training",
        "status": "running",
        "failure_stage": "queued",
        "started_at": 0.0,
        "initial_policy_version": engine.policy_version,
        "initial_adapter_hash": engine.adapter_hash,
        "requested_steps": steps,
        "completed_updates": 0,
        "resume_task_cursor": dict(resume_cursor),
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
        "restored_from_run_id": prior_logical_job.get("run_id"),
    })
    runtime.runs[run_id] = state
    await coordinator.execute(
        run_id,
        SimpleNamespace(kind="training", steps=steps, task_ids=None),
    )
    if state.get("status") != "succeeded":
        raise RuntimeError(f"minimal coordinator run failed: {state.get('error')}")
    return state


def run_minimal(checkpoint_root: str | Path | None = None) -> dict[str, Any]:
    """Run one coordinator update, restore a new engine, and compare next updates."""
    if checkpoint_root is None:
        with tempfile.TemporaryDirectory(prefix="qwen-rl-minimal-") as temporary:
            result = run_minimal(Path(temporary))
            result["checkpoint"] = "temporary checkpoint verified and removed"
            result["checkpoint_persisted"] = False
            return result

    root = Path(checkpoint_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("minimal example checkpoint directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    first_dir = root / "first"
    continued_dir = root / "continued"
    restored_dir = root / "restored"
    engine = ToyLearnerEngine(first_dir)
    first_record = asyncio.run(run_training_update(
        engine, first_dir, run_id="minimal-first-update", steps=1,
    ))
    if first_record["completed_updates"] != 1 or engine.optimizer_step != 1:
        raise RuntimeError("the core coordinator did not commit the first Adam update")

    shutil.copytree(first_dir, continued_dir)
    shutil.copytree(first_dir, restored_dir)
    continued = ToyLearnerEngine(continued_dir)
    restored = ToyLearnerEngine(restored_dir)
    if not restored.recovery_metadata["resumed"]:
        raise RuntimeError("new toy engine did not restore the full checkpoint")
    continued_record = asyncio.run(run_training_update(
        continued, continued_dir, run_id="minimal-follow-up", steps=1,
    ))
    restored_record = asyncio.run(run_training_update(
        restored, restored_dir, run_id="minimal-follow-up", steps=1,
    ))
    if continued.state_values() != restored.state_values():
        raise RuntimeError("restored continuation differs from uninterrupted toy state")
    return {
        "framework_namespace": "qwen-rl-framework",
        "path": "RunCoordinator -> AsyncGroupStream -> RolloutBatch -> UpdateTransaction -> LearnerRunEngineProtocol",
        "environment": "binary-choice",
        "backend": "stdlib scalar Bernoulli policy with Adam",
        "optimizer_updates": 3,
        "first_update_policy_version": 1,
        "continued_policy_version": continued.policy_version,
        "restored_policy_version": restored.policy_version,
        "recovery_match": True,
        "group_size": 2,
        "completed_updates": continued_record["completed_updates"],
        "restored_run_status": restored_record["status"],
        "checkpoint": str(restored.recovery_metadata["snapshot_path"]),
        "checkpoint_persisted": True,
        "scope": "CPU toy protocol smoke only; no LLM, GPU, or multi-node claim",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_minimal(args.checkpoint_dir), sort_keys=True, indent=2))


__all__ = [
    "BinaryChoiceEnvironment",
    "BinaryChoiceTask",
    "TargetRewardScorer",
    "ToyLearnerEngine",
    "run_minimal",
    "run_training_update",
]
