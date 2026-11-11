"""Coordinator regression for empty central-gradient rounds and policy handoff fencing."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from qwen_rl.learner.run.coordinator import RunCoordinator, RunCoordinatorServices
from qwen_rl.contracts.environment import Reward
from qwen_rl.learner.run.handoff import PolicyHandoffCoordinator
from qwen_rl.learner.run.state import PolicyReferenceLedger, RunExecutionState, RunState
from qwen_rl.learner.telemetry import TrainingTelemetry
from qwen_rl.learner.stop_control import StopAfterCheckpointControl
from qwen_rl.rollout.groups import PolicySnapshot
from qwen_rl.rollout.pipeline import AsyncGroupStream

_EXECUTOR_WORKERS = {"executor-a": "worker-a", "executor-b": "worker-b"}


class _ComputeOwner:
    def __init__(self):
        self.before_update = None

    async def run(self, function, *args, **kwargs):
        if getattr(function, "__name__", None) == "update" and self.before_update is not None:
            await self.before_update()
        return function(*args, **kwargs)


class _ObservedAsyncGroupStream(AsyncGroupStream):
    def __init__(self, *args, events, counts, handoff_settled, **kwargs):
        self._events = events
        self._counts = counts
        self._handoff_settled = handoff_settled
        super().__init__(*args, **kwargs)

    async def claim_round(self, learner_start_version, round_id=None, **kwargs):
        if self._counts["claim_calls"] >= 2:
            assert self._handoff_settled["version_1"]
        self._counts["claim_calls"] += 1
        self._events.append(("claim_start", learner_start_version))
        claim = await super().claim_round(learner_start_version, round_id, **kwargs)
        self._events.append(("claim_result", claim.actual_groups, claim.requested_groups))
        return claim

    async def finish_round(self, round_id, *, update_committed):
        self._events.append((
            "finish_round", update_committed, self._counts["optimizer_steps"],
            self._counts["scheduler_steps"], self._counts["policy_version"],
        ))
        return await super().finish_round(round_id, update_committed=update_committed)

    async def publish(self, snapshot):
        self._events.append(("stream_publish", snapshot.version))
        return await super().publish(snapshot)

    async def prefetch_round(self, claimed_round_id, snapshot, round_id):
        self._events.append(("prefetch_round", snapshot.version))
        return await super().prefetch_round(claimed_round_id, snapshot, round_id)

    async def stop(self, *, drain=True):
        self._events.append("stream_stop")
        return await super().stop(drain=drain)


def _make_coordinator_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_steps: int,
    rewards_by_group: tuple[tuple[float, float, float], ...],
    max_filtered: int,
    stop_control=None,
):
    events = []
    counts = {"optimizer_steps": 0, "scheduler_steps": 0, "policy_version": 0, "claim_calls": 0}
    handoff_settled = {"version_1": False, "version_2": False}
    recovery_cursors = []
    initial_policy = {
        "version": 0,
        "base_fingerprint": "base-fingerprint",
        "artifact_fingerprint": "artifact-fingerprint",
        "source_adapter_sha256": "a" * 64,
        "files": {"adapter_model.safetensors": {"sha256": "a" * 64}},
    }

    class Engine:
        def __init__(self):
            self.config = {"update_backend": "central_gradient"}
            self.policy_version = 0
            self.recovery_metadata = {
                "enabled": True, "dirty": False, "policy_version": 0,
                "durable_policy_version": 0, "checkpoint_state": "durable",
                "snapshot_id": "0" * 32, "lineage_id": "lineage-test",
                "engine_epoch": "epoch-test", "update_context": None,
                "can_publish": True,
            }
            self.acquired = []
            self.released = []

        def acquire_policy(self, version):
            self.acquired.append(version)

        def release_policy(self, version):
            self.released.append(version)

        def begin_staged_round(self, round_id, first_ordinal):
            events.append(("begin_staged_round", round_id, first_ordinal))

        def note_sampling_outcome(self, round_id, ordinal):
            events.append(("sampling_outcome", round_id, ordinal))

        def stage_group(self, round_id, rows, *, logical_run_id=None):
            events.append(("stage_group", round_id, len(rows), logical_run_id))

        def cancel_staged_round(self, round_id=None):
            events.append(("cancel_staged_round", round_id))
            return 0

        def update(self, rows, **kwargs):
            assert counts["policy_version"] == self.policy_version
            assert len(rows) == (3 if self.policy_version == 0 else 6)
            events.append("engine_update")
            counts["optimizer_steps"] += 1
            counts["scheduler_steps"] += 1
            counts["policy_version"] += 1
            self.policy_version = counts["policy_version"]
            self.recovery_metadata.update(
                policy_version=self.policy_version,
                durable_policy_version=self.policy_version,
                checkpoint_state="durable",
                snapshot_id=str(self.policy_version) * 32,
                update_context=kwargs.get("recovery_context"),
            )
            adapter_hash = chr(ord("a") + self.policy_version) * 64
            return {
                "updated": True,
                "gradient_norm_before_clip": 0.5,
                "publication": {"adapter_hash": adapter_hash, "version": self.policy_version},
            }

        def wait_for_durable(self):
            assert self.recovery_metadata["durable_policy_version"] == self.policy_version

    engine = Engine()
    telemetry = TrainingTelemetry()
    runtime = SimpleNamespace(
        policies={0: initial_policy},
        policy_refs={},
        timings={"learner": {"update_seconds": 0.0, "publish_seconds": 0.0}},
        external_stage_timings={},
        training_telemetry=telemetry,
        active_sampling_groups=0,
        active_run_id="roundtest",
        last_policy_acks={},
        serve_policy_version=None,
        seed_scheduler=None,
        rollout_stream=None,
        runs={},
        run_commit_view=SimpleNamespace(
            begin=lambda *_args, **_kwargs: None,
            finish=lambda *_args, **_kwargs: None,
        ),
    )
    config = {
        "run_dir": str(tmp_path),
        "workers": [{"id": worker_id} for worker_id in _EXECUTOR_WORKERS.values()],
        "learner": {
            "update_backend": "central_gradient",
            "gradient_executor_worker_ids": dict(_EXECUTOR_WORKERS),
        },
        "rollout": {
            "group_size": 3,
            "concurrent_tasks": 2,
            "ready_queue_batches": 1,
        },
        "dapo": {"max_filtered_groups": max_filtered},
    }
    record = RunState("roundtest", {
        "run_id": "roundtest",
        "kind": "diagnostic",
        "status": "running",
        "completed_updates": 0,
        "requested_steps": target_steps,
        "initial_policy_version": 0,
        "initial_adapter_hash": "a" * 64,
        "filtered_groups": 0,
        "diagnostic_course": "cpu-round-regression",
        "groups": [],
        "cohorts": [],
        "metrics": {
            "learner": {
                "update_seconds": 0.0,
                "publish_seconds": 0.0,
                "worker_sync_seconds": 0.0,
            },
            "workers": {
                worker_id: {"generation_calls": 0, "generation_seconds": 0.0}
                for worker_id in _EXECUTOR_WORKERS.values()
            },
            "arc": {
                "scored_episodes": 0,
                "groups_completed": 0,
                "agent_service_http_seconds": 0.0,
            },
        },
    })
    runtime.runs["roundtest"] = record
    snapshot = PolicySnapshot(0, "a" * 64, "base-fingerprint", "artifact-fingerprint")
    references = PolicyReferenceLedger(engine, runtime.policy_refs)
    selected_tasks = [SimpleNamespace(task_id="task-a"), SimpleNamespace(task_id="task-b")]
    execution = RunExecutionState(
        record=record,
        request=SimpleNamespace(kind="diagnostic", steps=target_steps),
        started_monotonic=0.0,
        selected_tasks=selected_tasks,
        target_steps=target_steps,
        remaining_steps=target_steps,
        completed_at_start=0,
        task_cursor_base=0,
        needed_groups=len(_EXECUTOR_WORKERS),
        references=references,
        accepted_task_ids_by_version=defaultdict(set),
        cohorts_by_batch_id={},
    )

    async def generate_group(_engine, _managers, _config, task, group_id, policy, **kwargs):
        ordinal = int(group_id.split("-", 2)[1])
        owner_worker_id = kwargs["owner_worker_id"]
        rewards = rewards_by_group[ordinal]
        events.append(("generated", ordinal, policy.version, owner_worker_id))
        worker_ids = [owner_worker_id] * len(rewards)
        trajectories = [
            {
                "group_id": group_id,
                "row_kind": "fresh_student",
                "policy_version": policy.version,
                "input_ids": [1, 2],
                "policy_mask": [0, 1],
                "behavior_logprobs": [0.0, -0.1],
                "reward": reward,
            }
            for reward in rewards
        ]
        return {
            "task": task,
            "group_id": group_id,
            "policy_version": policy.version,
            "outputs": [SimpleNamespace(metrics=SimpleNamespace(compute_score=0.0)) for _ in rewards],
            "trajectories": trajectories,
            "worker_ids": worker_ids,
            "rewards": list(rewards),
            "exact": [False] * len(rewards),
            "costs": [0.0] * len(rewards),
            "counters": {owner_worker_id: {"calls": len(rewards), "seconds": 0.01}},
            "initial_prompt_prefix_sha256": "c" * 64,
        }

    diagnostic_fixture = SimpleNamespace(
        task_id="diagnostic-fixture", split="diagnostic", pool="diagnostic",
    )
    for task in selected_tasks:
        task.split = "train"
        task.pool = "test"

    class TestEnvironment:
        name = "test"
        prompt_identity = "test/v1"
        reward_schema = "test-reward/v1"
        metrics_key = "environment"
        source_identity = "test-task-source/v1"
        tasks = {task.task_id: task for task in selected_tasks}

        def fixture_task(self):
            return diagnostic_fixture

        async def generate_group(self, *args, **kwargs):
            result = await generate_group(*args, **kwargs)
            result["metrics"] = {
                "worker_generation": {
                    worker_id: {
                        "generation_calls": counts["calls"],
                        "generation_seconds": counts["seconds"],
                    }
                    for worker_id, counts in result["counters"].items()
                },
            }
            return result

    class TestRewardScorer:
        def score(self, _task, trajectories):
            return [Reward(float(row["reward"])) for row in trajectories]

    async def assert_workers_current(version):
        events.append(("assert_workers_current", version))

    async def sync_workers(policy):
        version = policy["version"]
        events.append(("sync_workers_start", version))
        await asyncio.sleep(0)
        acknowledgements = {
            worker_id: {"version": version, "status": "acknowledged"}
            for worker_id in _EXECUTOR_WORKERS.values()
        }
        runtime.last_policy_acks = acknowledgements
        events.append(("sync_workers_done", version))
        return acknowledgements

    async def publish_current(publication=None):
        version = engine.policy_version
        events.append(("publish_current", version))
        policy = {
            **initial_policy,
            "version": version,
            "source_adapter_sha256": (publication or {}).get("adapter_hash", "a" * 64),
            "files": {"adapter_model.safetensors": {
                "sha256": (publication or {}).get("adapter_hash", "a" * 64),
            }},
        }
        runtime.policies[version] = policy
        return policy

    def make_recovery_context(**kwargs):
        recovery_cursors.append({
            "next_task_index": kwargs["next_task_index"],
            "next_group_index": kwargs["next_group_index"],
        })
        return {
            "new_policy_version": int(engine.policy_version) + 1,
            "logical_job": {
                "run_id": kwargs["run_id"],
                "kind": kwargs["request"].kind,
                "requested_steps": int(kwargs["state"]["requested_steps"]),
                "completed_updates": kwargs["next_completed_updates"],
                "task_ids": [task.task_id for task in kwargs["selected_tasks"]],
            },
        }

    def stream_factory(**kwargs):
        return _ObservedAsyncGroupStream(
            **kwargs, events=events, counts=counts, handoff_settled=handoff_settled,
        )

    services = RunCoordinatorServices(
        config=config,
        runtime=runtime,
        engine=engine,
        managers={},
        worker_pool=SimpleNamespace(),
        environment=TestEnvironment(),
        reward_scorer=TestRewardScorer(),
        compute_owner=_ComputeOwner(),
        bepa=None,
        evaluation=SimpleNamespace(runtime=None, pause_training=False),
        seed_service=SimpleNamespace(engine=engine),
        update_lock=asyncio.Lock(),
        set_monitor_phase=lambda phase: events.append(("monitor_phase", phase)),
        latest_policy=lambda: runtime.policies[engine.policy_version],
        publish_current=publish_current,
        make_recovery_context=make_recovery_context,
        engine_recovery_metadata=lambda: engine.recovery_metadata,
        assert_workers_current=assert_workers_current,
        sync_workers=sync_workers,
        persist_group=lambda *_args, **_kwargs: "trajectory-artifact",
        stream_factory=stream_factory,
        measure_alignment=lambda *_args, **_kwargs: {"status": "not_measured"},
        alignment_from_proximal_scores=lambda *_args, **_kwargs: {"status": "not_measured"},
        log_alignment_warning=lambda *_args, **_kwargs: None,
        publication_from_update=lambda *_args, **_kwargs: pytest.fail("unexpected publication fallback"),
        prune_synced_policies=lambda *_args, **_kwargs: events.append("prune_synced_policies"),
        diagnostic_course="cpu-round-regression",
        diagnostic_run_kinds=frozenset({"diagnostic"}),
        alignment_thresholds={},
        stop_control=stop_control,
    )

    original_settle = PolicyHandoffCoordinator.settle

    async def observe_settle(coordinator, *, wait):
        had_pending = coordinator.execution.pending_policy_handoff is not None
        pending_version = (
            coordinator.execution.record["policy_handoffs"][-1]["policy_version"]
            if had_pending else None
        )
        result = await original_settle(coordinator, wait=wait)
        if wait and had_pending and coordinator.execution.pending_policy_handoff is None:
            handoff_settled[f"version_{pending_version}"] = True
            events.append(f"handoff_q{pending_version}_settled")
        return result

    monkeypatch.setattr(PolicyHandoffCoordinator, "settle", observe_settle)
    coordinator = RunCoordinator(services)
    return (
        coordinator, execution, snapshot, selected_tasks, runtime, events, counts,
        handoff_settled, recovery_cursors,
    )


def _run_case(tmp_path, monkeypatch, *, target_steps, rewards_by_group, max_filtered):
    async def run():
        case = _make_coordinator_case(
            tmp_path, monkeypatch,
            target_steps=target_steps,
            rewards_by_group=rewards_by_group,
            max_filtered=max_filtered,
        )
        (
            coordinator, execution, snapshot, selected_tasks, _runtime, events, counts,
            handoff_settled, recovery_cursors,
        ) = case
        await coordinator._execute_locked(
            execution,
            use_bepa=False,
            max_filtered=max_filtered,
            seed_observability=None,
            fixture=SimpleNamespace(task_id="diagnostic-fixture"),
        )
        return execution, snapshot, selected_tasks, events, counts, handoff_settled, recovery_cursors

    return asyncio.run(run())


def test_coordinator_retries_k0_then_waits_for_handoff_before_next_claim(tmp_path, monkeypatch):
    zero = (0.0, 0.0, 0.0)
    rewards = (
        zero,
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 2.0),
        zero,
        (0.0, 1.0, 2.0),
        (2.0, 1.0, 0.0),
    )
    execution, _snapshot, _tasks, events, counts, handoff_settled, recovery_cursors = _run_case(
        tmp_path, monkeypatch,
        target_steps=2,
        rewards_by_group=rewards,
        max_filtered=10,
    )

    assert events.index(("sync_workers_done", 0)) < events.index(("assert_workers_current", 0))
    assert events.index(("assert_workers_current", 0)) < events.index(("claim_start", 0))
    round_results = [item for item in events if isinstance(item, tuple) and item[0] == "claim_result"]
    assert round_results == [
        ("claim_result", 0, 2),
        ("claim_result", 1, 2),
        ("claim_result", 2, 2),
    ]
    sampling_plans = [item for item in events if isinstance(item, tuple)
                      and item[0] == 'begin_staged_round']
    assert [item[2] for item in sampling_plans] == [0, 2, 4]
    for _, round_id, first in sampling_plans:
        assert sorted(item[2] for item in events if isinstance(item, tuple)
                      and item[:2] == ('sampling_outcome', round_id)) == [first, first + 1]
    first_empty_finish = next(
        item for item in events if isinstance(item, tuple) and item[0] == "finish_round"
    )
    assert first_empty_finish == ("finish_round", False, 0, 0, 0)
    assert events.count("engine_update") == 2
    assert counts == {
        "optimizer_steps": 2,
        "scheduler_steps": 2,
        "policy_version": 2,
        "claim_calls": 3,
    }
    assert execution.record["completed_updates"] == 2
    assert execution.record["status"] == "succeeded"
    assert recovery_cursors == [
        {"next_task_index": 4, "next_group_index": 4},
        {"next_task_index": 6, "next_group_index": 6},
    ]
    assert execution.record["cohorts"][0]["status"] == "skipped_empty_round"
    assert execution.record["cohorts"][0]["optimizer_consumed"] is False
    assert execution.record["cohorts"][1]["status"] == "optimizer_committed"
    assert [row["policy_version"] for row in execution.record["policy_handoffs"]] == [1, 2]
    assert [row["status"] for row in execution.record["policy_handoffs"]] == ["published", "published"]
    assert handoff_settled == {"version_1": True, "version_2": True}
    third_claim = next(index for index, item in enumerate(events)
                       if isinstance(item, tuple) and item[0] == "claim_start"
                       and index > events.index("handoff_q1_settled"))
    assert events.index(("stream_publish", 1)) < events.index("handoff_q1_settled") < third_claim


def test_stop_control_pauses_after_one_durable_update_and_finalizes_before_completed(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "runs" / "roundtest.json"

    class FinalizationObservedControl(StopAfterCheckpointControl):
        def complete_paused_run(self):
            manifest = json.loads(manifest_path.read_text())
            assert manifest["status"] == "paused"
            assert manifest["completed_updates"] == 1
            assert manifest["requested_steps"] == 2
            super().complete_paused_run()

    control = FinalizationObservedControl(
        tmp_path / "control" / "stop.json",
        lineage_id="lineage-test", engine_epoch="epoch-test",
    )
    coordinator, execution, _snapshot, _tasks, runtime, events, counts, _settled, cursors = (
        _make_coordinator_case(
            tmp_path, monkeypatch,
            target_steps=2,
            rewards_by_group=((0.0, 1.0, 2.0), (2.0, 1.0, 0.0)),
            max_filtered=10,
            stop_control=control,
        )
    )
    engine = coordinator.services.engine

    async def request_while_update_is_owned():
        receipt, close_stream, idle_wait = control.request(
            "stop-operation",
            run_id="roundtest",
            completed_update=0,
            requested_steps=2,
            current_policy_version=engine.policy_version,
            recovery=engine.recovery_metadata,
            bepa=None,
            supported=True,
        )
        assert receipt["target_update"] == 1
        assert receipt["target_policy_version"] == 1
        assert close_stream is True and idle_wait is False
        await runtime.rollout_stream.stop_admission()

    coordinator.services.compute_owner.before_update = request_while_update_is_owned

    async def run():
        await coordinator.execute(
            "roundtest", SimpleNamespace(kind="diagnostic", steps=2, task_ids=None),
        )

    asyncio.run(run())

    assert counts["optimizer_steps"] == counts["scheduler_steps"] == 1
    assert counts["policy_version"] == 1
    assert counts["claim_calls"] == 1
    assert not any(
        isinstance(item, tuple) and item[0] == "prefetch_round" for item in events
    )
    assert ("sync_workers_start", 1) not in events
    assert execution.record["status"] == "paused"
    assert execution.record["completed_updates"] == 1
    assert execution.record["requested_steps"] == 2
    assert runtime.active_run_id is None
    assert control.get("stop-operation")["status"] == "completed"
    assert control.get("stop-operation")["bepa_commit_state"] == "not_applicable"
    assert json.loads(manifest_path.read_text())["stop_control"]["status"] == "completed"
    assert engine.recovery_metadata["update_context"]["logical_job"]["run_id"] == "roundtest"
    assert engine.recovery_metadata["update_context"]["logical_job"]["completed_updates"] == 1
