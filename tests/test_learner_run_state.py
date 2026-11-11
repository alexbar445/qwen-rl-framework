from __future__ import annotations

import json

import pytest

from qwen_rl.learner.run.state import PolicyReferenceLedger, RunExecutionState, RunState


def test_run_state_remains_json_compatible_and_guards_terminal_transitions():
    state = RunState("run-1", {"run_id": "run-1", "status": "running", "groups": []})
    state.set_stage("optimizer")
    state.finish(
        status="succeeded",
        finished_at=12.0,
        elapsed_seconds=3.0,
        committed_policy_version=4,
        summary={"updates": 4},
    )

    assert isinstance(state, dict)
    assert json.loads(json.dumps(state))["committed_policy_version"] == 4
    with pytest.raises(RuntimeError, match="terminal run"):
        state.set_stage("worker_sync")
    with pytest.raises(RuntimeError, match="terminal status changed"):
        state.finish(
            status="failed",
            finished_at=13.0,
            elapsed_seconds=4.0,
            committed_policy_version=4,
            summary={},
        )


def test_run_state_failure_keeps_the_original_failure_stage_and_error_shape():
    state = RunState("run-2", {"run_id": "run-2", "status": "running"})
    state.set_stage("learner_update")
    state.fail(RuntimeError("durable commit failed"))

    assert state["status"] == "failed"
    assert state["failure_stage"] == "learner_update"
    assert state["error"] == {"type": "RuntimeError", "message": "durable commit failed"}


def test_execution_state_keeps_per_run_cursors_and_owned_resources_separate():
    record = RunState("run-3", {"run_id": "run-3", "status": "running"})
    references = PolicyReferenceLedger(engine=object(), shared_refs={})
    run = RunExecutionState(
        record=record,
        request=object(),
        started_monotonic=1.0,
        selected_tasks=["task"],
        target_steps=3,
        remaining_steps=1,
        completed_at_start=2,
        task_cursor_base=4,
        needed_groups=2,
        references=references,
        task_index=4,
        group_index=4,
    )

    assert run.record is record
    assert run.remaining_steps == run.target_steps - run.completed_at_start
    assert run.task_index == run.group_index == 4
    assert run.references.demand_refs == {}
    assert run.active_task_ids == set()


def test_execution_state_persists_task_owners_and_rescue_queue():
    record = RunState("run-rescue", {"run_id": "run-rescue", "status": "running"})
    run = RunExecutionState(
        record=record,
        request=object(),
        started_monotonic=1.0,
        selected_tasks=["task-a", "task-b"],
        target_steps=2,
        remaining_steps=2,
        completed_at_start=0,
        task_cursor_base=0,
        needed_groups=2,
        references=PolicyReferenceLedger(engine=object(), shared_refs={}),
    )
    run.task_worker_owners["task-a"] = "worker-a"
    run.enqueue_rescue_ready_task("task-a")
    run.enqueue_rescue_ready_task("task-a")
    run.enqueue_rescue_ready_task("task-b")
    assert run.pop_rescue_ready_task({"task-a"}) == "task-b"
    assert run.rescue_ready_task_ids == ["task-a"]
    assert json.loads(json.dumps(record))["task_worker_owners"] == {"task-a": "worker-a"}

    resumed = RunExecutionState(
        record=RunState.from_record(record),
        request=object(),
        started_monotonic=2.0,
        selected_tasks=["task-a", "task-b"],
        target_steps=2,
        remaining_steps=2,
        completed_at_start=0,
        task_cursor_base=0,
        needed_groups=2,
        references=PolicyReferenceLedger(engine=object(), shared_refs={}),
    )
    assert resumed.task_worker_owners == {"task-a": "worker-a"}
    assert resumed.rescue_ready_task_ids == ["task-a"]
