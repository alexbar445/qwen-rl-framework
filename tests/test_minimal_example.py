from __future__ import annotations

import subprocess
import sys

import pytest

from qwen_rl.examples.minimal import run_minimal
from qwen_rl.rollout.groups import PolicySnapshot, RolloutBatch


def test_minimal_example_updates_and_restores_full_toy_optimizer_state(tmp_path):
    result = run_minimal(tmp_path / "minimal")

    assert result["path"] == (
        "RunCoordinator -> AsyncGroupStream -> RolloutBatch -> "
        "UpdateTransaction -> LearnerRunEngineProtocol"
    )
    assert result["optimizer_updates"] == 3
    assert result["first_update_policy_version"] == 1
    assert result["continued_policy_version"] == result["restored_policy_version"] == 2
    assert result["recovery_match"] is True
    assert result["scope"].startswith("CPU toy protocol smoke")


def test_rollout_batch_rejects_incomplete_groups_and_wrong_policy_versions():
    digest = "a" * 64
    snapshot = PolicySnapshot(0, digest, "base", "artifact")
    row = {
        "group_id": "group-1",
        "policy_version": 0,
        "input_ids": [1, 2],
        "policy_mask": [0, 1],
        "behavior_logprobs": [0.0, -0.7],
    }
    complete_group = {
        "group_id": "group-1",
        "policy_version": 0,
        "record": {"adapter_sha256": digest},
        "trajectories": [row, {**row}],
        "worker_ids": ["worker-1", "worker-1"],
    }
    RolloutBatch(snapshot, (complete_group,)).validate(
        groups_per_update=1,
        group_size=2,
        learner_version=0,
    )

    incomplete = {
        **complete_group,
        "trajectories": [row],
        "worker_ids": ["worker-1"],
    }
    with pytest.raises(ValueError, match="incomplete group"):
        RolloutBatch(snapshot, (incomplete,)).validate(1, 2, 0)

    wrong_version = {
        **complete_group,
        "policy_version": 1,
        "trajectories": [{**row, "policy_version": 1}, {**row, "policy_version": 1}],
    }
    with pytest.raises(ValueError, match="identity differs"):
        RolloutBatch(snapshot, (wrong_version,)).validate(1, 2, 1)


def test_minimal_example_imports_without_torch_or_transformers():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import qwen_rl.examples.minimal; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_corrupt_checkpoint_pointer_is_rejected(tmp_path):
    import json
    from qwen_rl.examples.minimal import ToyLearnerEngine

    root = tmp_path / "checkpoint"
    ToyLearnerEngine(root)
    latest = root / "latest.json"
    pointer = json.loads(latest.read_text())
    pointer["policy_version"] += 1
    latest.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="checkpoint"):
        ToyLearnerEngine(root)


def test_demo_refuses_existing_nonempty_directory(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("existing user data")
    with pytest.raises(FileExistsError):
        run_minimal(tmp_path)
    assert marker.read_text() == "existing user data"
