from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qwen_rl.training.runner import train


_MODEL_IDENTITY = {
    "model_id": "/local/test-model",
    "model_revision": "operator-test-v1",
    "source_identity_mode": "operator_declared_local_revision",
    "revision_is_immutable_hub_commit": False,
    "tokenizer_id": "/local/test-model",
    "tokenizer_revision": "operator-test-v1",
    "device": "cpu",
    "dtype": "float32",
    "attention_implementation": "eager",
    "optimizer": "torch.optim.AdamW",
    "scheduler": "torch.optim.lr_scheduler.LambdaLR",
    "lora_rank": 2,
    "lora_alpha": 4,
    "lora_targets": ["q_proj", "v_proj"],
    "lora_dropout": 0.0,
    "choices": [" yes", " no"],
    "choice_token_ids": [10, 11],
    "epsilon": 0.5,
    "behavior_probability": "test-mixture",
    "loss": "test-loss",
    "importance_correction_cap": 3.0,
    "policy_identity_kind": "lineage_version_metadata_id_not_content_sha256",
    "publication_scope": "local_only",
    "checkpoint_schema": "test/v1",
    "base_source_proof": "test local fixture; no payload hash",
    "base_identity": {"model_id": "/local/test-model", "model_revision": "operator-test-v1"},
    "adapter_structure": [{"name": "lora", "shape": [2, 2], "dtype": "torch.float32"}],
    "adapter_weights_finite": True,
}


class _FakeEngine:
    """Torch-free control fixture; it does not establish real-model qualification."""

    def __init__(self, config: dict, run_dir: Path, *, resume: bool, filter_first: bool) -> None:
        self.config = dict(config)
        self.run_dir = run_dir
        self.group_size = int(config["group_size"])
        self.checkpoint = run_dir / "checkpoints" / "latest.pt"
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.evidence = {
            **_MODEL_IDENTITY,
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "tokenizer_id": config["model_id"],
            "tokenizer_revision": config["model_revision"],
            "base_identity": {
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
            },
        }
        self.sample_calls = 0
        self.sample_versions: list[int] = []
        self.filter_first = filter_first
        if resume:
            stored = json.loads(self.checkpoint.read_text(encoding="utf-8"))
            self.policy_version = stored["policy_version"]
            self.lineage_id = stored["lineage_id"]
            self._context = stored["update_context"]
            self.optimizer_step = stored["optimizer_step"]
        else:
            self.policy_version = 0
            self.lineage_id = "fixture-lineage"
            self._context = None
            self.optimizer_step = 0
            self._save()

    def _save(self) -> None:
        payload = {
            "policy_version": self.policy_version,
            "lineage_id": self.lineage_id,
            "update_context": self._context,
            "optimizer_step": self.optimizer_step,
        }
        self.checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    @property
    def recovery_metadata(self) -> dict:
        return {
            "enabled": True,
            "lineage_id": self.lineage_id,
            "policy_version": self.policy_version,
            "durable_policy_version": self.policy_version,
            "update_context": self._context,
            "dirty": False,
        }

    def policy_record(self) -> dict:
        identifier = hashlib.sha256(f"{self.lineage_id}:{self.policy_version}".encode()).hexdigest()
        return {
            "version": self.policy_version,
            "directory": f"local://fixture/{self.lineage_id}/{self.policy_version}",
            "base_fingerprint": "fixture-base@v1",
            "artifact_fingerprint": "fixture-base@v1",
            "source_adapter_sha256": identifier,
            "adapter_hash": identifier,
            "quantization": {"mode": "none"},
            "files": [],
            "evidence": {
                "identity_kind": "metadata_id_not_content_sha256",
                "publication_scope": "local_only",
            },
        }

    def publish(self) -> dict:
        return self.policy_record()

    def sample_group(self, prompt: str, group_size: int, policy_version: int) -> list[dict]:
        del prompt
        assert group_size == self.group_size
        assert policy_version == self.policy_version
        self.sample_calls += 1
        self.sample_versions.append(policy_version)
        choices = [0, 0] if self.filter_first and self.sample_calls == 1 else [0, 1]
        return [
            {
                "completion": (" yes" if choice == 0 else " no"),
                "text": ("yes" if choice == 0 else "no"),
                "input_ids": [1, 10 + choice],
                "policy_mask": [0, 1],
                "behavior_logprobs": [0.0, -0.6931471805599453],
                "choice_index": choice,
                "policy_version": policy_version,
            }
            for choice in choices
        ]

    def update(self, rows: list[dict], *, recovery_context=None, **_kwargs) -> dict:
        assert len(rows) == self.group_size
        assert len({row["group_id"] for row in rows}) == 1
        old_version = self.policy_version
        self.policy_version += 1
        self.optimizer_step += 1
        self._context = recovery_context
        self._save()
        return {
            "updated": True,
            "policy_version": self.policy_version,
            "learner_start_version": old_version,
            "policy_tokens": len(rows),
            "trajectory_count": len(rows),
            "group_count": 1,
            "gradient_norm_before_clip": 1.0,
            "publication": self.policy_record(),
        }

    def acquire_policy(self, version: int) -> None:
        assert 0 <= version <= self.policy_version

    def release_policy(self, version: int) -> None:
        assert 0 <= version <= self.policy_version

    def shutdown(self) -> None:
        pass


def _config(tmp_path: Path) -> dict:
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "tasks.jsonl"
    data.write_text(
        json.dumps({"task_id": "yes-one", "prompt": "Say yes.", "answer": "yes"}) + "\n",
        encoding="utf-8",
    )
    return {
        "model_id": str(model),
        "model_revision": "operator-test-v1",
        "device": "cpu",
        "dtype": "float32",
        "lora_rank": 2,
        "lora_alpha": 4,
        "lora_targets": ["q_proj", "v_proj"],
        "group_size": 2,
        "learning_rate": 0.001,
        "seed": 7,
        "epsilon": 0.5,
        "clip_ratio": 0.2,
        "importance_cap": 3.0,
        "num_threads": 1,
        "max_prompt_tokens": 64,
        "max_filtered_groups": 64,
        "choices": [" yes", " no"],
        "data_path": str(data),
        "reward_fn": "qwen_rl.training.tasks:exact_match_reward",
        "reward_schema": "exact-match/v1",
        "source_identity": "fixture-task-jsonl/v1",
    }


def test_task_jsonl_and_exact_reward(tmp_path: Path) -> None:
    from qwen_rl.training.tasks import Task, exact_match_reward, load_tasks_jsonl

    data = tmp_path / "tasks.jsonl"
    data.write_text('{"task_id":"t","prompt":"p","answer":"yes"}\n', encoding="utf-8")
    task, = load_tasks_jsonl(data)
    assert task == Task("t", "p", "yes")
    assert exact_match_reward(task, " yes\n") == 1.0
    assert exact_match_reward(task, "Yes") == 0.0
    data.write_text('{"task_id":"t","task_id":"other","prompt":"p","answer":"yes"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_tasks_jsonl(data)


def test_runner_uses_shared_coordinator_and_counts_filtered_dispatch_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qwen_rl.training.runner as runner

    config = _config(tmp_path)
    engines: list[_FakeEngine] = []

    def load_engine(engine_config, run_dir, *, resume):
        engine = _FakeEngine(engine_config, run_dir, resume=resume, filter_first=not resume)
        engines.append(engine)
        return engine

    monkeypatch.setattr(runner, "_load_engine", load_engine)
    run_dir = tmp_path / "run"
    result = train(config, run_dir, steps=1)

    assert result["status"] == "succeeded"
    assert result["completed_updates"] == result["policy_version"] == 1
    assert result["policy_identity_kind"] == "lineage_version_metadata_id_not_content_sha256"
    assert result["publication_scope"] == "local_only"
    assert result["summary"]["filtered_groups"] == 1
    assert result["summary"]["policy_acks"]["local-shared-engine"]["remote_worker_qualification"] is False
    assert engines[0].sample_calls == 2
    context = engines[0].recovery_metadata["update_context"]
    assert context["task_cursor"]["next_task_index"] == 2
    assert context["task_cursor"]["ordered_task_ids"] == ["yes-one"]
    assert (run_dir / "input" / "tasks.jsonl").is_file()


def test_resume_uses_frozen_inputs_and_completed_extension_is_a_new_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qwen_rl.training.runner as runner

    config = _config(tmp_path)
    engines: list[_FakeEngine] = []

    def load_engine(engine_config, run_dir, *, resume):
        engine = _FakeEngine(engine_config, run_dir, resume=resume, filter_first=False)
        engines.append(engine)
        return engine

    monkeypatch.setattr(runner, "_load_engine", load_engine)
    run_dir = tmp_path / "run"
    first = train(config, run_dir, steps=1)
    source_path = Path(config["data_path"])
    source_path.unlink()

    same = train(None, run_dir, steps=1, resume=True)
    assert same["run_id"] == first["run_id"]
    assert same["policy_version"] == 1

    continuation = train(None, run_dir, steps=2, resume=True)
    assert continuation["status"] == "succeeded"
    assert continuation["policy_version"] == 2
    assert continuation["completed_updates"] == continuation["target_updates"] == 2
    assert continuation["updates_this_invocation"] == 1
    assert continuation["run_id"] != first["run_id"]
    assert continuation["parent_run_id"] == first["run_id"]
    assert continuation["segment"] == 1
    assert json.loads((run_dir / "checkpoints" / "latest.pt").read_text())["optimizer_step"] == 2


def test_resume_rejects_changed_config_and_missing_checkpoint_before_engine_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qwen_rl.training.runner as runner

    config = _config(tmp_path)
    run_dir = tmp_path / "run"

    def load_engine(engine_config, directory, *, resume):
        return _FakeEngine(engine_config, directory, resume=resume, filter_first=False)

    monkeypatch.setattr(runner, "_load_engine", load_engine)
    train(config, run_dir, steps=1)
    changed = dict(config, learning_rate=0.002)
    monkeypatch.setattr(
        runner,
        "_load_engine",
        lambda *_args, **_kwargs: pytest.fail("engine must not load for a config mismatch"),
    )
    with pytest.raises(ValueError, match="config differs"):
        train(changed, run_dir, steps=1, resume=True)
    (run_dir / "checkpoints" / "latest.pt").unlink()
    with pytest.raises(FileNotFoundError, match="complete checkpoints/latest.pt"):
        train(None, run_dir, steps=1, resume=True)


def test_local_sequential_updates_sample_each_group_from_committed_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qwen_rl.training.runner as runner

    config = _config(tmp_path)
    engines: list[_FakeEngine] = []

    def load_engine(engine_config, run_dir, *, resume):
        engine = _FakeEngine(engine_config, run_dir, resume=resume, filter_first=False)
        engines.append(engine)
        return engine

    monkeypatch.setattr(runner, "_load_engine", load_engine)
    result = train(config, tmp_path / "run", steps=2)

    assert result["status"] == "succeeded"
    assert engines[0].sample_versions == [0, 1]
