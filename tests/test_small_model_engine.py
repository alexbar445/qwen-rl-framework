from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from qwen_rl.training.checkpoint import HFCheckpointError, HFCheckpointStore
from qwen_rl.training.hf_engine import HFEngine


def _fixture_components(seed: int = 7):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {
        "[UNK]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[PAD]": 3,
        "Question": 4,
        ":": 5,
        "Is": 6,
        "sky": 7,
        "blue": 8,
        "?": 9,
        "Answer": 10,
        "yes": 11,
        "no": 12,
    }
    raw = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        _attn_implementation="eager",
    )
    model = LlamaForCausalLM(config)
    return model, tokenizer


def _config(model_dir: Path) -> dict:
    return {
        "model_id": str(model_dir),
        "model_revision": "local-fixture/v1",
        "tokenizer_id": str(model_dir),
        "tokenizer_revision": "local-fixture/v1",
        "device": "cpu",
        "dtype": "float32",
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_targets": ["q_proj", "v_proj"],
        "lora_dropout": 0.0,
        "group_size": 4,
        "learning_rate": 1e-3,
        "seed": 7,
        "epsilon": 0.5,
        "clip_ratio": 0.2,
        "importance_cap": 3.0,
        "num_threads": 1,
        "max_prompt_tokens": 32,
        "choices": [" yes", " no"],
        "source_identity": {"dataset": "local-fixture/v1"},
        "reward_schema": {"name": "binary-choice/v1"},
    }


def _save_fixture_model(model_dir: Path, seed: int = 7) -> Path:
    if not model_dir.exists():
        model, tokenizer = _fixture_components(seed)
        model_dir.mkdir(parents=True)
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
    return model_dir


def _make_engine(run_dir: Path, *, resume: bool = False, seed: int = 7) -> HFEngine:
    model_dir = _save_fixture_model(run_dir.parent / "tiny-llama", seed)
    return HFEngine(_config(model_dir), run_dir, resume=resume)


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def _mixed_group(engine: HFEngine):
    for attempt in range(64):
        rows = engine.sample_group("Is sky blue?", 4, engine.policy_version)
        if {row["choice_index"] for row in rows} == {0, 1}:
            for row in rows:
                row["group_id"] = f"fixture-group-{attempt}"
                row["reward"] = float(row["choice_index"] == 0)
            return rows
    raise AssertionError("epsilon-mixture failed to sample both choices in 64 attempts")


def _mixed_group_from_policy(engine: HFEngine, *, version: int, group_id: str):
    for _attempt in range(64):
        rows = engine.sample_group("Is sky blue?", 4, version)
        if {row["choice_index"] for row in rows} == {0, 1}:
            for row in rows:
                row["group_id"] = group_id
                row["reward"] = float(row["choice_index"] == 0)
            return rows
    raise AssertionError("epsilon-mixture failed to sample both choices in 64 attempts")


def test_sample_group_uses_actual_epsilon_mixture_and_retains_requested_policy(tmp_path):
    engine = _make_engine(tmp_path / "sample")
    outputs = engine.sample_group("Is sky blue?", 4, 0)

    assert len(outputs) == 4
    assert all({"text", "completion", "input_ids", "policy_mask", "behavior_logprobs", "choice_index"} <= row.keys()
               for row in outputs)
    assert all(len(row["input_ids"]) == len(row["policy_mask"]) == len(row["behavior_logprobs"])
               for row in outputs)
    assert all(row["policy_mask"][-1] == 1 and sum(row["policy_mask"]) == 1 for row in outputs)
    for row in outputs:
        p_logprobs = engine.logprobs(row["input_ids"], row["policy_mask"])
        p = torch.tensor([p_logprobs[-1]])
        p_other = engine.logprobs(
            [*row["input_ids"][:-1], engine.choice_token_ids[1 - row["choice_index"]]],
            row["policy_mask"],
        )[-1]
        ordered = [p.item(), p_other] if row["choice_index"] == 0 else [p_other, p.item()]
        selected = torch.tensor(ordered).softmax(dim=0)[row["choice_index"]]
        expected = torch.log((1 - engine.epsilon) * selected + engine.epsilon / 2)
        assert row["behavior_logprobs"][-1] == pytest.approx(float(expected), abs=2e-5)

    before = engine.audit_state()["adapter_state"]
    engine.acquire_policy(0)
    engine.update(_mixed_group(engine), recovery_context={"next_task_index": 1, "cursor": 1})
    engine.sample_group("Is sky blue?", 4, 0)
    engine.release_policy(0)
    after = engine.audit_state()["adapter_state"]
    assert any(not torch.equal(before[name], after[name]) for name in before)
    assert engine.policy_version == 1
    assert engine.policy_record()["evidence"]["publication_scope"] == "local_only"
    assert engine.policy_record()["evidence"]["identity_kind"] == "metadata_id_not_content_sha256"
    engine.shutdown()


def test_update_and_fresh_process_resume_restore_full_training_state(tmp_path):
    run_dir = tmp_path / "resume"
    engine = _make_engine(run_dir)
    update = engine.update(
        _mixed_group(engine),
        recovery_context={"run_id": "fixture-run", "next_task_index": 1, "next_group_index": 1},
    )
    assert update["updated"] is True
    assert update["policy_version"] == 1
    assert update["gradient_norm_before_clip"] > 0
    assert update["policy_tokens"] == 4
    assert update["publication"]["version"] == 1
    assert engine.scheduler.last_epoch == 1
    expected = engine.audit_state()
    engine.shutdown()

    resumed = _make_engine(run_dir, resume=True)
    actual = resumed.audit_state()
    _assert_nested_equal(expected, actual)
    assert resumed.recovery_metadata["resumed"] is True
    assert resumed.recovery_metadata["update_context"] == {
        "run_id": "fixture-run", "next_task_index": 1, "next_group_index": 1,
    }
    resumed.shutdown()


def test_lag_one_uses_behavior_snapshot_for_b_and_current_policy_for_proximal(tmp_path):
    engine = _make_engine(tmp_path / "lag-one")
    engine.acquire_policy(0)
    engine.update(_mixed_group(engine))
    rows = _mixed_group_from_policy(engine, version=0, group_id="lagged-group")

    expected_weights = []
    for row in rows:
        proximal = engine.logprobs(row["input_ids"], row["policy_mask"])[-1]
        expected_weights.append(min(
            math.exp(proximal - row["behavior_logprobs"][-1]), engine.importance_cap,
        ))
    result = engine.update(rows)

    assert result["policy_lag"] == 1
    assert result["behavior_version"] == 0
    assert result["learner_start_version"] == 1
    assert result["mean_importance_correction"] == pytest.approx(sum(expected_weights) / 4, rel=2e-5)
    assert engine.policy_version == 2
    engine.release_policy(0)
    engine.shutdown()


def test_zero_variance_and_incomplete_checkpoint_fail_closed(tmp_path):
    run_dir = tmp_path / "invalid"
    engine = _make_engine(run_dir)
    rows = _mixed_group(engine)
    for row in rows:
        row["reward"] = 1.0
    with pytest.raises(ValueError, match="zero-variance"):
        engine.update(rows)
    assert engine.policy_version == 0
    assert not engine.optimizer.state
    engine.shutdown()

    store = HFCheckpointStore(run_dir / "checkpoints")
    payload = store.load()
    del payload["scheduler_state"]
    torch.save(payload, store.latest_path)
    with pytest.raises(HFCheckpointError, match="schema or required state"):
        _make_engine(run_dir, resume=True)


def test_public_cli_trains_and_resumes_saved_local_llama(tmp_path):
    import os
    import subprocess
    import sys

    def invoke(args):
        result = subprocess.run(
            [sys.executable, "-m", "qwen_rl", "train", *args],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    model_dir = _save_fixture_model(tmp_path / "cli-tiny-llama")
    data_path = tmp_path / "tasks.jsonl"
    data_path.write_text(json.dumps({
        "task_id": "binary-1",
        "prompt": "Is sky blue?",
        "answer": "yes",
        "split": "train",
    }) + "\n", encoding="utf-8")
    recipe = {
        "schema": "qwen-rl-training/v1",
        "model": {
            "id": str(model_dir),
            "revision": "local-fixture/v1",
            "device": "cpu",
            "dtype": "float32",
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_targets": ["q_proj", "v_proj"],
        },
        "training": {
            "seed": 7,
            "group_size": 4,
            "learning_rate": 0.001,
            "epsilon": 0.5,
            "clip_ratio": 0.2,
            "importance_cap": 3.0,
            "num_threads": 1,
            "max_prompt_tokens": 32,
            "max_filtered_groups": 64,
        },
        "task": {
            "data": str(data_path),
            "choices": [" yes", " no"],
            "reward": "qwen_rl.training.tasks:exact_match_reward",
            "reward_schema": "binary-choice/v1",
            "source_identity": "local-cli-fixture/v1",
        },
    }
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")

    split_run = tmp_path / "split-run"
    invoke(["--config", str(config_path), "--run-dir", str(split_run), "--steps", "1"])
    resumed_result = invoke(["--run-dir", str(split_run), "--steps", "2", "--resume"])
    assert resumed_result["policy_version"] == 2
    assert resumed_result["status"] == "succeeded"

    continuous_run = tmp_path / "continuous-run"
    continuous_result = invoke(["--config", str(config_path), "--run-dir", str(continuous_run), "--steps", "2"])
    assert continuous_result["policy_version"] == 2

    split_state = HFCheckpointStore(split_run / "checkpoints").load()
    continuous_state = HFCheckpointStore(continuous_run / "checkpoints").load()
    for field in ("adapter_state", "optimizer_state", "scheduler_state", "rng_state"):
        _assert_nested_equal(split_state[field], continuous_state[field])
