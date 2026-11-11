"""User-facing configuration and CLI checks need no model dependencies."""
import json
from pathlib import Path
import sys

import pytest
import yaml

from qwen_rl.training.config import TrainingConfigError, load_training_config


def _recipe(tmp_path):
    (tmp_path / "tasks.jsonl").write_text('{"task_id":"one","prompt":"True?","answer":"yes"}\n')
    return {
        "schema": "qwen-rl-training/v1",
        "model": {"id": "HuggingFaceTB/SmolLM2-135M", "revision": "a" * 40},
        "task": {"data": "tasks.jsonl", "choices": [" yes", " no"],
                 "source_identity": "fixture/v1", "reward_schema": "exact/v1"},
    }


def _load(tmp_path, recipe):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return load_training_config(path)


def test_minimal_config_normalizes_defaults_and_relative_data(tmp_path):
    cfg = _load(tmp_path, _recipe(tmp_path))
    assert cfg["device"] == "cpu" and cfg["dtype"] == "float32"
    assert cfg["group_size"] == 4
    assert cfg["data_path"] == str(tmp_path / "tasks.jsonl")
    assert cfg["choices"] == [" yes", " no"]


@pytest.mark.parametrize("section,key,value", [
    ("model", "revision", "main"), ("model", "device", "cuda:0"),
    ("model", "dtype", "bfloat16"), ("model", "lora_alpha", 8.5),
    ("model", "lora_targets", ["k_proj"]), ("task", "choices", ["a", "b", "c"]),
    ("training", "epsilon", 0), ("training", "epsilon", 1),
    ("training", "clip_ratio", 1), ("training", "group_size", True),
    ("training", "learning_rate", float("nan")), ("training", "seed", 2**32),
])
def test_unsupported_recipe_values_fail_before_model_loading(tmp_path, section, key, value):
    recipe = _recipe(tmp_path)
    recipe.setdefault(section, {})[key] = value
    with pytest.raises(TrainingConfigError):
        _load(tmp_path, recipe)


def test_duplicate_yaml_keys_and_unknown_options_are_rejected(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text('schema: first\nschema: second\n')
    with pytest.raises(TrainingConfigError, match="unique"):
        load_training_config(path)
    recipe = _recipe(tmp_path)
    recipe["model"]["trust_remote_code"] = True
    with pytest.raises(TrainingConfigError, match="unknown"):
        _load(tmp_path, recipe)


def test_failed_training_result_has_nonzero_cli_status(monkeypatch, tmp_path, capsys):
    import qwen_rl.training.runner as runner
    from qwen_rl.training.cli import main
    monkeypatch.setattr(runner, "train", lambda *_a, **_k: {"status": "filtered_limit", "completed_updates": 0})
    assert main(["--resume", "--run-dir", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "filtered_limit"


def test_resume_missing_directory_does_not_create_it(tmp_path):
    from qwen_rl.training import train
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        train(None, missing, resume=True)
    assert not missing.exists()


def test_simple_public_api_does_not_load_model_runtime():
    import subprocess
    code = "from qwen_rl.training import Task, RewardFn, train; import sys; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
