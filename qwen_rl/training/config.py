"""One strict, lightweight boundary for the local small-model recipe."""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

import yaml


SCHEMA = "qwen-rl-training/v1"


class TrainingConfigError(ValueError):
    pass


class _Loader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise TrainingConfigError("configuration keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _section(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{name} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise TrainingConfigError(f"unknown {name} fields: {unknown}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise TrainingConfigError(f"{name} must be a nonempty string")
    return value


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise TrainingConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, low: float, high: float, *, open_low=False, open_high=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TrainingConfigError(f"{name} must be finite")
    if value < low or value > high or open_low and value == low or open_high and value == high:
        raise TrainingConfigError(f"{name} is outside its supported range")
    return float(value)


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Normalize external YAML once; runtime components consume this typed shape."""
    path = Path(path).expanduser().resolve(strict=True)
    if path.stat().st_size > 1 << 20:
        raise TrainingConfigError("configuration exceeds 1 MiB")
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)
    except yaml.YAMLError as exc:
        raise TrainingConfigError("invalid training YAML") from exc
    root = _section(raw, "training recipe", {"schema", "model", "training", "task"})
    if root.get("schema") != SCHEMA:
        raise TrainingConfigError(f"schema must be {SCHEMA}")
    model = _section(root.get("model"), "model", {
        "id", "revision", "device", "dtype", "lora_rank", "lora_alpha", "lora_targets",
    })
    settings = _section(root.get("training", {}), "training", {
        "seed", "group_size", "learning_rate", "epsilon", "clip_ratio", "importance_cap",
        "num_threads", "max_prompt_tokens", "max_filtered_groups",
    })
    task = _section(root.get("task"), "task", {
        "data", "choices", "reward", "reward_schema", "source_identity",
    })
    model_id = _text(model.get("id"), "model.id")
    local = Path(model_id).expanduser()
    if not local.is_absolute():
        local = path.parent / local
    is_local = model_id.startswith(("/", ".", "~"))
    if is_local:
        if not local.is_dir():
            raise TrainingConfigError("local model directory does not exist")
        model_id = str(local.resolve())
    revision = _text(model.get("revision"), "model.revision")
    if not is_local and re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise TrainingConfigError("Hub model.revision must be an immutable 40-character commit")
    device = model.get("device", "cpu")
    if device != "cpu":
        raise TrainingConfigError("the local small-model profile currently supports model.device=cpu only")
    dtype = model.get("dtype", "float32")
    if dtype != "float32":
        raise TrainingConfigError("the local small-model profile currently supports model.dtype=float32 only")
    targets = model.get("lora_targets", ["q_proj", "v_proj"])
    if (not isinstance(targets, list) or not targets or len(set(map(str, targets))) != len(targets)
            or any(not isinstance(x, str) or not x for x in targets)):
        raise TrainingConfigError("model.lora_targets must be a nonempty list of unique names")
    if targets != ["q_proj", "v_proj"]:
        raise TrainingConfigError("this reference profile uses model.lora_targets=[q_proj, v_proj]")
    data = Path(_text(task.get("data"), "task.data")).expanduser()
    data = (data if data.is_absolute() else path.parent / data).resolve(strict=True)
    if not data.is_file():
        raise TrainingConfigError("task.data must be a JSONL file")
    choices = task.get("choices")
    if (not isinstance(choices, list) or len(choices) != 2
            or any(not isinstance(x, str) or not x.strip() for x in choices)
            or len(set(x.strip() for x in choices)) != len(choices)):
        raise TrainingConfigError("task.choices needs exactly two distinct nonempty answer strings")
    reward_fn = _text(task.get("reward", "qwen_rl.training.tasks:exact_match_reward"), "task.reward")
    if re.fullmatch(r"[A-Za-z_][\w.]*(?::[A-Za-z_]\w*)", reward_fn) is None:
        raise TrainingConfigError("task.reward must identify an importable module:function")
    seed = _integer(settings.get("seed", 7), "training.seed", 0)
    if seed >= 2**32:
        raise TrainingConfigError("training.seed must be less than 2**32")
    return {
        "model_id": model_id, "model_revision": revision, "device": device, "dtype": dtype,
        "lora_rank": _integer(model.get("lora_rank", 4), "model.lora_rank"),
        "lora_alpha": _integer(model.get("lora_alpha", 8), "model.lora_alpha"),
        "lora_targets": list(targets), "seed": seed,
        "group_size": _integer(settings.get("group_size", 4), "training.group_size", 2),
        "learning_rate": _number(settings.get("learning_rate", 0.001), "training.learning_rate", 0, 1, open_low=True),
        "epsilon": _number(settings.get("epsilon", 0.5), "training.epsilon", 0, 1, open_low=True, open_high=True),
        "clip_ratio": _number(settings.get("clip_ratio", 0.2), "training.clip_ratio", 0, 1, open_low=True, open_high=True),
        "importance_cap": _number(settings.get("importance_cap", 3), "training.importance_cap", 1, 100),
        "num_threads": _integer(settings.get("num_threads", 2), "training.num_threads"),
        "max_prompt_tokens": _integer(settings.get("max_prompt_tokens", 256), "training.max_prompt_tokens"),
        "max_filtered_groups": _integer(settings.get("max_filtered_groups", 64), "training.max_filtered_groups"),
        "data_path": str(data), "choices": list(choices), "reward_fn": reward_fn,
        "reward_schema": _text(task.get("reward_schema"), "task.reward_schema"),
        "source_identity": _text(task.get("source_identity"), "task.source_identity"),
    }
