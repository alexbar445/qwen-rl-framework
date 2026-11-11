"""CPU-only, strict configuration for the compression_recovery task."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_profile(path):
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ValueError("configuration keys must be unique strings")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping
    )
    data = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    if not isinstance(data, dict):
        raise ValueError("compute profile must be a mapping")
    return data


def objective(path):
    data = read_profile(path)
    training = data.get("training")
    if training is None:
        return "dapo"
    if (
        not isinstance(training, dict)
        or training.get("objective") != "compression_recovery"
    ):
        raise ValueError("unknown training.objective")
    return "compression_recovery"


def _keys(data, allowed, required, label):
    if (
        not isinstance(data, dict)
        or set(data) - set(allowed)
        or set(required) - set(data)
    ):
        raise ValueError(f"{label}: unknown or missing fields")


def _number(data, key, default, low, high, integer=False):
    value = data.get(key, default)
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"invalid {key}")
    return value


def file_spec(value, base, *, extra=()):
    _keys(
        value, ("path", "sha256", *extra), ("path", "sha256", *extra), "file identity"
    )
    if not isinstance(value["path"], str) or not value["path"].strip():
        raise ValueError("file path must be nonempty")
    if not isinstance(value["sha256"], str) or not re.fullmatch(
        "[0-9a-f]{64}", value["sha256"]
    ):
        raise ValueError("file identity requires a SHA256")
    path = (base / value["path"]).resolve()
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError(f"file SHA256 mismatch: {path}")
    return {**value, "path": str(path)}


@dataclass(frozen=True)
class CompressionConfig:
    path: Path
    run_dir: Path
    artifact: dict
    teacher_head: dict
    dataset_manifest: Path
    dataset_sha256: str
    trainable: tuple[str, ...]
    mla_bundle: Path | None
    mla_bundle_sha256: str | None
    gpu: int
    max_sequence_length: int
    records_per_update: int
    steps: int
    lr: float
    ce_weight: float
    kl_weight: float
    max_grad_norm: float
    vocab_chunk_rows: int
    gradient_checkpointing: bool
    activation_offload: bool
    recovery_mode: str
    max_wall_seconds: int
    vocab_scale_lr: float
    checkpoint_steps: int = 1


def load_config(path):
    path = Path(path).resolve()
    d = read_profile(path)
    _keys(
        d,
        (
            "schema_version",
            "run_dir",
            "artifact",
            "gpus",
            "training",
            "gradient_checkpointing",
            "activation_offload",
            "max_sequence_length",
            "microbatch_size",
            "lr",
            "max_grad_norm",
            "recovery_mode",
        ),
        ("schema_version", "run_dir", "artifact", "gpus", "training"),
        "compression profile",
    )
    if d["schema_version"] != "qwen-rl-ternary-compute/v1":
        raise ValueError("unsupported compute schema")
    t = d["training"]
    _keys(
        t,
        (
            "objective",
            "trainable",
            "teacher_head",
            "dataset_manifest",
            "mla_bundle",
            "steps",
            "records_per_update",
            "ce_weight",
            "kl_weight",
            "vocab_chunk_rows",
            "max_wall_seconds",
            "checkpoint_steps",
            "vocab_scale_lr",
        ),
        ("objective", "trainable", "teacher_head", "dataset_manifest", "steps"),
        "training",
    )
    if t["objective"] != "compression_recovery":
        raise ValueError("unknown training objective")
    if (
        not isinstance(t["trainable"], list)
        or not t["trainable"]
        or any(x not in ("lora", "mla_factors", "vocab_scales", "ternary_scales") for x in t["trainable"])
        or len(set(t["trainable"])) != len(t["trainable"])
    ):
        raise ValueError("trainable must list distinct supported families")
    if (
        not isinstance(d["gpus"], list)
        or len(d["gpus"]) != 1
        or type(d["gpus"][0]) is not int
        or d["gpus"][0] < 0
    ):
        raise ValueError("compression recovery requires one explicitly selected GPU")
    if d.get("microbatch_size", 1) != 1 or type(d.get("microbatch_size", 1)) is not int:
        raise ValueError("compression recovery currently requires microbatch_size=1")
    for key in ("gradient_checkpointing", "activation_offload"):
        if type(d.get(key, True)) is not bool:
            raise ValueError(f"{key} must be boolean")
    if d.get("recovery_mode", "auto") not in ("auto", "fresh"):
        raise ValueError("recovery_mode must be auto or fresh; it controls resume only")
    base = path.parent
    artifact = file_spec(d["artifact"], base, extra=("artifact_id",))
    if not isinstance(artifact["artifact_id"], str) or not re.fullmatch(
        "[0-9a-f]{32}", artifact["artifact_id"]
    ):
        raise ValueError("invalid artifact_id")
    teacher = file_spec(t["teacher_head"], base, extra=("key",))
    if not isinstance(teacher["key"], str) or not teacher["key"]:
        raise ValueError("teacher head key must be nonempty")
    dataset = (base / t["dataset_manifest"]).resolve()
    if not dataset.is_file():
        raise ValueError("missing dataset manifest")
    mla = (base / t["mla_bundle"]).resolve() if t.get("mla_bundle") else None
    if ("mla_factors" in t["trainable"]) != (mla is not None):
        raise ValueError("mla_factors requires an explicit MLA bundle, and vice versa")
    if mla is not None and not mla.is_file():
        raise ValueError("MLA bundle is missing")
    if "ternary_scales" in t["trainable"] and "vocab_scales" in t["trainable"]:
        raise ValueError("ternary_scales already includes vocab_scales")
    if "vocab_scale_lr" in t and not {"vocab_scales", "ternary_scales"}.intersection(t["trainable"]):
        raise ValueError("vocab_scale_lr requires vocabulary scale training")
    lr = _number(d, "lr", 1e-5, 1e-12, 1.0)
    vocab_scale_lr = _number(t, "vocab_scale_lr", lr, 1e-12, 1.0)
    ce, kl = (
        _number(t, "ce_weight", 1.0, 0.0, 100.0),
        _number(t, "kl_weight", 1.0, 0.0, 100.0),
    )
    if ce + kl <= 0:
        raise ValueError("at least one loss weight must be positive")
    run_dir = (base / d["run_dir"]).resolve()
    if any(
        parent == run_dir for parent in (Path(artifact["path"]).parent, dataset.parent)
    ):
        raise ValueError(
            "run_dir must be separate from input artifact/capture directories"
        )
    return CompressionConfig(
        path,
        run_dir,
        artifact,
        teacher,
        dataset,
        sha256(dataset),
        tuple(t["trainable"]),
        mla,
        sha256(mla) if mla else None,
        d["gpus"][0],
        _number(d, "max_sequence_length", 4096, 2, 65536, True),
        _number(t, "records_per_update", 1, 1, 1024, True),
        _number(t, "steps", 1, 1, 1000000, True),
        lr,
        ce,
        kl,
        _number(d, "max_grad_norm", 1.0, 1e-12, 1000.0),
        _number(t, "vocab_chunk_rows", 256, 1, 4096, True),
        d.get("gradient_checkpointing", True),
        d.get("activation_offload", True),
        d.get("recovery_mode", "auto"),
        _number(t, "max_wall_seconds", 3600, 1, 604800, True),
        vocab_scale_lr,
        _number(t, "checkpoint_steps", 1, 1, 1000000, True),
    )
