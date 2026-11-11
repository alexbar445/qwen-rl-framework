"""Library workflow for the canonical ``qwen-rl learner compute`` command."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .ternary_config import TernaryComputeConfig, load_ternary_compute_config


def _trajectory_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"trajectory JSONL line {line_number} is invalid") from exc
                if not isinstance(value, dict):
                    raise ValueError(  # noqa: TRY004 -- malformed JSONL is a data error
                        f"trajectory JSONL line {line_number} must be an object"
                    )
                rows.append(value)
    except OSError as exc:
        raise ValueError(f"cannot read trajectory JSONL: {path}") from exc
    if not rows:
        raise ValueError("trajectory JSONL contains no trajectory rows")
    return rows


def _self_smoke_rows(engine, *, response_length: int = 4) -> list[dict[str, Any]]:
    """Sample one bounded diagnostic GRPO group from the loaded policy itself."""
    import math

    import torch

    if response_length < 1:
        raise ValueError("response_length must be positive")
    prompt_length = 28
    if prompt_length + response_length > engine.max_sequence_length:
        raise ValueError("self-smoke prompt and response exceed max_sequence_length")
    tokenizer = engine.tokenizer
    pattern = tokenizer.encode(
        "Synthetic diagnostic prompt for validating the frozen T2 Learner update path.",
        add_special_tokens=False,
    )
    if not pattern:
        raise RuntimeError("embedded tokenizer returned no tokens for the self-smoke prompt")
    prompt_ids = (pattern * math.ceil(prompt_length / len(pattern)))[:prompt_length]
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, bool) or not isinstance(eos, int) or eos < 0:
        raise RuntimeError("embedded tokenizer must declare a valid eos_token_id for self-smoke")
    sampling_vocab_size = len(tokenizer)
    base_model = engine.model.get_base_model()
    model_vocab_size = int(base_model.config.vocab_size)
    if not 1 <= sampling_vocab_size <= model_vocab_size or eos >= model_vocab_size:
        raise RuntimeError("embedded tokenizer vocabulary is incompatible with the T2 model output head")

    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=engine.device)
    was_training = engine.model.training
    engine.model.eval()
    group_id = f"ternary-self-smoke-{uuid4().hex}"
    rows = []
    try:
        for index, reward in enumerate((0.0, 0.5, 1.0)):
            cuda_devices = [engine.device.index] if engine.device.type == "cuda" and engine.device.index is not None else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(1201 + index)
                if cuda_devices:
                    torch.cuda.manual_seed(1201 + index)
                generated = engine.model.generate(
                    input_ids=prompt,
                    attention_mask=torch.ones_like(prompt),
                    max_new_tokens=response_length,
                    min_new_tokens=response_length,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    use_cache=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=eos,
                    eos_token_id=eos,
                )
            sequences = generated.sequences
            transition_logprobs = engine.model.compute_transition_scores(
                sequences, generated.scores, normalize_logits=True
            )[0].float().cpu().tolist()
            response_ids = [int(value) for value in sequences[0, len(prompt_ids):].tolist()]
            response_count = min(response_length, len(response_ids), len(transition_logprobs))
            response_ids = response_ids[:response_count]
            response_logprobs = [float(value) for value in transition_logprobs[:response_count]]
            if response_count != response_length or not all(math.isfinite(value) for value in response_logprobs):
                raise RuntimeError("model generation did not return a full finite self-smoke response")
            rows.append({
                "episode_id": f"{group_id}-{index}",
                "group_id": group_id,
                "input_ids": [*prompt_ids, *response_ids],
                "policy_mask": [*([0] * len(prompt_ids)), *([1] * response_count)],
                "behavior_logprobs": [*([0.0] * len(prompt_ids)), *response_logprobs],
                "reward": reward,
                "policy_version": int(engine.policy_version),
                "row_kind": "fresh_student",
            })
    finally:
        engine.model.train(was_training)
    return rows


def _build_engine(profile: TernaryComputeConfig):
    config = profile.engine_config()
    if profile.parallelism == "single":
        # CUDA visibility must be fixed before the selected model loader imports Torch.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in profile.gpus)
        from .ternary_engine import TernaryLearnerEngine

        return TernaryLearnerEngine(config)
    if profile.parallelism == "pp":
        from .parallel_engine import ParallelLearnerEngine

        return ParallelLearnerEngine(config)
    raise ValueError(f"unsupported T2 Learner parallelism: {profile.parallelism!r}")


def run_compute(
    profile_path: str | Path,
    *,
    trajectories_path: str | Path | None = None,
    self_smoke: bool = False,
) -> dict[str, Any]:
    """Load one frozen T2 Learner, optionally apply one supplied batch, then exit.

    This command accepts completed trajectories or an explicitly local
    self-smoke group. It owns no Worker and never claims an end-to-end RL run.
    """
    profile = load_ternary_compute_config(profile_path)
    from .cuda_allocator import configure_learner_allocator
    configure_learner_allocator()
    if type(self_smoke) is not bool:
        raise TypeError("self_smoke must be a boolean")
    if self_smoke and trajectories_path is not None:
        raise ValueError("--self-smoke and --trajectories are mutually exclusive")
    if self_smoke and profile.parallelism != "single":
        raise ValueError("--self-smoke currently requires parallelism=single")
    if self_smoke and (profile.group_size != 3 or profile.groups_per_update != 1):
        raise ValueError("--self-smoke requires group_size=3 and groups_per_update=1")
    rows = None if trajectories_path is None else _trajectory_rows(Path(trajectories_path).expanduser())
    engine = _build_engine(profile)
    try:
        publication = engine.publish()
        recovery = engine.recovery_metadata
        result: dict[str, Any] = {
            "status": "ready",
            "operation": "compute_only",
            "worker_started": False,
            "external_rollout_started": False,
            "rollout_started": False,
            "end_to_end_rl": False,
            "self_sampled_diagnostic": False,
            "model_path": profile.model_path,
            "artifact_identity_path": str(profile.artifact_identity_path),
            "artifact_id": profile.identity.artifact_id,
            "artifact_sha256": profile.identity.artifact_sha256,
            "raw_gguf_status": profile.identity.document["source"]["raw_gguf"]["status"],
            "parallelism": profile.parallelism,
            "world_size": len(profile.gpus),
            "devices": list(profile.gpus),
            "tokenizer_files": engine.evidence.get("tokenizer_files", []),
            "quantization": dict(engine.evidence.get("quantization", {})),
            "policy_version": int(engine.policy_version),
            "publication": publication,
            "recovery": recovery,
        }
        if self_smoke:
            rows = _self_smoke_rows(engine)
            result["rollout_started"] = True
            result["self_sampled_diagnostic"] = True
            result["rollout_source"] = "learner_local_self_sample"
            result["sample_class"] = "synthetic prompt; self-sampled by the T2 policy; not ARC or Worker evidence"
            result["tokenizer_vocab_size"] = len(engine.tokenizer)
            result["model_vocab_size"] = int(engine._base_model.config.vocab_size)
            result["sampling_vocab_size"] = int(engine._base_model.config.vocab_size)
            result["sampling_vocab_constraint"] = "none; full model output vocabulary"
        if rows is not None:
            update = engine.update(rows)
            result["status"] = "updated"
            result["update"] = update
            result["policy_version"] = int(engine.policy_version)
            result["publication"] = update.get("publication")
            result["recovery"] = engine.recovery_metadata
        return result
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
