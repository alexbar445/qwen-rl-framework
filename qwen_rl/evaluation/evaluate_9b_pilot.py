#!/usr/bin/env python3
"""Paired end-to-end 9B ARC pilot on the frozen eight-task SFT holdout.

The snapshot is an input-only projection.  The model receives only a pinned system
prompt and ``visible_task``; reference outputs are loaded separately into the trusted
environment / direct-answer scorer.  Tool turns are generated in batches, then executed
by the existing ``ARCToolEnv``.  This is an evaluation entry point, not an RL run.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SNAPSHOT = Path("/work/outputs/9b-review/20261011/snapshot")
DEFAULT_HELDOUT_IDS = (
    "009d5c81", "1cf80156", "2013d3e2", "2072aba6",
    "21f83797", "25e02866", "506d28a5", "66e6c45b",
)
DIRECT_SYSTEM = """Solve the ARC task using only the visible training examples and test inputs.
Do not call tools or output code. Return only one JSON object mapping each target name
such as \"test/0\" to its predicted grid (a rectangular array of integer color values).
The test outputs are not shown. Do not include prose or markdown around the JSON."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _without_descriptions(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_descriptions(item) for key, item in value.items()
                if key != "description"}
    if isinstance(value, list):
        return [_without_descriptions(item) for item in value]
    return value


def _load_snapshot(snapshot: Path, selected_ids: list[str] | None) -> tuple[list[dict], dict]:
    """Validate and load only the input-only task projection."""
    source_path = snapshot / "source.json"
    eval_path = snapshot / "eval.jsonl"
    if not source_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError(f"snapshot needs source.json and eval.jsonl: {snapshot}")
    source = json.loads(source_path.read_text())
    if source.get("train_overlap") != []:
        raise ValueError("snapshot source.json reports train/eval task overlap")
    if "input-only" not in str(source.get("construction", "")):
        raise ValueError("snapshot is not attested as an input-only projection")
    heldout = source.get("heldout_ids")
    if not isinstance(heldout, list) or not heldout or any(not isinstance(x, str) for x in heldout):
        raise ValueError("source.json heldout_ids is missing or invalid")
    if heldout != sorted(set(heldout)):
        raise ValueError("source.json heldout_ids must be unique and sorted")
    if tuple(heldout) != DEFAULT_HELDOUT_IDS:
        raise ValueError("source.json heldout IDs differ from the reviewed 9B pilot set")

    eval_rows = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
    eval_ids = []
    for row in eval_rows:
        meta = row.get("meta")
        task_id = meta.get("task_id") if isinstance(meta, dict) else None
        if not isinstance(task_id, str):
            raise ValueError("eval.jsonl rows must contain meta.task_id")
        if set(row) != {"meta"}:
            raise ValueError("eval.jsonl must contain task IDs only; teacher rows are forbidden")
        eval_ids.append(task_id)
    if sorted(eval_ids) != heldout or len(set(eval_ids)) != len(eval_ids):
        raise ValueError("eval.jsonl task IDs do not match source.json heldout_ids")

    if selected_ids is None:
        wanted = heldout
    else:
        wanted = sorted(selected_ids)
        if len(set(wanted)) != len(wanted) or not set(wanted) <= set(heldout):
            raise ValueError("--task-ids must be a unique subset of source.json heldout_ids")

    record_by_id: dict[str, dict] = {}
    for path in sorted((snapshot / "records").glob("*.json")):
        record = json.loads(path.read_text())
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or task_id in record_by_id:
            raise ValueError(f"invalid or duplicate record task_id in {path.name}")
        if set(record) - {"task_id", "visible_task", "system_prompt", "tool_version", "native_tools"}:
            raise ValueError(f"record {path.name} contains non-projection fields")
        visible = record.get("visible_task")
        if not isinstance(visible, dict) or set(visible) != {"train", "test"}:
            raise ValueError(f"record {path.name} visible_task must contain train and test only")
        if not isinstance(visible["train"], list) or not isinstance(visible["test"], list) or not visible["test"]:
            raise ValueError(f"record {path.name} has invalid visible train/test rows")
        if any(not isinstance(pair, dict) or "input" not in pair or "output" not in pair
               for pair in visible["train"]):
            raise ValueError(f"record {path.name} has malformed visible training examples")
        if any(not isinstance(item, dict) or "input" not in item or "output" in item
               for item in visible["test"]):
            raise ValueError(f"record {path.name} exposes a held-out test output")
        system_prompt = record.get("system_prompt")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError(f"record {path.name} needs its frozen system_prompt")
        if not isinstance(record.get("tool_version"), str) or not record["tool_version"]:
            raise ValueError(f"record {path.name} needs its public tool_version")
        if not isinstance(record.get("native_tools"), list):
            raise ValueError(f"record {path.name} needs its public native_tools schemas")
        record_by_id[task_id] = record

    missing = set(wanted) - set(record_by_id)
    if missing:
        raise ValueError(f"snapshot is missing records for {sorted(missing)}")
    records = [record_by_id[task_id] for task_id in wanted]
    details = {
        "snapshot": str(snapshot.resolve()),
        "source_json_sha256": _sha256_file(source_path),
        "projected_eval_jsonl_sha256": _sha256_file(eval_path),
        "source_eval_jsonl_sha256": source.get("eval_jsonl_sha256"),
        "source_snapshot": source.get("source_snapshot"),
        "source_snapshot_metadata_sha256": source.get("snapshot_metadata_sha256"),
        "construction": source.get("construction"),
        "heldout_ids": heldout,
        "selected_ids": wanted,
        "train_tasks": source.get("train_tasks"),
        "train_overlap": source.get("train_overlap"),
        "source_record_sha256": source.get("source_records_sha256"),
    }
    return records, details


def _valid_grid(value: Any) -> bool:
    return (isinstance(value, list) and 1 <= len(value) <= 30
            and all(isinstance(row, list) and 1 <= len(row) <= 30 for row in value)
            and len({len(row) for row in value}) == 1
            and all(type(cell) is int and 0 <= cell <= 9 for row in value for cell in row))


def _parse_direct_answer(text: str, target_names: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the final JSON after an optional Qwen ``<think>`` block."""
    candidate = text.rsplit("</think>", 1)[-1].strip()
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"):
        candidate = candidate.replace(token, "").strip()
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    if not isinstance(value, dict) or set(value) != set(target_names):
        return None, "answer_keys_mismatch"
    if not all(_valid_grid(value[key]) for key in target_names):
        return None, "invalid_grid"
    return value, None


def _grids_equal(predicted: Any, expected: Any) -> bool:
    if not _valid_grid(predicted) or not _valid_grid(expected):
        return False
    return (len(predicted) == len(expected)
            and all(len(row_a) == len(row_b) and all(int(a) == int(b)
                    for a, b in zip(row_a, row_b))
                    for row_a, row_b in zip(predicted, expected)))


def _event(writer, event: dict) -> None:
    event = {"ts_utc": datetime.now(timezone.utc).isoformat(), **event}
    writer.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    writer.flush()


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@dataclass
class Episode:
    task_id: str
    record: dict
    challenge: dict
    targets: dict[str, list[list[int]]]
    messages: list[dict]
    env: Any = None
    generated_tokens: int = 0
    context_tokens_peak: int = 0
    turns: int = 0
    errors: int = 0
    tool_call_failures: int = 0
    status: str = "running"
    terminal: bool = False
    final_exact: bool = False
    per_target: dict[str, bool] = field(default_factory=dict)
    started_at: float = 0.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True, choices=("base-tools", "tuned-tools", "base-direct"))
    parser.add_argument("--output", required=True, type=Path, help="new JSONL event file; existing files are rejected")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="directory containing ARC-AGI-2 challenge/solution JSON")
    parser.add_argument("--tokenizer", default=None, help="tokenizer path; defaults to --model")
    parser.add_argument("--toolkit", type=Path, default=None, help="optional toolkit tree for ARCToolEnv")
    parser.add_argument("--task-ids", nargs="+", default=None, help="subset of the frozen eight IDs; default runs all")
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="raw per-task budget: generated tokens plus tool feedback")
    parser.add_argument("--turn-tokens", type=int, default=4096, help="per-turn cap for tool arms")
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=65536)
    return parser


def _create_llm(args):
    from vllm import LLM

    return LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        kv_cache_dtype="auto",  # resolves to the BF16 model dtype
        enforce_eager=True,
        speculative_config=None,  # no MTP/speculative decoding in this comparison
        language_model_only=True,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        mamba_cache_mode="align",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
    )


def _parse_batch(parser, token_id_rows, schemas):
    async def parse_all():
        return await asyncio.gather(*(parser.extract_tool_calls(token_ids, schemas)
                                      for token_ids in token_id_rows))
    return asyncio.run(parse_all())


def _make_sampling(SamplingParams, max_tokens: int, tools_mode: bool):
    kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
    }
    if tools_mode:
        kwargs.update({"stop": ["</tool_call>"], "include_stop_str_in_output": True})
    return SamplingParams(**kwargs)


def _finalize(ep: Episode, status: str, writer=None, *, exact: bool = False,
              per_target: dict[str, bool] | None = None, reason: str | None = None) -> None:
    ep.status = status
    ep.terminal = True
    ep.final_exact = bool(exact)
    ep.per_target = dict(per_target or {})
    env = ep.env
    if env is not None:
        env_summary = env.close()
        observation_tokens = env_summary["obs_tokens"]
        tool_calls = env_summary["lines"]
        env_root = env_summary["root"]
    else:
        env_summary = None
        observation_tokens = 0
        tool_calls = 0
        env_root = None
    event = {
        "event": "task_final", "task_id": ep.task_id, "status": status,
        "exact": ep.final_exact, "per_target": ep.per_target,
        "reason": reason, "turns": ep.turns, "tool_calls": tool_calls,
        "generated_tokens": ep.generated_tokens,
        "observation_tokens": observation_tokens,
        "tokens_generated_plus_feedback": ep.generated_tokens + observation_tokens,
        "context_tokens_peak": ep.context_tokens_peak,
        "tool_error_count": ep.errors,
        "tool_call_failures": ep.tool_call_failures,
        "environment": env_summary, "environment_root": env_root,
    }
    if writer is not None:
        _event(writer, event)


def _reject_tool_generation(ep: Episode, *, reason: str, role: str,
                            assistant_message: dict, writer, turn_id: int,
                            max_failures: int, counts_as_failure: bool,
                            parse_status: str, generation_limit_reached: bool,
                            requested_tool_name: str | None = None) -> str | None:
    """Append one failed generation and corrective observation; return a terminal status."""
    if counts_as_failure:
        ep.tool_call_failures += 1
        if parse_status != "no_call":
            ep.errors += 1

    from qwen_rl.contracts.tool_call_failures import tool_call_failure_feedback

    feedback = tool_call_failure_feedback(
        reason, parse_status=parse_status,
        generation_limit_reached=generation_limit_reached,
        requested_tool_name=requested_tool_name,
        tool_call_failures=ep.tool_call_failures,
        done=(reason == "generation_limit" or not counts_as_failure or ep.tool_call_failures >= max_failures),
    )
    feedback.update(tool_call_failures=ep.tool_call_failures,
                    max_tool_call_failures=max_failures)
    observation = ep.env.render_budget_observation(feedback, role=role)
    ep.messages.append(assistant_message)
    ep.messages.append({"role": role, "content": observation})
    _event(writer, {
        "event": "tool_rejected", "task_id": ep.task_id, "turn_id": turn_id,
        "reason": reason, "tool_call_failures": ep.tool_call_failures,
        "max_tool_call_failures": max_failures,
        "observation_role": role,
    })

    if ep.env.tokens_used() >= ep.env.token_budget:
        return "budget_exhausted"
    if reason == "generation_limit":
        return "turn_budget_exhausted"
    if not counts_as_failure:
        return "tool_error"
    if ep.tool_call_failures >= max_failures:
        return "tool_error"
    return None


def _execute(args) -> int:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from qwen_rl.budget import CONTRACT
    from qwen_rl.contracts.tool_call_failures import (
        ENV_SCHEMA_VERSION, MAX_TOOL_CALL_FAILURES, TOOL_CALL_FAILURE_COUNTING,
    )
    from qwen_rl.environment import ARCToolEnv, DEFAULT_TOOLKIT
    from qwen_rl.prompt import action_tools, build_messages
    from qwen_rl.task_source import load_tasks
    from qwen_rl.verl_agent import _assistant_message, _policy_turn
    from qwen_rl.contracts.arc_tools import is_model_stop, turn_rejection_reason
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from vllm import SamplingParams

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_name(output.name + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite pilot output: {output}")

    records, snapshot_info = _load_snapshot(args.snapshot, args.task_ids)
    specs = load_tasks(pools=("arc2_train",), variants=False, dataset_dir=args.dataset_dir)
    spec_by_id = {task.family: task for task in specs}
    episode_specs = []
    for record in records:
        task_id = record["task_id"]
        spec = spec_by_id.get(task_id)
        if spec is None:
            raise ValueError(f"dataset-dir has no ARC-AGI-2 task {task_id}")
        if _canonical(spec.challenge) != _canonical(record["visible_task"]):
            raise ValueError(f"input-only snapshot challenge differs from dataset-dir for {task_id}")
        if not spec.targets:
            raise ValueError(f"dataset-dir has no reference target for {task_id}")
        episode_specs.append((record, spec))

    tools_mode = args.arm != "base-direct"
    enable_thinking = True
    tool_dicts = action_tools() if tools_mode else []
    if tools_mode:
        expected_schema = _canonical(_without_descriptions(tool_dicts))
        for record, _ in episode_specs:
            if _canonical(_without_descriptions(record["native_tools"])) != expected_schema:
                raise ValueError(f"frozen tool schema differs from qwen_RL tools for {record['task_id']}")
    tool_schemas = [OpenAIFunctionToolSchema.model_validate(item) for item in tool_dicts]
    tool_parser = None
    toolkit = Path(args.toolkit or DEFAULT_TOOLKIT).resolve()
    toolkit_sha = _sha256_file(toolkit / "arc_toolkit" / "runtime.py")
    # Same raw budget for each arm: generated tokens plus tool observations. Direct has
    # no observations, so its raw budget is all generation; tool feedback consumes it.
    pilot_contract = replace(CONTRACT, token_budget=args.max_tokens)

    system_hashes = sorted({_sha256_bytes(record["system_prompt"].encode())
                            for record, _ in episode_specs}) if tools_mode else []
    manifest = {
        "schema": "qwen-9b-arc-pilot-v1", "status": "initializing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm, "model": args.model, "tokenizer": args.tokenizer or args.model,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "enable_thinking": enable_thinking,
                     "tool_mode": tools_mode, "tensor_parallel": args.tensor_parallel,
                     "dtype": "bfloat16", "kv_cache_dtype": "auto (BF16 model dtype)",
                     "enforce_eager": True, "speculative_config": None, "MTP": False,
                     "language_model_only": True, "max_num_seqs": 4,
                     "max_num_batched_tokens": 2048, "mamba_cache_mode": "align",
                     "enable_prefix_caching": True,
                     "gpu_memory_utilization": 0.9, "max_model_len": args.max_model_len,
                     "token_budget_scope": "generated_plus_observation",
                     "max_raw_episode_tokens": args.max_tokens,
                     "max_generated_tokens_per_tool_turn": args.turn_tokens if tools_mode else None,
                     "max_turns": args.max_turns,
                     "arc_env_schema": ENV_SCHEMA_VERSION if tools_mode else None,
                     "max_tool_call_failures": MAX_TOOL_CALL_FAILURES if tools_mode else None,
                     "tool_call_failure_counting": TOOL_CALL_FAILURE_COUNTING if tools_mode else None,
                     "tool_stop": ["</tool_call>"] if tools_mode else None,
                     "include_stop_str_in_output": tools_mode,
                     "skip_special_tokens": False},
        "snapshot": snapshot_info,
        "record_system_prompt_sha256": system_hashes,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_tasks": [{"task_id": record["task_id"], "family": spec.family,
                           "targets": sorted(spec.targets), "tool_version": record["tool_version"]}
                          for record, spec in episode_specs],
        "toolkit": str(toolkit), "toolkit_runtime_sha256": toolkit_sha,
        "tool_schemas_sha256": _sha256_bytes(_canonical(tool_dicts).encode()) if tools_mode else None,
        "contract": pilot_contract.as_dict(), "contract_fingerprint": pilot_contract.fingerprint(),
        "output_jsonl": str(output),
        "interpretation": "deployment pilot; tool-mode comparison uses the shared ARC system prompt and current RL vendored runtime; not an SFT-only causal estimate",
    }
    _write_manifest(manifest_path, manifest)

    try:
        llm = _create_llm(args)
        tokenizer = llm.get_tokenizer()
        if tools_mode:
            tool_parser = ToolParser.get_tool_parser("qwen3_coder", tokenizer)
        manifest["tokenizer_class"] = type(tokenizer).__name__
        manifest["tokenizer_vocab_size"] = len(tokenizer)
        manifest["vllm_version"] = __import__("vllm").__version__
        manifest["status"] = "running"
        _write_manifest(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "infrastructure_error"
        manifest["initialization_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_manifest(manifest_path, manifest)
        with output.open("x", encoding="utf-8") as writer:
            _event(writer, {"event": "run_error", "classification": "infrastructure_error",
                            "stage": "model_initialization", "type": type(exc).__name__,
                            "message": str(exc)})
        raise

    episode_root = output.parent / (output.name + ".episodes")
    states: dict[str, Episode] = {}
    try:
        with output.open("x", encoding="utf-8") as writer:
            for record, spec in episode_specs:
                task_id = record["task_id"]
                base_messages = build_messages(record["visible_task"])
                if tools_mode:
                    messages = [{"role": "system", "content": record["system_prompt"]}, base_messages[-1]]
                    env = ARCToolEnv(
                        record["visible_task"], spec.targets, spec.public_id,
                        episode_root / task_id, pilot_contract, pilot_contract.token_budget,
                        tokenizer, toolkit=toolkit,
                    )
                    try:
                        reset_message = env.reset()
                        if reset_message != messages[-1]["content"]:
                            raise AssertionError("ARCToolEnv reset prompt differs from built user prompt")
                    except Exception:
                        env.close()
                        raise
                else:
                    direct_messages = [
                        {"role": "system", "content": DIRECT_SYSTEM},
                        base_messages[-1],
                    ]
                    env = None
                    messages = direct_messages
                states[task_id] = Episode(
                    task_id=task_id, record=record, challenge=record["visible_task"],
                    targets=spec.targets, messages=messages, env=env,
                    started_at=datetime.now(timezone.utc).timestamp(),
                )
                _event(writer, {"event": "task_start", "task_id": task_id,
                                "targets": sorted(spec.targets), "arm": args.arm,
                                "prompt_mode": "SFT-tool-system" if tools_mode else "direct-json"})

            if not tools_mode:
                active = [states[record["task_id"]] for record, _ in episode_specs]
                prompts = []
                params = []
                for ep in active:
                    prompt = tokenizer.apply_chat_template(
                        ep.messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                    context_tokens = len(tokenizer.apply_chat_template(
                        ep.messages, tokenize=True, add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    ))
                    ep.context_tokens_peak = context_tokens
                    if context_tokens >= args.max_model_len:
                        _finalize(ep, "context_limit", writer, reason="prompt exceeds max_model_len")
                        continue
                    generation_limit = min(args.max_tokens, args.max_model_len - context_tokens)
                    if generation_limit <= 0:
                        _finalize(ep, "context_limit", writer, reason="no generation room under max_model_len")
                        continue
                    prompts.append(prompt)
                    params.append(_make_sampling(SamplingParams, generation_limit, False))
                if prompts:
                    try:
                        outputs = llm.generate(prompts, params, use_tqdm=False)
                    except Exception as exc:
                        for ep in active:
                            if not ep.terminal:
                                _finalize(ep, "infrastructure_error", writer,
                                          reason=f"generate:{type(exc).__name__}:{exc}")
                        outputs = []
                    prompt_index = 0
                    for ep in active:
                        if ep.terminal:
                            continue
                        result = outputs[prompt_index]
                        prompt_index += 1
                        completion = result.outputs[0]
                        token_ids = list(completion.token_ids)
                        ep.turns = 1
                        ep.generated_tokens = len(token_ids)
                        text = completion.text
                        _event(writer, {"event": "model_turn", "task_id": ep.task_id,
                                        "turn_id": 0, "generated_text": text,
                                        "generated_token_ids": token_ids,
                                        "generated_tokens": len(token_ids),
                                        "prompt_tokens": len(result.prompt_token_ids),
                                        "finish_reason": completion.finish_reason,
                                        "stop_reason": getattr(completion, "stop_reason", None)})
                        predicted, parse_error = _parse_direct_answer(text, sorted(ep.targets))
                        if parse_error:
                            status = "budget_exhausted" if completion.finish_reason == "length" else "wrong_answer"
                            _finalize(ep, status, writer, reason=parse_error)
                            continue
                        per_target = {name: _grids_equal(predicted[name], target)
                                      for name, target in ep.targets.items()}
                        exact = bool(per_target) and all(per_target.values())
                        _event(writer, {"event": "direct_answer", "task_id": ep.task_id,
                                        "predicted": predicted, "per_target": per_target,
                                        "exact": exact})
                        _finalize(ep, "solved" if exact else "wrong_answer", writer,
                                  exact=exact, per_target=per_target)
            else:
                for turn_id in range(args.max_turns):
                    active = [ep for ep in states.values() if not ep.terminal]
                    if not active:
                        break
                    prompts = []
                    params = []
                    ready = []
                    generation_limits = []
                    for ep in active:
                        remaining = args.max_tokens - ep.env.tokens_used()
                        if remaining <= 0:
                            _finalize(ep, "budget_exhausted", writer,
                                      reason="raw generated-plus-feedback budget reached")
                            continue
                        prompt = tokenizer.apply_chat_template(
                            ep.messages, tools=tool_dicts, tokenize=False,
                            add_generation_prompt=True, enable_thinking=enable_thinking,
                        )
                        context_tokens = len(tokenizer.apply_chat_template(
                            ep.messages, tools=tool_dicts, tokenize=True,
                            add_generation_prompt=True, enable_thinking=enable_thinking,
                        ))
                        ep.context_tokens_peak = max(ep.context_tokens_peak, context_tokens)
                        if context_tokens >= args.max_model_len:
                            _finalize(ep, "context_limit", writer, reason="prompt exceeds max_model_len")
                            continue
                        generation_limit = min(args.turn_tokens, remaining,
                                               args.max_model_len - context_tokens)
                        if generation_limit <= 0:
                            _finalize(ep, "context_limit", writer,
                                      reason="no generation room under max_model_len")
                            continue
                        prompts.append(prompt)
                        params.append(_make_sampling(SamplingParams, generation_limit, True))
                        ready.append(ep)
                        generation_limits.append(generation_limit)
                    if not prompts:
                        continue
                    try:
                        outputs = llm.generate(prompts, params, use_tqdm=False)
                    except Exception as exc:
                        for ep in ready:
                            _finalize(ep, "infrastructure_error", writer,
                                      reason=f"generate:{type(exc).__name__}:{exc}")
                        continue
                    parsed_batch = _parse_batch(
                        tool_parser, [list(output.outputs[0].token_ids) for output in outputs],
                        tool_schemas,
                    )
                    for ep, result, parsed, generation_limit in zip(
                            ready, outputs, parsed_batch, generation_limits):
                        completion = result.outputs[0]
                        token_ids = list(completion.token_ids)
                        ep.turns = turn_id + 1
                        ep.generated_tokens += len(token_ids)
                        ep.env.record_generation(len(token_ids))
                        content, calls = parsed
                        _event(writer, {"event": "model_turn", "task_id": ep.task_id,
                                        "turn_id": turn_id, "generated_text": completion.text,
                                        "generated_token_ids": token_ids,
                                        "generated_tokens": len(token_ids),
                                        "prompt_tokens": len(result.prompt_token_ids),
                                        "finish_reason": completion.finish_reason,
                                        "stop_reason": getattr(completion, "stop_reason", None),
                                        "parser_content": content,
                                        "parsed_call_count": len(calls),
                                        "generation_limit_reached": (
                                            completion.finish_reason == "length"
                                            or len(token_ids) >= generation_limit)})
                        if ep.env.tokens_used() > ep.env.token_budget:
                            _finalize(ep, "budget_exhausted", writer,
                                      reason="raw generated-plus-feedback budget exceeded")
                            continue
                        limit_reached = (completion.finish_reason == "length"
                                         or len(token_ids) >= generation_limit)
                        selected_name, selected_arguments, parse_status, requested_tool_name = (
                            _policy_turn(calls, completion.text)
                        )
                        valid_call = parse_status == "valid"
                        assistant_message = (
                            _assistant_message(content, selected_name, selected_arguments)
                            if valid_call else {"role": "assistant", "content": completion.text}
                        )

                        rejection_reason = turn_rejection_reason(
                            parse_status, completion.finish_reason, limit_reached,
                        )
                        if is_model_stop(parse_status, completion.finish_reason, limit_reached):
                            ep.messages.append(assistant_message)
                            exact, per_target = ep.env.score_targets()
                            _event(writer, {"event": "model_stop", "task_id": ep.task_id,
                                            "turn_id": turn_id, "exact": exact,
                                            "per_target": per_target})
                            _finalize(ep, "solved" if exact else "wrong_answer", writer,
                                      exact=exact, per_target=per_target, reason="model_stop")
                            continue

                        if rejection_reason is not None:
                            failure = (parse_status != "valid"
                                       and rejection_reason != "generation_limit")
                            terminal_status = _reject_tool_generation(
                                ep, reason=rejection_reason,
                                role="tool" if valid_call else "user",
                                assistant_message=assistant_message, writer=writer,
                                turn_id=turn_id, max_failures=MAX_TOOL_CALL_FAILURES,
                                counts_as_failure=failure, parse_status=parse_status,
                                generation_limit_reached=limit_reached,
                                requested_tool_name=requested_tool_name,
                            )
                            if terminal_status is not None:
                                _finalize(ep, terminal_status, writer, reason=rejection_reason)
                            continue

                        name, arguments = selected_name, selected_arguments
                        if name == "arc_action":
                            action = arguments
                            step = ep.env.step(action)
                            response = ep.env.state.responses[-1]
                            infra_abort = bool(step.info.get("aborted"))
                            tool_failed = step.info.get("status") != "ok" and not infra_abort
                            if tool_failed:
                                ep.errors += 1
                                ep.tool_call_failures += 1
                            _event(writer, {"event": "tool_result", "task_id": ep.task_id,
                                            "turn_id": turn_id, "tool_name": name,
                                            "tool_action": action,
                                            "tool_status": step.info.get("status"),
                                            "tool_error": step.info.get("error"),
                                            "tool_response": response, "observation": step.observation,
                                            "tool_call_failures": ep.tool_call_failures,
                                            "max_tool_call_failures": MAX_TOOL_CALL_FAILURES,
                                            "tokens_generated": ep.generated_tokens,
                                            "observation_tokens": ep.env.state.obs_tokens,
                                            "exact": step.info.get("exact")})
                            if infra_abort:
                                _finalize(ep, "infrastructure_error", writer,
                                          reason=step.info.get("reason") or "tool environment aborted")
                                continue
                            if tool_failed:
                                failure_limit_reached = ep.tool_call_failures >= MAX_TOOL_CALL_FAILURES
                                budget_exhausted = ep.env.tokens_used() >= ep.env.token_budget
                                if step.done or failure_limit_reached or budget_exhausted:
                                    status = "budget_exhausted" if budget_exhausted else "tool_error"
                                    _finalize(ep, status, writer,
                                              reason=step.info.get("error") or "tool_execution_error")
                                    continue
                                ep.messages.append(_assistant_message(content, name, arguments))
                                ep.messages.append({"role": "tool", "content": step.observation})
                                continue
                            if step.done:
                                exact = bool(step.info.get("exact"))
                                status = "solved" if exact else (
                                    "budget_exhausted" if step.info.get("reason") == "token budget exhausted"
                                    else "episode_terminal")
                                _finalize(ep, status, writer, exact=exact,
                                          per_target=step.info.get("per_target"),
                                          reason=step.info.get("reason"))
                                continue
                            ep.messages.append(_assistant_message(content, name, arguments))
                            ep.messages.append({"role": "tool", "content": step.observation})
                        else:
                            _finalize(ep, "tool_error", writer, reason=f"unknown_tool:{name}")
                for ep in states.values():
                    if not ep.terminal:
                        if tools_mode and ep.env.tokens_used() >= pilot_contract.token_budget:
                            _finalize(ep, "budget_exhausted", writer,
                                      reason="raw generated-plus-feedback budget reached")
                        else:
                            _finalize(ep, "turn_limit", writer, reason="max_turns reached")

    except Exception as exc:
        # Exceptions outside a single model request or a task-local parse/action result
        # are environment/runner failures; they are never counted as wrong ARC answers.
        with output.open("a", encoding="utf-8") as recovery_writer:
            for ep in states.values():
                if not ep.terminal:
                    _finalize(ep, "infrastructure_error", recovery_writer,
                              reason=f"runner:{type(exc).__name__}:{exc}")
            _event(recovery_writer, {"event": "run_error", "classification": "infrastructure_error",
                                     "stage": "runner", "type": type(exc).__name__,
                                     "message": str(exc)})
        manifest["status"] = "infrastructure_error"
        manifest["runner_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=10)

    finals = []
    for ep in states.values():
        finals.append({"task_id": ep.task_id, "status": ep.status,
                       "exact": ep.final_exact, "generated_tokens": ep.generated_tokens,
                       "turns": ep.turns, "errors": ep.errors,
                       "tool_call_failures": ep.tool_call_failures})
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["task_results"] = finals
    manifest["summary"] = {
        "tasks": len(finals), "exact": sum(row["exact"] for row in finals),
        "wrong_or_incomplete": sum(row["status"] in {
            "wrong_answer", "episode_terminal", "model_stop",
            "turn_limit", "turn_budget_exhausted", "budget_exhausted"
        } for row in finals),
        "tool_errors": sum(row["status"] == "tool_error" for row in finals),
        "tool_call_failures": sum(row["tool_call_failures"] for row in finals),
        "infrastructure_errors": sum(row["status"] == "infrastructure_error" for row in finals),
    }
    _write_manifest(manifest_path, manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for name in ("tensor_parallel", "max_tokens", "turn_tokens", "max_turns", "max_model_len"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return _execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
