#!/usr/bin/env python3
"""Read-only independent receipt checks for the fixed 9B eight-task pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
TASK_IDS = (
    "009d5c81", "1cf80156", "2013d3e2", "2072aba6",
    "21f83797", "25e02866", "506d28a5", "66e6c45b",
)
RAW_LIMIT = 16384
STOP = ("<|im_end|>", "<|endoftext|>", "<|fim_suffix|>")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("not an object")
            rows.append(value)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"line_{n}:{type(exc).__name__}")
    return rows, errors


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def strict_grid(value: Any) -> tuple[bool, str | None, tuple[int, int] | None]:
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        return False, "height_out_of_range", None
    if not all(isinstance(row, list) and 1 <= len(row) <= 30 for row in value):
        return False, "row_width_out_of_range", None
    widths = {len(row) for row in value}
    if len(widths) != 1:
        return False, "ragged_rows", None
    if any(type(cell) is not int for row in value for cell in row):
        return False, "cell_not_strict_int", None
    if any(cell < 0 or cell > 9 for row in value for cell in row):
        return False, "cell_outside_0_to_9", None
    return True, None, (len(value), next(iter(widths)))


def compare_grid(prediction: Any, target: Any) -> dict:
    pred_ok, pred_error, pred_shape = strict_grid(prediction)
    target_ok, target_error, target_shape = strict_grid(target)
    return {
        "exact": bool(pred_ok and target_ok and prediction == target),
        "prediction_valid": pred_ok,
        "target_valid": target_ok,
        "prediction_shape": pred_shape,
        "target_shape": target_shape,
        "prediction_error": pred_error,
        "target_error": target_error,
    }


def parse_direct(text: str, targets: list[str]) -> tuple[dict | None, str | None]:
    candidate = text.rsplit("</think>", 1)[-1].strip()
    for token in STOP:
        candidate = candidate.replace(token, "").strip()
    if not candidate.startswith("{"):
        return None, "raw_text_not_json_object"
    try:
        value, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as exc:
        return None, f"raw_json_error:{exc.msg}@{exc.pos}"
    trailing = candidate[end:].strip()
    for token in STOP:
        trailing = trailing.replace(token, "").strip()
    if trailing:
        return None, "trailing_non_json_text"
    if not isinstance(value, dict) or set(value) != set(targets):
        return None, "raw_json_target_keys_mismatch"
    return value, None


def load_tasks(dataset_dir: Path) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from qwen_rl.task_source import load_tasks as load_task_specs

    specs = load_task_specs(
        pools=("arc2_train",), variants=False, dataset_dir=dataset_dir
    )
    return {spec.family: spec for spec in specs}


def resolve_model_config(manifest: dict) -> Path | None:
    model = manifest.get("model")
    if not isinstance(model, str):
        return None
    paths = [Path(model) / "config.json"]
    sft = ROOT.parent / "qwen_sft"
    if "9b-qlora-both" in model:
        paths.append(sft / "outputs/9b-qlora-both/merged/config.json")
    elif "Qwen3.5-9B" in model:
        paths.append(sft / "models/Qwen3.5-9B/config.json")
    return next((path for path in paths if path.is_file()), None)


def validate_tool_submission(
    episode_root: Path, toolkit_root: Path, spec,
) -> tuple[dict[str, Any], dict]:
    details: dict[str, Any] = {"submission_refs": {}, "session_inputs_match": None}
    if not (episode_root / "session.json").is_file():
        return {}, {"error": "session_missing"}
    if not (episode_root / ".runtime.lock").is_file():
        return {}, {"error": "existing_session_lock_missing"}
    if str(toolkit_root) not in sys.path:
        sys.path.insert(0, str(toolkit_root))
    try:
        from arc_toolkit.runtime import Session

        session = Session.open(episode_root)
        state = read_json(episode_root / "session.json")
    except Exception as exc:
        return {}, {"error": f"session_open:{type(exc).__name__}"}

    inputs_match = True
    for split in ("train", "test"):
        for index, row in enumerate(spec.challenge.get(split, [])):
            fields = ("input", "output") if split == "train" else ("input",)
            for field in fields:
                try:
                    artifact = session.read(f"{split}/{index}/{field}")
                    inputs_match &= artifact.kind == "Grid" and artifact.data == row[field]
                except Exception:
                    inputs_match = False
    details["session_inputs_match"] = bool(inputs_match)
    refs_by_target = state.get("submissions", {})
    predictions = {}
    for target in sorted(spec.targets):
        refs = refs_by_target.get(target, []) if isinstance(refs_by_target, dict) else []
        slots = {}
        details["submission_refs"][target] = {}
        for ref in refs:
            try:
                submitted = session.read(ref)
                if (submitted.kind != "Submission" or not isinstance(submitted.data, dict)
                        or submitted.data.get("target") != target
                        or submitted.data.get("attempt") not in (1, 2)):
                    raise ValueError("invalid_submission")
                attempt = submitted.data["attempt"]
                slots[attempt] = submitted.data["prediction"]
                details["submission_refs"][target][str(attempt)] = ref
            except Exception as exc:
                details.setdefault("errors", []).append(
                    f"{target}:submission_read:{type(exc).__name__}"
                )
        predictions[target] = slots
    return predictions, details


def validate_run(jsonl_path: Path, dataset_dir: Path, toolkit_root: Path,
                 selected_ids: set[str] | None) -> dict:
    manifest_path = Path(str(jsonl_path) + ".manifest.json")
    manifest = read_json(manifest_path)
    events, jsonl_errors = read_jsonl(jsonl_path)
    all_specs = load_tasks(dataset_dir)
    records = manifest.get("dataset_tasks", [])
    manifest_ids = {
        row.get("task_id") for row in records if isinstance(row, dict)
    } if isinstance(records, list) else set()
    starts = [e.get("task_id") for e in events if e.get("event") == "task_start"]
    finals = [e.get("task_id") for e in events if e.get("event") == "task_final"]
    manifest_list_ids = [row.get("task_id") for row in records if isinstance(row, dict)] \
        if isinstance(records, list) else []
    start_ids, final_ids = set(starts), set(finals)
    start_dupes = sorted(task for task, n in Counter(starts).items() if n > 1)
    dupes = sorted(task for task, n in Counter(finals).items() if n > 1)
    chosen = set(TASK_IDS) if selected_ids is None else selected_ids
    task_results = []
    by_task = {}
    arm = manifest.get("arm")
    episodes_root = Path(str(jsonl_path) + ".episodes")

    for task_id in sorted(chosen & set(TASK_IDS)):
        final_rows = [e for e in events if e.get("event") == "task_final"
                      and e.get("task_id") == task_id]
        starts_for_task = [e for e in events if e.get("event") == "task_start"
                           and e.get("task_id") == task_id]
        task_events = [e for e in events if e.get("task_id") == task_id]
        result: dict[str, Any] = {
            "task_id": task_id,
            "start_seen": bool(starts_for_task),
            "final_seen": len(final_rows) == 1,
            "reported_status": final_rows[-1].get("status") if final_rows else None,
            "reported_exact": final_rows[-1].get("exact") if final_rows else None,
        }
        spec = all_specs.get(task_id)
        if spec is None:
            result["error"] = "dataset_truth_missing"
            task_results.append(result)
            by_task[task_id] = result
            continue
        targets = sorted(spec.targets)
        result["targets"] = targets
        snapshot_records = list((ROOT / "outputs/9b-review/20261011/snapshot/records").glob(
            f"{task_id}*.json"
        ))
        if len(snapshot_records) == 1:
            record = read_json(snapshot_records[0])
            visible = record.get("visible_task")
            result["snapshot_input_match"] = (
                isinstance(visible, dict) and canonical(visible) == canonical(spec.challenge)
            )
        else:
            result["snapshot_input_match"] = False
        if not final_rows:
            result.update({"independent_exact": None, "within_budget_success": None,
                           "status": "partial"})
            task_results.append(result)
            by_task[task_id] = result
            continue
        final = final_rows[-1]
        turns = sorted((e for e in task_events if e.get("event") == "model_turn"),
                       key=lambda e: e.get("turn_id", -1))
        tool_results = sorted((e for e in task_events if e.get("event") == "tool_result"),
                              key=lambda e: e.get("turn_id", -1))
        errors = []
        generated = 0
        generated_by_turn = {}
        for turn in turns:
            ids = turn.get("generated_token_ids")
            count = turn.get("generated_tokens")
            if not isinstance(ids, list) or any(type(token) is not int for token in ids):
                errors.append(f"turn_{turn.get('turn_id')}:token_ids_invalid")
                n_ids = None
            else:
                n_ids = len(ids)
                generated += n_ids
            if n_ids is None or count != n_ids:
                errors.append(f"turn_{turn.get('turn_id')}:generated_count_mismatch")
            generated_by_turn[turn.get("turn_id")] = generated
        if generated != final.get("generated_tokens"):
            errors.append("final_generated_sum_mismatch")
        obs_values = [e.get("observation_tokens") for e in tool_results]
        if any(type(n) is not int or n < 0 for n in obs_values):
            errors.append("tool_observation_count_invalid")
        if any(a > b for a, b in zip(obs_values, obs_values[1:]) if
               isinstance(a, int) and isinstance(b, int)):
            errors.append("tool_observation_count_decreased")
        obs = obs_values[-1] if obs_values and isinstance(obs_values[-1], int) else 0
        if obs != final.get("observation_tokens"):
            errors.append("final_observation_count_mismatch")
        environment = final.get("environment")
        if isinstance(environment, dict):
            if environment.get("out_tokens") != generated:
                errors.append("environment_generated_count_mismatch")
            if environment.get("obs_tokens") != obs:
                errors.append("environment_observation_count_mismatch")
            if environment.get("tokens_used") != generated + obs:
                errors.append("environment_raw_count_mismatch")
        raw = generated + obs
        if final.get("tokens_generated_plus_feedback") != raw:
            errors.append("final_raw_count_mismatch")
        for event in tool_results:
            turn_id = event.get("turn_id")
            if event.get("tokens_generated") != generated_by_turn.get(turn_id):
                errors.append(f"tool_result_{turn_id}:generated_cumulative_mismatch")
        if final.get("turns") != len(turns):
            errors.append("final_turn_count_mismatch")
        if arm != "base-direct" and final.get("tool_calls") != len(tool_results):
            errors.append("final_tool_call_count_mismatch")
        infra = (
            (isinstance(environment, dict) and environment.get("aborted") is True)
            or final.get("status") == "infrastructure_error"
            or bool(final.get("reason") and "infrastructure" in str(final.get("reason")).lower())
        )
        predictions, source_details = {}, {}
        extraction_error = None
        if arm == "base-direct":
            if turns:
                predictions, extraction_error = parse_direct(
                    turns[-1].get("generated_text", ""), targets
                )
            else:
                extraction_error = "raw_model_turn_missing"
            predictions = predictions or {}
        else:
            predictions, source_details = validate_tool_submission(
                episodes_root / task_id, toolkit_root, spec
            )
        source_blocked = arm != "base-direct" and bool(
            source_details.get("error") or source_details.get("errors")
        )
        per_target = {}
        for target in targets:
            candidate = predictions.get(target)
            if arm == "base-direct":
                per_target[target] = compare_grid(candidate, spec.targets[target])
            else:
                attempts = {str(slot): compare_grid(grid, spec.targets[target])
                            for slot, grid in (candidate or {}).items()}
                per_target[target] = {"exact": any(row["exact"] for row in attempts.values()),
                                      "attempts": attempts}
            if source_blocked:
                per_target[target]["exact"] = None
        exact = (None if source_blocked else
                 bool(per_target) and all(item["exact"] for item in per_target.values()))
        reported_per_target = final.get("per_target")
        per_target_match = (
            None if exact is None or not isinstance(reported_per_target, dict)
            or not reported_per_target else all(
                reported_per_target.get(t) == row["exact"]
                for t, row in per_target.items()
            )
        )
        result.update({
            "independent_exact": exact,
            "per_target": per_target,
            "raw_json_error": extraction_error,
            "source_validation": source_details,
            "accounting": {
                "turns": len(turns),
                "generated_from_ids": generated,
                "reported_generated": final.get("generated_tokens"),
                "tool_results": len(tool_results),
                "observation_cumulative": obs,
                "reported_observation": final.get("observation_tokens"),
                "raw_tokens": raw,
                "raw_limit": RAW_LIMIT,
                "within_raw_limit": raw <= RAW_LIMIT,
                "infrastructure_or_abort": infra,
                "errors": sorted(set(errors)),
            },
            "reported_exact_matches": None if exact is None else final.get("exact") == exact,
            "reported_per_target_matches": per_target_match,
            "within_budget_success": bool(exact and raw <= RAW_LIMIT and not infra),
            "status": "validated" if not errors and not source_blocked else "mismatch",
        })
        task_results.append(result)
        by_task[task_id] = result

    sampling = manifest.get("sampling", {})
    config_path = resolve_model_config(manifest)
    quantization_none = None
    if config_path:
        config = read_json(config_path)
        quantization_none = not bool(config.get("quantization_config"))
    stage_complete = manifest.get("status") == "complete"
    expected = set(TASK_IDS)
    scope_complete = chosen == expected
    ids_complete = (
        manifest_ids == expected and len(manifest_list_ids) == len(TASK_IDS)
        and start_ids == expected and len(starts) == len(TASK_IDS)
        and final_ids == expected and len(finals) == len(TASK_IDS)
    )
    complete = (
        stage_complete and scope_complete and ids_complete and not jsonl_errors
        and not start_dupes and not dupes
    )
    validation_errors = []
    if not complete:
        validation_errors.append("run_incomplete_or_id_set_mismatch")
    if jsonl_errors:
        validation_errors.extend(jsonl_errors)
    if dupes:
        validation_errors.append("duplicate_task_finals")
    if start_dupes:
        validation_errors.append("duplicate_task_starts")
    if len(manifest_list_ids) != len(TASK_IDS):
        validation_errors.append("manifest_task_id_count_mismatch")
    validation_errors.extend(
        f"{task_id}:{error}"
        for task_id, result in by_task.items()
        for error in result.get("accounting", {}).get("errors", [])
    )
    for task_id, result in by_task.items():
        source = result.get("source_validation", {})
        if source.get("error"):
            validation_errors.append(f"{task_id}:{source['error']}")
        if source.get("session_inputs_match") is False:
            validation_errors.append(f"{task_id}:session_input_mismatch")
        validation_errors.extend(
            f"{task_id}:{error}" for error in source.get("errors", [])
        )
        if result.get("snapshot_input_match") is False:
            validation_errors.append(f"{task_id}:snapshot_input_mismatch")
        if result.get("reported_exact_matches") is False:
            validation_errors.append(f"{task_id}:reported_exact_mismatch")
        if result.get("reported_per_target_matches") is False:
            validation_errors.append(f"{task_id}:reported_per_target_mismatch")
    config_checks = {
        "model_config_found": config_path is not None,
        "model_config_quantization_none": quantization_none,
        "dtype_bfloat16": str(sampling.get("dtype", "")).lower() in ("bf16", "bfloat16"),
        "MTP_disabled": sampling.get("MTP") is False,
        "kv_cache_auto": str(sampling.get("kv_cache_dtype", "")).lower().startswith("auto"),
    }
    runtime_path = toolkit_root / "arc_toolkit/runtime.py"
    expected_runtime_hash = manifest.get("toolkit_runtime_sha256")
    runtime_hash_match = (
        runtime_path.is_file()
        and isinstance(expected_runtime_hash, str)
        and hashlib.sha256(runtime_path.read_bytes()).hexdigest() == expected_runtime_hash
    )
    if not runtime_hash_match:
        validation_errors.append("toolkit_runtime_sha256_mismatch")
    if not all(value is True for value in config_checks.values()):
        validation_errors.append("model_config_or_sampling_check_failed")
    validation_valid = (
        complete
        and not validation_errors
        and all(result.get("independent_exact") is not None for result in task_results)
    )
    return {
        "arm": arm,
        "jsonl": str(jsonl_path),
        "manifest": str(manifest_path),
        "manifest_status": manifest.get("status"),
        "thinking_enabled": sampling.get("enable_thinking"),
        "stage_complete": stage_complete,
        "complete_eight_task_receipt": complete,
        "independent_validation_valid": validation_valid,
        "manifest_task_ids_match": manifest_ids == expected,
        "task_start_ids_match": start_ids == expected,
        "task_final_ids_match": final_ids == expected,
        "toolkit_runtime_hash_matches_manifest": runtime_hash_match,
        "filtered_scope": not scope_complete,
        "model_config": str(config_path) if config_path else None,
        "model_checks": config_checks,
        "model_checks_pass": all(value is True for value in config_checks.values()),
        "task_results": task_results,
        "summary": {
            "finals": len(final_ids & chosen),
            "independent_exact": sum(r.get("independent_exact") is True for r in task_results),
            "within_budget_success": sum(r.get("within_budget_success") is True for r in task_results),
            "validation_errors": validation_errors,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", required=True,
                        help="pilot JSONL; repeat once per arm")
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT.parents[1] / "arc/arc2/dataset")
    parser.add_argument("--toolkit", type=Path, default=ROOT / "vendor/paired-factory-v1")
    parser.add_argument("--task-id", action="append",
                        help="validate selected task IDs for a partial CPU spot check")
    args = parser.parse_args()
    selected = set(args.task_id) if args.task_id else None
    runs = [validate_run(Path(path), args.dataset_dir, args.toolkit, selected)
            for path in args.jsonl]
    print(json.dumps({
        "schema": "qwen-9b-pilot-validation-v1",
        "read_only": True,
        "prediction_sources": {
            "base-direct": "raw model_turn.generated_text JSON",
            "tool-arms": "Session.open(episode_root).read(submission_ref)",
        },
        "runs": runs,
    }, ensure_ascii=False, indent=2))
    return 0 if all(run["complete_eight_task_receipt"] for run in runs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
