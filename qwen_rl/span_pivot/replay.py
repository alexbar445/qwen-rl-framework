"""Isolated replay of ARC tool-call lines against the pinned toolkit runtime.

The receipt is deliberately built from the runtime's persisted session and actual
responses. It is evidence about a replay, not a reconstruction of an agent's
unseen observations or reasoning.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_TOOLKIT_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "paired-factory-v1"
_CALL_NAME = re.compile(r"^\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*\(")
_TASK_TARGET = re.compile(r"test/(\d+)\Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _runtime(toolkit_root: Path):
    """Load the selected runtime and reject an already-loaded different copy.

    Python caches imported packages, so compare distinct toolkit roots in
    separate processes.
    """
    root = Path(toolkit_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("toolkit_root must be an existing real directory")
    root = root.resolve(strict=True)
    path_text = str(root)
    inserted = path_text not in sys.path
    if inserted:
        sys.path.insert(0, path_text)
    try:
        runtime = importlib.import_module("arc_toolkit.runtime")
        common = importlib.import_module("arc_toolkit.common")
        contracts = importlib.import_module("arc_toolkit.trajectory.contracts")
        module_path = Path(runtime.__file__).resolve(strict=True)
        if not module_path.is_relative_to(root):
            raise RuntimeError(
                f"arc_toolkit.runtime was loaded from {module_path}, outside pinned root {root}"
            )
        return runtime.Session, common, contracts, root
    finally:
        if inserted:
            try:
                sys.path.remove(path_text)
            except ValueError:
                pass


def _implementation_fingerprint(contracts, root: Path) -> tuple[dict[str, str], str]:
    hashes = contracts.implementation_hashes(root)
    if not hashes:
        raise ValueError("toolkit implementation hashes are unavailable")
    normalized = dict(hashes)
    plugin_resource = root / "integrations" / "hermes_plugin" / "plugin.yaml"
    if plugin_resource.exists():
        if plugin_resource.is_symlink() or not plugin_resource.is_file():
            raise ValueError("toolkit plugin resource must be a regular file")
        normalized["integrations/hermes_plugin/plugin.yaml"] = _sha256(plugin_resource.read_bytes())
    normalized = dict(sorted(normalized.items()))
    return normalized, _digest(normalized)


def _session_state(session) -> dict[str, Any]:
    path = session.root / "session.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("session state must be a regular file")
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_projection(state: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    record = state["artifacts"][artifact_id]
    return {
        "ref": artifact_id,
        "type": record["type"],
        # Session.execute validates the persisted tree before each subsequent
        # call; a final Session.open validates the last boundary. The runtime's
        # content hash is the canonical persisted artifact data hash.
        "data_sha256": record["content_hash"],
        "parents": list(record["parents"]),
    }


def _submission_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    submissions: list[dict[str, Any]] = []
    for target in sorted(state["submissions"], key=lambda value: (
            int(value.split("/", 1)[1]) if value.startswith("test/") else -1, value)):
        for ordinal, artifact_id in enumerate(state["submissions"][target]):
            artifact = _artifact_projection(state, artifact_id)
            if artifact["type"] != "Submission":
                raise ValueError(f"submission index references a non-submission artifact {artifact_id}")
            submissions.append({
                "target": target,
                "ordinal": ordinal,
                "artifact_id": artifact_id,
                "artifact_data_sha256": artifact["data_sha256"],
            })
    return submissions


def _read_artifact_data(session, state: dict[str, Any], artifact_id: str) -> Any:
    record = state["artifacts"][artifact_id]
    path = session.root / record["path"]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact data for {artifact_id} must be a regular file")
    payload = path.read_bytes()
    if _sha256(payload) != record["content_hash"]:
        raise ValueError(f"artifact data hash mismatch for {artifact_id}")
    value = json.loads(payload)
    if _canonical_bytes(value) != payload:
        raise ValueError(f"artifact data is not canonical JSON for {artifact_id}")
    return value


def _submission_projection(session, state: dict[str, Any]) -> list[dict[str, Any]]:
    submissions: list[dict[str, Any]] = []
    for item in _submission_state(state):
        artifact_id = item["artifact_id"]
        data = _read_artifact_data(session, state, artifact_id)
        if data.get("target") != item["target"]:
            raise ValueError(f"submission index disagrees with artifact {artifact_id}")
        grid = data.get("prediction")
        submissions.append({
            **item,
            "grid": grid,
            "data_sha256": _digest(grid),
        })
    return submissions


def _business_state(session, state: dict[str, Any], catalog_sha256: str,
                    implementation_sha256: str,
                    implementation_hashes: dict[str, str]) -> dict[str, Any]:
    inputs = {
        alias: {"type": record["kind"], "data_sha256": record["sha256"]}
        for alias, record in sorted(state["inputs"].items())
    }
    artifacts = [
        _artifact_projection(state, artifact_id)
        for artifact_id in sorted(state["artifacts"], key=lambda value: int(value[1:]))
    ]
    return {
        "tool_version": state["tool_version"],
        "task_context": state.get("task_context"),
        "inputs": inputs,
        "artifacts": artifacts,
        "submissions": _submission_state(state),
        "catalog_sha256": catalog_sha256,
        "implementation_sha256": implementation_sha256,
        "implementation_hashes": implementation_hashes,
    }


def _call_class(line: str) -> tuple[str | None, str]:
    match = _CALL_NAME.match(line)
    name = match.group(1) if match else None
    if name in {"tools", "inspect"}:
        return name, "read_only"
    return name, "side_effect"


def _required_targets(task: dict[str, Any]) -> list[str]:
    if isinstance(task.get("submission_targets"), list):
        return list(task["submission_targets"])
    return [f"test/{index}" for index in range(len(task.get("test", [])))]


def _coverage(required: list[str], submissions: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for submission in submissions:
        target = submission["target"]
        counts[target] = counts.get(target, 0) + 1
    required_set = set(required)
    actual_set = set(counts)
    return {
        "required_targets": required,
        "submitted_targets": sorted(actual_set),
        "missing_targets": sorted(required_set - actual_set),
        "unexpected_targets": sorted(actual_set - required_set),
        "submission_counts": dict(sorted(counts.items())),
        "complete": required_set == actual_set and all(counts.get(target, 0) >= 1 for target in required),
    }


def run_replay(task: dict, task_id: str, lines: list[str], output_dir: Path, *,
               toolkit_root: Path = DEFAULT_TOOLKIT_ROOT) -> dict[str, Any]:
    """Execute every provided call line in a fresh isolated Session.

    ``output_dir`` must be new or empty and is used only as the sandbox session
    root. Errors returned by the toolkit are preserved as actual responses and
    make the overall receipt unsuccessful; later lines are still executed.
    """
    receipt: dict[str, Any] = {
        "schema_version": "arc-span-replay-v1",
        "task_id": task_id,
        "successful": False,
        "execution_complete": False,
        "error": None,
        "calls": [],
        "submissions": [],
        "call_counts": {
            "total": 0, "successful": 0, "errors": 0,
            "read_only": 0, "side_effect": 0, "tools": 0, "inspect": 0,
            "read_only_state_changes": 0,
        },
    }
    session = None
    try:
        if not isinstance(lines, list) or any(type(line) is not str for line in lines):
            raise ValueError("lines must be a list of call strings")
        output_dir = Path(output_dir)
        if output_dir.is_symlink():
            raise ValueError("output_dir may not be a symlink")
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise ValueError("output_dir must be new or empty")
        Session, common, contracts, toolkit_root = _runtime(Path(toolkit_root))
        implementation_hashes, implementation_sha256 = _implementation_fingerprint(contracts, toolkit_root)
        session = Session.create(
            output_dir, task, task_id=task_id, max_calls=max(1, len(lines)),
        )
        catalog_sha256 = common.content_hash(session.catalog())
        initial_state = _business_state(
            session, _session_state(session), catalog_sha256,
            implementation_sha256, implementation_hashes,
        )
        receipt.update({
            "session_dir": str(session.root),
            "toolkit": {
                "root": str(toolkit_root),
                "tool_version": common.VERSION,
                "catalog_sha256": catalog_sha256,
                "implementation_sha256": implementation_sha256,
                "implementation_hashes": implementation_hashes,
            },
            "initial_state_sha256": _digest(initial_state),
            "required_submission_targets": _required_targets(task),
        })

        for call_index, line in enumerate(lines):
            name, classification = _call_class(line)
            before_state = _business_state(
                session, _session_state(session), catalog_sha256,
                implementation_sha256, implementation_hashes,
            )
            started = time.monotonic()
            response = session.execute(line)
            elapsed_seconds = time.monotonic() - started
            state_after_call = _session_state(session)
            after_state = _business_state(
                session, state_after_call, catalog_sha256,
                implementation_sha256, implementation_hashes,
            )
            successful = response.get("status") == "ok"
            artifacts = []
            artifact_id = response.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id in state_after_call["artifacts"]:
                artifacts.append(_artifact_projection(state_after_call, artifact_id))
            state_changed = _digest(before_state) != _digest(after_state)
            record = {
                "call_index": call_index,
                "line": line,
                "name": name,
                "classification": classification,
                "elapsed_seconds": elapsed_seconds,
                "successful": successful,
                "response": response,
                "artifacts": artifacts,
                "semantic_state_before_sha256": _digest(before_state),
                "semantic_state_after_sha256": _digest(after_state),
                "business_state_changed": state_changed,
            }
            receipt["calls"].append(record)
            counts = receipt["call_counts"]
            counts["total"] += 1
            counts["successful" if successful else "errors"] += 1
            counts[classification] += 1
            if name in {"tools", "inspect"}:
                counts[name] += 1
            if classification == "read_only" and state_changed:
                counts["read_only_state_changes"] += 1

        # Validate the final persisted tree once, then read actual submission
        # grids directly from their registered artifacts below.
        Session.open(session.root)
        final_state = _business_state(
            session, _session_state(session), catalog_sha256,
            implementation_sha256, implementation_hashes,
        )
        submissions = _submission_projection(session, _session_state(session))
        receipt["submissions"] = submissions
        receipt["submission_coverage"] = _coverage(receipt["required_submission_targets"], submissions)
        receipt["final_state_sha256"] = _digest(final_state)
        receipt["semantic_state"] = final_state
        receipt["execution_complete"] = True
        receipt["successful"] = receipt["call_counts"]["errors"] == 0
        return receipt
    except Exception as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if session is not None:
            try:
                receipt["session_dir"] = str(session.root)
            except Exception:
                pass
        return receipt


def _call_state_after(receipt: dict[str, Any], call_index: int | None) -> str:
    if call_index is None:
        value = receipt.get("final_state_sha256")
    else:
        calls = receipt.get("calls", [])
        if type(call_index) is not int or not 0 <= call_index < len(calls):
            raise ValueError("pivot end indices must refer to existing zero-based call indices")
        value = calls[call_index].get("semantic_state_after_sha256")
    if not isinstance(value, str):
        raise ValueError("replay receipt has no semantic state at the requested boundary")
    return value


def _read_only_proof(receipt: dict[str, Any]) -> dict[str, Any]:
    actions = []
    counts = {"tools": 0, "inspect": 0}
    for call in receipt.get("calls", []):
        if call.get("classification") != "read_only":
            continue
        if call.get("name") in counts:
            counts[call["name"]] += 1
        before = call.get("semantic_state_before_sha256")
        after = call.get("semantic_state_after_sha256")
        actions.append({
            "call_index": call.get("call_index"),
            "line": call.get("line"),
            "response": call.get("response"),
            "response_sha256": _digest(call.get("response")),
            "state_preserving": before == after,
        })
    return {
        "actual_calls": actions,
        "all_state_preserving": all(item["state_preserving"] for item in actions),
        "count": len(actions),
        "counts": counts,
    }


def _side_effect_results(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "line": call.get("line"),
            "successful": call.get("successful"),
            "response": call.get("response"),
            "artifacts": call.get("artifacts", []),
        }
        for call in receipt.get("calls", [])
        if call.get("classification") == "side_effect"
    ]


def compare_replays(original: dict, rewritten: dict, *,
                    pivot_end_indices: tuple[int, int] | None = None) -> dict[str, Any]:
    """Conservatively compare two actual replay receipts.

    ``pivot_end_indices`` gives (original, rewritten) zero-based call indices;
    each checkpoint is the semantic state immediately after that actual call.
    Read-only outputs are reported as observed, never synthesized or inferred.
    A recovered, identical original tool error may be faithfully replayed;
    execution completeness, all successful reads, submission coverage and
    exact side-effect response equality are required separately. This proves
    execution equivalence, not model continuation or distribution equivalence.
    """
    original_pivot = rewritten_pivot = None
    try:
        if pivot_end_indices is not None:
            if (not isinstance(pivot_end_indices, (tuple, list))
                    or len(pivot_end_indices) != 2):
                raise ValueError("pivot_end_indices must contain original and rewritten indices")
            original_pivot = _call_state_after(original, pivot_end_indices[0])
            rewritten_pivot = _call_state_after(rewritten, pivot_end_indices[1])
        original_final = _call_state_after(original, None)
        rewritten_final = _call_state_after(rewritten, None)
        original_reads = _read_only_proof(original)
        rewritten_reads = _read_only_proof(rewritten)
        state_match = original_final == rewritten_final
        pivot_match = None if pivot_end_indices is None else original_pivot == rewritten_pivot
        results_match = _side_effect_results(original) == _side_effect_results(rewritten)
        submissions_match = original.get("submissions", []) == rewritten.get("submissions", [])
        complete = (
            original.get("submission_coverage", {}).get("complete") is True
            and rewritten.get("submission_coverage", {}).get("complete") is True
        )
        runs_successful = original.get("successful") is True and rewritten.get("successful") is True
        runs_complete = original.get("execution_complete") is True and rewritten.get("execution_complete") is True
        reads_successful = all(call.get("successful") is True
            for receipt in (original, rewritten) for call in receipt.get("calls", [])
            if call.get("classification") == "read_only")
        equivalent = bool(
            runs_complete and state_match and results_match and submissions_match
            and complete and original_reads["all_state_preserving"]
            and rewritten_reads["all_state_preserving"]
            and reads_successful and pivot_match is not False
        )
        differences = []
        if not runs_complete:
            differences.append("one_or_both_replays_incomplete")
        if not reads_successful:
            differences.append("read_only_call_failed")
        if pivot_match is False:
            differences.append("pivot_state_differs")
        if not state_match:
            differences.append("business_state_differs")
        if not results_match:
            differences.append("side_effect_results_differ")
        if not submissions_match:
            differences.append("actual_submissions_differ")
        if not complete:
            differences.append("required_submissions_incomplete")
        if not original_reads["all_state_preserving"] or not rewritten_reads["all_state_preserving"]:
            differences.append("read_only_action_changed_business_state")
        return {
            "equivalent": equivalent,
            "successful_replays": runs_successful,
            "complete_replays": runs_complete,
            "recovered_errors_preserved": results_match and not runs_successful,
            "business_state_match": state_match,
            "pivot_state_match": pivot_match,
            "side_effect_results_match": results_match,
            "actual_submissions_match": submissions_match,
            "complete_submissions": complete,
            "read_only_proof": {"original": original_reads, "rewritten": rewritten_reads},
            "differences": differences,
        }
    except Exception as exc:
        return {
            "equivalent": False,
            "successful_replays": False,
            "business_state_match": False,
            "pivot_state_match": None,
            "side_effect_results_match": False,
            "actual_submissions_match": False,
            "complete_submissions": False,
            "read_only_proof": {
                "original": _read_only_proof(original) if isinstance(original, dict) else {"actual_calls": [], "all_state_preserving": False, "count": 0},
                "rewritten": _read_only_proof(rewritten) if isinstance(rewritten, dict) else {"actual_calls": [], "all_state_preserving": False, "count": 0},
            },
            "differences": ["comparison_error"],
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def trusted_score(submissions: list[dict[str, Any]], task_id: str,
                  solutions_path: Path) -> dict[str, Any]:
    """Score a frozen prediction payload against ARC's private solution file.

    The caller must freeze and hash the prediction payload before calling this
    function. The return contains only a boolean and fingerprints; target grids
    are never returned or copied into an error message.
    """
    if not isinstance(submissions, list) or not submissions:
        raise ValueError("submissions must be a non-empty frozen prediction list")
    normalized: list[dict[str, Any]] = []
    for item in submissions:
        if not isinstance(item, dict) or not isinstance(item.get("target"), str):
            raise ValueError("each submission must contain a target and grid")
        target = item["target"]
        match = _TASK_TARGET.fullmatch(target)
        grid = item.get("grid")
        if not match or not isinstance(grid, list):
            raise ValueError("each submission must name a test target and include its grid")
        normalized.append({"target": target, "grid": grid})
    prediction_sha256 = _digest(normalized)

    path = Path(solutions_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("solutions_path must be a regular file")
    solution_bytes = path.read_bytes()
    try:
        solution_data = json.loads(solution_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("solutions file is not valid JSON") from None
    if not isinstance(solution_data, dict) or task_id not in solution_data:
        raise ValueError("task_id is unavailable in solutions file")
    answers = solution_data[task_id]
    if not isinstance(answers, list) or not answers:
        raise ValueError("solutions for task_id must be a non-empty list")

    by_target: dict[str, list[list[list[int]]]] = {}
    for submission in normalized:
        by_target.setdefault(submission["target"], []).append(submission["grid"])
    expected_targets = {f"test/{index}" for index in range(len(answers))}
    exact = set(by_target) == expected_targets and all(
        len(grids) == 1 and grids[0] == answers[int(target.split("/", 1)[1])]
        for target, grids in by_target.items()
    )
    return {
        "exact": bool(exact),
        "submission_count": len(normalized),
        "expected_target_count": len(answers),
        "prediction_sha256": prediction_sha256,
        "solutions_file_sha256": _sha256(solution_bytes),
    }
