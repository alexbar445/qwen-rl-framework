"""Frozen, answer-free manifests for the official ARC-AGI-2 eval pool or its sorted tail."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..task_catalog import load_source_catalog
from ..task_source import POOLS, TaskSpec, load_tasks
from .dataset import (
    EvaluationDataset,
    EvaluationDatasetError,
    _load_bepa_inventory,
    _manifest_hash,
    _public_challenge,
    _read_sft_proof,
    _require_sha256,
    _sha256_file,
    _task_seed,
    canonical_json_sha256,
)

OFFICIAL_ARC2_MANIFEST_SCHEMA = "qwen-rl-arc2-official-evaluation-v1"
OFFICIAL_ARC2_DATASET_NAME = "arc2_eval_official"


def _task_challenge_hash(spec: TaskSpec) -> str:
    challenge = _public_challenge(spec.challenge, field=f"task {spec.task_id}")
    return canonical_json_sha256(challenge)


def _task_aliases(spec: TaskSpec) -> set[str]:
    aliases = {spec.task_id, spec.public_id, spec.family}
    if spec.pool in {"arc2_train", "arc2_eval"}:
        aliases.update({f"arc2_train:{spec.family}", f"arc2_eval:{spec.family}"})
        if spec.variant is None:
            aliases.update(
                f"{pool}:{spec.family}:v{index}"
                for pool in ("arc2_train", "arc2_eval")
                for index in range(len(spec.challenge.get("train", [])))
            )
    return {value for value in aliases if isinstance(value, str) and value}


def _training_inventory(dataset_dir: Path) -> tuple[
    set[str], set[str], set[str], dict[str, str], dict[str, Any]
]:
    """Read the mixed train source through its common TaskSpec interface."""
    try:
        tasks = tuple(load_tasks(
            pools=("arc2_train",), variants=True, dataset_dir=Path(dataset_dir)
        ))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvaluationDatasetError(f"cannot load mixed training task source: {exc}") from exc
    if not tasks:
        raise EvaluationDatasetError("mixed training task source is empty")
    ids: set[str] = set()
    families: set[str] = set()
    challenges: set[str] = set()
    alias_families: dict[str, str] = {}
    inventory_rows: list[dict[str, str]] = []
    for spec in tasks:
        if spec.split != "train":
            continue
        if (not isinstance(spec.task_id, str) or not spec.task_id
                or not isinstance(spec.family, str) or not spec.family):
            raise EvaluationDatasetError("training TaskSpec is missing its task ID or family")
        if spec.task_id in ids:
            raise EvaluationDatasetError(f"duplicate training TaskSpec ID: {spec.task_id}")
        challenge_hash = _task_challenge_hash(spec)
        ids.add(spec.task_id)
        families.add(spec.family)
        challenges.add(challenge_hash)
        for alias in _task_aliases(spec):
            previous = alias_families.get(alias)
            if previous is not None and previous != spec.family:
                raise EvaluationDatasetError(f"ambiguous training family alias: {alias}")
            alias_families[alias] = spec.family
        inventory_rows.append({
            "task_id": spec.task_id,
            "family": spec.family,
            "public_challenge_sha256": challenge_hash,
        })
    if not inventory_rows:
        raise EvaluationDatasetError("mixed training source has no train-split tasks")
    inventory_rows.sort(key=lambda row: (row["task_id"], row["family"]))
    inventory = {
        "task_count": len(inventory_rows),
        "unique_families": len(families),
        "unique_public_challenges": len(challenges),
        "inventory_sha256": canonical_json_sha256(inventory_rows),
    }
    return ids, families, challenges, alias_families, inventory


def _official_tasks(dataset_dir: Path) -> tuple[tuple[TaskSpec, ...], list[dict[str, Any]]]:
    try:
        tasks = tuple(load_tasks(
            pools=("arc2_eval",), variants=False, dataset_dir=Path(dataset_dir)
        ))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvaluationDatasetError(f"cannot load official ARC2 evaluation source: {exc}") from exc
    if not tasks:
        raise EvaluationDatasetError("official ARC2 evaluation source is empty")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for spec in tasks:
        if spec.pool != "arc2_eval" or spec.split != "test" or spec.variant is not None:
            raise EvaluationDatasetError(
                f"official ARC2 evaluation requires original, unvarianted test tasks: {spec.task_id}"
            )
        if (not isinstance(spec.task_id, str) or not spec.task_id
                or not isinstance(spec.family, str) or not spec.family
                or spec.task_id in seen):
            raise EvaluationDatasetError("official ARC2 evaluation task IDs/families are invalid")
        challenge_hash = _task_challenge_hash(spec)
        expected_targets = {f"test/{index}" for index in range(len(spec.challenge["test"]))}
        if set(spec.targets) != expected_targets:
            raise EvaluationDatasetError(f"official ARC2 trusted targets are incomplete: {spec.task_id}")
        seen.add(spec.task_id)
        rows.append({
            "task_id": spec.task_id,
            "family": spec.family,
            "public_challenge_sha256": challenge_hash,
        })
    rows.sort(key=lambda row: row["task_id"])
    tasks_by_id = {spec.task_id: spec for spec in tasks}
    ordered = tuple(tasks_by_id[row["task_id"]] for row in rows)
    return ordered, rows


def _official_source_file_inventory(dataset_dir: Path) -> dict[str, dict[str, int | str]]:
    """Fingerprint the eval challenges and trusted solution file without copying answers."""
    try:
        catalog = load_source_catalog(dataset_dir)
    except (OSError, ValueError) as exc:
        raise EvaluationDatasetError(f"cannot resolve official ARC2 source root: {exc}") from exc
    source = catalog.source_for("arc2_train") if catalog is not None else None
    root = source.path if source is not None else Path(dataset_dir)
    inventory: dict[str, dict[str, int | str]] = {}
    for name in POOLS["arc2_eval"]:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise EvaluationDatasetError(f"official ARC2 source file is missing or unsafe: {path}")
        inventory[name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return inventory


def _sft_families(proof: Mapping[str, Any], alias_families: Mapping[str, str]) -> set[str]:
    return {
        alias_families[task_id]
        for task_id in proof["sft_task_ids"]
        if task_id in alias_families
    }


def _overlap_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    sft_ids: set[str],
    sft_families: set[str],
    sft_hashes: set[str],
    train_ids: set[str],
    train_families: set[str],
    train_hashes: set[str],
    bepa_hashes: set[str],
) -> dict[str, int]:
    counts = {
        "sft_task_ids": 0,
        "sft_families": 0,
        "sft_public_challenges": 0,
        "rl_train_task_ids": 0,
        "rl_train_families": 0,
        "rl_train_public_challenges": 0,
        "bepa_registered_public_challenges": 0,
    }
    for row in rows:
        task_id = row["task_id"]
        family = row["family"]
        challenge_hash = row["public_challenge_sha256"]
        overlaps = (
            task_id in sft_ids,
            family in sft_families,
            challenge_hash in sft_hashes,
            task_id in train_ids,
            family in train_families,
            challenge_hash in train_hashes,
            challenge_hash in bepa_hashes,
        )
        for name, overlap in zip(counts, overlaps):
            counts[name] += int(overlap)
    return counts


def _frozen_row(row: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "family": row["family"],
        "public_challenge_sha256": row["public_challenge_sha256"],
        "task_seed": _task_seed(seed, row["task_id"]),
    }


def _build_context(
    dataset_dir: Path,
    *,
    sft_proof_path: Path,
    teacher_db_path: Path,
    last_n: int | None = None,
) -> tuple[
    tuple[TaskSpec, ...], list[dict[str, Any]], dict[str, Any],
    set[str], set[str], set[str], set[str], dict[str, Any], dict[str, Any],
    dict[str, dict[str, int | str]],
]:
    tasks, rows = _official_tasks(Path(dataset_dir))
    if last_n is not None and (type(last_n) is not int or not 1 <= last_n <= len(rows)):
        raise EvaluationDatasetError("last_n must be a positive integer within the official pool size")
    selected_rows = rows if last_n is None else rows[-last_n:]
    source_files = _official_source_file_inventory(Path(dataset_dir))
    proof = _read_sft_proof(Path(sft_proof_path))
    train_ids, train_families, train_hashes, alias_families, train_inventory = (
        _training_inventory(Path(dataset_dir))
    )
    for spec in tasks:
        for alias in _task_aliases(spec):
            previous = alias_families.get(alias)
            if previous is not None and previous != spec.family:
                raise EvaluationDatasetError(f"ambiguous ARC2 family alias: {alias}")
            alias_families[alias] = spec.family
    bepa_hashes, bepa_inventory = _load_bepa_inventory(Path(teacher_db_path))
    sft_families = _sft_families(proof, alias_families)
    overlaps = _overlap_counts(
        selected_rows,
        sft_ids=set(proof["sft_task_ids"]),
        sft_families=sft_families,
        sft_hashes=set(proof["sft_public_challenge_sha256"]),
        train_ids=train_ids,
        train_families=train_families,
        train_hashes=train_hashes,
        bepa_hashes=bepa_hashes,
    )
    if any(overlaps.values()):
        offenders = ", ".join(f"{name}={value}" for name, value in overlaps.items() if value)
        raise EvaluationDatasetError(
            f"official ARC2 evaluation overlaps training or teacher provenance: {offenders}"
        )
    return (tasks, rows, proof, train_ids, train_families, train_hashes, sft_families,
            train_inventory, {"bepa": bepa_inventory, "overlaps": overlaps,
                              "bepa_hashes": bepa_hashes}, source_files)


def build_official_arc2_evaluation_dataset(
    dataset_dir: Path,
    *,
    sft_proof_path: Path,
    teacher_db_path: Path,
    base_fingerprint: str,
    seed: int = 20261012,
    last_n: int | None = None,
) -> EvaluationDataset:
    """Freeze original official tasks, optionally selecting the last N sorted IDs."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluationDatasetError("seed must be a non-negative integer")
    base_fingerprint = _require_sha256(base_fingerprint, "base_fingerprint")
    (tasks, candidate_rows, proof, _train_ids, _train_families, _train_hashes, sft_families,
     train_inventory, provenance, source_files) = _build_context(
        Path(dataset_dir), sft_proof_path=sft_proof_path, teacher_db_path=teacher_db_path,
        last_n=last_n,
    )
    selected_rows = candidate_rows if last_n is None else candidate_rows[-last_n:]
    selected_tasks = tasks if last_n is None else tasks[-last_n:]
    frozen_rows = [_frozen_row(row, seed) for row in selected_rows]
    official_inventory_sha = canonical_json_sha256(candidate_rows)
    source_hashes = {
        "official_public_inventory_sha256": official_inventory_sha,
        "official_source_file_inventory_sha256": canonical_json_sha256(source_files),
        "sft_exclusion_proof_sha256": proof["proof_sha256"],
        "rl_training_public_inventory_sha256": train_inventory["inventory_sha256"],
        "bepa_registered_public_challenges_sha256": (
            provenance["bepa"]["registered_public_challenges_sha256"]
        ),
    }
    manifest: dict[str, Any] = {
        "schema": OFFICIAL_ARC2_MANIFEST_SCHEMA,
        "dataset_name": OFFICIAL_ARC2_DATASET_NAME,
        "split": "test",
        "seed": seed,
        "count": len(frozen_rows),
        "base_fingerprint": base_fingerprint,
        "sources_sha256": canonical_json_sha256(source_hashes),
        "source_hashes": source_hashes,
        "task_ids": [row["task_id"] for row in frozen_rows],
        "tasks": frozen_rows,
        "eligibility_audit": {
            "candidate_count": len(candidate_rows),
            "selected_overlap_counts": provenance["overlaps"],
            "sft": {
                "task_count": proof["sft_task_count"],
                "public_challenge_count": len(proof["sft_public_challenge_sha256"]),
                "family_count_resolved_from_mixed_train_ids": len(sft_families),
                "family_scope": "SFT IDs matched to mixed train and official eval TaskSpec aliases; all SFT challenges checked by canonical hash",
                "proof_sha256": proof["proof_sha256"],
            },
            "rl_training": train_inventory,
            "bepa_registry": provenance["bepa"],
            "policy": ("complete_official_pool_or_fail_on_any_overlap" if last_n is None
                       else "sorted_tail_or_fail_on_selected_overlap"),
        },
    }
    if last_n is not None:
        manifest["selection"] = {"method": "task_id_sorted_tail", "count": last_n}
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    return EvaluationDataset(selected_tasks, manifest)


def load_frozen_official_arc2_evaluation_dataset(
    dataset_dir: Path,
    *,
    manifest_path: Path,
    sft_proof_path: Path,
    teacher_db_path: Path,
    base_fingerprint: str,
    frozen: dict[str, Any] | None = None,
) -> EvaluationDataset:
    """Load the exact official pool and reject changed sources or new leakage."""
    from .dataset import _read_json

    manifest = frozen if frozen is not None else _read_json(Path(manifest_path))
    if manifest.get("schema") != OFFICIAL_ARC2_MANIFEST_SCHEMA:
        raise EvaluationDatasetError("unsupported official ARC2 evaluation manifest")
    if _manifest_hash(manifest) != manifest.get("manifest_sha256"):
        raise EvaluationDatasetError("official ARC2 evaluation manifest digest mismatch")
    expected_base = _require_sha256(base_fingerprint, "base_fingerprint")
    if manifest.get("base_fingerprint") != expected_base:
        raise EvaluationDatasetError("official ARC2 evaluation base fingerprint mismatch")
    seed, count = manifest.get("seed"), manifest.get("count")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluationDatasetError("official ARC2 evaluation seed is invalid")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise EvaluationDatasetError("official ARC2 evaluation count is invalid")
    selection = manifest.get("selection")
    last_n = None
    if selection is not None:
        if (not isinstance(selection, dict) or set(selection) != {"method", "count"}
                or selection.get("method") != "task_id_sorted_tail"
                or type(selection.get("count")) is not int or selection["count"] != count):
            raise EvaluationDatasetError("official ARC2 tail selection is invalid")
        last_n = count
    source_hashes = manifest.get("source_hashes")
    expected_source_names = {
        "official_public_inventory_sha256",
        "official_source_file_inventory_sha256",
        "sft_exclusion_proof_sha256",
        "rl_training_public_inventory_sha256",
        "bepa_registered_public_challenges_sha256",
    }
    if (not isinstance(source_hashes, dict)
            or set(source_hashes) != expected_source_names
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in source_hashes.values())
            or manifest.get("sources_sha256") != canonical_json_sha256(source_hashes)):
        raise EvaluationDatasetError("official ARC2 source digest summary is invalid")

    (tasks, candidate_rows, proof, train_ids, train_families, train_hashes, sft_families,
     _train_inventory, provenance, source_files) = _build_context(
        Path(dataset_dir), sft_proof_path=Path(sft_proof_path),
        teacher_db_path=Path(teacher_db_path), last_n=last_n,
    )
    selected_rows = candidate_rows if last_n is None else candidate_rows[-last_n:]
    selected_tasks = tasks if last_n is None else tasks[-last_n:]
    if source_hashes.get("official_public_inventory_sha256") != canonical_json_sha256(candidate_rows):
        raise EvaluationDatasetError("official ARC2 public evaluation source changed")
    if source_hashes.get("official_source_file_inventory_sha256") != canonical_json_sha256(source_files):
        raise EvaluationDatasetError("official ARC2 evaluation challenge/solution files changed")
    if source_hashes.get("sft_exclusion_proof_sha256") != proof["proof_sha256"]:
        raise EvaluationDatasetError("current SFT exclusion proof differs from official eval provenance")

    rows = manifest.get("tasks")
    task_ids = manifest.get("task_ids")
    if (manifest.get("dataset_name") != OFFICIAL_ARC2_DATASET_NAME
            or manifest.get("split") != "test"
            or not isinstance(rows, list) or len(rows) != count
            or count != len(selected_rows)
            or not isinstance(task_ids, list) or len(task_ids) != count):
        raise EvaluationDatasetError("official ARC2 manifest differs from its frozen pool selection")
    expected_rows = [_frozen_row(row, seed) for row in selected_rows]
    if rows != expected_rows or task_ids != [row["task_id"] for row in expected_rows]:
        raise EvaluationDatasetError("official ARC2 frozen tasks differ from the current source")

    audit = manifest.get("eligibility_audit")
    recorded = audit.get("selected_overlap_counts") if isinstance(audit, dict) else None
    if (not isinstance(recorded, dict) or set(recorded) != set(provenance["overlaps"])
            or audit.get("candidate_count") != len(candidate_rows)):
        raise EvaluationDatasetError("official ARC2 frozen eligibility audit is invalid")
    if any(type(value) is not int or value != 0 for value in recorded.values()):
        raise EvaluationDatasetError("official ARC2 manifest was not leakage-free when created")
    current = _overlap_counts(
        expected_rows,
        sft_ids=set(proof["sft_task_ids"]),
        sft_families=sft_families,
        sft_hashes=set(proof["sft_public_challenge_sha256"]),
        train_ids=train_ids,
        train_families=train_families,
        train_hashes=train_hashes,
        bepa_hashes=provenance["bepa_hashes"],
    )
    if any(current.values()):
        leaked = ", ".join(name for name, value in current.items() if value)
        raise EvaluationDatasetError(f"official ARC2 evaluation now overlaps provenance: {leaked}")
    return EvaluationDataset(selected_tasks, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--sft-exclusion-proof", type=Path, required=True)
    parser.add_argument("--teacher-db", type=Path, required=True)
    parser.add_argument("--base-fingerprint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--last-n", type=int, help="Select the last N task IDs in sorted order")
    args = parser.parse_args(argv)
    dataset = build_official_arc2_evaluation_dataset(
        args.dataset_dir,
        sft_proof_path=args.sft_exclusion_proof,
        teacher_db_path=args.teacher_db,
        base_fingerprint=args.base_fingerprint,
        seed=args.seed, last_n=args.last_n,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dataset.manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "manifest_path": str(args.output.resolve()),
        "manifest_sha256": dataset.manifest["manifest_sha256"],
        "task_count": len(dataset.tasks),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
