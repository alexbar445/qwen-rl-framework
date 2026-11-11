"""Frozen, provenance-checked tasksets for periodic ARC capability evaluation.

The public manifest contains only task identities, public metadata, and hashes of
public challenge inputs.  ``TaskSpec.targets`` stay in memory for the trusted
verifier and are never serialized here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from ..task_source import TaskSpec
from ..task_source_generated import SCHEMA_VERSION, dataset_roots, load_tasks


MANIFEST_SCHEMA = "qwen-rl-evaluation-taskset-v1"
EXCLUSION_PROOF_SCHEMA = "qwen-rl-evaluation-exclusion-proof-v1"
DEFAULT_DATASET_NAME = "template-holdout-600-v2"
DEFAULT_SPLIT = "test"
DEFAULT_SEED = 20261012
DEFAULT_TASK_COUNT = 20
_DIFFICULTIES = ("easy", "medium", "hard")
_BEPA_REFERENCE_TABLES = ("episodes", "plans", "rescue_jobs", "cache_entries", "reservations")


class EvaluationDatasetError(ValueError):
    """The frozen evaluation set or its eligibility evidence is invalid."""


@dataclass(frozen=True)
class EvaluationDataset:
    """Trusted task specs paired with a public, answer-free frozen manifest."""

    tasks: tuple[TaskSpec, ...]
    manifest: dict[str, Any]

    def task_seed(self, task_id: str) -> int:
        """Return the seed fixed for this task across policies and evaluation rounds."""
        for row in self.manifest["tasks"]:
            if row["task_id"] == task_id:
                return int(row["task_seed"])
        raise KeyError(task_id)

    @property
    def task_seeds(self) -> dict[str, int]:
        return {row["task_id"]: int(row["task_seed"]) for row in self.manifest["tasks"]}


def canonical_json_sha256(value: Any) -> str:
    """SHA-256 of stable JSON, shared with evaluation result writers."""
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationDatasetError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDatasetError(f"cannot read JSON provenance: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationDatasetError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise EvaluationDatasetError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDatasetError(f"cannot read JSONL provenance: {path}") from exc
    return rows


def _require_sha256(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise EvaluationDatasetError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _public_challenge(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"train", "test"}:
        raise EvaluationDatasetError(f"{field} must contain only train and test")
    train, test = value["train"], value["test"]
    if not isinstance(train, list) or not isinstance(test, list) or not test:
        raise EvaluationDatasetError(f"{field} has invalid train/test rows")
    for index, pair in enumerate(train):
        if (not isinstance(pair, dict) or not isinstance(pair.get("input"), list)
                or not isinstance(pair.get("output"), list)):
            raise EvaluationDatasetError(f"{field}.train[{index}] is malformed")
    for index, item in enumerate(test):
        if (not isinstance(item, dict) or not isinstance(item.get("input"), list)
                or "output" in item):
            raise EvaluationDatasetError(f"{field}.test[{index}] must be input-only")
    return {"train": train, "test": test}


def _task_challenge_sha256(challenge: Mapping[str, Any]) -> str:
    return canonical_json_sha256(_public_challenge(dict(challenge), field="challenge"))


def build_sft_exclusion_proof(
    snapshot_dir: Path,
    sft_data_dir: Path,
    training_challenges_path: Path,
) -> dict[str, Any]:
    """Build a host-verified, public-only inventory of SFT/teacher task exposure.

    Only task IDs and hashes of model-visible challenge inputs leave this helper.
    Assistant/teacher trajectories and private ARC test solutions are not copied.
    """
    snapshot_dir, sft_data_dir = Path(snapshot_dir), Path(sft_data_dir)
    training_challenges_path = Path(training_challenges_path)
    selection_path = snapshot_dir / "selection.jsonl"
    snapshot_manifest_path = snapshot_dir / "SNAPSHOT.json"
    sft_manifest_path = sft_data_dir / "manifest.json"
    train_path, eval_path = sft_data_dir / "train.jsonl", sft_data_dir / "eval.jsonl"
    required = (selection_path, snapshot_manifest_path, sft_manifest_path,
                train_path, eval_path, training_challenges_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise EvaluationDatasetError(f"missing SFT provenance files: {missing}")

    snapshot_manifest = _read_json(snapshot_manifest_path)
    if not isinstance(snapshot_manifest.get("counts"), dict):
        raise EvaluationDatasetError("SFT snapshot manifest has no task counts")
    selection_rows = _read_jsonl(selection_path)
    selection_ids = [row.get("task_id") for row in selection_rows]
    if (not selection_ids or any(not isinstance(task_id, str) or not task_id
                                 for task_id in selection_ids)
            or len(selection_ids) != len(set(selection_ids))):
        raise EvaluationDatasetError("SFT selection task IDs are missing or duplicated")
    if snapshot_manifest["counts"].get("tasks") != len(selection_ids):
        raise EvaluationDatasetError("SFT snapshot task count differs from its selection")

    record_challenges: dict[str, str] = {}
    record_root = snapshot_dir / "records"
    if not record_root.is_dir():
        raise EvaluationDatasetError("SFT snapshot records directory is missing")
    for path in sorted(record_root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        record = _read_json(path)
        task_id, visible = record.get("task_id"), record.get("visible_task")
        if not isinstance(task_id, str) or not task_id or task_id in record_challenges:
            raise EvaluationDatasetError(f"invalid or duplicate SFT task in {path.name}")
        if not isinstance(visible, dict) or not {"train", "test"} <= set(visible):
            raise EvaluationDatasetError(f"SFT record has no public challenge: {path.name}")
        challenge = _public_challenge(
            {"train": visible["train"], "test": visible["test"]},
            field=f"{path.name}.visible_task",
        )
        record_challenges[task_id] = canonical_json_sha256(challenge)
    if set(selection_ids) != set(record_challenges):
        raise EvaluationDatasetError("SFT selection IDs do not match snapshot records")

    sft_manifest = _read_json(sft_manifest_path)
    workers = sft_manifest.get("workers")
    dataset = sft_manifest.get("dataset")
    if not isinstance(workers, list) or not workers or not isinstance(dataset, dict):
        raise EvaluationDatasetError("SFT data manifest lacks worker/source provenance")
    worker_ids = {row.get("task_id") for row in workers if isinstance(row, dict)}
    if (len(worker_ids) != len(workers)
            or any(not isinstance(task_id, str) or not task_id for task_id in worker_ids)):
        raise EvaluationDatasetError("SFT worker task IDs are invalid")
    train_ids: set[str] = set()
    for path in (train_path, eval_path):
        for row in _read_jsonl(path):
            meta = row.get("meta")
            task_id = meta.get("task_id") if isinstance(meta, dict) else None
            if not isinstance(task_id, str) or not task_id:
                raise EvaluationDatasetError(f"SFT corpus row lacks task_id: {path.name}")
            train_ids.add(task_id)
    if train_ids != worker_ids:
        raise EvaluationDatasetError("SFT corpus task IDs differ from its source manifest")

    try:
        training_challenges = _read_json(training_challenges_path)
    except EvaluationDatasetError:
        raise
    expected_training_sha = _require_sha256(dataset.get("sha256"), "SFT source dataset sha256")
    if _sha256_file(training_challenges_path) != expected_training_sha:
        raise EvaluationDatasetError("ARC training challenge source differs from SFT manifest")
    if worker_ids - set(training_challenges):
        raise EvaluationDatasetError("SFT worker task IDs are absent from ARC training source")

    task_ids = set(selection_ids) | worker_ids
    tasks: list[dict[str, str]] = []
    for task_id in sorted(task_ids):
        challenge_hash = record_challenges.get(task_id)
        if challenge_hash is None:
            challenge_hash = canonical_json_sha256(_public_challenge(
                training_challenges[task_id], field=f"ARC training task {task_id}"
            ))
        tasks.append({"task_id": task_id, "public_challenge_sha256": challenge_hash})

    selection_projection = [{
        "task_id": row["task_id"],
        "split": row.get("split"),
    } for row in selection_rows]
    corpus_id_projection = sorted(train_ids)
    record_projection = [
        {"task_id": task_id, "public_challenge_sha256": record_challenges[task_id]}
        for task_id in sorted(record_challenges)
    ]
    source_artifacts = [
        {"name": "qwen_sft_snapshot_manifest", "original_path": str(snapshot_manifest_path.resolve()),
         "sha256": _sha256_file(snapshot_manifest_path), "sha256_scope": "file",
         "verified_on_host": True},
        {"name": "qwen_sft_selection_task_projection", "original_path": str(selection_path.resolve()),
         "sha256": canonical_json_sha256(selection_projection),
         "sha256_scope": "task_id_split_projection", "verified_on_host": True},
        {"name": "qwen_sft_snapshot_public_task_projection", "original_path": str(record_root.resolve()),
         "sha256": canonical_json_sha256(record_projection),
         "sha256_scope": "public_challenge_projection", "verified_on_host": True},
        {"name": "qwen_sft_data_manifest", "original_path": str(sft_manifest_path.resolve()),
         "sha256": _sha256_file(sft_manifest_path), "sha256_scope": "file",
         "verified_on_host": True},
        {"name": "qwen_sft_corpus_task_id_projection",
         "original_path": f"{train_path.resolve()} + {eval_path.resolve()}",
         "sha256": canonical_json_sha256(corpus_id_projection),
         "sha256_scope": "task_id_projection", "verified_on_host": True},
        {"name": "arc2_training_public_source", "original_path": str(training_challenges_path.resolve()),
         "sha256": expected_training_sha, "sha256_scope": "file", "verified_on_host": True},
    ]
    proof: dict[str, Any] = {
        "schema": EXCLUSION_PROOF_SCHEMA,
        "host_verified": True,
        "sources": source_artifacts,
        "sft_task_count": len(tasks),
        "sft_task_ids": sorted(task_ids),
        "sft_public_challenge_sha256": sorted({row["public_challenge_sha256"] for row in tasks}),
        "task_inventory": tasks,
        "task_inventory_sha256": canonical_json_sha256(tasks),
    }
    proof["proof_sha256"] = canonical_json_sha256(proof)
    return proof


def _read_sft_proof(path: Path) -> dict[str, Any]:
    proof = _read_json(path)
    if proof.get("schema") != EXCLUSION_PROOF_SCHEMA or proof.get("host_verified") is not True:
        raise EvaluationDatasetError("SFT exclusion proof is missing host-verified provenance")
    expected_proof_sha = _require_sha256(proof.get("proof_sha256"), "proof_sha256")
    body = {key: value for key, value in proof.items() if key != "proof_sha256"}
    if canonical_json_sha256(body) != expected_proof_sha:
        raise EvaluationDatasetError("SFT exclusion proof digest mismatch")
    sources = proof.get("sources")
    required_sources = {
        "qwen_sft_snapshot_manifest", "qwen_sft_selection_task_projection",
        "qwen_sft_snapshot_public_task_projection", "qwen_sft_data_manifest",
        "qwen_sft_corpus_task_id_projection", "arc2_training_public_source",
    }
    if not isinstance(sources, list):
        raise EvaluationDatasetError("SFT exclusion proof has no source receipts")
    source_names: set[str] = set()
    for row in sources:
        if not isinstance(row, dict) or row.get("verified_on_host") is not True:
            raise EvaluationDatasetError("SFT source receipt was not verified on the host")
        name = row.get("name")
        if not isinstance(name, str) or name in source_names:
            raise EvaluationDatasetError("SFT source receipt names are missing or duplicated")
        if not isinstance(row.get("original_path"), str) or not row["original_path"]:
            raise EvaluationDatasetError(f"SFT source {name} has no original path")
        _require_sha256(row.get("sha256"), f"SFT source {name} sha256")
        source_names.add(name)
    if not required_sources <= source_names:
        raise EvaluationDatasetError(
            f"SFT exclusion proof lacks sources: {sorted(required_sources - source_names)}"
        )
    inventory = proof.get("task_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise EvaluationDatasetError("SFT exclusion proof has no task inventory")
    task_ids: set[str] = set()
    challenge_hashes: set[str] = set()
    for row in inventory:
        if not isinstance(row, dict):
            raise EvaluationDatasetError("SFT inventory rows must be objects")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in task_ids:
            raise EvaluationDatasetError("SFT inventory task IDs are invalid or duplicated")
        challenge_hashes.add(_require_sha256(
            row.get("public_challenge_sha256"), "SFT public_challenge_sha256"
        ))
        task_ids.add(task_id)
    if sorted(task_ids) != proof.get("sft_task_ids"):
        raise EvaluationDatasetError("SFT proof task ID summary differs from inventory")
    if sorted(challenge_hashes) != proof.get("sft_public_challenge_sha256"):
        raise EvaluationDatasetError("SFT proof challenge summary differs from inventory")
    if (proof.get("sft_task_count") != len(inventory)
            or proof.get("task_inventory_sha256") != canonical_json_sha256(inventory)):
        raise EvaluationDatasetError("SFT proof inventory digest/count mismatch")
    return proof


def _public_manifest_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise EvaluationDatasetError("generated manifest has no files table")
    public_files = {name: files[name] for name in sorted(files)
                    if isinstance(name, str) and name.startswith("public/")}
    if not public_files:
        raise EvaluationDatasetError("generated manifest has no public file digests")
    return {
        "schema_version": manifest.get("schema_version"),
        "generator_version": manifest.get("generator_version"),
        "counts": manifest.get("counts"),
        "files": public_files,
    }


def _read_public_rows(root: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvaluationDatasetError(f"unsupported generated dataset: {root}")
    relative = f"public/{split}.jsonl"
    expected = manifest.get("files", {}).get(relative)
    if not isinstance(expected, dict):
        raise EvaluationDatasetError(f"manifest has no {relative} digest")
    path = root / relative
    if not path.is_file():
        raise EvaluationDatasetError(f"generated public split is missing: {path}")
    raw_sha = _sha256_file(path)
    if (expected.get("sha256") != raw_sha
            or expected.get("bytes") != path.stat().st_size):
        raise EvaluationDatasetError(f"generated public split differs from manifest: {path}")
    rows = _read_jsonl(path)
    expected_count = manifest.get("counts", {}).get(split)
    if expected_count != len(rows):
        raise EvaluationDatasetError(f"generated public split count mismatch: {path}")
    public_manifest_sha = canonical_json_sha256(_public_manifest_projection(manifest))
    return rows, {
        "dataset": root.name,
        "split": split,
        "path": f"{root.name}/{relative}",
        "file_sha256": raw_sha,
        "public_manifest_sha256": public_manifest_sha,
        "row_count": len(rows),
    }


def _load_training_inventory(dataset_dir: Path) -> tuple[set[str], set[str], dict[str, Any]]:
    roots = dataset_roots(Path(dataset_dir))
    if not roots:
        raise EvaluationDatasetError(f"no generated training datasets under {dataset_dir}")
    task_ids: set[str] = set()
    challenge_hashes: set[str] = set()
    source_rows: list[dict[str, Any]] = []
    task_rows = 0
    for root in roots:
        rows, source = _read_public_rows(root, "train")
        source_rows.append(source)
        task_rows += len(rows)
        for row in rows:
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise EvaluationDatasetError(f"training task has no ID in {source['path']}")
            challenge = _public_challenge(row.get("challenge"), field=f"{source['path']}:{task_id}")
            task_ids.add(task_id)
            challenge_hashes.add(canonical_json_sha256(challenge))
    source_rows.sort(key=lambda row: row["dataset"])
    inventory = {
        "datasets": source_rows,
        "task_rows": task_rows,
        "unique_task_ids": len(task_ids),
        "unique_public_challenges": len(challenge_hashes),
        "inventory_sha256": canonical_json_sha256(sorted(challenge_hashes)),
    }
    return task_ids, challenge_hashes, inventory


def _load_bepa_inventory(path: Path) -> tuple[set[str], dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise EvaluationDatasetError(f"BEPA provenance database is missing: {path}")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"tasks", *_BEPA_REFERENCE_TABLES} <= tables:
            raise EvaluationDatasetError("BEPA DB is missing required task/exposure tables")
        task_rows = connection.execute("SELECT task_key, spec_json FROM tasks").fetchall()
        public_by_key: dict[str, str] = {}
        registered_hashes: set[str] = set()
        for task_key, spec_text in task_rows:
            if not isinstance(task_key, str) or task_key in public_by_key:
                raise EvaluationDatasetError("BEPA task keys are invalid or duplicated")
            try:
                spec = json.loads(spec_text)
            except (TypeError, json.JSONDecodeError) as exc:
                raise EvaluationDatasetError("BEPA task spec JSON is invalid") from exc
            if not isinstance(spec, dict) or not isinstance(spec.get("public_challenge"), dict):
                raise EvaluationDatasetError("BEPA task has no public challenge provenance")
            public_challenge = _public_challenge(
                spec["public_challenge"], field="BEPA public_challenge"
            )
            challenge_hash = canonical_json_sha256(public_challenge)
            public_by_key[task_key] = challenge_hash
            registered_hashes.add(challenge_hash)
        if not task_rows:
            raise EvaluationDatasetError("BEPA DB has no registered task provenance")

        for table in _BEPA_REFERENCE_TABLES:
            rows = connection.execute(f'SELECT task_key FROM "{table}"').fetchall()
            for (task_key,) in rows:
                if task_key not in public_by_key:
                    raise EvaluationDatasetError(
                        f"BEPA {table} references an unregistered task"
                    )
    except sqlite3.Error as exc:
        raise EvaluationDatasetError(f"cannot read BEPA DB read-only: {path}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    registered_rows_hash = canonical_json_sha256(sorted(public_by_key.values()))
    inventory = {
        "registered_task_rows": len(task_rows),
        "unique_public_challenges": len(registered_hashes),
        "registered_public_challenges_sha256": registered_rows_hash,
        "exposure_tables": list(_BEPA_REFERENCE_TABLES),
    }
    return registered_hashes, inventory


def _category(row: Mapping[str, Any], spec: TaskSpec) -> int:
    category = row.get("category")
    if type(category) is not int or category <= 0:
        raise EvaluationDatasetError(f"invalid category for {spec.task_id}")
    family_category = spec.family.split(":", 1)[0]
    if family_category.isdigit() and int(family_category) != category:
        raise EvaluationDatasetError(f"TaskSpec family/category mismatch for {spec.task_id}")
    return category


def _load_candidate_tasks(
    dataset_dir: Path, dataset_name: str,
) -> tuple[dict[str, TaskSpec], dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    roots = dataset_roots(Path(dataset_dir), names=(dataset_name,))
    if len(roots) != 1:
        raise EvaluationDatasetError(
            f"expected one dataset root named {dataset_name}, found {len(roots)}"
        )
    evaluation_root = roots[0]
    public_rows, evaluation_source = _read_public_rows(evaluation_root, DEFAULT_SPLIT)
    public_by_id: dict[str, dict[str, Any]] = {}
    public_hash_by_id: dict[str, str] = {}
    for row in public_rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in public_by_id:
            raise EvaluationDatasetError("evaluation split task IDs are invalid or duplicated")
        category = row.get("category")
        difficulty = row.get("difficulty")
        if type(category) is not int or category <= 0 or difficulty not in _DIFFICULTIES:
            raise EvaluationDatasetError(f"evaluation task metadata is invalid: {task_id}")
        challenge = _public_challenge(row.get("challenge"), field=f"evaluation task {task_id}")
        public_by_id[task_id] = row
        public_hash_by_id[task_id] = canonical_json_sha256(challenge)

    specs = load_tasks(Path(dataset_dir), split=DEFAULT_SPLIT, datasets=(dataset_name,))
    spec_by_id: dict[str, TaskSpec] = {}
    for spec in specs:
        if spec.split != DEFAULT_SPLIT:
            raise EvaluationDatasetError(f"non-test TaskSpec in evaluation split: {spec.task_id}")
        if spec.task_id in spec_by_id:
            raise EvaluationDatasetError(f"duplicate trusted TaskSpec: {spec.task_id}")
        spec_by_id[spec.task_id] = spec
    if set(spec_by_id) != set(public_by_id):
        raise EvaluationDatasetError("trusted TaskSpecs differ from the public evaluation split")
    for task_id, spec in spec_by_id.items():
        row = public_by_id[task_id]
        _category(row, spec)
        if spec.difficulty != row["difficulty"] or spec.challenge != row["challenge"]:
            raise EvaluationDatasetError(f"TaskSpec metadata/content mismatch: {task_id}")
    return spec_by_id, public_by_id, public_hash_by_id, evaluation_source


def _frozen_task_row(
    task_id: str, spec: TaskSpec, public_row: Mapping[str, Any],
    challenge_hash: str, seed: int,
) -> dict[str, Any]:
    category = _category(public_row, spec)
    difficulty = public_row["difficulty"]
    return {
        "task_id": task_id,
        "category": category,
        "family": spec.family,
        "difficulty": difficulty,
        "stratum": {"category": category, "difficulty": difficulty},
        "public_challenge_sha256": challenge_hash,
        "task_seed": _task_seed(seed, task_id),
    }


def _overlap_counts(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    sft_ids: set[str],
    sft_hashes: set[str],
    train_ids: set[str],
    train_hashes: set[str],
    teacher_hashes: set[str],
) -> dict[str, int]:
    counts = {
        "sft_task_ids": 0,
        "sft_public_challenges": 0,
        "rl_train_task_ids": 0,
        "rl_train_public_challenges": 0,
        "bepa_registered_public_challenges": 0,
    }
    for row in task_rows:
        task_id = row["task_id"]
        challenge_hash = row["public_challenge_sha256"]
        overlaps = (
            task_id in sft_ids,
            challenge_hash in sft_hashes,
            task_id in train_ids,
            challenge_hash in train_hashes,
            challenge_hash in teacher_hashes,
        )
        for name, overlap in zip(counts, overlaps):
            counts[name] += int(overlap)
    return counts


def _task_seed(seed: int, task_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _selection_key(seed: int, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0select\0{task_id}".encode("utf-8")).hexdigest()


def _stratum_key(seed: int, key: tuple[int, str]) -> str:
    return hashlib.sha256(f"{seed}\0stratum\0{key[0]}\0{key[1]}".encode("utf-8")).hexdigest()


def _stratified_select(
    candidates: Sequence[dict[str, Any]], count: int, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(row["category"], row["difficulty"])].append(row)
    if not groups or count > len(candidates):
        raise EvaluationDatasetError(
            f"requested {count} tasks from only {len(candidates)} eligible candidates"
        )
    exact = {key: count * len(rows) / len(candidates) for key, rows in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    order = sorted(groups, key=lambda key: (
        -(exact[key] - quotas[key]), _stratum_key(seed, key)
    ))
    for key in order[:remaining]:
        quotas[key] += 1
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: (
            _selection_key(seed, row["task_id"]), row["task_id"]
        ))
        selected.extend(ordered[:quotas[key]])
    selected.sort(key=lambda row: (
        row["category"], _DIFFICULTIES.index(row["difficulty"]),
        _selection_key(seed, row["task_id"]), row["task_id"],
    ))
    return selected, {
        f"category-{category}/{difficulty}": quota
        for (category, difficulty), quota in sorted(quotas.items())
    }


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    return canonical_json_sha256({key: value for key, value in manifest.items()
                                  if key != "manifest_sha256"})


def build_evaluation_dataset(
    dataset_dir: Path,
    *,
    sft_proof_path: Path,
    teacher_db_path: Path,
    base_fingerprint: str,
    seed: int = DEFAULT_SEED,
    count: int = DEFAULT_TASK_COUNT,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> EvaluationDataset:
    """Select a deterministic eligible split and build its public frozen manifest.

    SFT provenance is a host-verified public projection because the Qwen training
    container does not mount qwen_sft.  Current RL train splits and the live BEPA
    task registry are read and rechecked on every call.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluationDatasetError("seed must be a non-negative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise EvaluationDatasetError("count must be a positive integer")
    base_fingerprint = _require_sha256(base_fingerprint, "base_fingerprint")
    proof = _read_sft_proof(Path(sft_proof_path))
    train_ids, train_hashes, train_inventory = _load_training_inventory(Path(dataset_dir))
    teacher_hashes, teacher_inventory = _load_bepa_inventory(Path(teacher_db_path))
    spec_by_id, public_by_id, public_hash_by_id, evaluation_source = _load_candidate_tasks(
        Path(dataset_dir), dataset_name
    )
    sft_ids = set(proof["sft_task_ids"])
    sft_hashes = set(proof["sft_public_challenge_sha256"])
    eligible: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, Any]] = []
    for task_id, spec in spec_by_id.items():
        row = public_by_id[task_id]
        candidate = _frozen_task_row(
            task_id, spec, row, public_hash_by_id[task_id], seed
        )
        all_candidate_rows.append(candidate)
        overlaps = (
            task_id in sft_ids,
            candidate["public_challenge_sha256"] in sft_hashes,
            task_id in train_ids,
            candidate["public_challenge_sha256"] in train_hashes,
            candidate["public_challenge_sha256"] in teacher_hashes,
        )
        if any(overlaps):
            continue
        eligible.append(candidate)
    overlap_counts = _overlap_counts(
        all_candidate_rows, sft_ids=sft_ids, sft_hashes=sft_hashes,
        train_ids=train_ids, train_hashes=train_hashes, teacher_hashes=teacher_hashes,
    )
    if len(eligible) < count:
        raise EvaluationDatasetError(
            f"only {len(eligible)} eligible tasks remain; requested {count}"
        )
    selected, quotas = _stratified_select(eligible, count, seed)
    task_rows = selected

    source_hashes = {
        "evaluation_public_split_sha256": evaluation_source["file_sha256"],
        "evaluation_public_manifest_sha256": evaluation_source["public_manifest_sha256"],
        "rl_training_public_inventory_sha256": canonical_json_sha256(train_inventory),
        "sft_exclusion_proof_sha256": proof["proof_sha256"],
        "bepa_registered_public_challenges_sha256": (
            teacher_inventory["registered_public_challenges_sha256"]
        ),
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "dataset_name": dataset_name,
        "split": DEFAULT_SPLIT,
        "seed": seed,
        "count": count,
        "base_fingerprint": base_fingerprint,
        "sources_sha256": canonical_json_sha256(source_hashes),
        "source_hashes": source_hashes,
        "task_ids": [row["task_id"] for row in task_rows],
        "tasks": task_rows,
        "stratification": {
            "fields": ["category", "difficulty"],
            "eligible_strata": len({
                (row["category"], row["difficulty"]) for row in eligible
            }),
            "selected_quotas": quotas,
        },
        "eligibility_audit": {
            "candidate_count": len(spec_by_id),
            "eligible_count": len(eligible),
            "excluded_candidate_count": len(spec_by_id) - len(eligible),
            "overlap_counts": overlap_counts,
            "selected_overlap_counts": _overlap_counts(
                task_rows, sft_ids=sft_ids, sft_hashes=sft_hashes,
                train_ids=train_ids, train_hashes=train_hashes,
                teacher_hashes=teacher_hashes,
            ),
            "sft": {
                "task_count": proof["sft_task_count"],
                "public_challenge_count": len(sft_hashes),
                "proof_sha256": proof["proof_sha256"],
            },
            "rl_training": train_inventory,
            "bepa_registry": teacher_inventory,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    selected_specs = tuple(spec_by_id[row["task_id"]] for row in task_rows)
    return EvaluationDataset(selected_specs, manifest)


def load_frozen_evaluation_dataset(
    dataset_dir: Path,
    *,
    manifest_path: Path,
    sft_proof_path: Path,
    teacher_db_path: Path,
    base_fingerprint: str,
) -> EvaluationDataset:
    """Load the exact frozen tasks and reject only changed evidence or new leakage.

    The frozen task IDs never get reselected.  Current train and BEPA inventories
    may grow with unrelated tasks; only an ID or public challenge collision with a
    frozen evaluation task blocks loading.
    """
    frozen = _read_json(Path(manifest_path))
    if frozen.get("schema") == "qwen-rl-arc2-official-evaluation-v1":
        from .official_arc2 import load_frozen_official_arc2_evaluation_dataset

        return load_frozen_official_arc2_evaluation_dataset(
            dataset_dir,
            manifest_path=manifest_path,
            sft_proof_path=sft_proof_path,
            teacher_db_path=teacher_db_path,
            base_fingerprint=base_fingerprint,
            frozen=frozen,
        )
    if frozen.get("schema") != MANIFEST_SCHEMA:
        raise EvaluationDatasetError("unsupported evaluation taskset manifest")
    if canonical_json_sha256({key: value for key, value in frozen.items()
                              if key != "manifest_sha256"}) != frozen.get("manifest_sha256"):
        raise EvaluationDatasetError("evaluation manifest digest mismatch")
    expected_base = _require_sha256(base_fingerprint, "base_fingerprint")
    if frozen.get("base_fingerprint") != expected_base:
        raise EvaluationDatasetError("evaluation manifest base fingerprint mismatch")
    seed, count, dataset_name = frozen.get("seed"), frozen.get("count"), frozen.get("dataset_name")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluationDatasetError("frozen evaluation seed is invalid")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise EvaluationDatasetError("frozen evaluation task count is invalid")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise EvaluationDatasetError("frozen evaluation dataset name is invalid")
    source_hashes = frozen.get("source_hashes")
    if (not isinstance(source_hashes, dict)
            or frozen.get("sources_sha256") != canonical_json_sha256(source_hashes)):
        raise EvaluationDatasetError("evaluation source digest summary is invalid")
    proof = _read_sft_proof(Path(sft_proof_path))
    if source_hashes.get("sft_exclusion_proof_sha256") != proof["proof_sha256"]:
        raise EvaluationDatasetError("current SFT exclusion proof differs from frozen provenance")
    train_ids, train_hashes, _ = _load_training_inventory(Path(dataset_dir))
    teacher_hashes, _ = _load_bepa_inventory(Path(teacher_db_path))
    spec_by_id, public_by_id, public_hash_by_id, evaluation_source = _load_candidate_tasks(
        Path(dataset_dir), dataset_name
    )
    if (source_hashes.get("evaluation_public_split_sha256") != evaluation_source["file_sha256"]
            or source_hashes.get("evaluation_public_manifest_sha256")
            != evaluation_source["public_manifest_sha256"]):
        raise EvaluationDatasetError("frozen public evaluation source changed")

    task_rows = frozen.get("tasks")
    task_ids = frozen.get("task_ids")
    if (not isinstance(task_rows, list) or len(task_rows) != count
            or not isinstance(task_ids, list) or len(task_ids) != count):
        raise EvaluationDatasetError("frozen evaluation task list/count is invalid")
    row_ids = [row.get("task_id") if isinstance(row, dict) else None for row in task_rows]
    if (any(not isinstance(task_id, str) or not task_id for task_id in row_ids)
            or len(set(row_ids)) != count or task_ids != row_ids):
        raise EvaluationDatasetError("frozen evaluation task IDs are missing or duplicated")

    expected_task_keys = {
        "task_id", "category", "family", "difficulty", "stratum",
        "public_challenge_sha256", "task_seed",
    }
    selected_specs: list[TaskSpec] = []
    for row in task_rows:
        task_id = row["task_id"]
        spec = spec_by_id.get(task_id)
        public_row = public_by_id.get(task_id)
        if spec is None or public_row is None:
            raise EvaluationDatasetError(f"frozen evaluation task is absent from source: {task_id}")
        if set(row) != expected_task_keys:
            raise EvaluationDatasetError(f"frozen evaluation row schema is invalid: {task_id}")
        expected_row = _frozen_task_row(
            task_id, spec, public_row, public_hash_by_id[task_id], seed
        )
        if row != expected_row:
            raise EvaluationDatasetError(f"frozen task metadata/seed/source differs: {task_id}")
        selected_specs.append(spec)

    audit = frozen.get("eligibility_audit")
    recorded_overlaps = audit.get("selected_overlap_counts") if isinstance(audit, dict) else None
    if (not isinstance(recorded_overlaps, dict)
            or set(recorded_overlaps) != {
                "sft_task_ids", "sft_public_challenges", "rl_train_task_ids",
                "rl_train_public_challenges", "bepa_registered_public_challenges",
            }
            or any(value != 0 for value in recorded_overlaps.values())):
        raise EvaluationDatasetError("frozen evaluation set was not eligible at creation")
    current_overlaps = _overlap_counts(
        task_rows,
        sft_ids=set(proof["sft_task_ids"]),
        sft_hashes=set(proof["sft_public_challenge_sha256"]),
        train_ids=train_ids,
        train_hashes=train_hashes,
        teacher_hashes=teacher_hashes,
    )
    if any(current_overlaps.values()):
        leaked = [name for name, value in current_overlaps.items() if value]
        raise EvaluationDatasetError(
            f"frozen evaluation tasks now overlap current provenance: {', '.join(leaked)}"
        )
    if frozen.get("split") != DEFAULT_SPLIT:
        raise EvaluationDatasetError("frozen evaluation taskset must use the test split")
    return EvaluationDataset(tuple(selected_specs), frozen)
