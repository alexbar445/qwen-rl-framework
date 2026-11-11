#!/usr/bin/env python3
"""Task source: the ARC-AGI-2 public data (training 1000 + held-out evaluation set).

Pools
    arc2_train   arc-agi_training_challenges.json (1000 tasks) -- the RL curriculum source
    arc2_eval    arc-agi_evaluation_challenges.json (120 tasks) -- held-out reading, never trained

Variants (the reason this module is not a one-line JSON loader)
    931/1000 training tasks ship exactly one test input, so training on the raw task means
    training on one fixed instance over and over.  For every training task with k
    demonstration pairs we therefore also build k *variants*: variant i hides pair i, i.e.

        train = the other pairs        test = [pair i's input]        target = pair i's output

    That is "a familiar rule with a new input" (design §9.4's first category) and multiplies
    the usable instance count by ~3 without inventing any rule the data does not contain.
    The official task itself is kept too (its test inputs have no visible outputs).

Everything here is deterministic: ordering is by (task id, variant index) and sampling is
seeded by a stable hash, so a curriculum selection is reproducible.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

POOLS = {
    "arc2_train": ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
    "arc2_eval": ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"),
}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str                 # "<pool>:<arc id>" (+ ":v<i>" for a variant)
    family: str                  # the ARC task id -- the grouping key for reference costs
    pool: str                    # arc2_train | arc2_eval
    split: str                   # train | test  (test = the held-out evaluation pool)
    variant: int | None          # which demonstration pair was hidden (None = official task)
    challenge: dict              # {"train": [...], "test": [{"input": ...}]}
    targets: dict                # {"test/<i>": grid}
    public_id: str               # toolkit-safe runtime id
    # Generated curriculum (`arc-agent-task-v2`): all three are absent for ARC-AGI-2 tasks.
    instruction: str | None = None        # explicit task instruction (categories 1-4 have no demos)
    input_roles: dict[str, str] | None = None  # "solve" / reference-only role per visible input
    submission_targets: tuple[str, ...] = ()   # required targets; may exclude reference inputs
    difficulty: str | None = None         # generator difficulty label (easy | medium | hard | mixed)
    conventions: str | None = None        # public task-specific assumptions, not a hidden rule

    @property
    def train_pairs(self) -> int:
        return len(self.challenge["train"])

    @property
    def test_inputs(self) -> int:
        return len(self.challenge["test"])


def _load_pool(pool: str, dataset_dir: Path) -> tuple[dict, dict]:
    challenges_file, solutions_file = POOLS[pool]
    challenges = json.loads((dataset_dir / challenges_file).read_text())
    solutions = json.loads((dataset_dir / solutions_file).read_text())
    if not isinstance(challenges, dict) or not isinstance(solutions, dict):
        raise ValueError(f"{pool} challenges and solutions must be JSON objects")
    if set(challenges) != set(solutions):
        raise ValueError(f"{pool} solutions must cover exactly every public task")
    for task_id, challenge in challenges.items():
        if not isinstance(challenge, dict) or not isinstance(challenge.get("train"), list) \
                or not isinstance(challenge.get("test"), list) or not challenge["test"]:
            raise ValueError(f"{pool} task {task_id!r} has an invalid challenge")
        for pair in challenge["test"]:
            if not isinstance(pair, dict) or "input" not in pair or "output" in pair:
                raise ValueError(f"{pool} public test challenge contains an output: {task_id!r}")
        targets = solutions[task_id]
        if not isinstance(targets, list) or len(targets) != len(challenge["test"]) \
                or any(not isinstance(grid, list) or not grid for grid in targets):
            raise ValueError(f"{pool} private solutions are incomplete: {task_id!r}")
    return challenges, solutions


def _spec(pool: str, split: str, task_id: str, challenge: dict, targets: dict,
          variant: int | None) -> TaskSpec:
    suffix = f":v{variant}" if variant is not None else ""
    public = f"{task_id}{'-v' + str(variant) if variant is not None else ''}"
    return TaskSpec(task_id=f"{pool}:{task_id}{suffix}", family=task_id, pool=pool, split=split,
                    variant=variant, challenge=challenge, targets=targets, public_id=public)


def load_tasks(pools: list[str] | tuple[str, ...] = ("arc2_train",), variants: bool = True,
               *, dataset_dir: Path, include_official: bool = True) -> list[TaskSpec]:
    """All tasks of the given pools, deterministically ordered.

    A mixed catalog takes precedence. Legacy roots containing generated-curriculum bundles
    (`arc-agent-task-v2`) continue to be served by ``task_source_generated``.
    """
    dataset_dir = Path(dataset_dir)
    requested = tuple(pools)
    unknown = set(requested) - {*POOLS, "synthetic_arc"}
    if unknown:
        raise ValueError(f"unknown task pools: {sorted(unknown)}")
    from .task_catalog import load_source_catalog
    catalog = load_source_catalog(dataset_dir)
    from .task_source_generated import is_generated_root, load_tasks as load_generated_tasks
    if catalog is not None:
        tasks: list[TaskSpec] = []
        train_requested = "arc2_train" in requested
        for source in catalog.sources:
            if source.pool == "synthetic_arc" and (train_requested or "synthetic_arc" in requested):
                generated = load_generated_tasks(source.path, split="train")
                if not generated:
                    raise ValueError("mixed task catalog synthetic_arc source has no training tasks")
                tasks.extend(generated)
            elif source.pool == "arc2_train" and train_requested:
                challenges, solutions = _load_pool("arc2_train", source.path)
                if include_official:
                    for task_id in sorted(challenges):
                        task = challenges[task_id]
                        targets = {f"test/{index}": grid
                                   for index, grid in enumerate(solutions[task_id])}
                        tasks.append(_spec("arc2_train", "train", task_id, task,
                                           targets, None))
                # Catalog v1 fixes variants=false, regardless of the caller's legacy flag.
        if "arc2_eval" in requested:
            arc2 = catalog.source_for("arc2_train")
            if arc2 is None:
                raise ValueError("mixed task catalog has no official ARC2 source")
            challenges, solutions = _load_pool("arc2_eval", arc2.path)
            if include_official:
                for task_id in sorted(challenges):
                    task = challenges[task_id]
                    targets = {f"test/{index}": grid
                               for index, grid in enumerate(solutions[task_id])}
                    tasks.append(_spec("arc2_eval", "test", task_id, task,
                                       targets, None))
        return tasks
    if is_generated_root(dataset_dir):
        return load_generated_tasks(dataset_dir)
    tasks: list[TaskSpec] = []
    for pool in pools:
        challenges, solutions = _load_pool(pool, dataset_dir)
        split = "test" if pool == "arc2_eval" else "train"
        for task_id in sorted(challenges):
            task = challenges[task_id]
            if include_official:
                targets = {f"test/{index}": grid
                           for index, grid in enumerate(solutions.get(task_id, []))}
                tasks.append(_spec(pool, split, task_id, task, targets, None))
            if not variants or pool != "arc2_train":
                continue
            pairs = task["train"]
            for index, pair in enumerate(pairs):
                if not isinstance(pair.get("output"), list):
                    continue
                variant_challenge = {
                    "train": [p for j, p in enumerate(pairs) if j != index],
                    "test": [{"input": pair["input"]}],
                }
                tasks.append(_spec(pool, split, task_id, variant_challenge,
                                   {"test/0": pair["output"]}, index))
    return tasks


def _order_key(spec: TaskSpec, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{spec.task_id}".encode()).hexdigest()


def sample_tasks(split: str, pools: list[str], n_tasks: int, seed: int = 20261008,
                 variants: bool = True, *, dataset_dir: Path,
                 families: list[str] | None = None) -> list[TaskSpec]:
    """Deterministic sample of n tasks from the given pools/split (one spec per family).

    ``families`` restricts the sample to those public task ids.  The design trains only on
    families with a non-zero success rate, so a curriculum is an explicit family list rather
    than a re-rolled seed.
    """
    candidates = [t for t in load_tasks(pools, variants=variants, dataset_dir=dataset_dir)
                  if t.split == split]
    if families is not None:
        wanted = set(families)
        candidates = [t for t in candidates if t.family in wanted]
        missing = wanted - {t.family for t in candidates}
        if missing:
            raise ValueError(f"no {split} task found for families: {sorted(missing)}")
    # one instance per family per sample keeps the reading interpretable; the variant
    # index rotates with the seed so repeated sweeps cover different instances
    by_family: dict[str, list[TaskSpec]] = {}
    for spec in candidates:
        by_family.setdefault(spec.family, []).append(spec)
    chosen: list[TaskSpec] = []
    for family in sorted(by_family, key=lambda f: _order_key(by_family[f][0], seed)):
        options = sorted(by_family[family], key=lambda s: (s.variant is None, s.variant or 0))
        chosen.append(options[seed % len(options)])
        if len(chosen) >= n_tasks:
            break
    return chosen


def index_summary(dataset_dir: Path) -> dict:
    """Counts per pool, for the record."""
    dataset_dir = Path(dataset_dir)
    summary: dict[str, dict] = {}
    from .task_catalog import load_source_catalog
    catalog = load_source_catalog(dataset_dir)
    if catalog is not None:
        arc2 = catalog.source_for("arc2_train")
        if arc2 is None:
            raise ValueError("mixed task catalog has no official ARC2 source")
        for pool in POOLS:
            challenges, solutions = _load_pool(pool, arc2.path)
            summary[pool] = {
                "tasks": len(challenges),
                "with_solution": sum(1 for task_id in challenges if task_id in solutions),
                "test_inputs": sum(len(task["test"]) for task in challenges.values()),
                "variants": 0,
            }
        from .task_source_generated import load_tasks as load_generated_tasks
        generated = load_generated_tasks(
            catalog.source_for("synthetic_arc").path, split="train")
        if not generated:
            raise ValueError("mixed task catalog synthetic_arc source has no training tasks")
        summary["synthetic_arc"] = {
            "tasks": len(generated),
            "with_solution": sum(bool(task.targets) for task in generated),
            "test_inputs": sum(task.test_inputs for task in generated),
            "variants": 0,
        }
        return summary
    for pool in POOLS:
        challenges, solutions = _load_pool(pool, dataset_dir)
        variants = sum(len(t["train"]) for t in challenges.values()) if pool == "arc2_train" else 0
        summary[pool] = {
            "tasks": len(challenges),
            "with_solution": sum(1 for t in challenges if t in solutions),
            "test_inputs": sum(len(t["test"]) for t in challenges.values()),
            "variants": variants,
        }
    return summary


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ARC-AGI-2 task source summary")
    ap.add_argument("--dataset-dir", type=Path, required=True)
    args = ap.parse_args()
    summary = index_summary(args.dataset_dir)
    print(json.dumps({"dataset_dir": str(args.dataset_dir), **summary}, indent=2,
                     ensure_ascii=False))
    sample = sample_tasks("train", ["arc2_train"], 3, seed=20261008,
                          dataset_dir=args.dataset_dir)
    for spec in sample:
        print(f"  {spec.task_id} train={spec.train_pairs} test={spec.test_inputs} "
              f"targets={sorted(spec.targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
