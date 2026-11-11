#!/usr/bin/env python3
"""Task source for the generated instruction curriculum (`arc-agent-task-v2`).

The curriculum is produced by the generator workspace `qwen_RL_dataset` (vendored here at
``vendor/arc-taskgen-v2``).  Each dataset directory carries the standard public/private
layout::

    public/<split>.jsonl    category / difficulty / challenge / instruction / input_roles /
                            submission_targets / task_id  (everything the student may see)
    private/<split>.jsonl   provenance + verifier.targets        (trusted side only)
    manifest.json           schema_version + counts + per-file sha256

Two properties of this curriculum drive the code below:

* categories 1-4 ship **no demonstrations** (`train: []`) and are legal only together with the
  explicit instruction contract (`instruction` / `input_roles` / `submission_targets`); the
  vendored toolkit's ``validate_task`` accepts exactly ``{train,test}`` or
  ``{train,test,instruction,input_roles,submission_targets}``.
* the generator supplies trusted target verification; the runtime computes its own
  success-only continuous cost reward without length taper. This module never widens
  targets or invents demonstrations.

Nothing here talks to a model: it only pairs a public task with its private target record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .paths import ROOT
from .task_source import TaskSpec
from .prompt import RULE_INFERENCE_INSTRUCTION, GENERATED_INFERENCE_CONVENTIONS

SCHEMA_VERSION = "arc-agent-task-v2"
VENDOR = ROOT / "vendor" / "arc-taskgen-v2"
EXTENDED_FIELDS = ("instruction", "input_roles", "submission_targets")
POOL = "synthetic_arc"
# Curriculum order: category 5 (rule inference from demonstrations -- the closest match to the SFT
# regime, one target per task) first, then category 6 (multi-target composition, also with
# demonstrations), then the explicit-instruction categories 1-4 that ship no demonstrations, with
# the light-reasoning templates (computed paths/regions) last.  Every task stays loaded; only the
# order the learner walks them changes.
CURRICULUM_ORDER = (5, 6, 1, 2, 3, 4)
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "mixed": 1, "hard": 2}


def _curriculum_rank(family: str, difficulty: str | None) -> tuple[int, int, str]:
    """Sort key: category rank, then difficulty rank, then the family string itself."""
    category = family.split(":", 1)[0]
    try:
        index = CURRICULUM_ORDER.index(int(category))
    except (ValueError, TypeError):
        index = len(CURRICULUM_ORDER)
    return (index, DIFFICULTY_ORDER.get(difficulty or "", len(DIFFICULTY_ORDER)), family)


def _generator():
    """Import the vendored generator package (pure Python, no third-party dependencies)."""
    vendored = str(VENDOR)
    if vendored not in sys.path:
        sys.path.insert(0, vendored)
    from arc_taskgen import dataset as generator_dataset      # noqa: E402
    from arc_taskgen import verifier as generator_verifier    # noqa: E402

    return generator_dataset, generator_verifier


def dataset_roots(root: Path, names: list[str] | tuple[str, ...] | None = None) -> list[Path]:
    """Every v2 dataset directory under ``root`` (or ``root`` itself), deterministically ordered."""
    from .task_catalog import generated_source_path
    root = generated_source_path(Path(root))
    if (root / "manifest.json").is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
    else:
        candidates = []
    roots: list[Path] = []
    for path in candidates:
        manifest = path / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text())
        except ValueError:
            continue
        if data.get("schema_version") != SCHEMA_VERSION:
            continue
        if names and path.name not in names:
            continue
        roots.append(path)
    return roots


def is_generated_root(root: Path) -> bool:
    """True when ``root`` looks like a generated-curriculum directory (cheap, no generator import)."""
    try:
        return bool(dataset_roots(root))
    except OSError:
        return False


def load_tasks(dataset_dir: Path, *, split: str = "train",
               datasets: list[str] | tuple[str, ...] | None = None) -> list[TaskSpec]:
    """Every task of every v2 dataset under ``dataset_dir``, ordered by task id.

    ``datasets`` selects dataset directories by name; ``None`` mixes all of them.  Trusted
    loading audits each bundle against its manifest (the generator's own ``load_split``).
    """
    generator_dataset, _ = _generator()
    by_id: dict[str, TaskSpec] = {}
    for root in dataset_roots(dataset_dir, datasets):
        for public, private in generator_dataset.load_split(root, split, trusted=True):
            spec = _spec(public, private)
            previous = by_id.get(spec.task_id)
            if previous is not None and previous != spec:
                raise ValueError(f"generated task ID collision has different task specs: {spec.task_id}")
            by_id[spec.task_id] = spec
    tasks = list(by_id.values())
    tasks.sort(key=lambda spec: (_curriculum_rank(spec.family, spec.difficulty), spec.task_id))
    return tasks


def public_instruction(public: dict) -> str:
    """Rule-inference categories share the official ARC task instruction."""
    return (RULE_INFERENCE_INSTRUCTION if public["category"] in (5, 6)
            else public["instruction"])


def _spec(public: dict, private: dict) -> TaskSpec:
    """Bridge one generated task into the frozen ``TaskSpec`` shape (trusted targets included)."""
    instruction = public_instruction(public)
    return TaskSpec(
        task_id=public["task_id"],
        family=private["provenance"]["family"],
        pool=POOL,
        split=private["split"],
        variant=None,
        challenge=json.loads(json.dumps(public["challenge"])),
        targets=json.loads(json.dumps(private["verifier"]["targets"])),
        public_id=public["task_id"],
        instruction=instruction,
        conventions=(GENERATED_INFERENCE_CONVENTIONS if public["category"] in (5, 6) else None),
        input_roles=dict(public["input_roles"]),
        submission_targets=tuple(public["submission_targets"]),
        difficulty=public.get("difficulty"),
    )


def task_payload(spec: TaskSpec) -> dict:
    """The toolkit task dict for ``Session.create``: visible task plus its explicit context."""
    payload = json.loads(json.dumps(spec.challenge))
    if spec.instruction is None:
        return payload
    payload["instruction"] = spec.instruction
    payload["input_roles"] = dict(spec.input_roles or {})
    payload["submission_targets"] = list(spec.submission_targets)
    return payload
