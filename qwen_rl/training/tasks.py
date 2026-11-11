"""Public task and reward contracts for local model training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Task:
    """One prompt/answer example exposed to a user reward function."""

    task_id: str
    prompt: str
    answer: str
    split: str = "train"
    pool: str | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "prompt", "answer", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or (name != "answer" and not value.strip()):
                raise ValueError(f"Task.{name} must be a non-empty string")
        if self.pool is not None and (
            not isinstance(self.pool, str) or not self.pool.strip()
        ):
            raise ValueError("Task.pool must be None or a non-empty string")


RewardFn = Callable[[Task, str], float]


def exact_match_reward(task: Task, completion: str) -> float:
    """Case-sensitive exact match after trimming surrounding whitespace."""
    if not isinstance(completion, str):
        raise TypeError("completion must be a string")
    return float(completion.strip() == task.answer.strip())

_TASK_FIELDS = frozenset({"task_id", "prompt", "answer", "split", "pool"})


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_tasks_jsonl(path: str | Path) -> tuple[Task, ...]:
    """Validate the external JSONL once and return immutable normalized tasks."""
    source = Path(path)
    tasks: list[Task] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank task row at {source}:{line_number}")
            try:
                value = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid task JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"task row must be a JSON object at {source}:{line_number}")
            unknown = set(value) - _TASK_FIELDS
            missing = {"task_id", "prompt", "answer"} - set(value)
            if unknown or missing:
                raise ValueError(
                    f"invalid task fields at {source}:{line_number}: "
                    f"unknown={sorted(unknown)}, missing={sorted(missing)}"
                )
            try:
                task = Task(**value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid task at {source}:{line_number}: {exc}") from exc
            if task.task_id in seen_ids:
                raise ValueError(f"duplicate task_id at {source}:{line_number}: {task.task_id}")
            seen_ids.add(task.task_id)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"task JSONL must contain at least one task: {source}")
    return tuple(tasks)


def normalized_tasks_jsonl(tasks: Iterable[Task]) -> bytes:
    """Serialize normalized tasks canonically for a run's immutable input copy."""
    rows = [asdict(task) for task in tasks]
    if not rows:
        raise ValueError("cannot serialize an empty task set")
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        .encode("utf-8")
        for row in rows
    )


def tasks_sha256(tasks: Iterable[Task]) -> str:
    return hashlib.sha256(normalized_tasks_jsonl(tasks)).hexdigest()
