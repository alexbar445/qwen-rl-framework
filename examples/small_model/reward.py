"""Small custom reward example for the JSONL task contract."""

from qwen_rl.training.tasks import Task


def reward(task: Task, completion: str) -> float:
    """Return 1 for the expected yes/no label and 0 otherwise."""
    return float(completion.strip() == task.answer.strip())
