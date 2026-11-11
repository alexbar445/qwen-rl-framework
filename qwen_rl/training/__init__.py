"""Public API for small, local Hugging Face training."""

from .tasks import RewardFn, Task, exact_match_reward

__all__ = ["RewardFn", "Task", "exact_match_reward", "train"]


def __getattr__(name: str):
    if name == "train":
        from .runner import train

        return train
    raise AttributeError(name)
