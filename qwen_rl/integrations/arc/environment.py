"""Adapter from the existing ARC rollout path to generic learner contracts."""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...contracts.environment import GroupMetrics, GroupResult, Reward, Task, Trajectory


GroupGenerator = Callable[..., Awaitable[GroupResult]]


@dataclass(frozen=True)
class ArcEnvironment:
    """Keep ARC's established source, rollout generator, and fixture behind an adapter."""

    tasks: Mapping[str, Task]
    fixture: Task
    generator: GroupGenerator
    prompt_identity: str
    reward_schema: str
    source_identity: str = "arc-adapter/v1"
    metrics_key: str = "arc"
    name: str = "arc"

    def fixture_task(self) -> Task:
        return self.fixture

    async def generate_group(self, *args: Any, **kwargs: Any) -> GroupResult:
        result = await self.generator(*args, **kwargs)
        counters = result.get("counters", {})
        outputs = [*result.get("outputs", ())]
        values = {
            "agent_service_http_seconds": sum(
                float(getattr(output.metrics, "compute_score", 0.0))
                for output in outputs
            )
        }
        if result.get("guided_output") is not None:
            values["agent_service_http_seconds"] += float(
                getattr(result["guided_output"].metrics, "compute_score", 0.0)
            )
        metrics: GroupMetrics = {
            "episode_count": len(outputs) + int(result.get("guided_output") is not None),
            "worker_generation": {
                worker_id: {
                    "generation_calls": int(count.get("calls", 0)),
                    "generation_seconds": float(count.get("seconds", 0.0)),
                }
                for worker_id, count in counters.items()
            },
            "values": values,
        }
        result["metrics"] = metrics
        return result


class ArcRewardScorer:
    """Carry through rewards already produced by the trusted ARC service."""

    def score(self, task: Task, trajectories: Sequence[Trajectory]) -> list[Reward]:
        del task
        rewards: list[Reward] = []
        for trajectory in trajectories:
            value = trajectory.get("reward")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("ARC service trajectory omitted its numeric reward")
            components = trajectory.get("reward_components", {})
            if not isinstance(components, Mapping):
                raise ValueError("ARC service trajectory reward_components must be a mapping")
            rewards.append(Reward(float(value), components))
        return rewards
