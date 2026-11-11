# Extending the framework

The generic extension point is an `Environment` plus a `RewardScorer`. The environment supplies tasks and complete rollout groups; the scorer assigns one reward per trajectory. The small example below can be imported and constructed without starting a service, loading a model, or using a GPU.

For a bounded real-model task that fits a JSONL dataset and a scalar reward,
start with the [small-model CPU recipe](../examples/small_model/README.md).
That command path accepts the task rows and reward directly; it does not ask
you to build an `Environment`, engine, Worker manager, or pool. It validates
JSONL records once at input, then passes typed `Task` objects to the reward
function. The generic `Environment` example below remains useful when a task
needs custom rollout behavior or application composition.

```python
import asyncio
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qwen_rl.contracts.environment import (
    GroupResult,
    Reward,
    RewardScorer,
    Trajectory,
    score_group,
    validate_environment,
    validate_group_result,
    validate_reward_scorer,
)
from qwen_rl.rollout.groups import PolicySnapshot


@dataclass(frozen=True)
class ChoiceTask:
    task_id: str
    split: str = "train"
    pool: str | None = None
    target_token: int = 3


class ChoiceEnvironment:
    name = "binary-choice"
    prompt_identity = "binary-choice-prompt/v1"
    reward_schema = "target-token/v1"
    metrics_key = "choice"
    source_identity = "binary-choice-source/v1"

    def __init__(self) -> None:
        task = ChoiceTask("choice-1")
        self.tasks = {task.task_id: task}

    def fixture_task(self) -> ChoiceTask:
        return ChoiceTask("fixture", split="diagnostic", pool="diagnostic")

    async def generate_group(
        self,
        engine: Any,
        managers: Mapping[str, object],
        config: Mapping[str, object],
        task: ChoiceTask,
        group_id: str,
        policy: PolicySnapshot,
        *,
        worker_pool: Any,
        diagnostic_course: str | None = None,
        owner_worker_id: str | None = None,
        guided_plan: str | None = None,
    ) -> GroupResult:
        del engine, managers, config, task, worker_pool
        del diagnostic_course, owner_worker_id, guided_plan
        trajectories: list[Trajectory] = []
        for answer_token in (2, 3):
            trajectories.append({
                "input_ids": [1, answer_token],
                "policy_mask": [0, 1],
                "behavior_logprobs": [0.0, math.log(0.5)],
                "group_id": group_id,
                "policy_version": policy.version,
            })
        return {
            "trajectories": trajectories,
            "worker_ids": ["toy-worker", "toy-worker"],
            "policy_version": policy.version,
        }


class TargetTokenScorer(RewardScorer[ChoiceTask]):
    def score(
        self,
        task: ChoiceTask,
        trajectories: Sequence[Trajectory],
    ) -> Sequence[Reward]:
        return [
            Reward(
                value=float(row["input_ids"][-1] == task.target_token),
                components={"target_match": row["input_ids"][-1] == task.target_token},
            )
            for row in trajectories
        ]


async def check_contracts() -> list[float]:
    environment = validate_environment(ChoiceEnvironment())
    scorer = validate_reward_scorer(TargetTokenScorer())
    task = environment.tasks["choice-1"]
    policy = PolicySnapshot(0, "a" * 64, "toy-base", "toy-artifact")
    raw_group = await environment.generate_group(
        None, {}, {}, task, "group-1", policy, worker_pool=None,
    )
    group = validate_group_result(raw_group, policy_version=policy.version)
    return score_group(scorer, task, group)["rewards"]


print(asyncio.run(check_contracts()))  # [0.0, 1.0]
```

The snippet constructs the typed adapter views, generates one in-memory group, checks it at the group boundary, and scores it. It does not run a Learner or start a Worker. The fixed log-probabilities are contract-test fixtures only. A real adapter must preserve the actual generated token IDs, masks and sampled behavior log-probabilities returned by its Worker.

## Contract requirements

- Each task has a non-empty `task_id` matching its key in `environment.tasks`, a non-empty `split`, and an optional string `pool`. `fixture_task()` returns a task with the same metadata shape.
- `name`, `reward_schema`, `metrics_key`, and `source_identity` are non-empty strings. The metrics key cannot be `learner`, `workers`, or `timings`. `prompt_identity` is a stable value or callable.
- `generate_group()` returns at least one trajectory and one non-empty Worker ID per trajectory. All rows share the group's requested policy version and `group_id`. Each row has aligned, non-empty `input_ids`, `policy_mask`, and `behavior_logprobs`, plus `group_id` and `policy_version`.
- `RewardScorer.score(task, trajectories)` returns exactly one `Reward` for each trajectory. Reward values must be finite numbers; optional components are a mapping for diagnostics.

Keep `source_identity` stable for unchanged task and rule semantics, and change it when those semantics change. Recovery rejects a missing or different environment name/source identity so an old checkpoint cannot silently resume against a changed adapter. This string is supplied by the adapter owner; it is a semantic version label, not an automatically computed content hash.

## Use in an application

The application factory is `create_learner_app` in `qwen_rl.learner.app`. Pass your prepared `environment=` and `reward_scorer=` when constructing the application; the factory also needs the application's config and, for a controlled local setup, its chosen engine and Worker managers. This is an application integration point, not a standalone command. When a custom environment is supplied, a scorer is required; `arc_app` cannot be combined with that path. BEPA and held-out evaluation currently require the ARC adapter and are rejected for custom environments. For the simpler JSONL-based real-model flow, use `qwen-rl train` and the [small-model recipe](../examples/small_model/README.md) instead of constructing this application factory.

If both arguments are omitted, the factory selects the existing ARC environment and scorer. See [Architecture](ARCHITECTURE.md) for the shared path and [Support](SUPPORT.md) for the evidence boundary.
