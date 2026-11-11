"""Small, domain-neutral contracts for rollout tasks and rewards.

The contracts contain only standard-library types.  ARC integrations, vendored
toolkits, model runtimes, and service clients belong in adapters at the boundary.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, TypeVar, cast

if TYPE_CHECKING:
    from ..rollout.groups import PolicySnapshot
    from ..worker.placement import EpisodeWorkerPool


class Task(Protocol):
    """The task metadata required by scheduling and recovery."""

    task_id: str
    split: str
    pool: str | None


TaskT = TypeVar("TaskT", bound=Task)
_RESERVED_METRICS_KEYS = frozenset({"learner", "workers", "timings"})


class Trajectory(TypedDict):
    """The policy-learning fields shared by every environment adapter."""

    input_ids: list[int]
    policy_mask: list[int]
    behavior_logprobs: list[float]
    group_id: str
    policy_version: int
    reward: NotRequired[float]
    reward_components: NotRequired[dict[str, object]]


class GroupResult(TypedDict):
    """A complete environment-produced group before learner admission."""

    trajectories: list[Trajectory]
    worker_ids: list[str]
    policy_version: int
    metrics: NotRequired[GroupMetrics]
    rewards: NotRequired[list[float]]
    outputs: NotRequired[list[object]]


class WorkerGenerationMetrics(TypedDict, total=False):
    generation_calls: int
    generation_seconds: float


class GroupMetrics(TypedDict, total=False):
    """Environment measurements; the coordinator accumulates values by metric key."""

    episode_count: int
    worker_generation: Mapping[str, WorkerGenerationMetrics]
    values: Mapping[str, float]
    annotations: Mapping[str, object]


@dataclass(frozen=True)
class Reward:
    """One scalar policy-learning reward plus optional domain diagnostics."""

    value: float
    components: Mapping[str, object] = field(default_factory=dict)


class Environment(Protocol[TaskT]):
    """Provides tasks and complete rollout groups to the shared coordinator."""

    tasks: Mapping[str, TaskT]
    name: str
    prompt_identity: object
    reward_schema: str
    metrics_key: str
    source_identity: str

    def fixture_task(self) -> TaskT: ...

    async def generate_group(
        self,
        engine: object,
        managers: Mapping[str, object],
        config: Mapping[str, object],
        task: TaskT,
        group_id: str,
        policy: PolicySnapshot,
        *,
        worker_pool: EpisodeWorkerPool,
        diagnostic_course: str | None = None,
        owner_worker_id: str | None = None,
        guided_plan: str | None = None,
    ) -> GroupResult: ...


class RewardScorer(Protocol[TaskT]):
    """Scores the episodes in a complete group, preserving group-level rewards."""

    def score(
        self,
        task: TaskT,
        trajectories: Sequence[Trajectory],
    ) -> Sequence[Reward]: ...


@dataclass(frozen=True)
class ValidatedEnvironment:
    """Typed, immutable view created once at the external adapter boundary."""

    adapter: Environment[Task]
    tasks: Mapping[str, Task]
    fixture: Task
    name: str
    prompt_identity: object
    reward_schema: str
    metrics_key: str
    source_identity: str

    def fixture_task(self) -> Task:
        return self.fixture

    async def generate_group(self, *args: object, **kwargs: object) -> GroupResult:
        return await self.adapter.generate_group(*args, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ValidatedRewardScorer:
    """Typed scorer view created once at the external adapter boundary."""

    adapter: RewardScorer[Task]

    def score(self, task: Task, trajectories: Sequence[Trajectory]) -> Sequence[Reward]:
        return self.adapter.score(task, trajectories)


def validate_environment(environment: object) -> ValidatedEnvironment:
    """Validate an adapter once and return the typed view used by core code."""
    if isinstance(environment, ValidatedEnvironment):
        return environment
    tasks = getattr(environment, "tasks", None)
    if not isinstance(tasks, Mapping):
        raise TypeError("environment.tasks must be a mapping")
    checked_tasks: dict[str, Task] = {}
    for key, task in tasks.items():
        task_id = getattr(task, "task_id", None)
        split = getattr(task, "split", None)
        pool = getattr(task, "pool", None)
        if (not isinstance(key, str) or not key or task_id != key
                or not isinstance(split, str) or not split
                or pool is not None and not isinstance(pool, str)):
            raise TypeError("environment tasks require matching task_id, non-empty split, and string pool metadata")
        checked_tasks[key] = cast(Task, task)
    fixture_factory = getattr(environment, "fixture_task", None)
    generator = getattr(environment, "generate_group", None)
    name = getattr(environment, "name", None)
    prompt_identity = getattr(environment, "prompt_identity", None)
    reward_schema = getattr(environment, "reward_schema", None)
    metrics_key = getattr(environment, "metrics_key", None)
    source_identity = getattr(environment, "source_identity", None)
    if not callable(fixture_factory) or not callable(generator):
        raise TypeError("environment must provide fixture_task() and generate_group()")
    if any(not isinstance(value, str) or not value for value in (name, reward_schema, metrics_key)):
        raise TypeError("environment name, reward_schema, and metrics_key must be non-empty strings")
    if metrics_key in _RESERVED_METRICS_KEYS:
        raise ValueError(f"environment metrics_key is reserved by the Learner: {metrics_key}")
    if prompt_identity is None:
        raise TypeError("environment prompt_identity must be a stable value or callable")
    if not isinstance(source_identity, str) or not source_identity.strip():
        raise TypeError("environment source_identity must be a stable non-empty string")
    fixture = fixture_factory()
    fixture_id = getattr(fixture, "task_id", None)
    fixture_split = getattr(fixture, "split", None)
    fixture_pool = getattr(fixture, "pool", None)
    if (not isinstance(fixture_id, str) or not fixture_id
            or not isinstance(fixture_split, str) or not fixture_split
            or fixture_pool is not None and not isinstance(fixture_pool, str)):
        raise TypeError("environment fixture task has invalid task metadata")
    return ValidatedEnvironment(
        adapter=cast(Environment[Task], environment),
        tasks=MappingProxyType(checked_tasks),
        fixture=cast(Task, fixture),
        name=name,
        prompt_identity=prompt_identity,
        reward_schema=reward_schema,
        metrics_key=metrics_key,
        source_identity=source_identity,
    )


def validate_reward_scorer(scorer: object) -> RewardScorer[Task]:
    """Validate the scorer entrypoint once before the run starts."""
    if isinstance(scorer, ValidatedRewardScorer):
        return scorer
    score = getattr(scorer, "score", None)
    if not callable(score):
        raise TypeError("reward_scorer must provide score(task, trajectories)")
    return ValidatedRewardScorer(cast(RewardScorer[Task], scorer))


def validate_recovery_source_identity(
    stored: object,
    environment: ValidatedEnvironment,
) -> None:
    """Fail closed when a checkpoint names a different or missing adapter source."""
    if (not isinstance(stored, Mapping)
            or stored.get("environment") != environment.name
            or stored.get("adapter_source_identity") != environment.source_identity):
        raise ValueError("recovery source identity does not match the active environment")


def validate_group_result(result: object, *, policy_version: int) -> GroupResult:
    """Convert one environment response into the internal group contract."""
    if not isinstance(result, dict):
        raise TypeError("environment.generate_group must return a mapping")
    trajectories = result.get("trajectories")
    worker_ids = result.get("worker_ids")
    version = result.get("policy_version")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("environment group must contain at least one trajectory")
    if (not isinstance(worker_ids, list) or len(worker_ids) != len(trajectories)
            or any(not isinstance(worker_id, str) or not worker_id for worker_id in worker_ids)):
        raise ValueError("environment group must identify one Worker per trajectory")
    if type(version) is not int or version != policy_version:
        raise ValueError("environment group policy version differs from its requested snapshot")
    fields = {"input_ids", "policy_mask", "behavior_logprobs", "group_id", "policy_version"}
    for trajectory in trajectories:
        if not isinstance(trajectory, dict) or not fields.issubset(trajectory):
            raise ValueError("environment trajectory is missing required learner fields")
        input_ids = trajectory["input_ids"]
        policy_mask = trajectory["policy_mask"]
        behavior_logprobs = trajectory["behavior_logprobs"]
        if (not isinstance(input_ids, list) or not isinstance(policy_mask, list)
                or not isinstance(behavior_logprobs, list)
                or not input_ids or len(input_ids) != len(policy_mask)
                or len(input_ids) != len(behavior_logprobs)):
            raise ValueError("environment trajectory token arrays are incomplete or misaligned")
        if (any(type(token) is not int or token < 0 for token in input_ids)
                or any(type(active) is not int or active not in (0, 1) for active in policy_mask)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in behavior_logprobs)):
            raise ValueError("environment trajectory token arrays contain invalid values")
        if (not isinstance(trajectory["group_id"], str) or not trajectory["group_id"]
                or type(trajectory["policy_version"]) is not int
                or trajectory["policy_version"] != version):
            raise ValueError("environment trajectory identity differs from its group")
    metrics = result.get("metrics")
    if metrics is not None and not isinstance(metrics, Mapping):
        raise TypeError("environment group metrics must be a mapping")
    return cast(GroupResult, result)


def score_group(
    scorer: RewardScorer[Task],
    task: Task,
    group: GroupResult,
) -> GroupResult:
    """Score a complete group once and attach the normalized learner rewards."""
    scores = scorer.score(task, group["trajectories"])
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        raise TypeError("reward scorer must return a sequence of Reward values")
    if len(scores) != len(group["trajectories"]):
        raise ValueError("reward scorer must return one reward per trajectory")
    rewards: list[float] = []
    for trajectory, score in zip(group["trajectories"], scores, strict=True):
        if not isinstance(score, Reward):
            raise TypeError("reward scorer entries must be Reward values")
        value = score.value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("reward scorer returned a non-finite numeric reward")
        if not isinstance(score.components, Mapping):
            raise TypeError("reward components must be a mapping")
        reward = float(value)
        trajectory["reward"] = reward
        if score.components:
            trajectory["reward_components"] = dict(score.components)
        rewards.append(reward)
    group["rewards"] = rewards
    return group
