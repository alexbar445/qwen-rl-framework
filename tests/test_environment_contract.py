from __future__ import annotations

import math
import subprocess
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from qwen_rl.contracts.environment import (
    Reward,
    score_group,
    validate_environment,
    validate_group_result,
    validate_recovery_source_identity,
    validate_reward_scorer,
)
from qwen_rl.integrations.arc.environment import ArcEnvironment, ArcRewardScorer


@dataclass(frozen=True)
class ToyTask:
    task_id: str
    split: str = "train"
    pool: str | None = None


def _group(policy_version: int = 0):
    return {
        "trajectories": [{
            "input_ids": [1, 2],
            "policy_mask": [0, 1],
            "behavior_logprobs": [0.0, -0.5],
            "group_id": "g1",
            "policy_version": policy_version,
        }],
        "worker_ids": ["worker-1"],
        "policy_version": policy_version,
    }


class EnvironmentContractTests(unittest.TestCase):
    def test_contract_and_arc_adapter_import_without_arc_runtime_or_torch(self):
        code = (
            "import sys; import qwen_rl.contracts.environment; "
            "import qwen_rl.integrations.arc.environment; "
            "import qwen_rl.learner.app; "
            "blocked = {'torch', 'transformers', 'qwen_rl.environment', "
            "'qwen_rl.service', 'qwen_rl.prompt', 'qwen_rl.task_source'}; "
            "assert not blocked.intersection(sys.modules), "
            "blocked.intersection(sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_generic_adapter_is_validated_once_and_exposes_immutable_tasks(self):
        task = ToyTask("task-1")
        fixture = ToyTask("fixture", split="diagnostic", pool="diagnostic")

        class Environment:
            tasks = {task.task_id: task}
            name = "toy"
            prompt_identity = "toy-prompt/v1"
            reward_schema = "toy-reward/v1"
            metrics_key = "toy"
            source_identity = "toy-data/v1"

            def fixture_task(self):
                return fixture

            async def generate_group(self, *_args, **_kwargs):
                return _group()

        environment = validate_environment(Environment())
        self.assertEqual(environment.name, "toy")
        self.assertEqual(environment.fixture_task(), fixture)
        validate_recovery_source_identity(
            {"environment": "toy", "adapter_source_identity": "toy-data/v1"},
            environment,
        )
        for stored in (
            {"environment": "toy"},
            {"environment": "toy", "adapter_source_identity": "toy-data/v2"},
        ):
            with self.assertRaisesRegex(ValueError, "source identity"):
                validate_recovery_source_identity(stored, environment)
        with self.assertRaises(TypeError):
            environment.tasks["task-2"] = ToyTask("task-2")

        with self.assertRaises(TypeError):
            validate_environment(SimpleNamespace(
                tasks={"wrong-key": task}, fixture_task=lambda: fixture,
                generate_group=lambda: None, name="toy", prompt_identity="v1",
                reward_schema="v1", metrics_key="toy", source_identity="toy/v1",
            ))

        for reserved_key in ("learner", "workers", "timings"):
            candidate = Environment()
            candidate.metrics_key = reserved_key
            with self.assertRaisesRegex(ValueError, "metrics_key is reserved"):
                validate_environment(candidate)

    def test_incomplete_group_is_rejected_at_environment_boundary(self):
        with self.assertRaisesRegex(ValueError, "one Worker per trajectory"):
            validate_group_result({**_group(), "worker_ids": []}, policy_version=0)
        bad_trajectory = _group()
        del bad_trajectory["trajectories"][0]["behavior_logprobs"]
        with self.assertRaisesRegex(ValueError, "missing required learner fields"):
            validate_group_result(bad_trajectory, policy_version=0)
        with self.assertRaisesRegex(ValueError, "policy version"):
            validate_group_result(_group(policy_version=1), policy_version=0)

    def test_reward_scorer_length_and_finite_value_are_checked_before_admission(self):
        task = ToyTask("task-1")
        group = validate_group_result(_group(), policy_version=0)

        class WrongLengthScorer:
            def score(self, _task, _trajectories):
                return []

        with self.assertRaisesRegex(ValueError, "one reward per trajectory"):
            score_group(validate_reward_scorer(WrongLengthScorer()), task, group)

        class NonFiniteScorer:
            def score(self, _task, _trajectories):
                return [Reward(math.nan)]

        with self.assertRaisesRegex(ValueError, "non-finite"):
            score_group(validate_reward_scorer(NonFiniteScorer()), task, group)

    def test_arc_adapter_preserves_existing_rewards_and_adds_generic_metrics(self):
        task = ToyTask("arc-task")
        row = {**_group()["trajectories"][0], "reward": 1.25,
               "reward_components": {"exact": 1.0}}
        output = SimpleNamespace(metrics=SimpleNamespace(compute_score=0.25))

        async def generate(*_args, **_kwargs):
            return {
                **_group(),
                "trajectories": [row],
                "outputs": [output],
                "worker_ids": ["worker-1"],
                "counters": {"worker-1": {"calls": 1, "seconds": 0.5}},
                "rewards": [1.25],
                "exact": [True],
                "costs": [3.0],
            }

        import asyncio

        adapter = ArcEnvironment(
            tasks={task.task_id: task},
            fixture=ToyTask("fixture", split="diagnostic", pool="diagnostic"),
            generator=generate,
            prompt_identity={"fingerprint": "arc-prompt"},
            reward_schema="arc-reward/v13",
        )
        result = asyncio.run(adapter.generate_group(None, {}, {}, task, "g1", None,
                                                    worker_pool=None))
        score = ArcRewardScorer().score(task, result["trajectories"])

        self.assertEqual(result["trajectories"][0]["reward"], 1.25)
        self.assertEqual(result["exact"], [True])
        self.assertEqual(result["costs"], [3.0])
        self.assertEqual(score, [Reward(1.25, {"exact": 1.0})])
        self.assertEqual(result["metrics"]["values"]["agent_service_http_seconds"], 0.25)
        self.assertEqual(result["metrics"]["worker_generation"]["worker-1"]["generation_calls"], 1)


if __name__ == "__main__":
    unittest.main()
