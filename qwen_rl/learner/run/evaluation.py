"""Learner-owned native evaluation orchestration.

The coordinator owns lazy runtime initialization and the policy-reference fence.
HTTP status codes and request authentication remain in ``learner.app``.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from ...evaluation.policy import EvaluationPolicy


class EvaluationCoordinator:
    """Create, observe, and close the evaluation runtime for one Learner app."""

    def __init__(
        self,
        *,
        settings: dict[str, Any],
        dataset: Any | None,
        engine: Any,
        policies: dict[int, dict[str, Any]],
        policy_refs: dict[int, int],
        current_policy: Callable[[], dict[str, Any]],
        recovery_metadata: Callable[[], dict[str, Any] | None],
        assert_workers_current: Callable[[int], Any],
        run_episode: Callable[..., Any],
        on_initialized: Callable[[Any], None] | None = None,
        runtime_factory: Callable[..., Any] | None = None,
        evaluation_scope: Callable[[EvaluationPolicy], AbstractAsyncContextManager[None]] | None = None,
    ) -> None:
        self.settings = settings
        self.dataset = dataset
        self.engine = engine
        self.policies = policies
        self.policy_refs = policy_refs
        self.current_policy = current_policy
        self.recovery_metadata = recovery_metadata
        self.assert_workers_current = assert_workers_current
        self.run_episode = run_episode
        self.on_initialized = on_initialized
        self.runtime_factory = runtime_factory
        self.evaluation_scope = evaluation_scope
        self.runtime: Any | None = None
        self._init_lock = asyncio.Lock()

    @classmethod
    def load_dataset(cls, config: dict[str, Any]) -> Any | None:
        """Load the frozen evaluation set only when the feature is enabled."""
        settings = config.get("evaluation", {})
        if not settings.get("enabled"):
            return None
        from ...evaluation.dataset import load_frozen_evaluation_dataset

        manifest = json.loads(Path(settings["manifest_path"]).read_text())
        return load_frozen_evaluation_dataset(
            Path(config["dataset_dir"]),
            manifest_path=Path(settings["manifest_path"]),
            sft_proof_path=Path(settings["sft_exclusion_proof"]),
            teacher_db_path=Path(config["bepa"]["state_dir"]) / "state.sqlite3",
            base_fingerprint=manifest["base_fingerprint"],
        )

    @staticmethod
    def validate_task_separation(training_tasks: Iterable[Any], dataset: Any | None) -> None:
        if dataset is None:
            return
        from ...evaluation.dataset import canonical_json_sha256

        eval_ids = {task.task_id for task in dataset.tasks}
        eval_families = {
            family for task in dataset.tasks
            if getattr(task, "pool", None) in {"arc2_train", "arc2_eval"}
            and isinstance((family := getattr(task, "family", None)), str) and family
        }
        eval_challenges = {
            canonical_json_sha256(task.challenge) for task in dataset.tasks
            if isinstance(getattr(task, "challenge", None), dict)
        }
        train_ids: set[str] = set()
        train_families: set[str] = set()
        train_challenges: set[str] = set()
        for task in training_tasks:
            # The mounted ARC service also contains test tasks so evaluation episodes can
            # resolve their IDs. Only train-split entries are training exposure.
            if getattr(task, "split", "train") != "train":
                continue
            train_ids.add(task.task_id)
            family = getattr(task, "family", None)
            # ARC families identify an original puzzle across variants. Generated
            # families identify shared templates, including legitimate held-out tasks.
            if (getattr(task, "pool", None) in {"arc2_train", "arc2_eval"}
                    and isinstance(family, str) and family):
                train_families.add(family)
            challenge = getattr(task, "challenge", None)
            if isinstance(challenge, dict):
                train_challenges.add(canonical_json_sha256(challenge))
        if (eval_ids.intersection(train_ids)
                or eval_families.intersection(train_families)
                or eval_challenges.intersection(train_challenges)):
            raise ValueError(
                "evaluation task overlaps the training source by task ID, family, or public challenge"
            )

    @property
    def enabled(self) -> bool:
        return bool(self.settings.get("enabled"))

    @property
    def count(self) -> int | None:
        """Successful training count when the evaluation runtime is initialized."""
        return self.runtime.count if self.runtime is not None else None

    @property
    def pause_training(self) -> bool:
        return self.enabled and self.settings.get("pause_training") is True

    def has_pending_requests(self) -> bool:
        return self.runtime is not None and self.runtime.has_pending_requests()

    async def drain_pending_at_boundary(self) -> None:
        if not self.pause_training:
            return
        runtime = await self.ensure_ready()
        if runtime is not None:
            await runtime.run_pending_at_boundary()

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "phase": "not_initialized"}
        if self.runtime is None:
            return {"enabled": True, "phase": "not_initialized"}
        return self.runtime.monitor()

    async def ensure_ready(self) -> Any | None:
        if not self.enabled:
            return None
        async with self._init_lock:
            if self.runtime is not None:
                return self.runtime

            await self.assert_workers_current(int(self.engine.policy_version))
            metadata = self.recovery_metadata()
            if metadata is None:
                raise RuntimeError("evaluation requires a verified full training lineage")
            current = self.current_policy()
            if current["base_fingerprint"] != self.dataset.manifest["base_fingerprint"]:
                raise RuntimeError("evaluation taskset differs from the loaded model base")

            def pin(publication: dict[str, Any]) -> None:
                version = publication["version"]
                existing = self.policies.get(version)
                if existing is not None and existing["files"] != publication["files"]:
                    raise RuntimeError("evaluation policy identity conflicts with a training publication")
                if existing is not None:
                    from ...rollout.groups import PolicySnapshot
                    if (PolicySnapshot.from_policy(existing) != PolicySnapshot.from_policy(publication)
                            or existing.get("artifact_id") != publication.get("artifact_id")):
                        raise RuntimeError("evaluation policy snapshot conflicts with a training publication")
                else:
                    self.policies[version] = publication
                self.policy_refs[version] = self.policy_refs.get(version, 0) + 1

            def unpin(version: int) -> None:
                remaining = self.policy_refs.get(version, 0) - 1
                if remaining < 0:
                    raise RuntimeError("evaluation policy reference underflow")
                if remaining:
                    self.policy_refs[version] = remaining
                else:
                    self.policy_refs.pop(version, None)

            if self.runtime_factory is None:
                from ...evaluation.runtime import EvaluationRuntime

                runtime_factory = EvaluationRuntime
            else:
                runtime_factory = self.runtime_factory
            runtime = runtime_factory(
                settings=self.settings,
                dataset=self.dataset,
                recovery=metadata,
                current_policy=self.current_policy,
                publish_pin=pin,
                release_pin=unpin,
                run_episode=self.run_episode,
                evaluation_scope=self.evaluation_scope,
            )
            try:
                await runtime.initialize()
            except BaseException:
                await runtime.close()
                raise
            self.runtime = runtime
            if self.on_initialized is not None:
                self.on_initialized(runtime)
            return runtime

    async def request_current(self) -> Any:
        if not self.enabled:
            raise RuntimeError("native evaluation is disabled in this deployment")
        runtime = await self.ensure_ready()
        return runtime.request_current()

    async def retry_errors(self, request_id: str) -> Any:
        if not self.enabled:
            raise RuntimeError("native evaluation is disabled in this deployment")
        runtime = await self.ensure_ready()
        return runtime.retry_errors(request_id)

    def observe_commit(self, publication: dict[str, Any] | None) -> None:
        if self.runtime is not None:
            self.runtime.observe_commit(publication)

    async def close(self) -> None:
        if self.runtime is not None:
            await self.runtime.close()
