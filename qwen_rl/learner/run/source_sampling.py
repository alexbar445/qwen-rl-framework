"""Independent weighted random source draws with checkpointed seed and cursor."""
from __future__ import annotations

import hashlib
from fractions import Fraction
import json
import random
import secrets
from typing import Any, Sequence

from ...task_catalog import SourceCatalog

SOURCE_ORDER = ("synthetic_arc", "arc2_train")
SCHEMA = "qwen-rl-mixed-source-sampling/v1"


class MixedSourceSampling:
    """Draw a source per group; retain the next draw across managed resumes."""

    def __init__(
        self,
        *,
        catalog: SourceCatalog,
        selected_tasks: Sequence[Any],
        worker_ids: Sequence[str],
        tasks_per_worker: int,
        record: dict[str, Any],
        worker_slots: Sequence[str] | None = None,
    ) -> None:
        catalog_pools = tuple(source.pool for source in catalog.sources)
        if len(catalog_pools) != len(SOURCE_ORDER) or set(catalog_pools) != set(SOURCE_ORDER):
            raise ValueError("mixed source catalog must contain synthetic_arc and arc2_train")
        workers = tuple(worker_ids)
        if not workers or any(not isinstance(worker, str) or not worker for worker in workers):
            raise ValueError("mixed source dispatch requires explicit paired Worker IDs")
        if len(set(workers)) != len(workers):
            raise ValueError("mixed source dispatch Worker IDs must be unique")
        if (isinstance(tasks_per_worker, bool) or not isinstance(tasks_per_worker, int)
                or tasks_per_worker < 1):
            raise ValueError("mixed source dispatch requires a positive tasks_per_worker count")

        weights = {source.pool: Fraction(str(source.weight)) for source in catalog.sources}
        self.generated_share = weights["synthetic_arc"] / sum(weights.values())
        self.catalog = catalog
        self.worker_ids = workers
        self.tasks_per_worker = tasks_per_worker
        self.worker_slots = tuple(worker_slots) if worker_slots is not None else workers * tasks_per_worker
        self.tasks_by_pool: dict[str, list[Any]] = {pool: [] for pool in SOURCE_ORDER}
        for task in selected_tasks:
            pool = getattr(task, "pool", None)
            if pool not in self.tasks_by_pool or getattr(task, "split", None) != "train":
                raise ValueError("mixed source selection contains a task outside the two training pools")
            self.tasks_by_pool[pool].append(task)
        for pool, tasks in self.tasks_by_pool.items():
            task_ids = [getattr(task, "task_id", None) for task in tasks]
            if not tasks or any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
                raise ValueError(f"mixed source {pool} has no selected training tasks")
            if len(set(task_ids)) != len(task_ids):
                raise ValueError(f"mixed source {pool} contains duplicate task IDs")

        catalog_sources = []
        for pool in SOURCE_ORDER:
            source = catalog.source_for(pool)
            if source is None:
                raise ValueError(f"mixed source catalog is missing {pool}")
            catalog_sources.append({
                "pool": pool,
                "path": source.relative_path,
                "weight": source.weight,
                "variants": source.variants,
            })
        task_ids = {
            pool: sorted(task.task_id for task in self.tasks_by_pool[pool])
            for pool in SOURCE_ORDER
        }
        identity_payload = {
            "catalog_schema": getattr(catalog, "schema", "qwen-rl-mixed-task-sources/v1"),
            "sources": catalog_sources,
            "task_ids": task_ids,
        }
        taskset_identity = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        static_state = {
            "schema": SCHEMA,
            "taskset_identity": taskset_identity,
            "source_order": list(SOURCE_ORDER),
            "source_task_counts": {pool: len(task_ids[pool]) for pool in SOURCE_ORDER},
        }
        saved = record.get("source_sampling")
        if saved is None:
            saved = {**static_state, "draw_counts": {pool: 0 for pool in SOURCE_ORDER}}
            record["source_sampling"] = saved
        elif not isinstance(saved, dict) or any(saved.get(key) != value for key, value in static_state.items()):
            raise ValueError("persisted mixed source sampling identity differs from the selected taskset")
        draws = saved.get("draw_counts")
        if (not isinstance(draws, dict) or set(draws) != set(SOURCE_ORDER)
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in draws.values())):
            raise ValueError("persisted mixed source draw counts are invalid")
        # Initialize new random-sampling state without resetting existing counts.
        # The counts are the authoritative cursor, independent of process ordinals
        # and Worker topology. A seed is generated only when first needed.
        if "random_seed" not in saved:
            saved["random_seed"] = secrets.token_hex(16)
        seed = saved["random_seed"]
        if (not isinstance(seed, str) or len(seed) != 32
                or any(c not in "0123456789abcdef" for c in seed)):
            raise ValueError("persisted mixed source random seed is invalid")
        saved["method"] = "independent-weighted-random/v1"
        self.record = record

    def source_for_dispatch(self, offset: int = 0) -> str:
        """Peek at the next independent source draw; note_dispatch consumes it."""
        state = self.record["source_sampling"]
        cursor = sum(state["draw_counts"].values()) + offset
        rng = random.Random(f'mixed-source-random-v1:{state["random_seed"]}:{cursor}')
        return "synthetic_arc" if rng.random() < self.generated_share else "arc2_train"

    def round_sources(self) -> list[str]:
        """Keep each Worker's random draws, dispatching its ARC2 slots first.

        Planning only peeks; note_dispatch still counts actual dispatches.
        Full checkpoints are written after the round has drained.
        """
        sources = [self.source_for_dispatch(offset) for offset in range(len(self.worker_slots))]
        for worker in self.worker_ids:
            indices = [i for i, value in enumerate(self.worker_slots) if value == worker]
            ordered = sorted((sources[i] for i in indices), key=lambda pool: pool != "arc2_train")
            for i, source in zip(indices, ordered):
                sources[i] = source
        return sources

    def tasks_for(self, pool: str) -> list[Any]:
        if pool not in self.tasks_by_pool:
            raise ValueError(f"unsupported mixed source pool: {pool!r}")
        return self.tasks_by_pool[pool]

    def pool_for(self, task: Any) -> str | None:
        pool = getattr(task, "pool", None)
        return pool if pool in self.tasks_by_pool else None

    def note_dispatch(self, pool: str) -> None:
        if pool not in self.tasks_by_pool:
            raise ValueError(f"unsupported mixed source pool: {pool!r}")
        self.record["source_sampling"]["draw_counts"][pool] += 1
