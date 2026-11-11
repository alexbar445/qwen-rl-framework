"""Export public prompts; hidden solutions stay in the trusted environment service."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .prompt import build_task_messages
from .task_source import sample_tasks


def prepare_dataset(dataset_dir: Path, output_dir: Path, train_tasks: int = 32,
                    validation_tasks: int = 8, seed: int = 20261009,
                    families: list[str] | None = None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if train_tasks < 1 or validation_tasks < 1:
        raise ValueError("train_tasks and validation_tasks must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    train_families = list(families) if families is not None else None
    for name, split, pool, count in (
        ("train", "train", "arc2_train", train_tasks),
        ("validation", "test", "arc2_eval", validation_tasks),
    ):
        if name == "train" and train_families is not None:
            # A curriculum names its families, so train_tasks is an upper bound there.
            count = min(count, len(train_families))
        specs = sample_tasks(split, [pool], count, seed=seed, dataset_dir=dataset_dir,
                             families=train_families if name == "train" else None)
        if len(specs) != count:
            raise ValueError(f"requested {count} {name} tasks, found {len(specs)}")
        rows = [{"data_source": "arc", "prompt": build_task_messages(t),
                 "agent_name": "arc", "ability": "arc",
                 "reward_model": {"style": "rule", "ground_truth": t.task_id},
                 "extra_info": {"task_id": t.task_id, "family": t.family,
                                "validation": name == "validation", "index": i}}
                for i, t in enumerate(specs)]
        with NamedTemporaryFile(dir=output_dir, suffix=".parquet", delete=False) as f:
            temporary = Path(f.name)
        try:
            pq.write_table(pa.Table.from_pylist(rows), temporary)
            os.replace(temporary, output_dir / f"{name}.parquet")
        finally:
            temporary.unlink(missing_ok=True)
        counts[name] = len(rows)
    (output_dir / "manifest.json").write_text(json.dumps(
        {"schema": "arc-public-prompts-v1", "seed": seed, "counts": counts,
         "families": sorted(train_families) if train_families is not None else None,
         "solutions_exported": False}, indent=2) + "\n")
    return counts
