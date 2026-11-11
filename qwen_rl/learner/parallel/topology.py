"""Topology parsing and deterministic, no-duplicate data-parallel row layout."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


_MODES = frozenset({"single", "tp", "ddp", "fsdp2", "pp"})


@dataclass(frozen=True)
class ParallelTopology:
    mode: str
    world_size: int
    devices: tuple[int, ...]
    compute_timeout: float = 3600.0
    cuda_memory_fraction: float = 1.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ParallelTopology":
        if not isinstance(config, Mapping):
            raise TypeError("parallel config must be a mapping")
        mode = config.get("parallelism")
        if mode not in _MODES:
            raise ValueError(f"parallel mode must be one of {sorted(_MODES)}, got {mode!r}")
        raw_devices = config.get("gpus")
        if not isinstance(raw_devices, (list, tuple)):
            raise ValueError("gpus must be a non-empty list or tuple of CUDA indices")
        if any(type(value) is not int for value in raw_devices):
            raise ValueError("gpus must contain non-negative integer CUDA indices")
        devices = tuple(raw_devices)
        if not devices or any(value < 0 for value in devices) or len(set(devices)) != len(devices):
            raise ValueError("devices must be a non-empty list of unique non-negative CUDA indices")
        world_size = len(devices)
        if mode == "single" and world_size != 1:
            raise ValueError("single mode requires exactly one device")
        if mode in {"tp", "fsdp2", "pp"} and world_size < 2:
            raise ValueError(f"{mode} mode requires at least two devices")
        if mode == "pp" and world_size != 4:
            raise ValueError("pp mode requires exactly four pipeline devices")
        if mode == "pp":
            if config.get("model_backend") != "ternary":
                raise ValueError("pp mode is currently supported only by the verified ternary model backend")
            if config.get("pipeline_stages") != 4:
                raise ValueError("ternary pp mode requires pipeline_stages=4")
        timeout = config.get("compute_timeout", 3600.0)
        fraction = config.get("cuda_memory_fraction", 1.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("compute_timeout must be finite and positive")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("cuda_memory_fraction must be in (0, 1]")
        return cls(mode, world_size, devices, float(timeout), float(fraction))

    @property
    def is_data_parallel(self) -> bool:
        return self.mode in {"ddp", "fsdp2"}

    def signature(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "world_size": self.world_size,
            "devices": list(self.devices),
            "compute_timeout": self.compute_timeout,
            "cuda_memory_fraction": self.cuda_memory_fraction,
        }


def _row_work(row: Mapping[str, Any]) -> int:
    ids = row.get("input_ids")
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError("every real trajectory must have a non-empty input_ids sequence")
    return len(ids)


def balance_rows_for_data_parallel(
    rows: list[dict[str, Any]],
    *,
    world_size: int,
    microbatch_size: int,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Return a rank-chunked global list plus rank-local rows padded with dummies.

    Longest rows are greedily assigned to the least-loaded rank with remaining
    capacity, then restored to their original within-rank order. The capacity
    is the smallest microbatch multiple that can hold all rows across the
    world. Padding rows are explicit zero-credit records and never copy a
    trajectory.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty trajectory list")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("data parallel world_size must be positive")
    if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or microbatch_size < 1:
        raise ValueError("microbatch_size must be a positive integer")
    assignments: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    local_capacity = max(
        microbatch_size,
        math.ceil(len(rows) / (world_size * microbatch_size)) * microbatch_size,
    )
    order = sorted(enumerate(rows), key=lambda item: (-_row_work(item[1]), item[0]))
    for source_index, row in order:
        available = [rank for rank in range(world_size) if len(assignments[rank]) < local_capacity]
        if not available:
            raise RuntimeError("data-parallel row capacity was exhausted before assigning all trajectories")
        rank = min(available, key=lambda value: (loads[value], len(assignments[value]), value))
        copied = dict(row)
        copied["_parallel_source_index"] = source_index
        copied["_parallel_row_id"] = source_index
        assignments[rank].append((source_index, copied))
        loads[rank] += _row_work(row)
    for rank_rows in assignments:
        rank_rows.sort(key=lambda item: item[0])

    rank_chunks: list[list[dict[str, Any]]] = []
    for rank, rank_rows in enumerate(assignments):
        chunk = [row for _, row in rank_rows]
        while len(chunk) < local_capacity:
            dummy_id = f"zero-credit-{rank}-{len(chunk)}"
            chunk.append({
                "input_ids": (0, 0),
                "policy_mask": (0, 1),
                "reward": 0.0,
                "group_id": dummy_id,
                "row_kind": "zero_credit",
                "_parallel_dummy": True,
                "_parallel_advantage": 0.0,
                "_parallel_row_id": dummy_id,
            })
        rank_chunks.append(chunk)
    return [row for chunk in rank_chunks for row in chunk], rank_chunks
