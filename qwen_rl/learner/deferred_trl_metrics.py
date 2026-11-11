"""Window-batched, semantics-preserving collection of pinned TRL DAPO metrics."""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch


_REDUCE_CALLS_PER_MICROBATCH = 4
_GATHER_CALLS_PER_MICROBATCH = 2
_METRIC_KEYS = (
    "entropy",
    "clip_ratio/low_mean",
    "clip_ratio/high_mean",
    "clip_ratio/region_mean",
    "clip_ratio/low_min",
    "clip_ratio/high_max",
)
_PATCH_LOCK = threading.RLock()


def _nan_extreme(values: torch.Tensor, *, minimum: bool, dim: int) -> torch.Tensor:
    """Vectorized nanmin/nanmax with TRL's all-NaN result and no device sync."""
    nan = torch.isnan(values)
    fill = float("inf") if minimum else float("-inf")
    reduced = values.masked_fill(nan, fill).amin(dim=dim) if minimum else values.masked_fill(nan, fill).amax(dim=dim)
    has_value = (~nan).any(dim=dim)
    return torch.where(has_value, reduced, torch.full_like(reduced, float("nan")))


class DeferredDapoMetricWindow:
    """Hold per-microbatch DAPO metric sufficient statistics until one gather."""

    def __init__(self, *, expected_microbatches: int, rows_per_microbatch: int):
        if type(expected_microbatches) is not int or expected_microbatches < 1:
            raise ValueError("expected_microbatches must be a positive integer")
        if type(rows_per_microbatch) is not int or rows_per_microbatch < 1:
            raise ValueError("rows_per_microbatch must be a positive integer")
        self.expected_microbatches = expected_microbatches
        self.rows_per_microbatch = rows_per_microbatch
        self._batches: list[tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]] = []

    def add_microbatch(
        self,
        reduce_inputs: Sequence[torch.Tensor],
        gather_inputs: Sequence[torch.Tensor],
    ) -> None:
        if len(reduce_inputs) != _REDUCE_CALLS_PER_MICROBATCH:
            raise RuntimeError(
                "pinned TRL DAPO metric contract expected four [sum,count] reductions per microbatch, "
                f"got {len(reduce_inputs)}"
            )
        if len(gather_inputs) != _GATHER_CALLS_PER_MICROBATCH:
            raise RuntimeError(
                "pinned TRL DAPO metric contract expected low-min and high-max gathers per microbatch, "
                f"got {len(gather_inputs)}"
            )
        reduced = tuple(value.detach() for value in reduce_inputs)
        if any(tuple(value.shape) != (2,) for value in reduced):
            raise RuntimeError("deferred DAPO metric reductions must each have [sum,count] shape")
        gathered = tuple(value.detach() for value in gather_inputs)
        if any(value.ndim != 1 or value.numel() != self.rows_per_microbatch for value in gathered):
            raise RuntimeError(
                "deferred DAPO clip extrema require one row statistic per local microbatch row"
            )
        devices = {value.device for value in (*reduced, *gathered)}
        if len(devices) != 1:
            raise RuntimeError("deferred DAPO metric tensors must share one device")
        self._batches.append((reduced, gathered[0], gathered[1]))

    def pack(self) -> torch.Tensor:
        if len(self._batches) != self.expected_microbatches:
            raise RuntimeError(
                "deferred DAPO metric window is incomplete: "
                f"expected {self.expected_microbatches}, got {len(self._batches)} microbatches"
            )
        rows = []
        for reduced, low_clip_rows, high_clip_rows in self._batches:
            means = torch.stack(reduced).reshape(-1)
            rows.append(torch.cat((means, low_clip_rows, high_clip_rows)))
        return torch.stack(rows)

    def decode(self, gathered: torch.Tensor, *, world_size: int) -> dict[str, list[float]]:
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        columns = 2 * _REDUCE_CALLS_PER_MICROBATCH + 2 * self.rows_per_microbatch
        if not isinstance(gathered, torch.Tensor) or tuple(gathered.shape) != (
            world_size * self.expected_microbatches,
            columns,
        ):
            raise RuntimeError(
                "packed DAPO metric gather has an unexpected shape: "
                f"got {tuple(getattr(gathered, 'shape', ()))}, "
                f"expected {(world_size * self.expected_microbatches, columns)}"
            )
        by_rank = gathered.reshape(world_size, self.expected_microbatches, columns)
        reduced = by_rank[:, :, : 2 * _REDUCE_CALLS_PER_MICROBATCH].reshape(
            world_size, self.expected_microbatches, _REDUCE_CALLS_PER_MICROBATCH, 2
        )
        global_totals = reduced.sum(dim=0)
        means = global_totals[:, :, 0] / global_totals[:, :, 1].clamp(min=1.0)

        low_start = 2 * _REDUCE_CALLS_PER_MICROBATCH
        high_start = low_start + self.rows_per_microbatch
        low_rows = by_rank[:, :, low_start:high_start].permute(1, 0, 2).flatten(1)
        high_rows = by_rank[:, :, high_start:].permute(1, 0, 2).flatten(1)
        low_min = _nan_extreme(low_rows, minimum=True, dim=1)
        high_max = _nan_extreme(high_rows, minimum=False, dim=1)

        # One device-to-host transfer follows the single cross-rank gather.
        mean_values = means.cpu().tolist()
        low_values = low_min.cpu().tolist()
        high_values = high_max.cpu().tolist()
        return {
            _METRIC_KEYS[0]: [float(row[0]) for row in mean_values],
            _METRIC_KEYS[1]: [float(row[1]) for row in mean_values],
            _METRIC_KEYS[2]: [float(row[2]) for row in mean_values],
            _METRIC_KEYS[3]: [float(row[3]) for row in mean_values],
            _METRIC_KEYS[4]: [float(value) for value in low_values],
            _METRIC_KEYS[5]: [float(value) for value in high_values],
        }


class _DapoMetricCapture:
    def __init__(self, *, rows_per_microbatch: int):
        self.rows_per_microbatch = rows_per_microbatch
        self.reduce_inputs: list[torch.Tensor] = []
        self.gather_inputs: list[torch.Tensor] = []
        self._reduce_placeholder = torch.tensor([0.0, 1.0], dtype=torch.float32, device="cpu")

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum", scale: float = 1.0) -> torch.Tensor:
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != (2,)
            or reduction != "sum"
            or scale != 1.0
        ):
            raise RuntimeError("unexpected collective inside the deferred TRL DAPO metrics scope")
        self.reduce_inputs.append(tensor.detach())
        return self._reduce_placeholder

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 1
            or tensor.numel() != self.rows_per_microbatch
        ):
            raise RuntimeError("unexpected gather inside the deferred TRL DAPO metrics scope")
        self.gather_inputs.append(tensor.detach())
        return torch.zeros((tensor.numel(),), dtype=tensor.dtype, device="cpu")

    def validate(self) -> None:
        if len(self.reduce_inputs) != _REDUCE_CALLS_PER_MICROBATCH:
            raise RuntimeError(
                "pinned TRL DAPO metric contract changed: expected four reduce calls, "
                f"got {len(self.reduce_inputs)}"
            )
        if len(self.gather_inputs) != _GATHER_CALLS_PER_MICROBATCH:
            raise RuntimeError(
                "pinned TRL DAPO metric contract changed: expected two gather calls, "
                f"got {len(self.gather_inputs)}"
            )


@contextmanager
def capture_trl_dapo_metric_collectives(
    accelerator: Any,
    *,
    rows_per_microbatch: int,
) -> Iterator[_DapoMetricCapture]:
    """Replace only the known metric collectives during one upstream loss call."""
    capture = _DapoMetricCapture(rows_per_microbatch=rows_per_microbatch)
    instance_dict = getattr(accelerator, "__dict__", {})
    had_reduce = "reduce" in instance_dict
    had_gather = "gather" in instance_dict
    previous_reduce = instance_dict.get("reduce")
    previous_gather = instance_dict.get("gather")
    with _PATCH_LOCK:
        try:
            accelerator.reduce = capture.reduce
            accelerator.gather = capture.gather
            yield capture
        finally:
            if had_reduce:
                accelerator.reduce = previous_reduce
            else:
                delattr(accelerator, "reduce")
            if had_gather:
                accelerator.gather = previous_gather
            else:
                delattr(accelerator, "gather")


__all__ = ["DeferredDapoMetricWindow", "capture_trl_dapo_metric_collectives"]
