"""Bounded asynchronous activation copies for the single-device T2 Learner.

The existing pinned CPU arena owns saved tensor payloads. This policy adds one
communication stream, bounded in-flight D2H work, and a one-ticket H2D
prefetch cache. Predictions are hints only: an unpack uses the cache only when
its weakly referenced ticket is the exact requested object.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any
import weakref

import torch

from .bounded_offload import BoundedPinnedSaveOnCPU, _PinnedTicket
from .ternary_head_offload import RecomputeTernaryHeadSaveOnCPU


DEFAULT_ASYNC_COPY_BUDGET_BYTES = 512 * 1024**2


@dataclass
class _D2HTransfer:
    event: Any
    budget_bytes: int
    source: torch.Tensor
    source_version: int


@dataclass
class _PrefetchEntry:
    ticket: weakref.ReferenceType[_PinnedTicket]
    tensor: torch.Tensor
    ready_event: Any
    size_bytes: int


class AsyncActivationOffload(BoundedPinnedSaveOnCPU):
    """Add async copies to the existing bounded pinned CPU arena."""

    def __init__(
        self,
        base_pack_hook,
        base_unpack_hook,
        *,
        budget_bytes: int = DEFAULT_ASYNC_COPY_BUDGET_BYTES,
        **kwargs: Any,
    ) -> None:
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ValueError("budget_bytes must be a positive integer")
        super().__init__(base_pack_hook, base_unpack_hook, **kwargs)
        self.budget_bytes = budget_bytes
        self._copy_stream: torch.cuda.Stream | None = None
        self._stream_device: torch.device | None = None
        self._closed = False
        self._operation_lock = threading.Lock()
        self._inflight: deque[_D2HTransfer] = deque()
        self._inflight_bytes = 0
        self._pack_order: list[weakref.ReferenceType[_PinnedTicket]] = []
        self._ticket_indices: dict[
            int, tuple[weakref.ReferenceType[_PinnedTicket], int]
        ] = {}
        self._prefetch: _PrefetchEntry | None = None
        self._async_stats: dict[str, Any] = {
            "d2h_async_count": 0,
            "d2h_peak_outstanding_bytes": 0,
            "d2h_backpressure_seconds": 0.0,
            "demand_hits": 0,
            "demand_misses": 0,
            "predictions": 0,
            "predicted_bytes": 0,
            "peak_speculative_bytes": 0,
            "unused_predictions": 0,
            "over_cap_skips": 0,
            "native_or_oversized_pack": 0,
        }

    @property
    def report(self) -> dict[str, Any]:
        return {
            **super().report,
            "stream_prefetch": {
                **self._async_stats,
                "budget_bytes": self.budget_bytes,
                "pending_prediction": self._prefetch is not None,
                "closed": self._closed,
            },
        }

    @staticmethod
    def _d2h_budget_bytes(tensor: torch.Tensor) -> int:
        """Bound both logical transfer bytes and source storage kept alive."""
        payload_bytes = tensor.numel() * tensor.element_size()
        source_storage_bytes = tensor.untyped_storage().nbytes()
        return max(payload_bytes, source_storage_bytes)

    @staticmethod
    def _version(tensor: torch.Tensor) -> int | None:
        try:
            return int(tensor._version)
        except RuntimeError:
            # Inference tensors have no version counter. Keep those on the
            # caller's native path instead of guessing about mutation safety.
            return None

    def _ensure_copy_stream(self, device: torch.device) -> torch.cuda.Stream:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        selected_device = torch.device("cuda", device_index)
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=selected_device)
            self._stream_device = selected_device
        elif self._stream_device != selected_device:
            raise RuntimeError(
                "one AsyncActivationOffload policy cannot span multiple CUDA devices"
            )
        return self._copy_stream

    @staticmethod
    def _check_source_version(transfer: _D2HTransfer) -> None:
        if transfer.source._version != transfer.source_version:
            raise RuntimeError("saved activation mutated before asynchronous D2H retired")

    def _retire_completed(self) -> None:
        while self._inflight and self._inflight[0].event.query():
            transfer = self._inflight.popleft()
            self._inflight_bytes -= transfer.budget_bytes
            self._check_source_version(transfer)

    def _wait_for_d2h_capacity(self, required_bytes: int) -> None:
        self._retire_completed()
        while self._inflight and self._inflight_bytes + required_bytes > self.budget_bytes:
            transfer = self._inflight.popleft()
            started = time.perf_counter()
            try:
                transfer.event.synchronize()
                self._check_source_version(transfer)
            finally:
                self._inflight_bytes -= transfer.budget_bytes
                self._async_stats["d2h_backpressure_seconds"] += (
                    time.perf_counter() - started
                )

    def pack(self, tensor: torch.Tensor) -> Any:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("concurrent asynchronous offload hooks are unsupported")
        try:
            return self._pack(tensor)
        finally:
            self._operation_lock.release()

    def _pack(self, tensor: torch.Tensor) -> Any:
        if self._closed:
            # Saved-tensor hooks may retain this callback beyond context exit.
            # Late checkpoint recomputation remains correct, without launching
            # new async work, through the parent implementation.
            return super().pack(tensor)
        if tensor.device.type != "cuda" or not self._eligible(tensor):
            self._async_stats["native_or_oversized_pack"] += 1
            return super().pack(tensor)
        if tensor.numel() * tensor.element_size() < self.min_tensor_bytes:
            self._async_stats["native_or_oversized_pack"] += 1
            return super().pack(tensor)

        budget_bytes = self._d2h_budget_bytes(tensor)
        if budget_bytes > self.budget_bytes:
            self._async_stats["native_or_oversized_pack"] += 1
            return super().pack(tensor)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("asynchronous activation offload does not support CUDA capture")

        source_version = self._version(tensor)
        if source_version is None:
            self._async_stats["native_or_oversized_pack"] += 1
            return super().pack(tensor)

        copy_stream = self._ensure_copy_stream(tensor.device)
        self._wait_for_d2h_capacity(budget_bytes)
        source = tensor.detach()
        producer_stream = torch.cuda.current_stream(tensor.device)
        producer_ready = producer_stream.record_event()
        try:
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(producer_ready)
                ticket = super().pack(tensor)
                # Allocator lifetime only; the version check independently
                # rejects in-place writes while the DMA read is outstanding.
                source.record_stream(copy_stream)
                transfer_done = copy_stream.record_event()
        except BaseException:
            # If record_stream fails after a copy was queued, finish the DMA
            # before dropping this stack's source reference.
            copy_stream.synchronize()
            raise

        transfer = _D2HTransfer(
            event=transfer_done,
            budget_bytes=budget_bytes,
            source=source,
            source_version=source_version,
        )
        self._inflight.append(transfer)
        self._inflight_bytes += budget_bytes
        self._async_stats["d2h_async_count"] += 1
        self._async_stats["d2h_peak_outstanding_bytes"] = max(
            self._async_stats["d2h_peak_outstanding_bytes"], self._inflight_bytes
        )

        if isinstance(ticket, _PinnedTicket):
            ticket_ref = weakref.ref(ticket)
            self._ticket_indices[id(ticket)] = (ticket_ref, len(self._pack_order))
            self._pack_order.append(ticket_ref)
        return ticket

    def unpack(self, packed: Any) -> torch.Tensor:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("concurrent asynchronous offload hooks are unsupported")
        try:
            return self._unpack(packed)
        finally:
            self._operation_lock.release()

    def _unpack(self, packed: Any) -> torch.Tensor:
        if self._closed or not isinstance(packed, _PinnedTicket):
            return super().unpack(packed)

        cache = self._prefetch
        if cache is not None and cache.ticket() is packed:
            consumer_stream = torch.cuda.current_stream(packed.device)
            consumer_stream.wait_event(cache.ready_event)
            cache.tensor.record_stream(consumer_stream)
            result = cache.tensor
            self._prefetch = None
            self._async_stats["demand_hits"] += 1
        else:
            self._async_stats["demand_misses"] += 1
            if cache is not None and cache.ready_event.query():
                self._prefetch = None
                self._async_stats["unused_predictions"] += 1
            result = super().unpack(packed)

        self._predict_next(packed)
        return result

    def _predict_next(self, ticket: _PinnedTicket) -> None:
        if self._closed or self._prefetch is not None or self._copy_stream is None:
            return
        entry = self._ticket_indices.get(id(ticket))
        if entry is None or entry[0]() is not ticket:
            return

        for index in range(entry[1] - 1, -1, -1):
            next_ticket = self._pack_order[index]()
            if next_ticket is None:
                continue
            size_bytes = next_ticket.cpu_tensor.numel() * next_ticket.cpu_tensor.element_size()
            if size_bytes > self.budget_bytes:
                self._async_stats["over_cap_skips"] += 1
                continue
            with torch.cuda.stream(self._copy_stream):
                tensor = super().unpack(next_ticket)
                ready_event = self._copy_stream.record_event()
            self._prefetch = _PrefetchEntry(
                ticket=weakref.ref(next_ticket),
                tensor=tensor,
                ready_event=ready_event,
                size_bytes=size_bytes,
            )
            self._async_stats["predictions"] += 1
            self._async_stats["predicted_bytes"] += size_bytes
            self._async_stats["peak_speculative_bytes"] = max(
                self._async_stats["peak_speculative_bytes"], size_bytes
            )
            return

    def close(self) -> None:
        # Context exit is outside hook callbacks. Wait for any callback already
        # in progress before closing, then keep cleanup idempotent.
        with self._operation_lock:
            if self._closed:
                return
            error: BaseException | None = None
            try:
                if self._copy_stream is not None:
                    self._copy_stream.synchronize()
                for transfer in self._inflight:
                    self._check_source_version(transfer)
            except BaseException as exc:
                error = exc
            finally:
                if self._prefetch is not None:
                    self._async_stats["unused_predictions"] += 1
                self._prefetch = None
                self._inflight.clear()
                self._inflight_bytes = 0
                self._pack_order.clear()
                self._ticket_indices.clear()
                self._closed = True
            if error is not None:
                raise error


class AsyncTernaryHeadSaveOnCPU(RecomputeTernaryHeadSaveOnCPU):
    """Retain frozen T2 head recomputation while cleaning the async policy."""

    def __init__(self, offload: AsyncActivationOffload) -> None:
        super().__init__(offload.pack, offload.unpack)
        self.bounded_offload = offload

    def __exit__(self, *args: Any) -> Any:
        active_error = args[1] if len(args) > 1 else None
        super_error: BaseException | None = None
        try:
            return super().__exit__(*args)
        except BaseException as exc:
            super_error = exc
            raise
        finally:
            try:
                self.bounded_offload.close()
            except BaseException as cleanup_error:
                self.report["status"] = "failed"
                primary_error = active_error if active_error is not None else super_error
                if primary_error is not None:
                    primary_error.add_note(
                        f"Asynchronous activation offload cleanup failed: {cleanup_error}"
                    )
                else:
                    raise
