"""Saved-tensor offload through one process-wide, fixed-capacity pinned arena.

Backing storage grows lazily in independent slabs and can be reclaimed when
leases and their recorded copies are finished. Tensors that do not fit use the
caller's native pageable hooks, so pinned allocation stays within the cap.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import weakref
from typing import Any, Callable, Mapping

import torch


DEFAULT_MAX_PINNED_BYTES = 16 * 1024**3
DEFAULT_MIN_TENSOR_BYTES = 1024 * 1024
_ARENA_ALIGNMENT = 64
_DEFAULT_ARENA_SLAB_BYTES = 64 * 1024**2


def _pinned_chunk_size_plan(
    minimum_size: int,
    remaining_bytes: int,
    slab_bytes: int,
    *,
    reserve_factor: int = 3,
) -> int:
    """Return a cap-safe power-of-two pinned backing size, or zero if it cannot fit."""
    if (
        minimum_size <= 0
        or remaining_bytes <= 0
        or slab_bytes <= 0
        or reserve_factor <= 0
    ):
        return 0
    preferred_size = max(slab_bytes, minimum_size * reserve_factor)
    preferred_size = 1 << (preferred_size - 1).bit_length()
    if preferred_size <= remaining_bytes:
        return preferred_size
    minimum_power_of_two = 1 << (minimum_size - 1).bit_length()
    if minimum_power_of_two <= remaining_bytes:
        return minimum_power_of_two
    return 0


@dataclass
class _ArenaChunk:
    backing: torch.Tensor
    size: int


@dataclass
class _FreeBlock:
    chunk_id: int
    offset: int
    size: int
    # One latest event per CUDA stream is enough: stream order makes it cover
    # all earlier copies recorded on that stream.
    events: dict[tuple[int, int], tuple[int, Any]]


class _ArenaLease:
    def __init__(self, arena: "PinnedCPUArena", block: _FreeBlock, payload_size: int) -> None:
        self._arena = arena
        self.chunk_id = block.chunk_id
        self.offset = block.offset
        self.size = block.size
        self.payload_size = payload_size
        self._events = dict(block.events)
        self._view_refs: list[Any] = []
        self._released = False
        self._lock = threading.Lock()

    def record_event(self, stream: Any, device_index: int) -> Any:
        key = (device_index, int(stream.cuda_stream))
        with self._lock:
            if self._released:
                raise RuntimeError("cannot record a transfer on a released arena lease")
            sequence, event = self._arena._record_event(key, stream)
            previous = self._events.get(key)
            if previous is None or sequence > previous[0]:
                self._events[key] = (sequence, event)
            return event

    def event_snapshot(self) -> dict[tuple[int, int], tuple[int, Any]]:
        with self._lock:
            return dict(self._events)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._view_refs = [view_ref for view_ref in self._view_refs if view_ref() is not None]
            if self._view_refs:
                # A tensor view keeps the lease alive too, so a ticket's host
                # storage cannot be recycled while the view is still reachable.
                return
            self._released = True
            events = dict(self._events)
        self._arena._release(self.chunk_id, self.offset, self.size, self.payload_size, events)

    def __del__(self) -> None:
        self.release()


class PinnedCPUArena:
    """A fixed-capacity arena that allocates CPU backing in independent slabs.

    ``pin_memory=False`` is intended for CPU-only contract tests. Production
    instances use pinned storage and are shared process-wide by default.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        pin_memory: bool = True,
        slab_bytes: int = _DEFAULT_ARENA_SLAB_BYTES,
    ) -> None:
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be a positive integer")
        if type(slab_bytes) is not int or slab_bytes <= 0:
            raise ValueError("slab_bytes must be a positive integer")
        self.capacity_bytes = capacity_bytes
        self.pin_memory = bool(pin_memory)
        self.slab_bytes = slab_bytes
        self._chunks: dict[int, _ArenaChunk] = {}
        self._next_chunk_id = 0
        self._allocated_bytes = 0
        self._free: list[_FreeBlock] = []
        self._live_bytes = 0
        self._live_payload_bytes = 0
        self._max_live_bytes = 0
        self._lock = threading.RLock()
        self._stream_sequences: dict[tuple[int, int], int] = {}

    @property
    def initialized(self) -> bool:
        with self._lock:
            return bool(self._chunks)

    def initialize(self) -> "PinnedCPUArena":
        """Warm one slab now, without allocating the full configured capacity."""
        with self._lock:
            if not self._chunks:
                warm_size = min(self.slab_bytes, self.capacity_bytes)
                if self.pin_memory:
                    warm_size = 1 << (warm_size.bit_length() - 1)
                if not self._allocate_chunk(warm_size, reserve_factor=1):
                    raise RuntimeError("cannot warm a pinned arena slab within its fixed capacity")
        return self

    @property
    def allocated_bytes(self) -> int:
        """Bytes in arena-owned backing chunks; excludes Torch allocator cache."""
        with self._lock:
            return self._allocated_bytes

    @property
    def chunk_count(self) -> int:
        """Number of backing chunks currently owned by this arena."""
        with self._lock:
            return len(self._chunks)

    @property
    def payload_bytes(self) -> int:
        """Logical bytes requested by live leases, excluding alignment padding."""
        with self._lock:
            return self._live_payload_bytes

    @property
    def slack_bytes(self) -> int:
        """Allocated bytes not occupied by live lease payloads."""
        with self._lock:
            return self._allocated_bytes - self._live_payload_bytes

    @property
    def resident_bytes(self) -> int:
        """Arena-owned pinned backing bytes, excluding Torch cache after trimming."""
        with self._lock:
            return self._allocated_bytes if self.pin_memory else 0

    @property
    def live_bytes(self) -> int:
        """Live arena reservations, including alignment padding."""
        with self._lock:
            return self._live_bytes

    @property
    def max_live_bytes(self) -> int:
        with self._lock:
            return self._max_live_bytes

    def try_acquire(self, size: int) -> tuple[_ArenaLease | None, str | None]:
        if size <= 0:
            return None, "empty"
        reserved_size = (size + _ARENA_ALIGNMENT - 1) // _ARENA_ALIGNMENT * _ARENA_ALIGNMENT
        if reserved_size > self.capacity_bytes:
            return None, "oversized"
        with self._lock:
            block_index = next(
                (index for index, block in enumerate(self._free) if block.size >= reserved_size),
                None,
            )
            if block_index is None:
                available_bytes = self.capacity_bytes - self._allocated_bytes
                if available_bytes >= reserved_size and self._allocate_chunk(reserved_size):
                    block_index = next(
                        index for index, block in enumerate(self._free)
                        if block.size >= reserved_size
                    )
                else:
                    free_bytes = sum(block.size for block in self._free)
                    reason = "fragmented" if free_bytes >= reserved_size else "arena_full"
                    return None, reason

            block = self._free[block_index]
            lease_block = _FreeBlock(
                block.chunk_id, block.offset, reserved_size, dict(block.events)
            )
            if block.size == reserved_size:
                del self._free[block_index]
            else:
                self._free[block_index] = _FreeBlock(
                    block.chunk_id,
                    block.offset + reserved_size,
                    block.size - reserved_size,
                    dict(block.events),
                )
            self._live_bytes += reserved_size
            self._live_payload_bytes += size
            self._max_live_bytes = max(self._max_live_bytes, self._live_bytes)
            return _ArenaLease(self, lease_block, size), None

    def tensor_view(
        self,
        lease: _ArenaLease,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        byte_count: int,
    ) -> torch.Tensor:
        if lease._arena is not self:
            raise ValueError("arena lease belongs to a different pinned CPU arena")
        with lease._lock:
            if lease._released:
                raise RuntimeError("cannot create a tensor view from a released arena lease")
            if byte_count < 0 or byte_count > lease.payload_size:
                raise ValueError("tensor view exceeds its arena lease payload")
            with self._lock:
                chunk = self._chunks.get(lease.chunk_id)
                if chunk is None:
                    raise RuntimeError("arena lease backing chunk is no longer allocated")
                byte_view = chunk.backing.narrow(0, lease.offset, byte_count)
                view = byte_view.view(dtype).view(shape)
                lease._view_refs = [
                    view_ref for view_ref in lease._view_refs if view_ref() is not None
                ]
                lease._view_refs.append(weakref.ref(view))
                # The production ticket owns this direct tensor view. Keeping
                # its lease on the view also protects it during ticket teardown.
                view._pinned_cpu_arena_lease = lease
                return view

    def wait_for_prior_copies(self, lease: _ArenaLease, stream: Any, device_index: int) -> None:
        for (event_device, _stream_id), (_sequence, event) in lease.event_snapshot().items():
            if event_device == device_index:
                stream.wait_event(event)
            else:
                # Cross-device stream waits are not portable. This path is not
                # used by the single-GPU Learner, but remains safe if shared.
                event.synchronize()

    def _allocate_chunk(self, minimum_size: int, *, reserve_factor: int = 3) -> bool:
        remaining = self.capacity_bytes - self._allocated_bytes
        if self.pin_memory:
            chunk_size = _pinned_chunk_size_plan(
                minimum_size, remaining, self.slab_bytes, reserve_factor=reserve_factor
            )
        else:
            slab_rounded_size = (
                (minimum_size + self.slab_bytes - 1) // self.slab_bytes
            ) * self.slab_bytes
            chunk_size = min(remaining, slab_rounded_size)
        if chunk_size < minimum_size:
            return False
        backing = torch.empty(
            (chunk_size,), dtype=torch.uint8, device="cpu", pin_memory=self.pin_memory
        )
        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        self._chunks[chunk_id] = _ArenaChunk(backing, chunk_size)
        self._allocated_bytes += chunk_size
        self._free.append(_FreeBlock(chunk_id, 0, chunk_size, {}))
        self._free.sort(key=lambda block: (block.chunk_id, block.offset))
        return True

    def _record_event(self, key: tuple[int, int], stream: Any) -> tuple[int, Any]:
        with self._lock:
            event = torch.cuda.Event(enable_timing=False)
            event.record(stream)
            sequence = self._stream_sequences.get(key, 0) + 1
            self._stream_sequences[key] = sequence
            return sequence, event

    def _release(
        self,
        chunk_id: int,
        offset: int,
        size: int,
        payload_size: int,
        events: dict[tuple[int, int], tuple[int, Any]],
    ) -> None:
        with self._lock:
            chunk = self._chunks.get(chunk_id)
            if chunk is None or offset < 0 or offset + size > chunk.size:
                raise RuntimeError("pinned arena lease refers to an invalid backing range")
            self._live_bytes -= size
            self._live_payload_bytes -= payload_size
            if self._live_bytes < 0 or self._live_payload_bytes < 0:
                raise RuntimeError("pinned arena lease accounting underflow")
            self._free.append(_FreeBlock(chunk_id, offset, size, dict(events)))
            self._free.sort(key=lambda block: (block.chunk_id, block.offset))
            merged: list[_FreeBlock] = []
            for block in self._free:
                if (
                    merged
                    and merged[-1].chunk_id == block.chunk_id
                    and merged[-1].offset + merged[-1].size == block.offset
                ):
                    previous = merged[-1]
                    previous.size += block.size
                    for key, event_entry in block.events.items():
                        current = previous.events.get(key)
                        if current is None or event_entry[0] > current[0]:
                            previous.events[key] = event_entry
                else:
                    merged.append(block)
            self._free = merged

    def prepare_idle_retention(self) -> bool:
        """Retain backing only after copies finish; discard completed event metadata."""
        with self._lock:
            if self._live_bytes or self._live_payload_bytes:
                return False
            if any(not self._events_complete(block.events) for block in self._free):
                return False
            for block in self._free:
                block.events.clear()
            self._stream_sequences.clear()
            return True

    def trim_idle(self) -> int:
        """Drop arena references to fully free chunks after recorded copies complete.

        The returned byte count is no longer owned by this arena; PyTorch's
        pinned host allocator may cache the underlying storage.
        """
        with self._lock:
            idle_ids: set[int] = set()
            for block in self._free:
                chunk = self._chunks.get(block.chunk_id)
                if (
                    chunk is not None
                    and block.offset == 0
                    and block.size == chunk.size
                    and self._events_complete(block.events)
                ):
                    idle_ids.add(block.chunk_id)
            if not idle_ids:
                return 0
            reclaimed = sum(self._chunks[chunk_id].size for chunk_id in idle_ids)
            self._free = [block for block in self._free if block.chunk_id not in idle_ids]
            for chunk_id in idle_ids:
                del self._chunks[chunk_id]
            self._allocated_bytes -= reclaimed
            return reclaimed

    @staticmethod
    def _events_complete(events: dict[tuple[int, int], tuple[int, Any]]) -> bool:
        for _sequence, event in events.values():
            try:
                if not event.query():
                    return False
            except Exception:
                return False
        return True


_PROCESS_ARENA: PinnedCPUArena | None = None
_PROCESS_ARENA_LOCK = threading.Lock()


def _process_arena(capacity_bytes: int) -> PinnedCPUArena:
    global _PROCESS_ARENA
    with _PROCESS_ARENA_LOCK:
        if _PROCESS_ARENA is None:
            _PROCESS_ARENA = PinnedCPUArena(capacity_bytes, pin_memory=True)
        elif _PROCESS_ARENA.capacity_bytes != capacity_bytes:
            raise ValueError(
                "the process-wide pinned arena already has a different fixed capacity"
            )
        return _PROCESS_ARENA


def reclaim_idle_pinned_cache(
    *,
    empty_host_cache: Callable[[], None],
    host_memory_stats: Callable[[], Mapping[str, Any]],
    retain_if_safe: bool = False,
    memory_capacity: Callable[[], Mapping[str, Any]] | None = None,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Release idle backing, or retain bounded backing with verified headroom.

    The caller must reach a quiescent training boundary and synchronize CUDA
    copies before calling this helper. It only inspects an already-created
    process arena; it never creates or warms one. Live leases and chunks with
    incomplete copy events remain owned by the arena. Empty host allocator
    cache is delegated to the pinned Torch runtime.
    Retention requires a finite observed cgroup budget, room for peer roles
    and the caller's residency backup reserve, and no live arena leases.
    Unavailable observations retain the original reclaim behavior.
    """
    if not callable(empty_host_cache):
        raise TypeError("empty_host_cache must be callable")
    if not callable(host_memory_stats):
        raise TypeError("host_memory_stats must be callable")

    with _PROCESS_ARENA_LOCK:
        arena = _PROCESS_ARENA

    def read_host_stats() -> dict[str, Any]:
        value = host_memory_stats()
        if not isinstance(value, Mapping):
            raise TypeError("host_memory_stats must return a mapping; use {} when unavailable")
        return dict(value)

    def arena_stats() -> dict[str, Any]:
        if arena is None:
            return {
                "arena_capacity_bytes": None,
                "arena_allocated_bytes": None,
                "arena_live_bytes": None,
                "arena_live_payload_bytes": None,
                "arena_chunk_count": None,
            }
        return {
            "arena_capacity_bytes": arena.capacity_bytes,
            "arena_allocated_bytes": arena.allocated_bytes,
            "arena_live_bytes": arena.live_bytes,
            "arena_live_payload_bytes": arena.payload_bytes,
            "arena_chunk_count": arena.chunk_count,
        }

    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a non-negative integer")
    before_arena = arena_stats()
    host_before = read_host_stats()
    decision = {"retained": False, "reason": "explicit_reclaim"}
    if retain_if_safe and arena is not None and arena.allocated_bytes:
        try:
            if memory_capacity is None:
                # Reuse the same read-only host/cgroup observer as preflight.
                from qwen_rl.deployment.instance_storage import _memory_capacity
                memory_capacity = _memory_capacity
            memory = dict(memory_capacity())
            limit = memory["cgroup_max_bytes"]
            available = [memory["host_available_bytes"], memory["cgroup_available_bytes"]]
            if (type(limit) is not int or limit <= 0
                    or any(type(value) is not int or value < 0 for value in available)):
                raise ValueError("finite host/cgroup memory budget is unavailable")
            # Keep spare capacity for peer roles as well as this owner's backup.
            required = max(2 * arena.capacity_bytes, limit // 4) + reserve_bytes
            decision.update(memory=memory, required_available_bytes=required,
                            available_bytes=min(available))
            if arena.live_bytes or arena.payload_bytes:
                decision["reason"] = "live_leases"
            elif min(available) < required:
                decision["reason"] = "memory_pressure"
            elif not arena.prepare_idle_retention():
                decision["reason"] = "pending_copies"
            else:
                return {
                    "status": "completed", "arena_present": True,
                    "arena_trimmed_bytes": 0, "arena_before": before_arena,
                    "arena_after": arena_stats(), "host_memory_before": host_before,
                    "host_memory_after": host_before, "empty_host_cache_called": False,
                    "cache_policy": {**decision, "retained": True, "reason": "headroom"},
                }
        except (OSError, ValueError, TypeError, KeyError) as exc:
            decision.update(reason="memory_observation_unavailable", error=str(exc))
    trimmed_bytes = 0 if arena is None else arena.trim_idle()
    # Drop only unoccupied blocks from the runtime's pinned allocator. The
    # operation intentionally stays after lease/event-aware arena trimming.
    empty_host_cache()
    after_arena = arena_stats()
    host_after = read_host_stats()
    return {
        "status": "completed",
        "arena_present": arena is not None,
        "arena_trimmed_bytes": trimmed_bytes,
        "arena_before": before_arena,
        "arena_after": after_arena,
        "host_memory_before": host_before,
        "host_memory_after": host_after,
        "empty_host_cache_called": True,
        "cache_policy": decision,
    }


@dataclass
class _PinnedTicket:
    lease: _ArenaLease
    cpu_tensor: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool
    d2h_event: Any | None


def _non_overlapping_dense(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    dimensions = sorted(
        (step, length) for length, step in zip(shape, stride, strict=True) if length > 1
    )
    expected = 1
    for step, length in dimensions:
        if step != expected:
            return False
        expected *= length
    return True


class BoundedPinnedSaveOnCPU:
    """Use a fixed pinned arena for admitted tensors, then delegate to native hooks.

    Share the default process arena across contexts and models. An explicit
    ``arena`` is provided for isolated CPU tests or managed integration tests.
    """

    def __init__(
        self,
        base_pack_hook: Callable[[torch.Tensor], Any],
        base_unpack_hook: Callable[[Any], torch.Tensor],
        *,
        max_pinned_bytes: int = DEFAULT_MAX_PINNED_BYTES,
        min_tensor_bytes: int = DEFAULT_MIN_TENSOR_BYTES,
        arena: PinnedCPUArena | None = None,
        device_type: str = "cuda",
    ) -> None:
        if type(max_pinned_bytes) is not int or max_pinned_bytes <= 0:
            raise ValueError("max_pinned_bytes must be a positive integer")
        if type(min_tensor_bytes) is not int or min_tensor_bytes < 0:
            raise ValueError("min_tensor_bytes must be a non-negative integer")
        if device_type not in {"cuda", "cpu"}:
            raise ValueError("device_type must be 'cuda' or 'cpu'")
        self.base_pack_hook = base_pack_hook
        self.base_unpack_hook = base_unpack_hook
        self.arena = arena if arena is not None else _process_arena(max_pinned_bytes)
        self.min_tensor_bytes = min_tensor_bytes
        self.device_type = device_type
        self._hooks = None
        self._status = "ready"
        self._counts = {
            "base_pack_delegations": 0,
            "base_unpack_delegations": 0,
            "pinned_pack_count": 0,
            "pinned_unpack_count": 0,
            "admitted_bytes": 0,
            "pageable_bytes": 0,
            "pageable_tensor_count": 0,
            "pageable_admission_reasons": {},
        }

    @property
    def report(self) -> dict[str, Any]:
        return {
            "status": self._status,
            **self._counts,
            "arena_capacity_bytes": self.arena.capacity_bytes,
            "arena_allocated_bytes": self.arena.allocated_bytes,
            "arena_live_payload_bytes": self.arena.payload_bytes,
            "arena_slack_bytes": self.arena.slack_bytes,
            "arena_chunk_count": self.arena.chunk_count,
            "arena_initialized": self.arena.initialized,
            "live_arena_bytes": self.arena.live_bytes,
            "max_live_arena_bytes": self.arena.max_live_bytes,
            "live_pinned_bytes": self.arena.live_bytes if self.arena.pin_memory else 0,
            "max_live_pinned_bytes": self.arena.max_live_bytes if self.arena.pin_memory else 0,
            "resident_pinned_bytes": self.arena.resident_bytes,
        }

    def pack(self, tensor: torch.Tensor) -> Any:
        if not self._eligible(tensor):
            self._counts["base_pack_delegations"] += 1
            return self.base_pack_hook(tensor)

        num_bytes = tensor.numel() * tensor.element_size()
        if num_bytes < self.min_tensor_bytes:
            self._counts["base_pack_delegations"] += 1
            return self.base_pack_hook(tensor)

        lease, reason = self.arena.try_acquire(num_bytes)
        if lease is None:
            self._counts["base_pack_delegations"] += 1
            if reason != "empty":
                self._counts["pageable_bytes"] += num_bytes
                self._counts["pageable_tensor_count"] += 1
                reasons = self._counts["pageable_admission_reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
            return self.base_pack_hook(tensor)

        stream = torch.cuda.current_stream(tensor.device) if tensor.device.type == "cuda" else None
        device_index = (
            tensor.device.index if tensor.device.index is not None else torch.cuda.current_device()
        ) if stream is not None else 0
        try:
            if stream is not None:
                self.arena.wait_for_prior_copies(lease, stream, device_index)
            cpu_tensor = self.arena.tensor_view(
                lease, tensor.dtype, tuple(tensor.shape), num_bytes
            )
            with torch.no_grad():
                cpu_tensor.copy_(tensor, non_blocking=bool(stream is not None and self.arena.pin_memory))
            d2h_event = None
            if stream is not None:
                d2h_event = lease.record_event(stream, device_index)
            self._counts["pinned_pack_count"] += 1
            self._counts["admitted_bytes"] += num_bytes
            return _PinnedTicket(
                lease=lease,
                cpu_tensor=cpu_tensor,
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                device=tensor.device,
                requires_grad=bool(tensor.requires_grad),
                d2h_event=d2h_event,
            )
        except BaseException:
            # If a copy was queued before an error, synchronize before making
            # its host range reusable. This is an exceptional correctness path.
            if stream is not None:
                stream.synchronize()
            lease.release()
            raise

    def unpack(self, packed: Any) -> torch.Tensor:
        if not isinstance(packed, _PinnedTicket):
            self._counts["base_unpack_delegations"] += 1
            return self.base_unpack_hook(packed)

        stream = torch.cuda.current_stream(packed.device) if packed.device.type == "cuda" else None
        try:
            if stream is not None and packed.d2h_event is not None:
                stream.wait_event(packed.d2h_event)
            if _non_overlapping_dense(packed.shape, packed.stride):
                restored = torch.empty_strided(
                    packed.shape, packed.stride, dtype=packed.dtype, device=packed.device
                )
            else:
                restored = torch.empty(packed.shape, dtype=packed.dtype, device=packed.device)
            with torch.no_grad():
                restored.copy_(
                    packed.cpu_tensor,
                    non_blocking=bool(stream is not None and self.arena.pin_memory),
                )
            if packed.requires_grad:
                restored.requires_grad_(True)
            if stream is not None:
                device_index = (
                    packed.device.index if packed.device.index is not None
                    else torch.cuda.current_device()
                )
                packed.lease.record_event(stream, device_index)
        except BaseException:
            # The host range must stay reserved until any queued H2D read ends.
            if stream is not None:
                stream.synchronize()
            raise
        self._counts["pinned_unpack_count"] += 1
        return restored

    def _eligible(self, tensor: torch.Tensor) -> bool:
        if type(tensor) is not torch.Tensor:
            return False
        if tensor.device.type != self.device_type or tensor.layout != torch.strided:
            return False
        if tensor.is_quantized or tensor.numel() == 0:
            return False
        if isinstance(tensor, torch.nn.Parameter):
            return False
        buffer_type = getattr(torch.nn, "Buffer", None)
        if buffer_type is not None and isinstance(tensor, buffer_type):
            return False
        return True

    def __enter__(self) -> "BoundedPinnedSaveOnCPU":
        if self._hooks is not None:
            raise RuntimeError("bounded offload context is already active")
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self._hooks.__enter__()
        self._status = "active"
        return self

    def __exit__(self, *args: Any) -> Any:
        if self._hooks is None:
            raise RuntimeError("bounded offload context was not entered")
        try:
            return self._hooks.__exit__(*args)
        finally:
            self._hooks = None
            self._status = "failed" if args and args[0] is not None else "completed"
