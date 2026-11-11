"""Bounded host streaming for immutable packed T2 decoder weights.

The offloader recognizes frozen packed T2 linears across primal and rotated
input bases, plus explicitly marked T2 TP base modules. It moves only uint8
packed codes and uint16 scale words; bias, Hadamard metadata, and trainable
LoRA parameters stay where the model placed them. CPU masters are ordinary
immutable-by-contract tensors. Each module
group receives a fresh CUDA lease, so a tensor saved by autograd is never
overwritten or recycled.  This is intended for per-layer non-reentrant
checkpointing, where checkpoint recomputation releases the original forward
leases before rebuilding them during backward.

For CuMem phase sharing, call :meth:`park` after backward and before
``CudaResidency.park``; call ``resume('forward')`` after the CuMem pool wakes.
Streamed leases use the normal CUDA allocator and are not part of the CuMem
pool.  The caller owns phase boundaries and must not park while retaining an
autograd graph.
"""
from __future__ import annotations

import threading
import time
import weakref
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Self

import torch
from torch import nn

from .ternary import is_t2_linear

Phase = Literal["forward", "backward", "idle"]
_PAYLOAD_DTYPES = {"packed_codes": torch.uint8, "scale_words": torch.uint16}
_ALIGNMENT = 16
_TRANSFER_DETAIL_LIMIT = 512


class TernaryTPOffloadError(RuntimeError):
    """A streaming lifecycle or packed-buffer invariant was violated."""


@dataclass(frozen=True, slots=True)
class PackedBufferEntry:
    module: nn.Module
    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class PackedGroupPlan:
    index: int
    owner: nn.Module
    entries: tuple[PackedBufferEntry, ...]
    payload_bytes: int
    storage_bytes: int


@dataclass(frozen=True, slots=True)
class PackedTernaryTPPlan:
    """Exact host/staging estimate built before any model buffers are changed."""

    groups: tuple[PackedGroupPlan, ...]
    device: torch.device
    cpu_master_bytes: int
    max_group_bytes: int
    payload_bytes: int

    @property
    def pinned_staging_bytes(self) -> int:
        return 2 * self.max_group_bytes

    @property
    def total_host_bytes(self) -> int:
        return self.cpu_master_bytes + self.pinned_staging_bytes


@dataclass(slots=True)
class _RuntimeEntry:
    plan: PackedBufferEntry
    cpu_view: torch.Tensor


@dataclass(slots=True)
class _RuntimeGroup:
    plan: PackedGroupPlan
    master: torch.Tensor
    entries: tuple[_RuntimeEntry, ...]
    version: int


@dataclass(slots=True)
class _Lease:
    flat: torch.Tensor
    ready: torch.cuda.Event


@dataclass(slots=True)
class _LeaseRecord:
    group_index: int
    nbytes: int
    views: tuple[weakref.ReferenceType[torch.Tensor], ...]


@dataclass(slots=True)
class _Transfer:
    group_index: int
    nbytes: int
    staging_cpu_seconds: float
    start: torch.cuda.Event
    end: torch.cuda.Event


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    if actual.type != expected.type:
        return False
    return expected.index is None or actual.index == expected.index


def _packed_payloads(module: nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    """Return only the supported frozen packed payload buffers for ``module``."""
    if is_t2_linear(module):
        payloads = (("packed_codes", module.packed_codes), ("scale_words", module.scale_words))
    elif getattr(module, "_t2_tp_base", False) is True:
        iterator = getattr(module, "iter_packed_weight_buffers", None)
        if not callable(iterator):
            raise TypeError("marked T2 TP base must implement iter_packed_weight_buffers()")
        payloads = tuple(iterator())
    else:
        return ()

    seen: set[str] = set()
    normalized: list[tuple[str, torch.Tensor]] = []
    for item in payloads:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("packed buffer iterator must yield (name, tensor) pairs")
        name, tensor = item
        if name not in _PAYLOAD_DTYPES or name in seen:
            raise ValueError("T2 offload accepts exactly one packed_codes and one scale_words buffer")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype != _PAYLOAD_DTYPES[name]:
            raise TypeError(f"{name} must use dtype {_PAYLOAD_DTYPES[name]}, got {tensor.dtype}")
        registered = module._buffers.get(name)
        if registered is not tensor:
            raise ValueError(f"{name} must be the registered module buffer returned by the iterator")
        if tensor.requires_grad:
            raise ValueError(f"{name} must be frozen")
        if tensor.layout is not torch.strided or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous strided tensor")
        if tensor.numel() <= 0 or any(int(dim) <= 0 for dim in tensor.shape):
            raise ValueError(f"{name} must have nonempty dimensions")
        seen.add(name)
        normalized.append((name, tensor))
    if seen != set(_PAYLOAD_DTYPES):
        raise ValueError("each T2 base must expose packed_codes and scale_words")
    return tuple(sorted(normalized, key=lambda pair: pair[0]))


def build_packed_weight_plan(
    groups: Iterable[nn.Module], *, device: torch.device | str | None = None
) -> PackedTernaryTPPlan:
    """Build a CPU-only structural plan and exact packed-host byte estimate.

    This function does not allocate pinned memory or CUDA storage and is safe
    for CPU-only structural checks.  ``groups`` are outer decoder layers or
    equivalent execution units whose forward hooks enclose all packed T2
    projections that should be leased together.
    """
    if isinstance(groups, (str, bytes)):
        raise TypeError("groups must be an iterable of torch modules, not text")
    try:
        group_modules = tuple(groups)
    except TypeError as exc:
        raise TypeError("groups must be an iterable of torch modules") from exc
    if not group_modules:
        raise ValueError("groups must be a nonempty sequence of decoder modules")
    if any(not isinstance(group, nn.Module) for group in group_modules):
        raise TypeError("every streaming group must be a torch module")

    expected_device = torch.device(device) if device is not None else None
    planned_groups: list[PackedGroupPlan] = []
    seen_buffers: set[tuple[int, str]] = set()
    seen_storages: set[tuple[str, int]] = set()
    inferred_device: torch.device | None = None
    payload_total = 0
    cpu_master_total = 0
    max_group_bytes = 0

    for index, owner in enumerate(group_modules):
        entries: list[PackedBufferEntry] = []
        offset = 0
        payload_bytes = 0
        for module in owner.modules():
            for name, tensor in _packed_payloads(module):
                key = (id(module), name)
                if key in seen_buffers:
                    raise ValueError("a packed T2 buffer cannot belong to multiple streaming groups")
                seen_buffers.add(key)

                if expected_device is not None and not _same_device(tensor.device, expected_device):
                    raise ValueError(
                        f"packed T2 buffer must be on {expected_device}; found {tensor.device}"
                    )
                if inferred_device is None:
                    inferred_device = tensor.device
                elif tensor.device != inferred_device:
                    raise ValueError(
                        f"packed T2 buffers must share device {inferred_device}; found {tensor.device}"
                    )
                storage = tensor.untyped_storage()
                storage_key = (str(tensor.device), int(storage.data_ptr()))
                if storage_key in seen_storages:
                    raise ValueError("packed T2 payload buffers must not share underlying storage")
                seen_storages.add(storage_key)

                offset = (offset + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
                nbytes = tensor.numel() * tensor.element_size()
                entries.append(PackedBufferEntry(
                    module=module,
                    name=name,
                    offset=offset,
                    nbytes=nbytes,
                    shape=tuple(int(dim) for dim in tensor.shape),
                    dtype=tensor.dtype,
                ))
                offset += nbytes
                payload_bytes += nbytes
        if not entries:
            raise ValueError(f"streaming group {index} contains no supported packed T2 buffers")
        group = PackedGroupPlan(index, owner, tuple(entries), payload_bytes, offset)
        planned_groups.append(group)
        payload_total += payload_bytes
        cpu_master_total += offset
        max_group_bytes = max(max_group_bytes, offset)

    if inferred_device is None:
        raise ValueError("the packed T2 plan could not infer a source device")
    return PackedTernaryTPPlan(
        groups=tuple(planned_groups),
        device=inferred_device,
        cpu_master_bytes=cpu_master_total,
        max_group_bytes=max_group_bytes,
        payload_bytes=payload_total,
    )


def _shared_masters_for_plan(
    master_provider: Any | None,
    plan: PackedTernaryTPPlan,
) -> tuple[torch.Tensor, ...] | None:
    """Validate and retain zero-copy group views supplied by a shared store."""
    if master_provider is None:
        return None
    required = ("group_tensor", "check_unchanged", "report")
    missing = [name for name in required if not callable(getattr(master_provider, name, None))]
    if missing:
        raise TypeError(f"shared packed master provider is missing methods: {missing}")
    expected_sizes = tuple(group.storage_bytes for group in plan.groups)
    if getattr(master_provider, "group_count", None) != len(plan.groups):
        raise ValueError("shared packed master group count differs from the decoder plan")
    if tuple(getattr(master_provider, "group_sizes", ())) != expected_sizes:
        raise ValueError("shared packed master group sizes differ from the decoder plan")
    master_provider.check_unchanged()
    masters = tuple(master_provider.group_tensor(group.index) for group in plan.groups)
    for group, master in zip(plan.groups, masters, strict=True):
        if (not isinstance(master, torch.Tensor) or master.device.type != "cpu"
                or master.dtype != torch.uint8 or master.ndim != 1
                or not master.is_contiguous() or master.numel() != group.storage_bytes):
            raise ValueError(
                f"shared packed master group {group.index} must be a flat "
                f"CPU uint8 view of {group.storage_bytes} bytes"
            )
    return masters


class PackedTernaryTPOffloader:
    """Stream packed T2 buffers for ordered decoder groups with bounded staging.

    ``max_cpu_master_bytes`` and ``max_pinned_bytes`` are mandatory preflight
    caps. Construction validates the full plan against both caps before
    allocating or replacing any model buffer. The only pinned allocations are
    two reusable staging slots, each no larger than the largest decoder group.

    Lifecycle contract:

    * ``set_phase('forward')`` before model forward and
      ``set_phase('backward')`` before backward. With non-reentrant per-layer
      checkpointing, layers are revisited in reverse order during recompute.
    * ``set_phase('idle')`` after backward. ``park()`` then proves there are no
      active hooks, in-flight copies, or autograd-owned leases before a caller
      parks a shared CuMem pool. Resume with ``resume('forward')``.
    * ``close()`` is allowed only after the graph has released all leases. It
      removes hooks and releases pinned staging. Packed module buffers remain
      CPU views into the ordinary CPU masters; it does not repopulate the full
      CUDA model.
    """

    def __init__(
        self,
        groups: Iterable[nn.Module],
        *,
        device: torch.device | str,
        max_cpu_master_bytes: int,
        max_pinned_bytes: int,
        prefetch: bool = True,
        master_provider: Any | None = None,
    ) -> None:
        if type(max_cpu_master_bytes) is not int or max_cpu_master_bytes <= 0:
            raise ValueError("max_cpu_master_bytes must be a positive integer")
        if type(max_pinned_bytes) is not int or max_pinned_bytes <= 0:
            raise ValueError("max_pinned_bytes must be a positive integer")
        if type(prefetch) is not bool:
            raise TypeError("prefetch must be a bool")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("packed T2 streaming requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("packed T2 streaming requires CUDA to be available")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        group_modules = tuple(groups)
        plan = build_packed_weight_plan(group_modules, device=self.device)
        if plan.cpu_master_bytes > max_cpu_master_bytes:
            raise ValueError(
                "CPU master allocation exceeds explicit cap "
                f"({plan.cpu_master_bytes} > {max_cpu_master_bytes} bytes)"
            )
        if plan.pinned_staging_bytes > max_pinned_bytes:
            raise ValueError(
                "two pinned staging slots exceed explicit cap "
                f"({plan.pinned_staging_bytes} > {max_pinned_bytes} bytes)"
            )

        shared_masters = _shared_masters_for_plan(master_provider, plan)

        self._plan: PackedTernaryTPPlan | None = plan
        self._max_cpu_master_bytes = max_cpu_master_bytes
        self._max_pinned_bytes = max_pinned_bytes
        self._prefetch_enabled = prefetch
        # Keep the file-backed owner alive while installed buffers still refer
        # to its mmap. close() must not explicitly unmap those tensor views.
        self._master_provider = master_provider
        self._groups: list[_RuntimeGroup] = []
        self._layers = group_modules
        self._hooks: list[Any] = []
        self._pending: dict[int, Future[_Lease]] = {}
        self._active: set[int] = set()
        self._active_views: dict[tuple[int, str], torch.Tensor] = {}
        self._lease_records: list[_LeaseRecord] = []
        # Keep bounded report detail only. Slot reuse and compute readiness use
        # their own event references in _slot_events and _Lease.
        self._transfers: deque[_Transfer] = deque(maxlen=_TRANSFER_DETAIL_LIMIT)
        self._transfer_count = 0
        self._transfer_bytes = 0
        self._lock = threading.Lock()
        self._recompute_lock = threading.RLock()
        self._wait_seconds = 0.0
        self._acquire_count = 0
        self._slot_events: list[torch.cuda.Event | None] = [None, None]
        self._slot_cursor = 0
        self._closed = False
        self._parked = False
        self._phase: Phase = "idle"
        self._closed_report: dict[str, Any] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._slots: list[torch.Tensor] = []

        originals: list[tuple[nn.Module, str, torch.Tensor]] = []
        try:
            # Allocate and populate all pageable CPU masters first. If host
            # allocation fails, the live model has not yet been changed.
            for group_plan in plan.groups:
                master = (
                    shared_masters[group_plan.index]
                    if shared_masters is not None
                    else torch.empty(group_plan.storage_bytes, dtype=torch.uint8, device="cpu")
                )
                runtime_entries: list[_RuntimeEntry] = []
                for entry in group_plan.entries:
                    source = entry.module._buffers[entry.name]
                    if source.device != self.device:
                        raise ValueError(
                            f"{entry.name} moved after planning: expected {self.device}, found {source.device}"
                        )
                    originals.append((entry.module, entry.name, source))
                    if master_provider is None:
                        raw = source.detach().view(torch.uint8).reshape(-1)
                        master.narrow(0, entry.offset, entry.nbytes).copy_(raw)
                    cpu_view = self._view(master, entry)
                    runtime_entries.append(_RuntimeEntry(entry, cpu_view))
                self._groups.append(_RuntimeGroup(
                    group_plan, master, tuple(runtime_entries), master._version
                ))

            # Pinned allocation is exact and fail-closed; there is no pageable
            # or smaller-slot fallback that could silently change the cap.
            max_group = plan.max_group_bytes
            self._slots = [
                torch.empty(max_group, dtype=torch.uint8, device="cpu", pin_memory=True)
                for _ in range(2)
            ]
            self._slot_events = [None, None]
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t2-tp-offload")

            # Install CPU master views transactionally. The original GPU
            # tensors remain in ``originals`` only until all hooks are ready.
            for group in self._groups:
                for entry in group.entries:
                    entry.plan.module._buffers[entry.plan.name] = entry.cpu_view
            for index, layer in enumerate(self._layers):
                self._hooks.append(layer.register_forward_pre_hook(self._pre_hook(index)))
                self._hooks.append(layer.register_forward_hook(
                    self._post_hook(index), always_call=True
                ))
        except BaseException:
            for handle in self._hooks:
                handle.remove()
            for module, name, source in originals:
                module._buffers[name] = source
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            self._slots.clear()
            self._groups.clear()
            raise

        # Release the constructor's temporary references to the original CUDA
        # payloads now that each module points at its immutable CPU master.
        originals.clear()

    @staticmethod
    def _view(flat: torch.Tensor, entry: PackedBufferEntry) -> torch.Tensor:
        return flat.narrow(0, entry.offset, entry.nbytes).view(entry.dtype).reshape(entry.shape)

    def _check_open(self) -> None:
        if self._closed:
            raise TernaryTPOffloadError("packed T2 offloader is closed")

    def _check_master(self, group: _RuntimeGroup) -> None:
        """Fail if the selected immutable CPU master changed after planning."""
        if group.master._version != group.version:
            raise TernaryTPOffloadError(
                f"CPU packed master for group {group.plan.index} was mutated"
            )

    def _check_masters(self) -> None:
        """Audit every master at lifecycle boundaries, outside the copy hot path."""
        for group in self._groups:
            self._check_master(group)
        self._assert_cpu_views_installed()
        self._check_provider()

    def _check_provider(self) -> None:
        if self._master_provider is not None:
            self._master_provider.check_unchanged()

    def _assert_cpu_views_installed(self) -> None:
        """Fail before compute if a later model.to() rematerialized packed buffers."""
        for group in self._groups:
            for runtime_entry in group.entries:
                entry = runtime_entry.plan
                if entry.module._buffers.get(entry.name) is not runtime_entry.cpu_view:
                    raise TernaryTPOffloadError(
                        f"{entry.name} was moved or replaced after packed offload installation"
                    )

    def _copy_group(self, index: int) -> _Lease:
        self._check_open()
        if self._copy_stream is None or not self._slots:
            raise TernaryTPOffloadError("CUDA staging resources are unavailable")
        self._check_provider()
        group = self._groups[index]
        with torch.cuda.device(self.device):
            # Only this group is about to be copied. Full-model version scans
            # belong at lifecycle boundaries, not once per layer transfer.
            self._check_master(group)
            slot_index = self._slot_cursor % 2
            self._slot_cursor += 1
            previous = self._slot_events[slot_index]
            if previous is not None:
                previous.synchronize()

            stage = self._slots[slot_index]
            start_cpu = time.perf_counter()
            self._copy_master_to_stage(group, stage)
            staging_seconds = time.perf_counter() - start_cpu

            with torch.cuda.stream(self._copy_stream):
                flat = torch.empty_like(group.master, device=self.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(self._copy_stream)
                flat.copy_(stage[: group.master.numel()], non_blocking=True)
                end.record(self._copy_stream)
            self._slot_events[slot_index] = end
            self._record_transfer(_Transfer(
                index, group.master.numel(), staging_seconds, start, end
            ))
            return _Lease(flat, end)

    def _record_transfer(self, transfer: _Transfer) -> None:
        """Retain recent detail while preserving lifetime transfer totals."""
        with self._lock:
            self._transfers.append(transfer)
            self._transfer_count += 1
            self._transfer_bytes += transfer.nbytes

    def _copy_master_to_stage(self, group: _RuntimeGroup, stage: torch.Tensor) -> None:
        """Copy one immutable master and recheck its identity before H2D enqueue."""
        stage[: group.master.numel()].copy_(group.master)
        # Bracket the host copy so a changed mmap cannot become a mixed staging
        # snapshot and proceed into a CUDA lease.
        self._check_provider()
        self._check_master(group)

    def _prefetch(self, index: int) -> None:
        if not self._prefetch_enabled or not 0 <= index < len(self._groups):
            return
        if index in self._pending or index in self._active:
            return
        if self._executor is None:
            raise TernaryTPOffloadError("packed T2 prefetch executor is unavailable")
        self._pending[index] = self._executor.submit(self._copy_group, index)

    def _pre_hook(self, index: int):
        def enter(module: nn.Module, args: tuple[Any, ...]) -> None:
            self._check_open()
            if self._parked:
                raise TernaryTPOffloadError("packed T2 offloader is parked")
            if self._phase not in ("forward", "backward"):
                raise TernaryTPOffloadError(
                    "set_phase('forward'|'backward') before executing a decoder group"
                )
            if index in self._active:
                raise TernaryTPOffloadError(f"nested lease for decoder group {index}")
            self._prefetch(index)
            future = self._pending.pop(index, None)
            if future is None:
                if self._executor is None:
                    raise TernaryTPOffloadError("packed T2 prefetch executor is unavailable")
                future = self._executor.submit(self._copy_group, index)
            started = time.perf_counter()
            lease = future.result()
            self._wait_seconds += time.perf_counter() - started
            self._acquire_count += 1

            compute_stream = torch.cuda.current_stream(self.device)
            compute_stream.wait_event(lease.ready)
            # All typed views below share this fresh storage. Recording the
            # storage on the consumer stream prevents allocator reuse before
            # the compute stream has finished using it.
            lease.flat.record_stream(compute_stream)
            group = self._groups[index]
            views: list[torch.Tensor] = []
            try:
                for runtime_entry in group.entries:
                    entry = runtime_entry.plan
                    current = entry.module._buffers.get(entry.name)
                    if current is not runtime_entry.cpu_view:
                        raise TernaryTPOffloadError(
                            f"{entry.name} changed outside the packed T2 offloader"
                        )
                    view = self._view(lease.flat, entry)
                    entry.module._buffers[entry.name] = view
                    views.append(view)
            except BaseException:
                for runtime_entry in group.entries:
                    runtime_entry.plan.module._buffers[runtime_entry.plan.name] = runtime_entry.cpu_view
                raise

            self._lease_records.append(_LeaseRecord(
                index, group.plan.storage_bytes, tuple(weakref.ref(view) for view in views)
            ))
            for runtime_entry, view in zip(group.entries, views, strict=True):
                entry = runtime_entry.plan
                self._active_views[(id(entry.module), entry.name)] = view
            self._active.add(index)
            direction = 1 if self._phase == "forward" else -1
            self._prefetch(index + direction)

        return enter

    def _post_hook(self, index: int):
        def leave(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            if index not in self._active:
                return
            group = self._groups[index]
            try:
                for runtime_entry in group.entries:
                    entry = runtime_entry.plan
                    entry.module._buffers[entry.name] = runtime_entry.cpu_view
                    self._active_views.pop((id(entry.module), entry.name), None)
            finally:
                self._active.remove(index)

        return leave

    @contextmanager
    def recompute_group(self, index: int) -> Iterator[None]:
        """Temporarily lease one packed group during nested backward recompute.

        If the ordinary decoder hook already owns this group, the nested
        recompute borrows that lease and leaves its lifetime to the outer hook.
        Otherwise, acquire and release through the same hooks used by normal
        decoder execution. Saved tensors may keep the CUDA lease alive after
        the module buffers have been restored to their CPU views.
        """
        # Autograd may visit sibling checkpoint nodes on different threads.
        # Serialize their Python lease scopes; a same-thread nested context
        # can still borrow its outer lease without releasing that owner.
        with self._recompute_lock:
            self._check_open()
            if type(index) is not int:
                raise TypeError("decoder group index must be a built-in int")
            if index < 0 or index >= len(self._groups):
                raise IndexError(f"decoder group index out of range: {index}")
            if self._parked:
                raise TernaryTPOffloadError("packed T2 offloader is parked")
            if self._phase != "backward":
                raise TernaryTPOffloadError("recompute_group is only allowed during backward phase")

            group = self._groups[index]
            owner = group.plan.owner
            if index in self._active:
                yield
                return

            pre_hook = self._pre_hook(index)
            post_hook = self._post_hook(index)
            try:
                pre_hook(owner, ())
            except BaseException:
                # A pre-hook can fail after installing its views (for example, if
                # prefetch scheduling fails). Retire that partial lease as well.
                if index in self._active:
                    post_hook(owner, (), None)
                raise
            try:
                yield
            finally:
                post_hook(owner, (), None)

    def _drain_pending(self) -> None:
        pending, self._pending = self._pending, {}
        futures: list[Future[_Lease]] = []
        for future in pending.values():
            if not future.cancel():
                futures.append(future)
        if futures:
            # Wait for every started copy before releasing either pinned slot.
            wait(futures)
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
        first_error = next((future.exception() for future in futures if future.exception()), None)
        if first_error is not None:
            raise first_error

    def set_phase(self, phase: Phase | str) -> None:
        """Select forward-order, reverse-order, or idle hook prefetching."""
        self._check_open()
        if phase not in ("forward", "backward", "idle"):
            raise ValueError("phase must be 'forward', 'backward', or 'idle'")
        if self._parked:
            raise TernaryTPOffloadError("resume the parked offloader before changing phase")
        if self._active:
            raise TernaryTPOffloadError(
                f"cannot change phase with active decoder groups {sorted(self._active)}"
            )
        self._drain_pending()
        self._check_masters()
        self._phase = phase  # type: ignore[assignment]
        if phase == "forward":
            self._prefetch(0)
        elif phase == "backward":
            self._prefetch(len(self._groups) - 1)

    def _live_lease_totals(self) -> tuple[int, int]:
        live: list[_LeaseRecord] = []
        live_bytes = 0
        for record in self._lease_records:
            if any(reference() is not None for reference in record.views):
                live.append(record)
                live_bytes += record.nbytes
        self._lease_records = live
        return len(live), live_bytes

    def _assert_no_live_leases(self, operation: str) -> tuple[int, int]:
        groups, nbytes = self._live_lease_totals()
        if groups:
            raise TernaryTPOffloadError(
                f"cannot {operation} while autograd retains {groups} CUDA lease(s) "
                f"({nbytes} bytes); finish backward and release retained graphs first"
            )
        return groups, nbytes

    def park(self) -> dict[str, Any]:
        """Drain streaming and guard a CuMem park boundary.

        Call after backward and before ``CudaResidency.park``. If a graph still
        owns any packed CUDA view, the operation fails closed and leaves hooks
        in idle mode so the caller can finish/release that graph and retry.
        """
        self._check_open()
        if self._active:
            raise TernaryTPOffloadError(
                f"cannot park with active decoder groups {sorted(self._active)}"
            )
        self._assert_cpu_views_installed()
        if not self._parked:
            self.set_phase("idle")
        torch.cuda.synchronize(self.device)
        self._assert_no_live_leases("park")
        self._parked = True
        return self.report()

    def resume(self, phase: Literal["forward", "backward"] = "forward") -> None:
        """Resume hooks after the shared CuMem pool has returned awake."""
        self._check_open()
        if not self._parked:
            raise TernaryTPOffloadError("resume requires a parked packed T2 offloader")
        if phase not in ("forward", "backward"):
            raise ValueError("resume phase must be 'forward' or 'backward'")
        self._parked = False
        try:
            self.set_phase(phase)
        except BaseException:
            self._parked = True
            raise

    def _report_open(self) -> dict[str, Any]:
        plan = self._plan
        if plan is None:
            raise TernaryTPOffloadError("offload report is unavailable")
        with self._lock:
            transfer_rows = tuple(self._transfers)
            transfer_count = self._transfer_count
            transfer_bytes = self._transfer_bytes
        transfers = []
        for transfer in transfer_rows:
            transfer.end.synchronize()
            transfers.append({
                "group": transfer.group_index,
                "bytes": transfer.nbytes,
                "staging_cpu_seconds": transfer.staging_cpu_seconds,
                "h2d_cuda_ms": transfer.start.elapsed_time(transfer.end),
            })
        live_groups, live_bytes = self._live_lease_totals()
        module_buffers_match = all(
            runtime_entry.plan.module._buffers.get(runtime_entry.plan.name)
            is self._active_views.get(
                (id(runtime_entry.plan.module), runtime_entry.plan.name), runtime_entry.cpu_view
            )
            for group in self._groups
            for runtime_entry in group.entries
        )
        try:
            shared_master = (
                self._master_provider.report() if self._master_provider is not None else None
            )
            shared_master_integrity_error = None
        except BaseException as exc:
            shared_master = None
            shared_master_integrity_error = f"{type(exc).__name__}: {exc}"
        return {
            "groups": len(plan.groups),
            "phase": self._phase,
            "parked": self._parked,
            "closed": self._closed,
            "payload_bytes": plan.payload_bytes,
            "cpu_master_bytes": plan.cpu_master_bytes,
            "cpu_master_cap_bytes": self._max_cpu_master_bytes,
            "max_group_bytes": plan.max_group_bytes,
            "pinned_staging_bytes": plan.pinned_staging_bytes,
            "pinned_staging_cap_bytes": self._max_pinned_bytes,
            "total_host_bytes": plan.total_host_bytes,
            "max_fresh_gpu_lease_bytes": plan.max_group_bytes,
            "transfers": transfers,
            "transfer_count": transfer_count,
            "transfer_bytes": transfer_bytes,
            "transfer_totals_scope": "offloader_lifetime",
            "transfer_details_scope": "most_recent_offloader_lifetime",
            "transfer_details_limit": _TRANSFER_DETAIL_LIMIT,
            "transfer_details_truncated": transfer_count > len(transfer_rows),
            "cpu_future_wait_seconds": self._wait_seconds,
            "acquire_count": self._acquire_count,
            "pending_groups": sorted(self._pending),
            "active_groups": sorted(self._active),
            "live_autograd_lease_groups": live_groups,
            "live_autograd_lease_bytes": live_bytes,
            "cpu_masters_unmodified": all(
                group.master._version == group.version for group in self._groups
            ),
            "module_buffers_match_offloader": module_buffers_match,
            "executor_alive": self._executor is not None,
            "pinned_slots_allocated": len(self._slots),
            "shared_master": shared_master,
            "shared_master_integrity_error": shared_master_integrity_error,
        }

    def report(self) -> dict[str, Any]:
        """Return lifetime totals and bounded recent transfer details."""
        if self._closed:
            return dict(self._closed_report or {"closed": True})
        return self._report_open()

    def close(self) -> dict[str, Any]:
        """Remove hooks and release staging after all AD leases are gone."""
        if self._closed:
            return self.report()
        if self._active:
            raise TernaryTPOffloadError(
                f"cannot close with active decoder groups {sorted(self._active)}"
            )
        shared_master_error = None
        try:
            self._check_provider()
        except BaseException as exc:
            # Cleanup must still remove hooks and release staging after an
            # integrity failure; include the failure in the final report.
            shared_master_error = f"{type(exc).__name__}: {exc}"
        if not self._parked:
            # Close must still release resources if a master was mutated. Phase
            # transitions fail fast on all masters; final report keeps the
            # global mutation audit without making cleanup conditional on it.
            self._drain_pending()
            self._phase = "idle"
        torch.cuda.synchronize(self.device)
        self._assert_no_live_leases("close")

        # Make the final snapshot before dropping references to the model and
        # masters. CPU views stay installed in modules as their packed state.
        self._closed_report = self._report_open()
        self._closed_report["phase"] = "idle"
        self._closed_report["parked"] = self._parked
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
            self._copy_stream = None
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._slots.clear()
        self._slot_events.clear()
        self._pending.clear()
        self._active.clear()
        self._active_views.clear()
        self._lease_records.clear()
        self._groups.clear()
        self._layers = ()
        self._plan = None
        self._closed = True
        self._closed_report["closed"] = True
        self._closed_report["executor_alive"] = False
        self._closed_report["pinned_slots_allocated"] = 0
        if shared_master_error is not None:
            self._closed_report["shared_master_integrity_error"] = shared_master_error
        return self.report()

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
