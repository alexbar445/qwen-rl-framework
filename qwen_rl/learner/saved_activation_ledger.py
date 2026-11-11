"""Inventory saved tensors and safely share exact-source tickets.

This layer decorates an existing saved-tensor policy. It never owns the source
Tensor: source identities are held through weak references and StorageWeakRef.
The wrapped policy ticket remains the only owner of any CPU payload/lease.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import weakref
from typing import Any
from weakref import WeakValueDictionary

import torch
from torch.multiprocessing.reductions import StorageWeakRef


@dataclass(frozen=True)
class _ObjectKey:
    object_id: int
    storage: StorageWeakRef
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    version: int


@dataclass(frozen=True)
class _ViewKey:
    storage: StorageWeakRef
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    version: int


@dataclass(frozen=True)
class _SourceSnapshot:
    object_id: int
    source_ref: weakref.ReferenceType[torch.Tensor] | None
    storage_ref: StorageWeakRef | None
    storage_bytes: int
    version: int | None
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    logical_bytes: int
    cache_eligible: bool

    @property
    def object_key(self) -> _ObjectKey | None:
        if self.version is None or not self.cache_eligible or self.storage_ref is None:
            return None
        return _ObjectKey(
            self.object_id, self.storage_ref, self.storage_offset, self.shape,
            self.stride, self.dtype, self.device, self.layout, self.version,
        )

    @property
    def view_key(self) -> _ViewKey | None:
        if self.version is None or not self.cache_eligible or self.storage_ref is None:
            return None
        return _ViewKey(
            self.storage_ref, self.storage_offset, self.shape, self.stride,
            self.dtype, self.device, self.layout, self.version,
        )


@dataclass(frozen=True)
class _PayloadFootprint:
    storage: StorageWeakRef
    storage_bytes: int
    # None means this payload keeps the entire CPU Storage live. A span is an
    # arena lease, for which only the reserved range is live activation data.
    span: tuple[int, int] | None
    logical_bytes: int


@dataclass
class _LivePayloadStorage:
    storage: StorageWeakRef
    storage_bytes: int
    whole_tokens: set[tuple[int, int]] = field(default_factory=set)
    spans: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)

    def live_bytes(self) -> int:
        if self.whole_tokens:
            return self.storage_bytes
        ranges = sorted((start, start + length) for start, length in self.spans.values())
        if not ranges:
            return 0
        total = 0
        start, end = ranges[0]
        for next_start, next_end in ranges[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                total += end - start
                start, end = next_start, next_end
        return total + end - start


class _LedgerTicket:
    """The autograd ticket: source is weak, native policy ticket is preserved."""

    __slots__ = (
        "payload", "source_ref", "source_storage_ref", "source_key",
        "source_version", "source_metadata", "_ledger", "_payload_token",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        payload: Any,
        source: _SourceSnapshot,
        ledger: "SavedActivationLedger",
        payload_token: int,
    ) -> None:
        self.payload = payload
        self.source_ref = source.source_ref
        self.source_storage_ref = source.storage_ref
        self.source_key = source.object_key
        self.source_version = source.version
        self.source_metadata = (
            source.storage_offset, source.shape, source.stride, source.dtype,
            source.device, source.layout,
        )
        self._ledger = ledger
        self._payload_token = payload_token

    def __del__(self) -> None:
        try:
            self._ledger._release_payload(self._payload_token)
        except BaseException:
            # Destructors must not replace an active exception during GC.
            pass


class SavedActivationLedger:
    """Count saved-tensor traffic and weakly deduplicate identical source objects.

    Exact sharing requires the same Python Tensor object, StorageImpl, version,
    offset, shape, stride, dtype, device, and layout. Other Python tensors that
    happen to present the same storage view are only counted as candidates.
    """

    def __init__(self, base_pack, base_unpack, *, deduplicate: bool = True) -> None:
        if type(deduplicate) is not bool:
            raise ValueError("deduplicate must be a boolean")
        self._base_pack = base_pack
        self._base_unpack = base_unpack
        self._deduplicate = deduplicate
        self._lock = threading.RLock()
        self._cache: WeakValueDictionary[_ObjectKey, _LedgerTicket] = WeakValueDictionary()
        self._view_sources: dict[_ViewKey, list[weakref.ReferenceType[torch.Tensor]]] = {}
        self._source_storages: dict[StorageWeakRef, tuple[int, int]] = {}
        self._source_objects: dict[
            int, tuple[weakref.ReferenceType[torch.Tensor], StorageWeakRef, int]
        ] = {}
        self._source_generation = 0
        self._source_live_count = 0
        self._source_live_bytes = 0
        self._payload_storages: dict[StorageWeakRef, _LivePayloadStorage] = {}
        self._payload_tokens: dict[int, list[tuple[StorageWeakRef, tuple[int, int]]]] = {}
        self._live_ticket_tokens: set[int] = set()
        self._next_ticket_token = 0
        self._closed = False
        self._status = "ready"
        self._stats: dict[str, Any] = {
            "pack_calls": 0,
            "unpack_calls": 0,
            "late_pack_calls": 0,
            "pack_errors": 0,
            "unpack_errors": 0,
            "logical_saved_bytes": 0,
            "logical_saved_bytes_by_device": {},
            "logical_saved_bytes_by_dtype": {},
            "eligible_pack_calls": 0,
            "eligible_logical_bytes": 0,
            "same_object_dedup_hits": 0,
            "same_object_dedup_logical_bytes": 0,
            "same_object_cuda_logical_bytes_avoided": 0,
            "dedup_cpu_copy_payload_bytes_avoided": 0,
            "same_object_cpu_logical_bytes_reused": 0,
            "cross_object_exact_view_candidates": 0,
            "cross_object_candidate_logical_bytes": 0,
            "version_unavailable_count": 0,
            "unsupported_metadata_count": 0,
            "native_cpu_payload_ticket_count": 0,
            "arena_cpu_payload_ticket_count": 0,
            "regeneration_ticket_count": 0,
            "marker_ticket_count": 0,
            "opaque_ticket_count": 0,
            "cpu_payload_logical_bytes_total": 0,
            "cpu_payload_unique_bytes_observed": 0,
            "cpu_payload_peak_storage_count": 0,
            "cpu_payload_peak_bytes": 0,
            "source_peak_storage_count": 0,
            "source_peak_storage_bytes": 0,
        }

    @staticmethod
    def _version(tensor: torch.Tensor) -> int | None:
        try:
            return int(tensor._version)
        except RuntimeError:
            return None

    @staticmethod
    def _object_id(tensor: torch.Tensor) -> int:
        return id(tensor)

    def _snapshot(self, tensor: torch.Tensor) -> _SourceSnapshot | None:
        try:
            source_ref = weakref.ref(tensor)
        except TypeError:
            return None
        try:
            storage = tensor.untyped_storage()
            storage_ref = StorageWeakRef(storage)
            storage_bytes = int(storage.nbytes())
        except (AttributeError, RuntimeError, TypeError):
            return None
        logical_bytes = int(tensor.numel()) * int(tensor.element_size())
        version = self._version(tensor)
        try:
            shape = tuple(int(value) for value in tensor.shape)
            stride = tuple(int(value) for value in tensor.stride())
            storage_offset = int(tensor.storage_offset())
            dtype = tensor.dtype
            device = tensor.device
            layout = tensor.layout
        except (RuntimeError, TypeError, AttributeError):
            return None
        cache_eligible = (
            type(tensor) is torch.Tensor
            and tensor.layout == torch.strided
            and not tensor.is_quantized
            and tensor.device.type != "meta"
            and tensor.numel() > 0
            and version is not None
        )
        return _SourceSnapshot(
            object_id=self._object_id(tensor),
            source_ref=source_ref,
            storage_ref=storage_ref,
            storage_bytes=storage_bytes,
            version=version,
            storage_offset=storage_offset,
            shape=shape,
            stride=stride,
            dtype=dtype,
            device=device,
            layout=layout,
            logical_bytes=logical_bytes,
            cache_eligible=cache_eligible,
        )

    def _record_source(self, snapshot: _SourceSnapshot) -> None:
        if snapshot.source_ref is None or snapshot.storage_ref is None:
            return
        tensor = snapshot.source_ref()
        if tensor is None:
            return
        with self._lock:
            object_id = snapshot.object_id
            existing = self._source_objects.get(object_id)
            if existing is not None:
                existing_ref, existing_storage, _generation = existing
                existing_tensor = existing_ref()
                if existing_tensor is tensor and existing_storage == snapshot.storage_ref:
                    return
                # This also handles Python id reuse if an old weakref callback
                # has not run yet, and a Tensor whose set_ changed StorageImpl.
                self._source_objects.pop(object_id, None)
                self._drop_source_storage_locked(existing_storage)

            self._source_generation += 1
            generation = self._source_generation
            ledger_ref = weakref.ref(self)
            storage_ref = snapshot.storage_ref

            def source_released(dead_ref, *, oid=object_id, storage=storage_ref,
                                gen=generation, owner_ref=ledger_ref):
                owner = owner_ref()
                if owner is not None:
                    owner._source_released(oid, dead_ref, storage, gen)

            tracked_ref = weakref.ref(tensor, source_released)
            self._source_objects[object_id] = (tracked_ref, storage_ref, generation)
            bucket = self._source_storages.get(storage_ref)
            if bucket is None:
                self._source_storages[storage_ref] = (snapshot.storage_bytes, 1)
                self._source_live_count += 1
                self._source_live_bytes += snapshot.storage_bytes
            else:
                self._source_storages[storage_ref] = (bucket[0], bucket[1] + 1)
            self._stats["source_peak_storage_count"] = max(
                self._stats["source_peak_storage_count"], self._source_live_count
            )
            self._stats["source_peak_storage_bytes"] = max(
                self._stats["source_peak_storage_bytes"], self._source_live_bytes
            )

    def _source_released(
        self,
        object_id: int,
        dead_ref: weakref.ReferenceType[torch.Tensor],
        storage_ref: StorageWeakRef,
        generation: int,
    ) -> None:
        with self._lock:
            current = self._source_objects.get(object_id)
            if current is None or current[0] is not dead_ref or current[2] != generation:
                return
            self._source_objects.pop(object_id, None)
            self._drop_source_storage_locked(storage_ref)

    def _drop_source_storage_locked(self, storage_ref: StorageWeakRef) -> None:
        bucket = self._source_storages.get(storage_ref)
        if bucket is None:
            return
        if bucket[1] > 1:
            self._source_storages[storage_ref] = (bucket[0], bucket[1] - 1)
            return
        self._source_storages.pop(storage_ref, None)
        self._source_live_count -= 1
        self._source_live_bytes -= bucket[0]

    def _record_candidate(self, snapshot: _SourceSnapshot) -> None:
        key = snapshot.view_key
        if key is None or snapshot.source_ref is None:
            return
        with self._lock:
            refs = self._view_sources.setdefault(key, [])
            alive: list[weakref.ReferenceType[torch.Tensor]] = []
            seen_same = False
            seen_other = False
            for ref in refs:
                source = ref()
                if source is None:
                    continue
                alive.append(ref)
                if source is snapshot.source_ref():
                    seen_same = True
                else:
                    seen_other = True
            if seen_other:
                self._stats["cross_object_exact_view_candidates"] += 1
                self._stats["cross_object_candidate_logical_bytes"] += snapshot.logical_bytes
            if not seen_same:
                alive.append(snapshot.source_ref)
            self._view_sources[key] = alive

    def _record_input(self, tensor: torch.Tensor, snapshot: _SourceSnapshot | None) -> None:
        with self._lock:
            self._stats["pack_calls"] += 1
            if self._closed:
                self._stats["late_pack_calls"] += 1
            logical_bytes = int(tensor.numel()) * int(tensor.element_size())
            device = str(tensor.device)
            dtype = str(tensor.dtype)
            self._stats["logical_saved_bytes"] += logical_bytes
            device_bytes = self._stats["logical_saved_bytes_by_device"]
            device_bytes[device] = device_bytes.get(device, 0) + logical_bytes
            dtype_bytes = self._stats["logical_saved_bytes_by_dtype"]
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + logical_bytes
            if snapshot is None:
                self._stats["unsupported_metadata_count"] += 1
                return
            self._record_source(snapshot)
            if snapshot.version is None:
                self._stats["version_unavailable_count"] += 1
            if snapshot.cache_eligible:
                self._stats["eligible_pack_calls"] += 1
                self._stats["eligible_logical_bytes"] += snapshot.logical_bytes
                self._record_candidate(snapshot)

    def pack(self, tensor: torch.Tensor) -> Any:
        snapshot = self._snapshot(tensor)
        self._record_input(tensor, snapshot)
        key = None if snapshot is None else snapshot.object_key
        if key is not None and self._deduplicate and not self._closed:
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    if (
                        cached.source_ref is not None
                        and cached.source_ref() is tensor
                        and cached.source_storage_ref is not None
                        and not cached.source_storage_ref.expired()
                        and cached.source_key == key
                        and cached.source_version == snapshot.version
                        and cached.source_metadata == (
                            snapshot.storage_offset, snapshot.shape, snapshot.stride,
                            snapshot.dtype, snapshot.device, snapshot.layout,
                        )
                    ):
                        self._stats["same_object_dedup_hits"] += 1
                        self._stats["same_object_dedup_logical_bytes"] += snapshot.logical_bytes
                        if snapshot.device.type == "cuda":
                            self._stats["same_object_cuda_logical_bytes_avoided"] += snapshot.logical_bytes
                            payload_category, _ = self._classify_payload(cached.payload)
                            if payload_category in {"native", "arena"}:
                                self._stats["dedup_cpu_copy_payload_bytes_avoided"] += snapshot.logical_bytes
                        elif snapshot.device.type == "cpu":
                            self._stats["same_object_cpu_logical_bytes_reused"] += snapshot.logical_bytes
                        return cached
                    self._cache.pop(key, None)
        try:
            payload = self._base_pack(tensor)
        except BaseException:
            with self._lock:
                self._stats["pack_errors"] += 1
            raise
        with self._lock:
            self._next_ticket_token += 1
            token = self._next_ticket_token
        ticket = _LedgerTicket(
            payload=payload, source=(snapshot or self._fallback_snapshot(tensor)),
            ledger=self, payload_token=token,
        )
        self._register_ticket(ticket, payload)
        after_version = self._version(tensor)
        if key is not None and self._deduplicate and not self._closed and after_version == snapshot.version:
            with self._lock:
                self._cache[key] = ticket
        return ticket

    def _fallback_snapshot(self, tensor: torch.Tensor) -> _SourceSnapshot:
        # Unsupported tensor/storage implementations are still delegated and
        # accounted logically, but cannot participate in storage-based reuse.
        try:
            source_ref = weakref.ref(tensor)
        except TypeError:
            source_ref = None
        try:
            version = self._version(tensor)
            shape = tuple(int(v) for v in tensor.shape)
            stride = tuple(int(v) for v in tensor.stride())
            storage_offset = int(tensor.storage_offset())
            dtype, device, layout = tensor.dtype, tensor.device, tensor.layout
        except (RuntimeError, TypeError, AttributeError):
            version, shape, stride, storage_offset = None, (), (), 0
            dtype, device, layout = torch.uint8, torch.device("meta"), torch.strided
        return _SourceSnapshot(
            self._object_id(tensor), source_ref, None, 0, version,
            storage_offset, shape, stride, dtype, device, layout,
            int(tensor.numel()) * int(tensor.element_size()), False,
        )

    def _classify_payload(self, payload: Any) -> tuple[str, list[_PayloadFootprint]]:
        cpu_tensor = None
        lease = None
        if isinstance(payload, tuple) and len(payload) >= 2 and isinstance(payload[1], torch.Tensor):
            if payload[1].device.type == "cpu":
                cpu_tensor = payload[1]
        else:
            candidate = getattr(payload, "cpu_tensor", None)
            if isinstance(candidate, torch.Tensor) and candidate.device.type == "cpu":
                cpu_tensor = candidate
                lease = getattr(payload, "lease", None)
        if cpu_tensor is not None:
            try:
                storage = cpu_tensor.untyped_storage()
                storage_ref = StorageWeakRef(storage)
                storage_bytes = int(storage.nbytes())
                logical_bytes = int(cpu_tensor.numel()) * int(cpu_tensor.element_size())
                span = None
                if lease is not None:
                    offset = getattr(lease, "offset", None)
                    size = getattr(lease, "size", None)
                    if type(offset) is int and type(size) is int and offset >= 0 and size >= 0:
                        span = (offset, size)
                category = "arena" if span is not None else "native"
                return category, [_PayloadFootprint(storage_ref, storage_bytes, span, logical_bytes)]
            except (AttributeError, RuntimeError, TypeError):
                pass
        name = type(payload).__name__
        if name == "_HeadTicket" or (hasattr(payload, "head") and hasattr(payload, "buffers")):
            return "regeneration", []
        if name == "_FrozenWeightTicket":
            return "marker", []
        return "opaque", []

    def _register_ticket(self, ticket: _LedgerTicket, payload: Any) -> None:
        category, footprints = self._classify_payload(payload)
        with self._lock:
            token = ticket._payload_token
            if token in self._live_ticket_tokens:
                raise RuntimeError("saved activation ledger ticket token collision")
            self._live_ticket_tokens.add(token)
            if category == "native":
                self._stats["native_cpu_payload_ticket_count"] += 1
            elif category == "arena":
                self._stats["arena_cpu_payload_ticket_count"] += 1
            elif category == "regeneration":
                self._stats["regeneration_ticket_count"] += 1
            elif category == "marker":
                self._stats["marker_ticket_count"] += 1
            else:
                self._stats["opaque_ticket_count"] += 1
            if footprints:
                self._stats["cpu_payload_logical_bytes_total"] += sum(
                    footprint.logical_bytes for footprint in footprints
                )
                payload_entries: list[tuple[StorageWeakRef, tuple[int, int]]] = []
                before = self._payload_live_locked()[1]
                for index, footprint in enumerate(footprints):
                    entry_token = (token, index)
                    bucket = self._payload_storages.get(footprint.storage)
                    if bucket is None:
                        bucket = _LivePayloadStorage(
                            footprint.storage, footprint.storage_bytes
                        )
                        self._payload_storages[footprint.storage] = bucket
                    if footprint.span is None:
                        bucket.whole_tokens.add(entry_token)
                    else:
                        bucket.spans[entry_token] = footprint.span
                    payload_entries.append((footprint.storage, entry_token))
                self._payload_tokens[token] = payload_entries
                after_count, after_bytes = self._payload_live_locked()
                self._stats["cpu_payload_unique_bytes_observed"] += max(0, after_bytes - before)
                self._stats["cpu_payload_peak_storage_count"] = max(
                    self._stats["cpu_payload_peak_storage_count"], after_count
                )
                self._stats["cpu_payload_peak_bytes"] = max(
                    self._stats["cpu_payload_peak_bytes"], after_bytes
                )

    def _release_payload(self, token: int) -> None:
        with self._lock:
            if token not in self._live_ticket_tokens:
                return
            self._live_ticket_tokens.remove(token)
            entries = self._payload_tokens.pop(token, [])
            for storage_ref, entry_token in entries:
                bucket = self._payload_storages.get(storage_ref)
                if bucket is None:
                    continue
                bucket.whole_tokens.discard(entry_token)
                bucket.spans.pop(entry_token, None)
                if not bucket.whole_tokens and not bucket.spans:
                    self._payload_storages.pop(storage_ref, None)

    def unpack(self, packed: Any) -> torch.Tensor:
        if isinstance(packed, _LedgerTicket) and packed._ledger is self:
            payload = packed.payload
        else:
            payload = packed
        with self._lock:
            self._stats["unpack_calls"] += 1
        try:
            return self._base_unpack(payload)
        except BaseException:
            with self._lock:
                self._stats["unpack_errors"] += 1
            raise

    def _source_live_locked(self) -> tuple[int, int]:
        # Tensor weakref callbacks maintain counts on release. A final scan also
        # catches storage metadata changes made without another saved-tensor pack.
        stale_ids = [
            object_id
            for object_id, (source_ref, storage_ref, _generation) in self._source_objects.items()
            if source_ref() is None or storage_ref.expired()
        ]
        for object_id in stale_ids:
            current = self._source_objects.pop(object_id, None)
            if current is not None:
                self._drop_source_storage_locked(current[1])
        return self._source_live_count, self._source_live_bytes

    def _payload_live_locked(self) -> tuple[int, int]:
        dead = [storage for storage in self._payload_storages if storage.expired()]
        for storage in dead:
            self._payload_storages.pop(storage, None)
        live_bytes = sum(bucket.live_bytes() for bucket in self._payload_storages.values())
        return len(self._payload_storages), live_bytes

    def close(self, *, failed: bool = False) -> None:
        with self._lock:
            self._closed = True
            self._status = "failed" if failed else "completed"
            self._cache.clear()
            self._view_sources.clear()

    @property
    def report(self) -> dict[str, Any]:
        with self._lock:
            source_count, source_bytes = self._source_live_locked()
            payload_count, payload_bytes = self._payload_live_locked()
            return {
                "status": self._status,
                **self._stats,
                "source_live_unique_storage_count": source_count,
                "source_live_unique_storage_bytes": source_bytes,
                "cpu_payload_live_storage_count": payload_count,
                "cpu_payload_live_bytes": payload_bytes,
                "live_ticket_count": len(self._live_ticket_tokens),
                "cache_entries": len(self._cache),
                "closed": self._closed,
                "deduplicate_enabled": self._deduplicate,
                "cpu_payload_live_bytes_scope": (
                    "unique live CPU payload storage bytes or arena lease spans; "
                    "does not attribute shared host PSS or allocator-resident memory"
                ),
                "dedup_cpu_copy_payload_bytes_avoided_scope": (
                    "logical saved payload bytes skipped for CUDA sources whose cached ticket "
                    "contains a native CPU payload or arena lease; regen/marker and CPU-source "
                    "hits are zero; this is pack traffic, not physical RAM or PSS saved"
                ),
                "dedup_logical_bytes_scope": (
                    "saved tensor logical bytes; they do not by themselves measure additional "
                    "CPU RAM, shared PSS, or allocator-resident bytes"
                ),
            }


class LedgerSavedTensorContext:
    """Decorate an existing saved-tensor context without nesting hook pairs.

    The original context owns hook installation/cleanup. Its callback attributes
    are temporarily replaced with ledger callbacks, so the wrapped policy still
    receives and returns its own native tickets.
    """

    def __init__(self, inner_context: Any, *, deduplicate: bool = True) -> None:
        if type(deduplicate) is not bool:
            raise ValueError("deduplicate must be a boolean")
        self.inner_context = inner_context
        self._deduplicate = deduplicate
        self._target, self._pack_attr, self._unpack_attr = self._resolve_callbacks(inner_context)
        self.ledger = SavedActivationLedger(
            getattr(self._target, self._pack_attr),
            getattr(self._target, self._unpack_attr),
            deduplicate=deduplicate,
        )
        self._patch_active = False
        self._entered = False
        self._status = "ready"

    @classmethod
    def _resolve_callbacks(cls, context: Any) -> tuple[Any, str, str]:
        if callable(getattr(context, "pack_hook", None)) and callable(getattr(context, "unpack_hook", None)):
            return context, "pack_hook", "unpack_hook"
        if callable(getattr(context, "pack", None)) and callable(getattr(context, "unpack", None)):
            return context, "pack", "unpack"
        bounded = getattr(context, "bounded_offload", None)
        if bounded is not None and bounded is not context:
            return cls._resolve_callbacks(bounded)
        raise TypeError("inner saved-tensor context must expose pack/unpack callbacks")

    def pack(self, tensor: torch.Tensor) -> Any:
        return self.ledger.pack(tensor)

    def unpack(self, packed: Any) -> torch.Tensor:
        return self.ledger.unpack(packed)

    def __enter__(self) -> "LedgerSavedTensorContext":
        if self._entered:
            raise RuntimeError("ledger saved-tensor context is already entered")
        original_pack = getattr(self._target, self._pack_attr)
        original_unpack = getattr(self._target, self._unpack_attr)
        try:
            setattr(self._target, self._pack_attr, self.pack)
            setattr(self._target, self._unpack_attr, self.unpack)
            self._patch_active = True
            self.inner_context.__enter__()
        except BaseException:
            self._restore_callbacks(original_pack, original_unpack)
            self.ledger.close(failed=True)
            self._status = "failed"
            raise
        self._entered = True
        self._status = "active"
        self.ledger._status = "active"
        return self

    def _restore_callbacks(self, original_pack: Any, original_unpack: Any) -> None:
        if self._patch_active:
            setattr(self._target, self._pack_attr, original_pack)
            setattr(self._target, self._unpack_attr, original_unpack)
            self._patch_active = False

    def __exit__(self, *args: Any) -> Any:
        if not self._entered:
            raise RuntimeError("ledger saved-tensor context was not entered")
        original_pack = self.ledger._base_pack
        original_unpack = self.ledger._base_unpack
        active_error = bool(args and args[0] is not None)
        try:
            result = self.inner_context.__exit__(*args)
        except BaseException:
            self.ledger.close(failed=True)
            self._status = "failed"
            raise
        else:
            self.ledger.close(failed=active_error)
            self._status = "failed" if active_error else "completed"
            return result
        finally:
            self._restore_callbacks(original_pack, original_unpack)
            self._entered = False

    @property
    def bounded_offload(self) -> Any:
        return getattr(self.inner_context, "bounded_offload", None)

    @property
    def report(self) -> dict[str, Any]:
        base_report = getattr(self.inner_context, "report", None)
        if isinstance(base_report, dict):
            report = dict(base_report)
        else:
            report = {"status": self._status}
        bounded = self.bounded_offload
        if bounded is not None and "arena_capacity_bytes" not in report:
            report["bounded_offload"] = bounded.report
        report["saved_activation_ledger"] = self.ledger.report
        return report
