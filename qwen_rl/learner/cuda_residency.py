"""Persistent CUDA residency using pinned vLLM 0.29 / Torch 2.13 pools.

Use Torch MemPool owner retirement to reclaim whole inactive segments through
its allocator callbacks. Active allocations retain their original addresses.
Current and retired pools share one CuMem registry and park/wake inventory;
never unmap a cached segment behind Torch's allocator metadata.
"""
from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
from importlib import metadata
import os
import gc
import threading
from typing import Any, Callable, Iterator, Mapping


PINNED_VLLM_VERSION = "0.29.0"
PINNED_TORCH_VERSION = "2.13.0"
DEFAULT_TAG = "default"
_ALLOCATOR_ENVIRONMENTS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


@contextmanager
def _measure_allocator_gc(timer):
    """Observe upstream full-GC time without changing collection policy."""
    thread_id = threading.get_ident()
    active = {}

    def observe(phase, info):
        if threading.get_ident() != thread_id:
            return
        generation = info['generation']
        if phase == 'start':
            active[generation] = timer.start(f'python_gc_generation_{generation}')
        elif generation in active:
            timer.finish(active.pop(generation))

    gc.callbacks.append(observe)
    try:
        yield
    finally:
        gc.callbacks.remove(observe)


def materialize_adam_state(optimizer) -> None:
    """Use pinned Torch's own lazy initializer without an optimizer step.

    Call during model construction inside the persistent pool. Temporary zero
    gradient metadata activates Adam's initializer, then is restored to None.
    No parameter, step counter, scheduler or random generator is advanced.
    """
    import torch
    if str(torch.__version__).split("+", 1)[0] != PINNED_TORCH_VERSION:
        raise CudaResidencyError("Adam materialization requires pinned Torch 2.13.0")
    if type(optimizer) is not torch.optim.AdamW:
        raise CudaResidencyError("phase_shared requires Trainer-owned torch.optim.AdamW")
    with torch.no_grad():
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if not parameter.requires_grad or parameter.grad is not None:
                    raise CudaResidencyError("initialize Adam before any gradients on trainable parameters only")
                if optimizer.state.get(parameter):
                    continue
                parameter.grad = torch.zeros_like(parameter)
                try:
                    optimizer._init_group({**group, "params": [parameter]}, *[[] for _ in range(6)])
                finally:
                    parameter.grad = None


class CudaResidencyError(RuntimeError):
    """The process cannot safely park or resume its CUDA allocation pool."""


class CudaResidencyBudgetError(CudaResidencyError):
    """The active CuMem allocations exceed the configured host backup budget."""


class CudaResidency:
    """Own one persistent CuMem pool and preserve its CUDA virtual addresses.

    Model construction, optimizer-state allocation and full-state restoration
    run inside allocation_context(); forward/backward temporaries use native CUDA. The caller drains asynchronous
    offload before park(); the helper then synchronizes the selected device.
    """

    def __init__(
        self,
        *,
        host_backup_budget_bytes: int,
        device: int | str | None = None,
        allocator: Any | None = None,
        torch_module: Any | None = None,
        pool_context_factory: Callable[..., Any] | None = None,
        frozen_backup_store: Any | None = None,
    ) -> None:
        if type(host_backup_budget_bytes) is not int or host_backup_budget_bytes <= 0:
            raise ValueError("host_backup_budget_bytes must be a positive integer")
        if device is not None and type(device) is not int and not (
            isinstance(device, str) and device
        ):
            raise TypeError("device must be an integer, a non-empty string, or None")
        injected = (
            allocator is not None,
            torch_module is not None,
            pool_context_factory is not None,
        )
        if any(injected) and not all(injected):
            raise ValueError(
                "allocator, torch_module, and pool_context_factory must be injected together"
            )

        self._frozen_backup_store = frozen_backup_store
        self.last_park_timing = None
        self.last_wake_timing = None
        self._shared_backup_bytes: dict[int, int] = {}
        self._shared_backup_created_bytes = 0
        self.host_backup_budget_bytes = host_backup_budget_bytes
        self.device = device
        self._allocator = allocator
        self._torch = torch_module
        self._pool_context_factory = pool_context_factory
        self._runtime_injected = all(injected)
        self._blas_workspace_scope = None
        self._pool: Any | None = None
        self._pool_owner_entry: Any | None = None
        # Retain the plugin allocator, never the retired MemPool Python owner.
        # Live tensors in retired pools still call through this allocator.
        self._retired_pool_owners: dict[tuple[int, int], Any] = {}
        self._cache_reclaim_count = 0
        self._last_cache_reclaim: dict[str, Any] | None = None
        self._parked_handles: dict[int, tuple[Any, ...]] | None = None
        self._state = "uninitialized"
        self._active_operations = 0
        self._lock = threading.Lock()

        # This check happens before importing Torch or initializing CUDA.
        self._validate_allocator_configuration()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _validate_allocator_configuration(self) -> None:
        """Allow expandable temporaries, never an alternate allocator backend.

        Pinned Torch 2.13 gates expandable segments on !active_user_pool in
        CUDACachingAllocator::malloc. CuMem's custom MemPool still invokes its
        plugin callbacks; only the ordinary temporary pool uses expansion.
        Runtime snapshot/ownership checks independently enforce that boundary.
        """
        configured: list[tuple[str, bool]] = []
        for env_name in _ALLOCATOR_ENVIRONMENTS:
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            found: bool | None = None
            for item in raw.split(","):
                key, separator, value = item.partition(":")
                if not separator:
                    key, separator, value = item.partition("=")
                if separator and key.strip() == "backend" and value.strip() != "native":
                    raise CudaResidencyError("CuMem residency requires the native allocator backend")
                if not separator or key.strip() != "expandable_segments":
                    continue
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes"}:
                    current = True
                elif normalized in {"false", "0", "no"}:
                    current = False
                else:
                    raise CudaResidencyError(
                        f"{env_name} has an invalid expandable_segments value: {value!r}"
                    )
                if found is not None and found != current:
                    raise CudaResidencyError(
                        f"{env_name} sets expandable_segments more than once inconsistently"
                    )
                found = current
            if found is not None:
                configured.append((env_name, found))

        if len({value for _, value in configured}) > 1:
            raise CudaResidencyError(
                "conflicting expandable_segments settings in allocator environments"
            )
        if not configured:
            raise CudaResidencyError(
                "CuMem residency requires expandable_segments to be explicit in "
                "PYTORCH_ALLOC_CONF or PYTORCH_CUDA_ALLOC_CONF before importing Torch"
            )

    def _resolve_runtime(self) -> None:
        if self._runtime_injected:
            return
        try:
            installed_vllm = metadata.version("vllm")
        except metadata.PackageNotFoundError as exc:
            raise CudaResidencyError("CuMem residency requires pinned vLLM 0.29.0") from exc
        if installed_vllm != PINNED_VLLM_VERSION:
            raise CudaResidencyError(
                f"CuMem residency is pinned to vLLM {PINNED_VLLM_VERSION}; "
                f"found {installed_vllm}"
            )

        import torch

        installed_torch = str(torch.__version__).split("+", 1)[0]
        if installed_torch != PINNED_TORCH_VERSION:
            raise CudaResidencyError(
                f"CuMem residency is pinned to Torch {PINNED_TORCH_VERSION}; "
                f"found {torch.__version__}"
            )
        # Torch 2.13 caches BLAS workspaces per handle/stream. Initializing
        # the handle inside CuMem reproduced illegal accesses in the T2 large
        # base GEMM -> small LoRA GEMM path. Keep this small workspace outside
        # the releasable pool; ordinary role switches never clear its cache.
        # No random values are consumed, so model/optimizer RNG is unchanged.
        with torch.cuda.device(self.device if self.device is not None else torch.cuda.current_device()):
            torch.backends.cuda.preferred_blas_library("cublaslt")
            warm = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
            result = warm @ warm
            torch.cuda.synchronize()
            del result, warm
        self._blas_workspace_scope = "default_allocator_before_cumem"
        try:
            from vllm.device_allocator import cumem
            from vllm.device_allocator.cumem import CuMemAllocator, use_memory_pool_with_allocator
            # This process owns the only pool. Pageable copies bound CPU storage
            # by mapped bytes; the pinned host allocator can reserve much more
            # than the logical backup and retain it across training phases.
            cumem.PIN_MEMORY = False
            allocator = CuMemAllocator.get_instance()
        except Exception as exc:
            raise CudaResidencyError(
                f"cannot initialize pinned vLLM CuMemAllocator: {exc}"
            ) from exc
        from .cumem_backup import ReusableBackupAllocator
        self._allocator = ReusableBackupAllocator(
            allocator, cumem, torch, self.host_backup_budget_bytes)
        self._torch = torch
        self._pool_context_factory = use_memory_pool_with_allocator
        self._validate_runtime_objects()

    def _validate_runtime_objects(self) -> None:
        allocator = self._allocator
        torch_module = self._torch
        required_allocator = (
            "allocator_and_pools",
            "pointer_to_data",
            "current_tag",
            "python_malloc_callback",
            "python_free_callback",
            "sleep",
            "wake_up",
        )
        missing = [name for name in required_allocator if not hasattr(allocator, name)]
        if missing:
            raise CudaResidencyError(
                f"pinned CuMemAllocator is missing required fields: {missing}"
            )
        if not callable(getattr(torch_module.cuda.memory, "use_mem_pool", None)):
            raise CudaResidencyError(
                "pinned Torch has no torch.cuda.memory.use_mem_pool API"
            )
        if not callable(self._pool_context_factory):
            raise CudaResidencyError("pinned vLLM pool context factory is unavailable")
        if not isinstance(allocator.allocator_and_pools, dict) or not isinstance(
            allocator.pointer_to_data, dict
        ):
            raise CudaResidencyError("pinned CuMemAllocator state has an unsupported shape")

    def _capture_pool(self) -> None:
        self._validate_runtime_objects()
        allocator = self._allocator
        pools = allocator.allocator_and_pools
        if set(pools) != {DEFAULT_TAG}:
            raise CudaResidencyError(
                "CuMem residency requires exactly one process-owned default memory pool"
            )
        entry = pools[DEFAULT_TAG]
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise CudaResidencyError(
                "pinned vLLM allocator_and_pools entry must be (MemPool, allocator owner)"
            )
        pool = entry[0]
        if not callable(getattr(pool, "snapshot", None)):
            raise CudaResidencyError("pinned CuMem MemPool has no snapshot()")
        self._pool = pool
        # Retain both objects: MemPool does not own its CUDAPluggableAllocator.
        self._pool_owner_entry = entry
        self._validate_default_tag_entries(allow_asleep=False)

    def _assert_pool_ownership(self, *, allow_asleep: bool = False) -> None:
        if self._pool is None or self._pool_owner_entry is None:
            raise CudaResidencyError("CuMem pool has not been initialized")
        self._validate_runtime_objects()
        pools = self._allocator.allocator_and_pools
        if (
            set(pools) != {DEFAULT_TAG}
            or pools.get(DEFAULT_TAG) is not self._pool_owner_entry
        ):
            raise CudaResidencyError(
                "CuMem pool ownership changed; refusing to use a replaced allocator"
            )
        if self._pool_owner_entry[0] is not self._pool:
            raise CudaResidencyError("CuMem pool identity changed")
        self._validate_default_tag_entries(allow_asleep=allow_asleep)

    def _validate_default_tag_entries(self, *, allow_asleep: bool) -> None:
        for pointer, data in self._allocator.pointer_to_data.items():
            if getattr(data, "tag", None) != DEFAULT_TAG:
                raise CudaResidencyError(
                    f"CuMem pointer {pointer} has a non-default tag"
                )
            if not allow_asleep and bool(getattr(data, "is_asleep", False)):
                raise CudaResidencyError(
                    f"CuMem pointer {pointer} is unexpectedly asleep"
                )

    def _begin_operation(self) -> bool:
        self._validate_allocator_configuration()
        with self._lock:
            if self._state == "uninitialized":
                self._state = "initializing"
                self._active_operations = 1
                return True
            if self._state != "awake":
                raise CudaResidencyError(
                    f"compute/allocation is refused while residency state is {self._state}"
                )
            if self._active_operations:
                raise CudaResidencyError(
                    "concurrent or nested CUDA residency operations are unsupported"
                )
            self._active_operations = 1
            return False

    def _end_operation(self) -> None:
        with self._lock:
            if self._active_operations != 1:
                self._state = "failed"
                self._active_operations = 0
                raise CudaResidencyError("CUDA residency operation accounting is incomplete")
            self._active_operations = 0
            if self._state == "initializing":
                self._state = "failed"
                raise CudaResidencyError("initial CuMem pool setup did not complete")

    @contextmanager
    def _guard_context_exit(self, manager: Any) -> Iterator[Any]:
        body_error: BaseException | None = None
        try:
            with manager as entry:
                try:
                    yield entry
                except BaseException as exc:
                    body_error = exc
                    raise
        except BaseException as exc:
            if body_error is None or exc is not body_error:
                with self._lock:
                    self._state = "failed"
            raise

    @contextmanager
    def _reuse_pool_scope(self) -> Iterator[None]:
        try:
            self._assert_pool_ownership()
            allocator = self._allocator
            old_tag = allocator.current_tag
            if old_tag != DEFAULT_TAG:
                raise CudaResidencyError(
                    f"CuMem current_tag drifted from default: {old_tag!r}"
                )
        except BaseException:
            with self._lock:
                self._state = "failed"
            raise
        try:
            allocator.current_tag = DEFAULT_TAG
            manager = self._torch.cuda.memory.use_mem_pool(self._pool)
            with self._guard_context_exit(manager):
                yield
        finally:
            allocator.current_tag = old_tag

    @contextmanager
    def allocation_context(self) -> Iterator[None]:
        """Route allocations/compute through the single persistent default pool."""
        first = self._begin_operation()
        try:
            if first:
                if self._allocator is None:
                    self._resolve_runtime()
                if self._allocator.allocator_and_pools or self._allocator.pointer_to_data:
                    raise CudaResidencyError(
                        "CuMem singleton is already in use; residency must own its first pool"
                    )
                with self._guard_context_exit(self._pool_context_factory(
                    self._allocator.python_malloc_callback, self._allocator.python_free_callback
                )) as entry:
                    self._allocator.allocator_and_pools[DEFAULT_TAG] = entry
                    self._capture_pool()
                    with self._lock:
                        self._state = "awake"
                    yield
            else:
                with self._reuse_pool_scope():
                    yield
        except BaseException:
            with self._lock:
                if self._state == "initializing":
                    self._state = "failed"
            raise
        finally:
            self._end_operation()

    def _pool_snapshot(self) -> list[dict[str, Any]]:
        """Snapshot all CuMem-owned pools, including retired live segments."""
        if self._pool is None:
            return []
        rows = list(self._pool.snapshot())
        for pool_id in self._retired_pool_owners:
            rows.extend(self._torch.cuda.memory_snapshot(pool_id))
        return rows

    @staticmethod
    def _active_cache_boundary(rows) -> dict[int, int]:
        # Whole-segment retirement must preserve every live allocation's address.
        active = {}
        for row in rows:
            blocks = row.get("blocks")
            if not isinstance(blocks, list):
                raise CudaResidencyError("cache retirement requires block-level pool snapshots")
            for block in blocks:
                state = block.get("state")
                if state == "active_awaiting_free":
                    raise CudaResidencyError("drain pending CUDA frees before cache retirement")
                if state == "active_allocated":
                    active[int(block["address"])] = int(block["size"])
                elif state != "inactive":
                    raise CudaResidencyError("unsupported allocator block state at cache boundary")
        return active

    def reclaim_inactive_cache(self) -> dict[str, Any]:
        """Retire the current pool owner; Torch frees only whole inactive blocks.

        Replacing the allocation pool does not move live Tensor/Adam storage.
        Never call vLLM's raw one-shot trim or adjust Torch reference counts.
        """
        with self._lock:
            if self._state != "awake" or self._active_operations:
                raise CudaResidencyError("cache retirement requires an awake, quiescent residency")
            self._active_operations = 1
        try:
            self._assert_pool_ownership()
            if self._pool.use_count() != 1:
                raise CudaResidencyError("cache retirement requires exactly one MemPool owner")
            self._torch.cuda.synchronize(self.device)
            # Process completed stream frees; retained current-pool cache remains
            # protected until its owner is retired below.
            self._torch.cuda.empty_cache()
            before_rows = self._pool_snapshot()
            before = self._allocation_rows_from_snapshot(before_rows)
            self._awake_inventory(before)
            active_before = self._active_cache_boundary(before_rows)
            before_pointers = set(self._allocator.pointer_to_data)
            before_bytes = sum(int(v.handle[1]) for v in self._allocator.pointer_to_data.values())
            free_before, _ = self._torch.cuda.mem_get_info(self.device)
            current_rows = self._pool.snapshot()
            cached = sum(int(row["total_size"]) - int(row["active_size"]) for row in current_rows)
            report = {"strategy": "torch_mempool_owner_retirement",
                      "status": "completed", "rotated": False,
                      "reclaimed_bytes": 0, "released_segments": 0,
                      "current_inactive_bytes_before": cached,
                      "active_bytes_before": sum(active_before.values())}
            if cached:
                old_id = tuple(self._pool.id)
                old_plugin_owner = self._pool_owner_entry[1]
                # Do not keep the old entry or pool in a local variable: the last
                # Python owner must disappear so MemPool::~MemPool informs Torch
                # and triggers the CuMem free callbacks for whole free segments.
                with self._pool_context_factory(
                    self._allocator.python_malloc_callback,
                    self._allocator.python_free_callback,
                ) as entry:
                    self._retired_pool_owners[old_id] = old_plugin_owner
                    self._allocator.allocator_and_pools[DEFAULT_TAG] = entry
                    self._capture_pool()
                gc.collect()
                self._torch.cuda.empty_cache()
                report.update(rotated=True, retired_pool_id=list(old_id),
                              current_pool_id=list(self._pool.id))
            self._torch.cuda.synchronize(self.device)
            self._assert_pool_ownership()
            after_rows = self._pool_snapshot()
            after = self._allocation_rows_from_snapshot(after_rows)
            self._awake_inventory(after)
            if self._active_cache_boundary(after_rows) != active_before:
                raise CudaResidencyError("cache retirement changed active CUDA allocations")
            after_pointers = set(self._allocator.pointer_to_data)
            if after_pointers - before_pointers:
                raise CudaResidencyError("cache retirement unexpectedly created CUDA segments")
            after_bytes = sum(int(v.handle[1]) for v in self._allocator.pointer_to_data.values())
            free_after, _ = self._torch.cuda.mem_get_info(self.device)
            report.update(reclaimed_bytes=before_bytes - after_bytes,
                          released_segments=len(before_pointers - after_pointers),
                          tracked_bytes_before=before_bytes, tracked_bytes_after=after_bytes,
                          cuda_free_delta_bytes=int(free_after) - int(free_before),
                          active_addresses_preserved=True)
            self._cache_reclaim_count += 1
            self._last_cache_reclaim = report
            return dict(report)
        except BaseException:
            with self._lock:
                self._state = "failed"
            raise
        finally:
            self._end_operation()

    @staticmethod
    def _allocation_rows(pool: Any) -> dict[int, int]:
        return CudaResidency._allocation_rows_from_snapshot(pool.snapshot())

    @staticmethod
    def _allocation_rows_from_snapshot(rows) -> dict[int, int]:
        if not isinstance(rows, (tuple, list)):
            raise CudaResidencyError("CuMem pool snapshot must be a list of allocations")
        result: dict[int, int] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise CudaResidencyError("CuMem pool snapshot contains a malformed row")
            if row.get("is_expandable", False):
                raise CudaResidencyError("CuMem persistent pool contains an expandable segment")
            pointer = row.get("address")
            allocated_size = row.get("allocated_size")
            if type(pointer) is not int or pointer <= 0:
                raise CudaResidencyError("CuMem pool snapshot has an invalid address")
            if type(allocated_size) is not int or allocated_size < 0:
                raise CudaResidencyError(
                    f"CuMem snapshot allocation {pointer} has an invalid allocated_size"
                )
            if pointer in result:
                raise CudaResidencyError(
                    f"CuMem pool snapshot repeats allocation address {pointer}"
                )
            result[pointer] = allocated_size
        return result

    def _awake_inventory(self, rows: Mapping[int, int]) -> tuple[set[int], int]:
        allocator = self._allocator
        snapshot_pointers = set(rows)
        tracked = set(allocator.pointer_to_data)
        if snapshot_pointers != tracked:
            missing = sorted(snapshot_pointers - tracked)
            extra = sorted(tracked - snapshot_pointers)
            raise CudaResidencyError(
                "CuMem pool snapshot and pointer_to_data differ "
                f"(missing={missing[:4]}, extra={extra[:4]})"
            )

        total_bytes = 0
        for pointer, data in allocator.pointer_to_data.items():
            if data.tag != DEFAULT_TAG or data.is_asleep:
                raise CudaResidencyError(
                    f"CuMem pointer {pointer} is not awake in the default pool"
                )
            handle = data.handle
            if not isinstance(handle, (tuple, list)) or len(handle) < 3:
                raise CudaResidencyError(
                f"CuMem pointer {pointer} has a malformed handle"
            )
            if type(handle[1]) is not int or handle[1] <= 0 or handle[2] != pointer:
                raise CudaResidencyError(f"CuMem pointer {pointer} handle is inconsistent")
            if rows[pointer] > handle[1]:
                raise CudaResidencyError(
                    f"CuMem snapshot allocation {pointer} exceeds its mapped handle"
                )
            if data.cpu_backup_tensor is not None:
                raise CudaResidencyError(
                    f"awake CuMem pointer {pointer} unexpectedly retains a CPU backup"
                )
            total_bytes += handle[1]
        return snapshot_pointers, total_bytes

    @staticmethod
    def _tensor_bytes(tensor: Any) -> int:
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if callable(numel) and callable(element_size):
            value = numel() * element_size()
            if type(value) is int and value >= 0:
                return value
        value = getattr(tensor, "nbytes", None)
        if type(value) is int and value >= 0:
            return value
        raise CudaResidencyError("CuMem CPU backup has no measurable byte size")

    @staticmethod
    def _cpu_backup_bytes(data: Any, pointer: int) -> int:
        backup = data.cpu_backup_tensor
        if backup is None:
            raise CudaResidencyError(
                f"asleep CuMem pointer {pointer} is missing its CPU backup"
            )
        device = getattr(backup, "device", None)
        if getattr(device, "type", None) != "cpu":
            raise CudaResidencyError(
                f"CuMem backup for pointer {pointer} is not a CPU tensor"
            )
        return CudaResidency._tensor_bytes(backup)

    def assert_tensors_owned(self, tensors) -> dict[str, int]:
        """Prove every persistent CUDA storage belongs to this awake pool."""
        if self.state != "awake":
            raise CudaResidencyError("persistent storage checks require awake state")
        self._assert_pool_ownership()
        handles = self._allocator.pointer_to_data
        starts = sorted(handles)
        storages = {}
        tensor_count = 0
        for label, tensor in tensors:
            if tensor.device.type != "cuda":
                continue
            storage = tensor.untyped_storage()
            pointer, size = int(storage.data_ptr()), int(storage.nbytes())
            if size == 0:
                continue
            index = bisect_right(starts, pointer) - 1
            if index < 0:
                raise CudaResidencyError(f"persistent CUDA storage is outside the pool: {label}")
            begin = starts[index]
            if pointer + size > begin + int(handles[begin].handle[1]):
                raise CudaResidencyError(f"persistent CUDA storage is outside the pool: {label}")
            storages[pointer] = size
            tensor_count += 1
        return {"tensor_count": tensor_count, "unique_storages": len(storages),
                "storage_bytes": sum(storages.values())}

    def park(
        self, *, drain: Callable[[], None], frozen_tensors=(), mutable_tensors=(),
        expected_backup_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Drain work and back up all pool segments, including inactive cache."""
        from .gradient_executor.phase_timing import JobPhaseTiming

        if (expected_backup_bytes is not None
                and (type(expected_backup_bytes) is not int or expected_backup_bytes < 0)):
            raise ValueError("expected_backup_bytes must be a non-negative integer")
        timer = JobPhaseTiming()
        self.last_park_timing = None
        if not callable(drain):
            raise TypeError("park requires a callable drain callback")
        with self._lock:
            if self._state != "awake":
                raise CudaResidencyError(
                    f"park requires awake state, found {self._state}"
                )
            if self._active_operations:
                raise CudaResidencyError(
                    "cannot park while a CUDA allocation/compute operation is active"
                )
            self._state = "parking"

        try:
            with timer.measure('caller_cuda_drain'):
                drain()
        except BaseException:
            with self._lock:
                self._state = "failed"
            raise

        try:
            self._torch.cuda.synchronize(self.device)
            # Forward/backward temporaries use the normal allocator. Release
            # its inactive cache before lending the device to inference.
            self._torch.cuda.empty_cache()
            self._assert_pool_ownership()
            with timer.measure('pool_snapshot_inventory'):
                pool_snapshot = self._pool_snapshot()
                rows = self._allocation_rows_from_snapshot(pool_snapshot)
                pointers, backup_bytes = self._awake_inventory(rows)
            if (expected_backup_bytes is not None
                    and backup_bytes > expected_backup_bytes):
                raise CudaResidencyError(
                    "CuMem segment inventory exceeded host-cache headroom preflight "
                    f"({backup_bytes} > {expected_backup_bytes})"
                )
            if backup_bytes > self.host_backup_budget_bytes:
                raise CudaResidencyBudgetError(
                    "tracked CuMem bytes exceed the host backup budget "
                    f"({backup_bytes} > {self.host_backup_budget_bytes})"
                )
            self._parked_handles = {
                pointer: tuple(self._allocator.pointer_to_data[pointer].handle)
                for pointer in pointers
            }
            candidates = {}
            if self._frozen_backup_store is not None:
                from .shared_frozen_backup import frozen_segment_ranges
                def storage_ranges(tensors):
                    result = []
                    for tensor in tensors:
                        if tensor.device.type == "cuda":
                            storage = tensor.untyped_storage()
                            if storage.nbytes():
                                result.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
                    return result
                # No active operations are allowed in parking state; reuse the
                # exact snapshot whose pointer inventory was just validated.
                with timer.measure('frozen_segment_classification'):
                    candidates = frozen_segment_ranges(pool_snapshot,
                        storage_ranges(frozen_tensors), storage_ranges(mutable_tensors))
            prepare = getattr(self._allocator, 'prepare_frozen_backups', None)
            if prepare is not None:
                prepare(candidates, self._frozen_backup_store)
            with timer.measure('allocator_sleep'), _measure_allocator_gc(timer):
                self._allocator.sleep(offload_tags=(DEFAULT_TAG,))
            self._shared_backup_bytes = {}
            self._shared_backup_created_bytes = 0
            for pointer, ranges in candidates.items():
                data = self._allocator.pointer_to_data[pointer]
                # Only the complete, same-sized CuMem segment may be replaced.
                size = self._tensor_bytes(data.cpu_backup_tensor)
                if size != data.handle[1]:
                    raise CudaResidencyError("frozen backup segment size differs from its handle")
                streamed = getattr(self._allocator, 'streamed_backups', {}).get(pointer)
                if streamed is not None:
                    _, created = streamed
                else:
                    with timer.measure('frozen_backup_share'):
                        mapped, created = self._frozen_backup_store.share(data.cpu_backup_tensor, ranges)
                    data.cpu_backup_tensor = mapped
                forget = getattr(self._allocator, 'finish_shared_backup', None)
                if forget is not None:
                    forget(pointer)
                self._shared_backup_bytes[pointer] = size
                if created:
                    self._shared_backup_created_bytes += size
            self._validate_asleep_inventory(pointers, backup_bytes)
            with self._lock:
                self._state = "asleep"
            self.last_park_timing = timer.snapshot()
            return self.memory_report()
        except CudaResidencyBudgetError:
            with self._lock:
                self._state = "failed"
            raise
        except BaseException:
            with self._lock:
                self._state = "failed"
            raise

    def _validate_asleep_inventory(
        self,
        pointers: set[int],
        expected_backup_bytes: int,
    ) -> None:
        current = self._allocator.pointer_to_data
        if set(current) != pointers:
            raise CudaResidencyError("CuMem pointer inventory changed while sleeping")
        actual_backup_bytes = 0
        for pointer, data in current.items():
            if data.tag != DEFAULT_TAG or not data.is_asleep:
                raise CudaResidencyError(
                    f"CuMem pointer {pointer} did not enter the asleep state"
                )
            if tuple(data.handle) != self._parked_handles[pointer]:
                raise CudaResidencyError(
                    f"CuMem pointer {pointer} handle changed while sleeping"
                )
            actual_backup_bytes += self._cpu_backup_bytes(data, pointer)
        if actual_backup_bytes != expected_backup_bytes:
            raise CudaResidencyError(
                "CuMem CPU backup inventory is incomplete "
                f"({actual_backup_bytes} != {expected_backup_bytes})"
            )
        if actual_backup_bytes > self.host_backup_budget_bytes:
            raise CudaResidencyError("CuMem sleep exceeded the configured host backup budget")

    def wake(self) -> dict[str, Any]:
        """Remap every parked allocation at its original pointer and restore data."""
        with self._lock:
            if self._state != "asleep":
                raise CudaResidencyError(
                    f"wake requires asleep state, found {self._state}"
                )
            if self._active_operations:
                raise CudaResidencyError("cannot wake during a CUDA residency operation")
            self._state = "waking"

        try:
            self._assert_pool_ownership(allow_asleep=True)
            if self._parked_handles is None:
                raise CudaResidencyError("parked pointer inventory is missing")
            current = self._allocator.pointer_to_data
            pointers = set(self._parked_handles)
            if set(current) != pointers:
                raise CudaResidencyError("CuMem pointer inventory is incomplete before wake")
            backup_bytes = 0
            for pointer, data in current.items():
                if data.tag != DEFAULT_TAG or not data.is_asleep:
                    raise CudaResidencyError(
                        f"CuMem pointer {pointer} is not parked in the default pool"
                    )
                if tuple(data.handle) != self._parked_handles[pointer]:
                    raise CudaResidencyError(
                        f"CuMem pointer {pointer} handle changed before wake"
                    )
                backup_bytes += self._cpu_backup_bytes(data, pointer)
            if backup_bytes > self.host_backup_budget_bytes:
                raise CudaResidencyBudgetError(
                    "parked CPU backups exceed the host backup budget"
                )

            from .gradient_executor.phase_timing import JobPhaseTiming
            timer = JobPhaseTiming()
            with timer.measure('allocator_wake'):
                self._allocator.wake_up(tags=[DEFAULT_TAG])
            self.last_wake_timing = timer.snapshot()
            if set(self._allocator.pointer_to_data) != pointers:
                raise CudaResidencyError("CuMem pointer inventory changed while waking")
            for pointer, data in self._allocator.pointer_to_data.items():
                if data.tag != DEFAULT_TAG or data.is_asleep:
                    raise CudaResidencyError(
                        f"CuMem pointer {pointer} did not return to the awake state"
                    )
                if tuple(data.handle) != self._parked_handles[pointer]:
                    raise CudaResidencyError(
                        f"CuMem pointer {pointer} changed its handle during wake"
                    )
                if data.cpu_backup_tensor is not None:
                    raise CudaResidencyError(
                        f"CuMem pointer {pointer} retained a CPU backup after wake"
                    )
            with self._lock:
                self._state = "awake"
            self._parked_handles = None
            return self.memory_report()
        except BaseException:
            with self._lock:
                self._state = "failed"
            raise

    def memory_report(self) -> dict[str, Any]:
        """Return CUDA free memory and tracked pool/host bytes at a quiescent point."""
        with self._lock:
            state = self._state
            if self._active_operations:
                raise CudaResidencyError(
                    "memory_report requires a quiescent CUDA residency operation"
                )
            if state in {"initializing", "parking", "waking"}:
                raise CudaResidencyError(
                    f"memory_report is unavailable during {state}"
                )
        report: dict[str, Any] = {
            "state": state,
            "device": self.device,
            "host_backup_budget_bytes": self.host_backup_budget_bytes,
            "blas_workspace_scope": self._blas_workspace_scope,
            "host_backup_pin_memory": False,
            "cuda_free_bytes": None,
            "cuda_total_bytes": None,
            "pool_tracked_bytes": 0,
            "pool_awake_bytes": 0,
            "pool_snapshot_active_bytes": None,
            "retired_pool_count": len(self._retired_pool_owners),
            "cache_reclaim_count": self._cache_reclaim_count,
            "last_cache_reclaim": self._last_cache_reclaim,
            "host_backup_bytes": 0,
            # Logical mapped bytes are NOT per-rank physical RAM or a saving.
            "host_backup_shared_mapped_bytes": 0,
            "host_backup_private_bytes": 0,
            "last_park_shared_created_bytes": self._shared_backup_created_bytes,
            "frozen_backup_sharing_enabled": self._frozen_backup_store is not None,
        }
        if self._allocator is None:
            return report
        if self.device is None:
            free_bytes, total_bytes = self._torch.cuda.mem_get_info()
        else:
            free_bytes, total_bytes = self._torch.cuda.mem_get_info(self.device)
        report["cuda_free_bytes"] = int(free_bytes)
        report["cuda_total_bytes"] = int(total_bytes)

        rows = self._allocator.pointer_to_data
        for pointer, data in rows.items():
            if data.tag != DEFAULT_TAG:
                raise CudaResidencyError(
                    f"memory report found non-default CuMem pointer {pointer}"
                )
            handle_bytes = int(data.handle[1])
            report["pool_tracked_bytes"] += handle_bytes
            if not data.is_asleep:
                report["pool_awake_bytes"] += handle_bytes
            if data.cpu_backup_tensor is not None:
                size = self._tensor_bytes(data.cpu_backup_tensor)
                report["host_backup_bytes"] += size
                if pointer in self._shared_backup_bytes:
                    report["host_backup_shared_mapped_bytes"] += size
                else:
                    report["host_backup_private_bytes"] += size
        if state == "awake" and self._pool is not None:
            allocations = self._allocation_rows_from_snapshot(self._pool_snapshot())
            report["pool_snapshot_active_bytes"] = sum(allocations.values())
        retained = int(getattr(self._allocator, 'retained_backup_bytes', 0))
        idle = int(getattr(self._allocator, 'idle_backup_bytes', 0))
        report['host_backup_retained_bytes'] = retained
        report['host_backup_storage_bytes'] = report['host_backup_bytes'] + idle
        report['backup_sleep_timings'] = dict(getattr(self._allocator, 'last_sleep_timings', {}))
        if report['host_backup_storage_bytes'] > self.host_backup_budget_bytes:
            raise CudaResidencyBudgetError('retained and active CPU backup storage exceeds host budget')
        return report
