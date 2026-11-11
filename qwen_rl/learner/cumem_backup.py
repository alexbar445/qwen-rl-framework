"""Owned default-pool sleep with bounded reusable pageable CPU backups.

Uses pinned vLLM's registry, checked cudaMemcpy and unmap primitives. Allocation
callbacks and wake remain the upstream owner's methods; no Torch global patch.
"""
import gc
from time import perf_counter


FROZEN_SCRATCH_BYTES = 64 * 1024**2


class ReusableBackupAllocator:
    def __init__(self, base, runtime, torch_module, budget):
        self.base, self.runtime, self.torch = base, runtime, torch_module
        self.budget = budget
        self.buffers = {}
        self.last_sleep_timings = {}
        self.scratch = None
        self.frozen_ranges = {}
        self.frozen_store = None
        self.streamed_backups = {}

    def prepare_frozen_backups(self, ranges, store):
        self.frozen_ranges, self.frozen_store = ranges, store

    def _scratch_bytes(self):
        return 0 if self.scratch is None else self.scratch.numel()

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def retained_backup_bytes(self):
        return self._scratch_bytes() + sum(t.numel()*t.element_size() for t in self.buffers.values())

    @property
    def idle_backup_bytes(self):
        return self._scratch_bytes() + sum(t.numel()*t.element_size() for ptr,t in self.buffers.items()
                   if ptr not in self.base.pointer_to_data
                   or self.base.pointer_to_data[ptr].cpu_backup_tensor is not t)

    def finish_shared_backup(self, pointer):
        # A COW mapping and its reusable private staging buffer are distinct
        # storage. Retain both only when their combined logical bytes fit.
        active = sum(data.cpu_backup_tensor.numel()*data.cpu_backup_tensor.element_size()
                     for data in self.base.pointer_to_data.values()
                     if data.is_asleep and data.cpu_backup_tensor is not None)
        if active + self.idle_backup_bytes > self.budget:
            self.buffers.pop(pointer, None)

    def sleep(self, offload_tags):
        if tuple(offload_tags) != ('default',):
            raise ValueError('owned backup allocator supports only the default pool')
        entries = self.base.pointer_to_data
        if any(data.tag != 'default' or data.is_asleep for data in entries.values()):
            raise RuntimeError('owned backup sleep requires an entirely awake default pool')
        if sum(int(data.handle[1]) for data in entries.values()) > self.budget:
            raise RuntimeError('CPU backup inventory exceeds its host budget')
        for ptr in list(self.buffers):
            if ptr not in entries or self.buffers[ptr].numel() != entries[ptr].handle[1]:
                del self.buffers[ptr]
        total = sum(int(data.handle[1]) for data in entries.values())
        scratch_bytes = min(FROZEN_SCRATCH_BYTES,
                            max((int(entries[p].handle[1]) for p in self.frozen_ranges), default=0))
        # Do not degrade to millions of tiny copies when budget headroom is low.
        if total + scratch_bytes > self.budget:
            scratch_bytes = 0
        if scratch_bytes:
            for pointer in self.frozen_ranges:
                self.buffers.pop(pointer, None)
        if not scratch_bytes:
            self.scratch = None
        elif self.scratch is None or self.scratch.numel() != scratch_bytes:
            self.scratch = self.torch.empty(scratch_bytes, dtype=self.torch.uint8, device='cpu', pin_memory=False)
        self.streamed_backups = {}
        timings = dict(allocate_seconds=0.,copy_seconds=0.,unmap_seconds=0.,reused_bytes=0, frozen_stream_seconds=0.)
        for ptr, data in entries.items():
            size = int(data.handle[1]);start = perf_counter()
            if ptr in self.frozen_ranges and self.scratch is not None:
                self.buffers.pop(ptr, None)
                def read_chunk(offset, count):
                    start = perf_counter()
                    self.runtime.libcudart.cudaMemcpy(self.scratch.data_ptr(),ptr+offset,count)
                    timings['copy_seconds'] += perf_counter()-start
                    return self.scratch[:count]
                started = perf_counter()
                backup, created = self.frozen_store.share_stream(
                    size, self.frozen_ranges[ptr], read_chunk, self.scratch.numel())
                timings['frozen_stream_seconds'] += perf_counter()-started
                self.streamed_backups[ptr] = (size, created)
                data.cpu_backup_tensor = backup
                start = perf_counter()
                try:
                    self.runtime.unmap_and_release(data.handle)
                finally:
                    data.is_asleep = True
                timings['unmap_seconds'] += perf_counter()-start
                continue
            backup = self.buffers.get(ptr)
            if backup is None:
                backup = self.torch.empty(size,dtype=self.torch.uint8,device='cpu',pin_memory=False)
                self.buffers[ptr] = backup
            else:
                timings['reused_bytes'] += size
            timings['allocate_seconds'] += perf_counter()-start
            start = perf_counter()
            self.runtime.libcudart.cudaMemcpy(backup.data_ptr(),ptr,size)
            timings['copy_seconds'] += perf_counter()-start
            data.cpu_backup_tensor = backup
            start = perf_counter()
            try:
                self.runtime.unmap_and_release(data.handle)
            finally:
                data.is_asleep = True
            timings['unmap_seconds'] += perf_counter()-start
        start = perf_counter();gc.collect();self.torch.cuda.empty_cache()
        timings['gc_and_empty_cache_seconds'] = perf_counter()-start
        self.last_sleep_timings = timings
