"""Microbatch-local T2 GEMM output snapshots; the original autograd VJP stays intact.

No dispatcher modes or compiled-region changes. Selected projections must execute
once in the original forward. Nested checkpoint replay may read a snapshot again.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatchcase
import re

import torch

from .bounded_offload import PinnedCPUArena


DEFAULT_CPU_BYTES = 2 * 1024**3
DEFAULT_STAGING_BYTES = 512 * 1024**2
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def resolve_parameters(model, selectors):
    """Match actual LoRA parameter FQNs, deduplicating A/B into one base site."""
    from ..models.ternary import FrozenTernaryLinear

    if not isinstance(selectors, (list, tuple)) or any(
        not isinstance(s, str) or not s for s in selectors
    ):
        raise ValueError("activation_cache_parameter_names must be a list of nonempty strings")
    modules = dict(model.named_modules())
    parameters = dict(model.named_parameters())
    sites = {}
    for selector in selectors:
        matches = [name for name in parameters if fnmatchcase(name, selector)]
        if not matches:
            raise ValueError(f"activation cache parameter selector matched nothing: {selector}")
        for name in matches:
            owner = name.split(".lora_A.")[0].split(".lora_B.")[0]
            projection = modules.get(owner)
            base = getattr(projection, "base_layer", None)
            layer = _LAYER.search(owner)
            if owner == name or not isinstance(base, FrozenTernaryLinear) or layer is None:
                raise ValueError(f"activation cache requires a decoder T2 LoRA parameter: {name}")
            key = id(base)
            if key not in sites:
                sites[key] = dict(module=base, module_name=owner + ".base_layer",
                                  layer=int(layer[1]), parameters=[])
            sites[key]["parameters"].append(name)
    return sorted(sites.values(), key=lambda s: (s["layer"], s["module_name"]))


@dataclass
class _Entry:
    lease: object
    cpu: torch.Tensor
    metadata: tuple
    ready: object = None
    device_value: torch.Tensor | None = None
    restored: object = None


class _Site:
    def __init__(self, cache, index):
        self.cache, self.index = cache, index

    def restore(self, inputs):
        return self.cache.restore(self.index, inputs)

    def save(self, inputs, output):
        self.cache.save(self.index, inputs, output)


class ParameterActivationCache:
    def __init__(self, model, selectors, *, cpu_bytes=DEFAULT_CPU_BYTES,
                 staging_bytes=DEFAULT_STAGING_BYTES, device_type="cuda"):
        for value in (cpu_bytes, staging_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("activation cache byte budgets must be positive integers")
        self.model = model
        self.sites = resolve_parameters(model, selectors)
        self.arena = PinnedCPUArena(cpu_bytes, pin_memory=device_type == "cuda")
        self.staging_bytes, self.device_type = staging_bytes, device_type
        self.entries, self.seen, self.handles = {}, set(), []
        self.pending = deque()
        self.pending_bytes = 0
        self.stream = None
        self.phase = "new"
        self.stats = dict(saved=0, hits=0, skipped=0, prefetched=0, prefetch_hits=0,
                          d2h_bytes=0, h2d_bytes=0, peak_snapshot_bytes=0,
                          peak_staging_bytes=0, budget_skips=0)

    @property
    def report(self):
        return {**self.stats, "phase": self.phase,
                "cpu_capacity_bytes": self.arena.capacity_bytes,
                "cpu_peak_bytes": self.arena.max_live_bytes,
                "cpu_live_bytes": self.arena.live_bytes,
                "staging_capacity_bytes": self.staging_bytes,
                "sites": [{k: v for k, v in s.items() if k != "module"} for s in self.sites],
                "implementation": "native T2 autograd forward-output cache; original VJP"}

    def __enter__(self):
        if self.phase != "new":
            raise RuntimeError("activation cache cannot be reused across microbatches")
        if any(hasattr(s["module"], "_activation_cache_site") for s in self.sites):
            raise RuntimeError("activation cache sites already owned")
        self.versions = [(p, p._version) for p in self.model.parameters()]
        self.phase = "forward"
        try:
            for i, site in enumerate(self.sites):
                site["module"]._activation_cache_site = _Site(self, i)
            # Every decoder entry provides lookahead even across sparse selections.
            for name, module in self.model.named_modules():
                if re.search(r"(?:^|\.)layers\.\d+$", name):
                    layer = int(name.rsplit(".", 1)[1])
                    self.handles.append(module.register_forward_pre_hook(
                        lambda _m, _a, layer=layer: self._enter_layer(layer)))
        except BaseException:
            self.close()
            raise
        return self

    @staticmethod
    def _metadata(inputs):
        return tuple(inputs.shape), inputs.dtype, inputs.device

    def _reap(self, *, all_pending=False):
        while self.pending and (all_pending or self.pending[0][0].query()):
            event, snapshot, size = self.pending.popleft()
            if all_pending:
                event.synchronize()
            self.pending_bytes -= size

    def save(self, index, inputs, output):
        if self.phase == "backward":
            return
        if self.phase != "forward" or index in self.seen:
            raise RuntimeError("selected projection must execute once per original forward")
        self.seen.add(index)
        if output.device.type != self.device_type or output.layout != torch.strided:
            raise ValueError("activation cache device/layout mismatch")
        size = output.numel() * output.element_size()
        if size > self.staging_bytes:
            self.stats["skipped"] += 1
            return
        lease, _ = self.arena.try_acquire(size)
        if lease is None:
            self.stats["skipped"] += 1
            return
        try:
            cpu = self.arena.tensor_view(lease, output.dtype, tuple(output.shape), size)
            entry = _Entry(lease, cpu, self._metadata(inputs))
            if self.device_type == "cuda":
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("activation cache does not support CUDA graph capture")
                if self.stream is None:
                    self.stream = torch.cuda.Stream(device=output.device)
                    self.device = output.device
                if self.device != output.device:
                    raise ValueError("activation cache cannot span CUDA devices")
                self._reap()
                while self.pending and self.pending_bytes + size > self.staging_bytes:
                    self.pending[0][0].synchronize()
                    self._reap()
                # Snapshot before PEFT's in-place LoRA addition; the copy stream
                # may read it while subsequent GPU computation modifies output.
                snapshot = output.detach().clone()
                producer = torch.cuda.current_stream(output.device).record_event()
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(producer)
                    cpu.copy_(snapshot, non_blocking=True)
                    entry.ready = lease.record_event(self.stream, output.device.index)
                    snapshot.record_stream(self.stream)
                self.pending.append((entry.ready, snapshot, size))
                self.pending_bytes += size
                self.stats["peak_snapshot_bytes"] = max(self.stats["peak_snapshot_bytes"], self.pending_bytes)
            else:
                cpu.copy_(output.detach())
            self.entries[index] = entry
            self.stats["saved"] += 1
            self.stats["d2h_bytes"] += size
        except BaseException:
            if self.stream is not None:
                self.stream.synchronize()
            lease.release()
            raise

    def _staged_bytes(self):
        return sum(e.cpu.numel() * e.cpu.element_size() for e in self.entries.values()
                   if e.device_value is not None)

    def _stage(self, index, *, speculative=False):
        entry = self.entries[index]
        if entry.device_value is not None:
            return
        size = entry.cpu.numel() * entry.cpu.element_size()
        if self._staged_bytes() + size > self.staging_bytes:
            if speculative:
                self.stats["budget_skips"] += 1
                return
            # Only speculative buffers are owned here. Consumer clones belong
            # to autograd; record_stream keeps any outstanding use safe.
            for other in self.entries.values():
                other.device_value = None
                other.restored = None
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(entry.ready)
                entry.device_value = entry.cpu.to(self.device, non_blocking=True)
                entry.restored = entry.lease.record_event(self.stream, self.device.index)
        else:
            entry.device_value = entry.cpu.clone()
        self.stats["h2d_bytes"] += size
        self.stats["peak_staging_bytes"] = max(self.stats["peak_staging_bytes"], self._staged_bytes())
        if speculative:
            self.stats["prefetched"] += 1

    def _enter_layer(self, layer):
        if self.phase != "backward":
            return
        for i, entry in self.entries.items():
            if self.sites[i]["layer"] > layer:
                entry.device_value = None
                entry.restored = None
        # Prefetch the next selected layer while this decoder computes. Do not
        # assume selected layers are adjacent or checkpoint regions are flat.
        following = [s["layer"] for i, s in enumerate(self.sites)
                     if i in self.entries and s["layer"] < layer]
        if following:
            next_layer = max(following)
            for i in self.entries:
                if self.sites[i]["layer"] == next_layer:
                    self._stage(i, speculative=True)

    def restore(self, index, inputs):
        if self.phase == "forward":
            return None
        if self.phase != "backward":
            raise RuntimeError("activation cache accessed outside its microbatch")
        entry = self.entries.get(index)
        if entry is None:
            return None  # Recorded admission miss; normal T2 forward recomputes.
        if entry.metadata != self._metadata(inputs):
            raise RuntimeError("activation cache replay shape/dtype/device changed")
        if entry.device_value is not None:
            self.stats["prefetch_hits"] += 1
        self._stage(index)
        if self.stream is not None:
            consumer = torch.cuda.current_stream(inputs.device)
            consumer.wait_event(entry.restored)
            entry.device_value.record_stream(consumer)
        self.stats["hits"] += 1
        # An owned output is required by the existing in-place LoRA epilogue.
        return entry.device_value.clone()

    @contextmanager
    def backward(self):
        if self.phase != "forward":
            raise RuntimeError("activation cache supports one backward per microbatch")
        if any(p._version != v for p, v in self.versions):
            raise RuntimeError("parameters changed between cached forward and backward")
        self.phase = "backward"
        self._reap(all_pending=True)
        if self.entries:
            last = max(self.sites[i]["layer"] for i in self.entries)
            for i in self.entries:
                if self.sites[i]["layer"] == last:
                    self._stage(i, speculative=True)
        yield

    def close(self):
        try:
            if self.stream is not None:
                self.stream.synchronize()
            self._reap(all_pending=True)
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            for site in self.sites:
                installed = getattr(site["module"], "_activation_cache_site", None)
                if installed is not None and installed.cache is self:
                    del site["module"]._activation_cache_site
            for entry in self.entries.values():
                entry.lease.release()
            self.entries.clear()
            self.arena.trim_idle()
            self.phase = "closed"

    def __exit__(self, *_):
        self.close()
