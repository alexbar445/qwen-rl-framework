"""Persistent single-process model ownership over the existing CuMem pool."""
from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any


class GradientOwnerError(RuntimeError):
    """The persistent gradient model cannot safely change GPU ownership."""


class PersistentGradientOwner:
    """Build once in ``CudaResidency`` and park the same model between windows.

    ``model_factory`` is called exactly once inside the residency allocation
    context. It may return a gradient-only engine/trainer; ``model_getter``
    extracts the persistent ``torch.nn.Module`` whose parameters back that
    engine. Gradient computation itself runs outside ``allocation_context`` so
    forward/backward temporaries use the ordinary CUDA allocator.

    This owner deliberately excludes optimizer state. Callers must copy each
    completed gradient contribution to owned CPU memory and clear all parameter
    ``.grad`` fields before calling ``park()``.
    """

    def __init__(
        self,
        residency: Any,
        *,
        model_factory: Callable[..., Any],
        model_getter: Callable[[Any], Any],
        model_preparer: Callable[[], Any] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if not callable(model_factory) or not callable(model_getter):
            raise TypeError("model_factory and model_getter must be callable")
        if not callable(getattr(residency, "allocation_context", None)):
            raise TypeError("residency must provide allocation_context()")
        if not callable(getattr(residency, "park", None)) or not callable(
            getattr(residency, "wake", None)
        ):
            raise TypeError("residency must provide park() and wake()")
        self.residency = residency
        self._factory = model_factory
        self._model_getter = model_getter
        if model_preparer is not None and not callable(model_preparer):
            raise TypeError("model_preparer must be callable when provided")
        self._model_preparer = model_preparer
        self._prepared_model: Any | None = None
        self._preparation_lock = threading.Lock()
        self._torch = torch_module
        self._engine: Any | None = None
        self._model: Any | None = None
        self._failed = False
        self.last_park_timing = None
        self.last_host_cache_reclaim = None
        self._host_cache_monitor = None

    @property
    def loaded(self) -> bool:
        return self._engine is not None and self._model is not None

    @property
    def prepared(self) -> bool:
        """Whether CPU-only assembly is ready for the next fenced GPU wake."""
        return self._prepared_model is not None

    def prepare_cpu(self) -> None:
        """Build packed model tensors in RAM without taking Worker GPU residency."""
        if self.loaded or self.prepared or self._model_preparer is None:
            return
        if self._failed:
            raise GradientOwnerError("gradient owner failed before CPU preparation")
        with self._preparation_lock:
            if self.loaded or self.prepared:
                return
            prepared = self._model_preparer()
            if prepared is None:
                raise GradientOwnerError("CPU model preparer returned no prepared model")
            self._prepared_model = prepared

    @property
    def engine(self) -> Any:
        if self._engine is None:
            raise GradientOwnerError("gradient engine has not been loaded")
        return self._engine

    @property
    def model(self) -> Any:
        if self._model is None:
            raise GradientOwnerError("gradient model has not been loaded")
        return self._model

    def _named_parameters(self) -> list[tuple[str, Any]]:
        named_parameters = getattr(self.model, "named_parameters", None)
        if not callable(named_parameters):
            raise GradientOwnerError("model_factory result has no named_parameters() model")
        return list(named_parameters())

    def _named_buffers(self) -> list[tuple[str, Any]]:
        named_buffers = getattr(self.model, "named_buffers", None)
        if not callable(named_buffers):
            raise GradientOwnerError("gradient model has no named_buffers()")
        return list(named_buffers())

    def _persistent_inventory(self) -> tuple[list[tuple[str, Any]], list[Any], list[Any]]:
        parameters = self._named_parameters()
        buffers = self._named_buffers()
        if not parameters:
            raise GradientOwnerError("gradient model has no parameters")
        trainable = [value for _, value in parameters if bool(getattr(value, "requires_grad", False))]
        if not trainable:
            raise GradientOwnerError("gradient model has no trainable parameters")
        if any(getattr(getattr(value, "device", None), "type", None) != "cuda"
               for value in trainable):
            raise GradientOwnerError("every trainable gradient parameter must be CUDA resident")
        frozen = [value for _, value in parameters if not bool(getattr(value, "requires_grad", False))]
        frozen.extend(value for _, value in buffers)
        mutable = trainable
        labeled = [(f"parameter:{name}", value) for name, value in parameters]
        labeled.extend((f"buffer:{name}", value) for name, value in buffers)
        return labeled, frozen, mutable

    def _packed_weight_offloader(self):
        return getattr(self.engine, "packed_weight_offloader", None)

    def _reclaim_host_cache(self, *, segment_bytes: int | None = None):
        import atexit
        from functools import partial
        from ..bounded_offload import reclaim_idle_pinned_cache
        from ..cumem_backup import FROZEN_SCRATCH_BYTES
        from ..pinned_cache import IdlePinnedCacheMonitor

        torch = self._torch_module()
        if type(segment_bytes) is not int or segment_bytes <= 0:
            segment_bytes = None
        budget = getattr(self.residency, "host_backup_budget_bytes", 0)
        reserve_bytes = budget
        reserve_source = "configured_budget_fallback"
        if (segment_bytes is not None
                and type(budget) is int and budget >= 0):
            reserve_bytes = min(budget, segment_bytes + FROZEN_SCRATCH_BYTES)
            reserve_source = "validated_cumem_segments"
        reclaim = partial(
            reclaim_idle_pinned_cache,
            empty_host_cache=torch.accelerator.empty_host_cache,
            host_memory_stats=torch.cuda.memory.host_memory_stats,
            retain_if_safe=True,
            reserve_bytes=reserve_bytes,
        )
        report = reclaim()
        policy = report.get("cache_policy")
        if not isinstance(policy, dict):
            policy = {}
            report["cache_policy"] = policy
        policy.update({
            "segment_backup_bytes": segment_bytes,
            "reserve_bytes": reserve_bytes,
            "reserve_source": reserve_source,
        })
        if report["cache_policy"]["retained"]:
            if self._host_cache_monitor is None:
                # Once asleep, the backup is already charged to current usage.
                self._host_cache_monitor = IdlePinnedCacheMonitor(
                    partial(reclaim, reserve_bytes=0),
                )
                # Registered after CuMem initialization: stop before its teardown.
                atexit.register(self._host_cache_monitor.close)
        return report

    def _assert_no_gradients(self) -> None:
        active = [name for name, parameter in self._named_parameters()
                  if getattr(parameter, "grad", None) is not None]
        if active:
            sample = ", ".join(active[:4])
            raise GradientOwnerError(
                "copy gradient contributions to owned CPU memory and clear parameter grads before park; "
                f"still set on {sample}"
            )

    def _load_once(self) -> dict[str, Any]:
        if self._failed:
            raise GradientOwnerError("persistent gradient owner failed during model initialization")
        if self.loaded:
            raise GradientOwnerError("persistent gradient model is already loaded")
        if getattr(self.residency, "state", None) != "uninitialized":
            raise GradientOwnerError(
                f"cannot construct gradient model while residency is {self.residency.state}"
            )
        try:
            with self.residency.allocation_context():
                prepared = self._prepared_model
                if prepared is not None:
                    engine = self._factory(prepared)
                else:
                    engine = self._factory()
                model = self._model_getter(engine)
                if model is None:
                    raise GradientOwnerError("model_getter returned no gradient model")
                self._engine = engine
                self._model = model
                self._prepared_model = None
                inventory, _, _ = self._persistent_inventory()
                ownership = self.residency.assert_tensors_owned(inventory)
            self._assert_no_gradients()
            # Streaming replaces the initially loaded CUDA packed buffers with
            # CPU masters. Release those now-inactive pool blocks before any
            # forward allocates temporaries in the ordinary allocator.
            if self._packed_weight_offloader() is not None:
                self.residency.reclaim_inactive_cache()
            if not isinstance(ownership, dict) or ownership.get("tensor_count", 0) <= 0:
                raise GradientOwnerError("initial model load did not prove persistent CUDA ownership")
            return {"state": "awake", "loaded": True, "persistent_storage": ownership}
        except BaseException:
            self._failed = True
            raise

    def wake(self) -> dict[str, Any]:
        """Load the model once, then restore the same parked allocations."""
        if self._host_cache_monitor is not None:
            self._host_cache_monitor.stop()
        if self._failed:
            raise GradientOwnerError("persistent gradient owner is failed")
        if not self.loaded:
            return self._load_once()
        state = getattr(self.residency, "state", None)
        if state == "asleep":
            try:
                result = self.residency.wake()
                offloader = self._packed_weight_offloader()
                if offloader is not None:
                    offloader.resume("forward")
            except BaseException:
                self._failed = True
                raise
        elif state == "awake":
            result = self.residency.memory_report()
        else:
            raise GradientOwnerError(f"cannot wake gradient model from residency state {state}")
        self.health("awake")
        return {"state": "awake", "loaded": True, "residency": result}

    def _torch_module(self) -> Any:
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _drain_cuda(self) -> None:
        torch = self._torch_module()
        device = getattr(self.residency, "device", None)
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device)

    def park(self) -> dict[str, Any]:
        """Clear inactive cache, prove model ownership, and park its live pool."""
        from .phase_timing import JobPhaseTiming

        timer = JobPhaseTiming()
        try:
            return self._park(timer)
        finally:
            self.last_park_timing = timer.snapshot()

    def _park(self, timer) -> dict[str, Any]:
        self.last_host_cache_reclaim = None
        if self._failed:
            raise GradientOwnerError("failed gradient owner cannot safely park")
        if not self.loaded:
            raise GradientOwnerError("cannot park an unloaded gradient model")
        if getattr(self.residency, "state", None) != "awake":
            raise GradientOwnerError(
                f"cannot park gradient model while residency is {self.residency.state}"
            )
        trainer = getattr(self._engine, "trainer", None)
        if getattr(trainer, "_gradient_compute_active", False):
            raise GradientOwnerError("cannot park while a gradient computation is active")
        with timer.measure('gradient_clear_check'):
            self._assert_no_gradients()
        offloader = self._packed_weight_offloader()
        host_cache = None
        backup_segment_bytes = None
        if offloader is not None:
            # No CPU backup or Worker wake may start while streaming leases or
            # DMA transfers still retain an autograd-owned CUDA packed view.
            with timer.measure('packed_offloader_park'):
                offloader.park()
        with timer.measure('inactive_cuda_cache_reclaim'):
            cache = self.residency.reclaim_inactive_cache()
        if isinstance(cache, dict):
            value = cache.get("tracked_bytes_after")
            if type(value) is int and value > 0:
                backup_segment_bytes = value
        if offloader is not None:
            with timer.measure('inactive_pinned_cache_reclaim'):
                host_cache = self._reclaim_host_cache(segment_bytes=backup_segment_bytes)
            if isinstance(host_cache, dict):
                policy = host_cache.get("cache_policy")
                if isinstance(policy, dict):
                    keys = (
                        "retained", "reason", "segment_backup_bytes", "reserve_bytes",
                        "reserve_source", "required_available_bytes", "available_bytes",
                    )
                    self.last_host_cache_reclaim = {
                        key: policy.get(key) for key in keys
                    }
        with timer.measure('persistent_tensor_inventory'):
            inventory, frozen, mutable = self._persistent_inventory()
        with timer.measure('persistent_ownership_check'):
            ownership = self.residency.assert_tensors_owned(inventory)
        with timer.measure('residency_backup_and_park'):
            park_args = {
                "drain": self._drain_cuda,
                "frozen_tensors": frozen,
                "mutable_tensors": mutable,
            }
            if offloader is not None and backup_segment_bytes is not None:
                park_args["expected_backup_bytes"] = backup_segment_bytes
            result = self.residency.park(**park_args)
        if not isinstance(result, dict) or result.get("state") not in {"asleep", "parked"}:
            raise GradientOwnerError("CuMem residency did not prove the gradient model is parked")
        if self.last_host_cache_reclaim is not None:
            actual_segment_bytes = result.get("pool_tracked_bytes")
            if type(actual_segment_bytes) is int and actual_segment_bytes >= 0:
                self.last_host_cache_reclaim["actual_segment_bytes"] = actual_segment_bytes
        if (host_cache is not None
                and host_cache.get("cache_policy", {}).get("retained")
                and self._host_cache_monitor is not None):
            self._host_cache_monitor.start()
        return {
            **result,
            "loaded": True,
            "persistent_storage": ownership,
            "cache_reclaim": cache,
            "host_cache_reclaim": host_cache,
            "mutable_parameter_count": len(mutable),
            "frozen_tensor_count": len(frozen),
        }

    def health(self, expected_state: str) -> dict[str, Any]:
        """Return a live owner proof for lifecycle health callbacks."""
        if expected_state not in {"awake", "parked"}:
            raise ValueError("expected_state must be awake or parked")
        if self._failed:
            raise GradientOwnerError("compute health cannot pass for a failed gradient owner")
        if not self.loaded:
            raise GradientOwnerError("compute health cannot pass before the model is loaded")
        residency_state = getattr(self.residency, "state", None)
        if expected_state == "awake":
            if residency_state != "awake":
                raise GradientOwnerError(f"compute residency is {residency_state}, expected awake")
            inventory, _, _ = self._persistent_inventory()
            ownership = self.residency.assert_tensors_owned(inventory)
            return {"state": "awake", "ready": True, "persistent_storage": ownership}
        if residency_state != "asleep":
            raise GradientOwnerError(f"compute residency is {residency_state}, expected parked")
        self._assert_no_gradients()
        return {"state": "parked", "ready": False}
