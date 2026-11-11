"""Shared T2 packed-weight plan, CPU master, and phase-lifecycle integration."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any


class T2PackedOffloadError(RuntimeError):
    """The identity-bound packed T2 offloader could not be installed safely."""


@dataclass(frozen=True)
class T2PackedWeightLayout:
    base: Any
    decoder: tuple[Any, ...]
    plan: Any
    identity: dict[str, Any]
    group_sizes: tuple[int, ...]
    device: Any

    def write_group(self, index, stream) -> None:
        import torch

        group = self.plan.groups[index]
        cursor = 0
        for entry in group.entries:
            if entry.offset > cursor:
                stream.write(bytes(entry.offset - cursor))
            source = entry.module._buffers.get(entry.name)
            if (not isinstance(source, torch.Tensor) or source.device != self.device
                    or source.requires_grad):
                raise ValueError(f"{entry.name} is not the expected frozen CUDA buffer")
            raw = source.detach().view(torch.uint8).reshape(-1).cpu()
            if raw.numel() != entry.nbytes:
                raise ValueError(f"{entry.name} changed byte size while writing shared master")
            stream.write(memoryview(raw.numpy()))
            cursor = entry.offset + entry.nbytes
        if cursor < group.storage_bytes:
            stream.write(bytes(group.storage_bytes - cursor))


def build_t2_packed_weight_layout(
    model,
    *,
    device,
    artifact_identity: Mapping[str, Any],
    max_cpu_master_bytes: int,
    max_pinned_bytes: int,
) -> T2PackedWeightLayout:
    """Build the canonical decoder layout used by DDP and gradient executors."""
    import torch
    from qwen_rl.models.ternary_tp_offload import build_packed_weight_plan

    if not isinstance(artifact_identity, Mapping) or not artifact_identity:
        raise ValueError("packed T2 offload requires an artifact identity")
    if type(max_cpu_master_bytes) is not int or max_cpu_master_bytes <= 0:
        raise ValueError("packed T2 CPU master cap must be a positive integer")
    if type(max_pinned_bytes) is not int or max_pinned_bytes <= 0:
        raise ValueError("packed T2 pinned staging cap must be a positive integer")
    device = torch.device(device)
    get_base = getattr(model, "get_base_model", None)
    base = get_base() if callable(get_base) else model
    raw_decoder = getattr(getattr(base, "model", None), "layers", None)
    if raw_decoder is None:
        raise ValueError("T2 model has no ordered decoder layer inventory")
    decoder = tuple(raw_decoder)
    plan = build_packed_weight_plan(decoder, device=device)
    if plan.cpu_master_bytes > max_cpu_master_bytes:
        raise ValueError(
            "shared CPU master exceeds configured cap "
            f"({plan.cpu_master_bytes} > {max_cpu_master_bytes} bytes)"
        )
    if plan.pinned_staging_bytes > max_pinned_bytes:
        raise ValueError(
            "per-executor pinned staging exceeds configured cap "
            f"({plan.pinned_staging_bytes} > {max_pinned_bytes} bytes)"
        )
    module_paths = {id(module): name for name, module in base.named_modules()}
    group_inventory = []
    for group in plan.groups:
        owner_path = module_paths.get(id(group.owner))
        if owner_path is None:
            raise ValueError(f"decoder group {group.index} has no stable module path")
        entries = []
        for entry in group.entries:
            module_path = module_paths.get(id(entry.module))
            if module_path is None:
                raise ValueError("packed T2 module has no stable model path")
            entries.append({
                "module_path": module_path,
                "buffer": entry.name,
                "offset": entry.offset,
                "nbytes": entry.nbytes,
                "shape": list(entry.shape),
                "dtype": str(entry.dtype),
            })
        group_inventory.append({
            "index": group.index,
            "owner_path": owner_path,
            "storage_bytes": group.storage_bytes,
            "entries": entries,
        })
    identity = {
        "artifact_identity": dict(artifact_identity),
        "layout": "qwen35_t2_ddp_decoder_packed/v1",
        "groups": group_inventory,
    }
    return T2PackedWeightLayout(
        base=base,
        decoder=decoder,
        plan=plan,
        identity=identity,
        group_sizes=tuple(group.storage_bytes for group in plan.groups),
        device=device,
    )


def open_t2_shared_packed_store(layout: T2PackedWeightLayout, store_root: str | Path):
    """Open the identity-keyed canonical master shared by local executors/ranks."""
    from qwen_rl.models.shared_packed_store import SharedPackedStore

    root = Path(store_root).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("packed-weight store root must be an absolute non-symlink directory")
    return SharedPackedStore.open_or_create(
        root, layout.identity, layout.group_sizes, write_group=layout.write_group,
    )


def install_t2_packed_offloader(
    model,
    layout: T2PackedWeightLayout,
    store,
    *,
    config: Mapping[str, Any],
):
    """Install the formal bounded streamer and optional nested decoder checkpoints."""
    from qwen_rl.models.ternary_tp_offload import PackedTernaryTPOffloader

    max_cpu_master_bytes = int(config["packed_weight_offload_max_cpu_master_bytes"])
    max_pinned_bytes = int(config["packed_weight_offload_max_pinned_bytes"])
    prefetch = config["packed_weight_offload_prefetch"]
    if type(prefetch) is not bool:
        raise ValueError("packed_weight_offload_prefetch must be a boolean")
    offloader = None
    nested = None
    try:
        offloader = PackedTernaryTPOffloader(
            layout.decoder,
            device=layout.device,
            max_cpu_master_bytes=max_cpu_master_bytes,
            max_pinned_bytes=max_pinned_bytes,
            prefetch=prefetch,
            master_provider=store,
        )
        if config.get("gradient_checkpointing", True) and config.get("activation_offload", True):
            from qwen_rl.learner.nested_decoder_checkpoint import NestedDecoderCheckpoints

            nested = NestedDecoderCheckpoints(layout.decoder, offloader)
            nested.install()
    except BaseException:
        if nested is not None:
            nested.close()
        if offloader is not None:
            try:
                offloader.close()
            except Exception:
                pass
        raise

    # Publish the managed marker only after the CPU views and all hooks install.
    layout.base._t2_packed_weight_offloader = offloader
    model._t2_packed_weight_offloader = offloader
    model._t2_managed_packed_weights = True
    return offloader, nested


def install_t2_frozen_embedding_input_policy(model, *, enabled: bool):
    """Install the shared non-reentrant LoRA input policy before Trainer setup."""
    if not enabled or getattr(model, "_t2_gdn_execution", None) is None:
        return None
    from qwen_rl.models.ternary_checkpoint_inputs import FrozenEmbeddingInputPolicy

    get_base = getattr(model, "get_base_model", None)
    base = get_base() if callable(get_base) else model
    policy = FrozenEmbeddingInputPolicy(base, use_reentrant=False)
    model._t2_frozen_embedding_input_policy = policy
    return policy


def protect_preplaced_model(accelerator, target_model) -> None:
    """Keep Accelerator from rematerializing a managed CPU-view T2 base."""
    original = getattr(accelerator, "prepare_model", None)
    if not callable(original):
        raise T2PackedOffloadError("Accelerator has no callable prepare_model method")

    @wraps(original)
    def prepare_model(model, *args, **kwargs):
        if model is target_model:
            if args:
                args = (False, *args[1:])
            else:
                kwargs["device_placement"] = False
        return original(model, *args, **kwargs)

    accelerator.prepare_model = prepare_model


def packed_weight_phase_hooks(model, offloader) -> list[Any]:
    """Set the forward phase around calls that bypass accelerator.backward."""
    import torch

    def enter_forward(_module, _args):
        offloader.set_phase("forward")

    def leave_no_grad_forward(_module, _args, _output):
        if not torch.is_grad_enabled():
            offloader.set_phase("idle")

    return [
        model.register_forward_pre_hook(enter_forward),
        model.register_forward_hook(leave_no_grad_forward, always_call=True),
    ]


def wrap_packed_weight_backward(backward, offloader):
    """Bracket checkpoint recomputation with reverse-order packed leases."""
    @wraps(backward)
    def wrapped(loss, *args, **kwargs):
        offloader.set_phase("backward")
        try:
            return backward(loss, *args, **kwargs)
        finally:
            offloader.set_phase("idle")

    return wrapped


def t2_packed_offload_report(
    layout: T2PackedWeightLayout,
    store,
    offloader,
    nested,
    *,
    mode: str,
    max_pinned_bytes: int,
    retained_device_components: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": mode,
        "nested_decoder_checkpoint": nested.report if nested is not None else None,
        "artifact_identity_bound": True,
        "layout_group_count": len(layout.plan.groups),
        "cpu_master_bytes": layout.plan.cpu_master_bytes,
        "cpu_master_copy_scope": "single canonical file-backed mapping shared by local ranks",
        "persistent_per_rank_cpu_master_allocations": 0,
        "shared_file_writer_scope": "single lock winner, one packed source buffer at a time",
        "max_packed_source_buffer_bytes": max(
            entry.nbytes for group in layout.plan.groups for entry in group.entries
        ),
        "per_rank_pinned_staging_bytes": layout.plan.pinned_staging_bytes,
        "per_rank_pinned_staging_cap_bytes": max_pinned_bytes,
        "lease_scope": "one decoder layer per forward or checkpoint recompute",
        "moved_buffers": ["packed_codes", "scale_words"],
        "retained_device_components": list(retained_device_components),
        "shared_store": store.report(),
        "offloader": offloader.report(),
        "non_reentrant_gradient_checkpointing": True,
        "accelerator_model_placement": "preserved_preplaced_target_model_only",
    }
