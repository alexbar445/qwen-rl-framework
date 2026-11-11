"""Warm actual Flash Linear Attention GDN backward autotune profiles."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch


_GDN_CLASS_NAME = "Qwen3_5GatedDeltaNet"
_L2NORM_NB_TOKENS = 2048 * 32
_ProfileKey = tuple[str, int, int | None, torch.dtype]
_ATTEMPTED_PROFILES: set[_ProfileKey] = set()
_WARMED_PROFILES: set[_ProfileKey] = set()
_FAILED_PROFILES: dict[_ProfileKey, str] = {}


@dataclass(frozen=True)
class L2NormBackwardProfile:
    device: torch.device
    head_dim: int
    num_value_heads: int
    nb: int | None
    dtype: torch.dtype
    synthetic_tokens: int
    actual_shapes: tuple[str, ...]

    @property
    def key(self) -> _ProfileKey:
        return (
            str(self.device),
            self.head_dim,
            self.nb,
            self.dtype,
        )

    @property
    def label(self) -> str:
        nb_label = "na" if self.nb is None else str(self.nb)
        return f"{self.device}/D{self.head_dim}/H{self.num_value_heads}/NB{nb_label}/{self.dtype}"


class GDNWarmupError(RuntimeError):
    """A GDN autotune warmup failed before learner backward began."""

    def __init__(self, message: str, *, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


def _make_profile(
    *,
    device: torch.device,
    head_dim: int,
    num_value_heads: int,
    batch_size: int,
    sequence_length: int,
    dtype: torch.dtype,
    actual_shape: str,
) -> L2NormBackwardProfile:
    if head_dim < 1 or num_value_heads < 1 or batch_size < 1 or sequence_length < 1:
        raise ValueError("GDN warmup dimensions must all be positive")
    if dtype is not torch.bfloat16:
        raise ValueError(
            "CUDA GDN warmup supports only the configured BF16 compute profile; "
            f"received {dtype}"
        )

    flattened_tokens = batch_size * sequence_length * num_value_heads
    # FLA 0.5.2 keys its grouped l2norm autotuner on D and
    # NB=ceil(T/(2048*32)); T itself is excluded from specialization.
    nb = math.ceil(flattened_tokens / _L2NORM_NB_TOKENS) if head_dim <= 512 else None
    # Keep T representative of the selected learner layout. For rows sharing
    # this D/NB/dtype cache profile, _merge_profile_shapes retains the smallest
    # actual flattened T rather than the artificial lower edge of the NB bucket.
    synthetic_tokens = flattened_tokens
    return L2NormBackwardProfile(
        device=device,
        head_dim=head_dim,
        num_value_heads=num_value_heads,
        nb=nb,
        dtype=dtype,
        synthetic_tokens=synthetic_tokens,
        actual_shapes=(actual_shape,),
    )


def _merge_profile_shapes(
    profiles: dict[_ProfileKey, L2NormBackwardProfile],
    profile: L2NormBackwardProfile,
) -> None:
    previous = profiles.get(profile.key)
    if previous is None:
        profiles[profile.key] = profile
        return
    profiles[profile.key] = L2NormBackwardProfile(
        device=profile.device,
        head_dim=profile.head_dim,
        num_value_heads=profile.num_value_heads,
        nb=profile.nb,
        dtype=profile.dtype,
        synthetic_tokens=min(previous.synthetic_tokens, profile.synthetic_tokens),
        actual_shapes=(*previous.actual_shapes, *profile.actual_shapes),
    )


def plan_gdn_l2norm_backward_warmup(
    model: torch.nn.Module,
    microbatches: list[dict[str, Any]],
    *,
    compute_dtype: torch.dtype | None,
) -> tuple[list[L2NormBackwardProfile], dict[str, Any]]:
    """Plan profiles from final trimmed microbatch tensors and real HF GDN attrs."""
    gdn_modules = [module for module in model.modules() if type(module).__name__ == _GDN_CLASS_NAME]
    if not gdn_modules:
        return [], {"status": "skipped", "reason": "non_gdn", "microbatches": len(microbatches)}

    devices: set[torch.device] = set()
    for module in gdn_modules:
        try:
            parameter = next(module.parameters())
        except StopIteration as exc:
            raise ValueError("Qwen3_5GatedDeltaNet has no parameters to identify its execution device") from exc
        devices.add(parameter.device)
    if len(devices) != 1:
        raise ValueError(f"GDN layers span multiple devices; refusing ambiguous warmup: {sorted(map(str, devices))}")
    device = next(iter(devices))
    if device.type != "cuda":
        return [], {"status": "skipped", "reason": "non_cuda", "device": str(device), "microbatches": len(microbatches)}
    if compute_dtype is None or compute_dtype is not torch.bfloat16:
        raise ValueError(
            "CUDA Qwen3_5GatedDeltaNet warmup requires an explicit BF16 compute dtype; "
            f"received {compute_dtype}"
        )

    profiles: dict[_ProfileKey, L2NormBackwardProfile] = {}
    actual_shapes: list[str] = []
    for batch_index, batch in enumerate(microbatches):
        prompt_ids = batch.get("prompt_ids")
        completion_ids = batch.get("completion_ids")
        if not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2:
            raise ValueError(f"GDN warmup microbatch {batch_index} has no 2-D prompt_ids")
        if not isinstance(completion_ids, torch.Tensor) or completion_ids.ndim != 2:
            raise ValueError(f"GDN warmup microbatch {batch_index} has no 2-D completion_ids")
        if prompt_ids.shape[0] != completion_ids.shape[0]:
            raise ValueError(f"GDN warmup microbatch {batch_index} prompt/completion row counts differ")
        batch_size = int(prompt_ids.shape[0])
        sequence_length = int(prompt_ids.shape[1] + completion_ids.shape[1])
        actual_shapes.append(f"B{batch_size}xL{sequence_length}")
        for module in gdn_modules:
            head_dim = getattr(module, "head_k_dim", None)
            num_value_heads = getattr(module, "num_v_heads", None)
            if (isinstance(head_dim, bool) or not isinstance(head_dim, int)
                    or isinstance(num_value_heads, bool) or not isinstance(num_value_heads, int)):
                raise ValueError("Qwen3_5GatedDeltaNet must expose integer head_k_dim and num_v_heads")
            profile = _make_profile(
                device=device,
                head_dim=head_dim,
                num_value_heads=num_value_heads,
                batch_size=batch_size,
                sequence_length=sequence_length,
                dtype=compute_dtype,
                actual_shape=f"mb{batch_index}:B{batch_size}xL{sequence_length}",
            )
            _merge_profile_shapes(profiles, profile)

    planned = list(profiles.values())
    return planned, {
        "status": "planned",
        "device": str(device),
        "microbatches": len(microbatches),
        "actual_shapes": actual_shapes,
        "profiles_required": len(planned),
        "profile_labels": [profile.label for profile in planned],
    }


def warmup_gdn_l2norm_backward(
    model: torch.nn.Module,
    microbatches: list[dict[str, Any]],
    *,
    compute_dtype: torch.dtype | None,
) -> dict[str, Any]:
    """Run one public FLA backward call for each unseen actual profile per process."""
    profiles, metadata = plan_gdn_l2norm_backward_warmup(
        model, microbatches, compute_dtype=compute_dtype
    )
    if not profiles:
        return metadata

    previously_failed = [profile for profile in profiles if profile.key in _FAILED_PROFILES]
    if previously_failed:
        profile = previously_failed[0]
        metadata.update({
            "status": "previous_attempt_failed",
            "failed_profile": profile.label,
            "profiles_already_attempted": sum(
                item.key in _ATTEMPTED_PROFILES for item in profiles
            ),
            "profiles_attempted_total_process": len(_ATTEMPTED_PROFILES),
            "profiles_warmed_total_process": len(_WARMED_PROFILES),
        })
        raise GDNWarmupError(
            f"FLA l2norm_bwd profile {profile.label} failed earlier in this process; "
            f"automatic retry is disabled; previous_error={_FAILED_PROFILES[profile.key]}",
            metadata=metadata,
        )

    new_profiles = [profile for profile in profiles if profile.key not in _ATTEMPTED_PROFILES]
    metadata["profiles_already_attempted"] = len(profiles) - len(new_profiles)
    metadata["profiles_already_warmed"] = sum(
        profile.key in _WARMED_PROFILES for profile in profiles
    )
    metadata["profiles_attempted_this_update"] = 0
    metadata["profiles_warmed_this_update"] = 0
    metadata["warmup_seconds"] = 0.0
    metadata["profiles_attempted_total_process"] = len(_ATTEMPTED_PROFILES)
    metadata["profiles_warmed_total_process"] = len(_WARMED_PROFILES)
    if not new_profiles:
        metadata["status"] = "already_warmed"
        return metadata

    # Release only inactive cached blocks once per warmup phase. There is no retry.
    torch.cuda.empty_cache()
    from fla.modules.l2norm import l2norm_bwd

    started = time.perf_counter()
    for profile in new_profiles:
        # Record attempts before execution so a failed autotune is never retried
        # implicitly by a later batch in the same learner process.
        _ATTEMPTED_PROFILES.add(profile.key)
        metadata["profiles_attempted_this_update"] += 1
        metadata["profiles_attempted_total_process"] = len(_ATTEMPTED_PROFILES)
        try:
            with torch.no_grad():
                y = torch.zeros(
                    (profile.synthetic_tokens, profile.head_dim),
                    dtype=profile.dtype,
                    device=profile.device,
                )
                dy = torch.zeros(
                    (profile.synthetic_tokens, profile.head_dim),
                    dtype=profile.dtype,
                    device=profile.device,
                )
                rstd = torch.ones(
                    (profile.synthetic_tokens,), dtype=torch.float32, device=profile.device
                )
                dx = l2norm_bwd(y, rstd, dy)
                torch.cuda.synchronize(profile.device)
                del dx, rstd, dy, y
        except Exception as exc:
            _FAILED_PROFILES[profile.key] = f"{type(exc).__name__}: {exc}"
            metadata.update({
                "status": "failed",
                "failed_profile": profile.label,
                "warmup_seconds": time.perf_counter() - started,
            })
            raise GDNWarmupError(
                f"FLA l2norm_bwd warmup failed for {profile.label}; metadata={metadata}",
                metadata=metadata,
            ) from exc
        _WARMED_PROFILES.add(profile.key)
        metadata["profiles_warmed_this_update"] += 1

    metadata["status"] = "warmed"
    metadata["profiles_warmed_total_process"] = len(_WARMED_PROFILES)
    metadata["warmup_seconds"] = time.perf_counter() - started
    return metadata
