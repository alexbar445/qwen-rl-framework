"""Learner-only CUDA allocator defaults; does not import Torch or initialize CUDA."""
from __future__ import annotations

import os
from collections.abc import MutableMapping

PYTORCH_ALLOC_CONF_ENV = "PYTORCH_ALLOC_CONF"
PYTORCH_CUDA_ALLOC_CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"
LEARNER_DEFAULT_ALLOCATOR_CONF = "expandable_segments:True"
PHASE_SHARED_ALLOCATOR_CONF = "expandable_segments:False"


def configure_learner_allocator(environ: MutableMapping[str, str] | None = None) -> None:
    """Apply the validated default before CUDA initialization, preserving operator settings."""
    target = os.environ if environ is None else environ
    if PYTORCH_ALLOC_CONF_ENV not in target and PYTORCH_CUDA_ALLOC_CONF_ENV not in target:
        target[PYTORCH_CUDA_ALLOC_CONF_ENV] = LEARNER_DEFAULT_ALLOCATOR_CONF


def configure_phase_shared_allocator(
    environ: MutableMapping[str, str], *, model_backend: str | None, gpu_sharing: str | None
) -> bool:
    """Set allocator policy for a phase-shared Learner child environment only.

    Pinned Torch 2.13 disables expandable segments inside custom MemPools.
    T2 persistent storage therefore stays in CuMem while ordinary training
    temporaries use expandable segments without the legacy split-size cap.
    Callers pass the child environment; this
    helper does not mutate process-global state or import Torch.
    """
    if gpu_sharing != "phase_shared":
        return False
    environ.pop(PYTORCH_CUDA_ALLOC_CONF_ENV, None)
    environ[PYTORCH_ALLOC_CONF_ENV] = (
        LEARNER_DEFAULT_ALLOCATOR_CONF
        if model_backend == "ternary"
        else PHASE_SHARED_ALLOCATOR_CONF
    )
    return True
