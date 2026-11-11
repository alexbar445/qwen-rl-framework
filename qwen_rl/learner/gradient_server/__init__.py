"""Central async gradient aggregation and durable adapter optimizer."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import (
        DirtyGradientCoreError,
        GradientCheckpointError,
        GradientServerCore,
        WindowStepReceipt,
    )

__all__ = [
    "DirtyGradientCoreError",
    "GradientCheckpointError",
    "GradientServerCore",
    "WindowStepReceipt",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    core = import_module(".core", __name__)
    value = getattr(core, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
