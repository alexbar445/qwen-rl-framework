"""Gradient-only TRL DAPO execution and ownership primitives."""

from .compute import GradientComputeExecutor, build_compute
from .lifecycle import (
    GradientLifecycleError,
    LifecycleGradientCompute,
    SameNodeGradientCycle,
)
from .owner import GradientOwnerError, PersistentGradientOwner

__all__ = [
    "GradientComputeExecutor",
    "GradientLifecycleError",
    "GradientOwnerError",
    "LifecycleGradientCompute",
    "PersistentGradientOwner",
    "SameNodeGradientCycle",
    "build_compute",
]
