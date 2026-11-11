"""Learner engines and training components.

Keep the package import light so command help and config inspection do not
initialize Torch or CUDA. The public ``LearnerEngine`` package export is
resolved only when a caller asks for it.
"""
from __future__ import annotations

from typing import Any

__all__ = ["LearnerEngine"]


def __getattr__(name: str) -> Any:
    if name == "LearnerEngine":
        from .engine import LearnerEngine

        globals()[name] = LearnerEngine
        return LearnerEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
