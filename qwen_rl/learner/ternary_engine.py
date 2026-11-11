"""Hard rejection of the retired single-card T2 builder.

T2 compute must use the maintained Learner assembly, including its managed
packed-weight offloader. This entry cannot silently construct a resident model.
"""
from __future__ import annotations

from typing import Any


class OutdatedTernaryBuilderError(RuntimeError):
    """The caller selected the retired resident single-card T2 builder."""


class TernaryLearnerEngine:
    def __init__(self, config: dict[str, Any]):
        raise OutdatedTernaryBuilderError(
            "outdated single-card T2 builder is disabled; "
            "use the maintained T2 Learner assembly with packed-weight offload; "
            "there is no resident-builder fallback"
        )


__all__ = ["OutdatedTernaryBuilderError", "TernaryLearnerEngine"]
