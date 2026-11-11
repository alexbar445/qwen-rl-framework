"""Error and result contracts shared by the complete-group stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class CollectionResult(Generic[T]):
    """Scored items available when a requested batch cannot be completed."""

    items: list[T]
    filtered_count: int
    admitted_count: int


class FilteredLimit(RuntimeError, Generic[T]):
    """Raised when filtered outcomes prevent the oldest batch from filling."""

    def __init__(self, result: CollectionResult[T], *, max_filtered: int) -> None:
        self.result = result
        self.items = result.items
        self.filtered_count = result.filtered_count
        self.admitted_count = result.admitted_count
        self.max_filtered = max_filtered
        super().__init__(
            "filtered group limit reached before the required version bucket filled "
            f"(ready={len(result.items)}, filtered={result.filtered_count}, "
            f"admitted={result.admitted_count}, max_filtered={max_filtered})"
        )
