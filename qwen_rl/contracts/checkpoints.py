"""CPU-only checkpoint retention policy shared by writers and readers."""
from __future__ import annotations


def retained_versions(current_version: int) -> set[int]:
    """Return the archive versions retained at this successful RL step."""
    if current_version < 0:
        raise ValueError("current_version must be non-negative")

    def multiples(step: int, earliest: int) -> set[int]:
        first = ((earliest + step - 1) // step) * step
        return set(range(first, current_version + 1, step))

    return (
        multiples(50, 1)
        | multiples(10, max(1, current_version - 50))
        | multiples(2, max(1, current_version - 10))
    )
