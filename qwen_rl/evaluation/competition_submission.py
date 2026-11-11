"""Convert independently recorded test-target guesses to ARC submission format."""
from __future__ import annotations


def competition_submission(task_id: str, candidates: list[dict[int, list[list[int]]]]) -> tuple[dict, list[int]]:
    """Return the submission and zero-based targets requiring a placeholder.

    A single candidate is duplicated. With no candidate, [[0]] is a syntactically
    valid placeholder, explicitly identified in the returned missing-target list.
    This function does not read or infer hidden target outputs.
    """
    if not task_id or not candidates:
        raise ValueError("task_id and at least one test target are required")
    rows, missing = [], []
    for index, slots in enumerate(candidates):
        if set(slots) - {1, 2}:
            raise ValueError("attempt slots must be 1 or 2")
        for grid in slots.values():
            if (not isinstance(grid, list) or not grid or len(grid) > 30
                    or any(not isinstance(row, list) or not row or len(row) > 30 for row in grid)
                    or any(len(row) != len(grid[0]) for row in grid)
                    or any(type(cell) is not int or not 0 <= cell <= 9 for row in grid for cell in row)):
                raise ValueError("candidate must be a rectangular ARC grid")
        if not slots:
            missing.append(index)
        first = slots.get(1, slots.get(2, [[0]]))
        second = slots.get(2, first)
        rows.append({"attempt_1": first, "attempt_2": second})
    return {task_id: rows}, missing
