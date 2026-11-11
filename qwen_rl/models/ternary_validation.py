"""Bounded CPU NumPy bitwise validator for packed T2 payloads."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any


_ROWS_PER_GROUP = 8192
_MAX_TEMP_ELEMENTS = 8 * 1024 * 1024
_T2_CODE_INVALID = "T2 code 0b10 is invalid; legal two's-complement codes are {-1, 0, +1}"
_T2_SCALE_INVALID = "T2 scales must be finite FP16 values"
_T2_INPUT_BASES = {"primal", "prism_rotated"}


def validate_t2_input_basis_inventory(
    actual: Mapping[str, str], expected: Mapping[str, str] | None = None,
) -> dict[str, int]:
    """Validate a model's per-module T2 basis inventory against artifact metadata."""
    if not isinstance(actual, Mapping) or not actual:
        raise ValueError("T2 input basis inventory must be a nonempty mapping")
    for name, basis in actual.items():
        if not isinstance(name, str) or not name:
            raise ValueError("T2 input basis module names must be nonempty strings")
        if not isinstance(basis, str) or basis not in _T2_INPUT_BASES:
            raise ValueError(f"unsupported T2 input basis for {name}: {basis!r}")
    if expected is not None:
        if not isinstance(expected, Mapping) or not expected:
            raise ValueError("artifact T2 input basis inventory must be a nonempty mapping")
        for name, basis in expected.items():
            if not isinstance(name, str) or not name:
                raise ValueError("artifact T2 basis module names must be nonempty strings")
            if not isinstance(basis, str) or basis not in _T2_INPUT_BASES:
                raise ValueError(f"unsupported artifact T2 input basis for {name}: {basis!r}")
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ValueError(
                f"T2 input basis module inventory differs from artifact metadata; "
                f"missing={missing[:4]}, extra={extra[:4]}"
            )
        mismatched = [name for name in actual if actual[name] != expected[name]]
        if mismatched:
            sample = mismatched[:4]
            raise ValueError(
                "assembled T2 module basis differs from artifact metadata: "
                + ", ".join(f"{name}={actual[name]} expected={expected[name]}" for name in sample)
            )
    return dict(Counter(actual.values()))


def validate_t2_cpu(packed_codes: Any, scale_words: Any) -> None:
    """Validate pre-shaped CPU T2 tensors with two bounded uint8 scratch tiles.

    The caller owns shape, dtype, and device validation. Each temporary byte
    array has at most 8 Mi elements; combined code-validation scratch is at
    most 16 MiB. The row-group/code-before-scale order matches the Torch path.
    """
    if packed_codes.device.type != "cpu" or scale_words.device.type != "cpu":
        raise ValueError("validate_t2_cpu requires CPU tensors")

    import numpy as np

    codes = packed_codes.detach().numpy()
    raw_scales = scale_words.detach().numpy()
    scales = raw_scales.view(np.float16) if raw_scales.dtype == np.uint16 else raw_scales

    rows, code_columns = codes.shape
    _, scale_columns = scales.shape
    for row_begin in range(0, rows, _ROWS_PER_GROUP):
        row_end = min(rows, row_begin + _ROWS_PER_GROUP)
        row_count = row_end - row_begin
        column_chunk = max(1, _MAX_TEMP_ELEMENTS // row_count)
        for column_begin in range(0, code_columns, column_chunk):
            code_tile = codes[
                row_begin:row_end,
                column_begin : min(code_columns, column_begin + column_chunk),
            ]
            invalid = code_tile.copy()
            np.right_shift(invalid, np.uint8(1), out=invalid)
            not_codes = code_tile.copy()
            np.bitwise_not(not_codes, out=not_codes)
            np.bitwise_and(not_codes, np.uint8(0x55), out=not_codes)
            np.bitwise_and(invalid, not_codes, out=invalid)
            if invalid.any():
                raise ValueError(_T2_CODE_INVALID)
            del invalid, not_codes

        for column_begin in range(0, scale_columns, column_chunk):
            scale_tile = scales[
                row_begin:row_end,
                column_begin : min(scale_columns, column_begin + column_chunk),
            ]
            if not np.isfinite(scale_tile).all():
                raise ValueError(_T2_SCALE_INVALID)
