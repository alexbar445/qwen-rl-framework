"""Strict public evidence for frozen native NVFP4 weights."""
from __future__ import annotations

from typing import Any

MODES = frozenset({"torchao"})


def quantization_evidence(value: Any, *, expected_mode: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("model omitted its quantization evidence")
    mode = value.get("mode")
    dtype = str(value.get("compute_dtype", "")).removeprefix("torch.").lower()
    if mode not in MODES or (expected_mode is not None and mode != expected_mode):
        raise ValueError("model quantization mode does not match the configured backend")
    if isinstance(value.get("bits"), bool) or value.get("bits") != 4 or value.get("packed_resident") is not True or dtype != "bfloat16":
        raise ValueError("model must prove frozen packed four-bit storage and BF16 compute")
    if mode == "torchao" and value.get("quantizer") != "NVFP4WeightOnlyConfig":
        raise ValueError("native model must prove NVFP4WeightOnlyConfig")
    return {**value, "compute_dtype": "bfloat16"}
