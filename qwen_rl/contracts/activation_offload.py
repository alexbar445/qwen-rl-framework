"""CPU-only activation-offload admission policy shared by production and probes."""
from __future__ import annotations

DEFAULT_OFFLOAD_MIN_TOKENS = 0


def offload_decision(*, batch_size: int, sequence_length: int,
                     min_tokens: int = DEFAULT_OFFLOAD_MIN_TOKENS) -> dict:
    for name, value in (("batch_size", batch_size), ("sequence_length", sequence_length),
                        ("min_tokens", min_tokens)):
        minimum = 0 if name == "min_tokens" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    tokens = batch_size * sequence_length
    enabled = tokens >= min_tokens
    return {"enabled": enabled, "batch_size": batch_size,
            "sequence_length": sequence_length, "padded_tokens": tokens,
            "min_tokens": min_tokens, "basis": "final_microbatch_padded_batch_times_sequence",
            "reason": "at_or_above_threshold" if enabled else "below_threshold"}
