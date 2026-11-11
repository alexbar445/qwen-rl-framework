"""Provider usage accounting for uncached input and raw completion tokens."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _nonnegative_int(value: Any) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def uncached_input_tokens(usage: Any) -> int | None:
    """Return prompt tokens excluding cache hits, or None for ambiguous usage."""
    if not isinstance(usage, Mapping):
        return None
    prompt = _nonnegative_int(usage.get("prompt_tokens"))
    if prompt is None:
        return None

    prompt_details = usage.get("prompt_tokens_details")
    input_details = usage.get("input_tokens_details")
    canonical = (
        (prompt_details, "cached_tokens"),
        (input_details, "cached_tokens"),
        (usage, "cached_tokens"),
    )
    cache_hits: list[int] = []
    for container, key in canonical:
        if isinstance(container, Mapping) and key in container:
            value = _nonnegative_int(container.get(key))
            if value is None:
                return None
            cache_hits.append(value)
    if cache_hits and any(value != cache_hits[0] for value in cache_hits[1:]):
        return None
    cached = cache_hits[0] if cache_hits else None

    # OpenAI Chat Completions defines prompt_tokens_details.cached_tokens as
    # the cache-hit count. Some compatible routers also fill extension fields
    # such as cache_read_input_tokens with zero defaults. When the standard
    # field is present, nonzero aliases must agree, while an unrelated zero
    # default does not erase a valid standard count.
    aliases = (
        (prompt_details, "cache_read_input_tokens"),
        (input_details, "cache_read_input_tokens"),
        (usage, "cache_read_input_tokens"),
        (usage, "prompt_cache_hit_tokens"),
    )
    alias_hits: list[int] = []
    for container, key in aliases:
        if isinstance(container, Mapping) and key in container:
            value = _nonnegative_int(container.get(key))
            if value is None:
                return None
            # A zero in a nonstandard extension field may be a default even
            # when the canonical cached_tokens field carries the real count.
            if cached is not None and cached > 0 and value == 0 and key == "cache_read_input_tokens":
                continue
            alias_hits.append(value)
    if cached is None and alias_hits:
        cached = alias_hits[0]
    if alias_hits and any(value != cached for value in alias_hits):
        return None
    if cached is not None and cached > prompt:
        return None

    # DeepSeek reports the complementary miss count too. When present, it must
    # agree with both prompt_tokens and every cache-hit alias. It also directly
    # reports uncached input when the hit field is omitted.
    if "prompt_cache_miss_tokens" in usage:
        miss = _nonnegative_int(usage.get("prompt_cache_miss_tokens"))
        if miss is None or miss > prompt:
            return None
        if cached is not None and miss != prompt - cached:
            return None
        return miss
    return prompt - (cached if cached is not None else 0)


def raw_completion_tokens(usage: Any) -> int | None:
    """Read completion_tokens or derive it from raw total_tokens - prompt_tokens."""
    if not isinstance(usage, Mapping):
        return None
    if "completion_tokens" in usage:
        completion = _nonnegative_int(usage.get("completion_tokens"))
        if completion is not None:
            return completion
        return None
    total = _nonnegative_int(usage.get("total_tokens"))
    prompt = _nonnegative_int(usage.get("prompt_tokens"))
    if total is None or prompt is None or total < prompt:
        return None
    return total - prompt


def sum_uncached_input_tokens(usages: Iterable[Mapping[str, Any]]) -> int | None:
    """Sum valid usage receipts; an empty set is zero and any invalid event is unknown."""
    try:
        iterator = iter(usages)
    except TypeError:
        return None
    total = 0
    for receipt in iterator:
        if not isinstance(receipt, Mapping) or "usage" not in receipt:
            return None
        count = uncached_input_tokens(receipt.get("usage"))
        if count is None:
            return None
        total += count
    return total
