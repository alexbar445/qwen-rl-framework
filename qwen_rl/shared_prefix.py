"""Stable system/tool boundaries from the selected CT renderer; no prompt rewriting."""
from __future__ import annotations


async def stable_system_prefix(build_initial_tokens, messages, tools):
    leading = []
    for message in messages:
        if message.get("role") not in {"system", "developer"}:
            break
        leading.append(dict(message))
    refs = [await build_initial_tokens(
        leading + [{"role": "user", "content": value}], tools=tools,
    ) for value in ("A", "Z")]
    n = 0
    for a, b in zip(*refs):
        if a != b:
            break
        n += 1
    if n == min(map(len, refs)):
        raise ValueError("CT renderer did not distinguish different user messages")
    return tuple(refs[0][:n])
