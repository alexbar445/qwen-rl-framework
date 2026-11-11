"""References held while a published policy file is being served."""
from __future__ import annotations

from contextlib import contextmanager
import threading
from collections.abc import Iterator, Mapping
from typing import Any


class PolicyFileReferences:
    """Keep a policy publication alive until its FileResponse finishes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._references: dict[int, int] = {}

    def acquire(self, version: int, policies: Mapping[int, dict[str, Any]]) -> dict[str, Any] | None:
        with self._lock:
            policy = policies.get(version)
            if policy is None:
                return None
            self._references[version] = self._references.get(version, 0) + 1
            return policy

    def release(self, version: int) -> None:
        with self._lock:
            references = self._references.get(version, 0)
            if references < 1:
                raise ValueError(f"policy version {version} has no outstanding file response")
            if references == 1:
                del self._references[version]
            else:
                self._references[version] = references - 1

    @contextmanager
    def cleanup(self) -> Iterator[frozenset[int]]:
        """Serialize directory retirement with new file-response references."""
        with self._lock:
            yield frozenset(self._references)
