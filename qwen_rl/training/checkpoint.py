"""Atomic, weights-only checkpoint storage for the local HF LoRA engine."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import torch


class HFCheckpointError(RuntimeError):
    """A local HF training checkpoint is incomplete or cannot be committed."""


class HFCheckpointStore:
    """Keep one complete recoverable state and replace it atomically."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.latest_path = self.directory / "latest.pt"

    def exists(self) -> bool:
        return self.latest_path.exists()

    def save(self, state: dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.latest_path.is_symlink():
            raise HFCheckpointError("latest checkpoint must not be a symlink")
        fd, temporary_name = tempfile.mkstemp(prefix=".latest.", suffix=".tmp", dir=self.directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.latest_path)
            directory_fd = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise HFCheckpointError(f"could not commit latest checkpoint: {exc}") from exc
        return self.latest_path

    def load(self) -> dict[str, Any]:
        try:
            if self.latest_path.is_symlink() or not self.latest_path.is_file():
                raise HFCheckpointError("latest checkpoint is missing or not a regular file")
            payload = torch.load(self.latest_path, map_location="cpu", weights_only=True)
        except HFCheckpointError:
            raise
        except (OSError, RuntimeError, ValueError, EOFError, TypeError) as exc:
            raise HFCheckpointError(f"could not load latest checkpoint: {exc}") from exc
        if not isinstance(payload, dict):
            raise HFCheckpointError("latest checkpoint root must be a mapping")
        return payload
