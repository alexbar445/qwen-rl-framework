"""Process locks for immutable, node-local executor policy caches."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import fcntl
import os
from pathlib import Path
import stat


def open_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise ValueError('policy cache directory cannot be a symlink')
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        os.close(fd)
        raise ValueError('policy cache lock must be a regular file owned by the executor')
    return fd


@asynccontextmanager
async def policy_lock(path: Path, *, exclusive: bool = False):
    fd = open_lock(path)
    try:
        while True:
            try:
                fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.05)
        yield
    finally:
        os.close(fd)
