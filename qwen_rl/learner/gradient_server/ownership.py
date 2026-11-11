"""Lifetime process ownership for a central optimizer's durable directory."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import stat


def _lock_path_for_checkpoint_dir(directory: str | Path) -> Path:
    root = Path(directory).expanduser().resolve()
    if not root.name:
        raise ValueError('checkpoint root must not be the filesystem root')
    return root.parent / f'.{root.name}.gradient-optimizer.lock'


class ProcessOwnerLock:
    """Hold an OS lock until explicit shutdown or process exit.

    The lock lives beside the checkpoint root so genesis state creation still
    requires an empty state directory. Resolving the root makes alternate paths
    to the same state contend for the same lock. No stale PID heuristic can
    release a live optimizer owner.
    """

    def __init__(self, path: Path):
        self._fd: int | None = None
        self.path = path
        self._owner_pid = os.getpid()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f'{self._owner_pid}\n'.encode('ascii'))
            info = os.fstat(descriptor)
            self._device_inode = (info.st_dev, info.st_ino)
        except BaseException:
            os.close(descriptor)
            raise RuntimeError('central optimizer state is already owned or cannot be locked') from None
        self._fd = descriptor

    @classmethod
    def for_checkpoint_dir(cls, directory: str | Path) -> 'ProcessOwnerLock':
        return cls(_lock_path_for_checkpoint_dir(directory))

    def assert_owned_for(self, checkpoint_dir: str | Path) -> None:
        """Verify this process still owns the lock beside the requested checkpoint root."""
        descriptor = self._fd
        if descriptor is None:
            raise RuntimeError('central optimizer owner lock is closed')
        if self._owner_pid != os.getpid():
            raise RuntimeError('central optimizer owner lock was inherited by another process')
        expected = _lock_path_for_checkpoint_dir(checkpoint_dir)
        if self.path != expected:
            raise RuntimeError('central optimizer owner lock belongs to a different checkpoint root')
        try:
            opened = os.fstat(descriptor)
            current = self.path.lstat()
        except OSError as exc:
            raise RuntimeError('central optimizer owner lock is no longer live') from exc
        identity = (opened.st_dev, opened.st_ino)
        if (not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(current.st_mode)
                or stat.S_ISLNK(current.st_mode) or identity != self._device_inode
                or (current.st_dev, current.st_ino) != self._device_inode):
            raise RuntimeError('central optimizer owner lock inode changed')
        try:
            owner_record = os.pread(descriptor, 64, 0)
        except OSError as exc:
            raise RuntimeError('central optimizer owner lock is no longer readable') from exc
        if owner_record != f'{self._owner_pid}\n'.encode('ascii'):
            raise RuntimeError('central optimizer owner lock PID does not match this process')
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('central optimizer owner lock is not held by this process') from exc
        try:
            current = self.path.lstat()
            opened = os.fstat(descriptor)
        except OSError as exc:
            raise RuntimeError('central optimizer owner lock changed during validation') from exc
        if ((opened.st_dev, opened.st_ino) != self._device_inode
                or (current.st_dev, current.st_ino) != self._device_inode
                or stat.S_ISLNK(current.st_mode)):
            raise RuntimeError('central optimizer owner lock inode changed during validation')

    def close(self) -> None:
        descriptor, self._fd = self._fd, None
        if descriptor is not None:
            os.close(descriptor)

    def __enter__(self) -> 'ProcessOwnerLock':
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except (OSError, AttributeError):
            pass
