"""Process-local attestation for an immutable, deployment-pinned T2 base.

File metadata detects accidental mutation/replacement, not a malicious host.
Only the deployment identity loader may establish an attestation; never deserialize one
from persistent cache. All adapter and output hashes remain independent.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import stat

SCHEMA = "qwen-rl-pinned-base/v1"


def create_attestation_fd() -> int:
    """Use Linux's memfd API directly, including Conda Python builds without os.memfd_create."""
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    create.restype = ctypes.c_int
    # Linux UAPI: MFD_CLOEXEC=1, MFD_ALLOW_SEALING=2.
    descriptor = create(b"qwen-rl-pinned-base", 3)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor

def file_identity(path: Path) -> list[int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("pinned base inputs must remain regular files")
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]

@dataclass(frozen=True)
class PinnedBase:
    base_path: Path
    report_path: Path
    base_stat: tuple[int, ...]
    report_stat: tuple[int, ...]
    sha256: str

    @classmethod
    def capture(cls, base_path, report_path, before_base, before_report, sha256):
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError("pinned base requires a deployment SHA-256 declaration")
        result = cls(base_path, report_path, tuple(before_base), tuple(before_report), sha256)
        result.check()
        return result

    def check(self):
        if (tuple(file_identity(self.base_path)) != self.base_stat
                or tuple(file_identity(self.report_path)) != self.report_stat):
            raise ValueError("frozen base or conversion report changed since startup pinning")

    @contextmanager
    def descriptor(self):
        self.check()
        fd = create_attestation_fd()
        try:
            data = json.dumps({"schema": SCHEMA, "base_path": str(self.base_path),
                "report_path": str(self.report_path), "base_stat": self.base_stat,
                "report_stat": self.report_stat, "sha256": self.sha256}).encode()
            with os.fdopen(os.dup(fd), "wb") as stream:
                stream.write(data)
            os.lseek(fd, 0, os.SEEK_SET)
            # Linux UAPI seal operations/mask, independent of Python build headers.
            fcntl.fcntl(fd, 1033, 15)  # F_ADD_SEALS, WRITE|GROW|SHRINK|SEAL
            yield fd
            self.check()
        finally:
            os.close(fd)
