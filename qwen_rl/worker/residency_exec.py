"""Exec a shared-GPU native engine without synchronous THP compaction.

Linux PR_SET_THP_DISABLE is process-local and survives exec. No host sysfs
setting or parent process changes; the native PID and signal semantics survive.
"""
from __future__ import annotations

import ctypes
import os
import signal
import sys


def disable_transparent_hugepages() -> None:
    if sys.platform != "linux":
        raise RuntimeError("shared native residency requires Linux THP control")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    prctl.restype = ctypes.c_int
    if prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if prctl(42, 0, 0, 0, 0) != 1:  # PR_GET_THP_DISABLE
        raise RuntimeError("native process did not retain THP disablement")


def main(argv: list[str] | None = None) -> int:
    command = list(sys.argv[1:] if argv is None else argv)
    if not command or not os.path.isabs(command[0]):
        raise ValueError("residency exec requires an absolute native executable")
    parent = os.environ.pop("QWEN_NINFER_PARENT_PID", None)
    if parent is not None:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot bind Native lifetime to Worker parent")
        if os.getppid() != int(parent):
            raise RuntimeError("Worker parent exited before Native startup")
    disable_transparent_hugepages()
    os.execvpe(command[0], command, os.environ)
    return 1  # exec never returns successfully


if __name__ == "__main__":
    raise SystemExit(main())
