"""Tie an explicitly owned read-only observer to its parent's stdin pipe."""
from __future__ import annotations
import os
import signal
import sys
import threading


def terminate_on_stdin_eof() -> None:
    """Exit this observer when its owner disconnects, including during config loading.

    Used only by `watch --exit-on-stdin-eof`. Docker forwards the owner's pipe via
    `exec -i`; closing that pipe terminates the container observer even though killing
    the host Docker client alone does not terminate a docker-exec process.
    """
    descriptor = os.dup(sys.stdin.fileno())

    def read_owner() -> None:
        try:
            while os.read(descriptor, 4096):
                pass
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            os.close(descriptor)

    threading.Thread(target=read_owner, name='watch-stdin-owner', daemon=True).start()
