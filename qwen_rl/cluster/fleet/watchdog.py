"""Minimal systemd sd_notify support for role-neutral Fleet services."""
from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable, Mapping


class SystemdWatchdog:
    """Send READY and rate-limited WATCHDOG datagrams to systemd."""

    def __init__(
        self,
        *,
        notify_socket: str | None,
        watchdog_usec: int | None,
        watchdog_pid: int | None,
        current_pid: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._address = self._socket_address(notify_socket)
        self._watchdog_usec = watchdog_usec if watchdog_usec and watchdog_usec > 0 else None
        self._eligible = watchdog_pid is None or watchdog_pid == current_pid
        self._main_pid = current_pid if watchdog_pid is None else watchdog_pid
        self._clock = clock
        self._last_watchdog: float | None = None
        self._ready_sent = False
        self._lock = threading.Lock()

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        current_pid: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> SystemdWatchdog:
        """Read systemd's standard notification variables without exposing them."""
        env = os.environ if environ is None else environ
        try:
            watchdog_usec = int(env["WATCHDOG_USEC"]) if env.get("WATCHDOG_USEC") else None
        except (TypeError, ValueError):
            watchdog_usec = None
        try:
            watchdog_pid = int(env["WATCHDOG_PID"]) if env.get("WATCHDOG_PID") else None
        except (TypeError, ValueError):
            watchdog_pid = -1
        return cls(
            notify_socket=env.get("NOTIFY_SOCKET"),
            watchdog_usec=watchdog_usec,
            watchdog_pid=watchdog_pid,
            current_pid=os.getpid() if current_pid is None else current_pid,
            clock=clock,
        )

    @staticmethod
    def _socket_address(value: str | None) -> str | bytes | None:
        if not value or "\x00" in value:
            return None
        if value.startswith("@"):
            try:
                return b"\x00" + value[1:].encode("utf-8")
            except UnicodeError:
                return None
        return value

    def _send(self, message: bytes) -> bool:
        if self._address is None or not self._eligible or os.getpid() != self._main_pid:
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                client.setblocking(False)
                client.connect(self._address)
                return client.send(message) == len(message)
        except (OSError, TypeError, ValueError):
            # In particular, never include NOTIFY_SOCKET or the datagram in logs.
            return False

    def notify_ready(self) -> bool:
        """Send READY=1 once; return False when disabled or delivery failed."""
        with self._lock:
            if self._ready_sent or not self._eligible:
                return False
            if not self._send(b"READY=1"):
                return False
            self._ready_sent = True
            return True

    def notify_watchdog(self) -> bool:
        """Send WATCHDOG=1 at most once per half-timeout interval."""
        if self._watchdog_usec is None or not self._eligible or self._address is None:
            return False
        interval = self._watchdog_usec / 2_000_000
        now = self._clock()
        with self._lock:
            if self._last_watchdog is not None and now - self._last_watchdog < interval:
                return False
            if not self._send(b"WATCHDOG=1"):
                return False
            self._last_watchdog = now
            return True


_default_lock = threading.Lock()
_default_signature: tuple[str | None, str | None, str | None, int] | None = None
_default_notifier: SystemdWatchdog | None = None


def _default() -> SystemdWatchdog:
    global _default_signature, _default_notifier
    with _default_lock:
        signature = (
            os.environ.get("NOTIFY_SOCKET"),
            os.environ.get("WATCHDOG_USEC"),
            os.environ.get("WATCHDOG_PID"),
            os.getpid(),
        )
        if signature != _default_signature:
            _default_notifier = SystemdWatchdog.from_environment()
            _default_signature = signature
        assert _default_notifier is not None
        return _default_notifier


def notify_ready() -> bool:
    """Send a secret-free READY=1 notification when systemd supplied a socket."""
    return _default().notify_ready()


def notify_watchdog() -> bool:
    """Send a rate-limited watchdog ping when systemd configured one."""
    return _default().notify_watchdog()
