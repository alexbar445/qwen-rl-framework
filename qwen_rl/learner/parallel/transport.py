"""Authenticated, length-prefixed pickle transport over a private Unix socket.

The socket is created inside a mode-0700 directory and its path and random
token are passed only to the sibling torchrun worker. This transport is for
trusted local processes, never for a network listener.
"""
from __future__ import annotations

import pickle
import socket
import struct
import sys
from typing import Any


_HEADER = struct.Struct("!Q")


def send_message(connection: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=5)
    connection.sendall(_HEADER.pack(len(payload)))
    connection.sendall(payload)


def _read_exact(connection: socket.socket, size: int) -> bytearray:
    if size < 0 or size > sys.maxsize:
        raise ValueError("parallel learner message length does not fit this platform")
    payload = bytearray(size)
    view = memoryview(payload)
    offset = 0
    while offset < size:
        received = connection.recv_into(view[offset:], size - offset)
        if received == 0:
            raise EOFError("parallel learner control socket closed mid-message")
        offset += received
    return payload


def receive_message(connection: socket.socket) -> Any:
    size = _HEADER.unpack(_read_exact(connection, _HEADER.size))[0]
    payload = _read_exact(connection, size)
    return pickle.loads(memoryview(payload))
