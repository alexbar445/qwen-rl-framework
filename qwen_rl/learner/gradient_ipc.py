"""Same-host immutable FP32 gradient pages; sockets carry only metadata and FDs."""
from __future__ import annotations

import fcntl
import json
import logging
import math
import mmap
import os
import socket
import struct
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_MAX_MESSAGE = 1024 * 1024
# Linux UAPI constants: also work in Conda builds lacking fcntl seal names.
_ADD_SEALS, _GET_SEALS = 1033, 1034
_SEALS = 8 | 4 | 2


def namespace_identity():
    return {"boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            "network_namespace": os.stat('/proc/self/ns/net').st_ino, "uid": os.geteuid()}


def _address(value):
    if not isinstance(value, str) or not value.startswith('@qwen-gradient-'):
        raise ValueError('invalid local gradient socket address')
    return '\0' + value[1:]


def _read(sock, size):
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise ConnectionError('local gradient peer disconnected')
        result.extend(chunk)
    return bytes(result)


def _message(sock, value):
    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > _MAX_MESSAGE:
        raise ValueError('local gradient metadata exceeds limit')
    sock.sendall(struct.pack('!I', len(raw)) + raw)


def _receive_message(sock):
    size, = struct.unpack('!I', _read(sock, 4))
    if not 0 < size <= _MAX_MESSAGE:
        raise ValueError('invalid local gradient metadata size')
    return json.loads(_read(sock, size))


class SharedGradientPayload:
    """Own an immutable memfd, with zero-copy CPU tensor views on receipt.

    Tensor views retain the mmap buffer if a consumer outlives store retirement.
    Closing this owner drops its views/FD; it never invalidates a live consumer.
    """
    def __init__(self, fd, metadata, size, gradients=None, mapping=None):
        self.fd = fd
        self.metadata = metadata
        self.size = size
        self.gradients = {} if gradients is None else gradients
        self._mapping = mapping

    def __len__(self):
        return self.size

    @classmethod
    def pack(cls, contribution):
        import torch

        from qwen_rl.worker.pinned_base import create_attestation_fd

        present = sorted(name for name, value in contribution.gradients.items() if value is not None)
        metadata = json.loads(contribution.to_metadata_json())
        metadata['present_parameters'] = present
        size = sum(contribution.gradients[name].numel() * 4 for name in present)
        fd = create_attestation_fd()
        mapping = None
        try:
            # Empty/all-None updates still have an owned, sealable descriptor.
            os.ftruncate(fd, max(1, size))
            if size:
                mapping = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
                offset = 0
                for name in present:
                    source = contribution.gradients[name]
                    if source.device.type != 'cpu' or source.dtype != torch.float32:
                        raise ValueError('shared gradients require CPU FP32 tensors')
                    count = source.numel()
                    if count:
                        target = torch.frombuffer(mapping, dtype=torch.float32, count=count, offset=offset)
                        try:
                            target.copy_(source.reshape(-1))
                        finally:
                            del target
                    offset += count * 4
                mapping.close()
                mapping = None
            fcntl.fcntl(fd, _ADD_SEALS, _SEALS | 1)
            return cls(fd, metadata, size)
        except BaseException:
            if mapping is not None:
                mapping.close()
            os.close(fd)
            raise

    @classmethod
    def receive(cls, fd, metadata, layout, max_bytes):
        import torch
        present = metadata.get('present_parameters')
        shapes = {p.name: tuple(p.shape) for p in layout}
        if (not isinstance(present, list) or any(not isinstance(n, str) for n in present)
                or present != sorted(set(present)) or not set(present) <= shapes.keys()):
            raise ValueError('shared gradient parameter layout differs from the job')
        size = sum(math.prod(shapes[name]) * 4 for name in present)
        if size > max_bytes or os.fstat(fd).st_size != max(1, size):
            raise ValueError('shared gradient size differs from layout or exceeds limit')
        if fcntl.fcntl(fd, _GET_SEALS) & _SEALS != _SEALS:
            raise ValueError('shared gradient pages must be immutable')
        mapping = mmap.mmap(fd, max(1, size), access=mmap.ACCESS_COPY)
        gradients = {}
        offset = 0
        for name in present:
            count = math.prod(shapes[name])
            gradients[name] = (torch.frombuffer(mapping, dtype=torch.float32, count=count, offset=offset)
                               .reshape(shapes[name]) if count else torch.empty(shapes[name], dtype=torch.float32))
            offset += count * 4
        return cls(fd, metadata, size, gradients, mapping)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        # Borrowers can retain this payload between get() and copying its views.
        # Keep the immutable views alive until the last Python owner releases them.
        self._mapping = None

    def __del__(self):
        self.close()


def send_contribution(address, token, executor_id, job_id, lease_id, lease_epoch, payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(120)
        sock.connect(_address(address))
        socket.send_fds(sock, [b'G'], [payload.fd])
        _message(sock, {'token': token, 'executor_id': executor_id, 'job_id': job_id,
                        'lease_id': lease_id, 'lease_epoch': lease_epoch,
                        'contribution': payload.metadata})
        return _receive_message(sock)


class LocalGradientServer:
    def __init__(self, accept, *, workers=4):
        self.address = '@qwen-gradient-' + uuid.uuid4().hex
        self.namespace = namespace_identity()
        self._accept = accept
        self.after_accept = None
        self._closed = threading.Event()
        self._notification_lock = threading.Lock()
        self._notifications = set()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='gradient-ipc')
        self._slots = threading.Semaphore(workers)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(_address(self.address))
        self._socket.listen(workers)
        self._socket.settimeout(0.25)
        self._thread = threading.Thread(target=self._serve, name='gradient-ipc-listener', daemon=True)
        self._thread.start()

    def descriptor(self):
        return {'address': self.address, 'namespace': self.namespace}

    def _serve(self):
        while not self._closed.is_set():
            if not self._slots.acquire(timeout=0.25):
                continue
            try:
                peer, _ = self._socket.accept()
            except socket.timeout:
                self._slots.release()
                continue
            except OSError:
                self._slots.release()
                break
            self._pool.submit(self._handle, peer)

    def _handle(self, peer):
        descriptors = []
        try:
            with peer:
                peer.settimeout(10)
                try:
                    _, uid, _ = struct.unpack('3i', peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != os.geteuid():
                        raise PermissionError('local gradient peer must use the same UID')
                    marker, descriptors, flags, _ = socket.recv_fds(peer, 1, 1)
                    if marker != b'G' or flags & socket.MSG_CTRUNC or len(descriptors) != 1:
                        raise ValueError('local gradient request requires exactly one descriptor')
                    request = _receive_message(peer)
                    response = self._accept(request, descriptors[0])
                    try:
                        _message(peer, {'status_code': 200, **response})
                    finally:
                        # Acceptance must still notify the central owner if the
                        # sender disappears before its small completion request.
                        if self.after_accept is not None:
                            try:
                                self.after_accept(response)
                            except Exception:
                                logging.getLogger(__name__).exception("shared gradient apply notification failed")
                except Exception as exc:
                    from fastapi import HTTPException

                    from .gradient_server.http import _http_error
                    error = exc if isinstance(exc, HTTPException) else _http_error(exc)
                    try:
                        _message(peer, {'status_code': error.status_code, 'detail': error.detail})
                    except OSError:
                        pass
        finally:
            for fd in descriptors:
                os.close(fd)
            self._slots.release()

    def join_notification(self, future):
        # A timed-out peer cannot abandon an accepted optimizer operation. Keep
        # it owned until completion, and drain before Core/store shutdown.
        with self._notification_lock:
            self._notifications.add(future)
        def finished(done):
            with self._notification_lock:
                self._notifications.discard(done)
        future.add_done_callback(finished)
        return future.result(timeout=120)

    def close(self):
        self._closed.set()
        self._thread.join(timeout=1)
        self._socket.close()
        self._pool.shutdown(wait=True, cancel_futures=False)
        with self._notification_lock:
            notifications = tuple(self._notifications)
        for future in notifications:
            try:
                future.result()
            except Exception:
                logging.getLogger(__name__).exception("shared gradient notification failed during shutdown")
