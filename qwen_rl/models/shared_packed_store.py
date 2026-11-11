"""One immutable file-backed packed master shared by local learner ranks.

Only page-backed views are returned: no rank-local master allocation is made.
ACCESS_COPY leaves clean pages shared and prevents a stray tensor write from
modifying the backing file; the offloader must still check tensor versions.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import stat
from typing import Any, BinaryIO
from uuid import uuid4

import torch

_SCHEMA = "qwen-rl-shared-packed-master/v1"


class SharedPackedStoreError(RuntimeError):
    """A shared master is incomplete, changed, or incompatible."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _signature(info: os.stat_result) -> dict[str, int]:
    return {name: int(getattr(info, "st_" + name))
            for name in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _private_directory(path: Path) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise SharedPackedStoreError("shared master root must be an absolute non-symlink directory")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise SharedPackedStoreError("shared master directory must be private and owned by the rank user")


@contextmanager
def _locked(path: Path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise SharedPackedStoreError("shared master lock is not owned by the rank user")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


class _CountingWriter:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.count = 0

    def write(self, data: Any) -> int:
        view = memoryview(data).cast("B")
        offset = 0
        while offset < len(view):
            written = self.stream.write(view[offset:])
            if written is None or written <= 0:
                raise SharedPackedStoreError("shared master write made no progress")
            self.count += written
            offset += written
        return offset

    def tell(self) -> int:
        return self.count


class SharedPackedStore:
    """Own a pinned descriptor and zero-copy views of one canonical master file."""

    @classmethod
    def open_or_create(
        cls,
        root: str | Path,
        identity: Mapping[str, Any],
        group_sizes: Sequence[int],
        write_group: Callable[[int, BinaryIO], None] | None = None,
    ) -> "SharedPackedStore":
        root = Path(root)
        sizes = tuple(group_sizes)
        if not sizes or any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError("shared packed group sizes must be positive integers")
        if not isinstance(identity, Mapping) or not identity:
            raise ValueError("shared packed master requires a nonempty identity")
        offsets = []
        position = 0
        for size in sizes:
            offsets.append(position)
            position += size
        descriptor = {"schema": _SCHEMA, "identity": dict(identity),
                      "group_sizes": list(sizes), "group_offsets": offsets,
                      "payload_bytes": position}
        # Round-trip rejects non-JSON identity and canonicalizes tuples.
        descriptor = json.loads(_canonical(descriptor))
        key = hashlib.sha256(_canonical(descriptor)).hexdigest()
        _private_directory(root)
        entry = root / key
        created = False
        with _locked(root / (key + ".lock")):
            if entry.is_symlink():
                raise SharedPackedStoreError("shared master entry cannot be a symlink")
            if not entry.exists():
                if write_group is None:
                    raise SharedPackedStoreError("shared master has not been published")
                temporary = root / ("." + key + "." + uuid4().hex + ".tmp")
                temporary.mkdir(mode=0o700)
                try:
                    data_path = temporary / "packed.bin"
                    with data_path.open("xb") as stream:
                        writer = _CountingWriter(stream)
                        for index, expected_size in enumerate(sizes):
                            before = writer.count
                            write_group(index, writer)
                            if writer.count - before != expected_size:
                                raise SharedPackedStoreError("shared group writer produced a different byte count")
                        stream.flush()
                        os.fchmod(stream.fileno(), 0o444)
                        os.fsync(stream.fileno())
                        file_signature = _signature(os.fstat(stream.fileno()))
                    manifest = {"descriptor": descriptor, "store_key": key,
                                "file_signature": file_signature}
                    manifest_path = temporary / "manifest.json"
                    with manifest_path.open("xb") as stream:
                        stream.write(_canonical(manifest) + b"\n")
                        stream.flush()
                        os.fchmod(stream.fileno(), 0o444)
                        os.fsync(stream.fileno())
                    _sync_directory(temporary)
                    os.rename(temporary, entry)
                    _sync_directory(root)
                    created = True
                except BaseException:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
            owner = cls(entry, descriptor, key)
            owner.created_here = created
            return owner

    def __init__(self, entry: Path, descriptor: dict[str, Any], key: str):
        _private_directory(entry)
        manifest_path = entry / "manifest.json"
        fd = os.open(manifest_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            manifest = json.load(stream)
        if manifest.get("descriptor") != descriptor or manifest.get("store_key") != key:
            raise SharedPackedStoreError("shared packed identity or group layout differs")
        self.path = entry / "packed.bin"
        self._fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        self._mapping = None
        try:
            info = os.fstat(self._fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o444
                    or _signature(info) != manifest.get("file_signature")
                    or info.st_size != descriptor["payload_bytes"]):
                raise SharedPackedStoreError("shared packed payload identity changed")
            self._signature = _signature(info)
            self._key = key
            self.group_sizes = tuple(descriptor["group_sizes"])
            self.group_count = len(self.group_sizes)
            self._offsets = tuple(descriptor["group_offsets"])
            self._mapping = mmap.mmap(self._fd, 0, access=mmap.ACCESS_COPY)
            self._tensors = tuple(torch.frombuffer(
                self._mapping, dtype=torch.uint8, count=size, offset=offset,
            ) for size, offset in zip(self.group_sizes, self._offsets, strict=True))
            self._versions = tuple(tensor._version for tensor in self._tensors)
            self.check_unchanged()
        except BaseException:
            self.close()
            raise

    def check_unchanged(self) -> None:
        if self._fd is None or self._mapping is None:
            raise SharedPackedStoreError("shared packed store is closed")
        if any(tensor._version != version for tensor, version in zip(
            self._tensors, self._versions, strict=True,
        )):
            raise SharedPackedStoreError("shared packed CPU view was modified")
        try:
            info = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise SharedPackedStoreError("shared packed payload is unavailable") from exc
        if not stat.S_ISREG(info.st_mode) or _signature(info) != self._signature or _signature(os.fstat(self._fd)) != self._signature:
            raise SharedPackedStoreError("shared packed payload was modified or replaced")

    def group_tensor(self, index: int) -> torch.Tensor:
        self.check_unchanged()
        if type(index) is not int or not 0 <= index < self.group_count:
            raise IndexError("shared packed group index is out of range")
        return self._tensors[index]

    def report(self) -> dict[str, Any]:
        self.check_unchanged()
        return {"schema": _SCHEMA, "store_key": self._key, "path": str(self.path),
                "group_count": self.group_count, "group_sizes": list(self.group_sizes),
                "group_offsets": list(self._offsets), "cpu_master_bytes": sum(self.group_sizes),
                "file_signature": dict(self._signature), "content_verification": "disabled",
                "backing": "shared_file_pages_with_private_copy_on_write_mapping",
                "private_cpu_master_allocations": 0, "file_backed": True,
                "created_here": getattr(self, "created_here", False)}

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        # torch.frombuffer retains the mmap object while any tensor view lives.
        # Never explicitly unmap behind those views; their final release owns it.
        self._tensors = ()
        self._mapping = None
