"""Process-scoped, file-backed sharing of proven frozen CuMem segments.

Use private (copy-on-write) mappings: readers share physical pages, while an
accidental tensor write cannot modify another rank's backup. Files belong to
the controller's private temporary runtime directory and are not checkpoints.
"""
from __future__ import annotations

import fcntl
from collections import OrderedDict
import hashlib
import os
import stat
import tempfile
from pathlib import Path

from .byte_compare import equal_cpu_bytes


def frozen_segment_ranges(snapshot, frozen, mutable):
    """Return segment-relative live ranges only when every live block is known.

    Inputs are storage intervals, not tensor views. Missing block metadata,
    pending frees, mutable aliases and unknown live allocations stay private.
    Padding and inactive bytes have no live tensor owner and may be zeroed.
    """
    result = {}
    for segment in snapshot:
        begin = segment["address"]
        end = begin + segment["total_size"]
        blocks = segment.get("blocks")
        if not blocks or any(a < end and b > begin for a, b in mutable):
            continue
        ranges = []
        valid = True
        for block in blocks:
            state = block.get("state")
            if state == "inactive":
                continue
            address = block.get("address")
            requested = block.get("requested_size")
            if (state != "active_allocated" or type(address) is not int
                    or type(requested) is not int or requested <= 0
                    or not begin <= address < address + requested <= end
                    or not any(a <= address and address + requested <= b for a, b in frozen)):
                valid = False
                break
            ranges.append((address - begin, address + requested - begin))
        ranges.sort()
        if valid and ranges and all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])):
            result[begin] = ranges
    return result


class FrozenBackupStore:
    """Content-addressed frozen bytes, shared across ranks of one service only."""

    def __init__(self, directory):
        self._recent = OrderedDict()
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("frozen backup directory must be owned, private and not a symlink")

    def share(self, backup, ranges):
        import torch

        if (backup.device.type != "cpu" or backup.dtype != torch.uint8
                or backup.ndim != 1 or not backup.is_contiguous()):
            raise ValueError("frozen backup must be a contiguous CPU byte vector")
        size = backup.numel()
        previous = 0
        for start, stop in ranges:
            if not previous <= start < stop <= size:
                raise ValueError("invalid frozen byte ranges")
            previous = stop
        if not ranges:
            raise ValueError("frozen backup must contain live bytes")
        # Canonicalize only proven allocator padding/inactive bytes. This makes
        # equivalent allocations share even when their uninitialized gaps differ.
        previous = 0
        for start, stop in ranges:
            backup[previous:start].zero_()
            previous = stop
        backup[previous:].zero_()
        view = memoryview(backup.numpy())
        key = (size, tuple(ranges))
        created = False
        lock_fd = os.open(self.directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            # Repeated frozen segments commonly have identical bytes. Compare
            # the complete current backup to a freshly mapped, checked file;
            # a hit avoids a redundant serial BLAKE2 pass, never byte validation.
            previous_path = self._recent.get(key)
            if previous_path is not None and previous_path.exists():
                mapped = self._map_checked(previous_path, size)
                if equal_cpu_bytes(mapped, backup):
                    self._recent.move_to_end(key)
                    return mapped, False
            # This digest names only ephemeral files. Changed data follows the
            # full content-addressed path and is independently checked below.
            digest = hashlib.blake2b(view, digest_size=32).hexdigest()
            path = self.directory / (digest + ".bin")
            if not path.exists():
                fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=self.directory)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(view)
                    os.chmod(temporary, 0o400)
                    os.replace(temporary, path)
                    created = True
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            mapped = self._map_checked(path, size)
            # Actual byte validation protects restoration even if an ephemeral
            # file was truncated/replaced/corrupted. No metadata-only trust.
            if not equal_cpu_bytes(mapped, backup):
                raise RuntimeError("shared frozen backup content differs")
            self._recent[key] = path
            self._recent.move_to_end(key)
            if len(self._recent) > 64:
                self._recent.popitem(last=False)
            return mapped, created
        finally:
            os.close(lock_fd)

    def share_stream(self, size, ranges, read_chunk, chunk_bytes):
        """Copy/compare a frozen segment using caller-owned bounded CPU scratch.

        read_chunk(offset, count) returns the current CPU bytes of that device
        range. All live bytes are copied on every park, including cache hits.
        """
        import torch

        previous = 0
        for start, stop in ranges:
            if not previous <= start < stop <= size:
                raise ValueError("invalid frozen byte ranges")
            previous = stop
        if not ranges or chunk_bytes <= 0:
            raise ValueError("frozen stream requires live ranges and positive scratch")

        def chunks():
            for offset in range(0, size, chunk_bytes):
                count = min(chunk_bytes, size-offset)
                value = read_chunk(offset, count)
                if (value.device.type != 'cpu' or value.dtype != torch.uint8
                        or value.ndim != 1 or value.numel() != count or not value.is_contiguous()):
                    raise ValueError('frozen stream returned invalid scratch')
                cursor = offset
                for start, stop in ranges:
                    left, right = max(start, offset), min(stop, offset+count)
                    if left < right:
                        value[cursor-offset:left-offset].zero_()
                        cursor = right
                value[cursor-offset:].zero_()
                yield offset, value

        key = (size, tuple(ranges))
        lock_fd = os.open(self.directory / '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        temporary = None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            previous_path = self._recent.get(key)
            if previous_path is not None and previous_path.exists():
                mapped = self._map_checked(previous_path, size)
                if all(equal_cpu_bytes(mapped[offset:offset+value.numel()], value)
                       for offset, value in chunks()):
                    self._recent.move_to_end(key)
                    return mapped, False
            fd, temporary = tempfile.mkstemp(prefix='.publish-', dir=self.directory)
            digest = hashlib.blake2b(digest_size=32)
            with os.fdopen(fd, 'wb') as stream:
                for offset, value in chunks():
                    view = memoryview(value.numpy())
                    digest.update(view)
                    stream.write(view)
            path = self.directory / (digest.hexdigest()+'.bin')
            created = not path.exists()
            if created:
                os.chmod(temporary, 0o400)
                os.replace(temporary, path)
                temporary = None
            mapped = self._map_checked(path, size)
            # Independently check the final mapping against the current device
            # content, also catching corrupt pre-existing digest-named files.
            if not all(equal_cpu_bytes(mapped[offset:offset+value.numel()], value)
                       for offset, value in chunks()):
                raise RuntimeError('shared frozen backup content differs')
            self._recent[key] = path
            self._recent.move_to_end(key)
            if len(self._recent) > 64:
                self._recent.popitem(last=False)
            return mapped, created
        finally:
            if temporary is not None:
                os.unlink(temporary)
            os.close(lock_fd)

    @staticmethod
    def _map_checked(path, size):
        import torch

        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_size != size or stat.S_IMODE(info.st_mode) != 0o400):
            raise RuntimeError("invalid shared frozen backup file")
        return torch.from_file(str(path), shared=False, size=size, dtype=torch.uint8)
