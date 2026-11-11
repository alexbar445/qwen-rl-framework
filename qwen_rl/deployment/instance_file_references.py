"""Bulk process-reference audit for source files considered for cleanup."""
from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Iterable


def audit_file_references(paths: list[Path]) -> dict[str, bool | None]:
    """Public bulk audit used by source cleanup; it reads the host `/proc` once."""
    return _audit_file_references(paths, proc_root=Path("/proc"))


def _audit_file_references(
    paths: Iterable[str | Path], *, proc_root: str | Path,
) -> dict[str, bool | None]:
    """Audit candidate inodes against every visible process's FDs and maps.

    ``True`` means a reference was found, ``False`` means the full scan was
    readable and found none, and ``None`` means the audit was incomplete. The
    scan uses device/inode identity so bind-mounted path aliases still match.
    One `/proc` walk serves the whole candidate set.
    """
    candidates: dict[str, tuple[int, int]] = {}
    paths_by_inode: dict[tuple[int, int], list[str]] = {}
    results: dict[str, bool | None] = {}
    for value in paths:
        path = Path(value).expanduser().absolute()
        key = str(path)
        if key in results:
            continue
        try:
            info = path.lstat()
        except OSError:
            results[key] = None
            continue
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            results[key] = None
            continue
        identity = (int(info.st_dev), int(info.st_ino))
        candidates[key] = identity
        paths_by_inode.setdefault(identity, []).append(key)
        results[key] = False
    if not candidates:
        return results

    by_maps_id = {
        (os.major(device), os.minor(device), inode): keys
        for (device, inode), keys in paths_by_inode.items()
    }
    incomplete = False
    try:
        processes = tuple(item for item in Path(proc_root).iterdir() if item.name.isdecimal())
    except OSError:
        return {key: (None if value is False else value) for key, value in results.items()}

    def mark(identity: tuple[int, int]) -> None:
        for key in paths_by_inode.get(identity, ()):
            results[key] = True

    def mark_maps(identity: tuple[int, int, int]) -> None:
        for key in by_maps_id.get(identity, ()):
            results[key] = True

    for process in processes:
        try:
            descriptors = tuple((process / "fd").iterdir())
        except FileNotFoundError:
            continue  # process exited during the scan
        except OSError:
            incomplete = True
            descriptors = ()
        for descriptor in descriptors:
            try:
                info = descriptor.stat()
            except FileNotFoundError:
                continue  # descriptor closed during the scan
            except OSError:
                incomplete = True
                continue
            mark((int(info.st_dev), int(info.st_ino)))

        maps_path = process / "maps"
        try:
            maps = maps_path.open("r", encoding="ascii", errors="replace")
        except FileNotFoundError:
            continue  # process exited during the scan
        except OSError:
            incomplete = True
            continue
        try:
            with maps:
                for line in maps:
                    fields = line.split(None, 5)
                    if len(fields) < 5:
                        incomplete = True
                        continue
                    try:
                        major_text, minor_text = fields[3].split(":", 1)
                        inode = int(fields[4], 10)
                        identity = (int(major_text, 16), int(minor_text, 16), inode)
                    except (ValueError, IndexError):
                        incomplete = True
                        continue
                    if inode > 0:
                        mark_maps(identity)
        except OSError:
            incomplete = True

        if all(results[key] is True for key in candidates):
            return results

    if incomplete:
        for key in candidates:
            if results[key] is False:
                results[key] = None
    return results
