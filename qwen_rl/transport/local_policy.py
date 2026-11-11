"""Managed same-instance Native policy files, with private Worker lifetimes."""
from __future__ import annotations

from contextlib import ExitStack
import errno
import json
import os
from pathlib import Path
import shutil
import stat
from uuid import uuid4

from qwen_rl.contracts.worker import NATIVE_ADAPTER_FILES, PolicyManifest


def shared_policy_root(value: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError("shared_policy_root must be an absolute directory")
    root = Path(value)
    if ".." in root.parts:
        raise ValueError("shared_policy_root cannot contain '..'")
    _directories(root, missing=True)
    return root


def _directories(path: Path, *, missing: bool = False) -> None:
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            if missing:
                continue
            raise
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("shared policy paths must contain only real directories")


def link_native_policy(root: Path, cache: Path, manifest: PolicyManifest) -> tuple[Path, int]:
    """Reuse immutable files; cross-filesystem copies stay local and are reported.

    The source root is deployment configuration, never a manifest-provided path.
    Native format/header/site validation still happens in the inference backend.
    """
    names = {item.name for item in manifest.files}
    if (names != NATIVE_ADAPTER_FILES or len(manifest.files) != len(names)
            or manifest.quantization.get("mode") != "ternary"):
        raise ValueError("shared policy transport requires a complete Native T2 manifest")
    source = root / f"v{manifest.version:08d}"
    _directories(source)
    if source.stat().st_uid != os.geteuid():
        raise ValueError("shared policy must belong to the Worker uid")
    if {p.name for p in source.iterdir()} != names:
        raise ValueError("shared policy file inventory differs from manifest")
    _directories(cache)
    if cache.stat().st_uid != os.geteuid():
        raise ValueError("private policy cache must belong to the Worker uid")
    final = cache / f"v{manifest.version}"
    stage = cache / f".v{manifest.version}.{uuid4().hex}.tmp"
    copied_bytes = 0
    with ExitStack() as stack:
        opened = {}
        for item in manifest.files:
            fd = os.open(source / item.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            stack.callback(os.close, fd)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_mode & 0o222 or item.size_bytes is None
                    or info.st_size != item.size_bytes):
                raise ValueError("shared policy file must be owned, read-only and match its declared size")
            opened[item.name] = (fd, info)
        descriptor_fd, descriptor_info = opened["policy.lora.ninfer.json"]
        if descriptor_info.st_size > 16 * 1024 * 1024:
            raise ValueError("shared policy descriptor exceeds the metadata limit")
        descriptor = json.loads(os.pread(descriptor_fd, descriptor_info.st_size, 0))
        expected = {"version": manifest.version,
                    "source_adapter_sha256": manifest.adapter_hash,
                    "artifact_id": manifest.artifact_id}
        if not isinstance(descriptor, dict) or any(descriptor.get(k) != v for k, v in expected.items()):
            raise ValueError("shared policy descriptor differs from manifest identity")
        if final.exists() or final.is_symlink():
            _directories(final)
            if {p.name for p in final.iterdir()} != names:
                raise ValueError("private policy cache has an unexpected inventory")
            if all((final / name).lstat().st_ino == info.st_ino
                   and (final / name).lstat().st_dev == info.st_dev for name, (_, info) in opened.items()):
                return final, 0
            # No loaded adapter is replaced: WorkerRuntime returns early for an
            # already-installed version. Open Native handles retain their inode.
            for path in final.iterdir():
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise ValueError("private policy cache contains an unsafe entry")
        stage.mkdir(mode=0o700)
        try:
            for name, (fd, info) in opened.items():
                target = stage / name
                try:
                    os.link(source / name, target, follow_symlinks=False)
                    linked = target.lstat()
                    if (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
                        raise ValueError("shared policy changed during linking")
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    with os.fdopen(os.dup(fd), "rb") as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, 1024 * 1024)
                    os.chmod(target, 0o444)
                    copied_bytes += info.st_size
                after = os.fstat(fd)
                if (after.st_size != info.st_size or after.st_mtime_ns != info.st_mtime_ns
                        or after.st_mode & 0o222 or target.stat().st_size != info.st_size):
                    raise ValueError("shared policy changed during materialization")
            if final.exists():
                shutil.rmtree(final)
            stage.replace(final)
            return final, copied_bytes
        finally:
            if stage.exists():
                shutil.rmtree(stage)
