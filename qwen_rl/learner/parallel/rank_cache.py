"""Rank-private persistent compiler caches for phase-shared T2 TP workers."""
from __future__ import annotations

import os
import stat
from pathlib import Path


def _index(value: int | str, name: str, *, minimum: int) -> int:
    if type(value) is int:
        result = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        result = int(value)
    else:
        raise ValueError(f"{name} must be a decimal integer")
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _private_directory(path: Path, uid: int) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"compiler cache directory must not be a symlink or non-directory: {path}")
    if info.st_uid != uid:
        raise ValueError(f"compiler cache directory must be owned by uid {uid}: {path}")
    if stat.S_IMODE(info.st_mode) != 0o700:
        os.chmod(path, 0o700, follow_symlinks=False)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError(f"compiler cache directory must be uid-owned mode 0700: {path}")


def configure_t2_tp_compiler_cache(
    recovery_dir: str | os.PathLike[str] | None,
    global_rank: int | str,
    world_size: int | str,
    *,
    expected_world_size: int,
) -> dict[str, str]:
    """Point this T2 TP rank at durable, private Triton and Inductor caches."""
    rank = _index(global_rank, "global rank", minimum=0)
    world = _index(world_size, "world size", minimum=1)
    expected_world = _index(expected_world_size, "expected world size", minimum=1)
    if world != expected_world:
        raise ValueError(f"torchrun world size {world} does not match topology {expected_world}")
    if rank >= world:
        raise ValueError(f"global rank {rank} is outside world size {world}")
    if not isinstance(recovery_dir, (str, os.PathLike)) or not os.fspath(recovery_dir):
        raise ValueError("T2 TP phase_shared requires an explicit recovery directory")

    root = Path(recovery_dir)
    if not root.is_absolute():
        raise ValueError("recovery directory must be absolute")
    try:
        root_info = root.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"recovery directory does not exist: {root}") from exc
    uid = os.geteuid()
    if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != uid:
        raise ValueError("recovery directory must be a non-symlink directory owned by the current uid")

    cache_root = root / "compiler-cache"
    rank_root = cache_root / f"rank-{rank}"
    triton = rank_root / "triton"
    inductor = rank_root / "inductor"
    for directory in (cache_root, rank_root, triton, inductor):
        _private_directory(directory, uid)

    paths = {"TRITON_CACHE_DIR": str(triton), "TORCHINDUCTOR_CACHE_DIR": str(inductor)}
    os.environ.update(paths)
    return paths
