"""Remove obsolete delivery copies within one role's adapter cache."""
from pathlib import Path
import shutil


def prune_adapter_cache(root: Path, current_version: int, *, width: int,
                        keep_versions: frozenset[int] = frozenset()) -> list[int]:
    """Delete only older canonical vN directories, never symlinks or other names."""
    if isinstance(current_version, bool) or not isinstance(current_version, int) or current_version < 0:
        raise ValueError("current_version must be a non-negative integer")
    root = Path(root)
    if root.is_symlink():
        raise ValueError("adapter cache root cannot be a symlink")
    removed = []
    if not root.exists():
        return removed
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_dir() or not path.name.startswith("v"):
            continue
        suffix = path.name[1:]
        if not suffix.isascii() or not suffix.isdecimal():
            continue
        version = int(suffix)
        if path.name != f"v{version:0{width}d}" or version >= current_version or version in keep_versions:
            continue
        shutil.rmtree(path)
        removed.append(version)
    return removed
