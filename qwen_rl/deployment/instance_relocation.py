"""Explicit path relocation checks for durable Learner inputs."""
from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


_ADAPTER_IDENTITY_FIELDS = (
    "initial_adapter_provenance_sha256",
    "initial_adapter_config_sha256",
    "initial_adapter_tensor_digest",
)


def _absolute_path(value: Any, *, label: str) -> str:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} must be an absolute path string")
    text = str(value)
    if not text.strip() or "\x00" in text:
        raise ValueError(f"{label} must be an absolute path string")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path string")
    return os.path.normpath(str(path))


def normalize_path_mappings(
    value: Any, *, staging_roots: Mapping[str | Path, str | Path] | None = None,
) -> dict[str, str]:
    """Canonicalize existing targets and validate nonoverlapping old-to-new path roots.

    During atomic import, ``staging_roots`` resolves future final paths through the
    copied tree, then maps the canonical relative suffix back to its final root.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("recovery_path_mappings must be an absolute path mapping")
    if staging_roots is not None and not isinstance(staging_roots, Mapping):
        raise TypeError("staging_roots must map final roots to existing temporary roots")
    staged_targets: dict[str, str] = {}
    for final_root, staging_root in (staging_roots or {}).items():
        final_text = _absolute_path(final_root, label="final staging root")
        final = str(Path(final_text).resolve(strict=False))
        staging = _absolute_path(staging_root, label="temporary staging root")
        try:
            resolved_staging = Path(staging).resolve(strict=True)
        except OSError as exc:
            raise ValueError("temporary staging root must exist") from exc
        if Path(staging).is_symlink() or not resolved_staging.is_dir():
            raise ValueError("temporary staging root must be a real directory")
        if final in staged_targets:
            raise ValueError("final staging roots must be unique")
        staged_targets[final] = staging

    final_roots = sorted(staged_targets)
    for index, final in enumerate(final_roots):
        final_path = Path(final)
        for other in final_roots[index + 1:]:
            other_path = Path(other)
            if final_path in other_path.parents or other_path in final_path.parents:
                raise ValueError("final staging roots may not overlap")
    mappings: dict[str, str] = {}
    for source, target in value.items():
        old = _absolute_path(source, label="recovery path map source")
        new_text = str(Path(_absolute_path(target, label="recovery path map target")).resolve(strict=False))
        try:
            new = str(Path(new_text).resolve(strict=True))
        except OSError:
            staging_match = next((
                (Path(final), Path(staging))
                for final, staging in staged_targets.items()
                if Path(new_text) == Path(final) or Path(final) in Path(new_text).parents
            ), None)
            if staging_match is None:
                raise ValueError("recovery path map target must exist") from None
            final_root, staging_root = staging_match
            target_relative = Path(new_text).relative_to(final_root)
            staged_target = staging_root / target_relative
            try:
                resolved_staged = staged_target.resolve(strict=True)
                resolved_staging = staging_root.resolve(strict=True)
                resolved_relative = resolved_staged.relative_to(resolved_staging)
            except (OSError, ValueError) as exc:
                raise ValueError("staged recovery path target is missing or escapes its import root") from exc
            new = str(final_root.resolve(strict=False) / resolved_relative)
        if Path(old) == Path(Path(old).anchor):
            raise ValueError("recovery path map source cannot be a filesystem root")
        if old == new or old in mappings or new in mappings.values():
            raise ValueError("recovery path map roots must be unique and nonidentical")
        mappings[old] = new

    for roots, label in ((list(mappings), "source"), (list(mappings.values()), "target")):
        for index, root in enumerate(roots):
            path = Path(root)
            if any(path in Path(other).parents or Path(other) in path.parents
                   for other in roots[index + 1:]):
                raise ValueError(f"recovery path map {label} roots may not overlap")
    return dict(sorted(mappings.items()))


def _mapped_path(path: str, mappings: Mapping[str, str]) -> str:
    candidate = Path(path)
    for source, target in mappings.items():
        try:
            suffix = candidate.relative_to(source)
        except ValueError:
            continue
        return os.path.normpath(str(Path(target) / suffix))
    return path


def path_identity_matches(saved_path: Any, current_path: Any, mappings: Any = None) -> bool:
    """Match a stored path directly or through one explicit old-to-new mapping."""
    saved = _absolute_path(saved_path, label="saved input path")
    current = _absolute_path(current_path, label="current input path")
    normalized = normalize_path_mappings(mappings)
    return saved == current or _mapped_path(saved, normalized) == current


def compatibility_matches(
    expected: Mapping[str, Any], saved: Mapping[str, Any], *, path_mappings: Any = None,
) -> bool:
    """Allow only mapped adapter paths whose persisted identities still match."""
    if not isinstance(expected, Mapping) or not isinstance(saved, Mapping):
        return False
    current = deepcopy(dict(expected))
    previous = deepcopy(dict(saved))
    current_path = current.get("initial_adapter_path")
    saved_path = previous.get("initial_adapter_path")
    if current_path != saved_path:
        try:
            mapped = path_identity_matches(saved_path, current_path, path_mappings)
        except (TypeError, ValueError):
            return False
        if (not mapped or any(current.get(key) is None or current.get(key) != previous.get(key)
                              for key in _ADAPTER_IDENTITY_FIELDS)):
            return False
        previous["initial_adapter_path"] = current_path
    return current == previous
