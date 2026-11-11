"""Metadata-only reader for a self-contained mixed task dataset."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

SCHEMA = "qwen-rl-mixed-task-sources/v1"
CATALOG_NAME = "task-sources.json"
OFFICIAL_FILES = (
    "arc-agi_training_challenges.json",
    "arc-agi_training_solutions.json",
    "arc-agi_evaluation_challenges.json",
    "arc-agi_evaluation_solutions.json",
)


@dataclass(frozen=True)
class CatalogSource:
    pool: str
    path: Path
    relative_path: str
    weight: float
    variants: bool | None = None


@dataclass(frozen=True)
class SourceCatalog:
    root: Path
    sources: tuple[CatalogSource, ...]

    @property
    def schema(self) -> str:
        return SCHEMA

    def source_for(self, pool: str) -> CatalogSource | None:
        return next((source for source in self.sources if source.pool == pool), None)


def _fail(message: str) -> None:
    raise ValueError(f"invalid mixed task catalog: {message}")


def _contained_source(root: Path, relative: str) -> Path:
    candidate_rel = Path(relative)
    if candidate_rel.is_absolute() or not relative or any(part in {".", ".."} for part in candidate_rel.parts):
        _fail(f"source path must be a clean relative path: {relative!r}")
    candidate = root / candidate_rel
    current = root
    for part in candidate_rel.parts:
        current = current / part
        if current.is_symlink():
            _fail(f"source path contains a symlink: {relative!r}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        _fail(f"source path escapes or is missing: {relative!r}")
    if not resolved.is_dir():
        _fail(f"source path is not a directory: {relative!r}")
    return resolved


def _reject_tree_symlinks(source: Path) -> None:
    def fail_walk(error: OSError) -> None:
        raise error

    for current, directories, files in os.walk(source, followlinks=False, onerror=fail_walk):
        parent = Path(current)
        for name in (*directories, *files):
            path = parent / name
            if path.is_symlink():
                _fail(f"source tree contains a symlink: {path.relative_to(source)}")


def load_source_catalog(dataset_dir: str | Path) -> SourceCatalog | None:
    """Load and validate ``task-sources.json`` without reading task payloads.

    ``None`` means this remains a legacy single-source dataset root.  Once the catalog
    exists, malformed metadata, missing children, and missing official files fail closed.
    """
    root_input = Path(dataset_dir).expanduser()
    catalog_path = root_input / CATALOG_NAME
    if not catalog_path.exists() and not catalog_path.is_symlink():
        return None
    if catalog_path.is_symlink() or not catalog_path.is_file():
        _fail("task-sources.json must be a regular file")
    root = root_input.resolve(strict=True)
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot read task-sources.json ({type(exc).__name__})")
    if not isinstance(data, dict) or set(data) != {"schema", "sources"} or data.get("schema") != SCHEMA:
        _fail(f"expected schema {SCHEMA!r} and only schema/sources fields")
    entries = data.get("sources")
    if not isinstance(entries, list) or len(entries) != 2:
        _fail("sources must contain exactly synthetic_arc and arc2_train")

    expected = {
        "synthetic_arc": ("generated", {"pool", "path", "weight"}, None),
        "arc2_train": ("arc2", {"pool", "path", "weight", "variants"}, False),
    }
    sources: list[CatalogSource] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            _fail("each source entry must be an object")
        pool = entry.get("pool")
        if pool not in expected or pool in seen:
            _fail(f"duplicate or unsupported source pool: {pool!r}")
        relative, fields, variants = expected[pool]
        weight = entry.get("weight")
        if set(entry) != fields or entry.get("path") != relative:
            _fail(f"source {pool!r} must match the v1 path and fields")
        if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 < weight < 1:
            _fail("source weights must be finite numbers strictly between 0 and 1")
        if variants is not None and entry.get("variants") is not variants:
            _fail("arc2_train variants must be false")
        child = _contained_source(root, relative)
        _reject_tree_symlinks(child)
        sources.append(CatalogSource(pool, child, relative, weight, variants))
        seen.add(pool)
    if seen != set(expected):
        _fail("sources must contain exactly synthetic_arc and arc2_train")

    if not math.isclose(sum(source.weight for source in sources), 1.0, rel_tol=0, abs_tol=1e-12):
        _fail("source weights must sum to 1")

    arc2 = next(source.path for source in sources if source.pool == "arc2_train")
    missing = [name for name in OFFICIAL_FILES if not (arc2 / name).is_file() or (arc2 / name).is_symlink()]
    if missing:
        _fail(f"arc2 source is missing regular challenge/solution files: {', '.join(missing)}")
    return SourceCatalog(root=root, sources=tuple(sources))


def generated_source_path(dataset_dir: str | Path) -> Path:
    """Resolve a catalog's generated child, without opening its data bundles."""
    root = Path(dataset_dir).expanduser()
    catalog = load_source_catalog(root)
    if catalog is None:
        return root
    source = catalog.source_for("synthetic_arc")
    if source is None:  # guarded by catalog validation; keeps the helper total
        _fail("synthetic_arc source is absent")
    return source.path
