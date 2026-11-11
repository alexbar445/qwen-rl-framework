"""Immutable served-policy materialization; bulk filesystem work stays off-loop."""
from __future__ import annotations
import errno
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from ..contracts.worker import NATIVE_ADAPTER_FILES
from .policy_maintenance import remove_conflicting_publication_directory


def published_policy_for_restore(policies, live_version, recovery=None):
    """Select registered bytes while a parked executor restores its Worker."""
    if recovery is not None and recovery.get("enabled") is True:
        if recovery.get("dirty") is not False or recovery.get("can_publish") is not True:
            raise RuntimeError("central recovery does not permit policy restoration")
    version = max(policies, default=-1)
    if version < 0 or not 0 <= live_version - version <= 1:
        raise RuntimeError("published adapter is unavailable or stale")
    return policies[version]


def materialize_served_policy(run_dir, version, source_files, published, model_backend,
                              existing_policy=None):
    started = time.perf_counter()
    cfg = {'run_dir': run_dir, 'model_backend': model_backend}
    native = model_backend == 'ternary'
    metrics = {'linked_bytes': 0, 'copied_bytes': 0}
    immutable_dir = Path(cfg["run_dir"]) / "policies" / f"v{version:08d}"
    if immutable_dir.exists() or immutable_dir.is_symlink():
        if immutable_dir.is_symlink() or not immutable_dir.is_dir():
            raise RuntimeError(f"immutable policy version {version} has an unsafe directory")
        entries = list(immutable_dir.iterdir())
        expected_names = set(source_files)
        actual_names = {entry.name for entry in entries}
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise RuntimeError(f"immutable policy version {version} contains unsafe entries")
        conflict = actual_names != expected_names or any(
            (immutable_dir / name).stat().st_size != source["size"]
            for name, source in source_files.items()
            if name in actual_names
        )
        if not conflict and cfg.get("model_backend", "default") == "ternary":
            descriptor_path = immutable_dir / "policy.lora.ninfer.json"
            try:
                descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                conflict = True
            else:
                native = published.get("native")
                expected_artifact_id = native.get("artifact_id") if isinstance(native, dict) else None
                conflict = (
                    not isinstance(descriptor, dict)
                    or descriptor.get("version") != version
                    or descriptor.get("source_adapter_sha256") != published.get("adapter_hash")
                    or descriptor.get("artifact_id") != expected_artifact_id
                )
        elif not conflict and "adapter_config.json" in source_files:
            config_path = immutable_dir / "adapter_config.json"
            conflict = config_path.read_bytes() != source_files["adapter_config.json"]["path"].read_bytes()
        if conflict:
            if existing_policy is not None:
                raise RuntimeError(f"published policy version {version} changed after publication")
            remove_conflicting_publication_directory(
                immutable_dir,
                version=version,
                kind="served adapter",
            )
    immutable_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, source in source_files.items():
        destination = immutable_dir / name
        source_size = source["size"]
        if destination.exists():
            if destination.is_symlink() or not destination.is_file() or destination.stat().st_size != source_size:
                raise RuntimeError(f"immutable policy version {version} has an unsafe or different-size {name}")
        else:
            temporary = immutable_dir / f".{name}.{uuid.uuid4().hex}.tmp"
            try:
                if native:
                    try:
                        os.link(source['path'], temporary, follow_symlinks=False)
                        metrics['linked_bytes'] += source_size
                    except OSError as exc:
                        if exc.errno != errno.EXDEV:
                            raise
                        shutil.copyfile(source['path'], temporary)
                        metrics['copied_bytes'] += source_size
                else:
                    shutil.copyfile(source['path'], temporary)
                    metrics['copied_bytes'] += source_size
                if temporary.stat().st_size != source_size:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"immutable policy copy size mismatch for {name}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        files[name] = {"path": destination, "sha256": None, "size": source_size}
        if cfg.get("model_backend") == "ternary" and name in NATIVE_ADAPTER_FILES:
            # Managed same-instance Workers link these immutable bytes into
            # private version directories. Unlinking either owner is safe.
            destination.chmod(0o444)
    metrics['seconds'] = time.perf_counter() - started
    return files, metrics
