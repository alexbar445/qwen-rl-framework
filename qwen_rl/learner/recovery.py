"""Atomic, single-slot recovery snapshots for the Learner training state.

The numbered adapter archives remain a separate inference/history product. This
module owns one current full optimizer snapshot per training root and refuses to
guess when that snapshot is missing, structurally invalid, or incompatible.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time
import uuid
from typing import Any

from .trainer_checkpoint_adapter import (
    TrainerCheckpointAdapterError,
    load_payload as load_trainer_payload,
    save_payload as save_trainer_payload,
)


LATEST_SCHEMA = "qwen-rl-latest/v1"
SNAPSHOT_SCHEMA = "qwen-rl-training-state/v1"
_ID = re.compile(r"[0-9a-f]{32}\Z")
_REQUIRED_FILES = {
    "adapter_config.json",
    "adapter_state.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "state.json",
}
_OPTIONAL_FILES = {"scaler.pt"}
_LOG = logging.getLogger(__name__)


class RecoveryError(RuntimeError):
    """A full training state cannot be restored without guessing."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecoveryError(f"recovery metadata is not strict JSON: {exc}") from exc


def _jsonable(value: Any) -> Any:
    """Convert configuration objects to a small, strict JSON value tree."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise RecoveryError("recovery metadata cannot contain NaN or infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=lambda item: _canonical_bytes(item))
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    raise RecoveryError(f"unsupported recovery metadata value: {type(value).__name__}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RecoveryError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=unique_pairs)
    except RecoveryError:
        raise
    except Exception as exc:
        raise RecoveryError(f"cannot read recovery JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"recovery JSON must contain an object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_payload_file(path: Path) -> float:
    started = time.perf_counter()
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    return time.perf_counter() - started


def _fsync_payload_files(
    entries: list[Path],
    performance: dict[str, float],
) -> float:
    # Drain every future before returning or raising so failed staging can be cleaned safely.
    started = time.perf_counter()
    futures = {}
    first_error: BaseException | None = None
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="recovery-payload-fsync"
    ) as executor:
        try:
            for entry in entries:
                if entry.is_file():
                    futures[entry.name] = executor.submit(_fsync_payload_file, entry)
        except BaseException as exc:
            first_error = exc

        for name, future in futures.items():
            try:
                performance[name] = future.result()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

    if first_error is not None:
        raise first_error
    return time.perf_counter() - started


def _write_file_fsynced(
    path: Path,
    data: bytes,
    *,
    performance: dict[str, Any] | None = None,
    prefix: str | None = None,
) -> None:
    started = time.perf_counter()
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        if performance is not None and prefix is not None:
            performance[f"{prefix}_write_seconds"] = time.perf_counter() - started
        fsync_started = time.perf_counter()
        os.fsync(stream.fileno())
        if performance is not None and prefix is not None:
            performance[f"{prefix}_fsync_seconds"] = time.perf_counter() - fsync_started


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    performance: dict[str, Any] | None = None,
    prefix: str = "atomic_json",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise RecoveryError(f"refusing recovery write through symlink directory: {path.parent}")
    if path.is_symlink():
        raise RecoveryError(f"refusing to replace symlink recovery file: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        serialize_started = time.perf_counter()
        payload = _canonical_bytes(value) + b"\n"
        if performance is not None:
            performance[f"{prefix}_serialize_seconds"] = time.perf_counter() - serialize_started
        _write_file_fsynced(
            temporary, payload, performance=performance, prefix=f"{prefix}_temporary"
        )
        replace_started = time.perf_counter()
        os.replace(temporary, path)
        if performance is not None:
            performance[f"{prefix}_replace_seconds"] = time.perf_counter() - replace_started
        directory_fsync_started = time.perf_counter()
        _fsync_directory(path.parent)
        if performance is not None:
            performance[f"{prefix}_directory_fsync_seconds"] = (
                time.perf_counter() - directory_fsync_started
            )
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_if_unchanged(
    path: Path, value: Mapping[str, Any], *, expected_bytes: bytes,
) -> str | None:
    """Commit one pointer only if it still names the snapshot we inspected.

    The rename is the commit point. Errors before it leave the original pointer
    untouched; directory-fsync errors after it are returned as a durability
    warning so callers do not report a failed operation after the pointer moved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise RecoveryError(f"refusing recovery pointer update through symlink: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_file_fsynced(temporary, _canonical_bytes(value) + b"\n")
        if path.read_bytes() != expected_bytes:
            raise RecoveryError("latest recovery pointer changed during rollout contract update")
        os.replace(temporary, path)
        try:
            _fsync_directory(path.parent)
        except Exception as exc:
            return f"directory fsync after pointer commit failed: {type(exc).__name__}: {exc}"
        return None
    finally:
        if temporary.exists():
            temporary.unlink()


def _normal_uuid(value: str | None, *, field: str) -> str:
    if value is None:
        return uuid.uuid4().hex
    try:
        normalized = uuid.UUID(str(value)).hex
    except (ValueError, AttributeError, TypeError) as exc:
        raise RecoveryError(f"{field} must be a UUID") from exc
    if not _ID.fullmatch(normalized):
        raise RecoveryError(f"{field} must be a UUID")
    return normalized


def _absolute_unresolved(path: str | Path) -> Path:
    """Make an absolute path without following the final symlink."""
    return Path(os.path.abspath(Path(path).expanduser()))


def _is_real_dir(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _require_safe_directories(paths: list[Path]) -> None:
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise RecoveryError(f"recovery path component is not a real directory: {path}")


def _tree_sha256(
    path: Path,
    *,
    include_suffixes: set[str] | None = None,
    exclude_dirs: set[str] | None = None,
) -> str:
    """Hash file contents and relative paths, following safe in-tree symlinks."""
    try:
        root = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise RecoveryError(f"semantic input path is missing: {path}") from exc
    if root.is_file():
        return _sha256_file(root)
    if not root.is_dir():
        raise RecoveryError(f"semantic input is not a file or directory: {path}")

    digest = hashlib.sha256()

    def visit(directory: Path, prefix: str, active: set[Path]) -> None:
        real = directory.resolve(strict=True)
        if real in active:
            raise RecoveryError(f"semantic input contains a directory symlink cycle: {directory}")
        if not real.is_dir():
            raise RecoveryError(f"semantic input directory changed while hashing: {directory}")
        active.add(real)
        try:
            for child in sorted(real.iterdir(), key=lambda item: item.name):
                if child.is_dir() and child.name in (exclude_dirs or set()):
                    continue
                relative = f"{prefix}/{child.name}" if prefix else child.name
                target = child.resolve(strict=True) if child.is_symlink() else child
                if target.is_file() and include_suffixes is not None and target.suffix not in include_suffixes:
                    continue
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                if target.is_dir():
                    digest.update(b"D\0")
                    visit(target, relative, active)
                elif target.is_file():
                    digest.update(b"F\0")
                    digest.update(str(target.stat().st_size).encode("ascii"))
                    digest.update(b"\0")
                    digest.update(bytes.fromhex(_sha256_file(target)))
                else:
                    raise RecoveryError(f"unsupported semantic input entry: {child}")
        finally:
            active.remove(real)

    visit(root, "", set())
    return digest.hexdigest()


def _read_rollout_contract(path: str | Path) -> tuple[Path, dict[str, Any], str, str]:
    """Read a complete episode contract and return path, fields, file SHA, and fingerprint."""
    from ..budget import CONTRACT_FIELDS, SEPARATE_BUDGET_FIELDS, Contract

    contract_path = Path(path).expanduser()
    if contract_path.is_symlink() or not contract_path.is_file():
        raise RecoveryError(f"rollout contract path is missing or unsafe: {contract_path}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RecoveryError(f"duplicate key {key!r} in rollout contract {contract_path}")
            result[key] = value
        return result

    try:
        document = json.loads(contract_path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    except RecoveryError:
        raise
    except Exception as exc:
        raise RecoveryError(f"cannot read rollout contract {contract_path}: {exc}") from exc
    if (not isinstance(document, dict)
            or set(document) not in (set(CONTRACT_FIELDS),
                                    set(CONTRACT_FIELDS) | set(SEPARATE_BUDGET_FIELDS))):
        raise RecoveryError("rollout contract must have the complete canonical field set")
    token_budget = document.get("token_budget")
    if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 1:
        raise RecoveryError("rollout contract token_budget must be a positive integer")
    try:
        contract = Contract(**document)
    except (TypeError, ValueError) as exc:
        raise RecoveryError(f"rollout contract is invalid: {exc}") from exc
    return (
        contract_path.resolve(), document, _sha256_file(contract_path), contract.fingerprint()
    )


def _semantic_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    supplied = config.get("recovery_semantics")
    if supplied is not None:
        if not isinstance(supplied, Mapping) or not supplied:
            raise RecoveryError("recovery_semantics must be a non-empty object")

    semantics: dict[str, Any] = {}
    dataset_dir = config.get("dataset_dir")
    if dataset_dir:
        semantics["dataset_tree_sha256"] = _tree_sha256(Path(str(dataset_dir)))

    arc = config.get("arc", {})
    contract_path = arc.get("contract_path") if isinstance(arc, Mapping) else None
    if contract_path:
        semantics["arc_contract_sha256"] = _tree_sha256(Path(str(contract_path)))

    project_root = Path(__file__).resolve().parents[2]
    toolkit_dependency = project_root / "toolkit-dependency.json"
    if toolkit_dependency.is_file():
        semantics["toolkit_dependency_sha256"] = _tree_sha256(toolkit_dependency)
    toolkit_root = project_root / "vendor" / "paired-factory-v1" / "arc_toolkit"
    if toolkit_root.is_dir():
        semantics["arc_toolkit_tree_sha256"] = _tree_sha256(
            toolkit_root,
            include_suffixes={".py"},
            exclude_dirs={"__pycache__"},
        )

    bepa = config.get("bepa", {})
    if isinstance(bepa, Mapping) and bool(bepa.get("enabled")):
        teacher = bepa.get("teacher", {})
        teacher_model = teacher.get("model") if isinstance(teacher, Mapping) else None
        source = {
            "enabled": True,
            "teacher_model": teacher_model,
        }
        for key in ("source_mode", "plan_projection", "student_projection", "guided_replay_contract"):
            if key in bepa:
                source[key] = _jsonable(bepa[key])
        semantics["bepa_source"] = source

    async_t2_semantics = async_t2_recovery_semantics(config)
    if async_t2_semantics is not None:
        semantics.update(async_t2_semantics)

    if not semantics:
        if supplied is not None:
            return _jsonable(supplied)
        raise RecoveryError(
            "recovery requires semantic dataset/tool identity; provide recovery_semantics or full runtime config"
        )
    rollout = config.get("rollout")
    constrained = rollout.get("constrained_decoding", False) if isinstance(rollout,Mapping) else config.get("constrained_decoding", False)
    expected_distribution = "grammar_conditioned_policy/v1" if constrained else "raw_policy"
    semantics["policy_distribution"] = expected_distribution
    grammar_identity = {}
    if constrained:
        from ..contracts.grammar_compiler import native_grammar_compiler_sha256
        from ..prompt import load_toolkit_runtime
        load_toolkit_runtime()
        from arc_toolkit.constrained_actions import constraint_receipt
        grammar_identity = {"grammar_policy": {"compiler_sha256":native_grammar_compiler_sha256(),
                            "descriptor_sha256":constraint_receipt()["sha256"]}}
        semantics.update(grammar_identity)
    normalized = _jsonable(semantics)
    if supplied is not None:
        supplied_normalized = _jsonable(supplied)
        declared_distribution = supplied_normalized.get("policy_distribution")
        if declared_distribution is not None and declared_distribution != expected_distribution:
            raise RecoveryError("supplied recovery policy distribution differs from the configured mode")
        if "grammar_policy" in supplied_normalized and supplied_normalized["grammar_policy"] != grammar_identity.get("grammar_policy"):
            raise RecoveryError("supplied recovery grammar policy differs from the configured compiler/descriptor")
        if async_t2_semantics is not None:
            extension = _jsonable(async_t2_semantics)
            if supplied_normalized not in (extension, normalized):
                raise RecoveryError("supplied T2 recovery semantics differ from the verified async runtime identity")
            return normalized
        # RuntimeConfig preflight and the app both call this helper over the
        # complete source config. The Learner constructor receives only its
        # sub-config plus this precomputed canonical object.
        has_full_semantic_inputs = bool(config.get("dataset_dir")) or bool(
            isinstance(config.get("arc"), Mapping) and config["arc"].get("contract_path")
        )
        if has_full_semantic_inputs and supplied_normalized != normalized:
            raise RecoveryError("supplied recovery_semantics differs from the canonical runtime-derived identity")
        return {**supplied_normalized, "policy_distribution":expected_distribution, **grammar_identity}
    return normalized


def async_t2_recovery_semantics(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Derive the T2 additions for the canonical asynchronous ARC runtime identity.

    The standalone T2 compute CLI has a different caller-supplied rollout contract
    and no Workers/dataset/ARC service. Keep it on its own recovery semantics.
    """
    if config.get("model_backend") != "ternary":
        return None
    arc = config.get("arc")
    learners = config.get("learner")
    workers = config.get("workers")
    supplied = config.get("recovery_semantics")
    full_async_config = (
        isinstance(config.get("dataset_dir"), str)
        and isinstance(arc, Mapping) and isinstance(arc.get("contract_path"), str)
        and isinstance(learners, Mapping)
        and isinstance(config.get("rollout"), Mapping)
        and isinstance(workers, (list, tuple)) and bool(workers)
    )
    precomputed_async_config = (
        isinstance(supplied, Mapping)
        and supplied.get("rollout_pipeline") == "bounded_complete_group_stream"
    )
    if not (full_async_config or precomputed_async_config):
        return None
    identity = config.get("artifact_identity")
    try:
        from ..contracts.ternary import TERNARY_QG_LAYOUT_VERSION, validate_artifact_identity

        validated = validate_artifact_identity(identity)
        artifact = validated["source"]["ninfer_artifact"]
        model_path = config.get("model_path")
        if (not isinstance(model_path, str)
                or Path(model_path).expanduser().resolve() != Path(artifact["path"]).expanduser().resolve()):
            raise RecoveryError("T2 model_path differs from the verified recovery artifact identity")
        return {
            "model_backend": "ternary",
            "rollout_pipeline": "bounded_complete_group_stream",
            "artifact_identity_schema": validated["schema_version"],
            "base_artifact_id": artifact["artifact_id"],
            "base_artifact_sha256": artifact["sha256"],
            "qg_layout": TERNARY_QG_LAYOUT_VERSION,
        }
    except RecoveryError:
        raise
    except Exception as exc:
        raise RecoveryError(f"cannot derive T2 recovery identity ({type(exc).__name__})") from exc


def preflight_compatibility(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the manager-computable compatibility subset without loading a model.

    Geometry, endpoints, manager paths, source-code hashes, and recovery mode are
    deliberately excluded so performance and code changes can resume a lineage.
    """
    if not isinstance(config, Mapping):
        raise TypeError("recovery compatibility config must be a mapping")
    learner = config.get("learner", {})
    if not isinstance(learner, Mapping):
        learner = {}
    lora = config.get("lora", {})
    if not isinstance(lora, Mapping):
        lora = {}

    def choose(name: str, flat_name: str, default: Any) -> Any:
        return learner.get(name, config.get(flat_name, default))

    model_path = config.get("model_path")
    if not model_path:
        raise RecoveryError("recovery compatibility requires model_path")
    try:
        normalized_model_path = str(_absolute_unresolved(str(model_path)))
    except OSError as exc:
        raise RecoveryError(f"cannot normalize base model path {model_path}: {exc}") from exc

    rank = lora.get("rank", config.get("lora_rank", 16))
    alpha = lora.get("alpha", config.get("lora_alpha", 16))
    betas = choose("adam_betas", "adam_betas", (0.9, 0.999))
    cap = choose("importance_sampling_cap", "importance_sampling_cap", 3.0)
    optimizer_hyperparameters = {
        "learning_rate": float(choose("lr", "lr", 1e-6)),
        "weight_decay": float(choose("weight_decay", "weight_decay", 0.0)),
        "betas": [float(value) for value in betas],
        "eps": float(choose("adam_eps", "adam_eps", 1e-8)),
        "max_grad_norm": float(choose("max_grad_norm", "max_grad_norm", 1.0)),
    }
    return _jsonable({
        "base": {"model_path": normalized_model_path},
        "peft": {"rank": int(rank), "alpha": int(alpha)},
        "objective": {
            "family": "grpo",
            "loss_type": "dapo",
            "epsilon": 0.2,
            "epsilon_high": 0.28,
            "scale_rewards": "group",
            "kl_beta": 0.0,
            "loss_aggregation": "global-token-mean",
            "importance_sampling_cap": float(cap),
        },
        "optimizer": {"hyperparameters": optimizer_hyperparameters},
        "semantics": _semantic_identity(config),
    })


def _matches_subset(expected: Any, actual: Any, path: str = "compatibility") -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            key in actual and _matches_subset(value, actual[key], f"{path}.{key}")
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            _matches_subset(left, right, path) for left, right in zip(expected, actual, strict=True)
        )
    return expected == actual


def _legacy_adapter_history(state_dir: Path) -> bool:
    """Detect old manager-run adapter archives that cannot restore Adam state."""
    runs = state_dir.parent / "runs"
    if not _is_real_dir(runs):
        return False
    patterns = (
        "*/artifacts/checkpoints/version-*",
        "*/artifacts/publications/version-*",
        "*/artifacts/policies/v*",
        "*/artifacts/worker-adapters/*/v*",
    )
    for pattern in patterns:
        for archive in runs.glob(pattern):
            if archive.is_symlink() or not archive.is_dir():
                continue
            names = {
                entry.name for entry in archive.iterdir()
                if entry.is_file() and not entry.is_symlink()
            }
            if {"adapter_config.json", "adapter_model.safetensors"} <= names:
                return True
    return False


def _verify_snapshot(state_dir: Path, pointer: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    expected_pointer_keys = {"schema", "lineage_id", "snapshot_id", "policy_version", "manifest_sha256"}
    if set(pointer) != expected_pointer_keys or pointer.get("schema") != LATEST_SCHEMA:
        raise RecoveryError("latest recovery pointer has an unsupported schema or fields")
    lineage_id = pointer.get("lineage_id")
    snapshot_id = pointer.get("snapshot_id")
    if not isinstance(lineage_id, str) or not _ID.fullmatch(lineage_id):
        raise RecoveryError("latest recovery pointer has an invalid lineage_id")
    if not isinstance(snapshot_id, str) or not _ID.fullmatch(snapshot_id):
        raise RecoveryError("latest recovery pointer has an invalid snapshot_id")
    manifest_sha = pointer.get("manifest_sha256")
    if manifest_sha is not None and (
        not isinstance(manifest_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha)
    ):
        raise RecoveryError("latest recovery pointer has invalid historical manifest digest metadata")
    version = pointer.get("policy_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise RecoveryError("latest recovery pointer has an invalid policy version")

    lineages_dir = state_dir / "lineages"
    lineage_dir = lineages_dir / lineage_id
    snapshots_dir = lineage_dir / "snapshots"
    snapshot_dir = snapshots_dir / snapshot_id
    _require_safe_directories([state_dir, lineages_dir, lineage_dir, snapshots_dir, snapshot_dir])
    if snapshot_dir.is_symlink() or not _is_real_dir(snapshot_dir):
        raise RecoveryError("latest recovery snapshot is missing or unsafe")
    entries = list(snapshot_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise RecoveryError("latest recovery snapshot contains an unsafe file or directory")
    manifest_path = snapshot_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RecoveryError("latest recovery snapshot has no manifest")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise RecoveryError("latest recovery snapshot has an unsupported schema")
    if manifest.get("lineage_id") != lineage_id or manifest.get("snapshot_id") != snapshot_id:
        raise RecoveryError("latest recovery snapshot identity does not match the pointer")
    if manifest.get("policy_version") != version:
        raise RecoveryError("latest recovery snapshot version does not match the pointer")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RecoveryError("latest recovery manifest has no payload file map")
    # v35 digest values are historical metadata; only names and any size inventory gate restore.
    names = set(files)
    if not _REQUIRED_FILES <= names or names - (_REQUIRED_FILES | _OPTIONAL_FILES):
        raise RecoveryError("latest recovery manifest has an incomplete or unknown payload list")
    if {entry.name for entry in entries} != names | {"manifest.json"}:
        raise RecoveryError("latest recovery snapshot contains missing or unlisted payload files")

    file_sizes = manifest.get("file_sizes")
    if "file_sizes" in manifest and (
        not isinstance(file_sizes, dict)
        or set(file_sizes) != names
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 0
               for size in file_sizes.values())
    ):
        raise RecoveryError("latest recovery manifest has an invalid payload file size map")
    for name in names:
        payload_path = snapshot_dir / name
        if payload_path.is_symlink() or not payload_path.is_file():
            raise RecoveryError(f"latest recovery payload is missing or unsafe for {name}")
        if file_sizes is not None and payload_path.stat().st_size != file_sizes[name]:
            raise RecoveryError(f"latest recovery payload size mismatch for {name}")

    state = _read_json(snapshot_dir / "state.json")
    for key in (
        "policy_version", "completed_updates", "lineage_id", "engine_epoch",
        "update_context", "publication_hashes",
    ):
        if state.get(key) != manifest.get(key):
            raise RecoveryError(f"latest recovery state/manifest mismatch at {key}")
    return snapshot_dir, manifest


def _inspect_recovery_snapshot(
    state_dir: str | Path,
    *,
    expected_compatibility: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], Path, dict[str, Any]] | None,
]:
    """Inspect recovery and return same-call verified restore data when resumable."""
    root = _absolute_unresolved(state_dir)
    result: dict[str, Any] = {
        "status": "empty",
        "state_dir": str(root),
        "lineage_id": None,
        "snapshot_id": None,
        "policy_version": None,
        "manifest_sha256": None,
        "compatibility": None,
        "update_context": None,
        "engine_epoch": None,
        "error": None,
    }
    if root.is_symlink():
        result.update(status="corrupt", error="training state directory is a symlink")
        return result, None
    pointer_path = root / "latest.json"
    if not pointer_path.exists() and not pointer_path.is_symlink():
        lineages = root / "lineages"
        if lineages.exists() or lineages.is_symlink():
            result.update(status="corrupt", error="lineage history exists but latest pointer is missing")
        elif _legacy_adapter_history(root):
            result.update(
                status="legacy_adapter_only",
                error=("adapter-only history exists without optimizer state; cannot resume it as full training state"),
            )
        return result, None
    if pointer_path.is_symlink() or not pointer_path.is_file():
        result.update(status="corrupt", error="latest recovery pointer is missing or unsafe")
        return result, None
    verified: tuple[dict[str, Any], Path, dict[str, Any]] | None = None
    try:
        pointer = _read_json(pointer_path)
        snapshot_dir, manifest = _verify_snapshot(root, pointer)
        compatibility = manifest.get("compatibility")
        if not isinstance(compatibility, dict):
            raise RecoveryError("latest recovery manifest has no compatibility identity")
        compatibility_hash = manifest.get("compatibility_sha256")
        if not isinstance(compatibility_hash, str) or _sha256_bytes(_canonical_bytes(compatibility)) != compatibility_hash:
            raise RecoveryError("latest recovery compatibility hash is invalid")
        if expected_compatibility is not None and not _matches_subset(expected_compatibility, compatibility):
            result.update(
                status="incompatible",
                error="latest full optimizer snapshot is incompatible with the current training identity",
            )
            return result, None
        result.update(
            status="resume",
            lineage_id=manifest["lineage_id"],
            snapshot_id=manifest["snapshot_id"],
            policy_version=manifest["policy_version"],
            manifest_sha256=pointer["manifest_sha256"],
            compatibility=compatibility,
            update_context=manifest.get("update_context"),
            engine_epoch=manifest.get("engine_epoch"),
            snapshot_path=str(snapshot_dir),
        )
        verified = (pointer, snapshot_dir, manifest)
    except Exception as exc:
        result.update(status="corrupt", error=str(exc))
    return result, verified


def inspect_recovery(
    state_dir: str | Path,
    *,
    expected_compatibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read-only CPU preflight for lineage, schema, sizes, and compatibility."""
    result, _verified = _inspect_recovery_snapshot(
        state_dir,
        expected_compatibility=expected_compatibility,
    )
    return result


def _parameter_layout(model: Any, optimizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names_by_id = {id(parameter): name for name, parameter in named}
    inventory = [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        for name, parameter in named
    ]
    groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        group_names = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise RecoveryError(f"optimizer group {group_index} contains an untracked parameter")
            group_names.append(name)
        options = {key: _jsonable(value) for key, value in group.items() if key != "params"}
        groups.append({"parameter_names": group_names, "options": options})
    return inventory, groups


def training_compatibility(
    config: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    trainer: Any,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Add loaded artifact, tokenizer, PEFT and optimizer structure to preflight identity."""
    identity = preflight_compatibility(config)
    base = dict(identity["base"])
    for key in ("base_fingerprint", "artifact_fingerprint"):
        value = evidence.get(key)
        if not isinstance(value, str) or not value:
            raise RecoveryError(f"loaded model evidence is missing {key}")
        base[key] = value
    if "quantization" in evidence:
        base["quantization"] = _jsonable(evidence["quantization"])
    identity["base"] = base

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise RecoveryError("loaded tokenizer cannot provide a stable vocabulary")
    tokenizer_identity = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "chat_template": getattr(tokenizer, "chat_template", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "vocabulary": get_vocab(),
    }
    identity["tokenizer_sha256"] = _sha256_bytes(_canonical_bytes(_jsonable(tokenizer_identity)))

    peft_configs = getattr(model, "peft_config", {})
    if isinstance(peft_configs, Mapping):
        peft_config_identity = {
            str(name): _jsonable(value.to_dict() if hasattr(value, "to_dict") else value)
            for name, value in peft_configs.items()
        }
    else:
        peft_config_identity = {}
    identity["peft"]["config_sha256"] = _sha256_bytes(_canonical_bytes(peft_config_identity))

    inventory, groups = _parameter_layout(model, trainer.optimizer)
    identity["trainable_parameters"] = inventory
    identity["optimizer"]["class"] = (
        f"{type(trainer.optimizer).__module__}.{type(trainer.optimizer).__qualname__}"
    )
    identity["optimizer"]["parameter_groups"] = groups
    scheduler = getattr(trainer, "lr_scheduler", None)
    identity["scheduler"] = {
        "class": None if scheduler is None else f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        "type": str(getattr(getattr(trainer, "args", None), "lr_scheduler_type", "constant")),
    }
    return _jsonable(identity)


def source_provenance() -> dict[str, Any]:
    """Record software/source hashes for audit only; they do not gate restore."""
    import importlib.metadata

    versions = {}
    for package in ("torch", "transformers", "trl", "peft", "accelerate", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_files = {}
    for path in (Path(__file__), Path(__file__).with_name("engine.py"), Path(__file__).with_name("trainer.py")):
        if path.is_file():
            source_files[path.name] = _sha256_file(path)
    return {"package_versions": versions, "on_disk_source_sha256": source_files}


def _optimizer_state_parameter_check(state: Any, groups: list[dict[str, Any]], model: Any) -> None:
    if isinstance(state, dict) and state.get("schema") == "qwen-rl-parallel-optimizer/v1":
        from .parallel_recovery import validate_parallel_optimizer
        validate_parallel_optimizer(state, groups, model)
        return
    if not isinstance(state, dict) or not isinstance(state.get("param_groups"), list) or not isinstance(state.get("state"), dict):
        raise RecoveryError("optimizer checkpoint has an invalid state_dict structure")
    if len(state["param_groups"]) != len(groups):
        raise RecoveryError("optimizer checkpoint parameter group count changed")
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    id_to_name: dict[Any, str] = {}
    for saved_group, group_layout in zip(state["param_groups"], groups, strict=True):
        parameter_ids = saved_group.get("params")
        names = group_layout["parameter_names"]
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(names):
            raise RecoveryError("optimizer checkpoint parameter ordering is incomplete")
        for parameter_id, name in zip(parameter_ids, names, strict=True):
            id_to_name[parameter_id] = name
    for parameter_id, parameter_state in state["state"].items():
        name = id_to_name.get(parameter_id)
        if name is None or not isinstance(parameter_state, dict):
            raise RecoveryError("optimizer checkpoint contains state for an unknown parameter")
        parameter = current[name]
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            tensor = parameter_state.get(key)
            if tensor is not None and (not hasattr(tensor, "shape") or tuple(tensor.shape) != tuple(parameter.shape)):
                raise RecoveryError(f"optimizer tensor shape mismatch for {name}.{key}")


class RecoveryStore:
    """Manage one atomically pointed full-state snapshot and its lineage."""

    def __init__(
        self,
        state_dir: str | Path,
        *,
        mode: str = "auto",
        lineage_id: str | None = None,
    ):
        if mode not in {"auto", "fresh"}:
            raise RecoveryError("recovery_mode must be 'auto' or 'fresh'")
        self.root = _absolute_unresolved(state_dir)
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise RecoveryError(f"training state root is not a safe directory: {self.root}")
        self.mode = mode
        self.pointer: dict[str, Any] | None = None
        self.current_snapshot_path: Path | None = None
        self.resumed = False
        self.parent_lineage: str | None = None

        if mode == "fresh":
            self.lineage_id = _normal_uuid(lineage_id, field="recovery_lineage_id")
            lineages = self.root / "lineages"
            if lineages.is_symlink() or (lineages.exists() and not lineages.is_dir()):
                raise RecoveryError("fresh recovery lineages path is unsafe")
            existing = self.root / "lineages" / self.lineage_id
            if existing.exists() or existing.is_symlink():
                raise RecoveryError("fresh recovery_lineage_id already exists; manager must allocate a new UUID")
            current = inspect_recovery(self.root)
            self.parent_lineage = current.get("lineage_id")
        else:
            preflight = inspect_recovery(self.root)
            if preflight["status"] == "resume":
                self.pointer = _read_json(self.root / "latest.json")
                self.lineage_id = preflight["lineage_id"]
                self.current_snapshot_path = Path(preflight["snapshot_path"])
                self.resumed = True
            elif preflight["status"] == "empty":
                self.lineage_id = uuid.uuid4().hex
                self.mode = "fresh"
            elif preflight["status"] == "legacy_adapter_only":
                raise RecoveryError(
                    "adapter-only history has no optimizer/RNG state and cannot be resumed; "
                    "stop the runtime and use --fresh to start a new lineage"
                )
            else:
                raise RecoveryError(
                    f"full training recovery is {preflight['status']}: {preflight['error']}; "
                    "no adapter-only fallback is allowed; use --fresh to start a new lineage"
                )
            if lineage_id is not None and lineage_id != self.lineage_id:
                raise RecoveryError("manager-selected lineage_id does not match the latest recovery pointer")

        self.lineage_dir = self.root / "lineages" / self.lineage_id
        self.snapshot_dir = self.lineage_dir / "snapshots"
        self.checkpoint_dir = self.lineage_dir / "checkpoints"
        self.publication_dir = self.lineage_dir / "publications"
        self.trainer_dir = self.lineage_dir / "trainer"

    def _write_adapter(self, model: Any, directory: Path) -> dict[str, Any]:
        from safetensors.torch import save_file

        state: dict[str, Any] = {}
        copy_started = time.perf_counter()
        copied_bytes = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                value = parameter.detach().to(device="cpu").contiguous().clone()
                state[name] = value
                copied_bytes += value.numel() * value.element_size()
        if not state:
            raise RecoveryError("cannot save a recovery snapshot with no trainable parameters")
        copy_seconds = time.perf_counter() - copy_started
        save_started = time.perf_counter()
        save_file(state, str(directory / "adapter_state.safetensors"), metadata={"format": "pt"})
        safetensors_save_seconds = time.perf_counter() - save_started
        peft_configs = getattr(model, "peft_config", None)
        if isinstance(peft_configs, Mapping):
            configs = {str(name): _jsonable(config.to_dict() if hasattr(config, "to_dict") else config)
                       for name, config in peft_configs.items()}
        else:
            configs = {}
        config_started = time.perf_counter()
        config_bytes = _canonical_bytes(configs) + b"\n"
        (directory / "adapter_config.json").write_bytes(config_bytes)
        config_write_seconds = time.perf_counter() - config_started
        return {
            "adapter_cpu_copy_seconds": copy_seconds,
            "adapter_cpu_copy_bytes": copied_bytes,
            "adapter_safetensors_save_wall_seconds": safetensors_save_seconds,
            "adapter_config_write_wall_seconds": config_write_seconds,
            "adapter_config_bytes": len(config_bytes),
        }

    def load(
        self,
        *,
        model: Any,
        trainer: Any,
        compatibility: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Validate and restore the full state. Empty/fresh lineages return None."""
        if not self.resumed:
            return None
        preflight, verified = _inspect_recovery_snapshot(
            self.root,
            expected_compatibility=compatibility,
        )
        if preflight["status"] != "resume":
            raise RecoveryError(
                f"full training snapshot is {preflight['status']}: {preflight['error']}; "
                "no adapter-only fallback is allowed; use --fresh for a new lineage"
            )
        if verified is None:
            raise RecoveryError("full training snapshot verification did not produce restore data")
        pointer, snapshot_dir, manifest = verified
        saved_compat = manifest["compatibility"]
        if _canonical_bytes(_jsonable(compatibility)) != _canonical_bytes(saved_compat):
            raise RecoveryError("full training snapshot compatibility identity differs from the current engine")

        import torch
        from safetensors.torch import load_file

        inventory, groups = _parameter_layout(model, trainer.optimizer)
        if inventory != manifest.get("parameter_inventory"):
            raise RecoveryError("trainable parameter inventory differs from the saved optimizer mapping")
        if groups != manifest.get("optimizer_parameter_groups"):
            raise RecoveryError("optimizer parameter groups/order/options differ from the saved mapping")

        adapter_state = load_file(str(snapshot_dir / "adapter_state.safetensors"), device="cpu")
        parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        if set(adapter_state) != set(parameters):
            raise RecoveryError("saved trainable adapter keys differ from the current model")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = adapter_state[name]
                if tuple(value.shape) != tuple(parameter.shape) or value.dtype != parameter.dtype:
                    raise RecoveryError(f"saved adapter tensor shape/dtype mismatch for {name}")
                parameter.copy_(value.to(device=parameter.device))

        optimizer_state = torch.load(
            snapshot_dir / "optimizer.pt", map_location="cpu", weights_only=True
        )
        _optimizer_state_parameter_check(optimizer_state, groups, model)
        try:
            load_trainer_payload(trainer, snapshot_dir)
        except TrainerCheckpointAdapterError as exc:
            raise RecoveryError(str(exc)) from exc
        metadata = _read_json(snapshot_dir / "state.json")
        self.pointer = pointer
        self.current_snapshot_path = snapshot_dir
        self._append_journal("restored", metadata, pointer["manifest_sha256"])
        return {
            **metadata,
            "snapshot_id": manifest["snapshot_id"],
            "snapshot_sha256": pointer["manifest_sha256"],
            "path": str(snapshot_dir),
            "compatibility_sha256": manifest["compatibility_sha256"],
        }

    def update_rollout_contract(
        self,
        *,
        expected_compatibility: Mapping[str, Any],
        old_contract_path: str | Path,
        new_contract_path: str | Path,
        turn_max_tokens: int,
        runtime_limits: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Revise only the episode token budget in a stopped lineage's latest snapshot.

        This operation is CPU-only. It verifies the old full snapshot, proves that
        the old and new contracts differ only in ``token_budget``, checks every
        manager-computable compatibility field, copies all immutable payloads,
        and atomically advances ``latest.json`` to a new snapshot ID at the same
        policy version. It never loads or saves model, optimizer, scheduler, or RNG
        state and never prunes the old snapshot.
        """
        if not self.resumed or self.pointer is None or self.current_snapshot_path is None:
            raise RecoveryError("rollout contract update requires a verified full recovery snapshot")
        if isinstance(turn_max_tokens, bool) or not isinstance(turn_max_tokens, int) or turn_max_tokens < 1:
            raise RecoveryError("turn_max_tokens must be a positive integer")
        expected_runtime_limits = {
            "rollout.turn_max_tokens",
            "worker_inference.max_generated_tokens",
            "arc.max_generated_tokens",
        }
        limits = _jsonable(runtime_limits)
        if not isinstance(limits, dict) or set(limits) != expected_runtime_limits:
            raise RecoveryError("runtime_limits must record the three generation caps")
        for name, delta in limits.items():
            if not isinstance(delta, dict) or set(delta) != {"from", "to"}:
                raise RecoveryError(f"runtime limit delta is malformed at {name}")
            for value in delta.values():
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise RecoveryError(f"runtime limit must be a positive integer at {name}")
        if limits["rollout.turn_max_tokens"]["to"] != turn_max_tokens:
            raise RecoveryError("turn_max_tokens does not match the recorded runtime limit")
        if limits["worker_inference.max_generated_tokens"]["to"] < turn_max_tokens:
            raise RecoveryError("Worker generation cap is below rollout.turn_max_tokens")

        old_path, old_contract, old_contract_sha, old_fingerprint = _read_rollout_contract(
            old_contract_path
        )
        new_path, new_contract, new_contract_sha, new_fingerprint = _read_rollout_contract(
            new_contract_path
        )
        if new_contract["token_budget"] < turn_max_tokens:
            raise RecoveryError("new episode token_budget must be at least rollout.turn_max_tokens")

        expected = _jsonable(expected_compatibility)
        expected_semantics = expected.get("semantics") if isinstance(expected, dict) else None
        if not isinstance(expected_semantics, dict):
            raise RecoveryError("current recovery identity has no semantic contract identity")
        if expected_semantics.get("arc_contract_sha256") != new_contract_sha:
            raise RecoveryError("new rollout contract file does not match current recovery identity")

        latest_path = self.root / "latest.json"
        try:
            original_pointer_bytes = latest_path.read_bytes()
            current_pointer = _read_json(latest_path)
        except Exception as exc:
            raise RecoveryError(f"cannot re-read latest recovery pointer: {exc}") from exc
        if current_pointer != self.pointer:
            raise RecoveryError("latest recovery pointer changed since this store was opened")

        preflight, verified = _inspect_recovery_snapshot(self.root)
        if preflight.get("status") != "resume" or verified is None:
            raise RecoveryError("latest full optimizer snapshot did not pass integrity verification")
        pointer, previous_snapshot_dir, previous_manifest = verified
        if pointer != current_pointer:
            raise RecoveryError("latest recovery pointer changed during integrity verification")
        old_compatibility = previous_manifest.get("compatibility")
        if not isinstance(old_compatibility, dict):
            raise RecoveryError("latest snapshot has no compatibility identity")
        old_semantics = old_compatibility.get("semantics")
        if not isinstance(old_semantics, dict):
            raise RecoveryError("latest snapshot has no semantic compatibility identity")

        state = _read_json(previous_snapshot_dir / "state.json")
        update_context = state.get("update_context")
        if not isinstance(update_context, dict):
            raise RecoveryError("latest snapshot has no logical-run update context")
        logical_job = update_context.get("logical_job")
        task_cursor = update_context.get("task_cursor")
        source_identity = update_context.get("source_identity")
        if not isinstance(logical_job, dict) or not isinstance(task_cursor, dict) or not isinstance(source_identity, dict):
            raise RecoveryError("latest snapshot has incomplete logical-run source identity")
        if (state.get("lineage_id") != pointer.get("lineage_id")
                or state.get("policy_version") != pointer.get("policy_version")
                or state.get("completed_updates") != previous_manifest.get("completed_updates")
                or logical_job.get("run_id") is None
                or task_cursor.get("next_group_index") is None):
            raise RecoveryError("latest checkpoint lineage, policy version, or run cursor is incomplete")
        for key, value in old_semantics.items():
            if source_identity.get(key) != value:
                raise RecoveryError(f"latest logical-run source identity differs at {key}")

        # A repeated invocation after a committed update must be a no-op, never a
        # second copy at the same policy version.
        if (old_semantics.get("arc_contract_sha256") == new_contract_sha
                and source_identity.get("arc_contract_sha256") == new_contract_sha
                and _matches_subset(expected, old_compatibility)):
            return {
                "status": "already_current",
                "lineage_id": pointer["lineage_id"],
                "snapshot_id": pointer["snapshot_id"],
                "policy_version": pointer["policy_version"],
                "run_id": logical_job.get("run_id"),
                "contract": {
                    "path": str(new_path), "sha256": new_contract_sha,
                    "fingerprint": new_fingerprint, "token_budget": new_contract["token_budget"],
                },
            }

        if old_path == new_path:
            raise RecoveryError("rollout contract update requires distinct old and new contract files")
        changed_fields = sorted(
            key for key in set(old_contract) | set(new_contract)
            if old_contract.get(key) != new_contract.get(key)
        )
        if changed_fields != ["token_budget"]:
            raise RecoveryError(
                "old and new rollout contracts may differ only in token_budget; "
                f"found {changed_fields}"
            )
        stored_old_contract_sha = old_semantics.get("arc_contract_sha256")
        if stored_old_contract_sha != old_contract_sha:
            raise RecoveryError("previous contract SHA does not match the latest snapshot identity")
        if source_identity.get("arc_contract_sha256") != old_contract_sha:
            raise RecoveryError("latest logical-run context does not match the previous contract SHA")

        expected_old_contract = _jsonable(expected)
        expected_old_contract["semantics"]["arc_contract_sha256"] = old_contract_sha
        if not _matches_subset(expected_old_contract, old_compatibility):
            raise RecoveryError(
                "current model, optimizer, data, objective, toolkit, or artifact identity "
                "differs from the latest recovery snapshot"
            )

        new_snapshot_id = uuid.uuid4().hex
        snapshot_dir = self.root / "lineages" / self.lineage_id / "snapshots"
        _require_safe_directories([self.root, self.root / "lineages", self.lineage_dir, snapshot_dir])
        staging = snapshot_dir / f".staging-{new_snapshot_id}"
        final = snapshot_dir / new_snapshot_id
        if final.exists() or final.is_symlink():
            raise RecoveryError("random snapshot ID collision")
        staging.mkdir()
        committed = False
        receipt_staging: Path | None = None
        receipt_final: Path | None = None
        try:
            new_context = _jsonable(update_context)
            new_context["source_identity"]["arc_contract_sha256"] = new_contract_sha
            transition = {
                "schema": "qwen-rl-rollout-contract-update/v1",
                "reason": "rollout_contract_update",
                "lineage_id": pointer["lineage_id"],
                "from_snapshot_id": previous_manifest["snapshot_id"],
                "from_manifest_sha256": pointer["manifest_sha256"],
                "to_snapshot_id": new_snapshot_id,
                "policy_version": pointer["policy_version"],
                "completed_updates": previous_manifest["completed_updates"],
                "changed_contract_fields": ["token_budget"],
                "from_contract": {
                    "path": str(old_path), "sha256": old_contract_sha,
                    "fingerprint": old_fingerprint, "token_budget": old_contract["token_budget"],
                },
                "to_contract": {
                    "path": str(new_path), "sha256": new_contract_sha,
                    "fingerprint": new_fingerprint, "token_budget": new_contract["token_budget"],
                },
                "runtime_limits": limits,
            }
            new_state = _jsonable(state)
            new_state["reason"] = "rollout_contract_update"
            new_state["update_context"] = new_context
            new_state["rollout_contract_update"] = transition
            state_bytes = _canonical_bytes(new_state) + b"\n"
            old_files = previous_manifest.get("files")
            if not isinstance(old_files, dict) or not _REQUIRED_FILES <= set(old_files):
                raise RecoveryError("latest snapshot has an incomplete payload file map")
            old_file_sizes = previous_manifest.get("file_sizes")
            if old_file_sizes is not None and (
                not isinstance(old_file_sizes, dict)
                or set(old_file_sizes) != set(old_files)
                or any(isinstance(size, bool) or not isinstance(size, int) or size < 0
                       for size in old_file_sizes.values())
            ):
                raise RecoveryError("latest snapshot has an invalid payload file size map")
            file_sizes: dict[str, int] = {}
            digest_free_files: dict[str, None] = {}
            for name in sorted(old_files):
                source = previous_snapshot_dir / name
                target = staging / name
                if source.is_symlink() or not source.is_file():
                    raise RecoveryError(f"immutable recovery payload is missing or unsafe for {name}")
                source_size = source.stat().st_size
                if old_file_sizes is not None and source_size != old_file_sizes[name]:
                    raise RecoveryError(f"immutable recovery payload size mismatch for {name}")
                if name == "state.json":
                    _write_file_fsynced(target, state_bytes)
                else:
                    shutil.copyfile(source, target)
                    with target.open("rb") as stream:
                        os.fsync(stream.fileno())
                copied_size = target.stat().st_size
                if name != "state.json" and copied_size != source_size:
                    raise RecoveryError(f"immutable recovery payload size changed while copying {name}")
                file_sizes[name] = copied_size
                digest_free_files[name] = None

            new_compatibility = _jsonable(old_compatibility)
            new_compatibility["semantics"]["arc_contract_sha256"] = new_contract_sha
            compatibility_sha = _sha256_bytes(_canonical_bytes(new_compatibility))
            new_manifest = _jsonable(previous_manifest)
            new_manifest.update({
                "snapshot_id": new_snapshot_id,
                "update_context": new_context,
                "compatibility": new_compatibility,
                "compatibility_sha256": compatibility_sha,
                "files": digest_free_files,
                "file_sizes": file_sizes,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "reason": "rollout_contract_update",
                "rollout_contract_update": transition,
            })
            manifest_bytes = _canonical_bytes(new_manifest) + b"\n"
            _write_file_fsynced(staging / "manifest.json", manifest_bytes)
            _fsync_directory(staging)
            os.rename(staging, final)
            _fsync_directory(snapshot_dir)

            # Re-read both source contracts immediately before pointer commit. A
            # changed source file invalidates the proof and leaves latest untouched.
            current_old_path, current_old_contract, current_old_sha, current_old_fingerprint = (
                _read_rollout_contract(old_path)
            )
            current_new_path, current_new_contract, current_new_sha, current_new_fingerprint = (
                _read_rollout_contract(new_path)
            )
            if ((current_old_path, current_old_contract, current_old_sha, current_old_fingerprint)
                    != (old_path, old_contract, old_contract_sha, old_fingerprint)
                    or (current_new_path, current_new_contract, current_new_sha, current_new_fingerprint)
                    != (new_path, new_contract, new_contract_sha, new_fingerprint)):
                raise RecoveryError("rollout contract source changed before pointer commit")

            receipt_dir = self.lineage_dir / "contract-updates"
            if receipt_dir.is_symlink() or (receipt_dir.exists() and not receipt_dir.is_dir()):
                raise RecoveryError("rollout contract receipt directory is unsafe")
            receipt_dir.mkdir(exist_ok=True)
            _require_safe_directories([self.lineage_dir, receipt_dir])
            receipt_final = receipt_dir / f"{new_snapshot_id}.json"
            if receipt_final.exists() or receipt_final.is_symlink():
                raise RecoveryError("rollout contract receipt ID collision")
            receipt_staging = receipt_dir / f".{new_snapshot_id}.staging"
            receipt = {
                "schema": "qwen-rl-rollout-contract-update-receipt/v1",
                "operation_id": new_snapshot_id,
                "transition": transition,
                "from_manifest_sha256": pointer.get("manifest_sha256"),
                "from_manifest": previous_manifest,
                "from_payload_sha256": previous_manifest.get("files"),
                "to_manifest_sha256": None,
                "to_manifest": new_manifest,
                "to_payload_sha256": None,
                "to_file_sizes": file_sizes,
            }
            receipt_bytes = _canonical_bytes(receipt) + b"\n"
            _write_file_fsynced(receipt_staging, receipt_bytes)
            os.rename(receipt_staging, receipt_final)
            receipt_staging = None
            _fsync_directory(receipt_dir)
            _fsync_directory(self.lineage_dir)
            new_pointer = {
                "schema": LATEST_SCHEMA,
                "lineage_id": pointer["lineage_id"],
                "snapshot_id": new_snapshot_id,
                "policy_version": pointer["policy_version"],
                "manifest_sha256": None,
            }
            durability_warning = _atomic_json_if_unchanged(
                latest_path, new_pointer, expected_bytes=original_pointer_bytes
            )
            committed = True
            self.pointer = new_pointer
            self.current_snapshot_path = final
            self.resumed = True
            self.mode = "resumed"
            self._append_journal("rollout_contract_updated", new_state, None)
            return {
                "status": ("committed_durability_unconfirmed" if durability_warning else "updated"),
                "lineage_id": pointer["lineage_id"],
                "previous_snapshot_id": previous_manifest["snapshot_id"],
                "snapshot_id": new_snapshot_id,
                "policy_version": pointer["policy_version"],
                "completed_updates": previous_manifest["completed_updates"],
                "run_id": logical_job.get("run_id"),
                "from_contract": transition["from_contract"],
                "to_contract": transition["to_contract"],
                "compatibility_sha256": compatibility_sha,
                "manifest_sha256": None,
                "contract_update_receipt": str(receipt_final.relative_to(self.lineage_dir)),
                "contract_update_receipt_sha256": None,
                "durability_warning": durability_warning,
            }
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if receipt_staging is not None and receipt_staging.exists():
                receipt_staging.unlink()
            if not committed:
                try:
                    current_latest = _read_json(latest_path)
                    latest_references_new_snapshot = current_latest.get("snapshot_id") == new_snapshot_id
                except Exception:
                    latest_references_new_snapshot = False
                if not latest_references_new_snapshot:
                    if final.exists():
                        shutil.rmtree(final, ignore_errors=True)
                    if receipt_final is not None and receipt_final.exists():
                        receipt_final.unlink()
            raise

    def save(
        self,
        *,
        model: Any,
        trainer: Any,
        compatibility: Mapping[str, Any],
        policy_version: int,
        engine_epoch: str,
        update_context: Mapping[str, Any] | None,
        reason: str,
        publication_hashes: Mapping[str, str | None] | None = None,
        source_provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a complete state and atomically advance the latest pointer."""
        save_started = time.perf_counter()
        controller_seconds: dict[str, float] = {}
        performance: dict[str, Any] = {
            "schema": "qwen-rl-checkpoint-performance/v1",
            "rank_collection": _jsonable(
                getattr(trainer, "checkpoint_collection_performance", {}) or {}
            ),
            "controller_seconds": controller_seconds,
            "file_bytes": {},
            "file_fsync_seconds": {},
        }
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise RecoveryError("policy_version must be a non-negative integer")
        if not reason:
            raise RecoveryError("snapshot reason must be non-empty")
        engine_epoch = _normal_uuid(engine_epoch, field="engine_epoch")
        compatibility_started = time.perf_counter()
        normalized_compat = _jsonable(compatibility)
        compatibility_sha = _sha256_bytes(_canonical_bytes(normalized_compat))
        normalized_context = None if update_context is None else _jsonable(update_context)
        normalized_publication_hashes = (
            None if publication_hashes is None else _jsonable(publication_hashes)
        )
        if isinstance(normalized_publication_hashes, dict) and all(
            value is None for value in normalized_publication_hashes.values()
        ):
            normalized_publication_hashes = None
        controller_seconds["compatibility_normalize_hash_seconds"] = (
            time.perf_counter() - compatibility_started
        )
        directory_started = time.perf_counter()
        lineage_dir = self.lineage_dir
        if self.root.is_symlink():
            raise RecoveryError("training state root is a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise RecoveryError("training state root is unsafe")
        lineages_dir = self.root / "lineages"
        if lineages_dir.is_symlink() or (lineages_dir.exists() and not lineages_dir.is_dir()):
            raise RecoveryError("training lineages path is unsafe")
        lineages_dir.mkdir(exist_ok=True)
        if lineage_dir.is_symlink() or (lineage_dir.exists() and not lineage_dir.is_dir()):
            raise RecoveryError("training lineage path is unsafe")
        lineage_dir.mkdir(exist_ok=True)
        if self.snapshot_dir.is_symlink() or (self.snapshot_dir.exists() and not self.snapshot_dir.is_dir()):
            raise RecoveryError("training snapshot path is unsafe")
        self.snapshot_dir.mkdir(exist_ok=True)
        _require_safe_directories([self.root, lineages_dir, lineage_dir, self.snapshot_dir])

        snapshot_id = uuid.uuid4().hex
        staging = self.snapshot_dir / f".staging-{snapshot_id}"
        final = self.snapshot_dir / snapshot_id
        staging.mkdir()
        controller_seconds["directory_prepare_seconds"] = time.perf_counter() - directory_started
        try:
            adapter_performance = self._write_adapter(model, staging)
            performance.update(adapter_performance)
            controller_seconds["adapter_copy_and_save_wall_seconds"] = (
                adapter_performance["adapter_cpu_copy_seconds"]
                + adapter_performance["adapter_safetensors_save_wall_seconds"]
                + adapter_performance["adapter_config_write_wall_seconds"]
            )
            parameter_layout_started = time.perf_counter()
            inventory, groups = _parameter_layout(model, trainer.optimizer)
            controller_seconds["parameter_layout_seconds"] = (
                time.perf_counter() - parameter_layout_started
            )
            try:
                trainer_save_performance = save_trainer_payload(trainer, staging)
            except TrainerCheckpointAdapterError as exc:
                raise RecoveryError(str(exc)) from exc
            performance["trainer_save"] = trainer_save_performance
            controller_seconds["trainer_payload_save_wall_seconds"] = (
                trainer_save_performance.get("trainer_optimizer_scheduler_helper_wall_seconds", 0.0)
                + trainer_save_performance.get("trainer_rng_helper_wall_seconds", 0.0)
                + trainer_save_performance.get("trainer_state_json_save_wall_seconds", 0.0)
                + trainer_save_performance.get("trainer_scaler_save_wall_seconds", 0.0)
            )
            trainer_state = getattr(trainer, "state", None)
            counters = {
                "trainer_global_step": int(getattr(trainer_state, "global_step", 0) or 0),
                "trainer_epoch": getattr(trainer_state, "epoch", None),
                "num_input_tokens_seen": int(getattr(trainer_state, "num_input_tokens_seen", 0) or 0),
            }
            metadata = {
                "schema": SNAPSHOT_SCHEMA,
                "lineage_id": self.lineage_id,
                "policy_version": policy_version,
                "completed_updates": policy_version,
                "engine_epoch": engine_epoch,
                "update_context": normalized_context,
                "publication_hashes": normalized_publication_hashes,
                "reason": reason,
                "trainer_counters": counters,
                "source_provenance": _jsonable(source_provenance or {}),
                "parent_lineage": self.parent_lineage,
            }
            state_write_started = time.perf_counter()
            state_bytes = _canonical_bytes(metadata) + b"\n"
            (staging / "state.json").write_bytes(state_bytes)
            performance["state_metadata_bytes"] = len(state_bytes)
            controller_seconds["state_metadata_serialize_write_seconds"] = (
                time.perf_counter() - state_write_started
            )
            names = {entry.name for entry in staging.iterdir()}
            if not _REQUIRED_FILES <= names or names - (_REQUIRED_FILES | _OPTIONAL_FILES):
                raise RecoveryError(f"Trainer produced an incomplete or unknown recovery payload: {sorted(names)}")

            payload_entries = sorted(staging.iterdir(), key=lambda item: item.name)
            file_sizes: dict[str, int] = {}
            safe_payload_entries = []
            for entry in payload_entries:
                if entry.is_symlink() or not entry.is_file():
                    raise RecoveryError(f"Trainer produced an unsafe recovery payload: {entry.name}")
                file_sizes[entry.name] = entry.stat().st_size
                safe_payload_entries.append(entry)

            payload_fsync_started = time.perf_counter()
            fsync_error = None
            try:
                _fsync_payload_files(
                    safe_payload_entries, performance["file_fsync_seconds"]
                )
            except BaseException as exc:
                fsync_error = exc
            controller_seconds["payload_file_fsync_wall_seconds"] = (
                time.perf_counter() - payload_fsync_started
            )
            performance["file_bytes"].update(file_sizes)
            if fsync_error is not None:
                raise fsync_error
            manifest = {
                "schema": SNAPSHOT_SCHEMA,
                "lineage_id": self.lineage_id,
                "snapshot_id": snapshot_id,
                "policy_version": policy_version,
                "completed_updates": policy_version,
                "engine_epoch": engine_epoch,
                "update_context": normalized_context,
                "publication_hashes": normalized_publication_hashes,
                "compatibility": normalized_compat,
                "compatibility_sha256": compatibility_sha,
                "parameter_inventory": inventory,
                "optimizer_parameter_groups": groups,
                "trainer_counters": counters,
                # Null values make the disabled payload-digest policy explicit.
                "files": {name: None for name in file_sizes},
                "file_sizes": file_sizes,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_started = time.perf_counter()
            manifest_bytes = _canonical_bytes(manifest) + b"\n"
            controller_seconds["manifest_json_serialize_seconds"] = (
                time.perf_counter() - manifest_started
            )
            performance["file_bytes"]["manifest.json"] = len(manifest_bytes)
            _write_file_fsynced(
                staging / "manifest.json",
                manifest_bytes,
                performance=controller_seconds,
                prefix="manifest",
            )
            staging_directory_fsync_started = time.perf_counter()
            _fsync_directory(staging)
            controller_seconds["staging_directory_fsync_seconds"] = (
                time.perf_counter() - staging_directory_fsync_started
            )
            if final.exists() or final.is_symlink():
                raise RecoveryError("random snapshot ID collision")
            rename_started = time.perf_counter()
            os.rename(staging, final)
            controller_seconds["snapshot_rename_seconds"] = time.perf_counter() - rename_started
            snapshot_directory_fsync_started = time.perf_counter()
            _fsync_directory(self.snapshot_dir)
            controller_seconds["snapshot_directory_fsync_seconds"] = (
                time.perf_counter() - snapshot_directory_fsync_started
            )

            pointer = {
                "schema": LATEST_SCHEMA,
                "lineage_id": self.lineage_id,
                "snapshot_id": snapshot_id,
                "policy_version": policy_version,
                "manifest_sha256": None,
            }
            _atomic_json(
                self.root / "latest.json",
                pointer,
                performance=controller_seconds,
                prefix="latest_pointer",
            )
            self.pointer = pointer
            self.current_snapshot_path = final
            self.resumed = True
            self.mode = "resumed"
            journal_metadata = {
                **metadata,
                "checkpoint_performance": _jsonable(performance),
            }
            journal_started = time.perf_counter()
            self._append_journal("snapshot_committed", journal_metadata, None)
            controller_seconds["journal_append_wall_seconds"] = time.perf_counter() - journal_started
            prune_started = time.perf_counter()
            self._prune_snapshots(keep=(self.lineage_id, snapshot_id))
            controller_seconds["snapshot_prune_seconds"] = time.perf_counter() - prune_started
            controller_seconds["total_save_wall_seconds"] = time.perf_counter() - save_started
            return {
                **metadata,
                "snapshot_id": snapshot_id,
                "snapshot_sha256": None,
                "path": str(final),
                "compatibility_sha256": compatibility_sha,
                "checkpoint_performance": performance,
            }
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def _append_journal(
        self,
        event: str,
        metadata: Mapping[str, Any],
        manifest_sha: str | None,
    ) -> None:
        try:
            self.lineage_dir.mkdir(parents=True, exist_ok=True)
            journal = self.lineage_dir / "journal.jsonl"
            if journal.is_symlink():
                raise RecoveryError("recovery journal is a symlink")
            record = {
                "event": event,
                "policy_version": metadata.get("policy_version"),
                "snapshot_id": self.pointer.get("snapshot_id") if self.pointer else None,
                "manifest_sha256": manifest_sha,
                "engine_epoch": metadata.get("engine_epoch"),
                "update_context": metadata.get("update_context"),
                "at": datetime.now(timezone.utc).isoformat(),
            }
            if "checkpoint_performance" in metadata:
                record["checkpoint_performance"] = metadata["checkpoint_performance"]
            with journal.open("ab") as stream:
                stream.write(_canonical_bytes(record) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(self.lineage_dir)
        except Exception:
            # The pointer and full snapshot are the commit record. The journal is
            # an audit aid and can be reconstructed from the pointed manifest.
            _LOG.exception("could not append non-authoritative recovery journal event")

    def _prune_snapshots(self, *, keep: tuple[str, str]) -> None:
        lineages = self.root / "lineages"
        if not _is_real_dir(lineages):
            return
        for lineage in lineages.iterdir():
            if lineage.is_symlink() or not lineage.is_dir():
                continue
            snapshots = lineage / "snapshots"
            if snapshots.is_symlink() or not snapshots.is_dir():
                continue
            for snapshot in snapshots.iterdir():
                if lineage.name == keep[0] and snapshot.name == keep[1]:
                    continue
                if snapshot.is_symlink():
                    continue
                if snapshot.is_dir():
                    shutil.rmtree(snapshot)
                elif snapshot.is_file():
                    snapshot.unlink()


__all__ = [
    "RecoveryError",
    "RecoveryStore",
    "inspect_recovery",
    "preflight_compatibility",
    "source_provenance",
    "training_compatibility",
]
