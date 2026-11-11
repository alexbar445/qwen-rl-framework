"""Shared model installation references and validated identity-sidecar relocation."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_rl import paths as qwen_paths
from qwen_rl.contracts.ternary import (
    CODER390_MODEL_NAME,
    MODEL_NAME,
    TernaryContractError,
    artifact_report,
    load_artifact_identity,
    load_pinned_artifact_identity,
)


def _regular_target_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise TernaryContractError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TernaryContractError(f"{label} is missing or unreadable: {path}") from exc
    if not resolved.is_file():
        raise TernaryContractError(f"{label} must be a regular file: {path}")
    return resolved


def _relocated_document(
    source_document: dict[str, Any],
    *,
    model_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Copy an identity and change only its two supported location fields."""
    document = json.loads(json.dumps(source_document))
    source = document["source"]
    source["ninfer_artifact"]["path"] = str(model_path)
    artifact_report(document)
    model_id = document["model"]["id"]
    if model_id == MODEL_NAME:
        report_key = "conversion_report"
    elif model_id == CODER390_MODEL_NAME:
        report_key = "artifact_report"
    else:
        raise TernaryContractError("artifact report lookup has an unsupported model profile")
    source[report_key]["path"] = str(report_path)
    return document


def _load_matching_target(
    target_path: Path,
    expected_document: dict[str, Any],
) -> Path:
    """Accept an existing sidecar only when it is this exact validated copy."""
    if target_path.is_symlink():
        raise TernaryContractError(f"target artifact identity must not be a symlink: {target_path}")
    existing = load_artifact_identity(target_path, verify_files=False).as_dict()
    if existing != expected_document:
        raise TernaryContractError(
            f"target artifact identity already exists with different contents: {target_path}"
        )
    load_pinned_artifact_identity(target_path)
    return target_path


def relocate_artifact_identity(
    source_identity_path: str | Path,
    target_identity_path: str | Path,
    *,
    model_path: str | Path,
    conversion_report_path: str | Path,
) -> Path:
    """Write a validated sidecar for a copied NInfer base and report.

    Only ``source.ninfer_artifact.path`` and the profile-selected report
    descriptor path are changed. Stable artifact identity data, including
    declared SHA values, format/layout, and input-basis metadata, is copied as
    recorded. The model payload is never hashed.

    The source sidecar is loaded without checking its old file locations, so
    those files may already have been removed. The new sidecar is validated
    with ``load_pinned_artifact_identity`` against the target files before it
    is installed. Repeating the operation with the same paths is safe.
    """
    source_path = Path(source_identity_path).expanduser()
    source_resolved = source_path.resolve(strict=True)
    target_path = Path(target_identity_path).expanduser()
    if target_path.is_symlink():
        raise TernaryContractError(f"target artifact identity must not be a symlink: {target_path}")
    target_path = target_path.resolve(strict=False)
    if source_resolved == target_path:
        raise TernaryContractError("source and target artifact identity paths must differ")

    model = _regular_target_file(model_path, "target NInfer artifact")
    report_path = _regular_target_file(conversion_report_path, "target conversion report")

    source_identity = load_artifact_identity(source_path, verify_files=False)
    expected_document = _relocated_document(
        source_identity.as_dict(), model_path=model, report_path=report_path
    )

    if target_path.exists():
        return _load_matching_target(target_path, expected_document)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".tmp", dir=target_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(expected_document, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        # Validate the complete target binding before making the sidecar visible.
        load_pinned_artifact_identity(temporary_path)
        try:
            # A same-directory hard link installs atomically without replacing
            # a sidecar created concurrently by another relocation attempt.
            os.link(temporary_path, target_path)
        except FileExistsError:
            return _load_matching_target(target_path, expected_document)
        directory_fd = os.open(target_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass

    return target_path


@dataclass(frozen=True, slots=True)
class ModelInstallation:
    """References to an existing base/report and their runtime identity sidecar."""

    artifact_path: Path
    conversion_report_path: Path
    identity_path: Path

    @classmethod
    def from_source_layout(cls, layout: Mapping[str, Any]) -> ModelInstallation:
        return cls(*(Path(layout[key]).expanduser().resolve(strict=False) for key in (
            "model_path", "conversion_report_path", "artifact_identity_path",
        )))

    @classmethod
    def from_runtime(cls, runtime: Mapping[str, Any]) -> ModelInstallation:
        model = runtime.get("model")
        worker = runtime.get("worker_inference")
        model = model if isinstance(model, Mapping) else {}
        worker = worker if isinstance(worker, Mapping) else {}

        def resolve(value: Any, label: str) -> Path:
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"runtime {label} path is missing")
            path = Path(value).expanduser()
            return (path if path.is_absolute() else qwen_paths.ROOT / path).resolve(strict=False)

        artifact = resolve(runtime.get("model_path"), "model")
        for field in ("ninfer_artifact_path", "model_path"):
            if worker.get(field) and resolve(worker[field], field) != artifact:
                raise ValueError(f"worker_inference.{field} differs from the model installation")
        identity = model.get("artifact_identity_path") or runtime.get("artifact_identity_path")
        return cls(
            artifact,
            resolve(worker.get("ninfer_conversion_report"), "conversion report"),
            resolve(identity, "artifact identity"),
        )

    def with_identity(self, target_path: str | Path) -> ModelInstallation:
        target = Path(target_path).expanduser().resolve(strict=False)
        relocate_artifact_identity(
            self.identity_path, target,
            model_path=self.artifact_path,
            conversion_report_path=self.conversion_report_path,
        )
        return ModelInstallation(self.artifact_path, self.conversion_report_path, target)

    def apply_to_runtime(self, runtime: dict[str, Any]) -> dict[str, Any]:
        model = runtime.setdefault("model", {})
        worker = runtime.setdefault("worker_inference", {})
        if not isinstance(model, dict) or not isinstance(worker, dict):
            raise TypeError("runtime model and worker_inference must be objects")
        runtime["model_path"] = str(self.artifact_path)
        model["artifact_identity_path"] = str(self.identity_path)
        worker.update(
            model_path=str(self.artifact_path),
            ninfer_artifact_path=str(self.artifact_path),
            ninfer_conversion_report=str(self.conversion_report_path),
        )
        runtime.pop("artifact_identity_path", None)
        return runtime

    def public_dict(self) -> dict[str, str]:
        return {
            "artifact_path": str(self.artifact_path),
            "conversion_report_path": str(self.conversion_report_path),
            "identity_path": str(self.identity_path),
        }

__all__ = ["ModelInstallation", "relocate_artifact_identity"]
