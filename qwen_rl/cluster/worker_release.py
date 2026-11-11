"""Strict, explicit Worker release catalog and GPU image selection."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from .images import ImageArchiveDescriptor


_RELEASE_SCHEMA = "qwen-rl-worker-release/v1"
_COMPUTE_CAPABILITY = re.compile(r"^(?:0|[1-9][0-9]?)\.[0-9]$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CUDA_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?$")
_RUNTIME_CAPABILITIES_BY_BUILD_ARCH = {
    "70": frozenset({"7.0"}),
    "80": frozenset({"8.0"}),
    "86": frozenset({"8.6", "8.9"}),
    "89": frozenset({"8.9"}),
    "120a": frozenset({"12.0"}),
}


class WorkerReleaseError(ValueError):
    """A Worker release catalog is invalid or has no exact GPU image match."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class WorkerImageVariant(_StrictFrozenModel):
    """One image build that explicitly serves one or more GPU capabilities."""

    compute_capabilities: tuple[StrictStr, ...] = Field(min_length=1)
    build_arch: StrictStr
    cuda_version: StrictStr = Field(min_length=1, max_length=40)
    image: ImageArchiveDescriptor

    @field_validator("compute_capabilities")
    @classmethod
    def canonical_compute_capabilities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not _COMPUTE_CAPABILITY.fullmatch(value):
                raise ValueError("compute capabilities must use canonical major.minor notation")
        if len(set(values)) != len(values):
            raise ValueError("a variant cannot declare the same compute capability twice")
        return values

    @field_validator("build_arch")
    @classmethod
    def supported_build_arch(cls, value: str) -> str:
        if value not in _RUNTIME_CAPABILITIES_BY_BUILD_ARCH:
            raise ValueError("build_arch must be one of the supported NInfer targets: 70, 80, 86, 89, 120a")
        return value

    @field_validator("cuda_version")
    @classmethod
    def nonblank_cuda_version(cls, value: str) -> str:
        if not _CUDA_VERSION.fullmatch(value):
            raise ValueError("cuda_version must be a numeric major.minor[.patch] toolkit version")
        return value

    @model_validator(mode="after")
    def build_contract_matches_declared_gpus(self) -> "WorkerImageVariant":
        unsupported = set(self.compute_capabilities) - _RUNTIME_CAPABILITIES_BY_BUILD_ARCH[self.build_arch]
        if unsupported:
            raise ValueError(
                f"build_arch {self.build_arch} is not documented for compute capabilities "
                f"{', '.join(sorted(unsupported))}"
            )
        version_match = _CUDA_VERSION.fullmatch(self.cuda_version)
        if version_match is None:  # field validation reports the detailed schema error
            return self
        version = tuple(int(part or 0) for part in version_match.groups())
        if self.build_arch == "70":
            compatible_toolkits = {(12, 8), (12, 9)}
            if version[:2] not in compatible_toolkits:
                raise ValueError("build_arch 70 requires CUDA toolkit 12.8 or 12.9")
        elif self.build_arch == "120a":
            if version < (13, 1, 0):
                raise ValueError("build_arch 120a requires CUDA toolkit 13.1 or newer")
        elif version < (12, 8, 0):
            raise ValueError(f"build_arch {self.build_arch} requires CUDA toolkit 12.8 or newer")
        return self


class WorkerReleaseIdentity(_StrictFrozenModel):
    """Code identity shared by every architecture variant of one release."""

    release_id: StrictStr = Field(min_length=1)
    worker_source_sha256: StrictStr
    engine_revision: StrictStr
    backend: Literal["ninfer"]

    @field_validator("release_id")
    @classmethod
    def nonblank_release_id(cls, value: str) -> str:
        if not value.strip() or any(char in value for char in "\r\n\x00"):
            raise ValueError("release_id must be a nonempty single-line identifier")
        return value

    @field_validator("worker_source_sha256")
    @classmethod
    def valid_source_sha(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("worker_source_sha256 must be a lowercase 64-character SHA-256")
        return value

    @field_validator("engine_revision")
    @classmethod
    def valid_engine_revision(cls, value: str) -> str:
        if not _GIT_REVISION.fullmatch(value):
            raise ValueError("engine_revision must be a full lowercase 40-character git revision")
        return value


class WorkerRelease(WorkerReleaseIdentity):
    """Validated manifest for a single NInfer Worker source/engine release."""

    schema_version: Literal["qwen-rl-worker-release/v1"] = Field(alias="schema")
    variants: tuple[WorkerImageVariant, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def capabilities_are_unambiguous(self) -> "WorkerRelease":
        seen: set[str] = set()
        for variant in self.variants:
            overlap = seen.intersection(variant.compute_capabilities)
            if overlap:
                values = ", ".join(sorted(overlap))
                raise ValueError(f"compute capability is declared by multiple variants: {values}")
            seen.update(variant.compute_capabilities)
        return self

    @property
    def release_identity(self) -> WorkerReleaseIdentity:
        return WorkerReleaseIdentity(
            release_id=self.release_id,
            worker_source_sha256=self.worker_source_sha256,
            engine_revision=self.engine_revision,
            backend=self.backend,
        )


class ResolvedWorkerImage(_StrictFrozenModel):
    """A release image selected by an exact GPU compute-capability match."""

    descriptor: ImageArchiveDescriptor
    release_identity: WorkerReleaseIdentity
    compute_capability: StrictStr
    build_arch: StrictStr
    cuda_version: StrictStr


def expected_worker_image_labels(resolved: ResolvedWorkerImage) -> dict[str, str]:
    """Return the exact image labels required to match a resolved release variant."""
    if not isinstance(resolved, ResolvedWorkerImage):
        raise WorkerReleaseError("resolved worker image must be a ResolvedWorkerImage")
    return {
        "com.ninfer.engine.revision": resolved.release_identity.engine_revision,
        "com.qwen-rl.worker.source.sha256": resolved.release_identity.worker_source_sha256,
        "com.qwen-rl.worker.cuda-arch": resolved.build_arch,
        "com.qwen-rl.worker.cuda-version": resolved.cuda_version,
    }


def load_worker_release(
    source: str | Path | Mapping[str, Any] | WorkerRelease,
) -> WorkerRelease:
    """Load a release from JSON or a strict mapping and validate its full schema."""
    if isinstance(source, WorkerRelease):
        raw: Any = source.model_dump(mode="python", by_alias=True)
    elif isinstance(source, Mapping):
        raw = dict(source)
    elif isinstance(source, (str, Path)):
        try:
            text = Path(source).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise WorkerReleaseError("worker release file could not be read") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WorkerReleaseError("worker release file is not valid JSON") from exc
    else:
        raise WorkerReleaseError("worker release source must be a path, mapping, or WorkerRelease")

    if not isinstance(raw, Mapping):
        raise WorkerReleaseError("worker release must be a JSON object")
    try:
        normalized = _normalize_manifest_arrays(raw)
        return WorkerRelease.model_validate(normalized, strict=True)
    except (ValidationError, TypeError, ValueError) as exc:
        raise WorkerReleaseError(f"worker release does not satisfy {_RELEASE_SCHEMA}") from exc


def select_worker_image(
    release: WorkerRelease,
    compute_capability: str,
) -> ResolvedWorkerImage:
    """Resolve only an explicitly declared compute capability; never guess/fallback."""
    try:
        validated = load_worker_release(release)
    except WorkerReleaseError:
        raise
    if not isinstance(compute_capability, str) or not _COMPUTE_CAPABILITY.fullmatch(compute_capability):
        raise WorkerReleaseError("GPU compute capability must use canonical major.minor notation")

    matches = [
        variant for variant in validated.variants
        if compute_capability in variant.compute_capabilities
    ]
    if not matches:
        raise WorkerReleaseError(
            f"Worker release {validated.release_id!r} does not declare GPU compute capability "
            f"{compute_capability}"
        )
    if len(matches) != 1:
        raise WorkerReleaseError(
            f"Worker release {validated.release_id!r} has ambiguous images for compute capability {compute_capability}"
        )
    variant = matches[0]
    return ResolvedWorkerImage(
        descriptor=variant.image,
        release_identity=validated.release_identity,
        compute_capability=compute_capability,
        build_arch=variant.build_arch,
        cuda_version=variant.cuda_version,
    )


def _normalize_manifest_arrays(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Convert JSON arrays to tuples so the validated frozen catalog is deeply immutable."""
    normalized = dict(raw)
    variants = normalized.get("variants")
    if isinstance(variants, list):
        rows: list[Any] = []
        for row in variants:
            if isinstance(row, Mapping):
                row = dict(row)
                capabilities = row.get("compute_capabilities")
                if isinstance(capabilities, list):
                    row["compute_capabilities"] = tuple(capabilities)
            rows.append(row)
        normalized["variants"] = tuple(rows)
    return normalized
