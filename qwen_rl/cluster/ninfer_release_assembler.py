"""Assemble a strict NInfer Worker release from verified, already-built image archives.

This tool does not build Docker images or synthesize missing architecture entries.
It verifies each ImageArchiveDescriptor against the content-addressed archive and
its embedded Docker config labels, then emits the existing WorkerRelease schema.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
from typing import Any, Sequence
import zlib

from .artifacts import ArtifactIntegrityError, ArtifactNotFound, ContentAddressedStore
from .images import ImageArchiveDescriptor
from .settings import load_settings_json
from .worker_release import WorkerRelease, WorkerReleaseError, load_worker_release


_RELEASE_SCHEMA = "qwen-rl-worker-release/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CONVERTER_REVISION = re.compile(r"^[0-9a-f]{40}(?:-dirty)?$")
_LABEL_ENGINE_REVISION = "com.ninfer.engine.revision"
_LABEL_ENGINE_SOURCE = "com.ninfer.source.sha256"
_LABEL_ENGINE_VERSION = "com.ninfer.version"
_LABEL_CONVERTER_REVISION = "com.ninfer.converter.revision"
_LABEL_CONVERTER_SOURCE = "com.ninfer.converter.source.sha256"
_LABEL_WORKER_SOURCE = "com.qwen-rl.worker.source.sha256"
_LABEL_BUILD_ARCH = "com.qwen-rl.worker.cuda-arch"
_LABEL_CUDA_VERSION = "com.qwen-rl.worker.cuda-version"
_LABEL_OCI_REVISION = "org.opencontainers.image.revision"
_LABEL_OCI_VERSION = "org.opencontainers.image.version"
_ARCH_CONTRACT = {
    "70": ("12.9.1", ("7.0",)),
    "86": ("13.1.0", ("8.6",)),
    "89": ("13.1.0", ("8.9",)),
    "120a": ("13.1.0", ("12.0",)),
}
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024
_MAX_CONFIG_BYTES = 8 * 1024 * 1024


class NInferReleaseAssemblyError(ValueError):
    """Built images or descriptors do not satisfy the NInfer release contract."""


def _read_tar_member(tar: tarfile.TarFile, member: tarfile.TarInfo, limit: int, label: str) -> bytes:
    if not member.isfile() or member.size < 0 or member.size > limit:
        raise NInferReleaseAssemblyError(f"Docker image archive has an invalid {label} member")
    stream = tar.extractfile(member)
    if stream is None:
        raise NInferReleaseAssemblyError(f"Docker image archive cannot read its {label} member")
    with stream:
        payload = stream.read(limit + 1)
    if len(payload) != member.size or len(payload) > limit:
        raise NInferReleaseAssemblyError(f"Docker image archive {label} member size is invalid")
    return payload


def _safe_tar_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise NInferReleaseAssemblyError(f"Docker image archive has an invalid {label} path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".."} for part in path.parts):
        raise NInferReleaseAssemblyError(f"Docker image archive has an unsafe {label} path")
    return value


def _config_labels_from_archive(archive_path: Path, image_id: str) -> dict[str, str]:
    """Stream Docker save metadata in any tar order without extracting image layers."""
    config_digest = image_id.removeprefix("sha256:")
    config_names = {f"{config_digest}.json", f"blobs/sha256/{config_digest}"}
    manifest_bytes: bytes | None = None
    config_bytes: bytes | None = None
    config_member_name: str | None = None
    try:
        with gzip.open(archive_path, "rb") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|") as archive:
                for member in archive:
                    if member.name == "manifest.json":
                        if manifest_bytes is not None:
                            raise NInferReleaseAssemblyError("Docker image archive has duplicate manifest.json files")
                        manifest_bytes = _read_tar_member(
                            archive, member, _MAX_MANIFEST_BYTES, "manifest",
                        )
                        continue
                    if not member.isfile():
                        continue

                    basename = PurePosixPath(member.name).name
                    likely_config = member.name in config_names or basename.endswith(".json")
                    if not likely_config:
                        continue
                    if member.size > _MAX_CONFIG_BYTES:
                        if member.name in config_names:
                            raise NInferReleaseAssemblyError("Docker image config exceeds its size limit")
                        continue
                    payload = _read_tar_member(archive, member, _MAX_CONFIG_BYTES, "candidate config")
                    if hashlib.sha256(payload).hexdigest() == config_digest:
                        if config_bytes is not None:
                            raise NInferReleaseAssemblyError("Docker image archive has duplicate matching configs")
                        config_bytes = payload
                        config_member_name = member.name
    except NInferReleaseAssemblyError:
        raise
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError, zlib.error) as exc:
        raise NInferReleaseAssemblyError("Docker image archive is not a readable gzip tar") from exc

    if manifest_bytes is None:
        raise NInferReleaseAssemblyError("Docker image archive has no manifest.json")
    if config_bytes is None or config_member_name is None:
        raise NInferReleaseAssemblyError("Docker image archive has no config matching ImageArchiveDescriptor.image_id")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NInferReleaseAssemblyError("Docker image archive manifest is invalid JSON") from exc
    if not isinstance(manifest, list) or len(manifest) != 1 or not isinstance(manifest[0], dict):
        raise NInferReleaseAssemblyError("Docker image archive must contain exactly one image manifest")
    config_path = _safe_tar_path(manifest[0].get("Config"), "config")
    if config_path != config_member_name:
        raise NInferReleaseAssemblyError("Docker manifest Config path differs from the matching image config member")

    try:
        config = json.loads(config_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NInferReleaseAssemblyError("Docker image config is invalid JSON") from exc
    if not isinstance(config, dict) or not isinstance(config.get("config"), dict):
        raise NInferReleaseAssemblyError("Docker image config has no Config object")
    labels = config["config"].get("Labels")
    if not isinstance(labels, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in labels.items()):
        raise NInferReleaseAssemblyError("Docker image config labels must be string pairs")
    return labels


def _verified_labels(labels: dict[str, str], requested_arch: str) -> dict[str, str]:
    required = (
        _LABEL_ENGINE_REVISION,
        _LABEL_ENGINE_SOURCE,
        _LABEL_ENGINE_VERSION,
        _LABEL_CONVERTER_REVISION,
        _LABEL_CONVERTER_SOURCE,
        _LABEL_WORKER_SOURCE,
        _LABEL_BUILD_ARCH,
        _LABEL_CUDA_VERSION,
        _LABEL_OCI_REVISION,
        _LABEL_OCI_VERSION,
    )
    missing = [key for key in required if not labels.get(key)]
    if missing:
        raise NInferReleaseAssemblyError(f"built image is missing release labels: {', '.join(missing)}")

    engine_revision = labels[_LABEL_ENGINE_REVISION]
    engine_source = labels[_LABEL_ENGINE_SOURCE]
    engine_version = labels[_LABEL_ENGINE_VERSION]
    converter_revision = labels[_LABEL_CONVERTER_REVISION]
    converter_source = labels[_LABEL_CONVERTER_SOURCE]
    worker_source = labels[_LABEL_WORKER_SOURCE]
    build_arch = labels[_LABEL_BUILD_ARCH]
    cuda_version = labels[_LABEL_CUDA_VERSION]
    if not _GIT_REVISION.fullmatch(engine_revision):
        raise NInferReleaseAssemblyError("NInfer engine revision label must be a full lowercase git SHA")
    if not _SHA256.fullmatch(engine_source):
        raise NInferReleaseAssemblyError("NInfer source fingerprint label must be a lowercase SHA-256")
    if not _SHA256.fullmatch(worker_source):
        raise NInferReleaseAssemblyError("Qwen Worker source fingerprint label must be a lowercase SHA-256")
    if not _CONVERTER_REVISION.fullmatch(converter_revision):
        raise NInferReleaseAssemblyError("NInfer converter revision label must be a Git SHA with an optional -dirty suffix")
    if not _SHA256.fullmatch(converter_source):
        raise NInferReleaseAssemblyError("NInfer converter source fingerprint label must be a lowercase SHA-256")
    if any(char in engine_version for char in "\r\n\x00"):
        raise NInferReleaseAssemblyError("NInfer version label must be a single line")
    if labels[_LABEL_OCI_REVISION] != engine_revision or labels[_LABEL_OCI_VERSION] != engine_version:
        raise NInferReleaseAssemblyError("OCI revision/version labels disagree with NInfer release labels")
    if build_arch != requested_arch or build_arch not in _ARCH_CONTRACT:
        raise NInferReleaseAssemblyError("requested architecture differs from the built image label")
    expected_cuda, compute_capabilities = _ARCH_CONTRACT[build_arch]
    if cuda_version != expected_cuda:
        raise NInferReleaseAssemblyError(
            f"SM{build_arch} image must use CUDA {expected_cuda}; found {cuda_version}"
        )
    return {
        "engine_revision": engine_revision,
        "engine_source_sha256": engine_source,
        "engine_version": engine_version,
        "converter_revision": converter_revision,
        "converter_source_sha256": converter_source,
        "worker_source_sha256": worker_source,
        "build_arch": build_arch,
        "cuda_version": cuda_version,
        "compute_capabilities": list(compute_capabilities),
    }


def _descriptor_from_file(path: str | Path) -> ImageArchiveDescriptor:
    source = Path(path)
    try:
        source.lstat()
    except OSError as exc:
        raise NInferReleaseAssemblyError("image archive descriptor is unavailable") from exc
    if source.is_symlink() or not source.is_file():
        raise NInferReleaseAssemblyError("image archive descriptor must be a regular file")
    try:
        value = load_settings_json(source)
        return ImageArchiveDescriptor.model_validate(value, strict=True)
    except (OSError, ValueError, TypeError) as exc:
        raise NInferReleaseAssemblyError("image archive descriptor is invalid") from exc


def assemble_worker_release(
    store_root: str | Path,
    variants: Sequence[tuple[str, ImageArchiveDescriptor | dict[str, Any]]],
) -> WorkerRelease:
    """Create the existing WorkerRelease schema from only the archives supplied."""
    if not variants:
        raise NInferReleaseAssemblyError("at least one already-built image variant is required")
    store = ContentAddressedStore(store_root)
    seen_arches: set[str] = set()
    common_identity: tuple[str, str, str, str, str, str] | None = None
    validated_variants: list[dict[str, Any]] = []

    for requested_arch, descriptor_value in variants:
        if requested_arch not in _ARCH_CONTRACT:
            raise NInferReleaseAssemblyError(f"unsupported built architecture: {requested_arch}")
        if requested_arch in seen_arches:
            raise NInferReleaseAssemblyError(f"architecture {requested_arch} was listed more than once")
        seen_arches.add(requested_arch)
        try:
            descriptor = ImageArchiveDescriptor.from_value(descriptor_value).validate_content_url()
            archive_record = store.resolve(
                descriptor.archive_sha256,
                expected_size=descriptor.size_bytes,
            )
        except (ArtifactNotFound, ArtifactIntegrityError, TypeError, ValueError) as exc:
            raise NInferReleaseAssemblyError(
                f"SM{requested_arch} archive does not match its strict ImageArchiveDescriptor"
            ) from exc

        labels = _config_labels_from_archive(archive_record.path, descriptor.image_id)
        identity = _verified_labels(labels, requested_arch)
        identity_tuple = (
            identity["engine_revision"],
            identity["engine_source_sha256"],
            identity["engine_version"],
            identity["worker_source_sha256"],
            identity["converter_revision"],
            identity["converter_source_sha256"],
        )
        if common_identity is None:
            common_identity = identity_tuple
        elif identity_tuple != common_identity:
            raise NInferReleaseAssemblyError(
                "all image variants must share the same NInfer and Qwen Worker source identities"
            )
        validated_variants.append({
            "compute_capabilities": identity["compute_capabilities"],
            "build_arch": identity["build_arch"],
            "cuda_version": identity["cuda_version"],
            "image": descriptor.model_dump(mode="json"),
        })

    assert common_identity is not None
    (engine_revision, engine_source_sha256, engine_version, worker_source_sha256,
     converter_revision, converter_source_sha256) = common_identity
    validated_variants.sort(key=lambda item: item["build_arch"])
    identity_payload = {
        "schema": _RELEASE_SCHEMA,
        "backend": "ninfer",
        "engine_revision": engine_revision,
        "engine_source_sha256": engine_source_sha256,
        "engine_version": engine_version,
        "converter_revision": converter_revision,
        "converter_source_sha256": converter_source_sha256,
        "worker_source_sha256": worker_source_sha256,
        "variants": validated_variants,
    }
    canonical = json.dumps(
        identity_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    release_id = "ninfer-" + hashlib.sha256(canonical).hexdigest()[:20]
    manifest = {
        "schema": _RELEASE_SCHEMA,
        "release_id": release_id,
        "worker_source_sha256": worker_source_sha256,
        "engine_revision": engine_revision,
        "backend": "ninfer",
        "variants": validated_variants,
    }
    try:
        return load_worker_release(manifest)
    except WorkerReleaseError as exc:
        raise NInferReleaseAssemblyError("assembled release failed the canonical WorkerRelease schema") from exc


def write_release_once(release: WorkerRelease, output_path: str | Path) -> Path:
    """Write a validated release without replacing an existing manifest."""
    output = Path(output_path).expanduser()
    if output.name in {"", ".", ".."}:
        raise NInferReleaseAssemblyError("release output must name a file")
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise NInferReleaseAssemblyError("release output parent must already exist") from exc
    if not parent.is_dir():
        raise NInferReleaseAssemblyError("release output parent is not a directory")
    target = parent / output.name
    if target.exists() or target.is_symlink():
        raise NInferReleaseAssemblyError("release output already exists; choose a new immutable path")

    payload = json.dumps(
        release.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()
        dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qwen_rl.cluster.ninfer_release_assembler",
        description="Assemble an immutable NInfer WorkerRelease from existing verified Docker image archives.",
    )
    parser.add_argument("--store", type=Path, required=True,
                        help="content-addressed image archive store used by cluster publish-image")
    parser.add_argument("--output", type=Path, required=True,
                        help="new path for the strict qwen-rl-worker-release/v1 JSON")
    parser.add_argument(
        "--variant",
        action="append",
        nargs=2,
        metavar=("BUILD_ARCH", "DESCRIPTOR_JSON"),
        required=True,
        help="one already-built image variant and its publish-image descriptor; repeat for built images",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        variants: list[tuple[str, ImageArchiveDescriptor]] = []
        for build_arch, descriptor_path in args.variant:
            variants.append((build_arch, _descriptor_from_file(descriptor_path)))
        release = assemble_worker_release(args.store, variants)
        output = write_release_once(release, args.output)
    except (NInferReleaseAssemblyError, OSError, ValueError) as exc:
        print(f"ninfer-release-assembler: {exc}", file=sys.stderr)
        return 2

    summary = {
        "release_id": release.release_id,
        "backend": release.backend,
        "engine_revision": release.engine_revision,
        "worker_source_sha256": release.worker_source_sha256,
        "built_variants": [
            {
                "build_arch": variant.build_arch,
                "compute_capabilities": list(variant.compute_capabilities),
                "cuda_version": variant.cuda_version,
                "image_id": variant.image.image_id,
                "archive_sha256": variant.image.archive_sha256,
            }
            for variant in release.variants
        ],
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
