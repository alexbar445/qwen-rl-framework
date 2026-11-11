"""Immutable adapter publication shared by independent compute engines."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError

from qwen_rl.learner.checkpoints import ADAPTER_FILES

from .adapter_namespace import prepare_worker_adapter
from .policy_maintenance import remove_conflicting_publication_directory as _remove_conflicting_publication_directory

_CANONICAL_VERSION_DIRECTORY = re.compile(r"version-([0-9]{6,})\Z")
_NATIVE_POLICY_FILENAME = "policy.lora.ninfer"
_NATIVE_DESCRIPTOR_FILENAME = "policy.lora.ninfer.json"
_NATIVE_DESCRIPTOR_SCHEMA_V1 = "qwen-rl-native-policy/v1"
_NATIVE_DESCRIPTOR_SCHEMA_V2 = "qwen-rl-native-policy/v2"
_NATIVE_DESCRIPTOR_FIELDS = frozenset({
    "schema", "version", "source_adapter_sha256", "adapter_config_sha256",
    "base_ninfer_artifact_id", "base_ninfer_sha256", "rank", "alpha",
    "input_basis", "base_weight_profile", "artifact_sha256", "artifact_id",
    "sites", "timings", "artifact_sha256_verified",
})
_NATIVE_SITE_FIELDS_V1 = frozenset({"site", "a_object", "b_object", "a_shape", "b_shape"})
_NATIVE_SITE_FIELDS_V2 = _NATIVE_SITE_FIELDS_V1 | {"input_basis"}
_NATIVE_TIMING_FIELDS = frozenset({
    "mapping_seconds", "write_seconds", "hash_seconds", "export_seconds"
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOGGER = logging.getLogger(__name__)




def _canonical_version_directory(name: str) -> int | None:
    match = _CANONICAL_VERSION_DIRECTORY.fullmatch(name)
    if match is None:
        return None
    try:
        version = int(match.group(1))
    except ValueError:
        return None
    return version if name == f"version-{version:06d}" else None


def _validate_native_publication_directory(
    path: Path,
    *,
    expected_version: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise RuntimeError(f"cannot inspect native publication directory: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise RuntimeError(f"native publication target is not a real directory: {path}")
    try:
        entries = list(path.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot inspect native publication directory: {path}") from exc
    expected_files = {_NATIVE_POLICY_FILENAME, _NATIVE_DESCRIPTOR_FILENAME}
    if {entry.name for entry in entries} != expected_files:
        raise RuntimeError(f"native publication has an unexpected file set: {path}")
    for entry in entries:
        try:
            entry_mode = entry.lstat().st_mode
        except OSError as exc:
            raise RuntimeError(f"cannot inspect native publication file: {entry}") from exc
        if not stat.S_ISREG(entry_mode):
            raise RuntimeError(f"native publication file is not a safe regular file: {entry}")

    descriptor_path = path / _NATIVE_DESCRIPTOR_FILENAME
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read native publication descriptor: {descriptor_path}") from exc
    fields = set(descriptor) if isinstance(descriptor, dict) else set()
    required_fields = _NATIVE_DESCRIPTOR_FIELDS - {"artifact_sha256_verified"}
    if not required_fields <= fields <= _NATIVE_DESCRIPTOR_FIELDS:
        raise RuntimeError(f"native publication descriptor has an unexpected key set: {descriptor_path}")
    schema = descriptor.get("schema")
    if schema not in (_NATIVE_DESCRIPTOR_SCHEMA_V1, _NATIVE_DESCRIPTOR_SCHEMA_V2):
        raise RuntimeError(f"native publication descriptor has an unsupported schema: {descriptor_path}")
    if schema == _NATIVE_DESCRIPTOR_SCHEMA_V1:
        expected_basis = "prism_rotated"
        expected_profile = "t2_g128_fp16_rotated"
        site_fields = _NATIVE_SITE_FIELDS_V1
    else:
        expected_basis = "per_use"
        expected_profile = "t2_g128_fp16_per_use"
        site_fields = _NATIVE_SITE_FIELDS_V2
    if descriptor.get("input_basis") != expected_basis:
        raise RuntimeError(
            f"native publication descriptor basis does not match {schema}: {descriptor_path}"
        )
    if descriptor.get("base_weight_profile") != expected_profile:
        raise RuntimeError(
            f"native publication descriptor weight profile does not match {schema}: {descriptor_path}"
        )
    version = descriptor.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise RuntimeError(f"native publication descriptor has an invalid version: {descriptor_path}")
    if expected_version is not None and version != expected_version:
        raise RuntimeError(f"native publication directory/version mismatch: {path}")
    directory_version = _canonical_version_directory(path.name)
    if directory_version is not None and directory_version != version:
        raise RuntimeError(f"native publication directory/version mismatch: {path}")

    for key in ("source_adapter_sha256", "base_ninfer_sha256"):
        value = descriptor.get(key)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise RuntimeError(f"native publication descriptor has an invalid {key}: {descriptor_path}")
    for key in ("adapter_config_sha256", "artifact_sha256"):
        value = descriptor.get(key)
        if value is not None and (not isinstance(value, str) or _SHA256.fullmatch(value) is None):
            raise RuntimeError(f"native publication descriptor has an invalid {key}: {descriptor_path}")
    if ("artifact_sha256_verified" in descriptor
            and descriptor["artifact_sha256_verified"] is not False):
        raise RuntimeError(f"native publication descriptor claims an unperformed artifact SHA check: {descriptor_path}")
    for key in ("rank", "alpha"):
        value = descriptor.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError(f"native publication descriptor has an invalid {key}: {descriptor_path}")
    for key in ("base_ninfer_artifact_id", "artifact_id"):
        value = descriptor.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"native publication descriptor has an invalid {key}: {descriptor_path}")

    sites = descriptor.get("sites")
    if not isinstance(sites, list) or any(
        not isinstance(site, dict) or set(site) != site_fields for site in sites
    ):
        raise RuntimeError(f"native publication descriptor has invalid site records: {descriptor_path}")
    if schema == _NATIVE_DESCRIPTOR_SCHEMA_V2 and any(
        site.get("input_basis") not in ("prism_rotated", "primal") for site in sites
    ):
        raise RuntimeError(
            f"native v2 publication site has a missing or unsupported input basis: {descriptor_path}"
        )
    timings = descriptor.get("timings")
    if not isinstance(timings, dict) or set(timings) != _NATIVE_TIMING_FIELDS:
        raise RuntimeError(f"native publication descriptor has invalid timings: {descriptor_path}")
    for key, value in timings.items():
        if key == "hash_seconds" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise RuntimeError(f"native publication descriptor has invalid timings: {descriptor_path}")

    policy_path = path / _NATIVE_POLICY_FILENAME
    files = [
        {"name": _NATIVE_POLICY_FILENAME, "sha256": None, "size": policy_path.stat().st_size},
        {
            "name": _NATIVE_DESCRIPTOR_FILENAME,
            "sha256": None,
            "size": descriptor_path.stat().st_size,
        },
    ]
    return descriptor, files


def _native_source_identity(
    *,
    version: int,
    adapter_sha256: str,
    adapter_config_sha256: str | None,
    base_identity: dict[str, Any],
) -> dict[str, Any]:
    artifact = base_identity["source"]["ninfer_artifact"]
    adapter = base_identity["learner"]["adapter"]
    representation = base_identity["representation"]
    input_basis = adapter["input_basis"]
    if input_basis == "prism_rotated":
        base_weight_profile = f"{representation['format']}_rotated"
    elif input_basis == "per_use":
        base_weight_profile = "t2_g128_fp16_per_use"
    else:
        raise RuntimeError(f"native publication has an unsupported adapter input basis: {input_basis!r}")
    return {
        "version": version,
        "source_adapter_sha256": adapter_sha256,
        "adapter_config_sha256": adapter_config_sha256,
        "base_ninfer_artifact_id": artifact["artifact_id"],
        "base_ninfer_sha256": artifact["sha256"],
        "rank": adapter["rank"],
        "alpha": adapter["alpha"],
        "input_basis": input_basis,
        "base_weight_profile": base_weight_profile,
    }


def _native_state_from_model(
    model: Any,
    parameter_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    from peft import get_peft_model_state_dict

    active = getattr(model, "active_adapters", ["default"])
    adapter_name = active[0] if isinstance(active, (list, tuple)) else str(active)
    if adapter_name != "default":
        raise RuntimeError(f"native T2 publication requires the default adapter, got {adapter_name!r}")

    source_state = parameter_state
    if source_state is None:
        source_state = {
            name: parameter.detach().to(device="cpu")
            for name, parameter in model.named_parameters()
            if "lora_" in name
        }
    if not source_state:
        raise RuntimeError("native T2 publication has no live LoRA parameters")
    state = get_peft_model_state_dict(
        model,
        state_dict=source_state,
        save_embedding_layers=False,
        adapter_name=adapter_name,
    )
    if not isinstance(state, dict) or not state:
        raise RuntimeError("PEFT produced no canonical native T2 state")
    for name, tensor in state.items():
        if ".default." in name:
            raise RuntimeError(f"PEFT native state retained an adapter-name segment: {name}")
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise RuntimeError(f"PEFT native state is not a CPU tensor: {name}")
    return state


def _validate_uncommitted_adapter_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise RuntimeError(f"cannot inspect uncommitted adapter directory: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise RuntimeError(f"uncommitted adapter target is not a real directory: {path}")
    try:
        entries = list(path.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot inspect uncommitted adapter directory: {path}") from exc
    if len(entries) != len(ADAPTER_FILES) or {entry.name for entry in entries} != set(ADAPTER_FILES):
        raise RuntimeError(f"uncommitted adapter directory has an unexpected file set: {path}")
    for entry in entries:
        try:
            entry_mode = entry.lstat().st_mode
        except OSError as exc:
            raise RuntimeError(f"cannot inspect uncommitted adapter file: {entry}") from exc
        if not stat.S_ISREG(entry_mode):
            raise RuntimeError(f"uncommitted adapter file is not a safe regular file: {entry}")


def tensor_digest(named_parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        tensor = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()


class AdapterPublicationStore:
    """Own published files, hashes, and in-flight policy references."""

    def __init__(self, publication_dir: str | Path, checkpoint_dir: str | Path):
        self.publication_dir = Path(publication_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.published_versions: dict[int, tuple[str, dict[str, str | None]]] = {}
        self.native_published_versions: dict[int, tuple[dict[str, Any], dict[str, str | None]]] = {}
        self._published_file_sizes: dict[int, dict[str, int]] = {}
        self.native_publications_enabled = False
        self.policy_references: dict[int, int] = {}
        self.lock = threading.RLock()

    def hashes_for(self, version: int) -> dict[str, str | None] | None:
        with self.lock:
            entry = self.published_versions.get(version)
            return None if entry is None else dict(entry[1])

    def retained_versions(self) -> tuple[int, ...]:
        with self.lock:
            return tuple(sorted(self.published_versions))

    def acquire(self, version: int) -> None:
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("policy version must be a non-negative integer")
        with self.lock:
            expected = self.published_versions.get(version)
            if expected is None:
                raise ValueError(f"policy version {version} has not been published or was already released")
            version_dir = self.publication_dir / f"version-{version:06d}"
            if version_dir.is_symlink() or not version_dir.is_dir():
                raise RuntimeError(f"published policy directory is missing or unsafe: {version_dir}")
            entries = list(version_dir.iterdir())
            if {entry.name for entry in entries} != set(ADAPTER_FILES) or any(
                entry.is_symlink() or not entry.is_file() for entry in entries
            ):
                raise RuntimeError(f"published policy directory is incomplete or unsafe: {version_dir}")
            self.policy_references[version] = self.policy_references.get(version, 0) + 1

    def release(self, version: int, *, current_version: int) -> None:
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("policy version must be a non-negative integer")
        with self.lock:
            references = self.policy_references.get(version, 0)
            if references < 1:
                raise ValueError(f"policy version {version} has no outstanding reference to release")
            if references == 1:
                del self.policy_references[version]
            else:
                self.policy_references[version] = references - 1
            self._prune_locked(current_version)

    def prune(self, current_version: int) -> None:
        with self.lock:
            self._prune_locked(current_version)

    def _prune_locked(self, current_version: int) -> None:
        root = self.publication_dir
        if root.is_symlink():
            raise RuntimeError(f"refusing to prune symlink publication root: {root}")
        if not root.exists():
            return
        if not root.is_dir():
            raise RuntimeError(f"expected a real publication directory: {root}")

        keep = set(self.policy_references)
        keep.add(current_version)
        if current_version > 0:
            # The serving policy may lag the Learner by one version until all
            # Workers ACK the next handoff. Keep that policy acquirable during
            # the handoff gap; explicit references protect older in-flight work.
            keep.add(current_version - 1)
        for entry in root.iterdir():
            name = entry.name
            prefix = "version-"
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            if not suffix.isascii() or not suffix.isdecimal():
                continue
            version = int(suffix)
            if name != f"version-{version:06d}" or version >= current_version or version in keep:
                continue
            if entry.is_symlink() or not entry.is_dir():
                continue
            shutil.rmtree(entry)

        for version in list(self.published_versions):
            if version < current_version and version not in keep:
                del self.published_versions[version]
                self._published_file_sizes.pop(version, None)
        if self.native_publications_enabled:
            self._prune_native_locked(current_version, keep)

    def publish(
        self,
        model: Any,
        evidence: dict[str, Any],
        version: int,
        *,
        model_backend: str | None = None,
        artifact_identity: dict[str, Any] | None = None,
        parameter_state: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            return self._publish_locked(
                model,
                evidence,
                version,
                model_backend=model_backend,
                artifact_identity=artifact_identity,
                parameter_state=parameter_state,
            )

    def _publish_locked(
        self,
        model: Any,
        evidence: dict[str, Any],
        version: int,
        *,
        model_backend: str | None = None,
        artifact_identity: dict[str, Any] | None = None,
        parameter_state: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        backend = evidence.get("model_backend") if model_backend is None else model_backend
        if backend == "ternary":
            if not isinstance(artifact_identity, dict):
                raise RuntimeError("native T2 publication requires the full artifact identity document")
            from qwen_rl.contracts.ternary import validate_artifact_identity

            artifact_identity = validate_artifact_identity(artifact_identity)
            self.native_publications_enabled = True
        version_dir = self.publication_dir / f"version-{version:06d}"
        if self.publication_dir.is_symlink():
            raise RuntimeError(f"refusing to publish through symlink directory: {self.publication_dir}")
        self.publication_dir.mkdir(parents=True, exist_ok=True)
        if self.publication_dir.is_symlink() or not self.publication_dir.is_dir():
            raise RuntimeError(f"expected a real publication directory: {self.publication_dir}")

        if parameter_state is None:
            source_state = {
                name: parameter.detach()
                for name, parameter in model.named_parameters()
                if "lora_" in name
            }
        else:
            source_state = parameter_state
            expected_names = {
                name for name, parameter in model.named_parameters() if "lora_" in name
            }
            if set(source_state) != expected_names:
                raise RuntimeError("publication snapshot LoRA parameter names differ from the model")
            for name, parameter in model.named_parameters():
                if "lora_" not in name:
                    continue
                value = source_state[name]
                if (not isinstance(value, torch.Tensor)
                        or tuple(value.shape) != tuple(parameter.shape)
                        or value.dtype != parameter.dtype):
                    raise RuntimeError(f"publication snapshot tensor layout differs: {name}")
        state_digest = tensor_digest(source_state.items())
        previous_hashes: dict[str, str | None] = {}
        if version in self.published_versions:
            previous_state, recorded_hashes = self.published_versions[version]
            if state_digest != previous_state:
                raise RuntimeError(f"adapter weights changed after version {version} was published")
            if version_dir.is_symlink() or not version_dir.is_dir():
                raise RuntimeError(f"published adapter directory disappeared: {version_dir}")
            entries = list(version_dir.iterdir())
            if {entry.name for entry in entries} != set(ADAPTER_FILES) or any(
                entry.is_symlink() or not entry.is_file() for entry in entries
            ):
                raise RuntimeError(f"published adapter directory is incomplete or unsafe: {version_dir}")
            if set(recorded_hashes) == set(ADAPTER_FILES):
                previous_hashes = dict(recorded_hashes)
        elif version_dir.exists() or version_dir.is_symlink():
            _remove_conflicting_publication_directory(
                version_dir,
                version=version,
                kind="adapter",
            )

        # Historical file digests may identify restored versions, but new policy
        # identity comes from the live LoRA tensor digest. No adapter file is scanned.
        hashes = previous_hashes or {name: None for name in ADAPTER_FILES}
        adapter_hash = hashes.get("adapter_model.safetensors")
        historical_identity = (
            isinstance(adapter_hash, str) and _SHA256.fullmatch(adapter_hash) is not None
        )
        if not historical_identity:
            adapter_hash = state_digest
        timings = {}

        def write_standard():
            started = perf_counter()
            try:
                return self._write_standard_policy(
                    model, evidence, version, version_dir, backend, parameter_state)
            finally:
                timings['standard_save_seconds'] = perf_counter() - started

        def write_native():
            started = perf_counter()
            try:
                return self._publish_native_policy(
                    model, version=version, adapter_sha256=adapter_hash,
                    adapter_config_sha256=hashes.get('adapter_config.json'),
                    base_identity=artifact_identity, parameter_state=parameter_state)
            finally:
                timings['native_export_validate_seconds'] = perf_counter() - started

        # Only the central engine's independent CPU snapshot can be read by two
        # writers. A live model (parameter_state=None) retains serial ownership.
        if (backend == 'ternary' and parameter_state is not None
                and all(t.device.type == 'cpu' for t in parameter_state.values())):
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='native-publication') as pool:
                native_future = pool.submit(write_native)
                files = write_standard()
                native = native_future.result()
            # The scoped pool joins its writer even when either format fails.
        else:
            files = write_standard()
            native = write_native() if backend == 'ternary' else None
        self._published_file_sizes[version] = {item["name"]: item["size"] for item in files}
        self.published_versions[version] = (state_digest, hashes)
        evidence['adapter_hash'] = adapter_hash
        evidence['adapter_hash_kind'] = (
            'historical_declared_file_sha256' if historical_identity else 'lora_tensor_state_sha256')
        self._prune_locked(version)
        publication = {
            "version": version,
            "directory": str(version_dir),
            "base_fingerprint": evidence["base_fingerprint"],
            "artifact_fingerprint": evidence["artifact_fingerprint"],
            "quantization": evidence["quantization"],
            "files": files,
            "adapter_hash": adapter_hash,
            "evidence": dict(evidence),
        }
        publication['timings'] = timings
        if native is not None:
            publication["native"] = {
                **native,
                "base_identity": artifact_identity,
            }
        return publication

    def _write_standard_policy(self, model, evidence, version, version_dir, backend, parameter_state):
        if not version_dir.exists():
            temporary_dir = Path(tempfile.mkdtemp(prefix=f".{version_dir.name}.", dir=str(version_dir.parent)))
            try:
                model.save_pretrained(
                    temporary_dir,
                    safe_serialization=True,
                    save_embedding_layers=False,
                    **({"state_dict": parameter_state} if parameter_state is not None else {}),
                )
                if not (temporary_dir / "adapter_config.json").is_file() or not (
                    temporary_dir / "adapter_model.safetensors"
                ).is_file():
                    raise RuntimeError("PEFT did not create standard adapter publication files")
                namespace = (
                    None
                    if backend == "ternary"
                    else prepare_worker_adapter(temporary_dir, evidence)
                )
                if namespace is not None:
                    evidence["worker_adapter_namespace"] = namespace
                readme = temporary_dir / "README.md"
                if readme.is_file():
                    readme.unlink()
                unexpected = {path.name for path in temporary_dir.iterdir()} - set(ADAPTER_FILES)
                if unexpected:
                    raise RuntimeError(f"unexpected files in adapter publication: {sorted(unexpected)}")
                if version_dir.exists() or version_dir.is_symlink():
                    raise FileExistsError(f"immutable publication path appeared during save: {version_dir}")
                os.rename(temporary_dir, version_dir)
            finally:
                if temporary_dir.exists():
                    shutil.rmtree(temporary_dir, ignore_errors=True)

        config_path = version_dir / "adapter_config.json"
        model_path = version_dir / "adapter_model.safetensors"
        if (version_dir.is_symlink() or config_path.is_symlink() or model_path.is_symlink()
                or not config_path.is_file() or not model_path.is_file()):
            raise RuntimeError(f"PEFT publication missing standard adapter files in {version_dir}")
        directory_entries = list(version_dir.iterdir())
        directory_files = {path.name for path in directory_entries}
        if directory_files != set(ADAPTER_FILES) or any(
            path.is_symlink() or not path.is_file() for path in directory_entries
        ):
            raise RuntimeError(f"adapter directory contains non-standard files: {sorted(directory_files)}")

        files = [
            {"name": path.name, "sha256": None, "size": path.stat().st_size}
            for path in (config_path, model_path)
        ]
        sizes = {item["name"]: item["size"] for item in files}
        expected_sizes = self._published_file_sizes.get(version)
        if expected_sizes is not None and sizes != expected_sizes:
            raise FileExistsError(f"immutable publication file sizes changed: {version_dir}")
        return files

    def _publish_native_policy(
        self,
        model: Any,
        *,
        version: int,
        adapter_sha256: str,
        adapter_config_sha256: str | None,
        base_identity: dict[str, Any],
        parameter_state: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        from .native_export import export_native_policy

        source_identity = _native_source_identity(
            version=version,
            adapter_sha256=adapter_sha256,
            adapter_config_sha256=adapter_config_sha256,
            base_identity=base_identity,
        )
        root = self.publication_dir.parent / "native-policies"
        if root.is_symlink():
            raise RuntimeError(f"refusing to publish through symlink native policy root: {root}")
        if root.exists() and not root.is_dir():
            raise RuntimeError(f"expected a real native policy root: {root}")
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"expected a real native policy root: {root}")

        version_dir = root / f"version-{version:06d}"
        if version_dir.exists() or version_dir.is_symlink():
            previous = self.native_published_versions.get(version)
            if version_dir.is_symlink() or not version_dir.is_dir():
                raise RuntimeError(f"native publication path is unsafe: {version_dir}")
            try:
                descriptor, native_files = _validate_native_publication_directory(
                    version_dir, expected_version=version
                )
            except (OSError, RuntimeError):
                if previous is not None:
                    raise
                _remove_conflicting_publication_directory(
                    version_dir,
                    version=version,
                    kind="native adapter",
                )
            else:
                identity_matches = all(
                    descriptor.get(key) == value for key, value in source_identity.items()
                )
                if not identity_matches:
                    if previous is not None:
                        raise RuntimeError(
                            f"native publication changed after version {version} was published"
                        )
                    _remove_conflicting_publication_directory(
                        version_dir,
                        version=version,
                        kind="native adapter",
                    )
                else:
                    hashes = {item["name"]: item["sha256"] for item in native_files}
                    if previous is not None and (
                        previous[0] != source_identity or previous[1] != hashes
                    ):
                        raise RuntimeError(
                            f"native publication changed after version {version} was published"
                        )
                    self.native_published_versions[version] = (dict(source_identity), hashes)
                    return {
                        "directory": str(version_dir),
                        "files": native_files,
                        "artifact_id": descriptor["artifact_id"],
                    }

        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{version_dir.name}.", dir=str(root)))
        created_final = False
        try:
            output = temporary_dir / _NATIVE_POLICY_FILENAME
            written_descriptor = export_native_policy(
                _native_state_from_model(model)
                if parameter_state is None else _native_state_from_model(model, parameter_state),
                output=output,
                version=version,
                adapter_sha256=adapter_sha256,
                adapter_config_sha256=adapter_config_sha256,
                base_identity=base_identity,
                rank=source_identity["rank"],
                alpha=source_identity["alpha"],
            )
            descriptor, native_files = _validate_native_publication_directory(
                temporary_dir, expected_version=version
            )
            if written_descriptor != descriptor:
                raise RuntimeError("native exporter return value differs from its written descriptor")
            if any(descriptor.get(key) != value for key, value in source_identity.items()):
                raise RuntimeError(f"native exporter wrote a different source identity for version {version}")
            if version_dir.exists() or version_dir.is_symlink():
                raise FileExistsError(f"native publication path appeared during export: {version_dir}")
            os.rename(temporary_dir, version_dir)
            created_final = True
            if version_dir.is_symlink() or not version_dir.is_dir():
                raise RuntimeError(f"native publication path became unsafe after rename: {version_dir}")
            hashes = {item["name"]: item["sha256"] for item in native_files}
            self.native_published_versions[version] = (dict(source_identity), hashes)
            return {"directory": str(version_dir), "files": native_files, "artifact_id": descriptor["artifact_id"]}
        except Exception:
            if created_final and version_dir.exists() and not version_dir.is_symlink():
                shutil.rmtree(version_dir)
                self.native_published_versions.pop(version, None)
            raise
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)

    def _prune_native_locked(self, current_version: int, keep: set[int]) -> None:
        root = self.publication_dir.parent / "native-policies"
        if root.is_symlink():
            raise RuntimeError(f"refusing to prune symlink native policy root: {root}")
        if not root.exists():
            return
        if not root.is_dir():
            raise RuntimeError(f"expected a real native policy root: {root}")
        for entry in root.iterdir():
            version = _canonical_version_directory(entry.name)
            if version is None or version >= current_version or version in keep:
                continue
            if entry.is_symlink() or not entry.is_dir():
                continue
            try:
                _validate_native_publication_directory(entry, expected_version=version)
            except (OSError, RuntimeError):
                continue
            shutil.rmtree(entry)
            self.native_published_versions.pop(version, None)

    def adopt_restored(
        self,
        model: Any,
        evidence: dict[str, Any],
        version: int,
        expected_hashes: dict[str, str | None] | None,
    ) -> None:
        version_dir = self.publication_dir / f"version-{version:06d}"
        if not version_dir.exists() and not version_dir.is_symlink():
            return
        if version_dir.is_symlink() or not version_dir.is_dir():
            raise RuntimeError("restored adapter publication is missing or unsafe")
        entries = list(version_dir.iterdir())
        if {entry.name for entry in entries} != set(ADAPTER_FILES) or any(
            entry.is_symlink() or not entry.is_file() for entry in entries
        ):
            raise RuntimeError("restored adapter publication has an invalid file set")
        hashes = (
            {
                name: value if (
                    isinstance(value, str) and _SHA256.fullmatch(value) is not None
                ) else None
                for name, value in expected_hashes.items()
            }
            if isinstance(expected_hashes, dict) and set(expected_hashes) == set(ADAPTER_FILES)
            else {name: None for name in ADAPTER_FILES}
        )
        from peft import get_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights

        active = getattr(model, "active_adapters", ["default"])
        adapter_name = active[0] if isinstance(active, (list, tuple)) else str(active)
        try:
            saved_state = load_peft_weights(str(version_dir), device="cpu")
        except (OSError, TypeError, ValueError, SafetensorError) as exc:
            _LOGGER.warning(
                "discarding unreadable unadopted adapter publication v%s (%s)",
                version, type(exc).__name__,
            )
            _remove_conflicting_publication_directory(
                version_dir,
                version=version,
                kind="restored adapter",
            )
            return
        current_state = get_peft_model_state_dict(model, adapter_name=adapter_name)
        if set(saved_state) != set(current_state):
            _remove_conflicting_publication_directory(
                version_dir,
                version=version,
                kind="restored adapter",
            )
            return
        for name, saved in saved_state.items():
            current = current_state[name].detach().to(device="cpu")
            if saved.dtype != current.dtype or tuple(saved.shape) != tuple(current.shape) or not torch.equal(saved, current):
                _remove_conflicting_publication_directory(
                    version_dir,
                    version=version,
                    kind="restored adapter",
                )
                return
        digest = tensor_digest(
            (name, parameter) for name, parameter in model.named_parameters() if "lora_" in name
        )
        with self.lock:
            self.published_versions[version] = (digest, hashes)
        declared_identity = hashes.get("adapter_model.safetensors")
        evidence["adapter_hash"] = (
            declared_identity
            if isinstance(declared_identity, str) and _SHA256.fullmatch(declared_identity) is not None
            else digest
        )
        evidence["adapter_hash_kind"] = (
            "historical_declared_file_sha256"
            if isinstance(declared_identity, str) and _SHA256.fullmatch(declared_identity) is not None
            else "lora_tensor_state_sha256"
        )

    def discard_uncommitted_after(self, durable_version: int) -> dict[str, Any]:
        """Remove unpublished-by-recovery artifacts newer than the durable snapshot.

        The caller must first validate and adopt the durable publication. Every
        matching directory in both roots is checked before any deletion starts,
        so an unexpected or unsafe candidate leaves all candidates untouched.
        """
        if isinstance(durable_version, bool) or not isinstance(durable_version, int) or durable_version < 0:
            raise ValueError("durable policy version must be a non-negative integer")

        with self.lock:
            active_references = {
                version: count for version, count in self.policy_references.items() if count != 0
            }
            if active_references:
                raise RuntimeError(
                    "cannot discard uncommitted publications while policy references are outstanding: "
                    f"{sorted(active_references.items())}"
                )

            candidates: list[tuple[int, Path]] = []
            native_candidates: list[tuple[int, Path]] = []
            roots = (self.publication_dir, self.checkpoint_dir)
            for root in roots:
                try:
                    root_mode = root.lstat().st_mode
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise RuntimeError(f"cannot inspect publication cleanup root: {root}") from exc
                if not stat.S_ISDIR(root_mode):
                    raise RuntimeError(f"publication cleanup root is not a real directory: {root}")
                try:
                    entries = list(root.iterdir())
                except OSError as exc:
                    raise RuntimeError(f"cannot inspect publication cleanup root: {root}") from exc
                for entry in entries:
                    version = _canonical_version_directory(entry.name)
                    if version is not None and version > durable_version:
                        candidates.append((version, entry))

            if self.native_publications_enabled:
                native_root = self.publication_dir.parent / "native-policies"
                try:
                    native_root_mode = native_root.lstat().st_mode
                except FileNotFoundError:
                    native_root_mode = None
                except OSError as exc:
                    raise RuntimeError(
                        f"cannot inspect native publication cleanup root: {native_root}"
                    ) from exc
                if native_root_mode is not None:
                    if not stat.S_ISDIR(native_root_mode):
                        raise RuntimeError(
                            f"native publication cleanup root is not a real directory: {native_root}"
                        )
                    try:
                        native_entries = list(native_root.iterdir())
                    except OSError as exc:
                        raise RuntimeError(
                            f"cannot inspect native publication cleanup root: {native_root}"
                        ) from exc
                    for entry in native_entries:
                        version = _canonical_version_directory(entry.name)
                        if version is not None and version > durable_version:
                            native_candidates.append((version, entry))

            # Validate the complete deletion set before removing anything.
            candidates.sort(key=lambda item: (item[0], str(item[1])))
            native_candidates.sort(key=lambda item: (item[0], str(item[1])))
            for _, candidate in candidates:
                _validate_uncommitted_adapter_directory(candidate)
            for version, candidate in native_candidates:
                _validate_native_publication_directory(candidate, expected_version=version)

            # Recheck the complete set after discovery, still before deleting
            # anything, in case a candidate changed during validation.
            for _, candidate in candidates:
                _validate_uncommitted_adapter_directory(candidate)
            for version, candidate in native_candidates:
                _validate_native_publication_directory(candidate, expected_version=version)

            removed_paths: list[str] = []
            removed_versions = {version for version, _ in candidates}
            removed_versions.update(version for version, _ in native_candidates)
            for version, candidate in candidates:
                shutil.rmtree(candidate)
                removed_paths.append(str(candidate))
            for version, candidate in native_candidates:
                shutil.rmtree(candidate)
                removed_paths.append(str(candidate))
                self.native_published_versions.pop(version, None)

            removed_versions.update(
                version for version in self.published_versions if version > durable_version
            )
            for version in list(self.published_versions):
                if version > durable_version:
                    del self.published_versions[version]
                    self._published_file_sizes.pop(version, None)

            return {
                "durable_version": durable_version,
                "discarded_versions": sorted(removed_versions),
                "discarded_paths": removed_paths,
            }
