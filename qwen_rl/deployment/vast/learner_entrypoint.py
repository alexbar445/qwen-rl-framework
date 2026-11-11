"""Fail-closed in-container entrypoint for a prebuilt Vast Learner image.

This validates the pinned runtime, unified immutable asset tree, credential
bindings, and visible GPU metadata before it starts the existing Learner role
manager. It stages only the pinned Learner asset manifest; it never installs
software, downloads Worker models, runs an optimizer, or starts the Teacher.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
import stat
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{36}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_CREDENTIALS = frozenset({
    "PATH", "HOME", "PYTHONPATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES", "LD_PRELOAD", "LD_LIBRARY_PATH",
    "ARC_RL_ROOT", "RL_PYTHON", "TMPDIR", "XDG_CACHE_HOME", "HF_HOME",
    "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HF_ASSETS_CACHE",
    "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
    "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH",
    "FLASHINFER_WORKSPACE_BASE",
})
_HASH_CHUNK = 8 * 1024 * 1024
_WORK_ROOT = Path("/work")


class VastLearnerEntrypointError(RuntimeError):
    """A safe-to-display deployment preflight or lifecycle failure."""

    def __init__(self, message: str, *, code: str = "preflight_failed") -> None:
        super().__init__(message)
        self.code = code


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _read_private_file(path: Path, *, label: str, maximum_bytes: int = 32 * 1024 * 1024) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise VastLearnerEntrypointError(f"{label} is unavailable", code=f"{label}_unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise VastLearnerEntrypointError(f"{label} must be a private regular file",
                                             code=f"{label}_unsafe")
        chunks = bytearray()
        while len(chunks) <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) > maximum_bytes:
            raise VastLearnerEntrypointError(f"{label} exceeds its size limit", code=f"{label}_too_large")
        return bytes(chunks)
    finally:
        os.close(descriptor)


def _strict_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise VastLearnerEntrypointError(f"{label} is invalid strict JSON", code=f"{label}_invalid") from exc


def _absolute_container_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise VastLearnerEntrypointError(f"{label} must be an absolute path", code="path_invalid")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise VastLearnerEntrypointError(f"{label} must be normalized and absolute", code="path_invalid")
    return path


def _reject_symlink_components(path: Path, *, label: str, code: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise VastLearnerEntrypointError(f"{label} must be an absolute normalized path", code=code)
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise VastLearnerEntrypointError(f"{label} path contains a symlink", code=code)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise VastLearnerEntrypointError(f"{label} path is unavailable", code=code) from exc


def _private_state_root(path: Path) -> Path:
    _reject_symlink_components(path, label="state root", code="state_root_unsafe")
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        _reject_symlink_components(path, label="state root", code="state_root_unsafe")
        info = path.lstat()
    except OSError as exc:
        raise VastLearnerEntrypointError("private state root is unavailable", code="state_root_unsafe") from exc
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise VastLearnerEntrypointError("state root must be owned by the runtime UID with mode 0700",
                                         code="state_root_unsafe")
    return path.resolve(strict=True)


def _write_private_receipt(path: Path, payload: dict[str, Any], *, state_root: Path) -> None:
    receipt = path.expanduser()
    if not receipt.is_absolute() or ".." in receipt.parts or receipt.parent.resolve(strict=True) != state_root:
        raise VastLearnerEntrypointError("receipt must be a file directly under the private state root",
                                         code="receipt_unsafe")
    if receipt.is_symlink():
        raise VastLearnerEntrypointError("receipt must not be a symlink", code="receipt_unsafe")
    temporary = state_root / f".{receipt.name}.{uuid.uuid4().hex}.tmp"
    encoded = (json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, receipt)
        directory_fd = os.open(state_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise VastLearnerEntrypointError("private startup receipt could not be written",
                                         code="receipt_write_failed") from exc


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_HASH_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _asset_target(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    current = root
    for index, part in enumerate(path.parts):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise VastLearnerEntrypointError("manifested asset is missing", code="asset_missing") from exc
        if stat.S_ISLNK(info.st_mode):
            raise VastLearnerEntrypointError("asset tree contains a symlink", code="asset_unsafe")
        if index < len(path.parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise VastLearnerEntrypointError("asset path crosses a non-directory", code="asset_unsafe")
    if not stat.S_ISREG(current.stat().st_mode):
        raise VastLearnerEntrypointError("manifested asset is not a regular file", code="asset_unsafe")
    return current


def _verify_asset_tree(asset_root: Path, manifest: Any) -> set[str]:
    _reject_symlink_components(asset_root, label="asset root", code="asset_root_invalid")
    if not asset_root.is_dir():
        raise VastLearnerEntrypointError("asset root is missing or unsafe", code="asset_root_invalid")
    root = asset_root.resolve(strict=True)
    expected = {item.path for item in manifest.files}
    observed: set[str] = set()
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(current_root)
        for name in directory_names:
            item = base / name
            info = item.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise VastLearnerEntrypointError("asset tree contains an unsafe directory entry",
                                                 code="asset_unsafe")
        for name in file_names:
            item = base / name
            info = item.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise VastLearnerEntrypointError("asset tree contains a non-regular file",
                                                 code="asset_unsafe")
            observed.add(item.relative_to(root).as_posix())
    if observed != expected:
        raise VastLearnerEntrypointError("asset tree file set differs from its pinned manifest",
                                         code="asset_inventory_mismatch")
    for item in manifest.files:
        path = _asset_target(root, item.path)
        try:
            observed_sha, observed_size = _hash_file(path)
        except OSError as exc:
            raise VastLearnerEntrypointError("manifested asset could not be hashed", code="asset_hash_failed") from exc
        if observed_size != item.size or not hmac.compare_digest(observed_sha, item.sha256):
            raise VastLearnerEntrypointError("asset size or SHA-256 differs from its pinned manifest",
                                             code="asset_integrity_failed")
    return expected


def _require_asset_path(value: str | Path, *, root: Path, files: set[str], label: str,
                        directory: bool) -> None:
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise VastLearnerEntrypointError(f"runtime {label} must be an absolute asset path",
                                         code="asset_reference_invalid")
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise VastLearnerEntrypointError(f"runtime {label} is outside the immutable asset root",
                                         code="asset_reference_invalid") from exc
    if directory:
        if not relative.parts or not path.is_dir():
            raise VastLearnerEntrypointError(f"runtime {label} must be a manifested asset directory",
                                             code="asset_reference_invalid")
        prefix = relative.as_posix().rstrip("/") + "/"
        if not any(item.startswith(prefix) for item in files):
            raise VastLearnerEntrypointError(f"runtime {label} has no files in the manifest",
                                             code="asset_reference_missing")
    elif relative.as_posix() not in files or not path.is_file():
        raise VastLearnerEntrypointError(f"runtime {label} is not listed in the asset manifest",
                                         code="asset_reference_missing")


def _require_state_path(value: str | Path | None, *, root: Path, label: str) -> None:
    if value in (None, ""):
        return
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise VastLearnerEntrypointError(f"runtime {label} must be an absolute state path",
                                         code="state_reference_invalid")
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise VastLearnerEntrypointError(f"runtime {label} is outside the private state root",
                                         code="state_reference_invalid") from exc


def _expected_credentials(config: Any) -> set[str]:
    names = {"ARC_ENV_TOKEN"}
    for worker in config.workers:
        if worker.scope == "local":
            raise VastLearnerEntrypointError("Vast Learner may not start local Worker processes",
                                             code="worker_scope_invalid")
        if worker.scope not in {"service", "gateway"}:
            raise VastLearnerEntrypointError("Vast Learner Worker scope must be service or gateway",
                                             code="worker_scope_invalid")
        names.add(worker.auth_token_env)
    if config.bepa.enabled:
        names.update({config.bepa.teacher.api_key_env, config.bepa.teacher.token_env})
    if (any(not isinstance(name, str) or _ENV_NAME.fullmatch(name) is None
            or name in _FORBIDDEN_CREDENTIALS or name.startswith(("CUDA_", "LD_", "PYTHON"))
            for name in names)):
        raise VastLearnerEntrypointError("runtime credential names contain a forbidden environment binding",
                                         code="credential_names_invalid")
    return names


def _load_credentials(credential_map_path: Path, *, expected_names: set[str]) -> dict[str, str]:
    payload = _read_private_file(credential_map_path, label="credential_map", maximum_bytes=64 * 1024)
    mapping = _strict_json(payload, label="credential_map")
    if not isinstance(mapping, dict) or set(mapping) != expected_names:
        raise VastLearnerEntrypointError("credential map names differ from the runtime allowlist",
                                         code="credential_map_mismatch")
    _reject_symlink_components(credential_map_path.parent, label="credential directory",
                               code="credential_directory_unsafe")
    secrets_directory = credential_map_path.parent.resolve(strict=True)
    if (secrets_directory.name != "secrets"
            or stat.S_IMODE(secrets_directory.stat().st_mode) != 0o700
            or secrets_directory.stat().st_uid != os.getuid()):
        raise VastLearnerEntrypointError("credential directory must be private and owned by the runtime UID",
                                         code="credential_directory_unsafe")
    from qwen_rl.deployment.learner.container import _read_secret_file

    result: dict[str, str] = {}
    for name in sorted(expected_names):
        value = mapping[name]
        if not isinstance(value, str):
            raise VastLearnerEntrypointError("credential map values must be absolute file paths",
                                             code="credential_path_invalid")
        secret_path = Path(value)
        _reject_symlink_components(secret_path.parent, label="credential path",
                                   code="credential_path_invalid")
        if (not secret_path.is_absolute() or ".." in secret_path.parts
                or secret_path.parent.resolve(strict=False) != secrets_directory
                or secret_path.name != name or secret_path.is_symlink()):
            raise VastLearnerEntrypointError("credential path does not match its private named file",
                                             code="credential_path_invalid")
        try:
            result[name] = _read_secret_file(secret_path, expected_uid=os.getuid())
        except Exception as exc:
            raise VastLearnerEntrypointError("private credential file failed owner, mode, or content checks",
                                             code="credential_invalid") from exc
    return result


def _expected_compute_capability(cuda_arch: str) -> str:
    if not isinstance(cuda_arch, str) or re.fullmatch(r"[0-9]{2,3}a?", cuda_arch) is None:
        raise VastLearnerEntrypointError("deployment CUDA architecture is invalid", code="gpu_contract_invalid")
    digits = cuda_arch.removesuffix("a")
    major, minor = (digits[0], digits[1]) if len(digits) == 2 else (digits[:-1], digits[-1])
    return f"{int(major)}.{int(minor)}"


def _nvidia_smi_rows() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VastLearnerEntrypointError("NVIDIA GPU metadata is unavailable", code="gpu_metadata_unavailable") from exc
    if result.returncode != 0:
        raise VastLearnerEntrypointError("NVIDIA GPU metadata query failed", code="gpu_metadata_unavailable")
    rows: list[dict[str, str]] = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        values = [item.strip() for item in row]
        if len(values) != 3 or _GPU_UUID.fullmatch(values[0]) is None:
            raise VastLearnerEntrypointError("NVIDIA GPU metadata has an invalid UUID row",
                                             code="gpu_metadata_invalid")
        rows.append({"uuid": values[0], "name": values[1], "compute_cap": values[2]})
    if not rows or len({item["uuid"] for item in rows}) != len(rows):
        raise VastLearnerEntrypointError("NVIDIA GPU metadata is empty or duplicated", code="gpu_metadata_invalid")
    return rows


def _logical_cuda_device_count() -> int:
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception as exc:
        raise VastLearnerEntrypointError("prebuilt PyTorch could not enumerate visible CUDA devices",
                                         code="gpu_visibility_invalid") from exc


def _selected_gpu_rows(rows: list[dict[str, str]], visible: str | None) -> list[dict[str, str]]:
    if visible is None or visible.strip().lower() == "all":
        return list(rows)
    tokens = [item.strip() for item in visible.split(",")]
    if not tokens or any(not item for item in tokens):
        raise VastLearnerEntrypointError("CUDA_VISIBLE_DEVICES selects no logical GPU", code="gpu_visibility_invalid")
    selected: list[dict[str, str]] = []
    for token in tokens:
        if token.isdecimal():
            index = int(token)
            if index >= len(rows):
                raise VastLearnerEntrypointError("CUDA_VISIBLE_DEVICES index is outside the visible GPU list",
                                                 code="gpu_visibility_invalid")
            selected.append(rows[index])
        else:
            match = next((item for item in rows if item["uuid"] == token), None)
            if match is None:
                raise VastLearnerEntrypointError("CUDA_VISIBLE_DEVICES UUID is not visible in the container",
                                                 code="gpu_visibility_invalid")
            selected.append(match)
    if len({item["uuid"] for item in selected}) != len(selected):
        raise VastLearnerEntrypointError("CUDA_VISIBLE_DEVICES repeats a physical GPU", code="gpu_visibility_invalid")
    return selected


def _verify_gpus(deployment: dict[str, Any], config: Any) -> list[dict[str, str]]:
    expected_count = deployment.get("gpu_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise VastLearnerEntrypointError("deployment GPU count is invalid", code="gpu_contract_invalid")
    logical_ids = tuple(config.learner.gpus)
    if logical_ids != tuple(range(expected_count)):
        raise VastLearnerEntrypointError("runtime logical GPU IDs must be exactly 0..gpu_count-1",
                                         code="gpu_config_mismatch")
    rows = _nvidia_smi_rows()
    selected = _selected_gpu_rows(rows, os.environ.get("CUDA_VISIBLE_DEVICES"))
    logical_count = _logical_cuda_device_count()
    if len(selected) != expected_count or logical_count != expected_count:
        raise VastLearnerEntrypointError("visible logical GPU count differs from the deployment profile",
                                         code="gpu_count_mismatch")
    expected_cap = _expected_compute_capability(deployment.get("cuda_arch"))
    expected_name = re.sub(r"[^a-z0-9]", "", str(deployment.get("gpu_name", "")).lower())
    for item in selected:
        actual_name = re.sub(r"[^a-z0-9]", "", item["name"].lower())
        if item["compute_cap"] != expected_cap or not expected_name or expected_name not in actual_name:
            raise VastLearnerEntrypointError("visible GPU architecture or model differs from the deployment profile",
                                             code="gpu_model_mismatch")
    return selected


def _validate_worker_routes(deployment: dict[str, Any], config: Any) -> None:
    routes = deployment.get("worker_routes")
    if not isinstance(routes, list) or not routes:
        raise VastLearnerEntrypointError("deployment worker routes are unavailable", code="worker_routes_invalid")
    expected = {item.get("worker_id"): item for item in routes if isinstance(item, dict)}
    if len(expected) != len(routes) or any(not isinstance(key, str) for key in expected):
        raise VastLearnerEntrypointError("deployment worker routes are invalid", code="worker_routes_invalid")
    workers = list(config.workers)
    if {worker.id for worker in workers} != set(expected):
        raise VastLearnerEntrypointError("runtime Workers differ from the prepared tunnel routes",
                                         code="worker_routes_mismatch")
    service_credentials: set[str] = set()
    for worker in workers:
        if worker.scope not in {"service", "gateway"}:
            raise VastLearnerEntrypointError("Vast Learner cannot launch local Worker processes",
                                             code="worker_scope_invalid")
        route = expected[worker.id]
        endpoint = urlsplit(worker.endpoint)
        local = urlsplit(str(route.get("local_url", "")))
        remote_port = route.get("remote_port")
        if (endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1"
                or endpoint.port is None or endpoint.path not in ("", "/")
                or local.scheme != "http" or local.hostname != "127.0.0.1"
                or local.port is None or local.path not in ("", "/")
                or local.username or local.password or local.query or local.fragment
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
                or isinstance(remote_port, bool) or not isinstance(remote_port, int)
                or endpoint.port != remote_port or worker.port != remote_port or worker.host is not None):
            raise VastLearnerEntrypointError("runtime Worker endpoint differs from its loopback tunnel route",
                                             code="worker_routes_mismatch")
        if (worker.scope == "service" and worker.url_prefix != "") or (
                worker.scope == "gateway" and worker.url_prefix != f"/v1/workers/{worker.id}"):
            raise VastLearnerEntrypointError("runtime Worker prefix differs from its external route contract",
                                             code="worker_routes_mismatch")
        if worker.auth_token_env == "ARC_ENV_TOKEN":
            raise VastLearnerEntrypointError("external Worker credentials must differ from ARC_ENV_TOKEN",
                                             code="credential_names_invalid")
        if worker.scope == "service":
            if worker.auth_token_env in service_credentials:
                raise VastLearnerEntrypointError("each service Worker must have its own credential name",
                                                 code="credential_names_invalid")
            service_credentials.add(worker.auth_token_env)


def _validate_state_paths(config: Any, state_root: Path) -> None:
    values: list[tuple[str, Any]] = [("run_dir", config.run_dir)]
    if config.learner.recovery_dir:
        values.append(("learner.recovery_dir", config.learner.recovery_dir))
    if config.bepa.state_dir:
        values.append(("bepa.state_dir", config.bepa.state_dir))
    if config.evaluation.state_dir:
        values.append(("evaluation.state_dir", config.evaluation.state_dir))
    for label, value in values:
        _require_state_path(value, root=state_root, label=label)


def _private_directory(path: Path) -> Path:
    if path.is_symlink():
        raise VastLearnerEntrypointError("cache directory contains a symlink", code="cache_path_unsafe")
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.lstat()
    except OSError as exc:
        raise VastLearnerEntrypointError("private cache directory is unavailable",
                                         code="cache_path_unsafe") from exc
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise VastLearnerEntrypointError("cache directory must be private and owned by the runtime UID",
                                         code="cache_path_unsafe")
    return path.resolve(strict=True)


def _configure_cache_environment(state_root: Path) -> None:
    cache_root = _private_directory(state_root / "cache")
    cache_environment = {
        "XDG_CACHE_HOME": "xdg",
        "HF_HOME": "huggingface",
        "HF_DATASETS_CACHE": "hf-datasets",
        "HF_HUB_CACHE": "hf-hub-cache",
        "HF_ASSETS_CACHE": "hf-assets",
        "HUGGINGFACE_HUB_CACHE": "hf-hub",
        "TRANSFORMERS_CACHE": "transformers",
        "VLLM_CACHE_ROOT": "vllm",
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "CUDA_CACHE_PATH": "cuda",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    }
    values = {name: str(_private_directory(cache_root / subdir))
              for name, subdir in cache_environment.items()}
    os.environ.update(values)


def _require_work_resource(value: str | Path, *, label: str) -> None:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path == Path("/work"):
        raise VastLearnerEntrypointError(f"runtime {label} is not a valid baked /work file",
                                         code="baked_resource_invalid")
    try:
        path.resolve(strict=True).relative_to(_WORK_ROOT.resolve(strict=True))
        _reject_symlink_components(path, label=f"runtime {label}", code="baked_resource_unsafe")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise VastLearnerEntrypointError(f"runtime {label} must be a regular baked /work file",
                                             code="baked_resource_invalid")
    except (OSError, ValueError) as exc:
        raise VastLearnerEntrypointError(f"runtime {label} must be an existing baked /work file",
                                         code="baked_resource_invalid") from exc


def _verify_baked_contracts() -> dict[str, Any]:
    expected_files = (
        "assets/arc-system-rules.txt",
        "assets/sft-tools.json",
        "toolkit-dependency.json",
    )
    file_receipts: dict[str, dict[str, Any]] = {}
    file_payloads: dict[str, bytes] = {}
    for relative in expected_files:
        path = _WORK_ROOT / relative
        _require_work_resource(path, label=relative)
        try:
            sha256, size = _hash_file(path)
            file_payloads[relative] = path.read_bytes()
        except OSError as exc:
            raise VastLearnerEntrypointError("baked Learner contract could not be read",
                                             code="baked_resource_invalid") from exc
        file_receipts[relative] = {"sha256": sha256, "size_bytes": size}

    if not file_payloads["assets/arc-system-rules.txt"].strip():
        raise VastLearnerEntrypointError(
            "baked ARC system rules are empty", code="baked_system_rules_invalid",
        )
    tools = _strict_json(file_payloads["assets/sft-tools.json"], label="baked_sft_tools")
    if not isinstance(tools, list) or not tools or any(not isinstance(item, dict) for item in tools):
        raise VastLearnerEntrypointError("baked SFT tool contract is invalid", code="baked_sft_invalid")
    toolkit = _strict_json(file_payloads["toolkit-dependency.json"], label="toolkit_dependency")
    bundle = toolkit.get("bundle") if isinstance(toolkit, dict) else None
    injection = bundle.get("injection") if isinstance(bundle, dict) else None
    system_message = injection.get("system_message") if isinstance(injection, dict) else None
    if (not isinstance(toolkit, dict) or toolkit.get("schema_version") != "arc-toolkit-dependency-v1"
            or not isinstance(bundle, dict) or not isinstance(bundle.get("catalog"), dict)
            or not isinstance(system_message, dict) or system_message.get("role") != "system"
            or not isinstance(system_message.get("content"), str) or not system_message["content"]):
        raise VastLearnerEntrypointError("baked toolkit dependency contract is invalid",
                                         code="baked_toolkit_invalid")

    toolkit_root = _WORK_ROOT / "vendor" / "paired-factory-v1" / "arc_toolkit"
    _reject_symlink_components(toolkit_root, label="baked ARC toolkit", code="baked_toolkit_invalid")
    if not toolkit_root.is_dir():
        raise VastLearnerEntrypointError("baked ARC toolkit source is unavailable",
                                         code="baked_toolkit_invalid")
    toolkit_init = toolkit_root / "__init__.py"
    _require_work_resource(toolkit_init, label="vendor/paired-factory-v1/arc_toolkit/__init__.py")
    init_sha, init_size = _hash_file(toolkit_init)
    file_receipts["vendor/paired-factory-v1/arc_toolkit/__init__.py"] = {
        "sha256": init_sha, "size_bytes": init_size,
    }
    return {
        "files": file_receipts,
        "arc_toolkit_import_root": str(toolkit_root),
    }


def _require_asset_or_work_file(value: str | Path, *, root: Path, files: set[str], label: str) -> None:
    try:
        _require_asset_path(value, root=root, files=files, label=label, directory=False)
    except VastLearnerEntrypointError as exc:
        if exc.code not in {"asset_reference_invalid", "asset_reference_missing"}:
            raise
        _require_work_resource(value, label=label)


def _validate_assets(config: Any, root: Path, files: set[str]) -> None:
    _require_asset_path(config.model, root=root, files=files, label="model_path", directory=True)
    _require_asset_path(config.dataset, root=root, files=files, label="dataset_dir", directory=True)
    _require_asset_or_work_file(config.contract, root=root, files=files, label="arc.contract_path")
    evaluation = config.evaluation
    if evaluation.enabled:
        for name in ("manifest_path", "sft_exclusion_proof", "initial_count_proof"):
            value = getattr(evaluation, name)
            if value:
                _require_asset_or_work_file(value, root=root, files=files, label=f"evaluation.{name}")


def _validate_manifest_references(config: Any, root: Path, manifest: Any) -> None:
    """Reject an incomplete config/manifest contract before any asset transfer."""
    files = {item.path for item in manifest.files}
    root_path = root.resolve(strict=False)

    def relative_asset(value: str | Path, *, label: str) -> PurePosixPath:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise VastLearnerEntrypointError(f"runtime {label} must be an absolute asset path",
                                             code="asset_reference_invalid")
        try:
            relative = path.resolve(strict=False).relative_to(root_path)
        except ValueError as exc:
            raise VastLearnerEntrypointError(f"runtime {label} is outside the immutable asset root",
                                             code="asset_reference_invalid") from exc
        if not relative.parts:
            raise VastLearnerEntrypointError(f"runtime {label} cannot name the asset root itself",
                                             code="asset_reference_invalid")
        return PurePosixPath(relative.as_posix())

    for label, value in (("model_path", config.model), ("dataset_dir", config.dataset)):
        relative = relative_asset(value, label=label)
        prefix = relative.as_posix().rstrip("/") + "/"
        if not any(item.startswith(prefix) for item in files):
            raise VastLearnerEntrypointError(f"runtime {label} has no files in the pinned manifest",
                                             code="asset_reference_missing")

    resources = [("arc.contract_path", config.contract)]
    evaluation = config.evaluation
    if evaluation.enabled:
        resources.extend((f"evaluation.{name}", getattr(evaluation, name))
                         for name in ("manifest_path", "sft_exclusion_proof", "initial_count_proof")
                         if getattr(evaluation, name))
    for label, value in resources:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise VastLearnerEntrypointError(f"runtime {label} must be an absolute resource path",
                                             code="asset_reference_invalid")
        try:
            relative = path.resolve(strict=False).relative_to(root_path)
        except ValueError:
            _require_work_resource(path, label=label)
            continue
        if relative.as_posix() not in files:
            raise VastLearnerEntrypointError(f"runtime {label} is absent from the pinned manifest",
                                             code="asset_reference_missing")


def _download_assets(manifest_path: Path, asset_root: Path, *, state_root: Path,
                     expected_sha256: str) -> dict[str, Any]:
    from qwen_rl.deployment.vast.download import DownloadError, download_manifest

    try:
        receipt = download_manifest(
            manifest_path,
            asset_root,
            receipt_path=state_root / "vast-learner-download.json",
            expected_sha256=expected_sha256,
        )
    except DownloadError as exc:
        raise VastLearnerEntrypointError("Learner asset staging failed",
                                         code=f"asset_download_{exc.code}") from exc
    if receipt.get("manifest_sha256") != expected_sha256 or receipt.get("status") != "verified":
        raise VastLearnerEntrypointError("Learner asset staging receipt does not match the pinned manifest",
                                         code="asset_download_receipt_invalid")
    return receipt


def _load_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Any, Any, list[dict[str, str]], dict[str, str]]:
    from qwen_rl.deployment.vast.config import validate_deployment_payload
    from qwen_rl.deployment.vast.download import DownloadError, parse_manifest_bytes
    from qwen_rl.runtime.config import load_config

    deployment_path = _absolute_container_path(args.deployment, label="deployment")
    runtime_path = _absolute_container_path(args.runtime_config, label="runtime config")
    manifest_path = _absolute_container_path(args.asset_manifest, label="asset manifest")
    asset_root = _absolute_container_path(args.asset_root, label="asset root")
    state_root_arg = _absolute_container_path(args.state_root, label="state root")
    credential_map_path = _absolute_container_path(args.credential_map, label="credential map")
    receipt_path = _absolute_container_path(args.receipt, label="receipt")

    try:
        deployment = validate_deployment_payload(_strict_json(
            _read_private_file(deployment_path, label="deployment"), label="deployment"))
    except VastLearnerEntrypointError:
        raise
    except Exception as exc:
        raise VastLearnerEntrypointError("Learner deployment contract is invalid",
                                         code="deployment_invalid") from exc
    if deployment.get("role") != "learner":
        raise VastLearnerEntrypointError("deployment role must be learner", code="deployment_role_invalid")
    remote_root = Path(deployment["remote_root"])
    if (deployment_path != remote_root / "config" / "deployment.json"
            or runtime_path != remote_root / "config" / "runtime.json"
            or manifest_path != remote_root / "config" / "asset-manifest.json"
            or credential_map_path != remote_root / "secrets" / "credential-map.json"):
        raise VastLearnerEntrypointError("deployment input paths differ from the prepared private layout",
                                         code="deployment_layout_invalid")
    if asset_root != Path(deployment["asset_root_container"]):
        raise VastLearnerEntrypointError("asset root differs from the prepared deployment", code="asset_root_mismatch")
    if state_root_arg != Path(deployment["state_root_container"]):
        raise VastLearnerEntrypointError("state root differs from the prepared deployment", code="state_root_mismatch")
    state_root = _private_state_root(state_root_arg)
    expected_receipt = state_root / "vast-learner-entrypoint.json"
    if receipt_path != expected_receipt:
        raise VastLearnerEntrypointError("receipt path differs from the private state contract",
                                         code="receipt_path_mismatch")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "role": "learner",
        "deployment_id": deployment["deployment_id"],
        "status": "preflight_running",
        "runtime_config_sha256": deployment["runtime_config_sha256"],
        "asset_manifest_sha256": deployment["asset_manifest_sha256"],
        "ready_asserted": False,
    }
    _write_private_receipt(receipt_path, receipt, state_root=state_root)

    runtime_bytes = _read_private_file(runtime_path, label="runtime_config")
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    if not hmac.compare_digest(runtime_sha, deployment["runtime_config_sha256"]):
        raise VastLearnerEntrypointError("runtime config differs from its pinned digest",
                                         code="runtime_config_digest_mismatch")
    try:
        manifest_bytes = _read_private_file(manifest_path, label="asset_manifest")
        manifest = parse_manifest_bytes(manifest_bytes, expected_sha256=deployment["asset_manifest_sha256"])
    except DownloadError as exc:
        raise VastLearnerEntrypointError("asset manifest differs from its pinned contract",
                                         code="asset_manifest_invalid") from exc

    try:
        config = load_config(runtime_path)
    except Exception as exc:
        raise VastLearnerEntrypointError("runtime config failed the Qwen RuntimeConfig validator",
                                         code="runtime_config_invalid") from exc
    if config.learner.host != "127.0.0.1" or config.learner.port != deployment["learner_port"]:
        raise VastLearnerEntrypointError("Learner listener must match the prepared loopback tunnel",
                                         code="learner_listener_mismatch")
    if (len(config.learner.gpus) != deployment["gpu_count"]
            or tuple(config.learner.gpus) != tuple(range(deployment["gpu_count"]))):
        raise VastLearnerEntrypointError("runtime logical GPU IDs differ from the profile GPU count",
                                         code="gpu_config_mismatch")
    _validate_worker_routes(deployment, config)

    _validate_state_paths(config, state_root)
    _reject_symlink_components(asset_root, label="asset root", code="asset_root_invalid")
    _validate_manifest_references(config, asset_root, manifest)
    baked_contracts = _verify_baked_contracts()

    expected_credentials = _expected_credentials(config)
    if set(deployment["credential_envs"]) != expected_credentials:
        raise VastLearnerEntrypointError("profile credential names differ from RuntimeConfig",
                                         code="credential_names_mismatch")
    credentials = _load_credentials(credential_map_path, expected_names=expected_credentials)
    arc_token = credentials["ARC_ENV_TOKEN"]
    for worker in config.workers:
        if worker.scope in {"service", "gateway"} and credentials[worker.auth_token_env] == arc_token:
            raise VastLearnerEntrypointError("external Worker credential must differ from ARC_ENV_TOKEN",
                                             code="credential_values_invalid")

    gpus = _verify_gpus(deployment, config)
    download_receipt = _download_assets(
        manifest_path,
        asset_root,
        state_root=state_root,
        expected_sha256=deployment["asset_manifest_sha256"],
    )
    asset_files = _verify_asset_tree(asset_root, manifest)
    _validate_assets(config, asset_root, asset_files)
    _configure_cache_environment(state_root)
    receipt.update({
        "status": "preflight_passed",
        "asset_staging": {
            "status": download_receipt["status"],
            "mode": download_receipt["mode"],
            "manifest_sha256": download_receipt["manifest_sha256"],
            "files": download_receipt["files"],
            "timings_seconds": download_receipt["timings_seconds"],
        },
        "asset_file_count": len(manifest.files),
        "asset_total_size_bytes": manifest.total_size,
        "gpu_count": len(gpus),
        "gpu_uuids": [item["uuid"] for item in gpus],
        "gpu_names": [item["name"] for item in gpus],
        "compute_capabilities": [item["compute_cap"] for item in gpus],
        "credential_env_names": sorted(expected_credentials),
        "baked_contracts": baked_contracts,
    })
    _write_private_receipt(receipt_path, receipt, state_root=state_root)
    for name, value in credentials.items():
        os.environ[name] = value
    return deployment, config, manifest, gpus, credentials


def _write_preflight_failure(args: argparse.Namespace, *, error_code: str) -> None:
    """Replace a running/missing preflight receipt only for a validated bundle layout."""
    try:
        from qwen_rl.deployment.vast.config import validate_deployment_payload

        deployment_path = _absolute_container_path(args.deployment, label="deployment")
        runtime_path = _absolute_container_path(args.runtime_config, label="runtime config")
        manifest_path = _absolute_container_path(args.asset_manifest, label="asset manifest")
        asset_root = _absolute_container_path(args.asset_root, label="asset root")
        state_root_arg = _absolute_container_path(args.state_root, label="state root")
        credential_map_path = _absolute_container_path(args.credential_map, label="credential map")
        receipt_path = _absolute_container_path(args.receipt, label="receipt")
        deployment = validate_deployment_payload(_strict_json(
            _read_private_file(deployment_path, label="deployment"), label="deployment"))
        remote_root = Path(deployment["remote_root"])
        state_root = _private_state_root(state_root_arg)
        if (deployment.get("role") != "learner"
                or deployment_path != remote_root / "config" / "deployment.json"
                or runtime_path != remote_root / "config" / "runtime.json"
                or manifest_path != remote_root / "config" / "asset-manifest.json"
                or credential_map_path != remote_root / "secrets" / "credential-map.json"
                or asset_root != Path(deployment["asset_root_container"])
                or state_root_arg != Path(deployment["state_root_container"])
                or receipt_path != state_root / "vast-learner-entrypoint.json"):
            return
        _write_private_receipt(receipt_path, {
            "schema_version": 1,
            "role": "learner",
            "deployment_id": deployment["deployment_id"],
            "status": "failed",
            "ready_asserted": False,
            "error_code": error_code,
            "runtime_config_sha256": deployment["runtime_config_sha256"],
            "asset_manifest_sha256": deployment["asset_manifest_sha256"],
        }, state_root=state_root)
    except Exception:  # noqa: BLE001 - failure recording must never weaken the original error path.
        # A receipt is not worth weakening the fail-closed path and ownership checks.
        return


def _run_supervisor(config_path: Path, config: Any) -> int:
    from qwen_rl.deployment.learner.container import run_container_supervisor
    from qwen_rl.runtime import manager

    return run_container_supervisor(
        config_path,
        config,
        runtime_start=manager.start,
        runtime_status=manager.status,
        runtime_stop=manager.stop,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--credential-map", required=True)
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt_path = Path(args.receipt).expanduser()
    preflight_complete = False
    try:
        deployment, config, _manifest, _gpu_rows, credentials = _load_preflight(args)
        preflight_complete = True
        # `credentials` is intentionally referenced only to keep validation and
        # allowlisted process-environment setup in the same preflight transaction.
        del credentials
        result = _run_supervisor(Path(args.runtime_config), config)
        _write_private_receipt(receipt_path, {
            "schema_version": 1,
            "role": "learner",
            "deployment_id": deployment["deployment_id"],
            "status": "supervisor_exited",
            "ready_asserted": False,
            "runtime_config_sha256": deployment["runtime_config_sha256"],
            "asset_manifest_sha256": deployment["asset_manifest_sha256"],
            "supervisor_exit_code": int(result),
        }, state_root=Path(deployment["state_root_container"]))
        return int(result)
    except VastLearnerEntrypointError as exc:
        _write_preflight_failure(args, error_code=exc.code)
        print(f"vast-learner-entrypoint: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if not preflight_complete:
            _write_preflight_failure(args, error_code="preflight_interrupted")
            return 130
        return 0
    except Exception:  # noqa: BLE001 - keep logs safe and mark only a validated bundle failed.
        _write_preflight_failure(
            args,
            error_code="supervisor_failed" if preflight_complete else "preflight_internal_error",
        )
        print("vast-learner-entrypoint: preflight or supervisor failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
