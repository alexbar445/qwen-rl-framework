"""Gate policy manifests on immutable Dragonfly pre-seeding at the site Gateway."""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from qwen_rl.contracts import worker as wire
from qwen_rl.transport.tensor_codec import MAX_TENSOR_RAW_BYTES, max_encoded_size

from .artifacts import (
    ArtifactIntegrityError,
    ArtifactMirror,
    _append_jsonl,
    _inspect_tensor_file,
    _is_private_or_local,
    policy_file_cache_key,
)
from .dragonfly import DragonflyDownloader

_MAX_POLICY_JSON_BYTES = 16 * 1024 * 1024


class PolicyPreseedError(RuntimeError):
    """The current policy is not fully verified in the site's Dragonfly seed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PolicyPreseeder:
    """Ensure current policy objects reach the seed before Gateway policy ACK.

    The mirror and seed both use the Worker policy object route. Manifest file
    sizes stay in raw-file bytes; Dragonfly validates the transferred wire size.
    Tensor objects use the explicit LORABLS2 cache key and frame contract.
    """

    def __init__(
        self,
        mirror: ArtifactMirror,
        *,
        artifact_seed_origin: str,
        site_id: str,
        state_dir: str | Path,
        token: str,
        downloader: DragonflyDownloader | None = None,
    ) -> None:
        parts = urlsplit(artifact_seed_origin)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or not _is_private_or_local(parts.hostname) or parts.username is not None
                or parts.password is not None or parts.query or parts.fragment
                or parts.path not in ("", "/")):
            raise ValueError("artifact_seed_origin must be a private/local HTTP(S) origin")
        if not site_id or any(char.isspace() or char in "\r\n\x00" for char in site_id):
            raise ValueError("site_id must be a nonempty single-line identity")
        lineage_id = getattr(mirror, "lineage_id", None)
        if not isinstance(lineage_id, str) or not lineage_id \
                or any(char.isspace() or char in "\r\n\x00" for char in lineage_id):
            raise ValueError("ArtifactMirror must expose its immutable lineage_id")
        state_path = Path(state_dir)
        if downloader is None:
            downloader = DragonflyDownloader(
                "/usr/local/bin/dfget",
                state_path / "artifact-cache",
                allowed_hosts={parts.hostname},
                site_id=site_id,
                site_role="seed",
            )
        if downloader.site_id != site_id or downloader.site_role != "seed":
            raise ValueError("PolicyPreseeder requires the matching site seed downloader")
        if not isinstance(token, str) or not token \
                or any(char.isspace() or char in "\r\n\x00" for char in token):
            raise ValueError("Gateway service bearer must be one nonempty token")
        self.mirror = mirror
        self.downloader = downloader
        self.artifact_seed_origin = artifact_seed_origin.rstrip("/")
        self.site_id = site_id
        self.lineage_id = lineage_id
        self.state_dir = state_path
        self._headers = {"Authorization": f"Bearer {token}"}
        self._receipt_lock = threading.Lock()
        self._locks_guard = asyncio.Lock()
        self._locks: dict[tuple[int, str, str], asyncio.Lock] = {}
        self._seeded: set[tuple[int, str, str]] = set()

    async def _lock_for(self, key: tuple[int, str, str]) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(key, asyncio.Lock())

    def _record(self, record: dict[str, Any]) -> None:
        _append_jsonl(self.state_dir / "policy-preseed-receipts.jsonl", record, self._receipt_lock)

    @staticmethod
    def _validate_manifest(manifest: Any) -> tuple[wire.PolicyManifest, dict[str, Any]]:
        try:
            parsed = wire.PolicyManifest.model_validate(manifest)
        except Exception as exc:
            raise PolicyPreseedError("policy hook received a malformed Worker manifest") from exc
        expected_files = (
            wire.NATIVE_ADAPTER_FILES
            if parsed.quantization.get("mode") == "ternary"
            else wire.ADAPTER_FILES
        )
        names = [item.name for item in parsed.files]
        if set(names) != expected_files or len(names) != len(expected_files):
            raise PolicyPreseedError("policy hook received an incomplete or invalid adapter file set")
        for item in parsed.files:
            if item.url != f"/policy/{parsed.version}/{item.name}":
                raise PolicyPreseedError("policy hook requires the exact versioned Worker file route")
        return parsed, parsed.model_dump(mode="json")

    async def _wire_size(
        self,
        path: Path,
        item: wire.PolicyFile,
    ) -> int:
        if path.is_symlink() or not path.is_file():
            raise PolicyPreseedError("policy mirror did not return a regular cached file")
        try:
            info = await asyncio.to_thread(path.stat)
        except OSError as exc:
            raise PolicyPreseedError("policy mirror file is unavailable") from exc
        if item.name.endswith(".json"):
            raw_limit = _MAX_POLICY_JSON_BYTES if item.size_bytes is None else item.size_bytes
            if raw_limit > _MAX_POLICY_JSON_BYTES or info.st_size > raw_limit:
                raise PolicyPreseedError("policy metadata exceeds the transfer limit")
            if item.size_bytes is not None and info.st_size != item.size_bytes:
                raise PolicyPreseedError("policy metadata raw size differs from its manifest")
            return info.st_size

        raw_limit = MAX_TENSOR_RAW_BYTES if item.size_bytes is None else item.size_bytes
        if raw_limit > MAX_TENSOR_RAW_BYTES:
            raise PolicyPreseedError("policy tensor exceeds the raw transfer limit")
        wire_limit = max_encoded_size(raw_limit)
        if not 0 < info.st_size <= wire_limit:
            raise PolicyPreseedError("policy tensor wire size exceeds its bounded frame limit")
        try:
            observed_raw_size = await asyncio.to_thread(
                _inspect_tensor_file,
                path,
                max_raw_bytes=raw_limit,
                safetensors=item.name == "adapter_model.safetensors",
            )
        except (ArtifactIntegrityError, OSError, TypeError, ValueError, BufferError) as exc:
            raise PolicyPreseedError("policy tensor frame is invalid") from exc
        if item.size_bytes is not None and observed_raw_size != item.size_bytes:
            raise PolicyPreseedError("policy tensor raw size differs from its manifest")
        return info.st_size

    async def __call__(self, manifest: dict[str, Any]) -> None:
        parsed_manifest, normalized = self._validate_manifest(manifest)
        version = parsed_manifest.version
        try:
            pinned = await self.mirror.policy(version)
        except Exception as exc:
            raise PolicyPreseedError("policy version is unavailable from the configured origin") from exc
        if pinned != normalized:
            raise PolicyPreseedError("policy manifest differs from its immutable lineage/version record")

        for item in parsed_manifest.files:
            name = item.name
            try:
                object_key = policy_file_cache_key(
                    normalized,
                    name,
                    origin=self.artifact_seed_origin,
                )
            except (TypeError, ValueError) as exc:
                raise PolicyPreseedError("policy object metadata is invalid") from exc
            key = (version, name, object_key)
            lock = await self._lock_for(key)
            async with lock:
                if key in self._seeded:
                    continue
                started = _now()
                outcome = "failed"
                error_name: str | None = None
                wire_size: int | None = None
                try:
                    path = await self.mirror.policy_file(version, name)
                    wire_size = await self._wire_size(path, item)
                    url = (
                        f"{self.artifact_seed_origin}/policy/{version}/{name}"
                        f"?policy_object_key={object_key}"
                    )
                    parsed_url = urlsplit(url)
                    source = urlsplit(self.artifact_seed_origin)
                    query = parse_qs(parsed_url.query, keep_blank_values=True, strict_parsing=True)
                    if (
                        parsed_url.scheme.lower() != source.scheme.lower()
                        or parsed_url.hostname != source.hostname
                        or parsed_url.port != source.port
                        or parsed_url.path != f"/policy/{version}/{name}"
                        or query != {"policy_object_key": [object_key]}
                        or parsed_url.fragment
                    ):
                        raise PolicyPreseedError("seed source URL escaped the configured Gateway policy route")
                    seeded_path = await self.downloader.download_policy_object(
                        url,
                        object_key,
                        wire_size,
                        headers=self._headers,
                        allow_lan_source_fallback=True,
                        force_transfer=True,
                    )
                    seeded_wire_size = await self._wire_size(Path(seeded_path), item)
                    if seeded_wire_size != wire_size:
                        raise PolicyPreseedError("seeded policy object wire size changed during transfer")
                    self._seeded.add(key)
                    outcome = "seeded"
                except asyncio.CancelledError:
                    outcome = "cancelled"
                    error_name = "CancelledError"
                    raise
                except Exception as exc:
                    error_name = type(exc).__name__
                    raise PolicyPreseedError(
                        f"policy v{version} file {name} is not ready in the site seed"
                    ) from exc
                finally:
                    self._record({
                        "kind": "policy_preseed",
                        "site_id": self.site_id,
                        "lineage_id": self.lineage_id,
                        "version": version,
                        "filename": name,
                        "object_key": object_key,
                        "raw_size_bytes": item.size_bytes,
                        "wire_size_bytes": wire_size,
                        "transport": "identity" if name.endswith(".json") else "LORABLS2",
                        "outcome": outcome,
                        "error": error_name,
                        "started_at": started,
                        "finished_at": _now(),
                    })
