"""Owned single-slot writer for immutable central-gradient checkpoint payloads."""
from __future__ import annotations

import os
import re
import json
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from ...contracts.checkpoints import retained_versions

_RETENTION_LOCK = threading.RLock()

_POINTER_SCHEMA = "central-gradient-checkpoint-pointer/v1"


class CheckpointWriteError(RuntimeError):
    """Checkpoint write failed before durable latest-pointer completion."""

    def __init__(self, message: str, *, pointer_replaced: bool):
        super().__init__(message)
        self.pointer_replaced = pointer_replaced


@dataclass(frozen=True)
class CheckpointWriteResult:
    generation: int
    snapshot_id: str
    snapshot_name: str
    size_bytes: int
    durations_ns: dict[str, int]
    spans: dict[str, tuple[dict[str, Any], dict[str, Any]]]
    maintenance_error: str | None = None


def _point() -> dict[str, Any]:
    return {
        "monotonic_ns": time.perf_counter_ns(),
        "utc": datetime.now(UTC).isoformat(timespec="microseconds"),
    }


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_latest_pointer(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    pointer_replaced = False
    try:
        with open(temporary, "xb") as stream:
            stream.write(
                (__import__("json").dumps(
                    value, sort_keys=True, separators=(",", ":"),
                ) + "\n").encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        pointer_replaced = True
        _fsync_dir(path.parent)
    except Exception as exc:
        raise CheckpointWriteError(
            f"latest pointer write failed: {type(exc).__name__}: {exc}",
            pointer_replaced=pointer_replaced,
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_commit_record(pointer: Any) -> None:
    fields = {"schema", "generation", "snapshot_id", "snapshot", "size_bytes",
              "policy_version", "optimizer_steps"}
    if (not isinstance(pointer, Mapping) or set(pointer) != fields
            or pointer.get("schema") != _POINTER_SCHEMA
            or any(type(pointer.get(k)) is not int or pointer[k] < 0
                   for k in ("generation", "size_bytes", "policy_version", "optimizer_steps"))
            or not isinstance(pointer.get("snapshot_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", pointer["snapshot_id"]) is None
            or pointer.get("snapshot") != f"state-{pointer['generation']:08d}-{pointer['snapshot_id']}.pt"):
        raise RuntimeError("invalid committed checkpoint metadata")


def _record_committed_snapshot(root: Path, pointer: Mapping[str, Any]) -> None:
    """A small receipt proves the snapshot was published, without loading tensors."""
    _validate_commit_record(pointer)
    path = root / (pointer["snapshot"] + ".commit.json")
    if path.is_symlink():
        raise RuntimeError("checkpoint commit metadata cannot be a symlink")
    if path.exists():
        if json.loads(path.read_text()) != pointer:
            raise RuntimeError("immutable checkpoint commit metadata differs")
        return
    _atomic_latest_pointer(path, pointer)


def _prune_snapshots(root: Path, current: str) -> None:
    # Writers and asynchronous maintenance share this lock. The authoritative
    # latest pointer may have advanced beyond the maintenance request's version.
    del current
    with _RETENTION_LOCK:
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError("cannot prune checkpoints from a missing or unsafe root")
        latest_path = root / "latest.json"
        if latest_path.is_symlink() or not latest_path.is_file():
            raise RuntimeError("cannot prune checkpoints without a regular latest pointer")
        pointer = json.loads(latest_path.read_text())
        _validate_commit_record(pointer)
        latest = root / pointer["snapshot"]
        if latest.is_symlink() or not latest.is_file() or latest.stat().st_size != pointer["size_bytes"]:
            raise RuntimeError("cannot prune checkpoints while latest snapshot is unavailable")
        _record_committed_snapshot(root, pointer)
        records = []
        for path in root.glob("state-*.pt.commit.json"):
            if path.is_symlink():
                raise RuntimeError("checkpoint commit metadata cannot be a symlink")
            record = json.loads(path.read_text())
            _validate_commit_record(record)
            if path.name != record["snapshot"] + ".commit.json":
                raise RuntimeError("checkpoint commit metadata filename differs")
            if record["generation"] > pointer["generation"]:
                continue
            if record["generation"] == pointer["generation"] and record["snapshot"] != pointer["snapshot"]:
                continue
            records.append(record)
        keep_versions = retained_versions(pointer["policy_version"]) | {pointer["policy_version"]}
        newest = {}
        for record in sorted(records, key=lambda r: r["generation"], reverse=True):
            newest.setdefault(record["policy_version"], record)
        keep = {record["snapshot"] for version, record in newest.items()
                if version in keep_versions or version > pointer["policy_version"]}
        for record in records:
            if record["snapshot"] in keep:
                path = root / record["snapshot"]
                if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size_bytes"]:
                    raise RuntimeError("retained checkpoint payload is missing or differs in size")
        for record in records:
            if record["snapshot"] in keep:
                continue
            path = root / record["snapshot"]
            if path.is_symlink():
                raise RuntimeError("checkpoint payload cannot be a symlink")
            path.unlink(missing_ok=True)
            (root / (record["snapshot"] + ".commit.json")).unlink()
        # Files without a commit receipt are not claimed as committed archives
        # and are never guessed from their generation number or tensor contents.
        _fsync_dir(root)


def _span(
    spans: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    durations: dict[str, int],
    name: str,
    start: dict[str, Any],
    end: dict[str, Any],
) -> None:
    spans[name] = (start, end)
    durations[name] = end["monotonic_ns"] - start["monotonic_ns"]


def write_checkpoint_payload(
    root: str | Path,
    payload: Mapping[str, Any],
    *,
    prune: bool = True,
) -> CheckpointWriteResult:
    """Write a completed immutable checkpoint payload and atomically commit latest.json.

    The payload must already own CPU copies of all model, optimizer, scheduler,
    RNG, and receipt state. This function never reaches into a live core.
    """
    root = Path(root)
    generation = payload.get("generation")
    snapshot_id = payload.get("snapshot_id")
    if type(generation) is not int or generation < 0:
        raise ValueError("checkpoint payload generation must be a non-negative integer")
    if not isinstance(snapshot_id, str) or re.fullmatch(r"[0-9a-f]{32}", snapshot_id) is None:
        raise ValueError("checkpoint payload snapshot_id must be UUID hex")
    if root.is_symlink() or not root.is_dir():
        raise CheckpointWriteError("checkpoint root is missing or unsafe", pointer_replaced=False)

    snapshot_name = f"state-{generation:08d}-{snapshot_id}.pt"
    snapshot = root / snapshot_name
    temporary = root / f".{snapshot_name}.tmp"
    durations: dict[str, int] = {}
    spans: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    pointer_replaced = False
    metadata_error = None
    try:
        with open(temporary, "xb") as stream:
            started = _point()
            torch.save(payload, stream)
            _span(spans, durations, "checkpoint_torch_serialize", started, _point())
            started = _point()
            stream.flush()
            _span(spans, durations, "checkpoint_file_flush", started, _point())
            started = _point()
            os.fsync(stream.fileno())
            _span(spans, durations, "checkpoint_payload_fsync", started, _point())
        started = _point()
        os.replace(temporary, snapshot)
        _span(spans, durations, "checkpoint_snapshot_replace", started, _point())
        started = _point()
        _fsync_dir(root)
        _span(spans, durations, "checkpoint_root_directory_fsync", started, _point())
        size_bytes = snapshot.stat().st_size

        started = _point()
        pointer = {
            "schema": _POINTER_SCHEMA, "generation": generation,
            "snapshot_id": snapshot_id, "snapshot": snapshot_name,
            "size_bytes": size_bytes, "policy_version": payload["policy_version"],
            "optimizer_steps": payload["optimizer_steps"],
        }
        _validate_commit_record(pointer)
        with _RETENTION_LOCK:
            previous_path = root / "latest.json"
            previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
            _atomic_latest_pointer(previous_path, pointer)
            pointer_replaced = True
            try:
                if previous is not None:
                    _record_committed_snapshot(root, previous)
                _record_committed_snapshot(root, pointer)
            except Exception as exc:  # Metadata maintenance cannot revoke a durable latest.
                metadata_error = f"{type(exc).__name__}: {exc}"
        _span(spans, durations, "checkpoint_latest_pointer_atomic", started, _point())
    except CheckpointWriteError:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    except Exception as exc:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise CheckpointWriteError(
            f"atomic central checkpoint failed: {type(exc).__name__}: {exc}",
            pointer_replaced=pointer_replaced,
        ) from exc

    maintenance_error = metadata_error
    if prune:
        started = _point()
        try:
            _prune_snapshots(root, snapshot_name)
        except Exception as exc:  # noqa: BLE001 -- post-commit pruning cannot revoke durability
            maintenance_error = f"{type(exc).__name__}: {exc}"
        _span(spans, durations, "checkpoint_prune", started, _point())
    return CheckpointWriteResult(
        generation=generation,
        snapshot_id=snapshot_id,
        snapshot_name=snapshot_name,
        size_bytes=size_bytes,
        durations_ns=durations,
        spans=spans,
        maintenance_error=maintenance_error,
    )


class OwnedCheckpointWriter:
    """Single-worker writer with at most one submitted checkpoint operation."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="central-checkpoint")
        self._maintenance_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="central-checkpoint-maintenance",
        )
        self._lock = threading.Lock()
        self._future: Future[CheckpointWriteResult] | None = None
        self._maintenance_futures: set[Future[None]] = set()
        self._maintenance_error: str | None = None
        self._closed = False

    @property
    def maintenance_error(self) -> str | None:
        with self._lock:
            return self._maintenance_error

    def _schedule_prune(self, root: Path, current: str) -> None:
        try:
            with self._lock:
                future = self._maintenance_executor.submit(_prune_snapshots, root, current)
                self._maintenance_futures.add(future)
        except Exception as exc:  # noqa: BLE001 -- pruning cannot revoke a durable checkpoint
            with self._lock:
                self._maintenance_error = f"{type(exc).__name__}: {exc}"
            return

        def record_completion(completed: Future[None]) -> None:
            error = None
            try:
                completed.result()
            except BaseException as exc:  # noqa: BLE001 -- report maintenance failure only
                error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._maintenance_futures.discard(completed)
                if error is not None:
                    self._maintenance_error = error

        future.add_done_callback(record_completion)

    def submit(
        self,
        root: str | Path,
        payload: Mapping[str, Any],
        *,
        on_reserved: Callable[[Future[CheckpointWriteResult]], None] | None = None,
    ) -> Future[CheckpointWriteResult]:
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint writer is closed")
            if self._future is not None and not self._future.done():
                raise RuntimeError("checkpoint writer already has an active operation")
            future: Future[CheckpointWriteResult] = Future()
            self._future = future
            if on_reserved is not None:
                on_reserved(future)

            def run_write() -> None:
                if not future.set_running_or_notify_cancel():
                    return
                try:
                    root_path = Path(root)
                    result = write_checkpoint_payload(root_path, payload, prune=False)
                except BaseException as exc:  # noqa: BLE001 -- propagate worker failure through Future
                    future.set_exception(exc)
                else:
                    self._schedule_prune(root_path, result.snapshot_name)
                    future.set_result(result)

            try:
                self._executor.submit(run_write)
            except Exception as exc:  # noqa: BLE001 -- report submit failure through Future
                future.set_exception(exc)
            return future

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait)
        self._maintenance_executor.shutdown(wait=wait, cancel_futures=False)
