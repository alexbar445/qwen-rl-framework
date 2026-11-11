"""Pre-Gloo exact signatures and rank-slot reconstruction for DDP snapshots."""
from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import struct
import time
from typing import Any, Callable

import torch


MAX_DDP_OPTIMIZER_PICKLE_BYTES = 4 * 1024**3
MAX_DDP_SNAPSHOT_METADATA_BYTES = 64 * 1024**2


class OptimizerSnapshotPreflightError(ValueError):
    """A rank could not prove a supported optimizer snapshot signature."""


class _BoundedBytesIO(io.BytesIO):
    def __init__(self, max_bytes: int):
        super().__init__()
        self.max_bytes = max_bytes

    def write(self, data):
        if self.tell() + len(data) > self.max_bytes:
            raise OptimizerSnapshotPreflightError(
                f"serialized snapshot exceeds the {self.max_bytes}-byte bound"
            )
        return super().write(data)


def _field(digest, tag: bytes, payload: bytes = b"") -> None:
    digest.update(len(tag).to_bytes(2, "big"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _qualified_type(value: Any) -> bytes:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}".encode("utf-8", "surrogatepass")


def optimizer_state_signature(state: Any) -> tuple[str, int]:
    """Return a deterministic SHA-256 proof and logical tensor-byte count.

    The encoding preserves concrete container/scalar types and iteration order,
    tensor dtype/shape/stride/storage offset and raw backing storage bytes. It
    rejects unsupported Python and tensor layouts so a failed proof can never
    silently collapse distinct optimizer states.
    """
    if type(state) is not dict:
        raise OptimizerSnapshotPreflightError("optimizer state root must be a plain dict")

    digest = hashlib.sha256()
    seen_containers: dict[int, int] = {}
    seen_tensors: dict[int, int] = {}
    storage_ids: dict[int, int] = {}
    logical_tensor_bytes = 0

    def visit(value: Any) -> None:
        nonlocal logical_tensor_bytes
        value_type = type(value)
        if value is None:
            _field(digest, b"none", _qualified_type(value))
        elif value_type is bool:
            _field(digest, b"bool", b"1" if value else b"0")
        elif value_type is int:
            _field(digest, b"int", str(value).encode("ascii"))
        elif value_type is float:
            _field(digest, b"float64", struct.pack("!d", value))
        elif value_type is str:
            _field(digest, b"str", value.encode("utf-8", "surrogatepass"))
        elif value_type is bytes:
            _field(digest, b"bytes", value)
        elif value_type in (dict, list, tuple):
            identity = id(value)
            if identity in seen_containers:
                ref = seen_containers[identity]
                _field(digest, b"container-ref", value_type.__name__.encode() + b":" + str(ref).encode())
                return
            ref = len(seen_containers)
            seen_containers[identity] = ref
            _field(digest, b"container", value_type.__name__.encode() + b":" + str(ref).encode())
            if value_type is dict:
                _field(digest, b"length", str(len(value)).encode("ascii"))
                for key, item in value.items():
                    visit(key)
                    visit(item)
            else:
                _field(digest, b"length", str(len(value)).encode("ascii"))
                for item in value:
                    visit(item)
        elif isinstance(value, torch.Tensor):
            if value_type is not torch.Tensor:
                raise OptimizerSnapshotPreflightError(
                    f"unsupported optimizer tensor subclass {_qualified_type(value).decode()}"
                )
            identity = id(value)
            if identity in seen_tensors:
                _field(digest, b"tensor-ref", str(seen_tensors[identity]).encode("ascii"))
                return
            tensor_ref = len(seen_tensors)
            seen_tensors[identity] = tensor_ref
            if (value.device.type != "cpu" or value.layout != torch.strided
                    or value.is_quantized or value.requires_grad
                    or value.is_conj() or value.is_neg()):
                raise OptimizerSnapshotPreflightError(
                    "optimizer tensors must be ordinary detached CPU strided tensors"
                )
            try:
                storage = value.untyped_storage()
                storage_key = int(storage._cdata)
                storage_size = int(storage.nbytes())
                storage_offset = int(value.storage_offset())
                stride = tuple(int(part) for part in value.stride())
                shape = tuple(int(part) for part in value.shape)
                tensor_metadata = {
                    "tensor_ref": tensor_ref,
                    "dtype": str(value.dtype),
                    "layout": str(value.layout),
                    "device": str(value.device),
                    "shape": shape,
                    "stride": stride,
                    "storage_offset": storage_offset,
                    "storage_nbytes": storage_size,
                    "requires_grad": bool(value.requires_grad),
                    "is_conj": bool(value.is_conj()),
                    "is_neg": bool(value.is_neg()),
                    "is_quantized": bool(value.is_quantized),
                    "is_pinned": bool(value.is_pinned()),
                    "names": tuple(value.names) if hasattr(value, "names") else None,
                }
                metadata_bytes = json.dumps(
                    tensor_metadata, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            except BaseException as exc:
                raise OptimizerSnapshotPreflightError(
                    f"cannot read optimizer tensor layout: {type(exc).__name__}: {exc}"
                ) from exc
            _field(digest, b"tensor", metadata_bytes)
            logical_tensor_bytes += int(value.numel()) * int(value.element_size())
            if storage_key in storage_ids:
                _field(digest, b"storage-ref", str(storage_ids[storage_key]).encode("ascii"))
            else:
                storage_ref = len(storage_ids)
                storage_ids[storage_key] = storage_ref
                _field(digest, b"storage", str(storage_ref).encode("ascii"))
                _field(digest, b"storage-size", str(storage_size).encode("ascii"))
                try:
                    raw = torch.empty(0, dtype=torch.uint8).set_(
                        storage, 0, (storage_size,), (1,)
                    )
                    if storage_size:
                        digest.update(memoryview(raw.numpy()).cast("B"))
                except BaseException as exc:
                    raise OptimizerSnapshotPreflightError(
                        f"cannot read optimizer tensor bytes: {type(exc).__name__}: {exc}"
                    ) from exc
        else:
            raise OptimizerSnapshotPreflightError(
                f"unsupported optimizer snapshot value type {_qualified_type(value).decode()}"
            )

    visit(state)
    return digest.hexdigest(), logical_tensor_bytes


def _validate_statuses(statuses: Any, world_size: int) -> list[dict[str, Any]]:
    if not isinstance(statuses, list) or len(statuses) != world_size:
        raise OptimizerSnapshotPreflightError(
            "optimizer signature all-gather did not return one status per rank"
        )
    failures = []
    for rank, status in enumerate(statuses):
        if (not isinstance(status, dict) or type(status.get("rank")) is not int
                or status.get("rank") != rank):
            failures.append(f"rank {rank}: malformed signature status")
        elif status.get("ok") is not True:
            detail = status.get("error")
            failures.append(f"rank {rank}: {detail if isinstance(detail, str) else 'signature failed'}")
    if failures:
        raise OptimizerSnapshotPreflightError(
            "DDP optimizer snapshot signature failed closed: " + "; ".join(failures)
        )
    for rank, status in enumerate(statuses):
        signature = status.get("signature")
        if (not isinstance(signature, str) or len(signature) != 64
                or any(char not in "0123456789abcdef" for char in signature)):
            raise OptimizerSnapshotPreflightError(f"rank {rank}: malformed optimizer signature")
        count = status.get("logical_tensor_bytes")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise OptimizerSnapshotPreflightError(f"rank {rank}: malformed optimizer tensor byte count")
        proof_seconds = status.get("digest_seconds")
        if (isinstance(proof_seconds, bool) or not isinstance(proof_seconds, (int, float))
                or not math.isfinite(float(proof_seconds)) or proof_seconds < 0):
            raise OptimizerSnapshotPreflightError(f"rank {rank}: malformed digest timing")
    return statuses


def preflight_ddp_optimizer_snapshot(
    state: Any,
    *,
    rank: int,
    world_size: int,
    all_gather_object: Callable[[list[Any], dict[str, Any]], None],
    local_error: BaseException | str | None = None,
) -> dict[str, Any]:
    """All-gather only digests/status, including returned local failures."""
    started = time.perf_counter()
    safe_rank = rank if type(rank) is int else -1
    digest_started = time.perf_counter()
    if local_error is not None:
        if isinstance(local_error, BaseException):
            error_text = f"{type(local_error).__name__}: {local_error}"
        else:
            error_text = str(local_error)
        local_status = {
            "rank": safe_rank, "ok": False,
            "error": error_text[:768],
            "digest_seconds": time.perf_counter() - digest_started,
        }
    else:
        try:
            signature, logical_tensor_bytes = optimizer_state_signature(state)
            local_status = {
                "rank": safe_rank, "ok": True, "signature": signature,
                "logical_tensor_bytes": logical_tensor_bytes,
                "digest_seconds": time.perf_counter() - digest_started,
            }
        except BaseException as exc:
            local_status = {
                "rank": safe_rank, "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:768]}",
                "digest_seconds": time.perf_counter() - digest_started,
            }
    statuses: list[Any] = [None] * int(world_size)
    collective_started = time.perf_counter()
    try:
        all_gather_object(statuses, local_status)
    except BaseException as exc:
        raise OptimizerSnapshotPreflightError(
            f"DDP optimizer signature all-gather failed: {type(exc).__name__}: {exc}"
        ) from exc
    collective_seconds = time.perf_counter() - collective_started
    proof_seconds = time.perf_counter() - started
    validated = _validate_statuses(statuses, int(world_size))
    representatives: list[int] = []
    digest_to_representative: dict[str, int] = {}
    for record in validated:
        signature = record["signature"]
        representatives.append(digest_to_representative.setdefault(signature, int(record["rank"])))
    # A SHA-256 signature includes the complete recursive encoding and tensor
    # byte count; equal digests with different counts indicate a corrupt status.
    byte_counts_by_signature: dict[str, int] = {}
    for record in validated:
        signature = record["signature"]
        count = int(record["logical_tensor_bytes"])
        previous = byte_counts_by_signature.setdefault(signature, count)
        if previous != count:
            raise OptimizerSnapshotPreflightError(
                "DDP optimizer signature collision has conflicting tensor byte counts"
            )
    return {
        "rank_representatives": representatives,
        "signatures": [record["signature"] for record in validated],
        "logical_tensor_bytes_by_rank": [int(record["logical_tensor_bytes"]) for record in validated],
        "digest_seconds_by_rank": [float(record["digest_seconds"]) for record in validated],
        "proof_seconds": float(proof_seconds),
        "signature_collective_seconds": float(collective_seconds),
    }


def synchronize_ddp_snapshot_payload_ready(
    *,
    rank: int,
    world_size: int,
    local_error: BaseException | str | None,
    all_gather_object: Callable[[list[Any], dict[str, Any]], None],
    status_fields: dict[str, Any] | None = None,
    expected_representatives: list[int] | None = None,
    expected_transfer_bytes_by_rank: list[int] | None = None,
    expected_received_bytes_by_rank: list[int] | None = None,
    bounded_fields: dict[str, int] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Make local preparation/transfer errors unanimous before the next collective."""
    started = time.perf_counter()
    safe_rank = rank if type(rank) is int else -1
    status: dict[str, Any] = {"rank": safe_rank, "ok": local_error is None}
    if status_fields:
        status.update(status_fields)
    if local_error is not None:
        if isinstance(local_error, BaseException):
            error_text = f"{type(local_error).__name__}: {local_error}"
        else:
            error_text = str(local_error)
        status["error"] = error_text[:768]
    statuses: list[Any] = [None] * int(world_size)
    try:
        all_gather_object(statuses, status)
    except BaseException as exc:
        raise OptimizerSnapshotPreflightError(
            f"DDP optimizer readiness all-gather failed: {type(exc).__name__}: {exc}"
        ) from exc
    elapsed = time.perf_counter() - started
    if not isinstance(statuses, list) or len(statuses) != int(world_size):
        raise OptimizerSnapshotPreflightError(
            "optimizer readiness all-gather did not return one status per rank"
        )
    failures = []
    for expected_rank, record in enumerate(statuses):
        if (not isinstance(record, dict) or type(record.get("rank")) is not int
                or record.get("rank") != expected_rank):
            failures.append(f"rank {expected_rank}: malformed readiness status")
        elif record.get("ok") is not True:
            detail = record.get("error")
            failures.append(f"rank {expected_rank}: {detail if isinstance(detail, str) else 'preparation failed'}")
    if failures:
        raise OptimizerSnapshotPreflightError(
            "DDP optimizer snapshot readiness failed closed: " + "; ".join(failures)
        )
    for expected_rank, record in enumerate(statuses):
        for field, maximum in (bounded_fields or {}).items():
            value = record.get(field)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < 0 or value > maximum):
                raise OptimizerSnapshotPreflightError(
                    f"rank {expected_rank}: invalid or oversized {field}"
                )
    if expected_representatives is not None:
        if len(expected_representatives) != int(world_size):
            raise OptimizerSnapshotPreflightError("optimizer representative plan has the wrong world size")
        for expected_rank, representative in enumerate(expected_representatives):
            value = statuses[expected_rank].get("optimizer_pickle_bytes")
            should_send = expected_rank > 0 and representative == expected_rank
            if (isinstance(value, bool) or not isinstance(value, int) or value < 0
                    or (should_send and value == 0) or (not should_send and value != 0)):
                raise OptimizerSnapshotPreflightError(
                    f"rank {expected_rank}: optimizer serialized bytes disagree with representative plan"
                )
    if expected_transfer_bytes_by_rank is not None:
        if len(expected_transfer_bytes_by_rank) != int(world_size):
            raise OptimizerSnapshotPreflightError("optimizer transfer plan has the wrong world size")
        for expected_rank, expected in enumerate(expected_transfer_bytes_by_rank):
            value = statuses[expected_rank].get("optimizer_transfer_bytes")
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value != expected):
                raise OptimizerSnapshotPreflightError(
                    f"rank {expected_rank}: optimizer transfer byte receipt differs from plan"
                )
    if expected_received_bytes_by_rank is not None:
        if len(expected_received_bytes_by_rank) != int(world_size):
            raise OptimizerSnapshotPreflightError("optimizer receive plan has the wrong world size")
        for expected_rank, expected in enumerate(expected_received_bytes_by_rank):
            value = statuses[expected_rank].get("optimizer_received_bytes")
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value != expected):
                raise OptimizerSnapshotPreflightError(
                    f"rank {expected_rank}: optimizer receive byte receipt differs from plan"
                )
    return elapsed, statuses


def stage_ddp_snapshot_payload(
    payload: dict[str, Any], *, rank: int, plan: dict[str, Any]
) -> dict[str, Any]:
    """Keep heavy state local or in a prepared P2P buffer; leave only small metadata."""
    representatives = plan["rank_representatives"]
    if not isinstance(payload, dict) or "optimizer_state" not in payload:
        raise OptimizerSnapshotPreflightError("rank snapshot is missing its optimizer state")
    if type(rank) is not int or not 0 <= rank < len(representatives):
        raise OptimizerSnapshotPreflightError("rank is outside the optimizer snapshot plan")
    representative = int(representatives[rank])
    state = payload.pop("optimizer_state")
    if type(state) is not dict:
        raise OptimizerSnapshotPreflightError("rank optimizer state is not a plain dict")
    adapter_state = payload.pop("adapter_state", None)
    if rank == 0:
        if not isinstance(adapter_state, dict):
            raise OptimizerSnapshotPreflightError("rank 0 is missing its full adapter state")
    elif adapter_state is not None:
        raise OptimizerSnapshotPreflightError("a nonzero rank unexpectedly supplied adapter state")

    local_optimizer_state = None
    send_stream = None
    send_tensor = None
    optimizer_pickle_bytes = 0
    if rank == 0:
        local_optimizer_state = state
    elif representative == rank:
        logical_bytes = int(plan["logical_tensor_bytes_by_rank"][rank])
        if logical_bytes > MAX_DDP_OPTIMIZER_PICKLE_BYTES:
            raise OptimizerSnapshotPreflightError(
                f"logical optimizer state size {logical_bytes} exceeds the transfer bound"
            )
        send_stream = _BoundedBytesIO(MAX_DDP_OPTIMIZER_PICKLE_BYTES)
        pickle.Pickler(send_stream, protocol=pickle.HIGHEST_PROTOCOL).dump(state)
        optimizer_pickle_bytes = send_stream.tell()
        if not 0 < optimizer_pickle_bytes <= MAX_DDP_OPTIMIZER_PICKLE_BYTES:
            raise OptimizerSnapshotPreflightError(
                f"serialized optimizer state size {optimizer_pickle_bytes} exceeds the transfer bound"
            )
        send_tensor = torch.frombuffer(send_stream.getbuffer(), dtype=torch.uint8)
    metadata = {
        "rank": rank,
        "representative": representative,
        "signature": plan["signatures"][rank],
        "logical_tensor_bytes": int(plan["logical_tensor_bytes_by_rank"][rank]),
        "digest_seconds": float(plan["digest_seconds_by_rank"][rank]),
        "proof_seconds": float(plan["proof_seconds"]),
        "signature_collective_seconds": float(plan["signature_collective_seconds"]),
        "payload_ready_collective_seconds": 0.0,
        "optimizer_pickle_bytes": optimizer_pickle_bytes,
        "optimizer_transfer_bytes": 0,
        "small_gather_pickle_bytes": 0,
        "adapter_tensor_bytes_local": _nested_tensor_bytes(adapter_state) if rank == 0 else 0,
    }
    payload["_optimizer_snapshot_pre_gloo"] = metadata
    return {
        "optimizer_state": local_optimizer_state,
        "adapter_state": adapter_state if rank == 0 else None,
        "send_stream": send_stream,
        "send_tensor": send_tensor,
        "optimizer_pickle_bytes": optimizer_pickle_bytes,
    }


def _nested_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    if type(value) is dict:
        return sum(_nested_tensor_bytes(item) for item in value.values())
    if type(value) in (list, tuple):
        return sum(_nested_tensor_bytes(item) for item in value)
    return 0


def measure_small_gather_pickle_bytes(local_result: dict[str, Any]) -> int:
    """Record the exact default-Pickler byte length used by Gloo gather_object."""
    if not isinstance(local_result, dict):
        raise OptimizerSnapshotPreflightError("small rank result is not a dictionary")
    payload = local_result.get("payload")
    metadata = payload.get("_optimizer_snapshot_pre_gloo") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise OptimizerSnapshotPreflightError("small rank result lacks snapshot metadata")
    for _ in range(4):
        buffer = _BoundedBytesIO(MAX_DDP_SNAPSHOT_METADATA_BYTES)
        pickle.Pickler(buffer).dump(local_result)
        byte_count = buffer.tell()
        if metadata.get("small_gather_pickle_bytes") == byte_count:
            return byte_count
        metadata["small_gather_pickle_bytes"] = byte_count
    raise OptimizerSnapshotPreflightError("small gather pickle length did not stabilize")


def load_optimizer_state_from_pickle_buffer(buffer: torch.Tensor, byte_count: int) -> dict[str, Any]:
    if (not isinstance(buffer, torch.Tensor) or buffer.device.type != "cpu"
            or buffer.dtype != torch.uint8 or buffer.ndim != 1):
        raise OptimizerSnapshotPreflightError("optimizer receive buffer has an invalid tensor layout")
    if (isinstance(byte_count, bool) or not isinstance(byte_count, int)
            or not 0 < byte_count <= buffer.numel()
            or byte_count > MAX_DDP_OPTIMIZER_PICKLE_BYTES):
        raise OptimizerSnapshotPreflightError("optimizer pickle byte count is outside the receive buffer")
    view = memoryview(buffer[:byte_count].numpy()).cast("B")
    state = pickle.loads(view)
    if type(state) is not dict:
        raise OptimizerSnapshotPreflightError("received optimizer pickle is not a plain dict")
    return state


def transfer_ddp_snapshot_heavy_payloads(
    gathered: list[dict[str, Any]] | None,
    local_heavy: dict[str, Any],
    *,
    rank: int,
    world_size: int,
    optimizer_pickle_bytes_by_rank: list[int],
    receive_buffer: torch.Tensor | None,
    dist: Any,
    group: Any,
) -> dict[str, Any]:
    """Transfer only nonzero unique optimizer states, then restore rank-0 local trees."""
    started = time.perf_counter()
    error_text = None
    transfer_bytes = 0
    received_bytes = 0
    if rank == 0:
        try:
            if not isinstance(gathered, list) or len(gathered) != world_size:
                raise OptimizerSnapshotPreflightError("small Gloo gather omitted rank envelopes")
            for item in gathered:
                envelope = item.get("payload") if isinstance(item, dict) else None
                if not isinstance(envelope, dict):
                    raise OptimizerSnapshotPreflightError("small Gloo gather has a malformed rank envelope")
                envelope["adapter_state"] = None
            gathered[0]["payload"]["optimizer_state"] = local_heavy["optimizer_state"]
            gathered[0]["payload"]["adapter_state"] = local_heavy["adapter_state"]
        except BaseException as exc:
            error_text = f"{type(exc).__name__}: {str(exc)[:768]}"
        for source_rank in range(1, world_size):
            byte_count = int(optimizer_pickle_bytes_by_rank[source_rank])
            if byte_count == 0:
                continue
            try:
                if receive_buffer is None or receive_buffer.numel() < byte_count:
                    raise OptimizerSnapshotPreflightError("optimizer receive buffer is smaller than its announced payload")
                dist.recv(receive_buffer[:byte_count], src=source_rank, group=group)
                received_bytes += byte_count
            except BaseException as exc:
                error_text = error_text or f"{type(exc).__name__}: {str(exc)[:768]}"
                break
            try:
                state = load_optimizer_state_from_pickle_buffer(receive_buffer, byte_count)
                gathered[source_rank]["payload"]["optimizer_state"] = state
            except BaseException as exc:
                error_text = error_text or f"{type(exc).__name__}: {str(exc)[:768]}"
    else:
        send_tensor = local_heavy.get("send_tensor")
        if send_tensor is not None:
            try:
                dist.send(send_tensor, dst=0, group=group)
                transfer_bytes = int(send_tensor.numel())
            except BaseException as exc:
                error_text = f"{type(exc).__name__}: {str(exc)[:768]}"
    return {
        "rank": rank,
        "error": error_text,
        "optimizer_transfer_bytes": transfer_bytes,
        "optimizer_received_bytes": received_bytes,
        "transfer_seconds": time.perf_counter() - started,
    }

def reconstruct_ddp_optimizer_states(
    snapshots: list[dict[str, Any]], *, world_size: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild rank slots from local rank-0 and unique P2P optimizer payloads."""
    if not isinstance(snapshots, list) or len(snapshots) != world_size:
        raise OptimizerSnapshotPreflightError("Gloo snapshot gather omitted DDP ranks")
    metadata = [item.pop("_optimizer_snapshot_pre_gloo", None) for item in snapshots]
    validated = _validate_statuses([
        {
            "rank": row.get("rank") if isinstance(row, dict) else None,
            "ok": True,
            "signature": row.get("signature") if isinstance(row, dict) else None,
            "logical_tensor_bytes": row.get("logical_tensor_bytes") if isinstance(row, dict) else None,
            "digest_seconds": row.get("digest_seconds") if isinstance(row, dict) else None,
        }
        for row in metadata
    ], world_size)
    representatives = []
    for rank, row in enumerate(metadata):
        representative = row.get("representative") if isinstance(row, dict) else None
        if type(representative) is not int or not 0 <= representative <= rank:
            raise OptimizerSnapshotPreflightError(f"rank {rank}: malformed optimizer representative")
        if validated[rank]["signature"] != validated[representative]["signature"]:
            raise OptimizerSnapshotPreflightError(f"rank {rank}: representative signature differs")
        representatives.append(representative)
    expected_representatives: list[int] = []
    first_rank_by_signature: dict[str, int] = {}
    for rank, status in enumerate(validated):
        expected_representatives.append(first_rank_by_signature.setdefault(status["signature"], rank))
    if representatives != expected_representatives:
        raise OptimizerSnapshotPreflightError("Gloo optimizer representative map is inconsistent")

    unique_states: dict[int, dict[str, Any]] = {}
    for rank, (snapshot, representative) in enumerate(zip(snapshots, representatives)):
        if representative == rank:
            state = snapshot.pop("optimizer_state", None)
            if type(state) is not dict:
                raise OptimizerSnapshotPreflightError(
                    f"rank {rank}: representative optimizer state is missing or malformed"
                )
            unique_states[rank] = state
        elif "optimizer_state" in snapshot:
            raise OptimizerSnapshotPreflightError(
                f"rank {rank}: duplicate optimizer state crossed the Gloo object gather"
            )
    if set(unique_states) != set(representatives):
        raise OptimizerSnapshotPreflightError("Gloo transfer is missing a representative optimizer state")

    byte_counts = [int(row["logical_tensor_bytes"]) for row in metadata]
    total_bytes = sum(byte_counts)
    unique_bytes = sum(byte_counts[rank] for rank in unique_states)
    pickle_bytes = []
    transfer_bytes = []
    received_bytes = []
    small_gather_bytes = []
    proof_seconds = []
    signature_collective_seconds = []
    readiness_seconds = []
    adapter_local_bytes = 0
    for rank, row in enumerate(metadata):
        for key in (
            "proof_seconds", "signature_collective_seconds", "payload_ready_collective_seconds",
            "optimizer_pickle_bytes", "optimizer_transfer_bytes", "optimizer_received_bytes",
            "small_gather_pickle_bytes",
            "adapter_tensor_bytes_local",
        ):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OptimizerSnapshotPreflightError(f"rank {rank}: malformed {key} telemetry")
            if not math.isfinite(float(value)) or value < 0:
                raise OptimizerSnapshotPreflightError(f"rank {rank}: malformed {key} telemetry")
        size = int(row["optimizer_pickle_bytes"])
        actual = int(row["optimizer_transfer_bytes"])
        should_send = rank > 0 and representatives[rank] == rank
        if size > MAX_DDP_OPTIMIZER_PICKLE_BYTES or (should_send and size == 0):
            raise OptimizerSnapshotPreflightError(f"rank {rank}: optimizer pickle size violates the transfer plan")
        if not should_send and size != 0:
            raise OptimizerSnapshotPreflightError(f"rank {rank}: non-sender has optimizer pickle bytes")
        if actual != (size if should_send else 0):
            raise OptimizerSnapshotPreflightError(f"rank {rank}: optimizer P2P transfer receipt differs from plan")
        received = int(row["optimizer_received_bytes"])
        expected_received = sum(
            int(metadata[source]["optimizer_pickle_bytes"])
            for source in range(1, world_size)
            if representatives[source] == source
        ) if rank == 0 else 0
        if received != expected_received:
            raise OptimizerSnapshotPreflightError(f"rank {rank}: optimizer P2P receive receipt differs from plan")
        small_size = int(row["small_gather_pickle_bytes"])
        if not 0 < small_size <= MAX_DDP_SNAPSHOT_METADATA_BYTES:
            raise OptimizerSnapshotPreflightError(f"rank {rank}: small Gloo envelope exceeds its byte bound")
        pickle_bytes.append(size)
        transfer_bytes.append(actual)
        received_bytes.append(received)
        small_gather_bytes.append(small_size)
        proof_seconds.append(float(row["proof_seconds"]))
        signature_collective_seconds.append(float(row["signature_collective_seconds"]))
        readiness_seconds.append(float(row["payload_ready_collective_seconds"]))
        adapter_bytes = int(row["adapter_tensor_bytes_local"])
        if rank == 0:
            adapter_local_bytes = adapter_bytes
        elif adapter_bytes != 0:
            raise OptimizerSnapshotPreflightError(f"rank {rank}: nonzero rank adapter bytes were not expected")

    padded_small_gather_bytes = max(small_gather_bytes, default=0) * world_size
    p2p_optimizer_bytes = sum(transfer_bytes)
    report = {
        "recipe": "pre_gloo_exact_ddp_optimizer_signature/v2_p2p",
        "rank_slots": world_size,
        "unique_states": len(unique_states),
        "rank_representatives": representatives,
        "signatures": [row["signature"] for row in metadata],
        "logical_tensor_bytes_before": total_bytes,
        "logical_tensor_bytes_after": unique_bytes,
        "logical_tensor_bytes_avoided": total_bytes - unique_bytes,
        "bytes_avoided_before_gloo": total_bytes - unique_bytes,
        "logical_tensor_bytes_omitted_from_gather_object": total_bytes,
        "optimizer_state_tensor_bytes_in_gather_object": 0,
        "adapter_state_tensor_bytes_in_gather_object": 0,
        "optimizer_pickle_bytes_by_rank": pickle_bytes,
        "optimizer_p2p_transfer_bytes_by_rank": transfer_bytes,
        "optimizer_p2p_received_bytes_by_rank": received_bytes,
        "optimizer_p2p_wire_bytes_to_root": p2p_optimizer_bytes,
        "remote_heavy_serialized_bytes": sum(pickle_bytes),
        "small_gather_object_pickle_bytes_by_rank": small_gather_bytes,
        "small_gather_object_padded_payload_bytes_to_root": padded_small_gather_bytes,
        "snapshot_object_and_p2p_payload_bytes_to_root": padded_small_gather_bytes + p2p_optimizer_bytes,
        "root_local_optimizer_tensor_bytes": byte_counts[0],
        "adapter_tensor_bytes_kept_local_on_rank0": adapter_local_bytes,
        "root_local_heavy_tensor_bytes_excluded_from_wire": byte_counts[0] + adapter_local_bytes,
        "digest_seconds_by_rank": [float(row["digest_seconds"]) for row in metadata],
        "proof_seconds_by_rank": proof_seconds,
        "proof_seconds_max": max(proof_seconds, default=0.0),
        "signature_collective_seconds_max": max(signature_collective_seconds, default=0.0),
        "payload_ready_collective_seconds_max": max(readiness_seconds, default=0.0),
        "comparison_seconds": max(proof_seconds, default=0.0),
        "scope": (
            "optimizer signatures precede Gloo; object gather contains small rank metadata only; "
            "rank-0 adapter stays local and nonzero unique optimizer states use variable-size Gloo P2P"
        ),
    }
    return [unique_states[representative] for representative in representatives], report

def intern_ddp_optimizer_states(states: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Intern already-collected CPU states with the same fail-closed signature."""
    if not isinstance(states, list) or not states or any(type(state) is not dict for state in states):
        raise ValueError("expected one optimizer state dictionary per DDP rank")
    started = time.perf_counter()
    compact: list[dict[str, Any]] = []
    representatives: list[int] = []
    digest_to_representative: dict[str, int] = {}
    byte_counts: list[int] = []
    signatures: list[str] = []
    for rank, state in enumerate(states):
        signature, tensor_bytes = optimizer_state_signature(state)
        representative = digest_to_representative.setdefault(signature, rank)
        representatives.append(representative)
        signatures.append(signature)
        byte_counts.append(tensor_bytes)
        if representative == rank:
            compact.append(state)
        else:
            compact.append(compact[representative])
    before = sum(byte_counts)
    after = sum(byte_counts[rank] for rank in sorted(set(representatives)))
    elapsed = time.perf_counter() - started
    return compact, {
        "recipe": "exact_cpu_ddp_optimizer_signature/v1",
        "rank_slots": len(states),
        "unique_states": len(set(representatives)),
        "rank_representatives": representatives,
        "signatures": signatures,
        "comparison_seconds": elapsed,
        "logical_tensor_bytes_before": before,
        "logical_tensor_bytes_after": after,
        "logical_tensor_bytes_avoided": before - after,
        "scope": "post-Gloo CPU snapshot interning",
    }
