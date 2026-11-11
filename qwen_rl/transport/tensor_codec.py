"""Bounded lossless transport for safetensors and native adapter blobs."""

from __future__ import annotations

import json
import struct
import sys
from typing import Any

import blosc2

TENSOR_PAYLOAD_MAGIC = b"LORABLS2"
TENSOR_PAYLOAD_ENVELOPE = struct.Struct("<8sQQ")
SAFETENSORS_LENGTH_PREFIX = struct.Struct("<Q")
TENSOR_PAYLOAD_CONTENT_TYPE = "application/vnd.qwen.tensor-payload"
TENSOR_PAYLOAD_TRANSPORT = "blosc2-zstd-lossless-v2"

# compress2 writes a single Blosc2 frame. Keep transport payloads below its
# single-buffer limit while leaving a bound that fits the current LoRA state.
MAX_TENSOR_RAW_BYTES = 1 << 30
MAX_SAFETENSORS_HEADER_BYTES = 100_000_000
MAX_FRAME_OVERHEAD = int(blosc2.MAX_OVERHEAD)

_UINT64_MAX = (1 << 64) - 1
_FILTERS = [blosc2.Filter.NOFILTER] * 5 + [blosc2.Filter.SHUFFLE]
_FILTERS_META = [0] * 6


def _as_byte_view(value: object, *, name: str) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as exc:
        raise TypeError(f"{name} must support the buffer protocol") from exc
    if not view.contiguous:
        view.release()
        raise TypeError(f"{name} must be contiguous")
    try:
        byte_view = view.cast("B")
    except (TypeError, ValueError) as exc:
        view.release()
        raise TypeError(f"{name} must expose a contiguous byte buffer") from exc
    if byte_view is not view:
        view.release()
    return byte_view


def _parse_bounded_json_int(value: str) -> int:
    number = int(value)
    if number < 0 or number > _UINT64_MAX:
        raise ValueError("safetensors header integer is outside uint64 bounds")
    return number


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant in safetensors header: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in safetensors header")
        result[key] = value
    return result


def _validate_safetensors_header(
    raw_prefix_and_header: bytes | memoryview,
    *,
    raw_size: int,
    header_size: int,
) -> bytes:
    """Check the plaintext SafeTensors framing and tensor byte ranges."""
    if raw_size < SAFETENSORS_LENGTH_PREFIX.size:
        raise ValueError("safetensors payload is shorter than its length prefix")
    if header_size > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError("safetensors header exceeds the configured limit")
    if header_size > raw_size - SAFETENSORS_LENGTH_PREFIX.size:
        raise ValueError("safetensors header length exceeds the raw payload")

    prefix_and_header = bytes(raw_prefix_and_header)
    if len(prefix_and_header) != SAFETENSORS_LENGTH_PREFIX.size + header_size:
        raise ValueError("safetensors plaintext header is truncated")
    declared_header_size = SAFETENSORS_LENGTH_PREFIX.unpack_from(prefix_and_header)[0]
    if declared_header_size != header_size:
        raise ValueError("safetensors header length does not match the envelope")

    header_bytes = prefix_and_header[SAFETENSORS_LENGTH_PREFIX.size :]
    try:
        header = json.loads(
            header_bytes.decode("utf-8"),
            parse_int=_parse_bounded_json_int,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensors header is not valid UTF-8 JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")  # noqa: TRY004

    body_size = raw_size - SAFETENSORS_LENGTH_PREFIX.size - header_size
    ranges: list[tuple[int, int]] = []
    for name, tensor_info in header.items():
        if name == "__metadata__":
            if not isinstance(tensor_info, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in tensor_info.items()
            ):
                raise ValueError("safetensors metadata must contain string keys and values")
            continue
        if not isinstance(name, str) or not isinstance(tensor_info, dict):
            raise ValueError("safetensors tensor entries must be named objects")  # noqa: TRY004
        dtype = tensor_info.get("dtype")
        shape = tensor_info.get("shape")
        offsets = tensor_info.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("safetensors tensor dtype must be a non-empty string")
        if not isinstance(shape, list):
            raise ValueError("safetensors tensor shape must be a list")  # noqa: TRY004
        elements = 1
        for dimension in shape:
            if type(dimension) is not int or dimension > _UINT64_MAX:
                raise ValueError("safetensors tensor dimensions must be uint64 integers")
            if dimension == 0:
                elements = 0
            elif elements and elements > body_size // dimension:
                raise ValueError("safetensors tensor shape exceeds the data body")
            else:
                elements *= dimension
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("safetensors data_offsets must be a pair")
        start, end = offsets
        if (
            type(start) is not int
            or type(end) is not int
            or start > body_size
            or end > body_size
            or start > end
        ):
            raise ValueError("safetensors data_offsets exceed the data body")
        if end - start < elements:
            raise ValueError("safetensors tensor shape exceeds its data range")
        ranges.append((start, end))

    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("safetensors data ranges contain a gap or overlap")
        cursor = end
    if cursor != body_size:
        raise ValueError("safetensors data ranges do not cover the data body")
    return prefix_and_header


def _compression_params(compression: bool) -> blosc2.CParams:
    return blosc2.CParams(
        typesize=4,
        clevel=1 if compression else 0,
        codec=blosc2.Codec.ZSTD,
        nthreads=4,
        filters=_FILTERS,
        filters_meta=_FILTERS_META,
    )


def max_encoded_size(raw_size: int) -> int:
    """Return the conservative wire ceiling for a raw transport payload."""
    if type(raw_size) is not int or raw_size < 0 or raw_size > MAX_TENSOR_RAW_BYTES:
        raise ValueError("raw_size is outside the tensor transport limit")
    # In safetensors mode the plaintext prefix/header is added back while the
    # same bytes are removed from the compressed region, so they cancel out.
    return raw_size + TENSOR_PAYLOAD_ENVELOPE.size + MAX_FRAME_OVERHEAD


def declared_tensor_raw_size(
    packed: bytes | bytearray | memoryview, *, max_raw_bytes: int
) -> int:
    """Read the bounded raw size from a bytes-like or mmap-backed payload."""
    if type(max_raw_bytes) is not int or max_raw_bytes < 0:
        raise ValueError("max_raw_bytes must be a non-negative integer")
    packed_view = _as_byte_view(packed, name="packed")
    try:
        if packed_view.nbytes < TENSOR_PAYLOAD_ENVELOPE.size:
            raise ValueError("tensor payload is shorter than its envelope")
        magic, raw_size, header_size = TENSOR_PAYLOAD_ENVELOPE.unpack_from(packed_view)
        if magic != TENSOR_PAYLOAD_MAGIC:
            raise ValueError("tensor payload has the wrong magic")
        if raw_size > min(max_raw_bytes, MAX_TENSOR_RAW_BYTES) or raw_size > sys.maxsize:
            raise ValueError("tensor payload exceeds the configured raw size limit")
        if header_size:
            if header_size > MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError("safetensors header exceeds the configured limit")
            if raw_size < SAFETENSORS_LENGTH_PREFIX.size or header_size > raw_size - 8:
                raise ValueError("safetensors header length exceeds the raw payload")
            if packed_view.nbytes < TENSOR_PAYLOAD_ENVELOPE.size + 8 + header_size:
                raise ValueError("safetensors plaintext header is truncated")
        return raw_size
    except struct.error as exc:
        raise ValueError("tensor payload envelope is truncated") from exc
    finally:
        packed_view.release()


def encode_tensor_payload(
    raw: bytes, *, safetensors: bool = True, compression: bool = True,
) -> bytes:
    """Frame a tensor payload; level 0 stores bytes without compression/filters."""
    if type(compression) is not bool:
        raise TypeError("compression must be a bool")
    if type(safetensors) is not bool:
        raise TypeError("safetensors must be a bool")
    raw_view = _as_byte_view(raw, name="raw")
    data_view = raw_view
    try:
        raw_size = raw_view.nbytes
        if raw_size > MAX_TENSOR_RAW_BYTES or raw_size > sys.maxsize:
            raise ValueError("raw tensor payload exceeds the configured limit")

        header_size = 0
        plaintext = b""
        if safetensors:
            if raw_size < SAFETENSORS_LENGTH_PREFIX.size:
                raise ValueError("safetensors payload is shorter than its length prefix")
            header_size = SAFETENSORS_LENGTH_PREFIX.unpack_from(raw_view)[0]
            if header_size > MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError("safetensors header exceeds the configured limit")
            prefix_header_size = SAFETENSORS_LENGTH_PREFIX.size + header_size
            if prefix_header_size > raw_size:
                raise ValueError("safetensors header length exceeds the raw payload")
            header_view = raw_view[:prefix_header_size]
            try:
                plaintext = _validate_safetensors_header(
                    header_view, raw_size=raw_size, header_size=header_size
                )
            finally:
                header_view.release()
            data_view = raw_view[prefix_header_size:]

        if data_view.nbytes:
            try:
                frame = blosc2.compress2(data_view, cparams=_compression_params(compression))
            except Exception as exc:
                raise ValueError("could not compress tensor transport payload") from exc
        else:
            # SafeTensors may contain only empty tensors and a native blob may
            # be empty. An empty body has no frame and therefore needs no allocation.
            frame = b""

        envelope = TENSOR_PAYLOAD_ENVELOPE.pack(TENSOR_PAYLOAD_MAGIC, raw_size, header_size)
        return envelope + plaintext + frame
    finally:
        if data_view is not raw_view:
            data_view.release()
        raw_view.release()


def _inspect_tensor_payload_parts(
    packed: bytes | bytearray | memoryview,
    *,
    max_raw_bytes: int,
    safetensors: bool = True,
) -> tuple[int, bytes, memoryview, int]:
    """Validate the envelope, plaintext header, and compressed frame header."""
    if type(max_raw_bytes) is not int or max_raw_bytes < 0:
        raise ValueError("max_raw_bytes must be a non-negative integer")
    if type(safetensors) is not bool:
        raise TypeError("safetensors must be a bool")
    packed_view = _as_byte_view(packed, name="packed")
    frame: memoryview | None = None
    try:
        raw_size = declared_tensor_raw_size(packed_view, max_raw_bytes=max_raw_bytes)
        if packed_view.nbytes > max_encoded_size(raw_size):
            raise ValueError("tensor payload exceeds the configured encoded size limit")
        _, _, header_size = TENSOR_PAYLOAD_ENVELOPE.unpack_from(packed_view)

        payload_offset = TENSOR_PAYLOAD_ENVELOPE.size
        if safetensors:
            prefix_header_size = SAFETENSORS_LENGTH_PREFIX.size + header_size
            if header_size > MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError("safetensors header exceeds the configured limit")
            if prefix_header_size > raw_size or prefix_header_size > sys.maxsize:
                raise ValueError("safetensors header length exceeds the raw payload")
            if packed_view.nbytes < payload_offset + prefix_header_size:
                raise ValueError("safetensors plaintext header is truncated")
            header_view = packed_view[payload_offset : payload_offset + prefix_header_size]
            try:
                plaintext = _validate_safetensors_header(
                    header_view, raw_size=raw_size, header_size=header_size
                )
            finally:
                header_view.release()
            data_offset = payload_offset + prefix_header_size
            expected_frame_output = raw_size - prefix_header_size
        else:
            if header_size != 0:
                raise ValueError("native adapter payload must have a zero header size")
            plaintext = b""
            data_offset = payload_offset
            expected_frame_output = raw_size

        frame = packed_view[data_offset:]
        if expected_frame_output == 0:
            if frame.nbytes != 0:
                raise ValueError("empty tensor body must not contain a compressed frame")
            return raw_size, plaintext, frame, expected_frame_output
        if frame.nbytes == 0:
            raise ValueError("tensor payload is missing its compressed frame")

        try:
            declared_size, declared_frame_size, _ = blosc2.get_cbuffer_sizes(frame)
        except Exception as exc:
            raise ValueError("tensor payload has an invalid compressed frame") from exc
        if declared_size != expected_frame_output:
            raise ValueError("compressed frame output size does not match the envelope")
        if declared_size > min(max_raw_bytes, MAX_TENSOR_RAW_BYTES):
            raise ValueError("compressed frame declares an output above the raw size limit")
        if declared_frame_size != frame.nbytes:
            raise ValueError("compressed frame is truncated or has trailing bytes")

        return raw_size, plaintext, frame, expected_frame_output
    except BaseException:
        if frame is not None:
            frame.release()
        raise
    finally:
        packed_view.release()


def inspect_tensor_payload(
    packed: bytes | bytearray | memoryview,
    *,
    max_raw_bytes: int,
    safetensors: bool = True,
) -> int:
    """Validate transport structure and frame sizes without decompressing."""
    raw_size, _, frame, _ = _inspect_tensor_payload_parts(
        packed, max_raw_bytes=max_raw_bytes, safetensors=safetensors
    )
    frame.release()
    return raw_size


def decode_tensor_payload(
    packed: bytes | bytearray | memoryview,
    *,
    max_raw_bytes: int,
    safetensors: bool = True,
) -> bytes:
    """Decode a bounded tensor payload, checking the declared frame first."""
    _, plaintext, frame, expected_frame_output = _inspect_tensor_payload_parts(
        packed, max_raw_bytes=max_raw_bytes, safetensors=safetensors
    )
    try:
        if expected_frame_output == 0:
            return plaintext
        destination = bytearray(expected_frame_output)
        try:
            blosc2.decompress2(frame, dst=destination, nthreads=4)
        except Exception as exc:
            raise ValueError("tensor payload compressed frame is corrupt") from exc
        return plaintext + bytes(destination)
    finally:
        frame.release()


__all__ = [
    "MAX_FRAME_OVERHEAD",
    "MAX_SAFETENSORS_HEADER_BYTES",
    "MAX_TENSOR_RAW_BYTES",
    "TENSOR_PAYLOAD_CONTENT_TYPE",
    "TENSOR_PAYLOAD_ENVELOPE",
    "TENSOR_PAYLOAD_MAGIC",
    "TENSOR_PAYLOAD_TRANSPORT",
    "declared_tensor_raw_size",
    "decode_tensor_payload",
    "encode_tensor_payload",
    "inspect_tensor_payload",
    "max_encoded_size",
]
