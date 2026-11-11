"""Read-only, bounded-range access to NInfer-v3 T2 tensors for the Bonsai Learner.

This module reads only the small v3 directory at open and then uses ``pread`` for requested
row ranges. It never loads an artifact payload wholesale and it does not depend on the NInfer
converter's Python environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from math import prod
from pathlib import Path
from types import TracebackType
from typing import Any

import torch

from .ternary import FrozenTernaryEmbedding, FrozenTernaryLinear, dequantize_t2

_HEADER = struct.Struct("<8sQ16s")
_MAGIC = b"NINFER\x00\x03"
_PAYLOAD_ALIGNMENT = 4096
_PLANE_ALIGNMENT = 256
_T2_FORMAT = "t2_g128_fp16"
_T2_LAYOUT = "row_split_k128_v1"
_Q8_FORMAT = "q8_g32_fp16"
_Q8_LAYOUT = "row_split_k128_v1"
_DIRECT_DTYPES = {
    "bf16": (torch.uint16, 2),
    "fp32": (torch.float32, 4),
    "int32": (torch.int32, 4),
}


class TernaryArtifactError(ValueError):
    """An NInfer-v3 artifact cannot satisfy the frozen T2 reader contract."""


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _read_exact(fd: int, size: int, offset: int, label: str) -> bytes:
    data = os.pread(fd, size, offset)
    if len(data) != size:
        raise TernaryArtifactError(f"{label}: expected {size} bytes, got {len(data)}")
    return data


class TernaryArtifactStore:
    """Read NInfer-v3 T2 matrices and sign auxiliaries without dense model materialization."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            size = os.fstat(self._fd).st_size
            header = _read_exact(self._fd, _HEADER.size, 0, "NInfer-v3 header")
            magic, json_bytes, self.artifact_id = _HEADER.unpack(header)
            if magic != _MAGIC:
                raise TernaryArtifactError(f"{self.path}: expected an NInfer-v3 artifact")
            if json_bytes <= 0 or json_bytes > size - _HEADER.size:
                raise TernaryArtifactError(f"{self.path}: invalid NInfer directory length {json_bytes}")
            directory_bytes = _read_exact(self._fd, json_bytes, _HEADER.size, "NInfer directory")
            try:
                directory = json.loads(directory_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise TernaryArtifactError(f"{self.path}: malformed NInfer directory") from error
            if not isinstance(directory, dict):
                raise TernaryArtifactError(f"{self.path}: NInfer directory must be an object")
            self.payload_offset = _align_up(_HEADER.size + json_bytes, _PAYLOAD_ALIGNMENT)
            files = directory.get("files")
            if not isinstance(files, list) or len(files) != 1:
                raise TernaryArtifactError("T2 loader requires one self-contained NInfer entry file")
            payload_bytes = files[0].get("payload_bytes") if isinstance(files[0], dict) else None
            if type(payload_bytes) is not int or payload_bytes <= 0:
                raise TernaryArtifactError("NInfer file table has an invalid payload size")
            if self.payload_offset + payload_bytes != size:
                raise TernaryArtifactError(
                    f"{self.path}: entry file length differs from its declared payload "
                    f"({size} vs {_HEADER.size}+{json_bytes}+padding+{payload_bytes})"
                )
            self.payload_bytes = payload_bytes
            self.directory = directory
            self.components = self._mapping(directory.get("components"), "components")
            self.bindings = self._mapping(directory.get("bindings"), "bindings")
            self._uses = directory.get("uses")
            if not isinstance(self._uses, list):
                raise TernaryArtifactError("NInfer directory has no Uses list")
            self._objects = self._index_objects(directory.get("objects"))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _mapping(value: Any, label: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TernaryArtifactError(f"NInfer directory {label} must be an object")
        return value

    @staticmethod
    def _index_objects(value: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(value, list):
            raise TernaryArtifactError("NInfer directory has no object list")
        result: dict[str, dict[str, Any]] = {}
        for record in value:
            if not isinstance(record, dict):
                raise TernaryArtifactError("NInfer object record must be an object")
            object_id = record.get("id")
            if not isinstance(object_id, str) or not object_id or object_id in result:
                raise TernaryArtifactError("NInfer object IDs must be unique nonempty strings")
            result[object_id] = record
        return result

    @property
    def text_config(self) -> dict[str, Any]:
        text = self.components.get("text")
        config = text.get("config") if isinstance(text, dict) else None
        if not isinstance(config, dict):
            raise TernaryArtifactError("NInfer artifact has no text model config")
        return config

    @property
    def text_resources(self) -> dict[str, str]:
        text = self.components.get("text")
        resources = text.get("resources") if isinstance(text, dict) else None
        if not isinstance(resources, dict):
            raise TernaryArtifactError("NInfer artifact has no text resource map")
        if any(not isinstance(name, str) or not isinstance(object_id, str) for name, object_id in resources.items()):
            raise TernaryArtifactError("NInfer text resource map contains an invalid entry")
        return resources

    def read_component_resource(self, component: str, name: str) -> bytes:
        component_record = self.components.get(component)
        resources = component_record.get("resources") if isinstance(component_record, dict) else None
        if not isinstance(resources, dict):
            raise TernaryArtifactError(f"NInfer artifact has no resources for component {component!r}")
        object_id = resources.get(name)
        if not isinstance(object_id, str):
            raise TernaryArtifactError(f"NInfer artifact has no {component} resource {name!r}")
        return self.read_resource(object_id)

    def text_resource_hashes(self) -> dict[str, dict[str, Any]]:
        """Return exact SHA256 and byte size for each embedded text resource."""

        result = {}
        for name, object_id in sorted(self.text_resources.items()):
            if Path(name).name != name:
                raise TernaryArtifactError(f"unsafe text resource filename {name!r}")
            raw = self.read_resource(object_id)
            result[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return result

    def materialize_text_resources(self, destination: str | Path) -> dict[str, dict[str, Any]]:
        """Copy embedded tokenizer/template files to an explicit directory for HF loading.

        Existing files are accepted only when their bytes match the artifact. New files are
        atomically replaced from same-directory temporaries; no path is written implicitly.
        """

        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        result = {}
        for name, object_id in sorted(self.text_resources.items()):
            if Path(name).name != name or name in ("", ".", ".."):
                raise TernaryArtifactError(f"unsafe text resource filename {name!r}")
            target = root / name
            raw = self.read_resource(object_id)
            digest = hashlib.sha256(raw).hexdigest()
            if target.is_symlink():
                raise TernaryArtifactError(f"refusing to replace symlinked tokenizer resource {target}")
            if target.exists():
                if not target.is_file() or target.read_bytes() != raw:
                    raise TernaryArtifactError(f"existing tokenizer resource differs from the artifact: {target}")
            else:
                fd, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=root)
                try:
                    with os.fdopen(fd, "wb") as output:
                        output.write(raw)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, target)
                except BaseException:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
                    raise
            result[name] = {"path": str(target), "bytes": len(raw), "sha256": digest}
        return result

    def _object(self, object_id: str) -> dict[str, Any]:
        try:
            return self._objects[object_id]
        except KeyError as error:
            raise TernaryArtifactError(f"NInfer artifact has no object {object_id!r}") from error

    def _read_payload(self, offset: int, size: int, label: str) -> bytes:
        if type(offset) is not int or type(size) is not int or offset < 0 or size < 0:
            raise TernaryArtifactError(f"{label}: invalid payload range")
        if offset + size > self.payload_bytes:
            raise TernaryArtifactError(f"{label}: range extends beyond declared payload")
        return _read_exact(self._fd, size, self.payload_offset + offset, label)

    def _parts(self, parameter: str) -> list[tuple[dict[str, Any], int, int]]:
        try:
            binding = self.bindings[parameter]
        except KeyError as error:
            raise TernaryArtifactError(f"NInfer artifact has no parameter {parameter!r}") from error
        if not isinstance(binding, dict):
            raise TernaryArtifactError(f"{parameter}: binding must be an object")
        raw_parts = binding.get("parts")
        if raw_parts is None:
            object_id = binding.get("object")
            if not isinstance(object_id, str):
                raise TernaryArtifactError(f"{parameter}: binding must name an object or parts")
            obj = self._object(object_id)
            shape = obj.get("shape")
            if not isinstance(shape, list) or any(type(dim) is not int or dim <= 0 for dim in shape):
                raise TernaryArtifactError(f"{parameter}: bound object has an invalid shape")
            bounds = binding.get("range", [0, prod(shape)])
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise TernaryArtifactError(f"{parameter}: direct binding range is malformed")
            raw_parts = [{"object": object_id, "range": bounds}]
        if not isinstance(raw_parts, list):
            raise TernaryArtifactError(f"{parameter}: binding parts must be a list")
        parts = []
        for part in raw_parts:
            if not isinstance(part, dict) or not isinstance(part.get("range"), list):
                raise TernaryArtifactError(f"{parameter}: malformed binding part")
            bounds = part["range"]
            if len(bounds) != 2 or any(type(v) is not int for v in bounds):
                raise TernaryArtifactError(f"{parameter}: binding range must be two integers")
            begin, end = bounds
            obj = self._object(part.get("object"))
            parts.append((obj, begin, end))
        if not parts:
            raise TernaryArtifactError(f"{parameter}: binding has no storage parts")
        return parts

    @staticmethod
    def _t2_geometry(obj: dict[str, Any]) -> tuple[int, int, int, int, int]:
        if obj.get("kind") != "tensor" or obj.get("format") != _T2_FORMAT or obj.get("layout") != _T2_LAYOUT:
            raise TernaryArtifactError(
                f"{obj.get('id')}: expected {_T2_FORMAT}/{_T2_LAYOUT} tensor"
            )
        shape = obj.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape):
            raise TernaryArtifactError(f"{obj.get('id')}: T2 shape must be positive rank-2 [N,K]")
        n, k = shape
        if k % 128:
            raise TernaryArtifactError(f"{obj.get('id')}: Bonsai T2 K={k} must be divisible by 128")
        groups = k // 128
        code_row_bytes = groups * 32
        scale_row_bytes = groups * 2
        base_bytes = n * code_row_bytes
        scale_offset = _align_up(base_bytes, _PLANE_ALIGNMENT)
        expected_bytes = scale_offset + n * scale_row_bytes
        if obj.get("bytes") != expected_bytes:
            raise TernaryArtifactError(
                f"{obj.get('id')}: T2 payload has {obj.get('bytes')} bytes, expected {expected_bytes}"
            )
        offset = obj.get("offset")
        if type(offset) is not int or offset < 0 or offset % _PLANE_ALIGNMENT:
            raise TernaryArtifactError(f"{obj.get('id')}: T2 object offset is not 256-byte aligned")
        return n, k, code_row_bytes, scale_row_bytes, scale_offset

    def read_t2_parameter(
        self,
        parameter: str,
        *,
        expected_shape: tuple[int, int] | None = None,
        rows: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read T2 rows as packed uint8 codes and raw uint16 scales.

        ``rows`` selects a half-open range in the logical parameter, which permits small
        artifact parity checks without reading the full matrix.
        """

        code_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        input_width: int | None = None
        total_rows = 0
        selected_begin, selected_end = rows if rows is not None else (0, None)
        if type(selected_begin) is not int or (selected_end is not None and type(selected_end) is not int):
            raise TypeError("T2 row bounds must be integers")
        if selected_begin < 0 or (selected_end is not None and selected_end <= selected_begin):
            raise ValueError("T2 row range must be nonempty and nonnegative")
        for obj, begin, end in self._parts(parameter):
            n, k, code_row_bytes, scale_row_bytes, scale_offset = self._t2_geometry(obj)
            if begin < 0 or end <= begin or end > n * k or begin % k or end % k:
                raise TernaryArtifactError(
                    f"{parameter}: binding range [{begin},{end}) is not a whole in-bounds row span"
                )
            if input_width is None:
                input_width = k
            elif input_width != k:
                raise TernaryArtifactError(f"{parameter}: bound T2 parts use different input widths")
            first, last = begin // k, end // k
            part_rows = last - first
            part_begin = max(selected_begin, total_rows)
            part_end = min(selected_end if selected_end is not None else total_rows + part_rows, total_rows + part_rows)
            if part_begin < part_end:
                storage_first = first + part_begin - total_rows
                rows_in_part = part_end - part_begin
                object_offset = obj["offset"]
                code_bytes = self._read_payload(
                    object_offset + storage_first * code_row_bytes,
                    rows_in_part * code_row_bytes,
                    f"{parameter} packed codes",
                )
                scale_bytes = self._read_payload(
                    object_offset + scale_offset + storage_first * scale_row_bytes,
                    rows_in_part * scale_row_bytes,
                    f"{parameter} scale words",
                )
                # frombuffer retains its private writable bytearray owner. A second
                # clone copies the full payload without adding lifetime or isolation.
                code_parts.append(
                    torch.frombuffer(bytearray(code_bytes), dtype=torch.uint8)
                    .reshape(rows_in_part, code_row_bytes)
                )
                scale_parts.append(
                    torch.frombuffer(bytearray(scale_bytes), dtype=torch.uint16)
                    .reshape(rows_in_part, k // 128)
                )
            total_rows += part_rows

        if input_width is None:
            raise TernaryArtifactError(f"{parameter}: binding has no T2 rows")
        if selected_end is None:
            selected_end = total_rows
        if selected_end > total_rows or selected_begin >= selected_end:
            raise TernaryArtifactError(
                f"{parameter}: row range [{selected_begin},{selected_end}) exceeds {total_rows} rows"
            )
        selected_rows = selected_end - selected_begin
        shape = (selected_rows, input_width)
        if expected_shape is not None and tuple(expected_shape) != shape:
            raise TernaryArtifactError(
                f"{parameter}: decoded shape {shape} differs from expected {tuple(expected_shape)}"
            )
        codes = code_parts[0] if len(code_parts) == 1 else torch.cat(code_parts, dim=0)
        scales = scale_parts[0] if len(scale_parts) == 1 else torch.cat(scale_parts, dim=0)
        if codes.shape[0] != selected_rows:
            raise TernaryArtifactError(f"{parameter}: row selection left a storage gap")
        return codes, scales

    def read_q8_parameter(
        self,
        parameter: str,
        *,
        expected_shape: tuple[int, int] | None = None,
        rows: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read MTP Q8 rows as signed int8 codes and raw uint16 FP16 scale words.

        Bonsai's text causal LM is T2/BF16/FP32. The bundled ``q8_g32_fp16`` tensors belong to
        the separate MTP component; this method preserves that boundary while exposing their
        exact packed values to a future MTP-specific loader.
        """

        code_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        input_width: int | None = None
        total_rows = 0
        selected_begin, selected_end = rows if rows is not None else (0, None)
        if type(selected_begin) is not int or (selected_end is not None and type(selected_end) is not int):
            raise TypeError("Q8 row bounds must be integers")
        if selected_begin < 0 or (selected_end is not None and selected_end <= selected_begin):
            raise ValueError("Q8 row range must be nonempty and nonnegative")
        for obj, begin, end in self._parts(parameter):
            if obj.get("kind") != "tensor" or obj.get("format") != _Q8_FORMAT or obj.get("layout") != _Q8_LAYOUT:
                raise TernaryArtifactError(f"{parameter}: expected {_Q8_FORMAT}/{_Q8_LAYOUT} tensor")
            shape = obj.get("shape")
            if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
                raise TernaryArtifactError(f"{parameter}: Q8 shape must be positive rank-2 [N,K]")
            n, k = shape
            if k % 32:
                raise TernaryArtifactError(f"{parameter}: Q8 K={k} must be divisible by 32")
            base_row_bytes = k
            scale_row_bytes = k // 32 * 2
            base_bytes = n * base_row_bytes
            scale_offset = _align_up(base_bytes, _PLANE_ALIGNMENT)
            expected_bytes = scale_offset + n * scale_row_bytes
            if obj.get("bytes") != expected_bytes:
                raise TernaryArtifactError(f"{parameter}: Q8 payload size does not match its shape")
            if begin < 0 or end <= begin or end > n * k or begin % k or end % k:
                raise TernaryArtifactError(f"{parameter}: Q8 binding range is not a whole row span")
            if input_width is None:
                input_width = k
            elif input_width != k:
                raise TernaryArtifactError(f"{parameter}: Q8 parts use different input widths")
            first, last = begin // k, end // k
            part_rows = last - first
            part_begin = max(selected_begin, total_rows)
            part_end = min(selected_end if selected_end is not None else total_rows + part_rows, total_rows + part_rows)
            if part_begin < part_end:
                storage_first = first + part_begin - total_rows
                count = part_end - part_begin
                object_offset = obj.get("offset", -1)
                code_raw = self._read_payload(
                    object_offset + storage_first * base_row_bytes,
                    count * base_row_bytes,
                    f"{parameter} Q8 codes",
                )
                scale_raw = self._read_payload(
                    object_offset + scale_offset + storage_first * scale_row_bytes,
                    count * scale_row_bytes,
                    f"{parameter} Q8 scales",
                )
                codes = torch.frombuffer(bytearray(code_raw), dtype=torch.uint8).reshape(count, k).clone()
                for row in range(0, count, 8192):
                    if bool((codes[row : row + 8192] == 128).any()):
                        raise TernaryArtifactError(f"{parameter}: Q8 code -128 is outside the stored range")
                scales = torch.frombuffer(bytearray(scale_raw), dtype=torch.uint16).reshape(count, k // 32).clone()
                if not bool(torch.isfinite(scales.view(torch.float16)).all()):
                    raise TernaryArtifactError(f"{parameter}: Q8 scales must be finite FP16 values")
                code_parts.append(codes.view(torch.int8))
                scale_parts.append(scales)
            total_rows += part_rows
        if input_width is None:
            raise TernaryArtifactError(f"{parameter}: binding has no Q8 rows")
        if selected_end is None:
            selected_end = total_rows
        if selected_end > total_rows or selected_begin >= selected_end:
            raise TernaryArtifactError(f"{parameter}: Q8 row range exceeds {total_rows} rows")
        shape = (selected_end - selected_begin, input_width)
        if expected_shape is not None and tuple(expected_shape) != shape:
            raise TernaryArtifactError(f"{parameter}: decoded shape {shape} differs from {tuple(expected_shape)}")
        codes = code_parts[0] if len(code_parts) == 1 else torch.cat(code_parts, dim=0)
        scales = scale_parts[0] if len(scale_parts) == 1 else torch.cat(scale_parts, dim=0)
        if codes.shape[0] != shape[0]:
            raise TernaryArtifactError(f"{parameter}: Q8 row selection left a storage gap")
        return codes, scales

    def load_linear(
        self,
        parameter: str,
        *,
        expected_shape: tuple[int, int] | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.bfloat16,
        primal: bool = False,
    ) -> FrozenTernaryLinear:
        if type(primal) is not bool:
            raise TypeError("primal must be boolean")
        if primal and self.input_basis(parameter) != "primal":
            raise TernaryArtifactError("cannot load a rotated Use as primal")
        codes, scales = self.read_t2_parameter(parameter, expected_shape=expected_shape)
        if primal:
            from .ternary_recovery import PrimalTernaryLinear
            return PrimalTernaryLinear(codes, scales, output_dtype=output_dtype, compute_dtype=compute_dtype)
        signs = self.read_hadamard_signs(parameter, codes.shape[1] * 4)
        return FrozenTernaryLinear(
            codes,
            scales,
            signs,
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
        )

    def dequantize_t2_rows(
        self,
        parameter: str,
        row_begin: int,
        row_end: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Decode a bounded T2 row slice, useful for vocabulary-chunked log-probability heads."""

        codes, scales = self.read_t2_parameter(parameter, rows=(row_begin, row_end))
        return dequantize_t2(codes, scales, dtype=dtype)

    def load_fused_linear(
        self,
        parameters: list[str] | tuple[str, ...],
        *,
        expected_shape: tuple[int, int],
        layout: str = "concat",
        head_size: int | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.bfloat16,
        primal: bool = False,
    ) -> FrozenTernaryLinear:
        """Combine T2 logical matrices into an HF fused projection without dequantizing.

        ``concat`` preserves parameter row order, used for GDN Q/K/V. ``head_interleave``
        places each head's full query rows before its full gate rows, matching Qwen3.5's
        ``view(..., num_heads, 2 * head_dim).chunk(2, dim=-1)`` split.
        """

        if type(primal) is not bool:
            raise TypeError("primal must be boolean")
        if primal and any(self.input_basis(name) != "primal" for name in parameters):
            raise TernaryArtifactError("cannot load rotated fused Uses as primal")
        if not parameters:
            raise ValueError("fused T2 linear requires at least one logical parameter")
        if layout not in ("concat", "head_interleave"):
            raise ValueError("T2 fused layout must be 'concat' or 'head_interleave'")
        if layout == "head_interleave" and (len(parameters) != 2 or type(head_size) is not int or head_size <= 0):
            raise ValueError("head_interleave requires two matrices and a positive head_size")
        matrices = [self.read_t2_parameter(name) for name in parameters]
        input_widths = {codes.shape[1] * 4 for codes, _ in matrices}
        if len(input_widths) != 1:
            raise TernaryArtifactError("fused T2 parts use different input widths")
        widths = input_widths.pop()
        if not primal:
            sign_vectors = [self.read_hadamard_signs(name, widths) for name in parameters]
            if any(not torch.equal(sign_vectors[0], signs) for signs in sign_vectors[1:]):
                raise TernaryArtifactError("fused T2 parts use different Hadamard input bases")
        if layout == "concat":
            packed_codes = torch.cat([codes for codes, _ in matrices], dim=0)
            scale_words = torch.cat([scales for _, scales in matrices], dim=0)
        else:
            rows = matrices[0][0].shape[0]
            if any(codes.shape[0] != rows for codes, _ in matrices):
                raise TernaryArtifactError("head-interleaved T2 parts must have equal row counts")
            if rows % head_size:
                raise TernaryArtifactError(f"T2 rows {rows} are not divisible by head_size {head_size}")
            head_count = rows // head_size
            packed_codes = torch.stack(
                [codes.reshape(head_count, head_size, -1) for codes, _ in matrices], dim=1
            ).reshape(head_count * head_size * len(matrices), -1)
            scale_words = torch.stack(
                [scales.reshape(head_count, head_size, -1) for _, scales in matrices], dim=1
            ).reshape(head_count * head_size * len(matrices), -1)
        if (packed_codes.shape[0], widths) != tuple(expected_shape):
            raise TernaryArtifactError(
                f"fused T2 shape {(packed_codes.shape[0], widths)} differs from {tuple(expected_shape)}"
            )
        if primal:
            from .ternary_recovery import PrimalTernaryLinear
            return PrimalTernaryLinear(packed_codes, scale_words, output_dtype=output_dtype,
                                        compute_dtype=compute_dtype)
        return FrozenTernaryLinear(
            packed_codes,
            scale_words,
            sign_vectors[0],
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
        )

    def read_resource(self, object_id: str) -> bytes:
        obj = self._object(object_id)
        if obj.get("kind") != "resource" or obj.get("encoding") != "raw_bytes_v1":
            raise TernaryArtifactError(f"{object_id}: expected a raw_bytes_v1 resource")
        offset, size = obj.get("offset"), obj.get("bytes")
        return self._read_payload(offset, size, object_id)

    def input_basis(self, parameter: str) -> str:
        """Read explicit Use semantics; absence of rotation means primal, not H(ones*x)."""
        uses = [u for u in self._uses if isinstance(u, dict) and u.get("parameter") == parameter]
        if not uses:
            raise TernaryArtifactError(f"{parameter}: no Uses")
        modes = set()
        for use in uses:
            aux = use.get("auxiliaries", {})
            if not isinstance(aux, dict) or set(aux) - {"hadamard_signs"}:
                raise TernaryArtifactError(f"{parameter}: unsupported input transform")
            modes.add("hadamard1024_signed" if "hadamard_signs" in aux else "primal")
        if len(modes) != 1:
            raise TernaryArtifactError(f"{parameter}: Uses disagree on input basis")
        return modes.pop()

    def read_hadamard_signs(self, use_parameter: str, width: int) -> torch.Tensor:
        """Read BF16 +/-1 signs from the selected Use's ``hadamard_signs`` tensor."""

        object_ids = {
            sign_record.get("object")
            for use in self._uses
            if isinstance(use, dict)
            and use.get("parameter") == use_parameter
            and isinstance(use.get("auxiliaries"), dict)
            and isinstance((sign_record := use["auxiliaries"].get("hadamard_signs")), dict)
            and isinstance(sign_record.get("object"), str)
        }
        if not object_ids:
            raise TernaryArtifactError(f"{use_parameter}: Use has no hadamard_signs resource")
        raw_values = []
        for object_id in sorted(object_ids):
            obj = self._object(object_id)
            if (
                obj.get("kind") != "tensor"
                or obj.get("format") != "bf16"
                or obj.get("layout") != "contiguous_le_v1"
                or obj.get("shape") != [width]
                or obj.get("bytes") != width * 2
            ):
                raise TernaryArtifactError(
                    f"{use_parameter}: Hadamard sign auxiliary must be BF16 contiguous tensor [{width}]"
                )
            raw_values.append(self._read_payload(obj["offset"], obj["bytes"], object_id))
        if any(raw != raw_values[0] for raw in raw_values[1:]):
            raise TernaryArtifactError(f"{use_parameter}: Uses disagree on the Hadamard sign vector")
        raw = raw_values[0]
        words = torch.frombuffer(bytearray(raw), dtype=torch.uint16).clone()
        signs = words.view(torch.bfloat16)
        if not bool(torch.all((signs == 1) | (signs == -1))):
            raise TernaryArtifactError(f"{use_parameter}: Hadamard signs are not BF16 +/-1")
        return signs

    def load_embedding(
        self,
        parameter: str = "text/token_embedding",
        *,
        signs_parameter: str = "text/output_head",
        expected_shape: tuple[int, int] | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        padding_idx: int | None = None,
    ) -> FrozenTernaryEmbedding:
        codes, scales = self.read_t2_parameter(parameter, expected_shape=expected_shape)
        width = codes.shape[1] * 4
        signs = self.read_hadamard_signs(signs_parameter, width)
        return FrozenTernaryEmbedding(
            codes,
            scales,
            signs,
            padding_idx=padding_idx,
            output_dtype=output_dtype,
        )

    def read_direct_parameter(
        self,
        parameter: str,
        *,
        expected_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Read a direct BF16/FP32/INT32 parameter, without decoding grouped weights."""

        pieces = []
        source_dtype: torch.dtype | None = None
        for obj, begin, end in self._parts(parameter):
            fmt = obj.get("format")
            if fmt not in _DIRECT_DTYPES or obj.get("kind") != "tensor":
                raise TernaryArtifactError(f"{parameter}: {fmt!r} is not a supported direct format")
            current_dtype, current_bytes = _DIRECT_DTYPES[fmt]
            shape = obj.get("shape")
            if (
                obj.get("layout") != "contiguous_le_v1"
                or not isinstance(shape, list)
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or obj.get("bytes") != prod(shape) * current_bytes
            ):
                raise TernaryArtifactError(f"{parameter}: direct object geometry does not match its format")
            if source_dtype is None:
                source_dtype = current_dtype
            elif source_dtype != current_dtype:
                raise TernaryArtifactError(f"{parameter}: direct parts use different numeric formats")
            if type(begin) is not int or type(end) is not int or begin < 0 or end <= begin or end > prod(shape):
                raise TernaryArtifactError(f"{parameter}: invalid direct element range")
            count = end - begin
            raw = self._read_payload(
                obj.get("offset", -1) + begin * current_bytes,
                count * current_bytes,
                parameter,
            )
            tensor = torch.frombuffer(bytearray(raw), dtype=current_dtype).clone()
            if fmt == "bf16":
                tensor = tensor.view(torch.bfloat16)
            pieces.append(tensor)
        flat = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
        if expected_shape is not None:
            if flat.numel() != prod(expected_shape):
                raise TernaryArtifactError(
                    f"{parameter}: {flat.numel()} values do not fit shape {tuple(expected_shape)}"
                )
            flat = flat.reshape(expected_shape)
        if dtype is not None:
            flat = flat.to(dtype=dtype)
        return flat

    def close(self) -> None:
        fd = getattr(self, "_fd", -1)
        if fd >= 0:
            os.close(fd)
            self._fd = -1

    def __enter__(self) -> TernaryArtifactStore:  # noqa: PYI034 -- typing.Self is unavailable on supported Python 3.10
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["TernaryArtifactError", "TernaryArtifactStore"]
