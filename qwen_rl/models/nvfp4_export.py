"""CPU-capable, atomic exporter for native Qwen3.5 TorchAO NVFP4 artifacts."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .artifact import expected_nvfp4_skip_modules, read_nvfp4_artifact
from .artifacts import source_model_fingerprint
from .nvfp4 import quantize_nvfp4_linears


_ASSET_FILES = {
    "added_tokens.json",
    "audio_preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "image_processor_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "vocab.txt",
}
_TOKENIZER_FILES = {
    "added_tokens.json", "chat_template.jinja", "merges.txt", "special_tokens_map.json",
    "spiece.model", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "vocab.json", "vocab.txt", "generation_config.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors_dtype(dtype: Any) -> str:
    return {
        "torch.bool": "BOOL",
        "torch.uint8": "U8",
        "torch.int8": "I8",
        "torch.int16": "I16",
        "torch.int32": "I32",
        "torch.int64": "I64",
        "torch.float16": "F16",
        "torch.bfloat16": "BF16",
        "torch.float32": "F32",
        "torch.float64": "F64",
    }.get(str(dtype), "")


def _file_records(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]


def _read_source_mtp(source: Path, *, expected_count: int) -> tuple[dict[str, Any], dict[str, str]]:
    from safetensors import safe_open

    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file() or index_path.is_symlink():
        if expected_count == 0:
            single_file = source / "model.safetensors"
            if single_file.is_file() and not single_file.is_symlink():
                return {}, {}
        raise ValueError("NVFP4 export source must have a safetensors shard index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError("source model.safetensors.index.json has no weight_map")
    mtp_map = {
        name: shard for name, shard in weight_map.items()
        if isinstance(name, str) and name.startswith("mtp.")
    }
    if len(mtp_map) != expected_count:
        raise ValueError(
            f"NVFP4 export expected {expected_count} source MTP tensors, found {len(mtp_map)}"
        )
    tensors: dict[str, Any] = {}
    by_shard: dict[str, list[str]] = {}
    for name, shard in mtp_map.items():
        if not isinstance(shard, str) or Path(shard).is_absolute() or ".." in Path(shard).parts:
            raise ValueError(f"source MTP tensor references an invalid shard: {name}")
        by_shard.setdefault(shard, []).append(name)
    for shard, names in by_shard.items():
        shard_path = source / shard
        if shard_path.is_symlink() or not shard_path.is_file() or source not in shard_path.resolve().parents:
            raise ValueError(f"source MTP shard is missing or escapes the source: {shard}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in sorted(names):
                if name not in available:
                    raise ValueError(f"source MTP index references a missing tensor: {name}")
                tensors[name] = handle.get_tensor(name).detach().cpu().contiguous().clone()
    return tensors, mtp_map


def _flush_shard(
    stage: Path,
    shard_number: int,
    tensor_data: dict[str, Any],
    metadata: dict[str, str],
    weight_map: dict[str, str],
    total_size: int,
) -> tuple[int, int]:
    from safetensors.torch import save_file

    if not tensor_data:
        return shard_number, total_size
    shard_name = f"model-{shard_number:05d}.safetensors"
    save_file(tensor_data, str(stage / shard_name), metadata=metadata)
    for tensor_name in tensor_data:
        if tensor_name in weight_map:
            raise ValueError(f"duplicate flattened tensor key during NVFP4 export: {tensor_name}")
        weight_map[tensor_name] = shard_name
    total_size += sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensor_data.values())
    return shard_number + 1, total_size


def _serialize_model(
    model: Any,
    stage: Path,
    *,
    max_shard_bytes: int,
    mtp_tensors: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict

    state = model.state_dict()
    state_mtp_names = {name for name in state if name.startswith("mtp.")}
    if state_mtp_names and state_mtp_names != set(mtp_tensors):
        raise ValueError("loaded model MTP state differs from the preserved source MTP inventory")
    if state_mtp_names:
        for name in state_mtp_names:
            if not torch.equal(state[name].detach().cpu(), mtp_tensors[name]):
                raise ValueError(f"loaded model MTP tensor differs from source: {name}")

    weight_map: dict[str, str] = {}
    current_data: dict[str, Any] = {}
    current_metadata: dict[str, str] = {}
    current_bytes = 0
    shard_number = 1
    total_size = 0

    for logical_name in sorted(state):
        if logical_name.startswith("mtp."):
            continue
        flattened, logical_metadata = flatten_tensor_state_dict({logical_name: state[logical_name]})
        tensor_names = json.loads(logical_metadata.pop("tensor_names"))
        if tensor_names != [logical_name]:
            raise ValueError(f"TorchAO flatten returned inconsistent tensor_names for {logical_name}")
        group: dict[str, Any] = {}
        group_size = 0
        for name, value in flattened.items():
            if name in current_data or name in group or name in weight_map:
                raise ValueError(f"duplicate flattened tensor key during NVFP4 export: {name}")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"TorchAO flatten returned a non-tensor for {name}")
            tensor = value.detach().to(device="cpu", copy=True).contiguous()
            group[name] = tensor
            group_size += int(tensor.numel()) * int(tensor.element_size())
        if current_data and current_bytes + group_size > max_shard_bytes:
            shard_number, total_size = _flush_shard(
                stage, shard_number, current_data, current_metadata, weight_map, total_size
            )
            current_data, current_metadata, current_bytes = {}, {}, 0
        if current_bytes and any(name in current_metadata for name in logical_metadata):
            raise ValueError(f"duplicate TorchAO metadata key for {logical_name}")
        current_data.update(group)
        current_metadata.update(logical_metadata)
        current_metadata["tensor_names"] = json.dumps(
            json.loads(current_metadata.get("tensor_names", "[]")) + [logical_name]
        )
        current_bytes += group_size

    shard_number, total_size = _flush_shard(
        stage, shard_number, current_data, current_metadata, weight_map, total_size
    )
    if not weight_map:
        raise RuntimeError("NVFP4 model state serialization produced no tensor files")

    mtp_path: Path | None = None
    if mtp_tensors:
        mtp_path = stage / "model_mtp.safetensors"
        copied = {name: value.detach().cpu().contiguous().clone() for name, value in mtp_tensors.items()}
        save_file(copied, str(mtp_path), metadata={"format": "pt"})
        for name in copied:
            if name in weight_map:
                raise ValueError(f"MTP key collides with model key: {name}")
            weight_map[name] = mtp_path.name
        total_size += sum(int(value.numel()) * int(value.element_size()) for value in copied.values())

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (stage / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "weight_map": index["weight_map"],
        "shard_count": len(set(weight_map.values())),
        "total_size": total_size,
        "mtp_path": mtp_path.name if mtp_path is not None else None,
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(source, destination)
        return
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd, os.fsencode(source), at_fdcwd, os.fsencode(destination), rename_noreplace
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error, os.strerror(error), str(destination))


def export_nvfp4_checkpoint(
    source_model: str | Path,
    output_model: str | Path,
    *,
    max_shard_bytes: int = 2 * 1024**3,
    cpu_threads: int = 4,
    expected_mtp_tensor_count: int = 15,
) -> dict[str, Any]:
    """Export a local dense Qwen3.5 model as a validated TorchAO NVFP4 artifact."""
    if isinstance(max_shard_bytes, bool) or not isinstance(max_shard_bytes, int) or max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be a positive integer")
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads <= 0:
        raise ValueError("cpu_threads must be a positive integer")
    if (isinstance(expected_mtp_tensor_count, bool) or not isinstance(expected_mtp_tensor_count, int)
            or expected_mtp_tensor_count < 0):
        raise ValueError("expected_mtp_tensor_count must be a non-negative integer")
    source = Path(source_model).expanduser().resolve(strict=True)
    requested_source = Path(source_model).expanduser()
    if requested_source.is_symlink() or not source.is_dir():
        raise ValueError("NVFP4 source must be a local, non-symlink model directory")
    destination = Path(output_model).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"NVFP4 output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("NVFP4 output parent may not be a symlink")
    destination = destination.parent.resolve() / destination.name
    if source == destination or source in destination.parents:
        raise ValueError("NVFP4 output must be outside the source model directory")

    lock_path = destination.parent / f".{destination.name}.nvfp4.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage: Path | None = None
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        os.fsync(lock_fd)
        os.close(lock_fd)
        lock_fd = -1
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"NVFP4 output already exists: {destination}")

        base_fingerprint, source_manifest = source_model_fingerprint(str(source))
        mtp_tensors, mtp_map = _read_source_mtp(source, expected_count=expected_mtp_tensor_count)
        if any(not _safetensors_dtype(tensor.dtype) for tensor in mtp_tensors.values()):
            raise ValueError("source MTP inventory has an unsupported safetensors dtype")
        source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        skips = expected_nvfp4_skip_modules(source_config)

        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))

        import torch
        from transformers import Qwen3_5ForConditionalGeneration, TorchAoConfig
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig

        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(min(2, cpu_threads))
        except RuntimeError:
            pass
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(source),
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
            local_files_only=True,
            trust_remote_code=True,
        )
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise RuntimeError("NVFP4 export model unexpectedly loaded a non-CPU parameter")
        model.config.quantization_config = TorchAoConfig(
            NVFP4WeightOnlyConfig(), modules_to_not_convert=skips
        ).to_dict()
        model.config.torch_dtype = torch.bfloat16
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "torch_dtype"):
            text_config.torch_dtype = torch.bfloat16
        quantization_report = quantize_nvfp4_linears(
            model, modules_to_not_convert=skips
        )
        if set(quantization_report["fp32_gdn_modules"]) != {
            name for name in skips if name.endswith(("in_proj_a", "in_proj_b"))
            and name in dict(model.named_modules())
        }:
            raise RuntimeError("quantized GDN skip inventory differs from loaded Qwen modules")
        if set(state_name for state_name in model.state_dict() if state_name.startswith("mtp.")):
            raise RuntimeError("Transformers model instantiated standalone MTP keys; refusing to quantize them")

        model.config.save_pretrained(stage)
        for asset in sorted(_ASSET_FILES):
            source_asset = source / asset
            if source_asset.is_file() and not source_asset.is_symlink():
                shutil.copy2(source_asset, stage / asset)

        serialization = _serialize_model(
            model, stage, max_shard_bytes=max_shard_bytes, mtp_tensors=mtp_tensors
        )
        if expected_mtp_tensor_count > 0 and serialization["mtp_path"] is None:
            raise RuntimeError("source MTP tensors were not preserved in a dedicated sidecar")

        quantization_report.update({
            "algorithm": "TorchAO",
            "quantizer": "NVFP4WeightOnlyConfig",
            "bits": 4,
            "compute_dtype": "torch.bfloat16",
            "packed_resident": True,
            "source_fingerprint": base_fingerprint,
            "modules_to_not_convert": skips,
            "mtp_preservation": {
                "quantized": False,
                "runtime_enabled": False,
                "tensor_count": len(mtp_tensors),
                "path": serialization["mtp_path"],
            },
            "serialized_shard_count": serialization["shard_count"],
            "serialized_total_size": serialization["total_size"],
        })
        report_path = stage / "quantization_report.json"
        report_path.write_text(
            json.dumps(quantization_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        weight_paths = sorted(stage.glob("*.safetensors"))
        tokenizer_paths = sorted(
            path for path in stage.iterdir() if path.name in _TOKENIZER_FILES and path.is_file()
        )
        if not any(path.name in {"tokenizer.json", "vocab.json", "tokenizer.model"} for path in tokenizer_paths):
            raise RuntimeError("source model has no loadable tokenizer vocabulary")
        auxiliary_paths = sorted(
            path for path in stage.iterdir()
            if path.is_file() and path.name not in {
                "config.json", "model.safetensors.index.json", "source_fingerprint.json"
            } and path.suffix != ".safetensors"
        )
        lineage = {
            "base_fingerprint": base_fingerprint,
            "source_model_path": str(source),
            "source_manifest": source_manifest,
            "source_weight_files": source_manifest["weight_files"],
            "quantization_method": "torchao",
            "quantizer": "NVFP4WeightOnlyConfig",
            "modules_to_not_convert": skips,
            "quantized_weight_files": _file_records(stage, weight_paths),
            "tokenizer_files": _file_records(stage, tokenizer_paths),
            "auxiliary_files": _file_records(stage, auxiliary_paths),
            "packed_module_count": quantization_report["packed_module_count"],
            "packed_uint8_tensor_count": quantization_report["packed_uint8_tensor_count"],
            "packed_uint8_numel": quantization_report["packed_uint8_numel"],
        }
        if mtp_tensors:
            lineage["mtp_preservation"] = {
                "path": serialization["mtp_path"],
                "tensor_count": len(mtp_tensors),
                "tensors": [
                    {
                        "name": name,
                        "dtype": _safetensors_dtype(tensor.dtype),
                        "shape": list(tensor.shape),
                    }
                    for name, tensor in sorted(mtp_tensors.items())
                ],
                "source_shards": sorted(set(mtp_map.values())),
                "quantized": False,
                "runtime_enabled": False,
            }
        (stage / "source_fingerprint.json").write_text(
            json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        after_fingerprint, after_manifest = source_model_fingerprint(str(source))
        if after_fingerprint != base_fingerprint or after_manifest != source_manifest:
            raise RuntimeError("NVFP4 source model changed during export")
        _, _, identity = read_nvfp4_artifact(stage)
        _rename_noreplace(stage, destination)
        stage = None
        return {
            "output_model": str(destination),
            "base_fingerprint": base_fingerprint,
            "artifact_fingerprint": identity["artifact_fingerprint"],
            "packed_module_count": identity["artifact_manifest"]["packed_module_count"],
            "packed_uint8_tensor_count": identity["artifact_manifest"]["packed_uint8_tensor_count"],
            "packed_uint8_numel": identity["artifact_manifest"]["packed_uint8_numel"],
            "mtp_tensor_count": len(mtp_tensors),
            "serialized_shard_count": serialization["shard_count"],
        }
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--max-shard-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--expected-mtp-tensor-count", type=int, default=15)
    args = parser.parse_args()
    result = export_nvfp4_checkpoint(
        args.source_model,
        args.output_model,
        max_shard_bytes=args.max_shard_bytes,
        cpu_threads=args.cpu_threads,
        expected_mtp_tensor_count=args.expected_mtp_tensor_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
