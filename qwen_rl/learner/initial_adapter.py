"""LoRA-only initialization with structural checks and a new optimizer lineage."""
from pathlib import Path
import json


def read_initial_adapter(path, *, base_sha256, lora_rank, lora_alpha):
    from safetensors import SafetensorError
    from safetensors.torch import load_file
    directory = Path(path)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("initial adapter directory is missing or unsafe")
    provenance = directory / "provenance.json"
    if provenance.is_symlink() or not provenance.is_file():
        raise ValueError("initial adapter provenance is missing or unsafe")
    record = json.loads(provenance.read_text())
    if (record.get("schema") != "qwen-rl-initial-adapter/v1"
            or record.get("base_sha256") != base_sha256
            or record.get("optimizer") != "new"):
        raise ValueError("initial adapter provenance/base identity is invalid")
    files = record.get("files")
    if not isinstance(files, dict) or set(files) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("initial adapter requires a complete file inventory")
    sizes = {}
    for name in files:
        target = directory / name
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"initial adapter file is missing or unsafe: {name}")
        sizes[name] = target.stat().st_size
    config = json.loads((directory / "adapter_config.json").read_text())
    if (config.get("r") != lora_rank or config.get("lora_alpha") != lora_alpha
            or config.get("bias", "none") != "none" or config.get("use_dora", False)
            or config.get("use_rslora", False) or config.get("rank_pattern") or config.get("alpha_pattern")):
        raise ValueError("initial adapter LoRA geometry differs from the selected model")
    try:
        state = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    except (OSError, SafetensorError) as exc:
        raise ValueError("initial adapter tensor format is invalid") from exc
    if not state:
        raise ValueError("initial adapter has no tensors")
    return state, {"source_policy_version": record["source_policy_version"],
                   "source_snapshot_sha256": record["source_snapshot_sha256"],
                   "base_sha256": base_sha256, "files": files, "file_sizes": sizes,
                   "content_verification": "disabled", "optimizer": "new"}
