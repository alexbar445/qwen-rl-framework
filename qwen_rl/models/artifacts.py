"""Model metadata fingerprints shared by native quantization backends."""
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def source_model_fingerprint(model_path: str) -> tuple[str, dict[str, Any]]:
    """Identify a generic source model from metadata without reading weight payloads.

    Config, index, and tokenizer assets retain their content fingerprints. A raw
    source model with no sidecar weight manifest uses only weight shard names and
    sizes; those entries deliberately carry ``sha256=None``.
    """
    root = Path(model_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"model_path does not exist: {root}")
    config_path = root / "config.json" if root.is_dir() else root
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found under model_path: {root}")
    config_bytes = config_path.read_bytes()
    index_path = root / "model.safetensors.index.json" if root.is_dir() else None
    if index_path is not None and index_path.is_file():
        index_data = json.loads(index_path.read_text())
        weight_map = index_data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map \
                or any(not isinstance(value, str) or not value for value in weight_map.values()):
            raise ValueError(f"invalid model weight index: {index_path}")
        weight_names = sorted(set(weight_map.values()))
        files = [root / name for name in weight_names]
    else:
        files = sorted(
            p for p in root.iterdir()
            if p.is_file() and (p.name.endswith(".safetensors") or p.name.endswith(".bin"))
        ) if root.is_dir() else []
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"model weight shards missing: {missing[:3]}")
    if not files:
        raise FileNotFoundError(f"no model weight shards found under model_path: {root}")
    for path in files:
        if root.is_dir() and root not in path.resolve().parents:
            raise ValueError(f"model weight index references a path outside model_path: {path.name}")
    manifest = [
        {"name": path.name, "size": path.stat().st_size, "sha256": None}
        for path in files
    ]
    index_sha256 = (
        _sha256_file(index_path)
        if index_path is not None and index_path.is_file()
        else None
    )
    tokenizer_files = []
    if root.is_dir():
        for name in (
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt", "chat_template.jinja", "added_tokens.json",
            "generation_config.json",
        ):
            path = root / name
            if path.is_file():
                tokenizer_files.append({
                    "name": name,
                    "sha256": _sha256_file(path),
                    "size": path.stat().st_size,
                })
    content_manifest = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "index_sha256": index_sha256,
        "weight_files": manifest,
        "tokenizer_files": tokenizer_files,
    }
    canonical = json.dumps(content_manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), {
        "kind": "metadata-model-manifest",
        "content_verification": "disabled",
        "identity_basis": "config_index_weight_names_and_sizes_tokenizer_content",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "index_sha256": index_sha256,
        "weight_files": manifest,
        "tokenizer_files": tokenizer_files,
    }
