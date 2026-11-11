"""Serialize projected text Learner adapters in the Worker artifact's HF namespace."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


_PROJECTED = "base_model.model.model.layers."
_CONDITIONAL = "base_model.model.model.language_model.layers."


def prepare_worker_adapter(directory: Path, evidence: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize only a newly staged PEFT publication, preserving all tensor values.

    The native NVFP4 Learner projects the original Qwen conditional artifact into a
    causal text model. vLLM retains the conditional module hierarchy; its
    standard HF mapper requires the original ``model.language_model`` prefix.
    Learner recovery state and existing immutable publications are untouched.
    """
    model_path = evidence.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        return None
    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        return None

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    path = Path(directory) / "adapter_model.safetensors"
    weights = load_file(str(path), device="cpu")
    projected = [name for name in weights if name.startswith(_PROJECTED)]
    if not projected:
        return {"view": "original_hf_conditional", "renamed_tensor_count": 0}
    if len(projected) != len(weights):
        raise ValueError("mixed adapter namespaces in projected text publication")
    mapped = {_CONDITIONAL + name[len(_PROJECTED):]: value for name, value in weights.items()}
    if len(mapped) != len(weights):
        raise ValueError("adapter namespace conversion produced duplicate tensor names")
    with safe_open(str(path), framework="pt", device="cpu") as source:
        metadata = source.metadata()
    fd, temp_name = tempfile.mkstemp(prefix=".adapter-worker-", suffix=".safetensors", dir=directory)
    os.close(fd)
    staged = Path(temp_name)
    try:
        save_file(mapped, str(staged), metadata=metadata)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return {"view": "original_hf_conditional", "renamed_tensor_count": len(mapped),
            "tensor_values_preserved": True}
