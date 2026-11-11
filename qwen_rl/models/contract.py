"""Learner compute and PEFT projection contracts."""

from typing import Any

import torch
from torch import nn

_LORA_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_SDPA_BACKENDS_BY_IMPLEMENTATION = {
    "attention_mask_absent": "efficient_attention",
    "attention_mask_present": "efficient_attention",
}

NUMERICAL_POLICY = {
    "non_quantized_parameter_dtype": "torch.bfloat16",
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "attention_implementation": "arc_sdpa",
    "attention_mask": "sdpa",
    "attention_kv_head_expansion": "hf_repeat_kv_when_mask_absent",
    "sdpa_backend_by_attention_mask": dict(_SDPA_BACKENDS_BY_IMPLEMENTATION),
}


def _enforce_numerical_policy() -> dict[str, Any]:
    """Set shared safe SDPA availability and return the model's routed policy."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    if (
        torch.get_float32_matmul_precision() != "highest"
        or torch.backends.cuda.matmul.allow_tf32 is not False
        or torch.backends.cudnn.allow_tf32 is not False
    ):
        raise RuntimeError("could not enforce the canonical BF16 numerical policy")
    if (
        not torch.backends.cuda.flash_sdp_enabled()
        or not torch.backends.cuda.mem_efficient_sdp_enabled()
        or torch.backends.cuda.math_sdp_enabled()
        or torch.backends.cuda.cudnn_sdp_enabled()
    ):
        raise RuntimeError(
            "could not enforce explicit Flash/Efficient SDPA availability"
        )
    policy = dict(NUMERICAL_POLICY)
    policy["sdpa_backend_by_attention_mask"] = dict(_SDPA_BACKENDS_BY_IMPLEMENTATION)
    return policy


def _text_lora_targets(model: nn.Module) -> list[str]:
    """Select the configured text projections, excluding vision modules."""
    names: list[str] = []
    for name, module in model.named_modules():
        if not name or not name.startswith("model."):
            continue
        if name.startswith(("model.visual.", "model.vision_tower.")):
            continue
        if name.rsplit(".", 1)[-1] in _LORA_SUFFIXES and isinstance(module, nn.Module):
            names.append(name)
    if not names:
        raise RuntimeError(
            "no Qwen text projection modules matched the LoRA target set "
            f"{_LORA_SUFFIXES}; refusing to create a no-op adapter"
        )
    return names


def _set_embedding_and_head_dtype(model: nn.Module) -> dict[str, Any]:
    input_embedding = model.get_input_embeddings()
    output_head = model.get_output_embeddings()
    if input_embedding is None or output_head is None:
        raise RuntimeError("causal model must expose input embeddings and output head")
    input_weight = getattr(input_embedding, "weight", None)
    output_weight = getattr(output_head, "weight", None)
    if input_weight is None or output_weight is None:
        raise RuntimeError("embedding and output head must expose a weight tensor")
    if not input_weight.is_floating_point() or not output_weight.is_floating_point():
        raise TypeError("native NVFP4 loader requires floating-point embeddings and output head")
    if input_weight.dtype != torch.bfloat16:
        input_weight.data = input_weight.data.to(dtype=torch.bfloat16)
    if output_weight is not input_weight and output_weight.dtype != torch.bfloat16:
        output_weight.data = output_weight.data.to(dtype=torch.bfloat16)
    if (
        input_embedding.weight.dtype != torch.bfloat16
        or output_head.weight.dtype != torch.bfloat16
    ):
        raise RuntimeError("input embedding and output head must remain BF16")
    if input_embedding.weight.dtype != output_head.weight.dtype:
        raise RuntimeError("input embedding and output head dtype must match")
    return {
        "input_embedding": str(input_embedding.weight.dtype),
        "output_head": str(output_head.weight.dtype),
        "tied": input_embedding.weight is output_head.weight,
    }
