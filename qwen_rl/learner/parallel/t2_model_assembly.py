"""Shared identity-bound Qwen3.5 T2 model assembly for Learner execution modes."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class T2ModelAssemblyError(RuntimeError):
    """The pinned T2 artifact cannot produce the supported Learner model."""


def prepare_t2_model_in_cpu(config: Mapping[str, Any]):
    """Assemble the verified packed T2 model on CPU while its paired Worker runs.

    No dense BF16 base is materialized. The existing assembly reads packed
    artifact tensors and constructs the same T2 layers with CPU buffers; the
    caller moves the completed model to CUDA only inside the GPU handoff fence.
    """
    model, tokenizer, evidence = load_t2_model(
        config, device="cpu", parallelism="single", world_size=1,
    )
    if (any(parameter.device.type != "cpu" for parameter in model.parameters())
            or any(buffer.device.type != "cpu" for buffer in model.buffers())):
        raise T2ModelAssemblyError("CPU T2 preparation unexpectedly allocated a non-CPU tensor")
    evidence = dict(evidence)
    evidence["preparation_device"] = "cpu"
    return model, tokenizer, evidence


def load_t2_model(
    config: Mapping[str, Any],
    *,
    device: Any,
    parallelism: str,
    world_size: int,
    materialize_tokenizer_files: Callable[[Any, Path], dict[str, Any]] | None = None,
):
    """Load the current Qwen3.5 T2 model and its evidence from one provider contract.

    DDP supplies a rank-zero tokenizer materializer that broadcasts its receipt;
    a managed single-device executor materializes the same resources locally.
    Both modes then use the same provider, model builder, LoRA checks, and evidence.
    """
    import torch
    from transformers import AutoTokenizer

    from qwen_rl.contracts.ternary import validate_artifact_identity
    from qwen_rl.learner.parallel.qwen35_pipeline import build_qwen35_causal_lm
    from qwen_rl.models.ternary import is_t2_linear, t2_linear_input_basis
    from qwen_rl.models.ternary_fused_hadamard import kernel_build_info
    from qwen_rl.models.ternary_provider import TernaryModelProvider

    identity_path = config.get("artifact_identity_path")
    tokenizer_path = config.get("tokenizer_path")
    declared_identity = config.get("artifact_identity")
    if not isinstance(identity_path, str) or not identity_path:
        raise T2ModelAssemblyError("T2 model assembly requires artifact_identity_path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise T2ModelAssemblyError("T2 model assembly requires tokenizer_path")
    if parallelism not in {"single", "ddp"}:
        raise T2ModelAssemblyError("T2 shared model assembly supports single or ddp")
    if type(world_size) is not int or world_size < 1:
        raise T2ModelAssemblyError("T2 model assembly world_size must be positive")
    if parallelism == "single" and world_size != 1:
        raise T2ModelAssemblyError("single T2 model assembly requires world_size=1")
    try:
        declared_identity = validate_artifact_identity(declared_identity)
    except Exception as exc:
        raise T2ModelAssemblyError(f"T2 artifact identity is invalid: {exc}") from exc

    # T2 training defaults to FFPA. Fail during assembly, including CPU
    # preparation, before a managed executor acquires its GPU.
    from importlib import metadata
    try:
        if metadata.version("ffpa-attn") != "0.2.4":
            raise T2ModelAssemblyError("T2 Learner requires ffpa-attn==0.2.4")
        from ffpa_attn.triton import _ffpa_attn_forward_triton, _ffpa_attn_backward_triton
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise T2ModelAssemblyError(
            "T2 Learner requires ffpa-attn==0.2.4; install the Learner dependencies"
        ) from exc

    hadamard_build = kernel_build_info(check_device=torch.device(device).type == "cuda")
    with TernaryModelProvider(identity_path, verify_files=False) as provider:
        if provider.identity.as_dict() != declared_identity:
            raise T2ModelAssemblyError("T2 artifact sidecar changed after profile validation")
        tokenizer_destination = Path(tokenizer_path).expanduser().resolve()
        if materialize_tokenizer_files is None:
            tokenizer_files = provider.materialize_text_resources(tokenizer_destination)
        else:
            tokenizer_files = materialize_tokenizer_files(provider, tokenizer_destination)
        if not isinstance(tokenizer_files, dict) or not tokenizer_files:
            raise T2ModelAssemblyError("T2 tokenizer resource inventory is missing")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model, model_evidence = build_qwen35_causal_lm(
            provider.text_config,
            provider,
            lora_rank=int(config.get("lora_rank", 16)),
            lora_alpha=int(config.get("lora_alpha", 16)),
            device=device,
        )
        provider_evidence = provider.evidence()

    initial_adapter = None
    initial_adapter_path = config.get("initial_adapter_path")
    if initial_adapter_path:
        from ..initial_adapter import read_initial_adapter

        state, initial_adapter = read_initial_adapter(
            initial_adapter_path,
            base_sha256=declared_identity["source"]["ninfer_artifact"]["sha256"],
            lora_rank=int(config.get("lora_rank", 16)),
            lora_alpha=int(config.get("lora_alpha", 16)),
        )
        load_t2_adapter_state(model, state)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.dtype is not torch.float32 for parameter in trainable):
        raise T2ModelAssemblyError("T2 model assembly requires FP32 LoRA master parameters")
    packed_linears = [module for module in model.modules() if is_t2_linear(module)]
    if not packed_linears or any(
        module.compute_dtype is not torch.bfloat16 or module.output_dtype is not torch.bfloat16
        for module in packed_linears
    ):
        raise T2ModelAssemblyError("T2 model assembly requires the packed BF16 compute/output contract")

    evidence = dict(model_evidence)
    evidence.update(provider_evidence)
    if initial_adapter is not None:
        evidence["initial_adapter"] = initial_adapter
    evidence.update(
        packed_module_count=len(packed_linears),
        packed_module_input_basis_counts={
            basis: sum(t2_linear_input_basis(module) == basis for module in packed_linears)
            for basis in ("primal", "prism_rotated")
        },
        is_loaded_in_4bit=False,
        base_trainable_parameters=0,
    )
    evidence.update({
        "model_backend": "ternary",
        "parallelism": parallelism,
        "world_size": world_size,
        "is_loaded_in_4bit": False,
        "base_frozen": True,
        "base_trainable_parameters": 0,
        "adapter_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "lora_master_dtype": "torch.float32",
        "hadamard": {
            "forward": "ninfer_fused_bf16",
            "transpose_vjp": "dao_fp32_original_order",
            "build": hadamard_build,
        },
        "tokenizer_files": tokenizer_files,
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "tokenizer_length": len(tokenizer),
        "output_head": "trl_fused_selected_token_t2_rotated",
    })
    return model, tokenizer, evidence


def load_t2_adapter_state(model, state: dict[str, Any]) -> None:
    """Load a complete T2 PEFT state and verify every canonical live tensor."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    if not isinstance(state, dict) or not state:
        raise ValueError("T2 PEFT snapshot must contain a non-empty adapter state")
    canonical_state = {}
    for name, value in state.items():
        canonical = name.replace(".default.", ".") if isinstance(name, str) else name
        if canonical in canonical_state:
            raise ValueError(f"T2 PEFT snapshot has duplicate canonical key: {canonical}")
        canonical_state[canonical] = value
    result = set_peft_model_state_dict(model, canonical_state, adapter_name="default")
    if result.unexpected_keys or getattr(result, "mismatched_keys", ()):
        raise ValueError("T2 PEFT snapshot has unexpected or mismatched adapter tensors")
    loaded = get_peft_model_state_dict(model, adapter_name="default")
    if set(loaded) != set(canonical_state):
        raise ValueError("T2 PEFT restore did not load the complete adapter inventory")
    for name, value in canonical_state.items():
        if (not isinstance(value, torch.Tensor) or tuple(loaded[name].shape) != tuple(value.shape)
                or loaded[name].dtype != value.dtype
                or not torch.equal(loaded[name].detach().cpu(), value.detach().cpu())):
            raise ValueError(f"T2 PEFT restore differs from its snapshot tensor: {name}")
