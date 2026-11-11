"""Complete Qwen3.5 T2 model assembly for explicit four-rank tensor parallelism."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from qwen_rl.models.ternary import FrozenTernaryEmbedding, FrozenTernaryLinear
from qwen_rl.models.ternary_tp_head import (
    VocabularyParallelTernaryEmbedding,
    VocabularyParallelTernaryHead,
)
from qwen_rl.models.ternary_tp_layers import (
    ColumnParallelT2LoRALinear,
    RowParallelT2LoRALinear,
    adapter_placement_records,
    split_hadamard_blocks,
)
from qwen_rl.models.ternary_tp_state import (
    load_t2_tp_peft_state_dict,
    normalize_t2_tp_placements,
)

T2_TP_MODEL_RECIPE = "qwen35_bonsai_t2_tp4_trl_head/v1"
_SUPPORTED_WORLD_SIZE = 4


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_runtime(rank: int, local_rank: int, world_size: int,
                      control_group: Any) -> None:
    if type(world_size) is not int or world_size != _SUPPORTED_WORLD_SIZE:
        raise ValueError("complete Qwen3.5 T2 tensor parallel assembly requires world_size=4")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be inside the four-rank TP world")
    if type(local_rank) is not int or local_rank < 0:
        raise ValueError("local_rank must be a nonnegative integer")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("T2 TP model assembly requires an initialized process group")
    if dist.get_world_size() != world_size or dist.get_rank() != rank:
        raise RuntimeError("loader rank/world_size differs from the initialized TP process group")
    if control_group is not None and (
        dist.get_world_size(group=control_group) != world_size
        or dist.get_rank(group=control_group) != rank
    ):
        raise ValueError("control_group must contain the same ordered four TP ranks")


def _run_rank_stage(stage: str, function, *, rank: int, world_size: int,
                    control_group: Any, cleanup=None):
    """Run a local stage and report its result/error to every rank before continuing."""
    result = None
    error = None
    try:
        result = function()
    except Exception as exc:  # noqa: BLE001 -- every rank must exchange failure status
        error = f"{type(exc).__name__}: {exc}"

    reports: list[Any] = [None] * world_size
    try:
        dist.all_gather_object(
            reports,
            {"rank": rank, "stage": stage, "error": error},
            group=control_group,
        )
    except Exception:
        if cleanup is not None and result is not None:
            cleanup(result)
        raise

    failures = [item for item in reports if item is not None and item.get("error")]
    if failures:
        if cleanup is not None and result is not None:
            cleanup(result)
        detail = "; ".join(
            f"rank {item['rank']}: {item['error']}" for item in failures
        )
        raise RuntimeError(f"T2 TP loader {stage} failed across ranks: {detail}")
    return result


def _tokenizer_vocab_evidence(tokenizer: Any, model_vocab_size: int) -> dict[str, int]:
    """Validate identity-bound tokenizer IDs against a padded model vocabulary."""
    if type(model_vocab_size) is not int or model_vocab_size <= 0:
        raise ValueError("model vocabulary size must be a positive integer")
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("identity-bound tokenizer must report a positive base vocabulary size")
    token_length = len(tokenizer)
    if isinstance(token_length, bool) or not isinstance(token_length, int) or token_length <= 0:
        raise ValueError("identity-bound tokenizer length must be a positive integer")
    if vocab_size > token_length or token_length > model_vocab_size:
        raise ValueError(
            "identity-bound tokenizer geometry must satisfy "
            "0 < vocab_size <= len(tokenizer) <= model_vocab_size"
        )
    get_vocab = getattr(tokenizer, "get_vocab", None)
    token_map = get_vocab() if callable(get_vocab) else None
    if not isinstance(token_map, Mapping) or not token_map:
        raise ValueError("identity-bound tokenizer must expose its token-to-ID mapping")
    token_ids = list(token_map.values())
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in token_ids):
        raise ValueError("identity-bound tokenizer contains an invalid token ID")
    maximum = max(token_ids)
    if maximum >= model_vocab_size:
        raise ValueError(
            f"identity-bound tokenizer token ID {maximum} exceeds model vocabulary {model_vocab_size}"
        )
    return {
        "tokenizer_vocab_size": vocab_size,
        "tokenizer_length": token_length,
        "tokenizer_max_token_id": maximum,
        "model_vocab_size": model_vocab_size,
        "model_vocab_padding_rows": model_vocab_size - token_length,
    }


def _validated_artifact_config(config: Mapping[str, Any]) -> dict[str, Any]:
    from qwen_rl.contracts.ternary import (
        load_artifact_identity,
        validate_artifact_identity,
    )

    if not isinstance(config, Mapping):
        raise TypeError("T2 TP model config must be a mapping")
    if config.get("model_backend") != "ternary":
        raise ValueError("T2 TP loader requires model_backend='ternary'")
    if config.get("parallelism", "tp") != "tp":
        raise ValueError("T2 TP loader requires parallelism='tp'")

    identity_path = config.get("artifact_identity_path")
    tokenizer_path = config.get("tokenizer_path")
    if not isinstance(identity_path, str) or not identity_path:
        raise ValueError("T2 TP config requires artifact_identity_path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise ValueError("T2 TP config requires tokenizer_path")

    declared = validate_artifact_identity(config.get("artifact_identity"))
    sidecar = load_artifact_identity(identity_path, verify_files=False).as_dict()
    if sidecar != declared:
        raise ValueError("T2 artifact identity sidecar changed after profile validation")

    source = declared["source"]["ninfer_artifact"]
    model_path = config.get("model_path")
    if not isinstance(model_path, str) or Path(model_path).expanduser().resolve() != Path(
        source["path"]
    ).expanduser().resolve():
        raise ValueError("T2 model_path differs from the validated artifact identity")

    adapter = declared["learner"]["adapter"]
    lora_rank = _positive_int(config.get("lora_rank"), "lora_rank")
    lora_alpha = _positive_int(config.get("lora_alpha"), "lora_alpha")
    if lora_rank != adapter["rank"] or lora_alpha != adapter["alpha"]:
        raise ValueError("T2 TP LoRA rank/alpha differs from the artifact identity")

    output_chunk_rows = config.get("tp_output_chunk_rows", 8192)
    if isinstance(output_chunk_rows, bool) or not isinstance(output_chunk_rows, int) or output_chunk_rows < 1:
        raise ValueError("tp_output_chunk_rows must be a positive integer")

    gradient_checkpointing = config.get("gradient_checkpointing", True)
    if type(gradient_checkpointing) is not bool:
        raise ValueError("gradient_checkpointing must be a boolean")

    offload_enabled = config.get("packed_weight_offload", False)
    if type(offload_enabled) is not bool:
        raise ValueError("packed_weight_offload must be a boolean")
    cpu_cap = config.get("packed_weight_offload_max_cpu_master_bytes", 4 * 1024**3)
    pinned_cap = config.get("packed_weight_offload_max_pinned_bytes", 512 * 1024**2)
    offload_prefetch = config.get("packed_weight_offload_prefetch", True)
    if type(offload_prefetch) is not bool:
        raise ValueError("packed_weight_offload_prefetch must be a boolean")
    if offload_enabled:
        _positive_int(cpu_cap, "packed_weight_offload_max_cpu_master_bytes")
        _positive_int(pinned_cap, "packed_weight_offload_max_pinned_bytes")

    return {
        "identity_path": identity_path,
        "identity": declared,
        "tokenizer_path": tokenizer_path,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "tp_output_chunk_rows": output_chunk_rows,
        "gradient_checkpointing": gradient_checkpointing,
        "packed_weight_offload": offload_enabled,
        "packed_weight_offload_max_cpu_master_bytes": cpu_cap,
        "packed_weight_offload_max_pinned_bytes": pinned_cap,
        "packed_weight_offload_prefetch": offload_prefetch,
    }


def _active_adapter_weights(module: nn.Module) -> tuple[FrozenTernaryLinear, torch.Tensor | nn.Parameter,
                                                          torch.Tensor | nn.Parameter, float]:
    get_base = getattr(module, "get_base_layer", None)
    if not callable(get_base):
        raise TypeError("Qwen3.5 MLP projections must be initialized standard PEFT LoRA modules")
    source = get_base()
    if type(source) is not FrozenTernaryLinear:
        raise TypeError(
            f"Qwen3.5 MLP TP requires a FrozenTernaryLinear base, got {type(source).__name__}"
        )
    adapter_names = list(getattr(module, "lora_A", {}).keys())
    active = getattr(module, "active_adapters", adapter_names)
    if isinstance(active, str):
        active = [active]
    selected = [name for name in adapter_names if name in active]
    if len(selected) != 1:
        raise ValueError("T2 MLP TP requires exactly one active standard PEFT adapter")
    adapter_name = selected[0]
    if bool(getattr(module, "merged", False)) or getattr(module, "merged_adapters", ()):
        raise ValueError("merged PEFT adapters cannot be converted into T2 TP shards")
    dropout = module.lora_dropout[adapter_name]
    if float(getattr(dropout, "p", 0.0)) != 0.0:
        raise ValueError("T2 MLP TP requires zero-dropout LoRA adapters")
    lora_a = module.lora_A[adapter_name]
    lora_b = module.lora_B[adapter_name]
    if not hasattr(lora_a, "weight") or not hasattr(lora_b, "weight"):
        raise TypeError("T2 MLP LoRA factors must expose standard Linear weights")
    return source, lora_a.weight, lora_b.weight, float(module.scaling[adapter_name])


def _replace_mlp_projection(parent: nn.Module, name: str, old: nn.Module,
                            replacement: nn.Module) -> None:
    if getattr(parent, name) is not old:
        raise RuntimeError(f"MLP projection {name} changed during T2 TP installation")
    setattr(parent, name, replacement)


def _install_mlp_tensor_parallel(text_model: nn.Module, *, world_size: int,
                                 rank: int, group=None) -> dict[str, Any]:
    """Install paired MLP column/row shards, retaining complete Hadamard blocks."""
    layers = getattr(text_model, "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)) or not layers:
        raise TypeError("Qwen3.5 text model must expose decoder layers")

    planned: list[tuple[nn.Module, list[tuple[str, nn.Module, nn.Module]]]] = []
    manifest_layers = []
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise TypeError(f"Qwen3.5 decoder layer {index} has no MLP")
        gate = getattr(mlp, "gate_proj", None)
        up = getattr(mlp, "up_proj", None)
        down = getattr(mlp, "down_proj", None)
        if any(module is None for module in (gate, up, down)):
            raise TypeError(f"Qwen3.5 decoder layer {index} has an incomplete gated MLP")

        gate_source, gate_a, gate_b, gate_scaling = _active_adapter_weights(gate)
        up_source, up_a, up_b, up_scaling = _active_adapter_weights(up)
        down_source, down_a, down_b, down_scaling = _active_adapter_weights(down)
        intermediate = int(gate_source.out_features)
        if (gate_source.in_features != up_source.in_features
                or gate_source.in_features != down_source.out_features
                or gate_source.out_features != up_source.out_features
                or down_source.in_features != intermediate):
            raise ValueError(f"Qwen3.5 decoder layer {index} MLP dimensions do not pair")

        start, end = split_hadamard_blocks(intermediate, world_size, rank)
        gate_parallel = ColumnParallelT2LoRALinear(
            gate_source, shard=(start, end), lora_a_weight=gate_a, lora_b_weight=gate_b,
            scaling=gate_scaling, group=group,
        )
        up_parallel = ColumnParallelT2LoRALinear(
            up_source, shard=(start, end), lora_a_weight=up_a, lora_b_weight=up_b,
            scaling=up_scaling, group=group,
        )
        down_parallel = RowParallelT2LoRALinear(
            down_source, shard=(start, end), lora_a_weight=down_a, lora_b_weight=down_b,
            scaling=down_scaling, output_mode="replicated", group=group,
        )
        for replacement, old in (
            (gate_parallel, gate), (up_parallel, up), (down_parallel, down)
        ):
            replacement.train(old.training)
        planned.append((mlp, [
            ("gate_proj", gate, gate_parallel),
            ("up_proj", up, up_parallel),
            ("down_proj", down, down_parallel),
        ]))
        manifest_layers.append({
            "layer": index,
            "intermediate_size": intermediate,
            "hadamard_block_size": 1024,
            "local_intermediate_rows": [start, end],
            "local_intermediate_width": end - start,
            "gate_projection": "column_parallel_t2_lora",
            "up_projection": "column_parallel_t2_lora",
            "down_projection": "row_parallel_t2_lora_replicated_output",
        })

    # All validation and allocation succeeds before the first model mutation.
    for mlp, entries in planned:
        for name, old, replacement in entries:
            _replace_mlp_projection(mlp, name, old, replacement)

    return {
        "recipe": "qwen35_t2_paired_mlp_column_row/v1",
        "rank": rank,
        "world_size": world_size,
        "layers": manifest_layers,
        "activation_layout": "local_gate_up_rows_then_row_parallel_down",
        "adapter_basis": "rotated_hadamard_input_with_separate_base_and_lora_vjps",
    }


def _collect_adapter_placements(model: nn.Module, *, rank: int,
                                world_size: int) -> dict[str, dict[str, Any]]:
    placements: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, (ColumnParallelT2LoRALinear, RowParallelT2LoRALinear)):
            continue
        if module.local_lora_rank == 0:
            continue
        records = adapter_placement_records(name, module)
        overlap = set(placements).intersection(records)
        if overlap:
            raise ValueError(f"duplicate canonical TP adapter placements: {sorted(overlap)[:4]}")
        placements.update(records)
    normalized = normalize_t2_tp_placements(
        model, placements, rank=rank, world_size=world_size
    )
    if not normalized:
        raise ValueError("T2 TP model has no trainable LoRA placement records")
    return normalized


def _set_accelerate_tp_size(model: nn.Module, base_model: nn.Module,
                            world_size: int) -> dict[str, Any]:
    """Advertise the explicit custom TP layout to Accelerate's Trainer integration."""
    # Transformers exposes tp_size as a read-only property over _tp_size.
    # Accelerate checks this property before it prepares the model.
    base_model._tp_size = world_size
    config = getattr(base_model, "config", None)
    if config is None:
        raise TypeError("Qwen3.5 base model must expose its config for Accelerate TP metadata")
    try:
        config.distributed_config = {"tp_size": world_size}
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("Qwen3.5 config cannot expose distributed_config TP metadata") from exc
    if getattr(model, "tp_size", None) != world_size:
        raise RuntimeError("PEFT wrapper did not expose the base model Accelerate tp_size marker")
    if getattr(config, "distributed_config", {}).get("tp_size") != world_size:
        raise RuntimeError("Qwen3.5 config did not retain its TP topology metadata")
    return {
        "model_tp_size": int(model.tp_size),
        "config_distributed_config": {"tp_size": world_size},
        "parallelism_owner": "qwen_rl_explicit_t2_tp_modules",
        "transformers_tp_plan": "unused",
    }


def _set_nonregistered_module_reference(owner: nn.Module, name: str, value: nn.Module) -> None:
    """Expose a bridge module without adding a second state-dict registration path."""
    object.__setattr__(owner, name, value)


def _cleanup_assembly(result: dict[str, Any]) -> None:
    handle = result.get("sequence_parallel_handle")
    if handle is not None:
        handle.close()


def load_t2_tp_policy(
    config: Mapping[str, Any],
    *,
    local_rank: int,
    rank: int,
    world_size: int,
    control_group=None,
):
    """Load an identity-bound T2 Qwen3.5 policy and install its complete TP4 layout.

    The PEFT model is first built on CPU using the normal T2 artifact provider and
    standard adapter injection. Local TP shards are installed before CUDA placement.
    TP collectives use the initialized default process group; control_group is used
    only to coordinate loader stages and error reports.
    """
    _validate_runtime(rank, local_rank, world_size, control_group)
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 T2 TP model assembly requires CUDA workers")
    if local_rank >= torch.cuda.device_count():
        raise ValueError("local_rank is outside the visible CUDA device count")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    inputs = _run_rank_stage(
        "identity_validation",
        lambda: _validated_artifact_config(config),
        rank=rank,
        world_size=world_size,
        control_group=control_group,
    )

    def build_cpu_model():
        from transformers import AutoTokenizer

        from qwen_rl.learner.parallel.qwen35_pipeline import build_qwen35_causal_lm
        from qwen_rl.models.ternary_provider import TernaryModelProvider

        with TernaryModelProvider(inputs["identity_path"], verify_files=False) as provider:
            if provider.identity.as_dict() != inputs["identity"]:
                raise ValueError("T2 provider identity changed before TP model assembly")
            tokenizer_files = provider.materialize_text_resources(inputs["tokenizer_path"])
            tokenizer = AutoTokenizer.from_pretrained(
                inputs["tokenizer_path"], local_files_only=True
            )
            model, build_evidence = build_qwen35_causal_lm(
                provider.text_config,
                provider,
                lora_rank=inputs["lora_rank"],
                lora_alpha=inputs["lora_alpha"],
                device="cpu",
                with_lora=True,
            )
            provider_evidence = provider.evidence()
        expected_vocab = int(model.get_base_model().config.vocab_size)
        tokenizer_evidence = _tokenizer_vocab_evidence(tokenizer, expected_vocab)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "build_evidence": build_evidence,
            "provider_evidence": provider_evidence,
            "tokenizer_files": tokenizer_files,
            "tokenizer_evidence": tokenizer_evidence,
        }

    built = _run_rank_stage(
        "cpu_model_construction",
        build_cpu_model,
        rank=rank,
        world_size=world_size,
        control_group=control_group,
    )
    model = built["model"]

    def install_tp_layout():
        base = model.get_base_model()
        text_model = base.model
        base.config.use_cache = False
        model.train()

        original_head = base.get_output_embeddings()
        if type(original_head) is not FrozenTernaryLinear:
            raise TypeError("T2 TP requires an unsharded FrozenTernaryLinear output head")
        parallel_head = VocabularyParallelTernaryHead.from_source(original_head)
        base.lm_head = parallel_head
        del original_head

        original_embedding = base.get_input_embeddings()
        if type(original_embedding) is not FrozenTernaryEmbedding:
            raise TypeError("T2 TP requires an unsharded FrozenTernaryEmbedding token table")
        parallel_embedding = VocabularyParallelTernaryEmbedding(original_embedding)
        text_model.embed_tokens = parallel_embedding
        del original_embedding

        # The full-head scorer created by the ordinary single-rank builder owns
        # another reference to the source output module. Remove it after sharding.
        if hasattr(base, "_ternary_fused_head_adapter"):
            delattr(base, "_ternary_fused_head_adapter")
        if hasattr(base, "_ternary_fused_head_bridge_required"):
            delattr(base, "_ternary_fused_head_bridge_required")
        _set_nonregistered_module_reference(base, "_t2_tp_head", parallel_head)
        base._t2_tp_head_bridge_required = True
        _set_nonregistered_module_reference(model, "_t2_tp_head", parallel_head)
        model._t2_tp_head_bridge_required = True

        mlp_evidence = _install_mlp_tensor_parallel(
            text_model, world_size=world_size, rank=rank
        )
        from qwen_rl.models.ternary_tp_blocks import install_qwen35_attention_gdn_tp

        attention_evidence = install_qwen35_attention_gdn_tp(
            model,
            output_chunk_rows=inputs["tp_output_chunk_rows"],
        )

        placements = _collect_adapter_placements(
            model, rank=rank, world_size=world_size
        )

        initial_adapter_evidence = None
        if config.get("initial_adapter_path"):
            from qwen_rl.learner.initial_adapter import read_initial_adapter

            full_state, initial_adapter_evidence = read_initial_adapter(
                config["initial_adapter_path"],
                base_sha256=inputs["identity"]["source"]["ninfer_artifact"]["sha256"],
                lora_rank=inputs["lora_rank"],
                lora_alpha=inputs["lora_alpha"],
            )
            load_t2_tp_peft_state_dict(
                model, full_state, placements, rank=rank, world_size=world_size
            )

        lingering_base = [
            name for name, module in model.named_modules()
            if isinstance(module, (FrozenTernaryLinear, FrozenTernaryEmbedding))
        ]
        if lingering_base:
            raise RuntimeError(
                "complete T2 TP assembly left unsharded T2 modules: "
                + ", ".join(lingering_base[:8])
            )

        accelerator_tp = _set_accelerate_tp_size(model, base, world_size)
        for owner in (model, base):
            owner._t2_tp_model = True
            owner._t2_tp_placements = placements
            owner._t2_tp_layout_recipe = T2_TP_MODEL_RECIPE
        return {
            "base": base,
            "placements": placements,
            "mlp_evidence": mlp_evidence,
            "attention_evidence": attention_evidence,
            "accelerator_tp": accelerator_tp,
            "initial_adapter": initial_adapter_evidence,
            "head": parallel_head,
            "embedding": parallel_embedding,
        }

    assembled = _run_rank_stage(
        "tp_module_installation",
        install_tp_layout,
        rank=rank,
        world_size=world_size,
        control_group=control_group,
        cleanup=_cleanup_assembly,
    )
    base = assembled["base"]

    def move_to_device():
        model.to(device)
        from qwen_rl.learner.sequence_packing import enable_qwen35_packing
        from qwen_rl.models.ternary_blas import configure_ternary_blas

        blas = configure_ternary_blas(device)
        packing = enable_qwen35_packing(model)
        return {"blas": blas, "packing": packing}

    runtime_evidence = _run_rank_stage(
        "cuda_placement",
        move_to_device,
        rank=rank,
        world_size=world_size,
        control_group=control_group,
    )

    def install_sequence_and_checkpointing():
        from qwen_rl.models.ternary_tp_sequence import install_sequence_parallel

        sequence_handle = install_sequence_parallel(base.model)
        checkpointing = {"enabled": False, "checkpointed_layer_count": 0}
        if inputs["gradient_checkpointing"]:
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if not callable(enable):
                sequence_handle.close()
                raise TypeError("Qwen3.5 PEFT policy cannot enable gradient checkpointing")
            try:
                enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                layers = tuple(base.model.layers)
                if not layers or any(
                    not bool(getattr(layer, "gradient_checkpointing", False))
                    for layer in layers
                ):
                    raise RuntimeError(
                        "non-reentrant checkpointing was not enabled on every Qwen3.5 layer"
                    )
                checkpointing = {
                    "enabled": True,
                    "checkpointed_layer_count": len(layers),
                    "use_reentrant": False,
                }
            except Exception:
                sequence_handle.close()
                raise
        if getattr(base, "_require_grads_hook", None) is None:
            enable_inputs = getattr(model, "enable_input_require_grads", None)
            if not callable(enable_inputs):
                sequence_handle.close()
                raise TypeError("Qwen3.5 PEFT policy cannot enable input gradients")
            try:
                enable_inputs()
            except Exception:
                sequence_handle.close()
                raise
        for owner in (model, base):
            owner._t2_tp_sequence_parallel_handle = sequence_handle
        return {
            "sequence_parallel_handle": sequence_handle,
            "checkpointing": checkpointing,
            "input_requires_grad_hook_installed": getattr(base, "_require_grads_hook", None) is not None,
        }

    sequence_and_checkpointing = _run_rank_stage(
        "sequence_parallel_checkpointing",
        install_sequence_and_checkpointing,
        rank=rank,
        world_size=world_size,
        control_group=control_group,
        cleanup=_cleanup_assembly,
    )
    assembled.update(sequence_and_checkpointing)

    offloader = None

    def install_weight_offloader():
        nonlocal offloader
        if not inputs["packed_weight_offload"]:
            return None
        from qwen_rl.models.ternary_tp_offload import PackedTernaryTPOffloader

        offloader = PackedTernaryTPOffloader(
            tuple(base.model.layers),
            device=device,
            max_cpu_master_bytes=inputs["packed_weight_offload_max_cpu_master_bytes"],
            max_pinned_bytes=inputs["packed_weight_offload_max_pinned_bytes"],
            prefetch=inputs["packed_weight_offload_prefetch"],
        )
        base._t2_tp_weight_offloader = offloader
        return offloader.report()

    def cleanup_offloader(result):
        if offloader is not None:
            offloader.close()

    offload_report = _run_rank_stage(
        "packed_weight_offload_installation",
        install_weight_offloader,
        rank=rank,
        world_size=world_size,
        control_group=control_group,
        cleanup=cleanup_offloader,
    )

    placements = assembled["placements"]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    packed_tp_modules = sum(1 for module in model.modules()
                            if getattr(module, "_t2_tp_base", False) is True)
    evidence = dict(built["build_evidence"])
    evidence.update(built["provider_evidence"])
    evidence.update({
        "model_backend": "ternary",
        "parallelism": "tp",
        "rank": rank,
        "world_size": world_size,
        "tp_size": world_size,
        "tp_layout_recipe": T2_TP_MODEL_RECIPE,
        "accelerate_tp": assembled["accelerator_tp"],
        "base_frozen": True,
        "base_trainable_parameters": 0,
        "adapter_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "adapter_trainable_tensor_count": len(trainable),
        "lora_master_dtype": "torch.float32",
        "lora_format": "standard_peft_lora",
        "t2_tp_adapter_placement_count": len(placements),
        "t2_tp_adapter_target_count": len(placements) // 2,
        "t2_tp_packed_module_count": packed_tp_modules,
        "t2_tp_mlp": assembled["mlp_evidence"],
        "t2_tp_attention_gdn": assembled["attention_evidence"],
        "t2_tp_sequence_parallel": assembled["sequence_parallel_handle"].evidence,
        "gradient_checkpointing": assembled["checkpointing"],
        "input_requires_grad_hook_installed": assembled["input_requires_grad_hook_installed"],
        "blas": runtime_evidence["blas"],
        "sequence_packing": runtime_evidence["packing"],
        "packed_weight_offload": {
            "enabled": offload_report is not None,
            "report": offload_report,
        },
        "output_head": "t2_tp_vocabulary_parallel_selected_token",
        "tokenizer_files": built["tokenizer_files"],
        **built["tokenizer_evidence"],
        "initial_adapter": assembled["initial_adapter"],
        "hadamard": {
            "forward": "complete_1024_block_rotation_before_base_and_lora",
            "vjp": "separate_base_and_lora_transposes",
        },
        "is_loaded_in_4bit": False,
    })
    return model, built["tokenizer"], evidence


__all__ = ["T2_TP_MODEL_RECIPE", "load_t2_tp_policy"]
