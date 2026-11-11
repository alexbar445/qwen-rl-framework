"""Qwen3.5 text-stage assembly and a TRL fused head bridge for T2 PP training.

TernaryArtifactStore supplies packed Uses and their basis metadata. Model linears apply each
Use's declared sign/Hadamard transform, and PEFT LoRA-A follows its target module's basis. This
file assembles official Transformers decoder layers and passes hidden streams through the
upstream distributed pipeline.
"""
from __future__ import annotations

import hashlib
import json
import threading
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from itertools import pairwise
from types import SimpleNamespace
from typing import Any, Protocol

import torch
from torch import nn

from .pipeline import PipelineContractError

PIPELINE_LAYER_RANGES = ((0, 16), (16, 32), (32, 48), (48, 64))
DEFAULT_LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_HF_LAYER_BUILD_LOCK = threading.RLock()


class Qwen35TernaryProvider(Protocol):
    """The low-level artifact API consumed by this stage builder."""

    artifact_id: bytes
    path: Any
    payload_bytes: int
    text_config: dict[str, Any]

    def text_resource_hashes(self) -> dict[str, str]: ...

    def load_embedding(self, logical_name: str, *, expected_shape: tuple[int, int]) -> nn.Module: ...

    def load_linear(
        self,
        logical_name: str,
        *,
        expected_shape: tuple[int, int],
        compute_dtype: torch.dtype,
    ) -> nn.Module: ...

    def load_fused_linear(
        self,
        logical_names: Sequence[str],
        *,
        expected_shape: tuple[int, int],
        layout: str,
        head_size: int | None = None,
        compute_dtype: torch.dtype,
    ) -> nn.Module: ...

    def read_direct_parameter(
        self,
        logical_name: str,
        *,
        expected_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor: ...

def _direct_parameter(
    provider: Qwen35TernaryProvider,
    module: nn.Module,
    logical_name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    value = provider.read_direct_parameter(logical_name, expected_shape=shape, dtype=dtype)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise PipelineContractError(f"direct Qwen tensor {logical_name!r} has the wrong shape")
    module.weight = nn.Parameter(value.to(device=device, dtype=dtype), requires_grad=False)


def _replace_layer_weights(
    layer: nn.Module,
    *,
    layer_index: int,
    config: Any,
    provider: Qwen35TernaryProvider,
    device: torch.device,
) -> None:
    prefix = f"text/layers/{layer_index}"
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    dtype = torch.bfloat16
    if getattr(layer, "layer_type", None) == "linear_attention":
        linear = layer.linear_attn
        key_dim = int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
        value_dim = int(config.linear_num_value_heads) * int(config.linear_value_head_dim)
        linear.in_proj_qkv = provider.load_fused_linear(
            (
                f"{prefix}/gdn/query",
                f"{prefix}/gdn/key",
                f"{prefix}/gdn/value",
            ),
            expected_shape=(2 * key_dim + value_dim, hidden),
            layout="concat",
            compute_dtype=dtype,
        ).to(device)
        linear.in_proj_z = provider.load_linear(
            f"{prefix}/gdn/z", expected_shape=(value_dim, hidden), compute_dtype=dtype
        ).to(device)
        _direct_parameter(
            provider,
            linear.in_proj_b,
            f"{prefix}/gdn/b_projection",
            (int(config.linear_num_value_heads), hidden),
            device=device,
            dtype=dtype,
        )
        _direct_parameter(
            provider,
            linear.in_proj_a,
            f"{prefix}/gdn/a_projection",
            (int(config.linear_num_value_heads), hidden),
            device=device,
            dtype=dtype,
        )
        linear.out_proj = provider.load_linear(
            f"{prefix}/gdn/output", expected_shape=(hidden, value_dim), compute_dtype=dtype
        ).to(device)
        conv_shape = (int(config.linear_conv_kernel_dim), int(linear.conv_dim))
        conv = provider.read_direct_parameter(
            f"{prefix}/gdn/convolution", expected_shape=conv_shape, dtype=dtype
        )
        if tuple(conv.shape) != conv_shape:
            raise PipelineContractError(f"Qwen GDN convolution weight shape differs at layer {layer_index}")
        linear.conv1d.weight = nn.Parameter(
            conv.transpose(0, 1).unsqueeze(1).contiguous().to(device=device, dtype=dtype),
            requires_grad=False,
        )
        linear.A_log = nn.Parameter(
            provider.read_direct_parameter(
                f"{prefix}/gdn/a_log",
                expected_shape=(int(config.linear_num_value_heads),),
                dtype=torch.float32,
            ).to(device=device, dtype=torch.float32),
            requires_grad=False,
        )
        linear.dt_bias = nn.Parameter(
            provider.read_direct_parameter(
                f"{prefix}/gdn/dt_bias",
                expected_shape=(int(config.linear_num_value_heads),),
                dtype=torch.float32,
            ).to(device=device, dtype=torch.float32),
            requires_grad=False,
        )
        _direct_parameter(provider, linear.norm, f"{prefix}/gdn/norm", (int(config.linear_value_head_dim),), device=device, dtype=dtype)
    elif getattr(layer, "layer_type", None) == "full_attention":
        attention = layer.self_attn
        head_size = int(config.head_dim)
        query_width = int(config.num_attention_heads) * head_size
        key_value_width = int(config.num_key_value_heads) * head_size
        attention.q_proj = provider.load_fused_linear(
            (f"{prefix}/attention/query", f"{prefix}/attention/gate"),
            expected_shape=(2 * query_width, hidden),
            layout="head_interleave",
            head_size=head_size,
            compute_dtype=dtype,
        ).to(device)
        attention.k_proj = provider.load_linear(
            f"{prefix}/attention/key", expected_shape=(key_value_width, hidden), compute_dtype=dtype
        ).to(device)
        attention.v_proj = provider.load_linear(
            f"{prefix}/attention/value", expected_shape=(key_value_width, hidden), compute_dtype=dtype
        ).to(device)
        attention.o_proj = provider.load_linear(
            f"{prefix}/attention/output", expected_shape=(hidden, query_width), compute_dtype=dtype
        ).to(device)
        _direct_parameter(provider, attention.q_norm, f"{prefix}/attention/query_norm", (head_size,), device=device, dtype=dtype)
        _direct_parameter(provider, attention.k_norm, f"{prefix}/attention/key_norm", (head_size,), device=device, dtype=dtype)
    else:
        raise PipelineContractError(f"unsupported Qwen3.5 layer type at index {layer_index}")

    layer.mlp.gate_proj = provider.load_linear(
        f"{prefix}/mlp/gate", expected_shape=(intermediate, hidden), compute_dtype=dtype
    ).to(device)
    layer.mlp.up_proj = provider.load_linear(
        f"{prefix}/mlp/up", expected_shape=(intermediate, hidden), compute_dtype=dtype
    ).to(device)
    layer.mlp.down_proj = provider.load_linear(
        f"{prefix}/mlp/down", expected_shape=(hidden, intermediate), compute_dtype=dtype
    ).to(device)
    _direct_parameter(provider, layer.input_layernorm, f"{prefix}/input_norm", (hidden,), device=device, dtype=dtype)
    _direct_parameter(provider, layer.post_attention_layernorm, f"{prefix}/post_attention_norm", (hidden,), device=device, dtype=dtype)


class Qwen35PipelineStage(nn.Module):
    """A contiguous range of official Qwen3.5 text decoder layers."""

    def __init__(
        self,
        config: Any,
        *,
        stage_index: int,
        layer_start: int,
        layer_end: int,
        layers: dict[int, nn.Module],
        embed_tokens: nn.Module | None,
        rotary_emb: nn.Module | None,
        final_norm: nn.Module | None,
        output_head: nn.Module | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_tags: list[str] | None = None
        self.stage_index = stage_index
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({str(index): layers[index] for index in range(layer_start, layer_end)})
        if embed_tokens is not None:
            self.model.embed_tokens = embed_tokens
        if rotary_emb is not None:
            self.model.rotary_emb = rotary_emb
        if final_norm is not None:
            self.model.norm = final_norm
        if output_head is not None:
            self.lm_head = output_head
        self._batch_attention_mask: torch.Tensor | None = None
        self._batch_position_ids: torch.Tensor | None = None

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == 3

    def get_output_embeddings(self) -> nn.Module | None:
        """Return this partition's output head, if present.

        A headless PP partition is still a valid Trainer model. In particular,
        upstream TRL checks this method while deciding whether a PEFT adapter
        wraps the LM head.
        """
        return getattr(self, "lm_head", None)

    def get_input_embeddings(self) -> nn.Module:
        try:
            return self.model.embed_tokens
        except AttributeError as exc:
            raise PipelineContractError("only pipeline stage 0 owns the Qwen token embedding") from exc

    def add_model_tags(self, tags: list[str] | str) -> None:
        """Store the standard Transformers model tags as local metadata."""
        if isinstance(tags, str):
            tags = [tags]
        if self.model_tags is None:
            self.model_tags = []
        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)

    def enable_input_require_grads(self) -> None:
        """PEFT hook is unnecessary; PipelineStage owns activation-gradient flow."""
        return

    def configure_gradient_checkpointing(
        self,
        enabled: bool,
        *,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Toggle Transformers' official decoder-layer checkpoint hook locally."""
        if type(enabled) is not bool:
            raise TypeError("gradient_checkpointing enabled must be a boolean")
        kwargs = {"use_reentrant": False}
        if gradient_checkpointing_kwargs is not None:
            if not isinstance(gradient_checkpointing_kwargs, dict):
                raise TypeError("gradient_checkpointing_kwargs must be a dictionary")
            kwargs.update(gradient_checkpointing_kwargs)
        if kwargs.get("use_reentrant") is not False:
            raise PipelineContractError("Qwen PP requires non-reentrant gradient checkpointing")
        from torch.utils.checkpoint import checkpoint
        from transformers.modeling_layers import GradientCheckpointingLayer

        local_layers = list(self.model.layers.values())
        if not local_layers or any(not isinstance(layer, GradientCheckpointingLayer) for layer in local_layers):
            raise PipelineContractError("every PP decoder block must use the official GradientCheckpointingLayer hook")
        checkpoint_fn = partial(checkpoint, **kwargs)
        for layer in local_layers:
            layer.gradient_checkpointing = enabled
            layer._gradient_checkpointing_func = checkpoint_fn
        self.gradient_checkpointing = enabled
        return {
            "enabled": enabled,
            "checkpointed_layer_count": len(local_layers) if enabled else 0,
            "use_reentrant": False,
        }

    @contextmanager
    def bind_batch_context(self, attention_mask: torch.Tensor, position_ids: torch.Tensor):
        """Bind this update's immutable mask/positions locally on every PP rank."""
        if self._batch_attention_mask is not None or self._batch_position_ids is not None:
            raise PipelineContractError("a Qwen pipeline batch context is already bound")
        if attention_mask.ndim != 2 or position_ids.shape != (4, *attention_mask.shape):
            raise PipelineContractError("bound Qwen context must be [batch,sequence] and [4,batch,sequence]")
        if attention_mask.requires_grad or position_ids.requires_grad:
            raise PipelineContractError("attention and position context tensors must be frozen")
        self._batch_attention_mask = attention_mask
        self._batch_position_ids = position_ids
        try:
            yield self
        finally:
            self._batch_attention_mask = None
            self._batch_position_ids = None

    def forward(self, *args: torch.Tensor):
        from transformers.masking_utils import create_causal_mask

        attention_context = self._batch_attention_mask
        position_context = self._batch_position_ids
        if attention_context is None or position_context is None:
            raise PipelineContractError("bind a full-update Qwen batch context before a pipeline step")
        if self.is_first:
            if len(args) != 2:
                raise PipelineContractError("Qwen stage 0 expects input_ids and float32 batch row_ids")
            input_ids, row_ids = args
            hidden_states = self.model.embed_tokens(input_ids)
        else:
            if len(args) != 2:
                raise PipelineContractError("Qwen stages 1-3 expect hidden_states and float32 batch row_ids")
            hidden_states, row_ids = args
        if (not isinstance(row_ids, torch.Tensor) or row_ids.ndim != 1
                or row_ids.dtype != torch.float32 or row_ids.shape[0] != hidden_states.shape[0]
                or not bool(torch.isfinite(row_ids).all())
                or not bool((row_ids == row_ids.round()).all())):
            raise PipelineContractError("Qwen PP row_ids must be finite integer-valued float32 indices")
        row_indices = row_ids.to(dtype=torch.long)
        if row_indices.numel() and (int(row_indices.min()) < 0 or int(row_indices.max()) >= attention_context.shape[0]):
            raise PipelineContractError("Qwen PP row_ids are outside the bound update context")
        attention_mask = attention_context.index_select(0, row_indices)
        position_ids = position_context.index_select(1, row_indices)
        if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
            raise PipelineContractError("Qwen PP hidden states and local attention context do not align")
        # Keep a zero-gradient edge to row_ids so upstream Schedule1F1B receives
        # a defined gradient for every floating-point pipeline output.
        hidden_states = hidden_states + row_ids.to(hidden_states.dtype).view(-1, 1, 1) * 0
        text_position_ids = position_ids[0]
        rope_position_ids = position_ids[1:]
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = self.model.rotary_emb(hidden_states, rope_position_ids)
        for global_index in range(self.layer_start, self.layer_end):
            layer = self.model.layers[str(global_index)]
            layer_mask = attention_mask if layer.layer_type == "linear_attention" else causal_mask
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
        if self.is_last:
            hidden_states = self.model.norm(hidden_states)
            return hidden_states
        return hidden_states, row_ids

    def pipeline_input_args(self, microbatch_rows: int, sequence_length: int) -> tuple[torch.Tensor, ...]:
        if microbatch_rows <= 0 or sequence_length < 2:
            raise ValueError("Qwen pipeline example shape must be positive with at least two tokens")
        hidden_size = int(self.config.hidden_size)
        ids = torch.empty((microbatch_rows, sequence_length), dtype=torch.long, device="meta")
        row_ids = torch.empty((microbatch_rows,), dtype=torch.float32, device="meta")
        hidden = torch.empty((microbatch_rows, sequence_length, hidden_size), dtype=torch.bfloat16, device="meta")
        if self.is_first:
            return (ids, row_ids)
        return hidden, row_ids

    def pipeline_output_args(self, microbatch_rows: int, sequence_length: int) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if microbatch_rows <= 0 or sequence_length < 2:
            raise ValueError("Qwen pipeline example shape must be positive with at least two tokens")
        hidden = torch.empty(
            (microbatch_rows, sequence_length, int(self.config.hidden_size)),
            dtype=torch.bfloat16,
            device="meta",
        ).requires_grad_(True)
        if self.is_last:
            return hidden
        row_ids = torch.empty((microbatch_rows,), dtype=torch.float32, device="meta").requires_grad_(True)
        return hidden, row_ids

    @staticmethod
    def input_chunk_specs():
        from torch.distributed.pipelining.microbatch import TensorChunkSpec

        return (
            TensorChunkSpec(0),
            TensorChunkSpec(0),
        )


def _construct_layer_meta(config: Any, layer_index: int, modeling_module: Any) -> nn.Module:
    # The optional fused gated norm constructor explicitly asks CUDA for its
    # device. The ordinary HF norm has the same numerical definition and keeps
    # the meta skeleton independent of all GPU devices.
    with _HF_LAYER_BUILD_LOCK:
        fused_norm = getattr(modeling_module, "FusedRMSNormGated", None)
        if fused_norm is not None:
            modeling_module.FusedRMSNormGated = None
        try:
            with torch.device("meta"):
                return modeling_module.Qwen3_5DecoderLayer(config, layer_index)
        finally:
            if fused_norm is not None:
                modeling_module.FusedRMSNormGated = fused_norm


def configure_qwen35_attention(config: Any) -> dict[str, str]:
    """Register the Learner's SDPA attention and mask interfaces for Qwen3.5."""
    try:
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask

        from qwen_rl.models.attention import arc_sdpa_attention_forward
    except ImportError as exc:
        raise RuntimeError("Qwen3.5 PP requires the pinned Transformers arc_sdpa interfaces") from exc
    AttentionInterface.register("arc_sdpa", arc_sdpa_attention_forward)
    AttentionMaskInterface.register("arc_sdpa", sdpa_mask)
    try:
        config._attn_implementation = "arc_sdpa"
    except (AttributeError, TypeError) as exc:
        raise PipelineContractError("Qwen3.5 text config cannot select the arc_sdpa implementation") from exc
    if getattr(config, "_attn_implementation", None) != "arc_sdpa":
        raise PipelineContractError("Qwen3.5 config did not retain the arc_sdpa implementation")
    return {"attention_implementation": "arc_sdpa", "attention_mask": "sdpa_mask"}


def _make_lora_config(
    target_names: Sequence[str], lora_rank: int, lora_alpha: int, *, task_type: str | None
):
    from peft import LoraConfig

    return LoraConfig(
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        lora_dropout=0.0,
        bias="none",
        task_type=task_type,
        target_modules=list(target_names),
    )


def _attach_ternary_lora(
    model: nn.Module,
    *,
    target_names: Sequence[str],
    lora_rank: int,
    lora_alpha: int,
    task_type: str | None,
    input_basis_by_module: Mapping[str, str],
) -> tuple[nn.Module, dict[str, Any]]:
    from qwen_rl.models.ternary import attach_ternary_peft
    from qwen_rl.models.ternary_validation import validate_t2_input_basis_inventory

    peft_model, evidence = attach_ternary_peft(
        model,
        _make_lora_config(target_names, lora_rank, lora_alpha, task_type=task_type),
        input_basis_by_module=input_basis_by_module,
    )
    if not isinstance(evidence, dict):
        raise PipelineContractError("attach_ternary_peft did not return its evidence dictionary")
    basis_counts = validate_t2_input_basis_inventory(
        evidence.get("adapter_input_basis_by_module", {}), input_basis_by_module,
    )
    expected = {
        "selected_target_count": len(target_names),
        "rotated_lora_a_count": basis_counts.get("prism_rotated", 0),
        "primal_lora_a_count": basis_counts.get("primal", 0),
        "non_lora_trainable_parameter_count": 0,
        "all_trainable_names_lora_only": True,
        "all_lora_dropout_zero": True,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise PipelineContractError(
                f"T2 PEFT evidence {key}={evidence.get(key)!r}, expected {value!r}"
            )
    selected_names = evidence.get("selected_target_names")
    normalized_selected_names = (
        [name.removeprefix("base_model.model.") for name in selected_names]
        if isinstance(selected_names, list) and all(isinstance(name, str) for name in selected_names)
        else None
    )
    if normalized_selected_names is None or sorted(normalized_selected_names) != sorted(target_names):
        raise PipelineContractError("T2 PEFT evidence target names differ from the Qwen stage inventory")
    if int(evidence.get("merge_guard_count", 0)) <= 0:
        raise PipelineContractError("T2 immutable-base merge guard evidence is missing")
    if int(evidence.get("trainable_lora_parameter_count", 0)) <= 0:
        raise PipelineContractError("T2 PEFT evidence reports no trainable LoRA parameters")
    return peft_model, dict(evidence)


def build_qwen35_pipeline_stage(
    config: Any,
    provider: Qwen35TernaryProvider,
    *,
    stage_index: int,
    lora_rank: int,
    lora_alpha: int,
    layer_ranges: Sequence[tuple[int, int]] = PIPELINE_LAYER_RANGES,
    lora_target_suffixes: Sequence[str] = DEFAULT_LORA_TARGET_SUFFIXES,
    device: torch.device | str,
    gradient_checkpointing: bool = True,
) -> nn.Module:
    """Load only this process's four-stage Qwen3.5 T2/LoRA partition."""
    if isinstance(config, dict):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        config = Qwen3_5TextConfig(**config)
    if type(gradient_checkpointing) is not bool:
        raise TypeError("gradient_checkpointing must be a boolean")
    configure_qwen35_attention(config)
    if len(layer_ranges) != 4 or any(
        left[1] != right[0] for left, right in pairwise(layer_ranges)
    ) or layer_ranges[0][0] != 0 or layer_ranges[-1][1] != int(config.num_hidden_layers):
        raise PipelineContractError("layer_ranges must partition every Qwen layer into four contiguous stages")
    if isinstance(stage_index, bool) or not 0 <= stage_index < 4:
        raise PipelineContractError("stage_index must be between zero and three")
    if isinstance(lora_rank, bool) or lora_rank <= 0 or isinstance(lora_alpha, bool) or lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive integers")
    if getattr(config, "tie_word_embeddings", False):
        raise PipelineContractError("this Bonsai2 artifact has untied token embedding and output head")
    from transformers.models.qwen3_5 import modeling_qwen3_5

    device = torch.device(device)
    layer_start, layer_end = layer_ranges[stage_index]
    local_layers: dict[int, nn.Module] = {}
    for layer_index in range(layer_start, layer_end):
        layer = _construct_layer_meta(config, layer_index, modeling_qwen3_5)
        _replace_layer_weights(
            layer,
            layer_index=layer_index,
            config=config,
            provider=provider,
            device=device,
        )
        local_layers[layer_index] = layer

    rotary_emb = modeling_qwen35_rotary(config)
    rotary_emb.to(device)
    embed_tokens = final_norm = output_head = None
    if stage_index == 0:
        embed_tokens = provider.load_embedding(
            "text/token_embedding",
            expected_shape=(int(config.vocab_size), int(config.hidden_size)),
        ).to(device)
    if stage_index == 3:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        with torch.device("meta"):
            final_norm = Qwen3_5RMSNorm(int(config.hidden_size), eps=config.rms_norm_eps)
        final_norm.weight = nn.Parameter(
            provider.read_direct_parameter(
                "text/final_norm",
                expected_shape=(int(config.hidden_size),),
                dtype=torch.bfloat16,
            ).to(device=device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        output_head = provider.load_linear(
            "text/output_head",
            expected_shape=(int(config.vocab_size), int(config.hidden_size)),
            compute_dtype=torch.bfloat16,
        ).to(device)

    stage = Qwen35PipelineStage(
        config,
        stage_index=stage_index,
        layer_start=layer_start,
        layer_end=layer_end,
        layers=local_layers,
        embed_tokens=embed_tokens,
        rotary_emb=rotary_emb,
        final_norm=final_norm,
        output_head=output_head,
    )
    # GRPOTrainer reads and aligns this standard metadata during construction,
    # although generation itself remains owned by the Learner client/Worker.
    from transformers import GenerationConfig

    stage.generation_config = GenerationConfig.from_model_config(config)
    # TRL's Trainer constructor resolves the PEFT base model and installs its
    # own fused causal-LM head. The PP adapter has a separate rank-3 head and
    # uses a stage marker to bypass that whole-model patch during construction.
    stage._qwen35_pipeline_stage = stage_index
    stage.to(device)
    stage_checkpointing = stage.configure_gradient_checkpointing(gradient_checkpointing)
    for parameter in stage.parameters():
        parameter.requires_grad_(False)
    target_names, input_basis_by_module = _identity_lora_target_inventory(
        stage,
        provider,
        layer_range=(layer_start, layer_end),
        fallback_suffixes=lora_target_suffixes,
    )
    peft_model, lora_evidence = _attach_ternary_lora(
        stage,
        target_names=target_names,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        task_type=None,
        input_basis_by_module=input_basis_by_module,
    )
    trainable = [name for name, parameter in peft_model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise PipelineContractError("Qwen pipeline stage must train only LoRA A/B parameters")
    if any(parameter.is_meta for parameter in peft_model.parameters()):
        raise PipelineContractError("Qwen pipeline stage retained an uninitialized meta parameter")
    peft_model.config = config
    peft_model._qwen35_pipeline_stage = stage_index
    peft_model._qwen35_layer_range = (layer_start, layer_end)
    peft_model._qwen35_lora_target_modules = tuple(target_names)
    peft_model._qwen35_build_evidence = dict(lora_evidence)
    peft_model._qwen35_gradient_checkpointing = stage_checkpointing
    peft_model._qwen35_attention = {"attention_implementation": "arc_sdpa", "attention_mask": "sdpa_mask"}
    return peft_model


def _verified_store_evidence(provider: Any) -> dict[str, Any]:
    artifact_id = getattr(provider, "artifact_id", None)
    store_path = getattr(provider, "path", None)
    payload_bytes = getattr(provider, "payload_bytes", None)
    text_config = getattr(provider, "text_config", None)
    text_hashes_fn = getattr(provider, "text_resource_hashes", None)
    if not isinstance(artifact_id, bytes) or len(artifact_id) != 16:
        raise PipelineContractError("TernaryArtifactStore must expose its verified 16-byte artifact_id")
    if store_path is None or isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes <= 0:
        raise PipelineContractError("TernaryArtifactStore identity is incomplete")
    if not isinstance(text_config, dict) or not callable(text_hashes_fn):
        raise PipelineContractError("TernaryArtifactStore must expose text_config and text_resource_hashes()")
    text_hashes = text_hashes_fn()
    if not isinstance(text_hashes, dict) or not text_hashes:
        raise PipelineContractError("tokenizer/config/chat-template resource hashes must be materialized before loading")
    return {
        "artifact_id": artifact_id.hex(),
        "artifact_path": str(store_path),
        "payload_bytes": payload_bytes,
        "text_config_sha256": hashlib.sha256(
            json.dumps(text_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "text_resource_hashes": dict(text_hashes),
    }


def _target_module_names(model: nn.Module, suffixes: Sequence[str]) -> list[str]:
    suffix_set = set(suffixes)
    names = [
        name for name, module in model.named_modules()
        if name.startswith("model.layers.") and name.rsplit(".", 1)[-1] in suffix_set
    ]
    if not names:
        raise PipelineContractError("Qwen3.5 causal model has no requested text LoRA targets")
    return names


def _identity_lora_target_inventory(
    model: nn.Module,
    provider: Any,
    *,
    layer_range: tuple[int, int] | None = None,
    fallback_suffixes: Sequence[str] = DEFAULT_LORA_TARGET_SUFFIXES,
) -> tuple[list[str], dict[str, str]]:
    """Resolve exact PEFT modules and bases from the pinned artifact identity."""
    identity = getattr(getattr(provider, "identity", None), "document", None)
    if not isinstance(identity, dict):
        names = _target_module_names(model, fallback_suffixes)
        from qwen_rl.models.ternary import t2_linear_input_basis

        selected = set(names)
        basis_by_module = {
            name: basis for name, module in model.named_modules()
            if name in selected and (basis := t2_linear_input_basis(module)) is not None
        }
        if set(basis_by_module) != selected:
            missing = sorted(selected - set(basis_by_module))
            raise PipelineContractError(
                "Qwen fallback LoRA targets must have packed-base input metadata: "
                + ", ".join(missing[:8])
            )
        return names, basis_by_module

    from qwen_rl.contracts.lora_targets import (
        gdn_lora_enabled,
        projection_input_bases,
        projection_specs,
    )

    try:
        include_gdn = gdn_lora_enabled(identity)
        specs = projection_specs(identity)
        expected_basis = projection_input_bases(identity)
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineContractError(f"artifact LoRA projection contract is invalid: {exc}") from exc
    if set(specs) != set(expected_basis):
        raise PipelineContractError("artifact LoRA projection specs and basis inventory differ")
    declared_basis = getattr(provider, "input_basis_by_module", None)
    adapter = identity.get("learner", {}).get("adapter", {})
    if declared_basis is None or declared_basis == {}:
        if include_gdn or adapter.get("input_basis") == "per_use":
            raise PipelineContractError("per-use artifact LoRA identity is missing its module-basis map")
    elif not isinstance(declared_basis, Mapping) or dict(declared_basis) != expected_basis:
        raise PipelineContractError("provider LoRA basis map differs from the artifact projection contract")

    if layer_range is None:
        selected = expected_basis
    else:
        start, end = layer_range
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise PipelineContractError("LoRA target layer range is invalid")
        selected = {
            name: basis for name, basis in expected_basis.items()
            if (parts := name.split("."))[0:2] == ["model", "layers"]
            and len(parts) == 5 and parts[2].isdecimal()
            and start <= int(parts[2]) < end
        }
    if not selected:
        raise PipelineContractError("artifact identity selects no LoRA projections for this model stage")
    modules = dict(model.named_modules())
    missing_modules = sorted(set(selected) - set(modules))
    if missing_modules:
        raise PipelineContractError(
            "artifact LoRA target modules are absent from the assembled model: "
            + ", ".join(missing_modules[:8])
        )
    from qwen_rl.models.ternary import t2_linear_input_basis

    for name, basis in selected.items():
        module = modules[name]
        spec = specs[name]
        if (getattr(module, "in_features", None) != spec["input_width"]
                or getattr(module, "out_features", None) != spec["output_width"]):
            raise PipelineContractError(
                f"assembled LoRA projection {name} dimensions differ from identity specs"
            )
        if name.endswith((".linear_attn.in_proj_a", ".linear_attn.in_proj_b")):
            if not include_gdn or basis != "primal":
                raise PipelineContractError(f"dense GDN LoRA projection {name} must use primal input")
        else:
            actual_basis = t2_linear_input_basis(module)
            if actual_basis != basis:
                raise PipelineContractError(
                    f"assembled LoRA projection {name} basis {actual_basis!r} differs from identity {basis!r}"
                )
    return sorted(selected), dict(selected)


def _validate_model_config(config: Any, provider: Any) -> None:
    store_config = provider.text_config
    checks = {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "vocab_size": 248320,
        "num_hidden_layers": 64,
        "tie_word_embeddings": False,
    }
    for key, expected in checks.items():
        observed = getattr(config, key, None)
        manifest_value = store_config.get(key)
        if observed != expected or manifest_value != expected:
            raise PipelineContractError(
                f"Bonsai2 text config mismatch for {key}: model={observed!r}, manifest={manifest_value!r}"
            )
    if list(getattr(config, "layer_types", ())) != list(store_config.get("layer_types", ())):
        raise PipelineContractError("Qwen3.5 layer_types differ from the Ternary artifact config")


def _construct_hf_causal_lm_meta(config: Any, modeling_module: Any) -> nn.Module:
    with _HF_LAYER_BUILD_LOCK:
        fused_norm = getattr(modeling_module, "FusedRMSNormGated", None)
        if fused_norm is not None:
            modeling_module.FusedRMSNormGated = None
        try:
            with torch.device("meta"):
                return modeling_module.Qwen3_5ForCausalLM(config)
        finally:
            if fused_norm is not None:
                modeling_module.FusedRMSNormGated = fused_norm


def build_qwen35_causal_lm(
    config: Any,
    provider: Qwen35TernaryProvider,
    *,
    lora_rank: int,
    lora_alpha: int,
    device: torch.device | str,
    with_lora: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the full T2 Qwen3.5 HF causal model for the existing single Learner engine."""
    if isinstance(config, dict):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        config = Qwen3_5TextConfig(**config)
    _validate_model_config(config, provider)
    from qwen_rl.models.ternary_attention import configure_bounded_t2_attention

    bounded_attention = configure_bounded_t2_attention(config)
    store_evidence = _verified_store_evidence(provider)
    if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or lora_rank <= 0:
        raise ValueError("lora_rank must be a positive integer")
    if isinstance(lora_alpha, bool) or not isinstance(lora_alpha, int) or lora_alpha <= 0:
        raise ValueError("lora_alpha must be a positive integer")
    device = torch.device(device)
    from qwen_rl.models.ternary_blas import configure_ternary_blas

    blas_evidence = configure_ternary_blas(device)
    from transformers.models.qwen3_5 import modeling_qwen3_5

    base = _construct_hf_causal_lm_meta(config, modeling_qwen3_5)
    if len(base.model.layers) != 64:
        raise PipelineContractError("official Qwen3.5 HF skeleton does not contain all 64 decoder layers")
    for layer_index, layer in enumerate(base.model.layers):
        _replace_layer_weights(
            layer,
            layer_index=layer_index,
            config=config,
            provider=provider,
            device=device,
        )
    base.model.embed_tokens = provider.load_embedding(
        "text/token_embedding",
        expected_shape=(int(config.vocab_size), int(config.hidden_size)),
    ).to(device)
    base.model.rotary_emb = modeling_qwen35_rotary(config).to(device)
    _direct_parameter(
        provider,
        base.model.norm,
        "text/final_norm",
        (int(config.hidden_size),),
        device=device,
        dtype=torch.bfloat16,
    )
    base.lm_head = provider.load_linear(
        "text/output_head",
        expected_shape=(int(config.vocab_size), int(config.hidden_size)),
        compute_dtype=torch.bfloat16,
    ).to(device)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    meta_parameters = [name for name, parameter in base.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise PipelineContractError("Qwen3.5 artifact loader left meta parameters: " + ", ".join(meta_parameters[:8]))

    from qwen_rl.models.ternary_gated_norm import install_fused_gated_norms

    gated_norm_evidence = install_fused_gated_norms(base.model.layers)
    rms_norm_evidence = {"implementation": "transformers.Qwen3_5RMSNorm", "device_type": device.type}
    gdn_evidence = {"implementation": "transformers.Qwen3_5GatedDeltaNet"}
    if device.type == "cuda":
        from qwen_rl.models.ternary_rms_norm import install_bounded_qwen_rms_norms
        from qwen_rl.models.ternary_gdn_execution import CompactGDNExecution
        rms_norm_evidence = install_bounded_qwen_rms_norms(base)
        base._t2_gdn_execution = CompactGDNExecution(base.model.layers)
        gdn_evidence = base._t2_gdn_execution.report

    if type(with_lora) is not bool:
        raise TypeError("with_lora must be boolean")
    if not with_lora:
        return base, {**store_evidence, "model_backend": "ternary", "base_frozen": True,
                      "attention": bounded_attention,
                      "gated_norm": gated_norm_evidence, "rms_norm": rms_norm_evidence, "gdn_execution": gdn_evidence, "blas": blas_evidence, "lora_attached": False}

    target_names, input_basis_by_module = _identity_lora_target_inventory(base, provider)
    from qwen_rl.models.ternary_validation import validate_t2_input_basis_inventory
    from qwen_rl.models.ternary import is_t2_linear

    assembled_modules = dict(base.named_modules())
    packed_lora_target_count = sum(
        is_t2_linear(assembled_modules[name]) for name in target_names
    )
    artifact_input_basis = getattr(provider, "input_basis_by_module", None)
    adapter_identity = getattr(getattr(provider, "identity", None), "document", {})
    adapter_contract = adapter_identity.get("learner", {}).get("adapter", {})
    if artifact_input_basis is None or artifact_input_basis == {}:
        if adapter_contract.get("input_basis") == "per_use":
            raise PipelineContractError("per-use T2 identity is missing its module basis inventory")
        artifact_input_basis = input_basis_by_module
    try:
        input_basis_counts = validate_t2_input_basis_inventory(
            input_basis_by_module, artifact_input_basis,
        )
    except (TypeError, ValueError) as exc:
        raise PipelineContractError(f"Qwen T2 input basis inventory is invalid: {exc}") from exc
    peft_model, lora_evidence = _attach_ternary_lora(
        base,
        target_names=target_names,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        task_type="CAUSAL_LM",
        input_basis_by_module=input_basis_by_module,
    )
    identity_scale_evidence = {"implementation": "PEFT vanilla LoRA", "device_type": device.type}
    if device.type == "cuda":
        from qwen_rl.models.ternary_lora_scale import install_identity_scale_lora

        identity_scale_evidence = install_identity_scale_lora(
            peft_model, expected_count=packed_lora_target_count
        )

    full_model = peft_model.get_base_model()
    full_model._ternary_fused_head_adapter = TRLFusedTernaryHead(config, full_model.get_output_embeddings())
    full_model._ternary_fused_head_bridge_required = True
    evidence = {
        **store_evidence,
        "architecture": "Qwen3_5ForCausalLM",
        "component": "text",
        "num_hidden_layers": 64,
        "layer_ranges": [[0, 64]],
        "hidden_size": int(config.hidden_size),
        "vocab_size": int(config.vocab_size),
        "uses_mtp": False,
        "full_vocab_logits_materialized": False,
        "gated_norm": gated_norm_evidence,
        "rms_norm": rms_norm_evidence,
        "gdn_execution": gdn_evidence,
        "blas": blas_evidence,
        "attention": bounded_attention,
        "selected_token_head": "trl.add_fused_lm_head with transient T2 W-prime dequantization",
        "lora_target_modules": list(target_names),
        "lora": lora_evidence,
        "lora_input_basis_counts": input_basis_counts,
        "lora_input_basis_by_module": input_basis_by_module,
        "lora_identity_scale": identity_scale_evidence,
    }
    peft_model._qwen35_ternary_evidence = evidence
    return peft_model, evidence


def modeling_qwen35_rotary(config: Any) -> nn.Module:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    return Qwen3_5TextRotaryEmbedding(config=config, device="cpu")


class _RotatedHiddenBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_states: torch.Tensor | None = None

    def forward(self, *args, **kwargs):
        if self.hidden_states is None:
            raise PipelineContractError("TRL fused head proxy has no final pipeline hidden states")
        return SimpleNamespace(last_hidden_state=self.hidden_states)


class _TernaryHeadWeightView(nn.Module):
    """Expose one temporarily dequantized T2 head weight to TRL's public patch."""

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_weight_value", None)

    @property
    def weight(self) -> torch.Tensor:
        value = self._weight_value
        if not isinstance(value, torch.Tensor):
            raise PipelineContractError("T2 lm_head weight must be bound only while the TRL fused head runs")
        return value

    @property
    def bias(self):
        return None

    @contextmanager
    def bind(self, weight: torch.Tensor):
        if self._weight_value is not None:
            raise PipelineContractError("nested dense T2 output-head bindings are unsupported")
        object.__setattr__(self, "_weight_value", weight)
        try:
            yield
        finally:
            object.__setattr__(self, "_weight_value", None)


class _PipelineCausalLMProxy(nn.Module):
    """Minimal forward bridge so TRL's fused-head patch can score PP hidden states."""

    def __init__(self, config: Any, backbone: nn.Module, output_head: _TernaryHeadWeightView):
        super().__init__()
        self.config = config
        self.base_model = backbone
        self.lm_head = output_head

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, *args, **kwargs):
        raise RuntimeError("call this proxy only with fused_lm_head=True")


class TRLFusedTernaryHead:
    """Keep TRL's selected-token kernel while exposing the T2 head correctly.

    The 2.54 GB BF16 head is transient and is never stored in the model state or
    checkpoint. The backbone rotates final hidden states with the output-head
    sign vector before TRL reads the dequantized W' matrix.
    """

    def __init__(self, config: Any, output_head: nn.Module, *, temperature: float = 1.0):
        if not callable(getattr(output_head, "rotate_inputs", None)):
            raise TypeError("T2 output head must expose rotate_inputs(hidden)")
        if not callable(getattr(output_head, "dequantize_weight", None)):
            raise TypeError("T2 output head must expose dequantize_weight(dtype=...)")
        from trl.trainer.utils import add_fused_lm_head

        self.output_head = output_head
        self.backbone = _RotatedHiddenBackbone()
        self.head_view = _TernaryHeadWeightView()
        self.model = _PipelineCausalLMProxy(config, self.backbone, self.head_view)
        add_fused_lm_head(
            self.model,
            temperature=temperature,
            cast_lm_head_to_fp32=False,
            outputs=("log_probs", "entropy"),
        )

    @contextmanager
    def _bind_hidden_and_weight(self, hidden_states: torch.Tensor):
        rotated = self.output_head.rotate_inputs(hidden_states)
        weight = self.output_head.dequantize_weight(dtype=hidden_states.dtype)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise PipelineContractError("dequantized T2 output head must be a matrix")
        self.backbone.hidden_states = rotated
        try:
            from ..ternary_head_offload import transient_head_scope

            with transient_head_scope(self.output_head, weight), self.head_view.bind(weight):
                yield
        finally:
            self.backbone.hidden_states = None

    def score(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        shift_labels: torch.Tensor | None = None,
    ):
        """Return TRL FusedCausalLMOutput without materializing vocabulary logits."""
        if (labels is None) == (shift_labels is None):
            raise PipelineContractError("provide exactly one of labels or shift_labels to the fused T2 head")
        with self._bind_hidden_and_weight(hidden_states):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                shift_labels=shift_labels,
                fused_lm_head=True,
            )

    def score_policy_tokens(
        self,
        trainer: Any,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_length: int,
        policy_target_mask: torch.Tensor,
    ):
        """Run the existing LearnerGRPOTrainer fused scorer over selected targets."""
        with trainer._policy_target_mask_scope(policy_target_mask), self._bind_hidden_and_weight(hidden_states):
            return trainer._get_per_token_logps_and_entropies(
                self.model,
                input_ids,
                attention_mask,
                completion_length,
                compute_entropy=True,
                compute_aux_loss=False,
            )

    def dapo_microbatch_loss(self, trainer: Any, hidden_states: torch.Tensor, inputs: dict[str, Any]) -> torch.Tensor:
        """Let the pinned TRL DAPO code own clipping, masks, and token reduction."""
        target_mask = trainer._completion_policy_mask(inputs)
        with trainer._policy_target_mask_scope(target_mask), self._bind_hidden_and_weight(hidden_states):
            return trainer._compute_loss(self.model, inputs)


def make_pipeline_trl_dapo_loss_fn(
    trainer: Any,
    head: TRLFusedTernaryHead,
    batch_context: Any,
    *,
    process_group: Any,
    microbatches: int,
):
    """Bind packed microbatch metadata to the existing TRL DAPO implementation."""
    if not callable(getattr(batch_context, "trl_inputs", None)):
        raise TypeError("batch_context must expose trl_inputs(target_microbatch, trainer, process_group, microbatches)")

    def loss_fn(final_hidden: torch.Tensor, target_microbatch: torch.Tensor) -> torch.Tensor:
        inputs = batch_context.trl_inputs(
            target_microbatch,
            trainer=trainer,
            process_group=process_group,
            microbatches=microbatches,
        )
        return head.dapo_microbatch_loss(trainer, final_hidden, inputs)

    return loss_fn


def make_pipeline_proximal_score_fn(
    trainer: Any,
    head: TRLFusedTernaryHead,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    tool_mask: torch.Tensor,
):
    """Freeze p with the same TRL target mask and selected-token head as DAPO."""
    if completion_mask.shape != tool_mask.shape or completion_mask.ndim != 2:
        raise PipelineContractError("completion and tool masks must match [batch, completion]")
    policy_target_mask = completion_mask.bool() & tool_mask.bool()

    def score_fn(final_hidden: torch.Tensor) -> torch.Tensor:
        logps, _entropy, _aux = head.score_policy_tokens(
            trainer,
            final_hidden,
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_length=int(completion_mask.shape[1]),
            policy_target_mask=policy_target_mask,
        )
        return logps.detach()

    return score_fn


def install_trl_fused_head_bridge(model: nn.Module) -> dict[str, Any]:
    """Install the T2-aware forward bridge after TRL's normal fused-head patch."""
    base_model = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    head_adapter = getattr(base_model, "_ternary_fused_head_adapter", None)
    if not isinstance(head_adapter, TRLFusedTernaryHead):
        output_head = base_model.get_output_embeddings()
        head_adapter = TRLFusedTernaryHead(base_model.config, output_head)
        base_model._ternary_fused_head_adapter = head_adapter
    current_forward = base_model.forward
    if getattr(current_forward, "__func__", None) is _ternary_fused_lm_head_forward:
        return {
            "installed": True,
            "idempotent_reuse": True,
            "head": "trl_chunked_selected_token_t2",
        }

    base_model._ternary_fused_head_bridge_original_forward = current_forward
    base_model.forward = types.MethodType(_ternary_fused_lm_head_forward, base_model)
    return {
        "installed": True,
        "idempotent_reuse": False,
        "head": "trl_chunked_selected_token_t2",
        "dense_head_bytes_per_call": int(base_model.config.vocab_size)
        * int(base_model.config.hidden_size)
        * 2,
    }


def _ternary_fused_lm_head_forward(
    self,
    *args,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep=0,
    fused_lm_head=False,
    shift_labels=None,
    **kwargs,
):
    """T2-aware selected-head dispatch installed after TRL patches a full model."""
    if args:
        if input_ids is not None or len(args) > 1:
            raise PipelineContractError("ternary fused-head bridge accepts one positional input_ids tensor")
        input_ids = args[0]
    original_forward = self._ternary_fused_head_bridge_original_forward
    if not fused_lm_head:
        kwargs.pop("fused_lm_head", None)
        kwargs.pop("shift_labels", None)
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        **kwargs,
    )
    return self._ternary_fused_head_adapter.score(
        outputs.last_hidden_state,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        shift_labels=shift_labels,
    )
