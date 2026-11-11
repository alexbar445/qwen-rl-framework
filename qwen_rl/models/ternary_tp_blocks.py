"""Head-parallel Qwen3.5 attention and Gated DeltaNet for packed T2 weights.

The official Qwen3.5 attention/GDN forward methods remain responsible for RoPE,
attention dispatch, convolution, packed-sequence boundaries, and the FLA scan.
This installer only changes projection shards and GDN head metadata. The output
projections redistribute local head features into complete 1024-wide T2
Hadamard blocks, then sum their local output contributions.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .ternary import FrozenTernaryLinear
from .ternary_tp_layers import (
    ColumnParallelT2LoRALinear,
    RowParallelT2LoRALinear,
    adapter_placement_records,
    replicate_to_tp,
    split_hadamard_blocks,
)
from .ternary_tp_shared_input import install_shared_input_regions

RECIPE = "qwen35_t2_attention_gdn_head_parallel/v1"


def _world(group=None) -> tuple[int, int]:
    if not dist.is_initialized():
        raise RuntimeError("Qwen3.5 attention/GDN TP requires an initialized process group")
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if size not in (2, 4) or not 0 <= rank < size:
        raise ValueError("Qwen3.5 T2 TP supports groups of two or four members")
    return size, rank


def _feature_interval(total: int, world_size: int, rank: int) -> tuple[int, int]:
    if total % world_size:
        raise ValueError("head feature width must divide evenly across TP ranks")
    width = total // world_size
    return rank * width, (rank + 1) * width


def _intersection_width(left_a: int, right_a: int, left_b: int, right_b: int) -> int:
    return max(0, min(right_a, right_b) - max(left_a, left_b))


class _RedistributeHeadFeatures(torch.autograd.Function):
    """Permute source head slices into destination Hadamard-block slices."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, total_features: int, group):
        size, rank = _world(group)
        if value.ndim != 2:
            raise ValueError("head redistribution expects flattened [tokens, local_features]")
        local_width = value.shape[-1]
        source_start, source_end = _feature_interval(total_features, size, rank)
        if source_end - source_start != local_width:
            raise ValueError("local head feature width differs from the TP source interval")

        destinations = [
            split_hadamard_blocks(total_features, size, destination)
            for destination in range(size)
        ]
        send_counts = [
            _intersection_width(source_start, source_end, start, end)
            for start, end in destinations
        ]
        recv_counts = [
            _intersection_width(*_feature_interval(total_features, size, source_rank),
                                *destinations[rank])
            for source_rank in range(size)
        ]
        send_chunks = []
        for (start, end), count in zip(destinations, send_counts, strict=True):
            if count:
                left = max(source_start, start) - source_start
                send_chunks.append(value.narrow(-1, left, count).movedim(-1, 0).contiguous())
        send_buffer = torch.cat(send_chunks, dim=0)
        token_rows = value.shape[0]
        received = value.new_empty((sum(recv_counts), token_rows))
        dist.all_to_all_single(
            received,
            send_buffer,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group,
        )
        ctx.group = group
        ctx.rank = rank
        ctx.send_counts = tuple(send_counts)
        ctx.recv_counts = tuple(recv_counts)
        ctx.input_shape = tuple(value.shape)
        # Source intervals are ordered by global feature index. Their pieces in
        # received are therefore already in destination-block Hadamard order.
        return received.movedim(0, -1).contiguous()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        moved = gradient.movedim(-1, 0).contiguous()
        returned = gradient.new_empty((sum(ctx.send_counts), gradient.shape[0]))
        # This all-to-all is the exact inverse permutation. Each head feature
        # has one owner, so its VJP is routed without a gradient reduction.
        dist.all_to_all_single(
            returned,
            moved,
            output_split_sizes=list(ctx.send_counts),
            input_split_sizes=list(ctx.recv_counts),
            group=ctx.group,
        )
        return returned.movedim(0, -1).reshape(ctx.input_shape).contiguous(), None, None


class RedistributedRowParallelT2LoRALinear(RowParallelT2LoRALinear):
    """Redistribute local heads to full Hadamard blocks, then row-project."""

    def __init__(self, source: FrozenTernaryLinear, *, shard: tuple[int, int],
                 local_feature_width: int,
                 output_chunk_rows: int = 8192,
                 lora_a_weight: torch.Tensor | nn.Parameter | None = None,
                 lora_b_weight: torch.Tensor | nn.Parameter | None = None,
                 scaling: float | None = None, group=None):
        size, rank = _world(group)
        if type(local_feature_width) is not int or local_feature_width <= 0:
            raise ValueError("local head feature width must be a positive integer")
        if type(output_chunk_rows) is not int or output_chunk_rows <= 0:
            raise ValueError("output projection chunk rows must be a positive integer")
        if size * local_feature_width != source.in_features:
            raise ValueError("equal local head features must cover the row-projection input")
        expected_shard = split_hadamard_blocks(source.in_features, size, rank)
        if shard != expected_shard:
            raise ValueError("row projection must own its balanced full 1024-Hadamard-block interval")
        super().__init__(source, shard=shard, lora_a_weight=lora_a_weight,
                         lora_b_weight=lora_b_weight, scaling=scaling,
                         output_mode="replicated", group=group)
        self.local_feature_width = local_feature_width
        self.output_chunk_rows = output_chunk_rows
        self.global_input_width = source.in_features
        source_start, source_end = _feature_interval(source.in_features, size, rank)
        destination_ranges = [
            split_hadamard_blocks(source.in_features, size, destination)
            for destination in range(size)
        ]
        self.send_feature_counts = tuple(
            _intersection_width(source_start, source_end, start, end)
            for start, end in destination_ranges
        )
        self.recv_feature_counts = tuple(
            _intersection_width(*_feature_interval(source.in_features, size, source_rank), *shard)
            for source_rank in range(size)
        )

    def get_extra_state(self) -> dict[str, Any]:
        state = super().get_extra_state()
        state.update({
            "source_head_width": self.local_feature_width,
            "output_chunk_rows": self.output_chunk_rows,
            "global_input_width": self.global_input_width,
            "send_feature_counts": list(self.send_feature_counts),
            "recv_feature_counts": list(self.recv_feature_counts),
            "feature_redistribution": "all_to_all_hadamard_blocks",
            "redistribution_backward": "inverse_all_to_all_no_sum",
            "hadamard_block_size": 1024,
        })
        return state

    def forward(self, local_features: torch.Tensor) -> torch.Tensor:
        self._check_topology()
        if local_features.shape[-1] != self.local_feature_width:
            raise ValueError(
                f"local head feature width must be {self.local_feature_width}, "
                f"got {local_features.shape[-1]}"
            )
        flat = local_features.reshape(-1, self.local_feature_width)
        if flat.shape[0] == 0:
            raise ValueError("attention/GDN output projection cannot redistribute an empty token batch")
        outputs = []
        for begin in range(0, flat.shape[0], self.output_chunk_rows):
            local_chunk = flat[begin:begin + self.output_chunk_rows]
            block_chunk = _RedistributeHeadFeatures.apply(
                local_chunk, self.global_input_width, self.tp_group
            )
            outputs.append(super().forward(block_chunk))
        return torch.cat(outputs, dim=0).reshape(*local_features.shape[:-1], self.out_features)


class ColumnParallelDenseLinear(nn.Module):
    """Frozen dense output-row shard with summed input gradients.

    Qwen3.5 GDN's per-value-head A/B scalars are ordinary BF16/FP32 Linear
    projections in the learner artifact, while Q/K/V/Z use packed T2. Keep
    these small dense projections sharded too, without treating them as T2.
    """

    parallel_axis = "output_rows"
    _t2_tp_column_parallel_input = True

    def __init__(self, source: nn.Linear, *, shard: tuple[int, int], group=None):
        super().__init__()
        if type(source) is not nn.Linear:
            raise TypeError(f"dense GDN TP requires nn.Linear, got {type(source).__name__}")
        size, rank = _world(group)
        if (not isinstance(shard, tuple) or len(shard) != 2
                or type(shard[0]) is not int or type(shard[1]) is not int
                or not 0 <= shard[0] < shard[1] <= source.out_features):
            raise ValueError("dense column-parallel shard must be a valid output-row interval")
        start, end = shard
        self.in_features, self.out_features = source.in_features, source.out_features
        self.local_out_features = end - start
        self.shard_segments = ((start, end),)
        self.tp_rank, self.tp_world_size, self.tp_group = rank, size, group
        local_weight = source.weight.detach()[start:end].contiguous().clone()
        if source.weight.requires_grad:
            self.weight = nn.Parameter(local_weight, requires_grad=True)
        else:
            self.register_buffer("weight", local_weight, persistent=True)
        if source.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            local_bias = source.bias.detach()[start:end].contiguous().clone()
            if source.bias.requires_grad:
                self.bias = nn.Parameter(local_bias, requires_grad=True)
            else:
                self.register_buffer("bias", local_bias, persistent=True)
        self.train(source.training)

    def get_extra_state(self) -> dict[str, Any]:
        start, end = self.shard_segments[0]
        return {"recipe": RECIPE, "rank": self.tp_rank, "world_size": self.tp_world_size,
                "projection": self.parallel_axis,
                "global_shape": [self.out_features, self.in_features],
                "shard_segments": [[start, end]], "format": "dense_linear"}

    def set_extra_state(self, state) -> None:
        if state != self.get_extra_state():
            raise ValueError("dense TP shard state belongs to a different rank, layout or topology")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if _world(self.tp_group) != (self.tp_world_size, self.tp_rank):
            raise RuntimeError("dense TP process-group topology changed")
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"dense column-parallel input width must be {self.in_features}")
        shared_required = getattr(self, "_t2_tp_shared_input_required", False)
        shared_slot = getattr(self, "_t2_tp_shared_input_slot", None)
        if shared_required and shared_slot is None:
            raise RuntimeError("shared dense TP column projection is missing its owner slot")
        replicated = replicate_to_tp(
            inputs, group=self.tp_group,
            shared_slot=shared_slot if shared_required else None,
        )
        return F.linear(replicated, self.weight, self.bias)


@dataclass(frozen=True)
class _AdapterWeights:
    a: torch.Tensor | nn.Parameter | None = None
    b: torch.Tensor | nn.Parameter | None = None
    scaling: float | None = None
    name: str | None = None


def _projection_source(module: nn.Module) -> tuple[FrozenTernaryLinear, _AdapterWeights]:
    """Unwrap one frozen T2 projection and at most one active PEFT adapter."""
    wrapped = hasattr(module, "get_base_layer") and callable(module.get_base_layer)
    source = module.get_base_layer() if wrapped else module
    if type(source) is not FrozenTernaryLinear:
        raise TypeError(
            "Qwen3.5 TP expects FrozenTernaryLinear projection bases; "
            f"got {type(source).__name__}"
        )
    if not wrapped or not hasattr(module, "lora_A"):
        return source, _AdapterWeights()

    names = list(module.lora_A.keys())
    if not names:
        return source, _AdapterWeights()
    active = getattr(module, "active_adapters", names)
    if isinstance(active, str):
        active = [active]
    selected = [name for name in names if name in active]
    if len(selected) != 1:
        raise ValueError("T2 attention/GDN TP requires exactly one active LoRA adapter per projection")
    adapter = selected[0]
    if bool(getattr(module, "merged", False)) or getattr(module, "merged_adapters", ()):
        raise ValueError("merged PEFT adapters cannot be converted into T2 TP shards")
    dropout = module.lora_dropout[adapter]
    if float(getattr(dropout, "p", 0.0)) != 0.0:
        raise ValueError("T2 TP requires zero-dropout LoRA adapters")
    a_module, b_module = module.lora_A[adapter], module.lora_B[adapter]
    if not hasattr(a_module, "weight") or not hasattr(b_module, "weight"):
        raise TypeError("T2 TP requires Linear-compatible PEFT LoRA A and B factors")
    scale = float(module.scaling[adapter])
    return source, _AdapterWeights(a_module.weight, b_module.weight, scale, adapter)


def _parallel_column(module: nn.Module, *, shard: tuple[int, int] | None = None,
                     row_segments: tuple[tuple[int, int], ...] | None = None,
                     group=None) -> ColumnParallelT2LoRALinear:
    source, adapter = _projection_source(module)
    kwargs: dict[str, Any] = {
        "lora_a_weight": adapter.a,
        "lora_b_weight": adapter.b,
        "scaling": adapter.scaling,
        "group": group,
    }
    replacement = (ColumnParallelT2LoRALinear(source, shard=shard, **kwargs)
                   if shard is not None else
                   ColumnParallelT2LoRALinear(source, row_segments=row_segments, **kwargs))
    replacement.train(module.training)
    return replacement


def _parallel_dense_column(module: nn.Module, *, shard: tuple[int, int], group=None) -> ColumnParallelDenseLinear:
    if hasattr(module, "get_base_layer") or hasattr(module, "lora_A"):
        raise TypeError("dense GDN A/B TP projections do not accept PEFT wrappers")
    replacement = ColumnParallelDenseLinear(module, shard=shard, group=group)
    replacement.train(module.training)
    return replacement


def _parallel_redistributed_row(module: nn.Module, *, shard: tuple[int, int],
                                local_feature_width: int, output_chunk_rows: int,
                                group=None) -> RedistributedRowParallelT2LoRALinear:
    source, adapter = _projection_source(module)
    replacement = RedistributedRowParallelT2LoRALinear(
        source, shard=shard, local_feature_width=local_feature_width,
        output_chunk_rows=output_chunk_rows,
        lora_a_weight=adapter.a, lora_b_weight=adapter.b,
        scaling=adapter.scaling, group=group,
    )
    replacement.train(module.training)
    return replacement


def _adapter_placements(fqn: str, module: nn.Module) -> dict[str, dict[str, Any]]:
    if getattr(module, "local_lora_rank", 0) == 0:
        return {}
    return adapter_placement_records(fqn, module)


def _config_value(config: Any, key: str) -> int:
    value = config.get(key) if isinstance(config, Mapping) else getattr(config, key, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Qwen3.5 config must expose a positive integer {key}")
    return value


def _find_layers(model: nn.Module) -> tuple[nn.Module, list[nn.Module], str]:
    get_base_model = getattr(model, "get_base_model", None)
    base = get_base_model() if callable(get_base_model) else model
    candidates = ((base, ""), (getattr(base, "model", None), "model"),
                  (getattr(getattr(base, "model", None), "language_model", None), "model.language_model"),
                  (getattr(base, "language_model", None), "language_model"))
    for candidate, prefix in candidates:
        layers = getattr(candidate, "layers", None) if candidate is not None else None
        if isinstance(layers, (nn.ModuleList, list, tuple)) and layers:
            return base, list(layers), f"{prefix + '.' if prefix else ''}layers"
    raise ValueError("could not locate Qwen3.5 decoder layers under model.layers")


def _replace(parent: nn.Module, name: str, old: nn.Module, new: nn.Module) -> None:
    if getattr(parent, name) is not old:
        raise RuntimeError(f"projection {name} changed while installing TP")
    setattr(parent, name, new)


def _attention_tp(attention: nn.Module, *, config: Any, size: int, rank: int,
                  fqn: str, output_chunk_rows: int, group=None) -> dict[str, Any]:
    heads = _config_value(config, "num_attention_heads")
    kv_heads = _config_value(config, "num_key_value_heads")
    head_dim = _config_value(config, "head_dim")
    hidden = _config_value(config, "hidden_size")
    if heads % size or kv_heads % size or heads % kv_heads:
        raise ValueError("attention Q/KV head counts must divide TP and preserve the GQA ratio")
    local_q_heads, local_kv_heads = heads // size, kv_heads // size
    group_ratio = heads // kv_heads
    if local_q_heads // local_kv_heads != group_ratio:
        raise ValueError("local attention shard does not preserve Qwen3.5 GQA grouping")

    q = attention.q_proj
    q_source, _ = _projection_source(q)
    q_rows_per_head = 2 * head_dim  # HF Qwen3.5 stores Q and gate interleaved per head.
    if q_source.in_features != hidden or q_source.out_features != heads * q_rows_per_head:
        raise ValueError("Qwen3.5 fused Q/gate projection shape differs from its config")
    q_start = rank * local_q_heads * q_rows_per_head
    q_end = q_start + local_q_heads * q_rows_per_head
    q_parallel = _parallel_column(q, shard=(q_start, q_end), group=group)

    kv_records = {}
    kv_replacements: dict[str, tuple[nn.Module, nn.Module]] = {}
    for name in ("k_proj", "v_proj"):
        old = getattr(attention, name)
        source, _ = _projection_source(old)
        if source.in_features != hidden or source.out_features != kv_heads * head_dim:
            raise ValueError(f"Qwen3.5 {name} shape differs from its config")
        start = rank * local_kv_heads * head_dim
        end = start + local_kv_heads * head_dim
        new = _parallel_column(old, shard=(start, end), group=group)
        kv_replacements[name] = (old, new)
        kv_records[name] = {"global_rows": [start, end], "local_heads": local_kv_heads}

    o = attention.o_proj
    o_source, _ = _projection_source(o)
    full_attention_width = heads * head_dim
    if o_source.in_features != full_attention_width or o_source.out_features != hidden:
        raise ValueError("Qwen3.5 attention output projection shape differs from its config")
    block_start, block_end = split_hadamard_blocks(full_attention_width, size, rank)
    local_head_width = local_q_heads * head_dim
    o_parallel = _parallel_redistributed_row(
        o, shard=(block_start, block_end), local_feature_width=local_head_width,
        output_chunk_rows=output_chunk_rows, group=group
    )
    # The upstream attention dispatch repeats each local KV head by the original
    # GQA ratio, so keep this value at its global value and verify it stays valid.
    existing_ratio = getattr(attention, "num_key_value_groups", group_ratio)
    if int(existing_ratio) != group_ratio:
        raise ValueError("upstream attention module has an unexpected num_key_value_groups")
    # Commit all four projections only after every shape, adapter and block-range
    # contract has passed, avoiding half-installed layers on malformed artifacts.
    for name, (old, new) in kv_replacements.items():
        _replace(attention, name, old, new)
    _replace(attention, "q_proj", q, q_parallel)
    _replace(attention, "o_proj", o, o_parallel)
    return {
        "kind": "full_attention",
        "global_heads": heads,
        "global_kv_heads": kv_heads,
        "local_heads": local_q_heads,
        "local_kv_heads": local_kv_heads,
        "head_dim": head_dim,
        "gqa_group_size": group_ratio,
        "q_proj_rows": [q_start, q_end],
        "kv_projections": kv_records,
        "o_proj_input_blocks": [block_start, block_end],
        "o_proj_local_head_width": local_head_width,
        "o_proj_output_chunk_rows": output_chunk_rows,
        "o_proj_feature_redistribution": "all_to_all_hadamard_blocks",
        "o_proj_redistribution_backward": "inverse_all_to_all_no_sum",
        "o_proj_output": "replicated_full_hidden_after_sum",
        "adapter_placements": {
            **_adapter_placements(f"{fqn}.q_proj", q_parallel),
            **_adapter_placements(f"{fqn}.k_proj", attention.k_proj),
            **_adapter_placements(f"{fqn}.v_proj", attention.v_proj),
            **_adapter_placements(f"{fqn}.o_proj", o_parallel),
        },
    }


def _local_conv1d(source: nn.Conv1d, segments: tuple[tuple[int, int], ...]) -> nn.Conv1d:
    channels = sum(end - start for start, end in segments)
    if source.in_channels != source.out_channels or source.groups != source.in_channels:
        raise ValueError("Qwen3.5 GDN convolution must be depthwise over the fused Q/K/V channels")
    weight = torch.cat([source.weight[start:end] for start, end in segments], dim=0).detach().contiguous()
    if weight.shape[0] != channels:
        raise RuntimeError("local GDN convolution channel assembly failed")
    local = nn.Conv1d(
        channels, channels, kernel_size=source.kernel_size, stride=source.stride,
        padding=source.padding, dilation=source.dilation, groups=channels,
        bias=source.bias is not None, padding_mode=source.padding_mode,
        device=source.weight.device, dtype=source.weight.dtype,
    )
    local.weight = nn.Parameter(weight.clone(), requires_grad=source.weight.requires_grad)
    if source.bias is not None:
        bias = torch.cat([source.bias[start:end] for start, end in segments], dim=0).detach().contiguous()
        local.bias = nn.Parameter(bias.clone(), requires_grad=source.bias.requires_grad)
    local.train(source.training)
    return local


def _slice_parameter(source: nn.Parameter, start: int, end: int) -> nn.Parameter:
    if source.ndim != 1 or not 0 <= start < end <= source.shape[0]:
        raise ValueError("GDN head parameter has an invalid shard range")
    return nn.Parameter(source.detach()[start:end].contiguous().clone(), requires_grad=source.requires_grad)


def _gdn_tp(linear: nn.Module, *, config: Any, size: int, rank: int,
            fqn: str, output_chunk_rows: int, group=None) -> dict[str, Any]:
    key_heads = _config_value(config, "linear_num_key_heads")
    value_heads = _config_value(config, "linear_num_value_heads")
    key_head_dim = _config_value(config, "linear_key_head_dim")
    value_head_dim = _config_value(config, "linear_value_head_dim")
    hidden = _config_value(config, "hidden_size")
    if key_heads % size or value_heads % size or value_heads % key_heads:
        raise ValueError("GDN K/V heads must divide TP and preserve the value-to-key grouping")
    local_key_heads = key_heads // size
    local_value_heads = value_heads // size
    if local_value_heads % local_key_heads:
        raise ValueError("local GDN shard must preserve the value-to-key head ratio")
    key_dim, value_dim = key_heads * key_head_dim, value_heads * value_head_dim
    local_key_dim, local_value_dim = local_key_heads * key_head_dim, local_value_heads * value_head_dim
    if (linear.num_k_heads, linear.num_v_heads, linear.head_k_dim, linear.head_v_dim) != (
        key_heads, value_heads, key_head_dim, value_head_dim
    ):
        raise ValueError("upstream Qwen3.5 GDN attributes differ from its config")
    if (linear.key_dim, linear.value_dim, linear.conv_dim) != (
        key_dim, value_dim, 2 * key_dim + value_dim
    ):
        raise ValueError("upstream Qwen3.5 GDN widths differ from its head metadata")

    q_start = rank * local_key_dim
    q_end = q_start + local_key_dim
    k_start = key_dim + q_start
    k_end = k_start + local_key_dim
    v_start = 2 * key_dim + rank * local_value_dim
    v_end = v_start + local_value_dim
    qkv_segments = ((q_start, q_end), (k_start, k_end), (v_start, v_end))
    qkv_old = linear.in_proj_qkv
    qkv_source, _ = _projection_source(qkv_old)
    expected_qkv_rows = 2 * key_dim + value_dim
    if qkv_source.in_features != hidden or qkv_source.out_features != expected_qkv_rows:
        raise ValueError("Qwen3.5 fused GDN Q/K/V projection shape differs from its config")
    qkv_parallel = _parallel_column(qkv_old, row_segments=qkv_segments, group=group)

    local_records = {"in_proj_qkv": [list(x) for x in qkv_segments]}
    adapter_placements = _adapter_placements(f"{fqn}.in_proj_qkv", qkv_parallel)
    projection_replacements: dict[str, tuple[nn.Module, nn.Module]] = {}
    for name, start, width in (
        ("in_proj_z", rank * local_value_dim, local_value_dim),
        ("in_proj_a", rank * local_value_heads, local_value_heads),
        ("in_proj_b", rank * local_value_heads, local_value_heads),
    ):
        old = getattr(linear, name)
        end = start + width
        if not isinstance(old, nn.Linear) or old.in_features != hidden:
            raise ValueError(f"Qwen3.5 GDN {name} input width differs from hidden size")
        expected_out = value_dim if name == "in_proj_z" else value_heads
        if old.out_features != expected_out:
            raise ValueError(f"Qwen3.5 GDN {name} output width differs from its config")
        if name in ("in_proj_a", "in_proj_b"):
            new = _parallel_dense_column(old, shard=(start, end), group=group)
        else:
            new = _parallel_column(old, shard=(start, end), group=group)
        projection_replacements[name] = (old, new)
        local_records[name] = [start, end]
        adapter_placements.update(_adapter_placements(f"{fqn}.{name}", new))

    full_out = linear.out_proj
    out_source, _ = _projection_source(full_out)
    if out_source.in_features != value_dim or out_source.out_features != hidden:
        raise ValueError("Qwen3.5 GDN output projection shape differs from its config")
    block_start, block_end = split_hadamard_blocks(value_dim, size, rank)
    out_parallel = _parallel_redistributed_row(
        full_out, shard=(block_start, block_end), local_feature_width=local_value_dim,
        output_chunk_rows=output_chunk_rows, group=group
    )

    conv_old = linear.conv1d
    conv_local = _local_conv1d(conv_old, qkv_segments)
    a_log = _slice_parameter(linear.A_log, rank * local_value_heads,
                             (rank + 1) * local_value_heads)
    dt_bias = _slice_parameter(linear.dt_bias, rank * local_value_heads,
                               (rank + 1) * local_value_heads)

    # As with attention, finish validation and local allocation before changing
    # the upstream module's forward-visible projection or head attributes.
    for name, (old, new) in projection_replacements.items():
        _replace(linear, name, old, new)
    _replace(linear, "in_proj_qkv", qkv_old, qkv_parallel)
    _replace(linear, "out_proj", full_out, out_parallel)
    _replace(linear, "conv1d", conv_old, conv_local)
    linear.A_log = a_log
    linear.dt_bias = dt_bias
    linear.num_k_heads = local_key_heads
    linear.num_v_heads = local_value_heads
    linear.key_dim = local_key_dim
    linear.value_dim = local_value_dim
    linear.conv_dim = 2 * local_key_dim + local_value_dim
    linear._t2_tp_head_partition = {
        "rank": rank, "world_size": size,
        "key_heads": [rank * local_key_heads, (rank + 1) * local_key_heads],
        "value_heads": [rank * local_value_heads, (rank + 1) * local_value_heads],
        "qkv_row_segments": [list(x) for x in qkv_segments],
        "conv_channel_segments": [list(x) for x in qkv_segments],
        "out_proj_input_blocks": [block_start, block_end],
    }
    return {
        "kind": "gated_delta_net",
        "global_key_heads": key_heads,
        "global_value_heads": value_heads,
        "local_key_heads": local_key_heads,
        "local_value_heads": local_value_heads,
        "key_head_dim": key_head_dim,
        "value_head_dim": value_head_dim,
        "value_to_key_group_size": local_value_heads // local_key_heads,
        "local_projection_segments": local_records,
        "projection_formats": {"in_proj_qkv": "t2", "in_proj_z": "t2",
                               "in_proj_a": "dense_linear", "in_proj_b": "dense_linear"},
        "conv_channel_segments": [list(x) for x in qkv_segments],
        "A_log_rows": [rank * local_value_heads, (rank + 1) * local_value_heads],
        "dt_bias_rows": [rank * local_value_heads, (rank + 1) * local_value_heads],
        "norm_weight": "replicated_per_head_dimension",
        "out_proj_input_blocks": [block_start, block_end],
        "out_proj_local_head_width": local_value_dim,
        "out_proj_output_chunk_rows": output_chunk_rows,
        "out_proj_feature_redistribution": "all_to_all_hadamard_blocks",
        "out_proj_redistribution_backward": "inverse_all_to_all_no_sum",
        "out_proj_output": "replicated_full_hidden_after_sum",
        "adapter_placements": {
            **adapter_placements,
            **_adapter_placements(f"{fqn}.out_proj", out_parallel),
        },
    }


def install_qwen35_attention_gdn_tp(model: nn.Module, *, group=None,
                                    output_chunk_rows: int = 8192) -> dict[str, Any]:
    """Shard all Qwen3.5 full-attention and GDN heads in an initialized TP group.

    Install after LoRA injection and before moving the model to its final device,
    constructing the optimizer, enabling checkpoint recomputation, or compiling.
    Each decoder layer continues to receive and return replicated full hidden states.
    Packed conv sequence indices and FLA cumulative sequence lengths pass through
    the original upstream forward unchanged.
    """
    size, rank = _world(group)
    if type(output_chunk_rows) is not int or output_chunk_rows <= 0:
        raise ValueError("attention/GDN output chunk rows must be a positive integer")
    base, layers, layers_fqn = _find_layers(model)
    config = getattr(base, "config", None)
    config = getattr(config, "text_config", config)
    if config is None:
        raise ValueError("Qwen3.5 model must expose its text config")
    if getattr(base, "_t2_tp_attention_gdn_installed", False):
        raise ValueError("Qwen3.5 attention/GDN TP installation is not repeatable")

    records = []
    attention_count = gdn_count = 0
    for index, layer in enumerate(layers):
        layer_fqn = f"{layers_fqn}.{index}"
        attention = getattr(layer, "self_attn", None)
        linear = getattr(layer, "linear_attn", None)
        if attention is not None:
            record = _attention_tp(attention, config=config, size=size, rank=rank,
                                   fqn=f"{layer_fqn}.self_attn",
                                   output_chunk_rows=output_chunk_rows, group=group)
            records.append({"layer": index, **record})
            attention_count += 1
        elif linear is not None:
            record = _gdn_tp(linear, config=config, size=size, rank=rank,
                             fqn=f"{layer_fqn}.linear_attn",
                             output_chunk_rows=output_chunk_rows, group=group)
            records.append({"layer": index, **record})
            gdn_count += 1
        else:
            raise ValueError(f"Qwen3.5 decoder layer {index} has no attention or GDN module")
    if not records or attention_count + gdn_count != len(layers):
        raise RuntimeError("Qwen3.5 TP did not cover every decoder token mixer")
    shared_input_regions = install_shared_input_regions(model)

    manifest = {
        "recipe": RECIPE,
        "rank": rank,
        "world_size": size,
        "hidden_size": _config_value(config, "hidden_size"),
        "full_attention_layers": attention_count,
        "gdn_layers": gdn_count,
        "layers": records,
        "hidden_state_placement": "replicated_full_input_and_output",
        "output_chunk_rows": output_chunk_rows,
        "feature_redistribution": "all_to_all_hadamard_blocks",
        "shared_input_regions": shared_input_regions,
        "shared_input_reduction": {
            "low_precision_local_sum": "projection_vjps_lift_to_fp32_then_cast_once",
            "tp_sum": "one_per_shared_primal_activation",
            "scope": "owner_forward_with_checkpoint_reentry_cleanup",
        },
        "attention_dispatch": "upstream_qwen35_local_heads",
        "gdn_dispatch": "upstream_qwen35_causal_conv_and_fla",
        "packed_boundaries": {"causal_conv": "seq_idx_preserved", "fla": "cu_seq_lens_q_preserved"},
        "adapter_state_contract": "use_local_peft_adapter_state_and_load_standard_peft_adapter_state",
        "production_enabled": False,
    }
    base._t2_tp_attention_gdn_installed = True
    base._t2_tp_attention_gdn_manifest = manifest
    return manifest


__all__ = [
    "RECIPE",
    "RedistributedRowParallelT2LoRALinear",
    "install_qwen35_attention_gdn_tp",
]
