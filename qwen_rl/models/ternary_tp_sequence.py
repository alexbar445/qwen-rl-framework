"""Sequence-parallel residual storage for the experimental packed-T2 TP path.

The decoder kernels still run on a complete token sequence. Between decoder
layers, each rank stores only its contiguous token shard. This keeps the
residuals captured by Transformers' GradientCheckpointingLayer local while
preserving the ordinary model modules and their state-dict names.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd import Function

RECIPE = "qwen35_t2_sequence_parallel_residuals/v1"
SUPPORTED_WORLD_SIZES = (2, 4)
_LAYER_CLASS_NAME = "Qwen3_5DecoderLayer"
_DYNAMIC_LAYER_TYPES: dict[type, type] = {}


def _world(group=None) -> tuple[int, int]:
    if not dist.is_initialized():
        raise RuntimeError("sequence parallelism requires an initialized process group")
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if size not in SUPPORTED_WORLD_SIZES or not 0 <= rank < size:
        raise ValueError("sequence parallelism supports TP groups of two or four ranks")
    return size, rank


def _partition(total_tokens: int, size: int) -> tuple[int, ...]:
    if type(total_tokens) is not int or total_tokens < 1:
        raise ValueError("the global sequence must contain at least one token")
    quotient, remainder = divmod(total_tokens, size)
    return tuple(quotient + (rank < remainder) for rank in range(size))


def _offsets(lengths: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    offset = 0
    for length in lengths:
        result.append(offset)
        offset += length
    return tuple(result)


def _all_gather_variable_tokens(
    value: torch.Tensor,
    *,
    total_tokens: int,
    group=None,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    size, rank = _world(group)
    if value.ndim != 3:
        raise ValueError("sequence-parallel hidden states must have shape [batch, tokens, width]")
    if type(total_tokens) is not int or total_tokens < 1:
        raise ValueError("global token count must be a positive integer")
    lengths = _partition(total_tokens, size)
    if value.shape[1] != lengths[rank]:
        raise ValueError(
            f"rank {rank} owns {lengths[rank]} tokens for global length {total_tokens}, "
            f"received {value.shape[1]}"
        )
    if value.shape[0] < 1 or value.shape[2] < 1:
        raise ValueError("hidden state batch and width must be positive")

    # Equal shards need no padding. The current Torch single-buffer API writes
    # directly into one rank-major output; for B=1 that is already the compact
    # [batch, tokens, width] order, so no receive list or cat allocation is
    # needed. Larger batches need one layout copy to restore batch-major order.
    if total_tokens % size == 0:
        gathered = value.new_empty((size * value.shape[0], value.shape[1], value.shape[2]))
        input_value = value if value.is_contiguous() else value.contiguous()
        dist.all_gather_single(gathered, input_value, group=group)
        rank_major = gathered.view(size, value.shape[0], value.shape[1], value.shape[2])
        if value.shape[0] == 1:
            full = rank_major.view(1, total_tokens, value.shape[2])
        else:
            full = rank_major.permute(1, 0, 2, 3).reshape(
                value.shape[0], total_tokens, value.shape[2]
            )
        return full, lengths

    # Every rank pads to the same max shard length for the backend collective,
    # then trims its received tensor before concatenating real tokens only.
    padded_length = max(lengths)
    padded = value.new_zeros((value.shape[0], padded_length, value.shape[2]))
    if value.shape[1]:
        padded[:, : value.shape[1]].copy_(value)
    received = [torch.empty_like(padded) for _ in range(size)]
    dist.all_gather(received, padded, group=group)
    full = torch.cat([part[:, :length] for part, length in zip(received, lengths, strict=True)], dim=1)
    return full, lengths


class _ScatterSequence(Function):
    @staticmethod
    def forward(ctx, full: torch.Tensor, group):
        size, rank = _world(group)
        if full.ndim != 3:
            raise ValueError("sequence-parallel hidden states must have shape [batch, tokens, width]")
        lengths = _partition(full.shape[1], size)
        offset = _offsets(lengths)[rank]
        ctx.group = group
        ctx.total_tokens = full.shape[1]
        # copy_ is deliberate: a narrow view would retain the full hidden-state
        # storage and defeat residual sharding under checkpointing.
        return full[:, offset : offset + lengths[rank]].clone()

    @staticmethod
    def backward(ctx, local_gradient: torch.Tensor):
        full_gradient, _ = _all_gather_variable_tokens(
            local_gradient, total_tokens=ctx.total_tokens, group=ctx.group
        )
        return full_gradient, None


class _GatherSequence(Function):
    @staticmethod
    def forward(ctx, local: torch.Tensor, total_tokens: int, group):
        full, lengths = _all_gather_variable_tokens(local, total_tokens=total_tokens, group=group)
        ctx.group = group
        ctx.lengths = lengths
        ctx.rank = dist.get_rank(group)
        ctx.offset = _offsets(lengths)[ctx.rank]
        ctx.local_length = lengths[ctx.rank]
        return full

    @staticmethod
    def backward(ctx, full_gradient: torch.Tensor):
        # Decoder output is replicated after TP's paired projections. Each rank
        # selects the gradient for its token shard; there is no token-gradient SUM.
        local = full_gradient[:, ctx.offset : ctx.offset + ctx.local_length].clone()
        return local, None, None


def scatter_sequence_shard(full: torch.Tensor, *, group=None) -> torch.Tensor:
    """Return this rank's balanced contiguous token shard from a full tensor."""
    return _ScatterSequence.apply(full, group)


def gather_sequence_shards(
    local: torch.Tensor, *, total_tokens: int, group=None
) -> torch.Tensor:
    """Gather contiguous token shards, padding transfers but trimming fake tokens."""
    return _GatherSequence.apply(local, total_tokens, group)


def _full_token_count(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    position_embeddings = kwargs.get("position_embeddings")
    if position_embeddings is None and len(args) > 1:
        position_embeddings = args[1]
    if isinstance(position_embeddings, (tuple, list)) and position_embeddings:
        first = position_embeddings[0]
        if isinstance(first, torch.Tensor) and first.ndim >= 2:
            return int(first.shape[-2])
    position_ids = kwargs.get("position_ids")
    if position_ids is None and len(args) > 2:
        position_ids = args[2]
    if isinstance(position_ids, torch.Tensor) and position_ids.ndim in (2, 3):
        return int(position_ids.shape[-1])
    raise RuntimeError(
        "Qwen decoder sequence parallelism requires global position_embeddings or position_ids"
    )


def _sequence_parallel_layer_type(original_type: type) -> type:
    cached = _DYNAMIC_LAYER_TYPES.get(original_type)
    if cached is not None:
        return cached

    parallel_type: type | None = None

    def forward(self, *args, **kwargs):
        if args:
            hidden_states, tail = args[0], args[1:]
            call_args = (hidden_states, *tail)
            call_kwargs = kwargs
        else:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                raise TypeError("decoder forward must receive hidden_states")
            call_args = ()
            call_kwargs = dict(kwargs)

        group = self._sequence_parallel_group
        total_tokens = _full_token_count(call_args, call_kwargs)
        full_hidden = gather_sequence_shards(hidden_states, total_tokens=total_tokens, group=group)
        if call_args:
            full_args = (full_hidden, *call_args[1:])
            output = super(parallel_type, self).forward(*full_args, **call_kwargs)
        else:
            full_kwargs = dict(call_kwargs)
            full_kwargs["hidden_states"] = full_hidden
            output = super(parallel_type, self).forward(**full_kwargs)

        if isinstance(output, torch.Tensor):
            return scatter_sequence_shard(output, group=group)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (scatter_sequence_shard(output[0], group=group), *output[1:])
        raise TypeError("decoder forward must return hidden states as a tensor or tuple[0]")

    parallel_type = type(
        _LAYER_CLASS_NAME,
        (original_type,),
        {
            "__module__": original_type.__module__,
            "__qualname__": original_type.__qualname__,
            "__doc__": original_type.__doc__,
            "forward": forward,
        },
    )
    _DYNAMIC_LAYER_TYPES[original_type] = parallel_type
    return parallel_type


@dataclass
class SequenceParallelHandle:
    """Reversible installation state; ordinary state_dict keys stay unchanged."""

    text_model: nn.Module
    group: Any
    layers: tuple[nn.Module, ...]
    original_types: tuple[type, ...]
    hooks: tuple[Any, ...]
    current_tokens: ContextVar[int | None]
    call_stack: ContextVar[tuple[Any, ...]]
    closed: bool = False

    @property
    def evidence(self) -> dict[str, Any]:
        size, rank = _world(self.group)
        return {
            "recipe": RECIPE,
            "rank": rank,
            "world_size": size,
            "layer_count": len(self.layers),
            "activation_layout": "balanced_contiguous_token_shards_between_layers",
            "decoder_compute": "full_sequence_per_rank_with_TP_replicated_output",
            "gather_backward": "slice_local_token_gradient",
            "scatter_backward": "all_gather_local_token_gradients_no_sum",
            "trainable_final_norm_gradient": "sum_token_partials",
            "checkpoint_input": "local_token_shard",
            "state_dict_fqns_preserved": True,
            "full_module_pickle_supported": False,
            "production_enabled": False,
        }

    def close(self) -> None:
        if self.closed:
            return
        for hook in self.hooks:
            hook.remove()
        for layer, original_type in zip(self.layers, self.original_types, strict=True):
            layer.__class__ = original_type
            delattr(layer, "_sequence_parallel_group")
        self.closed = True


def _bind_input(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, index: int):
    if len(args) > index:
        return args[index], ("args", index)
    return kwargs.get(name), ("kwargs", name)


def _set_input(args: tuple[Any, ...], kwargs: dict[str, Any], slot, value):
    kind, key = slot
    if kind == "args":
        mutable = list(args)
        mutable[key] = value
        return tuple(mutable), kwargs
    copied = dict(kwargs)
    copied[key] = value
    return args, copied


_REPLICATED_TENSOR_METADATA = (
    "position_ids", "seq_idx", "cu_seq_lens_q", "cu_seq_lens_k",
    "cu_seqlens_q", "cu_seqlens_k", "cache_position",
)
_REPLICATED_SCALAR_METADATA = (
    "max_length_q", "max_length_k", "max_seqlen_q", "max_seqlen_k",
    "max_seqlength_q", "max_seqlength_k",
)
_METADATA_DTYPE_CODES = {
    torch.bool: 1, torch.uint8: 2, torch.int8: 3, torch.int16: 4,
    torch.int32: 5, torch.int64: 6, torch.float16: 7, torch.bfloat16: 8,
    torch.float32: 9, torch.float64: 10,
}
_METADATA_DIMS = 4
_METADATA_FIELDS = 1 + 1 + 1 + 1 + 1 + _METADATA_DIMS + 1
_PACKED_BOUNDARY_METADATA = (
    "seq_idx", "cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens_q", "cu_seqlens_k",
)
_INTEGER_METADATA_DTYPES = (torch.int32, torch.int64)


def _packed_metadata_contract_valid(
    input_ids: Any, position_ids: Any, kwargs: dict[str, Any]
) -> bool:
    """Require and verify the complete official packer contract when boundaries exist."""
    has_packed_metadata = any(kwargs.get(name) is not None for name in _PACKED_BOUNDARY_METADATA)
    if not has_packed_metadata:
        return True
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return False

    sequence_ids = kwargs.get("seq_idx")
    cu_q = kwargs.get("cu_seq_lens_q")
    cu_k = kwargs.get("cu_seq_lens_k")
    if not all(isinstance(value, torch.Tensor) for value in (
        position_ids, sequence_ids, cu_q, cu_k
    )):
        return False

    token_count = int(input_ids.shape[-1])
    if (position_ids.ndim not in (2, 3) or position_ids.shape[-1] != token_count
            or position_ids.dtype not in _INTEGER_METADATA_DTYPES
            or sequence_ids.ndim not in (1, 2) or sequence_ids.shape[-1] != token_count
            or sequence_ids.numel() != token_count
            or sequence_ids.dtype not in _INTEGER_METADATA_DTYPES):
        return False
    if (cu_q.ndim != 1 or cu_k.ndim != 1 or cu_q.numel() < 2
            or cu_q.shape != cu_k.shape or cu_q.dtype not in _INTEGER_METADATA_DTYPES
            or cu_k.dtype not in _INTEGER_METADATA_DTYPES):
        return False
    tensors = (position_ids, sequence_ids, cu_q, cu_k)
    if any(value.device != input_ids.device for value in tensors):
        return False

    cu_q64 = cu_q.to(torch.int64)
    cu_k64 = cu_k.to(torch.int64)
    if (not torch.equal(cu_q64, cu_k64) or int(cu_q64[0]) != 0
            or int(cu_q64[-1]) != token_count
            or not bool((cu_q64[1:] > cu_q64[:-1]).all())):
        return False

    # Official flattening assigns sequence IDs in offset order and resets every
    # position axis to 0..length-1 at the same cu_seq_lens boundaries.
    token_positions = torch.arange(token_count, device=input_ids.device, dtype=torch.int64)
    expected_sequence_ids = torch.bucketize(token_positions, cu_q64[1:-1], right=True)
    if not torch.equal(sequence_ids.reshape(-1).to(torch.int64), expected_sequence_ids):
        return False
    local_positions = token_positions - cu_q64.index_select(0, expected_sequence_ids)
    position_rows = position_ids.reshape(-1, token_count).to(torch.int64)
    if not torch.equal(position_rows, local_positions.unsqueeze(0).expand_as(position_rows)):
        return False

    # Some backbones accept aliases as well as the collator's canonical names.
    # If present, aliases must describe the exact same packed boundaries.
    for alias_name, canonical in (
        ("cu_seqlens_q", cu_q64), ("cu_seqlens_k", cu_k64),
    ):
        alias = kwargs.get(alias_name)
        if alias is not None and (
            not isinstance(alias, torch.Tensor) or alias.ndim != 1
            or alias.dtype not in _INTEGER_METADATA_DTYPES
            or alias.device != input_ids.device
            or not torch.equal(alias.to(torch.int64), canonical)
        ):
            return False
    return True


def _encode_replicated_metadata(value: Any) -> tuple[list[int], bool]:
    """Encode sidecar presence, shape, dtype, or scalar in a fixed small row."""
    encoded = [0] * _METADATA_FIELDS
    if value is None:
        return encoded, True
    if isinstance(value, torch.Tensor):
        dtype_code = _METADATA_DTYPE_CODES.get(value.dtype)
        if (dtype_code is None or value.layout != torch.strided
                or value.ndim > _METADATA_DIMS):
            encoded[:2] = [1, 3]
            return encoded, False
        encoded[:5] = [1, 1, dtype_code, value.ndim, value.numel()]
        encoded[5:5 + value.ndim] = [int(dim) for dim in value.shape]
        return encoded, True
    if isinstance(value, int) and not isinstance(value, bool):
        encoded[0], encoded[1], encoded[-1] = 1, 2, int(value)
        return encoded, True
    encoded[:2] = [1, 3]
    return encoded, False


def _replicated_tensor_matches_rank_zero(
    value: torch.Tensor, *, rank: int, group, control_device: torch.device
) -> bool:
    """Broadcast a sidecar from rank zero, then compare without Python lists."""
    local = value.detach().to(device=control_device).contiguous()
    reference = (local.clone() if rank == 0 else
                 torch.empty(tuple(value.shape), dtype=value.dtype, device=control_device))
    source_rank = 0 if group is None else dist.get_global_rank(group, 0)
    dist.broadcast(reference, src=source_rank, group=group)
    return torch.equal(local, reference)


def _root_preflight(
    module, args, kwargs, *, group, attention_implementation, current_tokens, call_stack
):
    input_ids, _ = _bind_input(args, kwargs, "input_ids", 0)
    attention_mask, mask_slot = _bind_input(args, kwargs, "attention_mask", 1)
    position_ids, position_slot = _bind_input(args, kwargs, "position_ids", 2)
    past_key_values, _ = _bind_input(args, kwargs, "past_key_values", 3)
    inputs_embeds, _ = _bind_input(args, kwargs, "inputs_embeds", 4)
    use_cache, use_cache_slot = _bind_input(args, kwargs, "use_cache", 5)

    valid_input = isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2
    batch, tokens = (int(input_ids.shape[0]), int(input_ids.shape[1])) if valid_input else (-1, -1)
    valid_input = valid_input and batch > 0 and tokens > 0 and inputs_embeds is None
    valid_mask = attention_mask is None
    if attention_mask is not None:
        valid_mask = isinstance(attention_mask, torch.Tensor) and attention_mask.numel() > 0
        if valid_mask:
            valid_mask = bool(torch.all(attention_mask == 1))
    valid_positions = position_ids is None or (
        isinstance(position_ids, torch.Tensor)
        and position_ids.ndim in (2, 3)
        and position_ids.shape[-1] == tokens
    )
    no_cache = past_key_values is None
    has_packed_metadata = any(
        kwargs.get(name) is not None for name in _PACKED_BOUNDARY_METADATA
    )
    valid_packed_backend = (
        not has_packed_metadata or attention_implementation == "flash_attention_2"
    )
    valid_packed_contract = _packed_metadata_contract_valid(
        input_ids, position_ids, kwargs
    )
    valid_aux_outputs = not bool(kwargs.get("output_hidden_states")) and not bool(
        kwargs.get("output_attentions")
    )
    backend = str(dist.get_backend(group)).lower()
    control_device = torch.device("cpu")
    if "nccl" in backend:
        control_device = (
            input_ids.device
            if isinstance(input_ids, torch.Tensor) and input_ids.is_cuda
            else torch.device("cuda", torch.cuda.current_device())
        )
    sidecar_values = {"position_ids": position_ids}
    sidecar_values.update({name: kwargs.get(name) for name in _REPLICATED_TENSOR_METADATA[1:]})
    sidecar_values.update({name: kwargs.get(name) for name in _REPLICATED_SCALAR_METADATA})
    encoded_sidecars: dict[str, list[int]] = {}
    valid_sidecar_types = True
    for name, value in sidecar_values.items():
        encoded, valid_type = _encode_replicated_metadata(value)
        encoded_sidecars[name] = encoded
        valid_sidecar_types &= valid_type
    metadata_values = [
        batch, tokens, int(valid_input), int(valid_mask), int(valid_positions),
        int(no_cache), int(valid_packed_backend), int(valid_aux_outputs),
        int(valid_sidecar_types), int(valid_packed_contract),
    ]
    for name in (*_REPLICATED_TENSOR_METADATA, *_REPLICATED_SCALAR_METADATA):
        metadata_values.extend(encoded_sidecars[name])
    metadata = torch.tensor(
        metadata_values,
        dtype=torch.int64,
        device=control_device,
    )
    size, _ = _world(group)
    replicas = [torch.empty_like(metadata) for _ in range(size)]
    dist.all_gather(replicas, metadata, group=group)
    rows = [tuple(int(value) for value in row.tolist()) for row in replicas]
    if any(row[2] != 1 for row in rows):
        raise ValueError("sequence parallelism requires replicated full input_ids, not inputs_embeds")
    if any(row[3] != 1 for row in rows):
        raise ValueError("sequence parallelism requires unpadded inputs; attention_mask must be all ones or None")
    if any(row[9] != 1 for row in rows):
        raise ValueError(
            "packed inputs require official position_ids, seq_idx, and matching "
            "cu_seq_lens_q/k with exact per-sequence resets"
        )
    if any(row[4] != 1 for row in rows):
        raise ValueError("position_ids must be global and match the full input token count")
    if any(row[5] != 1 for row in rows):
        raise ValueError("sequence parallelism does not accept past_key_values")
    if any(row[6] != 1 for row in rows):
        raise ValueError("flattened packed inputs require flash_attention_2 variable-length attention")
    if any(row[7] != 1 for row in rows):
        raise ValueError("sequence parallelism returns only the full final hidden state")
    if len({row[:2] for row in rows}) != 1:
        raise ValueError("all sequence-parallel ranks must receive the same full batch and token count")

    sidecar_metadata: dict[str, tuple[int, ...]] = {}
    metadata_mismatch_names = []
    sidecar_offset = 10
    for name in (*_REPLICATED_TENSOR_METADATA, *_REPLICATED_SCALAR_METADATA):
        reference = rows[0][sidecar_offset:sidecar_offset + _METADATA_FIELDS]
        sidecar_metadata[name] = reference
        if any(row[sidecar_offset:sidecar_offset + _METADATA_FIELDS] != reference
               for row in rows[1:]):
            metadata_mismatch_names.append(name)
        sidecar_offset += _METADATA_FIELDS

    replicated_metadata_mismatch = bool(metadata_mismatch_names) or any(row[8] != 1 for row in rows)
    content_mismatch_names = []
    if not replicated_metadata_mismatch:
        rank = dist.get_rank(group)
        for name in _REPLICATED_TENSOR_METADATA:
            metadata_row = sidecar_metadata[name]
            if metadata_row[0] == 0:
                continue
            if metadata_row[1] != 1:
                replicated_metadata_mismatch = True
                metadata_mismatch_names.append(name)
                continue
            if not _replicated_tensor_matches_rank_zero(
                sidecar_values[name], rank=rank, group=group, control_device=control_device
            ):
                replicated_metadata_mismatch = True
                content_mismatch_names.append(name)

    invalid_type_names = [
        name for name, metadata_row in sidecar_metadata.items()
        if metadata_row[0] != 0 and metadata_row[1] == 3
    ]
    mismatch_flags = [int(replicated_metadata_mismatch)]
    mismatch_flags.extend(int(name in content_mismatch_names) for name in _REPLICATED_TENSOR_METADATA)
    mismatch_flag = torch.tensor(mismatch_flags, dtype=torch.int32, device=control_device)
    if size > 1:
        dist.all_reduce(mismatch_flag, op=dist.ReduceOp.MAX, group=group)
    global_mismatch = mismatch_flag.tolist()
    if global_mismatch[0] or any(global_mismatch[1:]):
        names = set(metadata_mismatch_names + invalid_type_names)
        names.update(name for name, mismatch in zip(
            _REPLICATED_TENSOR_METADATA, global_mismatch[1:], strict=True
        ) if mismatch)
        name_text = ", ".join(sorted(names))
        raise ValueError(
            "TP ranks disagree on replicated sequence metadata"
            + (f": {name_text}" if name_text else "")
        )

    if position_ids is None:
        position_ids = torch.arange(tokens, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        args, kwargs = _set_input(args, kwargs, position_slot, position_ids)
    if attention_mask is not None:
        args, kwargs = _set_input(args, kwargs, mask_slot, None)
    if use_cache is not False:
        args, kwargs = _set_input(args, kwargs, use_cache_slot, False)
    token = current_tokens.set(tokens)
    call_stack.set((*call_stack.get(), token))
    return args, kwargs


def _root_postflight(module, args, kwargs, output, *, current_tokens, call_stack):
    stack = call_stack.get()
    if stack:
        current_tokens.reset(stack[-1])
        call_stack.set(stack[:-1])
    return output


def _tensor_hook(function):
    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("sequence-parallel embedding and final norm must return a tensor")
        return function(output)
    return hook


def _sum_partial_parameter_gradient(gradient: torch.Tensor, *, group=None) -> torch.Tensor:
    # Final norm runs on each rank's local tokens before the gathered hidden is
    # sent to the vocabulary head. A trainable norm therefore has token-partial
    # parameter VJPs that must be summed to recover the full-sequence gradient.
    result = gradient.contiguous().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
    return result


def install_sequence_parallel(text_model: nn.Module, *, group=None) -> SequenceParallelHandle:
    """Install token-sharded residuals on a Qwen3.5 text model in place.

    The embedding output is copied to a local token shard. Each existing decoder
    module keeps its object, parameters, FQNs and hooks; its forward gathers the
    local residual, calls the original implementation, then scatters the result.
    The final norm gathers for the downstream head. Checkpointing therefore sees
    only local decoder inputs. Save and restore with state_dict, not full-module
    pickle.
    """
    _world(group)
    required = ("embed_tokens", "layers", "norm")
    if any(not hasattr(text_model, name) for name in required):
        raise TypeError("expected a Qwen3.5 text backbone with embed_tokens, layers and norm")
    layers = tuple(text_model.layers)
    if not layers:
        raise ValueError("sequence parallelism requires at least one decoder layer")
    config = getattr(text_model, "config", None)
    attention_implementation = getattr(config, "_attn_implementation", None)
    if attention_implementation is not None and attention_implementation not in {
        "sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4"
    }:
        raise ValueError(
            "sequence parallelism requires an attention backend that handles causal masking "
            "without a materialized local-length mask"
        )
    if getattr(config, "output_hidden_states", False) or getattr(config, "output_attentions", False):
        raise ValueError("sequence parallelism does not capture per-layer hidden states or attentions")
    if any(hasattr(layer, "_sequence_parallel_group") for layer in layers):
        raise ValueError("sequence parallelism is already installed on this decoder stack")
    for layer in layers:
        if not isinstance(layer, nn.Module) or not callable(getattr(layer, "forward", None)):
            raise TypeError("every decoder layer must be a torch module with forward")

    current_tokens: ContextVar[int | None] = ContextVar(
        f"qwen35_sequence_parallel_tokens_{id(text_model)}", default=None
    )
    call_stack: ContextVar[tuple[Any, ...]] = ContextVar(
        f"qwen35_sequence_parallel_call_stack_{id(text_model)}", default=()
    )
    original_types = tuple(type(layer) for layer in layers)
    hooks = []

    def preflight(module, args, kwargs):
        return _root_preflight(
            module, args, kwargs, group=group,
            attention_implementation=attention_implementation,
            current_tokens=current_tokens, call_stack=call_stack,
        )

    def postflight(module, args, kwargs, output):
        return _root_postflight(
            module, args, kwargs, output,
            current_tokens=current_tokens, call_stack=call_stack,
        )

    hooks.append(text_model.register_forward_pre_hook(preflight, with_kwargs=True))
    hooks.append(text_model.register_forward_hook(postflight, with_kwargs=True, always_call=True))
    hooks.append(text_model.embed_tokens.register_forward_hook(
        _tensor_hook(lambda output: scatter_sequence_shard(output, group=group))
    ))

    def gather_final_hidden(output):
        total_tokens = current_tokens.get()
        if total_tokens is None:
            raise RuntimeError("final norm ran outside a sequence-parallel text-model forward")
        return gather_sequence_shards(output, total_tokens=total_tokens, group=group)

    hooks.append(text_model.norm.register_forward_hook(_tensor_hook(gather_final_hidden)))
    for parameter in text_model.norm.parameters():
        if parameter.requires_grad:
            hooks.append(parameter.register_hook(
                lambda gradient, group=group: _sum_partial_parameter_gradient(gradient, group=group)
            ))

    installed_layers = []
    try:
        for layer, original_type in zip(layers, original_types, strict=True):
            installed_layers.append((layer, original_type))
            layer._sequence_parallel_group = group
            layer.__class__ = _sequence_parallel_layer_type(original_type)
    except Exception:
        for layer, original_type in reversed(installed_layers):
            layer.__class__ = original_type
            delattr(layer, "_sequence_parallel_group")
        for hook in hooks:
            hook.remove()
        raise

    handle = SequenceParallelHandle(
        text_model=text_model,
        group=group,
        layers=layers,
        original_types=original_types,
        hooks=tuple(hooks),
        current_tokens=current_tokens,
        call_stack=call_stack,
    )
    return handle


__all__ = [
    "RECIPE",
    "SequenceParallelHandle",
    "gather_sequence_shards",
    "install_sequence_parallel",
    "scatter_sequence_shard",
]
