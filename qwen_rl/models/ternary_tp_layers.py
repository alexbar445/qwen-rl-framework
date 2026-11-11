"""Paired column/row tensor-parallel primitives for packed T2 and LoRA.

Column projections keep the complete 1024-block input transform and return only
their local output rows. A paired row projection consumes whole 1024-feature
input blocks and reduces the partial output, so an MLP never gathers its
17408-wide gate/up activation. LoRA state remains expressible as ordinary PEFT
``lora_A.weight`` / ``lora_B.weight`` tensors through explicit placement maps.

These are building blocks; callers own sequence placement, loss normalization,
gradient synchronization before clipping/Adam, optimizer recovery, and model
assembly.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from .ternary import (
    HADAMARD_BLOCK,
    FrozenTernaryLinear,
    _dequantize_t2_unchecked,
    _TernaryLinearFunction,
    hadamard_rows,
)
from .ternary_tp_shared_input import share_replicated_tp_input

RECIPE = "t2_paired_column_row_lora/v1"
_TP_SIZES = (2, 4)


def _cuda_autocast_enabled() -> bool:
    return bool(torch.is_autocast_enabled("cuda"))


def _lora_linear_input(inputs: torch.Tensor, weight: torch.Tensor, *,
                       autocast_enabled: bool | None = None) -> torch.Tensor:
    """Avoid an explicit large FP32 activation copy under CUDA autocast.

    PyTorch autocast handles the FP32 LoRA master weights at the GEMM. Outside
    CUDA autocast the input is explicitly matched to the PEFT weight dtype, as
    it is in the ordinary adapter path. ``autocast_enabled`` is an internal
    test seam for exercising this device path without allocating a GPU tensor.
    """
    if autocast_enabled is None:
        autocast_enabled = inputs.is_cuda and _cuda_autocast_enabled()
    return inputs if autocast_enabled else inputs.to(dtype=weight.dtype)


def _world(group=None) -> tuple[int, int]:
    if not dist.is_initialized():
        raise RuntimeError("T2 tensor parallelism requires an initialized process group")
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if size not in _TP_SIZES or not 0 <= rank < size:
        raise ValueError("T2 tensor parallelism supports groups of two or four members")
    return size, rank


def _validate_shard(start: int, end: int, total: int, *, block_aligned: bool) -> None:
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= total:
        raise ValueError(f"invalid shard [{start}, {end}) for dimension {total}")
    if block_aligned and (start % HADAMARD_BLOCK or end % HADAMARD_BLOCK):
        raise ValueError("row-parallel T2 input shards must align to full 1024-feature Hadamard blocks")


def split_hadamard_blocks(total_features: int, world_size: int, rank: int) -> tuple[int, int]:
    """Balanced contiguous whole-1024 shard; lower ranks receive the remainder.

    For Qwen's 17408-wide MLP intermediate at TP4 this returns widths
    5120/4096/4096/4096, preserving all seventeen independent Hadamard blocks.
    """
    if type(total_features) is not int or total_features <= 0 or total_features % HADAMARD_BLOCK:
        raise ValueError("total feature width must be a positive multiple of 1024")
    if type(world_size) is not int or world_size not in _TP_SIZES:
        raise ValueError("T2 tensor parallelism supports world sizes two and four")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be inside the tensor-parallel group")
    blocks = total_features // HADAMARD_BLOCK
    if blocks < world_size:
        raise ValueError("at least one complete Hadamard block is required on every rank")
    quotient, remainder = divmod(blocks, world_size)
    local_blocks = quotient + int(rank < remainder)
    begin_block = rank * quotient + min(rank, remainder)
    start = begin_block * HADAMARD_BLOCK
    return start, start + local_blocks * HADAMARD_BLOCK


class _ReplicatedInput(Function):
    """Identity forward; sum partial input gradients from distinct output shards."""
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, gradient):
        result = gradient.contiguous().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=ctx.group)
        return result, None


class _ReducedOutput(Function):
    """Sum partial outputs forward; replicated-output loss has identity backward."""
    @staticmethod
    def forward(ctx, value, group):
        result = value.contiguous().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class _ReduceScatterSequence(Function):
    """Sum then split sequence rows; backward gathers the full sequence gradient."""
    @staticmethod
    def forward(ctx, value, sequence_dim: int, group):
        size, rank = _world(group)
        sequence_dim = sequence_dim % value.ndim
        if sequence_dim == value.ndim - 1:
            raise ValueError("sequence dimension cannot be the feature dimension")
        moved = value.movedim(sequence_dim, 0).contiguous()
        if moved.shape[0] % size:
            raise ValueError("sequence length must divide the tensor-parallel world size")
        ctx.group, ctx.size, ctx.rank = group, size, rank
        ctx.sequence_dim, ctx.input_shape = sequence_dim, tuple(value.shape)
        local_shape = (moved.shape[0] // size, *moved.shape[1:])
        output = torch.empty(local_shape, device=value.device, dtype=value.dtype)
        backend = str(dist.get_backend(group)).lower()
        if backend == "gloo":
            # Gloo's explicitly selected CPU test path uses a full reduction;
            # this branch is never selected for CUDA/NCCL execution.
            dist.all_reduce(moved, op=dist.ReduceOp.SUM, group=group)
            output.copy_(moved.narrow(0, rank * local_shape[0], local_shape[0]))
        elif backend == "nccl":
            dist.reduce_scatter_tensor(output, moved, op=dist.ReduceOp.SUM, group=group)
        else:
            raise RuntimeError(f"sequence reduce-scatter does not support backend {backend!r}")
        return output.movedim(0, sequence_dim).contiguous()

    @staticmethod
    def backward(ctx, gradient):
        moved = gradient.movedim(ctx.sequence_dim, 0).contiguous()
        # all_gather_into_tensor concatenates the contiguous sequence chunks.
        full_shape = list(moved.shape)
        full_shape[0] *= ctx.size
        full = torch.empty(tuple(full_shape), device=gradient.device, dtype=gradient.dtype)
        dist.all_gather_into_tensor(full, moved, group=ctx.group)
        return full.movedim(0, ctx.sequence_dim).contiguous(), None, None


class _SeparateT2RotationVJP(Function):
    """Reuse a rotation value while retaining an independent LoRA transpose VJP."""
    @staticmethod
    def forward(ctx, inputs, rotated_value, signs):
        ctx.input_dtype = inputs.dtype
        ctx.compute_dtype = torch.float64 if inputs.dtype == torch.float64 else torch.float32
        ctx.save_for_backward(signs)
        return rotated_value

    @staticmethod
    def backward(ctx, gradient):
        from .ternary import _hadamard_rows_impl
        (signs,) = ctx.saved_tensors
        grad_input = _hadamard_rows_impl(
            gradient, signs, inverse=False, block_size=HADAMARD_BLOCK,
            output_dtype=ctx.input_dtype, compute_dtype=ctx.compute_dtype, transpose=True,
        )
        return grad_input, None, None


class _RowParallelTernaryPartial(Function):
    """Compute one packed-T2 input-column partial with FP32 accumulation.

    The operands remain in the configured T2 compute dtype. CUDA BF16/FP16
    GEMMs request FP32 output accumulation directly from ``torch.mm``; the
    portable CPU oracle materializes only a transient FP32 operand view. The
    row-parallel SUM then combines partials before the single output cast.
    """
    @staticmethod
    def forward(ctx, inputs, packed_codes, scale_words, bias, compute_dtype):
        if not inputs.is_floating_point():
            raise TypeError("T2 row-parallel inputs must be floating point")
        if inputs.device != packed_codes.device:
            raise ValueError("T2 row-parallel input and packed weights must share a device")
        rows, in_features = packed_codes.shape[0], packed_codes.shape[1] * 4
        if inputs.shape[-1] != in_features:
            raise ValueError(f"T2 row-parallel expected local input width {in_features}")
        ctx.save_for_backward(packed_codes, scale_words)
        ctx.input_shape, ctx.input_dtype = tuple(inputs.shape), inputs.dtype
        ctx.compute_dtype = compute_dtype
        ctx.needs_input_gradient = ctx.needs_input_grad[0]

        weight = _dequantize_t2_unchecked(
            packed_codes, scale_words, rows, in_features, compute_dtype
        )
        flat_inputs = inputs.reshape(-1, in_features)
        if compute_dtype in (torch.bfloat16, torch.float16):
            if inputs.is_cuda:
                # Supported pinned CUDA path: preserve BF16/FP16 operands and
                # accumulate each local row shard in FP32 Tensor Core output.
                low_precision_inputs = flat_inputs.to(dtype=compute_dtype)
                local = torch.mm(
                    low_precision_inputs, weight.transpose(0, 1), out_dtype=torch.float32
                )
            else:
                # CPU/Gloo oracle only; do not use this activation copy on CUDA.
                local = torch.mm(
                    flat_inputs.to(torch.float32), weight.to(torch.float32).transpose(0, 1)
                )
        else:
            local = torch.mm(
                flat_inputs.to(dtype=compute_dtype), weight.transpose(0, 1)
            )
            if local.dtype != torch.float32 and compute_dtype != torch.float64:
                local = local.to(torch.float32)
        if bias is not None:
            local = local + bias.to(device=local.device, dtype=local.dtype)
        return local.reshape(*inputs.shape[:-1], rows)

    @staticmethod
    def backward(ctx, grad_output):
        packed_codes, scale_words = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_gradient:
            in_features = packed_codes.shape[1] * 4
            weight = _dequantize_t2_unchecked(
                packed_codes, scale_words, packed_codes.shape[0], in_features,
                ctx.compute_dtype,
            )
            grad_flat = grad_output.reshape(-1, packed_codes.shape[0]).to(ctx.compute_dtype)
            # Match the base T2 transpose VJP: compute in configured dtype,
            # then cast dInput back to the activation dtype independently.
            grad_input = grad_flat.matmul(weight).reshape(ctx.input_shape).to(ctx.input_dtype)
        return grad_input, None, None, None, None


class _RowParallelLoRAAPartial(Function):
    """Low-precision LoRA-A GEMM with FP32 partial output and master VJP.

    This avoids an explicit FP32 copy of the potentially long row activation.
    The FP32 master A is cast only to the small compute weight used by the
    low-precision GEMM; backward returns an FP32 master gradient.
    """
    @staticmethod
    def forward(ctx, inputs, master_weight):
        if inputs.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("row LoRA-A accumulator expects BF16 or FP16 activations")
        if master_weight.ndim != 2 or inputs.shape[-1] != master_weight.shape[1]:
            raise ValueError("row LoRA-A activation and weight shapes differ")
        weight = master_weight.to(dtype=inputs.dtype)
        ctx.save_for_backward(inputs, weight)
        ctx.input_shape = tuple(inputs.shape)
        ctx.master_weight_dtype = master_weight.dtype
        flat = inputs.reshape(-1, inputs.shape[-1])
        if inputs.is_cuda:
            output = torch.mm(flat, weight.transpose(0, 1), out_dtype=torch.float32)
        else:
            # CPU/Gloo oracle only. The production CUDA path above keeps the
            # long activation BF16 and accumulates directly into FP32 output.
            output = torch.mm(flat.float(), weight.float().transpose(0, 1))
        return output.reshape(*inputs.shape[:-1], master_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1]).to(inputs.dtype)
        grad_input = flat_grad.matmul(weight).reshape(ctx.input_shape)
        if inputs.is_cuda:
            grad_weight = torch.mm(
                flat_grad.transpose(0, 1), flat_inputs, out_dtype=torch.float32
            )
        else:
            grad_weight = torch.mm(flat_grad.float().transpose(0, 1), flat_inputs.float())
        # Match the ordinary BF16/FP16 PEFT A VJP before the cast back to the
        # FP32 master: accumulation may use FP32, but the source adapter VJP is
        # rounded once to its activation dtype.
        return grad_input, grad_weight.to(inputs.dtype).to(ctx.master_weight_dtype)


def replicate_to_tp(value: torch.Tensor, *, group=None,
                    shared_slot: str | None = None) -> torch.Tensor:
    """Map a replicated activation to distinct output shards (backward SUM)."""
    _world(group)
    if shared_slot is not None:
        return share_replicated_tp_input(
            value, group=group, slot=shared_slot,
            create=lambda: _ReplicatedInput.apply(value, group),
        )
    return _ReplicatedInput.apply(value, group)


def reduce_from_tp(value: torch.Tensor, *, group=None) -> torch.Tensor:
    """Sum a feature-sharded contribution to a replicated full output."""
    _world(group)
    return _ReducedOutput.apply(value, group)


def reduce_scatter_sequence(value: torch.Tensor, *, sequence_dim: int = -2, group=None) -> torch.Tensor:
    """Sum row-parallel partial outputs and distribute disjoint sequence chunks."""
    _world(group)
    return _ReduceScatterSequence.apply(value, sequence_dim, group)


class _AllGatherSequence(Function):
    """Gather distinct sequence shards; replicated full-output backward selects local rows."""
    @staticmethod
    def forward(ctx, value, sequence_dim: int, group):
        size, rank = _world(group)
        sequence_dim = sequence_dim % value.ndim
        if sequence_dim == value.ndim - 1:
            raise ValueError("sequence dimension cannot be the feature dimension")
        local = value.movedim(sequence_dim, 0).contiguous()
        parts = [torch.empty_like(local) for _ in range(size)]
        dist.all_gather(parts, local, group=group)
        ctx.group, ctx.size, ctx.rank, ctx.sequence_dim = group, size, rank, sequence_dim
        return torch.cat(parts, dim=0).movedim(0, sequence_dim).contiguous()

    @staticmethod
    def backward(ctx, gradient):
        moved = gradient.movedim(ctx.sequence_dim, 0).contiguous()
        local_shape = (moved.shape[0] // ctx.size, *moved.shape[1:])
        local = torch.empty(local_shape, device=gradient.device, dtype=gradient.dtype)
        backend = str(dist.get_backend(ctx.group)).lower()
        if backend == "gloo":
            dist.all_reduce(moved, op=dist.ReduceOp.SUM, group=ctx.group)
            local.copy_(moved.narrow(0, ctx.rank * local_shape[0], local_shape[0]))
        elif backend == "nccl":
            dist.reduce_scatter_tensor(local, moved, op=dist.ReduceOp.SUM, group=ctx.group)
        else:
            raise RuntimeError(f"sequence reduce-scatter does not support backend {backend!r}")
        return local.contiguous().movedim(0, ctx.sequence_dim), None, None


def gather_sequence_from_tp(value: torch.Tensor, *, sequence_dim: int = -2, group=None) -> torch.Tensor:
    """Gather local sequence chunks to full sequence on every rank."""
    _world(group)
    return _AllGatherSequence.apply(value, sequence_dim, group)


def _tensor_like(value: torch.Tensor | nn.Parameter) -> torch.Tensor:
    return value.detach() if isinstance(value, nn.Parameter) else value


def _parameter(value: torch.Tensor | nn.Parameter | None) -> nn.Parameter | None:
    if value is None:
        return None
    if isinstance(value, nn.Parameter):
        return value
    return nn.Parameter(value.detach().clone())


def _validate_lora(a: torch.Tensor | nn.Parameter | None,
                   b: torch.Tensor | nn.Parameter | None,
                   in_features: int, out_features: int) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    if a is None and b is None:
        return 0, None, None
    if a is None or b is None:
        raise ValueError("LoRA A and B weights must be supplied together")
    a_value, b_value = _tensor_like(a), _tensor_like(b)
    if a_value.ndim != 2 or b_value.ndim != 2:
        raise ValueError("LoRA weights must be matrices A[r,in] and B[out,r]")
    if a_value.shape[1] != in_features or b_value.shape[0] != out_features:
        raise ValueError("LoRA weights do not match the full T2 projection dimensions")
    if a_value.shape[0] != b_value.shape[1] or a_value.shape[0] <= 0:
        raise ValueError("LoRA A/B rank dimensions must match and be positive")
    if a_value.dtype != b_value.dtype or not a_value.is_floating_point():
        raise TypeError("LoRA A/B weights must use the same floating-point dtype")
    return int(a_value.shape[0]), a_value, b_value


def _row_indices(segments: Sequence[tuple[int, int]], device) -> torch.Tensor:
    return torch.cat([torch.arange(start, end, device=device) for start, end in segments])


class _T2TPBase(nn.Module):
    t2_packed_weight_shard = True
    _t2_tp_base = True
    packed_weight_format = "t2_2bit_group128_fp16scale/v1"

    def _set_base(self, source: FrozenTernaryLinear, codes: torch.Tensor,
                  scales: torch.Tensor, signs: torch.Tensor, bias: torch.Tensor | None,
                  *, global_input: int, global_output: int) -> None:
        self.in_features, self.out_features = int(global_input), int(global_output)
        self.output_dtype, self.compute_dtype = source.output_dtype, source.compute_dtype
        self.register_buffer("packed_codes", codes.detach().contiguous().clone(), persistent=True)
        self.register_buffer("scale_words", scales.detach().contiguous().clone(), persistent=True)
        self.register_buffer("input_hadamard_signs", signs.detach().contiguous().to(torch.int8).clone(), persistent=True)
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous().clone(), persistent=True)
        self.tp_group = None

    def iter_packed_weight_buffers(self):
        """Yield only offloadable immutable T2 payload buffers, never a dense weight."""
        yield "packed_codes", self.packed_codes
        yield "scale_words", self.scale_words

    @property
    def packed_weight_nbytes(self) -> int:
        return (self.packed_codes.numel() * self.packed_codes.element_size()
                + self.scale_words.numel() * self.scale_words.element_size())

    def get_extra_state(self) -> dict[str, Any]:
        return {"recipe": RECIPE, "rank": self.tp_rank, "world_size": self.tp_world_size,
                "projection": self.parallel_axis, "global_shape": [self.out_features, self.in_features],
                "shard_segments": [list(x) for x in self.shard_segments],
                "output_mode": getattr(self, "output_mode", None)}

    def set_extra_state(self, state) -> None:
        if state != self.get_extra_state():
            raise ValueError("T2 TP shard state belongs to a different rank, layout or topology")

    def dequantize_weight(self, dtype=torch.float32):
        raise RuntimeError("T2 TP stores only local packed T2 shards; full dense weight is unavailable")

    def _check_topology(self) -> None:
        if _world(self.tp_group) != (self.tp_world_size, self.tp_rank):
            raise RuntimeError("T2 TP process-group topology changed")


class ColumnParallelT2LoRALinear(_T2TPBase):
    """Full-axis T2 input rotation with local output rows and local LoRA B rows.

    ``row_segments`` may describe noncontiguous fused projections such as
    Q/K/V. Local rows are concatenated in listed order; ``shard`` is the common
    single contiguous interval form used by MLP gate/up.
    """
    parallel_axis = "output_rows"
    _t2_tp_column_parallel_input = True

    def __init__(self, source: FrozenTernaryLinear, *, shard: tuple[int, int] | None = None,
                 row_segments: Sequence[tuple[int, int]] | None = None,
                 lora_a_weight: torch.Tensor | nn.Parameter | None = None,
                 lora_b_weight: torch.Tensor | nn.Parameter | None = None,
                 alpha: float | None = None, scaling: float | None = None, group=None):
        super().__init__()
        if type(source) is not FrozenTernaryLinear:
            raise TypeError("column-parallel T2 requires an unsharded FrozenTernaryLinear")
        if source.in_features % HADAMARD_BLOCK:
            raise ValueError("column-parallel T2 input width must contain complete 1024-feature Hadamard blocks")
        size, rank = _world(group)
        if (shard is None) == (row_segments is None):
            raise ValueError("supply exactly one of shard or row_segments")
        segments = [tuple(shard)] if shard is not None else [tuple(x) for x in row_segments]
        if not segments or any(len(x) != 2 for x in segments):
            raise ValueError("row segments must be a nonempty sequence of (start,end) pairs")
        for start, end in segments:
            _validate_shard(start, end, source.out_features, block_aligned=False)
        if any(a[1] > b[0] for a, b in pairwise(segments)):
            # Disjoint segments may be ordered by fused projection component,
            # not by global row number; reject overlap but preserve that order.
            ordered = sorted(segments)
            if any(a[1] > b[0] for a, b in pairwise(ordered)):
                raise ValueError("column row segments cannot overlap")
        indices = _row_indices(segments, source.packed_codes.device)
        local_out = int(indices.numel())
        rank_lora, a_value, b_value = _validate_lora(
            lora_a_weight, lora_b_weight, source.in_features, source.out_features
        )
        self.tp_rank, self.tp_world_size, self.tp_group = rank, size, group
        self.shard_segments = tuple(segments)
        self.local_out_features = local_out
        if alpha is not None and scaling is not None:
            raise ValueError("supply raw alpha or PEFT scaling, not both")
        self.local_lora_rank = rank_lora
        self.lora_scaling = float(scaling if scaling is not None else
                                  ((alpha if alpha is not None else rank_lora) / rank_lora
                                   if rank_lora else 1.0))
        if rank_lora and (not torch.isfinite(torch.tensor(self.lora_scaling)) or self.lora_scaling <= 0):
            raise ValueError("LoRA scaling must be finite and positive")
        local_codes = source.packed_codes.index_select(0, indices)
        local_scales = source.scale_words.index_select(0, indices)
        local_bias = None if source.bias is None else source.bias.index_select(0, indices)
        self._set_base(source, local_codes, local_scales, source.input_hadamard_signs,
                       local_bias, global_input=source.in_features, global_output=source.out_features)
        if rank_lora:
            self.lora_A = nn.Linear(source.in_features, rank_lora, bias=False,
                                    device=source.packed_codes.device, dtype=a_value.dtype)
            self.lora_B = nn.Linear(rank_lora, local_out, bias=False,
                                    device=source.packed_codes.device, dtype=b_value.dtype)
            with torch.no_grad():
                self.lora_A.weight.copy_(a_value)
                self.lora_B.weight.copy_(b_value.index_select(0, indices.to(b_value.device)))
        else:
            self.lora_A = self.lora_B = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._check_topology()
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"column-parallel input width must be {self.in_features}")
        # Map the shared primal activation once. Base and LoRA retain separate
        # Hadamard transpose VJPs below, which accumulate before this one SUM.
        shared_required = getattr(self, "_t2_tp_shared_input_required", False)
        shared_slot = getattr(self, "_t2_tp_shared_input_slot", None)
        if shared_required and shared_slot is None:
            raise RuntimeError("shared T2 TP column projection is missing its owner slot")
        primal_input = replicate_to_tp(
            inputs, group=self.tp_group,
            shared_slot=shared_slot if shared_required else None,
        )
        rotated = hadamard_rows(primal_input, self.input_hadamard_signs, output_dtype=inputs.dtype)
        if self.bias is not None and self.bias.dtype != rotated.dtype:
            bias = self.bias.to(dtype=rotated.dtype)
        else:
            bias = self.bias
        output = _TernaryLinearFunction.apply(
            rotated, self.packed_codes, self.scale_words, bias, self.compute_dtype
        )
        if self.local_lora_rank:
            # Share the forward rotation value but keep the LoRA transpose VJP
            # independent from the frozen-base VJP, matching the PEFT contract.
            adapter_rotated = _SeparateT2RotationVJP.apply(
                primal_input, rotated.detach(), self.input_hadamard_signs
            )
            a_input = _lora_linear_input(adapter_rotated, self.lora_A.weight)
            latent = F.linear(a_input, self.lora_A.weight)
            adapter = F.linear(_lora_linear_input(latent, self.lora_B.weight),
                               self.lora_B.weight) * self.lora_scaling
            output = (output.to(dtype=adapter.dtype) + adapter).to(dtype=output.dtype)
        return output


class RowParallelT2LoRALinear(_T2TPBase):
    """Whole-block input-column shard and reduced full output.

    LoRA A is column-sharded over the input features; its rank activations are
    reduced before replicated LoRA B. ``sequence_shard`` uses reduce-scatter
    over sequence rows and therefore returns disjoint token chunks.
    """
    parallel_axis = "input_columns"

    def __init__(self, source: FrozenTernaryLinear, *, shard: tuple[int, int],
                 lora_a_weight: torch.Tensor | nn.Parameter | None = None,
                 lora_b_weight: torch.Tensor | nn.Parameter | None = None,
                 alpha: float | None = None, scaling: float | None = None,
                 output_mode: str = "replicated", group=None):
        super().__init__()
        if type(source) is not FrozenTernaryLinear:
            raise TypeError("row-parallel T2 requires an unsharded FrozenTernaryLinear")
        if source.in_features % HADAMARD_BLOCK:
            raise ValueError("row-parallel T2 input width must contain complete 1024-feature Hadamard blocks")
        size, rank = _world(group)
        if output_mode not in ("replicated", "sequence_shard"):
            raise ValueError("output_mode must be 'replicated' or 'sequence_shard'")
        start, end = shard
        _validate_shard(start, end, source.in_features, block_aligned=True)
        if end - start < HADAMARD_BLOCK:
            raise ValueError("every row-parallel rank needs at least one full Hadamard block")
        rank_lora, a_value, b_value = _validate_lora(
            lora_a_weight, lora_b_weight, source.in_features, source.out_features
        )
        if rank_lora and output_mode == "sequence_shard" and any(
            p.device != source.packed_codes.device for p in (a_value, b_value)
        ):
            raise ValueError("T2 base and LoRA tensors must share a device")
        self.tp_rank, self.tp_world_size, self.tp_group = rank, size, group
        self.shard_segments = ((start, end),)
        self.local_in_features = end - start
        self.output_mode = output_mode
        if alpha is not None and scaling is not None:
            raise ValueError("supply raw alpha or PEFT scaling, not both")
        self.local_lora_rank = rank_lora
        self.lora_scaling = float(scaling if scaling is not None else
                                  ((alpha if alpha is not None else rank_lora) / rank_lora
                                   if rank_lora else 1.0))
        if rank_lora and (not torch.isfinite(torch.tensor(self.lora_scaling)) or self.lora_scaling <= 0):
            raise ValueError("LoRA scaling must be finite and positive")
        byte_start, byte_end = start // 4, end // 4
        scale_start, scale_end = start // 128, end // 128
        local_codes = source.packed_codes[:, byte_start:byte_end]
        local_scales = source.scale_words[:, scale_start:scale_end]
        local_bias = source.bias if rank == 0 else None
        local_signs = source.input_hadamard_signs[start:end]
        self._set_base(source, local_codes, local_scales, local_signs, local_bias,
                       global_input=source.in_features, global_output=source.out_features)
        # PEFT state is A[r,K], B[N,r]. The local A stores the source's
        # column interval while B remains a complete replica.
        local_a = None if a_value is None else a_value[:, start:end].contiguous()
        if rank_lora:
            self.lora_A = nn.Linear(end - start, rank_lora, bias=False,
                                    device=source.packed_codes.device, dtype=local_a.dtype)
            self.lora_B = nn.Linear(rank_lora, source.out_features, bias=False,
                                    device=source.packed_codes.device, dtype=b_value.dtype)
            with torch.no_grad():
                self.lora_A.weight.copy_(local_a)
                self.lora_B.weight.copy_(b_value)
        else:
            self.lora_A = self.lora_B = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._check_topology()
        if inputs.shape[-1] != self.local_in_features:
            raise ValueError(f"row-parallel local input width must be {self.local_in_features}")
        rotated = hadamard_rows(inputs, self.input_hadamard_signs, output_dtype=inputs.dtype)
        bias = self.bias
        if bias is not None and bias.dtype != rotated.dtype:
            bias = bias.to(dtype=rotated.dtype)
        local_base = _RowParallelTernaryPartial.apply(
            rotated, self.packed_codes, self.scale_words, bias, self.compute_dtype
        )
        local_lora = None
        lora_latent_dtype = None
        if self.local_lora_rank:
            adapter_rotated = _SeparateT2RotationVJP.apply(
                inputs, rotated.detach(), self.input_hadamard_signs
            )
            a_input = _lora_linear_input(adapter_rotated, self.lora_A.weight)
            if a_input.dtype in (torch.bfloat16, torch.float16):
                latent = _RowParallelLoRAAPartial.apply(a_input, self.lora_A.weight)
                lora_latent_dtype = a_input.dtype
            else:
                latent = F.linear(a_input, self.lora_A.weight)
                lora_latent_dtype = latent.dtype
            local_lora = latent
        if local_lora is None:
            if self.output_mode == "replicated":
                return reduce_from_tp(local_base, group=self.tp_group).to(dtype=rotated.dtype)
            return reduce_scatter_sequence(
                local_base, sequence_dim=-2, group=self.tp_group
            ).to(dtype=rotated.dtype)
        # Reduce local base contributions and the rank-sized LoRA A latent in
        # one collective. LoRA B is applied only after the latent is complete,
        # so its replica gradient has the declared mean/SUM semantics.
        reduction_dtype = torch.promote_types(local_base.dtype, local_lora.dtype)
        combined = torch.cat((local_base.to(dtype=reduction_dtype),
                              local_lora.to(dtype=reduction_dtype)), dim=-1)
        if self.output_mode == "replicated":
            reduced = reduce_from_tp(combined, group=self.tp_group)
        else:
            reduced = reduce_scatter_sequence(combined, sequence_dim=-2, group=self.tp_group)
        # Match the original row projection's output dtype after FP32 partial
        # accumulation, and cast the complete A latent to PEFT's live dtype
        # before replicated B consumes it.
        global_base = reduced[..., : self.out_features].to(dtype=rotated.dtype)
        global_latent = reduced[..., self.out_features :].to(dtype=lora_latent_dtype)
        adapter = F.linear(_lora_linear_input(global_latent, self.lora_B.weight), self.lora_B.weight)
        return (global_base + adapter * self.lora_scaling).to(dtype=rotated.dtype)


def adapter_placement_records(canonical_fqn: str, module: _T2TPBase) -> dict[str, dict[str, Any]]:
    """Describe the local LoRA factors using canonical standard PEFT keys."""
    if not canonical_fqn or canonical_fqn.endswith("."):
        raise ValueError("canonical module FQN must be nonempty and have no trailing dot")
    if not isinstance(module, (ColumnParallelT2LoRALinear, RowParallelT2LoRALinear)):
        raise TypeError("placement records require a paired T2 TP linear")
    if module.local_lora_rank == 0:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for factor, parameter in (("lora_A.weight", module.lora_A.weight),
                              ("lora_B.weight", module.lora_B.weight)):
        full_key = f"{canonical_fqn}.{factor}"
        if isinstance(module, ColumnParallelT2LoRALinear):
            if factor == "lora_A.weight":
                record = {"role": "replicated", "global_shape": list(parameter.shape),
                          "local_shape": list(parameter.shape), "shard_dim": None,
                          "shard_start": None, "shard_end": None,
                          "grad_reduce": "sum_partial", "norm_count": "count_once"}
            else:
                segments = [list(x) for x in module.shard_segments]
                continuous = len(segments) == 1
                record = {"role": "sharded", "global_shape": [module.out_features, module.local_lora_rank],
                          "local_shape": list(parameter.shape), "shard_dim": 0,
                          "shard_start": segments[0][0] if continuous else None,
                          "shard_end": segments[0][1] if continuous else None,
                          "grad_reduce": "none", "norm_count": "local_shard"}
                if not continuous:
                    record["shard_segments"] = segments
        else:
            start, end = module.shard_segments[0]
            if factor == "lora_A.weight":
                record = {"role": "sharded", "global_shape": [module.local_lora_rank, module.in_features],
                          "local_shape": list(parameter.shape), "shard_dim": 1,
                          "shard_start": start, "shard_end": end,
                          "grad_reduce": "none", "norm_count": "local_shard"}
            else:
                record = {"role": "replicated", "global_shape": list(parameter.shape),
                          "local_shape": list(parameter.shape), "shard_dim": None,
                          "shard_start": None, "shard_end": None,
                          "grad_reduce": "mean_complete" if module.output_mode == "replicated" else "sum_partial",
                          "norm_count": "count_once"}
        record.setdefault("shard_segments", None)
        record.update({"tp_rank": module.tp_rank, "tp_world_size": module.tp_world_size})
        records[full_key] = record
    return records


def local_peft_adapter_state(module: _T2TPBase, canonical_fqn: str) -> dict[str, torch.Tensor]:
    """Return this rank's factors under ordinary PEFT adapter state keys."""
    if not isinstance(module, (ColumnParallelT2LoRALinear, RowParallelT2LoRALinear)):
        raise TypeError("adapter state requires a paired T2 TP linear")
    if module.local_lora_rank == 0:
        return {}
    return {f"{canonical_fqn}.lora_A.weight": module.lora_A.weight,
            f"{canonical_fqn}.lora_B.weight": module.lora_B.weight}


def load_standard_peft_adapter_state(module: _T2TPBase, canonical_fqn: str,
                                     state: Mapping[str, torch.Tensor]) -> None:
    """Load full standard PEFT tensors into the local row/column factor shards."""
    local = local_peft_adapter_state(module, canonical_fqn)
    records = adapter_placement_records(canonical_fqn, module)
    with torch.no_grad():
        for key, target in local.items():
            if key not in state:
                raise KeyError(f"standard PEFT state is missing {key}")
            source = state[key]
            record = records[key]
            if record["role"] == "replicated":
                value = source
            elif record.get("shard_segments") is not None:
                value = torch.cat([source[start:end] for start, end in record["shard_segments"]], dim=0)
            else:
                value = source.narrow(record["shard_dim"], record["shard_start"],
                                      record["shard_end"] - record["shard_start"])
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"local PEFT tensor shape mismatch for {key}: {tuple(value.shape)} != {tuple(target.shape)}")
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def synchronize_t2_lora_gradients(model: nn.Module, *, group=None) -> dict[str, Any]:
    """Apply canonical TP state-layer gradient semantics before clipping/Adam."""
    size, rank = _world(group)
    placements: dict[str, dict[str, Any]] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, (ColumnParallelT2LoRALinear, RowParallelT2LoRALinear)):
            continue
        if module.local_lora_rank == 0:
            continue
        placements.update(adapter_placement_records(module_name or "_", module))
    if not placements:
        return {"collectives": 0, "sum_partial": 0, "mean_complete": 0, "local_shards": 0}
    from .ternary_tp_state import reduce_t2_tp_gradients_
    return reduce_t2_tp_gradients_(model, placements, rank=rank, world_size=size, group=group)


__all__ = [
    "ColumnParallelT2LoRALinear", "RowParallelT2LoRALinear", "adapter_placement_records",
    "gather_sequence_from_tp", "load_standard_peft_adapter_state", "local_peft_adapter_state",
    "reduce_from_tp", "reduce_scatter_sequence", "replicate_to_tp", "split_hadamard_blocks",
    "synchronize_t2_lora_gradients",
]
