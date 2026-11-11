"""Vocabulary-parallel scoring for packed, frozen T2 language-model heads.

Each rank owns a contiguous vocabulary-row shard.  The head never creates a
dense full-vocabulary weight or logits tensor: it dequantizes bounded row
chunks, computes online softmax statistics, and recomputes those chunks in the
custom backward.  Inputs, labels, masks, and returned scores are replicated
across the TP group.  Labels are aligned to hidden rows; causal shifting is a
caller responsibility.
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from .ternary import (
    FrozenTernaryEmbedding,
    FrozenTernaryLinear,
    _dequantize_t2_unchecked,
    hadamard_rows,
)

RECIPE = "t2_vocabulary_row_shards_streamed_score/v2"
_SUPPORTED_TP_SIZES = (2, 4)
DEFAULT_CHUNK_ROWS = 32768
DEFAULT_MAX_LOGITS_BYTES = 256 * 1024**2
_LOGIT_SCRATCH_BYTES_PER_ELEMENT = 16
_TORCH_ORACLE_CHUNK_ROWS = 2048


def _topology(group: Any = None) -> tuple[int, int]:
    if not dist.is_initialized():
        raise RuntimeError("vocabulary-parallel T2 requires an initialized process group")
    size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if size not in _SUPPORTED_TP_SIZES or not 0 <= rank < size:
        raise ValueError("vocabulary-parallel T2 supports TP groups of two or four ranks")
    return size, rank


def _row_shard(rows: int, size: int, rank: int) -> tuple[int, int]:
    base, remainder = divmod(rows, size)
    count = base + (1 if rank < remainder else 0)
    start = rank * base + min(rank, remainder)
    if count <= 0:
        raise ValueError("vocabulary must provide at least one row per TP rank")
    return start, count


def _raise_if_any(local_error: bool | torch.Tensor, *, device: torch.device,
                  group: Any, message: str) -> None:
    flag = torch.as_tensor(local_error, device=device, dtype=torch.int32).reshape(())
    if dist.get_world_size(group=group) > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    if bool(flag.item()):
        raise ValueError(message)


def _gather_counts(count: int, *, device: torch.device, group: Any,
                   size: int, message: str) -> None:
    local = torch.tensor([count], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(local) for _ in range(size)]
    dist.all_gather(gathered, local, group=group)
    if any(int(value.item()) != count for value in gathered):
        raise ValueError(message)


def _require_replicated_rows(rows: torch.Tensor, *, group: Any, size: int,
                             message: str) -> None:
    """Fail closed when TP ranks disagree on token positions or token IDs."""
    flat = rows.reshape(-1).contiguous()
    _gather_counts(flat.numel(), device=flat.device, group=group, size=size,
                   message=f"{message}: rank-local lengths differ")
    if flat.numel() == 0:
        return
    gathered = [torch.empty_like(flat) for _ in range(size)]
    dist.all_gather(gathered, flat, group=group)
    if any(not torch.equal(value, gathered[0]) for value in gathered[1:]):
        raise ValueError(message)


def _accumulation_dtype(compute_dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if compute_dtype == torch.float64 else torch.float32


def _transform_logits(raw_logits: torch.Tensor, *, temperature: float,
                      logit_scale: float, softcap: float | None) -> torch.Tensor:
    logits = raw_logits * logit_scale
    if softcap is not None:
        logits = softcap * (2.0 * torch.sigmoid(2.0 * logits / softcap) - 1.0)
    return logits / temperature


def _logit_transform_gradient(raw_logits: torch.Tensor, *, temperature: float,
                              logit_scale: float, softcap: float | None) -> torch.Tensor:
    derivative = logit_scale / temperature
    if softcap is None:
        return torch.full_like(raw_logits, derivative)
    scaled = raw_logits * logit_scale
    tanh_value = 2.0 * torch.sigmoid(2.0 * scaled / softcap) - 1.0
    return (1.0 - tanh_value.square()) * derivative


def _position_tile_rows(token_count: int, hidden_size: int, chunk_rows: int,
                         compute_dtype: torch.dtype, accumulation_dtype: torch.dtype,
                         max_logits_bytes: int) -> int:
    # Bound the logits/probability scratch plus the dHidden tile.  The estimate
    # intentionally excludes the dequantized weight chunk and saved activations.
    bytes_per_token = (
        chunk_rows * _LOGIT_SCRATCH_BYTES_PER_ELEMENT
        + hidden_size * (
            torch.empty((), dtype=compute_dtype).element_size()
            + torch.empty((), dtype=accumulation_dtype).element_size()
        )
    )
    return max(1, min(token_count, max_logits_bytes // bytes_per_token))


class _TRLLogProbKernels(NamedTuple):
    triton: Any
    token_chunk_size: int
    block_size: int
    addmm_fp32: Any
    forward: Any
    backward: Any


def _trl_logprob_kernels() -> _TRLLogProbKernels:
    """Use named bindings to the pinned upstream CUDA kernels, without fallback."""
    import triton
    from trl.kernels.chunked_logprob import (
        _BLOCK_SIZE,
        TOKEN_CHUNK_SIZE,
        _addmm_fp32,
        _backward_kernel,
        _forward_kernel,
    )
    return _TRLLogProbKernels(
        triton=triton, token_chunk_size=TOKEN_CHUNK_SIZE, block_size=_BLOCK_SIZE,
        addmm_fp32=_addmm_fp32, forward=_forward_kernel, backward=_backward_kernel,
    )


class _VocabularyParallelT2Score(Function):
    """Streaming selected-token log-probability and entropy with exact TP VJP."""

    @staticmethod
    def forward(
        ctx,
        rotated_hidden: torch.Tensor,
        labels: torch.Tensor,
        packed_codes: torch.Tensor,
        scale_words: torch.Tensor,
        bias: torch.Tensor,
        compute_dtype: torch.dtype,
        accumulation_dtype: torch.dtype,
        chunk_rows: int,
        max_logits_bytes: int,
        temperature: float,
        logit_scale: float,
        softcap: float | None,
        group: Any,
        vocab_start: int,
        vocab_size: int,
        has_bias: bool,
        world_size: int,
        rank: int,
    ):
        ctx.set_materialize_grads(False)
        if rotated_hidden.ndim != 2 or labels.ndim != 1:
            raise ValueError("internal T2 scorer expects [selected_tokens, hidden] and [selected_tokens]")
        if rotated_hidden.shape[0] != labels.numel() or labels.dtype != torch.int64:
            raise ValueError("internal T2 scorer inputs are misaligned")

        token_count, hidden_size = rotated_hidden.shape
        local_rows = packed_codes.shape[0]
        position_rows = _position_tile_rows(
            token_count, hidden_size, chunk_rows, compute_dtype, accumulation_dtype,
            max_logits_bytes,
        )
        local_max = torch.full((token_count,), -torch.inf, dtype=accumulation_dtype,
                               device=rotated_hidden.device)
        local_sum = torch.zeros_like(local_max)
        local_weighted_sum = torch.zeros_like(local_max)
        local_target = torch.zeros_like(local_max)
        bad_logits = torch.zeros((), dtype=torch.int32, device=rotated_hidden.device)

        # Decode each packed vocab chunk once, then stream all selected target
        # positions through it in bounded tiles.
        for begin in range(0, local_rows, chunk_rows):
            end = min(begin + chunk_rows, local_rows)
            rows = end - begin
            weight = _dequantize_t2_unchecked(
                packed_codes[begin:end], scale_words[begin:end], rows, hidden_size,
                compute_dtype,
            )
            local_bias = bias[begin:end].to(compute_dtype) if has_bias else None
            for token_begin in range(0, token_count, position_rows):
                token_end = min(token_begin + position_rows, token_count)
                tile_hidden = rotated_hidden[token_begin:token_end]
                tile_labels = labels[token_begin:token_end]
                tile_max = local_max[token_begin:token_end]
                tile_sum = local_sum[token_begin:token_end]
                tile_weighted_sum = local_weighted_sum[token_begin:token_end]
                tile_target = local_target[token_begin:token_end]
                raw_logits = F.linear(tile_hidden.to(compute_dtype), weight, local_bias)
                raw_logits = raw_logits.to(accumulation_dtype)
                logits = _transform_logits(
                    raw_logits, temperature=temperature, logit_scale=logit_scale,
                    softcap=softcap,
                )
                finite = torch.isfinite(logits).all()
                bad_logits = torch.maximum(bad_logits, (~finite).to(torch.int32))

                chunk_max = logits.max(dim=1).values
                next_max = torch.maximum(tile_max, chunk_max)
                old_scale = torch.exp(tile_max - next_max)
                old_scale = torch.where(torch.isfinite(tile_max), old_scale, 0.0)
                exp_logits = torch.exp(logits - next_max.unsqueeze(1))
                local_sum[token_begin:token_end] = tile_sum * old_scale + exp_logits.sum(dim=1)
                local_weighted_sum[token_begin:token_end] = (
                    tile_weighted_sum * old_scale + (exp_logits * logits).sum(dim=1)
                )
                local_max[token_begin:token_end] = next_max

                target_cols = tile_labels - vocab_start - begin
                in_chunk = (target_cols >= 0) & (target_cols < rows)
                safe_cols = target_cols.clamp(min=0, max=rows - 1)
                selected = logits.gather(1, safe_cols.unsqueeze(1)).squeeze(1)
                local_target[token_begin:token_end] = torch.where(in_chunk, selected, tile_target)

        # Every rank reaches this guard before reductions, even if only one
        # local shard encountered an overflow.
        _raise_if_any(bad_logits, device=rotated_hidden.device, group=group,
                      message="nonfinite T2 vocabulary logits")

        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
        rescale = torch.exp(local_max - global_max)
        scaled = torch.stack((
            local_sum * rescale,
            local_weighted_sum * rescale,
            local_target,
        ), dim=1)
        dist.all_reduce(scaled, op=dist.ReduceOp.SUM, group=group)
        global_sum = scaled[:, 0]
        global_weighted_sum = scaled[:, 1]
        global_target = scaled[:, 2]
        log_normalizer = global_max + torch.log(global_sum)
        expected_logit = global_weighted_sum / global_sum
        log_probs = global_target - log_normalizer
        entropy = log_normalizer - expected_logit
        _raise_if_any(~(torch.isfinite(log_probs).all() & torch.isfinite(entropy).all()),
                      device=rotated_hidden.device, group=group,
                      message="nonfinite T2 vocabulary score")

        ctx.group = group
        ctx.compute_dtype = compute_dtype
        ctx.accumulation_dtype = accumulation_dtype
        ctx.chunk_rows = chunk_rows
        ctx.max_logits_bytes = max_logits_bytes
        ctx.temperature = temperature
        ctx.logit_scale = logit_scale
        ctx.softcap = softcap
        ctx.vocab_start = vocab_start
        ctx.vocab_size = vocab_size
        ctx.has_bias = has_bias
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.save_for_backward(rotated_hidden, labels, packed_codes, scale_words,
                              bias, log_normalizer, expected_logit)
        return log_probs, entropy

    @staticmethod
    def backward(ctx, grad_log_probs, grad_entropy):
        (rotated_hidden, labels, packed_codes, scale_words, bias,
         log_normalizer, expected_logit) = ctx.saved_tensors
        if ctx.world_size != dist.get_world_size(group=ctx.group) or ctx.rank != dist.get_rank(group=ctx.group):
            raise RuntimeError("vocabulary-parallel T2 topology changed before backward")
        has_logprob_gradient = grad_log_probs is not None
        has_entropy_gradient = grad_entropy is not None
        if has_logprob_gradient and not bool(torch.isfinite(grad_log_probs).all()) or has_entropy_gradient and not bool(torch.isfinite(grad_entropy).all()):
            bad_gradient = True
        else:
            bad_gradient = False
        _raise_if_any(bad_gradient, device=rotated_hidden.device, group=ctx.group,
                      message="nonfinite T2 vocabulary output gradient")

        token_count, hidden_size = rotated_hidden.shape
        position_rows = _position_tile_rows(
            token_count, hidden_size, ctx.chunk_rows, ctx.compute_dtype,
            ctx.accumulation_dtype, ctx.max_logits_bytes,
        )
        local_gradient = torch.empty(
            (token_count, hidden_size), dtype=rotated_hidden.dtype, device=rotated_hidden.device
        )
        for token_begin in range(0, token_count, position_rows):
            token_end = min(token_begin + position_rows, token_count)
            tile_hidden = rotated_hidden[token_begin:token_end]
            tile_labels = labels[token_begin:token_end]
            tile_log_normalizer = log_normalizer[token_begin:token_end]
            tile_expected_logit = expected_logit[token_begin:token_end]
            tile_gradient = torch.zeros(
                (token_end - token_begin, hidden_size), dtype=ctx.accumulation_dtype,
                device=rotated_hidden.device,
            )
            bad_logits = torch.zeros((), dtype=torch.int32, device=rotated_hidden.device)
            for begin in range(0, packed_codes.shape[0], ctx.chunk_rows):
                end = min(begin + ctx.chunk_rows, packed_codes.shape[0])
                rows = end - begin
                weight = _dequantize_t2_unchecked(
                    packed_codes[begin:end], scale_words[begin:end], rows, hidden_size,
                    ctx.compute_dtype,
                )
                local_bias = bias[begin:end].to(ctx.compute_dtype) if ctx.has_bias else None
                weight_acc = weight.to(ctx.accumulation_dtype)
                raw_logits = F.linear(tile_hidden.to(ctx.compute_dtype), weight, local_bias)
                raw_logits = raw_logits.to(ctx.accumulation_dtype)
                logits = _transform_logits(
                    raw_logits, temperature=ctx.temperature, logit_scale=ctx.logit_scale,
                    softcap=ctx.softcap,
                )
                bad_logits = torch.maximum(
                    bad_logits, (~torch.isfinite(logits).all()).to(torch.int32)
                )
                probabilities = torch.exp(logits - tile_log_normalizer.unsqueeze(1))
                coefficients = torch.zeros_like(logits)
                if has_logprob_gradient:
                    tile_grad_log_probs = grad_log_probs[token_begin:token_end].to(
                        ctx.accumulation_dtype
                    )
                    coefficients.add_(
                        -tile_grad_log_probs.unsqueeze(1) * probabilities
                    )
                    target_cols = tile_labels - ctx.vocab_start - begin
                    in_chunk = (target_cols >= 0) & (target_cols < rows)
                    safe_cols = target_cols.clamp(min=0, max=rows - 1)
                    target_add = torch.where(in_chunk, tile_grad_log_probs, 0)
                    coefficients.scatter_add_(1, safe_cols.unsqueeze(1), target_add.unsqueeze(1))
                if has_entropy_gradient:
                    tile_grad_entropy = grad_entropy[token_begin:token_end].to(
                        ctx.accumulation_dtype
                    )
                    coefficients.add_(
                        tile_grad_entropy.unsqueeze(1) * probabilities
                        * (tile_expected_logit.unsqueeze(1) - logits)
                    )
                coefficients.mul_(_logit_transform_gradient(
                    raw_logits, temperature=ctx.temperature, logit_scale=ctx.logit_scale,
                    softcap=ctx.softcap,
                ))
                tile_gradient.add_(coefficients.matmul(weight_acc))

            bad_gradient = torch.maximum(
                bad_logits, (~torch.isfinite(tile_gradient).all()).to(torch.int32)
            )
            _raise_if_any(bad_gradient, device=rotated_hidden.device, group=ctx.group,
                          message="nonfinite T2 vocabulary hidden gradient")
            # TRL accumulates vocabulary-chunk GEMMs in FP32. Sum each bounded
            # dHidden tile across disjoint vocab shards before casting once.
            dist.all_reduce(tile_gradient, op=dist.ReduceOp.SUM, group=ctx.group)
            _raise_if_any(~torch.isfinite(tile_gradient).all(), device=rotated_hidden.device,
                          group=ctx.group, message="nonfinite reduced T2 vocabulary hidden gradient")
            local_gradient[token_begin:token_end].copy_(tile_gradient.to(rotated_hidden.dtype))
        return (local_gradient.to(rotated_hidden.dtype), None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None)


class _TRLTiledVocabularyParallelT2Score(Function):
    """Use the pinned TRL CUDA MM/Triton kernels on each local vocab shard."""

    @staticmethod
    def forward(
        ctx,
        rotated_hidden: torch.Tensor,
        labels: torch.Tensor,
        packed_codes: torch.Tensor,
        scale_words: torch.Tensor,
        bias: torch.Tensor,
        chunk_rows: int,
        max_logits_bytes: int,
        temperature: float,
        logit_scale: float,
        softcap: float | None,
        group: Any,
        vocab_start: int,
        vocab_size: int,
        has_bias: bool,
        world_size: int,
        rank: int,
    ):
        ctx.set_materialize_grads(False)
        if not rotated_hidden.is_cuda:
            raise RuntimeError("TRL tiled T2 scoring requires CUDA; select scoring_backend='torch' for CPU oracle")
        if rotated_hidden.ndim != 2 or labels.ndim != 1 or labels.dtype != torch.int64:
            raise ValueError("TRL tiled T2 scorer expects aligned [tokens, hidden] and int64 labels")
        if rotated_hidden.shape[0] != labels.numel():
            raise ValueError("TRL tiled T2 scorer inputs are misaligned")
        kernels = _trl_logprob_kernels()
        token_chunk_size, block_size = kernels.token_chunk_size, kernels.block_size
        forward_kernel = kernels.forward
        device = rotated_hidden.device
        compute_dtype = (
            torch.get_autocast_dtype(device.type)
            if torch.is_autocast_enabled(device.type)
            else rotated_hidden.dtype
        )
        token_count, hidden_size = rotated_hidden.shape
        local_rows = packed_codes.shape[0]
        projection_rows = min(chunk_rows, local_rows)
        tile_bytes = min(token_count, token_chunk_size) * projection_rows * torch.empty(
            (), dtype=compute_dtype
        ).element_size()
        if tile_bytes > max_logits_bytes:
            raise ValueError(
                f"TRL BF16 MM tile needs {tile_bytes} bytes, over the configured "
                f"{max_logits_bytes}-byte output-tile budget"
            )

        running_max = torch.full((token_count,), -torch.inf, device=device, dtype=torch.float32)
        sum_exp = torch.zeros((token_count,), device=device, dtype=torch.float32)
        x_sum_exp = torch.zeros_like(sum_exp)
        sq_sum_exp = torch.zeros_like(sum_exp)
        z_sum = torch.zeros_like(sum_exp)
        max_before_target = torch.full_like(sum_exp, -torch.inf)
        target_logit = torch.zeros_like(sum_exp)
        mm_buffer = torch.empty(
            (min(token_count, token_chunk_size), projection_rows),
            device=device,
            dtype=compute_dtype,
        )
        kernel_args = {
            "logit_scale": logit_scale,
            "softcap": softcap or 1.0,
            "inv_t": 1.0 / temperature,
            "HAS_SOFTCAP": softcap is not None,
        }
        output_flags = {
            "HAS_ENTROPY": True,
            "HAS_LOG_SUM_SQ_PROBS": False,
            "HAS_MEAN_LOGITS": False,
            "HAS_IS_TOP1": False,
        }
        for token_begin in range(0, token_count, token_chunk_size):
            token_end = min(token_begin + token_chunk_size, token_count)
            count = token_end - token_begin
            hidden_tile = rotated_hidden[token_begin:token_end].to(compute_dtype)
            for local_begin in range(0, local_rows, chunk_rows):
                local_end = min(local_begin + chunk_rows, local_rows)
                rows = local_end - local_begin
                weight = _dequantize_t2_unchecked(
                    packed_codes[local_begin:local_end],
                    scale_words[local_begin:local_end],
                    rows,
                    hidden_size,
                    compute_dtype,
                )
                local_bias = bias[local_begin:local_end].to(compute_dtype) if has_bias else None
                tile = mm_buffer[:count, :rows]
                # Match pinned TRL exactly: write the projection into a BF16
                # (or active autocast dtype) tile before Triton's FP32 stats.
                torch.mm(hidden_tile, weight.t(), out=tile)
                if local_bias is not None:
                    tile.add_(local_bias)
                vocab_chunk_start = vocab_start + local_begin
                forward_kernel[(count,)](
                    tile,
                    tile.stride(0),
                    labels[token_begin:token_end],
                    running_max[token_begin:token_end],
                    sum_exp[token_begin:token_end],
                    x_sum_exp[token_begin:token_end],
                    sq_sum_exp[token_begin:token_end],
                    z_sum[token_begin:token_end],
                    max_before_target[token_begin:token_end],
                    target_logit[token_begin:token_end],
                    vocab_chunk_start,
                    rows,
                    BLOCK_SIZE=block_size,
                    **output_flags,
                    **kernel_args,
                )

        local_bad = not bool(
            torch.isfinite(running_max).all()
            & torch.isfinite(sum_exp).all()
            & torch.isfinite(x_sum_exp).all()
            & torch.isfinite(target_logit).all()
        )
        _raise_if_any(local_bad, device=device, group=group,
                      message="nonfinite local TRL T2 vocabulary statistics")
        global_max = running_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
        rescale = torch.exp(running_max - global_max)
        statistics = torch.stack((sum_exp * rescale, x_sum_exp * rescale, target_logit), dim=1)
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM, group=group)
        global_sum_exp = statistics[:, 0]
        global_x_sum_exp = statistics[:, 1]
        global_target_logit = statistics[:, 2]
        log_normalizer = global_max + torch.log(global_sum_exp)
        entropy = log_normalizer - global_x_sum_exp / global_sum_exp
        log_probs = global_target_logit - log_normalizer
        _raise_if_any(~(torch.isfinite(log_probs).all() & torch.isfinite(entropy).all()),
                      device=device, group=group,
                      message="nonfinite global TRL T2 vocabulary score")

        ctx.group = group
        ctx.compute_dtype = compute_dtype
        ctx.chunk_rows = chunk_rows
        ctx.temperature = temperature
        ctx.logit_scale = logit_scale
        ctx.softcap = softcap
        ctx.vocab_start = vocab_start
        ctx.vocab_size = vocab_size
        ctx.has_bias = has_bias
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.save_for_backward(rotated_hidden, labels, packed_codes, scale_words,
                              bias, log_normalizer, entropy)
        return log_probs, entropy

    @staticmethod
    def backward(ctx, grad_log_probs, grad_entropy):
        (rotated_hidden, labels, packed_codes, scale_words, bias,
         log_normalizer, entropy) = ctx.saved_tensors
        if ctx.world_size != dist.get_world_size(group=ctx.group) or ctx.rank != dist.get_rank(group=ctx.group):
            raise RuntimeError("TRL tiled T2 vocabulary topology changed before backward")
        kernels = _trl_logprob_kernels()
        triton, token_chunk_size, block_size = kernels.triton, kernels.token_chunk_size, kernels.block_size
        addmm_fp32, backward_kernel = kernels.addmm_fp32, kernels.backward
        if grad_log_probs is not None and not bool(torch.isfinite(grad_log_probs).all()) or grad_entropy is not None and not bool(torch.isfinite(grad_entropy).all()):
            bad_gradient = True
        else:
            bad_gradient = False
        _raise_if_any(bad_gradient, device=rotated_hidden.device, group=ctx.group,
                      message="nonfinite TRL T2 vocabulary output gradient")

        token_count, hidden_size = rotated_hidden.shape
        local_rows = packed_codes.shape[0]
        projection_rows = min(ctx.chunk_rows, local_rows)
        mm_buffer = torch.empty(
            (min(token_count, token_chunk_size), projection_rows),
            device=rotated_hidden.device,
            dtype=ctx.compute_dtype,
        )
        grad_log_probs = grad_log_probs.float().contiguous() if grad_log_probs is not None else None
        grad_entropy = grad_entropy.float().contiguous() if grad_entropy is not None else None
        grad_hidden = torch.zeros(
            rotated_hidden.shape, device=rotated_hidden.device, dtype=torch.float32
        )
        kernel_args = {
            "logit_scale": ctx.logit_scale,
            "softcap": ctx.softcap or 1.0,
            "inv_t": 1.0 / ctx.temperature,
            "HAS_SOFTCAP": ctx.softcap is not None,
        }
        bad_logits = False
        for token_begin in range(0, token_count, token_chunk_size):
            token_end = min(token_begin + token_chunk_size, token_count)
            count = token_end - token_begin
            hidden_tile = rotated_hidden[token_begin:token_end].to(ctx.compute_dtype)
            for local_begin in range(0, local_rows, ctx.chunk_rows):
                local_end = min(local_begin + ctx.chunk_rows, local_rows)
                rows = local_end - local_begin
                weight = _dequantize_t2_unchecked(
                    packed_codes[local_begin:local_end],
                    scale_words[local_begin:local_end],
                    rows,
                    hidden_size,
                    ctx.compute_dtype,
                )
                local_bias = bias[local_begin:local_end].to(ctx.compute_dtype) if ctx.has_bias else None
                tile = mm_buffer[:count, :rows]
                torch.mm(hidden_tile, weight.t(), out=tile)
                if local_bias is not None:
                    tile.add_(local_bias)
                vocab_chunk_start = ctx.vocab_start + local_begin
                backward_kernel[(count, triton.cdiv(rows, block_size))](
                    tile,
                    tile.stride(0),
                    labels[token_begin:token_end],
                    log_normalizer[token_begin:token_end],
                    entropy[token_begin:token_end],
                    grad_log_probs[token_begin:token_end] if grad_log_probs is not None else log_normalizer,
                    grad_entropy[token_begin:token_end] if grad_entropy is not None else log_normalizer,
                    vocab_chunk_start,
                    rows,
                    HAS_LOGPROB_GRAD=grad_log_probs is not None, HAS_ENTROPY_GRAD=grad_entropy is not None, BLOCK_SIZE=block_size,
                    **kernel_args,
                )
                bad_logits = bad_logits or not bool(torch.isfinite(tile).all())
                # TRL's kernel writes BF16 logits gradients into `tile`, then
                # accumulates every vocabulary GEMM into an FP32 dHidden buffer.
                addmm_fp32(grad_hidden[token_begin:token_end], tile, weight)

        _raise_if_any(bad_logits or not bool(torch.isfinite(grad_hidden).all()),
                      device=rotated_hidden.device, group=ctx.group,
                      message="nonfinite TRL T2 vocabulary hidden gradient")
        dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=ctx.group)
        _raise_if_any(~torch.isfinite(grad_hidden).all(), device=rotated_hidden.device,
                      group=ctx.group, message="nonfinite reduced TRL T2 vocabulary hidden gradient")
        result = grad_hidden.to(rotated_hidden.dtype)
        return (result, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None)


class VocabularyParallelTernaryHead(nn.Module):
    """Frozen T2 head with local packed vocabulary rows and streamed scoring.

    `score` selects the rows where ``target_mask`` is true before the full-width
    input Hadamard.  It returns float32 log-probabilities and entropies (float64
    only for a float64 T2 compute oracle), shaped like ``labels``.  Masked rows
    are zero and receive zero hidden gradient.  The input source is copied into
    local packed row buffers and is not retained by this module; callers should
    release their full-vocabulary source module after construction.
    """

    def __init__(self, source: FrozenTernaryLinear, *, group: Any = None,
                 chunk_rows: int = DEFAULT_CHUNK_ROWS,
                 max_logits_bytes: int = DEFAULT_MAX_LOGITS_BYTES,
                 scoring_backend: str = "trl"):
        super().__init__()
        if type(source) is not FrozenTernaryLinear:
            raise TypeError("vocabulary TP requires an unsharded FrozenTernaryLinear source")
        if any(parameter.requires_grad for parameter in source.parameters()):
            raise ValueError("vocabulary T2 base weights must be frozen")
        if type(chunk_rows) is not int or chunk_rows <= 0:
            raise ValueError("T2 vocabulary row chunk size must be a positive integer")
        if type(max_logits_bytes) is not int or max_logits_bytes <= 0:
            raise ValueError("T2 maximum logits working-set budget must be a positive byte count")
        if scoring_backend not in ("trl", "torch"):
            raise ValueError("T2 scoring_backend must explicitly be 'trl' or the CPU 'torch' oracle")
        size, rank = _topology(group)
        start, count = _row_shard(source.out_features, size, rank)
        self.in_features = int(source.in_features)
        self.out_features = int(source.out_features)
        self.vocab_size = int(source.out_features)
        self.local_vocab_start = start
        self.local_vocab_size = count
        self.tp_world_size = size
        self.tp_rank = rank
        self.chunk_rows = chunk_rows
        self.max_logits_bytes = max_logits_bytes
        self.scoring_backend = scoring_backend
        if scoring_backend == "trl":
            minimum_budget = min(chunk_rows, count) * torch.empty(
                (), dtype=source.compute_dtype
            ).element_size()
        else:
            minimum_budget = (
                chunk_rows * _LOGIT_SCRATCH_BYTES_PER_ELEMENT
                + source.in_features * (
                    torch.empty((), dtype=source.compute_dtype).element_size()
                    + torch.empty((), dtype=_accumulation_dtype(source.compute_dtype)).element_size()
                )
            )
        if max_logits_bytes < minimum_budget:
            raise ValueError(
                f"T2 maximum logits budget must be at least {minimum_budget} bytes for one token tile"
            )
        self.compute_dtype = source.compute_dtype
        self.output_dtype = source.output_dtype
        object.__setattr__(self, "tp_group", group)
        self.register_buffer("packed_codes", source.packed_codes[start:start + count].contiguous().clone())
        self.register_buffer("scale_words", source.scale_words[start:start + count].contiguous().clone())
        self.register_buffer("input_hadamard_signs", source.input_hadamard_signs.contiguous().clone())
        local_bias = None if source.bias is None else source.bias[start:start + count].contiguous().clone()
        self.register_buffer("bias", local_bias)

    @classmethod
    def from_source(cls, source: FrozenTernaryLinear, *, group: Any = None,
                    chunk_rows: int = DEFAULT_CHUNK_ROWS,
                    max_logits_bytes: int = DEFAULT_MAX_LOGITS_BYTES,
                    scoring_backend: str = "trl") -> VocabularyParallelTernaryHead:
        return cls(source, group=group, chunk_rows=chunk_rows,
                   max_logits_bytes=max_logits_bytes, scoring_backend=scoring_backend)

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "recipe": RECIPE,
            "rank": self.tp_rank,
            "world_size": self.tp_world_size,
            "vocab_size": self.vocab_size,
            "vocab_start": self.local_vocab_start,
            "local_vocab_size": self.local_vocab_size,
            "hidden_size": self.in_features,
            "chunk_rows": self.chunk_rows,
            "max_logits_bytes": self.max_logits_bytes,
            "scoring_backend": self.scoring_backend,
            "compute_dtype": str(self.compute_dtype),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state != self.get_extra_state():
            raise ValueError("T2 vocabulary shard state belongs to a different rank, layout or topology")

    def rotate_inputs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from .ternary_shared_rotation import rotate_base
        return rotate_base(self, hidden_states)

    def dequantize_weight(self, *args, **kwargs):
        raise RuntimeError("vocabulary TP retains only local packed rows; dense head materialization is unavailable")

    @property
    def weight(self) -> torch.Tensor:
        raise RuntimeError("vocabulary TP has no dense lm_head.weight tensor")

    def score(self, hidden_states: torch.Tensor, labels: torch.Tensor,
              target_mask: torch.Tensor | None = None, *, temperature: float = 1.0,
              logit_scale: float = 1.0, softcap: float | None = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_floating_point():
            raise TypeError("T2 vocabulary hidden states must be a floating point tensor")
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.in_features:
            raise ValueError(f"T2 vocabulary hidden states must end in width {self.in_features}")
        if not isinstance(labels, torch.Tensor) or labels.dtype not in (torch.int32, torch.int64):
            raise TypeError("T2 vocabulary labels must be int32 or int64")
        if labels.shape != hidden_states.shape[:-1] or labels.device != hidden_states.device:
            raise ValueError("T2 vocabulary labels must match hidden leading dimensions and device")
        if hidden_states.device != self.packed_codes.device:
            raise ValueError("T2 vocabulary hidden states and local packed rows must share a device")
        if type(temperature) not in (int, float) or not math.isfinite(float(temperature)):
            raise ValueError("T2 vocabulary temperature must be finite")
        if temperature <= 0:
            raise ValueError("T2 vocabulary temperature must be greater than zero")
        if type(logit_scale) not in (int, float) or not math.isfinite(float(logit_scale)):
            raise ValueError("T2 vocabulary logit_scale must be finite")
        if softcap is not None and (
            type(softcap) not in (int, float) or not math.isfinite(float(softcap))
            or softcap <= 0
        ):
            raise ValueError("T2 vocabulary softcap must be finite and greater than zero")
        temperature = float(temperature)
        logit_scale = float(logit_scale)
        softcap = None if softcap is None else float(softcap)
        if target_mask is None:
            target_mask = labels != -100
        elif (not isinstance(target_mask, torch.Tensor) or target_mask.dtype != torch.bool
              or target_mask.shape != labels.shape or target_mask.device != labels.device):
            raise ValueError("T2 vocabulary target_mask must be a boolean tensor matching labels")

        size, rank = _topology(self.tp_group)
        if (size, rank) != (self.tp_world_size, self.tp_rank):
            raise RuntimeError("T2 vocabulary process-group topology changed")
        mask_flat = target_mask.reshape(-1).contiguous()
        active_indices = mask_flat.nonzero(as_tuple=False).reshape(-1)
        flat_labels = labels.reshape(-1)
        active_labels = flat_labels.index_select(0, active_indices).to(torch.int64).contiguous()
        bad_labels = bool(((active_labels < 0) | (active_labels >= self.vocab_size)).any())
        _raise_if_any(bad_labels, device=hidden_states.device, group=self.tp_group,
                      message="active T2 vocabulary labels must be within the global vocabulary")
        _gather_counts(active_indices.numel(), device=hidden_states.device,
                       group=self.tp_group, size=size,
                       message="TP ranks have different active target counts")
        if active_indices.numel():
            target_rows = torch.stack((active_indices.to(torch.int64), active_labels), dim=1)
            _require_replicated_rows(target_rows, group=self.tp_group, size=size,
                                     message="TP ranks disagree on active target positions or labels")

        flat_hidden = hidden_states.reshape(-1, self.in_features)
        selected_hidden = flat_hidden.index_select(0, active_indices)
        bad_hidden = not bool(torch.isfinite(selected_hidden).all())
        _raise_if_any(bad_hidden, device=hidden_states.device, group=self.tp_group,
                      message="nonfinite active T2 vocabulary hidden state")

        score_dtype = _accumulation_dtype(self.compute_dtype)
        if active_indices.numel() == 0:
            # Match TRL's all-masked microbatch contract: return a connected
            # differentiable zero so DAPO can safely skip the tokenless rows.
            empty = torch.nan_to_num(hidden_states[..., 0].to(score_dtype)) * 0.0
            return empty, empty.clone()
        rotated = self.rotate_inputs(selected_hidden)
        bias = self.bias if self.bias is not None else torch.empty(
            0, device=self.packed_codes.device, dtype=self.compute_dtype
        )
        if self.scoring_backend == "torch":
            if hidden_states.device.type != "cpu":
                raise RuntimeError("scoring_backend='torch' is a CPU oracle; CUDA requires scoring_backend='trl'")
            selected_log_probs, selected_entropy = _VocabularyParallelT2Score.apply(
                rotated,
                active_labels,
                self.packed_codes,
                self.scale_words,
                bias,
                self.compute_dtype,
                score_dtype,
                self.chunk_rows,
                self.max_logits_bytes,
                temperature,
                logit_scale,
                softcap,
                self.tp_group,
                self.local_vocab_start,
                self.vocab_size,
                self.bias is not None,
                self.tp_world_size,
                self.tp_rank,
            )
        else:
            if hidden_states.device.type != "cuda":
                raise RuntimeError("scoring_backend='trl' requires CUDA; there is no runtime Torch fallback")
            selected_log_probs, selected_entropy = _TRLTiledVocabularyParallelT2Score.apply(
                rotated,
                active_labels,
                self.packed_codes,
                self.scale_words,
                bias,
                self.chunk_rows,
                self.max_logits_bytes,
                temperature,
                logit_scale,
                softcap,
                self.tp_group,
                self.local_vocab_start,
                self.vocab_size,
                self.bias is not None,
                self.tp_world_size,
                self.tp_rank,
            )
        log_probs = torch.zeros(labels.numel(), dtype=score_dtype, device=labels.device)
        entropy = torch.zeros_like(log_probs)
        log_probs = log_probs.index_copy(0, active_indices, selected_log_probs)
        entropy = entropy.index_copy(0, active_indices, selected_entropy)
        return log_probs.reshape(labels.shape), entropy.reshape(labels.shape)

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor,
                target_mask: torch.Tensor | None = None, *, temperature: float = 1.0,
                logit_scale: float = 1.0, softcap: float | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.score(hidden_states, labels, target_mask, temperature=temperature,
                          logit_scale=logit_scale, softcap=softcap)


class VocabularyParallelTernaryEmbedding(nn.Module):
    """Optional packed vocabulary-row-sharded input embedding.

    Token IDs must be replicated across the TP group when the lookup runs.  It
    decodes only rows owned by this rank, restores the primal Hadamard basis,
    and sums local contributions so every rank receives the same embeddings.
    """

    def __init__(self, source: FrozenTernaryEmbedding, *, group: Any = None):
        super().__init__()
        if type(source) is not FrozenTernaryEmbedding:
            raise TypeError("vocabulary TP requires an unsharded FrozenTernaryEmbedding source")
        size, rank = _topology(group)
        start, count = _row_shard(source.num_embeddings, size, rank)
        self.num_embeddings = int(source.num_embeddings)
        self.embedding_dim = int(source.embedding_dim)
        self.local_vocab_start = start
        self.local_vocab_size = count
        self.tp_world_size = size
        self.tp_rank = rank
        self.padding_idx = source.padding_idx
        self.output_dtype = source.output_dtype
        object.__setattr__(self, "tp_group", group)
        self.register_buffer("packed_codes", source.packed_codes[start:start + count].contiguous().clone())
        self.register_buffer("scale_words", source.scale_words[start:start + count].contiguous().clone())
        self.register_buffer("hadamard_signs", source.hadamard_signs.contiguous().clone())

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "recipe": "t2_vocabulary_row_shards_embedding/v1",
            "rank": self.tp_rank,
            "world_size": self.tp_world_size,
            "vocab_size": self.num_embeddings,
            "vocab_start": self.local_vocab_start,
            "local_vocab_size": self.local_vocab_size,
            "embedding_dim": self.embedding_dim,
            "padding_idx": self.padding_idx,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state != self.get_extra_state():
            raise ValueError("T2 vocabulary embedding state belongs to a different rank, layout or topology")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("T2 vocabulary embedding IDs must use int32 or int64")
        if input_ids.device != self.packed_codes.device:
            raise ValueError("T2 vocabulary IDs and local packed rows must share a device")
        size, rank = _topology(self.tp_group)
        if (size, rank) != (self.tp_world_size, self.tp_rank):
            raise RuntimeError("T2 vocabulary embedding process-group topology changed")
        flat = input_ids.reshape(-1).to(torch.int64).contiguous()
        invalid = bool(((flat < 0) | (flat >= self.num_embeddings)).any())
        _raise_if_any(invalid, device=input_ids.device, group=self.tp_group,
                      message="T2 vocabulary embedding token ID is outside the global table")
        _require_replicated_rows(flat, group=self.tp_group, size=size,
                                 message="TP ranks disagree on replicated T2 embedding token IDs")
        output = torch.zeros((flat.numel(), self.embedding_dim), dtype=self.output_dtype,
                             device=input_ids.device)
        local = (flat >= self.local_vocab_start) & (
            flat < self.local_vocab_start + self.local_vocab_size
        )
        if self.padding_idx is not None:
            local &= flat != self.padding_idx
        positions = local.nonzero(as_tuple=False).reshape(-1)
        if positions.numel():
            local_rows = flat.index_select(0, positions) - self.local_vocab_start
            codes = self.packed_codes.index_select(0, local_rows)
            scales = self.scale_words.index_select(0, local_rows)
            rotated = _dequantize_t2_unchecked(
                codes, scales, positions.numel(), self.embedding_dim, torch.float32
            )
            primal = hadamard_rows(
                rotated, self.hadamard_signs, inverse=True, output_dtype=self.output_dtype
            )
            output.index_copy_(0, positions, primal)
        if flat.numel():
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return output.reshape(*input_ids.shape, self.embedding_dim)


__all__ = ["VocabularyParallelTernaryEmbedding", "VocabularyParallelTernaryHead"]
