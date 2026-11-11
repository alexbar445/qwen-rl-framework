"""Exact causal CE/KL with vocabulary-row streaming and recomputed backward.

Never save [tokens,vocabulary] logits/probabilities. Teacher hidden states are
offline captures in the original teacher basis. Student head rotation remains
owned by the shared packed-T2 module.
"""

from __future__ import annotations

import torch

from qwen_rl.models.ternary import FrozenTernaryLinear
from qwen_rl.models.ternary_recovery import ScaleTernaryLinear, decode, scale_vjp


class TensorRows:
    def __init__(self, weight):
        if weight.ndim != 2 or weight.requires_grad:
            raise ValueError("teacher/dense reference head must be a frozen matrix")
        self.weight = weight
        self.shape = tuple(weight.shape)

    def rows(self, begin, end, *, device, dtype):
        return self.weight[begin:end].to(device=device, dtype=dtype)


class SafetensorRows:
    def __init__(self, path, key):
        from safetensors import safe_open

        self.path, self.key = str(path), key
        with safe_open(self.path, framework="pt", device="cpu") as f:
            self.shape = tuple(f.get_slice(key).get_shape())
        if len(self.shape) != 2:
            raise ValueError("teacher head must be a matrix")

    def rows(self, begin, end, *, device, dtype):
        from safetensors import safe_open

        with safe_open(self.path, framework="pt", device="cpu") as f:
            value = f.get_slice(self.key)[begin:end]
        return value.to(device=device, dtype=dtype)


class StudentRows:
    def __init__(self, head):
        self.head = head
        self.base = head.base if isinstance(head, ScaleTernaryLinear) else head
        if not isinstance(self.base, FrozenTernaryLinear):
            raise TypeError("recovery requires the existing packed T2 output head")
        if self.base.bias is not None:
            raise ValueError("recovery T2 head must have no bias")
        self.shape = (self.base.out_features, self.base.in_features)

    @property
    def delta(self):
        return (
            self.head.log_scale_delta
            if isinstance(self.head, ScaleTernaryLinear)
            else None
        )

    def rows(self, begin, end, *, device, dtype):
        if isinstance(self.head, ScaleTernaryLinear):
            return self.head.rows(begin, end, dtype)
        return decode(
            self.base.packed_codes[begin:end], self.base.scale_words[begin:end], dtype
        )

    def rotate(self, hidden):
        return self.base.rotate_inputs(hidden)

    def scale_gradient(self, begin, end, dw):
        return scale_vjp(
            self.base.packed_codes[begin:end],
            self.base.scale_words[begin:end],
            self.delta[begin:end],
            dw,
        )


class _StreamingDistillation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, teacher_x, labels, delta, student, teacher, ce, kl, denominator, chunk
    ):
        if student.shape != teacher.shape or x.shape != teacher_x.shape:
            raise ValueError("teacher/student head or hidden geometry differs")
        count, _ = x.shape
        vocab = student.shape[0]
        lse_s = torch.full((count,), -torch.inf, device=x.device, dtype=torch.float32)
        lse_t = torch.full_like(lse_s, -torch.inf)
        weighted_difference = torch.zeros_like(lse_s)
        targets = torch.zeros_like(lse_s)
        for begin in range(0, vocab, chunk):
            end = min(begin + chunk, vocab)
            s = (x @ student.rows(begin, end, device=x.device, dtype=x.dtype).T).float()
            t = (
                teacher_x
                @ teacher.rows(begin, end, device=x.device, dtype=teacher_x.dtype).T
            ).float()
            lse_s = torch.logaddexp(lse_s, torch.logsumexp(s, -1))
            next_lse_t = torch.logaddexp(lse_t, torch.logsumexp(t, -1))
            # Online teacher-normalized E_t[teacher_logit-student_logit].
            # This avoids a second forward pass over the 248K vocabulary.
            weighted_difference = weighted_difference * (lse_t - next_lse_t).exp() + (
                (t - next_lse_t[:, None]).exp() * (t - s)
            ).sum(-1)
            lse_t = next_lse_t
            active = (labels >= begin) & (labels < end)
            targets[active] = s[active, labels[active] - begin]
        ce_sum = (lse_s - targets).double().sum()
        kl_sum = (weighted_difference + lse_s - lse_t).double().sum()
        result = (ce * ce_sum + kl * kl_sum) / denominator
        if not bool(torch.isfinite(result)):
            raise FloatingPointError("nonfinite recovery CE/KL")
        ctx.save_for_backward(x, teacher_x, labels, delta, lse_s, lse_t)
        ctx.student, ctx.teacher = student, teacher
        ctx.ce, ctx.kl, ctx.denominator, ctx.chunk = ce, kl, denominator, chunk
        ctx.mark_non_differentiable(ce_sum, kl_sum)
        return result, ce_sum, kl_sum

    @staticmethod
    def backward(ctx, grad, _ce_grad, _kl_grad):
        x, tx, labels, delta, lse_s, lse_t = ctx.saved_tensors
        dx = torch.zeros_like(x, dtype=torch.float32)
        ds = torch.zeros_like(delta) if delta.numel() else None
        for begin in range(0, ctx.student.shape[0], ctx.chunk):
            end = min(begin + ctx.chunk, ctx.student.shape[0])
            w = ctx.student.rows(begin, end, device=x.device, dtype=x.dtype)
            p = ((x @ w.T).float() - lse_s[:, None]).exp()
            dl = (ctx.ce + ctx.kl) * p
            if ctx.kl:
                tw = ctx.teacher.rows(begin, end, device=x.device, dtype=tx.dtype)
                pt = ((tx @ tw.T).float() - lse_t[:, None]).exp()
                dl -= ctx.kl * pt
            active = (labels >= begin) & (labels < end)
            dl[active, labels[active] - begin] -= ctx.ce
            dl *= grad.float() / ctx.denominator
            dx += dl.to(x.dtype).matmul(w).float()
            if ds is not None:
                dw = dl.to(x.dtype).T.matmul(x)
                ds[begin:end] = ctx.student.scale_gradient(begin, end, dw)
        return dx.to(x.dtype), None, None, ds, None, None, None, None, None, None


def causal_distillation(
    hidden,
    teacher_hidden,
    input_ids,
    attention_mask,
    loss_mask,
    student,
    teacher,
    *,
    ce_weight=1.0,
    kl_weight=1.0,
    chunk_rows=256,
    denominator=None,
):
    if (
        hidden.ndim != 3
        or hidden.shape != teacher_hidden.shape
        or input_ids.shape != hidden.shape[:2]
    ):
        raise ValueError(
            "recovery tensors require aligned [B,T,H] hidden states and [B,T] IDs"
        )
    if (
        input_ids.dtype != torch.int64
        or attention_mask.dtype != torch.bool
        or loss_mask.dtype != torch.bool
    ):
        raise TypeError("recovery IDs must be int64 and masks boolean")
    if attention_mask.shape != input_ids.shape or loss_mask.shape != input_ids.shape:
        raise ValueError("recovery mask geometry differs")
    active = attention_mask[:, :-1] & attention_mask[:, 1:] & loss_mask[:, 1:]
    targets = input_ids[:, 1:][active]
    count = targets.numel()
    if not count:
        raise ValueError("recovery batch has no causal targets")
    if bool(((targets < 0) | (targets >= student.shape[0])).any()):
        raise ValueError("recovery target outside full vocabulary")
    if denominator is None:
        denominator = count
    if (
        denominator < count
        or chunk_rows < 1
        or ce_weight < 0
        or kl_weight < 0
        or ce_weight + kl_weight <= 0
    ):
        raise ValueError("invalid recovery normalization, chunk size or loss weights")
    x = hidden[:, :-1][active]
    if isinstance(student, StudentRows):
        x = student.rotate(x)
    tx = teacher_hidden[:, :-1][active].detach().to(device=x.device, dtype=x.dtype)
    delta = getattr(student, "delta", None)
    delta = x.new_empty(0) if delta is None else delta
    value, ce, kl = _StreamingDistillation.apply(
        x,
        tx,
        targets,
        delta,
        student,
        teacher,
        ce_weight,
        kl_weight,
        denominator,
        chunk_rows,
    )
    return value, {"scored_tokens": count, "ce_total": ce.item(), "kl_total": kl.item()}
