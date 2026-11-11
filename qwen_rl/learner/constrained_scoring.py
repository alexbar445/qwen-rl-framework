"""Bounded, exact grammar-conditioned head scoring for single-device Learners.

The backbone sees the original full history. Only generated targets reach the
head; one small head block is checkpointed at a time. Grammar masks are replayed
from original sampled IDs, not from re-tokenized text or behavior normalizers.
"""
from __future__ import annotations

from typing import Any
from collections.abc import Iterator

import torch
from torch.utils.checkpoint import checkpoint

from ..constrained_policy import GrammarPolicy


def _allowed_tokens(bits: torch.Tensor, vocab_size: int, public_token_count: int,
                    device: torch.device) -> torch.Tensor:
    """Expand the compiler's signed int32 words without widening the dense gather."""
    ids = torch.arange(vocab_size, device=device, dtype=torch.int32)
    words = bits.to(device=device, dtype=torch.int32)
    allowed = ((words[:, ids//32] >> (ids%32)) & 1).bool()
    allowed &= ids < public_token_count
    return allowed


def _row_for(ids: list[int], rows: list[dict]) -> dict:
    candidates = [row for row in rows if list(row['input_ids']) == ids]
    if not candidates:
        raise ValueError('constrained microbatch does not match an original trajectory')
    identity = lambda r: (r.get('constraint_segments'), list(r['policy_mask']))
    if any(identity(r) != identity(candidates[0]) for r in candidates[1:]):
        raise ValueError('identical token sequences have conflicting constraint provenance')
    return candidates[0]


def _masks_for_row(policy: GrammarPolicy, row: dict) -> Iterator[tuple[int, torch.Tensor]]:
    segments = row.get('constraint_segments')
    if not isinstance(segments, list) or not segments:
        raise ValueError('grammar-conditioned row requires its original generation segments')
    covered = set(); previous_stop = 0
    ids = list(row['input_ids']); policy_mask = list(row['policy_mask'])
    for segment in segments:
        if set(segment) != {'start','stop','grammar_sha256','tokenizer_sha256','public_token_count','compiler_sha256'}:
            raise ValueError('invalid constraint segment metadata')
        start, stop = segment['start'], segment['stop']
        if (type(start) is not int or type(stop) is not int or not previous_stop <= start < stop <= len(ids)):
            raise ValueError('constraint segments overlap or exceed original tokens')
        if (segment['grammar_sha256'] != policy.identity.grammar_sha256
                or segment['tokenizer_sha256'] != policy.identity.tokenizer_sha256
                or segment['public_token_count'] != policy.identity.public_token_count
                or segment['compiler_sha256'] != policy.identity.compiler_sha256):
            raise ValueError('constraint grammar/compiler/tokenizer/vocabulary differs from the Learner')
        state = policy.session()
        for position in range(start, stop):
            if not policy_mask[position]:
                raise ValueError('constraint segment includes an environment token')
            state.next_bitmask()
            bits = state.bitmask[0].clone()
            covered.add(position)
            if not state.matcher.accept_token(ids[position]):
                raise ValueError('original sampled token violates recorded grammar')
            yield position, bits
        previous_stop = stop
    if covered != {i for i,value in enumerate(policy_mask) if value}:
        raise ValueError('constraint segments do not cover exactly the sampled policy tokens')


def score_constrained_rows(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                           labels: torch.Tensor, rows: list[dict], tokenizer: Any,
                           *, policy: GrammarPolicy | None = None, policy_cache: dict | None = None, chunk_rows: int = 32):
    """Return causal [batch, sequence] q_theta logprobs and entropies.

    Entry is explicit: DDP/TP wrappers are rejected instead of bypassing their
    reducers or silently scoring an unconstrained distribution.
    """
    if hasattr(model, 'module') or getattr(model, '_t2_tp_model', False):
        raise ValueError('constrained head scoring requires a single-device Learner')
    base = model.get_base_model() if callable(getattr(model, 'get_base_model', None)) else model
    if getattr(base, '_t2_tp_model', False):
        raise ValueError('constrained head scoring requires a single-device Learner')
    backbone = getattr(base, 'model', None)
    head = base.get_output_embeddings()
    if not isinstance(backbone, torch.nn.Module) or not isinstance(head, torch.nn.Module):
        raise ValueError('constrained Learner requires a text backbone and output head')
    if input_ids.ndim != 2 or labels.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise ValueError('constrained scoring requires aligned token, label and attention matrices')
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('head chunk_rows must be positive')
    from .ternary_tp_bridge import _trl_logit_parameters, _forward_text_backbone
    scale, softcap = _trl_logit_parameters(base.config)
    text_config = base.config.get_text_config() if callable(getattr(base.config,'get_text_config',None)) else base.config
    vocab_size = int(text_config.vocab_size)
    # Validation is host work. Copy each matrix once rather than synchronizing
    # CUDA for every sampled label and token in a long ARC trajectory.
    host_ids = input_ids.detach().cpu().tolist()
    host_attention = attention_mask.detach().cpu().tolist()
    host_labels = labels.detach().cpu().tolist()
    matched = []
    for b in range(input_ids.shape[0]):
        valid = [p for p,value in enumerate(host_attention[b]) if value]
        if not valid or valid != list(range(valid[0], valid[-1]+1)):
            raise ValueError('constrained attention rows must be contiguous')
        original = host_ids[b][valid[0]:valid[-1]+1]
        matched.append((valid[0], _row_for(original, rows)))
    credited = [row for _,row in matched if row.get('row_kind') != 'zero_credit']
    if not credited:
        if any(value != -100 for row in host_labels for value in row):
            raise ValueError('zero-credit rows must not contain head targets')
        hidden = _forward_text_backbone(base,input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True)
        zero = torch.nan_to_num(hidden[...,0].float()) * 0
        return zero,zero.clone()
    public = credited[0]['constraint_segments'][0]['public_token_count']
    if policy is None:
        first = credited[0]['constraint_segments'][0]
        cache_key = (first['grammar_sha256'], first['tokenizer_sha256'], vocab_size, public, first['compiler_sha256'])
        policy = policy_cache.get(cache_key) if policy_cache is not None else None
        if policy is None:
            policy = GrammarPolicy.for_arc(tokenizer,vocab_size=vocab_size,public_token_count=public)
            if (policy.identity.grammar_sha256 != first['grammar_sha256']
                    or policy.identity.tokenizer_sha256 != first['tokenizer_sha256']
                    or policy.identity.compiler_sha256 != first['compiler_sha256']):
                raise ValueError('recorded constraint identity does not match the loaded toolkit/tokenizer')
            if policy_cache is not None:
                policy_cache[cache_key] = policy
    if policy.identity.vocab_size != vocab_size:
        raise ValueError('constraint policy vocabulary differs from output head')
    selections = []; packed_masks = []
    # Intern repeated masks (free reasoning and repeated JSON states) on CPU.
    intern = {}
    for b,(padding,row) in enumerate(matched):
        if row.get('row_kind') == 'zero_credit':
            if any(value != -100 for value in host_labels[b]):
                raise ValueError('zero-credit rows must not contain head targets')
            continue
        masks = _masks_for_row(policy,row)
        for original, bits in masks:
            pos = padding + original
            if host_labels[b][pos] == -100:
                continue
            token = host_ids[b][pos]
            if pos == 0 or host_labels[b][pos] != token:
                raise ValueError('constrained target has no causal predecessor or changed ID')
            if (not 0 <= token < policy.identity.public_token_count
                    or not ((int(bits[token//32]) >> (token%32)) & 1)):
                raise ValueError('selected token outside replayed grammar support')
            key=bits.numpy().tobytes()
            packed_masks.append(intern.setdefault(key,bits))
            selections.append((b,pos))
    expected = {(b,p) for b,row in enumerate(host_labels) for p,value in enumerate(row) if value != -100}
    if set(selections) != expected:
        raise ValueError('constrained head labels differ from sampled target coverage')
    hidden = _forward_text_backbone(base,input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True)
    accumulator = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    result = torch.nan_to_num(hidden[...,0].to(accumulator))*0
    entropy_result = result.clone()
    if not selections:
        return result,entropy_result
    index_b=torch.tensor([b for b,p in selections],device=hidden.device)
    index_p=torch.tensor([p for b,p in selections],device=hidden.device)
    selected=hidden[index_b,index_p-1]
    target=input_ids[index_b,index_p]
    from ..models.ternary import FrozenTernaryLinear
    from ..models.ternary_weight_lease import DecodedTernaryWeightLease
    weight_lease = None
    if type(head) is FrozenTernaryLinear and selected.requires_grad:
        weight_lease = DecodedTernaryWeightLease(head)
        # All head branches join here. Retire the matrix before the gather and
        # backbone VJPs, including when head blocks checkpoint their forward.
        selected.register_hook(weight_lease.release)
    logps=[]; entropies=[]
    try:
        for begin in range(0,len(selections),chunk_rows):
            end=min(begin+chunk_rows,len(selections))
            bits=torch.stack(packed_masks[begin:end])
            def block(h, targets, *, bits=bits):
                logits=(head(h, decoded_weight_lease=weight_lease)
                        if weight_lease is not None else head(h)).to(accumulator)*scale
                if softcap is not None:
                    logits=softcap*torch.tanh(logits/softcap)
                allowed = _allowed_tokens(bits, vocab_size, policy.identity.public_token_count, logits.device)
                q=logits.masked_fill(~allowed,-torch.inf).log_softmax(-1)
                entropy=-(q.exp()*torch.where(allowed,q,0)).sum(-1)
                return q.gather(1,targets[:,None]).squeeze(1),entropy
            lp,en=checkpoint(block,selected[begin:end],target[begin:end],use_reentrant=False)
            logps.append(lp);entropies.append(en)
        result=result.index_put((index_b,index_p),torch.cat(logps))
        entropy_result=entropy_result.index_put((index_b,index_p),torch.cat(entropies))
        return result,entropy_result
    except BaseException:
        if weight_lease is not None:
            weight_lease.release()
        raise
