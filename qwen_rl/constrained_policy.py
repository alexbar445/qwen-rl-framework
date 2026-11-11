"""Grammar-conditioned sampling and differentiable policy scoring.

This is the shared mathematical contract for constrained rollout and Learner
scoring. It never substitutes raw-policy probabilities for the masked policy.
Grammar state is request-local and reconstructed from original token IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch

DISTRIBUTION = 'grammar_conditioned_policy/v1'


@dataclass(frozen=True)
class PolicyIdentity:
    grammar_sha256: str
    tokenizer_sha256: str
    vocab_size: int
    public_token_count: int | None = None
    compiler_sha256: str | None = None

    def __post_init__(self):
        public = self.vocab_size if self.public_token_count is None else self.public_token_count
        if type(self.vocab_size) is not int or type(public) is not int or not 0 < public <= self.vocab_size:
            raise ValueError("invalid constrained-policy vocabulary domain")
        object.__setattr__(self, "public_token_count", public)


class GrammarPolicy:
    def __init__(self, compiled: Any, identity: PolicyIdentity):
        self.compiled = compiled
        self.identity = identity

    @classmethod
    def for_arc(cls, tokenizer: Any, *, vocab_size: int, public_token_count: int | None = None, thinking: bool = True):
        import xgrammar as xgr
        from .prompt import load_toolkit_runtime
        load_toolkit_runtime()
        from arc_toolkit.constrained_actions import constraint_receipt
        receipt = constraint_receipt(thinking=thinking)
        from .contracts.grammar_compiler import native_grammar_compiler_sha256
        compiler_id = native_grammar_compiler_sha256()
        backend = getattr(tokenizer, 'backend_tokenizer', None)
        if backend is None:
            raise ValueError('constrained ARC requires a serializable fast tokenizer')
        tokenizer_hash = hashlib.sha256(backend.to_str().encode()).hexdigest()
        hf_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        # Match qwen3_5::Frontend exactly: invalid/padded tokens and special
        # tokens are absent from the grammar lexicon, except </think>. EOS is
        # separately licensed by the compiler's stop-token contract.
        decoded = list(hf_info.decoded_vocab)
        for token_id, added in backend.get_added_tokens_decoder().items():
            if token_id < len(decoded) and added.special and str(added) != "</think>":
                decoded[token_id] = b""
        info = xgr.TokenizerInfo(decoded, vocab_type=xgr.VocabType.RAW,
            vocab_size=vocab_size, stop_token_ids=hf_info.stop_token_ids)
        compiler = xgr.GrammarCompiler(info)
        compiled = compiler.compile_structural_tag(receipt['descriptor'])
        return cls(compiled, PolicyIdentity(receipt['sha256'], tokenizer_hash, vocab_size, public_token_count, compiler_id))

    def session(self):
        return GrammarPolicySession(self)

    def score(self, logits: torch.Tensor, token_ids: list[int]):
        """Return exact q_theta logprobs/entropy with autograd through normalization.

        logits[i] predicts token_ids[i]. This interface accepts one generation
        segment; tool observations are never scored or passed to the matcher.
        It also accepts a truncated valid prefix, without inventing closure tokens.
        """
        if logits.ndim != 2 or logits.shape != (len(token_ids), self.identity.vocab_size):
            raise ValueError('score logits must align with original segment token IDs and vocabulary')
        state = self.session()
        logps, entropies = [], []
        for row, token in zip(logits, token_ids, strict=True):
            logp, entropy = state.score_next(row, token)
            logps.append(logp); entropies.append(entropy)
        if not logps:
            return logits.sum(-1), logits.sum(-1)
        return torch.stack(logps), torch.stack(entropies)


class GrammarPolicySession:
    def __init__(self, policy: GrammarPolicy):
        import xgrammar as xgr
        self.policy = policy
        self.matcher = xgr.GrammarMatcher(policy.compiled)
        self.bitmask = xgr.allocate_token_bitmask(1, policy.identity.vocab_size)
        self.position = 0

    def next_bitmask(self):
        if not self.matcher.fill_next_token_bitmask(self.bitmask):
            self.bitmask.fill_(-1)
        return self.bitmask

    def _log_distribution(self, logits: torch.Tensor):
        import xgrammar as xgr
        if logits.ndim != 1 or logits.numel() != self.policy.identity.vocab_size:
            raise ValueError('one vocabulary logit vector is required')
        if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
            raise ValueError('raw logits must be finite floating point values')
        self.next_bitmask()
        # Expand one row only. No dense episode-length token x vocabulary mask is
        # stored; replay derives the next mask from the immutable compiled grammar.
        ids = torch.arange(logits.numel(), device=logits.device, dtype=torch.int64)
        words = self.bitmask[0].to(device=logits.device, dtype=torch.int64)
        allowed = ((words[ids // 32] >> (ids % 32)) & 1).bool()
        allowed &= ids < self.policy.identity.public_token_count
        if not bool(allowed.any()):
            raise ValueError('grammar has no admissible next token')
        masked = logits.masked_fill(~allowed, -torch.inf)
        return masked.log_softmax(-1), allowed

    def score_next(self, logits: torch.Tensor, token: int):
        logps, allowed = self._log_distribution(logits)
        if type(token) is not int or not 0 <= token < logits.numel() or not bool(allowed[token]):
            raise ValueError(f'token at segment position {self.position} violates grammar')
        # Avoid 0 * -inf in the entropy and its gradient.
        finite_logps = torch.where(allowed, logps, 0)
        entropy = -(logps.exp() * finite_logps).sum()
        if not self.matcher.accept_token(token):
            raise RuntimeError('mask and matcher disagree')
        self.position += 1
        return logps[token], entropy

    def sample_next(self, logits: torch.Tensor, *, generator=None):
        logps, _ = self._log_distribution(logits)
        token = int(torch.multinomial(logps.exp(), 1, generator=generator).item())
        if not self.matcher.accept_token(token):
            raise RuntimeError('sampled token violates applied grammar mask')
        self.position += 1
        return token, float(logps[token].detach())
