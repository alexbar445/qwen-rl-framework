"""CPU-only grammar policy identity and loss-preserving segment projection."""
from __future__ import annotations
import hashlib
import json
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

DISTRIBUTION = "grammar_conditioned_policy/v1"

class GrammarConstraint(BaseModel):
    """Engine-independent structural-tag descriptor, bound to its canonical bytes."""
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["structural-tag-policy/v1"]
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    descriptor: dict[str, Any]

    @model_validator(mode="after")
    def verify_descriptor(self):
        if self.descriptor.get("type") != "structural_tag" or "format" not in self.descriptor:
            raise ValueError("constraint requires a complete structural-tag descriptor")
        encoded = json.dumps(self.descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > 2 * 1024**2:
            raise ValueError("constraint descriptor exceeds 2 MiB")
        if hashlib.sha256(encoded).hexdigest() != self.sha256:
            raise ValueError("constraint descriptor fingerprint mismatch")
        return self



def rebase_constraint_segments(segments, original_ids, original_mask, projected_ids, projected_mask,
                               *, original_prefix: int, projected_prefix: int):
    """Move generation spans after replacing only the non-policy prompt prefix."""
    if (list(original_ids[original_prefix:]) != list(projected_ids[projected_prefix:])
            or list(original_mask[original_prefix:]) != list(projected_mask[projected_prefix:])):
        raise ValueError("constraint projection changed sampled body tokens or masks")
    covered=set(); result=[]; previous=original_prefix
    for segment in segments:
        start,stop=segment['start'],segment['stop']
        if type(start) is not int or type(stop) is not int or not previous<=start<stop<=len(original_ids):
            raise ValueError("constraint projection has overlapping or invalid spans")
        if any(not original_mask[p] for p in range(start,stop)):
            raise ValueError("constraint projection includes environment tokens")
        covered.update(range(start,stop));previous=stop
        result.append({**segment,'start':start+projected_prefix-original_prefix,
                       'stop':stop+projected_prefix-original_prefix})
    if covered!={i for i,value in enumerate(original_mask) if value}:
        raise ValueError("constraint projection does not cover the policy mask")
    return result
