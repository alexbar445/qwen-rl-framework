"""CPU-only request and artifact contracts for the standalone Worker HTTP ABI."""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

MAX_CONTEXT_TOKENS = 131_072
# Absolute wire ceiling for one generation request. The *effective* bound is per deployment:
# `worker_inference.max_generated_tokens` is copied verbatim into each Worker's /config/worker.json,
# so changing the number is a config edit plus a restart - never a Worker image rebuild. The
# Learner's rollout.turn_max_tokens must stay <= that bound and is checked when the config loads.
MAX_GENERATED_TOKENS = 65_536
ADAPTER_FILES = frozenset({"adapter_config.json", "adapter_model.safetensors"})
NATIVE_ADAPTER_FILES = frozenset({"policy.lora.ninfer", "policy.lora.ninfer.json"})
MAX_REQUEST_SEED = (1 << 32) - 1
SAMPLING_CONSTANTS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
}
NEUTRALIZED_SAMPLING_FIELDS = (
    "typical_p",
    "min_p",
    "epsilon_cutoff",
    "eta_cutoff",
    "no_repeat_ngram_size",
    "sequence_bias",
    "suppress_tokens",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


from .constraints import GrammarConstraint


class SamplingParams(_StrictModel):
    """The shared Worker raw-policy sampling fields accepted on the wire."""

    max_tokens: StrictInt = Field(ge=1, le=MAX_GENERATED_TOKENS)
    logprobs: Literal[True]
    temperature: StrictInt | StrictFloat = 1
    top_p: StrictInt | StrictFloat = 1
    top_k: StrictInt = 0
    repetition_penalty: StrictInt | StrictFloat = 1
    constraint: GrammarConstraint | None = None

    @model_validator(mode="after")
    def require_raw_policy_distribution(self) -> "SamplingParams":
        values = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
        for name, expected in SAMPLING_CONSTANTS.items():
            if values[name] != expected:
                raise ValueError(f"{name} is fixed at {expected}")
        return self

    def engine_params(self) -> dict[str, Any]:
        return {
            **SAMPLING_CONSTANTS,
            "max_tokens": self.max_tokens,
            "logprobs": True,
            **({"constraint": self.constraint.model_dump()} if self.constraint is not None else {}),
        }


def normalize_sampling_params(params: Any, *, limit: int) -> tuple[int, dict[str, Any]]:
    """Validate the shared sampling shape and apply a deployment's stricter cap."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_GENERATED_TOKENS:
        raise ValueError(f"sampling limit must be between 1 and {MAX_GENERATED_TOKENS}")
    try:
        sampling = SamplingParams.model_validate(params)
    except ValidationError as exc:
        if any(error.get("type") == "extra_forbidden" for error in exc.errors()):
            raise ValueError("sampling_params contains an unsupported or unsafe field") from exc
        raise ValueError(str(exc)) from exc
    if sampling.max_tokens > limit:
        raise ValueError(f"max_tokens must be between 1 and {limit}")
    return sampling.max_tokens, sampling.engine_params()


class WorkerStartupEvidence(_StrictModel):
    """Normalized facts each backend must prove before the shared runtime is ready."""

    backend: Literal["vllm", "ninfer"]
    base_fingerprint: StrictStr = Field(min_length=1)
    artifact_fingerprint: StrictStr = Field(min_length=1)
    source_artifact_fingerprint: StrictStr | None = None
    quantization: dict[str, Any]
    runtime_quantization: dict[str, Any]
    max_model_len: StrictInt = Field(ge=1)
    max_generated_tokens: StrictInt = Field(ge=1)
    max_num_seqs: StrictInt = Field(ge=1)
    max_loras: StrictInt = Field(ge=1)
    supports_full_token_logprobs: StrictBool
    behavior_distribution: Literal["raw_policy", "raw_policy_public_vocab"]
    logical_model_vocab_size: StrictInt | None = Field(default=None, ge=1)
    public_token_count: StrictInt | None = Field(default=None, ge=1)
    base_trainable_parameters: StrictInt = Field(ge=0)
    backend_evidence: dict[str, Any]

    @model_validator(mode="after")
    def validate_distribution_evidence(self) -> "WorkerStartupEvidence":
        if self.quantization.get("compute_dtype") != "bfloat16":
            raise ValueError("normalized Worker quantization must use canonical bfloat16 compute_dtype")
        logical_vocab = self.logical_model_vocab_size
        public_tokens = self.public_token_count
        if self.behavior_distribution == "raw_policy_public_vocab":
            if logical_vocab is None or public_tokens is None:
                raise ValueError("public-vocabulary behavior distribution requires valid vocabulary sizes")
            if public_tokens > logical_vocab:
                raise ValueError("public token count exceeds logical model vocabulary")
        elif (logical_vocab is None) != (public_tokens is None):
            raise ValueError("raw-policy vocabulary evidence must be complete when present")
        elif (logical_vocab is not None and public_tokens is not None
              and public_tokens > logical_vocab):
            raise ValueError("public token count exceeds logical model vocabulary")
        return self


class GenerationResponse(_StrictModel):
    """Shared generation response accepted by both the HTTP server and client."""

    token_ids: list[StrictInt] = Field(min_length=1)
    log_probs: list[StrictFloat] = Field(min_length=1)
    stop_reason: StrictStr
    extra_fields: dict[str, Any]
    worker_id: StrictStr
    timing: dict[str, StrictFloat]

    @field_validator("token_ids")
    @classmethod
    def nonnegative_token_ids(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("token_ids must be nonnegative")
        return values

    @field_validator("log_probs")
    @classmethod
    def finite_log_probs(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) for value in values):
            raise ValueError("log_probs must be finite")
        return values

    @model_validator(mode="after")
    def aligned_scores(self) -> "GenerationResponse":
        if len(self.token_ids) != len(self.log_probs):
            raise ValueError("token_ids and log_probs must have the same length")
        return self


class GenerateRequest(_StrictModel):
    policy_version: StrictInt = Field(ge=0)
    episode_id: StrictStr = Field(min_length=1, max_length=200)
    adapter_hash: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: StrictStr = Field(min_length=1, max_length=200)
    prompt_ids: list[StrictInt] = Field(min_length=1, max_length=MAX_CONTEXT_TOKENS)
    shared_prefix_tokens: StrictInt = Field(default=0, ge=0, lt=MAX_CONTEXT_TOKENS)
    sampling_params: dict[str, Any]
    purpose: Literal["training", "evaluation"] = "training"
    seed: StrictInt | None = Field(default=None, ge=0, le=MAX_REQUEST_SEED)

    @model_validator(mode="after")
    def shared_prefix_within_prompt(self) -> "GenerateRequest":
        if self.shared_prefix_tokens >= len(self.prompt_ids):
            raise ValueError("shared prefix must end before the final prompt token")
        return self

    @field_validator("prompt_ids")
    @classmethod
    def nonnegative_prompt_ids(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("prompt_ids must be nonnegative token ids")
        return values


class PolicyRequest(_StrictModel):
    version: StrictInt = Field(ge=0)


class EpisodeLeaseRequest(_StrictModel):
    episode_id: StrictStr = Field(min_length=1, max_length=200)
    policy_version: StrictInt = Field(ge=0)
    adapter_hash: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    purpose: Literal["training", "evaluation"] = "training"


class PolicyFile(_StrictModel):
    name: StrictStr
    sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    size_bytes: StrictInt | None = Field(default=None, ge=0)
    url: StrictStr = Field(min_length=1)


class PolicyManifest(_StrictModel):
    version: StrictInt = Field(ge=0)
    base_fingerprint: StrictStr = Field(min_length=1)
    artifact_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    quantization: dict[str, Any]
    files: list[PolicyFile] = Field(min_length=2, max_length=2)
    source_adapter_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_id: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")

    @model_validator(mode="after")
    def validate_source_adapter_identity(self) -> "PolicyManifest":
        ternary = self.quantization.get("mode") == "ternary"
        if ternary and self.artifact_id is None:
            raise ValueError("native T2 policies require the LoRA header artifact_id")
        if not ternary and self.artifact_id is not None:
            raise ValueError("artifact_id is only valid for native T2 policies")
        return self

    @property
    def adapter_hash(self) -> str:
        return self.source_adapter_sha256
