"""HTTP client for the pinned NInfer RL serving ABI.

The Worker owns policy version identity, file-size checks, and episode leases. This
backend converts the selected adapter into NInfer's native artifact, installs its header
artifact_id and source policy identity, and returns the model-distribution logprob for each
generated token as the behavior logprob consumed by the existing rollout contract.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlsplit

import httpx

from .backend import BackendError, NInferOptions, DEFAULT_WORKER_MAX_MODEL_LEN
from .pinned_base import PinnedBase, file_identity
from .generation import GenerationResult, TOOL_CALL_STOP_STRING
from .native_artifact import (
    validate_policy_quantization as validate_policy_artifact_quantization,
    WorkerArtifactError,
    validate_ternary_runtime_quantization,
    validate_ternary_runtime_profile,
)
from ..contracts import worker as wire
from ..contracts.ternary import SUPPORTED_T2_LORA_PROFILES


class _NativeModelBusy(BackendError):
    """Native refused a transition while a short registry lease was active."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_POLICY_SCHEMA = "qwen-rl-native-policy/v1"
_CODER390_NATIVE_POLICY_SCHEMA = "qwen-rl-native-policy/v2"
_NATIVE_POLICY_TIMINGS = frozenset({"mapping_seconds", "write_seconds", "hash_seconds", "export_seconds"})
_MODEL_VOCAB_SIZE = 248_320
_SAMPLING_VOCAB_SIZE = 248_077
_BEHAVIOR_DISTRIBUTION = "raw_policy_public_vocab"
_T2_BASE_ARTIFACT_KIND = "bonsai2_t2"
_CODER390_BASE_ARTIFACT_KIND = "coder390_t2_full_scales_step512"
_T2_ADAPTER_FORMAT = "ninfer-lora-t2-v1"
_T2_MODEL_FAMILY = "qwen3_5_dense"
_T2_INPUT_BASIS = "prism_rotated"
_T2_BASE_WEIGHT_PROFILE = "t2_g128_fp16_rotated"
_T2_SCALE_FOLDING = "alpha_over_r_into_B"
_T2_TARGET_MODULES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
})
_LOGPROB_SENTINEL = -9999.0
_LOGPROB_ORDER_TOL = 1e-5
_POLICY_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "min_p": 0.0,
}
_STOP = {"strings": [TOOL_CALL_STOP_STRING], "include_stop": True}
_STOP_REASONS = frozenset({"tool_call", "eos", "length"})
_NATIVE_ERROR_BODY_LIMIT_BYTES = 8 * 1024
_NATIVE_ERROR_FIELD_LIMITS = {"code": 80, "param": 120, "message": 240}


@dataclass(frozen=True, slots=True)
class NInferPolicyHandle:
    """Opaque Worker policy handle returned after a versioned policy ACK."""

    version: int
    adapter_hash: str
    adapter_artifact_sha256: str | None
    adapter_ninfer_artifact_id: str
    artifact_path: Path
    sync_timings: dict[str, float | bool | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NInferPreparedPolicy:
    """Validated on-disk adapter identity, ready for native installation after wake."""

    version: int
    adapter_hash: str
    adapter_config_hash: str | None
    adapter_artifact_sha256: str | None
    adapter_ninfer_artifact_id: str
    artifact_path: Path
    sync_timings: Mapping[str, float | bool | None]


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BackendError(f"NInfer {name} must be a lowercase SHA-256 digest")
    return value


def _optional_sha(value: Any, name: str) -> str | None:
    return None if value is None else _require_sha(value, name)


def _validate_generation_capacity(capabilities: dict[str, Any], inference: dict[str, Any]) -> int:
    configured = inference.get("max_generated_tokens", wire.MAX_GENERATED_TOKENS)
    if isinstance(configured, bool) or not isinstance(configured, int) or configured < 1:
        raise BackendError("worker_inference.max_generated_tokens must be a positive integer")
    native = capabilities.get("max_generated_tokens")
    if isinstance(native, bool) or not isinstance(native, int) or native < configured:
        raise BackendError("NInfer runtime max_generated_tokens is below the configured Worker limit")
    return native


def _configured_ninfer_speculative_backend(inference: dict[str, Any]) -> str:
    options = inference.get("ninfer_options")
    if options is None:
        return NInferOptions().spec
    if not isinstance(options, dict):
        raise BackendError("worker_inference.ninfer_options must be an object")
    value = options.get("spec", NInferOptions().spec)
    if not isinstance(value, str) or value not in {"none", "mtp"}:
        raise BackendError("NInfer Worker supports only explicitly configured none or mtp speculation")
    return value


def _validate_ternary_source_artifact(
    inference: dict[str, Any], *, lora_rank: int = 16, lora_alpha: int = 16,
) -> tuple[Any, Path, Path]:
    """Read the explicit T2 identity contract without routing through HF/Q4 evidence."""
    model_path = inference.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise BackendError("NInfer T2 Worker requires inference.model_path")
    requested_base = Path(str(inference.get("ninfer_artifact_path", ""))).expanduser()
    if requested_base.is_symlink():
        raise BackendError("T2 ninfer_artifact_path must be a regular file, not a symlink")
    try:
        base_path = requested_base.resolve(strict=True)
    except OSError as exc:
        raise BackendError("configured T2 NInfer artifact is missing or unreadable") from exc
    report_value = inference.get("ninfer_conversion_report")
    if not isinstance(report_value, str) or not report_value.strip():
        raise BackendError("T2 NInfer backend requires the conversion report used by the identity contract")
    report_path = Path(report_value).expanduser()
    if report_path.is_symlink():
        raise BackendError("T2 conversion report must be a regular file, not a symlink")
    try:
        report_path = report_path.resolve(strict=True)
    except OSError as exc:
        raise BackendError("configured T2 conversion report is missing or unreadable") from exc
    identity_value = inference.get("artifact_identity_path")
    if (lora_rank, lora_alpha) == (32, 32) and not identity_value:
        raise BackendError("rank-32 T2 Worker requires the explicit pinned artifact_identity_path")
    try:
        from .native_artifact import read_ternary_worker_artifact

        declared = _require_sha(inference.get("ninfer_artifact_sha256"), "deployment base SHA-256")
        evidence = read_ternary_worker_artifact(
            base_path,
            report_path,
            declared_artifact_sha256=declared,
            artifact_identity_path=identity_value,
            expected_lora_rank=lora_rank,
            expected_lora_alpha=lora_alpha,
        )
    except Exception as exc:
        raise BackendError("NInfer Worker could not validate the frozen T2 artifact") from exc
    return evidence, base_path, report_path


def _validate_ternary_capabilities(
    capabilities: Any,
    *,
    source: Any,
    inference: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(capabilities, dict):
        raise BackendError("NInfer capabilities response must be an object")
    if capabilities.get("ready") is not True or capabilities.get("protocol_version") != 1:
        raise BackendError("NInfer T2 runtime is not ready on the supported protocol")
    if capabilities.get("backend") != "ninfer":
        raise BackendError("NInfer capabilities identify a different backend")
    max_generated_tokens = _validate_generation_capacity(capabilities, inference)
    build_id = capabilities.get("build_id")
    if not isinstance(build_id, str) or not build_id:
        raise BackendError("NInfer T2 capabilities omitted build_id")
    expected_speculative_backend = _configured_ninfer_speculative_backend(inference)
    raw_ninfer_options = inference.get("ninfer_options")
    if raw_ninfer_options is None:
        raw_ninfer_options = {}
    if not isinstance(raw_ninfer_options, dict):
        raise BackendError("worker_inference.ninfer_options must be an object")
    configured_gdn = raw_ninfer_options.get("lora_gdn_projections", False)
    if type(configured_gdn) is not bool:
        raise BackendError("worker_inference.ninfer_options.lora_gdn_projections must be a boolean")
    if configured_gdn is not source.gdn_projections:
        raise BackendError("configured GDN LoRA option differs from the T2 adapter identity")
    exact = {
        "base artifact kind": capabilities.get("base_artifact_kind") == source.base_artifact_kind,
        "base fingerprint": capabilities.get("base_fingerprint") == source.base_fingerprint,
        "artifact fingerprint": capabilities.get("artifact_fingerprint") == source.artifact_fingerprint,
        "NInfer artifact ID": capabilities.get("ninfer_artifact_id") == source.ninfer_artifact_id,
        "NInfer artifact SHA declaration": capabilities.get("ninfer_artifact_sha256") == source.ninfer_artifact_sha256,
        "NInfer artifact SHA verification state": capabilities.get("ninfer_artifact_sha256_verified") is False,
        "LoRA input basis": capabilities.get("input_basis") == source.input_basis,
        "T2 base weight profile": capabilities.get("base_weight_profile") == source.base_weight_profile,
        "T2 base is not four-bit": capabilities.get("is_loaded_in_4bit") is False,
        "frozen base": (
            isinstance(capabilities.get("base_trainable_parameters"), int)
            and not isinstance(capabilities.get("base_trainable_parameters"), bool)
            and capabilities["base_trainable_parameters"] == 0
        ),
        "full token logprobs": capabilities.get("supports_full_token_logprobs") is True,
        "neutral full-vocabulary sampling": capabilities.get("neutral_full_vocab_sampling") is True,
        "behavior distribution": capabilities.get("behavior_distribution") == _BEHAVIOR_DISTRIBUTION,
        "model vocabulary size": capabilities.get("model_vocab_size") == _MODEL_VOCAB_SIZE,
        "sampling vocabulary size": capabilities.get("sampling_vocab_size") == _SAMPLING_VOCAB_SIZE,
        "LoRA support": capabilities.get("supports_lora") is True,
    }
    if "supports_gdn_lora_projections" in capabilities:
        exact["GDN LoRA capability state"] = (
            capabilities.get("supports_gdn_lora_projections") is configured_gdn
        )
    else:
        exact["GDN LoRA support"] = not configured_gdn
    missing = [name for name, valid in exact.items() if not valid]
    if missing:
        raise BackendError("NInfer T2 capabilities are incomplete: " + ", ".join(missing))
    runtime_quantization = capabilities.get("runtime_quantization")
    try:
        runtime_quantization = validate_ternary_runtime_quantization(runtime_quantization)
    except WorkerArtifactError as exc:
        raise BackendError("NInfer T2 runtime did not prove native t2_g128_fp16 execution") from exc
    runtime_quantization = {"mode": "ternary", **runtime_quantization}
    source_quantization = capabilities.get("source_quantization")
    if source_quantization != source.quantization:
        raise BackendError("NInfer T2 source quantization differs from the identity-bound artifact")
    try:
        validate_ternary_runtime_profile(source.as_dict())
    except WorkerArtifactError as exc:
        raise BackendError("NInfer T2 source does not match its supported runtime profile") from exc
    quantized_count = capabilities.get("quantized_tensor_count")
    if isinstance(quantized_count, bool) or not isinstance(quantized_count, int) or quantized_count <= 0:
        raise BackendError("NInfer T2 runtime did not report loaded quantized tensors")
    configured_context = inference.get("max_model_len", DEFAULT_WORKER_MAX_MODEL_LEN)
    configured_concurrency = inference.get("max_num_seqs", 4)
    configured_slots = inference.get("max_loras", 2)
    for value, name in ((configured_context, "max_model_len"),
                        (configured_concurrency, "max_num_seqs"),
                        (configured_slots, "max_loras")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BackendError(f"worker_inference.{name} must be a positive integer")
    for capability_name, minimum, label in (
        ("max_context", configured_context, "context capacity"),
        ("max_concurrency", configured_concurrency, "generation concurrency"),
        ("policy_slots", max(2, configured_slots), "adapter slots"),
    ):
        value = capabilities.get(capability_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise BackendError(f"NInfer T2 {label} is below Worker configuration")
    if capabilities["max_concurrency"] > 8:
        raise BackendError("NInfer T2 runtime concurrency exceeds the supported limit")
    report_identity = (
        source.artifact_identity["source"].get("conversion_report")
        or source.artifact_identity["source"].get("artifact_report")
    )
    if capabilities.get("supports_shared_cache_prefix") is not True:
        raise BackendError("NInfer must support the raw shared-prefix API")
    return {
        "backend": "ninfer",
        "base_artifact_kind": source.base_artifact_kind,
        "build_id": build_id,
        "model_path": source.model_path,
        "ninfer_artifact_path": source.ninfer_artifact_path,
        "ninfer_artifact_origin_path": report_identity.get("recorded_output_path"),
        "base_fingerprint": source.base_fingerprint,
        "artifact_fingerprint": source.artifact_fingerprint,
        "ninfer_artifact_id": source.ninfer_artifact_id,
        "ninfer_artifact_sha256": source.ninfer_artifact_sha256,
        "ninfer_conversion_report": source.conversion_report_path,
        "source_report_sha256": source.source_report_sha256,
        "source_report_kind": source.source_report_kind,
        "artifact_identity": source.artifact_identity,
        "gdn_projections": source.gdn_projections,
        "input_basis": source.input_basis,
        "base_weight_profile": source.base_weight_profile,
        "source_quantization": source.quantization,
        "speculative_backend": expected_speculative_backend,
        "runtime_quantization": runtime_quantization,
        "is_loaded_in_4bit": False,
        "quantized_tensor_count": quantized_count,
        "base_trainable_parameters": 0,
        "max_model_len": capabilities["max_context"],
        "max_generated_tokens": max_generated_tokens,
        "max_num_seqs": capabilities["max_concurrency"],
        "max_loras": capabilities["policy_slots"],
        "supports_full_token_logprobs": True,
        "neutral_full_vocab_sampling": True,
        "behavior_distribution": _BEHAVIOR_DISTRIBUTION,
        "model_vocab_size": _MODEL_VOCAB_SIZE,
        "sampling_vocab_size": _SAMPLING_VOCAB_SIZE,
        "logical_model_vocab_size": _MODEL_VOCAB_SIZE,
        "public_token_count": _SAMPLING_VOCAB_SIZE,
        "supports_lora": True,
        "supports_shared_cache_prefix": True,
        "supports_grammar_policy": capabilities.get("supports_grammar_policy") is True,
        "grammar_compiler_sha256": capabilities.get("grammar_compiler_sha256"),
    }


def _strict_policy_sampling(params: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise BackendError("Worker sampling_params must be an object")
    allowed = {"max_tokens", "logprobs", *wire.SAMPLING_CONSTANTS}
    if set(params) - allowed:
        raise BackendError("NInfer Worker accepts only the fixed raw-policy sampling parameters")
    if params.get("max_tokens", max_tokens) != max_tokens:
        raise BackendError("NInfer max_tokens differs from the Worker request")
    if params.get("logprobs") is not True:
        raise BackendError("NInfer Worker requires sampled-token logprobs")
    for name, expected in wire.SAMPLING_CONSTANTS.items():
        value = params.get(name, expected)
        valid = type(value) is int if name == "top_k" else (
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        )
        if not valid or value != expected:
            raise BackendError(f"NInfer {name} must remain at its raw-policy value {expected}")
    return dict(_POLICY_SAMPLING)


def _effective_seed(worker_seed: Any, request_id: str, requested_seed: int | None) -> tuple[int, str]:
    if requested_seed is not None:
        if isinstance(requested_seed, bool) or not isinstance(requested_seed, int) or not 0 <= requested_seed <= wire.MAX_REQUEST_SEED:
            raise BackendError("NInfer request seed is outside the Worker seed range")
        return requested_seed, "request"
    if isinstance(worker_seed, bool) or not isinstance(worker_seed, int) or not 0 <= worker_seed <= wire.MAX_REQUEST_SEED:
        raise BackendError("NInfer Worker requires its injected stable uint32 seed")
    material = f"{worker_seed}\0{request_id}".encode("utf-8")
    effective = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
    return effective, "worker_rng"


def _expected_t2_adapter_sites(
    identity: dict[str, Any], *, rank: int | None = None, alpha: int | None = None,
) -> dict[str, dict[str, Any]]:
    model = identity.get("model")
    geometry = model.get("geometry") if isinstance(model, dict) else None
    layers = model.get("layer_types") if isinstance(model, dict) else None
    if not isinstance(geometry, dict) or not isinstance(layers, list) or len(layers) != 64:
        raise BackendError("T2 identity omitted the fixed Qwen3.5 layer geometry")
    learner = identity.get("learner")
    adapter = learner.get("adapter") if isinstance(learner, dict) else None
    if not isinstance(adapter, dict):
        raise BackendError("T2 identity omitted its LoRA profile")
    identity_profile = (adapter.get("rank"), adapter.get("alpha"))
    if (type(identity_profile[0]) is not int or type(identity_profile[1]) is not int
            or identity_profile not in SUPPORTED_T2_LORA_PROFILES):
        raise BackendError("T2 identity has an unsupported LoRA rank/alpha profile")
    if (rank is None) != (alpha is None):
        raise BackendError("T2 selected LoRA rank and alpha must be configured together")
    if rank is None:
        rank, alpha = identity_profile
    if (type(rank) is not int or type(alpha) is not int
            or (rank, alpha) not in SUPPORTED_T2_LORA_PROFILES):
        raise BackendError("T2 selected LoRA profile must be rank/alpha 16/16 or 32/32")
    if (rank, alpha) != identity_profile:
        raise BackendError("selected T2 LoRA profile differs from the validated identity")
    input_basis = adapter.get("input_basis") if isinstance(adapter, dict) else None
    if input_basis not in {"prism_rotated", "per_use"}:
        raise BackendError("T2 identity omitted the supported LoRA input basis")
    module_basis = adapter.get("input_basis_by_module") if isinstance(adapter, dict) else None
    if input_basis == "per_use" and not isinstance(module_basis, dict):
        raise BackendError("per-use T2 identity omitted its PEFT target basis map")
    try:
        from ..contracts.lora_targets import (
            gdn_lora_enabled,
            projection_input_bases,
        )

        include_gdn = gdn_lora_enabled(identity)
        if include_gdn and input_basis != "per_use":
            raise BackendError("all-GDN T2 adapters must declare per-use input bases")
        if input_basis == "per_use" and module_basis != projection_input_bases(identity):
            raise BackendError("per-use T2 identity has an incomplete or incorrect projection basis map")
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendError(f"T2 projection basis contract is invalid: {exc}") from exc
    hidden = geometry.get("hidden_size")
    intermediate = geometry.get("intermediate_size")
    heads = geometry.get("num_attention_heads")
    kv_heads = geometry.get("num_key_value_heads")
    head_dim = geometry.get("head_dim")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (hidden, intermediate, heads, kv_heads, head_dim)):
        raise BackendError("T2 identity has invalid LoRA target dimensions")
    query_width = heads * head_dim
    kv_width = kv_heads * head_dim
    result: dict[str, dict[str, Any]] = {}

    def add(
        layer: int,
        name: str,
        input_width: int,
        output_width: int,
        *,
        shared_a: str | None = None,
        module_path: str | None = None,
    ) -> None:
        site = f"text/layers/{layer}/{name}"
        record = {
            "site": site,
            "a_object": shared_a or site + "/lora_a",
            "b_object": site + "/lora_b",
            "a_shape": [rank, input_width],
            "b_shape": [output_width, rank],
        }
        if input_basis == "per_use":
            if module_path is None or module_path not in module_basis:
                raise BackendError(f"per-use T2 target {site} has no PEFT module basis")
            basis = module_basis[module_path]
            if basis not in {"prism_rotated", "primal"}:
                raise BackendError(f"per-use T2 target {site} has an unsupported basis")
            record["input_basis"] = basis
        result[site] = record

    for layer, layer_type in enumerate(layers):
        add(layer, "mlp/gate", hidden, intermediate,
            module_path=f"model.layers.{layer}.mlp.gate_proj")
        add(layer, "mlp/up", hidden, intermediate,
            module_path=f"model.layers.{layer}.mlp.up_proj")
        add(layer, "mlp/down", intermediate, hidden,
            module_path=f"model.layers.{layer}.mlp.down_proj")
        if layer_type == "full_attention":
            shared_a = f"text/layers/{layer}/attention/query_gate/lora_a"
            module_prefix = f"model.layers.{layer}.self_attn."
            add(layer, "attention/query", hidden, query_width, shared_a=shared_a,
                module_path=module_prefix + "q_proj")
            add(layer, "attention/attention_gate", hidden, query_width, shared_a=shared_a,
                module_path=module_prefix + "q_proj")
            add(layer, "attention/key", hidden, kv_width, module_path=module_prefix + "k_proj")
            add(layer, "attention/value", hidden, kv_width, module_path=module_prefix + "v_proj")
            add(layer, "attention/output", query_width, hidden, module_path=module_prefix + "o_proj")
        elif layer_type != "linear_attention":
            raise BackendError("T2 identity has an unsupported layer type")
        elif include_gdn:
            linear_key_heads = geometry.get("linear_num_key_heads")
            linear_key_dim = geometry.get("linear_key_head_dim")
            linear_value_heads = geometry.get("linear_num_value_heads")
            linear_value_dim = geometry.get("linear_value_head_dim")
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in (linear_key_heads, linear_key_dim, linear_value_heads, linear_value_dim)):
                raise BackendError("all-GDN T2 identity has invalid linear-attention geometry")
            key_width = linear_key_heads * linear_key_dim
            value_width = linear_value_heads * linear_value_dim
            module_prefix = f"model.layers.{layer}.linear_attn."
            shared_a = f"text/layers/{layer}/gdn/query_key_value/lora_a"
            add(layer, "gdn/query", hidden, key_width, shared_a=shared_a,
                module_path=module_prefix + "in_proj_qkv")
            add(layer, "gdn/key", hidden, key_width, shared_a=shared_a,
                module_path=module_prefix + "in_proj_qkv")
            add(layer, "gdn/value", hidden, value_width, shared_a=shared_a,
                module_path=module_prefix + "in_proj_qkv")
            add(layer, "gdn/z", hidden, value_width,
                module_path=module_prefix + "in_proj_z")
            add(layer, "gdn/a_projection", hidden, linear_value_heads,
                module_path=module_prefix + "in_proj_a")
            add(layer, "gdn/b_projection", hidden, linear_value_heads,
                module_path=module_prefix + "in_proj_b")
            add(layer, "gdn/output", value_width, hidden,
                module_path=module_prefix + "out_proj")
    return result


def _validate_t2_native_policy(
    adapter_dir: Path, *, version: int, rank: int, alpha: int, evidence: dict[str, Any],
) -> tuple[str, str | None, str | None, str, dict[str, float | None]]:
    if adapter_dir.is_symlink() or not adapter_dir.is_dir():
        raise BackendError("T2 native policy path must be a non-symlink directory")
    artifact_path = adapter_dir / "policy.lora.ninfer"
    descriptor_path = adapter_dir / "policy.lora.ninfer.json"
    for path in (artifact_path, descriptor_path):
        if path.is_symlink() or not path.is_file():
            raise BackendError(f"T2 native policy is missing regular file {path.name}")
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError("T2 native policy descriptor is invalid JSON") from exc
    required_fields = {
        "schema", "version", "source_adapter_sha256", "adapter_config_sha256",
        "base_ninfer_artifact_id", "base_ninfer_sha256", "rank", "alpha",
        "input_basis", "base_weight_profile", "artifact_sha256", "artifact_id",
        "sites", "timings",
    }
    gdn_enabled = evidence.get("gdn_projections", False)
    if type(gdn_enabled) is not bool:
        raise BackendError("T2 Worker evidence omitted a boolean GDN projection identity")
    per_use_adapter = evidence.get("input_basis") == "per_use"
    if gdn_enabled and not per_use_adapter:
        raise BackendError("all-GDN T2 adapters must use per-use input bases")
    allowed_fields = required_fields | {"artifact_sha256_verified"}
    if per_use_adapter:
        allowed_fields.add("gdn_projections")
    if gdn_enabled:
        required_fields.add("gdn_projections")
        allowed_fields.add("gdn_projections")
        if not isinstance(descriptor, dict) or "gdn_projections" not in descriptor:
            raise BackendError("all-GDN native policy descriptor omitted its GDN projection flag")
    descriptor_fields = set(descriptor) if isinstance(descriptor, dict) else set()
    if not required_fields <= descriptor_fields <= allowed_fields:
        raise BackendError("T2 native policy descriptor has an unsupported schema")
    schema = _CODER390_NATIVE_POLICY_SCHEMA if per_use_adapter else _NATIVE_POLICY_SCHEMA
    if descriptor.get("schema") != schema:
        raise BackendError("T2 native policy descriptor has an unsupported schema")
    descriptor_version = descriptor.get("version")
    descriptor_rank = descriptor.get("rank")
    descriptor_alpha = descriptor.get("alpha")
    if (isinstance(descriptor_version, bool) or not isinstance(descriptor_version, int)
            or descriptor_version != version):
        raise BackendError("T2 native policy descriptor version differs from the requested policy")
    if (isinstance(descriptor_rank, bool) or not isinstance(descriptor_rank, int)
            or descriptor_rank != rank or isinstance(descriptor_alpha, bool)
            or not isinstance(descriptor_alpha, int) or descriptor_alpha != alpha):
        raise BackendError("T2 native policy descriptor rank or alpha differs from Worker configuration")
    source_hash = _require_sha(descriptor.get("source_adapter_sha256"), "source adapter identity")
    config_hash = descriptor.get("adapter_config_sha256")
    if config_hash is not None:
        config_hash = _require_sha(config_hash, "declared adapter config SHA-256")
    if descriptor.get("base_ninfer_artifact_id") != evidence.get("ninfer_artifact_id"):
        raise BackendError("T2 native policy descriptor names a different base NInfer artifact ID")
    base_sha = _require_sha(descriptor.get("base_ninfer_sha256"), "declared base NInfer artifact SHA-256")
    if base_sha != evidence.get("ninfer_artifact_sha256"):
        raise BackendError("T2 native policy descriptor names a different declared base artifact identity")
    if descriptor.get("input_basis") != evidence.get("input_basis"):
        raise BackendError("T2 native policy descriptor uses a different LoRA input basis")
    if descriptor.get("base_weight_profile") != evidence.get("base_weight_profile"):
        raise BackendError("T2 native policy descriptor names a different base weight profile")
    if "gdn_projections" in descriptor:
        if (type(descriptor["gdn_projections"]) is not bool
                or descriptor["gdn_projections"] is not gdn_enabled):
            raise BackendError("T2 native policy descriptor GDN projection flag differs from its base identity")
    elif gdn_enabled:
        raise BackendError("all-GDN native policy descriptor omitted its GDN projection flag")
    sites = descriptor.get("sites")
    expected_sites = _expected_t2_adapter_sites(
        evidence.get("artifact_identity", {}), rank=rank, alpha=alpha,
    )
    if not isinstance(sites, list) or len(sites) != len(expected_sites):
        raise BackendError("T2 native policy site inventory is incomplete")
    observed: dict[str, dict[str, Any]] = {}
    for record in sites:
        if not isinstance(record, dict) or not isinstance(record.get("site"), str):
            raise BackendError("T2 native policy site record has an unsupported schema")
        label = record.get("site")
        if not isinstance(label, str) or label in observed:
            raise BackendError("T2 native policy site inventory has a missing or duplicate site")
        expected = expected_sites.get(label)
        if expected is None or set(record) != set(expected):
            raise BackendError("T2 native policy site record has an unsupported schema")
        observed[label] = record
    if observed != expected_sites:
        raise BackendError("T2 native policy sites, factor shapes, or input bases differ from the identity")
    timings = descriptor.get("timings")
    if not isinstance(timings, dict) or set(timings) != _NATIVE_POLICY_TIMINGS:
        raise BackendError("T2 native policy descriptor timing fields are incomplete")
    clean_timings: dict[str, float | None] = {}
    for key, value in timings.items():
        if key == "hash_seconds" and value is None:
            clean_timings[key] = None
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise BackendError("T2 native policy descriptor timings must be finite nonnegative seconds")
        clean_timings[key] = float(value)
    artifact_sha = descriptor.get("artifact_sha256")
    if artifact_sha is not None:
        artifact_sha = _require_sha(artifact_sha, "declared native adapter artifact SHA-256")
    if ("artifact_sha256_verified" in descriptor
            and descriptor.get("artifact_sha256_verified") is not False):
        raise BackendError("T2 native descriptor claims an unperformed artifact SHA check")
    artifact_id = descriptor.get("artifact_id")
    if not isinstance(artifact_id, str) or re.fullmatch(r"[0-9a-f]{32}", artifact_id) is None:
        raise BackendError("T2 native policy descriptor omitted a valid NInfer artifact ID")
    return source_hash, config_hash, artifact_sha, artifact_id, clean_timings


def _native_error_detail(response: httpx.Response) -> str | None:
    """Extract only bounded structured NativeAPI error fields, never the full body."""
    if len(response.content) > _NATIVE_ERROR_BODY_LIMIT_BYTES:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None

    fields: list[str] = []
    for name, limit in _NATIVE_ERROR_FIELD_LIMITS.items():
        value = error.get(name)
        if not isinstance(value, str):
            continue
        normalized = "".join(char for char in " ".join(value.split()) if char.isprintable())
        normalized = normalized[:limit]
        if normalized:
            fields.append(f"{name}={normalized}")
    return ", ".join(fields) if fields else None


class NInferBackend:
    """Client implementing the same async duck-typed boundary as VLLMBackend."""

    engine_process = None
    _process_resume_ready = False

    def __init__(
        self,
        *,
        client: httpx.Client,
        inference: dict[str, Any],
        lora_rank: int,
        lora_alpha: int,
        evidence: dict[str, Any],
        base_artifact_path: Path,
        base_conversion_report_path: Path,
        pinned_base: PinnedBase | None = None,
    ) -> None:
        self.client = client
        self.inference = dict(inference)
        generation_timeout = inference.get("ninfer_generation_timeout_seconds")
        self.generation_timeout_seconds = 1200.0 if generation_timeout is None else float(generation_timeout)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.evidence = evidence
        self.base_artifact_path = base_artifact_path
        self.base_conversion_report_path = base_conversion_report_path
        self.pinned_base = pinned_base
        self.policies: dict[int, NInferPolicyHandle] = {}
        self.model_id: str | None = None
        self.closed = False
        self.startup_evidence = wire.WorkerStartupEvidence(
            backend="ninfer",
            base_fingerprint=evidence["base_fingerprint"],
            artifact_fingerprint=evidence["artifact_fingerprint"],
            quantization=evidence["source_quantization"],
            runtime_quantization=evidence["runtime_quantization"],
            max_model_len=evidence["max_model_len"],
            max_generated_tokens=evidence["max_generated_tokens"],
            max_num_seqs=evidence["max_num_seqs"],
            max_loras=evidence["max_loras"],
            supports_full_token_logprobs=evidence["supports_full_token_logprobs"],
            behavior_distribution=evidence["behavior_distribution"],
            logical_model_vocab_size=evidence["logical_model_vocab_size"],
            public_token_count=evidence["public_token_count"],
            base_trainable_parameters=evidence["base_trainable_parameters"],
            backend_evidence=evidence,
        )

    def validate_policy_quantization(self, policy_quantization: dict[str, Any]) -> None:
        validate_policy_artifact_quantization(
            "ternary", self.startup_evidence.quantization, policy_quantization,
        )

    def _request(
        self, method: str, path: str, *, json_body: dict[str, Any] | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {"json": json_body}
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        try:
            response = self.client.request(method, path, **request_kwargs)
        except httpx.HTTPError as exc:
            raise BackendError(f"NInfer {method} {path} request failed ({type(exc).__name__})") from exc
        if response.status_code != 200:
            detail = _native_error_detail(response)
            suffix = f" ({detail})" if detail else ""
            error_type = BackendError
            if response.status_code == 409:
                try:
                    native_error = response.json().get("error", {})
                except (ValueError, AttributeError):
                    native_error = {}
                if isinstance(native_error, dict) and native_error.get("code") == "model_busy":
                    error_type = _NativeModelBusy
            raise error_type(
                f"NInfer {method} {path} returned HTTP {response.status_code}{suffix}"
            )
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise BackendError(f"NInfer {method} {path} returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise BackendError(f"NInfer {method} {path} response must be an object")
        return body

    async def _thread_call_without_early_cancellation(
        self,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Wait for an already-started blocking operation to finish before cancellation returns.

        Cancelling ``asyncio.to_thread`` only cancels its awaiter. The thread continues, so the
        caller must retain its surrounding lock or lease until the result is consumed.
        """
        operation_task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(operation_task)
        except asyncio.CancelledError as cancellation:
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            while not operation_task.done():
                try:
                    await asyncio.shield(operation_task)
                except asyncio.CancelledError:
                    if current is not None:
                        current.uncancel()
                    continue
                except Exception:
                    break
            try:
                operation_task.result()
            except BaseException:
                # Preserve cancellation while consuming any terminal thread exception.
                pass
            raise cancellation

    async def _request_without_early_cancellation(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> dict[str, Any]:
        """Hold the caller until a sync HTTP call finishes, even after cancellation."""
        return await self._thread_call_without_early_cancellation(
            self._request,
            method,
            path,
            json_body=json_body,
            timeout=timeout,
        )

    async def prepare_adapter(
        self,
        version: int,
        path: Path,
        *,
        adapter_hash: str,
    ) -> NInferPreparedPolicy:
        """Validate one identity-bound native T2 policy without converting PEFT files."""
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise BackendError("NInfer policy version must be a nonnegative integer")
        adapter_hash = _require_sha(adapter_hash, "policy adapter identity")
        if self.pinned_base is None:
            raise BackendError("T2 native policy requires deployment-pinned frozen base identity")
        self.pinned_base.check()
        adapter_dir = Path(path)
        started = perf_counter()
        declared_hash, config_hash, artifact_sha, artifact_id, export_timings = await asyncio.to_thread(
            _validate_t2_native_policy,
            adapter_dir,
            version=version,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            evidence=self.evidence,
        )
        if declared_hash != adapter_hash:
            raise BackendError("T2 native policy source identity differs from the Worker manifest")
        self.pinned_base.check()
        timings: dict[str, float | bool | None] = {
            "native_direct_validation_seconds": max(0.0, perf_counter() - started),
            "native_direct_export": True,
            "converter_seconds": 0.0,
        }
        for key, value in export_timings.items():
            timings[f"learner_native_export_{key}"] = value
        return NInferPreparedPolicy(
            version=version,
            adapter_hash=adapter_hash,
            adapter_config_hash=config_hash,
            adapter_artifact_sha256=artifact_sha,
            adapter_ninfer_artifact_id=artifact_id,
            artifact_path=adapter_dir / "policy.lora.ninfer",
            sync_timings=MappingProxyType(dict(timings)),
        )

    async def add_adapter(
        self,
        version: int,
        path: Path,
        *,
        adapter_hash: str,
    ) -> NInferPolicyHandle:
        prepared = await self.prepare_adapter(version, path, adapter_hash=adapter_hash)
        previous = self.policies.get(prepared.version)
        if previous is not None:
            if previous.adapter_hash != prepared.adapter_hash:
                raise BackendError("NInfer policy version was reused with a different adapter identity")
            return previous

        timings = dict(prepared.sync_timings)
        native_install_started = perf_counter()
        ack = await self._request_without_early_cancellation(
            "POST",
            "/v1/rl/policies",
            json_body={
                "version": prepared.version,
                "adapter_sha256": prepared.adapter_hash,
                "artifact_path": str(prepared.artifact_path.resolve()),
                "artifact_id": prepared.adapter_ninfer_artifact_id,
                "artifact_sha256": prepared.adapter_artifact_sha256,
            },
        )
        if (
            "artifact_sha256" not in ack
            or "adapter_config_sha256" not in ack
            or ack.get("version") != prepared.version
            or ack.get("adapter_sha256") != prepared.adapter_hash
            or ack.get("artifact_id") != prepared.adapter_ninfer_artifact_id
            or ack.get("artifact_sha256") != prepared.adapter_artifact_sha256
            or ack.get("artifact_sha256_verified") is not False
            or ack.get("adapter_config_sha256") != prepared.adapter_config_hash
        ):
            raise BackendError("NInfer policy ACK differs from versioned source and header artifact identity")
        timings["native_install_seconds"] = max(0.0, perf_counter() - native_install_started)
        handle = NInferPolicyHandle(
            prepared.version,
            prepared.adapter_hash,
            prepared.adapter_artifact_sha256,
            prepared.adapter_ninfer_artifact_id,
            prepared.artifact_path,
            sync_timings=timings,
        )
        self.policies[prepared.version] = handle
        return handle

    async def remove_adapter(self, request: NInferPolicyHandle) -> None:
        if not isinstance(request, NInferPolicyHandle):
            raise BackendError("NInfer remove_adapter requires its policy handle")
        ack = await asyncio.to_thread(
            self._request, "DELETE", f"/v1/rl/policies/{request.version}"
        )
        if (
            "artifact_sha256" not in ack
            or ack.get("version") != request.version
            or ack.get("adapter_sha256") != request.adapter_hash
            or ack.get("artifact_sha256") != request.adapter_artifact_sha256
            or ack.get("artifact_sha256_verified") is not False
            or ack.get("artifact_id") != request.adapter_ninfer_artifact_id
        ):
            raise BackendError("NInfer policy removal ACK differs from the resident policy identity")
        self.policies.pop(request.version, None)

    def _resolve_model_id(self) -> str:
        if self.model_id is not None:
            return self.model_id
        catalog = self._request("GET", "/v1/models")
        models = catalog.get("data")
        if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
            raise BackendError("NInfer model catalog must contain exactly one served model")
        model_id = models[0].get("id")
        if not isinstance(model_id, str) or not model_id:
            raise BackendError("NInfer model catalog omitted the served model ID")
        self.model_id = model_id
        return model_id

    def _model_residency_path(self) -> str:
        return f"/v1/models/{quote(self._resolve_model_id(), safe='')}/residency"

    def _attest_registry_admission(self, response: dict[str, Any]) -> dict[str, Any]:
        """Verify the single-model registry gate, which owns HTTP auto-wake.

        EngineModelService::suspend sets engine auto_resume=true even for a held
        registry sleep. On this pinned native server, a Sleeping registry model
        has /health=503 iff its admission is held. Keep raw auto_resume evidence;
        never rewrite it to imply an engine-level fence.
        """
        suspended = response.get("state") == "suspended"
        expected_status = "sleeping" if suspended else "loaded"
        if (response.get("model") != self.model_id or response.get("enabled") is not True
                or response.get("state") not in {"resident", "suspended"}
                or response.get("status") != expected_status
                or response.get("last_error") is not None):
            raise BackendError(f"NInfer registry residency is not verified: {response}")
        health = self.client.get("/health", timeout=10.0)
        expected_http, expected_health = (503, "unavailable") if suspended else (200, "ok")
        if health.status_code != expected_http or health.json().get("status") != expected_health:
            raise BackendError("NInfer registry health does not prove the expected admission gate")
        return {**response, "registry_admission_held": suspended}

    async def get_residency(self) -> dict[str, Any]:
        if self.engine_process is not None and self.engine_process.state == "stopped":
            return self.engine_process.stopped_status(self.model_id)
        if self.model_id is None:
            await asyncio.to_thread(self._resolve_model_id)
        response = await self._request_without_early_cancellation(
            "GET", self._model_residency_path(), timeout=600.0,
        )
        return await asyncio.to_thread(self._attest_registry_admission, response)

    async def suspend_model(self) -> dict[str, Any]:
        if self.engine_process is not None:
            from .ninfer_process import settle
            if self.engine_process.state == "stopped":
                return self.engine_process.stopped_status(self.model_id)
            if self.model_id is None:
                await asyncio.to_thread(self._resolve_model_id)
            self._process_resume_ready = False
            await settle(self.engine_process.stop())
            return self.engine_process.stopped_status(self.model_id)
        if self.model_id is None:
            await asyncio.to_thread(self._resolve_model_id)
        # Read-only Native observers briefly hold registry leases. Only the
        # explicit, side-effect-free busy refusal may be retried; uncertain
        # failures still fence the lifecycle without waking compute.
        deadline = perf_counter() + 10.0
        while True:
            try:
                response = await self._request_without_early_cancellation(
                    "POST", f"/v1/models/{quote(self.model_id, safe='')}/suspend",
                    json_body={"auto_resume": False}, timeout=600.0,
                )
                break
            except _NativeModelBusy:
                if perf_counter() >= deadline:
                    raise
                await asyncio.sleep(0.05)
        if (response.get("model") != self.model_id or response.get("state") != "suspended"
                or response.get("enabled") is not True
                or (response.get("last_suspend") or {}).get("cache_discarded") is not True):
            raise BackendError(f"NInfer did not confirm suspended residency: {response}")
        return await asyncio.to_thread(self._attest_registry_admission, response)

    async def resume_model(self) -> dict[str, Any]:
        if self.engine_process is not None:
            from .ninfer_process import settle
            return await settle(self._restart_native())
        if self.model_id is None:
            await asyncio.to_thread(self._resolve_model_id)
        response = await self._request_without_early_cancellation(
            "POST", f"/v1/models/{quote(self.model_id, safe='')}/resume", json_body={},
            timeout=600.0,
        )
        if (response.get("model") != self.model_id or response.get("state") != "resident"
                or response.get("enabled") is not True):
            raise BackendError("NInfer did not confirm resident model memory after wake")
        return await asyncio.to_thread(self._attest_registry_admission, response)

    async def _restart_native(self) -> dict[str, Any]:
        if self.engine_process.state == "running" and self._process_resume_ready:
            return await self.get_residency()
        self._process_resume_ready = False
        old = dict(self.policies)
        try:
            if self.pinned_base is not None:
                self.pinned_base.check()
            await self.engine_process.start()
            capabilities = await asyncio.to_thread(_capabilities_request, self.client)
            # Every restart is gated on the same runtime profile as initial load.
            for key in ("ninfer_artifact_id", "ninfer_artifact_sha256", "base_fingerprint",
                        "artifact_fingerprint", "input_basis", "base_weight_profile", "build_id"):
                if capabilities.get(key) != self.evidence.get(key):
                    raise BackendError(f"Native restart identity differs: {key}")
            self.model_id = None
            self.policies.clear()
            for handle in old.values():
                installed = await self.add_adapter(handle.version, handle.artifact_path.parent,
                                                   adapter_hash=handle.adapter_hash)
                if installed.adapter_ninfer_artifact_id != handle.adapter_ninfer_artifact_id:
                    raise BackendError("Native replay changed adapter artifact identity")
                # Existing Worker entries retain these exact opaque handles.
                self.policies[handle.version] = handle
            result = await self.get_residency()
            self._process_resume_ready = True
            return result
        except BaseException:
            self.policies = old
            if self.engine_process.state == "running":
                await self.engine_process.stop()
            raise

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        max_tokens: int,
        sampling_params: dict[str, Any],
        lora_request: NInferPolicyHandle,
        token_count_callback: Callable[[int], None] | None = None,
        seed: int | None = None,
        shared_prefix_tokens: int = 0,
    ) -> GenerationResult:
        if not isinstance(lora_request, NInferPolicyHandle):
            raise BackendError("NInfer generation requires a resident policy handle")
        resident = self.policies.get(lora_request.version)
        if resident != lora_request:
            raise BackendError("NInfer generation policy is not resident on this Worker")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
            raise BackendError("NInfer request_id must be a non-empty string of at most 200 characters")
        limit = min(int(self.inference.get("max_generated_tokens", wire.MAX_GENERATED_TOKENS)),
                    wire.MAX_GENERATED_TOKENS)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= limit:
            raise BackendError(f"NInfer max_tokens is outside the Worker generation limit ({limit})")
        if (
            not isinstance(prompt_ids, list) or not prompt_ids
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in prompt_ids)
        ):
            raise BackendError("NInfer prompt_ids must be a non-empty list of nonnegative token IDs")
        if type(shared_prefix_tokens) is not int or not 0 <= shared_prefix_tokens < len(prompt_ids):
            raise BackendError("shared prefix must end before the final prompt token")
        constraint = sampling_params.get("constraint")
        sampling = _strict_policy_sampling({k:v for k,v in sampling_params.items() if k != "constraint"}, max_tokens)
        if constraint is not None:
            constraint = wire.GrammarConstraint.model_validate(constraint).model_dump()
            if self.evidence.get("grammar_compiler_sha256") != constraint["compiler_sha256"]:
                raise BackendError("Native grammar compiler differs from the requested Learner compiler")
            if self.evidence.get("supports_grammar_policy") is not True:
                raise BackendError("Native engine does not support grammar-conditioned policy; enable structured_output and use a matching Native build")
        effective_seed, seed_mode = _effective_seed(
            self.inference.get("seed"), request_id, seed,
        )
        body = {
            "request_id": request_id,
            "policy_version": lora_request.version,
            "adapter_sha256": lora_request.adapter_hash,
            "prompt_token_ids": list(prompt_ids),
            "max_tokens": max_tokens,
            "sampling": {**sampling, "seed": effective_seed},
            "stop": _STOP,
        }
        if constraint is not None:
            body["constraint"] = constraint
        if shared_prefix_tokens:
            body["shared_prefix_tokens"] = shared_prefix_tokens
        response = await self._request_without_early_cancellation(
            "POST", "/v1/rl/generate", json_body=body, timeout=httpx.Timeout(self.generation_timeout_seconds),
        )
        token_ids, log_probs, delta_metrics = self._parse_generation_response(
            response,
            request_id=request_id,
            policy=lora_request,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            constraint=constraint,
        )
        if token_count_callback is not None:
            token_count_callback(len(token_ids))
        metrics = response.get("metrics")
        assert isinstance(metrics, dict)
        metrics = dict(metrics)
        metrics.update(delta_metrics)
        metrics["behavior_logprob_source"] = "sampling_logprob"
        metrics["behavior_distribution"] = "grammar_conditioned_policy/v1" if constraint is not None else _BEHAVIOR_DISTRIBUTION
        if constraint is not None:
            metrics["grammar_sha256"] = constraint["sha256"]
            metrics["grammar_compiler_sha256"] = self.evidence["grammar_compiler_sha256"]
        metrics["model_vocab_size"] = _MODEL_VOCAB_SIZE
        metrics["sampling_vocab_size"] = _SAMPLING_VOCAB_SIZE
        metrics["seed_mode"] = seed_mode
        metrics["effective_seed"] = effective_seed
        metrics["base_fingerprint"] = self.evidence["base_fingerprint"]
        metrics["artifact_fingerprint"] = self.evidence["artifact_fingerprint"]
        metrics["ninfer_artifact_id"] = self.evidence["ninfer_artifact_id"]
        metrics["ninfer_artifact_sha256"] = self.evidence["ninfer_artifact_sha256"]
        metrics["ninfer_artifact_sha256_verified"] = False
        metrics["base_artifact_kind"] = self.evidence["base_artifact_kind"]
        metrics["input_basis"] = self.evidence["input_basis"]
        metrics["base_weight_profile"] = self.evidence["base_weight_profile"]
        metrics["speculative_backend"] = self.evidence["speculative_backend"]
        metrics["adapter_sha256"] = lora_request.adapter_hash
        metrics["adapter_artifact_sha256"] = lora_request.adapter_artifact_sha256
        metrics["artifact_id"] = lora_request.adapter_ninfer_artifact_id
        metrics["artifact_sha256_verified"] = False
        return GenerationResult(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=response["stop_reason"],
            metrics=metrics,
        )

    def _parse_generation_response(
        self,
        response: dict[str, Any],
        *,
        request_id: str,
        policy: NInferPolicyHandle,
        prompt_ids: list[int],
        max_tokens: int,
        constraint: dict | None = None,
    ) -> tuple[list[int], list[float], dict[str, float]]:
        if response.get("request_id") != request_id:
            raise BackendError("NInfer generation response request_id mismatch")
        if response.get("policy_version") != policy.version:
            raise BackendError("NInfer generation response policy version mismatch")
        if response.get("adapter_sha256") != policy.adapter_hash:
            raise BackendError("NInfer generation response adapter identity mismatch")
        if ("artifact_sha256" not in response
                or response.get("artifact_sha256") != policy.adapter_artifact_sha256):
            raise BackendError("NInfer generation response declared adapter artifact SHA mismatch")
        if response.get("artifact_sha256_verified") is not False:
            raise BackendError("NInfer generation response claims an unperformed adapter artifact SHA check")
        if response.get("artifact_id") != policy.adapter_ninfer_artifact_id:
            raise BackendError("NInfer generation response adapter artifact ID mismatch")
        if response.get("base_fingerprint") != self.evidence["base_fingerprint"]:
            raise BackendError("NInfer generation response base fingerprint mismatch")
        if response.get("artifact_fingerprint") != self.evidence["artifact_fingerprint"]:
            raise BackendError("NInfer generation response artifact fingerprint mismatch")
        if (response.get("ninfer_artifact_sha256_verified") is not False
                or response.get("ninfer_artifact_sha256") != self.evidence["ninfer_artifact_sha256"]):
            raise BackendError("NInfer generation response T2 base artifact identity mismatch")
        if (response.get("base_artifact_kind") != self.evidence["base_artifact_kind"]
                or response.get("input_basis") != self.evidence["input_basis"]
                or response.get("base_weight_profile") != self.evidence["base_weight_profile"]):
            raise BackendError("NInfer generation response T2 artifact type or LoRA basis mismatch")
        if response.get("ninfer_artifact_id") != self.evidence["ninfer_artifact_id"]:
            raise BackendError("NInfer generation response representation ID mismatch")
        expected_distribution = "grammar_conditioned_policy/v1" if constraint is not None else self.evidence.get("behavior_distribution")
        if response.get("behavior_distribution") != expected_distribution:
            raise BackendError("NInfer generation response behavior distribution mismatch")
        if constraint is not None and (response.get("grammar_compiler_sha256") != constraint["compiler_sha256"]
                or response.get("grammar_sha256") != constraint["sha256"]):
            raise BackendError("NInfer generation response grammar identity mismatch")
        if response.get("model_vocab_size") != self.evidence.get("model_vocab_size"):
            raise BackendError("NInfer generation response model vocabulary size mismatch")
        if response.get("sampling_vocab_size") != self.evidence.get("sampling_vocab_size"):
            raise BackendError("NInfer generation response sampling vocabulary size mismatch")
        token_ids = response.get("token_ids")
        records = response.get("rl_tokens")
        if (
            not isinstance(token_ids, list) or not token_ids
            or len(token_ids) > max_tokens
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids)
        ):
            raise BackendError("NInfer returned invalid or over-budget token_ids")
        if not isinstance(records, list) or len(records) != len(token_ids):
            raise BackendError("NInfer returned misaligned generated-token logprobs")
        log_probs: list[float] = []
        deltas: list[float] = []
        for token_id, record in zip(token_ids, records):
            record_id = record.get("id") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or isinstance(record_id, bool)
                or not isinstance(record_id, int)
                or record_id != token_id
            ):
                raise BackendError("NInfer generated token ID and logprob record are misaligned")
            model_lp = record.get("model_logprob")
            sampling_lp = record.get("sampling_logprob")
            if (
                isinstance(model_lp, bool) or not isinstance(model_lp, (int, float))
                or not math.isfinite(model_lp)
                or isinstance(sampling_lp, bool) or not isinstance(sampling_lp, (int, float))
                or not math.isfinite(sampling_lp)
            ):
                raise BackendError("NInfer returned a non-finite model or sampling logprob")
            if float(sampling_lp) == _LOGPROB_SENTINEL:
                raise BackendError("NInfer returned a sentinel instead of an exact sampled-token behavior logprob")
            if float(sampling_lp) > 0.0:
                raise BackendError("NInfer sampled-token behavior logprob must be non-positive")
            if float(model_lp) > float(sampling_lp) + _LOGPROB_ORDER_TOL:
                raise BackendError("NInfer full-model logprob exceeds the public-vocabulary behavior logprob")
            deltas.append(abs(float(model_lp) - float(sampling_lp)))
            # b is the probability under the exact distribution that sampled this token.
            log_probs.append(float(sampling_lp))
        if response.get("stop_reason") not in _STOP_REASONS:
            raise BackendError("NInfer returned an unsupported stop_reason")
        metrics = response.get("metrics")
        if not isinstance(metrics, dict):
            raise BackendError("NInfer generation response omitted metrics")
        speculative = metrics.get("speculative")
        if (not isinstance(speculative, dict)
                or speculative.get("backend") != self.evidence["speculative_backend"]):
            actual_spec = speculative.get("backend") if isinstance(speculative, dict) else None
            raise BackendError(
                "NInfer generation used a speculative backend different from Worker configuration "
                f"(observed={actual_spec!r}, expected={self.evidence['speculative_backend']!r})"
            )
        if metrics.get("prompt_tokens") != len(prompt_ids):
            raise BackendError("NInfer generation metrics prompt token count mismatch")
        if metrics.get("completion_tokens") != len(token_ids):
            raise BackendError("NInfer generation metrics completion token count mismatch")
        for key in ("prompt_seconds", "generation_seconds"):
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise BackendError(f"NInfer generation metrics {key} must be finite and nonnegative")
        return token_ids, log_probs, {
            "model_vs_sampling_logprob_max_abs_delta": max(deltas, default=0.0),
            "model_vs_sampling_logprob_rms_delta": math.sqrt(
                sum(delta * delta for delta in deltas) / len(deltas)
            ),
        }

    async def log_stats(self) -> None:
        """NInfer owns structured request metrics; no synthetic scheduler snapshot is added."""

    def resource_snapshot(self) -> dict[str, Any] | None:
        return None

    def close(self) -> None:
        if not self.closed:
            if self.engine_process is not None:
                self.engine_process.close()
            self.client.close()
            self.closed = True


async def load_ninfer_backend(
    inference: dict[str, Any],
    *,
    lora_rank: int,
    lora_alpha: int,
    http_transport: httpx.BaseTransport | None = None,
) -> NInferBackend:
    """Validate one explicit native T2 artifact and connect to its NInfer service."""
    if not isinstance(inference, dict) or inference.get("quantization") != "ternary":
        raise BackendError("NInfer Worker supports only explicitly configured T2 artifacts")
    for key in ("ninfer_url", "ninfer_artifact_path", "ninfer_conversion_report", "ninfer_artifact_sha256"):
        if not isinstance(inference.get(key), str) or not inference[key].strip():
            raise BackendError(f"worker_inference.{key} must be an explicit non-empty value")
    parsed = urlsplit(inference["ninfer_url"])
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise BackendError("worker_inference.ninfer_url must be an HTTP(S) engine origin without credentials, path, query, or fragment")
    for value, label in ((lora_rank, "LoRA rank"), (lora_alpha, "LoRA alpha")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BackendError(f"NInfer published {label} must be positive")
    if (lora_rank, lora_alpha) not in SUPPORTED_T2_LORA_PROFILES:
        raise BackendError("NInfer Worker supports only rank/alpha 16/16 or 32/32 T2 adapters")
    worker_seed = inference.get("seed")
    if (isinstance(worker_seed, bool) or not isinstance(worker_seed, int)
            or not 0 <= worker_seed <= wire.MAX_REQUEST_SEED):
        raise BackendError("NInfer Worker requires its injected stable uint32 seed")

    requested_base = Path(inference["ninfer_artifact_path"]).expanduser()
    requested_report = Path(inference["ninfer_conversion_report"]).expanduser()
    before_base = file_identity(requested_base)
    before_report = file_identity(requested_report)
    source, base_artifact_path, base_conversion_report_path = await asyncio.to_thread(
        _validate_ternary_source_artifact, inference, lora_rank=lora_rank, lora_alpha=lora_alpha,
    )
    pinned_base = PinnedBase.capture(
        base_artifact_path,
        base_conversion_report_path,
        before_base,
        before_report,
        source.ninfer_artifact_sha256,
    )
    client = httpx.Client(
        base_url=inference["ninfer_url"].rstrip("/"),
        timeout=httpx.Timeout(180.0),
        transport=http_transport,
    )
    try:
        capabilities = await asyncio.to_thread(lambda: _capabilities_request(client))
        evidence = _validate_ternary_capabilities(
            capabilities,
            source=source,
            inference=inference,
        )
        evidence["base_content_verification"] = "deployment_declared"
        return NInferBackend(
            client=client,
            inference=inference,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            evidence=evidence,
            base_artifact_path=base_artifact_path,
            base_conversion_report_path=base_conversion_report_path,
            pinned_base=pinned_base,
        )
    except Exception:
        client.close()
        raise


def _capabilities_request(client: httpx.Client) -> dict[str, Any]:
    try:
        response = client.get("/v1/rl/capabilities")
    except httpx.HTTPError as exc:
        raise BackendError(f"NInfer capabilities request failed ({type(exc).__name__})") from exc
    if response.status_code != 200:
        raise BackendError(f"NInfer capabilities returned HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise BackendError("NInfer capabilities returned invalid JSON") from exc
    if not isinstance(body, dict):
        raise BackendError("NInfer capabilities response must be an object")
    return body
