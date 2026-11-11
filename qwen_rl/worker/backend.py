"""Backend-neutral contracts and lazy selection for generation Workers."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from ..contracts import worker as wire
from .generation import GenerationResult


class BackendError(RuntimeError):
    """A Worker inference backend could not prove or execute its configured policy."""


class WorkerBackend(Protocol):
    """The runtime surface shared by generation engines."""

    evidence: dict[str, Any]
    startup_evidence: wire.WorkerStartupEvidence

    async def add_adapter(self, version: int, path: Path, *, adapter_hash: str) -> Any: ...

    async def remove_adapter(self, request: Any) -> None: ...

    def validate_policy_quantization(self, policy_quantization: dict[str, Any]) -> None: ...

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        max_tokens: int,
        sampling_params: dict[str, Any],
        lora_request: Any,
        token_count_callback: Any = None,
        seed: int | None = None,
    ) -> GenerationResult: ...

    async def log_stats(self) -> None: ...

    def close(self) -> None: ...


NINFER_FIELDS = frozenset({
    "ninfer_url",
    "ninfer_artifact_path",
    "ninfer_conversion_report",
    "ninfer_converter_python",
    "ninfer_converter_path",
})
NINFER_CONFIG_FIELDS = NINFER_FIELDS | {"ninfer_options", "ninfer_artifact_sha256", "ninfer_generation_timeout_seconds"}
DEFAULT_WORKER_MAX_MODEL_LEN = 65_536
DEFAULT_WORKER_MAX_NUM_SEQS = 4
DEFAULT_WORKER_MAX_LORAS = 2
# Per-deployment request bound for one generation. The deployment renders
# worker_inference.max_generated_tokens into /config/worker.json, so raising the bound is a
# config edit plus a restart; the wire ceiling still caps it.
DEFAULT_MAX_GENERATED_TOKENS = 4_096
DEFAULT_LORA_RANK = 16


@dataclass(frozen=True)
class NInferOptions:
    """Typed defaults for native NInfer engine flags owned by the Worker config."""

    kv_capacity: str = "auto"
    cuda_graph_allowance_mib: int = 640
    spec: str = "mtp"
    draft_tokens: int = 3
    ngram_draft_tokens: int = 0
    kv_dtype: str = "rk4v4"
    prefill_cublas: bool = True
    prefill_chunk: int = 4096
    log_stats_interval_ms: int = 1000
    device_profile: str = "auto"
    host_kv_mib: int | None = None
    host_state_slots: int | None = None
    lora_gdn_projections: bool = False
    structured_output: bool = True

# These options configure the vLLM engine itself. NInfer uses the Worker-level
# context and concurrency limits but receives none of these engine arguments.
VLLM_ONLY_DEFAULTS = {
    "gpu_memory_utilization": 0.65,
    "kv_cache_dtype": "auto",
    "kv_cache_memory_bytes": None,
    "max_num_batched_tokens": 2048,
    "enable_prefix_caching": True,
    "language_model_only": True,
    "mamba_cache_mode": "align",
    "enforce_eager": True,
    "speculative_config": None,
}


def _ninfer_options(value: Any) -> NInferOptions:
    if value is None:
        options = NInferOptions()
    elif isinstance(value, NInferOptions):
        options = value
    elif isinstance(value, dict):
        allowed = {item.name for item in fields(NInferOptions)}
        extra = sorted(set(value) - allowed)
        if extra:
            raise ValueError(f"NInfer ninfer_options contains unsupported field(s): {extra}")
        try:
            options = NInferOptions(**value)
        except TypeError as exc:
            raise ValueError("NInfer ninfer_options must match its typed contract") from exc
    else:
        raise ValueError("NInfer ninfer_options must be an object or null")

    if type(options.structured_output) is not bool:
        raise ValueError("NInfer structured_output must be a boolean")
    if not isinstance(options.kv_capacity, str) or not (
        options.kv_capacity == "auto" or options.kv_capacity.isdecimal()
        and int(options.kv_capacity) > 0
    ):
        raise ValueError("NInfer ninfer_options.kv_capacity must be 'auto' or a positive integer string")
    if (isinstance(options.cuda_graph_allowance_mib, bool)
            or not isinstance(options.cuda_graph_allowance_mib, int)
            or options.cuda_graph_allowance_mib < 0):
        raise ValueError("NInfer ninfer_options.cuda_graph_allowance_mib must be a nonnegative integer")
    if options.host_kv_mib is not None and (type(options.host_kv_mib) is not int or options.host_kv_mib < 0):
        raise ValueError("NInfer ninfer_options.host_kv_mib must be a nonnegative integer when set")
    if (options.host_state_slots is not None
            and (type(options.host_state_slots) is not int or options.host_state_slots < 0)):
        raise ValueError("NInfer ninfer_options.host_state_slots must be a nonnegative integer when set")
    if not isinstance(options.spec, str) or options.spec not in {"none", "mtp"}:
        raise ValueError("NInfer ninfer_options.spec must be 'none' or 'mtp'")
    if options.spec == "none":
        if options.draft_tokens != 0 or options.ngram_draft_tokens != 0:
            raise ValueError("NInfer no-spec mode requires draft_tokens=0 and ngram_draft_tokens=0")
    else:
        if (isinstance(options.draft_tokens, bool) or not isinstance(options.draft_tokens, int)
                or not 1 <= options.draft_tokens <= 15):
            raise ValueError("NInfer ninfer_options.draft_tokens must be an integer in [1, 15]")
        if (isinstance(options.ngram_draft_tokens, bool)
                or not isinstance(options.ngram_draft_tokens, int)
                or not 0 <= options.ngram_draft_tokens <= 15):
            raise ValueError("NInfer ninfer_options.ngram_draft_tokens must be an integer in [0, 15]")
    if not isinstance(options.kv_dtype, str) or not options.kv_dtype.strip():
        raise ValueError("NInfer ninfer_options.kv_dtype must be a non-empty string")
    if type(options.prefill_cublas) is not bool:
        raise ValueError("NInfer ninfer_options.prefill_cublas must be a boolean")
    for name, option_value in (("prefill_chunk", options.prefill_chunk),
                                ("log_stats_interval_ms", options.log_stats_interval_ms)):
        if isinstance(option_value, bool) or not isinstance(option_value, int) or option_value < 1:
            raise ValueError(f"NInfer ninfer_options.{name} must be a positive integer")
    if not isinstance(options.device_profile, str) or not options.device_profile.strip():
        raise ValueError("NInfer ninfer_options.device_profile must be a non-empty string")
    if type(options.lora_gdn_projections) is not bool:
        raise ValueError("NInfer ninfer_options.lora_gdn_projections must be a boolean")
    return options


def ninfer_engine_config_arguments(inference: dict[str, Any]) -> list[str]:
    """Map the typed NInfer defaults to native engine flags."""
    if inference.get("speculative_config") is not None:
        raise ValueError("NInfer requires speculative_config=null; use ninfer_options for native MTP settings")
    options = _ninfer_options(inference.get("ninfer_options"))
    if options.lora_gdn_projections and inference.get("quantization") != "ternary":
        raise ValueError("NInfer GDN LoRA projections require quantization=ternary")
    result = [
        "--kv-capacity", options.kv_capacity,
        "--cuda-graph-allowance-mib", str(options.cuda_graph_allowance_mib),
    ]
    if options.spec == "mtp":
        result.extend(("--spec", "mtp", "--draft-tokens", str(options.draft_tokens),
                       "--ngram-draft-tokens", str(options.ngram_draft_tokens)))
    if options.structured_output:
        result.append("--structured-output")
    if options.lora_gdn_projections:
        result.append("--rl-lora-gdn-projections")
    result.extend(("--kv-dtype", options.kv_dtype))
    if options.host_kv_mib is not None:
        result.extend(("--host-kv-mib", str(options.host_kv_mib)))
    if options.host_state_slots is not None:
        result.extend(("--host-state-slots", str(options.host_state_slots)))
    if options.prefill_cublas:
        result.append("--prefill-cublas")
    result.extend((
        "--prefill-chunk", str(options.prefill_chunk),
        "--log-stats-interval-ms", str(options.log_stats_interval_ms),
        "--device-profile", options.device_profile,
    ))
    return result


def validate_backend_config(inference: dict[str, Any]) -> str:
    """Validate backend-specific fields without importing either engine."""
    if not isinstance(inference, dict):
        raise ValueError("worker inference config must be an object")
    backend = inference.get("backend", "vllm")
    if not isinstance(backend, str) or backend not in {"vllm", "ninfer"}:
        raise ValueError("worker_inference.backend must be vllm or ninfer")
    quantization = inference.get("quantization")
    if not isinstance(quantization, str) or quantization not in {"torchao", "ternary"}:
        raise ValueError("worker_inference.quantization must explicitly be torchao or ternary")

    if backend == "vllm":
        if quantization == "ternary":
            raise ValueError("ternary Worker artifacts require inference.backend=ninfer")
        configured = sorted(name for name in NINFER_CONFIG_FIELDS if inference.get(name) is not None)
        if configured:
            raise ValueError(f"vLLM backend does not accept NInfer field(s): {configured}")
        return backend

    generation_timeout = inference.get("ninfer_generation_timeout_seconds")
    if generation_timeout is not None and (
            isinstance(generation_timeout, bool)
            or not isinstance(generation_timeout, (int, float))
            or not math.isfinite(generation_timeout) or generation_timeout <= 0):
        raise ValueError("worker_inference.ninfer_generation_timeout_seconds must be finite and positive")

    if quantization != "ternary":
        raise ValueError("NInfer Worker supports only explicit T2 artifacts")

    # NInfer has one native engine contract. Its shared vLLM speculative field
    # must remain null so the engine cannot receive two competing settings.
    ninfer_engine_config_arguments(inference)
    for name, default in VLLM_ONLY_DEFAULTS.items():
        if inference.get(name, default) != default:
            raise ValueError(f"NInfer backend does not accept vLLM-specific worker_inference.{name}")

    concurrency = inference.get("max_num_seqs", 4)
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or not 1 <= concurrency <= 8:
        raise ValueError("NInfer worker_inference.max_num_seqs must be an integer between 1 and 8")

    converter_fields = {"ninfer_converter_python", "ninfer_converter_path"}
    required = (NINFER_FIELDS - converter_fields) | {"ninfer_artifact_sha256"}
    missing = sorted(name for name in required
                     if not isinstance(inference.get(name), str) or not inference[name].strip())
    if missing:
        raise ValueError(f"NInfer backend requires non-empty worker_inference field(s): {missing}")

    endpoint = urlsplit(inference["ninfer_url"])
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
            or endpoint.path not in ("", "/")):
        raise ValueError("worker_inference.ninfer_url must be an HTTP(S) engine origin without credentials, path, query, or fragment")
    path_fields = ["ninfer_artifact_path", "ninfer_conversion_report"]
    if quantization != "ternary":
        path_fields.extend(("ninfer_converter_python", "ninfer_converter_path"))
    for name in path_fields:
        if not Path(inference[name]).is_absolute():
            raise ValueError(f"worker_inference.{name} must be an absolute path")
    if Path(inference["ninfer_artifact_path"]).suffix != ".ninfer":
        raise ValueError("worker_inference.ninfer_artifact_path must name a .ninfer artifact")
    if Path(inference["ninfer_conversion_report"]).suffix != ".json":
        raise ValueError("worker_inference.ninfer_conversion_report must name a JSON report")
    if quantization == "ternary" and inference.get("ninfer_converter_path") is not None:
        converter_name = Path(inference["ninfer_converter_path"]).name
        if converter_name != "qwen3_5_bonsai2_t2_lora.py":
            raise ValueError("ternary NInfer requires qwen3_5_bonsai2_t2_lora.py when a converter is configured")
    if quantization != "ternary":
        converter_name = Path(inference["ninfer_converter_path"]).name
        if converter_name == "qwen3_5_bonsai2_t2_lora.py":
            raise ValueError("qwen3_5_bonsai2_t2_lora.py requires quantization=ternary")
    return backend


def load_worker_backend(
    inference: dict[str, Any],
    *,
    lora_rank: int,
    lora_alpha: int,
) -> Any:
    """Load only the selected backend and pass it only its supported options."""
    try:
        backend = validate_backend_config(inference)
    except ValueError as exc:
        raise BackendError(str(exc)) from exc

    options = dict(inference)
    options.pop("backend", None)
    if backend == "vllm":
        options.pop("ninfer_options", None)
        options.pop("ninfer_generation_timeout_seconds", None)
        for name in NINFER_FIELDS:
            options.pop(name, None)
        from .vllm_backend import load_vllm_backend

        loader = load_vllm_backend
    else:
        for name in VLLM_ONLY_DEFAULTS:
            options.pop(name, None)
        # Preserve the stable per-Worker seed. The selected backend combines
        # it with request identity when a request does not supply its own seed.
        from .ninfer_backend import load_ninfer_backend

        loader = load_ninfer_backend
    return loader(options, lora_rank=lora_rank, lora_alpha=lora_alpha)
