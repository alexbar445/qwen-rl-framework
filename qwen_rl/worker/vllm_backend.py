"""vLLM inference and adapter bridge for one quantized worker model."""
from __future__ import annotations

import math
import os
from uuid import uuid4
from pathlib import Path
from typing import Any, Callable

from ..contracts import worker as wire
from ..models.quantization import quantization_evidence
from .backend import BackendError
from .generation import GenerationResult, TOOL_CALL_STOP_STRING
from .metrics import KVCacheMetricsObserver
from .native_artifact import (
    WorkerArtifactError,
    artifact_lineage,
    read_worker_artifact,
    validate_policy_quantization as validate_policy_artifact_quantization,
)


SOURCE_FINGERPRINT_FILE = "source_fingerprint.json"
_NVFP4_QUANTIZER = "NVFP4WeightOnlyConfig"


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        return None
    return result


def _timestamp_delta(metrics: Any, start_field: str, end_field: str) -> float | None:
    """Derive an interval only from two present engine-monotonic timestamps."""
    start = getattr(metrics, start_field, None)
    end = getattr(metrics, end_field, None)
    if (isinstance(start, bool) or not isinstance(start, (int, float))
            or isinstance(end, bool) or not isinstance(end, (int, float))):
        return None
    start_value = float(start)
    end_value = float(end)
    # vLLM uses 0.0 for absent RequestStateStats timestamps. These are monotonic
    # engine timestamps, so require positive anchors and never mix in arrival_time.
    if (not math.isfinite(start_value) or not math.isfinite(end_value)
            or start_value <= 0.0 or end_value <= 0.0 or end_value < start_value):
        return None
    return end_value - start_value


def _generation_metrics(
    *,
    prompt_ids: list[int],
    token_ids: list[int],
    request_output: Any,
    finished_stats: Any,
    finished_stats_observer_available: bool,
    finished_stats_error: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe receipt from public vLLM request telemetry."""
    prompt_tokens = len(prompt_ids)
    generated_tokens = len(token_ids)
    raw_cached_tokens = getattr(request_output, "num_cached_tokens", None)
    cached_tokens: int | None = None
    if (isinstance(raw_cached_tokens, int) and not isinstance(raw_cached_tokens, bool)
            and 0 <= raw_cached_tokens <= prompt_tokens):
        cached_tokens = raw_cached_tokens

    request_metrics = getattr(request_output, "metrics", None)
    finished = finished_stats if isinstance(finished_stats, dict) else {}
    result: dict[str, Any] = {
        "schema": "qwen-rl-generation-metrics/v1",
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_available": True,
        "prompt_tokens_source": "request.prompt_token_ids",
        "generated_tokens": generated_tokens,
        "generated_tokens_available": True,
        "generated_tokens_source": "request_output.outputs[0].token_ids",
        "num_cached_tokens": cached_tokens,
        "num_cached_tokens_available": cached_tokens is not None,
        "num_cached_tokens_source": (
            "request_output.num_cached_tokens" if cached_tokens is not None else None
        ),
        "cache_hit_fraction": (
            cached_tokens / prompt_tokens
            if cached_tokens is not None and prompt_tokens > 0 else None
        ),
        "cache_hit_fraction_available": cached_tokens is not None and prompt_tokens > 0,
        "cache_hit_fraction_source": (
            "request_output.num_cached_tokens/request.prompt_token_ids"
            if cached_tokens is not None and prompt_tokens > 0 else None
        ),
        "request_output_metrics_count": 1 if request_metrics is not None else 0,
        "finished_request_stats_count": 1 if isinstance(finished_stats, dict) else 0,
        "finished_request_stats_observer_available": finished_stats_observer_available,
        "finished_request_stats_error": finished_stats_error,
    }

    # FinishedRequestStats are the primary phase source when vLLM's public
    # observer captured the request. The observer returns None for absent or
    # invalid timestamps rather than encoding them as zero.
    phase_specs = {
        "queue_seconds": ("queue_seconds", "queued_ts", "scheduled_ts"),
        "prefill_seconds": ("prefill_seconds", "scheduled_ts", "first_token_ts"),
        "decode_seconds": ("decode_seconds", "first_token_ts", "last_token_ts"),
        "inference_seconds": ("inference_seconds", "scheduled_ts", "last_token_ts"),
    }
    for name, (finished_key, start_field, end_field) in phase_specs.items():
        value = _finite_nonnegative(finished.get(finished_key))
        source = "vllm_finished_request_stats" if value is not None else None
        if value is None:
            value = _timestamp_delta(request_metrics, start_field, end_field)
            if value is not None:
                source = f"request_output.metrics.{start_field}/{end_field}"
        result[name] = value
        result[f"{name}_available"] = value is not None
        result[f"{name}_source"] = source

    ttft = _finite_nonnegative(getattr(request_metrics, "first_token_latency", None))
    # RequestStateStats initializes first_token_latency to 0.0 before a first
    # token exists, so zero is unknown here rather than a measured latency.
    if ttft == 0.0:
        ttft = None
    result["ttft_seconds"] = ttft
    result["ttft_seconds_available"] = ttft is not None
    result["ttft_seconds_source"] = (
        "request_output.metrics.first_token_latency" if ttft is not None else None
    )
    return result


def _normalize_quantization(value: Any) -> str:
    if value != "torchao":
        raise BackendError("vLLM Worker supports only explicit torchao NVFP4 artifacts")
    return value


class VLLMBackend:
    """Thin async boundary around the pinned vLLM AsyncLLM API."""

    def __init__(
        self,
        engine: Any,
        *,
        sampling_params_type: type,
        lora_request_type: type,
        evidence: dict[str, Any],
        resource_observer: KVCacheMetricsObserver | None = None,
    ) -> None:
        self.engine = engine
        self.sampling_params_type = sampling_params_type
        self.lora_request_type = lora_request_type
        self.evidence = evidence
        self.resource_observer = resource_observer
        quantization = quantization_evidence(evidence["quantization"], expected_mode="torchao")
        source_fingerprint = evidence["source_fingerprint"]
        self.startup_evidence = wire.WorkerStartupEvidence(
            backend="vllm",
            base_fingerprint=evidence["base_fingerprint"],
            artifact_fingerprint=evidence["artifact_fingerprint"],
            source_artifact_fingerprint=source_fingerprint.get("artifact_fingerprint"),
            quantization=quantization,
            runtime_quantization=evidence["runtime_quantization"],
            max_model_len=evidence["max_model_len"],
            max_generated_tokens=evidence["max_model_len"],
            max_num_seqs=evidence["max_num_seqs"],
            max_loras=evidence["max_loras"],
            supports_full_token_logprobs=True,
            behavior_distribution="raw_policy",
            base_trainable_parameters=evidence["base_trainable_parameters"],
            backend_evidence=evidence,
        )

    def validate_policy_quantization(self, policy_quantization: dict[str, Any]) -> None:
        validate_policy_artifact_quantization(
            "torchao", self.startup_evidence.quantization, policy_quantization,
        )

    def resource_snapshot(self) -> dict[str, Any] | None:
        if self.resource_observer is None:
            return None
        return self.resource_observer.snapshot().as_dict()

    async def log_stats(self) -> None:
        """Flush the public vLLM stat loggers for the worker lifecycle loop."""
        do_log_stats = getattr(self.engine, "do_log_stats", None)
        if not callable(do_log_stats):
            raise BackendError("vLLM AsyncLLM does not expose public do_log_stats")
        try:
            await do_log_stats()
        except Exception as exc:
            raise BackendError("vLLM failed to log worker statistics") from exc

    def _take_finished_request_metrics(
        self,
        engine_request_id: str,
    ) -> tuple[dict[str, Any] | None, bool, str | None]:
        observer = self.resource_observer
        take_metrics = getattr(observer, "take_finished_request_metrics", None)
        if not callable(take_metrics):
            return None, False, None
        try:
            metrics = take_metrics(engine_request_id)
        except Exception as exc:
            return None, True, type(exc).__name__
        return (metrics if isinstance(metrics, dict) else None), True, None

    async def add_adapter(self, version: int, path: Path, *, adapter_hash: str | None = None):
        request = self.lora_request_type(
            f"policy-v{version}",
            version + 1,
            str(path),
        )
        if await self.engine.add_lora(request) is not True:
            raise BackendError(f"vLLM refused policy adapter version {version}")
        try:
            ranks = await self.engine.collective_rpc(
                "inspect_lora_namespace", args=(request.lora_int_id,)
            )
            if not isinstance(ranks, list) or not ranks:
                raise BackendError("Worker returned no LoRA namespace evidence")
            for rank in ranks:
                matched = rank.get("matched_module_count") if isinstance(rank, dict) else None
                if (not isinstance(rank, dict) or rank.get("registered") is not True
                        or isinstance(matched, bool) or not isinstance(matched, int) or matched < 1):
                    raise BackendError("LoRA checkpoint matches no Worker model modules; verify its model namespace")
            self.last_adapter_namespace_evidence = ranks
        except Exception as exc:
            if await self.engine.remove_lora(request.lora_int_id) is not True:
                raise BackendError("Worker failed to remove a namespace-invalid adapter") from exc
            if isinstance(exc, BackendError):
                raise
            raise BackendError("Worker could not validate the adapter namespace") from exc
        return request

    async def remove_adapter(self, request: Any) -> None:
        if request is None:
            return
        if await self.engine.remove_lora(request.lora_int_id) is not True:
            raise BackendError(f"vLLM refused to remove policy adapter {request.lora_name}")

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        max_tokens: int,
        sampling_params: dict[str, Any],
        lora_request: Any,
        token_count_callback: Callable[[int], None] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        if sampling_params.get("constraint") is not None:
            raise BackendError("grammar-conditioned ARC policy requires the NInfer backend; vLLM raw policy remains supported")
        sampling_kwargs = dict(
            n=1,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            top_k=sampling_params["top_k"],
            repetition_penalty=sampling_params["repetition_penalty"],
            presence_penalty=0.0,
            frequency_penalty=0.0,
            min_p=0.0,
            max_tokens=max_tokens,
            logprobs=1,
            stop=[TOOL_CALL_STOP_STRING],
            include_stop_str_in_output=True,
            skip_special_tokens=False,
            ignore_eos=False,
        )
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (1 << 32) - 1:
                raise BackendError("request seed must be an integer between 0 and 2**32-1")
            sampling_kwargs["seed"] = seed
        params = self.sampling_params_type(**sampling_kwargs)
        response = None
        # vLLM request IDs must be unique while concurrent streams are active.
        engine_request_id = f"{request_id[:120]}:{uuid4().hex}"
        try:
            async for result in self.engine.generate(
                {"prompt_token_ids": list(prompt_ids)},
                params,
                engine_request_id,
                lora_request=lora_request,
            ):
                if token_count_callback is not None:
                    # SamplingParams defaults to cumulative RequestOutput values in
                    # the pinned vLLM release, so this is a live per-request total.
                    outputs = getattr(result, "outputs", None)
                    if isinstance(outputs, list) and outputs:
                        token_ids = getattr(outputs[0], "token_ids", None)
                        if isinstance(token_ids, (list, tuple)):
                            token_count_callback(len(token_ids))
                response = result
        except Exception as exc:
            # A failed request must not leave a terminal stats receipt buffered
            # in the observer for a later request.
            self._take_finished_request_metrics(engine_request_id)
            raise BackendError("vLLM async generation failed") from exc
        finished_stats, finished_stats_observer_available, finished_stats_error = (
            self._take_finished_request_metrics(engine_request_id)
        )
        if response is None or not getattr(response, "finished", True):
            raise BackendError("vLLM did not return a final request output")
        returned_prompt = getattr(response, "prompt_token_ids", None)
        if returned_prompt is None or list(returned_prompt) != prompt_ids:
            raise BackendError("vLLM changed the caller-provided prompt token IDs")
        completions = getattr(response, "outputs", None)
        if not isinstance(completions, list) or len(completions) != 1:
            raise BackendError("vLLM returned a different number of completions than requested")
        completion = completions[0]
        token_ids = [int(value) for value in completion.token_ids]
        if not token_ids:
            raise BackendError("vLLM returned no sampled tokens")
        token_logprobs = completion.logprobs
        if token_logprobs is None or len(token_logprobs) != len(token_ids):
            raise BackendError("vLLM sampled-token logprobs do not align with token IDs")
        log_probs: list[float] = []
        for token_id, candidates in zip(token_ids, token_logprobs):
            selected = candidates.get(token_id) if candidates is not None else None
            logprob = getattr(selected, "logprob", None)
            if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
                raise BackendError(f"vLLM omitted the sampled token logprob for token {token_id}")
            value = float(logprob)
            if not math.isfinite(value):
                raise BackendError("vLLM returned a non-finite sampled-token logprob")
            log_probs.append(value)
        stop_reason = getattr(completion, "stop_reason", None)
        finish_reason = getattr(completion, "finish_reason", None)
        if stop_reason == TOOL_CALL_STOP_STRING:
            reason = "tool_call"
        elif finish_reason == "length":
            reason = "length"
        elif finish_reason == "stop":
            reason = "eos"
        else:
            raise BackendError(f"vLLM returned an unsupported finish reason: {finish_reason!r}")
        return GenerationResult(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=reason,
            metrics=_generation_metrics(
                prompt_ids=prompt_ids,
                token_ids=token_ids,
                request_output=response,
                finished_stats=finished_stats,
                finished_stats_observer_available=finished_stats_observer_available,
                finished_stats_error=finished_stats_error,
            ),
        )

    def close(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if callable(shutdown):
            shutdown(timeout=10)


def _loaded_quantized_modules(model: Any) -> dict[str, Any]:
    """Prove serialized NVFP4 modules and packed qdata in every vLLM model worker."""
    module_count = 0
    packed_uint8_tensor_count = 0
    packed_uint8_numel = 0
    runtime_skip_modules: list[str] | None = None
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if method is None:
            continue
        if type(method).__name__ == "UnquantizedLinearMethod":
            continue
        quant_config = getattr(method, "quant_config", None)
        torchao_config = getattr(quant_config, "torchao_config", None)
        if type(torchao_config).__name__ != _NVFP4_QUANTIZER:
            raise BackendError("vLLM Worker loaded a non-NVFP4 quantization method")
        if getattr(quant_config, "is_checkpoint_torchao_serialized", None) is not True:
            raise BackendError("vLLM TorchAO NVFP4 method is not loading a serialized checkpoint")
        weight = getattr(module, "weight", None)
        weight_data = getattr(weight, "data", weight)
        qdata = getattr(weight_data, "qdata", None)
        if qdata is None or type(weight_data).__name__ != "NVFP4Tensor":
            raise BackendError("vLLM TorchAO NVFP4 module has no loaded NVFP4Tensor qdata")
        if str(getattr(qdata, "dtype", None)) != "torch.uint8" or int(qdata.numel()) <= 0:
            raise BackendError("vLLM TorchAO NVFP4 qdata is not non-empty packed uint8")
        skips = getattr(quant_config, "skip_modules", None)
        if not isinstance(skips, list):
            raise BackendError("vLLM TorchAO config has no modules_to_not_convert")
        if runtime_skip_modules is None:
            runtime_skip_modules = list(skips)
        elif runtime_skip_modules != skips:
            raise BackendError("vLLM TorchAO modules disagree on skip metadata")
        module_count += 1
        packed_uint8_tensor_count += 1
        packed_uint8_numel += int(qdata.numel())

    trainable_parameters = sum(
        int(parameter.numel()) for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "module_count": module_count,
        "quantizer_counts": {_NVFP4_QUANTIZER: module_count},
        "base_trainable_parameters": trainable_parameters,
        "packed_uint8_tensor_count": packed_uint8_tensor_count,
        "packed_uint8_numel": packed_uint8_numel,
        "runtime_modules_to_not_convert": runtime_skip_modules,
    }


class QuantizationInspector:
    """Expose a named local vLLM RPC; no callable/pickle transport is needed."""

    def inspect_lora_namespace(self, adapter_id: int) -> dict[str, Any]:
        """Check loaded adapter tensors against real module names before ACKing it.

        An all-zero policy-0 delta remains valid when its A/B tensors have real
        destinations. A checkpoint with no matching destinations is rejected.
        """
        import torch

        manager = self.model_runner.lora_manager._adapter_manager
        adapter = manager.get_adapter(adapter_id)
        if adapter is None:
            return {"registered": False, "matched_module_count": 0}

        def present(value: Any) -> bool:
            if isinstance(value, torch.Tensor):
                return value.numel() > 0
            if isinstance(value, (list, tuple)):
                return any(present(item) for item in value)
            return False

        matched = []
        for name in manager.modules:
            weights = adapter.get_lora(name)
            if (weights is not None and present(weights.lora_a) and present(weights.lora_b)):
                matched.append(name)
        return {"registered": True, "adapter_id": adapter_id,
                "matched_module_count": len(matched), "model_module_count": len(manager.modules),
                "matched_module_names_preview": matched[:4]}

    def inspect_quantization(self) -> dict[str, Any]:
        model = self.model_runner.model
        # vLLM inference mode does not require all parameter flags to be false.
        # This Worker never trains, so freeze them explicitly before serving.
        model.requires_grad_(False)
        return _loaded_quantized_modules(model)


async def load_vllm_backend(
    inference: dict[str, Any],
    *,
    lora_rank: int,
    lora_alpha: int,
) -> VLLMBackend:
    """Load one explicit NVFP4 model and prove its runtime state through AsyncLLM."""
    if not isinstance(inference, dict):
        raise BackendError("worker inference config must be an object")
    kwargs = dict(inference)
    model_path = kwargs.pop("model_path", None)
    if not isinstance(model_path, str) or not model_path:
        raise BackendError("worker inference config must provide model_path")
    requested_quantization = _normalize_quantization(kwargs.get("quantization"))
    configured_generated_limit = kwargs.pop("max_generated_tokens", 4_096)
    if (isinstance(configured_generated_limit, bool)
            or not isinstance(configured_generated_limit, int)
            or not 1 <= configured_generated_limit <= wire.MAX_GENERATED_TOKENS):
        raise BackendError("worker max_generated_tokens is outside the wire limit")
    if kwargs.get("enable_prefix_caching") is not True:
        raise BackendError("worker inference config must enable vLLM prefix KV caching")
    requested_speculation = kwargs.get("speculative_config")
    if requested_speculation is None:
        speculative_tokens = None
    elif isinstance(requested_speculation, dict):
        if requested_speculation.get("method") not in {"ngram", "mtp"}:
            raise BackendError("worker speculative_config method must be ngram, mtp, or null")
        speculative_tokens = requested_speculation.get("num_speculative_tokens")
        if (isinstance(speculative_tokens, bool)
                or not isinstance(speculative_tokens, int)
                or speculative_tokens < 1):
            raise BackendError("speculative_config.num_speculative_tokens must be a positive integer")
    else:
        raise BackendError("worker speculative_config must be an object or null")
    max_loras = kwargs.pop("max_loras", 2)
    if isinstance(max_loras, bool) or not isinstance(max_loras, int) or max_loras < 2:
        raise BackendError("worker inference max_loras must be at least 2 for atomic adapter replacement")
    requested_lora_rank = kwargs.pop("max_lora_rank", lora_rank)
    if (isinstance(requested_lora_rank, bool)
            or not isinstance(requested_lora_rank, int)
            or requested_lora_rank < lora_rank):
        raise BackendError("worker inference max_lora_rank must cover the published adapter rank")
    if kwargs.pop("enable_lora", True) is not True:
        raise BackendError("vLLM worker requires enable_lora=true")

    try:
        artifact = read_worker_artifact(model_path, requested_quantization)
    except WorkerArtifactError as exc:
        raise BackendError(str(exc)) from exc
    lineage = artifact_lineage(artifact)
    model_config = artifact.config

    # This Worker process owns one AsyncLLM instance.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.lora.request import LoRARequest

    resource_observer = KVCacheMetricsObserver()
    engine_args = AsyncEngineArgs(
        model=str(Path(model_path).expanduser().resolve()),
        worker_extension_cls="qwen_rl.worker.vllm_backend.QuantizationInspector",
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=requested_lora_rank,
        **kwargs,
    )
    engine = AsyncLLM.from_engine_args(
        engine_args,
        stat_loggers=[resource_observer.stat_logger_factory()],
    )
    try:
        effective_method = getattr(engine.model_config, "quantization", None)
        if _normalize_quantization(effective_method) != artifact.mode:
            raise BackendError("vLLM runtime did not retain the configured NVFP4 quantization method")
        vllm_config = engine.vllm_config
        prefix_cache = bool(vllm_config.cache_config.enable_prefix_caching)
        if not prefix_cache:
            raise BackendError("vLLM runtime disabled prefix KV caching")
        runtime_skip_modules = getattr(
            getattr(vllm_config, "quant_config", None), "skip_modules", None,
        )
        expected_skips = artifact.quantization["modules_to_not_convert"]
        if runtime_skip_modules != expected_skips:
            raise BackendError(
                "vLLM runtime modules_to_not_convert differs from the validated NVFP4 artifact"
            )

        quantizer_results = await engine.collective_rpc("inspect_quantization")
        if not isinstance(quantizer_results, list) or not quantizer_results:
            raise BackendError("vLLM did not return runtime NVFP4 evidence")
        module_count = 0
        trainable_parameters = 0
        runtime_packed_uint8_tensor_count = 0
        runtime_packed_uint8_numel = 0
        inspector_skip_modules: list[str] | None = None
        for item in quantizer_results:
            if not isinstance(item, dict):
                raise BackendError("vLLM runtime quantizer evidence is malformed")
            local_count = item.get("module_count")
            counts = item.get("quantizer_counts")
            packed_count = item.get("packed_uint8_tensor_count")
            packed_numel = item.get("packed_uint8_numel")
            trainable = item.get("base_trainable_parameters")
            if (isinstance(local_count, bool) or not isinstance(local_count, int) or local_count <= 0
                    or not isinstance(counts, dict)
                    or counts.get(_NVFP4_QUANTIZER) != local_count
                    or set(counts) != {_NVFP4_QUANTIZER}
                    or isinstance(packed_count, bool) or not isinstance(packed_count, int)
                    or packed_count != local_count
                    or isinstance(packed_numel, bool) or not isinstance(packed_numel, int)
                    or packed_numel <= 0
                    or isinstance(trainable, bool) or not isinstance(trainable, int)
                    or trainable < 0):
                raise BackendError("vLLM runtime NVFP4 evidence is incomplete or inconsistent")
            module_count += local_count
            runtime_packed_uint8_tensor_count += packed_count
            runtime_packed_uint8_numel += packed_numel
            trainable_parameters += trainable
            runtime_skips = item.get("runtime_modules_to_not_convert")
            if not isinstance(runtime_skips, list):
                raise BackendError("vLLM runtime skip metadata is missing")
            if inspector_skip_modules is None:
                inspector_skip_modules = list(runtime_skips)
            elif inspector_skip_modules != runtime_skips:
                raise BackendError("vLLM workers disagree on NVFP4 skip metadata")
        if trainable_parameters != 0:
            raise BackendError("vLLM loaded trainable base-model parameters")
        if inspector_skip_modules != expected_skips:
            raise BackendError("vLLM model workers did not retain the NVFP4 skip recipe")

        speculative = getattr(vllm_config, "speculative_config", None)
        if requested_speculation is None:
            if speculative is not None:
                raise BackendError("vLLM runtime enabled speculative decoding when config requested null")
            speculative_evidence = None
        else:
            speculative_evidence = {
                "method": getattr(speculative, "method", None),
                "num_speculative_tokens": getattr(speculative, "num_speculative_tokens", None),
            } if speculative is not None else None
            if (speculative_evidence is None
                    or speculative_evidence["method"] != requested_speculation["method"]
                    or speculative_evidence["num_speculative_tokens"] != speculative_tokens):
                raise BackendError("vLLM runtime did not retain the configured speculative decoder")

        max_model_len = engine.model_config.max_model_len
        max_num_seqs = engine.scheduler_config.max_num_seqs
        if (isinstance(max_model_len, bool) or not isinstance(max_model_len, int)
                or max_model_len <= 0 or configured_generated_limit > max_model_len):
            raise BackendError("vLLM context capacity is below the Worker generation limit")
        if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs < 1:
            raise BackendError("vLLM runtime reported an invalid generation concurrency")

        runtime_quantization = {
            "mode": "torchao",
            "runtime_method": effective_method,
            "packed_module_count": module_count,
            "packed_uint8_tensor_count": runtime_packed_uint8_tensor_count,
            "packed_uint8_numel": runtime_packed_uint8_numel,
        }
        quantization_evidence = {
            **artifact.quantization,
            "dtype": str(kwargs.get("dtype", "auto")),
            "quantized_weight_count": lineage["quantized_weight_count"],
            "quantized_module_count": module_count,
        }
        evidence = {
            "backend": "vllm",
            "sampling_backend": "torch",
            "seed": kwargs.get("seed", 0),
            "model_path": lineage["model_path"],
            "base_fingerprint": lineage["base_fingerprint"],
            "artifact_fingerprint": lineage["artifact_fingerprint"],
            "source_fingerprint": lineage,
            "quantization": quantization_evidence,
            "runtime_quantization": runtime_quantization,
            "is_loaded_in_4bit": True,
            "module_count": module_count,
            "packed_module_count": module_count,
            "packed_uint8_tensor_count": lineage.get("packed_uint8_tensor_count", 0),
            "packed_uint8_numel": lineage.get("packed_uint8_numel", 0),
            "kv_cache_enabled": True,
            "max_model_len": max_model_len,
            "max_generated_tokens": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_loras": max_loras,
            "prefix_caching_enabled": prefix_cache,
            "speculative_decoding": speculative_evidence,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "base_trainable_parameters": trainable_parameters,
            "supports_full_token_logprobs": True,
            "behavior_distribution": "raw_policy",
            "logical_model_vocab_size": None,
            "public_token_count": None,
            "model_config_quantization": model_config["quantization_config"],
        }
        return VLLMBackend(
            engine,
            sampling_params_type=SamplingParams,
            lora_request_type=LoRARequest,
            evidence=evidence,
            resource_observer=resource_observer,
        )
    except Exception:
        engine.shutdown(timeout=10)
        raise
