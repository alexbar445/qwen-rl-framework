from __future__ import annotations

import subprocess
import sys

import pytest

from qwen_rl.contracts import worker as wire


def test_worker_wire_contract_is_cpu_only_and_not_reexported_by_app() -> None:
    code = """
import sys
from qwen_rl.contracts import worker as wire
assert wire.MAX_REQUEST_SEED == (1 << 32) - 1
assert wire.MAX_CONTEXT_TOKENS == 131072
assert wire.MAX_GENERATED_TOKENS == 65536
assert wire.ADAPTER_FILES == {"adapter_config.json", "adapter_model.safetensors"}
assert "torch" not in sys.modules
assert "verl" not in sys.modules
assert "transformers" not in sys.modules
from qwen_rl.worker import limits
assert not hasattr(limits, "MAX_CONTEXT_TOKENS")
assert not hasattr(limits, "MAX_GENERATED_TOKENS")
assert limits.wire is wire
import qwen_rl.worker.app as app
for name in (
    "StrictModel", "GenerateRequest", "PolicyRequest", "EpisodeLeaseRequest",
    "PolicyFile", "PolicyManifest", "ADAPTER_FILES", "MAX_REQUEST_SEED",
    "SAMPLING_CONSTANTS", "NEUTRALIZED_SAMPLING_FIELDS",
):
    assert not hasattr(app, name), name
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_client_and_server_share_sampling_acceptance_and_preserve_http_422_mapping():
    from qwen_rl.worker.app import WorkerFailure, _normalize_sampling_params
    from qwen_rl.worker.client import _safe_sampling_params

    valid = (
        {"max_tokens": 1, "logprobs": True},
        {"max_tokens": 64, "logprobs": True, "temperature": 1.0, "top_k": 0},
    )
    for params in valid:
        assert _safe_sampling_params(params) == _normalize_sampling_params(
            params, limit=wire.MAX_GENERATED_TOKENS,
        )[1]

    invalid = (
        {"max_tokens": True, "logprobs": True},
        {"max_tokens": 4, "logprobs": False},
        {"max_tokens": 4, "logprobs": True, "top_p": 0.9},
        {"max_tokens": 4, "logprobs": True, "min_p": 0.1},
    )
    for params in invalid:
        with pytest.raises(ValueError):
            _safe_sampling_params(params)
        with pytest.raises(WorkerFailure) as failure:
            _normalize_sampling_params(params, limit=wire.MAX_GENERATED_TOKENS)
        assert failure.value.status_code == 422
        assert failure.value.code == "invalid_sampling_params"


def test_shared_generation_response_rejects_alignment_and_nonfinite_logprobs():
    valid = {
        "token_ids": [1, 2],
        "log_probs": [-0.1, -0.2],
        "stop_reason": "eos",
        "extra_fields": {"episode_purpose": "evaluation", "sampling_config": {"seed": 7}},
        "worker_id": "worker-1",
        "timing": {"generation_seconds": 0.1},
    }
    assert wire.GenerationResponse.model_validate(valid).token_ids == [1, 2]
    with pytest.raises(ValueError, match="same length"):
        wire.GenerationResponse.model_validate({**valid, "log_probs": [-0.1]})
    with pytest.raises(ValueError, match="finite"):
        wire.GenerationResponse.model_validate({**valid, "log_probs": [-0.1, float("nan")]})


def test_server_keeps_the_actual_deployment_generation_limit():
    from qwen_rl.worker.app import WorkerFailure, _normalize_sampling_params
    from qwen_rl.worker.client import _safe_sampling_params

    request = {"max_tokens": 2048, "logprobs": True}
    assert _safe_sampling_params(request)["max_tokens"] == 2048
    with pytest.raises(WorkerFailure) as failure:
        _normalize_sampling_params(request, limit=1024)
    assert failure.value.status_code == 422
    assert _normalize_sampling_params(request, limit=4096)[0] == 2048


def test_startup_wire_requires_normalized_quantization_dtype():
    evidence = {
        "backend": "vllm",
        "base_fingerprint": "base",
        "artifact_fingerprint": "artifact",
        "quantization": {"mode": "torchao", "compute_dtype": "bfloat16"},
        "runtime_quantization": {"mode": "torchao"},
        "max_model_len": 4096,
        "max_generated_tokens": 1024,
        "max_num_seqs": 1,
        "max_loras": 1,
        "supports_full_token_logprobs": True,
        "behavior_distribution": "raw_policy",
        "base_trainable_parameters": 0,
        "backend_evidence": {},
    }

    assert wire.WorkerStartupEvidence.model_validate(evidence).quantization["compute_dtype"] == "bfloat16"
    with pytest.raises(ValueError, match="canonical bfloat16"):
        wire.WorkerStartupEvidence.model_validate({
            **evidence,
            "quantization": {"mode": "torchao", "compute_dtype": "torch.bfloat16"},
        })


@pytest.mark.parametrize("boundary", [-1, True, 1.5, 3, 65536])
def test_shared_prefix_cannot_escape_raw_prompt(boundary):
    with pytest.raises(ValueError):
        wire.GenerateRequest.model_validate({
            "policy_version": 1, "episode_id": "e", "adapter_hash": "a" * 64,
            "request_id": "r", "prompt_ids": [1, 2, 3],
            "shared_prefix_tokens": boundary, "sampling_params": {},
        })


def test_shared_prefix_is_a_boundary_hint_and_preserves_prompt_ids():
    r = wire.GenerateRequest.model_validate({
        "policy_version": 1, "episode_id": "e", "adapter_hash": "a" * 64,
        "request_id": "r", "prompt_ids": [1, 2, 3],
        "shared_prefix_tokens": 2, "sampling_params": {},
    })
    assert r.prompt_ids == [1, 2, 3] and r.shared_prefix_tokens == 2


def test_128k_wire_prompt_boundary_keeps_generation_ceiling_independent():
    payload = {'policy_version': 1, 'episode_id': 'e', 'adapter_hash': 'a' * 64,
               'request_id': 'r', 'prompt_ids': [1] * 131072, 'sampling_params': {}}
    assert len(wire.GenerateRequest.model_validate(payload).prompt_ids) == 131072
    payload['prompt_ids'].append(1)
    with pytest.raises(ValueError):
        wire.GenerateRequest.model_validate(payload)
    assert wire.MAX_GENERATED_TOKENS == 65536
