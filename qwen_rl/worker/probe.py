"""Check the real vLLM Worker HTTP ABI with saved Learner adapters; no speed benchmark."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from qwen_rl.runtime.config import load_config
from qwen_rl.worker.app import create_worker_app
from qwen_rl.contracts import worker as wire
from qwen_rl.worker.client import HTTPGenerationManager


async def run_probe(config_path: str, method: str, adapters: list[Path], output: Path) -> dict:
    from transformers import AutoTokenizer
    from qwen_rl.learner.app import _identity_course_messages, _identity_fixture

    config = load_config(config_path)
    inference = asdict(config.worker_inference)
    inference["speculative_config"] = (
        {"method": "mtp", "num_speculative_tokens": 1} if method == "mtp" else
        {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_min": 2, "prompt_lookup_max": 5}
    )
    from .native_artifact import artifact_lineage, read_worker_artifact
    artifact = read_worker_artifact(Path(inference["model_path"]), inference["quantization"])
    lineage = artifact_lineage(artifact)
    adapter_files = []
    for directory in adapters:
        adapter_files.append({name: (directory / name).read_bytes() for name in wire.ADAPTER_FILES})
    learner_version = 0

    def learner_handler(request):
        if request.url.path == "/policy":
            files = [{"name": name, "sha256": hashlib.sha256(data).hexdigest(),
                      "url": f"/policy/{learner_version}/{name}"}
                     for name, data in adapter_files[learner_version].items()]
            manifest = {
                "version": learner_version, "base_fingerprint": lineage["base_fingerprint"],
                "artifact_fingerprint": lineage["artifact_fingerprint"],
                "quantization": artifact.quantization,
                "files": files,
            }
            wire.PolicyManifest.model_validate(manifest)
            return httpx.Response(200, json=manifest)
        parts = request.url.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "policy":
            return httpx.Response(200, content=adapter_files[int(parts[1])][parts[2]])
        return httpx.Response(404)

    output.parent.mkdir(parents=True, exist_ok=True)
    # This diagnostic token is local to the in-process HTTP fixture, never an external service credential.
    os.environ["ARC_ENV_TOKEN"] = "vllm-worker-abi-diagnostic"
    app = create_worker_app({
        "worker_id": "worker-probe", "inference": inference,
        "learner_url": "http://learner.invalid", "adapter_cache": str(output.parent / "adapter-cache" / method),
        "telemetry_dir": str(output.parent / "telemetry" / method),
        "lora_rank": config.lora.rank, "lora_alpha": config.lora.alpha,
    }, http_transport=httpx.MockTransport(learner_handler))
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    messages = _identity_course_messages(_identity_fixture())
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=True, return_dict=False)
    assert isinstance(prompt_ids, list) and prompt_ids and all(type(token) is int for token in prompt_ids)
    result = {
        "status": "running", "diagnostic_only": True, "method": method,
        "scope": "real CUDA vLLM and saved real PEFT adapters; in-process Worker HTTP and Learner manifest fixture; no Learner optimizer run or throughput benchmark",
        "source_model_fingerprint": lineage["base_fingerprint"], "prompt_length": len(prompt_ids),
        "adapter_sources": [str(p) for p in adapters], "versions": [],
    }
    async with app.router.lifespan_context(app):
        backend = app.state.runtime.backend
        raw_responses = []
        original_generate = backend.engine.generate

        async def captured_generate(*args, **kwargs):
            async for response in original_generate(*args, **kwargs):
                raw_responses.append(response)
                yield response

        backend.engine.generate = captured_generate
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://worker.invalid/",
            headers={"Authorization": "Bearer vllm-worker-abi-diagnostic"}, timeout=300,
        ) as client:
            for version in range(len(adapters)):
                learner_version = version
                ack = await client.post("policy", json={"version": version})
                assert ack.status_code == 200, ack.text
                manager = HTTPGenerationManager("http://worker.invalid", policy_version=version, client=client)
                episode_id = f"worker-abi-{method}-v{version}"
                adapter_hash = ack.json()["adapter_hash"]
                await manager.acquire_episode(
                    episode_id=episode_id,
                    policy_version=version,
                    adapter_hash=adapter_hash,
                )
                try:
                    generated = await manager.generate(
                        request_id=episode_id, prompt_ids=prompt_ids,
                        sampling_params={"max_tokens": wire.MAX_GENERATED_TOKENS, "logprobs": True},
                        episode_id=episode_id,
                        adapter_hash=adapter_hash,
                    )
                finally:
                    await manager.release_episode(
                        episode_id=episode_id,
                        policy_version=version,
                        adapter_hash=adapter_hash,
                    )
                raw = raw_responses[-1]
                completion = raw.outputs[0]
                assert list(raw.prompt_token_ids) == prompt_ids
                assert generated.token_ids == list(completion.token_ids)
                expected_logprobs = [entry[token].logprob for token, entry in zip(
                    completion.token_ids, completion.logprobs, strict=True)]
                assert generated.log_probs == expected_logprobs
                assert all(math.isfinite(lp) for lp in generated.log_probs)
                assert generated.extra_fields["min_global_steps"] == generated.extra_fields["max_global_steps"] == version
                health = (await client.get("health")).json()
                assert health["policy_ready"] and health["policy_version"] == version
                assert health["kv_cache_enabled"] and health["prefix_caching_enabled"]
                assert health["speculative_decoding"]["method"] == method
                assert health["adapter_hash"] == ack.json()["adapter_hash"]
                result["versions"].append({
                    "version": version, "ack": ack.json(), "health": health,
                    "token_ids": generated.token_ids, "log_probs": generated.log_probs,
                    "stop_reason": generated.stop_reason,
                    "text": tokenizer.decode(generated.token_ids, skip_special_tokens=False),
                    "exact_raw_vllm_token_and_logprob_match": True,
                })
            result["evidence"] = backend.evidence
    result["status"] = "passed"
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("ngram", "mtp"), required=True)
    parser.add_argument("--adapter", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = asyncio.run(run_probe(args.config, args.method, args.adapter, args.output))
    print(json.dumps({"status": result["status"], "method": args.method,
                      "versions": [v["version"] for v in result["versions"]]}))


if __name__ == "__main__":
    main()
