from __future__ import annotations

import subprocess
import sys


def test_service_client_probes_unpinned_then_pins_active_requests_and_allows_stale_release():
    code = r"""
import asyncio
import os
import sys

import httpx
from qwen_rl.worker.client import GenerationContractError, HTTPGenerationManager
assert "verl" not in sys.modules

seen = []
os.environ["QWEN_LOCAL_WORKER_ABC_TOKEN"] = "isolated-worker-token"
os.environ["ARC_ENV_TOKEN"] = "central-arc-token"
node_id = "manager-node"
worker_id = "worker-abc"
boot_id = "kernel-boot-a"
epoch = "worker-epoch-a"
adapter_hash = "a" * 64

def handler(request):
    seen.append(request)
    assert request.headers.get("authorization") == "Bearer isolated-worker-token"
    if request.url.path.endswith("/health"):
        assert request.headers.get("x-expected-worker-boot-id") is None
        assert request.headers.get("x-expected-worker-telemetry-epoch") is None
        return httpx.Response(200, json={
            "role": "worker", "ready": True, "worker_id": worker_id,
            "boot_id": boot_id, "telemetry_epoch": epoch,
            "deployment": {"scope": "service", "node_id": node_id, "site_id": None},
        }, headers={"X-Worker-Node-ID": node_id,
                    "X-Worker-Boot-ID": boot_id,
                    "X-Worker-Telemetry-Epoch": epoch})
    assert request.headers.get("x-expected-worker-boot-id") == boot_id
    assert request.headers.get("x-expected-worker-telemetry-epoch") == epoch
    if request.url.path.endswith("/episode/acquire"):
        return httpx.Response(200, json={
            "episode_id": "episode-1", "policy_version": 7,
            "adapter_hash": adapter_hash, "purpose": "training", "acquired": True,
        }, headers={"X-Worker-Node-ID": node_id,
                    "X-Worker-Boot-ID": boot_id,
                    "X-Worker-Telemetry-Epoch": epoch})
    if request.url.path.endswith("/episode/release"):
        return httpx.Response(200, json={
            "episode_id": "episode-1", "purpose": "training", "released": True,
        }, headers={"X-Worker-Node-ID": node_id,
                    "X-Worker-Boot-ID": "kernel-boot-b",
                    "X-Worker-Telemetry-Epoch": "worker-epoch-b"})
    raise AssertionError(request.url.path)

async def run():
    manager = HTTPGenerationManager(
        "http://worker.local:8241", scope="service", worker_id=worker_id,
        node_id=node_id, auth_token_env="QWEN_LOCAL_WORKER_ABC_TOKEN",
        http_transport=httpx.MockTransport(handler),
    )
    health = await manager.get_health()
    assert health["boot_id"] == boot_id and health["telemetry_epoch"] == epoch
    try:
        await manager.acquire_episode(
            episode_id="episode-1", policy_version=7, adapter_hash=adapter_hash,
        )
    except GenerationContractError as exc:
        assert "unpinned" in str(exc)
    else:
        raise AssertionError("service Worker accepted an active request before identity pinning")
    assert len(seen) == 1
    manager.pin_worker_identity(boot_id, epoch)
    assert await manager.acquire_episode(
        episode_id="episode-1", policy_version=7, adapter_hash=adapter_hash,
    ) is True
    assert await manager.release_episode(
        episode_id="episode-1", policy_version=7, adapter_hash=adapter_hash,
    ) is True
    assert [request.url.path for request in seen] == [
        "/health", "/episode/acquire", "/episode/release",
    ]
    await manager.aclose()

asyncio.run(run())
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
