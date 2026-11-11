"""Worker health, policy verification, and authenticated sync control plane."""
from __future__ import annotations

from ...rollout.groups import PolicySnapshot

import asyncio
import hmac
import os
import time
import ssl
from functools import partial

import certifi
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from ...worker.policy_sync import sync_worker_policy
from ...worker.errors import WorkerUnavailableError


class WorkerControlPlane:
    """Use each Worker descriptor's route and token for Learner-to-Worker calls."""

    def __init__(
        self,
        *,
        workers: list[dict[str, Any]],
        request_timeout: float,
        quantization_mode: str,
        policies: dict[int, dict[str, Any]],
        policy_refs: dict[int, int],
        run_dir: str | Path,
        training_telemetry: Any,
        timings: dict[str, Any],
        external_stage_timings: dict[str, Any],
        get_monitor_phase: Callable[[], str],
        set_monitor_phase: Callable[[str], None],
        set_serve_version: Callable[[int], None],
        set_last_acks: Callable[[dict[str, dict[str, Any]]], None],
        prune_synced_policies: Callable[..., Any],
        policy_file_references: Any | None = None,
        required_policy_worker_ids: tuple[str, ...] = (),
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.workers = workers
        self.request_timeout = float(request_timeout)
        self.quantization_mode = quantization_mode
        self.policies = policies
        self.policy_refs = policy_refs
        self.run_dir = Path(run_dir)
        self.training_telemetry = training_telemetry
        self.timings = timings
        self.external_stage_timings = external_stage_timings
        self.get_monitor_phase = get_monitor_phase
        self.set_monitor_phase = set_monitor_phase
        self.set_serve_version = set_serve_version
        self.set_last_acks = set_last_acks
        self.prune_synced_policies = prune_synced_policies
        self.policy_file_references = policy_file_references
        self.required_policy_worker_ids = tuple(required_policy_worker_ids)
        # Preserve TLS verification and trust_env=False; load the default CA
        # store once per controller instead of once per control request. Test or
        # caller-supplied factories keep complete ownership of their transports.
        self.client_factory = client_factory or partial(
            httpx.AsyncClient, verify=ssl.create_default_context(cafile=certifi.where()))
        from ..policy_maintenance import PolicyMaintenance
        self._maintenance = PolicyMaintenance(
            self._prune_policy_cache,
            lambda status: self.external_stage_timings.update(policy_cleanup=status))

    def _prune_policy_cache(self, policy, acknowledgements, worker_ids):
        # Resolve live rollout references when maintenance actually starts;
        # file-response references are acquired by the pruner under its lock.
        return self.prune_synced_policies(
            self.run_dir, self.policies, policy, acknowledgements, worker_ids,
            frozenset(self.policy_refs), self.policy_file_references)

    async def drain_policy_cleanup(self):
        await asyncio.to_thread(self._maintenance.drain)

    def close(self):
        self._maintenance.close()

    def worker(self, worker_id: str) -> dict[str, Any]:
        worker = next((item for item in self.workers if item["id"] == worker_id), None)
        if worker is None:
            raise ValueError(f"unknown worker id {worker_id}")
        return worker

    def worker_url(self, worker_id: str) -> str:
        worker = self.worker(worker_id)
        scope = worker.get("scope", "local")
        if scope in {"service", "gateway"}:
            endpoint = worker.get("endpoint")
            prefix = worker.get("url_prefix", "")
            if not isinstance(endpoint, str) or not endpoint:
                raise ValueError(f"{scope} Worker {worker_id} is missing its endpoint")
            if not isinstance(prefix, str) or (prefix and not prefix.startswith("/")):
                raise ValueError(f"{scope} Worker {worker_id} has an invalid url_prefix")
            return endpoint.rstrip("/") + prefix.rstrip("/")
        return f"http://{worker['host']}:{worker['port']}"

    def auth_headers(self, worker: dict[str, Any]) -> dict[str, str]:
        token_env = worker.get("auth_token_env", "ARC_ENV_TOKEN")
        if not isinstance(token_env, str) or not token_env:
            raise ValueError(f"Worker {worker['id']} has an invalid auth_token_env")
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(f"{token_env} must be set for Worker {worker['id']} access")
        if worker.get("scope") in {"service", "gateway"}:
            central_token = os.environ.get("ARC_ENV_TOKEN")
            if central_token and hmac.compare_digest(token, central_token):
                raise ValueError(f"Worker {worker['id']} auth token must differ from ARC_ENV_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}
        identity = self._expected_identity(worker)
        if identity is not None:
            headers.update({
                "X-Expected-Worker-Boot-ID": identity["boot_id"],
                "X-Expected-Worker-Telemetry-Epoch": identity["telemetry_epoch"],
            })
        return headers

    @staticmethod
    def _expected_identity(worker: dict[str, Any]) -> dict[str, str] | None:
        scope = worker.get("scope", "local")
        if scope not in {"local", "service", "gateway"}:
            raise ValueError(f"Worker {worker.get('id')} has an invalid scope")
        if scope == "local":
            return None
        values = {
            "node_id": worker.get("node_id"),
            "boot_id": worker.get("boot_id"),
            "telemetry_epoch": worker.get("telemetry_epoch"),
        }
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError(
                f"{scope.title()} Worker {worker.get('id')} requires explicit node, boot, and telemetry identity"
            )
        return values

    def verify_worker_response_identity(
        self, worker: dict[str, Any], response: httpx.Response,
    ) -> None:
        identity = self._expected_identity(worker)
        if identity is None:
            return
        expected = {
            "X-Worker-Node-ID": identity["node_id"],
            "X-Worker-Boot-ID": identity["boot_id"],
            "X-Worker-Telemetry-Epoch": identity["telemetry_epoch"],
        }
        values = {name: response.headers.get_list(name) for name in expected}
        if not response.is_success and not any(values.values()):
            return
        for name, expected_value in expected.items():
            received = values[name]
            if len(received) != 1 or received[0] != expected_value:
                raise RuntimeError(
                    f"Worker {worker['id']} response identity {name} differs from its pinned descriptor"
                )

    def _verify_external_health(self, worker: dict[str, Any], health: dict[str, Any]) -> None:
        scope = worker.get("scope", "local")
        if scope not in {"service", "gateway"}:
            return
        identity = self._expected_identity(worker)
        deployment = health.get("deployment")
        # Descriptor scope is the request route. A Gateway can proxy either
        # a direct service Worker or a Dragonfly-deployed Worker. Its actual
        # node/site/boot/process identity must still match every pinned field.
        deployment_scopes = {"service", "gateway"} if scope == "gateway" else {"service"}
        if (not isinstance(deployment, dict)
                or deployment.get("scope") not in deployment_scopes
                or deployment.get("node_id") != identity["node_id"]
                or deployment.get("site_id") != worker.get("site_id")
                or health.get("boot_id") != identity["boot_id"]
                or health.get("telemetry_epoch") != identity["telemetry_epoch"]):
            raise RuntimeError(f"Worker {worker['id']} health body differs from its pinned deployment identity")

    async def health_of_worker(self, worker_id: str) -> dict[str, Any]:
        worker = self.worker(worker_id)
        async with self.client_factory(
            timeout=self.request_timeout,
            headers=self.auth_headers(worker),
            trust_env=False,
        ) as client:
            try:
                response = await client.get(f"{self.worker_url(worker_id)}/health")
            except httpx.HTTPError as exc:
                raise WorkerUnavailableError(
                    f"worker {worker_id} health request failed ({type(exc).__name__})",
                    worker_id=worker_id,
                ) from exc
        self.verify_worker_response_identity(worker, response)
        if response.status_code >= 500:
            raise WorkerUnavailableError(
                f"worker {worker_id} health returned HTTP {response.status_code}",
                worker_id=worker_id,
            )
        if response.status_code != 200:
            raise RuntimeError(f"worker {worker_id} health returned HTTP {response.status_code}")
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError(f"worker {worker_id} health response is invalid")
        if data.get("worker_id") != worker_id:
            raise RuntimeError(f"worker {worker_id} health identifies a different Worker")
        self._verify_external_health(worker, data)
        return data

    def _selected_workers(self, worker_ids: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
        if worker_ids is None:
            return list(self.workers)
        if (not isinstance(worker_ids, (tuple, list)) or not worker_ids
                or any(not isinstance(worker_id, str) or not worker_id for worker_id in worker_ids)
                or len(worker_ids) != len(set(worker_ids))):
            raise ValueError("selected worker IDs must be a nonempty unique string sequence")
        by_id = {worker["id"]: worker for worker in self.workers}
        unknown = set(worker_ids) - set(by_id)
        if unknown:
            raise ValueError("selected worker IDs are not configured: " + ", ".join(sorted(unknown)))
        return [by_id[worker_id] for worker_id in worker_ids]

    async def assert_workers_current(
        self,
        expected_version: int,
        *,
        worker_ids: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        selected_workers = self._selected_workers(worker_ids)
        policy = self.policies.get(expected_version)
        if policy is None:
            raise RuntimeError(f"learner policy {expected_version} is not published")
        expected_adapter_hash = PolicySnapshot.from_policy(policy).adapter_sha256
        observations = await asyncio.gather(
            *(self.health_of_worker(worker['id']) for worker in selected_workers),
            return_exceptions=True)
        for worker, health in zip(selected_workers, observations, strict=True):
            if isinstance(health, BaseException):
                raise health
            quantization = health.get("quantization")
            identity_matches = (
                health.get("artifact_fingerprint") == policy["artifact_fingerprint"]
                and health.get("base_fingerprint") == policy["base_fingerprint"]
            )
            if self.quantization_mode == "ternary":
                from ...worker.native_artifact import (
                    validate_policy_quantization, validate_ternary_runtime_profile,
                )

                try:
                    model_id = policy["_native_export_identity"]["model"]["id"]
                    validate_ternary_runtime_profile(health, model_id=model_id)
                    validate_policy_quantization(
                        "ternary", quantization, policy.get("quantization")
                    )
                except (KeyError, TypeError, ValueError):
                    ternary_quantization_matches = False
                else:
                    ternary_quantization_matches = True
                artifact_sha = policy["base_fingerprint"]
                tensor_count = health.get("quantized_tensor_count")
                mode_ready = (
                    health.get("backend") == "ninfer"
                    and health.get("is_loaded_in_4bit") is False
                    and health.get("ninfer_artifact_sha256") == artifact_sha
                    and policy.get("artifact_fingerprint") == artifact_sha
                    and isinstance(quantization, dict)
                    and quantization.get("mode") == "ternary"
                    and quantization.get("bits") == 2
                    and quantization.get("packed_resident") is True
                    and health.get("base_trainable_parameters") == 0
                    and isinstance(tensor_count, int) and not isinstance(tensor_count, bool)
                    and tensor_count > 0
                    and ternary_quantization_matches
                )
            else:
                module_count = health.get("module_count")
                mode_ready = (
                    health.get("backend") == "vllm"
                    and health.get("is_loaded_in_4bit") is True
                    and isinstance(module_count, int) and not isinstance(module_count, bool)
                    and module_count > 0
                    and isinstance(quantization, dict)
                    and quantization.get("bits") == 4
                    and quantization.get("mode") == self.quantization_mode
                )
            if (health.get("ready") is not True or health.get("role") != "worker"
                    or not mode_ready or health.get("policy_ready") is not True
                    or health.get("worker_id") != worker["id"]
                    or health.get("policy_version") != expected_version
                    or not identity_matches
                    or health.get("adapter_hash") != expected_adapter_hash):
                raise RuntimeError(f"worker {worker['id']} is not ready on policy {expected_version}")

    def _gateway_inventory_pending(
        self, members: list[dict[str, Any]], rows: list[Any], snapshot: PolicySnapshot,
        site_id: str,
    ) -> list[str]:
        by_node = {
            self._expected_identity(worker)["node_id"]: worker for worker in members
        }
        found: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("Gateway inventory contains a malformed worker row")
            row_node = row.get("node_id")
            row_worker = row.get("worker_id")
            node_id = row_node if isinstance(row_node, str) and row_node in by_node else row_worker
            if not isinstance(node_id, str) or node_id not in by_node:
                continue
            worker = by_node[node_id]
            expected = self._expected_identity(worker)
            if row_node != node_id or row_worker != node_id:
                raise RuntimeError(
                    f"Gateway inventory crossed the pinned node identity for Worker {worker['id']}"
                )
            if row.get("site_id") != site_id:
                raise RuntimeError(
                    f"Gateway inventory crossed the pinned site identity for Worker {worker['id']}"
                )
            if row.get("boot_id") != expected["boot_id"]:
                raise RuntimeError(
                    f"Gateway inventory crossed the pinned boot identity for Worker {worker['id']}"
                )
            if node_id in found:
                raise RuntimeError(f"Gateway inventory contains duplicate rows for node {node_id}")
            found[node_id] = row

        pending = []
        for node_id, worker in by_node.items():
            row = found.get(node_id)
            if row is None:
                pending.append(node_id)
                continue
            expected = self._expected_identity(worker)
            health = row.get("worker_health")
            if not isinstance(health, dict):
                pending.append(node_id)
                continue
            health_boot = health.get("boot_id")
            telemetry_epoch = health.get("telemetry_epoch")
            if health_boot is not None and health_boot != expected["boot_id"]:
                raise RuntimeError(
                    f"Gateway Worker {worker['id']} health crossed the pinned boot identity"
                )
            if telemetry_epoch is not None and telemetry_epoch != expected["telemetry_epoch"]:
                raise RuntimeError(
                    f"Gateway Worker {worker['id']} health crossed the pinned telemetry epoch"
                )
            if health_boot is None or telemetry_epoch is None:
                pending.append(node_id)
                continue
            policy_version = health.get("policy_version")
            matches_policy = (
                type(policy_version) is int and policy_version == snapshot.version
                and health.get("adapter_hash") == snapshot.adapter_sha256
            )
            if row.get("eligible") is not True or not matches_policy:
                pending.append(node_id)
        return pending

    async def _wait_gateway_policy_eligibility(
        self,
        policy: dict[str, Any],
        *,
        worker_ids: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """Wait until Gateway inventory reflects the just-ACKed policy."""
        gateway_workers = [
            worker for worker in self._selected_workers(worker_ids)
            if worker.get("scope") == "gateway"
        ]
        if not gateway_workers:
            return
        snapshot = PolicySnapshot.from_policy(policy)
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for worker in gateway_workers:
            endpoint = worker.get("endpoint")
            token_env = worker.get("auth_token_env")
            site_id = worker.get("site_id")
            if (not isinstance(endpoint, str) or not endpoint.strip()
                    or not isinstance(token_env, str) or not token_env.strip()
                    or not isinstance(site_id, str) or not site_id.strip()):
                raise ValueError(f"Gateway Worker {worker.get('id')} has incomplete inventory routing")
            self._expected_identity(worker)
            groups.setdefault((endpoint.rstrip("/"), token_env, site_id), []).append(worker)

        all_node_ids = [self._expected_identity(worker)["node_id"] for worker in gateway_workers]
        if len(all_node_ids) != len(set(all_node_ids)):
            raise ValueError("Gateway workers must map to unique node IDs")
        deadline = time.monotonic() + self.request_timeout
        all_node_ids = sorted(all_node_ids)
        last_observation = "inventory has not been queried"
        while True:
            pending_nodes = []
            for (endpoint, _token_env, site_id), members in groups.items():
                node_ids = [self._expected_identity(worker)["node_id"] for worker in members]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pending_nodes.extend(node_ids)
                    continue
                auth = self.auth_headers(members[0])["Authorization"]
                try:
                    async with self.client_factory(
                        timeout=min(self.request_timeout, remaining),
                        headers={"Authorization": auth, "Accept": "application/json"},
                        trust_env=False,
                    ) as client:
                        response = await client.get(f"{endpoint}/v1/workers")
                except httpx.HTTPError as exc:
                    last_observation = f"Gateway inventory request failed ({type(exc).__name__})"
                    pending_nodes.extend(node_ids)
                    continue
                if response.status_code in {401, 403}:
                    raise RuntimeError(
                        f"Gateway inventory authorization failed with HTTP {response.status_code}"
                    )
                if response.status_code == 503:
                    try:
                        body = response.json()
                    except ValueError:
                        body = {}
                    detail = body.get("detail") if isinstance(body, dict) else None
                    if not isinstance(detail, dict) or detail.get("code") != "site_inventory_stale":
                        raise RuntimeError("Gateway inventory returned an unexpected HTTP 503")
                    last_observation = "Gateway site inventory is stale"
                    pending_nodes.extend(node_ids)
                    continue
                if response.status_code != 200:
                    raise RuntimeError(f"Gateway inventory returned HTTP {response.status_code}")
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RuntimeError("Gateway inventory returned invalid JSON") from exc
                rows = payload.get("workers") if isinstance(payload, dict) else None
                if not isinstance(rows, list):
                    raise RuntimeError("Gateway inventory response has no worker list")
                pending_nodes.extend(self._gateway_inventory_pending(
                    members, rows, snapshot, site_id,
                ))
                if pending_nodes:
                    last_observation = "Gateway inventory is missing or has not published the expected policy"

            pending_nodes = sorted(set(pending_nodes))
            if not pending_nodes:
                if time.monotonic() <= deadline:
                    return
                pending_nodes = all_node_ids
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Gateway policy eligibility timed out after {self.request_timeout:g}s "
                    f"for {', '.join(pending_nodes)}: {last_observation}"
                )
            await asyncio.sleep(min(0.25, remaining))

    async def sync_workers(self, policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
        expected_version = policy["version"]
        previous_phase = self.get_monitor_phase()
        if self.training_telemetry.active is None:
            self.set_monitor_phase("worker_sync")
        expected_files = policy["files"]
        started = time.perf_counter()

        async def sync_one(worker: dict[str, Any]) -> dict[str, Any]:
            async with self.client_factory(
                timeout=self.request_timeout,
                headers=self.auth_headers(worker),
                trust_env=False,
            ) as client:
                fenced_client = _IdentityFencedClient(self, worker, client)
                return await sync_worker_policy(
                    fenced_client,
                    worker_id=worker["id"],
                    worker_url=self.worker_url(worker["id"]),
                    version=expected_version,
                    expected_files=expected_files,
                    expected_adapter_hash=PolicySnapshot.from_policy(policy).adapter_sha256,
                    request_timeout=self.request_timeout,
                )

        results = await asyncio.gather(
            *(sync_one(worker) for worker in self.workers),
            return_exceptions=True,
        )
        acknowledgements: dict[str, dict[str, Any]] = {}
        for worker, result in zip(self.workers, results):
            if isinstance(result, BaseException):
                raise result
            acknowledgements[worker["id"]] = result

        rpc_seconds = time.perf_counter() - started
        from ..policy_maintenance import validate_policy_acknowledgements
        validate_policy_acknowledgements(policy, acknowledgements, {w['id'] for w in self.workers})
        health_started = time.perf_counter()
        await self.assert_workers_current(expected_version)
        health_seconds = time.perf_counter() - health_started
        eligibility_started = time.perf_counter()
        await self._wait_gateway_policy_eligibility(policy)
        self.external_stage_timings['worker_sync_health_seconds'] = health_seconds
        self.external_stage_timings['worker_sync_eligibility_seconds'] = time.perf_counter() - eligibility_started
        self.set_serve_version(expected_version)
        if self.training_telemetry.active is None and self.get_monitor_phase() == "worker_sync":
            self.set_monitor_phase(
                previous_phase if previous_phase in {"awaiting_groups", "pre_alignment"} else "idle"
            )
        self.set_last_acks(acknowledgements)
        # Only fully validated ACKs may schedule reference-aware maintenance.
        self._maintenance.schedule((policy, acknowledgements, {w['id'] for w in self.workers}))
        sync_elapsed = time.perf_counter() - started
        self.timings["learner"]["worker_sync_seconds"] += sync_elapsed
        self.external_stage_timings["worker_sync_seconds"] = sync_elapsed
        self.external_stage_timings["worker_sync_policy_version"] = expected_version
        self.external_stage_timings['worker_sync_stages'] = {
            'policy_version': expected_version, 'worker_ids': [w['id'] for w in self.workers],
            'rpc_seconds': rpc_seconds, 'health_seconds': health_seconds,
            'eligibility_seconds': self.external_stage_timings['worker_sync_eligibility_seconds'],
            'total_seconds': sync_elapsed}
        return acknowledgements

    async def sync_selected_workers(
        self,
        policy: dict[str, Any],
        worker_ids: tuple[str, ...] | list[str],
        *,
        wait_gateway: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Install one immutable policy on an explicit subset and prove its ACKs."""
        started = time.perf_counter()
        selected_workers = self._selected_workers(worker_ids)
        expected_version = policy.get("version")
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("selected Worker policy version must be a non-negative integer")
        expected_files = policy["files"]
        expected_hash = PolicySnapshot.from_policy(policy).adapter_sha256

        async def sync_one(worker: dict[str, Any]) -> dict[str, Any]:
            async with self.client_factory(
                timeout=self.request_timeout,
                headers=self.auth_headers(worker),
                trust_env=False,
            ) as client:
                fenced_client = _IdentityFencedClient(self, worker, client)
                return await sync_worker_policy(
                    fenced_client,
                    worker_id=worker["id"],
                    worker_url=self.worker_url(worker["id"]),
                    version=expected_version,
                    expected_files=expected_files,
                    expected_adapter_hash=expected_hash,
                    request_timeout=self.request_timeout,
                )

        results = await asyncio.gather(
            *(sync_one(worker) for worker in selected_workers),
            return_exceptions=True,
        )
        acknowledgements: dict[str, dict[str, Any]] = {}
        for worker, result in zip(selected_workers, results, strict=True):
            if isinstance(result, BaseException):
                raise result
            acknowledgements[worker["id"]] = result
        rpc_seconds = time.perf_counter() - started
        health_started = time.perf_counter()
        await self.assert_workers_current(expected_version, worker_ids=list(acknowledgements))
        health_seconds = time.perf_counter() - health_started
        eligibility_seconds = 0.0
        required_workers = set(self.required_policy_worker_ids)
        publishes_training_policy = bool(required_workers) and set(acknowledgements) == required_workers
        if publishes_training_policy or wait_gateway:
            eligibility_started = time.perf_counter()
            await self._wait_gateway_policy_eligibility(
                policy, worker_ids=tuple(acknowledgements),
            )
            eligibility_seconds = time.perf_counter() - eligibility_started
        if publishes_training_policy:
            from ..policy_maintenance import validate_policy_acknowledgements
            validate_policy_acknowledgements(policy, acknowledgements, required_workers)
            self.set_serve_version(expected_version)
            self.set_last_acks(acknowledgements)
            self._maintenance.schedule((policy, acknowledgements, required_workers))
        self.external_stage_timings['worker_sync_stages'] = {
            'policy_version': expected_version, 'worker_ids': list(acknowledgements),
            'rpc_seconds': rpc_seconds, 'health_seconds': health_seconds,
            'eligibility_seconds': eligibility_seconds,
            'total_seconds': time.perf_counter() - started}
        return acknowledgements


class _IdentityFencedClient:
    """Add response identity checks to the policy-sync POST operation."""

    def __init__(self, control: WorkerControlPlane, worker: dict[str, Any], client: Any) -> None:
        self.control = control
        self.worker = worker
        self.client = client

    @property
    def headers(self):
        return self.client.headers

    async def post(self, *args, **kwargs):
        response = await self.client.post(*args, **kwargs)
        self.control.verify_worker_response_identity(self.worker, response)
        return response
