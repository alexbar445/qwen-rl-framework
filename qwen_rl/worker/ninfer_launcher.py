"""Supervise one NInfer engine and the existing authenticated Qwen Worker API.

This launcher is deliberately separate from the normal Worker entry point. The
engine must prove the RL policy capabilities required by the Worker before the
HTTP service is started; there is no vLLM fallback path here.
"""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .residency import parse_residency_settings


class LauncherError(RuntimeError):
    """A startup, readiness, or child-process failure."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_VOCAB_SIZE = 248_320
_SAMPLING_VOCAB_SIZE = 248_077
_BEHAVIOR_DISTRIBUTION = "raw_policy_public_vocab"
_T2_BASE_ARTIFACT_KINDS = frozenset({"bonsai2_t2", "coder390_t2_full_scales_step512"})


def _private_token_file(path: Path) -> None:
    from ..cluster.server import read_token_file

    read_token_file(path)


def _absolute_config_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LauncherError(f"worker config requires a non-empty {name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise LauncherError(f"worker config {name} must be an absolute path")
    return path.resolve(strict=True)


def _absolute_directory_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LauncherError(f"worker config requires a non-empty {name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise LauncherError(f"worker config {name} must be an absolute path")
    if path.is_symlink():
        raise LauncherError(f"worker config {name} must not be a symlink")
    return path.resolve(strict=False)


def _load_worker_settings(config_path: Path) -> tuple[dict[str, Any], Path, Path, Path, str]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LauncherError(f"cannot read worker JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise LauncherError("worker JSON must contain an object")
    inference = config.get("inference")
    if not isinstance(inference, dict):
        raise LauncherError("worker JSON inference must be an object")
    if inference.get("backend") != "ninfer":
        raise LauncherError("worker JSON must explicitly select inference.backend=ninfer")
    artifact = _absolute_config_path(inference.get("ninfer_artifact_path"), "ninfer_artifact_path")
    if artifact.suffix != ".ninfer" or not artifact.is_file():
        raise LauncherError("inference.ninfer_artifact_path must name an existing .ninfer file")
    report = _absolute_config_path(
        inference.get("ninfer_conversion_report"), "ninfer_conversion_report"
    )
    if not report.is_file():
        raise LauncherError("inference.ninfer_conversion_report must name an existing file")
    policy_root = _absolute_directory_path(config.get("adapter_cache"), "adapter_cache")
    if policy_root.exists() and not policy_root.is_dir():
        raise LauncherError("worker adapter_cache must be a directory")
    policy_root.mkdir(parents=True, exist_ok=True)
    ninfer_url = inference.get("ninfer_url")
    if not isinstance(ninfer_url, str) or not ninfer_url.strip():
        raise LauncherError("worker JSON requires inference.ninfer_url")
    return config, artifact, report, policy_root, ninfer_url


def _conversion_identity(
    report_path: Path,
    *,
    artifact_path: Path | None = None,
    quantization: str | None = None,
    declared_artifact_sha256: str | None = None,
    artifact_identity_path: str | Path | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 16,
) -> dict[str, Any]:
    if quantization != "ternary":
        raise LauncherError("NInfer launcher supports only explicit T2 artifacts")
    if artifact_path is None:
        raise LauncherError("T2 startup requires the configured .ninfer artifact path")
    try:
        from .native_artifact import read_ternary_worker_artifact

        if not isinstance(declared_artifact_sha256, str) or _SHA256.fullmatch(declared_artifact_sha256) is None:
            raise LauncherError("T2 startup requires deployment base SHA-256 declaration")
        evidence = read_ternary_worker_artifact(
            artifact_path,
            report_path,
            declared_artifact_sha256=declared_artifact_sha256,
            artifact_identity_path=artifact_identity_path,
            expected_lora_rank=lora_rank,
            expected_lora_alpha=lora_alpha,
        )
    except Exception as exc:
        raise LauncherError("NInfer T2 artifact identity is invalid") from exc
    return {
        "base_artifact_kind": evidence.base_artifact_kind,
        "base_fingerprint": evidence.base_fingerprint,
        "artifact_fingerprint": evidence.artifact_fingerprint,
        "ninfer_artifact_id": evidence.ninfer_artifact_id,
        "ninfer_artifact_sha256": evidence.ninfer_artifact_sha256,
        "source_quantization": evidence.quantization,
        "gdn_projections": evidence.gdn_projections,
        "input_basis": evidence.input_basis,
        "base_weight_profile": evidence.base_weight_profile,
    }


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _worker_engine_url(ninfer_url: str, *, host: str, port: int) -> str:
    parsed = urlsplit(ninfer_url)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise LauncherError("inference.ninfer_url must be an http loopback engine origin without a path")
    if not _is_loopback(parsed.hostname) or not _is_loopback(host):
        raise LauncherError("the NInfer RL engine must bind to loopback on this single-host launcher")
    effective_port = parsed.port or 80
    if effective_port != port:
        raise LauncherError(
            f"inference.ninfer_url port {effective_port} does not match engine port {port}"
        )
    return f"http://{host}:{port}"


def _read_json(url: str, timeout: float) -> tuple[int, dict[str, Any] | None]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read(1024 * 1024)
            status = response.status
    except HTTPError as exc:
        payload = exc.read(1024 * 1024)
        status = exc.code
    except (OSError, URLError, TimeoutError):
        return 0, None
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, None
    return status, parsed if isinstance(parsed, dict) else None


def _validate_capabilities(
    capabilities: dict[str, Any] | None,
    *,
    expected: dict[str, Any],
    max_context: int,
    max_concurrency: int,
    policy_slots: int,
) -> None:
    if not isinstance(capabilities, dict):
        raise LauncherError("NInfer returned an invalid RL capabilities document")
    if expected.get("base_artifact_kind") not in _T2_BASE_ARTIFACT_KINDS:
        raise LauncherError("NInfer launcher expected identity is not a supported T2 profile")
    runtime_quantization = capabilities.get("runtime_quantization")
    try:
        from .native_artifact import validate_ternary_runtime_quantization

        validate_ternary_runtime_quantization(runtime_quantization)
        valid_t2_runtime_formats = True
    except ValueError:
        valid_t2_runtime_formats = False
    required_t2 = {
        "ready": capabilities.get("ready") is True,
        "protocol version": capabilities.get("protocol_version") == 1,
        "backend": capabilities.get("backend") == "ninfer",
        "build ID": isinstance(capabilities.get("build_id"), str) and bool(capabilities.get("build_id")),
        "base artifact kind": capabilities.get("base_artifact_kind") == expected["base_artifact_kind"],
        "base fingerprint": capabilities.get("base_fingerprint") == expected["base_fingerprint"],
        "artifact fingerprint": capabilities.get("artifact_fingerprint") == expected["artifact_fingerprint"],
        "NInfer artifact ID": capabilities.get("ninfer_artifact_id") == expected["ninfer_artifact_id"],
        "NInfer artifact SHA": capabilities.get("ninfer_artifact_sha256") == expected["ninfer_artifact_sha256"],
        "input basis": capabilities.get("input_basis") == expected["input_basis"],
        "base weight profile": capabilities.get("base_weight_profile") == expected["base_weight_profile"],
        "not loaded as four-bit": capabilities.get("is_loaded_in_4bit") is False,
        "frozen base": (
            isinstance(capabilities.get("base_trainable_parameters"), int)
            and not isinstance(capabilities.get("base_trainable_parameters"), bool)
            and capabilities["base_trainable_parameters"] == 0
        ),
        "full token logprobs": capabilities.get("supports_full_token_logprobs") is True,
        "raw shared-prefix API": capabilities.get("supports_shared_cache_prefix") is True,
        "neutral full-vocabulary sampling": capabilities.get("neutral_full_vocab_sampling") is True,
        "behavior distribution": capabilities.get("behavior_distribution") == _BEHAVIOR_DISTRIBUTION,
        "model vocabulary size": capabilities.get("model_vocab_size") == _MODEL_VOCAB_SIZE,
        "sampling vocabulary size": capabilities.get("sampling_vocab_size") == _SAMPLING_VOCAB_SIZE,
        "LoRA support": capabilities.get("supports_lora") is True,
        "source T2 quantization": capabilities.get("source_quantization") == expected["source_quantization"],
        "runtime T2 formats": valid_t2_runtime_formats,
        "loaded quantized tensors": (
            isinstance(capabilities.get("quantized_tensor_count"), int)
            and not isinstance(capabilities.get("quantized_tensor_count"), bool)
            and capabilities["quantized_tensor_count"] > 0
        ),
        "max generated tokens": (
            isinstance(capabilities.get("max_generated_tokens"), int)
            and not isinstance(capabilities.get("max_generated_tokens"), bool)
            and capabilities["max_generated_tokens"] > 0
        ),
        "at least two policy slots": (
            isinstance(capabilities.get("policy_slots"), int)
            and not isinstance(capabilities.get("policy_slots"), bool)
            and capabilities["policy_slots"] >= max(2, policy_slots)
        ),
        "configured context capacity": (
            isinstance(capabilities.get("max_context"), int)
            and not isinstance(capabilities.get("max_context"), bool)
            and capabilities["max_context"] >= max_context
        ),
        "configured generation concurrency": (
            isinstance(capabilities.get("max_concurrency"), int)
            and not isinstance(capabilities.get("max_concurrency"), bool)
            and max_concurrency <= capabilities["max_concurrency"] <= 8
        ),
    }
    if expected.get("gdn_projections") is True:
        required_t2["all-GDN LoRA support"] = (
            capabilities.get("supports_gdn_lora_projections") is True
        )
    elif "supports_gdn_lora_projections" in capabilities:
        required_t2["GDN LoRA remains disabled"] = (
            capabilities.get("supports_gdn_lora_projections") is False
        )
    missing_t2 = [name for name, valid in required_t2.items() if not valid]
    if missing_t2:
        raise LauncherError("NInfer T2 RL capabilities are incomplete: " + ", ".join(missing_t2))


def _capabilities_url(engine_base_url: str) -> str:
    return engine_base_url.rstrip("/") + "/v1/rl/capabilities"


async def _wait_for_engine(
    engine: subprocess.Popen,
    *,
    base_url: str,
    expected_identity: dict[str, Any],
    max_context: int,
    max_concurrency: int,
    policy_slots: int,
    timeout_seconds: float,
    shutdown_event: asyncio.Event,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = base_url.rstrip("/") + "/health"
    caps_url = _capabilities_url(base_url)
    while time.monotonic() < deadline:
        if shutdown_event.is_set():
            raise LauncherError("shutdown requested before NInfer became ready")
        if engine.poll() is not None:
            raise LauncherError(f"NInfer exited before readiness with code {engine.returncode}")
        # Startup-only probes are short and synchronous. This keeps signal
        # handling and the test surface independent of a background thread.
        health_status, health = _read_json(health_url, 1.0)
        if health_status == 200 and isinstance(health, dict) and health.get("status") == "ok":
            caps_status, capabilities = _read_json(caps_url, 2.0)
            if caps_status == 200:
                _validate_capabilities(
                    capabilities,
                    expected=expected_identity,
                    max_context=max_context,
                    max_concurrency=max_concurrency,
                    policy_slots=policy_slots,
                )
                return
            if caps_status not in (0, 503):
                raise LauncherError(
                    f"NInfer RL capabilities endpoint returned HTTP {caps_status}"
                )
        await asyncio.sleep(0.2)
    raise LauncherError(f"NInfer did not prove readiness within {timeout_seconds:g} seconds")


async def _stop_process(
    process: subprocess.Popen | None,
    *,
    grace_seconds: float,
) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(_wait_process(process), timeout=grace_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await _wait_process(process)


async def _wait_process(process: subprocess.Popen) -> int:
    """Poll and reap a child without depending on an asyncio watcher thread."""
    while True:
        code = process.poll()
        if code is not None:
            return code
        await asyncio.sleep(0.05)


def _shell_exit_code(code: int) -> int:
    return 128 + abs(code) if code < 0 else code


async def supervise(
    *,
    engine_command: Sequence[str],
    worker_command: Sequence[str],
    engine_base_url: str,
    expected_identity: dict[str, Any],
    max_context: int,
    max_concurrency: int,
    policy_slots: int,
    engine_environment: dict[str, str],
    worker_environment: dict[str, str],
    ready_timeout: float,
    shutdown_grace: float,
    shutdown_event: asyncio.Event,
    shutdown_signal: list[int | None],
    device_profile_cache=None,
    device_profile_path: Path | None = None,
    worker_owns_engine: bool = False,
) -> int:
    """Start the engine, gate the Worker on RL readiness, and own both PIDs."""
    engine: subprocess.Popen | None = None
    worker: subprocess.Popen | None = None
    try:
        if worker_owns_engine:
            worker = subprocess.Popen(worker_command, env=worker_environment, start_new_session=True)
            while worker.poll() is None and not shutdown_event.is_set():
                await asyncio.sleep(.05)
            if shutdown_event.is_set():
                await _stop_process(worker, grace_seconds=shutdown_grace)
                return 128 + (shutdown_signal[0] or signal.SIGTERM)
            return _shell_exit_code(worker.returncode)
        engine = subprocess.Popen(engine_command, env=engine_environment, start_new_session=True)
        await _wait_for_engine(
            engine,
            base_url=engine_base_url,
            expected_identity=expected_identity,
            max_context=max_context,
            max_concurrency=max_concurrency,
            policy_slots=policy_slots,
            timeout_seconds=ready_timeout,
            shutdown_event=shutdown_event,
        )
        if device_profile_cache is not None and device_profile_path is not None:
            device_profile_cache.publish(device_profile_path)
        if shutdown_event.is_set():
            return 128 + (shutdown_signal[0] or signal.SIGTERM)
        worker = subprocess.Popen(worker_command, env=worker_environment, start_new_session=True)
        engine_wait = asyncio.create_task(_wait_process(engine))
        worker_wait = asyncio.create_task(_wait_process(worker))
        shutdown_wait = asyncio.create_task(shutdown_event.wait())
        done, pending = await asyncio.wait(
            {engine_wait, worker_wait, shutdown_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if shutdown_wait in done and shutdown_event.is_set():
            await _stop_process(worker, grace_seconds=shutdown_grace)
            await _stop_process(engine, grace_seconds=shutdown_grace)
            return 128 + (shutdown_signal[0] or signal.SIGTERM)
        if engine_wait in done:
            raw_code = engine.poll()
            code = _shell_exit_code(raw_code) if raw_code is not None else 1
            await _stop_process(worker, grace_seconds=shutdown_grace)
            return code if code != 0 else 1
        raw_code = worker.poll()
        code = _shell_exit_code(raw_code) if raw_code is not None else 1
        await _stop_process(engine, grace_seconds=shutdown_grace)
        return code
    except (OSError, LauncherError) as exc:
        print(f"ninfer-launcher: {exc}", file=sys.stderr, flush=True)
        await _stop_process(worker, grace_seconds=shutdown_grace)
        await _stop_process(engine, grace_seconds=shutdown_grace)
        return 1
    finally:
        await _stop_process(worker, grace_seconds=shutdown_grace)
        await _stop_process(engine, grace_seconds=shutdown_grace)


_NINFER_CONFIG_VALUE_OPTIONS = (
    "--kv-capacity", "--cuda-graph-allowance-mib", "--spec", "--draft-tokens",
    "--ngram-draft-tokens", "--kv-dtype", "--prefill-chunk",
    "--log-stats-interval-ms", "--device-profile", "--device-profile-path", "--host-kv-mib", "--host-state-slots",
)


def _take_value_options(
    engine_args: Sequence[str], options: Sequence[str],
) -> tuple[dict[str, str], list[str]]:
    found: dict[str, str] = {}
    remainder: list[str] = []
    index = 0
    while index < len(engine_args):
        token = engine_args[index]
        option, separator, inline_value = token.partition("=")
        if option not in options:
            remainder.append(token)
            index += 1
            continue
        if option in found:
            raise LauncherError(f"engine option {option} was specified more than once")
        if separator:
            if not inline_value:
                raise LauncherError(f"engine option {option} requires a value")
            value = inline_value
            index += 1
        else:
            if index + 1 >= len(engine_args) or engine_args[index + 1].startswith("--"):
                raise LauncherError(f"engine option {option} requires a value")
            value = engine_args[index + 1]
            index += 2
        found[option] = value
    return found, remainder


def _engine_arguments(
    engine_args: Sequence[str], *, configured_runtime_args: Sequence[str] = (),
) -> list[str]:
    """Combine safe extras with the single typed NInfer options contract."""
    residency_owned = {"--model-suspend", "--suspend-weights", "--suspend-snapshot", "--suspend-discard-cache", "--no-auto-resume"}
    for value in engine_args:
        option = value.split("=", 1)[0]
        if option in residency_owned:
            raise LauncherError(f"engine option {option} is owned by worker JSON residency settings")
    controlled = {
        "--host", "--port", "--rl-artifact-root", "--rl-lora-slots",
        "--rl-lora-rank-ceiling", "--rl-lora-gdn-projections",
        "--max-context", "--max-concurrency",
        "--device", "--devices", "--api-key", "--rl-pinned-base-sha256",
    }
    explicit_runtime, remaining = _take_value_options(engine_args, _NINFER_CONFIG_VALUE_OPTIONS)
    if explicit_runtime:
        option = next(iter(explicit_runtime))
        raise LauncherError(
            f"extra engine option {option} is owned by worker_inference.ninfer_options"
        )
    for value in remaining:
        option = value.split("=", 1)[0]
        if option in {"--prefill-cublas", "--no-prefill-cublas"}:
            raise LauncherError(f"engine option {option} is owned by worker_inference.ninfer_options")

    for value in remaining:
        option = value.split("=", 1)[0]
        if option in controlled:
            raise LauncherError(f"engine option {option} is owned by the launcher")
    return [*configured_runtime_args, *remaining]


def _residency_engine_arguments(worker_config: dict[str, Any]) -> list[str]:
    """Enable native held suspension only for an explicitly configured Worker."""
    try:
        settings = parse_residency_settings(worker_config)
    except ValueError as exc:
        raise LauncherError(str(exc)) from exc
    if not settings.enabled:
        return []
    inference = worker_config.get("inference")
    if not isinstance(inference, dict) or inference.get("backend") != "ninfer":
        raise LauncherError("residency.enabled requires inference.backend=ninfer")
    return [
        "--model-suspend",
        "--suspend-discard-cache",
        "--suspend-weights", settings.weight_source,
        "--suspend-snapshot", "pageable",
        "--no-auto-resume",
    ]


def _resolve_executable(value: str, name: str) -> str:
    found = shutil.which(value)
    if found is None:
        raise LauncherError(f"{name} is not executable or not on PATH: {value}")
    return str(Path(found).absolute())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qwen_rl.worker.ninfer_launcher",
        description="Run one NInfer engine and the existing authenticated Qwen cluster Worker.",
    )
    parser.add_argument("--worker-config", type=Path, required=True, help="standard cluster Worker JSON")
    parser.add_argument("--token-file", type=Path, required=True, help="private per-Worker bearer file")
    parser.add_argument("--engine-binary", required=True, help="patched ninfer-serve executable")
    parser.add_argument("--worker-python", default=sys.executable, help="Python containing qwen_rl and its Worker runtime")
    parser.add_argument("--gpu-uuid", help="selected physical GPU identity; defaults to GPU_UUID")
    parser.add_argument("--engine-host", default="127.0.0.1", help="loopback-only NInfer bind host")
    parser.add_argument("--engine-port", type=int, default=8088)
    parser.add_argument("--worker-host", default="0.0.0.0", help="Qwen Worker HTTP bind host")
    parser.add_argument("--worker-port", type=int, default=8241)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--shutdown-grace", type=float, default=15.0)
    parser.add_argument("engine_args", nargs=argparse.REMAINDER,
                        help="additional ninfer-serve flags after `--`")
    return parser


def _validate_cli(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path, Path, Path, str, str]:
    if not 1 <= args.engine_port <= 65535 or not 1 <= args.worker_port <= 65535:
        raise LauncherError("engine and Worker ports must be in [1, 65535]")
    if args.engine_port == args.worker_port and args.engine_host == args.worker_host:
        raise LauncherError("NInfer and Worker HTTP ports must differ")
    if args.ready_timeout <= 0 or args.shutdown_grace <= 0:
        raise LauncherError("readiness and shutdown timeouts must be positive")
    if not _is_loopback(args.engine_host):
        raise LauncherError("engine-host must be loopback; the policy API is private to this Worker")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible.lower() == "all" or len([part for part in visible.split(",") if part.strip()]) != 1:
        raise LauncherError("set CUDA_VISIBLE_DEVICES explicitly to exactly one GPU before launch")
    gpu_uuid = (args.gpu_uuid or os.environ.get("GPU_UUID", "")).strip()
    if not gpu_uuid:
        raise LauncherError("selected GPU identity is required via --gpu-uuid or GPU_UUID")

    config_path = args.worker_config.expanduser().resolve(strict=True)
    token_path = args.token_file.expanduser().resolve(strict=True)
    _private_token_file(token_path)
    config, artifact, report, policy_root, ninfer_url = _load_worker_settings(config_path)
    base_url = _worker_engine_url(ninfer_url, host=args.engine_host, port=args.engine_port)
    return config, artifact, report, policy_root, token_path, base_url, gpu_uuid


async def _async_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, artifact, report, policy_root, token_path, engine_base_url, gpu_uuid = _validate_cli(args)
        expected_identity = _conversion_identity(
            report,
            artifact_path=artifact,
            quantization=config["inference"].get("quantization"),
            declared_artifact_sha256=config["inference"].get("ninfer_artifact_sha256"),
            artifact_identity_path=config.get("artifact_identity_path"),
            lora_rank=config.get("lora_rank", 16),
            lora_alpha=config.get("lora_alpha", 16),
        )
        if config["inference"].get("quantization") == "ternary":
            options = config["inference"].get("ninfer_options")
            if options is None:
                options = {}
            if not isinstance(options, dict):
                raise LauncherError("worker_inference.ninfer_options must be an object")
            configured_gdn = options.get("lora_gdn_projections", False)
            if configured_gdn is not expected_identity.get("gdn_projections", False):
                raise LauncherError(
                    "worker ninfer_options.lora_gdn_projections differs from the T2 adapter identity"
                )
        from .backend import (
            DEFAULT_LORA_RANK,
            DEFAULT_WORKER_MAX_LORAS,
            DEFAULT_WORKER_MAX_MODEL_LEN,
            DEFAULT_WORKER_MAX_NUM_SEQS,
            ninfer_engine_config_arguments,
            validate_backend_config,
        )

        try:
            validate_backend_config(config["inference"])
            configured_runtime_args = ninfer_engine_config_arguments(config["inference"])
            configured_runtime_args.extend(_residency_engine_arguments(config))
        except ValueError as exc:
            raise LauncherError(str(exc)) from exc
        device_profile_path = policy_root / "device-profiles.json"
        device_profile_cache = None
        cache_directory = os.environ.get("QWEN_NINFER_DEVICE_PROFILE_CACHE")
        # Cache is an optional startup optimization. Native compatibility checks stay authoritative.
        if cache_directory and (config["inference"].get("ninfer_options") or {}).get("device_profile", "auto") == "auto":
            from .device_profiles import DeviceProfileCache
            try:
                device_profile_cache = DeviceProfileCache(Path(cache_directory))
                device_profile_cache.seed(device_profile_path)
            except (OSError, ValueError) as exc:
                print(f"ninfer-launcher: device profile cache unavailable: {exc}", file=sys.stderr, flush=True)
        configured_runtime_args.extend((
            "--device-profile-path", str(device_profile_path),
        ))
        engine_args = args.engine_args[1:] if args.engine_args[:1] == ["--"] else args.engine_args
        extra_engine_args = _engine_arguments(
            engine_args, configured_runtime_args=configured_runtime_args,
        )
        engine_binary = _resolve_executable(args.engine_binary, "engine binary")
        worker_python = _resolve_executable(args.worker_python, "worker Python")
        inference = config["inference"]
        max_context = inference.get("max_model_len", DEFAULT_WORKER_MAX_MODEL_LEN)
        max_concurrency = inference.get("max_num_seqs", DEFAULT_WORKER_MAX_NUM_SEQS)
        policy_slots = inference.get("max_loras", DEFAULT_WORKER_MAX_LORAS)
        if isinstance(max_context, bool) or not isinstance(max_context, int) or max_context <= 0:
            raise LauncherError("worker JSON inference.max_model_len must be a positive integer")
        if (isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int)
                or not 1 <= max_concurrency <= 8):
            raise LauncherError("worker JSON inference.max_num_seqs must be between 1 and 8")
        if isinstance(policy_slots, bool) or not isinstance(policy_slots, int) or policy_slots < 2:
            raise LauncherError("worker JSON inference.max_loras must be at least 2")
        lora_rank = config.get("lora_rank", DEFAULT_LORA_RANK)
        if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or lora_rank < 1:
            raise LauncherError("worker JSON lora_rank must be a positive integer")
        engine_command = [
            engine_binary,
            str(artifact),
            "--host", args.engine_host,
            "--port", str(args.engine_port),
            "--device", "0",
            "--max-context", str(max_context),
            "--max-concurrency", str(max_concurrency),
            "--rl-artifact-root", str(policy_root),
            "--rl-lora-slots", str(policy_slots),
            "--rl-lora-rank-ceiling", str(lora_rank),
            *extra_engine_args,
        ]
        if config["inference"].get("quantization") == "ternary":
            engine_command.extend(["--rl-pinned-base-sha256", expected_identity["ninfer_artifact_sha256"]])
        if parse_residency_settings(config).enabled:
            engine_command = [worker_python, "-m", "qwen_rl.worker.residency_exec", *engine_command]
        worker_command = [
            worker_python, "-m", "qwen_rl.worker.ninfer_service",
            "--config", str(args.worker_config.expanduser().resolve()),
            "--host", args.worker_host,
            "--port", str(args.worker_port),
        ]
        engine_environment = os.environ.copy()
        engine_environment.pop("ARC_ENV_TOKEN", None)
        engine_environment.pop("ARC_ENV_TOKEN_FILE", None)
        engine_environment["GPU_UUID"] = gpu_uuid
        # Bind native CUDA selection by physical identity; NVML may enumerate
        # multiple host GPUs even when the container has one assigned device.
        engine_environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        worker_environment = engine_environment.copy()
        worker_environment["ARC_ENV_TOKEN_FILE"] = str(token_path)
        worker_owns_engine = parse_residency_settings(config).engine_lifecycle == "restart"
        if worker_owns_engine:
            contract_path = policy_root / "native-process.json"
            if contract_path.is_symlink():
                raise LauncherError("Native process contract must not be a symlink")
            contract = dict(command=engine_command, base_url=engine_base_url,
                            identity=expected_identity, max_context=max_context,
                            max_concurrency=max_concurrency, policy_slots=policy_slots,
                            ready_timeout=args.ready_timeout, shutdown_grace=args.shutdown_grace,
                            device_profile_cache=cache_directory if device_profile_cache else None,
                            device_profile_path=str(device_profile_path))
            contract_path.write_text(json.dumps(contract))
            contract_path.chmod(0o600)
            worker_environment["QWEN_NINFER_ENGINE_CONTRACT"] = str(contract_path)
    except (OSError, ValueError, LauncherError) as exc:
        print(f"ninfer-launcher: {exc}", file=sys.stderr, flush=True)
        return 2

    shutdown_event = asyncio.Event()
    shutdown_signal: list[int | None] = [None]
    loop = asyncio.get_running_loop()

    def request_shutdown(signum: int) -> None:
        shutdown_signal[0] = signum
        shutdown_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, request_shutdown, signum)
    try:
        return await supervise(
            engine_command=engine_command,
            worker_command=worker_command,
            engine_base_url=engine_base_url,
            expected_identity=expected_identity,
            max_context=max_context,
            max_concurrency=max_concurrency,
            policy_slots=policy_slots,
            engine_environment=engine_environment,
            worker_environment=worker_environment,
            ready_timeout=args.ready_timeout,
            shutdown_grace=args.shutdown_grace,
            shutdown_event=shutdown_event,
            shutdown_signal=shutdown_signal,
            device_profile_cache=device_profile_cache,
            device_profile_path=device_profile_path,
            worker_owns_engine=worker_owns_engine,
        )
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
