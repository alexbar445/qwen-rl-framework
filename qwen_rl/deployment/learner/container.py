"""Learner image entrypoint and private in-container command bridge.

The default entrypoint supervises one runtime-manager manifest as PID 1. The
host control plane may also invoke read-only status/watch and explicit run or
activation operations through ``docker exec``; credentials are read from the
mounted private files inside this process, never passed as command arguments.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import stat
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from .config import (
    LearnerDeployment,
    LearnerDeploymentError,
    check_learner_profile,
    load_learner_deployment,
)

_CREDENTIAL_MAX_BYTES = 16_384


def _container_profile(deployment: LearnerDeployment) -> LearnerDeployment:
    """Translate host source paths to the declared in-container mount paths."""
    raw = deployment.model_dump(mode="json")
    for mount in raw["mounts"].values():
        mount["host_path"] = mount["container_path"]
    for item in raw["resources"]:
        item["host_path"] = item["container_path"]
    for item in raw["credentials"].values():
        item["host_path"] = item["container_path"]
    # The private SSH client identity is host-only and never enters the Learner
    # process namespace or runtime profile passed to training code.
    raw["ssh"] = None
    return load_learner_deployment(
        raw,
        allow_gpu_autodetect=False,
        allow_rootless_container_user=os.environ.get("QWEN_LEARNER_ROOTLESS_CONTAINER_USER") == "1",
    )


def _read_secret_file(path: Path, *, expected_uid: int | None = None) -> str:
    owner_uid = os.getuid() if expected_uid is None else expected_uid
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LearnerDeploymentError(f"credential file is unavailable: {path.name}") from exc
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != owner_uid
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise LearnerDeploymentError(f"credential file has unsafe owner or mode: {path.name}")
        chunks = bytearray()
        while len(chunks) <= _CREDENTIAL_MAX_BYTES:
            chunk = os.read(descriptor, min(4096, _CREDENTIAL_MAX_BYTES + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) > _CREDENTIAL_MAX_BYTES:
            raise LearnerDeploymentError(f"credential file is too large: {path.name}")
        try:
            value = chunks.decode("utf-8").strip()
        except UnicodeError:
            raise LearnerDeploymentError(f"credential file is not UTF-8: {path.name}") from None
        if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise LearnerDeploymentError(f"credential file is empty or invalid: {path.name}")
        return value
    finally:
        os.close(descriptor)


def load_credential_environment(deployment: LearnerDeployment) -> dict[str, str]:
    """Read every profile credential file and return only its allowlisted binding."""
    container_uid = int(deployment.container_user.split(":", 1)[0])
    values: dict[str, str] = {}
    for name, credential in deployment.credentials.items():
        if credential.container_path != f"/run/secrets/{name}":
            raise LearnerDeploymentError(f"credential binding path does not match {name}")
        values[name] = _read_secret_file(credential.host_path, expected_uid=container_uid)
    if "ARC_ENV_TOKEN" not in values:
        raise LearnerDeploymentError("ARC_ENV_TOKEN credential file is required")
    for name, value in values.items():
        if name not in deployment.credentials:
            raise LearnerDeploymentError("credential environment contains an undeclared name")
        os.environ[name] = value
    return values


def _runtime_context(deployment: LearnerDeployment):
    container_deployment = _container_profile(deployment)
    check_learner_profile(
        container_deployment,
        require_assets=True,
        allow_rootless_container_user=os.environ.get("QWEN_LEARNER_ROOTLESS_CONTAINER_USER") == "1",
    )
    load_credential_environment(container_deployment)
    from ...runtime.config import load_config

    runtime_config = load_config(container_deployment.runtime_config.host_path)
    if any(worker.scope == "local" for worker in runtime_config.workers):
        raise LearnerDeploymentError("Learner image cannot launch local Worker processes")
    state_cache = PurePosixPath(container_deployment.state.container_path) / "cache"
    home = PurePosixPath(container_deployment.state.container_path) / "home"
    for directory in (
        state_cache,
        state_cache / "huggingface",
        state_cache / "huggingface" / "datasets",
        state_cache / "huggingface" / "hub",
        state_cache / "huggingface" / "transformers",
        state_cache / "vllm",
        state_cache / "triton",
        state_cache / "torchinductor",
        state_cache / "torch",
        state_cache / "torch-extensions",
        state_cache / "cuda",
        state_cache / "flashinfer",
        home,
    ):
        Path(str(directory)).mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ.update({
        "HOME": str(home),
        "XDG_CACHE_HOME": str(state_cache),
        "HF_HOME": str(state_cache / "huggingface"),
        "HF_DATASETS_CACHE": str(state_cache / "huggingface" / "datasets"),
        "HUGGINGFACE_HUB_CACHE": str(state_cache / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(state_cache / "huggingface" / "transformers"),
        "VLLM_CACHE_ROOT": str(state_cache / "vllm"),
        "TRITON_CACHE_DIR": str(state_cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(state_cache / "torchinductor"),
        "TORCH_HOME": str(state_cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(state_cache / "torch-extensions"),
        "CUDA_CACHE_PATH": str(state_cache / "cuda"),
        "FLASHINFER_WORKSPACE_BASE": str(state_cache / "flashinfer"),
    })
    return container_deployment, runtime_config


def run_container_supervisor(
    config_path: Path,
    runtime_config: Any,
    *,
    runtime_start: Callable[..., dict[str, Any]],
    runtime_status: Callable[..., dict[str, Any]],
    runtime_stop: Callable[..., dict[str, Any]],
    sleep: Callable[[float], None] = time.sleep,
    signal_register: Callable[..., Any] = signal.signal,
    should_wait: Callable[[], bool] | None = None,
) -> int:
    """Supervise the existing process manifest and stop only its recorded roles."""
    received: dict[str, int | None] = {"signal": None}

    def request_shutdown(signum, _frame):
        received["signal"] = signum
        raise KeyboardInterrupt

    prior_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            prior_handlers[signum] = signal_register(signum, request_shutdown)
        except ValueError:
            # Only the main Python thread owns process signal handlers. The
            # real container entrypoint always runs on that thread; an injected
            # test thread simply exercises the lifecycle functions directly.
            continue
    start_succeeded = False
    stopped = False

    def stop_owned_runtime(*, suppress_errors: bool) -> None:
        nonlocal stopped
        if not start_succeeded or stopped:
            return
        stopped = True
        try:
            runtime_stop(runtime_config)
        except BaseException:
            if not suppress_errors:
                raise

    try:
        runtime_start(
            runtime_config,
            config_path,
            skip_selftest=True,
            # Return after READY so this PID 1 loop can catch a single exited
            # role instead of waiting inside manager.start for every role.
            foreground=False,
            fresh=False,
            start_teacher=False,
            resume_interrupted_run=False,
        )
        start_succeeded = True
        if received["signal"] is not None:
            return 0
        wait = should_wait or (lambda: True)
        while wait():
            report = runtime_status(runtime_config, config_path=config_path)
            if report.get("reuse_ready") is not True:
                return 1
            sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        # Once start returns READY, stop its manifest on every exit, including
        # status/probe exceptions. Failed new-run startup is rolled back by the
        # runtime manager; stopping here on a failed call could kill a reused run.
        stop_owned_runtime(suppress_errors=True)
        for signum, handler in prior_handlers.items():
            try:
                signal_register(signum, handler)
            except ValueError:
                pass
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-rl-learner-container")
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--operation", choices=("supervise", "status", "watch", "activate", "run", "stop"),
                        default="supervise")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--with-teacher", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _operation(args: argparse.Namespace) -> int:
    deployment = load_learner_deployment(
        args.deployment,
        allow_rootless_container_user=os.environ.get("QWEN_LEARNER_ROOTLESS_CONTAINER_USER") == "1",
    )
    container_deployment, config = _runtime_context(deployment)
    from ...runtime import manager

    if args.operation == "supervise":
        return run_container_supervisor(
            container_deployment.runtime_config.host_path,
            config,
            runtime_start=manager.start,
            runtime_status=manager.status,
            runtime_stop=manager.stop,
        )
    if args.operation == "status":
        _print(manager.status(config, config_path=container_deployment.runtime_config.host_path))
        return 0
    if args.operation == "stop":
        _print(manager.stop(config))
        return 0
    if args.operation == "watch":
        from ...runtime.watch import watch

        asyncio.run(watch(config, interval=args.interval, timeout=args.timeout,
                          once=args.once, json_output=args.json))
        return 0
    if args.operation == "activate":
        _print(manager.activate_runtime(
            config,
            container_deployment.runtime_config.host_path,
            start_teacher=args.with_teacher,
            resume_interrupted_run=args.resume,
        ))
        return 0
    if args.operation == "run":
        if args.steps is None:
            raise LearnerDeploymentError("run operation requires --steps")
        current = manager.status(config, config_path=container_deployment.runtime_config.host_path)
        recovery = current.get("recovery") if isinstance(current, dict) else None
        pending = recovery.get("pending_run") if isinstance(recovery, dict) else None
        resumed = current.get("resume") if isinstance(current, dict) else None
        if pending not in (None, False) and not isinstance(resumed, dict):
            raise LearnerDeploymentError("pending checkpoint run requires explicit `learner start --resume`")
        if isinstance(resumed, dict) and resumed.get("status") == "running":
            raise LearnerDeploymentError("a resumed logical run is active; wait for it before starting another run")
        manager.activate_runtime(
            config,
            container_deployment.runtime_config.host_path,
            start_teacher=config.bepa.enabled,
            resume_interrupted_run=False,
        )
        _print(manager.run_training(config, steps=args.steps, task_ids=args.task_ids))
        return 0
    raise LearnerDeploymentError("unknown internal Learner operation")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.resume and args.operation != "activate":
            raise LearnerDeploymentError("--resume is only valid for explicit activation")
        if args.with_teacher and args.operation not in {"activate"}:
            raise LearnerDeploymentError("--with-teacher is only valid for explicit activation")
        return _operation(args)
    except KeyboardInterrupt:
        return 0
    except (LearnerDeploymentError, RuntimeError, ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"learner container: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
