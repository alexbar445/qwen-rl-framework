"""Internal and grouped CLI for the Learner / Worker runtime."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .config import RuntimeConfig, load_config
from .manager import (RuntimeErrorDetail, request_selftest, start, status, stop,
                      bepa_request, run_teacher, run_training, update_rollout_contract,
                      validate_bepa_environment,
                      validate_environment, validate_paths, request_evaluation)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _role_config_data(config: RuntimeConfig, role: str, *, managed: bool) -> dict[str, Any]:
    data = config.as_dict()
    if not managed:
        if role == "learner":
            # Keep optimizer recovery outside the per-invocation standalone artifact path.
            data["learner"]["recovery_dir"] = str(config.run / "training")
            data["learner"]["recovery_mode"] = "auto"
            data["learner"]["recovery_lineage_id"] = None
        data["run_dir"] = str(config.run / "standalone" / f"{role}-{uuid.uuid4().hex}")
    return data


def _serve_role(config: RuntimeConfig, role: str, worker_id: str | None = None) -> int:
    managed = bool(os.environ.get("QWEN_RL_MANIFEST_RUN_ID"))
    if role == "learner" and not managed and any(
        worker.scope in {"gateway", "service"} for worker in config.workers
    ):
        raise RuntimeErrorDetail(
            "service and Gateway Workers require a manager-resolved active membership snapshot; "
            "start the runtime with `qwen-rl runtime start`"
        )
    config_data = _role_config_data(config, role, managed=managed)
    if role == "learner":
        from ..learner.cuda_allocator import configure_learner_allocator
        configure_learner_allocator()
        os.environ.pop(config.bepa.teacher.api_key_env, None)
        os.environ["CUDA_VISIBLE_DEVICES"] = ("" if config.quantization.mode == "torchao"
                                             or config.learner.parallelism != "single"
                                             else str(config.learner.gpus[0]))
        from ..learner.app import create_learner_app
        import uvicorn

        app = create_learner_app(config_data)
        uvicorn.run(app, host=config.learner.host, port=config.learner.port,
                    workers=1, access_log=False, log_level="info")
        return 0
    if role == "worker":
        worker = next((item for item in config.workers if item.id == worker_id), None)
        if worker is None:
            raise RuntimeErrorDetail(f"unknown worker id: {worker_id}")
        if worker.scope != "local":
            raise RuntimeErrorDetail(
                f"Worker {worker.id} is externally managed and cannot be started by the central process manager"
            )
        os.environ.pop(config.bepa.teacher.api_key_env, None)
        os.environ.pop(config.bepa.teacher.token_env, None)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.gpu)
        from ..worker.app import create_worker_app
        import uvicorn

        worker_config = {
            "worker_id": worker.id,
            "inference": dict(config_data["worker_inference"]),
            "learner_url": config.learner_url,
            "adapter_cache": str(Path(config_data["run_dir"]) / "worker-adapters" / worker.id),
            "telemetry_dir": str(Path(config_data["run_dir"]) / "telemetry" / "workers" / worker.id),
            "lora_rank": config.lora.rank,
            "lora_alpha": config.lora.alpha,
            "request_timeout": config.learner.request_timeout,
            "host": worker.host,
            "port": worker.port,
            "temperature": config.rollout.temperature,
            "top_p": config.rollout.top_p,
            "top_k": config.rollout.top_k,
            "repetition_penalty": config.rollout.repetition_penalty,
            "turn_max_tokens": config.rollout.turn_max_tokens,
        }
        if config.model_backend == "ternary":
            worker_config["artifact_identity_path"] = config.artifact_identity_path
        app = create_worker_app(worker_config)
        uvicorn.run(app, host=worker.host, port=worker.port, workers=1,
                    access_log=False, log_level="info")
        return 0
    raise ValueError(f"unsupported role: {role}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-rl runtime",
        description="Run a four-bit LoRA Learner and a configured HTTP Worker pool.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "stop", "status", "selftest", "learner", "worker", "teacher"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True,
                             help="strict YAML configuration")
        command.add_argument(
            "--environment-file", type=Path,
            help="owned mode-0600 Worker-auth exports for service and Gateway Workers",
        )
        if name == "start":
            command.add_argument(
                "--fresh", action="store_true",
                help="start a new training lineage; existing roles must be stopped first, old checkpoints are preserved",
            )
            command.add_argument("--ready-timeout", type=float, default=900.0,
                                 help="seconds allowed for /health on all configured processes")
            command.add_argument("--selftest-timeout", type=float, default=3600.0,
                                 help="seconds allowed for the real-model identity connectivity self-test")
            command.add_argument(
                "--skip-selftest", action="store_true",
                help=("skip the real-model identity self-test; health and policy ACKs still run, "
                      "but startup will not validate model generation or training mechanics"),
            )
            command.add_argument("--foreground", action="store_true",
                                 help="keep this manager attached; Ctrl-C stops only its manifest processes")
        elif name == "stop":
            command.add_argument("--timeout", type=float, default=15.0,
                                 help="graceful shutdown timeout before escalating signals")
        elif name == "selftest":
            command.add_argument("--timeout", type=float, default=3600.0)
        elif name == "worker":
            command.add_argument("--worker-id", required=True)
        elif name == "teacher":
            command.add_argument("--worker-id", default="teacher-manual")
            command.add_argument("--once", action="store_true",
                                 help="claim at most one job and exit when the queue is empty")
    contract_update = sub.add_parser(
        "update-rollout-contract",
        help="revise only an episode token budget in a verified stopped training snapshot",
    )
    contract_update.add_argument("--config", type=Path, required=True)
    monitor = sub.add_parser("watch", help="read-only live logs, throughput, occupancy, rewards and Teacher status")
    monitor.add_argument("--config", type=Path, required=True)
    monitor.add_argument("--environment-file", type=Path)
    monitor.add_argument("--interval", type=float, default=2.0)
    monitor.add_argument("--window", type=float, default=30.0, help="rolling throughput window in seconds")
    monitor.add_argument("--timeout", type=float, default=3.0)
    monitor.add_argument("--once", action="store_true", help="print one snapshot and exit")
    monitor.add_argument("--json", action="store_true", help="one JSON object per snapshot")
    monitor.add_argument("--details", action="store_true", help="show bounded per-task BEPA diagnostics")
    monitor.add_argument("--log-lines", type=int, default=3, help="recent lines per process log; zero hides logs")
    monitor.add_argument("--no-gpu", action="store_true", help="skip optional nvidia-smi probes")
    monitor.add_argument("--exit-on-stdin-eof", action="store_true",
                         help="terminate this read-only observer when its owning stdin pipe closes")
    evaluate = sub.add_parser('evaluate', help='native heldout evaluation; never updates the learner')
    evaluate.add_argument('--config', type=Path, required=True)
    evaluate.add_argument('--environment-file', type=Path)
    evaluation_action = evaluate.add_mutually_exclusive_group()
    evaluation_action.add_argument('--status', action='store_true', help='read evaluation progress without generating')
    evaluation_action.add_argument('--retry-errors', metavar='REQUEST_ID',
                                   help='retry only infrastructure-error rows from a completed request')
    run = sub.add_parser("run", help="submit a bounded training run to the existing Learner API")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--environment-file", type=Path)
    run.add_argument("--steps", type=int, required=True)
    run.add_argument("--task-ids", nargs="+", default=None)

    bepa = sub.add_parser("bepa", help="manage the Learner's BEPA seed queue")
    bepa_sub = bepa.add_subparsers(dest="bepa_operation", required=True)
    for operation in ("bootstrap", "status", "jobs", "plans", "forget-plan", "budget", "settle-lower-bound"):
        command = bepa_sub.add_parser(operation)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--environment-file", type=Path)
        if operation == "bootstrap":
            command.add_argument("--task-ids", nargs="+", default=None)
        if operation == "forget-plan":
            command.add_argument("--plan-id", required=True)
        if operation == "settle-lower-bound":
            command.add_argument("--job-id", required=True)
            command.add_argument("--evidence-ref", required=True)
            command.add_argument("--reason", required=True)
            command.add_argument("--accept-unreported-usage", action="store_true", required=True)
        if operation == "budget":
            command.add_argument("--max-job-tokens", type=int)
            command.add_argument("--max-total-tokens", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "watch" and args.exit_on_stdin_eof:
            from .stdin_lifetime import terminate_on_stdin_eof
            terminate_on_stdin_eof()
        config_path = args.config.expanduser().resolve()
        # Status before host planning reads metadata only; GPU role entry points
        # keep the mandatory service-membership validation.
        config = (load_config(config_path, for_local_worker_plan=True)
                  if args.command in {"status", "update-rollout-contract"}
                  else load_config(config_path))
        if args.command == "update-rollout-contract":
            result = update_rollout_contract(config, config_path)
            _print(result)
            if result.get("status") == "committed_durability_unconfirmed":
                return 3
            return 0
        worker_environment_names = {
            worker.auth_token_env for worker in config.workers
            if worker.scope in {"service", "gateway"}
        }
        if args.command == "teacher":
            if args.environment_file is not None:
                raise RuntimeErrorDetail(
                    "BEPA Teacher must not load service and Gateway Worker auth bindings"
                )
            # A manually launched Teacher can inherit a shell with Worker auth
            # variables. Keep those credentials outside the Teacher role too.
            for name in worker_environment_names:
                os.environ.pop(name, None)
        else:
            if worker_environment_names and args.environment_file is None:
                raise RuntimeErrorDetail(
                    "service and Gateway Workers require --environment-file with their owned private auth bindings"
                )
            if args.environment_file is not None:
                from ..cluster.runtime_entry import load_runtime_environment

                os.environ.update(load_runtime_environment(
                    args.environment_file, required_names=worker_environment_names,
                ))
        if args.command == "start":
            result = start(config, config_path, ready_timeout=args.ready_timeout,
                           selftest_timeout=args.selftest_timeout, foreground=args.foreground,
                           skip_selftest=args.skip_selftest, fresh=args.fresh,
                           worker_environment_file=args.environment_file)
            _print(result)
            return 0
        if args.command == "stop":
            _print(stop(config, timeout=args.timeout))
            return 0
        if args.command == "watch":
            from .watch import watch
            try:
                asyncio.run(watch(config, interval=args.interval, window=args.window,
                    timeout=args.timeout, once=args.once, json_output=args.json,
                    log_lines=args.log_lines, gpu=not args.no_gpu, details=args.details))
            except KeyboardInterrupt:
                pass
            return 0
        if args.command == "status":
            # Reading the manifest and probing HTTP endpoints never imports a model runtime.
            _print(status(config, config_path=config_path))
            return 0
        if args.command == "selftest":
            _print(request_selftest(config, timeout=args.timeout))
            return 0
        if args.command == "run":
            _print(run_training(config, steps=args.steps, task_ids=args.task_ids))
            return 0
        if args.command == 'evaluate':
            _print(request_evaluation(config, read_only=args.status,
                                      retry_errors=args.retry_errors))
            return 0
        if args.command == "bepa":
            _print(bepa_request(config, args.bepa_operation,
                                task_ids=getattr(args, "task_ids", None),
                                max_job_tokens=getattr(args, "max_job_tokens", None),
                                max_total_tokens=getattr(args, "max_total_tokens", None),
                                job_id=getattr(args, "job_id", None),
                                plan_id=getattr(args, "plan_id", None),
                                evidence_ref=getattr(args, "evidence_ref", None),
                                reason=getattr(args, "reason", None),
                                accept_unreported_usage=getattr(args, "accept_unreported_usage", False)))
            return 0
        if args.command == "teacher":
            if not config.bepa.enabled:
                raise RuntimeErrorDetail("BEPA teacher runner requires bepa.enabled=true")
            validate_bepa_environment(config)
            os.environ.pop("ARC_ENV_TOKEN", None)
            _print(asyncio.run(run_teacher(config, worker_id=args.worker_id, once=args.once)))
            return 0
        if args.command in {"learner", "worker"}:
            validate_paths(config, role=args.command)
            validate_environment(config)
        if args.command == "learner":
            return _serve_role(config, "learner")
        if args.command == "worker":
            return _serve_role(config, "worker", args.worker_id)
        parser.error(f"unknown command: {args.command}")
    except (RuntimeErrorDetail, ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"qwen-rl: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
