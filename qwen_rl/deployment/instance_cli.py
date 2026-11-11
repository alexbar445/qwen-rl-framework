"""CLI for one local or Vast multi-process deployment instance."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Sequence


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated nonnegative integers") from exc
    if not parts or any(gpu < 0 for gpu in parts) or len(set(parts)) != len(parts):
        raise argparse.ArgumentTypeError("GPU IDs must be unique nonnegative integers")
    return parts


def _assignment(value: str) -> tuple[str, str]:
    key, separator, path = value.partition("=")
    if not separator or not key or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH with nonempty values")
    return key, path


def _assignments(values, parser, flag):
    result = {}
    for name, value in values:
        if name in result:
            parser.error(flag + " entries must have unique names")
        result[name] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-rl instance",
        description="Set up, check, and operate one RL instance on Docker, Vast, or AutoDL.",
        epilog="First use: setup --interactive --output-dir /path/to/config. Then use one deployment.json for doctor, serve, watch, and stop-now or stop-after-checkpoint.",
    )
    parser.add_argument("--deployment", type=Path, help="deployment JSON path; required except for setup")
    parser.add_argument(
        "--gpus",
        type=_parse_gpus,
        help="override deployment GPU IDs, for example 0,1,2,3",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="generate a local deployment from prepared model/data paths")
    setup.add_argument("--runtime-config", type=Path,
                       default=Path(__file__).resolve().parents[2] / "configs/local-bonsai2-async-bepa.yaml",
                       help="source runtime YAML; defaults to the Bonsai2 template")
    setup.add_argument("--output-dir", required=True, type=Path, help="new directory for runtime.yaml and deployment.json")
    setup.add_argument("--model-path", type=Path)
    setup.add_argument("--dataset-dir", type=Path)
    setup.add_argument("--artifact-identity", type=Path, help="prepared artifact identity file; never rewritten")
    setup.add_argument("--conversion-report", type=Path, help="prepared model conversion report")
    setup.add_argument("--state-dir", type=Path, help="supervisor state directory; does not replace checkpoint recovery_dir")
    setup.add_argument("--cache-root", type=Path, help="explicit location for compiler and runtime caches")
    setup.add_argument("--packed-cache-root", type=Path,
                       help="explicit packed-weight cache location; defaults to a deployment-scoped /dev/shm path")
    setup.add_argument("--python", type=Path, help="installed runtime Python")
    setup.add_argument("--native-binary", type=Path, help="installed ninfer-serve")
    setup.add_argument("--tasks-per-worker", type=int, help="explicit future tasks per Worker override")
    setup.add_argument("--trajectories-per-group", type=int, help="explicit future native trajectories per group override")
    setup.add_argument("--install-root", type=Path, help="runtime source directory")
    setup.add_argument("--credential", action="append", type=_assignment, default=[], metavar="ENV=PATH",
                       help="credential file path, never a credential value")
    setup.add_argument("--teacher-base-url", help="reachable Teacher provider or bridge URL")
    setup.add_argument("--path-map", action="append", type=_assignment, default=[], metavar="OLD=NEW",
                       help="explicit mapping for known runtime paths; does not rewrite identity documents")
    setup.add_argument("--interactive", action="store_true", help="prompt for missing inputs using template defaults")
    doctor = commands.add_parser("doctor", help="read-only configuration, prepared-input, and instance diagnosis")
    doctor.add_argument("--json", action="store_true", help="print the diagnostic report as JSON")
    resume = commands.add_parser("resume-plan", help="inspect recovery compatibility before starting roles")
    resume.add_argument("--json", action="store_true", help="print the machine-readable recovery plan")
    resume.add_argument("--inspect-checkpoint", action="store_true",
                        help="explicitly inspect trusted checkpoint metadata on CPU without restoring or changing it")
    check = commands.add_parser("check", help="validate the deployment and print its startup plan")
    check_format = check.add_mutually_exclusive_group()
    check_format.add_argument("--summary", action="store_true", help="print only the read-only operator summary as JSON")
    check_format.add_argument("--human", action="store_true", help="show the existing operator summary as readable text")
    serve = commands.add_parser("serve", help="run the full instance in the foreground")
    serve.add_argument("--preflight-receipt", type=Path, help=argparse.SUPPRESS)
    execute_resume = commands.add_parser("resume", help="preflight, recover, start or reuse the instance, and verify readiness")
    execute_resume.add_argument("--wait", choices=("ready", "update"), default="ready")
    execute_resume.add_argument("--timeout", type=float, help="readiness observation timeout in seconds; timeout leaves the instance supervised")
    execute_resume.add_argument("--json", action="store_true")
    relocate = commands.add_parser("relocate", help="relocate managed state/cache while reusing the fixed model installation")
    relocate.add_argument("--data-root", type=Path, required=True)
    relocate.add_argument("--cache-root", type=Path, required=True)
    relocate.add_argument("--resume", action="store_true", help="resume the original workload after target CPU validation")
    relocate.add_argument("--cleanup-source", action="store_true", help="remove only verified unreferenced source runtime files after target validation")
    relocate.add_argument("--timeout", type=float, help="target readiness/progress observation budget, excluding CPU and copy stages")
    relocate.add_argument("--json", action="store_true")
    cache = commands.add_parser("cache", help="inspect storage usage, references, peak space and safe cleanup plan without deletion")
    cache.add_argument("--json", action="store_true")
    migrate = commands.add_parser("migrate", help="stage, export, transfer, and import a stopped instance migration")
    migration_commands = migrate.add_subparsers(dest="migration_command", required=True)
    migration_prepare = migration_commands.add_parser("prepare", help="pre-stage immutable inputs without stopping the source")
    migration_prepare.add_argument("--output", type=Path, required=True)
    migration_export = migration_commands.add_parser("export", help="capture latest full state after the source has stopped")
    migration_export.add_argument("--output", type=Path, required=True)
    migration_export.add_argument("--refresh", action="store_true", help="recapture all stopped durable state before any target claim")
    migration_export.add_argument("--stop", action="store_true", help="explicitly stop this owned source before exporting")
    migration_import = migration_commands.add_parser("import", help="verify and install an exported migration into an independent target directory")
    migration_import.add_argument("--bundle", type=Path, required=True)
    migration_import.add_argument("--output-dir", type=Path, required=True)
    migration_import.add_argument("--path-map", action="append", type=_assignment, default=[], metavar="OLD=NEW")
    migration_transfer = migration_commands.add_parser("transfer", help="resume SSH transfer of an existing staged or exported bundle")
    migration_transfer.add_argument("--bundle", type=Path, required=True)
    migration_transfer.add_argument("--to", required=True, help="known SSH host and absolute destination: user@host:/path")
    migration_transfer.add_argument("--ssh-port", type=int, help="override the SSH-configured port")
    migration_transfer.add_argument("--identity-file", type=Path)
    migration_cancel = migration_commands.add_parser("cancel", help="release a source migration fence only after confirming the destination is stopped")
    migration_cancel.add_argument("--acknowledge-destination-stopped", action="store_true", required=True)
    for migration_parser in (migration_prepare, migration_export, migration_import, migration_transfer):
        migration_parser.add_argument("--json", action="store_true")
    status = commands.add_parser("status", help="show instance status")
    status_format = status.add_mutually_exclusive_group()
    status_format.add_argument("--summary", action="store_true", help="print only the read-only operator summary as JSON")
    status_format.add_argument("--human", action="store_true", help="show the existing operator summary as readable text")
    commands.add_parser("stop-now", help="stop owned roles now and discard uncommitted staged work")
    stop_after = commands.add_parser("stop-after-checkpoint", help="stop after the next completed durable update")
    stop_after.add_argument("--timeout", type=float, default=30.0,
                            help="how long this CLI waits; the persisted Learner request continues after timeout")
    run = commands.add_parser("run", help="submit a bounded training run")
    run.add_argument("--steps", required=True, type=int, help="number of training steps")
    evaluate = commands.add_parser("evaluate", help="submit or inspect native evaluation")
    evaluate.add_argument("--status", action="store_true", help="read evaluation status without submitting")
    watch = commands.add_parser("watch", help="observe the running instance")
    watch.add_argument("--details", action="store_true")
    watch.add_argument("--json", action="store_true", help="print JSON snapshots")
    watch.add_argument("--once", action="store_true", help="print one snapshot and exit")
    watch.add_argument("--interval", type=float, help="seconds between snapshots")
    watch.add_argument("--log-lines", type=int, help="recent lines per process log; zero hides logs")
    return parser


def _load_api():
    """Import the deployment implementation only when an operation is requested."""
    from .instance import InstanceDeploymentError, InstanceSupervisor, load_deployment

    return load_deployment, InstanceSupervisor, InstanceDeploymentError


def _print_json(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _report_failure(error: Exception, *, operation: str, deployment_path: Path | None,
                    json_output: bool = False) -> int:
    from .instance_diagnostics import safe_failure

    if isinstance(getattr(error, "report", None), dict) and error.report.get("schema") == "qwen-rl-instance-stop/v1":
        report = error.report
        _print_json(report)
        return 2
    if type(error).__module__ == "qwen_rl.deployment.instance_relocate" and isinstance(getattr(error, "report", None), dict):
        _print_json(error.report)
        return 1
    if type(error).__module__ == "qwen_rl.deployment.instance_preflight" and isinstance(getattr(error, "report", None), dict):
        report = error.report
        if json_output:
            _print_json(report)
        else:
            for issue in report.get("errors", []):
                print(f"{issue.get('stage', 'preflight')}: {issue.get('message', 'startup check failed')}", file=sys.stderr)
                if issue.get("field"):
                    print("Field: " + issue["field"], file=sys.stderr)
                if issue.get("next_command"):
                    print("Next: " + issue["next_command"], file=sys.stderr)
        return 1
    if type(error).__module__ == "qwen_rl.deployment.instance_migration" and type(error).__name__ == "InstanceMigrationError":
        failure = {"status": "blocked", "reason_code": "migration_validation_failed",
                   "message": str(error),
                   "next_command": (f"qwen-rl instance --deployment {deployment_path} status --human"
                                    if deployment_path is not None else "qwen-rl instance migrate transfer --help")}
        _print_json(failure) if json_output else print(f"Migration blocked: {failure['message']}\nNext: {failure['next_command']}", file=sys.stderr)
        return 1
    failure = safe_failure(error, operation=operation, deployment_path=deployment_path)
    if json_output:
        _print_json(failure)
    else:
        print(f"qwen-rl instance [{failure['reason_code']}]: {failure['message']}", file=sys.stderr)
        if failure.get("field"):
            print("Field: " + str(failure["field"]), file=sys.stderr)
        if failure.get("next_command"):
            print("Next: " + str(failure["next_command"]), file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    needs_deployment = args.command != "setup" and not (
        args.command == "migrate" and args.migration_command == "transfer")
    if needs_deployment and args.deployment is None:
        parser.error("--deployment is required for " + args.command)
    if args.command == "setup" and args.deployment is not None:
        parser.error("setup uses --runtime-config and --output-dir; --deployment selects an existing instance")

    try:
        if args.command == "setup":
            from .instance_setup import setup_instance

            credentials = _assignments(args.credential, parser, "--credential")
            mappings = _assignments(args.path_map, parser, "--path-map")
            result = setup_instance(
                runtime_config=args.runtime_config, output_dir=args.output_dir, gpus=args.gpus,
                model_path=args.model_path, dataset_dir=args.dataset_dir,
                artifact_identity_path=args.artifact_identity, conversion_report=args.conversion_report,
                state_dir=args.state_dir, cache_root=args.cache_root,
                packed_cache_root=args.packed_cache_root, python=args.python, native_binary=args.native_binary,
                install_root=args.install_root, credential_paths=credentials,
                teacher_base_url=args.teacher_base_url, path_mappings=mappings,
                interactive=args.interactive, tasks_per_worker=args.tasks_per_worker,
                trajectories_per_group=args.trajectories_per_group,
            )
            _print_json(result)
            return 0
        if args.command == "doctor":
            from .instance_diagnostics import diagnose_instance, format_diagnostics

            result = diagnose_instance(args.deployment, gpus=args.gpus)
            if args.json:
                _print_json(result)
            else:
                print(format_diagnostics(result))
            return 1 if result["status"] == "blocked" else 0
        if args.command == "resume-plan":
            from .instance_resume import build_resume_plan, format_resume_plan

            result = build_resume_plan(args.deployment, gpus=args.gpus,
                                       inspect_checkpoint=args.inspect_checkpoint)
            _print_json(result) if args.json else print(format_resume_plan(result))
            return {"passed": 0, "blocked": 1, "incomplete": 2}.get(result["status"], 2)
        if args.command == "migrate" and args.migration_command == "transfer":
            from .instance_transfer import transfer_migration
            result = transfer_migration(args.bundle, args.to, ssh_port=args.ssh_port,
                                        identity_file=args.identity_file)
            _print_json(result)
            return 0 if result["status"] == "transferred" else 1
        load_deployment, InstanceSupervisor, _InstanceDeploymentError = _load_api()
        deployment = load_deployment(args.deployment)
        if args.gpus is not None:
            deployment = replace(deployment, gpus=args.gpus)
        supervisor = InstanceSupervisor(deployment)

        if args.command == "relocate":
            from .instance_relocate import relocate_instance
            def relocation_progress(event):
                print(f"[{event.get('elapsed_seconds', 0):.1f}s] {event.get('stage')}: {event.get('status', 'running')}", file=sys.stderr, flush=True)
            result = relocate_instance(supervisor, data_root=args.data_root, cache_root=args.cache_root,
                                       resume=args.resume, cleanup_source=args.cleanup_source,
                                       timeout_seconds=args.timeout, progress=relocation_progress)
            if args.json:
                _print_json(result)
            else:
                print(f"Relocation: {result.get('status')}; stage: {result.get('current_stage', 'validation')}")
                if result.get("target_deployment"):
                    print("Target deployment: " + result["target_deployment"])
                if result.get("failure"):
                    _print_json(result["failure"])
                for key in ("storage", "online_preflight_failure", "target_cpu_failure"):
                    detail = result.get(key)
                    if isinstance(detail, dict) and detail.get("status") == "blocked":
                        _print_json({key: detail})
                if result.get("next_command"):
                    print("Next: " + result["next_command"])
            return {"passed": 0, "prepared": 0, "pending": 2}.get(result.get("status"), 1)
        if args.command == "cache":
            from .instance_storage import preflight_storage
            plan = supervisor.prepare()
            result = preflight_storage(deployment, plan, plan["runtime"])
            if args.json:
                _print_json(result)
            else:
                print("Storage: " + result.get("status", "unknown"))
                for row in result.get("filesystems", []):
                    print(f"{row.get('mount_path', row.get('path', 'filesystem'))}: "
                          f"free {row.get('free_bytes', 0):,} bytes; required {row.get('required_free_bytes', 0):,} bytes")
                for row in result.get("inventory", []):
                    print(f"{row.get('category')}: {row.get('path')} ({row.get('allocated_bytes', 0):,} bytes; {row.get('reference_status', 'unknown')})")
                if result.get("errors"):
                    _print_json(result["errors"])
            return 0 if result.get("status") == "passed" else 1
        if args.command == "resume":
            from .instance_operations import resume_instance

            def progress(event):
                detail = ""
                if event.get("teacher_provider_status") == "not_performed":
                    detail = "; Teacher catalog not_performed while local roles are pending"
                print(f"[{event['elapsed_seconds']:.1f}s] {event['stage']}: {event['status']}" + detail,
                      file=sys.stderr, flush=True)

            result = resume_instance(supervisor, wait=args.wait, timeout_seconds=args.timeout, progress=progress)
            if args.json:
                _print_json(result)
            else:
                print("Resume: " + result["status"] + (" (existing instance)" if result.get("reused") else ""))
                observation = result.get("observation", {})
                if observation:
                    print(f"Policy: {observation.get('policy_version')}; run: {observation.get('run_id')}; "
                          f"optimizer progress observed: {result.get('optimizer_update_observed', False)}")
                if result.get("reason_code"):
                    print("Reason: " + result["reason_code"])
                if result.get("status") != "passed":
                    _print_json({key: result[key] for key in ("preflight", "recovery", "detail", "diagnosis", "failure_locations", "supervisor_log") if key in result})
                    print("Next: " + result["next_command"])
            return {"passed": 0, "pending": 2}.get(result.get("status"), 1)
        if args.command == "migrate":
            from . import instance_migration as migration

            last_migration_progress = [None, 0.0]
            def migration_progress(event):
                stage = event.get("stage", "migration")
                now = time.monotonic()
                if stage == last_migration_progress[0] and now - last_migration_progress[1] < 2:
                    return
                last_migration_progress[:] = [stage, now]
                fields = " ".join(f"{name}={event[name]}" for name in (
                    "files_done", "current_file_bytes", "current_file_total_bytes", "bytes_total",
                    "owned_roles_remaining", "remaining_timeout_seconds"
                ) if name in event)
                print(stage + (": " + fields if fields else ""), file=sys.stderr, flush=True)

            if args.migration_command == "prepare":
                result = migration.prepare_instance_migration(deployment, args.output, progress=migration_progress)
            elif args.migration_command == "export":
                result = migration.export_instance_migration(deployment, args.output, stop=args.stop, refresh=args.refresh, progress=migration_progress)
            elif args.migration_command == "import":
                mappings = _assignments(args.path_map, parser, "--path-map")
                result = migration.import_instance_migration(args.bundle, args.output_dir,
                                                             deployment=deployment, path_mappings=mappings,
                                                             progress=migration_progress)
            elif args.migration_command == "cancel":
                result = migration.cancel_instance_migration(
                    deployment, acknowledge_destination_stopped=args.acknowledge_destination_stopped)
            _print_json(result)
            return 0 if result.get("status") in {"passed", "prepared", "exported", "imported", "claimed", "cancelled", "no_migration_fence"} else 1

        if args.command == "check":
            result = supervisor.prepare()
            if args.human:
                from .instance_diagnostics import format_operator_summary
                print(format_operator_summary(result["operator"], deployment_path=args.deployment))
                if (result.get("status") in {"failed", "stop_incomplete", "orphaned"}
                        or result.get("error") or result.get("error_message")):
                    from .instance_failure import failure_diagnosis
                    _print_json(failure_diagnosis(supervisor, result))
            else:
                _print_json(result["operator"] if args.summary else result)
            return 0
        if args.command == "serve":
            return int(supervisor.serve(preflight_receipt=args.preflight_receipt)
                       if args.preflight_receipt is not None else supervisor.serve())
        if args.command == "status":
            result = supervisor.status()
            if args.human:
                from .instance_diagnostics import format_operator_summary
                print(format_operator_summary(result["operator"], deployment_path=args.deployment))
                if (result.get("status") in {"failed", "stop_incomplete", "orphaned"}
                        or result.get("error") or result.get("error_message")):
                    from .instance_failure import failure_diagnosis
                    _print_json(failure_diagnosis(supervisor, result))
            else:
                _print_json(result["operator"] if args.summary else result)
            return 0
        if args.command == "stop-now":
            def stop_progress(event):
                print(f"stop-now: {event.get('stage', 'stopping')}", file=sys.stderr, flush=True)
            _print_json(supervisor.stop_now(progress=stop_progress))
            return 0
        if args.command == "stop-after-checkpoint":
            def stop_progress(event):
                target = event.get("target_update")
                suffix = f"; target update {target}" if target is not None else ""
                print(f"stop-after-checkpoint: {event.get('status', 'pending')}{suffix}", file=sys.stderr, flush=True)
            result = supervisor.stop_after_checkpoint(timeout_seconds=args.timeout, progress=stop_progress)
            _print_json(result)
            return {"completed": 0, "pending": 2}.get(result.get("manager_status"), 1)
        if args.command == "run":
            _print_json(supervisor.run(args.steps))
            return 0
        if args.command == "evaluate":
            _print_json(supervisor.evaluate(status_only=args.status))
            return 0
        if args.command == "watch":
            arguments: list[str] = []
            for flag in ("details", "json", "once"):
                if getattr(args, flag):
                    arguments.append(f"--{flag}")
            if args.interval is not None:
                arguments.extend(("--interval", str(args.interval)))
            if args.log_lines is not None:
                arguments.extend(("--log-lines", str(args.log_lines)))
            return int(supervisor.watch(arguments))
    except Exception as error:
        return _report_failure(error, operation=args.command, deployment_path=args.deployment,
                               json_output=getattr(args, "json", False))

    return _report_failure(RuntimeError(), operation=args.command, deployment_path=args.deployment)


if __name__ == "__main__":
    raise SystemExit(main())
