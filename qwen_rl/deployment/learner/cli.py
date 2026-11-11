"""Host-only commands for a profile-owned portable Learner container."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .config import LearnerDeploymentError, load_learner_deployment
from .manager import LearnerManager, LearnerManagerError

_DEPLOYMENT_COMMANDS = {"check", "bundle", "start", "status", "watch", "run", "stop"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-rl learner",
        description="Manage one profile-owned Learner container on this host.",
    )
    parser.add_argument("--deployment", type=Path,
                        help="strict qwen-rl-learner-deployment/v1 JSON profile")
    parser.add_argument("--fleet-rootless", action="store_true", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check", help="validate assets, image, GPU UUIDs, and publish port")

    bundle = commands.add_parser("bundle", help="build the private Learner deployment bundle")
    bundle.add_argument("--image-archive", type=Path, required=True,
                        help="previously exported exact Learner Docker image archive")
    bundle.add_argument("--output", type=Path, required=True,
                        help="new destination path for the private bundle")
    bundle.add_argument("--include-state", action="store_true",
                        help="include the explicitly reviewed full persistent state snapshot")

    starter = commands.add_parser("starter", help="print a verified one-paste private installer command")
    starter.add_argument("--installer-url", required=True)
    starter.add_argument("--installer-sha256", required=True)
    starter.add_argument("--bundle-url", required=True)
    starter.add_argument("--bundle-sha256", required=True)
    starter.add_argument("--install-root", type=Path,
                         default=Path("/var/lib/qwen-rl/learner"))

    start = commands.add_parser("start", help="start the Learner and wait for READY plus policy ACKs")
    start.add_argument("--timeout", type=float, default=900.0,
                       help="seconds allowed for the container runtime to become READY")
    start.add_argument("--resume", action="store_true",
                       help="explicitly resume a verified pending full-state Learner run")
    start.add_argument("--with-teacher", action="store_true",
                       help="explicitly start the configured BEPA Teacher after READY")

    commands.add_parser("status", help="show owned container and Learner runtime status")

    watch = commands.add_parser("watch", help="follow the owned container logs")
    watch.add_argument("--tail", type=int, default=100)
    watch.add_argument("--timeout", type=float, default=3600.0)
    watch.add_argument("--no-follow", action="store_true",
                       help="print the requested tail and return")

    run = commands.add_parser("run", help="explicitly run a new bounded Learner training operation")
    run.add_argument("--steps", type=int, required=True)
    run.add_argument("--task-ids", nargs="+",
                     help="optional exact task IDs for a controlled run")
    run.add_argument("--timeout", type=float, default=60.0,
                     help="seconds allowed for the run request to be accepted")

    compute = commands.add_parser(
        "compute",
        help="run a standalone T2 Learner operation without starting Workers or rollouts",
    )
    compute.add_argument("--config", type=Path, required=True,
                         help="strict qwen-rl-ternary-compute/v1 JSON or YAML profile")
    compute.add_argument("--trajectories", type=Path,
                         help="optional JSONL batch of completed trajectories for one update")
    compute.add_argument("--self-smoke", action="store_true",
                         help="sample one local 3-row diagnostic group and apply one update")

    stop = commands.add_parser("stop", help="stop and remove only this profile's owned container")
    stop.add_argument("--timeout", type=float, default=60.0)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _dispatch(args: argparse.Namespace) -> int:
    if args.fleet_rootless and args.command in {"bundle", "starter"}:
        raise LearnerDeploymentError("--fleet-rootless is valid only for current-user runtime management")
    if args.command == "compute":
        from qwen_rl.learner.ternary_compute import run_compute

        _print(run_compute(
            args.config,
            trajectories_path=args.trajectories,
            self_smoke=args.self_smoke,
        ))
        return 0
    if args.command == "starter":
        from .bundle import render_learner_starter_command

        print(render_learner_starter_command(
            installer_url=args.installer_url,
            installer_sha256=args.installer_sha256,
            bundle_url=args.bundle_url,
            bundle_sha256=args.bundle_sha256,
            install_root=args.install_root,
        ))
        return 0
    if args.command == "bundle":
        from .bundle import build_learner_bundle

        bundle = build_learner_bundle(
            Path.cwd(),
            profile=args.deployment,
            image_archive=args.image_archive,
            destination=args.output,
            include_state=args.include_state,
        )
        # The bundle receipt deliberately exposes only output metadata, digest,
        # size, file count, and pinned image ID; it never serializes its manifest.
        _print(bundle.as_dict())
        return 0
    if args.deployment is None:
        raise LearnerDeploymentError(f"{args.command} requires --deployment PROFILE")
    host_uid = host_gid = None
    if args.fleet_rootless:
        from qwen_rl.cluster.fleet.host import current_user_account
        from qwen_rl.cluster.fleet.roles import inspect_docker_access, probe_docker_uid_mapping

        account = current_user_account()
        access = inspect_docker_access()
        if not access.rootless:
            raise LearnerDeploymentError("--fleet-rootless requires an inspected rootless Docker daemon")
        deployment = load_learner_deployment(
            args.deployment,
            allow_rootless_container_user=True,
        )
        if deployment.container_user != "0:0":
            raise LearnerDeploymentError("rootless Learner profile container_user must be exactly 0:0")
        mapping = probe_docker_uid_mapping(
            access,
            image=deployment.image_id,
            account=account,
            probe_dir=deployment.host_control_path / "uid-map-probe",
            python_executable="/work/envs/rl-vllm/bin/python",
        )
        if (mapping.get("rootless") is not True
                or mapping.get("uid_mapping_verified") is not True
                or (mapping.get("container_uid"), mapping.get("container_gid")) != (0, 0)
                or (mapping.get("host_uid"), mapping.get("host_gid")) != (account.uid, account.gid)):
            raise LearnerDeploymentError("rootless Docker bind mapping does not map container 0:0 to this account")
        host_uid, host_gid = account.uid, account.gid
    else:
        deployment = load_learner_deployment(args.deployment)
    if args.command == "check":
        # Profile validation is read-only. Host-specific checks additionally
        # inspect only the exact image ID, selected GPUs, and publish port.
        _print(LearnerManager(
            deployment,
            allow_rootless_container_user=args.fleet_rootless,
            host_uid=host_uid,
            host_gid=host_gid,
        ).check())
        return 0
    manager = LearnerManager(
        deployment,
        allow_rootless_container_user=args.fleet_rootless,
        host_uid=host_uid,
        host_gid=host_gid,
    )
    if args.command == "start":
        _print(manager.start(timeout=args.timeout, resume=args.resume,
                             with_teacher=args.with_teacher))
    elif args.command == "status":
        _print(manager.status())
    elif args.command == "watch":
        manager.watch(tail=args.tail, follow=not args.no_follow, timeout=args.timeout)
    elif args.command == "run":
        _print(manager.run(steps=args.steps, task_ids=args.task_ids,
                           timeout=args.timeout))
    elif args.command == "stop":
        _print(manager.stop(timeout=args.timeout))
    else:  # pragma: no cover - argparse requires one of the listed commands.
        raise LearnerDeploymentError("unknown Learner command")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in _DEPLOYMENT_COMMANDS and args.deployment is None:
        parser.error(f"{args.command} requires --deployment PROFILE")
    try:
        return _dispatch(args)
    except (LearnerDeploymentError, LearnerManagerError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"qwen-rl learner: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
