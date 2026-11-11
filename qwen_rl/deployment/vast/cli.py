"""Role-explicit host CLI for CPU preparation and SSH attach to Vast instances."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import VastDeploymentError
from .prepare import (
    check_prepared_deployment,
    check_vast_profile,
    prepare_vast_deployment,
)

_ROLES = ("worker", "learner")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-rl vast",
        description="Prepare or inspect an explicit Vast Worker/Learner deployment; no create/start API calls are exposed.",
    )
    roles = parser.add_subparsers(dest="role", required=True, metavar="ROLE")
    for role in _ROLES:
        role_parser = roles.add_parser(role, help=f"prepare or attach the {role.title()} role")
        actions = role_parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

        prepare = actions.add_parser("prepare", help="validate inputs and build a private deployment package")
        prepare.add_argument("--config", type=Path, required=True,
                             help=f"strict qwen-rl Vast {role.title()} profile JSON")

        check = actions.add_parser("check", help="check a profile or an already prepared package")
        target = check.add_mutually_exclusive_group(required=True)
        target.add_argument("--config", type=Path, help=f"strict {role.title()} profile to check without writing")
        target.add_argument("--deployment-dir", type=Path, help="prepared private package to verify")

        actions.add_parser("attach", help="attach to an already running instance over verified SSH")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "prepare":
        _print(prepare_vast_deployment(args.config, expected_role=args.role))
        return 0
    if args.command == "check":
        result = (check_vast_profile(args.config, expected_role=args.role) if args.config is not None
                  else check_prepared_deployment(args.deployment_dir, expected_role=args.role))
        _print(result)
        return 0
    raise VastDeploymentError("attach dispatch must use the role-specific SSH helper")  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) >= 2 and values[0] in _ROLES and values[1] == "attach":
        try:
            from .remote import main as remote_main
        except ImportError as exc:
            raise SystemExit(f"qwen-rl vast: SSH attach helper is not installed: {exc}") from exc
        return int(remote_main(values[2:], expected_role=values[0]) or 0)
    parser = build_parser()
    args = parser.parse_args(values)
    try:
        return _dispatch(args)
    except (VastDeploymentError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"qwen-rl vast: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
