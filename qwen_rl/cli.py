"""Lightweight command router for the Qwen RL workspace."""
from __future__ import annotations

import argparse
from importlib import import_module
import sys
from typing import Sequence


_GROUPS = {
    "instance": ("qwen_rl.deployment.instance_cli", "Advanced: profile-managed setup, recovery, monitoring, and shutdown"),
    "vast": ("qwen_rl.deployment.vast.cli", "Advanced: prepare and attach prebuilt Vast roles without renting"),
    "local": ("qwen_rl.deployment.local", "Advanced: explicit host Docker deployment profiles and credential bridge"),
    "learner": ("qwen_rl.deployment.learner.cli", "Portable profile-owned Learner deployment"),
    "runtime": ("qwen_rl.runtime.cli", "Learner, Worker, training, and runtime operations"),
    "cluster": ("qwen_rl.cluster.__main__", "LAN cluster control plane and node operations"),
    "dashboard": ("qwen_rl.dashboard.cli", "Read-only local training dashboard"),
    "gradients": ("qwen_rl.learner.gradient_cli", "Central LoRA gradient server and compute executors"),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the root parser without importing any command implementation."""
    parser = argparse.ArgumentParser(
        prog="qwen-rl",
        description="Distributed reinforcement-learning and cluster tools.",
        epilog=("CPU example: python -m qwen_rl.examples.minimal. "
                "GPU and cluster deployments require explicit backend dependencies and profiles."),
    )
    commands = parser.add_subparsers(dest="group", metavar="GROUP")
    for name, (_module, help_text) in _GROUPS.items():
        commands.add_parser(name, help=help_text, description=help_text)
    return parser


def _dispatch(group: str, argv: Sequence[str]) -> int:
    module_name, _help_text = _GROUPS[group]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_group_modules = {module_name}
        parent = module_name.rpartition(".")[0]
        while parent and parent != "qwen_rl":
            missing_group_modules.add(parent)
            parent = parent.rpartition(".")[0]
        if exc.name in missing_group_modules:
            print(f"qwen-rl: the {group} command group is not included in this installation", file=sys.stderr)
            return 2
        raise
    command = getattr(module, "main", None)
    if not callable(command):
        raise RuntimeError(f"command group {group} has no main(argv) entry")
    result = command(list(argv))
    return 0 if result is None else int(result)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one public group, importing its implementation only on demand."""
    values = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not values or values[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    group = values[0]
    if group not in _GROUPS:
        parser.error(f"invalid choice: {group!r} (choose from {', '.join(_GROUPS)})")
    return _dispatch(group, values[1:])
