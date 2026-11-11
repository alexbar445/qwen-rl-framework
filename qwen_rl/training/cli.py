"""One normal-user entry point for local training and complete-state resume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_training_config


def _steps(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("steps must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qwen-rl train",
        description="Train a local small language model through the shared RL core.",
        epilog="--steps is the total committed-update target, not an additional count. Resume never resets optimizer state.",
    )
    parser.add_argument("--config", type=Path, help="training YAML; required for a new run")
    parser.add_argument("--run-dir", type=Path, required=True, help="new run directory, or the existing directory with --resume")
    parser.add_argument("--steps", type=_steps, help="total committed-update target; required for a new run")
    parser.add_argument("--resume", action="store_true", help="load the complete checkpoint and recorded configuration")
    args = parser.parse_args(argv)
    if not args.resume and (args.config is None or args.steps is None):
        parser.error("a new run requires --config and --steps")
    try:
        config = load_training_config(args.config) if args.config is not None else None
        from .runner import train

        result = train(config, args.run_dir, steps=args.steps, resume=args.resume)
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "transformers", "peft", "safetensors", "accelerate"}:
            parser.exit(2, "qwen-rl train: install the small-model optional dependencies in this environment; see examples/small_model/README.md\n")
        parser.exit(2, f"qwen-rl train: cannot import Python module {exc.name!r}; check the selected dependencies and reward module\n")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"qwen-rl train: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
