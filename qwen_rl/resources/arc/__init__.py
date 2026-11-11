"""ARC-only text resources.

The framework package bundles its original prompt rules here. ARC runtime code,
toolkit identity contracts, task data, and verifiers must be supplied separately.
"""
from importlib.resources import files


def read_system_rules() -> str:
    """Read the bundled ARC prompt rules from an installed package."""
    return files(__package__).joinpath("arc-system-rules.txt").read_text(encoding="utf-8")


__all__ = ["read_system_rules"]
