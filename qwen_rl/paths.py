"""Locate the checked-out workspace on each node, including non-editable installs."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("ARC_RL_ROOT", Path(__file__).resolve().parents[1])).resolve()
