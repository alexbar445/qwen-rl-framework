"""Atomic per-run manifests and Linux process identity checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return data


_TERMINAL_PROCESS_STATES = {"Z", "X", "x"}


def process_state(pid: int) -> str | None:
    """Return the Linux `/proc` state, with an unknown sentinel for unreadable live PIDs."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError:
        # Permission or transient read failures must not make a potentially live
        # foreign PID look exited to a manager that might start on the same ports.
        return "?"
    close = stat.rfind(")")
    if close < 0:
        return "?"
    tail = stat[close + 1:].split()
    if not tail or len(tail[0]) != 1:
        return "?"
    return tail[0]


def process_is_running(pid: int) -> bool:
    """Treat zombies and exited tasks as gone, but unreadable identities as possibly live."""
    state = process_state(pid)
    return state is not None and state not in _TERMINAL_PROCESS_STATES


def process_starttime(pid: int) -> str | None:
    """Return Linux /proc starttime ticks, or None when the process no longer exists."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    # The comm field is parenthesized and may itself contain spaces or ')' characters.
    close = stat.rfind(")")
    if close < 0:
        return None
    tail = stat[close + 1:].split()  # starts at field 3 (state)
    if len(tail) <= 19:
        return None
    return tail[19]  # field 22: starttime


def process_cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    if not raw:
        return None
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def process_matches(record: dict[str, Any]) -> bool:
    """Require PID, starttime and the recorded role/config identity before signalling."""
    pid = record.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    state = process_state(pid)
    if state is None or state in _TERMINAL_PROCESS_STATES:
        return False
    if process_starttime(pid) != str(record.get("starttime", "")):
        return False
    args = process_cmdline(pid)
    if not args:
        return False
    expected = record.get("identity_args")
    if not isinstance(expected, list) or any(not isinstance(item, str) for item in expected):
        return False
    # Each identity token must be present as an exact argv element. This prevents matching a
    # reused PID whose command merely contains a substring of this run's command.
    return all(token in args for token in expected)


def live_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    processes = manifest.get("processes", [])
    if not isinstance(processes, list):
        return []
    return [record for record in processes
            if isinstance(record, dict) and process_matches(record)]


def active_manifest_path(run_root: Path) -> Path | None:
    pointer = run_root / "active.json"
    if not pointer.exists():
        return None
    data = read_json(pointer)
    relative = data.get("manifest")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid active run pointer: {pointer}")
    candidate = (run_root / relative).resolve()
    root = run_root.resolve()
    if root not in candidate.parents:
        raise ValueError("active run manifest must remain below run_dir")
    return candidate
