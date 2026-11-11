"""Read-only collection layer: five sources -> plain JSON-serialisable dicts.

Design rules kept deliberately strict, because a live training process owns these files:

* SQLite is opened with ``file:...?mode=ro`` + ``PRAGMA query_only`` so the trainer keeps
  its write lock and we can never corrupt a ledger mid-update.
* Every path that comes from a URL is validated against an allow-list root before use.
* Nothing here writes, creates or deletes anything inside the run tree.
* Missing data is reported as ``None`` with a reason, never as a fabricated zero -- the
  watch contract makes the same promise and the UI renders ``—`` for it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .watch_stream import LocalWatchStream

from .sources import (
    DEFAULT_HISTORY_SAMPLES, LiveSampler, PROJECT_ROOT as SOURCE_PROJECT_ROOT,
    learner_watch_snapshot, resolve_source, runtime_paths, runtime_watch_snapshot, safe_run_id,
    safe_runtime_document, sanitized_runtime,
)

SCHEMA = "qwen-rl-dashboard-v2"
PROJECT_ROOT = SOURCE_PROJECT_ROOT
_LOCAL_COMMAND_ENV = ("PATH", "HOME", "LANG", "LC_ALL")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_RECORDS = 500
MAX_TRAJECTORIES = 400
MAX_RUN_ROWS = 1000
MAX_LEDGER_ROWS = 1000
_SHA_RE = re.compile(r"^[0-9a-f]{6,128}$")


def _read_json(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> Any | None:
    """Read a bounded regular JSON file; symlinks and oversized payloads stay unavailable."""
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            return None
        with path.open("rb") as handle:
            blob = handle.read(max_bytes + 1)
        if len(blob) > max_bytes:
            return None
        return json.loads(blob.decode("utf-8", "replace"))
    except Exception:
        return None

def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _stat(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        return {"bytes": info.st_size, "mtime": info.st_mtime, "exists": True}
    except OSError:
        return {"bytes": None, "mtime": None, "exists": False}


def _tail(path: Path, limit: int) -> list[str]:
    """Return the last ``limit`` lines without loading the whole file."""
    if limit <= 0:
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    window = min(size, max(64_000, limit * 400))
    try:
        with path.open("rb") as handle:
            handle.seek(size - window)
            blob = handle.read(window)
    except OSError:
        return []
    text = blob.decode("utf-8", "replace")
    lines = text.splitlines()
    if window < size and lines:
        lines = lines[1:]  # drop the partially read first line
    return lines[-limit:]


# Credential shapes that may be echoed by a command line in a log file.  Only
# value-shaped matches are replaced, so adapter SHAs and other identifiers stay readable.
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+/=]{8,}")
_ASSIGNED = re.compile(
    r"(?i)\b([A-Za-z0-9_]*(?:token|api[_-]?key|secret|password)[A-Za-z0-9_]*)\s*[:=]\s*[\"']?[^\s\"',;}]+")


def _redact(line: str) -> str:
    """Strip credential-looking values before log text leaves this process."""
    line = _BEARER.sub("Bearer ***", line)
    return _ASSIGNED.sub(lambda match: match.group(1) + "=***", line)


def _connect_ro(db: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
    except sqlite3.DatabaseError:
        pass
    return connection


def _rows(connection: sqlite3.Connection, sql: str, args: Iterable[Any] = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in connection.execute(sql, tuple(args))]
    except sqlite3.DatabaseError:
        return []


def _sha_ok(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA_RE.fullmatch(value))


def _limited_rows(connection: sqlite3.Connection, query: str,
                  args: Iterable[Any] = ()) -> tuple[list[dict[str, Any]], bool]:
    try:
        rows = [dict(row) for row in connection.execute(
            query + " LIMIT ?", (*tuple(args), MAX_LEDGER_ROWS + 1)
        )]
    except sqlite3.DatabaseError:
        return [], False
    truncated = len(rows) > MAX_LEDGER_ROWS
    return rows[:MAX_LEDGER_ROWS], truncated


class Collector:
    """Read-only view over one deployment's ``outputs/<run-root>`` tree."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        source: str | None = None,
        deployment: Path | str | None = None,
        bepa_store: Path | str | None = None,
        instance_manifest: Path | str | None = None,
        live: bool = True,
    ) -> None:
        if source is None:
            source = "offline" if root is not None and deployment is None and not live else "local"
        self.source = resolve_source(source, deployment=deployment, root=root, bepa_store=bepa_store,
                                     instance_manifest=instance_manifest)
        self.root = Path(root).expanduser().resolve(strict=False) if root is not None else self.source.root
        self.deployment = self.source.deployment_path
        self._deployment_explicit = deployment is not None
        self.live = bool(live and self.source.kind != "offline")
        self.live_sampler = LiveSampler()
        self._watch_stream = None
        self._watch_stream_key = None
        self._watch_stream_lock = threading.Lock()
        self._source_resolution_error = self.source.resolution_error
        self._instance_generation_id: str | None = None
        self._instance_logical_run_id: str | None = None

    @property
    def runs_dir(self) -> Path:
        return (self.source.runtime_registry_root or self.root) / "runs"

    def run_dirs(self) -> list[Path]:
        try:
            base = self.runs_dir.resolve(strict=False)
            entries = []
            for path in self.runs_dir.iterdir():
                run_id = safe_run_id(path.name)
                if run_id is None or path.is_symlink() or not (path / "manifest.json").is_file():
                    continue
                resolved = path.resolve(strict=False)
                try:
                    resolved.relative_to(base)
                except ValueError:
                    continue
                entries.append(path)
        except OSError:
            return []
        return sorted(entries, key=_mtime, reverse=True)[:500]

    def active_run_id(self) -> str | None:
        if self.source.instance_manifest_path is not None:
            from .central import current_instance
            control = current_instance(self.source)
            generation_id = control.get("run_id") if control else None
            if generation_id != self._instance_generation_id:
                self._instance_generation_id = generation_id
                self._instance_logical_run_id = None
            return self._instance_logical_run_id
        if self.source.runtime_registry_root is not None:
            from .central import current_control
            control = current_control(self.source)
            return control.get("run_id") if control else None
        pointer = _read_json(self.root / "active.json") or {}
        value = safe_run_id(pointer.get("run_id"))
        manifest_ref = pointer.get("manifest")
        if value is None or not isinstance(manifest_ref, str):
            return None
        if Path(manifest_ref).parts != ("runs", value, "manifest.json"):
            return None
        manifest = self._read_run_manifest(value)
        if (manifest is None or manifest.get("run_id") != value
                or not self.source.matches_profile(manifest, value)):
            return None
        return value

    def runs(self) -> list[dict[str, Any]]:
        active = self.active_run_id()
        out: list[dict[str, Any]] = []
        for path in self.run_dirs():
            manifest = _read_json(path / "manifest.json") or {}
            record_dir = path / "artifacts" / "runs"
            records = list(record_dir.glob("*.json")) if record_dir.is_dir() and not record_dir.is_symlink() else []
            records = sorted(records, key=_mtime)
            out.append({
                "run_id": path.name,
                "active": path.name == active,
                "status": manifest.get("status"),
                "created_at": manifest.get("created_at"),
                "stopped_at": manifest.get("stopped_at"),
                "selftest": (manifest.get("selftest") or {}).get("status")
                if isinstance(manifest.get("selftest"), dict) else manifest.get("selftest"),
                "topology": manifest.get("topology"),
                "records": min(len(records), MAX_RECORDS),
                "records_truncated": len(records) > MAX_RECORDS,
                "mtime": _mtime(path),
            })
        return out

    def run_detail(self, run_id: str) -> dict[str, Any] | None:
        path = self._run_path(run_id)
        if path is None:
            return None
        manifest = _read_json(path / "manifest.json")
        if not isinstance(manifest, dict):
            return None
        record_dir = path / "artifacts" / "runs"
        try:
            record_paths = sorted(record_dir.glob("*.json"), key=_mtime) if record_dir.is_dir() and not record_dir.is_symlink() else []
        except OSError:
            record_paths = []
        records_truncated = len(record_paths) > MAX_RECORDS
        records: list[dict[str, Any]] = []
        for record_path in record_paths[:MAX_RECORDS]:
            payload = _read_json(record_path)
            if not isinstance(payload, dict):
                continue
            groups = payload.get("groups") or []
            groups_truncated = isinstance(groups, list) and len(groups) > MAX_RUN_ROWS
            records.append({
                "record_id": record_path.stem,
                "kind": payload.get("kind"),
                "status": payload.get("status"),
                "started_at": payload.get("started_at"),
                "finished_at": payload.get("finished_at"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "updates": payload.get("completed_updates"),
                "policy_version": payload.get("committed_policy_version"),
                "requested_steps": payload.get("requested_steps"),
                "error": payload.get("error"),
                "summary": payload.get("summary") or {},
                "groups": groups[:MAX_RUN_ROWS] if isinstance(groups, list) else [],
                "groups_truncated": groups_truncated,
                "metrics": payload.get("metrics") or {},
                "progress": payload.get("progress"),
                "update_intervals": payload.get("update_intervals"),
                "logprob_alignment": payload.get("logprob_alignment"),
                "pre_update_logprob_alignment": payload.get("pre_update_logprob_alignment"),
                "cohorts": payload.get("cohorts"),
                "stream_dispatches": payload.get("stream_dispatches"),
                "policy_handoffs": payload.get("policy_handoffs"),
                "bepa_seed_observability": payload.get("bepa_seed_observability"),
            })
        trajectory_rows = self.trajectories(run_id)
        all_trajectory_paths = path / "artifacts" / "trajectories"
        try:
            trajectory_count = sum(1 for item in all_trajectory_paths.glob("*.json")) if all_trajectory_paths.is_dir() else 0
        except OSError:
            trajectory_count = 0
        return {
            "run_id": run_id,
            "dir": str(path),
            "manifest": manifest,
            "records": records,
            "logs_available": sorted(self._role_log_map(run_id)),
            "trajectories": trajectory_rows,
            "artifacts": self.run_artifacts(run_id),
            "truncation": {
                "records": records_truncated,
                "trajectories": trajectory_count > MAX_TRAJECTORIES,
            },
        }

    def trajectories(self, run_id: str) -> list[dict[str, Any]]:
        path = self._run_path(run_id)
        if path is None:
            return []
        base = path / "artifacts" / "trajectories"
        if not base.is_dir() or base.is_symlink():
            return []
        try:
            rows = sorted(base.glob("*.json"), key=_mtime, reverse=True)[:MAX_TRAJECTORIES]
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for item in rows:
            info = _stat(item)
            out.append({"name": item.name, "bytes": info["bytes"], "mtime": info["mtime"]})
        return out

    def trajectory(self, run_id: str, name: str) -> dict[str, Any] | None:
        if Path(name).name != name or not name.endswith(".json"):
            return None
        run_dir = self._run_path(run_id)
        if run_dir is None:
            return None
        path = run_dir / "artifacts" / "trajectories" / name
        payload = _read_json(path)
        if not isinstance(payload, dict):
            return None
        items = payload.get("trajectories") or []
        rows: list[dict[str, Any]] = []
        for item in items[:MAX_RUN_ROWS] if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            messages = item.get("messages") or []
            rows.append({
                "episode_id": item.get("episode_id"),
                "row_kind": item.get("row_kind"),
                "worker_id": item.get("worker_id"),
                "policy_version": item.get("policy_version"),
                "reward": item.get("reward"),
                "exact": item.get("exact"),
                "generated_tokens": item.get("generated_tokens"),
                "observation_tokens": item.get("observation_tokens"),
                "input_tokens": len(item["input_ids"]) if isinstance(item.get("input_ids"), list) else None,
                "cost": item.get("cost"),
                "tool_events": len(item["tool_events"]) if isinstance(item.get("tool_events"), list) else None,
                "messages": [
                    {"role": m.get("role"),
                     "chars": len(m["content"]) if isinstance(m.get("content"), str) else None}
                    for m in messages[:200] if isinstance(m, dict)
                ],
            })
        return {
            "file": name,
            "group_id": payload.get("group_id"),
            "phase": payload.get("phase"),
            "policy_version": payload.get("policy_version"),
            "adapter_sha256": payload.get("adapter_sha256"),
            "schema": payload.get("schema"),
            "rows": rows,
            "truncated": isinstance(items, list) and len(items) > MAX_RUN_ROWS,
        }

    def run_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        run_dir = self._run_path(run_id)
        if run_dir is None:
            return []
        base = run_dir / "artifacts"
        if not base.is_dir() or base.is_symlink():
            return []
        from .cache import dir_stats
        out: list[dict[str, Any]] = []
        try:
            children = sorted(base.iterdir())
        except OSError:
            return []
        for child in children[:100]:
            if not child.is_dir() or child.is_symlink():
                continue
            files, total, truncated = dir_stats(child)
            try:
                names = sorted(p.name for p in child.iterdir() if p.is_dir() and not p.is_symlink())
            except OSError:
                names = []
            out.append({"name": child.name, "files": files, "bytes": total,
                        "truncated": truncated, "children": names[:12],
                        "children_truncated": len(names) > 12})
        return out

    def logs(self, run_id: str, role: str, tail: int = 200) -> dict[str, Any]:
        tail = max(1, min(int(tail or 200), 1000))
        if not isinstance(role, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", role):
            return {"role": role, "lines": [], "error": "unknown role"}
        logs = self._role_log_map(run_id)
        path = logs.get(role)
        if path is None:
            return {"role": role, "lines": [], "error": "unknown role"}
        info = _stat(path)
        return {"role": role, "file": str(path.relative_to(self._run_path(run_id))),
                "mtime": info["mtime"], "bytes": info["bytes"],
                "lines": [_redact(line) for line in _tail(path, tail)]}

    def gpu(self) -> list[dict[str, Any]] | None:
        if self.source.kind in {"learner", "offline"}:
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                 "clocks.mem,clocks.max.mem,power.draw,power.limit,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10.0)
        except Exception:
            return None
        if result.returncode != 0:
            return None
        cards: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 10:
                continue
            def num(value: str) -> float | None:
                try:
                    return float(value)
                except ValueError:
                    return None
            cards.append({
                "index": int(parts[0]) if parts[0].isdigit() else None,
                "name": parts[1],
                "source": ("selected_runtime_host" if self.source.kind == "runtime"
                           else "local_dashboard_host"),
                "util_percent": num(parts[2]),
                "memory_used_mib": num(parts[3]),
                "memory_total_mib": num(parts[4]),
                "memory_clock_mhz": num(parts[5]),
                "memory_clock_max_mhz": num(parts[6]),
                "power_watts": num(parts[7]),
                "power_limit_watts": num(parts[8]),
                "temperature_c": num(parts[9]),
            })
        return cards

    # -------------------------------------------------------------- selected runtime / BEPA
    def _run_path(self, run_id: str) -> Path | None:
        safe = safe_run_id(run_id)
        if safe is None:
            return None
        base = self.runs_dir.resolve(strict=False)
        candidate = self.runs_dir / safe
        if candidate.is_symlink():
            return None
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(base)
        except ValueError:
            return None
        if not candidate.is_dir() or not (candidate / "manifest.json").is_file():
            return None
        return resolved

    def _read_run_manifest(self, run_id: str) -> dict[str, Any] | None:
        path = self._run_path(run_id)
        if path is None:
            return None
        value = _read_json(path / "manifest.json")
        return value if isinstance(value, dict) else None

    def _role_log_map(self, run_id: str) -> dict[str, Path]:
        run_dir = self._run_path(run_id)
        manifest = self._read_run_manifest(run_id)
        if run_dir is None or manifest is None:
            return {}
        out: dict[str, Path] = {}
        if self.source.runtime_registry_root is not None:
            for item in (manifest.get("roles") or [])[:100]:
                key = item.get("executor_id") or ("learner" if item.get("role") == "server" else item.get("role"))
                if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", key):
                    continue
                candidate = self.source.map_path(item.get("log_file"))
                if candidate is None or candidate.is_symlink() or candidate.suffix != ".log":
                    continue
                if candidate.resolve(strict=False).is_relative_to(run_dir.resolve(strict=False)):
                    out[key] = candidate
            return out
        processes = manifest.get("processes")
        if isinstance(processes, list):
            for item in processes[:100]:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                worker_id = item.get("worker_id")
                if role == "worker":
                    key = worker_id
                elif role in {"learner", "teacher", "manager"}:
                    key = role
                elif isinstance(role, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", role):
                    key = role + ("-" + worker_id if isinstance(worker_id, str)
                                  and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", worker_id) else "")
                else:
                    continue
                if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", key) or key in out:
                    continue
                candidate = self.source.map_path(item.get("log"))
                if candidate is None:
                    raw_path = Path(str(item.get("log") or ""))
                    candidate = raw_path.resolve(strict=False) if raw_path.is_absolute() else None
                if candidate is None or candidate.is_symlink() or candidate.suffix != ".log":
                    continue
                resolved = candidate.resolve(strict=False)
                try:
                    resolved.relative_to(run_dir.resolve(strict=False))
                except ValueError:
                    continue
                out[key] = resolved
        compute_root = run_dir / "artifacts" / "compute"
        try:
            services = sorted(compute_root.iterdir())[:50] if compute_root.is_dir() and not compute_root.is_symlink() else []
        except OSError:
            services = []
        for service in services:
            if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", service.name)
                    or service.is_symlink() or not service.is_dir()):
                continue
            log = service / "rank-service.log"
            if log.is_symlink():
                continue
            try:
                log.resolve(strict=False).relative_to(compute_root.resolve(strict=False))
            except ValueError:
                continue
            if log.is_file():
                out["compute-" + service.name] = log.resolve(strict=False)
        return out

    def _runtime_selection(self, run_id: str | None) -> tuple[str | None, Path | None, dict[str, Any] | None, str]:
        active = self.active_run_id()
        selected = run_id if run_id is not None else active
        if self.source.runtime_registry_root is not None:
            if selected == active and active is not None:
                return selected, self.source.runtime_config_path, self.source.runtime, "central_control_runtime"
            return selected, None, None, "selected_run_unavailable"
        if self.source.instance_manifest_path is not None and selected == active and active is not None:
            return selected, self.source.runtime_config_path, self.source.runtime, "instance_runtime_yaml"
        if selected is not None and safe_run_id(selected) is None:
            return None, None, None, "invalid_run_id"
        if selected is not None:
            run_dir = self._run_path(selected)
            manifest = self._read_run_manifest(selected)
            if run_dir is None or manifest is None:
                if run_id is not None:
                    return selected, None, None, "selected_run_unavailable"
                selected = None
            elif run_id is not None or self.source.matches_profile(manifest, selected):
                candidate = run_dir / "runtime.yaml"
                if candidate.is_file() and not candidate.is_symlink():
                    return selected, candidate, safe_runtime_document(candidate), "selected_runtime_yaml"
                return selected, None, None, "selected_runtime_yaml_missing"
        if self.source.runtime_config_path is not None:
            return None, self.source.runtime_config_path, self.source.runtime, "profile_runtime_config"
        return selected, None, None, self._source_resolution_error or "runtime_config_unavailable"

    def source_info(self, run_id: str | None = None) -> dict[str, Any]:
        selected, runtime_path, runtime, runtime_source = self._runtime_selection(run_id)
        path_values: dict[str, Path | None] = {
            "bepa_store": None, "recovery_dir": None, "evaluation_state_dir": None,
        }
        path_errors: dict[str, str] = {}
        if isinstance(runtime, dict):
            path_values, path_errors = runtime_paths(runtime, self.source)
        elif runtime_source not in {"runtime_config_unavailable", "selected_runtime_yaml_missing",
                                   "selected_run_unavailable", "invalid_run_id"}:
            runtime_source = "runtime_config_invalid"
        if self.source.kind == "offline":
            if run_id is not None and runtime_source in {"invalid_run_id", "selected_run_unavailable"}:
                path_errors["bepa_store"] = "selected run is unavailable"
            elif (run_id is not None and runtime_source == "selected_runtime_yaml_missing"
                    and self.source.explicit_bepa_store is None):
                path_errors["bepa_store"] = "selected run has no runtime.yaml; pass --bepa-store explicitly"
            elif self.source.explicit_bepa_store is not None:
                explicit = self.source.explicit_bepa_store
                path_values["bepa_store"] = explicit.parent if explicit.name == "state.sqlite3" else explicit
                path_errors.pop("bepa_store", None)
                runtime_source = runtime_source + "+explicit_bepa_store"
            elif path_values.get("bepa_store") is None and run_id is None:
                candidates = self._offline_store_candidates()
                if len(candidates) == 1:
                    path_values["bepa_store"] = candidates[0]["store"]
                    path_errors.pop("bepa_store", None)
                    runtime_source = runtime_source + "+unique_offline_store"
                elif len(candidates) > 1:
                    path_errors["bepa_store"] = "ambiguous offline BEPA stores; pass --bepa-store"
                else:
                    path_errors["bepa_store"] = "no BEPA store selected"
        candidates = self._offline_store_candidates() if self.source.kind == "offline" else []
        if self.source.kind == "offline" and self.source.explicit_bepa_store is not None:
            explicit = self.source.explicit_bepa_store
            explicit = explicit.parent if explicit.name == "state.sqlite3" else explicit
            candidates.insert(0, {"path": str(explicit), "store": str(explicit),
                                  "db": str(explicit / "state.sqlite3"), "selected": True})
        return {
            "kind": self.source.kind,
            "deployment": str(self.deployment) if self.deployment else None,
            "deployment_id": self.source.deployment_id,
            "root": str(self.root),
            "runtime_config": str(runtime_path) if runtime_path else None,
            "instance_manifest": (str(self.source.instance_manifest_path)
                                  if self.source.instance_manifest_path else None),
            "runtime": sanitized_runtime(runtime),
            "runtime_source": runtime_source,
            "run_id": selected,
            "is_current": run_id is None or run_id == self.active_run_id(),
            "paths": {key: str(value) if value is not None else None
                      for key, value in path_values.items()},
            "path_errors": path_errors,
            "store_candidates": [
                {key: str(value) if isinstance(value, Path) else value
                 for key, value in item.items()}
                for item in candidates
            ],
            "history_limit": self.live_sampler.history_limit,
            "source_identity": self.source.identity,
        }

    def _offline_store_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for relative in ("validation-stream/bepa", "bepa"):
            store = (self.root / relative).resolve(strict=False)
            db = store / "state.sqlite3"
            if db.is_file() and store not in seen:
                seen.add(store)
                info = _stat(db)
                candidates.append({"path": relative, "store": store, "db": db,
                                   "mtime": info["mtime"], "bytes": info["bytes"]})
        return candidates

    def store_candidates(self, run_id: str | None = None) -> list[dict[str, Any]]:
        info = self.source_info(run_id)
        candidates = [dict(item) for item in (info.get("store_candidates") or [])]
        selected = info.get("paths", {}).get("bepa_store")
        if selected:
            selected_path = Path(selected)
            if not any(item.get("store") == str(selected_path) for item in candidates):
                db = selected_path / "state.sqlite3"
                candidates.insert(0, {"path": str(selected_path), "store": str(selected_path),
                                      "db": str(db), "mtime": _mtime(db),
                                      "bytes": _stat(db)["bytes"], "selected": True})
        return candidates

    def store_dir(self, run_id: str | None = None) -> Path | None:
        selected = self.source_info(run_id).get("paths", {}).get("bepa_store")
        return Path(selected) if isinstance(selected, str) and selected else None

    def _bepa(self, run_id: str | None = None) -> dict[str, Any]:
        info = self.source_info(run_id)
        store = self.store_dir(run_id)
        candidates = self.store_candidates(run_id)
        if store is None:
            reason = (info.get("path_errors") or {}).get("bepa_store", "no selected BEPA store")
            return {"available": False, "reason": reason, "candidates": candidates}
        db = store / "state.sqlite3"
        if not db.is_file():
            return {"available": False, "reason": "selected store has no state.sqlite3",
                    "path": str(store), "candidates": candidates}
        try:
            connection = _connect_ro(db)
        except sqlite3.Error as exc:
            return {"available": False, "reason": f"sqlite open failed: {exc}",
                    "path": str(store), "candidates": candidates}
        try:
            scopes, scopes_cut = _limited_rows(connection, "SELECT * FROM budget_scopes ORDER BY created_at")
            jobs, jobs_cut = _limited_rows(connection, "SELECT * FROM rescue_jobs ORDER BY updated_at DESC")
            plans, plans_cut = _limited_rows(connection, "SELECT * FROM plans ORDER BY updated_at DESC")
            cache, cache_cut = _limited_rows(connection, "SELECT * FROM cache_entries ORDER BY created_at DESC")
        finally:
            connection.close()
        try:
            artifacts = sorted(p.name for p in (store / "artifacts").iterdir()) if (store / "artifacts").is_dir() else []
            sessions = sorted(p.name for p in (store / "sessions").iterdir()) if (store / "sessions").is_dir() else []
        except OSError:
            artifacts, sessions = [], []
        return {
            "available": True, "path": str(store),
            "candidates": candidates, "scopes": scopes, "jobs": jobs,
            "plans": plans, "cache": cache, "artifacts": artifacts[:MAX_LEDGER_ROWS],
            "sessions": sessions[:MAX_LEDGER_ROWS],
            "truncation": {
                "scopes": scopes_cut, "jobs": jobs_cut, "plans": plans_cut, "cache": cache_cut,
                "artifacts": len(artifacts) > MAX_LEDGER_ROWS,
                "sessions": len(sessions) > MAX_LEDGER_ROWS,
            },
        }

    def bepa_overview(self, run_id: str | None = None) -> dict[str, Any]:
        raw = self._bepa(run_id)
        if not raw.get("available"):
            return raw
        def count(key: str, field: str, value: Any) -> int:
            return sum(1 for row in raw[key] if row.get(field) == value)
        return {
            "available": True,
            "path": raw["path"],
            "candidates": raw["candidates"],
            "scopes": raw["scopes"],
            "artifact_count": len(raw["artifacts"]),
            "session_count": len(raw["sessions"]),
            "jobs": {"total": len(raw["jobs"]),
                     "by_state": {state: count("jobs", "state", state)
                                  for state in sorted({r.get("state") for r in raw["jobs"]})}},
            "plans": {"total": len(raw["plans"]),
                      "by_status": {status: count("plans", "status", status)
                                    for status in sorted({r.get("status") for r in raw["plans"]})}},
            "cache": {"total": len(raw["cache"]),
                      "by_kind": {kind: count("cache", "row_kind", kind)
                                  for kind in sorted({r.get("row_kind") for r in raw["cache"]})}},
            "truncation": raw.get("truncation", {}),
        }

    def bepa_detail(self, run_id: str | None = None) -> dict[str, Any]:
        raw = self._bepa(run_id)
        if not raw.get("available"):
            return raw
        return {key: raw[key] for key in
                ("available", "path", "candidates", "scopes", "jobs", "plans", "cache", "truncation")}

    def artifact(self, ref: str, run_id: str | None = None) -> dict[str, Any] | None:
        store = self.store_dir(run_id)
        if store is None or not _sha_ok(ref):
            return None
        artifacts = store / "artifacts"
        path = (artifacts / ref / "artifact.json").resolve(strict=False)
        try:
            path.relative_to(artifacts.resolve(strict=False))
        except ValueError:
            return None
        payload = _read_json(path)
        return None if payload is None else {"ref": ref, "payload": payload,
                                             "bytes": _stat(path)["bytes"], "store": str(store)}

    def watch_snapshot(self, timeout: float = 90.0,
                       run_id: str | None = None) -> dict[str, Any]:
        """Read live metadata; a local owner runs one persistent JSONL watch process.

        Its initial startup retains full artifact verification. Every subsequent frame
        reuses the established authenticated WatchClient and feeds history immediately.
        """
        active = self.active_run_id()
        if run_id is not None and safe_run_id(run_id) is None:
            return {"ok": False, "snapshot": None, "error": "invalid run_id"}
        if run_id is not None and run_id != active:
            return {"ok": False, "snapshot": None, "error": "historical run has no live snapshot"}
        if not self.live or self.source.kind == "offline":
            return {"ok": False, "snapshot": None, "error": "live collection disabled"}
        if self.source.kind == "learner":
            result = learner_watch_snapshot(
                self.source, active_run_id=active, selected_run_id=run_id, timeout=min(timeout, 8.0),
            )
        elif self.source.kind == "runtime":
            result = runtime_watch_snapshot(
                self.source, active_run_id=active, selected_run_id=run_id,
                timeout=min(timeout, 8.0),
            )
        else:
            command = [str(PROJECT_ROOT / "bin" / "qwen-rl"), "local"]
            if self._deployment_explicit and self.deployment is not None:
                command.extend(("--deployment", str(self.deployment)))
            command.extend(("watch", "--json", "--no-gpu", "--log-lines", "0", "--interval", "1", "--exit-on-stdin-eof"))
            env = {name: os.environ[name] for name in _LOCAL_COMMAND_ENV if name in os.environ}
            if self.source.runtime_registry_root is not None:
                from .central import watch_command
                command, env = watch_command(self.source, env, [
                    "--json", "--no-gpu", "--log-lines", "0", "--interval", "1", "--exit-on-stdin-eof",
                ])
            files = (self.deployment, self.source.runtime_config_path)
            revision = tuple((str(path), _mtime(path)) for path in files if path is not None)
            key = (active, self.source.identity, revision)
            with self._watch_stream_lock:
                if self._watch_stream is None or key != self._watch_stream_key:
                    if self._watch_stream is not None:
                        self._watch_stream.close()
                    self._watch_stream = LocalWatchStream(
                        command=command, env=env,
                        expected_run_id=None if self.source.runtime_registry_root is not None else active,
                        on_frame=lambda frame: self._observe_watch(
                            {"ok": True, "snapshot": frame, "error": None}, active, None),
                    )
                    self._watch_stream_key = key
                stream = self._watch_stream
            result = stream.get(timeout=timeout)
            return result if result.get("ok") is True else self._observe_watch(result, active, run_id)
        return self._observe_watch(result, active, run_id)

    def _observe_watch(self, result: dict[str, Any], active: str | None,
                       run_id: str | None) -> dict[str, Any]:
        if result.get("ok") is not True:
            result = {**result, "expected_runtime_run_id": active}
        snapshot = result.get("snapshot") if isinstance(result, dict) else None
        if isinstance(snapshot, dict) and self.source.runtime_registry_root is not None:
            from .central import bind_snapshot
            try:
                snapshot = bind_snapshot(self.source, snapshot, active)
                result = {**result, "snapshot": snapshot}
            except (OSError, ValueError):
                result = {"ok": False, "snapshot": None, "error": "central control/live identity mismatch"}
                snapshot = None
        elif isinstance(snapshot, dict) and self.source.instance_manifest_path is not None:
            from .central import bind_instance_snapshot
            try:
                snapshot = bind_instance_snapshot(self.source, snapshot, run_id)
                result = {**result, "snapshot": snapshot}
                self._instance_generation_id = snapshot["source_identity"]["instance_run_id"]
                self._instance_logical_run_id = snapshot["source_identity"].get("logical_run_id")
            except (OSError, ValueError):
                self._instance_logical_run_id = None
                result = {"ok": False, "snapshot": None,
                          "error": "instance manifest/live identity mismatch"}
                snapshot = None
        if isinstance(snapshot, dict):
            lifecycle = snapshot.get("lifecycle") if isinstance(snapshot.get("lifecycle"), dict) else {}
            snapshot_run_id = lifecycle.get("run_id")
            if active is not None and snapshot_run_id is not None and snapshot_run_id != active:
                result = {"ok": False, "snapshot": None,
                          "error": "live runtime identity differs from selected manifest"}
            elif run_id is not None and snapshot_run_id not in (None, run_id):
                result = {"ok": False, "snapshot": None,
                          "error": "live runtime identity differs from selected run"}
        runtime = self.source.runtime if isinstance(self.source.runtime, dict) else {}
        workers = runtime.get("workers") if isinstance(runtime.get("workers"), (list, tuple)) else []
        expected = tuple(item["id"] for item in workers
                         if isinstance(item, dict) and isinstance(item.get("id"), str))
        if (isinstance(snapshot, dict) and result.get('ok') is True
                and isinstance(snapshot.get("sampled_at"), (int, float))
                and 0 <= time.time() - snapshot["sampled_at"] <= 5):
            from .native_metrics import observe_native_metrics
            snapshot = observe_native_metrics(self.source, snapshot)
            result = {**result, 'snapshot': snapshot}
        sampler_identity = self.source.identity
        source_identity = snapshot.get("source_identity") if isinstance(snapshot, dict) else None
        if isinstance(source_identity, dict):
            encoded = json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            sampler_identity += ":" + hashlib.sha256(encoded).hexdigest()[:24]
        return self.live_sampler.observe(
            result, source_identity=sampler_identity, expected_worker_ids=expected,
        )

    def history(self, limit: int = DEFAULT_HISTORY_SAMPLES,
                run_id: str | None = None) -> dict[str, Any]:
        if run_id is not None and safe_run_id(run_id) != self.active_run_id():
            return {"samples": [], "truncated": False, "total": 0,
                    "limit": max(1, min(int(limit), self.live_sampler.history_limit)),
                    "retained": 0, "reason": "historical run has no live history"}
        return self.live_sampler.history_state(limit)

    def close(self) -> None:
        """Stop only the JSONL reader process owned by this Collector."""
        with self._watch_stream_lock:
            if self._watch_stream is not None:
                self._watch_stream.close()
                self._watch_stream = None
                self._watch_stream_key = None

    def overview(self, watch: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "generated_at": time.time(),
            "root": str(self.root),
            "source": self.source_info(),
            "active_run_id": self.active_run_id(),
            "runs": self.runs(),
            "gpus": self.gpu(),
            "bepa": self.bepa_overview(),
            "watch": watch if watch is not None else self.watch_snapshot(),
        }
