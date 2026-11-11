"""Cheap presence check for source files required by managed instance startup."""
from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any

_SCHEMA = "qwen-rl-instance-source-package-preflight/v1"
from .autodl_install import CRITICAL_INSTALLED_FILES

_REQUIRED_FILES = tuple(name.removeprefix("source/")
                        for name in CRITICAL_INSTALLED_FILES if name.startswith("source/"))


def source_package_preflight(source_root: str | Path) -> dict[str, Any]:
    """Check startup-critical source package entries using metadata only."""
    root = Path(os.path.abspath(Path(source_root).expanduser()))
    report: dict[str, Any] = {
        "schema": _SCHEMA,
        "status": "blocked",
        "source_root": str(root),
        "checked_files": list(_REQUIRED_FILES),
        "errors": [],
        "scope": "required_source_paths_only_no_imports_no_gpu_or_payload_hashes",
    }
    try:
        root_info = root.lstat()
    except OSError:
        report["errors"].append({
            "reason_code": "source_package_root_missing",
            "field": "source_package",
            "message": "The managed instance source package root is missing or unreadable.",
        })
        return report
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        report["errors"].append({
            "reason_code": "source_package_root_invalid",
            "field": "source_package",
            "message": "The managed instance source package root is not a real directory.",
        })
        return report

    for relative in _REQUIRED_FILES:
        current = root
        code = None
        for part in Path(relative).parts[:-1]:
            current /= part
            try:
                info = current.lstat()
            except FileNotFoundError:
                code = "source_package_file_missing"
                break
            except OSError:
                code = "source_package_path_unavailable"
                break
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                code = "source_package_path_invalid"
                break
        if code is None:
            try:
                info = (current / Path(relative).name).lstat()
            except FileNotFoundError:
                code = "source_package_file_missing"
            except OSError:
                code = "source_package_path_unavailable"
            else:
                if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    code = "source_package_file_invalid"
        if code is None:
            continue
        messages = {
            "source_package_file_missing": f"A required source package file is missing: {relative}.",
            "source_package_path_invalid": f"A source package directory for {relative} is invalid.",
            "source_package_file_invalid": f"A required source package entry is not a regular file: {relative}.",
            "source_package_path_unavailable": f"A required source package path is unreadable: {relative}.",
        }
        report["errors"].append({"reason_code": code, "field": relative,
                                 "path": relative, "message": messages[code]})

    report["status"] = "blocked" if report["errors"] else "passed"
    return report
