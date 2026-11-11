"""Prepare an ACL-free copy of the fixed Bonsai2 model inputs."""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

MODEL_NAME = "Ternary-Bonsai-2-27B-ARC-K2-MTP-eos.ninfer"
REPORT_NAME = MODEL_NAME + ".conversion.json"
MODEL_SIZE_BYTES = 7_831_249_152
REPORT_SIZE_BYTES = 300_492
_ACL_XATTR = "system.posix_acl_access"
_UNAVAILABLE_XATTR_ERRORS = {
    errno.ENODATA,
    getattr(errno, "ENOATTR", errno.ENODATA),
    errno.ENOTSUP,
    errno.EOPNOTSUPP,
}


class ModelContextError(Exception):
    """A sanitized input or copy failure."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _snapshot(path: Path, expected_size: int) -> tuple[int, int, int, int, int, int]:
    if path.is_symlink():
        raise ModelContextError("model_input_invalid")
    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        raise ModelContextError("model_input_invalid") from None
    if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
        raise ModelContextError("model_input_invalid")
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _remove_acl(path: Path) -> None:
    try:
        os.removexattr(path, _ACL_XATTR)
    except OSError as error:
        if error.errno not in _UNAVAILABLE_XATTR_ERRORS:
            raise ModelContextError("acl_normalization_failed") from None
    try:
        attributes = os.listxattr(path)
    except OSError as error:
        if error.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
            return
        raise ModelContextError("acl_normalization_failed") from None
    if _ACL_XATTR in attributes:
        raise ModelContextError("acl_normalization_failed")


def _remove_stage(stage: Path | None) -> None:
    if stage is not None:
        shutil.rmtree(stage, ignore_errors=True)


def prepare_model_context(input_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    """Copy the two verified inputs to a new private directory and publish atomically."""
    source_root = Path(input_dir)
    target = Path(output_dir).absolute()
    if source_root.is_symlink() or not source_root.is_dir() or not target.name:
        raise ModelContextError("model_input_invalid")
    parent = target.parent
    if not parent.is_dir():
        raise ModelContextError("output_parent_invalid")
    if target.exists() or target.is_symlink():
        raise ModelContextError("output_exists")

    names = (MODEL_NAME, REPORT_NAME)
    expected = {MODEL_NAME: MODEL_SIZE_BYTES, REPORT_NAME: REPORT_SIZE_BYTES}
    try:
        entries = {entry.name for entry in source_root.iterdir()}
    except OSError:
        raise ModelContextError("model_input_invalid") from None
    if entries != set(names):
        raise ModelContextError("model_input_invalid")
    sources = {name: source_root / name for name in names}
    identities = {name: _snapshot(path, expected[name]) for name, path in sources.items()}
    try:
        with sources[MODEL_NAME].open("rb") as stream:
            if not stream.read(1):
                raise ModelContextError("model_input_invalid")
        report = json.loads(sources[REPORT_NAME].read_text(encoding="utf-8"))
    except ModelContextError:
        raise
    except Exception:
        raise ModelContextError("model_input_invalid") from None
    if not isinstance(report, dict):
        raise ModelContextError("model_input_invalid")

    stage: Path | None = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
        os.chmod(stage, 0o700)
        for name in names:
            source, copied = sources[name], stage / name
            subprocess.run(
                ["cp", "--reflink=auto", "--no-preserve=all", "--", str(source), str(copied)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _remove_acl(copied)
            os.chmod(copied, 0o444)
            source_info, copied_info = source.stat(), copied.stat(follow_symlinks=False)
            if (not stat.S_ISREG(copied_info.st_mode)
                    or copied_info.st_size != expected[name]
                    or stat.S_IMODE(copied_info.st_mode) != 0o444
                    or (source_info.st_dev, source_info.st_ino) == (copied_info.st_dev, copied_info.st_ino)):
                raise ModelContextError("model_copy_invalid")
        if any(_snapshot(sources[name], expected[name]) != identities[name] for name in names):
            raise ModelContextError("model_input_changed_during_copy")
        if target.exists() or target.is_symlink():
            raise ModelContextError("output_exists")
        os.rename(stage, target)
        stage = None
    except ModelContextError:
        _remove_stage(stage)
        raise
    except Exception:
        _remove_stage(stage)
        raise ModelContextError("model_copy_failed") from None

    return {
        "status": "prepared",
        "prepared_model_dir": str(target),
        "copied_files": list(names),
        "gpu_started": False,
        "training_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare_model_context(args.input_dir, args.output_dir)
    except ModelContextError as error:
        print(json.dumps({"status": "failed", "reason_code": error.code}), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
