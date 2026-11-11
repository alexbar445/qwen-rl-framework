#!/usr/bin/env python3
"""Export reviewed committed source, without research data or private Git history."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
# This list defines the source distribution, independently of ignored local files.
INCLUDE = (
    "qwen_rl/*", "examples/*", ".github/workflows/*",
    "README.md", "README.zh-CN.md", "LICENSE", "NOTICE",
    "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "SECURITY.md",
    "pyproject.toml", "MANIFEST.in", ".gitignore", ".gitleaks.toml", "bin/qwen-rl",
    "docs/PUBLIC_RELEASE.md", "docs/GETTING_STARTED.md", "docs/ARCHITECTURE.md",
    "docs/EXTENDING.md", "docs/SUPPORT.md", "docs/RELEASE_REVIEW.md",
    "requirements/public-*.txt", "provenance/initial-20261111/*.json",
    "tools/build_public_snapshot.py",
    "tests/test_cli_entry.py", "tests/test_worker_contract.py",
    "tests/test_worker_service_identity_client.py", "tests/test_learner_run_state.py",
    "tests/test_heterogeneous_sampling.py", "tests/test_gradient_round_coordinator.py",
    "tests/test_instance_release.py",
    "tests/test_environment*.py", "tests/test_generic*.py",
    "tests/test_minimal_example.py", "tests/test_public*.py",
)
REQUIRED = (
    "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
    "qwen_rl/cli.py", "qwen_rl/contracts/worker.py",
)


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def export(root: Path, output: Path, ref: str = "HEAD") -> dict:
    root = root.resolve()
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("output must be outside the source checkout")
    if output.exists():
        raise FileExistsError("output already exists; select a new directory")
    commit = git(root, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}").decode().strip()
    selected = []
    for row in git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if not row:
            continue
        head, raw_path = row.split(b"\t", 1)
        name = raw_path.decode("utf-8")
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in INCLUDE):
            continue
        mode, kind, _oid = head.decode().split()
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name:
            raise ValueError(f"unsafe source path: {name}")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"public source must be a regular file: {name}")
        selected.append(name)
    missing = sorted(set(REQUIRED) - set(selected))
    if missing:
        raise ValueError(f"missing public release inputs: {missing}")
    archive = git(root, "archive", "--format=tar", commit, "--", *selected)
    output.mkdir(parents=True)
    entries = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        for member in source:
            if member.isdir():
                continue
            if not member.isfile() or member.name not in selected:
                raise ValueError(f"unexpected archive entry: {member.name}")
            content = source.extractfile(member).read()
            target = output.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
            entries[member.name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    manifest = {
        "schema": "qwen-rl-public-source/v1", "source_commit": commit,
        "files": entries,
        "scope": "Committed reviewed-source allowlist only. No private Git history, models, datasets, runtime state, external vendor trees or native patches. Licensed package-local _vendor code retains its notices. File identity is not a secret or licensing audit.",
    }
    (output / "PUBLIC_SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD", help="committed release revision")
    args = parser.parse_args()
    result = export(ROOT, args.output, args.ref)
    print(json.dumps({"source_commit": result["source_commit"], "files": len(result["files"]), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
