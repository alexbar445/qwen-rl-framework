"""Public exports must not expose local state, dirty edits, links or old history."""
import importlib.util
from pathlib import Path
import subprocess

import pytest


SPEC = importlib.util.spec_from_file_location("public_snapshot", Path(__file__).resolve().parents[1] / "tools/build_public_snapshot.py")
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in snapshot.REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("committed public source\n")
    for name in ["outputs/runtime.token", "data/generated/private/task.json", "vendor/private.py", "journal/private.md", "deploy/worker/native-shared-prefix.patch"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not for publication\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Snapshot test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    return root


def test_only_committed_allowlist_is_exported(repo, tmp_path):
    (repo / "README.md").write_text("unreviewed dirty text\n")
    (repo / "qwen_rl/untracked.py").write_text("unreviewed new code\n")
    out = tmp_path / "public"
    result = snapshot.export(repo, out)
    assert (out / "README.md").read_text() == "committed public source\n"
    assert not (out / "qwen_rl/untracked.py").exists()
    for name in [".git", "outputs", "data", "vendor", "journal", "deploy"]:
        assert not (out / name).exists()
    assert set(result["files"]) == set(snapshot.REQUIRED)


def test_link_to_private_file_is_rejected(repo, tmp_path):
    (repo / "qwen_rl/leak.py").symlink_to("../outputs/runtime.token")
    subprocess.run(["git", "-C", str(repo), "add", "qwen_rl/leak.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Snapshot test", "-c", "user.email=test@example.invalid", "commit", "-qm", "link"], check=True)
    with pytest.raises(ValueError, match="regular file"):
        snapshot.export(repo, tmp_path / "public")


def test_existing_output_and_in_checkout_output_are_rejected(repo, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "keep"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        snapshot.export(repo, out)
    assert marker.read_text() == "keep"
    with pytest.raises(ValueError, match="outside"):
        snapshot.export(repo, repo / "public")
