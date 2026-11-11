from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from qwen_rl import cli


ROOT = Path(__file__).resolve().parents[1]


def test_root_help_and_import_are_lazy() -> None:
    script = """
import importlib.abc
import sys
class HeavyImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith(('torch.', 'transformers', 'trl', 'verl', 'vllm')):
            raise AssertionError('root help imported a model runtime: ' + fullname)
sys.meta_path.insert(0, HeavyImportBlocker())
import qwen_rl.cli
assert not any(name == 'qwen_rl.runtime' or name.startswith('qwen_rl.runtime.') for name in sys.modules)
assert qwen_rl.cli.main(['--help']) == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert all(group in result.stdout for group in ("local", "runtime", "cluster", "dashboard", "vast"))


def test_all_canonical_entry_forms_use_the_same_root_groups() -> None:
    module = subprocess.run(
        [sys.executable, "-m", "qwen_rl", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    script = subprocess.run(
        [str(ROOT / "bin/qwen-rl"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert module.returncode == script.returncode == 0
    assert module.stdout == script.stdout

    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    scripts = project.split("[project.scripts]", 1)[1].split("[", 1)[0]
    assert [line.strip() for line in scripts.splitlines() if line.strip()] == [
        'qwen-rl = "qwen_rl.cli:main"',
    ]
    assert not (ROOT / "bin/qwen-cluster").exists()
    assert not (ROOT / "bin/qwen-rl-dash").exists()
    assert not (ROOT / "qwen_rl/dashboard/__main__.py").exists()


def test_core_install_keeps_training_and_display_dependencies_optional() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    core = project.split("[project]\n", 1)[1].split("[project.optional-dependencies]", 1)[0]
    dependencies = next(line for line in core.splitlines() if line.startswith("dependencies ="))
    assert all(package not in dependencies for package in (
        "numpy", "pyarrow", "transformers", "hydra-core", "rich", "torch", "trl", "verl", "vllm",
    ))
    learner = project.split("learner = [", 1)[1].split("]", 1)[0]
    worker = next(line for line in project.splitlines() if line.startswith("worker = "))
    runtime = next(line for line in project.splitlines() if line.startswith("runtime = "))
    assert all(package in learner for package in (
        "numpy", "pyarrow", "transformers", "hydra-core", "torchao==0.18.0",
    ))
    assert '"vllm==0.29.0"' in worker and '"torchao==0.18.0"' in worker
    assert '"transformers==5.12.1"' in worker
    assert '"rich==15.0.0"' in runtime


def test_dispatch_imports_only_selected_group(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []

    class FakeGroup:
        @staticmethod
        def main(argv):
            return 7 if argv == ["worker", "--config", "worker.json"] else 11

    def import_one(name: str):
        imported.append(name)
        return FakeGroup

    monkeypatch.setattr(cli, "import_module", import_one)
    assert cli.main(["cluster", "worker", "--config", "worker.json"]) == 7
    assert imported == ["qwen_rl.cluster.__main__"]


def test_minimal_install_reports_omitted_group_without_masking_missing_dependencies(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def import_missing_group(_name: str):
        raise ModuleNotFoundError("missing optional command package", name="qwen_rl.deployment")

    monkeypatch.setattr(cli, "import_module", import_missing_group)
    assert cli.main(["local", "check"]) == 2
    assert "local command group is not included" in capsys.readouterr().err

    def import_missing_dependency(_name: str):
        raise ModuleNotFoundError("missing required dependency", name="fastapi")

    monkeypatch.setattr(cli, "import_module", import_missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="missing required dependency"):
        cli.main(["cluster", "worker"])


def test_old_top_level_runtime_and_dashboard_aliases_are_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["start", "--config", "runtime.yaml"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cluster_does_not_expose_a_second_runtime_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["cluster", "runtime", "status", "--config", "runtime.yaml"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
