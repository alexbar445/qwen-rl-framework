"""First-user and domain-isolation regressions found during the final review."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("group", ["train", "instance", "vast", "local", "learner", "runtime",
                                  "cluster", "dashboard", "gradients"])
def test_public_command_help_does_not_import_missing_profiles_or_model_runtimes(group):
    code = """
import sys
from qwen_rl import cli
try:
    status = cli.main([sys.argv[1], '--help'])
except SystemExit as error:
    status = error.code
assert status == 0
assert not {'torch', 'transformers', 'vllm', 'trl', 'verl'}.intersection(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code, group], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "qwen-rl" in result.stdout


def test_dashboard_missing_selector_is_actionable_before_collection(capsys):
    from qwen_rl.dashboard.cli import main
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    assert "--deployment PATH" in capsys.readouterr().err


def test_dashboard_can_read_empty_offline_history_without_runtime(tmp_path, capsys):
    import json
    from qwen_rl.dashboard.cli import main
    assert main(["--root", str(tmp_path), "--once", "--no-live"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), dict)


def test_generic_environment_named_arc_does_not_enable_arc_catalog(monkeypatch, tmp_path):
    import qwen_rl.examples.minimal as example
    import qwen_rl.task_catalog as catalog

    original = example._coordinator

    def configure(*args, **kwargs):
        coordinator, runtime = original(*args, **kwargs)
        coordinator.services.config["dataset_dir"] = str(tmp_path / "absent-catalog")
        coordinator.services.config["rollout"]["category_sampling"] = True
        return coordinator, runtime

    def forbidden(*args, **kwargs):
        pytest.fail("generic adapter activated ARC source loading")

    monkeypatch.setattr(example.BinaryChoiceEnvironment, "name", "arc")
    monkeypatch.setattr(example, "_coordinator", configure)
    monkeypatch.setattr(catalog, "load_source_catalog", forbidden)
    assert example.run_minimal(tmp_path / "run")["recovery_match"] is True
