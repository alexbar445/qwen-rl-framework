"""Protect the release boundary without starting model or deployment roles."""
from pathlib import Path
import tomllib

import pytest

from qwen_rl.deployment.instance_cli import build_parser
from qwen_rl.deployment import local

ROOT = Path(__file__).resolve().parents[1]


def test_specialized_research_entrypoints_are_not_distributed():
    for name in ["span_pivot", "evaluation/rank32_transition.py",
                 "evaluation/evaluate_9b_pilot.py", "evaluation/validate_9b_pilot.py"]:
        assert not (ROOT / "qwen_rl" / name).exists()
    assert (ROOT / "qwen_rl/rollout/pipeline.py").is_file()
    assert (ROOT / "qwen_rl/learner/gradient_server/store.py").is_file()
    assert (ROOT / "qwen_rl/worker/inference_pool.py").is_file()


def test_instance_setup_requires_an_explicit_template(tmp_path):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["setup", "--output-dir", str(tmp_path / "new")])
    assert error.value.code == 2
    args = build_parser().parse_args([
        "setup", "--runtime-config", str(tmp_path / "runtime.yaml"),
        "--output-dir", str(tmp_path / "new"),
    ])
    assert args.runtime_config == tmp_path / "runtime.yaml"


def test_local_missing_profile_is_rejected_before_any_deployment_operation(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("missing profile reached deployment loading")
    monkeypatch.setattr(local, "load_deployment", forbidden)
    assert local.main(["start"]) == 2
    assert "--deployment PATH is required" in capsys.readouterr().err
    assert local.main(["--help"]) == 0


def test_public_ternary_extra_does_not_resolve_an_unpublished_native_wheel():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    assert "ternary-kernels" not in extras
    assert extras["ternary-deps"]
    assert not any("qwen-rl-ternary-kernels" in dep for deps in extras.values() for dep in deps)
