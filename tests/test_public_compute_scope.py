from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from qwen_rl.deployment.learner.cli import main as learner_main
from qwen_rl.learner.ternary_compute import run_compute
from qwen_rl.learner.ternary_config import TernaryComputeConfigError


ROOT = Path(__file__).resolve().parents[1]


def test_compression_training_modules_are_removed_but_artifact_provenance_remains() -> None:
    removed = (
        "learner/compression_config.py",
        "learner/compression_data.py",
        "learner/compression_loss.py",
        "learner/compression_model.py",
        "learner/compression_recovery.py",
        "learner/compression_trainer.py",
        "models/mla_recovery.py",
    )
    assert all(not (ROOT / "qwen_rl" / path).exists() for path in removed)

    artifact_contract = (ROOT / "qwen_rl/contracts/ternary.py").read_text(encoding="utf-8")
    assert 'provenance.get("compression_recovery")' in artifact_contract


def test_legacy_compression_profile_is_rejected_before_allocator_setup(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = tmp_path / "compression.json"
    profile.write_text(json.dumps({
        "schema_version": "qwen-rl-ternary-compute/v1",
        "training": {"objective": "compression_recovery"},
    }), encoding="utf-8")

    import qwen_rl.learner.cuda_allocator as cuda_allocator

    def allocator_must_not_run() -> None:
        pytest.fail("invalid profile reached allocator setup")

    monkeypatch.setattr(cuda_allocator, "configure_learner_allocator", allocator_must_not_run)
    with pytest.raises(TernaryComputeConfigError, match="compression_recovery profiles are no longer supported"):
        run_compute(profile)


def test_compute_keeps_t2_trajectory_and_self_smoke_inputs_without_check_mode() -> None:
    parameters = inspect.signature(run_compute).parameters
    assert {"trajectories_path", "self_smoke"}.issubset(parameters)
    assert "check_only" not in parameters

    source = (ROOT / "qwen_rl/learner/ternary_compute.py").read_text(encoding="utf-8")
    assert "def _trajectory_rows" in source
    assert "def _self_smoke_rows" in source
    assert "compression_recovery" not in source


def test_learner_compute_no_longer_accepts_compression_check_flag(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        learner_main(["compute", "--config", str(tmp_path / "unused.json"), "--check"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --check" in capsys.readouterr().err
