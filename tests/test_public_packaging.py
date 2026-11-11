from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def _built_wheel() -> Path:
    configured = os.environ.get("PUBLIC_WHEEL_PATH")
    wheels = [Path(configured)] if configured else sorted((REPO_ROOT / "dist").glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one built wheel, found {wheels}"
    wheel = wheels[0].resolve()
    assert wheel.is_file(), f"wheel does not exist: {wheel}"
    return wheel


def test_wheel_contains_only_explicit_arc_resources_and_not_unknown_vendor() -> None:
    with zipfile.ZipFile(_built_wheel()) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")

    assert "License-Expression: Apache-2.0" in metadata
    assert "qwen_rl/resources/arc/arc-system-rules.txt" in names
    assert "qwen_rl/resources/arc/README.md" in names
    assert "qwen_rl/resources/arc/__init__.py" in names
    assert not any("toolkit-dependency.json" in name for name in names)
    assert not any("vendor/paired-factory-v1" in name for name in names)
    assert not any("vendor/arc-taskgen-v2" in name for name in names)
    assert "qwen_rl/dashboard/static/vendor/ECHARTS-LICENSE.txt" in names
    assert "qwen_rl/dashboard/static/vendor/ECHARTS-NOTICE.txt" in names
    assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
    assert any(name.endswith(".dist-info/licenses/NOTICE") for name in names)
    assert any(name.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md") for name in names)
    assert "Provides-Extra: learner" in metadata
    assert "Provides-Extra: worker" in metadata
    for dependency in ("torch", "torchao", "vllm", "transformers", "peft", "trl", "verl"):
        rows = [line for line in metadata.splitlines()
                if line.lower().startswith(f"requires-dist: {dependency}")]
        assert rows, (dependency, "training dependency disappeared from its optional extra")
        assert all("; extra ==" in line for line in rows), (dependency, rows)


def test_installed_wheel_cli_demo_and_core_contract_work_outside_checkout(tmp_path: Path) -> None:
    wheel = _built_wheel()
    wheelhouse = Path(os.environ["PUBLIC_WHEELHOUSE"]).resolve()
    assert wheelhouse.is_dir(), f"wheelhouse does not exist: {wheelhouse}"
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin" / "python"
    qwen_cli = venv / "bin" / "qwen-rl"
    env = os.environ.copy()
    env.pop("ARC_RL_ROOT", None)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), str(wheel)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], cwd=tmp_path, env=env, check=True,
                   capture_output=True, text=True)

    module_check = (
        "from pathlib import Path; import sys, qwen_rl; "
        "from qwen_rl.contracts import environment; "
        "import qwen_rl.learner.app; "
        "from qwen_rl.integrations.arc.environment import ArcEnvironment, ArcRewardScorer; "
        "from qwen_rl.resources.arc import read_system_rules; "
        "assert Path(qwen_rl.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
        "assert read_system_rules().strip(); "
        "assert all(name not in sys.modules for name in ('qwen_rl.prompt', 'qwen_rl.environment', "
        "'qwen_rl.service', 'qwen_rl.task_source', 'torch', 'transformers')); "
        "assert ArcEnvironment is not None and ArcRewardScorer is not None; "
        "assert environment is not None"
    )
    subprocess.run([str(python), "-c", module_check], cwd=tmp_path, env=env, check=True)

    help_output = subprocess.run(
        [str(qwen_cli), "--help"], cwd=tmp_path, env=env,
        check=True, capture_output=True, text=True,
    ).stdout
    assert "qwen-rl" in help_output
    assert "instance" in help_output
    assert str(REPO_ROOT) not in help_output
    assert "docs/QUICKSTART.md" not in help_output

    subprocess.run([str(python), "-m", "qwen_rl", "--help"], cwd=tmp_path, env=env, check=True,
                   capture_output=True, text=True)
    subprocess.run([str(python), "-m", "qwen_rl.examples.minimal"], cwd=tmp_path, env=env, check=True,
                   capture_output=True, text=True)

    missing_arc = (
        "from qwen_rl.prompt import system_prompt\n"
        "try:\n"
        "    system_prompt()\n"
        "except RuntimeError as exc:\n"
        "    assert 'ARC toolkit is not bundled' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('an ARC toolkit was silently substituted')\n"
    )
    subprocess.run([str(python), "-c", missing_arc], cwd=tmp_path, env=env, check=True,
                   capture_output=True, text=True)
