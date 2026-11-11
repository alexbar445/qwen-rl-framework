from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_rl.deployment.instance_release import main, preflight_base_image


BASE = "ghcr.io/example/qwen-rl-instance@sha256:" + "a" * 64
MODEL_PATH = Path("/opt/models/bonsai2/base.ninfer")
CONVERSION_REPORT = Path("/opt/models/bonsai2/base.ninfer.conversion.json")
MODEL_SIZE_BYTES = 7_831_249_152
COMPAT = {
    "architectures": ["8.6", "8.9"],
    "cuda": "13.0",
    "cpu_checks": "passed",
    "cuobjdump_check": "passed",
    "gpu_checks": "not_requested",
    "liger_commit": "ff8ce5857420cee06865f40ffa8a60309bf2b458",
    "liger_provenance": "git-direct-url",
    "torchao_native_libraries": ["/private/path/libtorchao.so"],
}
SECRET_TEXT = "Authorization: Bearer never-print-this"


def _completed(argv: list[str], *, payload: dict | None = None, returncode: int = 0,
               stderr: str = SECRET_TEXT) -> SimpleNamespace:
    return SimpleNamespace(
        args=argv,
        returncode=returncode,
        stdout=json.dumps(payload if payload is not None else COMPAT),
        stderr=stderr,
    )


def test_model_preflight_reuses_cpu_compatibility_gate_without_gpu_or_pull():
    commands = []

    def runner(argv, **kwargs):
        commands.append((argv, kwargs))
        return _completed(argv)

    report = preflight_base_image(BASE, kind="model", runner=runner)

    assert report["status"] == "ready"
    assert report["kind"] == "model"
    assert report["compatibility"] == {
        "architectures": ["8.6", "8.9"],
        "cuda": "13.0",
        "cpu_checks": "passed",
        "cuobjdump_check": "passed",
        "gpu_checks": "not_requested",
        "liger_commit": COMPAT["liger_commit"],
        "liger_provenance": "git-direct-url",
    }
    assert report["model_access"] == "not_checked"
    assert report["next_action"] == "build_full_model_image"
    assert len(commands) == 1
    argv, kwargs = commands[0]
    assert argv[:4] == ["docker", "run", "--pull=never", "--rm"]
    assert "--cidfile" in argv
    assert argv[argv.index("--cidfile") + 1].endswith("/container.id")
    assert "CUDA_VISIBLE_DEVICES=" in argv
    assert "NVIDIA_VISIBLE_DEVICES=void" in argv
    assert "PYTHONPATH=/work" in argv
    assert "--gpus" not in argv
    assert argv[-3:] == ["-B", "-m", "qwen_rl.deployment.instance_compat"]
    assert kwargs["timeout"] == 300
    assert kwargs["capture_output"] is True and kwargs["text"] is True
    rendered = json.dumps(report)
    assert SECRET_TEXT not in rendered
    assert "/private/path/libtorchao.so" not in rendered


def test_runtime_overlay_also_probes_model_and_report_as_uid_10001():
    commands = []

    def runner(argv, **kwargs):
        commands.append((argv, kwargs))
        if "-m" in argv:
            return _completed(argv)
        return _completed(argv, payload={"model_access": "passed"})

    report = preflight_base_image(
        BASE, kind="runtime-overlay", model_path=MODEL_PATH,
        conversion_report=CONVERSION_REPORT, model_size_bytes=MODEL_SIZE_BYTES, runner=runner,
    )

    assert report["status"] == "ready"
    assert report["model_access"] == "passed_as_uid_10001"
    assert report["next_action"] == "build_runtime_overlay"
    assert len(commands) == 2
    probe_argv, probe_kwargs = commands[1]
    assert "--user" in probe_argv
    assert probe_argv[probe_argv.index("--user") + 1] == "10001:10001"
    assert probe_argv[-2] == "-c"
    probe_source = probe_argv[-1]
    assert "model.open('rb')" in probe_source
    assert "report.open('rb')" in probe_source
    assert str(MODEL_PATH) in probe_source
    assert str(CONVERSION_REPORT) in probe_source
    assert f"model.stat().st_size == {MODEL_SIZE_BYTES}" in probe_source
    assert "os.access(parent, os.X_OK)" in probe_source
    assert "--gpus" not in probe_argv
    assert probe_kwargs["timeout"] == 300


@pytest.mark.parametrize("message,code", [
    ("installed T2 binary lacks required cubin: sm_89", "base_missing_sm89"),
    ("instance wheel lacks required CUDA targets: 8.9", "base_missing_sm89"),
    ("T2 kernel wheel must match the installed Torch/CUDA", "base_torch_cuda_kernel_mismatch"),
])
def test_known_parent_gate_failures_have_specific_redacted_codes(message, code):
    def runner(argv, **kwargs):
        return _completed(argv, returncode=1, stderr=message + "\n" + SECRET_TEXT)
    report = preflight_base_image(BASE, kind="model", runner=runner)
    assert report["reason_codes"] == [code]
    assert report["status"] == "blocked"
    assert SECRET_TEXT not in json.dumps(report)


def test_unavailable_local_parent_is_incomplete_without_rebuild_advice():
    def runner(argv, **kwargs):
        return _completed(argv, returncode=125)
    report = preflight_base_image(BASE, kind="model", runner=runner)
    assert report["status"] == "incomplete"
    assert report["reason_codes"] == ["base_container_unavailable"]
    assert report["next_action"] == "check_local_pinned_image_and_docker"


@pytest.mark.parametrize("change,status", [
    ({"liger_commit": None}, "blocked"),
    ({"liger_provenance": ["secret-must-not-echo"]}, "blocked"),
    ({"liger_provenance": None}, "blocked"),
])
def test_source_provenance_is_required_even_after_generic_cpu_success(change, status):
    report = preflight_base_image(BASE, kind="model", runner=lambda argv, **kwargs:
                                  _completed(argv, payload={**COMPAT, **change}))
    assert report["status"] == status
    assert report["reason_codes"] == ["base_liger_provenance_invalid"]
    assert "secret-must-not-echo" not in json.dumps(report)


def test_legacy_parent_gate_without_provenance_is_incomplete():
    legacy = {key: value for key, value in COMPAT.items() if not key.startswith("liger_")}
    report = preflight_base_image(BASE, kind="model", runner=lambda argv, **kwargs:
                                  _completed(argv, payload=legacy))
    assert report["status"] == "incomplete"
    assert report["reason_codes"] == ["base_liger_provenance_missing"]


def test_missing_sm89_is_blocked_before_the_runtime_overlay_probe():
    commands = []

    def runner(argv, **kwargs):
        commands.append(argv)
        bad = {**COMPAT, "architectures": ["8.6"]}
        return _completed(argv, payload=bad)

    report = preflight_base_image(
        BASE, kind="runtime-overlay", model_path=MODEL_PATH,
        conversion_report=CONVERSION_REPORT, model_size_bytes=MODEL_SIZE_BYTES, runner=runner,
    )

    assert report["status"] == "blocked"
    assert report["reason_codes"] == ["base_runtime_incompatible"]
    assert report["next_action"] == "choose_compatible_parent"
    assert len(commands) == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"cpu_checks": "failed"},
        {"cuobjdump_check": "skipped"},
        {"gpu_checks": "requested"},
    ],
)
def test_parent_must_pass_cpu_cuobjdump_and_no_gpu_checks(changed):
    def runner(argv, **kwargs):
        return _completed(argv, payload={**COMPAT, **changed})

    report = preflight_base_image(BASE, kind="model", runner=runner)

    assert report["status"] == "blocked"
    assert report["reason_codes"] == ["base_runtime_incompatible"]
    assert report["compatibility"]["gpu_checks"] == changed.get("gpu_checks", "not_requested")


def test_malformed_compatibility_json_is_incomplete_and_does_not_leak_output():
    def runner(argv, **kwargs):
        return SimpleNamespace(returncode=0, stdout="not-json " + SECRET_TEXT, stderr=SECRET_TEXT)

    report = preflight_base_image(BASE, kind="model", runner=runner)

    assert report["status"] == "incomplete"
    assert report["reason_codes"] == ["base_gate_output_invalid"]
    assert SECRET_TEXT not in json.dumps(report)


def test_unrecognized_compatibility_values_are_not_copied_into_report():
    def runner(argv, **kwargs):
        payload = {
            **COMPAT,
            "architectures": ["8.6", "8.9", SECRET_TEXT],
            "cpu_checks": SECRET_TEXT,
            "cuobjdump_check": SECRET_TEXT,
            "gpu_checks": SECRET_TEXT,
        }
        return _completed(argv, payload=payload)

    report = preflight_base_image(BASE, kind="model", runner=runner)

    assert report["status"] == "blocked"
    assert report["compatibility"]["architectures"] == ["8.6", "8.9"]
    assert report["compatibility"]["cpu_checks"] == "failed"
    assert SECRET_TEXT not in json.dumps(report)


def test_runtime_overlay_permission_failure_is_a_safe_reason_code():
    calls = 0

    def runner(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _completed(argv)
        return _completed(argv, returncode=1)

    report = preflight_base_image(
        BASE, kind="runtime-overlay", model_path=MODEL_PATH,
        conversion_report=CONVERSION_REPORT, model_size_bytes=MODEL_SIZE_BYTES, runner=runner,
    )

    assert report["status"] == "blocked"
    assert report["model_access"] == "not_checked"
    assert report["reason_codes"] == ["model_access_probe_failed"]
    assert report["next_action"] == "choose_model_image_with_runtime_uid_access"
    assert SECRET_TEXT not in json.dumps(report)


def test_timeout_and_failed_gate_have_stable_reasons():
    container_id = "c" * 64
    cleanup_commands = []

    def timeout_runner(argv, **kwargs):
        if argv[:2] == ["docker", "run"]:
            assert kwargs["timeout"] == 300
            cidfile = Path(argv[argv.index("--cidfile") + 1])
            cidfile.write_text(container_id)
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"], stderr=SECRET_TEXT)
        cleanup_commands.append((argv, kwargs))
        return _completed(argv)

    timeout = preflight_base_image(BASE, kind="model", runner=timeout_runner)
    assert timeout["status"] == "incomplete"
    assert timeout["reason_codes"] == ["base_preflight_timeout"]
    assert timeout["next_action"] == "retry_preflight"
    assert timeout["timeout_container_cleanup"] == "confirmed"
    assert cleanup_commands == [(
        ["docker", "rm", "--force", container_id],
        {"check": False, "capture_output": True, "text": True, "timeout": 30},
    )]

    failed = preflight_base_image(
        BASE, kind="model", runner=lambda argv, **_kwargs: _completed(argv, returncode=1),
    )
    assert failed["status"] == "blocked"
    assert failed["reason_codes"] == ["base_cpu_gate_failed_or_image_unavailable"]
    assert SECRET_TEXT not in json.dumps(failed)


def test_cli_timeout_progress_keeps_json_stdout_clean_and_retry_command_exact(capsys):
    container_id = "d" * 64
    commands = []

    def timeout_runner(argv, **kwargs):
        commands.append((argv, kwargs))
        if argv[:2] == ["docker", "run"]:
            assert kwargs["timeout"] == 417
            cidfile = Path(argv[argv.index("--cidfile") + 1])
            cidfile.write_text(container_id)
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output='partial', stderr=SECRET_TEXT)
        return _completed(argv)

    exit_code = main(
        ["--base-image", BASE, "--kind", "runtime-overlay",
         "--model-path", str(MODEL_PATH), "--conversion-report", str(CONVERSION_REPORT),
         "--model-size-bytes", str(MODEL_SIZE_BYTES), "--timeout", "417", "--json"],
        runner=timeout_runner,
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    report = json.loads(captured.out)
    expected_retry = shlex.join([
        sys.executable, "-m", "qwen_rl.deployment.instance_release",
        "--base-image", BASE, "--kind", "runtime-overlay",
        "--model-path", str(MODEL_PATH), "--conversion-report", str(CONVERSION_REPORT),
        "--model-size-bytes", str(MODEL_SIZE_BYTES), "--timeout", "417", "--json",
    ])
    assert report["status"] == "incomplete"
    assert report["reason_codes"] == ["base_preflight_timeout"]
    assert report["retry_command"] == expected_retry
    assert report["timeout_container_cleanup"] == "confirmed"
    assert len(commands) == 2
    assert commands[1][0] == ["docker", "rm", "--force", container_id]
    assert commands[0][0][:4] == ["docker", "run", "--pull=never", "--rm"]
    assert commands[0][0][commands[0][0].index("--cidfile") + 1].endswith("/container.id")
    assert SECRET_TEXT not in captured.out + captured.err
    assert "parent CPU compatibility gate: started; per-stage timeout=417s" in captured.err
    assert "parent CPU compatibility gate: timed out" in captured.err
    assert "timeout cleanup: removed the exact timed-out container" in captured.err
    assert captured.out.count("\n") == 1


def test_timeout_without_a_valid_container_id_stays_incomplete_without_guessing_target():
    commands = []

    def missing_id_runner(argv, **_kwargs):
        commands.append(argv)
        if argv[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(argv, 1)
        pytest.fail("cleanup must not guess a container target")

    report = preflight_base_image(BASE, kind="model", runner=missing_id_runner, timeout=1)

    assert report["status"] == "incomplete"
    assert report["reason_codes"] == ["base_preflight_timeout_cleanup_unconfirmed"]
    assert report["timeout_container_cleanup"] == "unconfirmed"
    assert report["next_action"] == "retry_preflight"
    assert len(commands) == 1


def test_cleanup_failure_does_not_turn_a_timed_out_gate_into_qualification():
    container_id = "e" * 64

    def timeout_runner(argv, **_kwargs):
        if argv[:2] == ["docker", "run"]:
            Path(argv[argv.index("--cidfile") + 1]).write_text(container_id)
            raise subprocess.TimeoutExpired(argv, 1)
        assert argv == ["docker", "rm", "--force", container_id]
        return _completed(argv, returncode=1, stderr=SECRET_TEXT)

    report = preflight_base_image(BASE, kind="model", runner=timeout_runner, timeout=1)

    assert report["status"] == "incomplete"
    assert report["reason_codes"] == ["base_preflight_timeout_cleanup_unconfirmed"]
    assert report["timeout_container_cleanup"] == "unconfirmed"
    assert SECRET_TEXT not in json.dumps(report)


@pytest.mark.parametrize("timeout", [0, -1, True, 1.5])
def test_invalid_api_timeout_fails_before_running_a_gate(timeout):
    with pytest.raises(ValueError, match="positive integer"):
        preflight_base_image(BASE, kind="model", timeout=timeout,
                             runner=lambda *_args, **_kwargs: pytest.fail("must reject timeout first"))


def test_runtime_overlay_without_model_profile_fails_closed_before_running_a_gate():
    report = preflight_base_image(
        BASE, kind="runtime-overlay",
        runner=lambda *_args, **_kwargs: pytest.fail("missing model metadata must not reach Docker"),
    )
    assert report["status"] == "incomplete"
    assert report["model_access"] == "unknown"
    assert report["reason_codes"] == ["model_access_profile_missing"]
    assert report["next_action"] == "provide_model_access_profile"


def test_runtime_overlay_cli_without_model_profile_returns_incomplete_json(capsys):
    exit_code = main(
        ["--base-image", BASE, "--kind", "runtime-overlay", "--json"],
        runner=lambda *_args, **_kwargs: pytest.fail("missing model metadata must not reach Docker"),
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    report = json.loads(captured.out)
    assert report["status"] == "incomplete"
    assert report["reason_codes"] == ["model_access_profile_missing"]
    assert report["model_access"] == "unknown"


@pytest.mark.parametrize("model_path,report_path,size", [
    (Path("relative/model.ninfer"), CONVERSION_REPORT, MODEL_SIZE_BYTES),
    (MODEL_PATH, MODEL_PATH, MODEL_SIZE_BYTES),
    (MODEL_PATH, CONVERSION_REPORT, True),
    (MODEL_PATH, CONVERSION_REPORT, 0),
    (MODEL_PATH, CONVERSION_REPORT, 1.5),
])
def test_invalid_runtime_model_profile_fails_closed_before_running_a_gate(model_path, report_path, size):
    report = preflight_base_image(
        BASE, kind="runtime-overlay", model_path=model_path, conversion_report=report_path,
        model_size_bytes=size,
        runner=lambda *_args, **_kwargs: pytest.fail("invalid model metadata must not reach Docker"),
    )
    assert report["status"] == "incomplete"
    assert report["model_access"] == "unknown"
    assert report["reason_codes"] == ["model_access_profile_invalid"]


@pytest.mark.parametrize("size", ["0", "-2", "1.5", "nope"])
def test_cli_rejects_invalid_model_size(size):
    from qwen_rl.deployment.instance_release import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--base-image", BASE, "--kind", "runtime-overlay",
            "--model-path", str(MODEL_PATH), "--conversion-report", str(CONVERSION_REPORT),
            "--model-size-bytes", size,
        ])


@pytest.mark.parametrize("timeout", ["0", "-2", "1.5", "nope"])
def test_cli_rejects_invalid_timeout(timeout):
    from qwen_rl.deployment.instance_release import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--base-image", BASE, "--kind", "model", "--timeout", timeout])


def test_invalid_base_reference_never_invokes_docker():
    def runner(*_args, **_kwargs):
        pytest.fail("invalid base reference must be rejected before Docker")

    report = preflight_base_image("latest", kind="model", runner=runner)

    assert report["status"] == "blocked"
    assert report["reason_codes"] == ["base_image_not_immutable"]
    assert report["next_action"] == "choose_compatible_parent"
