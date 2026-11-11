"""CPU-only preflight for immutable instance image parents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

from .instance_compat import LIGER_COMMIT

_BASE_IMAGE = re.compile(
    r"^(?:[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?/)?"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
    r"@sha256:[0-9a-f]{64}$"
)
_CUDA_VERSION = re.compile(r"^[0-9]{1,2}\.[0-9]{1,2}$")
_PYTHON = "/work/envs/rl-vllm/bin/python"
_MAX_REPORT_BYTES = 1024 * 1024
_TIMEOUT_SECONDS = 300
_CLEANUP_TIMEOUT_SECONDS = 30
_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")

def _model_access_probe(model_path: Path, conversion_report: Path, model_size_bytes: int) -> str:
    return f"""import json, os
from pathlib import Path
model = Path({str(model_path)!r})
report = Path({str(conversion_report)!r})
assert os.geteuid() == 10001
for parent in model.parents:
    assert os.access(parent, os.X_OK), 'model parent traversal denied'
assert model.stat().st_size == {model_size_bytes}
with model.open('rb') as stream:
    assert len(stream.read(16)) == 16
info = report.stat()
assert info.st_size <= {_MAX_REPORT_BYTES}
with report.open('rb') as stream:
    payload = stream.read({_MAX_REPORT_BYTES + 1})
assert len(payload) == info.st_size <= {_MAX_REPORT_BYTES}
assert isinstance(json.loads(payload), dict)
print(json.dumps({{'model_access': 'passed'}}))
"""


def _model_access_profile(
    model_path: str | Path | None,
    conversion_report: str | Path | None,
    model_size_bytes: int | None,
) -> tuple[Path, Path, int] | str:
    if model_path is None or conversion_report is None or model_size_bytes is None:
        return "missing"
    if (not isinstance(model_path, (str, Path))
            or not isinstance(conversion_report, (str, Path))
            or isinstance(model_size_bytes, bool) or not isinstance(model_size_bytes, int)
            or model_size_bytes <= 0):
        return "invalid"
    model = Path(model_path)
    report = Path(conversion_report)
    if not model.is_absolute() or not report.is_absolute() or model == report:
        return "invalid"
    return model, report, model_size_bytes


def _report(base_image: str | None, kind: str, *, status: str, reasons: list[str],
            next_action: str, compatibility: dict[str, Any] | None = None,
            model_access: str = "not_checked",
            timeout_container_cleanup: str = "not_needed") -> dict[str, Any]:
    return {
        "schema": "qwen-rl-instance-release-preflight/v2",
        "base_image": base_image,
        "kind": kind,
        "status": status,
        "compatibility": compatibility or {
            "architectures": [], "cuda": None, "cpu_checks": "unknown",
            "cuobjdump_check": "unknown", "gpu_checks": "not_requested",
        },
        "model_access": model_access,
        "reason_codes": reasons,
        "next_action": next_action,
        "retry_command": None,
        "timeout_container_cleanup": timeout_container_cleanup,
    }


def _docker_command(base_image: str, command: list[str], *, cidfile: str, uid: bool = True) -> list[str]:
    # Docker writes this run's full ID before starting it; timeout cleanup then
    # targets only that container, never a name pattern or the daemon globally.
    args = [
        "docker", "run", "--pull=never", "--rm", "--cidfile", cidfile, "--network", "none",
        "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
        "-e", "PYTHONPATH=/work", "--read-only", "--tmpfs", "/tmp",
    ]
    if uid:
        args.extend(("--user", "10001:10001"))
    args.extend(("--entrypoint", _PYTHON, base_image, "-B", *command))
    return args


def _cleanup_timed_out_container(
    cidfile: Path,
    *,
    runner: Callable[..., Any],
) -> tuple[bool, str]:
    try:
        container_id = cidfile.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return False, "no valid container ID was recorded"
    if _CONTAINER_ID.fullmatch(container_id) is None:
        return False, "no valid container ID was recorded"
    try:
        result = runner(
            ["docker", "rm", "--force", container_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=_CLEANUP_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "exact container removal could not be confirmed"
    if result.returncode == 0:
        return True, "removed the exact timed-out container"
    stderr = getattr(result, "stderr", "")
    if isinstance(stderr, str) and re.search(
        rf"\bNo such container:\s*{re.escape(container_id)}(?:\s|$)", stderr,
    ):
        return True, "exact timed-out container was already absent"
    return False, "exact container removal could not be confirmed"


def _run_json_gate(
    base_image: str,
    command: list[str],
    *,
    runner: Callable[..., Any],
    timeout: int,
    stage: str,
    progress: Callable[[str], None] | None,
) -> tuple[str, Any | None]:
    started = time.monotonic()
    if progress is not None:
        progress(f"{stage}: started; per-stage timeout={timeout}s")

    def finish(status: str, value: Any | None = None, *, progress_status: str | None = None) -> tuple[str, Any | None]:
        if progress is not None:
            elapsed = time.monotonic() - started
            progress(f"{stage}: {progress_status or status} after {elapsed:.1f}s")
        return status, value

    try:
        temporary = tempfile.TemporaryDirectory(prefix="qwen-rl-release-preflight-")
    except OSError:
        return finish("identity_unavailable", progress_status="could not allocate a private container ID file")
    with temporary as temporary_path:
        cidfile = Path(temporary_path) / "container.id"
        try:
            result = runner(
                _docker_command(base_image, command, cidfile=str(cidfile)),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            cleanup_confirmed, cleanup_detail = _cleanup_timed_out_container(cidfile, runner=runner)
            cleanup_progress = f"timeout cleanup: {cleanup_detail}"
            if progress is not None:
                progress(cleanup_progress)
            status = "timeout" if cleanup_confirmed else "timeout_cleanup_unconfirmed"
            return finish(status, progress_status=f"timed out; {cleanup_detail}")
        except OSError:
            return finish("docker_unavailable")
    if result.returncode != 0:
        # Map only known gate messages to static codes; never echo stderr.
        stderr = getattr(result, "stderr", "")
        stderr = stderr[:32768] if isinstance(stderr, str) else ""
        known = {
            "installed T2 binary lacks required cubin: sm_89": "base_missing_sm89",
            "installed T2 binary lacks required cubin: sm_86": "base_missing_sm86",
            "instance wheel lacks required CUDA targets: 8.9": "base_missing_sm89",
            "instance wheel lacks required CUDA targets: 8.6": "base_missing_sm86",
            "T2 kernel wheel must match the installed Torch/CUDA": "base_torch_cuda_kernel_mismatch",
            "model parent traversal denied": "model_parent_traversal_denied",
        }
        for message, code in known.items():
            if message in stderr:
                return finish(code)
        if result.returncode == 125:
            return finish("base_container_unavailable")
        return finish("gate_failed")
    stdout = result.stdout
    if not isinstance(stdout, str) or len(stdout.encode("utf-8", errors="replace")) > 64 * 1024:
        return finish("invalid_output")
    try:
        value = json.loads(stdout)
    except (json.JSONDecodeError, UnicodeError):
        return finish("invalid_output")
    if not isinstance(value, dict):
        return finish("invalid_output")
    return finish("passed", value)


def preflight_base_image(
    base_image: str,
    *,
    kind: str,
    model_path: str | Path | None = None,
    conversion_report: str | Path | None = None,
    model_size_bytes: int | None = None,
    runner: Callable[..., Any] | None = None,
    timeout: int = _TIMEOUT_SECONDS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Check a local immutable parent without pulling, building, or using GPUs."""
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer number of seconds")
    if runner is None:
        runner = subprocess.run
    supported_kind = kind if isinstance(kind, str) and kind in {"model", "runtime-overlay"} else "unsupported"
    if not isinstance(base_image, str) or _BASE_IMAGE.fullmatch(base_image) is None:
        return _report(
            None, supported_kind,
            status="blocked", reasons=["base_image_not_immutable"],
            next_action="choose_compatible_parent",
        )
    if supported_kind == "unsupported":
        return _report(
            base_image, "unsupported", status="blocked", reasons=["unsupported_kind"],
            next_action="select_supported_image_kind",
        )
    model_profile: tuple[Path, Path, int] | None = None
    if kind == "runtime-overlay":
        checked_profile = _model_access_profile(model_path, conversion_report, model_size_bytes)
        if isinstance(checked_profile, str):
            missing = checked_profile == "missing"
            return _report(
                base_image, kind, status="incomplete",
                reasons=["model_access_profile_missing" if missing else "model_access_profile_invalid"],
                next_action="provide_model_access_profile",
                model_access="unknown",
            )
        model_profile = checked_profile

    gate_status, compat = _run_json_gate(
        base_image,
        ["-m", "qwen_rl.deployment.instance_compat"],
        runner=runner,
        timeout=timeout,
        stage="parent CPU compatibility gate",
        progress=progress,
    )
    if gate_status != "passed":
        if gate_status == "timeout":
            reason, action = "base_preflight_timeout", "retry_preflight"
            status = "incomplete"
        elif gate_status == "timeout_cleanup_unconfirmed":
            reason, action = "base_preflight_timeout_cleanup_unconfirmed", "retry_preflight"
            status = "incomplete"
        elif gate_status == "identity_unavailable":
            reason, action = "preflight_container_identity_unavailable", "retry_preflight"
            status = "incomplete"
        elif gate_status == "docker_unavailable":
            reason, action = "docker_unavailable", "make_docker_available_and_retry"
            status = "blocked"
        elif gate_status == "invalid_output":
            reason, action = "base_gate_output_invalid", "choose_compatible_parent"
            status = "incomplete"
        elif gate_status in {"base_missing_sm89", "base_missing_sm86", "base_torch_cuda_kernel_mismatch"}:
            reason, action, status = gate_status, "choose_compatible_parent", "blocked"
        elif gate_status == "base_container_unavailable":
            reason, action, status = gate_status, "check_local_pinned_image_and_docker", "incomplete"
        else:
            reason, action = "base_cpu_gate_failed_or_image_unavailable", "choose_compatible_parent"
            status = "blocked"
        cleanup = (
            "confirmed" if gate_status == "timeout"
            else "unconfirmed" if gate_status == "timeout_cleanup_unconfirmed"
            else "not_needed"
        )
        return _report(
            base_image, kind, status=status, reasons=[reason], next_action=action,
            timeout_container_cleanup=cleanup,
        )

    architectures = compat.get("architectures")
    cuda = compat.get("cuda")
    cpu_checks = compat.get("cpu_checks")
    cuobjdump_check = compat.get("cuobjdump_check")
    gpu_checks = compat.get("gpu_checks")
    fields_valid = (
        isinstance(architectures, list) and all(isinstance(value, str) for value in architectures)
        and isinstance(cuda, str) and _CUDA_VERSION.fullmatch(cuda) is not None
        and isinstance(cpu_checks, str) and isinstance(cuobjdump_check, str)
        and isinstance(gpu_checks, str)
    )
    compatibility = {
        "architectures": sorted(set(architectures) & {"8.6", "8.9"}) if fields_valid else [],
        "cuda": cuda if fields_valid else None,
        "cpu_checks": "passed" if fields_valid and cpu_checks == "passed" else "failed" if fields_valid else "unknown",
        "cuobjdump_check": (
            cuobjdump_check if fields_valid and cuobjdump_check in {"passed", "skipped"}
            else "failed" if fields_valid else "unknown"
        ),
        "gpu_checks": "not_requested" if fields_valid and gpu_checks == "not_requested" else "requested" if fields_valid else "not_requested",
    }
    if not fields_valid:
        return _report(
            base_image, kind, status="incomplete", reasons=["base_gate_output_incomplete"],
            next_action="retry_preflight", compatibility=compatibility,
        )
    required = {"8.6", "8.9"}
    if (not required.issubset(compatibility["architectures"])
            or cpu_checks != "passed" or cuobjdump_check != "passed"
            or gpu_checks != "not_requested"):
        return _report(
            base_image, kind, status="blocked", reasons=["base_runtime_incompatible"],
            next_action="choose_compatible_parent", compatibility=compatibility,
        )
    # A legacy gate can report generic CPU success without the frozen-source
    # check. Its result is insufficient for a new release.
    if "liger_commit" not in compat or "liger_provenance" not in compat:
        return _report(
            base_image, kind, status="incomplete", reasons=["base_liger_provenance_missing"],
            next_action="choose_qualified_parent_with_source_provenance", compatibility=compatibility,
        )
    if (compat["liger_commit"] != LIGER_COMMIT
            or not isinstance(compat["liger_provenance"], str)
            or compat["liger_provenance"] not in {"git-direct-url", "offline-wheel-receipt"}):
        return _report(
            base_image, kind, status="blocked", reasons=["base_liger_provenance_invalid"],
            next_action="choose_qualified_parent_with_source_provenance", compatibility=compatibility,
        )
    compatibility.update(liger_commit=LIGER_COMMIT, liger_provenance=compat["liger_provenance"])

    model_access = "not_checked"
    if kind == "runtime-overlay":
        access_status, access = _run_json_gate(
            base_image,
            ["-c", _model_access_probe(*model_profile)],
            runner=runner,
            timeout=timeout,
            stage="runtime-user model access probe",
            progress=progress,
        )
        if access_status == "timeout":
            return _report(
                base_image, kind, status="incomplete", reasons=["model_access_probe_timeout"],
                next_action="retry_preflight", compatibility=compatibility,
                timeout_container_cleanup="confirmed",
            )
        if access_status == "timeout_cleanup_unconfirmed":
            return _report(
                base_image, kind, status="incomplete",
                reasons=["model_access_probe_timeout_cleanup_unconfirmed"],
                next_action="retry_preflight", compatibility=compatibility,
                timeout_container_cleanup="unconfirmed",
            )
        if access_status == "identity_unavailable":
            return _report(
                base_image, kind, status="incomplete",
                reasons=["preflight_container_identity_unavailable"],
                next_action="retry_preflight", compatibility=compatibility,
            )
        if access_status == "docker_unavailable":
            return _report(
                base_image, kind, status="blocked", reasons=["docker_unavailable"],
                next_action="make_docker_available_and_retry", compatibility=compatibility,
            )
        if access_status in {"gate_failed", "model_parent_traversal_denied"}:
            return _report(
                base_image, kind, status="blocked", reasons=[
                    "model_parent_traversal_denied" if access_status == "model_parent_traversal_denied"
                    else "model_access_probe_failed"],
                next_action="choose_model_image_with_runtime_uid_access", compatibility=compatibility,
            )
        if access_status != "passed" or access != {"model_access": "passed"}:
            return _report(
                base_image, kind, status="incomplete", reasons=["model_access_probe_incomplete"],
                next_action="retry_preflight", compatibility=compatibility,
            )
        model_access = "passed_as_uid_10001"

    return _report(
        base_image, kind, status="ready", reasons=[],
        next_action=("build_full_model_image" if kind == "model" else "build_runtime_overlay"),
        compatibility=compatibility, model_access=model_access,
    )


def _positive_timeout(value: str) -> int:
    try:
        timeout = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer number of seconds") from error
    if timeout <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer number of seconds")
    return timeout


def _positive_size_bytes(value: str) -> int:
    try:
        size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer number of bytes") from error
    if size <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer number of bytes")
    return size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True, help="local immutable repository@sha256 digest")
    parser.add_argument("--kind", required=True)
    parser.add_argument("--model-path", type=Path,
                        help="absolute model path inside the parent image (runtime-overlay only)")
    parser.add_argument("--conversion-report", type=Path,
                        help="absolute NInfer conversion report path inside the parent image")
    parser.add_argument("--model-size-bytes", type=_positive_size_bytes,
                        help="exact expected model size from the selected profile metadata")
    parser.add_argument("--timeout", type=_positive_timeout, default=_TIMEOUT_SECONDS,
                        metavar="SECONDS", help="timeout for each parent container gate (default: 300)")
    parser.add_argument("--json", action="store_true", help="print the sanitized machine report")
    return parser


def _retry_command(base_image: str, kind: str, timeout: int,
                   model_profile: tuple[Path, Path, int] | None = None) -> str:
    command = [
        sys.executable, "-m", "qwen_rl.deployment.instance_release",
        "--base-image", base_image, "--kind", kind,
    ]
    if model_profile is not None:
        model_path, conversion_report, model_size_bytes = model_profile
        command.extend((
            "--model-path", str(model_path),
            "--conversion-report", str(conversion_report),
            "--model-size-bytes", str(model_size_bytes),
        ))
    command.extend(("--timeout", str(timeout), "--json"))
    return shlex.join(command)


def _stderr_progress(message: str) -> None:
    print(f"instance release preflight: {message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None, *, runner: Callable[..., Any] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_profile = None
    if (args.kind == "runtime-overlay" and args.model_path is not None
            and args.conversion_report is not None and args.model_size_bytes is not None):
        model_profile = (args.model_path, args.conversion_report, args.model_size_bytes)
    result = preflight_base_image(
        args.base_image, kind=args.kind, model_path=args.model_path,
        conversion_report=args.conversion_report, model_size_bytes=args.model_size_bytes,
        runner=runner, timeout=args.timeout, progress=_stderr_progress,
    )
    if result["next_action"] == "retry_preflight":
        result["retry_command"] = _retry_command(args.base_image, args.kind, args.timeout, model_profile)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        outcome = result["status"]
        detail = ",".join(result["reason_codes"]) or "all required checks passed"
        retry = f"; retry_command={result['retry_command']}" if result["retry_command"] else ""
        print(f"{outcome}: {detail}; next_action={result['next_action']}{retry}")
    return {"ready": 0, "blocked": 1, "incomplete": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
