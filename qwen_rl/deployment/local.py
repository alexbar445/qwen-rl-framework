"""Secure host launcher for the fixed local Docker deployment."""
from __future__ import annotations

import fcntl
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..cluster.worker_release import WorkerRelease


ROOT = Path(__file__).resolve().parents[2]
GRADIENT_DEPLOYMENT_PATH = ROOT / "configs" / "local-bonsai2-async-bepa-deployment.json"
SECRET_NAMES = ("ARC_ENV_TOKEN", "BEPA_TEACHER_TOKEN", "DEEPSEEK_API_KEY")
_DOCKER_IMAGE_REF = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}|@sha256:[0-9a-f]{64})$"
)
EXPECTED_BASE_URL_ENV = "QWEN_RL_EXPECTED_TEACHER_BASE_URL"
EXPECTED_MODEL_ENV = "QWEN_RL_EXPECTED_TEACHER_MODEL"
# Role stdout is redirected straight into runs/<run_id>/logs/<role>-<worker|main>.log.  Python
# block-buffers stdout when it is a file, so without this the run log stays empty until the
# 8 KiB buffer fills (the dashboard log view then shows only the startup lines).
UNBUFFERED_ENV = "PYTHONUNBUFFERED"
COMMANDS = {"start", "check", "status", "watch", "run", "stop", "selftest", "bepa", "evaluate"}
PREFLIGHT = r'''
import json, os, sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from qwen_rl.runtime.config import load_config
from qwen_rl.runtime.manager import validate_bepa_environment, validate_environment, validate_paths

def fail(label):
    print("local deployment preflight failed: " + label, file=sys.stderr)
    raise SystemExit(2)

try:
    # Read-only host preflight; the generated runtime is validated again at start.
    config = load_config(Path(sys.argv[1]), for_local_worker_plan=True)
    validate_paths(config, role="cluster")
    validate_environment()
    teacher = None
    expected_url = None
    if config.bepa.enabled:
        validate_bepa_environment(config)
        teacher = config.bepa.teacher
        expected_url = os.environ.get("QWEN_RL_EXPECTED_TEACHER_BASE_URL", "")
        expected_model = os.environ.get("QWEN_RL_EXPECTED_TEACHER_MODEL", "")
        if teacher.base_url.rstrip("/") != expected_url:
            fail("configured teacher endpoint does not match the inspected Docker bridge")
        if teacher.model != expected_model:
            fail("configured teacher model does not match the deployment model")
        parsed = urlsplit(teacher.base_url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
            fail("configured teacher endpoint is invalid")
        url = parsed.scheme + "://" + parsed.netloc + os.environ["QWEN_RL_MODELS_PATH"]
        api_key = os.environ.get(teacher.api_key_env, "")
        request = Request(url, headers={"Authorization": "Bearer " + api_key,
                                       "Accept": "application/json"})
        try:
            with urlopen(request, timeout=float(os.environ["QWEN_RL_PROBE_TIMEOUT"])) as response:
                payload = json.loads(response.read())
        except Exception:
            fail("authenticated teacher model probe failed")
        rows = payload.get("data") if isinstance(payload, dict) else None
        available = {row.get("id") for row in rows if isinstance(row, dict)} if isinstance(rows, list) else set()
        if teacher.model not in available:
            fail("configured teacher model is unavailable")
except SystemExit:
    raise
except Exception as exc:
    fail("container runtime validation failed: " + str(exc))

print(json.dumps({
    "status": "passed",
    "teacher": "ready" if teacher is not None else "disabled",
    "model": teacher.model if teacher is not None else None,
    "teacher_sdk": "openai==3.26.1" if teacher is not None else None,
    "teacher_endpoint": expected_url,
}))
'''


class DeploymentError(RuntimeError):
    """An expected, user-safe local deployment failure."""


class CommandExit(Exception):
    def __init__(self, status: int):
        self.status = status


@dataclass(frozen=True)
class DeploymentConfig:
    path: Path
    container: str
    container_workdir: str
    container_python: str
    container_uid: str
    worker_image: str | None
    worker_release: WorkerRelease | None
    default_config: str
    runtime_env: Mapping[str, str]
    arc_token_path: str
    teacher_token_path: str
    provider_key_path: str
    bridge_listen_port: int
    bridge_upstream_port: int
    models_path: str
    required_model: str
    probe_timeout: float
    bridge_start_timeout: float
    bridge_stop_timeout: float
    docker_timeout: float
    start_timeout: float
    worker_start_timeout: float
    command_timeout: float


_TOP_KEYS = {
    "schema_version", "container", "container_workdir", "container_python",
    "container_uid", "worker_image", "default_config", "runtime_env", "credentials", "bridge", "timeouts",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("deployment JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("deployment JSON cannot contain non-finite numbers")


def _object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DeploymentError(f"deployment {label} fields do not match the strict schema")
    return value


def _string(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        raise DeploymentError(f"deployment {label} must be a non-empty string")
    return value


def _positive_number(value: Any, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise DeploymentError(f"deployment {label} must be positive")
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 65535:
        raise DeploymentError(f"deployment {label} must be a valid TCP port")
    return value


def load_deployment(path: str | Path) -> DeploymentConfig:
    deployment_path = Path(path).expanduser().resolve(strict=True)
    try:
        raw = json.loads(deployment_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                         parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise DeploymentError("deployment JSON cannot be read") from None
    required_top_keys = _TOP_KEYS - {"worker_image"}
    if (not isinstance(raw, dict) or not required_top_keys.issubset(raw)
            or set(raw) - (_TOP_KEYS | {"worker_release"})):
        raise DeploymentError("deployment root fields do not match the strict schema")
    top = raw
    if top["schema_version"] != "qwen-rl-local-deployment/v1":
        raise DeploymentError("deployment schema_version is unsupported")
    runtime_env = _object(top["runtime_env"], {"ARC_RL_ROOT", "HF_HOME", "CUDA_HOME"}, "runtime_env")
    credentials = _object(top["credentials"], {"arc_token", "teacher_token", "provider_key"}, "credentials")
    bridge = _object(top["bridge"], {
        "gateway_listen_port", "upstream_host_port", "models_path", "required_model",
        "probe_timeout_seconds", "start_timeout_seconds", "stop_timeout_seconds",
    }, "bridge")
    timeouts = _object(top["timeouts"], {
        "docker_seconds", "manager_start_seconds", "worker_start_seconds", "command_seconds",
    }, "timeouts")
    if any(not isinstance(value, str) for value in runtime_env.values()):
        raise DeploymentError("deployment runtime_env values must be strings")
    worker_image_value = top.get("worker_image")
    worker_release_value = top.get("worker_release")
    if (worker_image_value is None) == (worker_release_value is None):
        raise DeploymentError("deployment must configure exactly one of worker_image or worker_release")
    worker_image = _string(worker_image_value, "worker_image") if worker_image_value is not None else None
    worker_release = None
    if worker_release_value is not None:
        configured_release = Path(_string(worker_release_value, "worker_release")).expanduser()
        if not configured_release.is_absolute():
            configured_release = deployment_path.parent / configured_release
        if configured_release.is_symlink():
            raise DeploymentError("deployment worker_release must not be a symbolic link")
        try:
            worker_release = configured_release.resolve(strict=True)
        except OSError:
            raise DeploymentError("deployment worker_release path is unavailable") from None
        if not worker_release.is_file():
            raise DeploymentError("deployment worker_release must be a regular JSON file")
        try:
            from ..cluster.worker_release import WorkerReleaseError, load_worker_release

            worker_release = load_worker_release(worker_release)
        except WorkerReleaseError as exc:
            raise DeploymentError(f"deployment worker_release is invalid: {exc}") from exc
    config = DeploymentConfig(
        path=deployment_path,
        container=_string(top["container"], "container"),
        container_workdir=_string(top["container_workdir"], "container_workdir"),
        container_python=_string(top["container_python"], "container_python"),
        container_uid=_string(top["container_uid"], "container_uid"),
        worker_image=worker_image,
        worker_release=worker_release,
        default_config=_string(top["default_config"], "default_config"),
        runtime_env=dict(runtime_env),
        arc_token_path=_string(credentials["arc_token"], "credentials.arc_token"),
        teacher_token_path=_string(credentials["teacher_token"], "credentials.teacher_token"),
        provider_key_path=_string(credentials["provider_key"], "credentials.provider_key"),
        bridge_listen_port=_positive_int(bridge["gateway_listen_port"], "bridge.gateway_listen_port"),
        bridge_upstream_port=_positive_int(bridge["upstream_host_port"], "bridge.upstream_host_port"),
        models_path=_string(bridge["models_path"], "bridge.models_path"),
        required_model=_string(bridge["required_model"], "bridge.required_model"),
        probe_timeout=_positive_number(bridge["probe_timeout_seconds"], "bridge.probe_timeout_seconds"),
        bridge_start_timeout=_positive_number(bridge["start_timeout_seconds"], "bridge.start_timeout_seconds"),
        bridge_stop_timeout=_positive_number(bridge["stop_timeout_seconds"], "bridge.stop_timeout_seconds"),
        docker_timeout=_positive_number(timeouts["docker_seconds"], "timeouts.docker_seconds"),
        start_timeout=_positive_number(timeouts["manager_start_seconds"], "timeouts.manager_start_seconds"),
        worker_start_timeout=_positive_number(timeouts["worker_start_seconds"], "timeouts.worker_start_seconds"),
        command_timeout=_positive_number(timeouts["command_seconds"], "timeouts.command_seconds"),
    )
    if re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", config.container_uid) is None:
        raise DeploymentError("deployment container_uid must be a non-root numeric uid:gid pair")
    if config.worker_image is not None and _DOCKER_IMAGE_REF.fullmatch(config.worker_image) is None:
        raise DeploymentError("deployment worker_image must be an explicit image tag or SHA-256 reference")
    if config.container_workdir != "/work" or not config.default_config.startswith("/work/"):
        raise DeploymentError("deployment container paths must use the /work mount")
    if not config.container_python.startswith("/work/"):
        raise DeploymentError("deployment Python path must be inside /work")
    if config.runtime_env != {"ARC_RL_ROOT": "/work", "HF_HOME": "/work/envs/hf-cache",
                              "CUDA_HOME": "/usr/local/cuda"}:
        raise DeploymentError("deployment runtime_env does not match the local container contract")
    if not config.models_path.startswith("/") or not config.models_path.endswith("/models"):
        raise DeploymentError("deployment bridge models_path must be an absolute /models path")
    return config


def _read_secret_file(path: Path, *, label: str, uid: int | None = None) -> str:
    expected_uid = os.getuid() if uid is None else uid
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise DeploymentError(f"{label} credential file is missing or is not a regular file: {path}") from None
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != expected_uid
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise DeploymentError(f"{label} credential file has unsafe type, owner, link count, or mode: {path}")
        chunks: list[bytes] = []
        remaining = 16385
        while remaining:
            chunk = os.read(fd, min(4096, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > 16384:
            raise DeploymentError(f"{label} credential file is too large: {path}")
        try:
            secret = content.decode("utf-8").strip()
        except UnicodeError:
            raise DeploymentError(f"{label} credential file is not valid UTF-8: {path}") from None
        if not secret or any(ord(character) < 32 or ord(character) == 127 for character in secret):
            raise DeploymentError(f"{label} credential file is empty or invalid: {path}")
        return secret
    finally:
        os.close(fd)


def _create_secret_file(path: Path, *, label: str, uid: int | None = None) -> str:
    expected_uid = os.getuid() if uid is None else uid
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    value = secrets.token_urlsafe(48)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        return _read_secret_file(path, label=label, uid=expected_uid)
    except OSError:
        raise DeploymentError(f"could not securely create {label} credential file") from None
    try:
        os.fchmod(fd, 0o600)
        info = os.fstat(fd)
        if info.st_uid != expected_uid or stat.S_IMODE(info.st_mode) != 0o600:
            raise DeploymentError(f"new {label} credential file has unexpected owner or mode")
        data = (value + "\n").encode("ascii")
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.close(fd)
    return value


def read_or_create_secret(path: Path, *, label: str, create: bool) -> str:
    try:
        path.lstat()
    except FileNotFoundError:
        if create:
            return _create_secret_file(path, label=label)
        if label == "ARC":
            raise DeploymentError(
                f"ARC credential file is missing: {path}; run ./bin/qwen-rl check or start to initialize it"
            ) from None
        raise DeploymentError(f"{label} credential file is missing: {path}") from None
    except OSError:
        raise DeploymentError(f"{label} credential file cannot be inspected") from None
    return _read_secret_file(path, label=label)


def _proc_identity(pid: int, proc_root: Path = Path("/proc")) -> tuple[int, tuple[str, ...]] | None:
    try:
        stat_line = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
        cmdline = (proc_root / str(pid) / "cmdline").read_bytes()
    except (OSError, UnicodeError):
        return None
    end = stat_line.rfind(")")
    if end < 0:
        return None
    fields = stat_line[end + 1:].split()
    if len(fields) <= 19:
        return None
    try:
        starttime = int(fields[19])
        argv = tuple(item.decode("utf-8") for item in cmdline.rstrip(b"\0").split(b"\0"))
    except (ValueError, UnicodeError):
        return None
    if not argv:
        return None
    return starttime, argv


def bridge_process_matches(state: Mapping[str, Any], *, proc_root: Path = Path("/proc")) -> bool:
    pid = state.get("pid")
    expected_start = state.get("starttime")
    expected_argv = state.get("argv")
    if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1
            or isinstance(expected_start, bool) or not isinstance(expected_start, int)
            or not isinstance(expected_argv, list)
            or any(not isinstance(item, str) for item in expected_argv)):
        return False
    current = _proc_identity(pid, proc_root)
    return current == (expected_start, tuple(expected_argv))


class LocalDeployment:
    def __init__(
        self,
        root: Path,
        config: DeploymentConfig,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        environ: Mapping[str, str] | None = None,
        stdin: Any = None,
        stdout: Any = None,
        stderr: Any = None,
        proc_root: Path = Path("/proc"),
        socket_factory: Callable[..., socket.socket] = socket.socket,
        signal_sender: Callable[[int, int], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.root = Path(root).resolve()
        self.config = config
        self.runner = runner
        self.popen = popen
        self.environ = dict(os.environ if environ is None else environ)
        self.stdin = sys.stdin if stdin is None else stdin
        self.stdout = sys.stdout if stdout is None else stdout
        self.stderr = sys.stderr if stderr is None else stderr
        self.proc_root = Path(proc_root)
        self.socket_factory = socket_factory
        self.signal_sender = os.kill if signal_sender is None else signal_sender
        self.sleep = sleep
        self.monotonic = monotonic
        self.runtime_dir = self.root / "outputs" / "local-runtime"
        self.bridge_state_path = self.runtime_dir / "teacher-bridge.json"
        self.lock_path = self.runtime_dir / "deployment.lock"
        self._active_redactions: tuple[str, ...] = ()

    def _worker_fleet(self):
        from .workers import LocalWorkerFleet

        return LocalWorkerFleet(
            root=self.root,
            deployment_path=self.config.path,
            manager_container=self.config.container,
            worker_image=self.config.worker_image,
            worker_release=self.config.worker_release,
            container_uid=self.config.container_uid,
            default_config=self.config.default_config,
            container_workdir=self.config.container_workdir,
            container_python=self.config.container_python,
            worker_start_timeout=self.config.worker_start_timeout,
            command_env=self._base_child_env(),
            runner=self.runner,
            sleep=self.sleep,
            monotonic=self.monotonic,
        )

    def _configured_update_backend(self) -> str:
        """Read the source profile without starting roles or touching GPU state."""
        configured = Path(self.config.default_config)
        try:
            relative = configured.relative_to("/work")
        except ValueError:
            raise DeploymentError("local runtime profile must be inside the /work mount") from None
        profile = self.root / relative
        try:
            from ..runtime.config import load_config

            return load_config(profile, for_local_worker_plan=True).learner.update_backend
        except Exception as exc:
            raise DeploymentError(f"local runtime profile is invalid ({type(exc).__name__})") from exc

    def _assert_default_gradient_inactive(self) -> None:
        """Do not let an explicitly selected legacy deployment stop the default stack."""
        gradient_path = self.root / "configs" / GRADIENT_DEPLOYMENT_PATH.name
        if not gradient_path.is_file() or self.config.path.resolve() == gradient_path.resolve():
            return
        try:
            legacy_credentials = self._load_credentials(teacher=False)
            legacy_snapshot = self._manager_snapshot(legacy_credentials)
            if legacy_snapshot is None:
                raise DeploymentError("cannot verify the selected legacy RuntimeManager before stop")
            legacy_live = any(row.get("identity_matches") is True for row in legacy_snapshot["processes"])
            gradient_config = load_deployment(gradient_path)
            gradient_deployment = LocalDeployment(
                self.root, gradient_config, runner=self.runner, popen=self.popen,
                environ=self.environ, stdin=self.stdin, stdout=self.stdout, stderr=self.stderr,
                proc_root=self.proc_root, socket_factory=self.socket_factory,
                signal_sender=self.signal_sender, sleep=self.sleep, monotonic=self.monotonic,
            )
            from .local_gradients import LocalGradientHost

            status = LocalGradientHost(gradient_deployment).status(gradient_deployment._worker_fleet())
        except Exception as exc:
            raise DeploymentError(f"cannot verify the default central-gradient deployment ({type(exc).__name__})") from exc
        if status.get("status") != "stopped":
            raise DeploymentError("the default central-gradient deployment is active; stop it with its deployment profile")
        try:
            from ..runtime.config import load_config

            central_profile = self.root / Path(gradient_config.default_config).relative_to("/work")
            central_runtime = load_config(central_profile, for_local_worker_plan=True)
            LocalGradientHost(gradient_deployment)._assert_learner_port_free(central_runtime)
        except DeploymentError:
            if not legacy_live:
                raise
        except Exception as exc:
            raise DeploymentError(f"cannot verify central-gradient port ownership ({type(exc).__name__})") from exc

    def _runtime_paths(self) -> tuple[str, str | None]:
        persisted = self._worker_fleet().existing_runtime_paths()
        if persisted is None:
            return self.config.default_config, None
        return persisted

    @contextmanager
    def lifecycle_lock(self) -> Iterator[None]:
        self.runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            info = self.runtime_dir.lstat()
        except OSError:
            raise DeploymentError("local runtime state directory cannot be inspected") from None
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
            raise DeploymentError("local runtime state directory must be private and owned by the caller")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.lock_path, flags, 0o600)
        except OSError:
            raise DeploymentError("deployment lifecycle lock cannot be opened") from None
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600):
                raise DeploymentError("deployment lifecycle lock has unsafe owner or mode")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _credential_paths(self) -> tuple[Path, Path, Path]:
        def resolve(value: str) -> Path:
            path = Path(value).expanduser()
            return path if path.is_absolute() else self.root / path

        arc = resolve(self.config.arc_token_path)
        teacher = resolve(self.config.teacher_token_path)
        provider = resolve(self.config.provider_key_path)
        return arc, teacher, provider

    def _load_credentials(
        self, *, teacher: bool, create_arc: bool | None = None,
    ) -> dict[str, str]:
        arc_path, teacher_path, provider_path = self._credential_paths()
        arc_token = read_or_create_secret(
            arc_path, label="ARC", create=teacher if create_arc is None else create_arc,
        )
        values = {"ARC_ENV_TOKEN": arc_token}
        if teacher:
            teacher_token = read_or_create_secret(teacher_path, label="BEPA teacher", create=True)
            provider_key = read_or_create_secret(provider_path, label="provider", create=False)
            if len({arc_token, teacher_token, provider_key}) != 3:
                raise DeploymentError("ARC, teacher, and provider credentials must be distinct")
            values["BEPA_TEACHER_TOKEN"] = teacher_token
            values["DEEPSEEK_API_KEY"] = provider_key
        return values

    def _base_child_env(self, injected: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(self.environ)
        for key in (*SECRET_NAMES, EXPECTED_BASE_URL_ENV, EXPECTED_MODEL_ENV, "QWEN_RL_MODELS_PATH",
                    "QWEN_RL_PROBE_TIMEOUT", *self.config.runtime_env):
            env.pop(key, None)
        env.update(self.config.runtime_env)
        if injected:
            env.update(injected)
        return env

    def _redact(self, text: str, secrets_to_redact: Sequence[str] | None = None) -> str:
        secrets_ordered = sorted((s for s in (secrets_to_redact or self._active_redactions) if s),
                                 key=len, reverse=True)
        for value in secrets_ordered:
            text = text.replace(value, "[REDACTED]")
        return text

    def _capture(self, argv: Sequence[str], *, env: Mapping[str, str], timeout: float) -> subprocess.CompletedProcess[str]:
        try:
            result = self.runner(list(argv), env=dict(env), text=True, capture_output=True,
                                 timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            raise DeploymentError("Docker operation timed out") from None
        except OSError:
            raise DeploymentError("Docker command could not be started") from None
        return subprocess.CompletedProcess(list(argv), result.returncode,
                                           self._redact(result.stdout or ""),
                                           self._redact(result.stderr or ""))

    def _inherit_output(self, argv: Sequence[str], *, env: Mapping[str, str],
                        timeout: float) -> subprocess.CompletedProcess[str]:
        try:
            return self.runner(list(argv), env=dict(env), stdin=None, stdout=None, stderr=None,
                               timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            raise DeploymentError("Docker command timed out") from None
        except OSError:
            raise DeploymentError("Docker command could not be started") from None

    def _exec_container(
        self,
        container_argv: Sequence[str],
        *,
        credentials: Mapping[str, str] | None = None,
        injected: Mapping[str, str] | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        values = dict(injected or {})
        if credentials:
            values.update(credentials)
            joined_args = "\0".join(container_argv)
            if any(value and value in joined_args for value in credentials.values()):
                raise DeploymentError("credential values cannot be passed as command-line arguments")
        env_names = [*self.config.runtime_env]
        if credentials:
            env_names.extend(credentials)
        if injected:
            env_names.extend(injected)
        env_names.append(UNBUFFERED_ENV)
        child_env = self._base_child_env({**values, UNBUFFERED_ENV: "1"})
        argv = ["docker", "exec"]
        try:
            stdin_tty = bool(self.stdin.isatty())
            stdout_tty = bool(self.stdout.isatty())
            stderr_tty = bool(self.stderr.isatty())
        except (AttributeError, OSError):
            stdin_tty = stdout_tty = stderr_tty = False
        owned_watch = bool(container_argv and container_argv[0] == "watch"
                           and "--exit-on-stdin-eof" in container_argv)
        if stdin_tty or owned_watch:
            argv.append("-i")
        if stdout_tty and stderr_tty:
            argv.append("-t")
        argv.extend(("-u", self.config.container_uid, "-w", self.config.container_workdir))
        for name in env_names:
            if name not in child_env:
                raise DeploymentError(f"required environment value {name} is missing")
            argv.extend(("-e", name))
        argv.extend((self.config.container, self.config.container_python, "-B", "-m", "qwen_rl", "runtime"))
        argv.extend(container_argv)
        self._active_redactions = tuple((credentials or {}).values())
        try:
            if stream:
                result = self._inherit_output(argv, env=child_env,
                                              timeout=timeout or self.config.command_timeout)
            else:
                result = self._capture(argv, env=child_env, timeout=timeout or self.config.command_timeout)
        finally:
            self._active_redactions = ()
        return result

    def _inspect_network(self) -> tuple[str, str]:
        argv = ["docker", "inspect", "--format", "{{json .NetworkSettings.Networks}}", self.config.container]
        result = self._capture(argv, env=self._base_child_env(), timeout=self.config.docker_timeout)
        if result.returncode != 0:
            raise DeploymentError("Docker container network inspection failed")
        try:
            networks = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            raise DeploymentError("Docker container network data is invalid") from None
        if not isinstance(networks, dict):
            raise DeploymentError("Docker container network data is invalid")
        candidates: list[tuple[str, str]] = []
        for item in networks.values():
            if not isinstance(item, dict):
                continue
            address, gateway = item.get("IPAddress"), item.get("Gateway")
            if not isinstance(address, str) or not isinstance(gateway, str) or not address or not gateway:
                continue
            try:
                ip = ipaddress.ip_address(address)
                gw = ipaddress.ip_address(gateway)
            except ValueError:
                continue
            if ip.version == gw.version == 4:
                candidates.append((str(gw), str(ip)))
        if len(candidates) != 1:
            raise DeploymentError("Docker container must have exactly one inspectable IPv4 bridge network")
        return candidates[0]

    def _expected_bridge(self, gateway: str, container_ip: str) -> tuple[str, list[str]]:
        binary = shutil.which("socat")
        if not binary:
            raise DeploymentError("socat is required to create the restricted teacher bridge")
        upstream = [
            "socat",
            f"TCP4-LISTEN:{self.config.bridge_listen_port},bind={gateway},fork,range={container_ip}/32",
            f"TCP4:127.0.0.1:{self.config.bridge_upstream_port}",
        ]
        resolved_binary = str(Path(binary).resolve())
        upstream[0] = resolved_binary
        base_url = f"http://{gateway}:{self.config.bridge_listen_port}/v1"
        return base_url, upstream

    def _state(self) -> dict[str, Any] | None:
        try:
            info = self.bridge_state_path.lstat()
        except FileNotFoundError:
            return None
        except OSError:
            raise DeploymentError("teacher bridge state cannot be inspected") from None
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
            raise DeploymentError("teacher bridge state has unsafe type, owner, or mode")
        try:
            value = json.loads(self.bridge_state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise DeploymentError("teacher bridge state is invalid") from None
        self._validate_bridge_state(value)
        return value

    def _validate_bridge_state(self, value: Mapping[str, Any]) -> None:
        expected_fields = {
            "schema_version", "deployment_path", "pid", "starttime", "argv", "gateway",
            "container_ip", "base_url", "created_at",
        }
        if (set(value) != expected_fields or value.get("schema_version") != "qwen-rl-teacher-bridge/v1"):
            raise DeploymentError("teacher bridge state has an unsupported schema")
        if value["deployment_path"] != str(self.config.path):
            raise DeploymentError("teacher bridge state belongs to a different deployment file")
        if (isinstance(value["pid"], bool) or not isinstance(value["pid"], int) or value["pid"] <= 1
                or isinstance(value["starttime"], bool) or not isinstance(value["starttime"], int)
                or value["starttime"] < 0):
            raise DeploymentError("teacher bridge process identity is invalid")
        try:
            gateway = str(ipaddress.IPv4Address(value["gateway"]))
            container_ip = str(ipaddress.IPv4Address(value["container_ip"]))
            expected_url, expected_argv = self._expected_bridge(gateway, container_ip)
        except (ipaddress.AddressValueError, TypeError):
            raise DeploymentError("teacher bridge network identity is invalid") from None
        if (value["gateway"] != gateway or value["container_ip"] != container_ip
                or value["argv"] != expected_argv or value["base_url"] != expected_url):
            raise DeploymentError("teacher bridge state does not match this deployment's socat command")
        if isinstance(value["created_at"], bool) or not isinstance(value["created_at"], (int, float)):
            raise DeploymentError("teacher bridge creation time is invalid")

    def _write_state(self, state: Mapping[str, Any]) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        encoded = (json.dumps(dict(state), sort_keys=True, indent=2) + "\n").encode("utf-8")
        fd, temporary = tempfile.mkstemp(prefix=".teacher-bridge-", dir=self.runtime_dir)
        temp_path = Path(temporary)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.bridge_state_path)
        except BaseException:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

    def _probe_bind_available(self, gateway: str) -> None:
        sock = self.socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((gateway, self.config.bridge_listen_port))
        except OSError:
            raise DeploymentError("Docker gateway teacher port is occupied by a listener not owned by this launcher") from None
        finally:
            sock.close()

    def _process_matches(self, state: Mapping[str, Any]) -> bool:
        return bridge_process_matches(state, proc_root=self.proc_root)

    def _terminate_bridge(self, state: Mapping[str, Any]) -> None:
        self._validate_bridge_state(state)
        if not self._process_matches(state):
            self._remove_bridge_state()
            return
        pid = state["pid"]
        try:
            self.signal_sender(pid, signal.SIGTERM)
        except ProcessLookupError:
            self._remove_bridge_state()
            return
        except OSError:
            raise DeploymentError("owned teacher bridge did not accept a graceful stop") from None
        deadline = self.monotonic() + self.config.bridge_stop_timeout
        while self.monotonic() < deadline:
            if not self._process_matches(state):
                self._remove_bridge_state()
                return
            self.sleep(min(0.1, max(0.0, deadline - self.monotonic())))
        raise DeploymentError("owned teacher bridge did not stop before the bounded timeout")

    def _remove_bridge_state(self) -> None:
        try:
            self.bridge_state_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            raise DeploymentError("teacher bridge state could not be removed") from None

    def _ensure_bridge(self, gateway: str, container_ip: str) -> bool:
        base_url, argv = self._expected_bridge(gateway, container_ip)
        state = self._state()
        if state is not None:
            if self._process_matches(state):
                if (state.get("gateway") != gateway or state.get("container_ip") != container_ip
                        or state.get("argv") != argv):
                    raise DeploymentError("owned teacher bridge belongs to a different Docker network")
                return False
            # The PID is stale or reused. Never signal it; clear only this stale identity record.
            self._remove_bridge_state()
        self._probe_bind_available(gateway)
        try:
            process = self.popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
        except OSError:
            raise DeploymentError("socat bridge process could not be started") from None
        deadline = self.monotonic() + self.config.bridge_start_timeout
        current: tuple[int, tuple[str, ...]] | None = None
        while self.monotonic() < deadline:
            current = _proc_identity(process.pid, self.proc_root)
            if current is not None and current[1] == tuple(argv):
                break
            if process.poll() is not None:
                break
            self.sleep(0.1)
        if current is None or current[1] != tuple(argv):
            current = _proc_identity(process.pid, self.proc_root)
        if current is None or current[1] != tuple(argv):
            if current is not None:
                provisional = {
                    "pid": process.pid, "starttime": current[0], "argv": list(argv),
                }
                if self._process_matches(provisional):
                    try:
                        self.signal_sender(process.pid, signal.SIGTERM)
                    except OSError:
                        pass
            raise DeploymentError("socat bridge did not acquire the expected process identity")
        new_state = {
            "schema_version": "qwen-rl-teacher-bridge/v1",
            "deployment_path": str(self.config.path),
            "pid": process.pid,
            "starttime": current[0],
            "argv": list(argv),
            "gateway": gateway,
            "container_ip": container_ip,
            "base_url": base_url,
            "created_at": time.time(),
        }
        try:
            self._write_state(new_state)
        except BaseException:
            # This exact process was just created by this invocation; identity-check before signaling.
            if self._process_matches(new_state):
                self._terminate_bridge(new_state)
            raise
        return True

    def _preflight(self, credentials: Mapping[str, str], gateway: str | None) -> subprocess.CompletedProcess[str]:
        injected = {"QWEN_RL_PROBE_TIMEOUT": str(self.config.probe_timeout)}
        if gateway is not None:
            injected.update({
                EXPECTED_BASE_URL_ENV: f"http://{gateway}:{self.config.bridge_listen_port}/v1",
                EXPECTED_MODEL_ENV: self.config.required_model,
                "QWEN_RL_MODELS_PATH": self.config.models_path,
            })
        argv = ["-c", PREFLIGHT, self.config.default_config]
        command = ["docker", "exec", "-u", self.config.container_uid, "-w", self.config.container_workdir]
        values = self._base_child_env({**credentials, **injected})
        env_names = [*self.config.runtime_env, *credentials, *injected]
        for name in env_names:
            command.extend(("-e", name))
        command.extend((self.config.container, self.config.container_python, "-B", *argv))
        self._active_redactions = tuple(credentials.values())
        try:
            result = self._capture(command, env=values, timeout=self.config.docker_timeout + self.config.probe_timeout + 5)
        finally:
            self._active_redactions = ()
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "container returned no diagnostic").strip()
            detail = " ".join(detail.split())[-1200:]
            raise DeploymentError(f"container preflight failed before GPU role startup: {detail}")
        return result

    def _manager_snapshot(
        self, credentials: Mapping[str, str], *,
        runtime_paths: tuple[str, str | None] | None = None,
    ) -> dict[str, Any] | None:
        runtime_config, environment_file = runtime_paths or self._runtime_paths()
        arguments = ["status", "--config", runtime_config]
        if environment_file is not None:
            arguments.extend(("--environment-file", environment_file))
        result = self._exec_container(
            arguments, credentials={"ARC_ENV_TOKEN": credentials["ARC_ENV_TOKEN"]},
            timeout=self.config.docker_timeout,
        )
        if result.returncode != 0:
            return None
        try:
            body = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return None
        processes = body.get("processes") if isinstance(body, dict) else None
        if (not isinstance(processes, list) or not isinstance(body.get("status"), str)
                or not body["status"]):
            return None
        for item in processes:
            if not isinstance(item, dict) or not isinstance(item.get("identity_matches"), bool):
                return None
        return body

    def _manager_status(
        self, credentials: Mapping[str, str], *,
        runtime_paths: tuple[str, str | None] | None = None,
    ) -> bool | None:
        runtime_config, environment_file = runtime_paths or self._runtime_paths()
        arguments = ["status", "--config", runtime_config]
        if environment_file is not None:
            arguments.extend(("--environment-file", environment_file))
        result = self._exec_container(
            arguments,
            credentials={"ARC_ENV_TOKEN": credentials["ARC_ENV_TOKEN"]},
            timeout=self.config.docker_timeout,
        )
        if result.returncode != 0:
            return None
        try:
            snapshot = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return None
        processes = snapshot.get("processes") if isinstance(snapshot, dict) else None
        if (not isinstance(processes, list) or not isinstance(snapshot.get("status"), str)
                or any(not isinstance(item, dict) or not isinstance(item.get("identity_matches"), bool)
                       for item in processes)):
            return None
        return not any(item["identity_matches"] for item in processes)

    def _cleanup_new_bridge_if_inactive(self, created: bool, credentials: Mapping[str, str]) -> None:
        if not created:
            return
        inactive = self._manager_status(credentials)
        if inactive is not True:
            return
        state = self._state()
        if state is not None and self._process_matches(state):
            self._terminate_bridge(state)
        elif state is not None:
            self._remove_bridge_state()

    @staticmethod
    def _format_result(result: subprocess.CompletedProcess[str]) -> None:
        if result.stdout:
            sys.stdout.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if result.stderr:
            sys.stderr.write(result.stderr)
            if not result.stderr.endswith("\n"):
                sys.stderr.write("\n")

    def _container_call(
        self,
        command: str,
        arguments: Sequence[str],
        credentials: Mapping[str, str],
        *,
        runtime_paths: tuple[str, str | None] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        runtime_config, environment_file = runtime_paths or self._runtime_paths()
        full = [command, "--config", runtime_config]
        if environment_file is not None:
            full.extend(("--environment-file", environment_file))
        full.extend(arguments)
        stream = command == "watch" and "--once" not in arguments
        timeout = self.config.start_timeout if command == "start" else self.config.command_timeout
        result = self._exec_container(full, credentials=credentials,
                                      timeout=timeout, stream=stream)
        self._format_result(result)
        return result

    def _help_call(self, command: str, arguments: Sequence[str]) -> int:
        full = [command, *arguments]
        result = self._exec_container(full, timeout=self.config.docker_timeout)
        self._format_result(result)
        return result.returncode

    def check(self) -> int:
        with self.lifecycle_lock():
            plan = self._worker_fleet().prepare()
            teacher_required = plan.runtime_config.bepa.enabled
            credentials = self._load_credentials(teacher=teacher_required, create_arc=True)
            created = False
            gateway = None
            if teacher_required:
                gateway, container_ip = self._inspect_network()
                created = self._ensure_bridge(gateway, container_ip)
            try:
                result = self._preflight(credentials, gateway)
            except (DeploymentError, KeyboardInterrupt):
                try:
                    self._cleanup_new_bridge_if_inactive(created, credentials)
                except DeploymentError:
                    pass
                raise
            try:
                payload = json.loads(result.stdout)
            except (json.JSONDecodeError, TypeError):
                raise DeploymentError("container preflight returned invalid result data") from None
            backend = plan.runtime_config.learner.update_backend
            detail = {}
            if backend == "central_gradient":
                from .local_gradients import LocalGradientHost

                detail = LocalGradientHost(self).check(plan)
            print(json.dumps({
                "status": "ready",
                "backend": backend,
                "bridge": "verified" if teacher_required else "not_required",
                "preflight": payload,
                "worker_image": "verified", "workers": len(plan.specs),
                **detail,
            }, indent=2))
            return 0

    def start(self, arguments: Sequence[str]) -> int:
        if self._configured_update_backend() == "central_gradient":
            with self.lifecycle_lock():
                fleet = self._worker_fleet()
                plan = fleet.prepare()
                if plan.runtime_config.learner.update_backend != "central_gradient":
                    raise DeploymentError("source and generated runtime backend differ")
                from .local_gradients import LocalGradientHost

                return LocalGradientHost(self).start(fleet, plan, arguments)
        arc_credentials = self._load_credentials(teacher=False)
        with self.lifecycle_lock():
            initial_paths = self._runtime_paths()
            initial_snapshot = self._manager_snapshot(
                arc_credentials, runtime_paths=initial_paths,
            )
            if initial_snapshot is None:
                raise DeploymentError("manager status could not be verified before start")
            initial_active = any(item["identity_matches"] for item in initial_snapshot["processes"])
            fresh = "--fresh" in arguments
            if initial_active and fresh:
                raise DeploymentError(
                    "stop the active runtime before starting a fresh lineage; Worker sidecars were left untouched"
                )
            if initial_active and initial_paths[1] is None:
                raise DeploymentError(
                    "an active runtime without a managed Worker service profile must be stopped before local startup"
                )
            if initial_active and not initial_snapshot.get("config_compatible", False):
                raise DeploymentError(
                    "the active runtime uses a different config; stop it before changing the local Worker profile"
                )

            fleet = self._worker_fleet()
            plan = fleet.prepare()
            runtime_paths = (plan.runtime_config_container_path,
                             plan.runtime_environment_container_path)
            snapshot = self._manager_snapshot(arc_credentials, runtime_paths=runtime_paths)
            if snapshot is None:
                raise DeploymentError("manager status could not be verified before start")
            active = any(item["identity_matches"] for item in snapshot["processes"])
            if active and fresh:
                raise DeploymentError(
                    "the runtime became active while preparing local Workers; no Worker container was started"
                )
            if active and not snapshot.get("config_compatible", False):
                raise DeploymentError(
                    "the active runtime uses a different config; stop it before changing the local Worker profile"
                )
            if active and snapshot.get("reuse_ready") is True:
                # The manager confirms healthy, same-config roles. Reuse needs only
                # the ARC token and does not probe the Teacher provider or bridge.
                started = fleet.start(plan)
                return self._start_manager_with_plan(
                    fleet, plan, started, arguments, arc_credentials, runtime_paths,
                )

            teacher_required = plan.runtime_config.bepa.enabled
            credentials = self._load_credentials(teacher=True) if teacher_required else arc_credentials
            gateway = None
            created = False
            if teacher_required:
                gateway, container_ip = self._inspect_network()
                created = self._ensure_bridge(gateway, container_ip)
            started: tuple[tuple[str, str], ...] = ()
            try:
                self._preflight(credentials, gateway)
                started = fleet.start(plan)
                return self._start_manager_with_plan(
                    fleet, plan, started, arguments, credentials, runtime_paths,
                    bridge_created=created,
                )
            except KeyboardInterrupt:
                try:
                    self._cleanup_new_bridge_if_inactive(created, credentials)
                except DeploymentError:
                    pass
                self._rollback_workers_if_inactive(fleet, plan, started, credentials, runtime_paths)
                raise
            except DeploymentError:
                try:
                    self._cleanup_new_bridge_if_inactive(created, credentials)
                except DeploymentError:
                    pass
                self._rollback_workers_if_inactive(fleet, plan, started, credentials, runtime_paths)
                raise

    def _rollback_workers_if_inactive(self, fleet, plan, started, credentials,
                                      runtime_paths: tuple[str, str | None] | None = None) -> None:
        if not started:
            return
        try:
            if self._manager_status(credentials, runtime_paths=runtime_paths) is True:
                fleet.rollback_started(plan, started)
        except DeploymentError:
            return

    def _start_manager_with_plan(self, fleet, plan, started, arguments, credentials,
                                 runtime_paths, *, bridge_created: bool = False) -> int:
        try:
            result = self._container_call("start", arguments, credentials,
                                          runtime_paths=runtime_paths)
            if result.returncode == 0:
                return 0
            self._rollback_workers_if_inactive(fleet, plan, started, credentials, runtime_paths)
            if bridge_created:
                try:
                    self._cleanup_new_bridge_if_inactive(bridge_created, credentials)
                except DeploymentError:
                    pass
            raise CommandExit(result.returncode)
        except (DeploymentError, KeyboardInterrupt):
            self._rollback_workers_if_inactive(fleet, plan, started, credentials, runtime_paths)
            if bridge_created:
                try:
                    self._cleanup_new_bridge_if_inactive(bridge_created, credentials)
                except DeploymentError:
                    pass
            raise

    def stop(self, arguments: Sequence[str]) -> int:
        if self._configured_update_backend() == "central_gradient":
            with self.lifecycle_lock():
                from .local_gradients import LocalGradientHost

                return LocalGradientHost(self).stop(self._worker_fleet(), arguments)
        self._assert_default_gradient_inactive()
        credentials = self._load_credentials(teacher=False)
        with self.lifecycle_lock():
            runtime_paths = self._runtime_paths()
            result = self._container_call("stop", arguments, credentials,
                                          runtime_paths=runtime_paths)
            if result.returncode != 0:
                raise CommandExit(result.returncode)
            try:
                payload = json.loads(result.stdout)
            except (json.JSONDecodeError, TypeError):
                raise DeploymentError("manager stop returned invalid result data; teacher bridge remains active") from None
            if not isinstance(payload, dict) or payload.get("status") != "stopped":
                raise DeploymentError("manager stop is incomplete; teacher bridge remains active")
            self._worker_fleet().stop_owned()
            state = self._state()
            if state is not None:
                self._terminate_bridge(state)
            return 0

    def forward(self, command: str, arguments: Sequence[str]) -> int:
        if command == "selftest" and self._configured_update_backend() == "central_gradient":
            raise DeploymentError(f"{command} is not supported by the central-gradient local profile")
        if command == "status" and self._configured_update_backend() == "central_gradient":
            from .local_gradients import LocalGradientHost

            status = LocalGradientHost(self).status(self._worker_fleet())
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0 if status.get("status") != "degraded" else 2
        credentials = self._load_credentials(teacher=False)
        runtime_paths = self._runtime_paths()
        if command == "bepa":
            if not arguments or arguments[0] not in {"bootstrap", "status", "jobs", "budget"}:
                raise DeploymentError("bepa requires bootstrap, status, jobs, or budget")
            runtime_config, environment_file = runtime_paths
            full = ["bepa", arguments[0], "--config", runtime_config]
            if environment_file is not None:
                full.extend(("--environment-file", environment_file))
            full.extend(arguments[1:])
            stream = False
            result = self._exec_container(full, credentials=credentials,
                                          timeout=self.config.command_timeout, stream=stream)
            self._format_result(result)
            if result.returncode:
                raise CommandExit(result.returncode)
            return 0
        result = self._container_call(command, arguments, credentials,
                                      runtime_paths=runtime_paths)
        if result.returncode:
            raise CommandExit(result.returncode)
        return 0


def _extract_host_option(argv: Sequence[str]) -> tuple[Path, list[str]]:
    values = list(argv)
    deployment = None
    if values and values[0] == "--deployment":
        if len(values) < 2:
            raise DeploymentError("--deployment requires a path before the command")
        deployment = Path(values[1]).expanduser()
        values = values[2:]
    elif values and values[0].startswith("--deployment="):
        deployment = Path(values[0].split("=", 1)[1]).expanduser()
        values = values[1:]
    if not values:
        raise DeploymentError("a qwen-rl command is required")
    if deployment is None:
        raise DeploymentError("--deployment PATH is required before COMMAND; this distribution has no default deployment profile")
    if any(arg == "--deployment" or arg.startswith("--deployment=") for arg in values):
        raise DeploymentError("--deployment is a host option and must appear before the command")
    if any(arg == "--config" or arg.startswith("--config=") for arg in values):
        raise DeploymentError("--config is fixed by the deployment file and cannot be forwarded")
    return deployment, values


def _usage() -> str:
    return (
        "Usage: qwen-rl local --deployment PATH COMMAND [ARGS...]\n"
        "Commands: start [--fresh], check, status, watch, run, stop, selftest, bepa, evaluate [--status]\n"
        "Supply a prepared deployment profile; no default model/deployment profile is bundled.\n"
        "check verifies the Worker image/profile, provider route and deployment mounts without starting roles.\n"
        "start reuses healthy same-config roles; default central-gradient rejects --fresh (Trainer profiles opt in).\n"
        "The deployment flag must appear before COMMAND. The runtime config is fixed by deployment JSON.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values in (["--help"], ["-h"]):
        print(_usage(), end="")
        return 0
    try:
        deployment_path, command_argv = _extract_host_option(values)
        command, *arguments = command_argv
        if command in {"--help", "-h"}:
            print(_usage(), end="")
            return 0
        if command not in COMMANDS:
            raise DeploymentError("unknown qwen-rl command")
        config = load_deployment(deployment_path)
        deployment = LocalDeployment(ROOT, config)
        if any(arg in {"--help", "-h"} for arg in arguments):
            if command == "check":
                print("check validates the Worker image/profile, credentials and teacher route without starting roles")
                return 0
            return deployment._help_call(command, arguments)
        if command == "check":
            if arguments:
                raise DeploymentError("check does not accept forwarded arguments")
            return deployment.check()
        if command == "start":
            return deployment.start(arguments)
        if command == "stop":
            return deployment.stop(arguments)
        return deployment.forward(command, arguments)
    except CommandExit as exc:
        return exc.status
    except KeyboardInterrupt:
        return 130
    except DeploymentError as exc:
        print(f"qwen-rl: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError):
        print("qwen-rl: local deployment failed", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - keep unexpected host failures in the user-safe CLI boundary.
        print(f"qwen-rl: local deployment failed ({type(exc).__name__})", file=sys.stderr)
        return 2
