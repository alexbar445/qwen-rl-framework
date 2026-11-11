"""Parent-process controller for a persistent native TorchAO learner service."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Callable

from .topology import ParallelTopology
from .transport import receive_message, send_message


class ParallelBackendError(RuntimeError):
    """A persistent parallel Learner world failed its controller contract."""


class ParallelBackendClient:
    """Own one compute process group and a private authenticated UDS control link.

    Public calls are synchronous and serialized. `cached_status` is memory-only:
    reading it never sends a command, waits for the service, or touches CUDA.
    """

    _STARTUP_TIMEOUT_SECONDS = 900.0

    def __init__(self, config: dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError("ParallelBackendClient config must be a dict")
        self.config = copy.deepcopy(config)
        self.topology = ParallelTopology.from_config(config)
        self.compute_timeout = self.topology.compute_timeout
        self.compute_python = config.get("compute_python")
        self.compute_pythonpath = config.get("compute_pythonpath")
        if not isinstance(self.compute_python, str) or not self.compute_python:
            raise ValueError("compute_python must be explicitly configured")
        if not isinstance(self.compute_pythonpath, str) or not self.compute_pythonpath:
            raise ValueError("compute_pythonpath must be explicitly configured")
        self._mutex = threading.RLock()
        self._runtime_dir: Path | None = None
        self._log_dir: Path | None = None
        self._log_dir_persistent = False
        self._listener: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._process: subprocess.Popen | None = None
        self._log_stream = None
        self._token: str | None = None
        self._started = False
        self._closed = False
        self._request_sequence = 0
        self._evidence: dict[str, Any] | None = None
        self._cached_status: dict[str, Any] = {
            "state": "created",
            "mode": self.topology.mode,
            "world_size": self.topology.world_size,
            "devices": list(self.topology.devices),
            "phase": "not_started",
        }

    @property
    def cached_status(self) -> dict[str, Any]:
        """Return the latest locally cached progress without RPC or GPU access."""
        process = self._process
        if (process is not None and not self._closed
                and self._cached_status.get("state") not in {"failed", "closed"}):
            exit_code = process.poll()
            if exit_code is not None:
                self._cached_status = {
                    **self._cached_status,
                    "state": "failed",
                    "phase": "process_exit",
                    "error": f"owned parallel rank launcher exited with code {exit_code}",
                    "exit_code": exit_code,
                }
        return copy.deepcopy(self._cached_status)

    @property
    def evidence(self) -> dict[str, Any]:
        if self._evidence is None:
            return {}
        return copy.deepcopy(self._evidence)

    @property
    def log_path(self) -> str | None:
        if self._log_dir is None:
            return None
        return str(self._log_dir / "rank-service.log")

    def start(self) -> dict[str, Any]:
        with self._mutex:
            if self._started:
                return self.evidence
            if self._closed:
                raise ParallelBackendError("parallel backend client has been shut down")
            self._launch()
            deadline = time.monotonic() + self._STARTUP_TIMEOUT_SECONDS
            try:
                self._connection = self._accept_authenticated(deadline)
                event = self._receive_until(deadline)
                while event.get("kind") == "status":
                    update = event.get("payload")
                    if isinstance(update, dict):
                        self._cached_status = {**self._cached_status, **update}
                    event = self._receive_until(deadline)
                if event.get("kind") != "ready":
                    raise ParallelBackendError(f"rank service did not become ready: {event!r}")
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    raise ParallelBackendError("rank service returned malformed startup evidence")
                self._evidence = copy.deepcopy(payload)
                if self._process is None:
                    raise ParallelBackendError("parallel rank process was not recorded")
                self._evidence.update({
                    "controller_pid": os.getpid(),
                    "owned_process_pid": self._process.pid,
                    "owned_process_group": os.getpgid(self._process.pid),
                })
                self._started = True
                self._cached_status = {
                    "state": "ready",
                    "phase": "idle",
                    "mode": self.topology.mode,
                    "world_size": self.topology.world_size,
                    "devices": list(self.topology.devices),
                    "evidence": copy.deepcopy(self._evidence),
                }
                return self.evidence
            except BaseException as exc:
                log_tail = self._read_log_tail()
                self._stop_process_group()
                self._close_sockets()
                if self._runtime_dir is not None:
                    shutil.rmtree(self._runtime_dir, ignore_errors=True)
                if log_tail:
                    raise ParallelBackendError(f"{exc}\nrank-service log tail:\n{log_tail}") from exc
                raise

    def logprobs(
        self,
        input_ids: list[int],
        policy_mask: list[int],
        attention_mask: list[int] | None = None,
    ) -> list[float]:
        result = self._command("logprobs", {
            "input_ids": input_ids,
            "policy_mask": policy_mask,
            "attention_mask": attention_mask,
        })
        return list(result["logprobs"])

    def probe(self, lengths: list[int]) -> dict[str, Any]:
        """Run the loaded model's fused scoring path at the requested lengths."""
        result = self._command("probe", {"lengths": lengths})
        return result

    def execution_diagnostics(self, *, input_ids=None, policy_mask=None) -> dict[str, Any]:
        """Explicitly hash frozen T2 TP execution state on every rank.

        This command is never part of startup or snapshot collection. It is
        serialized with compute commands and must run while phase-shared storage
        is awake because parked CUDA pointers may not be readable.
        """
        if self.topology.mode != "tp" or self.config.get("model_backend") != "ternary":
            raise ParallelBackendError("execution diagnostics are available only for T2 TP")
        with self._mutex:
            status = self._cached_status
            residency = status.get("residency")
            if status.get("state") == "parked" or (
                isinstance(residency, dict) and residency.get("state") in {"parked", "asleep"}
            ):
                raise ParallelBackendError("execution diagnostics require awake T2 TP storage")
            payload = {}
            if input_ids is not None or policy_mask is not None:
                if input_ids is None or policy_mask is None:
                    raise ValueError("forward diagnostics require both input_ids and policy_mask")
                if not 2 <= len(input_ids) <= 256 or len(policy_mask) != len(input_ids):
                    raise ValueError("forward diagnostics require aligned inputs of 2..256 tokens")
                payload = {"input_ids": input_ids, "policy_mask": policy_mask}
            return self._command("execution_diagnostics", payload)

    def update(
        self,
        rows: list[dict[str, Any]],
        *,
        before_optimizer_step: Callable[[], None] | None = None,
        recovery_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if before_optimizer_step is not None and not callable(before_optimizer_step):
            raise TypeError("before_optimizer_step must be callable")
        self._cached_status = {
            **self._cached_status,
            "state": "updating",
            "phase": "rollout_validation",
            "started_at": time.time(),
        }
        result = self._command(
            "update",
            {"rows": rows, "recovery_context": recovery_context},
            before_optimizer_step=before_optimizer_step,
        )
        self._cached_status = {
            **self._cached_status,
            "state": "ready",
            "phase": "idle",
            "finished_at": time.time(),
            "last_update": copy.deepcopy(result.get("status", {})),
            "telemetry": copy.deepcopy(result.get("telemetry", {})),
            "quality": copy.deepcopy(result.get("quality", {})),
        }
        return result

    def export_adapter(self, path: str | Path) -> dict[str, Any]:
        result = self._command("export_adapter", {"path": str(Path(path).expanduser().resolve())})
        return result

    def snapshot(self) -> dict[str, Any]:
        result = self._command("snapshot", {})
        if not isinstance(result, dict):
            raise ParallelBackendError("rank service returned a malformed state snapshot")
        return result

    def restore(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("parallel restore payload must be a dict")
        return self._command("restore", {"snapshot": payload})

    def park(self) -> dict[str, Any]:
        result = self._command("park", {})
        self._cached_status = {**self._cached_status, "state": "parked", "residency": result}
        return result

    def wake(self) -> dict[str, Any]:
        result = self._command("wake", {})
        self._cached_status = {**self._cached_status, "state": "ready", "residency": result}
        return result

    def shutdown(self) -> None:
        acquired = self._mutex.acquire(blocking=False)
        if not acquired:
            # A concurrent update may be blocked in the private IPC callback.
            # Terminate only this client's dedicated process group to unblock it.
            self._stop_process_group()
            self._close_sockets()
            if self._runtime_dir is not None:
                shutil.rmtree(self._runtime_dir, ignore_errors=True)
            if self._log_dir is not None and not self._log_dir_persistent:
                shutil.rmtree(self._log_dir, ignore_errors=True)
            self._closed = True
            self._started = False
            self._cached_status = {**self._cached_status, "state": "closed", "phase": "closed"}
            return
        try:
            if self._closed:
                return
            try:
                if (self._started and self._connection is not None and self._process is not None
                        and self._cached_status.get("state") != "failed"):
                    self._command("shutdown", {})
            finally:
                self._stop_process_group()
                self._close_sockets()
                if self._runtime_dir is not None:
                    shutil.rmtree(self._runtime_dir, ignore_errors=True)
                if self._log_dir is not None and not self._log_dir_persistent:
                    shutil.rmtree(self._log_dir, ignore_errors=True)
                self._closed = True
                self._started = False
                self._cached_status = {**self._cached_status, "state": "closed", "phase": "closed"}
        finally:
            self._mutex.release()

    def _launch(self) -> None:
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="qwen-rl-parallel-"))
        os.chmod(self._runtime_dir, 0o700)
        configured_log_root = self.config.get("compute_runtime_dir")
        self._log_dir_persistent = configured_log_root is not None
        if configured_log_root is None:
            self._log_dir = Path(tempfile.mkdtemp(prefix="qwen-rl-parallel-logs-"))
        else:
            if not isinstance(configured_log_root, (str, Path)) or not str(configured_log_root):
                raise ValueError("compute_runtime_dir must be a non-empty path when configured")
            log_root = Path(configured_log_root).expanduser().resolve()
            log_root.mkdir(parents=True, exist_ok=True)
            self._log_dir = Path(tempfile.mkdtemp(prefix="rank-service-", dir=str(log_root)))
        os.chmod(self._log_dir, 0o700)
        socket_path = self._runtime_dir / "control.sock"
        if len(os.fsencode(socket_path)) >= 100:
            raise ParallelBackendError("runtime socket path is too long for AF_UNIX")
        self._token = secrets.token_hex(32)
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        self._listener.listen(1)
        self._listener.settimeout(0.25)

        runtime = {
            "config": self.config,
            "topology": self.topology.signature(),
            "socket_path": str(socket_path),
            "auth_token": self._token,
            "runtime_dir": str(self._runtime_dir),
        }
        runtime_path = self._runtime_dir / "runtime.json"
        runtime_path.write_text(json.dumps(runtime, sort_keys=True), encoding="utf-8")
        os.chmod(runtime_path, 0o600)

        env = self._compute_environment()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in self.topology.devices)
        env["PYTHONUNBUFFERED"] = "1"
        env["QWEN_RL_PARALLEL_PARENT_PID"] = str(os.getpid())
        env["QWEN_RL_PARALLEL_NPROC"] = f"--nproc_per_node={self.topology.world_size}"
        env["QWEN_RL_PARALLEL_CONFIG"] = str(runtime_path)
        service_log = self._log_dir / "rank-service.log"
        self._log_stream = service_log.open("ab", buffering=0)
        command = [self.compute_python, "-m", "qwen_rl.learner.parallel.launcher"]
        self._process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[3]),
            stdin=subprocess.DEVNULL,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self._cached_status = {
            "state": "starting",
            "phase": "process_group_start",
            "mode": self.topology.mode,
            "world_size": self.topology.world_size,
            "devices": list(self.topology.devices),
            "log_path": str(service_log),
        }

    def _compute_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        for name in tuple(env):
            upper = name.upper()
            provider_or_arc = upper.startswith((
                "ARC_", "OPENAI_", "ANTHROPIC_", "GOOGLE_", "GEMINI_", "COHERE_",
                "MISTRAL_", "VLLM_", "PROVIDER_", "WANDB_", "HF_", "HUGGINGFACE",
            ))
            if upper in {"API_KEY", "API_TOKEN", "AUTH_TOKEN", "ACCESS_TOKEN"} or upper.endswith((
                "_API_KEY", "_API_TOKEN", "_AUTH_TOKEN", "_ACCESS_TOKEN", "_TOKEN", "_SECRET", "_KEY"
            )) or (
                provider_or_arc and ("TOKEN" in upper or "KEY" in upper or "SECRET" in upper)
            ):
                env.pop(name, None)
        project_root = str(Path(__file__).resolve().parents[3])
        python_paths = [value for value in self.compute_pythonpath.split(os.pathsep) if value]
        if project_root not in python_paths:
            python_paths.append(project_root)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        for name in ("PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL", "ACCELERATE_USE_CPU"):
            env.pop(name, None)
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                     "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
            env.pop(name, None)
        for name in tuple(env):
            if name.startswith("TORCHELASTIC_"):
                env.pop(name, None)
        if self.config.get("gpu_sharing") == "phase_shared":
            # CuMem provides VMM residency itself. Never combine it with
            # PyTorch expandable segments. T2 also bounds allocator splitting;
            # this environment is for the Learner ranks, never Workers.
            from ..cuda_allocator import configure_phase_shared_allocator

            configure_phase_shared_allocator(
                env,
                model_backend=self.config.get("model_backend"),
                gpu_sharing=self.config.get("gpu_sharing"),
            )
            # A recovery root survives managed runtime restarts. Native Torch /
            # Triton cache keys still select compatible compiler/GPU/shape entries.
            # This cache is neither a model checkpoint nor an autotune warmup receipt.
            recovery_root = self.config.get("recovery_dir")
            cache_parent = Path(recovery_root or self.config.get("compute_runtime_dir") or self._runtime_dir)
            cache_root = cache_parent / "compiler-cache"
            cache_parent.mkdir(parents=True, exist_ok=True)
            if cache_parent.is_symlink():
                raise ParallelBackendError("compiler cache parent cannot be a symlink")
            for directory in (cache_root, cache_root / "inductor", cache_root / "triton"):
                if directory.is_symlink():
                    raise ParallelBackendError("compiler cache cannot be a symlink")
                directory.mkdir(mode=0o700, exist_ok=True)
                info = directory.stat()
                if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                    raise ParallelBackendError("compiler cache must be a private directory owned by the compute user")
            env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "inductor")
            env["TRITON_CACHE_DIR"] = str(cache_root / "triton")
        return env

    def _accept_authenticated(self, deadline: float) -> socket.socket:
        assert self._listener is not None and self._token is not None
        while time.monotonic() < deadline:
            self._raise_if_process_exited("rank service exited before connecting")
            try:
                connection, _ = self._listener.accept()
                connection.settimeout(self.compute_timeout)
                hello = receive_message(connection)
                if not isinstance(hello, dict) or hello.get("kind") != "auth" or not secrets.compare_digest(
                    str(hello.get("token", "")), self._token
                ):
                    connection.close()
                    raise ParallelBackendError("rank service failed private-socket authentication")
                send_message(connection, {"kind": "auth_ack"})
                self._listener.close()
                self._listener = None
                return connection
            except socket.timeout:
                continue
        raise TimeoutError("timed out waiting for rank service to connect")

    def _command(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        before_optimizer_step: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        with self._mutex:
            if not self._started or self._connection is None:
                if operation == "start":
                    raise ParallelBackendError("invalid recursive start request")
                self.start()
            assert self._connection is not None
            self._request_sequence += 1
            request_id = f"{self._request_sequence}-{uuid.uuid4().hex}"
            try:
                send_message(self._connection, {
                    "kind": "command",
                    "request_id": request_id,
                    "operation": operation,
                    "payload": payload,
                })
            except BaseException as exc:
                self._cached_status = {
                    **self._cached_status,
                    "state": "failed",
                    "phase": "rpc_send",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                raise
            deadline = time.monotonic() + self.compute_timeout
            while time.monotonic() < deadline:
                try:
                    event = self._receive_until(deadline)
                except BaseException as exc:
                    self._cached_status = {
                        **self._cached_status,
                        "state": "failed",
                        "phase": "rpc_wait",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if isinstance(exc, TimeoutError):
                        details = json.dumps(self._cached_status, sort_keys=True, default=str)
                        raise TimeoutError(
                            f"{exc}; status={details}; rank-service log={self.log_path}"
                        ) from exc
                    raise
                kind = event.get("kind")
                if kind == "status":
                    update = event.get("payload")
                    if isinstance(update, dict):
                        self._cached_status = {**self._cached_status, **update}
                    continue
                if kind == "before_optimizer_step":
                    callback_error = None
                    if before_optimizer_step is not None:
                        try:
                            before_optimizer_step()
                        except BaseException as exc:
                            callback_error = {
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                    send_message(self._connection, {
                        "kind": "callback_ack",
                        "request_id": request_id,
                        "error": callback_error,
                    })
                    continue
                if kind != "response" or event.get("request_id") != request_id:
                    raise ParallelBackendError(f"unexpected rank-service message: {event!r}")
                if event.get("ok") is not True:
                    self._cached_status = {
                        **self._cached_status,
                        "state": "failed",
                        "phase": event.get("phase", "unknown"),
                        "error": event.get("error"),
                    }
                    raise ParallelBackendError(
                        f"parallel {operation} failed at {event.get('phase', 'unknown')}: "
                        f"{event.get('error', 'unknown rank-service error')}"
                    )
                result = event.get("payload")
                if not isinstance(result, dict):
                    raise ParallelBackendError(f"parallel {operation} returned a non-object result")
                return result
            raise TimeoutError(
                f"parallel {operation} exceeded {self.compute_timeout:g}s; "
                f"status={json.dumps(self._cached_status, sort_keys=True, default=str)}; "
                f"rank-service log={self.log_path}"
            )

    def _receive_until(self, deadline: float) -> dict[str, Any]:
        connection = self._connection
        if connection is None:
            raise ParallelBackendError("rank service control socket is not connected")
        while True:
            self._raise_if_process_exited("rank service exited during an operation")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for rank service")
            readable, _, _ = select.select([connection], [], [], min(remaining, 0.25))
            if not readable:
                continue
            try:
                connection.settimeout(max(0.1, remaining))
                message = receive_message(connection)
            except (EOFError, OSError, socket.timeout) as exc:
                self._raise_if_process_exited("rank service control socket closed")
                raise ParallelBackendError(f"rank service control socket failed: {exc}") from exc
            if not isinstance(message, dict):
                raise ParallelBackendError("rank service sent a malformed control message")
            return message

    def _raise_if_process_exited(self, message: str) -> None:
        process = self._process
        if process is not None and process.poll() is not None:
            log_text = self._read_log_tail()
            raise ParallelBackendError(f"{message} (exit={process.returncode}; log={log_text})")

    def _read_log_tail(self) -> str:
        if self.log_path and Path(self.log_path).is_file():
            try:
                return Path(self.log_path).read_text(encoding="utf-8", errors="replace")[-8000:]
            except OSError:
                pass
        return ""

    def _stop_process_group(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        self._process = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def _close_sockets(self) -> None:
        for connection in (self._connection, self._listener):
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
        self._connection = None
        self._listener = None
