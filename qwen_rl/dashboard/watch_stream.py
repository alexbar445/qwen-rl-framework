"""Bounded, restartable reader for the local runtime watch JSONL stream."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
import time
from typing import Any, Callable

MAX_WATCH_BODY_BYTES = 1 * 1024 * 1024
DEFAULT_STALE_AFTER_SECONDS = 5.0
DEFAULT_RESTART_BACKOFF_SECONDS = 5.0
_DEFAULT_TERMINATE_TIMEOUT_SECONDS = 1.0
_DEFAULT_KILL_TIMEOUT_SECONDS = 1.0
_DEFAULT_READER_JOIN_TIMEOUT_SECONDS = 1.0
_STDERR_CHUNK_BYTES = 4096
_OVERSIZED_LINE = object()


def _failure(error: str) -> dict[str, Any]:
    return {"ok": False, "snapshot": None, "error": error}


def _readline_bounded(stream: Any, max_bytes: int) -> bytes | object | None:
    """Read one line using bounded memory, draining oversize lines before continuing."""
    buffered = bytearray()
    oversized = False

    while True:
        size = (_STDERR_CHUNK_BYTES if oversized
                else min(_STDERR_CHUNK_BYTES, max(1, max_bytes - len(buffered) + 1)))
        try:
            chunk = stream.readline(size)
        except (OSError, ValueError):
            return None
        if not chunk:
            if oversized:
                return _OVERSIZED_LINE
            return bytes(buffered) if buffered else None

        if not oversized:
            if len(buffered) + len(chunk) > max_bytes:
                oversized = True
                buffered.clear()
            else:
                buffered.extend(chunk)

        if chunk.endswith(b"\n"):
            return _OVERSIZED_LINE if oversized else bytes(buffered)


class LocalWatchStream:
    """Own one local watch process and cache only its most recent validated frame.

    The command is an argv vector, never a shell string. The caller owns this object and must
    call close(); cleanup signals only the process group created for this command.
    """

    def __init__(
        self,
        command: list[str],
        env: dict[str, str],
        expected_run_id: str | None,
        on_frame: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        *,
        max_line_bytes: int = MAX_WATCH_BODY_BYTES,
        stale_after: float = DEFAULT_STALE_AFTER_SECONDS,
        restart_backoff: float = DEFAULT_RESTART_BACKOFF_SECONDS,
        terminate_timeout: float = _DEFAULT_TERMINATE_TIMEOUT_SECONDS,
        kill_timeout: float = _DEFAULT_KILL_TIMEOUT_SECONDS,
        reader_join_timeout: float = _DEFAULT_READER_JOIN_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(command, list) or not command or any(
            not isinstance(item, str) or not item for item in command
        ):
            raise ValueError("command must be a non-empty argv list")
        if not isinstance(env, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in env.items()
        ):
            raise ValueError("env must map strings to strings")
        if expected_run_id is not None and (
            not isinstance(expected_run_id, str) or not expected_run_id
        ):
            raise ValueError("expected_run_id must be a non-empty string or None")
        if on_frame is not None and not callable(on_frame):
            raise ValueError("on_frame must be callable or None")
        if type(max_line_bytes) is not int or max_line_bytes < 1:
            raise ValueError("max_line_bytes must be a positive integer")
        for name, value in (
            ("stale_after", stale_after),
            ("restart_backoff", restart_backoff),
            ("terminate_timeout", terminate_timeout),
            ("kill_timeout", kill_timeout),
            ("reader_join_timeout", reader_join_timeout),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        self.command = list(command)
        self.env = dict(env)
        self.expected_run_id = expected_run_id
        self.on_frame = on_frame
        self.max_line_bytes = max_line_bytes
        self.stale_after = float(stale_after)
        self.restart_backoff = float(restart_backoff)
        self.terminate_timeout = float(terminate_timeout)
        self.kill_timeout = float(kill_timeout)
        self.reader_join_timeout = float(reader_join_timeout)

        self._condition = threading.Condition(threading.Lock())
        self._restart_lock = threading.Lock()
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._ended_at: float | None = None
        self._last_error: str | None = None
        self._latest_result: dict[str, Any] | None = None
        self._latest_at: float | None = None

        with self._restart_lock:
            with self._condition:
                self._spawn_locked()

    @property
    def pid(self) -> int | None:
        with self._condition:
            return self._process.pid if self._process is not None else None

    def get(self, timeout: float = 0.0) -> dict[str, Any]:
        """Return the latest processed frame, waiting at most timeout for the first frame."""
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        deadline = time.monotonic() + float(timeout)

        while True:
            self._restart_if_due()
            with self._condition:
                if self._closed:
                    return _failure("watch_stream_closed")

                now = time.monotonic()
                if self._latest_result is not None and self._latest_at is not None:
                    if now - self._latest_at <= self.stale_after:
                        return dict(self._latest_result)
                    if self._ended_at is None:
                        self._ended_at = now
                        self._last_error = "stale_snapshot"
                    return _failure("stale_snapshot")

                remaining = deadline - now
                if remaining <= 0:
                    return _failure(self._last_error or "initial_snapshot_timeout")

                wait_for = remaining
                if self._ended_at is not None:
                    until_retry = self.restart_backoff - (now - self._ended_at)
                    if until_retry > 0:
                        wait_for = min(wait_for, until_retry)
                self._condition.wait(wait_for)

    def close(self) -> None:
        """Stop this command's process group and join its two bounded readers."""
        with self._restart_lock:
            with self._condition:
                if self._closed:
                    return
                self._closed = True
                process = self._process
                stdout_thread = self._stdout_thread
                stderr_thread = self._stderr_thread
                self._process = None
                self._condition.notify_all()

            if process is not None:
                self._stop_process_group(process)

            deadline = time.monotonic() + self.reader_join_timeout
            for thread in (stdout_thread, stderr_thread):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))

            if process is not None:
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except OSError:
                            pass

    def _spawn_locked(self) -> None:
        """Start a child while the condition lock is held."""
        if self._closed:
            return
        try:
            process = subprocess.Popen(
                self.command,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
            )
        except Exception as exc:
            self._process = None
            self._ended_at = time.monotonic()
            self._last_error = f"watch_spawn_error:{type(exc).__name__}"
            self._condition.notify_all()
            return

        self._process = process
        self._latest_result = None
        self._latest_at = None
        self._ended_at = None
        self._last_error = None
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(process, process.stdout),
            name=f"local-watch-stdout-{process.pid}",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process, process.stderr),
            name=f"local-watch-stderr-{process.pid}",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._condition.notify_all()

    def _restart_if_due(self) -> None:
        with self._restart_lock:
            with self._condition:
                if self._closed or self._ended_at is None:
                    return
                if time.monotonic() - self._ended_at < self.restart_backoff:
                    return
                process = self._process
                stdout_thread = self._stdout_thread
                stderr_thread = self._stderr_thread
                self._process = None

            if process is not None:
                self._stop_process_group(process)

            deadline = time.monotonic() + self.reader_join_timeout
            for thread in (stdout_thread, stderr_thread):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))

            with self._condition:
                if not self._closed:
                    self._spawn_locked()

    def _read_stdout(self, process: subprocess.Popen[bytes], stream: Any) -> None:
        try:
            while True:
                line = _readline_bounded(stream, self.max_line_bytes)
                if line is None:
                    break
                if line is _OVERSIZED_LINE:
                    self._publish_failure(process, "watch_frame_too_large")
                    continue
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                    self._publish_failure(process, "watch_frame_invalid_json")
                    continue
                if not isinstance(frame, dict):
                    self._publish_failure(process, "watch_frame_invalid_shape")
                    continue

                error = self._validate_frame(frame)
                if error is not None:
                    self._publish_failure(process, error)
                    continue

                received_at = time.monotonic()
                try:
                    result = (
                        self.on_frame(frame)
                        if self.on_frame is not None
                        else {"ok": True, "snapshot": frame, "error": None}
                    )
                except Exception as exc:
                    self._publish_failure(
                        process, f"watch_frame_processor_error:{type(exc).__name__}"
                    )
                    continue

                if (
                    not isinstance(result, dict)
                    or type(result.get("ok")) is not bool
                    or "snapshot" not in result
                    or ("error" in result and result["error"] is not None
                        and not isinstance(result["error"], str))
                    or "error" not in result
                ):
                    self._publish_failure(process, "watch_frame_processor_result_invalid")
                    continue

                with self._condition:
                    if self._closed or process is not self._process:
                        continue
                    self._latest_result = dict(result)
                    self._latest_at = received_at
                    self._last_error = result.get("error")
                    self._condition.notify_all()
        except Exception as exc:
            self._publish_failure(process, f"watch_stream_reader_error:{type(exc).__name__}")
        finally:
            return_code = process.poll()
            exit_error = (
                f"watch_stream_exit:{return_code}"
                if return_code not in (None, 0)
                else "watch_stream_eof"
            )
            with self._condition:
                if not self._closed and process is self._process:
                    self._ended_at = time.monotonic()
                    self._latest_result = _failure(exit_error)
                    self._latest_at = time.monotonic()
                    self._last_error = exit_error
                    self._condition.notify_all()

    def _validate_frame(self, frame: dict[str, Any]) -> str | None:
        if frame.get("schema") != "qwen-rl-watch-v1":
            return "watch_frame_schema_invalid"
        sampled_at = frame.get("sampled_at")
        if (
            not isinstance(sampled_at, (int, float))
            or isinstance(sampled_at, bool)
            or not math.isfinite(sampled_at)
        ):
            return "watch_frame_sampled_at_invalid"
        if time.time() - sampled_at > self.stale_after:
            return "watch_frame_sampled_at_stale"
        if sampled_at - time.time() > 30:
            return "watch_frame_clock_ahead"
        lifecycle = frame.get("lifecycle")
        if not isinstance(lifecycle, dict):
            return "watch_frame_runtime_identity_invalid"
        run_id = lifecycle.get("run_id")
        if run_id is not None and not isinstance(run_id, str):
            return "watch_frame_runtime_identity_invalid"
        if self.expected_run_id is not None and run_id != self.expected_run_id:
            return "watch_frame_runtime_identity_mismatch"
        return None

    def _publish_failure(self, process: subprocess.Popen[bytes], error: str) -> None:
        with self._condition:
            if self._closed or process is not self._process:
                return
            self._latest_result = _failure(error)
            self._latest_at = time.monotonic()
            self._last_error = error
            self._condition.notify_all()

    @staticmethod
    def _drain_stderr(process: subprocess.Popen[bytes], stream: Any) -> None:
        """Drain stderr without retaining or exposing any bytes from it."""
        del process
        try:
            while stream.read(_STDERR_CHUNK_BYTES):
                pass
        except (OSError, ValueError):
            return

    def _stop_process_group(self, process: subprocess.Popen[bytes]) -> None:
        """Signal only the new session created by this object's Popen."""
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if "--exit-on-stdin-eof" in self.command:
            try:
                process.wait(timeout=self.terminate_timeout)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                process.terminate()
            except OSError:
                pass

        try:
            process.wait(timeout=self.terminate_timeout)
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=self.kill_timeout)
        except subprocess.TimeoutExpired:
            pass
