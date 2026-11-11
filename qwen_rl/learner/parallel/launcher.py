"""Single-thread parent-death guard before execing the compute service."""
from __future__ import annotations

import ctypes
import json
import os
import signal
import sys


def main() -> int:
    if sys.platform != "linux":
        raise RuntimeError("parallel Learner subprocess ownership requires Linux PR_SET_PDEATHSIG")
    expected_parent = int(os.environ["QWEN_RL_PARALLEL_PARENT_PID"])
    parent_pid = os.getppid()
    if parent_pid != expected_parent:
        return 1
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PDEATHSIG = 1. Keep this launcher free of threads and perform the
    # syscall before torchrun imports Torch, CUDA, or initializes any workers.
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    if os.getppid() != parent_pid:
        return 1
    config_path = os.environ["QWEN_RL_PARALLEL_CONFIG"]
    with open(config_path, encoding="utf-8") as stream:
        runtime = json.load(stream)
    topology = runtime["topology"]
    if topology["mode"] == "single":
        os.environ["QWEN_RL_PARALLEL_RANK_PARENT_PID"] = str(parent_pid)
        command = [sys.executable, "-m", "qwen_rl.learner.parallel.rank_service", "--config", config_path]
    else:
        os.environ["QWEN_RL_PARALLEL_TORCHRUN_PID"] = str(os.getpid())
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            os.environ["QWEN_RL_PARALLEL_NPROC"],
            "-m",
            "qwen_rl.learner.parallel.rank_service",
            "--config",
            config_path,
        ]
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
