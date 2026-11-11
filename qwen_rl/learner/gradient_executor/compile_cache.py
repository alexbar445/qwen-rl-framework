"""Observe persistent compiler caches without clearing or changing them."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

from qwen_rl.runtime.manifest import write_json_atomic

_CACHE_ENV = ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH')


def cache_counters():
    from torch._dynamo.utils import counters
    return {group: {str(key): int(value) for key, value in list(values.items())
                    if isinstance(value, int)}
            for group, values in list(counters.items())
            if group in {'inductor', 'aot_autograd', 'stats'} or 'cache' in group}


def counter_delta(current, baseline):
    return {group: {key: value - baseline.get(group, {}).get(key, 0)
                    for key, value in values.items()}
            for group, values in current.items()}


class CompileCacheMonitor:
    """Record process counters from startup, including lazy first compilation."""

    def __init__(self, runtime_dir, *, read_counters=cache_counters, interval=5):
        self.root = Path(runtime_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.read_counters = read_counters
        self.baseline = read_counters()
        self.started = time.time()
        self.paths = {key: os.environ.get(key) for key in _CACHE_ENV}
        self.interval = interval
        self.stop = threading.Event()
        self.previous = None
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True, name='compile-cache-monitor')
        self.thread.start()

    def sample(self):
        current = self.read_counters()
        payload = {'schema': 'qwen-rl-compile-cache/v1', 'pid': os.getpid(),
                   'process_monitor_started_at': self.started, 'sampled_at': time.time(),
                   'cache_paths': self.paths, 'counters': current,
                   'since_monitor_start': counter_delta(current, self.baseline),
                   'scope': 'pytorch_process_counters; not per-kernel Triton or CUDA hit rates'}
        write_json_atomic(self.root / 'compile-cache.json', payload)
        if current != self.previous:
            with (self.root / 'compile-cache.jsonl').open('a') as stream:
                stream.write(json.dumps(payload, allow_nan=False) + '\n')
            self.previous = current

    def _run(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except Exception:
                # Observability must not interrupt a gradient job.
                import logging
                logging.getLogger(__name__).exception('compiler cache telemetry failed')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        self.sample()
