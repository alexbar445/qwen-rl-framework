"""First-boot seed selection shared by local mounts and the SSH bridge."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time

from qwen_rl.deployment.instance_compat import _read_regular_json, verify_seed
from qwen_rl.runtime.manifest import write_json_atomic

RUN = 'bonsai2-async-bepa-20261019'


def receive(state=Path('/state'), stream=None):
    """Publish a complete trusted-server bundle atomically, never a partial copy."""
    state.mkdir(parents=True, exist_ok=True)
    target = state / 'latest-seed'
    if target.exists():
        raise RuntimeError('a synchronized seed already exists')
    with tempfile.TemporaryDirectory(prefix='seed-incoming-', dir=state) as tmp:
        with tarfile.open(fileobj=stream or sys.stdin.buffer, mode='r|') as archive:
            archive.extractall(tmp, filter='data')
        incoming = Path(tmp)
        if not (incoming / 'seed-manifest.json').is_file():
            raise RuntimeError('incomplete synchronized seed')
        os.rename(incoming, target)


def select_seed(bundled: Path, work: Path, state: Path):
    installed = state / 'seed-sync-installed.json'
    existing = work / 'outputs' / RUN / 'checkpoints/latest.json'
    if existing.exists():
        # A local production mount or a restarted container owns its own progress.
        try:
            installed.lstat()
            installed_present = True
        except FileNotFoundError:
            installed_present = False
        if installed_present:
            marker = _read_regular_json(installed, limit=64 * 1024)
            candidate = Path(marker['seed_path'])
            if not candidate.is_absolute():
                raise RuntimeError('installed seed marker has no absolute seed path')
            verify_seed(candidate, staged_inputs=True)
            return candidate
        return bundled
    server = os.environ.get('QWEN_RL_SERVER_ROOT')
    if server:
        import subprocess
        source = state / 'local-latest-seed'
        if source.exists(): shutil.rmtree(source)
        state.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-B', '-m', 'qwen_rl.deployment.seed_export',
                        '--root', server, '--output', str(source)], check=True)
    else:
        source = os.environ.get('QWEN_RL_SEED_DIR')
    if source:
        candidate = Path(source)
    elif os.environ.get('QWEN_RL_SEED_SYNC', 'required') == 'bundled':
        return bundled
    else:
        candidate = state / 'latest-seed'
        deadline = time.monotonic() + int(os.environ.get('QWEN_RL_SEED_WAIT_SECONDS', '1800'))
        print('Waiting for latest complete checkpoint from local server; no training roles started.', flush=True)
        while not (candidate / 'seed-manifest.json').is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError('latest checkpoint sync timed out; refusing stale fallback')
            time.sleep(2)
    result = verify_seed(candidate, staged_inputs=True)
    # No output/checkpoint is replaced. Only the matching immutable inputs are
    # installed before prepare() seeds the still-empty training directories.
    for name in ['data', 'configs']:
        for child in (candidate / name).iterdir():
            target = work / name / child.name
            if target.exists():
                if target.is_dir(): shutil.rmtree(target)
                else: target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            if child.is_dir(): shutil.copytree(child, target)
            else: shutil.copy2(child, target)
    state.mkdir(parents=True, exist_ok=True)
    candidate = candidate.absolute()
    write_json_atomic(state / 'seed-sync-selected.json', {'seed_path': str(candidate), **result})
    return candidate


if __name__ == '__main__':
    if sys.argv[1:] != ['receive']: raise SystemExit('usage: seed_sync receive')
    receive()
