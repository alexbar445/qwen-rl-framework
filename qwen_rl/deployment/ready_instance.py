"""Prepare the bundled offline state and launch a clean GPU instance."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys
import time
import urllib.request

from .instance_compat import _read_regular_json, read_checkpoint_metadata


_PRIVATE_UIDS = {0, 10001, os.getuid()}


def _private_directory(path: Path) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid not in _PRIVATE_UIDS):
        raise RuntimeError('credential directory must be a private owned directory')
    return path


def _read_private_secret(path: Path) -> str:
    for parent in (path.parent, path.parent.parent):
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError('credential path parents must be real directories')
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
        raise RuntimeError('credential file must be private, regular and singly linked')
    try:
        from .local import _read_secret_file
        return _read_secret_file(path, label='ready instance', uid=info.st_uid)
    except Exception as exc:
        raise RuntimeError('credential file could not be read safely') from exc


def _write_private_secret(path: Path, value: str) -> None:
    parent = _private_directory(path.parent)
    try:
        prior = path.lstat()
    except FileNotFoundError:
        prior = None
    if prior is not None and (not stat.S_ISREG(prior.st_mode) or stat.S_ISLNK(prior.st_mode)
                              or prior.st_nlink != 1 or prior.st_uid not in _PRIVATE_UIDS
                              or stat.S_IMODE(prior.st_mode) != 0o600):
        raise RuntimeError('existing credential path is not a private regular file')
    temporary = parent / ('.' + path.name + '.' + secrets.token_hex(8) + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        if os.geteuid() == 0:
            os.fchown(fd, 10001, 10001)
        payload = value.encode('utf-8')
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    if path.is_symlink():
        temporary.unlink(missing_ok=True)
        raise RuntimeError('credential target became a symlink')
    os.replace(temporary, path)


def retire_stale_packed_cache(root: Path):
    """Only discard reproducible packed caches; never touch training checkpoints."""
    retired = []
    if not root.exists(): return retired
    if root.is_symlink(): raise RuntimeError('packed cache root must not be a symlink')
    for manifest in root.glob('*/manifest.json'):
        entry = manifest.parent
        if entry.is_symlink(): raise RuntimeError('packed cache entry must not be a symlink')
        saved = json.loads(manifest.read_text())
        payload = entry / 'packed.bin'
        if payload.is_symlink(): raise RuntimeError('packed cache payload must not be a symlink')
        info = payload.stat() if payload.exists() else None
        actual = {k: int(getattr(info, 'st_' + k)) for k in
                  ['dev', 'ino', 'size', 'mtime_ns', 'ctime_ns']} if info else None
        if (actual != saved.get('file_signature') or info is None
                or info.st_uid != 10001 or stat.S_IMODE(info.st_mode) != 0o444):
            retired.append({'entry': entry.name, 'reason': 'filesystem identity changed; rebuild from baked immutable model',
                            'previous_signature': saved.get('file_signature'), 'current_signature': actual})
            shutil.rmtree(entry)
    return retired


def prepare(seed: Path, work: Path, state: Path):
    seed = Path(seed)
    seed_info = seed.lstat()
    if not stat.S_ISDIR(seed_info.st_mode) or stat.S_ISLNK(seed_info.st_mode):
        raise RuntimeError('seed root must be a real directory')
    manifest = _read_regular_json(seed / 'seed-manifest.json')
    run = work / 'outputs/bonsai2-async-bepa-20261019'
    existing_checkpoint = (run / 'checkpoints/latest.json').exists()
    for source in (seed / 'outputs').iterdir():
        target = work / 'outputs' / source.name
        if not target.exists():
            shutil.copytree(source, target)
    if not (run / 'checkpoints/latest.json').is_file():
        raise RuntimeError('existing state is incomplete; refusing to overwrite it with image seed')
    selected_checkpoint = read_checkpoint_metadata(run / 'checkpoints')
    retired = retire_stale_packed_cache(run / 'frozen-base')
    state.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        for root in [work / 'outputs', state]:
            for base, dirs, files in os.walk(root):
                if Path(base).stat().st_uid != 10001 or Path(base).stat().st_gid != 10001:
                    os.chown(base, 10001, 10001)
                for name in files:
                    p = Path(base) / name
                    if not p.is_symlink() and (p.stat().st_uid != 10001 or p.stat().st_gid != 10001):
                        os.chown(p, 10001, 10001)
    receipt = {'image_seed_checkpoint': manifest['checkpoint'],
               'selected_checkpoint': selected_checkpoint,
               'selected_checkpoint_source': 'existing_training_state' if existing_checkpoint else 'image_seed',
               'restore_validation': 'deferred_to_managed_instance_start',
               'retired_derived_caches': retired,
               'compile_cache_policy': 'preserve /state/jit across process restarts',
               'existing_training_state_overwritten': False}
    (state / 'ready-seed-receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('command', nargs='?', default='serve', choices=['serve', 'status', 'stop', 'run'])
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    seed = Path('/opt/qwen-rl-seed'); work = Path('/work'); state = Path('/state')
    if args.command != 'serve':
        path = state / 'instance-ready.json'
        if not path.is_file(): raise RuntimeError('instance has not been prepared yet')
        command = [args.command]
        if args.command == 'run':
            if args.steps is None or args.steps < 1: parser.error('run requires --steps N')
            command += ['--steps', str(args.steps)]
        if os.geteuid() == 0:
            os.setgroups([]); os.setgid(10001); os.setuid(10001)
        os.execv(sys.executable, [sys.executable, '-B', '-m', 'qwen_rl', 'instance',
                                 '--deployment', str(path), *command])

    if shutil.disk_usage(work).free < 24 * 1024**3:
        raise RuntimeError('RL needs at least 24 GiB free writable space; rent 64 GiB or more (128 GiB recommended)')
    from qwen_rl.deployment.seed_sync import select_seed
    seed = select_seed(seed, work, state)
    receipt = prepare(seed, work, state)
    selected = state / 'seed-sync-selected.json'
    if selected.exists():
        os.replace(selected, state / 'seed-sync-installed.json')
    if args.prepare_only:
        print(json.dumps(receipt)); return
    import yaml
    cfg = yaml.safe_load((seed / 'runtime.yaml').read_text())
    url = os.environ.get('QWEN_RL_TEACHER_BASE_URL', 'http://127.0.0.1:18137/v1')
    input_credential_dir = state / 'ready-credentials'
    credential_dir = _private_directory(state / 'ready-credentials-managed')
    key = os.environ.get('DEEPSEEK_API_KEY')
    key_path = input_credential_dir / 'deepseek_api_key'
    managed_key_path = credential_dir / 'deepseek_api_key'
    deadline = time.monotonic() + int(os.environ.get('QWEN_RL_TEACHER_WAIT_SECONDS', '900'))
    announced = False
    while cfg['bepa']['enabled']:
        if not key:
            try:
                key = _read_private_secret(key_path)
            except FileNotFoundError:
                try:
                    key = _read_private_secret(managed_key_path)
                except FileNotFoundError:
                    pass
        if key:
            try:
                req = urllib.request.Request(url.rstrip('/') + '/models',
                    headers={'Authorization': 'Bearer ' + key})
                with urllib.request.urlopen(req, timeout=3) as response:
                    if response.status == 200: break
            except Exception:
                pass
        if not announced:
            print('Waiting for local teacher bridge/API secret; no training roles started.', flush=True)
            announced = True
        if time.monotonic() >= deadline:
            raise RuntimeError('teacher bridge/API secret did not become ready before the startup deadline')
        time.sleep(2)
    if key: os.environ['DEEPSEEK_API_KEY'] = key
    cfg['bepa']['teacher']['base_url'] = url or cfg['bepa']['teacher']['base_url']
    if os.environ.get('QWEN_RL_TEACHER_MODEL'):
        cfg['bepa']['teacher']['model'] = os.environ['QWEN_RL_TEACHER_MODEL']
    cfg['learner']['gradient_local_work_stealing'] = True
    cfg['learner']['gradient_tasks_per_worker'] = 2
    cfg['evaluation']['concurrency'] = 64
    credentials = {}
    for name in ['ARC_ENV_TOKEN', 'BEPA_TEACHER_TOKEN', 'DEEPSEEK_API_KEY']:
        path = credential_dir / name.lower()
        value = os.environ.get(name)
        if value is None:
            try:
                value = _read_private_secret(path)
            except FileNotFoundError:
                if name == 'DEEPSEEK_API_KEY':
                    continue
                value = secrets.token_hex(32)
        try:
            existing = _read_private_secret(path)
        except FileNotFoundError:
            existing = None
        if existing != value:
            _write_private_secret(path, value)
        try:
            _read_private_secret(path)
        except FileNotFoundError:
            continue
        credentials[name] = str(path)
    indices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    inventory = {int(row.split(',')[0]): row.split(',')[1].strip() for row in indices.splitlines() if row.strip()}
    available = list(inventory)
    gpus = [int(x) for x in os.environ['QWEN_RL_GPUS'].split(',')] if os.environ.get('QWEN_RL_GPUS') else available
    if len(gpus) < 2 or len(set(gpus)) != len(gpus) or not set(gpus) <= set(available):
        raise RuntimeError('select at least two distinct available GPUs for the default shared Learner profile')
    runtime_path = state / 'ready-runtime.yaml'; runtime_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    deployment = {'schema': 'qwen-rl-instance/v1', 'runtime_config': str(runtime_path),
                  'state_dir': str(state / 'instance'), 'gpus': gpus,
                  'engine_binary': '/usr/local/bin/ninfer-serve',
                  'python': sys.executable, 'worker_python': sys.executable,
                  'start_timeout': 1800, 'stop_timeout': 1200,
                  'teacher_base_url': cfg['bepa']['teacher']['base_url'],
                  'credential_files': credentials, 'dashboard_port': 11507, 'dashboard_host': '0.0.0.0'}
    path = state / 'instance-ready.json'; path.write_text(json.dumps(deployment, indent=2))
    if os.geteuid() == 0:
        for p in [credential_dir, *credential_dir.iterdir(), runtime_path, path]:
            os.chown(p, 10001, 10001)
        os.setgroups([]); os.setgid(10001); os.setuid(10001)
    os.environ.update(HOME='/state/home', ARC_RL_ROOT='/work', PYTHONPATH='/work',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # A separate process releases every CUDA context before the CPU supervisor
    # starts. UUID selection also handles Docker's CUDA ordinal remapping.
    smoke_env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(inventory[gpu] for gpu in gpus))
    smoke = subprocess.run([sys.executable, '-B', '-m', 'qwen_rl.deployment.instance_compat', '--gpu'],
                           env=smoke_env, capture_output=True, text=True)
    (state / 'ready-cuda-preflight.log').write_text(smoke.stdout + smoke.stderr)
    if smoke.returncode:
        raise RuntimeError('GPU preflight failed; inspect /state/ready-cuda-preflight.log')
    hardware = json.loads(smoke.stdout.strip().splitlines()[-1])
    (state / 'ready-cuda-receipt.json').write_text(json.dumps(hardware, indent=2))
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    os.execv(sys.executable, [sys.executable, '-B', '-m', 'qwen_rl', 'instance',
                             '--deployment', str(path), 'serve'])


if __name__ == '__main__': main()
