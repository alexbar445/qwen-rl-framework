"""Freeze a complete durable checkpoint plus offline RL inputs; never copy secrets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve(); out = a.output.resolve()
    # A portable checkpoint and BEPA ledger must share one stopped owner boundary.
    # Reuse the central writer's lock; never export an active training owner.
    from qwen_rl.learner.gradient_server.ownership import ProcessOwnerLock
    runtime = root / 'outputs/bonsai2-async-bepa-20261019'
    with ProcessOwnerLock.for_checkpoint_dir(runtime / 'checkpoints'):
        _write_seed(root, out)


def _write_seed(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    runtime = root / 'outputs/bonsai2-async-bepa-20261019'
    bepa = root / 'outputs/bonsai2-t2-bepa-20261015-r1/bepa'
    seed_bepa = out / 'outputs/bonsai2-t2-bepa-20261015-r1/bepa'
    seed_bepa.mkdir(parents=True)
    source = sqlite3.connect('file:' + str(bepa / 'state.sqlite3') + '?mode=ro', uri=True)
    # Pin the budget/exposure snapshot before selecting the durable checkpoint.
    source.execute('BEGIN'); source.execute('SELECT name FROM sqlite_master LIMIT 1').fetchone()
    pointer = json.loads((runtime / 'checkpoints/latest.json').read_text())
    cp = out / 'outputs/bonsai2-async-bepa-20261019/checkpoints'; cp.mkdir(parents=True)
    h = hashlib.sha256()
    with (runtime / 'checkpoints' / pointer['snapshot']).open('rb') as src, (cp / pointer['snapshot']).open('xb') as dst:
        for chunk in iter(lambda: src.read(8 * 1024**2), b''):
            h.update(chunk); dst.write(chunk)
    assert (cp / pointer['snapshot']).stat().st_size == pointer['size_bytes']
    (cp / 'latest.json').write_text(json.dumps(pointer))
    with sqlite3.connect(seed_bepa / 'state.sqlite3') as destination:
        source.backup(destination)
    source.close()
    for name in ['artifacts', 'teacher-outbox', 'sessions']:
        if (bepa / name).exists(): shutil.copytree(bepa / name, seed_bepa / name)
    for name in ['tokenizer', 'resume-adapter-v3']:
        shutil.copytree(runtime / name, cp.parent / name)
    shutil.copytree(root / 'data/generated', out / 'data/generated')
    shutil.copytree(root / 'data/evaluation', out / 'data/evaluation')
    active_root = root / 'outputs/local-runtime/container-state-managed/instance'
    active = json.loads((active_root / 'active.json').read_text())
    shutil.copyfile(active_root / 'runs' / active['run_id'] / 'runtime.yaml', out / 'runtime.yaml')
    shutil.copytree(root / 'configs', out / 'configs')
    identity = Path('outputs/ternary-lora-pp/20261014/bonsai2-t2-identity.json')
    (out / identity).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / identity, out / identity)
    # Generated private verifier data keeps its required private permissions.
    for private in (out / 'data/generated').glob('*/private'):
        private.chmod(0o700)
        for f in private.rglob('*'): f.chmod(0o700 if f.is_dir() else 0o600)
    import torch
    state = torch.load(cp / pointer['snapshot'], map_location='cpu', mmap=True, weights_only=False)
    for key in ['policy_version', 'optimizer_steps', 'scheduler_steps']:
        assert state[key] == pointer['policy_version'], key
    assert state.get('pending_attempt') is None and not state.get('prepared_windows')
    manifest = {'schema': 'qwen-rl-offline-seed/v1', 'frozen_at': time.time(),
                'checkpoint': pointer, 'checkpoint_sha256': h.hexdigest(),
                'secrets_included': False, 'packed_and_compile_caches_included': False}
    transitions = state.get('last_update_context', {}).get('arc_contract_transitions')
    if transitions:
        manifest['arc_contract_transitions'] = transitions
    (out / 'seed-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__': main()
