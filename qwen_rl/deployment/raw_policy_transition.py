"""Audited CPU checkpoint transition from conditioned to raw ARC policy.

The owning instance must be stopped and all staged jobs from the old policy
resolved by the manager. The command intentionally does not stop roles, edit
runtime config, migrate staged jobs, or bypass checkpoint ownership.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from .arc_transition import assert_same


class TransitionError(RuntimeError):
    pass


def derive_context(context, old_semantics, new_semantics):
    """Allow only the policy distribution and its grammar identity to change."""
    old = copy.deepcopy(old_semantics)
    new = copy.deepcopy(new_semantics)
    if old.get('policy_distribution') != 'grammar_conditioned_policy/v1' or new.get('policy_distribution') != 'raw_policy':
        raise TransitionError('transition requires conditioned source and raw target')
    if not isinstance(old.get('grammar_policy'), dict) or 'grammar_policy' in new:
        raise TransitionError('grammar identity must belong only to the source policy')
    changes = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    if changes != {'policy_distribution', 'grammar_policy'}:
        raise TransitionError('unexpected semantic changes: ' + repr(sorted(changes)))
    old_identity = context.get('source_identity')
    if not isinstance(old_identity, dict) or any(old_identity.get(k) != v for k, v in old.items()):
        raise TransitionError('checkpoint source identity does not match source configuration')
    result = copy.deepcopy(context)
    identity = result['source_identity']
    for key in changes:
        if key in new:
            identity[key] = copy.deepcopy(new[key])
        else:
            identity.pop(key, None)
    return result, {key: {'from': old.get(key), 'to': new.get(key)} for key in sorted(changes)}


def transition(old_config_path: Path, new_config_path: Path, audit_root: Path, *, apply: bool = False):
    import torch
    from qwen_rl.learner.app import _load_engine
    from qwen_rl.deployment.instance_recovery import _close_engine
    from qwen_rl.learner.recovery import preflight_compatibility
    from qwen_rl.runtime.config import load_config

    torch.set_num_threads(2)
    old = load_config(old_config_path).as_dict()
    new = load_config(new_config_path).as_dict()
    expected = copy.deepcopy(old)
    if expected['rollout']['constrained_decoding'] is not True:
        raise TransitionError('source rollout must use constrained decoding')
    expected['rollout']['constrained_decoding'] = False
    if expected != new:
        raise TransitionError('target runtime may only disable constrained decoding')
    old_semantics = preflight_compatibility(old)['semantics']
    new_semantics = preflight_compatibility(new)['semantics']
    root = Path(old['learner']['recovery_dir']).resolve()
    pointer_path = root / 'latest.json'
    pointer = json.loads(pointer_path.read_text())
    source = root / pointer['snapshot']
    if source.is_symlink() or not source.is_file() or source.stat().st_size != pointer['size_bytes']:
        raise TransitionError('source checkpoint pointer does not identify a regular complete snapshot')
    payload = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    if (payload.get('snapshot_id') != pointer['snapshot_id'] or payload.get('policy_version') != pointer['policy_version']
            or payload.get('optimizer_steps') != pointer['optimizer_steps']):
        raise TransitionError('checkpoint identity differs from pointer')
    if payload.get('prepared_windows') or payload.get('pending_attempt') is not None:
        raise TransitionError('checkpoint contains unresolved optimizer work')
    for name in ('prepared-window.json', 'pending-attempt.json', 'async-checkpoint-intent.json'):
        if (root / name).exists():
            raise TransitionError('live transaction sidecar exists: ' + name)
    context, changes = derive_context(payload['last_update_context'], old_semantics, new_semantics)
    compatibility = copy.deepcopy(payload['compatibility'])
    receipt = {'schema': 'qwen-rl-raw-policy-transition/v1', 'source_checkpoint': pointer,
               'source_checkpoint_sha256': hashlib.file_digest(source.open('rb'), 'sha256').hexdigest(),
               'semantic_changes': changes,
               'applied': False}
    if not apply:
        return receipt
    audit = audit_root / uuid.uuid4().hex
    audit.mkdir(parents=True, exist_ok=False)
    try:
        os.link(source, audit / source.name)
    except OSError:
        shutil.copy2(source, audit / source.name)
    shutil.copy2(pointer_path, audit / 'source-latest.json')
    (audit / 'transition.json').write_text(json.dumps(receipt, indent=2) + '\n')
    engine = _load_engine(old, recovery_only=True, preserve_control_files=True)
    try:
        status = engine.core.status()
        if (not status.get('resumed') or status.get('dirty') or status.get('pending_window')
                or status.get('pending_attempt') is not None or json.loads(pointer_path.read_text()) != pointer):
            raise TransitionError('checkpoint owner is not quiescent or latest advanced')
        assert_same(payload['compatibility'], engine.core.compatibility, 'compatibility')
        engine.core.compatibility = compatibility
        engine.core.save_checkpoint(update_context=context, reason='explicit_raw_policy_transition',
                                    rng_state=payload['rng_state'])
        target = json.loads(pointer_path.read_text())
        saved = torch.load(root / target['snapshot'], map_location='cpu', mmap=True, weights_only=False)
        allowed = {'generation', 'snapshot_id', 'last_update_context', 'checkpoint_reason', 'compatibility'}
        for key in payload:
            if key not in allowed:
                assert_same(payload[key], saved[key], key)
        assert_same(context, saved['last_update_context'], 'last_update_context')
        assert_same(compatibility, saved['compatibility'], 'compatibility')
        receipt.update(applied=True, target_checkpoint=target, preserved_state_equal=True,
                       target_checkpoint_sha256=hashlib.file_digest((root / target['snapshot']).open('rb'), 'sha256').hexdigest())
        (audit / 'transition.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt
    finally:
        _close_engine(engine)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-config', type=Path, required=True)
    p.add_argument('--new-config', type=Path, required=True)
    p.add_argument('--audit-root', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    print(json.dumps(transition(args.old_config, args.new_config, args.audit_root, apply=args.apply)), flush=True)


if __name__ == '__main__':
    main()
