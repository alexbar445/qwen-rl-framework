"""Explicit, audited checkpoint transition to the d85 ARC action contract.

Run only after stopping the owning instance, or on an isolated seed copy. The
central engine's normal CPU lock and checkpoint writer remain authoritative.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

OLD_TOOLKIT = {
    'arc_toolkit_tree_sha256': 'd93287c1fd616cc95f138cdee6ff4c478c76a193159fa54438f1824126c674c7',
    'toolkit_dependency_sha256': 'd719a72411ce91b86648be07308b8a9155fa8e72beafd1214d7df76b6504f946',
}


def assert_same(left, right, path='state'):
    import torch
    import numpy as np
    if isinstance(left, torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.dtype!=right.dtype or left.shape!=right.shape or not torch.equal(left,right):
            raise RuntimeError('transition changed tensor: '+path)
    elif isinstance(left,np.ndarray):
        if not isinstance(right,np.ndarray) or not np.array_equal(left,right): raise RuntimeError('transition changed RNG: '+path)
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys(): raise RuntimeError('transition changed mapping: '+path)
        for key in left: assert_same(left[key],right[key],path+'.'+str(key))
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right): raise RuntimeError('transition changed sequence: '+path)
        for i,(a,b) in enumerate(zip(left,right)): assert_same(a,b,path+'.'+str(i))
    elif left!=right:
        raise RuntimeError('transition changed value: '+path)


def transition(config_path: Path, audit_root: Path, *, apply=False):
    import torch
    from qwen_rl.runtime.config import load_config
    from qwen_rl.learner.recovery import _semantic_identity
    from qwen_rl.prompt import validate_toolkit_binding, load_toolkit_runtime
    cfg=load_config(config_path);raw=cfg.as_dict()
    binding=validate_toolkit_binding()
    assert load_toolkit_runtime()._parse_call("O.recolor(source='test/0/input', mapping={4:8})")[1]['mapping']=={'4':8}
    cp=Path(raw['run_dir'])/'checkpoints'
    pointer=json.loads((cp/'latest.json').read_text());parent=cp/pointer['snapshot']
    original=torch.load(parent,map_location='cpu',mmap=True,weights_only=False)
    context=copy.deepcopy(original['last_update_context'])
    old=context['source_identity'];new=_semantic_identity(raw)
    changed={k for k,v in new.items() if old.get(k)!=v}
    if changed!=set(OLD_TOOLKIT): raise RuntimeError('transition permits only the two known toolkit identity changes: '+repr(sorted(changed)))
    if any(old.get(k)!=v for k,v in OLD_TOOLKIT.items()): raise RuntimeError('checkpoint is not the reviewed pre-d85 toolkit')
    if original.get('prepared_windows') or original.get('pending_attempt') is not None:
        raise RuntimeError('checkpoint has outstanding optimizer work')
    for name in ['prepared-window.json','pending-attempt.json','async-checkpoint-intent.json']:
        if (cp/name).exists(): raise RuntimeError('checkpoint has a live transaction sidecar: '+name)
    parent_sha=hashlib.file_digest(parent.open('rb'),'sha256').hexdigest()
    receipt={'schema':'qwen-rl-arc-contract-transition/v1','transition_id':uuid.uuid4().hex,
        'target_commit':'d85f9c7ef1bab30408e81f934279b719812a3eea','source_checkpoint':pointer,
        'source_checkpoint_sha256':parent_sha,'action_language_fingerprint':binding['fingerprint'],
        'identity_changes':{k:{'from':old[k],'to':new[k]} for k in sorted(changed)},
        'policy_version':original['policy_version'],'optimizer_steps':original['optimizer_steps'],
        'logical_run':{k:v for k,v in context['logical_job'].items() if k!='task_ids'},
        'preserve_model_optimizer_scheduler_rng_cursor_budget':True,'applied':False}
    if not apply: return receipt
    audit=audit_root/receipt['transition_id'];audit.mkdir(parents=True,exist_ok=False)
    try: os.link(parent,audit/parent.name)
    except OSError: shutil.copy2(parent,audit/parent.name)
    shutil.copy2(cp/'latest.json',audit/'parent-latest.json')
    (audit/'transition.json').write_text(json.dumps(receipt,indent=2))
    from qwen_rl.learner.app import _load_engine
    engine=_load_engine(raw)
    try:
        status=engine.core.status()
        if status['dirty'] or status['pending_window'] or status['pending_attempt'] is not None:
            raise RuntimeError('central engine is not quiescent')
        if json.loads((cp/'latest.json').read_text())!=pointer: raise RuntimeError('checkpoint advanced during transition preparation')
        context['source_identity'].update({k:new[k] for k in changed})
        history=context.setdefault('arc_contract_transitions',[])
        history.append({k:receipt[k] for k in ['transition_id','target_commit','source_checkpoint_sha256','action_language_fingerprint','identity_changes']})
        engine.core.save_checkpoint(update_context=context,reason='explicit_arc_contract_transition_d85f9c7e')
        engine.core.close()
        after_pointer=json.loads((cp/'latest.json').read_text())
        after=torch.load(cp/after_pointer['snapshot'],map_location='cpu',mmap=True,weights_only=False)
        allowed={'generation','snapshot_id','last_update_context','checkpoint_reason'}
        for key in original:
            if key not in allowed: assert_same(original[key],after[key],key)
        assert_same(context,after['last_update_context'],'last_update_context')
        receipt.update(applied=True,target_checkpoint=after_pointer,all_preserved_state_equal=True,
            target_checkpoint_sha256=hashlib.file_digest((cp/after_pointer['snapshot']).open('rb'),'sha256').hexdigest())
        (audit/'transition.json').write_text(json.dumps(receipt,indent=2))
        return receipt
    finally:
        engine.shutdown()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--apply',action='store_true');a=p.parse_args()
    print(json.dumps(transition(a.config,a.audit_root,apply=a.apply)),flush=True)

if __name__=='__main__': main()
