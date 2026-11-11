"""Read-only event presentation; derived classifications retain their evidence basis."""
from __future__ import annotations
import json
import math
from datetime import datetime
import sqlite3
from typing import Any, Mapping

_INFO = frozenset({'activated','claimed','committed','configured','heartbeat','max_job_tokens_updated',
    'max_total_tokens_updated','optimizer_entered','prepared','queued','ready','released',
    'reservations_released','reserved','seed_cache_bound','seed_claimed','seed_generation',
    'seed_succeeded','source_submitted','source_verified','stored','usage_recorded',
    'cache_read_excluded_from_settled_usage','running','pending','succeeded','completed',
    'acknowledged','published','snapshot_recorded'})
_WARNING = frozenset({'aborted','seed_failed','exhausted','expired','expired_after_update',
    'cache_read_reconciliation_incomplete','retry_wait','cancelled','filtered_limit'})
_ERROR = frozenset({'failed','error','commit_unknown','outcome_unknown'})
_ENTITY = {'job':('rescue_jobs','job_id'),'plan':('plans','plan_id'),
    'cache':('cache_entries','cache_entry_id'),'episode':('episodes','episode_id'),'reservation':('reservations','reservation_id'),
    'update':('update_attempts','attempt_id')}

def severity_fields(action: Any) -> dict[str, str]:
    severity = ('error' if action in _ERROR else 'warning' if action in _WARNING
                else 'info' if action in _INFO else 'unknown') if isinstance(action,str) else 'unknown'
    return {'severity':severity,'severity_basis':f'derived from recorded action/status: {action}'
            if severity!='unknown' else 'recorded action/status has no severity classification'}

def _version(value: Any) -> int | None:
    return value if type(value) is int and value>=0 else None

def policy_fields(value: Any, basis: str, role: str) -> dict[str, Any]:
    version = _version(value)
    return {'policy_version':version,'policy_applicability':'known' if version is not None else 'unknown',
            'policy_basis':basis if version is not None else 'applicable policy version was not recorded',
            'policy_role':role}

def event_timestamp(value: Any) -> float | None:
    if type(value) in (int,float): return float(value) if math.isfinite(value) else None
    if isinstance(value,str):
        try:
            stamp=datetime.fromisoformat(value.replace('Z','+00:00'))
            return stamp.timestamp() if stamp.tzinfo is not None else None
        except ValueError: pass
    return None

def _seed_policy(connection: sqlite3.Connection, row: Mapping[str,Any], detail: Mapping[str,Any]):
    attempt = detail.get('attempt_id')
    if not isinstance(attempt,str): return None
    prior = connection.execute(
        "SELECT sequence,details_json FROM events WHERE entity_type='plan' AND entity_id=? "
        "AND action='seed_claimed' AND sequence<? ORDER BY sequence DESC LIMIT 40",
        (row.get('entity_id'),row['sequence'])).fetchall()
    for candidate in prior:
        try: data=json.loads(candidate['details_json'])
        except (ValueError,TypeError): continue
        if not isinstance(data,dict) or data.get('attempt_id')!=attempt: continue
        if any(key in detail and key in data and detail[key]!=data[key]
               for key in ('run_id','generation_id','fence','engine_epoch')): continue
        if _version(data.get('policy_version')) is not None:
            return data['policy_version'],candidate['sequence']
    return None

def _related(connection: sqlite3.Connection, entity: str, entity_id: Any) -> dict[str, Any]:
    spec = _ENTITY.get(entity)
    if spec is None or not isinstance(entity_id,str): return {}
    table, key = spec  # Closed identifiers, values remain bound parameters.
    found = connection.execute(f'SELECT * FROM {table} WHERE {key}=?',(entity_id,)).fetchone()
    return dict(found) if found is not None else {}

def ledger_event(connection: sqlite3.Connection, row: Mapping[str, Any], db: str) -> dict[str, Any]:
    raw = row.get('details_json')
    detail: dict[str, Any] = {}
    detail_status = 'not_recorded' if raw is None else 'invalid'
    if isinstance(raw,str) and len(raw)<=65536:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded,dict): detail,detail_status=decoded,'recorded'
        except (ValueError,TypeError): pass
    entity, action = str(row['entity_type']),str(row['action'])
    related = _related(connection,entity,row.get('entity_id'))
    event = {'id':f"bepa:{row['sequence']}",'type':'bepa_'+entity,'at':row.get('created_at'),
        'title':action,'action':action,'sequence':row['sequence'],'entity_type':entity,
        'entity_id':row.get('entity_id'),'detail':detail,'detail_status':detail_status,
        'reason':row.get('reason'),'source':'BEPA event ledger',
        'ref':f"{db}#event-{row['sequence']}",'evidence_scope':'shared_bepa_ledger',
        **severity_fields(action)}
    for key in ('task_key','group_id','budget_scope','engine_epoch','batch_id'):
        value = detail.get(key,related.get(key))
        if isinstance(value,str) and value: event[key]=value
    if entity=='task': event['task_key']=row.get('entity_id')
    if entity=='budget_scope': event['budget_scope']=row.get('entity_id')
    if entity=='batch': event['batch_id']=row.get('entity_id')
    candidates=[('policy_version','recorded_policy'),('start_policy_version','update_start'),
                ('new_policy_version','committed_result'),('source_policy_version','replay_source')]
    for key,role in candidates:
        if _version(detail.get(key)) is not None:
            event.update(policy_fields(detail[key],'event.details_json.'+key,role));break
    else:
        value, basis, role = None,'',''
        if entity in {'cache','episode'}:
            value,basis,role=related.get('source_policy_version'),'associated '+entity+'.source_policy_version','replay_source'
        elif entity=='plan' and detail.get('cache_entry_id'):
            cache=_related(connection,'cache',detail['cache_entry_id'])
            value,basis,role=cache.get('source_policy_version'),'event.cache_entry_id -> cache.source_policy_version','replay_source'
        elif entity=='plan' and action.startswith('seed_'):
            seed=_seed_policy(connection,row,detail)
            if seed: value,basis,role=seed[0],f'matching attempt seed_claimed event #{seed[1]}.policy_version','seed_policy'
        elif entity=='reservation' and related.get('update_attempt_id'):
            update = _related(connection,'update',related['update_attempt_id'])
            value,basis,role=update.get('start_policy_version'),'associated update.start_policy_version (not event-time measurement)','update_start'
        elif entity=='update':
            # A committed record may gain new_policy_version later: do not attach that later
            # version to an earlier prepared event.
            key='new_policy_version' if action=='committed' else 'start_policy_version'
            value,basis,role=related.get(key),'associated update.'+key,'committed_result' if action=='committed' else 'update_start'
        if _version(value) is not None:
            event.update(policy_fields(value,basis,role))
        elif entity in {'task','budget_scope','job','batch'} or (entity=='plan' and action=='ready'):
            event.update(policy_version=None,policy_applicability='not_applicable',
                policy_basis='task/budget/teacher bookkeeping is not a Qwen policy update',policy_role=None)
        else:
            event.update(policy_fields(None,'','unrecorded'))
    return event
