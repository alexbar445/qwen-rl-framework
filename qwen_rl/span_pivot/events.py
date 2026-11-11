"""Attach persisted replay evidence to native messages for conservative dedup."""
from __future__ import annotations

import copy
import hashlib
import json

from .dedup import deduplicate_rechecks


def deduplicate_replayed(messages: list[dict], replay: dict, *, retry_message_indices=()) -> dict:
    if replay.get('execution_complete') is not True:
        raise ValueError('dedup requires a complete real replay')
    events = copy.deepcopy(messages)
    calls = iter(replay['calls'])
    receipts = {}
    proposals = set(retry_message_indices)
    if any(type(index) is not int or not 0 <= index < len(events) for index in proposals):
        raise ValueError('invalid retry message index')
    for index, event in enumerate(events):
        event['event_id'] = str(index)
        if index in proposals:
            if event['role'] != 'assistant':
                raise ValueError('retry proposal must select an assistant event')
            event['proposal'] = {'kind': 'retry_only'}
        for call in event.get('tool_calls', []):
            actual = next(calls, None)
            arguments = call['function']['arguments']
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            if actual is None or call['function']['name'] != 'arc_action' or arguments['line'] != actual['line']:
                raise ValueError('native calls disagree with replay')
            if call['id'] in receipts:
                raise ValueError('duplicate native call ID')
            receipts[call['id']] = actual
            # Some source transports retain redundant linkage aliases. Their
            # values must agree before the derived proof view drops them.
            if set(call) - {'id', 'type', 'function', 'call_id', 'response_item_id'}:
                continue  # Unknown metadata keeps this turn ineligible.
            if 'call_id' in call and call['call_id'] != call['id']:
                raise ValueError('native call linkage aliases disagree')
            call.pop('call_id', None)
            call.pop('response_item_id', None)
        if event['role'] == 'tool':
            actual = receipts.get(event['tool_call_id'])
            if actual is None or json.loads(event['content']) != actual['response']:
                raise ValueError('native observation disagrees with replay')
            body = json.dumps(actual['response'], sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            event['receipt'] = {
                'state_hash_before': actual['semantic_state_before_sha256'],
                'state_hash_after': actual['semantic_state_after_sha256'],
                'observation_hash': hashlib.sha256(body.encode()).hexdigest(),
                'side_effect_free': actual['classification'] == 'read_only' and not actual['business_state_changed'],
                'complete': actual['successful'] and not actual['response'].get('truncated', False),
                'toolkit_implementation_sha256': replay['toolkit']['implementation_sha256'],
            }
    if next(calls, None) is not None:
        raise ValueError('replay contains additional calls')
    result = deduplicate_rechecks(events)
    result['messages'] = [copy.deepcopy(messages[int(event['event_id'])]) for event in result['events']]
    return result
