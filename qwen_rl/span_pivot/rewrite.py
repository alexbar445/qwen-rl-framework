"""Exact-source edits: numerical facts are witnessed by a preceding real tool."""
from __future__ import annotations

import copy
import hashlib
import json
import re


_FACT = re.compile(r' bbox \[([0-9, ]+)\], area ([0-9]+)\.')


def rewrite_observed_facts(source: dict, *, message_id: int, expected_text_sha256: str,
                           insert_at: int, replace_until: int, observation_call_index: int) -> dict:
    index = source['message_ids'].index(message_id)
    original = source['messages'][index]
    text = original.get('reasoning_content')
    if original['role'] != 'assistant' or not isinstance(text, str):
        raise ValueError('pivot must be a captured assistant reasoning message')
    if hashlib.sha256(text.encode()).hexdigest() != expected_text_sha256:
        raise ValueError('pivot source text hash mismatch')
    if not 0 <= insert_at < replace_until <= len(text):
        raise ValueError('invalid split position')
    before = sum(len([c for c in m.get('tool_calls', [])
                     if c['function']['name'] == 'arc_action']) for m in source['messages'][:index])
    if not 0 <= observation_call_index < before:
        raise ValueError('witness must already have executed before the pivot')
    response = source['runtime_calls'][observation_call_index]['response']
    if response.get('status') != 'ok' or response.get('type') != 'ObjectSet' or not response.get('artifact_id'):
        raise ValueError('witness must be a successful ObjectSet artifact')
    objects = response.get('summary', {}).get('objects_preview', [])
    objects = {item['id']: item for item in objects}
    edits = []
    # Only the explicitly delimited opening object list is eligible. Its
    # arithmetic/inference suffix remains untouched, including Open/Closed.
    for match in _FACT.finditer(text[insert_at:replace_until]):
        start, end = insert_at + match.start(), insert_at + match.end()
        line_start = text.rfind('\n', 0, start) + 1
        prefix = text[line_start:start]
        if not re.fullmatch(r'- o[0-9]+:', prefix):
            continue
        object_id = prefix[2:-1]
        object_ = objects.get(object_id)
        bbox = [int(n.strip()) for n in match.group(1).split(',')]
        area = int(match.group(2))
        if object_ is None or object_.get('bbox') != bbox or object_.get('area') != area:
            raise ValueError(f'computed facts lack a matching tool witness: {object_id}')
        edits.append({'start': start, 'end': end, 'original': text[start:end],
                      'sha256': hashlib.sha256(text[start:end].encode()).hexdigest(),
                      'object_id': object_id, 'bbox': bbox, 'area': area,
                      'witness_call_index': observation_call_index})
    if not edits:
        raise ValueError('no witnessed bbox/area facts in the selected opening list')
    cursor, parts = insert_at, []
    for edit in edits:
        parts.append(text[cursor:edit['start']]); cursor = edit['end']
    parts.append(text[cursor:])
    remainder = ''.join(parts)
    first = copy.deepcopy(original)
    first['reasoning_content'] = text[:insert_at]
    tool_id = 'span-pivot-inspect'
    if any(c['id'] == tool_id for m in source['messages'] for c in m.get('tool_calls', [])):
        raise ValueError('inserted tool call ID collides with the source')
    first['tool_calls'] = [{'id': tool_id, 'type': 'function', 'function': {
        'name': 'arc_action', 'arguments': json.dumps({'line':
            f'inspect(source="{response["artifact_id"]}", limit=5)'}, ensure_ascii=False)}}]
    second = copy.deepcopy(original)
    second['reasoning_content'] = remainder
    messages = copy.deepcopy(source['messages'])
    # The actual response is filled by the replay runner, never authored here.
    messages[index:index + 1] = [first, {'role': 'tool', 'name': 'arc_action',
        'tool_call_id': tool_id, 'content': None}, second]
    return {'messages': messages, 'pivot_source_message_id': message_id,
            'inserted_message_index': index, 'inserted_call_index': before,
            'edits': edits, 'preserved_reasoning_characters': len(text) - sum(e['end']-e['start'] for e in edits),
            'original_reasoning_characters': len(text), 'noncomputational_text_preserved_exactly': True,
            'reasoning_order_preserved': True, 'generated_bridge_text': ''}
