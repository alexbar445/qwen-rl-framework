"""Read stopped, parent-verified solver records without modifying their source."""
from __future__ import annotations

import hashlib
import base64
import json
from pathlib import Path
import sqlite3


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


def read_verified_source(ready_path: Path) -> dict:
    ready_path = Path(ready_path).resolve(strict=True)
    ready_bytes = ready_path.read_bytes()
    ready = json.loads(ready_bytes)
    if ready.get('eligible') is not True or ready.get('test_exact') is not True:
        raise ValueError('source lacks a parent-verified successful freeze')
    freeze_path = Path(ready['freeze_path']).resolve(strict=True)
    freeze_bytes = freeze_path.read_bytes()
    if digest(freeze_bytes) != ready['freeze_sha256']:
        raise ValueError('source freeze hash mismatch')
    freeze = json.loads(freeze_bytes)
    worker = Path(ready['source_path']).resolve(strict=True)
    database = worker / 'private-hermes/state.db'
    # These records belong to a stopped solver. Read-only SQLite includes its
    # recorded WAL; no rows are changed and the source files are never rewritten.
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM messages WHERE session_id=? ORDER BY id',
                            (ready['session_id'],)).fetchall()
    if not rows:
        raise ValueError('source session contains no messages')
    messages, raw_rows, ids = [], [], []
    for row in rows:
        raw = dict(row)
        raw_rows.append({key: {'encoding': 'base64', 'bytes': base64.b64encode(value).decode('ascii')}
                         if isinstance(value, bytes) else value for key, value in raw.items()})
        if raw.get('_compressed_summary') or raw.get('compacted'):
            raise ValueError('compressed source messages cannot prove verbatim preservation')
        role = raw['role']
        if role not in {'system', 'user', 'assistant', 'tool'}:
            raise ValueError(f'unsupported source role: {role}')
        message = {'role': role, 'content': raw.get('content') or ''}
        if role == 'assistant':
            reasoning = raw.get('reasoning_content') or ''
            mirror = raw.get('reasoning') or ''
            if mirror and mirror != reasoning:
                raise ValueError('source reasoning fields disagree; select a fully captured source')
            message['reasoning_content'] = reasoning
            if raw.get('tool_calls'):
                calls = json.loads(raw['tool_calls'])
                for call in calls:
                    if call.get('type') != 'function' or not isinstance(call.get('function'), dict):
                        raise ValueError('invalid native function call envelope')
                    if call['function'].get('name') not in {'arc_action'}:
                        raise ValueError('source contains an unsupported native tool')
                    args = call['function']['arguments']
                    json.loads(args) if isinstance(args, str) else args
                message['tool_calls'] = calls
        if role == 'tool':
            message.update(name=raw['tool_name'], tool_call_id=raw['tool_call_id'])
        messages.append(message)
        ids.append(raw['id'])
    calls_path = worker / 'episode/calls.jsonl'
    calls_bytes = calls_path.read_bytes()
    runtime_calls = [json.loads(line) for line in calls_bytes.splitlines() if line.strip()]
    native_lines = []
    for message in messages:
        for call in message.get('tool_calls', []):
            function = call['function']
            if function['name'] == 'arc_action':
                args = function['arguments']
                args = json.loads(args) if isinstance(args, str) else args
                native_lines.append(args['line'])
    lines = [row['request']['line'] for row in runtime_calls]
    if native_lines != lines:
        raise ValueError('native messages and actual runtime calls disagree')
    task = json.loads((worker / 'challenge.json').read_text())
    return {'schema': 'arc-span-source/v1', 'task_id': freeze['task_id'],
            'source_model': ready['source_model'], 'source_origin': 'captured_native',
            'ready': ready, 'ready_sha256': digest(ready_bytes),
            'freeze_sha256': digest(freeze_bytes), 'source_database_sha256': digest(database.read_bytes()),
            'calls_sha256': digest(calls_bytes), 'task': task, 'messages': messages,
            'message_ids': ids, 'raw_database_rows': raw_rows,
            'runtime_calls': runtime_calls, 'tool_lines': lines}
