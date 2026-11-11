"""Replace witnessed row rereading, preserving all non-factual text verbatim.

This uses the shipped read-only inspect control. It does not invent a dense
row-runs endpoint or change the production toolkit's output representation.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re


_HEADINGS = re.compile(r'Let me re-read train(?P<input_index>\d+) input rows (?P<first>\d+)-(?P<last>\d+):\n|Output rows \(train(?P<output_index>\d+) output\):\n')
_ROW = re.compile(r'row(?P<row>\d+): `(?P<vector>\[[^`]+\])` → (?P<description>[^\n]*)(?:\n|$)')
_SAME = re.compile(r'row(?P<row>\d+): same as row(?P<other>\d+)\.(?:\n|$)')
_CLAUSE = re.compile(r'(?P<label>cols?|cols)\s*(?P<first>\d+)(?:-(?P<last>\d+))?(?:=(?P<color>\d+))?')


def _claims(description: str, grid_row: list[int]) -> tuple[str, str]:
    """Validate literal cell/range claims; return facts and retained annotation."""
    annotation = ''
    match = re.search(r'\s*(\([^()]+\))\s*$', description)
    if match:
        annotation = description[match.start():]
        description = description[:match.start()]
    facts = description
    # A leading scope ("cols16-21:") names the area being discussed; all
    # actual coordinate claims still have to be independently grounded.
    scope = re.match(r'cols(?P<first>\d+)-(?P<last>\d+):\s*', description)
    if scope:
        a, b = int(scope['first']), int(scope['last'])
        if not 0 <= a <= b < len(grid_row):
            raise ValueError('row scope is outside the grid')
        description = description[scope.end():]
    clauses = [part.strip().rstrip('.') for part in re.split(r'[,;]', description)]
    if not clauses or any(not clause for clause in clauses):
        raise ValueError('unsupported row description')
    for clause in clauses:
        claim = _CLAUSE.fullmatch(clause)
        if claim is None:
            raise ValueError(f'unsupported row description: {clause}')
        first = int(claim['first']); last = int(claim['last'] or claim['first'])
        if not 0 <= first <= last < len(grid_row):
            raise ValueError('row coordinate is outside the grid')
        # The captured notation omits =1 for color-1 border cells. Other
        # colors must be explicit; ambiguous descriptions fail closed.
        color = int(claim['color'] or '1')
        if any(grid_row[col] != color for col in range(first, last + 1)):
            raise ValueError('row coordinate/color claim disagrees with source')
    return facts, annotation


def rewrite_grid_rows(source: dict, *, message_id: int, expected_text_sha256: str) -> dict:
    index = source['message_ids'].index(message_id)
    original = source['messages'][index]
    text = original.get('reasoning_content')
    if original['role'] != 'assistant' or not isinstance(text, str):
        raise ValueError('pivot must be a captured assistant reasoning message')
    if hashlib.sha256(text.encode()).hexdigest() != expected_text_sha256:
        raise ValueError('pivot source text hash mismatch')
    before = sum(1 for m in source['messages'][:index] for c in m.get('tool_calls', [])
                 if c['function']['name'] == 'arc_action')
    blocks, edits = [], []
    for heading in _HEADINGS.finditer(text):
        train_index = int(heading['input_index'] or heading['output_index'])
        side = 'input' if heading['input_index'] is not None else 'output'
        grid = source['task']['train'][train_index][side]
        alias = f'train/{train_index}/{side}'
        start = heading.end()
        end = text.find('\n\n', start)
        if end == -1:
            end = len(text)
        else:
            end += 1  # Consume the final row's newline, leaving the separator.
        cursor, rows = start, []
        while cursor < end:
            row_match = _ROW.match(text, cursor)
            same_match = _SAME.match(text, cursor)
            if row_match:
                row = int(row_match['row'])
                if not 0 <= row < len(grid):
                    raise ValueError('row is outside the grid')
                vector = ast.literal_eval(row_match['vector'])
                if (not isinstance(vector, list) or any(type(cell) is not int for cell in vector)
                        or vector != grid[row]):
                    raise ValueError('copied row disagrees with source')
                facts, annotation = _claims(row_match['description'], grid[row])
                if annotation:
                    # Preserve the original row label so the conceptual
                    # annotation remains attached to the same row.
                    a = row_match.start('vector') - 1  # Opening backtick.
                    b = row_match.start('description') + len(facts)
                else:
                    a, b = row_match.span()
                cursor = row_match.end()
            elif same_match:
                row, other = int(same_match['row']), int(same_match['other'])
                if not (0 <= row < len(grid) and other in rows and grid[row] == grid[other]):
                    raise ValueError('same-row claim lacks an earlier exact witness')
                a, b = same_match.span(); cursor = b
            else:
                raise ValueError('unsupported text inside a row rereading block')
            if row in rows:
                raise ValueError('duplicate row in a rereading block')
            rows.append(row)
            edits.append({'start': a, 'end': b, 'original': text[a:b],
                'sha256': hashlib.sha256(text[a:b].encode()).hexdigest(),
                'kind': 'row_lookup', 'source': alias, 'row': row})
        if heading['first'] is not None and rows != list(range(int(heading['first']), int(heading['last']) + 1)):
            raise ValueError('heading row range disagrees with its body')
        if not rows:
            raise ValueError('empty row rereading block')
        blocks.append({'at': start, 'source': alias, 'rows': rows,
                       'line': f'inspect(source="{alias}", limit={max(len(grid), len(grid[0]))})'})
    if not blocks:
        raise ValueError('no supported grid row rereading blocks')
    prior_ids = {c['id'] for m in source['messages'] for c in m.get('tool_calls', [])}
    span, insertions = [], []
    boundaries = [0] + [block['at'] for block in blocks] + [len(text)]
    for part_index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        content, cursor = [], left
        for edit in edits:
            if left <= edit['start'] < right:
                if edit['end'] > right:
                    raise ValueError('edit crosses a tool boundary')
                content.append(text[cursor:edit['start']]); cursor = edit['end']
        content.append(text[cursor:right])
        message = copy.deepcopy(original)
        message['reasoning_content'] = ''.join(content)
        if part_index < len(blocks):
            block = blocks[part_index]
            call_id = f'span-grid-inspect-{part_index}'
            if call_id in prior_ids:
                raise ValueError('inserted tool call ID collides with the source')
            message['tool_calls'] = [{'id': call_id, 'type': 'function', 'function': {
                'name': 'arc_action', 'arguments': json.dumps({'line': block['line']})}}]
            # Visible content belongs to the original final assistant turn.
            message['content'] = ''
            span.append(message)
            insertions.append({**block, 'call_index': before + part_index,
                               'message_index': index + len(span)})
            span.append({'role': 'tool', 'name': 'arc_action', 'tool_call_id': call_id, 'content': None})
        else:
            span.append(message)  # Carries the original action and content.
    messages = copy.deepcopy(source['messages'])
    messages[index:index + 1] = span
    return {'messages': messages, 'pivot_source_message_id': message_id,
            'inserted_message_index': index, 'inserted_call_index': before,
            'insertions': insertions, 'span_message_count': len(span), 'edits': edits,
            'preserved_reasoning_characters': len(text) - sum(e['end']-e['start'] for e in edits),
            'original_reasoning_characters': len(text), 'reasoning_order_preserved': True,
            'generated_bridge_text': '', 'strategy': 'grid-rows',
            'observation_format': 'unmodified_pinned_toolkit_inspect'}
