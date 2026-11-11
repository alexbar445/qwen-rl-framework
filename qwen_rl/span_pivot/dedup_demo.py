"""Real-tool mechanism fixture, explicitly not a model-authored trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .events import deduplicate_replayed
from .pipeline import native_lines, write
from .replay import compare_replays, run_replay


def run_demo(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    task = {'train': [{'input': [[0, 1]], 'output': [[0, 1]]}], 'test': [{'input': [[0, 1]]}]}
    lines = ["inspect(source='test/0/input')", "inspect(source='test/0/input')", "submit(source='test/0/input')"]
    original = run_replay(task, 'dedup-mechanism', lines, output / 'original-session')
    write(output / 'original-replay.json', original)
    messages = []
    for index, call in enumerate(original['calls']):
        call_id = f'fixture-call-{index}'
        messages += [{'role': 'assistant', 'content': '', 'reasoning_content':
            ('Wait, let me retry.' if index == 1 else 'Inspect the input.' if index == 0 else 'Submit the input.'),
            'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': 'arc_action',
                'arguments': json.dumps({'line': call['line']})}}]},
            {'role': 'tool', 'name': 'arc_action', 'tool_call_id': call_id,
             'content': json.dumps(call['response'])}]
    write(output / 'constructed-fixture-messages.json', messages)
    dedup = deduplicate_replayed(messages, original, retry_message_indices=(2,))
    write(output / 'dedup-audit.json', dedup)
    rewritten = run_replay(task, 'dedup-mechanism', native_lines(dedup['messages']), output / 'dedup-session')
    write(output / 'dedup-replay.json', rewritten)
    report = {'schema': 'arc-span-dedup-demo/v1',
        'source_origin': 'constructed_mechanism_fixture_not_model_data',
        'tool_observations': 'actual_Session_execute',
        'original_tool_calls': len(lines), 'retained_tool_calls': len(rewritten['calls']),
        'deleted_retry_turns': len(dedup['deletions']), 'training_eligible': False,
        'comparison': compare_replays(original, rewritten)}
    write(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    print(json.dumps(run_demo(parser.parse_args().output), indent=2))
