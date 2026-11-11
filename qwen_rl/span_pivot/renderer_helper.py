"""Fresh-process use of the actual upstream native observation renderer."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def fingerprint(root: Path) -> dict:
    files = sorted((root / 'arc_toolkit').rglob('*.py'))
    files += sorted((root / 'integrations/hermes_plugin').rglob('*.py'))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    before = fingerprint(root)
    sys.path.insert(0, str(root))
    from arc_toolkit import model_observation, common
    from arc_toolkit.observation_codec import decode_observation_v2
    if not Path(model_observation.__file__).resolve().is_relative_to(root):
        raise ValueError('renderer imported outside the selected upstream root')
    if common.VERSION != '0.20.0' or model_observation.OBSERVATION_SCHEMA != 'arc-model-observation-v2':
        raise ValueError('expected the updated v0.20/v2 renderer')
    data = json.loads(args.input.read_text())
    result = {'renderer': {'root':str(root), 'tool_version':common.VERSION,
                          'observation_schema':model_observation.OBSERVATION_SCHEMA}, 'branches':{}}
    for label, branch in data.items():
        messages = copy.deepcopy(branch['messages'])
        v1_messages = copy.deepcopy(branch['messages'])
        replay = branch['replay']
        actual_by_id, source_by_id, cursor, checks = {}, {}, 0, []
        for index, message in enumerate(messages):
            for call in message.get('tool_calls', []):
                if call['id'] in actual_by_id:
                    raise ValueError('duplicate native call ID')
                arguments = call['function']['arguments']
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                actual = replay['calls'][cursor]
                if call['function']['name'] != 'arc_action' or arguments['line'] != actual['line']:
                    raise ValueError('native command differs from the real replay')
                actual_by_id[call['id']] = actual
                source_by_id[call['id']] = cursor
                cursor += 1
            if message['role'] != 'tool':
                continue
            actual = actual_by_id[message['tool_call_id']]
            full = actual['response']
            if json.loads(message['content']) != full:
                raise ValueError('source observation differs from the actual full replay response')
            original_full_sha = hashlib.sha256(json.dumps(full,sort_keys=True).encode()).hexdigest()
            text = model_observation.serialize_response(full)
            projected_v1 = model_observation.project_source_response(full, model_observation.V1_OBSERVATION_SCHEMA)
            # Catalog schema is the presentation marker, intentionally v2.
            # All callable definitions and actual runtime version still have
            # to match the v1 audit projection exactly.
            expected_decoded = copy.deepcopy(projected_v1)
            if 'catalog' in expected_decoded:
                expected_decoded['catalog']['model_observation_schema'] = model_observation.OBSERVATION_SCHEMA
            if decode_observation_v2(json.loads(text)) != expected_decoded:
                raise ValueError('v2 decoding changed the source audit projection')
            if hashlib.sha256(json.dumps(full,sort_keys=True).encode()).hexdigest() != original_full_sha:
                raise ValueError('renderer changed a full audit response')
            message['content'] = text
            v1_messages[index]['content'] = json.dumps(projected_v1,ensure_ascii=False,sort_keys=True,separators=(',',':'))
            checks.append({'call_index':source_by_id[message['tool_call_id']], 'type':full.get('type'),
                'audit_sha256':original_full_sha, 'roundtrip_equal':True,
                'v2_text_sha256':hashlib.sha256(text.encode()).hexdigest()})
        if cursor != len(replay['calls']) or len(checks) != cursor:
            raise ValueError('not every actual action had a matching native tool message')
        result['branches'][label] = {'messages':messages, 'v1_messages':v1_messages, 'checks':checks}
    after = fingerprint(root)
    if before != after:
        raise ValueError('upstream renderer implementation changed during the test')
    result['renderer'].update(implementation_stable=True,files_sha256=before,
        fingerprint_sha256=hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
