"""Re-test a frozen row-rewrite with the current upstream native renderer.

Full tool execution uses the historical runtime. A separate process renders
both branches with v0.20; raw source archives and old experiments stay intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .events import deduplicate_replayed
from .grid_rows import rewrite_grid_rows
from .pipeline import native_lines, token_cost, write
from .replay import compare_replays, run_replay, trusted_score
from .source import digest, read_verified_source


def field_cost(messages, tokenizer):
    # No chat-template/prefill work is needed for this v1 sizing control.
    generated = observations = 0
    for message in messages:
        if message['role'] == 'assistant':
            fields = [message.get('reasoning_content') or '', message.get('content') or '']
            for call in message.get('tool_calls', []):
                arguments = call['function']['arguments']
                fields += [call['function']['name'], arguments if isinstance(arguments,str)
                           else json.dumps(arguments,ensure_ascii=False)]
            generated += sum(len(tokenizer.encode(field,add_special_tokens=False)) for field in fields)
        elif message['role'] == 'tool':
            observations += len(tokenizer.encode(message['content'],add_special_tokens=False))
    return {'assistant_tokens':generated,'observation_tokens':observations,
            'weighted_token_cost':4*generated+observations,
            'repeated_prefill_tokens':None,'model_decode_seconds':None,
            'basis':'target_tokenizer_offline_fields_not_provider_usage'}


def run_test(*, baseline:Path, source_ready:Path, renderer_root:Path,
             tokenizer:Path, solutions:Path, output:Path) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    source = read_verified_source(source_ready)
    previous_source = json.loads((baseline/'original-source.json').read_text())
    if source != previous_source:
        raise ValueError('live frozen source changed since the baseline experiment')
    write(output/'original-source.json',source)
    plan = json.loads((baseline/'edit-plan.json').read_text())
    text = source['messages'][source['message_ids'].index(plan['pivot_source_message_id'])]['reasoning_content']
    candidate = rewrite_grid_rows(source,message_id=plan['pivot_source_message_id'],expected_text_sha256=digest(text))
    a = run_replay(source['task'],source['task_id'],source['tool_lines'],output/'A-session')
    b = run_replay(source['task'],source['task_id'],native_lines(candidate['messages']),output/'B-session')
    write(output/'A-replay.json',a);write(output/'B-replay.json',b)
    if not a['execution_complete'] or not b['execution_complete']:
        raise ValueError('incomplete fresh replay')
    source_match = all(actual['response']==expected['response'] for actual,expected in zip(a['calls'],source['runtime_calls'])) and len(a['calls'])==len(source['runtime_calls'])
    for insertion in candidate['insertions']:
        actual = b['calls'][insertion['call_index']]['response']
        _, train_index, side = insertion['source'].split('/')
        if actual.get('truncated') is not False or actual.get('data') != source['task']['train'][int(train_index)][side]:
            raise ValueError('new inspection failed to witness every row fact')
        candidate['messages'][insertion['message_index']]['content'] = json.dumps(actual,ensure_ascii=False)
    raw_b = deduplicate_replayed(candidate['messages'],b)['messages']
    if raw_b != json.loads((baseline/'rewritten-messages.json').read_text()):
        raise ValueError('raw B differs from the baseline; output-only comparison is invalid')
    comparison = compare_replays(a,b,pivot_end_indices=(candidate['inserted_call_index'],candidate['inserted_call_index']+len(candidate['insertions'])))
    if not source_match or not comparison['equivalent']:
        raise ValueError('execution or original-source responses changed')
    write(output/'predictions-frozen.json',{'A':a['submissions'],'B':b['submissions']})
    scores = {label:trusted_score(replay['submissions'],source['task_id'],solutions) for label,replay in (('A',a),('B',b))}
    write(output/'gold-score.json',scores)
    write(output/'render-input.json',{'A':{'messages':source['messages'],'replay':a},'B':{'messages':raw_b,'replay':b}})
    script = renderer_root/'scripts/arc-toolkit-dependency.py'
    dependency = subprocess.run([sys.executable,str(script),'sync',str(output/'renderer-dependency.json'),
        '--upstream',str(renderer_root)],capture_output=True,text=True,check=True)
    (output/'dependency-sync.txt').write_text(dependency.stdout+dependency.stderr)
    helper = Path(__file__).with_name('renderer_helper.py')
    process = subprocess.run([sys.executable,str(helper),'--root',str(renderer_root),
        '--input',str(output/'render-input.json'),'--output',str(output/'rendered.json')],
        capture_output=True,text=True,timeout=120,check=True)
    (output/'renderer-process.txt').write_text(process.stdout+process.stderr)
    rendered = json.loads((output/'rendered.json').read_text())
    from transformers import AutoTokenizer
    model_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer),local_files_only=True,fix_mistral_regex=False)
    branches = rendered['branches']
    costs = {};v1_costs={}
    for label in ('A','B'):
        messages = branches[label]['messages']
        costs[label]=token_cost(messages,model_tokenizer)
        v1_costs[label]=field_cost(branches[label]['v1_messages'],model_tokenizer)
        write(output/f'{label}-v2-messages.json',messages)
        write(output/f'{label}-v1-sizing-control.json',branches[label]['v1_messages'])
    original_assistants = [[m for m in messages if m['role']=='assistant'] for messages in (source['messages'],raw_b)]
    assistant_unchanged = all([m for m in branches[label]['messages'] if m['role']=='assistant']==expected
        for label,expected in zip(('A','B'),original_assistants))
    pivot_index=candidate['inserted_message_index']
    common_prefix=branches['A']['messages'][:pivot_index]==branches['B']['messages'][:pivot_index]
    span=branches['B']['messages'][pivot_index:pivot_index+candidate['span_message_count']]
    write(output/'B-span-v2.json',span)
    (output/'B-span-v2-readable.txt').write_text('\n\n'.join(
        '[assistant reasoning]\n'+m['reasoning_content'] if m['role']=='assistant'
        else '[actual v2 tool observation]\n'+m['content'] for m in span),encoding='utf-8')
    inserted = []
    for insertion in candidate['insertions']:
        index=insertion['message_index']
        inserted.append({'source':insertion['source'],
            'full_response_tokens':len(model_tokenizer.encode(raw_b[index]['content'],add_special_tokens=False)),
            'v1_projection_tokens':len(model_tokenizer.encode(branches['B']['v1_messages'][index]['content'],add_special_tokens=False)),
            'v2_observation_tokens':len(model_tokenizer.encode(branches['B']['messages'][index]['content'],add_special_tokens=False))})
    exact=all(score['exact'] for score in scores.values())
    saved=costs['A']['weighted_token_cost']-costs['B']['weighted_token_cost']
    report={'schema':'arc-span-v020-retest/v1','task_id':source['task_id'],
        'construction':'historical_execution_reobserved_with_current_native_renderer',
        'execution_toolkit':a['toolkit'],'renderer':rendered['renderer'],
        'baseline_report_sha256':digest((baseline/'report.json').read_bytes()),
        'source_unchanged':True,'raw_rewrite_identical_to_baseline':True,
        'assistant_messages_unchanged_by_renderer':assistant_unchanged,'common_prefix_equal':common_prefix,
        'replay_equivalent':comparison['equivalent'],'comparison':comparison,'test_exact':exact,
        'rendered_responses_roundtrip_equal':all(check['roundtrip_equal'] for label in ('A','B') for check in branches[label]['checks']),
        'rendered_response_counts':{label:len(branches[label]['checks']) for label in ('A','B')},
        'old_full_response_costs':json.loads((baseline/'report.json').read_text())['cost'],
        'v1_projection_sizing_control':v1_costs,'current_v2_costs':costs,'inserted_observations':inserted,
        'weighted_tokens_saved':saved,'weighted_cost_reduction_fraction':saved/costs['A']['weighted_token_cost'],
        'token_proxy_improved':saved>0,'training_eligible':False,
        'unmeasured':['native_model_continuation','distribution_equivalence','model_decode_seconds','end_to_end_speedup'],
        'prompt_scope':'source_prompt_preserved_for_offline_sizing_only_current_v2_injection_not_applied',
        'tokenizer_json_sha256':digest((tokenizer/'tokenizer.json').read_bytes())}
    if not (assistant_unchanged and common_prefix and exact):
        raise ValueError('source reasoning, common prefix or exact answer failed validation')
    write(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','source-ready','renderer-root','tokenizer','solutions','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    result=run_test(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ('replay_equivalent','test_exact','inserted_observations',
        'current_v2_costs','weighted_tokens_saved','weighted_cost_reduction_fraction','training_eligible')},indent=2))
