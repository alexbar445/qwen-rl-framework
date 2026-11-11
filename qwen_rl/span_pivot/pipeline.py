"""Preserve a captured A, execute B tools, and gate an offline training pair."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .source import read_verified_source, digest
from .rewrite import rewrite_observed_facts
from .replay import run_replay, compare_replays, trusted_score
from .events import deduplicate_replayed
from .grid_rows import rewrite_grid_rows


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def native_lines(messages):
    result = []
    for message in messages:
        for call in message.get('tool_calls', []):
            function = call['function']
            if function['name'] == 'arc_action':
                args = function['arguments']
                args = json.loads(args) if isinstance(args, str) else args
                result.append(args['line'])
    return result


def token_cost(messages, tokenizer):
    generated, observations, prefills = 0, 0, 0
    for index, message in enumerate(messages):
        if message['role'] == 'assistant':
            fields = [message.get('reasoning_content') or '', message.get('content') or '']
            for call in message.get('tool_calls', []):
                arguments = call['function']['arguments']
                fields += [call['function']['name'], arguments if isinstance(arguments, str)
                           else json.dumps(arguments, ensure_ascii=False)]
            generated += sum(len(tokenizer.encode(value, add_special_tokens=False)) for value in fields)
            prefix = tokenizer.apply_chat_template(messages[:index], tokenize=False,
                add_generation_prompt=True, enable_thinking=True) if index else ''
            prefills += len(tokenizer.encode(prefix, add_special_tokens=False))
        elif message['role'] == 'tool':
            observations += len(tokenizer.encode(message['content'], add_special_tokens=False))
    return {'assistant_tokens': generated, 'observation_tokens': observations,
            'repeated_prefill_tokens': prefills, 'weighted_token_cost': 4 * generated + observations,
            'basis': 'target_tokenizer_offline_counts_not_provider_usage',
            'model_decode_seconds': None}


def run_pilot(*, source_ready: Path, message_id: int, source_text_sha256: str,
              insert_at: int, replace_until: int, witness_call_index: int,
              solutions: Path, tokenizer: Path, output: Path,
              strategy: str = 'observed-facts') -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = read_verified_source(source_ready)
    write(output / 'original-source.json', source)
    if strategy == 'grid-rows':
        candidate = rewrite_grid_rows(source, message_id=message_id,
            expected_text_sha256=source_text_sha256)
    elif strategy == 'observed-facts':
        candidate = rewrite_observed_facts(source, message_id=message_id,
            expected_text_sha256=source_text_sha256, insert_at=insert_at,
            replace_until=replace_until, observation_call_index=witness_call_index)
        candidate['span_message_count'] = 3
        candidate['insertions'] = [{'message_index': candidate['inserted_message_index'] + 1,
            'call_index': candidate['inserted_call_index']}]
    else:
        raise ValueError('unsupported rewrite strategy')
    write(output / 'edit-plan.json', {k: v for k, v in candidate.items() if k != 'messages'})
    source_index = source['message_ids'].index(message_id)
    original_text = source['messages'][source_index]['reasoning_content']
    (output / 'A-span.txt').write_text(original_text, encoding='utf-8')
    original = run_replay(source['task'], source['task_id'], source['tool_lines'], output / 'A-session')
    rewritten = run_replay(source['task'], source['task_id'], native_lines(candidate['messages']), output / 'B-session')
    write(output / 'A-replay.json', original); write(output / 'B-replay.json', rewritten)
    # Store actual predictions before the trusted parent-side gold read.
    write(output / 'predictions-frozen.json', {'A': original['submissions'], 'B': rewritten['submissions']})
    source_match = len(original['calls']) == len(source['runtime_calls']) and all(
        actual['response'] == expected['response']
        for actual, expected in zip(original['calls'], source['runtime_calls']))
    messages = candidate['messages']
    for insertion in candidate['insertions']:
        tool = messages[insertion['message_index']]
        actual = rewritten['calls'][insertion['call_index']]
        if strategy == 'grid-rows':
            _, train_index, side = insertion['source'].split('/')
            grid = source['task']['train'][int(train_index)][side]
            if actual['response'].get('status') != 'ok' or actual['response'].get('truncated') is not False or actual['response'].get('data') != grid:
                raise ValueError('actual inspection does not fully witness the removed row facts')
        tool['content'] = json.dumps(actual['response'], ensure_ascii=False)
    # Untagged rechecks are retained. The proof engine remains available for
    # separately proposed retry-only events, with actual runtime receipts.
    dedup = deduplicate_replayed(messages, rewritten)
    write(output / 'dedup-audit.json', {k: v for k, v in dedup.items() if k != 'messages'})
    messages = dedup['messages']
    insertion_count = len(candidate['insertions'])
    comparison = compare_replays(original, rewritten,
        pivot_end_indices=(candidate['inserted_call_index'], candidate['inserted_call_index'] + insertion_count))
    entry_index = candidate['inserted_call_index']
    entry_state = original['calls'][entry_index - 1]['semantic_state_after_sha256'] if entry_index else original['initial_state_sha256']
    comparison['entry_state_match'] = (entry_state
        == rewritten['calls'][entry_index + insertion_count - 1]['semantic_state_after_sha256'])
    write(output / 'rewritten-messages.json', messages)
    span_count = candidate['span_message_count']
    span = messages[source_index:source_index + span_count]
    write(output / 'B-span.json', span)
    (output / 'B-span-readable.txt').write_text('\n\n'.join(
        '[assistant reasoning]\n' + m['reasoning_content'] + '\n[tool calls]\n' + json.dumps(m.get('tool_calls', []), ensure_ascii=False)
        if m['role'] == 'assistant' else '[actual tool result]\n' + m['content'] for m in span), encoding='utf-8')
    scores = {'A': trusted_score(original['submissions'], source['task_id'], solutions),
              'B': trusted_score(rewritten['submissions'], source['task_id'], solutions)}
    write(output / 'gold-score.json', scores)
    from transformers import AutoTokenizer
    # Keep the shipped Qwen pretokenizer; Mistral's regex patch is unrelated.
    model_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True,
                                                  fix_mistral_regex=False)
    costs = {'A': token_cost(source['messages'], model_tokenizer),
             'B': token_cost(messages, model_tokenizer)}
    for label, replay in (('A', original), ('B', rewritten)):
        costs[label]['tool_seconds'] = sum(c['elapsed_seconds'] for c in replay['calls'])
    kept = candidate['preserved_reasoning_characters']
    expected_retained = original_text
    for edit in reversed(candidate['edits']):
        expected_retained = expected_retained[:edit['start']] + expected_retained[edit['end']:]
    verbatim = ''.join(m['reasoning_content'] for m in span if m['role'] == 'assistant') == expected_retained
    outside_unchanged = (messages[:source_index] == source['messages'][:source_index]
        and messages[source_index + span_count:] == source['messages'][source_index + 1:])
    inserted_indices = {item['call_index'] for item in candidate['insertions']}
    original_calls_unchanged = ([line for i, line in enumerate(native_lines(messages))
        if i not in inserted_indices] == source['tool_lines'])
    reasons = []
    if not (verbatim and outside_unchanged and original_calls_unchanged):
        reasons.append('source_reasoning_or_original_flow_changed')
    if not source_match: reasons.append('source_runtime_responses_changed')
    if not comparison['equivalent']: reasons.append('replay_or_suffix_state_mismatch')
    exact = all(s.get('exact') is True for s in scores.values())
    if not exact: reasons.append('gold_exact_failed')
    if costs['B']['weighted_token_cost'] >= costs['A']['weighted_token_cost']:
        reasons.append('no_weighted_token_cost_improvement')
    # This compiler does not measure source-model likelihood or live decode
    # time; those are separate proof levels and cannot be filled with guesses.
    reasons.append('live_decode_and_distribution_not_measured')
    report = {'schema': 'arc-span-pilot/v1', 'status': 'rejected' if reasons else 'verified',
        'rewrite_strategy': strategy,
        'task_id': source['task_id'], 'source_model': source['source_model'],
        'replay_equivalent': comparison['equivalent'], 'source_responses_unchanged': source_match,
        'exact': exact, 'comparison': comparison, 'reasoning_preservation': {
            'original_characters': len(original_text), 'retained_characters': kept,
            'retained_fraction': kept / len(original_text),
            'source_text_sha256': digest(original_text),
            'noncomputational_text_verbatim': verbatim, 'hypothesis_order_verbatim': verbatim,
            'outside_pivot_messages_unchanged': outside_unchanged,
            'original_action_sequence_unchanged': original_calls_unchanged,
            'summary_generated': False, 'original_source_saved': True,
            'posterior_distribution_equivalence': 'unmeasured'},
        'cost': costs, 'rejection_reasons': reasons,
        'training_eligible': False, 'duplicate_rechecks_removed': 0}
    report['tool_output_density'] = {'format': 'actual_unmodified_pinned_toolkit',
        'future_dense_format_savings': 'unmeasured_and_not_counted',
        'inserted_observations': [{'source': insertion.get('source'),
            'response_characters': len(messages[insertion['message_index']]['content']),
            'observation_tokens': len(model_tokenizer.encode(messages[insertion['message_index']]['content'], add_special_tokens=False))}
            for insertion in candidate['insertions']]}
    import transformers
    report['tokenizer'] = {'path': str(tokenizer.resolve()),
        'transformers_version': transformers.__version__,
        'tokenizer_json_sha256': digest((tokenizer / 'tokenizer.json').read_bytes()),
        'chat_template_sha256': digest(model_tokenizer.chat_template)}
    report['pivot_tokens'] = {'A': len(model_tokenizer.encode(original_text, add_special_tokens=False)),
        'B_reasoning_blocks': [len(model_tokenizer.encode(m['reasoning_content'], add_special_tokens=False))
            for m in span
            if m['role'] == 'assistant']}
    write(output / 'report.json', report)
    return report
