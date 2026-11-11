"""Read-only terminal monitor. Poll metadata; never import a model or call generation."""
from __future__ import annotations
import asyncio
from collections import deque
from dataclasses import replace
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

import httpx
from .manifest import active_manifest_path, read_json
from .watch_metrics import (
    COMMITTED_COUNTERS,
    RollingCounterWindow,
    SampledActivityWindow,
    classify_worker_activity,
    classify_learner_activity,
)


class Rates:
    """Rolling wall-clock rate from monotonic counters within one process epoch."""
    def __init__(self, window=30.0):
        self.window = window
        self.series = {}

    def sample(self, key, epoch, count, now):
        if not epoch or type(count) not in (int, float) or not math.isfinite(count) or count < 0:
            self.series.pop(key, None)
            return None
        prior_epoch, points = self.series.get(key, (epoch, deque()))
        if prior_epoch != epoch or (points and count < points[-1][1]):
            points = deque()
        points.append((now, count))
        while len(points) > 2 and points[1][0] <= now-self.window:
            points.popleft()
        self.series[key] = (epoch, points)
        if len(points) < 2 or now <= points[0][0]:
            return None
        return (count-points[0][1])/(now-points[0][0])


def gpu_snapshot():
    if shutil.which('nvidia-smi') is None:
        return []
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True)
        rows = []
        for values in csv.reader(io.StringIO(result.stdout)):
            index, utilization, used, total = [v.strip() for v in values]
            rows.append(dict(index=int(index), utilization_percent=None if utilization in {'N/A','[N/A]'} else float(utilization),
                             used_mib=float(used), total_mib=float(total)))
        return rows
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def clean_text(value, secrets=()):
    value = str(value)
    for secret in secrets:
        if secret:
            value = value.replace(secret, '[REDACTED]')
    value = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', value)
    return ''.join(c for c in value if c in '\n\t' or ord(c) >= 32 and ord(c) != 127)


def _fmt(value, suffix='', digits=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return '—'
    return f'{value:,.{digits}f}{suffix}'


def _fmt_general(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return '—'
    return f'{value:.3g}'


_PHASE_LABELS = {
    'idle': '閒置', 'optimizer': '優化器更新', 'starting': '更新初始化',
    'pre_alignment': '更新前對齊', 'proximal': '近端策略評分',
    'loss_forward': '損失前向', 'loss_backward': '損失反向',
    'forward': '前向', 'backward': '反向',
    'gradient_clip': '梯度裁剪', 'optimizer_step': '優化器步進',
    'post_optimizer_validation': '更新後驗證', 'publication': '策略發布',
    'checkpoint': '完整狀態保存', 'worker_sync': 'Worker 策略同步',
    'awaiting_groups': '等待訓練組',
}


def _phase_label(value):
    if not isinstance(value, str) or not value:
        return '未知'
    prefix, separator, suffix = value.partition(':')
    if separator and prefix in {'compute', 'foreground'}:
        return _PHASE_LABELS.get(suffix, suffix)
    if separator and prefix == 'wait':
        return f"等待訓練組：{suffix}"
    if separator and prefix == 'wait_groups':
        return f"等待訓練組：{suffix}"
    if value == 'wait_sync':
        return '等待 Worker 策略同步'
    return _PHASE_LABELS.get(value, value)


def _activity_label(state):
    labels = {
        'inference_inflight': '推理中',
        'queued': '排隊中',
        'between_generations': 'episode引用保留',
        'not_ready': '未就緒/載入中',
        'unknown': '未知',
        'idle': '閒置',
    }
    if isinstance(state, str) and state.startswith('wait_dispatch:'):
        return f"派送等待[{state.split(':', 1)[1]}]"
    if isinstance(state, str) and state.startswith('wait:'):
        return f"等待訓練組[{state.split(':', 1)[1]}]"
    if isinstance(state, str) and state.startswith('wait_groups:'):
        return f"等組[{state.split(':', 1)[1]}]"
    if state == 'wait_sync':
        return '等待 Worker 同步'
    if isinstance(state, str) and state.startswith(('compute:', 'foreground:')):
        return _phase_label(state)
    return labels.get(state, str(state or '未知'))


def _committed_rate_text(record):
    record = record if isinstance(record, dict) else {}
    if not record.get('available'):
        observed = record.get('observed_seconds')
        window = record.get('window_seconds', 300)
        if record.get('unavailable_reason') == 'stale':
            return f"未知（Learner metadata 過期 {_fmt(record.get('last_sample_age_seconds'), '秒')}）"
        if record.get('unavailable_reason') == 'counter_unavailable':
            return '未知（提交計數缺失）'
        if isinstance(observed, (int, float)) and observed > 0:
            return f"未知（基线 {_fmt(observed, '秒', 0)}/{_fmt(window, '秒', 0)}）"
        age = record.get('last_sample_age_seconds')
        if age is not None:
            return f"未知（最近样本 {_fmt(age, '秒')} 前）"
        return '未知（等待差分基线）'
    rates = record.get('rates_per_hour') or {}
    elapsed = record.get('observed_seconds')
    window = record.get('window_seconds', 300)
    return (
        f"輸入 {_fmt(rates.get('input_tokens'), '/時', 0)} | "
        f"策略 {_fmt(rates.get('policy_tokens'), '/時', 0)} | "
        f"軌跡 {_fmt(rates.get('trajectories'), '/時', 0)} | "
        f"更新 {_fmt(rates.get('updates'), '/時', 1)} "
        f"({_fmt(elapsed, '秒', 0)}/{_fmt(window, '秒', 0)})"
    )


def _compact_committed_rate_text(record):
    record = record if isinstance(record, dict) else {}
    if not record.get('available'):
        if record.get('unavailable_reason') == 'stale':
            return f"未知／stale {_fmt(record.get('last_sample_age_seconds'), '秒')}"
        return '未知／等待差分基線'
    rates = record.get('rates_per_hour') or {}
    return (
        f"入 {_fmt(rates.get('input_tokens'), '', 0)} 策 {_fmt(rates.get('policy_tokens'), '', 0)} "
        f"軌 {_fmt(rates.get('trajectories'), '', 0)} 更 {_fmt(rates.get('updates'), '', 1)}/時 "
        f"({_fmt(record.get('observed_seconds'), '秒', 0)})"
    )


def _lifetime_rate_text(record):
    record = record if isinstance(record, dict) else {}
    rates = record.get('lifetime_rates_per_hour') or {}
    values = [rates.get(name) for name in COMMITTED_COUNTERS]
    if not any(value is not None for value in values):
        return '未知'
    return (
        f"輸入 {_fmt(rates.get('input_tokens'), '/時', 0)} | "
        f"策略 {_fmt(rates.get('policy_tokens'), '/時', 0)} | "
        f"軌跡 {_fmt(rates.get('trajectories'), '/時', 0)} | "
        f"更新 {_fmt(rates.get('updates'), '/時', 1)}"
    )


def _totals_text(totals):
    totals = totals if isinstance(totals, dict) else {}
    return (
        f"輸入 {_fmt(totals.get('input_tokens'), '', 0)} | "
        f"策略 {_fmt(totals.get('policy_tokens'), '', 0)} | "
        f"軌跡 {_fmt(totals.get('trajectories'), '', 0)} | "
        f"更新 {_fmt(totals.get('updates'), '', 0)}"
    )


def _stream_filter_guard_text(stream):
    stream = stream if isinstance(stream, dict) else {}

    def count(name):
        value = stream.get(name)
        return str(value) if type(value) is int and value >= 0 else '—'

    return (
        f"filtered {count('filtered_groups_total')} total | "
        f"no-progress guard {count('filtered_groups_since_update')}/"
        f"{count('max_filtered_groups_per_update')} per committed update"
    )


def _stream_filter_guard_compact_text(stream):
    stream = stream if isinstance(stream, dict) else {}

    def count(name):
        value = stream.get(name)
        return str(value) if type(value) is int and value >= 0 else '—'

    return (
        f"filtered {count('filtered_groups_total')} total | "
        f"no-progress guard {count('filtered_groups_since_update')}/"
        f"{count('max_filtered_groups_per_update')} per committed update"
    )


def _evaluation_comparison_text(value):
    if not isinstance(value, dict):
        return '未知'
    exact = _evaluation_outcome(value, 'native_exact', 'exact_solve_count', 'exact_known_count',
                                'unknown_exact_count', 'exact_rate')
    passed = _evaluation_outcome(value, 'native_pass', 'pass_count', 'pass_known_count',
                                 'unknown_pass_count', 'pass_rate')
    qualifier = '' if exact['complete'] else (
        f" (known rows only; whole-set exact rate unknown; unknown "
        f"{exact['unknown'] if exact['unknown'] is not None else '—'}/"
        f"{exact['total'] if exact['total'] is not None else '—'})"
    )
    return (f"v{value.get('policy_version', '—')} exact {exact['count']}/{exact['known']} "
            f"pass {passed['count']}/{passed['known']}{qualifier}")


def _evaluation_outcome(evaluation, metric_key, count_key, known_key, unknown_key, rate_key):
    """Separate a whole-task score from the rate over only known rows."""
    if not isinstance(evaluation, dict):
        return {'count': None, 'known': None, 'unknown': None, 'total': None, 'complete': False}
    source = evaluation.get('summary') if isinstance(evaluation.get('summary'), dict) else evaluation
    metric = source.get(metric_key, evaluation.get(metric_key))
    count = source.get(count_key)
    if type(count) is not int:
        count = source.get(f'{metric_key}_count', evaluation.get(f'{metric_key}_count'))
    if type(count) is not int and isinstance(metric, dict) and type(metric.get('count')) is int:
        count = metric['count']
    known = source.get(known_key)
    if type(known) is not int:
        known = source.get(f'{metric_key}_total', evaluation.get(f'{metric_key}_total'))
    if type(known) is not int and isinstance(metric, dict) and type(metric.get('total')) is int:
        known = metric['total']
    if type(known) is not int:
        known = evaluation.get(f'{metric_key}_total')
    total = source.get('task_count', source.get('tasks_total', evaluation.get('tasks_total', evaluation.get('total_tasks'))))
    if type(total) is not int and type(known) is int:
        total = known
    unknown = source.get(unknown_key)
    if type(unknown) is not int and type(total) is int and type(known) is int:
        unknown = max(0, total - known)
    rate = source.get(rate_key, evaluation.get(rate_key))
    done = evaluation.get('tasks_done')
    all_done = type(done) is not int or type(total) is not int or done >= total
    complete_flag = source.get('complete_valid_evaluation')
    complete = (
        type(count) is int and type(known) is int and type(total) is int
        and known == total and unknown == 0 and all_done
        and complete_flag is not False
        and (rate is None or (isinstance(rate, (int, float)) and not isinstance(rate, bool)
                              and math.isfinite(rate) and 0.0 <= rate <= 1.0))
    )
    return {'count': count, 'known': known, 'unknown': unknown, 'total': total,
            'complete': complete, 'rate': rate}


def _evaluation_summary_text(evaluation, *, live=None, age_seconds=None):
    if not isinstance(evaluation, dict) or not evaluation:
        return '未知／尚無獨立 ARC 評估狀態'
    phase = evaluation.get('phase', evaluation.get('current_phase', 'unknown'))
    phase_label = {
        'idle': '閒置', 'baseline_evaluation': 'SFT基線評估中',
        'milestone_evaluation': 'RL檢查點評估中', 'queued': '待評估',
        'baseline_pending': 'SFT基線待評估', 'baseline_retry_pending': 'SFT基線待重試',
        'milestone_pending': 'RL檢查點待評估', 'milestone_retry_pending': 'RL檢查點待重試',
        'manual_evaluation': '手動策略評估中', 'manual_pending': '手動策略待評估',
        'manual_retry_pending': '手動策略待重試',
        'running': '評估中', 'complete': '已完成', 'completed': '已完成',
    }.get(phase, str(phase or '未知'))
    version = evaluation.get('policy_version', evaluation.get('version'))
    version_text = f'v{version}' if type(version) is int else '版本未知'
    done = evaluation.get('tasks_done', evaluation.get('completed_tasks'))
    total = evaluation.get('tasks_total', evaluation.get('total_tasks'))
    progress = f'{done}/{total}' if type(done) is int and type(total) is int else '未知'
    exact = _evaluation_outcome(evaluation, 'native_exact', 'exact_solve_count', 'exact_known_count',
                                'unknown_exact_count', 'exact_rate')
    passed = _evaluation_outcome(evaluation, 'native_pass', 'pass_count', 'pass_known_count',
                                 'unknown_pass_count', 'pass_rate')
    def outcome_text(outcome):
        count, known, total = outcome.get('count'), outcome.get('known'), outcome.get('total')
        if outcome.get('complete'):
            rate = outcome.get('rate')
            if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate):
                rate = count / total if total else None
            return f'{count}/{total} ({rate * 100:.1f}%)' if rate is not None else f'{count}/{total}'
        return f'{count if count is not None else "—"}/{known if known is not None else "—"}'

    exact_text = outcome_text(exact)
    pass_text = outcome_text(passed)
    outcome_scope = '' if exact.get('complete') else (
        f" (known rows only; whole-set exact rate 未知; unknown "
        f"{exact.get('unknown') if exact.get('unknown') is not None else '—'}/"
        f"{exact.get('total') if exact.get('total') is not None else '—'})"
    )
    current_task = evaluation.get('current_task_id')
    task_phase = evaluation.get('current_task_phase')
    task_text = f' | 題目 {current_task}:{task_phase}' if current_task else (
        f' | 題目階段 {task_phase}' if task_phase else ''
    )
    freshness = '' if live is not False else f' | metadata過期 {_fmt(age_seconds, "秒")}'
    return (
        f'{phase_label} {version_text} | tasks {progress}{task_text} | '
        f'native exact {exact_text} / pass {pass_text}{outcome_scope}{freshness}'
    )


def _evaluation_progress_text(evaluation):
    if not isinstance(evaluation, dict) or not evaluation:
        return '評估進度未知'
    baseline = _evaluation_comparison_text(evaluation.get('compared_baseline'))
    previous = _evaluation_comparison_text(evaluation.get('compared_previous'))
    gains = evaluation.get('gains', evaluation.get('gain_count'))
    regressions = evaluation.get('regressions', evaluation.get('regression_count'))
    errors = evaluation.get('errors', evaluation.get('error_count'))
    if isinstance(errors, (dict, list)):
        errors = len(errors)
    next_milestone = evaluation.get('next_milestone_update_count', evaluation.get('next_milestone'))
    next_text = str(next_milestone) if type(next_milestone) is int else '未知'
    return (
        f'相較基線 {baseline} | 前次 {previous} | gains {gains if gains is not None else "未知"} '
        f'regressions {regressions if regressions is not None else "未知"} '
        f'errors {errors if type(errors) is int else "未知"} | 下個里程碑 {next_text}'
    )


def _evaluation_runtime_error_text(evaluation):
    if not isinstance(evaluation, dict):
        return None
    error = evaluation.get('runtime_error')
    if isinstance(error, dict):
        kind = error.get('type', 'EvaluationError')
        message = error.get('message', 'unknown error')
        return clean_text(f'{kind}: {message}')
    if isinstance(error, str) and error:
        return clean_text(error)
    return None


def _evaluation_task_table(evaluation, *, max_rows=12):
    from rich.table import Table

    rows = evaluation.get('result_rows') if isinstance(evaluation, dict) else None
    rows = list(rows) if isinstance(rows, list) else []
    partial = evaluation.get('partial_tasks') if isinstance(evaluation, dict) else None
    if isinstance(partial, dict):
        present = {row.get('task_id') for row in rows if isinstance(row, dict)}
        for task_id, task in partial.items():
            if task_id not in present:
                attempts = task.get('attempts') or []
                latest = attempts[-1] if attempts else {}
                rows.append({
                    'task_id': task_id, 'status': task.get('status', 'unknown'),
                    'episode_id': latest.get('episode_id'), 'exact': None, 'passed': None,
                    'error': latest.get('reason'),
                    'partial_artifact_refs': latest.get('artifact_refs', []),
                })
    title = f"Native ARC 評估題目結果 · 顯示 {min(len(rows), max_rows)}/{len(rows)}"
    table = Table(title=title)
    for name in ('task', '狀態', 'exact', 'pass', 'error', 'episode / artifact'):
        table.add_column(name, no_wrap=name != 'episode / artifact', overflow='ellipsis')
    status_labels = {
        'completed': '完成', 'in_progress': '執行中', 'retry_pending': '待重試',
        'pending': '待執行', 'partial': '部分產物', 'unknown': '未知',
    }
    for row in rows[:max_rows]:
        if not isinstance(row, dict):
            continue
        exact = row.get('exact') if type(row.get('exact')) is bool else None
        passed = row.get('passed') if type(row.get('passed')) is bool else None
        artifact = row.get('partial_artifact_refs') or row.get('result_ref') or '—'
        if isinstance(artifact, (list, dict)):
            artifact = json.dumps(artifact, ensure_ascii=False, sort_keys=True)
        episode = row.get('episode_id') or '—'
        table.add_row(
            str(row.get('task_id') or '—'),
            status_labels.get(row.get('status'), str(row.get('status') or '未知')),
            '是' if exact is True else '否' if exact is False else '未知',
            '是' if passed is True else '否' if passed is False else '未知',
            str(row.get('error') or '—'),
            f'{episode} / {artifact}',
        )
    if not rows:
        table.add_row('—', '尚無結果列', '未知', '未知', '未知', '未知')
    return table


def _quality_lines(quality):
    quality = quality if isinstance(quality, dict) else {}
    grad = quality.get('gradient_norm_before_clip', quality.get('gradient_norm'))
    learning_rate = quality.get('learning_rate')
    clip_ratio = quality.get('clip_ratio') or {}
    if isinstance(clip_ratio, dict) and clip_ratio:
        clip_text = ','.join(f"{key}={_fmt(value, '', 3)}" for key, value in sorted(clip_ratio.items()))
    else:
        clip_text = _fmt(quality.get('clip_fraction'))
    correction = quality.get('importance_correction') or {}
    cap = correction.get('cap', quality.get('importance_sampling_cap')) if isinstance(correction, dict) else None
    truncation = quality.get('importance_capped_fraction')
    if truncation is None and isinstance(correction, dict):
        truncation = correction.get('truncated_fraction')
    truncation_denominator = quality.get('importance_capped_fraction_denominator_fresh_policy_tokens')
    if truncation_denominator is None and isinstance(correction, dict):
        truncation_denominator = correction.get('fresh_policy_tokens')
    quality_prefix = '上次成功更新品質' if quality.get('available') is True else '更新品質未知'
    truncation_text = (
        f"校正上限 {_fmt_general(cap)} | 超限比例 {_fmt_general(truncation)} / "
        f"{_fmt(truncation_denominator, ' 個 fresh 策略 token', 0)}"
        if truncation is not None or cap is not None else '校正上限/超限比例 —'
    )
    line = (
        f"{quality_prefix}：損失 {_fmt_general(quality.get('loss'))} | "
        f"梯度 {_fmt_general(grad)} | 學習率 {_fmt_general(learning_rate)} | 裁剪 {clip_text}"
    )
    alignment = quality.get('alignment')
    if not isinstance(alignment, dict):
        return line, truncation_text, '更新前對齊未知'
    status = alignment.get('status', 'measured')
    if status == 'policy_lag':
        behavior = alignment.get('behavior_version', '—')
        learner_start = alignment.get('learner_start_version', '—')
        lag = alignment.get('policy_lag', '—')
        return line, truncation_text, f"更新前對齊：策略版本延遲 v{behavior}→v{learner_start}（lag {lag}），未測同策略數值差"
    metrics = ' '.join(
        f"{name}={_fmt(alignment.get(key), '', 4)}"
        for name, key in (('mean', 'mean_abs_error'), ('RMS', 'rms_error'), ('max', 'max_abs_error'))
        if alignment.get(key) is not None
    )
    scope = quality.get('alignment_scope')
    label = 'proximal前向對齊' if alignment.get('source') == 'trainer_proximal_scores_before_optimizer' else '更新前對齊'
    return line, truncation_text, f"{label}[{scope or '最新嘗試'}] {status}" + (f" {metrics}" if metrics else '')


def _bepa_summary(bepa):
    diagnostics = bepa.get('diagnostics') if isinstance(bepa, dict) else None
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    plans = diagnostics.get('plans') or {}
    cache = plans.get('cache_metadata') or {}
    blocked = {
        name: cache.get(key)
        for name, key in (
            ('age', 'blocked_age'), ('quota', 'blocked_quota'),
            ('pending', 'blocked_pending'), ('unknown', 'blocked_unknown'),
            ('unverified', 'blocked_unverified'), ('future', 'future_policy'),
            ('inactive', 'inactive'),
        )
    }
    blocked_total = (
        sum(value for value in blocked.values() if type(value) is int and value >= 0)
        if all(type(value) is int and value >= 0 for value in blocked.values()) else None
    )
    eligible = cache.get('metadata_eligible')
    mismatches = plans.get('unverified_or_mismatch')
    other_blocked = (
        blocked_total - sum(blocked[key] for key in ('age', 'quota', 'pending', 'unknown'))
        if blocked_total is not None else None
    )
    cache_text = (
        f"可claim候選 {_fmt(eligible, '', 0)}（metadata，來源未重驗） | "
        f"封鎖 {_fmt(blocked_total, '', 0)}：過期 {_fmt(blocked['age'], '', 0)} "
        f"額度 {_fmt(blocked['quota'], '', 0)} 預留 {_fmt(blocked['pending'], '', 0)} "
        f"用量未知 {_fmt(blocked['unknown'], '', 0)} 其他 {_fmt(other_blocked, '', 0)} | "
        f"無cache {_fmt(cache.get('absent'), '', 0)} 不匹配 {_fmt(mismatches, '', 0)}"
        if diagnostics else 'Cache候選/封鎖未知'
    )
    states = plans.get('states') or {}
    generation = plans.get('seed_generation') or {}
    plans_text = (
        f"Plan待引導 {_fmt(states.get('ready'), '', 0)} | Qwen引導中 {_fmt(states.get('seed_running'), '', 0)} | "
        f"已消費 {_fmt(states.get('consumed'), '', 0)} | 耗盡 {_fmt(states.get('exhausted'), '', 0)} | "
        f"失敗嘗試 {_fmt(generation.get('failed_attempts'), '', 0)} | "
        f"重試用盡 {_fmt(generation.get('attempt_cap_reached'), '', 0)} 題 | "
        f"每題嘗試上限 {_fmt(generation.get('attempt_cap'), '', 0)}"
    ) if diagnostics else 'Plan/seed結果未知'
    current = diagnostics.get('current_run') or {}
    reservations = current.get('reservations') or {}
    usage = plans.get('usage') or {}
    run_text = (
        f"replay已消費（記憶體） {_fmt(current.get('optimizer_consumed_replay_groups'), '', 0)} | "
        f"Cache全epoch預留 {_fmt(usage.get('pending_reserved'), '', 0)} / "
        f"提交未知 {_fmt(usage.get('pending_commit_unknown'), '', 0)} | "
        f"已觀測cohort ledger預留/提交/釋放/提交未知 "
        f"{_fmt(reservations.get('reserved'), '', 0)}/{_fmt(reservations.get('committed'), '', 0)}/"
        f"{_fmt(reservations.get('released'), '', 0)}/{_fmt(reservations.get('commit_unknown'), '', 0)} "
        f"(batches {_fmt(reservations.get('observed_batch_count'), '', 0)}; {reservations.get('coverage','unknown')}) | "
        f"待救援 {_fmt(current.get('held_groups'), '', 0)}"
    ) if diagnostics else '本run消費/預留未知'
    return diagnostics, cache_text, plans_text, run_text, blocked


def _short_count(value):
    if type(value) is not int:
        return '—'
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f}k"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _short_rate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return '—'
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f}k"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return _fmt_general(value)


def _sum_counts(values):
    if not isinstance(values, dict) or not values:
        return None
    if not all(type(value) is int and value >= 0 for value in values.values()):
        return None
    return sum(values.values())


def _compact_committed_rate_text(record):
    record = record if isinstance(record, dict) else {}
    if not record.get('available'):
        if record.get('unavailable_reason') == 'stale':
            return f"未知/過期{_fmt(record.get('last_sample_age_seconds'), '秒', 0)}"
        return '未知/等待基線'
    rates = record.get('rates_per_hour') or {}
    observed_value = record.get('observed_seconds')
    observed = _short_count(int(round(observed_value))) if (
        isinstance(observed_value, (int, float)) and not isinstance(observed_value, bool)
        and math.isfinite(observed_value) and observed_value >= 0
    ) else '—'
    return (
        f"入{_short_rate(rates.get('input_tokens'))} 策{_short_rate(rates.get('policy_tokens'))} "
        f"軌{_short_rate(rates.get('trajectories'))} 更{_fmt(rates.get('updates'), '', 1)}/時 "
        f"({observed}秒)"
    )


def _compact_bepa_summaries(diagnostics):
    if not isinstance(diagnostics, dict):
        return ('Cache候選未知', 'Plan/seed結果未知', 'Run消費/預留未知')
    plans = diagnostics.get('plans') or {}
    cache = plans.get('cache_metadata') or {}
    blocked_names = ('blocked_age', 'blocked_quota', 'blocked_pending', 'blocked_unknown',
                     'blocked_unverified', 'future_policy', 'inactive')
    blocked = [cache.get(name) for name in blocked_names]
    blocked_total = sum(value for value in blocked if type(value) is int) \
        if all(type(value) is int for value in blocked) else None
    cache_line = (
        f"Cache候選{_short_count(cache.get('metadata_eligible'))}(未重驗) | "
        f"封鎖{_short_count(blocked_total)} 過期{_short_count(cache.get('blocked_age'))} "
        f"額度{_short_count(cache.get('blocked_quota'))} 預留{_short_count(cache.get('blocked_pending'))} "
        f"未知{_short_count(cache.get('blocked_unknown'))} | 無cache{_short_count(cache.get('absent'))}"
    )
    states = plans.get('states') or {}
    generation = plans.get('seed_generation') or {}
    plan_line = (
        f"Plan待引導{_short_count(states.get('ready'))} 引導中{_short_count(states.get('seed_running'))} "
        f"已消費{_short_count(states.get('consumed'))} 耗盡{_short_count(states.get('exhausted'))} "
        f"| 失敗嘗試{_short_count(generation.get('failed_attempts'))}；"
        f"重試用盡{_short_count(generation.get('attempt_cap_reached'))}題"
    )
    current = diagnostics.get('current_run') or {}
    reservations = current.get('reservations') or {}
    usage = plans.get('usage') or {}
    run_line = (
        f"Run replay已消費{_short_count(current.get('optimizer_consumed_replay_groups'))}（記憶體） "
        f"| Cache全epoch預留 {_short_count(usage.get('pending_reserved'))}/未知{_short_count(usage.get('pending_commit_unknown'))} "
        f"| cohort帳 {_short_count(reservations.get('reserved'))}/"
        f"{_short_count(reservations.get('committed'))}/{_short_count(reservations.get('commit_unknown'))} "
        f"({_short_count(reservations.get('observed_batch_count'))}批) "
        f"| 待救援{_short_count(current.get('held_groups'))}"
    )
    return cache_line, plan_line, run_line


def _compact_teacher_summaries(bepa, diagnostics):
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    budget = diagnostics.get('budget_scope') or {}
    tokens = budget.get('tokens') or {}
    job_states = budget.get('job_states') or {}
    counts_known = isinstance(budget.get('job_states'), dict)
    active = (sum(job_states.get(name, 0) for name in
                  ('teacher_running', 'source_submitted', 'source_verified'))
              if counts_known else None)
    queued = (sum(job_states.get(name, 0) for name in ('queued', 'retry_wait'))
              if counts_known else None)
    exhausted = job_states.get('exhausted', 0) if counts_known else None
    reasons = budget.get('terminal_reasons') or []
    top_reason = next((item for item in reasons if isinstance(item, dict)
                       and item.get('reason') == 'token_budget_exhausted'), None)
    reason_text = (
        f"預算{_short_count(top_reason.get('count'))}" if top_reason is not None
        else '原因未提供'
    )
    budget_line = (
        f"教師在跑{_short_count(active)}/排隊{_short_count(queued)}；"
        f"{budget.get('scope_id','scope?')}耗盡{_short_count(exhausted)}({reason_text})；"
        f"剩{_short_count(tokens.get('remaining'))}"
        if budget else '教師/預算狀態未知'
    )
    return budget_line


def _teacher_terminal_reason_summary(diagnostics, *, limit=3):
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    budget = diagnostics.get('budget_scope') or {}
    reasons = budget.get('terminal_reasons') or []
    valid = [item for item in reasons if isinstance(item, dict)
             and isinstance(item.get('reason'), str)
             and type(item.get('count')) is int and item['count'] >= 0]
    valid.sort(key=lambda item: (-item['count'], item['reason']))
    if not valid:
        return f"{budget.get('scope_id','scope?')} 終止原因：未提供"
    top = ', '.join(f"{item['reason']}:{item['count']}" for item in valid[:limit])
    return f"{budget.get('scope_id','scope?')} 終止原因（最多{limit}項）：{top}"


def log_tail(manifest, lines, secrets):
    """Bounded reads of only this run's recorded log files; no subprocess tail."""
    result = []
    for process in manifest.get('processes', []):
        path = process.get('log')
        if not path:
            continue
        try:
            with Path(path).open('rb') as handle:
                handle.seek(max(0, os.fstat(handle.fileno()).st_size-32768))
                tail = handle.read().decode(errors='replace').splitlines()[-lines:]
            result.append(dict(role=process.get('worker_id') or process.get('role'),
                               path=path, lines=[clean_text(line, secrets) for line in tail]))
        except OSError:
            continue
    return result


class WatchClient:
    def __init__(self, config, *, timeout=3.0, window=30.0, log_lines=3, gpu=True,
                 transport=None, clock=time.monotonic, interval=2.0):
        self.config = config
        self.clock = clock
        self.rates = Rates(window)
        self.committed_rates = RollingCounterWindow(300.0)
        self.worker_activity = SampledActivityWindow(
            300.0, max_observation_gap_seconds=max(3.0 * interval, timeout + interval)
        )
        self.learner_activity = SampledActivityWindow(
            300.0, max_observation_gap_seconds=max(3.0 * interval, timeout + interval),
            basis='sampled_learner_walltime_not_gpu_utilization',
        )
        self.log_lines = log_lines
        self.gpu = gpu
        self.timeout = timeout
        self.interval = interval
        token = os.environ.get('ARC_ENV_TOKEN')
        if not token:
            raise ValueError('ARC_ENV_TOKEN is required for watch')
        external_worker_envs = {worker.auth_token_env for worker in config.workers
                                if worker.scope in {'gateway', 'service'}}
        self.secrets = tuple(os.environ.get(n) for n in (
            'ARC_ENV_TOKEN', config.bepa.teacher.token_env,
            config.bepa.teacher.api_key_env, *sorted(external_worker_envs),
        ))
        self.client = httpx.AsyncClient(timeout=timeout, transport=transport)
        self.last = {}
        self.runtime_identity = None

    async def close(self):
        await self.client.aclose()

    async def get(self, key, url, *, auth_token_env='ARC_ENV_TOKEN',
                  worker_id=None, node_id=None, boot_id=None, telemetry_epoch=None):
        token = os.environ.get(auth_token_env)
        if not token:
            return key, None, f'missing auth environment {auth_token_env}'
        headers = {'Authorization': 'Bearer ' + token}
        if worker_id is not None:
            if not boot_id or not telemetry_epoch:
                return key, None, 'active external Worker identity is missing'
            headers.update({
                'X-Expected-Worker-Boot-ID': boot_id,
                'X-Expected-Worker-Telemetry-Epoch': telemetry_epoch,
            })
        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            if worker_id is not None:
                expected = {
                    'X-Worker-Node-ID': node_id or worker_id,
                    'X-Worker-Boot-ID': boot_id,
                    'X-Worker-Telemetry-Epoch': telemetry_epoch,
                }
                if any(response.headers.get(name) != value for name, value in expected.items()):
                    raise ValueError('external Worker identity differs from active membership')
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError('expected metadata object')
            return key, payload, None
        except httpx.HTTPStatusError as exc:
            return key, None, f'HTTP {exc.response.status_code}'
        except (httpx.HTTPError, ValueError) as exc:
            return key, None, type(exc).__name__

    def _active_workers(self, manifest):
        configured = {worker.id: worker for worker in self.config.workers}
        membership = manifest.get('membership')
        rows = membership.get('active_workers') if isinstance(membership, dict) else None
        if (isinstance(membership, dict) and membership.get('phase') != 'active'
                and isinstance(membership.get('startup_workers'), list)):
            rows = membership['startup_workers']
        if not isinstance(rows, list):
            return tuple(self.config.workers)
        active = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('worker_id'), str):
                continue
            worker = configured.get(row['worker_id'])
            if worker is None:
                continue
            if worker.scope in {'gateway', 'service'}:
                if (row.get('scope') != worker.scope or not row.get('boot_id')
                        or not row.get('telemetry_epoch')):
                    continue
                worker = replace(
                    worker,
                    boot_id=row['boot_id'],
                    telemetry_epoch=row['telemetry_epoch'],
                    model_sha256=row.get('model_sha256', ''),
                )
            active.append(worker)
        return tuple(active)

    async def poll(self):
        now = self.clock()
        path = active_manifest_path(self.config.run)
        manifest = read_json(path) if path and path.is_file() else {}
        identity = manifest.get('run_id')
        workers = self._active_workers(manifest)
        if identity != self.runtime_identity:
            self.last.clear()
            self.rates.series.clear()
            self.committed_rates.clear()
            self.worker_activity.clear()
            self.learner_activity.clear()
            self.runtime_identity = identity
        stopped = manifest.get('status') in {'stopped', 'failed'}
        requests = [
            ('learner', self.config.learner_url + '/monitor', 'ARC_ENV_TOKEN', None, None, None, None),
            ('learner_health', self.config.learner_url + '/health', 'ARC_ENV_TOKEN', None, None, None, None),
            ('arc', self.config.learner_url + '/arc/health', 'ARC_ENV_TOKEN', None, None, None, None),
            ('bepa', self.config.learner_url + '/bepa/status', 'ARC_ENV_TOKEN', None, None, None, None),
        ]
        requests += [
            (worker.id, worker.url + '/health', worker.auth_token_env,
             worker.id if worker.scope in {'gateway', 'service'} else None,
             worker.node_id if worker.scope in {'gateway', 'service'} else None,
             worker.boot_id or None, worker.telemetry_epoch or None)
            for worker in workers
        ]
        replies = [] if stopped else await asyncio.gather(*(
            self.get(key, url, auth_token_env=auth_env, worker_id=worker_id, node_id=node_id,
                     boot_id=boot_id, telemetry_epoch=telemetry_epoch)
            for key, url, auth_env, worker_id, node_id, boot_id, telemetry_epoch in requests
        ))
        now = self.clock()
        data, errors, freshness = {}, {}, {}
        foreign_runtime = any(key == 'learner' and error is None and manifest
            and payload.get('runtime_run_id') != manifest.get('run_id') for key,payload,error in replies)
        for key, payload, error in replies:
            if foreign_runtime:
                error = 'runtime identity differs from selected manifest'
            if error is None:
                if key == 'learner' and manifest and payload.get('runtime_run_id') != manifest.get('run_id'):
                    error = 'runtime identity differs from selected manifest'
                if key in {w.id for w in workers} and payload.get('worker_id') != key:
                    error = 'worker identity differs from configuration'
            if error is None:
                self.last[key] = (now,payload)
                data[key] = payload
                freshness[key] = dict(live=True, age_seconds=0)
            else:
                errors[key] = error
                if key in self.last:
                    timestamp, previous = self.last[key]
                    data[key] = previous
                    freshness[key] = dict(live=False, age_seconds=max(0,now-timestamp))
        learner = data.get('learner') or {}
        training = learner.get('training') or {}
        learner_live = freshness.get('learner',{}).get('live',False)
        committed = self._committed_snapshot(training, run=learner.get('run') or {},
                                               now=now, fresh=learner_live)
        learner_activity_state = classify_learner_activity(learner, live=learner_live)
        self.learner_activity.observe(
            'learner', learner_activity_state, now=now,
            epoch=training.get('telemetry_epoch') if isinstance(training, dict) else None,
        )
        worker_rows = []
        all_live = True
        aggregate = []
        stream = (learner.get('sampling') or {}).get('stream') or {}
        producer_reason = stream.get('producer_backpressure_reason')
        for worker in workers:
            h = data.get(worker.id) or {}
            live = freshness.get(worker.id,{}).get('live',False)
            rate = self.rates.sample(worker.id, h.get('telemetry_epoch') if live else None,
                                     h.get('generated_tokens_total'), now)
            all_live &= live
            aggregate.append(rate)
            active, maximum = h.get('active_generations'), h.get('max_num_seqs')
            worker_rows.append(dict(id=worker.id, gpu=worker.gpu, local=worker.scope == 'local',
                scope=worker.scope, site_id=worker.site_id, node_id=worker.node_id,
                port=worker.port, endpoint=worker.endpoint or None, live=live,
                age_seconds=freshness.get(worker.id,{}).get('age_seconds'),
                active=active, maximum=maximum, queued=h.get('queued_generations'),
                slot_use_percent=100*active/maximum if type(active) is int and type(maximum) is int and maximum>0 else None,
                generated_tokens=h.get('generated_tokens_total'), tokens_per_second=rate,
                completed=h.get('completed_generations'), failed=h.get('failed_generations'),
                policy_version=h.get('policy_version'), ready=h.get('ready'),
                policy_ready=h.get('policy_ready'), episode_refs=h.get('episode_refs'),
                telemetry_epoch=h.get('telemetry_epoch'), resident_versions=h.get('resident_versions'),
                telemetry=h.get('telemetry'),
                runtime={key: h.get(key) for key in (
                    'backend', 'quantization_mode', 'is_loaded_in_4bit',
                    'artifact_fingerprint', 'base_fingerprint', 'adapter_hash',
                    'max_model_len', 'max_generated_tokens', 'speculative_decoding',
                    'prefix_caching_enabled', 'packed_module_count')},
                kv_cache=h.get('kv_cache')))
            worker_rows[-1]['activity_state'] = classify_worker_activity(
                worker_rows[-1], producer_reason
            )
            self.worker_activity.observe(
                worker.id, worker_rows[-1]['activity_state'], now=now,
                epoch=worker_rows[-1].get('telemetry_epoch'),
            )
        forward = (learner.get('training') or {}).get('forward') or {}
        forward_scope = forward.get('measurement_scope', 'global')
        forward_epoch = forward.get('telemetry_epoch')
        rate_epoch = f'{forward_epoch}:{forward_scope}' if learner_live and forward_epoch else None
        learning_rate = self.rates.sample('learner_forward', rate_epoch,
                                          forward.get('forward_input_tokens'),now)
        if stopped:
            self.last.clear()
        gpu = await asyncio.to_thread(gpu_snapshot) if self.gpu else []
        snapshot = dict(schema='qwen-rl-watch-v1', sampled_at=time.time(), window_seconds=self.rates.window,
            lifecycle=dict(status=manifest.get('status','unmanaged'), run_id=manifest.get('run_id'),
                           manifest=str(path) if path else None),
            membership=manifest.get('membership'),
            learner=learner,
            learner_health=data.get('learner_health') if learner_live else None,
            learner_gpus=list(self.config.learner.gpus),
            parallelism=self.config.learner.parallelism,
            arc=data.get('arc'), bepa=data.get('bepa'), workers=worker_rows,
            evaluation=learner.get('evaluation'), evaluation_live=learner_live,
            throughput=dict(aggregate_output_tokens_per_second=sum(aggregate) if all_live and all(v is not None for v in aggregate) else None,
                            learner_forward_input_tokens_per_second=learning_rate,
                            learner_forward_scope=forward_scope),
            committed=committed,
            worker_activity=self.worker_activity.summary(now=now),
            learner_activity=self.learner_activity.summary(now=now),
            freshness=freshness, errors=errors, gpus=gpu,
            logs=log_tail(manifest,self.log_lines,self.secrets) if self.log_lines else [])
        def redact(value):
            if isinstance(value, str):
                return clean_text(value, self.secrets)
            if isinstance(value, dict):
                return {key:redact(item) for key,item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value
        return redact(snapshot)

    def _committed_snapshot(self, training, *, run, now, fresh):
        process = training.get('process_committed') or {}
        if not process:
            # Keep older /monitor payloads usable while treating missing
            # trajectories as unknown rather than inventing a zero count.
            process = {
                'scope': 'current_process',
                'telemetry_epoch': training.get('telemetry_epoch'),
                'elapsed_seconds': training.get('uptime_seconds'),
                'updates': training.get('committed_updates'),
                'input_tokens': training.get('committed_input_tokens'),
                'policy_tokens': training.get('committed_policy_tokens'),
                'trajectories': training.get('committed_trajectories'),
            }
        process_epoch = process.get('telemetry_epoch') or training.get('telemetry_epoch')
        process_rate = self.committed_rates.observe(
            'process', epoch=process_epoch,
            counters={name: process.get(name) for name in COMMITTED_COUNTERS},
            now=now, fresh=fresh,
        )

        current_run = training.get('run_committed') or {}
        run_id = current_run.get('run_id') or run.get('run_id')
        run_epoch_parts = (current_run.get('telemetry_epoch'), run_id, current_run.get('scope'))
        run_epoch = run_epoch_parts if all(value is not None for value in run_epoch_parts) else None
        run_rate = self.committed_rates.observe(
            'current_process_run', epoch=run_epoch,
            counters={name: current_run.get(name) for name in COMMITTED_COUNTERS},
            now=now, fresh=fresh and run_epoch is not None,
        )

        def totals(source):
            return {name: source.get(name) for name in COMMITTED_COUNTERS}

        def lifetime_rates(source):
            if not fresh:
                return {name: None for name in COMMITTED_COUNTERS}
            elapsed = source.get('elapsed_seconds')
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
                return {name: None for name in COMMITTED_COUNTERS}
            values = totals(source)
            return {
                name: (value * 3600.0 / elapsed
                       if type(value) is int and value >= 0 else None)
                for name, value in values.items()
            }

        run_lifetime_rates = lifetime_rates(current_run)

        return {
            'window_seconds': 300.0,
            'process': {
                'live': fresh,
                'scope': process.get('scope', 'current_process'),
                'telemetry_epoch': process_epoch,
                'elapsed_seconds': process.get('elapsed_seconds'),
                'totals': totals(process),
                'lifetime_rates_per_hour': lifetime_rates(process),
                'rolling': process_rate,
            },
            'run': {
                'live': fresh,
                'scope': current_run.get('scope'),
                'run_id': run_id,
                'elapsed_seconds': current_run.get('elapsed_seconds'),
                'logical_completed_updates': current_run.get('logical_completed_updates'),
                'prior_completed_updates': current_run.get('prior_completed_updates'),
                'prior_resumed_token_totals': current_run.get('prior_resumed_token_totals', 'unknown'),
                'totals': totals(current_run),
                'lifetime_rates_per_hour': run_lifetime_rates,
                'rolling': run_rate,
            },
        }


def _bepa_state_label(value, kind):
    labels = {
        'priority': {'held': '待救援', 'seed_running': '引導中', 'plan_ready': '已有 Plan', 'job_pending': '教師任務中'},
        'job': {
            'queued': '排隊中', 'retry_wait': '等待重試', 'teacher_running': '教師執行中',
            'source_submitted': '來源已提交', 'source_verified': '來源已驗證', 'cache_ready': 'Cache已建立',
            'failed': '失敗', 'exhausted': '額度耗盡', 'outcome_unknown': '結果未知',
            'rejected': '拒絕', 'cancelled': '已取消',
        },
        'plan': {'ready': 'Plan待引導', 'seed_running': 'Qwen引導中', 'consumed': '已使用', 'exhausted': '已耗盡'},
        'cache': {
            'metadata_eligible': 'metadata候選（未重驗）', 'blocked_age': '已過期',
            'blocked_quota': '額度用盡', 'blocked_pending': '已有預留',
            'blocked_unknown': '用量未知', 'blocked_unverified': '來源不匹配',
            'future_policy': '未來策略版本', 'inactive': '已停用', 'absent': '無 Cache',
        },
        'reservation': {
            'reserved': '預留待提交', 'committed': '已提交', 'released': '已釋放',
            'commit_unknown': '提交結果未知',
        },
        'wait': {
            'guided_seed_running': 'Qwen引導執行中', 'blocked_pending': '已有預留',
            'blocked_unknown': '用量未知', 'blocked_age': '已過期',
            'blocked_quota': '額度用盡', 'verified_plan_ready': 'Plan可引導',
            'seed_attempts_exhausted': '引導次數耗盡',
        },
        'basis': {
            'current_process_observed_cohorts': '本process已觀測cohort',
            'persisted_reservation_batch_ids': '已觀測batch ledger',
            'engine_epoch_latest': '本epoch最近持久記錄',
            'empty_current_process_cohort_set': '本process尚無cohort',
        },
    }
    return labels.get(kind, {}).get(value, str(value or '—'))


def _stage_duration_lines(training):
    active = training.get('active_update') is not None
    attempt = training.get('progress') if active else training.get('last_attempt')
    durations = (attempt or {}).get('stage_seconds') or {}
    text = ' | '.join(f"{_phase_label(name)} {_fmt(seconds, '秒', 2)}"
                      for name, seconds in durations.items()) or '尚無階段測量'
    external = training.get('external_stages') or {}
    return [f"{'當前' if active else '最近'}更新wall-clock區間：{text}",
            f"外部區間：對齊 {_fmt(external.get('pre_alignment_seconds'), '秒')} | "
            f"Worker同步v{external.get('worker_sync_policy_version', '—')} "
            f"{_fmt(external.get('worker_sync_seconds'), '秒')}（可重疊，不相加；非CUDA kernel時間）"]


def _bepa_timeline_table(diagnostics, *, max_rows):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from rich.table import Table
    def at(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            return '—'
        return datetime.fromtimestamp(value, ZoneInfo('Asia/Taipei')).strftime('%m-%d %H:%M:%S')
    table = Table(title='BEPA 時間鏈 +08 · 持久記錄（cache／plan可重用）')
    table.add_column('題目', no_wrap=True, max_width=36)
    table.add_column('計畫 → 引導開始 → cache → 預留 → 最近提交', overflow='fold')
    for task in (diagnostics.get('tasks') or [])[:max_rows]:
        plan, cache, reservation = task.get('plan') or {}, task.get('cache') or {}, task.get('reservation') or {}
        values = [plan.get('available_at'), plan.get('attempt_started_at'), cache.get('created_at'),
                  reservation.get('created_at'), reservation.get('committed_at')]
        table.add_row(task.get('task_id') or '未映射題目', ' → '.join(at(value) for value in values))
    return table


def _bepa_task_table(diagnostics, *, max_rows):
    from rich.table import Table
    from rich.text import Text
    tasks = diagnostics.get('tasks') if isinstance(diagnostics, dict) else None
    tasks = tasks if isinstance(tasks, list) else []
    context = diagnostics.get('context') if isinstance(diagnostics, dict) else {}
    context = context if isinstance(context, dict) else {}
    basis = context.get('task_basis', 'unknown basis')
    truncated = ' · 伺服端也有截斷' if context.get('task_rows_truncated') else ''
    table = Table(title=f"BEPA 任務生命週期 · {basis} · 顯示 {min(len(tasks), max_rows)}/{len(tasks)}{truncated}")
    for name in ('任務', '優先序', '等待原因 / 時長', '教師任務', 'Plan', 'Cache metadata', 'Reservation / 範圍'):
        table.add_column(name, no_wrap=True, overflow='ellipsis')
    for task in tasks[:max_rows]:
        job = task.get('job') or {}
        plan = task.get('plan') or {}
        cache = task.get('cache') or {}
        reservation = task.get('reservation') or {}
        wait = task.get('wait') or {}
        if wait.get('reason') is None:
            wait_text = Text('未等待')
        else:
            elapsed = _fmt(wait.get('elapsed_seconds'), '秒') if wait.get('time_basis') != 'unavailable' else '時長未知'
            reason = wait.get('reason')
            wait_text = Text(f"{_bepa_state_label(reason, 'wait')} [{reason}] / {elapsed}")
        task_id = task.get('task_id')
        task_key = task.get('task_key')
        if not task_id and isinstance(task_key, str):
            task_id = f"sha:{task_key[:12]}"
        elif task_id == task_key and isinstance(task_key, str) and len(task_key) == 64:
            task_id = f"sha:{task_key[:12]}"
        table.add_row(
            str(task_id or '—'),
            _bepa_state_label(task.get('priority'), 'priority'),
            wait_text,
            _bepa_state_label(job.get('state'), 'job'),
            _bepa_state_label(plan.get('state'), 'plan'),
            _bepa_state_label(cache.get('metadata_state'), 'cache'),
            f"{_bepa_state_label(reservation.get('state'), 'reservation')} "
            f"({_bepa_state_label(reservation.get('basis'), 'basis')})"
            if reservation.get('state') else '無 Reservation',
        )
    if len(tasks) > max_rows:
        table.add_row(f"另有 {len(tasks)-max_rows} 列", '—', 'JSON保留伺服端回傳diagnostics', '—', '—', '—', '—')
    return table


def render(snapshot, height=None, *, details=False):
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    def number(value, suffix=''):
        return _fmt(value, suffix)

    learner = snapshot.get('learner') or {}
    evaluation = snapshot.get('evaluation')
    if evaluation is None:
        evaluation = learner.get('evaluation')
    evaluation_freshness = (snapshot.get('freshness') or {}).get('learner') or {}
    evaluation_text = _evaluation_summary_text(
        evaluation,
        live=snapshot.get('evaluation_live', evaluation_freshness.get('live')),
        age_seconds=evaluation_freshness.get('age_seconds'),
    )
    evaluation_progress = _evaluation_progress_text(evaluation)
    evaluation_error = _evaluation_runtime_error_text(evaluation)
    training = learner.get('training') or {}
    active = training.get('active_update') or {}
    last = training.get('last_update') or {}
    run = learner.get('run') or {}
    sampling = learner.get('sampling') or {}
    stream = sampling.get('stream') or {}
    reward = learner.get('rewards') or {}
    committed = snapshot.get('committed') or {}
    process_commit = committed.get('process') or {}
    run_commit = committed.get('run') or {}
    progress = training.get('progress') or {}
    activity = training.get('activity') or {}
    quality_lines = _quality_lines(training.get('quality') or {})

    phase = _phase_label(activity.get('state') or progress.get('stage') or training.get('stage'))
    stage_elapsed = progress.get('stage_elapsed_seconds')
    if stage_elapsed is None:
        stage_elapsed = activity.get('since_elapsed_seconds')
    if stage_elapsed is None:
        stage_elapsed = active.get('stage_elapsed_seconds')
    update_elapsed = active.get('elapsed_seconds')
    current_microbatch = progress.get('current_microbatch')
    planned_microbatches = progress.get('planned_microbatches')
    if (type(current_microbatch) is int and current_microbatch > 0
            and type(planned_microbatches) is int and planned_microbatches > 0):
        microbatch = f"{current_microbatch}/{planned_microbatches}"
    elif type(current_microbatch) is int and current_microbatch > 0:
        microbatch = f"{current_microbatch}/?"
    else:
        microbatch = '—'

    worker_activity = snapshot.get('worker_activity') or {}
    states = worker_activity.get('states') or {}
    activity_parts = []
    labels = {
        'inference_inflight': '推理',
        'queued': '排隊',
        'between_generations': 'episode引用',
        'not_ready': '未就緒',
        'unknown': '未知',
    }
    dispatch_states = []
    for state, item in states.items():
        if state.startswith('wait_dispatch:'):
            dispatch_states.append((item.get('seconds', 0.0), state, item))
        elif state in labels:
            activity_parts.append(f"{labels[state]} {number(item.get('percent_of_observed_worker_walltime'), '%')}")
    for _seconds, state, item in sorted(dispatch_states, reverse=True)[:2]:
            activity_parts.append(f"派送等待[{state.split(':', 1)[1]}] {number(item.get('percent_of_observed_worker_walltime'), '%')}")
    coverage = (
        f"known {number(worker_activity.get('known_percent'), '%')} / unknown "
        f"{number(worker_activity.get('unknown_percent'), '%')} "
        f"({_fmt(worker_activity.get('observed_worker_seconds'), 's', 0)}/"
        f"{_fmt(worker_activity.get('expected_worker_seconds'), 'worker-s', 0)})"
        if worker_activity.get('expected_worker_seconds') else 'no activity baseline'
    )
    worker_activity_text = (
        f"5m Worker採樣時長（非GPU閒置率）：{coverage}"
        + (' | ' + ' | '.join(activity_parts) if activity_parts else '')
    )
    learner_activity = snapshot.get('learner_activity') or {}
    learner_states = learner_activity.get('states') or {}
    learner_state_parts = [
        f"{_activity_label(state)} {number(item.get('percent_of_observed_worker_walltime'), '%')}"
        for state, item in sorted(
            learner_states.items(), key=lambda pair: pair[1].get('seconds', 0.0), reverse=True
        )[:3]
    ]
    learner_activity_text = (
        f"5m Learner採樣時長（非GPU閒置率）：已知 {number(learner_activity.get('known_percent'), '%')} / "
        f"未知 {number(learner_activity.get('unknown_percent'), '%')} "
        f"({_fmt(learner_activity.get('observed_wall_seconds'), 's', 0)}/"
        f"{_fmt(learner_activity.get('expected_wall_seconds'), 's', 0)})"
        + (' | ' + ' | '.join(learner_state_parts) if learner_state_parts else '')
    )

    bepa = snapshot.get('bepa') or {}
    diagnostics, cache_summary, plan_summary, consume_summary, _blocked = _bepa_summary(bepa)
    bepa_freshness = (snapshot.get('freshness') or {}).get('bepa') or {}
    if bepa_freshness and not bepa_freshness.get('live'):
        cache_summary = f"BEPA狀態過期 {_fmt(bepa_freshness.get('age_seconds'), '秒')}：{cache_summary}"
    global_text = (
        f"Teacher {bepa.get('teacher_model','—')} | BEPA={bepa.get('enabled','—')} 模式={bepa.get('mode','—')} | "
        f"全scope歷史job狀態 {bepa.get('jobs',{})} | "
        f"全epoch active cache {_fmt(bepa.get('active_cache_entries'), '', 0)}（非可claim候選數）"
    )
    teacher_lines = [global_text]
    seed_scheduler = sampling.get('seed_scheduler') or {}
    if seed_scheduler:
        teacher_lines.append(
            f"Qwen引導排程={seed_scheduler.get('status','—')} | 任務={seed_scheduler.get('active_task_id') or '—'} | "
            f"嘗試 {seed_scheduler.get('attempts','—')} / Cache {seed_scheduler.get('succeeded','—')} / "
            f"失敗 {seed_scheduler.get('failed','—')} | 最久待救援 {number(seed_scheduler.get('oldest_hold_seconds'),'秒')}"
        )
    budget_texts = []
    if diagnostics:
        context = diagnostics.get('context') or {}
        budget_scope = diagnostics.get('budget_scope') or {}
        budget_tokens = budget_scope.get('tokens') or {}
        budget_texts.append(
            f"預算 {budget_scope.get('scope_id','—')}：用量 {_fmt(budget_tokens.get('used'),' tok',0)}/"
            f"{_fmt(budget_tokens.get('cap'),' tok',0)}，預留 {_fmt(budget_tokens.get('reserved'),' tok',0)}，"
            f"unknown={budget_tokens.get('usage_unknown','—')}"
        )
        teacher_lines.extend([
            f"BEPA診斷 {diagnostics.get('diagnostics_version','—')} | 預算範圍={context.get('budget_scope','—')} | "
            f"註冊題目 {_fmt(context.get('registered_task_count'), '', 0)} | basis={context.get('task_basis','—')} | "
            f"來源關聯未重讀",
        ])
        durable = ((diagnostics.get('current_run') or {}).get('reservations') or {})
        if durable:
            teacher_lines.append(
                f"本process更新帳：預留 {_fmt(durable.get('reserved'),'')} / 已提交 {_fmt(durable.get('committed'),'')} / "
                f"釋放 {_fmt(durable.get('released'),'')} / 提交未知 {_fmt(durable.get('commit_unknown'),'')} | "
                f"coverage={durable.get('coverage','unknown')}，恢復前 uses 未知"
            )
        update_states = ((diagnostics.get('current_run') or {}).get('updates') or {})
        if update_states:
            teacher_lines.append(f"本process update attempts：{update_states}")
    else:
        budget_values = bepa.get('budgets') or []
        for item in budget_values[:2]:
            budget_texts.append(
                f"預算 {item.get('scope_id','—')}：用量 {_fmt(item.get('used_tokens'),' tok',0)}/"
                f"{_fmt(item.get('max_total_tokens'),' tok',0)}，預留 {_fmt(item.get('reserved_tokens'),' tok',0)}，"
                f"unknown={item.get('usage_unknown','—')}"
            )
    teacher_lines.extend(budget_texts)
    for budget in bepa.get('budgets',[]):
        if diagnostics and budget.get('scope_id') == (diagnostics.get('context') or {}).get('budget_scope'):
            continue
        teacher_lines.append(
            f"歷史預算 {budget.get('scope_id','—')}：用量 {_fmt(budget.get('used_tokens'),' tok',0)} / "
            f"預留 {_fmt(budget.get('reserved_tokens'),' tok',0)} / "
            f"上限 {_fmt(budget.get('max_total_tokens'),' tok',0)}；unknown={budget.get('usage_unknown','—')}"
        )
    for job in bepa.get('active_teacher_jobs',[]):
        teacher_lines.append(
            f"教師任務 {job.get('task_id','—')} | {job.get('state','—')} | 工具呼叫={job.get('tool_calls','—')} | "
            f"已確認未快取輸入 {_fmt(job.get('confirmed_uncached_input_tokens'),' tok',0)}（僅工具回執，非完整job） | "
            f"source_verified={job.get('source_verified','—')}"
        )
    operation = bepa.get('operation') or {}
    teacher_lines.append(
        f"Bootstrap={operation.get('status','—')}，seeded={operation.get('seeded','—')} | "
        f"全域預留狀態={bepa.get('reservations',{})}"
    )

    summary = Table.grid(padding=(0, 2))
    summary.add_column(no_wrap=True, overflow='ellipsis')
    summary.add_column(no_wrap=True, overflow='ellipsis')
    summary.add_row('運行 / 策略', Text(
        f"{run.get('kind','—')} / {run.get('status',snapshot.get('lifecycle',{}).get('status','—'))} | "
        f"learner v{learner.get('policy_version','—')} / serve v{learner.get('serve_policy_version','—')} | "
        f"updates {run.get('completed_updates','—')}/{run.get('requested_steps','—')}"
    ))
    throughput = snapshot.get('throughput') or {}
    forward_label = 'rank0 前向' if throughput.get('learner_forward_scope') == 'rank0_partial' else '前向'
    summary.add_row('採樣／前向（30s）', Text(
        f"輸出 {number(throughput.get('aggregate_output_tokens_per_second'),' tok/s')} | "
        f"Learner {forward_label} {number(throughput.get('learner_forward_input_tokens_per_second'),' input tok/s')} | "
        f"上次提交 {number(last.get('input_tokens_per_second'),' input tok/s')} / "
        f"{number(last.get('policy_tokens_per_second'),' policy tok/s')}"
    ))
    summary.add_row('有效提交速率（5m，/h）', Text(
        f"run/current process: {_committed_rate_text(run_commit.get('rolling'))} | "
        f"process: {_committed_rate_text(process_commit.get('rolling'))}"
    ))
    logical_updates = run_commit.get('logical_completed_updates')
    run_totals = run_commit.get('totals') or {}
    process_totals = process_commit.get('totals') or {}
    summary.add_row('Run / process totals', Text(
        f"run logical updates {logical_updates if logical_updates is not None else '—'}; "
        f"current-process-run {'live' if run_commit.get('live') else 'stale'} {_totals_text(run_totals)} (prior resumed tokens: "
        f"{run_commit.get('prior_resumed_token_totals','unknown')}); "
        f"process {'live' if process_commit.get('live') else 'stale'} {_totals_text(process_totals)}"
    ))
    summary.add_row('Lifetime average（/h）', Text(
        f"current process-run: {_lifetime_rate_text(run_commit)} | "
        f"process: {_lifetime_rate_text(process_commit)}"
    ))
    summary.add_row('Learner 前台 / microbatch', Text(
        f"{phase} | stage {number(stage_elapsed,'s')} | update {number(update_elapsed,'s')} | "
        f"microbatch {microbatch} | trajectories {active.get('trajectories','—')}"
    ))
    summary.add_row('更新品質', Text(quality_lines[0]))
    summary.add_row('重要性校正', Text(quality_lines[1]))
    summary.add_row('更新前對齊', Text(quality_lines[2]))
    summary.add_row('Native ARC evaluation', Text(evaluation_text))
    summary.add_row('ARC comparison', Text(evaluation_progress))
    if evaluation_error:
        summary.add_row('ARC evaluation error', Text(evaluation_error))
    summary.add_row('Learner activity（5m）', Text(learner_activity_text))
    summary.add_row('Worker activity（5m）', Text(worker_activity_text))
    summary.add_row('採樣組', Text(
        f"{sampling.get('active_groups','—')}/{sampling.get('maximum_groups','—')} 活躍 | "
        f"每題 n={sampling.get('group_size','—')}"
    ))
    if stream:
        summary.add_row('持續隊列 / 等待原因', Text(
            f"ready {stream.get('ready_groups','—')} | 在途 {stream.get('inflight_groups','—')} | "
            f"準備 {stream.get('preparing_groups','—')} | 待救援 {stream.get('deferred_groups','—')} | "
            f"訓練中 {stream.get('claimed_groups','—')} | "
            f"額度 {stream.get('outstanding_groups','—')}/{stream.get('maximum_outstanding_groups','—')} | "
            f"producer {stream.get('producer_backpressure_reason') or '無'} / "
            f"learner {stream.get('learner_backpressure_reason') or '無'}"
        ))
        summary.add_row('過濾 / 無進展守衛', Text(_stream_filter_guard_text(stream)))
        summary.add_row('隊列版本', Text(
            f"ready {stream.get('ready_by_version',{})} | 在途 {stream.get('inflight_by_version',{})} | "
            f"待救援 {stream.get('deferred_by_version',{})}"
        ))
    summary.add_row('BEPA Cache候選', Text(cache_summary))
    summary.add_row('BEPA Plan / 引導結果', Text(plan_summary))
    summary.add_row('BEPA 本process消費 / 預留', Text(consume_summary))
    budget_summary = budget_texts[0] if budget_texts else '預算未知'
    summary.add_row('Teacher / 預算', Text(f"{global_text} | {budget_summary}"))
    if details:
        summary.add_row('終止原因', Text(_teacher_terminal_reason_summary(diagnostics)))
    summary.add_row('獎勵 / 資料', Text(
        f"原生非零 {reward.get('native_nonzero_groups','—')} | 原生有差異 {reward.get('native_variable_groups','—')} | "
        f"全失敗 {reward.get('all_fail_groups','—')} | replay {reward.get('replay_groups','—')} | "
        f"混合有差異 {reward.get('mixed_variable_groups','—')} | "
        f"exact {reward.get('native_exact_episodes','—')}/{reward.get('native_scored_episodes','—')} | "
        f"zero-var {reward.get('zero_variance_filtered','—')} | 丟棄 {reward.get('discarded_groups','—')}"
    ))

    arc = snapshot.get('arc') or {}
    learner_devices = snapshot.get('learner_gpus', [])
    learner_cards = [g for g in snapshot.get('gpus', []) if g['index'] in learner_devices]
    learner_hardware = {}
    if learner_cards:
        for key in ('used_mib', 'total_mib'):
            values = [g.get(key) for g in learner_cards]
            learner_hardware[key] = sum(values) if all(type(v) in (int, float) for v in values) else None
        values = [g.get('utilization_percent') for g in learner_cards]
        learner_hardware['utilization_percent'] = sum(values) / len(values) if all(type(v) in (int, float) for v in values) else None
    summary.add_row('ARC / Learner GPU', Text(
        f"ARC 活躍 {arc.get('active_groups','—')}；登記 {arc.get('registered_groups','—')}/"
        f"{arc.get('maximum_registered_groups','—')}（含保留結果） | GPU "
        f"{number(learner_hardware.get('utilization_percent'),'%')} | VRAM "
        f"{number(learner_hardware.get('used_mib'))}/{number(learner_hardware.get('total_mib'))} MiB"
    ))

    activity_by_worker = worker_activity.get('workers') or {}
    worker_table = Table(title='Worker：槽位與 GPU 計算利用率分開；Activity 為採樣時長')
    for name in ('Worker','策略','狀態','已知/未知 5m','活躍/上限','排隊','佔槽','KV %','Prefix 累計%','生成 tok/s','GPU %','VRAM MiB','完成/失敗'):
        worker_table.add_column(name, no_wrap=True, overflow='ellipsis')
    hardware = {g['index']:g for g in snapshot.get('gpus', [])}
    for row in snapshot.get('workers', []):
        gpu = hardware.get(row['gpu'],{}) if row['local'] else {}
        kv = row.get('kv_cache') or {}
        usage = kv.get('usage_fraction') if row['live'] and kv.get('available') and not kv.get('stale') else None
        kv_percent = 100 * usage if type(usage) in (int, float) and math.isfinite(usage) else None
        prefix = kv.get('prefix_cache') or {}
        hit_fraction = prefix.get('hit_fraction') if row['live'] and prefix.get('available') and not prefix.get('stale') else None
        prefix_percent = 100 * hit_fraction if type(hit_fraction) in (int, float) and math.isfinite(hit_fraction) else None
        state_window = activity_by_worker.get(row['id']) or {}
        state_pct = state_window.get('state_percent') or {}
        observed = state_window.get('observed_seconds')
        if isinstance(observed, (int, float)) and observed > 0:
            unknown_pct = state_pct.get('unknown', 0.0)
            known_pct = 100.0 - unknown_pct if isinstance(unknown_pct, (int, float)) else None
        else:
            unknown_pct = known_pct = None
        worker_table.add_row(
            row['id']+(' [stale/offline]' if not row['live'] else ''), str(row.get('policy_version','—')),
            _activity_label(row.get('activity_state','unknown')),
            f"{number(known_pct,'%')}/{number(unknown_pct,'%')}",
            f"{row.get('active','—')}/{row.get('maximum','—')}", str(row.get('queued','—')),
            number(row.get('slot_use_percent'),'%'), number(kv_percent), number(prefix_percent), number(row.get('tokens_per_second')),
            number(gpu.get('utilization_percent')),
            f"{number(gpu.get('used_mib'))}/{number(gpu.get('total_mib'))}",
            f"{row.get('completed','—')}/{row.get('failed','—')}",
        )

    compact_worker_table = Table(title='Worker 狀態與 5m 採樣時長')
    for name in ('Worker ID', '狀態 / 等待原因', '槽位', '排隊', '已知/未知', 'tok/s', 'GPU%'):
        compact_worker_table.add_column(name, no_wrap=True, overflow='ellipsis',
                                       min_width=8 if name == 'Worker ID' else None)
    for row in snapshot.get('workers', []):
        state_window = activity_by_worker.get(row['id']) or {}
        state_pct = state_window.get('state_percent') or {}
        observed = state_window.get('observed_seconds')
        if isinstance(observed, (int, float)) and observed > 0:
            unknown_pct = state_pct.get('unknown', 0.0)
            known_pct = 100.0 - unknown_pct if isinstance(unknown_pct, (int, float)) else None
        else:
            unknown_pct = known_pct = None
        compact_worker_table.add_row(
            str(row['id']), _activity_label(row.get('activity_state','unknown')),
            f"{row.get('active','—')}/{row.get('maximum','—')}", str(row.get('queued','—')),
            f"{number(known_pct,'%')}/{number(unknown_pct,'%')}",
            number(row.get('tokens_per_second')),
            number((hardware.get(row['gpu'], {}) if row['local'] else {}).get('utilization_percent')),
        )

    errors = [f"{k}: {v}" for k,v in (snapshot.get('errors') or {}).items()]
    errors += [f"{k}: stale {v['age_seconds']:.1f}s" for k,v in (snapshot.get('freshness') or {}).items() if not v['live']]
    if run.get('error'):
        errors.append(str(run['error']))
    logs = [f"[{log['role']}]\n"+'\n'.join(log['lines']) for log in snapshot.get('logs', [])]

    task_table = None
    timeline_table = None
    evaluation_table = None
    if details:
        task_limit = 8 if height is None else min(8, max(1, int((height - 18) / 2)))
        task_table = _bepa_task_table(diagnostics, max_rows=task_limit)
        timeline_table = _bepa_timeline_table(diagnostics, max_rows=min(task_limit, 4))
        if isinstance(evaluation, dict):
            evaluation_table = _evaluation_task_table(evaluation, max_rows=max(1, min(16, task_limit)))
    if height is not None and height < 44:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        timestamp = datetime.fromtimestamp(snapshot.get('sampled_at', 0),ZoneInfo('Asia/Macau')).strftime('%H:%M:%S +08')
        compact = [
            Text(f"Qwen RL | {timestamp} | {run.get('kind','—')}/{run.get('status',snapshot.get('lifecycle',{}).get('status','—'))} | v{learner.get('policy_version','—')} → serve v{learner.get('serve_policy_version','—')} | {run.get('completed_updates','—')}/{run.get('requested_steps','—')}", no_wrap=True, overflow='ellipsis'),
            Text(f"5m/時 本process-run：{_compact_committed_rate_text(run_commit.get('rolling'))}", no_wrap=True, overflow='ellipsis'),
            Text(f"輸出 {number(throughput.get('aggregate_output_tokens_per_second'),' tok/s')} | "
                 f"{forward_label} {number(throughput.get('learner_forward_input_tokens_per_second'),' input tok/s')} | "
                 f"上次提交 {number(last.get('input_tokens_per_second'),' input tok/s')}", no_wrap=True, overflow='ellipsis'),
            Text(f"前台 {_phase_label(phase)} {number(stage_elapsed,'秒')} | update {number(update_elapsed,'秒')} | microbatch {microbatch} | 軌跡 {active.get('trajectories','—')}", no_wrap=True, overflow='ellipsis'),
            Text(quality_lines[0], no_wrap=True, overflow='ellipsis'),
            Text(quality_lines[1], no_wrap=True, overflow='ellipsis'),
            Text(quality_lines[2], no_wrap=True, overflow='ellipsis'),
            Text('Native ARC evaluation: '+evaluation_text, no_wrap=True, overflow='ellipsis'),
            Text('Native ARC baseline / progress: '+evaluation_progress,
                 no_wrap=True, overflow='ellipsis'),
            Text(learner_activity_text, no_wrap=True, overflow='ellipsis'),
            Text(worker_activity_text, no_wrap=True, overflow='ellipsis'),
            Text('BEPA '+_compact_bepa_summaries(diagnostics)[0], no_wrap=True, overflow='ellipsis'),
            Text('BEPA '+_compact_bepa_summaries(diagnostics)[1], no_wrap=True, overflow='ellipsis'),
            Text('BEPA '+_compact_bepa_summaries(diagnostics)[2], no_wrap=True, overflow='ellipsis'),
            Text(f"BEPA {_compact_teacher_summaries(bepa, diagnostics)}", no_wrap=True, overflow='ellipsis'),
            Text(f"採樣 {sampling.get('active_groups','—')}/{sampling.get('maximum_groups','—')} | "
                 f"ready {stream.get('ready_groups','—')} 在途 {stream.get('inflight_groups','—')} "
                 f"準備 {stream.get('preparing_groups','—')} 待救援 {stream.get('deferred_groups','—')} "
                 f"訓練中 {stream.get('claimed_groups','—')} | "
                 f"額度 {stream.get('outstanding_groups','—')}/{stream.get('maximum_outstanding_groups','—')}",
                 no_wrap=True, overflow='ellipsis'),
            Text(f"獎勵: 原生非零 {reward.get('native_nonzero_groups','—')} / 有差異 {reward.get('native_variable_groups','—')} | replay {reward.get('replay_groups','—')} / 混合差異 {reward.get('mixed_variable_groups','—')} | exact {reward.get('native_exact_episodes','—')}/{reward.get('native_scored_episodes','—')} | 丟棄 {reward.get('discarded_groups','—')}", no_wrap=True, overflow='ellipsis'),
            compact_worker_table,
            Text('健康: '+('; '.join(errors) or '正常 / 等待metadata'), no_wrap=True, overflow='ellipsis'),
        ]
        if evaluation_error:
            compact.insert(9, Text('Native ARC evaluation error: '+evaluation_error,
                                   no_wrap=True, overflow='ellipsis'))
        if stream:
            compact[7:7] = [
                Text(
                    f"stream wait {stream.get('learner_backpressure_reason') or '無'} / "
                    f"{stream.get('producer_backpressure_reason') or '無'}",
                    no_wrap=True, overflow='ellipsis',
                ),
                Text(_stream_filter_guard_compact_text(stream), no_wrap=True, overflow='ellipsis'),
            ]
        if details and task_table is not None:
            compact.insert(14, Text('\n'.join(_stage_duration_lines(training))))
            compact.insert(15, timeline_table)
            compact.insert(16, task_table)
            if evaluation_table is not None:
                compact.insert(17, evaluation_table)
        visible_logs = [f"[{log['role']}] {log['lines'][-1]}" for log in snapshot.get('logs', []) if log.get('lines')]
        compact.append(Panel(Text('\n'.join(visible_logs[-max(1,height-19):]) or '無日誌', no_wrap=True, overflow='ellipsis'), title='最近日誌；--log-lines / --json 查看完整快照'))
        return Group(*compact)
    sections = [Panel(summary, title='Qwen RL 即時監控（只讀）')]
    if details and task_table is not None:
        sections.append(Panel(Text('\n'.join(_stage_duration_lines(training))), title='Learner 階段耗時'))
        sections.append(timeline_table)
        sections.append(task_table)
        if evaluation_table is not None:
            sections.append(evaluation_table)
    sections.append(worker_table)
    sections.extend([
        Panel(Text('\n'.join(teacher_lines)), title='教師 / BEPA / 預算'),
        Panel(Text('\n'.join(errors) or '無連線錯誤'), title='健康 / stale'),
        Panel(Text('\n'.join(logs) or '無本機manifest日誌'), title='最近日誌'),
    ])
    return Group(*sections)


async def watch(config, *, interval=2.0, timeout=3.0, window=30.0, once=False,
                json_output=False, log_lines=3, gpu=True, details=False, snapshot_transform=None):
    if not all(math.isfinite(v) for v in (interval,timeout,window)) or interval < 0.2 or timeout <= 0 or window <= 0 or log_lines < 0:
        raise ValueError('watch requires interval>=0.2, positive timeout/window, log-lines>=0')
    from rich.console import Console
    from rich.live import Live
    client = WatchClient(config,timeout=timeout,window=window,log_lines=log_lines,
                         gpu=gpu, interval=interval)
    console = Console()
    async def poll():
        item = await client.poll()
        return snapshot_transform(item) if snapshot_transform is not None else item
    try:
        if once:
            item = await poll()
            if json_output:
                print(json.dumps(item,ensure_ascii=False))
            else:
                console.print(render(item,height=console.size.height,details=details))
            return
        if json_output:
            while True:
                print(json.dumps(await poll(),ensure_ascii=False),flush=True)
                await asyncio.sleep(interval)
        else:
            with Live(console=console,screen=console.is_terminal,auto_refresh=False,vertical_overflow='crop') as live:
                while True:
                    live.update(render(await poll(),height=console.size.height,details=details),refresh=True)
                    await asyncio.sleep(interval)
    finally:
        await client.close()
