"""Operational summaries derived from fresh, source-bound metadata only."""
from __future__ import annotations

import math
from typing import Any

WAIT_LABELS = {
    "finite_update_budget": "本次请求的更新额度已用完",
    "filtered_group_limit": "连续无有效训练信号的题组达到限制",
    "ready_queue_full": "训练队列额度已满",
    "policy_sync": "等待 Worker 确认新策略",
    "awaiting_ready_groups": "等待同版本有效题组收齐",
    "policy_lag": "等待可消费的行为策略版本",
    "awaiting_worker_capacity": "等待 Worker 容量",
}


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _seconds(value: Any) -> float | None:
    if type(value) not in (int, float) or value < 0:
        return None
    try:
        seconds = float(value)
    except OverflowError:
        return None
    return seconds if math.isfinite(seconds) else None


def operations_for(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not snapshot:
        return {"training_pipeline": [], "insights": [], "activity": None}
    learner = snapshot.get("learner") or {}
    freshness = (snapshot.get("freshness") or {}).get("learner") or {}
    if freshness.get("live") is not True:
        return {"training_pipeline": [], "insights": [{
            "id": "learner_observation_stale", "severity": "unknown",
            "title": "Learner 即时状态未知", "message": "最新探测未成功；历史状态不能证明当前训练仍在前进。",
            "source": "watch.freshness.learner"}], "activity": None}
    sampling = learner.get("sampling") or {}
    stream = sampling.get("stream") or {}
    training = learner.get("training") or {}
    progress = training.get("progress") or {}
    activity = training.get("activity") or {}
    active_update = training.get("active_update")
    lifecycle = snapshot.get("lifecycle")
    source_bound_central = (
        snapshot.get("owned_source_verified") is True
        and isinstance(lifecycle, dict)
        and lifecycle.get("backend") == "central_gradient"
    )
    central_update_active = source_bound_central and isinstance(active_update, dict)
    phase = progress.get("stage") or training.get("stage")
    phase_source = "watch.learner.training.progress"
    if central_update_active:
        # The central Learner owns aggregation/optimizer, while GPU executors
        # perform backward asynchronously. The current watch telemetry does not
        # expose their active substage, so report the whole update window only.
        phase = "梯度/优化器更新（子阶段未知）"
        phase_source = "source-bound central_gradient + watch.learner.training.active_update"
    phase_elapsed = (_seconds(active_update.get("elapsed_seconds")) if central_update_active
                     else _seconds(progress.get("stage_elapsed_seconds")))
    handoff = learner.get("pending_policy_handoff")
    workers = snapshot.get("workers") or []
    worker_counts = [_count(row.get("active")) for row in workers if isinstance(row, dict)]
    generating = (sum(worker_counts) if workers and len(worker_counts) == len(workers)
                  and all(value is not None for value in worker_counts)
                  and all(row.get("live") is True for row in workers) else None)
    update_count = (None if "active_update" not in training else
                    0 if active_update is None else
                    1 if isinstance(active_update, dict) else None)
    checkpoint_count = (None if central_update_active else
                        int(str(phase).split(":")[-1] == "checkpoint") if phase else None)
    handoff_count = (int(isinstance(handoff, dict)) if "pending_policy_handoff" in learner else None)

    def stage(key, label, count, unit, source):
        return {"id": key, "label": label, "count": count, "unit": unit,
                "known": count is not None, "state": "active" if count else ("idle" if count == 0 else "unknown"),
                "source": source}

    pipeline = [
        stage("generating", "Worker 生成", generating, "请求", "watch.workers[].active"),
        stage("ready", "有效题组就绪", _count(stream.get("ready_groups")), "组", "watch.learner.sampling.stream.ready_groups"),
        stage("claimed", "已领取训练组", _count(stream.get("claimed_groups")), "组", "watch.learner.sampling.stream.claimed_groups"),
        stage("optimizer", "梯度/优化器更新（子阶段未知）" if central_update_active else "Learner 更新",
              update_count, "次", "watch.learner.training.active_update"),
        stage("checkpoint", "完整状态保存", checkpoint_count, "次", "watch.learner.training.progress.stage"),
        stage("handoff", "策略交接", handoff_count, "项", "watch.learner.pending_policy_handoff"),
    ]
    insights = []
    reason = stream.get("producer_backpressure_reason")
    if isinstance(reason, str) and reason:
        insights.append({"id": "sampling_wait", "severity": "warning" if reason == "filtered_group_limit" else "info",
                         "title": "采样派送正在等待", "message": WAIT_LABELS.get(reason, reason),
                         "source": "watch.learner.sampling.stream.producer_backpressure_reason", "reason": reason})
    if phase:
        insights.append({"id": "learner_phase", "severity": "info", "title": "Learner 当前阶段",
                         "message": str(phase), "elapsed_seconds": phase_elapsed,
                         "source": phase_source,
                         **({"reported_stage": active_update.get("stage")}
                            if central_update_active else {})})
    health = snapshot.get("learner_health") or {}
    compute = health.get("compute") or {}
    if compute.get("state") == "failed":
        insights.insert(0, {"id": "compute_failed", "severity": "error", "title": "Learner compute 已失败",
                            "message": str(compute.get("error") or "请查看 compute 角色日志和完整状态收据。"),
                            "source": "watch.learner_health.compute"})
    if isinstance(handoff, dict):
        insights.append({"id": "policy_handoff", "severity": "info", "title": "策略交接尚未结束",
                         "message": "Learner 和 serve 版本分开显示；以 Worker SHA ACK 及交接收据确认发布。",
                         "source": "watch.learner.pending_policy_handoff", "ref": handoff.get("version")})
    run = learner.get("run") or {}
    if run.get("error"):
        insights.insert(0, {"id": "run_error", "severity": "error", "title": "本次运行报告错误",
                           "message": str(run["error"]), "source": "watch.learner.run.error"})
    activity_view = {
        "phase": phase,
        "state": "active" if central_update_active else activity.get("state"),
        "elapsed_seconds": (phase_elapsed if central_update_active
                            else _seconds(activity.get("since_elapsed_seconds"))),
        "time_basis": ("reported_update_walltime" if central_update_active
                       else "reported_foreground_walltime"),
        "phase_source": phase_source,
    }
    if central_update_active:
        activity_view["reported_stage"] = active_update.get("stage")
        activity_view["reported_stage_source"] = "watch.learner.training.active_update.stage"
    return {"training_pipeline": pipeline, "insights": insights, "activity": activity_view}
