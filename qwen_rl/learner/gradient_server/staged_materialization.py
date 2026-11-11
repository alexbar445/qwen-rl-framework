"""Materialize complete immutable groups before the Learner GPU-release fence."""
from __future__ import annotations

import time
from dataclasses import dataclass

from ...contracts.gradients import GradientContribution
from .core import PreparedGradientWindow

from .group_staging import reweight_group_contributions, rows_fingerprint
from .staging_timing import StagingTiming


@dataclass(frozen=True)
class MaterializedRound:
    contributions: tuple[GradientContribution, ...]
    preparation: PreparedGradientWindow


def materialize_staged_round(engine, round_id, final_window, job_rows):
    timing = StagingTiming()
    assembled = {}
    sources = []
    status = "failed"
    preparation = None
    deadline = time.monotonic() + engine.gradient_wait_timeout

    def check_running():
        if engine._shutdown_requested.is_set():
            raise RuntimeError("central shutdown cancelled the pre-optimizer gradient wait")
        if time.monotonic() >= deadline:
            raise TimeoutError("staged gradients did not finish before timeout")

    try:
        engine.store.seal_work_round(round_id)
        with engine._staging_lock:
            staged_group_ids = {group_id for (source_round, group_id) in engine._staged_groups
                                if source_round == round_id}
        if staged_group_ids != {job.group_id for job in final_window.jobs}:
            raise ValueError("final optimizer window does not cover the complete staged round")
        pending = []
        for job in final_window.jobs:
            with engine._staging_lock:
                staged = engine._staged_groups.get((round_id, job.group_id))
            if staged is None or staged["fingerprint"] != rows_fingerprint(job_rows[job.job_id]["rows"]):
                raise ValueError("sealed round rows differ from the staged complete group")
            pending.append((job, staged["window"]))
            sources.append(staged["window"])

        while pending:
            check_running()
            progressed = False
            pending_snapshot = list(pending)
            with timing.measure("gradient_ready_query"):
                progress = engine.store.staged_window_progress([
                    (source.window_id, source.attempt_id) for _, source in pending_snapshot
                ])
            for (job, source), state in zip(pending_snapshot, progress, strict=True):
                check_running()
                if state["error"] or state["state"] in {"cancelled", "failed"}:
                    raise RuntimeError("staged gradient was cancelled or failed")
                ready, total = state["accepted"], state["total"]
                if ready != total or total != len(source.jobs):
                    continue
                # A group may wait behind earlier CPU materialization after the
                # batch snapshot. Recheck its state/count just before consuming it.
                with timing.measure("gradient_ready_query"):
                    current = engine.store.staged_window_progress(
                        [(source.window_id, source.attempt_id)],
                    )[0]
                if current["error"] or current["state"] in {"cancelled", "failed"}:
                    raise RuntimeError("staged gradient was cancelled or failed")
                ready, total = current["accepted"], current["total"]
                if ready != total or total != len(source.jobs):
                    continue
                # Accepted payloads are immutable. CPU work does not acquire or
                # mutate the model, so it may overlap remaining GPU backward.
                contributions = engine.store.iter_contributions(
                    source.window_id, source.attempt_id, timing=timing,
                )
                assembled[job.job_id] = reweight_group_contributions(
                    contributions, source_window=source,
                    final_window=final_window, final_job=job, timing=timing,
                )
                check_running()
                with timing.measure("payload_retire"):
                    engine.store.release_materialized_staged_payloads(source.window_id, source.attempt_id)
                pending.remove((job, source))
                progressed = True
            if not progressed:
                with timing.measure("gradient_ready_wait"):
                    engine._shutdown_requested.wait(min(0.05, max(0, deadline - time.monotonic())))

        contributions = tuple(assembled[job.job_id] for job in final_window.jobs)
        # The final CPU scan/sum uses privately owned buffers. In particular it
        # does not attach model .grad or enter the optimizer/checkpoint boundary.
        with timing.measure("final_gradient_prepare"):
            preparation = engine.core.prepare_window_gradients(final_window, contributions)

        # Every Learner must prove park/offload and parked health before CPU
        # optimizer entry. Worker restart may continue behind the still-active
        # scheduler fence; publication waits for its separate release receipt.
        check_running()
        with timing.measure("compute_park_wait"):
            engine.store.wait_for_sessions_parked(
                round_id, timeout=max(0, deadline - time.monotonic()),
                cancel_event=engine._shutdown_requested,
            )
        check_running()
        status = "ready"
        # Readiness order is not floating-point accumulation/work order.
        return MaterializedRound(contributions, preparation)
    finally:
        cleanup_errors = []
        if status != "ready":
            if preparation is not None:
                engine.core.discard_prepared_gradients(preparation)
            # The exception traceback may outlive the failed update. Do not let
            # its locals retain completed group tensors or the final tuple.
            assembled.clear()
            contributions = ()
            for source in sources:
                try:
                    # These windows only stage local means; this function has
                    # never entered the optimizer. Fence late uploads as well.
                    engine.store.release_staged_window(
                        source.window_id, source.attempt_id,
                        reason="materialization_failed_before_optimizer_entry",
                    )
                except Exception as exc:  # noqa: BLE001 - preserve the primary materialization failure.
                    cleanup_errors.append(type(exc).__name__)
        summary = timing.snapshot(status=status, completed_groups=len(assembled))
        if cleanup_errors:
            summary["cleanup_errors"] = cleanup_errors
        evidence = getattr(engine, "evidence", None)
        if isinstance(evidence, dict):
            evidence["last_staging_timings"] = summary
        record = getattr(getattr(engine, "core", None), "record_staging_timing", None)
        if callable(record):
            record(final_window.window_id, final_window.attempt_id, summary)
