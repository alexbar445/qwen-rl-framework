"""Immutable trajectory partitions of one complete, staged GRPO group.

The group is the statistical unit. A partition is only a unit of computation:
all executors receive the complete group and use its common token denominator.
"""
from __future__ import annotations

from collections.abc import Mapping
import ipaddress
from urllib.parse import urlsplit


DEFAULT_GRADIENT_ACCUMULATION_ROWS = 1

PARTITION_KEY = "trajectory_partitions"
STAGED_SCOPE = "local_group_mean_staging/v1"


def accumulation_partitions(rows, max_rows):
    """Bundle complete rows for one backward accumulation/export transaction.

    IDs retain their original complete-group positions. Microbatch size stays
    independent: the executor can backward one sequence at a time and export
    only after all rows in this immutable lease have contributed.
    """
    if type(max_rows) is not int or max_rows < 1:
        raise ValueError("gradient_accumulation_rows must be a positive integer")
    return [list(range(start, min(start + max_rows, len(rows))))
            for start in range(0, len(rows), max_rows)]


def local_executor_pools(config):
    """Only deployment-selected, loopback paired GPUs may share trajectory work."""
    learner = config["learner"]
    if not learner.get("gradient_local_work_stealing", False):
        return {}
    selected = set(learner.get("gradient_local_executor_worker_ids", ()))
    workers = {worker["id"]: worker for worker in config["workers"]}
    scope = config.get("node_scope", {})
    result = {}
    for executor, worker_id in learner["gradient_executor_worker_ids"].items():
        if worker_id not in selected:
            continue
        worker = workers[worker_id]
        kind = worker.get("scope", "local")
        host = (urlsplit(worker["endpoint"]).hostname if kind == "service"
                else worker.get("host", "127.0.0.1"))
        try:
            local = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        if kind not in {"local", "service"} or not local:
            raise ValueError("work stealing requires an explicitly local loopback Worker")
        site = worker.get("site_id") or scope.get("site_id")
        node = worker.get("node_id") or scope.get("node_id")
        if not site or not node:
            raise ValueError("work stealing requires a trusted site/node identity")
        result[executor] = f"{site}/{node}"
    return result


def gradient_executor_pools(config):
    """Trusted cross-node dedicated executors may consume paired owners' partitions."""
    learner = config["learner"]
    remote = learner.get("gradient_remote_executor_ids", ())
    if not remote:
        return local_executor_pools(config)
    return {executor: "cross-node-gradient" for executor in
            (*learner["gradient_executor_worker_ids"], *learner.get("sampling_only_worker_ids", {}), *remote)}


def validate_partitions(context, jobs):
    partitions = context.get(PARTITION_KEY)
    if partitions is None:
        return
    if (context.get("gradient_scope") != STAGED_SCOPE
            or not isinstance(context.get("staged_round_id"), str) or not context["staged_round_id"]
            or not isinstance(partitions, Mapping)
            or set(partitions) != {job.job_id for job in jobs}
            or len({job.group_id for job in jobs}) != 1):
        raise ValueError("trajectory partitions require one complete staged group")
    coverage = []
    for job in jobs:
        ids = partitions[job.job_id]
        if (not isinstance(ids, (list, tuple)) or not ids
                or any(type(i) is not int or i < 0 for i in ids)
                or list(ids) != sorted(set(ids))):
            raise ValueError("partition row IDs must be nonempty, sorted and unique")
        coverage.extend(ids)
    if sorted(coverage) != list(range(len(coverage))):
        raise ValueError("trajectory partitions must cover every group row exactly once")
    lengths = context.get("trajectory_input_tokens")
    if lengths is not None and (
        not isinstance(lengths, Mapping) or set(lengths) != set(partitions)
        or any(type(value) is not int or value < 1 for value in lengths.values())
    ):
        raise ValueError("trajectory input-token costs must cover each partition with positive integers")
    peaks = context.get('trajectory_peak_input_tokens')
    if peaks is not None and (
        not isinstance(peaks, Mapping) or set(peaks) != set(partitions)
        or any(type(value) is not int or value < 1 for value in peaks.values())
    ):
        raise ValueError('trajectory peak input-token costs must cover each partition')


def selected_rows(window, job, rows):
    """Validate full-group coverage, then select rows without changing advantages."""
    partitions = (window.update_context or {}).get(PARTITION_KEY)
    if partitions is None:
        return list(rows)
    validate_partitions(window.update_context, window.jobs)
    by_id = {row.get("row_id"): row for row in rows}
    coverage = [i for ids in partitions.values() for i in ids]
    if (any(type(row.get("row_id")) is not int for row in rows)
            or len(by_id) != len(rows) or set(by_id) != set(coverage)):
        raise ValueError("partition context differs from complete group row coverage")
    total = sum(sum(row["policy_mask"][1:]) for row in rows)
    if total != window.n_window_tokens:
        raise ValueError("partition full-group token denominator differs")
    result = [by_id[i] for i in partitions[job.job_id]]
    lengths = (window.update_context or {}).get("trajectory_input_tokens")
    if lengths is not None and lengths[job.job_id] != sum(len(row["input_ids"]) for row in result):
        raise ValueError("trajectory input-token cost differs from immutable row coverage")
    peaks = (window.update_context or {}).get('trajectory_peak_input_tokens')
    if peaks is not None and peaks[job.job_id] != max(len(row['input_ids']) for row in result):
        raise ValueError('trajectory peak input-token cost differs from immutable row coverage')
    if sum(sum(row["policy_mask"][1:]) for row in result) != job.token_count:
        raise ValueError("partition token coverage differs from the leased job")
    return result
