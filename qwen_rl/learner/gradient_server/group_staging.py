"""Local group means, staged without optimizer authority, then centrally weighted."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import torch

STAGED_GROUP_SCOPE = "local_group_mean_staging/v1"


def reweight_group_contributions(contributions, *, source_window, final_window, final_job, timing=None):
    """Reduce immutable partitions using private buffers and fixed logical order.

    A store iterator streams payloads in work order. An already materialized
    sequence may arrive in any order and is indexed without copying tensors.
    """
    from collections.abc import Sequence
    from qwen_rl.contracts.gradient_partitions import validate_partitions
    from .staging_timing import measured

    context = source_window.update_context or {}
    if "trajectory_partitions" not in context:
        values = iter(contributions)
        missing = object()
        contribution = next(values, missing)
        if contribution is missing or next(values, missing) is not missing:
            raise ValueError("unpartitioned group must contribute exactly once")
        return reweight_contribution(contribution, source_window=source_window,
                                     final_window=final_window, final_job=final_job, timing=timing)
    validate_partitions(context, source_window.jobs)
    if isinstance(contributions, Sequence):
        by_id = {c.job_id: c for c in contributions}
        if len(by_id) != len(contributions) or set(by_id) != set(source_window.work_order):
            raise ValueError("partition contributions have missing or duplicate coverage")
        contributions = (by_id[job.job_id] for job in source_window.jobs)
    values = iter(contributions)
    if (context.get("gradient_scope") != STAGED_GROUP_SCOPE
            or sum(job.token_count for job in source_window.jobs) != final_job.token_count
            or source_window.n_window_tokens != final_job.token_count):
        raise ValueError("partition group token coverage differs from sealed round")
    if source_window.parameter_layout != final_window.parameter_layout:
        raise ValueError("partition parameter layout differs from sealed round")
    shapes = {p.name: tuple(p.shape) for p in source_window.parameter_layout}
    parameter_names = set(shapes)
    gradients = {}
    counts = {}
    template = None
    missing = object()
    for job in source_window.jobs:
        c = next(values, missing)  # Store I/O has its own disjoint phase timers.
        if c is missing or c.job_id != job.job_id:
            raise ValueError("partition contributions have missing or duplicate ordered coverage")
        with measured(timing, "gradient_merge_validate"):
            for name in ("lineage_id", "engine_epoch", "window_id", "attempt_id", "job_id",
                         "group_id", "owner_executor_id", "policy_version", "token_count",
                         "n_window_tokens", "layout_fingerprint", "compute_profile_fingerprint"):
                if getattr(c, name) != getattr(job, name):
                    raise ValueError(f"partition contribution differs from its lease: {name}")
            for name in ("lineage_id", "engine_epoch", "group_id", "owner_executor_id",
                         "policy_version", "layout_fingerprint", "compute_profile_fingerprint"):
                if getattr(job, name) != getattr(final_job, name):
                    raise ValueError(f"partition group differs from sealed round: {name}")
            if set(c.gradients) != parameter_names:
                raise ValueError("partition gradient parameter presence differs from group layout")
            if template is None:
                template = replace(c, gradients={name: None for name in shapes})
            present = []
            for name, value in c.gradients.items():
                if value is None:
                    gradients.setdefault(name, None)
                else:
                    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                            or value.device.type != "cpu" or tuple(value.shape) != shapes[name]):
                        raise ValueError("partition gradients must be CPU FP32 tensors with the sealed shape")
                    present.append((name, value))
            for name in ("fresh_policy_tokens", "fresh_behavior_tokens", "replay_policy_tokens",
                         "correction_capped_tokens", "dapo_loss_sum", "dapo_loss_microbatch_count"):
                counts[name] = counts.get(name, 0) + (c.diagnostics or {}).get(name, 0)
        with measured(timing, "gradient_accumulator_write"), torch.no_grad():
            targets, sources = [], []
            for name, value in present:
                if gradients.get(name) is None:
                    gradients[name] = value.clone()
                else:
                    targets.append(gradients[name])
                    sources.append(value)
            if targets:
                torch._foreach_add_(targets, sources)
        del c, present, targets, sources
    if next(values, missing) is not missing:
        raise ValueError("partition contributions have missing or duplicate coverage")
    weight = source_window.n_window_tokens / final_window.n_window_tokens
    with measured(timing, "gradient_accumulator_write"), torch.no_grad():
        present = [value for value in gradients.values() if value is not None]
        if present:
            torch._foreach_mul_(present, weight)
    return replace(
        template, window_id=final_job.window_id, attempt_id=final_job.attempt_id,
        job_id=final_job.job_id, token_count=final_job.token_count,
        n_window_tokens=final_window.n_window_tokens,
        gradients=gradients,
        diagnostics={**counts, "partition_coverage": dict(context["trajectory_partitions"]),
                     "staged_normalization": {"local_tokens": source_window.n_window_tokens,
                                              "global_tokens": final_window.n_window_tokens,
                                              "central_weight": weight}},
    )


def rows_fingerprint(rows):
    canonical = [{k: v for k, v in row.items() if k != "row_id"} for row in rows]
    return hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def reweight_contribution(contribution, *, source_window, final_window, final_job, timing=None):
    """Convert a proven local mean into its share of the final global mean.

    Missing gradients remain None (never manufacture zeros for unused params).
    Only the central engine calls this, after exact staged-row validation.
    """
    jobs = source_window.jobs
    if (
        len(jobs) != 1
        or (source_window.update_context or {}).get("gradient_scope")
        != STAGED_GROUP_SCOPE
    ):
        raise ValueError("gradient source is not an isolated staged group")
    source = jobs[0]
    for name in (
        "lineage_id",
        "engine_epoch",
        "policy_version",
        "layout_fingerprint",
        "compute_profile_fingerprint",
    ):
        if getattr(contribution, name) != getattr(source, name) or getattr(
            source, name
        ) != getattr(final_job, name):
            raise ValueError(f"staged gradient identity differs: {name}")
    for name in (
        "window_id",
        "attempt_id",
        "job_id",
        "group_id",
        "owner_executor_id",
        "n_window_tokens",
        "token_count",
    ):
        if getattr(contribution, name) != getattr(source, name):
            raise ValueError(f"staged contribution differs from its lease: {name}")
    if (
        source.n_window_tokens != source.token_count
        or source.group_id != final_job.group_id
        or source.owner_executor_id != final_job.owner_executor_id
        or source.token_count != final_job.token_count
        or source.token_count > final_window.n_window_tokens
    ):
        raise ValueError(
            "staged group owner/token coverage differs from the sealed round"
        )
    weight = source.token_count / final_window.n_window_tokens
    from .staging_timing import measured
    gradients = {}
    for name, value in contribution.gradients.items():
        with measured(timing, "gradient_merge_validate"):
            if value is not None and (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
            ):
                raise ValueError("staged gradients must be FP32 tensors or None")
        with measured(timing, "gradient_accumulator_write"), torch.no_grad():
            gradients[name] = None if value is None else value * weight
    return replace(
        contribution,
        window_id=final_job.window_id,
        attempt_id=final_job.attempt_id,
        job_id=final_job.job_id,
        n_window_tokens=final_window.n_window_tokens,
        gradients=gradients,
        diagnostics={
            **dict(contribution.diagnostics or {}),
            "staged_normalization": {
                "source_window_id": source.window_id,
                "source_attempt_id": source.attempt_id,
                "local_tokens": source.token_count,
                "global_tokens": final_window.n_window_tokens,
                "central_weight": weight,
            },
        },
    )
