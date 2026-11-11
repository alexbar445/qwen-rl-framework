"""Durable, CPU-only scheduling state for independent native ARC evaluations."""
from __future__ import annotations

import copy
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..runtime.manifest import read_json, write_json_atomic


_SCHEMA = "qwen-rl-evaluation-state-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _count(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class EvaluationRegistry:
    """Persist evaluation requests, pins, task outcomes, and milestone decisions.

    The registry does not decide whether a training update succeeded. Its caller
    must invoke :meth:`observe_training_update` only with the lineage-wide count
    after a real training optimizer commit. Self-tests and diagnostics have no
    scheduling entry point here.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        lineage_id: str,
        evalset_sha256: str,
        task_ids: Iterable[str],
        milestone_interval: int = 10,
    ) -> None:
        self.path = Path(path)
        self.lineage_id = _nonempty(lineage_id, "lineage_id")
        self.evalset_sha256 = _sha256(evalset_sha256, "evalset_sha256")
        ordered_task_ids = tuple(_nonempty(task_id, "task_id") for task_id in task_ids)
        if not ordered_task_ids:
            raise ValueError("task_ids must contain at least one task")
        if len(set(ordered_task_ids)) != len(ordered_task_ids):
            raise ValueError("task_ids must be unique while preserving manifest order")
        self.task_ids = ordered_task_ids
        if type(milestone_interval) is not int or milestone_interval <= 0:
            raise ValueError("milestone_interval must be a positive integer")
        self.milestone_interval = milestone_interval
        self._lock = threading.RLock()
        if self.path.exists():
            self._state = read_json(self.path)
            self._validate_identity(self._state)
        else:
            self._state = self._empty_state()

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "lineage_id": self.lineage_id,
            "evalset_sha256": self.evalset_sha256,
            "task_ids": list(self.task_ids),
            "milestone_interval": self.milestone_interval,
            "lineage_successful_update_count": None,
            "current_policy": None,
            "baseline_request_id": None,
            "pending_request_id": None,
            "pending_milestone_request_id": None,
            "retry_request_id": None,
            "active_request_id": None,
            "last_completed_request_id": None,
            "request_order": [],
            "requests": {},
            "skipped_milestones": [],
            "skipped_evaluations": [],
            "events": [],
        }

    def _validate_identity(self, state: dict[str, Any]) -> None:
        expected = {
            "schema": _SCHEMA,
            "lineage_id": self.lineage_id,
            "evalset_sha256": self.evalset_sha256,
            "task_ids": list(self.task_ids),
            "milestone_interval": self.milestone_interval,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"evaluation registry identity mismatch for {key}")
        for key in ("requests", "skipped_milestones", "skipped_evaluations", "events", "request_order"):
            if not isinstance(state.get(key), (dict if key == "requests" else list)):
                raise ValueError(f"evaluation registry has invalid {key}")

    def _save(self) -> None:
        write_json_atomic(self.path, self._state)

    def snapshot(self) -> dict[str, Any]:
        """Return a detached copy of the durable registry state."""
        with self._lock:
            return copy.deepcopy(self._state)

    def initialize_lineage_count(self, successful_update_count: int, *, source: str) -> None:
        """Set the first audited lineage count before scheduling begins.

        This is for bootstrapping the evaluation feature from an already-running
        lineage. It does not fabricate evaluations for prior milestones.
        """
        count = _count(successful_update_count, "successful_update_count")
        source = _nonempty(source, "source")
        with self._lock:
            current = self._state["lineage_successful_update_count"]
            if current is not None:
                if count < current:
                    raise ValueError("lineage successful update count cannot move backwards")
                if count == current:
                    return
                self._reconcile_count_without_event(current, count, source)
            else:
                self._state["lineage_successful_update_count"] = count
                self._state["events"].append({
                    "type": "lineage_count_initialized",
                    "successful_update_count": count,
                    "source": source,
                    "at": time.time(),
                })
            self._save()

    def _reconcile_count_without_event(self, previous: int, current: int, source: str) -> None:
        """Advance an audited count without claiming missed checkpoint identities."""
        interval = self.milestone_interval
        first = (previous // interval + 1) * interval
        for milestone in range(first, current + 1, interval):
            self._skip_milestone(milestone, "count_reconciled_without_checkpoint_event", source=source)
        self._state["lineage_successful_update_count"] = current
        self._state["current_policy"] = None
        self._state["events"].append({
            "type": "lineage_count_reconciled",
            "previous_count": previous,
            "successful_update_count": current,
            "source": source,
            "at": time.time(),
        })

    def disable_initial_baseline(self) -> None:
        """Retire unfinished automatic baselines without rewriting completed results."""
        with self._lock:
            baseline_id = self._state.get("baseline_request_id")
            baseline = self._state["requests"].get(baseline_id) if baseline_id else None
            if baseline is None or baseline["status"] == "completed":
                return
            baseline["status"] = "coalesced"
            baseline["skip_reason"] = "initial_baseline_disabled"
            for key in ("baseline_request_id", "pending_request_id", "pending_milestone_request_id",
                        "retry_request_id", "active_request_id"):
                if self._state.get(key) == baseline_id:
                    self._state[key] = None
            self._state["events"].append({"type": "initial_baseline_disabled",
                                           "request_id": baseline_id, "at": time.time()})
            self._save()

    def ensure_baseline(
        self,
        *,
        policy_version: int,
        adapter_sha256: str,
        policy_archive_ref: str | None = None,
    ) -> dict[str, Any]:
        """Create or return the one mandatory SFT baseline request."""
        version = _count(policy_version, "policy_version")
        adapter_sha256 = _sha256(adapter_sha256, "adapter_sha256")
        with self._lock:
            request_id = self._state["baseline_request_id"]
            if request_id is None:
                request = self._new_request(
                    kind="baseline",
                    milestone_update_count=None,
                    successful_update_count=0,
                    policy_version=version,
                    adapter_sha256=adapter_sha256,
                    policy_archive_ref=policy_archive_ref,
                )
                request_id = request["request_id"]
                self._state["baseline_request_id"] = request_id
                self._state["events"].append({
                    "type": "baseline_scheduled",
                    "request_id": request_id,
                    "at": time.time(),
                })
                self._save()
            else:
                request = self._state["requests"][request_id]
                if (request["policy_version"] != version
                        or request["adapter_sha256"] != adapter_sha256):
                    raise ValueError("baseline is already pinned to a different SFT policy")
                if policy_archive_ref and request.get("policy_archive_ref") not in (None, policy_archive_ref):
                    raise ValueError("baseline policy archive reference changed")
                if policy_archive_ref and request.get("policy_archive_ref") is None:
                    request["policy_archive_ref"] = policy_archive_ref
                    self._save()
            return self._public_request(self._state["requests"][request_id])

    def observe_training_update(
        self,
        successful_update_count: int,
        *,
        policy_version: int,
        adapter_sha256: str,
        policy_archive_ref: str | None = None,
    ) -> dict[str, Any] | None:
        """Observe a committed training update and queue an exact milestone.

        Calls with repeated counts are idempotent. A count jump over a milestone
        records that milestone as skipped because the supplied policy cannot be
        honestly labeled as that older checkpoint.
        """
        count = _count(successful_update_count, "successful_update_count")
        version = _count(policy_version, "policy_version")
        adapter_sha256 = _sha256(adapter_sha256, "adapter_sha256")
        with self._lock:
            previous = self._state["lineage_successful_update_count"]
            if previous is None:
                raise ValueError("initialize the audited lineage count before observing updates")
            if count < previous:
                raise ValueError("lineage successful update count cannot move backwards")
            policy = {
                "policy_version": version,
                "adapter_sha256": adapter_sha256,
                "policy_archive_ref": policy_archive_ref,
            }
            if count == previous:
                prior_policy = self._state.get("current_policy")
                if prior_policy is not None and prior_policy != policy:
                    raise ValueError("policy identity changed without a successful lineage update")
                return None

            interval = self.milestone_interval
            first = (previous // interval + 1) * interval
            due = list(range(first, count + 1, interval))
            self._state["lineage_successful_update_count"] = count
            self._state["current_policy"] = policy
            self._state["events"].append({
                "type": "training_update_observed",
                "successful_update_count": count,
                **policy,
                "at": time.time(),
            })

            schedulable = count if due and count % interval == 0 else None
            for milestone in due:
                if milestone != schedulable:
                    reason = "checkpoint_event_observed_after_milestone" if milestone < count else "milestone_checkpoint_unavailable"
                    self._skip_milestone(milestone, reason, source="training_update_observation")

            scheduled = None
            if schedulable is not None:
                scheduled = self._schedule_milestone(schedulable, policy)
            self._save()
            return self._public_request(scheduled) if scheduled else None

    def request_current_policy(
        self,
        successful_update_count: int,
        *,
        policy_version: int,
        adapter_sha256: str,
        policy_archive_ref: str | None = None,
    ) -> dict[str, Any]:
        """Queue one manual evaluation for the current policy without advancing training state."""
        count = _count(successful_update_count, "successful_update_count")
        version = _count(policy_version, "policy_version")
        adapter_sha256 = _sha256(adapter_sha256, "adapter_sha256")
        with self._lock:
            observed = self._state["lineage_successful_update_count"]
            if observed is None:
                raise ValueError("initialize the audited lineage count before requesting evaluation")
            if count != observed:
                raise ValueError("manual evaluation count must match the audited lineage count")
            policy = {
                "policy_version": version,
                "adapter_sha256": adapter_sha256,
                "policy_archive_ref": policy_archive_ref,
            }
            current_policy = self._state.get("current_policy")
            if current_policy is not None and current_policy != policy:
                raise ValueError("manual evaluation policy does not match the current lineage policy")

            for prior_id in reversed(self._state["request_order"]):
                prior = self._state["requests"][prior_id]
                if (prior["successful_update_count"] == count
                        and prior["policy_version"] == version
                        and prior["adapter_sha256"] == adapter_sha256
                        and prior["evalset_sha256"] == self.evalset_sha256
                        and prior["status"] != "coalesced"
                        and not (prior["status"] == "completed"
                                 and prior.get("summary", {}).get("complete_valid_evaluation") is False)):
                    return self._public_request(prior)

            self._coalesce_pending(new_kind="manual", new_count=count)
            request = self._new_request(
                kind="manual",
                milestone_update_count=None,
                successful_update_count=count,
                policy_version=version,
                adapter_sha256=adapter_sha256,
                policy_archive_ref=policy_archive_ref,
            )
            self._state["pending_request_id"] = request["request_id"]
            self._state["events"].append({
                "type": "manual_evaluation_scheduled",
                "request_id": request["request_id"],
                "successful_update_count": count,
                "policy_version": version,
                "adapter_sha256": adapter_sha256,
                "at": time.time(),
            })
            self._save()
            return self._public_request(request)

    def retry_errors(self, request_id: str) -> dict[str, Any]:
        """Retry only completed task rows with infrastructure/error evidence."""
        request_id = _nonempty(request_id, "request_id")
        with self._lock:
            if self._state.get("active_request_id") is not None:
                raise RuntimeError("cannot retry evaluation errors while a request is active")
            request = self._request(request_id)
            if request.get("status") not in {"completed", "cancelled", "interrupted"}:
                raise ValueError("evaluation error retry requires a completed or interrupted request")
            retry_id = self._state.get("retry_request_id")
            if request.get("kind") != "baseline" and retry_id not in (None, request_id):
                raise RuntimeError("another evaluation retry is already queued")

            now = time.time()
            retry_task_ids = []
            for task_id, task in request["tasks"].items():
                summary = task.get("summary")
                if (task.get("status") != "completed" or not isinstance(summary, dict)
                        or not summary.get("error")):
                    continue
                history = task.setdefault("result_history", [])
                if not isinstance(history, list):
                    raise ValueError(f"evaluation task {task_id} has invalid result history")
                history.append({
                    "result_ref": task.get("result_ref"),
                    "result_sha256": task.get("result_sha256"),
                    "summary": copy.deepcopy(summary),
                    "archived_at": now,
                })
                task["status"] = "retry_pending"
                task["result_ref"] = None
                task["result_sha256"] = None
                task["summary"] = None
                retry_task_ids.append(task_id)

            already_retryable = any(task.get("status") == "retry_pending"
                                    for task in request["tasks"].values())
            if not retry_task_ids and not already_retryable:
                raise ValueError("evaluation request has no error rows to retry")

            request_history = request.setdefault("result_history", [])
            if not isinstance(request_history, list):
                raise ValueError("evaluation request has invalid result history")
            if (request.get("result_ref") is not None or request.get("result_sha256") is not None
                    or request.get("summary") is not None):
                request_history.append({
                    "result_ref": request.get("result_ref"),
                    "result_sha256": request.get("result_sha256"),
                    "summary": copy.deepcopy(request.get("summary")),
                    "started_at": request.get("started_at"),
                    "completed_at": request.get("completed_at"),
                    "retry_count": request.get("retry_count", 0),
                    "archived_at": now,
                })
            request.update(
                status="interrupted",
                result_ref=None,
                result_sha256=None,
                summary=None,
                started_at=None,
                completed_at=None,
                cancel_reason="retry_errors_requested",
            )
            request["events"].append({
                "type": "errors_requeued",
                "task_ids": retry_task_ids,
                "at": now,
            })
            if request.get("kind") != "baseline":
                self._state["retry_request_id"] = request_id
            if self._state.get("last_completed_request_id") == request_id:
                self._state["last_completed_request_id"] = next((
                    previous for previous in reversed(self._state["request_order"])
                    if self._state["requests"][previous].get("status") == "completed"
                ), None)
            self._state["events"].append({
                "type": "evaluation_errors_requeued",
                "request_id": request_id,
                "task_ids": retry_task_ids,
                "at": now,
            })
            self._save()
            return self._public_request(request)

    def _new_request(
        self,
        *,
        kind: str,
        milestone_update_count: int | None,
        successful_update_count: int,
        policy_version: int,
        adapter_sha256: str,
        policy_archive_ref: str | None,
    ) -> dict[str, Any]:
        request_id = f"eval-{uuid.uuid4().hex}"
        request = {
            "request_id": request_id,
            "kind": kind,
            "milestone_update_count": milestone_update_count,
            "successful_update_count": successful_update_count,
            "lineage_id": self.lineage_id,
            "evalset_sha256": self.evalset_sha256,
            "task_ids": list(self.task_ids),
            "policy_version": policy_version,
            "adapter_sha256": adapter_sha256,
            "policy_archive_ref": policy_archive_ref,
            "comparison": {"baseline_request_id": None, "previous_request_id": None},
            "status": "pending",
            "retry_count": 0,
            "tasks": {task_id: {"status": "pending", "attempts": [], "result_history": []}
                      for task_id in self.task_ids},
            "result_history": [],
            "result_ref": None,
            "result_sha256": None,
            "summary": None,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "cancel_reason": None,
            "events": [],
        }
        self._state["requests"][request_id] = request
        self._state["request_order"].append(request_id)
        return request

    def _schedule_milestone(self, milestone: int, policy: dict[str, Any]) -> dict[str, Any]:
        self._coalesce_pending(new_kind="milestone", new_count=milestone)
        request = self._new_request(
            kind="milestone",
            milestone_update_count=milestone,
            successful_update_count=milestone,
            policy_version=policy["policy_version"],
            adapter_sha256=policy["adapter_sha256"],
            policy_archive_ref=policy.get("policy_archive_ref"),
        )
        self._state["pending_request_id"] = request["request_id"]
        self._state["pending_milestone_request_id"] = request["request_id"]
        self._state["events"].append({
            "type": "milestone_scheduled",
            "request_id": request["request_id"],
            "milestone_update_count": milestone,
            "at": time.time(),
        })
        return request

    def _coalesce_pending(self, *, new_kind: str, new_count: int) -> None:
        old_id = self._state.get("pending_request_id") or self._state.get("pending_milestone_request_id")
        if not old_id:
            return
        old = self._state["requests"][old_id]
        if old["status"] not in {"pending", "cancelled", "interrupted"}:
            return
        old["status"] = "coalesced"
        old["coalesced_by"] = {"kind": new_kind, "successful_update_count": new_count}
        if old["kind"] == "milestone":
            self._skip_milestone(old["milestone_update_count"], "coalesced_by_newer_pending_evaluation",
                                 superseded_by_kind=new_kind, superseded_by_count=new_count)
        else:
            self._state["skipped_evaluations"].append({
                "request_id": old_id,
                "kind": old["kind"],
                "successful_update_count": old["successful_update_count"],
                "reason": "coalesced_by_newer_pending_evaluation",
                "superseded_by_kind": new_kind,
                "superseded_by_count": new_count,
                "at": time.time(),
            })
        self._state["events"].append({
            "type": "evaluation_coalesced",
            "request_id": old_id,
            "superseded_by_kind": new_kind,
            "superseded_by_count": new_count,
            "at": time.time(),
        })
        self._state["pending_request_id"] = None
        self._state["pending_milestone_request_id"] = None

    def _skip_milestone(self, milestone: int, reason: str, **extra: Any) -> None:
        row = {"milestone_update_count": milestone, "reason": reason, "at": time.time(), **extra}
        if row not in self._state["skipped_milestones"]:
            self._state["skipped_milestones"].append(row)

    def claim_next(self) -> dict[str, Any] | None:
        """Claim the mandatory baseline first, otherwise the latest pending milestone."""
        with self._lock:
            active_id = self._state.get("active_request_id")
            if active_id:
                active = self._state["requests"][active_id]
                if active["status"] == "running":
                    return None
                raise ValueError("active evaluation request is not running; recover it explicitly")

            baseline_id = self._state.get("baseline_request_id")
            baseline = self._state["requests"].get(baseline_id) if baseline_id else None
            if baseline and baseline["status"] != "completed":
                request = baseline
                request_id = baseline_id
            else:
                retry_id = self._state.get("retry_request_id")
                retry = self._state["requests"].get(retry_id) if retry_id else None
                if retry and retry["status"] in {"cancelled", "interrupted", "pending"}:
                    request = retry
                    request_id = retry_id
                    self._state["retry_request_id"] = None
                else:
                    pending_id = self._state.get("pending_request_id") or self._state.get("pending_milestone_request_id")
                    request = self._state["requests"].get(pending_id) if pending_id else None
                    request_id = pending_id

            if request is None:
                return None
            if request["status"] not in {"pending", "cancelled", "interrupted"}:
                if request["status"] in {"completed", "coalesced"}:
                    return None
                raise ValueError(f"evaluation request cannot be claimed from {request['status']!r}")

            if request["kind"] != "baseline":
                baseline_id = self._state.get("baseline_request_id")
                baseline = self._state["requests"].get(baseline_id) if baseline_id else None
                request["comparison"]["baseline_request_id"] = (
                    baseline_id if baseline is not None and baseline["status"] == "completed" else None
                )
                request["comparison"]["previous_request_id"] = self._state.get("last_completed_request_id")
                if request["comparison"]["previous_request_id"] == request_id:
                    request["comparison"]["previous_request_id"] = None
                if self._state.get("pending_request_id") == request_id:
                    self._state["pending_request_id"] = None
                if self._state.get("pending_milestone_request_id") == request_id:
                    self._state["pending_milestone_request_id"] = None

            if request["status"] in {"cancelled", "interrupted"}:
                request["retry_count"] += 1
            request["status"] = "running"
            request["started_at"] = time.time()
            request["cancel_reason"] = None
            self._state["active_request_id"] = request_id
            request["events"].append({"type": "claimed", "at": time.time()})
            self._save()
            return self._public_request(request)

    def record_task_started(
        self,
        request_id: str,
        task_id: str,
        episode_id: str | None = None,
        *,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        if episode_id is not None:
            episode_id = _nonempty(episode_id, "episode_id")
        if attempt_id is not None:
            attempt_id = _nonempty(attempt_id, "attempt_id")
        with self._lock:
            request = self._running_request(request_id)
            task = self._task(request, task_id)
            if task["status"] == "completed":
                raise ValueError(f"completed evaluation task cannot be generated again: {task_id}")
            if task["status"] == "in_progress":
                latest = task["attempts"][-1] if task["attempts"] else None
                if latest and (latest.get("attempt_id") == attempt_id or latest.get("episode_id") == episode_id):
                    if latest.get("episode_id") is None and episode_id is not None:
                        latest["episode_id"] = episode_id
                        self._save()
                    return self._public_request(request)
                raise ValueError(f"evaluation task already has an in-progress attempt: {task_id}")
            task["attempts"].append({
                "attempt_id": attempt_id,
                "episode_id": episode_id,
                "status": "in_progress",
                "artifact_refs": [],
            })
            task["status"] = "in_progress"
            request["events"].append({
                "type": "task_started", "task_id": task_id, "attempt_id": attempt_id,
                "episode_id": episode_id, "at": time.time(),
            })
            self._save()
            return self._public_request(request)

    def record_partial(
        self,
        request_id: str,
        task_id: str,
        episode_id: str | None,
        artifact_refs: Any,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            request = self._running_request(request_id)
            task = self._task(request, task_id)
            if task["status"] == "completed":
                raise ValueError(f"completed evaluation task cannot be changed: {task_id}")
            episode_id = episode_id or (task["attempts"][-1].get("episode_id") if task["attempts"] else None)
            if task["attempts"] and task["attempts"][-1].get("episode_id") == episode_id:
                attempt = task["attempts"][-1]
                if (task["status"] == "retry_pending" and attempt.get("status") == "partial"
                        and attempt.get("artifact_refs") == artifact_refs
                        and attempt.get("reason") == reason):
                    return self._public_request(request)
            else:
                attempt = {"episode_id": episode_id, "artifact_refs": []}
                task["attempts"].append(attempt)
            attempt["status"] = "partial"
            attempt["artifact_refs"] = copy.deepcopy(artifact_refs)
            if reason is not None:
                attempt["reason"] = str(reason)
            task["status"] = "retry_pending"
            request["events"].append({
                "type": "task_partial",
                "task_id": task_id,
                "episode_id": episode_id,
                "artifact_refs": copy.deepcopy(artifact_refs),
                "reason": reason,
                "at": time.time(),
            })
            self._save()
            return self._public_request(request)

    def record_task_result(
        self,
        request_id: str,
        task_id: str,
        result_ref: str,
        result_sha256: str,
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        result_ref = _nonempty(result_ref, "result_ref")
        result_sha256 = _sha256(result_sha256, "result_sha256")
        if not isinstance(summary, Mapping):
            raise ValueError("summary must be a mapping")
        summary_copy = copy.deepcopy(dict(summary))
        with self._lock:
            request = self._running_request(request_id)
            task = self._task(request, task_id)
            if task["status"] == "completed":
                if (task.get("result_ref") == result_ref and task.get("result_sha256") == result_sha256
                        and task.get("summary") == summary_copy):
                    return self._public_request(request)
                raise ValueError(f"conflicting result for completed evaluation task: {task_id}")
            if task["status"] not in {"in_progress", "retry_pending"}:
                raise ValueError(f"evaluation task must have a recorded attempt before result: {task_id}")
            if task["attempts"]:
                task["attempts"][-1]["status"] = "completed"
                result_episode_id = summary_copy.get("episode_id")
                if result_episode_id is not None:
                    task["attempts"][-1]["episode_id"] = result_episode_id
                artifact_ref = summary_copy.get("artifact_ref")
                if artifact_ref:
                    task["attempts"][-1]["artifact_refs"] = [artifact_ref]
            task.update({
                "status": "completed",
                "result_ref": result_ref,
                "result_sha256": result_sha256,
                "summary": summary_copy,
            })
            request["events"].append({"type": "task_completed", "task_id": task_id, "result_ref": result_ref, "at": time.time()})
            self._save()
            return self._public_request(request)

    def mark_cancelled(self, request_id: str, reason: str) -> dict[str, Any]:
        reason = _nonempty(reason, "reason")
        with self._lock:
            request = self._request(request_id)
            if request["status"] == "completed":
                return self._public_request(request)
            if request["status"] == "coalesced":
                return self._public_request(request)
            if request["status"] == "cancelled" and request.get("cancel_reason") == reason:
                return self._public_request(request)
            if request["status"] not in {"running", "pending", "cancelled", "interrupted"}:
                raise ValueError(f"cannot cancel evaluation request from {request['status']!r}")
            self._mark_running_tasks_retryable(request, reason=reason, event_type="task_cancelled")
            request["status"] = "cancelled"
            request["cancel_reason"] = reason
            request["events"].append({"type": "cancelled", "reason": reason, "at": time.time()})
            if self._state.get("active_request_id") == request_id:
                self._state["active_request_id"] = None
                if request["kind"] != "baseline" and not self._state.get("retry_request_id"):
                    self._state["retry_request_id"] = request_id
            self._save()
            return self._public_request(request)

    def mark_interrupted(self, request_id: str | None = None) -> dict[str, Any] | None:
        """Record process interruption while preserving task attempts and artifacts."""
        with self._lock:
            request_id = request_id or self._state.get("active_request_id")
            if request_id is None:
                return None
            request = self._request(request_id)
            if request["status"] in {"completed", "interrupted"}:
                return self._public_request(request)
            self._mark_running_tasks_retryable(request, reason="process_interrupted", event_type="task_interrupted")
            request["status"] = "interrupted"
            request["events"].append({"type": "interrupted", "at": time.time()})
            if self._state.get("active_request_id") == request_id:
                self._state["active_request_id"] = None
                if request["kind"] != "baseline" and not self._state.get("retry_request_id"):
                    self._state["retry_request_id"] = request_id
            self._save()
            return self._public_request(request)

    def _mark_running_tasks_retryable(self, request: dict[str, Any], *, reason: str, event_type: str) -> None:
        for task_id, task in request["tasks"].items():
            if task["status"] != "in_progress":
                continue
            if task["attempts"]:
                attempt = task["attempts"][-1]
                attempt["status"] = "interrupted"
                attempt["reason"] = reason
            task["status"] = "retry_pending"
            request["events"].append({
                "type": event_type,
                "task_id": task_id,
                "episode_id": task["attempts"][-1].get("episode_id") if task["attempts"] else None,
                "artifact_refs": copy.deepcopy(task["attempts"][-1].get("artifact_refs", [])) if task["attempts"] else [],
                "reason": reason,
                "at": time.time(),
            })

    def complete(
        self,
        request_id: str,
        result_ref: str,
        result_sha256: str,
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        result_ref = _nonempty(result_ref, "result_ref")
        result_sha256 = _sha256(result_sha256, "result_sha256")
        if not isinstance(summary, Mapping):
            raise ValueError("summary must be a mapping")
        summary_copy = copy.deepcopy(dict(summary))
        with self._lock:
            existing = self._request(request_id)
            if existing["status"] == "completed":
                if (existing.get("result_ref") == result_ref and existing.get("result_sha256") == result_sha256
                        and existing.get("summary") == summary_copy):
                    return self._public_request(existing)
                raise ValueError("conflicting result for completed evaluation request")
            request = self._running_request(request_id)
            incomplete = [task_id for task_id, task in request["tasks"].items() if task["status"] != "completed"]
            if incomplete:
                raise ValueError(f"evaluation cannot complete with unfinished tasks: {incomplete}")
            request.update({
                "status": "completed",
                "result_ref": result_ref,
                "result_sha256": result_sha256,
                "summary": summary_copy,
                "completed_at": time.time(),
            })
            request["events"].append({"type": "completed", "result_ref": result_ref, "at": time.time()})
            self._state["active_request_id"] = None
            self._state["last_completed_request_id"] = request_id
            self._save()
            return self._public_request(request)

    def monitor_snapshot(self) -> dict[str, Any]:
        """Return a compact, native-evaluation-only view suitable for /monitor."""
        with self._lock:
            state = copy.deepcopy(self._state)
        requests = state["requests"]
        active_id = state.get("active_request_id")
        current = requests.get(active_id) if active_id else None
        latest_id = state.get("last_completed_request_id")
        latest = requests.get(latest_id) if latest_id else None
        baseline_id = state.get("baseline_request_id")
        baseline = requests.get(baseline_id) if baseline_id else None
        retry_id = state.get("retry_request_id")
        retry = requests.get(retry_id) if retry_id else None
        pending_id = state.get("pending_request_id") or state.get("pending_milestone_request_id")
        pending = requests.get(pending_id) if pending_id else None
        queued = baseline if baseline and baseline.get("status") != "completed" else retry or pending
        display = current or queued or latest
        task_rows = []
        exact_known = 0
        exact_count = 0
        pass_known = 0
        pass_count = 0
        task_done = 0
        errors = 0
        if display:
            for task_id in display["task_ids"]:
                task = display["tasks"][task_id]
                summary = task.get("summary") or {}
                exact = summary.get("exact") if type(summary.get("exact")) is bool else None
                passed = summary.get("passed") if type(summary.get("passed")) is bool else None
                error = summary.get("error")
                if task["status"] == "completed":
                    task_done += 1
                    if exact is not None:
                        exact_known += 1
                        exact_count += int(exact)
                    if passed is not None:
                        pass_known += 1
                        pass_count += int(passed)
                    if error:
                        errors += 1
                attempts = task.get("attempts") or []
                latest_attempt = attempts[-1] if attempts else {}
                task_rows.append({
                    "task_id": task_id,
                    "status": task.get("status", "unknown"),
                    "episode_id": latest_attempt.get("episode_id") or summary.get("episode_id"),
                    "attempt_id": latest_attempt.get("attempt_id"),
                    "exact": exact,
                    "passed": passed,
                    "error": error,
                    "result_ref": task.get("result_ref"),
                    "result_sha256": task.get("result_sha256"),
                    "partial_artifact_refs": latest_attempt.get("artifact_refs", []),
                })

        summary = (display.get("summary") or {}) if display else {}
        if display and not summary and task_done:
            from .results import aggregate_episode_rows
            summary = aggregate_episode_rows([
                task['summary'] for task in display['tasks'].values()
                if task['status'] == 'completed'])
        current_task_id = None
        current_task_phase = None
        if current:
            current_task_id = next((task_id for task_id, task in current["tasks"].items()
                                    if task.get("status") == "in_progress"), None)
            current_task = current["tasks"].get(current_task_id) if current_task_id else None
            if current_task is not None:
                current_task_phase = "native_episode_running"
            elif any(task.get("status") == "retry_pending" for task in current["tasks"].values()):
                current_task_phase = "retry_pending"
            else:
                current_task_phase = "awaiting_task_start"
        comparison = display.get("comparison", {}) if display else {}
        baseline_comparison_id = comparison.get("baseline_request_id")
        previous_comparison_id = comparison.get("previous_request_id")
        if display and display.get("kind") == "milestone":
            baseline_comparison_id = baseline_comparison_id or baseline_id
            previous_comparison_id = previous_comparison_id or latest_id
        compared_baseline = self._comparison_view(requests.get(baseline_comparison_id))
        compared_previous = self._comparison_view(requests.get(previous_comparison_id))
        interval = state["milestone_interval"]
        observed = state.get("lineage_successful_update_count")
        next_milestone = ((observed // interval) + 1) * interval if type(observed) is int else None
        phase = (
            "baseline_evaluation" if current and current["kind"] == "baseline"
            else "milestone_evaluation" if current and current["kind"] == "milestone"
            else "manual_evaluation" if current and current["kind"] == "manual"
            else "baseline_retry_pending" if queued is baseline and baseline and baseline.get("status") in {"cancelled", "interrupted"}
            else "baseline_pending" if queued is baseline and baseline
            else f"{retry['kind']}_retry_pending" if queued is retry and retry
            else f"{pending['kind']}_pending" if queued is pending and pending
            else "idle"
        )
        policy = display or {}
        reported_errors = summary.get("error_count")
        if reported_errors is None:
            reported_errors = errors if task_done else None
        result = {
            "schema": "qwen-rl-native-evaluation-monitor-v1",
            "enabled": True,
            "lineage_id": state["lineage_id"],
            "evalset_sha256": state["evalset_sha256"],
            "phase": phase,
            "current_task_id": current_task_id,
            "current_task_phase": current_task_phase,
            "request_id": display.get("request_id") if display else None,
            "request_kind": display.get("kind") if display else None,
            "policy_version": policy.get("policy_version"),
            "adapter_sha256": policy.get("adapter_sha256"),
            "successful_update_count": state.get("lineage_successful_update_count"),
            "milestone_update_count": display.get("milestone_update_count") if display else None,
            "tasks_done": task_done,
            "tasks_total": len(display["task_ids"]) if display else len(state["task_ids"]),
            "native_exact": {
                "count": exact_count if exact_known else None,
                "total": exact_known,
            },
            "native_pass": {
                "count": pass_count if pass_known else None,
                "total": pass_known,
            },
            "exact_rate": summary.get("exact_rate") if task_done == len(state['task_ids']) else None,
            "known_exact_rate": summary.get("known_exact_rate", exact_count / exact_known if exact_known else None),
            "unknown_exact_count": len(state['task_ids']) - exact_known,
            "complete_valid_evaluation": task_done == len(state['task_ids']) and summary.get("complete_valid_evaluation", False),
            "generated_tokens": summary.get("generated_tokens"),
            "observation_tokens": summary.get("observation_tokens"),
            "total_tokens": summary.get("total_tokens"),
            "duration_seconds_total": summary.get("duration_seconds_total"),
            "tool_errors": summary.get("tool_errors"),
            "tool_calls": summary.get("tool_calls"),
            "tool_error_rate": summary.get("tool_error_rate") if task_done == len(state['task_ids']) else None,
            "known_tool_error_rate": summary.get("known_tool_error_rate"),
            "tool_error_rate_known_calls": summary.get("tool_error_rate_known_calls"),
            "tool_error_unknown_calls": summary.get("tool_error_unknown_calls"),
            "tool_error_unknown_episodes": (len(state['task_ids']) - task_done
                + summary.get("tool_error_unknown_episodes", 0)),
            "gains": summary.get("gains"),
            "regressions": summary.get("regressions"),
            "errors": reported_errors,
            "next_milestone_update_count": next_milestone,
            "compared_baseline": compared_baseline,
            "compared_previous": compared_previous,
            "result_rows": task_rows,
            "partial_tasks": {
                task_id: copy.deepcopy(task)
                for task_id, task in ((display or {}).get("tasks") or {}).items()
                if task.get("status") in {"in_progress", "retry_pending"}
            },
            "latest_completed_request_id": latest_id,
            "baseline_status": baseline.get("status") if baseline else "not_scheduled",
            "pending_milestone_update_count": (
                pending.get("milestone_update_count")
                if pending and pending.get("kind") == "milestone" else None
            ),
            "pending_request_kind": pending.get("kind") if pending else None,
            "pending_manual_update_count": (
                pending.get("successful_update_count")
                if pending and pending.get("kind") == "manual" else None
            ),
            "retrying_milestone_update_count": (
                requests.get(state.get("retry_request_id"), {}).get("milestone_update_count")
                if state.get("retry_request_id")
                and requests.get(state.get("retry_request_id"), {}).get("kind") == "milestone" else None
            ),
            "retrying_request_kind": retry.get("kind") if retry else None,
            "skipped_evaluations": copy.deepcopy(state["skipped_evaluations"]),
            "skipped_milestones": copy.deepcopy(state["skipped_milestones"]),
        }
        return result

    @staticmethod
    def _comparison_view(request: dict[str, Any] | None) -> dict[str, Any] | None:
        if not request or request.get("status") != "completed":
            return None
        summary = request.get("summary") or {}
        return {
            "request_id": request["request_id"],
            "kind": request["kind"],
            "successful_update_count": request["successful_update_count"],
            "policy_version": request["policy_version"],
            "adapter_sha256": request["adapter_sha256"],
            "result_ref": request.get("result_ref"),
            "summary": copy.deepcopy(summary),
        }

    def _running_request(self, request_id: str) -> dict[str, Any]:
        request = self._request(request_id)
        if self._state.get("active_request_id") != request_id or request["status"] != "running":
            raise ValueError("evaluation request is not the active running request")
        return request

    def _request(self, request_id: str) -> dict[str, Any]:
        try:
            return self._state["requests"][request_id]
        except KeyError as exc:
            raise KeyError(f"unknown evaluation request: {request_id}") from exc

    @staticmethod
    def _task(request: dict[str, Any], task_id: str) -> dict[str, Any]:
        try:
            return request["tasks"][task_id]
        except KeyError as exc:
            raise KeyError(f"task is not in the frozen evaluation set: {task_id}") from exc

    @staticmethod
    def _public_request(request: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(request)
        result["completed_tasks"] = {
            task_id: copy.deepcopy(task)
            for task_id, task in request["tasks"].items()
            if task.get("status") == "completed"
        }
        result["partial_tasks"] = {
            task_id: copy.deepcopy(task)
            for task_id, task in request["tasks"].items()
            if task.get("status") in {"in_progress", "retry_pending"}
        }
        return result
