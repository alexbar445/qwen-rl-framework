"""Local trajectory scheduling and the GPU-return fence before optimizer apply.

Placement is supplied by the trusted deployment, never by a claiming node.
The HTTP protocol remains outbound-pull and can later support other placements.
"""
from __future__ import annotations

import json
import time

from qwen_rl.contracts.gradient_partitions import STAGED_SCOPE


def partition_peak_input_tokens(row):
    """Read maximum sequential row length checked against the immutable rowbundle."""
    context = json.loads(row['window_json']).get('update_context') or {}
    costs = context.get('trajectory_peak_input_tokens', {})
    cost = costs.get(row['job_id'])
    if type(cost) is not int or cost < 1:
        raise ValueError('host compute admission requires validated trajectory input length')
    return cost


def longest_available_partition(rows):
    """Cost-order one eligible oldest round without changing logical work order.

    Only central-declared input costs checked against immutable rowbundles are
    scheduling evidence. Ordinary jobs without partition costs keep their
    existing order; policy-token count is never substituted for input length.
    """
    if not rows:
        return None
    first = json.loads(rows[0]["window_json"])
    context = first.get("update_context") or {}
    round_id = context.get("staged_round_id")
    if "trajectory_input_tokens" not in context or not round_id:
        return rows[0]
    candidates = []
    for index, row in enumerate(rows):
        window = json.loads(row["window_json"])
        ctx = window.get("update_context") or {}
        if (ctx.get("staged_round_id") == round_id
                and all(window.get(key) == first.get(key)
                        for key in ("lineage_id", "engine_epoch", "policy_version"))):
            cost = ctx.get("trajectory_input_tokens", {}).get(row["job_id"])
            if type(cost) is not int or cost < 1:
                raise ValueError("cost-ordered round has a partition without validated input length")
            candidates.append((-cost, index, row))
    return min(candidates, key=lambda item: item[:2])[2]


def _staged_round_key(window):
    context = window.get("update_context") or {}
    if (context.get("gradient_scope") != STAGED_SCOPE
            or "trajectory_partitions" not in context
            or not context.get("staged_round_id")):
        return None
    return (window["lineage_id"], window["engine_epoch"], window["policy_version"],
            context["staged_round_id"])


class WorkStealingStore:
    @staticmethod
    def _validate_compute_session_state(row):
        active, parked = row["active"], row["parked"]
        if (type(active) is not int or active not in (0, 1)
                or type(parked) is not int or parked not in (0, 1)):
            raise ValueError("compute session has an invalid active or parked state")

    def _compute_session_rows(self, round_id=None):
        with self._db() as db:
            if round_id is None:
                rows = db.execute(
                    "SELECT executor_id,session_id,round_id,active,parked FROM compute_sessions"
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT executor_id,session_id,round_id,active,parked FROM compute_sessions "
                    "WHERE round_id=?", (round_id,),
                ).fetchall()
        for row in rows:
            self._validate_compute_session_state(row)
        return rows

    def begin_sampling_round(self, round_id, *, lineage_id, engine_epoch,
                             policy_version, executor_ids, tasks_per_worker, first_ordinal, owner_slots=None):
        """The trusted controller declares every sampling slot before staging."""
        if (not isinstance(round_id, str) or not round_id
                or type(policy_version) is not int or policy_version < 0
                or type(tasks_per_worker) is not int or tasks_per_worker < 1
                or type(first_ordinal) is not int or first_ordinal < 0
                or isinstance(executor_ids, (str, bytes))
                or not executor_ids or len(set(executor_ids)) != len(executor_ids)
                or any(not isinstance(value, str) or not value
                       for value in (lineage_id, engine_epoch, *executor_ids))):
            raise ValueError('invalid central sampling round')
        slots = tuple(owner_slots) if owner_slots is not None else tuple(executor_ids) * tasks_per_worker
        if not slots or set(slots) != set(executor_ids):
            raise ValueError('invalid central sampling slots')
        plan = (lineage_id, engine_epoch, policy_version, slots, first_ordinal)
        with self._lock:
            previous = self._sampling_rounds.get(round_id)
            if previous is not None and previous['plan'] != plan:
                raise ValueError('central sampling round plan changed')
            if previous is None:
                self._sampling_rounds[round_id] = {'plan': plan, 'outcomes': {}}

    def note_sampling_outcome(self, round_id, ordinal):
        """Count accepted and filtered outcomes; duplicate callbacks are idempotent."""
        with self._lock:
            state = self._sampling_rounds[round_id]
            slots, first = state['plan'][3:]
            if type(ordinal) is not int or not first <= ordinal < first + len(slots):
                raise ValueError('invalid sampling outcome ordinal')
            owner = slots[ordinal % len(slots)]
            outcomes = state['outcomes']
            if ordinal not in outcomes:
                outcomes[ordinal] = owner

    def discard_sampling_round(self, round_id=None):
        with self._lock:
            if round_id is None:
                self._sampling_rounds.clear()
            else:
                self._sampling_rounds.pop(round_id, None)

    def _sampling_allows_compute(self, db, window, executor_id):
        """Require owner outcomes unless central physical-role admission fences sampling."""
        context = window.get('update_context') or {}
        if context.get('gradient_scope') != STAGED_SCOPE:
            return True
        round_id = context.get('staged_round_id')
        state = self._sampling_rounds.get(round_id)
        if state is not None:
            lineage, epoch, version, executors, _first = state['plan']
            if (window['lineage_id'], window['engine_epoch'], window['policy_version']) != (lineage, epoch, version):
                raise ValueError('staged window differs from its sampling attestation')
            if executor_id in self.remote_executor_ids:
                return True
            if executor_id not in executors:
                return False
            scheduler = self.episode_queue
            worker = self.executor_worker_ids().get(executor_id)
            if scheduler is not None and worker in scheduler.pools:
                # claim() still atomically calls compute_admission: generating
                # episodes, own pending work, transfers and quarantine block it.
                # Terminal episodes awaiting scores no longer need this GPU.
                return True
            if sum(owner == executor_id for owner in state['outcomes'].values()) == executors.count(executor_id):
                return True
        # A full round seal is also a central proof that every rollout ended.
        return bool(db.execute('SELECT 1 FROM store_metadata WHERE key=?',
                               ('work_round_sealed:' + str(round_id),)).fetchone())

    def _help_candidates(self, db, executor_id):
        pool = self.work_stealing_pools.get(executor_id)
        if pool is None:
            return []
        rows = db.execute(
            "SELECT j.*,w.window_json,w.compute_profile_json,w.created_at AS window_created_at FROM jobs j "
            "JOIN windows w USING(window_id,attempt_id) "
            "WHERE w.state='pending' AND j.state='queued' "
            "ORDER BY w.created_at,j.work_order DESC"
        ).fetchall()
        eligible = []
        for row in rows:
            owner = row["owner_executor_id"]
            if owner == executor_id or self.work_stealing_pools.get(owner) != pool:
                continue
            window = json.loads(row["window_json"])
            key = _staged_round_key(window)
            if key is None:
                continue
            if not self._sampling_allows_compute(db, window, executor_id):
                continue
            # The sampling attestation or round seal proves eligibility even if
            # the owner has not claimed a sibling yet. Physical role admission
            # and the helper's residency probe remain separate gates in claim().
            eligible.append(row)
        return eligible

    def seal_work_round(self, round_id):
        """Central-only proof that all P rollout outcomes (including skips) arrived."""
        with self._lock, self._db() as db:
            db.execute("INSERT OR IGNORE INTO store_metadata(key,value) VALUES(?, 'true')",
                       ("work_round_sealed:" + round_id,))

    def _open_work_session(self, db, executor_id, session_id, window_json):
        context = json.loads(window_json).get("update_context", {})
        prior = db.execute("SELECT * FROM compute_sessions WHERE executor_id=?", (executor_id,)).fetchone()
        if prior is not None:
            self._validate_compute_session_state(prior)
            active = prior["active"]
        else:
            active = 0
        if active == 1:
            if prior["parked"]:
                raise ValueError("parked active compute session must be released before claiming work")
            if "trajectory_partitions" not in context:
                raise ValueError("executor already has an active compute session")
            if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
                raise ValueError("partition claims require a bounded compute session identity")
            round_id = context["staged_round_id"]
            if prior["session_id"] == session_id and prior["round_id"] == round_id:
                return
            pending = db.execute("SELECT window_json FROM windows WHERE state IN ('pending','applying')").fetchall()
            if any(json.loads(item["window_json"]).get("update_context", {}).get("staged_round_id")
                   == prior["round_id"] for item in pending):
                raise ValueError("executor must return its previous GPU session before claiming another round")
            raise ValueError("executor already has another active compute session")
        if "trajectory_partitions" not in context:
            return
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("partition claims require a bounded compute session identity")
        round_id = context["staged_round_id"]
        db.execute(
            "INSERT INTO compute_sessions(executor_id,session_id,round_id,active,parked) VALUES(?,?,?,1,0) "
            "ON CONFLICT(executor_id) DO UPDATE SET session_id=excluded.session_id,"
            "round_id=excluded.round_id,active=1,parked=0", (executor_id, session_id, round_id),
        )

    def park_compute_session(self, executor_id, session_id):
        """Record GPU park without releasing the executor or Worker compute lease."""
        if not isinstance(executor_id, str) or not executor_id or len(executor_id) > 128:
            raise ValueError("compute session executor identity is malformed")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("compute session parked identity is malformed")
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT session_id,active,parked FROM compute_sessions WHERE executor_id=?",
                    (executor_id,),
                ).fetchone()
                if row is None or int(row["active"]) != 1:
                    raise ValueError("parked receipt requires a currently active compute session")
                if row["session_id"] != session_id:
                    raise ValueError("parked receipt belongs to another executor boot")
                self._validate_compute_session_state(row)
                leased = db.execute(
                    "SELECT 1 FROM jobs WHERE executor_id=? AND state='leased' LIMIT 1",
                    (executor_id,),
                ).fetchone()
                if leased is not None:
                    raise ValueError("cannot attest a parked session with a leased unaccepted job")
                if row["parked"] == 0:
                    updated = db.execute(
                        "UPDATE compute_sessions SET parked=1 "
                        "WHERE executor_id=? AND session_id=? AND active=1 AND parked=0",
                        (executor_id, session_id),
                    )
                    if updated.rowcount != 1:
                        raise ValueError("compute session changed before the parked receipt committed")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        with self._changed:
            self._changed.notify_all()

    def release_compute_session(self, executor_id, session_id):
        """Caller attests successful park/wake; a stale daemon cannot release a newer fence."""
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("compute session release identity is malformed")
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT * FROM compute_sessions WHERE executor_id=?", (executor_id,)).fetchone()
                if row is not None:
                    self._validate_compute_session_state(row)
                if row and row["session_id"] != session_id:
                    raise ValueError("compute session release belongs to another executor boot")
                if row:
                    # The caller has joined every compute/upload owner before
                    # returning this exact session. Expiry alone is insufficient:
                    # only this terminal attestation may release its RAM claims.
                    db.execute(
                        "UPDATE jobs SET state='queued',executor_id=NULL,lease_id=NULL,"
                        "lease_epoch=lease_epoch+1,lease_expires_at=NULL "
                        "WHERE executor_id=? AND state='leased' AND EXISTS "
                        "(SELECT 1 FROM windows w WHERE w.window_id=jobs.window_id "
                        "AND w.attempt_id=jobs.attempt_id AND w.state='pending' AND "
                        "json_extract(w.window_json,'$.update_context.staged_round_id')=?)",
                        (executor_id, row["round_id"]),
                    )
                db.execute("UPDATE compute_sessions SET active=0 WHERE executor_id=? AND session_id=?",
                           (executor_id, session_id))
                scheduler = self.episode_queue
                worker_id = self.executor_worker_ids().get(executor_id)
                if scheduler is not None and worker_id is not None:
                    scheduler.release_compute(worker_id, session_id)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        with self._changed:
            self._changed.notify_all()

    def active_compute_sessions(self, round_id=None):
        return [
            {"executor_id": row["executor_id"], "session_id": row["session_id"],
             "round_id": row["round_id"]}
            for row in self._compute_session_rows(round_id) if row["active"] == 1
        ]

    def unparked_compute_sessions(self, round_id=None):
        return [
            {"executor_id": row["executor_id"], "session_id": row["session_id"],
             "round_id": row["round_id"]}
            for row in self._compute_session_rows(round_id)
            if row["active"] == 1 and row["parked"] == 0
        ]

    def wait_for_sessions_parked(self, round_id, *, timeout, cancel_event=None):
        deadline = time.monotonic() + timeout
        with self._changed:
            while self.unparked_compute_sessions(round_id):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("central shutdown cancelled the pre-optimizer park wait")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("gradient GPU sessions have not safely parked")
                self._changed.wait(timeout=min(0.1, remaining))

    def wait_for_sessions_released(self, round_id, *, timeout, cancel_event=None):
        deadline = time.monotonic() + timeout
        with self._changed:
            while self.active_compute_sessions(round_id):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("central shutdown cancelled the Worker return wait")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("gradient GPU sessions have not safely returned to Worker mode")
                self._changed.wait(timeout=min(0.1, remaining))
