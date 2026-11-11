"""Central whole-episode queue and mutually exclusive physical GPU roles.

The controller and gradient store share this object, not the executor processes.
Compute leases never expire into sampling: only an acknowledged GPU return can
release them. Worker health and the existing drain/wake protocol remain required.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import threading
import time


def episode_queue_enabled(learner, *, helper_worker_ids=()):
    """Enable whole-episode placement for local stealing or explicit helpers."""
    if (learner.get('update_backend') != 'central_gradient'
            or learner.get('microbatch_size') != 1):
        return False
    if helper_worker_ids:
        return True
    pairs = learner.get('gradient_executor_worker_ids', {})
    return (learner.get('gradient_local_work_stealing') is True
            and len(pairs) >= 2
            and set(learner.get('gradient_local_executor_worker_ids', ())) == set(pairs.values()))


@dataclass(frozen=True)
class EpisodeRequest:
    owner: str
    policy: tuple
    cost_key: str
    seconds: float
    allow_help: bool


class EpisodeRoleQueue:
    def __init__(self, worker_pools, slots, *, episode_seconds=120.0,
                 switch_seconds=35.0, margin_seconds=10.0, clock=time.monotonic,
                 helper_worker_ids=(), external_owner_ids=()):
        owner_ids = set(worker_pools)
        helper_ids = set(helper_worker_ids)
        if (not worker_pools or owner_ids & helper_ids
                or set(slots) != owner_ids | helper_ids):
            raise ValueError('episode queue needs owner pools and capacities for owners and helpers')
        if any(type(n) is not int or n < 1 for n in slots.values()):
            raise ValueError('episode slots must be positive integers')
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v <= 0
               for v in (episode_seconds, switch_seconds, margin_seconds)):
            raise ValueError('episode and switch cost priors must be finite and positive')
        self.lock = threading.RLock()
        self.pools, self.slots = dict(worker_pools), dict(slots)
        self.helper_worker_ids = frozenset(helper_ids)
        self.external_owner_ids = frozenset(external_owner_ids)
        self.helper_policy = None
        self.pending = {}
        self.active = {}
        self.generation_finished = {}
        self.compute = {}
        self.transfers = {}
        self.handoffs_inflight = set()
        self.migrated = set()
        self.quarantined = set()
        self.estimates = {}
        self.episode_seconds = episode_seconds
        self.switch_seconds = switch_seconds
        self.margin_seconds = margin_seconds
        self.clock = clock
        self.decisions = {w: 'worker' for w in slots}

    def quarantine_worker(self, worker_id: str) -> None:
        """Fence uncertain physical work without releasing compute or episode leases."""
        with self.lock:
            if worker_id not in self.slots:
                raise ValueError("cannot quarantine an unknown Worker")
            self.quarantined.add(worker_id)

    def set_helper_policy(self, policy: tuple[int, str] | None) -> None:
        """Allow external helpers to claim only the acknowledged policy."""
        with self.lock:
            self.helper_policy = policy

    def can_sample_for(self, owner: str, producer: str) -> bool:
        """Authorize an owner, same-pool owner peer, or explicit helper."""
        return (owner in self.pools and producer in self.slots
                and (producer in self.helper_worker_ids
                     or (producer in self.pools and self.pools[producer] == self.pools[owner])))

    def enqueue(self, key, owner, policy, cost_key, *, allow_help=True):
        with self.lock:
            if owner not in self.pools or key in self.pending or key in self.active:
                raise ValueError('unknown owner or duplicate episode identity')
            self.pending[key] = EpisodeRequest(owner, policy, cost_key,
                                               self.estimates.get(cost_key, self.episode_seconds), allow_help)

    def _occupants(self, worker):
        return [value for value in self.active.values() if value[1] == worker]

    def _generating(self, worker):
        return [value for key, value in self.active.items()
                if value[1] == worker and key not in self.generation_finished]

    def note_generation_finished(self, key):
        """End physical generation only after terminal finish and Worker release ACK."""
        with self.lock:
            if key in self.active and key not in self.generation_finished:
                self.generation_finished[key] = max(0.001, self.clock() - self.active[key][2])
                self.transfers.pop(key, None)
                self.handoffs_inflight.discard(key)
                return self.active[key][1]
            return None

    def _owner_delay(self, key):
        """List-schedule existing episodes and the FIFO prefix onto owner slots."""
        request = self.pending[key]
        loads = [0.0] * self.slots[request.owner]
        now = self.clock()
        for active_key, (active, worker, started) in self.active.items():
            if worker == request.owner and active_key not in self.generation_finished:
                i = min(range(len(loads)), key=loads.__getitem__)
                loads[i] += max(1.0, active.seconds - (now - started))
        for pending_key, pending in self.pending.items():
            if pending.owner == request.owner:
                i = min(range(len(loads)), key=loads.__getitem__)
                if pending_key == key:
                    return loads[i]
                loads[i] += pending.seconds
        return 0.0

    def _gain(self, key, worker):
        request = self.pending[key]
        if (not request.allow_help or worker == request.owner
                or not self.can_sample_for(request.owner, worker)):
            return 0.0
        # A helper is borrowed only while otherwise idle. Its own pending
        # episodes always have priority, so helping cannot strand its own work.
        if any(p.owner == worker for p in self.pending.values()):
            return 0.0
        return self._owner_delay(key)

    def claim_episode(self, key, available):
        with self.lock:
            request = self.pending[key]
            reserved = {target for _, target in self.transfers.values()}
            eligible = [w for w in self.slots if w in available and w not in self.compute
                        and w not in reserved and w not in self.quarantined
                        and len(self._generating(w)) < self.slots[w]
                        and self.can_sample_for(request.owner, w)
                        and (w not in self.helper_worker_ids | self.external_owner_ids or self.helper_policy == request.policy)
                        and all(p.policy == request.policy for p, _, _ in self._generating(w))]
            # The earliest unstarted episode for this owner has first claim.
            if next(k for k, p in self.pending.items() if p.owner == request.owner) != key:
                return None
            target = request.owner if request.owner in eligible else None
            if target is None:
                helpers = [w for w in eligible
                           if request.allow_help and (
                               w in self.helper_worker_ids
                               or self._gain(key, w) > self.margin_seconds)]
                if helpers:
                    target = (min(helpers, key=lambda w: len(self._generating(w)))
                              if self.helper_worker_ids else helpers[0])
            if target is None:
                return None
            self.active[key] = (self.pending.pop(key), target, self.clock())
            self.decisions[target] = 'sampling'
            return target

    def begin_external_episode(self, key, worker):
        """Evaluation and other ordinary leases share the same physical fence."""
        with self.lock:
            if (worker in self.compute or worker in self.quarantined
                    or any(worker in pair for pair in self.transfers.values())):
                return False
            self.active[key] = (EpisodeRequest(worker, None, 'external', self.episode_seconds, False),
                                worker, self.clock())
            return True

    def finish_episode(self, key, *, completed=False):
        with self.lock:
            self.pending.pop(key, None)
            self.transfers.pop(key, None)
            self.handoffs_inflight.discard(key)
            self.migrated.discard(key)
            value = self.active.pop(key, None)
            generation_seconds = self.generation_finished.pop(key, None)
            if value is not None and completed:
                request, _, started = value
                seconds = generation_seconds if generation_seconds is not None else max(0.001, self.clock() - started)
                previous = self.estimates.get(request.cost_key, self.episode_seconds)
                self.estimates[request.cost_key] = 0.75 * previous + 0.25 * seconds

    def _prefer_sampling(self, worker):
        if any(p.owner == worker for p in self.pending.values()):
            return True
        threshold = 2 * self.switch_seconds + self.margin_seconds
        return any(self._gain(key, worker) > threshold for key in self.pending)

    @contextmanager
    def compute_admission(self, worker, session):
        """Atomically arbitrate episode vs compute admission, rolling back errors."""
        with self.lock:
            previous = self.compute.get(worker)
            allowed = (worker in self.pools and worker not in self.quarantined
                       and not any(worker in pair for pair in self.transfers.values())) and (not self._generating(worker)
                       and previous in (None, session)
                       and not self._prefer_sampling(worker))
            if allowed:
                if not isinstance(session, str) or not session:
                    raise ValueError('GPU compute lease requires a session identity')
                self.compute[worker] = session
                self.decisions[worker] = 'learner'
            else:
                self.decisions[worker] = 'keep_worker' if previous is None else 'return_to_worker'
            try:
                yield allowed
            except BaseException:
                if allowed and previous is None:
                    self.compute.pop(worker, None)
                raise

    def release_compute(self, worker, session):
        with self.lock:
            if self.compute.get(worker) not in (None, session):
                raise ValueError('stale GPU compute lease release')
            self.compute.pop(worker, None)
            self.decisions[worker] = 'worker'

    def rebalance_target(self, key, available):
        """Reserve a half split, consumed only at individual generation boundaries."""
        with self.lock:
            if self.pending:
                self.cancel_rebalance_plans()
                return None
            if key not in self.active or key in self.generation_finished:
                return None
            if key in self.transfers:
                if self.transfers[key][1] in available:
                    self.handoffs_inflight.add(key)
                    return self.transfers[key][1]
                return None
            request, source, _ = self.active[key]
            if (not request.allow_help or key in self.migrated or source in self.quarantined
                    or source not in self.pools):
                return None
            members = [k for k, (_, w, _) in self.active.items()
                       if w == source and k not in self.generation_finished]
            planned = sum(src == source for src, _ in self.transfers.values())
            if len(members) - planned <= 4:
                return None
            count = len(members) // 2
            reserved = {target for _, target in self.transfers.values()}
            candidates = [w for w in self.pools if w != source and w in available
                          and self.pools[w] == self.pools[source] and w not in self.compute
                          and w not in self.quarantined and w not in reserved
                          and not self._generating(w) and self.slots[w] >= count]
            if not candidates:
                return None
            target = candidates[0]
            movable = [k for k in members if self.active[k][0].allow_help
                       and k not in self.migrated and k not in self.transfers
                       and self.active[k][0].policy == request.policy]
            if key not in movable:
                return None
            # This caller is already at a safe boundary; the rest wait for theirs.
            chosen = [key] + [k for k in movable if k != key]
            for candidate in chosen[:count]:
                self.transfers[candidate] = (source, target)
            self.handoffs_inflight.add(key)
            return target

    def cancel_rebalance_plans(self):
        with self.lock:
            self.transfers = {key: pair for key, pair in self.transfers.items()
                              if key in self.handoffs_inflight}

    def commit_transfer(self, key, source, target):
        with self.lock:
            if self.transfers.get(key) != (source, target) or self.active[key][1] != source:
                raise ValueError('episode handoff reservation changed')
            request, _, started = self.active[key]
            self.active[key] = (request, target, started)
            self.transfers.pop(key)
            self.handoffs_inflight.discard(key)
            self.migrated.add(key)

    def abort_transfer(self, key, *, uncertain_worker=None):
        with self.lock:
            self.transfers.pop(key, None)
            self.handoffs_inflight.discard(key)
            if uncertain_worker is not None:
                self.quarantined.add(uncertain_worker)

    def snapshot(self):
        with self.lock:
            return {'pending_episodes': len(self.pending), 'active_episodes': len(self.active),
                    'generating_episodes': len(self.active) - len(self.generation_finished),
                    'generating_by_worker': {w: len(self._generating(w)) for w in self.slots},
                    'scoring_by_worker': {w: len(self._occupants(w)) - len(self._generating(w)) for w in self.slots},
                    'compute_leases': dict(self.compute), 'decisions': dict(self.decisions),
                    'pending_handoffs': len(self.transfers), 'migrated_episodes': len(self.migrated),
                    'quarantined_workers': sorted(self.quarantined),
                    'episode_seconds': self.episode_seconds, 'switch_seconds': self.switch_seconds,
                    'estimates': dict(self.estimates)}


def validate_sampling_placement(row, owner, scheduler):
    """Require every physical producer to be authorized for the durable owner."""
    physical = row.get('worker_id', owner)
    if physical is None:
        physical = owner
    if row.get("generation_worker_id", physical) != physical:
        raise ValueError("generation receipt differs from its physical Worker identity")
    history = row.get('generation_worker_history', [physical])
    if not isinstance(history, list) or not history or history[-1] != physical:
        raise ValueError('generation Worker history differs from its final receipt')
    for producer in history:
        if producer != owner and (scheduler is None or not scheduler.can_sample_for(owner, producer)):
            raise ValueError('generation Worker history leaves its authorized sampling placement')
    if physical == owner:
        return
    if (scheduler is None or not scheduler.can_sample_for(owner, physical)
            or row.get('generation_worker_id') != physical
            or row.get('source_worker_id') != owner):
        raise ValueError('fresh Worker identity differs from its authorized sampling placement')


def sampling_group_owner(group, scheduler=None):
    """Recover durable ownership without relabeling actual generation workers."""
    workers = group['worker_ids']
    owner = group.get('owner_worker_id')
    if owner is None and len(set(workers)) == 1:
        owner = workers[0]
    if owner is None:
        raise ValueError('sampling group has no durable owner')
    if set(workers) != {owner}:
        rows = group.get('trajectories', ())
        if len(rows) != len(workers):
            raise ValueError('sampling placement coverage differs from complete group')
        for row, worker in zip(rows, workers, strict=True):
            validate_sampling_placement({**row, 'worker_id': worker}, owner, scheduler)
    return owner
