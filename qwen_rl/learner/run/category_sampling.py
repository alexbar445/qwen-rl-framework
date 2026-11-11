"""Bounded recent all-correct-group category sampling; no difficulty or reward shaping."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import random

ALPHA = 0.7
WINDOW = 20
EPSILON = 0.1
SCHEMA = 'category-all-correct-group-sampling/v1'


def category(task):
    if getattr(task, 'pool', None) != 'synthetic_arc' or getattr(task, 'split', None) != 'train':
        return None
    value = str(getattr(task, 'family', '')).partition(':')[0]
    return value if value in {'1','2','3','4','5','6'} else None


def task_set_key(tasks):
    data = sorted((t.task_id, category(t)) for t in tasks)
    return hashlib.sha256(json.dumps(data, separators=(',', ':')).encode()).hexdigest()


def distribution(counts, rates):
    """Actual conditional distribution over currently eligible tasks/categories."""
    total = sum(counts.values())
    if total <= 0:
        raise ValueError('category sampling has no eligible tasks')
    base = {c:1/len(counts) for c in counts}
    weights = {c:base[c] * (1 - rates[c] + EPSILON) for c in base}
    norm = sum(weights.values())
    return {c:(1-ALPHA)*base[c] + ALPHA*weights[c]/norm for c in base}


class CategorySampler:
    def __init__(self, record, tasks):
        self.record = record
        self.tasks = {t.task_id:t for t in tasks}
        self.key = task_set_key(tasks)
        state = record.get('category_sampling')
        if state is None:
            state = {'schema':SCHEMA, 'alpha':ALPHA, 'window':WINDOW, 'task_set':self.key,
                     'draws':0, 'history':{}, 'seen_groups':[], 'frozen':None}
            record['category_sampling'] = state
        self.state = state
        if (not isinstance(state, dict) or state.get('schema') != SCHEMA
                or state.get('alpha') != ALPHA
                or type(state.get('window')) is not int or state['window'] <= 0
                or state.get('task_set') != self.key
                or type(state.get('draws')) is not int or state['draws'] < 0):
            raise ValueError('invalid category sampling checkpoint identity or cursor')
        history, seen = state.get('history'), state.get('seen_groups')
        if (not isinstance(history, dict) or any(c not in {'1','2','3','4','5','6'}
                or not isinstance(v,list) or len(v)>state['window'] or any(type(x) is not bool for x in v)
                for c,v in history.items()) or not isinstance(seen,list)
                or any(not isinstance(x,str) or not x for x in seen) or len(seen)!=len(set(seen))):
            raise ValueError('invalid category all-correct group-history checkpoint')
        frozen = state.get('frozen')
        if frozen is not None and (not isinstance(frozen,dict) or not isinstance(frozen.get('scope'),str)
                or set(frozen.get('rates',{})) != {category(t) for t in tasks}
                or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1
                       for v in frozen['rates'].values())):
            raise ValueError('invalid frozen category sampling probabilities')
        # A window change keeps observations and the draw cursor, but invalidates
        # the round estimate computed under the previous setting.
        if state['window'] != WINDOW:
            for entries in history.values():
                entries[:] = entries[-WINDOW:]
            state['window'] = WINDOW
            state['frozen'] = None
        self.seen = set(seen)

    def rates(self):
        history = self.state['history']
        values = [x for entries in history.values() for x in entries]
        prior = sum(values)/len(values) if values else 0.5
        rates = {}
        for c in {category(t) for t in self.tasks.values()}:
            groups = history.get(c,[])
            n = len(groups)
            # N/WINDOW confidence backoff; a full window uses the raw group rate.
            rates[c] = (sum(groups) + (WINDOW-n)*prior)/WINDOW
        return rates

    def observe(self, task, group_id, exacts):
        if task.task_id not in self.tasks or group_id in self.seen:
            return
        if not isinstance(group_id,str) or not group_id or not exacts or any(type(x) is not bool for x in exacts):
            raise ValueError('category observations require a complete scored exact group')
        c = category(task)
        groups = self.state['history'].setdefault(c,[])
        groups[:] = (groups + [all(exacts)])[-WINDOW:]
        self.seen.add(group_id)
        self.state['seen_groups'].append(group_id)

    def choose(self, eligible, scope):
        if not eligible:
            raise ValueError('category sampling has no eligible tasks')
        frozen = self.state['frozen']
        if frozen is None or frozen['scope'] != scope:
            frozen = {'scope':scope, 'rates':self.rates()}
            self.state['frozen'] = frozen
        ordered = sorted(eligible, key=lambda t:t.task_id)
        counts = Counter(category(t) for t in ordered)
        probabilities = distribution(counts, frozen['rates'])
        draw = self.state['draws']
        rng = random.Random(f'category-all-correct-group-v1:{self.record["run_id"]}:{draw}')
        threshold, cumulative = rng.random(), 0.0
        chosen = sorted(counts)[-1]
        for c in sorted(counts):
            cumulative += probabilities[c]
            if threshold < cumulative:
                chosen = c
                break
        candidates = [t for t in ordered if category(t)==chosen]
        task = candidates[rng.randrange(len(candidates))]
        self.state['draws'] += 1
        return task, {'schema':SCHEMA, 'draw':draw, 'scope':scope, 'alpha':ALPHA,
            'window':WINDOW, 'category':chosen, 'category_probabilities':probabilities,
            'task_probability':probabilities[chosen]/len(candidates),
            'eligible_tasks':len(ordered), 'frozen_rates':dict(frozen['rates']),
            'group_counts':{c:len(self.state['history'].get(c,[])) for c in counts},
            'basis':'conditional on Worker affinity, availability and unseen-task preference'}
