"""Round slots for paired and sampling-only Workers."""

def owner_map(learner):
    return {**learner.get("gradient_executor_worker_ids", {}),
            **learner.get("sampling_only_worker_ids", {})}

def owner_slots(learner, *, rotation=0):
    owners = owner_map(learner)
    paired = learner.get("gradient_executor_worker_ids", {})
    quotas = learner.get("sampling_group_quotas", {})
    default = learner.get("gradient_tasks_per_worker", 1)
    counts = {owner: quotas.get(worker, default if owner in paired else 1)
              for owner, worker in owners.items()}
    shared = learner.get("sampling_shared_worker_pairs", {})
    if not isinstance(shared, dict):
        raise ValueError("sampling_shared_worker_pairs must be a mapping")
    by_worker = {worker: owner for owner, worker in learner.get("sampling_only_worker_ids", {}).items()}
    used = set()
    for first, second in shared.items():
        if (not isinstance(first, str) or not isinstance(second, str) or first == second
                or first not in by_worker or second not in by_worker
                or first in used or second in used
                or counts[by_worker[first]] != 1 or counts[by_worker[second]] != 1):
            raise ValueError("shared sampling pairs require disjoint sampling-only Workers with one group each")
        used.update((first, second))
        counts[by_worker[second if rotation % 2 == 0 else first]] = 0
    return tuple(owner for level in range(max(counts.values(), default=0))
                 for owner in sorted(owners) if counts[owner] > level)


def owner_slots_for_ordinal(learner, ordinal):
    """Choose one stable pair owner for each complete sampling round."""
    size = len(owner_slots(learner))
    return owner_slots(learner, rotation=ordinal // size)


def remap_paired_quotas(learner, new_pairs):
    """Follow managed pair order when a deployment renames paired Workers."""
    old_workers = [worker for _, worker in sorted(learner["gradient_executor_worker_ids"].items())]
    new_workers = [worker for _, worker in sorted(new_pairs.items())]
    remap = dict(zip(old_workers, new_workers))
    quotas = learner.get("sampling_group_quotas", {})
    if set(quotas) & (set(old_workers) - set(remap)):
        raise ValueError("explicit sampling quota refers to a removed paired Worker")
    return {remap.get(worker, worker): count for worker, count in quotas.items()}
