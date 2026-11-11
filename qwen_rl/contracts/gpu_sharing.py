"""Explicit local Worker selection for fixed-topology GPU role sharing."""
from collections.abc import Mapping


def shared_workers(config: Mapping) -> tuple[dict, ...]:
    learner = config["learner"]
    selected = learner.get("shared_worker_ids", ())
    if (not isinstance(selected, (list, tuple)) or not selected
            or any(not isinstance(value, str) or not value.strip() for value in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("learner.shared_worker_ids must explicitly name unique shared Workers")
    workers = config["workers"]
    by_id = {worker["id"]: worker for worker in workers}
    if len(by_id) != len(workers):
        raise ValueError("Worker IDs must be unique before selecting shared GPUs")
    if set(selected) - by_id.keys():
        raise ValueError("learner.shared_worker_ids contains unknown Worker IDs")
    node = config.get("node_scope", {"site_id": "central", "node_id": "central"})
    result = tuple(by_id[key] for key in selected)
    for worker in result:
        if (worker.get("scope", "local") not in {"local", "service"}
                or any(worker.get(key) != node[key] for key in ("site_id", "node_id"))):
            raise ValueError("shared Workers must belong to the local Learner node; remote sharing is forbidden")
    learner_gpus = learner["gpus"]
    if (not isinstance(learner_gpus, (list, tuple)) or not learner_gpus
            or any(type(gpu) is not int or gpu < 0 for gpu in learner_gpus)
            or len(set(learner_gpus)) != len(learner_gpus)):
        raise ValueError("Learner GPUs must be unique nonnegative local indices")
    gpus = [worker["gpu"] for worker in result]
    if (any(type(gpu) is not int or gpu < 0 for gpu in gpus)
            or len(set(gpus)) != len(gpus) or set(gpus) != set(learner_gpus)):
        raise ValueError("phase_shared requires exactly one selected Worker per Learner GPU")
    return result
