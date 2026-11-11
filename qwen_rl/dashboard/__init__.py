"""Deployment-bound, read-only Qwen RL observation workspace.

The collector selects local, independent Learner or offline evidence by explicit
source, runtime identity and lineage. It never runs model work, calls a Teacher
provider, changes a ledger or controls training roles. The operation contract is
in docs/DASHBOARD.md; sampled history and persisted events are separate.
"""

from .collect import Collector
