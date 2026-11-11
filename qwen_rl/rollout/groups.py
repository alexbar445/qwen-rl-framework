"""Immutable policy identity and complete-batch boundaries for the rollout pipeline."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicySnapshot:
    version: int
    adapter_sha256: str
    base_fingerprint: str
    artifact_fingerprint: str

    @classmethod
    def from_policy(cls, policy: dict[str, Any]):
        version = policy["version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("policy snapshot requires a nonnegative integer version")
        digest = policy.get("source_adapter_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("policy snapshot requires an adapter SHA256")
        return cls(version, digest, policy["base_fingerprint"], policy["artifact_fingerprint"])


@dataclass(frozen=True)
class RolloutBatch:
    """A sealed cohort: only complete valid groups, in dispatch order."""
    policy: PolicySnapshot
    groups: tuple[dict[str, Any], ...]

    @property
    def trajectories(self):
        return [row for group in self.groups for row in group["trajectories"]]

    @property
    def worker_ids(self):
        return [worker for group in self.groups for worker in group["worker_ids"]]

    def validate(self, groups_per_update: int, group_size: int, learner_version: int):
        if len(self.groups) != groups_per_update:
            raise ValueError("training requires exactly the configured complete groups")
        if not 0 <= learner_version - self.policy.version <= 1:
            raise ValueError("rollout batch policy lag must be 0 or 1")
        seen = set()
        for group in self.groups:
            if (group.get("policy_version") != self.policy.version
                    or group["record"]["adapter_sha256"] != self.policy.adapter_sha256):
                raise ValueError("rollout batch group identity differs from its policy snapshot")
            rows = group["trajectories"]
            if len(rows) != group_size or len(group["worker_ids"]) != group_size:
                raise ValueError("rollout batch contains an incomplete group")
            ids = {row["group_id"] for row in rows}
            if len(ids) != 1 or seen.intersection(ids):
                raise ValueError("rollout batch contains mixed or duplicate group IDs")
            seen.update(ids)
            if {row["policy_version"] for row in rows} != {self.policy.version}:
                raise ValueError("rollout batch contains mixed behavior policy versions")
