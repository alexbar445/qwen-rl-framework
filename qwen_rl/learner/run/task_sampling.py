"""Bounded BEPA practice set and ARC2-only review sampling."""
from __future__ import annotations

import hashlib
import json
import random
import secrets
from typing import Any


class ActiveTaskSampling:
    CAPACITY = 16
    REVIEW_SHARE = 0.2
    SCHEMA = "bepa-active-arc2/v1"

    def __init__(self, record: dict, tasks: list[Any]) -> None:
        self.tasks = {t.task_id: t for t in tasks
                      if getattr(t, "pool", None) == "arc2_train"
                      and getattr(t, "split", None) == "train"}
        if not self.tasks:
            raise ValueError("active BEPA sampling requires ARC2 training tasks")
        identity = hashlib.sha256(json.dumps(sorted(self.tasks)).encode()).hexdigest()
        state = record.get("bepa_task_sampling")
        if state is None:
            state = {"schema": self.SCHEMA, "taskset_identity": identity,
                     "active": [], "review": [], "waiting_for_active": [],
                     "random_seed": secrets.token_hex(16), "draws": 0,
                     "dispatch_counts": {"A": 0, "B": 0, "arc2": 0}}
            record["bepa_task_sampling"] = state
        if (not isinstance(state, dict) or state.get("schema") != self.SCHEMA
                or state.get("taskset_identity") != identity):
            raise ValueError("active BEPA taskset identity differs from checkpoint")
        seen = set()
        for name in ("active", "review"):
            ids = state.get(name)
            if (not isinstance(ids, list) or any(not isinstance(t, str) or t not in self.tasks for t in ids)
                    or len(set(ids)) != len(ids) or seen.intersection(ids)):
                raise ValueError("invalid or overlapping BEPA task queues")
            seen.update(ids)
        waiting = state.get("waiting_for_active")
        if (not isinstance(waiting, list)
                or any(not isinstance(t, str) or t not in state["review"] for t in waiting)
                or len(set(waiting)) != len(waiting)):
            raise ValueError("invalid BEPA B-to-A waiting order")
        if len(state["active"]) > self.CAPACITY:
            raise ValueError("active BEPA queue exceeds sixteen tasks")
        seed = state.get("random_seed")
        counts = state.get("dispatch_counts")
        if (not isinstance(seed, str) or len(seed) != 32
                or any(c not in "0123456789abcdef" for c in seed)
                or type(state.get("draws")) is not int or state["draws"] < 0
                or not isinstance(counts, dict) or set(counts) != {"A", "B", "arc2"}
                or any(type(n) is not int or n < 0 for n in counts.values())
                or sum(counts.values()) != state["draws"]):
            raise ValueError("invalid active BEPA sampling cursor")
        self.state = state
        self.state["active_capacity"] = self.CAPACITY
        self._fill()

    def offer(self, task_id: str) -> bool:
        """Request practice; overflow remains sampleable in B until A has room."""
        if (task_id not in self.tasks or task_id in self.state["active"]
                or task_id in self.state["waiting_for_active"]):
            return False
        if task_id not in self.state["review"]:
            self.state["review"].append(task_id)
        self.state["waiting_for_active"].append(task_id)
        self._fill()
        return True

    def _fill(self) -> None:
        while self.state["waiting_for_active"] and len(self.state["active"]) < self.CAPACITY:
            waiting = self.state["waiting_for_active"]
            task_id = waiting.pop(0)
            self.state["review"].remove(task_id)
            self.state["active"].append(task_id)

    def offer_verified_plan(self, task_id: str) -> bool:
        """Keep a verified plan available while preserving A membership and FIFO."""
        return self.offer(task_id)

    def observe_progress(self, task_id: str, *, retired: bool, reopened: bool = False,
                         needs_practice: bool = False) -> None:
        if task_id not in self.tasks:
            return
        if retired:
            for name in ("active", "waiting_for_active"):
                if task_id in self.state[name]:
                    self.state[name].remove(task_id)
            if task_id not in self.state["review"]:
                self.state["review"].append(task_id)
            self._fill()
        elif reopened or needs_practice:
            self.offer(task_id)

    @staticmethod
    def active_priority_for_dispatch(ordinal: int, round_size: int) -> bool:
        """Allocate one quarter of each complete round to A-priority slots.

        For sizes not divisible by four, alternate round quotas so the
        cumulative reservation remains one quarter. Rotate the slots so the
        same paired Workers do not always receive the A-priority slots.
        """
        if type(ordinal) is not int or ordinal < 0 or type(round_size) is not int or round_size < 1:
            raise ValueError("invalid active sampling dispatch ordinal or round size")
        round_index, slot = divmod(ordinal, round_size)
        count = ((round_index + 1) * round_size // 4) - (round_index * round_size // 4)
        rotation = round_index % round_size
        active_slots = {(offset + rotation) % round_size for offset in range(count)}
        return slot in active_slots

    def choose(self, eligible_task_ids: set[str], *, prefer_active: bool = True) -> tuple[Any, str] | None:
        """Optionally prefer eligible A; otherwise draw 20% B / 80% ARC2."""
        if type(prefer_active) is not bool:
            raise ValueError("prefer_active must be a boolean")
        state = self.state
        active = [t for t in state["active"] if t in eligible_task_ids]
        rng = random.Random(f'{self.SCHEMA}:{state["random_seed"]}:{state["draws"]}')
        if prefer_active and active:
            task_id, source = rng.choice(active), "A"
        else:
            review = sorted(set(state["review"]) & eligible_task_ids)
            original = sorted((eligible_task_ids & self.tasks.keys())
                              - set(state["review"]) - set(state["active"]))
            if not review and not original:
                return None
            source = "B" if review and (not original or rng.random() < self.REVIEW_SHARE) else "arc2"
            task_id = rng.choice(review if source == "B" else original)
        state["draws"] += 1
        state["dispatch_counts"][source] += 1
        return self.tasks[task_id], source
