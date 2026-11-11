"""Hashed offline teacher captures; no rollout rewards or behavior probabilities."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch

from .compression_config import sha256


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


class RecoveryDataset:
    def __init__(
        self, path, *, teacher_head_sha256, max_sequence_length, hidden_size, vocab_size
    ):
        self.path = Path(path)
        self.manifest = json.loads(self.path.read_text())
        d = self.manifest
        if (
            d.get("schema") != "qwen-rl-recovery-dataset/v1"
            or d.get("teacher_head_sha256") != teacher_head_sha256
            or not isinstance(d.get("teacher_identity"), str)
            or not re.fullmatch("[0-9a-f]{64}", d["teacher_identity"])
            or not isinstance(d.get("tokenizer_sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", d["tokenizer_sha256"])
        ):
            raise ValueError("teacher dataset identity is incomplete or mismatched")
        self.rows = d.get("records")
        if not isinstance(self.rows, list) or not self.rows:
            raise ValueError("dataset has no records")
        self.hidden_size, self.vocab_size = hidden_size, vocab_size
        self.max_sequence_length = max_sequence_length
        self.splits = {"train": [], "validation": []}
        samples, paths, groups = set(), set(), {"train": set(), "validation": set()}
        for i, row in enumerate(self.rows):
            if not isinstance(row, dict) or row.get("split") not in self.splits:
                raise ValueError(
                    "only explicit train/validation captures are accepted; test stays closed"
                )
            if set(row) != {
                "sample_id",
                "source_group_id",
                "split",
                "path",
                "sha256",
                "input_ids_sha256",
                "sequence_length",
                "scored_tokens",
            }:
                raise ValueError("capture record has unknown or missing fields")
            if not all(
                isinstance(row[k], str) and row[k]
                for k in ["sample_id", "source_group_id", "path"]
            ):
                raise ValueError("capture IDs and paths must be nonempty")
            p = (self.path.parent / row["path"]).resolve()
            if row["sample_id"] in samples or p in paths:
                raise ValueError("duplicate sample or capture path")
            samples.add(row["sample_id"])
            paths.add(p)
            groups[row["split"]].add(row["source_group_id"])
            self.splits[row["split"]].append(i)
            # Validate every capture before allocating any CUDA model.
            self.load(i)
        if not all(self.splits.values()) or groups["train"] & groups["validation"]:
            raise ValueError(
                "train and validation must be nonempty with disjoint source groups"
            )

    def load(self, index):
        row = self.rows[index]
        path = (self.path.parent / row["path"]).resolve()
        length = row["sequence_length"]
        if type(length) is not int or not 2 <= length <= self.max_sequence_length:
            raise ValueError("capture length exceeds the configured boundary")
        if (
            path.is_file()
            and path.stat().st_size
            > length * self.hidden_size * 8 + length * 64 + 1048576
        ):
            raise ValueError("capture file exceeds its bounded tensor geometry")
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError("capture SHA mismatch")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or set(data) != {
            "input_ids",
            "attention_mask",
            "loss_mask",
            "teacher_hidden_states",
            "teacher_identity",
            "teacher_head_sha256",
            "sample_id",
        }:
            raise ValueError("capture payload has unknown or missing fields")
        if (
            data["teacher_identity"] != self.manifest["teacher_identity"]
            or data["teacher_head_sha256"] != self.manifest["teacher_head_sha256"]
            or data["sample_id"] != row["sample_id"]
        ):
            raise ValueError("capture teacher/sample provenance mismatch")
        ids, hidden = data["input_ids"], data["teacher_hidden_states"]
        am, lm = data["attention_mask"], data["loss_mask"]
        if (
            type(length) is not int
            or not 2 <= length <= self.max_sequence_length
            or not isinstance(ids, torch.Tensor)
            or ids.dtype != torch.int64
            or ids.shape != (length,)
            or not isinstance(hidden, torch.Tensor)
            or hidden.shape != (length, self.hidden_size)
            or hidden.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or not bool(torch.isfinite(hidden).all())
        ):
            raise ValueError("capture IDs/hidden geometry, dtype or values invalid")
        if (
            bool(((ids < 0) | (ids >= self.vocab_size)).any())
            or ids_hash(ids.tolist()) != row["input_ids_sha256"]
        ):
            raise ValueError("capture token identity mismatch")
        if (
            not isinstance(am, torch.Tensor)
            or not isinstance(lm, torch.Tensor)
            or am.dtype != torch.bool
            or lm.dtype != torch.bool
            or am.shape != ids.shape
            or lm.shape != ids.shape
        ):
            raise ValueError("capture masks must be aligned boolean tensors")
        targets = int((am[:-1] & am[1:] & lm[1:]).sum())
        if (
            not targets
            or type(row["scored_tokens"]) is not int
            or targets != row["scored_tokens"]
        ):
            raise ValueError("capture target denominator mismatch")
        return data
