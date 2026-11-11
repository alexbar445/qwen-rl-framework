"""Immutable BEPA JSON artifacts and exact-token replay projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from .contracts import (
    ContractError,
    SourceEpisode,
    SourcePolicy,
    SourceProjection,
    TaskSpec,
    TrustedReceipt,
    canonical_json_bytes,
    content_hash,
    replay_row_from_artifacts as _replay_row_from_contracts,
)


@dataclass(frozen=True)
class EpisodeArtifactRefs:
    raw_artifact_ref: str
    projection_artifact_ref: str
    receipt_artifact_ref: str


def project_prefix_replacement(
    *,
    episode_id: str,
    task_key: str,
    source_policy: SourcePolicy,
    row_kind: str,
    input_ids: Sequence[int],
    policy_mask: Sequence[int],
    guided_prefix_length: int,
    clean_prefix_ids: Sequence[int],
    tokenizer_fingerprint: str,
    max_context_tokens: int | None = None,
) -> SourceProjection:
    """Replace the complete guided prompt prefix with clean prompt tokens.

    The body is copied by token ID from the original sequence. No decoding,
    retokenization, truncation, or reward insertion occurs here. The source
    trace stays untouched and must be written separately as a raw artifact.
    """
    if not input_ids or len(input_ids) != len(policy_mask):
        raise ContractError("input_ids and policy_mask must be non-empty and exactly aligned")
    if not isinstance(guided_prefix_length, int) or isinstance(guided_prefix_length, bool):
        raise ContractError("guided_prefix_length must be an integer")
    if not 0 < guided_prefix_length <= len(input_ids):
        raise ContractError("guided_prefix_length must identify a non-empty prompt prefix")
    if any(isinstance(mask_value, float) or mask_value not in (0, 1, False, True) for mask_value in policy_mask):
        raise ContractError("policy_mask must contain only 0/1 values")
    if any(policy_mask[:guided_prefix_length]):
        raise ContractError("guided prefix contains policy-loss tokens; refusing to shift masks")
    body_ids = tuple(input_ids[guided_prefix_length:])
    body_mask = tuple(int(value) for value in policy_mask[guided_prefix_length:])
    if not body_ids or not any(body_mask):
        raise ContractError("projection body is empty or has no valid policy-mask token")
    clean_prefix = tuple(clean_prefix_ids)
    if not clean_prefix or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in clean_prefix):
        raise ContractError("clean_prefix_ids must be non-empty non-negative integer token IDs")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in input_ids):
        raise ContractError("input_ids must contain non-negative integer token IDs")
    if max_context_tokens is not None:
        if not isinstance(max_context_tokens, int) or isinstance(max_context_tokens, bool) or max_context_tokens <= 0:
            raise ContractError("max_context_tokens must be a positive integer")
        if len(clean_prefix) + len(body_ids) > max_context_tokens:
            raise ContractError(
                "clean-prefix projection exceeds the student context limit; refusing to truncate the episode"
            )
    if row_kind == "native" and tuple(input_ids[:guided_prefix_length]) != clean_prefix:
        raise ContractError("native episode prefix must exactly match the clean prompt token IDs")
    if row_kind not in {"native", "guided"}:
        raise ContractError("row_kind must be 'native' or 'guided'")
    projected_ids = clean_prefix + body_ids
    projected_mask = (0,) * len(clean_prefix) + body_mask
    return SourceProjection(
        episode_id=episode_id,
        task_key=task_key,
        source_policy=source_policy,
        row_kind="native_replay" if row_kind == "native" else "guided_replay",
        input_ids=projected_ids,
        policy_mask=projected_mask,
        prompt_token_count=len(clean_prefix),
        body_token_count=len(body_ids),
        body_token_hash=content_hash(list(body_ids)),
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


def project_source_episode(episode: SourceEpisode, task: TaskSpec) -> SourceProjection:
    """Project a source episode after checking task/tokenizer compatibility."""
    if episode.task_key != task.task_key:
        raise ContractError("source episode task key does not match the task compatibility key")
    if episode.tokenizer_fingerprint != task.tokenizer_fingerprint:
        raise ContractError("source episode tokenizer does not match the task tokenizer fingerprint")
    projected = project_prefix_replacement(
        episode_id=episode.episode_id,
        task_key=episode.task_key,
        source_policy=episode.source_policy,
        row_kind=episode.row_kind,
        input_ids=episode.input_ids,
        policy_mask=episode.policy_mask,
        guided_prefix_length=episode.guided_prefix_length,
        clean_prefix_ids=episode.clean_prefix_ids,
        tokenizer_fingerprint=episode.tokenizer_fingerprint,
        max_context_tokens=episode.max_context_tokens,
    )
    # Keep the original object intact; this check only confirms our source
    # envelope itself remains JSON-storable before publishing the projection.
    canonical_json_bytes(episode)
    return projected


class BepaArtifacts:
    """Content-addressed JSON artifacts with atomic directory publication."""

    def __init__(self, root: str | Path):
        """Use a BEPA state directory and publish under its ``artifacts/`` child."""
        self.bepa_root = Path(root)
        self.root = self.bepa_root / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    def path_for(self, digest: str) -> Path:
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("artifact reference must be a lowercase sha256 digest")
        return self.root / digest / "artifact.json"

    def write_json(self, value: Any) -> str:
        """Durably publish one canonical JSON value and return its sha256."""
        payload = canonical_json_bytes(value)
        digest = hashlib.sha256(payload).hexdigest()
        target_dir = self.root / digest
        target_file = target_dir / "artifact.json"
        if target_file.exists():
            existing = target_file.read_bytes()
            if existing != payload or hashlib.sha256(existing).hexdigest() != digest:
                raise RuntimeError(f"content-addressed artifact collision at {target_file}")
            return digest

        temporary_dir = Path(tempfile.mkdtemp(prefix=".artifact-", dir=self.root))
        try:
            temporary_file = temporary_dir / "artifact.json"
            with temporary_file.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_file, 0o444)
            directory_fd = os.open(temporary_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            try:
                os.rename(temporary_dir, target_dir)
            except OSError:
                if not target_file.exists():
                    raise
                existing = target_file.read_bytes()
                if existing != payload:
                    raise RuntimeError(f"content-addressed artifact collision at {target_file}")
            root_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(root_fd)
            finally:
                os.close(root_fd)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
        return digest

    def read_json(self, digest: str) -> Any:
        """Read and re-hash an artifact so a damaged file is never trusted."""
        path = self.path_for(digest)
        try:
            payload = path.read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"BEPA artifact {digest} does not exist") from exc
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"BEPA artifact {digest} is not valid JSON") from exc
        if content_hash(value) != digest or canonical_json_bytes(value) != payload:
            raise RuntimeError(f"BEPA artifact {digest} failed its content hash")
        return value

    def write_episode(
        self,
        raw: SourceEpisode,
        projection: SourceProjection,
        receipt: TrustedReceipt,
    ) -> EpisodeArtifactRefs:
        """Store raw source, clean projection, and central verifier receipt.

        The raw source is preserved independently of the training projection.
        The store checks receipt identity and exactness; construction of the
        receipt itself remains the central verifier's responsibility.
        """
        episode = raw
        if projection.episode_id != episode.episode_id or projection.task_key != episode.task_key:
            raise ContractError("projection does not identify this raw source episode")
        expected_projection = project_prefix_replacement(
            episode_id=episode.episode_id,
            task_key=episode.task_key,
            source_policy=episode.source_policy,
            row_kind=episode.row_kind,
            input_ids=episode.input_ids,
            policy_mask=episode.policy_mask,
            guided_prefix_length=episode.guided_prefix_length,
            clean_prefix_ids=episode.clean_prefix_ids,
            tokenizer_fingerprint=episode.tokenizer_fingerprint,
            max_context_tokens=episode.max_context_tokens,
        )
        if projection != expected_projection:
            raise ContractError("projection does not exactly match source prefix replacement")
        if (receipt.episode_id, receipt.task_key) != (episode.episode_id, episode.task_key):
            raise ContractError("trusted receipt does not identify this source episode")
        if not receipt.exact or receipt.verifier_id != "central_verifier":
            raise ContractError("only exact central-verifier episodes may be stored as cache candidates")
        raw_ref = self.write_json(
            {
                "schema": "bepa-source-episode-v1",
                "episode": asdict(episode),
                "raw_trace": episode.raw_trace,
            }
        )
        projection_ref = self.write_json(
            {
                "schema": "bepa-source-projection-v1",
                "projection": asdict(projection),
            }
        )
        receipt_ref = self.write_json(
            {"schema": "bepa-trusted-receipt-v1", "receipt": asdict(receipt)}
        )
        return EpisodeArtifactRefs(raw_ref, projection_ref, receipt_ref)


def replay_row_from_artifacts(
    *,
    projection: SourceProjection | dict[str, Any],
    receipt: TrustedReceipt | dict[str, Any],
    projection_artifact_ref: str,
    group_id: str,
    cache_entry_id: str,
):
    """Load JSON artifact envelopes or typed objects into a replay row."""
    projection_value: Any = projection
    receipt_value: Any = receipt
    if isinstance(projection_value, dict) and "projection" in projection_value:
        projection_value = projection_value["projection"]
    if isinstance(projection_value, dict):
        data = dict(projection_value)
        source_policy = data.get("source_policy")
        if isinstance(source_policy, dict):
            data["source_policy"] = SourcePolicy(**source_policy)
        data["input_ids"] = tuple(data["input_ids"])
        data["policy_mask"] = tuple(data["policy_mask"])
        projection_value = SourceProjection(**data)
    if isinstance(receipt_value, dict) and "receipt" in receipt_value:
        receipt_value = receipt_value["receipt"]
    if isinstance(receipt_value, dict):
        receipt_value = TrustedReceipt(**receipt_value)
    if not isinstance(projection_value, SourceProjection) or not isinstance(receipt_value, TrustedReceipt):
        raise ContractError("replay artifacts must contain a source projection and trusted receipt")
    return _replay_row_from_contracts(
        projection=projection_value,
        receipt=receipt_value,
        projection_artifact_ref=projection_artifact_ref,
        group_id=group_id,
        cache_entry_id=cache_entry_id,
    )
