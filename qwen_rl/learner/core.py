"""Backend-neutral learner state, trajectory, publication, and recovery API."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .batch import validate_batch
from .checkpoints import ADAPTER_FILES
from .publication import AdapterPublicationStore
from .recovery import RecoveryError, source_provenance


class LearnerEngineCore:
    """Small shared boundary; model construction and optimizer updates stay in subclasses."""

    def _configure_publication_state(self) -> None:
        self._publication_store = AdapterPublicationStore(self.publication_dir, self.checkpoint_dir)
        backend = self.config.get("model_backend", "default")
        self._publication_store.native_publications_enabled = backend == "ternary"
        # These state views are also used by recovery and are kept on the engine
        # so the persisted publication identity remains part of the engine boundary.
        self._published_versions = self._publication_store.published_versions
        self._policy_references = self._publication_store.policy_references
        self._publication_lock = self._publication_store.lock

    def _validated_batch(
        self,
        trajectories: list[dict[str, Any]],
        *,
        learner_start_version: int | None = None,
    ) -> tuple[list[dict[str, Any]], int, dict[Any, str], dict[Any, int]]:
        if learner_start_version is None:
            learner_start_version = self.policy_version
        return validate_batch(
            trajectories,
            learner_start_version=learner_start_version,
            max_sequence_length=self.max_sequence_length,
            group_size=self.group_size,
            groups_per_update=self.groups_per_update,
        )

    @property
    def recovery_metadata(self) -> dict[str, Any]:
        """Current durable recovery identity and latest run/update cursor."""
        return copy.deepcopy(self._recovery_metadata)

    @property
    def recovery(self) -> dict[str, Any]:
        """Short read-only view used by runtime integrations."""
        return self.recovery_metadata

    def _refresh_recovery_metadata(
        self,
        *,
        snapshot_sha256: str | None,
        snapshot_id: str | None,
        snapshot_path: str | None,
        compatibility_sha256: str | None = None,
    ) -> None:
        store = self.recovery_store
        self._recovery_metadata = {
            "schema": "qwen-rl-training-state/v1",
            "enabled": store is not None,
            "mode": self._recovery_mode,
            "resumed": self._recovery_resumed,
            "lineage_id": None if store is None else store.lineage_id,
            "engine_epoch": self._engine_epoch,
            "policy_version": self.policy_version,
            "durable_policy_version": self._durable_policy_version,
            "snapshot_id": snapshot_id,
            "snapshot_sha256": snapshot_sha256,
            "snapshot_path": snapshot_path,
            "loaded_snapshot": copy.deepcopy(self._loaded_snapshot),
            "update_context": copy.deepcopy(self._last_update_context),
            "compatibility_sha256": compatibility_sha256,
            "dirty": self._recovery_dirty,
        }

    def _write_recovery_snapshot(
        self,
        *,
        reason: str,
        update_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self.recovery_store is None or self._recovery_compatibility is None or self._engine_epoch is None:
            raise RecoveryError("training recovery is not configured")
        with self._recovery_lock:
            durable = self._durable_policy_version
            if durable is not None and self.policy_version not in (durable, durable + 1):
                raise RecoveryError(
                    f"policy version {self.policy_version} cannot advance durable recovery step {durable}"
                )
            publication_hashes = self._publication_store.hashes_for(self.policy_version)
            if publication_hashes is not None and all(
                value is None for value in publication_hashes.values()
            ):
                publication_hashes = None
            snapshot = self.recovery_store.save(
                model=self.model,
                trainer=self.trainer,
                compatibility=self._recovery_compatibility,
                policy_version=self.policy_version,
                engine_epoch=self._engine_epoch,
                update_context=update_context,
                reason=reason,
                publication_hashes=publication_hashes,
                source_provenance=source_provenance(),
            )
            self._last_update_context = copy.deepcopy(update_context)
            self._durable_policy_version = self.policy_version
            if self.recovery_store.pointer is not None:
                self._refresh_recovery_metadata(
                    snapshot_sha256=self.recovery_store.pointer["manifest_sha256"],
                    snapshot_id=self.recovery_store.pointer["snapshot_id"],
                    snapshot_path=snapshot["path"],
                    compatibility_sha256=snapshot["compatibility_sha256"],
                )
            return snapshot

    def save_recovery_snapshot(
        self,
        *,
        update_context: dict[str, Any] | None = None,
        reason: str = "explicit",
    ) -> dict[str, Any]:
        """Persist current model/optimizer/RNG state at the current policy step."""
        if self.recovery_store is None:
            raise RecoveryError("training recovery is not configured")
        if self._recovery_dirty or self.policy_version != self._durable_policy_version:
            raise RecoveryError("cannot snapshot an update whose durable commit failed; restart from latest state")
        return self._write_recovery_snapshot(reason=reason, update_context=update_context)

    def _adopt_restored_publication(
        self,
        expected_hashes: dict[str, Any] | None,
    ) -> None:
        """Validate the durable adapter, then discard artifacts newer than recovery."""
        store = self.recovery_store
        pointer = None if store is None else store.pointer
        if (
            store is None
            or store.resumed is not True
            or self._recovery_mode != "resumed"
            or self._recovery_resumed is not True
        ):
            raise RecoveryError("publication cleanup requires a fully resumed recovery snapshot")
        if (
            not isinstance(pointer, dict)
            or isinstance(pointer.get("policy_version"), bool)
            or pointer.get("policy_version") != self.policy_version
            or isinstance(self._durable_policy_version, bool)
            or self._durable_policy_version != self.policy_version
        ):
            raise RecoveryError("recovery pointer and durable policy version do not match the loaded policy")
        if expected_hashes is not None and (
            not isinstance(expected_hashes, dict)
            or set(expected_hashes) != set(ADAPTER_FILES)
        ):
            raise RecoveryError("full recovery snapshot has invalid adapter identity metadata")

        try:
            self._publication_store.adopt_restored(
                self.model, self.evidence, self.policy_version, expected_hashes
            )
            cleanup = self._publication_store.discard_uncommitted_after(self._durable_policy_version)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RecoveryError(str(exc)) from exc
        self.evidence["recovery_publication_cleanup"] = cleanup

    def _refresh_after_failed_durability(self) -> None:
        pointer = None if self.recovery_store is None else self.recovery_store.pointer
        snapshot_path = None if self.recovery_store is None else self.recovery_store.current_snapshot_path
        self._refresh_recovery_metadata(
            snapshot_sha256=None if pointer is None else pointer.get("manifest_sha256"),
            snapshot_id=None if pointer is None else pointer.get("snapshot_id"),
            snapshot_path=None if snapshot_path is None else str(snapshot_path),
            compatibility_sha256=self._recovery_metadata.get("compatibility_sha256"),
        )

    @property
    def retained_policy_versions(self) -> tuple[int, ...]:
        return self._publication_store.retained_versions()

    def acquire_policy(self, version: int) -> None:
        self._publication_store.acquire(version)

    def release_policy(self, version: int) -> None:
        self._publication_store.release(version, current_version=self.policy_version)

    def publish(
        self, *, parameter_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        backend = self.config.get("model_backend", self.evidence.get("model_backend", "default"))
        identity = None
        if backend == "ternary":
            identity = self.config.get("artifact_identity")
            if not isinstance(identity, dict):
                identity = getattr(self, "_ternary_identity", None)
            if not isinstance(identity, dict):
                identity_path = self.config.get("artifact_identity_path")
                if not isinstance(identity_path, str) or not identity_path:
                    raise RuntimeError(
                        "native T2 publication requires an artifact identity document or path"
                    )
                from qwen_rl.contracts.ternary import load_artifact_identity

                identity = load_artifact_identity(identity_path, verify_files=False).as_dict()
            from qwen_rl.contracts.ternary import validate_artifact_identity

            identity = validate_artifact_identity(identity)
        publication_kwargs = {
            "model_backend": backend,
            "artifact_identity": identity,
        }
        if parameter_state is not None:
            publication_kwargs["parameter_state"] = parameter_state
        return self._publication_store.publish(
            self.model, self.evidence, self.policy_version, **publication_kwargs,
        )
