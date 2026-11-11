"""Narrow adapter for the pinned Transformers Trainer checkpoint helpers.

RecoveryStore owns the snapshot layout and transaction. This module only
invokes Trainer serialization, restore, and scaler APIs for that snapshot.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any


class TrainerCheckpointAdapterError(RuntimeError):
    """The pinned Trainer checkpoint interface is unavailable or invalid."""


def save_payload(trainer: Any, directory: Path) -> dict[str, Any]:
    if trainer is None:
        raise TrainerCheckpointAdapterError(
            "full recovery requires the Trainer-owned optimizer and scheduler"
        )
    required = ("_save_optimizer_and_scheduler", "_save_rng_state")
    if any(not callable(getattr(trainer, name, None)) for name in required):
        raise TrainerCheckpointAdapterError("pinned Trainer save helpers are unavailable")

    performance: dict[str, Any] = {}
    started = time.perf_counter()
    trainer._save_optimizer_and_scheduler(str(directory))
    performance["trainer_optimizer_scheduler_helper_wall_seconds"] = (
        time.perf_counter() - started
    )
    performance.update(getattr(trainer, "checkpoint_save_performance", {}))

    save_scaler = getattr(trainer, "_save_scaler", None)
    if callable(save_scaler):
        started = time.perf_counter()
        save_scaler(str(directory))
        performance["trainer_scaler_save_wall_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    trainer._save_rng_state(str(directory))
    performance["trainer_rng_helper_wall_seconds"] = time.perf_counter() - started
    performance.update(getattr(trainer, "checkpoint_save_performance", {}))

    state = getattr(trainer, "state", None)
    if state is None or not callable(getattr(state, "save_to_json", None)):
        raise TrainerCheckpointAdapterError("Trainer state cannot be saved")
    started = time.perf_counter()
    state.save_to_json(str(directory / "trainer_state.json"))
    performance["trainer_state_json_save_wall_seconds"] = time.perf_counter() - started
    return performance


def load_payload(trainer: Any, snapshot_dir: Path) -> None:
    if trainer is None:
        raise TrainerCheckpointAdapterError(
            "full recovery requires the Trainer-owned optimizer and scheduler"
        )
    if not callable(getattr(trainer, "_load_optimizer_and_scheduler", None)):
        raise TrainerCheckpointAdapterError(
            "pinned Trainer optimizer/scheduler loader is unavailable"
        )
    trainer._load_optimizer_and_scheduler(str(snapshot_dir))

    scaler_path = snapshot_dir / "scaler.pt"
    if scaler_path.exists():
        if getattr(getattr(trainer, "accelerator", None), "scaler", None) is None:
            raise TrainerCheckpointAdapterError(
                "snapshot has a scaler but current Trainer does not"
            )
        loader = getattr(trainer, "_load_scaler", None)
        if not callable(loader):
            raise TrainerCheckpointAdapterError("pinned Trainer scaler loader is unavailable")
        loader(str(snapshot_dir))
    elif getattr(getattr(trainer, "accelerator", None), "scaler", None) is not None:
        raise TrainerCheckpointAdapterError(
            "current Trainer has a scaler but snapshot does not"
        )

    trainer._load_rng_state(str(snapshot_dir))
    try:
        from transformers.trainer_callback import TrainerState

        trainer.state = TrainerState.load_from_json(str(snapshot_dir / "trainer_state.json"))
    except Exception as exc:
        raise TrainerCheckpointAdapterError(
            f"cannot restore pinned Trainer counters: {exc}"
        ) from exc
