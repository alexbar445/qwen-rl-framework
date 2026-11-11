"""Import one complete DDP recovery snapshot into a fresh central optimizer.

This is a CPU-only initializer. It imports the LoRA parameters, identical
per-rank AdamW moments/steps, and constant scheduler state by parameter name.
It keeps the source runtime context as provenance and never resumes its run
cursor, RNG, DDP workers, or publication lifecycle.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from qwen_rl.learner.parallel_recovery import (
    OPTIMIZER_SCHEMA,
    validate_parallel_optimizer,
)
from qwen_rl.learner.recovery import (
    RecoveryError,
    _canonical_bytes,
    _jsonable,
    _read_json,
    inspect_recovery,
)
from qwen_rl.learner.gradient_server.core import DDPCounterInitializationError


_INITIALIZATION_SCHEMA = "central-gradient-ddp-initialization/v1"
_SCHEDULER_SCHEMA = "qwen-rl-parallel-scheduler/v1"
_RNG_CONTEXT_NOTE = (
    "initial-state-only; DDP run cursor, run ID, RNG, Worker state, and "
    "Trainer lifecycle are not resumed"
)


class DDPInitializationError(RuntimeError):
    """A DDP recovery snapshot cannot safely seed a central optimizer."""


@dataclass(frozen=True)
class DDPRecoverySource:
    """A validated immutable source snapshot and its CPU scheduler state."""

    training_dir: Path
    snapshot_dir: Path
    manifest: dict[str, Any]
    receipt: dict[str, Any]
    scheduler_state: dict[str, Any]
    scheduler_rank_states: tuple[dict[str, Any], ...]
    parameter_groups: tuple[dict[str, Any], ...]
    active_parameter_group: int


def _is_safe_regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(_jsonable(value))).hexdigest()


def _peft_config_hash(model: torch.nn.Module) -> str:
    configs = getattr(model, "peft_config", {})
    if not isinstance(configs, Mapping):
        configs = {}
    identity = {
        str(name): _jsonable(config.to_dict() if callable(getattr(config, "to_dict", None)) else config)
        for name, config in configs.items()
    }
    return _canonical_hash(identity)


def _tokenizer_hash(tokenizer: Any) -> str:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise DDPInitializationError("central tokenizer cannot provide a vocabulary for recovery matching")
    identity = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "chat_template": getattr(tokenizer, "chat_template", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "vocabulary": get_vocab(),
    }
    return _canonical_hash(identity)


def _trainable_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    values: dict[str, torch.nn.Parameter] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" in name:
            if name in values:
                raise DDPInitializationError(f"central model repeats a trainable parameter name: {name}")
            if parameter.device.type != "cpu":
                raise DDPInitializationError("central DDP initialization requires CPU LoRA parameters")
            values[name] = parameter
    if not values:
        raise DDPInitializationError("central model has no trainable LoRA parameters")
    return values


def _inventory_by_name(inventory: Any, *, label: str) -> dict[str, tuple[tuple[int, ...], str]]:
    if not isinstance(inventory, list) or not inventory:
        raise DDPInitializationError(f"{label} has no trainable parameter inventory")
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for item in inventory:
        if not isinstance(item, Mapping):
            raise DDPInitializationError(f"{label} parameter inventory is malformed")
        name = item.get("name")
        shape = item.get("shape")
        dtype = item.get("dtype")
        if (not isinstance(name, str) or not name or name in result
                or not isinstance(shape, (list, tuple))
                or any(type(size) is not int or size < 1 for size in shape)
                or not isinstance(dtype, str) or not dtype):
            raise DDPInitializationError(f"{label} parameter inventory contains an invalid entry")
        result[name] = (tuple(shape), dtype)
    return result


def _require_same_inventory(
    source_inventory: Any,
    parameters: Mapping[str, torch.nn.Parameter],
    *,
    label: str,
) -> None:
    source = _inventory_by_name(source_inventory, label=label)
    current = {
        name: (tuple(parameter.shape), str(parameter.dtype))
        for name, parameter in parameters.items()
    }
    if source != current:
        raise DDPInitializationError(f"{label} LoRA names, shapes, or dtypes differ from central")


def _same_tree(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_same_tree(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_same_tree(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DDPInitializationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DDPInitializationError(f"{label} must be finite")
    return result


def _inspect_optimizer_groups(
    compatibility: Mapping[str, Any],
    manifest: Mapping[str, Any],
    parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float,
) -> tuple[list[dict[str, Any]], int]:
    source_groups = manifest.get("optimizer_parameter_groups")
    source_compat = compatibility.get("optimizer")
    if (not isinstance(source_groups, list) or not source_groups
            or not isinstance(source_compat, Mapping)
            or source_compat.get("parameter_groups") != source_groups):
        raise DDPInitializationError("DDP manifest and compatibility optimizer mappings disagree")
    _require_same_inventory(
        manifest.get("parameter_inventory"), parameters, label="DDP recovery",
    )
    compat_inventory = compatibility.get("trainable_parameters")
    if compat_inventory is not None:
        _require_same_inventory(compat_inventory, parameters, label="DDP compatibility")

    target_groups = optimizer.param_groups
    if len(target_groups) != 1:
        raise DDPInitializationError("central optimizer must own one named LoRA parameter group")
    target_ids = {id(parameter) for parameter in parameters.values()}
    target_group_ids = [id(parameter) for parameter in target_groups[0]["params"]]
    if len(target_group_ids) != len(set(target_group_ids)) or set(target_group_ids) != target_ids:
        raise DDPInitializationError("central optimizer parameter mapping is incomplete or duplicated")

    active = []
    seen_names: set[str] = set()
    for index, group in enumerate(source_groups):
        if (not isinstance(group, Mapping)
                or not isinstance(group.get("parameter_names"), list)
                or not isinstance(group.get("options"), Mapping)):
            raise DDPInitializationError("DDP optimizer group metadata is malformed")
        names = group["parameter_names"]
        if any(not isinstance(name, str) or name not in parameters for name in names):
            raise DDPInitializationError("DDP optimizer group contains an unknown parameter name")
        if seen_names.intersection(names):
            raise DDPInitializationError("DDP optimizer groups repeat parameter names")
        seen_names.update(names)
        if names:
            active.append(index)
    if seen_names != set(parameters) or len(active) != 1:
        raise DDPInitializationError(
            "central mapping requires exactly one nonempty DDP optimizer group covering every LoRA name"
        )

    source_parallel = compatibility.get("parallel")
    semantics = compatibility.get("semantics")
    if (not isinstance(source_parallel, Mapping)
            or source_parallel.get("mode") != "ddp"
            or type(source_parallel.get("world_size")) is not int
            or source_parallel["world_size"] < 2):
        raise DDPInitializationError("source recovery is not a multi-rank DDP snapshot")
    if not isinstance(semantics, Mapping) or semantics.get("model_backend") != "ternary":
        raise DDPInitializationError("source DDP recovery is not the compatible T2 Learner")

    target = target_groups[0]
    target_lr = _float(target.get("lr"), "central AdamW learning rate")
    target_betas = [_float(item, "central AdamW beta") for item in target.get("betas", ())]
    target_eps = _float(target.get("eps"), "central AdamW epsilon")
    target_weight_decay = _float(target.get("weight_decay"), "central AdamW weight decay")
    source_hyper = source_compat.get("hyperparameters")
    if not isinstance(source_hyper, Mapping):
        raise DDPInitializationError("DDP optimizer hyperparameters are missing")
    expected_hyper = {
        "learning_rate": target_lr,
        "betas": target_betas,
        "eps": target_eps,
        "weight_decay": target_weight_decay,
        "max_grad_norm": _float(max_grad_norm, "central max_grad_norm"),
    }
    for key, expected in expected_hyper.items():
        actual = source_hyper.get(key)
        if key == "betas":
            if not isinstance(actual, (list, tuple)) or [_float(item, f"DDP {key}") for item in actual] != expected:
                raise DDPInitializationError("DDP and central AdamW beta values differ")
        elif _float(actual, f"DDP optimizer {key}") != expected:
            raise DDPInitializationError(f"DDP and central optimizer {key} values differ")

    for index, group in enumerate(source_groups):
        options = group["options"]
        required_options = {
            "lr": target_lr,
            "initial_lr": target_lr,
            "betas": target_betas,
            "eps": target_eps,
            "weight_decay": target_weight_decay,
        }
        for key, expected in required_options.items():
            actual = options.get(key)
            if key == "betas":
                if not isinstance(actual, (list, tuple)) or [_float(item, f"DDP {key}") for item in actual] != expected:
                    raise DDPInitializationError(f"DDP optimizer group {index} beta values differ")
            elif _float(actual, f"DDP optimizer group {index} {key}") != expected:
                raise DDPInitializationError(f"DDP optimizer group {index} {key} differs")
        for key, expected in (
            ("amsgrad", False),
            ("maximize", False),
            ("capturable", False),
            ("differentiable", False),
        ):
            if options.get(key) is not expected:
                raise DDPInitializationError(
                    f"DDP AdamW option {key} is incompatible with central CPU AdamW"
                )
        if options.get("decoupled_weight_decay") is not True:
            raise DDPInitializationError("DDP optimizer group is not decoupled AdamW")
    return copy.deepcopy(source_groups), active[0]


def _inspect_scheduler(
    source_dir: Path,
    compatibility: Mapping[str, Any],
    parameter_groups: list[dict[str, Any]],
    active_group: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scheduler_compat = compatibility.get("scheduler")
    if (not isinstance(scheduler_compat, Mapping)
            or scheduler_compat.get("type") != "constant"):
        raise DDPInitializationError("source scheduler is not the constant schedule")
    if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        raise DDPInitializationError("central scheduler must be LambdaLR/constant")

    path = source_dir / "scheduler.pt"
    if not _is_safe_regular_file(path):
        raise DDPInitializationError("DDP recovery scheduler payload is missing or unsafe")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DDPInitializationError(
            f"cannot load DDP scheduler payload: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != _SCHEDULER_SCHEMA:
        raise DDPInitializationError("DDP scheduler payload schema is unsupported")
    parallel = compatibility.get("parallel", {})
    world_size = parallel.get("world_size") if isinstance(parallel, Mapping) else None
    states = payload.get("rank_states")
    if (not isinstance(states, list) or len(states) != world_size
            or not states or any(not isinstance(state, Mapping) for state in states)):
        raise DDPInitializationError("DDP scheduler payload does not contain every rank")
    first = dict(states[0])
    if any(not _same_tree(first, state) for state in states[1:]):
        raise DDPInitializationError("DDP scheduler states disagree across ranks")

    group_count = len(parameter_groups)
    normalized = copy.deepcopy(first)
    for key in ("base_lrs", "_last_lr", "lr_lambdas"):
        values = first.get(key)
        if not isinstance(values, list) or len(values) != group_count:
            raise DDPInitializationError(f"DDP scheduler {key} does not map to optimizer groups")
        selected = values[active_group]
        if any(not _same_tree(selected, value) for value in values):
            raise DDPInitializationError(f"DDP scheduler {key} differs across optimizer groups")
        normalized[key] = [copy.deepcopy(selected)]

    last_epoch = normalized.get("last_epoch")
    step_count = normalized.get("_step_count")
    if type(last_epoch) is not int or last_epoch < 0:
        raise DDPInitializationError("DDP scheduler last_epoch is invalid")
    if type(step_count) is not int or step_count < 1 or step_count != last_epoch + 1:
        raise DDPInitializationError("DDP scheduler step count is inconsistent")
    active_options = parameter_groups[active_group]["options"]
    active_lr = _float(active_options.get("lr"), "DDP active optimizer learning rate")
    if len(optimizer.param_groups) != 1 or _float(
        optimizer.param_groups[0].get("lr"), "central optimizer learning rate",
    ) != active_lr:
        raise DDPInitializationError("DDP and central scheduler learning rates differ")
    base_lrs = normalized["base_lrs"]
    last_lrs = normalized["_last_lr"]
    if _float(base_lrs[0], "DDP scheduler base learning rate") != active_lr:
        raise DDPInitializationError("DDP scheduler base learning rate differs")
    if _float(last_lrs[0], "DDP scheduler current learning rate") != active_lr:
        raise DDPInitializationError("DDP scheduler current learning rate differs")
    return [dict(state) for state in states], normalized


def inspect_ddp_recovery_source(
    training_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    tokenizer: Any,
    base_artifact_id: str,
    base_artifact_sha256: str,
    correction_cap: float,
    max_grad_norm: float,
) -> DDPRecoverySource:
    """Validate the source RecoveryStore snapshot without changing it."""
    raw_root = Path(training_dir).expanduser()
    if not raw_root.is_absolute():
        raise DDPInitializationError("DDP source training directory must be absolute")
    if raw_root.is_symlink():
        raise DDPInitializationError("DDP source training directory may not be a symlink")
    try:
        root = raw_root.resolve(strict=True)
        inspection = inspect_recovery(root)
    except (OSError, RecoveryError) as exc:
        raise DDPInitializationError(f"cannot inspect DDP RecoveryStore: {exc}") from exc
    if inspection.get("status") != "resume":
        raise DDPInitializationError(
            f"DDP source RecoveryStore is {inspection.get('status')}: {inspection.get('error')}"
        )
    snapshot_path = Path(str(inspection.get("snapshot_path", "")))
    if snapshot_path.is_symlink() or not snapshot_path.is_dir():
        raise DDPInitializationError("DDP RecoveryStore snapshot path is unsafe")

    try:
        manifest = _read_json(snapshot_path / "manifest.json")
        state = _read_json(snapshot_path / "state.json")
        trainer_state = _read_json(snapshot_path / "trainer_state.json")
    except RecoveryError as exc:
        raise DDPInitializationError(f"cannot read DDP RecoveryStore metadata: {exc}") from exc
    for key in ("lineage_id", "snapshot_id", "policy_version", "engine_epoch"):
        if manifest.get(key) != inspection.get(key):
            raise DDPInitializationError(f"DDP RecoveryStore manifest {key} disagrees")
    for key in ("lineage_id", "policy_version", "engine_epoch"):
        if state.get(key) != inspection.get(key):
            raise DDPInitializationError(f"DDP RecoveryStore state {key} disagrees")
    if manifest.get("update_context") != inspection.get("update_context"):
        raise DDPInitializationError("DDP runtime context differs from the verified RecoveryStore manifest")
    if manifest.get("schema") != state.get("schema"):
        raise DDPInitializationError("DDP RecoveryStore snapshot schemas disagree")
    compatibility = inspection.get("compatibility")
    if not isinstance(compatibility, Mapping) or manifest.get("compatibility") != compatibility:
        raise DDPInitializationError("DDP RecoveryStore compatibility metadata is inconsistent")

    parameters = _trainable_parameters(model)
    parameter_groups, active_group = _inspect_optimizer_groups(
        compatibility,
        manifest,
        parameters,
        optimizer,
        max_grad_norm=max_grad_norm,
    )
    base = compatibility.get("base")
    quantization = base.get("quantization") if isinstance(base, Mapping) else None
    if (not isinstance(base, Mapping)
            or base.get("base_fingerprint") != base_artifact_sha256
            or base.get("artifact_fingerprint") != base_artifact_sha256
            or not isinstance(quantization, Mapping)
            or quantization.get("mode") != "ternary"
            or quantization.get("artifact_id") != base_artifact_id
            or quantization.get("artifact_sha256") != base_artifact_sha256):
        raise DDPInitializationError("DDP source base artifact identity differs from central")
    objective = compatibility.get("objective")
    if not isinstance(objective, Mapping):
        raise DDPInitializationError("DDP objective identity is missing")
    expected_objective = {
        "family": "grpo",
        "loss_type": "dapo",
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "scale_rewards": "group",
        "kl_beta": 0.0,
        "loss_aggregation": "global-token-mean",
    }
    if any(objective.get(key) != value for key, value in expected_objective.items()):
        raise DDPInitializationError("DDP objective semantics are incompatible with central DAPO")
    if _float(objective.get("importance_sampling_cap"), "DDP correction cap") != _float(
        correction_cap, "central correction cap",
    ):
        raise DDPInitializationError("DDP and central importance-sampling caps differ")
    if compatibility.get("tokenizer_sha256") != _tokenizer_hash(tokenizer):
        raise DDPInitializationError("DDP and central tokenizer identities differ")
    peft = compatibility.get("peft")
    if (not isinstance(peft, Mapping)
            or peft.get("config_sha256") != _peft_config_hash(model)):
        raise DDPInitializationError("DDP and central PEFT configurations differ")

    source_groups, normalized_scheduler = _inspect_scheduler(
        snapshot_path,
        compatibility,
        parameter_groups,
        active_group,
        optimizer,
        scheduler,
    )
    policy_version = inspection.get("policy_version")
    optimizer_steps = normalized_scheduler["last_epoch"]
    scheduler_steps = normalized_scheduler["last_epoch"]
    if type(policy_version) is not int or policy_version < optimizer_steps:
        raise DDPInitializationError("DDP policy version precedes its optimizer step state")
    initial_policy_version = policy_version - optimizer_steps
    trainer_counters = manifest.get("trainer_counters")
    if (not isinstance(trainer_counters, Mapping)
            or trainer_state.get("global_step") != trainer_counters.get("trainer_global_step")):
        raise DDPInitializationError("DDP trainer counters are missing or inconsistent")

    selected_compatibility = {
        "base": copy.deepcopy(base),
        "objective": copy.deepcopy(dict(objective)),
        "optimizer": {
            "class": compatibility.get("optimizer", {}).get("class"),
            "hyperparameters": copy.deepcopy(
                compatibility.get("optimizer", {}).get("hyperparameters", {}),
            ),
            "parameter_groups": copy.deepcopy(parameter_groups),
        },
        "parallel": copy.deepcopy(dict(compatibility["parallel"])),
        "peft": copy.deepcopy(dict(peft)),
        "scheduler": copy.deepcopy(dict(compatibility["scheduler"])),
        "semantics": copy.deepcopy(dict(compatibility.get("semantics", {}))),
        "tokenizer_sha256": compatibility.get("tokenizer_sha256"),
    }
    receipt = {
        "schema": _INITIALIZATION_SCHEMA,
        "mode": "ddp_full_state_as_central_initialization",
        "source_training_dir": str(root),
        "source_lineage_id": inspection["lineage_id"],
        "source_snapshot_id": inspection["snapshot_id"],
        "source_manifest_sha256": inspection["manifest_sha256"],
        "source_policy_version": policy_version,
        "source_engine_epoch": inspection.get("engine_epoch"),
        "source_optimizer_steps": optimizer_steps,
        "source_scheduler_steps": scheduler_steps,
        "source_initial_policy_version": initial_policy_version,
        "source_parallel": copy.deepcopy(dict(compatibility["parallel"])),
        "source_optimizer_groups": copy.deepcopy(parameter_groups),
        "source_scheduler_rank_states": copy.deepcopy(source_groups),
        "source_trainer_counters": copy.deepcopy(dict(trainer_counters)),
        "source_runtime_context": copy.deepcopy(inspection.get("update_context")),
        "source_compatibility": selected_compatibility,
        "mapping_contract": {
            "parameter_mapping": "exact_trainable_names_shapes_dtypes",
            "optimizer_state_mapping": "parameter_name_after_rank_group_id_resolution",
            "rank_state_policy": "all_ddp_ranks_must_match_exactly",
            "central_optimizer_groups": 1,
            "active_source_group": active_group,
            "mapped_parameter_count": len(parameters),
        },
        "resume_contract": _RNG_CONTEXT_NOTE,
    }
    try:
        _canonical_bytes(receipt)
    except RecoveryError as exc:
        raise DDPInitializationError(f"DDP initialization evidence is not JSON-safe: {exc}") from exc
    return DDPRecoverySource(
        training_dir=root,
        snapshot_dir=snapshot_path,
        manifest=manifest,
        receipt=receipt,
        scheduler_state=normalized_scheduler,
        scheduler_rank_states=tuple(copy.deepcopy(source_groups)),
        parameter_groups=tuple(copy.deepcopy(parameter_groups)),
        active_parameter_group=active_group,
    )


def _rank_state_by_name(
    rank_optimizer: Any,
    parameter_groups: tuple[dict[str, Any], ...],
    *,
    rank: int,
) -> dict[str, dict[str, Any]]:
    if (not isinstance(rank_optimizer, Mapping)
            or not isinstance(rank_optimizer.get("param_groups"), list)
            or not isinstance(rank_optimizer.get("state"), Mapping)
            or len(rank_optimizer["param_groups"]) != len(parameter_groups)):
        raise DDPInitializationError(f"DDP rank {rank} optimizer state_dict is malformed")
    id_to_name: dict[Any, str] = {}
    for index, (saved_group, source_group) in enumerate(
        zip(rank_optimizer["param_groups"], parameter_groups, strict=True)
    ):
        names = source_group["parameter_names"]
        ids = saved_group.get("params") if isinstance(saved_group, Mapping) else None
        if not isinstance(ids, list) or len(ids) != len(names):
            raise DDPInitializationError(f"DDP rank {rank} optimizer group {index} mapping is incomplete")
        for parameter_id, name in zip(ids, names, strict=True):
            if parameter_id in id_to_name:
                raise DDPInitializationError(f"DDP rank {rank} optimizer repeats a parameter ID")
            id_to_name[parameter_id] = name
    state_by_name: dict[str, dict[str, Any]] = {}
    for parameter_id, state in rank_optimizer["state"].items():
        name = id_to_name.get(parameter_id)
        if name is None or not isinstance(state, Mapping):
            raise DDPInitializationError(f"DDP rank {rank} optimizer contains an unmapped parameter state")
        if name in state_by_name:
            raise DDPInitializationError(f"DDP rank {rank} optimizer repeats parameter state for {name}")
        state_by_name[name] = dict(state)
    if set(state_by_name) != set(id_to_name.values()):
        raise DDPInitializationError(f"DDP rank {rank} optimizer state coverage is incomplete")
    return state_by_name


def _step_value(state: Mapping[str, Any], name: str) -> int:
    value = state.get("step")
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise DDPInitializationError(f"AdamW step state is missing or invalid for {name}")
    number = float(value.item())
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise DDPInitializationError(f"AdamW step count is invalid for {name}")
    return int(number)


def import_ddp_recovery_state(
    source: DDPRecoverySource,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    """Restore validated source parameters/moments/scheduler into fresh CPU objects."""
    parameters = _trainable_parameters(model)
    if optimizer.state:
        raise DDPInitializationError("central AdamW already has state; DDP seeding is empty-checkpoint only")
    if len(optimizer.param_groups) != 1:
        raise DDPInitializationError("central optimizer must have one parameter group")
    current_scheduler_state = scheduler.state_dict()
    if current_scheduler_state.get("last_epoch") != 0 or current_scheduler_state.get("_step_count") != 1:
        raise DDPInitializationError("central scheduler is not at its empty initialization state")

    adapter_path = source.snapshot_dir / "adapter_state.safetensors"
    optimizer_path = source.snapshot_dir / "optimizer.pt"
    if not _is_safe_regular_file(adapter_path) or not _is_safe_regular_file(optimizer_path):
        raise DDPInitializationError("DDP adapter or optimizer payload is missing or unsafe")
    try:
        from safetensors.torch import load_file

        adapter_state = load_file(str(adapter_path), device="cpu")
        optimizer_payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DDPInitializationError(
            f"cannot load DDP adapter/optimizer state: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(adapter_state, Mapping) or set(adapter_state) != set(parameters):
        raise DDPInitializationError("DDP adapter state names differ from central LoRA names")
    for name, parameter in parameters.items():
        value = adapter_state[name]
        if (not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or not bool(torch.isfinite(value).all())):
            raise DDPInitializationError(f"DDP adapter tensor is invalid: {name}")

    if not isinstance(optimizer_payload, Mapping) or optimizer_payload.get("schema") != OPTIMIZER_SCHEMA:
        raise DDPInitializationError("DDP optimizer payload schema is unsupported")
    if (optimizer_payload.get("parallelism") != "ddp"
            or optimizer_payload.get("world_size") != source.receipt["source_parallel"]["world_size"]
            or optimizer_payload.get("parameter_groups") != list(source.parameter_groups)):
        raise DDPInitializationError("DDP optimizer topology or named groups differ from RecoveryStore metadata")
    try:
        validate_parallel_optimizer(
            dict(optimizer_payload),
            [copy.deepcopy(group) for group in source.parameter_groups],
            model,
        )
    except RecoveryError as exc:
        raise DDPInitializationError(f"DDP optimizer mapping failed schema validation: {exc}") from exc
    ranks = optimizer_payload.get("rank_states")
    if not isinstance(ranks, list) or len(ranks) != source.receipt["source_parallel"]["world_size"]:
        raise DDPInitializationError("DDP optimizer payload is missing rank state")
    by_rank = [
        _rank_state_by_name(rank_state, source.parameter_groups, rank=index)
        for index, rank_state in enumerate(ranks)
    ]
    reference = by_rank[0]
    if any(not _same_tree(reference, other) for other in by_rank[1:]):
        raise DDPInitializationError("DDP AdamW moments or steps disagree across ranks")

    optimizer_state_by_name: dict[str, dict[str, Any]] = {}
    expected_steps = source.receipt["source_optimizer_steps"]
    for name, parameter in parameters.items():
        state = reference[name]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise DDPInitializationError(f"AdamW state fields are incomplete or unsupported for {name}")
        if _step_value(state, name) != expected_steps:
            raise DDPInitializationError(f"AdamW and scheduler step counts disagree for {name}")
        converted = {}
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if (not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                    or not bool(torch.isfinite(value).all())):
                raise DDPInitializationError(f"AdamW {key} is invalid for {name}")
            converted[key] = value.detach().clone()
        converted["step"] = state["step"].detach().clone()
        optimizer_state_by_name[name] = converted

    # Recheck the pointer after all payload reads. RecoveryStore snapshots are
    # immutable, but a changed latest pointer means this import no longer names
    # the explicitly inspected source state.
    latest = inspect_recovery(source.training_dir)
    if (latest.get("status") != "resume"
            or latest.get("snapshot_id") != source.receipt["source_snapshot_id"]
            or latest.get("lineage_id") != source.receipt["source_lineage_id"]):
        raise DDPInitializationError("DDP latest recovery snapshot changed during initialization")

    previous_model = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    previous_optimizer = dict(optimizer.state)
    previous_scheduler = copy.deepcopy(scheduler.state_dict())
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(adapter_state[name])
        optimizer.state.clear()
        for name, parameter in parameters.items():
            optimizer.state[parameter] = optimizer_state_by_name[name]
        scheduler.load_state_dict(copy.deepcopy(source.scheduler_state))
    except BaseException as exc:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(previous_model[name])
        optimizer.state.clear()
        optimizer.state.update(previous_optimizer)
        scheduler.load_state_dict(previous_scheduler)
        raise DDPInitializationError(
            f"DDP state could not be installed atomically: {type(exc).__name__}: {exc}"
        ) from exc
    return copy.deepcopy(source.receipt)


def persist_ddp_initial_counters(core: Any, receipt: Mapping[str, Any]) -> None:
    """Write imported policy/step counters into the central genesis snapshot."""
    try:
        core.initialize_ddp_counters(receipt, commit=True)
    except DDPCounterInitializationError as exc:
        raise DDPInitializationError(str(exc)) from exc


def validate_ddp_initialized_core(core: Any, receipt: Mapping[str, Any]) -> None:
    """Reject a central checkpoint that predates or misstates the DDP seed."""
    try:
        core.initialize_ddp_counters(receipt, commit=False)
    except DDPCounterInitializationError as exc:
        raise DDPInitializationError(str(exc)) from exc


__all__ = [
    "DDPInitializationError",
    "DDPRecoverySource",
    "inspect_ddp_recovery_source",
    "import_ddp_recovery_state",
    "persist_ddp_initial_counters",
    "validate_ddp_initialized_core",
]
