"""Prepare an isolated rank-32 view of the durable zero-residual SFT baseline."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..rollout.groups import PolicySnapshot
from ..runtime.manifest import read_json, write_json_atomic
from .policy import EvaluationPolicy, _verify_standard_lora_config, export_zero_residual_sft_policy
from .state import EvaluationRegistry


_NATIVE_FILES = {"policy.lora.ninfer", "policy.lora.ninfer.json"}
_TRANSITION_SCHEMA = "qwen-rl-rank32-evaluation-transition/v1"


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file, not a symlink")
    return path.resolve(strict=True)


def _target_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    from ..contracts.ternary import validate_artifact_identity

    identity = validate_artifact_identity(value)
    model = identity.get("model")
    adapter = identity.get("learner", {}).get("adapter", {})
    artifact = identity.get("source", {}).get("ninfer_artifact", {})
    if not isinstance(model, dict) or model.get("id") != "bonsai2-27b-arc-k2-mtp":
        raise ValueError("rank-32 evaluation baseline requires the Bonsai2 T2 model identity")
    if (adapter.get("rank"), adapter.get("alpha"), adapter.get("scale_alpha_over_rank")) != (32, 32, 1.0):
        raise ValueError("rank-32 evaluation baseline requires identity rank/alpha 32/32 and scale 1")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str):
        raise ValueError("rank-32 evaluation identity omitted its pinned base SHA")
    return identity


def _source_registry_snapshot(
    source_state_dir: Path,
    *,
    lineage_id: str,
    evalset_sha256: str,
    task_ids: Iterable[str],
    milestone_interval: int,
) -> dict[str, Any]:
    _regular_file(source_state_dir / "registry.json", "rank-16 evaluation registry")
    registry = EvaluationRegistry(
        source_state_dir / "registry.json",
        lineage_id=lineage_id,
        evalset_sha256=evalset_sha256,
        task_ids=task_ids,
        milestone_interval=milestone_interval,
    )
    snapshot = registry.snapshot()
    baseline_id = snapshot.get("baseline_request_id")
    baseline = snapshot.get("requests", {}).get(baseline_id) if baseline_id else None
    if not isinstance(baseline, dict) or baseline.get("kind") != "baseline":
        raise ValueError("rank-16 evaluation history has no recorded SFT baseline request")
    return snapshot


def _expand_rank16_factors(source_dir: Path, destination: Path, *, identity: dict[str, Any]) -> dict[str, Any]:
    """Expand standard rank-16 A/B tensors by zero-padding the new factor dimensions."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    import torch

    from ..learner.native_export import _identity_contract, _state_keys
    from ..learner.publication import tensor_digest

    config_path = _regular_file(source_dir / "adapter_config.json", "rank-16 SFT adapter config")
    state_path = _regular_file(source_dir / "adapter_model.safetensors", "rank-16 SFT adapter tensors")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("rank-16 SFT adapter config must be an object")
    _verify_standard_lora_config(config)
    if (type(config.get("r")) is not int or type(config.get("lora_alpha")) is not int
            or (config["r"], config["lora_alpha"]) != (16, 16)
            or config.get("lora_dropout") != 0.0
            or config.get("rank_pattern") not in (None, {}, [])
            or config.get("alpha_pattern") not in (None, {}, [])):
        raise ValueError("source SFT adapter must use uniform rank-16/alpha-16 LoRA with zero dropout")

    model_contract = _identity_contract(identity)
    expected_keys = _state_keys(
        model_contract["layer_types"], include_gdn=model_contract.get("gdn_projections", False),
    )
    with safe_open(str(state_path), framework="pt", device="cpu") as stream:
        metadata = stream.metadata()
    state = load_file(str(state_path), device="cpu")
    if set(state) != expected_keys:
        missing = sorted(expected_keys - set(state))
        extra = sorted(set(state) - expected_keys)
        raise ValueError(f"rank-16 SFT target-module inventory differs from the pinned T2 identity: missing={missing[:4]}, extra={extra[:4]}")

    expanded: dict[str, torch.Tensor] = {}
    pair_prefixes: dict[str, set[str]] = {}
    nonzero_b = 0
    for key, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.layout != torch.strided or tensor.ndim != 2
                or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all().item())):
            raise ValueError(f"rank-16 SFT factor is not a finite dense CPU matrix: {key}")
        if key.endswith(".lora_A.weight"):
            prefix = key[:-len(".lora_A.weight")]
            pair_prefixes.setdefault(prefix, set()).add("A")
            if tensor.shape[0] != 16:
                raise ValueError(f"rank-16 SFT A factor has the wrong rank: {key}")
            padded = torch.zeros((32, tensor.shape[1]), dtype=tensor.dtype, device="cpu")
            padded[:16].copy_(tensor)
            if torch.count_nonzero(padded[16:]).item() != 0:
                raise RuntimeError(f"new rank-32 A rows are not zero: {key}")
            expanded[key] = padded
        elif key.endswith(".lora_B.weight"):
            prefix = key[:-len(".lora_B.weight")]
            pair_prefixes.setdefault(prefix, set()).add("B")
            if tensor.shape[1] != 16:
                raise ValueError(f"rank-16 SFT B factor has the wrong rank: {key}")
            nonzero = int(torch.count_nonzero(tensor).item())
            nonzero_b += nonzero
            if nonzero:
                raise ValueError("persisted SFT baseline contains nonzero LoRA B; refusing to label it zero-residual")
            padded = torch.zeros((tensor.shape[0], 32), dtype=tensor.dtype, device="cpu")
            padded[:, :16].copy_(tensor)
            if torch.count_nonzero(padded).item() != 0:
                raise RuntimeError(f"expanded rank-32 B factor is not zero: {key}")
            expanded[key] = padded
        else:
            raise ValueError(f"rank-16 SFT adapter contains an unsupported tensor key: {key}")
    if not pair_prefixes or any(sides != {"A", "B"} for sides in pair_prefixes.values()):
        raise ValueError("rank-16 SFT adapter has an incomplete LoRA A/B inventory")

    expanded_config = dict(config)
    expanded_config["r"] = 32
    expanded_config["lora_alpha"] = 32
    if "rank_pattern" in expanded_config:
        expanded_config["rank_pattern"] = {}
    if "alpha_pattern" in expanded_config:
        expanded_config["alpha_pattern"] = {}
    digest = tensor_digest(expanded.items())
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "adapter_config.json").write_text(
        json.dumps(expanded_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    save_file(expanded, str(destination / "adapter_model.safetensors"), metadata=metadata)
    return {
        "rank": 32,
        "alpha": 32,
        "source_rank": 16,
        "source_alpha": 16,
        "module_count": len(pair_prefixes),
        "factor_tensor_count": len(expanded),
        "source_nonzero_b_elements": nonzero_b,
        "new_a_rows_zero": True,
        "all_b_elements_zero": True,
        "adapter_tensor_digest": digest,
    }


def _read_existing_transition(
    target_root: Path,
    *,
    source_state_dir: Path,
    lineage_id: str,
    checkpoint_count: int,
    evalset_sha256: str,
    target_identity: dict[str, Any],
    milestone_interval: int,
) -> dict[str, Any] | None:
    receipt_path = target_root / "rank32-transition.json"
    if not target_root.exists():
        return None
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise FileExistsError("rank-32 evaluation target exists without a valid transition receipt")
    receipt = read_json(receipt_path)
    adapter = target_identity["learner"]["adapter"]
    artifact = target_identity["source"]["ninfer_artifact"]
    expected = {
        "schema": _TRANSITION_SCHEMA,
        "lineage_id": lineage_id,
        "source_state_dir": str(source_state_dir),
        "target_identity": {
            "rank": adapter["rank"], "alpha": adapter["alpha"],
            "artifact_id": artifact["artifact_id"], "artifact_sha256": artifact["sha256"],
            "conversion_report_sha256": target_identity["source"]["conversion_report"]["sha256"],
        },
        "evalset_sha256": evalset_sha256,
        "milestone_interval": milestone_interval,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("existing rank-32 evaluation transition has a different source or target identity")
    seeded_count = receipt.get("checkpoint_count")
    if type(seeded_count) is not int or checkpoint_count < seeded_count:
        raise ValueError("rank-32 evaluation checkpoint count is older than the prepared transition")
    registry = EvaluationRegistry(
        target_root / "registry.json", lineage_id=lineage_id,
        evalset_sha256=evalset_sha256, task_ids=receipt.get("task_ids", []),
        milestone_interval=milestone_interval,
    ).snapshot()
    if registry.get("lineage_successful_update_count", -1) < seeded_count:
        raise ValueError("rank-32 evaluation registry lost its checkpoint counter seed")
    return receipt


def refresh_rank32_evaluation_transition_count(
    source_state_dir: str | Path,
    target_state_dir: str | Path,
    *,
    lineage_id: str,
    final_checkpoint_count: int,
    source_checkpoint_snapshot_id: str,
    target_checkpoint_snapshot_id: str,
    source_stop_receipt: str | Path,
) -> dict[str, Any]:
    """Advance the prepared target counter from a real final stopped-source checkpoint."""
    if type(final_checkpoint_count) is not int or final_checkpoint_count < 0:
        raise ValueError("final_checkpoint_count must be a non-negative integer")
    if (not isinstance(source_checkpoint_snapshot_id, str) or not source_checkpoint_snapshot_id.strip()
            or not isinstance(target_checkpoint_snapshot_id, str) or not target_checkpoint_snapshot_id.strip()):
        raise ValueError("source and target checkpoint snapshot IDs must be non-empty")
    stop = read_json(_regular_file(Path(source_stop_receipt).expanduser(), "managed source stop receipt"))
    learner = stop.get("learner_receipt")
    selected = stop.get("selected_checkpoint")
    stop_lineage = stop.get("lineage_id")
    learner_lineage = learner.get("lineage_id") if isinstance(learner, dict) else None
    learner_snapshot = (learner.get("checkpoint_snapshot_id") or learner.get("snapshot_id")
                        if isinstance(learner, dict) else None)
    selected_snapshot = (selected.get("checkpoint_snapshot_id") or selected.get("snapshot_id")
                         if isinstance(selected, dict) else None)
    operation_snapshot = stop.get("checkpoint_snapshot_id") or stop.get("snapshot_id")
    if (stop.get("schema") != "qwen-rl-instance-stop-operation/v1"
            or stop.get("mode") != "after_checkpoint"
            or stop.get("manager_status") != "completed"
            or stop.get("phase") != "managed_roles_stopped"
            or not isinstance(stop.get("operation_id"), str) or not stop["operation_id"].strip()
            or not isinstance(stop.get("deployment_id"), str) or not stop["deployment_id"].strip()
            or not isinstance(stop.get("manager_run_id"), str) or not stop["manager_run_id"].strip()
            or (stop_lineage is not None and stop_lineage != lineage_id)
            or learner_lineage != lineage_id
            or not isinstance(learner, dict)
            or learner.get("status") != "completed"
            or learner.get("request_accepted") is not True
            or learner_snapshot != source_checkpoint_snapshot_id
            or (operation_snapshot is not None and operation_snapshot != source_checkpoint_snapshot_id)
            or not isinstance(selected, dict)
            or selected_snapshot != source_checkpoint_snapshot_id):
        raise ValueError("source stop receipt does not prove a completed managed stop at the final checkpoint")

    source_root = Path(source_state_dir).expanduser()
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source evaluation state must be a real directory")
    source_root = source_root.resolve(strict=True)
    target_state_root = Path(target_state_dir).expanduser()
    if target_state_root.is_symlink() or not target_state_root.is_dir():
        raise ValueError("target evaluation state must be an existing real directory")
    target_state_root = target_state_root.resolve(strict=True)
    target_root = target_state_root / lineage_id
    if target_root.is_symlink() or not target_root.is_dir():
        raise ValueError("prepared rank-32 evaluation target is unavailable")

    receipt_path = _regular_file(target_root / "rank32-transition.json", "rank-32 transition receipt")
    receipt = read_json(receipt_path)
    if (receipt.get("schema") != _TRANSITION_SCHEMA
            or receipt.get("lineage_id") != lineage_id
            or receipt.get("source_state_dir") != str(source_root)
            or receipt.get("target_lineage_dir") != str(target_root)):
        raise ValueError("rank-32 transition receipt has a different source or lineage")
    prepared_count = receipt.get("prepared_checkpoint_count", receipt.get("checkpoint_count"))
    if type(prepared_count) is not int or prepared_count < 0:
        raise ValueError("rank-32 transition receipt has no valid prepared checkpoint count")
    if final_checkpoint_count < prepared_count:
        raise ValueError("final checkpoint count is older than the prepared rank-32 candidate")

    source_snapshot = _source_registry_snapshot(
        source_root,
        lineage_id=lineage_id,
        evalset_sha256=receipt.get("evalset_sha256"),
        task_ids=receipt.get("task_ids", []),
        milestone_interval=receipt.get("milestone_interval"),
    )
    source_history_count = source_snapshot.get("lineage_successful_update_count")
    if type(source_history_count) is not int or final_checkpoint_count < source_history_count:
        raise ValueError("final checkpoint is older than stopped-source evaluation history")
    prepared_source_count = receipt.get("source_history_count_at_prepare", receipt.get("source_registry_count"))
    if type(prepared_source_count) is not int or source_history_count < prepared_source_count:
        raise ValueError("stopped-source evaluation history moved behind its prepared count")

    registry = EvaluationRegistry(
        target_root / "registry.json",
        lineage_id=lineage_id,
        evalset_sha256=receipt["evalset_sha256"],
        task_ids=receipt["task_ids"],
        milestone_interval=receipt["milestone_interval"],
    )
    current_count = registry.snapshot().get("lineage_successful_update_count")
    prior_final = receipt.get("final_checkpoint_count")
    if prior_final is not None:
        if (prior_final != final_checkpoint_count
                or receipt.get("final_source_checkpoint_snapshot_id") != source_checkpoint_snapshot_id
                or receipt.get("final_target_checkpoint_snapshot_id") != target_checkpoint_snapshot_id
                or receipt.get("source_stop_operation_id") != stop["operation_id"]):
            raise ValueError("rank-32 transition was already refreshed from a different final checkpoint")
        if current_count != final_checkpoint_count:
            raise ValueError("rank-32 evaluation registry differs from its final checkpoint receipt")
        return receipt
    if current_count not in {prepared_count, final_checkpoint_count}:
        raise ValueError("rank-32 evaluation registry changed outside the prepared/final checkpoint transition")

    registry.initialize_lineage_count(
        final_checkpoint_count,
        source=f"rank32_transition_final_source_checkpoint:{source_checkpoint_snapshot_id}",
    )
    receipt.update(
        final_checkpoint_count=final_checkpoint_count,
        final_source_checkpoint_snapshot_id=source_checkpoint_snapshot_id,
        final_target_checkpoint_snapshot_id=target_checkpoint_snapshot_id,
        source_history_count_at_final=source_history_count,
        source_stop_operation_id=stop["operation_id"],
        source_stop_receipt={
            "schema": stop["schema"],
            "deployment_id": stop["deployment_id"],
            "manager_run_id": stop["manager_run_id"],
            "phase": stop["phase"],
            "learner_status": learner["status"],
            "checkpoint_snapshot_id": source_checkpoint_snapshot_id,
        },
    )
    write_json_atomic(receipt_path, receipt)
    return receipt


def prepare_rank32_evaluation_transition(
    source_state_dir: str | Path,
    target_state_dir: str | Path,
    *,
    lineage_id: str,
    checkpoint_count: int,
    checkpoint_snapshot_id: str,
    target_identity: Mapping[str, Any],
    quantization: dict[str, Any],
    evalset_sha256: str,
    task_ids: Iterable[str],
    milestone_interval: int = 10,
) -> dict[str, Any]:
    """Prepare a separate rank-32 evaluation candidate at a pinned checkpoint count.

    The source evaluation state is read only. Its version-0 zero-residual PEFT policy is
    expanded with zero-filled new A rows and B columns, then the standard evaluation
    zeroer creates the target baseline archive. The candidate registry is seeded at the
    pinned checkpoint count without replaying history; after stopping the source, refresh
    the counter from its final checkpoint before using this state for cutover.
    """
    from ..contracts.ternary import SHA256_RE, validate_artifact_identity

    if type(checkpoint_count) is not int or checkpoint_count < 0:
        raise ValueError("checkpoint_count must be a non-negative integer")
    if not isinstance(checkpoint_snapshot_id, str) or not checkpoint_snapshot_id.strip():
        raise ValueError("checkpoint_snapshot_id must be non-empty")
    if (not isinstance(evalset_sha256, str) or not SHA256_RE.fullmatch(evalset_sha256)
            or type(milestone_interval) is not int or milestone_interval <= 0):
        raise ValueError("evaluation transition taskset SHA or milestone interval is invalid")
    ordered_task_ids = tuple(task_ids)
    if not ordered_task_ids or len(ordered_task_ids) != len(set(ordered_task_ids)):
        raise ValueError("evaluation transition task IDs must be non-empty and unique")
    if not isinstance(quantization, dict) or quantization.get("mode") != "ternary" or quantization.get("bits") != 2:
        raise ValueError("rank-32 evaluation transition requires the pinned T2 quantization profile")
    identity = validate_artifact_identity(target_identity)
    identity = _target_identity(identity)

    source_root = Path(source_state_dir).expanduser()
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source evaluation state must be a real directory")
    source_root = source_root.resolve(strict=True)
    target_state_root = Path(target_state_dir).expanduser()
    if target_state_root.is_symlink():
        raise ValueError("target evaluation state directory cannot be a symlink")
    target_state_root = target_state_root.resolve(strict=False)
    target_root = target_state_root / lineage_id
    if source_root == target_root or source_root in target_root.parents or target_root in source_root.parents:
        raise ValueError("rank-32 evaluation state must be separate from rank-16 history")

    existing = _read_existing_transition(
        target_root,
        source_state_dir=source_root,
        lineage_id=lineage_id,
        checkpoint_count=checkpoint_count,
        evalset_sha256=evalset_sha256,
        target_identity=identity,
        milestone_interval=milestone_interval,
    )
    if existing is not None:
        return existing

    source_snapshot = _source_registry_snapshot(
        source_root, lineage_id=lineage_id, evalset_sha256=evalset_sha256,
        task_ids=ordered_task_ids, milestone_interval=milestone_interval,
    )
    source_count = source_snapshot.get("lineage_successful_update_count")
    if type(source_count) is not int or source_count < 0:
        raise ValueError("rank-16 evaluation history has no valid lineage count")
    baseline_id = source_snapshot["baseline_request_id"]
    old_baseline_request = source_snapshot["requests"][baseline_id]
    old_policy_archive_ref = old_baseline_request.get("policy_archive_ref")

    source_catalog = read_json(_regular_file(source_root / "policies.json", "rank-16 evaluation catalog"))
    old_baseline = source_catalog.get("0") if isinstance(source_catalog, dict) else None
    if (not isinstance(old_baseline, dict) or old_baseline.get("role") != "baseline_sft"
            or set(old_baseline.get("files", {})) != _NATIVE_FILES):
        raise ValueError("rank-16 history has no native zero-residual SFT baseline catalog entry")
    source_baseline_dir = source_root / "baseline-sft"
    provenance = read_json(_regular_file(source_baseline_dir / "evaluation-policy.json", "rank-16 SFT baseline provenance"))
    baseline_identity = provenance.get("baseline_policy")
    if (not isinstance(baseline_identity, dict)
            or baseline_identity.get("role") != "baseline_sft"
            or baseline_identity.get("version") != 0
            or provenance.get("schema") != "qwen-rl-evaluation-policy-v1"):
        raise ValueError("source evaluation baseline provenance is not version-0 SFT policy evidence")
    if (old_baseline.get("source_adapter_sha256") != baseline_identity.get("adapter_sha256")
            or old_baseline.get("base_fingerprint") != baseline_identity.get("base_fingerprint")
            or old_baseline.get("artifact_fingerprint") != baseline_identity.get("artifact_fingerprint")):
        raise ValueError("rank-16 baseline catalog differs from its SFT provenance")
    target_artifact = identity["source"]["ninfer_artifact"]
    if (baseline_identity.get("base_fingerprint") != target_artifact.get("sha256")
            or baseline_identity.get("artifact_fingerprint") != target_artifact.get("sha256")
            or old_baseline.get("quantization") != quantization
            or provenance.get("quantization") != quantization):
        raise ValueError("rank-32 target identity differs from the rank-16 baseline frozen base")

    old_baseline_directory = source_root / "baseline-sft" / "native"
    if (source_baseline_dir.is_symlink() or not source_baseline_dir.is_dir()
            or old_baseline_directory.is_symlink() or not old_baseline_directory.is_dir()):
        raise ValueError("rank-16 SFT baseline archive directories must be regular directories")
    old_descriptor_path = _regular_file(old_baseline_directory / "policy.lora.ninfer.json",
                                        "rank-16 baseline native descriptor")
    old_descriptor = read_json(old_descriptor_path)
    if (old_descriptor.get("schema") != "qwen-rl-native-policy/v1"
            or old_descriptor.get("version") != 0
            or (old_descriptor.get("rank"), old_descriptor.get("alpha")) != (16, 16)
            or old_descriptor.get("base_ninfer_artifact_id") != target_artifact.get("artifact_id")
            or old_descriptor.get("base_ninfer_sha256") != target_artifact.get("sha256")
            or old_descriptor.get("source_adapter_sha256") != old_baseline.get("source_adapter_sha256")):
        raise ValueError("rank-16 baseline native descriptor differs from its policy catalog identity")

    baseline_source_policy = EvaluationPolicy(
        "baseline_sft",
        PolicySnapshot(
            0,
            baseline_identity["adapter_sha256"],
            baseline_identity["base_fingerprint"],
            baseline_identity["artifact_fingerprint"],
        ),
        str(baseline_identity.get("source") or provenance.get("construction")),
        str(source_baseline_dir),
    )
    target_state_root.mkdir(parents=True, exist_ok=True)
    if target_root.exists() or target_root.is_symlink():
        raise FileExistsError("rank-32 evaluation target exists without a matching transition receipt")
    stage = Path(tempfile.mkdtemp(prefix=f".{lineage_id}.rank32-", suffix=".tmp", dir=target_state_root))
    final_source_dir = target_root / "baseline-sft-source"
    staged_source_dir = stage / "baseline-sft-source"
    try:
        expanded = _expand_rank16_factors(source_baseline_dir, staged_source_dir, identity=identity)
        target_adapter_sha256 = expanded["adapter_tensor_digest"]
        rank32_source_policy = EvaluationPolicy(
            "baseline_sft",
            PolicySnapshot(
                0, target_adapter_sha256,
                target_artifact["sha256"], target_artifact["sha256"],
            ),
            "rank32_expanded_zero_residual_sft",
            str(final_source_dir),
        )
        exported = export_zero_residual_sft_policy(
            staged_source_dir,
            stage / "baseline-sft",
            source_policy=rank32_source_policy,
            quantization=quantization,
        )
        target_registry = EvaluationRegistry(
            stage / "registry.json", lineage_id=lineage_id,
            evalset_sha256=evalset_sha256, task_ids=ordered_task_ids,
            milestone_interval=milestone_interval,
        )
        target_registry.initialize_lineage_count(
            checkpoint_count,
            source=f"rank32_transition_checkpoint:{checkpoint_snapshot_id}",
        )
        receipt = {
            "schema": _TRANSITION_SCHEMA,
            "lineage_id": lineage_id,
            "source_state_dir": str(source_root),
            "target_state_dir": str(target_state_root),
            "target_lineage_dir": str(target_root),
            "source_registry_count": source_count,
            "source_registry_baseline_request_id": baseline_id,
            "source_registry_baseline_status": old_baseline_request.get("status"),
            "source_baseline_policy": baseline_identity,
            "source_baseline_registry_archive_ref": old_policy_archive_ref,
            "source_baseline_native_rank": old_descriptor.get("rank"),
            "source_baseline_native_alpha": old_descriptor.get("alpha"),
            "source_baseline_b_tensor_count": expanded["factor_tensor_count"] // 2,
            "source_baseline_nonzero_b_elements": expanded["source_nonzero_b_elements"],
            "target_identity": {
                "rank": 32, "alpha": 32,
                "artifact_id": target_artifact["artifact_id"],
                "artifact_sha256": target_artifact["sha256"],
                "conversion_report_sha256": identity["source"]["conversion_report"]["sha256"],
            },
            "target_baseline_policy": exported.policy.as_dict(),
            "rank_expansion": {
                "source_rank": 16, "target_rank": 32,
                "new_a_rows_zero": expanded["new_a_rows_zero"],
                "all_b_elements_zero": expanded["all_b_elements_zero"],
                "function_equivalence": "same_frozen_base_zero_residual_lora",
            },
            "checkpoint_snapshot_id": checkpoint_snapshot_id,
            "checkpoint_count": checkpoint_count,
            "prepared_checkpoint_count": checkpoint_count,
            "source_history_count_at_prepare": source_count,
            "evalset_sha256": evalset_sha256,
            "task_ids": list(ordered_task_ids),
            "milestone_interval": milestone_interval,
            "historical_milestones_replayed": [],
            "prepared_at_unix": time.time(),
        }
        write_json_atomic(stage / "rank32-transition.json", receipt)
        stage.rename(target_root)
        return receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


async def initialize_rank32_evaluation_cpu_gate(
    *,
    settings: dict[str, Any],
    dataset: Any,
    recovery: dict[str, Any],
    current_policy: dict[str, Any],
    target_identity: Mapping[str, Any],
    checkpoint_count: int,
) -> dict[str, Any]:
    """Run the real EvaluationRuntime initializer without a Worker or GPU callback."""
    from ..contracts.ternary import validate_artifact_identity
    from .runtime import EvaluationRuntime

    if settings.get("pause_training") is not True:
        raise ValueError("rank-32 evaluation CPU gate requires pause_training=true")
    if type(checkpoint_count) is not int or checkpoint_count < 0:
        raise ValueError("checkpoint_count must be a non-negative integer")
    identity = validate_artifact_identity(target_identity)
    adapter_identity = identity["learner"]["adapter"]
    artifact = identity["source"]["ninfer_artifact"]
    dataset_manifest = getattr(dataset, "manifest", None)
    if (not isinstance(dataset_manifest, dict)
            or dataset_manifest.get("base_fingerprint") != artifact.get("sha256")):
        raise ValueError("CPU evaluation gate dataset differs from the pinned target base")
    current_identity = current_policy.get("_native_export_identity")
    if not isinstance(current_identity, dict) or validate_artifact_identity(current_identity) != identity:
        raise ValueError("CPU evaluation gate current policy does not carry the target pinned identity")
    if (adapter_identity.get("rank"), adapter_identity.get("alpha")) != (32, 32):
        raise ValueError("CPU evaluation gate requires rank-32 target identity")
    if (current_policy.get("base_fingerprint") != artifact.get("sha256")
            or current_policy.get("artifact_fingerprint") != artifact.get("sha256")
            or current_policy.get("quantization", {}).get("mode") != "ternary"):
        raise ValueError("CPU evaluation gate current policy differs from the pinned T2 base")

    calls: list[str] = []

    def publish_pin(_publication: dict) -> None:
        calls.append("publish_pin")

    def release_pin(_version: int) -> None:
        calls.append("release_pin")

    async def forbidden_episode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("CPU evaluation initialization gate must not run an episode")

    runtime = EvaluationRuntime(
        settings=settings,
        dataset=dataset,
        recovery=recovery,
        current_policy=lambda: current_policy,
        publish_pin=publish_pin,
        release_pin=release_pin,
        run_episode=forbidden_episode,
    )
    try:
        expected_root = Path(settings["state_dir"]).expanduser().resolve() / recovery["lineage_id"]
        if runtime.root.resolve() != expected_root:
            raise ValueError("EvaluationRuntime initialized outside the rank-32 target state root")
        initialized_count, _count_source = runtime._initial_count()
        if initialized_count != checkpoint_count:
            raise ValueError("CPU evaluation gate count differs from the full checkpoint update proof")
        await runtime.initialize()
        if runtime.worker is not None:
            raise RuntimeError("CPU evaluation initialization gate unexpectedly started its worker")
        if runtime.count != checkpoint_count:
            raise RuntimeError("EvaluationRuntime count differs from the target checkpoint count")
        transition = read_json(runtime.root / "rank32-transition.json")
        effective_count = transition.get("final_checkpoint_count", transition.get("prepared_checkpoint_count",
                                                                                   transition.get("checkpoint_count")))
        transition_target_snapshot_id = transition.get("final_target_checkpoint_snapshot_id")
        loaded_snapshot = recovery.get("loaded_snapshot") or {}
        recovery_snapshot_id = (recovery.get("snapshot_id") or loaded_snapshot.get("snapshot_id")
                                or (recovery.get("update_context") or {}).get("checkpoint_snapshot_id"))
        if (transition.get("schema") != _TRANSITION_SCHEMA
                or effective_count != checkpoint_count
                or (transition_target_snapshot_id is not None
                    and transition_target_snapshot_id != recovery_snapshot_id)
                or transition.get("target_identity", {}).get("artifact_sha256") != artifact.get("sha256")
                or transition.get("target_identity", {}).get("conversion_report_sha256")
                != identity["source"]["conversion_report"]["sha256"]):
            raise RuntimeError("rank-32 evaluation transition receipt differs from the initialized runtime")
        state = runtime.registry.snapshot()
        if state.get("baseline_request_id") is not None:
            raise RuntimeError("rank-32 evaluation initialization scheduled an initial baseline")
        if any(row.get("kind") == "milestone" for row in state.get("requests", {}).values()):
            raise RuntimeError("rank-32 evaluation initialization replayed a historical milestone")
        if state.get("pending_milestone_request_id") is not None:
            raise RuntimeError("rank-32 evaluation initialization queued a historical milestone")
        # Check the real converted adapter profile without exporting/evaluating v0.
        peft_files = current_policy.get("_peft_files") or current_policy.get("_files", {})
        config_entry = peft_files.get("adapter_config.json")
        if not isinstance(config_entry, dict) or "path" not in config_entry:
            raise RuntimeError("rank-32 CPU gate requires the converted PEFT adapter profile")
        profile = read_json(Path(config_entry["path"]))
        if (profile.get("r"), profile.get("lora_alpha")) != (32, 32):
            raise RuntimeError("converted evaluation adapter profile is not rank/alpha 32/32")
        return {
            "status": "passed",
            "lineage_id": recovery["lineage_id"],
            "checkpoint_count": checkpoint_count,
            "prepared_checkpoint_count": transition.get("prepared_checkpoint_count",
                                                           transition.get("checkpoint_count")),
            "baseline_request_id": None,
            "baseline_policy_version": None,
            "baseline_adapter_sha256": None,
            "native_rank": profile["r"],
            "native_alpha": profile["lora_alpha"],
            "validation_scope": "CPU registry initialization and converted PEFT adapter profile",
            "native_transport_validated": False,
            "target_state_root": str(runtime.root),
            "milestone_request_count": sum(
                row.get("kind") == "milestone" for row in state.get("requests", {}).values()
            ),
            "pin_callbacks": calls,
            "gpu_or_episode_work": False,
        }
    finally:
        await runtime.close()
