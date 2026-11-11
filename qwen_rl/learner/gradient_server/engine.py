"""CPU-only LearnerEngine integration for centrally owned asynchronous gradients."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import resource
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch

from qwen_rl.contracts.gradients import (
    GRADIENT_LOSS_PROFILE,
    GradientJobSpec,
)
from qwen_rl.contracts.gradient_partitions import DEFAULT_GRADIENT_ACCUMULATION_ROWS
from qwen_rl.learner.core import LearnerEngineCore
from qwen_rl.learner.gradient_server.core import (
    DirtyGradientCoreError,
    GradientServerCore,
    WindowStepReceipt,
)
from qwen_rl.learner.gradient_server.store import SQLiteGradientJobStore
from qwen_rl.learner.gradient_server.ddp_initialization import (
    inspect_ddp_recovery_source,
    import_ddp_recovery_state,
    persist_ddp_initial_counters,
    validate_ddp_initialized_core,
)
from qwen_rl.learner.publication import (
    tensor_digest,
)


_EXECUTOR_CAPABILITIES = (
    "claim",
    "rowbundle.read",
    "heartbeat",
    "policy.read",
    "contribution.upload",
    "receipt.read",
    "status.read",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _central_checkpoint_is_empty(path: Path) -> bool:
    if path.is_symlink():
        raise ValueError("central recovery directory may not be a symlink")
    if not path.exists():
        return True
    if not path.is_dir():
        raise ValueError("central recovery path must be a directory")
    return next(path.iterdir(), None) is None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_stale_training_policy_caches(
    root: str | Path,
    prefix: str,
    *,
    owner_uid: int | None = None,
) -> tuple[Path, ...]:
    """Remove only this recovery directory's prior same-UID real cache roots."""
    root = Path(root)
    owner_uid = os.getuid() if owner_uid is None else owner_uid
    if not prefix or Path(prefix).name != prefix or "/" in prefix:
        raise ValueError("training policy cache prefix must be one path component")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("training policy cache root must be a real directory")
    removed = []
    for entry in root.iterdir():
        if not entry.name.startswith(prefix):
            continue
        try:
            entry_stat = entry.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(entry_stat.st_mode) or entry_stat.st_uid != owner_uid:
            continue
        shutil.rmtree(entry)
        removed.append(entry)
    return tuple(removed)


def _training_names(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Restore the exact PEFT live-parameter namespace without loading the base."""
    if not isinstance(state, Mapping) or not state:
        raise ValueError("initial adapter must contain at least one LoRA tensor")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("initial adapter tensors must have string names and tensor values")
        suffix = next((item for item in (".lora_A.weight", ".lora_B.weight")
                       if name.endswith(item)), None)
        if suffix is None:
            raise ValueError(f"initial adapter contains an unsupported PEFT tensor: {name}")
        live_name = name[:-len(".weight")] + ".default.weight"
        if live_name in result:
            raise ValueError("initial adapter namespace conversion produced duplicate parameters")
        # PEFT's default autocast_adapter_dtype promotes BF16/FP16 adapter
        # tensors to FP32 trainable masters. Keep the central optimizer layout
        # identical. copy=True also owns FP32 sources without cloning an
        # already copied BF16/FP16 upcast a second time.
        result[live_name] = tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    prefixes: dict[str, set[str]] = {}
    for name in result:
        if name.endswith(".lora_A.default.weight"):
            prefix = name.removesuffix(".lora_A.default.weight")
            prefixes.setdefault(prefix, set()).add("A")
        elif name.endswith(".lora_B.default.weight"):
            prefix = name.removesuffix(".lora_B.default.weight")
            prefixes.setdefault(prefix, set()).add("B")
        else:
            raise ValueError(f"initial adapter has an unsupported live LoRA parameter: {name}")
    if not prefixes or any(sides != {"A", "B"} for sides in prefixes.values()):
        raise ValueError("initial adapter must contain complete LoRA A/B factor pairs")
    return dict(sorted(result.items()))


def _adapter_payload(config: Mapping[str, Any]) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    from qwen_rl.contracts.ternary import (
        TERNARY_QG_LAYOUT_VERSION,
        artifact_report,
        load_artifact_identity,
        validate_artifact_identity,
    )
    from qwen_rl.learner.initial_adapter import read_initial_adapter
    from ..parallel_engine import _AdapterView

    identity_path = config.get("artifact_identity_path")
    identity = config.get("artifact_identity")
    if not isinstance(identity_path, str) or not identity_path:
        raise ValueError("central_gradient requires the manager-validated T2 artifact identity path")
    if not isinstance(identity, dict):
        raise ValueError("central_gradient requires the manager-validated T2 artifact identity")
    identity = validate_artifact_identity(identity)
    sidecar = load_artifact_identity(identity_path, verify_files=False).as_dict()
    if sidecar != identity:
        raise ValueError("T2 artifact identity sidecar changed after runtime profile validation")
    artifact = identity["source"]["ninfer_artifact"]
    if Path(config.get("model_path", "")).expanduser().resolve() != Path(artifact["path"]).expanduser().resolve():
        raise ValueError("central_gradient model_path differs from its verified T2 artifact identity")
    adapter_identity = identity["learner"]["adapter"]
    rank = int(config.get("lora_rank", -1))
    alpha = int(config.get("lora_alpha", -1))
    if (rank != adapter_identity["rank"] or alpha != adapter_identity["alpha"]):
        raise ValueError("central LoRA geometry differs from the verified T2 artifact identity")

    initial_path = config.get("initial_adapter_path")
    if not isinstance(initial_path, str) or not Path(initial_path).is_absolute():
        raise ValueError("central_gradient requires an explicit absolute initial_adapter_path")
    raw_state, provenance = read_initial_adapter(
        initial_path,
        base_sha256=artifact["sha256"],
        lora_rank=rank,
        lora_alpha=alpha,
    )
    config_path = Path(initial_path) / "adapter_config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("initial adapter config must be a regular file")
    config_bytes = config_path.read_bytes()
    adapter_config = json.loads(config_bytes)
    from peft import LoraConfig

    accepted_config_fields = set(inspect.signature(LoraConfig).parameters)
    unknown_config = {key: value for key, value in adapter_config.items()
                      if key not in accepted_config_fields}
    if any(value not in (None, False, {}) for value in unknown_config.values()):
        raise ValueError("initial adapter config contains unsupported non-default PEFT options")
    adapter_config = {key: value for key, value in adapter_config.items()
                      if key in accepted_config_fields}
    # The configured initial adapter is serialized for inference. The central
    # CPU view is a trainable LoRA owner, matching the DDP recovery's PEFT mode.
    adapter_config["inference_mode"] = False
    training_state = _training_names(raw_state)
    model = _AdapterView({
        "peft_config": {"default": adapter_config},
        "adapter_state": training_state,
    })
    parameters = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    if (not parameters or any(parameter.device.type != "cpu" for _, parameter in parameters)
            or any(not bool(torch.isfinite(parameter).all()) for _, parameter in parameters)):
        raise RuntimeError("central adapter view must own only finite CPU LoRA parameters")
    parameter_count = sum(parameter.numel() for _, parameter in parameters)
    adapter_tensor_digest = tensor_digest(parameters)
    provenance_path = Path(initial_path) / "provenance.json"
    compatibility = {
        "schema": "central-gradient-runtime-compatibility/v1",
        "base_artifact_id": artifact["artifact_id"],
        "base_artifact_sha256": artifact["sha256"],
        "artifact_identity_schema": identity["schema_version"],
        "qg_layout": TERNARY_QG_LAYOUT_VERSION,
        "initial_adapter_path": str(Path(initial_path).resolve()),
        "initial_adapter_provenance": provenance,
        "initial_adapter_provenance_sha256": _sha256_bytes(provenance_path.read_bytes()),
        "initial_adapter_config_sha256": _sha256_bytes(config_bytes),
        "initial_adapter_tensor_digest": adapter_tensor_digest,
        "trainable_master_dtype": "float32",
        "lora_rank": rank,
        "lora_alpha": alpha,
        "parameter_layout": [
            {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in parameters
        ],
    }
    evidence = {
        "model_backend": "ternary",
        "compute_backend": "central_gradient",
        "parallelism": "central_gradient",
        "world_size": 0,
        "full_base_loaded": False,
        "is_loaded_in_4bit": False,
        "base_trainable_parameters": 0,
        "adapter_state_device": "cpu",
        "trainable_master_dtype": "float32",
        "adapter_trainable_parameters": parameter_count,
        "optimizer_owner": "central_cpu_adamw",
        "vocab_size": int(identity["model"]["geometry"]["vocab_size"]),
        "lora": {
            "all_trainable_names_lora_only": True,
            "non_lora_trainable_parameter_count": 0,
            "trainable_lora_parameter_count": parameter_count,
        },
        "base_fingerprint": artifact["sha256"],
        "artifact_fingerprint": artifact["sha256"],
        "artifact_identity": {
            "schema_version": identity["schema_version"],
            "model_id": identity["model"]["id"],
            "artifact_id": artifact["artifact_id"],
            "artifact_sha256": artifact["sha256"],
            "artifact_report_sha256": artifact_report(identity)["sha256"],
            "raw_gguf": identity["source"]["raw_gguf"],
            "mtp_checkpoint": {
                "source_status": identity["mtp_checkpoint"]["source_status"],
                "local_copy": identity["mtp_checkpoint"]["local_copy"],
                "is_lora_adapter": False,
            },
        },
        "quantization": {
            "mode": "ternary",
            "bits": 2,
            "packed_resident": True,
            "compute_dtype": "bfloat16",
            "format": identity["representation"]["format"],
            "layout": identity["representation"]["layout"],
            "group_size_k": identity["representation"]["group_size_k"],
            "t2_object_count": identity["representation"]["t2_object_count"],
            "t2_logical_parameter_count": identity["representation"]["t2_logical_parameter_count"],
            "artifact_id": artifact["artifact_id"],
            "artifact_sha256": artifact["sha256"],
            "artifact_sha256_verified": False,
            "lora_input_basis": adapter_identity["input_basis"],
                **({"gdn_projections": True} if adapter_identity.get("gdn_projections", False) else {}),
            "qg_layout": TERNARY_QG_LAYOUT_VERSION,
        },
        "initial_adapter": provenance,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_tensor_digest": adapter_tensor_digest,
        "model_path": str(artifact["path"]),
        "loaded_components": ["tokenizer", "canonical_lora_adapter"],
    }
    return model, identity, {"compatibility": compatibility, "evidence": evidence}


def central_training_compatibility(
    adapter_compatibility: Mapping[str, Any], identity: Mapping[str, Any],
    optimizer: torch.optim.Optimizer, *, lr: float, max_grad_norm: float,
    importance_sampling_cap: float, group_size: int, microbatch_size: int,
    max_sequence_length: int,
) -> dict[str, Any]:
    """The optimizer/window contract used by both live restore and offline expansion."""
    result = copy.deepcopy(dict(adapter_compatibility))
    result.update({
            "optimizer": {
                "class": "torch.optim.AdamW",
                "lr": lr,
                "betas": [float(value) for value in optimizer.defaults["betas"]],
                "eps": float(optimizer.defaults["eps"]),
                "weight_decay": float(optimizer.defaults["weight_decay"]),
                "parameter_dtype": "float32",
                "max_grad_norm": max_grad_norm,
                "scheduler": "torch.optim.lr_scheduler.LambdaLR/constant",
            },
            "compute_profile": {
                "base_artifact_id": identity["source"]["ninfer_artifact"]["artifact_id"],
                "loss_profile": GRADIENT_LOSS_PROFILE,
                "correction_cap": importance_sampling_cap,
            },
            "window_geometry": {
                "group_size": group_size,
                "microbatch_size": microbatch_size,
                "max_sequence_length": max_sequence_length,
            },
        })
    result.update(result["compute_profile"])
    return result


class GradientLearnerEngine(LearnerEngineCore):
    """Central optimizer with a CPU LoRA view and authenticated HTTP executors."""

    def __init__(
        self, config: Mapping[str, Any], *, tokenizer: Any = None,
        process_owner_lock: Any = None, recovery_only: bool = False,
        preserve_control_files: bool | None = None,
    ):
        if not isinstance(config, Mapping):
            raise TypeError("central gradient engine config must be a mapping")
        if type(recovery_only) is not bool:
            raise TypeError("recovery_only must be a boolean")
        if preserve_control_files is None:
            preserve_control_files = recovery_only
        elif type(preserve_control_files) is not bool:
            raise TypeError("preserve_control_files must be a boolean or None")
        if preserve_control_files and not recovery_only:
            raise ValueError("preserve_control_files requires recovery_only=True")
        self._recovery_only = recovery_only
        self.config = dict(config)
        if (self.config.get("update_backend") != "central_gradient"
                or self.config.get("model_backend") != "ternary"
                or self.config.get("parallelism") != "single"
                or self.config.get("gpus") not in ((), [], None)):
            raise ValueError("central_gradient requires the explicit T2 CPU-controller configuration")
        if self.config.get("gpu_sharing", "dedicated") != "dedicated":
            raise ValueError("central_gradient does not use the local Trainer GPU-sharing lifecycle")

        self.model_path = str(self.config["model_path"])
        self.group_size = int(self.config.get("group_size", 3))
        configured_groups_per_update = int(self.config.get("groups_per_update", 2))
        self.groups_per_update = configured_groups_per_update
        self.groups_per_update_auto = self.config.get("groups_per_update_auto", False)
        self.microbatch_size = int(self.config.get("microbatch_size", 2))
        self.max_sequence_length = int(self.config.get("max_sequence_length", 65536))
        self.lr = float(self.config.get("lr", 1e-6))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.importance_sampling_cap = float(self.config.get("importance_sampling_cap", 3.0))
        self.gradient_wait_timeout = float(self.config.get("gradient_wait_timeout", 3600.0))
        if (self.group_size < 2 or configured_groups_per_update < 1 or self.microbatch_size < 1
                or type(self.groups_per_update_auto) is not bool
                or self.max_sequence_length < 2 or not math.isfinite(self.lr) or self.lr <= 0
                or not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
                or not math.isfinite(self.importance_sampling_cap) or self.importance_sampling_cap < 1
                or not math.isfinite(self.gradient_wait_timeout) or self.gradient_wait_timeout <= 0):
            raise ValueError("central gradient update geometry or optimizer settings are invalid")

        self.device = torch.device("cpu")
        adapter_started = time.perf_counter()
        self.model, identity, model_payload = _adapter_payload(self.config)
        self.startup_load_measurements = {
            "initial_adapter_path": self.config.get("initial_adapter_path"),
            "initial_adapter_seconds": time.perf_counter() - adapter_started,
            "initial_adapter_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        self._ternary_identity = identity
        self._compatibility = model_payload["compatibility"]
        self.evidence = model_payload["evidence"]
        tokenizer_path = str(self.config.get("tokenizer_path") or self.model_path)
        if tokenizer is None:
            from transformers import AutoTokenizer

            if self.config.get("model_backend") == "ternary":
                from qwen_rl.contracts.ternary import validate_artifact_identity
                from qwen_rl.models.ternary_provider import TernaryModelProvider

                identity_path = self.config.get("artifact_identity_path")
                declared_identity = self.config.get("artifact_identity")
                if not isinstance(identity_path, str) or not isinstance(declared_identity, dict):
                    raise ValueError("central T2 tokenizer requires the validated artifact identity")
                declared_identity = validate_artifact_identity(declared_identity)
                with TernaryModelProvider(identity_path, verify_files=False) as provider:
                    if provider.identity.as_dict() != declared_identity:
                        raise ValueError("T2 tokenizer provider identity differs from runtime validation")
                    tokenizer_files = provider.materialize_text_resources(tokenizer_path)
                    if not isinstance(tokenizer_files, dict) or not tokenizer_files:
                        raise ValueError("T2 artifact contains no materialized tokenizer resources")
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path, local_files_only=True,
                    )
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.evidence["tokenizer_path"] = tokenizer_path
        self.evidence["tokenizer_length"] = len(tokenizer)
        self.evidence["tokenizer_vocab_size"] = int(getattr(tokenizer, "vocab_size", 0))

        parameters = [parameter for _, parameter in self.model.named_parameters()]
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lr,
            betas=tuple(self.config.get("adam_betas", (0.9, 0.999))),
            eps=float(self.config.get("adam_eps", 1e-8)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self._compatibility = central_training_compatibility(
            self._compatibility, identity, self.optimizer, lr=self.lr,
            max_grad_norm=self.max_grad_norm,
            importance_sampling_cap=self.importance_sampling_cap,
            group_size=self.group_size, microbatch_size=self.microbatch_size,
            max_sequence_length=self.max_sequence_length,
        )
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir") or (
            Path(self.config["gradient_server_dir"]).expanduser() / "publications"
        )).expanduser().resolve()
        self.publication_dir = self.checkpoint_dir.parent / "publications"
        if not recovery_only:
            self._configure_publication_state()

        recovery_dir = self.config.get("recovery_dir")
        if not isinstance(recovery_dir, str) or not Path(recovery_dir).is_absolute():
            raise ValueError("central_gradient requires an explicit absolute central recovery_dir")
        self.central_checkpoint_dir = Path(recovery_dir).expanduser().resolve()
        from .ownership import ProcessOwnerLock

        if process_owner_lock is None:
            self._process_owner_lock = ProcessOwnerLock.for_checkpoint_dir(self.central_checkpoint_dir)
        else:
            if not isinstance(process_owner_lock, ProcessOwnerLock):
                raise TypeError("process_owner_lock must be a ProcessOwnerLock")
            process_owner_lock.assert_owned_for(self.central_checkpoint_dir)
            self._process_owner_lock = process_owner_lock
        try:
            source_training_dir = self.config.get("gradient_initial_training_dir")
            ddp_source = None
            imported_ddp_state = False
            if source_training_dir is not None:
                if not isinstance(source_training_dir, str) or not Path(source_training_dir).is_absolute():
                    raise ValueError("gradient_initial_training_dir must be an explicit absolute path")
                ddp_source = inspect_ddp_recovery_source(
                    source_training_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    tokenizer=self.tokenizer,
                    base_artifact_id=identity["source"]["ninfer_artifact"]["artifact_id"],
                    base_artifact_sha256=identity["source"]["ninfer_artifact"]["sha256"],
                    correction_cap=self.importance_sampling_cap,
                    max_grad_norm=self.max_grad_norm,
                )
                if ddp_source.training_dir == self.central_checkpoint_dir:
                    raise ValueError("DDP source and central recovery directories must be separate")
                self._compatibility["ddp_initialization"] = copy.deepcopy(ddp_source.receipt)
                self.evidence["ddp_initialization"] = copy.deepcopy(ddp_source.receipt)
                if _central_checkpoint_is_empty(self.central_checkpoint_dir):
                    import_ddp_recovery_state(
                        ddp_source,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )
                    imported_ddp_state = True

            self.core = GradientServerCore(
                self.model,
                self.optimizer,
                self.scheduler,
                self.central_checkpoint_dir,
                max_grad_norm=self.max_grad_norm,
                policy_version=0,
                auto_resume=True,
                async_checkpoint=True,
                # The offline recovery owner holds the checkpoint lock and uses
                # immutable atomic snapshots. Normal live training retains the
                # ordinary owned CPU load and does not depend on a mapped file.
                snapshot_mmap=recovery_only,
                compatibility=self._compatibility,
                compatibility_path_mappings=self.config.get("recovery_path_mappings", {}),
                preserve_control_files=preserve_control_files,
            )
            if ddp_source is not None:
                if imported_ddp_state:
                    persist_ddp_initial_counters(self.core, ddp_source.receipt)
                else:
                    validate_ddp_initialized_core(self.core, ddp_source.receipt)
        except BaseException:
            self._process_owner_lock.close()
            self._process_owner_lock = None
            raise
        initial_core_status = self.core.status()
        if (not recovery_only
                and initial_core_status.get("checkpoint_state") == "recovery_required"):
            try:
                self.core.close()
            finally:
                self._process_owner_lock.close()
                self._process_owner_lock = None
            raise DirtyGradientCoreError(
                "an interrupted asynchronous checkpoint requires recovery before publishing or sampling; "
                "the last complete checkpoint is preserved and speculative policy data must remain isolated"
            )
        self.policy_version = initial_core_status["policy_version"]
        if recovery_only:
            self._gradient_store_path()
            pairs = self.config.get("gradient_executor_worker_ids", {})
            tasks = self.config.get("gradient_tasks_per_worker", 1)
            self._validate_gradient_memory_capacity(len(pairs) * tasks)
            # Managed recovery consumes the restored Core directly. It has no
            # scheduler/runtime window policy until a normal engine is started.
            self.window_policy = None
            self._sync_recovery_metadata()
            return
        self._update_lock = threading.Lock()
        self._shutdown_requested = threading.Event()
        self._staged_groups = {}
        self._staging_lock = threading.RLock()
        self._last_reward_stats: dict[str, Any] = {}
        self._last_gradient_diagnostics: dict[str, Any] = {}

        run_server_dir = self._gradient_store_path()
        run_server_dir.mkdir(parents=True, exist_ok=True)
        if run_server_dir.is_symlink() or not run_server_dir.is_dir():
            raise ValueError("gradient server directory must be a real directory")
        self.gradient_server_dir = run_server_dir
        self.store = SQLiteGradientJobStore(
            run_server_dir / "jobs.sqlite3",
            run_server_dir / "artifacts",
            group_size=self.group_size,
            max_memory_bytes=self.config.get("gradient_memory_limit_bytes"),
            max_inflight_input_tokens=self.config.get("gradient_max_inflight_input_tokens"),
        )
        token_envs = self.config.get("gradient_executor_token_envs")
        if not isinstance(token_envs, Mapping) or not token_envs:
            raise ValueError("central_gradient requires an explicit gradient executor allowlist")
        import re

        sync_worker_ids = tuple(self.config.get("gradient_worker_sync_ids", ()))
        allowlist = {}
        paired_ids = set(self.config.get("gradient_executor_worker_ids", {}))
        for executor_id, env_name in token_envs.items():
            if (not isinstance(executor_id, str) or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", executor_id) is None
                    or not isinstance(env_name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name) is None):
                raise ValueError("gradient executor ID/token environment allowlist is malformed")
            token = os.environ.get(env_name)
            if not token:
                raise ValueError(f"gradient executor token environment variable is missing: {env_name}")
            capabilities = list(_EXECUTOR_CAPABILITIES)
            if sync_worker_ids and executor_id in paired_ids:
                capabilities.append("worker.sync")
            allowlist[executor_id] = {"token": token, "capabilities": capabilities}
        self.executor_allowlist = allowlist
        executor_worker_ids = self.config.get("gradient_executor_worker_ids")
        remote_ids = tuple(self.config.get("gradient_remote_executor_ids", ()))
        if (not isinstance(executor_worker_ids, Mapping)
                or set(executor_worker_ids) & set(remote_ids)
                or set(executor_worker_ids) | set(remote_ids) != set(allowlist)
                or any(not isinstance(worker_id, str) or not worker_id.strip()
                       for worker_id in executor_worker_ids.values())
                or len(set(executor_worker_ids.values())) != len(executor_worker_ids)):
            raise ValueError("central_gradient requires one explicit configured Worker for every executor")
        self.executor_worker_ids = dict(executor_worker_ids)
        from qwen_rl.contracts.sampling_topology import owner_map, owner_slots
        self.sampling_owner_ids = owner_map(self.config)
        self.store.sampling_only_owner_ids = frozenset(self.config.get("sampling_only_worker_ids", {}))
        self.store.remote_executor_ids = frozenset(remote_ids)
        self.store.remote_min_queued_trajectories = self.config.get(
            "gradient_remote_min_queued_trajectories", 1)
        self.store.work_stealing_pools = dict(self.config.get("gradient_work_stealing_pools", {}))
        if not set(self.store.work_stealing_pools) <= set(allowlist) | self.store.sampling_only_owner_ids:
            raise ValueError("work stealing pool contains an unregistered executor")
        self.worker_executor_ids = {
            worker_id: executor_id for executor_id, worker_id in self.sampling_owner_ids.items()
        }
        # Each central round requests one outcome from every explicit pair.
        # The optimizer consumes K complete valid groups for 0 <= K <= P;
        # configured_groups_per_update remains evidence about the config but
        # cannot change the P-outcome round contract.
        tasks_per_worker = self.config.get("gradient_tasks_per_worker", 1)
        if type(tasks_per_worker) is not int or not 1 <= tasks_per_worker <= 8:
            raise ValueError("gradient_tasks_per_worker must be an integer in [1, 8]")
        self.groups_per_update = len(owner_slots(self.config))
        self.groups_per_update_capacity = self.groups_per_update
        # A partial group cannot be consumed yet. Admit enough RAM for a whole
        # round plus one upload, otherwise pressure retries could never make progress.
        self._validate_gradient_memory_capacity(self.groups_per_update)
        self.window_policy = {
            "groups_per_update": self.groups_per_update,
            "groups_per_update_capacity": self.groups_per_update_capacity,
            "groups_per_update_auto": self.groups_per_update_auto,
            "configured_groups_per_update": configured_groups_per_update,
            "gradient_executor_worker_ids": dict(self.executor_worker_ids),
        }
        self.evidence["central_gradient_window_policy"] = copy.deepcopy(self.window_policy)
        self.current_served_policy_provider: Callable[[], Mapping[str, Any] | None] | None = None
        self.ensure_selected_workers: Callable[[str, Mapping[str, Any]], Any] | None = None
        worker_cohorts = (
            {executor_id: list(sync_worker_ids) for executor_id in self.executor_worker_ids}
            if sync_worker_ids else {}
        )

        # A complete local-group staging window is not an optimizer window.
        # Retire those pre-optimizer payloads before HTTP binds a possibly new
        # executor/Worker map, but only when the restored central state is clean
        # and durable. A real prepared optimizer window remains the authority.
        self.store.bind_identity(self.core.lineage_id, self.core.engine_epoch)
        self._release_staged_windows_if_safe(
            reason="startup_staged_group_recompute_required",
        )

        from .http import create_app

        self.gradient_http_app = create_app(
            self.store,
            self.core,
            self.executor_allowlist,
            executor_worker_ids=self.sampling_owner_ids,
            policy_provider=self._training_policy_provider,
            current_policy_provider=self._current_served_policy,
            worker_sync_provider=self._ensure_selected_workers,
            worker_cohorts=worker_cohorts,
            local_executor_ids=tuple(executor_id for executor_id, worker_id in self.executor_worker_ids.items()
                                     if worker_id in self.config.get("gradient_local_executor_worker_ids", ())),
            prefix="",
        )
        self._sync_recovery_metadata()
        restored_publication = self.publication_dir / f"version-{self.policy_version:06d}"
        if restored_publication.exists() or restored_publication.is_symlink():
            # On restore, the durable CPU adapter is the source of truth. Adopt an
            # existing immutable publication only after PEFT verifies its tensors.
            self._publication_store.adopt_restored(
                self.model, self.evidence, self.policy_version, None,
            )

        policy_cache_root = self.publication_dir.parent / 'training-policy-cache'
        if policy_cache_root.is_symlink():
            raise RuntimeError('training policy cache root may not be a symlink')
        policy_cache_root.mkdir(parents=True, exist_ok=True)
        cache_scope = hashlib.sha256(
            str(self.central_checkpoint_dir.resolve()).encode("utf-8")
        ).hexdigest()[:16]
        self._training_policy_cache_prefix = (
            f"qwen-gradient-policies-u{os.getuid()}-{cache_scope}-"
        )
        _cleanup_stale_training_policy_caches(
            policy_cache_root, self._training_policy_cache_prefix, owner_uid=os.getuid(),
        )
        self._training_policy_cache = tempfile.TemporaryDirectory(
            prefix=self._training_policy_cache_prefix,
            dir=str(policy_cache_root),
        )
        self.training_policy_dir = Path(self._training_policy_cache.name)
        self._training_policy_digests: dict[int, str] = {}
        self._training_policy_lock = threading.RLock()
        self._training_policy_write_events: dict[tuple[str, str, int], threading.Event] = {}
        self._training_policy_write_errors: dict[tuple[str, str, int], str] = {}
        try:
            self._checkpoint_publications = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="central-lora-publication",
            )
        except BaseException:
            self._training_policy_cache.cleanup()
            raise

    def _gradient_store_path(self) -> Path:
        """Validate the derived store location without creating runtime state."""
        path = Path(self.config.get("gradient_server_dir") or (
            Path(self.config["checkpoint_dir"]).parent / "gradient-server"
        )).expanduser()
        if path.is_symlink():
            raise ValueError("gradient server directory may not be a symlink")
        if path.exists() and not path.is_dir():
            raise ValueError("gradient server directory must be a real directory")
        return path.resolve()

    def _validate_gradient_memory_capacity(self, groups_per_update: int) -> None:
        """Keep the layout-dependent RAM check even without runtime services."""
        partitions_per_group = (
            math.ceil((self.group_size + 1) / self.config.get(
                "gradient_accumulation_rows", DEFAULT_GRADIENT_ACCUMULATION_ROWS,
            )) if self.config.get("gradient_work_stealing_pools") else 1
        )
        raw_gradient_bytes = sum(math.prod(parameter.shape) * 4 for parameter in self.core.parameter_layout)
        required_gradient_bytes = ((groups_per_update * partitions_per_group + 1)
                                   * (raw_gradient_bytes + 1024**2))
        limit = self.config.get("gradient_memory_limit_bytes")
        if limit is not None and (type(limit) is not int or limit < required_gradient_bytes):
            raise ValueError("gradient_memory_limit_bytes cannot hold one full round and an upload; "
                             f"requires at least {required_gradient_bytes} bytes")
        self.evidence["gradient_memory_required_bytes"] = required_gradient_bytes

    def open_recovery_window_store(self):
        """Open the scheduler store only when a recovery window needs inspection."""
        if not self._recovery_only:
            return self.store
        pending = self.core.status().get("pending_window")
        if pending is None:
            return None
        existing = getattr(self, "store", None)
        if existing is not None:
            return existing

        run_server_dir = self._gradient_store_path()
        run_server_dir.mkdir(parents=True, exist_ok=True)
        if run_server_dir.is_symlink() or not run_server_dir.is_dir():
            raise ValueError("gradient server directory must be a real directory")
        store = SQLiteGradientJobStore(
            run_server_dir / "jobs.sqlite3",
            run_server_dir / "artifacts",
            group_size=self.group_size,
            max_memory_bytes=self.config.get("gradient_memory_limit_bytes"),
            max_inflight_input_tokens=self.config.get("gradient_max_inflight_input_tokens"),
        )
        store.bind_identity(self.core.lineage_id, self.core.engine_epoch)
        self.gradient_server_dir = run_server_dir
        self.store = store
        return store

    def _current_served_policy(self) -> Mapping[str, Any] | None:
        provider = self.current_served_policy_provider
        if provider is None:
            return None
        return provider()

    def _ensure_selected_workers(self, executor_id: str, payload: Mapping[str, Any]) -> Any:
        provider = self.ensure_selected_workers
        if provider is None:
            raise RuntimeError("selected Worker policy sync provider is unavailable")
        return provider(executor_id, payload)

    def _training_policy_key(self, version: int) -> tuple[str, str, int]:
        return (str(self.core.lineage_id), str(self.core.engine_epoch), version)

    def _begin_training_policy_write(
        self, version: int,
    ) -> tuple[tuple[str, str, int], threading.Event]:
        key = self._training_policy_key(version)
        with self._training_policy_lock:
            if self._shutdown_requested.is_set():
                raise RuntimeError("learner is shutting down")
            previous = self._training_policy_write_events.get(key)
            if previous is not None and not previous.is_set():
                raise RuntimeError(f"training policy write is already active for version {version}")
            event = threading.Event()
            self._training_policy_write_events[key] = event
            self._training_policy_write_errors.pop(key, None)
            return key, event

    def _finish_training_policy_write(
        self,
        key: tuple[str, str, int],
        error: BaseException | str | None = None,
    ) -> None:
        with self._training_policy_lock:
            event = self._training_policy_write_events.get(key)
            if event is None:
                return
            if self._shutdown_requested.is_set():
                error = self._training_policy_write_errors.get(key) or "learner is shutting down"
            if error is None:
                self._training_policy_write_errors.pop(key, None)
            elif isinstance(error, BaseException):
                self._training_policy_write_errors[key] = f"{type(error).__name__}: {error}"
            else:
                self._training_policy_write_errors[key] = str(error)
            event.set()

    def _wake_training_policy_waiters(self) -> None:
        with self._training_policy_lock:
            for key, event in self._training_policy_write_events.items():
                if not event.is_set():
                    self._training_policy_write_errors[key] = "learner is shutting down"
                    event.set()

    def _wait_for_training_policy_write(self, version: int) -> None:
        key = self._training_policy_key(version)
        with self._training_policy_lock:
            event = self._training_policy_write_events.get(key)
            error = self._training_policy_write_errors.get(key)
        if error is not None:
            raise RuntimeError(f"training policy {version} publication failed: {error}")
        if self._shutdown_requested.is_set():
            raise RuntimeError("learner is shutting down")
        if event is not None and not event.is_set():
            if not event.wait(timeout=self.gradient_wait_timeout):
                raise TimeoutError(f"training policy {version} is still being written")
        with self._training_policy_lock:
            error = self._training_policy_write_errors.get(key)
        if error is not None:
            raise RuntimeError(f"training policy {version} publication failed: {error}")
        if self._shutdown_requested.is_set():
            raise RuntimeError("learner is shutting down")

    def _training_policy_provider(self, *identity: Any) -> Path:
        if len(identity) == 1:
            version = identity[0]
            status = self.core.status()
            lineage_id = status.get("lineage_id")
            engine_epoch = status.get("engine_epoch")
        elif len(identity) == 3:
            lineage_id, engine_epoch, version = identity
            status = self.core.status()
        else:
            raise ValueError("training policy provider requires version or lineage/epoch/version")
        if type(version) is not int or version < 0:
            raise ValueError("training policy version must be a non-negative integer")
        if (lineage_id != status.get("lineage_id")
                or engine_epoch != status.get("engine_epoch")):
            raise ValueError("executor requested a policy from a different central lineage/epoch")
        self._wait_for_training_policy_write(version)
        if self.training_policy_dir.is_symlink() or not self.training_policy_dir.is_dir():
            raise ValueError("training policy root is missing or unsafe")
        lineage_dir = self.training_policy_dir / str(lineage_id)
        epoch_dir = lineage_dir / str(engine_epoch)
        if lineage_dir.is_symlink() or epoch_dir.is_symlink():
            raise FileNotFoundError("central lineage/epoch training policy directory is unsafe")
        if not lineage_dir.is_dir() or not epoch_dir.is_dir():
            if (version != status.get("durable_policy_version")
                    or status.get("checkpoint_state") != "durable"):
                raise FileNotFoundError("central lineage/epoch training policy directory is unavailable")
            with self._training_policy_lock:
                lineage_dir.mkdir(exist_ok=True)
                if lineage_dir.is_symlink() or not lineage_dir.is_dir():
                    raise FileNotFoundError("central lineage training policy directory is unsafe")
                epoch_dir.mkdir(exist_ok=True)
                if epoch_dir.is_symlink() or not epoch_dir.is_dir():
                    raise FileNotFoundError("central epoch training policy directory is unsafe")
        if lineage_dir.is_symlink() or epoch_dir.is_symlink():
            raise FileNotFoundError("central lineage/epoch training policy directory is unavailable")
        path = (self.training_policy_dir / str(lineage_id) / str(engine_epoch)
                / f"version-{version:06d}")
        if path.is_symlink() or not path.is_dir():
            with self._training_policy_lock:
                if path.is_symlink() or not path.is_dir():
                    current = self.core.status()
                    if (version != current.get("durable_policy_version")
                            or current.get("checkpoint_state") != "durable"):
                        raise FileNotFoundError(
                            f"canonical LoRA training state is unavailable for version {version}"
                        )
                    publication_dir = self.publication_dir / f"version-{version:06d}"
                    entry = self._publication_store.published_versions.get(version)
                    if (publication_dir.is_symlink() or not publication_dir.is_dir()
                            or entry is None):
                        raise FileNotFoundError(
                            f"durable adapter publication is unavailable for training policy {version}"
                        )
                    hashes = entry[1]
                    declared_adapter_hash = hashes.get("adapter_model.safetensors")
                    if not isinstance(declared_adapter_hash, str):
                        declared_adapter_hash = entry[0]
                    write_key = self._training_policy_key(version)
                    previous_error = self._training_policy_write_errors.get(write_key)
                    if previous_error is not None:
                        raise RuntimeError(
                            f"training policy {version} publication failed: {previous_error}"
                        )
                    write_key, _ = self._begin_training_policy_write(version)
                    try:
                        self._write_training_policy(
                            version,
                            {
                                "directory": str(publication_dir),
                                "adapter_hash": declared_adapter_hash,
                            },
                            parameter_state=self.core.adapter_state_dict(),
                            state_digest=entry[0],
                        )
                    except BaseException as exc:
                        self._finish_training_policy_write(write_key, exc)
                        raise
                    else:
                        self._finish_training_policy_write(write_key)
        return path

    def _record_timing_span(self, name: str, start: Mapping[str, Any], end: Mapping[str, Any] | None = None) -> None:
        core = getattr(self, "core", None)
        if core is not None:
            core.record_latest_timing_span(name, start, end, persist=False)

    def _note_timing(self, event: str, point: Mapping[str, Any] | None = None) -> None:
        core = getattr(self, "core", None)
        if core is not None:
            core.note_latest_timing_event(event, point, persist=False)

    def _prune_training_policies(self) -> None:
        # Publication references cover rollout/evaluation; scheduler windows
        # additionally pin versions still needed for gradient recovery.
        with self._publication_store.lock:
            keep = set(self.retained_policy_versions) | self.store.active_policy_versions()
            keep.update((self.policy_version, max(0, self.policy_version - 1)))
            namespace = self.training_policy_dir / self.core.lineage_id / self.core.engine_epoch
            if not namespace.exists():
                return
            if namespace.is_symlink():
                raise ValueError("training policy namespace cannot be a symlink")
            for entry in namespace.iterdir():
                if entry.is_symlink() or not entry.is_dir():
                    continue
                suffix = entry.name.removeprefix("version-")
                if not suffix.isascii() or not suffix.isdecimal():
                    continue
                version = int(suffix)
                if entry.name == f"version-{version:06d}" and version < self.policy_version and version not in keep:
                    shutil.rmtree(entry)

    def _schedule_training_policy_prune(self) -> None:
        publisher = getattr(self, "_checkpoint_publications", None)
        if publisher is None:
            return
        try:
            publisher.submit(self._prune_training_policies)
        except RuntimeError:
            # Shutdown disposes of the volatile cache after publishers join.
            pass

    def release_policy(self, version: int) -> None:
        super().release_policy(version)
        self._schedule_training_policy_prune()

    def _write_training_policy(
        self,
        version: int,
        publication: Mapping[str, Any],
        *,
        parameter_state: Mapping[str, torch.Tensor] | None = None,
        state_digest: str | None = None,
    ) -> Path:
        from ..training_state import materialize_training_state

        policy_total_started = self.core.timing_mark()
        standard_dir = Path(publication["directory"]).resolve(strict=True)
        source_config = standard_dir / "adapter_config.json"
        status = self.core.status()
        lineage_id = status.get("lineage_id")
        engine_epoch = status.get("engine_epoch")
        target_root = self.training_policy_dir / str(lineage_id) / str(engine_epoch)
        for parent in (self.training_policy_dir, self.training_policy_dir / str(lineage_id)):
            if parent.is_symlink():
                raise RuntimeError("training policy namespace may not traverse symlink directories")
        target_root.mkdir(parents=True, exist_ok=True)
        if target_root.is_symlink() or not target_root.is_dir():
            raise RuntimeError("training policy namespace must be a real directory")
        target = target_root / f"version-{version:06d}"
        state_copy_started = self.core.timing_mark()
        state = self.core.adapter_state_dict() if parameter_state is None else parameter_state
        state_copy_finished = self.core.timing_mark()
        self._record_timing_span(
            "training_policy_snapshot_reuse" if parameter_state is not None
            else "training_policy_checkpoint_state_copy",
            state_copy_started,
            state_copy_finished,
        )
        layout = {item.name: (item.shape, item.dtype) for item in self.core.parameter_layout}
        if set(state) != set(layout) or any(
            not isinstance(state[name], torch.Tensor)
            or state[name].device.type != "cpu"
            or tuple(state[name].shape) != layout[name][0]
            or str(state[name].dtype).removeprefix("torch.") != layout[name][1]
            for name in layout
        ):
            raise RuntimeError("central training policy snapshot differs from the checkpoint LoRA layout")
        if state_digest is None:
            digest_started = self.core.timing_mark()
            state_digest = tensor_digest(state.items())
            self._record_timing_span(
                "training_policy_tensor_digest", digest_started, self.core.timing_mark(),
            )
        directory_started = self.core.timing_mark()
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise RuntimeError("existing training-state version is unsafe")
            config_path = target / "adapter_config.json"
            state_path = target / "adapter_state.safetensors"
            manifest_path = target / "training-state.json"
            existing_files = {path.name for path in target.iterdir()}
            expected_files = {config_path.name, state_path.name, manifest_path.name}
            if existing_files - expected_files or any(
                path.is_symlink() or not path.is_file() for path in target.iterdir()
            ):
                raise RuntimeError("existing training-state version contains unsafe entries")
            matches = self._training_policy_digests.get(version) == state_digest
            try:
                if matches and existing_files == expected_files:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    matches = (
                        config_path.read_bytes() == source_config.read_bytes()
                        and manifest.get("schema") == 'central-gradient-training-policy/v2'
                        and manifest.get("storage") == 'standard_peft_fp32'
                        and manifest.get("policy_version") == version
                        and manifest.get("layout_fingerprint") == self.core.layout_fingerprint
                        and manifest.get("tensor_digest") == state_digest
                    )
            except (OSError, ValueError, json.JSONDecodeError):
                matches = False
            existing_validation_finished = self.core.timing_mark()
            if matches:
                self._record_timing_span(
                    "training_policy_existing_state_validation",
                    directory_started,
                    existing_validation_finished,
                )
                self._record_timing_span("training_policy_total", policy_total_started, existing_validation_finished)
                return target
            raise RuntimeError(f"existing immutable training policy differs for version {version}")

        directory_finished = self.core.timing_mark()
        self._record_timing_span("training_policy_directory_setup", directory_started, directory_finished)
        temporary_started = self.core.timing_mark()
        temporary = Path(tempfile.mkdtemp(prefix=f".version-{version:06d}-", dir=target_root))
        try:
            (temporary / "adapter_config.json").write_bytes(source_config.read_bytes())
            temporary_finished = self.core.timing_mark()
            self._record_timing_span("training_policy_temp_prepare", temporary_started, temporary_finished)
            safetensors_started = self.core.timing_mark()
            backing = materialize_training_state(
                standard_dir / 'adapter_model.safetensors',
                temporary / 'adapter_state.safetensors',
                {name: spec[0] for name, spec in layout.items()},
            )
            safetensors_finished = self.core.timing_mark()
            self._record_timing_span("training_policy_backing_materialize", safetensors_started, safetensors_finished)
            manifest = {
                "schema": "central-gradient-training-policy/v2",
                "storage": "standard_peft_fp32",
                "materialization": backing,
                "policy_version": version,
                "lineage_id": lineage_id,
                "engine_epoch": engine_epoch,
                "layout_fingerprint": self.core.layout_fingerprint,
                "parameter_count": sum(math.prod(layout[name][0]) for name in layout),
                "tensor_digest": state_digest,
                "adapter_hash": publication["adapter_hash"],
            }
            manifest_write_started = self.core.timing_mark()
            (temporary / "training-state.json").write_bytes(_canonical_json(manifest) + b"\n")
            manifest_write_finished = self.core.timing_mark()
            self._record_timing_span(
                "training_policy_manifest_write", manifest_write_started, manifest_write_finished,
            )
            rename_started = self.core.timing_mark()
            os.rename(temporary, target)
            rename_finished = self.core.timing_mark()
            self._record_timing_span("training_policy_rename", rename_started, rename_finished)
            self._training_policy_digests[version] = state_digest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        policy_total_finished = self.core.timing_mark()
        self._record_timing_span("training_policy_total", policy_total_started, policy_total_finished)
        return target

    def publish(
        self,
        *,
        rollout_ready: Callable[[dict[str, Any]], None] | None = None,
        parameter_state: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        write_key, _ = self._begin_training_policy_write(self.policy_version)
        try:
            publish_started = self.core.timing_mark()
            self._note_timing("publication_started", publish_started)
            standard_started = self.core.timing_mark()
            publication = super().publish(parameter_state=parameter_state)
            standard_finished = self.core.timing_mark()
            self._record_timing_span("publication_standard_policy_store", standard_started, standard_finished)
            # Rollout needs only the immutable inference adapter. The FP32 training
            # bundle is for the following backward and has its own durability path.
            callback_error: BaseException | None = None
            if rollout_ready is not None:
                try:
                    rollout_ready(publication)
                except BaseException as exc:
                    callback_error = exc
            training_started = self.core.timing_mark()
            entry = self._publication_store.published_versions.get(self.policy_version)
            state_digest = None if entry is None else entry[0]
            self._write_training_policy(
                self.policy_version,
                publication,
                parameter_state=parameter_state,
                state_digest=state_digest,
            )
        except BaseException as exc:
            self._finish_training_policy_write(write_key, exc)
            raise
        self._finish_training_policy_write(write_key)
        self._schedule_training_policy_prune()
        training_finished = self.core.timing_mark()
        self._record_timing_span("publication_training_policy_write", training_started, training_finished)
        self._note_timing("publication_completed", training_finished)
        self._record_timing_span("publication_total", publish_started, training_finished)
        if callback_error is not None:
            raise callback_error
        return publication

    def _sync_recovery_metadata(self) -> dict[str, Any]:
        status = self.core.status()
        self.policy_version = int(status["policy_version"])
        checkpoint_path = status.get("checkpoint_path")
        snapshot_id = status.get("checkpoint_id")
        self._recovery_metadata = {
            "schema": "qwen-rl-central-gradient-state/v1",
            "enabled": True,
            "mode": "resumed" if status.get("resumed") is True else "fresh",
            "resumed": status.get("resumed") is True,
            "lineage_id": status.get("lineage_id"),
            # BEPA stores its ledger under this stable lineage UUID.
            "engine_epoch": status.get("engine_epoch"),
            "policy_version": self.policy_version,
            "durable_policy_version": status.get("durable_policy_version", self.policy_version),
            "checkpoint_state": status.get("checkpoint_state", "durable"),
            "can_optimize": status.get("can_optimize", status.get("dirty") is not True),
            "can_publish": status.get("can_publish", status.get("dirty") is not True),
            "checkpoint_intent": copy.deepcopy(status.get("checkpoint_intent")),
            "snapshot_id": snapshot_id,
            "snapshot_sha256": None,
            "snapshot_path": checkpoint_path,
            "loaded_snapshot": {
                "policy_version": status.get("durable_policy_version", self.policy_version),
                "snapshot_id": snapshot_id,
                "checkpoint_generation": status.get("checkpoint_generation"),
                "path": checkpoint_path,
            },
            "update_context": copy.deepcopy(status.get("last_update_context")),
            "compatibility_sha256": _sha256_bytes(_canonical_json(self._compatibility)),
            "dirty": status.get("dirty") is True,
            "pending_window": copy.deepcopy(status.get("pending_window")),
            "last_receipt": copy.deepcopy(status.get("last_receipt")),
            "optimizer_steps": status.get("optimizer_steps"),
            "durable_optimizer_steps": status.get("durable_optimizer_steps", status.get("optimizer_steps")),
            "scheduler_steps": status.get("scheduler_steps"),
            # Batch cardinality and executor/Worker affinity are intentionally
            # mutable across a clean optimizer resume and do not alter Adam state.
            "window_policy": copy.deepcopy(self.window_policy),
        }
        return self._recovery_metadata

    @property
    def recovery_metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._sync_recovery_metadata())

    @property
    def retained_policy_versions(self) -> tuple[int, ...]:
        return self._publication_store.retained_versions()

    def logprobs(self, input_ids: list[int], policy_mask: list[int]) -> list[float]:
        raise RuntimeError("central_gradient has no loaded base scorer; token logprob alignment is unsupported")

    def _resume_ready_windows_sync(self) -> None:
        async def resume() -> None:
            from .http import resume_ready_windows

            await resume_ready_windows(self.store, self.core)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(resume())
            return
        failures: list[BaseException] = []
        completed = threading.Event()

        def worker() -> None:
            try:
                asyncio.run(resume())
            except BaseException as exc:
                failures.append(exc)
            finally:
                completed.set()

        thread = threading.Thread(target=worker, name="central-gradient-resume", daemon=True)
        thread.start()
        completed.wait()
        if failures:
            raise failures[0]

    async def resume_ready_gradient_windows(
        self,
        *,
        callback_factory: Callable[[Any], Callable[[], None] | None] | None = None,
    ) -> None:
        # Pre-optimizer staged means have no durable optimizer authority. On a
        # new process, cancel their old leases before fresh rounds can dispatch.
        self._release_staged_windows_if_safe(
            reason="startup_staged_group_recompute_required",
        )
        status = self.core.status()
        pending = status.get("pending_window")
        if pending is not None:
            if status.get("dirty") is True:
                raise DirtyGradientCoreError(status.get("dirty_reason") or "central core is fenced")
            window = self.core.prepared_window(pending["window_id"], pending["attempt_id"])
            if window is None:
                raise RuntimeError("central core status names a missing prepared window")
            try:
                accepted, total = await asyncio.to_thread(
                    self.store.accepted_count, window.window_id, window.attempt_id,
                )
            except KeyError:
                accepted, total = 0, 0
            window_ready = total > 0 and accepted == total
            if not window_ready:
                reason = "startup_incomplete_gradient_window_cancelled_before_optimizer_entry"
                try:
                    cancelled = await asyncio.to_thread(
                        self.store.cancel_window, window.window_id, window.attempt_id, reason,
                    )
                except KeyError:
                    # A crash may have persisted the core prepared-window sidecar
                    # before the scheduler transaction created its first row.
                    cancelled = True
                if not cancelled:
                    # The final contribution may win the race with cancellation.
                    # Recheck readiness before deciding to fence the core window.
                    accepted, total = await asyncio.to_thread(
                        self.store.accepted_count, window.window_id, window.attempt_id,
                    )
                    window_ready = total > 0 and accepted == total
                if not window_ready:
                    await asyncio.to_thread(
                        self.core.cancel_prepared_window, window, reason=reason,
                    )
                    pending = None
                    # The prepared window fenced staged cleanup at startup.
                    # Once it is cancelled before optimizer entry, retire its
                    # orphan group means before any new executor can claim them.
                    self._release_staged_windows_if_safe(
                        reason="startup_cancelled_prepared_window_staged_recompute_required",
                    )
            if window_ready:
                context = window.update_context
                if (isinstance(context, Mapping) and context.get("bepa_attempt_id") is not None
                        and callback_factory is None):
                    raise RuntimeError("recovered BEPA window cannot step without its optimizer-entry callback")
                if callback_factory is not None:
                    callback = callback_factory(window)
                    if callback is not None:
                        self.core.register_before_optimizer_step(window, callback)
                    elif isinstance(context, Mapping) and context.get("bepa_attempt_id") is not None:
                        raise RuntimeError("recovered BEPA window callback factory returned no callback")
        from .http import resume_ready_windows

        if pending is not None:
            await resume_ready_windows(self.store, self.core)
        for stale in self.gradient_server_dir.glob("gradient-rowbundle-*"):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
        self._sync_recovery_metadata()

    @staticmethod
    def _staged_release_is_safe(status: Mapping[str, Any]) -> bool:
        return bool(
            status.get("dirty") is False
            and status.get("checkpoint_state") == "durable"
            and status.get("can_optimize") is True
            and status.get("pending_attempt") is None
            and status.get("pending_window") is None
        )

    def _release_staged_windows_if_safe(self, *, reason: str) -> int:
        status = self.core.status()
        if not self._staged_release_is_safe(status):
            return 0
        released = self.store.release_staged_windows(reason=reason)
        self.store.release_terminal_window_payloads()
        return released

    def _rowbundle_paths(
        self,
        window,
        job_rows: Mapping[str, Mapping[str, Any]],
    ) -> tuple[tempfile.TemporaryDirectory, dict[str, str]]:
        temporary = tempfile.TemporaryDirectory(prefix="gradient-rowbundle-", dir=self.gradient_server_dir)
        root = Path(temporary.name)
        paths: dict[str, str] = {}
        profile = dict(window.compute_profile)
        for job in window.jobs:
            group = job_rows[job.job_id]
            rows = group["rows"]
            payload = {
                "schema": "gradient_rowbundle/v1",
                "window_id": window.window_id,
                "attempt_id": window.attempt_id,
                "lineage_id": window.lineage_id,
                "engine_epoch": window.engine_epoch,
                "job_id": job.job_id,
                "group_id": job.group_id,
                "owner_executor_id": job.owner_executor_id,
                "source_worker_id": group["source_worker_id"],
                "group_size": self.group_size,
                "policy_version": window.policy_version,
                "n_window_tokens": window.n_window_tokens,
                "token_count": job.token_count,
                "layout_fingerprint": window.layout_fingerprint,
                "compute_profile_fingerprint": job.compute_profile_fingerprint,
                "compute_profile": profile,
                "rows": rows,
            }
            path = root / f"{job.job_id}.json"
            path.write_bytes(_canonical_json(payload))
            paths[job.job_id] = str(path)
        return temporary, paths

    def _aggregate_diagnostics(self, contributions: list[Any]) -> dict[str, Any]:
        fields = (
            "fresh_policy_tokens", "replay_policy_tokens", "fresh_behavior_tokens",
            "correction_capped_tokens", "dapo_loss_microbatch_count",
        )
        result: dict[str, Any] = {}
        for name in fields:
            values = [item.diagnostics[name] for item in contributions
                      if isinstance(item.diagnostics, Mapping) and name in item.diagnostics]
            if values and all(type(value) is int for value in values):
                result[name] = sum(values)
        correction_tokens = result.get("fresh_behavior_tokens")
        capped = result.get("correction_capped_tokens")
        if isinstance(correction_tokens, int) and correction_tokens > 0 and isinstance(capped, int):
            result["correction_capped_fraction"] = capped / correction_tokens
        losses = [item.diagnostics.get("dapo_loss_mean") for item in contributions
                  if isinstance(item.diagnostics, Mapping)
                  and isinstance(item.diagnostics.get("dapo_loss_mean"), (int, float))]
        if losses and all(math.isfinite(float(value)) for value in losses):
            result["dapo_loss_job_mean"] = sum(float(value) for value in losses) / len(losses)
        return result

    def begin_staged_round(self, round_id: str, first_ordinal: int) -> None:
        from qwen_rl.contracts.sampling_topology import owner_slots_for_ordinal
        slots = owner_slots_for_ordinal(self.config, first_ordinal)
        self.store.begin_sampling_round(
            round_id, lineage_id=self.core.lineage_id, engine_epoch=self.core.engine_epoch,
            policy_version=self.policy_version, executor_ids=tuple(sorted(set(slots))),
            tasks_per_worker=int(self.config.get('gradient_tasks_per_worker', 1)),
            first_ordinal=first_ordinal, owner_slots=slots,
        )

    def note_sampling_outcome(self, round_id: str, ordinal: int) -> None:
        self.store.note_sampling_outcome(round_id, ordinal)

    def stage_group(
        self, round_id: str, trajectories: list[dict[str, Any]], *,
        logical_run_id: str | None = None,
    ) -> None:
        """Enqueue one complete group now; never enter an optimizer here.

        Unscoped diagnostics may omit the logical run. Runtime training supplies
        it explicitly so executor telemetry cannot leak into another live run.
        """
        if logical_run_id is not None and (
            not isinstance(logical_run_id, str) or not logical_run_id.strip()
        ):
            raise ValueError("logical_run_id must be a nonempty string when provided")
        from .group_staging import STAGED_GROUP_SCOPE, rows_fingerprint
        from qwen_rl.contracts.gradients import (
            GradientJob,
            PreparedWindow,
            PREPARED_WINDOW_SCHEMA,
            GRADIENT_CONTRIBUTION_CONTRACT,
            compute_profile_fingerprint,
        )
        from qwen_rl.learner.batch import validate_batch

        with self._staging_lock:
            rows, tokens, _, _ = validate_batch(
                trajectories,
                learner_start_version=self.policy_version,
                max_sequence_length=self.max_sequence_length,
                group_size=self.group_size,
                groups_per_update=1,
            )
            groups = {row["group_id"] for row in rows}
            if len(groups) != 1:
                raise ValueError("staging requires exactly one complete group")
            group_id = next(iter(groups))
            workers = {
                row.get("source_worker_id", row.get("worker_id"))
                for row in rows
                if row.get("row_kind", "fresh_student") == "fresh_student"
            }
            if len(workers) != 1 or next(iter(workers)) not in self.worker_executor_ids:
                raise ValueError("staged group must retain one configured owner")
            worker = next(iter(workers))
            owner = self.worker_executor_ids[worker]
            for i, row in enumerate(rows):
                row["row_id"] = i
                if row.get("row_kind", "fresh_student") == "fresh_student":
                    from ...worker.episode_queue import validate_sampling_placement
                    validate_sampling_placement(row, worker, self.store.episode_queue)
                    row["source_worker_id"] = worker
            fingerprint = rows_fingerprint(rows)
            key = (round_id, group_id)
            previous = self._staged_groups.get(key)
            if previous is not None:
                if (previous["window"].update_context or {}).get("logical_run_id") != logical_run_id:
                    raise ValueError("staged group identity was reused by a different logical run")
                if previous["fingerprint"] != fingerprint:
                    raise ValueError(
                        "staged group identity was reused with changed rows"
                    )
                return
            profile = {
                "base_artifact_id": self._ternary_identity["source"]["ninfer_artifact"][
                    "artifact_id"
                ],
                "loss_profile": GRADIENT_LOSS_PROFILE,
                "correction_cap": self.importance_sampling_cap,
            }
            digest = _sha256_bytes(
                _canonical_json(
                    [
                        round_id,
                        self.core.lineage_id,
                        self.core.engine_epoch,
                        self.policy_version,
                        fingerprint,
                    ]
                )
            )
            wid, aid = "sg-" + digest[:32], "sa-" + digest[32:]
            job = GradientJob(
                schema="gradient_job/v1",
                window_id=wid,
                attempt_id=aid,
                lineage_id=self.core.lineage_id,
                engine_epoch=self.core.engine_epoch,
                job_id=wid + "-g0",
                group_id=group_id,
                owner_executor_id=owner,
                policy_version=self.policy_version,
                n_window_tokens=tokens,
                token_count=tokens,
                work_index=0,
                layout_fingerprint=self.core.layout_fingerprint,
                compute_profile=profile,
                compute_profile_fingerprint=compute_profile_fingerprint(profile),
            )
            from dataclasses import replace
            from qwen_rl.contracts.gradient_partitions import accumulation_partitions
            partitioned = owner in self.store.work_stealing_pools
            partitions = accumulation_partitions(
                rows, self.config.get("gradient_accumulation_rows", DEFAULT_GRADIENT_ACCUMULATION_ROWS),
            ) if partitioned else []
            jobs = tuple(
                replace(job, job_id=wid + f"-r{i}", work_index=i,
                        token_count=sum(sum(rows[index]["policy_mask"][1:]) for index in ids))
                for i, ids in enumerate(partitions)
            ) if partitioned else (job,)
            window = PreparedWindow(
                schema=PREPARED_WINDOW_SCHEMA,
                aggregation=GRADIENT_CONTRIBUTION_CONTRACT,
                window_id=wid,
                attempt_id=aid,
                lineage_id=job.lineage_id,
                engine_epoch=job.engine_epoch,
                policy_version=job.policy_version,
                n_window_tokens=tokens,
                jobs=jobs,
                work_order=tuple(item.job_id for item in jobs),
                parameter_layout=self.core.parameter_layout,
                layout_fingerprint=job.layout_fingerprint,
                compute_profile=profile,
                update_context={
                    **({"trajectory_partitions": {item.job_id: ids for item, ids in zip(jobs, partitions)},
                        "trajectory_input_tokens": {
                            item.job_id: sum(len(rows[index]["input_ids"]) for index in ids)
                            for item, ids in zip(jobs, partitions)},
                        "trajectory_peak_input_tokens": {
                            item.job_id: max(len(rows[index]["input_ids"]) for index in ids)
                            for item, ids in zip(jobs, partitions)}}
                       if partitioned else {}),
                    "gradient_scope": STAGED_GROUP_SCOPE,
                    "staged_round_id": round_id,
                    "logical_run_id": logical_run_id,
                    "group_size": self.group_size,
                },
            )
            staged, paths = self._rowbundle_paths(
                window, {item.job_id: {"rows": rows, "source_worker_id": worker} for item in jobs}
            )
            try:
                self.store.enqueue_window(window, paths, compute_profile=profile)
            finally:
                staged.cleanup()
            self._staged_groups[key] = {"window": window, "fingerprint": fingerprint}

    def cancel_staged_round(self, round_id: str | None = None) -> int:
        if not self._staged_release_is_safe(self.core.status()):
            return 0
        released = 0
        with self._staging_lock:
            for key, value in list(self._staged_groups.items()):
                if round_id is not None and key[0] != round_id:
                    continue
                window = value["window"]
                released += int(self.store.release_staged_window(
                    window.window_id, window.attempt_id, reason="staged_round_released",
                ))
                del self._staged_groups[key]
            self.store.discard_sampling_round(round_id)
        return released

    def _sealed_staged_contributions(self, round_id, final_window, job_rows):
        from .staged_materialization import materialize_staged_round

        return materialize_staged_round(self, round_id, final_window, job_rows)

    def update(
        self,
        trajectories: list[dict[str, Any]],
        *,
        before_optimizer_step: Callable[[], None] | None = None,
        staged_round_id: str | None = None,
        recovery_context: dict[str, Any] | None = None,
        checkpoint_ready: Callable[[], None] | None = None,
        rollout_policy_ready: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if checkpoint_ready is not None:
            raise ValueError("central_gradient does not support local phase-shared checkpoint callbacks")
        if (isinstance(recovery_context, Mapping)
                and recovery_context.get("bepa_attempt_id") is not None
                and before_optimizer_step is None):
            raise ValueError("BEPA central-gradient updates require the optimizer-entry callback")
        if not trajectories:
            return {
                "updated": False,
                "skipped": True,
                "skip_reason": "empty_central_gradient_round",
                "policy_version": self.policy_version,
                "learner_start_version": self.policy_version,
                "groups_per_update": 0,
                "executor_capacity": self.groups_per_update_capacity,
                "gradient_window": None,
            }
        if not self._update_lock.acquire(blocking=False):
            raise RuntimeError("a central gradient update is already active")
        staged = None
        publication_future: Future[dict[str, Any]] | None = None
        timing_identity: tuple[str, str] | None = None

        def timing_event(name: str) -> None:
            if timing_identity is not None:
                self.core.note_timing_event(*timing_identity, name, persist=False)

        def timing_span(name: str, start: Mapping[str, Any]) -> None:
            if timing_identity is not None:
                self.core.record_timing_span(*timing_identity, name, start, persist=False)

        try:
            status = self.core.status()
            if status.get("dirty") is True:
                raise DirtyGradientCoreError(status.get("dirty_reason") or "central optimizer state is fenced")
            start_version = self.policy_version
            requested_groups = {
                row.get("group_id") for row in trajectories
                if isinstance(row, Mapping) and isinstance(row.get("group_id"), str)
            }
            if not requested_groups:
                raise ValueError("central-gradient round contains no complete group IDs")
            if len(requested_groups) > self.groups_per_update_capacity:
                raise ValueError(
                    f"central-gradient round has K={len(requested_groups)} groups above "
                    f"P={self.groups_per_update_capacity}"
                )
            from qwen_rl.learner.batch import validate_batch

            normalized, total_tokens, group_hashes, group_sizes = validate_batch(
                trajectories,
                learner_start_version=start_version,
                max_sequence_length=self.max_sequence_length,
                group_size=self.group_size,
                groups_per_update=len(requested_groups),
            )
            rows = list(normalized)
            grouped_rows: dict[str, list[dict[str, Any]]] = {}
            for row_id, row in enumerate(rows):
                group_id = row.get("group_id")
                if not isinstance(group_id, str) or not group_id.strip():
                    raise ValueError("central gradient groups require non-empty string group_id values")
                row["row_id"] = row_id
                grouped_rows.setdefault(group_id, []).append(row)
            actual_groups = len(grouped_rows)
            capacity = self.groups_per_update_capacity
            if actual_groups > self.groups_per_update_capacity:
                raise RuntimeError(
                    f"central-gradient round has K={actual_groups} groups above P="
                    f"{self.groups_per_update_capacity}"
                )
            if actual_groups < 1:
                raise RuntimeError("central batch validation did not produce the configured complete group count")

            reward_stds = []
            group_descriptors: list[dict[str, Any]] = []
            for group_id, group_rows in grouped_rows.items():
                fresh_rows = [row for row in group_rows if row.get("row_kind", "fresh_student") == "fresh_student"]
                source_workers = {
                    row.get("source_worker_id", row.get("worker_id")) for row in fresh_rows
                }
                if (len(fresh_rows) != self.group_size or len(source_workers) != 1
                        or not all(isinstance(worker_id, str) and worker_id.strip()
                                   for worker_id in source_workers)):
                    raise ValueError("each complete group must retain one source Worker across all fresh candidates")
                source_worker_id = next(iter(source_workers))
                from ...worker.episode_queue import validate_sampling_placement
                for row in fresh_rows:
                    validate_sampling_placement(row, source_worker_id, self.store.episode_queue)
                owner_executor_id = self.worker_executor_ids.get(source_worker_id)
                if owner_executor_id is None:
                    raise ValueError(
                        f"source Worker {source_worker_id!r} has no explicitly configured gradient executor"
                    )
                for row in fresh_rows:
                    row["source_worker_id"] = source_worker_id
                rewards = [float(row["reward"]) for row in group_rows]
                mean = sum(rewards) / len(rewards)
                reward_stds.append(math.sqrt(sum((value - mean) ** 2 for value in rewards) / (len(rewards) - 1)))
                token_count = sum(
                    sum(int(flag) for flag in row["policy_mask"][1:]) for row in group_rows
                )
                if token_count < 1:
                    raise ValueError("complete gradient group has zero causal policy tokens")
                group_descriptors.append({
                    "group_id": group_id,
                    "source_worker_id": source_worker_id,
                    "owner_executor_id": owner_executor_id,
                    "token_count": token_count,
                    "rows": group_rows,
                })
            self._last_reward_stats = {
                "group_reward_std_min": min(reward_stds),
                "group_reward_std_mean": sum(reward_stds) / len(reward_stds),
                "group_reward_std_max": max(reward_stds),
                "group_zero_std_fraction": 0.0,
            }
            behavior_versions = {
                row["policy_version"] for row in rows if row.get("row_kind") == "fresh_student"
            }
            if len(behavior_versions) != 1:
                raise RuntimeError("validated central window lost its single fresh behavior version")
            behavior_version = next(iter(behavior_versions))
            if behavior_version > start_version:
                raise RuntimeError("validated central window behavior policy is newer than central adapter")

            batch_fingerprint = _sha256_bytes(_canonical_json({
                "policy_version": start_version,
                "rows": rows,
                "update_context": recovery_context or {},
            }))
            window_id = f"w{start_version:06d}-{batch_fingerprint[:32]}"
            attempt_id = f"a{batch_fingerprint[32:64]}"
            job_rows: dict[str, dict[str, Any]] = {}
            specs = []
            for index, group in enumerate(group_descriptors):
                job_id = f"{window_id}-g{index:04d}"
                job_rows[job_id] = group
                specs.append(GradientJobSpec(
                    job_id,
                    group["token_count"],
                    group["group_id"],
                    group["owner_executor_id"],
                ))
            compute_profile = {
                "base_artifact_id": self._ternary_identity["source"]["ninfer_artifact"]["artifact_id"],
                "loss_profile": GRADIENT_LOSS_PROFILE,
                "correction_cap": self.importance_sampling_cap,
            }
            window = self.core.prepare_window(
                window_id,
                attempt_id,
                start_version,
                total_tokens,
                specs,
                compute_profile=compute_profile,
                update_context={
                    **dict(recovery_context or {}),
                    "group_size": self.group_size,
                    "groups_per_update": actual_groups,
                    "groups_per_update_capacity": capacity,
                    "groups_per_update_auto": self.groups_per_update_auto,
                },
                before_optimizer_step=before_optimizer_step,
            )
            def checkpoint_staged(notice: Mapping[str, Any]) -> None:
                nonlocal publication_future
                if publication_future is not None:
                    raise RuntimeError("checkpoint staged callback was invoked twice")
                if (notice.get("window_id") != window.window_id
                        or notice.get("attempt_id") != window.attempt_id):
                    raise RuntimeError("checkpoint staged callback named a different window")
                status = self.core.status()
                next_version = status["policy_version"]
                if next_version != start_version + 1:
                    raise RuntimeError("staged checkpoint did not advance exactly one policy")
                self.policy_version = next_version
                rollout_released = threading.Event()

                def release_rollout(publication: dict[str, Any]) -> None:
                    # Give the small immutable inference artifact first access
                    # to disk, then let network ACK and full-state I/O progress
                    # independently. A slow remote Worker cannot delay saving.
                    rollout_released.set()
                    if staged_round_id is not None:
                        wait_started = self.core.timing_mark()
                        try:
                            self.store.wait_for_sessions_released(
                                staged_round_id, timeout=self.gradient_wait_timeout,
                                cancel_event=self._shutdown_requested,
                            )
                        finally:
                            timing_span("worker_return_before_publication", wait_started)
                    if rollout_policy_ready is not None:
                        rollout_policy_ready(publication)

                def publish_for_checkpoint() -> dict[str, Any]:
                    try:
                        return self.publish(
                            rollout_ready=release_rollout,
                            parameter_state=notice["trainable_state_snapshot"],
                        )
                    finally:
                        # Publication failure must never strand a completed
                        # optimizer step without its full checkpoint attempt.
                        rollout_released.set()

                publication_future = self._checkpoint_publications.submit(publish_for_checkpoint)
                rollout_released.wait()

            self.core.register_checkpoint_staged(window, checkpoint_staged)
            timing_identity = (window.window_id, window.attempt_id)
            timing_event("learner_window_prepared")
            stage_started = self.core.timing_mark()
            staged_contributions = None
            if staged_round_id is not None:
                materialized = self._sealed_staged_contributions(staged_round_id,window,job_rows)
                staged_contributions = materialized.contributions
                timing_span("staged_materialization_total", stage_started)
                # One final global window is the sole optimizer/checkpoint authority.
                try:
                    raw_receipt = self.core.apply_prepared_window(window, materialized.preparation).to_dict()
                finally:
                    self.core.discard_prepared_gradients(materialized.preparation)
            else:
                staged, rowbundle_paths = self._rowbundle_paths(window, job_rows)
                timing_span("learner_rowbundle_stage", stage_started)
                stage_started = self.core.timing_mark()
                self.store.enqueue_window(
                    window,
                    rowbundle_paths,
                    compute_profile=compute_profile,
                )
                timing_span("learner_window_enqueue", stage_started)
                # Re-enter the HTTP apply path after registering the callback. This
                # also recovers a fully accepted window after a process restart.
                stage_started = self.core.timing_mark()
                self._resume_ready_windows_sync()
                timing_span("learner_ready_window_reconcile", stage_started)
                stage_started = self.core.timing_mark()
                raw_receipt = self.store.wait_for_window(
                    window.window_id,
                    window.attempt_id,
                    timeout=self.gradient_wait_timeout,
                )
                timing_span("learner_wait_for_window_receipt", stage_started)
            timing_event("learner_window_receipt_observed")
            receipt = WindowStepReceipt.from_dict(raw_receipt)
            if receipt.policy_version_before != start_version or receipt.policy_version != start_version + 1:
                raise RuntimeError("central gradient receipt did not prove exactly one policy version advance")
            self.policy_version = receipt.policy_version
            stage_started = self.core.timing_mark()
            self._sync_recovery_metadata()
            timing_span("post_commit_recovery_metadata_sync", stage_started)
            stage_started = self.core.timing_mark()
            diagnostic_contributions = staged_contributions if staged_contributions is not None else self.store.load_contribution_diagnostics(
                window.window_id, window.attempt_id,
            )
            timing_span("post_commit_contribution_reload", stage_started)
            stage_started = self.core.timing_mark()
            self._last_gradient_diagnostics = self._aggregate_diagnostics(diagnostic_contributions)
            if staged_contributions is not None:
                # Diagnostics are now scalar metadata; release the per-group tensors
                # and final private preparation before waiting for publication.
                del materialized
            del diagnostic_contributions, staged_contributions
            timing_span("post_commit_gradient_diagnostics", stage_started)
            self.evidence["last_update_gradient_norm_before_clip"] = receipt.gradient_norm_before_clip
            self.evidence["last_update_policy_tokens"] = total_tokens
            self.evidence["last_update_job_count"] = len(window.jobs)
            self.evidence["last_update_reward_stats"] = dict(self._last_reward_stats)
            self.evidence["last_update_gradient_diagnostics"] = dict(self._last_gradient_diagnostics)
            if publication_future is None:
                raise RuntimeError("completed async checkpoint omitted its publication task")
            publication = publication_future.result()
            timing_event("learner_publication_returned")
            row_kind_counts = {
                kind: sum(row.get("row_kind", "fresh_student") == kind for row in rows)
                for kind in ("fresh_student", "native_replay", "guided_replay", "zero_credit")
            }
            return {
                "updated": True,
                "policy_version": self.policy_version,
                "learner_start_version": start_version,
                "behavior_version": behavior_version,
                "policy_lag": start_version - behavior_version,
                "importance_correction_enabled": True,
                "importance_correction_cap": self.importance_sampling_cap,
                "importance_correction_metrics": dict(self._last_gradient_diagnostics),
                "gradient_norm_before_clip": receipt.gradient_norm_before_clip,
                "trajectory_count": len(rows),
                "group_count": len(group_sizes),
                "policy_tokens": total_tokens,
                "row_kind_counts": row_kind_counts,
                "groups_per_update": actual_groups,
                "groups_per_update_capacity": capacity,
                "microbatch_size": self.microbatch_size,
                "gradient_window": receipt.to_dict(),
                "initial_prompt_hashes": group_hashes,
                "reward_stats": dict(self._last_reward_stats),
                "publication": publication,
                "staging_timings": self.evidence.get("last_staging_timings") if staged_round_id is not None else None,
            }
        finally:
            # A failed update's traceback must not retain aggregate group tensors.
            materialized = None
            staged_contributions = None  # noqa: F841 - release tensors held by traceback locals.
            diagnostic_contributions = None  # noqa: F841 - release tensors held by traceback locals.
            # A failed checkpoint must not leave a publisher touching policy
            # files after update/run cleanup releases its references.
            if publication_future is not None:
                try:
                    publication_future.result()
                except BaseException:
                    # The original checkpoint/transport failure remains primary;
                    # success-path publication errors were raised above.
                    pass
            if staged is not None:
                cleanup_started = self.core.timing_mark()
                staged.cleanup()
                timing_span("learner_staged_rowbundle_cleanup", cleanup_started)
            sync_started = self.core.timing_mark()
            self._sync_recovery_metadata()
            timing_span("learner_final_recovery_metadata_sync", sync_started)
            timing_event("learner_update_cleanup_complete")
            if timing_identity is not None:
                self.core.persist_timing_receipt(*timing_identity)
            try:
                if staged_round_id is not None:
                    self.cancel_staged_round(staged_round_id)
            finally:
                self._update_lock.release()

    @property
    def recovery(self) -> dict[str, Any]:
        return self.recovery_metadata

    @property
    def supports_checkpoint_overlap(self) -> bool:
        return False

    def save_recovery_snapshot(
        self,
        *,
        update_context: dict[str, Any] | None = None,
        reason: str = "explicit",
    ) -> dict[str, Any]:
        generation = self.core.save_checkpoint(update_context=update_context, reason=reason)
        self._sync_recovery_metadata()
        return {
            "snapshot_id": self.core.status().get("checkpoint_id"),
            "snapshot_sha256": None,
            "path": self.core.status().get("checkpoint_path"),
            "checkpoint_generation": generation,
            "policy_version": self.policy_version,
        }

    def request_shutdown(self) -> None:
        """Wake pre-optimizer waits before joining the controller's update thread."""
        self._shutdown_requested.set()
        self._wake_training_policy_waiters()

    def shutdown(self) -> None:
        self._shutdown_requested.set()
        self._wake_training_policy_waiters()
        app_state = getattr(getattr(self, "gradient_http_app", None), "state", None)
        gradient_ipc = getattr(app_state, "gradient_ipc", None)
        try:
            try:
                if gradient_ipc is not None:
                    gradient_ipc.close()
            finally:
                try:
                    self.cancel_staged_round()
                    try:
                        self.core.close()
                    finally:
                        publisher = getattr(self, "_checkpoint_publications", None)
                        if publisher is not None:
                            publisher.shutdown(wait=True, cancel_futures=False)
                    self._sync_recovery_metadata()
                finally:
                    cache = getattr(self, "_training_policy_cache", None)
                    if cache is not None:
                        cache.cleanup()
        finally:
            store = getattr(self, "store", None)
            if store is not None:
                store.clear_memory_payloads()
            owner_lock = getattr(self, "_process_owner_lock", None)
            if owner_lock is not None:
                owner_lock.close()
                self._process_owner_lock = None


def build_gradient_learner_engine(
    config: Mapping[str, Any],
    *,
    tokenizer: Any = None,
    process_owner_lock: Any = None,
    recovery_only: bool = False,
    preserve_control_files: bool | None = None,
) -> GradientLearnerEngine:
    """Construct the central optimizer, optionally without runtime services."""
    engine = GradientLearnerEngine.__new__(GradientLearnerEngine)
    try:
        GradientLearnerEngine.__init__(
            engine, config, tokenizer=tokenizer, process_owner_lock=process_owner_lock,
            recovery_only=recovery_only, preserve_control_files=preserve_control_files,
        )
    except BaseException:
        app_state = getattr(getattr(engine, "gradient_http_app", None), "state", None)
        gradient_ipc = getattr(app_state, "gradient_ipc", None)
        if gradient_ipc is not None:
            try:
                gradient_ipc.close()
            except BaseException:
                pass
        core = getattr(engine, "core", None)
        if core is not None:
            try:
                core.close()
            except BaseException:
                pass
        publisher = getattr(engine, "_checkpoint_publications", None)
        if publisher is not None:
            try:
                publisher.shutdown(wait=True, cancel_futures=False)
            except BaseException:
                pass
        cache = getattr(engine, "_training_policy_cache", None)
        if cache is not None:
            try:
                cache.cleanup()
            except BaseException:
                pass
        store = getattr(engine, "store", None)
        if store is not None:
            try:
                store.clear_memory_payloads()
            except BaseException:
                pass
        owner_lock = getattr(engine, "_process_owner_lock", None)
        if owner_lock is not None:
            try:
                owner_lock.close()
            except BaseException:
                pass
        raise
    return engine


__all__ = ["GradientLearnerEngine", "build_gradient_learner_engine"]
