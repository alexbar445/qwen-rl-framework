"""CPU controller for the native NVFP4 rank service and durable LoRA state.

The controller owns publication and recovery. GPU models, Trainer updates and
collectives belong exclusively to the private rank service.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from typing import Any, Callable
from uuid import uuid4

import torch

from .backend import RankComputeBackend
from .core import LearnerEngineCore
from .parallel_recovery import OPTIMIZER_SCHEMA
from .recovery import RecoveryError, RecoveryStore, training_compatibility


class _AdapterView(torch.nn.Module):
    """A CPU copy of trainable LoRA tensors, never a second base model."""
    def __init__(self, payload: dict[str, Any]):
        super().__init__()
        from peft import LoraConfig
        self.adapter_parameters: dict[str, torch.nn.Parameter] = {}
        configs = payload["peft_config"]
        if set(configs) != {"default"}:
            raise RuntimeError("native learner requires exactly one default LoRA adapter")
        self.peft_config = {"default": LoraConfig(**configs["default"])}
        self.active_adapter = "default"
        self.refresh(payload["adapter_state"])
        self._build_tuner_metadata()

    def _build_tuner_metadata(self):
        """Build PEFT's module metadata with meta base layers and CPU LoRA only."""
        from peft.tuners.lora.layer import Linear
        config = self.peft_config["default"]
        if config.use_dora or config.bias != "none":
            raise ValueError("native controller supports ordinary bias-free LoRA")
        pairs = {}
        for name in self.adapter_parameters:
            if name.endswith(".lora_A.default.weight"):
                prefix = name.removesuffix(".lora_A.default.weight")
                other = prefix + ".lora_B.default.weight"
                if other not in self.adapter_parameters:
                    raise RuntimeError("incomplete LoRA A/B pair")
                pairs[prefix] = (name, other)
        if len(pairs) * 2 != len(self.adapter_parameters):
            raise RuntimeError("unsupported or incomplete LoRA parameter inventory")
        for prefix, (a_name, b_name) in pairs.items():
            a, b = self.adapter_parameters[a_name], self.adapter_parameters[b_name]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise RuntimeError("invalid full LoRA factor geometry")
            node = self
            parts = prefix.split(".")
            for part in parts[:-1]:
                if part not in node._modules:
                    node.add_module(part, torch.nn.Module())
                node = node._modules[part]
            base = torch.nn.Linear(a.shape[1], b.shape[0], bias=False, device="meta", dtype=torch.bfloat16)
            base.weight.requires_grad_(False)
            layer = Linear(base, adapter_name="default", config=config, r=a.shape[0],
                           lora_alpha=config.lora_alpha, lora_dropout=0.0, init_lora_weights=False)
            layer.lora_A["default"].weight = a
            layer.lora_B["default"].weight = b
            node.add_module(parts[-1], layer)

    def refresh(self, state: dict[str, torch.Tensor]) -> None:
        if not state or any("lora_" not in name for name in state):
            raise RuntimeError("rank service must return only complete trainable LoRA state")
        if self.adapter_parameters and set(self.adapter_parameters) != set(state):
            raise RuntimeError("rank service changed the adapter parameter inventory")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
                raise RuntimeError("rank service returned invalid CPU LoRA state")
            if name in self.adapter_parameters:
                target = self.adapter_parameters[name]
                if target.shape != value.shape or target.dtype != value.dtype:
                    raise RuntimeError("rank service changed LoRA parameter shape or dtype")
                with torch.no_grad():
                    target.copy_(value)
            else:
                self.adapter_parameters[name] = torch.nn.Parameter(value.detach().clone(), requires_grad=True)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        for name, parameter in self.adapter_parameters.items():
            yield (f"{prefix}.{name}" if prefix else name), parameter

    def state_dict(self):
        return {name: value.detach().clone() for name, value in self.adapter_parameters.items()}

    def save_pretrained(
        self,
        directory,
        *,
        safe_serialization=True,
        save_embedding_layers=False,
        state_dict=None,
    ):
        if safe_serialization is not True or save_embedding_layers is not False:
            raise ValueError("native publication requires standard safetensors LoRA only")
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        state = get_peft_model_state_dict(
            self,
            state_dict=self.state_dict() if state_dict is None else state_dict,
            save_embedding_layers=False,
            adapter_name="default",
        )
        # PEFT 0.21 serializes sets as lists without sorting. Keep the
        # immutable publication byte identity stable across Python hash seeds.
        config = copy.deepcopy(self.peft_config["default"])
        for name, value in vars(config).items():
            if isinstance(value, (set, frozenset)):
                setattr(config, name, sorted(value))
        config.save_pretrained(path)
        save_file({name: value.contiguous() for name, value in state.items()},
                  str(path / "adapter_model.safetensors"), metadata={"format": "pt"})


class _OptimizerView:
    """Optimizer group identity plus a complete, explicitly distributed payload."""
    def __init__(self, engine):
        self.engine = engine
        self.param_groups = []
        for group in engine._snapshot["parameter_groups"]:
            self.param_groups.append({
                **copy.deepcopy(group["options"]),
                "params": [engine.model.adapter_parameters[name] for name in group["parameter_names"]],
            })

    def state_dict(self):
        state = self.engine._snapshot
        result = {
            "schema": OPTIMIZER_SCHEMA,
            "parallelism": state["parallelism"], "world_size": state["world_size"],
            "layout_signature": state["layout_signature"],
            "parameter_groups": copy.deepcopy(state["parameter_groups"]),
            "rank_layouts": copy.deepcopy(state["rank_layouts"]),
            "rank_parameter_groups": copy.deepcopy(state.get("rank_parameter_groups")),
            "rank_states": state["optimizer_states"],
        }
        if (self.engine.config.get("model_backend") == "ternary"
                and self.engine.config.get("parallelism") == "tp"):
            signature = state.get("t2_tp_layout_signature")
            recipe = state.get("t2_tp_layout_recipe")
            if not isinstance(signature, str) or len(signature) != 64:
                raise RecoveryError("T2 TP snapshot has no topology layout signature")
            if not isinstance(recipe, str) or not recipe or recipe != recipe.strip():
                raise RecoveryError("T2 TP snapshot has no topology layout recipe")
            result["t2_tp_layout_signature"] = signature
            result["t2_tp_layout_recipe"] = recipe
        return result


class _CachedTelemetry:
    def __init__(self, backend):
        self.backend = backend

    def snapshot(self):
        return copy.deepcopy(self.backend.cached_status.get("telemetry", {}))


class _TrainerView:
    def __init__(self, engine):
        from transformers import TrainerState
        self.engine = engine
        self.optimizer = _OptimizerView(engine)
        self.lr_scheduler = SimpleNamespace()
        self.args = SimpleNamespace(lr_scheduler_type="constant")
        self.accelerator = SimpleNamespace(scaler=None)
        self.state = TrainerState(**engine._snapshot["trainer_state"])
        self.telemetry = _CachedTelemetry(engine.backend)
        self.failure_stage = "idle"
        self.gradient_norm_before_clip = None
        self.checkpoint_collection_performance = copy.deepcopy(
            engine._snapshot.get("checkpoint_performance", {})
        )
        self.checkpoint_save_performance: dict[str, float] = {}

    @property
    def failure_stage(self):
        if self._failure_stage == "optimizer":
            return self.engine.backend.cached_status.get("phase", "optimizer")
        return self._failure_stage

    @failure_stage.setter
    def failure_stage(self, stage):
        self._failure_stage = stage

    def quality_snapshot(self):
        return copy.deepcopy(self.engine.backend.cached_status.get("quality", {}))

    def _save_optimizer_and_scheduler(self, directory):
        path = Path(directory)
        self.checkpoint_save_performance = {}
        started = time.perf_counter()
        optimizer_state = self.optimizer.state_dict()
        self.checkpoint_save_performance["optimizer_view_state_dict_seconds"] = (
            time.perf_counter() - started
        )
        started = time.perf_counter()
        torch.save(optimizer_state, path / "optimizer.pt")
        self.checkpoint_save_performance["optimizer_torch_save_wall_seconds"] = (
            time.perf_counter() - started
        )
        del optimizer_state
        scheduler_state = {
            "schema": "qwen-rl-parallel-scheduler/v1",
            "rank_states": self.engine._snapshot["scheduler_states"],
        }
        started = time.perf_counter()
        torch.save(scheduler_state, path / "scheduler.pt")
        self.checkpoint_save_performance["scheduler_torch_save_wall_seconds"] = (
            time.perf_counter() - started
        )

    def _save_rng_state(self, directory):
        rng_state = {
            "schema": "qwen-rl-parallel-rng/v1",
            "rank_states": self.engine._snapshot["rng_states"],
        }
        started = time.perf_counter()
        torch.save(rng_state, Path(directory) / "rng_state.pth")
        self.checkpoint_save_performance["rng_torch_save_wall_seconds"] = (
            time.perf_counter() - started
        )

    def _load_optimizer_and_scheduler(self, directory):
        import json
        from .parallel_recovery import validate_parallel_optimizer
        path = Path(directory)
        optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
        validate_parallel_optimizer(optimizer, self.engine._snapshot["parameter_groups"], self.engine.model)
        if optimizer["layout_signature"] != self.engine._snapshot["layout_signature"]:
            raise RecoveryError("parallel optimizer topology/layout changed")
        is_t2_tp = (self.engine.config.get("model_backend") == "ternary"
                    and self.engine.config.get("parallelism") == "tp")
        tp_layout_fields = ("t2_tp_layout_signature", "t2_tp_layout_recipe")
        if is_t2_tp:
            for key in tp_layout_fields:
                saved, active = optimizer.get(key), self.engine._snapshot.get(key)
                if not isinstance(saved, str) or saved != active:
                    raise RecoveryError(f"T2 TP optimizer {key} differs from the active model layout")
            if (len(optimizer["t2_tp_layout_signature"]) != 64
                    or not optimizer["t2_tp_layout_recipe"].strip()):
                raise RecoveryError("T2 TP optimizer topology metadata is malformed")
        elif any(key in optimizer for key in tp_layout_fields):
            raise RecoveryError("non-TP optimizer snapshot contains T2 TP topology metadata")
        scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
        # NumPy RNG contains NumPy objects. This file belongs to a verified local
        # snapshot, not to an untrusted model or remote Worker response.
        rng = torch.load(path / "rng_state.pth", map_location="cpu", weights_only=False)
        if scheduler.get("schema") != "qwen-rl-parallel-scheduler/v1" or rng.get("schema") != "qwen-rl-parallel-rng/v1":
            raise RecoveryError("parallel scheduler/RNG snapshot schema changed")
        size = self.engine._snapshot["world_size"]
        if len(scheduler["rank_states"]) != size or len(rng["rank_states"]) != size:
            raise RecoveryError("parallel snapshot is missing scheduler/RNG rank state")
        payload = dict(self.engine._snapshot)
        for key in tp_layout_fields:
            payload.pop(key, None)
        payload.update({
            "adapter_state": self.engine.model.state_dict(),
            "optimizer_states": optimizer["rank_states"],
            "rank_layouts": optimizer["rank_layouts"],
            "rank_parameter_groups": optimizer.get("rank_parameter_groups"),
            "scheduler_states": scheduler["rank_states"],
            "rng_states": rng["rank_states"],
            "trainer_state": json.loads((path / "trainer_state.json").read_text()),
        })
        if is_t2_tp:
            payload.update({key: optimizer[key] for key in tp_layout_fields})
        self.engine.backend.restore(payload)
        self.engine._refresh_snapshot()

    def _load_rng_state(self, directory):
        # Collective restore above installs each rank's own RNG last, after
        # model/optimizer setup. Do not overwrite it with controller CPU RNG.
        return None


class ParallelLearnerEngine(LearnerEngineCore):
    """Native rank service behind the existing publication/recovery contract."""
    def __init__(self, config: dict[str, Any], *, backend=None, tokenizer=None):
        from .parallel.client import ParallelBackendClient
        self.config = dict(config)
        self.max_padded_tokens_per_microbatch = self.config.get("max_padded_tokens_per_microbatch")
        if (self.max_padded_tokens_per_microbatch is not None
                and (type(self.max_padded_tokens_per_microbatch) is not int
                     or self.max_padded_tokens_per_microbatch < 1)):
            raise ValueError("max_padded_tokens_per_microbatch must be a positive integer")
        if (self.max_padded_tokens_per_microbatch is not None
                and self.config.get("parallelism", "single") != "single"):
            raise ValueError("max_padded_tokens_per_microbatch requires parallelism='single'")
        if self.config.get("parallelism") == "pp":
            if self.config.get("model_backend") != "ternary" or self.config.get("pipeline_stages") != 4:
                raise ValueError("PP Learner requires the ternary backend and exactly four pipeline stages")
        self.model_path = str(config["model_path"])
        self.tokenizer_path = str(config.get("tokenizer_path") or self.model_path)
        self.group_size = int(config.get("group_size", 3))
        self.groups_per_update = int(config.get("groups_per_update", 2))
        self.microbatch_size = int(config.get("microbatch_size", 1))
        self.max_sequence_length = int(config.get("max_sequence_length", 65536))
        self.lora_rank = int(config.get("lora_rank", 16))
        self.lora_alpha = int(config.get("lora_alpha", 16))
        self.lr = float(config.get("lr", 1e-6))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.importance_sampling_cap = float(config.get("importance_sampling_cap", 3.0))
        if not 2 <= self.max_sequence_length <= 65536 or self.group_size < 2 or self.groups_per_update < 1 or self.microbatch_size < 1:
            raise ValueError("invalid native learner sequence or group geometry")
        if not all(math.isfinite(v) and v > 0 for v in (self.lr, self.max_grad_norm)) or not math.isfinite(self.importance_sampling_cap) or self.importance_sampling_cap < 1:
            raise ValueError("invalid native optimizer/correction settings")
        self.device = torch.device("cpu")
        self.backend: RankComputeBackend = backend if backend is not None else ParallelBackendClient(config)
        try:
            self.evidence = dict(self.backend.start())
            self.evidence.update({
                "max_padded_tokens_per_microbatch": self.max_padded_tokens_per_microbatch,
                "dynamic_microbatch_enabled": self.max_padded_tokens_per_microbatch is not None,
            })
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
            self.tokenizer = tokenizer
            declared_tokenizer_files = config.get("tokenizer_files")
            if declared_tokenizer_files is not None:
                if not isinstance(declared_tokenizer_files, list) or not declared_tokenizer_files:
                    raise ValueError("tokenizer_files must be a non-empty hashed inventory when configured")
                self.evidence["tokenizer_files"] = copy.deepcopy(declared_tokenizer_files)
                self.evidence["tokenizer_path"] = self.tokenizer_path
            snapshot_started = time.perf_counter()
            self._snapshot = self.backend.snapshot()
            self._snapshot.setdefault("checkpoint_performance", {})[
                "controller_snapshot_rpc_wall_seconds"
            ] = time.perf_counter() - snapshot_started
            self.model = _AdapterView(self._snapshot)
            # AdapterView owns its own stable CPU parameter storage; retaining
            # the transport payload kept a second full adapter resident.
            self._snapshot.pop("adapter_state", None)
            self.trainer = _TrainerView(self)
            self._recovery_lock = threading.RLock()
            self.policy_version = 0
            self._durable_policy_version = None
            self._last_update_context = None
            self._recovery_dirty = False
            self._recovery_resumed = False
            self._loaded_snapshot = None
            self._engine_epoch = uuid4().hex
            self._recovery_metadata = {"enabled": False, "resumed": False}
            root = config.get("recovery_dir")
            self.recovery_store = RecoveryStore(
                root, mode=config.get("recovery_mode", "auto"),
                lineage_id=config.get("recovery_lineage_id"),
            ) if root else None
            self._recovery_mode = "resumed" if self.recovery_store and self.recovery_store.resumed else "fresh"
            self.checkpoint_dir = (self.recovery_store.checkpoint_dir if self.recovery_store else
                                   Path(config["checkpoint_dir"]).resolve())
            self.publication_dir = (self.recovery_store.publication_dir if self.recovery_store else
                                    self.checkpoint_dir.parent / "publications")
            self._configure_publication_state()
            self._recovery_compatibility = training_compatibility(
                config, model=self.model, tokenizer=self.tokenizer, trainer=self.trainer, evidence=self.evidence
            )
            self._recovery_compatibility["parallel"] = {
                "mode": self._snapshot["parallelism"], "world_size": self._snapshot["world_size"],
                "layout_signature": self._snapshot["layout_signature"],
            }
            if (self.config.get("model_backend") == "ternary"
                    and self.config.get("parallelism") == "tp"):
                signature = self._snapshot.get("t2_tp_layout_signature")
                recipe = self._snapshot.get("t2_tp_layout_recipe")
                if not isinstance(signature, str) or len(signature) != 64:
                    raise RuntimeError("T2 TP RankService omitted its topology layout signature")
                if not isinstance(recipe, str) or not recipe or recipe != recipe.strip():
                    raise RuntimeError("T2 TP RankService omitted its topology layout recipe")
                self._recovery_compatibility["parallel"].update({
                    "t2_tp_layout_signature": signature,
                    "t2_tp_layout_recipe": recipe,
                })
            if self.evidence.get("initial_adapter") is not None:
                self._recovery_compatibility["initial_adapter"] = copy.deepcopy(self.evidence["initial_adapter"])
            if self.recovery_store:
                restored = self.recovery_store.load(
                    model=self.model, trainer=self.trainer, compatibility=self._recovery_compatibility
                )
                if restored:
                    self.policy_version = int(restored["policy_version"])
                    self._engine_epoch = restored["engine_epoch"]
                    self._last_update_context = copy.deepcopy(restored.get("update_context"))
                    self._recovery_resumed = True
                    self._loaded_snapshot = {"policy_version": self.policy_version,
                                             "snapshot_sha256": restored["snapshot_sha256"]}
                    self._durable_policy_version = self.policy_version
                    self._refresh_recovery_metadata(
                        snapshot_sha256=restored["snapshot_sha256"], snapshot_id=restored["snapshot_id"],
                        snapshot_path=restored["path"], compatibility_sha256=restored["compatibility_sha256"]
                    )
                    self._adopt_restored_publication(restored.get("publication_hashes"))
                else:
                    self.publish()
                    self._write_recovery_snapshot(reason="initial", update_context=None)
        except BaseException:
            self.backend.shutdown()
            raise

    def _refresh_snapshot(self):
        from transformers import TrainerState
        snapshot_started = time.perf_counter()
        payload = self.backend.snapshot()
        checkpoint_performance = payload.get("checkpoint_performance")
        if not isinstance(checkpoint_performance, dict):
            checkpoint_performance = {}
        checkpoint_performance["controller_snapshot_rpc_wall_seconds"] = (
            time.perf_counter() - snapshot_started
        )
        payload["checkpoint_performance"] = checkpoint_performance
        if hasattr(self, "_snapshot") and payload["layout_signature"] != self._snapshot["layout_signature"]:
            raise RuntimeError("rank service changed its topology/parameter layout")
        if (self.config.get("model_backend") == "ternary"
                and self.config.get("parallelism") == "tp"):
            for key in ("t2_tp_layout_signature", "t2_tp_layout_recipe"):
                if payload.get(key) != self._snapshot.get(key):
                    raise RuntimeError(f"rank service changed T2 TP {key}")
        self.model.refresh(payload["adapter_state"])
        # Keep only optimizer/scheduler/RNG state after copying the adapter into
        # the controller view. The transport adapter is no longer needed.
        payload.pop("adapter_state", None)
        self._snapshot = payload
        self.trainer.state = TrainerState(**payload["trainer_state"])
        self.trainer.checkpoint_collection_performance = copy.deepcopy(checkpoint_performance)
        self.evidence.update(payload.get("evidence", {}))

    def logprobs(self, input_ids: list[int], policy_mask: list[int]) -> list[float]:
        if len(input_ids) != len(policy_mask) or not 2 <= len(input_ids) <= self.max_sequence_length:
            raise ValueError("native logprob input length or mask is invalid")
        values = self.backend.logprobs(input_ids, policy_mask)
        if len(values) != len(input_ids) or any(not math.isfinite(float(v)) for v in values):
            raise RuntimeError("rank service returned invalid causal logprobs")
        return [float(value) for value in values]

    def probe(self, lengths: list[int]):
        return self.backend.probe(lengths)

    @property
    def compute_status(self):
        status = self.backend.cached_status
        return {key: copy.deepcopy(status[key]) for key in (
            "state", "phase", "mode", "world_size", "devices", "error"
        ) if key in status}

    @property
    def supports_checkpoint_overlap(self) -> bool:
        return (
            self.config.get("model_backend") == "ternary"
            and self.config.get("parallelism") == "ddp"
            and self.config.get("gpu_sharing") == "phase_shared"
            and self.recovery_store is not None
        )

    def park_for_checkpoint(self):
        """Release ranks at the owned immutable-CPU checkpoint boundary only."""
        if not getattr(self, "_checkpoint_boundary_open", False):
            raise RecoveryError("checkpoint parking requires the suspended update callback boundary")
        return self.park()

    def update(self, trajectories, *, before_optimizer_step: Callable | None = None,
               recovery_context: dict | None = None,
               checkpoint_ready: Callable[[], None] | None = None):
        if self._recovery_dirty:
            raise RecoveryError("native learner has an uncommitted durability failure")
        if checkpoint_ready is not None and (
            not callable(checkpoint_ready) or not self.supports_checkpoint_overlap
        ):
            raise ValueError("checkpoint overlap requires a recoverable phase-shared T2 DDP engine")
        start_version = self.policy_version
        rows, total, prefixes, sizes = self._validated_batch(trajectories, learner_start_version=start_version)
        self.trainer.failure_stage = "optimizer"
        optimizer_entered = False
        def before_step():
            nonlocal optimizer_entered
            if before_optimizer_step is not None:
                before_optimizer_step()
            optimizer_entered = True
        try:
            result = self.backend.update(rows, before_optimizer_step=before_step,
                                         recovery_context=recovery_context)
        except BaseException:
            if optimizer_entered:
                self._recovery_dirty = True
                self._refresh_after_failed_durability()
            raise
        norm = result.get("gradient_norm_before_clip")
        if result.get("updated") is not True or isinstance(norm, bool) or not isinstance(norm, (float, int)) or not math.isfinite(norm) or norm <= 0:
            if optimizer_entered:
                self._recovery_dirty = True
                self._refresh_after_failed_durability()
            raise RuntimeError("native rank service did not prove one finite nonzero optimizer update")
        self.policy_version += 1
        self.trainer.gradient_norm_before_clip = float(norm)
        self._recovery_dirty = self.recovery_store is not None
        checkpoint_snapshot = None
        try:
            self._refresh_snapshot()
            self.evidence["last_update_gradient_norm_before_clip"] = float(norm)
            self.trainer.failure_stage = "publication"
            publication = self.publish()
            if checkpoint_ready is not None:
                # All rank RPCs and CPU captures have finished. The callback may
                # park ranks and start fenced Worker wake; it must not publish
                # or admit this version before the durable save below succeeds.
                self._checkpoint_boundary_open = True
                try:
                    checkpoint_ready()
                finally:
                    self._checkpoint_boundary_open = False
            if self.recovery_store:
                self.trainer.failure_stage = "checkpoint"
                checkpoint_snapshot = self._write_recovery_snapshot(
                    reason="optimizer_update", update_context=recovery_context
                )
            self._recovery_dirty = False
            if self.recovery_store:
                pointer = self.recovery_store.pointer
                self._refresh_recovery_metadata(
                    snapshot_sha256=pointer["manifest_sha256"], snapshot_id=pointer["snapshot_id"],
                    snapshot_path=str(self.recovery_store.current_snapshot_path),
                    compatibility_sha256=self._recovery_compatibility and self._recovery_metadata.get("compatibility_sha256"),
                )
        except BaseException:
            self._refresh_after_failed_durability()
            raise
        self.trainer.failure_stage = "idle"
        behavior = next(row["policy_version"] for row in rows if row["row_kind"] == "fresh_student")
        gradient_accumulation_steps = result.get("gradient_accumulation_steps")
        microbatch_plan = copy.deepcopy(result.get("microbatch_plan", {}))
        if isinstance(gradient_accumulation_steps, int) and not isinstance(gradient_accumulation_steps, bool):
            self.evidence["last_update_gradient_accumulation_steps"] = gradient_accumulation_steps
        self.evidence["last_microbatch_plan"] = copy.deepcopy(microbatch_plan)
        return {**result, "updated": True, "policy_version": self.policy_version,
                "learner_start_version": start_version, "behavior_version": behavior,
                "policy_lag": start_version - behavior, "policy_tokens": total,
                "max_padded_tokens_per_microbatch": self.max_padded_tokens_per_microbatch,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "microbatch_plan": microbatch_plan,
                "trajectory_count": len(rows), "group_count": len(sizes),
                "initial_prompt_hashes": prefixes, "publication": publication,
                "checkpoint_performance": None if checkpoint_snapshot is None else copy.deepcopy(
                    checkpoint_snapshot.get("checkpoint_performance", {})
                )}

    def park(self):
        if self.config.get("gpu_sharing") != "phase_shared":
            raise RuntimeError("Learner parking requires phase_shared")
        return self.backend.park()

    def wake(self):
        if self.config.get("gpu_sharing") != "phase_shared":
            raise RuntimeError("Learner wake requires phase_shared")
        return self.backend.wake()

    def shutdown(self):
        self.backend.shutdown()
