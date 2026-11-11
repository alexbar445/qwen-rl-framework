"""Real CPU HF causal-LM + PEFT LoRA learner for the small-model profile."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import threading
from typing import Any
import uuid

import numpy as np
import torch

from .checkpoint import HFCheckpointError, HFCheckpointStore


HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
HF_MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
_CHECKPOINT_SCHEMA = "qwen-rl-small-hf-lora/v1"
_METADATA_ID_SCHEMA = "local-policy-identity/v1"


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class HFEngine:
    """Single-process learner using an actual HF model and trainable PEFT LoRA.

    The public constructor accepts the flat runner configuration. Optional
    ``model`` and ``tokenizer`` arguments are an in-memory fixture seam; normal
    runs load the pinned public Hub model and tokenizer on CPU.
    """

    def __init__(
        self,
        config: dict[str, Any],
        run_dir: str | Path,
        resume: bool = False,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if not isinstance(config, dict):
            raise TypeError("HFEngine config must be a dict")
        if type(resume) is not bool:
            raise TypeError("resume must be a bool")
        self.config = dict(config)
        fixture_mode = model is not None and tokenizer is not None
        self._validate_config(allow_fixture=fixture_mode)
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.checkpoint_store = HFCheckpointStore(self.run_dir / "checkpoints")
        self._lock = threading.RLock()
        self._policy_refs: dict[int, int] = {}
        self._policy_snapshots: dict[int, dict[str, torch.Tensor]] = {}
        self._recovery_resumed = False
        self._recovery_dirty = False
        self._last_update_context: dict[str, Any] | None = None
        self._snapshot_id: str | None = None

        self.model_id = str(self.config.get("model_id", HF_MODEL_ID))
        self.model_revision = str(self.config.get("model_revision", HF_MODEL_REVISION))
        self._is_local_model = Path(self.model_id).expanduser().exists()
        self.device = torch.device("cpu")
        self.seed = int(self.config.get("seed", 7))
        self.group_size = int(self.config.get("group_size", 4))
        self.max_prompt_tokens = int(self.config.get("max_prompt_tokens", 256))
        self.epsilon = float(self.config.get("epsilon", 0.5))
        self.clip_ratio = float(self.config.get("clip_ratio", 0.2))
        self.importance_cap = float(self.config.get("importance_cap", 3.0))
        self.learning_rate = float(self.config.get("learning_rate", 1e-3))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.choices = tuple(self.config.get("choices", (" yes", " no")))
        self.lora_rank = int(self.config.get("lora_rank", 4))
        self.lora_alpha = int(self.config.get("lora_alpha", 8))
        self.lora_targets = tuple(self.config.get("lora_targets", ("q_proj", "v_proj")))

        self._seed_process(self.seed)
        torch.set_num_threads(int(self.config["num_threads"]))
        if model is None or tokenizer is None:
            if model is not None or tokenizer is not None:
                raise ValueError("model and tokenizer fixture values must be supplied together")
            model, tokenizer = self._load_hub_model()
        self.tokenizer = tokenizer
        self._validate_model_source(model)
        self.model = self._wrap_lora(model)
        self.model.to(device=self.device, dtype=torch.float32)
        self._configure_deterministic_eager_model()
        self._adapter_parameters = self._trainable_adapter_parameters()
        self._parameter_by_name = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.choice_token_ids = self._choice_ids()

        self.optimizer = torch.optim.AdamW(
            self._adapter_parameters,
            lr=self.learning_rate,
            betas=tuple(self.config.get("adam_betas", (0.9, 0.999))),
            eps=float(self.config.get("adam_eps", 1e-8)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda _step: 1.0,
        )
        self.policy_version = 0
        self.lineage_id = uuid.uuid4().hex
        self.model_identity = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": str(self.config.get("tokenizer_id", self.model_id)),
            "tokenizer_revision": str(self.config.get("tokenizer_revision", self.model_revision)),
        }
        self.compatibility = self._compatibility_identity()

        self.evidence = {
            "backend": "huggingface-transformers-peft-lora",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.model_identity["tokenizer_id"],
            "tokenizer_revision": self.model_identity["tokenizer_revision"],
            "base_identity": dict(self.model_identity),
            "adapter_structure": copy.deepcopy(self.compatibility["lora"]["trainable_structure"]),
            "adapter_weights_finite": True,
            "revision_is_immutable_hub_commit": (
                not self._is_local_model and bool(re.fullmatch(r"[0-9a-f]{40}", self.model_revision))
            ),
            "source_identity_mode": (
                "immutable_hub_commit"
                if not self._is_local_model and re.fullmatch(r"[0-9a-f]{40}", self.model_revision)
                else "operator_declared_local_revision"
            ),
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": "eager",
            "optimizer": "torch.optim.AdamW",
            "scheduler": "torch.optim.lr_scheduler.LambdaLR",
            "scheduler_schedule": "constant",
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_targets": list(self.lora_targets),
            "lora_dropout": 0.0,
            "choices": list(self.choices),
            "choice_token_ids": list(self.choice_token_ids),
            "epsilon": self.epsilon,
            "behavior_probability": "log((1-epsilon)*restricted_softmax_probability+epsilon/2)",
            "loss": "token-mean clipped PPO using detached proximal p and capped detached exp(p-b); group-normalized advantages",
            "importance_correction_cap": self.importance_cap,
            "policy_identity_kind": "lineage_version_metadata_id_not_content_sha256",
            "publication_scope": "local_only",
            "checkpoint_schema": _CHECKPOINT_SCHEMA,
            "base_source_proof": "model id/revision plus lightweight structure and finite checks; no model payload hash",
        }

        if resume:
            self._restore_checkpoint()
            self._recovery_resumed = True
        else:
            if self.checkpoint_store.exists():
                raise HFCheckpointError(
                    "checkpoint already exists; pass resume=True to restore or choose a new run directory"
                )
            self._durable_policy_version = 0
            self._save_checkpoint(reason="lineage_initialized", update_context=None)
        self._policy_snapshots[self.policy_version] = self._adapter_state()
        self._set_runtime_threads()

    def _validate_config(self, *, allow_fixture: bool = False) -> None:
        defaults = {
            "model_id": HF_MODEL_ID,
            "model_revision": HF_MODEL_REVISION,
            "device": "cpu",
            "dtype": "float32",
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_targets": ["q_proj", "v_proj"],
            "lora_dropout": 0.0,
            "group_size": 4,
            "learning_rate": 1e-3,
            "seed": 7,
            "epsilon": 0.5,
            "clip_ratio": 0.2,
            "importance_cap": 3.0,
            "num_threads": 2,
            "max_prompt_tokens": 256,
            "max_filtered_groups": 64,
            "choices": [" yes", " no"],
        }
        for key, default in defaults.items():
            self.config.setdefault(key, default)
        if self.config["device"] != "cpu" or self.config["dtype"] != "float32":
            raise ValueError("the small HF backend requires device='cpu' and dtype='float32'")
        revision = self.config["model_revision"]
        model_id = self.config["model_id"]
        if (not isinstance(revision, str) or not revision
                or (not allow_fixture and not Path(str(model_id)).exists()
                    and not re.fullmatch(r"[0-9a-f]{40}", revision))):
            raise ValueError("Hub model_revision must be a fixed 40-character commit SHA")
        for key in ("lora_rank", "lora_alpha"):
            value = self.config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.config["lora_dropout"] != 0.0:
            raise ValueError("the small HF profile requires LoRA dropout 0.0")
        if list(self.config["lora_targets"]) != ["q_proj", "v_proj"]:
            raise ValueError("the small HF profile targets q_proj and v_proj only")
        for key, minimum in (("group_size", 2), ("max_prompt_tokens", 1), ("num_threads", 1)):
            value = self.config.get(key, defaults.get(key))
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key, value, lower, upper, include_lower in (
            ("epsilon", self.config["epsilon"], 0.0, 1.0, False),
            ("clip_ratio", self.config["clip_ratio"], 0.0, 1.0, False),
            ("importance_cap", self.config["importance_cap"], 1.0, math.inf, True),
            ("learning_rate", self.config["learning_rate"], 0.0, math.inf, False),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or (value < lower if include_lower else value <= lower)
                    or value >= upper):
                raise ValueError(f"{key} is outside its supported range")
        choices = self.config["choices"]
        if (not isinstance(choices, (list, tuple)) or len(choices) != 2
                or any(not isinstance(choice, str) or not choice for choice in choices)
                or choices[0] == choices[1]):
            raise ValueError("choices must contain two distinct nonempty strings")
        if not _is_nonnegative_int(self.config["seed"]):
            raise ValueError("seed must be a nonnegative integer")

    def _set_runtime_threads(self) -> None:
        torch.set_num_threads(int(self.config["num_threads"]))

    @staticmethod
    def _seed_process(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _load_hub_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = str(self.config["model_id"])
        revision = str(self.config["model_revision"])
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.get("tokenizer_id", model_id)),
            revision=str(self.config.get("tokenizer_revision", revision)),
            use_fast=True,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            trust_remote_code=False,
        )
        return model, tokenizer

    def _validate_model_source(self, model: Any) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("HF model must be a torch.nn.Module")
        model_config = getattr(model, "config", None)
        if model_config is None:
            raise ValueError("HF causal LM is missing its model config")
        if not bool(getattr(model_config, "is_encoder_decoder", False)):
            if not callable(getattr(model, "forward", None)):
                raise ValueError("HF causal LM has no forward method")
        for _name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        for field in ("attention_dropout", "dropout", "hidden_dropout_prob"):
            if hasattr(model_config, field):
                setattr(model_config, field, 0.0)
        setattr(model_config, "_attn_implementation", "eager")
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter.detach()).all()):
                raise ValueError(f"base model contains non-finite parameters at {name}")

    def _wrap_lora(self, model: torch.nn.Module) -> torch.nn.Module:
        from peft import LoraConfig, TaskType, get_peft_model

        if hasattr(model, "peft_config"):
            raise ValueError("HFEngine expects a fresh base model; LoRA is applied exactly once")
        lora = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=list(self.lora_targets),
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, lora)

    def _configure_deterministic_eager_model(self) -> None:
        self.model.config.use_cache = False
        setattr(self.model.config, "_attn_implementation", "eager")
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.model.eval()

    def _trainable_adapter_parameters(self) -> list[torch.nn.Parameter]:
        trainable = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise RuntimeError(f"non-LoRA parameter is unexpectedly trainable: {name}")
                if parameter.dtype != torch.float32 or parameter.device.type != "cpu":
                    raise RuntimeError(f"LoRA parameter is not CPU float32: {name}")
                trainable.append(parameter)
        if not trainable:
            raise RuntimeError("PEFT did not create trainable LoRA parameters")
        return trainable

    def _choice_ids(self) -> tuple[int, int]:
        encoded = [self.tokenizer.encode(choice, add_special_tokens=False) for choice in self.choices]
        if any(not isinstance(ids, list) or len(ids) != 1 for ids in encoded):
            raise ValueError("each configured yes/no choice must encode to exactly one token")
        ids = (int(encoded[0][0]), int(encoded[1][0]))
        if ids[0] == ids[1]:
            raise ValueError("yes/no choice tokens must be distinct")
        return ids

    def _adapter_state(self) -> dict[str, torch.Tensor]:
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if set(state) != set(self._parameter_by_name):
            raise RuntimeError("trainable LoRA parameter names changed unexpectedly")
        for name, value in state.items():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"LoRA state contains non-finite values: {name}")
        return state

    def _load_adapter_state(self, state: Any) -> None:
        if not isinstance(state, dict) or set(state) != set(self._parameter_by_name):
            raise HFCheckpointError("checkpoint LoRA parameter names do not match the loaded model")
        for name, tensor in state.items():
            parameter = self._parameter_by_name[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                    or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())):
                raise HFCheckpointError(f"checkpoint LoRA tensor is incompatible: {name}")
        with torch.no_grad():
            for name, tensor in state.items():
                self._parameter_by_name[name].copy_(tensor.to(device="cpu", dtype=torch.float32))

    def _compatibility_identity(self) -> dict[str, Any]:
        structures = [
            {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        model_structure = [
            {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in self.model.named_parameters()
            if "lora_" not in name
        ]
        try:
            source_identity = json.loads(json.dumps(
                self.config.get("source_identity"), sort_keys=True, allow_nan=False,
            ))
            reward_schema = json.loads(json.dumps(
                self.config.get("reward_schema"), sort_keys=True, allow_nan=False,
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError("source_identity and reward_schema must be JSON-safe") from exc
        return {
            "model_identity": dict(self.model_identity),
            "base_parameter_structure": model_structure,
            "lora": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "targets": list(self.lora_targets),
                "dropout": 0.0,
                "trainable_structure": structures,
            },
            "choices": list(self.choices),
            "choice_token_ids": list(self.choice_token_ids),
            "optimizer": "AdamW",
            "optimizer_config": {
                "learning_rate": self.learning_rate,
                "betas": list(self.config.get("adam_betas", (0.9, 0.999))),
                "eps": float(self.config.get("adam_eps", 1e-8)),
                "weight_decay": float(self.config.get("weight_decay", 0.0)),
            },
            "scheduler": "LambdaLR-constant",
            "learning_rate": self.learning_rate,
            "group_size": self.group_size,
            "seed": self.seed,
            "num_threads": int(self.config["num_threads"]),
            "max_prompt_tokens": self.max_prompt_tokens,
            "epsilon": self.epsilon,
            "clip_ratio": self.clip_ratio,
            "importance_cap": self.importance_cap,
            "source_identity": source_identity,
            "reward_schema": reward_schema,
        }

    def _numpy_rng_state(self) -> dict[str, Any]:
        algorithm, keys, position, has_gauss, cached = np.random.get_state()
        return {
            "algorithm": algorithm,
            "keys": torch.from_numpy(keys.astype(np.int64, copy=True)),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        }

    @staticmethod
    def _restore_numpy_rng(state: Any) -> None:
        required = {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
        if not isinstance(state, dict) or set(state) != required:
            raise HFCheckpointError("checkpoint NumPy RNG state is incomplete")
        keys = state["keys"]
        if not isinstance(keys, torch.Tensor) or keys.ndim != 1 or keys.dtype != torch.int64:
            raise HFCheckpointError("checkpoint NumPy RNG key array is invalid")
        np.random.set_state((
            str(state["algorithm"]),
            keys.cpu().numpy().astype(np.uint32, copy=True),
            int(state["position"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        ))

    def _rng_state(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": self._numpy_rng_state(),
            "torch_cpu": torch.get_rng_state().clone(),
        }

    def _restore_rng_state(self, state: Any) -> None:
        required = {"python", "numpy", "torch_cpu"}
        if not isinstance(state, dict) or set(state) != required:
            raise HFCheckpointError("checkpoint RNG state is incomplete")
        torch_state = state["torch_cpu"]
        if not isinstance(torch_state, torch.Tensor) or torch_state.dtype != torch.uint8:
            raise HFCheckpointError("checkpoint PyTorch CPU RNG state is invalid")
        try:
            random.setstate(state["python"])
            self._restore_numpy_rng(state["numpy"])
            torch.set_rng_state(torch_state.cpu())
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HFCheckpointError(f"checkpoint RNG state is incompatible: {exc}") from exc

    @staticmethod
    def _copy_optimizer_state(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: HFEngine._copy_optimizer_state(item) for key, item in value.items()}
        if isinstance(value, list):
            return [HFEngine._copy_optimizer_state(item) for item in value]
        if isinstance(value, tuple):
            return tuple(HFEngine._copy_optimizer_state(item) for item in value)
        return copy.deepcopy(value)

    def _checkpoint_payload(self, *, reason: str, update_context: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "model_identity": dict(self.model_identity),
            "compatibility": copy.deepcopy(self.compatibility),
            "lineage_id": self.lineage_id,
            "policy_version": self.policy_version,
            "optimizer_step": self.policy_version,
            "scheduler_step": int(self.scheduler.last_epoch),
            "adapter_state": self._adapter_state(),
            "optimizer_state": self._copy_optimizer_state(self.optimizer.state_dict()),
            "scheduler_state": self._copy_optimizer_state(self.scheduler.state_dict()),
            "rng_state": self._rng_state(),
            "recovery_context": copy.deepcopy(update_context),
            "reason": str(reason),
        }

    def _save_checkpoint(self, *, reason: str, update_context: dict[str, Any] | None) -> Path:
        if update_context is not None and not isinstance(update_context, dict):
            raise TypeError("recovery_context must be a dict or None")
        if update_context is not None:
            try:
                update_context = json.loads(json.dumps(update_context, sort_keys=True, allow_nan=False))
            except (TypeError, ValueError) as exc:
                raise ValueError("recovery_context must be JSON-safe") from exc
        path = self.checkpoint_store.save(
            self._checkpoint_payload(reason=reason, update_context=update_context)
        )
        self._durable_policy_version = self.policy_version
        self._last_update_context = copy.deepcopy(update_context)
        self._snapshot_id = f"{self.lineage_id}:v{self.policy_version}"
        return path

    def _restore_checkpoint(self) -> None:
        payload = self.checkpoint_store.load()
        required = {
            "schema", "model_identity", "compatibility", "lineage_id", "policy_version",
            "optimizer_step", "scheduler_step", "adapter_state", "optimizer_state",
            "scheduler_state", "rng_state", "recovery_context", "reason",
        }
        if set(payload) != required or payload.get("schema") != _CHECKPOINT_SCHEMA:
            raise HFCheckpointError("checkpoint schema or required state fields do not match")
        if payload.get("model_identity") != self.model_identity:
            raise HFCheckpointError("checkpoint model/tokenizer identity differs from this run")
        if payload.get("compatibility") != self.compatibility:
            raise HFCheckpointError("checkpoint LoRA or optimizer compatibility differs from this run")
        lineage_id = payload.get("lineage_id")
        version = payload.get("policy_version")
        if not isinstance(lineage_id, str) or not lineage_id or not _is_nonnegative_int(version):
            raise HFCheckpointError("checkpoint lineage or policy version is invalid")
        if payload.get("optimizer_step") != version:
            raise HFCheckpointError("checkpoint optimizer step and policy version differ")
        if not isinstance(payload.get("recovery_context"), (dict, type(None))):
            raise HFCheckpointError("checkpoint recovery context must be a mapping or None")
        self._load_adapter_state(payload["adapter_state"])
        try:
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.scheduler.load_state_dict(payload["scheduler_state"])
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise HFCheckpointError(f"checkpoint optimizer/scheduler state is incompatible: {exc}") from exc
        if int(self.scheduler.last_epoch) != int(payload["scheduler_step"]):
            raise HFCheckpointError("checkpoint scheduler cursor differs from its stored scheduler state")
        if int(payload["scheduler_step"]) != version:
            raise HFCheckpointError("checkpoint scheduler step differs from the committed policy version")
        if version > 0 and not self.optimizer.state:
            raise HFCheckpointError("updated checkpoint is missing AdamW moment state")
        if version > 0 and int(self.scheduler.last_epoch) < version:
            raise HFCheckpointError("updated checkpoint scheduler is behind its optimizer")
        self._restore_rng_state(payload["rng_state"])
        self.policy_version = version
        self.lineage_id = lineage_id
        self._durable_policy_version = version
        self._last_update_context = copy.deepcopy(payload["recovery_context"])
        self._snapshot_id = f"{lineage_id}:v{version}"

    def _encode_prompt(self, prompt: str) -> list[int]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        rendered = f"Question: {prompt.strip()}\nAnswer:"
        ids = self.tokenizer.encode(rendered, add_special_tokens=True)
        if not isinstance(ids, list) or not ids or len(ids) > self.max_prompt_tokens:
            raise ValueError("encoded prompt is empty or exceeds max_prompt_tokens")
        return [int(token) for token in ids]

    def _constrained_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, -1, list(self.choice_token_ids)].float()
        return torch.log_softmax(logits, dim=-1)

    def _with_policy_version(self, version: int):
        class _PolicyContext:
            def __init__(inner_self, engine: "HFEngine", requested: int):
                inner_self.engine = engine
                inner_self.requested = requested
                inner_self.current = None
                inner_self.was_training = False

            def __enter__(inner_self):
                engine = inner_self.engine
                if inner_self.requested not in engine._policy_snapshots:
                    raise ValueError(f"policy version {inner_self.requested} is not retained")
                inner_self.was_training = engine.model.training
                if inner_self.requested != engine.policy_version:
                    inner_self.current = engine._adapter_state()
                    engine._load_adapter_state(engine._policy_snapshots[inner_self.requested])
                engine.model.eval()
                return engine

            def __exit__(inner_self, exc_type, exc, traceback):
                engine = inner_self.engine
                try:
                    if inner_self.current is not None:
                        engine._load_adapter_state(inner_self.current)
                finally:
                    engine.model.train(inner_self.was_training)

        return _PolicyContext(self, version)

    def sample_group(self, prompt: str, group_size: int, policy_version: int) -> list[dict[str, Any]]:
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
            raise ValueError("group_size must be an integer >= 2")
        if group_size != self.group_size:
            raise ValueError(f"group_size={group_size} differs from configured group_size={self.group_size}")
        if not _is_nonnegative_int(policy_version):
            raise ValueError("policy_version must be a nonnegative integer")
        prompt_ids = self._encode_prompt(prompt)
        with self._lock, torch.no_grad(), self._with_policy_version(policy_version):
            if self._recovery_dirty:
                raise HFCheckpointError("cannot sample from a policy with an uncommitted optimizer update")
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            log_p = self._constrained_log_probs(ids)[0]
            p = torch.exp(log_p)
            q = (1.0 - self.epsilon) * p + self.epsilon / 2.0
            # A single multinomial call samples each row independently with replacement.
            choices = torch.multinomial(q, num_samples=group_size, replacement=True)
            sampled = choices.cpu().tolist()
            behavior = torch.log(q[choices]).cpu().tolist()
            outputs = []
            for choice_index, behavior_logprob in zip(sampled, behavior, strict=True):
                answer = self.choices[int(choice_index)]
                full_ids = [*prompt_ids, self.choice_token_ids[int(choice_index)]]
                mask = [0] * len(prompt_ids) + [1]
                behavior_logprobs = [0.0] * len(prompt_ids) + [float(behavior_logprob)]
                outputs.append({
                    "text": answer.strip(),
                    "completion": answer,
                    "input_ids": full_ids,
                    "policy_mask": mask,
                    "behavior_logprobs": behavior_logprobs,
                    "choice_index": int(choice_index),
                    "policy_version": int(policy_version),
                })
            return outputs

    @torch.no_grad()
    def logprobs(self, input_ids: list[int], policy_mask: list[int]) -> list[float]:
        if not isinstance(input_ids, list) or not isinstance(policy_mask, list):
            raise ValueError("input_ids and policy_mask must be lists")
        if len(input_ids) != len(policy_mask) or len(input_ids) < 2:
            raise ValueError("input_ids and policy_mask must align and contain a prompt plus one answer")
        if (any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in input_ids)
                or len(input_ids) - 1 > self.max_prompt_tokens):
            raise ValueError("input_ids contain an invalid token or exceed max_prompt_tokens")
        active = [index for index, value in enumerate(policy_mask) if value in (1, True)]
        if (active != [len(input_ids) - 1]
                or any(type(value) not in (int, bool) or value not in (0, 1) for value in policy_mask)
                or policy_mask[0] != 0):
            raise ValueError("small profile requires exactly one final policy token")
        try:
            choice_index = self.choice_token_ids.index(input_ids[-1])
        except ValueError as exc:
            raise ValueError("final policy token is outside the configured yes/no choices") from exc
        ids = torch.tensor([input_ids[:-1]], dtype=torch.long, device=self.device)
        with self._lock:
            if self._recovery_dirty:
                raise HFCheckpointError("cannot score a policy with an uncommitted optimizer update")
            was_training = self.model.training
            self.model.eval()
            try:
                log_p = self._constrained_log_probs(ids)[0, choice_index]
            finally:
                self.model.train(was_training)
        result = [0.0] * len(input_ids)
        result[-1] = float(log_p.cpu())
        return result

    def _validate_rows(self, trajectories: list[dict[str, Any]]):
        if not isinstance(trajectories, list) or not trajectories:
            raise ValueError("update requires at least one trajectory")
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        versions: set[int] = set()
        for index, raw in enumerate(trajectories):
            if not isinstance(raw, dict):
                raise ValueError(f"trajectory {index} must be a mapping")
            required = {"input_ids", "policy_mask", "behavior_logprobs", "policy_version", "reward", "group_id"}
            missing = required - raw.keys()
            if missing:
                raise ValueError(f"trajectory {index} is missing {sorted(missing)}")
            ids, mask, behavior = raw["input_ids"], raw["policy_mask"], raw["behavior_logprobs"]
            if (not isinstance(ids, (list, tuple)) or not isinstance(mask, (list, tuple))
                    or not isinstance(behavior, (list, tuple))):
                raise ValueError(f"trajectory {index} token, mask, and behavior fields must be sequences")
            if len(ids) != len(mask) or len(ids) != len(behavior) or len(ids) < 2:
                raise ValueError(f"trajectory {index} token, mask, and behavior lengths differ")
            if (any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids)
                    or any(type(flag) not in (int, bool) or flag not in (0, 1) for flag in mask)
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in behavior)):
                raise ValueError(f"trajectory {index} contains an invalid token, mask, or behavior value")
            ids = list(ids)
            mask = [int(flag) for flag in mask]
            behavior = [float(value) for value in behavior]
            if (any(flag not in (0, 1) for flag in mask)
                    or mask[0] != 0 or mask[:-1] != [0] * (len(mask) - 1) or mask[-1] != 1
                    or not all(math.isfinite(value) for value in behavior)
                    or any(value != 0.0 for value in behavior[:-1])):
                raise ValueError(f"trajectory {index} violates the one-token action layout")
            choice_index = raw.get("choice_index")
            if choice_index is None and ids[-1] in self.choice_token_ids:
                choice_index = self.choice_token_ids.index(ids[-1])
            if (isinstance(choice_index, bool) or not isinstance(choice_index, int)
                    or choice_index not in (0, 1)
                    or ids[-1] != self.choice_token_ids[choice_index]):
                raise ValueError(f"trajectory {index} choice_index does not match its selected token")
            version = raw["policy_version"]
            if not _is_nonnegative_int(version) or version > self.policy_version or self.policy_version - version > 1:
                raise ValueError(f"trajectory {index} behavior policy version is outside the allowed lag")
            if version not in self._policy_snapshots:
                raise ValueError(f"trajectory {index} behavior policy adapter snapshot is not retained")
            reward = raw["reward"]
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                raise ValueError(f"trajectory {index} reward must be finite")
            if not isinstance(raw["group_id"], (str, int, tuple)):
                raise ValueError(f"trajectory {index} group_id has an unsupported type")
            try:
                hash(raw["group_id"])
            except TypeError as exc:
                raise ValueError(f"trajectory {index} group_id must be hashable") from exc
            versions.add(version)
            groups[raw["group_id"]].append({
                **raw,
                "input_ids": ids,
                "policy_mask": mask,
                "behavior_logprobs": behavior,
                "policy_version": version,
                "reward": float(reward),
                "choice_index": choice_index,
            })
        if len(versions) != 1:
            raise ValueError("all rows in one update must share the same behavior policy version")
        for group_id, rows in groups.items():
            if len(rows) != self.group_size:
                raise ValueError(f"group {group_id!r} is incomplete; expected {self.group_size} rows")
            prompt_ids = rows[0]["input_ids"][:-1]
            if any(row["input_ids"][:-1] != prompt_ids for row in rows):
                raise ValueError(f"group {group_id!r} contains mixed prompt token sequences")
            if len(prompt_ids) > self.max_prompt_tokens:
                raise ValueError(f"group {group_id!r} exceeds max_prompt_tokens")
        return groups, next(iter(versions))

    def _group_log_probs(self, group: list[dict[str, Any]], *, grad: bool) -> torch.Tensor:
        prompt_ids = group[0]["input_ids"][:-1]
        batched = torch.tensor([prompt_ids] * len(group), dtype=torch.long, device=self.device)
        if grad:
            return self._constrained_log_probs(batched)
        with torch.no_grad():
            return self._constrained_log_probs(batched)

    def update(
        self,
        trajectories: list[dict[str, Any]],
        *,
        before_optimizer_step: Callable[[], None] | None = None,
        recovery_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._recovery_dirty:
                raise HFCheckpointError("an optimizer step is newer than the durable checkpoint; resume latest")
            groups, behavior_version = self._validate_rows(trajectories)
            learner_start_version = self.policy_version
            if learner_start_version - behavior_version not in (0, 1):
                raise ValueError("behavior policy lag must be zero or one")

            prepared: list[
                tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ] = []
            all_advantages: list[float] = []
            for group in groups.values():
                rewards = torch.tensor([row["reward"] for row in group], dtype=torch.float32)
                if not bool(torch.isfinite(rewards).all()):
                    raise ValueError("group rewards must be finite")
                std = rewards.std(correction=1)
                if not bool(torch.isfinite(std)):
                    raise ValueError("group reward standard deviation is invalid")
                if float(std) <= 0.0:
                    raise ValueError(
                        "zero-variance groups must be filtered and refilled before optimizer entry"
                    )
                advantages = (rewards - rewards.mean()) / (std + 1e-4)
                all_advantages.extend(float(value) for value in advantages.tolist())
                with self._with_policy_version(behavior_version), torch.no_grad():
                    behavior_logps = self._group_log_probs(group, grad=False)
                with self._with_policy_version(learner_start_version), torch.no_grad():
                    proximal_logps = self._group_log_probs(group, grad=False)
                selected = torch.tensor([row["choice_index"] for row in group], dtype=torch.long)
                sampled_policy_logp = behavior_logps.gather(1, selected[:, None]).squeeze(1).detach()
                old_selected = proximal_logps.gather(1, selected[:, None]).squeeze(1).detach()
                stored_behavior = torch.tensor(
                    [row["behavior_logprobs"][-1] for row in group], dtype=torch.float32
                )
                expected_behavior = torch.log(
                    (1.0 - self.epsilon) * sampled_policy_logp.exp()
                    + self.epsilon / 2.0
                )
                if not torch.allclose(stored_behavior, expected_behavior, atol=2e-5, rtol=2e-5):
                    raise ValueError("stored behavior logprob does not match the actual epsilon-mixture policy")
                correction = torch.exp(old_selected - stored_behavior).clamp(max=self.importance_cap).detach()
                prepared.append((group, selected, old_selected, advantages, correction))

            if not any(abs(value) > 1e-12 for value in all_advantages):
                raise ValueError("all complete groups have zero reward variance; no optimizer or scheduler step was taken")
            policy_token_count = sum(
                sum(row["policy_mask"][1:]) for group in groups.values() for row in group
            )
            if policy_token_count != len(trajectories) or policy_token_count <= 0:
                raise ValueError("the actual policy-token denominator is invalid")

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum: torch.Tensor | None = None
            for group, selected, old_logp, advantage, correction in prepared:
                new_logps = self._group_log_probs(group, grad=True)
                current = new_logps.gather(1, selected[:, None]).squeeze(1)
                ratio = torch.exp(current - old_logp)
                if not bool(torch.isfinite(ratio).all()):
                    raise FloatingPointError("PPO ratio is not finite")
                clipped = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage
                row_loss = -torch.minimum(ratio * advantage, clipped) * correction
                group_sum = row_loss.sum()
                loss_sum = group_sum if loss_sum is None else loss_sum + group_sum
            assert loss_sum is not None
            loss = loss_sum / policy_token_count
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("small HF PPO loss is not finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(self._adapter_parameters, self.max_grad_norm)
            gradient_norm_value = float(gradient_norm.detach().cpu())
            if not math.isfinite(gradient_norm_value) or gradient_norm_value <= 0.0:
                self.optimizer.zero_grad(set_to_none=True)
                self.model.eval()
                raise RuntimeError("LoRA update produced no finite nonzero gradient; no optimizer step was taken")
            if before_optimizer_step is not None:
                before_optimizer_step()
            try:
                self.optimizer.step()
                self.scheduler.step()
            except BaseException:
                self._recovery_dirty = True
                self.model.eval()
                raise
            self.policy_version = learner_start_version + 1
            self._recovery_dirty = True
            self._policy_snapshots[self.policy_version] = self._adapter_state()
            try:
                path = self._save_checkpoint(
                    reason="optimizer_update",
                    update_context=recovery_context,
                )
                self._recovery_dirty = False
                publication = self.policy_record()
            except BaseException:
                self.model.eval()
                raise
            self.model.eval()
            self._prune_policy_snapshots()
            return {
                "updated": True,
                "policy_version": self.policy_version,
                "behavior_version": behavior_version,
                "learner_start_version": learner_start_version,
                "policy_lag": learner_start_version - behavior_version,
                "trajectory_count": len(trajectories),
                "group_count": len(groups),
                "policy_tokens": policy_token_count,
                "gradient_norm_before_clip": gradient_norm_value,
                "loss": float(loss.detach().cpu()),
                "importance_correction_cap": self.importance_cap,
                "mean_importance_correction": float(torch.cat([
                    item[4].detach().cpu() for item in prepared
                ]).mean()),
                "scheduler_step": int(self.scheduler.last_epoch),
                "checkpoint_path": str(path),
                "publication": publication,
            }

    def _identity_for_version(self, version: int) -> str:
        raw = f"{_METADATA_ID_SCHEMA}:{self.lineage_id}:{version}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def policy_record(self) -> dict[str, Any]:
        with self._lock:
            if self._recovery_dirty:
                raise HFCheckpointError("cannot publish a policy with an uncommitted optimizer update")
            identity = self._identity_for_version(self.policy_version)
            return {
                "version": self.policy_version,
                "directory": f"local://small-hf/{self.lineage_id}/version-{self.policy_version:06d}",
                "base_fingerprint": f"{self.model_id}@{self.model_revision}",
                "artifact_fingerprint": f"{self.model_id}@{self.model_revision}",
                "source_adapter_sha256": identity,
                "adapter_hash": identity,
                "quantization": {"mode": "none", "dtype": "float32"},
                "files": [],
                "evidence": {
                    "identity_kind": "metadata_id_not_content_sha256",
                    "policy_identity_kind": "metadata_id_not_content_sha256",
                    "identity_schema": _METADATA_ID_SCHEMA,
                    "lineage_id": self.lineage_id,
                    "version": self.policy_version,
                    "publication_scope": "local_only",
                    "adapter_payload_hash_computed": False,
                },
            }

    def publish(self) -> dict[str, Any]:
        return self.policy_record()

    def _prune_policy_snapshots(self) -> None:
        minimum_retained = max(0, self.policy_version - 1)
        for version in list(self._policy_snapshots):
            if version < minimum_retained and self._policy_refs.get(version, 0) == 0:
                del self._policy_snapshots[version]

    def acquire_policy(self, version: int) -> None:
        with self._lock:
            if not _is_nonnegative_int(version) or version not in self._policy_snapshots:
                raise ValueError(f"policy version {version!r} is not retained")
            self._policy_refs[version] = self._policy_refs.get(version, 0) + 1

    def release_policy(self, version: int) -> None:
        with self._lock:
            count = self._policy_refs.get(version, 0)
            if count < 1:
                raise RuntimeError(f"policy reference underflow for version {version}")
            if count == 1:
                self._policy_refs.pop(version)
            else:
                self._policy_refs[version] = count - 1
            self._prune_policy_snapshots()

    @property
    def retained_policy_versions(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._policy_snapshots))

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_store.latest_path

    @property
    def recovery_metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "mode": "resumed" if self._recovery_resumed else "fresh",
                "resumed": self._recovery_resumed,
                "lineage_id": self.lineage_id,
                "policy_version": self.policy_version,
                "durable_policy_version": self._durable_policy_version,
                "snapshot_id": self._snapshot_id,
                "snapshot_path": str(self.checkpoint_store.latest_path),
                "update_context": copy.deepcopy(self._last_update_context),
                "dirty": self._recovery_dirty,
            }

    def save_recovery_snapshot(
        self,
        *,
        update_context: dict[str, Any] | None = None,
        reason: str = "explicit",
    ) -> dict[str, Any]:
        with self._lock:
            if self._recovery_dirty:
                raise HFCheckpointError("cannot snapshot an update that has not committed durably")
            if update_context is None:
                update_context = self._last_update_context
            path = self._save_checkpoint(reason=reason, update_context=update_context)
            return {
                "path": str(path),
                "snapshot_id": self._snapshot_id,
                "policy_version": self.policy_version,
                "lineage_id": self.lineage_id,
            }

    def audit_state(self) -> dict[str, Any]:
        """Return exact state copies for split/resume equality checks."""
        with self._lock:
            return {
                "lineage_id": self.lineage_id,
                "policy_version": self.policy_version,
                "adapter_state": self._adapter_state(),
                "optimizer_state": self._copy_optimizer_state(self.optimizer.state_dict()),
                "scheduler_state": self._copy_optimizer_state(self.scheduler.state_dict()),
                "rng_state": self._copy_optimizer_state(self._rng_state()),
                "recovery_context": copy.deepcopy(self._last_update_context),
            }

    def shutdown(self) -> None:
        with self._lock:
            if self._recovery_dirty:
                raise HFCheckpointError("cannot shut down with an uncommitted optimizer update")
            self.model.eval()
