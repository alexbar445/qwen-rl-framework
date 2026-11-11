"""Strict configuration for the Native Learner and scoped Worker pool."""
from __future__ import annotations

from ..contracts.activation_offload import DEFAULT_OFFLOAD_MIN_TOKENS
from ..contracts.gradient_partitions import DEFAULT_GRADIENT_ACCUMULATION_ROWS

import ipaddress
import json
import math
import re
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints
from urllib.parse import urlsplit

import yaml

from ..contracts import worker as wire
from ..deployment.instance_relocation import normalize_path_mappings
from ..paths import ROOT
from ..worker.backend import (
    DEFAULT_LORA_RANK,
    DEFAULT_MAX_GENERATED_TOKENS,
    DEFAULT_WORKER_MAX_LORAS,
    DEFAULT_WORKER_MAX_MODEL_LEN,
    DEFAULT_WORKER_MAX_NUM_SEQS,
    NInferOptions,
    validate_backend_config,
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _matches_type(value: Any, expected: Any) -> bool:
    if expected is type(None):
        return value is None
    origin = get_origin(expected)
    if origin is not None:
        if origin is tuple:
            return isinstance(value, tuple)
        return isinstance(value, origin) if isinstance(origin, type) else True
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, type):
        return isinstance(value, expected)
    return True


def _strict_dataclass(cls, value: Any, name: str):
    data = _mapping(value, name)
    known = {item.name for item in fields(cls) if item.init}
    extra = set(data) - known
    if extra:
        raise ValueError(f"unknown {name} field(s): {sorted(extra)}")
    required = {
        item.name for item in fields(cls)
        if item.init and item.default is MISSING and item.default_factory is MISSING
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"missing {name} field(s): {sorted(missing)}")
    hints = get_type_hints(cls)
    for key, item in data.items():
        expected = hints.get(key)
        origin = get_origin(expected)
        args = get_args(expected)
        if expected is str:
            actual_ok = isinstance(item, str)
        elif expected is bool:
            actual_ok = isinstance(item, bool)
        elif expected is int:
            actual_ok = isinstance(item, int) and not isinstance(item, bool)
        elif expected is float:
            actual_ok = isinstance(item, (int, float)) and not isinstance(item, bool)
        elif origin in (Union, UnionType):
            actual_ok = any(_matches_type(item, part) for part in args)
        elif origin is tuple and args == (str, Ellipsis):
            actual_ok = isinstance(item, tuple) and all(isinstance(part, str) for part in item)
        else:
            actual_ok = True
        if not actual_ok:
            raise ValueError(f"{name}.{key} has the wrong type")
        if expected is float and not math.isfinite(float(item)):
            raise ValueError(f"{name}.{key} must be finite")
    return cls(**data)


@dataclass(frozen=True)
class LearnerConfig:
    host: str = "0.0.0.0"
    port: int = 8240
    gpus: tuple[int, ...] = (0,)
    parallelism: str = "single"
    update_backend: str = "trainer"
    gradient_executor_token_envs: dict[str, str] = field(default_factory=dict)
    gradient_executor_worker_ids: dict[str, str] = field(default_factory=dict)
    sampling_only_worker_ids: dict[str, str] = field(default_factory=dict)
    sampling_group_quotas: dict[str, int] = field(default_factory=dict)
    sampling_shared_worker_pairs: dict[str, str] = field(default_factory=dict)
    sampling_episode_slots: dict[str, int] = field(default_factory=dict)
    gradient_remote_executor_ids: tuple[str, ...] = ()
    gradient_remote_min_queued_trajectories: int = 1
    gradient_worker_sync_ids: tuple[str, ...] = ()
    gradient_local_executor_worker_ids: tuple[str, ...] = ()
    gradient_initial_training_dir: str | None = None
    gradient_wait_timeout: float = 3600.0
    gradient_tasks_per_worker: int = 1
    gradient_local_work_stealing: bool = False
    episode_duration_prior_seconds: float = 120.0
    gpu_role_switch_seconds: float = 35.0
    gpu_role_switch_margin_seconds: float = 10.0
    gradient_accumulation_rows: int = DEFAULT_GRADIENT_ACCUMULATION_ROWS
    gradient_memory_limit_bytes: int | None = None
    gradient_max_inflight_input_tokens: int | None = None
    gpu_sharing: str = "dedicated"
    shared_worker_ids: tuple[str, ...] = ()
    initial_adapter_path: str | None = None
    worker_suspend_weight_source: str = "host"
    worker_engine_lifecycle: str = "suspend"
    gpu_sharing_timeout: float = 3600.0
    residency_host_budget_gib: float = 12.0
    compute_python: str | None = None
    compute_pythonpath: str | None = None
    compute_timeout: float = 3600.0
    gradient_checkpointing: bool = True
    cuda_memory_fraction: float = 1.0
    lr: float = 1.0e-6
    request_timeout: float = 180.0
    max_sequence_length: int = 65536
    groups_per_update: int | str = 2
    groups_per_update_auto: bool = False
    microbatch_size: int = 2
    max_padded_tokens_per_microbatch: int | None = None
    importance_sampling_cap: float = 3.0
    activation_offload: bool = True
    activation_offload_min_tokens: int = DEFAULT_OFFLOAD_MIN_TOKENS
    activation_cache_parameter_names: tuple[str, ...] = ()
    activation_cache_cpu_bytes: int = 2 * 1024**3
    activation_cache_staging_bytes: int = 512 * 1024**2
    checkpoint_input_recompute_span: int = 1
    ternary_regional_compile: bool = True
    packed_weight_offload: bool = False
    packed_weight_offload_prefetch: bool = True
    packed_weight_offload_max_cpu_master_bytes: int = 4 * 1024**3
    packed_weight_offload_max_pinned_bytes: int = 512 * 1024**2
    recovery_dir: str | None = None
    recovery_mode: str = "auto"
    recovery_lineage_id: str | None = None
    recovery_path_mappings: dict[str, str] = field(default_factory=dict)
    tp_output_chunk_rows: int = 8192


@dataclass(frozen=True)
class WorkerConfig:
    id: str
    host: str | None = None
    port: int = 8241
    gpu: int = 0
    scope: str = "local"
    site_id: str = ""
    node_id: str = ""
    endpoint: str = ""
    url_prefix: str = ""
    auth_token_env: str = "ARC_ENV_TOKEN"
    boot_id: str = ""
    telemetry_epoch: str = ""
    model_sha256: str = ""

    @property
    def url(self) -> str:
        if self.scope in {"gateway", "service"}:
            return f"{self.endpoint.rstrip('/')}{self.url_prefix}"
        if self.host is None:
            raise ValueError(f"local Worker {self.id} has no host")
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class NodeScopeConfig:
    """Physical node identity for roles owned by this central runtime host."""

    site_id: str = "central"
    node_id: str = "central"


@dataclass(frozen=True)
class RolloutConfig:
    group_size: int = 3
    concurrent_tasks: int = 4
    ready_queue_batches: int = 2
    max_turns: int = 64
    prompt_length: int = 8192
    response_length: int = 57344
    turn_max_tokens: int = wire.MAX_GENERATED_TOKENS
    constrained_decoding: bool = True
    max_filtered_groups: int = 12
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    task_ids: tuple[str, ...] = ()
    category_sampling: bool = True
    helper_worker_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DapoConfig:
    updates: int = 1
    max_filtered_groups: int = 12


@dataclass(frozen=True)
class QuantizationConfig:
    mode: str = "torchao"
    bits: int = 4
    packed_resident: bool = True
    compute_dtype: str = "bfloat16"


@dataclass(frozen=True)
class WorkerInferenceConfig:
    model_path: str
    backend: str = "vllm"
    ninfer_url: str | None = None
    ninfer_generation_timeout_seconds: float | int | None = None
    ninfer_artifact_path: str | None = None
    ninfer_artifact_sha256: str | None = None
    ninfer_conversion_report: str | None = None
    ninfer_converter_python: str | None = None
    ninfer_converter_path: str | None = None
    quantization: str = "torchao"
    dtype: str = "bfloat16"
    max_model_len: int = DEFAULT_WORKER_MAX_MODEL_LEN
    # Effective per-request generation bound for this deployment. It is rendered into the
    # Worker's /config/worker.json, so raising it needs no Worker image rebuild; the wire
    # ceiling (wire.MAX_GENERATED_TOKENS) still caps it and the loader checks that
    # rollout.turn_max_tokens never exceeds it.
    max_generated_tokens: int = DEFAULT_MAX_GENERATED_TOKENS
    gpu_memory_utilization: float = 0.65
    kv_cache_dtype: str = "auto"
    kv_cache_memory_bytes: int | None = None
    max_num_seqs: int = DEFAULT_WORKER_MAX_NUM_SEQS
    max_loras: int = DEFAULT_WORKER_MAX_LORAS
    max_num_batched_tokens: int = 2048
    enable_prefix_caching: bool = True
    language_model_only: bool = True
    mamba_cache_mode: str = "align"
    enforce_eager: bool = True
    speculative_config: dict[str, Any] | None = None
    ninfer_options: NInferOptions | None = None


@dataclass(frozen=True)
class LoraConfig:
    rank: int = DEFAULT_LORA_RANK
    alpha: int = 16


@dataclass(frozen=True)
class ArcConfig:
    contract_path: str = "/work/configs/train-contract.json"
    group_timeout: float = 1200.0
    max_active_groups: int = 64
    max_generated_tokens: int = 4096
    overlong_buffer: int = 512


@dataclass(frozen=True)
class BepaBudgetConfig:
    max_total_tokens: int = 0
    max_job_tokens: int = 0
    job_timeout_seconds: float = 600.0
    unknown_usage_policy: str = "confirmed_lower_bound"


@dataclass(frozen=True)
class BepaTeacherConfig:
    base_url: str = ""
    model: str = "deepseek-v4.1-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    token_env: str = "BEPA_TEACHER_TOKEN"
    max_active_jobs: int = 1
    max_pending_jobs: int = 32
    max_completion_tokens: int = 32768
    budget: BepaBudgetConfig = field(default_factory=BepaBudgetConfig)
    max_stream_retries: int = 1


@dataclass(frozen=True)
class BepaBootstrapConfig:
    target_cache_tasks: int = 2
    max_tasks: int = 8


@dataclass(frozen=True)
class BepaSeedConfig:
    attempts_per_plan: int = 2
    max_active_episodes: int = 1


@dataclass(frozen=True)
class BepaCacheConfig:
    max_age_updates: int = 10
    max_committed_uses: int = 5


@dataclass(frozen=True)
class BepaScheduleConfig:
    ready_every_groups: int = 4


@dataclass(frozen=True)
class BepaConfig:
    enabled: bool = False
    state_dir: str = ""
    budget_scope: str = "default"
    teacher: BepaTeacherConfig = field(default_factory=BepaTeacherConfig)
    bootstrap: BepaBootstrapConfig = field(default_factory=BepaBootstrapConfig)
    seed: BepaSeedConfig = field(default_factory=BepaSeedConfig)
    cache: BepaCacheConfig = field(default_factory=BepaCacheConfig)
    schedule: BepaScheduleConfig = field(default_factory=BepaScheduleConfig)


def _bepa_config(value: Any) -> BepaConfig:
    data = dict(_mapping(value, "bepa"))
    teacher = dict(_mapping(data.get("teacher", {}), "bepa.teacher"))
    teacher["budget"] = _strict_dataclass(BepaBudgetConfig, teacher.get("budget", {}), "bepa.teacher.budget")
    data["teacher"] = _strict_dataclass(BepaTeacherConfig, teacher, "bepa.teacher")
    for name, cls in (("bootstrap", BepaBootstrapConfig), ("seed", BepaSeedConfig),
                      ("cache", BepaCacheConfig), ("schedule", BepaScheduleConfig)):
        data[name] = _strict_dataclass(cls, data.get(name, {}), f"bepa.{name}")
    return _strict_dataclass(BepaConfig, data, "bepa")


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = False
    pause_training: bool = False
    every_updates: int = 10
    manifest_path: str = ""
    sft_exclusion_proof: str = ""
    initial_count_proof: str = ""
    state_dir: str = ""
    concurrency: int = 1


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: str
    dataset_dir: str
    run_dir: str
    worker_inference: WorkerInferenceConfig
    model_backend: str = "default"
    artifact_identity_path: str | None = None
    artifact_identity: dict[str, Any] | None = None
    tokenizer_path: str | None = None
    recovery_semantics: dict[str, Any] | None = None
    node_scope: NodeScopeConfig = field(default_factory=NodeScopeConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    workers: tuple[WorkerConfig, ...] = ()
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    dapo: DapoConfig = field(default_factory=DapoConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    arc: ArcConfig = field(default_factory=ArcConfig)
    bepa: BepaConfig = field(default_factory=BepaConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def learner_url(self) -> str:
        host = "127.0.0.1" if self.learner.host in {"0.0.0.0", "::"} else self.learner.host
        return f"http://{host}:{self.learner.port}"

    @property
    def model(self) -> Path:
        return _resolve_path(self.model_path)

    @property
    def dataset(self) -> Path:
        return _resolve_path(self.dataset_dir)

    @property
    def run(self) -> Path:
        return _resolve_path(self.run_dir)

    @property
    def contract(self) -> Path:
        return _resolve_path(self.arc.contract_path)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        data = asdict(self)
        inference = data.get("worker_inference")
        options = inference.get("ninfer_options") if isinstance(inference, dict) else None
        if isinstance(options, dict) and options.get("lora_gdn_projections") is False:
            options.pop("lora_gdn_projections")
        return data


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _private_or_local_host(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        normalized = host.rstrip(".").lower()
        return (normalized == "localhost"
                or normalized.endswith((".local", ".lan", ".localhost", ".cluster.local", ".svc")))
    return (address.is_private or address.is_loopback) and not (
        address.is_link_local or address.is_unspecified or address.is_multicast
    )


def _tuple_of_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return tuple(value)


def _resolve_groups_per_update(learner_data: dict[str, Any]) -> bool:
    """Resolve central auto-B from explicit pairs and verify serialized metadata."""
    has_serialized_marker = "groups_per_update_auto" in learner_data
    serialized_marker = learner_data.pop("groups_per_update_auto", None)
    if has_serialized_marker and type(serialized_marker) is not bool:
        raise ValueError("learner.groups_per_update_auto must be a runtime boolean")
    value = learner_data.get("groups_per_update", 2)
    auto_requested = value == "auto"
    if has_serialized_marker and serialized_marker is False and auto_requested:
        raise ValueError("learner.groups_per_update='auto' conflicts with groups_per_update_auto=false")
    if not auto_requested and not (has_serialized_marker and serialized_marker is True):
        return False
    if learner_data.get("update_backend", "trainer") != "central_gradient":
        raise ValueError("learner.groups_per_update='auto' requires update_backend='central_gradient'")
    pairs = learner_data.get("gradient_executor_worker_ids")
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError("learner.groups_per_update='auto' requires explicit gradient executor/Worker pairs")
    tasks_per_worker = learner_data.get("gradient_tasks_per_worker", 1)
    if type(tasks_per_worker) is not int or not 1 <= tasks_per_worker <= 8:
        raise ValueError("gradient_tasks_per_worker must be an integer in [1, 8]")
    from qwen_rl.contracts.sampling_topology import owner_slots
    sampling_only = learner_data.get("sampling_only_worker_ids", {})
    quotas = learner_data.get("sampling_group_quotas", {})
    if (not isinstance(sampling_only, dict) or not isinstance(quotas, dict)
            or any(not isinstance(k, str) or re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}", k) is None
                   or not isinstance(v, str) or not v for k, v in sampling_only.items())
            or any(not isinstance(k, str) or not k or type(v) is not int or not 1 <= v <= 8
                   for k, v in quotas.items())):
        raise ValueError("invalid sampling owners or group quotas")
    expected_groups = len(owner_slots(learner_data))
    if auto_requested:
        learner_data["groups_per_update"] = expected_groups
    elif type(value) is not int or value != expected_groups:
        raise ValueError(
            "serialized groups_per_update_auto=true must match the explicit executor/Worker pair count"
        )
    return True


def load_config(path: str | Path, *, for_local_worker_plan: bool = False) -> RuntimeConfig:
    """Read a YAML file and validate the selected model and runtime topology."""
    path = Path(path).expanduser().resolve()
    contents = path.read_text()
    # The manager writes a per-run JSON document (which is valid YAML too); parse it as JSON
    # so exponent-form floats retain JSON numeric types under PyYAML's YAML 1.1 resolver.
    loaded = json.loads(contents) if contents.lstrip().startswith("{") else yaml.safe_load(contents)
    data = _mapping(loaded, "configuration")
    quantization_data = _mapping(data.get("quantization", {}), "quantization")
    if quantization_data.get("mode") == "compressed-tensors":
        raise ValueError(
            "quantization.mode='compressed-tensors' is retired with GPTQ; "
            "select native NVFP4 with mode='torchao' or the explicit T2 backend"
        )
    raw_inference = _mapping(data.get("worker_inference"), "worker_inference")
    if raw_inference.get("quantization") == "compressed-tensors":
        raise ValueError(
            "worker_inference.quantization='compressed-tensors' is retired with GPTQ; "
            "select native NVFP4 with quantization='torchao' or the explicit T2 backend"
        )
    if "marlin_input_dtype" in raw_inference:
        raise ValueError("worker_inference.marlin_input_dtype was removed with GPTQ support")
    allowed = {item.name for item in fields(RuntimeConfig) if item.init} | {"model"}
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"unknown configuration field(s): {sorted(extra)}")
    for name in ("model_path", "dataset_dir", "run_dir"):
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise ValueError(f"{name} must be a non-empty path string")

    model_data = _mapping(data.get("model", {}), "model")
    extra_model_fields = set(model_data) - {"backend", "artifact_identity_path"}
    if extra_model_fields:
        raise ValueError(f"unknown model field(s): {sorted(extra_model_fields)}")
    if model_data and ("model_backend" in data or "artifact_identity_path" in data):
        raise ValueError("model backend and artifact identity must use one config form")
    model_backend = model_data.get("backend", data.get("model_backend", "default"))
    artifact_identity_path = model_data.get(
        "artifact_identity_path", data.get("artifact_identity_path"))

    node_scope = _strict_dataclass(NodeScopeConfig, data.get("node_scope", {}), "node_scope")
    learner_data = dict(_mapping(data.get("learner", {}), "learner"))
    groups_per_update_auto = _resolve_groups_per_update(learner_data)
    learner_data["groups_per_update_auto"] = groups_per_update_auto
    if "gpus" in learner_data:
        if not isinstance(learner_data["gpus"], (list, tuple)):
            raise ValueError("learner.gpus must be a non-empty list of GPU IDs")
        learner_data["gpus"] = tuple(learner_data["gpus"])
    if "shared_worker_ids" in learner_data:
        learner_data["shared_worker_ids"] = _tuple_of_strings(learner_data["shared_worker_ids"], "learner.shared_worker_ids")
    if "activation_cache_parameter_names" in learner_data:
        learner_data["activation_cache_parameter_names"] = _tuple_of_strings(
            learner_data["activation_cache_parameter_names"], "learner.activation_cache_parameter_names")
    if "gradient_worker_sync_ids" in learner_data:
        learner_data["gradient_worker_sync_ids"] = _tuple_of_strings(
            learner_data["gradient_worker_sync_ids"], "learner.gradient_worker_sync_ids")
    if "gradient_local_executor_worker_ids" in learner_data:
        learner_data["gradient_local_executor_worker_ids"] = _tuple_of_strings(
            learner_data["gradient_local_executor_worker_ids"], "learner.gradient_local_executor_worker_ids")
    if "gradient_remote_executor_ids" in learner_data:
        learner_data["gradient_remote_executor_ids"] = _tuple_of_strings(
            learner_data["gradient_remote_executor_ids"], "learner.gradient_remote_executor_ids")
    try:
        learner_data["recovery_path_mappings"] = normalize_path_mappings(
            learner_data.get("recovery_path_mappings", {})
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"learner.recovery_path_mappings is invalid: {exc}") from None
    learner = _strict_dataclass(LearnerConfig, learner_data, "learner")
    rollout_data = dict(_mapping(data.get("rollout", {}), "rollout"))
    if "task_ids" in rollout_data:
        rollout_data["task_ids"] = _tuple_of_strings(rollout_data["task_ids"], "rollout.task_ids")
    if "helper_worker_ids" in rollout_data:
        rollout_data["helper_worker_ids"] = _tuple_of_strings(
            rollout_data["helper_worker_ids"], "rollout.helper_worker_ids")
    rollout = _strict_dataclass(RolloutConfig, rollout_data, "rollout")
    dapo = _strict_dataclass(DapoConfig, data.get("dapo", {}), "dapo")
    quantization = _strict_dataclass(QuantizationConfig, data.get("quantization", {}), "quantization")
    lora = _strict_dataclass(LoraConfig, data.get("lora", {}), "lora")
    arc = _strict_dataclass(ArcConfig, data.get("arc", {}), "arc")
    bepa = _bepa_config(data.get("bepa", {}))
    inference_data = dict(raw_inference)
    if inference_data.get("ninfer_options") is not None:
        inference_data["ninfer_options"] = _strict_dataclass(
            NInferOptions, inference_data["ninfer_options"], "worker_inference.ninfer_options")
    inference = _strict_dataclass(WorkerInferenceConfig, inference_data, "worker_inference")
    evaluation = _strict_dataclass(EvaluationConfig, data.get("evaluation", {}), "evaluation")

    raw_workers = data.get("workers")
    if not isinstance(raw_workers, list):
        raise ValueError("workers must be a list")
    workers_loaded = tuple(_strict_dataclass(WorkerConfig, value, f"workers[{index}]")
                           for index, value in enumerate(raw_workers))
    workers = tuple(
        replace(worker,
                site_id=worker.site_id or node_scope.site_id,
                node_id=worker.node_id or node_scope.node_id)
        if worker.scope in {"local", "service"} else worker
        for worker in workers_loaded
    )
    config = RuntimeConfig(model_path=data["model_path"], dataset_dir=data["dataset_dir"],
                           run_dir=data["run_dir"], worker_inference=inference,
                           model_backend=model_backend,
                           artifact_identity_path=artifact_identity_path,
                           tokenizer_path=data.get("tokenizer_path"),
                           recovery_semantics=data.get("recovery_semantics"),
                           node_scope=node_scope, learner=learner, workers=workers, rollout=rollout, dapo=dapo,
                           quantization=quantization, lora=lora, arc=arc, bepa=bepa,
                           evaluation=evaluation)
    if config.model_backend == "ternary":
        if not isinstance(config.artifact_identity_path, str) or not config.artifact_identity_path.strip():
            raise ValueError("model.artifact_identity_path is required for the ternary backend")
        try:
            from ..contracts.ternary import artifact_report, load_pinned_artifact_identity

            identity_path = _resolve_path(config.artifact_identity_path)
            identity = load_pinned_artifact_identity(identity_path)
        except Exception as exc:
            raise ValueError(f"T2 artifact identity verification failed ({type(exc).__name__})") from exc
        document = identity.as_dict()
        supplied_identity = data.get("artifact_identity")
        if supplied_identity is not None and supplied_identity != document:
            raise ValueError("T2 artifact identity sidecar changed after runtime profile validation")
        source = document["source"]
        artifact = source["ninfer_artifact"]
        report = artifact_report(document)
        if Path(artifact["path"]).resolve() != config.model.resolve():
            raise ValueError("model_path differs from the T2 artifact identity path")
        if _resolve_path(inference.model_path) != config.model.resolve():
            raise ValueError("Worker model_path differs from the T2 artifact identity path")
        if _resolve_path(inference.ninfer_artifact_path or "") != config.model.resolve():
            raise ValueError("worker_inference.ninfer_artifact_path differs from the T2 artifact identity path")
        if _resolve_path(inference.ninfer_conversion_report or "") != Path(report["path"]).resolve():
            raise ValueError("Worker conversion report differs from the T2 artifact identity")
        if inference.ninfer_artifact_sha256 not in (None, artifact["sha256"]):
            raise ValueError("Worker declared base SHA differs from deployment identity")
        config = replace(config, artifact_identity_path=str(identity_path), artifact_identity=document,
                         worker_inference=replace(inference, ninfer_artifact_sha256=artifact["sha256"]))
    elif config.model_backend == "default":
        if config.artifact_identity_path is not None:
            raise ValueError("model.artifact_identity_path requires model.backend=ternary")
        if config.tokenizer_path is not None or config.recovery_semantics is not None:
            raise ValueError("T2 manager fields require model.backend=ternary")
    else:
        raise ValueError("model.backend must be default or ternary")
    validate_config(config, for_local_worker_plan=for_local_worker_plan)
    return config


def validate_config(config: RuntimeConfig, *, for_local_worker_plan: bool = False) -> None:
    ternary = config.model_backend == "ternary"
    if ternary:
        if (config.quantization.mode != "ternary" or config.quantization.bits != 2
                or config.quantization.packed_resident is not True):
            raise ValueError("T2 requires explicit native ternary quantization with bits=2")
    elif (config.quantization.mode != "torchao"
          or config.quantization.bits != 4 or config.quantization.packed_resident is not True):
        raise ValueError("native Learner requires explicitly configured resident four-bit NVFP4")
    learner = config.learner
    central_gradient = learner.update_backend == "central_gradient"
    from ..task_catalog import load_source_catalog
    if load_source_catalog(config.dataset) is not None:
        if not central_gradient:
            raise ValueError("mixed task sources require central_gradient")
    if learner.update_backend not in {"trainer", "central_gradient"}:
        raise ValueError("learner.update_backend must be trainer or central_gradient")
    if any(type(gpu) is not int or gpu < 0 for gpu in learner.gpus) or len(set(learner.gpus)) != len(learner.gpus):
        raise ValueError("learner.gpus must contain distinct nonnegative GPU IDs")
    if central_gradient:
        if learner.gpus:
            raise ValueError("central_gradient requires learner.gpus=[]; executors select their own GPUs")
    elif not learner.gpus:
        raise ValueError("learner.gpus must contain at least one GPU for the Trainer backend")
    if learner.parallelism not in {"single", "tp", "ddp", "fsdp2"}:
        raise ValueError("learner.parallelism must be single, tp, ddp or fsdp2")
    if central_gradient:
        if learner.parallelism != "single":
            raise ValueError("central_gradient requires independent single-GPU executors; local DDP/TP are unsupported")
    elif learner.parallelism == "single" and len(learner.gpus) != 1:
        raise ValueError("single requires one Learner GPU")
    elif learner.parallelism in {"tp", "fsdp2"} and len(learner.gpus) < 2:
        raise ValueError("tp/fsdp2 require at least two Learner GPUs")
    if (isinstance(learner.gradient_wait_timeout, bool)
            or not isinstance(learner.gradient_wait_timeout, (int, float))
            or not math.isfinite(learner.gradient_wait_timeout)
            or learner.gradient_wait_timeout <= 0):
        raise ValueError("learner.gradient_wait_timeout must be finite and positive")
    token_envs = learner.gradient_executor_token_envs
    if not isinstance(token_envs, dict):
        raise ValueError("learner.gradient_executor_token_envs must be a mapping")
    for executor_id, env_name in token_envs.items():
        if (not isinstance(executor_id, str) or not executor_id or executor_id != executor_id.strip()
                or not isinstance(env_name, str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name) is None):
            raise ValueError("gradient executor IDs and token environment names must be non-empty valid strings")
    if len(set(token_envs.values())) != len(token_envs):
        raise ValueError("each gradient executor must use a distinct token environment variable")
    if central_gradient and not token_envs:
        raise ValueError("central_gradient requires at least one explicitly allowlisted executor")
    if not central_gradient and token_envs:
        raise ValueError("gradient executor tokens require learner.update_backend='central_gradient'")
    if type(learner.gradient_local_work_stealing) is not bool:
        raise ValueError("gradient_local_work_stealing must be bool")
    if learner.gradient_memory_limit_bytes is not None and (
            type(learner.gradient_memory_limit_bytes) is not int or learner.gradient_memory_limit_bytes <= 0):
        raise ValueError("gradient_memory_limit_bytes must be null or a positive integer")
    if type(learner.gradient_accumulation_rows) is not int or learner.gradient_accumulation_rows < 1:
        raise ValueError("gradient_accumulation_rows must be a positive integer")
    if learner.gradient_max_inflight_input_tokens is not None:
        if (type(learner.gradient_max_inflight_input_tokens) is not int
                or learner.gradient_max_inflight_input_tokens < learner.max_sequence_length):
            raise ValueError("gradient_max_inflight_input_tokens must cover max_sequence_length")
        if (not central_gradient or learner.microbatch_size != 1
                or not learner.gradient_local_work_stealing):
            raise ValueError("gradient_max_inflight_input_tokens requires central trajectory partitions")
    if learner.gradient_local_work_stealing and (not central_gradient or learner.microbatch_size != 1):
        raise ValueError("local work stealing requires central_gradient and microbatch_size=1")
    executor_worker_ids = learner.gradient_executor_worker_ids
    if not isinstance(executor_worker_ids, dict):
        raise ValueError("learner.gradient_executor_worker_ids must be a mapping")
    for executor_id, worker_id in executor_worker_ids.items():
        if (not isinstance(executor_id, str) or not executor_id or executor_id != executor_id.strip()
                or not isinstance(worker_id, str) or not worker_id or worker_id != worker_id.strip()):
            raise ValueError("gradient executor Worker mapping must contain non-empty exact IDs")
    if central_gradient:
        remote_ids = learner.gradient_remote_executor_ids
        if (type(learner.gradient_remote_min_queued_trajectories) is not int
                or learner.gradient_remote_min_queued_trajectories < 1):
            raise ValueError("gradient_remote_min_queued_trajectories must be a positive integer")
        if (len(set(remote_ids)) != len(remote_ids) or set(remote_ids) & set(executor_worker_ids)
                or set(executor_worker_ids) | set(remote_ids) != set(token_envs)):
            raise ValueError("executor allowlist must contain exactly the paired and dedicated remote executors")
        if remote_ids and (learner.microbatch_size != 1 or learner.gradient_accumulation_rows != 1):
            raise ValueError("remote executors require microbatch_size=1 and one trajectory per gradient")
        sampling_only = learner.sampling_only_worker_ids
        quotas = learner.sampling_group_quotas
        if (not isinstance(sampling_only, dict) or not isinstance(quotas, dict)
                or any(not isinstance(k, str) or re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}", k) is None
                       or not isinstance(v, str) or not v for k, v in sampling_only.items())
                or set(sampling_only) & (set(executor_worker_ids) | set(remote_ids))
                or len(set((*executor_worker_ids.values(), *sampling_only.values()))) != len(executor_worker_ids) + len(sampling_only)
                or not set(sampling_only.values()) <= set(config.rollout.helper_worker_ids)
                or not set(quotas) <= set((*executor_worker_ids.values(), *sampling_only.values()))
                or any(type(v) is not int or not 1 <= v <= 8 for v in quotas.values())):
            raise ValueError("invalid sampling-only Workers or group quotas")
        from qwen_rl.contracts.sampling_topology import owner_slots
        shared = learner.sampling_shared_worker_pairs
        slots = learner.sampling_episode_slots
        # owner_slots validates pair membership and overlap before scheduler construction.
        owner_slots(vars(learner))
        if (not isinstance(slots, dict) or set(slots) - set(sampling_only.values())
                or any(type(value) is not int or not 1 <= value <= config.worker_inference.max_num_seqs
                       for value in slots.values())):
            raise ValueError("invalid sampling-only episode slot limits")
        worker_config = {worker.id: worker for worker in config.workers}
        if (set(shared) | set(shared.values())) - set(worker_config):
            raise ValueError("shared sampling Workers must be configured")
        for first, second in shared.items():
            a, b = worker_config[first], worker_config[second]
            if (a.scope != "gateway" or b.scope != "gateway" or not a.boot_id
                    or (a.endpoint, a.site_id, a.boot_id) != (b.endpoint, b.site_id, b.boot_id)
                    or first not in slots or second not in slots
                    or slots[first] + slots[second] < config.rollout.group_size):
                raise ValueError("shared sampling Workers require the same Gateway host and sufficient slots")
        if sampling_only and (not remote_ids or not learner.gradient_local_work_stealing):
            raise ValueError("sampling-only Workers require remote gradient executors and local work stealing")
        if learner.groups_per_update_auto and learner.groups_per_update != len(owner_slots(vars(learner))):
            raise ValueError("auto groups_per_update differs from the sampling slots")
        if len(set(executor_worker_ids.values())) != len(executor_worker_ids):
            raise ValueError("one configured Worker may belong to only one gradient executor")
        worker_by_id = {worker.id: worker for worker in config.workers}
        unknown = set(executor_worker_ids.values()) - set(worker_by_id)
        if unknown:
            raise ValueError("gradient_executor_worker_ids contains unconfigured Workers: " + ", ".join(sorted(unknown)))
        physical_gpu_owners = set()
        for worker_id in executor_worker_ids.values():
            worker = worker_by_id[worker_id]
            if worker.scope not in {"local", "service", "gateway"}:
                raise ValueError(
                    "gradient executors must map only to configured local/service/gateway Workers"
                )
            node_identity = worker.node_id or worker.endpoint or worker.host or config.node_scope.node_id
            physical_gpu = (worker.site_id or config.node_scope.site_id,
                            node_identity, worker.gpu)
            if physical_gpu in physical_gpu_owners:
                raise ValueError("one configured Worker GPU may belong to only one gradient executor")
            physical_gpu_owners.add(physical_gpu)
    elif (executor_worker_ids or learner.gradient_remote_executor_ids
          or learner.sampling_only_worker_ids or learner.sampling_group_quotas
          or learner.sampling_shared_worker_pairs or learner.sampling_episode_slots):
        raise ValueError("sampling topology requires learner.update_backend='central_gradient'")
    local_executor_ids = learner.gradient_local_executor_worker_ids
    if local_executor_ids:
        if not central_gradient:
            raise ValueError("gradient_local_executor_worker_ids requires central_gradient")
        if len(set(local_executor_ids)) != len(local_executor_ids):
            raise ValueError("gradient_local_executor_worker_ids must contain unique Worker IDs")
        worker_by_id = {worker.id: worker for worker in config.workers}
        if not set(local_executor_ids) <= set(executor_worker_ids.values()):
            raise ValueError("local executors must belong to explicit gradient executor/Worker pairs")
        if any(worker_by_id[name].scope not in {"local", "service"} for name in local_executor_ids):
            raise ValueError("local executors cannot be assigned to remote Gateway Workers")
    if learner.gradient_local_work_stealing and len(local_executor_ids) < 2:
        raise ValueError("local work stealing requires at least two explicitly local paired Workers")
    for name in ('episode_duration_prior_seconds', 'gpu_role_switch_seconds',
                 'gpu_role_switch_margin_seconds'):
        value = getattr(learner, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if learner.gradient_initial_training_dir is not None:
        source_state = learner.gradient_initial_training_dir
        if not central_gradient or not isinstance(source_state, str) or not Path(source_state).is_absolute():
            raise ValueError("gradient_initial_training_dir requires central_gradient and an absolute source directory")
        if learner.recovery_dir and Path(source_state).resolve() == Path(learner.recovery_dir).resolve():
            raise ValueError("initial DDP training state and central checkpoint directory must be separate")
    if type(learner.groups_per_update_auto) is not bool:
        raise ValueError("learner.groups_per_update_auto must be a runtime boolean")
    if learner.groups_per_update_auto and not central_gradient:
        raise ValueError("auto groups_per_update requires learner.update_backend='central_gradient'")
    helper_worker_ids = config.rollout.helper_worker_ids
    if helper_worker_ids:
        if len(set(helper_worker_ids)) != len(helper_worker_ids):
            raise ValueError("rollout.helper_worker_ids must contain unique Worker IDs")
        if not central_gradient or learner.microbatch_size != 1:
            raise ValueError("rollout.helper_worker_ids requires central_gradient and microbatch_size=1")
        if config.evaluation.enabled and config.evaluation.pause_training:
            raise ValueError("rollout.helper_worker_ids requires evaluation.pause_training=false")
        worker_by_id = {worker.id: worker for worker in config.workers}
        unknown_helpers = set(helper_worker_ids) - set(worker_by_id)
        if unknown_helpers:
            raise ValueError("rollout.helper_worker_ids contains unconfigured Workers: "
                             + ", ".join(sorted(unknown_helpers)))
        if any(worker_by_id[worker_id].scope != "gateway" for worker_id in helper_worker_ids):
            raise ValueError("rollout.helper_worker_ids may select only configured Gateway Workers")
        owner_worker_ids = set(learner.gradient_executor_worker_ids.values())
        if set(helper_worker_ids) & owner_worker_ids:
            raise ValueError("rollout.helper_worker_ids cannot overlap gradient executor Worker owners")
        if set(helper_worker_ids) & set(learner.gradient_worker_sync_ids):
            raise ValueError("rollout.helper_worker_ids cannot be included in gradient_worker_sync_ids")
    if learner.gradient_worker_sync_ids:
        if not central_gradient:
            raise ValueError("gradient_worker_sync_ids requires learner.update_backend='central_gradient'")
        worker_by_id = {worker.id: worker for worker in config.workers}
        unknown = set(learner.gradient_worker_sync_ids) - set(worker_by_id)
        if unknown:
            raise ValueError("gradient_worker_sync_ids contains unconfigured Workers: " + ", ".join(sorted(unknown)))
        if any(worker_by_id[worker_id].scope not in {"local", "service", "gateway"}
               for worker_id in learner.gradient_worker_sync_ids):
            raise ValueError("gradient_worker_sync_ids may select only configured local/service/gateway Workers")
        for worker_id in learner.gradient_worker_sync_ids:
            worker = worker_by_id[worker_id]
            if worker.scope == "gateway" and not all((worker.boot_id, worker.telemetry_epoch, worker.model_sha256)):
                raise ValueError(
                    f"gradient gateway Worker {worker_id} requires pinned boot_id, telemetry_epoch and model_sha256"
                )
    dynamic_microbatch_cap = learner.max_padded_tokens_per_microbatch
    if dynamic_microbatch_cap is not None:
        if type(dynamic_microbatch_cap) is not int or dynamic_microbatch_cap < 1:
            raise ValueError("learner.max_padded_tokens_per_microbatch must be a positive integer")
        if learner.parallelism != "single":
            raise ValueError("learner.max_padded_tokens_per_microbatch requires learner.parallelism='single'")
    if type(learner.checkpoint_input_recompute_span) is not int or learner.checkpoint_input_recompute_span != 1:
        raise ValueError("learner.checkpoint_input_recompute_span must be1")
    selectors = learner.activation_cache_parameter_names
    if not isinstance(selectors, (list, tuple)) or any(not isinstance(s, str) or not s for s in selectors):
        raise ValueError("activation_cache_parameter_names must be a list of nonempty strings")
    if selectors and not (central_gradient and config.model_backend == "ternary"):
        raise ValueError("parameter activation cache requires central T2 gradients")
    for budget in (learner.activation_cache_cpu_bytes, learner.activation_cache_staging_bytes):
        if type(budget) is not int or budget <= 0:
            raise ValueError("activation cache byte budgets must be positive integers")
    if type(learner.activation_offload_min_tokens) is not int or learner.activation_offload_min_tokens < 0:
        raise ValueError("learner.activation_offload_min_tokens must be an integer >= 0")
    if type(learner.packed_weight_offload) is not bool or type(learner.packed_weight_offload_prefetch) is not bool:
        raise ValueError("learner packed weight offload flags must be booleans")
    if isinstance(learner.tp_output_chunk_rows, bool) or not isinstance(learner.tp_output_chunk_rows, int) or learner.tp_output_chunk_rows < 1:
        raise ValueError("learner.tp_output_chunk_rows must be a positive integer")
    for name in ("packed_weight_offload_max_cpu_master_bytes", "packed_weight_offload_max_pinned_bytes"):
        value = getattr(learner, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"learner.{name} must be a positive integer")
    if not 0 < learner.cuda_memory_fraction <= 1:
        raise ValueError("learner.cuda_memory_fraction must be in (0, 1]")
    if isinstance(learner.compute_timeout, bool) or not math.isfinite(learner.compute_timeout) or learner.compute_timeout <= 0:
        raise ValueError("learner.compute_timeout must be finite and positive")
    if config.quantization.mode == "torchao":
        if not learner.compute_python or not learner.compute_pythonpath:
            raise ValueError("native Learner requires explicit compute_python and compute_pythonpath")
        if config.worker_inference.speculative_config is not None:
            raise ValueError("native NVFP4 Worker requires speculative_config=null until separately validated")
    evaluation = config.evaluation
    if type(evaluation.every_updates) is not int or evaluation.every_updates < 1:
        raise ValueError("evaluation.every_updates must be a positive integer")
    if type(evaluation.concurrency) is not int or not 1 <= evaluation.concurrency <= 64:
        raise ValueError("evaluation.concurrency must be an integer between 1 and 64")
    if type(evaluation.pause_training) is not bool:
        raise ValueError("evaluation.pause_training must be a boolean")
    if evaluation.pause_training and (
            not evaluation.enabled
            or config.learner.update_backend != "central_gradient"
            or config.worker_inference.backend != "ninfer"):
        raise ValueError("evaluation.pause_training requires enabled central_gradient evaluation with NInfer Workers")
    if evaluation.enabled:
        if any(not isinstance(value, str) or not value.strip() for value in
               (evaluation.manifest_path, evaluation.sft_exclusion_proof, evaluation.state_dir)):
            raise ValueError("evaluation requires its frozen manifest, exclusion proof and durable state directory")
        # Evaluation on drained helpers needs the active head and one frozen
        # policy, just like a paused native evaluation.
        isolated_native = (
            config.worker_inference.backend == "ninfer"
            and (not config.bepa.enabled or evaluation.pause_training or bool(helper_worker_ids))
        )
        required_loras = 2 if isolated_native else 4
        if config.worker_inference.max_loras < required_loras:
            raise ValueError(f"evaluation requires adapter capacity of at least {required_loras}")
    bepa = config.bepa
    learner = config.learner
    if learner.recovery_mode not in {"auto", "fresh"}:
        raise ValueError("learner.recovery_mode must be auto or fresh")
    if learner.recovery_dir is not None and (
            not isinstance(learner.recovery_dir, str) or not learner.recovery_dir.strip()):
        raise ValueError("learner.recovery_dir must be a non-empty path when set")
    if learner.recovery_lineage_id is not None and (
            not isinstance(learner.recovery_lineage_id, str) or not learner.recovery_lineage_id.strip()):
        raise ValueError("learner.recovery_lineage_id must be a non-empty string when set")
    if learner.recovery_mode == "fresh" and learner.recovery_lineage_id is None:
        raise ValueError("learner.recovery_lineage_id is required in fresh mode")
    if learner.recovery_mode == "auto" and learner.recovery_lineage_id is not None:
        raise ValueError("learner.recovery_lineage_id is only valid in fresh mode")
    for name, value in (("max_pending_jobs", bepa.teacher.max_pending_jobs),
                        ("max_completion_tokens", bepa.teacher.max_completion_tokens),
                        ("max_tasks", bepa.bootstrap.max_tasks),
                        ("target_cache_tasks", bepa.bootstrap.target_cache_tasks),
                        ("attempts_per_plan", bepa.seed.attempts_per_plan),
                        ("max_age_updates", bepa.cache.max_age_updates),
                        ("max_committed_uses", bepa.cache.max_committed_uses),
                        ("ready_every_groups", bepa.schedule.ready_every_groups)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"bepa.{name} must be a positive integer")
    if bepa.teacher.max_active_jobs != 1 or bepa.seed.max_active_episodes != 1:
        raise ValueError("BEPA v1 requires one teacher job and one guided episode at a time")
    if bepa.bootstrap.target_cache_tasks > bepa.bootstrap.max_tasks:
        raise ValueError("BEPA bootstrap target exceeds its candidate task budget")
    if bepa.enabled:
        endpoint = urlsplit(bepa.teacher.base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or endpoint.username or endpoint.password:
            raise ValueError("bepa.teacher.base_url must be an HTTP URL without embedded credentials")
        if not bepa.teacher.model or not bepa.teacher.api_key_env or not bepa.teacher.token_env:
            raise ValueError("BEPA teacher model and credential environment names are required")
        names = (bepa.teacher.api_key_env, bepa.teacher.token_env, "ARC_ENV_TOKEN")
        if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None for name in names):
            raise ValueError("BEPA credential names must be valid environment variable names")
        if len(set(names)) != len(names):
            raise ValueError("BEPA provider, teacher and ARC credential names must be distinct")
        if type(bepa.teacher.max_stream_retries) is not int or not 0 <= bepa.teacher.max_stream_retries <= 2:
            raise ValueError("BEPA max_stream_retries must be an integer from 0 to 2")
        budget = bepa.teacher.budget
        if budget.unknown_usage_policy not in {"block", "confirmed_lower_bound"}:
            raise ValueError("BEPA unknown_usage_policy must be block or confirmed_lower_bound")
        if (budget.max_job_tokens < 1 or budget.max_total_tokens < budget.max_job_tokens
                or not math.isfinite(budget.job_timeout_seconds) or budget.job_timeout_seconds <= 0):
            raise ValueError("BEPA requires explicit positive teacher job/total token budgets and deadline")
    if (isinstance(config.learner.importance_sampling_cap, bool)
            or not isinstance(config.learner.importance_sampling_cap, (int, float))
            or not math.isfinite(config.learner.importance_sampling_cap)
            or config.learner.importance_sampling_cap < 1.0):
        raise ValueError("learner.importance_sampling_cap must be finite and >= 1")
    if learner.worker_engine_lifecycle not in {"suspend", "restart"}:
        raise ValueError("learner.worker_engine_lifecycle must be suspend or restart")
    if learner.worker_engine_lifecycle == "restart" and config.worker_inference.backend != "ninfer":
        raise ValueError("restart Worker engine lifecycle requires NInfer")
    if learner.worker_suspend_weight_source not in {"host", "artifact"}:
        raise ValueError("learner.worker_suspend_weight_source must be host or artifact")
    if learner.initial_adapter_path is not None:
        allowed_initial_adapter = (
            (ternary and learner.parallelism in {"ddp", "tp"})
            or (central_gradient and ternary and learner.parallelism == "single")
        )
        if (not allowed_initial_adapter
                or not isinstance(learner.initial_adapter_path, str)
                or not Path(learner.initial_adapter_path).is_absolute()):
            raise ValueError("learner.initial_adapter_path requires an absolute T2 DDP/TP or central LoRA adapter directory")
    if central_gradient:
        if not ternary:
            raise ValueError("central_gradient currently supports only the verified T2 model backend")
        if learner.initial_adapter_path is None:
            raise ValueError("central_gradient requires an explicit provenance-qualified initial_adapter_path")
        if learner.gpu_sharing != "dedicated":
            raise ValueError("central_gradient does not use the existing local GPU sharing lifecycle")
        if learner.recovery_mode != "auto" or learner.recovery_lineage_id is not None:
            raise ValueError("central_gradient requires automatic resume of its dedicated checkpoint schema")
        if not isinstance(learner.recovery_dir, str) or not Path(learner.recovery_dir).is_absolute():
            raise ValueError("central_gradient requires an explicit absolute learner.recovery_dir")
        central_state = Path(learner.recovery_dir).resolve()
        formal_state = (Path(config.run_dir).expanduser().resolve() / "training").resolve()
        if central_state == formal_state:
            raise ValueError("central_gradient checkpoint directory must be separate from the formal training state")
    if learner.gpu_sharing not in {"dedicated", "phase_shared"}:
        raise ValueError("learner.gpu_sharing must be dedicated or phase_shared")
    if learner.gpu_sharing == "phase_shared" and not ternary:
        raise ValueError("phase_shared currently requires the T2 backend")
    if learner.gpu_sharing == "dedicated" and learner.shared_worker_ids:
        raise ValueError("learner.shared_worker_ids requires phase_shared")
    for name in ("gpu_sharing_timeout", "residency_host_budget_gib"):
        value = getattr(learner, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"learner.{name} must be finite and positive")
    inference = config.worker_inference
    if learner.packed_weight_offload and not (
        ternary and (learner.parallelism in {"ddp", "tp"} or central_gradient)
    ):
        raise ValueError("packed_weight_offload requires managed T2 DDP, TP or central gradients")
    if learner.packed_weight_offload and not learner.gradient_checkpointing:
        raise ValueError("packed_weight_offload requires gradient_checkpointing for bounded leases")
    if not ternary and (inference.quantization != "torchao"
                        or inference.quantization != config.quantization.mode):
        raise ValueError("worker inference requires a verified four-bit quantization backend")
    try:
        validate_backend_config(asdict(inference))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not inference.model_path.strip():
        raise ValueError("worker_inference.model_path is required")
    if ternary:
        if learner.parallelism == "tp" and len(learner.gpus) != 4:
            raise ValueError("T2 TP currently requires exactly four Learner GPUs")
        if inference.backend != "ninfer" or inference.quantization != "ternary":
            raise ValueError("T2 Worker requires worker_inference.backend=ninfer and quantization=ternary")
        if learner.parallelism not in {"single", "ddp", "tp"}:
            raise ValueError("the asynchronous T2 Learner requires single, ddp or tp parallelism")
        local_workers = [worker for worker in config.workers if worker.scope in {"local", "service"}]
        local_gpus = [worker.gpu for worker in local_workers]
        if len(local_gpus) != len(set(local_gpus)):
            raise ValueError("T2 local Workers require distinct GPU assignments")
        if learner.gpu_sharing == "phase_shared":
            from ..contracts.gpu_sharing import shared_workers
            selected = shared_workers(config.as_dict())
            if learner.parallelism not in {"ddp", "tp"}:
                raise ValueError("T2 phase_shared requires DDP or TP")
            if any(worker["scope"] != "service" and not (
                    for_local_worker_plan and worker["scope"] == "local") for worker in selected):
                raise ValueError("phase_shared requires managed local service Workers for the shared selection")
        elif set(local_gpus) & set(learner.gpus):
            raise ValueError("dedicated T2 Learner and Worker GPUs must be distinct")
        if not config.artifact_identity or not config.artifact_identity_path:
            raise ValueError("T2 requires a verified artifact identity before runtime startup")
        adapter = config.artifact_identity.get("learner", {}).get("adapter", {})
        if (config.lora.rank, config.lora.alpha) != (adapter.get("rank"), adapter.get("alpha")):
            raise ValueError("T2 LoRA rank/alpha differs from the pinned artifact identity")
        if config.tokenizer_path is not None and (
                not isinstance(config.tokenizer_path, str) or not Path(config.tokenizer_path).is_absolute()):
            raise ValueError("T2 tokenizer_path must be an absolute manager-selected path")
        if config.recovery_semantics is not None:
            from ..learner.recovery import async_t2_recovery_semantics

            expected_semantics = async_t2_recovery_semantics(config.as_dict())
            if config.recovery_semantics != expected_semantics:
                raise ValueError("T2 recovery_semantics do not match the canonical async runtime identity")
    elif inference.quantization != config.quantization.mode:
        raise ValueError("worker inference requires a verified four-bit quantization backend")
    if inference.dtype != "bfloat16" or not inference.enable_prefix_caching or inference.mamba_cache_mode != "align":
        raise ValueError("Qwen Worker requires BF16 compute and prefix caching with mamba_cache_mode=align")
    if not 0 < inference.gpu_memory_utilization < 1 or min(inference.max_model_len, inference.max_num_seqs, inference.max_num_batched_tokens) < 1 or inference.max_loras < 2:
        raise ValueError("worker inference memory, sequence and LoRA limits must be positive")
    if inference.kv_cache_dtype not in ("auto", "fp8", "fp8_e5m2", "fp8_e4m3"):
        raise ValueError("worker_inference.kv_cache_dtype must be auto, fp8, fp8_e5m2 or fp8_e4m3")
    if inference.speculative_config is not None:
        speculative = _mapping(inference.speculative_config, "worker_inference.speculative_config")
        if speculative.get("method") not in {"ngram", "mtp"}:
            raise ValueError("speculative method must be ngram or mtp")
        tokens = speculative.get("num_speculative_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError("num_speculative_tokens must be a positive integer")
    cache_bytes = config.worker_inference.kv_cache_memory_bytes
    if cache_bytes is not None and (isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes <= 0):
        raise ValueError("kv_cache_memory_bytes must be a positive integer")
    if not ternary and (config.quantization.mode != "torchao"
                        or config.quantization.bits != 4 or config.quantization.packed_resident is not True):
        raise ValueError("native Learner requires explicitly configured resident four-bit NVFP4")
    if config.quantization.compute_dtype != "bfloat16":
        raise ValueError("Learner and Worker compute_dtype must be bfloat16")
    if Path(config.model_path).expanduser().resolve() != Path(config.worker_inference.model_path).expanduser().resolve():
        raise ValueError("Learner and Worker must use the same selected model artifact")
    sequence_ceiling = wire.MAX_CONTEXT_TOKENS if config.learner.update_backend == "central_gradient" else 65536
    if (isinstance(config.learner.max_sequence_length, bool)
            or not isinstance(config.learner.max_sequence_length, int)
            or not 2 <= config.learner.max_sequence_length <= sequence_ceiling):
        raise ValueError(f"learner.max_sequence_length must be an integer between 2 and {sequence_ceiling}")
    if config.rollout.prompt_length + config.rollout.response_length > config.learner.max_sequence_length:
        raise ValueError("rollout prompt plus response exceeds the Learner sequence limit")
    if config.lora.rank < 1 or config.lora.alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive integers")
    if not config.workers:
        raise ValueError("at least one Worker is required")
    for name, value, lower in (
        ("rollout.group_size", config.rollout.group_size, 2),
        ("rollout.concurrent_tasks", config.rollout.concurrent_tasks, 1),
        ("rollout.ready_queue_batches", config.rollout.ready_queue_batches, 1),
        ("learner.groups_per_update", config.learner.groups_per_update, 1),
        ("learner.microbatch_size", config.learner.microbatch_size, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if type(config.rollout.category_sampling) is not bool:
        raise ValueError("rollout.category_sampling must be boolean")
    samples = config.rollout.group_size * config.learner.groups_per_update
    if samples % config.learner.microbatch_size:
        raise ValueError("groups_per_update * group_size must be divisible by microbatch_size")
    if config.rollout.concurrent_tasks > config.arc.max_active_groups:
        raise ValueError("concurrent_tasks exceeds ARC active-group capacity")
    ids = [worker.id for worker in config.workers]
    if len(set(ids)) != len(config.workers) or any(not item for item in ids):
        raise ValueError("worker ids must be unique and non-empty")
    if not config.learner.host:
        raise ValueError("learner host must be non-empty")
    if (isinstance(config.learner.port, bool) or not isinstance(config.learner.port, int)
            or not 1 <= config.learner.port <= 65535):
        raise ValueError("learner.port must be a valid TCP port")

    # Device and listener numbers are local to a physical node. Native Learner
    # ranks occupy all configured Learner GPUs on the manager node; Gateway
    # Workers on other nodes may reuse their node-local GPU and port indices.
    manager_node = (config.node_scope.site_id, config.node_scope.node_id)
    if any(not isinstance(part, str) or not part.strip() for part in manager_node):
        raise ValueError("node_scope requires non-empty site_id and node_id")
    if (re.fullmatch(r"[A-Za-z0-9_-]{1,80}", manager_node[0]) is None
            or re.fullmatch(r"[A-Za-z0-9_-]{1,100}", manager_node[1]) is None):
        raise ValueError("node_scope site_id and node_id may contain only letters, digits, '_' and '-'")
    manager_node_workers = {(worker.site_id, worker.node_id) for worker in config.workers
                            if worker.scope in {"local", "service"}}
    if len(manager_node_workers) > 1:
        raise ValueError("local and service Workers must share the local Learner node scope")
    learner_node = manager_node
    if manager_node_workers and manager_node not in manager_node_workers:
        raise ValueError("local and service Worker scopes must match runtime node_scope")
    if any(not isinstance(part, str) or not part.strip() for part in learner_node):
        raise ValueError("local Worker site_id and node_id must be non-empty")
    used_ports = {(learner_node[0], learner_node[1], config.learner.port): "Learner"}
    used_gpus = {
        (learner_node[0], learner_node[1], gpu): f"Learner rank {rank}"
        for rank, gpu in enumerate(config.learner.gpus)
        if learner.gpu_sharing != "phase_shared"
    }
    gateway_auth: dict[tuple[str, str], str] = {}
    service_auth_names: set[str] = set()
    for worker in config.workers:
        if not isinstance(worker.id, str) or not worker.id.strip():
            raise ValueError("worker ids must be unique and non-empty")
        if worker.scope not in {"local", "service", "gateway"}:
            raise ValueError(f"Worker {worker.id} scope must be local, service or gateway")
        if any(not isinstance(part, str) or not part.strip() for part in (worker.site_id, worker.node_id)):
            raise ValueError(f"Worker {worker.id} requires an explicit non-empty site_id and node_id")
        if isinstance(worker.port, bool) or not isinstance(worker.port, int) or not 1 <= worker.port <= 65535:
            raise ValueError(f"Worker {worker.id} port must be a valid node-local TCP port")
        if isinstance(worker.gpu, bool) or not isinstance(worker.gpu, int) or worker.gpu < 0:
            raise ValueError(f"Worker {worker.id} GPU must be a nonnegative node-local index")
        port_key = (worker.site_id, worker.node_id, worker.port)
        gpu_key = (worker.site_id, worker.node_id, worker.gpu)
        if port_key in used_ports:
            raise ValueError(
                f"duplicate TCP port {worker.port}; listener ports must be unique TCP ports on node "
                f"{worker.site_id}/{worker.node_id} "
                f"for {used_ports[port_key]} and Worker {worker.id}"
            )
        if gpu_key in used_gpus:
            raise ValueError(
                f"duplicate GPU {worker.gpu}; roles on one node must use a different GPU on node "
                f"{worker.site_id}/{worker.node_id} "
                f"for {used_gpus[gpu_key]} and Worker {worker.id}"
            )
        used_ports[port_key] = worker.id
        used_gpus[gpu_key] = worker.id
        if worker.scope == "local":
            if worker.host is None or not worker.host.strip():
                raise ValueError(f"local Worker {worker.id} requires host")
            if worker.endpoint or worker.url_prefix or worker.auth_token_env != "ARC_ENV_TOKEN":
                raise ValueError(f"local Worker {worker.id} cannot configure Gateway routing or credentials")
            if (worker.site_id, worker.node_id) != learner_node:
                raise ValueError(f"local Worker {worker.id} must use the local Learner node scope")
            if worker.boot_id or worker.telemetry_epoch:
                raise ValueError(f"local Worker {worker.id} cannot carry a remote membership epoch")
            if worker.model_sha256:
                raise ValueError(f"local Worker {worker.id} cannot carry a Gateway model SHA")
            continue

        if worker.scope == "service":
            if worker.host is not None:
                raise ValueError(f"service Worker {worker.id} must use its direct endpoint, not host")
            if worker.url_prefix:
                raise ValueError(f"service Worker {worker.id} cannot use a Gateway route prefix")
            if (worker.site_id, worker.node_id) != learner_node:
                raise ValueError(f"service Worker {worker.id} must share the local Learner node scope")
            endpoint = urlsplit(worker.endpoint)
            try:
                endpoint_port = endpoint.port
            except ValueError as exc:
                raise ValueError(f"service Worker {worker.id} endpoint has an invalid port") from exc
            if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
                    or endpoint.username or endpoint.password or endpoint.path not in {"", "/"}
                    or endpoint.query or endpoint.fragment or endpoint_port is None):
                raise ValueError(f"service Worker {worker.id} endpoint must be an explicit origin")
            if not _private_or_local_host(endpoint.hostname):
                raise ValueError(f"service Worker {worker.id} endpoint must use a private or local host")
            if endpoint_port != worker.port:
                raise ValueError(f"service Worker {worker.id} endpoint port must match worker.port")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", worker.auth_token_env) is None:
                raise ValueError(f"service Worker {worker.id} auth_token_env is invalid")
            if worker.auth_token_env == "ARC_ENV_TOKEN":
                raise ValueError(f"service Worker {worker.id} must use its per-Worker token environment name")
            if worker.auth_token_env in service_auth_names:
                raise ValueError("each service Worker must have a distinct auth_token_env")
            service_auth_names.add(worker.auth_token_env)
            if not worker.boot_id:
                raise ValueError(f"service Worker {worker.id} requires the local host boot_id pin")
            if worker.model_sha256 and re.fullmatch(r"[0-9a-f]{64}", worker.model_sha256) is None:
                raise ValueError(f"service Worker {worker.id} model_sha256 must be lowercase SHA-256")
            continue

        if worker.host is not None:
            raise ValueError(f"Gateway Worker {worker.id} must not configure a direct node host")
        if worker.id != worker.node_id:
            raise ValueError(f"Gateway Worker {worker.id} id must equal its registered node_id")
        if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", worker.node_id) is None:
            raise ValueError(f"Gateway Worker {worker.id} node_id contains unsafe route characters")
        if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", worker.site_id) is None:
            raise ValueError(f"Gateway Worker {worker.id} site_id contains invalid characters")
        endpoint = urlsplit(worker.endpoint)
        if (endpoint.scheme not in {"http", "https"} or not endpoint.netloc
                or endpoint.username or endpoint.password or endpoint.path not in {"", "/"}
                or endpoint.query or endpoint.fragment):
            raise ValueError(f"Gateway Worker {worker.id} endpoint must be an origin URL without a path")
        if worker.url_prefix != f"/v1/workers/{worker.node_id}":
            raise ValueError(f"Gateway Worker {worker.id} url_prefix must match its exact registered node route")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", worker.auth_token_env) is None:
            raise ValueError(f"Gateway Worker {worker.id} auth_token_env is not a valid environment variable name")
        if worker.auth_token_env == "ARC_ENV_TOKEN":
            raise ValueError(f"Gateway Worker {worker.id} must use a Gateway token separate from ARC_ENV_TOKEN")
        gateway_key = (worker.site_id, worker.endpoint.rstrip("/"))
        previous_auth = gateway_auth.get(gateway_key)
        if previous_auth is not None and previous_auth != worker.auth_token_env:
            raise ValueError(f"Gateway site {worker.site_id} has inconsistent central Gateway settings")
        gateway_auth[gateway_key] = worker.auth_token_env
        if bool(worker.boot_id) != bool(worker.telemetry_epoch):
            raise ValueError(f"Gateway Worker {worker.id} boot_id and telemetry_epoch must be pinned together")
        if worker.id in helper_worker_ids and not worker.boot_id:
            raise ValueError(f"helper Worker {worker.id} requires pinned boot_id and telemetry_epoch")
        if worker.model_sha256 and re.fullmatch(r"[0-9a-f]{64}", worker.model_sha256) is None:
            raise ValueError(f"Gateway Worker {worker.id} model_sha256 must be lowercase SHA-256")
    if config.bepa.enabled:
        teacher_tokens = {config.bepa.teacher.api_key_env, config.bepa.teacher.token_env}
        worker_auth_names = set(gateway_auth.values()) | service_auth_names
        if worker_auth_names & teacher_tokens:
            raise ValueError("Worker authentication environment names must be distinct from BEPA credentials")
    if set(gateway_auth.values()) & service_auth_names:
        raise ValueError("service and Gateway Workers must use distinct authentication environment names")
    if not 0 < config.learner.lr < 1:
        raise ValueError("learner.lr must be positive and less than 1")
    if type(config.rollout.constrained_decoding) is not bool:
        raise ValueError("rollout.constrained_decoding must be boolean")
    if config.rollout.constrained_decoding:
        if config.worker_inference.backend != "ninfer":
            raise ValueError("grammar-conditioned ARC policy currently requires the NInfer backend")
        if config.learner.parallelism != "single":
            raise ValueError("grammar-conditioned training requires single-device gradient executors")
        if config.worker_inference.backend == "ninfer":
            options = config.worker_inference.ninfer_options or NInferOptions()
            if not options.structured_output:
                raise ValueError("grammar-conditioned Native rollout requires ninfer_options.structured_output=true")
    if config.rollout.max_turns < 1 or config.rollout.turn_max_tokens < 1:
        raise ValueError("rollout turn and token limits must be positive")
    inference_limit = config.worker_inference.max_generated_tokens
    if inference_limit < 1 or inference_limit > wire.MAX_GENERATED_TOKENS:
        raise ValueError(
            f"worker_inference.max_generated_tokens must be between 1 and {wire.MAX_GENERATED_TOKENS}")
    if config.rollout.turn_max_tokens > inference_limit:
        raise ValueError(
            "rollout.turn_max_tokens must not exceed worker_inference.max_generated_tokens "
            f"({config.rollout.turn_max_tokens} > {inference_limit}); raise the Worker bound first")
    if inference.max_model_len > wire.MAX_CONTEXT_TOKENS:
        raise ValueError(f"worker_inference.max_model_len must not exceed {wire.MAX_CONTEXT_TOKENS}")
    if config.rollout.prompt_length < 1 or config.rollout.response_length < 1:
        raise ValueError("rollout prompt_length and response_length must be positive")
    if config.rollout.max_filtered_groups < 1 or config.dapo.max_filtered_groups < 1:
        raise ValueError("max_filtered_groups must be positive")
    if config.dapo.updates < 1:
        raise ValueError("dapo.updates must be positive")
    if config.arc.group_timeout <= 0 or config.learner.request_timeout <= 0:
        raise ValueError("HTTP and group timeouts must be positive")
    if (config.rollout.temperature, config.rollout.top_p, config.rollout.top_k,
            config.rollout.repetition_penalty) != (1.0, 1.0, 0, 1.0):
        raise ValueError("sampling must stay at temperature=1, top_p=1, top_k=0, repetition_penalty=1")
    if config.arc.max_active_groups < config.rollout.group_size:
        raise ValueError("arc.max_active_groups must be at least group_size")
    if not 0 < config.arc.overlong_buffer < config.arc.max_generated_tokens:
        raise ValueError("arc.overlong_buffer must be positive and smaller than max_generated_tokens")
