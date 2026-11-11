"""Write the Learner's CPU LoRA state directly as a native NInfer T2 adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import tempfile
from time import perf_counter

import torch

from qwen_rl._vendor.ninfer_artifact.schema import TensorSpec
from .policy_artifact_writer import PolicyArtifactWriter as ArtifactWriter
from qwen_rl.contracts.lora_targets import gdn_lora_enabled, projection_input_bases

SCHEMA = "qwen-rl-native-policy/v1"
CODER390_SCHEMA = "qwen-rl-native-policy/v2"
MODEL_NAME = "bonsai2-27b-arc-k2-mtp"
RECIPE = "bonsai2_27b_ternary"
CODER390_MODEL_NAME = "coder390-t2-full-scales-step512"
ADAPTER_FORMAT = "ninfer-lora-t2-v1"
CODER390_ADAPTER_FORMAT = "ninfer-lora-t2-v2"
MODEL_FAMILY = "qwen3_5_dense"
INPUT_BASIS = "prism_rotated"
BASE_WEIGHT_PROFILE = "t2_g128_fp16_rotated"
SCALE_FOLDING = "alpha_over_r_into_B"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_ID = re.compile(r"[0-9a-f]{32}\Z")
_HASH_CHUNK = 8 * 1024 * 1024


def _require_digest(value: object, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _identity_contract(base_identity: dict) -> dict:
    if not isinstance(base_identity, dict):
        raise TypeError("base_identity must be the pinned artifact identity document")
    from qwen_rl.contracts.ternary import validate_artifact_identity

    try:
        base_identity = validate_artifact_identity(base_identity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"base identity violates the pinned T2 contract: {exc}") from exc
    if (base_identity.get("schema_version") != "qwen-rl-ternary-artifact/v1"
            or base_identity.get("artifact_kind") != "frozen_ninfer_t2_learner_base"):
        raise ValueError("base identity is not a frozen T2 Learner artifact")

    model = base_identity.get("model")
    if not isinstance(model, dict):
        raise ValueError("base identity omitted its model profile")
    model_id = model.get("id")
    if model_id == CODER390_MODEL_NAME:
        return _coder390_identity_contract(base_identity)
    if (model_id != MODEL_NAME
            or model.get("architecture") != "Qwen3_5ForCausalLM"
            or model.get("model_type") != "qwen3_5_text"):
        raise ValueError("base identity does not describe a supported Qwen3.5 T2 model")
    geometry = model.get("geometry")
    if not isinstance(geometry, dict):
        raise TypeError("base identity omitted model geometry")
    geometry_names = (
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
    )
    dimensions = {}
    for name in geometry_names:
        value = geometry.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"base identity has invalid {name}")
        dimensions[name] = value
    layers = model.get("layer_types")
    if (not isinstance(layers, list) or len(layers) != dimensions["num_hidden_layers"]
            or any(item not in ("linear_attention", "full_attention") for item in layers)):
        raise ValueError("base identity has unsupported layer_types")

    source = base_identity.get("source")
    ninfer = source.get("ninfer_artifact") if isinstance(source, dict) else None
    from qwen_rl.contracts.ternary import artifact_report

    report = artifact_report(base_identity)
    if not isinstance(ninfer, dict) or not isinstance(report, dict):
        raise TypeError("base identity omitted the NInfer artifact or conversion report")
    if ninfer.get("name") != MODEL_NAME or ninfer.get("recipe") != RECIPE or ninfer.get("converter") != "ninfer-v3":
        raise ValueError("base identity NInfer provenance is unsupported")
    artifact_id = _require_digest(ninfer.get("artifact_id"), "base NInfer artifact ID", _ARTIFACT_ID)
    artifact_sha = _require_digest(ninfer.get("sha256"), "base NInfer SHA-256", _SHA256)
    report_sha = _require_digest(report.get("sha256"), "base conversion report SHA-256", _SHA256)

    representation = base_identity.get("representation")
    if (not isinstance(representation, dict)
            or representation.get("format") != "t2_g128_fp16"
            or representation.get("layout") != "row_split_k128_v1"):
        raise ValueError("base identity T2 representation is unsupported")
    rotation = base_identity.get("rotation")
    if not isinstance(rotation, dict) or rotation.get("input_basis") != INPUT_BASIS:
        raise ValueError("base identity LoRA input basis is unsupported")
    if f"{representation['format']}_rotated" != BASE_WEIGHT_PROFILE:
        raise ValueError("base identity weight profile is unsupported")
    learner = base_identity.get("learner")
    adapter = learner.get("adapter") if isinstance(learner, dict) else None
    if (not isinstance(adapter, dict) or learner.get("base_trainable") is not False
            or adapter.get("format") != "standard_peft_lora"
            or adapter.get("input_basis") != ("per_use" if gdn_lora_enabled(base_identity) else INPUT_BASIS)
            or adapter.get("output_basis") != "projection_output"
            or adapter.get("scale_alpha_over_rank") != 1.0
            or adapter.get("dropout") != 0.0 or adapter.get("bias") != "none"):
        raise ValueError("base identity LoRA contract is unsupported")
    include_gdn = gdn_lora_enabled(base_identity)
    if include_gdn:
        from qwen_rl.contracts.ternary import validate_artifact_identity
        validate_artifact_identity(base_identity)
        for key in ("linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads", "linear_value_head_dim"):
            value = geometry.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"all-GDN base identity has invalid {key}")
            dimensions[key] = value
    return {
        "gdn_projections": include_gdn,
        "artifact_id": artifact_id,
        "artifact_sha256": artifact_sha,
        "conversion_report_sha256": report_sha,
        "geometry": dimensions,
        "layer_types": list(layers),
        "adapter": adapter,
        "input_basis": adapter["input_basis"],
        "input_basis_by_module": projection_input_bases(base_identity) if include_gdn else None,
        "base_weight_profile": BASE_WEIGHT_PROFILE,
        "artifact_report": report,
    }


def _coder390_identity_contract(base_identity: dict) -> dict:
    from qwen_rl.contracts.ternary import artifact_report, validate_artifact_identity

    try:
        validated = validate_artifact_identity(base_identity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"base identity is not a valid Coder390 T2 artifact: {exc}") from exc
    report = artifact_report(validated)
    model = validated["model"]
    geometry = model.get("geometry")
    if not isinstance(geometry, dict):
        raise TypeError("Coder390 identity omitted model geometry")
    geometry_names = (
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
    )
    dimensions: dict[str, int] = {}
    for name in geometry_names:
        value = geometry.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Coder390 identity has invalid {name}")
        dimensions[name] = value
    layers = model.get("layer_types")
    if (not isinstance(layers, list) or len(layers) != dimensions["num_hidden_layers"]
            or any(item not in ("linear_attention", "full_attention") for item in layers)):
        raise ValueError("Coder390 identity has unsupported layer_types")

    source = validated.get("source")
    ninfer = source.get("ninfer_artifact") if isinstance(source, dict) else None
    if not isinstance(ninfer, dict) or not isinstance(report, dict):
        raise TypeError("Coder390 identity omitted the NInfer artifact or run report")
    artifact_id = _require_digest(ninfer.get("artifact_id"), "base NInfer artifact ID", _ARTIFACT_ID)
    artifact_sha = _require_digest(ninfer.get("sha256"), "declared base NInfer SHA-256", _SHA256)
    if ninfer.get("profile") != CODER390_MODEL_NAME:
        raise ValueError("Coder390 identity NInfer artifact profile differs from its model ID")
    if (report.get("artifact_id") != artifact_id
            or report.get("t2_object_count") != 322
            or report.get("t2_logical_parameter_count") != 514):
        raise ValueError("Coder390 run report differs from the pinned full-scales artifact")

    representation = validated.get("representation")
    if (not isinstance(representation, dict)
            or representation.get("format") != "t2_g128_fp16"
            or representation.get("layout") != "row_split_k128_v1"):
        raise ValueError("Coder390 T2 representation is unsupported")
    rotation = validated.get("rotation")
    if not isinstance(rotation, dict) or rotation.get("input_basis") != "per_use":
        raise ValueError("Coder390 LoRA input basis must be per_use")
    learner = validated.get("learner")
    adapter = learner.get("adapter") if isinstance(learner, dict) else None
    if (not isinstance(adapter, dict) or learner.get("base_trainable") is not False
            or adapter.get("format") != "standard_peft_lora"
            or adapter.get("input_basis") != "per_use"
            or adapter.get("output_basis") != "projection_output"
            or adapter.get("scale_alpha_over_rank") != 1.0
            or adapter.get("dropout") != 0.0 or adapter.get("bias") != "none"):
        raise ValueError("Coder390 identity LoRA contract is unsupported")
    module_basis = adapter.get("input_basis_by_module")
    if not isinstance(module_basis, dict):
        raise TypeError("Coder390 identity omitted its PEFT module-basis map")
    return {
        "artifact_id": artifact_id,
        "artifact_sha256": artifact_sha,
        "conversion_report_sha256": None,
        "geometry": dimensions,
        "layer_types": list(layers),
        "adapter": adapter,
        "input_basis": "per_use",
        "input_basis_by_module": module_basis,
        "base_weight_profile": "t2_g128_fp16_per_use",
        "artifact_report": report,
    }


def _state_keys(layers: list[str], *, include_gdn: bool = False) -> set[str]:
    keys: set[str] = set()
    for layer, kind in enumerate(layers):
        prefix = f"base_model.model.model.layers.{layer}."
        modules = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        if kind == "full_attention":
            modules.extend((
                "self_attn.q_proj", "self_attn.k_proj",
                "self_attn.v_proj", "self_attn.o_proj",
            ))
        elif include_gdn:
            modules.extend("linear_attn." + name for name in (
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"))
        for module in modules:
            for side in ("A", "B"):
                keys.add(prefix + module + f".lora_{side}.weight")
    return keys


def _adapter_module_paths(layers: list[str], *, include_gdn: bool = False) -> set[str]:
    paths: set[str] = set()
    for layer, kind in enumerate(layers):
        prefix = f"model.layers.{layer}."
        paths.update(prefix + "mlp." + module for module in ("gate_proj", "up_proj", "down_proj"))
        if kind == "full_attention":
            paths.update(prefix + "self_attn." + module for module in (
                "q_proj", "k_proj", "v_proj", "o_proj",
            ))
        elif include_gdn:
            paths.update(prefix + "linear_attn." + name for name in (
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"))
    return paths


def _split_q_projection_b(weight: torch.Tensor, heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    expected_rows = 2 * heads * head_dim
    if weight.ndim != 2 or weight.shape[0] != expected_rows:
        raise ValueError("q_proj B must contain interleaved query/gate rows for every head")
    query_rows = []
    gate_rows = []
    for head in range(heads):
        start = head * 2 * head_dim
        query_rows.append(weight[start : start + head_dim])
        gate_rows.append(weight[start + head_dim : start + 2 * head_dim])
    return torch.cat(query_rows, dim=0), torch.cat(gate_rows, dim=0)


def _to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bfloat16:
        return tensor.contiguous()
    return tensor.to(dtype=torch.bfloat16).contiguous()


def _map_adapter_state(
    state: dict[str, torch.Tensor], *, layers: list[str], geometry: dict,
    rank: int, alpha: int, input_basis_by_module: dict[str, str] | None = None,
    include_gdn: bool = False,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    expected = _state_keys(layers, include_gdn=include_gdn)
    if any(not isinstance(key, str) for key in state):
        raise TypeError("adapter state keys must be strings")
    actual = set(state)
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"adapter tensor coverage mismatch: missing={missing[:8]} extra={extra[:8]}")
    if alpha / rank != 1.0:
        raise ValueError("this NInfer LoRA ABI requires alpha/rank == 1")
    if input_basis_by_module is not None:
        expected_modules = _adapter_module_paths(layers, include_gdn=include_gdn)
        if set(input_basis_by_module) != expected_modules:
            missing = sorted(expected_modules - set(input_basis_by_module))
            extra = sorted(set(input_basis_by_module) - expected_modules)
            raise ValueError(
                "adapter module-basis coverage mismatch: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        invalid = {
            module: basis for module, basis in input_basis_by_module.items()
            if basis not in ("prism_rotated", "primal")
        }
        if invalid:
            raise ValueError(f"adapter module-basis values are unsupported: {invalid}")
    if sys.byteorder != "little":
        raise RuntimeError("NInfer contiguous_le_v1 adapter output requires a little-endian host")

    factors: dict[str, torch.Tensor] = {}
    sites: list[dict] = []
    checked: dict[str, torch.Tensor] = {}

    def get(module: str, layer: int, side: str) -> torch.Tensor:
        name = f"base_model.model.model.layers.{layer}.{module}.lora_{side}.weight"
        if name in checked:
            return checked[name]
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}: LoRA factor must be a tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name}: native Learner export accepts CPU tensors only")
        if tensor.layout != torch.strided or tensor.ndim != 2:
            raise ValueError(f"{name}: LoRA factor must be a dense rank-2 tensor")
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"{name}: LoRA factor must be FP16, BF16, or FP32")
        checked[name] = tensor
        return tensor

    def add_site(
        layer: int, site: str, a: torch.Tensor, b: torch.Tensor,
        *, a_object: str | None = None, module_path: str | None = None,
    ) -> None:
        prefix = f"text/layers/{layer}/{site}"
        a_id = a_object or prefix + "/lora_a"
        b_id = prefix + "/lora_b"
        # Query/gate deliberately reference the same source A factor.
        if a_id not in factors:
            factors[a_id] = _to_bf16(a)
        factors[b_id] = _to_bf16(b)
        record = {
            "site": prefix, "a_object": a_id, "b_object": b_id,
            "a_shape": list(factors[a_id].shape), "b_shape": list(factors[b_id].shape),
        }
        if input_basis_by_module is not None:
            if module_path is None or module_path not in input_basis_by_module:
                raise ValueError(f"adapter site {prefix} omitted its PEFT module basis")
            record["input_basis"] = input_basis_by_module[module_path]
        sites.append(record)

    hidden = geometry["hidden_size"]
    intermediate = geometry["intermediate_size"]
    heads = geometry["num_attention_heads"]
    kv_heads = geometry["num_key_value_heads"]
    head_dim = geometry["head_dim"]
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim

    for layer, kind in enumerate(layers):
        for module, output_width, input_width in (
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        ):
            a = get("mlp." + module, layer, "A")
            b = get("mlp." + module, layer, "B")
            if tuple(a.shape) != (rank, input_width) or tuple(b.shape) != (output_width, rank):
                raise ValueError(f"layer {layer} MLP {module} LoRA dimensions disagree with base identity")
            add_site(
                layer, "mlp/" + module.removesuffix("_proj"), a, b,
                module_path=f"model.layers.{layer}.mlp.{module}",
            )

        if kind == "linear_attention" and include_gdn:
            key_width = geometry["linear_num_key_heads"] * geometry["linear_key_head_dim"]
            value_width = geometry["linear_num_value_heads"] * geometry["linear_value_head_dim"]
            control_width = geometry["linear_num_value_heads"]
            module_prefix = f"model.layers.{layer}.linear_attn."
            a = get("linear_attn.in_proj_qkv", layer, "A")
            b = get("linear_attn.in_proj_qkv", layer, "B")
            if tuple(a.shape) != (rank, hidden) or tuple(b.shape) != (2 * key_width + value_width, rank):
                raise ValueError(f"layer {layer} GDN QKV LoRA dimensions disagree with base identity")
            shared_a = f"text/layers/{layer}/gdn/query_key_value/lora_a"
            begin = 0
            for name, width in (("query", key_width), ("key", key_width), ("value", value_width)):
                add_site(layer, "gdn/" + name, a, b[begin:begin+width],
                         a_object=shared_a, module_path=module_prefix + "in_proj_qkv")
                begin += width
            for module, site, inp, out in (
                ("in_proj_z", "z", hidden, value_width),
                ("in_proj_a", "a_projection", hidden, control_width),
                ("in_proj_b", "b_projection", hidden, control_width),
                ("out_proj", "output", value_width, hidden),
            ):
                a = get("linear_attn." + module, layer, "A")
                b = get("linear_attn." + module, layer, "B")
                if tuple(a.shape) != (rank, inp) or tuple(b.shape) != (out, rank):
                    raise ValueError(f"layer {layer} GDN {module} LoRA dimensions disagree with base identity")
                add_site(layer, "gdn/" + site, a, b, module_path=module_prefix + module)
        if kind != "full_attention":
            continue
        a_q = get("self_attn.q_proj", layer, "A")
        b_q = get("self_attn.q_proj", layer, "B")
        a_k = get("self_attn.k_proj", layer, "A")
        b_k = get("self_attn.k_proj", layer, "B")
        a_v = get("self_attn.v_proj", layer, "A")
        b_v = get("self_attn.v_proj", layer, "B")
        a_o = get("self_attn.o_proj", layer, "A")
        b_o = get("self_attn.o_proj", layer, "B")
        if any(tuple(value.shape) != (rank, hidden) for value in (a_q, a_k, a_v)):
            raise ValueError(f"layer {layer} Q/K/V LoRA A dimensions disagree with base identity")
        if tuple(b_q.shape) != (2 * q_width, rank):
            raise ValueError(f"layer {layer} q_proj B must include query and attention-gate rows")
        if tuple(a_o.shape) != (rank, q_width) or tuple(b_o.shape) != (hidden, rank):
            raise ValueError(f"layer {layer} O LoRA dimensions disagree with base identity")
        if any(tuple(value.shape) != (kv_width, rank) for value in (b_k, b_v)):
            raise ValueError(f"layer {layer} K/V LoRA B dimensions disagree with base identity")
        b_query, b_gate = _split_q_projection_b(b_q, heads, head_dim)
        shared_a = f"text/layers/{layer}/attention/query_gate/lora_a"
        module_prefix = f"model.layers.{layer}.self_attn."
        add_site(
            layer, "attention/query", a_q, b_query, a_object=shared_a,
            module_path=module_prefix + "q_proj",
        )
        add_site(
            layer, "attention/attention_gate", a_q, b_gate, a_object=shared_a,
            module_path=module_prefix + "q_proj",
        )
        add_site(layer, "attention/key", a_k, b_k, module_path=module_prefix + "k_proj")
        add_site(layer, "attention/value", a_v, b_v, module_path=module_prefix + "v_proj")
        add_site(layer, "attention/output", a_o, b_o, module_path=module_prefix + "o_proj")
    return factors, sites


def _write_json_atomic(path: Path, value: dict) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        published = True
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
        raise


def export_native_policy(
    state: dict[str, torch.Tensor], *, output: Path, version: int,
    adapter_sha256: str, adapter_config_sha256: str | None, base_identity: dict,
    rank: int = 16, alpha: int = 16,
) -> dict:
    """Publish CPU PEFT factors with canonical PEFT keys and an adjacent descriptor.

    The descriptor sidecar is named by appending .json to output. Timings measure
    factor mapping, artifact writing, and artifact hashing; export_seconds is their sum.
    """
    if type(version) is not int or version < 0:
        raise ValueError("version must be a nonnegative integer")
    if type(rank) is not int or rank <= 0 or type(alpha) is not int or alpha <= 0:
        raise ValueError("rank and alpha must be positive integers")
    if not isinstance(state, dict):
        raise TypeError("state must be a dictionary of canonical PEFT tensor keys")
    source_sha = _require_digest(adapter_sha256, "adapter SHA-256", _SHA256)
    config_sha = (
        _require_digest(adapter_config_sha256, "adapter config SHA-256", _SHA256)
        if adapter_config_sha256 is not None else None
    )
    identity = _identity_contract(base_identity)
    base_adapter = identity["adapter"]
    if base_adapter.get("rank") != rank or base_adapter.get("alpha") != alpha:
        raise ValueError("rank/alpha differ from the pinned base identity LoRA contract")
    if alpha / rank != 1.0:
        raise ValueError("this NInfer LoRA ABI requires alpha/rank == 1")

    output = Path(output)
    descriptor_path = Path(str(output) + ".json")
    if output.is_symlink() or output.exists() or descriptor_path.is_symlink() or descriptor_path.exists():
        raise FileExistsError(f"native policy output already exists: {output} or {descriptor_path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    mapping_started = perf_counter()
    factors, sites = _map_adapter_state(
        state,
        layers=identity["layer_types"],
        geometry=identity["geometry"],
        rank=rank,
        alpha=alpha,
        input_basis_by_module=identity.get("input_basis_by_module"),
        include_gdn=identity.get("gdn_projections", False),
    )
    mapping_seconds = perf_counter() - mapping_started

    metadata = {
        "format": (
            CODER390_ADAPTER_FORMAT
            if identity["input_basis"] == "per_use"
            else ADAPTER_FORMAT
        ),
        "model_family": MODEL_FAMILY,
        "base_ninfer_artifact_id": identity["artifact_id"],
        "base_ninfer_sha256": identity["artifact_sha256"],
        "source_adapter_sha256": source_sha,
        "adapter_config_sha256": config_sha,
        "policy_version": version,
        "rank": rank,
        "alpha": alpha,
        "scale_folding": SCALE_FOLDING,
        "sites": sites,
        "input_basis": identity["input_basis"],
        "base_weight_profile": identity["base_weight_profile"],
    }
    if identity.get("gdn_projections", False):
        metadata["gdn_projections"] = True
    if identity["input_basis"] == "per_use":
        provenance = {
            "producer": "qwen-rl-native-policy-export",
            "base_ninfer_artifact_id": identity["artifact_id"],
            "base_ninfer_sha256": identity["artifact_sha256"],
            "base_artifact_report": identity["artifact_report"],
            "source_adapter_sha256": source_sha,
            "adapter_config_sha256": config_sha,
            "policy_version": version,
            "input_basis": identity["input_basis"],
            "base_weight_profile": identity["base_weight_profile"],
        }
    else:
        provenance = {
            "converter": "ninfer-v3-qwen3_5-bonsai2-t2-lora",
            "base_ninfer_artifact_id": identity["artifact_id"],
            "base_ninfer_sha256": identity["artifact_sha256"],
            "base_conversion_report_sha256": identity["conversion_report_sha256"],
            "source_adapter_sha256": source_sha,
            "adapter_config_sha256": config_sha,
            "policy_version": version,
            "input_basis": identity["input_basis"],
            "base_weight_profile": identity["base_weight_profile"],
        }
    specs = [
        TensorSpec(name, tuple(tensor.shape), "bf16", "contiguous_le_v1")
        for name, tensor in factors.items()
    ]
    components = {
        "adapter": {
            "config": {
                "architectures": ["NInferQwen3_5LoRA"],
                "model_type": "ninfer_lora",
            },
        },
    }

    artifact_created = False
    descriptor_created = False
    try:
        write_started = perf_counter()
        with ArtifactWriter(
            output,
            specs,
            components=components,
            bindings={},
            metadata={"ninfer_adapter": metadata},
            provenance=provenance,
        ) as writer:
            artifact_id = writer.artifact_id.hex()
            for name, tensor in factors.items():
                data = memoryview(tensor.detach().contiguous().view(torch.uint8).numpy()).cast("B")
                writer.write_object(name, data)
        artifact_created = True
        write_seconds = perf_counter() - write_started

        timings = {
            "mapping_seconds": mapping_seconds,
            "write_seconds": write_seconds,
            "hash_seconds": None,
            "export_seconds": mapping_seconds + write_seconds,
        }
        descriptor = {
            "schema": CODER390_SCHEMA if identity["input_basis"] == "per_use" else SCHEMA,
            "version": version,
            "source_adapter_sha256": source_sha,
            "adapter_config_sha256": config_sha,
            "base_ninfer_artifact_id": identity["artifact_id"],
            "base_ninfer_sha256": identity["artifact_sha256"],
            "rank": rank,
            "alpha": alpha,
            "input_basis": identity["input_basis"],
            "base_weight_profile": identity["base_weight_profile"],
            "artifact_sha256": None,
            "artifact_sha256_verified": False,
            "artifact_id": artifact_id,
            "sites": sites,
            "timings": timings,
        }
        if identity.get("gdn_projections", False):
            descriptor["gdn_projections"] = True
        _write_json_atomic(descriptor_path, descriptor)
        descriptor_created = True
        return descriptor
    except BaseException:
        if descriptor_created:
            descriptor_path.unlink(missing_ok=True)
        if artifact_created:
            output.unlink(missing_ok=True)
        raise
