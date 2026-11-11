"""Bounded compression task dispatched by the existing learner compute entry."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from .compression_config import load_config, sha256


def base_metadata(config):
    from qwen_rl.contracts.ternary import (
        EXPECTED_LAYER_TYPES,
        EXPECTED_TEXT_CONFIG,
        expected_t2_parameters,
    )
    from qwen_rl.models.ternary import validate_t2
    from qwen_rl.models.ternary_artifacts import TernaryArtifactStore

    with TernaryArtifactStore(config.artifact["path"]) as store:
        if store.artifact_id.hex() != config.artifact["artifact_id"]:
            raise ValueError("base artifact ID mismatch")
        c = store.text_config
        if (
            any(c.get(k) != v for k, v in EXPECTED_TEXT_CONFIG.items())
            or c.get("layer_types") != EXPECTED_LAYER_TYPES
            or c.get("tie_word_embeddings") is not False
        ):
            raise ValueError(
                "recovery builder requires the existing Qwen3.5 27B geometry"
            )
        shapes = {
            "text/token_embedding": (248320, 5120),
            "text/output_head": (248320, 5120),
        }
        for i, kind in enumerate(c["layer_types"]):
            prefix = f"text/layers/{i}/"
            shapes.update(
                {
                    prefix + "mlp/" + n: s
                    for n, s in [
                        ("gate", (17408, 5120)),
                        ("up", (17408, 5120)),
                        ("down", (5120, 17408)),
                    ]
                }
            )
            local = (
                [
                    ("attention/query", (6144, 5120)),
                    ("attention/gate", (6144, 5120)),
                    ("attention/key", (1024, 5120)),
                    ("attention/value", (1024, 5120)),
                    ("attention/output", (5120, 6144)),
                ]
                if kind == "full_attention"
                else [
                    ("gdn/query", (2048, 5120)),
                    ("gdn/key", (2048, 5120)),
                    ("gdn/value", (6144, 5120)),
                    ("gdn/z", (6144, 5120)),
                    ("gdn/output", (5120, 6144)),
                ]
            )
            shapes.update({prefix + n: s for n, s in local})
        if set(shapes) != expected_t2_parameters() - {"proposal/head"}:
            raise RuntimeError(
                "recovery text inventory differs from the shared architecture"
            )
        basis_counts = {"primal": 0, "hadamard1024_signed": 0}
        for name, expected in shapes.items():
            rows = 0
            for obj, begin, end in store._parts(name):
                n, k, *_ = store._t2_geometry(obj)
                if (
                    k != expected[1]
                    or begin % k
                    or end % k
                    or not 0 <= begin < end <= n * k
                ):
                    raise ValueError("invalid T2 binding geometry: " + name)
                rows += (end - begin) // k
            if rows != expected[0]:
                raise ValueError("T2 binding shape mismatch: " + name)
            codes, words = store.read_t2_parameter(
                name, rows=(0, 1), expected_shape=(1, expected[1])
            )
            validate_t2(codes, words)
            signs_name = "text/output_head" if name == "text/token_embedding" else name
            basis = store.input_basis(signs_name)
            basis_counts[basis] += 1
            if basis == "hadamard1024_signed":
                store.read_hadamard_signs(signs_name, expected[1])
            if (
                name in ("text/token_embedding", "text/output_head")
                and basis != "hadamard1024_signed"
            ):
                raise ValueError(
                    "this recovery vocabulary path requires the rotated T2 vocabulary pair"
                )
        for i, kind in enumerate(c["layer_types"]):
            fused = (
                [f"text/layers/{i}/attention/{n}" for n in ["query", "gate"]]
                if kind == "full_attention"
                else [f"text/layers/{i}/gdn/{n}" for n in ["query", "key", "value"]]
            )
            if len({store.input_basis(n) for n in fused}) != 1:
                raise ValueError("fused input bases disagree before CUDA construction")
        tokenizer_sha = store.text_resource_hashes()["tokenizer.json"]["sha256"]
        c = {**c, "recovery_basis_counts": basis_counts}
    return c, tokenizer_sha


def preflight(config):
    from .compression_data import RecoveryDataset
    from .compression_loss import SafetensorRows

    c, tokenizer_sha = base_metadata(config)
    teacher = SafetensorRows(config.teacher_head["path"], config.teacher_head["key"])
    if teacher.shape != (c["vocab_size"], c["hidden_size"]):
        raise ValueError(
            "teacher head does not match the student vocabulary/hidden size"
        )
    dataset = RecoveryDataset(
        config.dataset_manifest,
        teacher_head_sha256=config.teacher_head["sha256"],
        max_sequence_length=config.max_sequence_length,
        hidden_size=c["hidden_size"],
        vocab_size=c["vocab_size"],
    )
    if dataset.manifest["tokenizer_sha256"] != tokenizer_sha:
        raise ValueError("capture tokenizer differs from the artifact tokenizer")
    mla_tensor_sha = None
    if config.mla_bundle:
        from safetensors import safe_open

        spec = json.loads(config.mla_bundle.read_text())
        full = [
            i for i, kind in enumerate(c["layer_types"]) if kind == "full_attention"
        ]
        if (
            spec.get("schema") != "qwen-rl-mla-factors/v1"
            or spec.get("base_artifact_sha256") != config.artifact["sha256"]
            or spec.get("input_basis") != "primal"
            or spec.get("rope_policy") != "preserve_original_prefix"
            or spec.get("qk_norm") != "upstream_dynamic"
            or spec.get("layers") != full
        ):
            raise ValueError("MLA bundle base/layer identity differs")
        tensor_path = (config.mla_bundle.parent / spec["tensor_file"]).resolve()
        mla_tensor_sha = sha256(tensor_path)
        if mla_tensor_sha != spec["tensor_sha256"]:
            raise ValueError("MLA tensor SHA mismatch")
        with safe_open(str(tensor_path), framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != {
                f"{i}.{n}" for i in full for n in ["down", "up_k", "up_v", "rope"]
            }:
                raise ValueError("MLA tensor inventory differs")
            rd = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
            for i in full:
                rank, width = tensors.get_slice(f"{i}.down").get_shape()
                shapes = {
                    n: tuple(tensors.get_slice(f"{i}.{n}").get_shape())
                    for n in ["up_k", "up_v", "rope"]
                }
                if (
                    width != c["hidden_size"]
                    or rank <= 0
                    or rank % 128
                    or rank + c["num_key_value_heads"] * (rd + 1)
                    >= 2 * c["num_key_value_heads"] * c["head_dim"]
                    or shapes
                    != {
                        "up_k": (c["num_key_value_heads"] * (c["head_dim"] - rd), rank),
                        "up_v": (c["num_key_value_heads"] * c["head_dim"], rank),
                        "rope": (c["num_key_value_heads"] * rd, width),
                    }
                ):
                    raise ValueError(
                        "MLA factor shape mismatch before CUDA construction"
                    )
            import torch

            for key in tensors.keys():
                view = tensors.get_slice(key)
                for start in range(0, view.get_shape()[0], 256):
                    tile = view[start : start + 256]
                    if not tile.is_floating_point() or not bool(
                        torch.isfinite(tile).all()
                    ):
                        raise ValueError(
                            "MLA factors contain nonfinite/nonfloating values before CUDA"
                        )
    return (
        dataset,
        teacher,
        {
            "status": "cpu_preflight_passed",
            "objective": "compression_recovery",
            "artifact_sha256": config.artifact["sha256"],
            "artifact_id": config.artifact["artifact_id"],
            "dataset_sha256": config.dataset_sha256,
            "teacher_head_sha256": config.teacher_head["sha256"],
            "mla_bundle_sha256": config.mla_bundle_sha256,
            "mla_tensor_sha256": mla_tensor_sha,
            "records": {k: len(v) for k, v in dataset.splits.items()},
            "gpu_started": False,
            "test_opened": False,
            "workers_started": False,
        },
    )


def train(config, dataset, teacher, proof):
    import torch

    from .compression_model import load_model
    from .compression_trainer import CompressionRecoveryTrainer

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config.gpu):
        raise RuntimeError(
            "recovery worker GPU visibility does not match its explicit allocation"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("recovery requires exactly one visible CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, evidence = load_model(config, device)
    trainer = CompressionRecoveryTrainer(model, config, teacher, device=device)
    return training_loop(config, dataset, proof, model, evidence, trainer)


def training_loop(config, dataset, proof, model, evidence, trainer):
    """Task logic shared by the real CUDA entry and bounded CPU integration tests."""
    import torch
    from safetensors.torch import save_file

    from qwen_rl.models.ternary_recovery import (
        ScaleTernaryEmbedding,
        ScaleTernaryLinear,
    )

    from .recovery import RecoveryStore, _atomic_json, source_provenance

    compatibility = {
        k: proof[k]
        for k in [
            "artifact_sha256",
            "artifact_id",
            "dataset_sha256",
            "teacher_head_sha256",
            "mla_bundle_sha256",
            "mla_tensor_sha256",
        ]
    }
    compatibility.update(
        task="compression_recovery/v1",
        trainable=list(config.trainable),
        ce_weight=config.ce_weight,
        kl_weight=config.kl_weight,
        max_sequence_length=config.max_sequence_length,
        records_per_update=config.records_per_update,
        vocab_chunk_rows=config.vocab_chunk_rows,
        gradient_checkpointing=config.gradient_checkpointing,
        activation_offload=config.activation_offload,
        learning_rate=config.lr,
        vocab_scale_learning_rate=getattr(config, "vocab_scale_lr", config.lr),
    )
    store = RecoveryStore(
        config.run_dir / "compression_training", mode=config.recovery_mode
    )
    resumed = store.load(model=model, trainer=trainer, compatibility=compatibility)
    epoch = uuid4().hex
    software = source_provenance()
    import qwen_rl.models.ternary as shared_ternary
    import qwen_rl.models.ternary_recovery as recovery_ternary

    from . import compression_loss, compression_model, compression_trainer

    software["on_disk_compression_sources_at_start"] = {
        Path(module.__file__).name: sha256(module.__file__)
        for module in [
            sys.modules[__name__],
            compression_loss,
            compression_model,
            compression_trainer,
            shared_ternary,
            recovery_ternary,
        ]
    }

    def checkpoint(reason):
        return store.save(
            model=model,
            trainer=trainer,
            compatibility=compatibility,
            policy_version=trainer.state.global_step,
            engine_epoch=epoch,
            update_context={
                "objective": "compression_recovery",
                "record_cursor": trainer.state.global_step * config.records_per_update,
            },
            reason=reason,
            source_provenance=software,
        )

    if resumed is None:
        checkpoint("compression_initial_state")
    durable_step = trainer.state.global_step
    checkpoint_steps = getattr(config, "checkpoint_steps", 1)
    if type(checkpoint_steps) is not int or checkpoint_steps < 1:
        raise ValueError("checkpoint_steps must be a positive integer")
    if trainer.state.global_step > config.steps:
        raise ValueError("requested steps precede the durable recovery state")
    report = dict(
        proof,
        status="training",
        gpu_started=trainer.recovery_device.type == "cuda",
        model=evidence,
        resumed=resumed is not None,
        lineage_id=store.lineage_id,
        updates=[],
        checkpoint_steps=checkpoint_steps,
        durable_completed_updates=durable_step,
    )
    report_path = config.run_dir / "compression-recovery.json"
    if report_path.exists():
        prior_report = json.loads(report_path.read_text())
        if prior_report.get("lineage_id") == store.lineage_id:
            report["baseline_validation"] = prior_report.get("baseline_validation")
    _atomic_json(report_path, report)
    try:
        if trainer.state.global_step == 0 and not report.get("baseline_validation"):
            report["baseline_validation"] = trainer.evaluate_records(
                dataset.load(i) for i in dataset.splits["validation"]
            )
            _atomic_json(report_path, report)
        train_ids = dataset.splits["train"]
        while trainer.state.global_step < config.steps:
            cursor = trainer.state.global_step * config.records_per_update
            records = [
                dataset.load(train_ids[(cursor + i) % len(train_ids)])
                for i in range(config.records_per_update)
            ]
            metrics = trainer.update_records(records)
            save_now = (trainer.state.global_step % checkpoint_steps == 0
                        or trainer.state.global_step == config.steps)
            metrics["snapshot"] = (
                checkpoint("compression_optimizer_completed") if save_now else None
            )
            if save_now:
                durable_step = trainer.state.global_step
            metrics["durable_completed_updates"] = durable_step
            report["durable_completed_updates"] = durable_step
            report["updates"].append(metrics)
            _atomic_json(report_path, report)
        report["validation"] = trainer.evaluate_records(
            dataset.load(i) for i in dataset.splits["validation"]
        )
        export_dir = (
            store.lineage_dir
            / "compression_exports"
            / f"step-{trainer.state.global_step:08d}"
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        scale_state = {
            f"{name}.scales": module.exported_scale_words()
            .view(torch.float16)
            .cpu()
            .contiguous()
            for name, module in model.named_modules()
            if isinstance(module, (ScaleTernaryEmbedding, ScaleTernaryLinear))
        }
        if scale_state:
            destination = export_dir / "fp16-scales.safetensors"
            if destination.exists():
                from safetensors.torch import load_file

                prior = load_file(str(destination))
                if set(prior) != set(scale_state) or any(
                    not torch.equal(prior[n], v) for n, v in scale_state.items()
                ):
                    raise ValueError(
                        "existing immutable scale export differs from the restored state"
                    )
            else:
                save_file(scale_state, str(destination))
        export = dict(
            compatibility,
            schema="qwen-rl-compression-export/v1",
            completed_updates=trainer.state.global_step,
            full_training_snapshot=str(store.current_snapshot_path),
            fp16_scales="fp16-scales.safetensors" if scale_state else None,
            fp16_scales_sha256=sha256(export_dir / "fp16-scales.safetensors")
            if scale_state
            else None,
            native_deployable=False,
            requires_native_composition=True,
            mla_requires_requantization=config.mla_bundle is not None,
            workers_published=False,
        )
        from .compression_trainer import effective_vocab_scale_changes

        export["effective_vocab_scale_word_changes"] = effective_vocab_scale_changes(
            model
        )
        export["effective_vocab_forward_scale_changes"] = effective_vocab_scale_changes(
            model, compute=True
        )
        export_path = export_dir / "manifest.json"
        if export_path.exists():
            if json.loads(export_path.read_text()) != export:
                raise ValueError("existing immutable recovery export identity differs")
        else:
            _atomic_json(export_path, export)
        report.update(
            status="completed",
            completed_updates=trainer.state.global_step,
            export_manifest=str(export_dir / "manifest.json"),
            rl_policy_published=False,
            effective_vocab_scale_word_changes=export[
                "effective_vocab_scale_word_changes"
            ],
            effective_vocab_forward_scale_changes=export[
                "effective_vocab_forward_scale_changes"
            ],
        )
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        device = trainer.recovery_device
        report["gpu_started"] = device.type == "cuda"
        report["peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        report["peak_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        )
        _atomic_json(report_path, report)
    return report


def run_compression_recovery(profile, *, check_only=False):
    config = load_config(profile)
    _, _, proof = preflight(config)
    if check_only:
        return proof
    config.run_dir.mkdir(parents=True, exist_ok=True)
    with (config.run_dir / ".compression-recovery.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "this compression run directory already has an owner"
            ) from exc
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config.gpu))
        from .cuda_allocator import configure_learner_allocator

        configure_learner_allocator(env)
        command = [
            sys.executable,
            "-B",
            "-m",
            __name__,
            "--worker",
            str(config.path),
            sha256(config.path),
            config.dataset_sha256,
            config.mla_bundle_sha256 or "-",
        ]
        with (config.run_dir / "compression-worker.log").open("ab") as log:
            child = subprocess.Popen(
                command, env=env, stdout=log, stderr=log, start_new_session=True
            )
            try:
                code = child.wait(timeout=config.max_wall_seconds)
            except BaseException as exc:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                from .recovery import _atomic_json

                _atomic_json(
                    config.run_dir / "compression-stop.json",
                    {
                        "status": "stopped",
                        "objective": "compression_recovery",
                        "pid": child.pid,
                        "reason": type(exc).__name__,
                        "max_wall_seconds": config.max_wall_seconds,
                        "durable_state_preserved": True,
                    },
                )
                raise
        if code:
            raise RuntimeError(
                f"compression worker failed ({code}); see compression-worker.log"
            )
        report = json.loads((config.run_dir / "compression-recovery.json").read_text())
        if report.get("status") != "completed":
            raise RuntimeError("compression worker did not complete")
        return report


if __name__ == "__main__":
    if len(sys.argv) != 6 or sys.argv[1] != "--worker":
        raise SystemExit(
            "internal worker; use qwen-rl learner compute --config PROFILE"
        )
    _, _, profile, expected_profile, expected_dataset, expected_mla = sys.argv
    if sha256(profile) != expected_profile:
        raise SystemExit("profile changed before worker startup")
    config = load_config(profile)
    if (
        config.dataset_sha256 != expected_dataset
        or (config.mla_bundle_sha256 or "-") != expected_mla
    ):
        raise SystemExit("dataset or MLA manifest changed before worker startup")
    dataset, teacher, proof = preflight(config)
    train(config, dataset, teacher, proof)
