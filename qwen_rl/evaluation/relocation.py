"""Relocate native evaluation storage references and validate CPU initialization."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..runtime.manifest import read_json, write_json_atomic


class EvaluationRelocationError(RuntimeError):
    """An evaluation state tree cannot be safely relocated or initialized."""


def _absolute_path(value: str | Path, *, label: str) -> Path:
    path = Path(os.path.expanduser(str(value)))
    if not str(value).strip() or "\x00" in str(value) or not path.is_absolute():
        raise EvaluationRelocationError(f"{label} must be an absolute path")
    return Path(os.path.abspath(path))


def _normalize_mappings(path_mappings: Mapping[str | Path, str | Path]) -> tuple[tuple[Path, Path], ...]:
    if not isinstance(path_mappings, Mapping):
        raise EvaluationRelocationError("path_mappings must be an explicit old-root to new-root mapping")
    mappings: list[tuple[Path, Path]] = []
    for old, new in path_mappings.items():
        mappings.append((_absolute_path(old, label="old path mapping root"),
                         _absolute_path(new, label="new path mapping root")))
    return tuple(sorted(mappings, key=lambda pair: len(pair[0].parts), reverse=True))


def _relocate_path(value: Any, mappings: tuple[tuple[Path, Path], ...], *, label: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise EvaluationRelocationError(f"{label} must be a non-empty path or null")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        # Evaluation result and episode-artifact references are relative to the
        # result output root. They remain valid after the enclosing tree copy.
        return value
    path = _absolute_path(raw, label=label)
    for old, new in mappings:
        try:
            suffix = path.relative_to(old)
        except ValueError:
            continue
        return str(new / suffix)
    if any(path.is_relative_to(target) for _old, target in mappings):
        return value
    raise EvaluationRelocationError(f"{label} is outside explicit source and target path mappings")


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = read_json(path)
    except Exception as exc:
        raise EvaluationRelocationError(f"{label} is unreadable ({type(exc).__name__})") from exc
    if not isinstance(value, dict):
        raise EvaluationRelocationError(f"{label} must contain a JSON object")
    return value


def _write_if_changed(path: Path, value: dict[str, Any], before: dict[str, Any]) -> bool:
    if value == before:
        return False
    write_json_atomic(path, value)
    return True


def _validate_policy_publication(
    runtime: Any,
    publication: Mapping[str, Any],
    *,
    policy_version: int,
) -> None:
    from ..rollout.groups import PolicySnapshot

    if publication.get("version") != policy_version:
        raise EvaluationRelocationError("current policy publication does not match recovered policy_version")
    try:
        snapshot = PolicySnapshot.from_policy(dict(publication))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationRelocationError("current policy publication has invalid stable identity") from exc

    def validate_files(
        files_key: str,
        transport_key: str,
        *,
        declared_optional: bool = False,
    ) -> dict[str, int]:
        declared = publication.get(files_key)
        transport = publication.get(transport_key)
        if not isinstance(transport, Mapping):
            raise EvaluationRelocationError(f"current policy publication has invalid {files_key} transport")
        if declared is None and declared_optional:
            if set(transport) != {"adapter_config.json", "adapter_model.safetensors"}:
                raise EvaluationRelocationError("current policy publication has an invalid source PEFT inventory")
        elif not isinstance(declared, Mapping) or set(declared) != set(transport):
            raise EvaluationRelocationError(f"current policy publication has invalid {files_key} metadata")
        sizes: dict[str, int] = {}
        for name, entry in transport.items():
            if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), (str, Path)):
                raise EvaluationRelocationError(f"current policy publication has no file path for {name}")
            path = Path(entry["path"])
            if not path.is_file():
                raise EvaluationRelocationError(f"current policy publication file is missing: {name}")
            size = path.stat().st_size
            declared_entry = declared.get(name) if isinstance(declared, Mapping) else None
            if declared_entry is not None and not isinstance(declared_entry, Mapping):
                raise EvaluationRelocationError(f"current policy publication file metadata is invalid: {name}")
            expected_size = (declared_entry.get("size_bytes", declared_entry.get("size"))
                             if isinstance(declared_entry, Mapping) else None)
            transport_size = entry.get("size")
            if ((expected_size is not None and expected_size != size)
                    or (transport_size is not None and transport_size != size)):
                raise EvaluationRelocationError(f"current policy publication file size differs: {name}")
            sizes[name] = size
        return sizes

    files = validate_files("files", "_files")
    source_files = None
    if publication.get("_peft_files") is not None:
        source_files = validate_files("source_peft_files", "_peft_files", declared_optional=True)

    if str(policy_version) not in runtime.catalog:
        return
    try:
        archived = runtime._publication(policy_version)
        archived_snapshot = PolicySnapshot.from_policy(archived)
    except Exception as exc:
        raise EvaluationRelocationError("target catalog has an invalid recovered policy archive") from exc
    if (snapshot != archived_snapshot
            or publication.get("quantization") != archived.get("quantization")
            or publication.get("artifact_id") != archived.get("artifact_id")
            or files != {name: item["size_bytes"] for name, item in archived["files"].items()}):
        raise EvaluationRelocationError("current policy publication conflicts with the target evaluation catalog")
    archived_source_files = archived.get("_peft_files")
    if (source_files is None) != (archived_source_files is None):
        raise EvaluationRelocationError("current policy source-file inventory differs from the target catalog")
    if source_files is not None and source_files != {
            name: item["size"] for name, item in archived_source_files.items()}:
        raise EvaluationRelocationError("current policy source-file sizes differ from the target catalog")


def _copy_evaluation_probe_state(target_state_dir: Path, probe_state_dir: Path, lineage_id: str) -> Path:
    if Path(lineage_id).name != lineage_id or lineage_id in {".", ".."}:
        raise EvaluationRelocationError("managed recovery lineage_id is not a safe path component")
    source = target_state_dir / lineage_id
    probe = probe_state_dir / lineage_id
    probe.mkdir(parents=True)
    if source.exists():
        for name in ("policies.json", "registry.json"):
            original = source / name
            if original.is_symlink() or (original.exists() and not original.is_file()):
                raise EvaluationRelocationError(f"evaluation probe input is not a regular file: {name}")
            if original.exists():
                shutil.copyfile(original, probe / name)
        baseline = source / "baseline-sft"
        if baseline.is_symlink() or (baseline.exists() and not baseline.is_dir()):
            raise EvaluationRelocationError("evaluation baseline probe input must be a regular directory")
        if baseline.exists():
            if any(path.is_symlink() for path in baseline.rglob("*")):
                raise EvaluationRelocationError("evaluation baseline probe input must not contain symlinks")
            shutil.copytree(baseline, probe / "baseline-sft", symlinks=False)
    return probe


def _point_probe_baseline_at_copy(runtime: Any, target_lineage_root: Path, probe_lineage_root: Path) -> None:
    baseline = runtime.catalog.get("0")
    if not isinstance(baseline, dict) or baseline.get("role") != "baseline_sft":
        return
    # Check the actual target archive before rebasing this probe-only catalog row.
    runtime._publication(0)
    # Fresh Native v0 shares the immutable head archive with its baseline.
    # Like the recovered current policy, this validated archive is read-only
    # input to the isolated probe; only the copied catalog/registry are updated.
    if (baseline.get("quantization", {}).get("mode") == "ternary"
            and Path(baseline.get("directory", ""))
            == target_lineage_root / "policies" / "v00000000"):
        return
    target_baseline_root = target_lineage_root / "baseline-sft"
    probe_baseline_root = probe_lineage_root / "baseline-sft"
    try:
        suffix = Path(baseline["directory"]).relative_to(target_baseline_root)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationRelocationError("baseline catalog directory is outside the target baseline archive") from exc
    probe_directory = probe_baseline_root / suffix
    if not probe_directory.is_dir():
        raise EvaluationRelocationError("baseline probe copy is missing the catalogued policy directory")
    baseline["directory"] = str(probe_directory)
    baseline_request_id = runtime.registry._state.get("baseline_request_id")
    if baseline_request_id is not None:
        request = runtime.registry._state["requests"].get(baseline_request_id)
        if not isinstance(request, dict):
            raise EvaluationRelocationError("evaluation registry baseline request is missing")
        archive_ref = request.get("policy_archive_ref")
        if archive_ref:
            try:
                request_suffix = Path(archive_ref).relative_to(target_baseline_root)
            except ValueError as exc:
                raise EvaluationRelocationError(
                    "baseline request archive is outside the target baseline directory"
                ) from exc
            request["policy_archive_ref"] = str(probe_baseline_root / request_suffix)


def relocate_evaluation_references(
    target_state_root: str | Path,
    *,
    path_mappings: Mapping[str | Path, str | Path],
) -> dict[str, Any]:
    """Rebase known persisted evaluation path fields inside an isolated target copy.

    ``target_state_root`` is the extracted target ``evaluation.state_dir`` tree,
    before adding the lineage subdirectory. ``path_mappings`` must be the same
    explicit source-to-target storage roots used by instance migration. Only
    filesystem pointer fields owned by the evaluation schemas are rewritten;
    relative result and artifact references and all policy/content identities
    are retained.
    """
    root = _absolute_path(target_state_root, label="target evaluation state root")
    mappings = _normalize_mappings(path_mappings)
    if not root.is_dir():
        raise EvaluationRelocationError("target evaluation state root must be an existing directory")

    changed_files: list[str] = []
    references_updated = 0

    for path in sorted(root.glob("*/policies.json")):
        before = _load_object(path, label="evaluation policy catalog")
        catalog = json.loads(json.dumps(before))
        for version, record in catalog.items():
            if not isinstance(record, dict):
                raise EvaluationRelocationError(f"evaluation policy catalog entry {version!r} is invalid")
            old = record.get("directory")
            if not isinstance(old, str) or not old.strip():
                raise EvaluationRelocationError(f"evaluation policy {version} has no directory")
            new = _relocate_path(old, mappings, label=f"evaluation policy {version} directory")
            if new != old:
                record["directory"] = new
                references_updated += 1
        if _write_if_changed(path, catalog, before):
            changed_files.append(str(path))

    for path in sorted(root.glob("*/baseline-sft/evaluation-policy.json")):
        before = _load_object(path, label="evaluation baseline provenance")
        if before.get("schema") != "qwen-rl-evaluation-policy-v1":
            raise EvaluationRelocationError("evaluation baseline provenance schema is unsupported")
        provenance = json.loads(json.dumps(before))
        source_policy = provenance.get("source_policy")
        if not isinstance(source_policy, dict):
            raise EvaluationRelocationError("evaluation baseline provenance has no source_policy object")
        old = source_policy.get("policy_archive_ref")
        if not isinstance(old, str) or not old.strip():
            raise EvaluationRelocationError("evaluation baseline provenance has no source policy archive")
        new = _relocate_path(old, mappings, label="evaluation baseline source policy archive")
        if new != old:
            source_policy["policy_archive_ref"] = new
            references_updated += 1
        if _write_if_changed(path, provenance, before):
            changed_files.append(str(path))

    for path in sorted(root.glob("*/registry.json")):
        before = _load_object(path, label="evaluation registry")
        if before.get("schema") != "qwen-rl-evaluation-state-v1":
            raise EvaluationRelocationError("evaluation registry schema is unsupported")
        registry = json.loads(json.dumps(before))

        def relocate_policy_ref(policy: Any, *, label: str) -> None:
            nonlocal references_updated
            if policy is None:
                return
            if not isinstance(policy, dict):
                raise EvaluationRelocationError(f"{label} must be an object or null")
            old = policy.get("policy_archive_ref")
            new = _relocate_path(old, mappings, label=f"{label}.policy_archive_ref")
            if new != old:
                policy["policy_archive_ref"] = new
                references_updated += 1

        relocate_policy_ref(registry.get("current_policy"), label="evaluation registry current_policy")
        requests = registry.get("requests")
        if not isinstance(requests, dict):
            raise EvaluationRelocationError("evaluation registry requests must be an object")
        for request_id, request in requests.items():
            relocate_policy_ref(request, label=f"evaluation request {request_id}")

        events = registry.get("events")
        if not isinstance(events, list):
            raise EvaluationRelocationError("evaluation registry events must be a list")
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise EvaluationRelocationError(f"evaluation registry event {index} must be an object")
            if "policy_archive_ref" in event:
                old = event["policy_archive_ref"]
                new = _relocate_path(old, mappings, label=f"evaluation registry event {index} policy archive")
                if new != old:
                    event["policy_archive_ref"] = new
                    references_updated += 1

        if _write_if_changed(path, registry, before):
            changed_files.append(str(path))

    return {
        "target_state_root": str(root),
        "files_updated": changed_files,
        "references_updated": references_updated,
    }


async def preflight_evaluation_initialization(
    runtime_config: Any,
    *,
    recovery: Mapping[str, Any],
    current_policy: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    training_tasks: Iterable[Any] = (),
) -> dict[str, Any]:
    """Run the production eval dataset/runtime initializer on CPU without generation.

    ``current_policy`` must be a genuine managed-recovery publication. When
    omitted, the initializer loads the publication for the exact recovered
    policy version from the target evaluation catalog and checks its real files.
    No substitute or synthesized policy identity is accepted.
    """
    config = runtime_config.as_dict() if hasattr(runtime_config, "as_dict") else runtime_config
    if not isinstance(config, Mapping):
        raise EvaluationRelocationError("runtime_config must be a runtime mapping or RuntimeConfig")
    evaluation_settings = config.get("evaluation")
    if not isinstance(evaluation_settings, Mapping) or evaluation_settings.get("enabled") is not True:
        return {"enabled": False, "initialized": False, "generation_started": False}
    if not isinstance(recovery, Mapping):
        raise EvaluationRelocationError("recovery must be a managed recovery receipt")
    lineage_id = recovery.get("lineage_id")
    policy_version = recovery.get("policy_version")
    if not isinstance(lineage_id, str) or not lineage_id.strip():
        raise EvaluationRelocationError("managed recovery receipt omitted lineage_id")
    if type(policy_version) is not int or policy_version < 0:
        raise EvaluationRelocationError("managed recovery receipt omitted a valid policy_version")

    try:
        import torch
    except ImportError as exc:
        raise EvaluationRelocationError("CPU evaluation initialization requires torch") from exc
    if torch.cuda.is_initialized():
        raise EvaluationRelocationError("CPU evaluation initialization cannot run after CUDA initialization")

    from .runtime import EvaluationRuntime
    from ..learner.run.evaluation import EvaluationCoordinator

    dataset = EvaluationCoordinator.load_dataset(dict(config))
    if dataset is None:
        raise EvaluationRelocationError("enabled evaluation did not load its frozen taskset")
    EvaluationCoordinator.validate_task_separation(training_tasks, dataset)

    settings = dict(evaluation_settings)
    settings["pause_training"] = True
    target_state_dir = _absolute_path(settings["state_dir"], label="evaluation state_dir")
    if not target_state_dir.parent.is_dir():
        raise EvaluationRelocationError("evaluation state_dir parent must exist for an isolated CPU probe")

    def generation_forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise EvaluationRelocationError("CPU initialization attempted to start evaluation generation")

    with tempfile.TemporaryDirectory(prefix=".evaluation-preflight-", dir=target_state_dir.parent) as probe_root:
        probe_state_dir = Path(probe_root) / "state"
        probe_lineage_root = _copy_evaluation_probe_state(target_state_dir, probe_state_dir, lineage_id)
        settings["state_dir"] = str(probe_state_dir)
        publication_holder: dict[str, Mapping[str, Any]] = {}

        def current_policy_value() -> Mapping[str, Any]:
            if "value" not in publication_holder:
                raise EvaluationRelocationError("CPU evaluation policy was requested before recovery validation")
            return publication_holder["value"]

        runtime = EvaluationRuntime(
            settings=settings,
            dataset=dataset,
            recovery=dict(recovery),
            current_policy=current_policy_value,
            publish_pin=lambda _publication: None,
            release_pin=lambda _version: None,
            run_episode=generation_forbidden,
        )
        if current_policy is None:
            try:
                publication = runtime._publication(policy_version)
            except Exception as exc:
                raise EvaluationRelocationError(
                    "recovered current policy is absent or invalid in the target evaluation catalog"
                ) from exc
        else:
            publication = current_policy() if callable(current_policy) else current_policy
        if not isinstance(publication, Mapping):
            raise EvaluationRelocationError("current policy publication must be a managed publication mapping")
        _validate_policy_publication(runtime, publication, policy_version=policy_version)
        _point_probe_baseline_at_copy(runtime, target_state_dir / lineage_id, probe_lineage_root)
        publication_holder["value"] = dict(publication)
        try:
            await runtime.initialize()
            state = runtime.registry.snapshot()
            if torch.cuda.is_initialized():
                raise EvaluationRelocationError("evaluation initialization unexpectedly initialized CUDA")
            return {
                "enabled": True,
                "initialized": True,
                "generation_started": False,
                "probe_isolated": True,
                "lineage_id": lineage_id,
                "policy_version": policy_version,
                "evalset_sha256": dataset.manifest["manifest_sha256"],
                "task_count": len(dataset.tasks),
                "evaluation_state_root": str(target_state_dir / lineage_id),
                "baseline_request_id": state.get("baseline_request_id"),
                "catalog_policy_versions": sorted(int(key) for key in runtime.catalog),
            }
        finally:
            await runtime.close()
