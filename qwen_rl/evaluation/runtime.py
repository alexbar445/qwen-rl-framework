"""Learner-owned evaluation lifecycle; generation is an injected native callback."""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from pathlib import Path
from typing import Any

from ..rollout.groups import PolicySnapshot
from ..runtime.manifest import read_json, write_json_atomic
from .policy import EvaluationPolicy, policy_quantization_mode
from .runner import run_evaluation_request
from .state import EvaluationRegistry

_PEFT_FILES = {"adapter_config.json", "adapter_model.safetensors"}
_NATIVE_T2_FILES = {"policy.lora.ninfer", "policy.lora.ninfer.json"}


def _proof_file_sha256(path: Path) -> str:
    """Retain the independent audit-proof receipt; policy payloads are size-checked only."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def native_episode_settings(rollout: dict, contract: Any) -> dict:
    """Capture actual production settings and prompt/runtime contract fingerprints."""
    from ..contracts.tool_call_failures import (
        ENV_SCHEMA_VERSION, MAX_TOOL_CALL_FAILURES, TOOL_CALL_FAILURE_COUNTING,
    )

    settings = {key: rollout[key] for key in
        ('temperature', 'top_p', 'top_k', 'repetition_penalty', 'turn_max_tokens',
         'max_turns', 'response_length', 'prompt_length')}
    conditioned = rollout.get('constrained_decoding', False)
    if type(conditioned) is not bool:
        raise ValueError('evaluation constrained_decoding must be boolean')
    settings['policy_distribution'] = (
        'grammar_conditioned_policy/v1' if conditioned else 'raw_policy'
    )
    if conditioned:
        from ..prompt import load_toolkit_runtime
        from ..contracts.grammar_compiler import native_grammar_compiler_sha256
        load_toolkit_runtime()
        from arc_toolkit.constrained_actions import constraint_receipt
        settings.update(grammar_sha256=constraint_receipt(thinking=True)['sha256'],
                        grammar_compiler_sha256=native_grammar_compiler_sha256())
    payload = json.dumps(contract.as_dict(), sort_keys=True, separators=(',', ':')).encode()
    from ..prompt import prompt_contract, _action_language
    from ..rewards import REWARD_SCHEMA

    prompt_payload = json.dumps(
        prompt_contract(),
        ensure_ascii=False, sort_keys=True, separators=(',', ':')
    ).encode("utf-8")
    settings.update(reward_schema=REWARD_SCHEMA, arc_env_schema=ENV_SCHEMA_VERSION,
                    max_tool_call_failures=MAX_TOOL_CALL_FAILURES,
                    tool_call_failure_counting=TOOL_CALL_FAILURE_COUNTING,
                    enable_thinking=True, tool_protocol='qwen3_coder',
                    tool_names=['arc_action'], token_budget=contract.token_budget,
                    contract_fingerprint=hashlib.sha256(payload).hexdigest(),
                    prompt_fingerprint=hashlib.sha256(prompt_payload).hexdigest(),
                    action_language_fingerprint=_action_language()["fingerprint"])
    if contract.generated_token_budget is not None:
        settings.update(generated_token_budget=contract.generated_token_budget,
                        observation_token_budget=contract.observation_token_budget)
    return settings


class EvaluationRuntime:
    def __init__(self, *, settings: dict, dataset: Any, recovery: dict,
                 current_policy: Callable[[], dict], publish_pin: Callable[[dict], None],
                 release_pin: Callable[[int], None], run_episode: Callable,
                 evaluation_scope: Callable[[EvaluationPolicy], AbstractAsyncContextManager[None]] | None = None):
        self.settings, self.dataset, self.recovery = settings, dataset, recovery
        self.pause_training = settings.get('pause_training') is True
        self._gate_open = not self.pause_training
        self.current_policy, self.publish_pin, self.release_pin = current_policy, publish_pin, release_pin
        self.run_episode = run_episode
        self.evaluation_scope = evaluation_scope
        self.root = Path(settings['state_dir']) / recovery['lineage_id']
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / 'policies.json'
        self.catalog = read_json(self.catalog_path) if self.catalog_path.exists() else {}
        self.registry = EvaluationRegistry(self.root / 'registry.json',
            lineage_id=recovery['lineage_id'], evalset_sha256=dataset.manifest['manifest_sha256'],
            task_ids=[task.task_id for task in dataset.tasks], milestone_interval=settings['every_updates'])
        self.pins: set[int] = set()
        self.error: dict | None = None
        self.worker: asyncio.Task | None = None
        self.wake = asyncio.Event()
        self.closed = False
        self._count: int | None = None

    def _publication(self, version: int) -> dict:
        item = self.catalog[str(version)]
        directory = Path(item['directory'])
        native_t2 = policy_quantization_mode(item.get('quantization')) == 'ternary'
        expected_names = _NATIVE_T2_FILES if native_t2 else _PEFT_FILES
        if set(item.get('files', {})) != expected_names:
            if native_t2:
                raise RuntimeError('T2 evaluation catalog is PEFT-only or has an invalid native transport pair')
            raise RuntimeError('evaluation catalog has an invalid PEFT transport pair')
        source_adapter_sha256 = item.get('source_adapter_sha256')
        if (not isinstance(source_adapter_sha256, str) or len(source_adapter_sha256) != 64
                or any(char not in '0123456789abcdefABCDEF' for char in source_adapter_sha256)):
            raise RuntimeError('evaluation catalog omitted source_adapter_sha256')
        files = {}
        for name, metadata in item['files'].items():
            path = directory / name
            if not path.is_file():
                raise RuntimeError(f'evaluation policy archive is missing: {version}/{name}')
            expected_size = metadata.get('size') if isinstance(metadata, dict) else None
            size = path.stat().st_size
            if expected_size is not None and expected_size != size:
                raise RuntimeError(f'evaluation policy archive size mismatch: {version}/{name}')
            files[name] = {'path': path, 'sha256': None, 'size': size}
        publication = {'version': version, 'base_fingerprint': item['base_fingerprint'],
                       'artifact_fingerprint': item['artifact_fingerprint'],
                       'source_adapter_sha256': source_adapter_sha256.lower(),
                       'quantization': item['quantization'],
                       'files': {name: {'sha256': None, 'size_bytes': entry['size']}
                                 for name, entry in files.items()},
                       '_files': files}
        if native_t2:
            descriptor = read_json(directory / 'policy.lora.ninfer.json')
            artifact_id = descriptor.get('artifact_id')
            if not isinstance(artifact_id, str) or len(artifact_id) != 32:
                raise RuntimeError('T2 evaluation policy omitted its native artifact ID')
            publication['artifact_id'] = artifact_id
        if native_t2 and item.get('source_peft_files') is not None:
            source_peft_files = item['source_peft_files']
            if not isinstance(source_peft_files, dict) or set(source_peft_files) != _PEFT_FILES:
                raise RuntimeError('T2 evaluation catalog has an invalid source PEFT file inventory')
            peft_files = {}
            for name, metadata in source_peft_files.items():
                path = directory / name
                if not path.is_file():
                    raise RuntimeError(f'T2 evaluation source PEFT file is missing: {name}')
                expected_size = metadata.get('size') if isinstance(metadata, dict) else None
                size = path.stat().st_size
                if expected_size is not None and expected_size != size:
                    raise RuntimeError(f'T2 evaluation source PEFT file size mismatch: {name}')
                peft_files[name] = {'path': path, 'sha256': None, 'size': size}
            publication['_peft_files'] = peft_files
        return publication

    def archive(self, publication: dict, *, role: str = 'rl_checkpoint') -> dict:
        native_t2 = policy_quantization_mode(publication.get('quantization')) == 'ternary'
        version = int(publication['version'])
        directory = self.root / 'policies' / f'v{version:08d}'
        directory.mkdir(parents=True, exist_ok=True)
        expected_names = _NATIVE_T2_FILES if native_t2 else _PEFT_FILES
        transport_files = publication.get('_files')
        if not isinstance(transport_files, dict) or set(transport_files) != expected_names:
            if native_t2:
                raise RuntimeError('T2 evaluation publication must contain exactly the native transport pair')
            raise RuntimeError('evaluation publication must contain exactly the PEFT transport pair')
        existing = self.catalog.get(str(version))
        if native_t2 and isinstance(existing, dict) \
                and set(existing.get('files', {})) != _NATIVE_T2_FILES:
            raise RuntimeError('persisted T2 evaluation catalog is PEFT-only; refusing legacy file mapping')
        file_records = {}
        for name, entry in transport_files.items():
            source = entry.get('path') if isinstance(entry, dict) else None
            if not isinstance(source, (str, Path)) or not Path(source).is_file():
                raise RuntimeError(f'evaluation policy file is missing: {name}')
            source = Path(source)
            source_size = source.stat().st_size
            expected_size = entry.get('size')
            if expected_size is not None and expected_size != source_size:
                raise RuntimeError(f'evaluation policy file size differs from its publication: {name}')
            destination = directory / name
            if not destination.exists():
                tmp = directory / ('.' + name + '.tmp')
                shutil.copyfile(source, tmp)
                tmp.replace(destination)
            if not destination.is_file() or destination.stat().st_size != source_size:
                raise RuntimeError(f'immutable evaluation policy file size differs from its publication: {name}')
            file_records[name] = {'sha256': None, 'size': destination.stat().st_size}
        source_peft_records = None
        snapshot = PolicySnapshot.from_policy(publication)
        source_adapter_sha256 = snapshot.adapter_sha256
        if native_t2:
            source_peft_files = publication.get('_peft_files')
            if not isinstance(source_peft_files, dict) or set(source_peft_files) != _PEFT_FILES:
                raise RuntimeError('T2 evaluation archive requires Learner-exported source PEFT files')
            source_peft_records = {}
            for name, entry in source_peft_files.items():
                if not isinstance(entry, dict) or not isinstance(entry.get('path'), (str, Path)):
                    raise RuntimeError(f'T2 evaluation source PEFT entry is invalid: {name}')
                source = Path(entry['path'])
                if not source.is_file():
                    raise RuntimeError(f'T2 evaluation source PEFT file is missing: {name}')
                source_size = source.stat().st_size
                expected_size = entry.get('size')
                if expected_size is not None and expected_size != source_size:
                    raise RuntimeError(f'T2 evaluation source PEFT file size differs from its publication: {name}')
                destination = directory / name
                if not destination.exists():
                    tmp = directory / ('.' + name + '.tmp')
                    shutil.copyfile(source, tmp)
                    tmp.replace(destination)
                if not destination.is_file() or destination.stat().st_size != source_size:
                    raise RuntimeError(f'immutable T2 evaluation source PEFT file size differs: {name}')
                source_peft_records[name] = {'sha256': None, 'size': destination.stat().st_size}
        record = {'role': role, 'directory': str(directory), 'files': file_records,
                  'source_adapter_sha256': source_adapter_sha256,
                  'base_fingerprint': publication['base_fingerprint'],
                  'artifact_fingerprint': publication['artifact_fingerprint'],
                  'quantization': publication['quantization']}
        if native_t2:
            record['source_peft_files'] = source_peft_records
        if existing is not None and existing != record:
            raise RuntimeError('evaluation archive version identity changed')
        self.catalog[str(version)] = record
        write_json_atomic(self.catalog_path, self.catalog)
        return record

    def _initial_count(self) -> tuple[int, str]:
        context = self.recovery.get('update_context') or {}
        value = context.get('lineage_successful_training_updates')
        if type(value) is int and value >= 0:
            return value, 'verified_full_checkpoint_update_context'
        if self.recovery.get('policy_version') == 0:
            return 0, 'fresh_policy_before_any_optimizer_update'
        path = self.settings.get('initial_count_proof')
        if not path:
            raise RuntimeError('existing lineage needs an audited successful-training count proof')
        proof = read_json(Path(path))
        for key in ('lineage_id', 'engine_epoch', 'policy_version'):
            if proof.get(key) != self.recovery.get(key):
                raise RuntimeError(f'initial evaluation count proof differs from checkpoint: {key}')
        loaded = self.recovery.get('loaded_snapshot') or {}
        expected_sha256 = loaded.get('snapshot_sha256')
        if expected_sha256 is not None:
            if proof.get('snapshot_sha256') != expected_sha256:
                raise RuntimeError('initial evaluation count proof snapshot SHA differs')
        else:
            snapshot_id = self.recovery.get('snapshot_id') or loaded.get('snapshot_id')
            snapshot_path = self.recovery.get('snapshot_path') or loaded.get('path')
            if not isinstance(snapshot_id, str) or not snapshot_id.strip():
                raise RuntimeError('initial evaluation count proof checkpoint snapshot ID is unavailable')
            if not isinstance(snapshot_path, (str, Path)) or not str(snapshot_path).strip():
                raise RuntimeError('initial evaluation count proof checkpoint path is unavailable')
            checkpoint = Path(snapshot_path)
            if checkpoint.suffix == '.json':
                pointer = read_json(checkpoint)
                name = pointer.get('snapshot')
                if (pointer.get('schema') != 'central-gradient-checkpoint-pointer/v1'
                        or pointer.get('snapshot_id') != snapshot_id
                        or pointer.get('policy_version') != self.recovery.get('policy_version')
                        or not isinstance(name, str) or not name
                        or Path(name).name != name):
                    raise RuntimeError('initial evaluation count proof checkpoint pointer identity differs')
                checkpoint = checkpoint.parent / name
                declared_size = pointer.get('size_bytes')
                if type(declared_size) is not int or declared_size != proof.get('snapshot_size_bytes'):
                    raise RuntimeError('initial evaluation count proof snapshot size differs from pointer')
            try:
                if checkpoint.is_symlink() or not checkpoint.is_file():
                    raise RuntimeError('initial evaluation count proof checkpoint is not a regular file')
                snapshot_size = checkpoint.stat().st_size
            except OSError:
                raise RuntimeError('initial evaluation count proof checkpoint cannot be inspected') from None
            if proof.get('snapshot_id') != snapshot_id:
                raise RuntimeError('initial evaluation count proof snapshot ID differs')
            proof_size = proof.get('snapshot_size_bytes')
            if type(proof_size) is not int or proof_size <= 0 or proof_size != snapshot_size:
                raise RuntimeError('initial evaluation count proof snapshot size differs')
        count = proof.get('successful_training_updates')
        if type(count) is not int or count < 0:
            raise RuntimeError('initial evaluation count proof has no valid count')
        return count, 'audited_training_lineage_proof:' + _proof_file_sha256(Path(path))

    async def initialize(self) -> None:
        self.registry.mark_interrupted()
        self.registry.disable_initial_baseline()
        count, source = self._initial_count()
        current = self.current_policy()
        previous = self.registry.snapshot()['lineage_successful_update_count']
        if previous is None and count > 0 and count % self.settings['every_updates'] == 0:
            self.registry.initialize_lineage_count(0, source='start_tracking_before_audited_updates:' + source)
            previous = 0
        if previous is not None and count > previous and count % self.settings['every_updates'] == 0:
            archive_ref = self.archive(current)['directory']
            self.registry.observe_training_update(count, policy_version=current['version'],
                adapter_sha256=PolicySnapshot.from_policy(current).adapter_sha256,
                policy_archive_ref=archive_ref)
        else:
            self.registry.initialize_lineage_count(count, source=source)
        self._count = count
        self.refresh_pins()
        if not self.pause_training:
            self._start_worker()

    def _start_worker(self) -> None:
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._loop(), name='native-capability-evaluation')
        self.wake.set()

    @property
    def count(self) -> int:
        if self._count is None:
            raise RuntimeError('evaluation training count has not been initialized')
        return self._count

    def observe_commit(self, publication: dict | None) -> None:
        count = self.count + 1
        self._count = count
        try:
            if publication is None:
                raise RuntimeError('committed policy publication unavailable for evaluation')
            archive_ref = None
            if count % self.settings['every_updates'] == 0:
                archive_ref = self.archive(publication)['directory']
            self.registry.observe_training_update(count, policy_version=publication['version'],
                adapter_sha256=PolicySnapshot.from_policy(publication).adapter_sha256,
                policy_archive_ref=archive_ref)
            self.refresh_pins()
            self.wake.set()
        except Exception as exc:
            # Training durability precedes evaluation bookkeeping. The full
            # checkpoint carries count; a resumed runtime reconciles it.
            self.error = {'type': type(exc).__name__, 'message': str(exc)[:500],
                          'stage': 'observe_committed_training_update', 'successful_update_count': count}

    def retry_errors(self, request_id: str) -> dict:
        request = self.registry.retry_errors(request_id)
        self.error = None
        self.refresh_pins()
        if not self.pause_training:
            self._start_worker()
        else:
            self.wake.set()
        return request

    def request_current(self) -> dict:
        current = self.current_policy()
        archive_ref = (self.catalog['0']['directory'] if current['version'] == 0
                       and self.catalog.get('0', {}).get('role') == 'baseline_sft'
                       else self.archive(current)['directory'])
        request = self.registry.request_current_policy(self.count, policy_version=current['version'],
            adapter_sha256=PolicySnapshot.from_policy(current).adapter_sha256,
            policy_archive_ref=archive_ref)
        self.error = None
        self.refresh_pins()
        if not self.pause_training:
            self._start_worker()
        else:
            self.wake.set()
        return request

    def refresh_pins(self) -> None:
        state = self.registry.snapshot()
        wanted = {int(item['policy_version']) for item in state['requests'].values()
                  if item['status'] in {'pending', 'running', 'interrupted', 'cancelled'}}
        for version in wanted - self.pins:
            self.publish_pin(self._publication(version))
        for version in self.pins - wanted:
            self.release_pin(version)
        self.pins = wanted

    def has_pending_requests(self) -> bool:
        requests = self.registry.snapshot()['requests'].values()
        return any(item.get('status') in {'pending', 'running', 'interrupted', 'cancelled'}
                   for item in requests if isinstance(item, dict))

    async def run_pending_at_boundary(self) -> None:
        if not self.pause_training or not self.has_pending_requests():
            return
        self._gate_open = True
        self._start_worker()
        try:
            while self.has_pending_requests():
                if self.worker is not None and self.worker.done():
                    message = self.error.get('message', 'worker stopped with queued requests') if self.error else 'worker stopped with queued requests'
                    raise RuntimeError(f"native evaluation failed: {message}")
                await asyncio.sleep(0.05)
            if self.error is not None:
                raise RuntimeError(f"native evaluation failed: {self.error.get('message', 'unknown error')}")
        finally:
            self._gate_open = False
            self.wake.set()

    def monitor(self) -> dict:
        return {**self.registry.monitor_snapshot(), 'runtime_error': self.error,
                'baseline_semantics': 'same frozen base; zero residual LoRA',
                'dataset_scope': self.dataset.manifest.get('dataset_name', 'unavailable')}

    async def _loop(self) -> None:
        request = None
        try:
            while not self.closed:
                if self.pause_training and not self._gate_open:
                    self.wake.clear()
                    if not self._gate_open:
                        await self.wake.wait()
                        continue
                request = self.registry.claim_next()
                if request is None:
                    self.wake.clear()
                    try:
                        await asyncio.wait_for(self.wake.wait(), 0.5)
                    except TimeoutError:
                        pass
                    continue
                self.error = None
                version = int(request['policy_version'])
                record = self.catalog[str(version)]
                publication = self._publication(version)
                policy = EvaluationPolicy(record['role'], PolicySnapshot.from_policy(publication),
                    'zero_residual_sft_lora' if record['role'] == 'baseline_sft' else 'published_rl_checkpoint', record['directory'])
                state = self.registry.snapshot()
                comparisons = {key: {task_id: task['summary'] for task_id, task in item['tasks'].items()
                                      if task['status'] == 'completed'}
                               for key, item in state['requests'].items() if item['status'] == 'completed'}
                request_id = request['request_id']
                def started(task_id, attempt_id, request_id=request_id):
                    self.registry.record_task_started(request_id, task_id, attempt_id=attempt_id)
                def completed(row, request_id=request_id):
                    summary = row.summary()
                    self.registry.record_task_result(request_id, row.task_id,
                        row.artifact_ref, row.artifact_sha256, summary)
                scope = self.evaluation_scope(policy) if self.evaluation_scope is not None else nullcontext()
                async with scope:
                    receipt = await run_evaluation_request(request, self.dataset, policy, self.run_episode,
                        self.root / 'receipts', comparison_rows=comparisons,
                        concurrency=self.settings.get('concurrency', 1),
                        on_task_started=started, on_task_result=completed)
                    self.registry.complete(request['request_id'], receipt.artifact_ref,
                                           receipt.artifact_sha256, receipt.summary)
                    self.refresh_pins()
                    request = None
        except asyncio.CancelledError:
            if request is not None:
                self.registry.mark_cancelled(request['request_id'], 'runtime_stopped')
            raise
        except Exception as exc:
            self.error = {'type': type(exc).__name__, 'message': str(exc)[:500]}
            if request is not None:
                self.registry.mark_cancelled(request['request_id'], 'evaluation_failed_no_automatic_retry')

    async def close(self) -> None:
        self.closed = True
        if self.worker is not None and not self.worker.done():
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        for version in self.pins:
            self.release_pin(version)
        self.pins.clear()
