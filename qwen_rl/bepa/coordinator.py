"""CPU BEPA control plane; reuse the ARC runtime and the existing student loop."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import random
import time
import threading
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Sequence
import uuid

from .artifacts import BepaArtifacts, project_source_episode, replay_row_from_artifacts
from .contracts import (TaskSpec as BepaTask, SourcePolicy, SourceEpisode, TrustedReceipt,
                        content_hash, BudgetUnavailableError, ConflictError, ContractError,
                        LeaseLostError, validate_teacher_plan)
from .reset_receipt import read_checkpoint_reset_receipt, write_checkpoint_reset_receipt
from .store import BepaStore
from .token_usage import raw_completion_tokens, sum_uncached_input_tokens, uncached_input_tokens
from ..budget import load_contract
from ..environment import ARCToolEnv, DEFAULT_TOOLKIT
from ..prompt import action_tools, build_task_messages, prompt_identity
from ..rewards import REWARD_SCHEMA, VERIFICATION_COEFFICIENT, group_rewards
from ..contracts.arc_tools import is_model_stop, tool_schema_error, tool_schema_issue, turn_rejection_reason
from ..contracts.episode_budget import (BUDGET_POLICY_ID, budget_policy_identity,
                                       context_generation_limit, plan_generation, replay_response_offsets)
from ..contracts.worker import MAX_CONTEXT_TOKENS
from ..contracts.tool_call_failures import (ENV_SCHEMA_VERSION, MAX_TOOL_CALL_FAILURES,
                                           tool_call_failure_feedback)
from ..service import SCHEMA_VERSION, TurnRequest, TurnResponse, generated_group_rewards


def _toolkit_semantics_fingerprint(toolkit_root: str | Path) -> str:
    """Hash the vendored Python package and its runtime resources deterministically."""
    root = Path(toolkit_root)
    package = root / 'arc_toolkit'
    if not package.is_dir():
        raise FileNotFoundError(f'ARC toolkit package is missing: {package}')
    files: dict[str, str] = {}
    for path in sorted(package.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in {'.pyc', '.pyo'}:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if 'arc_toolkit/runtime.py' not in files:
        raise FileNotFoundError(f'ARC toolkit runtime is missing from {package}')
    metadata: dict[str, str] = {}
    for name in ('pyproject.toml',):
        path = root / name
        if path.is_file():
            metadata[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return content_hash({
        'distribution': 'paired-factory-v1',
        'files': files,
        'metadata': metadata,
    })


@dataclass
class TeacherSession:
    job_id: str
    fence: int
    task_id: str
    env: Any
    messages: list[dict]
    events: list[dict] = field(default_factory=list)
    replies: dict[int, tuple[str, dict]] = field(default_factory=dict)
    event_artifact_refs: list[str] = field(default_factory=list)
    confirmed_completion_tokens: int = 0
    receipt_ref: str | None = None
    source_hash: str | None = None
    accepted: bool = False
    finished: bool = False
    finish_hash: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class TrainingBatch:
    batch_id: str
    source: Any
    rows: list[dict]
    fresh_rows: list[dict]
    fresh_worker_ids: list[str]
    reservation_ids: tuple[str, ...]


class BepaCoordinator:
    def __init__(self, config, engine, tasks):
        self.config, self.engine, self.tasks = config, engine, tasks
        self.settings = config['bepa']
        recovery_metadata = getattr(engine, 'recovery_metadata', {})
        self.recovery_metadata = recovery_metadata if isinstance(recovery_metadata, dict) else {}
        engine_epoch = self.recovery_metadata.get('engine_epoch')
        self.epoch = engine_epoch if isinstance(engine_epoch, str) and engine_epoch else uuid.uuid4().hex
        root = Path(self.settings.get('state_dir') or (Path(config['run_dir']) / 'bepa'))
        self.root = root
        self.artifacts = BepaArtifacts(root)
        self.store = BepaStore(root,
            max_age_updates=int(self.settings['cache']['max_age_updates']),
            max_committed_uses=int(self.settings['cache']['max_committed_uses']))
        # Checkpoint-backed update reconciliation must happen after the Learner
        # publishes its restored adapter SHA. The app calls recover_startup()
        # before it becomes ready; generic recovery is deliberately deferred
        # until that verified checkpoint evidence is available.
        self.startup_recovery: dict[str, Any] | None = None
        budget = self.settings['teacher']['budget']
        self.scope = self.settings.get('budget_scope', 'default')
        scope_limits = self.store.configure_budget_scope(
            self.scope, max_total_tokens=int(budget['max_total_tokens']))
        budget['max_total_tokens'] = int(scope_limits['max_total_tokens'])
        persisted_job_cap = self.store.latest_max_job_tokens(self.scope)
        if persisted_job_cap is not None:
            budget['max_job_tokens'] = persisted_job_cap
        self.contract = load_contract(config['arc']['contract_path'])
        self.tokenizer = engine.tokenizer
        self.tokenizer_hash = content_hash({'template': self.tokenizer.chat_template,
                                          'vocab': self.tokenizer.get_vocab()})
        self.tool_hash = content_hash({'toolkit': _toolkit_semantics_fingerprint(DEFAULT_TOOLKIT),
                                      'tools': action_tools()})
        self.task_keys: dict[str, str] = {}
        self._registered_task_guards: dict[str, str] = {}
        self._registered_context_guard: str | None = None
        self.keys_to_tasks: dict[str, str] = {}
        self.sessions: dict[str, TeacherSession] = {}
        self._claim_lock = threading.RLock()
        self.teacher_last_seen: float | None = None
        self.operation: dict | None = None
        self.operation_task: asyncio.Task | None = None
        self.mode = 'idle'

    def write_checkpoint_reset_receipt(self, *, policy_version: int,
                                       adapter_sha256: str) -> dict[str, Any]:
        """Authorize the exact restored checkpoint against a freshly reset BEPA ledger."""
        if self.startup_recovery is not None:
            raise RuntimeError('BEPA checkpoint reset receipt must be written before startup recovery')
        recovery = getattr(self.engine, 'recovery_metadata', self.recovery_metadata)
        return write_checkpoint_reset_receipt(
            self.store,
            recovery=recovery,
            policy_version=policy_version,
            adapter_sha256=adapter_sha256,
            target_budget_scope=self.scope,
        )

    def _matching_checkpoint_reset_receipt(
        self, recovery: Mapping[str, Any], *, policy_version: int, adapter_sha256: str,
    ) -> dict[str, Any] | None:
        scope = getattr(self, 'scope', None)
        if not isinstance(scope, str) or not scope:
            return None
        return read_checkpoint_reset_receipt(
            self.store,
            recovery=recovery,
            policy_version=policy_version,
            adapter_sha256=adapter_sha256,
            target_budget_scope=scope,
        )

    def reconcile_checkpoint_attempt(self, *, policy_version: int,
                                      adapter_sha256: str) -> str | None:
        """Commit one BEPA attempt only when current engine metadata proves its snapshot."""
        metadata = getattr(self.engine, 'recovery_metadata', self.recovery_metadata)
        if not isinstance(metadata, dict):
            return None
        if metadata.get('policy_version') != policy_version:
            return None
        snapshot_id = metadata.get('snapshot_id')
        loaded = metadata.get('loaded_snapshot')
        if not isinstance(snapshot_id, str) and isinstance(loaded, dict):
            snapshot_id = loaded.get('snapshot_id')
        if (not isinstance(snapshot_id, str) or len(snapshot_id) != 32
                or any(char not in '0123456789abcdef' for char in snapshot_id)):
            return None
        context = metadata.get('update_context')
        if not isinstance(context, dict):
            return None
        attempt_id = context.get('bepa_attempt_id')
        batch_id = context.get('bepa_batch_id')
        input_batch_hash = context.get('input_batch_hash')
        if attempt_id is None:
            return None
        if not all(isinstance(value, str) and value for value in
                   (attempt_id, batch_id, input_batch_hash)):
            raise RuntimeError('recovery context has an incomplete BEPA attempt identity')
        attempt = self.store.get_update_attempt(attempt_id)
        if attempt is None:
            if self._matching_checkpoint_reset_receipt(
                    metadata, policy_version=policy_version,
                    adapter_sha256=adapter_sha256) is not None:
                return None
            raise RuntimeError('verified Learner checkpoint names a missing BEPA update attempt')
        if (attempt.engine_epoch != self.epoch or attempt.batch_id != batch_id
                or attempt.input_batch_hash != input_batch_hash):
            raise RuntimeError('checkpoint BEPA attempt identity does not match the durable ledger')
        if (context.get('new_policy_version') != policy_version
                or policy_version != attempt.start_policy_version + 1):
            raise RuntimeError('checkpoint version does not prove the optimizer-entered BEPA attempt')
        logical_job = context.get('logical_job')
        if (not isinstance(logical_job, dict)
                or isinstance(logical_job.get('completed_updates'), bool)
                or not isinstance(logical_job.get('completed_updates'), int)
                or logical_job['completed_updates'] < 1):
            raise RuntimeError('checkpoint context does not prove the BEPA logical update count')
        if attempt.state in {'optimizer_entered', 'commit_unknown'}:
            self.store.commit_update(
                attempt_id,
                new_policy_version=policy_version,
                adapter_sha256=adapter_sha256,
                commit_evidence={
                    'recovered_from_full_snapshot': True,
                    'snapshot_id': snapshot_id,
                    'checkpoint_policy_version': policy_version,
                    'input_batch_hash': input_batch_hash,
                    'lineage_id': metadata.get('lineage_id'),
                },
            )
            return attempt_id
        if attempt.state == 'committed':
            if attempt.new_policy_version != policy_version or attempt.adapter_sha256 != adapter_sha256:
                raise RuntimeError('checkpoint and committed BEPA attempt disagree on version or adapter SHA')
            return attempt_id
        raise RuntimeError(f'checkpoint names BEPA attempt in impossible recovery state {attempt.state!r}')

    def recover_startup(self, *, policy_version: int, adapter_sha256: str) -> dict[str, Any]:
        """Reconcile BEPA's optimizer ledger against one verified engine snapshot.

        The full learner snapshot is the proof that an optimizer-entered attempt
        reached its publication version. This is run before generic restart
        handling can quarantine that attempt as unknown.
        """
        if self.startup_recovery is not None:
            return self.startup_recovery
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise ContractError('recovered policy_version must be a non-negative integer')
        if (not isinstance(adapter_sha256, str) or len(adapter_sha256) != 64
                or any(char not in '0123456789abcdef' for char in adapter_sha256)):
            raise ContractError('recovered adapter SHA-256 is invalid')

        metadata = self.recovery_metadata
        resumed = metadata.get('resumed') is True
        loaded = metadata.get('loaded_snapshot')
        if resumed:
            if not isinstance(loaded, dict):
                raise RuntimeError('resumed Learner has no verified full-snapshot metadata')
            if loaded.get('policy_version') != policy_version:
                raise RuntimeError('Learner policy version differs from its verified recovery snapshot')
            snapshot_id = loaded.get('snapshot_id')
            if (not isinstance(snapshot_id, str) or len(snapshot_id) != 32
                    or any(char not in '0123456789abcdef' for char in snapshot_id)):
                raise RuntimeError('recovery snapshot does not expose a snapshot ID')
        else:
            snapshot_id = None

        latest = self.store.latest_committed_update(engine_epoch=self.epoch)
        if latest is not None and latest.new_policy_version is not None:
            if latest.new_policy_version > policy_version:
                raise RuntimeError(
                    'BEPA committed-update ledger is ahead of the verified Learner checkpoint '
                    f'({latest.new_policy_version} > {policy_version})'
                )
            if (latest.new_policy_version == policy_version
                    and latest.adapter_sha256 != adapter_sha256):
                raise RuntimeError('BEPA committed adapter SHA differs from the verified Learner checkpoint')

        reconciled_attempt_id = None
        reset_receipt_applied = None
        context = metadata.get('update_context')
        if resumed and isinstance(context, dict):
            attempt_id = context.get('bepa_attempt_id')
            batch_id = context.get('bepa_batch_id')
            input_batch_hash = context.get('input_batch_hash')
            if attempt_id is not None:
                if not all(isinstance(value, str) and value for value in
                           (attempt_id, batch_id, input_batch_hash)):
                    raise RuntimeError('recovery context has an incomplete BEPA attempt identity')
                attempt = self.store.get_update_attempt(attempt_id)
                if attempt is None:
                    reset_receipt = self._matching_checkpoint_reset_receipt(
                        metadata, policy_version=policy_version,
                        adapter_sha256=adapter_sha256,
                    )
                    if reset_receipt is None:
                        raise RuntimeError('verified Learner checkpoint names a missing BEPA update attempt')
                    reset_receipt_applied = {
                        'attempt_id': attempt_id,
                        'snapshot_id': snapshot_id,
                        'target_budget_scope': self.scope,
                    }
                else:
                    if (attempt.engine_epoch != self.epoch or attempt.batch_id != batch_id
                            or attempt.input_batch_hash != input_batch_hash):
                        raise RuntimeError('checkpoint BEPA attempt identity does not match the durable ledger')
                    if attempt.state in {'optimizer_entered', 'commit_unknown'}:
                        if (policy_version != attempt.start_policy_version + 1
                                or context.get('new_policy_version') != policy_version):
                            raise RuntimeError('checkpoint version does not prove the optimizer-entered BEPA attempt')
                        logical_job = context.get('logical_job')
                        if (not isinstance(logical_job, dict)
                                or isinstance(logical_job.get('completed_updates'), bool)
                                or not isinstance(logical_job.get('completed_updates'), int)
                                or logical_job['completed_updates'] < 1):
                            raise RuntimeError('checkpoint context does not prove the BEPA logical update count')
                        self.store.commit_update(
                            attempt_id,
                            new_policy_version=policy_version,
                            adapter_sha256=adapter_sha256,
                            commit_evidence={
                                'recovered_from_full_snapshot': True,
                                'snapshot_id': snapshot_id,
                                'checkpoint_policy_version': policy_version,
                                'input_batch_hash': input_batch_hash,
                                'lineage_id': metadata.get('lineage_id'),
                            },
                        )
                        reconciled_attempt_id = attempt_id
                    elif attempt.state == 'committed':
                        if attempt.new_policy_version != policy_version or attempt.adapter_sha256 != adapter_sha256:
                            raise RuntimeError('checkpoint and committed BEPA attempt disagree on version or adapter SHA')
                        reconciled_attempt_id = attempt_id
                    else:
                        raise RuntimeError(
                            f'checkpoint names BEPA attempt in impossible recovery state {attempt.state!r}'
                        )

        generic = self.store.recover_interrupted_updates()
        latest = self.store.latest_committed_update(engine_epoch=self.epoch)
        if latest is not None and latest.new_policy_version is not None:
            if latest.new_policy_version > policy_version:
                raise RuntimeError('BEPA committed-update ledger remains ahead after checkpoint reconciliation')
            if latest.new_policy_version == policy_version and latest.adapter_sha256 != adapter_sha256:
                raise RuntimeError('BEPA committed adapter SHA differs from the verified Learner checkpoint')
        self.startup_recovery = {
            'resumed': resumed,
            'engine_epoch': self.epoch,
            'policy_version': policy_version,
            'adapter_sha256': adapter_sha256,
            'reconciled_attempt_id': reconciled_attempt_id,
            'checkpoint_reset_receipt': reset_receipt_applied,
            'generic_recovery': generic,
            'latest_committed_update': (
                {'attempt_id': latest.attempt_id, 'policy_version': latest.new_policy_version,
                 'adapter_sha256': latest.adapter_sha256}
                if latest is not None else None
            ),
        }
        return self.startup_recovery

    @staticmethod
    def _registration_guard(task):
        # TaskSpec is frozen, but its nested grids and target mappings are mutable.
        return content_hash({name: getattr(task, name, None) for name in (
            'task_id', 'challenge', 'targets', 'instruction', 'input_roles',
            'submission_targets', 'conventions', 'split', 'pool',
        )})

    def _registration_context(self):
        return content_hash({
            'prompt': prompt_identity(), 'tool_hash': self.tool_hash,
            'env_contract': self.contract.as_dict(),
            'tokenizer': self.tokenizer_hash,
            'artifact': self.engine.evidence['artifact_fingerprint'],
            'budget_policy': budget_policy_identity(), 'arc_env_schema': SCHEMA_VERSION,
            'reward_schema': REWARD_SCHEMA,
            'constrained': bool(self.config.get('rollout', {}).get('constrained_decoding', False)),
        })

    def ensure_registered_tasks(self, tasks):
        """Reuse identities only after revalidating task content and shared contracts."""
        context = self._registration_context()
        refresh_all = context != self._registered_context_guard
        for task in tasks:
            key = self.task_keys.get(task.task_id)
            current = (not refresh_all and key is not None
                       and self._registered_task_guards.get(task.task_id) == self._registration_guard(task)
                       and self.store.task_registration_current(
                           key, content_hash(task.targets),
                           allowed=task.split == 'train' and task.pool != 'diagnostic'))
            if not current:
                self.register(task)
        if self._registration_context() != context:
            raise ContractError('BEPA registration contracts changed during task selection')
        self._registered_context_guard = context

    def register(self, task):
        messages = build_task_messages(task)
        spec = BepaTask.create(public_challenge=task.challenge,
            clean_prompt=json.dumps(messages, ensure_ascii=False, separators=(',', ':')),
            tool_contract={'sha256': self.tool_hash, 'tools': action_tools(), 'arc_env_schema': SCHEMA_VERSION,
                           'budget_policy': budget_policy_identity(),
                           **({'policy_distribution':'grammar_conditioned_policy/v1'} if self.config.get('rollout',{}).get('constrained_decoding',False) else {})},
            env_contract=self.contract.as_dict(), prompt_contract={'messages_hash': content_hash(messages)},
            tokenizer_fingerprint=self.tokenizer_hash,
            artifact_fingerprint=self.engine.evidence['artifact_fingerprint'],
            private_target_hash=content_hash(task.targets),
            verifier_contract={'toolkit': self.tool_hash, 'contract': self.contract.fingerprint(),
                               'generated': task.instruction is not None, 'reward_schema': REWARD_SCHEMA})
        self.store.put_task(spec, allowed=task.split == 'train' and task.pool != 'diagnostic')
        self.task_keys[task.task_id] = spec.task_key
        self._registered_task_guards[task.task_id] = self._registration_guard(task)
        self.keys_to_tasks[spec.task_key] = task.task_id
        return spec

    def observe_native_group(self, task, result, snapshot):
        exact = result.get('exact')
        if (not isinstance(exact, list) or len(exact) != int(self.engine.group_size)
                or any(type(value) is not bool for value in exact)):
            return None
        return self.store.observe_native_rescue_group(
            self.task_keys[task.task_id], engine_epoch=self.epoch,
            group_id=result['group_id'], policy_version=snapshot.version, native_exact=exact,
        )

    def enqueue(
        self,
        task,
        *,
        owner_worker_id=None,
        owner_executor_id=None,
        original_group_id=None,
    ):
        if (owner_worker_id is None) != (owner_executor_id is None):
            raise ContractError('rescue owner Worker and executor IDs must be supplied together')
        if original_group_id is not None and owner_worker_id is None:
            raise ContractError('original rescue group provenance requires its Worker/executor owner')
        if owner_worker_id is not None:
            if (not isinstance(owner_worker_id, str) or not owner_worker_id.strip()
                    or not isinstance(owner_executor_id, str) or not owner_executor_id.strip()):
                raise ContractError('rescue owner Worker and executor IDs must be non-empty strings')
            configured = getattr(self.engine, 'worker_executor_ids', None)
            if not isinstance(configured, Mapping) or configured.get(owner_worker_id) != owner_executor_id:
                raise ContractError('rescue owner does not match the current explicit Worker/executor map')
        with self._claim_lock:
            spec = self.register(task)
            plans = self.store.retained_plans([spec.task_key])
            if plans:
                # Teacher text survives contract/model updates. Its original
                # source receipt stays historical; only new student output is scored.
                plan = plans[0]
                job = self.store.get_job(plan.job_id)
                if owner_worker_id is not None:
                    self.store.bind_rescue_owner(
                        plan.job_id, task_key=spec.task_key, worker_id=owner_worker_id,
                        executor_id=owner_executor_id, original_group_id=original_group_id,
                    )
                return {'job_id': plan.job_id, 'plan_id': plan.plan_id,
                        'status': job.state, 'reused_teacher_plan': True}
            public_job_key = spec.public_job_key(source_contract={'toolkit': self.tool_hash, 'arc_env_schema': SCHEMA_VERSION},
                teacher_id=self.settings['teacher']['model'])
            max_job_tokens = int(self.settings['teacher']['budget']['max_job_tokens'])
            existing = self.store.get_job_by_public_key(self.scope, public_job_key)
            pending_alias_job = None
            if existing is None:
                aliases = self.store.teacher_pending_task_aliases((spec.task_key,), budget_scope=self.scope)
                matches = self.store.list_jobs(states=('queued','retry_wait','teacher_running'),
                    budget_scope=self.scope, task_allowlist=tuple(aliases), limit=1) if aliases else []
                pending_alias_job = matches[0] if matches else None
                existing = pending_alias_job
            if existing is None:
                keys = tuple(self.task_keys.values())
                aliases = self.store.teacher_pending_task_aliases(keys, budget_scope=self.scope)
                pending = self.store.list_jobs(
                    states=('queued','retry_wait','teacher_running','source_submitted','source_verified'),
                    budget_scope=self.scope, task_allowlist=(*keys, *aliases),
                )
                if len(pending) >= self.settings['teacher']['max_pending_jobs']:
                    return {'status': 'queue_full'}
            try:
                job = pending_alias_job or self.store.enqueue_job(spec.task_key, public_job_key=public_job_key,
                    budget_scope=self.scope, max_attempts=existing.max_attempts if existing else 3, max_tokens=max_job_tokens)
            except BudgetUnavailableError as exc:
                return {'status': 'budget_blocked', 'reason': exc.reason}
            if owner_worker_id is not None:
                self.store.bind_rescue_owner(
                    job.job_id,
                    task_key=spec.task_key,
                    worker_id=owner_worker_id,
                    executor_id=owner_executor_id,
                    original_group_id=original_group_id,
                )
            return {'job_id': job.job_id, 'status': job.state}

    def rescue_owner(self, task):
        """Return the durable owner only when its executor still owns that Worker."""
        spec = self.register(task)
        owner = self.store.rescue_owner_for_task(spec.task_key)
        if owner is None:
            return None
        configured = getattr(self.engine, 'worker_executor_ids', None)
        if (not isinstance(configured, Mapping)
                or configured.get(owner['worker_id']) != owner['executor_id']):
            raise ContractError('persisted rescue owner differs from the current explicit Worker/executor map')
        return owner

    def registered_rescue_owner(self, task, task_key):
        """Read an already registered task's owner without rebuilding its identity."""
        if (self.task_keys.get(task.task_id) != task_key
                or self._registered_task_guards.get(task.task_id) != self._registration_guard(task)):
            raise ContractError('rescue owner task identity differs from its registration')
        if not self.store.task_registration_current(
                task_key, content_hash(task.targets),
                allowed=task.split == 'train' and task.pool != 'diagnostic'):
            # Restore the same durable allowlist semantics as a fresh registration.
            self.register(task)
            if self.task_keys[task.task_id] != task_key:
                raise ContractError('rescue owner task identity changed during registration')
        owner = self.store.rescue_owner_for_task(task_key)
        if owner is not None:
            configured = getattr(self.engine, 'worker_executor_ids', None)
            if (not isinstance(configured, Mapping)
                    or configured.get(owner['worker_id']) != owner['executor_id']):
                raise ContractError('persisted rescue owner differs from the current explicit Worker/executor map')
        return owner

    def update_budget_limits(self, *, max_job_tokens=None, max_total_tokens=None):
        if max_job_tokens is not None and (type(max_job_tokens) is not int or max_job_tokens < 1):
            raise ContractError("max_job_tokens must be a positive integer")
        if max_total_tokens is not None and (type(max_total_tokens) is not int or max_total_tokens < 1):
            raise ContractError("max_total_tokens must be a positive integer")
        with self._claim_lock:
            updated = self.store.update_budget_limits(
                self.scope, max_job_tokens=max_job_tokens, max_total_tokens=max_total_tokens)
            if max_job_tokens is not None:
                self.settings["teacher"]["budget"]["max_job_tokens"] = max_job_tokens
            if max_total_tokens is not None:
                self.settings["teacher"]["budget"]["max_total_tokens"] = max_total_tokens
            retried = self.store.retry_unstarted_budget_jobs(self.scope, tuple(self.task_keys.values()))
            active = self.store.list_jobs(states=("teacher_running",))
        return {**updated,
                "unstarted_budget_jobs_requeued": retried,
                "max_job_tokens": int(self.settings["teacher"]["budget"]["max_job_tokens"]),
                "active_jobs_unchanged": len(active),
                "usage_accounting": "total_tokens minus cache_read input tokens"}

    def ready_task(self, task_ids, version, exclude=()):
        keys = [self.task_keys[t] for t in task_ids if t in self.task_keys]
        retired = self.store.retired_rescue_task_keys(keys, engine_epoch=self.epoch)
        keys = [key for key in keys if key not in retired]
        row = self.store.peek_ready(keys, engine_epoch=self.epoch, current_policy_version=version,
                                  exclude_task_keys=[self.task_keys[t] for t in exclude if t in self.task_keys])
        return self.keys_to_tasks[row.spec.task_key] if row is not None else None

    def _new_env(self, task, identity, *, token_budget=None):
        contract = self.contract if token_budget is None else replace(self.contract, token_budget=token_budget,
            generated_token_budget=None, observation_token_budget=None)
        env = ARCToolEnv(task.challenge, task.targets, task.public_id,
            workdir=self.root / 'sessions' / identity, contract=contract,
            token_budget=contract.token_budget, tokenizer=self.tokenizer,
            instruction=task.instruction, input_roles=task.input_roles,
            submission_targets=task.submission_targets, conventions=getattr(task, "conventions", None))
        env.reset()
        return env

    @staticmethod
    def _predictions(env):
        return {target: {str(attempt): env.session.read(artifact).data
                         for attempt, artifact in slots.items()}
                for target, slots in env.state.submissions.items()}

    def _verify_replay(self, task, events, *, input_ids, policy_mask, verification_evidence=None, termination_reason=None):
        env = self._new_env(task, 'replay-' + uuid.uuid4().hex)
        request_fields = tuple(TurnRequest.model_fields)
        response_fields = tuple(TurnResponse.model_fields)
        try:
            rollout = self.config["rollout"]
            turn_max_tokens = int(rollout["turn_max_tokens"])
            response_limit = int(rollout["response_length"])
        except (KeyError, TypeError, ValueError) as exc:
            env.close()
            raise ContractError("replay requires configured rollout token limits") from exc
        if turn_max_tokens < 1 or (response_limit is not None and response_limit < 1):
            env.close()
            raise ContractError("replay rollout token limits must be positive")

        terminal = False
        last_response = None
        tool_call_failures = 0

        def verify_response(event, recorded, *, observation, observation_role, done,
                            infra_abort, outcome, reason, retryable=False,
                            retries_remaining=0):
            expected = TurnResponse.model_validate({
                "observation": observation,
                "observation_role": observation_role,
                "done": done,
                "infra_abort": infra_abort,
                "outcome": outcome,
                "reason": reason,
                "retryable": retryable,
                "retries_remaining": retries_remaining,
                "token_budget": env.token_budget,
                "tokens_used": env.tokens_used(),
                "tokens_remaining": max(0, env.token_budget - env.tokens_used()),
                **({"generated_token_budget": env.contract.generated_token_budget,
                    "observation_token_budget": env.contract.observation_token_budget,
                    "generated_tokens": env.state.out_tokens,
                    "observation_tokens": env.state.obs_tokens}
                   if env.contract.generated_token_budget is not None else {}),
            })
            if expected != recorded:
                raise ContractError(
                    f"replay response or budget receipt differs at sequence {event['sequence']}")
            return expected

        try:
            if not isinstance(events, list) or not events:
                raise ContractError("successful replay requires a non-empty v2 event list")

            try:
                response_offsets = replay_response_offsets(events, input_ids, policy_mask)
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError("replay sampled-token/context layout is invalid") from exc
            for sequence, event in enumerate(events):
                if terminal:
                    raise ContractError("replay contains events after a terminal response")
                if not isinstance(event, dict):
                    raise ContractError("replay event must be an object")
                try:
                    request = TurnRequest.model_validate(
                        {key: event[key] for key in request_fields})
                    recorded = TurnResponse.model_validate(
                        {key: event[key] for key in response_fields if key in event})
                except (KeyError, TypeError, ValueError) as exc:
                    raise ContractError(
                        f"replay event {sequence} is missing or violates the {ENV_SCHEMA_VERSION} schema") from exc

                if request.sequence != sequence:
                    raise ContractError("replay turn sequence is not contiguous")

                remaining_before = env.token_budget - env.tokens_used()
                generation_plan = plan_generation(
                    remaining_budget=max(0, remaining_before), episode_budget=env.token_budget,
                    turn_max_tokens=turn_max_tokens,
                    remaining_response=max(0, response_limit - response_offsets[sequence]),
                    **env.generation_budget_fields(),
                )
                context_limit = min(
                    int(self.config.get("learner", {}).get("max_sequence_length", MAX_CONTEXT_TOKENS)),
                    int(self.config.get("worker_inference", {}).get("max_model_len", MAX_CONTEXT_TOKENS)),
                )
                expected_limit = context_generation_limit(
                    generation_plan, prompt_tokens=policy_mask.index(1) + response_offsets[sequence],
                    context_limit=context_limit,
                )
                if (event.get("budget_policy") != BUDGET_POLICY_ID
                        or event.get("budget_phase") != generation_plan.phase):
                    raise ContractError("replay generation budget policy or phase differs")
                if expected_limit < 1 or request.requested_max_tokens != expected_limit:
                    raise ContractError(
                        f"replay requested_max_tokens differs at sequence {sequence}")

                raw_length = len(request.generated_ids)
                if raw_length > expected_limit:
                    raise ContractError("replay sampled output exceeds its requested budget")
                env.record_generation(raw_length)

                if env.budget_exhausted(exceeded=True):
                    expected = verify_response(
                        event, recorded, observation="", observation_role="none", done=True,
                        infra_abort=False, outcome="episode_complete",
                        reason=env.budget_limit_reason())
                    terminal = expected.done
                    last_response = expected
                    continue

                schema_issue = (
                    tool_schema_issue(request.name, request.arguments)
                    if request.parse_status == "valid" else None
                )
                schema_error = schema_issue["code"] if schema_issue is not None else None
                rejection_reason = turn_rejection_reason(
                    request.parse_status, request.generation_stop_reason,
                    request.generation_limit_reached, schema_error,
                )
                if is_model_stop(request.parse_status, request.generation_stop_reason,
                                 request.generation_limit_reached):
                    expected = verify_response(
                        event, recorded, observation="", observation_role="none", done=True,
                        infra_abort=False, outcome="episode_complete", reason="model_stop",
                    )
                    terminal, last_response = True, expected
                    continue

                if rejection_reason is not None:
                    failed_tool_generation = (
                        request.parse_status != "valid" or schema_error is not None
                    ) and rejection_reason != "generation_limit"
                    if failed_tool_generation:
                        tool_call_failures += 1
                    if tool_call_failures > MAX_TOOL_CALL_FAILURES:
                        raise ContractError("replay exceeded the tool-call failure limit")
                    done = (not failed_tool_generation
                            or tool_call_failures >= MAX_TOOL_CALL_FAILURES)
                    role = "tool" if request.parse_status == "valid" else "user"
                    feedback = tool_call_failure_feedback(
                        rejection_reason, parse_status=request.parse_status,
                        generation_limit_reached=request.generation_limit_reached,
                        requested_tool_name=request.requested_tool_name,
                        validation_message=schema_issue["message"] if schema_issue else None,
                        tool_call_failures=tool_call_failures, done=done,
                    )
                    feedback.update(tool_call_failures=tool_call_failures,
                                    max_tool_call_failures=MAX_TOOL_CALL_FAILURES)
                    observation = env.render_budget_observation(feedback, role=role)
                    if env.budget_exhausted():
                        rejection_reason = env.budget_limit_reason()
                        done = True
                    expected = verify_response(
                        event, recorded, observation=observation, observation_role=role if observation else "none",
                        done=done, infra_abort=False,
                        outcome="episode_complete" if done else "rejection",
                        reason=rejection_reason, retryable=not done,
                        retries_remaining=(MAX_TOOL_CALL_FAILURES - tool_call_failures
                                           if not done else 0),
                    )
                    terminal = expected.done
                    last_response = expected
                    continue

                step = env.step(request.arguments)
                observation = step.observation
                infra_abort = bool(step.info.get("aborted"))
                used = env.tokens_used()
                over_budget = env.budget_exhausted(exceeded=True)
                tool_failed = step.info.get("status") != "ok" and not infra_abort
                if tool_failed:
                    tool_call_failures += 1
                failure_limit_reached = (
                    tool_failed and tool_call_failures >= MAX_TOOL_CALL_FAILURES
                )
                done = bool(step.done or infra_abort or failure_limit_reached
                            or env.budget_exhausted())
                expected = verify_response(event, recorded, observation=observation,
                    observation_role="tool" if observation else "none", done=done, infra_abort=infra_abort,
                    outcome=("rejection" if tool_failed and not done else
                             "episode_complete" if done or infra_abort else "tool_result"),
                    reason=("tool_error" if tool_failed else
                            env.budget_limit_reason() if (over_budget or
                            (env.contract.generated_token_budget is not None and env.budget_exhausted()))
                          else "action_executed"),
                    retryable=tool_failed and not done,
                    retries_remaining=(MAX_TOOL_CALL_FAILURES - tool_call_failures
                                       if tool_failed and not done else 0))

                terminal = expected.done
                last_response = expected

            exact, _ = env.score_targets()
            predictions = self._predictions(env)
            normal_reserve_stop = False
            if not terminal and termination_reason == "budget_reserve" and last_response is not None:
                remaining_response = max(0, response_limit - (len(input_ids) - policy_mask.index(1)))
                next_plan = plan_generation(
                    remaining_budget=max(0, env.token_budget - env.tokens_used()),
                    episode_budget=env.token_budget, turn_max_tokens=turn_max_tokens,
                    remaining_response=remaining_response, **env.generation_budget_fields(),
                )
                normal_reserve_stop = context_generation_limit(
                    next_plan, prompt_tokens=len(input_ids), context_limit=context_limit,
                ) == 0
            if ((not terminal and not normal_reserve_stop) or last_response is None
                    or (terminal and (not last_response.done or
                        (last_response.reason, last_response.observation_role) not in {
                        ("action_executed", "tool"), ("model_stop", "none"),
                        ("generated_budget_exceeded", "tool"), ("observation_budget_exceeded", "tool")}))
                    or last_response.infra_abort
                    or not exact or env.state.aborted or env.budget_exhausted(exceeded=True)):
                raise ContractError("student replay failed trusted exact successful-completion verification")
            if verification_evidence is not None:
                verification_evidence.update(env.training_verification.summary())
            return predictions, env.cost(), env.state.out_tokens, env.state.obs_tokens, env.state.successful_tool_calls, tool_call_failures
        finally:
            env.close()

    def cache_student(self, task, row, policy, clean_prefix_ids, *, guided=False, seed_claim=None, trusted_summary=None):
        if guided != (seed_claim is not None):
            raise ContractError('guided cache creation requires its active seed claim')
        spec = self.register(task)
        if row.get('exact') is not True or not row.get('tool_events'):
            raise ContractError('cache requires a real scored successful student episode')
        verification = {}
        predictions, cost, generated, observed, successful_calls, failures = self._verify_replay(task, row['tool_events'],
            input_ids=row['input_ids'], policy_mask=row['policy_mask'], verification_evidence=verification,
            termination_reason=row.get('termination_reason'))
        if (type(row.get('generated_tokens')) is not int or row['generated_tokens'] != generated
                or type(row.get('observation_tokens')) is not int
                or row['observation_tokens'] != observed):
            raise ContractError('student replay token totals differ from the scored episode artifact')
        split = row['policy_mask'].index(1)
        episode = SourceEpisode(episode_id=row['episode_id'], task_key=spec.task_key,
            source_policy=SourcePolicy(self.epoch, policy.version, policy.adapter_sha256),
            row_kind='guided' if guided else 'native', input_ids=row['input_ids'],
            policy_mask=row['policy_mask'], guided_prefix_length=split,
            clean_prefix_ids=clean_prefix_ids, raw_trace={**row, 'predictions': predictions,
                'source_origin': 'student', 'construction': 'native',
                'conditioning': 'task_plan' if guided else 'task',
                'generated_tokens': generated, 'observation_tokens': observed, 'cost': cost,
                'successful_tool_calls': successful_calls, 'tool_call_failures': failures,
                'training_verification': verification, 'reward_schema': REWARD_SCHEMA},
            tokenizer_fingerprint=self.tokenizer_hash,
            behavior_logprobs=row['behavior_logprobs'][split:],
            max_context_tokens=int(self.config['learner']['max_sequence_length']))
        projection = project_source_episode(episode, spec)
        receipt = TrustedReceipt(uuid.uuid4().hex, spec.task_key, episode.episode_id,
            spec.private_target_hash, content_hash(spec.verifier_contract), True, 1.0,
            content_hash(predictions))
        refs = self.artifacts.write_episode(raw=episode, projection=projection, receipt=receipt)
        cache = self.store.create_cache_entry(spec.task_key, engine_epoch=self.epoch, episode=episode,
            projection=projection, receipt=receipt, raw_artifact_ref=refs.raw_artifact_ref,
            projection_artifact_ref=refs.projection_artifact_ref, receipt_artifact_ref=refs.receipt_artifact_ref,
            seed_claim=seed_claim)
        if trusted_summary is not None:
            trusted_summary.update(exact=True, cost=cost, generated_tokens=generated,
                observation_tokens=observed, successful_tool_calls=successful_calls,
                tool_call_failures=failures, training_verification=verification)
        return cache

    def training_group_rewards(self, task, rows, policy_version):
        """Compute the shared reward over every member of the actual training group."""
        costs = [row['cost'] for row in rows]
        exacts = [row['exact'] for row in rows]
        coverages = [row['training_verification']['coverage'] if row['exact'] else 0.0 for row in rows]
        if task.instruction is not None:
            # Recompute the shared runtime reward over every fresh/replay row, without taper.
            members = [SimpleNamespace(policy_version=policy_version,
                exact=row['exact'], generated_tokens=row.get('generated_tokens', sum(row['policy_mask'])),
                tool_call_failures=row['tool_call_failures'],
                termination_reason=row.get('termination_reason'),
                env=SimpleNamespace(state=SimpleNamespace(
                    obs_tokens=row.get('observation_tokens', 0), aborted=False,
                    successful_tool_calls=row.get('successful_tool_calls', 0))))
                for row in rows]
            rewards = generated_group_rewards(SimpleNamespace(task_id=task.task_id), members, self.contract.epsilon,
                verification_coverages=coverages)
        else:
            rewards = group_rewards(costs, exacts, self.contract.epsilon,
                successful_tool_calls=[row.get('successful_tool_calls', 0) for row in rows],
                tool_call_failures=[row['tool_call_failures'] for row in rows],
                termination_reasons=[row.get('termination_reason') for row in rows],
                verification_coverages=coverages)
        return rewards

    def admit_group(self, result, task, policy, batch_id, *, current_policy_version: int):
        spec = self.register(task)
        winners = [row for row in result['trajectories'] if row.get('exact') is True]
        if winners and result.get("guided_rescue_replay") is not True:
            # A harness turn limit can score an already submitted answer exact
            # without an environment-terminal receipt. Such rows remain valid
            # fresh samples, but cannot serve as trusted replay cache sources.
            cache_winners = [row for row in winners if row.get('tool_events')
                             and row['tool_events'][-1].get('done') is True]
            if cache_winners:
                chosen = random.Random(content_hash(result['group_id'])).choice(cache_winners)
                prefix = chosen['input_ids'][:chosen['policy_mask'].index(1)]
                self.cache_student(task, chosen, policy, prefix)
            return None
        if any(result['exact']) and result.get('guided_rescue_replay') is not True:
            raise ContractError('scored exact data is inconsistent with source rows')
        guided_cache = result.get('guided_cache_entry_id') if result.get('guided_rescue_replay') is True else None
        reservation_kwargs = {'expected_cache_entry_id': guided_cache} if guided_cache is not None else {}
        reservation = self.store.reserve_cache(spec.task_key, engine_epoch=self.epoch,
            current_policy_version=current_policy_version, batch_id=batch_id, group_id=result['group_id'],
            **reservation_kwargs)
        if reservation is None:
            if guided_cache is None:
                result['rescue'] = self.enqueue(task)
            return None
        result['reservation'] = reservation
        return reservation

    def training_batch(self, source, batch_id, version):
        rows, fresh, workers, reservations = [], [], [], []
        engine_config = getattr(self.engine, 'config', {})
        central_gradient = (
            isinstance(engine_config, dict)
            and engine_config.get('update_backend') == 'central_gradient'
        )
        for group in source.groups:
            original = group['trajectories']
            if central_gradient:
                expected_group_size = int(self.config['rollout']['group_size'])
                if (len(original) != expected_group_size
                        or any(row.get('row_kind', 'fresh_student') != 'fresh_student'
                               for row in original)):
                    raise ContractError(
                        'central BEPA input must contain only the newly generated full native task group'
                    )
                if len(group.get('worker_ids', ())) != expected_group_size:
                    raise ContractError('central BEPA native group lost its pinned original Worker')
                from ..worker.episode_queue import sampling_group_owner
                try:
                    sampling_group_owner(
                        group, getattr(getattr(self.engine, 'store', None), 'episode_queue', None),
                    )
                except ValueError as exc:
                    raise ContractError('central BEPA native group lost its pinned original Worker') from exc
            final = [dict(row) for row in original]
            reservation = group.get('reservation')
            if central_gradient and reservation is not None and (
                group.get('guided_rescue_replay') is not True
                or reservation.cache_entry_id != group.get('guided_cache_entry_id')
            ):
                raise ContractError(
                    'central BEPA cannot append a historical guided cache to a new native group'
                )
            if central_gradient and group.get('guided_rescue_replay') is True and reservation is None:
                raise ContractError('central whole-group guided replay is missing its durable reservation')
            if reservation:
                task = group['task']
                cache_entry = self.store.get_cache_entry(reservation.cache_entry_id)
                raw = self.artifacts.read_json(cache_entry.raw_artifact_ref)
                projection = self.artifacts.read_json(reservation.projection_artifact_ref)
                receipt = self.artifacts.read_json(reservation.receipt_artifact_ref)
                raw_trace = raw['episode']['raw_trace']
                predictions = raw_trace['predictions']
                if (content_hash(task.targets) != receipt['receipt']['private_target_hash']
                        or set(predictions) != set(task.targets)
                        or any(not isinstance(predictions[target], dict)
                               or not any(value == expected for value in predictions[target].values())
                               for target, expected in task.targets.items())):
                    raise ContractError('cache targets no longer match current private task revision')
                replay = replay_row_from_artifacts(projection=projection, receipt=receipt,
                    projection_artifact_ref=reservation.projection_artifact_ref,
                    group_id=group['group_id'], cache_entry_id=reservation.cache_entry_id)
                replay_row = replay.to_training_dict()
                replay_row.update({'exact': True, 'cost': raw_trace['cost'],
                    'generated_tokens': raw_trace['generated_tokens'],
                    'observation_tokens': raw_trace['observation_tokens'],
                    'successful_tool_calls': raw_trace.get('successful_tool_calls', 0),
                    'tool_call_failures': raw_trace['tool_call_failures'],
                    'training_verification': raw_trace['training_verification'], 'reward_schema': REWARD_SCHEMA})
                segments = raw_trace.get('constraint_segments')
                if self.config.get('rollout',{}).get('constrained_decoding',False):
                    if not segments:
                        raise ContractError('grammar-conditioned cache entry lacks source constraint segments')
                    from ..contracts.constraints import rebase_constraint_segments
                    replay_row['constraint_segments'] = rebase_constraint_segments(
                        segments,raw_trace['input_ids'],raw_trace['policy_mask'],
                        replay_row['input_ids'],replay_row['policy_mask'],
                        original_prefix=raw['episode']['guided_prefix_length'],
                        projected_prefix=projection['projection']['prompt_token_count'])
                    replay_row['verification_receipt']['constraint_source_raw_artifact'] = cache_entry.raw_artifact_ref
                final.append(replay_row)
                rewards = self.training_group_rewards(task, final, source.policy.version)
                for row, reward in zip(final, rewards):
                    row['reward'] = reward
                    row['reward_schema'] = REWARD_SCHEMA
                    bonus = VERIFICATION_COEFFICIENT * row['training_verification']['coverage'] if row['exact'] else 0.0
                    row['reward_components'] = {'exact': float(row['exact']), 'verification': bonus,
                        'efficiency': reward - 1.0 - bonus if row['exact'] else 0.0,
                        'failure': reward if not row['exact'] else 0.0}
                reservations.append(reservation.reservation_id)
            for i, row in enumerate(final):
                rows.append(row)
                if row['row_kind'] == 'fresh_student':
                    fresh.append(row)
                    workers.append(group['worker_ids'][i])
        if not rows or not fresh:
            raise ContractError('BEPA batch requires remaining real fresh student rows')
        return TrainingBatch(batch_id, source, rows, fresh, workers, tuple(reservations))

    def record_provider_receipt(self, job_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._check_lease(job_id, payload['fence'])
        receipt = payload['receipt']
        index = receipt.get('index')
        if (type(index) is not int or index < 0 or receipt.get('request_id') != (
                    f'{job_id}-call-{index}' if payload['fence'] == 1
                    else f"{job_id}-fence-{payload['fence']}-call-{index}")
                or receipt.get('budget_token_basis') != 'flash_uncached_input'):
            raise ContractError('provider receipt does not identify this job call')
        usage = receipt.get('usage')
        tokens = uncached_input_tokens(usage) if usage is not None else None
        if usage is not None and tokens is None:
            # Ambiguous usage is retained as evidence, not counted as confirmed usage.
            tokens = None
        if tokens is not None:
            completion = raw_completion_tokens(usage)
            if (completion is None or type(usage.get('total_tokens')) is not int
                    or usage.get('prompt_tokens') + completion != usage['total_tokens']
                    or receipt.get('total_tokens') != tokens):
                raise ContractError('provider receipt has conflicting totals')
        ref = self.artifacts.write_json({'schema': 'bepa-provider-call-v1', 'job_id': job_id,
                                        'fence': payload['fence'], 'receipt': dict(receipt)})
        self.store.record_provider_receipt(job_id, worker_id=payload['worker_id'], fence=payload['fence'],
                                          index=index, evidence_ref=ref, tokens=tokens)
        return {'receipt_ref': ref, 'confirmed_tokens': tokens}

    def settle_unknown_usage_policy(self) -> list[dict[str, Any]]:
        """Isolate failed paid attempts under the selected accounting policy, never retry them."""
        policy = self.settings['teacher']['budget'].get('unknown_usage_policy', 'confirmed_lower_bound')
        if policy == 'block':
            return []
        if policy != 'confirmed_lower_bound':
            raise ContractError('unknown BEPA unknown_usage_policy')
        settled = []
        for job in self.store.list_jobs(states=('outcome_unknown','source_submitted','source_verified','plan_ready','cache_ready'), budget_scope=self.scope, limit=100, unsettled_usage_only=True):
            if not job.usage_unknown:
                continue
            # This is control-plane evidence, not a fabricated provider usage receipt.
            ref = self.artifacts.write_json({
                'schema': 'bepa-unknown-usage-policy-v1', 'job_id': job.job_id,
                'fence': job.fence, 'budget_scope': job.budget_scope,
                'failure_reason': job.fail_reason, 'accounting_basis': policy,
                'provider_usage_known': False,
            })
            try:
                settled.append(self.store.settle_unknown_lower_bound(
                    job.job_id, evidence_ref=ref, fence=job.fence,
                    reason='configured confirmed_lower_bound accounting policy',
                    acceptance_source='configured_policy', preserve_completed=job.state != 'outcome_unknown'))
            except ConflictError:
                # A concurrent request may already have accepted the same failed attempt.
                latest = self.store.get_job(job.job_id)
                if latest.fail_reason not in {'usage_lower_bound_settled', 'provider_usage_reconciled'}:
                    raise
        return settled

    def claim_job(self, worker_id, *, claim_id=None):
        self.teacher_last_seen = time.time()
        with self._claim_lock:
            self.settle_unknown_usage_policy()
            task_allowlist = tuple(
                task_key for task_id, task_key in self.task_keys.items()
                if task_id in self.tasks
                and getattr(self.tasks[task_id], 'split', None) == 'train'
                and getattr(self.tasks[task_id], 'pool', None) != 'diagnostic'
            )
            aliases = self.store.teacher_pending_task_aliases(task_allowlist, budget_scope=self.scope)
            for origin_key, current_key in aliases.items():
                self.keys_to_tasks[origin_key] = self.keys_to_tasks[current_key]
            task_allowlist = tuple(dict.fromkeys((*task_allowlist, *aliases)))
            lease = self.store.claim_job(
                worker_id,
                lease_seconds=float(self.settings['teacher']['budget']['job_timeout_seconds']),
                task_allowlist=task_allowlist,
                claim_id=claim_id,
                single_active_worker=True,
                budget_scope=self.scope,
            )
        if lease is None:
            # claim_job may just have fenced an expired lease. Heal its accounting
            # before the next poll, also covering a crash between failure and settlement.
            self.settle_unknown_usage_policy()
            return None
        task_id = self.keys_to_tasks[lease.task_key]
        job = self.store.get_job(lease.job_id)
        budget_remaining = min(job.reserved_tokens, max(0, job.max_tokens-job.used_tokens))
        return {'job_id': lease.job_id, 'task_id': task_id, 'public_task': lease.public_task,
                'worker_id': worker_id, 'fence': lease.fence,
                'max_job_tokens': job.max_tokens-job.used_tokens,
                'budget_remaining': budget_remaining,
                'deadline': lease.lease_expires_at}

    def _check_lease(self, job_id, fence):
        job = self.store.get_job(job_id)
        if job.fence != fence or job.state != 'teacher_running' or not job.lease_expires_at or time.time() >= job.lease_expires_at:
            raise LeaseLostError('teacher lease is no longer current')
        return job

    def source_start(self, job_id, payload):
        job = self._check_lease(job_id, payload['fence'])
        budget_remaining = min(job.reserved_tokens, max(0, job.max_tokens-job.used_tokens))
        if budget_remaining < 1:
            raise ContractError('teacher source grant is exhausted')
        prior = self.sessions.get(job_id)
        if prior is not None:
            if prior.fence == payload['fence']:
                session = prior
            else:
                if not prior.finished:
                    prior.env.close()
                prior = None
        if prior is None:
            task_id = self.keys_to_tasks[job.task_key]
            task = self.tasks[task_id]
            session = TeacherSession(job_id, payload['fence'], task_id,
                self._new_env(task, f'teacher-{job_id}-attempt-{job.fence}', token_budget=budget_remaining),
                build_task_messages(task))
            self.sessions[job_id] = session
        return {'session_id': job_id, 'messages': session.messages, 'tools': action_tools(),
                'max_tool_calls': int(self.config['rollout']['max_turns']),
                'max_completion_tokens': self.settings['teacher']['max_completion_tokens'],
                'max_stream_retries': self.settings['teacher'].get('max_stream_retries', 1),
                'unknown_usage_policy': self.settings['teacher']['budget'].get('unknown_usage_policy', 'confirmed_lower_bound'),
                'budget_remaining': budget_remaining}

    @staticmethod
    def _teacher_completion_count(payload, *, shared_turn=False):
        count = payload.get('completion_tokens')
        if type(count) is not int or count < 0:
            raise ContractError('teacher generation requires a non-negative provider token count')
        usage = payload.get('usage')
        if count == 0 and shared_turn:
            if usage is not None:
                raise ContractError('zero-token follow-up tool calls cannot carry a usage receipt')
            return count
        if not isinstance(usage, dict):
            raise ContractError('teacher generation requires the provider usage receipt')
        if type(usage.get('completion_tokens')) is not int or usage['completion_tokens'] != count:
            raise ContractError('completion_tokens differ from the provider usage receipt')
        prompt = usage.get('prompt_tokens')
        total = usage.get('total_tokens')
        if type(prompt) is not int or prompt < 0:
            raise ContractError('provider usage receipt requires non-negative prompt_tokens')
        if type(total) is not int or total != prompt + count or total < 1:
            raise ContractError('provider usage receipt has inconsistent total_tokens')
        return count

    async def source_tool(self, job_id, payload):
        self._check_lease(job_id, payload['fence'])
        session = self.sessions[job_id]
        async with session.lock:
            self._check_lease(job_id, payload['fence'])
            sequence = payload['sequence']
            digest = content_hash(payload)
            if sequence in session.replies:
                previous, response = session.replies[sequence]
                if previous != digest:
                    raise ContractError('tool sequence payload conflict')
                return response
            if session.finished or sequence != len(session.events):
                raise ContractError('tool sequence is not the next sequence')
            count = self._teacher_completion_count(payload, shared_turn=True)
            if count:
                await asyncio.to_thread(session.env.record_generation, count)
                session.env.state.out_count_mode = 'teacher_provider_usage'
            name, args = payload['name'], payload['arguments']
            if name == 'arc_action' and tool_schema_error(name, args) is None:
                step = await asyncio.to_thread(session.env.step, args)
                observed, done, aborted = step.observation, bool(step.done), bool(step.info.get('aborted'))
            else:
                raise ContractError('teacher tool shape is invalid')
            response = {'observation': observed, 'done': done, 'infra_abort': aborted}
            event = {**payload, 'observation': observed, 'done': done, 'infra_abort': aborted}
            event_ref = await asyncio.to_thread(self.artifacts.write_json, {
                'schema': 'bepa-teacher-tool-event-v1', 'job_id': job_id,
                'fence': payload['fence'], 'sequence': sequence,
                'request_hash': digest, 'event': event,
            })
            session.events.append(event)
            session.event_artifact_refs.append(event_ref)
            session.confirmed_completion_tokens += count
            session.replies[sequence] = (digest, response)
            return response

    async def source_finish(self, job_id, payload):
        self._check_lease(job_id, payload['fence'])
        session = self.sessions[job_id]
        async with session.lock:
            self._check_lease(job_id, payload['fence'])
            finish_hash = content_hash(payload)
            if session.finish_hash is not None and session.finish_hash != finish_hash:
                raise ContractError('source finish payload conflict')
            if not session.finished:
                termination = payload.get('termination')
                terminal_generation = None
                if termination == 'model_stop':
                    if not is_model_stop('no_call', payload.get('generation_stop_reason'), False):
                        raise ContractError('teacher model stop requires a normal generation stop')
                    count = self._teacher_completion_count(payload)
                    if session.finish_hash is None:
                        session.env.record_generation(count)
                        session.env.state.out_count_mode = 'teacher_provider_usage'
                        session.confirmed_completion_tokens += count
                    terminal_generation = {key: payload[key] for key in (
                        'generation_stop_reason', 'completion_tokens', 'usage')}
                    terminal_generation['generated_text'] = payload.get('generated_text', '')
                session.finish_hash = finish_hash
                exact, per_target = await asyncio.to_thread(session.env.score_targets)
                rejection_reason = None if termination == 'model_stop' else 'source_not_completed'
                if session.env.state.aborted:
                    rejection_reason = 'episode_aborted'
                elif session.env.tokens_used() > session.env.token_budget:
                    rejection_reason = 'token_budget_exceeded'
                elif not exact:
                    submitted = getattr(session.env.state, 'submissions', {})
                    rejection_reason = ('missing_submission' if any(t not in submitted for t in per_target)
                                        else 'submission_mismatch')
                session.accepted = rejection_reason is None
                source = {'source_origin': 'deepseek', 'construction': 'native',
                          'task_id': session.task_id, 'public_messages': session.messages,
                          'events': session.events, 'event_artifact_refs': session.event_artifact_refs,
                          'confirmed_completion_tokens': session.confirmed_completion_tokens,
                          'terminal_generation': terminal_generation,
                          'accepted': session.accepted}
                session.source_hash = await asyncio.to_thread(self.artifacts.write_json, source)
                session.receipt_ref = await asyncio.to_thread(self.artifacts.write_json, {
                    'job_id': job_id, 'source_hash': session.source_hash,
                    'exact': session.accepted, 'rejection_reason': rejection_reason, 'verifier': self.tool_hash,
                    'private_target_hash': content_hash(self.tasks[session.task_id].targets),
                })
                await asyncio.to_thread(self.store.record_source_validation, job_id, fence=payload['fence'],
                    accepted=session.accepted, reason=rejection_reason, receipt_ref=session.receipt_ref)
                await asyncio.to_thread(session.env.close)
                session.finished = True
                session.finish_hash = finish_hash
            return {'source_accepted': session.accepted,
                    'source_receipt': {'receipt_id': session.receipt_ref, 'source_hash': session.source_hash}}

    def complete_job(self, job_id, payload):
        job = self.store.get_job(job_id)
        if job.state == 'plan_ready':
            return self._repeat_completed_job(job, payload)
        session = self.sessions[job_id]
        job = self._check_lease(job_id, payload['fence'])
        if not session.finished or not session.accepted:
            raise ContractError('no centrally verified successful teacher source')
        plan = validate_teacher_plan(payload.get('plan'))
        messages = payload.get('native_messages')
        tool_messages = [m['content'] for m in messages if m.get('role') == 'tool'] if isinstance(messages, list) else []
        if tool_messages != [e['observation'] for e in session.events]:
            raise ContractError('teacher transcript tool observations differ from native execution')
        source_ref = self.artifacts.write_json(payload)
        usage = payload.get('token_total')
        if type(usage) is not int or usage < 0:
            if (payload.get('usage_known') is not False or usage is not None
                    or self.settings['teacher']['budget'].get('unknown_usage_policy', 'confirmed_lower_bound') != 'confirmed_lower_bound'):
                raise ContractError('provider usage is unavailable; refusing plan admission')
        self.store.complete_job(job_id, worker_id=job.lease_owner, fence=payload['fence'],
            idempotency_key=f'{job_id}-fence-{job.fence}-complete', source_artifact_refs={'native': source_ref}, usage_tokens=usage, usage_is_total=True)
        self.store.mark_source_verified(job_id, source_episode_id=job_id,
            trusted_receipt_ref=session.receipt_ref, idempotency_key=job_id+'-verify')
        plan_ref = self.artifacts.write_json({'plan': plan, 'source_hash': session.source_hash,
                                            'native_messages_ref': source_ref})
        record = self.store.mark_plan_ready(job_id, plan_artifact_ref=plan_ref, source_episode_id=job_id,
            trusted_receipt_ref=session.receipt_ref, idempotency_key=job_id+'-plan')
        self.settle_unknown_usage_policy()
        return {'job_id': job_id, 'status': 'plan_ready', 'plan_id': record.plan_id}

    def _repeat_completed_job(self, job, payload):
        """Accept only the byte-equivalent completion already committed as plan-ready."""
        if payload.get('fence') != job.fence:
            raise LeaseLostError('teacher completion fence is no longer current')
        refs = job.source_artifact_refs or {}
        native_ref = refs.get('native')
        if not native_ref:
            raise ContractError('completed job has no immutable native-source artifact')
        stored_payload = self.artifacts.read_json(native_ref)
        if content_hash(stored_payload) != content_hash(payload):
            raise ConflictError('completed job was retried with different request content')
        plan = self.store.get_plan_for_job(job.job_id)
        if plan is None:
            raise ContractError('plan-ready job has no durable plan record')
        if plan.status not in {'ready', 'seed_running', 'consumed', 'exhausted'}:
            raise ContractError('completed job plan is not in a durable state')
        if plan.plan_artifact_ref != job.plan_artifact_ref:
            raise ContractError('job and plan records identify different plan artifacts')
        plan_artifact = self.artifacts.read_json(plan.plan_artifact_ref)
        if plan_artifact.get('native_messages_ref') != native_ref:
            raise ContractError('durable plan does not identify the committed completion payload')
        return {'job_id': job.job_id, 'status': 'plan_ready', 'plan_id': plan.plan_id}

    def status(self):
        active = []
        for job in self.store.list_jobs(states=('teacher_running',), limit=20):
            if job.budget_scope != self.scope:
                continue
            session = self.sessions.get(job.job_id)
            events = list(session.events) if session else []
            active.append(dict(job_id=job.job_id, task_id=self.keys_to_tasks.get(job.task_key),
                state=job.state, lease_expires_at=job.lease_expires_at,
                reserved_tokens=job.reserved_tokens, max_tokens=job.max_tokens,
                tool_calls=len(events),
                confirmed_uncached_input_tokens=sum_uncached_input_tokens(
                    e for e in events if e.get('usage') is not None),
                token_accounting_basis='flash_uncached_input',
                source_verified=bool(session and session.finished and session.accepted)))
        return {**self.store.status(), 'enabled': True,
                'unknown_usage_policy': self.settings['teacher']['budget'].get('unknown_usage_policy', 'confirmed_lower_bound'), 'engine_epoch': self.epoch,
                'mode': self.mode, 'teacher_last_seen': self.teacher_last_seen,
                'current_budget_scope': self.scope, 'teacher_model': self.settings['teacher']['model'], 'active_teacher_jobs': active,
                'operation': self.operation, 'recovery': self.startup_recovery}

    def settle_unknown_lower_bound(self, job_id: str, evidence_ref: str, reason: str) -> dict[str, Any]:
        """Accept only the stored known-usage lower bound, never a caller-supplied debit."""
        artifact = self.artifacts.read_json(evidence_ref)
        if (not isinstance(artifact, Mapping) or artifact.get("schema") != "bepa-failed-teacher-v1"
                or artifact.get("job_id") != job_id):
            raise ContractError("lower-bound evidence does not identify this failed Teacher job")
        payload = artifact.get("payload")
        if (not isinstance(payload, Mapping) or payload.get("usage_known") is not False
                or payload.get("outcome_unknown") is not True):
            raise ContractError("lower-bound evidence must preserve unknown provider usage")
        return self.store.settle_unknown_lower_bound(
            job_id, evidence_ref=evidence_ref, fence=payload.get("fence"), reason=reason,
        )

    def reconcile_unknown_usage(self, job_id: str, evidence_ref: str) -> dict[str, Any]:
        """Resolve a locked job only from its immutable, complete provider receipts."""
        artifact = self.artifacts.read_json(evidence_ref)
        if (not isinstance(artifact, Mapping)
                or artifact.get("schema") != "bepa-failed-teacher-v1"
                or artifact.get("job_id") != job_id):
            raise ContractError("usage evidence does not identify this failed Teacher job")
        payload = artifact.get("payload")
        if (not isinstance(payload, Mapping) or payload.get("reason") != "usage_unknown"
                or payload.get("usage_known") is not False or payload.get("outcome_unknown") is not True):
            raise ContractError("usage evidence is not a recorded unknown-usage failure")
        calls = payload.get("usage_calls")
        if not isinstance(calls, list) or not calls:
            raise ContractError("unknown-usage evidence has no provider call receipts")
        evidence = payload.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("usage_calls") != calls:
            raise ContractError("failed-job evidence contains conflicting provider call receipts")
        current_fence = payload.get("fence", 1)
        if type(current_fence) is not int or current_fence < 1:
            raise ContractError("unknown-usage evidence has an invalid provider fence")
        call_tokens = []
        for index, receipt in enumerate(calls):
            if not isinstance(receipt, Mapping):
                raise ContractError("provider call receipt must be an object")
            prefix = job_id if current_fence == 1 else f"{job_id}-fence-{current_fence}"
            if (receipt.get("index") != index
                    or receipt.get("request_id") != f"{prefix}-call-{index}"
                    or receipt.get("budget_token_basis") != "flash_uncached_input"
                    or receipt.get("stream_complete") is not True
                    or not isinstance(receipt.get("finish_reason"), str)
                    or not receipt.get("finish_reason")
                    or not isinstance(receipt.get("response_id"), str)
                    or not receipt.get("response_id")):
                raise ContractError("provider call receipt is incomplete or out of order")
            usage = receipt.get("usage")
            tokens = uncached_input_tokens(usage)
            if tokens is None:
                raise ContractError(f"provider call {index} still has ambiguous cached-input usage")
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
            completion_tokens = raw_completion_tokens(usage)
            total = usage.get("total_tokens") if isinstance(usage, Mapping) else None
            if (type(prompt_tokens) is not int or completion_tokens is None or type(total) is not int
                    or prompt_tokens + completion_tokens != total):
                raise ContractError(f"provider call {index} has inconsistent raw usage totals")
            if receipt.get("total_tokens") is not None and receipt.get("total_tokens") != tokens:
                raise ContractError(f"provider call {index} disagrees with its recorded budget count")
            call_tokens.append(tokens)
        return self.store.reconcile_unknown_usage(
            job_id, evidence_ref=evidence_ref, call_tokens=call_tokens,
        )

    def diagnostics(
        self,
        *,
        current_policy_version: int,
        run_id: str | None = None,
        held_task_ids: Sequence[str] | None = None,
        run_batch_ids: Sequence[str] | None = None,
        task_limit: int = 20,
    ) -> dict[str, Any]:
        """Return bounded persisted diagnostics without changing coordinator state."""
        if (isinstance(held_task_ids, (str, bytes)) or isinstance(run_batch_ids, (str, bytes))
                or (held_task_ids is not None and any(not isinstance(item, str) or not item for item in held_task_ids))
                or (run_batch_ids is not None and any(not isinstance(item, str) or not item for item in run_batch_ids))):
            raise ContractError('held_task_ids and run_batch_ids must contain non-empty strings')
        held_keys = None
        unmapped_held_ids: list[str] = []
        if held_task_ids is not None:
            held_keys = []
            for task_id in dict.fromkeys(held_task_ids):
                task_key = self.task_keys.get(task_id)
                if task_key is None:
                    unmapped_held_ids.append(task_id)
                else:
                    held_keys.append(task_key)
        result = self.store.diagnostics(
            budget_scope=self.scope,
            engine_epoch=self.epoch,
            current_policy_version=current_policy_version,
            registered_task_keys=tuple(self.keys_to_tasks),
            run_id=run_id,
            held_task_keys=held_keys,
            run_batch_ids=run_batch_ids,
            task_limit=task_limit,
        )
        for task in result["tasks"]:
            task["task_id"] = self.keys_to_tasks.get(task["task_key"])
        result["context"]["held_task_ids"] = list(dict.fromkeys(held_task_ids or ()))
        result["context"]["unmapped_held_task_ids"] = unmapped_held_ids
        result["context"]["run_batch_ids"] = list(dict.fromkeys(run_batch_ids or ()))
        return result

    def close(self):
        for session in self.sessions.values():
            if session.env.session is not None:
                session.env.close()
        self.store.close()
