"""One-instance process lifecycle shared by local Docker and Vast.

Only the deployment boundary changes. Native Worker launchers and the existing
central-gradient supervisor retain their policy, recovery and GPU-sharing code.
"""
from __future__ import annotations

import copy
import asyncio
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import shlex
import signal
import socket
import stat
import subprocess
import time
import uuid
from typing import Any

import yaml

from .instance_plan import InstanceDeployment, InstanceDeploymentError, build_plan, load_deployment as load_deployment
from .instance_status import build_operator_summary, read_runtime_build_identity
from .local import read_or_create_secret
from .local_gradients import (
    LocalGradientError, _SAFE_INHERITED_ENV, _http_json, _http_request_json,
    _process_record, _proc_record, _read_private_json, _record_matches,
    _require_private_dir, _write_private_json, _write_private_bytes,
)

_SCHEMA = 'qwen-rl-instance-process/v1'
_STOP_OPERATION_SCHEMA = 'qwen-rl-instance-stop-operation/v1'
_STOP_HOLD_SCHEMA = 'qwen-rl-instance-stop-hold/v1'
_LOGGER = logging.getLogger(__name__)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_private_dir(path)


def _secret(path: Path) -> str:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077 or info.st_nlink != 1):
        raise InstanceDeploymentError('credential file must be private, regular and user-owned')
    if info.st_size > 16384:
        raise InstanceDeploymentError('credential file is too large')
    value = path.read_text().strip()
    if not value or any(c.isspace() for c in value):
        raise InstanceDeploymentError('credential file must contain one nonempty token')
    return value


def _port_free(port: int) -> None:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', port))
        except OSError:
            raise InstanceDeploymentError(f'instance port {port} is occupied; stop its owner before serving') from None


def _spawn(argv: list[str], env: dict[str, str], log: Path, *, cwd: Path) -> tuple[subprocess.Popen, dict]:
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        child = subprocess.Popen(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                                 stdout=fd, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        os.close(fd)
    try:
        record = _process_record(child.pid, argv)
        record["log_file"] = str(log)
    except BaseException:
        child.terminate()
        child.wait(timeout=10)
        raise
    return child, record


def _drain_role_group(children, timeout: float) -> bool:
    """Allow owned peers to drain concurrently without killing a timed-out role."""
    pending = []
    clean = True
    deadline = time.monotonic() + timeout
    for child, record in children:
        if child.poll() is not None:
            continue
        if not _record_matches(record):
            clean = False
            continue
        try:
            child.send_signal(signal.SIGINT)
        except ProcessLookupError:
            continue
        pending.append(child)
    for child in pending:
        try:
            child.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            clean = False
    return clean


def _owner_matches(record: dict | None) -> bool:
    if not isinstance(record, dict):
        return False
    pid, started, argv = record.get('pid'), record.get('starttime'), record.get('argv')
    if type(pid) is not int or pid < 1 or type(started) is not int or not isinstance(argv, list):
        return False
    return _proc_record(pid) == (started, tuple(argv))


def configure_cache_environment(deployment: InstanceDeployment) -> None:
    """Apply this deployment's explicit cache location to its own process tree."""
    root = getattr(deployment, "cache_root", None)
    if root is None:
        return
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise InstanceDeploymentError('cache_root must be a regular directory')
    for name, suffix in {
        'HF_HOME': 'huggingface', 'XDG_CACHE_HOME': 'xdg',
        'VLLM_CACHE_ROOT': 'vllm', 'TRITON_CACHE_DIR': 'triton',
        'TORCHINDUCTOR_CACHE_DIR': 'torchinductor', 'TORCH_EXTENSIONS_DIR': 'torch-extensions',
        'CUDA_CACHE_PATH': 'cuda', 'FLASHINFER_WORKSPACE_BASE': 'flashinfer',
    }.items():
        os.environ[name] = str(root / suffix)

    # Keep graph/autograd caches and shape decisions reusable after role restarts.
    for name in (
        'TORCHINDUCTOR_FX_GRAPH_CACHE', 'TORCHINDUCTOR_AUTOGRAD_CACHE',
        'TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO', 'TRITON_CACHE_AUTOTUNING',
        'FLA_CACHE_RESULTS',
    ):
        os.environ[name] = '1'
    os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '0'
    scope = hashlib.sha256(str(deployment.state_dir.resolve()).encode()).hexdigest()[:16]
    os.environ['TORCH_COMPILE_JOB_ID'] = f'qwen-rl-{scope}'


def _initialize_caches() -> None:
    for name in ('HOME', 'XDG_CACHE_HOME', 'HF_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
                 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH',
                 'FLASHINFER_WORKSPACE_BASE'):
        value = os.environ.get(name)
        if not value:
            continue
        directory = Path(value)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not directory.is_dir() or not os.access(directory, os.W_OK):
            raise InstanceDeploymentError(f'instance cache directory is not writable: {name}')


def _gradient_drained(path: Path) -> bool:
    manifest = _read_private_json(path)
    return bool(manifest is not None and isinstance(manifest.get('roles'), list)
                and all(isinstance(row, dict) and not _record_matches(row)
                        for row in manifest['roles']))


def _run_ids_referenced(
    run_dirs: list[Path], *, proc_root: Path = Path('/proc'),
) -> set[str] | None:
    """Return run IDs still named by a process, or None when /proc cannot be audited."""
    if not run_dirs:
        return set()
    try:
        needles = {path.name: os.fsencode(str(path.resolve())) for path in run_dirs}
        processes = tuple(proc_root.iterdir())
    except OSError:
        return None
    referenced: set[str] = set()
    for process in processes:
        if not process.name.isdecimal():
            continue
        try:
            command_line = (process / 'cmdline').read_bytes()
            current_dir = os.readlink(process / 'cwd')
        except FileNotFoundError:
            continue
        except OSError:
            return None
        for run_id, needle in needles.items():
            path = needle.decode(errors='surrogateescape')
            if needle in command_line or current_dir == path or current_dir.startswith(path + os.sep):
                referenced.add(run_id)
    return referenced


class InstanceSupervisor:
    def __init__(self, deployment: InstanceDeployment):
        self.deployment = deployment
        self.active_path = deployment.state_dir / 'active.json'
        self.stopping = False

    def prepare(self) -> dict[str, Any]:
        """Validate and describe the shared topology without starting CUDA or roles."""
        plan = build_plan(self.deployment)
        identity = json.dumps(plan["deployment"], sort_keys=True, default=str).encode()
        plan["config_fingerprint"] = hashlib.sha256(identity + self.deployment.runtime_config.read_bytes()).hexdigest()
        plan["deployment"]["id"] = hashlib.sha256(str(self.deployment.path).encode()).hexdigest()[:20]
        current = self.status()
        operator = current["operator"]
        operator["deployment"]["startup_plan"] = {
            "config_fingerprint": plan["config_fingerprint"],
            "gpu_assignments": [{key: row[key] for key in ("worker_id", "gpu_index", "gpu_uuid")}
                                for row in plan["workers"]],
            "scope": "validated_desired_plan_not_running_identity",
        }
        plan["operator"] = operator
        return plan

    def status(self) -> dict[str, Any]:
        manifest = _read_private_json(self.active_path)
        stop_operation = _read_private_json(self.deployment.state_dir / 'stop-operation.json')
        stop_hold = _read_private_json(self.deployment.state_dir / 'stop-hold.json')
        if stop_operation is not None and stop_operation.get('schema') != _STOP_OPERATION_SCHEMA:
            raise InstanceDeploymentError('instance stop operation receipt is unsupported')
        if stop_hold is not None and stop_hold.get('schema') != _STOP_HOLD_SCHEMA:
            raise InstanceDeploymentError('instance stop hold receipt is unsupported')
        if manifest is None:
            result = {'schema': _SCHEMA, 'status': 'stopped', 'roles': []}
            result.update(stop_operation=stop_operation, stop_hold=stop_hold)
            result['operator'] = build_operator_summary(
                self.deployment, result, observer_build=read_runtime_build_identity(
                    Path(__file__).resolve().parents[1]))
            return result
        if manifest.get('schema') != _SCHEMA:
            raise InstanceDeploymentError('instance ownership manifest is unsupported')
        live = _owner_matches(manifest.get('owner', {}))
        roles = [{**row, 'identity_matches': _record_matches(row)} for row in manifest.get('roles', [])]
        nested = None
        nested_path = manifest.get('gradient_manifest')
        if nested_path:
            nested = _read_private_json(Path(nested_path))
        gradient_live = bool(nested and (
            _record_matches(nested.get('supervisor', {}))
            or any(_record_matches(row) for row in nested.get('roles', []))))
        status = manifest.get('status') if live else (
            'orphaned' if gradient_live or any(r['identity_matches'] for r in roles) else 'stopped')
        error, error_message = manifest.get('error'), manifest.get('error_message')
        if live and nested and nested.get('status') in {'failed', 'stop_incomplete'}:
            status = 'failed'
            error = nested.get('error') or 'InstanceDeploymentError'
            error_message = nested.get('error_message') or 'gradient supervisor failed'
        result = {'schema': _SCHEMA, 'status': status, 'owner': manifest.get('owner'),
                  'roles': roles, 'gradient': nested,
                  'runtime_config': manifest.get('runtime_config'),
                  'run_id': manifest.get('run_id'), 'error': error,
                  'error_message': error_message, 'stop_operation': stop_operation,
                  'stop_hold': stop_hold}
        result['operator'] = build_operator_summary(
            self.deployment, result, owner_live=live,
            observer_build=read_runtime_build_identity(Path(__file__).resolve().parents[1]),
            manifest_build=manifest.get('observer_build'))
        return result

    def _prune_stopped_run_policy_caches(self, *, allow_current_owner: bool = False) -> int:
        """Remove only executor delivery caches after the managed run is drained.

        Runtime configs, role logs, worker pins and manifests remain as run evidence.
        The policy copies are reconstructible from the durable Learner policy store.
        """
        manifest = _read_private_json(self.active_path)
        if manifest is None or manifest.get('schema') != _SCHEMA or manifest.get('status') != 'stopped':
            return 0
        owner = manifest.get('owner')
        if not isinstance(owner, dict):
            return 0
        if _owner_matches(owner):
            # The serving process may prune its own cache only after clean drain.
            if not allow_current_owner or owner.get('pid') != os.getpid():
                return 0
        roles = manifest.get('roles')
        if not isinstance(roles, list) or any(not isinstance(row, dict) for row in roles):
            return 0
        if any(_record_matches(row) for row in roles):
            return 0
        gradient_manifest = manifest.get('gradient_manifest')
        if gradient_manifest:
            gradient_path = Path(gradient_manifest)
            try:
                gradient_path.resolve().relative_to(self.deployment.state_dir.resolve())
            except (OSError, ValueError):
                return 0
            nested = _read_private_json(gradient_path)
            if (nested is None or nested.get('status') != 'stopped'
                    or not _gradient_drained(gradient_path)):
                return 0

        runs_root = self.deployment.state_dir / 'runs'
        if runs_root.is_symlink() or not runs_root.is_dir():
            return 0
        try:
            _require_private_dir(runs_root)
            run_dirs = [path for path in runs_root.iterdir()
                        if path.is_dir() and not path.is_symlink()
                        and len(path.name) == 32
                        and all(character in '0123456789abcdef' for character in path.name)]
        except (OSError, InstanceDeploymentError, LocalGradientError):
            return 0
        referenced = _run_ids_referenced(run_dirs)
        if referenced is None:
            _LOGGER.warning('skipping stopped-run policy cleanup because process references are unreadable')
            return 0

        removed = 0
        for run_dir in run_dirs:
            if run_dir.name in referenced:
                continue
            try:
                executor_dirs = tuple(run_dir.iterdir())
            except OSError:
                continue
            for executor_dir in executor_dirs:
                if (executor_dir.is_symlink() or not executor_dir.is_dir()
                        or not executor_dir.name.startswith('executor-')):
                    continue
                policy_dir = executor_dir / 'policy'
                if policy_dir.is_symlink() or not policy_dir.is_dir():
                    continue
                try:
                    if policy_dir.resolve().parent != executor_dir.resolve():
                        continue
                    shutil.rmtree(policy_dir)
                except OSError as exc:
                    _LOGGER.warning('could not prune stopped executor policy cache (%s)', type(exc).__name__)
                    continue
                removed += 1
        if removed:
            _LOGGER.info('pruned %d stopped executor policy caches; run records and logs retained', removed)
        return removed

    def stop(self, *, progress=None) -> dict[str, Any]:
        current = self.status()
        if current['status'] == 'stopped':
            return current
        owner = current.get('owner')
        if not isinstance(owner, dict) or not _owner_matches(owner):
            raise InstanceDeploymentError('instance owner is absent; refusing to signal orphaned or unrelated processes')
        os.kill(owner['pid'], signal.SIGTERM)
        deadline = time.monotonic() + self.deployment.stop_timeout
        next_progress = 0.0
        while time.monotonic() < deadline:
            state = self.status()
            if state['status'] == 'stopped':
                return state
            if progress is not None and time.monotonic() >= next_progress:
                progress({'stage': 'draining', 'status': state['status'],
                          'owned_roles_remaining': sum(row.get('identity_matches') is True for row in state['roles']),
                          'remaining_timeout_seconds': round(max(0.0, deadline - time.monotonic()), 1),
                          'roles_waiting': [dict(role=r.get('role'), pid=r.get('pid'),
                                                 worker_id=r.get('worker_id'), executor_id=r.get('executor_id'))
                                            for r in state['roles'] if r.get('identity_matches') is True],
                          'gradient_state': (state.get('gradient') or {}).get('status', 'unavailable')})
                next_progress = time.monotonic() + 5
            time.sleep(0.2)
        error = InstanceDeploymentError('instance did not drain before timeout; no unrelated process was killed')
        error.report = {'schema': 'qwen-rl-instance-stop/v1', 'status': 'pending',
                        'reason_code': 'drain_timeout', 'field': 'state_dir.active_manifest',
                        'owned_roles_remaining': sum(row.get('identity_matches') is True for row in state['roles']),
                        'next_command': f'qwen-rl instance --deployment {self.deployment.path} status --human'}
        raise error

    def _control_context(self) -> tuple[dict, str, str]:
        state = self.status()
        if state['status'] != 'running' or not state.get('runtime_config'):
            raise InstanceDeploymentError('instance must be running for control operations')
        config = yaml.safe_load(Path(state['runtime_config']).read_text())
        token = os.environ.get('ARC_ENV_TOKEN')
        if not token:
            path = self.deployment.credential_files.get('ARC_ENV_TOKEN')
            if path is not None:
                token = _secret(Path(path))
        if not token:
            raise InstanceDeploymentError('required credential is unavailable: ARC_ENV_TOKEN')
        return config, token, f"http://127.0.0.1:{config['learner']['port']}"

    def stop_after_checkpoint(self, *, timeout_seconds: float = 30.0, progress=None) -> dict[str, Any]:
        from .instance_stop_control import stop_after_checkpoint
        return stop_after_checkpoint(self, timeout_seconds=timeout_seconds, progress=progress)

    def stop_now(self, *, progress=None) -> dict[str, Any]:
        from .instance_stop_control import stop_now
        return stop_now(self, progress=progress)

    def _poll_checkpoint_stop(self, run: Path, manifest: dict[str, Any]) -> bool:
        from .instance_stop_control import poll_checkpoint_stop
        return poll_checkpoint_stop(self, run, manifest)

    def run(self, steps: int) -> dict:
        if type(steps) is not int or not 1 <= steps <= 100000:
            raise InstanceDeploymentError('steps must be an integer between 1 and 100000')
        config, token, url = self._control_context()
        return _http_request_json(url + '/run', token, method='POST',
            body={'kind': 'training', 'steps': steps}, timeout=config['learner']['request_timeout'])[0]

    def evaluate(self, *, status_only: bool = False) -> dict:
        config, token, url = self._control_context()
        return _http_request_json(url + '/evaluation', token,
            method='GET' if status_only else 'POST', body=None if status_only else {},
            timeout=config['learner']['request_timeout'])[0]

    def watch(self, arguments: list[str]) -> int:
        config_data, token, _url = self._control_context()
        runtime_path = Path(self.status()['runtime_config'])
        from ..runtime.cli import build_parser
        from ..runtime.config import load_config
        from ..runtime.watch import watch, log_tail
        from ..dashboard.sources import resolve_source
        from ..dashboard.central import bind_instance_snapshot, current_instance
        from dataclasses import replace
        args = build_parser().parse_args(['watch', '--config', str(runtime_path), *arguments])
        config = replace(load_config(runtime_path), run_dir=str(runtime_path.parent))
        source = resolve_source('runtime', deployment=runtime_path,
                                instance_manifest=runtime_path.parent / 'gradient-manifest.json')
        def bind(item):
            logical_id = (item.get('learner', {}).get('run') or {}).get('run_id')
            bound = bind_instance_snapshot(source, item, logical_id) if logical_id else item
            if args.log_lines:
                control = current_instance(source)
                records = control.get('roles', []) if control else []
                processes = []
                for row in records:
                    path = Path(str(row.get('log_file', '')))
                    try:
                        path.resolve().relative_to(runtime_path.parent.resolve())
                    except ValueError:
                        continue
                    processes.append({'role': row.get('executor_id') or row.get('role'), 'log': str(path)})
                bound['logs'] = log_tail({'processes': processes}, args.log_lines, tuple(values.values()))
            return bound
        values = {'ARC_ENV_TOKEN': token}
        for row in config_data['workers']:
            values[row['auth_token_env']] = _secret(self.deployment.state_dir / 'tokens' / (row['id'] + '.token'))
        prior = {name: os.environ.get(name) for name in values}
        os.environ.update(values)
        try:
            asyncio.run(watch(config, interval=args.interval, window=args.window,
                timeout=args.timeout, once=args.once, json_output=args.json,
                log_lines=args.log_lines, gpu=not args.no_gpu, details=args.details, snapshot_transform=bind))
        except KeyboardInterrupt:
            pass
        finally:
            for name, value in prior.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        return 0

    def _credentials(self, plan: dict, run: Path) -> dict[str, str]:
        runtime = plan['runtime']
        names = {'ARC_ENV_TOKEN'}
        helper_ids = set(runtime['rollout'].get('helper_worker_ids', ()))
        names.update(row['auth_token_env'] for row in runtime['workers']
                     if row['id'] in helper_ids)
        names.update(runtime['learner']['gradient_executor_token_envs'][executor_id]
                     for executor_id in runtime['learner'].get('gradient_remote_executor_ids', ()))
        if runtime['bepa']['enabled']:
            teacher = runtime['bepa']['teacher']
            names |= {teacher['api_key_env'], teacher['token_env']}
        credentials = {}
        for name in names:
            value = os.environ.get(name)
            if value is None:
                path = self.deployment.credential_files.get(name)
                if path is not None:
                    value = _secret(Path(path))
            if not value or any(c.isspace() for c in value):
                raise InstanceDeploymentError(f'required credential is unavailable: {name}')
            credentials[name] = value
        for row in [*plan['workers'], *plan['executors']]:
            name = row.get('auth_token_env') or row['token_env']
            path = Path(row['auth_token_path'] if 'auth_token_env' in row else row['token_path'])
            _mkdir(path.parent)
            credentials[name] = read_or_create_secret(path, label=name, create=True)
        if len(set(credentials.values())) != len(credentials):
            raise InstanceDeploymentError('all role credentials must be distinct')
        return credentials

    def _pin_workers(self, plan: dict, children: list[tuple[subprocess.Popen, dict]], env: dict) -> list[dict]:
        pins = []
        expected_base = plan['runtime']['worker_inference']['ninfer_artifact_sha256']
        local_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        deadline = time.monotonic() + self.deployment.start_timeout
        for row in plan['workers']:
            while time.monotonic() < deadline and not self.stopping:
                if any(child.poll() is not None for child, _record in children):
                    raise InstanceDeploymentError('owned Worker exited during startup; inspect its private log')
                try:
                    health = _http_json(f"http://127.0.0.1:{row['port']}/health", env[row['auth_token_env']])
                except (OSError, ValueError, RuntimeError):
                    time.sleep(0.2)
                    continue
                deployment = health.get('deployment', {})
                if (health.get('ready') is True and health.get('role') == 'worker'
                        and health.get('worker_id') == row['worker_id']
                        and health.get('gpu_uuid') == row['gpu_uuid']
                        and health.get('boot_id') == local_boot
                        and isinstance(health.get('telemetry_epoch'), str) and health['telemetry_epoch']
                        and health.get('residency_enabled') is True
                        and all(health.get(k) == expected_base for k in (
                            'ninfer_artifact_sha256', 'base_fingerprint', 'artifact_fingerprint'))
                        and deployment.get('scope') == 'service'
                        and deployment.get('site_id') == row['site_id']
                        and deployment.get('node_id') == row['node_id']):
                    pin = {key: health[key] for key in ('boot_id', 'telemetry_epoch', 'base_fingerprint', 'artifact_fingerprint')}
                    pin.update(worker_id=row['worker_id'], gpu_uuid=row['gpu_uuid'], model_sha256=expected_base,
                               site_id=row['site_id'], node_id=row['node_id'], auth_token_env=row['auth_token_env'],
                               endpoint=f"http://127.0.0.1:{row['port']}")
                    pins.append(pin)
                    break
                time.sleep(0.2)
            else:
                raise InstanceDeploymentError('Worker readiness/identity pin did not complete')
        return pins

    def _gradient_plan(self, plan: dict, run: Path, env: dict, pins: list[dict]) -> Path:
        runtime = copy.deepcopy(plan['runtime'])
        by_id = {p['worker_id']: p for p in pins}
        for row in runtime['workers']:
            if row['id'] in by_id:
                row.update({key: by_id[row['id']][key]
                            for key in ('boot_id', 'telemetry_epoch', 'model_sha256')})
        runtime_path = run / 'runtime.yaml'
        _write_private_bytes(runtime_path, yaml.safe_dump(runtime, sort_keys=False).encode())
        server_url = f"http://127.0.0.1:{runtime['learner']['port']}"
        safe_names = set(env) - set(plan['secret_names'])
        server_names = safe_names | set(plan['secret_names'])
        roles = [{'role': 'server', 'argv': [self.deployment.python, '-B', '-m', 'qwen_rl', 'gradients',
                  'server', '--config', str(runtime_path)], 'environment_names': sorted(server_names),
                  'log_path': str(run / 'server.log')}]
        sync = []
        for row in plan['executors']:
            worker = next(w for w in plan['workers'] if w['worker_id'] == row['worker_id'])
            directory = run / row['executor_id']
            _mkdir(directory)
            profile_path = directory / 'compute-profile.json'
            profile = copy.deepcopy(row['compute_profile'])
            profile['run_dir'] = str(directory)
            _write_private_json(profile_path, profile)
            config = copy.deepcopy(row['executor_config'])
            config.update(compute_profile=str(profile_path), working_dir=str(directory))
            source_worker = next(w for w in runtime['workers'] if w['id'] == worker['worker_id'])
            allowed_worker_keys = {'id', 'scope', 'site_id', 'node_id', 'gpu', 'port', 'endpoint',
                                   'url_prefix', 'auth_token_env', 'boot_id', 'telemetry_epoch', 'model_sha256'}
            config['sharing']['workers'] = [{
                **{k: v for k, v in source_worker.items() if k in allowed_worker_keys},
                'gpu_uuid': worker['gpu_uuid']}]
            config_path = directory / 'executor.json'
            _write_private_json(config_path, config)
            from ..learner.gradient_cli import _validate_executor_config
            _validate_executor_config(config_path)
            roles.append({'role': 'executor', 'executor_id': row['executor_id'],
                'argv': [self.deployment.python, '-B', '-m', 'qwen_rl', 'gradients', 'executor', '--config', str(config_path)],
                'environment_names': sorted(safe_names | {row['token_env'], worker['auth_token_env']}),
                'log_path': str(run / f"{row['executor_id']}.log")})
            sync.append({'executor_id': row['executor_id'], 'token_env': row['token_env'],
                         'server_url': server_url, 'worker_ids': [worker['worker_id']]})
        if runtime['bepa']['enabled']:
            teacher = runtime['bepa']['teacher']
            roles.append({'role': 'teacher', 'argv': [self.deployment.python, '-B', '-m', 'qwen_rl', 'runtime',
                'teacher', '--config', str(runtime_path), '--worker-id', 'teacher-instance'],
                'environment_names': sorted(safe_names | {'ARC_ENV_TOKEN', teacher['api_key_env'], teacher['token_env']}),
                'log_path': str(run / 'teacher.log')})
        path = run / 'gradient-plan.json'
        argv = [self.deployment.python, '-B', '-m', 'qwen_rl.deployment.local_gradients', '_supervise', '--plan', str(path)]
        _write_private_json(path, {'schema': 'qwen-rl-local-gradient-plan/v1', 'deployment_id': plan['deployment']['id'],
            'run_id': run.name, 'config_fingerprint': hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            'source_config_fingerprint': plan['config_fingerprint'],
            'manifest_path': str(run / 'gradient-manifest.json'), 'supervisor_argv': argv,
            'workdir': str(Path(__file__).resolve().parents[2]),
            'server_ready_timeout_seconds': self.deployment.start_timeout,
            'role_stop_timeout_seconds': self.deployment.stop_timeout,
            'server_health': {'url': server_url + '/health', 'token_env': 'ARC_ENV_TOKEN'},
            'resume_url': server_url + '/resume',
            'resume_ledger_path': str(self.deployment.state_dir / 'resume-ledger.json'),
            'request_timeout_seconds': runtime['learner']['request_timeout'],
            'initial_policy_sync': sync, 'worker_pins': pins,
            'secret_environment_names': plan['secret_names'], 'roles': roles})
        return path

    def serve(self, *, preflight_receipt: Path | None = None) -> int:
        from .instance_migration import assert_not_migrating

        hold_path = self.deployment.state_dir / 'stop-hold.json'
        hold = _read_private_json(hold_path)
        if hold is not None and hold.get('schema') != _STOP_HOLD_SCHEMA:
            raise InstanceDeploymentError('instance stop hold receipt is unsupported')
        if hold is not None and preflight_receipt is None:
            # PID 1 stays alive without holding owner.lock or starting any role.
            # A resume child can start only through consume_handoff below.
            self._idle_hold()
            return 0
        assert_not_migrating(self.deployment)
        plan = self.prepare()
        state = self.deployment.state_dir
        _mkdir(state)
        with (state / 'owner.lock').open('a') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise InstanceDeploymentError('this instance deployment is already supervised') from None
            assert_not_migrating(self.deployment)
            if preflight_receipt is not None:
                result = self._serve_locked(plan, preflight_receipt=preflight_receipt)
            else:
                result = self._serve_locked(plan)
        if self._read_stop_hold() is not None:
            keep_idle_guard = preflight_receipt is None or os.getpid() == 1
            self._demote_active_owner(keep_idle_guard=keep_idle_guard)
            if keep_idle_guard:
                self._idle_hold()
        return result

    def _read_stop_hold(self) -> dict[str, Any] | None:
        hold = _read_private_json(self.deployment.state_dir / 'stop-hold.json')
        if hold is not None and hold.get('schema') != _STOP_HOLD_SCHEMA:
            raise InstanceDeploymentError('instance stop hold receipt is unsupported')
        return hold

    def _demote_active_owner(self, *, keep_idle_guard: bool) -> None:
        manifest = _read_private_json(self.active_path)
        if not manifest or manifest.get('schema') != _SCHEMA:
            return
        owner = manifest.get('owner')
        if isinstance(owner, dict) and _owner_matches(owner) and owner.get('pid') == os.getpid():
            manifest['stopped_owner'] = dict(owner)
            prior_guard = manifest.get('idle_guard')
            if keep_idle_guard and (not isinstance(prior_guard, dict) or not _owner_matches(prior_guard)
                                    or prior_guard.get('pid') == owner.get('pid')):
                manifest['idle_guard'] = {
                    key: owner[key] for key in ('pid', 'starttime', 'argv') if key in owner
                }
            if isinstance(manifest.get('idle_guard'), dict):
                manifest['idle_guard']['status'] = ('alive_without_owner_lock'
                    if _owner_matches(manifest['idle_guard']) and
                    (keep_idle_guard or manifest['idle_guard'].get('pid') != owner.get('pid'))
                    else 'exited')
            manifest['owner'] = None
            manifest['status'] = 'stopped'
            _write_private_json(self.active_path, manifest)

    def _idle_hold(self) -> None:
        stop_requested = []
        old_handlers = {}

        def request_stop(signum, _frame):
            stop_requested.append(signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, request_stop)
        try:
            while not stop_requested:
                time.sleep(1.0)
            active = _read_private_json(self.active_path)
            owner = active.get('owner') if active and active.get('schema') == _SCHEMA else None
            if isinstance(owner, dict) and owner.get('pid') != os.getpid() and _owner_matches(owner):
                from .instance_stop_control import _pidfd_open, _send
                descriptor = _pidfd_open(owner)
                try:
                    if descriptor >= 0:
                        _send(descriptor, signal.SIGTERM)
                        deadline = time.monotonic() + max(0.05, self.deployment.stop_timeout * 0.9)
                        while time.monotonic() < deadline:
                            current = _read_private_json(self.active_path)
                            current_owner = current.get('owner') if current else None
                            if (not isinstance(current_owner, dict)
                                    or not _owner_matches(current_owner)):
                                break
                            time.sleep(0.2)
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)

    def _serve_locked(self, plan: dict, *, preflight_receipt: Path | None = None) -> int:
        if self.status()['status'] != 'stopped':
            raise InstanceDeploymentError('owned instance roles remain active; stop them before serving')
        from .instance_preflight import InstancePreflightError, preflight

        configure_cache_environment(self.deployment)
        if preflight_receipt is not None:
            from .instance_startup import consume_handoff
            checked = consume_handoff(self, plan, preflight_receipt)
        else:
            checked = preflight(self, plan=plan)
        if checked.get('status') != 'passed':
            raise InstancePreflightError(checked)
        for binary in (self.deployment.python, self.deployment.worker_python, self.deployment.engine_binary):
            if not Path(binary).is_file() or not os.access(binary, os.X_OK):
                raise InstanceDeploymentError('instance runtime executable is unavailable; use the unified instance image')
        ports = [plan['runtime']['learner']['port'], *[r[k] for r in plan['workers'] for k in ('port', 'engine_port')]]
        if self.deployment.gateway_config is not None:
            ports.append(8310)
        if self.deployment.dashboard_port is not None:
            ports.append(self.deployment.dashboard_port)
        for port in ports:
            _port_free(port)
        _initialize_caches()
        self._prune_stopped_run_policy_caches()
        run = self.deployment.state_dir / 'runs' / uuid.uuid4().hex
        _mkdir(run)
        from .instance_stop_control import rebind_pending_checkpoint_stop
        rebind_pending_checkpoint_stop(self, run.name)
        secrets = self._credentials(plan, run)
        env = {k: v for k, v in os.environ.items() if k in _SAFE_INHERITED_ENV}
        env.update(secrets)
        env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   ARC_RL_ROOT=str(Path(__file__).resolve().parents[2]), PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        plan['secret_names'] = sorted(secrets)
        observer_build = read_runtime_build_identity(Path(__file__).resolve().parents[1])
        manifest = {'schema': _SCHEMA, 'status': 'starting', 'run_id': run.name,
                    'config_fingerprint': plan['config_fingerprint'],
                    'owner': {'pid': os.getpid(), 'starttime': _proc_record(os.getpid())[0],
                              'argv': list(_proc_record(os.getpid())[1])}, 'roles': [],
                    'observer_build': observer_build}
        children = []
        gradient = None
        old_handlers = {}
        def interrupted(_signum, _frame):
            self.stopping = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, interrupted)
        try:
            _write_private_json(self.active_path, manifest)
            for row in plan['workers']:
                directory = run / row['worker_id']
                _mkdir(directory)
                raw = copy.deepcopy(row['worker_config'])
                _mkdir(Path(raw['adapter_cache']).parent)
                config_path = directory / 'worker.json'
                _write_private_json(config_path, raw)
                child_env = {k: v for k, v in env.items() if k not in secrets}
                child_env.update(CUDA_VISIBLE_DEVICES=row['gpu_uuid'], GPU_UUID=row['gpu_uuid'],
                                 QWEN_SITE_ID=row['site_id'], QWEN_NODE_ID=row['node_id'])
                argv = [self.deployment.worker_python, '-B', '-m', 'qwen_rl.worker.ninfer_launcher',
                    '--worker-config', str(config_path), '--token-file', row['auth_token_path'],
                    '--engine-binary', str(self.deployment.engine_binary), '--worker-python', self.deployment.worker_python,
                    '--gpu-uuid', row['gpu_uuid'], '--engine-port', str(row['engine_port']),
                    '--worker-host', '127.0.0.1', '--worker-port', str(row['port']),
                    '--ready-timeout', str(self.deployment.start_timeout)]
                child, record = _spawn(argv, child_env, directory / 'launcher.log', cwd=run)
                record.update(role='worker', worker_id=row['worker_id'], gpu_uuid=row['gpu_uuid'])
                children.append((child, record));manifest['roles'].append(record)
                _write_private_json(self.active_path, manifest)
            pins = self._pin_workers(plan, children, env)
            gradient_path = self._gradient_plan(plan, run, env, pins)
            manifest.update(runtime_config=str(run / 'runtime.yaml'), gradient_manifest=str(run / 'gradient-manifest.json'))
            if self.deployment.gateway_config is not None:
                argv = [self.deployment.worker_python, '-B', '-m', 'qwen_rl', 'cluster', 'gateway',
                        '--config', str(self.deployment.gateway_config)]
                child, record = _spawn(argv, {k: v for k, v in env.items() if k not in secrets}, run / 'gateway.log', cwd=run)
                record.update(role='gateway'); children.append((child, record));manifest['roles'].append(record)
            gradient_argv = [self.deployment.python, '-B', '-m', 'qwen_rl.deployment.local_gradients', '_supervise', '--plan', str(gradient_path)]
            child, record = _spawn(gradient_argv, env, run / 'supervisor.log', cwd=run)
            gradient = (child, record)
            record.update(role='gradient-supervisor');children.append(gradient);manifest['roles'].append(record)
            _write_private_json(self.active_path, manifest)
            dashboard_started = False
            ready_deadline = time.monotonic() + self.deployment.start_timeout
            next_stop_poll = time.monotonic()
            while not self.stopping:
                if any(child.poll() is not None and record.get('role') != 'dashboard' for child, record in children):
                    raise InstanceDeploymentError('an owned role exited; inspect instance logs')
                nested = _read_private_json(run / 'gradient-manifest.json')
                if nested and nested.get('status') in {'failed', 'stop_incomplete', 'stopped'}:
                    raise InstanceDeploymentError(
                        nested.get('error_message') or 'gradient supervisor stopped unexpectedly')
                gateway_ready = True
                if self.deployment.gateway_config is not None:
                    try:
                        health = _http_json('http://127.0.0.1:8310/health', '')
                        gateway_ready = health.get('role') == 'gateway' and health.get('ready') is True
                    except (OSError, ValueError, RuntimeError):
                        gateway_ready = False
                if (nested and nested.get('status') == 'running' and not dashboard_started
                        and self.deployment.dashboard_port is not None):
                    dashboard_env = {k: v for k, v in env.items() if k not in secrets or k == 'ARC_ENV_TOKEN'}
                    argv = [self.deployment.python, '-B', '-m', 'qwen_rl', 'dashboard',
                            '--source', 'runtime', '--deployment', str(run / 'runtime.yaml'),
                            '--instance-manifest', str(run / 'gradient-manifest.json'),
                            '--host', self.deployment.dashboard_host, '--port', str(self.deployment.dashboard_port)]
                    dashboard, dashboard_record = _spawn(argv, dashboard_env, run / 'dashboard.log', cwd=run)
                    dashboard_record.update(role='dashboard')
                    children.append((dashboard, dashboard_record))
                    manifest['roles'].append(dashboard_record)
                    dashboard_started = True
                    _write_private_json(self.active_path, manifest)
                if nested and nested.get('status') == 'running' and gateway_ready and manifest['status'] != 'running':
                    manifest['status'] = 'running'
                    _write_private_json(self.active_path, manifest)
                if manifest['status'] != 'running' and time.monotonic() >= ready_deadline:
                    raise InstanceDeploymentError('instance roles did not finish readiness before timeout')
                if time.monotonic() >= next_stop_poll:
                    next_stop_poll = time.monotonic() + 1.0
                    if self._poll_checkpoint_stop(run, manifest):
                        self.stopping = True
                        break
                time.sleep(0.2)
        except BaseException as exc:
            manifest.update(status='failed', error=type(exc).__name__)
            if isinstance(exc, InstanceDeploymentError):
                manifest['error_message'] = str(exc)
            _write_private_json(self.active_path, manifest)
        finally:
            if self.stopping:
                manifest["status"] = "stopping"
                _write_private_json(self.active_path, manifest)
            # Drain central commits while Native Worker endpoints are still alive.
            writers = [(child, record) for child, record in children
                       if record.get('role') == 'gradient-supervisor']
            peers = [(child, record) for child, record in reversed(children)
                     if record.get('role') != 'gradient-supervisor']
            clean = _drain_role_group(writers, self.deployment.stop_timeout)
            if writers and not _gradient_drained(run / 'gradient-manifest.json'):
                clean = False
            if clean:
                clean = _drain_role_group(peers, self.deployment.stop_timeout)
            manifest['status'] = ('stopped' if self.stopping and clean else
                                  'stop_incomplete' if not clean else 'failed')
            _write_private_json(self.active_path, manifest)
            stop_operation = _read_private_json(self.deployment.state_dir / 'stop-operation.json')
            if (self.stopping and stop_operation is not None
                    and stop_operation.get('schema') == _STOP_OPERATION_SCHEMA
                    and stop_operation.get('mode') == 'after_checkpoint'
                    and stop_operation.get('manager_run_id') == run.name):
                stop_operation.update(
                    manager_status='completed' if manifest['status'] == 'stopped' else 'blocked',
                    phase='managed_roles_stopped' if manifest['status'] == 'stopped' else 'managed_shutdown_incomplete',
                    manager_completed_at=time.time(), updated_at=time.time(),
                    next_command=shlex.join(['qwen-rl', 'instance', '--deployment', str(self.deployment.path), 'resume']),
                    resume_command=shlex.join(['qwen-rl', 'instance', '--deployment', str(self.deployment.path), 'resume']),
                )
                stop_operation['selected_checkpoint'] = {
                    key: stop_operation[key] for key in (
                        'checkpoint_snapshot_id', 'checkpoint_generation', 'checkpoint_id',
                        'snapshot_id', 'generation', 'target_update', 'target_policy_version',
                        'completed_update', 'durable_policy_version',
                    )
                    if key in stop_operation
                }
                stop_operation['rollback'] = None
                _write_private_json(self.deployment.state_dir / 'stop-operation.json', stop_operation)
            if manifest['status'] == 'stopped':
                self._prune_stopped_run_policy_caches(allow_current_owner=True)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return 0 if manifest['status'] == 'stopped' else 1
