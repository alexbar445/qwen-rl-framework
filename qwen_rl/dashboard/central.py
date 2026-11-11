"""Read-only binding of central-gradient control receipts to dashboard telemetry."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time


def is_central(runtime):
    return isinstance(runtime, dict) and runtime.get('learner', {}).get('update_backend') == 'central_gradient'


def container_bindings(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('invalid dashboard controller name')
    result = subprocess.run(['docker', 'inspect', '--type', 'container', name],
                            capture_output=True, timeout=5, check=True)
    if len(result.stdout) > 2 * 1024 * 1024:
        raise ValueError('controller inspect response exceeds limit')
    rows = json.loads(result.stdout)
    if len(rows) != 1 or rows[0].get('Name', '').lstrip('/') != name:
        raise ValueError('controller identity differs from deployment')
    item = rows[0]
    mounts = []
    for mount in item.get('Mounts', []):
        if mount.get('Type') != 'bind':
            continue
        host, destination = Path(mount['Source']), PurePosixPath(mount['Destination'])
        if host.is_absolute() and destination.is_absolute() and str(destination) != '/' and '..' not in destination.parts:
            mounts.append((destination, host))
    return item['Id'], tuple(mounts)


def _read(path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            return None
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _control(root, profile):
    if root.is_symlink() or (root / 'runs').is_symlink() or profile.is_symlink():
        return None
    pointer = _read(root / 'active.json')
    if not pointer or pointer.get('schema') != 'qwen-rl-local-gradient-run/v1':
        return None
    if pointer.get('source_config_fingerprint') != hashlib.sha256(profile.read_bytes()).hexdigest():
        return None
    run_id = pointer.get('run_id')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        return None
    path = root / 'runs' / run_id / 'manifest.json'
    if path.parent.is_symlink() or path.parent.parent.is_symlink():
        return None
    manifest = _read(path)
    if (not manifest or manifest.get('schema') != 'qwen-rl-local-gradient-run/v1'
            or manifest.get('run_id') != run_id
            or manifest.get('deployment_id') != root.parent.name
            or manifest.get('config_fingerprint') != pointer.get('config_fingerprint')):
        return None
    return {**manifest, '_path': path}


def find_registry(project, profile):
    base = project / 'outputs' / 'local-worker-services'
    matches = []
    if base.is_dir() and not base.is_symlink():
        for fleet in sorted(base.iterdir())[:128]:
            if fleet.is_symlink() or not fleet.is_dir():
                continue
            root = fleet / 'gradient-runs'
            if root.is_symlink():
                continue
            if _control(root, profile) is not None:
                matches.append(root)
    if len(matches) != 1:
        raise ValueError('central dashboard requires exactly one matching control registry')
    return matches[0]


def current_control(source):
    if source.runtime_registry_root is None or source.profile_config_path is None:
        return None
    return _control(source.runtime_registry_root, source.profile_config_path)


def bind_snapshot(source, snapshot, expected_run_id):
    """Project the real control generation; retain the Learner's null manager ID."""
    control = current_control(source)
    if not control or control.get('run_id') != expected_run_id or control.get('status') != 'running':
        raise ValueError('central control generation is no longer running')
    health = snapshot.get('learner_health') or {}
    learner = snapshot.get('learner') or {}
    epoch = learner.get('telemetry_epoch')
    expected_base = ((source.runtime.get('artifact_identity') or {}).get('source') or {}).get('ninfer_artifact', {}).get('sha256')
    worker_base = (source.runtime.get('worker_inference') or {}).get('ninfer_artifact_sha256')
    if worker_base is not None and worker_base != expected_base:
        raise ValueError('Worker base declaration differs from the canonical artifact identity')
    if (not source.controller_id or not isinstance(epoch, str) or not epoch
            or not expected_base or health.get('base_fingerprint') != expected_base
            or health.get('role') != 'learner'):
        raise ValueError('central live telemetry differs from the selected controller/base')
    result = deepcopy(snapshot)
    result['lifecycle'] = {'status': control['status'], 'run_id': control['run_id'],
                           'manifest': str(control['_path']), 'backend': 'central_gradient'}
    result['owned_source_verified'] = True
    result['source_identity'] = {'container_id': source.controller_id,
        'profile_sha256': hashlib.sha256(source.profile_config_path.read_bytes()).hexdigest(),
        'telemetry_epoch': epoch, 'control_run_id': control['run_id']}
    result['execution_telemetry'] = executor_telemetry(source, control, result)
    return result

_OPERATION_SCHEMA = 'qwen-rl-operation-telemetry/v1'
_OPERATION_FRESH_SECONDS = 5.0
_OPERATION_MAX_BYTES = 128 * 1024
_SAFE_EXECUTOR = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}')
_SAFE_WORKER = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}')
_SAFE_RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}')
_SHA256 = re.compile(r'[0-9a-f]{64}')
_LINEAGE_ID = re.compile(r'[0-9a-f]{32}')
def _executor_id(value):
    return value if isinstance(value, str) and _SAFE_EXECUTOR.fullmatch(value) else None


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _safe_child(base, path):
    base, path = Path(base), Path(path)
    try:
        if base.is_symlink():
            return False
        relative = path.relative_to(base)
        cursor = base
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return False
        return path.resolve(strict=False).is_relative_to(base.resolve(strict=False))
    except (OSError, ValueError):
        return False


def _read_control_json(base, path, *, max_bytes):
    path = Path(path)
    try:
        if (not _safe_child(base, path) or path.is_symlink() or not path.is_file()):
            return None
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            return None
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, ValueError):
        return None


def _read_private_control_json(base, path, *, max_bytes, with_sha256=False):
    path = Path(path)
    try:
        if not _safe_child(base, path) or path.is_symlink():
            return None
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            info = os.fstat(descriptor)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                    or info.st_size > max_bytes):
                return None
            with os.fdopen(descriptor, 'rb', closefd=False) as handle:
                raw = handle.read(max_bytes + 1)
        finally:
            os.close(descriptor)
        if len(raw) > max_bytes:
            return None
        value = json.loads(raw.decode('utf-8'))
        if not isinstance(value, dict):
            return None
        return (value, hashlib.sha256(raw).hexdigest()) if with_sha256 else value
    except (OSError, UnicodeError, ValueError):
        return None


def _argv_config_path(record, flag):
    argv = record.get('argv') if isinstance(record, dict) else None
    if (not isinstance(argv, list) or any(not isinstance(item, str) for item in argv)
            or argv.count(flag) != 1):
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        return None
    return Path(argv[index + 1])


def current_instance(source):
    """Verify the explicit in-container instance generation and its config binding."""
    from ..deployment.local_gradients import _record_matches

    config_path = source.runtime_config_path
    manifest_path = source.instance_manifest_path
    if (config_path is None or manifest_path is None
            or config_path.name != 'runtime.yaml'
            or manifest_path.name != 'gradient-manifest.json'
            or config_path.resolve(strict=False).parent != manifest_path.resolve(strict=False).parent):
        return None
    base = config_path.parent
    if base.is_symlink() or not _safe_child(base, config_path) or not _safe_child(base, manifest_path):
        return None
    state_root = base.parent.parent
    if (base.parent.name != 'runs' or any(path.is_symlink() or not path.is_dir()
            for path in (state_root, base.parent, base))):
        return None
    try:
        config_fd = os.open(config_path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            config_info = os.fstat(config_fd)
            if (not stat.S_ISREG(config_info.st_mode) or config_info.st_uid != os.getuid()
                    or stat.S_IMODE(config_info.st_mode) != 0o600 or config_info.st_nlink != 1
                    or config_info.st_size > 2 * 1024 * 1024):
                return None
            with os.fdopen(config_fd, 'rb', closefd=False) as handle:
                config_bytes = handle.read(2 * 1024 * 1024 + 1)
        finally:
            os.close(config_fd)
        if len(config_bytes) > 2 * 1024 * 1024:
            return None
        config_sha = hashlib.sha256(config_bytes).hexdigest()
    except OSError:
        return None
    if config_sha != source.deployment_id:
        return None

    loaded = _read_private_control_json(base, manifest_path, max_bytes=2 * 1024 * 1024,
                                        with_sha256=True)
    if loaded is None:
        return None
    manifest, manifest_sha = loaded
    if (not manifest or manifest.get('schema') != 'qwen-rl-local-gradient-run/v1'
            or manifest.get('status') != 'running'
            or not isinstance(manifest.get('run_id'), str)
            or not _SAFE_RUN_ID.fullmatch(manifest['run_id'])
            or not isinstance(manifest.get('deployment_id'), str)
            or not re.fullmatch(r'[0-9a-f]{20}', manifest['deployment_id'])
            or not _SHA256.fullmatch(manifest.get('config_fingerprint', ''))
            or manifest.get('config_fingerprint') != config_sha
            or not _SHA256.fullmatch(manifest.get('source_config_fingerprint', ''))):
        return None

    plan_path = base / 'gradient-plan.json'
    supervisor = manifest.get('supervisor')
    plan_arg = _argv_config_path(supervisor, '--plan')
    if (plan_arg is None or plan_arg.resolve(strict=False) != plan_path.resolve(strict=False)
            or not _record_matches(supervisor)):
        return None
    plan = _read_private_control_json(base, plan_path, max_bytes=2 * 1024 * 1024)
    if (not plan or plan.get('schema') != 'qwen-rl-local-gradient-plan/v1'
            or plan.get('run_id') != manifest['run_id']
            or plan.get('deployment_id') != manifest['deployment_id']
            or plan.get('config_fingerprint') != config_sha
            or plan.get('source_config_fingerprint') != manifest['source_config_fingerprint']
            or Path(str(plan.get('manifest_path', ''))).resolve(strict=False)
               != manifest_path.resolve(strict=False)
            or plan.get('supervisor_argv') != supervisor.get('argv')):
        return None

    runtime = source.runtime if isinstance(source.runtime, dict) else {}
    learner = runtime.get('learner') if isinstance(runtime.get('learner'), dict) else {}
    executor_map = learner.get('gradient_executor_worker_ids')
    if not isinstance(executor_map, dict) or not executor_map:
        return None
    expected_executors = set(executor_map)
    roles = manifest.get('roles')
    if not isinstance(roles, list) or not 1 <= len(roles) <= 16:
        return None
    server_roles = [row for row in roles if isinstance(row, dict) and row.get('role') == 'server']
    executor_roles = [row for row in roles if isinstance(row, dict) and row.get('role') == 'executor']
    planned_roles = plan.get('roles')
    if not isinstance(planned_roles, list) or len(planned_roles) != len(roles):
        return None
    planned_by_key = {}
    for row in planned_roles:
        if not isinstance(row, dict) or row.get('role') not in {'server', 'executor', 'teacher'}:
            return None
        key = (row.get('role'), row.get('executor_id'))
        if key in planned_by_key:
            return None
        planned_by_key[key] = row
    for row in roles:
        if not isinstance(row, dict):
            return None
        key = (row.get('role'), row.get('executor_id'))
        planned = planned_by_key.get(key)
        if (planned is None or row.get('argv') != planned.get('argv')
                or row.get('environment_names') != planned.get('environment_names')
                or row.get('log_file') != planned.get('log_path')):
            return None
    observed_executors = [row.get('executor_id') for row in executor_roles]
    teacher_roles = [row for row in roles if isinstance(row, dict) and row.get('role') == 'teacher']
    bepa = runtime.get('bepa') if isinstance(runtime.get('bepa'), dict) else {}
    if (len(server_roles) != 1 or len(observed_executors) != len(set(observed_executors))
            or set(observed_executors) != expected_executors
            or len(teacher_roles) != int(bepa.get('enabled') is True)
            or len(roles) != 1 + len(executor_roles) + len(teacher_roles)):
        return None
    server_config = _argv_config_path(server_roles[0], '--config')
    if (server_config is None or server_config.resolve(strict=False) != config_path.resolve(strict=False)):
        return None
    if any((config_arg := _argv_config_path(row, '--config')) is None
           or config_arg.resolve(strict=False) != config_path.resolve(strict=False)
           for row in teacher_roles):
        return None
    if any(row.get('status') != 'running' or not _record_matches(row) for row in roles):
        return None

    configured_workers = runtime.get('workers')
    pins = manifest.get('worker_pins')
    if not isinstance(configured_workers, list) or not isinstance(pins, list):
        return None
    expected_pins = {}
    for worker in configured_workers:
        if (not isinstance(worker, dict) or not isinstance(worker.get('id'), str)
                or not _SAFE_WORKER.fullmatch(worker['id'])
                or not isinstance(worker.get('telemetry_epoch'), str) or not worker['telemetry_epoch']
                or not _SHA256.fullmatch(worker.get('model_sha256', ''))):
            return None
        expected_pins[worker['id']] = (worker.get('telemetry_epoch'), worker.get('model_sha256'))
    observed_pins = {}
    for pin in pins:
        if (not isinstance(pin, dict) or not isinstance(pin.get('worker_id'), str)
                or not _SAFE_WORKER.fullmatch(pin['worker_id'])):
            return None
        observed_pins[pin['worker_id']] = (pin.get('telemetry_epoch'), pin.get('model_sha256'))
    if (len(expected_pins) != len(configured_workers) or expected_pins != observed_pins
            or len(observed_pins) != len(pins)):
        return None
    return {**manifest, '_path': manifest_path, '_plan': plan_path,
            '_runtime_config_sha256': config_sha,
            '_gradient_manifest_sha256': manifest_sha}


def bind_instance_snapshot(source, snapshot, expected_run_id):
    """Bind direct Runtime Watch data to the current in-container instance generation."""
    control = current_instance(source)
    if not control:
        raise ValueError('instance manifest is not a verified live generation')
    if (expected_run_id is not None and
            (not isinstance(expected_run_id, str) or not _SAFE_RUN_ID.fullmatch(expected_run_id))):
        raise ValueError('selected logical run id is invalid')
    if not isinstance(snapshot, dict):
        raise ValueError('runtime Watch snapshot is unavailable')
    runtime = source.runtime if isinstance(source.runtime, dict) else {}
    artifact_identity = runtime.get('artifact_identity')
    identity_source = artifact_identity.get('source') if isinstance(artifact_identity, dict) else None
    ninfer_identity = identity_source.get('ninfer_artifact') if isinstance(identity_source, dict) else None
    expected_base = ninfer_identity.get('sha256') if isinstance(ninfer_identity, dict) else None
    worker_inference = runtime.get('worker_inference')
    worker_base = (worker_inference.get('ninfer_artifact_sha256')
                   if isinstance(worker_inference, dict) else None)
    if (not isinstance(expected_base, str) or not _SHA256.fullmatch(expected_base)
            or (worker_base is not None and worker_base != expected_base)):
        raise ValueError('Worker base declaration differs from the canonical artifact identity')
    health = snapshot.get('learner_health')
    health = health if isinstance(health, dict) else {}
    learner = snapshot.get('learner')
    learner = learner if isinstance(learner, dict) else {}
    freshness = snapshot.get('freshness')
    freshness = freshness if isinstance(freshness, dict) else {}
    if (not isinstance(freshness.get('learner'), dict)
            or freshness['learner'].get('live') is not True
            or not isinstance(freshness.get('learner_health'), dict)
            or freshness['learner_health'].get('live') is not True):
        raise ValueError('instance Learner metadata is not fresh')
    sampled_at = snapshot.get('sampled_at')
    if (not _finite_number(sampled_at) or time.time() - float(sampled_at) > 5.0
            or float(sampled_at) > time.time() + 1.0):
        raise ValueError('instance watch snapshot is stale')
    lineage = learner.get('recovery') if isinstance(learner.get('recovery'), dict) else {}
    epoch = learner.get('telemetry_epoch')
    if (not expected_base or health.get('base_fingerprint') != expected_base
            or health.get('role') != 'learner'
            or not isinstance(epoch, str) or not epoch
            or not isinstance(lineage.get('lineage_id'), str)
            or not _LINEAGE_ID.fullmatch(lineage['lineage_id'])
            or not isinstance(lineage.get('engine_epoch'), str) or not lineage['engine_epoch']):
        raise ValueError('instance live telemetry differs from the selected runtime/base/lineage')

    result = deepcopy(snapshot)
    lifecycle = result.get('lifecycle')
    if not isinstance(lifecycle, dict):
        lifecycle = {}
    run = learner.get('run')
    run = run if isinstance(run, dict) else {}
    monitor_run_id = run.get('run_id')
    if monitor_run_id is not None:
        if (not isinstance(monitor_run_id, str) or not _SAFE_RUN_ID.fullmatch(monitor_run_id)
                or not isinstance(run.get('status'), str) or not run['status']
                or not isinstance(run.get('kind'), str) or not run['kind']):
            raise ValueError('Learner monitor run identity is malformed')
    elif run.get('status') == 'running':
        raise ValueError('running Learner monitor has no logical run id')
    training = learner.get('training')
    training = training if isinstance(training, dict) else {}
    committed = training.get('run_committed')
    committed = committed if isinstance(committed, dict) else {}
    committed_run_id = committed.get('run_id')
    if committed_run_id is not None and (
            not isinstance(committed_run_id, str) or not _SAFE_RUN_ID.fullmatch(committed_run_id)):
        raise ValueError('Learner committed-run identity is malformed')
    if monitor_run_id is not None and committed_run_id is not None and monitor_run_id != committed_run_id:
        raise ValueError('Learner run and committed-run identities differ')
    reported_runtime_id = learner.get('runtime_run_id')
    if reported_runtime_id is not None and (
            not isinstance(reported_runtime_id, str) or not _SAFE_RUN_ID.fullmatch(reported_runtime_id)):
        raise ValueError('runtime monitor identity is malformed')
    reported_watch_id = lifecycle.get('run_id')
    if reported_watch_id is not None and (
            not isinstance(reported_watch_id, str) or not _SAFE_RUN_ID.fullmatch(reported_watch_id)):
        raise ValueError('runtime Watch identity is malformed')
    logical_run_id = monitor_run_id or committed_run_id
    if logical_run_id is None:
        logical_run_id = reported_runtime_id or reported_watch_id
    for reported in (reported_runtime_id, reported_watch_id):
        if reported is not None and logical_run_id is not None and reported != logical_run_id:
            raise ValueError('runtime Watch identity differs from the Learner monitor run')
    if (expected_run_id is not None and logical_run_id is not None
            and logical_run_id != expected_run_id):
        raise ValueError('runtime Learner run differs from the selected logical run')
    if expected_run_id is not None and logical_run_id is None:
        raise ValueError('selected logical run is absent from the current Learner monitor')
    lifecycle.update(status='running', run_id=logical_run_id, backend='central_gradient',
                     instance_run_id=control['run_id'])
    result['lifecycle'] = lifecycle
    result['owned_source_verified'] = True
    result['source_identity'] = {
        'instance_run_id': control['run_id'],
        'runtime_config_sha256': control['_runtime_config_sha256'],
        'gradient_manifest_sha256': control['_gradient_manifest_sha256'],
        'telemetry_epoch': epoch,
        'logical_run_id': logical_run_id,
    }
    return result


def _kernel_boot_id():
    try:
        value = Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip().lower()
    except (OSError, UnicodeError):
        return None
    return value if re.fullmatch(r'[0-9a-f-]{36}', value) else None


def _identity_tuple(value):
    if not isinstance(value, dict):
        return None
    owner = _executor_id(value.get('owner_executor_id'))
    lineage = value.get('lineage_id')
    epoch = value.get('engine_epoch')
    if (owner is None or not isinstance(lineage, str) or not lineage
            or not isinstance(epoch, str) or not epoch):
        return None
    for field in ('window_id', 'attempt_id', 'job_id', 'group_id'):
        if not isinstance(value.get(field), str) or not value[field]:
            return None
    policy_version = value.get('policy_version')
    if type(policy_version) is not int or policy_version < 0:
        return None
    logical_run_id = value.get('logical_run_id')
    if logical_run_id is not None and (not isinstance(logical_run_id, str) or not logical_run_id):
        return None
    return (owner, lineage, epoch, value['window_id'], value['attempt_id'], value['job_id'],
            value['group_id'], policy_version, logical_run_id)


def _identity_matches(identity, *, executor_id, lineage_id, engine_epoch, logical_run_id):
    key = _identity_tuple(identity)
    if key is None or key[0] != executor_id or key[1] != lineage_id or key[2] != engine_epoch:
        return False
    if logical_run_id is not None and key[8] != logical_run_id:
        return False
    return True


def _unknown_measurements():
    return {'forward': None, 'backward': None}


def _executor_row(executor_id, *, status='unknown', reason='telemetry unavailable'):
    return {
        'executor_id': executor_id, 'status': status, 'reason': reason,
        'sample_age_seconds': None, 'current_phase': None, 'phase_elapsed_seconds': None,
        'active_job': None, 'measurements': _unknown_measurements(),
    }


def _execution_status(rows, reason=None):
    states = [row.get('status') for row in rows]
    if not states:
        status = 'unknown'
    elif all(state == 'idle' for state in states):
        status = 'idle'
    elif all(state in {'active', 'waiting', 'idle'} for state in states):
        status = 'live'
    elif all(state == 'stale' for state in states):
        status = 'stale'
    elif all(state == 'unknown' for state in states):
        status = 'unknown'
    else:
        status = 'partial'
    return {
        'schema': 'qwen-rl-dashboard-executor-microbatch/v1',
        'source_schema': _OPERATION_SCHEMA, 'status': status, 'reason': reason,
        'executors': rows,
    }


def _configured_local_executors(runtime):
    learner = runtime.get('learner') if isinstance(runtime, dict) else None
    if not isinstance(learner, dict):
        return None
    mapping = learner.get('gradient_executor_worker_ids')
    local_workers = learner.get('gradient_local_executor_worker_ids')
    if not isinstance(mapping, dict) or not isinstance(local_workers, (list, tuple)):
        return None
    if any(not isinstance(worker, str) or not worker for worker in local_workers):
        return None
    if len(local_workers) != len(set(local_workers)):
        return None
    if not set(local_workers).issubset(set(mapping.values())):
        return None
    if any(_executor_id(executor) is None or not isinstance(worker, str) or not worker
           for executor, worker in mapping.items()):
        return None
    local = set(local_workers)
    return {executor for executor, worker in mapping.items() if worker in local}


def _project_last_completed(entry, phase, *, executor_id, lineage_id, engine_epoch,
                            logical_run_id, active_job, sample_at, sample_age, now,
                            series_epoch):
    if not isinstance(entry, dict):
        return None
    identity = entry.get('identity')
    if not _identity_matches(identity, executor_id=executor_id, lineage_id=lineage_id,
                              engine_epoch=engine_epoch, logical_run_id=logical_run_id):
        return None
    if active_job is not None and _identity_tuple(identity) != _identity_tuple(active_job):
        return None
    tokens = entry.get('input_tokens')
    elapsed = entry.get('cuda_elapsed_seconds')
    reported_rate = entry.get('tokens_per_second')
    completed_at = entry.get('completed_at_unix_seconds')
    index = entry.get('microbatch_index')
    sequence = entry.get('sequence')
    if (type(tokens) is not int or tokens < 0 or not _finite_number(elapsed) or elapsed <= 0
            or not _finite_number(reported_rate) or reported_rate < 0
            or not _finite_number(completed_at) or completed_at > sample_at + 1.0
            or completed_at > now + 1.0 or type(index) is not int or index < 0
            or type(sequence) is not int or sequence < 0):
        return None
    try:
        measured_rate = tokens / float(elapsed)
    except (OverflowError, ZeroDivisionError):
        return None
    if (not math.isfinite(measured_rate)
            or not math.isclose(float(reported_rate), measured_rate, rel_tol=0.01, abs_tol=0.001)):
        return None
    age = max(0.0, now - float(completed_at))
    state = 'stale' if sample_age > _OPERATION_FRESH_SECONDS else (
        'current_job' if active_job is not None else 'last_completed_idle')
    identity_epoch = hashlib.sha256(json.dumps(
        [series_epoch, _identity_tuple(identity)], separators=(',', ':')).encode('utf-8')).hexdigest()[:20]
    return {
        'tokens_per_second': measured_rate, 'input_tokens': tokens,
        'cuda_elapsed_seconds': float(elapsed),
        'completed_at_unix_seconds': float(completed_at), 'age_seconds': age,
        'microbatch_index': index, 'sequence': sequence, 'phase': phase,
        'measurement': 'cuda_event', 'status': state,
        'series_epoch': identity_epoch,
    }


def _process_identity_matches(process, role, boot_id):
    """Compare the producer's /proc decimal ticks with the control manifest."""
    return (isinstance(process, dict) and type(process.get('pid')) is int
            and isinstance(process.get('process_start_time_ticks'), str)
            and process.get('pid') == role['pid']
            and process.get('process_start_time_ticks') == str(role['starttime'])
            and process.get('boot_id') == boot_id)


def executor_telemetry(source, control, snapshot, *, now=None):
    """Project only current-control CUDA-event microbatch measurements; never sum executors."""
    now = time.time() if now is None else float(now)
    control_id = control.get('run_id') if isinstance(control, dict) else None
    run_id = control_id if isinstance(control_id, str) else ''
    run_dir = Path(source.runtime_registry_root) / 'runs' / run_id
    expected = _configured_local_executors(source.runtime or {})
    if expected is None:
        return _execution_status([], 'current control executor mapping is unavailable')
    if not expected:
        return _execution_status([], 'no local gradient executors are configured')
    health = snapshot.get('learner_health') if isinstance(snapshot, dict) else None
    recovery = health.get('recovery') if isinstance(health, dict) else None
    lineage_id = recovery.get('lineage_id') if isinstance(recovery, dict) else None
    engine_epoch = recovery.get('engine_epoch') if isinstance(recovery, dict) else None
    if (not isinstance(lineage_id, str) or not lineage_id
            or not isinstance(engine_epoch, str) or not engine_epoch):
        return _execution_status([_executor_row(key) for key in sorted(expected)],
                                 'current central lineage/engine epoch is unavailable')
    learner = snapshot.get('learner') if isinstance(snapshot.get('learner'), dict) else {}
    training = learner.get('training') if isinstance(learner.get('training'), dict) else {}
    logical_run = training.get('run_committed') if isinstance(training.get('run_committed'), dict) else {}
    logical_run_id = logical_run.get('run_id')
    if not isinstance(logical_run_id, str) or not logical_run_id:
        logical_run_id = None
    current_policy = health.get('policy_version') if isinstance(health, dict) else None
    if type(current_policy) is not int or current_policy < 0:
        current_policy = None
    boot_id = _kernel_boot_id()
    if boot_id is None:
        return _execution_status([_executor_row(key) for key in sorted(expected)],
                                 'current controller process epoch is unavailable')
    roles = control.get('roles') if isinstance(control, dict) else None
    if not isinstance(roles, list):
        return _execution_status([_executor_row(key) for key in sorted(expected)],
                                 'current control process manifest is unavailable')
    role_map = {}
    malformed_roles = False
    for role in roles[:128]:
        if not isinstance(role, dict) or role.get('role') != 'executor':
            continue
        executor_id = _executor_id(role.get('executor_id'))
        if executor_id is None or executor_id in role_map:
            malformed_roles = True
            break
        role_map[executor_id] = role
    if malformed_roles or set(role_map) != expected:
        return _execution_status([_executor_row(key, reason='control executor roster mismatch')
                                  for key in sorted(expected)],
                                 'current control executor roster differs from runtime config')

    rows = []
    for executor_id in sorted(expected):
        role = role_map[executor_id]
        row = _executor_row(executor_id)
        rows.append(row)
        if (role.get('status') != 'running' or type(role.get('pid')) is not int
                or role.get('pid') <= 1 or type(role.get('starttime')) is not int
                or role.get('starttime') <= 0):
            row['reason'] = 'executor role is not current/running'
            continue
        argv = role.get('argv')
        if (not isinstance(argv, list) or any(not isinstance(item, str) for item in argv)
                or argv.count('--config') != 1):
            row['reason'] = 'executor process manifest has no safe config path'
            continue
        config_pos = argv.index('--config')
        if config_pos + 1 >= len(argv):
            row['reason'] = 'executor process manifest config path is incomplete'
            continue
        executor_dir = run_dir / 'executors' / executor_id
        config_path = executor_dir / 'executor.json'
        if not _safe_child(run_dir, config_path):
            row['reason'] = 'executor config path is unsafe'
            continue
        try:
            role_config_path = source.map_path(argv[config_pos + 1])
        except (OSError, ValueError):
            role_config_path = None
        if role_config_path is None or role_config_path.resolve(strict=False) != config_path.resolve(strict=False):
            row['reason'] = 'executor role config differs from current control directory'
            continue
        executor_config = _read_control_json(run_dir, config_path, max_bytes=64 * 1024)
        profile_path = executor_dir / 'compute-profile.json'
        if (not isinstance(executor_config, dict)
                or executor_config.get('schema_version') != 'qwen-rl-gradient-executor/v1'
                or executor_config.get('executor_id') != executor_id):
            row['reason'] = 'current executor config is missing or mismatched'
            continue
        try:
            configured_work = source.map_path(executor_config.get('working_dir'))
            configured_profile = source.map_path(executor_config.get('compute_profile'))
        except (OSError, ValueError):
            configured_work, configured_profile = None, None
        if (configured_work is None or configured_profile is None
                or configured_work.resolve(strict=False) != executor_dir.resolve(strict=False)
                or configured_profile.resolve(strict=False) != profile_path.resolve(strict=False)):
            row['reason'] = 'executor working/profile paths differ from current control directory'
            continue
        profile = _read_control_json(run_dir, profile_path, max_bytes=256 * 1024)
        if not isinstance(profile, dict) or profile.get('schema_version') != 'qwen-rl-ternary-compute/v1':
            row['reason'] = 'current compute profile is missing or invalid'
            continue
        try:
            profile_work = source.map_path(profile.get('run_dir'))
        except (OSError, ValueError):
            profile_work = None
        if profile_work is None or profile_work.resolve(strict=False) != executor_dir.resolve(strict=False):
            row['reason'] = 'compute profile run_dir differs from its executor directory'
            continue
        compute_dir = executor_dir / 'compute'
        operation_path = compute_dir / 'operation-telemetry.json'
        telemetry = _read_control_json(run_dir, operation_path, max_bytes=_OPERATION_MAX_BYTES)
        if not isinstance(telemetry, dict) or telemetry.get('schema') != _OPERATION_SCHEMA:
            row['reason'] = 'operation telemetry missing or schema unknown'
            continue
        process = telemetry.get('process_identity')
        if not _process_identity_matches(process, role, boot_id):
            row['reason'] = 'operation telemetry process identity differs from current control role'
            continue
        sampled_at = telemetry.get('sampled_at_unix_seconds')
        if not _finite_number(sampled_at) or sampled_at > now + 1.0:
            row['reason'] = 'operation telemetry timestamp is invalid'
            continue
        sample_age = max(0.0, now - float(sampled_at))
        sample_fresh = sample_age <= _OPERATION_FRESH_SECONDS
        active_job = telemetry.get('active_job')
        if active_job is not None:
            if (not _identity_matches(active_job, executor_id=executor_id, lineage_id=lineage_id,
                                      engine_epoch=engine_epoch, logical_run_id=logical_run_id)
                    or (current_policy is not None and active_job.get('policy_version') != current_policy)):
                row['reason'] = 'active executor job differs from the current central lineage/run/policy'
                continue
        active_phase = telemetry.get('active_phase')
        phase = None
        phase_index = None
        phase_elapsed = None
        if active_job is None:
            if active_phase is not None:
                row['reason'] = 'idle executor still reports an active phase'
                continue
        elif active_phase is not None:
            if (not isinstance(active_phase, dict)
                    or active_phase.get('name') not in {'forward', 'backward'}
                    or type(active_phase.get('microbatch_index')) is not int
                    or active_phase['microbatch_index'] < 0):
                row['reason'] = 'active executor phase is malformed'
                continue
            phase = active_phase['name']
            phase_index = active_phase['microbatch_index']
            if _finite_number(active_phase.get('elapsed_wall_seconds')):
                phase_elapsed = float(active_phase['elapsed_wall_seconds'])
        row['status'] = ('stale' if not sample_fresh else
                         ('idle' if active_job is None else ('active' if phase else 'waiting')))
        row['reason'] = None if sample_fresh else 'operation telemetry sample is stale'
        row['sample_age_seconds'] = sample_age
        row['reported_phase'] = {'name': phase, 'microbatch_index': phase_index} if phase else None
        row['current_phase'] = (
            {'name': phase, 'microbatch_index': phase_index}
            if sample_fresh and phase else None
        )
        row['phase_elapsed_seconds'] = phase_elapsed if sample_fresh else None
        row['active_job'] = None if active_job is None else {
            key: active_job.get(key) for key in ('job_id', 'window_id', 'attempt_id',
                'group_id', 'policy_version', 'logical_run_id')
        }
        last_completed = telemetry.get('last_completed')
        if not isinstance(last_completed, dict):
            continue
        process_epoch = hashlib.sha256(json.dumps([source.controller_id, executor_id, role['pid'],
            role['starttime'], boot_id], separators=(',', ':')).encode('utf-8')).hexdigest()[:20]
        for name in ('forward', 'backward'):
            row['measurements'][name] = _project_last_completed(
                last_completed.get(name), name, executor_id=executor_id, lineage_id=lineage_id,
                engine_epoch=engine_epoch, logical_run_id=logical_run_id, active_job=active_job,
                sample_at=float(sampled_at), sample_age=sample_age, now=now, series_epoch=process_epoch,
            )
    return _execution_status(rows)


def watch_command(source, environment, arguments):
    """Run the observer in the pinned controller; never load codecs on the host."""
    from ..deployment.local import LocalDeployment, load_deployment

    config = load_deployment(source.deployment_path)
    from .sources import PROJECT_ROOT
    project = PROJECT_ROOT
    deployment = LocalDeployment(project, config, environ=environment)
    credentials = deployment._load_credentials(teacher=False, create_arc=False)
    child_env = deployment._base_child_env({**credentials, 'PYTHONUNBUFFERED': '1'})

    def inside(path):
        resolved = path.resolve(strict=True)
        for container, host in sorted(source.mounts, key=lambda row: len(row[1].parts), reverse=True):
            base = host.resolve(strict=False)
            if resolved.is_relative_to(base):
                return str(container / PurePosixPath(resolved.relative_to(base).as_posix()))
        raise ValueError('central observer file is outside verified controller mounts')

    env_file = source.runtime_config_path.parent / 'runtime-workers.env'
    if env_file.is_symlink() or not env_file.is_file():
        raise ValueError('central Worker credential environment is unavailable')
    command = ['docker', 'exec', '-i', '-u', config.container_uid, '-w', config.container_workdir]
    for name in [*config.runtime_env, 'ARC_ENV_TOKEN', 'PYTHONUNBUFFERED']:
        command.extend(['-e', name])
    command.extend([config.container, config.container_python, '-B', '-m', 'qwen_rl',
        'runtime', 'watch', '--config', inside(source.runtime_config_path),
        '--environment-file', inside(env_file), *arguments])
    return command, child_env
