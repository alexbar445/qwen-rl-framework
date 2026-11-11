"""Persistent manager stop controls for one deployment instance."""
from __future__ import annotations

import errno
import ctypes
import hashlib
import math
import os
from pathlib import Path
import shlex
import signal
import sys
import time
import uuid
from typing import Any

from .local_gradients import (
    _http_request_json, _proc_record, _read_private_json, _write_private_json,
)

STOP_OPERATION_SCHEMA = 'qwen-rl-instance-stop-operation/v1'
STOP_HOLD_SCHEMA = 'qwen-rl-instance-stop-hold/v1'
STOP_NOW_PROOF_SCHEMA = 'qwen-rl-instance-stop-now-proof/v1'
_FREEZE_SECONDS = 3.0
_CORE_TERM_SECONDS = 3.0
_ENDPOINT_TERM_SECONDS = 5.0
_TOTAL_SECONDS = 20.0


def _error(message: str):
    from .instance import InstanceDeploymentError
    return InstanceDeploymentError(message)


def _operation_path(supervisor) -> Path:
    return supervisor.deployment.state_dir / 'stop-operation.json'


def _hold_path(supervisor) -> Path:
    return supervisor.deployment.state_dir / 'stop-hold.json'


def _deployment_id(supervisor) -> str:
    return hashlib.sha256(str(supervisor.deployment.path).encode()).hexdigest()[:20]


def _resume_command(supervisor) -> str:
    return shlex.join(['qwen-rl', 'instance', '--deployment', str(supervisor.deployment.path), 'resume'])


def _write_hold(supervisor, operation: dict[str, Any]) -> None:
    _write_private_json(_hold_path(supervisor), {
        'schema': STOP_HOLD_SCHEMA, 'operation_id': operation['operation_id'],
        'mode': operation['mode'], 'deployment_id': operation['deployment_id'],
        'manager_run_id': operation.get('manager_run_id'), 'created_at': time.time(),
        'policy': 'hold_roles_until_explicit_resume',
    })


def clear_stop_hold_locked(supervisor) -> None:
    """Clear the persistent stop fence while the caller holds operation.lock."""
    path = _hold_path(supervisor)
    hold = _read_private_json(path)
    if hold is None:
        return
    if hold.get('schema') != STOP_HOLD_SCHEMA:
        raise _error('instance stop hold receipt is unsupported')
    path.unlink()
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def rebind_pending_checkpoint_stop(supervisor, manager_run_id: str) -> None:
    """Bind a persisted Learner latch to the explicitly resumed manager owner."""
    operation_path = _operation_path(supervisor)
    hold_path = _hold_path(supervisor)
    operation = _read_private_json(operation_path)
    hold = _read_private_json(hold_path)
    if (operation is None or hold is None
            or operation.get('schema') != STOP_OPERATION_SCHEMA
            or hold.get('schema') != STOP_HOLD_SCHEMA
            or operation.get('mode') != 'after_checkpoint'
            or hold.get('mode') != 'after_checkpoint'
            or hold.get('operation_id') != operation.get('operation_id')
            or operation.get('manager_status') not in {
                'requested', 'accepted', 'waiting_for_checkpoint', 'retrying',
                'learner_completed', 'pending',
            }):
        return
    previous = operation.get('manager_run_id')
    operation.update(manager_run_id=manager_run_id,
                     rebound_from_manager_run_id=previous,
                     phase='resumed_manager_bound', updated_at=time.time())
    hold['manager_run_id'] = manager_run_id
    hold['rebound_at'] = time.time()
    _write_private_json(operation_path, operation)
    _write_private_json(hold_path, hold)


def _merge_learner_receipt(operation: dict[str, Any], receipt: dict[str, Any]) -> None:
    if (receipt.get('operation_id') != operation.get('operation_id')
            or receipt.get('mode') != 'after_checkpoint'
            or receipt.get('status') not in {'accepted', 'waiting_durable', 'completed', 'blocked'}
            or type(receipt.get('request_accepted')) is not bool):
        raise _error('Learner stop endpoint returned a mismatched operation receipt')
    learner_run_id = receipt.get('run_id')
    if learner_run_id is not None and not isinstance(learner_run_id, str):
        raise _error('Learner stop endpoint returned an invalid logical run identity')
    if operation.get('learner_run_id') is not None and learner_run_id != operation['learner_run_id']:
        raise _error('Learner stop operation changed its bound logical run')
    target = receipt.get('target_update')
    if operation.get('target_update') is not None and target != operation['target_update']:
        raise _error('Learner stop operation changed its checkpoint target')
    for key in ('target_policy_version', 'lineage_id', 'engine_epoch'):
        if operation.get(key) is not None and receipt.get(key) != operation[key]:
            raise _error(f'Learner stop operation changed its bound {key}')


def _record_receipt(operation: dict[str, Any], receipt: dict[str, Any]) -> None:
    _merge_learner_receipt(operation, receipt)
    operation['learner_receipt'] = receipt
    operation['learner_run_id'] = receipt.get('run_id')
    for key in ('target_update', 'completed_update', 'target_policy_version',
                'durable_policy_version', 'checkpoint_snapshot_id', 'checkpoint_generation',
                'checkpoint_id', 'snapshot_id', 'generation', 'lineage_id', 'engine_epoch',
                'request_accepted'):
        if key in receipt:
            operation[key] = receipt[key]
    operation['updated_at'] = time.time()


def stop_after_checkpoint(supervisor, *, timeout_seconds: float = 30.0,
                          progress=None) -> dict[str, Any]:
    """Persist one authenticated Learner latch, then observe its durable result."""
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise _error('stop-after-checkpoint timeout must be a positive finite number')
    from .instance_operations import operation_lock

    path = _operation_path(supervisor)
    next_command = _resume_command(supervisor)
    wait_command = shlex.join(['qwen-rl', 'instance', '--deployment',
                               str(supervisor.deployment.path), 'stop-after-checkpoint',
                               '--timeout', str(timeout_seconds)])
    with operation_lock(supervisor.deployment.state_dir):
        state = supervisor.status()
        active = _read_private_json(supervisor.active_path)
        manager_run_id = active.get('run_id') if active else state.get('run_id')
        prior = _read_private_json(path)
        if (prior and prior.get('mode') == 'after_checkpoint'
                and prior.get('manager_run_id') == manager_run_id
                and not (prior.get('manager_status') == 'blocked'
                         and (prior.get('learner_receipt') or {}).get('request_accepted') is False)):
            operation = prior
        else:
            if state.get('status') != 'running' or not isinstance(manager_run_id, str):
                if prior and prior.get('mode') == 'after_checkpoint' and prior.get('manager_status') == 'completed':
                    return prior
                raise _error('stop-after-checkpoint requires a running managed instance')
            if (prior and prior.get('mode') == 'now'
                    and prior.get('manager_status') not in {'completed', 'blocked'}):
                raise _error('stop-now operation is still active; inspect its receipt before requesting a boundary stop')
            operation = {
                'schema': STOP_OPERATION_SCHEMA, 'operation_id': uuid.uuid4().hex,
                'mode': 'after_checkpoint', 'manager_status': 'requested',
                'phase': 'request_persisted', 'deployment_id': _deployment_id(supervisor),
                'manager_run_id': manager_run_id, 'learner_run_id': None,
                'learner_receipt': None, 'selected_snapshot': None,
                'next_command': wait_command, 'resume_command': next_command,
                'created_at': time.time(), 'updated_at': time.time(),
            }
            if prior and prior.get('operation_id'):
                operation['supersedes_operation_id'] = prior['operation_id']
            _write_private_json(path, operation)
        _write_hold(supervisor, operation)
        if operation.get('manager_status') not in {'completed', 'blocked'} and not operation.get('learner_receipt'):
            config, token, url = supervisor._control_context()
            receipt, _headers = _http_request_json(
                url + '/control/stop', token, method='POST',
                body={'operation_id': operation['operation_id'], 'mode': 'after_checkpoint'},
                timeout=min(float(config['learner']['request_timeout']), 30.0),
            )
            _record_receipt(operation, receipt)
            operation['manager_status'] = ('blocked' if receipt['status'] == 'blocked'
                                           else 'waiting_for_checkpoint')
            operation['phase'] = ('learner_blocked' if receipt['status'] == 'blocked'
                                  else 'learner_latch_accepted')
            if receipt['status'] == 'blocked':
                operation['next_command'] = shlex.join([
                    'qwen-rl', 'instance', '--deployment', str(supervisor.deployment.path),
                    'stop-now' if receipt['request_accepted'] else 'status',
                    *([] if receipt['request_accepted'] else ['--human']),
                ])
                if not receipt['request_accepted']:
                    clear_stop_hold_locked(supervisor)
            _write_private_json(path, operation)

    started = time.monotonic()
    next_report = 0.0
    while True:
        current = _read_private_json(path) or operation
        if current.get('manager_status') in {'completed', 'blocked'}:
            return current
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            current = dict(current)
            current.update(manager_status='pending', phase='waiting_for_durable_checkpoint',
                           wait_timed_out=True, wait_seconds=round(elapsed, 3),
                           next_command=wait_command, resume_command=next_command)
            return current
        if progress is not None and elapsed >= next_report:
            progress({'stage': current.get('phase', 'waiting_for_checkpoint'),
                      'status': current.get('manager_status', 'pending'),
                      'elapsed_seconds': round(elapsed, 1),
                      'target_update': current.get('target_update')})
            next_report = elapsed + 5.0
        time.sleep(min(0.25, max(0.0, timeout_seconds - elapsed)))


def poll_checkpoint_stop(supervisor, run: Path, manifest: dict[str, Any]) -> bool:
    path = _operation_path(supervisor)
    operation = _read_private_json(path)
    if (operation is None or operation.get('schema') != STOP_OPERATION_SCHEMA
            or operation.get('mode') != 'after_checkpoint'
            or operation.get('manager_run_id') != run.name
            or operation.get('manager_status') in {'completed', 'blocked'}):
        return False
    from .instance_operations import operation_lock
    try:
        with operation_lock(supervisor.deployment.state_dir):
            operation = _read_private_json(path) or operation
            if operation.get('manager_status') in {'completed', 'blocked'}:
                return False
            config, token, url = supervisor._control_context()
            if operation.get('learner_receipt'):
                receipt, _headers = _http_request_json(
                    url + '/control/stop/' + str(operation['operation_id']), token,
                    method='GET', timeout=min(float(config['learner']['request_timeout']), 5.0),
                )
            else:
                receipt, _headers = _http_request_json(
                    url + '/control/stop', token, method='POST',
                    body={'operation_id': operation['operation_id'], 'mode': 'after_checkpoint'},
                    timeout=min(float(config['learner']['request_timeout']), 5.0),
                )
            _record_receipt(operation, receipt)
            operation['last_error_type'] = None
            if receipt['status'] == 'blocked':
                operation.update(
                    manager_status='blocked', phase='learner_blocked',
                    next_command=shlex.join(['qwen-rl', 'instance', '--deployment',
                                             str(supervisor.deployment.path),
                                             *(['stop-now'] if receipt['request_accepted'] else ['status', '--human'])]),
                )
                if not receipt['request_accepted']:
                    clear_stop_hold_locked(supervisor)
            elif receipt['status'] == 'completed':
                operation.update(manager_status='learner_completed', phase='learner_checkpoint_durable')
            else:
                operation.update(manager_status='waiting_for_checkpoint', phase='waiting_for_durable_checkpoint')
            _write_private_json(path, operation)
            if receipt['status'] == 'completed':
                manifest['status'] = 'stopping'
                manifest['stop_operation_id'] = operation['operation_id']
                _write_private_json(supervisor.active_path, manifest)
                return True
            return False
    except BlockingIOError:
        return False
    except Exception as exc:
        operation['manager_status'] = 'retrying'
        operation['phase'] = 'learner_status_retry'
        operation['last_error_type'] = type(exc).__name__
        operation['updated_at'] = time.time()
        try:
            _write_private_json(path, operation)
        except Exception:
            pass
        return False


def _proc_details(pid: int, proc_root: Path = Path('/proc')) -> dict[str, Any] | None:
    try:
        stat_text = (proc_root / str(pid) / 'stat').read_text(encoding='ascii')
        cmdline = (proc_root / str(pid) / 'cmdline').read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _error(f'cannot inspect owned process tree ({type(exc).__name__})') from None
    close = stat_text.rfind(')')
    if close < 0:
        raise _error('owned process tree has an unreadable proc identity')
    fields = stat_text[close + 1:].split()
    if len(fields) <= 19:
        raise _error('owned process tree has an incomplete proc identity')
    try:
        parent_pid, starttime = int(fields[1]), int(fields[19])
    except ValueError:
        raise _error('owned process tree has an invalid proc identity') from None
    argv = [part.decode('utf-8', 'replace') for part in cmdline.split(b'\0') if part]
    return {'pid': pid, 'starttime': starttime, 'argv': argv,
            'parent_pid': parent_pid, 'state': fields[0]}


def _proc_stat(pid: int, proc_root: Path = Path('/proc')) -> dict[str, Any] | None:
    try:
        stat_text = (proc_root / str(pid) / 'stat').read_text(encoding='ascii')
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _error(f'cannot scan owned process ancestry ({type(exc).__name__})') from None
    close = stat_text.rfind(')')
    if close < 0:
        raise _error('owned process tree has unreadable ancestry')
    fields = stat_text[close + 1:].split()
    if len(fields) <= 19:
        raise _error('owned process tree has incomplete ancestry')
    try:
        return {'pid': pid, 'parent_pid': int(fields[1]), 'starttime': int(fields[19]), 'state': fields[0]}
    except ValueError:
        raise _error('owned process tree has invalid ancestry') from None


def _env_value(pid: int, name: str, proc_root: Path = Path('/proc')) -> str | None:
    try:
        values = (proc_root / str(pid) / 'environ').read_bytes().split(b'\0')
    except OSError as exc:
        raise _error(f'cannot verify native child GPU ownership ({type(exc).__name__})') from None
    prefix = name.encode('ascii') + b'='
    for value in values:
        if value.startswith(prefix):
            return value[len(prefix):].decode('utf-8', 'replace')
    return None


def _mark_terminal_descendant(record: dict[str, Any], detail: dict[str, Any]) -> None:
    if (record.get('argv') == [] and detail.get('argv') == []
            and detail.get('state') in {'Z', 'X'}
            and detail.get('starttime') == record.get('starttime')
            and detail.get('parent_pid') == record.get('parent_pid')):
        record.update(terminal_state=detail['state'], terminal_identity_verified=True)


def _pidfd_open(record: dict[str, Any]) -> int:
    open_pidfd = getattr(os, 'pidfd_open', None)
    if not _pidfd_signal_available():
        raise _error('stop-now requires Linux pidfd signal support')
    pid = record.get('pid')
    if type(pid) is not int or pid <= 1:
        raise _error('owned process identity is invalid')
    try:
        descriptor = open_pidfd(pid, 0) if open_pidfd is not None else _pidfd_open_syscall(pid)
    except ProcessLookupError:
        return -1
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return -1
        raise _error(f'cannot pin owned process identity ({type(exc).__name__})') from None
    detail = _proc_details(pid)
    if detail is None or detail['starttime'] != record.get('starttime'):
        os.close(descriptor)
        return -1
    if detail['state'] in {'Z', 'X'}:
        _mark_terminal_descendant(record, detail)
        os.close(descriptor)
        return -1
    if tuple(detail['argv']) != tuple(record.get('argv', ())):
        os.close(descriptor)
        raise _error('owned process command changed before stop-now')
    return descriptor


def _pidfd_signal_available() -> bool:
    if getattr(signal, 'pidfd_send_signal', None) is not None:
        return True
    if not sys.platform.startswith('linux'):
        return False
    try:
        getattr(ctypes.CDLL(None), 'pidfd_send_signal')
        return True
    except AttributeError:
        return True  # Linux pidfd_send_signal syscall number is stable across supported architectures.


def _pidfd_open_syscall(pid: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return int(result)


def _send(fd: int, sig: int) -> None:
    if fd < 0:
        return
    try:
        sender = getattr(signal, 'pidfd_send_signal', None)
        if sender is not None:
            sender(fd, sig)
            return
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            sender = libc.pidfd_send_signal
        except AttributeError:
            sender = libc.syscall
            sender.restype = ctypes.c_long
            result = sender(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(sig),
                            ctypes.c_void_p(), ctypes.c_uint(0))
        else:
            sender.restype = ctypes.c_int
            result = sender(fd, sig, None, 0)
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.ESRCH:
                return
            raise OSError(error_number, os.strerror(error_number))
    except ProcessLookupError:
        return
    except OSError as exc:
        raise _error(f'cannot signal pinned owned process ({type(exc).__name__})') from None


def _scan_tree(roots: list[dict[str, Any]], worker_gpu_uuids: dict[int, str]) -> list[dict[str, Any]]:
    proc_root = Path('/proc')
    try:
        pids = [int(path.name) for path in proc_root.iterdir() if path.name.isdecimal()]
    except OSError as exc:
        raise _error(f'cannot scan owned process tree ({type(exc).__name__})') from None
    ancestry = {}
    for pid in pids:
        row = _proc_stat(pid, proc_root)
        if row is not None:
            ancestry[pid] = row
    captured: dict[int, dict[str, Any]] = {}
    roots_by_pid = {int(row['pid']): row for row in roots}
    for root in roots:
        detail = _proc_details(int(root['pid']), proc_root)
        if detail is None or (detail['starttime'], tuple(detail['argv'])) != (
                root.get('starttime'), tuple(root.get('argv', ()))):
            raise _error('owned process changed during stable tree capture')
        captured[int(root['pid'])] = {
            **root, **detail, 'root_pid': int(root['pid']),
            'root_starttime': detail['starttime'], 'root_role': root.get('role'),
        }
    changed = True
    while changed:
        changed = False
        for pid, stat_row in ancestry.items():
            parent = captured.get(stat_row['parent_pid'])
            if parent is None or pid in captured:
                continue
            detail = _proc_details(pid, proc_root)
            if detail is None:
                continue
            if (detail['starttime'], detail['parent_pid']) != (stat_row['starttime'], stat_row['parent_pid']):
                raise _error('owned process ancestry changed during tree capture')
            expected_uuid = worker_gpu_uuids.get(parent['root_pid'])
            gpu_uuid_verified = False
            row = {**detail, 'root_pid': parent['root_pid'], 'root_role': parent.get('role'),
                   'role': parent.get('role'), 'worker_id': parent.get('worker_id'),
                   'executor_id': parent.get('executor_id'), 'gpu_uuid': expected_uuid,
                   'root_starttime': parent.get('root_starttime', parent['starttime']),
                   'parent_starttime': parent['starttime']}
            if expected_uuid:
                env_uuid = _env_value(pid, 'GPU_UUID', proc_root)
                cuda_uuid = _env_value(pid, 'CUDA_VISIBLE_DEVICES', proc_root)
                if env_uuid not in {None, expected_uuid} or cuda_uuid not in {None, expected_uuid}:
                    raise _error('Worker descendant GPU UUID disagrees with its owned Worker')
                native = any('ninfer-serve' in Path(arg).name or 'residency_exec' in arg for arg in row['argv'])
                if native and (env_uuid != expected_uuid or cuda_uuid != expected_uuid):
                    raise _error('native Worker child GPU UUID ownership is unknown')
                gpu_uuid_verified = env_uuid == expected_uuid and cuda_uuid == expected_uuid
            row['gpu_uuid_verified'] = gpu_uuid_verified
            _mark_terminal_descendant(row, detail)
            captured[pid] = row
            changed = True
    if any(row['root_pid'] not in roots_by_pid for row in captured.values()):
        raise _error('owned process tree root is invalid')
    return list(captured.values())


def _identity_state(record: dict[str, Any]) -> str:
    detail = _proc_details(int(record['pid']))
    if detail is None or detail['starttime'] != record['starttime'] or detail['state'] in {'Z', 'X'}:
        return 'exited'
    if tuple(detail['argv']) != tuple(record['argv']):
        expected = hashlib.sha256(repr(tuple(record['argv'])).encode()).hexdigest()[:12]
        current = hashlib.sha256(repr(tuple(detail['argv'])).encode()).hexdigest()[:12]
        role = record.get('root_role', record.get('role', 'process'))
        raise _error(f'owned {role} PID {record["pid"]} command identity changed ({expected}/{current})')
    return 'live'


def _read_managed_roots(supervisor, active: dict[str, Any]):
    from .instance import _SCHEMA

    if active.get('schema') != _SCHEMA or not isinstance(active.get('run_id'), str):
        raise _error('instance ownership manifest is missing or unsupported')
    owner = active.get('owner')
    top = active.get('roles')
    if owner is not None and not isinstance(owner, dict):
        raise _error('instance owner identity is malformed')
    if not isinstance(top, list) or any(not isinstance(row, dict) for row in top):
        raise _error('instance role identity inventory is malformed')
    records = []
    if owner:
        records.append({**owner, 'role': 'instance-supervisor'})
    top_records = [dict(row) for row in top]
    records.extend(top_records)
    nested_path = active.get('gradient_manifest')
    nested = None
    nested_absent = False
    if nested_path:
        expected = supervisor.deployment.state_dir / 'runs' / active['run_id'] / 'gradient-manifest.json'
        try:
            if Path(nested_path).resolve(strict=True) != expected.resolve(strict=False):
                raise _error('nested gradient manifest path is outside the selected run')
        except OSError:
            raise _error('nested gradient manifest path cannot be verified') from None
        nested = _read_private_json(Path(nested_path))
        top_gradient = next((row for row in top_records if row.get('role') == 'gradient-supervisor'), None)
        if nested is None:
            nested_absent = True
        else:
            if nested.get('schema') != 'qwen-rl-local-gradient-run/v1':
                raise _error('nested gradient process inventory is unsupported')
            gradient_supervisor = nested.get('supervisor')
            if not isinstance(gradient_supervisor, dict):
                raise _error('nested gradient supervisor identity is unavailable')
            if top_gradient and any(top_gradient.get(key) != gradient_supervisor.get(key)
                                    for key in ('pid', 'starttime', 'argv')):
                raise _error('top-level and nested gradient supervisor identities disagree')
            if not top_gradient:
                records.append({**gradient_supervisor, 'role': 'gradient-supervisor'})
            nested_roles = nested.get('roles')
            if not isinstance(nested_roles, list) or any(not isinstance(row, dict) for row in nested_roles):
                raise _error('nested gradient role identity inventory is malformed')
            records.extend(dict(row) for row in nested_roles)
    unique: dict[tuple[int, int, tuple[str, ...]], dict[str, Any]] = {}
    for row in records:
        pid, starttime, argv = row.get('pid'), row.get('starttime'), row.get('argv')
        if (type(pid) is not int or pid <= 1 or type(starttime) is not int
                or not isinstance(argv, list) or not argv):
            raise _error('managed process identity is incomplete')
        if any(not isinstance(arg, str) for arg in argv):
            raise _error('managed process command identity is malformed')
        key = (pid, starttime, tuple(argv))
        unique.setdefault(key, row)
    return list(unique.values()), top_records, nested_path, nested, nested_absent


def _block(supervisor, operation: dict[str, Any], phase: str, exc: Exception, *, persist: bool = True):
    path = _operation_path(supervisor)
    operation.update(manager_status='blocked', phase=phase,
                     error_type=type(exc).__name__, detail=str(exc)[:1000], updated_at=time.time(),
                     next_command=shlex.join(['qwen-rl', 'instance', '--deployment',
                                              str(supervisor.deployment.path), 'status', '--human']),
                     resume_command=_resume_command(supervisor))
    if persist:
        from .instance_operations import operation_lock
        try:
            with operation_lock(supervisor.deployment.state_dir):
                _write_private_json(path, operation)
        except BlockingIOError:
            pass
    from .instance import InstanceDeploymentError
    error = InstanceDeploymentError(str(exc))
    error.report = {
        'schema': 'qwen-rl-instance-stop/v1', 'status': 'pending',
        'reason_code': 'stop_now_incomplete', 'field': phase,
        'error_type': type(exc).__name__, 'detail': str(exc)[:1000],
        'operation_id': operation.get('operation_id'), 'phase': phase,
        'next_command': operation['next_command'], 'resume_command': operation['resume_command'],
    }
    raise error from None


def stop_now(supervisor, *, progress=None) -> dict[str, Any]:
    """Stop only the recorded instance process trees, then discard uncommitted work."""
    from .instance_operations import operation_lock
    from .instance import _SCHEMA, _write_private_json

    path = _operation_path(supervisor)
    next_command = shlex.join(['qwen-rl', 'instance', '--deployment',
                               str(supervisor.deployment.path), 'status', '--human'])
    resume_command = _resume_command(supervisor)
    operation = None
    phase = 'operation_lock'
    process_release_started = time.monotonic()
    stop_deadline = process_release_started + _TOTAL_SECONDS
    descriptors: dict[tuple[int, int, tuple[str, ...]], int] = {}
    frozen: set[tuple[int, int, tuple[str, ...]]] = set()

    def emit(stage: str, **fields: Any) -> None:
        if progress:
            progress({'stage': stage, **fields})

    def signal_record(key, sig):
        fd = descriptors.get(key, -1)
        _send(fd, sig)

    def state_for(key, row):
        if key in descriptors:
            detail = _proc_details(int(row['pid']))
            if detail is None or detail['starttime'] != row['starttime'] or detail['state'] in {'Z', 'X'}:
                state = 'exited'
            else:
                # pidfd pins the process selected while its saved argv matched;
                # a later exec with the same starttime is still that owned task.
                if tuple(detail['argv']) != tuple(row['argv']):
                    row['command_changed_after_pin'] = True
                state = 'live'
        else:
            state = _identity_state(row)
        if state != 'live':
            frozen.discard(key)
        return state

    def release_stopped():
        for key in tuple(frozen):
            try:
                signal_record(key, signal.SIGCONT)
            except Exception:
                pass
            frozen.discard(key)

    try:
        with operation_lock(supervisor.deployment.state_dir):
            active = _read_private_json(supervisor.active_path)
            if active is None or active.get('schema') != _SCHEMA:
                raise _error('active instance manifest is absent or unsupported')
            manager_run_id = active.get('run_id')
            prior = _read_private_json(path)
            if (prior and prior.get('mode') == 'now' and prior.get('manager_run_id') == manager_run_id
                    and prior.get('manager_status') == 'completed'):
                operation = prior
                proof = prior.get('proof')
                if (active.get('status') != 'stopped' or active.get('owner') is not None
                        or not isinstance(proof, dict) or proof.get('status') != 'stopped'
                        or proof.get('schema') != STOP_NOW_PROOF_SCHEMA
                        or proof.get('operation_id') != prior.get('operation_id')
                        or proof.get('run_id') != manager_run_id):
                    raise _error('completed stop-now receipt does not match the stopped instance')
                current_records, *_ = _read_managed_roots(supervisor, active)
                if any(_identity_state(row) == 'live' for row in current_records):
                    raise _error('a completed stop-now receipt has live managed roles')
                prior.pop('error_type', None)
                prior.pop('detail', None)
                _write_private_json(path, prior)
                return proof
            if (prior and prior.get('mode') == 'now' and prior.get('manager_run_id') == manager_run_id
                    and prior.get('manager_status') == 'blocked'):
                operation = prior
            else:
                operation = {
                    'schema': STOP_OPERATION_SCHEMA, 'operation_id': uuid.uuid4().hex,
                    'mode': 'now', 'manager_status': 'stopping', 'phase': 'intent_persisted',
                    'deployment_id': _deployment_id(supervisor), 'manager_run_id': manager_run_id,
                    'selected_snapshot': [], 'next_command': next_command,
                    'resume_command': resume_command, 'created_at': time.time(), 'updated_at': time.time(),
                }
                if prior and prior.get('operation_id'):
                    operation['supersedes_operation_id'] = prior['operation_id']
                    if prior.get('mode') == 'after_checkpoint':
                        prior_learner = prior.get('learner_receipt') or {}
                        run_id = prior_learner.get('run_id', prior.get('learner_run_id'))
                        target_update = prior_learner.get('target_update', prior.get('target_update'))
                        lineage_id = prior_learner.get('lineage_id', prior.get('lineage_id'))
                        engine_epoch = prior_learner.get('engine_epoch', prior.get('engine_epoch'))
                        target_policy = prior_learner.get('target_policy_version', prior.get('target_policy_version'))
                        if (prior_learner.get('request_accepted') is True
                                and prior_learner.get('status') in {'accepted', 'waiting_durable', 'blocked'}
                                and isinstance(run_id, str) and run_id
                                and type(target_update) is int
                                and isinstance(lineage_id, str) and lineage_id
                                and isinstance(engine_epoch, str) and engine_epoch
                                and type(target_policy) is int):
                            operation.update(
                                supersedes_learner_stop=True,
                                superseded_learner_run_id=run_id,
                                superseded_target_update=target_update,
                                superseded_lineage_id=lineage_id,
                                superseded_engine_epoch=engine_epoch,
                                superseded_target_policy_version=target_policy,
                            )
                _write_private_json(path, operation)
            _write_hold(supervisor, operation)
            phase = 'process_inventory'
            records, top_records, nested_path, nested, nested_absent = _read_managed_roots(supervisor, active)
            roots = []
            outcomes: dict[tuple[int, int, tuple[str, ...]], str] = {}
            for row in records:
                key = (row['pid'], row['starttime'], tuple(row['argv']))
                if _identity_state(row) == 'exited':
                    outcomes[key] = 'already_exited'
                    continue
                root = dict(row)
                root.setdefault('root_role', root.get('role'))
                roots.append(root)

            # A prior receipt may contain descendants that became orphaned after a
            # bounded attempt. Keep their saved identity and parent proof for retry.
            prior_snapshot = operation.get('selected_snapshot') or []
            if not isinstance(prior_snapshot, list):
                raise _error('saved stop-now process snapshot is malformed')
            if active.get('owner') is None:
                prior_snapshot = [row for row in prior_snapshot
                                  if not (isinstance(row, dict)
                                          and row.get('root_role') == 'instance-supervisor')]
            worker_uuids = {row['pid']: row['gpu_uuid'] for row in top_records
                            if row.get('role') == 'worker' and isinstance(row.get('gpu_uuid'), str)}
            for row in top_records:
                if row.get('role') == 'worker':
                    current = _proc_record(int(row['pid']))
                    if current is None or current[0] != row['starttime']:
                        continue
                    if current[1] != tuple(row['argv']):
                        raise _error('Worker PID is live with a different command; ownership is unknown')
                    gpu_uuid = row.get('gpu_uuid')
                    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith('GPU-'):
                        raise _error('owned Worker GPU UUID is unavailable')
                    argv = row.get('argv', [])
                    if not any(argv[index:index + 2] == ['--gpu-uuid', gpu_uuid]
                               for index in range(max(0, len(argv) - 1))):
                        raise _error('Worker process command does not prove its pinned GPU UUID')
                    env_uuid = _env_value(int(row['pid']), 'GPU_UUID')
                    cuda_uuid = _env_value(int(row['pid']), 'CUDA_VISIBLE_DEVICES')
                    if env_uuid != gpu_uuid or cuda_uuid != gpu_uuid:
                        raise _error('Worker launcher GPU UUID does not match its owned role')

            # Pin and freeze every manifest root before recursively discovering
            # children. This prevents a live launcher from creating an unobserved PID.
            phase = 'freezing_owned_tree'
            for row in roots:
                key = (row['pid'], row['starttime'], tuple(row['argv']))
                fd = _pidfd_open(row)
                if fd < 0:
                    outcomes[key] = 'exited_during_capture'
                    continue
                descriptors[key] = fd
                _send(fd, signal.SIGSTOP)
                frozen.add(key)
            live_roots = [row for row in roots if descriptors.get((row['pid'], row['starttime'], tuple(row['argv'])), -1) >= 0]
            stable = 0
            deadline = min(time.monotonic() + _FREEZE_SECONDS, stop_deadline)
            snapshot: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                snapshot = _scan_tree(live_roots, worker_uuids)
                if prior_snapshot:
                    snapshot.extend(row for row in prior_snapshot
                                    if isinstance(row, dict) and all(
                                        (item.get('pid'), item.get('starttime'), tuple(item.get('argv', ()))) !=
                                        (row.get('pid'), row.get('starttime'), tuple(row.get('argv', ())))
                                        for item in snapshot))
                new_count = 0
                for row in snapshot:
                    key = (row['pid'], row['starttime'], tuple(row['argv']))
                    if key in descriptors:
                        continue
                    fd = _pidfd_open(row)
                    if fd < 0:
                        outcomes[key] = 'exited_during_capture'
                        continue
                    descriptors[key] = fd
                    _send(fd, signal.SIGSTOP)
                    frozen.add(key)
                    new_count += 1
                for row in snapshot:
                    key = (row['pid'], row['starttime'], tuple(row['argv']))
                    if key in descriptors:
                        state_for(key, row)
                stopped = True
                for row in snapshot:
                    key = (row['pid'], row['starttime'], tuple(row['argv']))
                    if key in descriptors and state_for(key, row) == 'live':
                        detail = _proc_details(row['pid'])
                        if detail is not None and detail['state'] not in {'T', 't'}:
                            stopped = False
                stable = stable + 1 if new_count == 0 and stopped else 0
                if stable >= 2:
                    break
                time.sleep(0.02)
            else:
                raise _error('owned process tree did not reach a stable frozen snapshot')

            snapshot_by_key = {(row['pid'], row['starttime'], tuple(row['argv'])): row for row in snapshot}
            # The instance owner and gradient supervisor may not have unrecorded
            # direct children: those indicate a manifest write race.
            owner = active.get('owner')
            if owner and (owner['pid'], owner['starttime'], tuple(owner['argv'])) in snapshot_by_key:
                allowed = {int(row['pid']) for row in top_records}
                if any(row.get('parent_pid') == owner['pid'] and row['pid'] not in allowed
                       for row in snapshot):
                    raise _error('instance owner has an unrecorded direct child')
            gradient_supervisor = next((row for row in records if row.get('role') == 'gradient-supervisor'), None)
            nested_roles = (nested or {}).get('roles', [])
            if gradient_supervisor and (gradient_supervisor['pid'], gradient_supervisor['starttime'],
                                        tuple(gradient_supervisor['argv'])) in snapshot_by_key:
                allowed = {int(row['pid']) for row in nested_roles}
                if any(row.get('parent_pid') == gradient_supervisor['pid'] and row['pid'] not in allowed
                       for row in snapshot):
                    raise _error('gradient supervisor has an unrecorded direct child')

            selected = {}
            for row in [*records, *snapshot]:
                key = (row['pid'], row['starttime'], tuple(row['argv']))
                selected.setdefault(key, {**row, 'outcome': outcomes.get(key, 'selected')})
            operation.update(selected_snapshot=list(selected.values()), phase='process_tree_frozen',
                             updated_at=time.time())
            _write_private_json(path, operation)
            emit('process_tree_frozen', process_count=len(selected))

            # Stop Learner/gradient writers first. Worker launchers and native
            # endpoints stay available only until those writers have exited.
            owner_key = None
            if owner:
                owner_key = (owner['pid'], owner['starttime'], tuple(owner['argv']))
            core_roles = {'gradient-supervisor', 'server', 'executor', 'teacher'}
            groups = [
                [key for key, row in selected.items()
                 if key != owner_key and row.get('root_role', row.get('role')) in core_roles],
                [key for key, row in selected.items()
                 if key != owner_key and row.get('root_role', row.get('role')) not in core_roles],
            ]
            for group_index, keys in enumerate(groups):
                for key in keys:
                    if key not in descriptors or state_for(key, selected[key]) != 'live':
                        continue
                    _send(descriptors[key], signal.SIGTERM)
                    selected[key]['outcome'] = 'SIGTERM_sent'
                for key in keys:
                    if key in frozen:
                        _send(descriptors[key], signal.SIGCONT)
                        frozen.discard(key)
                group_budget = _CORE_TERM_SECONDS if group_index == 0 else _ENDPOINT_TERM_SECONDS
                group_deadline = min(time.monotonic() + group_budget, stop_deadline)
                while time.monotonic() < group_deadline:
                    if all(key not in descriptors or state_for(key, selected[key]) != 'live' for key in keys):
                        break
                    time.sleep(0.05)

            role_keys = [key for key in selected if key != owner_key]
            remaining = [key for key in role_keys if key in descriptors and state_for(key, selected[key]) == 'live']
            for key in remaining:
                _send(descriptors[key], signal.SIGKILL)
                selected[key]['outcome'] = 'SIGKILL_sent'
            while time.monotonic() < stop_deadline:
                remaining = [key for key in role_keys if key in descriptors and state_for(key, selected[key]) == 'live']
                if not remaining:
                    break
                time.sleep(0.05)
            remaining = [key for key in role_keys if key in descriptors and state_for(key, selected[key]) == 'live']
            if remaining:
                raise _error('owned role descendants remain after bounded SIGKILL')

            if owner_key and owner_key in descriptors:
                if owner_key in frozen:
                    _send(descriptors[owner_key], signal.SIGCONT)
                    frozen.discard(owner_key)
                _send(descriptors[owner_key], signal.SIGTERM)
                selected[owner_key]['outcome'] = 'owner_shutdown_requested'
                deadline = min(time.monotonic() + 5.0, stop_deadline)
                while time.monotonic() < deadline:
                    current = _read_private_json(supervisor.active_path)
                    if current and current.get('owner') is None and current.get('status') == 'stopped':
                        guard = current.get('idle_guard')
                        if (isinstance(guard, dict)
                                and all(guard.get(k) == owner.get(k) for k in ('pid', 'starttime', 'argv'))
                                and guard.get('status') == 'alive_without_owner_lock'
                                and _identity_state(guard) == 'live'):
                            break  # A persistent owner deliberately became the guard.
                    if state_for(owner_key, selected[owner_key]) != 'live':
                        break
                    time.sleep(0.05)

            if owner_key and owner_key in descriptors and state_for(owner_key, selected[owner_key]) == 'live':
                current = _read_private_json(supervisor.active_path) or {}
                current_guard = current.get('idle_guard')
                demoted = (current.get('owner') is None and current.get('status') == 'stopped'
                           and isinstance(current_guard, dict)
                           and current_guard.get('status') == 'alive_without_owner_lock'
                           and all(current_guard.get(k) == owner.get(k) for k in ('pid', 'starttime', 'argv'))
                           and _identity_state(current_guard) == 'live')
                if not demoted:
                    # All writers/endpoints are already stopped. A stuck host
                    # owner is part of this pinned tree; PID 1 is never killed.
                    _send(descriptors[owner_key], signal.SIGKILL)
                    selected[owner_key]['outcome'] = 'SIGKILL_sent'
                    while time.monotonic() < stop_deadline and state_for(owner_key, selected[owner_key]) == 'live':
                        time.sleep(0.02)

            # Re-read every exact manager inventory before any checkpoint cleanup.
            phase = 'verify_stopped'
            final_active = _read_private_json(supervisor.active_path)
            final_records, _final_top, final_nested_path, final_nested, final_nested_absent = _read_managed_roots(
                supervisor, final_active or {})
            for row in final_records:
                if _identity_state(row) == 'live':
                    raise _error(f'a manifest-owned {row.get("role", "process")} PID {row["pid"]} remains active')
            if final_active is None:
                raise _error('active instance manifest disappeared before cleanup')
            if active.get('owner') and not final_active.get('stopped_owner'):
                final_active['stopped_owner'] = {
                    key: active['owner'][key] for key in ('pid', 'starttime', 'argv') if key in active['owner']
                }
            guard = final_active.get('idle_guard')
            owner_summary = None
            if isinstance(guard, dict):
                if _identity_state(guard) == 'live':
                    if guard.get('status') == 'exited':
                        raise _error('idle guard marked exited is still live; terminal ownership is not proven')
                    guard['status'] = 'alive_without_owner_lock'
                    owner_summary = {key: guard[key] for key in ('pid', 'starttime', 'argv')}
                    owner_summary['status'] = 'alive_without_owner_lock'
                else:
                    guard['status'] = 'exited'
            if owner_summary is None:
                stopped_owner = final_active.get('stopped_owner')
                if (not isinstance(stopped_owner, dict) or stopped_owner.get('pid', 0) <= 1
                        or _identity_state(stopped_owner) != 'exited'):
                    raise _error('stop proof has neither a live idle guard nor a verified exited host owner')
            final_active.update(status='stopped', owner=None, stop_operation_id=operation['operation_id'])
            stopped_snapshot = []
            for key, row in selected.items():
                state = state_for(key, row)
                if state == 'live':
                    if (owner_summary is not None and key == owner_key
                            and all(row.get(k) == owner_summary.get(k) for k in ('pid', 'starttime', 'argv'))):
                        row.update(outcome='demoted_to_idle_guard', exited=False)
                        continue
                    raise _error('a selected process descendant remains active')
                row['outcome'] = row.get('outcome') if state == 'exited' else 'already_exited'
                row['exited'] = True
                stopped_snapshot.append(row)
            # SIGKILL or supervisor failure can prevent its final manifest write.
            # Reconcile metadata only after every exact root/descendant is gone.
            if final_nested is not None:
                final_nested = {**final_nested, 'status': 'stopped',
                                'stop_operation_id': operation['operation_id'], 'stopped_at': time.time(),
                                'supervisor': {**final_nested['supervisor'], 'status': 'exited'},
                                'roles': [{**row, 'status': 'exited'} for row in final_nested['roles']]}
                _write_private_json(Path(final_nested_path), final_nested)
            final_active['roles'] = [{**row, 'status': 'exited'} for row in final_active['roles']]
            _write_private_json(supervisor.active_path, final_active)
            from .instance_stop_now import normalize_stop_process_proof
            verified, descendant_audit = normalize_stop_process_proof(
                final_active, final_nested, stopped_snapshot)
            stopped_owner = final_active.get('stopped_owner')
            stopped_owner_proof = None
            if stopped_owner is not None:
                stopped_owner_proof = {key: stopped_owner[key]
                                       for key in ('pid', 'starttime', 'argv') if key in stopped_owner}
                if _identity_state(stopped_owner) == 'live':
                    if (owner_summary is None or
                            stopped_owner_proof != {key: owner_summary[key] for key in ('pid', 'starttime', 'argv')}):
                        raise _error('stopped owner identity remains live outside the idle guard')
                    stopped_owner_proof['outcome'] = 'demoted_to_idle_guard'
                else:
                    stopped_owner_proof['outcome'] = 'exited'
            proof = {
                'schema': STOP_NOW_PROOF_SCHEMA, 'status': 'stopped',
                'operation_id': operation['operation_id'], 'deployment_id': _deployment_id(supervisor),
                'run_id': final_active.get('run_id'), 'verified_processes': verified,
                'descendant_audit': descendant_audit,
                'stopped_owner': stopped_owner_proof,
                'nested_manifest_path': str(final_nested_path) if final_nested_path else None,
                'nested_manifest_absent': final_nested_absent,
                'idle_guard': owner_summary, 'selected_checkpoint': None,
                'discarded_work_summary': None, 'resume_command': resume_command,
            }
            if operation.get('supersedes_learner_stop'):
                proof['supersedes_operation_id'] = operation['supersedes_operation_id']
                proof['superseded_run_id'] = operation.get('superseded_learner_run_id')
                proof['superseded_learner_run_id'] = operation.get('superseded_learner_run_id')
                proof['superseded_target_update'] = operation.get('superseded_target_update')
                proof['superseded_lineage_id'] = operation.get('superseded_lineage_id')
                proof['superseded_engine_epoch'] = operation.get('superseded_engine_epoch')
                proof['superseded_target_policy_version'] = operation.get('superseded_target_policy_version')
            proof['process_release_seconds'] = round(time.monotonic() - process_release_started, 3)
            operation.update(manager_status='discarding', phase='processes_verified',
                             verified_processes=verified, idle_guard=owner_summary,
                             process_release_seconds=proof['process_release_seconds'],
                             updated_at=time.time())
            _write_private_json(path, operation)
            emit('processes_verified', process_count=len(verified), idle_guard=owner_summary)

            # The helper owns Core/window rollback under the optimizer lock. It
            # is deliberately after all writer and endpoint identities are gone.
            from .instance_stop_now import discard_stopped_instance_work
            phase = 'discard_uncommitted_work'
            cleanup_started = time.monotonic()
            discarded = discard_stopped_instance_work(
                supervisor.deployment, manifest_path=supervisor.active_path,
                operation_id=operation['operation_id'], stop_receipt=proof,
            )
            proof['cleanup'] = discarded
            proof['selected_checkpoint'] = dict(discarded['checkpoint'])
            proof['discarded_work_summary'] = {
                'classification': discarded['classification'],
                'staged_windows': len(discarded.get('staged_windows_discarded', [])),
                'core_cancellations': len(discarded.get('core_cancellations', [])),
                'quarantined_files': len(discarded.get('quarantined_files', [])),
                'bepa_ledger': 'unchanged' if discarded['bepa_ledger']['unchanged'] is True else 'changed',
            }
            proof['rollback'] = discarded.get('rollback')
            proof['phase_status'] = discarded.get('phase_status', 'checkpoint_preserved_work_discarded')
            proof['cpu_cleanup_seconds'] = round(time.monotonic() - cleanup_started, 3)
            proof['next_command'] = resume_command
            operation.pop('error_type', None)
            operation.pop('detail', None)
            operation.update(manager_status='completed', phase='completed', proof=proof,
                             selected_checkpoint=proof['selected_checkpoint'],
                             discarded_work_summary=proof['discarded_work_summary'],
                             rollback=proof.get('rollback'), next_command=resume_command,
                             updated_at=time.time())
            _write_private_json(path, operation)
            return proof
    except BlockingIOError:
        if operation is None:
            operation = {'operation_id': uuid.uuid4().hex, 'mode': 'now',
                         'deployment_id': _deployment_id(supervisor), 'manager_run_id': None}
        _block(supervisor, operation, 'operation_lock', _error('another managed operation holds operation.lock'),
               persist=False)
    except Exception as exc:
        release_stopped()
        if operation is None:
            operation = {'schema': STOP_OPERATION_SCHEMA, 'operation_id': uuid.uuid4().hex,
                         'mode': 'now', 'deployment_id': _deployment_id(supervisor),
                         'manager_run_id': None, 'selected_snapshot': []}
        _block(supervisor, operation, phase, exc, persist=operation.get('manager_run_id') is not None)
    finally:
        release_stopped()
        for descriptor in descriptors.values():
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
