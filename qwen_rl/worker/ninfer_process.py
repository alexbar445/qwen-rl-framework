"""Worker-owned Native child for the explicit stop-during-learning mode."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path


async def settle(operation):
    """Cancellation must not leave a child start/stop or policy replay running."""
    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        try:
            task.result()
        except BaseException:
            pass
        raise cancellation


class NInferProcess:
    def __init__(self, contract):
        self.contract = contract
        self.process = None
        self.state = 'new'
        self.closed = False
        self.failure_callback = None
        self._watcher = None

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def stopped_status(self, model_id):
        if self.state != 'stopped' or self.process is None or self.process.poll() is None:
            raise RuntimeError('Native child exit has not been proven')
        return {'model':model_id, 'enabled':True, 'state':'suspended',
                'registry_admission_held':True, 'engine_process_state':'stopped',
                'exited_pid':self.process.pid, 'exit_code':self.process.returncode,
                'last_suspend':{'cache_discarded':True}}

    async def start(self):
        if self.closed or self.state not in ('new', 'stopped'):
            raise RuntimeError(f'cannot start Native child from {self.state}')
        from .ninfer_launcher import _wait_for_engine
        self.state = 'starting'
        env = os.environ.copy()
        for key in ('ARC_ENV_TOKEN', 'ARC_ENV_TOKEN_FILE', 'QWEN_NINFER_ENGINE_CONTRACT'):
            env.pop(key, None)
        env['QWEN_NINFER_PARENT_PID'] = str(os.getpid())
        try:
            self.process = subprocess.Popen(self.contract['command'], env=env, start_new_session=True)
            await _wait_for_engine(
                self.process, base_url=self.contract['base_url'],
                expected_identity=self.contract['identity'],
                max_context=self.contract['max_context'],
                max_concurrency=self.contract['max_concurrency'],
                policy_slots=self.contract['policy_slots'],
                timeout_seconds=self.contract['ready_timeout'], shutdown_event=asyncio.Event(),
            )
            cache = self.contract.get('device_profile_cache')
            if cache:
                from .device_profiles import DeviceProfileCache
                DeviceProfileCache(Path(cache)).publish(Path(self.contract['device_profile_path']))
            self.state = 'running'
            if self._watcher is None:
                self._watcher = asyncio.create_task(self._watch())
        except BaseException:
            self.close_child()
            self.state = 'failed'
            raise

    async def _watch(self):
        while not self.closed:
            if self.state == 'running' and self.process.poll() is not None:
                self.state = 'failed'
                if self.failure_callback is not None:
                    await self.failure_callback()
                return
            await asyncio.sleep(.1)

    def close_child(self):
        process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=self.contract['shutdown_grace'])
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    async def stop(self):
        if self.state != 'running' or self.process.poll() is not None:
            raise RuntimeError('cannot stop an absent or failed Native child')
        self.state = 'stopping'
        try:
            await asyncio.to_thread(self.close_child)
            self.state = 'stopped'
        except BaseException:
            self.state = 'failed'
            raise

    def close(self):
        self.closed = True
        self.state = 'closing'
        self.close_child()
