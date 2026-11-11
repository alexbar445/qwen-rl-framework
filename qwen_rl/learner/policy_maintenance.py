"""One owned, coalescing cache-maintenance worker after policy ACKs."""
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import shutil
import stat
import threading
import time


def validate_policy_acknowledgements(policy, acknowledgements, worker_ids):
    """The authorization fence is synchronous; only file deletion is deferred."""
    if not worker_ids or set(acknowledgements) != set(worker_ids):
        raise ValueError('policy cleanup requires all configured Worker acknowledgements')
    for worker_id in worker_ids:
        ack = acknowledgements[worker_id]
        if (ack.get('worker_id') != worker_id or ack.get('version') != policy['version']
                or ack.get('files') != policy['files']
                or ack.get('artifact_id') != policy.get('artifact_id')):
            raise ValueError('policy cleanup requires matching version and immutable file metadata acknowledgements')


class PolicyMaintenance:
    def __init__(self, execute, publish_status):
        self.execute, self.publish_status = execute, publish_status
        self._lock = threading.RLock()
        self._pool = None
        self._future = None
        self._pending = None
        self._latest_version = -1
        self._closed = False

    def schedule(self, request):
        with self._lock:
            if self._closed:
                raise RuntimeError('policy maintenance is closed')
            version = request[0]['version']
            if version < self._latest_version:
                return
            self._latest_version = version
            self._pending = request
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='policy-maintenance')
            if self._future is None or self._future.done():
                self._future = self._pool.submit(self._run)

    def _run(self):
        while True:
            with self._lock:
                request, self._pending = self._pending, None
                if request is None:
                    # Change ownership under the same lock as schedule().
                    self._future = None
                    return
            version = request[0]['version']
            started = time.perf_counter()
            try:
                self.execute(*request)
            except Exception as exc:
                self.publish_status({'status':'failed','policy_version':version,
                                     'error_type':type(exc).__name__,
                                     'seconds':time.perf_counter()-started})
                logging.getLogger(__name__).exception('Policy cache maintenance failed')
            else:
                self.publish_status({'status':'completed','policy_version':version,
                                     'seconds':time.perf_counter()-started})

    def drain(self):
        while True:
            with self._lock:
                future = self._future
            if future is None:
                return
            future.result()

    def close(self):
        with self._lock:
            self._closed = True
            pool = self._pool
        if pool is not None:
            pool.shutdown(wait=True)


_LOGGER = logging.getLogger(__name__)


def remove_conflicting_publication_directory(
    path: str | Path,
    *,
    version: int,
    kind: str,
) -> bool:
    """Remove an unadopted version directory so publication can retry."""
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"cannot inspect conflicting {kind} publication: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise RuntimeError(f"conflicting {kind} publication is not a real directory: {path}")
    _LOGGER.warning(
        "clearing conflicting %s publication for policy v%s and retrying: %s",
        kind, version, path,
    )
    shutil.rmtree(path)
    return True
