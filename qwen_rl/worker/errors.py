"""Worker availability failures are distinct from invalid sampling evidence."""
from __future__ import annotations


class WorkerUnavailableError(RuntimeError):
    """The request has no usable result; uncertain requests require fencing."""

    def __init__(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        quarantinable: bool = True,
    ):
        super().__init__(message)
        self.worker_id = worker_id
        self.quarantinable = quarantinable


def worker_unavailable(error: BaseException) -> WorkerUnavailableError | None:
    """Recover the typed transport failure through an agent-loop wrapper."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, WorkerUnavailableError):
            return error
        error = error.__cause__ or error.__context__
    return None
