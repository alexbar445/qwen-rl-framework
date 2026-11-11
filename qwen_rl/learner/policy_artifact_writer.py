"""Policy-sized durable output that retains its immediately consumed hot pages."""
from qwen_rl._vendor.ninfer_artifact.file_io import Writeback, datasync
from qwen_rl._vendor.ninfer_artifact.writer import ArtifactWriter as BaseArtifactWriter


class PolicyWriteback(Writeback):
    # Bound policy dirty bytes independently of the generic offline-base writer.
    limit_bytes = 64 * 1024**2

    def written(self, fd, count):
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= self.limit_bytes:
            self.flush()

    def flush(self):
        for fd in self._fds:
            datasync(fd)
        self._fds.clear()
        self._bytes = 0


class PolicyArtifactWriter(BaseArtifactWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        writeback = PolicyWriteback()
        # Header bytes are already dirty after the base constructor.
        writeback._fds.update(self._writeback._fds)
        writeback._bytes = self._writeback._bytes
        self._writeback = writeback
