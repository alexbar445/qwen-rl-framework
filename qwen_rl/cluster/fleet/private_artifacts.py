"""SHA-addressed Learner bundles stored outside the public upload tree."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
from dataclasses import dataclass
from typing import BinaryIO


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BEARER = re.compile(r"^[A-Za-z0-9_-]{32,512}$")


class PrivateArtifactError(RuntimeError):
    """A private bundle could not be safely published or served."""


@dataclass(frozen=True)
class PublishedPrivateArtifact:
    sha256: str
    size_bytes: int
    path: Path
    publisher_bearer_file: Path
    publisher_bearer: str


def _private_dir(path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise PrivateArtifactError("private artifact root must be a real directory")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    if os.name == "posix" and path.stat().st_uid != os.geteuid():
        raise PrivateArtifactError("private artifact root must be owned by the Fleet service account")
    return path


def _assert_disjoint(private_root: Path, public_root: str | Path | None) -> None:
    if public_root is None:
        raise PrivateArtifactError("public upload root must be configured before publishing Learner bundles")
    public = Path(public_root).expanduser().resolve()
    private = private_root.resolve()
    if private == public or private.is_relative_to(public) or public.is_relative_to(private):
        raise PrivateArtifactError("private artifact storage must not overlap the public upload tree")


def _open_regular_source(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PrivateArtifactError("source bundle must be a readable regular file, not a symlink") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        os.close(descriptor)
        raise PrivateArtifactError("source bundle must be a nonempty regular file")
    return descriptor, info


def _write_private_token(path: Path) -> str:
    path = Path(path)
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise PrivateArtifactError("existing bundle bearer file must be a regular mode-0600 file")
        if os.name == "posix" and info.st_uid != os.geteuid():
            raise PrivateArtifactError("existing bundle bearer file has the wrong owner")
        token = path.read_text(encoding="ascii").strip()
        if _BEARER.fullmatch(token) is None:
            raise PrivateArtifactError("existing bundle bearer file is invalid")
        return token
    token = secrets.token_urlsafe(32)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        os.chmod(path, 0o600)
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return token
    except FileExistsError:
        return _write_private_token(path)
    except OSError as exc:
        raise PrivateArtifactError("could not create private bundle bearer file") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class PrivateArtifactStore:
    def __init__(self, state_dir: str | Path, *, public_upload_dir: str | Path | None):
        self.state_dir = _private_dir(Path(state_dir))
        self.root = _private_dir(self.state_dir / "private-artifacts")
        _assert_disjoint(self.state_dir, public_upload_dir)
        _assert_disjoint(self.root, public_upload_dir)
        self.public_upload_root = Path(public_upload_dir).expanduser().resolve() if public_upload_dir is not None else None

    def publish(self, source: str | Path) -> PublishedPrivateArtifact:
        source_path = Path(source).expanduser()
        if self.public_upload_root is None:
            raise PrivateArtifactError("public upload root must be configured before publishing Learner bundles")
        resolved_source = source_path.resolve()
        if (resolved_source == self.public_upload_root or resolved_source.is_relative_to(self.public_upload_root)):
            raise PrivateArtifactError("private Learner bundle source must be outside the public upload tree")
        descriptor, source_info = _open_regular_source(source_path)
        temporary_path: Path | None = None
        try:
            temp_fd, temp_name = tempfile.mkstemp(prefix=".publish-", dir=self.root)
            temporary_path = Path(temp_name)
            digest = hashlib.sha256()
            size = 0
            os.fchmod(temp_fd, 0o600)
            with os.fdopen(temp_fd, "wb") as destination, os.fdopen(os.dup(descriptor), "rb") as stream:
                while chunk := stream.read(4 * 1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if size != source_info.st_size:
                raise PrivateArtifactError("source bundle changed while it was being published")
            sha256 = digest.hexdigest()
            target = self.root / sha256
            if target.exists() or target.is_symlink():
                target_info = target.lstat()
                if (not stat.S_ISREG(target_info.st_mode) or stat.S_IMODE(target_info.st_mode) != 0o600
                        or target_info.st_uid != os.geteuid() or target_info.st_size != size):
                    raise PrivateArtifactError("existing private artifact has invalid ownership or mode")
                existing_digest = hashlib.sha256()
                with target.open("rb") as stream:
                    while chunk := stream.read(4 * 1024 * 1024):
                        existing_digest.update(chunk)
                if existing_digest.hexdigest() != sha256:
                    raise PrivateArtifactError("existing private artifact failed its content hash")
            else:
                os.link(temporary_path, target, follow_symlinks=False)
                os.chmod(target, 0o600)
                parent_fd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            publisher_bearer_file = self.root / f"{sha256}.publisher-bearer"
            publisher_bearer = _write_private_token(publisher_bearer_file)
            return PublishedPrivateArtifact(sha256=sha256, size_bytes=size, path=target,
                                            publisher_bearer_file=publisher_bearer_file,
                                            publisher_bearer=publisher_bearer)
        except FileExistsError:
            # A concurrent publisher won the same content-addressed name. Verify
            # it on the next invocation rather than replacing its immutable file.
            raise PrivateArtifactError("private artifact was published concurrently; retry publication")
        except OSError as exc:
            raise PrivateArtifactError("private bundle publication failed") from exc
        finally:
            os.close(descriptor)
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def serve_path(self, sha256: str) -> tuple[Path, int] | None:
        if _SHA256.fullmatch(sha256) is None:
            return None
        path = self.root / sha256
        try:
            info = path.lstat()
        except OSError:
            return None
        if (not stat.S_ISREG(info.st_mode) or path.is_symlink() or stat.S_IMODE(info.st_mode) != 0o600
                or (os.name == "posix" and info.st_uid != os.geteuid())):
            raise PrivateArtifactError("private artifact ownership or mode check failed")
        return path, info.st_size
