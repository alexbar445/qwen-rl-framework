"""Content-addressed Docker image archives for air-gapped Worker deployment."""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import hmac
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from .artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFound,
    ContentAddressedStore,
    _fsync_dir,
    _is_private_or_local,
)


_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


class ImageArchiveError(RuntimeError):
    """A local Docker image archive could not be exported or installed safely."""


class ImageArchiveDescriptor(BaseModel):
    """Immutable reference to one gzipped Docker save archive on the site LAN."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    image_id: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    archive_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: StrictInt = Field(gt=0)
    url: StrictStr = Field(min_length=1, max_length=2048)

    @classmethod
    def from_value(cls, value: "ImageArchiveDescriptor | Mapping[str, Any]") -> "ImageArchiveDescriptor":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("image archive descriptor must be a strict mapping")
        return cls.model_validate(value, strict=True)

    @model_validator(mode="after")
    def validate_content_url(self) -> "ImageArchiveDescriptor":
        parsed = urlsplit(self.url)
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("image archive URL has an invalid port") from exc
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("image archive URL must be an HTTP(S) LAN URL without credentials or query")
        if not _is_private_or_local(parsed.hostname):
            raise ValueError("image archive URL must point to a private or local host")
        if parsed.path.rstrip("/") != f"/v1/artifacts/{self.archive_sha256}":
            raise ValueError("image archive URL must use its pinned /v1/artifacts/<sha256> route")
        return self


@dataclass(frozen=True, slots=True)
class ExportedImageArchive:
    """Descriptor plus the local content-addressed blob path for a publisher."""

    descriptor: ImageArchiveDescriptor
    path: Path


class _HashingWriter:
    """Hash compressed bytes as gzip writes them, without making a tar file."""

    def __init__(self, output) -> None:
        self.output = output
        self.digest = hashlib.sha256()
        self.size_bytes = 0

    def write(self, data: bytes) -> int:
        count = self.output.write(data)
        if count is None:
            count = len(data)
        self.digest.update(memoryview(data)[:count])
        self.size_bytes += count
        return count

    def flush(self) -> None:
        self.output.flush()


CommandRunner = Callable[..., subprocess.CompletedProcess]
PopenFactory = Callable[..., Any]


def _image_id_from_inspect(
    image_ref: str,
    *,
    docker_binary: str | Path,
    command_runner: CommandRunner,
) -> str:
    if (not isinstance(image_ref, str) or not image_ref or image_ref.startswith("-")
            or any(char.isspace() or ord(char) < 32 for char in image_ref)):
        raise ValueError("image reference must be a nonempty single-line Docker image reference")
    try:
        result = command_runner(
            [str(docker_binary), "image", "inspect", "--format={{.Id}}", image_ref],
            check=False, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ImageArchiveError("Docker image inspect failed") from exc
    if getattr(result, "returncode", None) != 0:
        raise ImageArchiveError("Docker image inspect did not find the requested local image")
    value = getattr(result, "stdout", "")
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    image_id = value.strip() if isinstance(value, str) else ""
    if not _IMAGE_ID.fullmatch(image_id):
        raise ImageArchiveError("Docker inspect returned no full sha256 image ID")
    return image_id


def _content_origin(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("artifact_base_url must be a nonempty LAN origin")
    parsed = urlsplit(base_url)
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("artifact_base_url has an invalid port") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("artifact_base_url must be an HTTP(S) origin without credentials, path, or query")
    if not _is_private_or_local(parsed.hostname):
        raise ValueError("artifact_base_url must point to a private or local host")
    return base_url.rstrip("/")


def export_image_archive(
    image_ref: str,
    content_dir: str | Path,
    *,
    artifact_base_url: str,
    docker_binary: str | Path = "docker",
    command_runner: CommandRunner = subprocess.run,
    popen_factory: PopenFactory = subprocess.Popen,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    compression_level: int = 1,
) -> ExportedImageArchive:
    """Stream ``docker image save`` through gzip into the SHA-addressed content store.

    No uncompressed tar is written. The immutable local image ID is inspected
    first and used for the save command, so a mutable tag cannot change the
    bytes selected between inspect and export.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(compression_level, bool) or not isinstance(compression_level, int) or not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be an integer from 0 to 9")
    content_origin = _content_origin(artifact_base_url)
    image_id = _image_id_from_inspect(
        image_ref, docker_binary=docker_binary, command_runner=command_runner,
    )
    store = ContentAddressedStore(content_dir)
    stage_dir = store.blob_root / "00"
    stage_dir.mkdir(parents=True, exist_ok=True)
    fd, stage_name = tempfile.mkstemp(prefix=".docker-image.", suffix=".tar.gz.partial", dir=stage_dir)
    os.close(fd)
    stage = Path(stage_name)
    process = None
    try:
        with tempfile.TemporaryFile(mode="w+b") as error_output:
            try:
                process = popen_factory(
                    [str(docker_binary), "image", "save", image_id],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=error_output,
                    bufsize=0,
                )
            except (OSError, ValueError) as exc:
                raise ImageArchiveError("Docker image save could not be started") from exc
            stdout = getattr(process, "stdout", None)
            if stdout is None or not callable(getattr(stdout, "read", None)):
                if callable(getattr(process, "kill", None)):
                    process.kill()
                if callable(getattr(process, "wait", None)):
                    process.wait()
                raise ImageArchiveError("Docker image save has no readable archive stream")

            hashing = None
            try:
                with stage.open("wb") as output:
                    hashing = _HashingWriter(output)
                    with gzip.GzipFile(fileobj=hashing, mode="wb", filename="", mtime=0,
                                       compresslevel=compression_level) as compressor:
                        while chunk := stdout.read(chunk_size):
                            compressor.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException:
                if callable(getattr(process, "kill", None)):
                    process.kill()
                if callable(getattr(process, "wait", None)):
                    process.wait()
                raise
            finally:
                close_stdout = getattr(stdout, "close", None)
                if callable(close_stdout):
                    close_stdout()

            try:
                returncode = process.wait()
            except OSError as exc:
                raise ImageArchiveError("Docker image save did not exit cleanly") from exc
            if returncode != 0:
                error_output.seek(0, os.SEEK_END)
                end = error_output.tell()
                error_output.seek(max(0, end - 2048), os.SEEK_SET)
                detail = error_output.read(2048).decode("utf-8", errors="replace").strip()
                suffix = f" ({detail[-500:]})" if detail else ""
                raise ImageArchiveError(f"Docker image save failed with exit code {returncode}{suffix}")

        if hashing is None or hashing.size_bytes == 0:
            raise ImageArchiveError("Docker image save produced an empty archive")
        archive_sha256 = hashing.digest.hexdigest()
        archive_size = hashing.size_bytes
        destination = store.path_for(archive_sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            prior = store.resolve(archive_sha256, expected_size=archive_size)
        except (ArtifactNotFound, ArtifactIntegrityError):
            os.chmod(stage, 0o444)
            os.replace(stage, destination)
            _fsync_dir(destination.parent)
            store.remember_verified(archive_sha256, destination, expected_size=archive_size)
            stage = None
            archive_path = destination
        else:
            stage.unlink(missing_ok=True)
            stage = None
            archive_path = prior.path

        descriptor = ImageArchiveDescriptor(
            image_id=image_id,
            archive_sha256=archive_sha256,
            size_bytes=archive_size,
            url=f"{content_origin}/v1/artifacts/{archive_sha256}",
        )
        descriptor.validate_content_url()
        return ExportedImageArchive(descriptor=descriptor, path=archive_path)
    except BaseException:
        if stage is not None:
            stage.unlink(missing_ok=True)
        raise


def _docker_command(
    argv: list[str],
    *,
    command_runner: CommandRunner,
    timeout: float,
) -> subprocess.CompletedProcess:
    try:
        result = command_runner(
            argv, check=False, capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ImageArchiveError("Docker image archive command failed") from exc
    if getattr(result, "returncode", None) != 0:
        raise ImageArchiveError("Docker image archive command returned a nonzero exit status")
    return result


async def install_image_archive(
    descriptor: ImageArchiveDescriptor | Mapping[str, Any],
    downloader: Any,
    *,
    worker_token: str,
    docker_binary: str | Path = "docker",
    docker_args_prefix: Sequence[str] = (),
    command_runner: CommandRunner = subprocess.run,
    docker_timeout: float = 7200.0,
) -> dict[str, Any]:
    """Fetch a pinned archive through Dragonfly, load it, and verify its exact image ID."""
    image = ImageArchiveDescriptor.from_value(descriptor).validate_content_url()
    if (not isinstance(worker_token, str) or not worker_token
            or any(char.isspace() or char in "\r\n\x00" for char in worker_token)):
        raise ValueError("Worker bearer token must be a nonempty single-line value")
    if (isinstance(docker_timeout, bool) or not isinstance(docker_timeout, (int, float))
            or docker_timeout <= 0):
        raise ValueError("docker_timeout must be positive")

    try:
        archive_path = await downloader.download(
            image.url,
            image.archive_sha256,
            image.size_bytes,
            headers={"Authorization": f"Bearer {worker_token}"},
            allow_lan_source_fallback=False,
        )
    except Exception as exc:
        raise ImageArchiveError("Dragonfly could not download the pinned Worker image archive") from exc
    archive_path = Path(archive_path)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ImageArchiveError("Dragonfly image archive is not a regular local file")

    load_argv = [str(docker_binary), *docker_args_prefix, "image", "load", "--input", str(archive_path)]
    await asyncio.to_thread(
        _docker_command, load_argv, command_runner=command_runner, timeout=float(docker_timeout),
    )
    inspect_argv = [str(docker_binary), *docker_args_prefix,
                    "image", "inspect", "--format={{.Id}}", image.image_id]
    inspect_result = await asyncio.to_thread(
        _docker_command, inspect_argv, command_runner=command_runner, timeout=60.0,
    )
    observed = getattr(inspect_result, "stdout", "")
    if isinstance(observed, bytes):
        observed = observed.decode("ascii", errors="replace")
    observed = observed.strip() if isinstance(observed, str) else ""
    if not hmac.compare_digest(observed, image.image_id):
        raise ImageArchiveError("loaded Docker image ID differs from the pinned archive descriptor")
    return {
        "state": "loaded_verified",
        "image_id": image.image_id,
        "archive_sha256": image.archive_sha256,
        "size_bytes": image.size_bytes,
        "url_host": urlsplit(image.url).hostname,
    }
