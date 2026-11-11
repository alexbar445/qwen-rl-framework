"""Install and verify the pinned user-level Docker Compose CLI plugin."""
from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .paths import AccountPathError, ensure_account_path

if TYPE_CHECKING:
    from .contracts import UserAccount
    from .roles import DockerAccess


COMPOSE_VERSION = "5.6.0"
# Docker's Linux manual install documents this release URL and user config path;
# the hash is recorded in Docker's official docker-library/repo-info metadata.
COMPOSE_URL = (
    "https://github.com/docker/compose/releases/download/v5.6.0/"
    "docker-compose-linux-x86_64"
)
COMPOSE_SHA256 = "40343e21ca777173e69cff5dbafeb37c6f81f3b0d57d9e597f036e95eb63e76a"
_MAX_BINARY_BYTES = 128 * 1024 * 1024
_COMMAND_TIMEOUT_SECONDS = 20
_HTTP_TIMEOUT_SECONDS = 20
_VERSION_RE = re.compile(r"^v?(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?)$")


class ComposePluginError(RuntimeError):
    """The active Docker CLI cannot use a safely verified Compose plugin."""


def _docker_version(
    access: DockerAccess,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> str | None:
    try:
        result = runner(
            access.command("compose", "version", "--short"),
            check=False,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ComposePluginError("Docker Compose availability could not be checked") from exc
    if result.returncode != 0:
        return None
    raw = (result.stdout or "").strip()
    match = _VERSION_RE.fullmatch(raw)
    return match.group(1) if match else None


def _plugin_path(account: UserAccount) -> Path:
    configured = os.environ.get("DOCKER_CONFIG", "")
    if configured:
        root = Path(configured)
        if not root.is_absolute() or ".." in root.parts:
            raise ComposePluginError("DOCKER_CONFIG must be an absolute directory below the registered HOME")
    else:
        root = Path(account.home) / ".docker"
    home = Path(account.home)
    if not root.is_relative_to(home):
        raise ComposePluginError("Docker configuration must be below the registered HOME")
    return root / "cli-plugins" / "docker-compose"


def _download_binary(*, transport: httpx.BaseTransport | None) -> bytes:
    try:
        with httpx.Client(
            verify=True,
            transport=transport,
            timeout=httpx.Timeout(_HTTP_TIMEOUT_SECONDS, connect=10.0),
            follow_redirects=True,
        ) as client, client.stream("GET", COMPOSE_URL) as response:
            response.raise_for_status()
            if any(hop.url.scheme != "https" for hop in (*response.history, response)):
                raise ComposePluginError("Compose download did not remain on HTTPS")
            content_length = response.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > _MAX_BINARY_BYTES:
                        raise ComposePluginError("Compose download exceeds the size limit")
                except ValueError as exc:
                    raise ComposePluginError("Compose download has an invalid size header") from exc
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                size += len(chunk)
                if size > _MAX_BINARY_BYTES:
                    raise ComposePluginError("Compose download exceeds the size limit")
                chunks.append(chunk)
            return b"".join(chunks)
    except ComposePluginError:
        raise
    except httpx.HTTPError as exc:
        raise ComposePluginError("Pinned Docker Compose binary could not be downloaded") from exc


def _publish_binary(path: Path, payload: bytes, *, account: UserAccount) -> tuple[int, int]:
    temporary_path: Path | None = None
    identity: tuple[int, int] | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=".docker-compose-", dir=path.parent)
        temporary_path = Path(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o755)
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails if any plugin appeared after
        # the initial lstat, so this helper never replaces another installation.
        os.link(temporary_path, path, follow_symlinks=False)
        metadata = path.lstat()
        identity = (metadata.st_dev, metadata.st_ino)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != account.uid
                or metadata.st_gid != account.gid or stat.S_IMODE(metadata.st_mode) != 0o755):
            raise ComposePluginError("Installed Compose plugin ownership or mode is unsafe")
        temporary_path.unlink()
        temporary_path = None
        return identity
    except FileExistsError as exc:
        raise ComposePluginError("A Compose plugin appeared at the user install path; it was left untouched") from exc
    except ComposePluginError:
        _remove_our_install(path, identity)
        raise
    except OSError as exc:
        _remove_our_install(path, identity)
        raise ComposePluginError("Pinned Compose plugin could not be atomically installed") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _remove_our_install(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        metadata = path.lstat()
        if (stat.S_ISREG(metadata.st_mode)
                and (metadata.st_dev, metadata.st_ino) == identity):
            path.unlink()
    except OSError:
        pass


def ensure_compose_plugin(
    access: DockerAccess,
    *,
    account: UserAccount,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Ensure active Docker can invoke Compose, installing only the pinned user plugin.

    The active Docker environment and context are never rewritten. A working
    plugin is accepted as-is; an occupied user plugin path is never replaced.
    """
    existing_version = _docker_version(access, runner=runner)
    if existing_version is not None and int(existing_version.split(".", 1)[0]) >= 2:
        return {
            "status": "ready",
            "source": "existing",
            "version": existing_version,
            "path": None,
            "sha256": None,
        }

    path = _plugin_path(account)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        metadata = None
    except OSError as exc:
        raise ComposePluginError("User Compose plugin path could not be inspected") from exc
    if metadata is not None:
        raise ComposePluginError("An existing Compose plugin occupies the user install path; it was left untouched")

    payload = _download_binary(transport=transport)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != COMPOSE_SHA256:
        raise ComposePluginError("Downloaded Docker Compose binary failed its pinned SHA-256 check")
    try:
        ensure_account_path(path.parent, account=account, mode=0o700)
    except AccountPathError as exc:
        raise ComposePluginError("User Docker configuration path is unsafe below the registered HOME") from exc

    identity: tuple[int, int] | None = None
    try:
        identity = _publish_binary(path, payload, account=account)
        installed_version = _docker_version(access, runner=runner)
        if installed_version != COMPOSE_VERSION:
            raise ComposePluginError("Docker did not invoke the newly installed pinned Compose plugin")
    except ComposePluginError:
        _remove_our_install(path, identity)
        raise

    return {
        "status": "ready",
        "source": "pinned_user_plugin",
        "version": installed_version,
        "path": str(path),
        "sha256": digest,
    }
