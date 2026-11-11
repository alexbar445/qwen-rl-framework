"""Host prerequisites, package policy, and pinned host tooling."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Callable, Literal, Sequence
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contracts import GPUCapability


_DRAGONFLY_CLIENT_ARCHIVE_URL = (
    "https://github.com/dragonflyoss/client/releases/download/v1.5.7/"
    "dragonfly-client-v1.5.7-x86_64-unknown-linux-musl.tar.gz"
)
_NVIDIA_KEY_URL = "https://nvidia.github.io/libnvidia-container/gpgkey"
_NVIDIA_KEY_FINGERPRINT = "C95B321B61E88C1809C4F759DDCAE044F796ECB0"


class HostError(RuntimeError):
    """Expected failure while inspecting or preparing Ubuntu host prerequisites."""


class HostPrerequisites(BaseModel):
    """Typed, read-only host readiness receipt consumed by node activation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["passed", "needs_driver", "needs_host_packages"] = "passed"
    needs_driver: bool = False
    driver_reason: str | None = None
    needs_reboot: bool = False
    needs_host_packages: bool = False
    ubuntu_version: str | None = None
    architecture: str | None = None
    gpus: list[GPUCapability] = Field(default_factory=list)
    driver_versions: list[str] = Field(default_factory=list)
    model_path: str | None = None
    model_ready: bool = False
    state_path: str | None = None
    state_free_bytes: int = 0
    model_free_bytes: int = 0
    docker_runtime: str = "not_checked"


def _run_command(
    argv: Sequence[str], *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 60.0, input_text: str | None = None,
) -> subprocess.CompletedProcess:
    if not argv or any(not isinstance(part, str) or "\x00" in part for part in argv):
        raise HostError("invalid command argument")
    try:
        result = runner(list(argv), check=False, capture_output=True, text=True,
                        timeout=timeout, input=input_text)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostError(f"could not run {Path(argv[0]).name}") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        detail = stderr[-1][:240] if stderr else "command failed"
        raise HostError(f"{Path(argv[0]).name} failed: {detail}")
    return result

def detect_lan_host(*, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> str:
    """Choose one unambiguous private IPv4 address or fail for manual selection."""
    ip = shutil.which("ip")
    if not ip:
        raise HostError("iproute2 is required to discover the node LAN address")
    output = _run_command([ip, "-j", "-4", "addr", "show", "scope", "global"], runner=runner).stdout
    try:
        interfaces = json.loads(output)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HostError("iproute2 returned invalid JSON while discovering the node LAN address") from exc
    if not isinstance(interfaces, list):
        raise HostError("iproute2 returned an invalid address inventory")
    addresses: set[str] = set()
    for interface in interfaces:
        if not isinstance(interface, dict):
            raise HostError("iproute2 returned an invalid interface record")
        address_records = interface.get("addr_info", [])
        if not isinstance(address_records, list):
            raise HostError("iproute2 returned an invalid interface address list")
        for record in address_records:
            if not isinstance(record, dict):
                raise HostError("iproute2 returned an invalid interface address record")
            if record.get("family") != "inet":
                continue
            candidate = record.get("local")
            try:
                address = ipaddress.ip_address(candidate)
            except (TypeError, ValueError):
                continue
            prefix_length = record.get("prefixlen")
            if (address.version == 4 and isinstance(prefix_length, int)
                    and not isinstance(prefix_length, bool) and 0 <= prefix_length <= 32
                    and address.is_private and not address.is_loopback
                    and not address.is_link_local and not address.is_unspecified
                    and not address.is_multicast):
                addresses.add(str(address))
    if len(addresses) != 1:
        raise HostError("LAN address is ambiguous; pass --lan-host with the reachable site address")
    return next(iter(addresses))


def read_boot_id(path: Path = Path("/proc/sys/kernel/random/boot_id")) -> str:
    try:
        value = Path(path).read_text(encoding="ascii").strip()
    except OSError as exc:
        raise HostError("Linux boot ID is unavailable") from exc
    if not value or "\n" in value or "\x00" in value:
        raise HostError("Linux boot ID is invalid")
    return value


def probe_gpus(*, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[GPUCapability]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    args = [binary, "--query-gpu=uuid,index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"]
    try:
        result = runner(args, check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    output = result.stdout or ""
    gpus: list[GPUCapability] = []
    for row in output.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 5:
            raise HostError("nvidia-smi returned malformed GPU inventory")
        uuid, index, name, memory, capability = parts
        try:
            gpus.append(GPUCapability(uuid=uuid, index=int(index), name=name,
                                      memory_total_mib=int(memory), compute_capability=capability or None))
        except (ValueError, ValidationError) as exc:
            raise HostError("nvidia-smi returned invalid GPU data") from exc
    return gpus


def discover_gpus(*, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[GPUCapability]:
    gpus = probe_gpus(runner=runner)
    if not gpus:
        raise HostError("no usable NVIDIA GPU was detected; install a compatible driver before GPU activation")
    return gpus


def selected_gpu(gpus: Sequence[GPUCapability], requested_uuid: str | None = None) -> GPUCapability:
    if requested_uuid:
        for gpu in gpus:
            if gpu.uuid == requested_uuid:
                return gpu
        raise HostError("requested GPU UUID is not present")
    if len(gpus) != 1:
        raise HostError("node has multiple GPUs; select one by exact GPU UUID")
    return gpus[0]


def install_dragonfly_binaries(
    *, bin_dir: Path = Path("/usr/local/bin"), source_archive: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    http_transport: httpx.BaseTransport | None = None,
) -> dict[str, str]:
    """Install the pinned x86_64 musl dfget/dfdaemon pair atomically by SHA."""
    from .dragonfly import (
        DRAGONFLY_CLIENT_LINUX_AMD64_MUSL_ARCHIVE_SHA256,
        DRAGONFLY_CLIENT_VERSION,
        DRAGONFLY_DFDAEMON_LINUX_AMD64_MUSL_SHA256,
        DRAGONFLY_DFGET_LINUX_AMD64_MUSL_SHA256,
    )

    bin_dir = Path(bin_dir).expanduser().absolute()
    if bin_dir.exists() and (bin_dir.is_symlink() or not bin_dir.is_dir()):
        raise HostError("Dragonfly binary directory is not a directory")
    bin_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    targets = {
        "dfget": (DRAGONFLY_DFGET_LINUX_AMD64_MUSL_SHA256, bin_dir / "dfget"),
        "dfdaemon": (DRAGONFLY_DFDAEMON_LINUX_AMD64_MUSL_SHA256, bin_dir / "dfdaemon"),
    }

    def file_sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def version_ok(path: Path) -> bool:
        try:
            result = runner([str(path), "--version"], check=False, capture_output=True,
                            text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return False
        output = f"{result.stdout or ''}\n{result.stderr or ''}"
        return result.returncode == 0 and re.search(
            rf"\b[vV]?{re.escape(DRAGONFLY_CLIENT_VERSION)}\b", output,
        ) is not None

    if all(path.is_file() and not path.is_symlink() and file_sha(path) == expected and version_ok(path)
           for expected, path in targets.values()):
        return {name: str(path) for name, (_, path) in targets.items()}

    archive_tmp: Path | None = None
    if source_archive is not None:
        source_archive = Path(source_archive)
        if source_archive.is_symlink() or not source_archive.is_file():
            raise HostError("Dragonfly archive source must be a regular file")
        archive_path = source_archive
    else:
        fd, name = tempfile.mkstemp(prefix="qwen-dragonfly-v1.5.7-", suffix=".tar.gz")
        archive_tmp = Path(name)
        digest = hashlib.sha256()
        total = 0
        try:
            with os.fdopen(fd, "wb") as output:
                with httpx.Client(verify=True, transport=http_transport, timeout=60.0,
                                  follow_redirects=True, max_redirects=5) as client:
                    with client.stream("GET", _DRAGONFLY_CLIENT_ARCHIVE_URL) as response:
                        final = urlsplit(str(response.url))
                        if (response.status_code != 200 or final.scheme != "https"
                                or not (final.hostname == "github.com" or final.hostname.endswith(".githubusercontent.com"))):
                            raise HostError("Dragonfly client archive source is not the pinned HTTPS release")
                        length = response.headers.get("content-length")
                        if length and length.isdigit() and int(length) > 128 * 1024 * 1024:
                            raise HostError("Dragonfly client archive exceeds the size limit")
                        for chunk in response.iter_bytes():
                            total += len(chunk)
                            if total > 128 * 1024 * 1024:
                                raise HostError("Dragonfly client archive exceeds the size limit")
                            digest.update(chunk)
                            output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if not hmac.compare_digest(digest.hexdigest(), DRAGONFLY_CLIENT_LINUX_AMD64_MUSL_ARCHIVE_SHA256):
                raise HostError("Dragonfly client archive SHA-256 does not match the release pin")
            archive_path = archive_tmp
        except BaseException:
            archive_tmp.unlink(missing_ok=True)
            raise

    try:
        if file_sha(archive_path) != DRAGONFLY_CLIENT_LINUX_AMD64_MUSL_ARCHIVE_SHA256:
            raise HostError("Dragonfly client archive SHA-256 does not match the release pin")
        staged: dict[str, Path] = {}
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                parts = member.name.replace("\\", "/").split("/")
                if member.name.startswith("/") or any(part in {"", ".", ".."} for part in parts):
                    raise HostError("Dragonfly archive contains an unsafe path")
                basename = parts[-1]
                if basename not in targets:
                    continue
                if not member.isfile() or member.size > 256 * 1024 * 1024 or basename in staged:
                    raise HostError("Dragonfly archive has an invalid or duplicate executable")
                source = archive.extractfile(member)
                if source is None:
                    raise HostError("Dragonfly archive executable cannot be read")
                fd, stage_name = tempfile.mkstemp(prefix=f".{basename}.", dir=bin_dir)
                stage = Path(stage_name)
                staged[basename] = stage
                digest = hashlib.sha256()
                total = 0
                try:
                    with os.fdopen(fd, "wb") as output:
                        while chunk := source.read(1024 * 1024):
                            total += len(chunk)
                            digest.update(chunk)
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    os.chmod(stage, 0o755)
                finally:
                    source.close()
                expected = targets[basename][0]
                if total != member.size or not hmac.compare_digest(digest.hexdigest(), expected):
                    raise HostError(f"Dragonfly {basename} binary SHA-256 does not match the release pin")
                if not version_ok(stage):
                    raise HostError(f"Dragonfly {basename} version does not match {DRAGONFLY_CLIENT_VERSION}")
        if set(staged) != set(targets):
            raise HostError("Dragonfly release archive is missing a pinned executable")
        for name, stage in staged.items():
            os.replace(stage, targets[name][1])
        directory_fd = os.open(bin_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {name: str(path) for name, (_, path) in targets.items()}
    finally:
        if archive_tmp is not None:
            archive_tmp.unlink(missing_ok=True)


def preflight_host(
    *, model_path: Path, state_dir: Path, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    os_release: Path = Path("/etc/os-release"), minimum_free_bytes: int = 80 * 1024**3,
    require_runtime: bool = True, require_gpu: bool = True,
) -> HostPrerequisites:
    """Read-only preflight; missing driver/packages are returned as install states."""
    try:
        fields = {}
        for line in Path(os_release).read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                fields[key] = value.strip().strip('"')
    except OSError as exc:
        raise HostError("cannot read /etc/os-release") from exc
    if fields.get("ID") != "ubuntu":
        raise HostError("only Ubuntu hosts are supported")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise HostError("only x86_64 hosts are supported")
    gpu_rows: list[GPUCapability] = []
    driver_versions: list[str] = []
    needs_driver = False
    driver_reason: str | None = None
    if require_gpu:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            needs_driver = True
            driver_reason = "nvidia_driver_missing"
        else:
            gpu_rows = probe_gpus(runner=runner)
            if not gpu_rows:
                needs_driver = True
                driver_reason = "nvidia_driver_unavailable_or_no_gpu"
            else:
                driver_result = runner(
                    [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                    check=False, capture_output=True, text=True, timeout=20,
                )
                driver_versions = [line.strip() for line in (driver_result.stdout or "").splitlines() if line.strip()]
                if driver_result.returncode != 0 or len(driver_versions) < len(gpu_rows):
                    needs_driver = True
                    driver_reason = "nvidia_driver_version_unavailable"
                else:
                    try:
                        driver_branch = min(int(version.split(".", 1)[0]) for version in driver_versions)
                    except ValueError as exc:
                        raise HostError("NVIDIA driver version format is invalid") from exc
                    if driver_branch < 580:
                        needs_driver = True
                        driver_reason = "nvidia_driver_below_cuda13_minimum"
    needs_host_packages = False
    docker_runtime = "not_checked"
    docker = shutil.which("docker")
    if require_runtime and not docker:
        needs_host_packages = True
        docker_runtime = "docker_missing"
    if require_runtime:
        if docker:
            runtime_result = runner([docker, "--host", "unix:///var/run/docker.sock", "info", "--format",
                                     "{{json .Runtimes}}"], check=False, capture_output=True,
                                    text=True, timeout=30)
            if runtime_result.returncode != 0:
                needs_host_packages = True
                docker_runtime = "daemon_unavailable"
            else:
                try:
                    runtimes = json.loads(runtime_result.stdout or "")
                except json.JSONDecodeError as exc:
                    raise HostError("Docker runtime inventory is invalid") from exc
                has_nvidia = isinstance(runtimes, dict) and "nvidia" in runtimes
                docker_runtime = "nvidia" if has_nvidia else "docker"
                if require_gpu and not has_nvidia:
                    needs_host_packages = True
                if require_gpu and not needs_driver:
                    container_cli = shutil.which("nvidia-container-cli")
                    if not container_cli:
                        needs_host_packages = True
                    else:
                        probe = runner([container_cli, "info"], check=False, capture_output=True,
                                       text=True, timeout=30)
                        if probe.returncode != 0:
                            needs_host_packages = True
            compose = runner([docker, "--host", "unix:///var/run/docker.sock", "compose", "version"],
                             check=False, capture_output=True, text=True, timeout=30)
            if compose.returncode != 0:
                needs_host_packages = True
    model_path = Path(model_path).expanduser().absolute()
    model_ready = model_path.is_dir() and os.access(model_path, os.R_OK | os.X_OK)
    if model_path.exists() and not model_ready:
        raise HostError("existing model artifact path is unreadable or not a directory")
    state_dir = Path(state_dir).expanduser().absolute()
    def existing_anchor(path: Path) -> Path:
        candidate = path if path.exists() else path.parent
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        if not candidate.is_dir():
            raise HostError("filesystem anchor is not a directory")
        return candidate

    state_anchor = existing_anchor(state_dir)
    model_anchor = existing_anchor(model_path)
    for label, anchor in (("node state", state_anchor), ("model cache", model_anchor)):
        if not os.access(anchor, os.W_OK | os.X_OK):
            raise HostError(f"{label} filesystem is not writable")
    stat_state = os.statvfs(state_anchor)
    state_free_bytes = stat_state.f_bavail * stat_state.f_frsize
    stat_model = os.statvfs(model_anchor)
    model_free_bytes = stat_model.f_bavail * stat_model.f_frsize
    if state_free_bytes < minimum_free_bytes or model_free_bytes < minimum_free_bytes:
        raise HostError(f"node state or model filesystem has less than {minimum_free_bytes} free bytes")
    status = "needs_driver" if needs_driver else ("needs_host_packages" if needs_host_packages else "passed")
    return HostPrerequisites(
        status=status,
        needs_driver=needs_driver,
        driver_reason=driver_reason,
        needs_reboot=False,
        needs_host_packages=needs_host_packages,
        ubuntu_version=fields.get("VERSION_ID"),
        architecture=platform.machine(),
        gpus=gpu_rows,
        driver_versions=driver_versions,
        model_path=str(model_path),
        model_ready=model_ready,
        state_path=str(state_dir),
        state_free_bytes=state_free_bytes,
        model_free_bytes=model_free_bytes,
        docker_runtime=docker_runtime,
    )


def write_system_file(path: Path, content: bytes, *, mode: int) -> None:
    """Atomic write for apt/systemd destinations without chmodding their parent."""
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    if path.is_symlink():
        raise HostError(f"refusing to replace system symlink {path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _download_official_nvidia_key(
    *, transport: httpx.BaseTransport | None = None,
) -> bytes:
    try:
        with httpx.Client(verify=True, transport=transport, timeout=30, follow_redirects=False) as client:
            response = client.get(_NVIDIA_KEY_URL)
    except httpx.HTTPError as exc:
        raise HostError("could not fetch NVIDIA Container Toolkit repository key over HTTPS") from exc
    if response.status_code != 200 or response.url.scheme != "https" or response.url.host != "nvidia.github.io":
        raise HostError("NVIDIA repository key source did not match the official HTTPS origin")
    if not response.content or len(response.content) > 1024 * 1024:
        raise HostError("NVIDIA repository key payload is invalid")
    return response.content


def configure_nvidia_apt_repository(
    *, apt_root: Path = Path("/etc/apt"), runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    http_transport: httpx.BaseTransport | None = None,
) -> dict[str, str]:
    """Install only NVIDIA's verified GPG key and stable signed apt source."""
    apt_root = Path(apt_root).absolute()
    keyring_dir = apt_root / "keyrings"
    list_dir = apt_root / "sources.list.d"
    key_path = keyring_dir / "qwen-nvidia-container-toolkit.gpg"
    source_path = list_dir / "qwen-nvidia-container-toolkit.list"
    architecture = "amd64"
    source_text = (
        f"deb [arch={architecture} signed-by={key_path}] "
        "https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /\n"
    )
    gpg = shutil.which("gpg") or "gpg"
    if key_path.exists():
        if key_path.is_symlink() or not key_path.is_file():
            raise HostError("NVIDIA apt keyring path is not a regular file")
        shown = runner([gpg, "--show-keys", "--with-colons", "--fingerprint", str(key_path)],
                       check=False, capture_output=True, text=True, timeout=20)
        fingerprints = [line.split(":")[9].upper() for line in (shown.stdout or "").splitlines()
                        if line.startswith("fpr:") and len(line.split(":")) > 9]
        if shown.returncode != 0 or _NVIDIA_KEY_FINGERPRINT not in fingerprints:
            raise HostError("existing NVIDIA apt keyring did not match the pinned publisher key")
        if source_path.exists():
            if source_path.is_symlink() or not source_path.is_file():
                raise HostError("NVIDIA apt source path is not a regular file")
            if source_path.read_text(encoding="ascii") != source_text:
                raise HostError("existing NVIDIA apt source differs from the pinned stable source")
            return {"keyring": str(key_path), "source": str(source_path), "fingerprint": _NVIDIA_KEY_FINGERPRINT}
        write_system_file(source_path, source_text.encode("ascii"), mode=0o644)
        return {"keyring": str(key_path), "source": str(source_path), "fingerprint": _NVIDIA_KEY_FINGERPRINT}
    key_bytes = _download_official_nvidia_key(transport=http_transport)
    temp_dir = Path(tempfile.mkdtemp(prefix="qwen-nvidia-apt-key-"))
    raw_key = temp_dir / "nvidia.gpg"
    armored_keyring = temp_dir / "nvidia-keyring.gpg"
    try:
        raw_key.write_bytes(key_bytes)
        raw_key.chmod(0o600)
        shown = runner([gpg, "--show-keys", "--with-colons", "--fingerprint", str(raw_key)],
                       check=False, capture_output=True, text=True, timeout=20)
        if shown.returncode != 0:
            raise HostError("GPG could not inspect NVIDIA's repository key")
        fingerprints = [line.split(":")[9].upper() for line in shown.stdout.splitlines()
                        if line.startswith("fpr:") and len(line.split(":")) > 9]
        if _NVIDIA_KEY_FINGERPRINT not in fingerprints:
            raise HostError("NVIDIA repository key fingerprint did not match the pinned publisher key")
        _run_command([gpg, "--dearmor", "--output", str(armored_keyring), str(raw_key)],
                     runner=runner, timeout=20)
        if not armored_keyring.is_file() or not armored_keyring.stat().st_size:
            raise HostError("GPG did not create the NVIDIA apt keyring")
        write_system_file(key_path, armored_keyring.read_bytes(), mode=0o644)
        write_system_file(source_path, source_text.encode("ascii"), mode=0o644)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return {"keyring": str(key_path), "source": str(source_path), "fingerprint": _NVIDIA_KEY_FINGERPRINT}


def docker_has_running_jobs(
    *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    result = runner([docker, "--host", "unix:///var/run/docker.sock", "ps", "--quiet"],
                    check=False, capture_output=True, text=True, timeout=20)
    if result.returncode == 0:
        return bool((result.stdout or "").strip())
    systemctl = shutil.which("systemctl") or "systemctl"
    active = runner([systemctl, "is-active", "docker"], check=False, capture_output=True,
                    text=True, timeout=15)
    if active.returncode == 0 and (active.stdout or "").strip() == "active":
        raise HostError("could not determine whether Docker workloads are running")
    return False


def configure_nvidia_container_runtime(
    *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Configure Docker's NVIDIA runtime and reload only with no live containers."""
    docker = shutil.which("docker") or "/usr/bin/docker"
    systemctl = shutil.which("systemctl") or "systemctl"
    nvidia_ctk = shutil.which("nvidia-ctk") or "/usr/bin/nvidia-ctk"
    info = runner([docker, "--host", "unix:///var/run/docker.sock", "info", "--format", "{{json .Runtimes}}"],
                  check=False, capture_output=True, text=True, timeout=30)
    if info.returncode == 0:
        try:
            runtimes = json.loads(info.stdout or "")
        except json.JSONDecodeError as exc:
            raise HostError("Docker runtime inventory is invalid") from exc
        if isinstance(runtimes, dict) and "nvidia" in runtimes:
            return
    if docker_has_running_jobs(runner=runner):
        raise HostError("refusing to reload Docker while workloads are running")
    _run_command([nvidia_ctk, "runtime", "configure", "--runtime=docker"], runner=runner, timeout=30)
    active = runner([systemctl, "is-active", "docker"], check=False, capture_output=True,
                    text=True, timeout=15)
    if active.returncode == 0 and (active.stdout or "").strip() == "active":
        if docker_has_running_jobs(runner=runner):
            raise HostError("Docker acquired a running workload before its runtime reload")
        _run_command([systemctl, "restart", "docker"], runner=runner, timeout=120)
    else:
        _run_command([systemctl, "enable", "--now", "docker"], runner=runner, timeout=120)
    verified = runner([docker, "--host", "unix:///var/run/docker.sock", "info", "--format", "{{json .Runtimes}}"],
                      check=False, capture_output=True, text=True, timeout=30)
    try:
        runtimes = json.loads(verified.stdout or "") if verified.returncode == 0 else None
    except json.JSONDecodeError as exc:
        raise HostError("Docker runtime inventory after configuration is invalid") from exc
    if not isinstance(runtimes, dict) or "nvidia" not in runtimes:
        raise HostError("Docker still does not expose the NVIDIA runtime after configuration")


def install_host_packages(
    *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    packages: Sequence[str] = (
        "ca-certificates", "gnupg", "openssh-client", "openssh-server",
        "docker.io", "docker-compose-v2", "nvidia-container-toolkit",
    ),
    allow_package_changes: bool = False,
    apt_root: Path = Path("/etc/apt"),
    http_transport: httpx.BaseTransport | None = None,
) -> list[str]:
    """Install allowlisted packages without touching drivers or active jobs.

    This is intentionally opt-in. It refuses package changes while Docker has
    running containers and never runs a driver installer or reboot command.
    """
    if not allow_package_changes:
        raise HostError("package installation requires the explicit --install-host-packages option")
    allowed = {
        "ca-certificates", "gnupg", "openssh-client", "openssh-server",
        "docker.io", "docker-compose-v2", "nvidia-container-toolkit",
    }
    if not packages or any(package not in allowed for package in packages):
        raise HostError("installer package list contains an unsupported package")
    dpkg = shutil.which("dpkg-query") or "dpkg-query"
    missing: list[str] = []
    for package in dict.fromkeys(packages):
        result = runner([dpkg, "-W", "-f=${db:Status-Status}", package], check=False,
                        capture_output=True, text=True, timeout=20)
        if result.returncode != 0 or result.stdout.strip() != "installed":
            missing.append(package)
    docker = shutil.which("docker")
    if docker and "docker.io" in missing:
        missing.remove("docker.io")
    if docker and "docker-compose-v2" in missing:
        compose = runner([docker, "--host", "unix:///var/run/docker.sock", "compose", "version"],
                         check=False, capture_output=True, text=True, timeout=30)
        if compose.returncode == 0:
            missing.remove("docker-compose-v2")
    needs_nvidia_repo = "nvidia-container-toolkit" in missing
    has_changes = bool(missing)
    if not has_changes:
        if "nvidia-container-toolkit" in packages:
            configure_nvidia_container_runtime(runner=runner)
        return []
    if has_changes and docker_has_running_jobs(runner=runner):
        raise HostError("refusing host package changes while Docker jobs are running")
    apt = shutil.which("apt-get") or "apt-get"
    if missing:
        _run_command([apt, "update"], runner=runner, timeout=900)
        prerequisites = [name for name in ("ca-certificates", "gnupg") if name in missing]
        if prerequisites:
            _run_command([apt, "install", "--yes", "--no-install-recommends", *prerequisites],
                         runner=runner, timeout=900)
            missing = [name for name in missing if name not in prerequisites]
    if needs_nvidia_repo:
        configure_nvidia_apt_repository(apt_root=apt_root, runner=runner, http_transport=http_transport)
        if "nvidia-container-toolkit" in missing:
            _run_command([apt, "update"], runner=runner, timeout=900)
    if missing:
        _run_command([apt, "install", "--yes", "--no-install-recommends", *missing],
                     runner=runner, timeout=1800)
    if needs_nvidia_repo and shutil.which("nvidia-ctk"):
        configure_nvidia_container_runtime(runner=runner)
    return missing
