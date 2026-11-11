"""Build and describe the minimal, content-addressed node bootstrap bundle."""
from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import re
import shlex
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SITE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_RESERVED_SHELL_NAMES = {"HOME", "IFS", "PATH", "PWD", "SHELL", "TMUX", "TMUX_PANE"}
_MAX_BUNDLE_FILE = 128 * 1024 * 1024
_MAX_BUNDLE_UNPACKED = 256 * 1024 * 1024
_MAX_BUNDLE_MEMBERS = 4096
_DRAGONFLY_ARCHIVE_NAME = "dragonfly-client-v1.5.7-x86_64-unknown-linux-musl.tar.gz"
_PRIVATE_KEY_MARKER = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")


class BootstrapBundleError(ValueError):
    """The source tree or command inputs are not safe to package."""


@dataclass(frozen=True)
class BootstrapBundle:
    """A built bundle and its transport identity."""

    path: Path
    sha256: str
    size_bytes: int
    file_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _safe_regular(root: Path, relative: Path) -> Path:
    source = root / relative
    current = source
    while current != root:
        if current.is_symlink():
            raise BootstrapBundleError(f"bootstrap input traverses a symlink: {relative.as_posix()}")
        current = current.parent
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise BootstrapBundleError(f"required bootstrap file is missing: {relative.as_posix()}") from exc
    if not source.is_relative_to(root) or source.is_symlink() or not source.is_file():
        raise BootstrapBundleError(f"bootstrap input is not a regular file: {relative.as_posix()}")
    if metadata.st_size > _MAX_BUNDLE_FILE:
        raise BootstrapBundleError(f"bootstrap input exceeds the per-file size limit: {relative.as_posix()}")
    return source


def _excluded_deploy_file(relative: Path) -> bool:
    parts = relative.parts
    name = relative.name.lower()
    if "__pycache__" in parts or any(part == ".git" for part in parts):
        return True
    # Examples and real dotenv files are excluded alike. Runtime secrets must
    # be supplied through the existing private enrollment flow.
    return name.startswith(".env") or name.endswith((".env", ".pem", ".key"))


def _check_no_embedded_credentials(relative: str, source: Path) -> None:
    suffix = source.suffix.lower()
    if suffix == ".json":
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BootstrapBundleError(f"deployment JSON is invalid: {relative}") from exc

        def visit(item: object, key: str = "") -> None:
            if isinstance(item, dict):
                for child_key, child in item.items():
                    visit(child, str(child_key).lower())
            elif isinstance(item, list):
                for child in item:
                    visit(child, key)
            elif (isinstance(item, str) and item
                  and any(word in key for word in ("token", "secret", "password"))
                  and not key.endswith(("_file", "_path", "_url"))
                  and not item.startswith("file:")):
                raise BootstrapBundleError(f"deployment JSON embeds a credential value: {relative}")

        visit(value)
        return
    if suffix in {".py", ".md", ".sh", ".service", ".toml", ".txt", ".yaml", ".yml"} or relative == ".dockerignore":
        try:
            content = source.read_bytes()
        except OSError as exc:
            raise BootstrapBundleError(f"could not inspect bootstrap text file: {relative}") from exc
        if _PRIVATE_KEY_MARKER.search(content):
            raise BootstrapBundleError(f"bootstrap text file contains private key material: {relative}")


def _collect_sources(project_root: Path, assets_dir: Path | None) -> list[tuple[Path, str]]:
    root = project_root.expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise BootstrapBundleError("project root must be a real directory")
    root = root.resolve(strict=True)
    entries: dict[str, Path] = {}

    def add(relative: Path, *, archive_name: str | None = None) -> None:
        name = archive_name or PurePosixPath(*relative.parts).as_posix()
        if name in entries:
            return
        source = _safe_regular(root, relative)
        _check_no_embedded_credentials(name, source)
        entries[name] = source

    for fixed in (
        Path(".dockerignore"),
        Path("bin/qwen-rl"),
        Path("requirements/worker-cluster.txt"),
    ):
        add(fixed)

    for tree in (Path("qwen_rl/cluster"), Path("qwen_rl/worker"), Path("qwen_rl/contracts")):
        directory = root / tree
        if directory.is_symlink() or not directory.is_dir():
            raise BootstrapBundleError(f"required source package is missing: {tree.as_posix()}")
        for base, dirnames, filenames in os.walk(directory, followlinks=False):
            base_path = Path(base)
            kept: list[str] = []
            for dirname in sorted(dirnames):
                child = base_path / dirname
                if dirname == "__pycache__":
                    continue
                if base_path == root / "qwen_rl/cluster" and dirname == "fleet":
                    continue
                if child.is_symlink():
                    raise BootstrapBundleError(f"source package contains a symlink: {child.relative_to(root)}")
                kept.append(dirname)
            dirnames[:] = kept
            for filename in sorted(filenames):
                child = base_path / filename
                if child.suffix != ".py":
                    continue
                add(child.relative_to(root))

    for relative in (
        Path("qwen_rl/__init__.py"),
        Path("qwen_rl/__main__.py"),
        Path("qwen_rl/cli.py"),
        Path("qwen_rl/models/__init__.py"),
        Path("qwen_rl/models/artifacts.py"),
        Path("deploy/bootstrap/install.py"),
        Path("deploy/bootstrap/host-requirements.txt"),
        Path("deploy/bootstrap/README.md"),
    ):
        add(relative)
    add(Path("deploy/bootstrap/pyproject.toml"), archive_name="pyproject.toml")

    deploy = root / "deploy"
    if deploy.is_symlink() or not deploy.is_dir():
        raise BootstrapBundleError("deployment assets are missing")
    for base, dirnames, filenames in os.walk(deploy, followlinks=False):
        base_path = Path(base)
        relative_base = base_path.relative_to(root)
        kept = []
        for dirname in sorted(dirnames):
            child = base_path / dirname
            if dirname == "__pycache__":
                continue
            if base_path == deploy and dirname == "fleet":
                continue
            if child.is_symlink():
                raise BootstrapBundleError(f"deployment assets contain a symlink: {child.relative_to(root)}")
            kept.append(dirname)
        dirnames[:] = kept
        for filename in sorted(filenames):
            relative = relative_base / filename
            if _excluded_deploy_file(relative):
                continue
            source = base_path / filename
            if source.is_symlink():
                raise BootstrapBundleError(f"deployment assets contain a symlink: {relative.as_posix()}")
            if source.is_file():
                add(relative)

    if assets_dir is not None:
        asset_root = Path(assets_dir).expanduser().absolute()
        if asset_root.is_symlink() or not asset_root.is_dir():
            raise BootstrapBundleError("optional assets_dir must be a real directory")
        archive = asset_root / _DRAGONFLY_ARCHIVE_NAME
        if archive.is_symlink() or not archive.is_file():
            raise BootstrapBundleError(f"optional assets_dir must contain {_DRAGONFLY_ARCHIVE_NAME}")
        try:
            from .dragonfly import DRAGONFLY_CLIENT_LINUX_AMD64_MUSL_ARCHIVE_SHA256
        except ImportError as exc:  # pragma: no cover - package closure guard
            raise BootstrapBundleError("Dragonfly SHA pin is unavailable") from exc
        observed, size = _hash_file(archive)
        if size > _MAX_BUNDLE_FILE or not hmac.compare_digest(
            observed, DRAGONFLY_CLIENT_LINUX_AMD64_MUSL_ARCHIVE_SHA256,
        ):
            raise BootstrapBundleError("optional Dragonfly archive does not match the v1.5.7 SHA-256 pin")
        entries[f"assets/{_DRAGONFLY_ARCHIVE_NAME}"] = archive

    if len(entries) > _MAX_BUNDLE_MEMBERS:
        raise BootstrapBundleError("bootstrap bundle contains too many files")
    unpacked = sum(path.stat().st_size for path in entries.values())
    if unpacked > _MAX_BUNDLE_UNPACKED:
        raise BootstrapBundleError("bootstrap bundle exceeds the unpacked size limit")
    return [(path, name) for name, path in sorted(entries.items())]


def build_bootstrap_bundle(
    project_root: Path,
    destination: Path | None = None,
    *,
    assets_dir: Path | None = None,
) -> BootstrapBundle:
    """Write a deterministic SHA-addressed tar.gz of the minimal node runtime.

    The source allowlist contains the complete cluster and standalone Worker
    Python packages, the small model-artifact reader, and deployment assets.
    It never walks repository data, outputs, environments, or model weights.
    """
    sources = _collect_sources(Path(project_root), Path(assets_dir) if assets_dir is not None else None)
    if destination is None:
        descriptor, name = tempfile.mkstemp(prefix="qwen-cluster-bootstrap-", suffix=".tar.gz")
        os.close(descriptor)
        output = Path(name)
    else:
        output = Path(destination).expanduser().absolute()
        if output.is_symlink() or (output.exists() and not output.is_file()):
            raise BootstrapBundleError("bundle destination must be a regular file")
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        os.close(descriptor)
        temporary = Path(temp_name)
        output = temporary

    final_destination = Path(destination).expanduser().absolute() if destination is not None else output
    try:
        with output.open("wb") as raw:
            # Keep raw open after gzip finalizes so fsync includes the trailer.
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:  # noqa: SIM117
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for source, archive_name in sources:
                        metadata = source.stat()
                        if metadata.st_size > _MAX_BUNDLE_FILE:
                            raise BootstrapBundleError(f"bootstrap input grew beyond size limit: {archive_name}")
                        info = tarfile.TarInfo(archive_name)
                        info.size = metadata.st_size
                        info.mode = 0o755 if metadata.st_mode & 0o111 else 0o644
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        with source.open("rb") as stream:
                            archive.addfile(info, stream)
            raw.flush()
            os.fsync(raw.fileno())
        if destination is not None:
            os.replace(output, final_destination)
            output = final_destination
        digest, size = _hash_file(output)
        return BootstrapBundle(path=output, sha256=digest, size_bytes=size, file_count=len(sources))
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def _https_url(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) > 2048 or any(char in value for char in "\r\n\x00"):
        raise BootstrapBundleError(f"{name} must be a valid HTTPS URL")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise BootstrapBundleError(f"{name} must be a valid HTTPS URL") from exc
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise BootstrapBundleError(f"{name} must be a credential-free HTTPS URL")
    return value


def render_join_command(
    *,
    installer_url: str,
    installer_sha256: str,
    bundle_url: str,
    bundle_sha256: str,
    controller_url: str,
    site_id: str,
    code_env: str = "QWEN_CLUSTER_ENROLMENT_CODE",
    role: str = "worker",
    lan_host: str | None = None,
    gpu_uuid: str | None = None,
    state_dir: str = "/var/lib/qwen-cluster",
    project_root: str = "/opt/qwen-rl",
) -> str:
    """Render a copyable sudo command without embedding the enrollment code.

    The small installer is itself SHA-verified before execution. It then
    verifies the full source/deploy bundle against ``bundle_sha256`` over TLS.
    """
    installer_url = _https_url(installer_url, "installer_url")
    bundle_url = _https_url(bundle_url, "bundle_url")
    controller_url = _https_url(controller_url, "controller_url")
    if not _is_sha256(installer_sha256) or not _is_sha256(bundle_sha256):
        raise BootstrapBundleError("installer and bundle hashes must be lowercase SHA-256 values")
    if not isinstance(site_id, str) or not _SITE_ID.fullmatch(site_id):
        raise BootstrapBundleError("site_id is invalid")
    if role not in {"worker", "gateway"}:
        raise BootstrapBundleError("role must be worker or gateway")
    if not isinstance(code_env, str) or not _ENV_NAME.fullmatch(code_env):
        raise BootstrapBundleError("code_env must be a valid shell environment variable name")
    if code_env in _RESERVED_SHELL_NAMES:
        raise BootstrapBundleError("code_env is reserved by the host shell")
    for name, value in (("state_dir", state_dir), ("project_root", project_root)):
        if not isinstance(value, str) or not value.startswith("/") or any(c in value for c in "\r\n\x00"):
            raise BootstrapBundleError(f"{name} must be an absolute single-line path")
    if Path(project_root) != Path("/opt/qwen-rl"):
        raise BootstrapBundleError("project_root is fixed to /opt/qwen-rl for node bootstrap")
    for name, value in (("lan_host", lan_host), ("gpu_uuid", gpu_uuid)):
        if value is not None and (not isinstance(value, str) or not value or any(c in value for c in "\r\n\x00")):
            raise BootstrapBundleError(f"{name} must be a nonempty single-line value")

    # This tiny verifier uses only Ubuntu's system Python standard library.
    # It executes the installer only after TLS, final-origin, size and SHA
    # checks have passed. The enrollment code is stored in a private local file
    # and only its path crosses sudo and the installer process arguments.
    verifier = (
        "import hashlib,sys,urllib.request\n"
        "from urllib.parse import urlsplit\n"
        "url,expected,*args=sys.argv[1:]\n"
        "try:\n"
        " r=urllib.request.urlopen(url,timeout=60)\n"
        " final=urlsplit(r.geturl())\n"
        " data=r.read(1048577); r.close()\n"
        " if final.scheme!='https' or final.hostname!=urlsplit(url).hostname or len(data)>1048576: raise ValueError()\n"
        " if hashlib.sha256(data).hexdigest()!=expected: raise ValueError()\n"
        " sys.argv=[url,*args]\n"
        " exec(compile(data,url,'exec'),{'__name__':'__main__','__file__':url})\n"
        "except SystemExit:\n"
        " raise\n"
        "except BaseException:\n"
        " print('Qwen cluster installer TLS/SHA verification failed',file=sys.stderr)\n"
        " raise SystemExit(1)\n"
    )
    args = [
        "--bundle-url", bundle_url,
        "--bundle-sha256", bundle_sha256,
        "--controller", controller_url,
        "--site", site_id,
        "--role", role,
        "--state-dir", state_dir,
        "--project-root", project_root,
    ]
    if lan_host is not None:
        args.extend(("--lan-host", lan_host))
    if gpu_uuid is not None:
        args.extend(("--gpu-uuid", gpu_uuid))
    sudo_command = shlex.join([
        "sudo", "python3", "-c", verifier,
        installer_url, installer_sha256, *args,
    ]) + ' --code-file "$_QWEN_CLUSTER_CODE_FILE"'
    # A multiline shell block prompts without echoing the code, writes a
    # mode-0600 temporary file, and passes only that file path through sudo.
    return "\n".join((
        "set -eu",
        "umask 077",
        f"read -r -s -p 'Enrollment code: ' {code_env}",
        "printf '\\n'",
        "_QWEN_CLUSTER_CODE_FILE=\"$(mktemp /tmp/qwen-cluster-enrollment-XXXXXXXX)\"",
        f"trap 'rm -f -- \"$_QWEN_CLUSTER_CODE_FILE\"; unset {code_env}' EXIT",
        f"printf '%s' \"${code_env}\" > \"$_QWEN_CLUSTER_CODE_FILE\"",
        f"unset {code_env}",
        sudo_command,
    ))
