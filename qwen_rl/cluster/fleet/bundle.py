"""Build the public role-neutral Fleet bootstrap release.

The builder writes local artifacts and hash receipts only. Publication into the
upload directory is a separate, operator-reviewed action.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import hmac
import json
import os
import re
import shlex
import ssl
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_FILE_BYTES = 128 * 1024 * 1024
_MAX_UNPACKED_BYTES = 256 * 1024 * 1024
_MAX_MEMBERS = 4096
_PRIVATE_KEY_MARKER = re.compile(
    b"-----BEGIN " + rb"(?:RSA |EC |OPENSSH |DSA )?" + b"PRIVATE " + b"KEY-----"
)
UV_VERSION = "0.12.24"
UV_ARCHIVE_SHA256 = "b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6"
UV_ARCHIVE_FILENAME = f"uv-{UV_VERSION}-{UV_ARCHIVE_SHA256}.tar.gz"
UV_UPSTREAM_URL = (
    "https://releases.astral.sh/github/uv/releases/download/0.12.24/"
    "uv-x86_64-unknown-linux-gnu.tar.gz"
)
_EXACT_FILES = {
    "qwen_rl/__init__.py",
    "qwen_rl/__main__.py",
    "qwen_rl/cli.py",
    "qwen_rl/contracts/__init__.py",
    "qwen_rl/models/__init__.py",
    "qwen_rl/models/artifacts.py",
    "qwen_rl/deployment/__init__.py",
    "qwen_rl/cluster/fleet/paths.py",
    "qwen_rl/cluster/fleet/upgrade.py",
    "qwen_rl/cluster/fleet/watchdog.py",
    "deploy/fleet/bootstrap/install.py",
    "deploy/fleet/bootstrap/host-requirements.txt",
    "deploy/fleet/bootstrap/qwen-fleet-node.service",
    "deploy/bootstrap/install.py",
    "deploy/bootstrap/host-requirements.txt",
    "deploy/learner/install.py",
}


class FleetBundleError(ValueError):
    """Source closure, origin or public release input is unsafe."""


@dataclass(frozen=True)
class FleetSourceBundle:
    path: Path
    sha256: str
    size_bytes: int
    file_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FleetBootstrapRelease:
    output_dir: Path
    script_path: Path
    source_bundle_path: Path
    receipt_path: Path
    script_sha256: str
    source_bundle_sha256: str
    start_command: str
    uv_artifact_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "script_path": str(self.script_path),
            "source_bundle_path": str(self.source_bundle_path),
            "receipt_path": str(self.receipt_path),
            "script_sha256": self.script_sha256,
            "source_bundle_sha256": self.source_bundle_sha256,
            "start_command": self.start_command,
            "uv_artifact_path": str(self.uv_artifact_path),
        }


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _safe_regular(root: Path, relative: Path) -> Path:
    source = root / relative
    current = source
    while current != root:
        if current.is_symlink():
            raise FleetBundleError(f"Fleet source traverses a symlink: {relative.as_posix()}")
        current = current.parent
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise FleetBundleError(f"required Fleet source file is missing: {relative.as_posix()}") from exc
    if not source.is_relative_to(root) or not stat_is_regular(metadata.st_mode) or source.is_symlink():
        raise FleetBundleError(f"Fleet source is not a regular file: {relative.as_posix()}")
    if metadata.st_size > _MAX_FILE_BYTES:
        raise FleetBundleError(f"Fleet source file exceeds the size limit: {relative.as_posix()}")
    return source


def stat_is_regular(mode: int) -> bool:
    # Kept local so this source builder remains a standard-library utility.
    import stat
    return stat.S_ISREG(mode)


def _is_source_member(value: str) -> bool:
    if value in _EXACT_FILES:
        return True
    parts = PurePosixPath(value).parts
    # qwen_rl/transport is imported at module import time by qwen_rl/cluster/{artifacts,policy_seed,
    # adapter_transport}.py, so a bundle without it cannot even import the management agent.
    if len(parts) >= 3 and parts[:2] in {("qwen_rl", "cluster"), ("qwen_rl", "contracts"),
                                         ("qwen_rl", "transport")}:
        return value.endswith(".py")
    if len(parts) >= 4 and parts[:3] == ("qwen_rl", "deployment", "learner"):
        return value.endswith(".py")
    return False


def _check_public_source(relative: str, source: Path) -> None:
    data = source.read_bytes()
    if _PRIVATE_KEY_MARKER.search(data):
        raise FleetBundleError(f"Fleet public source contains private key material: {relative}")
    if source.suffix.lower() in {".json", ".jsonl", ".sqlite", ".db", ".pem", ".key"}:
        raise FleetBundleError(f"Fleet public bundle excludes data and credential file types: {relative}")
    if relative.startswith(("configs/", "outputs/", "data/", "private/", "catalog/")):
        raise FleetBundleError(f"Fleet public bundle excludes private catalogs and runtime data: {relative}")


def _collect_sources(project_root: Path) -> list[tuple[Path, str]]:
    requested = Path(project_root).expanduser().absolute()
    if requested.is_symlink() or not requested.is_dir():
        raise FleetBundleError("project root must be a real directory")
    root = requested.resolve(strict=True)
    entries: dict[str, Path] = {}

    def add(relative: Path) -> None:
        name = PurePosixPath(*relative.parts).as_posix()
        if not _is_source_member(name):
            raise FleetBundleError(f"source path is outside the public Fleet allowlist: {name}")
        if name in entries:
            return
        source = _safe_regular(root, relative)
        _check_public_source(name, source)
        entries[name] = source

    for tree in (Path("qwen_rl/cluster"), Path("qwen_rl/contracts"), Path("qwen_rl/transport"),
                 Path("qwen_rl/deployment/learner")):
        directory = root / tree
        if directory.is_symlink() or not directory.is_dir():
            raise FleetBundleError(f"required public source package is missing: {tree.as_posix()}")
        for base, dirnames, filenames in os.walk(directory, followlinks=False):
            base_path = Path(base)
            kept: list[str] = []
            for dirname in sorted(dirnames):
                child = base_path / dirname
                if dirname == "__pycache__":
                    continue
                if child.is_symlink():
                    raise FleetBundleError(f"Fleet package contains a symlink: {child.relative_to(root)}")
                kept.append(dirname)
            dirnames[:] = kept
            for filename in sorted(filenames):
                child = base_path / filename
                if child.suffix == ".py":
                    add(child.relative_to(root))

    for value in sorted(_EXACT_FILES):
        add(Path(value))
    if "qwen_rl/cluster/__init__.py" not in entries or "qwen_rl/cluster/fleet/agent.py" not in entries:
        raise FleetBundleError("Fleet agent package is incomplete")
    missing = _EXACT_FILES - entries.keys()
    if missing:
        raise FleetBundleError("Fleet public source closure is incomplete")
    if len(entries) > _MAX_MEMBERS:
        raise FleetBundleError("Fleet source bundle contains too many files")
    if sum(path.stat().st_size for path in entries.values()) > _MAX_UNPACKED_BYTES:
        raise FleetBundleError("Fleet source bundle exceeds the unpacked size limit")
    return [(source, name) for name, source in sorted(entries.items())]


def build_source_bundle(project_root: Path, destination: Path | None = None) -> FleetSourceBundle:
    """Build a deterministic public Fleet+role-installer source archive."""
    sources = _collect_sources(Path(project_root))
    if destination is None:
        descriptor, name = tempfile.mkstemp(prefix="qwen-fleet-source-", suffix=".tar.gz", dir="/tmp")
        os.close(descriptor)
        output = Path(name)
    else:
        output = Path(destination).expanduser().absolute()
        if output.is_symlink() or (output.exists() and not output.is_file()):
            raise FleetBundleError("source bundle destination must be a regular file")
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        os.close(descriptor)
        output = Path(name)
    final_destination = Path(destination).expanduser().absolute() if destination is not None else output
    try:
        with output.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for source, archive_name in sources:
                        info = tarfile.TarInfo(archive_name)
                        info.size = source.stat().st_size
                        info.mode = 0o644
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        with source.open("rb") as stream:
                            archive.addfile(info, stream)
            raw.flush()
            os.fsync(raw.fileno())
        if destination is not None:
            observed, _ = _sha256_file(output)
            if final_destination.exists():
                if final_destination.is_symlink() or not final_destination.is_file():
                    raise FleetBundleError("existing source bundle destination is unsafe")
                existing, _ = _sha256_file(final_destination)
                if not hmac.compare_digest(existing, observed):
                    raise FleetBundleError("source bundle destination already contains different bytes")
                output.unlink()
                output = final_destination
            else:
                os.replace(output, final_destination)
                output = final_destination
        digest, size = _sha256_file(output)
        return FleetSourceBundle(path=output, sha256=digest, size_bytes=size, file_count=len(sources))
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def _https_origin(value: str) -> str:
    if not isinstance(value, str) or len(value) > 2048 or any(c.isspace() or ord(c) < 32 for c in value):
        raise FleetBundleError("controller URL is invalid")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise FleetBundleError("controller URL is invalid") from exc
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise FleetBundleError("controller URL must be a credential-free HTTPS origin")
    return f"https://{parsed.netloc.lower()}"


def _validate_ca_pem(value: bytes) -> str:
    if not isinstance(value, bytes) or not value or len(value) > 64 * 1024:
        raise FleetBundleError("public controller certificate is invalid")
    if b"-----BEGIN CERTIFICATE-----" not in value or _PRIVATE_KEY_MARKER.search(value):
        raise FleetBundleError("public controller certificate contains invalid or private material")
    try:
        pem = value.decode("ascii")
        context = ssl.create_default_context()
        context.load_verify_locations(cadata=pem)
    except (UnicodeError, ssl.SSLError, ValueError) as exc:
        raise FleetBundleError("public controller certificate could not be parsed") from exc
    return pem


def render_node_bootstrap_script(
    *,
    controller_url: str,
    source_bundle_url: str,
    source_bundle_sha256: str,
    server_ca_pem: bytes,
    system_path: str = "/usr/sbin:/usr/bin:/sbin:/bin",
    template_path: Path | None = None,
) -> bytes:
    """Render the same role-free first-boot script for production or fixtures."""
    origin = _https_origin(controller_url)
    if not isinstance(source_bundle_sha256, str) or not _SHA256_RE.fullmatch(source_bundle_sha256):
        raise FleetBundleError("source bundle SHA-256 is invalid")
    try:
        source = urlsplit(source_bundle_url)
        source_port = source.port
        origin_parts = urlsplit(origin)
        origin_port = origin_parts.port
    except ValueError as exc:
        raise FleetBundleError("source bundle URL is invalid") from exc
    expected_path = f"/dl/upload/fleet-source-{source_bundle_sha256}.tar.gz"
    if (source.scheme != "https" or source.hostname != origin_parts.hostname
            or source_port != origin_port or source.username or source.password
            or source.path != expected_path or source.query or source.fragment):
        raise FleetBundleError("source bundle URL must be the same-origin SHA-addressed upload route")
    pem = _validate_ca_pem(server_ca_pem)
    if (not isinstance(system_path, str) or not system_path or len(system_path) > 4096
            or any(c in system_path for c in "\r\n\x00")):
        raise FleetBundleError("system executable path is invalid")
    path_entries = system_path.split(":")
    if any(not entry or not Path(entry).is_absolute() or ".." in Path(entry).parts for entry in path_entries):
        raise FleetBundleError("system executable path entries must be absolute and normalized")
    path = Path(template_path) if template_path is not None else Path(__file__).resolve().parents[3] / "deploy/fleet/bootstrap/node-bootstrap.sh"
    try:
        template = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise FleetBundleError("Fleet bootstrap script template is unavailable") from exc
    values = {
        "__FLEET_CONTROLLER_URL__": (repr(origin), 2),
        "__FLEET_SOURCE_BUNDLE_URL__": (repr(source_bundle_url), 2),
        "__FLEET_SOURCE_BUNDLE_SHA256__": (repr(source_bundle_sha256), 2),
        "__FLEET_SERVER_CA_PEM_B64__": (repr(base64.b64encode(pem.encode("ascii")).decode("ascii")), 2),
        "__FLEET_UV_ARTIFACT_URL__": (repr(f"{origin}/dl/upload/{UV_ARCHIVE_FILENAME}"), 2),
        "__FLEET_UV_ARTIFACT_SHA256__": (repr(UV_ARCHIVE_SHA256), 2),
        "__FLEET_UV_ARTIFACT_FILENAME__": (repr(UV_ARCHIVE_FILENAME), 2),
        "__FLEET_SYSTEM_PATH_SHELL__": (shlex.quote(system_path), 1),
    }
    rendered = template
    for token, (value, expected_count) in values.items():
        if rendered.count(token) != expected_count:
            raise FleetBundleError("Fleet bootstrap template has an invalid render token")
        rendered = rendered.replace(token, value)
    if any(token in rendered for token in values):
        raise FleetBundleError("Fleet bootstrap template contains an unresolved value")
    if "--no-check-certificate" in rendered or "Authorization" in rendered or "role" in rendered.lower():
        raise FleetBundleError("public Fleet bootstrap script contains a forbidden auth or role setting")
    return rendered.encode("utf-8")


def render_initial_fetch_command(script_url: str, script_sha256: str) -> str:
    """Render the externally pinned two-line plaintext initial script fetch."""
    if not isinstance(script_sha256, str) or not _SHA256_RE.fullmatch(script_sha256):
        raise FleetBundleError("bootstrap script SHA-256 is invalid")
    try:
        parsed = urlsplit(script_url)
        parsed.port
    except ValueError as exc:
        raise FleetBundleError("bootstrap script URL is invalid") from exc
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path != "/dl/upload/node-bootstrap.sh" or parsed.query or parsed.fragment):
        raise FleetBundleError("initial bootstrap script URL must be the fixed public HTTP download route")
    quoted_url = shlex.quote(script_url)
    first = (
        '(_QWEN_FLEET_BOOTSTRAP_TMP="$(mktemp /tmp/qwen-fleet-node-bootstrap.XXXXXX)" '
        f'&& wget --quiet --max-redirect=0 --output-document="$_QWEN_FLEET_BOOTSTRAP_TMP" {quoted_url} && \\'
    )
    second = (
        f'printf \'%s  %s\\n\' {shlex.quote(script_sha256)} "$_QWEN_FLEET_BOOTSTRAP_TMP" '
        '| sha256sum --check - && bash "$_QWEN_FLEET_BOOTSTRAP_TMP"; '
        '_QWEN_FLEET_STATUS=$?; rm -f "${_QWEN_FLEET_BOOTSTRAP_TMP:-}"; '
        'exit "$_QWEN_FLEET_STATUS" )'
    )
    return first + "\n" + second


def _atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise FleetBundleError("public release output path is unsafe")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            if target.read_bytes() != data:
                raise FleetBundleError("public release output already exists with different content")
            temporary.unlink()
        else:
            os.replace(temporary, target)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def build_public_release(
    *,
    project_root: Path,
    output_dir: Path,
    controller_url: str,
    server_ca_pem_path: Path,
) -> FleetBootstrapRelease:
    """Create public script, public source archive, and an integrity receipt.

    No network upload or external webroot write is performed.
    """
    origin = _https_origin(controller_url)
    output = Path(output_dir).expanduser().absolute()
    if output.is_symlink():
        raise FleetBundleError("release output directory cannot be a symlink")
    repo = Path(project_root).expanduser().absolute().resolve(strict=True)
    if output == repo or repo in output.parents:
        raise FleetBundleError("public bootstrap release artifacts must stay outside the project tree")
    created_output = not output.exists()
    output.mkdir(parents=True, exist_ok=True, mode=0o755)
    if output.is_symlink() or not output.is_dir():
        raise FleetBundleError("release output path must be a real directory")
    if created_output:
        os.chmod(output, 0o755)
    ca_path = Path(server_ca_pem_path)
    if ca_path.is_symlink() or not ca_path.is_file():
        raise FleetBundleError("server public certificate path is unsafe")
    ca_pem = ca_path.read_bytes()
    _validate_ca_pem(ca_pem)

    with tempfile.TemporaryDirectory(prefix="qwen-fleet-release-", dir="/tmp") as temp_name:
        temp_bundle = Path(temp_name) / "source.tar.gz"
        built = build_source_bundle(repo, temp_bundle)
        bundle_name = f"fleet-source-{built.sha256}.tar.gz"
        source_path = output / bundle_name
        if source_path.exists():
            existing_hash, _ = _sha256_file(source_path)
            if not hmac.compare_digest(existing_hash, built.sha256):
                raise FleetBundleError("existing public source archive does not match its content address")
        else:
            _atomic_write(source_path, temp_bundle.read_bytes(), 0o644)

    source_url = f"{origin}/dl/upload/{bundle_name}"
    script_bytes = render_node_bootstrap_script(
        controller_url=origin,
        source_bundle_url=source_url,
        source_bundle_sha256=built.sha256,
        server_ca_pem=ca_pem,
        template_path=repo / "deploy/fleet/bootstrap/node-bootstrap.sh",
    )
    script_sha, script_size = hashlib.sha256(script_bytes).hexdigest(), len(script_bytes)
    script_path = output / "node-bootstrap.sh"
    _atomic_write(script_path, script_bytes, 0o644)
    script_url = f"http://{urlsplit(origin).netloc}/dl/upload/node-bootstrap.sh"
    start_command = render_initial_fetch_command(script_url, script_sha)
    ca_der = ssl.PEM_cert_to_DER_cert(_validate_ca_pem(ca_pem))
    uv_artifact_url = f"{origin}/dl/upload/{UV_ARCHIVE_FILENAME}"
    receipt = {
        "schema": "qwen-fleet-public-bootstrap/v1",
        "controller_origin": origin,
        "initial_script_url": script_url,
        "source_bundle_url": source_url,
        "script": {"filename": script_path.name, "sha256": script_sha, "size_bytes": script_size},
        "source_bundle": {
            "filename": source_path.name,
            "sha256": built.sha256,
            "size_bytes": built.size_bytes,
            "file_count": built.file_count,
        },
        "runtime_artifacts": {
            "uv": {
                "filename": UV_ARCHIVE_FILENAME,
                "version": UV_VERSION,
                "sha256": UV_ARCHIVE_SHA256,
                "upstream_url": UV_UPSTREAM_URL,
                "url": uv_artifact_url,
                "controller_path": f"/dl/upload/{UV_ARCHIVE_FILENAME}",
                "reviewed_archive_path": str(output / UV_ARCHIVE_FILENAME),
            },
        },
        "server_ca_sha256": hashlib.sha256(ca_der).hexdigest(),
        "start_command": start_command,
    }
    receipt_path = output / "release.json"
    _atomic_write(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(), 0o644)
    return FleetBootstrapRelease(
        output_dir=output,
        script_path=script_path,
        source_bundle_path=source_path,
        receipt_path=receipt_path,
        script_sha256=script_sha,
        source_bundle_sha256=built.sha256,
        start_command=start_command,
        uv_artifact_path=output / UV_ARCHIVE_FILENAME,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a public, SHA-pinned Qwen Fleet bootstrap release")
    commands = parser.add_subparsers(dest="command", required=True)
    release = commands.add_parser("build-release")
    release.add_argument("--project-root", type=Path, default=Path.cwd())
    release.add_argument("--output-dir", type=Path, required=True)
    release.add_argument("--controller-url", required=True)
    release.add_argument("--server-ca-pem", type=Path, required=True)
    release.add_argument("--command-only", action="store_true",
                         help="print only the final two-line first-boot command")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        release = build_public_release(
            project_root=args.project_root,
            output_dir=args.output_dir,
            controller_url=args.controller_url,
            server_ca_pem_path=args.server_ca_pem,
        )
        if args.command_only:
            print(release.start_command)
        else:
            print(json.dumps(release.as_dict(), sort_keys=True))
        return 0
    except FleetBundleError as exc:
        print(f"Qwen Fleet release build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
