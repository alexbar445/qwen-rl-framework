"""Resumable migration transport using installed rsync and verified OpenSSH."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Callable

from .instance_migration import InstanceMigrationError, _read_manifest, _verify_bundle_files

_TARGET = re.compile(r"(?P<host>(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*):(?P<path>/[^\r\n\x00]*)\Z")


def transfer_migration(bundle: Path, target: str, *, ssh_port: int | None = None,
                       identity_file: Path | None = None,
                       runner: Callable = subprocess.run) -> dict:
    """Transfer static or final bundle files; importing remains a separate verified step.

    An interrupted copy is resumable by repeating the command. It never rents a
    server, installs SSH software, accepts a new host key, or restarts a runtime.
    """
    if ssh_port is not None and (type(ssh_port) is not int or not 1 <= ssh_port <= 65535):
        raise InstanceMigrationError("ssh_port must be an integer in 1..65535")
    match = _TARGET.fullmatch(target) if isinstance(target, str) else None
    if match is None or match['host'].startswith('-'):
        raise InstanceMigrationError("target must be user@known-host:/absolute/path")
    destination = match['path']
    if any(ord(char) < 32 for char in destination) or '..' in PurePosixPath(destination).parts:
        raise InstanceMigrationError("target path must not contain parent traversal or control characters")
    source = Path(bundle).expanduser().absolute()
    if source.is_symlink() or not source.is_dir() or not (source / "manifest.json").is_file():
        raise InstanceMigrationError("bundle must be a prepared migration directory containing manifest.json")
    manifest = _read_manifest(source)
    _verify_bundle_files(source, manifest)
    rsync, ssh = shutil.which('rsync'), shutil.which('ssh')
    if rsync is None or ssh is None:
        raise InstanceMigrationError("resumable migration transfer requires installed rsync and OpenSSH")
    transport = [ssh, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                 '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
    if ssh_port is not None:
        transport += ['-p', str(ssh_port)]
    if identity_file is not None:
        identity = Path(identity_file).expanduser().absolute()
        if identity.is_symlink() or not identity.is_file() or identity.stat().st_mode & 0o077:
            raise InstanceMigrationError("SSH identity file must be a private regular file")
        transport += ['-i', str(identity)]
    with tempfile.NamedTemporaryFile(prefix='qwen-rl-migration-files-', mode='wb') as file_list:
        names = ['manifest.json', *(entry['path'] for entry in manifest['files'])]
        file_list.write(b''.join(name.encode('utf-8') + b'\0' for name in sorted(set(names))))
        file_list.flush()
        command = [rsync, '-a', '--compress', '--partial', '--append-verify', '--protect-args', '--info=progress2',
                   '--from0', '--files-from', file_list.name,
                   '-e', shlex.join(transport), '--', str(source) + '/', target.rstrip('/') + '/']
        completed = runner(command, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr, check=False)
    return {'schema': 'qwen-rl-instance-migration-transfer/v1',
            'status': 'transferred' if completed.returncode == 0 else 'incomplete',
            'returncode': completed.returncode, 'resumable': True,
            'next_action': 'import_verified_bundle' if completed.returncode == 0 else 'repeat_transfer_command',
            'ssh_host_key_checking': 'strict', 'source_stopped_by_transfer': False}
