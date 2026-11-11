"""Mandatory, offline runtime checks shared by full and incremental images."""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import subprocess
import tempfile

REQUIRED_ARCHITECTURES = {'8.6', '8.9'}
LIGER_COMMIT = 'ff8ce5857420cee06865f40ffa8a60309bf2b458'
_POINTER_FIELDS = {'schema', 'generation', 'snapshot_id', 'snapshot', 'size_bytes',
                   'policy_version', 'optimizer_steps'}
_SNAPSHOT_NAME = re.compile(r'^state-(\d{8})-([0-9a-f]{32})\.pt$')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_MAX_SEED_JSON_BYTES = 1024 * 1024
_LIGER_RECEIPT = 'qwen-rl-provenance.json'
_LIGER_RECEIPT_SCHEMA = 'qwen-rl-installed-source/v1'
_MAX_LIGER_PY_BYTES = 2 * 1024 * 1024
_MAX_LIGER_TOTAL_BYTES = 16 * 1024 * 1024


def validate_kernel_build(info, torch_version, cuda_version):
    if info.get('torch_version') != torch_version or info.get('torch_cuda_version') != cuda_version:
        raise RuntimeError('T2 kernel wheel must match the installed Torch/CUDA')
    missing = REQUIRED_ARCHITECTURES - set(info.get('architectures', []))
    if missing:
        raise RuntimeError('instance wheel lacks required CUDA targets: ' + ', '.join(sorted(missing)))
    if info.get('api_version') != 1:
        raise RuntimeError('unsupported T2 kernel API')


def _distribution_info_dir(distribution) -> Path:
    path = Path(distribution._path)
    info = path.lstat()
    if not path.name.endswith('.dist-info') or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError('Liger distribution metadata directory is invalid')
    return path


def _installed_liger_python_files(package_root: str | Path) -> dict[str, str]:
    root = Path(package_root)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError('installed Liger package root is not a real directory')
    result = {}
    total = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in dirs:
            child = parent / name
            if child.is_symlink():
                raise RuntimeError('installed Liger package contains a directory symlink')
        for name in names:
            path = parent / name
            if path.suffix != '.py':
                continue
            relative = path.relative_to(root.parent).as_posix()
            posix = PurePosixPath(relative)
            if (posix.is_absolute() or '..' in posix.parts or not relative.startswith('liger_kernel/')):
                raise RuntimeError('installed Liger Python path is invalid')
            payload = _read_regular_bytes(path, limit=_MAX_LIGER_PY_BYTES)
            total += len(payload)
            if total > _MAX_LIGER_TOTAL_BYTES:
                raise RuntimeError('installed Liger Python files exceed the receipt limit')
            result[relative] = hashlib.sha256(payload).hexdigest()
    if 'liger_kernel/ops/rms_norm.py' not in result:
        raise RuntimeError('installed Liger package does not contain the pinned RMSNorm source')
    return dict(sorted(result.items()))


def write_installed_source_receipt(distribution, *, source_commit: str, files: dict[str, str]) -> Path:
    """Write verified offline-wheel provenance into this distribution's dist-info."""
    if distribution.version != '0.8.4' or source_commit != LIGER_COMMIT:
        raise RuntimeError('Liger receipt must name the accepted version and source commit')
    package_root = Path(distribution.locate_file('liger_kernel'))
    expected = _installed_liger_python_files(package_root)
    if files != expected:
        raise RuntimeError('Liger receipt file hashes differ from the installed package')
    receipt = {
        'schema': _LIGER_RECEIPT_SCHEMA, 'distribution': 'liger-kernel',
        'version': '0.8.4', 'source_commit': source_commit, 'files': expected,
    }
    path = _distribution_info_dir(distribution) / _LIGER_RECEIPT
    try:
        prior = path.lstat()
    except FileNotFoundError:
        prior = None
    if prior is not None and (not stat.S_ISREG(prior.st_mode) or stat.S_ISLNK(prior.st_mode)):
        raise RuntimeError('existing Liger provenance receipt is not a regular file')
    descriptor, temporary = tempfile.mkstemp(prefix='.qwen-rl-provenance-', dir=path.parent)
    try:
        os.fchmod(descriptor, 0o644)
        payload = (json.dumps(receipt, sort_keys=True, separators=(',', ':')) + '\n').encode()
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def _verify_liger_provenance(distribution, package_root: Path, rms_module_path: Path) -> str:
    if distribution.version != '0.8.4':
        raise RuntimeError('instance requires pinned Liger 0.8.4')
    root = Path(package_root)
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise RuntimeError('loaded Liger package root is not a real directory')
    root = root.resolve(strict=True)
    module = Path(rms_module_path)
    module_info = module.lstat()
    if not stat.S_ISREG(module_info.st_mode) or stat.S_ISLNK(module_info.st_mode):
        raise RuntimeError('loaded Liger RMSNorm module is not a regular file')
    module = module.resolve(strict=True)
    if not module.is_relative_to(root):
        raise RuntimeError('loaded Liger RMSNorm module is outside the installed package')

    direct_text = distribution.read_text('direct_url.json')
    if direct_text:
        try:
            direct = json.loads(direct_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError('Liger direct_url metadata is invalid') from exc
        vcs = direct.get('vcs_info') if isinstance(direct, dict) else None
        if isinstance(vcs, dict):
            if vcs.get('commit_id') != LIGER_COMMIT:
                raise RuntimeError('Liger source revision differs from the accepted frozen RMSNorm implementation')
            return 'git-direct-url'

    receipt_path = _distribution_info_dir(distribution) / _LIGER_RECEIPT
    receipt = _read_regular_json(receipt_path, limit=128 * 1024)
    required = {'schema', 'distribution', 'version', 'source_commit', 'files'}
    if (set(receipt) != required or receipt.get('schema') != _LIGER_RECEIPT_SCHEMA
            or receipt.get('distribution') != 'liger-kernel' or receipt.get('version') != '0.8.4'
            or receipt.get('source_commit') != LIGER_COMMIT or not isinstance(receipt.get('files'), dict)):
        raise RuntimeError('offline Liger source receipt is invalid')
    installed = _installed_liger_python_files(root)
    if receipt['files'] != installed:
        raise RuntimeError('offline Liger receipt hashes differ from the installed Python files')
    relative_module = module.relative_to(root.parent).as_posix()
    if relative_module not in installed:
        raise RuntimeError('loaded Liger RMSNorm source is absent from its provenance receipt')
    return 'offline-wheel-receipt'


def ensure_liger_build_dependency():
    """Build-time installer; never called by the runtime entrypoint."""
    import sys
    try:
        distribution = metadata.distribution('liger-kernel')
        direct = json.loads(distribution.read_text('direct_url.json') or '{}')
        current = direct.get('vcs_info', {}).get('commit_id')
    except metadata.PackageNotFoundError:
        distribution = None
        current = None
    if current == LIGER_COMMIT:
        return
    if distribution is not None:
        try:
            package = importlib.import_module('liger_kernel')
            rms = importlib.import_module('liger_kernel.ops.rms_norm')
            if _verify_liger_provenance(
                    distribution, Path(package.__file__).parent, Path(rms.__file__)) == 'offline-wheel-receipt':
                return
        except Exception:
            pass
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-deps',
                    'liger-kernel @ git+https://github.com/linkedin/Liger-Kernel.git@' + LIGER_COMMIT], check=True)


def _read_regular_bytes(path: Path, *, limit: int = _MAX_SEED_JSON_BYTES) -> bytes:
    """Read one bounded regular file without following symlinks."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_size > limit:
        raise RuntimeError('seed input is not a bounded regular file: ' + path.name)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_dev != info.st_dev
                or opened.st_ino != info.st_ino or opened.st_size > limit):
            raise RuntimeError('seed metadata changed during inspection: ' + path.name)
        chunks = bytearray()
        while len(chunks) <= limit:
            chunk = os.read(fd, min(64 * 1024, limit + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) > limit:
            raise RuntimeError('seed metadata exceeds the status size limit: ' + path.name)
    finally:
        os.close(fd)
    return bytes(chunks)


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError('seed metadata contains duplicate JSON keys')
        value[key] = item
    return value


def _read_regular_json(path: Path, *, limit: int = _MAX_SEED_JSON_BYTES):
    """Read bounded JSON metadata without following symlinks."""
    value = json.loads(_read_regular_bytes(path, limit=limit).decode('utf-8'),
                       object_pairs_hook=_unique_json_object)
    if not isinstance(value, dict):
        raise RuntimeError('seed metadata must be a JSON object: ' + path.name)
    return value


def _verified_seed_snapshot(seed: Path, manifest: dict, pointer: dict) -> tuple[Path, int]:
    if set(pointer) != _POINTER_FIELDS or pointer.get('schema') != 'central-gradient-checkpoint-pointer/v1':
        raise RuntimeError('seed checkpoint pointer schema is invalid')
    generation, snapshot_id = pointer.get('generation'), pointer.get('snapshot_id')
    size = pointer.get('size_bytes')
    if (type(generation) is not int or generation < 0
            or not isinstance(snapshot_id, str) or re.fullmatch(r'[0-9a-f]{32}', snapshot_id) is None
            or type(size) is not int or size <= 0
            or type(pointer.get('policy_version')) is not int or pointer['policy_version'] < 0
            or type(pointer.get('optimizer_steps')) is not int or pointer['optimizer_steps'] < 0):
        raise RuntimeError('seed checkpoint pointer counters are invalid')
    name = pointer.get('snapshot')
    match = _SNAPSHOT_NAME.fullmatch(name) if isinstance(name, str) else None
    if (match is None or int(match.group(1)) != generation or match.group(2) != snapshot_id
            or '/' in name or '\\' in name):
        raise RuntimeError('seed checkpoint snapshot name is invalid')
    expected_sha = manifest.get('checkpoint_sha256')
    if not isinstance(expected_sha, str) or _SHA256.fullmatch(expected_sha) is None:
        raise RuntimeError('seed checkpoint SHA256 receipt is invalid')

    seed = Path(seed)
    seed_info = seed.lstat()
    if not stat.S_ISDIR(seed_info.st_mode) or stat.S_ISLNK(seed_info.st_mode):
        raise RuntimeError('seed root must be a real directory')
    seed = seed.resolve(strict=True)
    for directory in (seed / 'outputs', seed / 'outputs/bonsai2-async-bepa-20261019',
                      seed / 'outputs/bonsai2-async-bepa-20261019/checkpoints'):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError('seed checkpoint directory is not a real directory')
    snapshot = seed / 'outputs/bonsai2-async-bepa-20261019/checkpoints' / name
    try:
        info = snapshot.lstat()
    except OSError as exc:
        raise RuntimeError('seed checkpoint snapshot is unavailable') from exc
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_size != size):
        raise RuntimeError('seed checkpoint snapshot file identity is invalid')
    fd = os.open(snapshot, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_dev != info.st_dev
                or opened.st_ino != info.st_ino or opened.st_size != size):
            raise RuntimeError('seed checkpoint snapshot changed during inspection')
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 8 * 1024**2)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        current = snapshot.lstat()
        if ((after.st_dev, after.st_ino, after.st_size) != (opened.st_dev, opened.st_ino, size)
                or (current.st_dev, current.st_ino, current.st_size) != (opened.st_dev, opened.st_ino, size)
                or digest.hexdigest() != expected_sha):
            raise RuntimeError('seed checkpoint SHA256 or file identity differs from its receipt')
        return snapshot, fd
    except BaseException:
        os.close(fd)
        raise


def read_checkpoint_metadata(checkpoint_dir: str | Path) -> dict:
    """Return a pointer identity after a no-follow metadata-only snapshot check."""
    root = Path(checkpoint_dir)
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise RuntimeError('checkpoint directory must be a real directory')
    pointer = _read_regular_json(root / 'latest.json', limit=64 * 1024)
    if set(pointer) != _POINTER_FIELDS or pointer.get('schema') != 'central-gradient-checkpoint-pointer/v1':
        raise RuntimeError('checkpoint pointer schema is invalid')
    generation, snapshot_id = pointer.get('generation'), pointer.get('snapshot_id')
    name, size = pointer.get('snapshot'), pointer.get('size_bytes')
    match = _SNAPSHOT_NAME.fullmatch(name) if isinstance(name, str) else None
    if (type(generation) is not int or generation < 0
            or not isinstance(snapshot_id, str) or re.fullmatch(r'[0-9a-f]{32}', snapshot_id) is None
            or match is None or int(match.group(1)) != generation or match.group(2) != snapshot_id
            or type(size) is not int or size <= 0
            or type(pointer.get('policy_version')) is not int or pointer['policy_version'] < 0
            or type(pointer.get('optimizer_steps')) is not int or pointer['optimizer_steps'] < 0):
        raise RuntimeError('checkpoint pointer identity is invalid')
    snapshot = root / name
    info = snapshot.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_size != size):
        raise RuntimeError('checkpoint snapshot file identity is invalid')
    fd = os.open(snapshot, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_dev != info.st_dev
                or opened.st_ino != info.st_ino or opened.st_size != size):
            raise RuntimeError('checkpoint snapshot changed during inspection')
    finally:
        os.close(fd)
    return {key: pointer[key] for key in sorted(_POINTER_FIELDS)}


def verify_seed(seed=Path('/opt/qwen-rl-seed'), *, staged_inputs=False):
    seed = Path(seed)
    try:
        seed_info = seed.lstat()
    except FileNotFoundError:
        return {'seed_check': 'not_bundled'}
    if not stat.S_ISDIR(seed_info.st_mode) or stat.S_ISLNK(seed_info.st_mode):
        raise RuntimeError('seed root must be a real directory')
    import torch
    import yaml
    from qwen_rl.learner.recovery import _semantic_identity
    manifest = _read_regular_json(seed / 'seed-manifest.json')
    cp = seed / 'outputs/bonsai2-async-bepa-20261019/checkpoints'
    for directory in (seed / 'outputs', seed / 'outputs/bonsai2-async-bepa-20261019', cp):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError('seed checkpoint directory is not a real directory')
    pointer = _read_regular_json(cp / 'latest.json')
    if pointer != manifest['checkpoint']:
        raise RuntimeError('seed pointer differs from manifest')
    snapshot, snapshot_fd = _verified_seed_snapshot(seed, manifest, pointer)
    try:
        runtime_path = seed / 'runtime.yaml'
        identity_path = seed / 'outputs/ternary-lora-pp/20261014/bonsai2-t2-identity.json'
        for directory in (seed / 'outputs/ternary-lora-pp',
                          seed / 'outputs/ternary-lora-pp/20261014'):
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise RuntimeError('seed identity directory is not a real directory')
        cfg = yaml.safe_load(_read_regular_bytes(runtime_path).decode('utf-8'))
        if not isinstance(cfg, dict):
            raise RuntimeError('seed runtime configuration is invalid')
        cfg['artifact_identity'] = _read_regular_json(identity_path)
        if staged_inputs:
            cfg['dataset_dir'] = str(seed / 'data/generated')
            contract = Path(cfg['arc']['contract_path'])
            cfg['arc']['contract_path'] = str(seed / 'configs' / contract.name)
        current = _semantic_identity(cfg)
        state = torch.load(f'/proc/self/fd/{snapshot_fd}', map_location='cpu', mmap=True, weights_only=False)
        expected = state['last_update_context']['source_identity']
        differences = [key for key, value in current.items() if expected.get(key) != value]
        if differences:
            raise RuntimeError('bundled checkpoint is incompatible with image semantics: ' + ', '.join(differences))
        if state.get('pending_attempt') is not None or state.get('prepared_windows'):
            raise RuntimeError('bundled checkpoint contains an unfinished optimizer transaction')
        for key in ['policy_version', 'optimizer_steps', 'scheduler_steps']:
            if state[key] != pointer['policy_version']:
                raise RuntimeError('bundled checkpoint counters disagree: ' + key)
        return {'seed_check': 'full checkpoint SHA and runtime semantics passed',
                'seed_policy_version': pointer['policy_version'],
                'seed_snapshot_id': pointer['snapshot_id']}
    finally:
        os.close(snapshot_fd)


def verify_cpu(*, native_binary='/usr/local/bin/ninfer-serve',
               cuobjdump='/usr/local/cuda/bin/cuobjdump'):
    import torch
    import qwen_rl_ternary_kernels as kernels
    info = kernels.build_info()
    validate_kernel_build(info, torch.__version__, torch.version.cuda)
    extension = importlib.import_module('qwen_rl_ternary_kernels._C')
    if cuobjdump is not None:
        cubins = subprocess.check_output([str(cuobjdump), '--list-elf', extension.__file__], text=True)
        for target in ['sm_86', 'sm_89']:
            if target not in cubins:
                raise RuntimeError('installed T2 binary lacks required cubin: ' + target)
    liger_distribution = metadata.distribution('liger-kernel')
    rms_module = importlib.import_module('liger_kernel.ops.rms_norm')
    liger_provenance = _verify_liger_provenance(
        liger_distribution, Path(importlib.import_module('liger_kernel').__file__).parent,
        Path(rms_module.__file__),
    )
    rms = rms_module.rms_norm_backward
    if 'needs_dw' not in inspect.signature(rms).parameters:
        raise RuntimeError('Liger RMSNorm backward lacks needs_dw')
    torchao = importlib.import_module('torchao')
    torchao_libraries = sorted(Path(torchao.__file__).resolve().parent.glob('_C*.so'))
    if not torchao_libraries:
        raise RuntimeError('TorchAO native extension files are missing')
    for library in torchao_libraries:
        if not library.is_file():
            raise RuntimeError('TorchAO native extension is missing: ' + str(library))
        try:
            torch.ops.load_library(str(library))
        except (OSError, RuntimeError) as exc:
            raise RuntimeError('TorchAO native extension failed to load: ' + str(library)) from exc
    for name in ['fast_hadamard_transform_cuda', 'fla.ops', 'trl', 'peft',
                 'qwen_rl.learner.trainer', 'qwen_rl.worker.ninfer_backend']:
        importlib.import_module(name)
    binary = Path(native_binary)
    if not binary.is_file():
        raise RuntimeError('NInfer runtime is missing')
    links = subprocess.run(['ldd', str(binary)], capture_output=True, text=True, check=True).stdout
    missing = [line for line in links.splitlines() if 'not found' in line and 'libcuda.so.1' not in line]
    if missing:
        raise RuntimeError('NInfer shared libraries missing: ' + '; '.join(missing))
    return {'architectures': info['architectures'], 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'liger_commit': LIGER_COMMIT,
            'liger_provenance': liger_provenance,
            'torchao_native_libraries': [str(path) for path in torchao_libraries],
            'cuobjdump_check': 'skipped' if cuobjdump is None else 'passed',
            'cpu_checks': 'passed', 'gpu_checks': 'not_requested'}


def verify_gpu():
    import torch
    from fast_hadamard_transform import hadamard_transform
    from qwen_rl.models.ternary_fused_hadamard import fused_hadamard_rows, kernel_build_info
    info = kernel_build_info(check_device=True)
    x = torch.randn((2, 1024), device='cuda', dtype=torch.bfloat16, requires_grad=True)
    y = hadamard_transform(x, scale=1024 ** -0.5)
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    z = fused_hadamard_rows(x.detach(), torch.ones(1024, device='cuda', dtype=torch.int8),
                           inverse=False, block_size=1024, output_dtype=torch.bfloat16)
    assert z.shape == x.shape and torch.isfinite(z).all()
    torch.cuda.synchronize()
    return {'capability': list(torch.cuda.get_device_capability()),
            'gpu_checks': 'Hadamard forward/backward and T2 fused forward passed',
            'architectures': info['architectures']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--native-binary', default='/usr/local/bin/ninfer-serve')
    cuobjdump_group = parser.add_mutually_exclusive_group()
    cuobjdump_group.add_argument('--cuobjdump', default='/usr/local/cuda/bin/cuobjdump')
    cuobjdump_group.add_argument('--no-cuobjdump', dest='cuobjdump', action='store_const', const=None)
    args = parser.parse_args()
    result = verify_cpu(native_binary=args.native_binary, cuobjdump=args.cuobjdump)
    result.update(verify_seed())
    if args.gpu:
        import torch
        count = torch.cuda.device_count()
        if not count: raise RuntimeError('GPU preflight requested but no CUDA GPU is visible')
        result['gpus'] = []
        for ordinal in range(count):
            with torch.cuda.device(ordinal):
                result['gpus'].append({'cuda_ordinal': ordinal,
                    'uuid': str(torch.cuda.get_device_properties(ordinal).uuid), **verify_gpu()})
        result['gpu_checks'] = 'all selected GPUs passed'
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
