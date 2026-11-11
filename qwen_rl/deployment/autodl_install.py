"""Install a prepared AutoDL runtime without Docker or on-host extension builds.

This bootstrap imports only the standard library until its private environment
has been installed. Model/data/checkpoint paths come from the user's runtime YAML.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

SCHEMA = "qwen-rl-autodl-bundle/v1"
PROFILES = {"cu129": ("12.9", (575, 51, 3)), "cu130": ("13.0", (580, 65, 6))}
GENERATED_TASK_PACKAGE = (
    "source/vendor/arc-taskgen-v2/pyproject.toml",
    "source/vendor/arc-taskgen-v2/arc_taskgen/__init__.py",
    "source/vendor/arc-taskgen-v2/arc_taskgen/dataset.py",
    "source/vendor/arc-taskgen-v2/arc_taskgen/verifier.py",
)
CRITICAL_INSTALLED_FILES = (
    "source/pyproject.toml",
    "source/toolkit-dependency.json",
    "source/assets/arc-system-rules.txt",
    "source/qwen_rl/paths.py",
    "source/qwen_rl/task_source.py",
    "source/qwen_rl/task_catalog.py",
    "source/qwen_rl/prompt.py",
    "source/qwen_rl/learner/recovery.py",
    "source/vendor/paired-factory-v1/arc_toolkit/__init__.py",
    "source/vendor/paired-factory-v1/arc_toolkit/runtime.py",
    "source/vendor/paired-factory-v1/arc_toolkit/dependencies.py",
    "source/qwen_rl/deployment/autodl_install.py",
    "source/qwen_rl/deployment/autodl_config.py",
    "source/qwen_rl/deployment/instance_setup.py",
    "source/qwen_rl/deployment/instance_compat.py",
    "source/qwen_rl/deployment/instance.py",
    "source/qwen_rl/deployment/instance_cli.py",
    "source/qwen_rl/deployment/instance_operations.py",
    "source/qwen_rl/deployment/instance_preflight.py",
    "source/qwen_rl/deployment/teacher_readiness.py",
    "source/qwen_rl/deployment/instance_storage.py",
    "source/qwen_rl/deployment/instance_startup.py",
    "source/qwen_rl/deployment/instance_failure.py",
    "source/qwen_rl/deployment/instance_artifact.py",
    "source/qwen_rl/deployment/instance_relocate.py",
    "source/qwen_rl/deployment/instance_source_package.py",
    "source/qwen_rl/deployment/instance_file_references.py",
    "source/qwen_rl/deployment/instance_stop_now.py",
    "source/qwen_rl/deployment/instance_stop_control.py",
    "source/qwen_rl/learner/stop_control.py",
    "source/qwen_rl/learner/app.py",
    "source/qwen_rl/learner/app_lifecycle.py",
    "source/qwen_rl/learner/run/coordinator.py",
    "source/qwen_rl/learner/run/source_sampling.py",
    "source/qwen_rl/evaluation/official_arc2.py",
    "source/qwen_rl/learner/run/checkpoint_overlap.py",
    "source/qwen_rl/learner/run/update.py",
    "source/qwen_rl/learner/run/state.py",
    "source/qwen_rl/learner/gradient_server/core.py",
    "source/qwen_rl/learner/gradient_server/engine.py",
    "source/qwen_rl/learner/gradient_server/store.py",
    "source/qwen_rl/rollout/pipeline.py",
    "source/qwen_rl/evaluation/relocation.py",
    "source/qwen_rl/deployment/instance_recovery.py",
    "source/qwen_rl/deployment/instance_relocation.py",
    "source/qwen_rl/deployment/instance_migration.py",
    "source/qwen_rl/deployment/instance_transfer.py",
    "source/qwen_rl/task_source_generated.py",
    "source/qwen_rl/service.py",
    *GENERATED_TASK_PACKAGE,
    "native/bin/ninfer-serve",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, **kwargs):
    subprocess.run([str(value) for value in command], check=True, **kwargs)


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or "\\" in name or str(relative) != name):
        raise ValueError("invalid bundle path: " + name)
    path = root.joinpath(*relative.parts)
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError("bundle paths must not contain symlinks: " + name)
    return path


def load_bundle(root: Path) -> tuple[dict, str]:
    root = root.resolve(strict=True)
    manifest_path = root / "bundle.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 4 * 1024**2:
        raise ValueError("invalid bundle manifest")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("profile") not in PROFILES
            or manifest.get("python") != "3.12" or not isinstance(manifest.get("files"), dict)
            or not manifest["files"] or type(manifest.get("installed_size_bytes")) is not int
            or manifest["installed_size_bytes"] <= 0):
        raise ValueError("unsupported AutoDL bundle")
    required = {"requirements.lock.txt", *CRITICAL_INSTALLED_FILES}
    required_vendor = manifest.get("required_vendor_python_files")
    if not isinstance(required_vendor, list) or not required_vendor:
        raise ValueError("bundle lacks required file field: required_vendor_python_files")
    if (any(not isinstance(name, str) or not name.startswith("source/vendor/")
            or not name.endswith(".py") for name in required_vendor)
            or len(set(required_vendor)) != len(required_vendor)):
        raise ValueError("bundle required_vendor_python_files field is invalid")
    required.update(required_vendor)
    missing_fields = sorted(required - manifest["files"].keys())
    if missing_fields:
        raise ValueError("bundle lacks required file field: " + ", ".join(missing_fields))
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(manifest["files"]) | {"bundle.json"}:
        raise ValueError("bundle file inventory differs from its manifest")
    for name, expected in manifest["files"].items():
        path = safe_path(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("bundle file differs from its receipt: " + name)
    return manifest, sha256(manifest_path)


def check_python(python: Path):
    code = ("import sys,os,fcntl,ctypes,platform;assert sys.version_info[:2]==(3,12), 'Python 3.12 required';"
            "assert sys.platform=='linux' and platform.machine()=='x86_64', 'Linux x86_64 required';"
            "lib=ctypes.CDLL(None,use_errno=True);lib.memfd_create.argtypes=(ctypes.c_char_p,ctypes.c_uint);"
            "lib.memfd_create.restype=ctypes.c_int;f=lib.memfd_create(b'install-check',3);"
            "assert f>=0, 'Linux memfd_create unavailable';fcntl.fcntl(f,1033,15);os.close(f)")
    run([python, "-B", "-c", code])


def select_bundle(root: Path, gpu_indices: list[int]) -> Path:
    """A catalog is an explicit directory containing the two supported profiles."""
    if (root / "bundle.json").is_file():
        return root
    reasons = []
    for profile in ["cu130", "cu129"]:
        candidate = root / profile
        if not (candidate / "bundle.json").is_file():
            continue
        try:
            check_driver(profile, gpu_indices)
        except ValueError as error:
            reasons.append(str(error))
            continue
        return candidate
    raise ValueError("no compatible cu129/cu130 release in catalog: " + "; ".join(reasons))


def validate_gpu_indices(gpu_indices):
    if len(gpu_indices) not in {2, 4, 8} or len(set(gpu_indices)) != len(gpu_indices) or min(gpu_indices) < 0:
        raise ValueError("select 2, 4, or 8 unique GPU indices")


def check_driver(profile: str, gpu_indices: list[int], *, query=None):
    validate_gpu_indices(gpu_indices)
    if query is None:
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,driver_version,compute_cap", "--format=csv,noheader,nounits"],
            text=True, timeout=20,
        )
    inventory = {}
    for line in query.splitlines():
        index, uuid, driver, capability = [value.strip() for value in line.split(",")]
        inventory[int(index)] = (tuple(int(value) for value in driver.split(".")), capability, uuid)
    minimum = PROFILES[profile][1]
    for index in gpu_indices:
        if index not in inventory:
            raise ValueError(f"GPU {index} is not present")
        driver, capability, uuid = inventory[index]
        if capability not in {"8.6", "8.9"}:
            raise ValueError(f"GPU {index} is not a supported SM86/SM89 device")
        if (*driver, 0, 0)[:3] < minimum:
            raise ValueError(f"{profile} requires driver {'.'.join(map(str, minimum))} or newer; GPU {index} has {driver}")
    return inventory


def runtime_library_path(prefix: Path) -> str:
    # NInfer and Torch must use the same CUDA libraries installed by pip.
    nvidia = prefix / ".venv/lib/python3.12/site-packages/nvidia"
    directories = sorted(path for path in nvidia.glob("*/lib") if path.is_dir())
    return ":".join(str(path) for path in [*directories, prefix / "native/lib"])


def runtime_environment(prefix: Path) -> dict:
    env = dict(os.environ)
    source = prefix / "source"
    vendor = source / "vendor/arc-taskgen-v2"
    env.update(ARC_RL_ROOT=str(source), PYTHONPATH=os.pathsep.join((str(source), str(vendor))),
               PYTHONDONTWRITEBYTECODE="1")
    env.update(HF_HOME=str(prefix / "cache/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    lib = runtime_library_path(prefix)
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    for key, name in [("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "torch"),
                      ("CUDA_CACHE_PATH", "cuda")]:
        env[key] = str(prefix / "cache" / name)
    return env


def _locked_packages(bundle: Path) -> dict[str, str]:
    packages = {}
    for line in (bundle / "requirements.lock.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = line.split(maxsplit=1)[0]
        if "==" not in requirement:
            raise ValueError("requirements.lock.txt must contain exact package versions")
        name, version = requirement.split("==", 1)
        canonical = name.lower().replace("_", "-").replace(".", "-")
        if not name or not version or canonical in packages:
            raise ValueError("requirements.lock.txt has an invalid or duplicate package")
        packages[canonical] = version
    if not packages:
        raise ValueError("requirements.lock.txt has no pinned packages")
    return dict(sorted(packages.items()))


def _launcher_text(prefix: Path, env: dict[str, str]) -> str:
    import shlex

    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for key in ["ARC_RL_ROOT", "PYTHONPATH", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH",
                "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        lines.append(f"export {key}={shlex.quote(env[key])}")
    lines.append('export LD_LIBRARY_PATH=' + shlex.quote(runtime_library_path(prefix))
                 + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}')
    lines.append("exec " + shlex.quote(str(prefix / ".venv/bin/python")) + ' -B -m qwen_rl "$@"')
    return "\n".join(lines) + "\n"


def _installation_proof(manifest: dict, packages: dict[str, str], prefix: Path) -> dict:
    critical_files = set(CRITICAL_INSTALLED_FILES)
    critical_files.update(manifest["required_vendor_python_files"])
    return {
        "schema": "qwen-rl-autodl-install-proof/v1",
        "dependency_versions": packages,
        "interpreter": {
            "python": str(prefix / ".venv/bin/python"),
            "prefix": str(prefix / ".venv"),
        },
        "critical_files": {name: manifest["files"][name] for name in sorted(critical_files)},
    }


def _dependency_identity_check_code() -> str:
    return """
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

expected = json.loads(sys.argv[1])
prefix = Path(sys.argv[2]).resolve(strict=True)
assert Path(sys.prefix).resolve(strict=True) == prefix, "Python is outside the private .venv"
assert Path(sys.executable).absolute() == Path(sys.argv[3]).absolute(), "Python executable path changed"
missing = []
for name, version in expected.items():
    try:
        dist = metadata.distribution(name)
        dist_info = Path(dist._path)
        assert dist.version == version and dist_info.is_dir() and not dist_info.is_symlink()
    except (metadata.PackageNotFoundError, AssertionError):
        missing.append(name)
assert not missing, "installed dependency identity differs: " + ", ".join(missing)
"""


def _validate_installed_runtime(prefix: Path, proof: dict, *, command_runner=run) -> None:
    target_python = prefix / ".venv/bin/python"
    if (prefix.is_symlink() or (prefix / ".venv").is_symlink()
            or (prefix / ".venv/bin").is_symlink()):
        raise ValueError("installation proof failed: private runtime directories must not be symlinks; use a fresh prefix")
    if not target_python.is_file() or not os.access(target_python, os.X_OK):
        raise ValueError("installation proof failed: private .venv interpreter is missing; use a fresh prefix")
    for name, digest in proof["critical_files"].items():
        path = prefix / name
        symlink = False
        for part in (path, *path.parents):
            if part == prefix:
                break
            if part.is_symlink():
                symlink = True
                break
        if symlink or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"installation proof failed: installed file is missing or changed: {name}; use a fresh prefix")
    launcher = prefix / "qwen-rl"
    expected_launcher = _launcher_text(prefix, runtime_environment(prefix))
    if launcher.is_symlink() or not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError("installation proof failed: qwen-rl launcher is missing or not executable; use a fresh prefix")
    if launcher.read_text(encoding="utf-8") != expected_launcher:
        raise ValueError("installation proof failed: qwen-rl launcher points outside the private runtime; use a fresh prefix")

    code = _dependency_identity_check_code() + """
import torch
assert torch.__version__ == expected["torch"], "Torch build differs from the offline release"
sys.path.insert(0, sys.argv[4])
from arc_taskgen import dataset, verifier
from qwen_rl.deployment.instance_compat import verify_cpu
verify_cpu(native_binary=Path(sys.argv[5]), cuobjdump=None)
"""
    env = runtime_environment(prefix)
    env["CUDA_VISIBLE_DEVICES"] = ""
    command_runner([
        target_python, "-B", "-c", code, json.dumps(proof["dependency_versions"], sort_keys=True),
        prefix / ".venv", target_python, prefix / "source/vendor/arc-taskgen-v2",
        prefix / "native/bin/ninfer-serve",
    ], env=env)


def doctor(prefix: Path, *, gpu=False, gpu_indices=None):
    python = prefix / ".venv/bin/python"
    env = runtime_environment(prefix)
    env["CUDA_VISIBLE_DEVICES"] = ""
    gpu_count = 0
    if gpu:
        receipt = json.loads((prefix / "install-receipt.json").read_text())
        deployment = prefix / "config/deployment.json"
        configured = json.loads(deployment.read_text())["gpus"] if deployment.is_file() else receipt["gpu_indices"]
        indices = gpu_indices if gpu_indices is not None else configured
        inventory = check_driver(receipt["profile"], indices)
        uuids = [inventory[index][2] for index in indices]
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=20)
        if set(busy.split()) & set(uuids):
            raise ValueError("selected GPUs have compute processes; use the normal manager to stop them before an explicit GPU check")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
        gpu_count = len(uuids)
    run([python, "-m", "pip", "check"], env=env)
    code = ("import json,sys,torch;from pathlib import Path;"
            "from qwen_rl.deployment.instance_compat import verify_cpu,verify_gpu;"
            "r=verify_cpu(native_binary=Path(sys.argv[1]),cuobjdump=None);"
            + (("assert torch.cuda.device_count()==" + str(gpu_count) + ";r['gpu']=[];"
                "\nfor ordinal in range(torch.cuda.device_count()):\n with torch.cuda.device(ordinal): r['gpu'].append(verify_gpu())\n") if gpu else "")
            + "print(json.dumps(r))")
    run([python, "-B", "-c", code, prefix / "native/bin/ninfer-serve"], env=env)


def install(bundle: Path, prefix: Path, python: Path, gpu_indices: list[int], *, command_runner=run, cpu_only=False):
    prefix = prefix.expanduser().absolute()
    if prefix.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("installation prefix must be outside the immutable release bundle")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(prefix.parent / ("." + prefix.name + ".install.lock"),
                         os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another installation already owns this prefix") from None
        return _install(bundle, prefix, python, gpu_indices, command_runner=command_runner, cpu_only=cpu_only)
    finally:
        os.close(descriptor)


def _install(bundle: Path, prefix: Path, python: Path, gpu_indices: list[int], *, command_runner=run, cpu_only=False):
    print("autodl-install: stage 1/6 verify offline bundle", file=sys.stderr)
    manifest, digest = load_bundle(bundle)
    prefix = prefix.expanduser().absolute()
    if prefix.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("installation prefix must be outside the immutable release bundle")
    if prefix.is_symlink():
        raise ValueError("installation prefix must not be a symlink")
    validate_gpu_indices(gpu_indices)
    if not cpu_only:
        print("autodl-install: stage 2/6 validate selected GPU driver", file=sys.stderr)
        check_driver(manifest["profile"], gpu_indices)
    packages = _locked_packages(bundle)
    if packages.get("torch") != manifest.get("torch_version"):
        raise ValueError("offline dependency lock Torch identity differs from the bundle profile")
    proof = _installation_proof(manifest, packages, prefix)
    receipt = prefix / "install-receipt.json"
    if receipt.exists():
        if receipt.is_symlink() or not receipt.is_file():
            raise ValueError("installation receipt must be a regular file; use a fresh prefix")
        installed = json.loads(receipt.read_text())
        if not isinstance(installed, dict):
            raise ValueError("installation receipt is invalid; use a fresh prefix")
        if installed.get("schema") != "qwen-rl-autodl-install/v1" or installed.get("bundle_sha256") != digest:
            raise ValueError("prefix contains a different release; select a new prefix")
        if installed.get("prefix") != str(prefix):
            raise ValueError("installation receipt belongs to a different prefix; select a new prefix")
        if installed.get("installation_proof") is not None:
            if installed["installation_proof"] != proof:
                raise ValueError("installation receipt proof differs from the verified bundle; select a new prefix")
            print("autodl-install: stage 4/6 reuse and verify installed runtime proof", file=sys.stderr)
            _validate_installed_runtime(prefix, proof, command_runner=command_runner)
        else:
            # Upgrade a receipt from the first installer version only after a full
            # dependency check. Later runs use the smaller proof validation.
            print("autodl-install: stage 4/6 verify existing runtime before recording proof", file=sys.stderr)
            env = runtime_environment(prefix)
            env["CUDA_VISIBLE_DEVICES"] = ""
            command_runner([prefix / ".venv/bin/python", "-m", "pip", "check"], env=env)
            _validate_installed_runtime(prefix, proof, command_runner=command_runner)
            installed["installation_proof"] = proof
        if not cpu_only:
            installed.update(gpu_indices=gpu_indices, driver_validation="passed")
        print("autodl-install: stage 6/6 reuse complete; no package reinstall performed", file=sys.stderr)
        installed["installation_proof"] = proof
        receipt.write_text(json.dumps(installed, indent=2) + "\n")
        return installed
    marker = prefix / ".installing.json"
    if prefix.exists() and any(prefix.iterdir()):
        if marker.is_symlink() or not marker.is_file() or json.loads(marker.read_text()).get("bundle_sha256") != digest:
            raise ValueError("refusing to overwrite an unmanaged/nonmatching installation directory")
    print("autodl-install: stage 3/6 validate Python 3.12 bootstrap", file=sys.stderr)
    check_python(python)
    prefix.mkdir(parents=True, exist_ok=True)
    required_space = manifest["installed_size_bytes"] + 1024**3
    if shutil.disk_usage(prefix).free < required_space:
        raise ValueError(f"installation needs at least {required_space // 1024**2} MiB free for software and extraction")
    marker.write_text(json.dumps({"bundle_sha256": digest}))
    print("autodl-install: stage 4/6 copy source and native runtime", file=sys.stderr)
    for name in ["source", "native"]:
        if (prefix / name).is_symlink():
            raise ValueError("managed installation path became a symlink")
        shutil.copytree(bundle / name, prefix / name, dirs_exist_ok=True)
    env = runtime_environment(prefix)
    env["CUDA_VISIBLE_DEVICES"] = ""
    target_python = prefix / ".venv/bin/python"
    if not target_python.exists():
        print("autodl-install: stage 5/6 create private .venv and install local wheels", file=sys.stderr)
        command_runner([python, "-m", "venv", prefix / ".venv"], env=env)
    else:
        print("autodl-install: stage 5/6 install local wheels into private .venv", file=sys.stderr)
    # Complete, hash-locked wheel inventory: never fetch an index or compile a package.
    command_runner([target_python, "-m", "pip", "install", "--no-compile", "--force-reinstall", "--no-index", "--only-binary=:all:",
                    "--no-deps", "--require-hashes", "--find-links", bundle / "wheelhouse",
                    "-r", bundle / "requirements.lock.txt"], env=env)
    env = runtime_environment(prefix)
    env["CUDA_VISIBLE_DEVICES"] = ""
    liger_proof = manifest["liger_provenance"]
    code = ("import json,sys;from importlib.metadata import distribution;from qwen_rl.deployment.instance_compat import write_installed_source_receipt;"
            "p=json.loads(sys.argv[1]);write_installed_source_receipt(distribution('liger-kernel'),"
            "source_commit=p['source_commit'],files=p['files'])")
    command_runner([target_python, "-B", "-c", code, json.dumps(liger_proof)], env=env)
    command_runner([target_python, "-m", "pip", "check"], env=env)
    for name in ["triton", "torch", "cuda"]:
        (prefix / "cache" / name).mkdir(parents=True, exist_ok=True)
    launcher = prefix / "qwen-rl"
    launcher.write_text(_launcher_text(prefix, env), encoding="utf-8")
    launcher.chmod(0o755)
    result = {"schema": "qwen-rl-autodl-install/v1", "bundle_sha256": digest,
              "profile": manifest["profile"], "prefix": str(prefix), "gpu_indices": gpu_indices,
              "installation_proof": proof,
              "driver_validation": "not_performed" if cpu_only else "passed",
              "training_started": False, "gpu_validation": "not_performed"}
    print("autodl-install: stage 6/6 verify runtime and write installation proof", file=sys.stderr)
    _validate_installed_runtime(prefix, proof, command_runner=command_runner)
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    marker.unlink()
    return result


def configure(args):
    from qwen_rl.deployment.autodl_config import write_autodl_configs
    credentials = dict(item.split("=", 1) for item in args.credential)
    mappings = dict(item.split("=", 1) for item in args.path_map)
    return write_autodl_configs(
        runtime_yaml=args.runtime_config, install_root=args.prefix / "source",
        venv_python=args.prefix / ".venv/bin/python", native_binary=args.prefix / "native/bin/ninfer-serve",
        output_dir=args.prefix / "config", state_dir=args.state_dir or args.prefix / "state",
        gpu_indices=args.gpus, credential_paths=credentials, path_mappings=mappings,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="prepared release directory, including its wheelhouse")
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpus", type=lambda value: [int(x) for x in value.split(",")])
    gpu_group.add_argument("--gpu-count", type=int, choices=[2, 4, 8])
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--credential", action="append", default=[], metavar="ENV=PATH")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", help="prepare software without GPU inventory; driver/GPU validation remains not performed")
    parser.add_argument("--doctor", action="store_true", help="check an installed runtime without starting roles")
    parser.add_argument("--gpu-check", action="store_true", help="explicitly run the small CUDA kernel check")
    args = parser.parse_args(argv)
    args.prefix = args.prefix.expanduser().absolute()
    explicit_gpus = args.gpus if args.gpus is not None else (list(range(args.gpu_count)) if args.gpu_count else None)
    args.gpus = explicit_gpus if explicit_gpus is not None else [0, 1]
    if args.configure_only:
        if not args.runtime_config:
            parser.error("--configure-only requires --runtime-config")
        if Path(sys.prefix).resolve() != (args.prefix / ".venv").resolve():
            run([args.prefix / ".venv/bin/python", "-B", "-m", "qwen_rl.deployment.autodl_install",
                 *(argv if argv is not None else sys.argv[1:])], env=runtime_environment(args.prefix))
            return 0
        print("Configured runtime:", *configure(args), sep="\n  ")
        print("Next: run", args.prefix / "qwen-rl", "instance --deployment",
              args.prefix / "config/deployment.json", "doctor")
        return 0
    if args.doctor:
        print("autodl-install: stage 1/1 check installed runtime", file=sys.stderr)
        doctor(args.prefix, gpu=args.gpu_check, gpu_indices=explicit_gpus)
        return 0
    if not args.bundle:
        parser.error("--bundle is required for installation")
    if args.cpu_only and not (args.bundle / "bundle.json").is_file():
        parser.error("--cpu-only requires a specific cu129/cu130 bundle, since driver selection is unavailable")
    release = select_bundle(args.bundle.resolve(), args.gpus)
    selected_python = Path(shutil.which(str(args.python)) or str(args.python)).expanduser().resolve(strict=True)
    result = install(release, args.prefix, selected_python, args.gpus, cpu_only=args.cpu_only)
    if args.runtime_config:
        print("autodl-install: stage 1/1 configure local runtime", file=sys.stderr)
        command = [args.prefix / ".venv/bin/python", "-B", "-m", "qwen_rl.deployment.autodl_install",
                   "--configure-only", "--prefix", args.prefix, "--runtime-config", args.runtime_config.resolve(),
                   "--gpus", ",".join(map(str, args.gpus))]
        if args.state_dir:
            command += ["--state-dir", args.state_dir]
        for value in args.credential:
            command += ["--credential", value]
        for value in args.path_map:
            command += ["--path-map", value]
        run(command, env=runtime_environment(args.prefix))
    print(json.dumps(result, indent=2))
    if args.runtime_config:
        print("Next: run", args.prefix / "qwen-rl", "instance --deployment",
              args.prefix / "config/deployment.json", "doctor")
    else:
        print("Next: rerun this command with --runtime-config /absolute/path/to/runtime.yaml, then run",
              args.prefix / "qwen-rl", "instance --deployment", args.prefix / "config/deployment.json", "doctor")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(f"autodl-install: current stage failed with exit code {error.returncode}", file=sys.stderr)
            print("Next: correct the failed stage above and rerun the same installer command; an incomplete install keeps its matching retry marker.", file=sys.stderr)
        else:
            print("autodl-install:", error, file=sys.stderr)
            if "installation proof failed" in str(error):
                print("Next: choose a fresh --prefix after preserving any user data in the existing directory.", file=sys.stderr)
            else:
                print("Next: correct the reported input or prerequisite and rerun the same installer command; use a fresh --prefix if the existing receipt belongs to another release.", file=sys.stderr)
        raise SystemExit(2)
