"""PEFT-backed FP32 transport with an explicit bijection to live LoRA names."""
from pathlib import Path
import errno
import os
import shutil


def storage_name(name):
    for side in ('A','B'):
        suffix = f'.lora_{side}.default.weight'
        if name.endswith(suffix):
            return name[:-len(suffix)] + f'.lora_{side}.weight'
    return name


def validate_training_state(path, layout):
    from safetensors import safe_open
    mapping = {storage_name(name): name for name in layout}
    if len(mapping) != len(layout):
        raise ValueError('training storage key mapping is not bijective')
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(mapping):
            raise ValueError('training storage names differ from the leased layout')
        for stored, canonical in mapping.items():
            view = handle.get_slice(stored)
            if tuple(view.get_shape()) != tuple(layout[canonical]):
                raise ValueError(f'training state shape differs for {canonical}')
            if view.get_dtype() not in {'F32','F16','BF16'}:
                raise ValueError(f'training state dtype is unsupported for {canonical}')


def load_training_state(path, expected_names):
    from safetensors.torch import load_file
    state = load_file(str(path), device='cpu')
    expected_names = tuple(expected_names)
    mapping = {storage_name(name):name for name in expected_names}
    if len(mapping) != len(expected_names) or set(state) != set(mapping):
        raise ValueError('training storage keys differ from the exact leased inventory')
    return {mapping[name]:tensor for name,tensor in state.items()}


def materialize_training_state(source, target, layout):
    """Share the immutable FP32 PEFT inode; cross-filesystem delivery copies bytes."""
    from safetensors import safe_open
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_file():
        raise ValueError('training backing file must be a regular PEFT artifact')
    validate_training_state(source, layout)
    with safe_open(str(source), framework='pt', device='cpu') as handle:
        if any(handle.get_slice(key).get_dtype() != 'F32' for key in handle.keys()):
            raise ValueError('shared training backing requires full FP32 masters')
    source.chmod(0o444)
    try:
        os.link(source, target, follow_symlinks=False)
        copied = False
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, target)
        copied = True
    target.chmod(0o444)
    return {'copied_bytes':source.stat().st_size if copied else 0,
            'linked_bytes':0 if copied else source.stat().st_size}
