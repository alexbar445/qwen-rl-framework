"""Bind the Learner's mask compiler to the Native engine's patched source tree."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import re


@lru_cache(maxsize=1)
def native_grammar_compiler_sha256() -> str:
    spec=importlib.util.find_spec('xgrammar')
    if spec is None or spec.origin is None:
        raise RuntimeError('conditioned ARC requires the matching Native XGrammar wheel')
    root=Path(spec.origin).parent
    manifest=root/'ninfer_policy.json'
    if not manifest.is_file():
        raise RuntimeError('stock XGrammar has different masks; build the matching wheel with scripts/build_native_xgrammar_wheel.py')
    value=json.loads(manifest.read_text())
    digest=value.get('compiler_sha256')
    library=value.get('library')
    if (value.get('schema')!='ninfer-xgrammar-policy/v1' or not isinstance(digest,str)
            or re.fullmatch('[0-9a-f]{64}',digest) is None or not isinstance(library,str)):
        raise RuntimeError('invalid Native grammar compiler manifest')
    path=(root/library).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise RuntimeError('invalid Native grammar compiler library path')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=value.get('library_sha256'):
        raise RuntimeError('Native grammar compiler library fingerprint mismatch')
    return digest
