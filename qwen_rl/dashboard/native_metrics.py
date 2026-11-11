"""Read in-flight decode counters from owned local Native engines, without GPU work."""
from __future__ import annotations

import json
import re
import subprocess
from urllib.parse import urlsplit

# Runs in the already-owned controller's network namespace. No credentials or model
# payloads enter this observer; only bounded capabilities and Prometheus counters.
_PROBE = r'''
import concurrent.futures,json,math,sys,time,urllib.request
items=json.loads(sys.argv[1])
def fetch(origin,path):
    with urllib.request.urlopen(origin+path,timeout=1.5) as response:
        raw=response.read(262145)
    if len(raw)>262144: raise ValueError('oversized metrics response')
    return raw.decode('utf-8')
def probe(item):
    result={'id':item['id'],'telemetry_epoch':item['epoch'],'live':False}
    try:
        cap=json.loads(fetch(item['origin'],'/v1/rl/capabilities'))
        if cap.get('backend')!='ninfer' or cap.get('base_fingerprint')!=item['base'] or cap.get('ready') is not True:
            raise ValueError('native identity/readiness mismatch')
        raw=fetch(item['origin'],'/metrics')
        values={}
        for line in raw.splitlines():
            if line and not line.startswith('#'):
                fields=line.split()
                if len(fields)==2 and fields[0]=='llamacpp:tokens_predicted_total':
                    value=float(fields[1])
                    if not math.isfinite(value) or value<0 or not value.is_integer():
                        raise ValueError('invalid Native decode counter')
                    values[fields[0]]=int(value)
        result.update(decode_tokens=values['llamacpp:tokens_predicted_total'],live=True,
                      monotonic_sample=time.monotonic(),sampled_at=time.time(),
                      artifact_id=cap.get('ninfer_artifact_id'),build_id=cap.get('build_id'))
    except Exception as exc:
        result['error']=type(exc).__name__
    return result
with concurrent.futures.ThreadPoolExecutor(max_workers=min(8,len(items))) as pool:
    print(json.dumps(list(pool.map(probe,items))))
'''


def _inputs(source, snapshot):
    """Use only the selected fleet's loopback engine origins and live Worker identities."""
    from .central import _read

    if (source.kind != 'local' or source.runtime_registry_root is None
            or not re.fullmatch(r'[0-9a-f]{64}', source.controller_id or '')):
        return []
    runtime = source.runtime or {}
    if (runtime.get('worker_inference') or {}).get('backend') != 'ninfer':
        return []
    manifest = _read(source.runtime_registry_root.parent / 'manifest.json')
    if (not manifest or manifest.get('schema') != 'qwen-rl-local-worker-fleet/v1'
            or manifest.get('manager_container') != source.deployment_id):
        return []
    configured = {w['id'] for w in runtime.get('workers', []) if isinstance(w, dict) and 'id' in w}
    rows = {w['id']: w for w in snapshot.get('workers', []) if isinstance(w, dict) and 'id' in w}
    origins = {w.get('worker_id'): w.get('ninfer_url') for w in manifest.get('workers', [])}
    items = []
    for name in sorted(configured):
        row = rows.get(name, {})
        origin = origins.get(name)
        if not isinstance(origin, str):
            return []
        try:
            parsed = urlsplit(origin)
            port = parsed.port
        except ValueError:
            return []
        if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
                or not port or parsed.username or parsed.password
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
            return []
        base = (row.get('runtime') or {}).get('base_fingerprint')
        if (row.get('live') is not True or not isinstance(row.get('telemetry_epoch'), str)
                or not isinstance(base, str) or not re.fullmatch(r'[0-9a-f]{64}', base)):
            return []
        items.append({'id': name, 'origin': origin.rstrip('/'),
                      'epoch': row['telemetry_epoch'], 'base': base})
    return items if 0 < len(items) <= 8 else []


def observe_native_metrics(source, snapshot, *, runner=subprocess.run):
    """Return a separate metric family; never turn computed tokens into accepted tokens."""
    items = _inputs(source, snapshot)
    if not items:
        return snapshot
    native = {'measurement_scope': 'engine_committed_decode_walltime', 'workers': []}
    try:
        result = runner(['docker', 'exec', source.controller_id, 'python3', '-c', _PROBE,
                         json.dumps(items, separators=(',', ':'))],
                        capture_output=True, text=True, timeout=5, check=False)
        if result.returncode != 0 or len(result.stdout) > 262144:
            raise ValueError('Native metrics observer unavailable')
        rows = json.loads(result.stdout)
        expected = {i['id']: i['epoch'] for i in items}
        if (not isinstance(rows, list) or len(rows) != len(items)
                or {r.get('id') for r in rows if isinstance(r, dict)} != set(expected)
                or any(not isinstance(r, dict) or r.get('telemetry_epoch') != expected.get(r.get('id')) for r in rows)):
            raise ValueError('Native metrics Worker identity mismatch')
        native['workers'] = rows
    except (OSError, ValueError, subprocess.SubprocessError):
        native['error'] = 'Native decode telemetry unavailable'
    return {**snapshot, 'native_metrics': native}
