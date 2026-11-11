"""Upgrade selected, approved, role-free Fleet nodes to the installed release."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..identity import atomic_write
from .registry import FleetRegistry, FleetRegistryError
from .settings import FleetSettings
from .ssh import FleetSSHError, run_admin_ssh

_SHA = re.compile(r"^[0-9a-f]{64}$")
_NODE = re.compile(r"^[A-Za-z0-9_.-]{1,253}$")
_ACTIVE = {"installing", "loading", "draining", "ready", "needs_reboot", "needs_driver", "needs_host_packages"}
_RELEASE_SCHEMA = "qwen-fleet-controller-deployment-receipt/v2"
_SPAN_TOKEN = re.compile(r"([0-9]+(?:\.[0-9]+)?)(usec|us|µs|μs|msec|ms|sec|s|min|hr|h|days|day|d|weeks|week|w)")
_SPAN_FACTOR = {
    "usec": 1, "us": 1, "µs": 1, "μs": 1, "msec": 1000, "ms": 1000,
    "sec": 1_000_000, "s": 1_000_000, "min": 60_000_000,
    "hr": 3_600_000_000, "h": 3_600_000_000, "day": 86_400_000_000,
    "days": 86_400_000_000, "d": 86_400_000_000,
    "week": 604_800_000_000, "weeks": 604_800_000_000, "w": 604_800_000_000,
}


class FleetUpgradeError(RuntimeError):
    """Safe operator-facing error for one-command node upgrades."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _regular(path: Path, *, mode: int | None = None) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise FleetUpgradeError("current Fleet deployment receipt or artifact is unavailable") from exc
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_uid != os.geteuid()
            or (mode is not None and stat.S_IMODE(info.st_mode) != mode)):
        raise FleetUpgradeError("current Fleet deployment receipt or artifact has unsafe ownership")
    return info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _systemd_timespan_usec(value: object) -> int | None:
    """Parse the human-readable WatchdogUSec value emitted by systemctl show."""
    if not isinstance(value, str) or not value.strip() or value.strip() == "infinity":
        return None
    source = value.strip()
    position = 0
    total = Decimal(0)
    for match in _SPAN_TOKEN.finditer(source):
        if source[position:match.start()].strip():
            return None
        try:
            total += Decimal(match.group(1)) * _SPAN_FACTOR[match.group(2)]
        except (InvalidOperation, KeyError):
            return None
        position = match.end()
    if position != len(source) or total != total.to_integral_value():
        return None
    return int(total)


def _watchdog_policy_ok(value: dict[str, Any]) -> bool:
    return (
        value.get("manager_active") is True
        and _systemd_timespan_usec(value.get("watchdog_usec")) == 120_000_000
        and value.get("notify_access") == "main" and value.get("restart") == "always"
        and value.get("tunnel_active") is True
        and _systemd_timespan_usec(value.get("tunnel_watchdog_usec")) == 60_000_000
        and value.get("tunnel_notify_access") == "main" and value.get("tunnel_restart") == "always"
        and value.get("ssh_active") is True
    )


def _published_release(config: FleetSettings) -> dict[str, str]:
    state = Path(config.state_dir).expanduser().resolve()
    receipt_path = state / "deployment-receipt.json"
    _regular(receipt_path, mode=0o600)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise FleetUpgradeError("current Fleet deployment receipt is invalid") from exc
    if not isinstance(receipt, dict):
        raise FleetUpgradeError("current Fleet deployment receipt is invalid")
    source_sha = receipt.get("source_bundle_sha256") if isinstance(receipt, dict) else None
    script_sha = receipt.get("script_sha256") if isinstance(receipt, dict) else None
    source_name = f"fleet-source-{source_sha}.tar.gz" if isinstance(source_sha, str) else ""
    hashes = receipt.get("public_hashes") if isinstance(receipt, dict) else None
    if (receipt.get("schema") != _RELEASE_SCHEMA or not isinstance(hashes, dict)
            or not isinstance(source_sha, str) or _SHA.fullmatch(source_sha) is None
            or not isinstance(script_sha, str) or _SHA.fullmatch(script_sha) is None
            or hashes.get("node-bootstrap.sh") != script_sha or hashes.get(source_name) != source_sha
            or not isinstance(receipt.get("public_files"), list)
            or not {"node-bootstrap.sh", source_name}.issubset(receipt["public_files"])):
        raise FleetUpgradeError("installed Fleet receipt does not identify the current published release")
    if config.public_origin is None or config.public_upload_dir is None:
        raise FleetUpgradeError("Fleet public origin and upload directory are required")
    origin = urlsplit(config.public_origin)
    if origin.scheme != "https" or not origin.netloc:
        raise FleetUpgradeError("Fleet public origin must use HTTPS")
    upload = Path(config.public_upload_dir).expanduser().resolve()
    script, archive = upload / "node-bootstrap.sh", upload / source_name
    _regular(script)
    _regular(archive)
    if _sha256(script) != script_sha or _sha256(archive) != source_sha:
        raise FleetUpgradeError("published Fleet artifacts do not match the installed release receipt")
    return {
        "source_sha256": source_sha,
        "script_sha256": script_sha,
        "bootstrap_url": f"https://{origin.netloc}/dl/upload/node-bootstrap.sh",
        "source_url": f"https://{origin.netloc}/dl/upload/{source_name}",
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    atomic_write(path, (json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(), mode=0o600)


def _receipt_dir(state_dir: Path, operation_id: str) -> Path:
    root = state_dir / "node-upgrade-operations"
    if not root.exists():
        root.mkdir(mode=0o700)
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or root.is_symlink() or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise FleetUpgradeError("Fleet upgrade receipt directory has unsafe ownership")
    target = root / operation_id
    target.mkdir(mode=0o700)
    return target


@contextmanager
def _node_lock(state_dir: Path, node_id: str):
    lock_dir = state_dir / "node-upgrade-locks"
    if not lock_dir.exists():
        try:
            lock_dir.mkdir(mode=0o700)
        except FileExistsError:
            pass
    info = lock_dir.lstat()
    if (not stat.S_ISDIR(info.st_mode) or lock_dir.is_symlink() or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise FleetUpgradeError("Fleet node-upgrade lock directory has unsafe ownership")
    name = hashlib.sha256(node_id.encode("ascii")).hexdigest() + ".lock"
    path = lock_dir / name
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as exc:
        raise FleetUpgradeError("Fleet node-upgrade lock could not be acquired safely") from exc
    try:
        file_info = os.fstat(fd)
        if (not stat.S_ISREG(file_info.st_mode) or file_info.st_uid != os.geteuid()
                or stat.S_IMODE(file_info.st_mode) != 0o600 or file_info.st_nlink != 1):
            raise FleetUpgradeError("Fleet node-upgrade lock file has unsafe ownership")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FleetUpgradeError("another upgrade operation already owns this node lock") from exc
        yield
    finally:
        os.close(fd)


def _resolve_node(registry: FleetRegistry, selector: str):
    matches = [node for node in registry.list_nodes() if node.node_id == selector or node.hostname == selector]
    if not matches:
        raise FleetUpgradeError(f"Fleet node {selector!r} was not found")
    if len(matches) != 1:
        raise FleetUpgradeError(f"Fleet node selector {selector!r} is ambiguous")
    return matches[0]


def _eligibility(node: Any) -> tuple[str | None, str]:
    if node.approval != "approved":
        return "blocked", "node is not approved"
    if node.heartbeat_stale:
        return "pending_unreachable", "heartbeat is stale; no remote action was attempted"
    if (node.desired_role is not None or node.observed_role is not None
            or node.desired_action not in {"none", "deactivate"}
            or node.deployment_id is not None or node.deployment_sha256 is not None):
        return "blocked", "node has an assigned role or deployment"
    if node.observed_state in _ACTIVE or node.observed_generation != node.desired_generation:
        return "blocked", "node has an in-flight or unacknowledged operation"
    receipt = node.observed_receipt or {}
    if not isinstance(receipt, dict) or receipt.get("role") in {"worker", "learner"}:
        return "blocked", "node receipt identifies a role"
    if receipt.get("status") in _ACTIVE or receipt.get("action") == "install":
        return "blocked", "node receipt identifies an in-flight operation"
    if node.desired_action == "deactivate" and (
            node.observed_state != "deactivated" or receipt.get("action") != "deactivate"
            or receipt.get("generation") != node.desired_generation):
        return "blocked", "current deactivation has not been acknowledged"
    if node.desired_action == "none" and node.observed_state == "deactivated":
        return "blocked", "registry state requires a current deactivation generation"
    return None, "eligible"


_REMOTE = r'''import hashlib,json,os,pwd,re,shlex,ssl,stat,subprocess,sys,urllib.request
from pathlib import Path
from urllib.parse import urlsplit
M="qwen-fleet-node.service"; T="qwen-fleet-node-tunnel.service"; S="qwen-fleet-node-ssh.service"
class Refuse(Exception): pass
def run(argv,timeout=20,ok=(0,)):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False)
 if p.returncode not in ok: raise Refuse("owned_command_failed")
 return p.stdout
def file(path):
 x=path.lstat()
 if not stat.S_ISREG(x.st_mode) or path.is_symlink() or x.st_uid!=os.geteuid() or stat.S_IMODE(x.st_mode)!=0o600 or path.resolve()!=path: raise Refuse("owned_file_invalid")
 return json.loads(path.read_text(encoding="utf-8"))
def private(path):
 x=path.lstat()
 if not stat.S_ISREG(x.st_mode) or path.is_symlink() or x.st_uid!=os.geteuid() or stat.S_IMODE(x.st_mode)!=0o600 or path.resolve()!=path: raise Refuse("owned_file_invalid")
def directory(path,create=False):
 if create and not path.exists(): path.mkdir(mode=0o700)
 x=path.lstat()
 if not stat.S_ISDIR(x.st_mode) or path.is_symlink() or x.st_uid!=os.geteuid() or path.resolve()!=path: raise Refuse("owned_directory_invalid")
 return path
def unit(name):
 out=run(["systemctl","--user","show","--no-pager","--property=LoadState,ActiveState,SubState,FragmentPath,Result,WatchdogUSec,NotifyAccess,Restart",name],10,(0,1))
 return dict(line.split("=",1) for line in out.splitlines() if "=" in line)
def unit_file(path):
 x=path.lstat()
 if not stat.S_ISREG(x.st_mode) or path.is_symlink() or x.st_uid!=os.geteuid(): raise Refuse("unit_not_owned")
 return path.read_text(encoding="utf-8")
def common(p):
 entry=pwd.getpwuid(os.geteuid()); home=Path(entry.pw_dir)
 if entry.pw_uid==0 or home!=Path.home(): raise Refuse("remote_account_invalid")
 root=directory(home/".local/share/qwen-fleet"); state=directory(root/"state"); mgmt=directory(root/"management")
 cfg=file(mgmt/"agent.json"); old=p.get("current_agent_version")
 if cfg.get("agent_version")!=old or cfg.get("state_dir")!=str(state): raise Refuse("source_version_mismatch")
 ca=home/".local/share/qwen-fleet/trust/upload-ca.pem"; private(ca)
 if cfg.get("ca_bundle")!=str(ca): raise Refuse("trust_path_mismatch")
 rel=directory(mgmt/"releases"/old); marker=file(rel/".qwen-fleet-release.json")
 if marker.get("bundle_sha256")!=old or not isinstance(marker.get("files"),dict): raise Refuse("source_receipt_mismatch")
 agent=file(state/"agent-state.json")
 if agent.get("node_id")!=p.get("node_id") or agent.get("schema")!="qwen-fleet-agent-state/v1": raise Refuse("identity_receipt_mismatch")
 if agent.get("terminal_rejection") or agent.get("active_role") is not None: raise Refuse("active_role")
 if agent.get("desired_role") is not None or agent.get("desired_action") not in ("none","deactivate"): raise Refuse("active_role")
 op=agent.get("operation")
 if op is not None and not (isinstance(op,dict) and op.get("status")=="deactivated" and op.get("role") is None and op.get("ack_accepted") is True): raise Refuse("node_operation_in_flight")
 roles=agent.get("owned_roles")
 if not isinstance(roles,dict) or any(not isinstance(v,dict) or v.get("status")!="stopped" for v in roles.values()): raise Refuse("active_role")
 ud=directory(home/".config/systemd/user"); services={}
 source_root=str(mgmt/"releases"/old)
 for name in (M,T,S):
  v=unit(name)
  if v.get("LoadState")!="loaded" or v.get("FragmentPath")!=str(ud/name): raise Refuse("unit_not_owned")
  contents=unit_file(ud/name)
  if source_root not in contents: raise Refuse("unit_source_mismatch")
  if name==M and "qwen_rl.cluster.fleet.agent" not in contents: raise Refuse("manager_exec_mismatch")
  services[name]=v
 if services[T].get("ActiveState")!="active" or services[S].get("ActiveState")!="active": raise Refuse("ssh_tunnel_unavailable")
 return home,state,mgmt,cfg,ca,services
def emit(x): print(json.dumps(x,sort_keys=True,separators=(",",":")))
def main(action,p):
 if action in ("status","cleanup"):
  entry=pwd.getpwuid(os.geteuid()); home=Path(entry.pw_dir)
  if entry.pw_uid==0 or home!=Path.home(): raise Refuse("remote_account_invalid")
  root=directory(home/".local/share/qwen-fleet"); state=directory(root/"state"); mgmt=directory(root/"management")
  cfg=file(mgmt/"agent.json"); ca=home/".local/share/qwen-fleet/trust/upload-ca.pem"; private(ca); sv={}
 else: home,state,mgmt,cfg,ca,sv=common(p)
 m=sv.get(M,{})
 if action in ("preflight","verify"):
  if action=="verify" and (cfg.get("agent_version")!=p.get("source_sha256") or m.get("ActiveState")!="active"): raise Refuse("manager_not_healthy")
  if action=="verify" and any(sv[n].get("ActiveState")!="active" for n in (T,S)): raise Refuse("ssh_tunnel_unavailable")
  return {"ok":True,"uid":os.geteuid(),"agent_version":cfg["agent_version"],"source_sha256":p["current_agent_version"],
          "manager_active":m.get("ActiveState")=="active","watchdog_usec":m.get("WatchdogUSec"),
          "notify_access":m.get("NotifyAccess"),"restart":m.get("Restart"),
          "tunnel_active":sv[T].get("ActiveState")=="active","tunnel_watchdog_usec":sv[T].get("WatchdogUSec"),
          "tunnel_notify_access":sv[T].get("NotifyAccess"),"tunnel_restart":sv[T].get("Restart"),
          "ssh_active":sv[S].get("ActiveState")=="active"}
 op=p["operation_id"]
 if not re.fullmatch(r"[0-9a-f]{32}",op): raise Refuse("operation_id_invalid")
 stem="qwen-fleet-upgrade-"+op; gs=stem+"-guard.service"; gt=stem+"-guard.timer"; rs=stem+"-runner.service"
 if action=="prepare":
  if p.get("source_sha256")!=p.get("target_source_sha256") or not re.fullmatch(r"[0-9a-f]{64}",p["source_sha256"]): raise Refuse("release_pin_invalid")
  tmp=directory(state/"tmp",True); opdir=tmp/("upgrade-"+op)
  if opdir.exists() or opdir.is_symlink(): raise Refuse("operation_directory_exists")
  opdir.mkdir(mode=0o700); unitdir=home/".config/systemd/user"; created=[]
  try:
   ctx=ssl.create_default_context(cafile=str(ca)); opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=ctx))
   for name,url,digest,limit in (("node-bootstrap.sh",p["bootstrap_url"],p["script_sha256"],4*1024*1024),("fleet-source-"+p["source_sha256"]+".tar.gz",p["source_url"],p["source_sha256"],256*1024*1024)):
    if not url.startswith("https://"): raise Refuse("artifact_url_not_https")
    req=urllib.request.Request(url,headers={"Accept":"application/octet-stream","User-Agent":"qwen-fleet-upgrade/1"}); count=0; sha=hashlib.sha256()
    try:
     with opener.open(req,timeout=30) as response:
      final=urlsplit(response.geturl()); original=urlsplit(url)
      if final.scheme!="https" or final.netloc.lower()!=original.netloc.lower(): raise Refuse("artifact_redirect_refused")
      with (opdir/name).open("xb") as out:
       os.chmod(opdir/name,0o600)
       while True:
        part=response.read(1024*1024)
        if not part: break
        count+=len(part)
        if count>limit: raise Refuse("artifact_too_large")
        sha.update(part); out.write(part)
    except Refuse: raise
    except Exception: raise Refuse("artifact_download_failed")
    if sha.hexdigest()!=digest: raise Refuse("artifact_sha256_mismatch")
   script=(opdir/"node-bootstrap.sh").read_bytes()
   if ("fleet-source-"+p["source_sha256"]+".tar.gz").encode() not in script: raise Refuse("bootstrap_pin_mismatch")
   runner="""#!/bin/bash\nset -Eeuo pipefail\numask 077\nrestore() { rc=$?; if ((rc != 0)); then /usr/bin/systemctl --user start qwen-fleet-node-tunnel.service >/dev/null 2>&1 || true; /usr/bin/systemctl --user start qwen-fleet-node.service >/dev/null 2>&1 || true; fi; exit \"$rc\"; }\ntrap restore EXIT\nbash ./node-bootstrap.sh >bootstrap.log 2>&1\n/usr/bin/systemctl --user start qwen-fleet-node-tunnel.service\n/usr/bin/systemctl --user start qwen-fleet-node.service\n/usr/bin/systemctl --user is-active --quiet qwen-fleet-node-ssh.service\n/usr/bin/systemctl --user is-active --quiet qwen-fleet-node.service\ntrap - EXIT\n"""
   with (opdir/"run.sh").open("x",encoding="utf-8") as out: out.write(runner)
   os.chmod(opdir/"run.sh",0o600)
   service=(f"[Unit]\nDescription=Qwen Fleet upgrade recovery guard {op}\n[Service]\nType=oneshot\nRuntimeMaxSec=240s\nUMask=0077\n"
            f"ExecStart=/usr/bin/systemctl --user start {T}\nExecStart=/usr/bin/systemctl --user start {M}\n")
   timer=(f"[Unit]\nDescription=Qwen Fleet upgrade recovery timer {op}\n[Timer]\nOnActiveSec=150s\nAccuracySec=1s\n" f"Unit={gs}\n")
   for name,text in ((gs,service),(gt,timer)):
    path=unitdir/name
    if path.exists() or path.is_symlink(): raise Refuse("operation_unit_exists")
    with path.open("x",encoding="utf-8") as out: out.write(text)
    os.chmod(path,0o600)
    created.append(name)
   run(["systemctl","--user","daemon-reload"])
   run(["systemctl","--user","start",gt],30)
   if unit(gt).get("ActiveState")!="active": raise Refuse("recovery_guard_not_active")
   run(["systemctl","--user","stop",M],45)
   return {"ok":True,"operation_id":op,"runner_service":rs,"guard_timer":gt}
  except Exception:
   try: run(["systemctl","--user","start",T],30); run(["systemctl","--user","start",M],30)
   except Exception: pass
   if created:
    for name in created: run(["systemctl","--user","stop",name],15,(0,1))
    for name in created:
     try: (unitdir/name).unlink()
     except OSError: pass
    run(["systemctl","--user","daemon-reload"],15,(0,1))
   raise
 if action=="launch":
  opdir=state/"tmp"/("upgrade-"+op); directory(opdir)
  unitdir=home/".config/systemd/user"; path=unitdir/rs
  if path.exists() or path.is_symlink(): raise Refuse("operation_unit_exists")
  working=str(opdir).replace("%","%%").replace(" ","\\x20")
  text=(f"[Unit]\nDescription=Qwen Fleet reviewed source upgrade {op}\n[Service]\nType=oneshot\nUMask=0077\nRuntimeMaxSec=240s\n"
        f"TimeoutStartSec=240s\nRemainAfterExit=yes\nWorkingDirectory={working}\nExecStart=/bin/bash run.sh\n")
  with path.open("x",encoding="utf-8") as out: out.write(text)
  os.chmod(path,0o600); run(["systemctl","--user","daemon-reload"]); run(["systemctl","--user","start","--no-block",rs],30)
  return {"ok":True,"runner_service":rs}
 if action=="status":
  value=unit(rs)
  status="failed" if value.get("ActiveState")=="failed" or value.get("Result") not in (None,"","success") else "succeeded" if value.get("ActiveState")=="active" and value.get("SubState")=="exited" else "running"
  return {"ok":True,"status":status}
 if action=="cleanup":
  unitdir=home/".config/systemd/user"
  descriptions={gs:"Qwen Fleet upgrade recovery guard "+op,gt:"Qwen Fleet upgrade recovery timer "+op,
                rs:"Qwen Fleet reviewed source upgrade "+op}
  owned=[]
  for name,description in descriptions.items():
   path=unitdir/name
   try:
    x=path.lstat()
    body=path.read_text(encoding="utf-8")
    if (stat.S_ISREG(x.st_mode) and not path.is_symlink() and x.st_uid==os.geteuid()
            and stat.S_IMODE(x.st_mode)==0o600 and "Description="+description in body): owned.append(name)
   except FileNotFoundError: pass
  for name in owned: run(["systemctl","--user","stop",name],20,(0,1))
  run(["systemctl","--user","start",T],30); run(["systemctl","--user","start",M],30)
  for name in owned:
   path=unitdir/name
   try:
    x=path.lstat()
    if (stat.S_ISREG(x.st_mode) and not path.is_symlink() and x.st_uid==os.geteuid()
            and stat.S_IMODE(x.st_mode)==0o600): path.unlink()
   except FileNotFoundError: pass
  run(["systemctl","--user","daemon-reload"])
  return {"ok":True,"manager_active":unit(M).get("ActiveState")=="active"}
 raise Refuse("unsupported_remote_action")
try: result=main(sys.argv[1],json.loads(sys.argv[2]))
except Refuse as exc: result={"ok":False,"code":str(exc) or "guard_refused"}
except Exception as exc: result={"ok":False,"code":"remote_error","error_type":type(exc).__name__}
print(json.dumps(result,sort_keys=True,separators=(",",":")))
'''


class NodeRemote:
    """Bounded OpenSSH RPC using the existing pinned admin SSH route."""

    def __init__(self, config: FleetSettings, runner: Callable[..., Any] | None = None) -> None:
        self.config = config
        self.runner = runner or run_admin_ssh

    def call(self, node: Any, action: str, payload: dict[str, Any], *, timeout: float = 30) -> dict[str, Any]:
        command = ["python3", "-B", "-c", _REMOTE, action,
                   json.dumps(payload, sort_keys=True, separators=(",", ":"))]
        result = self.runner(node, command, state_dir=Path(self.config.state_dir),
                             identity_file=Path(self.config.ssh_admin_identity_file),
                             timeout=timeout, capture_output=True)
        if result.returncode != 0:
            raise FleetUpgradeError("Fleet node SSH operation failed")
        try:
            value = json.loads(result.stdout)
        except (TypeError, ValueError) as exc:
            raise FleetUpgradeError("Fleet node returned an invalid upgrade receipt") from exc
        if not isinstance(value, dict) or type(value.get("ok")) is not bool:
            raise FleetUpgradeError("Fleet node returned an invalid upgrade receipt")
        if not value["ok"]:
            raise FleetUpgradeError(f"remote upgrade guard refused ({value.get('code', 'unknown')})")
        return value


def _payload(node: Any, release: dict[str, str], operation_id: str) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "current_agent_version": node.agent_version,
        "target_source_sha256": release["source_sha256"],
        **release,
        "operation_id": operation_id,
    }


def _management_is_current(node: Any) -> bool:
    receipt = node.observed_receipt or {}
    if node.heartbeat_stale or node.observed_generation != node.desired_generation:
        return False
    if node.desired_action == "deactivate":
        return (node.observed_state == "deactivated" and isinstance(receipt, dict)
                and receipt.get("action") == "deactivate" and receipt.get("generation") == node.desired_generation)
    return node.desired_action == "none" and node.observed_state != "deactivated"


def _acknowledged(node: Any, generation: int, source_sha: str) -> bool:
    receipt = node.observed_receipt or {}
    return bool(
        not node.heartbeat_stale and node.agent_version == source_sha
        and node.desired_generation == generation and node.observed_generation == generation
        and node.desired_action == "deactivate" and node.desired_role is None and node.observed_role is None
        and node.observed_state == "deactivated" and isinstance(receipt, dict)
        and receipt.get("action") == "deactivate" and receipt.get("generation") == generation
    )


def _remaining_timeout(deadline: float, ceiling: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise FleetUpgradeError("per-node upgrade deadline expired")
    return min(ceiling, remaining)


def _one_node(
    registry: FleetRegistry, remote: NodeRemote, config: FleetSettings, node: Any,
    release: dict[str, str], operation_id: str, *, wait_seconds: float, poll_seconds: float,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "node_id": node.node_id, "hostname": node.hostname,
        "from_source_sha256": node.agent_version, "to_source_sha256": release["source_sha256"],
        "started_at": _timestamp(), "status": "pending",
    }
    deadline = time.monotonic() + wait_seconds
    state, reason = _eligibility(node)
    if state is not None:
        record.update(status=state, detail=reason)
        return record
    payload = _payload(node, release, operation_id)
    try:
        remote_state = remote.call(node, "preflight", payload, timeout=_remaining_timeout(deadline, 30))
    except (FleetSSHError, FleetUpgradeError) as exc:
        record.update(status="pending_unreachable", detail=type(exc).__name__)
        return record
    if remote_state.get("uid") != node.account.uid:
        record.update(status="blocked", detail="SSH UID differs from registered account")
        return record
    if (remote_state.get("agent_version") != node.agent_version
            or remote_state.get("source_sha256") != node.agent_version):
        record.update(status="blocked", detail="node source receipt differs from Registry")
        return record
    if (node.agent_version == release["source_sha256"] and _management_is_current(node)
            and _watchdog_policy_ok(remote_state)):
        record.update(status="already_current", detail="source, heartbeat, watchdog, SSH and tunnel are healthy")
        return record

    try:
        remote.call(node, "prepare", payload, timeout=_remaining_timeout(deadline, 75))
    except (FleetSSHError, FleetUpgradeError) as exc:
        try:
            remote.call(node, "cleanup", payload, timeout=45)
            record["manager_recovery"] = "active"
        except (FleetSSHError, FleetUpgradeError):
            record["manager_recovery"] = "pending"
        record.update(status="failed", detail=f"guard or artifact preflight failed ({type(exc).__name__})")
        return record
    try:
        deactivation = registry.refresh_management(node.node_id)
    except Exception as exc:  # noqa: BLE001 - restore the node if the registry CAS fails unexpectedly.
        try:
            remote.call(node, "cleanup", payload, timeout=45)
            record["manager_recovery"] = "active"
        except (FleetSSHError, FleetUpgradeError):
            record["manager_recovery"] = "pending"
        code = exc.code if isinstance(exc, FleetRegistryError) else type(exc).__name__
        record.update(status="blocked", detail=f"Registry compare-and-swap refused ({code})")
        return record
    generation = deactivation.generation
    record["generation"] = generation
    try:
        remote.call(node, "launch", payload, timeout=_remaining_timeout(deadline, 30))
    except (FleetSSHError, FleetUpgradeError) as exc:
        try:
            remote.call(node, "cleanup", payload, timeout=45)
            record["manager_recovery"] = "active"
        except (FleetSSHError, FleetUpgradeError):
            record["manager_recovery"] = "pending"
        record.update(status="failed", detail=f"normal bootstrap could not launch ({type(exc).__name__})")
        return record

    status = "running"
    last_poll_error: str | None = None
    while time.monotonic() < deadline:
        try:
            status = remote.call(node, "status", payload,
                                 timeout=_remaining_timeout(deadline, 20)).get("status", "running")
            current = registry.get_node(node.node_id)
        except (FleetRegistryError, FleetSSHError, FleetUpgradeError) as exc:
            last_poll_error = type(exc).__name__
            sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
            continue
        if status == "failed":
            record.update(status="failed", detail="normal bootstrap installer failed")
            break
        if status == "succeeded" and _acknowledged(current, generation, release["source_sha256"]):
            try:
                verified = remote.call(
                    node, "verify", {**payload, "current_agent_version": release["source_sha256"]},
                    timeout=_remaining_timeout(deadline, 30),
                )
                if verified.get("uid") != node.account.uid or not _watchdog_policy_ok(verified):
                    raise FleetUpgradeError("watchdog unit policy or SSH UID differs from the accepted contract")
                latest = registry.get_node(node.node_id)
                if not _acknowledged(latest, generation, release["source_sha256"]):
                    raise FleetUpgradeError("node generation changed during final verification")
                record.update(status="upgraded", detail="source, generation ACK, watchdog services and WSS UID verified")
            except (FleetSSHError, FleetUpgradeError) as exc:
                record.update(status="failed", detail=f"final verification failed ({type(exc).__name__})")
            break
        if current.desired_role is not None or current.observed_role is not None or current.desired_action != "deactivate":
            record.update(status="blocked", detail="a role assignment raced with the management upgrade")
            break
        sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    else:
        state = "pending_unreachable" if last_poll_error else "pending"
        detail = (f"bounded status wait expired after transient errors ({last_poll_error})"
                  if last_poll_error else "bounded upgrade wait expired; independent recovery timer remains armed")
        record.update(status=state, detail=detail)

    if record["status"] in {"upgraded", "failed"}:
        try:
            remote.call(node, "cleanup", payload, timeout=45)
            record["manager_recovery"] = "active"
        except (FleetSSHError, FleetUpgradeError):
            record["manager_recovery"] = "pending"
            if record["status"] == "upgraded":
                record.update(status="failed", detail="upgrade verified but recovery guard cleanup failed")
    record["finished_at"] = _timestamp()
    return record


def upgrade_nodes(
    registry: FleetRegistry,
    config: FleetSettings,
    selectors: Sequence[str],
    *,
    ssh_runner: Callable[..., Any] | None = None,
    wait_seconds: float = 180,
    poll_seconds: float = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Upgrade selected nodes sequentially; offline/ineligible nodes remain explicit results."""
    if not selectors or len(selectors) != len(set(selectors)):
        raise FleetUpgradeError("provide one or more unique --node selectors")
    if any(not isinstance(value, str) or _NODE.fullmatch(value) is None for value in selectors):
        raise FleetUpgradeError("node selectors may contain only a registered node ID or hostname")
    if not 0 < wait_seconds <= 180 or poll_seconds <= 0:
        raise FleetUpgradeError("upgrade wait must be between 0 and 180 seconds")
    release = _published_release(config)
    operation_id = uuid.uuid4().hex
    state_dir = Path(config.state_dir).expanduser().resolve()
    operation_dir = _receipt_dir(state_dir, operation_id)
    receipt_path = operation_dir / "receipt.json"
    receipt: dict[str, Any] = {
        "schema": "qwen-fleet-node-upgrades/v1", "operation_id": operation_id,
        "started_at": _timestamp(),
        "release": {"source_sha256": release["source_sha256"], "script_sha256": release["script_sha256"]},
        "nodes": [],
    }
    _write_receipt(receipt_path, receipt)
    remote = NodeRemote(config, ssh_runner)
    for selector in selectors:
        try:
            node = _resolve_node(registry, selector)
            with _node_lock(state_dir, node.node_id):
                result = _one_node(registry, remote, config, node, release, operation_id,
                                   wait_seconds=wait_seconds, poll_seconds=poll_seconds, sleep=sleep)
        except (FleetUpgradeError, FleetRegistryError) as exc:
            result = {"selector": selector, "status": "blocked", "detail": str(exc)}
        receipt["nodes"].append(result)
        receipt["updated_at"] = _timestamp()
        _write_receipt(receipt_path, receipt)
    receipt["finished_at"] = _timestamp()
    _write_receipt(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt
