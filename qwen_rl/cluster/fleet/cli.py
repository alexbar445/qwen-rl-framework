"""Operator CLI for Fleet node approval and private role assignments."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Sequence

from pydantic import TypeAdapter, ValidationError

from ..identity import atomic_write, ensure_private_dir
from .contracts import FleetRoleTemplate
from .private_artifacts import PrivateArtifactError, PrivateArtifactStore
from .registry import FleetRegistry, FleetRegistryError
from .settings import FleetSettings
from .server import create_fleet_app, refresh_relay_authorized_keys, require_relay_setup


_ROLE_ADAPTER = TypeAdapter(FleetRoleTemplate)


class FleetCLIError(RuntimeError):
    """Safe operator-facing Fleet CLI failure."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _emit(value: Any) -> None:
    print(_json(value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-rl cluster fleet")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create private Fleet settings and SSH identities")
    init.add_argument("--state-dir", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--relay-user", default=getpass.getuser())
    init.add_argument("--relay-host", default="127.0.0.1")
    init.add_argument("--relay-port", type=int, default=22230)
    init.add_argument("--public-origin")
    init.add_argument("--public-upload-dir", type=Path)
    init.add_argument("--max-registered-nodes", type=int, default=1000)
    init.add_argument("--max-pending-nodes", type=int, default=100)
    init.add_argument("--max-rejected-tombstones", type=int, default=1000)
    init.add_argument("--registration-rate-per-minute", type=int, default=120)
    init.add_argument("--reverse-port-start", type=int, default=24000)
    init.add_argument("--reverse-port-end", type=int, default=34000)

    serve = sub.add_parser("serve", help="serve the loopback-only Fleet API")
    serve.add_argument("--config", type=Path, required=True)

    for name in ("list", "inspect", "audit", "approve", "reject", "assign", "deactivate", "refresh-management",
                 "upgrade-nodes", "operation-start", "operation-status", "operation-resume",
                 "ssh", "exec", "logs",
                 "publish-private-bundle"):
        help_text = ("advance a safe role-free node to a new management generation"
                     if name == "refresh-management" else
                     "upgrade approved role-free nodes to the currently installed public release"
                     if name == "upgrade-nodes" else None)
        command = sub.add_parser(name, help=help_text)
        operator_command = name == "upgrade-nodes" or name.startswith("operation-")
        default_config = (Path.home() / ".local/share/qwen-fleet-control/state/fleet.json"
                          if operator_command else None)
        command.add_argument("--config", type=Path, required=not operator_command, default=default_config)
        if name == "inspect":
            command.add_argument("node_id")
        elif name in {"approve", "reject"}:
            command.add_argument("node_id")
        elif name == "assign":
            command.add_argument("node_id")
            command.add_argument("--deployment-file", type=Path, required=True)
        elif name in {"deactivate", "refresh-management"}:
            command.add_argument("node_id")
        elif name == "upgrade-nodes":
            command.add_argument("--node", action="append", required=True,
                                 help="registered node ID or exact hostname; may be repeated")
        elif name == "operation-start":
            command.add_argument("--plan", type=Path, required=True)
        elif name in {"operation-status", "operation-resume"}:
            command.add_argument("--operation-id", required=True)
        elif name in {"ssh", "exec"}:
            command.add_argument("node_id")
            command.add_argument("argv", nargs=argparse.REMAINDER)
        elif name == "logs":
            command.add_argument("node_id")
            command.add_argument("--lines", type=int, default=100)
        elif name == "publish-private-bundle":
            command.add_argument("--source-file", type=Path, required=True)
    return parser


def _settings(args: argparse.Namespace) -> FleetSettings:
    try:
        return FleetSettings.load(args.config)
    except (OSError, ValueError, ValidationError) as exc:
        raise FleetCLIError("Fleet settings could not be loaded") from exc


def _registry(config: FleetSettings) -> FleetRegistry:
    return FleetRegistry(
        config.state_dir,
        heartbeat_ttl_seconds=config.heartbeat_ttl_seconds,
        max_registered_nodes=config.max_registered_nodes,
        max_pending_nodes=config.max_pending_nodes,
        max_rejected_tombstones=config.max_rejected_tombstones,
        registration_rate_per_minute=config.registration_rate_per_minute,
        reverse_port_start=config.reverse_port_start,
        reverse_port_end=config.reverse_port_end,
        public_origin=config.public_origin,
    )


def _private_regular_file(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser()
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FleetCLIError(f"{label} is unavailable") from exc
    if (not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or stat.S_IMODE(metadata.st_mode) != 0o600
            or (os.name == "posix" and metadata.st_uid != os.geteuid())):
        raise FleetCLIError(f"{label} must be an owner-matched regular mode-0600 file")
    return path


def _outside_public_upload(path: Path, public_upload_dir: str | Path | None, *, label: str) -> None:
    if public_upload_dir is None:
        return
    source = Path(path).resolve()
    public = Path(public_upload_dir).expanduser().resolve()
    if source == public or source.is_relative_to(public):
        raise FleetCLIError(f"{label} must be outside the public upload tree")


def _json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON value")


def _read_private_assignment(path: Path, *, public_upload_dir: str | Path | None):
    source = _private_regular_file(path, label="deployment document")
    _outside_public_upload(source, public_upload_dir, label="deployment document")
    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_json_object_pairs,
                           parse_constant=_reject_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise FleetCLIError("private deployment document is invalid") from exc
    if not isinstance(value, dict):
        raise FleetCLIError("private deployment document must be a JSON object")
    role = value.get("role")
    if role == "worker":
        secret_field = "worker_join_code"
    elif role == "learner":
        learner_secret_fields = [name for name in ("publisher_bearer", "external_bearer") if name in value]
        secret_field = learner_secret_fields[0] if len(learner_secret_fields) == 1 else None
    else:
        secret_field = None
    if secret_field is None:
        raise FleetCLIError("private deployment document has an unsupported role")
    secret_ref = value.get(secret_field)
    if not isinstance(secret_ref, str) or not secret_ref.startswith("file:"):
        raise FleetCLIError(f"{secret_field} must use a private file: reference")
    secret_path = Path(secret_ref[5:]).expanduser()
    if not secret_path.is_absolute():
        secret_path = source.resolve().parent / secret_path
    secret_path = _private_regular_file(secret_path, label=f"{secret_field} file")
    _outside_public_upload(secret_path, public_upload_dir, label=f"{secret_field} file")
    try:
        secret = secret_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise FleetCLIError(f"{secret_field} file is invalid") from exc
    if not secret or any(character.isspace() for character in secret):
        raise FleetCLIError(f"{secret_field} file is invalid")
    value[secret_field] = secret
    try:
        return _ROLE_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        # Validation exceptions can include field input; never print them because
        # the deployment document contains private role credentials.
        raise FleetCLIError("private deployment document does not match the Fleet contract") from exc


def _write_secret(path: Path, value: str) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FleetCLIError(f"refusing to overwrite existing private file: {path}")
    try:
        atomic_write(path, (value + "\n").encode("ascii"), mode=0o600)
    except Exception as exc:
        raise FleetCLIError("could not write Fleet admin bearer") from exc


def _init(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).expanduser().absolute()
    if args.public_upload_dir is not None:
        public_root = args.public_upload_dir.expanduser().resolve()
        resolved_state = state_dir.resolve()
        if (resolved_state == public_root or resolved_state.is_relative_to(public_root)
                or public_root.is_relative_to(resolved_state)):
            raise FleetCLIError("Fleet state must be separate from the entire public upload tree")
    ensure_private_dir(state_dir)
    config_path = Path(args.output).expanduser().absolute()
    if not config_path.resolve().is_relative_to(state_dir.resolve()):
        raise FleetCLIError("Fleet settings must be stored inside the private state directory")
    token_path = state_dir / "admin.token"
    if token_path.exists():
        token_path = _private_regular_file(token_path, label="Fleet admin bearer")
        token = token_path.read_text(encoding="ascii").strip()
        if not token or any(c.isspace() for c in token):
            raise FleetCLIError("existing Fleet admin bearer is invalid")
    else:
        token = secrets.token_urlsafe(32)
        _write_secret(token_path, token)

    try:
        from .ssh import prepare_fleet_ssh
    except ImportError as exc:
        raise FleetCLIError("Fleet SSH setup support is not installed") from exc
    try:
        ssh = prepare_fleet_ssh(state_dir, relay_user=args.relay_user,
                                relay_host=args.relay_host, relay_port=args.relay_port)
    except Exception as exc:
        raise FleetCLIError("Fleet SSH identities could not be prepared") from exc

    if config_path.exists() or config_path.is_symlink():
        raise FleetCLIError(f"refusing to overwrite existing Fleet settings: {config_path}")
    config = {
        "state_dir": str(state_dir),
        "admin_token": "file:" + str(token_path),
        "ssh_admin_identity_file": str(ssh.admin_identity_file),
        "ssh_relay_user": args.relay_user,
        "ssh_relay_host_key": ssh.relay_host_key,
        "admin_ssh_public_key": ssh.admin_public_key,
        "public_origin": args.public_origin,
        "public_upload_dir": str(args.public_upload_dir.expanduser().absolute()) if args.public_upload_dir else None,
        "max_registered_nodes": args.max_registered_nodes,
        "max_pending_nodes": args.max_pending_nodes,
        "max_rejected_tombstones": args.max_rejected_tombstones,
        "registration_rate_per_minute": args.registration_rate_per_minute,
        "reverse_port_start": args.reverse_port_start,
        "reverse_port_end": args.reverse_port_end,
        "ssh_relay_host": args.relay_host,
        "ssh_relay_port": args.relay_port,
        "api_host": "127.0.0.1",
        "api_port": 18138,
    }
    try:
        FleetSettings.model_validate({**config, "admin_token": token}, strict=True)
        atomic_write(config_path, (_json(config) + "\n").encode("utf-8"), mode=0o600)
        loaded = FleetSettings.load(config_path)
        _registry(loaded)
    except Exception as exc:
        raise FleetCLIError("Fleet settings or initial registry could not be created") from exc
    _emit({"config": str(config_path), "admin_token_file": str(token_path),
           "ssh_admin_identity_file": str(ssh.admin_identity_file),
           "relay_host_key": ssh.relay_host_key,
           "admin_ssh_public_key": ssh.admin_public_key,
           "private_artifacts_dir": str(loaded.private_artifacts_dir)})
    return 0


def _remote_command(args: argparse.Namespace) -> int:
    config = _settings(args)
    registry = _registry(config)
    node = registry.get_node(args.node_id)
    if args.command == "logs":
        if not 1 <= args.lines <= 10000:
            raise FleetCLIError("--lines must be between 1 and 10000")
        command = ["journalctl", "--user", "--unit", "qwen-fleet-node.service", "--no-pager", f"--lines={args.lines}"]
    else:
        command = list(args.argv)
        if command and command[0] == "--":
            command = command[1:]
        if args.command == "exec" and not command:
            raise FleetCLIError("exec requires a remote command after --")
    try:
        from .ssh import run_admin_ssh

        result = run_admin_ssh(node, command or None, state_dir=Path(config.state_dir),
                               identity_file=Path(config.ssh_admin_identity_file),
                               capture_output=False, interactive=(args.command == "ssh"),
                               timeout=None if args.command == "ssh" else 120.0)
    except FleetCLIError:
        raise
    except Exception as exc:
        raise FleetCLIError("Fleet SSH operation failed") from exc
    return int(getattr(result, "returncode", 0))


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "init":
        return _init(args)
    config = _settings(args)
    registry = _registry(config)

    if args.command.startswith("operation-"):
        from .operation_jobs import job_status, operation_dir, start_job
        from .operation_plan import load_plan

        try:
            if args.command == "operation-status":
                result = job_status(config, args.operation_id)
            else:
                plan = (args.plan if args.command == "operation-start" else
                        operation_dir(config, args.operation_id) / "plan.json")
                load_plan(plan)
                result = start_job(config, plan, config_path=args.config,
                                   resume_operation_id=(args.operation_id if args.command == "operation-resume" else None))
        except (OSError, ValueError, RuntimeError) as exc:
            raise FleetCLIError("Fleet operation input or state is invalid; inspect the private recipe and receipts") from exc
        _emit(result)
        return 0

    if args.command == "serve":
        import uvicorn

        refresh_relay_authorized_keys(registry, config)
        uvicorn.run(create_fleet_app(config, registry), host=config.api_host, port=config.api_port,
                    ws="websockets-sansio", ws_max_size=65_536, ws_max_queue=1,
                    ws_per_message_deflate=False, access_log=False, log_config=None)
        return 0
    if args.command == "list":
        _emit({"nodes": [node.model_dump(mode="json") for node in registry.list_nodes()]})
        return 0
    if args.command == "inspect":
        _emit(registry.get_node(args.node_id).model_dump(mode="json"))
        return 0
    if args.command == "audit":
        _emit({"rejections": [item.model_dump(mode="json") for item in registry.list_rejection_audit()]})
        return 0
    if args.command == "approve":
        require_relay_setup(config)
        result = registry.approve_node(args.node_id)
        refresh_relay_authorized_keys(registry, config)
        _emit(result.model_dump(mode="json"))
        return 0
    if args.command == "reject":
        _emit(registry.reject_node(args.node_id).model_dump(mode="json"))
        return 0
    if args.command == "assign":
        require_relay_setup(config)
        if config.public_upload_dir is None:
            raise FleetCLIError("assign requires public_upload_dir settings to protect private role documents")
        deployment = _read_private_assignment(args.deployment_file, public_upload_dir=config.public_upload_dir)
        _emit(registry.assign_node(args.node_id, deployment).model_dump(mode="json"))
        return 0
    if args.command == "deactivate":
        _emit(registry.deactivate_node(args.node_id).model_dump(mode="json"))
        return 0
    if args.command == "refresh-management":
        _emit(registry.refresh_management(args.node_id).model_dump(mode="json"))
        return 0
    if args.command == "upgrade-nodes":
        from .upgrade import FleetUpgradeError, upgrade_nodes

        try:
            result = upgrade_nodes(registry, config, args.node)
        except FleetUpgradeError as exc:
            raise FleetCLIError(str(exc)) from exc
        _emit(result)
        return 0 if all(item.get("status") in {"upgraded", "already_current"}
                        for item in result["nodes"]) else 1
    if args.command in {"ssh", "exec", "logs"}:
        return _remote_command(args)
    if args.command == "publish-private-bundle":
        if config.public_origin is None or config.public_upload_dir is None:
            raise FleetCLIError("private bundle publication requires public_origin and public_upload_dir settings")
        try:
            store = PrivateArtifactStore(config.state_dir, public_upload_dir=config.public_upload_dir)
            published = store.publish(args.source_file)
            registry.publish_private_artifact(published.sha256, publisher_bearer=published.publisher_bearer,
                                              size_bytes=published.size_bytes)
            url = config.private_artifact_url(published.sha256)
        except (PrivateArtifactError, FleetRegistryError, ValueError) as exc:
            raise FleetCLIError(str(exc)) from exc
        _emit({"url": url, "sha256": published.sha256, "size_bytes": published.size_bytes,
               "publisher_bearer_file": str(published.publisher_bearer_file)})
        return 0
    raise FleetCLIError("unsupported Fleet command")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return _dispatch(args)
    except (FleetCLIError, FleetRegistryError) as exc:
        print(f"qwen-rl cluster fleet: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
