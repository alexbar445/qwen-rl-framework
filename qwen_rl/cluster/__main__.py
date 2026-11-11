"""Controller, site, node and publication commands for the LAN cluster."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import shlex
import stat
import sys

from .settings import load_settings_json

NINFER_ENGINE_BINARY = "/usr/local/bin/ninfer-serve"
NINFER_WORKER_PYTHON = sys.executable


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def parser():
    p = argparse.ArgumentParser(prog="qwen-rl cluster")
    sub = p.add_subparsers(dest="command", required=True)
    fleet = sub.add_parser("fleet", help="Manage generic Ubuntu nodes and central role assignments",
                           add_help=False)
    fleet.add_argument("fleet_args", nargs=argparse.REMAINDER)
    for name, port in (("controller", 8300), ("gateway", 8310), ("worker", 8241)):
        q = sub.add_parser(name)
        q.add_argument("--config", type=Path, required=True)
        q.add_argument("--host", default="127.0.0.1" if name == "controller" else "0.0.0.0")
        q.add_argument("--port", type=int, default=port)
        if name == "controller":
            q.add_argument("--ssl-certfile", type=Path)
            q.add_argument("--ssl-keyfile", type=Path)
    q = sub.add_parser("init-controller")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--state-dir", type=Path, required=True)
    q.add_argument("--bootstrap-defaults", type=Path, required=True)
    q.add_argument("--policy-upstream-url")
    q.add_argument("--policy-token-file", type=Path)
    q.add_argument("--relay-state-dir", type=Path)
    q = sub.add_parser("enrol")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--site", required=True)
    q.add_argument("--max-nodes", type=int, default=50)
    q.add_argument("--ttl", type=int, default=3600)
    q.add_argument("--role", choices=("gateway", "worker"), default="worker")
    q = sub.add_parser("nodes")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--site")
    q = sub.add_parser("publish-image")
    choice = q.add_mutually_exclusive_group(required=True)
    choice.add_argument("--image")
    choice.add_argument("--archive-descriptor", type=Path)
    q.add_argument("--store", type=Path, required=True)
    q.add_argument("--read-url", required=True)
    q.add_argument("--gateway")
    q.add_argument("--token-file", type=Path)
    q.add_argument("--descriptor-output", type=Path, required=True)
    q = sub.add_parser("bootstrap-bundle")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--public-url", required=True)
    q.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    q.add_argument("--assets-dir", type=Path)
    q.add_argument("--site", required=True)
    q.add_argument("--role", choices=("gateway", "worker"), default="worker")
    q = sub.add_parser("relay")
    q.add_argument("--registry-state-dir", type=Path, required=True)
    q.add_argument("--state-dir", type=Path, required=True)
    q.add_argument("--host", default="0.0.0.0")
    q.add_argument("--port", type=int, default=22220)
    q.add_argument("--user", required=True)
    q.add_argument("--serve", action="store_true")
    q = sub.add_parser("runtime-config")
    q.add_argument("--template", type=Path, required=True)
    q.add_argument("--inventory", type=Path)
    q.add_argument("--gateway", required=True)
    q.add_argument("--token-file", type=Path, required=True)
    q.add_argument("--site", required=True)
    q.add_argument("--expected-workers", type=int, default=50)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--environment-output", type=Path, required=True)
    q = sub.add_parser("site-config")
    q.add_argument("--model-descriptor", type=Path, required=True)
    image_source = q.add_mutually_exclusive_group(required=True)
    image_source.add_argument("--image-descriptor", type=Path)
    image_source.add_argument("--worker-release", type=Path)
    q.add_argument("--gateway-image-descriptor", type=Path)
    q.add_argument("--lineage", required=True)
    q.add_argument("--gateway-lan-ip", required=True)
    q.add_argument("--admin-ssh-public-key", type=Path, required=True)
    q.add_argument("--profile", type=Path, required=True)
    q.add_argument("--reverse-ssh", type=Path)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("ssh")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--site", required=True)
    q.add_argument("--node", required=True)
    q.add_argument("--identity", type=Path, required=True)
    q.add_argument("--gateway-port", type=int, required=True)
    q.add_argument("--user", default="root")
    q.add_argument("argv", nargs=argparse.REMAINDER)
    q = sub.add_parser("node")
    ns = q.add_subparsers(dest="node_command", required=True)
    j = ns.add_parser("join")
    j.add_argument("--controller", required=True)
    j.add_argument("--site", required=True)
    j.add_argument("--code-file", type=Path, required=True)
    j.add_argument("--role", choices=("gateway", "worker"), required=True)
    j.add_argument("--state-dir", type=Path, default=Path("/var/lib/qwen-cluster"))
    j.add_argument("--lan-host")
    j.add_argument("--gpu-uuid")
    j.add_argument("--ca-bundle", type=Path)
    j.add_argument("--install-host-packages", action="store_true")
    j.add_argument("--activate", action="store_true")
    j.add_argument("--no-training-ssh-state", dest="write_training_ssh_state",
                   action="store_false", default=True)
    j.add_argument("--container-uid", type=int)
    j.add_argument("--container-gid", type=int)
    j.add_argument("--owner-uid", type=int)
    j.add_argument("--owner-gid", type=int)
    j.add_argument("--project-root", type=Path, default=Path("/opt/qwen-rl"))
    j.add_argument("--dragonfly-source-archive", type=Path)
    for name in ("run", "status", "verify-model", "install-image"):
        j = ns.add_parser(name)
        j.add_argument("--config", type=Path, required=True)
    q = sub.add_parser("seed")
    q.add_argument("seed_args", nargs=argparse.REMAINDER)
    return p


def _controller_tls_paths(certfile: Path | None, keyfile: Path | None) -> tuple[str | None, str | None]:
    """Validate an optional Uvicorn TLS pair without following private-key symlinks."""
    if (certfile is None) != (keyfile is None):
        raise ValueError("Controller TLS requires both --ssl-certfile and --ssl-keyfile")
    if certfile is None or keyfile is None:
        return None, None
    resolved: list[str] = []
    for label, requested, private in (
        ("certificate", certfile, False), ("private key", keyfile, True),
    ):
        path = Path(requested).expanduser().absolute()
        try:
            info = path.lstat()
        except OSError as exc:
            raise ValueError(f"Controller TLS {label} path is unavailable") from exc
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or not os.access(path, os.R_OK):
            raise ValueError(f"Controller TLS {label} must be a readable regular file")
        if private and (info.st_uid != os.geteuid() or info.st_mode & 0o077):
            raise ValueError("Controller TLS private key must be account-owned and private")
        resolved.append(str(path.resolve(strict=True)))
    return resolved[0], resolved[1]


async def join(args):
    from .bootstrap import join_cluster
    from .host import install_host_packages
    from .server import read_token_file
    if args.install_host_packages:
        packages = ("docker.io", "docker-compose-v2", "openssh-client", "openssh-server", "ca-certificates", "gnupg")
        if args.role == "worker":
            packages += ("nvidia-container-toolkit",)
        install_host_packages(packages=packages, allow_package_changes=True)
    key_creator = None
    if not args.write_training_ssh_state:
        # Fleet's account-owned SSH runtime already has pinned AsyncSSH and
        # creates real Ed25519 keys without invoking target OpenSSH binaries.
        from .fleet.user_ssh import create_asyncssh_ed25519_keypair

        key_creator = create_asyncssh_ed25519_keypair
    config = await join_cluster(controller_url=args.controller, site_id=args.site, code=read_token_file(args.code_file),
        role=args.role, state_dir=args.state_dir, lan_host=args.lan_host, gpu_uuid=args.gpu_uuid,
        ca_bundle=str(args.ca_bundle) if args.ca_bundle else True,
        container_uid=args.container_uid, container_gid=args.container_gid,
        owner_uid=args.owner_uid, owner_gid=args.owner_gid,
        write_training_ssh_state=args.write_training_ssh_state, key_creator=key_creator)
    installation = None
    if args.activate:
        from .install import install_node_services
        installation = await install_node_services(config, project_root=args.project_root,
                              python_executable=Path(sys.executable),
                              source_archive=args.dragonfly_source_archive)
    emit({"node_id": config.node_id, "site_id": config.site_id, "role": config.role,
          "config": str(args.state_dir / "node.json"),
          "needs_driver": (installation or {}).get("needs_driver", config.needs_driver),
          "needs_host_packages": (installation or {}).get("needs_host_packages", False),
          "installation": installation})


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "fleet":
        from .fleet.cli import main as fleet_main

        return fleet_main(arguments[1:])
    args = parser().parse_args(arguments)
    if args.command in {"controller", "gateway", "worker"}:
        import uvicorn
        if args.command == "controller":
            ssl_certfile, ssl_keyfile = _controller_tls_paths(args.ssl_certfile, args.ssl_keyfile)
            from .server import create_controller_server
            app = create_controller_server(args.config)
        elif args.command == "gateway":
            from .gateway import create_gateway_app, GatewaySettings
            app = create_gateway_app(GatewaySettings.load(args.config))
        else:
            worker_config = load_settings_json(args.config)
            inference = worker_config.get("inference")
            if isinstance(inference, dict) and inference.get("backend") == "ninfer":
                from urllib.parse import urlsplit

                token_file = os.environ.get("ARC_ENV_TOKEN_FILE")
                if not token_file:
                    raise ValueError("NInfer cluster Worker requires ARC_ENV_TOKEN_FILE")
                ninfer_url = inference.get("ninfer_url")
                if not isinstance(ninfer_url, str) or not ninfer_url:
                    raise ValueError("NInfer cluster Worker requires inference.ninfer_url")
                try:
                    endpoint = urlsplit(ninfer_url)
                    engine_port = endpoint.port
                except ValueError as exc:
                    raise ValueError("NInfer cluster Worker has an invalid inference.ninfer_url") from exc
                if engine_port is None:
                    raise ValueError("NInfer cluster Worker inference.ninfer_url must include its port")
                from ..worker.ninfer_launcher import main as launch_ninfer_worker

                return launch_ninfer_worker([
                    "--worker-config", str(args.config.resolve()),
                    "--token-file", token_file,
                    "--engine-binary", NINFER_ENGINE_BINARY,
                    "--worker-python", NINFER_WORKER_PYTHON,
                    "--engine-host", "127.0.0.1",
                    "--engine-port", str(engine_port),
                    "--worker-host", args.host,
                    "--worker-port", str(args.port),
                ])
            from .server import create_worker_server
            app = create_worker_server(worker_config,
                                       token_file=os.environ.get("ARC_ENV_TOKEN_FILE"))
        uvicorn.run(
            app, host=args.host, port=args.port, access_log=False,
            **({"ssl_certfile": ssl_certfile, "ssl_keyfile": ssl_keyfile}
               if args.command == "controller" else {}),
        )
    elif args.command == "init-controller":
        from .identity import atomic_write
        from .controller import ControllerSettings
        if args.output.exists():
            raise ValueError("Controller configuration already exists")
        token_path = args.output.parent / "admin.token"
        if token_path.exists():
            raise ValueError("admin token path already exists")
        token = secrets.token_urlsafe(36)
        if bool(args.policy_upstream_url) != bool(args.policy_token_file):
            raise ValueError("policy origin requires both URL and private token file")
        value = {"state_dir": str(args.state_dir.resolve()), "admin_token": "file:" + str(token_path.resolve()),
                 "bootstrap_defaults": load_settings_json(args.bootstrap_defaults)}
        if args.relay_state_dir:
            value["relay_state_dir"] = str(args.relay_state_dir.resolve())
        if args.policy_upstream_url:
            if args.policy_token_file is None:
                raise ValueError("policy origin requires --policy-token-file")
            value.update(policy_upstream_url=args.policy_upstream_url,
                         policy_upstream_token="file:" + str(args.policy_token_file.resolve()))
        ControllerSettings.model_validate({**value, "admin_token": token,
            **({"policy_upstream_token": token} if args.policy_upstream_url else {})})
        if args.policy_token_file is not None:
            from .server import read_token_file
            read_token_file(args.policy_token_file)
        atomic_write(token_path, (token + "\n").encode())
        atomic_write(args.output, (json.dumps(value, indent=2) + "\n").encode())
        emit({"config": str(args.output), "admin_token_file": str(token_path)})
    elif args.command in {"enrol", "nodes"}:
        from .controller import ControllerSettings
        from .registry import Registry
        from .contracts import EnrolmentRequest
        cfg = ControllerSettings.load(args.config)
        store = Registry(cfg.state_dir, heartbeat_ttl_seconds=cfg.heartbeat_ttl_seconds)
        if args.command == "enrol":
            emit(store.create_enrolment(EnrolmentRequest(site_id=args.site, role=args.role,
                max_nodes=args.max_nodes, ttl_seconds=args.ttl)))
        else:
            rows = store.list_nodes()
            emit([row for row in rows if args.site is None or row["site_id"] == args.site])
    elif args.command == "publish-image":
        from .images import export_image_archive, ImageArchiveDescriptor, ExportedImageArchive
        from .artifacts import ContentAddressedStore
        from .identity import atomic_write
        from .publication import publish_blob
        from .server import read_token_file
        if bool(args.gateway) != bool(args.token_file):
            raise ValueError("publishing to a site requires both --gateway and --token-file")
        if args.archive_descriptor:
            descriptor = ImageArchiveDescriptor.model_validate(load_settings_json(args.archive_descriptor))
            if descriptor.url != args.read_url.rstrip("/") + "/v1/artifacts/" + descriptor.archive_sha256:
                raise ValueError("archive descriptor belongs to a different site origin")
            cached = ContentAddressedStore(args.store).resolve(descriptor.archive_sha256, expected_size=descriptor.size_bytes)
            result = ExportedImageArchive(descriptor=descriptor, path=cached.path)
        else:
            result = export_image_archive(args.image, args.store, artifact_base_url=args.read_url)
        receipt = result.descriptor.model_dump()
        atomic_write(args.descriptor_output, (json.dumps(receipt, indent=2) + "\n").encode())
        if args.gateway:
            receipt["publication"] = asyncio.run(publish_blob(result.path,
                sha256=result.descriptor.archive_sha256, size_bytes=result.descriptor.size_bytes,
                gateway_url=args.gateway, token=read_token_file(args.token_file)))
        emit(receipt)
    elif args.command == "bootstrap-bundle":
        import hashlib
        from .bundle import build_bootstrap_bundle, render_join_command
        from .identity import atomic_write
        from .controller import ControllerSettings
        from .settings import validate_credential_origin
        config = ControllerSettings.load(args.config)
        validate_credential_origin(args.public_url)
        store = Path(config.state_dir) / "bootstrap"
        store.mkdir(mode=0o700, parents=True, exist_ok=True)
        bundle = build_bootstrap_bundle(args.project_root, destination=store / ".source.tar.gz", assets_dir=args.assets_dir)
        target = store / bundle.sha256
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != bundle.sha256:
                raise ValueError("bootstrap content identity conflict")
            bundle.path.unlink()
        else:
            bundle.path.replace(target)
        installer = (args.project_root / "deploy/bootstrap/install.py").read_bytes()
        installer_sha = hashlib.sha256(installer).hexdigest()
        atomic_write(store / installer_sha, installer)
        origin = args.public_url.rstrip("/")
        join_script = render_join_command(installer_url=f"{origin}/v1/bootstrap/{installer_sha}",
            installer_sha256=installer_sha, bundle_url=f"{origin}/v1/bootstrap/{bundle.sha256}",
            bundle_sha256=bundle.sha256, controller_url=origin, site_id=args.site, role=args.role)
        starter_sha = hashlib.sha256(join_script.encode()).hexdigest()
        atomic_write(store / starter_sha, join_script.encode())
        starter_url = f"{origin}/v1/bootstrap/{starter_sha}"
        launcher = ("import hashlib,subprocess,sys,urllib.request; "
            "u=sys.argv[1]; r=urllib.request.urlopen(u,timeout=60); "
            "b=r.read(1048577); "
            "valid=r.geturl()==u and len(b)<=1048576 and hashlib.sha256(b).hexdigest()==u.rsplit('/',1)[1]; "
            "valid or sys.exit('Starter TLS/SHA verification failed'); "
            "subprocess.run(['bash','-c',b.decode()],check=True)")
        emit({"bundle_sha256": bundle.sha256, "installer_sha256": installer_sha,
              "size_bytes": bundle.size_bytes, "file_count": bundle.file_count,
              "starter_sha256": starter_sha,
              "start_command": shlex.join(["python3", "-c", launcher, starter_url]),
              "join_script": join_script})
    elif args.command == "relay":
        from .relay import prepare_central_relay
        setup = prepare_central_relay(registry_state_dir=args.registry_state_dir, state_dir=args.state_dir,
                                     listen_host=args.host, port=args.port, user=args.user)
        if args.serve:
            os.execv(str(setup.sshd_path), list(setup.start_argv))
        emit({"config": str(setup.config_path), "host_public_key": setup.host_public_key,
              "start_argv": setup.start_argv, "reverse_listen_ports": setup.reverse_listen_ports})
    elif args.command == "site-config":
        from .deployment import prepare_bootstrap_defaults
        from .identity import atomic_write
        defaults = prepare_bootstrap_defaults(load_settings_json(args.model_descriptor),
            load_settings_json(args.image_descriptor) if args.image_descriptor else None,
            worker_release=args.worker_release,
            gateway_image_descriptor=(load_settings_json(args.gateway_image_descriptor)
                                      if args.gateway_image_descriptor else None),
            lineage_id=args.lineage,
            gateway_lan_ip=args.gateway_lan_ip, admin_ssh_public_key=args.admin_ssh_public_key.read_text().strip(),
            inference_profile=load_settings_json(args.profile),
            reverse_ssh=load_settings_json(args.reverse_ssh) if args.reverse_ssh else None)
        atomic_write(args.output, (defaults.model_dump_json(indent=2, by_alias=True) + "\n").encode())
        emit({"bootstrap_defaults": str(args.output),
              "image_mode": "release" if defaults.worker_release is not None else defaults.worker_image_mode})
    elif args.command == "runtime-config":
        from .runtime_entry import materialize_runtime_config
        from .server import read_token_file
        from .identity import atomic_write
        import httpx
        bearer = read_token_file(args.token_file)
        if args.inventory:
            inventory = args.inventory
        else:
            from .settings import validate_credential_origin
            validate_credential_origin(args.gateway)
            with httpx.Client(timeout=30, follow_redirects=False) as client:
                reply = client.get(args.gateway.rstrip("/") + "/v1/workers", params={"readiness": "model"},
                                   headers={"Authorization": "Bearer " + bearer})
                reply.raise_for_status()
                inventory = reply.json()
        result = materialize_runtime_config(args.template, inventory, site_id=args.site,
            gateway_endpoint=args.gateway, gateway_bearer=bearer, expected_workers=args.expected_workers)
        atomic_write(args.output, result.yaml_text.encode())
        environment = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in result.environment.items()) + "\n"
        atomic_write(args.environment_output, environment.encode())
        emit({"config": str(args.output), "environment_file": str(args.environment_output),
              "worker_ids": result.worker_ids, "worker_count": result.worker_count})
    elif args.command == "node":
        if args.node_command == "join":
            asyncio.run(join(args))
        elif args.node_command == "run":
            from .node import NodeDaemon
            asyncio.run(NodeDaemon(args.config).run())
        elif args.node_command == "verify-model":
            from .node_config import load_node_config
            from .node import install_node_model_artifact
            config, credentials = load_node_config(args.config)
            emit(asyncio.run(install_node_model_artifact(config, worker_token=credentials["worker_token"])))
        elif args.node_command == "install-image":
            from .node_config import load_node_config
            from .node import install_node_worker_image
            config, credentials = load_node_config(args.config)
            emit(asyncio.run(install_node_worker_image(config, worker_token=credentials["worker_token"])))
        else:
            from .node_config import load_node_config
            config, _secrets = load_node_config(args.config)
            emit({"node_id": config.node_id, "site_id": config.site_id, "role": config.role,
                  "gpu_uuid": config.gpu_uuid, "needs_driver": config.needs_driver})
    elif args.command == "ssh":
        from .controller import ControllerSettings
        from .ssh import SSHInventory, run_ssh
        cfg = ControllerSettings.load(args.config)
        inv = SSHInventory.load(Path(cfg.state_dir), args.site)
        path = Path(cfg.state_dir) / "ssh" / "sites" / args.site / "client.conf"
        inv.write_config(path, central_identity_file=args.identity, gateway_remote_port=args.gateway_port, user=args.user)
        command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        if not command:
            raise ValueError("ssh requires an explicit command after --")
        result = run_ssh(inv, args.node, command, config_path=path)
        print(result.stdout, end="")
    elif args.command == "seed":
        from .dragonfly import _preseed_cli
        return _preseed_cli(args.seed_args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        # Our expected errors contain action/contract descriptions. Validation
        # payloads and arbitrary provider exceptions can contain credentials.
        safe_errors = {"BootstrapError", "NodeInstallError", "RuntimeEntryError", "DeploymentConfigError",
                       "HostError", "IdentityError", "NodeConfigurationError",
                       "BootstrapBundleError", "RelayError", "ImageArchiveError", "NodeDaemonError"}
        detail = str(exc) if type(exc).__module__.startswith("qwen_rl.cluster") and type(exc).__name__ in safe_errors else "check configuration and local service logs"
        print(f"qwen-rl cluster failed ({type(exc).__name__}): {detail}", file=sys.stderr)
        raise SystemExit(1)
