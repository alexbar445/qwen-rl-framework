"""Read-only dashboard CLI for selected deployment or offline history."""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from pathlib import Path

from .app import BUILD, Dashboard, serve
from .collect import Collector

DEFAULT_PORT = 8799


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-rl dashboard",
        description="Read-only dashboard for a selected Qwen RL run source",
    )
    parser.add_argument("--source", choices=("local", "runtime", "learner"),
                        help="deployment source (default: local)")
    parser.add_argument("--deployment", type=Path,
                        help="selected local deployment JSON, native Runtime YAML, or Learner profile")
    parser.add_argument("--instance-manifest", type=Path,
                        help="bind a runtime source to its adjacent instance gradient manifest")
    parser.add_argument("--remote-profile", type=Path,
                        help="fixed upstream source selector profile for a local dashboard proxy")
    parser.add_argument("--root",
                        help="offline outputs/<run-root> directory; disables live collection")
    parser.add_argument("--bepa-store", type=Path,
                        help="explicit offline BEPA store directory or state.sqlite3 file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--once", action="store_true",
                        help="print one machine-readable snapshot and exit (no server)")
    parser.add_argument("--no-live", action="store_true",
                        help="never call the selected runtime metadata endpoints")
    parser.add_argument("--open", action="store_true", help="open a browser window")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.root is not None:
        if (args.source is not None or args.deployment is not None or args.remote_profile is not None
                or args.instance_manifest is not None):
            parser.error("--root selects offline mode; omit --source, --deployment, --instance-manifest, and --remote-profile")
        source = "offline"
        live = False
    else:
        if args.bepa_store is not None:
            parser.error("--bepa-store is available only with --root")
        source = args.source or "local"
        if source == "local" and args.deployment is None and args.remote_profile is None:
            parser.error("local dashboard requires --deployment PATH; use --root PATH for offline history")
        if args.instance_manifest is not None and source != "runtime":
            parser.error("--instance-manifest requires --source runtime")
        if source == "learner" and args.deployment is None:
            parser.error("--source learner requires --deployment PROFILE")
        if source == "runtime" and args.deployment is None:
            parser.error("--source runtime requires --deployment CONFIG")
        live = not args.no_live

    collector = Collector(
        args.root, source=source, deployment=args.deployment,
        bepa_store=args.bepa_store, instance_manifest=args.instance_manifest, live=live,
    )
    if args.once:
        dashboard = Dashboard(collector, background=False)
        try:
            print(json.dumps(dashboard.workspace(), ensure_ascii=False, allow_nan=False,
                             default=str, indent=2))
        finally:
            dashboard.close()
        return 0
    if not collector.root.is_dir():
        print("warning: %s does not exist yet; pages will show no data" % collector.root)
    if args.remote_profile is not None:
        if args.once:
            parser.error("--remote-profile requires dashboard server mode")
        from .upstream import load_remote_profile

        remote_sources, default_source = load_remote_profile(args.remote_profile)
        server = serve(collector, args.host, args.port,
                       remote_sources=remote_sources, default_source=default_source)
    else:
        server = serve(collector, args.host, args.port)
    host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    url = "http://%s:%d/" % (host, args.port)
    print("qwen-rl dashboard %s | root=%s" % (BUILD, collector.root))
    print("open %s   (Ctrl-C to stop; this server only reads)" % url)
    if args.open:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\ndashboard stopped")
    finally:
        server.server_close()
    return 0
