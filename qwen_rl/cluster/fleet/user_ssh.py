"""User-owned AsyncSSH server and authenticated reverse-tunnel client.

Both commands run as the registering Unix account. The server exposes only a
loopback listener, executes a login shell as the already-running user, and the
client reaches the central OpenSSH relay through the authenticated Fleet WSS
byte stream.
"""
from __future__ import annotations

import argparse
import asyncio
import errno
import fcntl
import importlib.metadata
import ipaddress
import json
import os
import pty
import pwd
import re
import shutil
import signal
import socket
import ssl
import struct
import subprocess
import sys
import termios
from functools import cache
from pathlib import Path
from typing import Any

from ..identity import IdentityError, atomic_write, ensure_private_dir
from . import watchdog
from .contracts import UserAccount
from .ssh import (
    _MAX_FRAME_BYTES,
    _RELAY_ALIAS,
    _SAFE_USER,
    FleetSSHError,
    _controller_websocket_url,
    _ed25519_public,
    _node_ssh_port,
    _port,
    _private_regular,
    _public_regular,
    _read_token,
)

_ASYNCSSH_VERSION = "2.24.1"
_SERVER_PORT = 22022
_MAX_SSH_READ = 65_536
_TUNNEL_CLOSE_TIMEOUT_SECONDS = 5.0
_SERVER_CONFIG_KEYS = {"version", "account", "host_key", "authorized_keys", "listen_host", "listen_port"}
_TUNNEL_CONFIG_KEYS = {
    "version", "account", "controller_url", "ca_bundle", "bearer_file",
    "reverse_port", "relay_port", "relay_user", "relay_host_key",
    "relay_known_hosts", "tunnel_key", "listen_host", "listen_port", "local_host", "local_port",
}


@cache
def _asyncssh() -> Any:
    try:
        import asyncssh
        version = importlib.metadata.version("asyncssh")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise FleetSSHError("the Fleet user runtime requires AsyncSSH 2.24.1") from exc
    if version != _ASYNCSSH_VERSION:
        raise FleetSSHError(f"Fleet user SSH requires AsyncSSH {_ASYNCSSH_VERSION}")
    return asyncssh


def create_asyncssh_ed25519_keypair(private_path: Path, public_path: Path, comment: str) -> None:
    """Create an OpenSSH-format Ed25519 identity without calling ssh-keygen."""
    asyncssh = _asyncssh()
    private_path, public_path = Path(private_path), Path(public_path)
    ensure_private_dir(private_path.parent)
    if private_path.exists() or public_path.exists():
        raise FleetSSHError("refusing to replace an existing node SSH keypair")
    if not isinstance(comment, str) or not comment or any(c in comment for c in "\r\n\x00"):
        raise FleetSSHError("node SSH key comment must be a non-empty single line")
    try:
        key = asyncssh.generate_private_key("ssh-ed25519", comment=comment)
        atomic_write(private_path, key.export_private_key(format_name="openssh"), mode=0o600)
        atomic_write(public_path, key.export_public_key(format_name="openssh"), mode=0o644)
    except (IdentityError, OSError, ValueError, asyncssh.Error) as exc:
        raise FleetSSHError("could not create the user-owned Ed25519 SSH identity") from exc


def read_asyncssh_public_key(private_path: Path) -> str:
    """Derive a public key from an identity without a target OpenSSH binary."""
    asyncssh = _asyncssh()
    _private_regular(Path(private_path))
    try:
        key = asyncssh.read_private_key(str(Path(private_path).absolute()))
        return _ed25519_public(key.export_public_key(format_name="openssh").decode("ascii"),
                               "node SSH public key")
    except (OSError, UnicodeError, ValueError, asyncssh.Error) as exc:
        raise FleetSSHError("node SSH private identity is unavailable or malformed") from exc


def _read_private_json(path: Path) -> dict[str, Any]:
    _private_regular(path)
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FleetSSHError("Fleet user SSH configuration cannot be read") from exc
    if not isinstance(value, dict):
        raise FleetSSHError("Fleet user SSH configuration is malformed")
    return value


def _current_account(account: UserAccount | dict[str, Any]) -> pwd.struct_passwd:
    if isinstance(account, dict):
        try:
            account = UserAccount.model_validate(account, strict=True)
        except Exception as exc:  # pydantic reports data-dependent validation failures
            raise FleetSSHError("Fleet SSH account record is invalid") from exc
    try:
        current = pwd.getpwuid(os.geteuid())
    except (KeyError, OSError) as exc:
        raise FleetSSHError("Fleet SSH must run as an existing local account") from exc
    if (os.geteuid() == 0 or os.getuid() != os.geteuid() or os.getgid() != os.getegid()
            or account.uid != os.geteuid() or account.gid != current.pw_gid
            or account.username != current.pw_name or account.home != current.pw_dir):
        raise FleetSSHError("Fleet SSH account must match the current non-root local UID, GID, name, and home")
    return current


def _account_record(account: UserAccount) -> dict[str, Any]:
    return account.model_dump(mode="json")


def _trusted_environment(account: UserAccount, current: pwd.struct_passwd, term_type: str | None) -> dict[str, str]:
    # Never merge process.env: USER/HOME and every other session variable are
    # taken from this local user's service environment or the local passwd row.
    env = {key: value for key, value in os.environ.items()
           if isinstance(key, str) and isinstance(value, str) and "\x00" not in key + value}
    env.update({"HOME": current.pw_dir, "USER": current.pw_name, "LOGNAME": current.pw_name,
                "SHELL": current.pw_shell or "/bin/sh", "PWD": current.pw_dir})
    if term_type and len(term_type) <= 64 and all(c.isalnum() or c in "._+-" for c in term_type):
        env["TERM"] = term_type
    return env


def _signal_number(name: str) -> int | None:
    normalized = name if name.startswith("SIG") else "SIG" + name
    return getattr(signal, normalized, None)


def _signal_process_group(pid: int, number: int) -> None:
    try:
        os.killpg(pid, number)
    except ProcessLookupError:
        pass


async def _pump_input(process: Any, child: asyncio.subprocess.Process) -> None:
    asyncssh = _asyncssh()
    while True:
        try:
            data = await process.stdin.read(_MAX_SSH_READ)
        except asyncssh.TerminalSizeChanged:
            continue
        except asyncssh.SignalReceived as exc:
            number = _signal_number(exc.signal)
            if number is not None:
                _signal_process_group(child.pid, number)
            continue
        except asyncssh.BreakReceived:
            _signal_process_group(child.pid, signal.SIGINT)
            continue
        if not data:
            if child.stdin is not None:
                child.stdin.close()
                try:
                    await child.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        if child.stdin is not None:
            child.stdin.write(data)
            await child.stdin.drain()


async def _pump_stream(source: asyncio.StreamReader, target: Any) -> None:
    while data := await source.read(_MAX_SSH_READ):
        target.write(data)
        await target.drain()


async def _wait_for_child(child: asyncio.subprocess.Process,
                          tasks: list[asyncio.Task[None]]) -> int:
    exit_task = asyncio.create_task(child.wait())
    try:
        while not exit_task.done():
            done, _pending = await asyncio.wait([exit_task, *tasks], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is exit_task:
                    continue
                exception = task.exception()
                if exception is not None:
                    raise exception
            tasks[:] = [task for task in tasks if not task.done()]
        return await exit_task
    finally:
        if not exit_task.done():
            exit_task.cancel()
            await asyncio.gather(exit_task, return_exceptions=True)


async def _terminate_child(child: asyncio.subprocess.Process) -> None:
    if child.returncode is not None:
        return
    _signal_process_group(child.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(child.wait(), timeout=2)
    except asyncio.TimeoutError:
        _signal_process_group(child.pid, signal.SIGKILL)
        await child.wait()


def _report_exit(process: Any, returncode: int) -> None:
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name.removeprefix("SIG")
        except ValueError:
            process.exit(128 - returncode)
        else:
            process.exit_with_signal(name)
    else:
        process.exit(min(returncode, 255))


async def _run_piped_process(process: Any, command: list[str], account: UserAccount,
                             current: pwd.struct_passwd) -> None:
    env = _trusted_environment(account, current, None)
    child = await asyncio.create_subprocess_exec(
        *command, cwd=current.pw_dir, env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    assert child.stdout is not None and child.stderr is not None
    tasks = [
        asyncio.create_task(_pump_input(process, child)),
        asyncio.create_task(_pump_stream(child.stdout, process.stdout)),
        asyncio.create_task(_pump_stream(child.stderr, process.stderr)),
    ]
    try:
        returncode = await _wait_for_child(child, tasks)
        # The child has closed both output pipes. Let the two output pumps send
        # their final buffered bytes before the SSH exit status closes the channel.
        await asyncio.gather(*tasks[1:], return_exceptions=False)
        _report_exit(process, returncode)
    except asyncio.CancelledError:
        await _terminate_child(child)
        raise
    except Exception:
        await _terminate_child(child)
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _read_fd(fd: int, limit: int) -> bytes:
    loop = asyncio.get_running_loop()
    ready: asyncio.Future[bytes] = loop.create_future()

    def readable() -> None:
        if ready.done():
            return
        try:
            ready.set_result(os.read(fd, limit))
        except BlockingIOError:
            return
        except OSError as exc:
            if exc.errno == errno.EIO:  # Linux PTY master reports EIO after the slave closes.
                ready.set_result(b"")
            else:
                ready.set_exception(exc)

    loop.add_reader(fd, readable)
    try:
        return await ready
    finally:
        loop.remove_reader(fd)


async def _write_fd(fd: int, payload: bytes) -> None:
    loop = asyncio.get_running_loop()
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
            view = view[written:]
        except BlockingIOError:
            ready: asyncio.Future[None] = loop.create_future()

            def writable(target: asyncio.Future[None] = ready) -> None:
                if not target.done():
                    target.set_result(None)

            loop.add_writer(fd, writable)
            try:
                await ready
            finally:
                loop.remove_writer(fd)


def _set_pty_size(fd: int, width: int, height: int, pixwidth: int, pixheight: int) -> None:
    columns, rows = max(1, width), max(1, height)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, pixheight, pixwidth))


def _apply_pty_modes(asyncssh: Any, fd: int, modes: dict[int, int]) -> None:
    """Apply RFC 4254 terminal modes to the native PTY before exec."""
    if not modes:
        return
    attrs = termios.tcgetattr(fd)
    control_chars = {
        "VINTR": "VINTR", "VQUIT": "VQUIT", "VERASE": "VERASE", "VKILL": "VKILL",
        "VEOF": "VEOF", "VEOL": "VEOL", "VEOL2": "VEOL2", "VSTART": "VSTART",
        "VSTOP": "VSTOP", "VSUSP": "VSUSP", "VDSUSP": "VDSUSP", "VREPRINT": "VREPRINT",
        "VWERASE": "VWERASE", "VLNEXT": "VLNEXT", "VFLUSH": "VFLUSH", "VSWTCH": "VSWTCH",
        "VSTATUS": "VSTATUS", "VDISCARD": "VDISCARD",
    }
    for option, name in control_chars.items():
        opcode = getattr(asyncssh, f"PTY_{option}", None)
        index = getattr(termios, name, None)
        if opcode in modes and index is not None:
            attrs[6][index] = bytes((modes[opcode] & 0xFF,))

    flag_groups = (
        (0, ("IGNPAR", "PARMRK", "INPCK", "ISTRIP", "INLCR", "IGNCR", "ICRNL", "IUCLC",
             "IXON", "IXANY", "IXOFF", "IMAXBEL", "IUTF8")),
        (1, ("OPOST", "OLCUC", "ONLCR", "OCRNL", "ONOCR", "ONLRET")),
        (2, ("PARENB", "PARODD")),
        (3, ("ISIG", "ICANON", "XCASE", "ECHO", "ECHOE", "ECHOK", "ECHONL", "NOFLSH",
             "TOSTOP", "IEXTEN", "ECHOCTL", "ECHOKE", "PENDIN")),
    )
    for attr_index, names in flag_groups:
        for name in names:
            opcode = getattr(asyncssh, f"PTY_{name}", None)
            flag = getattr(termios, name, None)
            if opcode not in modes or flag is None:
                continue
            if modes[opcode]:
                attrs[attr_index] |= flag
            else:
                attrs[attr_index] &= ~flag

    cs7 = getattr(asyncssh, "PTY_CS7", None)
    cs8 = getattr(asyncssh, "PTY_CS8", None)
    if modes.get(cs7):
        attrs[2] = (attrs[2] & ~termios.CSIZE) | termios.CS7
    elif modes.get(cs8):
        attrs[2] = (attrs[2] & ~termios.CSIZE) | termios.CS8
    for option, attr_index in (("ISPEED", 4), ("OSPEED", 5)):
        opcode = getattr(asyncssh, f"PTY_OP_{option}", None)
        speed = modes.get(opcode) if opcode is not None else None
        if speed is not None:
            baud = getattr(termios, f"B{speed}", None)
            if baud is not None:
                attrs[attr_index] = baud
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


async def _pump_pty_output(process: Any, master_fd: int) -> None:
    while data := await _read_fd(master_fd, _MAX_SSH_READ):
        process.stdout.write(data)
        await process.stdout.drain()


async def _wait_for_pty_child(child: subprocess.Popen[bytes]) -> int:
    """Poll and reap the PTY child without blocking the SSH event loop."""
    while (returncode := child.poll()) is None:
        await asyncio.sleep(0.02)
    return returncode


async def _drain_pty_output(process: Any, master_fd: int) -> None:
    """Flush bytes already buffered after the foreground process exits."""
    while True:
        try:
            data = os.read(master_fd, _MAX_SSH_READ)
        except BlockingIOError:
            return
        except OSError as exc:
            if exc.errno == errno.EIO:
                return
            raise
        if not data:
            return
        process.stdout.write(data)
        await process.stdout.drain()


async def _pump_pty_input(process: Any, master_fd: int, child_pid: int) -> None:
    asyncssh = _asyncssh()
    while True:
        try:
            data = await process.stdin.read(_MAX_SSH_READ)
        except asyncssh.TerminalSizeChanged as exc:
            _set_pty_size(master_fd, exc.width, exc.height, exc.pixwidth, exc.pixheight)
            _signal_process_group(child_pid, signal.SIGWINCH)
            continue
        except asyncssh.SignalReceived as exc:
            number = _signal_number(exc.signal)
            if number is not None:
                _signal_process_group(child_pid, number)
            continue
        except asyncssh.BreakReceived:
            _signal_process_group(child_pid, signal.SIGINT)
            continue
        if not data:
            # RFC 4254 has no half-close for a PTY. VEOF preserves the usual
            # terminal behavior for remote commands which read until EOF.
            try:
                attrs = termios.tcgetattr(master_fd)
                eof_char = attrs[6][termios.VEOF]
                if isinstance(eof_char, int):
                    eof_char = bytes((eof_char,))
                if eof_char and eof_char != b"\x00":
                    await _write_fd(master_fd, eof_char)
            except OSError:
                pass
            return
        try:
            await _write_fd(master_fd, data)
        except OSError as exc:
            if exc.errno in {errno.EIO, errno.EBADF, errno.EPIPE}:
                return
            raise


async def _run_pty_process(process: Any, command: list[str], account: UserAccount,
                           current: pwd.struct_passwd) -> None:
    env = _trusted_environment(account, current, process.term_type)
    master_fd, slave_fd = pty.openpty()
    width, height, pixwidth, pixheight = process.term_size
    _set_pty_size(slave_fd, width or 80, height or 24, pixwidth, pixheight)
    _apply_pty_modes(_asyncssh(), slave_fd, process.term_modes)

    setsid = shutil.which("setsid") or next(
        (str(candidate) for candidate in (Path("/usr/bin/setsid"), Path("/bin/setsid"))
         if candidate.is_file() and os.access(candidate, os.X_OK)),
        None,
    )
    if setsid is None:
        os.close(master_fd)
        os.close(slave_fd)
        raise FleetSSHError("PTY sessions require the host setsid utility")
    try:
        # Popen.poll() below provides cancellable, nonblocking wait/reap behavior
        # for the setsid --wait wrapper used to attach a controlling terminal.
        child = subprocess.Popen(  # noqa: ASYNC220
            [setsid, "--ctty", "--wait", "--", *command],
            cwd=current.pw_dir, env=env, stdin=slave_fd, stdout=slave_fd,
            stderr=slave_fd, close_fds=True,
        )
    except (OSError, subprocess.SubprocessError):
        os.close(master_fd)
        raise
    finally:
        try:
            os.close(slave_fd)
        except OSError:
            pass
    os.set_blocking(master_fd, False)
    try:
        process.set_redirect_pid(child.pid)
    except AttributeError:
        pass
    input_task = asyncio.create_task(_pump_pty_input(process, master_fd, child.pid))
    output_task = asyncio.create_task(_pump_pty_output(process, master_fd))
    wait_task = asyncio.create_task(_wait_for_pty_child(child))
    tasks = [input_task, output_task, wait_task]
    try:
        while not wait_task.done():
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is wait_task:
                    continue
                exception = task.exception()
                if exception is not None:
                    raise exception
            tasks[:] = [task for task in tasks if not task.done()]
        returncode = await wait_task
        if not output_task.done():
            output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)
        await process.stdout.drain()
        await _drain_pty_output(process, master_fd)
        _report_exit(process, returncode)
    except asyncio.CancelledError:
        _signal_process_group(child.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=2)
        except asyncio.TimeoutError:
            _signal_process_group(child.pid, signal.SIGKILL)
            await asyncio.shield(wait_task)
        raise
    except Exception:
        _signal_process_group(child.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=2)
        except asyncio.TimeoutError:
            _signal_process_group(child.pid, signal.SIGKILL)
            await asyncio.shield(wait_task)
        raise
    finally:
        for task in (input_task, output_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(input_task, output_task, return_exceptions=True)
        os.close(master_fd)


async def _handle_process(process: Any, account: UserAccount) -> None:
    current = _current_account(account)
    # A client may request arbitrary environment entries. They are deliberately
    # ignored, including USER and HOME, so they cannot change execution identity.
    process.channel.set_encoding(None)
    shell = current.pw_shell or "/bin/sh"
    argv = [shell, "-l"] if process.command is None else [shell, "-c", process.command]
    # AsyncSSH returns an empty terminal type when a client requested a PTY
    # with TERM unset. None is the only value which means no PTY request.
    if process.term_type is not None:
        await _run_pty_process(process, argv, account, current)
    else:
        await _run_piped_process(process, argv, account, current)


def _host_ssh_server(account: UserAccount, authorized_key: str) -> Any:
    asyncssh = _asyncssh()
    expected = _ed25519_public(authorized_key, "Fleet admin public key")

    class FleetUserSSHServer(asyncssh.SSHServer):
        def begin_auth(self, username: str) -> bool:
            return True

        def public_key_auth_supported(self) -> bool:
            return True

        def password_auth_supported(self) -> bool:
            return False

        def kbdint_auth_supported(self) -> bool:
            return False

        def host_based_auth_supported(self) -> bool:
            return False

        def validate_public_key(self, username: str, key: Any) -> bool:
            if username != account.username:
                return False
            try:
                offered = key.export_public_key(format_name="openssh").decode("ascii")
                return _ed25519_public(offered, "SSH client public key") == expected
            except (UnicodeError, FleetSSHError):
                return False

        def connection_requested(self, dest_host: str, dest_port: int,
                                 orig_host: str, orig_port: int) -> bool:
            try:
                address = ipaddress.ip_address(dest_host)
            except ValueError:
                return False
            return address.is_loopback and type(dest_port) is int and 1 <= dest_port <= 65535

    return FleetUserSSHServer


def _load_server_config(config_path: Path) -> tuple[dict[str, Any], UserAccount, pwd.struct_passwd]:
    config = _read_private_json(config_path)
    if set(config) != _SERVER_CONFIG_KEYS or config.get("version") != 1:
        raise FleetSSHError("Fleet user SSH server configuration has an unsupported shape")
    try:
        account = UserAccount.model_validate(config["account"], strict=True)
    except Exception as exc:
        raise FleetSSHError("Fleet user SSH account record is invalid") from exc
    current = _current_account(account)
    if config.get("listen_host") != "127.0.0.1" or config.get("listen_port") != _SERVER_PORT:
        raise FleetSSHError("Fleet user SSH server must use loopback port 22022")
    host_key = Path(config["host_key"])
    authorized_keys = Path(config["authorized_keys"])
    _private_regular(host_key)
    _private_regular(authorized_keys)
    return config, account, current


async def serve_user_ssh(config_path: Path) -> None:
    asyncssh = _asyncssh()
    config, account, _current = _load_server_config(config_path)
    try:
        authorized_lines = Path(config["authorized_keys"]).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise FleetSSHError("Fleet admin authorized key cannot be read") from exc
    if len(authorized_lines) != 1 or not authorized_lines[0].strip():
        raise FleetSSHError("Fleet admin authorized key file must contain exactly one key")
    authorized = _ed25519_public(authorized_lines[0].strip(), "Fleet admin public key")
    host_key = Path(config["host_key"])
    server = await asyncssh.listen(
        config["listen_host"], config["listen_port"],
        server_factory=_host_ssh_server(account, authorized),
        server_host_keys=[str(host_key)],
        process_factory=lambda process: _handle_process(process, account),
        sftp_factory=True,
        allow_scp=False,
        allow_pty=True,
        agent_forwarding=False,
        x11_forwarding=False,
        host_based_auth=False,
        password_auth=False,
        kbdint_auth=False,
        encoding=None,
        line_editor=False,
        config=[],
    )
    try:
        await server.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


def _load_tunnel_config(config_path: Path) -> tuple[dict[str, Any], UserAccount, pwd.struct_passwd]:
    config = _read_private_json(config_path)
    if set(config) != _TUNNEL_CONFIG_KEYS or config.get("version") != 1:
        raise FleetSSHError("Fleet reverse tunnel configuration has an unsupported shape")
    try:
        account = UserAccount.model_validate(config["account"], strict=True)
    except Exception as exc:
        raise FleetSSHError("Fleet reverse tunnel account record is invalid") from exc
    current = _current_account(account)
    if (config.get("listen_host") != "127.0.0.1" or config.get("listen_port") != _port(
            config.get("reverse_port"), "allocated reverse listener")
            or config.get("local_host") != "127.0.0.1"
            or config.get("local_port") != _node_ssh_port(config.get("local_port"))):
        raise FleetSSHError("Fleet reverse tunnel must use its assigned loopback listener and node SSH port")
    _controller_websocket_url(config.get("controller_url"))
    _port(config.get("relay_port"), "Fleet central relay port")
    if (not isinstance(config.get("relay_user"), str) or not _SAFE_USER.fullmatch(config["relay_user"])
            or config["relay_user"] == "root"):
        raise FleetSSHError("Fleet relay SSH user is invalid")
    _ed25519_public(config.get("relay_host_key"), "central relay host key")
    _private_regular(Path(config["tunnel_key"]))
    _public_regular(Path(config["ca_bundle"]))
    _private_regular(Path(config["bearer_file"]))
    known_hosts = Path(config["relay_known_hosts"])
    _private_regular(known_hosts)
    try:
        lines = [line.strip() for line in known_hosts.read_text(encoding="ascii").splitlines() if line.strip()]
    except (OSError, UnicodeError) as exc:
        raise FleetSSHError("pinned central relay key cannot be read") from exc
    expected = _ed25519_public(config["relay_host_key"], "central relay host key")
    if lines != [f"{_RELAY_ALIAS} {expected}"]:
        raise FleetSSHError("pinned central relay key file does not match the approved host key")
    return config, account, current


async def _websocket_socket_bridge(ws: Any, relay_socket: socket.socket) -> None:
    loop = asyncio.get_running_loop()

    async def wss_to_ssh() -> None:
        while True:
            payload = await ws.recv()
            if not isinstance(payload, bytes):
                raise FleetSSHError("Fleet SSH WebSocket returned a text frame")
            if len(payload) > _MAX_FRAME_BYTES:
                raise FleetSSHError("Fleet SSH WebSocket frame exceeded the configured limit")
            await loop.sock_sendall(relay_socket, payload)
            watchdog.notify_watchdog()

    async def ssh_to_wss() -> None:
        while payload := await loop.sock_recv(relay_socket, _MAX_FRAME_BYTES):
            await ws.send(payload)

    tasks = [asyncio.create_task(wss_to_ssh()), asyncio.create_task(ssh_to_wss())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if task.cancelled():
                continue
            exception = task.exception()
            if exception is not None:
                raise exception
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


_WS_CLOSE_CODES = {
    1000: "normal", 1001: "going_away", 1002: "protocol_error",
    1003: "unsupported_data", 1005: "no_status", 1006: "abnormal",
    1007: "invalid_payload", 1008: "policy", 1009: "message_too_big",
    1011: "internal_error", 1012: "service_restart", 1013: "try_again_later",
    1015: "tls_failure",
}
_WS_CLOSE_REASONS = {
    "normal closure": "normal", "going away": "going_away",
    "protocol error": "protocol_error", "unsupported data": "unsupported_data",
    "invalid frame payload data": "invalid_payload", "policy violation": "policy",
    "message too big": "message_too_big", "internal error": "internal_error",
    "service restart": "service_restart", "try again later": "try_again_later",
    "ping timeout": "ping_timeout", "keepalive ping timeout": "ping_timeout",
}
_ASYNCSSH_REASONS = {
    "connection lost": "connection_lost", "connection reset": "connection_reset",
    "channel open failed": "channel_open_failed", "key exchange failed": "key_exchange_failed",
    "authentication failed": "authentication_failed", "operation timed out": "timeout",
}
_TLS_LIBRARIES = {
    "SSL routines": "openssl_ssl", "BIO routines": "openssl_bio",
    "PEM routines": "openssl_pem", "common libcrypto routines": "openssl_crypto",
}


def _tunnel_exception_chain(exc: BaseException) -> list[dict[str, Any]]:
    """Return safe, bounded exception metadata without free-form messages."""
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 12:
        seen.add(id(current))
        cls = type(current)
        item: dict[str, Any] = {"type": f"{cls.__module__}.{cls.__name__}"}
        if isinstance(current, ssl.SSLError):
            reason = getattr(current, "reason", None)
            item["tls_reason_code"] = reason if isinstance(reason, str) and re.fullmatch(r"[A-Z0-9_]{1,64}", reason) else "unlisted"
            item["tls_library_code"] = _TLS_LIBRARIES.get(getattr(current, "library", None), "unlisted")
        if isinstance(current, OSError):
            error_code = errno.errorcode.get(current.errno) if isinstance(current.errno, int) else None
            allowed = {"ECONNRESET", "EPIPE", "ETIMEDOUT", "ECONNREFUSED", "ENETUNREACH", "EHOSTUNREACH", "ECONNABORTED"}
            item["os_errno_code"] = error_code if error_code in allowed else ("other" if current.errno is not None else "none")
        if cls.__module__.startswith("asyncssh."):
            code = getattr(current, "code", None)
            item["asyncssh_code"] = code if type(code) is int and 0 <= code <= 255 else ("other" if code is not None else "none")
            reason = getattr(current, "reason", None)
            item["asyncssh_reason_code"] = _ASYNCSSH_REASONS.get(reason.strip().lower(), "unlisted") if isinstance(reason, str) else "none"
        for key, label in (("rcvd", "websocket_received_close"), ("sent", "websocket_sent_close")):
            close = getattr(current, key, None)
            if close is not None:
                code = getattr(close, "code", None)
                reason = getattr(close, "reason", None)
                item[label] = {
                    "code": code if type(code) is int and code in _WS_CLOSE_CODES else "other",
                    "code_name": _WS_CLOSE_CODES.get(code, "other"),
                    "reason_code": _WS_CLOSE_REASONS.get(reason.strip().lower(), "unlisted") if isinstance(reason, str) and reason.strip() else "empty",
                }
        chain.append(item)
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return chain


def _log_tunnel_exception(exc: BaseException) -> None:
    print("Fleet reverse tunnel exception_chain=" + json.dumps(_tunnel_exception_chain(exc), sort_keys=True),
          file=sys.stderr, flush=True)


def _consume_task(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        try:
            task.exception()
        except BaseException:
            pass


async def _cleanup_reverse_tunnel(
    listener: Any, connection: Any, bridge_task: asyncio.Task[None] | None,
    closed_task: asyncio.Task[None] | None, ssh_socket: socket.socket | None,
    relay_socket: socket.socket | None,
) -> None:
    """Tear down a failed tunnel without waiting on a blocked SSH flush."""
    if listener is not None:
        try:
            listener.close()
        except Exception:
            pass
    if bridge_task is not None and not bridge_task.done():
        bridge_task.cancel()
    for sock in (ssh_socket, relay_socket):
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
    if connection is not None:
        try:
            connection.abort()
        except Exception:
            pass
        if closed_task is None:
            try:
                closed_task = asyncio.create_task(connection.wait_closed())
            except Exception:
                closed_task = None

    wait_tasks: set[asyncio.Task[Any]] = set()
    if closed_task is not None and not closed_task.done():
        wait_tasks.add(closed_task)
    if listener is not None:
        wait_closed = getattr(listener, "wait_closed", None)
        if callable(wait_closed):
            try:
                wait_tasks.add(asyncio.create_task(wait_closed()))
            except Exception:
                pass
    if bridge_task is not None and not bridge_task.done():
        wait_tasks.add(bridge_task)
    if wait_tasks:
        done, pending = await asyncio.wait(wait_tasks, timeout=_TUNNEL_CLOSE_TIMEOUT_SECONDS)
        for task in done:
            _consume_task(task)
        for task in pending:
            task.cancel()
            task.add_done_callback(_consume_task)


async def _open_loopback_bridge_pair() -> tuple[socket.socket, socket.socket]:
    """Create a loopback pair with the TCP peername tuple AsyncSSH expects."""
    loop = asyncio.get_running_loop()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setblocking(False)
    client.setblocking(False)
    accepted: socket.socket | None = None
    accept_task: asyncio.Task[tuple[socket.socket, tuple[Any, ...]]] | None = None
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        accept_task = asyncio.create_task(loop.sock_accept(listener))
        await loop.sock_connect(client, listener.getsockname())
        accepted, _address = await accept_task
        accepted.setblocking(False)
        if (accepted.getpeername() != client.getsockname()
                or client.getpeername() != accepted.getsockname()):
            raise FleetSSHError("Fleet SSH bridge sockets are not the same connected loopback pair")
        expected = ipaddress.IPv4Address("127.0.0.1")
        for endpoint in (client, accepted):
            if (endpoint.family != socket.AF_INET
                    or ipaddress.ip_address(endpoint.getsockname()[0]) != expected
                    or ipaddress.ip_address(endpoint.getpeername()[0]) != expected):
                raise FleetSSHError("Fleet SSH bridge sockets must remain on IPv4 loopback")
        return client, accepted
    except BaseException:
        if accept_task is not None:
            if not accept_task.done():
                accept_task.cancel()
            result = (await asyncio.gather(accept_task, return_exceptions=True))[0]
            if accepted is None and not isinstance(result, BaseException):
                accepted = result[0]
        client.close()
        if accepted is not None:
            accepted.close()
        raise
    finally:
        listener.close()


async def run_node_reverse_tunnel(config_path: Path) -> None:
    asyncssh = _asyncssh()
    import websockets

    config, _account, _current = _load_tunnel_config(config_path)
    url = _controller_websocket_url(config["controller_url"])
    token = _read_token(Path(config["bearer_file"]))
    tls = ssl.create_default_context(cafile=str(Path(config["ca_bundle"]).absolute()))
    tunnel_key = str(Path(config["tunnel_key"]).absolute())
    known_hosts = str(Path(config["relay_known_hosts"]).absolute())
    relay_socket: socket.socket | None = None
    ssh_socket: socket.socket | None = None
    connection: Any = None
    listener: Any = None
    bridge_task: asyncio.Task[None] | None = None
    closed_task: asyncio.Task[None] | None = None
    failure: BaseException | None = None
    try:
        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {token}"},
            ssl=tls,
            proxy=None,
            compression=None,
            max_size=_MAX_FRAME_BYTES,
            max_queue=1,
            write_limit=_MAX_FRAME_BYTES,
            open_timeout=15,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as ws:
            ssh_socket, relay_socket = await _open_loopback_bridge_pair()
            bridge_task = asyncio.create_task(_websocket_socket_bridge(ws, relay_socket))
            connection = await asyncssh.connect(
                sock=ssh_socket,
                username=config["relay_user"],
                client_keys=[tunnel_key],
                agent_path=None,
                known_hosts=known_hosts,
                host_key_alias=_RELAY_ALIAS,
                password_auth=False,
                kbdint_auth=False,
                host_based_auth=False,
                gss_auth=False,
                public_key_auth=True,
                preferred_auth="publickey",
                keepalive_interval=15,
                keepalive_count_max=3,
                connect_timeout=15,
                config=[],
            )
            # AsyncSSH now owns ssh_socket; the authorized_keys options on the
            # central relay independently fence this to the assigned listener.
            ssh_socket = None
            listener = await connection.forward_remote_port(
                "127.0.0.1", config["reverse_port"], "127.0.0.1", _SERVER_PORT,
            )
            if listener is None:
                raise FleetSSHError("central relay refused the assigned loopback SSH listener")
            watchdog.notify_ready()
            watchdog.notify_watchdog()
            closed_task = asyncio.create_task(connection.wait_closed())
            done, _pending = await asyncio.wait(
                [bridge_task, closed_task], return_when=asyncio.FIRST_COMPLETED,
            )
            if bridge_task in done and not bridge_task.cancelled():
                exception = bridge_task.exception()
                if exception is not None:
                    raise exception
            if closed_task in done and not closed_task.cancelled():
                exception = closed_task.exception()
                if exception is not None:
                    raise exception
            raise FleetSSHError("Fleet reverse tunnel closed and needs to reconnect")
    except FleetSSHError as exc:
        failure = exc
        _log_tunnel_exception(exc)
        raise
    except asyncio.CancelledError as exc:
        failure = exc
        raise
    except Exception as exc:
        _log_tunnel_exception(exc)
        failure = FleetSSHError("Fleet authenticated reverse SSH tunnel failed")
        raise failure from exc
    finally:
        try:
            await _cleanup_reverse_tunnel(
                listener, connection, bridge_task, closed_task, ssh_socket, relay_socket,
            )
        except BaseException:
            if failure is None:
                raise


def _cli() -> int:
    parser = argparse.ArgumentParser(prog="python -m qwen_rl.cluster.fleet.user_ssh")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("serve", "Run the user-owned loopback SSH server"),
                            ("tunnel", "Run the authenticated Fleet reverse tunnel")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "serve":
            asyncio.run(serve_user_ssh(args.config))
        elif args.command == "tunnel":
            asyncio.run(run_node_reverse_tunnel(args.config))
        return 0
    except (FleetSSHError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
