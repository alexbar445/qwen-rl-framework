"""JSON settings with explicit, field-scoped private secret-file references."""
from __future__ import annotations

import json
import ipaddress
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def validate_credential_origin(value: str) -> str:
    """Public credential-bearing services require TLS; plaintext is LAN-only."""
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ValueError("credential destination must be a valid HTTP(S) URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ValueError("credential destination must be a valid HTTP(S) URL") from exc
    if (parsed.scheme not in {"http", "https"} or not hostname
            or parsed.username is not None or parsed.password is not None):
        raise ValueError("credential destination must be an HTTP(S) origin without URL credentials")
    if parsed.scheme == "http":
        host = hostname.lower().rstrip(".")
        try:
            address = ipaddress.ip_address(host)
            local = address.is_private or address.is_loopback or address.is_link_local
        except ValueError:
            local = host == "localhost" or "." not in host or host.endswith((
                ".localhost", ".local", ".test", ".lan", ".internal",
                ".home.arpa", ".example", ".invalid",
            ))
        if not local:
            raise ValueError("public credential-bearing destinations require HTTPS")
    return value


def load_settings_json(path: str | Path, *, secret_fields: tuple[str, ...] = ()) -> dict[str, Any]:
    source = Path(path)
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def constant(_value):
        raise ValueError("non-finite JSON value")

    value = json.loads(source.read_text(), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("cluster settings must be a JSON object")
    for field in secret_fields:
        reference = value.get(field)
        if not isinstance(reference, str) or not reference.startswith("file:"):
            continue
        secret_path = Path(reference[5:])
        if not secret_path.is_absolute():
            secret_path = source.resolve().parent / secret_path
        metadata = secret_path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError(f"{field} secret must be a private regular file (0400/0600)")
        secret = secret_path.read_text().strip()
        if not secret or "\n" in secret or "\r" in secret:
            raise ValueError(f"{field} secret file must contain a nonempty single-line token")
        value[field] = secret
    return value
