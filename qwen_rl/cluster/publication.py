"""Stream already verified content to a site's authenticated publisher."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import httpx

from .settings import validate_credential_origin


async def publish_blob(path: Path, *, sha256: str, size_bytes: int,
                       gateway_url: str, token: str, client: httpx.AsyncClient | None = None) -> dict:
    """One WAN transfer per content SHA; repeated publications verify and skip."""
    from .artifacts import _validate_sha256
    from urllib.parse import urlsplit
    _validate_sha256(sha256)
    validate_credential_origin(gateway_url)
    origin = urlsplit(gateway_url)
    if origin.path not in {"", "/"} or origin.query or origin.fragment:
        raise ValueError("Gateway publisher must be an origin")
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size_bytes:
        raise ValueError("publication requires a regular file of the pinned size")
    if not token or any(c.isspace() for c in token):
        raise ValueError("publisher token must be a single-line value")

    def verify():
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError("publication content differs from its pinned SHA")
    await asyncio.to_thread(verify)
    owned = client is None
    http = client or httpx.AsyncClient(timeout=7200, follow_redirects=False)
    url = gateway_url.rstrip("/") + "/v1/artifacts/" + sha256
    headers = {"Authorization": "Bearer " + token}
    try:
        response = await http.head(url, headers=headers)
        if response.status_code == 200:
            if response.headers.get("etag") != f'"sha256:{sha256}"' or response.headers.get("content-length") != str(size_bytes):
                raise ValueError("site content identity conflicts with publication")
            return {"state": "already_present", "sha256": sha256, "size_bytes": size_bytes}
        if response.status_code != 404:
            raise RuntimeError(f"site publication HEAD returned HTTP {response.status_code}")

        async def chunks():
            with path.open("rb") as source:
                while chunk := await asyncio.to_thread(source.read, 8 * 1024 * 1024):
                    yield chunk
        response = await http.put(url, content=chunks(), headers={**headers,
            "Content-Length": str(size_bytes), "Content-Type": "application/octet-stream"})
        if response.status_code not in {200, 201}:
            raise RuntimeError(f"site publication PUT returned HTTP {response.status_code}")
        receipt = response.json()
        if receipt.get("sha256") != sha256 or receipt.get("size_bytes") != size_bytes:
            raise ValueError("site did not acknowledge the pinned content")
        return receipt
    finally:
        if owned:
            await http.aclose()
