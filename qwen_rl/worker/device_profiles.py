"""Preserve native route profiles across Worker states; NInfer owns compatibility.

Cache directories must be scoped to an immutable Worker image. The native engine
still checks hardware class, SM count and route-catalog identity before use.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
import tempfile

_LOG = logging.getLogger(__name__)
_MAX_BYTES = 4 * 1024 * 1024


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError("profile must be a regular file of at most 4 MiB")
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or data.get("schema") != "ninfer.device-route-profiles"
            or data.get("schema_version") != 2 or not isinstance(data.get("devices"), list)):
        raise ValueError("unsupported native device profile schema")
    keys = set()
    for device in data["devices"]:
        if not isinstance(device, dict):
            raise ValueError("invalid profile device")
        hardware, sms = device.get("hardware_class"), device.get("multiprocessors")
        calibration = device.get("calibration")
        if (not isinstance(hardware, str) or not hardware or type(sms) is not int or sms <= 0
                or not isinstance(device.get("routes"), dict)
                or not isinstance(calibration, dict)
                or not isinstance(calibration.get("route_catalog"), str)
                or not calibration["route_catalog"]):
            raise ValueError("profile lacks calibrated hardware/SM/catalog identity")
        key = hardware, sms
        if key in keys:
            raise ValueError("duplicate native hardware profile")
        keys.add(key)
    return data


def _write(path: Path, data: dict) -> None:
    content = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()
    if len(content) > _MAX_BYTES:
        raise ValueError("merged profile exceeds 4 MiB")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".device-profile-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class DeviceProfileCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        if not self.directory.is_absolute() or self.directory.is_symlink():
            raise ValueError("device profile cache must be an absolute directory without a symlink")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "device-profiles.json"

    @contextmanager
    def _lock(self):
        fd = os.open(self.directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def seed(self, private_path: Path) -> bool:
        """Copy only into a missing private profile. Never replace local calibration."""
        try:
            with self._lock():
                if private_path.exists() or private_path.is_symlink() or not self.path.exists():
                    return False
                _write(private_path, _read(self.path))
            _LOG.info("NInfer device profile seeded from persistent cache")
            return True
        except (OSError, ValueError) as exc:
            _LOG.warning("NInfer device profile cache miss: %s", exc)
            return False

    def publish(self, private_path: Path) -> bool:
        """Publish only after Engine readiness; merge devices under an interprocess lock."""
        try:
            fresh = _read(private_path)
            if not fresh["devices"]:
                return False
            with self._lock():
                try:
                    existing = _read(self.path) if self.path.exists() else {**fresh, "devices": []}
                except (OSError, ValueError):
                    existing = {**fresh, "devices": []}
                devices = {(x["hardware_class"], x["multiprocessors"]): x for x in existing["devices"]}
                devices.update({(x["hardware_class"], x["multiprocessors"]): x for x in fresh["devices"]})
                _write(self.path, {**fresh, "devices": [devices[k] for k in sorted(devices)]})
            _LOG.info("NInfer device profile saved to persistent cache")
            return True
        except (OSError, ValueError) as exc:
            _LOG.warning("NInfer device profile cache not saved: %s", exc)
            return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Preseed an image-scoped native device profile cache")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if not DeviceProfileCache(args.cache_dir).publish(args.source):
        return 1
    print("Profile cached; NInfer will verify hardware, SM count and route catalog at startup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
