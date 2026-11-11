"""Canonical user-owned filesystem paths for a Fleet node account."""
from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import UserAccount


_SAFE_UNIX_USERNAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class AccountPathError(ValueError):
    """A user-owned Fleet path cannot be safely created or used."""


def ensure_account_path(
    path: Path | str,
    *,
    account: UserAccount,
    mode: int = 0o700,
    create: bool = True,
) -> Path:
    """Validate or create a real account-owned directory below its passwd HOME.

    ``account.home`` must come from the local passwd record. Existing ancestors
    above HOME are checked for symlinks and non-directories; HOME and every
    descendant must also have the account's UID/GID. No existing directory is
    chmodded or chowned.
    """
    if (isinstance(mode, bool) or not isinstance(mode, int) or mode < 0 or mode > 0o777
            or not isinstance(create, bool)):
        raise AccountPathError("account path mode or creation flag is invalid")
    home = Path(account.home)
    target = Path(path)
    if (not home.is_absolute() or home == Path("/") or ".." in home.parts
            or not target.is_absolute() or ".." in target.parts
            or not target.is_relative_to(home)):
        raise AccountPathError("account path must be below the registered HOME")

    current = Path(home.anchor)
    for component in home.parts[1:]:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise AccountPathError("registered account HOME path is unavailable") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise AccountPathError("registered account HOME cannot traverse symlinks or non-directories")

    try:
        home_info = home.lstat()
    except OSError as exc:
        raise AccountPathError("registered account HOME cannot be inspected") from exc
    if (not stat.S_ISDIR(home_info.st_mode) or home_info.st_uid != account.uid
            or home_info.st_gid != account.gid or stat.S_IMODE(home_info.st_mode) & 0o022):
        raise AccountPathError("registered account HOME has unsafe ownership or permissions")

    current = home
    for component in target.relative_to(home).parts:
        current = current / component
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            if not create:
                raise AccountPathError("account path does not exist") from exc
            try:
                current.mkdir(mode=mode)
                metadata = current.lstat()
            except FileExistsError:
                try:
                    metadata = current.lstat()
                except OSError as inspect_exc:
                    raise AccountPathError("account path cannot be inspected") from inspect_exc
            except OSError as create_exc:
                raise AccountPathError("account path cannot be created below HOME") from create_exc
        except OSError as exc:
            raise AccountPathError("account path cannot be inspected") from exc
        if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != account.uid
                or metadata.st_gid != account.gid or stat.S_IMODE(metadata.st_mode) & 0o022):
            raise AccountPathError("account path contains a symlink, non-directory, or foreign-owned component")
    return target


@dataclass(frozen=True)
class UserPaths:
    """Paths rooted in the registered account home, never process environment."""

    home: Path
    root: Path
    management_dir: Path
    runtime_dir: Path
    trust_dir: Path
    state_dir: Path
    roles_dir: Path
    systemd_user_dir: Path

    @classmethod
    def from_account(cls, account: UserAccount) -> UserPaths:
        username = getattr(account, "username", None)
        uid = getattr(account, "uid", None)
        gid = getattr(account, "gid", None)
        home_value = getattr(account, "home", None)
        if (type(username) is not str or _SAFE_UNIX_USERNAME.fullmatch(username) is None
                or username in {"root", ".", ".."} or type(uid) is not int or uid <= 0
                or type(gid) is not int or gid < 0 or type(home_value) is not str):
            raise ValueError("account must provide valid username, non-root uid, gid, and home fields")
        posix_home = PurePosixPath(home_value)
        if (not posix_home.is_absolute() or home_value == "/" or ".." in posix_home.parts
                or len(home_value) > 4096
                or any(ord(char) < 32 or char in "\r\n\u0085\u2028\u2029" for char in home_value)):
            raise ValueError("account home must be an absolute non-root path without line breaks or parent traversal")
        home = Path(home_value)
        root = home / ".local" / "share" / "qwen-fleet"
        return cls(
            home=home,
            root=root,
            management_dir=root / "management",
            runtime_dir=root / "runtime",
            trust_dir=root / "trust",
            state_dir=root / "state",
            roles_dir=root / "roles",
            systemd_user_dir=home / ".config" / "systemd" / "user",
        )
