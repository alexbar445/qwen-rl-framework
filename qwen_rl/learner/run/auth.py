"""Least-privilege authentication for local service Worker policy reads."""
from __future__ import annotations

import hmac
import os
import re
from collections.abc import Iterable, Mapping

from fastapi import Header, HTTPException

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class PolicyReadAuthorizer:
    """Allow the Learner credential and configured service readers on policy GETs.

    This object is attached only to the three read-only policy routes. All other
    Learner, ARC, and management endpoints keep their existing central-token
    dependency.
    """

    __slots__ = ("central_token_env", "service_token_envs")

    def __init__(self, service_token_envs: Iterable[str], *, central_token_env: str = "ARC_ENV_TOKEN"):
        if _ENV_NAME.fullmatch(central_token_env) is None:
            raise ValueError("central policy-reader token environment name is invalid")
        names: list[str] = []
        for name in service_token_envs:
            if (not isinstance(name, str) or _ENV_NAME.fullmatch(name) is None
                    or name == central_token_env):
                raise ValueError("service policy-reader token environment name is invalid")
            if name in names:
                raise ValueError("service policy-reader token environment names must be unique")
            names.append(name)
        self.central_token_env = central_token_env
        self.service_token_envs = tuple(names)
        self.validate_configured_secrets()

    @classmethod
    def from_workers(cls, workers: Iterable[Mapping[str, object]]) -> PolicyReadAuthorizer:
        token_envs = []
        for worker in workers:
            if worker.get("scope", "local") != "service":
                continue
            token_env = worker.get("auth_token_env")
            if not isinstance(token_env, str):
                raise TypeError("service Worker is missing its auth_token_env policy credential")
            token_envs.append(token_env)
        return cls(token_envs)

    def validate_configured_secrets(self) -> None:
        """Fail closed on missing, shared, or central-token aliases at startup."""
        central = os.environ.get(self.central_token_env)
        seen: list[str] = []
        for name in self.service_token_envs:
            token = os.environ.get(name)
            if not token:
                raise RuntimeError(f"service policy-reader environment {name} must be set")
            if central and hmac.compare_digest(token, central):
                raise ValueError("service policy-reader credential must differ from the central token")
            if any(hmac.compare_digest(token, prior) for prior in seen):
                raise ValueError("service policy-reader credentials must be distinct")
            seen.append(token)

    def authorize(self, authorization: str | None) -> None:
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        expected_tokens = [os.environ.get(self.central_token_env, "")]
        expected_tokens.extend(os.environ.get(name, "") for name in self.service_token_envs)
        if any(expected and hmac.compare_digest(supplied, expected) for expected in expected_tokens):
            return
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def __call__(self, authorization: str | None = Header(default=None)) -> None:
        self.authorize(authorization)

    def __repr__(self) -> str:
        return (f"PolicyReadAuthorizer(central_token_env={self.central_token_env!r}, "
                f"service_token_envs={self.service_token_envs!r})")
