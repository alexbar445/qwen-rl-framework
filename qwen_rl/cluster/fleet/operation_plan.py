"""Strict, private recipes for the existing Fleet deployment primitives."""
from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$"
_SHA = r"^[0-9a-f]{64}$"


class OperationPlanError(ValueError):
    """A recipe failed a deployment boundary; input values are never echoed."""


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/"))
            or any(ord(c) < 32 for c in value)):
        raise ValueError("expected a relative path below the registered HOME")
    return value


def absolute_path(value: str) -> str:
    if (not Path(value).is_absolute() or ".." in Path(value).parts
            or any(ord(c) < 32 for c in value)):
        raise ValueError("expected an absolute path")
    return value


class Target(Contract):
    node: StrictStr = Field(pattern=_ID)
    role: Literal["worker", "learner"]


class Step(Contract):
    id: StrictStr = Field(pattern=_ID)
    depends_on: list[StrictStr] = Field(default_factory=list, max_length=500)
    timeout_seconds: StrictInt = Field(default=1800, ge=1, le=86400)


class TransferInput(Contract):
    source: StrictStr
    remote_path: StrictStr
    size: StrictInt = Field(ge=0)
    sha256: StrictStr = Field(pattern=_SHA)

    _source = field_validator("source")(absolute_path)
    _remote = field_validator("remote_path")(relative_path)


class TransferStep(Step):
    kind: Literal["transfer"]
    node: StrictStr = Field(pattern=_ID)
    remote_dir: StrictStr
    artifact_scope: Literal["worker_source", "worker_model", "worker_image", "site_source",
                            "site_private", "learner_bundle", "learner_image"]
    files: list[TransferInput] = Field(min_length=1, max_length=10000)
    tls_leaf_certificate: StrictStr | None = None

    _directory = field_validator("remote_dir")(relative_path)

    @model_validator(mode="after")
    def unique_files(self):
        names = [file.remote_path for file in self.files]
        if len(set(names)) != len(names):
            raise ValueError("transfer paths must be unique")
        if self.tls_leaf_certificate is not None:
            absolute_path(self.tls_leaf_certificate)
        return self


class ExecStep(Step):
    kind: Literal["exec"]
    node: StrictStr | None = Field(default=None, pattern=_ID)
    purpose: Literal["prepare", "deploy", "verify"]
    argv: list[StrictStr] = Field(min_length=1, max_length=128)
    cwd: StrictStr | None = None
    expected_json: dict[str, Any] | None = None

    @field_validator("argv")
    @classmethod
    def command(cls, value):
        for index, part in enumerate(value):
            if not part or any(c in part for c in "\x00\r\n"):
                raise ValueError("command arguments must be nonempty single-line strings")
            if "PRIVATE KEY-----" in part or re.search(r"\bBearer\s+\S+", part):
                raise ValueError("secrets must be file references, never command arguments")
            if re.match(r"^--(?:token|password|api-key|secret)(?:=|$)", part):
                raise ValueError("use the existing command's secret-file option")
            if re.match(r"^--(?:token|password|api-key|secret|bearer)-file(?:=|$)", part):
                ref = part.split("=", 1)[1] if "=" in part else value[index + 1] if index + 1 < len(value) else ""
                absolute_path(ref.removeprefix("file:"))
        # Recipes deploy idle services. Training remains a separately authorized operation.
        if (any("qwen_rl" in part or "qwen-rl" in part for part in value)
                and ("run" in value or "--teacher" in value)):
            raise ValueError("deployment recipes cannot start training or Teacher")
        if (any("qwen_rl" in part or "qwen-rl" in part for part in value)
                and ("teacher" in value or "bootstrap" in value
                     or (any(group in value for group in ("runtime", "local")) and "start" in value))):
            raise ValueError("runtime and Teacher startup must not bypass the role adapters")
        return value

    @model_validator(mode="after")
    def execution_location(self):
        if self.cwd is not None:
            absolute_path(self.cwd)
        if self.node is not None and self.cwd is not None:
            raise ValueError("remote commands specify their working directory in their argv")
        if self.expected_json is not None:
            json.dumps(self.expected_json, allow_nan=False)
        return self


class ExportImageStep(Step):
    kind: Literal["export_image"]
    node: StrictStr = Field(pattern=_ID)
    image_id: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    content_dir: StrictStr
    artifact_base_url: StrictStr
    python_executable: StrictStr | None = None
    docker_context: StrictStr = Field(default="fleet-rootless", pattern=_ID)

    _directory = field_validator("content_dir")(relative_path)


class AssignStep(Step):
    kind: Literal["assign"]
    node: StrictStr = Field(pattern=_ID)
    deployment_file: StrictStr

    _file = field_validator("deployment_file")(absolute_path)


class WaitRoleStep(Step):
    kind: Literal["wait_role"]
    node: StrictStr = Field(pattern=_ID)
    assignment_step: StrictStr = Field(pattern=_ID)


class UpgradeStep(Step):
    kind: Literal["upgrade_management"]
    node: StrictStr = Field(pattern=_ID)


OperationStep = Annotated[TransferStep | ExecStep | ExportImageStep | AssignStep | WaitRoleStep | UpgradeStep,
                          Field(discriminator="kind")]


class OperationPlan(Contract):
    schema_version: Literal["qwen-fleet-site-operation/v1"] = Field(alias="schema")
    site_id: StrictStr = Field(pattern=_ID)
    scope: Literal["preparation", "deployment"]
    targets: list[Target] = Field(min_length=1, max_length=1000)
    steps: list[OperationStep] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def coherent_recipe(self):
        targets = {target.node: target for target in self.targets}
        steps = {step.id: step for step in self.steps}
        if len(targets) != len(self.targets) or len(steps) != len(self.steps):
            raise ValueError("target names and step IDs must be unique")
        for step in self.steps:
            node = getattr(step, "node", None)
            if node is not None and node not in targets:
                raise ValueError("every remote step must name a declared target")
            if len(set(step.depends_on)) != len(step.depends_on) or any(x not in steps for x in step.depends_on):
                raise ValueError("dependencies must name unique existing steps")
            if isinstance(step, TransferStep):
                role = targets[step.node].role
                if (step.artifact_scope.startswith("worker_") and role != "worker"
                        or step.artifact_scope in {"site_source", "site_private", "learner_bundle", "learner_image"}
                        and role != "learner"):
                    raise ValueError("artifact scope does not match the declared role")
            if isinstance(step, WaitRoleStep):
                assignment = steps.get(step.assignment_step)
                if (not isinstance(assignment, AssignStep) or assignment.node != step.node
                        or step.assignment_step not in step.depends_on):
                    raise ValueError("readiness requires the same node's assignment dependency")
        assignments = [step for step in self.steps if isinstance(step, AssignStep)]
        waited = {step.assignment_step for step in self.steps if isinstance(step, WaitRoleStep)}
        if any(step.id not in waited for step in assignments):
            raise ValueError("every role assignment must have an actual readiness step")
        if self.scope == "deployment":
            roles = [target.role for target in self.targets]
            if roles.count("learner") != 1 or roles.count("worker") < 1:
                raise ValueError("deployment requires one Learner and at least one Worker")
            if {step.node for step in assignments} != set(targets):
                raise ValueError("deployment must assign and verify every declared target")
        self.ordered_steps()
        return self

    def ordered_steps(self) -> list[OperationStep]:
        done: set[str] = set()
        output = []
        while len(output) < len(self.steps):
            ready = [step for step in self.steps if step.id not in done and set(step.depends_on) <= done]
            if not ready:
                raise ValueError("operation dependency cycle")
            output.extend(ready)
            done.update(step.id for step in ready)
        return output


def load_plan(path: Path) -> OperationPlan:
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError("duplicate JSON key")
            output[key] = value
        return output

    try:
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_size > 8 * 1024 * 1024):
            raise ValueError("unsafe recipe file")
        value = json.loads(path.read_text(), object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        return OperationPlan.model_validate(value, strict=True)
    except (OSError, ValueError) as exc:
        raise OperationPlanError("operation_plan_invalid") from exc
