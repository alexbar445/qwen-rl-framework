#!/usr/bin/env python3
"""Shared ARC prompt construction and pinned toolkit signature validation."""
from __future__ import annotations

import json
import hashlib
import importlib
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
from .paths import ROOT
ASSETS = ROOT / "assets"
from .resources.arc import read_system_rules

USER_PREFIX = "Visible ARC task JSON:\n"
PROMPT_FORMAT = "arc-task-user-context/v2"
GENERATED_INFERENCE_CONVENTIONS = "The background color is 0; objects are same-color four-connected components by default."
RULE_INFERENCE_INSTRUCTION = (
    "This is a grid transformation task. All training input/output examples follow the same transformation rule. "
    "Infer that rule from the examples, apply it to every test input, and use the tools to submit the predicted output grids."
)


TOOLKIT_DEPENDENCY = ROOT / "toolkit-dependency.json"
_BOUND_TOOLKIT_ROOT: Path | None = None


def load_toolkit_runtime(toolkit: Path | None = None):
    """Bind one toolkit implementation per process; never silently reuse another."""
    global _BOUND_TOOLKIT_ROOT
    loaded = sys.modules.get("arc_toolkit.runtime")
    selected = toolkit if toolkit is not None else (_BOUND_TOOLKIT_ROOT or ROOT / "vendor" / "paired-factory-v1")
    root = Path(selected).resolve()
    expected = root / "arc_toolkit" / "runtime.py"
    if not expected.is_file():
        raise RuntimeError(
            "ARC toolkit is not bundled with qwen-rl-framework. Supply a reviewed toolkit directory "
            f"explicitly or set ARC_RL_ROOT to a compatible source bundle; expected {expected}"
        )
    if loaded is None:
        sys.path.insert(0, str(root))
        loaded = importlib.import_module("arc_toolkit.runtime")
    if Path(loaded.__file__).resolve() != expected:
        raise RuntimeError("a different ARC toolkit is already loaded; use a fresh process for this toolkit")
    _BOUND_TOOLKIT_ROOT = root
    return loaded


def _language_provider(runtime):
    provider = getattr(runtime, "action_language_contract", None)
    if not callable(provider):
        raise RuntimeError("ARC parser has no action-language contract; rebuild toolkit, prompt and dependency together")
    return provider


def _action_language(bundle: dict | None = None) -> dict:
    provider = _language_provider(load_toolkit_runtime())
    actual = provider()
    if bundle is None:
        bundle = json.loads(_read_toolkit_dependency())
    declared = bundle.get("bundle", {}).get("catalog", {}).get("action_language")
    if not declared or actual != declared:
        raise RuntimeError("loaded ARC parser differs from the declared action-language contract")
    return actual


def _read_toolkit_dependency() -> str:
    try:
        return TOOLKIT_DEPENDENCY.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ARC toolkit identity contract is not bundled with qwen-rl-framework. "
            "Set ARC_RL_ROOT to a compatible source bundle containing toolkit-dependency.json."
        ) from exc


def validate_toolkit_binding(toolkit: Path | None = None) -> dict:
    runtime = load_toolkit_runtime(toolkit)
    language = _action_language()
    validate_prompt_catalog(runtime.current_catalog())
    from arc_toolkit.dependencies import check_dependency
    status = check_dependency(TOOLKIT_DEPENDENCY, Path(runtime.__file__).resolve().parents[1])
    if status.get("status") != "current":
        raise RuntimeError("loaded ARC implementation differs from its pinned dependency; rebuild the toolkit bundle")
    return language


def validate_service_action_language(fingerprint: str | None) -> None:
    if fingerprint != _action_language()["fingerprint"]:
        raise RuntimeError("ARC service action-language contract is missing or differs from the sampling client")


def _toolkit_catalog(bundle: dict | None = None) -> list[dict]:
    """Structured dependency data is the only toolkit signature source."""
    if bundle is None:
        bundle = json.loads(_read_toolkit_dependency())
    catalog = bundle.get("bundle", {}).get("catalog", {}).get("tools")
    if not isinstance(catalog, list) or not catalog:
        raise AssertionError("toolkit dependency bundle carries no tool catalog")
    names = [tool.get("name") for tool in catalog]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise AssertionError("toolkit dependency catalog names are invalid")
    return catalog


def _render_tool_signature(tool: dict) -> str:
    arguments = []
    for parameter in tool["parameters"]:
        name = parameter["name"]
        annotation = parameter.get("type")
        if not isinstance(annotation, str) or not annotation:
            raise AssertionError(f"toolkit catalog parameter has no type: {tool['name']}.{name}")
        if parameter.get("enum"):
            annotation = "Literal[" + ",".join(repr(value) for value in parameter["enum"]) + "]"
        arguments.append(name + ":" + annotation + (
            "*" if parameter["required"] else "=" + json.dumps(parameter.get("default"), ensure_ascii=False)))
    signature = f"{tool['name']}({', '.join(arguments)}) -> {tool['returns']}"
    forms = tool.get("argument_forms")
    if forms is not None:
        if (not isinstance(forms, dict)
                or set(forms) != {"groups", "selection"}
                or forms.get("selection") != "exactly_one_non_null_group"
                or not isinstance(forms.get("groups"), list)
                or not forms["groups"]
                or any(not isinstance(group, list) or not group
                       or any(not isinstance(name, str) or not name for name in group)
                       for group in forms["groups"])):
            raise AssertionError(f"toolkit catalog argument forms are invalid: {tool['name']}")
        choices = " | ".join("+".join(group) for group in forms["groups"])
        signature += f" [exactly one non-null group: {choices}]"
    items_kind = tool.get("items_kind")
    if items_kind is not None:
        if not isinstance(items_kind, str) or not items_kind:
            raise AssertionError(f"toolkit catalog item kind is invalid: {tool['name']}")
        signature += f" [items:{items_kind}]"
    description = tool.get("description", "").split("\n", 1)[0].strip()
    if description and description != "Schema-only signature helper.":
        signature += " — " + description
    return signature


def system_prompt() -> str:
    """One shared rule set and typed catalog for ARC and generated ARC tasks."""
    runtime = load_toolkit_runtime()
    source_rules = ASSETS / "arc-system-rules.txt"
    try:
        rules_text = source_rules.read_text(encoding="utf-8")
    except FileNotFoundError:
        rules_text = read_system_rules()
    return _cached_system_prompt(
        rules_text,
        _read_toolkit_dependency(),
        str(Path(runtime.__file__).resolve()),
        _language_provider(runtime),
    )


@lru_cache(maxsize=1)
def _cached_system_prompt(rules_text, dependency_text, runtime_path, language_provider) -> str:
    # Exact file contents and the bound implementation identify the static parts.
    # Task instructions, grids and private targets never enter this cache.
    bundle = json.loads(dependency_text)
    rules = rules_text.strip()
    marker = "{{ARC_ACTION_LANGUAGE_RULES}}"
    if rules.count(marker) != 1:
        raise RuntimeError("ARC prompt must bind exactly one action-language contract")
    rules = rules.replace(marker, _action_language(bundle)["rules"])
    tools = _toolkit_catalog(bundle)
    lines = [rules, "", "Toolkit catalog:"]
    sections = (
        ("Analysis (A.*):", [tool for tool in tools if tool["name"].startswith("A.")]),
        ("Operations (O.*):", [tool for tool in tools if tool["name"].startswith("O.")]),
        ("Control:", [tool for tool in tools if not tool["name"].startswith(("A.", "O."))]),
    )
    for title, section_tools in sections:
        if section_tools:
            lines.extend(("", title))
            lines.extend(_render_tool_signature(tool) for tool in section_tools)
    return "\n".join(lines)


def action_tools() -> list[dict]:
    runtime = load_toolkit_runtime()
    from arc_toolkit.constrained_actions import compact_native_schema
    return deepcopy(_cached_action_tools(
        _read_toolkit_dependency(), str(Path(runtime.__file__).resolve()),
        _language_provider(runtime), compact_native_schema,
    ))


@lru_cache(maxsize=1)
def _cached_action_tools(dependency_text, runtime_path, language_provider, schema_provider) -> list[dict]:
    _action_language(json.loads(dependency_text))
    return [{"type": "function", "function": {"name": "arc_action",
        "description": "Execute one registered ARC action with action and args; saved references feed later actions.",
        "parameters": schema_provider()}}]


def grid_rows(grid: list[list[int]]) -> list[str]:
    """Use the pinned observation codec for initial model-visible grids too."""
    runtime = load_toolkit_runtime()
    runtime.common.validate_grid(grid)
    from arc_toolkit.observation_v3 import encode_observation_v3

    return encode_observation_v3({
        "type": "Grid", "summary": {"shape": [len(grid), len(grid[0])]},
        "data": grid,
    })["data"]


def task_json(challenge: dict) -> str:
    """Pack public grid fields without mutating the underlying task matrices."""
    packed = dict(challenge)
    for split in ("train", "test"):
        if split in challenge:
            packed[split] = [
                {key: grid_rows(value) if key in {"input", "output"} else value
                 for key, value in pair.items()}
                for pair in challenge[split]
            ]
    return json.dumps(packed, separators=(",", ":"), ensure_ascii=False)


def reference_names(challenge: dict) -> list[str]:
    """Public input references, in the toolkit's naming and visible order."""
    names = []
    for index, pair in enumerate(challenge.get("train") or []):
        for key in ("input", "output"):
            if isinstance(pair, dict) and key in pair:
                names.append(f"train/{index}/{key}")
    for index, pair in enumerate(challenge.get("test") or []):
        if isinstance(pair, dict) and "input" in pair:
            names.append(f"test/{index}/input")
    return names


def user_message(challenge: dict, instruction: str | None = None, *,
                 input_roles: dict[str, str] | None = None,
                 submission_targets: tuple[str, ...] | list[str] | None = None,
                 conventions: str | None = None) -> str:
    """Render only public task context; private target grids are never accepted."""
    instruction = instruction or RULE_INFERENCE_INSTRUCTION
    names = reference_names(challenge)
    roles = input_roles or {}
    inputs = [name + (f" ({roles[name]})" if name in roles else "") for name in names]
    targets = (list(submission_targets) if submission_targets is not None else
               [f"test/{index}" for index, _ in enumerate(challenge.get("test") or [])])
    return ("Task instruction:\n" + instruction
            + ("\n\nTask conventions:\n" + conventions if conventions else "")
            + "\n\nAvailable inputs:\n" + "\n".join(inputs)
            + "\n\nSubmit targets:\n" + "\n".join(targets)
            + "\n\n" + USER_PREFIX + task_json(challenge))


def build_messages(challenge: dict, instruction: str | None = None, *,
                   input_roles: dict[str, str] | None = None,
                   submission_targets: tuple[str, ...] | list[str] | None = None,
                   conventions: str | None = None) -> list[dict]:
    """One fixed system message and one public task message in every mode."""
    return [{"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_message(
                challenge, instruction, input_roles=input_roles,
                submission_targets=submission_targets, conventions=conventions)}]


def build_task_messages(task) -> list[dict]:
    """Shared TaskSpec entrypoint for native rollout, evaluation and BEPA."""
    return build_messages(task.challenge, task.instruction,
                          input_roles=getattr(task, "input_roles", None),
                          submission_targets=getattr(task, "submission_targets", None) or None,
                          conventions=getattr(task, "conventions", None))


def prompt_contract() -> dict:
    """Fingerprint the full public message format, not only the system prefix."""
    return {"format": PROMPT_FORMAT,
            "generated_inference_conventions": GENERATED_INFERENCE_CONVENTIONS,
            "messages": build_messages({"train": [], "test": []}),
            "tools": action_tools()}


def prompt_identity() -> dict[str, str]:
    payload = json.dumps(prompt_contract(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return {"format": PROMPT_FORMAT, "fingerprint": hashlib.sha256(payload).hexdigest()}


def validate_prompt_catalog(catalog: dict) -> None:
    """Require the rendered pinned catalog to match the live runtime catalog."""
    tools = catalog.get("tools") if isinstance(catalog, dict) else None
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        raise AssertionError("runtime catalog must contain a list of tool schemas")
    if catalog.get("action_language") != _action_language():
        raise AssertionError("runtime catalog parser contract differs from the prompt")
    live = {}
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name or name in live:
            raise AssertionError("runtime catalog has invalid or duplicate tool names")
        live[name] = tool
    rendered = set(system_prompt().splitlines())
    if "Toolkit catalog:" not in rendered:
        raise AssertionError("system prompt contains no toolkit catalog")
    for expected in _toolkit_catalog():
        name = expected["name"]
        actual = live.get(name)
        if actual is None:
            raise AssertionError(f"pinned toolkit tool is missing from runtime: {name}")
        if _render_tool_signature(expected) not in rendered:
            raise AssertionError(f"system prompt omits the pinned toolkit signature: {name}")
        expected_signature = [(p["name"], p["required"], p.get("type"))
                              for p in expected["parameters"]]
        actual_signature = [(p["name"], p["required"], p.get("type"))
                            for p in actual["parameters"]]
        if (actual_signature != expected_signature
                or actual.get("returns") != expected["returns"]
                or actual.get("items_kind") != expected.get("items_kind")
                or actual.get("returns_by_source") != expected.get("returns_by_source")
                or actual.get("argument_forms") != expected.get("argument_forms")
                or actual.get("argument_cases") != expected.get("argument_cases")):
            raise AssertionError(f"pinned toolkit signature differs from runtime: {name}")
