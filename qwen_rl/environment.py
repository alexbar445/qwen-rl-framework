#!/usr/bin/env python3
"""Gymnasium-style RL environment over the vendored ARC toolkit runtime.

Contract (RSI/docs/design/arc-agent-tool-library.md §8, revised 2026-10-09):

* one episode = one task; action = one existing tool-call line (or a submit call);
* observation = the real tool return plus the remaining token budget (the design's
  "剩余预算") -- the policy never sees target outputs;
* the env is the accountant and the exact-match gate only: 1.0 when every target input
  matches exactly, else 0.0.  The efficiency term is added by the group ranker
  (qwen_rl/rewards.py) because a within-group rank needs the whole group;
* cost is token based: w_output * generated + w_observation * fed_back.  The token budget
  is the only task budget (exhausting it is a failure); call_cap is a harness safety net;
  the wall clock is an infrastructure abort (rollout discarded, never scored as a task
  failure -- §8: 机器故障独立记录);
* the episode stays auditable: the session directory, the exact lines and the
  runtime responses are all kept on disk.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The vendored ARC runtime validates the tool signatures advertised by the
# model's SFT prompt. Its source hash is included in episode provenance.
from .paths import ROOT
DEFAULT_TOOLKIT = ROOT / "vendor" / "paired-factory-v1"

from .training_verification import TrainingVerification
from .budget import Contract               # noqa: E402  (single source of truth)
from .contracts.episode_budget import (BUDGET_POLICY_ID,
                                       budget_observation_fields)  # noqa: E402
from .prompt import validate_prompt_catalog, user_message, load_toolkit_runtime, grid_rows   # noqa: E402

_BUDGET_MARKER_MAX_ITERATIONS = 8

# The prompt is not built here.  qwen_rl/prompt.py renders the exact SFT-format
# conversation (system message carrying the toolkit catalog + user message carrying the
# visible task JSON); reset() returns the user-message text, which is what TRL appends to
# the last user turn of the dataset row.


def _count_text_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _tokenizer_vocab_snapshot(tokenizer: Any) -> tuple[Any, ...] | None:
    """Capture vocab identity without asking a fast tokenizer to merge its full vocab."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    get_vocab_size = getattr(backend, "get_vocab_size", None)
    if callable(get_vocab_size):
        base_vocab_size = get_vocab_size(with_added_tokens=False)
        get_added_tokens_decoder = getattr(backend, "get_added_tokens_decoder", None)
        if callable(get_added_tokens_decoder):
            added_tokens = get_added_tokens_decoder()
        else:
            added_tokens = getattr(tokenizer, "added_tokens_decoder", None)
        if added_tokens is None:
            raise ValueError("fast tokenizer does not expose its added-token decoder")

        added_token_snapshot = tuple(sorted(
            (
                token_id,
                getattr(token, "content", token),
                getattr(token, "single_word", None),
                getattr(token, "lstrip", None),
                getattr(token, "rstrip", None),
                getattr(token, "normalized", None),
                getattr(token, "special", None),
            )
            for token_id, token in added_tokens.items()
        ))
        return ("fast", base_vocab_size, added_token_snapshot)

    # Lightweight test/custom tokenizers may expose only Python's length protocol.
    try:
        return ("custom", len(tokenizer))
    except (AttributeError, TypeError):
        return None


@dataclass(frozen=True, slots=True)
class EnvironmentTokenFraming:
    """Fixed role-framing costs for one service's immutable tokenizer instance.

    Template, special-token map, and vocabulary identity are checked on injection.
    Fast tokenizers use their immutable base-vocabulary size plus the added-token
    decoder; the service must not mutate either part in place during its lifetime.
    """

    _tokenizer: Any = field(repr=False, compare=False)
    _chat_template_snapshot: Any = field(repr=False, compare=False)
    _special_tokens_snapshot: Any = field(repr=False, compare=False)
    _tokenizer_vocab: tuple[Any, ...] | None
    observation_overhead: int
    feedback_overhead: int

    @classmethod
    def measure(cls, tokenizer: Any) -> "EnvironmentTokenFraming":
        if tokenizer is None:
            raise ValueError("a tokenizer is required for environment framing")

        def rendered(messages: list[dict]) -> int:
            return len(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_dict=False))

        base = [{"role": "user", "content": "solve"}]
        call = {"role": "assistant", "tool_calls": [{"type": "function", "function": {
            "name": "arc_action", "arguments": {"action": "tools", "args": {}}}}]}
        result = {"role": "tool", "content": ""}
        observation_overhead = rendered(base + [call, result]) - rendered(base + [call])

        probe = "feedback"
        prefix = [
            {"role": "system", "content": "feedback framing"},
            {"role": "user", "content": "accounting context"},
        ]
        before = rendered(prefix)
        after = rendered(prefix + [{"role": "user", "content": probe}])
        feedback_overhead = max(0, after - before - _count_text_tokens(tokenizer, probe))
        tokenizer_vocab = _tokenizer_vocab_snapshot(tokenizer)
        return cls(
            tokenizer,
            copy.deepcopy(getattr(tokenizer, "chat_template", None)),
            copy.deepcopy(getattr(tokenizer, "special_tokens_map", None)),
            tokenizer_vocab,
            observation_overhead,
            feedback_overhead,
        )

    def validate_tokenizer(self, tokenizer: Any) -> None:
        if self._tokenizer is not tokenizer:
            raise ValueError("environment framing belongs to another tokenizer instance")
        if getattr(tokenizer, "chat_template", None) != self._chat_template_snapshot:
            raise ValueError("environment framing chat template changed during service lifetime")
        if getattr(tokenizer, "special_tokens_map", None) != self._special_tokens_snapshot:
            raise ValueError("environment framing special-token map changed during service lifetime")
        if self._tokenizer_vocab is not None:
            try:
                current_vocab = _tokenizer_vocab_snapshot(tokenizer)
            except (AttributeError, TypeError) as exc:
                raise ValueError(
                    "environment framing tokenizer vocabulary became unavailable"
                ) from exc
            if current_vocab != self._tokenizer_vocab:
                raise ValueError(
                    "environment framing tokenizer vocabulary changed during service lifetime"
                )


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass
class EpisodeState:
    lines: list[dict[str, Any]] = field(default_factory=list)
    responses: list[dict] = field(default_factory=list)
    errors: int = 0
    successful_tool_calls: int = 0
    submissions: dict[str, dict[int, str]] = field(default_factory=dict)  # target -> attempt -> grid artifact
    owned_artifacts: set[str] = field(default_factory=set)
    echoed_targets: int = 0
    latest_demo_verification: dict[str, bool | None] = field(default_factory=dict)
    verified: int = 0
    exact_train: int = 0
    started_at: float = 0.0
    out_tokens: int = 0
    obs_tokens: int = 0
    aborted: bool = False
    abort_reason: str | None = None
    out_count_mode: str = "generated_ids"
    visible_target_matches: int = 0
    observation_budget_blocked: bool = False


class ARCToolEnv:
    """One episode of the ARC toolkit; targets are held privately."""

    def __init__(self, challenge: dict, targets: dict[str, list[list[int]]], task_id: str,
                 workdir: Path, contract: Contract, token_budget: int,
                 tokenizer: Any, toolkit: Path = DEFAULT_TOOLKIT,
                 instruction: str | None = None, input_roles: dict[str, str] | None = None,
                 submission_targets: tuple[str, ...] | list[str] | None = None,
                 conventions: str | None = None,
                 token_framing: EnvironmentTokenFraming | None = None):
        if tokenizer is None:
            raise ValueError("a tokenizer is required for observation accounting")
        runtime = load_toolkit_runtime(toolkit)
        self._Session = runtime.Session
        self.challenge = challenge
        # Generated-curriculum tasks (`arc-agent-task-v2`) carry an explicit instruction contract.
        # The toolkit accepts the three extension fields only together, and requires them when
        # ``train`` is empty, so they travel as one task payload into ``Session.create``.
        self.instruction = instruction
        self.conventions = conventions
        self.input_roles = dict(input_roles or {})
        self.submission_targets = tuple(submission_targets or ())
        self.task_payload = dict(challenge)
        if instruction is not None:
            self.task_payload["instruction"] = instruction
            self.task_payload["input_roles"] = dict(self.input_roles)
            self.task_payload["submission_targets"] = list(self.submission_targets)
        self.targets = {k: v for k, v in targets.items()}   # never exposed
        self.required_submission_targets = tuple(self.targets)
        self.required_demo_verification_targets = tuple(
            f"train/{index}/output" for index, _ in enumerate(challenge.get("train") or [])
        )
        self.training_verification = TrainingVerification(self.required_demo_verification_targets)
        self.task_id = task_id
        self.workdir = Path(workdir)
        self.contract = contract          # weights / epsilon / caps: one definition site
        self.token_budget = int(token_budget)
        self.session = None
        self.state = EpisodeState(latest_demo_verification={
            target: None for target in self.required_demo_verification_targets
        })
        self._target_fingerprints = {k: json.dumps(v, separators=(",", ":")) for k, v in self.targets.items()}
        # outputs the challenge itself shows: a held target equal to one of them is a
        # duplicate grid, not a leak (ARC tasks reuse grids; variants hide one pair)
        self._visible_output_fingerprints = {
            json.dumps(pair.get("output"), separators=(",", ":"))
            for pair in (challenge.get("train") or []) if isinstance(pair, dict)}
        self._visible_grid_fingerprints = set(self._visible_output_fingerprints)
        for pair in [*(challenge.get("train") or []), *(challenge.get("test") or [])]:
            if isinstance(pair, dict) and isinstance(pair.get("input"), list):
                self._visible_grid_fingerprints.add(json.dumps(pair["input"], separators=(",", ":")))
        self.tokenizer = tokenizer
        self.token_counter_name = "qwen-chat-template"
        if token_framing is None:
            token_framing = EnvironmentTokenFraming.measure(tokenizer)
        elif not isinstance(token_framing, EnvironmentTokenFraming):
            raise TypeError("token_framing must be an EnvironmentTokenFraming")
        token_framing.validate_tokenizer(tokenizer)
        self.token_framing = token_framing
        self.observation_overhead = token_framing.observation_overhead
        self.feedback_overhead = token_framing.feedback_overhead
        runtime_py = Path(runtime.__file__).resolve()
        self.toolkit_sha256 = (hashlib.sha256(runtime_py.read_bytes()).hexdigest()
                               if runtime_py.exists() else None)

    # ------------------------------------------------------- token accounting
    def _count_text(self, text: str) -> int:
        """Count observation content using the deployment tokenizer."""
        return _count_text_tokens(self.tokenizer, text)

    def tokens_used(self) -> int:
        """Raw tokens the episode consumed (the quantity the token budget limits)."""
        return self.state.out_tokens + self.state.obs_tokens

    def record_generation(self, token_count: int) -> None:
        """Charge a model turn from the sampler's original token IDs.

        The service receives these IDs directly from the rollout worker, so this is the
        authoritative path for RL accounting.  It includes thinking and turns without a
        tool call and deliberately does not re-tokenize decoded text.
        """
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise ValueError("token_count must be a non-negative integer")
        self.state.out_count_mode = "generated_ids"
        self.state.out_tokens += token_count

    def record_observation(self, text: str) -> int:
        """Charge an externally rendered tool response."""
        count = self.observation_overhead + self._count_text(text)
        self.state.obs_tokens += count
        return count

    def record_feedback(self, text: str) -> int:
        """Charge trusted protocol feedback rendered as a user message, not a tool result."""
        if not isinstance(text, str):
            raise TypeError("feedback text must be a string")
        count = self.feedback_overhead + self._count_text(text)
        self.state.obs_tokens += count
        return count

    def cost(self) -> float:
        """Weighted cost (w_output : w_observation) used for continuous efficiency."""
        return self.contract.cost(self.state.out_tokens, self.state.obs_tokens)

    # ---------------------------------------------------------------- helpers
    def _render_observation(self, response: dict) -> str:
        from .observations import encode_observation
        return encode_observation(response)

    def budget_exhausted(self, *, exceeded: bool = False) -> bool:
        """Check each configured quota against the tokens actually delivered."""
        threshold = (lambda used, cap: used > cap) if exceeded else (lambda used, cap: used >= cap)
        if self.contract.generated_token_budget is not None:
            return (self.state.observation_budget_blocked
                    or threshold(self.state.out_tokens, self.contract.generated_token_budget)
                    or threshold(self.state.obs_tokens, self.contract.observation_token_budget))
        return threshold(self.tokens_used(), self.token_budget)

    def budget_limit_reason(self) -> str:
        if self.contract.generated_token_budget is not None:
            if (self.state.observation_budget_blocked
                    or self.state.obs_tokens >= self.contract.observation_token_budget):
                return "observation_budget_exceeded"
            return "generated_budget_exceeded"
        return "token_budget_exceeded"

    def generation_budget_fields(self) -> dict:
        if self.contract.generated_token_budget is None:
            return {}
        return {
            "remaining_generated_budget": max(0, self.contract.generated_token_budget - self.state.out_tokens),
            "remaining_observation_budget": max(0, self.contract.observation_token_budget - self.state.obs_tokens),
        }

    def render_budget_observation(self, response: dict, *, role: str = "tool") -> str:
        """Render and charge one budget-aware observation for its actual message role.

        The internal total marker means a safe lower bound after this body and its role
        framing are charged. Split quotas expose only their independent balances;
        combined quotas expose tokens_left. A self-consistent marker is exact. Tokenizer boundary or
        phase changes can make the marker equation oscillate, so the bounded search picks
        only a measured candidate that does not exceed its own resulting balance; if no
        candidate is safe, zero is rendered and charged. Trial renderings never mutate
        episode state, and the selected body is charged exactly once.
        """
        if not isinstance(response, dict):
            raise TypeError("response must be a dict")
        if role == "tool":
            overhead = self.observation_overhead
        elif role == "user":
            overhead = self.feedback_overhead
        else:
            raise ValueError("role must be 'tool' or 'user'")

        remaining_before = max(0, self.token_budget - self.tokens_used())
        measurements: dict[int, tuple[str, int, int]] = {}
        exact_marker: int | None = None

        def measure(marker: int) -> tuple[str, int, int]:
            # Budget metadata is part of the body, so phase and closing guidance take
            # part in the same self-referential tokenizer calculation as tokens_left.
            body = dict(response)
            for field_name in ("tokens_left", "budget_policy", "budget_phase", "budget_guidance",
                               "generated_tokens_left", "observation_tokens_left"):
                body.pop(field_name, None)
            fields = budget_observation_fields(marker, self.token_budget)
            if fields.get("budget_policy") != BUDGET_POLICY_ID:
                raise RuntimeError("budget observation policy identity mismatch")
            fields.pop("budget_policy")  # Static identity is in the episode contract/prompt.
            body.update(fields)
            if self.contract.generated_token_budget is not None:
                generated_left = max(0, self.contract.generated_token_budget - self.state.out_tokens)
                split_fields = budget_observation_fields(generated_left, self.contract.generated_token_budget)
                split_fields.pop("budget_policy")
                body.update(split_fields)
                body["generated_tokens_left"] = generated_left
                body["observation_tokens_left"] = max(0, marker - generated_left)
            else:
                body["tokens_left"] = marker
            text = self._render_observation(body)
            charged = overhead + self._count_text(text)
            actual_remaining = max(0, remaining_before - charged)
            return text, charged, actual_remaining

        marker = remaining_before
        for _ in range(_BUDGET_MARKER_MAX_ITERATIONS):
            if marker in measurements:
                break
            measurement = measure(marker)
            measurements[marker] = measurement
            _text, _charged, actual_remaining = measurement
            if actual_remaining == marker:
                exact_marker = marker
                break
            # A repeat means the marker/phase/guidance mapping is in a cycle. The
            # measured candidates below provide a bounded, fail-closed resolution.
            if actual_remaining in measurements:
                break
            marker = actual_remaining

        safe_markers = [candidate for candidate, (_, _, actual) in measurements.items()
                        if candidate <= actual]
        if exact_marker is not None:
            selected = exact_marker
        elif safe_markers:
            selected = max(safe_markers)
        else:
            # No fixed point or safe candidate was found within the iteration bound.
            # Zero is always a conservative lower bound, and this final body is still
            # fully counted with its phase and guidance before committing the charge.
            if 0 not in measurements:
                measurements[0] = measure(0)
            selected = 0

        text, charged, actual_remaining = measurements[selected]
        if selected > actual_remaining:
            raise AssertionError("selected budget marker exceeds its measured remaining balance")
        if (self.contract.observation_token_budget is not None
                and charged > self.contract.observation_token_budget - self.state.obs_tokens):
            # Keep the complete tool response in the audit, but never deliver or
            # charge an observation beyond its independent quota. Do not truncate JSON.
            self.state.observation_budget_blocked = True
            return ""
        self.state.obs_tokens += charged
        return text

    def _check_leak(self, response: dict, view: dict) -> bool:
        """Fail if a target grid shows up unless the episode produced it itself.

        The policy may legitimately see a grid it just built (its own artifact is
        echoed back); it must never see the held target through a task input or any
        other pre-existing source. Returns True when the hit was the episode's own.
        """
        if not isinstance(view, dict):
            raise TypeError("leak checks require decoded structured observations")
        text = json.dumps(view, ensure_ascii=False, separators=(",", ":"))
        hit = [t for t, fp in self._target_fingerprints.items() if fp in text]
        if not hit:
            return False
        if all(self._target_fingerprints[t] in self._visible_grid_fingerprints for t in hit):
            self.state.visible_target_matches += 1
            return True
        referenced = {response.get("artifact_id"), response.get("source")}
        referenced.update(response.get("parents") or [])
        if referenced & self.state.owned_artifacts:
            self.state.echoed_targets += 1
            return True
        raise AssertionError(f"target {hit} leaked into an observation: {text[:200]}")

    def score_attempts(self) -> dict[str, dict[int, bool]]:
        """Compare each occupied attempt privately against its own held target."""
        scored: dict[str, dict[int, bool]] = {}
        for target, expected in self.targets.items():
            scored[target] = {}
            for attempt, artifact in self.state.submissions.get(target, {}).items():
                try:
                    predicted = self.session.read(artifact).data
                    scored[target][attempt] = bool(
                        isinstance(predicted, list) and len(predicted) == len(expected)
                        and all(isinstance(r, list) and len(r) == len(e)
                                and all(int(a) == int(b) for a, b in zip(r, e))
                                for r, e in zip(predicted, expected)))
                except Exception:
                    scored[target][attempt] = False
        return scored

    def score_targets(self) -> tuple[bool, dict[str, bool]]:
        """A target succeeds when either of its occupied attempts matches."""
        scored = {target: any(attempts.values()) for target, attempts in self.score_attempts().items()}
        return all(scored.values()) and bool(scored), scored

    def _final_reward(self) -> tuple[float, dict]:
        """Exact-match gate only; the efficiency term is added by the group ranker."""
        exact, per_target = self.score_targets()
        return (1.0 if exact else 0.0), {
            "exact": exact, "per_target": per_target,
            "calls_used": len(self.state.lines), "out_tokens": self.state.out_tokens,
            "obs_tokens": self.state.obs_tokens, "tokens_used": self.tokens_used(),
            "cost": round(self.cost(), 1), "token_budget": self.token_budget,
            "aborted": self.state.aborted, "abort_reason": self.state.abort_reason,
            "contract": self.contract.as_dict(),
            "contract_fingerprint": self.contract.fingerprint(),
        }

    # ------------------------------------------------------------------- api
    @property
    def catalog(self) -> dict:
        return self.session.catalog()

    def reset(self) -> str:
        if self.session is not None:
            self.close()
        self.session = self._Session.create(self.workdir, self.task_payload,
                                            task_id=self.task_id, max_calls=self.contract.call_cap)
        self.training_verification = TrainingVerification(self.required_demo_verification_targets)
        self.state = EpisodeState(
            started_at=time.time(),
            latest_demo_verification={
                target: None for target in self.required_demo_verification_targets
            },
        )
        validate_prompt_catalog(self.catalog)      # validate the advertised tool signatures
        prompt = user_message(self.challenge, self.instruction, input_roles=self.input_roles,
                              submission_targets=self.submission_targets or None,
                              conventions=self.conventions)
        # The prompt carries the visible task JSON.  A held target may legitimately coincide
        # with a *visible* demonstration output (ARC reuses grids), so only a target that is
        # not among the visible outputs counts as a leak.
        leaked = [t for t, fp in self._target_fingerprints.items()
                  if fp not in self._visible_grid_fingerprints
                  and (fp in prompt or json.dumps(grid_rows(self.targets[t]),
                       separators=(",", ":")) in prompt)]
        if leaked:
            raise AssertionError(f"target {leaked} leaked into the episode prompt")
        return prompt

    def step(self, action: dict[str, Any]) -> StepResult:
        if self.session is None:
            raise RuntimeError("step() before reset()")
        response = self.session.execute_action(action)
        self.state.lines.append(action)
        self.state.responses.append(response)
        self.training_verification.observe(action, response, len(self.state.lines))
        status = response.get("status")
        if status != "ok":
            self.state.errors += 1
        else:
            self.state.successful_tool_calls += 1

        info: dict[str, Any] = {"status": status, "calls_used": len(self.state.lines),
                                "error": response.get("error") if status != "ok" else None}
        if action["action"] == "submit" and status == "ok":
            target = response["summary"]["target"]
            # the runtime returns a Submission *record*; the predicted grid is the
            # artifact named by the line's `source` argument.
            source = action["args"].get("source")
            if target and source:
                attempt = response["summary"]["attempt"]
                self.state.submissions.setdefault(target, {})[attempt] = source
        if action["action"] == "verify_example" and status == "ok":
            target = response.get("summary", {}).get("target")
            exact_match = response.get("summary", {}).get("exact_match")
            self.state.verified += 1
            if exact_match is True:
                self.state.exact_train += 1
            if target in self.state.latest_demo_verification and type(exact_match) is bool:
                self.state.latest_demo_verification[target] = exact_match

        if status == "ok":
            results = response["items"] if response.get("type") == "ObservationBatch" else [response]
            self.state.owned_artifacts.update(result["artifact_id"] for result in results
                                             if result.get("artifact_id"))

        # Generation is charged once by record_generation() before this action.
        # Charge only the real tool observation here.
        # Audit/scoring keep the complete runtime response. The model sees the
        # toolkit's compact view; budget framing is added and charged afterward.
        load_toolkit_runtime()
        from arc_toolkit.model_observation import OBSERVATION_SCHEMA, project_response
        from arc_toolkit.model_wire import unpack_observation

        model_response = project_response(response)
        # Scan exactly the model-visible fields, restoring packed grids first.
        # Audit-only resolved_params can echo the policy's own guesses; they are
        # retained in state.responses but never delivered as observations.
        leak_view = unpack_observation(model_response, OBSERVATION_SCHEMA)
        info["own_artifact_echo"] = self._check_leak(
            response, leak_view,
        )
        if response.get("type") == "Image":
            # The Qwen rollout transport accepts text observations only. Keep the
            # returned JSON image metadata visible while stating that no image was
            # attached to the model context.
            model_response["image_delivery"] = {
                "status": "not_attached",
                "reason": "Host route does not support native tool images. PNG saved; model received no image.",
            }
        observation = self.render_budget_observation(model_response)
        info["observation_schema"] = OBSERVATION_SCHEMA

        done = False
        reward = 0.0
        if len(self.state.submissions) == len(self.targets) and self.targets:
            reward, scoring = self._final_reward()
            info.update(scoring)
            # Submission records a prediction. The model can inspect or replace it
            # before ending its episode with a normal assistant completion.
        if self.budget_exhausted():
            info.update({"exact": False, "reason": "token budget exhausted"})
            done = True
        elif len(self.state.lines) >= self.contract.call_cap:
            self.state.aborted, self.state.abort_reason = True, "call cap (harness safety net)"
            info.update({"exact": False, "reason": self.state.abort_reason})
            done = True
        elif (self.contract.abort_seconds
              and time.time() - self.state.started_at > self.contract.abort_seconds):
            self.state.aborted = True
            self.state.abort_reason = "wall clock abort (infrastructure, not a task failure)"
            info.update({"exact": False, "reason": self.state.abort_reason})
            done = True
        info["tokens_used"] = self.tokens_used()
        info["tokens_left"] = max(0, self.token_budget - self.tokens_used())
        info["cost"] = round(self.cost(), 1)
        info["aborted"] = self.state.aborted
        info["contract_fingerprint"] = self.contract.fingerprint()
        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def workflow_coverage(self) -> dict[str, Any]:
        """Return public workflow coverage without consulting held-target scoring."""
        verification = {
            target: self.state.latest_demo_verification.get(target)
            for target in self.required_demo_verification_targets
        }
        submissions = {
            target: target in self.state.submissions
            for target in self.required_submission_targets
        }
        complete = (all(value is True for value in verification.values())
                    and all(submissions.values()))
        return {
            "latest_demo_verification": verification,
            "required_submissions": submissions,
            "submitted_targets": [target for target, submitted in submissions.items() if submitted],
            "missing_submissions": [target for target, submitted in submissions.items() if not submitted],
            "workflow_complete": complete,
            "training_export_ready": False,
            "training_export_gate": "independent exporter gate not run",
        }

    def close(self) -> dict:
        """Detach from the session (files stay for audit)."""
        summary = {"task_id": self.task_id, "lines": len(self.state.lines),
                   "successful_tool_calls": self.state.successful_tool_calls,
                   "errors": self.state.errors, "submissions": dict(self.state.submissions),
                   "verified": self.state.verified, "exact_train": self.state.exact_train,
                   "workflow": self.workflow_coverage(),
                   "training_verification": self.training_verification.summary(),
                   "echoed_targets": self.state.echoed_targets,
                   "visible_target_matches": self.state.visible_target_matches,
                   "root": str(self.workdir),
                   "out_tokens": self.state.out_tokens, "obs_tokens": self.state.obs_tokens,
                   "tokens_used": self.tokens_used(), "cost": round(self.cost(), 1),
                   "token_budget": self.token_budget, "aborted": self.state.aborted,
                   "abort_reason": self.state.abort_reason,
                   "token_counter": self.token_counter_name,
                   "out_count_mode": self.state.out_count_mode,
                   "contract": self.contract.as_dict(),
                   "contract_fingerprint": self.contract.fingerprint(),
                   "toolkit_runtime_sha256": self.toolkit_sha256}
        self.session = None
        return summary
