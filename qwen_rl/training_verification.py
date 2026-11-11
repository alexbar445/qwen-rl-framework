"""Environment-owned training verification evidence; never reads private answers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

VERIFICATION_SCHEMA = "training-verification/v1"


@dataclass
class TrainingVerification:
    targets: tuple[str, ...]
    artifacts: dict[str, dict] = field(default_factory=dict)
    latest: dict[str, dict] = field(default_factory=dict)
    first_submission_call: int | None = None

    def _uses_output(self, reference: str) -> bool:
        pending, seen = [reference], set()
        while pending:
            reference = pending.pop()
            if re.fullmatch(r"train/[0-9]+/output", reference):
                return True
            if re.fullmatch(r"(?:train|test)/[0-9]+/input", reference):
                continue
            match = re.fullmatch(r"artifacts/(a[0-9]{4,})/data\.json", reference)
            reference = match.group(1) if match else reference
            if reference not in self.artifacts:
                return True  # Unknown provenance earns no bonus.
            if reference in seen:
                continue
            seen.add(reference)
            artifact = self.artifacts[reference]
            pending.extend(artifact.get("parents", []))
            pending.extend(artifact.get("input_references", []))
        return False

    def observe(self, action: dict, response: dict, call: int) -> None:
        if response.get("status") == "ok":
            results = response["items"] if response.get("type") == "ObservationBatch" else [response]
            for result in results:
                if result.get("artifact_id"):
                    self.artifacts[result["artifact_id"]] = result
        if self.first_submission_call is not None:
            return
        name, args = action["action"], action["args"]
        if name == "submit" and response.get("status") == "ok":
            self.first_submission_call = call
        elif name == "verify_example" and args.get("target") in self.targets:
            target, prediction = args["target"], args.get("prediction")
            summary = response.get("summary", {})
            eligible = (response.get("status") == "ok"
                        and summary.get("target") == target
                        and summary.get("exact_match") is True
                        and isinstance(prediction, str)
                        and not self._uses_output(prediction))
            self.latest[target] = {"call": call, "prediction": prediction,
                "artifact_id": response.get("artifact_id"), "status": response.get("status"),
                "exact_match": summary.get("exact_match"), "eligible": bool(eligible)}

    def summary(self) -> dict[str, Any]:
        verified = [target for target in self.targets if self.latest.get(target, {}).get("eligible")]
        return {"schema": VERIFICATION_SCHEMA, "total": len(self.targets),
                "verified_targets": verified,
                "coverage": len(verified) / len(self.targets) if self.targets else 0.0,
                "first_submission_call": self.first_submission_call,
                "latest": {target: dict(event) for target, event in self.latest.items()}}
