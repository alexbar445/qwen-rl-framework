"""Identity-bound provider API for constructing frozen T2 Qwen3.5 modules."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import torch
from torch import nn

from qwen_rl.contracts.ternary import (
    TERNARY_QG_LAYOUT_VERSION,
    TernaryArtifactIdentity,
    artifact_report,
    load_artifact_identity,
)

from .ternary import install_lora_hadamard_inputs
from .ternary_artifacts import TernaryArtifactError, TernaryArtifactStore


class TernaryModelProvider:
    """Bind tensor reads to the exact artifact SHA and the T2 learner contract."""

    def __init__(
        self,
        identity_path: str | Path,
        *,
        verify_files: bool = True,
    ) -> None:
        self.identity: TernaryArtifactIdentity = load_artifact_identity(
            identity_path, verify_files=verify_files
        )
        source = self.identity.document["source"]["ninfer_artifact"]
        self.path = Path(source["path"])
        self.store = TernaryArtifactStore(self.path)
        if self.store.artifact_id.hex() != self.identity.artifact_id:
            self.store.close()
            raise TernaryArtifactError("NInfer header artifact ID differs from the verified identity")
        self.text_config = dict(self.store.text_config)
        model = self.identity.document["model"]
        geometry = model["geometry"]
        if (self.text_config.get("architectures") != [model.get("architecture")]
                or self.text_config.get("model_type") != model.get("model_type")
                or self.text_config.get("tie_word_embeddings") is not False
                or model.get("layer_types") != self.text_config.get("layer_types")
                or any(self.text_config.get(key) != value for key, value in geometry.items())):
            self.store.close()
            raise TernaryArtifactError("NInfer text config differs from the verified artifact identity")
        rotation = self.identity.document["rotation"]
        self._input_basis_by_parameter = dict(rotation.get("input_basis_by_parameter", {}))
        adapter = self.identity.document["learner"]["adapter"]
        self._input_basis_by_module = dict(adapter.get("input_basis_by_module", {}))
        if self.identity.document["model"]["id"] == "coder390-t2-full-scales-step512":
            actual_t2_parameters = set()
            for logical_name, binding in self.store.bindings.items():
                if not logical_name.startswith("text/"):
                    continue
                try:
                    parts = self.store._parts(logical_name)
                except TernaryArtifactError:
                    continue
                formats = {obj.get("format") for obj, _, _ in parts}
                if "t2_g128_fp16" in formats:
                    if formats != {"t2_g128_fp16"}:
                        self.store.close()
                        raise TernaryArtifactError(
                            f"NInfer text binding mixes T2 and non-T2 storage: {logical_name}"
                        )
                    actual_t2_parameters.add(logical_name)
            expected_t2_parameters = set(
                self.identity.document["representation"]["t2_logical_parameters"]
            )
            actual_t2_object_count = sum(
                obj.get("kind") == "tensor" and obj.get("format") == "t2_g128_fp16"
                for obj in self.store._objects.values()
            )
            if (actual_t2_parameters != expected_t2_parameters
                    or len(actual_t2_parameters) != 514
                    or actual_t2_object_count != 322):
                self.store.close()
                raise TernaryArtifactError(
                    "NInfer text T2 inventory differs from the verified Coder390 identity"
                )
            observed = {}
            for logical_name, declared_basis in self._input_basis_by_parameter.items():
                use_name = "text/output_head" if logical_name == "text/token_embedding" else logical_name
                try:
                    actual_basis = self.store.input_basis(use_name)
                except Exception as exc:
                    self.store.close()
                    raise TernaryArtifactError(
                        f"NInfer header has no valid input basis for {logical_name}"
                    ) from exc
                if actual_basis != declared_basis:
                    self.store.close()
                    raise TernaryArtifactError(
                        f"NInfer header input basis differs from identity for {logical_name}"
                    )
                observed[logical_name] = actual_basis
            if observed != self._input_basis_by_parameter:
                self.store.close()
                raise TernaryArtifactError("NInfer per-Use basis map differs from the verified identity")

    @property
    def artifact_id(self) -> bytes:
        return self.store.artifact_id

    @property
    def artifact_id_hex(self) -> str:
        return self.identity.artifact_id

    @property
    def payload_bytes(self) -> int:
        return self.store.payload_bytes

    @property
    def artifact_sha256(self) -> str:
        return self.identity.artifact_sha256

    @property
    def input_basis_by_module(self) -> dict[str, str]:
        """Effective input basis for the default PEFT targets, before PEFT prefixes are added."""
        return dict(self._input_basis_by_module)

    @property
    def input_basis_by_parameter(self) -> dict[str, str]:
        """NInfer Use input basis for each logical text T2 parameter."""
        return dict(self._input_basis_by_parameter)

    def input_basis(self, logical_name: str) -> str:
        """Return a logical parameter's effective input basis from its NInfer Uses."""
        use_name = "text/output_head" if logical_name == "text/token_embedding" else logical_name
        basis = self.store.input_basis(use_name)
        return "prism_rotated" if basis == "hadamard1024_signed" else basis

    def load_embedding(
        self,
        logical_name: str,
        *,
        expected_shape: tuple[int, int],
    ) -> nn.Module:
        return self.store.load_embedding(logical_name, expected_shape=expected_shape)

    def load_linear(
        self,
        logical_name: str,
        *,
        expected_shape: tuple[int, int],
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        use_name = "text/output_head" if logical_name == "text/token_embedding" else logical_name
        basis = self.store.input_basis(use_name)
        module = self.store.load_linear(
            logical_name,
            expected_shape=expected_shape,
            compute_dtype=compute_dtype,
            primal=basis == "primal",
        )
        module._t2_input_basis = "primal" if basis == "primal" else "prism_rotated"
        return module

    def load_fused_linear(
        self,
        logical_names: Sequence[str],
        *,
        expected_shape: tuple[int, int],
        layout: str,
        head_size: int | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        bases = {
            self.store.input_basis("text/output_head" if name == "text/token_embedding" else name)
            for name in logical_names
        }
        if len(bases) != 1:
            raise TernaryArtifactError("fused T2 projections use different input bases")
        basis = bases.pop()
        module = self.store.load_fused_linear(
            logical_names,
            expected_shape=expected_shape,
            layout=layout,
            head_size=head_size,
            compute_dtype=compute_dtype,
            primal=basis == "primal",
        )
        module._t2_input_basis = "primal" if basis == "primal" else "prism_rotated"
        return module

    def read_direct_parameter(
        self,
        logical_name: str,
        *,
        expected_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.store.read_direct_parameter(
            logical_name,
            expected_shape=expected_shape,
            dtype=dtype,
        )

    def install_lora_hadamard_inputs(self, model: nn.Module) -> int:
        return install_lora_hadamard_inputs(model)

    def text_resource_hashes(self) -> dict[str, dict[str, Any]]:
        return self.store.text_resource_hashes()

    def materialize_text_resources(self, destination: str | Path) -> dict[str, dict[str, Any]]:
        return self.store.materialize_text_resources(destination)

    def evidence(self) -> dict[str, Any]:
        document = self.identity.document
        representation = document["representation"]
        source = document["source"]
        raw_gguf = source.get("raw_gguf", {"path": None, "status": "not_recorded", "sha256": None})
        report = artifact_report(self.identity)
        evidence = {
            # The actual hash is the identity of the frozen, runnable base. The
            # separate raw_gguf status below preserves the weaker source lineage.
            "base_fingerprint": self.artifact_sha256,
            "artifact_fingerprint": self.artifact_sha256,
            "quantization": {
                "mode": "ternary",
                "bits": 2,
                "packed_resident": True,
                "compute_dtype": "bfloat16",
                "format": representation["format"],
                "layout": representation["layout"],
                "group_size_k": representation["group_size_k"],
                "t2_object_count": representation["t2_object_count"],
                "t2_logical_parameter_count": representation["t2_logical_parameter_count"],
                "artifact_id": self.artifact_id_hex,
                "artifact_sha256": self.artifact_sha256,
                "artifact_sha256_verified": False,
                "lora_input_basis": document["learner"]["adapter"]["input_basis"],
                **({"gdn_projections": True} if document["learner"]["adapter"].get("gdn_projections", False) else {}),
                "qg_layout": TERNARY_QG_LAYOUT_VERSION,
            },
            "artifact_identity": {
                "schema_version": document["schema_version"],
                "model_id": document["model"]["id"],
                "artifact_id": self.artifact_id_hex,
                "artifact_sha256": self.artifact_sha256,
                "artifact_report_sha256": report["sha256"],
                "artifact_report_kind": report.get("report_kind", "ninfer_conversion_report"),
                "raw_gguf": dict(raw_gguf),
                "mtp_checkpoint": {
                    "source_status": document["mtp_checkpoint"]["source_status"],
                    "local_copy": document["mtp_checkpoint"]["local_copy"],
                    "role": document["mtp_checkpoint"]["role"],
                    "is_lora_adapter": False,
                },
            },
            "tokenizer_files": self.text_resource_hashes(),
            "base_frozen": True,
            "lora_format": "standard_peft_lora",
        }
        if self._input_basis_by_module:
            evidence["lora_input_basis_by_module"] = self.input_basis_by_module
        if "conversion_report" in source:
            evidence["artifact_identity"]["conversion_report_sha256"] = report["sha256"]
        return evidence

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> TernaryModelProvider:  # noqa: PYI034 -- typing.Self is unavailable on supported Python 3.10
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["TernaryModelProvider"]
