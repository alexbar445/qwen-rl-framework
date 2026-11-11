"""Strict TRL fused-head bridge for vocabulary-parallel packed T2 models."""
from __future__ import annotations

import math
import types
from typing import Any

import torch
from torch import nn

from ..models.ternary import FrozenTernaryLinear
from ..models.ternary_tp_head import (
    DEFAULT_CHUNK_ROWS,
    DEFAULT_MAX_LOGITS_BYTES,
    VocabularyParallelTernaryHead,
)


class TernaryTPBridgeError(RuntimeError):
    """The model or batch cannot use the vocabulary-parallel fused-head path."""


def _base_model(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    base = get_base_model() if callable(get_base_model) else model
    if not isinstance(base, nn.Module):
        raise TypeError("T2 TP fused bridge requires a torch module or PEFT model")
    return base


def _replace_output_head(model: nn.Module, source: nn.Module,
                         replacement: VocabularyParallelTernaryHead) -> None:
    setter = getattr(model, "set_output_embeddings", None)
    if callable(setter):
        setter(replacement)
    elif getattr(model, "lm_head", None) is source:
        model.lm_head = replacement
    else:
        raise TernaryTPBridgeError(
            "model must expose set_output_embeddings() or a direct lm_head to release the full T2 head"
        )
    getter = getattr(model, "get_output_embeddings", None)
    if not callable(getter) or getter() is not replacement:
        raise TernaryTPBridgeError("replacing the dense T2 head did not update get_output_embeddings()")


def _trl_logit_parameters(config: Any) -> tuple[float, float | None]:
    text_config = config.get_text_config() if callable(getattr(config, "get_text_config", None)) else config
    final_softcap = getattr(text_config, "final_logit_softcapping", None)
    logit_scale = getattr(text_config, "logit_scale", None)
    if logit_scale is None:
        logit_scale = getattr(text_config, "output_multiplier", None)
    if logit_scale is None:
        logit_scale = getattr(text_config, "lm_head_multiplier", None)
    if logit_scale is None and getattr(text_config, "logits_scaling", None) is not None:
        scaling = text_config.logits_scaling
        logit_scale = scaling if getattr(text_config, "model_type", None) == "hyperclovax" else 1 / scaling
    return (1.0 if logit_scale is None else float(logit_scale),
            None if final_softcap is None else float(final_softcap))


def _forward_text_backbone(base, **kwargs):
    """Run the decoder with the packed-weight forward/backward ownership hooks."""
    backbone = getattr(base, "model", None)
    if not isinstance(backbone, nn.Module):
        raise TernaryTPBridgeError("model must expose its text backbone as .model")
    offloader = getattr(base, "_t2_packed_weight_offloader", None)
    if offloader is None:
        offloader = getattr(base, "_t2_tp_weight_offloader", None)
    if offloader is not None:
        if not callable(getattr(offloader, "set_phase", None)):
            raise TernaryTPBridgeError("packed-weight offloader must expose set_phase()")
        offloader.set_phase("forward")
    outputs = backbone(**kwargs)
    hidden = getattr(outputs, "last_hidden_state", None)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TernaryTPBridgeError("text backbone must return last_hidden_state shaped [batch, sequence, hidden]")
    if offloader is not None:
        if hidden.requires_grad:
            def enter_backward(gradient):
                offloader.set_phase("backward")
                return gradient
            hidden.register_hook(enter_backward)
        if not torch.is_grad_enabled():
            offloader.set_phase("idle")
    return hidden


def _t2_tp_fused_forward(
    self,
    *args,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep=0,
    fused_lm_head=False,
    shift_labels=None,
    num_items_in_batch=None,
    **kwargs,
):
    """Score causal targets from the real decoder hidden state without logits."""
    if not fused_lm_head:
        raise TernaryTPBridgeError(
            "vocabulary-parallel T2 models require fused_lm_head=True; dense logits and generation are unsupported"
        )
    if args:
        if input_ids is not None or len(args) != 1:
            raise TernaryTPBridgeError("fused T2 TP forward accepts at most one positional input_ids tensor")
        input_ids = args[0]
    if (labels is None) == (shift_labels is None):
        raise TernaryTPBridgeError("provide exactly one of labels or shift_labels to fused T2 TP forward")
    if past_key_values is not None:
        raise TernaryTPBridgeError("fused T2 TP training does not accept cached past_key_values")
    if logits_to_keep not in (0, None):
        raise TernaryTPBridgeError("logits_to_keep is not supported by selected-token T2 TP scoring")
    if input_ids is None and inputs_embeds is None:
        raise TernaryTPBridgeError("fused T2 TP forward requires input_ids or inputs_embeds")

    backbone_kwargs = dict(kwargs)
    backbone_kwargs.update(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = _forward_text_backbone(self, **backbone_kwargs)

    target_labels = labels if shift_labels is None else shift_labels
    if not isinstance(target_labels, torch.Tensor) or target_labels.dtype not in (torch.int32, torch.int64):
        raise TernaryTPBridgeError("causal labels must be an int32 or int64 tensor")
    if shift_labels is None:
        if target_labels.ndim != 2 or tuple(target_labels.shape) != tuple(hidden_states.shape[:2]):
            raise TernaryTPBridgeError("labels must match the text backbone [batch, sequence] dimensions")
        score_hidden = hidden_states[:, :-1]
        target_labels = target_labels[:, 1:]
    else:
        score_hidden = hidden_states
        if tuple(target_labels.shape) != tuple(score_hidden.shape[:2]):
            raise TernaryTPBridgeError("shift_labels must match the unshifted hidden-state dimensions")
    target_labels = target_labels.to(device=score_hidden.device)
    label_mask = target_labels != -100

    head = object.__getattribute__(self, "_t2_tp_vocabulary_head")
    logit_scale = object.__getattribute__(self, "_t2_tp_logit_scale")
    softcap = object.__getattribute__(self, "_t2_tp_softcap")
    temperature = object.__getattribute__(self, "_t2_tp_temperature")
    log_probs, entropy = head.score(
        score_hidden,
        target_labels,
        target_mask=label_mask,
        temperature=temperature,
        logit_scale=logit_scale,
        softcap=softcap,
    )
    denominator = label_mask.sum().clamp(min=1) if num_items_in_batch is None else num_items_in_batch
    loss = -log_probs.sum() / denominator
    from trl.trainer.utils import FusedCausalLMOutput

    return FusedCausalLMOutput(
        loss=loss,
        log_probs=log_probs,
        entropy=entropy,
        label_mask=label_mask,
    )


def install_t2_tp_fused_head_bridge(
    model: nn.Module,
    temperature: float = 1.0,
    *,
    group: Any = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    max_logits_bytes: int = DEFAULT_MAX_LOGITS_BYTES,
    scoring_backend: str = "trl",
) -> dict[str, Any]:
    """Shard and replace a frozen T2 output head, then install strict TRL scoring.

    The full source head is replaced in the model and not retained.  Training
    calls must pass ``fused_lm_head=True`` and exactly one of ``labels`` or
    pre-shifted ``shift_labels``.  Generation/dense-logit forwards fail clearly.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("T2 TP fused bridge requires a torch module")
    if type(temperature) not in (int, float) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("T2 TP fused bridge temperature must be finite and greater than zero")
    if scoring_backend not in ("trl", "torch"):
        raise ValueError("T2 TP scoring_backend must explicitly be 'trl' or the CPU 'torch' oracle")
    base_model = _base_model(model)
    current_forward = base_model.forward
    if getattr(current_forward, "__func__", None) is _t2_tp_fused_forward:
        installed_temperature = object.__getattribute__(base_model, "_t2_tp_temperature")
        installed_head = object.__getattribute__(base_model, "_t2_tp_vocabulary_head")
        if installed_temperature != float(temperature):
            raise TernaryTPBridgeError("T2 TP fused bridge is already installed with a different temperature")
        if installed_head.scoring_backend != scoring_backend:
            raise TernaryTPBridgeError("T2 TP fused bridge is already installed with a different scoring backend")
        return {
            "installed": True,
            "idempotent_reuse": True,
            "head": "vocabulary_sharded_packed_t2",
            "temperature": installed_temperature,
            "scoring_backend": installed_head.scoring_backend,
            "full_vocab_logits_materialized": False,
        }

    getter = getattr(base_model, "get_output_embeddings", None)
    if not callable(getter):
        raise TernaryTPBridgeError("T2 TP model must expose get_output_embeddings()")
    output_head = getter()
    if isinstance(output_head, VocabularyParallelTernaryHead):
        head = output_head
    elif type(output_head) is FrozenTernaryLinear:
        head = VocabularyParallelTernaryHead.from_source(
            output_head,
            group=group,
            chunk_rows=chunk_rows,
            max_logits_bytes=max_logits_bytes,
            scoring_backend=scoring_backend,
        )
        _replace_output_head(base_model, output_head, head)
    else:
        raise TernaryTPBridgeError(
            "output embeddings must be FrozenTernaryLinear or VocabularyParallelTernaryHead"
        )
    if head.scoring_backend != scoring_backend:
        raise TernaryTPBridgeError(
            f"output head backend is {head.scoring_backend!r}, expected {scoring_backend!r}"
        )

    config = getattr(base_model, "config", None)
    if config is None:
        raise TernaryTPBridgeError("T2 TP model must expose its text config")
    logit_scale, softcap = _trl_logit_parameters(config)
    if not math.isfinite(logit_scale) or (softcap is not None and (not math.isfinite(softcap) or softcap <= 0)):
        raise TernaryTPBridgeError("model config contains invalid TRL logit scale or softcap")
    object.__setattr__(base_model, "_t2_tp_vocabulary_head", head)
    object.__setattr__(base_model, "_t2_tp_logit_scale", logit_scale)
    object.__setattr__(base_model, "_t2_tp_softcap", softcap)
    object.__setattr__(base_model, "_t2_tp_temperature", float(temperature))
    # The prior single-rank TRL adapter retains the original output module.
    # Drop those references after replacing the registered lm_head so each
    # TP rank keeps only its packed vocabulary shard.
    for stale_name in (
        "_ternary_fused_head_adapter",
        "_pipeline_trl_fused_head",
        "_ternary_fused_head_bridge_original_forward",
    ):
        if hasattr(base_model, stale_name):
            delattr(base_model, stale_name)
    base_model._t2_tp_model = True
    model._t2_tp_model = True
    base_model.forward = types.MethodType(_t2_tp_fused_forward, base_model)
    return {
        "installed": True,
        "idempotent_reuse": False,
        "head": "vocabulary_sharded_packed_t2",
        "world_size": head.tp_world_size,
        "rank": head.tp_rank,
        "local_vocab_start": head.local_vocab_start,
        "local_vocab_size": head.local_vocab_size,
        "chunk_rows": head.chunk_rows,
        "max_logits_bytes": head.max_logits_bytes,
        "scoring_backend": head.scoring_backend,
        "temperature": float(temperature),
        "logit_scale": logit_scale,
        "softcap": softcap,
        "full_vocab_logits_materialized": False,
        "full_source_head_retained": False,
    }


__all__ = ["TernaryTPBridgeError", "install_t2_tp_fused_head_bridge"]
