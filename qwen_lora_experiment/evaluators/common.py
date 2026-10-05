"""Small model, tokenizer, and sequence helpers shared by evaluators."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol
import torch
from torch import Tensor, nn
from ..attention.common import run_attention_validation_forward


class ModelForward(Protocol):

    def __call__(
        self,
        *,
        model: object,
        input_ids: Tensor,
        kernel_bank: object | None,
        use_cache: bool,
    ) -> object: ...


@dataclass(frozen=True)
class EncodedContinuation:
    context: str
    continuation: str
    context_token_ids: tuple[int, ...]
    continuation_token_ids: tuple[int, ...]
    joint_token_ids: tuple[int, ...]
    full_token_ids: tuple[int, ...]


def tokenize_ids(tokenizer: object, text: str, *, label: str) -> tuple[int, ...]:
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    encoded = tokenizer(text, add_special_tokens=False)
    if not isinstance(encoded, Mapping):
        raise TypeError(f"{label} tokenizer output must be a mapping with input_ids")
    raw_ids = encoded.get("input_ids")
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise ValueError(f"{label} tokenizer input_ids must be a sequence")
    ids: list[int] = []
    for index, token_id in enumerate(raw_ids):
        if type(token_id) is not int or token_id < 0:
            raise ValueError(
                f"{label} tokenizer input_ids[{index}] must be a non-negative integer"
            )
        ids.append(token_id)
    if not ids:
        raise ValueError(f"{label} tokenizer input_ids must not be empty")
    return tuple(ids)


def encode_context_continuation(
    tokenizer: object, context: str, continuation: str, *, label: str
) -> EncodedContinuation:
    """Match lm-evaluation-harness causal `_encode_pair` exactly."""
    if not isinstance(context, str) or not context:
        raise ValueError(f"{label} context must be a non-empty string")
    if not isinstance(continuation, str):
        raise TypeError(f"{label} continuation must be a string")
    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    if not context:
        raise ValueError(f"{label} context must not contain only whitespace")
    context_ids = tokenize_ids(tokenizer, context, label=f"{label} context")
    joint_ids = tokenize_ids(
        tokenizer, context + continuation, label=f"{label} context plus continuation"
    )
    continuation_ids = joint_ids[len(context_ids) :]
    return EncodedContinuation(
        context=context,
        continuation=continuation,
        context_token_ids=context_ids,
        continuation_token_ids=continuation_ids,
        joint_token_ids=joint_ids,
        full_token_ids=context_ids + continuation_ids,
    )


def ensure_sequence_limit(token_count: int, sequence_limit: int, *, label: str) -> None:
    if type(sequence_limit) is not int or sequence_limit < 2:
        raise ValueError("sequence_limit must be an integer of at least two")
    if token_count > sequence_limit:
        raise ValueError(
            f"{label} requires {token_count} tokens but the sequence limit is {sequence_limit}"
        )


def resolve_device(model: object, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def forward_model(
    *,
    model: object,
    input_ids: Tensor,
    kernel_bank: object | None,
    model_forward: ModelForward | None,
) -> object:
    """Invoke a model or callback within the shared attention validation scope."""
    if model_forward is not None:
        if not callable(model_forward):
            raise TypeError("model_forward must be callable when provided")
        return run_attention_validation_forward(
            model_forward,
            model=model,
            input_ids=input_ids,
            kernel_bank=kernel_bank,
            use_cache=False,
        )
    if not callable(model):
        raise TypeError("model must be callable")
    return run_attention_validation_forward(model, input_ids=input_ids, use_cache=False)


def extract_logits(output: object, *, input_ids: Tensor, label: str) -> Tensor:
    """Validate the real, finite B=1 logits required by evaluator scoring."""
    if isinstance(output, Mapping):
        logits = output.get("logits")
    else:
        logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError(f"{label} model output must provide logits as a torch.Tensor")
    if logits.ndim != 3:
        raise ValueError(
            f"{label} logits must have shape [batch, sequence, vocabulary]"
        )
    if logits.shape[0] != 1:
        raise ValueError(f"{label} logits batch size must be exactly 1")
    if logits.shape[1] != input_ids.shape[1]:
        raise ValueError(f"{label} logits sequence length must match input_ids")
    if logits.shape[2] <= 0:
        raise ValueError(f"{label} logits vocabulary dimension must be positive")
    if not (logits.is_floating_point() or logits.is_complex()):
        raise TypeError(f"{label} logits must have floating dtype")
    if logits.is_complex():
        raise TypeError(f"{label} logits must have real floating dtype")
    if not torch.isfinite(logits).all().item():
        raise ValueError(f"{label} logits must be finite")
    return logits


def forward_logits(
    *,
    model: object,
    input_ids: Tensor,
    kernel_bank: object | None,
    model_forward: ModelForward | None,
    label: str,
) -> Tensor:
    """Run one evaluator forward and return its validated logits."""
    return extract_logits(
        forward_model(
            model=model,
            input_ids=input_ids,
            kernel_bank=kernel_bank,
            model_forward=model_forward,
        ),
        input_ids=input_ids,
        label=label,
    )


@contextmanager
def evaluation_mode(*objects: object | None):
    modules = [value for value in objects if isinstance(value, nn.Module)]
    states: list[tuple[nn.Module, bool]] = []
    seen: set[int] = set()
    for root in modules:
        for module in root.modules():
            if id(module) not in seen:
                states.append((module, module.training))
                seen.add(id(module))
    try:
        for root in modules:
            root.eval()
        yield
    finally:
        for module, was_training in states:
            module.training = was_training
