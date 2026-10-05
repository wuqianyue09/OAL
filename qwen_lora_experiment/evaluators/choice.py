"""Continuation-only likelihood scoring for HellaSwag choices."""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from .common import (
    ModelForward,
    encode_context_continuation,
    ensure_sequence_limit,
    evaluation_mode,
    forward_logits,
    resolve_device,
)


@dataclass(frozen=True)
class ContinuationScore:
    index: int
    text: str
    context_token_count: int
    full_token_count: int
    continuation_token_ids: tuple[int, ...]
    token_count: int
    total_log_likelihood: float
    mean_log_likelihood: float

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "context_token_count": self.context_token_count,
            "full_token_count": self.full_token_count,
            "continuation_token_ids": list(self.continuation_token_ids),
            "token_count": self.token_count,
            "total_log_likelihood": self.total_log_likelihood,
            "mean_log_likelihood": self.mean_log_likelihood,
        }


def score_continuation(
    *,
    model: object,
    tokenizer: object,
    context: str,
    continuation: str,
    candidate_index: int,
    sequence_limit: int,
    kernel_bank: object | None = None,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
) -> ContinuationScore:
    """Score one continuation using harness-compatible causal tokenization."""
    if type(candidate_index) is not int or candidate_index < 0:
        raise ValueError("candidate_index must be a non-negative integer")
    encoded = encode_context_continuation(
        tokenizer, context, continuation, label=f"candidate {candidate_index}"
    )
    if not encoded.continuation_token_ids:
        raise ValueError(f"candidate {candidate_index} has no continuation tokens")
    ensure_sequence_limit(
        len(encoded.full_token_ids),
        sequence_limit,
        label=f"candidate {candidate_index}",
    )
    resolved_device = resolve_device(model, device)
    input_ids = torch.tensor(
        [encoded.full_token_ids], dtype=torch.long, device=resolved_device
    )
    with evaluation_mode(model, kernel_bank), torch.inference_mode():
        logits = forward_logits(
            model=model,
            input_ids=input_ids,
            kernel_bank=kernel_bank,
            model_forward=model_forward,
            label=f"candidate {candidate_index}",
        )
        start = len(encoded.context_token_ids) - 1
        prediction_logits = logits[:, start : input_ids.shape[1] - 1, :]
        targets = torch.tensor(
            [encoded.continuation_token_ids], dtype=torch.long, device=resolved_device
        )
        if targets.max().item() >= prediction_logits.shape[-1]:
            raise ValueError("continuation token is outside the model vocabulary")
        log_probs = F.log_softmax(prediction_logits.float(), dim=-1)
        likelihoods = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        total = float(likelihoods.sum().item())
    if not math.isfinite(total):
        raise ValueError(f"candidate {candidate_index} likelihood is not finite")
    token_count = len(encoded.continuation_token_ids)
    return ContinuationScore(
        index=candidate_index,
        text=encoded.continuation,
        context_token_count=len(encoded.context_token_ids),
        full_token_count=len(encoded.full_token_ids),
        continuation_token_ids=encoded.continuation_token_ids,
        token_count=token_count,
        total_log_likelihood=total,
        mean_log_likelihood=total / token_count,
    )


def choose_first(values: object) -> int:
    iterator = iter(values)
    try:
        best = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot choose from an empty score sequence") from exc
    if not isinstance(best, float) or not math.isfinite(best):
        raise ValueError("choice scores must be finite floats")
    best_index = 0
    for index, value in enumerate(iterator, start=1):
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError("choice scores must be finite floats")
        if value > best:
            best = value
            best_index = index
    return best_index
