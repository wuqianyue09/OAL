"""Deterministic full-prefix generation for the GSM8K evaluator."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import torch
from .common import (
    ModelForward,
    ensure_sequence_limit,
    evaluation_mode,
    forward_logits,
    resolve_device,
    tokenize_ids,
)


@dataclass(frozen=True)
class GenerationResult:
    prompt_token_count: int
    generated_token_ids: tuple[int, ...]
    text: str
    raw_text: str
    stop_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "prompt_token_count": self.prompt_token_count,
            "generated_token_ids": list(self.generated_token_ids),
            "text": self.text,
            "raw_text": self.raw_text,
            "stop_reason": self.stop_reason,
        }


def greedy_generate_full_prefix(
    *,
    model: object,
    tokenizer: object,
    prompt: str,
    max_new_tokens: int,
    sequence_limit: int,
    stop_sequences: Sequence[str],
    kernel_bank: object | None = None,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
    eos_token_id: int | None = None,
) -> GenerationResult:
    """Generate greedily with B=1, the complete prefix, and no KV cache."""
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    stops = _validate_stop_sequences(stop_sequences)
    prompt_ids = tokenize_ids(tokenizer, prompt, label="generation prompt")
    ensure_sequence_limit(
        len(prompt_ids) + max_new_tokens,
        sequence_limit,
        label="generation prompt plus budget",
    )
    resolved_eos = (
        getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None
        else eos_token_id
    )
    if resolved_eos is not None and (type(resolved_eos) is not int or resolved_eos < 0):
        raise ValueError("eos_token_id must be None or a non-negative integer")
    resolved_device = resolve_device(model, device)
    generated: list[int] = []
    raw_text = ""
    visible_text = ""
    stop_reason = "max_new_tokens"
    with evaluation_mode(model, kernel_bank), torch.inference_mode():
        for _ in range(max_new_tokens):
            prefix = prompt_ids + tuple(generated)
            input_ids = torch.tensor([prefix], dtype=torch.long, device=resolved_device)
            logits = forward_logits(
                model=model,
                input_ids=input_ids,
                kernel_bank=kernel_bank,
                model_forward=model_forward,
                label="greedy generation",
            )
            token_id = int(torch.argmax(logits[0, -1, :]).item())
            generated.append(token_id)
            raw_text = _decode(tokenizer, generated)
            visible_text = raw_text
            if resolved_eos is not None and token_id == resolved_eos:
                stop_reason = "eos"
                break
            match = _first_stop(raw_text, stops)
            if match is not None:
                offset, stop = match
                visible_text = raw_text[:offset]
                stop_reason = f"stop:{stop}"
                break
    return GenerationResult(
        prompt_token_count=len(prompt_ids),
        generated_token_ids=tuple(generated),
        text=visible_text,
        raw_text=raw_text,
        stop_reason=stop_reason,
    )


def _validate_stop_sequences(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("stop_sequences must be a sequence of strings")
    stops: list[str] = []
    for index, stop in enumerate(value):
        if not isinstance(stop, str) or not stop:
            raise ValueError(f"stop_sequences[{index}] must be a non-empty string")
        if stop not in stops:
            stops.append(stop)
    return tuple(stops)


def _decode(tokenizer: object, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must provide decode() for generation")
    text = decode(token_ids, skip_special_tokens=True)
    if not isinstance(text, str):
        raise TypeError("tokenizer.decode() must return a string")
    return text


def _first_stop(text: str, stops: tuple[str, ...]) -> tuple[int, str] | None:
    matches = [(offset, stop) for stop in stops if (offset := text.find(stop)) >= 0]
    return (
        None
        if not matches
        else min(matches, key=lambda item: (item[0], stops.index(item[1])))
    )
