"""Task loop and equal-status exact-match metrics for GSM8K."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import torch
from ..gsm8k import extract_flexible_answer, extract_strict_answer, format_gsm8k_prompt
from .common import ModelForward, ensure_sequence_limit, tokenize_ids
from .generation import greedy_generate_full_prefix

GSM8K_MAX_NEW_TOKENS = 256
GSM8K_STOP_SEQUENCES = ("Question:", "</s>", "<|im_end|>")


@dataclass(frozen=True)
class Gsm8kScore:
    predictions: tuple[dict[str, object], ...]
    exact_match_strict: float
    exact_match_flexible: float

    @property
    def question_count(self) -> int:
        return len(self.predictions)

    def summary(self) -> dict[str, object]:
        return {
            "question_count": self.question_count,
            "exact_match_strict": self.exact_match_strict,
            "exact_match_flexible": self.exact_match_flexible,
        }


def reusable_gsm8k_score(
    *,
    predictions: Sequence[Mapping[str, object]],
    metrics: Mapping[str, object],
    expected_records: Mapping[str, Mapping[str, str]],
) -> Gsm8kScore:
    """Recompute exact-match metrics from complete, structurally valid saved rows."""
    expected_ids = tuple(expected_records)
    rows = tuple((dict(row) for row in predictions))
    if not expected_ids or tuple((row.get("id") for row in rows)) != expected_ids:
        raise ValueError("existing GSM8K sidecar IDs do not match the bundle order")
    fields = {
        "id",
        "gold_answer",
        "prompt_token_count",
        "prompt_sha256",
        "generated_token_ids",
        "text",
        "raw_text",
        "strict_answer",
        "flexible_answer",
        "strict_correct",
        "flexible_correct",
        "stop_reason",
    }
    strict_correct_count = 0
    flexible_correct_count = 0
    valid_stop_reasons = {
        "max_new_tokens",
        "eos",
        *(f"stop:{stop}" for stop in GSM8K_STOP_SEQUENCES),
    }
    for row_index, row in enumerate(rows):
        if set(row) != fields:
            raise ValueError(
                f"existing GSM8K prediction row {row_index} fields are invalid"
            )
        identifier = row["id"]
        assert isinstance(identifier, str)
        expected_record = expected_records[identifier]
        gold = row["gold_answer"]
        if gold != expected_record.get("gold_answer"):
            raise ValueError(
                f"existing GSM8K prediction row {row_index} gold answer is invalid"
            )
        prompt_count = row["prompt_token_count"]
        prompt_digest = row["prompt_sha256"]
        token_ids = row["generated_token_ids"]
        text = row["text"]
        raw_text = row["raw_text"]
        if type(prompt_count) is not int or prompt_count <= 0:
            raise ValueError(
                f"existing GSM8K prediction row {row_index} prompt count is invalid"
            )
        if (
            not isinstance(prompt_digest, str)
            or len(prompt_digest) != 64
            or any((character not in "0123456789abcdef" for character in prompt_digest))
        ):
            raise ValueError(
                f"existing GSM8K prediction row {row_index} prompt digest is invalid"
            )
        if prompt_digest != expected_record.get("prompt_sha256"):
            raise ValueError(
                f"existing GSM8K prediction row {row_index} prompt conflicts with the bundle"
            )
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                (type(token_id) is not int or token_id < 0 for token_id in token_ids)
            )
        ):
            raise ValueError(
                f"existing GSM8K prediction row {row_index} tokens are invalid"
            )
        if (
            not isinstance(text, str)
            or not isinstance(raw_text, str)
            or (not raw_text.startswith(text))
        ):
            raise ValueError(
                f"existing GSM8K prediction row {row_index} text is invalid"
            )
        strict = extract_strict_answer(text)
        flexible = extract_flexible_answer(text)
        if row["strict_answer"] != strict or row["flexible_answer"] != flexible:
            raise ValueError(
                f"existing GSM8K prediction row {row_index} answer is inconsistent"
            )
        strict_correct = strict == gold
        flexible_correct = flexible == gold
        if (
            row["strict_correct"] is not strict_correct
            or row["flexible_correct"] is not flexible_correct
        ):
            raise ValueError(
                f"existing GSM8K prediction row {row_index} correctness is inconsistent"
            )
        if row["stop_reason"] not in valid_stop_reasons:
            raise ValueError(
                f"existing GSM8K prediction row {row_index} stop reason is invalid"
            )
        strict_correct_count += int(strict_correct)
        flexible_correct_count += int(flexible_correct)
    count = len(rows)
    strict_metric = strict_correct_count / count
    flexible_metric = flexible_correct_count / count
    expected_metrics = {
        "question_count": count,
        "exact_match_strict": strict_metric,
        "exact_match_flexible": flexible_metric,
    }
    _require_reusable_metrics(metrics, expected_metrics)
    return Gsm8kScore(
        predictions=rows,
        exact_match_strict=strict_metric,
        exact_match_flexible=flexible_metric,
    )


def evaluate_gsm8k(
    *,
    model: object,
    tokenizer: object,
    test_rows: Sequence[Mapping[str, object]],
    assignments: Sequence[Mapping[str, object]],
    expected_ids: Sequence[str],
    sequence_limit: int,
    max_new_tokens: int = GSM8K_MAX_NEW_TOKENS,
    stop_sequences: Sequence[str] = GSM8K_STOP_SEQUENCES,
    kernel_bank: object | None = None,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
) -> Gsm8kScore:
    """Preflight all prompts, then run deterministic full-prefix generation."""
    rows, assignment_by_id = _validate_inputs(test_rows, assignments, expected_ids)
    prompts: dict[str, str] = {}
    for row in rows:
        identifier = row["id"]
        assert isinstance(identifier, str)
        assignment = assignment_by_id[identifier]
        demonstrations = assignment["demonstrations"]
        assert isinstance(demonstrations, list)
        prompt = format_gsm8k_prompt(demonstrations, row)
        prompt_ids = tokenize_ids(tokenizer, prompt, label=f"GSM8K {identifier} prompt")
        ensure_sequence_limit(
            len(prompt_ids) + max_new_tokens,
            sequence_limit,
            label=f"GSM8K {identifier} prompt plus generation budget",
        )
        prompts[identifier] = prompt
    predictions: list[dict[str, object]] = []
    strict_correct_count = 0
    flexible_correct_count = 0
    for row in rows:
        identifier = row["id"]
        gold = row["gold_answer"]
        assert isinstance(identifier, str) and isinstance(gold, str)
        prompt = prompts[identifier]
        generated = greedy_generate_full_prefix(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            sequence_limit=sequence_limit,
            stop_sequences=stop_sequences,
            kernel_bank=kernel_bank,
            model_forward=model_forward,
            device=device,
        )
        strict = extract_strict_answer(generated.text)
        flexible = extract_flexible_answer(generated.text)
        strict_correct = strict == gold
        flexible_correct = flexible == gold
        strict_correct_count += int(strict_correct)
        flexible_correct_count += int(flexible_correct)
        predictions.append(
            {
                "id": identifier,
                "gold_answer": gold,
                "prompt_token_count": generated.prompt_token_count,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "generated_token_ids": list(generated.generated_token_ids),
                "text": generated.text,
                "raw_text": generated.raw_text,
                "strict_answer": strict,
                "flexible_answer": flexible,
                "strict_correct": strict_correct,
                "flexible_correct": flexible_correct,
                "stop_reason": generated.stop_reason,
            }
        )
    count = len(predictions)
    return Gsm8kScore(
        predictions=tuple(predictions),
        exact_match_strict=strict_correct_count / count,
        exact_match_flexible=flexible_correct_count / count,
    )


def _validate_inputs(
    rows: Sequence[Mapping[str, object]],
    assignments: Sequence[Mapping[str, object]],
    expected_ids: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    expected = tuple(expected_ids)
    if not expected or any(
        (not isinstance(item, str) or not item for item in expected)
    ):
        raise ValueError("GSM8K expected IDs must be non-empty strings")
    prepared = [dict(row) for row in rows]
    if tuple((row.get("id") for row in prepared)) != expected:
        raise ValueError("GSM8K rows do not match the exact expected ID order")
    for index, row in enumerate(prepared):
        required = {"id", "question", "answer", "gold_answer"}
        if set(row) != required:
            raise ValueError(f"GSM8K test row {index} fields are invalid")
        if any(
            (not isinstance(row[field], str) or not row[field] for field in required)
        ):
            raise ValueError(f"GSM8K test row {index} string fields are invalid")
    if len(assignments) != len(expected):
        raise ValueError("GSM8K assignments must cover every expected ID")
    by_id: dict[str, dict[str, object]] = {}
    for index, assignment in enumerate(assignments):
        value = dict(assignment)
        identifier = value.get("id")
        if identifier != expected[index] or value.get("test_index") != index:
            raise ValueError(
                "GSM8K assignments do not match the exact expected ID order"
            )
        demonstrations = value.get("demonstrations")
        if not isinstance(demonstrations, list) or len(demonstrations) != 5:
            raise ValueError(
                f"GSM8K assignment {index} must contain five demonstrations"
            )
        assert isinstance(identifier, str)
        by_id[identifier] = value
    return (prepared, by_id)


def _require_reusable_metrics(
    metrics: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    if (
        set(metrics) != set(expected)
        or metrics.get("question_count") != expected["question_count"]
    ):
        raise ValueError("existing GSM8K sidecar metrics are invalid")
    for field in set(expected) - {"question_count"}:
        value = metrics.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("existing GSM8K sidecar metrics are invalid")
        if not math.isfinite(float(value)) or not math.isclose(
            float(value), float(expected[field]), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("existing GSM8K sidecar metrics are inconsistent")
