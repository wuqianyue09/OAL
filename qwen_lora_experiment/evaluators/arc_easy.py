"""Task loop and character-normalized metrics for ARC-E."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import torch
from .choice import ContinuationScore, choose_first, score_continuation
from .common import ModelForward, encode_context_continuation, ensure_sequence_limit


@dataclass(frozen=True)
class ArcEasyScore:
    predictions: tuple[dict[str, object], ...]
    accuracy: float
    accuracy_normalized: float

    @property
    def question_count(self) -> int:
        return len(self.predictions)

    def summary(self) -> dict[str, object]:
        return {
            "question_count": self.question_count,
            "accuracy": self.accuracy,
            "accuracy_normalized": self.accuracy_normalized,
        }


def reusable_arc_easy_score(
    *,
    predictions: Sequence[Mapping[str, object]],
    metrics: Mapping[str, object],
    expected_rows: Sequence[Mapping[str, object]],
) -> ArcEasyScore:
    """Recompute a saved ARC-E score against the exact bundle rows."""
    expected_ids = tuple((row.get("id") for row in expected_rows))
    expected = tuple(_validate_rows(expected_rows, expected_ids))
    rows = tuple((dict(row) for row in predictions))
    if tuple((row.get("id") for row in rows)) != expected_ids:
        raise ValueError("existing ARC-E sidecar IDs do not match the bundle order")
    row_fields = {
        "id",
        "row_index",
        "label",
        "answerKey",
        "prediction",
        "prediction_normalized",
        "raw_correct",
        "normalized_correct",
        "candidates",
    }
    candidate_fields = {
        "index",
        "text",
        "context_token_count",
        "full_token_count",
        "continuation_token_ids",
        "token_count",
        "total_log_likelihood",
        "mean_log_likelihood",
        "choice_label",
        "choice_text",
        "character_count",
        "character_normalized_log_likelihood",
    }
    raw_correct_count = 0
    normalized_correct_count = 0
    for row_index, row in enumerate(rows):
        expected_row = expected[row_index]
        if set(row) != row_fields:
            raise ValueError(
                f"existing ARC-E prediction row {row_index} fields are invalid"
            )
        if row["row_index"] != expected_row["row_index"]:
            raise ValueError(
                f"existing ARC-E prediction row {row_index} index conflicts with the bundle"
            )
        choices = expected_row["choices"]
        assert isinstance(choices, dict)
        choice_texts = choices["text"]
        choice_labels = choices["label"]
        assert isinstance(choice_texts, list) and isinstance(choice_labels, list)
        choice_count = len(choice_texts)
        label = _choice_index(row["label"], row_index, "label", choice_count)
        if (
            label != expected_row["label"]
            or row["answerKey"] != expected_row["answerKey"]
        ):
            raise ValueError(
                f"existing ARC-E prediction row {row_index} gold conflicts with the bundle"
            )
        prediction = _choice_index(
            row["prediction"], row_index, "prediction", choice_count
        )
        normalized_prediction = _choice_index(
            row["prediction_normalized"],
            row_index,
            "normalized prediction",
            choice_count,
        )
        candidates = row["candidates"]
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or len(candidates) != choice_count
        ):
            raise ValueError(
                f"existing ARC-E prediction row {row_index} candidates are invalid"
            )
        totals: list[float] = []
        normalized_scores: list[float] = []
        for candidate_index, candidate_value in enumerate(candidates):
            if not isinstance(candidate_value, Mapping):
                raise ValueError(
                    f"existing ARC-E row {row_index} candidate {candidate_index} is invalid"
                )
            candidate = dict(candidate_value)
            if (
                set(candidate) != candidate_fields
                or candidate.get("index") != candidate_index
            ):
                raise ValueError(
                    f"existing ARC-E row {row_index} candidate {candidate_index} fields are invalid"
                )
            choice_text = candidate["choice_text"]
            if (
                choice_text != choice_texts[candidate_index]
                or candidate["choice_label"] != choice_labels[candidate_index]
                or candidate["text"] != " " + choice_texts[candidate_index]
            ):
                raise ValueError(
                    f"existing ARC-E row {row_index} candidate {candidate_index} conflicts with the bundle"
                )
            context_count = candidate["context_token_count"]
            full_count = candidate["full_token_count"]
            token_count = candidate["token_count"]
            token_ids = candidate["continuation_token_ids"]
            if (
                type(context_count) is not int
                or context_count <= 0
                or type(full_count) is not int
                or (type(token_count) is not int)
                or (token_count <= 0)
                or (full_count != context_count + token_count)
                or (not isinstance(token_ids, list))
                or (len(token_ids) != token_count)
                or any(
                    (
                        type(token_id) is not int or token_id < 0
                        for token_id in token_ids
                    )
                )
            ):
                raise ValueError(
                    f"existing ARC-E row {row_index} candidate {candidate_index} tokens are invalid"
                )
            character_count = candidate["character_count"]
            if type(character_count) is not int or character_count != len(choice_text):
                raise ValueError(
                    f"existing ARC-E row {row_index} candidate {candidate_index} character count is invalid"
                )
            total = _finite_float(
                candidate["total_log_likelihood"],
                f"ARC-E row {row_index} candidate {candidate_index} total likelihood",
            )
            mean = _finite_float(
                candidate["mean_log_likelihood"],
                f"ARC-E row {row_index} candidate {candidate_index} mean likelihood",
            )
            normalized = _finite_float(
                candidate["character_normalized_log_likelihood"],
                f"ARC-E row {row_index} candidate {candidate_index} character score",
            )
            if not math.isclose(
                mean, total / token_count, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"existing ARC-E row {row_index} candidate {candidate_index} mean is invalid"
                )
            if not math.isclose(
                normalized, total / character_count, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"existing ARC-E row {row_index} candidate {candidate_index} character-normalized score is invalid"
                )
            totals.append(total)
            normalized_scores.append(normalized)
        if prediction != choose_first(totals) or normalized_prediction != choose_first(
            normalized_scores
        ):
            raise ValueError(
                f"existing ARC-E prediction row {row_index} prediction is inconsistent"
            )
        raw_correct = prediction == label
        normalized_correct = normalized_prediction == label
        if (
            row["raw_correct"] is not raw_correct
            or row["normalized_correct"] is not normalized_correct
        ):
            raise ValueError(
                f"existing ARC-E prediction row {row_index} correctness is inconsistent"
            )
        raw_correct_count += int(raw_correct)
        normalized_correct_count += int(normalized_correct)
    count = len(rows)
    expected_metrics = {
        "question_count": count,
        "accuracy": raw_correct_count / count,
        "accuracy_normalized": normalized_correct_count / count,
    }
    _require_metrics(metrics, expected_metrics)
    return ArcEasyScore(
        predictions=rows,
        accuracy=expected_metrics["accuracy"],
        accuracy_normalized=expected_metrics["accuracy_normalized"],
    )


def evaluate_arc_easy(
    *,
    model: object,
    tokenizer: object,
    rows: Sequence[Mapping[str, object]],
    expected_ids: Sequence[str],
    sequence_limit: int,
    kernel_bank: object | None = None,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
) -> ArcEasyScore:
    """Preflight and score every ARC-E choice without assuming four choices."""
    prepared = _validate_rows(rows, expected_ids)
    for row in prepared:
        identifier = row["id"]
        question = row["question"]
        choices = row["choices"]
        assert isinstance(identifier, str) and isinstance(question, str)
        assert isinstance(choices, dict)
        texts = choices["text"]
        assert isinstance(texts, list)
        context = _context(question)
        for candidate_index, choice_text in enumerate(texts):
            assert isinstance(choice_text, str)
            encoded = encode_context_continuation(
                tokenizer,
                context,
                " " + choice_text,
                label=f"ARC-E {identifier} candidate {candidate_index}",
            )
            if not encoded.continuation_token_ids:
                raise ValueError(
                    f"ARC-E {identifier} candidate {candidate_index} has no continuation tokens"
                )
            ensure_sequence_limit(
                len(encoded.full_token_ids),
                sequence_limit,
                label=f"ARC-E {identifier} candidate {candidate_index}",
            )
    predictions: list[dict[str, object]] = []
    raw_correct = 0
    normalized_correct = 0
    for row in prepared:
        identifier = row["id"]
        question = row["question"]
        choices = row["choices"]
        label = row["label"]
        assert isinstance(identifier, str) and isinstance(question, str)
        assert isinstance(choices, dict) and type(label) is int
        texts = choices["text"]
        choice_labels = choices["label"]
        assert isinstance(texts, list) and isinstance(choice_labels, list)
        context = _context(question)
        scores: tuple[ContinuationScore, ...] = tuple(
            (
                score_continuation(
                    model=model,
                    tokenizer=tokenizer,
                    context=context,
                    continuation=" " + choice_text,
                    candidate_index=index,
                    sequence_limit=sequence_limit,
                    kernel_bank=kernel_bank,
                    model_forward=model_forward,
                    device=device,
                )
                for (index, choice_text) in enumerate(texts)
            )
        )
        normalized_scores = tuple(
            (
                score.total_log_likelihood / len(choice_text)
                for (score, choice_text) in zip(scores, texts, strict=True)
            )
        )
        prediction = choose_first((score.total_log_likelihood for score in scores))
        prediction_normalized = choose_first(normalized_scores)
        raw_correct += int(prediction == label)
        normalized_correct += int(prediction_normalized == label)
        candidates: list[dict[str, object]] = []
        for score, choice_text, choice_label, normalized in zip(
            scores, texts, choice_labels, normalized_scores, strict=True
        ):
            candidates.append(
                {
                    **score.as_dict(),
                    "choice_label": choice_label,
                    "choice_text": choice_text,
                    "character_count": len(choice_text),
                    "character_normalized_log_likelihood": normalized,
                }
            )
        predictions.append(
            {
                "id": identifier,
                "row_index": row["row_index"],
                "label": label,
                "answerKey": row["answerKey"],
                "prediction": prediction,
                "prediction_normalized": prediction_normalized,
                "raw_correct": prediction == label,
                "normalized_correct": prediction_normalized == label,
                "candidates": candidates,
            }
        )
    count = len(predictions)
    return ArcEasyScore(
        predictions=tuple(predictions),
        accuracy=raw_correct / count,
        accuracy_normalized=normalized_correct / count,
    )


def _context(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def _validate_rows(
    rows: Sequence[Mapping[str, object]], expected_ids: Sequence[str]
) -> list[dict[str, object]]:
    expected = tuple(expected_ids)
    if not expected or any(
        (not isinstance(item, str) or not item for item in expected)
    ):
        raise ValueError("ARC-E expected IDs must be non-empty strings")
    if len(set(expected)) != len(expected):
        raise ValueError("ARC-E expected IDs must be unique")
    prepared = [dict(row) for row in rows]
    observed = tuple((row.get("id") for row in prepared))
    if observed != expected:
        raise ValueError("ARC-E rows do not match the exact expected ID order")
    required = {"id", "row_index", "question", "choices", "answerKey", "label"}
    for index, row in enumerate(prepared):
        if set(row) != required:
            raise ValueError(f"ARC-E row {index} fields are invalid")
        if row["row_index"] != index:
            raise ValueError(f"ARC-E row {index} index is invalid")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"ARC-E row {index} question is invalid")
        choices = row["choices"]
        if not isinstance(choices, Mapping) or set(choices) != {"text", "label"}:
            raise ValueError(f"ARC-E row {index} choices are invalid")
        texts = choices["text"]
        labels = choices["label"]
        if (
            not isinstance(texts, Sequence)
            or isinstance(texts, (str, bytes))
            or (not texts)
            or (not isinstance(labels, Sequence))
            or isinstance(labels, (str, bytes))
            or (len(labels) != len(texts))
            or any((not isinstance(item, str) or not item.strip() for item in texts))
            or any((not isinstance(item, str) or not item.strip() for item in labels))
            or (len(set(labels)) != len(labels))
        ):
            raise ValueError(f"ARC-E row {index} choices are invalid")
        answer_key = row["answerKey"]
        label = row["label"]
        if (
            not isinstance(answer_key, str)
            or answer_key not in labels
            or type(label) is not int
            or (label != list(labels).index(answer_key))
        ):
            raise ValueError(f"ARC-E row {index} label is invalid")
        row["choices"] = {"text": list(texts), "label": list(labels)}
    return prepared


def _choice_index(value: object, row_index: int, label: str, choice_count: int) -> int:
    if type(value) is not int or value not in range(choice_count):
        raise ValueError(
            f"existing ARC-E prediction row {row_index} {label} is invalid"
        )
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"existing {label} is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"existing {label} is invalid")
    return result


def _require_metrics(
    metrics: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    if (
        set(metrics) != set(expected)
        or metrics.get("question_count") != expected["question_count"]
    ):
        raise ValueError("existing ARC-E sidecar metrics are invalid")
    for field in set(expected) - {"question_count"}:
        actual = _finite_float(metrics.get(field), f"ARC-E metric {field}")
        if not math.isclose(actual, float(expected[field]), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("existing ARC-E sidecar metrics are inconsistent")
