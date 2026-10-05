"""Task loop and equal-status metrics for HellaSwag."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import torch
from .choice import ContinuationScore, choose_first, score_continuation
from .common import ModelForward, encode_context_continuation, ensure_sequence_limit


@dataclass(frozen=True)
class HellaSwagScore:
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


def reusable_hellaswag_score(
    *,
    predictions: Sequence[Mapping[str, object]],
    metrics: Mapping[str, object],
    expected_rows: Sequence[Mapping[str, object]],
) -> HellaSwagScore:
    """Recompute a saved score from complete, structurally valid prediction rows."""
    expected = tuple((dict(row) for row in expected_rows))
    expected_ids = tuple((row.get("id") for row in expected))
    rows = tuple((dict(row) for row in predictions))
    if not expected_ids or tuple((row.get("id") for row in rows)) != expected_ids:
        raise ValueError("existing HellaSwag sidecar IDs do not match the bundle order")
    raw_correct_count = 0
    normalized_correct_count = 0
    row_fields = {
        "id",
        "source_index",
        "label",
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
    }
    for row_index, row in enumerate(rows):
        expected_row = expected[row_index]
        if set(row) != row_fields:
            raise ValueError(
                f"existing HellaSwag prediction row {row_index} fields are invalid"
            )
        source_index = row["source_index"]
        if isinstance(source_index, bool) or not isinstance(source_index, (int, str)):
            raise ValueError(
                f"existing HellaSwag prediction row {row_index} source is invalid"
            )
        if source_index != expected_row.get("source_index"):
            raise ValueError(
                f"existing HellaSwag prediction row {row_index} source conflicts with the bundle"
            )
        label = _choice_index(row["label"], row_index, "label")
        if label != expected_row.get("label"):
            raise ValueError(
                f"existing HellaSwag prediction row {row_index} label conflicts with the bundle"
            )
        prediction = _choice_index(row["prediction"], row_index, "prediction")
        normalized_prediction = _choice_index(
            row["prediction_normalized"], row_index, "normalized prediction"
        )
        candidates = row["candidates"]
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or len(candidates) != 4
        ):
            raise ValueError(
                f"existing HellaSwag prediction row {row_index} candidates are invalid"
            )
        totals: list[float] = []
        means: list[float] = []
        for candidate_index, candidate_value in enumerate(candidates):
            if not isinstance(candidate_value, Mapping):
                raise ValueError(
                    f"existing HellaSwag row {row_index} candidate {candidate_index} is invalid"
                )
            candidate = dict(candidate_value)
            if (
                set(candidate) != candidate_fields
                or candidate.get("index") != candidate_index
            ):
                raise ValueError(
                    f"existing HellaSwag row {row_index} candidate {candidate_index} fields are invalid"
                )
            text = candidate["text"]
            context_count = candidate["context_token_count"]
            full_count = candidate["full_token_count"]
            token_count = candidate["token_count"]
            token_ids = candidate["continuation_token_ids"]
            if not isinstance(text, str) or not text.startswith(" "):
                raise ValueError(
                    f"existing HellaSwag row {row_index} candidate {candidate_index} text is invalid"
                )
            expected_continuations = expected_row.get("continuations")
            if (
                not isinstance(expected_continuations, Sequence)
                or isinstance(expected_continuations, (str, bytes))
                or len(expected_continuations) != 4
                or (text != expected_continuations[candidate_index])
            ):
                raise ValueError(
                    f"existing HellaSwag row {row_index} candidate {candidate_index} conflicts with the bundle"
                )
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
                    f"existing HellaSwag row {row_index} candidate {candidate_index} tokens are invalid"
                )
            total = _finite_float(
                candidate["total_log_likelihood"],
                f"HellaSwag row {row_index} candidate {candidate_index} total likelihood",
            )
            mean = _finite_float(
                candidate["mean_log_likelihood"],
                f"HellaSwag row {row_index} candidate {candidate_index} mean likelihood",
            )
            if not math.isclose(
                mean, total / token_count, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"existing HellaSwag row {row_index} candidate {candidate_index} mean is invalid"
                )
            totals.append(total)
            means.append(mean)
        if prediction != choose_first(totals) or normalized_prediction != choose_first(
            means
        ):
            raise ValueError(
                f"existing HellaSwag prediction row {row_index} prediction is inconsistent"
            )
        raw_correct = prediction == label
        normalized_correct = normalized_prediction == label
        if (
            row["raw_correct"] is not raw_correct
            or row["normalized_correct"] is not normalized_correct
        ):
            raise ValueError(
                f"existing HellaSwag prediction row {row_index} correctness is inconsistent"
            )
        raw_correct_count += int(raw_correct)
        normalized_correct_count += int(normalized_correct)
    count = len(rows)
    expected_metrics = {
        "question_count": count,
        "accuracy": raw_correct_count / count,
        "accuracy_normalized": normalized_correct_count / count,
    }
    _require_metrics(metrics, expected_metrics, task="HellaSwag")
    return HellaSwagScore(
        predictions=rows,
        accuracy=expected_metrics["accuracy"],
        accuracy_normalized=expected_metrics["accuracy_normalized"],
    )


def evaluate_hellaswag(
    *,
    model: object,
    tokenizer: object,
    rows: Sequence[Mapping[str, object]],
    expected_ids: Sequence[str],
    sequence_limit: int,
    kernel_bank: object | None = None,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
) -> HellaSwagScore:
    """Preflight and score every four-choice validation row."""
    prepared = _validate_rows(rows, expected_ids)
    for row in prepared:
        identifier = row["id"]
        context = row["context"]
        continuations = row["continuations"]
        assert isinstance(identifier, str) and isinstance(context, str)
        assert isinstance(continuations, list)
        for candidate_index, continuation in enumerate(continuations):
            assert isinstance(continuation, str)
            encoded = encode_context_continuation(
                tokenizer,
                context,
                continuation,
                label=f"HellaSwag {identifier} candidate {candidate_index}",
            )
            if not encoded.continuation_token_ids:
                raise ValueError(
                    f"HellaSwag {identifier} candidate {candidate_index} has no continuation tokens"
                )
            ensure_sequence_limit(
                len(encoded.full_token_ids),
                sequence_limit,
                label=f"HellaSwag {identifier} candidate {candidate_index}",
            )
    predictions: list[dict[str, object]] = []
    raw_correct = 0
    normalized_correct = 0
    for row in prepared:
        identifier = row["id"]
        context = row["context"]
        continuations = row["continuations"]
        label = row["label"]
        assert isinstance(identifier, str) and isinstance(context, str)
        assert isinstance(continuations, list) and type(label) is int
        scores: tuple[ContinuationScore, ...] = tuple(
            (
                score_continuation(
                    model=model,
                    tokenizer=tokenizer,
                    context=context,
                    continuation=continuation,
                    candidate_index=index,
                    sequence_limit=sequence_limit,
                    kernel_bank=kernel_bank,
                    model_forward=model_forward,
                    device=device,
                )
                for (index, continuation) in enumerate(continuations)
            )
        )
        prediction = choose_first((score.total_log_likelihood for score in scores))
        prediction_normalized = choose_first(
            (score.mean_log_likelihood for score in scores)
        )
        raw_correct += int(prediction == label)
        normalized_correct += int(prediction_normalized == label)
        predictions.append(
            {
                "id": identifier,
                "source_index": row["source_index"],
                "label": label,
                "prediction": prediction,
                "prediction_normalized": prediction_normalized,
                "raw_correct": prediction == label,
                "normalized_correct": prediction_normalized == label,
                "candidates": [score.as_dict() for score in scores],
            }
        )
    count = len(predictions)
    return HellaSwagScore(
        predictions=tuple(predictions),
        accuracy=raw_correct / count,
        accuracy_normalized=normalized_correct / count,
    )


def _validate_rows(
    rows: Sequence[Mapping[str, object]], expected_ids: Sequence[str]
) -> list[dict[str, object]]:
    expected = tuple(expected_ids)
    if not expected or any(
        (not isinstance(item, str) or not item for item in expected)
    ):
        raise ValueError("HellaSwag expected IDs must be non-empty strings")
    if len(set(expected)) != len(expected):
        raise ValueError("HellaSwag expected IDs must be unique")
    prepared = [dict(row) for row in rows]
    observed = tuple((row.get("id") for row in prepared))
    if observed != expected:
        raise ValueError("HellaSwag rows do not match the exact expected ID order")
    for index, row in enumerate(prepared):
        required = {"id", "source_index", "context", "continuations", "label"}
        if set(row) != required:
            raise ValueError(f"HellaSwag row {index} fields are invalid")
        if not isinstance(row["context"], str) or not row["context"]:
            raise ValueError(f"HellaSwag row {index} context is invalid")
        continuations = row["continuations"]
        if (
            not isinstance(continuations, Sequence)
            or isinstance(continuations, (str, bytes))
            or len(continuations) != 4
            or any(
                (
                    not isinstance(item, str) or not item.startswith(" ")
                    for item in continuations
                )
            )
        ):
            raise ValueError(f"HellaSwag row {index} continuations are invalid")
        row["continuations"] = list(continuations)
        if type(row["label"]) is not int or row["label"] not in range(4):
            raise ValueError(f"HellaSwag row {index} label is invalid")
    return prepared


def _choice_index(value: object, row_index: int, label: str) -> int:
    if type(value) is not int or value not in range(4):
        raise ValueError(
            f"existing HellaSwag prediction row {row_index} {label} is invalid"
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
    metrics: Mapping[str, object], expected: Mapping[str, object], *, task: str
) -> None:
    if (
        set(metrics) != set(expected)
        or metrics.get("question_count") != expected["question_count"]
    ):
        raise ValueError(f"existing {task} sidecar metrics are invalid")
    for field in set(expected) - {"question_count"}:
        actual = _finite_float(metrics.get(field), f"{task} metric {field}")
        if not math.isclose(actual, float(expected[field]), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"existing {task} sidecar metrics are inconsistent")
